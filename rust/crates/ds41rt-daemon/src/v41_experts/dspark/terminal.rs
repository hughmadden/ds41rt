use super::{DsparkConfidence, DsparkMarkov, DsparkWeights};
use crate::v41_memory::{DeviceAllocation, LoadStream};
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, V41DraftStep};
use std::ffi::c_void;

/// Five dependent Markov/sample positions followed by raw confidence, on one
/// stream. All draft tensors use [position,request,...] order at live row count.
pub(crate) struct DsparkTerminal<'weights, 'library> {
    stream: LoadStream<'library>,
    markov: DsparkMarkov<'weights, 'library>,
    confidence: DsparkConfidence<'weights, 'library>,
    sample: V41DraftStep<'library>,
    shared_logits: DeviceAllocation<'library>,
    adjusted_logits: DeviceAllocation<'library>,
    noise: DeviceAllocation<'library>,
    temperatures: DeviceAllocation<'library>,
    tokens: DeviceAllocation<'library>,
    capacity: usize,
    graph: Option<(*mut c_void, usize)>,
    ready_requests: Option<usize>,
}
impl<'library> DsparkWeights<'library> {
    /// Shared-head projection is produced upstream; this owner handles its
    /// Markov correction, sequential sampling and final confidence projection.
    pub fn terminal(&self, capacity: usize, budget: usize) -> Result<DsparkTerminal<'_, 'library>> {
        ensure!(
            DsparkTerminal::device_bytes(capacity)? <= budget,
            "dSpark terminal exceeds budget"
        );
        let library = self.experts[0].buffers[0].library;
        Ok(DsparkTerminal {
            stream: LoadStream {
                library,
                raw: library.cuda_stream_create()?,
            },
            markov: self.markov(capacity, DsparkMarkov::device_bytes(capacity)?)?,
            confidence: self
                .confidence(capacity * 5, DsparkConfidence::device_bytes(capacity * 5)?)?,
            sample: library.v41_draft_step()?,
            shared_logits: DeviceAllocation::new(library, capacity * 5 * 129280 * 4)?,
            adjusted_logits: DeviceAllocation::new(library, capacity * 5 * 129280 * 4)?,
            noise: DeviceAllocation::new(library, capacity * 5 * 129280 * 4)?,
            temperatures: DeviceAllocation::new(library, capacity * 4)?,
            tokens: DeviceAllocation::new(library, capacity * 6 * 4)?,
            capacity,
            graph: None,
            ready_requests: None,
        })
    }
}
impl DsparkTerminal<'_, '_> {
    pub fn additional_bytes(capacity: usize) -> Result<usize> {
        ensure!(
            (1..=16).contains(&capacity),
            "terminal request capacity must be 1 through 16"
        );
        Ok(capacity * (5 * 129280 * 4 * 3 + 7 * 4))
    }
    pub fn device_bytes(capacity: usize) -> Result<usize> {
        Ok(Self::additional_bytes(capacity)?
            + DsparkMarkov::device_bytes(capacity)?
            + DsparkConfidence::device_bytes(capacity * 5)?)
    }
    /// Stable inputs: raw shared-head logits [5,R,V], collapsed pre-final-norm
    /// hidden [5,R,5120], Exp(1) noise [5,R,V], temperatures [R], anchor IDs [R].
    /// Use live R densely, without capacity padding between positions. Never free
    /// or retain after drop; finish all producer writes before execute/replay.
    pub fn inputs(&self) -> [Ds41rtDeviceBuffer; 5] {
        let mut anchors = self.tokens.buffer;
        anchors.bytes = self.capacity * 4;
        [
            self.shared_logits.buffer,
            self.confidence.inputs()[0],
            self.noise.buffer,
            self.temperatures.buffer,
            anchors,
        ]
    }
    fn slice(
        buffer: Ds41rtDeviceBuffer,
        offset: usize,
        bytes: usize,
    ) -> Result<Ds41rtDeviceBuffer> {
        ensure!(
            offset
                .checked_add(bytes)
                .context("terminal slice overflow")?
                <= buffer.bytes,
            "terminal slice exceeds allocation"
        );
        Ok(Ds41rtDeviceBuffer {
            ptr: unsafe { buffer.ptr.cast::<u8>().add(offset).cast() },
            bytes,
            ..buffer
        })
    }
    unsafe fn enqueue(&self, requests: usize) -> Result<()> {
        ensure!(
            requests > 0 && requests <= self.capacity,
            "terminal requests exceed capacity"
        );
        let library = self.stream.library;
        let row_bytes = requests * 129280 * 4;
        unsafe {
            library.copy_d2d_async(
                self.markov.tokens(),
                self.tokens.buffer,
                requests * 4,
                self.stream.raw,
            )?;
            for position in 0..5 {
                self.markov.enqueue_on(requests, self.stream.raw)?;
                let [embedding, bias] = self.markov.storage();
                let confidence_embedding = Self::slice(
                    self.confidence.inputs()[1],
                    position * requests * 512,
                    requests * 512,
                )?;
                library.copy_d2d_async(
                    confidence_embedding,
                    embedding,
                    requests * 512,
                    self.stream.raw,
                )?;
                let next = Self::slice(
                    self.tokens.buffer,
                    (position + 1) * requests * 4,
                    requests * 4,
                )?;
                self.sample.launch(
                    Self::slice(self.shared_logits.buffer, position * row_bytes, row_bytes)?,
                    bias,
                    Self::slice(self.noise.buffer, position * row_bytes, row_bytes)?,
                    self.temperatures.buffer,
                    Self::slice(self.adjusted_logits.buffer, position * row_bytes, row_bytes)?,
                    next,
                    requests,
                    self.stream.raw,
                )?;
                if position < 4 {
                    library.copy_d2d_async(
                        self.markov.tokens(),
                        next,
                        requests * 4,
                        self.stream.raw,
                    )?;
                }
            }
            self.confidence.enqueue_on(requests * 5, self.stream.raw)?;
        }
        Ok(())
    }
    fn synchronize(&self) -> Result<()> {
        unsafe { self.stream.library.cuda_stream_synchronize(self.stream.raw) }
    }
    /// # Safety
    /// Inputs must satisfy `inputs()` layout with initialized BF16 hidden states,
    /// valid anchors <129280, finite shared+Markov logits, finite nonnegative
    /// temperatures and positive finite independent Exp(1) noise for stochastic
    /// rows. Producers must be complete; no input writes may race execution.
    pub unsafe fn execute(&mut self, requests: usize) -> Result<[Ds41rtDeviceBuffer; 3]> {
        self.ready_requests = None;
        unsafe {
            self.enqueue(requests)?;
        }
        self.synchronize()?;
        self.ready_requests = Some(requests);
        self.output()
    }
    /// # Safety
    /// Same input contract as execute; warms the complete sequence before capture.
    pub unsafe fn capture(&mut self, requests: usize) -> Result<()> {
        ensure!(self.graph.is_none(), "terminal graph already captured");
        unsafe {
            self.execute(requests)?;
        }
        self.ready_requests = None;
        unsafe {
            self.stream
                .library
                .cuda_graph_begin_capture(self.stream.raw)?;
        }
        let launch = unsafe { self.enqueue(requests) };
        let captured = unsafe { self.stream.library.cuda_graph_end_capture(self.stream.raw) };
        match (launch, captured) {
            (Ok(()), Ok(graph)) => {
                self.graph = Some((graph, requests));
                Ok(())
            }
            (Err(error), Ok(graph)) => {
                if let Err(cleanup) = unsafe { self.stream.library.cuda_graph_exec_destroy(graph) }
                {
                    tracing::error!(%cleanup, "destroying failed terminal capture");
                }
                Err(error)
            }
            (Err(error), Err(_)) | (Ok(()), Err(error)) => Err(error),
        }
    }
    /// # Safety
    /// Same input contract as execute; refresh inputs and noise before replay.
    pub unsafe fn replay(&mut self, requests: usize) -> Result<[Ds41rtDeviceBuffer; 3]> {
        self.ready_requests = None;
        let (graph, captured) = self.graph.context("terminal graph is not captured")?;
        ensure!(
            requests == captured,
            "terminal replay shape differs from capture"
        );
        unsafe {
            self.stream
                .library
                .cuda_graph_launch(graph, self.stream.raw)?;
        }
        self.synchronize()?;
        self.ready_requests = Some(requests);
        self.output()
    }
    pub fn clear_graph(&mut self) -> Result<()> {
        self.ready_requests = None;
        self.synchronize()?;
        if let Some((graph, _)) = self.graph.take() {
            unsafe {
                self.stream.library.cuda_graph_exec_destroy(graph)?;
            }
        }
        Ok(())
    }
    /// Tokens [6,R] (anchor then five drafts), corrected raw logits [5,R,V],
    /// raw confidence [5,R]. Borrowed until reuse/drop; no history is committed.
    pub fn output(&self) -> Result<[Ds41rtDeviceBuffer; 3]> {
        let requests = self
            .ready_requests
            .context("terminal output is not complete")?;
        Ok([
            Self::slice(self.tokens.buffer, 0, requests * 6 * 4)?,
            Self::slice(self.adjusted_logits.buffer, 0, requests * 5 * 129280 * 4)?,
            Self::slice(self.confidence.storage()[0], 0, requests * 5 * 4)?,
        ])
    }
}
impl Drop for DsparkTerminal<'_, '_> {
    fn drop(&mut self) {
        if let Err(error) = self.synchronize() {
            tracing::error!(%error, "draining dSpark terminal");
        }
        if let Some((graph, _)) = self.graph.take() {
            if let Err(error) = unsafe { self.stream.library.cuda_graph_exec_destroy(graph) } {
                tracing::error!(%error, "destroying dSpark terminal graph");
            }
        }
    }
}
