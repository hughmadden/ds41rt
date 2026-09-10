use super::{DsparkConfidence, DsparkMarkov, DsparkWeights};
use crate::v41_memory::{DeviceAllocation, LoadStream};
use crate::v41_tensors::VocabularyHead;
use anyhow::{ensure, Context, Result};
use ds41rt_core::DsparkRng;
use ds41rt_ffi::{Ds41rtDeviceBuffer, V41DraftStep, V41Hc, V41VocabularyProjection};
use std::ffi::c_void;

/// Five dependent Markov/sample positions followed by raw confidence, on one
/// stream. All draft tensors use [position,request,...] order at live row count.
pub(crate) struct DsparkTerminal<'weights, 'library> {
    stream: LoadStream<'library>,
    markov: DsparkMarkov<'weights, 'library>,
    confidence: DsparkConfidence<'weights, 'library>,
    sample: V41DraftStep<'library>,
    hc: V41Hc<'library>,
    residual: DeviceAllocation<'library>,
    pre_mix: DeviceAllocation<'library>,
    head_kernel: V41VocabularyProjection<'library>,
    _head_workspace: DeviceAllocation<'library>,
    normalized: DeviceAllocation<'library>,
    head: &'weights VocabularyHead<'library>,
    weights: &'weights DsparkWeights<'library>,
    shared_logits: DeviceAllocation<'library>,
    adjusted_logits: DeviceAllocation<'library>,
    rng: DeviceAllocation<'library>,
    sampling_requests: Option<usize>,
    temperatures: DeviceAllocation<'library>,
    tokens: DeviceAllocation<'library>,
    capacity: usize,
    graph: Option<(*mut c_void, usize)>,
    ready_requests: Option<usize>,
}
impl<'library> DsparkWeights<'library> {
    /// Collapse residual streams, normalize and project through the borrowed shared
    /// head, then apply Markov correction, sampling and confidence.
    pub fn terminal<'weights>(
        &'weights self,
        head: &'weights VocabularyHead<'library>,
        capacity: usize,
        budget: usize,
    ) -> Result<DsparkTerminal<'weights, 'library>> {
        ensure!(
            DsparkTerminal::device_bytes(capacity)? <= budget,
            "dSpark terminal exceeds budget"
        );
        let library = self.experts[0].buffers[0].library;
        head.weight()?;
        self.tensor("mtp.2.norm.weight")?;
        let head_workspace =
            DeviceAllocation::new(library, V41VocabularyProjection::WORKSPACE_BYTES)?;
        let head_kernel = unsafe { library.v41_vocabulary_head(head_workspace.buffer)? };
        Ok(DsparkTerminal {
            head_kernel,
            _head_workspace: head_workspace,
            normalized: DeviceAllocation::new(library, capacity * 5 * 10240)?,
            head,
            weights: self,
            stream: LoadStream {
                library,
                raw: library.cuda_stream_create()?,
            },
            markov: self.markov(capacity, DsparkMarkov::device_bytes(capacity)?)?,
            confidence: self
                .confidence(capacity * 5, DsparkConfidence::device_bytes(capacity * 5)?)?,
            sample: library.v41_draft_step()?,
            hc: library.v41_hc()?,
            residual: DeviceAllocation::new(library, capacity * 5 * 40960)?,
            pre_mix: DeviceAllocation::new(library, capacity * 5 * 16)?,
            shared_logits: DeviceAllocation::new(library, capacity * 5 * 129280 * 4)?,
            adjusted_logits: DeviceAllocation::new(library, capacity * 5 * 129280 * 4)?,
            rng: DeviceAllocation::new(library, capacity * 16)?,
            sampling_requests: None,
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
        Ok(
            capacity * (5 * 129280 * 4 * 2 + 7 * 4 + 16 + 5 * 10240 + 5 * (40960 + 16))
                + V41VocabularyProjection::WORKSPACE_BYTES,
        )
    }
    pub fn device_bytes(capacity: usize) -> Result<usize> {
        Ok(Self::additional_bytes(capacity)?
            + DsparkMarkov::device_bytes(capacity)?
            + DsparkConfidence::device_bytes(capacity * 5)?)
    }
    /// Stable inputs: BF16 residual [5,R,4,5120], FP32 incoming pre-mix [5,R,4]
    /// and anchor IDs [R]. Sampling is set via `prepare_sampling`.
    /// Use live R densely, without capacity padding between positions. Never free
    /// or retain after drop; finish all producer writes before execute/replay.
    pub fn inputs(&self) -> [Ds41rtDeviceBuffer; 3] {
        let mut anchors = self.tokens.buffer;
        anchors.bytes = self.capacity * 4;
        [self.residual.buffer, self.pre_mix.buffer, anchors]
    }
    /// Atomically admit a batch's RNG ranges before reserving any. Once reserved,
    /// failures/cancellation consume them; replay intentionally retains the same
    /// draws until this method prepares a new attempt. Temperatures are validated
    /// here and uploaded with request-owned seed/range metadata after prior work.
    pub fn prepare_sampling(
        &mut self,
        rngs: &mut [&mut DsparkRng],
        temperatures: &[f32],
    ) -> Result<()> {
        self.ready_requests = None;
        self.sampling_requests = None;
        ensure!(
            !rngs.is_empty() && rngs.len() <= self.capacity && rngs.len() == temperatures.len(),
            "invalid terminal sampling request count"
        );
        ensure!(
            temperatures.iter().all(|t| t.is_finite() && *t >= 0.0),
            "invalid draft temperature"
        );
        ensure!(
            rngs.iter().all(|rng| rng.can_reserve()),
            "dSpark RNG exhausted"
        );
        self.synchronize()?;
        let mut metadata = Vec::with_capacity(rngs.len() * 16);
        for rng in rngs.iter_mut() {
            let reservation = rng.reserve().context("dSpark RNG exhausted")?;
            metadata.extend_from_slice(&reservation.seed.to_ne_bytes());
            metadata.extend_from_slice(&reservation.first_subsequence.to_ne_bytes());
        }
        let mut temperature_bytes = Vec::with_capacity(temperatures.len() * 4);
        for temperature in temperatures {
            temperature_bytes.extend_from_slice(&temperature.to_ne_bytes());
        }
        self.stream.library.copy_h2d(self.rng.buffer, &metadata)?;
        self.stream
            .library
            .copy_h2d(self.temperatures.buffer, &temperature_bytes)?;
        self.sampling_requests = Some(rngs.len());
        Ok(())
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
        ensure!(
            self.sampling_requests == Some(requests),
            "sampling state does not match live requests"
        );
        let library = self.stream.library;
        let row_bytes = requests * 129280 * 4;
        unsafe {
            self.hc.pre(
                self.residual.buffer,
                self.pre_mix.buffer,
                self.confidence.inputs()[0],
                requests * 5,
                self.stream.raw,
            )?;
            // This RNE variant matches the reference's one final BF16 rounding.
            library.cuda_ds4_rmsnorm_bf16_rne_async(
                self.confidence.inputs()[0],
                self.weights.tensor("mtp.2.norm.weight")?,
                self.normalized.buffer,
                (requests * 5) as i32,
                5120,
                1e-6,
                self.stream.raw,
            )?;
            self.head_kernel.launch(
                self.normalized.buffer,
                self.head.weight()?,
                self.shared_logits.buffer,
                requests * 5,
                self.stream.raw,
            )?;
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
                    self.rng.buffer,
                    self.temperatures.buffer,
                    Self::slice(self.adjusted_logits.buffer, position * row_bytes, row_bytes)?,
                    next,
                    requests,
                    position,
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
    /// Inputs must satisfy `inputs()` layout with initialized BF16 residuals and
    /// FP32 incoming pre-mix coefficients,
    /// valid anchors <129280, finite shared+Markov logits and
    /// sampling state prepared for these requests. Producers must be complete;
    /// no input writes may race execution.
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
    /// Same input contract as execute; prepare new sampling state for a fresh attempt.
    pub unsafe fn replay(&mut self, requests: usize) -> Result<[Ds41rtDeviceBuffer; 3]> {
        self.ready_requests = None;
        ensure!(
            self.sampling_requests == Some(requests),
            "sampling state does not match replay requests"
        );
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
