//! All three draft transformer stages on one graph-owned stream.
use super::{DsparkStage, DsparkWeights};
use crate::v41_dspark_cache::{DsparkWindow, WindowLease, WindowRead};
use crate::v41_memory::LoadStream;
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::Ds41rtDeviceBuffer;
use std::ffi::c_void;

pub(crate) struct DsparkChain<'weights, 'library> {
    stream: LoadStream<'library>,
    stages: [DsparkStage<'weights, 'library>; 3],
    graph: Option<(*mut c_void, usize, [u64; 3])>,
    ready: Option<usize>,
}
impl<'library> DsparkWeights<'library> {
    pub fn chain_bytes(&self, requests: u32) -> Result<usize> {
        self.stage_bytes(requests)?
            .checked_mul(3)
            .context("dSpark chain budget overflow")
    }
    pub fn chain(&self, requests: u32, budget: usize) -> Result<DsparkChain<'_, 'library>> {
        ensure!(
            self.chain_bytes(requests)? <= budget,
            "dSpark chain exceeds budget"
        );
        let library = self.experts[0].buffers[0].library;
        let bytes = self.stage_bytes(requests)?;
        Ok(DsparkChain {
            stream: LoadStream {
                library,
                raw: library.cuda_stream_create()?,
            },
            stages: [
                self.stage(0, requests, bytes)?,
                self.stage(1, requests, bytes)?,
                self.stage(2, requests, bytes)?,
            ],
            graph: None,
            ready: None,
        })
    }
}
impl DsparkChain<'_, '_> {
    pub fn inputs(&self) -> [Ds41rtDeviceBuffer; 2] {
        self.stages[0].inputs()
    }
    fn invalidate(&mut self) {
        self.ready = None;
        for stage in &mut self.stages {
            stage.invalidate();
        }
    }
    fn synchronize(&self) -> Result<()> {
        unsafe { self.stream.library.cuda_stream_synchronize(self.stream.raw) }
    }
    fn prepare(
        &mut self,
        windows: [&DsparkWindow<'_>; 3],
        bindings: [&[(WindowLease, u64)]; 3],
    ) -> Result<[WindowRead; 3]> {
        self.invalidate();
        let reads = [
            self.stages[0].prepare(windows[0], bindings[0])?,
            self.stages[1].prepare(windows[1], bindings[1])?,
            self.stages[2].prepare(windows[2], bindings[2])?,
        ];
        ensure!(
            reads[0].owner != reads[1].owner
                && reads[0].owner != reads[2].owner
                && reads[1].owner != reads[2].owner,
            "dSpark stages require independent cache owners"
        );
        for stage in 1..3 {
            ensure!(
                bindings[stage].len() == bindings[0].len(),
                "dSpark chain stage counts differ"
            );
            for (&(first, end), &(lease, stage_end)) in bindings[0].iter().zip(bindings[stage]) {
                ensure!(
                    end == stage_end
                        && windows[0].request_id(first)? == windows[stage].request_id(lease)?,
                    "dSpark chain request or committed position differs"
                );
            }
        }
        Ok(reads)
    }
    fn upload(
        &mut self,
        reads: &[WindowRead; 3],
        bindings: [&[(WindowLease, u64)]; 3],
    ) -> Result<()> {
        for stage in 0..3 {
            self.stages[stage].upload(&reads[stage], bindings[stage])?;
        }
        Ok(())
    }
    unsafe fn enqueue(&mut self, reads: &[WindowRead; 3], requests: usize) -> Result<()> {
        for stage in 0..3 {
            if stage > 0 {
                let source = self.stages[stage - 1].output_storage();
                let destination = self.stages[stage].inputs();
                unsafe {
                    self.stream.library.copy_d2d_async(
                        destination[0],
                        source[0],
                        requests * 5 * 40960,
                        self.stream.raw,
                    )?;
                    self.stream.library.copy_d2d_async(
                        destination[1],
                        source[1],
                        requests * 5 * 16,
                        self.stream.raw,
                    )?;
                }
            }
            unsafe {
                self.stages[stage].enqueue_on(&reads[stage], requests, self.stream.raw)?;
            }
        }
        Ok(())
    }
    /// # Safety
    /// Initialize finite embedded BF16 residuals and incoming FP32 pre-mix in
    /// input row order; cache bindings across stages describe the same requests.
    /// All buffers are on this device and serialized through graph completion.
    pub unsafe fn execute(
        &mut self,
        windows: [&DsparkWindow<'_>; 3],
        bindings: [&[(WindowLease, u64)]; 3],
    ) -> Result<[Ds41rtDeviceBuffer; 2]> {
        let reads = self.prepare(windows, bindings)?;
        self.upload(&reads, bindings)?;
        let launched = unsafe { self.enqueue(&reads, bindings[0].len()) };
        let drained = self.synchronize();
        if let Err(error) = launched.and(drained) {
            self.invalidate();
            return Err(error);
        }
        self.ready = Some(bindings[0].len());
        self.output()
    }
    /// # Safety
    /// Same contract as execute; captures all three independent window owners.
    pub unsafe fn capture(
        &mut self,
        windows: [&DsparkWindow<'_>; 3],
        bindings: [&[(WindowLease, u64)]; 3],
    ) -> Result<()> {
        self.invalidate();
        ensure!(self.graph.is_none(), "dSpark chain already captured");
        unsafe {
            self.execute(windows, bindings)?;
        }
        let reads = self.prepare(windows, bindings)?;
        unsafe {
            self.stream
                .library
                .cuda_graph_begin_capture(self.stream.raw)?;
        }
        let launched = unsafe { self.enqueue(&reads, bindings[0].len()) };
        let captured = unsafe { self.stream.library.cuda_graph_end_capture(self.stream.raw) };
        self.invalidate();
        match (launched, captured) {
            (Ok(()), Ok(graph)) => {
                self.graph = Some((graph, bindings[0].len(), reads.each_ref().map(|r| r.owner)));
                Ok(())
            }
            (Err(error), Ok(graph)) => {
                unsafe {
                    self.stream.library.cuda_graph_exec_destroy(graph)?;
                }
                Err(error)
            }
            (Err(error), Err(_)) | (Ok(()), Err(error)) => Err(error),
        }
    }
    /// # Safety
    /// Same contract as execute; owner IDs and request count must match capture.
    pub unsafe fn replay(
        &mut self,
        windows: [&DsparkWindow<'_>; 3],
        bindings: [&[(WindowLease, u64)]; 3],
    ) -> Result<[Ds41rtDeviceBuffer; 2]> {
        let reads = self.prepare(windows, bindings)?;
        let (graph, count, owners) = self.graph.context("dSpark chain not captured")?;
        ensure!(
            count == bindings[0].len() && owners == reads.each_ref().map(|r| r.owner),
            "dSpark chain capture binding differs"
        );
        self.upload(&reads, bindings)?;
        let launched = unsafe {
            self.stream
                .library
                .cuda_graph_launch(graph, self.stream.raw)
        };
        let drained = self.synchronize();
        launched.and(drained)?;
        self.ready = Some(count);
        self.output()
    }
    pub fn output(&self) -> Result<[Ds41rtDeviceBuffer; 2]> {
        let requests = self.ready.context("dSpark chain output incomplete")?;
        let mut output = self.stages[2].output_storage();
        output[0].bytes = requests * 5 * 40960;
        output[1].bytes = requests * 5 * 16;
        Ok(output)
    }
}
impl Drop for DsparkChain<'_, '_> {
    fn drop(&mut self) {
        if let Err(error) = self.synchronize() {
            tracing::error!(%error,"draining dSpark chain");
        }
        if let Some((graph, _, _)) = self.graph.take() {
            if let Err(error) = unsafe { self.stream.library.cuda_graph_exec_destroy(graph) } {
                tracing::error!(%error,"destroying dSpark chain graph");
            }
        }
    }
}
