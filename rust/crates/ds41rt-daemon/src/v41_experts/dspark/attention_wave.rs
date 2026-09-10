//! One-stream draft attention with generation-checked committed cache reads.
use super::{DsparkAttentionOutput, DsparkProjection, DsparkWeights, ProjectionKind};
use crate::v41_dspark_cache::{DsparkWindow, WindowLease, WindowRead};
use crate::v41_memory::{DeviceAllocation, HostAllocation, LoadStream};
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary, V41AttentionOps, V41DsparkAttention};
use std::ffi::c_void;
pub(crate) struct DsparkAttentionWave<'weights, 'library> {
    stream: LoadStream<'library>,
    qa: DsparkProjection<'weights, 'library>,
    qb: DsparkProjection<'weights, 'library>,
    kv: DsparkProjection<'weights, 'library>,
    output: DsparkAttentionOutput<'weights, 'library>,
    ops: V41AttentionOps<'library>,
    attention: V41DsparkAttention<'library>,
    q_norm: Ds41rtDeviceBuffer,
    kv_norm: Ds41rtDeviceBuffer,
    sink: Ds41rtDeviceBuffer,
    query: DeviceAllocation<'library>,
    draft: DeviceAllocation<'library>,
    descriptors: DeviceAllocation<'library>,
    staging: HostAllocation<'library>,
    requests: u32,
    graph: Option<(*mut c_void, u32, u64)>,
    ready: Option<u32>,
}
impl<'library> DsparkWeights<'library> {
    pub fn attention_wave(
        &self,
        stage: usize,
        requests: u32,
        budget: usize,
    ) -> Result<DsparkAttentionWave<'_, 'library>> {
        ensure!(stage < 3, "invalid dSpark attention stage");
        let library = self.experts[stage].buffers[0].library;
        ensure!(
            DsparkAttentionWave::device_bytes(library, requests)? <= budget,
            "dSpark attention wave exceeds budget"
        );
        let rows = requests * 5;
        let capacity = DsparkAttentionWave::projection_capacity(requests)?;
        let projection = |kind| {
            self.projection(
                kind,
                capacity,
                DsparkProjection::device_bytes(library, kind, capacity)?,
            )
        };
        Ok(DsparkAttentionWave {
            stream: LoadStream {
                library,
                raw: library.cuda_stream_create()?,
            },
            qa: projection(ProjectionKind::QueryA(stage))?,
            qb: projection(ProjectionKind::QueryB(stage))?,
            kv: projection(ProjectionKind::Kv(stage))?,
            output: self.attention_output(
                stage,
                capacity,
                DsparkAttentionOutput::device_bytes(library, capacity)?,
            )?,
            ops: library.v41_attention_ops()?,
            attention: library.v41_dspark_attention()?,
            q_norm: self.tensor(&format!("mtp.{stage}.attn.q_norm.weight"))?,
            kv_norm: self.tensor(&format!("mtp.{stage}.attn.kv_norm.weight"))?,
            sink: self.tensor(&format!("mtp.{stage}.attn.attn_sink"))?,
            query: DeviceAllocation::new(library, rows as usize * 65536)?,
            draft: DeviceAllocation::new(library, rows as usize * 1024)?,
            descriptors: DeviceAllocation::new(library, 128)?,
            staging: HostAllocation::new(library, 128)?,
            requests,
            graph: None,
            ready: None,
        })
    }
}
impl DsparkAttentionWave<'_, '_> {
    fn projection_capacity(requests: u32) -> Result<u32> {
        ensure!(
            (1..=16).contains(&requests),
            "invalid attention request capacity"
        );
        // Compiled storage buckets; every launch still uses requests*5 live rows.
        Ok(if requests <= 3 { 16 } else { 80 })
    }
    pub fn additional_bytes(rows: u32) -> Result<usize> {
        ensure!((1..=4096).contains(&rows), "invalid attention wave rows");
        Ok(rows as usize * (65536 + 1024) + 128)
    }
    pub fn device_bytes(library: &NativeLibrary, requests: u32) -> Result<usize> {
        ensure!(
            (1..=16).contains(&requests),
            "invalid attention wave request capacity"
        );
        let rows = requests * 5;
        let capacity = Self::projection_capacity(requests)?;
        let mut bytes =
            Self::additional_bytes(rows)? + DsparkAttentionOutput::device_bytes(library, capacity)?;
        for kind in [
            ProjectionKind::QueryA(0),
            ProjectionKind::QueryB(0),
            ProjectionKind::Kv(0),
        ] {
            bytes = bytes
                .checked_add(DsparkProjection::device_bytes(library, kind, capacity)?)
                .context("attention wave storage overflow")?;
        }
        Ok(bytes)
    }
    pub fn input(&self) -> Ds41rtDeviceBuffer {
        self.qa.input()
    }
    pub fn frequencies(&self) -> Ds41rtDeviceBuffer {
        self.output.frequencies()
    }
    fn synchronize(&self) -> Result<()> {
        unsafe { self.stream.library.cuda_stream_synchronize(self.stream.raw) }
    }
    fn prepare(
        &mut self,
        window: &DsparkWindow<'_>,
        requests: &[(WindowLease, u64)],
    ) -> Result<WindowRead> {
        self.ready = None;
        ensure!(
            !requests.is_empty() && requests.len() <= self.requests as usize,
            "attention wave exceeds request capacity"
        );
        window.attention_read(requests)
    }
    fn upload(&mut self, read: &WindowRead) -> Result<()> {
        let bytes =
            unsafe { std::slice::from_raw_parts(read.descriptors.as_ptr().cast::<u8>(), 128) };
        self.staging.bytes_mut().copy_from_slice(bytes);
        self.stream
            .library
            .copy_h2d(self.descriptors.buffer, self.staging.bytes_mut())
    }
    unsafe fn enqueue(&mut self, read: &WindowRead, requests: u32) -> Result<()> {
        let rows = requests * 5;
        let stream = self.stream.raw;
        unsafe {
            self.qa
                .enqueue(self.qa.input(), self.qa.output_storage(), rows, stream)?;
            self.ops.norm(
                self.qa.output_storage(),
                self.q_norm,
                None,
                self.qb.input(),
                rows,
                1280,
                stream,
            )?;
            self.qb
                .enqueue(self.qb.input(), self.qb.output_storage(), rows, stream)?;
            self.ops.rope(
                self.qb.output_storage(),
                self.output.frequencies(),
                self.query.buffer,
                rows,
                64,
                false,
                stream,
            )?;
            self.kv
                .enqueue(self.qa.input(), self.kv.output_storage(), rows, stream)?;
            self.ops.kv(
                self.kv.output_storage(),
                self.kv_norm,
                self.output.frequencies(),
                self.draft.buffer,
                rows,
                stream,
            )?;
            self.attention.launch(
                self.query.buffer,
                read.ring,
                self.draft.buffer,
                self.sink,
                self.descriptors.buffer,
                self.output.input(),
                requests,
                read.slots,
                stream,
            )?;
            self.output.enqueue(rows, stream)
        }
    }
    /// # Safety
    /// Initialize finite normalized BF16 hidden rows [requests,5,5120] and FP32
    /// complex frequencies at each request's committed_end..committed_end+5;
    /// complete producer writes and serialize raw input/frequency view reuse.
    pub unsafe fn execute(
        &mut self,
        window: &DsparkWindow<'_>,
        requests: &[(WindowLease, u64)],
    ) -> Result<Ds41rtDeviceBuffer> {
        let read = self.prepare(window, requests)?;
        self.upload(&read)?;
        let launched = unsafe { self.enqueue(&read, requests.len() as u32) };
        let drained = self.synchronize();
        launched.and(drained)?;
        self.ready = Some(requests.len() as u32);
        self.output()
    }
    /// # Safety
    /// Same input contract as execute. The graph may only replay against this
    /// window owner, revalidated on every call; destruction never reads its ring.
    pub unsafe fn capture(
        &mut self,
        window: &DsparkWindow<'_>,
        requests: &[(WindowLease, u64)],
    ) -> Result<()> {
        ensure!(self.graph.is_none(), "attention wave already captured");
        unsafe {
            self.execute(window, requests)?;
        }
        let read = self.prepare(window, requests)?;
        unsafe {
            self.stream
                .library
                .cuda_graph_begin_capture(self.stream.raw)?;
        }
        let launched = unsafe { self.enqueue(&read, requests.len() as u32) };
        let captured = unsafe { self.stream.library.cuda_graph_end_capture(self.stream.raw) };
        match (launched, captured) {
            (Ok(()), Ok(graph)) => {
                self.graph = Some((graph, requests.len() as u32, read.owner));
                Ok(())
            }
            (Err(error), Ok(graph)) => {
                if let Err(cleanup) = unsafe { self.stream.library.cuda_graph_exec_destroy(graph) }
                {
                    tracing::error!(%cleanup,"destroying failed attention wave capture");
                }
                Err(error)
            }
            (Err(error), Err(_)) | (Ok(()), Err(error)) => Err(error),
        }
    }
    /// # Safety
    /// Same input contract as execute; cache owner and captured request count must match.
    pub unsafe fn replay(
        &mut self,
        window: &DsparkWindow<'_>,
        requests: &[(WindowLease, u64)],
    ) -> Result<Ds41rtDeviceBuffer> {
        let read = self.prepare(window, requests)?;
        let (graph, count, owner) = self.graph.context("attention wave not captured")?;
        ensure!(
            count as usize == requests.len() && owner == read.owner,
            "attention wave capture binding differs"
        );
        self.upload(&read)?;
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
    pub fn output(&self) -> Result<Ds41rtDeviceBuffer> {
        let requests = self.ready.context("attention wave output incomplete")?;
        let mut output = self.output.output_storage();
        output.bytes = requests as usize * 5 * 10240;
        Ok(output)
    }
}
impl Drop for DsparkAttentionWave<'_, '_> {
    fn drop(&mut self) {
        if let Err(error) = self.synchronize() {
            tracing::error!(%error,"draining attention wave");
        }
        if let Some((graph, _, _)) = self.graph.take() {
            if let Err(error) = unsafe { self.stream.library.cuda_graph_exec_destroy(graph) } {
                tracing::error!(%error,"destroying attention wave graph");
            }
        }
    }
}
