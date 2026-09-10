//! CSA2 source weights, immutable proposal execution and accepted-prefix state.
use crate::v41_memory::{DeviceAllocation, HostAllocation, LoadStream};
use crate::v41_tensors::NativeRtxTensors;
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary, V41AttentionOps, V41Compressor};
use ds41rt_loader::OfficialV41Catalog;
use std::{
    ffi::c_void,
    sync::atomic::{AtomicU64, Ordering},
};
static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);
fn ratio(layer: usize) -> Result<usize> {
    match layer {
        2 | 8 | 14 => Ok(2),
        20 => Ok(1),
        _ => anyhow::bail!("layer is not a V4.1 compressor source"),
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CompressorLease {
    owner: u64,
    slot: usize,
    generation: u64,
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct CompressorChunk {
    pub lease: CompressorLease,
    pub position: u64,
    pub tokens: u32,
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct CompressorLatentRow {
    pub lease: CompressorLease,
    pub source_row: u32,
    /// First token represented by this latent, for later RoPE/cache addressing.
    pub position: u64,
}
#[derive(Clone, Copy, Default)]
struct Slot {
    generation: u64,
    version: u64,
    request: Option<u64>,
    end: u64,
}
pub(crate) struct CompressorState<'a> {
    pending: Option<[DeviceAllocation<'a>; 2]>,
    slots: [Slot; 16],
    slot_count: usize,
    layer: usize,
    owner: u64,
}
impl<'a> CompressorState<'a> {
    pub fn device_bytes(layer: usize, slots: usize) -> Result<usize> {
        ensure!((1..=16).contains(&slots), "invalid compressor slot count");
        Ok(if ratio(layer)? == 2 { slots * 4096 } else { 0 })
    }
    pub fn new(
        library: &'a NativeLibrary,
        layer: usize,
        slots: usize,
        budget: usize,
    ) -> Result<Self> {
        ensure!(
            Self::device_bytes(layer, slots)? <= budget,
            "compressor state exceeds budget"
        );
        let owner = NEXT_OWNER
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| anyhow::anyhow!("compressor owner IDs exhausted"))?;
        Ok(Self {
            pending: if ratio(layer)? == 2 {
                Some([
                    DeviceAllocation::new(library, slots * 2048)?,
                    DeviceAllocation::new(library, slots * 2048)?,
                ])
            } else {
                None
            },
            slots: [Slot::default(); 16],
            slot_count: slots,
            layer,
            owner,
        })
    }
    pub fn begin_request(&mut self, slot: usize, request: u64) -> Result<CompressorLease> {
        ensure!(
            slot < self.slot_count && self.slots[slot].request.is_none(),
            "compressor slot unavailable"
        );
        ensure!(
            !self.slots.iter().any(|s| s.request == Some(request)),
            "duplicate compressor request"
        );
        let generation = self.slots[slot]
            .generation
            .checked_add(1)
            .context("compressor generation exhausted")?;
        self.slots[slot] = Slot {
            generation,
            request: Some(request),
            ..Slot::default()
        };
        // New requests start at zero and cannot read stale pending device rows.
        Ok(CompressorLease {
            owner: self.owner,
            slot,
            generation,
        })
    }
    fn validate(&self, lease: CompressorLease) -> Result<usize> {
        ensure!(
            lease.owner == self.owner && lease.slot < self.slot_count,
            "foreign compressor lease"
        );
        let slot = self.slots[lease.slot];
        ensure!(
            slot.request.is_some() && slot.generation == lease.generation,
            "stale compressor lease"
        );
        Ok(lease.slot)
    }
    pub fn request_id(&self, lease: CompressorLease) -> Result<u64> {
        self.slots[self.validate(lease)?]
            .request
            .context("compressor request missing")
    }
    pub fn committed_end(&self, lease: CompressorLease) -> Result<u64> {
        Ok(self.slots[self.validate(lease)?].end)
    }
    pub fn release(&mut self, lease: CompressorLease) -> Result<()> {
        let slot = self.validate(lease)?;
        self.slots[slot].request = None;
        Ok(())
    }
    /// Invalidate participating requests after an external cache transaction fails.
    pub fn invalidate(&mut self, leases: &[CompressorLease]) -> Result<()> {
        let slots = leases
            .iter()
            .map(|&l| self.validate(l))
            .collect::<Result<Vec<_>>>()?;
        for slot in slots {
            self.slots[slot].request = None;
        }
        Ok(())
    }
}
pub(crate) struct CompressorWeights<'a> {
    library: &'a NativeLibrary,
    tensors: NativeRtxTensors<'a>,
    layer: usize,
    ratio: usize,
    names: Vec<String>,
}
impl<'a> CompressorWeights<'a> {
    fn names(layer: usize) -> Result<Vec<String>> {
        let suffixes = if ratio(layer)? == 2 {
            vec!["wkv", "norm", "wgate"]
        } else {
            vec!["wkv", "norm"]
        };
        Ok(suffixes
            .into_iter()
            .map(|s| format!("layers.{layer}.attn.compressor.{s}.weight"))
            .collect())
    }
    pub fn device_bytes(catalog: &OfficialV41Catalog, layer: usize) -> Result<usize> {
        NativeRtxTensors::plan(catalog, &Self::names(layer)?)
    }
    pub fn load(
        library: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer: usize,
        budget: usize,
        staging_bytes: usize,
    ) -> Result<Self> {
        let names = Self::names(layer)?;
        let tensors = NativeRtxTensors::load(library, catalog, &names, budget, staging_bytes)?;
        Ok(Self {
            library,
            tensors,
            layer,
            ratio: ratio(layer)?,
            names,
        })
    }
    pub fn wave(&self, rows: usize, budget: usize) -> Result<CompressorWave<'_, 'a>> {
        ensure!(
            CompressorWave::device_bytes(self.layer, rows)? <= budget,
            "compressor wave exceeds budget"
        );
        let workspace = DeviceAllocation::new(self.library, V41Compressor::WORKSPACE_BYTES)?;
        ensure!(
            workspace.buffer.device_id == self.tensors.get(&self.names[0])?.device_id,
            "compressor weights device differs"
        );
        let kernel = unsafe { self.library.v41_compressor(workspace.buffer)? };
        Ok(CompressorWave {
            stream: LoadStream {
                library: self.library,
                raw: self.library.cuda_stream_create()?,
            },
            kernel,
            _workspace: workspace,
            norm: self.library.v41_attention_ops()?,
            weights: self,
            input: DeviceAllocation::new(self.library, rows * 10240)?,
            projected: DeviceAllocation::new(
                self.library,
                rows * 512 * if self.ratio == 2 { 4 } else { 2 },
            )?,
            scores: if self.ratio == 2 {
                Some(DeviceAllocation::new(self.library, rows * 2048)?)
            } else {
                None
            },
            descriptors: if self.ratio == 2 {
                Some(DeviceAllocation::new(self.library, rows * 8)?)
            } else {
                None
            },
            staging: if self.ratio == 2 {
                Some(HostAllocation::new(self.library, rows * 8)?)
            } else {
                None
            },
            output: DeviceAllocation::new(self.library, rows * 1024)?,
            capacity: rows,
            graph: None,
            ready: None,
        })
    }
}
struct Prepared {
    owner: u64,
    chunks: Vec<CompressorChunk>,
    versions: Vec<u64>,
    offsets: Vec<usize>,
    descriptors: Vec<u64>,
    completed: Vec<CompressorLatentRow>,
    rows: usize,
}
pub(crate) struct CompressorOutput<'a> {
    pub buffer: Ds41rtDeviceBuffer,
    pub completed: &'a [CompressorLatentRow],
}
pub(crate) struct CompressorWave<'w, 'a> {
    stream: LoadStream<'a>,
    kernel: V41Compressor<'a>,
    _workspace: DeviceAllocation<'a>,
    norm: V41AttentionOps<'a>,
    weights: &'w CompressorWeights<'a>,
    input: DeviceAllocation<'a>,
    projected: DeviceAllocation<'a>,
    scores: Option<DeviceAllocation<'a>>,
    descriptors: Option<DeviceAllocation<'a>>,
    staging: Option<HostAllocation<'a>>,
    output: DeviceAllocation<'a>,
    capacity: usize,
    graph: Option<(*mut c_void, usize, u64)>,
    ready: Option<Prepared>,
}
impl CompressorWave<'_, '_> {
    pub fn device_bytes(layer: usize, rows: usize) -> Result<usize> {
        ensure!(
            (1..=4096).contains(&rows),
            "invalid compressor row capacity"
        );
        Ok(V41Compressor::WORKSPACE_BYTES
            + rows
                * if ratio(layer)? == 2 {
                    10240 + 2048 + 2048 + 8 + 1024
                } else {
                    10240 + 1024 + 1024
                })
    }
    /// Packed BF16 [sum(chunk.tokens),5120], in chunk order. Finish all producer
    /// writes before execution, and keep this borrowed storage live until drop.
    pub fn input(&self) -> Ds41rtDeviceBuffer {
        self.input.buffer
    }
    fn synchronize(&self) -> Result<()> {
        unsafe { self.stream.library.cuda_stream_synchronize(self.stream.raw) }
    }
    fn prepare(
        &mut self,
        state: &CompressorState<'_>,
        chunks: &[CompressorChunk],
    ) -> Result<Prepared> {
        self.ready = None;
        ensure!(
            state.layer == self.weights.layer,
            "compressor source layer differs"
        );
        ensure!(
            !chunks.is_empty() && chunks.len() <= state.slot_count,
            "invalid compressor batch count"
        );
        if let Some(pending) = &state.pending {
            ensure!(
                pending[0].buffer.device_id == self.input.buffer.device_id,
                "compressor state device differs"
            );
        }
        let mut result = Prepared {
            owner: state.owner,
            chunks: chunks.to_vec(),
            versions: vec![],
            offsets: vec![],
            descriptors: vec![],
            completed: vec![],
            rows: 0,
        };
        let mut seen = [false; 16];
        for chunk in chunks {
            let slot = state.validate(chunk.lease)?;
            ensure!(
                !seen[slot] && chunk.tokens > 0,
                "duplicate or empty compressor chunk"
            );
            seen[slot] = true;
            ensure!(
                chunk.position == state.slots[slot].end,
                "compressor position is not the committed end"
            );
            chunk
                .position
                .checked_add(u64::from(chunk.tokens))
                .context("compressor position overflow")?;
            let end = result
                .rows
                .checked_add(chunk.tokens as usize)
                .context("compressor row overflow")?;
            ensure!(
                end <= self.capacity,
                "compressor batch exceeds row capacity"
            );
            result.versions.push(state.slots[slot].version);
            result.offsets.push(result.rows);
            for j in 0..chunk.tokens as usize {
                let pos = chunk.position + j as u64;
                let row = result.rows + j;
                if self.weights.ratio == 1 || pos % 2 == 1 {
                    result.completed.push(CompressorLatentRow {
                        lease: chunk.lease,
                        source_row: row as u32,
                        position: pos + 1 - self.weights.ratio as u64,
                    });
                }
                result.descriptors.push(if pos % 2 == 0 {
                    u64::MAX
                } else if j == 0 {
                    slot as u64
                } else {
                    (state.slot_count + row - 1) as u64
                });
            }
            result.rows = end;
        }
        Ok(result)
    }
    fn upload(&mut self, prepared: &Prepared) -> Result<()> {
        if let (Some(device), Some(host)) = (&self.descriptors, &mut self.staging) {
            let bytes = &mut host.bytes_mut()[..prepared.rows * 8];
            for (chunk, value) in bytes.chunks_exact_mut(8).zip(&prepared.descriptors) {
                chunk.copy_from_slice(&value.to_ne_bytes());
            }
            self.stream.library.copy_h2d(device.buffer, bytes)?;
        }
        Ok(())
    }
    unsafe fn enqueue(&self, state: &CompressorState<'_>, rows: usize) -> Result<()> {
        unsafe {
            self.kernel.project(
                self.input.buffer,
                self.weights.tensors.get(&self.weights.names[0])?,
                self.projected.buffer,
                rows,
                self.weights.ratio,
                self.stream.raw,
            )?;
            if let Some(pending) = &state.pending {
                self.kernel.project(
                    self.input.buffer,
                    self.weights.tensors.get(&self.weights.names[2])?,
                    self.scores.as_ref().unwrap().buffer,
                    rows,
                    2,
                    self.stream.raw,
                )?;
                self.kernel.pool(
                    self.projected.buffer,
                    self.scores.as_ref().unwrap().buffer,
                    pending[0].buffer,
                    pending[1].buffer,
                    self.descriptors.as_ref().unwrap().buffer,
                    self.weights.tensors.get(&self.weights.names[1])?,
                    self.output.buffer,
                    rows,
                    state.slot_count,
                    self.stream.raw,
                )?;
            } else {
                self.norm.norm(
                    self.projected.buffer,
                    self.weights.tensors.get(&self.weights.names[1])?,
                    None,
                    self.output.buffer,
                    rows as u32,
                    512,
                    self.stream.raw,
                )?;
            }
        }
        Ok(())
    }
    /// # Safety
    /// Finite input rows follow chunk order on this device, with producer writes
    /// complete. No external writes may race this wave or committed state.
    pub unsafe fn execute<'s>(
        &'s mut self,
        state: &'s CompressorState<'_>,
        chunks: &[CompressorChunk],
    ) -> Result<CompressorOutput<'s>> {
        let prepared = self.prepare(state, chunks)?;
        self.upload(&prepared)?;
        let launched = unsafe { self.enqueue(state, prepared.rows) };
        let drained = self.synchronize();
        launched.and(drained)?;
        self.ready = Some(prepared);
        self.output(state)
    }
    /// # Safety
    /// Same input contract as execute. Warmup is unpublished and never commits.
    pub unsafe fn capture(
        &mut self,
        state: &CompressorState<'_>,
        chunks: &[CompressorChunk],
    ) -> Result<()> {
        self.ready = None;
        ensure!(self.graph.is_none(), "compressor graph already captured");
        unsafe {
            self.execute(state, chunks)?;
        }
        let prepared = self.prepare(state, chunks)?;
        unsafe {
            self.stream
                .library
                .cuda_graph_begin_capture(self.stream.raw)?;
        }
        let launched = unsafe { self.enqueue(state, prepared.rows) };
        let captured = unsafe { self.stream.library.cuda_graph_end_capture(self.stream.raw) };
        match (launched, captured) {
            (Ok(()), Ok(graph)) => {
                self.graph = Some((graph, prepared.rows, state.owner));
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
    /// Same input contract as execute; state owner and live rows match capture.
    pub unsafe fn replay<'s>(
        &'s mut self,
        state: &'s CompressorState<'_>,
        chunks: &[CompressorChunk],
    ) -> Result<CompressorOutput<'s>> {
        let prepared = self.prepare(state, chunks)?;
        let (graph, rows, owner) = self.graph.context("compressor graph is not captured")?;
        ensure!(
            rows == prepared.rows && owner == state.owner,
            "compressor capture binding differs"
        );
        self.upload(&prepared)?;
        let launched = unsafe {
            self.stream
                .library
                .cuda_graph_launch(graph, self.stream.raw)
        };
        let drained = self.synchronize();
        launched.and(drained)?;
        self.ready = Some(prepared);
        self.output(state)
    }
    pub fn output<'s>(&'s self, state: &'s CompressorState<'_>) -> Result<CompressorOutput<'s>> {
        let prepared = self
            .ready
            .as_ref()
            .context("compressor output incomplete")?;
        ensure!(
            prepared.owner == state.owner,
            "compressor output state differs"
        );
        for (i, chunk) in prepared.chunks.iter().enumerate() {
            let slot = state.validate(chunk.lease)?;
            ensure!(
                state.slots[slot].version == prepared.versions[i]
                    && state.slots[slot].end == chunk.position,
                "stale compressor output"
            );
        }
        let mut buffer = self.output.buffer;
        buffer.bytes = prepared.rows * 1024;
        Ok(CompressorOutput {
            buffer,
            completed: &prepared.completed,
        })
    }
    /// Consume this proposal once. Validate all requests before any GPU write;
    /// zero acceptance also invalidates competing proposals via version advance.
    /// Returned rows name only accepted complete latents for later cache writes.
    pub fn commit(
        &mut self,
        state: &mut CompressorState<'_>,
        accepted: &[u32],
    ) -> Result<Vec<CompressorLatentRow>> {
        let prepared = self
            .ready
            .take()
            .context("compressor has no proposal to commit")?;
        ensure!(
            prepared.owner == state.owner && accepted.len() == prepared.chunks.len(),
            "compressor commit binding differs"
        );
        for (i, chunk) in prepared.chunks.iter().enumerate() {
            let slot = state.validate(chunk.lease)?;
            ensure!(
                state.slots[slot].version == prepared.versions[i]
                    && state.slots[slot].end == chunk.position,
                "stale compressor proposal"
            );
            ensure!(
                accepted[i] <= chunk.tokens,
                "compressor acceptance exceeds proposal"
            );
            state.slots[slot]
                .version
                .checked_add(1)
                .context("compressor version exhausted")?;
        }
        let slice = |b: Ds41rtDeviceBuffer, row: usize| Ds41rtDeviceBuffer {
            ptr: unsafe { b.ptr.cast::<u8>().add(row * 2048).cast() },
            bytes: 2048,
            ..b
        };
        let write = (|| -> Result<()> {
            if let Some(pending) = &state.pending {
                for (i, chunk) in prepared.chunks.iter().enumerate() {
                    let count = accepted[i] as usize;
                    if count > 0 && (chunk.position + count as u64) % 2 == 1 {
                        let row = prepared.offsets[i] + count - 1;
                        for (dst, src) in [
                            (pending[0].buffer, self.projected.buffer),
                            (pending[1].buffer, self.scores.as_ref().unwrap().buffer),
                        ] {
                            unsafe {
                                self.stream.library.copy_d2d_async(
                                    slice(dst, chunk.lease.slot),
                                    slice(src, row),
                                    2048,
                                    self.stream.raw,
                                )?;
                            }
                        }
                    }
                }
            }
            Ok(())
        })();
        let drained = self.synchronize();
        if let Err(error) = write.and(drained) {
            for chunk in &prepared.chunks {
                state.slots[chunk.lease.slot].request = None;
            }
            return Err(error);
        }
        let mut completed = vec![];
        for (i, chunk) in prepared.chunks.iter().enumerate() {
            state.slots[chunk.lease.slot].end = chunk.position + u64::from(accepted[i]);
            state.slots[chunk.lease.slot].version += 1;
            let end = prepared.offsets[i] + accepted[i] as usize;
            completed.extend(
                prepared
                    .completed
                    .iter()
                    .filter(|r| {
                        r.source_row as usize >= prepared.offsets[i]
                            && (r.source_row as usize) < end
                    })
                    .copied(),
            );
        }
        Ok(completed)
    }
    pub fn clear_graph(&mut self) -> Result<()> {
        self.ready = None;
        self.synchronize()?;
        if let Some((graph, _, _)) = self.graph.take() {
            unsafe {
                self.stream.library.cuda_graph_exec_destroy(graph)?;
            }
        }
        Ok(())
    }
}
impl Drop for CompressorWave<'_, '_> {
    fn drop(&mut self) {
        if let Err(error) = self.clear_graph() {
            tracing::error!(%error,"draining compressor graph");
        }
    }
}
