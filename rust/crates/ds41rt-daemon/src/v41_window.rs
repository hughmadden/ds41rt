//! Backbone FP8 window KV production, private proposals and accepted ring writes.
use crate::v41_memory::{DeviceAllocation, HostAllocation, LoadStream};
use crate::v41_tensors::NativeRtxTensors;
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary, V41AttentionOps, V41Fp8Plan, V41Kv};
use ds41rt_loader::OfficialV41Catalog;
use std::{
    ffi::c_void,
    marker::PhantomData,
    sync::atomic::{AtomicU64, Ordering},
};
static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);
mod prefix;
pub(crate) use prefix::{WindowPrefix, WINDOW_PREFIX_BYTES};
static NEXT_SNAPSHOT: AtomicU64 = AtomicU64::new(1);
fn next(counter: &AtomicU64) -> Result<u64> {
    counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
        .map_err(|_| anyhow::anyhow!("window IDs exhausted"))
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WindowLease {
    owner: u64,
    slot: usize,
    generation: u64,
}
#[derive(Clone, Copy)]
pub(crate) struct WindowChunk {
    pub lease: WindowLease,
    pub position: u64,
    pub tokens: u32,
}
#[derive(Clone, Copy, Default)]
struct Slot {
    request: Option<u64>,
    generation: u64,
    version: u64,
    end: u64,
    begin: u64,
}
pub(crate) struct WindowCacheView<'a> {
    pub values: Ds41rtDeviceBuffer,
    pub scales: Ds41rtDeviceBuffer,
    pub device_end: Ds41rtDeviceBuffer,
    pub end: u64,
    /// Earliest initialized logical row; preceding ring bytes are inaccessible.
    pub begin: u64,
    _owner: PhantomData<&'a ()>,
}
fn slice(mut b: Ds41rtDeviceBuffer, offset: usize, bytes: usize) -> Ds41rtDeviceBuffer {
    debug_assert!(offset + bytes <= b.bytes);
    b.ptr = unsafe { b.ptr.cast::<u8>().add(offset).cast() };
    b.bytes = bytes;
    b
}
pub(crate) struct WindowState<'a> {
    values: DeviceAllocation<'a>,
    scales: DeviceAllocation<'a>,
    ends: DeviceAllocation<'a>,
    slots: [Slot; 16],
    slot_count: usize,
    layer: usize,
    owner: u64,
}
impl<'a> WindowState<'a> {
    pub fn device_bytes(layer: usize, slots: usize) -> Result<usize> {
        ensure!(
            layer < 40 && (1..=16).contains(&slots),
            "invalid backbone window state shape"
        );
        Ok(slots * (128 * 528 + 8))
    }
    pub fn new(
        library: &'a NativeLibrary,
        layer: usize,
        slots: usize,
        budget: usize,
    ) -> Result<Self> {
        ensure!(
            Self::device_bytes(layer, slots)? <= budget,
            "window state exceeds budget"
        );
        let value = Self {
            values: DeviceAllocation::new(library, slots * 128 * 512)?,
            scales: DeviceAllocation::new(library, slots * 128 * 16)?,
            ends: DeviceAllocation::new(library, slots * 8)?,
            slots: [Slot::default(); 16],
            slot_count: slots,
            layer,
            owner: next(&NEXT_OWNER)?,
        };
        library.copy_h2d(value.ends.buffer, &vec![0; slots * 8])?;
        Ok(value)
    }
    pub fn begin_request(&mut self, slot: usize, request: u64) -> Result<WindowLease> {
        ensure!(
            slot < self.slot_count && self.slots[slot].request.is_none(),
            "window slot unavailable"
        );
        ensure!(
            !self.slots.iter().any(|s| s.request == Some(request)),
            "duplicate window request"
        );
        let generation = self.slots[slot]
            .generation
            .checked_add(1)
            .context("window generation exhausted")?;
        self.ends
            .library
            .copy_h2d(slice(self.ends.buffer, slot * 8, 8), &[0; 8])?;
        self.slots[slot] = Slot {
            request: Some(request),
            generation,
            ..Slot::default()
        };
        Ok(WindowLease {
            owner: self.owner,
            slot,
            generation,
        })
    }
    /// Start a fresh decoder window at the retained encoder suffix. No ring rows
    /// become valid: consumers must enforce the returned view's lower bound.
    /// Existing consumers must be drained; version advancement revokes proposals.
    pub fn begin_replay(&mut self, lease: WindowLease, position: u64) -> Result<()> {
        let slot = self.validate(lease)?;
        ensure!(self.layer >= 20 && position <= 1048576, "invalid decoder replay start");
        ensure!(self.slots[slot].end == 0 && self.slots[slot].version == 0,
            "decoder replay requires a fresh window lease");
        if let Err(error) = self.ends.library.copy_h2d(
            slice(self.ends.buffer, slot * 8, 8), &position.to_ne_bytes()) {
            self.slots[slot].request = None;
            return Err(error);
        }
        self.slots[slot].begin = position;
        self.slots[slot].end = position;
        self.slots[slot].version = 1;
        Ok(())
    }
    fn validate(&self, lease: WindowLease) -> Result<usize> {
        ensure!(
            lease.owner == self.owner && lease.slot < self.slot_count,
            "foreign window lease"
        );
        let s = self.slots[lease.slot];
        ensure!(
            s.request.is_some() && s.generation == lease.generation,
            "stale window lease"
        );
        Ok(lease.slot)
    }
    pub fn end(&self, lease: WindowLease) -> Result<u64> {
        Ok(self.slots[self.validate(lease)?].end)
    }
    pub fn request_id(&self, lease: WindowLease) -> Result<u64> {
        self.slots[self.validate(lease)?]
            .request
            .context("window request missing")
    }
    pub fn view(&self, lease: WindowLease) -> Result<WindowCacheView<'_>> {
        let slot = self.validate(lease)?;
        Ok(WindowCacheView {
            values: slice(self.values.buffer, slot * 128 * 512, 128 * 512),
            scales: slice(self.scales.buffer, slot * 128 * 16, 128 * 16),
            device_end: slice(self.ends.buffer, slot * 8, 8),
            end: self.slots[slot].end,
            begin: self.slots[slot].begin,
            _owner: PhantomData,
        })
    }
    /// Consumers must drain before release; generation checks revoke old views.
    pub fn release(&mut self, lease: WindowLease) -> Result<()> {
        let slot = self.validate(lease)?;
        self.slots[slot].request = None;
        self.ends
            .library
            .copy_h2d(slice(self.ends.buffer, slot * 8, 8), &[0; 8])
    }
    pub fn invalidate(&mut self, leases: &[WindowLease]) -> Result<()> {
        let slots = leases
            .iter()
            .map(|&l| self.validate(l))
            .collect::<Result<Vec<_>>>()?;
        for &slot in &slots {
            self.slots[slot].request = None;
        }
        let mut result = Ok(());
        for slot in slots {
            result = result.and(
                self.ends
                    .library
                    .copy_h2d(slice(self.ends.buffer, slot * 8, 8), &[0; 8]),
            );
        }
        result
    }
}
pub(crate) struct WindowWeights<'a> {
    library: &'a NativeLibrary,
    layer: usize,
    names: [String; 3],
    tensors: NativeRtxTensors<'a>,
    scales: DeviceAllocation<'a>,
}
impl<'a> WindowWeights<'a> {
    fn names(layer: usize) -> Result<[String; 3]> {
        ensure!(layer < 40, "invalid backbone window layer");
        Ok([
            format!("layers.{layer}.attn.wkv.weight"),
            format!("layers.{layer}.attn.wkv.scale"),
            format!("layers.{layer}.attn.kv_norm.weight"),
        ])
    }
    pub fn device_bytes(
        library: &NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer: usize,
    ) -> Result<usize> {
        let resident = NativeRtxTensors::plan(catalog, &Self::names(layer)?)?;
        ensure!(resident == 2_625_024, "unexpected window tensor sizes");
        Ok(resident
            + library
                .v41_fp8_matrix_info(1, 5120, 512)?
                .packed_weight_scale_bytes as usize)
    }
    pub fn load(
        library: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer: usize,
        budget: usize,
        staging: usize,
    ) -> Result<Self> {
        ensure!(
            Self::device_bytes(library, catalog, layer)? <= budget,
            "window weights exceed budget"
        );
        let names = Self::names(layer)?;
        let tensors = NativeRtxTensors::load(library, catalog, &names, budget, staging)?;
        let fp8 = library.v41_fp8_matrix_kernel(1, 5120, 512)?;
        let scales = DeviceAllocation::new(library, fp8.info().packed_weight_scale_bytes as usize)?;
        let stream = LoadStream {
            library,
            raw: library.cuda_stream_create()?,
        };
        let launched =
            unsafe { fp8.pack_scales(tensors.get(&names[1])?, scales.buffer, stream.raw) };
        launched.and(unsafe { library.cuda_stream_synchronize(stream.raw) })?;
        Ok(Self {
            library,
            layer,
            names,
            tensors,
            scales,
        })
    }
    pub fn wave(&self, capacity: u32, budget: usize) -> Result<WindowWave<'_, 'a>> {
        ensure!(
            WindowWave::device_bytes(self.library, capacity)? <= budget,
            "window wave exceeds budget"
        );
        let fp8 = self.library.v41_fp8_matrix_plan(capacity, 5120, 512)?;
        let rows = capacity as usize;
        let value = WindowWave {
            stream: LoadStream {
                library: self.library,
                raw: self.library.cuda_stream_create()?,
            },
            scratch: DeviceAllocation::new(self.library, fp8.info().scratch_bytes as usize)?,
            fp8,
            alpha: DeviceAllocation::new(self.library, 4)?,
            norm: self.library.v41_attention_ops()?,
            kv: self.library.v41_kv()?,
            weights: self,
            input: DeviceAllocation::new(self.library, rows * 10240)?,
            projected: DeviceAllocation::new(self.library, rows * 1024)?,
            normalized: DeviceAllocation::new(self.library, rows * 1024)?,
            positions: DeviceAllocation::new(self.library, rows * 8)?,
            frequencies: DeviceAllocation::new(self.library, rows * 256)?,
            values: DeviceAllocation::new(self.library, rows * 512)?,
            scales: DeviceAllocation::new(self.library, rows * 16)?,
            destinations: DeviceAllocation::new(self.library, rows * 8)?,
            staging: HostAllocation::new(self.library, rows * 8 + 128)?,
            capacity: rows,
            graph: None,
            ready: None,
        };
        ensure!(
            value.input.buffer.device_id == self.tensors.get(&self.names[0])?.device_id,
            "window weight device differs"
        );
        let launched = unsafe {
            value
                .fp8
                .initialize_scratch(value.scratch.buffer, value.alpha.buffer, value.stream.raw)
        };
        launched.and(value.synchronize())?;
        Ok(value)
    }
}
struct Prepared {
    snapshot: u64,
    owner: u64,
    chunks: Vec<WindowChunk>,
    offsets: Vec<usize>,
    versions: Vec<u64>,
    rows: usize,
}
pub(crate) struct WindowOutput<'a> {
    pub values: Ds41rtDeviceBuffer,
    pub scales: Ds41rtDeviceBuffer,
    pub projected: Ds41rtDeviceBuffer,
    pub normalized: Ds41rtDeviceBuffer,
    pub frequencies: Ds41rtDeviceBuffer,
    _owner: PhantomData<&'a ()>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WindowBinding {
    snapshot: u64,
    lease: WindowLease,
}
impl WindowBinding {
    pub fn same_pool(self, other: Self) -> bool {
        self.lease.owner == other.lease.owner
    }
}
pub(crate) struct WindowProposal<'a> {
    pub cache: WindowCacheView<'a>,
    pub values: Ds41rtDeviceBuffer,
    pub scales: Ds41rtDeviceBuffer,
    pub capacity: usize,
    pub request: u64,
    pub layer: usize,
    pub binding: WindowBinding,
    first: u64,
    tokens: usize,
    offset: usize,
    _wave: PhantomData<&'a ()>,
}
impl WindowProposal<'_> {
    /// Committed end, proposal offset/count and query position. Ring consumers
    /// use positions max(cache.begin,query+1-128)..=query; rows >=end use proposals.
    pub fn metadata(&self, position: u64) -> Result<[u64; 4]> {
        ensure!(
            position >= self.first && position < self.first + self.tokens as u64,
            "query outside window proposal"
        );
        Ok([self.first, self.offset as u64, self.tokens as u64, position])
    }
}
pub(crate) struct WindowWave<'w, 'a> {
    stream: LoadStream<'a>,
    fp8: V41Fp8Plan<'a>,
    scratch: DeviceAllocation<'a>,
    alpha: DeviceAllocation<'a>,
    norm: V41AttentionOps<'a>,
    kv: V41Kv<'a>,
    weights: &'w WindowWeights<'a>,
    input: DeviceAllocation<'a>,
    projected: DeviceAllocation<'a>,
    normalized: DeviceAllocation<'a>,
    positions: DeviceAllocation<'a>,
    frequencies: DeviceAllocation<'a>,
    values: DeviceAllocation<'a>,
    scales: DeviceAllocation<'a>,
    destinations: DeviceAllocation<'a>,
    staging: HostAllocation<'a>,
    capacity: usize,
    graph: Option<(*mut c_void, usize, u64)>,
    ready: Option<Prepared>,
}
impl WindowWave<'_, '_> {
    pub fn device_bytes(library: &NativeLibrary, capacity: u32) -> Result<usize> {
        Ok(library
            .v41_fp8_matrix_plan_info(capacity, 5120, 512)?
            .scratch_bytes as usize
            + 4
            + capacity as usize * 13088)
    }
    pub fn input(&self) -> Ds41rtDeviceBuffer {
        self.input.buffer
    }
    fn synchronize(&self) -> Result<()> {
        unsafe { self.stream.library.cuda_stream_synchronize(self.stream.raw) }
    }
    fn prepare(&mut self, state: &WindowState<'_>, chunks: &[WindowChunk]) -> Result<Prepared> {
        self.ready = None;
        ensure!(
            state.layer == self.weights.layer
                && state.values.buffer.device_id == self.input.buffer.device_id,
            "window state binding differs"
        );
        ensure!(
            !chunks.is_empty() && chunks.len() <= state.slot_count,
            "invalid window chunks"
        );
        let mut p = Prepared {
            snapshot: next(&NEXT_SNAPSHOT)?,
            owner: state.owner,
            chunks: chunks.to_vec(),
            offsets: vec![],
            versions: vec![],
            rows: 0,
        };
        let mut seen = [false; 16];
        for c in chunks {
            let slot = state.validate(c.lease)?;
            ensure!(
                !seen[slot] && c.tokens > 0,
                "duplicate or empty window chunk"
            );
            seen[slot] = true;
            ensure!(
                c.position == state.slots[slot].end
                    && c.position <= 1048576
                    && u64::from(c.tokens) <= 1048576 - c.position,
                "window position differs or exceeds context"
            );
            p.offsets.push(p.rows);
            p.versions.push(state.slots[slot].version);
            p.rows += c.tokens as usize;
            ensure!(p.rows <= self.capacity, "window rows exceed capacity");
        }
        Ok(p)
    }
    fn upload(&mut self, p: &Prepared) -> Result<()> {
        let bytes = self.staging.bytes_mut();
        for (i, c) in p.chunks.iter().enumerate() {
            for j in 0..c.tokens as usize {
                let row = p.offsets[i] + j;
                bytes[row * 8..row * 8 + 8].copy_from_slice(&(c.position + j as u64).to_ne_bytes());
            }
        }
        unsafe {
            self.stream.library.copy_host_buffer_h2d_async(
                self.positions.buffer, self.staging.buffer, p.rows * 8, self.stream.raw,
            )
        }
    }
    unsafe fn enqueue(&self, rows: usize) -> Result<()> {
        unsafe {
            self.norm.backbone_frequencies(
                self.positions.buffer,
                self.frequencies.buffer,
                rows as u32,
                self.weights.layer as u32,
                self.stream.raw,
            )?;
            self.fp8.launch(
                self.input.buffer,
                self.weights.tensors.get(&self.weights.names[0])?,
                self.weights.scales.buffer,
                self.scratch.buffer,
                self.alpha.buffer,
                self.projected.buffer,
                rows as u32,
                self.stream.raw,
            )?;
            self.norm.norm(
                self.projected.buffer,
                self.weights.tensors.get(&self.weights.names[2])?,
                None,
                self.normalized.buffer,
                rows as u32,
                512,
                self.stream.raw,
            )?;
            self.kv.pack(
                self.normalized.buffer,
                Some(self.frequencies.buffer),
                self.values.buffer,
                self.scales.buffer,
                rows,
                self.stream.raw,
            )?;
        }
        Ok(())
    }
    /// Consume the exact normalized hidden rows used by a bound attention query.
    /// # Safety
    /// Query producers have completed; no external writes race query or cache
    /// storage. The batch's request identities correspond to these hidden rows.
    pub unsafe fn execute_query<'s>(
        &'s mut self,
        state: &'s WindowState<'_>,
        chunks: &[WindowChunk],
        query: &crate::v41_attention_query::AttentionQueryOutput<'_>,
    ) -> Result<WindowOutput<'s>> {
        self.ready = None;
        let prepared = self.prepare(state, chunks)?;
        ensure!(query.binding()?.layer() == self.weights.layer
            && query.layer == self.weights.layer
            && query.rows == prepared.rows
            && query.hidden.bytes == prepared.rows * 10240
            && query.hidden.device_id == self.input.buffer.device_id
            && query.tokens()?.iter().copied().eq(chunks.iter().flat_map(|c|
                c.position..c.position + u64::from(c.tokens))),
            "window query layer, rows or positions differ");
        let executed = (|| -> Result<()> {
            unsafe {
                self.stream.library.copy_d2d_async(
                    self.input.buffer, query.hidden, query.hidden.bytes, self.stream.raw,
                )?;
            }
            if self.graph.is_none_or(|(_, rows, owner)| rows != prepared.rows || owner != state.owner) {
                self.clear_graph()?;
                unsafe { self.capture(state, chunks)?; }
            }
            unsafe { self.replay(state, chunks)?; }
            Ok(())
        })();
        if let Err(error) = executed {
            self.synchronize()?;
            self.ready = None;
            return Err(error);
        }
        self.output(state)
    }
    /// # Safety
    /// Finite attention-input hidden rows follow chunk order on this device,
    /// with producer writes complete. No external writes race this wave/state.
    pub unsafe fn execute<'s>(
        &'s mut self,
        state: &'s WindowState<'_>,
        chunks: &[WindowChunk],
    ) -> Result<WindowOutput<'s>> {
        let p = self.prepare(state, chunks)?;
        let launched = self.upload(&p).and_then(|()| unsafe { self.enqueue(p.rows) });
        launched.and(self.synchronize())?;
        self.ready = Some(p);
        self.output(state)
    }
    /// # Safety
    /// Same inputs as execute. Warmup is drained and unpublished after capture.
    pub unsafe fn capture(
        &mut self,
        state: &WindowState<'_>,
        chunks: &[WindowChunk],
    ) -> Result<()> {
        self.ready = None;
        ensure!(self.graph.is_none(), "window graph already captured");
        unsafe {
            self.execute(state, chunks)?;
        }
        let p = self.prepare(state, chunks)?;
        unsafe {
            self.stream
                .library
                .cuda_graph_begin_capture(self.stream.raw)?;
        }
        let launched = unsafe { self.enqueue(p.rows) };
        let captured = unsafe { self.stream.library.cuda_graph_end_capture(self.stream.raw) };
        match (launched, captured) {
            (Ok(()), Ok(g)) => self.graph = Some((g, p.rows, state.owner)),
            (Err(e), Ok(g)) => {
                unsafe {
                    self.stream.library.cuda_graph_exec_destroy(g)?;
                }
                return Err(e);
            }
            (Err(e), Err(_)) | (Ok(()), Err(e)) => return Err(e),
        }
        Ok(())
    }
    /// # Safety
    /// Same input contract as execute; capture fixes owner and live row count.
    pub unsafe fn replay<'s>(
        &'s mut self,
        state: &'s WindowState<'_>,
        chunks: &[WindowChunk],
    ) -> Result<WindowOutput<'s>> {
        let p = self.prepare(state, chunks)?;
        let (g, rows, owner) = self.graph.context("window graph missing")?;
        ensure!(
            rows == p.rows && owner == state.owner,
            "window capture binding differs"
        );
        let launched = self.upload(&p).and_then(|()| unsafe {
            self.stream.library.cuda_graph_launch(g, self.stream.raw)
        });
        launched.and(self.synchronize())?;
        self.ready = Some(p);
        self.output(state)
    }
    fn validate_ready(&self, state: &WindowState<'_>) -> Result<&Prepared> {
        let p = self.ready.as_ref().context("window output unpublished")?;
        ensure!(p.owner == state.owner, "window output owner differs");
        for (i, c) in p.chunks.iter().enumerate() {
            let slot = state.validate(c.lease)?;
            ensure!(
                state.slots[slot].version == p.versions[i] && state.slots[slot].end == c.position,
                "stale window output"
            );
        }
        Ok(p)
    }
    pub fn output<'s>(&'s self, state: &'s WindowState<'_>) -> Result<WindowOutput<'s>> {
        let rows = self.validate_ready(state)?.rows;
        Ok(WindowOutput {
            values: slice(self.values.buffer, 0, rows * 512),
            scales: slice(self.scales.buffer, 0, rows * 16),
            projected: slice(self.projected.buffer, 0, rows * 1024),
            normalized: slice(self.normalized.buffer, 0, rows * 1024),
            frequencies: slice(self.frequencies.buffer, 0, rows * 256),
            _owner: PhantomData,
        })
    }
    pub fn proposal<'s>(
        &'s self,
        state: &'s WindowState<'_>,
        lease: WindowLease,
    ) -> Result<WindowProposal<'s>> {
        let p = self.validate_ready(state)?;
        let i = p
            .chunks
            .iter()
            .position(|c| c.lease == lease)
            .context("request absent from window wave")?;
        let c = p.chunks[i];
        let o = self.output(state)?;
        Ok(WindowProposal {
            cache: state.view(lease)?,
            values: o.values,
            scales: o.scales,
            capacity: p.rows,
            request: state.request_id(lease)?,
            layer: self.weights.layer,
            binding: WindowBinding {
                snapshot: p.snapshot,
                lease,
            },
            first: c.position,
            tokens: c.tokens as usize,
            offset: p.offsets[i],
            _wave: PhantomData,
        })
    }
    /// Validate an enclosing cache transaction's exact request order/proposal.
    pub(crate) fn validate_batch(
        &self,
        state: &WindowState<'_>,
        chunks: &[WindowChunk],
    ) -> Result<()> {
        let prepared = self.validate_ready(state)?;
        ensure!(
            prepared.chunks.len() == chunks.len()
                && prepared.chunks.iter().zip(chunks).all(|(a, b)|
                    a.lease == b.lease && a.position == b.position && a.tokens == b.tokens),
            "window transaction proposal differs"
        );
        Ok(())
    }
    /// Consume a proposal once. Only the last 128 accepted rows per request are
    /// written, ensuring unique ring destinations even for large prefill chunks.
    pub fn commit(&mut self, state: &mut WindowState<'_>, accepted: &[u32]) -> Result<()> {
        let checked = self.validate_ready(state).map(|_| ());
        let p = self.ready.take().context("window output unpublished")?;
        checked?;
        ensure!(
            accepted.len() == p.chunks.len(),
            "window acceptance count differs"
        );
        let mut destinations = vec![u64::MAX; p.rows];
        let mut ends = vec![];
        for (i, c) in p.chunks.iter().enumerate() {
            ensure!(
                accepted[i] <= c.tokens,
                "window accepted prefix exceeds proposal"
            );
            state.slots[c.lease.slot]
                .version
                .checked_add(1)
                .context("window version exhausted")?;
            let n = accepted[i] as usize;
            for j in n.saturating_sub(128)..n {
                destinations[p.offsets[i] + j] =
                    (c.lease.slot * 128) as u64 + (c.position + j as u64) % 128;
            }
            ends.push(c.position + n as u64);
        }
        let staging = self.staging.bytes_mut();
        for (i, d) in destinations.iter().enumerate() {
            staging[i * 8..i * 8 + 8].copy_from_slice(&d.to_ne_bytes());
        }
        for (i, end) in ends.iter().enumerate() {
            staging[p.rows * 8 + i * 8..p.rows * 8 + i * 8 + 8].copy_from_slice(&end.to_ne_bytes());
        }
        let launched = (|| -> Result<()> {
            unsafe {
                self.stream.library.copy_h2d_async(
                    self.destinations.buffer,
                    &self.staging.bytes_mut()[..p.rows * 8],
                    self.stream.raw,
                )?;
                self.kv.store(
                    self.values.buffer,
                    self.scales.buffer,
                    self.destinations.buffer,
                    state.values.buffer,
                    state.scales.buffer,
                    p.rows,
                    state.slot_count * 128,
                    self.stream.raw,
                )?;
                for (i, c) in p.chunks.iter().enumerate() {
                    self.stream.library.copy_h2d_async(
                        slice(state.ends.buffer, c.lease.slot * 8, 8),
                        &self.staging.bytes_mut()[p.rows * 8 + i * 8..p.rows * 8 + i * 8 + 8],
                        self.stream.raw,
                    )?;
                }
            }
            Ok(())
        })();
        if let Err(error) = launched.and(self.synchronize()) {
            let leases = p.chunks.iter().map(|c| c.lease).collect::<Vec<_>>();
            if let Err(e) = state.invalidate(&leases) {
                tracing::error!(%e,"invalidating failed window transaction");
            }
            return Err(error);
        }
        for (i, c) in p.chunks.iter().enumerate() {
            state.slots[c.lease.slot].end = ends[i];
            state.slots[c.lease.slot].version += 1;
        }
        Ok(())
    }
    pub fn clear_graph(&mut self) -> Result<()> {
        self.ready = None;
        self.synchronize()?;
        if let Some((g, _, _)) = self.graph.take() {
            unsafe {
                self.stream.library.cuda_graph_exec_destroy(g)?;
            }
        }
        Ok(())
    }
}
impl Drop for WindowWave<'_, '_> {
    fn drop(&mut self) {
        if let Err(error) = self.clear_graph() {
            tracing::error!(%error,"draining window graph");
        }
    }
}

#[cfg(test)]
mod replay_tests {
    use super::*;

    #[test]
    fn decoder_replay_window_lease_boundaries() -> Result<()> {
        let Some(path) = std::env::var_os("DS41RT_WINDOW_REPLAY_LIBRARY") else {
            eprintln!("skip GPU window replay test: DS41RT_WINDOW_REPLAY_LIBRARY unset");
            return Ok(());
        };
        let library = unsafe { NativeLibrary::load(path)? };
        let mut state = WindowState::new(&library, 20, 16, WindowState::device_bytes(20, 16)?)?;
        let mut encoder = WindowState::new(&library, 19, 1, WindowState::device_bytes(19, 1)?)?;
        let encoder_lease = encoder.begin_request(0, 999)?;
        assert!(encoder.begin_replay(encoder_lease, 128).is_err());
        assert!(state.begin_replay(encoder_lease, 128).is_err());
        for slot in 0..16 {
            let lease = state.begin_request(slot, slot as u64 + 1)?;
            let before = state.slots[slot].version;
            let position = [0, 1, 127, 128, 16384, 1048576][slot % 6];
            assert!(state.begin_replay(lease, 1048577).is_err());
            assert_eq!(state.end(lease)?, 0);
            state.begin_replay(lease, position)?;
            assert_eq!(state.slots[slot].version, before + 1);
            assert!(state.begin_replay(lease, position).is_err());
            let view = state.view(lease)?;
            assert_eq!((view.begin, view.end), (position, position));
            let mut bytes = [0; 8];
            library.copy_d2h(&mut bytes, view.device_end)?;
            assert_eq!(u64::from_ne_bytes(bytes), position);
            state.release(lease)?;
            assert!(state.begin_replay(lease, 0).is_err());
            let replacement = state.begin_request(slot, slot as u64 + 101)?;
            assert!(state.view(lease).is_err());
            assert_eq!((state.view(replacement)?.begin, state.end(replacement)?), (0, 0));
            state.release(replacement)?;
        }
        eprintln!("PASS 16 decoder replay slots: bounds, device ends, versions, stale leases, encoder/foreign guards and reuse");
        Ok(())
    }
}
