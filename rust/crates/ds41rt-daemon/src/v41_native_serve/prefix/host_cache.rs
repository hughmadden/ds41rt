//! Host snapshot cache binding: the CUDA copy engine over the FFI and the glue between the
//! engine's retained snapshots (`Saved`) and `ds41rt_hostcache::HostCache`. Design of record:
//! recipes `dsv41-flash-tp4-engram/research/afd-hostcache-design.md` §4.6. Everything here runs
//! on the scheduler thread; the GPU copies asynchronously on two dedicated streams.
//!
//! A retained snapshot's device bytes are its compressor source pages (each page's rows in four
//! device buffers), its backbone tail in an arena slot and its dSpark rings. Everything else in
//! a `Saved` is small host data and travels as the cache payload (`HostSaved`), so restoring
//! is: allocate fresh pages and arena slots, copy the bytes back, rebuild the `Saved` from parts
//! and let the engine's own restore logic run unchanged.
use super::*;
use crate::v41_backbone_cache::BackbonePrefix;
use crate::v41_compressor::CompressorPrefix;
use crate::v41_dspark_cache::DsparkPrefix;
use crate::v41_memory::SnapshotStorage;
use crate::v41_window::WindowPrefix;
use ds41rt_core::EngramHistory;
use ds41rt_ffi::{Ds41rtDeviceBuffer, Ds41rtHostBuffer, NativeLibrary};
use ds41rt_hostcache::cache::{
    DevicePage, DeviceSnapshot, EvictDecision, HostCache, RestoreOutcome, RestoreTarget,
    StoreOutcome, StoreTicket,
};
use ds41rt_hostcache::config::Config;
use ds41rt_hostcache::copy::{CopyEngine, DeviceRange, Event, Stream};
use ds41rt_hostcache::metrics::Snapshot as MetricsSnapshot;
use ds41rt_hostcache::pool::{HostChunk, HostRange, Layout, PinnedMemory};
use ds41rt_hostcache::snapshot::{DevicePageId, Hit, SnapshotMeta};
use ds41rt_hostcache::COMPRESSORS;
use std::collections::VecDeque;
use std::ffi::c_void;
use std::time::{Duration, Instant};

/// The engine-side descriptors of a snapshot: everything in a `Saved` that is not device bytes.
/// `images` pins the snapshot's image key ids for as long as the host copy can be looked up.
pub(super) struct HostSaved {
    images: ImageKeys,
    history: EngramHistory,
    next: TokenScores,
    owner: u64,
    end: u64,
    /// `WindowPrefix` parts per backbone window.
    windows: Vec<(u64, u64, u64)>,
    /// Per compressor: owner, end, page count, rows.
    sources: Vec<(u64, u64, usize, usize)>,
    /// Per dSpark window: owner, end, ring bytes.
    draft: Option<Vec<(u64, u64, usize)>>,
}

fn range(buffer: Ds41rtDeviceBuffer) -> DeviceRange {
    DeviceRange {
        addr: buffer.ptr as u64,
        bytes: buffer.bytes,
    }
}

/// One CUDA stream with its completion probe: `cuda_stream_wait_event` on the probe followed by
/// `cuda_stream_query` answers "has this event completed" without blocking and without an
/// event-query entry point in the FFI. Events complete in issue order on a stream.
struct StreamState {
    raw: *mut c_void,
    probe: *mut c_void,
    pending: VecDeque<(u64, *mut c_void)>,
    probing: Option<u64>,
}

pub(crate) struct CudaCopyEngine<'a> {
    library: &'a NativeLibrary,
    template: Ds41rtDeviceBuffer,
    chunks: Vec<Option<Ds41rtHostBuffer>>,
    streams: [StreamState; 2],
    next_event: u64,
    started: Instant,
}

impl<'a> CudaCopyEngine<'a> {
    /// `template` supplies the device id and flags every engine buffer carries.
    pub fn new(library: &'a NativeLibrary, template: Ds41rtDeviceBuffer) -> Result<Self> {
        let mut streams = Vec::with_capacity(2);
        for _ in 0..2 {
            streams.push(StreamState {
                raw: library.cuda_stream_create()?,
                probe: library.cuda_stream_create()?,
                pending: VecDeque::new(),
                probing: None,
            });
        }
        let streams = streams.try_into().ok().expect("two streams");
        Ok(Self {
            library,
            template,
            chunks: Vec::new(),
            streams,
            next_event: 0,
            started: Instant::now(),
        })
    }
    fn state(&mut self, stream: Stream) -> &mut StreamState {
        &mut self.streams[stream as usize]
    }
    fn device(&self, range: DeviceRange) -> Ds41rtDeviceBuffer {
        Ds41rtDeviceBuffer {
            ptr: range.addr as *mut c_void,
            bytes: range.bytes,
            ..self.template
        }
    }
    fn host(&self, range: HostRange) -> Result<Ds41rtHostBuffer> {
        let chunk = self
            .chunks
            .get(range.chunk as usize)
            .and_then(Option::as_ref)
            .context("host cache chunk released")?;
        ensure!(
            range.offset + range.bytes <= chunk.bytes,
            "host range outside its chunk"
        );
        Ok(Ds41rtHostBuffer {
            ptr: unsafe { chunk.ptr.cast::<u8>().add(range.offset).cast() },
            bytes: range.bytes,
            flags: chunk.flags,
        })
    }
    /// Drain a stream (used before releasing device memory a timed-out restore may still write).
    pub fn synchronize(&mut self, stream: Stream) -> Result<()> {
        let raw = self.state(stream).raw;
        unsafe { self.library.cuda_stream_synchronize(raw) }
    }
    fn stream_of(&self, event: Event) -> Option<Stream> {
        [Stream::Store, Stream::Restore].into_iter().find(|&s| {
            self.streams[s as usize]
                .pending
                .iter()
                .any(|&(id, _)| id == event.0)
        })
    }
}

impl PinnedMemory for CudaCopyEngine<'_> {
    fn allocate_chunk(&mut self, bytes: usize) -> Result<HostChunk> {
        let buffer = self.library.alloc_host_buffer(bytes)?;
        self.chunks.push(Some(buffer));
        Ok(HostChunk {
            id: (self.chunks.len() - 1) as u32,
            bytes,
        })
    }
    fn release_chunk(&mut self, chunk: HostChunk) -> Result<()> {
        let mut buffer = self
            .chunks
            .get_mut(chunk.id as usize)
            .and_then(Option::take)
            .context("host cache chunk already released")?;
        self.library.free_host_buffer(&mut buffer)
    }
}

impl CopyEngine for CudaCopyEngine<'_> {
    fn d2h(&mut self, stream: Stream, src: DeviceRange, dst: HostRange) -> Result<()> {
        ensure!(src.bytes == dst.bytes, "copy length mismatch");
        let (host, device, raw) = (self.host(dst)?, self.device(src), self.state(stream).raw);
        unsafe {
            self.library
                .copy_d2h_host_buffer_async(host, device, src.bytes, raw)
        }
    }
    fn h2d(&mut self, stream: Stream, src: HostRange, dst: DeviceRange) -> Result<()> {
        ensure!(src.bytes == dst.bytes, "copy length mismatch");
        let (host, device, raw) = (self.host(src)?, self.device(dst), self.state(stream).raw);
        unsafe {
            self.library
                .copy_host_buffer_h2d_async(device, host, src.bytes, raw)
        }
    }
    fn record(&mut self, stream: Stream) -> Result<Event> {
        let event = self.library.cuda_event_create()?;
        let raw = self.state(stream).raw;
        unsafe { self.library.cuda_event_record(event, raw)? };
        self.next_event += 1;
        let id = self.next_event;
        self.state(stream).pending.push_back((id, event));
        Ok(Event(id))
    }
    fn completed(&mut self, event: Event) -> Result<bool> {
        let Some(stream) = self.stream_of(event) else {
            return Ok(true);
        };
        let library = self.library;
        let state = self.state(stream);
        let &(_, raw_event) = state
            .pending
            .iter()
            .find(|&&(id, _)| id == event.0)
            .expect("event pending");
        if state.probing != Some(event.0) {
            unsafe { library.cuda_stream_wait_event(state.probe, raw_event)? };
            state.probing = Some(event.0);
        }
        if !unsafe { library.cuda_stream_query(state.probe)? } {
            return Ok(false);
        }
        // In-order stream: everything up to and including this event is done.
        while let Some(&(id, raw)) = state.pending.front() {
            unsafe { library.cuda_event_destroy(raw)? };
            state.pending.pop_front();
            if id == event.0 {
                break;
            }
        }
        state.probing = None;
        Ok(true)
    }
    fn wait(&mut self, event: Event, budget_ns: u64) -> Result<bool> {
        let deadline = self.now_ns().saturating_add(budget_ns);
        loop {
            if self.completed(event)? {
                return Ok(true);
            }
            if self.now_ns() >= deadline {
                return Ok(false);
            }
            std::thread::sleep(Duration::from_micros(20));
        }
    }
    fn now_ns(&self) -> u64 {
        self.started.elapsed().as_nanos() as u64
    }
}

impl Drop for CudaCopyEngine<'_> {
    fn drop(&mut self) {
        for state in &mut self.streams {
            for &(_, event) in &state.pending {
                let _ = unsafe { self.library.cuda_event_destroy(event) };
            }
            let _ = unsafe { self.library.cuda_stream_destroy(state.probe) };
            let _ = unsafe { self.library.cuda_stream_destroy(state.raw) };
        }
        for buffer in self.chunks.iter_mut().flatten() {
            let _ = self.library.free_host_buffer(buffer);
        }
    }
}

/// The glue: builds `DeviceSnapshot`s from `Saved`s and `Saved`s from restored bytes.
pub(crate) struct HostCacheBinding<'a> {
    cache: HostCache<CudaCopyEngine<'a>, HostSaved>,
}

impl<'a> HostCacheBinding<'a> {
    /// `None` when the cache is disabled: the engine's paths stay untouched.
    pub fn new(
        library: &'a NativeLibrary,
        config: Config,
        template: Ds41rtDeviceBuffer,
    ) -> Result<Option<Self>> {
        config.validate()?;
        if !config.enabled() {
            return Ok(None);
        }
        let engine = CudaCopyEngine::new(library, template)?;
        let cache = HostCache::new(config, Layout::engine(0), engine)?;
        tracing::info!(target: "ds41rt::host_cache", config = ?cache.config(), "host snapshot cache enabled");
        Ok(Some(Self { cache }))
    }

    fn describe(
        &self,
        kind: SnapshotKind,
        keys: &[u32],
        saved: &Saved<'a>,
        requests: &Requests<'a>,
    ) -> Result<(DeviceSnapshot, HostSaved)> {
        let (backbone, history) = saved.target.parts();
        let (owner, end, tail, windows, sources) = backbone.parts();
        let caches = requests.cache().sources();
        let mut pages: [Vec<DevicePage>; COMPRESSORS] = Default::default();
        let mut source_parts = Vec::with_capacity(COMPRESSORS);
        for (c, prefix) in sources.iter().enumerate() {
            let (source_owner, source_end, source) = prefix.parts();
            let cache = caches[c].source_cache();
            pages[c] = source
                .pages()
                .iter()
                .map(|&page| DevicePage {
                    id: DevicePageId {
                        compressor: c as u8,
                        page,
                        generation: cache.page_generation(page),
                    },
                    segments: cache.page_segments(page).into_iter().map(range).collect(),
                })
                .collect();
            source_parts.push((
                source_owner,
                source_end,
                source.pages().len(),
                source.rows(),
            ));
        }
        let draft = saved.draft.as_ref().map(|d| {
            d.parts()
                .iter()
                .map(|p| p.parts())
                .map(|(o, e, ring)| ((o, e, ring.buffer.bytes), range(ring.buffer)))
                .unzip::<_, _, Vec<_>, Vec<_>>()
        });
        let (draft_parts, draft_ranges) = match draft {
            Some((parts, ranges)) => (Some(parts), Some(ranges)),
            None => (None, None),
        };
        let snapshot = DeviceSnapshot {
            meta: SnapshotMeta {
                kind,
                tokens: keys.to_vec(),
                end: end as u32,
                has_draft: draft_ranges.is_some(),
            },
            pages,
            tail: vec![range(tail.buffer)],
            draft: draft_ranges,
            scores: vec![],
        };
        let payload = HostSaved {
            images: saved._images.through(end as usize),
            history: history.fork()?,
            next: saved.next.clone(),
            owner,
            end,
            windows: windows.iter().map(WindowPrefix::parts).collect(),
            sources: source_parts,
            draft: draft_parts,
        };
        Ok((snapshot, payload))
    }

    /// Issue the write-behind copy of a freshly retained snapshot; the ticket lives in the `Saved`.
    pub(super) fn store(
        &mut self,
        kind: SnapshotKind,
        keys: &[u32],
        saved: &Saved<'a>,
        requests: &Requests<'a>,
    ) -> Result<Option<StoreTicket>> {
        let (snapshot, payload) = self.describe(kind, keys, saved, requests)?;
        Ok(match self.cache.store(&snapshot, payload) {
            StoreOutcome::Issued(ticket) | StoreOutcome::Deferred(ticket) => Some(ticket),
            StoreOutcome::Skipped(_) => None,
        })
    }
    /// The key-space tokens a resident host snapshot is keyed by.
    pub(super) fn snapshot_tokens(&self, key: ds41rt_hostcache::snapshot::Key) -> Option<Vec<u32>> {
        self.cache.snapshot_tokens(key).map(<[u32]>::to_vec)
    }
    pub(super) fn tick(&mut self) {
        self.cache.tick();
    }
    /// A host hit whose restore could not be carried out (for example no device pages for it):
    /// counted with the copy failures so `/v1/stats` shows every abandoned restore; the request
    /// prefills instead.
    pub(super) fn count_abandoned_restore(&mut self) {
        self.cache.metrics_mut().get_mut().restore_failures += 1;
    }
    /// The engine is dropping a `Saved`: let its copy finish within budget or count the loss.
    pub(super) fn before_evict(&mut self, ticket: Option<StoreTicket>) -> EvictDecision {
        self.cache.before_device_evict(ticket)
    }
    pub(super) fn lookup(&mut self, keys: &[u32]) -> Option<Hit> {
        self.cache.lookup(keys)
    }
    /// Rebuild a `Saved` from the host copy. `Ok(None)` when the restore timed out or failed (the
    /// caller prefills); allocations are released only after the restore stream drained.
    pub(super) fn restore(
        &mut self,
        hit: &Hit,
        requests: &Requests<'a>,
        draft: Option<&DraftRuntime<'_, 'a>>,
    ) -> Result<Option<Saved<'a>>> {
        // Take what the rebuilt `Saved` needs out of the payload before the mutable restore call.
        let (owner, end, windows, source_parts, draft_parts, history, next, images) = {
            let payload = self
                .cache
                .payload(hit.key)
                .context("host cache hit without payload")?;
            (
                payload.owner,
                payload.end,
                payload.windows.clone(),
                payload.sources.clone(),
                payload.draft.clone(),
                payload.history.fork()?,
                payload.next.clone(),
                payload.images.through(payload.end as usize),
            )
        };
        let backbone = requests.cache();
        let caches = backbone.sources();
        ensure!(
            draft_parts.is_some() == draft.is_some(),
            "host snapshot execution mode differs"
        );
        let mut sources = Vec::with_capacity(COMPRESSORS);
        let mut pages: [Vec<DevicePage>; COMPRESSORS] = Default::default();
        for (c, &(owner, end, count, rows)) in source_parts.iter().enumerate() {
            let cache = caches[c].source_cache();
            let source = cache.allocate_prefix(count, rows)?;
            pages[c] = source
                .pages()
                .iter()
                .map(|&page| DevicePage {
                    id: DevicePageId {
                        compressor: c as u8,
                        page,
                        generation: cache.page_generation(page),
                    },
                    segments: cache.page_segments(page).into_iter().map(range).collect(),
                })
                .collect();
            sources.push(CompressorPrefix::from_parts(owner, end, source));
        }
        let tail = SnapshotStorage::new(
            backbone.prefix_library(),
            BackbonePrefix::device_bytes(),
            backbone.prefix_pool(),
        )?;
        let rings = match (&draft_parts, draft) {
            (Some(parts), Some(runtime)) => Some(
                parts
                    .iter()
                    .zip(runtime.windows())
                    .map(|(&(_, _, bytes), window)| {
                        SnapshotStorage::new(window.library(), bytes, window.prefix_pool())
                    })
                    .collect::<Result<Vec<_>>>()?,
            ),
            _ => None,
        };
        let target = RestoreTarget {
            pages,
            tail: vec![range(tail.buffer)],
            draft: rings
                .as_ref()
                .map(|rings| rings.iter().map(|r| range(r.buffer)).collect()),
            scores: vec![],
        };
        match self.cache.restore(hit.key, &target) {
            RestoreOutcome::Done { .. } => {}
            outcome => {
                tracing::warn!(target: "ds41rt::host_cache", ?outcome, "host restore did not complete; prefilling");
                self.cache.engine_mut().synchronize(Stream::Restore)?;
                return Ok(None);
            }
        }
        let windows = windows
            .iter()
            .map(|&(o, e, b)| WindowPrefix::from_parts(o, e, b))
            .collect();
        let target_prefix = RequestPrefix::from_parts(
            BackbonePrefix::from_parts(owner, end, tail, windows, sources),
            history,
        );
        let draft = match (draft_parts.as_ref(), rings) {
            (Some(parts), Some(rings)) => Some(DraftPrefix::from_parts(
                parts
                    .iter()
                    .zip(rings)
                    .map(|(&(o, e, _), ring)| DsparkPrefix::from_parts(o, e, ring))
                    .collect(),
            )),
            _ => None,
        };
        Ok(Some(Saved {
            _images: images,
            target: target_prefix,
            draft,
            next,
            ticket: None,
        }))
    }
    pub(super) fn metrics(&self) -> MetricsSnapshot {
        self.cache.metrics()
    }
}
