//! Packed FP8 committed dSpark KV rings with generation-checked request slots.
use crate::v41_memory::{DeviceAllocation, HostAllocation, LoadStream};
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{
    Ds41rtDeviceBuffer, NativeLibrary, V41AttentionWindow, V41DsparkCache, V41KvWrite,
};
use std::{
    ffi::c_void,
    cell::Cell,
    rc::Rc,
    sync::atomic::{AtomicU64, Ordering},
};
static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);
mod prefix;
mod queued;
pub(crate) use queued::WindowWrite;
pub(crate) use prefix::DsparkPrefix;
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct WindowLease {
    owner: u64,
    slot: usize,
    generation: u64,
}
#[derive(Clone, Copy)]
pub(crate) struct WindowChunk {
    pub lease: WindowLease,
    pub position: u64,
    pub source_row: u32,
    pub tokens: u32,
}
#[derive(Clone, Copy, Default)]
struct Slot {
    generation: u64,
    request: Option<u64>,
    end: Option<u64>,
}
pub(crate) struct WindowView {
    pub buffer: Ds41rtDeviceBuffer,
    pub valid_rows: usize,
    pub committed_end: u64,
}
pub(crate) struct WindowRead {
    pub owner: u64,
    pub ring: Ds41rtDeviceBuffer,
    pub slots: u32,
    pub descriptors: [V41AttentionWindow; 16],
    _reservation: ReadReservation,
}
// Reservations outlive the short host borrow while a chain reads ring slots.
// Other request slots may be committed, but a live read slot cannot be rewritten
// or recycled until the chain has drained its GPU work.
#[derive(Clone, Default)]
struct SlotAccess(Rc<Cell<[u32; 16]>>);
struct ReadReservation { slots: SlotAccess, used: [bool; 16] }
const WRITE_RESERVED: u32 = u32::MAX;
struct WriteReservation { slots: SlotAccess, used: [bool; 16] }
impl SlotAccess {
    fn writable(&self, slot: usize) -> Result<()> {
        ensure!(self.0.get()[slot] == 0, "dSpark cache slot has outstanding access");
        Ok(())
    }
    fn readable(&self, slot: usize) -> Result<()> {
        ensure!(self.0.get()[slot] != WRITE_RESERVED, "dSpark cache write is unpublished");
        Ok(())
    }
    fn reserve_write(&self, used: [bool; 16]) -> Result<WriteReservation> {
        let mut counts = self.0.get();
        for (i, active) in used.iter().enumerate() {
            if *active { self.writable(i)?; counts[i] = WRITE_RESERVED; }
        }
        self.0.set(counts);
        Ok(WriteReservation { slots: self.clone(), used })
    }
    fn reserve(&self, used: [bool; 16]) -> Result<ReadReservation> {
        let mut counts = self.0.get();
        for (i, active) in used.iter().enumerate() {
            if *active {
                ensure!(counts[i] < WRITE_RESERVED-1, "dSpark slot is reserved or reader count exhausted");
                counts[i] += 1;
            }
        }
        self.0.set(counts);
        Ok(ReadReservation { slots: self.clone(), used })
    }
}
impl Drop for ReadReservation {
    fn drop(&mut self) {
        let mut counts = self.slots.0.get();
        for (i, active) in self.used.iter().enumerate() {
            if *active { counts[i] -= 1; }
        }
        self.slots.0.set(counts);
    }
}
impl Drop for WriteReservation {
    fn drop(&mut self) {
        let mut counts = self.slots.0.get();
        for (i, active) in self.used.iter().enumerate() {
            if *active { debug_assert_eq!(counts[i], WRITE_RESERVED); counts[i] = 0; }
        }
        self.slots.0.set(counts);
    }
}
pub(crate) struct DsparkWindow<'a> {
    prefix_copies: [crate::v41_memory::SnapshotCopies<'a, (WindowLease, DsparkPrefix<'a>, ReadReservation)>; 2],
    prefix_pool: Option<crate::v41_memory::SnapshotPool<'a>>,
    stream: LoadStream<'a>,
    kernel: V41DsparkCache<'a>,
    source: DeviceAllocation<'a>,
    ring: DeviceAllocation<'a>,
    descriptors: DeviceAllocation<'a>,
    staging: HostAllocation<'a>,
    graph: Option<*mut c_void>,
    slots: [Slot; 16],
    access: SlotAccess,
    slot_count: usize,
    source_rows: u32,
    owner: u64,
}
impl<'a> DsparkWindow<'a> {
    pub fn device(&self) -> crate::v41_memory::device::Device<'a> {
        crate::v41_memory::device::Device { library: self.stream.library, id: self.ring.buffer.device_id }
    }
    pub fn device_bytes(slots: usize, source_rows: u32) -> Result<usize> {
        ensure!(
            (1..=16).contains(&slots) && (1..=4096).contains(&source_rows),
            "invalid dSpark window capacity"
        );
        Ok(slots * V41DsparkCache::SLOT_BYTES + source_rows as usize * 1024 + 384)
    }
    pub fn new(
        library: &'a NativeLibrary,
        slots: usize,
        source_rows: u32,
        budget: usize,
    ) -> Result<Self> {
        ensure!(
            Self::device_bytes(slots, source_rows)? <= budget,
            "dSpark window exceeds budget"
        );
        let owner = NEXT_OWNER
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| anyhow::anyhow!("dSpark cache owner IDs exhausted"))?;
        let mut value = Self {
            prefix_copies: [crate::v41_memory::SnapshotCopies::new(library)?,
                crate::v41_memory::SnapshotCopies::new(library)?],
            prefix_pool: None,
            stream: LoadStream {
                library,
                raw: library.cuda_stream_create()?,
            },
            kernel: library.v41_dspark_cache()?,
            source: DeviceAllocation::new(library, source_rows as usize * 1024)?,
            ring: DeviceAllocation::new(library, slots * V41DsparkCache::SLOT_BYTES)?,
            descriptors: DeviceAllocation::new(library, 384)?,
            staging: HostAllocation::new(library, 384)?,
            graph: None,
            slots: [Slot::default(); 16],
            access: SlotAccess::default(),
            slot_count: slots,
            source_rows,
            owner,
        };
        // Inactive descriptors warm/capture the fixed launch without reading source.
        library.copy_h2d(value.descriptors.buffer, value.staging.bytes_mut())?;
        unsafe {
            value.enqueue()?;
        }
        value.synchronize()?;
        unsafe {
            library.cuda_graph_begin_capture(value.stream.raw)?;
        }
        let launched = unsafe { value.enqueue() };
        let captured = unsafe { library.cuda_graph_end_capture(value.stream.raw) };
        match (launched, captured) {
            (Ok(()), Ok(graph)) => value.graph = Some(graph),
            (Err(error), Ok(graph)) => {
                unsafe {
                    library.cuda_graph_exec_destroy(graph)?;
                }
                return Err(error);
            }
            (Err(error), Err(_)) | (Ok(()), Err(error)) => return Err(error),
        }
        Ok(value)
    }
    fn synchronize(&self) -> Result<()> {
        unsafe { self.stream.library.cuda_stream_synchronize(self.stream.raw) }
    }
    unsafe fn enqueue(&self) -> Result<()> {
        unsafe {
            self.kernel.launch(
                self.source.buffer,
                self.descriptors.buffer,
                self.ring.buffer,
                self.source_rows,
                self.slot_count as u32,
                self.stream.raw,
            )
        }
    }
    pub fn begin_request(&mut self, slot: usize, request: u64) -> Result<WindowLease> {
        ensure!(
            slot < self.slot_count,
            "dSpark window slot exceeds capacity"
        );
        ensure!(
            self.slots[slot].request.is_none(),
            "dSpark window slot is occupied"
        );
        self.access.writable(slot)?;
        let generation = self.slots[slot]
            .generation
            .checked_add(1)
            .context("cache generation exhausted")?;
        self.slots[slot] = Slot {
            generation,
            request: Some(request),
            end: None,
        };
        Ok(WindowLease {
            owner: self.owner,
            slot,
            generation,
        })
    }
    fn validate(&self, lease: WindowLease) -> Result<usize> {
        ensure!(
            lease.owner == self.owner && lease.slot < self.slot_count,
            "foreign dSpark cache lease"
        );
        let entry = self.slots[lease.slot];
        ensure!(
            entry.generation == lease.generation && entry.request.is_some(),
            "stale dSpark cache lease"
        );
        Ok(lease.slot)
    }
    /// Stable logical request identity after owner/generation validation.
    pub fn request_id(&self, lease: WindowLease) -> Result<u64> {
        let slot = self.validate(lease)?;
        self.slots[slot].request.context("dSpark cache request missing")
    }
    /// None denotes an admitted request with no published main context yet.
    pub fn committed_end(&self, lease: WindowLease) -> Result<Option<u64>> {
        Ok(self.slots[self.validate(lease)?].end)
    }
    pub fn release(&mut self, lease: WindowLease) -> Result<()> {
        let slot = self.validate(lease)?;
        self.access.writable(slot)?;
        self.slots[slot].request = None;
        self.slots[slot].end = None;
        Ok(())
    }
    #[cfg(test)]
    pub(crate) fn ring_for_test(&self) -> Ds41rtDeviceBuffer { self.ring.buffer }
    /// Stable BF16 [source_rows,512] destination for normalized, RoPE-applied KV.
    pub fn source(&self) -> Ds41rtDeviceBuffer {
        self.source.buffer
    }
    fn prepare_write(&self, chunks: &[WindowChunk], source_rows: u32)
        -> Result<([V41KvWrite; 16], [bool; 16], [Option<u64>; 16])>
    {
        ensure!(source_rows > 0 && source_rows <= self.source_rows, "invalid produced cache rows");
        ensure!(
            !chunks.is_empty() && chunks.len() <= self.slot_count,
            "invalid cache chunk count"
        );
        let mut seen = [false; 16];
        let mut descriptors = [V41KvWrite::default(); 16];
        let mut ends = [None; 16];
        for (i, chunk) in chunks.iter().enumerate() {
            let slot = self.validate(chunk.lease)?;
            self.access.writable(slot)?;
            ensure!(!seen[slot], "duplicate cache slot in batch");
            seen[slot] = true;
            ensure!(
                chunk.tokens > 0
                    && chunk.source_row < source_rows
                    && chunk.tokens <= source_rows - chunk.source_row,
                "invalid cache source span"
            );
            let end = chunk
                .position
                .checked_add(u64::from(chunk.tokens))
                .context("cache position overflow")?;
            if let Some(prior) = self.slots[slot].end {
                ensure!(
                    chunk.position == prior,
                    "non-contiguous committed cache append"
                );
            } else {
                ensure!(
                    chunk.position == 0 || chunk.tokens >= 128,
                    "cache tail seed must fill the window"
                );
            }
            descriptors[i] = V41KvWrite {
                position: chunk.position,
                source_row: chunk.source_row,
                token_count: chunk.tokens,
                slot: slot as u32,
                reserved: 0,
            };
            ends[slot] = Some(end);
        }
        Ok((descriptors, seen, ends))
    }
    /// Prevalidate an entire producer batch without changing cache state.
    pub fn validate_write(&self, chunks: &[WindowChunk], source_rows: u32) -> Result<()> {
        self.prepare_write(chunks, source_rows).map(|_| ())
    }
    /// # Safety
    /// Source rows must be initialized finite KV with producer writes completed;
    /// serialize source use and unreserved raw ring consumers. Reserved reads of
    /// other slots may remain in flight; overlapping writes are rejected. Chunks
    /// contain only committed main-model positions, never unaccepted draft KV.
    pub unsafe fn write(&mut self, chunks: &[WindowChunk]) -> Result<()> {
        let (descriptors, seen, ends) = self.prepare_write(chunks, self.source_rows)?;
        let bytes = unsafe { std::slice::from_raw_parts(descriptors.as_ptr().cast::<u8>(), 384) };
        self.staging.bytes_mut().copy_from_slice(bytes);
        let launched = (|| unsafe {
            self.stream
                .library
                .copy_h2d(self.descriptors.buffer, self.staging.bytes_mut())?;
            self.stream
                .library
                .cuda_graph_launch(self.graph.context("cache graph missing")?, self.stream.raw)
        })();
        let drained = self.synchronize();
        if let Err(error) = launched.and(drained) {
            // A partial GPU write cannot remain visible through old logical state.
            for slot in 0..self.slot_count {
                if seen[slot] {
                    self.slots[slot].request = None;
                    self.slots[slot].end = None;
                }
            }
            return Err(error);
        }
        for slot in 0..self.slot_count {
            if seen[slot] {
                self.slots[slot].end = ends[slot];
            }
        }
        Ok(())
    }
    /// Reserve the validated slots through GPU completion. The caller must keep
    /// the window owner alive; disjoint slots remain available to other lanes.
    pub fn attention_read(&self, requests: &[(WindowLease, u64)]) -> Result<WindowRead> {
        ensure!(
            (1..=16).contains(&requests.len()),
            "invalid attention request count"
        );
        let mut seen = [false; 16];
        let mut descriptors = [V41AttentionWindow::default(); 16];
        for (i, &(lease, expected_end)) in requests.iter().enumerate() {
            let slot = self.validate(lease)?;
            ensure!(!seen[slot], "duplicate attention cache slot");
            seen[slot] = true;
            let view = self.view(lease)?;
            ensure!(
                view.committed_end == expected_end,
                "attention cache position changed"
            );
            ensure!(
                expected_end >= 2 && expected_end.checked_add(5).is_some(),
                "invalid attention draft positions"
            );
            descriptors[i] = V41AttentionWindow {
                slot: slot as u32,
                valid_rows: view.valid_rows as u32,
            };
        }
        Ok(WindowRead {
            owner: self.owner,
            ring: self.ring.buffer,
            slots: self.slot_count as u32,
            descriptors,
            _reservation: self.access.reserve(seen)?,
        })
    }
    /// Packed [128,528] bytes: E4M3 values and E8M0 K32 scales per row.
    /// Physical ring order: attention reads 0..valid_rows, then its five private
    /// draft positions. No chronological reordering of this buffer is necessary.
    pub fn view(&self, lease: WindowLease) -> Result<WindowView> {
        let slot = self.validate(lease)?;
        self.access.readable(slot)?;
        let end = self.slots[slot]
            .end
            .context("dSpark window has not been seeded")?;
        let mut buffer = self.ring.buffer;
        buffer.ptr = unsafe { buffer.ptr.cast::<u8>().add(slot * V41DsparkCache::SLOT_BYTES).cast() };
        buffer.bytes = V41DsparkCache::SLOT_BYTES;
        Ok(WindowView {
            buffer,
            valid_rows: end.min(128) as usize,
            committed_end: end,
        })
    }
}
impl Drop for DsparkWindow<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.synchronize() {
            tracing::error!(%error,"draining dSpark window");
        }
        if let Some(graph) = self.graph.take() {
            if let Err(error) = unsafe { self.stream.library.cuda_graph_exec_destroy(graph) } {
                tracing::error!(%error,"destroying dSpark window graph");
            }
        }
    }
}

#[cfg(test)]
mod reservation_tests {
    use super::*;
    #[test]
    fn readers_block_only_their_slots_until_last_consumer_finishes() -> Result<()> {
        let slots = SlotAccess::default();
        let mut used = [false; 16]; used[3] = true; used[9] = true;
        let first = slots.reserve(used)?;
        let second = slots.reserve(used)?;
        assert!(slots.writable(3).is_err());
        assert!(slots.writable(9).is_err());
        slots.writable(4)?;
        drop(first);
        assert!(slots.writable(3).is_err());
        drop(second);
        slots.writable(3)?; slots.writable(9)?;
        Ok(())
    }
    #[test]
    fn writers_exclude_readers_and_recycle_without_blocking_peer_slots() -> Result<()> {
        let slots = SlotAccess::default();
        let mask = |slot| std::array::from_fn(|i| i == slot);
        let first = slots.reserve_write(mask(3))?;
        assert!(slots.readable(3).is_err());
        assert!(slots.reserve(mask(3)).is_err());
        assert!(slots.reserve_write(mask(3)).is_err());
        assert!(slots.writable(3).is_err());
        let peer = slots.reserve_write(mask(9))?;
        let reader = slots.reserve(mask(4))?;
        assert!(slots.reserve_write(mask(4)).is_err());
        // Failed multi-slot reservations must not reserve the earlier free slot.
        let overlapping = std::array::from_fn(|i| i == 1 || i == 3);
        assert!(slots.reserve_write(overlapping).is_err());
        assert!(slots.reserve(overlapping).is_err());
        slots.writable(1)?;
        drop(first);
        slots.readable(3)?;
        slots.writable(3)?;
        assert!(slots.readable(9).is_err());
        drop(peer);
        drop(reader);
        assert_eq!(slots.0.get(), [0; 16]);
        Ok(())
    }
    #[test]
    #[ignore = "requires DS41RT_NATIVE_LIB and CUDA"]
    fn native_queued_writes_publish_independently_and_revoke_partial_transactions() -> Result<()> {
        let lib = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        let mut window = DsparkWindow::new(&lib, 2, 2, usize::MAX)?;
        let first = window.begin_request(0, 11)?;
        let second = window.begin_request(1, 22)?;
        lib.copy_h2d(window.source(), &vec![0; 2048])?;
        let chunk = |lease, position| WindowChunk { lease, position, source_row: 0, tokens: 2 };
        unsafe { window.write(&[chunk(first, 0), chunk(second, 0)])?; }
        let first_write = window.prepare_async_write(&[chunk(first, 2)], 2)?;
        let second_write = window.prepare_async_write(&[chunk(second, 2)], 2)?;
        assert!(window.attention_read(&[(first, 2)]).is_err());
        assert!(window.retain_prefix(first).is_err());
        assert!(window.release(first).is_err());
        assert_eq!(window.committed_end(first)?, Some(2));
        let descriptors = [DeviceAllocation::new(&lib, 384)?, DeviceAllocation::new(&lib, 384)?];
        lib.copy_h2d(descriptors[0].buffer, first_write.descriptor_bytes())?;
        lib.copy_h2d(descriptors[1].buffer, second_write.descriptor_bytes())?;
        // Declare streams last: error unwinding drains them before tickets/buffers.
        let streams = [LoadStream { library: &lib, raw: lib.cuda_stream_create()? },
            LoadStream { library: &lib, raw: lib.cuda_stream_create()? }];
        unsafe {
            window.enqueue_write(&first_write, window.source(), descriptors[0].buffer, streams[0].raw)?;
            window.enqueue_write(&second_write, window.source(), descriptors[1].buffer, streams[1].raw)?;
            lib.cuda_stream_synchronize(streams[0].raw)?;
            window.publish_write(first_write)?;
        }
        assert_eq!(window.committed_end(first)?, Some(4));
        assert_eq!(window.committed_end(second)?, Some(2));
        drop(window.attention_read(&[(first, 4)])?);
        assert!(window.attention_read(&[(second, 2)]).is_err());
        unsafe {
            lib.cuda_stream_synchronize(streams[1].raw)?;
            window.publish_write(second_write)?;
        }
        assert_eq!(window.committed_end(second)?, Some(4));
        drop(streams);
        let failed = window.prepare_async_write(&[chunk(first, 4)], 2)?;
        lib.copy_h2d(descriptors[0].buffer, failed.descriptor_bytes())?;
        let stream = LoadStream { library: &lib, raw: lib.cuda_stream_create()? };
        unsafe {
            window.enqueue_write(&failed, window.source(), descriptors[0].buffer, stream.raw)?;
            lib.cuda_stream_synchronize(stream.raw)?;
            window.revoke_write(failed)?;
        }
        assert!(window.request_id(first).is_err());
        assert_eq!(window.committed_end(second)?, Some(4));
        drop(window.attention_read(&[(second, 4)])?);
        Ok(())
    }
    #[test]
    #[ignore = "requires DS41RT_NATIVE_LIB and CUDA"]
    fn native_reserved_reader_allows_disjoint_commit_and_blocks_recycle() -> Result<()> {
        let lib = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        let mut window = DsparkWindow::new(&lib, 2, 2, usize::MAX)?;
        let first = window.begin_request(0, 11)?;
        let second = window.begin_request(1, 22)?;
        lib.copy_h2d(window.source(), &vec![0; 2*1024])?;
        let chunk = |lease, position| WindowChunk { lease, position, source_row: 0, tokens: 2 };
        unsafe { window.write(&[chunk(first, 0), chunk(second, 0)])?; }
        let read = window.attention_read(&[(first, 2)])?;
        assert!(window.validate_write(&[chunk(first, 2)], 2).is_err());
        assert!(window.release(first).is_err());
        unsafe { window.write(&[chunk(second, 2)])?; }
        assert_eq!(window.committed_end(first)?, Some(2));
        assert_eq!(window.committed_end(second)?, Some(4));
        drop(read);
        unsafe { window.write(&[chunk(first, 2)])?; }
        window.release(first)?;
        assert!(window.request_id(first).is_err());
        let replacement = window.begin_request(0, 33)?;
        assert_eq!(window.request_id(replacement)?, 33);
        Ok(())
    }

}
