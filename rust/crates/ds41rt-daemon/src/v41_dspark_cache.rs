//! Committed dSpark KV rings with generation-checked request slots.
use crate::v41_memory::{DeviceAllocation, HostAllocation, LoadStream};
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{
    Ds41rtDeviceBuffer, NativeLibrary, V41AttentionWindow, V41DsparkCache, V41KvWrite,
};
use std::{
    ffi::c_void,
    sync::atomic::{AtomicU64, Ordering},
};
static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);
#[derive(Clone, Copy)]
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
}
pub(crate) struct DsparkWindow<'a> {
    stream: LoadStream<'a>,
    kernel: V41DsparkCache<'a>,
    source: DeviceAllocation<'a>,
    ring: DeviceAllocation<'a>,
    descriptors: DeviceAllocation<'a>,
    staging: HostAllocation<'a>,
    graph: Option<*mut c_void>,
    slots: [Slot; 16],
    slot_count: usize,
    source_rows: u32,
    owner: u64,
}
impl<'a> DsparkWindow<'a> {
    pub fn device_bytes(slots: usize, source_rows: u32) -> Result<usize> {
        ensure!(
            (1..=16).contains(&slots) && (1..=4096).contains(&source_rows),
            "invalid dSpark window capacity"
        );
        Ok(slots * 131072 + source_rows as usize * 1024 + 384)
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
            stream: LoadStream {
                library,
                raw: library.cuda_stream_create()?,
            },
            kernel: library.v41_dspark_cache()?,
            source: DeviceAllocation::new(library, source_rows as usize * 1024)?,
            ring: DeviceAllocation::new(library, slots * 131072)?,
            descriptors: DeviceAllocation::new(library, 384)?,
            staging: HostAllocation::new(library, 384)?,
            graph: None,
            slots: [Slot::default(); 16],
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
    pub fn release(&mut self, lease: WindowLease) -> Result<()> {
        let slot = self.validate(lease)?;
        self.slots[slot].request = None;
        self.slots[slot].end = None;
        Ok(())
    }
    /// Stable BF16 [source_rows,512] destination for normalized, RoPE-applied KV.
    pub fn source(&self) -> Ds41rtDeviceBuffer {
        self.source.buffer
    }
    /// # Safety
    /// Source rows must be initialized finite KV with producer writes completed;
    /// serialize source use and every consumer of a borrowed ring view. Chunks
    /// contain only committed main-model positions, never unaccepted draft KV.
    pub unsafe fn write(&mut self, chunks: &[WindowChunk]) -> Result<()> {
        ensure!(
            !chunks.is_empty() && chunks.len() <= self.slot_count,
            "invalid cache chunk count"
        );
        let mut seen = [false; 16];
        let mut descriptors = [V41KvWrite::default(); 16];
        let mut ends = [None; 16];
        for (i, chunk) in chunks.iter().enumerate() {
            let slot = self.validate(chunk.lease)?;
            ensure!(!seen[slot], "duplicate cache slot in batch");
            seen[slot] = true;
            ensure!(
                chunk.tokens > 0
                    && chunk.source_row < self.source_rows
                    && chunk.tokens <= self.source_rows - chunk.source_row,
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
    /// Validate the entire read batch before metadata upload; the caller holds
    /// this borrow through GPU completion, preventing safe concurrent mutation.
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
        })
    }
    /// Physical ring order: attention reads 0..valid_rows, then its five private
    /// draft positions. No chronological reordering of this buffer is necessary.
    pub fn view(&self, lease: WindowLease) -> Result<WindowView> {
        let slot = self.validate(lease)?;
        let end = self.slots[slot]
            .end
            .context("dSpark window has not been seeded")?;
        let mut buffer = self.ring.buffer;
        buffer.ptr = unsafe { buffer.ptr.cast::<u8>().add(slot * 131072).cast() };
        buffer.bytes = 131072;
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
