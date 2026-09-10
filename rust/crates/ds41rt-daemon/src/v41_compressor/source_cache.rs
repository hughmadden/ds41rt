//! Paired index/FP8 KV pages owned by compressor request leases.
use crate::v41_memory::{DeviceAllocation, HostAllocation};
use anyhow::{ensure, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary};
use std::ffi::c_void;

pub(super) const PAGE_ROWS: usize = 256;
pub(super) struct SourceCache<'a> {
    pub packed: DeviceAllocation<'a>,
    pub scales: DeviceAllocation<'a>,
    pub kv_values: DeviceAllocation<'a>,
    pub kv_scales: DeviceAllocation<'a>,
    pub capacity: usize,
    page_table: DeviceAllocation<'a>,
    lengths: DeviceAllocation<'a>,
    staging: HostAllocation<'a>,
    stride: usize,
    pages: [Vec<u32>; 16],
    free: Vec<u32>,
}
pub(super) struct IndexPlan {
    additions: Vec<(usize, Vec<u32>)>,
    used: usize,
    lengths: Vec<(usize, u64)>,
}
/// Only rows below `rows` are initialized. Logical row r uses physical page
/// pages[r / 256], offset r % 256. Drain device consumers before mutating owner.
/// The borrow prevents host-side release/commit while this view is in use.
pub(crate) struct IndexCacheView<'a> {
    pub packed: Ds41rtDeviceBuffer,
    pub scales: Ds41rtDeviceBuffer,
    pub pages: &'a [u32],
    pub rows: usize,
    /// U32 physical page IDs, capacity entries; only ceil(rows/256) are valid.
    pub device_pages: Ds41rtDeviceBuffer,
    /// U64 committed row count, published after accepted value/scale writes.
    pub device_rows: Ds41rtDeviceBuffer,
}
/// FP8 K32 serving KV shares physical pages and publication with index keys.
pub(crate) struct KvCacheView<'a> {
    pub values: Ds41rtDeviceBuffer,
    pub scales: Ds41rtDeviceBuffer,
    pub pages: &'a [u32],
    pub rows: usize,
    pub device_pages: Ds41rtDeviceBuffer,
    pub device_rows: Ds41rtDeviceBuffer,
}
impl<'a> SourceCache<'a> {
    pub fn device_bytes(pages: usize, slots: usize) -> Result<usize> {
        ensure!(
            (1..=65536).contains(&pages),
            "invalid index pool page count"
        );
        ensure!((1..=16).contains(&slots), "invalid index slot count");
        Ok(pages * PAGE_ROWS * (68 + 528) + slots * (pages.min(4096) * 4 + 8))
    }
    pub fn new(library: &'a NativeLibrary, pages: usize, slots: usize) -> Result<Self> {
        Self::device_bytes(pages, slots)?;
        let lengths = DeviceAllocation::new(library, slots * 8)?;
        library.copy_h2d(lengths.buffer, &vec![0; slots * 8])?;
        Ok(Self {
            page_table: DeviceAllocation::new(library, slots * pages.min(4096) * 4)?,
            lengths,
            staging: HostAllocation::new(library, pages * 4 + slots * 8)?,
            stride: pages.min(4096),
            packed: DeviceAllocation::new(library, pages * PAGE_ROWS * 64)?,
            scales: DeviceAllocation::new(library, pages * PAGE_ROWS * 4)?,
            kv_values: DeviceAllocation::new(library, pages * PAGE_ROWS * 512)?,
            kv_scales: DeviceAllocation::new(library, pages * PAGE_ROWS * 16)?,
            capacity: pages * PAGE_ROWS,
            pages: std::array::from_fn(|_| vec![]),
            free: (0..pages as u32).rev().collect(),
        })
    }
    pub fn view(&self, slot: usize, rows: usize) -> IndexCacheView<'_> {
        IndexCacheView {
            packed: self.packed.buffer,
            scales: self.scales.buffer,
            pages: &self.pages[slot],
            rows,
            device_pages: slice(
                self.page_table.buffer,
                slot * self.stride * 4,
                self.stride * 4,
            ),
            device_rows: slice(self.lengths.buffer, slot * 8, 8),
        }
    }
    pub fn kv_view(&self, slot: usize, rows: usize) -> KvCacheView<'_> {
        let index = self.view(slot, rows);
        KvCacheView {
            values: self.kv_values.buffer,
            scales: self.kv_scales.buffer,
            pages: index.pages,
            rows: index.rows,
            device_pages: index.device_pages,
            device_rows: index.device_rows,
        }
    }
    pub fn release(&mut self, slot: usize) -> Result<()> {
        // Caller first revokes the host lease, and has drained all consumers.
        self.free.extend(self.pages[slot].drain(..));
        self.lengths
            .library
            .copy_h2d(slice(self.lengths.buffer, slot * 8, 8), &[0; 8])
    }
    pub fn reset(&self, slot: usize) -> Result<()> {
        self.lengths
            .library
            .copy_h2d(slice(self.lengths.buffer, slot * 8, 8), &[0; 8])
    }
    /// # Safety
    /// Value/scales writes precede this call on stream. Drain the stream before
    /// reusing staging, releasing a slot or publishing the plan on the host.
    pub unsafe fn upload(&mut self, plan: &IndexPlan, stream: *mut c_void) -> Result<()> {
        let library = self.lengths.library;
        let mut offset = 0;
        for (slot, pages) in &plan.additions {
            let bytes = pages.len() * 4;
            if bytes == 0 {
                continue;
            }
            let staging = &mut self.staging.bytes_mut()[offset..offset + bytes];
            for (out, page) in staging.chunks_exact_mut(4).zip(pages) {
                out.copy_from_slice(&page.to_ne_bytes());
            }
            let dst = (slot * self.stride + self.pages[*slot].len()) * 4;
            unsafe {
                library.copy_h2d_async(
                    slice(self.page_table.buffer, dst, bytes),
                    staging,
                    stream,
                )?;
            }
            offset += bytes;
        }
        for &(slot, rows) in &plan.lengths {
            let staging = &mut self.staging.bytes_mut()[offset..offset + 8];
            staging.copy_from_slice(&rows.to_ne_bytes());
            unsafe {
                library.copy_h2d_async(slice(self.lengths.buffer, slot * 8, 8), staging, stream)?;
            }
            offset += 8;
        }
        Ok(())
    }
    /// Reserve all requests together without mutating pool metadata. Caller
    /// serializes this plan through apply/release and validates unique slots.
    pub fn reserve(&self, appends: &[(usize, usize, usize)]) -> Result<IndexPlan> {
        let mut plan = IndexPlan {
            additions: vec![],
            used: 0,
            lengths: vec![],
        };
        let mut seen = [false; 16];
        for &(slot, old, new) in appends {
            ensure!(slot < 16 && !seen[slot], "duplicate or invalid index slot");
            seen[slot] = true;
            ensure!(
                old <= new && new <= 1048576 && self.pages[slot].len() == old.div_ceil(PAGE_ROWS),
                "index history binding differs"
            );
            plan.lengths.push((slot, new as u64));
            let extra = new.div_ceil(PAGE_ROWS) - self.pages[slot].len();
            ensure!(
                extra <= self.free.len() - plan.used,
                "index cache pool exhausted"
            );
            let end = self.free.len() - plan.used;
            plan.additions.push((
                slot,
                self.free[end - extra..end].iter().rev().copied().collect(),
            ));
            plan.used += extra;
        }
        Ok(plan)
    }
    pub fn destination(&self, plan: &IndexPlan, slot: usize, row: usize) -> Result<u64> {
        let logical_page = row / PAGE_ROWS;
        let old = &self.pages[slot];
        let page = if logical_page < old.len() {
            old[logical_page]
        } else {
            let new = plan
                .additions
                .iter()
                .find(|(s, _)| *s == slot)
                .ok_or_else(|| anyhow::anyhow!("index append missing"))?;
            *new.1
                .get(logical_page - old.len())
                .ok_or_else(|| anyhow::anyhow!("index append outside reservation"))?
        };
        Ok(u64::from(page) * PAGE_ROWS as u64 + (row % PAGE_ROWS) as u64)
    }
    pub fn apply(&mut self, plan: IndexPlan) {
        self.free.truncate(self.free.len() - plan.used);
        for (slot, pages) in plan.additions {
            self.pages[slot].extend(pages);
        }
    }
}

fn slice(buffer: Ds41rtDeviceBuffer, offset: usize, bytes: usize) -> Ds41rtDeviceBuffer {
    debug_assert!(offset + bytes <= buffer.bytes);
    Ds41rtDeviceBuffer {
        ptr: unsafe { buffer.ptr.cast::<u8>().add(offset).cast() },
        bytes,
        ..buffer
    }
}
