//! Physical index pages owned by compressor request leases.
use crate::v41_memory::DeviceAllocation;
use anyhow::{ensure, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary};

pub(super) const PAGE_ROWS: usize = 256;
pub(super) struct IndexCache<'a> {
    pub packed: DeviceAllocation<'a>,
    pub scales: DeviceAllocation<'a>,
    pub capacity: usize,
    pages: [Vec<u32>; 16],
    free: Vec<u32>,
}
pub(super) struct IndexPlan {
    additions: Vec<(usize, Vec<u32>)>,
    used: usize,
}
/// Only rows below `rows` are initialized. Logical row r uses physical page
/// pages[r / 256], offset r % 256. Drain device consumers before mutating owner.
/// The borrow prevents host-side release/commit while this view is in use.
pub(crate) struct IndexCacheView<'a> {
    pub packed: Ds41rtDeviceBuffer,
    pub scales: Ds41rtDeviceBuffer,
    pub pages: &'a [u32],
    pub rows: usize,
}
impl<'a> IndexCache<'a> {
    pub fn device_bytes(pages: usize) -> Result<usize> {
        ensure!(
            (1..=65536).contains(&pages),
            "invalid index pool page count"
        );
        Ok(pages * PAGE_ROWS * 68)
    }
    pub fn new(library: &'a NativeLibrary, pages: usize) -> Result<Self> {
        Self::device_bytes(pages)?;
        Ok(Self {
            packed: DeviceAllocation::new(library, pages * PAGE_ROWS * 64)?,
            scales: DeviceAllocation::new(library, pages * PAGE_ROWS * 4)?,
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
        }
    }
    pub fn release(&mut self, slot: usize) {
        self.free.extend(self.pages[slot].drain(..));
    }
    /// Reserve all requests together without mutating pool metadata. Caller
    /// serializes this plan through apply/release and validates unique slots.
    pub fn reserve(&self, appends: &[(usize, usize, usize)]) -> Result<IndexPlan> {
        let mut plan = IndexPlan {
            additions: vec![],
            used: 0,
        };
        let mut seen = [false; 16];
        for &(slot, old, new) in appends {
            ensure!(slot < 16 && !seen[slot], "duplicate or invalid index slot");
            seen[slot] = true;
            ensure!(
                old <= new && new <= 1048576 && self.pages[slot].len() == old.div_ceil(PAGE_ROWS),
                "index history binding differs"
            );
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
