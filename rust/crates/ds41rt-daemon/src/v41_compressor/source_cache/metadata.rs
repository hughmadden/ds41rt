//! Canonical host ownership/reservation state for native compressed-source pages.
//! SourceCache uses this exact state; the executor command seam can test it without CUDA.
use super::*;

pub(crate) struct SourcePages {
    pub(super) pages: [Vec<u32>; 16],
    pub(super) rows: [usize; 16],
    pub(super) pool: Rc<RefCell<PagePool>>,
    pub(super) writing: Rc<std::cell::Cell<u16>>,
    slots: usize,
    stride: usize,
}
impl SourcePages {
    pub fn new(pages: usize, slots: usize) -> Result<Self> {
        SourceCache::device_bytes(pages, slots)?;
        Ok(Self { pages: std::array::from_fn(|_| vec![]), rows: [0; 16],
            pool: Rc::new(RefCell::new(PagePool::new(pages))), writing: Default::default(),
            slots, stride: pages.min(4096) })
    }
    pub fn free_pages(&self) -> usize { self.pool.borrow().free.len() }
    pub fn committed_rows(&self, slot: usize) -> Result<usize> {
        ensure!(slot < self.slots, "invalid source slot");
        Ok(self.rows[slot])
    }
    pub fn validate_restore(&self, slot: usize, prefix: &SourcePrefix) -> Result<()> {
        self.ensure_idle(slot)?;
        ensure!(self.pages[slot].is_empty() && Rc::ptr_eq(&self.pool, &prefix.pool)
            && prefix.pages.len() <= self.stride, "foreign source prefix or occupied destination");
        Ok(())
    }
    /// Device metadata must be installed before publishing the host attachment.
    pub fn restore_prefix(&mut self, slot: usize, prefix: &SourcePrefix) -> Result<()> {
        self.validate_restore(slot, prefix)?;
        self.pool.borrow_mut().retain(&prefix.pages);
        self.pages[slot] = prefix.pages.clone();
        self.rows[slot] = prefix.rows;
        Ok(())
    }
    pub fn ensure_idle(&self, slot: usize) -> Result<()> {
        ensure!(slot < self.slots && self.writing.get() & (1 << slot) == 0,
            "compressed cache slot has a pending append");
        Ok(())
    }
    pub fn release(&mut self, slot: usize) -> Result<()> {
        self.ensure_idle(slot)?;
        // Caller first revokes the host lease, and has drained all consumers.
        self.pool.borrow_mut().release(&self.pages[slot]);
        self.pages[slot].clear();
        self.rows[slot] = 0;
        // The lease is revoked and consumers drained. reset/restore installs
        // replacement device metadata before a new owner becomes usable.
        Ok(())
    }

    /// Identity generation of `page`: changes whenever the page is freed and reused.
    pub fn page_generation(&self, page: u32) -> u32 {
        self.pool.borrow().generation(page)
    }
    /// A prefix over `count` freshly allocated pages holding `rows` rows, for the host cache to
    /// fill; the prefix owns the pages. `SourcePoolExhausted` when fewer pages are free.
    pub fn allocate_prefix(&self, count: usize, rows: usize) -> Result<SourcePrefix> {
        ensure!(rows <= count * PAGE_ROWS, "allocated prefix rows exceed its pages");
        let mut pool = self.pool.borrow_mut();
        let available = pool.free.len();
        let pages = pool.allocate(count).ok_or(SourcePoolExhausted {
            work_index: 0,
            needed: count,
            available,
        })?;
        Ok(SourcePrefix {
            pool: Rc::clone(&self.pool),
            pages,
            rows,
        })
    }
    /// Retain initialized source rows without copying GPU data. Callers drain
    /// consumers and supply the authoritative committed row count.
    pub fn retain_prefix(&self, slot: usize, rows: usize) -> Result<SourcePrefix> {
        self.ensure_idle(slot)?;
        ensure!(
            slot < self.slots && rows <= self.rows[slot],
            "source prefix exceeds initialized page table"
        );
        let pages = self.pages[slot][..rows.div_ceil(PAGE_ROWS)].to_vec();
        self.pool.borrow_mut().retain(&pages);
        Ok(SourcePrefix {
            pool: Rc::clone(&self.pool),
            pages,
            rows,
        })
    }

    pub fn validate_plan(&self, plan: &IndexPlan) -> Result<()> {
        let reservation = plan.reservation.as_ref().ok_or_else(|| anyhow::anyhow!("source plan not reserved"))?;
        ensure!(Rc::ptr_eq(&reservation.pool, &self.pool)
            && self.writing.get() & reservation.mask == reservation.mask, "foreign or lost source reservation");
        Ok(())
    }
    /// Atomically claim the append slots and free pages after validating every
    /// participant. Disjoint plans may coexist and apply in either order. After
    /// queueing GPU writes, drain before applying or dropping the plan.
    pub fn reserve(&self, appends: &[(usize, usize, usize)]) -> Result<IndexPlan> {
        let mut plan = IndexPlan {
            additions: vec![],
            used: 0,
            lengths: vec![],
            replacements: vec![],
            reservation: None,
        };
        let mut pool = self.pool.borrow_mut();
        let mut seen = [false; 16];
        for &(slot, old, new) in appends {
            ensure!(
                slot < self.slots && !seen[slot],
                "duplicate or invalid index slot"
            );
            self.ensure_idle(slot)?;
            seen[slot] = true;
            ensure!(
                old == self.rows[slot]
                    && old <= new
                    && new <= 1048576
                    && self.pages[slot].len() == old.div_ceil(PAGE_ROWS),
                "index history binding differs"
            );
        }
        for (position, &(slot, old, new)) in appends.iter().enumerate() {
            plan.lengths.push((slot, new as u64));
            if new > old && old % PAGE_ROWS != 0 {
                let logical = old / PAGE_ROWS;
                let source = self.pages[slot][logical];
                if pool.shared(source) {
                    // If every owner appends in this transaction, one can keep
                    // the original. All tail copies precede every accepted write,
                    // including writes by that owner. A snapshot or non-appending
                    // owner prevents this optimization. Exclusive appends avoid
                    // this bounded (at most sixteen owners) scan entirely.
                    let mut writers = 0;
                    let mut last = position;
                    for (i, &(other, begin, end)) in appends.iter().enumerate() {
                        if end > begin
                            && begin % PAGE_ROWS != 0
                            && self.pages[other][begin / PAGE_ROWS] == source
                        {
                            writers += 1;
                            last = i;
                        }
                    }
                    if writers != pool.references(source) || position != last {
                        ensure!(
                            plan.used < pool.free.len(),
                            SourcePoolExhausted { work_index: position, needed: 1, available: pool.free.len() - plan.used }
                        );
                        let destination = pool.free[pool.free.len() - plan.used - 1];
                        plan.used += 1;
                        plan.replacements.push((slot, logical, source, destination));
                    }
                }
            }
            let extra = new.div_ceil(PAGE_ROWS) - self.pages[slot].len();
            ensure!(
                extra <= pool.free.len() - plan.used,
                SourcePoolExhausted { work_index: position, needed: extra, available: pool.free.len() - plan.used }
            );
            let end = pool.free.len() - plan.used;
            plan.additions.push((
                slot,
                pool.free[end - extra..end].iter().rev().copied().collect(),
            ));
            plan.used += extra;
        }
        let remaining = pool.free.len() - plan.used;
        let pages = pool.free.split_off(remaining);
        let mask = seen.iter().enumerate().fold(0u16, |mask, (slot, &used)|
            mask | if used { 1 << slot } else { 0 });
        self.writing.set(self.writing.get() | mask);
        plan.reservation = Some(PageReservation { pool: self.pool.clone(), pages,
            flags: self.writing.clone(), mask });
        Ok(plan)
    }
    pub fn destination(&self, plan: &IndexPlan, slot: usize, row: usize) -> Result<u64> {
        let logical_page = row / PAGE_ROWS;
        let old = &self.pages[slot];
        let page = if let Some(&(_, _, _, destination)) = plan
            .replacements
            .iter()
            .find(|&&(s, logical, _, _)| s == slot && logical == logical_page)
        {
            destination
        } else if logical_page < old.len() {
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
    pub fn apply(&mut self, mut plan: IndexPlan) {
        let mut reservation = plan.reservation.take().expect("source plan is not reserved");
        assert!(Rc::ptr_eq(&self.pool, &reservation.pool), "foreign source plan");
        let mut pool = self.pool.borrow_mut();
        for (slot, logical, old, new) in plan.replacements {
            pool.retain(&[new]);
            pool.release(&[old]);
            self.pages[slot][logical] = new;
        }
        for (slot, pages) in plan.additions {
            pool.retain(&pages);
            self.pages[slot].extend(pages);
        }
        for (slot, rows) in plan.lengths {
            self.rows[slot] = rows as usize;
        }
        reservation.pages.clear(); // Page references now belong to request tables.
        drop(pool); // Reservation drop must not reborrow an active pool borrow.
        drop(reservation);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jointly_appending_only_owners_copy_one_tail_and_preserve_native_page_generation() -> Result<()> {
        let mut pages = SourcePages::new(3, 2)?;
        let initial = pages.reserve(&[(0, 0, 127)])?;
        let original = pages.destination(&initial, 0, 0)? / 256;
        pages.apply(initial);
        let prefix = pages.retain_prefix(0, 127)?;
        pages.restore_prefix(1, &prefix)?;
        drop(prefix);
        let before = pages.page_generation(original as u32);
        let joint = pages.reserve(&[(0, 127, 128), (1, 127, 128)])?;
        assert_eq!(joint.tail_copies().len(), 1); // retained rule: last writer may retain original
        assert_ne!(pages.destination(&joint, 0, 127)?, pages.destination(&joint, 1, 127)?);
        pages.apply(joint);
        pages.release(0)?;
        assert_eq!(pages.page_generation(original as u32), before);
        pages.release(1)?;
        assert_eq!(pages.page_generation(original as u32), before + 1);
        assert_eq!(pages.free_pages(), 3);
        Ok(())
    }

    #[test]
    fn native_source_reservations_reject_foreign_owner_and_rollback_only_their_pages() -> Result<()> {
        let pages = SourcePages::new(5, 3)?;
        let foreign = SourcePages::new(5, 3)?;
        let a = pages.reserve(&[(0, 0, 257)])?;
        let b = pages.reserve(&[(1, 0, 256)])?;
        assert_eq!(pages.free_pages(), 2);
        assert!(foreign.validate_plan(&a).is_err());
        assert!(pages.reserve(&[(0, 0, 1)]).is_err());
        assert!(pages.reserve(&[(2, 0, 769)]).is_err());
        assert_eq!(pages.free_pages(), 2);
        drop(a);
        assert_eq!(pages.free_pages(), 4);
        assert!(pages.ensure_idle(1).is_err());
        drop(b);
        assert_eq!(pages.free_pages(), 5);
        pages.ensure_idle(0)?;
        pages.ensure_idle(1)?;
        Ok(())
    }
}
