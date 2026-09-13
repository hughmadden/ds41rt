//! Free pages are removed at reservation, not deferred until publication.
use super::*;

pub(super) struct PageReservation {
    pub pool: Rc<RefCell<PagePool>>,
    pub pages: Vec<u32>,
    pub flags: Rc<std::cell::Cell<u16>>,
    pub mask: u16,
}
impl Drop for PageReservation {
    fn drop(&mut self) {
        if !self.pages.is_empty() {
            self.pool.borrow_mut().free.append(&mut self.pages);
        }
        self.flags.set(self.flags.get() & !self.mask);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rollback_returns_only_its_pages_and_preserves_peer_reservation() {
        let pool = Rc::new(RefCell::new(PagePool::new(4)));
        let flags = Rc::new(std::cell::Cell::new(3));
        let original = pool.borrow().free.clone();
        let first = PageReservation { pool: pool.clone(), pages: pool.borrow_mut().free.split_off(2), flags: flags.clone(), mask: 1 };
        let second = PageReservation { pool: pool.clone(), pages: pool.borrow_mut().free.split_off(1), flags: flags.clone(), mask: 2 };
        assert_eq!(pool.borrow().free.len(), 1);
        drop(first);
        assert_eq!(flags.get(), 2);
        assert_eq!(pool.borrow().free.len(), 3);
        drop(second);
        assert_eq!(flags.get(), 0);
        let mut restored = pool.borrow().free.clone(); restored.sort_unstable();
        let mut expected = original; expected.sort_unstable();
        assert_eq!(restored, expected);
    }
    #[test]
    #[ignore = "requires DS41RT_NATIVE_LIB and CUDA"]
    fn native_disjoint_source_plans_publish_out_of_order_and_rollback() -> Result<()> {
        let lib = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        let mut cache = SourceCache::new(&lib, 8, 3)?;
        let first = cache.reserve(&[(0, 0, 257)])?;
        let second = cache.reserve(&[(1, 0, 512)])?;
        assert_eq!(cache.pool.borrow().free.len(), 4);
        assert!(cache.reserve(&[(0, 0, 1)]).is_err());
        assert!(cache.release(0).is_err());
        assert!(cache.retain_prefix(0, 0).is_err());
        let free = cache.pool.borrow().free.clone();
        assert!(cache.reserve(&[(2, 0, 1280)]).is_err());
        assert_eq!(cache.pool.borrow().free, free);
        let third = cache.reserve(&[(2, 0, 256)])?;
        let claimed = first.additions.iter().chain(&second.additions).chain(&third.additions)
            .flat_map(|(_, pages)| pages).copied().collect::<std::collections::BTreeSet<_>>();
        assert_eq!(claimed.len(), 5);
        drop(third);
        // Fill each claimed physical page before publishing its metadata.
        for (plan, value) in [(&first, 0x11), (&second, 0x22)] {
            for &page in plan.additions.iter().flat_map(|(_, pages)| pages) {
                for (buffer, width) in [(cache.packed.buffer, 64), (cache.scales.buffer, 4),
                    (cache.kv_values.buffer, KV_VALUES), (cache.kv_scales.buffer, KV_SCALES)] {
                    lib.copy_h2d(slice(buffer, page as usize*PAGE_ROWS*width, PAGE_ROWS*width),
                        &vec![value; PAGE_ROWS*width])?;
                }
            }
        }
        // The upload path is still synchronous/shared; this test overlaps page
        // ownership, not uploads. Producer-owned upload buffers are next.
        let stream = crate::v41_memory::LoadStream { library: &lib, raw: lib.cuda_stream_create()? };
        unsafe { cache.upload(&second, stream.raw)?; lib.cuda_stream_synchronize(stream.raw)?; }
        cache.apply(second);
        cache.ensure_idle(1)?;
        assert!(cache.ensure_idle(0).is_err());
        assert_eq!(cache.rows[0], 0);
        assert_eq!(cache.rows[1], 512);
        unsafe { cache.upload(&first, stream.raw)?; lib.cuda_stream_synchronize(stream.raw)?; }
        cache.apply(first);
        assert!(super::super::tests::read(&cache, 0, 257)?.iter().all(|&b| b == 0x11));
        assert!(super::super::tests::read(&cache, 1, 512)?.iter().all(|&b| b == 0x22));
        cache.release(0)?; cache.release(1)?;
        assert_eq!(cache.pool.borrow().free.len(), 8);
        let mut length = [0; 8];
        lib.copy_d2h(&mut length, slice(cache.lengths.buffer, 0, 8))?;
        assert_eq!(u64::from_ne_bytes(length), 257); // Revoked metadata need not be cleared.
        cache.reset(0)?;
        lib.copy_d2h(&mut length, slice(cache.lengths.buffer, 0, 8))?;
        assert_eq!(u64::from_ne_bytes(length), 0);
        let free = cache.pool.borrow().free.clone();
        let cancelled = cache.reserve(&[(0, 0, 257)])?;
        drop(cancelled);
        assert_eq!(cache.pool.borrow().free, free);
        cache.ensure_idle(0)?;
        Ok(())
    }
}
