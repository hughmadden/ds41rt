//! Host snapshots (packet HC-2): what the cache holds and how it finds it. A host snapshot is
//! the four parts of a retained snapshot in pinned slabs plus its key-space tokens, frontier and
//! kind. Pages are shared exactly as the device shares them: while a device page is allocated,
//! its identity maps to at most one host page, so a page is copied once however many snapshots
//! reference it. Lookups use the engine's own `Retention` radix (`ds41rt-core::prefix`), so a
//! host hit is exactly a snapshot the device tier would have chosen. Eviction follows the same
//! bank order (prompts before turns, oldest access first) and never touches a pinned snapshot.
use crate::pool::{Class, PoolExhausted, Slab, SlabPool};
use crate::SnapshotKind;
use crate::COMPRESSORS;
use ds41rt_core::prefix::Retention;
use serde::Serialize;
use std::collections::HashMap;

/// Identity of a device page while it is allocated: the engine's page index in its compressor
/// pool and the generation that increments on every free, so a reused index is a new identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct DevicePageId {
    pub compressor: u8,
    pub page: u32,
    pub generation: u32,
}

/// Cache-local snapshot id, unique for the life of the cache.
pub type Key = u64;

/// Index into the shared page table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct PageRef(pub u32);

/// What the engine tells the cache about a snapshot at store time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SnapshotMeta {
    pub kind: SnapshotKind,
    /// Key-space tokens (image spans already folded in by the engine).
    pub tokens: Vec<u32>,
    /// Tokens the snapshot covers; `tokens.len()` for a whole-prefix snapshot.
    pub end: u32,
    pub has_draft: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostSnapshot {
    pub key: Key,
    pub meta: SnapshotMeta,
    /// Per compressor, in logical order.
    pub pages: [Vec<PageRef>; COMPRESSORS],
    pub tail: Slab,
    pub draft: Option<Slab>,
    pub scores: Slab,
    pub last_access_ns: u64,
    /// Restores in flight; a pinned snapshot is never evicted.
    pub pins: u32,
}

/// The result of planning a store: slabs allocated for every part and for each page that is not
/// already shared, with the device identity each unshared page must be copied from.
#[derive(Debug)]
pub struct StorePlan {
    pub key: Key,
    pub meta: SnapshotMeta,
    /// `(compressor, logical index, device identity, destination slab)` for pages to copy.
    pub copies: Vec<(u8, u32, DevicePageId, Slab)>,
    /// Per compressor, in logical order, the final page references (shared or newly allocated).
    pub pages: [Vec<PageRef>; COMPRESSORS],
    pub tail: Slab,
    pub draft: Option<Slab>,
    pub scores: Slab,
}

/// A lookup hit: the engine's `(common, frontier)` semantics for the matched snapshot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hit {
    pub key: Key,
    pub kind: SnapshotKind,
    pub common: usize,
    pub frontier: usize,
}

/// The slab operations `Snapshots` needs from a pinned pool. `SlabPool` implements it; the
/// suites substitute a fake so this packet is testable before HC-1 lands.
pub trait Pool {
    fn take(&mut self, class: Class) -> Result<Slab, PoolExhausted>;
    fn give_back(&mut self, slab: Slab);
    fn bytes_used(&self) -> u64;
}

impl Pool for SlabPool {
    fn take(&mut self, class: Class) -> Result<Slab, PoolExhausted> {
        SlabPool::take(self, class)
    }
    fn give_back(&mut self, slab: Slab) {
        SlabPool::give_back(self, slab)
    }
    fn bytes_used(&self) -> u64 {
        SlabPool::bytes_used(self)
    }
}

/// One shared host page: its slab, how many snapshots reference it, and the device identity it
/// was copied from while that device page is alive.
struct PageEntry {
    slab: Slab,
    refs: u32,
    device: Option<DevicePageId>,
}

/// The store of host snapshots. Invariants: a page slab is held exactly while its reference
/// count is positive; the device map holds an entry only for a page that is both alive on the
/// device and present in a host slab; `bytes_used()` equals the pool's; eviction order equals
/// `Retention::evict_one` order over unpinned snapshots.
pub struct Snapshots<P: Pool = SlabPool> {
    pool: P,
    retention: Retention<Key>,
    snapshots: HashMap<Key, HostSnapshot>,
    /// Shared page table; a slot is `None` while its index is on `free_pages`.
    pages: Vec<Option<PageEntry>>,
    free_pages: Vec<u32>,
    /// Device identity to the one host page that holds its bytes, while it is shareable.
    device_map: HashMap<DevicePageId, PageRef>,
    next_key: Key,
}

impl<P: Pool> Snapshots<P> {
    pub fn new(pool: P) -> Self {
        Self {
            pool,
            retention: Retention::new(usize::MAX),
            snapshots: HashMap::new(),
            pages: Vec::new(),
            free_pages: Vec::new(),
            device_map: HashMap::new(),
            next_key: 0,
        }
    }

    /// Allocate slabs for `meta` and every page of `device_pages` not already shared. On
    /// `PoolExhausted` nothing is held. Duplicate tokens for the same kind replace the older
    /// snapshot on commit (the newer frontier wins, as in the engine's radix).
    pub fn plan_store(
        &mut self,
        meta: SnapshotMeta,
        device_pages: &[Vec<DevicePageId>; COMPRESSORS],
    ) -> Result<StorePlan, PoolExhausted> {
        let tail = self.pool.take(Class::Tail)?;
        let scores = match self.pool.take(Class::Scores) {
            Ok(slab) => slab,
            Err(exhausted) => {
                self.pool.give_back(tail);
                return Err(exhausted);
            }
        };
        let draft = if meta.has_draft {
            match self.pool.take(Class::Draft) {
                Ok(slab) => Some(slab),
                Err(exhausted) => {
                    self.release_slabs(tail, None, scores);
                    return Err(exhausted);
                }
            }
        } else {
            None
        };
        let mut pages: [Vec<PageRef>; COMPRESSORS] = std::array::from_fn(|_| Vec::new());
        let mut copies = Vec::new();
        for (compressor, list) in device_pages.iter().enumerate() {
            for (index, &id) in list.iter().enumerate() {
                match self.device_map.get(&id).copied() {
                    Some(page) => {
                        self.page_mut(page).refs += 1;
                        pages[compressor].push(page);
                    }
                    None => match self.pool.take(Class::Page) {
                        Ok(slab) => {
                            let page = self.insert_page(slab, id);
                            pages[compressor].push(page);
                            copies.push((compressor as u8, index as u32, id, slab));
                        }
                        Err(exhausted) => {
                            self.release_pages(&pages);
                            self.release_slabs(tail, draft, scores);
                            return Err(exhausted);
                        }
                    },
                }
            }
        }
        let key = self.next_key;
        self.next_key += 1;
        Ok(StorePlan {
            key,
            meta,
            copies,
            pages,
            tail,
            draft,
            scores,
        })
    }

    /// After every copy of `plan` completed: make the snapshot lookup-visible.
    pub fn commit_store(&mut self, plan: StorePlan, now_ns: u64) -> Key {
        let StorePlan {
            key,
            meta,
            copies: _,
            pages,
            tail,
            draft,
            scores,
        } = plan;
        let kind = meta.kind;
        let replaced = self
            .retention
            .bank_mut(kind)
            .lookup(&meta.tokens)
            .and_then(|(position, &old)| (position == meta.tokens.len()).then_some(old));
        if let Some(old) = replaced {
            if let Some(snapshot) = self.snapshots.remove(&old) {
                self.release_snapshot(snapshot);
            }
        }
        self.retention.bank_mut(kind).insert(&meta.tokens, key);
        self.snapshots.insert(
            key,
            HostSnapshot {
                key,
                meta,
                pages,
                tail,
                draft,
                scores,
                last_access_ns: now_ns,
                pins: 0,
            },
        );
        key
    }

    /// A store that failed: release its slabs and drop its page references.
    pub fn abort_store(&mut self, plan: StorePlan) {
        self.release_pages(&plan.pages);
        self.release_slabs(plan.tail, plan.draft, plan.scores);
    }

    /// The engine's reuse rule over host snapshots; refreshes the hit's access clock.
    pub fn lookup(&mut self, tokens: &[u32], now_ns: u64) -> Option<Hit> {
        let (common, frontier, key) = {
            let (common, frontier, &key) = self.retention.lookup_reusable(tokens)?;
            (common, frontier, key)
        };
        let snapshot = self.snapshots.get_mut(&key)?;
        snapshot.last_access_ns = now_ns;
        Some(Hit {
            key,
            kind: snapshot.meta.kind,
            common,
            frontier,
        })
    }

    pub fn get(&self, key: Key) -> Option<&HostSnapshot> {
        self.snapshots.get(&key)
    }

    pub fn pin(&mut self, key: Key) {
        if let Some(snapshot) = self.snapshots.get_mut(&key) {
            snapshot.pins += 1;
        }
    }

    pub fn unpin(&mut self, key: Key) {
        if let Some(snapshot) = self.snapshots.get_mut(&key) {
            snapshot.pins = snapshot.pins.saturating_sub(1);
        }
    }

    /// A device page was freed: its identity can no longer be shared from the device.
    pub fn device_page_freed(&mut self, id: DevicePageId) {
        if let Some(page) = self.device_map.remove(&id) {
            if let Some(entry) = self.pages[page.0 as usize].as_mut() {
                entry.device = None;
            }
        }
    }

    /// Evict in the engine's order until `bytes_used() <= quota`, skipping pinned snapshots;
    /// returns the evicted keys and the bytes freed. Stops early if only pinned remain.
    pub fn evict_to(&mut self, quota: u64) -> (Vec<Key>, u64) {
        let mut evicted = Vec::new();
        let mut freed = 0;
        while self.pool.bytes_used() > quota {
            let next = {
                let snapshots = &self.snapshots;
                self.retention.evict_one_where(&|key: &Key| {
                    snapshots.get(key).is_some_and(|snapshot| snapshot.pins > 0)
                })
            };
            let Some((_kind, key)) = next else {
                break;
            };
            let before = self.pool.bytes_used();
            if let Some(snapshot) = self.snapshots.remove(&key) {
                self.release_snapshot(snapshot);
                freed += before - self.pool.bytes_used();
                evicted.push(key);
            }
        }
        (evicted, freed)
    }

    pub fn remove(&mut self, key: Key) -> bool {
        let Some(snapshot) = self.snapshots.remove(&key) else {
            return false;
        };
        self.retention
            .bank_mut(snapshot.meta.kind)
            .remove_exact(&snapshot.meta.tokens);
        self.release_snapshot(snapshot);
        true
    }

    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn bytes_used(&self) -> u64 {
        self.pool.bytes_used()
    }

    pub fn pool(&self) -> &P {
        &self.pool
    }

    /// The radix, for the suites that assert a host hit equals a `Retention` hit.
    pub fn retention(&self) -> &Retention<Key> {
        &self.retention
    }

    /// The reference count of a shared page; zero when the index is free. Test support.
    #[doc(hidden)]
    pub fn page_ref_count(&self, page: PageRef) -> u32 {
        self.pages
            .get(page.0 as usize)
            .and_then(|entry| entry.as_ref())
            .map_or(0, |entry| entry.refs)
    }

    /// The device identity a shared page was copied from, if it is still shareable. Test support.
    #[doc(hidden)]
    pub fn page_device(&self, page: PageRef) -> Option<DevicePageId> {
        self.pages
            .get(page.0 as usize)
            .and_then(|entry| entry.as_ref())
            .and_then(|entry| entry.device)
    }

    /// The reference counts of every live shared page. Test support.
    #[doc(hidden)]
    pub fn page_ref_counts(&self) -> Vec<u32> {
        self.pages
            .iter()
            .flatten()
            .map(|entry| entry.refs)
            .collect()
    }

    /// The number of live shared pages. Test support.
    #[doc(hidden)]
    pub fn live_pages(&self) -> usize {
        self.pages.iter().filter(|entry| entry.is_some()).count()
    }

    /// The sum of every live page's reference count. Test support.
    #[doc(hidden)]
    pub fn page_ref_total(&self) -> u64 {
        self.pages
            .iter()
            .flatten()
            .map(|entry| u64::from(entry.refs))
            .sum()
    }

    /// The live page entry for `page`; a missing entry is a broken internal invariant.
    fn page_mut(&mut self, page: PageRef) -> &mut PageEntry {
        self.pages[page.0 as usize]
            .as_mut()
            .expect("device map points at a live page")
    }

    /// Claim a free page index (or grow the table) and map `id` to it with one reference.
    fn insert_page(&mut self, slab: Slab, id: DevicePageId) -> PageRef {
        let index = match self.free_pages.pop() {
            Some(index) => index,
            None => {
                self.pages.push(None);
                (self.pages.len() - 1) as u32
            }
        };
        self.pages[index as usize] = Some(PageEntry {
            slab,
            refs: 1,
            device: Some(id),
        });
        let page = PageRef(index);
        self.device_map.insert(id, page);
        page
    }

    /// Drop one reference to `page`; release its slab and device-map entry at zero.
    fn release_page(&mut self, page: PageRef) {
        let index = page.0 as usize;
        let Some(entry) = self.pages[index].as_mut() else {
            debug_assert!(false, "released a page that is not live");
            return;
        };
        entry.refs -= 1;
        if entry.refs > 0 {
            return;
        }
        let slab = entry.slab;
        if let Some(id) = entry.device.take() {
            self.device_map.remove(&id);
        }
        self.pages[index] = None;
        self.free_pages.push(page.0);
        self.pool.give_back(slab);
    }

    /// Drop one reference to every page in `pages`.
    fn release_pages(&mut self, pages: &[Vec<PageRef>; COMPRESSORS]) {
        for list in pages {
            for &page in list {
                self.release_page(page);
            }
        }
    }

    /// Release a snapshot's parts and every page reference it holds.
    fn release_snapshot(&mut self, snapshot: HostSnapshot) {
        self.release_pages(&snapshot.pages);
        self.release_slabs(snapshot.tail, snapshot.draft, snapshot.scores);
    }

    /// Return the non-page slabs of a plan or snapshot to the pool.
    fn release_slabs(&mut self, tail: Slab, draft: Option<Slab>, scores: Slab) {
        self.pool.give_back(scores);
        if let Some(draft) = draft {
            self.pool.give_back(draft);
        }
        self.pool.give_back(tail);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{test_layout, FakePool};

    fn id(page: u32) -> DevicePageId {
        DevicePageId {
            compressor: 0,
            page,
            generation: 0,
        }
    }

    fn meta(tokens: &[u32], has_draft: bool) -> SnapshotMeta {
        SnapshotMeta {
            kind: SnapshotKind::Turn,
            tokens: tokens.to_vec(),
            end: tokens.len() as u32,
            has_draft,
        }
    }

    fn pages(ids: &[DevicePageId]) -> [Vec<DevicePageId>; COMPRESSORS] {
        let mut pages: [Vec<DevicePageId>; COMPRESSORS] = std::array::from_fn(|_| Vec::new());
        pages[0].extend_from_slice(ids);
        pages
    }

    fn snapshots(quota: u64) -> Snapshots<FakePool> {
        Snapshots::new(FakePool::new(quota, test_layout()))
    }

    #[test]
    fn plan_allocates_one_slab_per_part_and_page() {
        let mut store = snapshots(1 << 30);
        let plan = store
            .plan_store(meta(&[1, 2, 3], true), &pages(&[id(1), id(2)]))
            .expect("plan");
        assert_eq!(plan.copies.len(), 2);
        assert_eq!(plan.pages[0].len(), 2);
        assert!(plan.draft.is_some());
        let layout = test_layout();
        let expected = (layout.tail + layout.scores + layout.draft + 2 * layout.page) as u64;
        assert_eq!(store.bytes_used(), expected);
        store.abort_store(plan);
        assert_eq!(store.bytes_used(), 0);
        assert_eq!(store.live_pages(), 0);
    }

    #[test]
    fn plan_exhaustion_releases_everything_it_allocated() {
        let layout = test_layout();
        // Room for the tail and scores but not one page.
        let mut store = snapshots((layout.tail + layout.scores) as u64);
        let error = store
            .plan_store(meta(&[1], false), &pages(&[id(1)]))
            .expect_err("page class exhausted");
        assert_eq!(error.class, Class::Page);
        assert_eq!(store.bytes_used(), 0);
        assert_eq!(store.live_pages(), 0);
    }

    #[test]
    fn shared_pages_are_referenced_not_recopied() {
        let mut store = snapshots(1 << 30);
        let first = store
            .plan_store(meta(&[1, 2], false), &pages(&[id(1), id(2)]))
            .expect("plan");
        assert_eq!(first.copies.len(), 2);
        let first_key = store.commit_store(first, 0);
        let second = store
            .plan_store(meta(&[1, 2, 3], false), &pages(&[id(1), id(2), id(3)]))
            .expect("plan");
        assert_eq!(second.copies.len(), 1);
        assert_eq!(second.copies[0].2, id(3));
        let second_key = store.commit_store(second, 1);
        assert_eq!(store.live_pages(), 3);
        assert_eq!(store.page_ref_total(), 5);
        assert_eq!(
            store.page_ref_count(store.get(first_key).expect("first").pages[0][0]),
            2
        );
        assert!(store.remove(first_key));
        assert_eq!(store.live_pages(), 3);
        assert!(store.remove(second_key));
        assert_eq!(store.live_pages(), 0);
        assert_eq!(store.bytes_used(), 0);
    }

    #[test]
    fn freeing_a_device_page_forgets_the_share_but_keeps_the_host_page() {
        let mut store = snapshots(1 << 30);
        let first = store
            .plan_store(meta(&[1], false), &pages(&[id(1)]))
            .expect("plan");
        let first_key = store.commit_store(first, 0);
        let page = store.get(first_key).expect("first").pages[0][0];
        store.device_page_freed(id(1));
        assert_eq!(store.page_device(page), None);
        assert_eq!(store.page_ref_count(page), 1);
        let second = store
            .plan_store(meta(&[1, 2], false), &pages(&[id(1), id(2)]))
            .expect("plan");
        assert_eq!(second.copies.len(), 2);
        // The old host page persists; the freed identity is copied into a new one.
        assert_eq!(store.live_pages(), 3);
    }

    #[test]
    fn commit_replaces_a_snapshot_with_the_same_tokens_and_kind() {
        let mut store = snapshots(1 << 30);
        let first = store
            .plan_store(meta(&[7, 8], false), &pages(&[id(1)]))
            .expect("plan");
        let first_key = store.commit_store(first, 0);
        let second = store
            .plan_store(meta(&[7, 8], false), &pages(&[id(2)]))
            .expect("plan");
        let second_key = store.commit_store(second, 1);
        assert_eq!(store.len(), 1);
        assert!(store.get(first_key).is_none());
        assert!(store.get(second_key).is_some());
        assert_eq!(store.live_pages(), 1);
        assert_eq!(
            store.bytes_used(),
            (test_layout().tail + test_layout().scores + test_layout().page) as u64
        );
    }

    #[test]
    fn eviction_skips_pinned_and_follows_bank_order() {
        let mut store = snapshots(1 << 30);
        let prompt = store
            .plan_store(
                SnapshotMeta {
                    kind: SnapshotKind::Prompt,
                    ..meta(&[1], false)
                },
                &pages(&[id(1)]),
            )
            .expect("plan");
        let prompt_key = store.commit_store(prompt, 0);
        let turn = store
            .plan_store(meta(&[2], false), &pages(&[id(2)]))
            .expect("plan");
        let turn_key = store.commit_store(turn, 1);
        store.pin(turn_key);
        let (evicted, freed) = store.evict_to(0);
        assert_eq!(evicted, vec![prompt_key]);
        assert!(freed > 0);
        assert_eq!(store.len(), 1);
        assert!(store.get(turn_key).is_some());
        store.unpin(turn_key);
        let (evicted, _) = store.evict_to(0);
        assert_eq!(evicted, vec![turn_key]);
        assert!(store.is_empty());
        assert_eq!(store.bytes_used(), 0);
    }

    #[test]
    fn lookup_matches_the_retention_rule_and_refreshes_access() {
        let mut store = snapshots(1 << 30);
        let plan = store
            .plan_store(meta(&[1, 2, 3, 4], false), &pages(&[id(1)]))
            .expect("plan");
        let key = store.commit_store(plan, 0);
        let hit = store.lookup(&[1, 2, 3, 4, 5], 42).expect("hit");
        assert_eq!(hit.key, key);
        assert_eq!(hit.kind, SnapshotKind::Turn);
        assert_eq!((hit.common, hit.frontier), (4, 4));
        assert_eq!(store.get(key).expect("snapshot").last_access_ns, 42);
        assert!(store.lookup(&[9], 43).is_none());
    }

    #[test]
    fn page_indices_are_reused_after_release() {
        let mut store = snapshots(1 << 30);
        let plan = store
            .plan_store(meta(&[1], false), &pages(&[id(1)]))
            .expect("plan");
        let key = store.commit_store(plan, 0);
        let page = store.get(key).expect("snapshot").pages[0][0];
        assert!(store.remove(key));
        let plan = store
            .plan_store(meta(&[2], false), &pages(&[id(2)]))
            .expect("plan");
        let key = store.commit_store(plan, 1);
        assert_eq!(store.get(key).expect("snapshot").pages[0][0], page);
    }

    #[test]
    fn layout_is_available_through_the_pool() {
        let store = snapshots(1 << 20);
        assert_eq!(store.pool().layout(), test_layout());
    }
}
