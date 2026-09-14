//! Host snapshots (packet HC-2): what the cache holds and how it finds it. A host snapshot is
//! the four parts of a retained snapshot in pinned slabs plus its key-space tokens, frontier and
//! kind. Pages are shared exactly as the device shares them: while a device page is allocated,
//! its identity maps to at most one host page, so a page is copied once however many snapshots
//! reference it. Lookups use the engine's own `Retention` radix (`ds41rt-core::prefix`), so a
//! host hit is exactly a snapshot the device tier would have chosen. Eviction follows the same
//! bank order (prompts before turns, oldest access first) and never touches a pinned snapshot.
use crate::pool::{Slab, SlabPool};
use crate::SnapshotKind;
use crate::COMPRESSORS;
use ds41rt_core::prefix::Retention;
use serde::Serialize;

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

/// The store of host snapshots. Invariants: a page slab is held exactly while its reference
/// count is positive; the device map holds an entry only for a page that is both alive on the
/// device and present in a host slab; `bytes_used()` equals the pool's; eviction order equals
/// `Retention::evict_one` order over unpinned snapshots.
pub struct Snapshots {
    _private: (),
}

impl Snapshots {
    pub fn new(pool: SlabPool) -> Self {
        let _ = pool;
        unimplemented!("HC-2")
    }
    /// Allocate slabs for `meta` and every page of `device_pages` not already shared. On
    /// `PoolExhausted` nothing is held. Duplicate tokens for the same kind replace the older
    /// snapshot on commit (the newer frontier wins, as in the engine's radix).
    pub fn plan_store(
        &mut self,
        meta: SnapshotMeta,
        device_pages: &[Vec<DevicePageId>; COMPRESSORS],
    ) -> Result<StorePlan, crate::pool::PoolExhausted> {
        let _ = (meta, device_pages);
        unimplemented!("HC-2")
    }
    /// After every copy of `plan` completed: make the snapshot lookup-visible.
    pub fn commit_store(&mut self, plan: StorePlan, now_ns: u64) -> Key {
        let _ = (plan, now_ns);
        unimplemented!("HC-2")
    }
    /// A store that failed: release its slabs and drop its page references.
    pub fn abort_store(&mut self, plan: StorePlan) {
        let _ = plan;
        unimplemented!("HC-2")
    }
    /// The engine's reuse rule over host snapshots; refreshes the hit's access clock.
    pub fn lookup(&mut self, tokens: &[u32], now_ns: u64) -> Option<Hit> {
        let _ = (tokens, now_ns);
        unimplemented!("HC-2")
    }
    pub fn get(&self, key: Key) -> Option<&HostSnapshot> {
        let _ = key;
        unimplemented!("HC-2")
    }
    pub fn pin(&mut self, key: Key) {
        let _ = key;
        unimplemented!("HC-2")
    }
    pub fn unpin(&mut self, key: Key) {
        let _ = key;
        unimplemented!("HC-2")
    }
    /// A device page was freed: its identity can no longer be shared from the device.
    pub fn device_page_freed(&mut self, id: DevicePageId) {
        let _ = id;
        unimplemented!("HC-2")
    }
    /// Evict in the engine's order until `bytes_used() <= quota`, skipping pinned snapshots;
    /// returns the evicted keys and the bytes freed. Stops early if only pinned remain.
    pub fn evict_to(&mut self, quota: u64) -> (Vec<Key>, u64) {
        let _ = quota;
        unimplemented!("HC-2")
    }
    pub fn remove(&mut self, key: Key) -> bool {
        let _ = key;
        unimplemented!("HC-2")
    }
    pub fn len(&self) -> usize {
        unimplemented!("HC-2")
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn bytes_used(&self) -> u64 {
        unimplemented!("HC-2")
    }
    pub fn pool(&self) -> &SlabPool {
        unimplemented!("HC-2")
    }
    /// The radix, for the suites that assert a host hit equals a `Retention` hit.
    pub fn retention(&self) -> &Retention<Key> {
        unimplemented!("HC-2")
    }
}
