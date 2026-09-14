//! Test support: a pinned-pool stand-in and a resident-snapshot builder for the suites and
//! benches while HC-1 is in flight. Not part of the cache's runtime surface.
use crate::pool::{Class, Layout, PoolExhausted, Slab};
use crate::snapshot::{DevicePageId, Key, Pool, SnapshotMeta, Snapshots};
use crate::{SnapshotKind, COMPRESSORS};
use std::collections::HashMap;

/// The layout the suites use: the engine's page, tail and draft sizes with a small scores row
/// so the scores class is exercised (the daemon binding disables it).
pub fn test_layout() -> Layout {
    Layout {
        page: crate::PAGE_BYTES,
        tail: crate::TAIL_BYTES,
        draft: crate::DRAFT_BYTES,
        scores: 4096,
    }
}

/// A pool that hands out slabs against a byte quota without carving real chunks. `take` fails
/// when the next slab would exceed the quota; `give_back` returns its bytes. Handles are unique.
pub struct FakePool {
    layout: Layout,
    quota: u64,
    used: u64,
    next_chunk: u32,
    next_index: u32,
    live: HashMap<(Class, u32, u32), usize>,
    in_use: HashMap<Class, u64>,
}

impl FakePool {
    pub fn new(quota: u64, layout: Layout) -> Self {
        Self {
            layout,
            quota,
            used: 0,
            next_chunk: 0,
            next_index: 0,
            live: HashMap::new(),
            in_use: HashMap::new(),
        }
    }
    pub fn layout(&self) -> Layout {
        self.layout
    }
    pub fn slabs_in_use(&self, class: Class) -> u64 {
        self.in_use.get(&class).copied().unwrap_or(0)
    }
}

impl Pool for FakePool {
    fn take(&mut self, class: Class) -> Result<Slab, PoolExhausted> {
        let bytes = self.layout.size(class);
        if bytes == 0 || self.used + bytes as u64 > self.quota {
            return Err(PoolExhausted {
                class,
                needed_bytes: bytes,
                free_bytes: self.quota.saturating_sub(self.used),
            });
        }
        let slab = Slab {
            class,
            chunk: self.next_chunk,
            index: self.next_index,
        };
        self.next_index += 1;
        if self.next_index == 8 {
            self.next_index = 0;
            self.next_chunk += 1;
        }
        self.used += bytes as u64;
        *self.in_use.entry(class).or_insert(0) += 1;
        self.live.insert((class, slab.chunk, slab.index), bytes);
        Ok(slab)
    }
    fn give_back(&mut self, slab: Slab) {
        let bytes = self.live.remove(&(slab.class, slab.chunk, slab.index));
        debug_assert!(bytes.is_some(), "give_back of a slab not held");
        if let Some(bytes) = bytes {
            self.used -= bytes as u64;
            if let Some(count) = self.in_use.get_mut(&slab.class) {
                *count = count.saturating_sub(1);
            }
        }
    }
    fn bytes_used(&self) -> u64 {
        self.used
    }
}

/// Build `count` resident snapshots of `tokens` tokens each sharing a `prefix`-token prefix.
/// Pages are modelled at one per 100 tokens so the shared prefix is shared exactly.
pub fn resident_snapshots(
    count: usize,
    tokens: usize,
    prefix: usize,
) -> (Snapshots<FakePool>, Vec<Key>) {
    let mut store = Snapshots::new(FakePool::new(1u64 << 42, test_layout()));
    let prefix_tokens: Vec<u32> = (0..prefix as u32).collect();
    let prefix_pages: Vec<DevicePageId> = (0..(prefix / 100) as u32)
        .map(|page| DevicePageId {
            compressor: 0,
            page,
            generation: 0,
        })
        .collect();
    let mut keys = Vec::with_capacity(count);
    for index in 0..count {
        let mut sequence = prefix_tokens.clone();
        sequence.extend((0..(tokens - prefix) as u32).map(|token| (index as u32) * 8192 + token));
        let mut pages: [Vec<DevicePageId>; COMPRESSORS] = std::array::from_fn(|_| Vec::new());
        pages[0].extend_from_slice(&prefix_pages);
        pages[0].extend(
            (0..((tokens - prefix) / 100) as u32).map(|page| DevicePageId {
                compressor: 0,
                page: 1_000_000 + (index as u32) * 1000 + page,
                generation: 0,
            }),
        );
        let meta = SnapshotMeta {
            kind: SnapshotKind::Turn,
            tokens: sequence,
            end: tokens as u32,
            has_draft: false,
        };
        let plan = store.plan_store(meta, &pages).expect("resident plan");
        keys.push(store.commit_store(plan, 0));
    }
    (store, keys)
}
