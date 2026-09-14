//! Shared model for the HC-2 suites: mirrors `Snapshots` with a separately maintained
//! `Retention<Key>` and exact page-reference and byte accounting, so a random or interleaved
//! sequence can be checked after every step.
#![allow(dead_code)]
use ds41rt_core::prefix::{Retention, SnapshotKind};
use ds41rt_hostcache::pool::Layout;
use ds41rt_hostcache::snapshot::{DevicePageId, Key, SnapshotMeta, Snapshots};
use ds41rt_hostcache::testing::{test_layout, FakePool};
use ds41rt_hostcache::COMPRESSORS;
use std::collections::HashMap;

/// The expected state after a sequence of operations.
pub struct Model {
    pub retention: Retention<Key>,
    pub snapshots: HashMap<Key, ModelSnapshot>,
    /// Model page id to reference count.
    pub page_refs: HashMap<u32, u32>,
    pub page_device: HashMap<u32, DevicePageId>,
    /// Device identity to the model page that currently holds its bytes.
    pub mapped: HashMap<DevicePageId, u32>,
    pub bytes: u64,
    pub next_key: Key,
    pub next_page: u32,
    pub layout: Layout,
}

pub struct ModelSnapshot {
    pub kind: SnapshotKind,
    pub tokens: Vec<u32>,
    pub pages: Vec<u32>,
    pub has_draft: bool,
    pub pins: u32,
}

impl Model {
    pub fn new() -> Self {
        Self {
            retention: Retention::new(usize::MAX),
            snapshots: HashMap::new(),
            page_refs: HashMap::new(),
            page_device: HashMap::new(),
            mapped: HashMap::new(),
            bytes: 0,
            next_key: 0,
            next_page: 0,
            layout: test_layout(),
        }
    }

    /// Mirror a successful `plan_store` + `commit_store`.
    pub fn store(
        &mut self,
        meta: &SnapshotMeta,
        device_pages: &[Vec<DevicePageId>; COMPRESSORS],
    ) -> Key {
        let mut refs = Vec::new();
        let mut new_pages = 0u64;
        for list in device_pages {
            for &id in list {
                let page = match self.mapped.get(&id).copied() {
                    Some(page) => page,
                    None => {
                        let page = self.next_page;
                        self.next_page += 1;
                        self.mapped.insert(id, page);
                        self.page_device.insert(page, id);
                        self.page_refs.insert(page, 0);
                        new_pages += 1;
                        page
                    }
                };
                *self.page_refs.get_mut(&page).expect("model page") += 1;
                refs.push(page);
            }
        }
        self.bytes += self.layout.tail as u64 + self.layout.scores as u64;
        if meta.has_draft {
            self.bytes += self.layout.draft as u64;
        }
        self.bytes += new_pages * self.layout.page as u64;
        let key = self.next_key;
        self.next_key += 1;
        let replaced = self
            .retention
            .bank_mut(meta.kind)
            .lookup(&meta.tokens)
            .and_then(|(position, &old)| (position == meta.tokens.len()).then_some(old));
        if let Some(old) = replaced {
            self.release(old);
        }
        self.retention.bank_mut(meta.kind).insert(&meta.tokens, key);
        self.snapshots.insert(
            key,
            ModelSnapshot {
                kind: meta.kind,
                tokens: meta.tokens.clone(),
                pages: refs,
                has_draft: meta.has_draft,
                pins: 0,
            },
        );
        key
    }

    /// Drop a snapshot and every page reference it holds.
    pub fn release(&mut self, key: Key) {
        let Some(snapshot) = self.snapshots.remove(&key) else {
            return;
        };
        self.bytes -= self.layout.tail as u64 + self.layout.scores as u64;
        if snapshot.has_draft {
            self.bytes -= self.layout.draft as u64;
        }
        for page in snapshot.pages {
            let count = self.page_refs.get_mut(&page).expect("model page");
            *count -= 1;
            if *count == 0 {
                self.page_refs.remove(&page);
                let device = self.page_device.remove(&page).expect("model device");
                if self.mapped.get(&device) == Some(&page) {
                    self.mapped.remove(&device);
                }
                self.bytes -= self.layout.page as u64;
            }
        }
    }

    pub fn lookup(&mut self, tokens: &[u32]) -> Option<(usize, usize, Key)> {
        self.retention
            .lookup_reusable(tokens)
            .map(|(common, frontier, &key)| (common, frontier, key))
    }

    pub fn evict_to(&mut self, quota: u64) -> (Vec<Key>, u64) {
        let mut evicted = Vec::new();
        let mut freed = 0;
        while self.bytes > quota {
            let next = {
                let snapshots = &self.snapshots;
                self.retention.evict_one_where(&|key: &Key| {
                    snapshots.get(key).is_some_and(|snapshot| snapshot.pins > 0)
                })
            };
            let Some((_kind, key)) = next else {
                break;
            };
            let before = self.bytes;
            self.release(key);
            freed += before - self.bytes;
            evicted.push(key);
        }
        (evicted, freed)
    }

    pub fn remove(&mut self, key: Key) -> bool {
        let Some(snapshot) = self.snapshots.get(&key) else {
            return false;
        };
        let kind = snapshot.kind;
        let tokens = snapshot.tokens.clone();
        self.retention.bank_mut(kind).remove_exact(&tokens);
        self.release(key);
        true
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

    pub fn device_page_freed(&mut self, id: DevicePageId) {
        self.mapped.remove(&id);
    }

    /// Assert the real store matches the model exactly.
    pub fn check(&self, store: &Snapshots<FakePool>) {
        assert_eq!(store.len(), self.snapshots.len(), "snapshot count");
        assert_eq!(store.bytes_used(), self.bytes, "bytes used");
        assert_eq!(store.live_pages(), self.page_refs.len(), "live pages");
        assert_eq!(
            store.page_ref_total(),
            self.page_refs
                .values()
                .map(|&refs| u64::from(refs))
                .sum::<u64>(),
            "page reference total"
        );
        let mut real = store.page_ref_counts();
        let mut model: Vec<u32> = self.page_refs.values().copied().collect();
        real.sort_unstable();
        model.sort_unstable();
        assert_eq!(real, model, "page reference counts");
        for kind in [SnapshotKind::Prompt, SnapshotKind::Turn] {
            assert_eq!(
                store.retention().bank(kind).entries(),
                self.retention.bank(kind).entries(),
                "retention entries for {kind:?}"
            );
        }
        for (&key, expected) in &self.snapshots {
            let actual = store.get(key).expect("snapshot present");
            assert_eq!(actual.meta.kind, expected.kind);
            assert_eq!(actual.meta.tokens, expected.tokens);
            assert_eq!(actual.pins, expected.pins);
        }
    }
}

/// One step of a suite's schedule.
#[derive(Clone, Debug)]
pub enum Op {
    Store {
        kind: SnapshotKind,
        tokens: Vec<u32>,
        has_draft: bool,
        pages: Vec<DevicePageId>,
    },
    Lookup {
        tokens: Vec<u32>,
    },
    Evict {
        quota: u64,
    },
    Free {
        id: DevicePageId,
    },
    Pin {
        key: Key,
    },
    Unpin {
        key: Key,
    },
    Remove {
        key: Key,
    },
}

/// Apply `op` to the real store and the model, then assert they agree.
pub fn apply(store: &mut Snapshots<FakePool>, model: &mut Model, op: &Op, now_ns: u64) {
    match op {
        Op::Store {
            kind,
            tokens,
            has_draft,
            pages,
        } => {
            let mut device_pages: [Vec<DevicePageId>; COMPRESSORS] =
                std::array::from_fn(|_| Vec::new());
            for (index, &id) in pages.iter().enumerate() {
                device_pages[index % COMPRESSORS].push(id);
            }
            let meta = SnapshotMeta {
                kind: *kind,
                tokens: tokens.clone(),
                end: tokens.len() as u32,
                has_draft: *has_draft,
            };
            let plan = store
                .plan_store(meta.clone(), &device_pages)
                .expect("plan succeeds");
            let key = store.commit_store(plan, now_ns);
            let model_key = model.store(&meta, &device_pages);
            assert_eq!(key, model_key, "key");
        }
        Op::Lookup { tokens } => {
            let hit = store.lookup(tokens, now_ns);
            let expected = model.lookup(tokens);
            match (hit, expected) {
                (Some(hit), Some((common, frontier, key))) => {
                    assert_eq!(hit.key, key, "lookup key");
                    assert_eq!(hit.common, common, "lookup common");
                    assert_eq!(hit.frontier, frontier, "lookup frontier");
                    assert_eq!(hit.kind, model.snapshots[&key].kind, "lookup kind");
                }
                (None, None) => {}
                (hit, expected) => panic!("lookup mismatch: {hit:?} vs {expected:?}"),
            }
        }
        Op::Evict { quota } => {
            let (keys, freed) = store.evict_to(*quota);
            let (model_keys, model_freed) = model.evict_to(*quota);
            assert_eq!(keys, model_keys, "evicted keys");
            assert_eq!(freed, model_freed, "freed bytes");
        }
        Op::Free { id } => {
            store.device_page_freed(*id);
            model.device_page_freed(*id);
        }
        Op::Pin { key } => {
            store.pin(*key);
            model.pin(*key);
        }
        Op::Unpin { key } => {
            store.unpin(*key);
            model.unpin(*key);
        }
        Op::Remove { key } => {
            assert_eq!(store.remove(*key), model.remove(*key), "remove");
        }
    }
    model.check(store);
}
