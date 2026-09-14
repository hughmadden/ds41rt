//! Functional suite for packet HC-2: host snapshots, exact page sharing, the reuse radix and
//! eviction. Every public operation on its happy path, every documented error path, and
//! proptest properties over random store/lookup/evict/free sequences against a model.
mod common;

use common::{apply, Model, Op};
use ds41rt_hostcache::pool::Class;
use ds41rt_hostcache::snapshot::{DevicePageId, SnapshotMeta, Snapshots};
use ds41rt_hostcache::testing::{resident_snapshots, test_layout, FakePool};
use ds41rt_hostcache::{SnapshotKind, COMPRESSORS};
use proptest::prelude::*;

fn id(page: u32) -> DevicePageId {
    DevicePageId {
        compressor: 0,
        page,
        generation: 0,
    }
}

fn meta(kind: SnapshotKind, tokens: &[u32], has_draft: bool) -> SnapshotMeta {
    SnapshotMeta {
        kind,
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

fn store(quota: u64) -> Snapshots<FakePool> {
    Snapshots::new(FakePool::new(quota, test_layout()))
}

#[test]
fn store_lookup_evict_happy_path() {
    let mut store = store(1 << 30);
    let plan = store
        .plan_store(
            meta(SnapshotKind::Turn, &[1, 2, 3], true),
            &pages(&[id(1), id(2)]),
        )
        .expect("plan");
    assert_eq!(plan.copies.len(), 2);
    assert!(plan.draft.is_some());
    let key = store.commit_store(plan, 10);
    assert_eq!(store.len(), 1);
    assert!(!store.is_empty());
    let hit = store.lookup(&[1, 2, 3, 4], 20).expect("hit");
    assert_eq!(hit.key, key);
    assert_eq!(hit.kind, SnapshotKind::Turn);
    assert_eq!((hit.common, hit.frontier), (3, 3));
    assert_eq!(store.get(key).expect("snapshot").last_access_ns, 20);
    let (evicted, freed) = store.evict_to(0);
    assert_eq!(evicted, vec![key]);
    assert!(freed > 0);
    assert!(store.is_empty());
    assert_eq!(store.bytes_used(), 0);
    assert_eq!(store.live_pages(), 0);
}

#[test]
fn plan_exhaustion_holds_nothing() {
    let layout = test_layout();
    let mut store = store((layout.tail + layout.scores) as u64);
    let error = store
        .plan_store(meta(SnapshotKind::Turn, &[1], false), &pages(&[id(1)]))
        .expect_err("page class exhausted");
    assert_eq!(error.class, Class::Page);
    assert_eq!(store.bytes_used(), 0);
    assert_eq!(store.live_pages(), 0);
    assert!(store.is_empty());
}

#[test]
fn exact_sharing_copies_shared_pages_once() {
    let mut store = store(1 << 30);
    let first = store
        .plan_store(
            meta(SnapshotKind::Turn, &[1, 2, 3], false),
            &pages(&[id(1), id(2), id(3)]),
        )
        .expect("plan");
    assert_eq!(first.copies.len(), 3);
    let first_key = store.commit_store(first, 0);
    let second = store
        .plan_store(
            meta(SnapshotKind::Turn, &[1, 2, 4], false),
            &pages(&[id(1), id(2), id(4)]),
        )
        .expect("plan");
    assert_eq!(second.copies.len(), 1);
    assert_eq!(second.copies[0].2, id(4));
    let second_key = store.commit_store(second, 1);
    assert_eq!(store.pool().slabs_in_use(Class::Page), 4);
    assert_eq!(store.live_pages(), 4);
    let shared = store.get(first_key).expect("first").pages[0][0];
    assert_eq!(store.page_ref_count(shared), 2);
    assert!(store.remove(first_key));
    // id(3) was private to the first snapshot; the two shared pages stay.
    assert_eq!(store.pool().slabs_in_use(Class::Page), 3);
    assert!(store.remove(second_key));
    assert_eq!(store.pool().slabs_in_use(Class::Page), 0);
    assert_eq!(store.bytes_used(), 0);
}

#[test]
fn freed_device_page_is_copied_again_while_the_old_host_page_persists() {
    let mut store = store(1 << 30);
    let first = store
        .plan_store(meta(SnapshotKind::Turn, &[1], false), &pages(&[id(1)]))
        .expect("plan");
    let first_key = store.commit_store(first, 0);
    let old_page = store.get(first_key).expect("first").pages[0][0];
    store.device_page_freed(id(1));
    assert_eq!(store.page_device(old_page), None);
    assert_eq!(store.page_ref_count(old_page), 1);
    let second = store
        .plan_store(
            meta(SnapshotKind::Turn, &[1, 2], false),
            &pages(&[id(1), id(2)]),
        )
        .expect("plan");
    assert_eq!(second.copies.len(), 2);
    let second_key = store.commit_store(second, 1);
    let new_page = store.get(second_key).expect("second").pages[0][0];
    assert_ne!(old_page, new_page);
    assert_eq!(store.page_device(new_page), Some(id(1)));
    assert_eq!(store.live_pages(), 3);
}

#[test]
fn replace_on_same_tokens_releases_the_old_snapshot() {
    let mut store = store(1 << 30);
    let first = store
        .plan_store(meta(SnapshotKind::Turn, &[5, 6], false), &pages(&[id(1)]))
        .expect("plan");
    let first_key = store.commit_store(first, 0);
    let second = store
        .plan_store(meta(SnapshotKind::Turn, &[5, 6], false), &pages(&[id(2)]))
        .expect("plan");
    let second_key = store.commit_store(second, 1);
    assert_eq!(store.len(), 1);
    assert!(store.get(first_key).is_none());
    assert!(store.get(second_key).is_some());
    assert_eq!(store.live_pages(), 1);
    assert_eq!(store.pool().slabs_in_use(Class::Page), 1);
}

#[test]
fn pin_blocks_eviction_until_unpinned() {
    let mut store = store(1 << 30);
    let a = store
        .plan_store(meta(SnapshotKind::Prompt, &[1], false), &pages(&[id(1)]))
        .expect("plan");
    let a_key = store.commit_store(a, 0);
    let b = store
        .plan_store(meta(SnapshotKind::Turn, &[2], false), &pages(&[id(2)]))
        .expect("plan");
    let b_key = store.commit_store(b, 1);
    store.pin(a_key);
    let (evicted, _) = store.evict_to(0);
    assert_eq!(evicted, vec![b_key]);
    assert!(store.get(a_key).is_some());
    let (evicted, _) = store.evict_to(0);
    assert!(evicted.is_empty());
    store.unpin(a_key);
    let (evicted, _) = store.evict_to(0);
    assert_eq!(evicted, vec![a_key]);
    assert!(store.is_empty());
}

#[test]
fn eviction_order_is_prompts_before_turns_oldest_first() {
    let mut store = store(1 << 30);
    let p1 = store
        .plan_store(meta(SnapshotKind::Prompt, &[1], false), &pages(&[id(1)]))
        .expect("plan");
    let p1 = store.commit_store(p1, 0);
    let p2 = store
        .plan_store(meta(SnapshotKind::Prompt, &[2], false), &pages(&[id(2)]))
        .expect("plan");
    let p2 = store.commit_store(p2, 1);
    let t1 = store
        .plan_store(meta(SnapshotKind::Turn, &[3], false), &pages(&[id(3)]))
        .expect("plan");
    let t1 = store.commit_store(t1, 2);
    assert!(store.lookup(&[2], 3).is_some());
    let (evicted, _) = store.evict_to(0);
    assert_eq!(evicted, vec![p1, p2, t1]);
}

#[test]
fn remove_reports_whether_the_key_was_present() {
    let mut store = store(1 << 30);
    let plan = store
        .plan_store(meta(SnapshotKind::Turn, &[1], false), &pages(&[id(1)]))
        .expect("plan");
    let key = store.commit_store(plan, 0);
    assert!(store.remove(key));
    assert!(!store.remove(key));
    assert!(store.lookup(&[1], 1).is_none());
}

#[test]
fn abort_store_releases_every_part() {
    let mut store = store(1 << 30);
    let plan = store
        .plan_store(
            meta(SnapshotKind::Turn, &[1, 2], true),
            &pages(&[id(1), id(2)]),
        )
        .expect("plan");
    store.abort_store(plan);
    assert_eq!(store.bytes_used(), 0);
    assert_eq!(store.live_pages(), 0);
    assert!(store.is_empty());
}

fn device_id() -> impl Strategy<Value = DevicePageId> {
    (0u8..4, 0u32..8, 0u32..3).prop_map(|(compressor, page, generation)| DevicePageId {
        compressor,
        page,
        generation,
    })
}

fn token_sequence() -> impl Strategy<Value = Vec<u32>> {
    prop::collection::vec(0u32..4, 1..8)
}

fn operation() -> impl Strategy<Value = Op> {
    prop_oneof![
        (
            any::<bool>(),
            token_sequence(),
            any::<bool>(),
            prop::collection::vec(device_id(), 0..6),
        )
            .prop_map(|(turn, tokens, has_draft, pages)| Op::Store {
                kind: if turn {
                    SnapshotKind::Turn
                } else {
                    SnapshotKind::Prompt
                },
                tokens,
                has_draft,
                pages,
            }),
        token_sequence().prop_map(|tokens| Op::Lookup { tokens }),
        (0u64..3).prop_map(|quota| Op::Evict {
            quota: quota * 1_000_000
        }),
        device_id().prop_map(|id| Op::Free { id }),
        (0u64..20).prop_map(|key| Op::Pin { key }),
        (0u64..20).prop_map(|key| Op::Unpin { key }),
        (0u64..20).prop_map(|key| Op::Remove { key }),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]
    #[test]
    fn random_sequences_match_the_model(ops in prop::collection::vec(operation(), 1..40)) {
        let mut store = Snapshots::new(FakePool::new(1 << 40, test_layout()));
        let mut model = Model::new();
        for (step, op) in ops.iter().enumerate() {
            apply(&mut store, &mut model, op, step as u64);
        }
    }
}

/// Order-of-magnitude floor: a broken fast path fails `cargo test`, not a human. Debug build,
/// so the bound is generous.
#[test]
fn lookup_floor_at_10k_snapshots() {
    let (mut store, keys) = resident_snapshots(10_000, 4096, 2048);
    assert_eq!(store.len(), 10_000);
    let tokens = store.get(keys[0]).expect("snapshot").meta.tokens.clone();
    let start = std::time::Instant::now();
    for step in 0..1000 {
        let hit = store.lookup(&tokens, step).expect("hit");
        assert_eq!(hit.key, keys[0]);
    }
    let elapsed = start.elapsed();
    assert!(
        elapsed < std::time::Duration::from_secs(2),
        "1000 lookups at 10k snapshots took {elapsed:?}"
    );
}
