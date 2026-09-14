//! Functional suite for the pinned slab pool (HC-1): every class round-trips, the quota is exact
//! at the boundary, chunks are claimed and released, locations never overlap, a failed
//! construction releases what it allocated, and random take/give_back sequences match a model.
use ds41rt_hostcache::pool::testing::FakePinned;
use ds41rt_hostcache::pool::{Class, ClassOccupancy, Layout, PoolExhausted, Slab, SlabPool};
use proptest::prelude::*;
use std::collections::HashSet;

const CHUNK: usize = 1 << 16;

fn layout() -> Layout {
    Layout {
        page: 4096,
        tail: 8192,
        draft: 2048,
        scores: 1024,
    }
}

fn pool(quota: u64) -> (SlabPool, FakePinned) {
    let mut mem = FakePinned::new(usize::MAX);
    let pool = SlabPool::new(quota, CHUNK, layout(), &mut mem).expect("pool");
    (pool, mem)
}

#[test]
fn every_class_round_trips() {
    let (mut pool, _mem) = pool(4 * CHUNK as u64);
    for class in Class::ALL {
        let size = layout().size(class);
        let slab = pool.take(class).expect("take");
        assert_eq!(slab.class, class);
        let range = pool.location(slab);
        assert_eq!(range.bytes, size);
        assert_eq!(range.offset % size, 0);
        assert!(range.offset + size <= CHUNK);
        assert_eq!(pool.bytes_used(), size as u64);
        pool.give_back(slab);
        assert_eq!(pool.bytes_used(), 0);
    }
}

#[test]
fn quota_boundary_exact() {
    let page = layout().page;
    let chunk = page * 4;
    let mut mem = FakePinned::new(usize::MAX);
    let mut pool = SlabPool::new(chunk as u64, chunk, layout(), &mut mem).expect("pool");
    for taken in 0..4u64 {
        assert_eq!(pool.free_bytes(Class::Page), (4 - taken) * page as u64);
        let _slab = pool.take(Class::Page).expect("take");
        assert_eq!(pool.bytes_used(), (taken + 1) * page as u64);
    }
    assert_eq!(pool.free_bytes(Class::Page), 0);
    let err = pool.take(Class::Page).unwrap_err();
    assert_eq!(
        err,
        PoolExhausted {
            class: Class::Page,
            needed_bytes: page,
            free_bytes: 0,
        }
    );
}

#[test]
fn free_bytes_counts_free_chunks() {
    let page = layout().page;
    let per_chunk = CHUNK / page;
    let (mut pool, _mem) = pool(2 * CHUNK as u64);
    assert_eq!(
        pool.free_bytes(Class::Page),
        2 * per_chunk as u64 * page as u64
    );
    let slab = pool.take(Class::Page).expect("take");
    assert_eq!(
        pool.free_bytes(Class::Page),
        (2 * per_chunk - 1) as u64 * page as u64
    );
    pool.give_back(slab);
    assert_eq!(
        pool.free_bytes(Class::Page),
        2 * per_chunk as u64 * page as u64
    );
}

#[test]
fn chunk_claim_and_release() {
    let (mut pool, _mem) = pool(CHUNK as u64);
    let page = pool.take(Class::Page).expect("take");
    let page_chunk = pool.location(page).chunk;
    assert_eq!(pool.occupancy()[0].1.chunks, 1);
    pool.give_back(page);
    assert_eq!(pool.occupancy()[0].1.chunks, 0);
    assert_eq!(
        pool.free_bytes(Class::Page),
        (CHUNK / layout().page) as u64 * layout().page as u64
    );
    let tail = pool.take(Class::Tail).expect("take");
    assert_eq!(pool.location(tail).chunk, page_chunk);
    assert_eq!(pool.occupancy()[1].1.chunks, 1);
    assert_eq!(pool.occupancy()[0].1.chunks, 0);
}

#[test]
fn location_offsets_never_overlap() {
    let (mut pool, _mem) = pool(CHUNK as u64);
    let size = layout().page;
    let mut ranges = Vec::new();
    while let Ok(slab) = pool.take(Class::Page) {
        ranges.push(pool.location(slab));
    }
    assert_eq!(ranges.len(), CHUNK / size);
    let mut seen = HashSet::new();
    for range in &ranges {
        assert!(range.offset + range.bytes <= CHUNK);
        assert!(seen.insert((range.chunk, range.offset)));
    }
}

#[test]
fn constructor_failure_releases_partial() {
    let mut mem = FakePinned::new(2);
    let result = SlabPool::new(5 * CHUNK as u64, CHUNK, layout(), &mut mem);
    assert!(result.is_err());
    assert_eq!(mem.allocations(), 2);
    assert_eq!(mem.releases(), 2);
    assert_eq!(mem.live_chunks(), 0);
    assert_eq!(mem.live_bytes(), 0);
}

#[test]
fn release_all_returns_every_chunk() {
    let (pool, mut mem) = pool(3 * CHUNK as u64);
    assert_eq!(mem.live_chunks(), 3);
    pool.release_all(&mut mem).expect("release_all");
    assert_eq!(mem.releases(), 3);
    assert_eq!(mem.live_chunks(), 0);
    assert_eq!(mem.live_bytes(), 0);
}

#[test]
fn quota_below_chunk_allocates_nothing() {
    let (mut pool, mem) = pool(CHUNK as u64 - 1);
    assert_eq!(mem.allocations(), 0);
    assert_eq!(pool.quota(), CHUNK as u64 - 1);
    assert_eq!(pool.free_bytes(Class::Page), 0);
    assert!(pool.take(Class::Page).is_err());
}

#[test]
fn occupancy_tracks_use() {
    let (mut pool, _mem) = pool(2 * CHUNK as u64);
    let a = pool.take(Class::Page).expect("take");
    let b = pool.take(Class::Page).expect("take");
    let c = pool.take(Class::Draft).expect("take");
    let occ = pool.occupancy();
    assert_eq!(
        occ[0].1,
        ClassOccupancy {
            slabs_in_use: 2,
            slabs_free: (CHUNK / layout().page - 2) as u64,
            chunks: 1,
        }
    );
    assert_eq!(
        occ[2].1,
        ClassOccupancy {
            slabs_in_use: 1,
            slabs_free: (CHUNK / layout().draft - 1) as u64,
            chunks: 1,
        }
    );
    assert_eq!(occ[1].1, ClassOccupancy::default());
    assert_eq!(occ[3].1, ClassOccupancy::default());
    pool.give_back(a);
    pool.give_back(b);
    pool.give_back(c);
    assert_eq!(pool.bytes_used(), 0);
    for (_, occupancy) in pool.occupancy() {
        assert_eq!(occupancy, ClassOccupancy::default());
    }
}

#[test]
fn bytes_used_exact_mixed() {
    let (mut pool, _mem) = pool(4 * CHUNK as u64);
    let mut held = Vec::new();
    let mut expected = 0u64;
    for class in Class::ALL {
        for _ in 0..3 {
            held.push(pool.take(class).expect("take"));
            expected += layout().size(class) as u64;
            assert_eq!(pool.bytes_used(), expected);
        }
    }
    for slab in held {
        pool.give_back(slab);
        expected -= layout().size(slab.class) as u64;
        assert_eq!(pool.bytes_used(), expected);
    }
    assert_eq!(expected, 0);
}

#[test]
fn layout_and_quota_accessors() {
    let (pool, _mem) = pool(2 * CHUNK as u64);
    assert_eq!(pool.layout(), layout());
    assert_eq!(pool.quota(), 2 * CHUNK as u64);
}

#[test]
fn new_rejects_oversized_class() {
    let mut mem = FakePinned::new(usize::MAX);
    let bad = Layout {
        page: CHUNK + 1,
        tail: 1,
        draft: 1,
        scores: 1,
    };
    assert!(SlabPool::new(CHUNK as u64, CHUNK, bad, &mut mem).is_err());
    assert_eq!(mem.allocations(), 0);
}

#[test]
fn new_rejects_zero_chunk() {
    let mut mem = FakePinned::new(usize::MAX);
    assert!(SlabPool::new(0, 0, layout(), &mut mem).is_err());
}

#[test]
fn engine_layout_round_trips() {
    let engine = Layout::engine(4096);
    let chunk = engine.tail;
    let mut mem = FakePinned::new(usize::MAX);
    let mut pool = SlabPool::new(chunk as u64, chunk, engine, &mut mem).expect("pool");
    let slab = pool.take(Class::Page).expect("take");
    assert_eq!(pool.location(slab).bytes, ds41rt_hostcache::PAGE_BYTES);
    pool.give_back(slab);
}

/// Performance floor: a broken fast path (e.g. a scan on every take) must fail `cargo test`, not
/// a human. Debug build, so the bound is generous.
#[test]
fn page_take_give_back_floor() {
    let (mut pool, _mem) = pool(CHUNK as u64);
    let keeper = pool.take(Class::Page).expect("take");
    let start = std::time::Instant::now();
    for _ in 0..100_000 {
        let slab = pool.take(Class::Page).expect("take");
        pool.give_back(slab);
    }
    let elapsed = start.elapsed();
    pool.give_back(keeper);
    assert!(
        elapsed < std::time::Duration::from_secs(2),
        "100k page take+give_back took {elapsed:?}"
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn random_take_give_back_matches_model(
        ops in prop::collection::vec((0u8..4, 0u8..2, 0usize..64), 0..3000)
    ) {
        let quota = 4 * CHUNK as u64;
        let (mut pool, _mem) = pool(quota);
        let mut held: Vec<Slab> = Vec::new();
        let mut held_set: HashSet<Slab> = HashSet::new();
        let mut used = 0u64;
        for (class_index, action, pick) in ops {
            let class = Class::ALL[class_index as usize];
            let size = layout().size(class) as u64;
            if action == 0 {
                match pool.take(class) {
                    Ok(slab) => {
                        prop_assert!(held_set.insert(slab), "live slab handed out twice: {slab:?}");
                        held.push(slab);
                        used += size;
                    }
                    Err(err) => {
                        prop_assert_eq!(err.class, class);
                        prop_assert_eq!(err.needed_bytes, layout().size(class));
                        prop_assert_eq!(err.free_bytes, 0);
                    }
                }
            } else {
                let of_class: Vec<usize> = held
                    .iter()
                    .enumerate()
                    .filter(|(_, slab)| slab.class == class)
                    .map(|(index, _)| index)
                    .collect();
                if !of_class.is_empty() {
                    let slab = held.swap_remove(of_class[pick % of_class.len()]);
                    held_set.remove(&slab);
                    pool.give_back(slab);
                    used -= size;
                }
            }
            prop_assert_eq!(pool.bytes_used(), used);
            prop_assert!(used <= quota);
            let in_use: u64 = pool.occupancy().iter().map(|(_, occ)| occ.slabs_in_use).sum();
            prop_assert_eq!(in_use, held.len() as u64);
        }
        for slab in held {
            pool.give_back(slab);
        }
        prop_assert_eq!(pool.bytes_used(), 0);
    }

    #[test]
    fn location_offsets_are_disjoint_within_chunk(
        ops in prop::collection::vec((0u8..4, 0usize..32), 0..2000)
    ) {
        let (mut pool, _mem) = pool(4 * CHUNK as u64);
        let mut held: Vec<Slab> = Vec::new();
        for (class_index, pick) in ops {
            let class = Class::ALL[class_index as usize];
            if pick % 3 == 0 && !held.is_empty() {
                let slab = held.swap_remove(pick % held.len());
                pool.give_back(slab);
            } else if let Ok(slab) = pool.take(class) {
                held.push(slab);
            }
        }
        let mut seen = HashSet::new();
        for slab in held {
            let range = pool.location(slab);
            prop_assert!(range.offset + range.bytes <= CHUNK);
            prop_assert!(
                seen.insert((range.chunk, range.offset)),
                "overlapping ranges in chunk {}",
                range.chunk
            );
        }
    }
}
