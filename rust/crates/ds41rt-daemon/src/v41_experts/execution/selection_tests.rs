//! CPU-only stub tests for the dedicated-state budget and selection tables.
//! These fake the per-capacity scratch table the native library would
//! provide; no GPU, model or daemon is required. The helpers under test are
//! the same ones `plan_execution` and `execution_state` use for serving and
//! graph dispatch, so a table pass here constrains the live selection.
use super::*;

/// Deterministic scratch bytes per exported Spark capacity, standing in for
/// `v41_expert_info(capacity).scratch_bytes` on a native library.
fn stub_scratch_bytes(capacity: u32) -> Result<u64> {
    match capacity {
        1 => Ok(1 << 20),
        16 => Ok(16 << 20),
        80 => Ok(80 << 20),
        256 => Ok(256 << 20),
        1024 => Ok(1024 << 20),
        4096 => Ok(4096 << 20),
        _ => anyhow::bail!("stub has no scratch for capacity {capacity}"),
    }
}

/// Which dedicated capacity serves `rows`, or the main capacity when `None`.
/// Mirrors the production mapping without touching GPU state.
fn selected_capacity(
    rows: u32,
    capacity: u32,
    role: u32,
) -> Result<u32> {
    let (decode, batch16, small) =
        dedicated_scratch_bytes(role, capacity, stub_scratch_bytes)?;
    Ok(match select_dedicated_state(
        rows,
        decode > 0,
        batch16 > 0,
        if small > 0 { Some(80) } else { None },
    ) {
        Some(DedicatedState::Decode) => 1,
        Some(DedicatedState::Batch16) => 16,
        Some(DedicatedState::Small) => 80,
        None => capacity,
    })
}

#[test]
fn role1_budget_counts_each_optional_state_once() -> Result<()> {
    // Capacity 1: the main state IS the capacity-1 kernel; no extras.
    assert_eq!(dedicated_scratch_bytes(1, 1, stub_scratch_bytes)?, (0, 0, 0));
    // Capacity 16: only decode (capacity 1) is below the configured capacity.
    let (decode, batch16, small) = dedicated_scratch_bytes(1, 16, stub_scratch_bytes)?;
    assert_eq!((decode, batch16, small), (1 << 20, 0, 0));
    // Capacity 80: decode + batch16; capacity-80 remains the main state.
    let (decode, batch16, small) = dedicated_scratch_bytes(1, 80, stub_scratch_bytes)?;
    assert_eq!((decode, batch16, small), (1 << 20, 16 << 20, 0));
    // Capacity 1024: all three dedicated states are budgeted.
    let (decode, batch16, small) = dedicated_scratch_bytes(1, 1024, stub_scratch_bytes)?;
    assert_eq!((decode, batch16, small), (1 << 20, 16 << 20, 80 << 20));
    // Role 0 keeps a single state at every capacity.
    for capacity in [1, 16, 80, 1024] {
        assert_eq!(
            dedicated_scratch_bytes(0, capacity, stub_scratch_bytes)?,
            (0, 0, 0),
            "role 0 capacity {capacity} must not plan dedicated states"
        );
    }
    Ok(())
}

#[test]
fn role1_selection_table_covers_small_row_boundaries() -> Result<()> {
    // (configured capacity, rows, expected serving capacity)
    let cases = [
        // Capacity 16: rows 2..16 have no batch16 state and fall to the main state.
        (16, 1, 1),
        (16, 2, 16),
        (16, 16, 16),
        // Capacity 80: rows 2..16 get the new batch16 state; 17..80 the cap80 main.
        (80, 1, 1),
        (80, 2, 16),
        (80, 16, 16),
        (80, 17, 80),
        (80, 80, 80),
        // Capacity 1024: decode / batch16 / cap80 / main tiers.
        (1024, 1, 1),
        (1024, 2, 16),
        (1024, 16, 16),
        (1024, 17, 80),
        (1024, 80, 80),
        (1024, 81, 1024),
        (1024, 1024, 1024),
    ];
    for (capacity, rows, expected) in cases {
        assert_eq!(
            selected_capacity(rows, capacity, 1)?,
            expected,
            "capacity {capacity} rows {rows}"
        );
    }
    Ok(())
}

#[test]
fn selection_falls_back_when_optional_state_is_absent() {
    // No dedicated states at all: everything serves on the main state.
    assert_eq!(
        select_dedicated_state(1, false, false, None),
        None
    );
    assert_eq!(select_dedicated_state(16, false, false, None), None);
    assert_eq!(select_dedicated_state(80, false, false, None), None);
    // Decode present, batch16 absent: rows 2..16 fall through to cap80.
    assert_eq!(
        select_dedicated_state(1, true, false, Some(80)),
        Some(DedicatedState::Decode)
    );
    assert_eq!(
        select_dedicated_state(2, true, false, Some(80)),
        Some(DedicatedState::Small)
    );
    assert_eq!(
        select_dedicated_state(16, true, false, Some(80)),
        Some(DedicatedState::Small)
    );
    // Small absent (capacity > 16 but <= 80): rows above decode serve the main state.
    assert_eq!(select_dedicated_state(2, true, true, None), Some(DedicatedState::Batch16));
    assert_eq!(select_dedicated_state(17, true, true, None), None);
    // Row 1 prefers decode even when batch16 could serve it.
    assert_eq!(
        select_dedicated_state(1, true, true, Some(80)),
        Some(DedicatedState::Decode)
    );
    // Beyond the small capacity: main state.
    assert_eq!(
        select_dedicated_state(81, true, true, Some(80)),
        None
    );
}

#[test]
fn budget_total_counts_batch16_scratch_without_overflow() -> Result<()> {
    let budget = ExpertExecutionBudget {
        scratch_bytes: 1024,
        decode_scratch_bytes: 2048,
        batch16_scratch_bytes: 4096,
        small_scratch_bytes: 8192,
        hidden_bytes: 16,
        routing_bytes: 8,
        output_and_shared_bytes: 4,
    };
    // The batch16 arena must be part of the tracked total: drop it and the
    // sum must shrink by exactly its size.
    assert_eq!(budget.total()?, 1024 + 2048 + 4096 + 8192 + 16 + 8 + 4);
    let without_batch16 = ExpertExecutionBudget {
        batch16_scratch_bytes: 0,
        ..budget
    };
    assert_eq!(budget.total()? - without_batch16.total()?, 4096);
    // Untracked allocation guard: an extreme batch16 arena surfaces as an
    // overflow error from total() instead of silently wrapping.
    let overflowing = ExpertExecutionBudget {
        batch16_scratch_bytes: usize::MAX,
        ..budget
    };
    assert!(overflowing.total().is_err());
    Ok(())
}

#[test]
fn selection_matches_budgeted_states_for_every_role1_capacity() -> Result<()> {
    // Cross-check the two helpers against each other: every row count the
    // configured capacity can serve must land on a state the budget planned
    // (dedicated states exist exactly when their scratch bytes are nonzero).
    for capacity in [1u32, 16, 80, 1024] {
        let (decode, batch16, small) =
            dedicated_scratch_bytes(1, capacity, stub_scratch_bytes)?;
        for rows in 1..=capacity.min(96) {
            let selected = match select_dedicated_state(
                rows,
                decode > 0,
                batch16 > 0,
                if small > 0 { Some(80) } else { None },
            ) {
                Some(DedicatedState::Decode) => 1,
                Some(DedicatedState::Batch16) => 16,
                Some(DedicatedState::Small) => 80,
                None => capacity,
            };
            assert!(
                selected <= capacity,
                "capacity {capacity} rows {rows}: selected capacity {selected} \
                 exceeds the configured capacity"
            );
            // A dedicated selection is only legal when its scratch was budgeted.
            // (selected == capacity means the main state, even when that
            // capacity happens to be 1 or 16.)
            if selected < capacity {
                match selected {
                    1 => assert!(decode > 0, "capacity {capacity} rows {rows}: unbudgeted decode"),
                    16 => assert!(
                        batch16 > 0,
                        "capacity {capacity} rows {rows}: unbudgeted batch16"
                    ),
                    80 => assert!(small > 0, "capacity {capacity} rows {rows}: unbudgeted small"),
                    _ => panic!("unexpected dedicated capacity {selected}"),
                }
            }
        }
    }
    Ok(())
}
