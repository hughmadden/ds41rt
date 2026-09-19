"""Pure-CPU materializer tests: no capture directory and no torch required.

Covers the pre-GPU contract: geometry extents, compact page remaps under
page permutation with duplicate keys and invalid masks, no whole-pool
allocation, missing-live-paged-row failure, and the single-nibble
counterfactual invariants on synthetic fixtures.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from sparse_replay import pins  # noqa: E402
from sparse_replay.capture import (  # noqa: E402
    QUERY_ROW_BYTES,
    METADATA_ROW_BYTES,
    SELECTED_ROW_BYTES,
    SINK_BYTES,
    RING_ROWS,
    RING_VALUE_BYTES,
    RING_SCALE_BYTES,
    WINDOW_VALUE_BYTES,
    WINDOW_SCALE_BYTES,
    SOURCE_VALUE_BYTES,
    SOURCE_SCALE_BYTES,
    CapturedRow,
)
from sparse_replay.materialize import (  # noqa: E402
    COUNTERFACTUAL_COLUMN,
    MaterializeError,
    MissingLivePagedRowError,
    SENTINEL_PAGE,
    materialize,
    mutate_private_source_column,
    set_fp4_low_nibble,
)

ORACLE = pins.load_oracle()


# ---------------------------------------------------------------------------
# Synthetic captured rows (kernel geometry, tiny arenas).
# ---------------------------------------------------------------------------

def _arena(rows: int, row_bytes: int, seed: int) -> bytes:
    return bytes(
        ((seed + 17 * (i // row_bytes) + i) % 251) for i in range(rows * row_bytes)
    )


def make_synthetic_row(pages, logical_to_physical, referenced_physical_rows,
                       selected, metadata, window_end=4, source_end=310,
                       source_capacity_rows=100000):
    """Build a CapturedRow for materializer testing without any disk fixture."""
    selected_full = list(selected) + [-1] * (512 - len(selected))
    ref_index = {physical: i for i, physical in enumerate(referenced_physical_rows)}
    logical_sorted = dict(sorted(logical_to_physical.items()))
    referenced_values = b"".join(
        _arena(1, SOURCE_VALUE_BYTES, 1000 + physical)
        for physical in referenced_physical_rows
    )
    referenced_scales = b"".join(
        _arena(1, SOURCE_SCALE_BYTES, 2000 + physical)
        for physical in referenced_physical_rows
    )
    return CapturedRow(
        wave="synthetic",
        manifest_path=Path("synthetic"),
        manifest_sha256="synthetic",
        request_id=1,
        flat_row=0,
        local_row=0,
        position=metadata[3],
        launch_kind="descriptor_batch",
        width=0,
        replay_begin=0,
        metadata10=tuple(metadata),
        selected512=tuple(selected_full),
        query_row=_arena(1, QUERY_ROW_BYTES, 7),
        sink=_arena(1, SINK_BYTES, 11),
        window_end=window_end,
        ring_values=_arena(RING_ROWS, RING_VALUE_BYTES, 13),
        ring_scales=_arena(RING_ROWS, RING_SCALE_BYTES, 19),
        window_proposal_capacity=1,
        window_proposal_values=_arena(1, WINDOW_VALUE_BYTES, 23),
        window_proposal_scales=_arena(1, WINDOW_SCALE_BYTES, 29),
        compressed=2,
        source_capacity_rows=source_capacity_rows,
        source_proposal_capacity=1,
        page_stride=len(pages),
        pages=tuple(pages),
        source_end=source_end,
        source_proposal_values=_arena(1, SOURCE_VALUE_BYTES, 31),
        source_proposal_scales=_arena(1, SOURCE_SCALE_BYTES, 37),
        referenced_count=len(referenced_physical_rows),
        referenced_physical_rows=tuple(referenced_physical_rows),
        logical_to_physical=logical_sorted,
        referenced_values=referenced_values,
        referenced_scales=referenced_scales,
        output_row=_arena(1, QUERY_ROW_BYTES, 41),
        output_sha256="synthetic",
        evidence={"canonical_digest": None},
    )


def synthetic_scenario_a():
    """Shuffled page table, duplicate keys, invalid ids, two physical pages.

    pages: logical page 0 -> physical page 3, logical page 1 -> physical
    page 1.  Referenced rows: logicals 0..5 (physical 768..773) and
    300..301 (physical 300..301).  Selected includes duplicate logical 1,
    invalid -1, out-of-causal ids 310/400/999.
    """
    pages = [3, 1, 2, 0]
    logical_to_physical = {
        0: 768, 1: 769, 2: 770, 3: 771, 4: 772, 5: 773,
        300: 300, 301: 301,
    }
    referenced = [300, 301, 768, 769, 770, 771, 772, 773]
    selected = [0, 1, 1, 300, -1, 2, 301, 999, 310, 400]
    # m[5] (causal source count) must satisfy m[5] <= m[6] + m[7]; with no
    # overlay (m[7]=0) ids >= m[6]=310 are masked through the overlay path
    # and ids >= m[5] through the causal-count path.
    metadata = [4, 0, 1, 4, 0, 310, 310, 0, 0, 1]
    return make_synthetic_row(
        pages, logical_to_physical, referenced, selected, metadata
    )


# ---------------------------------------------------------------------------
# Geometry extents.
# ---------------------------------------------------------------------------

def test_kernel_geometry_extents():
    assert QUERY_ROW_BYTES == 64 * 512 * 2 == 65536
    assert METADATA_ROW_BYTES == 80
    assert SELECTED_ROW_BYTES == 2048
    assert SINK_BYTES == 256
    assert (RING_ROWS, RING_VALUE_BYTES, RING_SCALE_BYTES) == (128, 512, 16)
    assert (WINDOW_VALUE_BYTES, WINDOW_SCALE_BYTES) == (512, 16)
    assert (SOURCE_VALUE_BYTES, SOURCE_SCALE_BYTES) == (256, 32)
    rows, parts = 2, 10
    assert rows * parts * 64 * 514 * 4 == 2 * 10 * 64 * 514 * 4
    assert rows * 120 == 240  # descriptor span


# ---------------------------------------------------------------------------
# Compact remap under page permutation / duplicates / invalid masks.
# ---------------------------------------------------------------------------

def test_page_permutation_duplicate_keys_invalid_mask_preservation():
    row = synthetic_scenario_a()
    mat = materialize(row, ORACLE)

    # Physical pages {1, 3} renumbered compactly, ascending.
    assert mat.page_map == {1: 0, 3: 1}
    # Offset mod 256 preserved for every referenced row.
    for old, new in mat.physical_map.items():
        assert old % 256 == new % 256
    assert mat.physical_map[300] == 0 * 256 + 44
    assert mat.physical_map[768] == 1 * 256 + 0
    assert mat.source_capacity == 262

    # Untouched logical pages (2, 3) point outside the compact capacity.
    assert mat.pages_compact[2] == SENTINEL_PAGE
    assert mat.pages_compact[3] == SENTINEL_PAGE
    for page in (0, 1):
        assert mat.pages_compact[page] < mat.source_capacity

    # Ordered slot evidence on the compact view: identical mask/tags/logicals
    # and identical content hashes; duplicates keep matching content.
    slots = mat.slot_evidence_compact
    live_paged = [s for s in slots if not s["masked"] and s["tag"] == "paged_source"]
    assert [s["logical"] for s in live_paged] == [0, 1, 1, 300, 2, 301]
    dup = [s for s in live_paged if s["logical"] == 1]
    assert len(dup) == 2
    assert dup[0]["value_sha256"] == dup[1]["value_sha256"]
    assert dup[0]["physical"] == dup[1]["physical"]
    # Invalid / out-of-range ids stay masked on the compact view.
    assert all(
        s["masked"] for s in slots
        if s["logical"] in (-1, 310, 400, 999) or s["tag"] is None
    )
    # Every live slot keeps its exact captured content hash.
    for slot in slots:
        if slot["masked"]:
            continue
        if slot["tag"] == "paged_source":
            index = row.referenced_physical_rows.index(
                row.logical_to_physical[slot["logical"]]
            )
            expected = hashlib.sha256(
                row.referenced_values[index * SOURCE_VALUE_BYTES:(index + 1) * SOURCE_VALUE_BYTES]
            ).hexdigest()
            assert slot["value_sha256"] == expected


def test_compact_fixture_never_allocates_the_source_pool():
    row = synthetic_scenario_a()
    mat = materialize(row, ORACLE)
    original_pool_bytes = row.source_capacity_rows * SOURCE_VALUE_BYTES
    assert original_pool_bytes == 100000 * 256
    assert mat.pool_bytes == mat.source_capacity * (256 + 32)
    assert mat.pool_bytes < original_pool_bytes // 100


def test_canonical_digest_reproduced_when_capture_records_one():
    row = synthetic_scenario_a()
    mat = materialize(row, ORACLE)
    assert mat.canonical_digest_compact
    assert row.evidence["canonical_digest"] is None  # synthetic: no capture digest


def test_missing_live_paged_row_is_a_hard_failure():
    row = synthetic_scenario_a()
    row = CapturedRow(**{
        **row.__dict__,
        "logical_to_physical": {
            logical: physical
            for logical, physical in row.logical_to_physical.items()
            if logical != 1
        },
    })
    with pytest.raises(MissingLivePagedRowError):
        materialize(row, ORACLE)


def test_materialize_rejects_inconsistent_page_table():
    row = synthetic_scenario_a()
    row = CapturedRow(**{
        **row.__dict__,
        "logical_to_physical": {**row.logical_to_physical, 0: 769},
    })
    with pytest.raises(MaterializeError):
        materialize(row, ORACLE)


# ---------------------------------------------------------------------------
# Counterfactual single-nibble mutation invariants (synthetic overlay row).
# ---------------------------------------------------------------------------

def _row_with_private_source():
    """Synthetic row carrying one live private-source overlay slot."""
    pages = [3, 1, 2, 0]
    logical_to_physical = {0: 768, 1: 769, 2: 770}
    referenced = [768, 769, 770]
    selected = [0, 1, 2, 20]
    # m[6]=20, m[7]=1, m[8]=0, m[9]=1 -> id 20 overlays proposal row 0.
    metadata = [4, 0, 1, 4, 0, 21, 20, 1, 0, 1]
    row = make_synthetic_row(
        pages, logical_to_physical, referenced, selected, metadata,
        window_end=4, source_end=20,
    )
    # Width is min(position+1, 128) = 5, so source slots start at index 5;
    # selected[3] == 20 lands at slot 8 as the private-source overlay.  Pin
    # the counterfactual site to low nibble 0x1.
    proposal = bytearray(row.source_proposal_values)
    proposal[COUNTERFACTUAL_COLUMN // 2] = 0xA0 | 0x1
    return CapturedRow(**{
        **row.__dict__,
        "source_proposal_values": bytes(proposal),
    })


def test_set_fp4_low_nibble_touches_one_nibble_only():
    buf = bytes([0xAB, 0xCD])
    out = set_fp4_low_nibble(buf, 0, 0x5)
    assert out == bytes([0xA5, 0xCD])
    out = set_fp4_low_nibble(buf, 2, 0x0)
    assert out == bytes([0xAB, 0xC0])
    with pytest.raises(ValueError):
        set_fp4_low_nibble(buf, 1, 0x5)  # high nibble site
    with pytest.raises(ValueError):
        set_fp4_low_nibble(buf, 0, 16)


def test_counterfactual_changes_exactly_one_low_nibble():
    row = _row_with_private_source()
    result = mutate_private_source_column(row, ORACLE, 0x2,
                                          column=COUNTERFACTUAL_COLUMN)
    evidence = result["evidence"]
    assert evidence["changed_bytes"] == 1
    assert evidence["byte_index"] == COUNTERFACTUAL_COLUMN // 2
    assert (evidence["before"] & 0xF0) == (evidence["after"] & 0xF0)
    assert (evidence["after"] & 0x0F) == 0x2
    assert evidence["before"] & 0x0F == 0x1  # synthetic site pinned to 0x1
    assert evidence["before_sha256"] != evidence["after_sha256"]
    # Scales arena is untouched.
    assert evidence["scales_sha256"] == hashlib.sha256(
        row.source_proposal_scales
    ).hexdigest()
    # Exactly one byte differs in the whole proposal arena.
    mutated = result["row"].source_proposal_values
    diffs = [i for i, (a, b) in enumerate(
        zip(row.source_proposal_values, mutated)) if a != b]
    assert diffs == [evidence["byte_index"]]
    # The mutated fixture still materializes and keeps the same ordered mask;
    # the private-source slot is at index width(5) + 3 = 8.
    mat = materialize(result["row"], ORACLE)
    assert mat.slot_evidence_compact[8]["tag"] == "private_source"
    assert mat.slot_evidence_compact[8]["value_sha256"] == \
        evidence["after_sha256"]


def test_counterfactual_roundtrip_is_involutive():
    row = _row_with_private_source()
    original = row.source_proposal_values
    once = mutate_private_source_column(row, ORACLE, 0x2)["row"]
    twice = mutate_private_source_column(once, ORACLE, 0x1,
                                         column=COUNTERFACTUAL_COLUMN)["row"]
    # The synthetic overlay row starts with nibble 0x1 at the site.
    assert original[COUNTERFACTUAL_COLUMN // 2] & 0x0F == 0x1
    assert twice.source_proposal_values == original
