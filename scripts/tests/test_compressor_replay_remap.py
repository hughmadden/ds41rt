"""CPU tests for descriptor remapping and the group-18 mutation geometry.

These exercise the pure byte bookkeeping behind the duplicate/mixed/swapped
counterfactual rows: a compact pending remap must address exactly the same FP32
operand bytes as the captured predecessor, and the group-18 column set must be
288..303 (16 columns), never word offsets 18..33.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from compressor_replay import experiments, remap  # noqa: E402
from compressor_replay.schema import DESCRIPTOR_SENTINEL  # noqa: E402

LATENT_ROW = 512 * 4


def _plane(marker: int, slots: int) -> bytes:
    return b"".join(bytes([(marker + slot) & 0xFF]) * LATENT_ROW
                    for slot in range(slots))


def test_group18_columns_are_288_to_303():
    assert experiments.MUTATION_GROUP == 18
    assert experiments.MUTATION_COLS == tuple(range(288, 304))
    regions = experiments.pack_affected_regions(experiments.MUTATION_COLS)
    # 2 FP4 values per value byte, 16 columns per E4M3 scale byte.
    assert regions["value_bytes"] == list(range(144, 152))
    assert regions["scale_bytes"] == [18]


def test_mutate_columns_flips_only_the_requested_bf16_elements():
    row_bytes = 8
    blob = b"\x00" * row_bytes
    mutated, evidence = experiments.mutate_columns(
        blob, row_bytes=row_bytes, row=0, cols=[1, 3], xor_byte=0x11
    )
    assert mutated[2] == 0x11 and mutated[3] == 0x01
    assert mutated[6] == 0x11 and mutated[7] == 0x01
    assert mutated[0] == 0x00 and mutated[1] == 0x00
    assert evidence["affected_pack_regions"]["scale_bytes"] == [0]
    assert evidence["mutated_sha256"] != evidence["original_sha256"]


def test_wave_remap_uses_slot_count_prefix():
    slot_count = 4
    remapped = remap.remap_batch(
        [DESCRIPTOR_SENTINEL, slot_count], [0, 1],
        original_slot_count=slot_count, row_map={0: 0, 1: 1},
        pending_kv=b"", pending_scores=b"", strategy="wave",
    )
    # Earlier-wave descriptor is slot_count + target, never the bare target
    # (which would be read as a pending slot).
    assert remapped.descriptors == (DESCRIPTOR_SENTINEL, slot_count)
    assert remapped.slot_count == slot_count


def test_compact_remap_preserves_earlier_wave_operand_bytes():
    slot_count = 4
    pending_kv = _plane(0x20, slot_count)
    pending_scores = _plane(0x60, slot_count)
    proposed = bytes([0xA1]) * LATENT_ROW
    scores = bytes([0xB2]) * LATENT_ROW

    batch = remap.remap_batch(
        [DESCRIPTOR_SENTINEL, slot_count], [0, 1],
        original_slot_count=slot_count, row_map={0: 0, 1: 1},
        pending_kv=pending_kv, pending_scores=pending_scores,
        strategy="compact_pending", compact_slots={0: 0},
    )
    assert batch.descriptors == (DESCRIPTOR_SENTINEL, 0)
    assert batch.slot_count == 1
    kv, sc = remap.build_compact_planes(
        batch.source_map, pending_kv=pending_kv, pending_scores=pending_scores,
        wave_kv_rows={0: proposed}, wave_scores_rows={0: scores},
        original_slot_count=slot_count,
    )
    assert kv == proposed
    assert sc == scores


def test_compact_remap_preserves_mixed_pending_and_wave_operands():
    slot_count = 4
    pending_kv = _plane(0x20, slot_count)
    pending_scores = _plane(0x60, slot_count)
    wave_kv = bytes([0x77]) * LATENT_ROW
    wave_scores = bytes([0x88]) * LATENT_ROW
    # row 0 pools pending slot 0; row 1 pools earlier wave row 0.
    batch = remap.remap_batch(
        [0, slot_count], [0, 1],
        original_slot_count=slot_count, row_map={0: 0, 1: 1},
        pending_kv=pending_kv, pending_scores=pending_scores,
        strategy="compact_pending", compact_slots={0: 2},
    )
    assert batch.descriptors == (0, 1)
    kv, sc = remap.build_compact_planes(
        batch.source_map, pending_kv=pending_kv, pending_scores=pending_scores,
        wave_kv_rows={2: wave_kv}, wave_scores_rows={2: wave_scores},
        original_slot_count=slot_count,
    )
    # source slot 0 is pending slot 0's row; source slot 1 is wave row 0.
    assert kv[:LATENT_ROW] == pending_kv[:LATENT_ROW]
    assert kv[LATENT_ROW:] == wave_kv
    assert sc[:LATENT_ROW] == pending_scores[:LATENT_ROW]
    assert sc[LATENT_ROW:] == wave_scores


def test_compact_remap_rejects_missing_wave_content_source():
    import pytest

    with pytest.raises(remap.RemapError):
        remap.remap_batch(
            [DESCRIPTOR_SENTINEL, 4], [0, 1],
            original_slot_count=4, row_map={0: 0, 1: 1},
            pending_kv=b"", pending_scores=b"", strategy="compact_pending",
        )


def test_compact_remap_preserves_swapped_row_operands():
    slot_count = 4
    pending_kv = _plane(0x20, slot_count)
    pending_scores = _plane(0x60, slot_count)
    proposed = bytes([0xC3]) * LATENT_ROW
    scores = bytes([0xD4]) * LATENT_ROW
    # New order [row1, row0]: row1's earlier-wave predecessor (row0) cannot
    # point forward in a wave remap, so it is preserved in a compact slot.
    batch = remap.remap_batch(
        [slot_count, DESCRIPTOR_SENTINEL], [1, 0],
        original_slot_count=slot_count, row_map={0: 1, 1: 0},
        pending_kv=pending_kv, pending_scores=pending_scores,
        strategy="compact_pending", compact_slots={0: 0},
    )
    assert batch.descriptors == (0, DESCRIPTOR_SENTINEL)
    kv, sc = remap.build_compact_planes(
        batch.source_map, pending_kv=pending_kv, pending_scores=pending_scores,
        wave_kv_rows={0: proposed}, wave_scores_rows={0: scores},
        original_slot_count=slot_count,
    )
    assert kv == proposed
    assert sc == scores


def test_compact_remap_deduplicates_duplicate_pending_rows():
    slot_count = 4
    pending_kv = _plane(0x20, slot_count)
    pending_scores = _plane(0x60, slot_count)
    # Duplicate rows both pool pending slot 0: one compact slot, same bytes.
    batch = remap.remap_batch(
        [0, 0], [0, 0],
        original_slot_count=slot_count, row_map={0: 0, 1: 1},
        pending_kv=pending_kv, pending_scores=pending_scores,
        strategy="compact_pending",
    )
    assert batch.descriptors == (0, 0)
    assert batch.slot_count == 1
    kv, sc = remap.build_compact_planes(
        batch.source_map, pending_kv=pending_kv, pending_scores=pending_scores,
        wave_kv_rows={}, wave_scores_rows={},
        original_slot_count=slot_count,
    )
    assert kv == pending_kv[:LATENT_ROW]
    assert sc == pending_scores[:LATENT_ROW]
