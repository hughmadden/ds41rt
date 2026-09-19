"""Real-capture CPU tests for the sparse replay harness.

Requires the cap16 p41/layer2 diagnostic captures (read-only).  Explicit CPU
target: no torch, no GPU, no dependency installation.  The capture directory
can be overridden with SPARSE_REPLAY_ACTIVATIONS; every test skips cleanly
when the captures are absent.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from sparse_replay import pins  # noqa: E402
from sparse_replay.capture import (  # noqa: E402
    load_activations,
    load_capture_dir,
    QUERY_ROW_BYTES,
    SINK_BYTES,
)
from sparse_replay.cases import (  # noqa: E402
    REF,
    CANDIDATE_LANE0,
    CANDIDATE_LANE0_ROW1,
    plan_case,
    standard_cases,
)
from sparse_replay.materialize import (  # noqa: E402
    COUNTERFACTUAL_COLUMN,
    SENTINEL_PAGE,
    materialize,
    mutate_private_source_column,
)
from sparse_replay.runner import VIEW_BYTES  # noqa: E402

ACTIVATIONS = Path(
    os.environ.get(
        "SPARSE_REPLAY_ACTIVATIONS",
        "/home/turq/.cache/afd-dsh-workers-20260919/"
        "flash-cap16-p2-inputs-diagnostic-01/activations",
    )
)

capture_required = pytest.mark.skipif(
    not ACTIVATIONS.is_dir(),
    reason=f"captures not present at {ACTIVATIONS}",
)

REFERENCE_CANONICAL_DIGEST = (
    "b14c07883d847e82d817eef14630e431bb19a61dbf1cde5bee3a446ac03a3254"
)
REFERENCE_SLOT62_VALUE_SHA = (
    "564defa666e07c9bb7b93d8212a239be80e9c3269f15a300fbff5ffc5144bb17"
)
CANDIDATE_SLOT62_VALUE_SHA = (
    "94d0c8a0ced818199cb296a83128a8eff6967ec3f622c662ebc7997509e2b313"
)


@pytest.fixture(scope="module")
def extractor():
    return pins.load_extractor()


@pytest.fixture(scope="module")
def oracle():
    return pins.load_oracle()


@pytest.fixture(scope="module")
def index(extractor):
    return load_activations(ACTIVATIONS, extractor)


# ---------------------------------------------------------------------------
# Raw-byte exact load, including BF16.
# ---------------------------------------------------------------------------

@capture_required
def test_query_row_is_raw_byte_exact_against_the_typed_file(index):
    row = index[REF]
    on_disk = row.manifest_path.parent / "layer2-attention-query.bin"
    blob = on_disk.read_bytes()
    assert len(blob) == QUERY_ROW_BYTES
    assert row.query_row == blob  # byte-for-byte, no float cast
    assert hashlib.sha256(row.query_row).hexdigest() == \
        row.evidence["query_sha256"]
    assert any(row.query_row)  # raw bytes, never a zero-filled fallback


@capture_required
def test_sink_is_shared_and_byte_exact(index):
    row = index[REF]
    assert len(row.sink) == SINK_BYTES
    for other in index.values():
        assert other.sink == row.sink


@capture_required
def test_captured_output_row_is_raw_byte_exact(index):
    row = index[REF]
    on_disk = row.manifest_path.parent / "layer2-attention-values.bin"
    blob = on_disk.read_bytes()
    flat = row.flat_row
    assert row.output_row == blob[flat * QUERY_ROW_BYTES:(flat + 1) * QUERY_ROW_BYTES]
    assert hashlib.sha256(row.output_row).hexdigest() == row.output_sha256


# ---------------------------------------------------------------------------
# Request identity / metadata offsets.
# ---------------------------------------------------------------------------

@capture_required
def test_reference_request_identity_and_metadata_offsets(index):
    row = index[REF]
    assert row.request_id == 2026091950
    assert row.position == 41
    assert row.metadata10[3] == row.position  # metadata position column
    assert row.evidence["position_matches_metadata"] is True
    assert row.evidence["flat_row"] == row.flat_row == 0
    assert row.evidence["canonical_digest"] == REFERENCE_CANONICAL_DIGEST
    assert row.launch_kind == "single_per_request"
    assert row.width == 0  # per-row auto width; the harness never changes it


@capture_required
def test_candidate_metadata_matches_reference_layout(index):
    ref = index[REF]
    candidate = index[CANDIDATE_LANE0]
    assert candidate.launch_kind == "descriptor_batch"
    same = [0, 2, 3, 4, 5, 6, 7, 9]
    for col in same:
        assert candidate.metadata10[col] == ref.metadata10[col], col
    # 1960/1962 share the reference overlay base; 1961/1963 differ by the
    # window/overlay base shift the root analysis already documented.
    shifted = index[CANDIDATE_LANE0_ROW1]
    assert shifted.metadata10[1] == 1 and shifted.metadata10[8] == 1
    assert candidate.metadata10[1] == 0 and candidate.metadata10[8] == 0


@capture_required
def test_extracted_evidence_matches_root_analysis(index):
    ref = index[REF]
    slot62 = ref.evidence["ordered_keyslots"][62]
    assert slot62["tag"] == "private_source"
    assert slot62["logical"] == 20
    assert slot62["value_sha256"] == REFERENCE_SLOT62_VALUE_SHA
    candidate = index[CANDIDATE_LANE0]
    cand62 = candidate.evidence["ordered_keyslots"][62]
    assert cand62["value_sha256"] == CANDIDATE_SLOT62_VALUE_SHA
    assert cand62["scale_sha256"] == slot62["scale_sha256"]


# ---------------------------------------------------------------------------
# Two distinct rows / batch anti-fake properties.
# ---------------------------------------------------------------------------

@capture_required
def test_reference_and_candidate_are_distinct_rows(index):
    ref = index[REF]
    candidate = index[CANDIDATE_LANE0]
    assert ref.key != candidate.key
    assert ref.metadata10 == candidate.metadata10  # same layout class
    diff = sum(1 for a, b in zip(ref.output_row, candidate.output_row) if a != b)
    assert 0 < diff <= QUERY_ROW_BYTES
    assert hashlib.sha256(ref.output_row).digest() != \
        hashlib.sha256(candidate.output_row).digest()


@capture_required
def test_batch_plan_expected_outputs_cannot_be_faked_by_duplicating_reference(index):
    """Each mixed-batch row names its own expected output hash.

    A harness that duplicated the reference output for every batch row would
    fail the candidate row outright, because the captured candidate output
    differs from the captured reference output byte-wise (checked above).
    The as-captured batches genuinely produce identical outputs for both
    rows, so there the anti-fake guarantee comes from the differing row
    inputs (metadata / pages), which the CPU materializer verifies.
    """
    cases = {case.name: case for case in standard_cases()}
    for name in ("batch_mixed_reference_candidate",
                 "batch_swapped_candidate_reference"):
        plan = plan_case(cases[name], index, repeats=3)
        assert len(plan["rows"]) == 2
        hashes = [r["expected_output_sha256"] for r in plan["rows"]]
        assert hashes[0] != hashes[1], name
        digests = [r["canonical_digest"] for r in plan["rows"]]
        assert digests[0] != digests[1], name
    for name in ("batch_as_captured_lane0", "batch_as_captured_lane1"):
        plan = plan_case(cases[name], index, repeats=3)
        hashes = [r["expected_output_sha256"] for r in plan["rows"]]
        assert hashes[0] == hashes[1]  # all four candidate outputs are equal
        metas = [tuple(r["actual_metadata10"]) for r in plan["rows"]]
        assert metas[0] != metas[1]  # but the row inputs are not identical
    # The mixed batch order is explicit and not the swapped order.
    mixed = plan_case(cases["batch_mixed_reference_candidate"], index, 3)
    assert mixed["rows"][0]["request_id"] == REF[1]
    assert mixed["rows"][1]["request_id"] == CANDIDATE_LANE0[1]
    swapped = plan_case(cases["batch_swapped_candidate_reference"], index, 3)
    assert swapped["rows"][0]["request_id"] == CANDIDATE_LANE0[1]
    assert swapped["rows"][1]["request_id"] == REF[1]


# ---------------------------------------------------------------------------
# Materialization of the real captures.
# ---------------------------------------------------------------------------

@capture_required
def test_real_captures_materialize_compactly(index, oracle):
    for key, row in sorted(index.items()):
        mat = materialize(row, oracle)
        assert mat.source_capacity == row.referenced_count == 20
        assert mat.pool_bytes == 20 * (256 + 32) == 5760
        original_pool = row.source_capacity_rows * 256
        assert mat.pool_bytes < original_pool // 1000
        assert mat.canonical_digest_compact == row.evidence["canonical_digest"]
        untouched = [p for p in mat.pages_compact if p == SENTINEL_PAGE]
        assert len(untouched) == row.page_stride - len(mat.page_map)


@capture_required
def test_counterfactual_nibble_lands_on_captured_opposite_hashes(index, oracle):
    candidate = index[CANDIDATE_LANE0]
    reference = index[REF]
    to_ref = mutate_private_source_column(candidate, oracle, 0x1,
                                          column=COUNTERFACTUAL_COLUMN)
    assert to_ref["evidence"]["changed_bytes"] == 1
    assert to_ref["evidence"]["byte_index"] == 147
    assert (to_ref["evidence"]["before"] & 0xF0) == \
           (to_ref["evidence"]["after"] & 0xF0)
    assert to_ref["evidence"]["after_sha256"] == REFERENCE_SLOT62_VALUE_SHA
    assert to_ref["evidence"]["scales_sha256"] == \
        reference.evidence["ordered_keyslots"][62]["scale_sha256"]

    to_cand = mutate_private_source_column(reference, oracle, 0x2,
                                           column=COUNTERFACTUAL_COLUMN)
    assert to_cand["evidence"]["after_sha256"] == CANDIDATE_SLOT62_VALUE_SHA
    assert to_cand["evidence"]["changed_bytes"] == 1
    # Only one byte differs across the whole proposal arena, both directions.
    for original, mutated in (
        (candidate.source_proposal_values,
         to_ref["row"].source_proposal_values),
        (reference.source_proposal_values,
         to_cand["row"].source_proposal_values),
    ):
        diffs = [i for i, (a, b) in enumerate(zip(original, mutated)) if a != b]
        assert diffs == [147]


# ---------------------------------------------------------------------------
# Failure modes: truncated capture / missing live paged row.
# ---------------------------------------------------------------------------

def _copy_ref_wave(tmp_path: Path) -> Path:
    src = ACTIVATIONS / REF[0]
    dst = tmp_path / REF[0]
    dst.mkdir(parents=True)
    for path in src.iterdir():
        if path.name.startswith("layer2-attention") or path.name in (
            "positions.json", "members.json"
        ):
            shutil.copy2(path, dst / path.name)
    return dst


@capture_required
def test_truncated_capture_fails(tmp_path, extractor):
    dst = _copy_ref_wave(tmp_path)
    query = dst / "layer2-attention-query.bin"
    query.write_bytes(query.read_bytes()[:-1])
    with pytest.raises(extractor.TraceError):
        load_capture_dir(dst / "layer2-attention-inputs.json", REF[1], extractor)


@capture_required
def test_missing_live_paged_row_fails(tmp_path, extractor, oracle):
    dst = _copy_ref_wave(tmp_path)
    manifest_path = dst / "layer2-attention-inputs.json"
    manifest = json.loads(manifest_path.read_bytes())
    referenced = manifest["requests"][0]["source"]["referenced"]
    referenced["logical_to_physical"] = [
        pair for pair in referenced["logical_to_physical"]
        if pair["logical"] != 0
    ]
    manifest_path.write_text(json.dumps(manifest))
    from sparse_replay.materialize import MissingLivePagedRowError
    # The pinned extractor rejects the row during load; if a capture ever
    # reached the materializer without that check, the materializer raises
    # MissingLivePagedRowError itself.  Either stage is a hard failure.
    with pytest.raises((MissingLivePagedRowError, extractor.TraceError)):
        row = load_capture_dir(manifest_path, REF[1], extractor)
        materialize(row, oracle)


# ---------------------------------------------------------------------------
# CLI: default CPU validate/plan mode.
# ---------------------------------------------------------------------------

@capture_required
def test_cli_default_mode_validates_and_plans(tmp_path):
    plan_out = tmp_path / "plan.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "replay-ds41-sparse-actual.py"),
         "--activations", str(ACTIVATIONS), "--plan-out", str(plan_out)],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["mode"] == "validate/plan (CPU only)"
    assert len(plan["cases"]) == 11
    assert all(v["canonical_digest_matches_capture"]
               for v in plan["materialized"].values())
    cf = plan["counterfactual_variants"]
    assert cf["lane0-batch81-Full-2rows:2026091960:candidate_to_reference"]["after_sha256"] == \
        REFERENCE_SLOT62_VALUE_SHA
    assert plan_out.is_file()


@capture_required
def test_cli_execute_requires_output_dir(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "replay-ds41-sparse-actual.py"),
         "--activations", str(ACTIVATIONS), "--execute"],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 2


@capture_required
def test_cli_rejects_wrong_extractor_sha(tmp_path):
    fake = tmp_path / "extractor.py"
    fake.write_text("# not the pinned extractor\n")
    result = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "replay-ds41-sparse-actual.py"),
         "--activations", str(ACTIVATIONS), "--extractor", str(fake),
         "--extractor-sha", "0" * 64],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 1
    assert "sha256 mismatch" in result.stderr


def test_view_struct_is_120_bytes():
    assert VIEW_BYTES == 120
