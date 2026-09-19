"""Tests for the replacement compressor row-case planners (families C/D).

Synthetic tests run against a tiny in-memory ``FakeCapture`` that satisfies
the capture API (``read_buffer``/``read_weight``/``device_descriptors``/
``slot_count``/``rows``/``wave_rows``/``chunks``) with DISTINCT current,
pending and frequency bytes per request and two odd p41 rows, so every
addressing or slicing mistake is detectable byte-for-byte.  The actual
immutable captures are loaded with the shared, corrected
``schema.load_capture``; there is no private capture adapter left in
``row_cases``.  Structural checks go through the shared
``validation.validate_experiment``; ``RowCaseError`` remains only for row
ownership / provenance mistakes.

CPU only: no GPU executor, no fake CUDA, no whole checkpoint or shared pool
load.  Everything here validates *plans* (roles, bytes, provenance, gates).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compressor_replay import row_cases as rc            # noqa: E402
from compressor_replay import schema                      # noqa: E402
from compressor_replay.validation import (               # noqa: E402
    ValidationError,
    validate_experiment,
)
from compressor_replay.runner import PlannedExperiment, PlannedOperand  # noqa: E402
from compressor_replay.schema import DESCRIPTOR_SENTINEL  # noqa: E402

ACTIVATIONS = Path(
    "/home/turq/.cache/afd-dsh-workers-20260919/"
    "flash-cap16-p2-compressor-diagnostic-01/activations"
)
REFERENCE_DIR = "lane0-batch20-Full-1rows"
CANDIDATE_DIRS = ("lane0-batch81-Full-2rows", "lane1-batch82-Full-2rows")

LATENT = 512
SOURCE = 5120
SLOTS = 4
PENDING_ROW = LATENT * 4          # FP32 [512]
FREQ_ROW = 32 * 2 * 4             # FP32 [32, 2]
INPUT_ROW = SOURCE * 2            # BF16 [5120]
NORM = LATENT * 2                 # BF16 [512]
PROJ_WEIGHT = LATENT * SOURCE * 2

# FP32 signalling-NaN / infinity / denormal patterns: raw bit patterns that a
# numeric cast would destroy.  They must pass through plans untouched.
ODD_BIT_PATTERNS = b"\x7f\x80\x00\x01" + b"\x7f\x80\x00\x00" + \
    b"\x00\x00\x80\x7f" + b"\x00\x00\x80\x00"


def tagged(tag: str, size: int) -> bytes:
    """Deterministic distinct bytes: the tag survives at both ends."""
    unit = f"[{tag}]".encode()
    body = (unit * (size // len(unit) + 1))[:size]
    return ODD_BIT_PATTERNS + body[len(ODD_BIT_PATTERNS):]


def u64(values) -> bytes:
    return b"".join(int(v).to_bytes(8, "little") for v in values)


def row(blob: bytes, index: int, row_bytes: int) -> bytes:
    return blob[index * row_bytes:(index + 1) * row_bytes]


SHARED_WEIGHTS = {
    "wkv": tagged("WKV", PROJ_WEIGHT),
    "wgate": tagged("WGATE", PROJ_WEIGHT),
    "norm": tagged("NORM", NORM),
}
P41_FREQ = tagged("FQ-p41", FREQ_ROW)


class FakeCapture:
    """Minimal capture API over in-memory bytes (distinct per request)."""

    layer = 2
    ratio = 2

    def __init__(self, name, *, rows, slot_count, buffers, weights,
                 device_descriptors, wave_rows, chunks):
        self.directory = Path(f"/fake/{name}")
        self.rows = rows
        self.slot_count = slot_count
        self.buffers = buffers
        self.weights = weights
        self.device_descriptors = tuple(device_descriptors)
        self.wave_rows = tuple(wave_rows)
        self.chunks = tuple(chunks)

    def read_buffer(self, name):
        return self.buffers[name]

    def read_weight(self, role):
        return self.weights[role]


def make_capture(name, *, rows=2, descs, leases, positions, chunk_of_row,
                 freq_rows=None, weights=None, request_base=9100,
                 latent_first_tokens=None):
    """Build a FakeCapture whose geometry is given explicitly.

    ``chunk_of_row[r]`` assigns row r to a chunk index; each chunk is a
    separate single-position request unless two rows share one chunk (the
    same-request wave geometry).  ``latent_first_tokens`` defaults to
    position-1 for odd rows.
    """
    chunk_ids = sorted(set(chunk_of_row))
    chunks = []
    for chunk_id in chunk_ids:
        member_rows = [r for r in range(rows) if chunk_of_row[r] == chunk_id]
        first = min(member_rows)
        chunks.append({
            "index": chunk_id,
            "request_id": request_base + chunk_id,
            "lease": {"slot": leases[first], "generation": 1},
            "version": 19,
            "position": positions[first],
            "tokens": len(member_rows),
            "prepared_offset": first,
            "flattened_row_range": [first, first + len(member_rows)],
            "absolute_positions": [positions[r] for r in member_rows],
        })
    latent_first_tokens = latent_first_tokens or {}
    wave_rows = []
    for r in range(rows):
        position = positions[r]
        descriptor = descs[r]
        if position % 2 == 1:
            first_token = latent_first_tokens.get(r, position - 1)
            latent = {
                "source_row": r,
                "first_token": first_token,
                "logical_compressed_row": first_token // 2,
                "request_id": request_base + chunk_of_row[r],
            }
        else:
            latent = None
        if descriptor == DESCRIPTOR_SENTINEL:
            predecessor = {"kind": "invalid_sentinel"}
        elif descriptor < SLOTS:
            predecessor = {"kind": "pending_slot", "slot": descriptor}
        else:
            predecessor = {"kind": "earlier_wave_row",
                           "wave_row": descriptor - SLOTS}
        wave_rows.append({
            "row": r, "chunk": chunk_of_row[r],
            "request_id": request_base + chunk_of_row[r],
            "absolute_position": position,
            "device_descriptor": descriptor,
            "predecessor": predecessor,
            "completed_latent": latent,
        })

    if freq_rows is None:
        freq_rows = [tagged(f"FQ-{name}-{r}", FREQ_ROW) for r in range(rows)]
    freq = freq_rows
    buffers = {
        "input": b"".join(tagged(f"IN-{name}-{r}", INPUT_ROW)
                          for r in range(rows)),
        "projected": b"".join(tagged(f"PR-{name}-{r}", PENDING_ROW)
                              for r in range(rows)),
        "scores": b"".join(tagged(f"SC-{name}-{r}", PENDING_ROW)
                           for r in range(rows)),
        "output": b"".join(tagged(f"OUT-{name}-{r}", LATENT * 2)
                           for r in range(rows)),
        "kv-values": b"".join(tagged(f"KV-{name}-{r}", 256) for r in range(rows)),
        "kv-scales": b"".join(tagged(f"KS-{name}-{r}", 32) for r in range(rows)),
        "frequencies": b"".join(freq),
        "positions": u64(positions),
        "pending-kv": b"".join(tagged(f"PK-{name}-{s}", PENDING_ROW)
                               for s in range(SLOTS)),
        "pending-scores": b"".join(tagged(f"PS-{name}-{s}", PENDING_ROW)
                                   for s in range(SLOTS)),
    }
    return FakeCapture(name, rows=rows, slot_count=SLOTS, buffers=buffers,
                       weights=weights or SHARED_WEIGHTS,
                       device_descriptors=descs, wave_rows=wave_rows,
                       chunks=chunks)


def two_request_capture(name, *, descs=(1, 2), leases=None, freq="shared",
                        weights=None):
    """The actual candidate geometry: two independent p41 requests."""
    leases = list(descs) if leases is None else list(leases)
    if freq == "shared":
        freq_rows = [P41_FREQ] * 2
    else:
        freq_rows = None       # distinct per request
    return make_capture(name, rows=2, descs=list(descs), leases=leases,
                        positions=[41, 41], chunk_of_row=[0, 1],
                        freq_rows=freq_rows, weights=weights)


def reference_capture(*, freq="shared", weights=None):
    """The actual reference geometry: ONE row, p41, its own pending slot 0."""
    freq_rows = [P41_FREQ] if freq == "shared" else None
    return make_capture("fake-ref-batch20-Full-1rows", rows=1, descs=[0],
                        leases=[0], positions=[41], chunk_of_row=[0],
                        freq_rows=freq_rows, weights=weights,
                        request_base=9200)


def operand(experiment, role):
    matches = [o for o in experiment.operands if o.role == role]
    assert len(matches) == 1, f"role {role!r} appears {len(matches)} times"
    return matches[0]


def by_name(experiments, suffix):
    matches = [e for e in experiments if e.name.endswith(suffix)]
    assert len(matches) == 1, f"{suffix}: {len(matches)} matches"
    return matches[0]


def assert_well_formed(experiments):
    names = [e.name for e in experiments]
    assert len(names) == len(set(names)), "experiment name alias collision"
    for experiment in experiments:
        validate_experiment(experiment)             # shared validator
        for op in experiment.operands:
            assert isinstance(op.blob, bytes)
        for stage in experiment.stages:
            assert stage["kind"] in ("project", "pool", "pack")


def notes_payload(experiment):
    for note in experiment.notes:
        if note.startswith("provenance:"):
            return json.loads(note[len("provenance:"):])
    raise AssertionError("no provenance note")


# ---------------------------------------------------------------------------
# Family C.
# ---------------------------------------------------------------------------

def plan_c_default():
    candidate = two_request_capture("fake-lane0-batch81-Full-2rows")
    reference = reference_capture()
    experiments = rc.plan_candidate_geometry(candidate, reference, tag="t")
    assert_well_formed(experiments)
    return candidate, reference, experiments


def test_c_case_set_is_all_counterfactual():
    _, _, experiments = plan_c_default()
    # 2 singles + 2 duplicates + 1 swapped + 2 mixed
    assert len(experiments) == 7
    for experiment in experiments:
        assert experiment.kind == "C"
        assert experiment.counterfactual is True
        assert experiment.expected == {}          # CF never gates
    assert sum(1 for e in experiments if e.name.endswith("_row0_single")) == 1
    assert sum(1 for e in experiments if e.name.endswith("_row1_single")) == 1
    assert sum(1 for e in experiments if "_duplicate" in e.name) == 2
    assert sum(1 for e in experiments if "_swapped_" in e.name) == 1
    assert sum(1 for e in experiments if "_mixed_" in e.name) == 2


def test_c_stage_wiring_projects_into_projected_and_scores():
    _, _, experiments = plan_c_default()
    for experiment in experiments:
        stages = experiment.stages
        assert [s["kind"] for s in stages] == ["project", "project", "pool",
                                               "pack"]
        assert stages[0]["weight"] == "wkv"
        assert stages[0]["output"] == "projected"
        assert stages[1]["weight"] == "wgate"
        assert stages[1]["output"] == "scores"
        assert stages[2]["kv"] == "projected"      # the B2/C bug fix
        assert stages[2]["scores"] == "scores"
        assert stages[3]["input"] == "output"
        assert set(experiment.outputs) == set(rc.C_OUTPUTS)


def test_c_no_stage_references_missing_kv_role():
    _, _, experiments = plan_c_default()
    for experiment in experiments:
        roles = {o.role for o in experiment.operands}
        assert "kv" not in roles
        for stage in experiment.stages:
            for value in stage.values():
                assert value != "kv"


def test_c_single_row_resolves_its_own_pending_slot():
    candidate, _, experiments = plan_c_default()
    single1 = by_name(experiments, "_row1_single")
    assert single1.rows == 1 and single1.slots == 1
    assert operand(single1, "predecessors").blob == u64([0])
    # descriptor 2 < slot_count: previous is the pending plane row 2,
    # byte-exact, no host substitution.
    plane = candidate.read_buffer("pending-kv")
    assert operand(single1, "pending_kv").blob == row(plane, 2, PENDING_ROW)
    scores_plane = candidate.read_buffer("pending-scores")
    assert operand(single1, "pending_scores").blob == row(scores_plane, 2,
                                                          PENDING_ROW)
    assert operand(single1, "input").blob == row(candidate.read_buffer("input"),
                                                 1, INPUT_ROW)
    assert operand(single1, "frequencies").blob == row(
        candidate.read_buffer("frequencies"), 1, FREQ_ROW)


def test_c_row_zero_is_not_token_forty():
    """A descriptor is an address: row 0 at p41 pools its own pending slot."""
    candidate, _, experiments = plan_c_default()
    single0 = by_name(experiments, "_row0_single")
    # row0's descriptor is 1 (a pending slot), NOT the sentinel and NOT an
    # earlier-wave row: row 0 of a batch does not imply token 40.
    assert candidate.device_descriptors[0] == 1
    assert candidate.wave_rows[0]["absolute_position"] == 41
    previous = rc.resolve_row_previous(candidate, 0)
    assert previous.kind == "pending"
    assert previous.index == 1
    assert operand(single0, "pending_kv").blob == row(
        candidate.read_buffer("pending-kv"), 1, PENDING_ROW)


def test_c_duplicate_duplicates_entire_row_bundle():
    candidate, _, experiments = plan_c_default()
    dup0 = by_name(experiments, "_row0_duplicate")
    assert dup0.rows == 2 and dup0.slots == 2
    assert operand(dup0, "predecessors").blob == u64([0, 1])
    plane = candidate.read_buffer("pending-kv")
    # BOTH compact slots hold row 0's previous pending bytes (desc 1), not
    # just the current input duplicated.
    assert operand(dup0, "pending_kv").blob == row(plane, 1, PENDING_ROW) * 2
    assert operand(dup0, "input").blob == row(
        candidate.read_buffer("input"), 0, INPUT_ROW) * 2
    assert operand(dup0, "frequencies").blob == row(
        candidate.read_buffer("frequencies"), 0, FREQ_ROW) * 2


def test_c_swapped_permutes_entire_row_bundle():
    candidate, _, experiments = plan_c_default()
    swapped = by_name(experiments, "_swapped_10")
    assert operand(swapped, "predecessors").blob == u64([0, 1])
    plane = candidate.read_buffer("pending-kv")
    input_blob = candidate.read_buffer("input")
    freq = candidate.read_buffer("frequencies")
    assert operand(swapped, "pending_kv").blob == (
        row(plane, 2, PENDING_ROW) + row(plane, 1, PENDING_ROW))
    assert operand(swapped, "input").blob == (
        row(input_blob, 1, INPUT_ROW) + row(input_blob, 0, INPUT_ROW))
    assert operand(swapped, "frequencies").blob == (
        row(freq, 1, FREQ_ROW) + row(freq, 0, FREQ_ROW))


def test_c_mixed_pairs_bundles_from_both_sources():
    candidate, reference, experiments = plan_c_default()
    mixed = by_name(
        experiments,
        "_mixed_fake-ref-batch20-Full-1rows_row0_plus_row0")
    assert mixed.rows == 2 and mixed.slots == 2
    ref_plane = reference.read_buffer("pending-kv")
    cand_plane = candidate.read_buffer("pending-kv")
    assert operand(mixed, "pending_kv").blob == (
        row(ref_plane, 0, PENDING_ROW) + row(cand_plane, 1, PENDING_ROW))
    assert operand(mixed, "input").blob == (
        row(reference.read_buffer("input"), 0, INPUT_ROW)
        + row(candidate.read_buffer("input"), 0, INPUT_ROW))
    payload = notes_payload(mixed)
    sources = payload["current_sources"]
    assert sources[0]["capture"] == "fake-ref-batch20-Full-1rows"
    assert sources[1]["capture"] == "fake-lane0-batch81-Full-2rows"
    assert payload["previous_sources"][0]["kind"] == "pending"
    assert payload["previous_sources"][1]["kind"] == "pending"
    # weights are recorded and identical across both sources
    assert payload["weights"] == payload["reference_weights"]


def test_c_frequencies_slice_one_per_selected_row():
    candidate, _, experiments = plan_c_default()
    freq = candidate.read_buffer("frequencies")
    ref_freq = reference_capture().read_buffer("frequencies")
    expected = {
        "_row0_single": row(freq, 0, FREQ_ROW),
        "_row1_single": row(freq, 1, FREQ_ROW),
        "_row0_duplicate": row(freq, 0, FREQ_ROW) * 2,
        "_row1_duplicate": row(freq, 1, FREQ_ROW) * 2,
        "_swapped_10": row(freq, 1, FREQ_ROW) + row(freq, 0, FREQ_ROW),
        "_mixed_fake-ref-batch20-Full-1rows_row0_plus_row0":
            row(ref_freq, 0, FREQ_ROW) + row(freq, 0, FREQ_ROW),
        "_mixed_row0_plus_fake-ref-batch20-Full-1rows_row0":
            row(freq, 0, FREQ_ROW) + row(ref_freq, 0, FREQ_ROW),
    }
    for suffix, blob in expected.items():
        experiment = by_name(experiments, suffix)
        got = operand(experiment, "frequencies")
        assert got.blob == blob, suffix
        assert got.dtype == "float32" and got.shape == (experiment.rows, 32, 2)


def test_c_without_reference_skips_mixed_cases():
    candidate = two_request_capture("fake-lane0-batch81-Full-2rows")
    experiments = rc.plan_candidate_geometry(candidate, None, tag="t")
    assert_well_formed(experiments)
    assert len(experiments) == 5
    assert all("_mixed_" not in e.name for e in experiments)


def test_c_records_own_raw_hashes_and_provenance():
    _, _, experiments = plan_c_default()
    single = by_name(experiments, "_row1_single")
    payload = notes_payload(single)
    assert payload["input_sha256"] == rc._sha256(
        operand(single, "input").blob)
    assert payload["pending_kv_sha256"] == rc._sha256(
        operand(single, "pending_kv").blob)
    assert payload["pending_scores_sha256"] == rc._sha256(
        operand(single, "pending_scores").blob)
    assert payload["compact_descriptors"] == [0]
    assert payload["slots"] == 1
    assert payload["weights"]["wkv"] == rc._sha256(SHARED_WEIGHTS["wkv"])


def test_c_weight_mismatch_is_rejected():
    other_weights = dict(SHARED_WEIGHTS)
    other_weights["wgate"] = tagged("WGATE-OTHER", PROJ_WEIGHT)
    candidate = two_request_capture("fake-lane0-batch81-Full-2rows")
    reference = reference_capture(weights=other_weights)
    with pytest.raises(rc.RowCaseError, match="wgate"):
        rc.plan_candidate_geometry(candidate, reference, tag="t")


# ---------------------------------------------------------------------------
# Family D.
# ---------------------------------------------------------------------------

def plan_d_default(freq="shared"):
    reference = reference_capture(freq=freq)
    candidate = two_request_capture("fake-lane0-batch81-Full-2rows", freq=freq)
    experiments = rc.plan_cross_substitution(reference, candidate, tag="t")
    assert_well_formed(experiments)
    return reference, candidate, experiments


def test_d_exists_for_one_row_reference_and_two_row_candidate():
    """Regression: the draft family D returned [] for a 1-row reference."""
    reference, candidate, experiments = plan_d_default()
    assert reference.rows == 1 and candidate.rows == 2
    assert len(experiments) >= 12
    # four combinations for EACH candidate row, named per member
    for r in (0, 1):
        for suffix in ("curR_prevR", "curC_prevC", "curC_prevR",
                       "curR_prevC"):
            by_name(experiments, f"_row{r}_{suffix}")
    # names stay unique per candidate capture name / member / tag
    assert all("fake-lane0-batch81-Full-2rows" in e.name for e in experiments)


def test_d_originals_gate_on_captured_oracles():
    reference, candidate, experiments = plan_d_default()
    rr = by_name(experiments, "_row0_curR_prevR")
    assert rr.counterfactual is False
    assert rr.expected == {
        "output": row(reference.read_buffer("output"), 0, LATENT * 2),
        "values": row(reference.read_buffer("kv-values"), 0, 256),
        "scales": row(reference.read_buffer("kv-scales"), 0, 32),
    }
    for r in (0, 1):
        cc = by_name(experiments, f"_row{r}_curC_prevC")
        assert cc.counterfactual is False
        assert cc.expected == {
            "output": row(candidate.read_buffer("output"), r, LATENT * 2),
            "values": row(candidate.read_buffer("kv-values"), r, 256),
            "scales": row(candidate.read_buffer("kv-scales"), r, 32),
        }


def test_d_current_only_swap_addresses_reference_previous():
    reference, candidate, experiments = plan_d_default()
    cr = by_name(experiments, "_row1_curC_prevR")
    assert cr.counterfactual is True and cr.expected == {}
    # current: candidate row 1 captured projections
    assert operand(cr, "projected").blob == row(
        candidate.read_buffer("projected"), 1, PENDING_ROW)
    assert operand(cr, "scores").blob == row(
        candidate.read_buffer("scores"), 1, PENDING_ROW)
    # previous: the REFERENCE row's own pending slot 0 content
    assert operand(cr, "pending_kv").blob == row(
        reference.read_buffer("pending-kv"), 0, PENDING_ROW)
    assert operand(cr, "predecessors").blob == u64([0])
    assert cr.rows == 1 and cr.slots == 1


def test_d_previous_only_swap_addresses_candidate_previous():
    reference, candidate, experiments = plan_d_default()
    rc_swap = by_name(experiments, "_row1_curR_prevC")
    assert rc_swap.counterfactual is True and rc_swap.expected == {}
    assert operand(rc_swap, "projected").blob == row(
        reference.read_buffer("projected"), 0, PENDING_ROW)
    # previous: candidate row 1's own pending slot (descriptor 2)
    assert operand(rc_swap, "pending_kv").blob == row(
        candidate.read_buffer("pending-kv"), 2, PENDING_ROW)


def test_d_norm_and_freq_come_from_current_row():
    reference, candidate, experiments = plan_d_default()
    cr = by_name(experiments, "_row0_curC_prevR")
    assert operand(cr, "frequencies").blob == row(
        candidate.read_buffer("frequencies"), 0, FREQ_ROW)
    assert operand(cr, "norm").blob == SHARED_WEIGHTS["norm"]
    rc_swap = by_name(experiments, "_row0_curR_prevC")
    assert operand(rc_swap, "frequencies").blob == row(
        reference.read_buffer("frequencies"), 0, FREQ_ROW)


def test_d_bundles_gate_original_bytes_only():
    reference, candidate, experiments = plan_d_default()
    originals = by_name(experiments, "_bundle_cand_originals")
    assert originals.counterfactual is False
    assert originals.expected["output"] == candidate.read_buffer("output")
    assert originals.expected["values"] == candidate.read_buffer("kv-values")
    assert originals.expected["scales"] == candidate.read_buffer("kv-scales")
    assert operand(originals, "predecessors").blob == u64([0, 1])
    # each output row has its own compact pending slot with its own content
    plane = candidate.read_buffer("pending-kv")
    assert operand(originals, "pending_kv").blob == (
        row(plane, 1, PENDING_ROW) + row(plane, 2, PENDING_ROW))

    duplicate = by_name(experiments, "_bundle_duplicate_ref_original")
    assert duplicate.counterfactual is False
    ref_out = row(reference.read_buffer("output"), 0, LATENT * 2)
    assert duplicate.expected["output"] == ref_out * 2
    assert operand(duplicate, "pending_kv").blob == (
        row(reference.read_buffer("pending-kv"), 0, PENDING_ROW) * 2)

    swapped = by_name(experiments, "_bundle_row0_swapped_originals")
    assert swapped.counterfactual is False
    assert swapped.expected["output"] == (
        row(candidate.read_buffer("output"), 0, LATENT * 2)
        + row(reference.read_buffer("output"), 0, LATENT * 2))

    mixed = by_name(experiments, "_bundle_row0_original_plus_prevswap")
    assert mixed.counterfactual is True and mixed.expected == {}
    assert operand(mixed, "pending_kv").blob == (
        row(reference.read_buffer("pending-kv"), 0, PENDING_ROW)
        + row(candidate.read_buffer("pending-kv"), 1, PENDING_ROW))


def test_d_p41_frequency_mismatch_is_rejected():
    # distinct per-request frequencies: combining R and C content in one row
    # is refused instead of silently mixing.
    reference = reference_capture(freq="distinct")
    candidate = two_request_capture("fake-lane0-batch81-Full-2rows",
                                    freq="distinct")
    with pytest.raises(rc.RowCaseError, match="frequency"):
        rc.plan_cross_substitution(reference, candidate, tag="t")


def test_d_weight_mismatch_is_rejected():
    other_weights = dict(SHARED_WEIGHTS)
    other_weights["wkv"] = tagged("WKV-OTHER", PROJ_WEIGHT)
    reference = reference_capture(weights=other_weights)
    candidate = two_request_capture("fake-lane0-batch81-Full-2rows")
    with pytest.raises(rc.RowCaseError, match="wkv"):
        rc.plan_cross_substitution(reference, candidate, tag="t")


def test_d_no_alias_collision_across_lanes():
    reference = reference_capture()
    lane0 = two_request_capture("fake-lane0-batch81-Full-2rows")
    lane1 = two_request_capture("fake-lane1-batch82-Full-2rows",
                                descs=(2, 3))
    experiments = (rc.plan_cross_substitution(reference, lane0, tag="t")
                   + rc.plan_cross_substitution(reference, lane1, tag="t"))
    names = [e.name for e in experiments]
    assert len(names) == len(set(names))
    assert_well_formed(experiments)


def test_d_lane_using_slots_two_and_three_maps_correctly():
    reference = reference_capture()
    lane1 = two_request_capture("fake-lane1-batch82-Full-2rows",
                                descs=(2, 3))
    experiments = rc.plan_cross_substitution(reference, lane1, tag="t")
    plane = lane1.read_buffer("pending-kv")
    for r, slot in ((0, 2), (1, 3)):
        cc = by_name(experiments, f"_row{r}_curC_prevC")
        assert operand(cc, "pending_kv").blob == row(plane, slot, PENDING_ROW)
    # bundle maps both rows to their own slots, in order
    originals = by_name(experiments, "_bundle_cand_originals")
    assert operand(originals, "pending_kv").blob == (
        row(plane, 2, PENDING_ROW) + row(plane, 3, PENDING_ROW))


# ---------------------------------------------------------------------------
# Predecessor resolution semantics.
# ---------------------------------------------------------------------------

def test_same_request_earlier_wave_is_supported():
    capture = make_capture(
        "fake-wave-batch7-Full-2rows", rows=2,
        descs=[DESCRIPTOR_SENTINEL, SLOTS + 0], leases=[0, 0],
        positions=[40, 41], chunk_of_row=[0, 0])
    previous = rc.resolve_row_previous(capture, 1)
    assert previous.kind == "wave"
    assert previous.index == 0
    assert previous.previous_position == 40
    assert previous.kv == row(capture.read_buffer("projected"), 0,
                              PENDING_ROW)
    assert previous.scores == row(capture.read_buffer("scores"), 0,
                                  PENDING_ROW)
    # the planner builds real cases on top of the earlier-wave previous
    experiments = rc.plan_candidate_geometry(capture, None, tag="t")
    assert_well_formed(experiments)
    single = by_name(experiments, "_row1_single")
    assert operand(single, "pending_kv").blob == row(
        capture.read_buffer("projected"), 0, PENDING_ROW)
    duplicate = by_name(experiments, "_row1_duplicate")
    assert operand(duplicate, "pending_kv").blob == row(
        capture.read_buffer("projected"), 0, PENDING_ROW) * 2


def test_cross_request_earlier_wave_is_rejected():
    capture = make_capture(
        "fake-crosswave-batch8-Full-2rows", rows=2,
        descs=[DESCRIPTOR_SENTINEL, SLOTS + 0], leases=[0, 1],
        positions=[40, 41], chunk_of_row=[0, 1])
    with pytest.raises(rc.RowCaseError, match="cross-request"):
        rc.resolve_row_previous(capture, 1)
    with pytest.raises(rc.RowCaseError, match="cross-request"):
        rc.plan_candidate_geometry(capture, None, tag="t")


def test_cross_request_pending_slot_is_rejected():
    # row 1's descriptor addresses slot 2, but its chunk leases slot 3:
    # pooling a pending row that is not this request's own lease is refused
    capture = make_capture("fake-badlease-batch9-Full-2rows", rows=2,
                           descs=[1, 2], leases=[1, 3],
                           positions=[41, 41], chunk_of_row=[0, 1])
    with pytest.raises(rc.RowCaseError, match="leases slot"):
        rc.resolve_row_previous(capture, 1)


def test_sentinel_on_selected_odd_row_is_rejected():
    capture = make_capture(
        "fake-sentinel-batchA-Full-2rows", rows=2,
        descs=[DESCRIPTOR_SENTINEL, 3], leases=[0, 3],
        positions=[41, 41], chunk_of_row=[0, 1])
    with pytest.raises(rc.RowCaseError, match="sentinel"):
        rc.plan_candidate_geometry(capture, None, tag="t")


def test_descriptor_disagrees_with_manifest_is_rejected():
    capture = two_request_capture("fake-descmismatch-batchB-Full-2rows")
    wave_rows = list(capture.wave_rows)
    wave_rows[1] = dict(wave_rows[1], device_descriptor=3)
    capture.wave_rows = tuple(wave_rows)
    with pytest.raises(rc.RowCaseError, match="manifest wave_rows"):
        rc.resolve_row_previous(capture, 1)


def test_completion_pair_invariant_is_enforced():
    # p43 must complete the latent [42, 43] at logical row 21
    good = make_capture("fake-p43-good", rows=1, descs=[0], leases=[0],
                        positions=[43], chunk_of_row=[0])
    assert rc.odd_position_rows(good) == [0]
    bad = make_capture("fake-p43-bad", rows=1, descs=[0], leases=[0],
                       positions=[43], chunk_of_row=[0],
                       latent_first_tokens={0: 43})
    with pytest.raises(rc.RowCaseError, match="complete"):
        rc.odd_position_rows(bad)


def test_operand_bytes_pass_through_untouched():
    candidate = two_request_capture("fake-lane0-batch81-Full-2rows")
    experiments = rc.plan_candidate_geometry(candidate, None, tag="t")
    single = by_name(experiments, "_row0_single")
    pending = operand(single, "pending_kv").blob
    assert ODD_BIT_PATTERNS in pending            # NaN/inf/denormal survive
    assert ODD_BIT_PATTERNS in operand(single, "input").blob
    reference = reference_capture()
    d_experiments = rc.plan_cross_substitution(reference, candidate, tag="t")
    cr = by_name(d_experiments, "_row0_curC_prevR")
    assert ODD_BIT_PATTERNS in operand(cr, "pending_kv").blob
    assert ODD_BIT_PATTERNS in operand(cr, "projected").blob


# ---------------------------------------------------------------------------
# Plan validation (negative cases) through the shared validator.
# ---------------------------------------------------------------------------

def pool_experiment(**overrides):
    """A minimal structurally valid pool-only plan (rows=1, slots=1)."""
    experiment = PlannedExperiment(
        name="minimal", kind="C", rows=1, slots=1, counterfactual=True,
        operands=[
            PlannedOperand("kv", tagged("K", PENDING_ROW), "float32",
                           (1, LATENT), "test"),
            PlannedOperand("scores", tagged("S", PENDING_ROW), "float32",
                           (1, LATENT), "test"),
            PlannedOperand("pending_kv", tagged("PK", PENDING_ROW), "float32",
                           (1, LATENT), "test"),
            PlannedOperand("pending_scores", tagged("PS", PENDING_ROW),
                           "float32", (1, LATENT), "test"),
            PlannedOperand("predecessors", u64([0]), "uint64", (1,), "test"),
            PlannedOperand("norm", tagged("N", NORM), "bfloat16", (LATENT,),
                           "test"),
        ],
        stages=[{"kind": "pool", "kv": "kv", "scores": "scores",
                 "pending_kv": "pending_kv", "pending_scores": "pending_scores",
                 "predecessors": "predecessors", "norm": "norm",
                 "output": "output", "slots": 1}],
        outputs=("output",), expected={}, notes=(),
    )
    for key, value in overrides.items():
        object.__setattr__(experiment, key, value)
    return experiment


def pack_experiment(**overrides):
    """A minimal structurally valid pack-only plan (rows=1)."""
    experiment = PlannedExperiment(
        name="pack-minimal", kind="C", rows=1, slots=1, counterfactual=True,
        operands=[
            PlannedOperand("output", tagged("O", LATENT * 2), "bfloat16",
                           (1, LATENT), "test"),
            PlannedOperand("frequencies", tagged("F", FREQ_ROW), "float32",
                           (1, 32, 2), "test"),
        ],
        stages=[{"kind": "pack", "input": "output", "frequencies":
                 "frequencies", "values": "values", "scales": "scales"}],
        outputs=("values", "scales"), expected={}, notes=(),
    )
    for key, value in overrides.items():
        object.__setattr__(experiment, key, value)
    return experiment


def test_validate_rejects_missing_stage_input_role():
    experiment = pool_experiment()
    next(s for s in experiment.stages if s["kind"] == "pool")["kv"] = \
        "missing_role"
    with pytest.raises(ValidationError, match="neither"):
        validate_experiment(experiment)


def test_validate_rejects_wrong_operand_length():
    experiment = pool_experiment()
    experiment.operands[0] = PlannedOperand(
        "kv", b"\x00" * 10, "float32", (1, LATENT), "test")
    with pytest.raises(ValidationError, match="bytes 10"):
        validate_experiment(experiment)


def test_validate_rejects_unallocated_produced_role():
    experiment = pool_experiment(outputs=())
    with pytest.raises(ValidationError, match="not listed in experiment.outputs"):
        validate_experiment(experiment)


def test_validate_tolerates_counterfactual_oracle_bytes():
    # The shared validator treats a counterfactual oracle as optional, but any
    # oracle it does carry must still have the exact output byte length.
    experiment = pool_experiment(expected={"output": bytes(1024)})
    validate_experiment(experiment)                 # tolerated, never required
    experiment.expected["output"] = bytes(1023)
    with pytest.raises(ValidationError, match="bytes but the output is"):
        validate_experiment(experiment)


def test_validate_rejects_gate_without_expected():
    experiment = pool_experiment(counterfactual=False)
    with pytest.raises(ValidationError, match="missing an exact oracle"):
        validate_experiment(experiment)


def test_validate_rejects_duplicate_operand_roles():
    experiment = pool_experiment()
    experiment.operands.append(experiment.operands[0])
    with pytest.raises(ValidationError, match="duplicate operand role"):
        validate_experiment(experiment)


def test_validate_rejects_misaligned_frequencies():
    experiment = pack_experiment()
    experiment.operands = [
        experiment.operands[0],
        PlannedOperand("frequencies", tagged("F2", FREQ_ROW * 2), "float32",
                       (2, 32, 2), "test"),
    ]
    with pytest.raises(ValidationError, match="expected 256"):
        validate_experiment(experiment)


# ---------------------------------------------------------------------------
# Shared-schema integration: schema -> row_cases -> shared validator.
# ---------------------------------------------------------------------------

def test_schema_rowcases_validator_integration(tmp_path):
    from compressor_replay.fixtures import write_synthetic_capture

    root = tmp_path / "activations"
    root.mkdir()
    freq_row = bytes((i * 17 + 5) & 0xFF for i in range(FREQ_ROW))
    write_synthetic_capture(root, "ref-single", geometry="single",
                            weights_writer=True, seed=11,
                            frequency_row=freq_row)
    write_synthetic_capture(root, "cand-pair", geometry="pair",
                            weights_writer=False, seed=12,
                            frequency_row=freq_row)
    # Load through the corrected shared schema (real manifest layout).
    reference = schema.load_capture(root / "ref-single")
    candidate = schema.load_capture(root / "cand-pair")
    c_cases = rc.plan_candidate_geometry(candidate, reference, tag="")
    d_cases = rc.plan_cross_substitution(reference, candidate, tag="")
    assert d_cases                                     # D exists for a 1-row ref
    assert any(not e.counterfactual for e in d_cases)  # D original gates
    # Every stage read is contract-valid on every planned experiment.
    for experiment in c_cases + d_cases:
        validate_experiment(experiment)
    singles = [e for e in c_cases if e.name.endswith("_single")]
    assert singles
    for experiment in singles:
        assert experiment.rows == 1 and experiment.slots == 1
        assert len(operand(experiment, "frequencies").blob) == FREQ_ROW == 256


# ---------------------------------------------------------------------------
# Actual captures through the shared schema loader (skipped when absent).
# ---------------------------------------------------------------------------

def test_shared_schema_loads_synthetic_capture_directory(tmp_path):
    from compressor_replay.fixtures import write_synthetic_capture

    directory = write_synthetic_capture(tmp_path, "synthetic-single",
                                        geometry="single", slot_count=4,
                                        seed=7)
    capture = schema.load_capture(directory)
    assert capture.rows == 1
    assert capture.slot_count == 4
    assert capture.device_descriptors[0] == 0     # single geometry, pending slot 0
    assert capture.wave_rows[0]["absolute_position"] == 41
    experiments = rc.plan_candidate_geometry(capture, None, tag="t")
    assert_well_formed(experiments)
    single = by_name(experiments, "_row0_single")
    assert operand(single, "pending_kv").blob == row(
        capture.read_buffer("pending-kv"), 0, PENDING_ROW)


# ---------------------------------------------------------------------------
# Read-only actual-plan smoke (skipped when the capture tree is absent).
# ---------------------------------------------------------------------------

pytestmark_actual = pytest.mark.skipif(
    not (ACTIVATIONS / REFERENCE_DIR).is_dir(),
    reason=f"actual captures not present under {ACTIVATIONS}",
)


@pytest.mark.parametrize("member", CANDIDATE_DIRS)
@pytestmark_actual
def test_actual_manifests_load_and_plan(member):
    reference = schema.load_capture(ACTIVATIONS / REFERENCE_DIR)
    candidate = schema.load_capture(ACTIVATIONS / member)
    experiments = (rc.plan_candidate_geometry(candidate, reference, tag="")
                   + rc.plan_cross_substitution(reference, candidate, tag=""))
    assert_well_formed(experiments)
    assert len(experiments) == 19                 # 7 C + 12 D
    for experiment in experiments:
        assert experiment.counterfactual or experiment.kind == "D"


@pytestmark_actual
def test_actual_reference_and_lane_geometry():
    reference = schema.load_capture(ACTIVATIONS / REFERENCE_DIR)
    assert reference.rows == 1
    assert reference.wave_rows[0]["request_id"] == 2026091950
    assert reference.wave_rows[0]["absolute_position"] == 41
    assert reference.device_descriptors == (0,)
    lane0 = schema.load_capture(ACTIVATIONS / CANDIDATE_DIRS[0])
    lane1 = schema.load_capture(ACTIVATIONS / CANDIDATE_DIRS[1])
    assert lane0.device_descriptors == (0, 1)
    assert lane0.wave_rows[0]["request_id"] == 2026091960
    assert lane0.wave_rows[1]["request_id"] == 2026091961
    assert lane1.device_descriptors == (2, 3)
    assert lane1.wave_rows[0]["request_id"] == 2026091962
    assert lane1.wave_rows[1]["request_id"] == 2026091963
    for capture in (reference, lane0, lane1):
        for r in range(capture.rows):
            assert capture.wave_rows[r]["absolute_position"] == 41
            latent = capture.wave_rows[r]["completed_latent"]
            assert latent["first_token"] == 40
            assert latent["logical_compressed_row"] == 20
            assert rc.resolve_row_previous(capture, r).kind == "pending"


@pytestmark_actual
def test_actual_weights_and_frequencies_match_across_sources():
    captures = [schema.load_capture(ACTIVATIONS / name)
                for name in (REFERENCE_DIR,) + CANDIDATE_DIRS]
    hashes = rc.assert_weights_compatible(*captures)
    assert set(hashes) == {"wkv", "wgate", "norm"}
    freq = {c.directory.name: c.read_buffer("frequencies") for c in captures}
    reference_freq = freq[REFERENCE_DIR]
    assert len(reference_freq) == FREQ_ROW
    for name, blob in freq.items():
        for r in range(len(blob) // FREQ_ROW):
            assert row(blob, r, FREQ_ROW) == reference_freq


@pytestmark_actual
def test_actual_d_gates_use_captured_oracles_and_lane_slots_map():
    reference = schema.load_capture(ACTIVATIONS / REFERENCE_DIR)
    lane0 = schema.load_capture(ACTIVATIONS / CANDIDATE_DIRS[0])
    lane1 = schema.load_capture(ACTIVATIONS / CANDIDATE_DIRS[1])
    for candidate, slots in ((lane0, (0, 1)), (lane1, (2, 3))):
        experiments = rc.plan_cross_substitution(reference, candidate, tag="")
        originals = by_name(experiments, "_bundle_cand_originals")
        assert originals.expected["output"] == candidate.read_buffer("output")
        assert originals.expected["values"] == (
            candidate.read_buffer("kv-values"))
        assert originals.expected["scales"] == (
            candidate.read_buffer("kv-scales"))
        plane = candidate.read_buffer("pending-kv")
        for r, slot in enumerate(slots):
            cc = by_name(experiments, f"_row{r}_curC_prevC")
            assert operand(cc, "pending_kv").blob == row(
                plane, slot, PENDING_ROW)


@pytestmark_actual
def test_actual_names_unique_across_both_lanes():
    captures = schema.load_captures(ACTIVATIONS)
    assert len(captures) == 3                       # three manifests
    assert sorted(c.rows for c in captures) == [1, 2, 2]
    assert sum(len(c.p41_rows()) for c in captures) == 5   # five p41 rows
    reference = schema.load_capture(ACTIVATIONS / REFERENCE_DIR)
    c_experiments, d_experiments = [], []
    for member in CANDIDATE_DIRS:
        candidate = schema.load_capture(ACTIVATIONS / member)
        c_member = rc.plan_candidate_geometry(candidate, reference, tag="")
        d_member = rc.plan_cross_substitution(reference, candidate, tag="")
        assert len(c_member) == 7                   # 2 singles + 2 dups + 1 swap + 2 mixed
        assert len(d_member) == 12                  # 4 per candidate row + 4 bundles
        c_experiments += c_member
        d_experiments += d_member
    assert len(c_experiments) == 14                 # C14 across both candidates
    assert len(d_experiments) == 24                 # D24 across both candidates
    names = [e.name for e in c_experiments + d_experiments]
    assert len(names) == len(set(names)) == 38
    gated = [e for e in d_experiments if not e.counterfactual]
    assert len(gated) == 14     # per candidate: 2 rows x 2 originals + 3 bundles
    assert all(e.kind == "D" for e in gated)
    assert all(e.expected for e in gated)           # D original gates present
