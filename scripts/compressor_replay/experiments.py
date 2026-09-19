"""Experiment planning: turn validated captures into replay experiments.

Families (three repeats each on GPU; raw outputs persisted):

A. Original-batch replay: recorded projected/scores/pending/descriptors
   through native pool then pack, byte-compared to the capture.  A separate
   direct-pack baseline packs the recorded BF16 output (isolates downstream).
B. Fresh projections: captured input x captured WKV/Wgate through project at
   the original batch size, FP32 bytes compared to the capture; then the
   fresh projections feed pool+pack with the captured pending state.
C. Candidate-geometry counterfactuals: each odd-position row of a two-row
   candidate capture rerun independently as rows=1, plus duplicate / mixed /
   swapped rows=2 regroupings.  Deliberate interventions — evidence, not
   baselines.  Planned by ``row_cases.plan_candidate_geometry``.
D. Cross-substitution baselines and counterfactuals: a reference capture (any
   row count, including one) paired with each two-row candidate; every row
   bundle's current projected/scores and previous pending operand are pooled
   in compact fixtures, with the originals gated against the captured
   oracles.  Planned by ``row_cases.plan_cross_substitution``.

Counterfactual experiments carry ``counterfactual=True``: their raw inputs
are hashed, every source and substitution is recorded, and no accuracy claim
is made — they never gate the component baseline.
"""

from __future__ import annotations

import hashlib
from typing import List, Optional, Sequence, Tuple

from .row_cases import plan_candidate_geometry, plan_cross_substitution
from .runner import PlannedExperiment, PlannedOperand
from .schema import ROW_BYTES, Capture

# Group-18 mutation target: scale group 18 covers columns 288..303 of a
# packed row (16 columns per E4M3 scale byte; 2 FP4 values per value byte).
MUTATION_GROUP = 18
MUTATION_COLS = tuple(range(MUTATION_GROUP * 16, MUTATION_GROUP * 16 + 16))


def pack_affected_regions(cols: Sequence[int]) -> dict:
    """Which compressed-pack output bytes a column mutation may influence.

    Structural byte bookkeeping only (no float arithmetic): value byte
    ``col // 2`` holds FP4 values for columns ``2*b, 2*b+1``; scale byte
    ``col // 16`` covers its 16-column group.
    """
    cols = list(cols)
    return {
        "value_bytes": sorted({c // 2 for c in cols}),
        "scale_bytes": sorted({c // 16 for c in cols}),
    }


def mutate_columns(blob: bytes, *, row_bytes: int, row: int,
                   cols: Sequence[int], xor_byte: int = 0x5A) -> Tuple[bytes, dict]:
    """Raw-bit mutation of one row's columns (BF16 element = 2 bytes).

    Returns the mutated blob plus evidence; hashes of every derived fixture
    must be regenerated from the mutated bytes, never reused from the base.
    """
    import hashlib

    cols = list(cols)
    data = bytearray(blob)
    base = row * row_bytes
    for col in cols:
        offset = base + col * 2
        data[offset] ^= xor_byte
        data[offset + 1] ^= xor_byte >> 4 if xor_byte >> 4 else xor_byte
    mutated = bytes(data)
    evidence = {
        "kind": "column_mutation",
        "row": row,
        "cols": cols,
        "group": MUTATION_GROUP,
        "affected_pack_regions": pack_affected_regions(cols),
        "original_sha256": hashlib.sha256(blob).hexdigest(),
        "mutated_sha256": hashlib.sha256(mutated).hexdigest(),
        "xor_byte": xor_byte,
    }
    return mutated, evidence


def row_slice(blob: bytes, row_bytes: int, index: int) -> bytes:
    return blob[index * row_bytes:(index + 1) * row_bytes]


def _operand(role, blob, dtype, shape, source) -> PlannedOperand:
    return PlannedOperand(role=role, blob=bytes(blob), dtype=dtype,
                          shape=tuple(shape), source=source)


def _pool_pack_stages(*, kv="kv", scores="scores", output="output",
                      frequencies="frequencies", values="values",
                      scales="scales", predecessors="predecessors",
                      pending_kv="pending_kv", pending_scores="pending_scores",
                      norm="norm", slots=None):
    stages = [{
        "kind": "pool", "kv": kv, "scores": scores,
        "pending_kv": pending_kv, "pending_scores": pending_scores,
        "predecessors": predecessors, "norm": norm, "output": output,
    }]
    if slots is not None:
        stages[0]["slots"] = slots
    stages.append({
        "kind": "pack", "input": output, "frequencies": frequencies,
        "values": values, "scales": scales,
    })
    return stages


def _project_stages() -> List[dict]:
    return [
        {"kind": "project", "input": "input", "weight": "wkv",
         "output": "projected"},
        {"kind": "project", "input": "input", "weight": "wgate",
         "output": "scores"},
    ]


# ---------------------------------------------------------------------------
# Family A and B (per capture; baselines that gate).
# ---------------------------------------------------------------------------

def plan_capture_baselines(capture: Capture, tag: str) -> List[PlannedExperiment]:
    """A1: pool+pack on recorded operands.  A2: direct pack of recorded
    output.  B1: fresh projections.  B2: fresh projections -> pool+pack."""
    name = capture.directory.name
    projected = capture.read_buffer("projected")
    scores = capture.read_buffer("scores")
    pending_kv = capture.read_buffer("pending-kv")
    pending_scores = capture.read_buffer("pending-scores")
    descriptors = capture.read_buffer("descriptors-device")
    norm = capture.read_weight("norm")
    frequencies = capture.read_buffer("frequencies")
    output = capture.read_buffer("output")
    kv_values = capture.read_buffer("kv-values")
    kv_scales = capture.read_buffer("kv-scales")
    rows, slots = capture.rows, capture.slot_count
    prefix = f"{tag}{name}"

    common_pool_operands = [
        _operand("kv", projected, "float32", (rows, 512),
                 f"capture:{name}:projected"),
        _operand("scores", scores, "float32", (rows, 512),
                 f"capture:{name}:scores"),
        _operand("pending_kv", pending_kv, "float32", (slots, 512),
                 f"capture:{name}:pending-kv"),
        _operand("pending_scores", pending_scores, "float32", (slots, 512),
                 f"capture:{name}:pending-scores"),
        _operand("predecessors", descriptors, "uint64", (rows,),
                 f"capture:{name}:descriptors-device"),
        _operand("norm", norm, "bfloat16", (512,),
                 f"capture:{name}:norm-weight"),
    ]

    a1 = PlannedExperiment(
        name=f"{prefix}_A1_pool_pack_recorded", kind="A", rows=rows, slots=slots,
        operands=common_pool_operands + [
            _operand("frequencies", frequencies, "float32", (rows, 32, 2),
                     f"capture:{name}:frequencies"),
        ],
        stages=_pool_pack_stages(),
        outputs=("output", "values", "scales"),
        expected={"output": output, "values": kv_values, "scales": kv_scales},
        notes=("A1: recorded projected/scores/pending/descriptors through "
               "native pool then pack; byte-compared to the capture.",),
    )

    a2 = PlannedExperiment(
        name=f"{prefix}_A2_direct_pack_recorded_output", kind="A", rows=rows,
        slots=slots,
        operands=[
            _operand("output", output, "bfloat16", (rows, 512),
                     f"capture:{name}:output"),
            _operand("frequencies", frequencies, "float32", (rows, 32, 2),
                     f"capture:{name}:frequencies"),
        ],
        stages=[{"kind": "pack", "input": "output",
                 "frequencies": "frequencies", "values": "values",
                 "scales": "scales"}],
        outputs=("values", "scales"),
        expected={"values": kv_values, "scales": kv_scales},
        notes=("A2: direct pack baseline of the recorded BF16 output; "
               "isolates the downstream pipeline from pooling.",),
    )

    proj_rows = rows
    b1 = PlannedExperiment(
        name=f"{prefix}_B1_project_wkv_wgate", kind="B", rows=proj_rows,
        slots=slots,
        operands=[
            _operand("input", capture.read_buffer("input"), "bfloat16",
                     (rows, 5120), f"capture:{name}:input"),
            _operand("wkv", capture.read_weight("wkv"), "bfloat16",
                     (512, 5120), f"capture:{name}:wkv-weight"),
            _operand("wgate", capture.read_weight("wgate"), "bfloat16",
                     (512, 5120), f"capture:{name}:wgate-weight"),
        ],
        stages=_project_stages(),
        outputs=("projected", "scores"),
        expected={"projected": projected, "scores": scores},
        notes=("B1: captured input x captured WKV/Wgate through project at "
               "the original batch size; FP32 bytes compared to capture.",),
    )

    b2 = PlannedExperiment(
        name=f"{prefix}_B2_fresh_projections_pool_pack", kind="B", rows=rows,
        slots=slots,
        operands=[
            _operand("input", capture.read_buffer("input"), "bfloat16",
                     (rows, 5120), f"capture:{name}:input"),
            _operand("wkv", capture.read_weight("wkv"), "bfloat16",
                     (512, 5120), f"capture:{name}:wkv-weight"),
            _operand("wgate", capture.read_weight("wgate"), "bfloat16",
                     (512, 5120), f"capture:{name}:wgate-weight"),
            _operand("pending_kv", pending_kv, "float32", (slots, 512),
                     f"capture:{name}:pending-kv"),
            _operand("pending_scores", pending_scores, "float32", (slots, 512),
                     f"capture:{name}:pending-scores"),
            _operand("predecessors", descriptors, "uint64", (rows,),
                     f"capture:{name}:descriptors-device"),
            _operand("norm", norm, "bfloat16", (512,),
                     f"capture:{name}:norm-weight"),
            _operand("frequencies", frequencies, "float32", (rows, 32, 2),
                     f"capture:{name}:frequencies"),
        ],
        # The fresh projections produce roles "projected"/"scores"; the pool
        # stage must read those names, not the recorded "kv"/"scores" operands.
        stages=_project_stages() + _pool_pack_stages(kv="projected"),
        outputs=("projected", "scores", "output", "values", "scales"),
        expected={"projected": projected, "scores": scores, "output": output,
                  "values": kv_values, "scales": kv_scales},
        notes=("B2: fresh WKV/Wgate projections feed pool+pack with captured "
               "pending state; end-to-end without recorded projections.",),
    )
    return [a1, a2, b1, b2]


def plan_mutation_variant(capture: Capture, tag: str) -> PlannedExperiment:
    """Counterfactual B-variant: raw-bit mutation of input columns 288..303
    (scale group 18) through project only.  Evidence that mutation hashes
    regenerate and that downstream bytes shift; never a baseline."""
    name = capture.directory.name
    input_blob = capture.read_buffer("input")
    mutated, evidence = mutate_columns(
        input_blob, row_bytes=ROW_BYTES["input"], row=0, cols=MUTATION_COLS,
    )
    return PlannedExperiment(
        name=f"{tag}{name}_B3_mutated_group18_input_project", kind="B",
        rows=capture.rows, slots=capture.slot_count, counterfactual=True,
        operands=[
            _operand("input", mutated, "bfloat16",
                     (capture.rows, 5120),
                     f"mutation:{name}:input cols{MUTATION_COLS[0]}-"
                     f"{MUTATION_COLS[-1]}"),
            _operand("wkv", capture.read_weight("wkv"), "bfloat16",
                     (512, 5120), f"capture:{name}:wkv-weight"),
            _operand("wgate", capture.read_weight("wgate"), "bfloat16",
                     (512, 5120), f"capture:{name}:wgate-weight"),
        ],
        stages=_project_stages(),
        outputs=("projected", "scores"),
        expected={},
        notes=("B3 counterfactual: columns 288..303 (scale group 18) of input "
               "row 0 mutated at raw-bit level.",
               f"mutation_evidence: {evidence}",),
    )


# ---------------------------------------------------------------------------
# Family P (fixed-size padded project-only placement counterfactuals).
# ---------------------------------------------------------------------------

PADDED_ROWS = (2, 16)
PADDED_SLOTS = {2: (0, 1), 16: (0, 1, 7, 15)}
DISTINCT_NEIGHBOR = {2: 1, 16: 15}


def current_input_row(capture: Capture) -> int:
    """The captured *current* row: the row that completes a latent."""
    for row, wave_row in enumerate(capture.wave_rows):
        if wave_row.get("completed_latent") is not None:
            return row
    return capture.rows - 1


def _padded_input_blob(capture: Capture, rows: int, placement: int,
                       distinct_neighbor: Optional[int],
                       current_row: int) -> Tuple[bytes, Tuple[str, ...]]:
    """Zero-fill ``rows`` input rows, put the real current row at ``placement``.

    With ``distinct_neighbor`` set, one *other* slot holds the same real row
    with the low bit of its last BF16 input value flipped, so that slot is a
    neighbour distinct from both zero padding and the exact row.  The exact
    provenance and hashes are returned for the record.
    """
    row_bytes = ROW_BYTES["input"]
    source = row_slice(capture.read_buffer("input"), row_bytes, current_row)
    padded = bytearray(rows * row_bytes)
    padded[placement * row_bytes:(placement + 1) * row_bytes] = source
    provenance = [
        f"zero-filled padding: {rows} rows, slot {placement} = real current "
        f"input row {current_row} (bytes {len(source)}, "
        f"sha256 {hashlib.sha256(source).hexdigest()})"
    ]
    if distinct_neighbor is not None:
        mutated = bytearray(source)
        # BF16 element = 2 little-endian bytes: the value's low bit is bit 0
        # of byte -2, not byte -1 (which holds the high 8 exponent/mantissa
        # bits).  The provenance offset below has always pointed at -2.
        mutated[-2] ^= 0x01
        padded[distinct_neighbor * row_bytes:
               (distinct_neighbor + 1) * row_bytes] = bytes(mutated)
        provenance.append(
            f"distinct neighbour: slot {distinct_neighbor} = real row "
            f"{current_row} with BF16 low bit of the last input value "
            f"flipped (offset {distinct_neighbor * row_bytes + len(mutated) - 2}"
            f", sha256 {hashlib.sha256(bytes(mutated)).hexdigest()})"
        )
    return bytes(padded), tuple(provenance)


def plan_padded_project_only(capture: Capture, tag: str = ""
                             ) -> List[PlannedExperiment]:
    """Fixed-size padded PROJECT-ONLY placement counterfactuals at rows 2 and 16.

    Each experiment runs the existing native project path only (WKV and Wgate)
    at a fixed padded batch size, with the capture's real current input row at
    one slot and every unused slot zero-filled.  Slots 0/1 are used at rows 2
    and slots 0/1/7/15 at rows 16; each batch size also gets one distinct-
    neighbour variant whose neighbour holds the real row with the low bit of
    its last BF16 input value flipped.

    These carry ``counterfactual=True`` and an empty ``expected`` dict: they
    are raw FP32 evidence for a later placement-invariance and exact-dot
    comparison, never a numeric gate, and no pooling runs on the padding.
    """
    name = capture.directory.name
    current_row = current_input_row(capture)
    wkv = capture.read_weight("wkv")
    wgate = capture.read_weight("wgate")
    experiments: List[PlannedExperiment] = []
    for rows in PADDED_ROWS:
        for placement in PADDED_SLOTS[rows]:
            blob, provenance = _padded_input_blob(
                capture, rows, placement, None, current_row
            )
            experiments.append(PlannedExperiment(
                name=f"{tag}{name}_P_pad{rows}_slot{placement}", kind="P",
                rows=rows, slots=capture.slot_count, counterfactual=True,
                operands=[
                    _operand("input", blob, "bfloat16", (rows, 5120),
                             f"padding:{name}:slot{placement} real row "
                             f"{current_row} + zero neighbors"),
                    _operand("wkv", wkv, "bfloat16", (512, 5120),
                             f"capture:{name}:wkv-weight"),
                    _operand("wgate", wgate, "bfloat16", (512, 5120),
                             f"capture:{name}:wgate-weight"),
                ],
                stages=_project_stages(),
                outputs=("projected", "scores"),
                expected={},
                notes=(
                    f"P padded project-only: rows {rows}, real current row "
                    f"{current_row} at slot {placement}, all other slots zero.",
                    "no pooling on padding; counterfactual projection evidence "
                    "only, no numeric gate.",
                    *provenance,
                ),
            ))
        neighbor = DISTINCT_NEIGHBOR[rows]
        placement = PADDED_SLOTS[rows][0]
        blob, provenance = _padded_input_blob(
            capture, rows, placement, neighbor, current_row
        )
        experiments.append(PlannedExperiment(
            name=f"{tag}{name}_P_pad{rows}_slot{placement}_distinct_nb{neighbor}",
            kind="P", rows=rows, slots=capture.slot_count, counterfactual=True,
            operands=[
                _operand("input", blob, "bfloat16", (rows, 5120),
                         f"padding:{name}:slot{placement} real row "
                         f"{current_row} + distinct neighbor slot {neighbor}"),
                _operand("wkv", wkv, "bfloat16", (512, 5120),
                         f"capture:{name}:wkv-weight"),
                _operand("wgate", wgate, "bfloat16", (512, 5120),
                         f"capture:{name}:wgate-weight"),
            ],
            stages=_project_stages(),
            outputs=("projected", "scores"),
            expected={},
            notes=(
                f"P padded project-only: rows {rows}, real current row "
                f"{current_row} at slot {placement}, distinct neighbour at "
                f"slot {neighbor} (last BF16 input value low bit flipped).",
                "no pooling on padding; counterfactual projection evidence "
                "only, no numeric gate.",
                *provenance,
            ),
        ))
    return experiments


# ---------------------------------------------------------------------------
# Top-level planning.
# ---------------------------------------------------------------------------

def _select_reference(captures: Sequence[Capture]) -> Tuple[Optional[Capture], str]:
    """Resolve the reference capture and the rule that selected it.

    Rules, in order:

    * ``explicit_name``: exactly one capture directory name mentions
      ``reference`` (its name is authoritative and preserved);
    * ``unique_single_row``: the name match is not unique (or absent), but
      exactly one capture is a one-row batch, so it is the only possible
      one-row reference;
    * ``none``: no unambiguous reference exists (A/B/P only).

    An ambiguous one-row fallback (multiple one-row captures) raises: the
    reference is never silently resolved to the first match.  The rule and
    resolved name are recorded in the CPU plan.
    """
    named = [c for c in captures if "reference" in c.directory.name.lower()]
    if len(named) == 1:
        return named[0], "explicit_name"

    single_row = [c for c in captures if c.rows == 1]
    if len(single_row) > 1:
        # A non-unique explicit name is itself ambiguous, so an ambiguous
        # one-row fallback is reported with both facts.
        detail = (
            "multiple names mention 'reference' "
            f"({', '.join(sorted(c.directory.name for c in named))}) and "
            if len(named) > 1 else ""
        )
        raise ValueError(
            "ambiguous reference: " + detail
            + "multiple one-row captures exist: "
            + ", ".join(sorted(c.directory.name for c in single_row))
        )
    if len(single_row) == 1:
        return single_row[0], "unique_single_row"
    return None, "none"


def select_reference(captures: Sequence[Capture]) -> Optional[Capture]:
    """The unambiguous reference capture, or None (A/B/P only)."""
    return _select_reference(captures)[0]


def reference_selection(captures: Sequence[Capture]) -> dict:
    """The reference selection recorded in the CPU plan (rule + name)."""
    reference, rule = _select_reference(captures)
    return {
        "rule": rule,
        "refname": reference.directory.name if reference is not None else None,
        "rows": reference.rows if reference is not None else None,
    }


def plan_experiments(captures: Sequence[Capture], *, tag: str = "") -> List[PlannedExperiment]:
    """Full experiment list for a set of validated captures.

    A/B run per capture, plus the mutation counterfactual and padded
    project-only placement variants.  C runs for every two-row candidate.
    D pairs the unambiguous reference (any row count, including one) with each
    two-row candidate.  When no unambiguous reference exists, D is omitted and
    C runs without mixed cases — recorded, never fabricated.
    """
    experiments: List[PlannedExperiment] = []
    for capture in captures:
        experiments += plan_capture_baselines(capture, tag)
        experiments.append(plan_mutation_variant(capture, tag))
        experiments += plan_padded_project_only(capture, tag)
    reference = select_reference(captures)
    candidates = [c for c in captures if c is not reference]
    two_row = [c for c in candidates if c.rows == 2]
    for capture in two_row:
        experiments += plan_candidate_geometry(capture, reference, tag)
    if reference is not None:
        for candidate in two_row:
            experiments += plan_cross_substitution(reference, candidate, tag)
    return experiments
