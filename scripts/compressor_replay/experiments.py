"""Experiment planning: turn validated captures into replay experiments.

Families (three repeats each on GPU; raw outputs persisted):

A. Original-batch replay: recorded projected/scores/pending/descriptors
   through native pool then pack, byte-compared to the capture.  A separate
   direct-pack baseline packs the recorded BF16 output (isolates downstream).
B. Fresh projections: captured input x captured WKV/Wgate through project at
   the original batch size, FP32 bytes compared to the capture; then the
   fresh projections feed pool+pack with the captured pending state.
C. Candidate-geometry counterfactuals: each input row of a two-row capture
   rerun independently as rows=1, plus duplicate / mixed / swapped rows=2
   regroupings.  Deliberate interventions — evidence, not baselines.
D. Cross-substitution counterfactuals: reference vs candidate current and
   predecessor projected/scores swapped separately/together into compact
   pool+pack fixtures, plus duplicate and swapped-order variants.

Counterfactual experiments carry ``counterfactual=True``: their raw inputs
are hashed, every source and substitution is recorded, and no accuracy claim
is made — they never gate the component baseline.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .remap import build_compact_planes, remap_batch
from .runner import PlannedExperiment, PlannedOperand
from .schema import (
    DESCRIPTOR_SENTINEL,
    PENDING_ROW_BYTES,
    ROW_BYTES,
    Capture,
    resolve_predecessor,
)

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
        stages=_project_stages() + _pool_pack_stages(),
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
# Family C (candidate 2-row capture geometry counterfactuals).
# ---------------------------------------------------------------------------

def plan_candidate_geometry(capture: Capture, reference: Optional[Capture],
                            tag: str) -> List[PlannedExperiment]:
    """Rows=1 independent reruns, duplicate, mixed and swapped rows=2.

    Every variant is a labelled counterfactual with its own input hashes.
    Projections run at the regrouped batch size; downstream pool+pack runs
    wherever row predecessors can be made authoritative: sentinel rows
    directly, earlier-wave rows through a content-preserving compact pending
    remap (the captured predecessor row's exact FP32 bytes, byte-copied).
    """
    if capture.rows != 2:
        return []
    name = capture.directory.name
    experiments: List[PlannedExperiment] = []
    input_blob = capture.read_buffer("input")
    wkv = capture.read_weight("wkv")
    wgate = capture.read_weight("wgate")
    norm = capture.read_weight("norm")
    frequencies = capture.read_buffer("frequencies")
    pending_kv = capture.read_buffer("pending-kv")
    pending_scores = capture.read_buffer("pending-scores")
    projected = capture.read_buffer("projected")
    scores = capture.read_buffer("scores")
    descriptors = capture.device_descriptors

    def downstream_for(new_descriptors, *, slots: int,
                       compact_kv: bytes = b"", compact_scores: bytes = b""):
        """Downstream pool+pack operands.  A sentinel-only batch still runs:
        the native zero path is the recorded semantics of those rows."""
        pend_kv = compact_kv or pending_kv
        pend_scores = compact_scores or pending_scores
        operands = [
            _operand("pending_kv", pend_kv, "float32", (slots, 512),
                     f"capture:{name}:pending-kv" if not compact_kv
                     else f"compact:{name}:predecessor rows"),
            _operand("pending_scores", pend_scores, "float32", (slots, 512),
                     f"capture:{name}:pending-scores" if not compact_scores
                     else f"compact:{name}:predecessor rows"),
            _operand("predecessors", _u64_blob(new_descriptors), "uint64",
                     (len(new_descriptors),), "remap:descriptors"),
            _operand("norm", norm, "bfloat16", (512,),
                     f"capture:{name}:norm-weight"),
            _operand("frequencies", frequencies, "float32",
                     (len(new_descriptors), 32, 2),
                     f"capture:{name}:frequencies"),
        ]
        return _pool_pack_stages(slots=slots), operands, slots

    def make(exp_name, *, input_rows, new_descriptors, note, slots,
             compact_kv=b"", compact_scores=b"", extra_operands=()):
        rows = len(input_rows)
        input_sel = b"".join(row_slice(input_blob, ROW_BYTES["input"], r)
                             for r in input_rows)
        operands = [
            _operand("input", input_sel, "bfloat16", (rows, 5120),
                     f"capture:{name}:input rows{list(input_rows)}"),
            _operand("wkv", wkv, "bfloat16", (512, 5120),
                     f"capture:{name}:wkv-weight"),
            _operand("wgate", wgate, "bfloat16", (512, 5120),
                     f"capture:{name}:wgate-weight"),
        ] + list(extra_operands)
        stages = _project_stages()
        ds_stages, ds_operands, ds_slots = downstream_for(
            new_descriptors, slots=slots, compact_kv=compact_kv,
            compact_scores=compact_scores,
        )
        operands += ds_operands
        stages += ds_stages
        outputs = ["projected", "scores"]
        if ds_stages:
            outputs += ["output", "values", "scales"]
        return PlannedExperiment(
            name=exp_name, kind="C", rows=rows, slots=ds_slots or slots,
            counterfactual=True, operands=operands, stages=stages,
            outputs=tuple(outputs), expected={}, notes=(note,),
        )

    slot_count = capture.slot_count

    # Independent rows=1 reruns of the exact captured input rows.
    for row in (0, 1):
        desc = descriptors[row]
        pred = resolve_predecessor(desc, row, slot_count)
        if pred["kind"] == "invalid_sentinel":
            new_desc = [DESCRIPTOR_SENTINEL]
            compact_kv = compact_scores = b""
            slots = slot_count
            note = (f"C single row{row}: sentinel row replays standalone; "
                    "downstream emits the native zero path.")
        elif pred["kind"] == "earlier_wave_row":
            source = pred["wave_row"]
            remap = remap_batch(
                [desc], [row], original_slot_count=slot_count,
                row_map={row: 0}, pending_kv=pending_kv,
                pending_scores=pending_scores, strategy="compact_pending",
                compact_slots={source: source},
            )
            compact_kv, compact_scores = build_compact_planes(
                remap.source_map, pending_kv=pending_kv,
                pending_scores=pending_scores,
                wave_kv_rows={source: row_slice(projected, ROW_BYTES["projected"], source)},
                wave_scores_rows={source: row_slice(scores, ROW_BYTES["scores"], source)},
                original_slot_count=slot_count,
            )
            new_desc = list(remap.descriptors)
            slots = remap.slot_count
            note = (f"C single row{row}: earlier-wave predecessor preserved "
                    "via compact pending slot (byte-identical operands).")
        else:
            # Pending-slot predecessor: keep the original pending plane row.
            new_desc = [desc]
            compact_kv = compact_scores = b""
            slots = slot_count
            note = (f"C single row{row}: pending-slot predecessor read from "
                    "the captured plane unchanged.")
        experiments.append(make(
            f"{tag}{name}_C_single_row{row}", input_rows=[row],
            new_descriptors=new_desc, note=note, slots=slots,
            compact_kv=compact_kv, compact_scores=compact_scores,
        ))

    # Duplicate rows=2: [row1, row1] with wave geometry (row1 pools row0).
    experiments.append(make(
        f"{tag}{name}_C_duplicate_row1", input_rows=[1, 1],
        new_descriptors=[DESCRIPTOR_SENTINEL, slot_count],
        note=("C duplicate: candidate row 1 twice under rows=2 wave geometry; "
              "the pooling row now pools its own duplicate."),
        slots=slot_count,
    ))

    # Swapped order rows=2: [row1, row0].  An earlier-wave reference cannot
    # point forward, so the pooling row's predecessor is preserved through a
    # compact pending slot holding row 0's exact captured FP32 bytes.
    swapped = remap_batch(
        [descriptors[1], descriptors[0]], [1, 0],
        original_slot_count=slot_count, row_map={0: 1, 1: 0},
        pending_kv=pending_kv, pending_scores=pending_scores,
        strategy="compact_pending", compact_slots={0: 0},
    )
    swapped_kv, swapped_scores = build_compact_planes(
        swapped.source_map, pending_kv=pending_kv, pending_scores=pending_scores,
        wave_kv_rows={0: row_slice(projected, ROW_BYTES["projected"], 0)},
        wave_scores_rows={0: row_slice(scores, ROW_BYTES["scores"], 0)},
        original_slot_count=slot_count,
    )
    experiments.append(make(
        f"{tag}{name}_C_swapped_order", input_rows=[1, 0],
        new_descriptors=list(swapped.descriptors),
        note=("C swapped order: rows [1, 0]; row 1's earlier-wave predecessor "
              "preserved via compact pending slot (a forward earlier-wave "
              f"reference is rejected, never clamped). {swapped.provenance}"),
        slots=swapped.slot_count,
        compact_kv=swapped_kv, compact_scores=swapped_scores,
    ))

    # Mixed reference/candidate rows=2 (needs a reference capture).
    if reference is not None and reference.rows == 2:
        ref_input = reference.read_buffer("input")
        ref_name = reference.directory.name
        mixed_input = (row_slice(ref_input, ROW_BYTES["input"], 0)
                       + row_slice(input_blob, ROW_BYTES["input"], 1))
        experiments.append(PlannedExperiment(
            name=f"{tag}{name}_C_mixed_{ref_name}_row0", kind="C", rows=2,
            slots=slot_count, counterfactual=True,
            operands=[
                _operand("input", mixed_input, "bfloat16", (2, 5120),
                         f"mix:{ref_name}:input row0 + {name}:input row1"),
                _operand("wkv", wkv, "bfloat16", (512, 5120),
                         f"capture:{name}:wkv-weight"),
                _operand("wgate", wgate, "bfloat16", (512, 5120),
                         f"capture:{name}:wgate-weight"),
            ] + downstream_for([DESCRIPTOR_SENTINEL, slot_count],
                               slots=slot_count)[1],
            stages=_project_stages() + _pool_pack_stages(),
            outputs=("projected", "scores", "output", "values", "scales"),
            expected={},
            notes=(f"C mixed: {ref_name} row 0 (previous) with {name} row 1 "
                   "(current) under candidate wave geometry; the pooling row "
                   "now pools the reference previous row.",),
        ))
    return experiments


def _u64_blob(values) -> bytes:
    return b"".join(int(v).to_bytes(8, "little") for v in values)


# ---------------------------------------------------------------------------
# Family D (reference vs candidate cross-substitution counterfactuals).
# ---------------------------------------------------------------------------

def plan_cross_substitution(reference: Capture, candidate: Capture,
                            tag: str) -> List[PlannedExperiment]:
    """Compact pool+pack fixtures at the FP32 operand level.

    Row roles in a two-row decode batch: row 0 is the *previous* row
    (sentinel descriptor), row 1 the *current* row pooling row 0.  Variants:
    both originals (baselines within D), current-only swap, previous-only
    swap, duplicate-previous, duplicate-current and swapped order.
    """
    if reference.rows != 2 or candidate.rows != 2:
        return []
    ref_name, cand_name = reference.directory.name, candidate.directory.name
    fixtures = []
    for role, capture in (("reference", reference), ("candidate", candidate)):
        fixtures.append({
            "role": role,
            "name": capture.directory.name,
            "kv": capture.read_buffer("projected"),
            "scores": capture.read_buffer("scores"),
            "descriptors": capture.device_descriptors,
            "pending_kv": capture.read_buffer("pending-kv"),
            "pending_scores": capture.read_buffer("pending-scores"),
            "norm": capture.read_weight("norm"),
            "frequencies": capture.read_buffer("frequencies"),
            "slots": capture.slot_count,
            "output": capture.read_buffer("output"),
            "values": capture.read_buffer("kv-values"),
            "scales": capture.read_buffer("kv-scales"),
        })
    ref, cand = fixtures

    def rows_of(blob, row_bytes):
        return [row_slice(blob, row_bytes, 0), row_slice(blob, row_bytes, 1)]

    ref_kv, ref_sc = rows_of(ref["kv"], ROW_BYTES["projected"]), rows_of(
        ref["scores"], ROW_BYTES["scores"])
    cand_kv, cand_sc = rows_of(cand["kv"], ROW_BYTES["projected"]), rows_of(
        cand["scores"], ROW_BYTES["scores"])

    def build(exp_name, *, kv_rows, scores_rows, descriptors, base, note,
              counterfactual, compact_kv=b"", compact_scores=b"",
              slots_override=None):
        rows = 2
        slots = slots_override or base["slots"]
        pend_kv = compact_kv or base["pending_kv"]
        pend_scores = compact_scores or base["pending_scores"]
        pend_src = (f"capture:{base['name']}:pending-kv" if not compact_kv
                    else "compact:predecessor rows")
        expected = {}
        if not counterfactual:
            expected = {"output": base["output"], "values": base["values"],
                        "scales": base["scales"]}
        return PlannedExperiment(
            name=exp_name, kind="D", rows=rows, slots=slots,
            counterfactual=counterfactual,
            operands=[
                _operand("kv", b"".join(r[0] for r in kv_rows), "float32",
                         (rows, 512),
                         f"mix:kv rows {[r[1] for r in kv_rows]}"),
                _operand("scores", b"".join(r[0] for r in scores_rows),
                         "float32", (rows, 512),
                         f"mix:scores rows {[r[1] for r in scores_rows]}"),
                _operand("pending_kv", pend_kv, "float32",
                         (slots, 512), pend_src),
                _operand("pending_scores", pend_scores, "float32",
                         (slots, 512), pend_src),
                _operand("predecessors", _u64_blob(descriptors), "uint64",
                         (rows,), f"capture:{base['name']}:descriptors-device"),
                _operand("norm", base["norm"], "bfloat16", (512,),
                         f"capture:{base['name']}:norm-weight"),
                _operand("frequencies", base["frequencies"], "float32",
                         (rows, 32, 2),
                         f"capture:{base['name']}:frequencies"),
            ],
            stages=_pool_pack_stages(),
            outputs=("output", "values", "scales"),
            expected=expected,
            notes=(note,),
        )

    experiments = [
        build(f"{tag}D_{ref_name}_original", kv_rows=[(ref_kv[0], 0),
                                                      (ref_kv[1], 1)],
              scores_rows=[(ref_sc[0], 0), (ref_sc[1], 1)],
              descriptors=ref["descriptors"], base=ref, counterfactual=False,
              note="D reference original (baseline within D)."),
        build(f"{tag}D_{cand_name}_original", kv_rows=[(cand_kv[0], 0),
                                                       (cand_kv[1], 1)],
              scores_rows=[(cand_sc[0], 0), (cand_sc[1], 1)],
              descriptors=cand["descriptors"], base=cand, counterfactual=False,
              note="D candidate original (baseline within D)."),
        build(f"{tag}D_current_only_swap", kv_rows=[(ref_kv[0], 0),
                                                    (cand_kv[1], 1)],
              scores_rows=[(ref_sc[0], 0), (cand_sc[1], 1)],
              descriptors=ref["descriptors"], base=ref, counterfactual=True,
              note=("D intervention: candidate current row substituted into "
                    "the reference fixture; previous row stays reference.")),
        build(f"{tag}D_previous_only_swap", kv_rows=[(cand_kv[0], 0),
                                                     (ref_kv[1], 1)],
              scores_rows=[(cand_sc[0], 0), (ref_sc[1], 1)],
              descriptors=ref["descriptors"], base=ref, counterfactual=True,
              note=("D intervention: candidate previous row substituted into "
                    "the reference fixture; current row stays reference.")),
        build(f"{tag}D_duplicate_previous", kv_rows=[(ref_kv[0], 0),
                                                     (ref_kv[0], 0)],
              scores_rows=[(ref_sc[0], 0), (ref_sc[0], 0)],
              descriptors=ref["descriptors"], base=ref, counterfactual=True,
              note=("D intervention: previous row duplicated; the current row "
                    "pools a copy of the previous row.")),
        build(f"{tag}D_duplicate_current", kv_rows=[(ref_kv[1], 1),
                                                    (ref_kv[1], 1)],
              scores_rows=[(ref_sc[1], 1), (ref_sc[1], 1)],
              descriptors=ref["descriptors"], base=ref, counterfactual=True,
              note=("D intervention: current row duplicated into both slots; "
                    "row 0 carries the sentinel so only row 1 emits.")),
    ]

    # Swapped row order [1, 0]: the pooling row's earlier-wave predecessor
    # cannot point forward, so it is preserved via a compact pending slot.
    swapped = remap_batch(
        list(ref["descriptors"]), [0, 1], original_slot_count=ref["slots"],
        row_map={0: 1, 1: 0}, pending_kv=ref["pending_kv"],
        pending_scores=ref["pending_scores"], strategy="compact_pending",
        compact_slots={0: 0},
    )
    swapped_kv, swapped_scores = build_compact_planes(
        swapped.source_map, pending_kv=ref["pending_kv"],
        pending_scores=ref["pending_scores"],
        wave_kv_rows={0: ref_kv[0]}, wave_scores_rows={0: ref_sc[0]},
        original_slot_count=ref["slots"],
    )
    experiments.append(build(
        f"{tag}D_swapped_order", kv_rows=[(ref_kv[1], 1), (ref_kv[0], 0)],
        scores_rows=[(ref_sc[1], 1), (ref_sc[0], 0)],
        descriptors=list(swapped.descriptors), base=ref, counterfactual=True,
        compact_kv=swapped_kv, compact_scores=swapped_scores,
        slots_override=swapped.slot_count,
        note=("D intervention: reference rows swapped; row 1's predecessor "
              "preserved via compact pending slot "
              f"({swapped.provenance})."),
    ))
    return experiments


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
        mutated[-1] ^= 0x01  # BF16 low bit of the row's last input value
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

def select_reference(captures: Sequence[Capture]) -> Optional[Capture]:
    """The reference capture: exactly one directory whose name mentions
    'reference', or None (single-capture replay then plans A/B only)."""
    refs = [c for c in captures if "reference" in c.directory.name.lower()]
    return refs[0] if len(refs) == 1 else None


def plan_experiments(captures: Sequence[Capture], *, tag: str = "") -> List[PlannedExperiment]:
    """Full experiment list for a set of validated captures.

    A/B run per capture.  C runs for every two-row capture.  D requires
    exactly one identifiable reference plus at least one two-row candidate;
    otherwise it is omitted (recorded in the plan notes, never fabricated).
    """
    experiments: List[PlannedExperiment] = []
    for capture in captures:
        experiments += plan_capture_baselines(capture, tag)
        experiments.append(plan_mutation_variant(capture, tag))
        experiments += plan_padded_project_only(capture, tag)
    reference = select_reference(captures)
    candidates = [c for c in captures if c is not reference]
    for capture in candidates:
        experiments += plan_candidate_geometry(capture, reference, tag)
    two_row = [c for c in candidates if c.rows == 2]
    if reference is not None and reference.rows == 2 and two_row:
        for candidate in two_row:
            experiments += plan_cross_substitution(reference, candidate, tag)
    return experiments
