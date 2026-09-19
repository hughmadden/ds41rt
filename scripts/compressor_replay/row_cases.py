"""Replacement compressor row-case planners (families C and D).

These planners replace the C/D families of ``experiments.py`` under the
semantics the root established after rejecting the draft version:

* A two-row candidate capture is TWO INDEPENDENT REQUESTS.  Both rows sit at
  absolute position 41 (each completing its own latent token pair 40/41,
  logical compressed row 20) and each pools its OWN previous pending row.
  They are NOT "previous token / current token of one wave".
* A device descriptor is an *address*.  ``desc < slot_count`` addresses the
  pending plane row ``desc`` (that request's own leased slot holding its p40
  pending state); ``desc >= slot_count`` addresses the earlier wave row
  ``desc - slot_count`` of the same chunk at the previous absolute position.
  Row 0 of a batch does NOT imply token 40.
* The reference capture ``lane0-batch20-Full-1rows`` has ONE row (request
  2026091950, p41, descriptor 0).  Family D must produce real cases for a
  1-row reference paired with each 2-row candidate (``lane0-batch81``:
  requests 2026091960/1961, descriptors 0/1; ``lane1-batch82``: requests
  2026091962/1963, descriptors 2/3).

Every regrouped batch is materialized as a *compact pool fixture*: for each
selected logical output row ``r`` the authoritative previous FP32 KV/scores
bytes are copied into compact pending slot ``r`` and ``descriptor[r] = r``
with ``slots = len(rows)`` — a content-preserving address remap.  Duplicate /
mixed / swapped cases duplicate or permute the ENTIRE row bundle (current
operands, previous pending content and the frequency slice), not just the
current input.

WKV is projected into the ``projected`` role and Wgate into ``scores``; the
pool stage consumes ``projected`` (the draft B2/C bug referenced a ``kv``
role that no stage ever produced).

Only C/D live here.  The two public planners are wire-compatible with
``runner.PlannedExperiment`` / ``runner.PlannedOperand``; nothing is imported
from ``experiments.py`` (no import cycles).  Captures are the shared, fully
validated ``schema.Capture`` objects produced by ``schema.load_captures`` /
``schema.load_capture``; there is no private capture adapter.  Structural plan
checks use the shared ``validation.validate_experiment``.  CPU only: no torch,
no CUDA, and no native library.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .runner import PlannedExperiment, PlannedOperand
from .schema import (
    DESCRIPTOR_SENTINEL,
    INVALID_SENTINEL,
    LATENT_DIM,
    PENDING_ROW_BYTES,
    PENDING_SLOT,
    ROW_BYTES,
    SOURCE_DIM,
    UNRESOLVED,
    resolve_predecessor,
)
from .validation import validate_experiment

RATIO = 2

# Roles the runner allocates/downloads for these families.
C_OUTPUTS = ("projected", "scores", "output", "values", "scales")
D_OUTPUTS = ("output", "values", "scales")

FREQ_ROW_BYTES = 32 * 2 * 4                 # FP32 [1, 32, 2]

WEIGHT_ROLES = ("wkv", "wgate", "norm")


class RowCaseError(Exception):
    """A row case cannot be planned from the actual capture state.

    Raised instead of any silent substitution, clamping, or reinterpretation
    of a descriptor, lease, weight or frequency mismatch.
    """


def _sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _row(blob: bytes, index: int, row_bytes: int) -> bytes:
    start = index * row_bytes
    if start + row_bytes > len(blob):
        raise RowCaseError(
            f"row {index} of {row_bytes} bytes exceeds buffer length {len(blob)}"
        )
    return blob[start:start + row_bytes]


def _u64_blob(values: Sequence[int]) -> bytes:
    return b"".join(int(v).to_bytes(8, "little") for v in values)


# ---------------------------------------------------------------------------
# Small explicit stage / operand constructors (nothing imported from
# experiments.py; these are the only stage shapes the runner understands).
# ---------------------------------------------------------------------------

def _operand(role: str, blob: bytes, dtype: str, shape, source: str
             ) -> PlannedOperand:
    return PlannedOperand(role=role, blob=bytes(blob), dtype=dtype,
                          shape=tuple(shape), source=source)


def _project_stages() -> List[dict]:
    """WKV -> ``projected`` and Wgate -> ``scores`` (the B2 role fix)."""
    return [
        {"kind": "project", "input": "input", "weight": "wkv",
         "output": "projected", "ratio": RATIO},
        {"kind": "project", "input": "input", "weight": "wgate",
         "output": "scores", "ratio": RATIO},
    ]


def _pool_pack_stages(slots: int) -> List[dict]:
    """Pool (consuming the ``projected`` role) then pack, on one stream."""
    return [
        {"kind": "pool", "kv": "projected", "scores": "scores",
         "pending_kv": "pending_kv", "pending_scores": "pending_scores",
         "predecessors": "predecessors", "norm": "norm",
         "output": "output", "slots": slots},
        {"kind": "pack", "input": "output", "frequencies": "frequencies",
         "values": "values", "scales": "scales"},
    ]


# ---------------------------------------------------------------------------
# Semantic predecessor resolution (chunks + wave_rows ownership; the device
# descriptor is the address, never reinterpreted).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PreviousOperand:
    """The authoritative previous FP32 KV/scores one selected row pools."""

    kind: str                    # "pending" | "wave"
    index: int                   # pending slot or earlier wave row
    kv: bytes                    # FP32 [512]
    scores: bytes                # FP32 [512]
    capture_name: str
    request_id: int
    position: int                # completing row's absolute position
    previous_position: int
    kv_source: str
    scores_source: str

    def identifying_tuple(self) -> tuple:
        return (self.capture_name, self.kind, self.index, self.request_id,
                self.position, self.previous_position)


def _chunk_of(capture: Any, row: int) -> dict:
    wave_row = capture.wave_rows[row]
    index = wave_row.get("chunk")
    for chunk in capture.chunks:
        if chunk.get("index") == index:
            return chunk
    raise RowCaseError(
        f"row {row} references unknown chunk {index!r} in "
        f"{capture.directory.name}"
    )


def _row_range(chunk: dict) -> Tuple[int, int]:
    if chunk.get("flattened_row_range"):
        lo, hi = chunk["flattened_row_range"]
        return lo, hi
    lo = chunk.get("prepared_offset", 0)
    return lo, lo + chunk.get("tokens", 1)


def odd_position_rows(capture: Any) -> List[int]:
    """Rows completing a latent (odd absolute position at ratio two).

    Even-position rows complete no latent and are never selected for row
    cases; an odd row carrying the sentinel is unsupported for selection.
    """
    selected = []
    for row in range(capture.rows):
        position = capture.wave_rows[row]["absolute_position"]
        if position % RATIO != 1:
            continue
        descriptor = capture.device_descriptors[row]
        if descriptor == DESCRIPTOR_SENTINEL:
            raise RowCaseError(
                f"{capture.directory.name} row {row}: selected odd position "
                f"{position} carries the sentinel descriptor; sentinel rows "
                "are unsupported for selected odd p41 rows"
            )
        latent = capture.wave_rows[row].get("completed_latent")
        if latent is None:
            raise RowCaseError(
                f"{capture.directory.name} row {row} at position {position} "
                "completes no latent"
            )
        first_token = position - 1
        if latent.get("first_token") != first_token or (
            latent.get("logical_compressed_row") != first_token // RATIO
        ):
            raise RowCaseError(
                f"{capture.directory.name} row {row}: completed latent "
                f"({latent.get('first_token')}, "
                f"{latent.get('logical_compressed_row')}) does not complete "
                f"the pair [{first_token}, {position}] at logical row "
                f"{first_token // RATIO}"
            )
        selected.append(row)
    return selected


def resolve_row_previous(capture: Any, row: int) -> PreviousOperand:
    """Resolve one row's previous pooled operand from the ACTUAL descriptor.

    ``desc < slot_count`` -> the pending plane row ``desc`` (must be this
    row's own chunk lease; using another request's active lease is rejected).
    ``desc >= slot_count`` -> the earlier wave row ``desc - slot_count`` (only
    same chunk and previous absolute position).  The sentinel is unsupported
    for selected odd rows.  No clamping, no host substitution, no cross-
    request borrowing.
    """
    name = capture.directory.name
    slot_count = capture.slot_count
    wave_row = capture.wave_rows[row]
    position = wave_row["absolute_position"]
    request_id = wave_row["request_id"]
    descriptor = capture.device_descriptors[row]
    if wave_row.get("device_descriptor") != descriptor:
        raise RowCaseError(
            f"{name} row {row}: device descriptor bytes {descriptor:#x} do "
            f"not match the manifest wave_rows entry "
            f"{wave_row.get('device_descriptor')!r}"
        )
    chunk = _chunk_of(capture, row)
    lo, hi = _row_range(chunk)
    if not lo <= row < hi:
        raise RowCaseError(
            f"{name} row {row} is not owned by chunk {chunk.get('index')} "
            f"(range [{lo}, {hi}))"
        )
    if chunk.get("request_id") != request_id:
        raise RowCaseError(
            f"{name} row {row} request {request_id} is not the owner of "
            f"chunk {chunk.get('index')} (request {chunk.get('request_id')})"
        )

    predecessor = resolve_predecessor(descriptor, row, slot_count)
    if predecessor["kind"] == INVALID_SENTINEL:
        raise RowCaseError(
            f"{name} row {row}: sentinel descriptor is unsupported for "
            f"selected odd position {position}"
        )
    if predecessor["kind"] == UNRESOLVED:
        raise RowCaseError(
            f"{name} row {row}: descriptor {descriptor:#x} is unresolvable "
            f"(slot_count {slot_count})"
        )

    if predecessor["kind"] == PENDING_SLOT:
        slot = predecessor["slot"]
        lease_slot = (chunk.get("lease") or {}).get("slot")
        if lease_slot != slot:
            raise RowCaseError(
                f"{name} row {row} (request {request_id}) pools pending slot "
                f"{slot} but its chunk leases slot {lease_slot}: using "
                "another request's pending state is rejected"
            )
        for other in capture.chunks:
            if other.get("index") != chunk.get("index"):
                other_slot = (other.get("lease") or {}).get("slot")
                if other_slot == slot:
                    raise RowCaseError(
                        f"{name} row {row}: pending slot {slot} is also "
                        f"leased by chunk {other.get('index')} (request "
                        f"{other.get('request_id')})"
                    )
        kv = _row(capture.read_buffer("pending-kv"), slot, PENDING_ROW_BYTES)
        scores = _row(capture.read_buffer("pending-scores"), slot,
                      PENDING_ROW_BYTES)
        return PreviousOperand(
            kind="pending", index=slot, kv=kv, scores=scores,
            capture_name=name, request_id=request_id, position=position,
            previous_position=position - 1,
            kv_source=f"capture:{name}:pending-kv slot{slot}",
            scores_source=f"capture:{name}:pending-scores slot{slot}",
        )

    # earlier wave row: only same chunk and previous absolute position
    earlier = predecessor["wave_row"]
    earlier_wave = capture.wave_rows[earlier]
    if earlier_wave.get("chunk") != chunk.get("index"):
        raise RowCaseError(
            f"{name} row {row}: earlier-wave row {earlier} belongs to chunk "
            f"{earlier_wave.get('chunk')}, not this row's chunk "
            f"{chunk.get('index')}: cross-request earlier-wave reference"
        )
    if earlier_wave.get("absolute_position") != position - 1:
        raise RowCaseError(
            f"{name} row {row}: earlier-wave row {earlier} sits at position "
            f"{earlier_wave.get('absolute_position')}, not the previous "
            f"absolute position {position - 1}"
        )
    kv = _row(capture.read_buffer("projected"), earlier, ROW_BYTES["projected"])
    scores = _row(capture.read_buffer("scores"), earlier, ROW_BYTES["scores"])
    return PreviousOperand(
        kind="wave", index=earlier, kv=kv, scores=scores,
        capture_name=name, request_id=request_id, position=position,
        previous_position=position - 1,
        kv_source=(f"capture:{name}:projected row{earlier} "
                   f"(earlier wave, same chunk, p{position - 1})"),
        scores_source=(f"capture:{name}:scores row{earlier} "
                       f"(earlier wave, same chunk, p{position - 1})"),
    )


# ---------------------------------------------------------------------------
# Row bundles: one selected row's complete operand set, with provenance.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RowBundle:
    """Everything one selected logical output row contributes to a fixture."""

    capture: Any
    row: int
    request_id: int
    position: int
    input_row: bytes            # BF16 [5120] (current wave input)
    projected_row: bytes        # FP32 [512] (captured WKV projection)
    scores_row: bytes           # FP32 [512] (captured Wgate projection)
    freq_row: bytes             # FP32 [32, 2]
    previous: PreviousOperand

    def current_source(self) -> dict:
        return {
            "capture": self.capture.directory.name, "row": self.row,
            "request_id": self.request_id, "position": self.position,
            "input_sha256": _sha256(self.input_row),
            "projected_sha256": _sha256(self.projected_row),
            "scores_sha256": _sha256(self.scores_row),
            "freq_sha256": _sha256(self.freq_row),
        }

    def previous_source(self) -> dict:
        prev = self.previous
        return {
            "capture": prev.capture_name, "kind": prev.kind,
            "index": prev.index, "request_id": prev.request_id,
            "position": prev.position, "previous_position": prev.previous_position,
            "kv_sha256": _sha256(prev.kv), "scores_sha256": _sha256(prev.scores),
            "identifying_tuple": list(prev.identifying_tuple()),
        }


def build_row_bundle(capture: Any, row: int) -> RowBundle:
    wave_row = capture.wave_rows[row]
    previous = resolve_row_previous(capture, row)
    return RowBundle(
        capture=capture, row=row, request_id=wave_row["request_id"],
        position=wave_row["absolute_position"],
        input_row=_row(capture.read_buffer("input"), row, ROW_BYTES["input"]),
        projected_row=_row(capture.read_buffer("projected"), row,
                           ROW_BYTES["projected"]),
        scores_row=_row(capture.read_buffer("scores"), row, ROW_BYTES["scores"]),
        freq_row=_row(capture.read_buffer("frequencies"), row, FREQ_ROW_BYTES),
        previous=previous,
    )


def _read_weights(capture: Any) -> Dict[str, bytes]:
    return {role: capture.read_weight(role) for role in WEIGHT_ROLES}


def assert_weights_compatible(*captures: Any) -> Dict[str, str]:
    """WKV/Wgate/norm must match byte-for-byte across all sources.

    Returns the shared hashes; raises :class:`RowCaseError` (never silently
    mixes weights) listing the offending role and captures.
    """
    base = captures[0]
    base_weights = _read_weights(base)
    for other in captures[1:]:
        other_weights = _read_weights(other)
        for role in WEIGHT_ROLES:
            if base_weights[role] != other_weights[role]:
                raise RowCaseError(
                    f"{role} weight differs between {base.directory.name} and "
                    f"{other.directory.name} "
                    f"({_sha256(base_weights[role])[:12]} vs "
                    f"{_sha256(other_weights[role])[:12]}): weights are not "
                    "silently mixed"
                )
    return {role: _sha256(blob) for role, blob in base_weights.items()}


def _assert_p41_freq_match(a: RowBundle, b: RowBundle) -> None:
    if a.freq_row != b.freq_row:
        raise RowCaseError(
            f"p{a.position} frequency slice of {a.capture.directory.name} "
            f"row {a.row} ({_sha256(a.freq_row)[:12]}) does not match "
            f"{b.capture.directory.name} row {b.row} "
            f"({_sha256(b.freq_row)[:12]}): combining R/C sources requires "
            "byte-identical p41 frequencies so the swap stays the only "
            "variable"
        )


def _output_oracle_row(capture: Any, row: int) -> Dict[str, bytes]:
    """The captured downstream oracle for one output row."""
    return {
        "output": _row(capture.read_buffer("output"), row, ROW_BYTES["output"]),
        "values": _row(capture.read_buffer("kv-values"), row,
                       ROW_BYTES["kv-values"]),
        "scales": _row(capture.read_buffer("kv-scales"), row,
                       ROW_BYTES["kv-scales"]),
    }


def _provenance_note(current: List[dict], previous: List[dict], **extra
                     ) -> str:
    payload = {
        "current_sources": current,
        "previous_sources": previous,
        **extra,
    }
    return "provenance:" + json.dumps(payload, sort_keys=True)


# ---------------------------------------------------------------------------
# Family C: candidate-geometry counterfactuals.
# ---------------------------------------------------------------------------

def plan_candidate_geometry(capture: Any, reference: Optional[Any],
                            tag: str) -> List[PlannedExperiment]:
    """Candidate-geometry row cases for one candidate capture.

    * independent rows=1 reruns of each actual candidate row;
    * duplicates ``[0, 0]`` and ``[1, 1]`` and swapped ``[1, 0]`` batches;
    * mixed ``[reference0, candidate0]`` and swapped ``[candidate0,
      reference0]`` batches when a reference capture is given.

    Every case projects the selected current input rows through WKV/Wgate
    (``projected`` / ``scores``), then pools+packs against a compact pending
    plane: output row ``r`` pools compact slot ``r``, which holds that row's
    authoritative previous FP32 bytes (its own pending slot, or the earlier
    wave row's captured projection).  Duplicate/mixed/swapped cases
    duplicate or permute the ENTIRE row bundle — current input, previous
    pending content and the per-row frequency slice.  All cases are explicit
    counterfactuals: no expected gate, no accuracy claim.
    """
    name = capture.directory.name
    selected = odd_position_rows(capture)
    if not selected:
        return []
    weights = _read_weights(capture)
    weight_hashes = {role: _sha256(blob) for role, blob in weights.items()}
    bundles = {row: build_row_bundle(capture, row) for row in selected}

    reference_bundle = None
    reference_weights_hashes = None
    if reference is not None:
        reference_selected = odd_position_rows(reference)
        if reference_selected:
            reference_weights_hashes = assert_weights_compatible(
                capture, reference)
            reference_bundle = build_row_bundle(reference,
                                                reference_selected[0])

    experiments: List[PlannedExperiment] = []

    def make_case(exp_name: str, case_bundles: List[RowBundle], summary: str,
                  extra: dict) -> PlannedExperiment:
        rows = len(case_bundles)
        input_blob = b"".join(b.input_row for b in case_bundles)
        pending_kv = b"".join(b.previous.kv for b in case_bundles)
        pending_scores = b"".join(b.previous.scores for b in case_bundles)
        frequencies = b"".join(b.freq_row for b in case_bundles)
        current_sources = [b.current_source() for b in case_bundles]
        previous_sources = [b.previous_source() for b in case_bundles]
        operands = [
            _operand("input", input_blob, "bfloat16", (rows, SOURCE_DIM),
                     f"capture: input rows "
                     f"{[b.capture.directory.name + ':' + str(b.row) for b in case_bundles]}"),
            _operand("wkv", weights["wkv"], "bfloat16", (LATENT_DIM, SOURCE_DIM),
                     f"capture:{name}:wkv-weight"),
            _operand("wgate", weights["wgate"], "bfloat16",
                     (LATENT_DIM, SOURCE_DIM),
                     f"capture:{name}:wgate-weight"),
            _operand("pending_kv", pending_kv, "float32", (rows, LATENT_DIM),
                     "compact: " + "; ".join(
                         f"row{i} <- {b.previous.kv_source}"
                         for i, b in enumerate(case_bundles))),
            _operand("pending_scores", pending_scores, "float32",
                     (rows, LATENT_DIM),
                     "compact: " + "; ".join(
                         f"row{i} <- {b.previous.scores_source}"
                         for i, b in enumerate(case_bundles))),
            _operand("predecessors", _u64_blob(range(rows)), "uint64",
                     (rows,), "remap: compact descriptor[r] = r "
                              "(content-preserving address remap)"),
            _operand("norm", weights["norm"], "bfloat16", (LATENT_DIM,),
                     f"capture:{name}:norm-weight"),
            _operand("frequencies", frequencies, "float32", (rows, 32, 2),
                     "compact: " + "; ".join(
                         f"row{i} <- {b.capture.directory.name}:frequencies "
                         f"row{b.row}" for i, b in enumerate(case_bundles))),
        ]
        note = _provenance_note(
            current_sources, previous_sources,
            weights=weight_hashes,
            reference_weights=reference_weights_hashes,
            input_sha256=_sha256(input_blob),
            pending_kv_sha256=_sha256(pending_kv),
            pending_scores_sha256=_sha256(pending_scores),
            frequencies_sha256=_sha256(frequencies),
            compact_descriptors=list(range(rows)),
            slots=rows,
            **extra,
        )
        experiment = PlannedExperiment(
            name=exp_name, kind="C", rows=rows, slots=rows,
            counterfactual=True, operands=operands,
            stages=_project_stages() + _pool_pack_stages(rows),
            outputs=C_OUTPUTS, expected={},
            notes=(summary, note),
        )
        validate_experiment(experiment)
        return experiment

    for row in selected:
        bundle = bundles[row]
        experiments.append(make_case(
            f"{tag}C_{name}_row{row}_single", [bundle],
            f"C counterfactual: independent rows=1 rerun of {name} row {row} "
            f"(request {bundle.request_id}, p{bundle.position}); compact "
            f"pending slot 0 holds its previous {bundle.previous.kind} "
            f"operand ({bundle.previous.kv_source}).",
            {"case": "single", "selected_rows": [row]},
        ))
        experiments.append(make_case(
            f"{tag}C_{name}_row{row}_duplicate", [bundle, bundle],
            f"C counterfactual: {name} row {row} duplicated as [row, row]; "
            "the ENTIRE bundle (input, previous pending content, frequency "
            "slice) is duplicated, each output row pooling its own compact "
            "copy.",
            {"case": "duplicate", "selected_rows": [row, row]},
        ))

    if len(selected) >= 2:
        reversed_rows = list(reversed(selected))
        experiments.append(make_case(
            f"{tag}C_{name}_swapped_{''.join(str(r) for r in reversed_rows)}",
            [bundles[r] for r in reversed_rows],
            f"C counterfactual: {name} rows swapped to {reversed_rows}; "
            "row order changes, each row's own previous operand and "
            "frequency slice travel with it.",
            {"case": "swapped", "selected_rows": reversed_rows},
        ))

    if reference_bundle is not None:
        reference_name = reference_bundle.capture.directory.name
        reference_row = reference_bundle.row
        first = bundles[selected[0]]
        experiments.append(make_case(
            f"{tag}C_{name}_mixed_{reference_name}_row{reference_row}"
            f"_plus_row{first.row}",
            [reference_bundle, first],
            f"C counterfactual: mixed rows=2 [{reference_name} row "
            f"{reference_row} (reference), {name} row {first.row} "
            f"(candidate)]; each row keeps its own previous pending content "
            "and frequency slice; weights are byte-identical across both "
            "captures.",
            {"case": "mixed", "selected_rows": [
                f"{reference_name}:{reference_row}", f"{name}:{first.row}"]},
        ))
        experiments.append(make_case(
            f"{tag}C_{name}_mixed_row{first.row}_plus_{reference_name}"
            f"_row{reference_row}",
            [first, reference_bundle],
            f"C counterfactual: mixed rows=2 swapped [{name} row {first.row} "
            f"(candidate), {reference_name} row {reference_row} "
            "(reference)].",
            {"case": "mixed_swapped", "selected_rows": [
                f"{name}:{first.row}", f"{reference_name}:{reference_row}"]},
        ))

    _assert_unique_names(experiments)
    return experiments


# ---------------------------------------------------------------------------
# Family D: reference-vs-candidate cross substitution.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _DRow:
    """One output row of a D fixture: a current bundle plus a previous."""

    current: RowBundle
    previous: PreviousOperand


def plan_cross_substitution(reference: Any, candidate: Any,
                            tag: str) -> List[PlannedExperiment]:
    """Cross-substitution row cases between reference row 0 and each
    candidate row.

    For reference row R and candidate row C, compact single-row pool+pack
    combinations (current, previous):

    * ``(R, R)`` and ``(C, C)`` originals — gated baselines comparing the
      exact captured output / kv-values / kv-scales of that row;
    * ``(C, R)`` current-only swap and ``(R, C)`` previous-only swap —
      interventions, counterfactual, never gated.

    Plus batched bundles of these variants (candidate originals, duplicated
    reference original, swapped originals, and an original+previous-swap
    mix).  Bundles made solely of original (R,R)/(C,C) bytes gate against the
    concatenated captured oracles; anything mixing an intervention is a
    counterfactual.  Every output row has its own compact pending slot.
    Norm and the frequency slice come from the combo's original (current)
    row; combining R and C in one row requires the two sources' p41
    frequency slices to be byte-identical, so the swap stays the only
    variable.
    """
    reference_selected = odd_position_rows(reference)
    if not reference_selected:
        raise RowCaseError(
            f"reference {reference.directory.name} exposes no odd-position "
            "row; family D needs a reference current row"
        )
    candidate_selected = odd_position_rows(candidate)
    if not candidate_selected:
        raise RowCaseError(
            f"candidate {candidate.directory.name} exposes no odd-position row"
        )
    assert_weights_compatible(reference, candidate)

    ref_name = reference.directory.name
    cand_name = candidate.directory.name
    ref_row = reference_selected[0]
    ref_bundle = build_row_bundle(reference, ref_row)
    experiments: List[PlannedExperiment] = []

    def oracle(bundle: RowBundle) -> Dict[str, bytes]:
        return _output_oracle_row(bundle.capture, bundle.row)

    def make_case(exp_name: str, drows: List[_DRow], gated: bool,
                  summary: str, extra: dict) -> PlannedExperiment:
        rows = len(drows)
        kv_blob = b"".join(d.current.projected_row for d in drows)
        scores_blob = b"".join(d.current.scores_row for d in drows)
        pending_kv = b"".join(d.previous.kv for d in drows)
        pending_scores = b"".join(d.previous.scores for d in drows)
        frequencies = b"".join(d.current.freq_row for d in drows)
        norm = _read_weights(drows[0].current.capture)["norm"]
        expected: Dict[str, bytes] = {}
        if gated:
            row_oracles = [oracle(d.current) for d in drows]
            for role in D_OUTPUTS:
                expected[role] = b"".join(o[role] for o in row_oracles)
        note = _provenance_note(
            [d.current.current_source() for d in drows],
            [
                {
                    "kind": d.previous.kind, "index": d.previous.index,
                    "capture": d.previous.capture_name,
                    "request_id": d.previous.request_id,
                    "kv_sha256": _sha256(d.previous.kv),
                    "scores_sha256": _sha256(d.previous.scores),
                    "identifying_tuple": list(d.previous.identifying_tuple()),
                }
                for d in drows
            ],
            weights={role: _sha256(blob) for role, blob
                     in _read_weights(drows[0].current.capture).items()},
            norm_source=f"capture:{drows[0].current.capture.directory.name}"
                        f":norm-weight",
            freq_source="; ".join(
                f"row{i} <- {d.current.capture.directory.name}:frequencies "
                f"row{d.current.row}" for i, d in enumerate(drows)),
            current_sha256=_sha256(kv_blob),
            pending_kv_sha256=_sha256(pending_kv),
            pending_scores_sha256=_sha256(pending_scores),
            gated=gated,
            expected_sha256={role: _sha256(blob)
                             for role, blob in expected.items()},
            compact_descriptors=list(range(rows)),
            slots=rows,
            **extra,
        )
        experiment = PlannedExperiment(
            name=exp_name, kind="D", rows=rows, slots=rows,
            counterfactual=not gated,
            operands=[
                _operand("projected", kv_blob, "float32", (rows, LATENT_DIM),
                         "compact: " + "; ".join(
                             f"row{i} <- {d.current.capture.directory.name}:"
                             f"projected row{d.current.row}"
                             for i, d in enumerate(drows))),
                _operand("scores", scores_blob, "float32", (rows, LATENT_DIM),
                         "compact: " + "; ".join(
                             f"row{i} <- {d.current.capture.directory.name}:"
                             f"scores row{d.current.row}"
                             for i, d in enumerate(drows))),
                _operand("pending_kv", pending_kv, "float32",
                         (rows, LATENT_DIM),
                         "compact: " + "; ".join(
                             f"row{i} <- {d.previous.kv_source}"
                             for i, d in enumerate(drows))),
                _operand("pending_scores", pending_scores, "float32",
                         (rows, LATENT_DIM),
                         "compact: " + "; ".join(
                             f"row{i} <- {d.previous.scores_source}"
                             for i, d in enumerate(drows))),
                _operand("predecessors", _u64_blob(range(rows)), "uint64",
                         (rows,),
                         "remap: compact descriptor[r] = r "
                         "(content-preserving address remap)"),
                _operand("norm", norm, "bfloat16", (LATENT_DIM,),
                         f"capture:{drows[0].current.capture.directory.name}"
                         f":norm-weight"),
                _operand("frequencies", frequencies, "float32",
                         (rows, 32, 2),
                         "compact: " + "; ".join(
                             f"row{i} <- {d.current.capture.directory.name}:"
                             f"frequencies row{d.current.row}"
                             for i, d in enumerate(drows))),
            ],
            stages=_pool_pack_stages(rows),
            outputs=D_OUTPUTS, expected=expected,
            notes=(summary, note),
        )
        validate_experiment(experiment)
        return experiment

    for row in candidate_selected:
        cand_bundle = build_row_bundle(candidate, row)
        # Combining R and C content inside one row requires matching p41
        # frequencies (norm already byte-identical by the weights check).
        _assert_p41_freq_match(ref_bundle, cand_bundle)
        variants = [
            ("curR_prevR", ref_bundle, ref_bundle.previous, False,
             f"D baseline: reference original (R, R) — {ref_name} row "
             f"{ref_row} current and its own previous; gates against the "
             "captured reference oracle."),
            ("curC_prevC", cand_bundle, cand_bundle.previous, False,
             f"D baseline: candidate original (C, C) — {cand_name} row {row} "
             "current and its own previous; gates against the captured "
             "candidate oracle."),
            ("curC_prevR", cand_bundle, ref_bundle.previous, True,
             f"D intervention: current-only swap (C, R) — {cand_name} row "
             f"{row} current pooling the reference previous operand; "
             "counterfactual, never gated."),
            ("curR_prevC", ref_bundle, cand_bundle.previous, True,
             f"D intervention: previous-only swap (R, C) — reference current "
             f"pooling {cand_name} row {row}'s previous operand; "
             "counterfactual, never gated."),
        ]
        for suffix, current, previous, counterfactual, summary in variants:
            experiments.append(make_case(
                f"{tag}D_{ref_name}_vs_{cand_name}_row{row}_{suffix}",
                [_DRow(current=current, previous=previous)],
                gated=not counterfactual, summary=summary,
                extra={"case": suffix, "candidate_row": row,
                       "reference_row": ref_row},
            ))

    # Batched bundles (bounded: four per reference/candidate pair).
    first_cand = build_row_bundle(candidate, candidate_selected[0])
    _assert_p41_freq_match(ref_bundle, first_cand)

    cand_originals = [
        _DRow(current=bundle, previous=bundle.previous)
        for bundle in (
            build_row_bundle(candidate, row) for row in candidate_selected
        )
    ]
    experiments.append(make_case(
        f"{tag}D_{ref_name}_vs_{cand_name}_bundle_cand_originals",
        cand_originals, gated=True,
        summary=(f"D bundle: every candidate original (C, C) row "
                 f"{candidate_selected} of {cand_name} as one batch, each "
                 "output row pooling its own compact pending slot; gates "
                 "against the concatenated captured candidate oracle."),
        extra={"case": "bundle_cand_originals",
               "candidate_rows": candidate_selected},
    ))

    ref_original = _DRow(current=ref_bundle, previous=ref_bundle.previous)
    experiments.append(make_case(
        f"{tag}D_{ref_name}_vs_{cand_name}_bundle_duplicate_ref_original",
        [ref_original, ref_original], gated=True,
        summary=("D bundle: reference original (R, R) duplicated; solely "
                 "original bytes, gates against the captured reference "
                 "oracle twice."),
        extra={"case": "bundle_duplicate_ref_original",
               "reference_row": ref_row},
    ))

    first_cand_original = cand_originals[0]
    experiments.append(make_case(
        f"{tag}D_{ref_name}_vs_{cand_name}_bundle_row"
        f"{candidate_selected[0]}_swapped_originals",
        [first_cand_original, ref_original], gated=True,
        summary=(f"D bundle: swapped originals [(C, C) row "
                 f"{candidate_selected[0]}, (R, R)]; solely original bytes, "
                 "gates against the concatenated captured oracles in the "
                 "swapped order."),
        extra={"case": "bundle_swapped_originals",
               "candidate_row": candidate_selected[0],
               "reference_row": ref_row},
    ))

    prev_swap_row = _DRow(current=ref_bundle,
                          previous=first_cand.previous)
    experiments.append(make_case(
        f"{tag}D_{ref_name}_vs_{cand_name}_bundle_row"
        f"{candidate_selected[0]}_original_plus_prevswap",
        [ref_original, prev_swap_row], gated=False,
        summary=(f"D bundle: original (R, R) followed by the previous-only "
                 f"swap (R, C row {candidate_selected[0]}); mixed "
                 "intervention, counterfactual, never gated."),
        extra={"case": "bundle_original_plus_prevswap",
               "candidate_row": candidate_selected[0],
               "reference_row": ref_row},
    ))

    _assert_unique_names(experiments)
    return experiments


def _assert_unique_names(experiments: List[PlannedExperiment]) -> None:
    names = [e.name for e in experiments]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise RowCaseError(f"experiment name alias collision: {duplicates}")


