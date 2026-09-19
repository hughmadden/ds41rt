"""Bounded, explicit replay case list (reference, candidates, counterfactuals).

Every case names its rows by (wave, request_id) and, for counterfactuals,
the single private-source nibble change plus which captured output each row
is compared against.  The plan is fully deterministic: the same case list is
used by the CPU validator and the GPU runner, and per-row expected output
hashes are recorded so batch results cannot be faked by duplicating
reference outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Capture keys.
REF = ("lane0-batch20-Full-1rows", 2026091950)
CANDIDATE_LANE0 = ("lane0-batch81-Full-2rows", 2026091960)
CANDIDATE_LANE0_ROW1 = ("lane0-batch81-Full-2rows", 2026091961)
CANDIDATE_LANE1 = ("lane1-batch82-Full-2rows", 2026091962)
CANDIDATE_LANE1_ROW1 = ("lane1-batch82-Full-2rows", 2026091963)

# Counterfactual variants.
CANDIDATE_TO_REFERENCE = "candidate_to_reference"  # FP4 low nibble 2 -> 1
REFERENCE_TO_CANDIDATE = "reference_to_candidate"  # FP4 low nibble 1 -> 2
COUNTERFACTUAL_NIBBLE = {
    CANDIDATE_TO_REFERENCE: 0x1,
    REFERENCE_TO_CANDIDATE: 0x2,
}

PARTS = 10  # compressed == 2 requires parts == 10 (batch rule and split)


@dataclass
class PlannedRow:
    """One row of a case: a capture key plus an optional counterfactual."""

    key: tuple
    counterfactual: Optional[str] = None
    expected_key: Optional[tuple] = None  # captured output to compare against

    def variant_id(self) -> str:
        base = f"{self.key[0]}:{self.key[1]}"
        if self.counterfactual:
            return f"{base}:{self.counterfactual}"
        return base


@dataclass
class Case:
    name: str
    kind: str  # "single" | "batch"
    rows: list
    parts: int = PARTS

    @property
    def row_count(self) -> int:
        return len(self.rows)


def standard_cases() -> list:
    """The bounded explicit case list (single + batched, both orders)."""
    return [
        Case("single_reference", "single",
             [PlannedRow(REF)]),
        Case("single_candidate_lane0", "single",
             [PlannedRow(CANDIDATE_LANE0)]),
        Case("single_candidate_lane1", "single",
             [PlannedRow(CANDIDATE_LANE1)]),
        Case("batch_as_captured_lane0", "batch",
             [PlannedRow(CANDIDATE_LANE0), PlannedRow(CANDIDATE_LANE0_ROW1)]),
        Case("batch_as_captured_lane1", "batch",
             [PlannedRow(CANDIDATE_LANE1), PlannedRow(CANDIDATE_LANE1_ROW1)]),
        Case("batch_mixed_reference_candidate", "batch",
             [PlannedRow(REF), PlannedRow(CANDIDATE_LANE0)]),
        Case("batch_swapped_candidate_reference", "batch",
             [PlannedRow(CANDIDATE_LANE0), PlannedRow(REF)]),
        Case("counterfactual_candidate_to_reference_single", "single",
             [PlannedRow(CANDIDATE_LANE0, CANDIDATE_TO_REFERENCE,
                         expected_key=REF)]),
        Case("counterfactual_candidate_to_reference_batch", "batch",
             [PlannedRow(CANDIDATE_LANE0, CANDIDATE_TO_REFERENCE,
                         expected_key=REF),
              PlannedRow(REF)]),
        Case("counterfactual_reference_to_candidate_single", "single",
             [PlannedRow(REF, REFERENCE_TO_CANDIDATE,
                         expected_key=CANDIDATE_LANE0)]),
        Case("counterfactual_reference_to_candidate_batch", "batch",
             [PlannedRow(REF, REFERENCE_TO_CANDIDATE,
                         expected_key=CANDIDATE_LANE0),
              PlannedRow(CANDIDATE_LANE0)]),
    ]


def select_cases(cases: list, names: Optional[list]) -> list:
    if not names:
        return list(cases)
    by_name = {case.name: case for case in cases}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError(f"unknown case(s): {', '.join(missing)}")
    return [by_name[name] for name in names]


def plan_case(case: Case, index: dict, repeats: int) -> dict:
    """CPU-side plan record for one case (JSON-serializable)."""
    rows = []
    for position, planned in enumerate(case.rows):
        row = index[planned.key]
        expected = index[planned.expected_key or planned.key]
        rows.append({
            "position": position,
            "wave": planned.key[0],
            "request_id": planned.key[1],
            "counterfactual": planned.counterfactual,
            "expected_output_wave": expected.wave,
            "expected_output_request_id": expected.request_id,
            "expected_output_sha256": expected.output_sha256,
            "canonical_digest": row.evidence["canonical_digest"],
            "query_sha256": row.evidence["query_sha256"],
            "actual_metadata10": list(row.metadata10),
            "replay_begin": row.replay_begin,
        })
    return {
        "name": case.name,
        "kind": case.kind,
        "parts": case.parts,
        "repeats": repeats,
        "rows": rows,
    }
