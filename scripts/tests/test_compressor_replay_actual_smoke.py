"""Actual-only smoke test for the compressor replay plan.

Independent of the synthetic fixtures: it loads the real activations root and
asserts the plan invariants the synthetic tests cannot prove (real weight
binding, real 3-manifest / 5-p41-row inventory and the actual padded planner
geometry).  Skipped when the read-only capture is not present.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

REAL_CAPTURE = Path(
    "/home/turq/.cache/afd-dsh-workers-20260919/"
    "flash-cap16-p2-compressor-diagnostic-01/activations"
)
SCRIPT = SCRIPT_DIR / "validate-compressor-replay-actual.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("vcr_actual_smoke", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not REAL_CAPTURE.is_dir(),
                    reason="actual compressor capture not present")
def test_actual_only_smoke_report_invariants():
    report = _load_module().build_report(REAL_CAPTURE)
    assert report["torch_imported"] is False
    assert len(report["captures"]) == 3
    assert report["p41_row_total"] == 5
    assert sorted(c["rows"] for c in report["captures"]) == [1, 2, 2]
    assert report["weights"]["directory"] == "projection-weights"
    assert report["weights"][
        "recorded_remote_directory_provenance_only"
    ] == "/trace/projection-weights"
    assert report["weights"]["first_writer_manifest"].endswith(
        "lane0-batch20-Full-1rows/layer2-compressor-inputs.json"
    )
    assert set(report["weights"]["tensors"]) == {"wkv", "wgate", "norm"}
    for ref in report["weights"]["tensors"].values():
        actual = (REAL_CAPTURE / "projection-weights" / ref["file"]).stat().st_size
        assert actual == ref["bytes"]

    assert report["baseline_count"] == report["baselines_with_oracle_bytes"]
    assert report["counterfactual_count"] == (
        report["counterfactuals_with_empty_oracles"]
    )
    # The replacement row-case planners replaced the old C8 (broken geometry)
    # with C14 and added D24; every experiment is validated before any device
    # work.  The deliberate new total is A6 + B9 + C14 + D24 + P24 = 77
    # (the old unvalidated draft reported 47).
    assert report["experiment_total"] == 77
    assert report["experiment_kinds"] == {"A": 6, "B": 9, "C": 14, "D": 24,
                                          "P": 24}
    assert report["baseline_count"] == 26          # A/B baselines + D originals
    assert report["counterfactual_count"] == 51
    assert report["reference_selection"] == {
        "rule": "unique_single_row",
        "refname": "lane0-batch20-Full-1rows",
        "rows": 1,
    }
    padded = report["padded"]
    assert padded["count"] == 24
    assert padded["slots"] == {"2": [0, 1], "16": [0, 1, 7, 15]}
    assert len(padded["distinct_neighbour_variants"]) == 6
    assert padded["all_counterfactual"] is True
    assert padded["all_project_only"] is True
