#!/usr/bin/env python3
"""Actual-only CPU smoke report for the compressor replay plan.

This is deliberately independent of the synthetic fixtures: it loads the real
activations root, resolves the shared weights, validates the padded project-only
planner geometry, and reports the baseline/expected and counterfactual hashes.
It performs no GPU, service, network, build or install work and never imports
torch.  Run it with an explicit ``--activations`` root and a fresh ``--output``.

    PYTHONDONTWRITEBYTECODE=1 python3 scripts/validate-compressor-replay-actual.py \
        --activations <activations-root> --output <new-json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compressor_replay import experiments, schema  # noqa: E402


def build_report(activations_root: Path) -> dict:
    captures = schema.load_captures(Path(activations_root))
    planned = experiments.plan_experiments(captures)
    baselines = [e for e in planned if not e.counterfactual]
    counterfactuals = [e for e in planned if e.counterfactual]
    padded = [e for e in planned if e.kind == "P"]

    weights = captures[0].weights if captures else {}
    return {
        "schema": 1,
        "kind": "compressor-replay-actual-smoke",
        "cpu_only": True,
        "torch_imported": "torch" in sys.modules,
        "activations_root": str(Path(activations_root)),
        "captures": [
            {
                "name": c.directory.name,
                "layer": c.layer,
                "ratio": c.ratio,
                "rows": c.rows,
                "slot_count": c.slot_count,
                "p41_rows": c.p41_rows(),
                "device_descriptors": list(c.device_descriptors),
                "device_matches_host": c.device_matches_host,
                "manifest_sha256": c.manifest_sha256,
                "buffers": {
                    name: {"file": ref.path.name, "dtype": ref.dtype,
                           "shape": list(ref.shape), "bytes": ref.bytes,
                           "sha256": ref.sha256}
                    for name, ref in c.buffers.items()
                },
            }
            for c in captures
        ],
        "p41_row_total": sum(len(c.p41_rows()) for c in captures),
        "reference_selection": experiments.reference_selection(captures),
        "weights": {
            "directory": captures[0].weights_provenance.get("weights_dir")
            if captures else None,
            "first_writer_manifest": captures[0].weights_provenance.get(
                "first_writer_manifest"
            ) if captures else None,
            "recorded_remote_directory_provenance_only": captures[
                0
            ].weights_provenance.get(
                "recorded_remote_directory_provenance_only"
            ) if captures else None,
            "tensors": {
                role: {"file": ref.path.name, "dtype": ref.dtype,
                       "shape": list(ref.shape), "bytes": ref.bytes,
                       "sha256": ref.sha256}
                for role, ref in weights.items()
            },
        },
        "experiment_kinds": dict(Counter(e.kind for e in planned)),
        "experiment_total": len(planned),
        "baseline_count": len(baselines),
        "baselines_with_oracle_bytes": sum(
            1 for e in baselines if e.expected
        ),
        "counterfactual_count": len(counterfactuals),
        "counterfactuals_with_empty_oracles": sum(
            1 for e in counterfactuals if not e.expected
        ),
        "counterfactual_input_hashes": sorted({
            hashlib.sha256(
                next(op.blob for op in e.operands if op.role == "input")
            ).hexdigest()
            for e in counterfactuals
            if any(op.role == "input" for op in e.operands)
        }),
        "padded": {
            "count": len(padded),
            "rows": dict(Counter(e.rows for e in padded)),
            "slots": {
                str(rows): sorted({
                    int(e.name.rsplit("slot", 1)[1].split("_")[0])
                    for e in padded if e.rows == rows
                })
                for rows in sorted({e.rows for e in padded})
            },
            "distinct_neighbour_variants": sorted(
                e.name for e in padded if "distinct" in e.name
            ),
            "all_counterfactual": all(e.counterfactual for e in padded),
            "all_project_only": all(
                [s["kind"] for s in e.stages] == ["project", "project"]
                for e in padded
            ),
        },
        "limitations": [
            "CPU-only: no GPU, service, network, build, install or native load.",
            "No timing or numerical accuracy claim is made.",
            "Weight hashes are computed locally; the pinned manifest has no "
            "expected SHA-256 field to compare against.",
            "The recorded remote weights directory is provenance only.",
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations", required=True)
    parser.add_argument("--output", required=True,
                        help="fresh report path (create-only)")
    args = parser.parse_args(argv)
    report = build_report(Path(args.activations))
    output = Path(args.output)
    with open(output, "x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({
        "ok": True,
        "output": str(output),
        "captures": len(report["captures"]),
        "p41_row_total": report["p41_row_total"],
        "experiments": report["experiment_total"],
        "padded": report["padded"]["count"],
        "torch_imported": report["torch_imported"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
