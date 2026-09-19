"""CLI for the actual-input sparse-attention component replay harness.

Default mode is CPU-only validate/plan: loads the captured rows through the
pinned extractor, materializes compact private fixtures, verifies them with
the integer oracle, and prints the bounded case plan (including per-row
expected output hashes).  GPU execution requires the explicit ``--execute``
flag and creates a fresh, non-overwritable output directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import pins
from .capture import CaptureError, load_activations
from .cases import plan_case, select_cases, standard_cases
from .materialize import (
    MaterializeError,
    materialize,
    mutate_private_source_column,
    COUNTERFACTUAL_COLUMN,
)

DEFAULT_ACTIVATIONS = Path(
    "/home/turq/.cache/afd-dsh-workers-20260919/flash-cap16-p2-inputs-diagnostic-01/"
    "activations"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="replay-ds41-sparse-actual",
        description=__doc__,
    )
    parser.add_argument(
        "--activations",
        type=Path,
        default=DEFAULT_ACTIVATIONS,
        help="capture directory holding */layer*-attention-inputs.json "
        "(default: the cap16 p41/layer2 diagnostic captures)",
    )
    parser.add_argument("--extractor", type=Path, default=pins.EXTRACTOR_PATH)
    parser.add_argument("--extractor-sha", default=pins.EXTRACTOR_SHA256)
    parser.add_argument("--oracle", type=Path, default=pins.ORACLE_PATH)
    parser.add_argument("--oracle-sha", default=pins.ORACLE_SHA256)
    parser.add_argument("--native", type=Path, default=pins.NATIVE_LIBRARY_PATH)
    parser.add_argument("--native-sha", default=pins.NATIVE_LIBRARY_SHA256)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="run the cases on CUDA (requires torch); without this flag the "
        "CLI only validates and plans on CPU",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="execution output directory (must not exist; created new)",
    )
    parser.add_argument("--repeats", type=int, default=3,
                        help="repeats per case (stability, not timing)")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--cases",
        nargs="*",
        default=None,
        help="bounded explicit case names (default: the standard list)",
    )
    parser.add_argument("--plan-out", type=Path,
                        help="write the CPU validation/plan JSON here too")
    return parser


def validate_and_plan(args) -> dict:
    """CPU-only: load captures, materialize, oracle-verify, build the plan."""
    extractor = pins.load_extractor(
        args.extractor, args.extractor_sha, args.oracle, args.oracle_sha
    )
    oracle = pins.load_oracle(args.oracle, args.oracle_sha)
    index = load_activations(args.activations, extractor)

    cases = select_cases(standard_cases(), args.cases)
    materialized = {}
    variants = {}
    counterfactual_materialized = {}
    for case in cases:
        for planned in case.rows:
            for key, counterfactual in (
                (planned.key, None),
                (planned.key, planned.counterfactual),
            ):
                if counterfactual is None:
                    if key not in materialized:
                        materialized[key] = materialize(index[key], oracle)
                else:
                    variant_key = (key, counterfactual)
                    if variant_key not in variants:
                        nibble = {
                            "candidate_to_reference": 0x1,
                            "reference_to_candidate": 0x2,
                        }[counterfactual]
                        variants[variant_key] = mutate_private_source_column(
                            index[key], oracle, nibble,
                            column=COUNTERFACTUAL_COLUMN,
                        )
                        # Materialize the mutated variant too: the compact
                        # fixture must reproduce the digest regenerated from
                        # the actual mutated bytes, and this failure must
                        # surface here on CPU rather than after a GPU launch.
                        counterfactual_materialized[variant_key] = materialize(
                            variants[variant_key]["row"], oracle
                        )

    plan = {
        "activations": str(args.activations),
        "manifest_sha256": {
            str(key): row.manifest_sha256 for key, row in sorted(index.items())
        },
        "request_rows": {
            f"{key[0]}:{key[1]}": {
                "flat_row": row.flat_row,
                "position": row.position,
                "launch_kind": row.launch_kind,
                "canonical_digest": row.evidence["canonical_digest"],
                "output_sha256": row.output_sha256,
                "referenced_rows": row.referenced_count,
                "page_stride": row.page_stride,
                "original_source_capacity_rows": row.source_capacity_rows,
            }
            for key, row in sorted(index.items())
        },
        "materialized": {
            f"{key[0]}:{key[1]}": {
                "source_capacity": mat.source_capacity,
                "pool_bytes": mat.pool_bytes,
                "page_map": {str(k): v for k, v in sorted(mat.page_map.items())},
                "untouched_pages_point_outside_capacity": True,
                "canonical_digest_compact": mat.canonical_digest_compact,
                "canonical_digest_matches_capture": (
                    mat.canonical_digest_compact
                    == index[key].evidence["canonical_digest"]
                ),
            }
            for key, mat in sorted(materialized.items())
        },
        "counterfactual_variants": {
            f"{key[0]}:{key[1]}:{cf}": record["evidence"]
            for (key, cf), record in sorted(variants.items())
        },
        "counterfactual_materialized": {
            f"{key[0]}:{key[1]}:{cf}": {
                "source_capacity": mat.source_capacity,
                "pool_bytes": mat.pool_bytes,
                "page_map": {str(k): v for k, v in sorted(mat.page_map.items())},
                "untouched_pages_point_outside_capacity": True,
                "canonical_digest_compact": mat.canonical_digest_compact,
                "canonical_digest_matches_mutation": (
                    mat.canonical_digest_compact
                    == variants[(key, cf)]["evidence"]["canonical_digest"]
                ),
                "original_canonical_digest": (
                    variants[(key, cf)]["evidence"]["original_canonical_digest"]
                ),
            }
            for (key, cf), mat in sorted(counterfactual_materialized.items())
        },
        "cases": [plan_case(case, index, args.repeats) for case in cases],
        "execute_command": (
            f"{Path(sys.argv[0]).name} --activations {args.activations} "
            f"--native {args.native} --output-dir <new-dir> --execute "
            f"--repeats {args.repeats}"
        ),
        "mode": "validate/plan (CPU only)",
    }
    return plan, index, cases, oracle


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan, index, cases, oracle = validate_and_plan(args)
    except (CaptureError, MaterializeError, ValueError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, indent=2),
              file=sys.stderr)
        return 1

    if args.plan_out:
        args.plan_out.write_text(json.dumps(plan, indent=2, sort_keys=True))

    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    if args.output_dir is None:
        print("--output-dir is required with --execute", file=sys.stderr)
        return 2
    # Deferred import: torch is only needed for execution.
    from .runner import run_cases

    summary = run_cases(
        cases, index, oracle, args.native, args.native_sha,
        args.output_dir, repeats=args.repeats, device=args.device,
    )
    print(json.dumps({
        "ok": True,
        "output_dir": str(args.output_dir),
        "native_library_sha256": summary["native_library_sha256"],
        "cases": len(summary["cases"]),
        "process_rc": summary["process_rc"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
