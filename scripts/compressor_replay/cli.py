"""CLI for the compressor component replay harness.

Default mode is CPU-only validate/plan: loads and validates the ratio-two
captures, resolves every device descriptor, plans experiments A-D with full
operand provenance and expected byte hashes, and writes the plan to a fresh
create-only output directory.  GPU execution requires the explicit
``--execute`` flag (the only code path that imports torch) plus the pinned
native library path and sha256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from . import experiments as exp
from .schema import (
    CaptureError,
    load_capture,
    load_captures,
    resolve_predecessor,
)
from .validation import validate_experiment


class CLIError(Exception):
    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="replay-ds41-compressor-actual",
        description=__doc__,
    )
    parser.add_argument("--activations", type=Path, required=True,
                        help="capture root: one capture directory holding "
                        "layer*-compressor-inputs.json, or a directory of "
                        "such capture directories")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="explicit manifest path inside a single capture "
                        "directory (disambiguates multiple captures)")
    parser.add_argument("--native-library", type=Path, required=True,
                        help="pinned libds41rt_native.so path")
    parser.add_argument("--native-sha256", required=True,
                        help="expected sha256 of the native library (no "
                        "default: pins are never guessed)")
    parser.add_argument("--output", type=Path, required=True,
                        help="output directory (must not exist; created new)")
    parser.add_argument("--execute", action="store_true",
                        help="run the planned experiments on CUDA (requires "
                        "torch); without this flag the CLI only validates "
                        "and plans on CPU")
    parser.add_argument("--repeats", type=int, default=3,
                        help="repeats per experiment (stability, not timing)")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--tag", default="",
                        help="experiment name prefix (separate runs)")
    return parser


def find_captures(root: Path, manifest: Path = None) -> list:
    """Resolve --activations into capture directories without guessing."""
    root = Path(root)
    if manifest is not None:
        return [(root, Path(manifest))]
    if not root.is_dir():
        raise CLIError(f"activations path not found: {root}")
    if list(root.glob("layer*-compressor-inputs.json")):
        return [(root, None)]
    subdirs = [d for d in sorted(root.iterdir())
               if d.is_dir() and list(d.glob("layer*-compressor-inputs.json"))]
    if not subdirs:
        raise CLIError(
            f"no layer*-compressor-inputs.json under {root} or its subdirectories"
        )
    return [(d, None) for d in subdirs]


def validate_and_plan(args) -> tuple:
    from .ffi import verify_native_library  # CPU-only: hashes, never loads

    native_sha256 = verify_native_library(args.native_library,
                                          args.native_sha256)
    if args.manifest is not None:
        manifest = Path(args.manifest)
        directory = manifest.parent
        # The weights live under the explicit activations root when it is a
        # real activations directory, otherwise beside the capture.
        root = Path(args.activations)
        if not (root / "projection-weights").is_dir():
            root = directory.parent
        captures = [load_capture(directory, activations_root=root,
                                 manifest_path=manifest)]
    else:
        captures = load_captures(Path(args.activations))
    planned = exp.plan_experiments(captures, tag=args.tag)
    # Structural gate: every selected experiment must validate before the
    # plan is written or any --execute path can reach the GPU runner.
    for experiment in planned:
        validate_experiment(experiment)

    weights = captures[0].weights_provenance if captures else {}
    plan = {
        "mode": "validate/plan (CPU only)",
        "activations": str(args.activations),
        "native_library": str(args.native_library),
        "native_library_sha256": native_sha256,
        "repeats": args.repeats,
        "weights": {
            "directory": weights.get("weights_dir"),
            "first_writer_manifest": weights.get("first_writer_manifest"),
            "first_writer_manifests": weights.get("first_writer_manifests", []),
            "recorded_remote_directory_provenance_only": weights.get(
                "recorded_remote_directory_provenance_only"
            ),
            "already_written_bindings": weights.get(
                "already_written_bindings", []
            ),
            "tensors": {
                role: {
                    "file": ref.path.name,
                    "dtype": ref.dtype,
                    "shape": list(ref.shape),
                    "bytes": ref.bytes,
                    "sha256": ref.sha256,
                }
                for role, ref in (captures[0].weights.items() if captures else ())
            },
        },
        "captures": [
            {
                "name": c.directory.name,
                "directory": str(c.directory),
                "manifest_sha256": c.manifest_sha256,
                "layer": c.layer,
                "ratio": c.ratio,
                "rows": c.rows,
                "slot_count": c.slot_count,
                "device_matches_host": c.device_matches_host,
                "p41_rows": c.p41_rows(),
                "predecessors": [
                    {"row": r,
                     "descriptor": c.device_descriptors[r],
                     **resolve_predecessor(c.device_descriptors[r], r,
                                           c.slot_count)}
                    for r in range(c.rows)
                ],
                "expected_output_hashes": c.expected_output_hashes(),
            }
            for c in captures
        ],
        "reference_selection": exp.reference_selection(captures),
        "experiments": [
            {
                "name": e.name,
                "kind": e.kind,
                "counterfactual": e.counterfactual,
                "rows": e.rows,
                "slots": e.slots,
                "stages": e.stages,
                "outputs": list(e.outputs),
                "expected_sha256": {
                    role: hashlib.sha256(blob).hexdigest()
                    for role, blob in e.expected.items()
                },
                "operand_sha256": {
                    op.role: hashlib.sha256(op.blob).hexdigest()
                    for op in e.operands
                },
                "operand_sources": {op.role: op.source for op in e.operands},
                "notes": list(e.notes),
            }
            for e in planned
        ],
        "execute_command": (
            f"{Path(sys.argv[0]).name} --activations {args.activations} "
            f"--native-library {args.native_library} "
            f"--native-sha256 {args.native_sha256} "
            f"--output <new-dir> --execute --repeats {args.repeats}"
        ),
    }
    return plan, captures, planned


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan, captures, planned = validate_and_plan(args)
    except (CLIError, CaptureError, ValueError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, indent=2),
              file=sys.stderr)
        return 1

    output = Path(args.output)
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(json.dumps({"ok": False,
                          "error": f"output path already exists: {output}"},
                         indent=2), file=sys.stderr)
        return 2
    (output / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True))

    if not args.execute:
        print(json.dumps({
            "ok": True,
            "mode": plan["mode"],
            "output": str(output),
            "captures": len(captures),
            "experiments": len(planned),
            "reference_selection": plan["reference_selection"],
            "counterfactual_experiments": sum(1 for e in planned
                                              if e.counterfactual),
            "execute_command": plan["execute_command"],
        }, indent=2, sort_keys=True))
        return 0

    # torch stays deferred to the execute path only.
    import torch

    from .runner import open_session, run_experiments

    with open_session(torch, args.native_library, args.native_sha256,
                      device=args.device) as session:
        summary = run_experiments(planned, session, output,
                                  repeats=args.repeats)
    print(json.dumps({
        "ok": True,
        "output": str(output),
        "experiments": len(summary["experiments"]),
        "process_rc": summary["process_rc"],
        # process_rc 0 alone is not a numerical pass: rejected launches are
        # unexecuted/unscored, never arithmetic failures.
        "component_baseline_pass": summary["component_baseline_pass"],
        "status": summary["status"],
        "unscored_experiments": summary["unscored_experiments"],
        "numerical_mismatch_experiments": summary["numerical_mismatch_experiments"],
        "counterfactual_experiments": summary["counterfactual_experiments"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
