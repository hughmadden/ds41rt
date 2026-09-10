#!/usr/bin/env python3
"""Convert DeepSeek V4 routed experts to calibrated native K2 EXL3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ds41rt_runtime.exl3_quantizer import (  # noqa: E402
    EXL3_ACTIVATION_RECIPE,
    EXL3_FORCED_ACTIVATION_RECIPE,
    EXL3_ROUTE_REPLAY_ARTIFACT_FILENAME,
    SafetensorsArtifactWriter,
    build_artifact_plan,
    load_activation_corpus,
    plan_summary,
    qualify_multigpu_batch_equivalence,
    run_layerwise_quantization,
)

MULTIGPU_QUALIFICATION_FILE = "ds41rt-exl3-multigpu-production.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--calibration-rows",
        type=int,
        default=512,
        help="equivalent row count used to normalize the analytic isotropic Hessian",
    )
    parser.add_argument(
        "--activation-corpus",
        type=Path,
        help=(
            "checkpoint-bound native activation manifest/root; routed v2 corpora "
            "use per-expert natural-route input and exact SwiGLU down Hessians"
        ),
    )
    parser.add_argument(
        "--forced-down-rows-per-expert",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--forced-down-expert-weight",
        type=float,
        default=0.25,
        help="expert-specific covariance weight; the remainder is the equal-expert pool",
    )
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--batch-experts", type=int, default=4)
    parser.add_argument(
        "--devices",
        default="0",
        help="visible CUDA ordinals used only for offline trellis search (for example 0,1)",
    )
    parser.add_argument(
        "--device-ratios",
        help="optional positive tile-search split ratios matching --devices",
    )
    parser.add_argument("--shard-size-gib", type=float, default=8.0)
    parser.add_argument("--debug-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--qualify-multigpu-only",
        action="store_true",
        help="run the repeated and production-shape one/two-GPU gate without writing an artifact",
    )
    parser.add_argument(
        "--qualification-output",
        type=Path,
        help="atomic JSON report path for --qualify-multigpu-only",
    )
    parser.add_argument(
        "--stop-after-layer",
        type=int,
        help="development checkpoint: leave a resumable, unpublished artifact",
    )
    parser.add_argument(
        "--only-layer",
        type=int,
        help="development qualification: encode only this layer into an unpublished artifact",
    )
    return parser.parse_args()


def parse_positive_int_list(raw: str, *, option: str, allow_zero: bool) -> tuple[int, ...]:
    try:
        values = tuple(int(value.strip()) for value in raw.split(","))
    except ValueError as error:
        raise SystemExit(f"{option} must be a comma-separated integer list") from error
    minimum = 0 if allow_zero else 1
    if not values or any(value < minimum for value in values):
        qualifier = "non-negative" if allow_zero else "positive"
        raise SystemExit(f"{option} must contain {qualifier} integers")
    return values


def write_json_atomic(path: Path, value: object) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    devices = parse_positive_int_list(args.devices, option="--devices", allow_zero=True)
    if devices[0] != 0 or len(set(devices)) != len(devices):
        raise SystemExit("--devices must contain unique visible ordinals beginning with 0")
    device_ratios = (
        parse_positive_int_list(
            args.device_ratios,
            option="--device-ratios",
            allow_zero=False,
        )
        if args.device_ratios
        else None
    )
    if device_ratios is not None and len(device_ratios) != len(devices):
        raise SystemExit("--device-ratios must contain one value per --devices entry")
    if args.shard_size_gib <= 0:
        raise SystemExit("--shard-size-gib must be positive")
    plan = build_artifact_plan(
        args.snapshot,
        max_shard_bytes=int(args.shard_size_gib * 1024**3),
    )
    activation_corpus = (
        load_activation_corpus(
            args.activation_corpus,
            snapshot=plan.snapshot,
            shape=plan.shape,
        )
        if args.activation_corpus is not None
        else None
    )
    if args.forced_down_rows_per_expert <= 0:
        raise SystemExit("--forced-down-rows-per-expert must be positive")
    if not 0.0 <= args.forced_down_expert_weight <= 1.0:
        raise SystemExit("--forced-down-expert-weight must be in 0..1")
    summary = plan_summary(plan)
    if activation_corpus is not None:
        summary.update(
            {
                "recipe": (
                    EXL3_ACTIVATION_RECIPE
                    if activation_corpus.route_aware
                    else EXL3_FORCED_ACTIVATION_RECIPE
                ),
                "activation_manifest": str(activation_corpus.manifest_path),
                "activation_corpus_sha256": activation_corpus.corpus_sha256,
                "activation_rows_by_layer": [
                    activation_corpus.rows_for_layer(layer_id)
                    for layer_id in range(activation_corpus.layer_count)
                ],
                "natural_routing": activation_corpus.route_aware,
                **(
                    {
                        "minimum_natural_routes_per_expert": (
                            activation_corpus.minimum_natural_routes_per_expert
                        ),
                        "route_replay_report": str(
                            activation_corpus.route_replay_report_path
                        ),
                        "route_replay_report_sha256": (
                            activation_corpus.route_replay_report_sha256
                        ),
                    }
                    if activation_corpus.route_aware
                    else {
                        "forced_down_rows_per_expert": (
                            args.forced_down_rows_per_expert
                        ),
                        "forced_down_expert_weight": (
                            args.forced_down_expert_weight
                        ),
                    }
                ),
            }
        )
    if args.plan_only and args.qualify_multigpu_only:
        raise SystemExit("--plan-only and --qualify-multigpu-only are mutually exclusive")
    if args.qualification_output is not None and not args.qualify_multigpu_only:
        raise SystemExit("--qualification-output requires --qualify-multigpu-only")
    if args.only_layer is not None and args.stop_after_layer is not None:
        raise SystemExit("--only-layer and --stop-after-layer are mutually exclusive")
    if args.plan_only:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if args.output is None and not args.qualify_multigpu_only:
        raise SystemExit("--output is required unless --plan-only is used")
    if args.qualify_multigpu_only and len(devices) <= 1:
        raise SystemExit("--qualify-multigpu-only requires at least two --devices")

    import _pinned_exllamav3

    debug_dir = args.debug_dir or (
        args.output / ".ds41rt-exl3-debug"
        if args.output is not None
        else Path(".ds41rt-cache/quality/exl3-multigpu-qualification-debug")
    )
    multigpu_qualification = qualify_multigpu_batch_equivalence(
        quantize_exl3_batch=_pinned_exllamav3.quantize_exl3_batch,
        quantization_devices=devices,
        device_ratios=device_ratios,
        seed=args.seed,
        debug_dir=debug_dir,
        production_projection_shapes=(
            (plan.shape.hidden_size, plan.shape.intermediate_size),
            (plan.shape.intermediate_size, plan.shape.hidden_size),
        ),
    )
    if multigpu_qualification is not None:
        print(json.dumps(multigpu_qualification, sort_keys=True), flush=True)
    if args.qualify_multigpu_only:
        if args.qualification_output is not None:
            write_json_atomic(args.qualification_output, multigpu_qualification)
        return

    assert args.output is not None
    calibration_override = None
    recipe = None
    if activation_corpus is not None:
        captured_rows = min(
            activation_corpus.rows_for_layer(layer_id)
            for layer_id in range(activation_corpus.layer_count)
        )
        recipe = (
            EXL3_ACTIVATION_RECIPE
            if activation_corpus.route_aware
            else EXL3_FORCED_ACTIVATION_RECIPE
        )
        calibration_override = {
            "method": (
                "layerwise_native_natural_routes"
                if activation_corpus.route_aware
                else "layerwise_native_activation_forced_down"
            ),
            "device": "cuda:0",
            "rows": captured_rows,
            "seed": args.seed,
            "hessian": (
                "per_expert_natural_route_gate_squared_covariance"
                if activation_corpus.route_aware
                else "native_sample_covariance_with_forced_down_shrinkage"
            ),
            "distribution": "checkpoint_bound_native_expert_inputs",
            "activation_manifest": str(activation_corpus.manifest_path),
            "activation_corpus_sha256": activation_corpus.corpus_sha256,
            "activation_checkpoint": str(activation_corpus.checkpoint),
            "activation_base_layers": activation_corpus.layer_count,
            **(
                {
                    "natural_routing": True,
                    "forced_expert_activation": False,
                    "route_gate_weighting": "squared_unit_rms",
                    "minimum_natural_routes_per_expert": (
                        activation_corpus.minimum_natural_routes_per_expert
                    ),
                    "route_replay_report": str(
                        EXL3_ROUTE_REPLAY_ARTIFACT_FILENAME
                    ),
                    "route_replay_report_sha256": (
                        activation_corpus.route_replay_report_sha256
                    ),
                }
                if activation_corpus.route_aware
                else {
                    "forced_down_rows_per_expert": (
                        args.forced_down_rows_per_expert
                    ),
                    "forced_down_expert_weight": args.forced_down_expert_weight,
                    "forced_down_pool_weight": (
                        1.0 - args.forced_down_expert_weight
                    ),
                }
            ),
            "mtp_calibration": "analytic_identity_pilot_only",
        }
    writer = SafetensorsArtifactWriter(
        plan,
        args.output,
        calibration_rows=args.calibration_rows,
        seed=args.seed,
        resume=args.resume,
        **({"recipe": recipe} if recipe is not None else {}),
        calibration_override=calibration_override,
    )
    if multigpu_qualification is not None:
        write_json_atomic(
            args.output / MULTIGPU_QUALIFICATION_FILE,
            multigpu_qualification,
        )
    writer.copy_native_tensors()
    report = run_layerwise_quantization(
        plan,
        writer,
        quantize_exl3_batch=_pinned_exllamav3.quantize_exl3_batch,
        calibration_rows=args.calibration_rows,
        seed=args.seed,
        batch_experts=args.batch_experts,
        debug_dir=debug_dir,
        stop_after_layer=args.stop_after_layer,
        quantization_devices=devices,
        device_ratios=device_ratios,
        multigpu_qualification=multigpu_qualification,
        activation_corpus=activation_corpus,
        forced_down_rows_per_expert=args.forced_down_rows_per_expert,
        forced_down_expert_weight=args.forced_down_expert_weight,
        only_layer=args.only_layer,
    )
    if report.get("incomplete"):
        print(json.dumps({**summary, "status": "resumable-incomplete"}, indent=2))
        return
    evidence_files = {}
    if activation_corpus is not None and activation_corpus.route_aware:
        assert activation_corpus.route_replay_report_path is not None
        evidence_files[EXL3_ROUTE_REPLAY_ARTIFACT_FILENAME] = (
            activation_corpus.route_replay_report_path
        )
    writer.finish(report, evidence_files=evidence_files)
    print(json.dumps({**summary, "status": "complete"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
