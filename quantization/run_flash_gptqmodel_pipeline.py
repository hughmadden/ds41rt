#!/usr/bin/env python3
"""Run base quantization, sampled dSpark overlay, and canonical export in order."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import quantize_flash_gptqmodel as base
from validate_projection_checkpoint_block import audit_block, write_json_atomic


DEFAULT_MTP_ANCHORS = 327_680
DEFAULT_MTP_SEED = 20_260_809


class PipelineError(RuntimeError):
    """The automatic two-phase quantization pipeline cannot continue safely."""


def _run(command: list[str], phase: str) -> None:
    print(f"ds4rt quantization pipeline: starting {phase}", flush=True)
    subprocess.run(command, check=True)
    print(f"ds4rt quantization pipeline: completed {phase}", flush=True)


def _remote_arguments(args: argparse.Namespace) -> list[str]:
    result: list[str] = []
    for declaration in args.remote_worker or ():
        result.extend(("--remote-worker", *declaration))
    if args.remote_worker:
        result.extend(
            (
                "--remote-token-env",
                args.remote_token_env,
                "--remote-timeout-seconds",
                str(args.remote_timeout_seconds),
                "--remote-max-attempts",
                str(args.remote_max_attempts),
            )
        )
    return result


def _base_command(args: argparse.Namespace, *, resume: bool) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).with_name("quantize_flash_gptqmodel.py")),
        "--snapshot",
        str(args.snapshot),
        "--calibration-jsonl",
        str(args.calibration_jsonl),
        "--calibration-manifest",
        str(args.calibration_manifest),
        "--route-screen-report",
        str(args.route_screen_report),
        "--preflight-report",
        str(args.preflight_report),
        "--output",
        str(args.base_output),
        "--run-state-dir",
        str(args.base_run_state_dir),
        "--projection-checkpoint-dir",
        str(args.base_projection_checkpoint_dir),
        "--active-layer-source-dir",
        str(args.base_active_layer_source_dir),
        "--offload-dir",
        str(args.base_offload_dir),
        "--mtp-prefix-store",
        str(args.mtp_prefix_store),
        "--gptqmodel-lock",
        str(args.gptqmodel_lock),
        "--bits",
        str(args.bits),
        "--coordinator-gpu-count",
        str(args.coordinator_gpu_count),
        "--batch-size",
        str(args.batch_size),
        "--mtp-replay-batch-size",
        str(args.mtp_replay_batch_size),
        "--mtp-execution-mode",
        "integrated" if args.legacy_integrated_parent else "external-overlay",
        *_remote_arguments(args),
    ]
    if resume:
        command.append("--resume")
    if args.legacy_integrated_parent:
        if not resume:
            raise PipelineError("legacy integrated handoff requires an existing base run")
        command.extend(("--execution-upgrade", "--mtp-overlay-handoff"))
    return command


def _overlay_command(args: argparse.Namespace, *, resume: bool) -> list[str]:
    base_run = args.base_run_state_dir
    command = [
        sys.executable,
        str(Path(__file__).with_name("quantize_flash_dspark_overlay.py")),
        "--snapshot",
        str(args.snapshot),
        "--parent-plan",
        str(base_run / base.PLAN_FILENAME),
        "--mtp-prefix-store",
        str(args.mtp_prefix_store),
        "--preflight-report",
        str(args.preflight_report),
        "--output",
        str(args.mtp_overlay_output),
        "--run-state-dir",
        str(args.mtp_overlay_run_state_dir),
        "--projection-checkpoint-dir",
        str(args.mtp_overlay_projection_checkpoint_dir),
        "--active-layer-source-dir",
        str(args.mtp_overlay_active_layer_source_dir),
        "--offload-dir",
        str(args.mtp_overlay_offload_dir),
        "--gptqmodel-lock",
        str(args.gptqmodel_lock),
        "--bits",
        str(args.bits),
        "--coordinator-gpu-count",
        str(args.coordinator_gpu_count),
        "--mtp-anchor-sample-count",
        str(args.mtp_anchor_sample_count),
        "--mtp-anchor-sample-seed",
        str(args.mtp_anchor_sample_seed),
        "--mtp-sequence-anchor-cap",
        str(args.mtp_sequence_anchor_cap),
        *_remote_arguments(args),
    ]
    if resume:
        command.append("--resume")
    owner_weights = getattr(args, "mtp_hessian_owner_device_weights", None)
    if owner_weights is not None:
        command.extend(
            (
                "--mtp-hessian-owner-device-weights",
                *(str(weight) for weight in owner_weights),
            )
        )
    return command


def _write_base_audits(args: argparse.Namespace) -> None:
    run_state = args.base_run_state_dir
    plan = base.read_json_object(run_state / base.PLAN_FILENAME)
    base.validate_base_prefix_completion(plan)
    args.base_block_audit_dir.mkdir(parents=True, exist_ok=True)
    layers = int(plan["source"]["geometry"]["num_hidden_layers"])
    for layer in range(layers):
        report = audit_block(
            run_state,
            block_namespace="base",
            logical_layer=layer,
        )
        output = args.base_block_audit_dir / (
            f"base-layer-{layer}-projection-audit.json"
        )
        write_json_atomic(output, report)
    print(
        f"ds4rt quantization pipeline: completed {layers} base block audits",
        flush=True,
    )


def _canonical_command(args: argparse.Namespace, *, resume: bool) -> list[str]:
    base_run = args.base_run_state_dir
    command = [
        sys.executable,
        str(Path(__file__).with_name("canonicalize_gptqmodel_composite_artifact.py")),
        "--base-run-state",
        str(base_run),
        "--mtp-overlay",
        str(args.mtp_overlay_output),
        "--source-snapshot",
        str(args.snapshot),
        "--base-block-audit-dir",
        str(args.base_block_audit_dir),
        "--mtp-block-audit-dir",
        str(args.mtp_overlay_output / "reports"),
        "--output",
        str(args.canonical_output),
        "--work-dir",
        str(args.canonical_work_dir),
    ]
    if resume:
        command.append("--resume")
    return command


def execute(args: argparse.Namespace) -> None:
    if args.canonical_output.exists():
        print(
            "ds4rt quantization pipeline: canonical output already exists; complete",
            flush=True,
        )
        return
    base_run = args.base_run_state_dir
    _run(
        _base_command(args, resume=base_run.exists()),
        "resumable base and MTP-prefix phase",
    )
    _write_base_audits(args)

    overlay_run = args.mtp_overlay_run_state_dir
    _run(
        _overlay_command(
            args,
            resume=overlay_run.exists() or args.mtp_overlay_output.exists(),
        ),
        f"sampled {args.mtp_anchor_sample_count}-anchor dSpark overlay",
    )

    if args.canonical_output.exists():
        print(
            "ds4rt quantization pipeline: canonical output already exists; complete",
            flush=True,
        )
        return
    _run(
        _canonical_command(args, resume=args.canonical_work_dir.exists()),
        "canonical composite export",
    )


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--calibration-jsonl", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--route-screen-report", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--base-output", type=Path, required=True)
    parser.add_argument("--base-run-state-dir", type=Path)
    parser.add_argument("--base-projection-checkpoint-dir", type=Path)
    parser.add_argument("--base-active-layer-source-dir", type=Path)
    parser.add_argument("--base-offload-dir", type=Path, required=True)
    parser.add_argument("--mtp-prefix-store", type=Path, required=True)
    parser.add_argument("--mtp-overlay-output", type=Path, required=True)
    parser.add_argument("--mtp-overlay-run-state-dir", type=Path)
    parser.add_argument("--mtp-overlay-projection-checkpoint-dir", type=Path)
    parser.add_argument("--mtp-overlay-active-layer-source-dir", type=Path)
    parser.add_argument("--mtp-overlay-offload-dir", type=Path, required=True)
    parser.add_argument("--base-block-audit-dir", type=Path, required=True)
    parser.add_argument("--canonical-output", type=Path, required=True)
    parser.add_argument(
        "--canonical-work-dir",
        type=Path,
        help=(
            "resumable canonical assembly directory; defaults to a hidden sibling "
            "of --canonical-output so final publication is an atomic rename"
        ),
    )
    parser.add_argument(
        "--gptqmodel-lock",
        type=Path,
        default=root / "third_party" / "gptqmodel.lock.json",
    )
    parser.add_argument("--bits", type=int, choices=(2, 3), required=True)
    parser.add_argument("--coordinator-gpu-count", type=int, choices=(1, 2), default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--mtp-replay-batch-size", type=int, default=4)
    parser.add_argument(
        "--mtp-anchor-sample-count", type=int, default=DEFAULT_MTP_ANCHORS
    )
    parser.add_argument("--mtp-anchor-sample-seed", type=int, default=DEFAULT_MTP_SEED)
    parser.add_argument(
        "--mtp-hessian-owner-device-weights",
        type=int,
        nargs="+",
        help=(
            "positive Hessian-capture ownership weights, one per coordinator "
            "GPU; defaults to equal ownership"
        ),
    )
    parser.add_argument("--mtp-sequence-anchor-cap", type=int, default=0)
    parser.add_argument("--legacy-integrated-parent", action="store_true")
    parser.add_argument("--remote-worker", action="append", nargs=3)
    parser.add_argument("--remote-token-env", default="DS4RT_EXL3_WORKER_TOKEN")
    parser.add_argument("--remote-timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--remote-max-attempts", type=int, default=2)
    args = parser.parse_args()
    for field in (
        "snapshot",
        "calibration_jsonl",
        "calibration_manifest",
        "route_screen_report",
        "preflight_report",
        "base_output",
        "mtp_prefix_store",
        "mtp_overlay_output",
        "base_block_audit_dir",
        "canonical_output",
        "gptqmodel_lock",
    ):
        setattr(args, field, getattr(args, field).expanduser().resolve())
    for field in ("base_offload_dir", "mtp_overlay_offload_dir"):
        setattr(args, field, getattr(args, field).expanduser().resolve())
    defaults = {
        "base_run_state_dir": args.base_output.with_name(
            f".{args.base_output.name}.ds4rt-run"
        ),
        "mtp_overlay_run_state_dir": args.mtp_overlay_output.with_name(
            f".{args.mtp_overlay_output.name}.ds4rt-run"
        ),
    }
    for field, default in defaults.items():
        value = getattr(args, field)
        setattr(args, field, (value or default).expanduser().resolve())
    subordinate_defaults = {
        "base_projection_checkpoint_dir": (
            args.base_run_state_dir / base.PROJECTION_CHECKPOINT_DIRNAME
        ),
        "base_active_layer_source_dir": (
            args.base_run_state_dir / base.ACTIVE_LAYER_SOURCE_DIRNAME
        ),
        "mtp_overlay_projection_checkpoint_dir": (
            args.mtp_overlay_run_state_dir / base.PROJECTION_CHECKPOINT_DIRNAME
        ),
        "mtp_overlay_active_layer_source_dir": (
            args.mtp_overlay_run_state_dir / base.ACTIVE_LAYER_SOURCE_DIRNAME
        ),
    }
    for field, default in subordinate_defaults.items():
        value = getattr(args, field)
        setattr(args, field, (value or default).expanduser().resolve())
    if args.canonical_work_dir is None:
        args.canonical_work_dir = args.canonical_output.with_name(
            f".{args.canonical_output.name}.ds4rt-assembly"
        )
    else:
        args.canonical_work_dir = args.canonical_work_dir.expanduser().resolve()
    if args.canonical_work_dir.parent != args.canonical_output.parent:
        parser.error(
            "--canonical-work-dir must be a sibling of --canonical-output "
            "for atomic publication"
        )
    if (
        args.batch_size <= 0
        or args.mtp_replay_batch_size <= 0
        or args.mtp_anchor_sample_count <= 0
        or args.mtp_sequence_anchor_cap < 0
        or args.remote_timeout_seconds <= 0
        or not 1 <= args.remote_max_attempts <= 10
    ):
        parser.error("batch, sample, and worker limits are invalid")
    paths = (
        args.base_output,
        args.base_run_state_dir,
        args.base_projection_checkpoint_dir,
        args.base_active_layer_source_dir,
        args.base_offload_dir,
        args.mtp_prefix_store,
        args.mtp_overlay_output,
        args.mtp_overlay_run_state_dir,
        args.mtp_overlay_projection_checkpoint_dir,
        args.mtp_overlay_active_layer_source_dir,
        args.mtp_overlay_offload_dir,
        args.base_block_audit_dir,
        args.canonical_work_dir,
    )
    if len(set(paths)) != len(paths):
        parser.error("pipeline writable and prefix paths must be distinct")
    if args.canonical_output != args.base_output and args.canonical_output in paths:
        parser.error("canonical output overlaps another pipeline path")
    return args


def main() -> int:
    args = parse_args()
    try:
        execute(args)
        return 0
    except (PipelineError, subprocess.CalledProcessError, OSError, ValueError) as error:
        print(f"run-flash-gptqmodel-pipeline: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
