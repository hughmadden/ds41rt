#!/usr/bin/env python3
"""Resolve or launch one DS41RT production profile."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from ds41rt_reference.serve_profiles import (
    resolve_serve_profile,
)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Resolve balanced/long/accuracy DS41RT launch settings."
    )
    parser.add_argument(
        "--profile",
        choices=("balanced", "long", "accuracy"),
        default="balanced",
    )
    parser.add_argument("--dspark", choices=("on", "off"), default="on")
    parser.add_argument(
        "--dspark-draft-policy",
        choices=("full", "adaptive"),
        default="full",
        help="verify all five native proposals or select a confidence-scored prefix",
    )
    parser.add_argument(
        "--model-id",
        default="deepseek-ai/DeepSeek-V4-Flash-0731",
        help="Hugging Face model repository ID",
    )
    parser.add_argument(
        "--model-variant",
        choices=("flash", "pro"),
        default="flash",
    )
    parser.add_argument(
        "--expert-format",
        choices=("native", "exl3"),
        default="native",
    )
    parser.add_argument("--headroom-gib", type=float, default=8.0)
    parser.add_argument("--max-context-tokens", type=int)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--kv-pool-tokens", type=int)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--spark-reduction-min-rows",
        type=int,
        default=16,
        help="minimum physical batch width reduced across the four Sparks",
    )
    parser.add_argument("--coordinator-gpu", type=int, default=0)
    parser.add_argument(
        "--coordinator-gpu-uuid",
        help="physical host GPU UUID exposed as container-local CUDA device 0",
    )
    parser.add_argument(
        "--coordinator-gpu-pci-bus-id",
        help="full physical PCI bus ID corresponding to --coordinator-gpu-uuid",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        help="artifact root; defaults to the source tree containing this tool",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-unqualified",
        action="store_true",
        help="permit a diagnostic launch despite explicit profile blockers",
    )
    parser.add_argument(
        "--launcher",
        help="launcher path; defaults to scripts/real-full-tcp-serve.sh",
    )
    return parser.parse_known_args()


def main() -> int:
    args, launcher_args = parse_args()
    if args.dry_run and launcher_args:
        print(
            "unrecognized profile arguments: " + " ".join(launcher_args),
            file=sys.stderr,
        )
        return 2
    repo_root = (
        args.repo_root.expanduser().resolve()
        if args.repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    resolved = resolve_serve_profile(
        repo_root=repo_root,
        profile=args.profile,
        model_id=args.model_id,
        model_variant=args.model_variant,
        expert_format=args.expert_format,
        dspark=args.dspark == "on",
        dspark_draft_policy=args.dspark_draft_policy,
        coordinator_gpu=args.coordinator_gpu,
        coordinator_gpu_uuid=args.coordinator_gpu_uuid,
        coordinator_gpu_pci_bus_id=args.coordinator_gpu_pci_bus_id,
        headroom_gib=args.headroom_gib,
        max_context_tokens=args.max_context_tokens,
        max_output_tokens=args.max_output_tokens,
        kv_pool_tokens=args.kv_pool_tokens,
        concurrency=args.concurrency,
        spark_reduction_min_rows=args.spark_reduction_min_rows,
        inherited_environment=os.environ,
    )
    if args.dry_run:
        print(resolved.to_json())
        return 0

    environment = os.environ.copy()
    environment.update(resolved.environment)
    if resolved.blockers and not args.allow_unqualified:
        print(
            json.dumps(
                {
                    "error": "profile has launch blockers",
                    "blockers": resolved.blockers,
                    "hint": "fix the blockers or use --allow-unqualified for diagnostics",
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2

    launcher = (
        Path(args.launcher).expanduser()
        if args.launcher
        else repo_root / "scripts" / "real-full-tcp-serve.sh"
    )
    os.execve(str(launcher), [str(launcher), *launcher_args], environment)
    raise AssertionError("execve returned")


if __name__ == "__main__":
    raise SystemExit(main())
