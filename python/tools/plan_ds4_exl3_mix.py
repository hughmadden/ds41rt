#!/usr/bin/env python3
"""Build a deterministic balanced K2/K3 expert-family selection manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ds4rt_runtime.exl3_mix import score_k2_k3_ledgers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k2", type=Path, required=True, help="canonical K2 snapshot")
    parser.add_argument("--k3", type=Path, required=True, help="canonical K3 snapshot")
    parser.add_argument("--target-bpw", default="2.1")
    parser.add_argument("--layers", type=int, default=46)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evidence = score_k2_k3_ledgers(
        args.k2 / "ds4rt-exl3-error-ledger.jsonl",
        args.k3 / "ds4rt-exl3-error-ledger.jsonl",
        layer_count=args.layers,
        experts_per_layer=args.experts,
    )
    plan = evidence.plan(target_bpw=args.target_bpw)
    report = evidence.summary(plan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
