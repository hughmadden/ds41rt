#!/usr/bin/env python3
"""Assemble one DS4RT model from a completed Flash base and dSpark overlay."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from canonicalize_gptqmodel_artifact import AssemblyError, canonicalize_composite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run-state", type=Path, required=True)
    parser.add_argument("--mtp-overlay", type=Path, required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--base-block-audit-dir", type=Path, required=True)
    parser.add_argument("--mtp-block-audit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        canonicalize_composite(
            base_run_state=args.base_run_state,
            mtp_overlay=args.mtp_overlay,
            source_snapshot=args.source_snapshot,
            base_block_audit_dir=args.base_block_audit_dir,
            mtp_block_audit_dir=args.mtp_block_audit_dir,
            output=args.output,
            work_dir=args.work_dir,
            resume=args.resume,
        )
        return 0
    except (AssemblyError, OSError, ValueError) as error:
        print(f"canonicalize-gptqmodel-composite: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
