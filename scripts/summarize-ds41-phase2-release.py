#!/usr/bin/env python3
"""Compact the matched one/two-RTX Phase 2 release benchmark artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


LAYOUTS = ("single", "dual")
CONCURRENCY_CASES = ("counting", "code", "topic")
EXPECTED_CONCURRENCY = [1, 2, 4, 8, 16]


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict:
    return {
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def summarize_decode(path: Path) -> dict:
    report = load(path)
    grouped: dict[str, list[float]] = {}
    for sample in report["samples"]:
        assert sample["passed"], (path, sample["case"], sample["repeat"])
        grouped.setdefault(sample["case"], []).append(
            sample["observed_decode_tokens_per_second"]
        )
    return {
        "passed": report["passed"],
        "repeats": report["repeats"],
        "median_weighted_tps": report[
            "median_weighted_observed_decode_tokens_per_second"
        ],
        "cases": {
            case: {
                "samples": len(values),
                "median_tps": statistics.median(values),
                "min_tps": min(values),
                "max_tps": max(values),
            }
            for case, values in sorted(grouped.items())
        },
    }


def summarize_concurrency(path: Path) -> dict:
    report = load(path)
    assert report["passed"]
    assert report["concurrency"] == EXPECTED_CONCURRENCY
    assert report["repeats"] == 3
    return {
        "passed": True,
        "prompt": report["prompt"],
        "repeats": report["repeats"],
        "summaries": report["summaries"],
    }


def summarize_mixed(paths: list[Path]) -> dict:
    grouped: dict[int, list[float]] = {}
    for path in paths:
        report = load(path)
        assert report["passed"]
        assert report["lifecycle_skipped"]
        for batch in report["batches"]:
            grouped.setdefault(batch["concurrency"], []).append(batch["aggregate_tps"])
    assert sorted(grouped) == EXPECTED_CONCURRENCY
    assert all(len(values) == 3 for values in grouped.values())
    return {
        "passed": True,
        "repeats": 3,
        "summaries": [
            {
                "concurrency": concurrency,
                "samples": len(grouped[concurrency]),
                "median_aggregate_tps": statistics.median(grouped[concurrency]),
                "min_aggregate_tps": min(grouped[concurrency]),
                "max_aggregate_tps": max(grouped[concurrency]),
            }
            for concurrency in EXPECTED_CONCURRENCY
        ],
    }


def summarize_prefill(path: Path) -> dict:
    report = load(path)
    assert report["passed"]
    assert report["repeats"] == 3
    assert all(sample["passed"] for sample in report["samples"])
    return {
        "passed": True,
        "bases": report["bases"],
        "suffixes": report["suffixes"],
        "repeats": report["repeats"],
        "warmups_per_cell": report["warmups_per_cell"],
        "context_sha256": report["context_sha256"],
        "cells": report["cells"],
    }


def summarize_retained(path: Path) -> dict:
    report = load(path)
    assert report["passed"]
    assert report["objective_checks_passed"]
    assert report["repeats"] == 3
    assert all(sample["passed"] for sample in report["samples"])
    return {
        "passed": True,
        "objective_checks_passed": True,
        "contexts": report["contexts"],
        "repeats": report["repeats"],
        "context_sha256": report["context_sha256"],
        "corpus_sha256": report["corpus_sha256"],
        "context_summaries": report["context_summaries"],
    }


def summarize_lifecycle(path: Path) -> dict:
    report = load(path)
    cancellation = report["lifecycle"]["cancellation_batch"]
    return {
        "passed": report["passed"],
        "needle_cold": report["lifecycle"]["needle_cold"]["result"]["text"],
        "needle_warm": report["lifecycle"]["needle_warm"]["result"]["text"],
        "retained_turn": report["lifecycle"]["retained_turn"]["result"]["text"],
        "cancelled": sum(row["cancel"] for row in cancellation),
        "survivors": sum(not row["cancel"] for row in cancellation),
        "post_cancellation": report["lifecycle"]["post_cancellation"]["result"]["text"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine-commit", required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")

    json_files = sorted(path for path in args.input.iterdir() if path.suffix == ".json")
    expected = {
        *(f"{layout}-{case}.json" for layout in LAYOUTS for case in CONCURRENCY_CASES),
        *(f"{layout}-decode.json" for layout in LAYOUTS),
        *(f"{layout}-target-decode.json" for layout in LAYOUTS),
        *(f"{layout}-prefill.json" for layout in LAYOUTS),
        *(f"{layout}-retained-decode.json" for layout in LAYOUTS),
        *(f"{layout}-mixed-{repeat}.json" for layout in LAYOUTS for repeat in range(1, 4)),
        "dual-lifecycle.json",
    }
    observed = {path.name for path in json_files}
    assert observed == expected, (sorted(expected - observed), sorted(observed - expected))
    startup_files = [
        args.input / "single-startup-seconds.txt",
        args.input / "dual-startup-seconds.txt",
    ]
    assert all(path.is_file() for path in startup_files)
    files = sorted([*json_files, *startup_files])

    report = {
        "schema": 1,
        "scope": "Matched one/two-RTX Phase 2 release performance qualification",
        "source_commit": args.source_commit,
        "engine_commit": args.engine_commit,
        "controls": {
            "rtx_power_limit_watts": 400,
            "rtx_memory": "standard 14,001 MHz maximum; no overclock",
            "temperature": 0,
            "thinking": "disabled",
            "repeats": 3,
        },
        "layouts": {},
        "startup_seconds": {
            layout: float((args.input / f"{layout}-startup-seconds.txt").read_text())
            for layout in LAYOUTS
        },
        "lifecycle": summarize_lifecycle(args.input / "dual-lifecycle.json"),
        "artifacts": [artifact(path) for path in files],
    }
    for layout in LAYOUTS:
        report["layouts"][layout] = {
            "rtx_gpus": 1 if layout == "single" else 2,
            "decode": summarize_decode(args.input / f"{layout}-decode.json"),
            "target_decode": summarize_decode(
                args.input / f"{layout}-target-decode.json"
            ),
            "concurrency": {
                case: summarize_concurrency(args.input / f"{layout}-{case}.json")
                for case in CONCURRENCY_CASES
            },
            "mixed": summarize_mixed(
                [args.input / f"{layout}-mixed-{repeat}.json" for repeat in range(1, 4)]
            ),
            "prefill": summarize_prefill(args.input / f"{layout}-prefill.json"),
            "retained_decode": summarize_retained(
                args.input / f"{layout}-retained-decode.json"
            ),
        }
    report["passed"] = report["lifecycle"]["passed"] and all(
        layout["decode"]["passed"]
        and layout["target_decode"]["passed"]
        and layout["mixed"]["passed"]
        and layout["prefill"]["passed"]
        and layout["retained_decode"]["passed"]
        and all(case["passed"] for case in layout["concurrency"].values())
        for layout in report["layouts"].values()
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
