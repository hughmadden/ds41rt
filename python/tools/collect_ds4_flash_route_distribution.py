#!/usr/bin/env python3
"""Collect exact native Flash expert-route distributions for JSONL corpora."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
from typing import Any

from collect_ds4_flash_calibration_activations import (
    NATIVE_RECIPE,
    load_corpus,
    request_completion,
    validate_runtime_evidence,
)
from collect_expert_route_bank import RouteFragment, read_trace_delta


SCHEMA = "ds41rt-flash-natural-route-distribution-v1"
PROGRESS_SCHEMA = "ds41rt-flash-natural-route-progress-v1"
BASE_LAYERS = 43
LAYERS = 46
EXPERTS = 256
TOP_K = 6


def corpus_argument(value: str) -> tuple[str, Path]:
    try:
        label, raw_path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("corpus must be LABEL=PATH") from error
    if not label or not label.replace("-", "_").isalnum():
        raise argparse.ArgumentTypeError("corpus label must be alphanumeric/hyphen")
    if not raw_path:
        raise argparse.ArgumentTypeError("corpus path must not be empty")
    return label, Path(raw_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", action="append", type=corpus_argument, required=True)
    parser.add_argument("--trace-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--url", default="http://127.0.0.1:8000/v1/chat/completions"
    )
    parser.add_argument(
        "--model", default="deepseek-ai/DeepSeek-V4-Flash-0731-full"
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def empty_counts() -> list[list[int]]:
    return [[0] * EXPERTS for _ in range(LAYERS)]


def empty_rows() -> list[int]:
    return [0] * LAYERS


def add_counts(left: list[list[int]], right: list[list[int]]) -> list[list[int]]:
    return [
        [left[layer][expert] + right[layer][expert] for expert in range(EXPERTS)]
        for layer in range(LAYERS)
    ]


def add_rows(left: list[int], right: list[int]) -> list[int]:
    return [left[layer] + right[layer] for layer in range(LAYERS)]


def validate_fragments(
    fragments: list[RouteFragment], runtime: dict[str, Any]
) -> tuple[list[list[int]], list[int], dict[str, int]]:
    if not fragments:
        raise ValueError("request produced no exact row-route fragments")
    layer_ids = {fragment.layer_id for fragment in fragments}
    if not set(range(BASE_LAYERS)).issubset(layer_ids) or any(
        layer_id not in range(LAYERS) for layer_id in layer_ids
    ):
        missing = sorted(set(range(BASE_LAYERS)) - layer_ids)
        extra = sorted(layer_ids - set(range(LAYERS)))
        raise ValueError(
            f"route trace layer coverage mismatch: missing={missing} extra={extra}"
        )
    counts = empty_counts()
    rows = empty_rows()
    cohorts: dict[str, int] = {}
    for fragment in fragments:
        rows[fragment.layer_id] += fragment.physical_m
        cohort = "+".join(fragment.source_kinds)
        cohorts[cohort] = cohorts.get(cohort, 0) + fragment.physical_m
        for route_row in fragment.routes:
            if len(route_row) != TOP_K:
                raise ValueError("route fragment does not carry exact top-k 6")
            for expert_id in route_row:
                counts[fragment.layer_id][expert_id] += 1
    base_rows = rows[:BASE_LAYERS]
    if len(set(base_rows)) != 1:
        raise ValueError(f"base-layer route row totals differ: {base_rows}")
    traced_rows = sum(base_rows)
    expected_rows = int(runtime["request_expert_batch_rows"])
    expected_routes = int(runtime["request_expert_batch_routes"])
    if (
        expected_rows < BASE_LAYERS
        or expected_rows % BASE_LAYERS
        or expected_routes != expected_rows * TOP_K
        or expected_rows > traced_rows
    ):
        raise ValueError(
            "terminal runtime route group is inconsistent with the full trace: "
            f"full_rows={traced_rows} terminal_rows={expected_rows} "
            f"terminal_routes={expected_routes}"
        )
    if any(sum(counts[layer]) != rows[layer] * TOP_K for layer in range(LAYERS)):
        raise ValueError("per-layer route counts do not equal rows times top-k")
    return counts, rows, cohorts


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def layer_summary(layer_id: int, counts: list[int], rows: int) -> dict[str, Any]:
    total = sum(counts)
    if total != rows * TOP_K or total < 1:
        raise ValueError(f"layer {layer_id} has invalid route total {total} for {rows} rows")
    ordered = sorted(counts)
    mean = total / EXPERTS
    probabilities = [count / total for count in counts if count]
    entropy = -sum(value * math.log(value) for value in probabilities)
    return {
        "layer_id": layer_id,
        "rows": rows,
        "routes": total,
        "counts": counts,
        "zero_hit_experts": sum(count == 0 for count in counts),
        "under_32_hit_experts": sum(count < 32 for count in counts),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p95": ordered[math.ceil(0.95 * EXPERTS) - 1],
        "max": ordered[-1],
        "mean": mean,
        "max_to_mean": ordered[-1] / mean,
        "normalized_entropy": entropy / math.log(EXPERTS),
    }


def distribution_summary(
    counts: list[list[int]], rows: list[int]
) -> dict[str, Any]:
    layers = [
        layer_summary(layer_id, counts[layer_id], rows[layer_id])
        for layer_id in range(LAYERS)
    ]
    global_counts = [
        sum(counts[layer][expert] for layer in range(LAYERS))
        for expert in range(EXPERTS)
    ]
    return {
        "rows": sum(rows),
        "routes": sum(sum(layer) for layer in counts),
        "layers": layers,
        "global_expert_counts": global_counts,
    }


def write_csv(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("split", "layer", "expert", "count", "fraction"))
            for split, distribution in report["distributions"].items():
                for layer in distribution["layers"]:
                    for expert_id, count in enumerate(layer["counts"]):
                        writer.writerow(
                            (
                                split,
                                layer["layer_id"],
                                expert_id,
                                count,
                                count / layer["routes"],
                            )
                        )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def collect(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.expanduser().resolve()
    csv_path = output.with_suffix(".csv")
    progress_path = output.with_suffix(".progress.json")
    if output.exists() or csv_path.exists():
        raise ValueError(f"refusing to overwrite route distribution output: {output}")
    if progress_path.exists() and not args.resume:
        raise ValueError(f"route progress exists; pass --resume: {progress_path}")
    if args.resume and not progress_path.exists():
        raise ValueError(f"cannot resume missing route progress: {progress_path}")
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    trace_log = args.trace_log.expanduser().resolve(strict=True)
    corpus_records: dict[str, list[dict[str, Any]]] = {}
    corpus_manifest: dict[str, dict[str, Any]] = {}
    schedule: list[tuple[str, int, dict[str, Any]]] = []
    labels: set[str] = set()
    for label, raw_path in args.corpus:
        if label in labels:
            raise ValueError(f"duplicate corpus label {label!r}")
        labels.add(label)
        path = raw_path.expanduser().resolve(strict=True)
        records, digest = load_corpus(path)
        corpus_records[label] = records
        corpus_manifest[label] = {
            "path": str(path),
            "sha256": digest,
            "prompts": len(records),
        }
        schedule.extend((label, index, record) for index, record in enumerate(records))

    if args.resume:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if (
            progress.get("schema") != PROGRESS_SCHEMA
            or progress.get("corpora") != corpus_manifest
            or progress.get("checkpoint") != str(checkpoint)
        ):
            raise ValueError("route progress identity does not match this run")
        prompt_records = progress["prompts"]
        counts_by_split = progress["counts_by_split"]
        rows_by_split = progress["rows_by_split"]
        cohorts_by_split = progress["cohorts_by_split"]
        if len(prompt_records) > len(schedule):
            raise ValueError("route progress exceeds corpus schedule")
    else:
        prompt_records = []
        counts_by_split = {label: empty_counts() for label in labels}
        rows_by_split = {label: empty_rows() for label in labels}
        cohorts_by_split = {label: {} for label in labels}

    def checkpoint_progress() -> None:
        write_json_atomic(
            progress_path,
            {
                "schema": PROGRESS_SCHEMA,
                "checkpoint": str(checkpoint),
                "corpora": corpus_manifest,
                "prompts": prompt_records,
                "counts_by_split": counts_by_split,
                "rows_by_split": rows_by_split,
                "cohorts_by_split": cohorts_by_split,
            },
        )

    with trace_log.open("rb") as trace:
        trace.seek(0, os.SEEK_END)
        trace_offset = trace.tell()
        for schedule_index in range(len(prompt_records), len(schedule)):
            label, corpus_index, item = schedule[schedule_index]
            result = request_completion(
                url=args.url,
                model=args.model,
                prompt=item["prompt"],
                max_tokens=item["max_tokens"],
                timeout=args.timeout,
            )
            runtime = validate_runtime_evidence(
                result,
                model=args.model,
                checkpoint=checkpoint,
                expected_quantization_recipe=NATIVE_RECIPE,
                expected_spark_targets=4,
                expected_dspark="on",
                expected_concurrency=1,
                require_dspark_execution=False,
            )
            fragments, trace_offset = read_trace_delta(
                trace, trace_offset, TOP_K, EXPERTS
            )
            counts, rows, cohorts = validate_fragments(fragments, runtime)
            counts_by_split[label] = add_counts(counts_by_split[label], counts)
            rows_by_split[label] = add_rows(rows_by_split[label], rows)
            for cohort, cohort_rows in cohorts.items():
                cohorts_by_split[label][cohort] = (
                    cohorts_by_split[label].get(cohort, 0) + cohort_rows
                )
            prompt_records.append(
                {
                    "schedule_index": schedule_index,
                    "split": label,
                    "corpus_index": corpus_index,
                    "id": item["id"],
                    "prompt_sha256": hashlib.sha256(item["prompt"].encode()).hexdigest(),
                    "fragments": len(fragments),
                    "rows": sum(rows),
                    "routes": sum(sum(layer) for layer in counts),
                    "mtp_verify_cycles": runtime["mtp_verify_cycles"],
                }
            )
            checkpoint_progress()
            print(
                json.dumps(
                    {
                        "event": "route-prompt-captured",
                        "progress": len(prompt_records),
                        "scheduled": len(schedule),
                        "split": label,
                        "id": item["id"],
                        "rows": sum(rows),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    if sum(int(record["mtp_verify_cycles"]) for record in prompt_records) < 1:
        raise ValueError("route corpus did not execute native dSpark verification")
    combined_counts = empty_counts()
    combined_rows = empty_rows()
    for label in sorted(labels):
        combined_counts = add_counts(combined_counts, counts_by_split[label])
        combined_rows = add_rows(combined_rows, rows_by_split[label])
    distributions = {
        label: distribution_summary(counts_by_split[label], rows_by_split[label])
        for label in sorted(labels)
    }
    distributions["combined"] = distribution_summary(combined_counts, combined_rows)
    report = {
        "schema": SCHEMA,
        "checkpoint": str(checkpoint),
        "model": args.model,
        "top_k": TOP_K,
        "experts": EXPERTS,
        "layers": LAYERS,
        "corpora": corpus_manifest,
        "prompts": prompt_records,
        "cohort_rows_by_split": cohorts_by_split,
        "distributions": distributions,
    }
    write_json_atomic(output, report)
    write_csv(csv_path, report)
    return report


def main() -> None:
    try:
        report = collect(parse_args())
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(
        json.dumps(
            {
                label: {
                    "rows": value["rows"],
                    "routes": value["routes"],
                    "zero_hit_layer_experts": sum(
                        layer["zero_hit_experts"] for layer in value["layers"]
                    ),
                }
                for label, value in report["distributions"].items()
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
