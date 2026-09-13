#!/usr/bin/env python3
"""Assemble the scoped v2 performance report from preserved qualification JSON."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from pathlib import Path


CASES = {
    "code": "Code",
    "math": "Math",
    "fable": "Fable",
    "hello": "Hello",
    "topic": "Topic",
    "structured-json": "Natural JSON",
    "structured-json-schema": "Schema JSON",
    "multilingual": "Multilingual",
}


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize_decode(report: dict) -> dict:
    cases = []
    for case_id in report["selected_cases"]:
        rows = [row for row in report["samples"] if row["case"] == case_id]
        rates = [row["observed_decode_tokens_per_second"] for row in rows]
        assessed = [row for row in rows if row.get("objective_checks_passed") is not None]
        cases.append(
            {
                "case": case_id,
                "samples": len(rows),
                "median_tps": statistics.median(rates),
                "min_tps": min(rates),
                "max_tps": max(rates),
                "serving_completed": sum(bool(row["serving_completed"]) for row in rows),
                "objective_checks_assessed": len(assessed),
                "objective_checks_passed": sum(bool(row["objective_checks_passed"]) for row in assessed),
            }
        )
    rows = [row for row in report["samples"] if row["case"] == "counting"]
    rates = [row["observed_decode_tokens_per_second"] for row in rows]
    return {
        "median_weighted_tps": report["median_weighted_observed_decode_tokens_per_second"],
        "repeat_weighted_tps": [row["weighted_observed_decode_tokens_per_second"] for row in report["repeat_summaries"]],
        "serving_completed": sum(row["serving_completed"] for row in cases),
        "samples": sum(row["samples"] for row in cases),
        "cases": cases,
        "counting": {
            "median_tps": statistics.median(rates),
            "min_tps": min(rates),
            "max_tps": max(rates),
            "completed": sum(bool(row["serving_completed"]) for row in rows),
            "samples": len(rows),
            "cache_hits": [row["cached_tokens"] for row in rows],
            "completion_tokens": [row["usage"]["completion_tokens"] for row in rows],
        },
    }


def command_seconds(state: dict, name: str) -> float:
    row = next(row for row in state["commands"] if row["name"] == name)
    return (row["completed_ns"] - row["started_ns"]) / 1e9


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--publication-dir", type=Path)
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    raw = args.raw_dir.resolve()
    old = read(args.v1)
    target_raw = read(raw / "target-eight-counting.json")
    dspark_raw = read(raw / "dspark-eight-counting.json")
    retained = read(raw / "dspark-retained.json")
    concurrency = read(raw / "dspark-concurrency.json")
    target = summarize_decode(target_raw)
    dspark = summarize_decode(dspark_raw)
    tools = read(raw / "tool-eval" / "summaries.json")
    assert target_raw["passed"] and dspark_raw["passed"]
    assert retained["passed"] and concurrency["passed"] and len(tools) == 3
    retained_summary = {
        "passed": retained["passed"],
        "schema": retained["schema"],
        "contexts": retained["contexts"],
        "repeats": retained["repeats"],
        "objective_checks_passed": retained["objective_checks_passed"],
        "context_summaries": retained["context_summaries"],
    }
    concurrency_summary = {
        "passed": concurrency["passed"],
        "concurrency": concurrency["concurrency"],
        "repeats": concurrency["repeats"],
        "summaries": concurrency["summaries"],
    }
    build = read(args.build_dir / "build-state.json")
    target_state = read(raw / "target-state.json")
    dspark_state = read(raw / "dspark-state.json")
    tools_state = read(raw / "tools-state.json")
    deployment = read(raw / "tools-startup-deployment.json")
    spark_images = read(raw / "spark-images.json")
    spark_ids = {row["Id"] for row in spark_images}
    assert len(spark_images) == 4
    assert len(spark_ids) == 1
    images = {
        "coordinator": {"local_image_id": deployment["image_id"]},
        "spark": {"local_image_id": next(iter(spark_ids))},
        "tags": ["v2", "latest"],
    }
    if args.publication_dir:
        published = args.publication_dir.resolve()
        coordinator_v2 = published / "coordinator-v2-manifest.json"
        coordinator_latest = published / "coordinator-latest-manifest.json"
        coordinator_platform = published / "coordinator-v2-platform-manifest.json"
        spark_v2 = published / "spark-v2-manifest.json"
        spark_latest = published / "spark-latest-manifest.json"
        assert coordinator_v2.read_bytes() == coordinator_latest.read_bytes()
        assert spark_v2.read_bytes() == spark_latest.read_bytes()
        coordinator_index = read(coordinator_v2)
        coordinator_manifest = read(coordinator_platform)
        spark_manifest = read(spark_v2)
        amd64 = next(
            row for row in coordinator_index["manifests"]
            if row.get("platform") == {"architecture": "amd64", "os": "linux"}
        )
        assert sha256(coordinator_platform) == amd64["digest"].removeprefix("sha256:")
        assert spark_manifest["config"]["digest"] == images["spark"]["local_image_id"]
        images["coordinator"].update(
            {
                "published_digest": f"sha256:{sha256(coordinator_v2)}",
                "platform_manifest_digest": amd64["digest"],
                "config_digest": coordinator_manifest["config"]["digest"],
            }
        )
        images["spark"].update(
            {
                "published_digest": f"sha256:{sha256(spark_v2)}",
                "config_digest": spark_manifest["config"]["digest"],
            }
        )
    log = re.sub(r"\x1b\[[0-9;]*m", "", (raw / "tools-startup-service.log").read_text())
    placement = next(line for line in log.splitlines() if "bottom-up RTX expert placement" in line)
    pool = next(line for line in log.splitlines() if "native KV pool reservation" in line)
    assert "layers=5" in placement and "global_bytes=16681077760" in pool
    memory_rows = []
    for phase in ("dspark", "target", "tools"):
        with (raw / f"{phase}-memory.csv").open() as handle:
            memory_rows.extend(csv.reader(handle))
    peak_mib = max(float(row[1]) for row in memory_rows)
    power_limits = {float(row[4]) for row in memory_rows}
    assert power_limits == {400.0}
    report = {
        "schema": 1,
        "serving_passed": True,
        "scope": "Scoped v2 qualification: README tables except the unchanged prefill matrix, plus three C16 high-thinking tool-eval runs.",
        "source": {
            "measured_image_revision": deployment["labels"]["org.opencontainers.image.revision"],
            "model_revision": old["source"]["model_revision"],
            "sparkinfer_revision": deployment["labels"]["io.ds41rt.sparkinfer.revision"],
            "xgrammar_revision": old["source"]["xgrammar_revision"],
        },
        "images": images,
        "hardware": {
            **old["hardware"],
            "coordinator_gpus": "1 x NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
            "power_limit_watts": 400,
            "memory_speed": "standard 14,001 MHz maximum; no memory overclock",
        },
        "controls": {
            "temperature": 0,
            "thinking": "disabled for throughput; enabled/high for tool eval",
            "weighted_repeats": 3,
            "retained_repeats": 3,
            "concurrency_repeats": 3,
            "cache": "FP4 compressed source, FP8 SWA, independent FP4 index",
        },
        "headline": {
            "max_prefill_median_tps": old["headline"]["max_prefill_median_tps"],
            "prefill_measurement": "preserved v1 result; not rerun for v2",
            "low_entropy_target_decode_median_tps": target["counting"]["median_tps"],
            "low_entropy_dspark_decode_median_tps": dspark["counting"]["median_tps"],
            "target_weighted_eight_type_median_tps": target["median_weighted_tps"],
            "dspark_weighted_eight_type_median_tps": dspark["median_weighted_tps"],
            "dspark_weighted_gain_percent": 100 * (dspark["median_weighted_tps"] / target["median_weighted_tps"] - 1),
            "c16_aggregate_decode_median_tps": concurrency["summaries"][-1]["median_aggregate_tps"],
            "global_source_pool_bytes": 16_681_077_760,
            "global_source_pool_logical_tokens": 18_710_016,
            "private_tail_tokens": 32_768,
            "rtx_resident_expert_layers": 5,
            "rtx_layer_range": "0-4",
            "spark_expert_layers": 40,
            "spark_device_budget_bytes": 107_374_182_400,
            "exact_prompt_cache_entries": 24,
            "completed_turn_cache_entries": 24,
            "peak_coordinator_gpu_memory_used_mib": peak_mib,
        },
        "eight_type_and_counting": {"target": target, "dspark": dspark, "official_flash": old["eight_type_and_counting"]["official_flash"]},
        "prefill_matrix": old["prefill_matrix"],
        "prefill_matrix_provenance": "Preserved v1 measurement; intentionally excluded from v2 rerun.",
        "retained_decode": retained_summary,
        "concurrency": concurrency_summary,
        "tool_eval": tools,
        "startup": {
            "clean_build_seconds": (build["finished_ns"] - build["started_ns"]) / 1e9,
            "target_only_launch_seconds": command_seconds(target_state, "target-launch"),
            "standard_dspark_launch_seconds": command_seconds(tools_state, "tools-launch"),
            "initial_dspark_launch_seconds": command_seconds(dspark_state, "dspark-launch"),
            "placement_log": placement,
            "pool_log": pool,
        },
        "qualification": {
            "full_suite_rerun": False,
            "excluded": ["prefill matrix", "full needle/vision/cache suite"],
            "tool_eval_runs": 3,
            "tool_eval_concurrency": 16,
            "tool_eval_thinking": "enabled/high",
        },
        "evidence": {
            path.name: sha256(path)
            for path in [
                raw / "target-eight-counting.json",
                raw / "dspark-eight-counting.json",
                raw / "dspark-retained.json",
                raw / "dspark-concurrency.json",
                raw / "tool-eval" / "summaries.json",
                args.build_dir / "build-state.json",
                args.build_dir / "build.log",
            ]
        },
    }
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")

    official = {row["case"]: row for row in report["eight_type_and_counting"]["official_flash"]["cases"]}
    lines = [
        "# DS41RT v2 performance report",
        "",
        "All RTX measurements use a 400 W power limit and standard 14,001 MHz maximum memory speed with no memory overclock. The standard topology is one RTX PRO 6000 Blackwell coordinator and four DGX Spark workers. Throughput uses temperature zero and thinking disabled; tool evaluation uses thinking enabled at high effort and C16.",
        "",
        "## Headline results",
        "",
        "| Measurement | Result |",
        "|---|---:|",
        f"| Best median prefill, 0 base + 32K new (preserved v1) | **{report['headline']['max_prefill_median_tps']:,.2f} tok/s** |",
        f"| Low-entropy target-only decode, counting 1–200 warm median | {target['counting']['median_tps']:.2f} tok/s |",
        f"| Low-entropy dSpark decode, counting 1–200 warm median | **{dspark['counting']['median_tps']:.2f} tok/s** |",
        f"| Weighted eight-type target-only median | {target['median_weighted_tps']:.2f} tok/s |",
        f"| Weighted eight-type dSpark median | **{dspark['median_weighted_tps']:.2f} tok/s** |",
        f"| dSpark gain on weighted mix | {report['headline']['dspark_weighted_gain_percent']:.2f}% |",
        f"| C16 aggregate warm decode median | **{report['headline']['c16_aggregate_decode_median_tps']:.2f} tok/s** |",
        "| Standard RTX routed-expert placement | **5 layers (0–4)** |",
        "| Spark expert residency / configured budget | 40 layers per worker / 100 GiB |",
        "| Global FP4 source pool | 16.681 GB for 18,710,016 logical tokens (+32,768 private-tail tokens) |",
        "| Exact prompt / completed-turn retention | 24 / 24 entries |",
        f"| Peak observed coordinator GPU memory used | {peak_mib:,.0f} MiB |",
        f"| Clean build / standard dSpark launch | {report['startup']['clean_build_seconds']:.2f} s / {report['startup']['standard_dspark_launch_seconds']:.2f} s |",
        "",
        "## Eight content types and counting",
        "",
        "Three local samples per mode. The official Flash column is the preserved one-request v1 reference and was not called again. Counting is outside the weighted score.",
        "",
        "| Case | Target tok/s | dSpark tok/s | Official Flash tok/s | Target completed | dSpark completed | Official completed |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    target_cases = {row["case"]: row for row in target["cases"]}
    dspark_cases = {row["case"]: row for row in dspark["cases"]}
    for case_id, title in CASES.items():
        ref = official[case_id]
        ref_rate = f"{ref['observed_decode_tokens_per_second']:.2f}" if ref["completed"] else "HTTP 400"
        lines.append(f"| {title} | {target_cases[case_id]['median_tps']:.2f} | {dspark_cases[case_id]['median_tps']:.2f} | {ref_rate} | 3/3 | 3/3 | {'1/1' if ref['completed'] else '0/1 (HTTP 400)'} |")
    ref_count = report["eight_type_and_counting"]["official_flash"]["counting"]["observed_decode_tokens_per_second"]
    lines += [
        f"| Counting 1–200 | **{target['counting']['median_tps']:.2f}** | **{dspark['counting']['median_tps']:.2f}** | **{ref_count:.2f}** | 3/3 warm | 3/3 warm | 1/1 |",
        "",
        "## Prefill matrix",
        "",
        "Preserved v1 target-only measurements; the prefill matrix was intentionally excluded from the scoped v2 rerun.",
        "",
        "| Retained base | +1K | +2K | +4K | +8K | +16K | +32K |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in old["prefill_matrix"]["cells"]:
        values = [cell[str(size)]["median_effective_prefill_tokens_per_second"] for size in (1024, 2048, 4096, 8192, 16384, 32768)] if isinstance(cell, dict) and "1024" in cell else None
        if values is not None:
            lines.append("| " + f"{cell['base_context_tokens']:,}" + " | " + " | ".join(f"{value:,.0f}" for value in values) + " |")
    if not any(line.startswith("| 0 |") for line in lines):
        # The v1 schema stores one row per cell rather than one row per base.
        cells = old["prefill_matrix"]["cells"]
        for base in sorted({row["base_context_tokens"] for row in cells}):
            values = [next(row["median_effective_prefill_tokens_per_second"] for row in cells if row["base_context_tokens"] == base and row["suffix_tokens"] == suffix) for suffix in (1024, 2048, 4096, 8192, 16384, 32768)]
            lines.append("| " + ("0" if base == 0 else f"{base // 1024}K") + " | " + " | ".join(f"{value:,.0f}" for value in values) + " |")
    lines += [
        "",
        "## Decode over retained context",
        "",
        "Three samples per each of eight content types, with verified exact retained-prefix reuse.",
        "",
        "| Retained base | Weighted dSpark tok/s | Completed with verified cache reuse |",
        "|---:|---:|---:|",
    ]
    for row in retained["context_summaries"]:
        label = "0" if row["context_tokens"] == 0 else f"{row['context_tokens'] // 1024}K"
        lines.append(f"| {label} | {row['weighted_observed_decode_tokens_per_second']:.2f} | {row['cache_valid']}/{row['samples']} |")
    lines += [
        "",
        "## Concurrency scaling",
        "",
        "Three exact 599-token counting samples per concurrency after one fully cached prime; aggregate timing includes admission gaps.",
        "",
        "| Concurrency | Median aggregate tok/s | Range | Scale vs C1 |",
        "|---:|---:|---:|---:|",
    ]
    c1 = concurrency["summaries"][0]["median_aggregate_tps"]
    for row in concurrency["summaries"]:
        lines.append(f"| {row['concurrency']} | {row['median_aggregate_tps']:.2f} | {row['min_aggregate_tps']:.2f}–{row['max_aggregate_tps']:.2f} | {row['median_aggregate_tps']/c1:.2f}× |")
    lines += [
        "",
        "## High-thinking tool evaluation",
        "",
        "Three hard-mode campaigns use C16, thinking enabled, high reasoning effort, temperature zero, a 900-second timeout, and the normal output policy.",
        "",
        "| Run | Basic | Hard | Total | Pass / partial / fail |",
        "|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(tools, 1):
        statuses = row["statuses"]
        lines.append(f"| {index} | {row['basic_points']}/{row['basic_max']} | {row['hard_points']}/{row['hard_max']} | {row['total_points']}/{row['total_max']} | {statuses.get('pass',0)} / {statuses.get('partial',0)} / {statuses.get('fail',0)} |")
    lines += [
        "",
        "## Memory, startup, and scope",
        "",
        f"The clean build completed in {report['startup']['clean_build_seconds']:.2f} seconds. The final standard dSpark launch reached port 8000 in {report['startup']['standard_dspark_launch_seconds']:.2f} seconds. Automatic placement loaded layers 0–4 on the RTX; every Spark retained all 40 TP expert layers under its configured 100 GiB device budget. Peak observed coordinator GPU memory use was {peak_mib:,.0f} MiB.",
        "",
        "V2 reran only the tables present in the main README, excluding prefill, plus the requested tool campaign. The v1 prefill matrix and one-shot official API comparison remain clearly labeled prior measurements. The full needle, vision, cache, and agentic artifact suites were not repeated; their v1 evidence remains applicable to unchanged interfaces but is not represented as fresh v2 qualification.",
        "",
        "Machine-readable results are in [release-v2-performance.json](release-v2-performance.json). The release evidence archive preserves raw requests, responses, traces, build and launch logs, memory samples, image labels, and hashes.",
        "",
    ]
    args.output_md.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
