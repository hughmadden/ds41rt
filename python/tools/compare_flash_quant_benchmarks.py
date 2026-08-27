#!/usr/bin/env python3
"""Compare matched Flash quant benchmarks and evaluate the 5% runtime gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any


CellKey = tuple[int, int, int, int, bool]
EXPECTED_CELLS = {
    (depth, concurrency, 2048, 128, False)
    for depth in (0, 4096, 8192)
    for concurrency in (1, 2, 4)
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-benchmark", type=Path, required=True)
    parser.add_argument("--candidate-benchmark", type=Path, required=True)
    parser.add_argument("--baseline-acceptance", type=Path, required=True)
    parser.add_argument("--candidate-acceptance", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--minimum-ratio", type=float, default=0.95)
    parser.add_argument(
        "--enforce",
        action="store_true",
        help="exit nonzero when one of the declared performance gates fails",
    )
    return parser.parse_args()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected one JSON object in {path}")
    return value


def positive_number(value: Any, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(f"invalid positive benchmark field {field}: {value!r}")
    return float(value)


def integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"invalid integer benchmark field {field}: {value!r}")
    return value


def cell_key(cell: dict[str, Any]) -> CellKey:
    context = integer(cell.get("context_size"), "context_size")
    concurrency = integer(cell.get("concurrency"), "concurrency", minimum=1)
    prompt = integer(cell.get("prompt_size"), "prompt_size", minimum=1)
    response = integer(cell.get("response_size"), "response_size", minimum=1)
    prefill_phase = cell.get("is_context_prefill_phase")
    if not isinstance(prefill_phase, bool):
        raise ValueError("benchmark cell has invalid is_context_prefill_phase")
    return context, concurrency, prompt, response, prefill_phase


def mean(cell: dict[str, Any], field: str) -> float:
    metric = cell.get(field)
    if not isinstance(metric, dict):
        raise ValueError(f"benchmark cell has no {field} metric")
    return positive_number(metric.get("mean"), f"{field}.mean")


def validate_samples(cell: dict[str, Any], field: str, expected: int) -> None:
    metric = cell.get(field)
    values = metric.get("values") if isinstance(metric, dict) else None
    if not isinstance(values, list) or len(values) != expected:
        raise ValueError(
            f"benchmark cell requires {expected} {field} samples, "
            f"found {0 if not isinstance(values, list) else len(values)}"
        )
    for value in values:
        positive_number(value, f"{field}.values")


def benchmark_cells(value: dict[str, Any]) -> tuple[str, dict[CellKey, dict[str, Any]]]:
    if value.get("latency_mode") != "generation":
        raise ValueError("benchmark must use generation latency mode")
    if value.get("prefix_caching_enabled") is not False:
        raise ValueError("benchmark must disable prefix caching")
    model = value.get("model")
    raw_cells = value.get("benchmarks")
    if not isinstance(model, str) or not model or not isinstance(raw_cells, list):
        raise ValueError("benchmark has no model or cell list")
    cells: dict[CellKey, dict[str, Any]] = {}
    for raw in raw_cells:
        if not isinstance(raw, dict):
            raise ValueError("benchmark cell is not an object")
        key = cell_key(raw)
        if key in cells:
            raise ValueError(f"benchmark contains duplicate cell {key}")
        # Validate every metric used below before accepting the cell.
        mean(raw, "tg_throughput")
        mean(raw, "pp_throughput")
        mean(raw, "e2e_ttft")
        validate_samples(raw, "tg_throughput", 3)
        validate_samples(raw, "pp_throughput", 3)
        validate_samples(raw, "e2e_ttft", 3 * key[1])
        cells[key] = raw
    if set(cells) != EXPECTED_CELLS:
        raise ValueError("benchmark does not contain the exact 3x3 Flash matrix")
    return model, cells


def acceptance(value: dict[str, Any], *, expected_model: str) -> dict[str, Any]:
    if value.get("schema") != "ds4rt-draft-acceptance-summary-v2":
        raise ValueError("acceptance summary has the wrong schema")
    if value.get("model") != expected_model:
        raise ValueError("acceptance summary and benchmark model differ")
    runtime_commit = value.get("runtime_commit")
    draft_policy = value.get("draft_policy")
    if (
        not isinstance(runtime_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", runtime_commit) is None
        or not isinstance(draft_policy, str)
        or not draft_policy
    ):
        raise ValueError("acceptance summary has no exact runtime or draft policy")
    requests = integer(value.get("measured_requests"), "measured_requests", minimum=1)
    requested = integer(
        value.get("requested_output_tokens"), "requested_output_tokens", minimum=1
    )
    observed = integer(
        value.get("observed_output_tokens"), "observed_output_tokens", minimum=1
    )
    if requested != observed:
        raise ValueError("acceptance summary did not complete exact output work")
    if requests != 63 or requested != 8_064:
        raise ValueError("acceptance summary is not the exact 63x128 work contract")
    proposed = integer(
        value.get("proposed_draft_tokens"), "proposed_draft_tokens", minimum=1
    )
    accepted = integer(value.get("accepted_draft_tokens"), "accepted_draft_tokens")
    rate = positive_number(value.get("strict_acceptance"), "strict_acceptance")
    if accepted > proposed or not math.isclose(
        rate, accepted / proposed, rel_tol=0.0, abs_tol=1e-15
    ):
        raise ValueError("acceptance summary totals and rate disagree")
    return {
        "measured_requests": requests,
        "requested_output_tokens": requested,
        "runtime_commit": runtime_commit,
        "draft_policy": draft_policy,
        "accepted_draft_tokens": accepted,
        "proposed_draft_tokens": proposed,
        "strict_acceptance": rate,
    }


def geometric_mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot compute an empty geometric mean")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def key_name(key: CellKey) -> str:
    context, concurrency, prompt, response, prefill_phase = key
    phase = "prefill" if prefill_phase else "generation"
    return f"d{context}-c{concurrency}-pp{prompt}-tg{response}-{phase}"


def compare(
    baseline_benchmark: dict[str, Any],
    candidate_benchmark: dict[str, Any],
    baseline_acceptance: dict[str, Any],
    candidate_acceptance: dict[str, Any],
    *,
    minimum_ratio: float,
) -> dict[str, Any]:
    if not math.isfinite(minimum_ratio) or not 0 < minimum_ratio <= 1:
        raise ValueError("minimum ratio must be in (0, 1]")
    baseline_version = baseline_benchmark.get("version")
    candidate_version = candidate_benchmark.get("version")
    if (
        not isinstance(baseline_version, str)
        or not baseline_version
        or candidate_version != baseline_version
    ):
        raise ValueError("baseline and candidate llama-benchy versions differ")
    baseline_model, baseline_cells = benchmark_cells(baseline_benchmark)
    candidate_model, candidate_cells = benchmark_cells(candidate_benchmark)
    if set(baseline_cells) != set(candidate_cells):
        raise ValueError("baseline and candidate benchmark cell sets differ")
    baseline_draft = acceptance(baseline_acceptance, expected_model=baseline_model)
    candidate_draft = acceptance(candidate_acceptance, expected_model=candidate_model)
    if (
        baseline_draft["measured_requests"] != candidate_draft["measured_requests"]
        or baseline_draft["requested_output_tokens"]
        != candidate_draft["requested_output_tokens"]
        or baseline_draft["draft_policy"] != candidate_draft["draft_policy"]
        or baseline_draft["runtime_commit"] != candidate_draft["runtime_commit"]
    ):
        raise ValueError("baseline and candidate acceptance work contracts differ")

    cells: dict[str, dict[str, Any]] = {}
    decode_ratios: list[float] = []
    prefill_ratios: list[float] = []
    ttft_ratios: list[float] = []
    for key in sorted(baseline_cells):
        baseline = baseline_cells[key]
        candidate = candidate_cells[key]
        baseline_decode = mean(baseline, "tg_throughput")
        candidate_decode = mean(candidate, "tg_throughput")
        baseline_prefill = mean(baseline, "pp_throughput")
        candidate_prefill = mean(candidate, "pp_throughput")
        baseline_ttft = mean(baseline, "e2e_ttft")
        candidate_ttft = mean(candidate, "e2e_ttft")
        decode_ratio = candidate_decode / baseline_decode
        prefill_ratio = candidate_prefill / baseline_prefill
        ttft_ratio = baseline_ttft / candidate_ttft
        decode_ratios.append(decode_ratio)
        prefill_ratios.append(prefill_ratio)
        ttft_ratios.append(ttft_ratio)
        cells[key_name(key)] = {
            "context_size": key[0],
            "concurrency": key[1],
            "prompt_size": key[2],
            "response_size": key[3],
            "baseline_decode_tps": baseline_decode,
            "candidate_decode_tps": candidate_decode,
            "decode_ratio": decode_ratio,
            "baseline_prefill_tps": baseline_prefill,
            "candidate_prefill_tps": candidate_prefill,
            "prefill_ratio": prefill_ratio,
            "baseline_ttft_ms": baseline_ttft,
            "candidate_ttft_ms": candidate_ttft,
            "inverse_ttft_ratio": ttft_ratio,
        }

    critical_keys = {
        "depth0_c1_decode": (0, 1, 2048, 128, False),
        "depth0_c4_decode": (0, 4, 2048, 128, False),
        "depth8192_c1_ingest": (8192, 1, 2048, 128, False),
    }
    missing = [name for name, key in critical_keys.items() if key not in baseline_cells]
    if missing:
        raise ValueError(f"benchmark is missing critical cells: {missing}")
    critical = {
        "depth0_c1_decode": cells[key_name(critical_keys["depth0_c1_decode"])][
            "decode_ratio"
        ],
        "depth0_c4_decode": cells[key_name(critical_keys["depth0_c4_decode"])][
            "decode_ratio"
        ],
        "depth8192_c1_ingest": cells[
            key_name(critical_keys["depth8192_c1_ingest"])
        ]["inverse_ttft_ratio"],
        "decode_geomean": geometric_mean(decode_ratios),
        "prefill_geomean": geometric_mean(prefill_ratios),
    }
    gates = {
        name: ratio >= minimum_ratio
        or math.isclose(ratio, minimum_ratio, rel_tol=0.0, abs_tol=1e-12)
        for name, ratio in critical.items()
    }
    return {
        "schema": "ds4rt-flash-quant-benchmark-comparison-v1",
        "baseline_model": baseline_model,
        "candidate_model": candidate_model,
        "llama_benchy_version": baseline_version,
        "minimum_ratio": minimum_ratio,
        "performance_gate_passed": all(gates.values()),
        "gates": gates,
        "critical_ratios": critical,
        "all_cell_geomeans": {
            "decode": geometric_mean(decode_ratios),
            "prefill": geometric_mean(prefill_ratios),
            "inverse_ttft": geometric_mean(ttft_ratios),
        },
        "acceptance": {
            "baseline": baseline_draft,
            "candidate": candidate_draft,
            "strict_acceptance_delta": (
                candidate_draft["strict_acceptance"]
                - baseline_draft["strict_acceptance"]
            ),
        },
        "cells": cells,
    }


def main() -> None:
    args = parse_args()
    report = compare(
        read_object(args.baseline_benchmark),
        read_object(args.candidate_benchmark),
        read_object(args.baseline_acceptance),
        read_object(args.candidate_acceptance),
        minimum_ratio=args.minimum_ratio,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if args.enforce and not report["performance_gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
