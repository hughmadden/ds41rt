from __future__ import annotations

import sys
from pathlib import Path

import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from compare_flash_quant_benchmarks import compare  # noqa: E402


def metric(value: float, samples: int = 1) -> dict:
    return {"mean": value, "std": 0.0, "values": [value] * samples}


def benchmark(model: str, ratio: float = 1.0) -> dict:
    cells = []
    for depth in (0, 4096, 8192):
        for concurrency in (1, 2, 4):
            cells.append(
                {
                    "context_size": depth,
                    "concurrency": concurrency,
                    "prompt_size": 2048,
                    "response_size": 128,
                    "is_context_prefill_phase": False,
                    "tg_throughput": metric(100.0 * ratio, 3),
                    "pp_throughput": metric(2000.0 * ratio, 3),
                    "e2e_ttft": metric(1000.0 / ratio, 3 * concurrency),
                }
            )
    return {
        "model": model,
        "version": "0.4.0",
        "latency_mode": "generation",
        "prefix_caching_enabled": False,
        "benchmarks": cells,
    }


def draft(model: str, accepted: int = 300) -> dict:
    return {
        "schema": "ds4rt-draft-acceptance-summary-v2",
        "model": model,
        "runtime_commit": "a" * 40,
        "draft_policy": "adaptive",
        "measured_requests": 63,
        "requested_output_tokens": 8064,
        "observed_output_tokens": 8064,
        "accepted_draft_tokens": accepted,
        "proposed_draft_tokens": 1000,
        "strict_acceptance": accepted / 1000,
    }


def test_accepts_candidate_at_five_percent_floor() -> None:
    report = compare(
        benchmark("k2"),
        benchmark("mixed", 0.95),
        draft("k2"),
        draft("mixed", 350),
        minimum_ratio=0.95,
    )
    assert report["performance_gate_passed"] is True
    assert all(report["gates"].values())
    assert report["acceptance"]["strict_acceptance_delta"] == pytest.approx(0.05)


def test_rejects_critical_decode_regression() -> None:
    candidate = benchmark("mixed")
    candidate["benchmarks"][0]["tg_throughput"] = metric(94.0, 3)
    report = compare(
        benchmark("k2"),
        candidate,
        draft("k2"),
        draft("mixed"),
        minimum_ratio=0.95,
    )
    assert report["performance_gate_passed"] is False
    assert report["gates"]["depth0_c1_decode"] is False


def test_rejects_different_cell_or_exact_work_contract() -> None:
    candidate = benchmark("mixed")
    candidate["benchmarks"].pop()
    with pytest.raises(ValueError, match="exact 3x3 Flash matrix"):
        compare(
            benchmark("k2"),
            candidate,
            draft("k2"),
            draft("mixed"),
            minimum_ratio=0.95,
        )

    candidate_draft = draft("mixed")
    candidate_draft["observed_output_tokens"] = 8000
    with pytest.raises(ValueError, match="did not complete exact output"):
        compare(
            benchmark("k2"),
            benchmark("mixed"),
            draft("k2"),
            candidate_draft,
            minimum_ratio=0.95,
        )
