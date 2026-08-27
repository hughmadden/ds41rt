from __future__ import annotations

import sys
from pathlib import Path

import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from summarize_openai_metrics_capture import summarize  # noqa: E402


def record(index: int, *, output_tokens: int = 128) -> dict:
    return {
        "id": f"chatcmpl-{index}",
        "status": 200,
        "requested_model": "mixed-v3",
        "requested_max_tokens": 128,
        "requested_min_tokens": 128,
        "requested_ignore_eos": True,
        "requested_stream": True,
        "requested_temperature": 0,
        "requested_enable_thinking": False,
        "prompt_tokens": 2048 + index,
        "output_tokens": output_tokens,
        "mtp_verify_cycles": 50,
        "mtp_draft_tokens": 250,
        "mtp_accepted_draft_tokens": 75,
    }


def test_summarizes_exact_measured_capture() -> None:
    result = summarize(
        [record(0), record(1), record(2)],
        model="mixed-v3",
        draft_policy="adaptive",
        runtime_commit="a" * 40,
        expected_requests=3,
        expected_output_tokens=128,
    )
    assert result["measured_requests"] == 3
    assert result["observed_output_tokens"] == 384
    assert result["strict_acceptance"] == 0.3
    assert result["accepted_drafts_per_verify_cycle"] == 1.5


def test_rejects_warmup_or_missing_measured_request() -> None:
    with pytest.raises(ValueError, match="expected 3 measured requests"):
        summarize(
            [record(0), record(1)],
            model="mixed-v3",
            draft_policy="adaptive",
            runtime_commit="a" * 40,
            expected_requests=3,
            expected_output_tokens=128,
        )


def test_rejects_non_exact_output() -> None:
    with pytest.raises(ValueError, match="emitted 127"):
        summarize(
            [record(0, output_tokens=127)],
            model="mixed-v3",
            draft_policy="adaptive",
            runtime_commit="a" * 40,
            expected_requests=1,
            expected_output_tokens=128,
        )


def test_rejects_thinking_or_non_greedy_request() -> None:
    thinking = record(0)
    thinking["requested_enable_thinking"] = True
    with pytest.raises(ValueError, match="did not disable thinking"):
        summarize(
            [thinking],
            model="mixed-v3",
            draft_policy="adaptive",
            runtime_commit="a" * 40,
            expected_requests=1,
            expected_output_tokens=128,
        )

    sampled = record(0)
    sampled["requested_temperature"] = 0.6
    with pytest.raises(ValueError, match="was not greedy"):
        summarize(
            [sampled],
            model="mixed-v3",
            draft_policy="adaptive",
            runtime_commit="a" * 40,
            expected_requests=1,
            expected_output_tokens=128,
        )


def test_rejects_impossible_acceptance() -> None:
    value = record(0)
    value["mtp_accepted_draft_tokens"] = 251
    with pytest.raises(ValueError, match="more drafts than proposed"):
        summarize(
            [value],
            model="mixed-v3",
            draft_policy="adaptive",
            runtime_commit="a" * 40,
            expected_requests=1,
            expected_output_tokens=128,
        )
