#!/usr/bin/env python3
"""Validate and summarize one measured-only DS41RT metrics-proxy capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-policy", required=True)
    parser.add_argument("--runtime-commit")
    parser.add_argument("--expected-requests", type=int, required=True)
    parser.add_argument("--expected-output-tokens", type=int, required=True)
    return parser.parse_args()


def read_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"metrics capture line {line_number} is not JSON: {error}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(
                    f"metrics capture line {line_number} is not an object"
                )
            records.append(value)
    if not records:
        raise ValueError("metrics capture is empty")
    return records


def positive_integer(value: Any, field: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"metrics capture has invalid {field}: {value!r}")
    return value


def summarize(
    records: list[dict[str, Any]],
    *,
    model: str,
    draft_policy: str,
    runtime_commit: str,
    expected_requests: int,
    expected_output_tokens: int,
) -> dict[str, Any]:
    if expected_requests <= 0 or expected_output_tokens <= 0:
        raise ValueError("expected request and output counts must be positive")
    if len(records) != expected_requests:
        raise ValueError(
            f"expected {expected_requests} measured requests, found {len(records)}"
        )
    ids: set[str] = set()
    proposed = 0
    accepted = 0
    cycles = 0
    prompt_tokens = 0
    output_tokens = 0
    for index, record in enumerate(records):
        request_id = record.get("id")
        if not isinstance(request_id, str) or not request_id or request_id in ids:
            raise ValueError(f"metrics capture request {index} has an invalid ID")
        ids.add(request_id)
        if record.get("status") != 200:
            raise ValueError(
                f"metrics capture request {request_id} returned {record.get('status')!r}"
            )
        if record.get("requested_stream") is not True:
            raise ValueError(f"metrics capture request {request_id} was not streamed")
        if record.get("requested_model") != model:
            raise ValueError(
                f"metrics capture request {request_id} selected a different model"
            )
        if record.get("requested_temperature") != 0:
            raise ValueError(
                f"metrics capture request {request_id} was not greedy"
            )
        if record.get("requested_enable_thinking") is not False:
            raise ValueError(
                f"metrics capture request {request_id} did not disable thinking"
            )
        if record.get("requested_max_tokens") != expected_output_tokens:
            raise ValueError(
                f"metrics capture request {request_id} did not request "
                f"exactly {expected_output_tokens} tokens"
            )
        if (
            record.get("requested_min_tokens") != expected_output_tokens
            or record.get("requested_ignore_eos") is not True
        ):
            raise ValueError(
                f"metrics capture request {request_id} did not enforce exact output"
            )
        observed = positive_integer(record.get("output_tokens"), "output_tokens")
        if observed != expected_output_tokens:
            raise ValueError(
                f"metrics capture request {request_id} emitted {observed}, "
                f"expected {expected_output_tokens} tokens"
            )
        request_proposed = positive_integer(
            record.get("mtp_draft_tokens"), "mtp_draft_tokens", allow_zero=True
        )
        request_accepted = positive_integer(
            record.get("mtp_accepted_draft_tokens"),
            "mtp_accepted_draft_tokens",
            allow_zero=True,
        )
        request_cycles = positive_integer(
            record.get("mtp_verify_cycles"), "mtp_verify_cycles", allow_zero=True
        )
        if request_accepted > request_proposed:
            raise ValueError(
                f"metrics capture request {request_id} accepted more drafts than proposed"
            )
        prompt_tokens += positive_integer(record.get("prompt_tokens"), "prompt_tokens")
        output_tokens += observed
        proposed += request_proposed
        accepted += request_accepted
        cycles += request_cycles
    if proposed == 0 or cycles == 0:
        raise ValueError("metrics capture did not execute dSpark verification")
    return {
        "schema": "ds41rt-draft-acceptance-summary-v2",
        "model": model,
        "runtime_commit": runtime_commit,
        "draft_policy": draft_policy,
        "measured_requests": len(records),
        "requested_output_tokens": expected_requests * expected_output_tokens,
        "observed_output_tokens": output_tokens,
        "prompt_tokens": prompt_tokens,
        "verify_cycles": cycles,
        "accepted_draft_tokens": accepted,
        "proposed_draft_tokens": proposed,
        "strict_acceptance": accepted / proposed,
        "accepted_drafts_per_verify_cycle": accepted / cycles,
        "proposed_drafts_per_verify_cycle": proposed / cycles,
    }


def git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def main() -> None:
    args = parse_args()
    summary = summarize(
        read_records(args.input),
        model=args.model,
        draft_policy=args.draft_policy,
        runtime_commit=args.runtime_commit or git_head(),
        expected_requests=args.expected_requests,
        expected_output_tokens=args.expected_output_tokens,
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
