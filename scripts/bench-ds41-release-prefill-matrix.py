#!/usr/bin/env python3
"""Measure exact cold/retained prefill rows over the release context matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import urllib.request

from tokenizers import Tokenizer


BOS = "<｜begin▁of▁sentence｜>"
EOS = "<｜end▁of▁sentence｜>"
USER = "<｜User｜>"
ASSISTANT = "<｜Assistant｜>"
NO_THINK = "</think>"
DEFAULT_BASES = (0, 32_768, 65_536, 131_072, 262_144)
DEFAULT_SUFFIXES = (1_024, 2_048, 4_096, 8_192, 16_384, 32_768)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def token_count(tokenizer: Tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


def fit_body(
    tokenizer: Tokenizer,
    source_ids: list[int],
    before: str,
    after: str,
    target_tokens: int,
) -> tuple[str, int]:
    """Fit a decoded source prefix so the complete rendered prompt is exact."""

    fixed = token_count(tokenizer, before + after)
    if fixed > target_tokens:
        raise ValueError(f"fixed prompt {fixed} exceeds target {target_tokens}")

    def candidate(count: int) -> tuple[str, int]:
        body = tokenizer.decode(source_ids[:count], skip_special_tokens=False)
        return body, token_count(tokenizer, before + body + after)

    low, high = 0, min(len(source_ids), target_tokens - fixed + 512)
    best_body, best_tokens, best_source = "", fixed, 0
    while low <= high:
        middle = (low + high) // 2
        body, count = candidate(middle)
        if count <= target_tokens:
            if count > best_tokens:
                best_body, best_tokens, best_source = body, count, middle
            low = middle + 1
        else:
            high = middle - 1
    for count in range(
        max(0, best_source - 256), min(len(source_ids), best_source + 256) + 1
    ):
        body, tokens = candidate(count)
        if tokens == target_tokens:
            return body, tokens
        if best_tokens < tokens < target_tokens:
            best_body, best_tokens = body, tokens
    # BPE boundary merges can make a decoded source prefix skip an isolated
    # token count. These separated fillers each tokenize linearly in the
    # official tokenizer and close the small residual without changing the
    # inert-context semantics.
    for filler in (" q", " x", " |"):
        for count in range(1, target_tokens - best_tokens + 65):
            body = best_body + filler * count
            tokens = token_count(tokenizer, before + body + after)
            if tokens == target_tokens:
                return body, tokens
            if tokens > target_tokens + 8:
                break
    if best_tokens != target_tokens:
        raise RuntimeError(f"could not fit {target_tokens} tokens; got {best_tokens}")
    return best_body, best_tokens


def markers(tokenizer: Tokenizer, count: int) -> list[str]:
    result: list[str] = []
    seen: set[int] = set()
    for codepoint in range(0x3400, 0xA000):
        char = chr(codepoint)
        ids = tokenizer.encode(char, add_special_tokens=False).ids
        if len(ids) != 1 or ids[0] in seen:
            continue
        if tokenizer.decode(ids, skip_special_tokens=False) != char:
            continue
        result.append(char)
        seen.add(ids[0])
        if len(result) == count:
            return result
    raise RuntimeError(f"only found {len(result)} distinct one-token markers")


def stream_request(base_url: str, body: dict, timeout: float) -> dict:
    started = time.perf_counter()
    first_content = None
    finish = None
    content = []
    usage = None
    fingerprint = None
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            elapsed = time.perf_counter() - started
            raw = line[6:].strip()
            if raw == b"[DONE]":
                continue
            event = json.loads(raw)
            if event.get("error"):
                raise RuntimeError(event["error"])
            fingerprint = event.get("system_fingerprint", fingerprint)
            usage = event.get("usage", usage)
            for choice in event.get("choices", []):
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    first_content = elapsed if first_content is None else first_content
                    content.append(delta["content"])
                if choice.get("finish_reason") is not None:
                    finish = elapsed
    if first_content is None or finish is None or usage is None:
        raise RuntimeError("incomplete streaming response")
    return {
        "first_content_seconds": first_content,
        "finish_seconds": finish,
        "content": "".join(content),
        "usage": usage,
        "system_fingerprint": fingerprint,
    }


def request_body(messages: list[dict], max_tokens: int) -> dict:
    return {
        "model": "deepseek-ai/DeepSeek-V4.1-Flash",
        "messages": messages,
        "thinking": {"type": "disabled"},
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--context-file", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--base", type=int, action="append")
    parser.add_argument("--suffix", type=int, action="append")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    bases = args.base or list(DEFAULT_BASES)
    suffixes = args.suffix or list(DEFAULT_SUFFIXES)
    if args.repeats < 1 or any(value < 0 for value in bases) or any(
        value <= 0 for value in suffixes
    ):
        parser.error("invalid matrix dimensions")

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    source = args.context_file.read_text()
    source_ids = tokenizer.encode(source, add_special_tokens=False).ids
    if not source_ids:
        parser.error("context source is empty")
    needed = max(bases) + max(suffixes) + 2048
    source_ids *= math.ceil(needed / len(source_ids))
    unique = iter(markers(tokenizer, len(bases) * (2 + len(suffixes) * args.repeats * 4)))
    report = {
        "schema": 1,
        "scope": __doc__,
        "label": args.label,
        "base_url": args.base_url,
        "bases": bases,
        "suffixes": suffixes,
        "repeats": args.repeats,
        "controls": {"temperature": 0, "thinking": "disabled", "max_tokens": 1},
        "tokenizer_sha256": sha256(args.tokenizer.read_bytes()),
        "context_sha256": sha256(source.encode()),
        "started_ns": time.time_ns(),
        "primes": [],
        "samples": [],
        "passed": False,
    }

    def save() -> None:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    for base in bases:
        seed_prompt = None
        seed_content = None
        expected_parent_hit = 0
        if base:
            marker = next(unique)
            before = marker + f" {args.label} inert base context.\n"
            after = "\nIgnore the inert source and reply only OK."
            body_text, fitted = fit_body(
                tokenizer,
                source_ids,
                BOS + USER + before,
                after + ASSISTANT + NO_THINK,
                base,
            )
            assert fitted == base
            seed_prompt = before + body_text + after
            result = stream_request(
                args.base_url,
                request_body([{"role": "user", "content": seed_prompt}], 16),
                args.timeout,
            )
            if result["usage"]["prompt_tokens"] != base:
                raise RuntimeError("server disagrees with fitted base token count")
            seed_content = result["content"]
            expected_parent_hit = result["usage"]["total_tokens"] - 1
            report["primes"].append(
                {
                    "base_context_tokens": base,
                    "prompt_sha256": sha256(seed_prompt.encode()),
                    "result": result,
                    "expected_parent_hit": expected_parent_hit,
                }
            )
            save()
            print(
                f"prime base={base} completion={result['usage']['completion_tokens']} "
                f"expected_hit={expected_parent_hit}",
                flush=True,
            )

        for suffix in suffixes:
            for repeat in range(1, args.repeats + 1):
                target_total = expected_parent_hit + suffix if base else suffix
                attempts = []
                sample = {
                    "base_context_tokens": base,
                    "suffix_tokens": suffix,
                    "repeat": repeat,
                    "attempts": attempts,
                }
                report["samples"].append(sample)
                for attempt in range(1, 5):
                    marker = next(unique)
                    before = marker + f" {args.label} inert suffix {base}/{suffix}/{repeat}/{attempt}.\n"
                    after = "\nIgnore the inert source and output only the digit 7."
                    if base:
                        rendered_before = (
                            BOS
                            + USER
                            + seed_prompt
                            + ASSISTANT
                            + NO_THINK
                            + seed_content
                            + EOS
                            + USER
                            + before
                        )
                        messages = [
                            {"role": "user", "content": seed_prompt},
                            {"role": "assistant", "content": seed_content},
                        ]
                    else:
                        rendered_before = BOS + USER + before
                        messages = []
                    suffix_body, fitted = fit_body(
                        tokenizer,
                        source_ids,
                        rendered_before,
                        after + ASSISTANT + NO_THINK,
                        target_total,
                    )
                    assert fitted == target_total
                    prompt = before + suffix_body + after
                    messages.append({"role": "user", "content": prompt})
                    result = stream_request(
                        args.base_url, request_body(messages, 1), args.timeout
                    )
                    usage = result["usage"]
                    miss = usage["prompt_cache_miss_tokens"]
                    row = {
                        "attempt": attempt,
                        "target_prompt_tokens": target_total,
                        "prompt_sha256": sha256(
                            json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()
                        ),
                        "result": result,
                    }
                    attempts.append(row)
                    save()
                    if miss == suffix:
                        sample.update(
                            {
                                "prompt_tokens": usage["prompt_tokens"],
                                "cached_tokens": usage["prompt_cache_hit_tokens"],
                                "new_tokens": miss,
                                "first_content_seconds": result["first_content_seconds"],
                                "effective_prefill_tokens_per_second": miss
                                / result["first_content_seconds"],
                                "system_fingerprint": result["system_fingerprint"],
                                "passed": True,
                            }
                        )
                        save()
                        print(
                            f"measure base={base} suffix={suffix} repeat={repeat} "
                            f"cached={sample['cached_tokens']} "
                            f"seconds={sample['first_content_seconds']:.6f} "
                            f"tps={sample['effective_prefill_tokens_per_second']:.2f}",
                            flush=True,
                        )
                        break
                    target_total += suffix - miss
                else:
                    raise RuntimeError(
                        f"failed to construct exact {suffix}-token suffix at base {base}"
                    )

    report["cells"] = []
    for base in bases:
        for suffix in suffixes:
            rows = [
                row
                for row in report["samples"]
                if row["base_context_tokens"] == base
                and row["suffix_tokens"] == suffix
            ]
            values = [row["effective_prefill_tokens_per_second"] for row in rows]
            report["cells"].append(
                {
                    "base_context_tokens": base,
                    "suffix_tokens": suffix,
                    "samples": len(rows),
                    "median_effective_prefill_tokens_per_second": statistics.median(values),
                    "min_effective_prefill_tokens_per_second": min(values),
                    "max_effective_prefill_tokens_per_second": max(values),
                }
            )
    report["completed_ns"] = time.time_ns()
    report["passed"] = all(row.get("passed") for row in report["samples"])
    save()


if __name__ == "__main__":
    main()
