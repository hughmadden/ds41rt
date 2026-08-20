#!/usr/bin/env python3
"""Append greedily selected natural-route prompt replays to a Flash corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import tempfile
from typing import Any


ROUTE_RECORD = struct.Struct("<Hf")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--minimum-natural-routes-per-expert", type=int, default=1024)
    parser.add_argument("--maximum-extension-token-fraction", type=float, default=0.25)
    return parser.parse_args()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_corpus(path: Path) -> tuple[bytes, list[dict[str, Any]]]:
    payload = path.read_bytes()
    records = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line]
    if not records or any(not isinstance(record, dict) for record in records):
        raise ValueError("coverage source corpus is empty or malformed")
    return payload, records


def write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError(f"coverage extension output already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def load_coverage_evidence(
    *, corpus_path: Path, progress_path: Path, minimum_routes: int
) -> tuple[
    bytes,
    list[dict[str, Any]],
    bytes,
    dict[str, Any],
    list[list[int]],
    list[int],
    dict[tuple[int, int], int],
    list[dict[tuple[int, int], int]],
    list[int],
]:
    corpus_payload, corpus = load_corpus(corpus_path)
    progress_payload = progress_path.read_bytes()
    progress = json.loads(progress_payload)
    prompts = progress.get("prompts")
    observed_counts = progress.get("observed_route_counts")
    observed_rows = progress.get("observed_rows_by_layer")
    if (
        not isinstance(prompts, list)
        or len(prompts) != len(corpus)
        or not isinstance(observed_counts, list)
        or not observed_counts
        or not isinstance(observed_rows, list)
        or len(observed_rows) != len(observed_counts)
    ):
        raise ValueError("coverage progress is incomplete or malformed")
    routed_experts = len(observed_counts[0])
    if routed_experts <= 0 or any(
        not isinstance(layer, list) or len(layer) != routed_experts
        for layer in observed_counts
    ):
        raise ValueError("coverage progress has inconsistent expert geometry")
    normalized_counts = [
        [int(count) for count in layer] for layer in observed_counts
    ]
    if any(count < 0 for layer in normalized_counts for count in layer):
        raise ValueError("coverage progress has a negative route count")
    normalized_rows = [int(rows) for rows in observed_rows]
    if any(rows <= 0 for rows in normalized_rows):
        raise ValueError("coverage progress has an invalid observed row count")
    deficient = {
        (layer_id, expert_id): minimum_routes - count
        for layer_id, layer in enumerate(normalized_counts)
        for expert_id, count in enumerate(layer)
        if count < minimum_routes
    }
    if not deficient:
        raise ValueError("base corpus already satisfies the natural-route floor")

    root = progress_path.parent.resolve()
    candidates: list[dict[tuple[int, int], int]] = []
    candidate_tokens: list[int] = []
    aggregate = {pair: 0 for pair in deficient}
    for index, (record, prompt) in enumerate(zip(corpus, prompts, strict=True)):
        prompt_text = record.get("prompt")
        if (
            not isinstance(prompt, dict)
            or prompt.get("index") != index
            or prompt.get("id") != record.get("id")
            or not isinstance(prompt_text, str)
            or prompt.get("prompt_sha256")
            != sha256_bytes(prompt_text.encode("utf-8"))
        ):
            raise ValueError(f"coverage prompt {index} does not match the corpus")
        layer_zero = next(
            (
                layer
                for layer in prompt.get("observed_layers", [])
                if layer.get("layer_id") == 0
            ),
            None,
        )
        if not isinstance(layer_zero, dict) or int(layer_zero.get("rows", 0)) <= 0:
            raise ValueError(f"coverage prompt {index} has no layer-zero token count")
        candidate_tokens.append(int(layer_zero["rows"]))
        contribution: dict[tuple[int, int], int] = {}
        for capture in prompt.get("capture_files", []):
            layer_id = int(capture.get("layer_id", -1))
            route_relative = capture.get("route_path")
            route_digest = capture.get("route_sha256")
            if not isinstance(route_relative, str) or not isinstance(route_digest, str):
                raise ValueError(f"coverage prompt {index} has an incomplete route capture")
            route_path = (root / route_relative).resolve(strict=True)
            try:
                route_path.relative_to(root)
            except ValueError as error:
                raise ValueError(
                    f"coverage route escapes the capture root: {route_path}"
                ) from error
            payload = route_path.read_bytes()
            if (
                not SHA256_RE.fullmatch(route_digest)
                or sha256_bytes(payload) != route_digest
            ):
                raise ValueError(f"coverage route digest changed: {route_path}")
            if len(payload) % ROUTE_RECORD.size:
                raise ValueError(f"coverage route payload is truncated: {route_path}")
            for expert_id, _ in ROUTE_RECORD.iter_unpack(payload):
                pair = (layer_id, expert_id)
                if pair in deficient:
                    contribution[pair] = contribution.get(pair, 0) + 1
                    aggregate[pair] += 1
        candidates.append(contribution)

    for pair, retained_count in aggregate.items():
        observed_count = normalized_counts[pair[0]][pair[1]]
        if retained_count != observed_count:
            raise ValueError(
                "deficient natural routes were not retained exactly for "
                f"layer/expert {pair}: retained={retained_count} observed={observed_count}"
            )
    return (
        corpus_payload,
        corpus,
        progress_payload,
        progress,
        normalized_counts,
        normalized_rows,
        deficient,
        candidates,
        candidate_tokens,
    )


def build_extension(
    *,
    corpus_path: Path,
    progress_path: Path,
    output_path: Path,
    manifest_path: Path,
    minimum_routes: int,
    maximum_extension_fraction: float,
) -> dict[str, Any]:
    if output_path == manifest_path:
        raise ValueError("coverage extension output and manifest must be distinct")
    for path in (output_path, manifest_path):
        if path.exists():
            raise ValueError(f"coverage extension output already exists: {path}")
    if minimum_routes <= 0:
        raise ValueError("minimum natural routes must be positive")
    if (
        not math.isfinite(maximum_extension_fraction)
        or not 0.0 < maximum_extension_fraction <= 1.0
    ):
        raise ValueError("maximum extension token fraction must be in (0, 1]")
    (
        corpus_payload,
        corpus,
        progress_payload,
        _progress,
        observed_counts,
        observed_rows,
        deficient,
        candidates,
        candidate_tokens,
    ) = load_coverage_evidence(
        corpus_path=corpus_path,
        progress_path=progress_path,
        minimum_routes=minimum_routes,
    )

    base_tokens = int(observed_rows[0])
    token_budget = int(base_tokens * maximum_extension_fraction)
    remaining = dict(deficient)
    selected_indices: list[int] = []
    selected_tokens = 0
    while remaining:
        best_index = -1
        best_gain = 0
        best_efficiency = -1.0
        for index, contribution in enumerate(candidates):
            gain = sum(
                min(count, remaining.get(pair, 0))
                for pair, count in contribution.items()
            )
            efficiency = gain / candidate_tokens[index]
            if (efficiency, gain, -index) > (
                best_efficiency,
                best_gain,
                -best_index,
            ):
                best_index = index
                best_gain = gain
                best_efficiency = efficiency
        if best_index < 0 or best_gain <= 0:
            unresolved = sorted(remaining.items(), key=lambda item: -item[1])[:8]
            raise ValueError(
                "base corpus has no replay candidate for deficient natural routes: "
                f"{unresolved}"
            )
        next_tokens = candidate_tokens[best_index]
        if selected_tokens + next_tokens > token_budget:
            raise ValueError(
                f"natural-route replay requires more than the {token_budget}-token budget"
            )
        selected_indices.append(best_index)
        selected_tokens += next_tokens
        for pair, count in candidates[best_index].items():
            if pair not in remaining:
                continue
            left = remaining[pair] - count
            if left <= 0:
                del remaining[pair]
            else:
                remaining[pair] = left

    extension_records = []
    replay_counts: dict[str, int] = {}
    for sequence, index in enumerate(selected_indices):
        source = corpus[index]
        source_id = str(source["id"])
        replay_counts[source_id] = replay_counts.get(source_id, 0) + 1
        extension_records.append(
            {
                **source,
                "id": f"coverage-{sequence:04d}",
                "coverage_replay_of": source_id,
                "coverage_replay_ordinal": replay_counts[source_id],
            }
        )
    output_payload = corpus_payload
    if output_payload and not output_payload.endswith(b"\n"):
        output_payload += b"\n"
    output_payload += b"".join(
        (
            json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8")
            + b"\n"
        )
        for record in extension_records
    )
    manifest = {
        "schema": "ds4rt-flash-natural-route-coverage-extension-v1",
        "base_corpus": str(corpus_path.resolve()),
        "base_corpus_sha256": sha256_bytes(corpus_payload),
        "progress": str(progress_path.resolve()),
        "progress_sha256": sha256_bytes(progress_payload),
        "output": str(output_path.resolve()),
        "output_sha256": sha256_bytes(output_payload),
        "minimum_natural_routes_per_expert": minimum_routes,
        "maximum_extension_token_fraction": maximum_extension_fraction,
        "base_tokens": base_tokens,
        "extension_tokens": selected_tokens,
        "extension_records": len(extension_records),
        "deficient_pairs_before": len(deficient),
        "minimum_routes_before": min(count for layer in observed_counts for count in layer),
        "predicted_minimum_routes_after": minimum_routes,
        "policy": "greedy_natural_route_prompt_replay_set_cover",
        "forced_expert_activation": False,
        "selected_source_indices": selected_indices,
    }
    write_atomic(output_path, output_payload)
    write_atomic(
        manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return manifest


def main() -> None:
    args = parse_args()
    try:
        manifest = build_extension(
            corpus_path=args.corpus.expanduser().resolve(strict=True),
            progress_path=args.progress.expanduser().resolve(strict=True),
            output_path=args.output.expanduser().resolve(),
            manifest_path=args.manifest.expanduser().resolve(),
            minimum_routes=args.minimum_natural_routes_per_expert,
            maximum_extension_fraction=args.maximum_extension_token_fraction,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
