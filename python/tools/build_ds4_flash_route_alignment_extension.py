#!/usr/bin/env python3
"""Append a minimal natural-prompt replay set for token/route alignment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from build_ds4_flash_route_coverage_extension import (
    load_coverage_evidence,
    sha256_bytes,
    write_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--minimum-natural-routes-per-expert", type=int, default=1024)
    parser.add_argument("--maximum-extension-token-fraction", type=float, default=0.05)
    return parser.parse_args()


def build_alignment_extension(
    *,
    corpus_path: Path,
    progress_path: Path,
    output_path: Path,
    manifest_path: Path,
    minimum_routes: int,
    maximum_extension_fraction: float,
) -> dict[str, Any]:
    if output_path == manifest_path:
        raise ValueError("route-alignment output and manifest must be distinct")
    for path in (output_path, manifest_path):
        if path.exists():
            raise ValueError(f"route-alignment output already exists: {path}")
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

    uncovered = set(deficient)
    available = set(range(len(candidates)))
    selected_indices: list[int] = []
    selected_tokens = 0
    token_budget = int(observed_rows[0] * maximum_extension_fraction)
    while uncovered:
        best_index = -1
        best_score = (-1.0, -1, 0)
        for index in available:
            gain = len(uncovered.intersection(candidates[index]))
            score = (gain / candidate_tokens[index], gain, -index)
            if score > best_score:
                best_index = index
                best_score = score
        if best_index < 0 or best_score[1] <= 0:
            unresolved = sorted(uncovered)[:8]
            raise ValueError(
                "natural corpus has no token-alignment prompt for deficient routes: "
                f"{unresolved}"
            )
        next_tokens = candidate_tokens[best_index]
        if selected_tokens + next_tokens > token_budget:
            raise ValueError(
                f"token-alignment replay requires more than the {token_budget}-token budget"
            )
        selected_indices.append(best_index)
        selected_tokens += next_tokens
        available.remove(best_index)
        uncovered.difference_update(candidates[best_index])

    extension_records = [
        {
            **corpus[index],
            "id": f"alignment-{sequence:04d}",
            "route_alignment_of": str(corpus[index]["id"]),
            "route_alignment_source_index": index,
        }
        for sequence, index in enumerate(selected_indices)
    ]
    output_payload = corpus_payload
    if output_payload and not output_payload.endswith(b"\n"):
        output_payload += b"\n"
    output_payload += b"".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
        for record in extension_records
    )
    manifest = {
        "schema": "ds41rt-flash-natural-route-token-alignment-extension-v1",
        "base_corpus": str(corpus_path.resolve()),
        "base_corpus_sha256": sha256_bytes(corpus_payload),
        "progress": str(progress_path.resolve()),
        "progress_sha256": sha256_bytes(progress_payload),
        "output": str(output_path.resolve()),
        "output_sha256": sha256_bytes(output_payload),
        "minimum_natural_routes_per_expert": minimum_routes,
        "maximum_extension_token_fraction": maximum_extension_fraction,
        "base_tokens": observed_rows[0],
        "extension_tokens": selected_tokens,
        "extension_records": len(extension_records),
        "deficient_pairs_before": len(deficient),
        "minimum_routes_before": min(
            count for layer in observed_counts for count in layer
        ),
        "policy": "greedy_natural_prompt_set_cover_for_token_alignment",
        "natural_routing": True,
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
        manifest = build_alignment_extension(
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
