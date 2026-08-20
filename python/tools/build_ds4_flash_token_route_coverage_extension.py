#!/usr/bin/env python3
"""Append stable token-ID repetitions selected from natural Flash routes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct
from typing import Any

from tokenizers import Tokenizer

from build_ds4_flash_route_coverage_extension import (
    load_corpus,
    sha256_bytes,
    write_atomic,
)
from deepseek_v4_benchmark import (
    DS4_ASSISTANT,
    DS4_BOS,
    DS4_THINK_CLOSE,
    DS4_USER,
    render_nonthinking_messages,
)


ROUTE_RECORD = struct.Struct("<Hf")
POSITION_RECORD = struct.Struct("<Q")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ROUTE_RECORD_FORMAT = "u16le_expert_id_f32le_gate_weight"
POSITION_RECORD_FORMAT = "u64le_absolute_token_position"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--minimum-natural-routes-per-expert", type=int, default=1024)
    parser.add_argument("--token-repetitions", type=int, default=1024)
    parser.add_argument("--maximum-token-sequence-length", type=int, default=3)
    parser.add_argument("--maximum-extension-token-fraction", type=float, default=0.25)
    parser.add_argument("--retry-alignment-output", type=Path)
    parser.add_argument("--retry-alignment-manifest", type=Path)
    return parser.parse_args()


def checked_capture_payload(
    *, root: Path, relative: object, digest: object, label: str
) -> tuple[Path, bytes]:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} has no relative path")
    if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
        raise ValueError(f"{label} has an invalid SHA-256")
    path = (root / relative).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes the capture root: {path}") from error
    payload = path.read_bytes()
    if sha256_bytes(payload) != digest:
        raise ValueError(f"{label} digest changed: {path}")
    return path, payload


def rendered_ids(tokenizer: Tokenizer, prompt: str) -> list[int]:
    rendered = render_nonthinking_messages([{"role": "user", "content": prompt}])
    return tokenizer.encode(rendered, add_special_tokens=False).ids


def content_token_bounds(tokenizer: Tokenizer, prompt: str) -> tuple[list[int], int, int]:
    token_ids = rendered_ids(tokenizer, prompt)
    prefix_ids = tokenizer.encode(
        f"{DS4_BOS}{DS4_USER}", add_special_tokens=False
    ).ids
    suffix_ids = tokenizer.encode(
        f"{DS4_ASSISTANT}{DS4_THINK_CLOSE}", add_special_tokens=False
    ).ids
    if (
        len(token_ids) < len(prefix_ids) + len(suffix_ids)
        or token_ids[: len(prefix_ids)] != prefix_ids
        or (suffix_ids and token_ids[-len(suffix_ids) :] != suffix_ids)
    ):
        raise ValueError("tokenizer does not preserve the canonical non-thinking envelope")
    return token_ids, len(prefix_ids), len(token_ids) - len(suffix_ids)


def repeated_token_prompt(
    tokenizer: Tokenizer, *, token_id: int, repetitions: int
) -> tuple[str, int] | None:
    return repeated_token_sequence_prompt(
        tokenizer, token_ids=(token_id,), repetitions=repetitions
    )


def repeated_token_sequence_prompt(
    tokenizer: Tokenizer, *, token_ids: tuple[int, ...], repetitions: int
) -> tuple[str, int] | None:
    if not token_ids or repetitions <= 0:
        return None
    repeated_ids = list(token_ids) * repetitions
    prompt = tokenizer.decode(repeated_ids, skip_special_tokens=False)
    if not prompt or "\ufffd" in prompt:
        return None
    actual_ids, content_start, content_end = content_token_bounds(tokenizer, prompt)
    if actual_ids[content_start:content_end] != repeated_ids:
        return None
    return prompt, len(actual_ids)


def build_retry_alignment_extension(
    *,
    corpus_path: Path,
    corpus_payload: bytes,
    corpus: list[dict[str, Any]],
    progress_path: Path,
    progress_payload: bytes,
    prompts: list[dict[str, Any]],
    root: Path,
    unresolved: dict[tuple[int, int], int],
    top_k: int,
    routed_experts: int,
    base_tokens: int,
    maximum_extension_fraction: float,
    output_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Replay untried natural prompts when positioned rows lack a stable token."""
    replayed_source_indices = {
        int(record["route_alignment_source_index"])
        for record in corpus
        if "route_alignment_source_index" in record
    }
    candidate_contributions: dict[int, dict[tuple[int, int], int]] = {}
    candidate_tokens: dict[int, int] = {}
    for index, (record, prompt_record) in enumerate(zip(corpus, prompts, strict=True)):
        if (
            index in replayed_source_indices
            or "route_alignment_source_index" in record
            or "coverage_token_id" in record
            or "coverage_token_ids" in record
        ):
            continue
        layer_zero = next(
            (
                layer
                for layer in prompt_record.get("observed_layers", [])
                if layer.get("layer_id") == 0
            ),
            None,
        )
        if not isinstance(layer_zero, dict) or int(layer_zero.get("rows", 0)) <= 0:
            raise ValueError(f"retry-alignment prompt {index} has no token count")
        candidate_tokens[index] = int(layer_zero["rows"])
        contribution: dict[tuple[int, int], int] = {}
        for capture in prompt_record.get("capture_files", []):
            layer_id = int(capture.get("layer_id", -1))
            rows = int(capture.get("rows", 0))
            routes_per_row = int(capture.get("routes_per_row", 0))
            if (
                layer_id < 0
                or rows <= 0
                or routes_per_row != top_k
                or capture.get("route_record_format") != ROUTE_RECORD_FORMAT
            ):
                raise ValueError("retry-alignment capture has invalid geometry or format")
            route_path, route_payload = checked_capture_payload(
                root=root,
                relative=capture.get("route_path"),
                digest=capture.get("route_sha256"),
                label="retry-alignment route capture",
            )
            if len(route_payload) != rows * top_k * ROUTE_RECORD.size:
                raise ValueError(
                    f"retry-alignment route payload has invalid size: {route_path}"
                )
            for expert_id, _gate in ROUTE_RECORD.iter_unpack(route_payload):
                if expert_id >= routed_experts:
                    raise ValueError(
                        f"retry-alignment route has invalid expert {expert_id}: {route_path}"
                    )
                pair = (layer_id, expert_id)
                if pair in unresolved:
                    contribution[pair] = contribution.get(pair, 0) + 1
        if contribution:
            candidate_contributions[index] = contribution

    uncovered = set(unresolved)
    selected_indices: list[int] = []
    selected_tokens = 0
    token_budget = int(base_tokens * maximum_extension_fraction)
    while uncovered:
        best_index = -1
        best_score = (-1, -1, -1, 0, 0)
        for index, contribution in candidate_contributions.items():
            if index in selected_indices:
                continue
            covered = uncovered.intersection(contribution)
            if not covered:
                continue
            reliability = min(contribution[pair] for pair in covered)
            total_hits = sum(contribution[pair] for pair in covered)
            score = (
                len(covered),
                reliability,
                total_hits,
                -candidate_tokens[index],
                -index,
            )
            if score > best_score:
                best_index = index
                best_score = score
        if best_index < 0:
            raise ValueError(
                "no untried natural prompt retains retry-alignment evidence for "
                f"{sorted(uncovered)[:8]}"
            )
        next_tokens = candidate_tokens[best_index]
        if selected_tokens + next_tokens > token_budget:
            raise ValueError(
                f"retry alignment requires more than the {token_budget}-token budget"
            )
        selected_indices.append(best_index)
        selected_tokens += next_tokens
        uncovered.difference_update(candidate_contributions[best_index])

    alignment_sequence = sum(
        "route_alignment_source_index" in record for record in corpus
    )
    extension_records = [
        {
            **corpus[index],
            "id": f"alignment-{alignment_sequence + sequence:04d}",
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
        "schema": "ds4rt-flash-natural-route-token-alignment-retry-v1",
        "base_corpus": str(corpus_path.resolve()),
        "base_corpus_sha256": sha256_bytes(corpus_payload),
        "progress": str(progress_path.resolve()),
        "progress_sha256": sha256_bytes(progress_payload),
        "output": str(output_path.resolve()),
        "output_sha256": sha256_bytes(output_payload),
        "maximum_extension_token_fraction": maximum_extension_fraction,
        "base_tokens": base_tokens,
        "extension_tokens": selected_tokens,
        "extension_records": len(extension_records),
        "unresolved_pairs_before": [
            {"layer_id": pair[0], "expert_id": pair[1], "routes_needed": needed}
            for pair, needed in sorted(unresolved.items())
        ],
        "previously_replayed_source_indices": sorted(replayed_source_indices),
        "selected_source_indices": selected_indices,
        "policy": "greedy_untried_natural_prompt_replay_for_stable_token_alignment",
        "natural_routing": True,
        "forced_expert_activation": False,
    }
    write_atomic(output_path, output_payload)
    write_atomic(
        manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return manifest


def build_token_coverage_extension(
    *,
    corpus_path: Path,
    progress_path: Path,
    tokenizer_path: Path,
    output_path: Path,
    manifest_path: Path,
    minimum_routes: int,
    token_repetitions: int,
    maximum_extension_fraction: float,
    maximum_token_sequence_length: int = 3,
    retry_alignment_output_path: Path | None = None,
    retry_alignment_manifest_path: Path | None = None,
) -> dict[str, Any]:
    if output_path == manifest_path:
        raise ValueError("token-route output and manifest must be distinct")
    for path in (output_path, manifest_path):
        if path.exists():
            raise ValueError(f"token-route output already exists: {path}")
    if (retry_alignment_output_path is None) != (
        retry_alignment_manifest_path is None
    ):
        raise ValueError("retry-alignment output and manifest must be provided together")
    retry_paths = [
        path
        for path in (retry_alignment_output_path, retry_alignment_manifest_path)
        if path is not None
    ]
    if len({output_path, manifest_path, *retry_paths}) != 2 + len(retry_paths):
        raise ValueError("token-route and retry-alignment outputs must be distinct")
    for path in retry_paths:
        if path.exists():
            raise ValueError(f"retry-alignment output already exists: {path}")
    if minimum_routes <= 0 or token_repetitions <= 0:
        raise ValueError("route floor and token repetitions must be positive")
    if not 1 <= maximum_token_sequence_length <= 3:
        raise ValueError("maximum token sequence length must be in [1, 3]")
    if token_repetitions < minimum_routes:
        raise ValueError("token repetitions must be at least the natural-route floor")
    if (
        not math.isfinite(maximum_extension_fraction)
        or not 0.0 < maximum_extension_fraction <= 1.0
    ):
        raise ValueError("maximum extension token fraction must be in (0, 1]")

    corpus_payload, corpus = load_corpus(corpus_path)
    progress_payload = progress_path.read_bytes()
    progress = json.loads(progress_payload)
    prompts = progress.get("prompts")
    observed_counts = progress.get("observed_route_counts")
    observed_rows = progress.get("observed_rows_by_layer")
    top_k = int(progress.get("top_k", 0))
    routed_scaling_factor = float(progress.get("routed_scaling_factor", 0.0))
    if (
        not isinstance(prompts, list)
        or len(prompts) != len(corpus)
        or not isinstance(observed_counts, list)
        or not observed_counts
        or not isinstance(observed_rows, list)
        or len(observed_rows) != len(observed_counts)
        or top_k <= 0
        or not math.isfinite(routed_scaling_factor)
        or routed_scaling_factor <= 0.0
    ):
        raise ValueError("token-route progress is incomplete or malformed")
    routed_experts = len(observed_counts[0])
    if routed_experts <= 0 or any(
        not isinstance(layer, list) or len(layer) != routed_experts
        for layer in observed_counts
    ):
        raise ValueError("token-route progress has inconsistent expert geometry")
    counts = [[int(count) for count in layer] for layer in observed_counts]
    if any(count < 0 for layer in counts for count in layer):
        raise ValueError("token-route progress has a negative route count")
    normalized_rows = [int(rows) for rows in observed_rows]
    if any(rows <= 0 for rows in normalized_rows) or any(
        sum(layer) != normalized_rows[layer_id] * top_k
        for layer_id, layer in enumerate(counts)
    ):
        raise ValueError("token-route progress has inconsistent row/route accounting")
    deficient = {
        (layer_id, expert_id): minimum_routes - count
        for layer_id, layer in enumerate(counts)
        for expert_id, count in enumerate(layer)
        if count < minimum_routes
    }
    if not deficient:
        raise ValueError("corpus already satisfies the natural-route floor")

    tried_token_sequences: set[tuple[int, ...]] = set()
    prior_coverage_tokens = 0
    coverage_record_count = 0
    for index, (record, prompt_record) in enumerate(zip(corpus, prompts, strict=True)):
        if "coverage_token_id" not in record and "coverage_token_ids" not in record:
            continue
        layer_zero = next(
            (
                layer
                for layer in prompt_record.get("observed_layers", [])
                if layer.get("layer_id") == 0
            ),
            None,
        )
        if not isinstance(layer_zero, dict) or int(layer_zero.get("rows", 0)) <= 0:
            raise ValueError(f"token-route prompt {index} has no layer-zero token count")
        rows = int(layer_zero["rows"])
        raw_token_ids = record.get("coverage_token_ids")
        if raw_token_ids is None:
            raw_token_ids = [record.get("coverage_token_id")]
        if (
            not isinstance(raw_token_ids, list)
            or not raw_token_ids
            or len(raw_token_ids) > maximum_token_sequence_length
            or any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                for token_id in raw_token_ids
            )
        ):
            raise ValueError(f"token-route prompt {index} has invalid coverage token IDs")
        token_sequence = tuple(raw_token_ids)
        legacy_token_id = record.get("coverage_token_id")
        repetitions = record.get("coverage_token_repetitions")
        if (
            token_sequence in tried_token_sequences
            or (
                legacy_token_id is not None
                and token_sequence != (legacy_token_id,)
            )
            or isinstance(repetitions, bool)
            or not isinstance(repetitions, int)
            or repetitions <= 0
        ):
            raise ValueError(f"token-route prompt {index} has invalid coverage metadata")
        tried_token_sequences.add(token_sequence)
        prior_coverage_tokens += rows
        coverage_record_count += 1
    base_tokens = normalized_rows[0] - prior_coverage_tokens
    if base_tokens <= 0:
        raise ValueError("token-route corpus has no non-coverage token rows")

    tokenizer_payload = tokenizer_path.read_bytes()
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    special_token_ids = {
        token_id
        for token_id, token in tokenizer.get_added_tokens_decoder().items()
        if token.special
    }
    root = progress_path.parent.resolve()
    candidate_pairs: dict[tuple[int, ...], set[tuple[int, int]]] = {}
    seen_content_rows: set[tuple[int, int, int]] = set()
    positioned_rows = 0
    for index, (record, prompt_record) in enumerate(zip(corpus, prompts, strict=True)):
        prompt = record.get("prompt")
        if (
            not isinstance(prompt, str)
            or not isinstance(prompt_record, dict)
            or prompt_record.get("index") != index
            or prompt_record.get("id") != record.get("id")
            or prompt_record.get("prompt_sha256")
            != sha256_bytes(prompt.encode("utf-8"))
        ):
            raise ValueError(f"token-route prompt {index} does not match the corpus")
        prompt_token_ids, content_start, content_end = content_token_bounds(
            tokenizer, prompt
        )
        for capture in prompt_record.get("capture_files", []):
            if "position_path" not in capture:
                continue
            layer_id = int(capture.get("layer_id", -1))
            rows = int(capture.get("rows", 0))
            routes_per_row = int(capture.get("routes_per_row", 0))
            if (
                not 0 <= layer_id < len(counts)
                or rows <= 0
                or routes_per_row != top_k
                or capture.get("route_record_format") != ROUTE_RECORD_FORMAT
                or capture.get("position_record_format") != POSITION_RECORD_FORMAT
            ):
                raise ValueError("token-route capture has invalid geometry or formats")
            route_path, route_payload = checked_capture_payload(
                root=root,
                relative=capture.get("route_path"),
                digest=capture.get("route_sha256"),
                label="token-route route capture",
            )
            position_path, position_payload = checked_capture_payload(
                root=root,
                relative=capture.get("position_path"),
                digest=capture.get("position_sha256"),
                label="token-route position capture",
            )
            if len(route_payload) != rows * top_k * ROUTE_RECORD.size:
                raise ValueError(f"token-route route payload has invalid size: {route_path}")
            if len(position_payload) != rows * POSITION_RECORD.size:
                raise ValueError(
                    f"token-route position payload has invalid size: {position_path}"
                )
            decoded_routes = ROUTE_RECORD.iter_unpack(route_payload)
            for (position,) in POSITION_RECORD.iter_unpack(position_payload):
                row_routes = [next(decoded_routes) for _ in range(top_k)]
                if not content_start <= position < content_end:
                    continue
                row_key = (index, layer_id, position)
                if row_key in seen_content_rows:
                    raise ValueError(
                        "token-route capture repeats a prompt/layer/token position: "
                        f"{row_key}"
                    )
                seen_content_rows.add(row_key)
                seen_experts: set[int] = set()
                gate_sum = 0.0
                for expert_id, gate in row_routes:
                    if expert_id >= routed_experts or expert_id in seen_experts:
                        raise ValueError(
                            f"token-route row has invalid expert {expert_id}: {route_path}"
                        )
                    if not math.isfinite(gate) or gate < 0.0:
                        raise ValueError(
                            f"token-route row has invalid gate {gate}: {route_path}"
                        )
                    seen_experts.add(expert_id)
                    gate_sum += gate
                if abs(gate_sum - routed_scaling_factor) > 1.0e-3:
                    raise ValueError(
                        f"token-route row gate sum {gate_sum} does not match "
                        f"{routed_scaling_factor}: {route_path}"
                    )
                row_pairs = {
                    (layer_id, expert_id)
                    for expert_id, _gate in row_routes
                    if (layer_id, expert_id) in deficient
                }
                token_id = prompt_token_ids[position]
                candidate_pairs.setdefault((token_id,), set()).update(row_pairs)
                if (token_id,) in tried_token_sequences or token_id in special_token_ids:
                    for sequence_length in range(
                        2, maximum_token_sequence_length + 1
                    ):
                        first_start = max(content_start, position - sequence_length + 1)
                        last_start = min(position, content_end - sequence_length)
                        for sequence_start in range(first_start, last_start + 1):
                            token_sequence = tuple(
                                prompt_token_ids[
                                    sequence_start : sequence_start + sequence_length
                                ]
                            )
                            candidate_pairs.setdefault(token_sequence, set()).update(
                                row_pairs
                            )
                positioned_rows += 1

    stable_candidates: dict[tuple[int, ...], set[tuple[int, int]]] = {}
    for token_sequence, pairs in candidate_pairs.items():
        if (
            not special_token_ids.intersection(token_sequence)
            and token_sequence not in tried_token_sequences
            and repeated_token_sequence_prompt(
                tokenizer, token_ids=token_sequence, repetitions=3
            )
            is not None
        ):
            stable_candidates[token_sequence] = pairs
    remaining = dict(deficient)
    selected_token_sequences: list[tuple[int, ...]] = []
    while remaining:
        best_token_sequence: tuple[int, ...] | None = None
        best_score: tuple[float, int, int, int, tuple[int, ...]] = (
            -1.0,
            -1,
            -1,
            0,
            (),
        )
        for token_sequence, pairs in stable_candidates.items():
            if token_sequence in selected_token_sequences:
                continue
            gain = sum(
                min(token_repetitions, remaining.get(pair, 0)) for pair in pairs
            )
            score = (
                gain / len(token_sequence),
                gain,
                len(pairs.intersection(remaining)),
                -len(token_sequence),
                tuple(-token_id for token_id in token_sequence),
            )
            if score > best_score:
                best_token_sequence = token_sequence
                best_score = score
        if best_token_sequence is None or best_score[0] <= 0:
            unresolved = sorted(remaining.items(), key=lambda item: -item[1])[:8]
            if (
                retry_alignment_output_path is not None
                and retry_alignment_manifest_path is not None
            ):
                return build_retry_alignment_extension(
                    corpus_path=corpus_path,
                    corpus_payload=corpus_payload,
                    corpus=corpus,
                    progress_path=progress_path,
                    progress_payload=progress_payload,
                    prompts=prompts,
                    root=root,
                    unresolved=remaining,
                    top_k=top_k,
                    routed_experts=routed_experts,
                    base_tokens=normalized_rows[0],
                    maximum_extension_fraction=maximum_extension_fraction,
                    output_path=retry_alignment_output_path,
                    manifest_path=retry_alignment_manifest_path,
                )
            raise ValueError(
                "positioned natural routes contain no stable token for deficient pairs: "
                f"{unresolved}"
            )
        selected_token_sequences.append(best_token_sequence)
        for pair in stable_candidates[best_token_sequence]:
            if pair not in remaining:
                continue
            left = remaining[pair] - token_repetitions
            if left <= 0:
                del remaining[pair]
            else:
                remaining[pair] = left

    extension_records = []
    extension_tokens = 0
    for sequence, token_sequence in enumerate(
        selected_token_sequences, start=coverage_record_count
    ):
        repeated = repeated_token_sequence_prompt(
            tokenizer, token_ids=token_sequence, repetitions=token_repetitions
        )
        if repeated is None:
            raise ValueError(
                f"selected token sequence {token_sequence} is not repetition-stable"
            )
        prompt, prompt_tokens = repeated
        extension_tokens += prompt_tokens
        record = {
            "id": f"token-coverage-{sequence:04d}",
            "prompt": prompt,
            "max_tokens": 2,
            "coverage_token_ids": list(token_sequence),
            "coverage_token_repetitions": token_repetitions,
        }
        if len(token_sequence) == 1:
            record["coverage_token_id"] = token_sequence[0]
        extension_records.append(record)
    token_budget = int(base_tokens * maximum_extension_fraction)
    cumulative_extension_tokens = prior_coverage_tokens + extension_tokens
    if cumulative_extension_tokens > token_budget:
        raise ValueError(
            "cumulative token-route coverage requires "
            f"{cumulative_extension_tokens} tokens, over budget {token_budget}"
        )

    output_payload = corpus_payload
    if output_payload and not output_payload.endswith(b"\n"):
        output_payload += b"\n"
    output_payload += b"".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
        for record in extension_records
    )
    selected_token_payload = b"".join(
        struct.pack("<I", len(token_sequence))
        + b"".join(struct.pack("<I", token_id) for token_id in token_sequence)
        for token_sequence in selected_token_sequences
    )
    tried_token_payload = b"".join(
        struct.pack("<I", len(token_sequence))
        + b"".join(struct.pack("<I", token_id) for token_id in token_sequence)
        for token_sequence in sorted(tried_token_sequences)
    )
    manifest = {
        "schema": "ds4rt-flash-natural-token-route-coverage-extension-v1",
        "base_corpus": str(corpus_path.resolve()),
        "base_corpus_sha256": sha256_bytes(corpus_payload),
        "progress": str(progress_path.resolve()),
        "progress_sha256": sha256_bytes(progress_payload),
        "tokenizer": str(tokenizer_path.resolve()),
        "tokenizer_sha256": sha256_bytes(tokenizer_payload),
        "output": str(output_path.resolve()),
        "output_sha256": sha256_bytes(output_payload),
        "minimum_natural_routes_per_expert": minimum_routes,
        "token_repetitions": token_repetitions,
        "maximum_token_sequence_length": maximum_token_sequence_length,
        "maximum_extension_token_fraction": maximum_extension_fraction,
        "base_tokens": base_tokens,
        "prior_coverage_records": coverage_record_count,
        "prior_coverage_tokens": prior_coverage_tokens,
        "extension_tokens": extension_tokens,
        "cumulative_extension_tokens": cumulative_extension_tokens,
        "extension_content_tokens": sum(
            len(token_sequence) * token_repetitions
            for token_sequence in selected_token_sequences
        ),
        "extension_records": len(extension_records),
        "deficient_pairs_before": len(deficient),
        "positioned_natural_rows": positioned_rows,
        "stable_candidate_token_ids": sum(
            len(token_sequence) == 1 for token_sequence in stable_candidates
        ),
        "stable_candidate_token_sequences": len(stable_candidates),
        "selected_token_ids": sum(
            len(token_sequence) == 1 for token_sequence in selected_token_sequences
        ),
        "selected_token_sequences": len(selected_token_sequences),
        "selected_token_ids_sha256": hashlib.sha256(selected_token_payload).hexdigest(),
        "tried_token_ids": sum(
            len(token_sequence) == 1 for token_sequence in tried_token_sequences
        ),
        "tried_token_sequences": len(tried_token_sequences),
        "tried_token_ids_sha256": hashlib.sha256(tried_token_payload).hexdigest(),
        "minimum_predicted_routes": minimum_routes,
        "policy": "greedy_valid_token_or_ngram_set_cover_repeated_to_route_floor",
        "natural_routing": True,
        "sample_isolated": True,
        "forced_expert_activation": False,
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
        manifest = build_token_coverage_extension(
            corpus_path=args.corpus.expanduser().resolve(strict=True),
            progress_path=args.progress.expanduser().resolve(strict=True),
            tokenizer_path=args.tokenizer.expanduser().resolve(strict=True),
            output_path=args.output.expanduser().resolve(),
            manifest_path=args.manifest.expanduser().resolve(),
            minimum_routes=args.minimum_natural_routes_per_expert,
            token_repetitions=args.token_repetitions,
            maximum_token_sequence_length=args.maximum_token_sequence_length,
            maximum_extension_fraction=args.maximum_extension_token_fraction,
            retry_alignment_output_path=(
                None
                if args.retry_alignment_output is None
                else args.retry_alignment_output.expanduser().resolve()
            ),
            retry_alignment_manifest_path=(
                None
                if args.retry_alignment_manifest is None
                else args.retry_alignment_manifest.expanduser().resolve()
            ),
        )
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
