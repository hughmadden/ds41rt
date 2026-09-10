#!/usr/bin/env python3
"""Collect bounded native Flash expert-input activations from a live DS41RT API."""

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
import urllib.request

from validate_ds4_flash_generation_ab import (
    NATIVE_RECIPE,
    validate_runtime_evidence,
)


SCHEMA = "ds41rt-flash-exl3-activation-corpus-v2"
PROGRESS_SCHEMA = "ds41rt-flash-exl3-activation-progress-v2"
LEGACY_CAPTURE_RE = re.compile(
    r"^layer_(?P<layer>[0-9]+)_rows_(?P<rows>[0-9]+)_expert_input\.bf16$"
)
ROUTED_CAPTURE_RE = re.compile(
    r"^capture_(?P<capture>[0-9]+_[0-9]+)_layer_(?P<layer>[0-9]+)_"
    r"rows_(?P<rows>[0-9]+)_expert_input\.bf16$"
)
OBSERVED_ROUTE_COUNTS_RE = re.compile(
    r"^capture_(?P<capture>[0-9]+_[0-9]+)_layer_(?P<layer>[0-9]+)_"
    r"rows_(?P<rows>[0-9]+)_observed_routes_u64\.bin$"
)
ROUTE_RECORD = struct.Struct("<Hf")
ROUTE_RECORD_FORMAT = "u16le_expert_id_f32le_gate_weight"
POSITION_RECORD = struct.Struct("<Q")
POSITION_RECORD_FORMAT = "u64le_absolute_token_position"
RETENTION_CONTROL_FILE = ".ds41rt_route_retention_control_v1.bin"
RETENTION_CONTROL_MAGIC = b"DS41RRC1"
IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--url", default="http://127.0.0.1:8000/v1/chat/completions"
    )
    parser.add_argument(
        "--model", default="deepseek-ai/DeepSeek-V4-Flash-0731-full"
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--retained-routes-per-expert",
        type=int,
        default=1024,
        help=(
            "retain joint activation/route rows until every layer/expert has at "
            "least this many retained natural routes"
        ),
    )
    parser.add_argument(
        "--minimum-natural-routes-per-expert",
        type=int,
        default=1024,
        help="fail unless the full natural-routing pass reaches this coverage floor",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-corpus-extension",
        action="store_true",
        help=(
            "when resuming, accept an appended corpus whose completed prefix "
            "matches the fsynced prompt identities"
        ),
    )
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_corpus(path: Path) -> tuple[list[dict[str, Any]], str]:
    raw = path.read_bytes()
    records: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid corpus JSON on line {line_number}: {error}") from error
        if not isinstance(record, dict):
            raise ValueError(f"corpus line {line_number} is not an object")
        identifier = record.get("id")
        prompt = record.get("prompt")
        max_tokens = record.get("max_tokens", 8)
        if not isinstance(identifier, str) or not IDENTIFIER_RE.fullmatch(identifier):
            raise ValueError(f"corpus line {line_number} has an invalid id")
        if identifier in identifiers:
            raise ValueError(f"duplicate corpus id {identifier!r}")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"corpus line {line_number} has an empty prompt")
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or not 2 <= max_tokens <= 256
        ):
            raise ValueError(
                f"corpus line {line_number} max_tokens must be in 2..256"
            )
        identifiers.add(identifier)
        records.append(
            {"id": identifier, "prompt": prompt, "max_tokens": max_tokens}
        )
    if not records:
        raise ValueError("calibration corpus is empty")
    return records, sha256_bytes(raw)


def checkpoint_shape(checkpoint: Path) -> tuple[int, int, int, int, int, float]:
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    hidden_size = int(config["hidden_size"])
    hidden_layers = int(config["num_hidden_layers"])
    dspark_layers = len(config["dspark_target_layer_ids"])
    routed_experts = int(config["n_routed_experts"])
    top_k = int(config["num_experts_per_tok"])
    routed_scaling_factor = float(config["routed_scaling_factor"])
    if (
        hidden_size <= 0
        or hidden_layers <= 0
        or dspark_layers <= 0
        or routed_experts <= 0
        or not 0 < top_k <= routed_experts
        or not math.isfinite(routed_scaling_factor)
        or routed_scaling_factor <= 0.0
    ):
        raise ValueError("checkpoint has invalid Flash calibration geometry")
    return (
        hidden_size,
        hidden_layers,
        dspark_layers,
        routed_experts,
        top_k,
        routed_scaling_factor,
    )


def request_completion(
    *, url: str, model: str, prompt: str, max_tokens: int, timeout: float
) -> dict[str, Any]:
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "enable_thinking": False,
        },
        ensure_ascii=False,
    ).encode()
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def move_capture_files(
    *,
    source: Path,
    destination: Path,
    hidden_size: int,
    routed_experts: int | None = None,
    expected_top_k: int | None = None,
    expected_gate_sum: float | None = None,
    require_routes: bool = False,
    record_root: Path | None = None,
) -> list[dict[str, Any]]:
    destination.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    for path in sorted(source.iterdir()):
        routed_match = ROUTED_CAPTURE_RE.fullmatch(path.name)
        match = routed_match or LEGACY_CAPTURE_RE.fullmatch(path.name)
        if match is None or not path.is_file():
            continue
        layer_id = int(match.group("layer"))
        rows = int(match.group("rows"))
        expected_bytes = rows * hidden_size * 2
        payload = path.read_bytes()
        if len(payload) != expected_bytes:
            raise ValueError(
                f"activation capture {path} has {len(payload)} bytes; "
                f"expected {expected_bytes} for {rows}x{hidden_size} BF16"
            )
        route_source: Path | None = None
        route_payload: bytes | None = None
        position_source: Path | None = None
        position_payload: bytes | None = None
        routes_per_row: int | None = None
        if routed_match is not None:
            base = path.name.removesuffix(".bf16")
            route_source = source / f"{base}_routes_u16_f32.bin"
            if not route_source.is_file():
                raise ValueError(
                    f"routed activation capture {path.name} has no route sidecar"
                )
            route_payload = route_source.read_bytes()
            denominator = rows * ROUTE_RECORD.size
            if denominator <= 0 or len(route_payload) % denominator:
                raise ValueError(
                    f"route capture {route_source} has {len(route_payload)} bytes; "
                    f"expected a whole number of {rows}-row u16/f32 route planes"
                )
            routes_per_row = len(route_payload) // denominator
            if routes_per_row <= 0 or (
                expected_top_k is not None and routes_per_row != expected_top_k
            ):
                raise ValueError(
                    f"route capture {route_source} has top-k {routes_per_row}; "
                    f"expected {expected_top_k}"
                )
            decoded = ROUTE_RECORD.iter_unpack(route_payload)
            for row_index in range(rows):
                seen_experts: set[int] = set()
                gate_sum = 0.0
                for _ in range(routes_per_row):
                    expert_id, gate_weight = next(decoded)
                    if routed_experts is not None and expert_id >= routed_experts:
                        raise ValueError(
                            f"route capture {route_source} row {row_index} contains "
                            f"expert {expert_id} outside 0..{routed_experts - 1}"
                        )
                    if expert_id in seen_experts:
                        raise ValueError(
                            f"route capture {route_source} row {row_index} repeats "
                            f"expert {expert_id}"
                        )
                    if not math.isfinite(gate_weight) or gate_weight < 0.0:
                        raise ValueError(
                            f"route capture {route_source} row {row_index} has invalid "
                            f"gate weight {gate_weight}"
                        )
                    seen_experts.add(expert_id)
                    gate_sum += gate_weight
                if expected_gate_sum is None:
                    valid_gate_sum = math.isfinite(gate_sum) and gate_sum > 0.0
                else:
                    valid_gate_sum = (
                        math.isfinite(gate_sum)
                        and abs(gate_sum - expected_gate_sum) <= 1.0e-3
                    )
                if not valid_gate_sum:
                    raise ValueError(
                        f"route capture {route_source} row {row_index} gate weights "
                        f"sum to {gate_sum}, expected {expected_gate_sum}"
                    )
            candidate_position_source = source / f"{base}_positions_u64.bin"
            if candidate_position_source.exists():
                if not candidate_position_source.is_file():
                    raise ValueError(
                        f"route position capture is not a file: {candidate_position_source}"
                    )
                position_source = candidate_position_source
                position_payload = position_source.read_bytes()
                expected_position_bytes = rows * POSITION_RECORD.size
                if len(position_payload) != expected_position_bytes:
                    raise ValueError(
                        f"route position capture {position_source} has "
                        f"{len(position_payload)} bytes; expected "
                        f"{expected_position_bytes} for {rows} rows"
                    )
        elif require_routes:
            raise ValueError(
                f"legacy activation capture {path.name} has no exact per-row routes"
            )

        target = destination / path.name
        route_target = destination / route_source.name if route_source is not None else None
        position_target = (
            destination / position_source.name
            if position_source is not None
            else None
        )
        if position_source is not None and position_target is not None:
            os.replace(position_source, position_target)
        if route_source is not None and route_target is not None:
            os.replace(route_source, route_target)
        os.replace(path, target)
        record_path = target if record_root is None else target.relative_to(record_root)
        record = {
            "path": str(record_path),
            "layer_id": layer_id,
            "rows": rows,
            "hidden_size": hidden_size,
            "bytes": len(payload),
            "sha256": sha256_bytes(payload),
        }
        if routed_match is not None:
            assert route_target is not None
            assert route_payload is not None
            assert routes_per_row is not None
            route_record_path = (
                route_target
                if record_root is None
                else route_target.relative_to(record_root)
            )
            record.update(
                {
                    "capture_id": routed_match.group("capture"),
                    "route_path": str(route_record_path),
                    "route_bytes": len(route_payload),
                    "route_sha256": sha256_bytes(route_payload),
                    "route_record_format": ROUTE_RECORD_FORMAT,
                    "routes_per_row": routes_per_row,
                }
            )
            if position_target is not None and position_payload is not None:
                position_record_path = (
                    position_target
                    if record_root is None
                    else position_target.relative_to(record_root)
                )
                record.update(
                    {
                        "position_path": str(position_record_path),
                        "position_bytes": len(position_payload),
                        "position_sha256": sha256_bytes(position_payload),
                        "position_record_format": POSITION_RECORD_FORMAT,
                    }
                )
        records.append(record)
    return records


def write_bytes_fsync(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def write_retention_control(
    *,
    path: Path,
    layer_count: int,
    routed_experts: int,
    target: int,
    route_counts: list[list[int]],
) -> None:
    if (
        not 0 < layer_count <= 0xFFFF_FFFF
        or not 0 < routed_experts <= 0xFFFF_FFFF
        or not 0 < target <= 0xFFFF_FFFF_FFFF_FFFF
        or len(route_counts) != layer_count
        or any(len(layer) != routed_experts for layer in route_counts)
    ):
        raise ValueError("invalid routed activation retention control geometry")
    payload = bytearray(RETENTION_CONTROL_MAGIC)
    payload.extend(struct.pack("<IIQ", layer_count, routed_experts, target))
    for layer in route_counts:
        for count in layer:
            if isinstance(count, bool) or not 0 <= count <= 0xFFFF_FFFF_FFFF_FFFF:
                raise ValueError("invalid routed activation retention count")
            payload.extend(struct.pack("<Q", count))
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


def compact_routed_capture_files(
    *,
    source: Path,
    destination: Path,
    hidden_size: int,
    layer_count: int,
    routed_experts: int,
    expected_top_k: int,
    expected_gate_sum: float,
    retained_routes_per_expert: int,
    retained_route_counts: list[list[int]],
    record_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Retain quota-useful rows while counting every naturally observed route."""
    if retained_routes_per_expert <= 0:
        raise ValueError("retained routes per expert must be positive")
    if len(retained_route_counts) != layer_count or any(
        len(layer) != routed_experts for layer in retained_route_counts
    ):
        raise ValueError("retained route counts do not match checkpoint geometry")

    destination.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    observed_rows = [0] * layer_count
    observed_counts = [[0] * routed_experts for _ in range(layer_count)]
    matched_capture = False
    observed_sidecars: dict[tuple[str, int], tuple[int, Path]] = {}
    for path in sorted(source.iterdir()):
        match = OBSERVED_ROUTE_COUNTS_RE.fullmatch(path.name)
        if match is None or not path.is_file():
            continue
        matched_capture = True
        capture_id = match.group("capture")
        layer_id = int(match.group("layer"))
        rows = int(match.group("rows"))
        key = (capture_id, layer_id)
        if key in observed_sidecars or not 0 <= layer_id < layer_count or rows <= 0:
            raise ValueError(f"invalid observed route-count capture {path}")
        payload = path.read_bytes()
        expected_bytes = routed_experts * struct.calcsize("<Q")
        if len(payload) != expected_bytes:
            raise ValueError(
                f"observed route-count capture {path} has {len(payload)} bytes; "
                f"expected {expected_bytes}"
            )
        counts = [value[0] for value in struct.iter_unpack("<Q", payload)]
        if sum(counts) != rows * expected_top_k:
            raise ValueError(
                f"observed route-count capture {path} has {sum(counts)} routes; "
                f"expected {rows * expected_top_k}"
            )
        observed_rows[layer_id] += rows
        for expert_id, count in enumerate(counts):
            observed_counts[layer_id][expert_id] += count
        observed_sidecars[key] = (rows, path)

    for path in sorted(source.iterdir()):
        match = ROUTED_CAPTURE_RE.fullmatch(path.name)
        if match is None or not path.is_file():
            continue
        matched_capture = True
        layer_id = int(match.group("layer"))
        rows = int(match.group("rows"))
        observed_sidecar = observed_sidecars.get((match.group("capture"), layer_id))
        if not 0 <= layer_id < layer_count:
            raise ValueError(
                f"activation capture {path} has layer {layer_id} outside "
                f"0..{layer_count - 1}"
            )
        expected_bytes = rows * hidden_size * 2
        payload = path.read_bytes()
        if len(payload) != expected_bytes:
            raise ValueError(
                f"activation capture {path} has {len(payload)} bytes; "
                f"expected {expected_bytes} for {rows}x{hidden_size} BF16"
            )
        route_source = source / (
            path.name.removesuffix(".bf16") + "_routes_u16_f32.bin"
        )
        if not route_source.is_file():
            raise ValueError(
                f"routed activation capture {path.name} has no route sidecar"
            )
        route_payload = route_source.read_bytes()
        expected_route_bytes = rows * expected_top_k * ROUTE_RECORD.size
        if len(route_payload) != expected_route_bytes:
            raise ValueError(
                f"route capture {route_source} has {len(route_payload)} bytes; "
                f"expected {expected_route_bytes} for {rows}x{expected_top_k} routes"
            )
        position_source = source / (
            path.name.removesuffix(".bf16") + "_positions_u64.bin"
        )
        position_payload: bytes | None = None
        if position_source.exists():
            if not position_source.is_file():
                raise ValueError(
                    f"route position capture is not a file: {position_source}"
                )
            position_payload = position_source.read_bytes()
            expected_position_bytes = rows * POSITION_RECORD.size
            if len(position_payload) != expected_position_bytes:
                raise ValueError(
                    f"route position capture {position_source} has "
                    f"{len(position_payload)} bytes; expected "
                    f"{expected_position_bytes} for {rows} rows"
                )

        selected_rows: list[int] = []
        decoded_rows: list[list[tuple[int, float]]] = []
        decoded = ROUTE_RECORD.iter_unpack(route_payload)
        for row_index in range(rows):
            row_routes: list[tuple[int, float]] = []
            seen_experts: set[int] = set()
            gate_sum = 0.0
            for _ in range(expected_top_k):
                expert_id, gate_weight = next(decoded)
                if expert_id >= routed_experts:
                    raise ValueError(
                        f"route capture {route_source} row {row_index} contains "
                        f"expert {expert_id} outside 0..{routed_experts - 1}"
                    )
                if expert_id in seen_experts:
                    raise ValueError(
                        f"route capture {route_source} row {row_index} repeats "
                        f"expert {expert_id}"
                    )
                if not math.isfinite(gate_weight) or gate_weight < 0.0:
                    raise ValueError(
                        f"route capture {route_source} row {row_index} has invalid "
                        f"gate weight {gate_weight}"
                    )
                seen_experts.add(expert_id)
                gate_sum += gate_weight
                row_routes.append((expert_id, gate_weight))
                if observed_sidecar is None:
                    observed_counts[layer_id][expert_id] += 1
            if (
                not math.isfinite(gate_sum)
                or abs(gate_sum - expected_gate_sum) > 1.0e-3
            ):
                raise ValueError(
                    f"route capture {route_source} row {row_index} gate weights "
                    f"sum to {gate_sum}, expected {expected_gate_sum}"
                )
            decoded_rows.append(row_routes)
            if any(
                retained_route_counts[layer_id][expert_id]
                < retained_routes_per_expert
                for expert_id, _ in row_routes
            ):
                selected_rows.append(row_index)
                for expert_id, _ in row_routes:
                    retained_route_counts[layer_id][expert_id] += 1
        if observed_sidecar is None:
            observed_rows[layer_id] += rows

        if selected_rows:
            row_bytes = hidden_size * 2
            retained_payload = b"".join(
                payload[row_index * row_bytes : (row_index + 1) * row_bytes]
                for row_index in selected_rows
            )
            retained_route_payload = b"".join(
                ROUTE_RECORD.pack(expert_id, gate_weight)
                for row_index in selected_rows
                for expert_id, gate_weight in decoded_rows[row_index]
            )
            retained_position_payload = (
                None
                if position_payload is None
                else b"".join(
                    position_payload[
                        row_index
                        * POSITION_RECORD.size : (row_index + 1)
                        * POSITION_RECORD.size
                    ]
                    for row_index in selected_rows
                )
            )
            retained_base = (
                f"capture_{match.group('capture')}_layer_{layer_id:02d}_"
                f"rows_{len(selected_rows)}_expert_input"
            )
            target = destination / f"{retained_base}.bf16"
            route_target = destination / f"{retained_base}_routes_u16_f32.bin"
            position_target = destination / f"{retained_base}_positions_u64.bin"
            # The activation file is the pair's commit marker, matching the daemon.
            if retained_position_payload is not None:
                write_bytes_fsync(position_target, retained_position_payload)
            write_bytes_fsync(route_target, retained_route_payload)
            write_bytes_fsync(target, retained_payload)
            record = {
                "path": str(target.relative_to(record_root)),
                "layer_id": layer_id,
                "rows": len(selected_rows),
                "hidden_size": hidden_size,
                "bytes": len(retained_payload),
                "sha256": sha256_bytes(retained_payload),
                "capture_id": match.group("capture"),
                "route_path": str(route_target.relative_to(record_root)),
                "route_bytes": len(retained_route_payload),
                "route_sha256": sha256_bytes(retained_route_payload),
                "route_record_format": ROUTE_RECORD_FORMAT,
                "routes_per_row": expected_top_k,
                "source_rows": (
                    rows if observed_sidecar is None else observed_sidecar[0]
                ),
            }
            if retained_position_payload is not None:
                record.update(
                    {
                        "position_path": str(position_target.relative_to(record_root)),
                        "position_bytes": len(retained_position_payload),
                        "position_sha256": sha256_bytes(retained_position_payload),
                        "position_record_format": POSITION_RECORD_FORMAT,
                    }
                )
            records.append(record)

        if position_payload is not None:
            position_source.unlink()
        route_source.unlink()
        path.unlink()

    for _, observed_path in observed_sidecars.values():
        observed_path.unlink()

    if not matched_capture:
        raise ValueError("request produced no routed activation captures")
    observed_layers = [
        {
            "layer_id": layer_id,
            "rows": rows,
            "routes": rows * expected_top_k,
            "expert_route_counts": observed_counts[layer_id],
        }
        for layer_id, rows in enumerate(observed_rows)
        if rows
    ]
    return records, observed_layers


def route_distribution(
    *,
    root: Path,
    prompt_records: list[dict[str, Any]],
    layer_count: int,
    routed_experts: int,
    top_k: int,
) -> list[dict[str, Any]]:
    counts = [[0] * routed_experts for _ in range(layer_count)]
    rows_by_layer = [0] * layer_count
    for prompt in prompt_records:
        for capture in prompt["capture_files"]:
            layer_id = int(capture["layer_id"])
            rows = int(capture["rows"])
            route_path = root / capture["route_path"]
            payload = route_path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != capture["route_sha256"]:
                raise ValueError(f"route capture SHA-256 changed: {route_path}")
            for expert_id, _ in ROUTE_RECORD.iter_unpack(payload):
                counts[layer_id][expert_id] += 1
            rows_by_layer[layer_id] += rows
    distribution = []
    for layer_id, layer_counts in enumerate(counts):
        ordered = sorted(layer_counts)
        routes = sum(layer_counts)
        expected_routes = rows_by_layer[layer_id] * top_k
        if routes != expected_routes:
            raise ValueError(
                f"layer {layer_id} route count {routes} does not match "
                f"{rows_by_layer[layer_id]} rows * top-k {top_k}"
            )
        distribution.append(
            {
                "layer_id": layer_id,
                "rows": rows_by_layer[layer_id],
                "routes": routes,
                "zero_hit_experts": sum(count == 0 for count in layer_counts),
                "min_hits": ordered[0],
                "p50_hits": ordered[len(ordered) // 2],
                "max_hits": ordered[-1],
                "expert_route_counts": layer_counts,
            }
        )
    return distribution


def summarize_route_counts(
    *,
    rows_by_layer: list[int],
    counts: list[list[int]],
    layer_count: int,
    routed_experts: int,
    top_k: int,
) -> list[dict[str, Any]]:
    if len(rows_by_layer) != layer_count or len(counts) != layer_count:
        raise ValueError("observed route coverage does not match layer count")
    distribution = []
    for layer_id in range(layer_count):
        layer_counts = counts[layer_id]
        if len(layer_counts) != routed_experts or any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in layer_counts
        ):
            raise ValueError(
                f"observed route counts for layer {layer_id} are invalid"
            )
        rows = rows_by_layer[layer_id]
        routes = sum(layer_counts)
        if routes != rows * top_k:
            raise ValueError(
                f"layer {layer_id} observed route count {routes} does not match "
                f"{rows} rows * top-k {top_k}"
            )
        ordered = sorted(layer_counts)
        distribution.append(
            {
                "layer_id": layer_id,
                "rows": rows,
                "routes": routes,
                "zero_hit_experts": sum(count == 0 for count in layer_counts),
                "min_hits": ordered[0],
                "p50_hits": ordered[len(ordered) // 2],
                "max_hits": ordered[-1],
                "expert_route_counts": layer_counts,
            }
        )
    return distribution


def add_observed_layers(
    *,
    observed_layers: list[dict[str, Any]],
    rows_by_layer: list[int],
    route_counts: list[list[int]],
    layer_count: int,
    routed_experts: int,
    top_k: int,
) -> None:
    seen_layers: set[int] = set()
    for layer in observed_layers:
        layer_id = int(layer["layer_id"])
        rows = int(layer["rows"])
        counts = layer["expert_route_counts"]
        if (
            layer_id in seen_layers
            or not 0 <= layer_id < layer_count
            or rows <= 0
            or not isinstance(counts, list)
            or len(counts) != routed_experts
            or sum(counts) != rows * top_k
        ):
            raise ValueError("request has invalid observed route coverage")
        seen_layers.add(layer_id)
        rows_by_layer[layer_id] += rows
        for expert_id, count in enumerate(counts):
            route_counts[layer_id][expert_id] += int(count)


def quarantine_path(path: Path, *, output: Path, label: str) -> None:
    if not path.exists():
        return
    quarantine_root = output / "uncommitted"
    quarantine_root.mkdir(parents=True, exist_ok=True)
    attempt = 0
    target = quarantine_root / f"{label}-{attempt}"
    while target.exists():
        attempt += 1
        target = quarantine_root / f"{label}-{attempt}"
    os.replace(path, target)


def is_live_capture_artifact(path: Path) -> bool:
    if not path.is_file():
        return False
    if (
        ROUTED_CAPTURE_RE.fullmatch(path.name)
        or LEGACY_CAPTURE_RE.fullmatch(path.name)
        or OBSERVED_ROUTE_COUNTS_RE.fullmatch(path.name)
    ):
        return True
    suffix = "_routes_u16_f32.bin"
    if path.name.endswith(suffix) and ROUTED_CAPTURE_RE.fullmatch(
        path.name.removesuffix(suffix) + ".bf16"
    ):
        return True
    suffix = "_positions_u64.bin"
    return path.name.endswith(suffix) and bool(
        ROUTED_CAPTURE_RE.fullmatch(path.name.removesuffix(suffix) + ".bf16")
    )


def quarantine_live_capture_artifacts(
    *, source: Path, output: Path, label: str
) -> None:
    paths = [path for path in source.iterdir() if is_live_capture_artifact(path)]
    if not paths:
        return
    quarantine_root = output / "uncommitted"
    quarantine_root.mkdir(parents=True, exist_ok=True)
    attempt = 0
    destination = quarantine_root / f"{label}-{attempt}"
    while destination.exists():
        attempt += 1
        destination = quarantine_root / f"{label}-{attempt}"
    destination.mkdir()
    for path in paths:
        os.replace(path, destination / path.name)


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


def collect(args: argparse.Namespace) -> dict[str, Any]:
    corpus_path = args.corpus.expanduser().resolve(strict=True)
    dump_dir = args.dump_dir.expanduser().resolve(strict=True)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    resume = bool(getattr(args, "resume", False))
    allow_corpus_extension = bool(
        getattr(args, "allow_corpus_extension", False)
    )
    if allow_corpus_extension and not resume:
        raise ValueError("corpus extension is valid only with --resume")
    retained_routes_per_expert = int(
        getattr(args, "retained_routes_per_expert", 1024)
    )
    minimum_natural_routes_per_expert = int(
        getattr(args, "minimum_natural_routes_per_expert", 0)
    )
    if retained_routes_per_expert <= 0:
        raise ValueError("retained routes per expert must be positive")
    if minimum_natural_routes_per_expert < 0:
        raise ValueError("minimum natural routes per expert must be nonnegative")
    if output.exists():
        if not output.is_dir():
            raise ValueError(f"capture output is not a directory: {output}")
        if any(output.iterdir()) and not resume:
            raise ValueError(f"capture output is not empty: {output}")
    elif resume:
        raise ValueError(f"cannot resume missing capture output: {output}")
    if output == dump_dir or dump_dir in output.parents:
        raise ValueError("capture output must not be inside the live dump directory")
    output.mkdir(parents=True, exist_ok=True)

    corpus, corpus_sha256 = load_corpus(corpus_path)
    (
        hidden_size,
        layer_count,
        dspark_layer_count,
        routed_experts,
        top_k,
        routed_scaling_factor,
    ) = checkpoint_shape(checkpoint)
    progress_path = output / "progress.json"
    if resume:
        if (output / "manifest.json").exists():
            raise ValueError(f"capture output is already complete: {output}")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        prior_corpus_sha256 = progress.get("corpus_sha256")
        if (
            not isinstance(prior_corpus_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", prior_corpus_sha256)
        ):
            raise ValueError("capture progress has an invalid corpus SHA-256")
        if prior_corpus_sha256 != corpus_sha256 and not allow_corpus_extension:
            raise ValueError(
                "capture progress corpus SHA-256 does not match; use "
                "--allow-corpus-extension only for a verified appended corpus"
            )
        expected_identity = {
            "schema": PROGRESS_SCHEMA,
            "checkpoint": str(checkpoint),
            "hidden_size": hidden_size,
            "layer_count": layer_count,
            "dspark_layer_count": dspark_layer_count,
            "routed_experts": routed_experts,
            "top_k": top_k,
            "routed_scaling_factor": routed_scaling_factor,
            "retained_routes_per_expert": retained_routes_per_expert,
            "minimum_natural_routes_per_expert": (
                minimum_natural_routes_per_expert
            ),
        }
        for key, expected in expected_identity.items():
            if progress.get(key) != expected:
                raise ValueError(
                    f"capture progress {key} {progress.get(key)!r} does not match {expected!r}"
                )
        startup_files = progress.get("startup_files")
        prompt_records = progress.get("prompts")
        observed_rows_by_layer = progress.get("observed_rows_by_layer")
        observed_route_counts = progress.get("observed_route_counts")
        retained_route_counts = progress.get("retained_route_counts")
        if not isinstance(startup_files, list) or not isinstance(prompt_records, list):
            raise ValueError("capture progress has invalid startup/prompt records")
        if (
            not isinstance(observed_rows_by_layer, list)
            or len(observed_rows_by_layer) != layer_count
            or not isinstance(observed_route_counts, list)
            or len(observed_route_counts) != layer_count
            or not isinstance(retained_route_counts, list)
            or len(retained_route_counts) != layer_count
            or any(
                not isinstance(layer, list) or len(layer) != routed_experts
                for matrix in (observed_route_counts, retained_route_counts)
                for layer in matrix
            )
        ):
            raise ValueError("capture progress has invalid route coverage state")
        if len(prompt_records) > len(corpus) or any(
            not isinstance(record, dict)
            or record.get("index") != index
            or record.get("id") != corpus[index]["id"]
            or record.get("prompt_sha256")
            != sha256_bytes(corpus[index]["prompt"].encode())
            or record.get("max_tokens") != corpus[index]["max_tokens"]
            for index, record in enumerate(prompt_records)
        ):
            raise ValueError("capture progress is not a valid corpus prefix")
        raw_corpus_history = progress.get(
            "corpus_history_sha256", [prior_corpus_sha256]
        )
        if (
            not isinstance(raw_corpus_history, list)
            or not raw_corpus_history
            or any(
                not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
                for digest in raw_corpus_history
            )
            or raw_corpus_history[-1] != prior_corpus_sha256
        ):
            raise ValueError("capture progress has invalid corpus history")
        corpus_history_sha256 = list(raw_corpus_history)
        if corpus_history_sha256[-1] != corpus_sha256:
            corpus_history_sha256.append(corpus_sha256)
        quarantine_live_capture_artifacts(
            source=dump_dir,
            output=output,
            label=f"prompt-{len(prompt_records):03d}-live",
        )
    else:
        # Startup prewarm exercises the same capture boundary but is not part
        # of the calibration corpus. Quarantine the complete joint file set,
        # including observed-count sidecars, before publishing quota state so
        # prompt zero cannot inherit startup routing coverage.
        quarantine_live_capture_artifacts(
            source=dump_dir,
            output=output,
            label="startup-live",
        )
        startup_files = []
        corpus_history_sha256 = [corpus_sha256]
        prompt_records = []
        observed_rows_by_layer = [0] * layer_count
        observed_route_counts = [[0] * routed_experts for _ in range(layer_count)]
        retained_route_counts = [[0] * routed_experts for _ in range(layer_count)]

    def checkpoint_progress() -> None:
        write_json_atomic(
            progress_path,
            {
                "schema": PROGRESS_SCHEMA,
                "corpus_sha256": corpus_sha256,
                "corpus_history_sha256": corpus_history_sha256,
                "corpus_records": len(corpus),
                "checkpoint": str(checkpoint),
                "hidden_size": hidden_size,
                "layer_count": layer_count,
                "dspark_layer_count": dspark_layer_count,
                "routed_experts": routed_experts,
                "top_k": top_k,
                "routed_scaling_factor": routed_scaling_factor,
                "retained_routes_per_expert": retained_routes_per_expert,
                "minimum_natural_routes_per_expert": (
                    minimum_natural_routes_per_expert
                ),
                "startup_files": startup_files,
                "prompts": prompt_records,
                "observed_rows_by_layer": observed_rows_by_layer,
                "observed_route_counts": observed_route_counts,
                "retained_route_counts": retained_route_counts,
            },
        )
        # Publish this only after progress is durable. A stale control can retain
        # extra rows on retry; a control ahead of progress could discard needed rows.
        write_retention_control(
            path=dump_dir / RETENTION_CONTROL_FILE,
            layer_count=layer_count,
            routed_experts=routed_experts,
            target=retained_routes_per_expert,
            route_counts=retained_route_counts,
        )

    checkpoint_progress()
    for index in range(len(prompt_records), len(corpus)):
        item = corpus[index]
        capture_destination = output / "captures" / f"{index:03d}-{item['id']}"
        quarantine_path(
            capture_destination,
            output=output,
            label=f"prompt-{index:03d}-partial-retention",
        )
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
        capture_files, observed_layers = compact_routed_capture_files(
            source=dump_dir,
            destination=capture_destination,
            hidden_size=hidden_size,
            layer_count=layer_count,
            routed_experts=routed_experts,
            expected_top_k=top_k,
            expected_gate_sum=routed_scaling_factor,
            retained_routes_per_expert=retained_routes_per_expert,
            retained_route_counts=retained_route_counts,
            record_root=output,
        )
        target_layers = {record["layer_id"] for record in observed_layers}
        if target_layers != set(range(layer_count)):
            missing = sorted(set(range(layer_count)) - target_layers)
            raise ValueError(
                f"prompt {item['id']!r} did not capture every target layer; missing {missing}"
            )
        add_observed_layers(
            observed_layers=observed_layers,
            rows_by_layer=observed_rows_by_layer,
            route_counts=observed_route_counts,
            layer_count=layer_count,
            routed_experts=routed_experts,
            top_k=top_k,
        )
        content = result["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise ValueError(f"prompt {item['id']!r} returned non-text content")
        prompt_record = {
            "index": index,
            "id": item["id"],
            "prompt_sha256": sha256_bytes(item["prompt"].encode()),
            "max_tokens": item["max_tokens"],
            "response_sha256": sha256_bytes(content.encode()),
            "completion_tokens": result.get("usage", {}).get("completion_tokens"),
            "mtp_verify_cycles": runtime["mtp_verify_cycles"],
            "capture_files": capture_files,
            "observed_layers": [
                {
                    "layer_id": layer["layer_id"],
                    "rows": layer["rows"],
                    "routes": layer["routes"],
                }
                for layer in observed_layers
            ],
        }
        prompt_records.append(prompt_record)
        checkpoint_progress()
        print(
            json.dumps(
                {
                    "event": "calibration-prompt-captured",
                    "id": item["id"],
                    "files": len(capture_files),
                    "retained_rows": sum(
                        record["rows"] for record in capture_files
                    ),
                    "observed_rows": sum(
                        layer["rows"] for layer in observed_layers
                    ),
                    "mtp_verify_cycles": runtime["mtp_verify_cycles"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    covered_layers = [
        layer_id
        for layer_id, rows in enumerate(observed_rows_by_layer)
        if rows > 0
    ]
    if covered_layers != list(range(layer_count)):
        missing = sorted(set(range(layer_count)) - set(covered_layers))
        raise ValueError(f"calibration corpus did not cover every layer; missing {missing}")
    mtp_verify_cycles = sum(
        int(prompt["mtp_verify_cycles"]) for prompt in prompt_records
    )
    if mtp_verify_cycles < 1:
        raise ValueError("calibration corpus did not execute native dSpark verification")
    routes = summarize_route_counts(
        rows_by_layer=observed_rows_by_layer,
        counts=observed_route_counts,
        layer_count=layer_count,
        routed_experts=routed_experts,
        top_k=top_k,
    )
    retained_routes = route_distribution(
        root=output,
        prompt_records=prompt_records,
        layer_count=layer_count,
        routed_experts=routed_experts,
        top_k=top_k,
    )
    actual_retained_counts = [
        layer["expert_route_counts"] for layer in retained_routes
    ]
    if actual_retained_counts != retained_route_counts:
        raise ValueError(
            "retained route sidecars do not match fsynced progress coverage"
        )
    weakest_natural_coverage = min(layer["min_hits"] for layer in routes)
    if weakest_natural_coverage < minimum_natural_routes_per_expert:
        raise ValueError(
            f"natural route coverage floor {weakest_natural_coverage} is below "
            f"required {minimum_natural_routes_per_expert}"
        )
    manifest = {
        "schema": SCHEMA,
        "corpus_path": str(corpus_path),
        "corpus_sha256": corpus_sha256,
        "corpus_history_sha256": corpus_history_sha256,
        "checkpoint": str(checkpoint),
        "model": args.model,
        "prompt_encoding": "deepseek_v4_single_user_nonthinking",
        "hidden_size": hidden_size,
        "layer_count": layer_count,
        "dspark_layer_count": dspark_layer_count,
        "routed_experts": routed_experts,
        "top_k": top_k,
        "routed_scaling_factor": routed_scaling_factor,
        "route_record_format": ROUTE_RECORD_FORMAT,
        "retention": {
            "policy": "first_joint_rows_until_per_layer_expert_route_quota",
            "routes_per_expert_target": retained_routes_per_expert,
            "minimum_natural_routes_per_expert": (
                minimum_natural_routes_per_expert
            ),
        },
        "startup_files": startup_files,
        "prompts": prompt_records,
        "route_distribution": routes,
        "retained_route_distribution": retained_routes,
        "summary": {
            "prompts": len(prompt_records),
            "mtp_verify_cycles": mtp_verify_cycles,
            "capture_files": sum(
                len(prompt["capture_files"]) for prompt in prompt_records
            ),
            "captured_rows": sum(
                record["rows"]
                for prompt in prompt_records
                for record in prompt["capture_files"]
            ),
            "routed_rows": sum(layer["rows"] for layer in routes),
            "route_records": sum(layer["routes"] for layer in routes),
            "retained_route_records": sum(
                layer["routes"] for layer in retained_routes
            ),
            "weakest_natural_route_coverage": weakest_natural_coverage,
            "covered_layers": covered_layers,
        },
    }
    write_json_atomic(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    try:
        manifest = collect(parse_args())
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(manifest["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
