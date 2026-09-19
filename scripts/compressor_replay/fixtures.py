"""Synthetic ratio-two capture fixtures for CPU development and tests.

The real producer capture arrives later; these fixtures follow the exact
schema-1 manifest layout of ``input_trace.rs`` (48f0c712) so the loader is
exercised against realistic bytes.  All payloads are deterministic
(RandomState seed) and finite (no inf/NaN bit patterns), but are otherwise
arbitrary raw bits — the harness never interprets operand values on CPU.

Geometries:
  * ``pair``  — the expected two-row decode batch: positions 40/41,
    descriptors [sentinel, slot_count + 0] (earlier-wave addressing);
  * ``single`` — a singleton position-41 batch whose predecessor is a pending
    slot (pending addressing; an earlier-wave reference is impossible alone).

Both exercise the nonzero absolute-position offset (first_token 40,
logical_compressed_row 20 for the p41 rows).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from .schema import (
    DESCRIPTOR_SENTINEL,
    LATENT_DIM,
    PENDING_ROW_BYTES,
    ROW_BYTES,
    SOURCE_DIM,
    WEIGHT_NORM_BYTES,
    WEIGHT_WGATE_BYTES,
    WEIGHT_WKV_BYTES,
)

PAIR = "pair"
SINGLE = "single"


def _finite_bf16(rng: np.random.RandomState, count: int) -> bytes:
    """Random BF16 bytes with exponent < 0x1F (finite by construction)."""
    value = rng.randint(0, 0x7F80, size=count).astype(np.uint16)
    sign = (rng.randint(0, 2, size=count) << 15).astype(np.uint16)
    return (value | sign).tobytes()


def _finite_f32(rng: np.random.RandomState, count: int) -> bytes:
    return rng.uniform(-2.0, 2.0, size=count).astype("<f4").tobytes()


def _raw_bytes(rng: np.random.RandomState, count: int) -> bytes:
    return rng.randint(0, 256, size=count).astype(np.uint8).tobytes()


def _u64(values) -> bytes:
    return b"".join(int(v).to_bytes(8, "little") for v in values)


def write_synthetic_capture(
    root: Path,
    name: str,
    *,
    geometry: str = PAIR,
    layer: int = 2,
    slot_count: int = 4,
    seed: int = 41,
    position: int = 40,
    weights_dir: Optional[Path] = None,
    write_weights: bool = True,
    device_matches_host: bool = True,
) -> Path:
    """Write one synthetic capture directory (plus shared weights on first
    use) and return the capture directory path."""
    if geometry not in (PAIR, SINGLE):
        raise ValueError(f"unknown geometry {geometry!r}")
    if not (1 <= slot_count <= 16):
        raise ValueError("slot_count outside 1..16")
    if geometry == PAIR:
        rows = 2
        positions = [position, position + 1]
        descriptors = [DESCRIPTOR_SENTINEL, slot_count]  # row1 pools row0
        chunks = [
            {"index": 0, "request_id": 7000, "lease": {"slot": 0, "generation": 3},
             "version": 9, "position": positions[0], "tokens": 1, "prepared_offset": 0},
            {"index": 1, "request_id": 7001, "lease": {"slot": 1, "generation": 4},
             "version": 9, "position": positions[1], "tokens": 1, "prepared_offset": 1},
        ]
    else:
        rows = 1
        positions = [position + 1]
        descriptors = [min(2, slot_count - 1)]  # pending-slot addressing
        chunks = [
            {"index": 0, "request_id": 7101, "lease": {"slot": descriptors[0], "generation": 5},
             "version": 10, "position": positions[0], "tokens": 1, "prepared_offset": 0},
        ]

    rng = np.random.RandomState(seed)
    directory = Path(root) / name
    directory.mkdir(parents=True, exist_ok=True)

    buffers = {}
    payloads = {}

    def put(buf_name: str, dtype: str, shape, blob: bytes):
        file = f"layer{layer}-compressor-{buf_name}.bin"
        (directory / file).write_bytes(blob)
        buffers[buf_name] = {"name": buf_name, "file": file, "dtype": dtype,
                             "shape": list(shape), "bytes": len(blob)}

    put("input", "bfloat16", (rows, SOURCE_DIM),
        _finite_bf16(rng, rows * SOURCE_DIM))
    put("projected", "float32", (rows, LATENT_DIM),
        _finite_f32(rng, rows * LATENT_DIM))
    put("scores", "float32", (rows, LATENT_DIM),
        _finite_f32(rng, rows * LATENT_DIM))
    put("output", "bfloat16", (rows, LATENT_DIM),
        _finite_bf16(rng, rows * LATENT_DIM))
    put("frequencies", "float32", (rows, 32, 2),
        _finite_f32(rng, rows * 64))
    put("positions", "uint64", (rows,), _u64(positions))
    put("descriptors-device", "uint64", (rows,), _u64(descriptors))
    put("kv-values", "fp4e2m1", (rows, 512), _raw_bytes(rng, rows * 256))
    put("kv-scales", "fp8e4m3", (rows, 32), _raw_bytes(rng, rows * 32))
    put("pending-kv", "float32", (slot_count, LATENT_DIM),
        _finite_f32(rng, slot_count * LATENT_DIM))
    put("pending-scores", "float32", (slot_count, LATENT_DIM),
        _finite_f32(rng, slot_count * LATENT_DIM))

    host_descriptors = list(descriptors)
    if not device_matches_host:
        host_descriptors = [d ^ 0x1 for d in host_descriptors]
    (directory / f"layer{layer}-compressor-descriptors-host.bin").write_bytes(
        _u64(host_descriptors)
    )

    wave_rows = []
    for row in range(rows):
        pos = positions[row]
        first_token = pos - (pos % 2)
        predecessor = (
            {"kind": "invalid_sentinel"}
            if descriptors[row] == DESCRIPTOR_SENTINEL
            else (
                {"kind": "pending_slot", "slot": descriptors[row]}
                if descriptors[row] < slot_count
                else {"kind": "earlier_wave_row", "wave_row": descriptors[row] - slot_count}
            )
        )
        wave_rows.append({
            "row": row,
            "chunk": chunks[row]["index"],
            "request_id": chunks[row]["request_id"],
            "absolute_position": pos,
            "device_descriptor": descriptors[row],
            "predecessor": predecessor,
            "completed_latent": {
                "source_row": row,
                "first_token": first_token,
                "logical_compressed_row": first_token // 2,
                "request_id": chunks[row]["request_id"],
            },
        })

    manifest = {
        "schema": 1,
        "kind": "compressor-inputs",
        "layer": layer,
        "ratio": 2,
        "rows": rows,
        "slot_count": slot_count,
        "proposal_snapshot": 123456,
        "chunks": [
            {**chunk, "flattened_row_range": [chunk["prepared_offset"],
                                              chunk["prepared_offset"] + chunk["tokens"]],
             "absolute_positions": list(range(chunk["position"],
                                              chunk["position"] + chunk["tokens"]))}
            for chunk in chunks
        ],
        "buffers": {k: buffers[k] for k in (
            "input", "projected", "scores", "output", "frequencies",
            "positions", "kv-values", "kv-scales")},
        "descriptors": {
            "device_file": f"layer{layer}-compressor-descriptors-device.bin",
            "host_file": f"layer{layer}-compressor-descriptors-host.bin",
            "dtype": "uint64",
            "rows": rows,
            "device_matches_host": device_matches_host,
            "note": "synthetic fixture; device bytes authoritative",
        },
        "pending": {
            "kv": buffers["pending-kv"],
            "scores": buffers["pending-scores"],
            "dtype": "float32",
            "shape": [slot_count, LATENT_DIM],
            "slots": [
                {"slot": s,
                 "state": "active" if any(c["lease"]["slot"] == s for c in chunks) else "unscored",
                 "request_id": next((c["request_id"] for c in chunks if c["lease"]["slot"] == s), None),
                 "generation": next((c["lease"]["generation"] for c in chunks if c["lease"]["slot"] == s), None)}
                for s in range(slot_count)
            ],
            "note": "synthetic fixture; whole bounded planes",
        },
        "wave_rows": wave_rows,
        "weights": {},
        "budget_bytes": 64 << 20,
        "captured_bytes": sum(b["bytes"] for b in buffers.values()),
        "notes": ["synthetic fixture for CPU harness development"],
    }

    if weights_dir is None:
        weights_dir = Path(root) / "weights"
    weights_dir = Path(weights_dir)
    weights_files = {
        "wkv-weight": ("wkv", WEIGHT_WKV_BYTES),
        "wgate-weight": ("wgate", WEIGHT_WGATE_BYTES),
        "norm-weight": ("norm", WEIGHT_NORM_BYTES),
    }
    if write_weights:
        weights_dir.mkdir(parents=True, exist_ok=True)
        for buf_name, (role, byte_count) in weights_files.items():
            file = f"layer{layer}-compressor-{buf_name}.bin"
            if not (weights_dir / file).exists():
                if buf_name == "norm-weight":
                    blob = _finite_bf16(rng, LATENT_DIM)
                else:
                    blob = _finite_bf16(rng, LATENT_DIM * SOURCE_DIM)
                assert len(blob) == byte_count
                (weights_dir / file).write_bytes(blob)
        manifest["weights"] = {
            "directory": str(weights_dir),
            "tensors": [
                {"tensor": f"layers.{layer}.attn.compressor.wkv.weight",
                 "dtype": "bfloat16", "shape": [LATENT_DIM, SOURCE_DIM],
                 "bytes": WEIGHT_WKV_BYTES,
                 "file": f"layer{layer}-compressor-wkv-weight.bin"},
                {"tensor": f"layers.{layer}.attn.compressor.wgate.weight",
                 "dtype": "bfloat16", "shape": [LATENT_DIM, SOURCE_DIM],
                 "bytes": WEIGHT_WGATE_BYTES,
                 "file": f"layer{layer}-compressor-wgate-weight.bin"},
                {"tensor": f"layers.{layer}.attn.compressor.norm.weight",
                 "dtype": "bfloat16", "shape": [LATENT_DIM],
                 "bytes": WEIGHT_NORM_BYTES,
                 "file": f"layer{layer}-compressor-norm-weight.bin"},
            ],
            "note": "synthetic shared weights (first writer wins)",
        }
    else:
        manifest["weights"] = {
            "directory": str(weights_dir),
            "already_written_for_root": True,
        }

    manifest_path = directory / f"layer{layer}-compressor-inputs.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return directory
