"""Synthetic ratio-two capture fixtures for CPU development and tests.

These fixtures follow the exact schema-1 manifest layout emitted by
``v41_compressor/input_trace.rs`` (48f0c712): public buffer keys
(``output_before_kv_pack``, ``kv_values``, ``kv_scales``), a ``pending``
section, a separate ``descriptors`` section with ``device_file``/``host_file``
and shared weights under ``<root>/projection-weights`` written by exactly one
first-writer manifest.  They are not a fictional layout: the loader is the
same code path that reads real captures.

Geometries:
  * ``single``          -- one row at absolute position 41 whose predecessor is
    a pending slot (pending addressing);
  * ``pair``            -- the real two-request decode batch: two chunks, both
    at position 41, each pooling its own pending slot, second chunk at
    prepared_offset 1 (nonzero offset);
  * ``pair_earlierwave`` -- one chunk of two tokens (positions 40, 41): the
    first row is the sentinel, the second pools earlier wave row 0 in the same
    chunk (earlier-wave addressing), nonzero first token 40.

All payloads are deterministic and finite; the harness never interprets
operand values on CPU.
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
    SOURCE_DIM,
    WEIGHT_NORM_BYTES,
    WEIGHT_WGATE_BYTES,
    WEIGHT_WKV_BYTES,
    WEIGHT_DIRNAME,
)

SINGLE = "single"
PAIR = "pair"
PAIR_EARLIERWAVE = "pair_earlierwave"
GEOMETRIES = (SINGLE, PAIR, PAIR_EARLIERWAVE)

POSITION = 41


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


def _chunk_geometry(geometry: str, slot_count: int):
    """Return ``(positions, descriptors, chunks)`` for one geometry."""
    if geometry == SINGLE:
        chunks = [
            {"index": 0, "request_id": 7101,
             "lease": {"slot": 0, "generation": 5}, "version": 10,
             "position": POSITION, "tokens": 1, "prepared_offset": 0},
        ]
        positions = [POSITION]
        descriptors = [0]                       # pending slot 0
    elif geometry == PAIR:
        chunks = [
            {"index": 0, "request_id": 7000,
             "lease": {"slot": 0, "generation": 3}, "version": 9,
             "position": POSITION, "tokens": 1, "prepared_offset": 0},
            {"index": 1, "request_id": 7001,
             "lease": {"slot": 1, "generation": 4}, "version": 9,
             "position": POSITION, "tokens": 1, "prepared_offset": 1},
        ]
        positions = [POSITION, POSITION]
        descriptors = [0, 1]                    # each its own pending slot
    elif geometry == PAIR_EARLIERWAVE:
        chunks = [
            {"index": 0, "request_id": 7200,
             "lease": {"slot": 2, "generation": 6}, "version": 11,
             "position": POSITION - 1, "tokens": 2, "prepared_offset": 0},
        ]
        positions = [POSITION - 1, POSITION]
        descriptors = [DESCRIPTOR_SENTINEL, slot_count]  # row 1 pools row 0
    else:
        raise ValueError(f"unknown geometry {geometry!r}")
    return positions, descriptors, chunks


def _weight_files(layer: int):
    return {
        "wkv": (f"layer{layer}-compressor-wkv-weight.bin",
                WEIGHT_WKV_BYTES, LATENT_DIM * SOURCE_DIM),
        "wgate": (f"layer{layer}-compressor-wgate-weight.bin",
                  WEIGHT_WGATE_BYTES, LATENT_DIM * SOURCE_DIM),
        "norm": (f"layer{layer}-compressor-norm-weight.bin",
                 WEIGHT_NORM_BYTES, LATENT_DIM),
    }


def write_projection_weights(root: Path, layer: int, rng,
                             weights_dir: Optional[Path] = None) -> Path:
    """Write the shared BF16 weight files once under ``projection-weights``."""
    root = Path(root)
    if weights_dir is None:
        weights_dir = root / WEIGHT_DIRNAME
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    for role, (file, byte_count, elements) in _weight_files(layer).items():
        path = weights_dir / file
        if not path.exists():
            blob = _finite_bf16(rng, elements)
            assert len(blob) == byte_count
            path.write_bytes(blob)
    return weights_dir


def write_synthetic_capture(
    root: Path,
    name: str,
    *,
    geometry: str = PAIR,
    layer: int = 2,
    slot_count: int = 4,
    seed: int = 41,
    weights_writer: bool = True,
    weights_dir: Optional[Path] = None,
    write_weights: bool = True,
    device_matches_host: bool = True,
) -> Path:
    """Write one synthetic capture directory and return its path.

    When ``weights_writer`` is true this manifest carries the full tensor list
    (a first writer) and writes the shared files; otherwise it records
    ``already_written_for_root`` and is bound to a sibling first writer.
    """
    if geometry not in GEOMETRIES:
        raise ValueError(f"unknown geometry {geometry!r}")
    if not (1 <= slot_count <= 16):
        raise ValueError("slot_count outside 1..16")
    root = Path(root)
    positions, descriptors, chunks = _chunk_geometry(geometry, slot_count)
    rows = len(positions)
    rng = np.random.RandomState(seed)
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)

    # ---- buffers (public keys, real internal names) ----------------------
    payloads = {}

    def put(public_key: str, internal: str, dtype: str, shape, blob: bytes):
        file = f"layer{layer}-compressor-{internal}.bin"
        (directory / file).write_bytes(blob)
        payloads[public_key] = {"name": internal, "file": file, "dtype": dtype,
                                "shape": list(shape), "bytes": len(blob)}

    put("input", "input", "bfloat16", (rows, SOURCE_DIM),
        _finite_bf16(rng, rows * SOURCE_DIM))
    put("projected", "projected", "float32", (rows, LATENT_DIM),
        _finite_f32(rng, rows * LATENT_DIM))
    put("scores", "scores", "float32", (rows, LATENT_DIM),
        _finite_f32(rng, rows * LATENT_DIM))
    put("output_before_kv_pack", "output", "bfloat16", (rows, LATENT_DIM),
        _finite_bf16(rng, rows * LATENT_DIM))
    put("frequencies", "frequencies", "float32", (rows, 32, 2),
        _finite_f32(rng, rows * 32 * 2))
    put("positions", "positions", "uint64", (rows,),
        _u64([0 if pos % 2 == 0 else pos - 1 for pos in positions]))
    put("kv_values", "kv-values", "fp4e2m1", (rows, LATENT_DIM),
        _raw_bytes(rng, rows * 256))
    put("kv_scales", "kv-scales", "fp8e4m3", (rows, 32),
        _raw_bytes(rng, rows * 32))

    # ---- pending planes and descriptor files -----------------------------
    pending_kv = _finite_f32(rng, slot_count * LATENT_DIM)
    pending_scores = _finite_f32(rng, slot_count * LATENT_DIM)
    (directory / f"layer{layer}-compressor-pending-kv.bin").write_bytes(pending_kv)
    (directory / f"layer{layer}-compressor-pending-scores.bin"
     ).write_bytes(pending_scores)
    (directory / f"layer{layer}-compressor-descriptors-device.bin").write_bytes(
        _u64(descriptors)
    )
    host_descriptors = list(descriptors)
    if not device_matches_host:
        host_descriptors = [d ^ 0x1 if d != DESCRIPTOR_SENTINEL
                            else DESCRIPTOR_SENTINEL for d in host_descriptors]
    (directory / f"layer{layer}-compressor-descriptors-host.bin").write_bytes(
        _u64(host_descriptors)
    )

    # ---- wave rows (manifest predecessor and completed latent) -----------
    wave_rows = []
    for row in range(rows):
        pos = positions[row]
        descriptor = descriptors[row]
        if descriptor == DESCRIPTOR_SENTINEL:
            predecessor = {"kind": "invalid_sentinel"}
        elif descriptor < slot_count:
            predecessor = {"kind": "pending_slot", "slot": descriptor}
        else:
            predecessor = {"kind": "earlier_wave_row",
                           "wave_row": descriptor - slot_count}
        # The chunk owning this row: chunks are contiguous by prepared_offset.
        chunk = next(c for c in chunks
                     if c["prepared_offset"] <= row
                     < c["prepared_offset"] + c["tokens"])
        if pos % 2 == 0:
            completed = None
        else:
            first_token = pos - (pos % 2)
            completed = {
                "source_row": row,
                "first_token": first_token,
                "logical_compressed_row": first_token // 2,
                "request_id": chunk["request_id"],
            }
        wave_rows.append({
            "row": row,
            "chunk": chunk["index"],
            "request_id": chunk["request_id"],
            "absolute_position": pos,
            "device_descriptor": descriptor,
            "predecessor": predecessor,
            "completed_latent": completed,
        })

    manifest_chunks = [
        {**chunk,
         "flattened_row_range": [chunk["prepared_offset"],
                                 chunk["prepared_offset"] + chunk["tokens"]],
         "absolute_positions": list(range(chunk["position"],
                                          chunk["position"] + chunk["tokens"]))}
        for chunk in chunks
    ]

    manifest = {
        "schema": 1,
        "kind": "compressor-inputs",
        "layer": layer,
        "ratio": 2,
        "rows": rows,
        "slot_count": slot_count,
        "proposal_snapshot": 123456,
        "chunks": manifest_chunks,
        "buffers": payloads,
        "descriptors": {
            "device_file": f"layer{layer}-compressor-descriptors-device.bin",
            "host_file": f"layer{layer}-compressor-descriptors-host.bin",
            "dtype": "uint64",
            "rows": rows,
            "device_matches_host": device_matches_host,
            "note": "synthetic fixture; device bytes authoritative",
        },
        "pending": {
            "kv": {"name": "pending-kv",
                   "file": f"layer{layer}-compressor-pending-kv.bin",
                   "dtype": "float32", "shape": [slot_count, LATENT_DIM],
                   "bytes": len(pending_kv)},
            "scores": {"name": "pending-scores",
                       "file": f"layer{layer}-compressor-pending-scores.bin",
                       "dtype": "float32", "shape": [slot_count, LATENT_DIM],
                       "bytes": len(pending_scores)},
            "dtype": "float32",
            "shape": [slot_count, LATENT_DIM],
            "slots": [
                {"slot": s,
                 "state": "active" if any(c["lease"]["slot"] == s
                                          for c in chunks) else "unscored",
                 "request_id": next((c["request_id"] for c in chunks
                                     if c["lease"]["slot"] == s), None),
                 "generation": next((c["lease"]["generation"] for c in chunks
                                     if c["lease"]["slot"] == s), None)}
                for s in range(slot_count)
            ],
            "note": "synthetic fixture; whole bounded planes",
        },
        "wave_rows": wave_rows,
        "weights": {},
        "budget_bytes": 64 << 20,
        "captured_bytes": (sum(b["bytes"] for b in payloads.values())
                           + len(pending_kv) + len(pending_scores)),
        "notes": ["synthetic fixture for CPU harness development"],
    }

    if write_weights:
        weights_root = write_projection_weights(root, layer, rng, weights_dir)
        files = _weight_files(layer)
        if weights_writer:
            manifest["weights"] = {
                "directory": "/trace/projection-weights",
                "tensors": [
                    {"tensor": f"layers.{layer}.attn.compressor.wkv.weight",
                     "dtype": "bfloat16", "shape": [LATENT_DIM, SOURCE_DIM],
                     "bytes": WEIGHT_WKV_BYTES, "file": files["wkv"][0]},
                    {"tensor": f"layers.{layer}.attn.compressor.wgate.weight",
                     "dtype": "bfloat16", "shape": [LATENT_DIM, SOURCE_DIM],
                     "bytes": WEIGHT_WGATE_BYTES, "file": files["wgate"][0]},
                    {"tensor": f"layers.{layer}.attn.compressor.norm.weight",
                     "dtype": "bfloat16", "shape": [LATENT_DIM],
                     "bytes": WEIGHT_NORM_BYTES, "file": files["norm"][0]},
                ],
                "note": "synthetic first-writer weights (shared files)",
            }
        else:
            manifest["weights"] = {
                "directory": "/trace/projection-weights",
                "already_written_for_root": True,
            }
    else:
        manifest["weights"] = {"already_written_for_root": True}

    manifest_path = directory / f"layer{layer}-compressor-inputs.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return directory
