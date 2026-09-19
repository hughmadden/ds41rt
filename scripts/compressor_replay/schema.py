"""Capture schema for the ratio-two compressor producer input trace.

Mirrors ``rust/crates/ds41rt-daemon/src/v41_compressor/input_trace.rs`` at
source commit 48f0c712 (schema 1, kind ``compressor-inputs``).  The production
manifest keys records by *public* JSON key; the internal planned-read names
(``output``, ``kv-values``, ``kv-scales``) are what the buffers are named in
memory and in the ``name`` field.  The device descriptor bytes are
authoritative; the staged host descriptors are evidence only.  Malformed,
missing or unsupported captures are rejected explicitly -- nothing is
fabricated to make a capture load.

Weights are resolved **only** under the explicit ``<activations root>/
projection-weights`` directory.  The absolute ``weights.directory`` recorded
in the manifest describes the remote machine and is provenance only.  At least
one first-writer manifest (one that carries the full tensor list) must be
present; manifests that only carry ``already_written_for_root`` are bound to
the validated first writer.  Weight files are read exclusively from the local
projection-weights directory and their actual byte sizes must match.

CPU-only: no torch, no CUDA, no native library.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

SCHEMA_VERSION = 1
CAPTURE_KIND = "compressor-inputs"

SOURCE_DIM = 5120
LATENT_DIM = 512
MAX_SLOTS = 16
DESCRIPTOR_SENTINEL = (1 << 64) - 1

SUPPORTED_RATIO = 2
SUPPORTED_ROW_COUNTS = (1, 2)

KV_VALUE_BYTES = 256
KV_SCALE_BYTES = 32
FREQUENCY_SHAPE = (32, 2)

# Quantization group geometry: group g covers packed columns [g*16, g*16+16).
QUANT_GROUP_WIDTH = 16
EVIDENCE_GROUP_INDEX = 18

# Fixed per-row extents, derived from the trace planner's push_read calls.
ROW_BYTES = {
    "input": SOURCE_DIM * 2,
    "projected": LATENT_DIM * 4,
    "scores": LATENT_DIM * 4,
    "output": LATENT_DIM * 2,
    "frequencies": 32 * 2 * 4,
    "positions": 8,
    "descriptors-device": 8,
    "kv-values": KV_VALUE_BYTES,
    "kv-scales": KV_SCALE_BYTES,
}
PENDING_ROW_BYTES = LATENT_DIM * 4
WEIGHT_WKV_BYTES = LATENT_DIM * SOURCE_DIM * 2
WEIGHT_WGATE_BYTES = LATENT_DIM * SOURCE_DIM * 2
WEIGHT_NORM_BYTES = LATENT_DIM * 2

DTYPE_BYTES = {
    "bfloat16": 2,
    "float32": 4,
    "uint64": 8,
    "fp8e4m3": 1,
    "fp4e2m1": None,  # packed two 4-bit elements per byte
}

# Internal buffer identity -> dtype (used by the reader and by fixtures).
BUFFER_DTYPES = {
    "input": "bfloat16",
    "projected": "float32",
    "scores": "float32",
    "output": "bfloat16",
    "frequencies": "float32",
    "positions": "uint64",
    "descriptors-device": "uint64",
    "kv-values": "fp4e2m1",
    "kv-scales": "fp8e4m3",
    "pending-kv": "float32",
    "pending-scores": "float32",
    "wkv-weight": "bfloat16",
    "wgate-weight": "bfloat16",
    "norm-weight": "bfloat16",
}

# Public JSON buffer key -> (internal name, dtype, shape function).
BUFFER_SPECS = {
    "input": ("input", "bfloat16", lambda rows, slots: [rows, SOURCE_DIM]),
    "projected": ("projected", "float32", lambda rows, slots: [rows, LATENT_DIM]),
    "scores": ("scores", "float32", lambda rows, slots: [rows, LATENT_DIM]),
    "output_before_kv_pack": ("output", "bfloat16",
                              lambda rows, slots: [rows, LATENT_DIM]),
    "frequencies": ("frequencies", "float32", lambda rows, slots: [rows, 32, 2]),
    "positions": ("positions", "uint64", lambda rows, slots: [rows]),
    "kv_values": ("kv-values", "fp4e2m1", lambda rows, slots: [rows, LATENT_DIM]),
    "kv_scales": ("kv-scales", "fp8e4m3",
                  lambda rows, slots: [rows, KV_SCALE_BYTES]),
}

PENDING_SPECS = {
    "kv": ("pending-kv", "float32", lambda rows, slots: [slots, LATENT_DIM]),
    "scores": ("pending-scores", "float32", lambda rows, slots: [slots, LATENT_DIM]),
}

# Weight role -> (dtype, shape).
WEIGHT_SPECS = {
    "wkv": ("bfloat16", (LATENT_DIM, SOURCE_DIM)),
    "wgate": ("bfloat16", (LATENT_DIM, SOURCE_DIM)),
    "norm": ("bfloat16", (LATENT_DIM,)),
}

# Manifest weight role -> buffer identity.
WEIGHT_BUFFER_NAMES = {
    "wkv": "wkv-weight",
    "wgate": "wgate-weight",
    "norm": "norm-weight",
}

WEIGHT_DIRNAME = "projection-weights"

# The pinned writer manifest carries no expected SHA-256 field.
WEIGHT_SHA_NOTE = (
    "the pinned writer manifest provides no expected SHA-256 field; these "
    "hashes are computed by this reader from the shared files and no "
    "comparison is claimed"
)


class CaptureError(Exception):
    """Base error for capture loading/validation failures."""


class MissingFileError(CaptureError):
    """A manifest-referenced raw file is absent."""


class MalformedCaptureError(CaptureError):
    """Manifest content disagrees with the raw bytes or the schema invariants."""


class PathEscapeError(MalformedCaptureError):
    """A manifest-recorded path is absolute, traverses, or escapes its root."""


class WeightError(CaptureError):
    """A weight manifest or weight file is invalid or cannot be bound."""


class UnsupportedCaptureError(CaptureError):
    """A well-formed capture this harness does not support (ratio, rows...)."""


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def typed_bytes(dtype: str, shape: Sequence[int]) -> int:
    """Exact byte extent of ``shape`` elements of ``dtype`` (schema identity)."""
    if dtype not in DTYPE_BYTES:
        raise MalformedCaptureError(f"unknown dtype {dtype!r}")
    elements = 1
    for dim in shape:
        elements *= int(dim)
    if dtype == "fp4e2m1":
        if elements % 2:
            raise MalformedCaptureError(
                f"fp4e2m1 element count {elements} is odd"
            )
        return elements // 2
    return elements * DTYPE_BYTES[dtype]


def resolve_confined(root: Path, name, label: str) -> Path:
    """Resolve a manifest-recorded relative path inside ``root``.

    Absolute paths and any ``..`` component are rejected outright; the
    resolved real path must stay under the real root so a symlink cannot
    escape.
    """
    if not isinstance(name, str) or not name:
        raise PathEscapeError(f"{label}: missing file name")
    candidate = Path(name)
    if candidate.is_absolute():
        raise PathEscapeError(f"{label}: absolute path {name!r} is not permitted")
    if any(part in ("..", "") for part in candidate.parts):
        raise PathEscapeError(f"{label}: traversal path {name!r} is not permitted")
    real_root = root.resolve()
    resolved = (real_root / candidate).resolve()
    if resolved != real_root and real_root not in resolved.parents:
        raise PathEscapeError(f"{label}: {name!r} escapes {real_root}")
    return resolved


# ---------------------------------------------------------------------------
# Predecessor resolution -- exact port of resolve_predecessor in input_trace.rs.
# ---------------------------------------------------------------------------

PENDING_SLOT = "pending_slot"
EARLIER_WAVE_ROW = "earlier_wave_row"
INVALID_SENTINEL = "invalid_sentinel"
UNRESOLVED = "unresolved"


def resolve_predecessor(descriptor: int, row: int, slot_count: int) -> dict:
    """Where one wave row's pooling predecessor lives.

    Pure and total: unresolvable values are reported, never substituted.
    ``row`` is the descriptor owner's index within its own invocation.
    """
    if descriptor == DESCRIPTOR_SENTINEL:
        return {"kind": INVALID_SENTINEL}
    if descriptor < slot_count:
        return {"kind": PENDING_SLOT, "slot": descriptor}
    earlier = descriptor - slot_count
    if earlier < row:
        return {"kind": EARLIER_WAVE_ROW, "wave_row": earlier}
    return {"kind": UNRESOLVED, "value": descriptor}


def predecessor_matches(predecessor: dict, descriptor: int, row: int,
                        slot_count: int) -> bool:
    """Cross-check a manifest-embedded predecessor record against the actual
    device descriptor resolution (the device bytes stay authoritative)."""
    return predecessor == resolve_predecessor(descriptor, row, slot_count)


# ---------------------------------------------------------------------------
# Capture container.
# ---------------------------------------------------------------------------

@dataclass
class BufferRef:
    """One validated raw buffer: typed identity plus the confined local path."""

    name: str
    path: Path
    dtype: str
    shape: Tuple[int, ...]
    bytes: int
    sha256: str = ""


@dataclass
class Capture:
    """One validated ratio-two compressor capture directory."""

    directory: Path
    manifest_path: Path
    manifest_sha256: str
    layer: int
    ratio: int
    rows: int
    slot_count: int
    buffers: Dict[str, BufferRef]               # internal name -> BufferRef
    device_descriptors: Tuple[int, ...]
    host_descriptors: Optional[Tuple[int, ...]]
    device_matches_host: Optional[bool]
    manifest_device_matches_host: Optional[bool]
    chunks: Tuple[dict, ...]
    row_map: Tuple[Tuple[int, int], ...]        # row -> (chunk index, j)
    active_slots: Tuple[int, ...]
    wave_rows: Tuple[dict, ...]
    pending_slots: Tuple[dict, ...]
    activations_root: Optional[Path] = None
    weights_dir: Optional[Path] = None
    weights: Dict[str, BufferRef] = field(default_factory=dict)
    weights_provenance: dict = field(default_factory=dict)

    def read_buffer(self, name: str) -> bytes:
        ref = self.buffers.get(name)
        if ref is None:
            raise MissingFileError(
                f"capture {self.directory}: buffer {name!r} not captured"
            )
        return ref.path.read_bytes()

    def read_weight(self, role: str) -> bytes:
        ref = self.weights.get(role)
        if ref is None:
            raise MissingFileError(
                f"capture {self.directory}: weight {role!r} not captured"
            )
        return ref.path.read_bytes()

    def expected_output_hashes(self) -> dict:
        """Byte oracles for the downstream pipeline (A / D baselines)."""
        return {
            "output": sha256_bytes(self.read_buffer("output")),
            "kv-values": sha256_bytes(self.read_buffer("kv-values")),
            "kv-scales": sha256_bytes(self.read_buffer("kv-scales")),
        }

    def completed_latent(self, row: int) -> Optional[dict]:
        return self.wave_rows[row].get("completed_latent")

    def p41_rows(self) -> List[int]:
        return [r for r, wr in enumerate(self.wave_rows)
                if wr.get("absolute_position") == 41]


# ---------------------------------------------------------------------------
# Low-level record validation.
# ---------------------------------------------------------------------------

def validate_record(record, dtype: str, shape: Sequence[int], base_dir: Path,
                    root: Path, label: str) -> BufferRef:
    """Validate one manifest buffer record against its typed identity.

    The record needs ``file``, ``dtype``, ``shape`` and ``bytes``; the
    production ``name`` field is informational and is never required, because
    the public JSON key already identifies the buffer.
    """
    if not isinstance(record, dict):
        raise MalformedCaptureError(f"{label}: record is not an object")
    for field_name in ("file", "dtype", "shape", "bytes"):
        if field_name not in record:
            raise MalformedCaptureError(f"{label}: record missing {field_name!r}")
    if record["dtype"] != dtype:
        raise MalformedCaptureError(
            f"{label}: dtype {record['dtype']!r} != expected {dtype!r}"
        )
    try:
        got_shape = tuple(int(dim) for dim in record["shape"])
    except (TypeError, ValueError) as error:
        raise MalformedCaptureError(f"{label}: malformed shape: {error}") from error
    want_shape = tuple(int(dim) for dim in shape)
    if got_shape != want_shape:
        raise MalformedCaptureError(
            f"{label}: shape {list(got_shape)} != expected {list(want_shape)}"
        )
    try:
        want = typed_bytes(dtype, got_shape)
    except MalformedCaptureError:
        raise
    if int(record["bytes"]) != want:
        raise MalformedCaptureError(
            f"{label}: manifest bytes {record['bytes']!r} != typed extent {want}"
        )
    path = resolve_confined(base_dir, record["file"], label)
    if not path.is_file():
        raise MissingFileError(f"{label}: file {path} is missing")
    actual = path.stat().st_size
    if actual != want:
        raise MalformedCaptureError(
            f"{label}: file {path.name} is {actual} bytes, expected {want}"
        )
    return BufferRef(label, path, dtype, got_shape, want, sha256_file(path))


def _validate_header(data, layer: Optional[int], manifest_path: Path) -> None:
    if not isinstance(data, dict):
        raise MalformedCaptureError(f"{manifest_path}: manifest is not an object")
    if data.get("schema") != SCHEMA_VERSION:
        raise UnsupportedCaptureError(
            f"{manifest_path}: schema {data.get('schema')!r} != {SCHEMA_VERSION}"
        )
    if data.get("kind") != CAPTURE_KIND:
        raise UnsupportedCaptureError(
            f"{manifest_path}: kind {data.get('kind')!r} != {CAPTURE_KIND!r}"
        )
    if layer is not None and data.get("layer") != layer:
        raise MalformedCaptureError(
            f"{manifest_path}: layer {data.get('layer')!r} != requested {layer}"
        )
    ratio = data.get("ratio")
    if ratio != SUPPORTED_RATIO:
        raise UnsupportedCaptureError(
            f"{manifest_path}: capture ratio {ratio!r}: this harness replays "
            "the ratio-two producer only"
        )
    rows = data.get("rows")
    if rows not in SUPPORTED_ROW_COUNTS:
        raise UnsupportedCaptureError(
            f"{manifest_path}: capture rows {rows!r}: expected a singleton or "
            "two-row decode batch"
        )
    slots = data.get("slot_count")
    if not isinstance(slots, int) or not (1 <= slots <= MAX_SLOTS):
        raise MalformedCaptureError(
            f"{manifest_path}: slot_count {slots!r} outside 1..{MAX_SLOTS}"
        )


# ---------------------------------------------------------------------------
# Chunk / row geometry.
# ---------------------------------------------------------------------------

def build_row_index(data: dict, manifest_path: Path) -> Tuple[list, list]:
    """Validate chunk layout and return ``(chunks, row_map)``.

    ``row_map[r] == (chunk_index, j)`` where ``j`` is the row's index inside
    its chunk.  Chunk offsets must be contiguous from zero and cover exactly
    the manifest row count.
    """
    chunks = data.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise MalformedCaptureError(f"{manifest_path}: manifest has no chunks")
    slot_count = data["slot_count"]
    rows = data["rows"]
    seen_slots = set()
    seen_indices = set()
    offset = 0
    row_map: list = []
    for ordinal, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise MalformedCaptureError(f"{manifest_path}: chunk is not an object")
        for field_name in ("index", "request_id", "lease", "version", "position",
                           "tokens"):
            if field_name not in chunk:
                raise MalformedCaptureError(
                    f"{manifest_path}: chunk missing {field_name!r}"
                )
        index = chunk["index"]
        if not isinstance(index, int) or index in seen_indices:
            raise MalformedCaptureError(
                f"{manifest_path}: chunk index {index!r} is not a unique integer"
            )
        seen_indices.add(index)
        if index != ordinal:
            raise MalformedCaptureError(
                f"{manifest_path}: chunk index {index!r} != native order {ordinal}"
            )
        position = chunk["position"]
        if not isinstance(position, int) or position < 0:
            raise MalformedCaptureError(
                f"{manifest_path}: chunk position must be a non-negative integer"
            )
        tokens = chunk["tokens"]
        if not isinstance(tokens, int) or tokens <= 0:
            raise MalformedCaptureError(
                f"{manifest_path}: chunk tokens must be positive"
            )
        lease = chunk["lease"]
        slot = lease.get("slot") if isinstance(lease, dict) else None
        if not isinstance(slot, int) or not (0 <= slot < slot_count):
            raise MalformedCaptureError(
                f"{manifest_path}: chunk slot outside slot_count"
            )
        if slot in seen_slots:
            raise MalformedCaptureError(
                f"{manifest_path}: duplicate chunk slot {slot}"
            )
        seen_slots.add(slot)
        prepared = chunk.get("prepared_offset", chunk.get("offset", chunk["index"]))
        if prepared != offset:
            raise MalformedCaptureError(
                f"{manifest_path}: chunk prepared_offset {prepared!r} != "
                f"expected {offset}"
            )
        flattened = chunk.get("flattened_row_range")
        if not isinstance(flattened, list) or [int(dim) for dim in flattened] != [
            offset, offset + tokens,
        ]:
            raise MalformedCaptureError(
                f"{manifest_path}: chunk {index} flattened_row_range "
                f"{flattened!r} != derived [{offset}, {offset + tokens})"
            )
        absolute = chunk.get("absolute_positions")
        expected_absolute = list(range(position, position + tokens))
        if not isinstance(absolute, list) or [
            int(dim) for dim in absolute
        ] != expected_absolute:
            raise MalformedCaptureError(
                f"{manifest_path}: chunk {index} absolute_positions "
                f"{absolute!r} != derived {expected_absolute}"
            )
        for j in range(tokens):
            row_map.append((ordinal, j))
        offset += tokens
    if offset != rows:
        raise MalformedCaptureError(
            f"{manifest_path}: chunks cover {offset} rows but manifest "
            f"declares {rows}"
        )
    return chunks, row_map


# ---------------------------------------------------------------------------
# Weight resolution (shared files under <activations>/projection-weights).
# ---------------------------------------------------------------------------

def expected_weight_tensor_names(layer: int) -> Dict[str, str]:
    """Exact full checkpoint tensor names for the requested layer, by role."""
    return {
        "wkv": f"layers.{layer}.attn.compressor.wkv.weight",
        "wgate": f"layers.{layer}.attn.compressor.wgate.weight",
        "norm": f"layers.{layer}.attn.compressor.norm.weight",
    }


def _parse_first_writer_weights(path: Path, data: dict, layer: int,
                                weights_root: Path, root: Path) -> dict:
    """Validate one full first-writer weight manifest into role -> BufferRef."""
    expected_names = expected_weight_tensor_names(layer)
    records: Dict[str, BufferRef] = {}
    tensors = data["weights"].get("tensors")
    if not isinstance(tensors, list):
        raise WeightError(f"{path}: first-writer manifest has no tensor list")
    for tensor in tensors:
        if not isinstance(tensor, dict) or "file" not in tensor:
            raise WeightError(f"{path}: malformed weight tensor record")
        name = tensor.get("tensor")
        role = next(
            (candidate for candidate, exact in expected_names.items()
             if name == exact),
            None,
        )
        if role is None:
            raise WeightError(
                f"{path}: weight tensor {name!r} is not one of the exact "
                f"expected layer-{layer} names {sorted(expected_names.values())}"
            )
        if role in records:
            raise WeightError(
                f"{path}: duplicate weight role {role!r}; weight roles must be "
                "unique"
            )
        dtype, shape = WEIGHT_SPECS[role]
        if tensor.get("dtype") != dtype:
            raise WeightError(
                f"{path}: weight {role} dtype {tensor.get('dtype')!r} != {dtype!r}"
            )
        got_shape = tuple(int(dim) for dim in tensor.get("shape", []))
        if got_shape != shape:
            raise WeightError(f"{path}: weight {role} shape {got_shape} != {shape}")
        want = typed_bytes(dtype, shape)
        if int(tensor.get("bytes", -1)) != want:
            raise WeightError(
                f"{path}: weight {role} bytes {tensor.get('bytes')!r} != {want}"
            )
        resolved = resolve_confined(weights_root, tensor["file"],
                                    f"weights.{role}")
        if not resolved.is_file():
            raise MissingFileError(f"weight {role} file is missing: {resolved}")
        actual = resolved.stat().st_size
        if actual != want:
            raise WeightError(
                f"weight {role} file {resolved.name} is {actual} bytes, "
                f"expected {want}"
            )
        records[role] = BufferRef(WEIGHT_BUFFER_NAMES[role], resolved, dtype,
                                  shape, want, sha256_file(resolved))
    missing = [role for role in ("wkv", "wgate", "norm") if role not in records]
    if missing:
        raise WeightError(f"{path}: first-writer manifest missing weights {missing}")
    return records


def _weight_inventory(records: Dict[str, BufferRef]) -> tuple:
    """Canonical role binding used to compare first-writer manifests."""
    return tuple(
        sorted(
            (
                role,
                WEIGHT_BUFFER_NAMES[role],
                ref.path.name,
                ref.dtype,
                tuple(ref.shape),
                ref.bytes,
            )
            for role, ref in records.items()
        )
    )


def _rel(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def resolve_weights(activations_root: Path, layer: int,
                    manifests: Sequence[Tuple[Path, dict]],
                    *, require_weights: bool = True) -> Tuple[dict, dict]:
    """Validate the shared BF16 weight files and bind later manifests.

    ``manifests`` is every parsed manifest discovered under the activations
    root (not only the selected capture).  The recorded remote ``directory``
    is provenance only; files are read exclusively from
    ``<activations_root>/projection-weights``.
    """
    root = Path(activations_root)
    weights_root = root / WEIGHT_DIRNAME
    first_writers = [
        (path, data) for path, data in manifests
        if isinstance(data.get("weights"), dict)
        and "tensors" in data["weights"]
    ]
    already = [
        (path, data) for path, data in manifests
        if isinstance(data.get("weights"), dict)
        and data["weights"].get("already_written_for_root") is True
    ]
    if not first_writers:
        if not require_weights:
            return {}, {"weights_dir": None, "reason": "weights not required"}
        raise WeightError(
            "missing first-writer weight manifest: no discovered manifest "
            "carries a full tensor list, so already_written_for_root entries "
            "cannot be bound"
        )
    if not weights_root.is_dir():
        raise WeightError(f"shared weight directory is missing: {weights_root}")

    inventories = []
    for path, data in first_writers:
        records = _parse_first_writer_weights(path, data, layer, weights_root,
                                              root)
        inventories.append((path, records, _weight_inventory(records)))
    canonical_path, canonical_records, canonical_inventory = inventories[0]
    for path, _records, inventory in inventories[1:]:
        if inventory != canonical_inventory:
            raise WeightError(
                f"conflicting first-writer weight manifests: {_rel(root, path)} "
                f"describes a different weight inventory than "
                f"{_rel(root, canonical_path)}; every first writer must agree "
                "exactly"
            )

    tensors = {}
    for role, ref in canonical_records.items():
        tensors[role] = BufferRef(ref.name, ref.path, ref.dtype, ref.shape,
                                  ref.bytes, sha256_file(ref.path))

    recorded_remote = None
    first = first_writers[0][1]["weights"].get("directory")
    if isinstance(first, str):
        recorded_remote = first

    bindings = []
    for path, data in already:
        bindings.append({
            "manifest": _rel(root, path),
            "recorded_remote_directory": data["weights"].get("directory"),
            "bound_to_first_writer": _rel(root, canonical_path),
            "note": "no host substitution: shared files validated from the "
                    "first writer",
        })

    provenance = {
        "weights_dir": _rel(root, weights_root),
        "first_writer_manifest": _rel(root, canonical_path),
        "first_writer_manifests": [_rel(root, path) for path, _ in first_writers],
        "already_written_bindings": bindings,
        "recorded_remote_directory_provenance_only": recorded_remote,
        "sha256_note": WEIGHT_SHA_NOTE,
    }
    return tensors, provenance


# ---------------------------------------------------------------------------
# Loading and validation.
# ---------------------------------------------------------------------------

def _parse_manifest(path: Path) -> dict:
    if not path.is_file():
        raise MissingFileError(f"manifest not found: {path}")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise MalformedCaptureError(f"manifest is not JSON: {error}") from error


def _discover_manifest_paths(activations_root: Path) -> List[Path]:
    """Manifests live in ``activations_root`` itself or its immediate subdirs.

    Deeper nesting is not part of the schema: capture directories are immediate
    subdirectories of the activations root that also holds ``projection-weights``.
    """
    root = Path(activations_root)
    if not root.is_dir():
        raise MissingFileError(f"activations root not found: {root}")
    paths = sorted(root.glob("layer*-compressor-inputs.json"))
    for subdir in sorted(root.iterdir()):
        if subdir.is_dir() and subdir.name != WEIGHT_DIRNAME:
            paths += sorted(subdir.glob("layer*-compressor-inputs.json"))
    return paths


def _validate_descriptors(data: dict, base_dir: Path, root: Path):
    descriptors = data.get("descriptors")
    if not isinstance(descriptors, dict):
        raise MalformedCaptureError("descriptors section missing at ratio two")
    rows = data["rows"]
    if descriptors.get("dtype") != "uint64":
        raise MalformedCaptureError(
            f"descriptors dtype {descriptors.get('dtype')!r} != uint64"
        )
    if descriptors.get("rows") != rows:
        raise MalformedCaptureError(
            f"descriptors rows {descriptors.get('rows')!r} != manifest rows {rows}"
        )
    device_path = resolve_confined(base_dir, descriptors.get("device_file"),
                                   "descriptors.device_file")
    host_path = resolve_confined(base_dir, descriptors.get("host_file"),
                                 "descriptors.host_file")
    want = rows * 8
    for label, path in (("descriptors.device_file", device_path),
                        ("descriptors.host_file", host_path)):
        if not path.is_file():
            raise MissingFileError(f"{label}: file {path} is missing")
        actual = path.stat().st_size
        if actual != want:
            raise MalformedCaptureError(
                f"{label}: {path.name} is {actual} bytes, expected {want}"
            )
    device = tuple(int.from_bytes(device_path.read_bytes()[i:i + 8], "little")
                   for i in range(0, want, 8))
    host = tuple(int.from_bytes(host_path.read_bytes()[i:i + 8], "little")
                 for i in range(0, want, 8))
    return device, host, device_path, host_path


def _load_one(directory: Path, manifest_path: Path, activations_root: Path,
              data: dict, weights: Dict[str, BufferRef],
              weights_provenance: dict) -> Capture:
    rows = data["rows"]
    slot_count = data["slot_count"]
    layer = data.get("layer")

    buffers_block = data.get("buffers") or {}
    buffers: Dict[str, BufferRef] = {}
    for public_key, (internal, dtype, shape_fn) in BUFFER_SPECS.items():
        record = buffers_block.get(public_key)
        if record is None:
            raise MalformedCaptureError(
                f"buffers.{public_key} is absent at ratio {SCHEMA_VERSION}"
            )
        buffers[internal] = validate_record(
            record, dtype, shape_fn(rows, slot_count), directory,
            activations_root, f"buffers.{public_key}",
        )

    pending = data.get("pending")
    if not isinstance(pending, dict):
        raise MalformedCaptureError("pending section missing at ratio two")
    for key, (internal, dtype, shape_fn) in PENDING_SPECS.items():
        record = pending.get(key)
        if record is None:
            raise MalformedCaptureError(
                f"pending.{key} is absent at ratio two"
            )
        buffers[internal] = validate_record(
            record, dtype, shape_fn(rows, slot_count), directory,
            activations_root, f"pending.{key}",
        )

    device, host, device_path, _host_path = _validate_descriptors(
        data, directory, activations_root
    )
    buffers["descriptors-device"] = BufferRef(
        "descriptors-device", device_path, "uint64", (rows,), rows * 8,
        sha256_file(device_path),
    )

    chunks, row_map = build_row_index(data, manifest_path)
    wave_rows = data.get("wave_rows")
    if not isinstance(wave_rows, list) or len(wave_rows) != rows:
        raise MalformedCaptureError(
            f"manifest wave_rows must list exactly {rows} rows"
        )
    active_slots = tuple(sorted({chunk["lease"]["slot"] for chunk in chunks}))

    capture = Capture(
        directory=directory,
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        layer=layer,
        ratio=data["ratio"],
        rows=rows,
        slot_count=slot_count,
        buffers=buffers,
        device_descriptors=device,
        host_descriptors=host,
        device_matches_host=device == host,
        manifest_device_matches_host=data.get("descriptors", {}).get(
            "device_matches_host"
        ),
        chunks=tuple(chunks),
        row_map=tuple(row_map),
        active_slots=active_slots,
        wave_rows=tuple(wave_rows),
        pending_slots=tuple(pending.get("slots") or ()),
        activations_root=Path(activations_root),
        weights_dir=weights_provenance.get("weights_dir"),
        weights=dict(weights),
        weights_provenance=dict(weights_provenance),
    )
    validate_wave_geometry(capture)
    return capture


def discover_captures(activations_root: Path, *, layer: Optional[int] = None
                      ) -> List[Path]:
    """Every manifest path under ``activations_root`` (immediate subdirs)."""
    paths = _discover_manifest_paths(Path(activations_root))
    if layer is not None:
        paths = [p for p in paths
                 if p.name.startswith(f"layer{layer}-compressor-inputs.json")]
    return paths


def load_captures(activations_root: Path, *, layer: Optional[int] = None,
                  require_weights: bool = True) -> List[Capture]:
    """Load and validate every capture manifest under ``activations_root``.

    The weights are resolved once for the whole root, exactly as the accepted
    CPU reader does: at least one first-writer manifest must be present when
    weights are required.
    """
    root = Path(activations_root)
    paths = discover_captures(root, layer=layer)
    if not paths:
        raise MissingFileError(
            f"no layer*-compressor-inputs.json under {root} or its immediate "
            "subdirectories"
        )
    parsed = []
    inferred_layer = layer
    for path in paths:
        data = _parse_manifest(path)
        _validate_header(data, None, path)
        if inferred_layer is None:
            inferred_layer = data.get("layer")
        parsed.append((path, data))
    weights, provenance = resolve_weights(
        root, inferred_layer if inferred_layer is not None else 0, parsed,
        require_weights=require_weights,
    )
    captures = []
    for path, data in parsed:
        if layer is not None and data.get("layer") != layer:
            continue
        captures.append(_load_one(path.parent, path, root, data, weights,
                                  provenance))
    return captures


def load_capture(directory: Path, *, activations_root: Optional[Path] = None,
                 manifest_path: Optional[Path] = None,
                 require_weights: bool = True) -> Capture:
    """Load and fully validate one capture directory (CPU only).

    ``activations_root`` is the directory that holds ``projection-weights/``
    (and normally the capture directories themselves).  It defaults to the
    capture directory's parent, which is the production layout.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise MissingFileError(f"capture directory not found: {directory}")
    if manifest_path is None:
        manifests = sorted(directory.glob("layer*-compressor-inputs.json"))
        if not manifests:
            raise MissingFileError(
                f"no layer*-compressor-inputs.json under {directory}"
            )
        if len(manifests) > 1:
            raise MalformedCaptureError(
                f"{directory} holds {len(manifests)} compressor manifests; "
                "pass manifest_path to select one explicitly"
            )
        manifest_path = manifests[0]
    else:
        manifest_path = Path(manifest_path)
    data = _parse_manifest(manifest_path)
    _validate_header(data, None, manifest_path)
    layer = data.get("layer")

    root = Path(activations_root) if activations_root is not None \
        else directory.parent
    # Gather every sibling manifest so the shared first writer can be found.
    siblings = []
    for path in _discover_manifest_paths(root):
        if path == manifest_path:
            siblings.append((path, data))
        else:
            try:
                sibling_data = _parse_manifest(path)
            except CaptureError:
                continue
            if sibling_data.get("layer") == layer:
                siblings.append((path, sibling_data))
    weights, provenance = resolve_weights(root, layer, siblings,
                                          require_weights=require_weights)
    return _load_one(directory, manifest_path, root, data, weights, provenance)


# ---------------------------------------------------------------------------
# Wave geometry validation.
# ---------------------------------------------------------------------------

def _u64_values(blob: bytes, what: str) -> Tuple[int, ...]:
    if len(blob) % 8:
        raise MalformedCaptureError(
            f"{what} byte length {len(blob)} is not a multiple of 8"
        )
    return tuple(int.from_bytes(blob[i:i + 8], "little")
                 for i in range(0, len(blob), 8))


def validate_wave_geometry(capture_like: Capture) -> None:
    """Owner/lease adjacency and latent mapping checks.

    Enforces, from the actual device descriptors, the typed per-row buffers and
    the chunk provenance:
      * a manifest ``device_descriptor`` that disagrees with the actual device
        file fails qualification (device bytes authoritative);
      * even absolute positions carry the sentinel, complete no latent and
        carry a zero positions entry;
      * odd absolute positions complete the pair ``(position-1, position)`` and
        their positions entry equals ``position-1``;
      * a pending predecessor must be the row's own lease slot;
      * an earlier-wave predecessor must be the same chunk's chronologically
        adjacent row (absolute position ``position-1``).
    """
    ratio = capture_like.ratio
    rows = capture_like.rows
    if len(capture_like.device_descriptors) != rows:
        raise MalformedCaptureError(
            f"device descriptor count {len(capture_like.device_descriptors)} "
            f"!= rows {rows}"
        )
    if capture_like.host_descriptors is not None \
            and len(capture_like.host_descriptors) != rows:
        raise MalformedCaptureError(
            f"host descriptor count {len(capture_like.host_descriptors)} "
            f"!= rows {rows}"
        )
    positions = _u64_values(capture_like.read_buffer("positions"),
                            "positions")
    if len(positions) != rows:
        raise MalformedCaptureError(
            f"positions count {len(positions)} != rows {rows}"
        )
    if len(capture_like.row_map) != rows or len(capture_like.wave_rows) != rows:
        raise MalformedCaptureError("wave row index length mismatch")

    for row in range(rows):
        chunk_ordinal, j = capture_like.row_map[row]
        chunk = capture_like.chunks[chunk_ordinal]
        wave_row = capture_like.wave_rows[row]
        if not isinstance(wave_row, dict):
            raise MalformedCaptureError(f"wave_rows[{row}] is not an object")
        position = chunk["position"] + j
        if wave_row.get("row") != row:
            raise MalformedCaptureError(
                f"wave_rows[{row}].row {wave_row.get('row')!r} != {row}"
            )
        if wave_row.get("chunk") != chunk["index"]:
            raise MalformedCaptureError(
                f"wave_rows[{row}].chunk {wave_row.get('chunk')!r} != "
                f"{chunk['index']!r}"
            )
        if wave_row.get("absolute_position") != position:
            raise MalformedCaptureError(
                f"wave_rows[{row}].absolute_position "
                f"{wave_row.get('absolute_position')!r} != derived {position}"
            )
        if wave_row.get("request_id") != chunk["request_id"]:
            raise MalformedCaptureError(
                f"wave_rows[{row}].request_id {wave_row.get('request_id')!r} "
                f"!= chunk {chunk['request_id']!r}"
            )

        descriptor = capture_like.device_descriptors[row]
        if wave_row.get("device_descriptor") != descriptor:
            raise MalformedCaptureError(
                f"row {row} manifest device_descriptor "
                f"{wave_row.get('device_descriptor')!r} differs from actual "
                f"device file value {descriptor}; qualification fails"
            )
        predecessor = resolve_predecessor(descriptor, row,
                                          capture_like.slot_count)
        if predecessor["kind"] == UNRESOLVED:
            raise MalformedCaptureError(
                f"row {row} descriptor {descriptor} is unresolvable "
                f"(slot_count {capture_like.slot_count})"
            )
        embedded = wave_row.get("predecessor")
        if embedded is not None and embedded != predecessor:
            raise MalformedCaptureError(
                f"row {row} manifest predecessor {embedded} disagrees with the "
                f"device descriptor resolution {predecessor}"
            )

        lease = chunk["lease"]
        own_slot = lease.get("slot") if isinstance(lease, dict) else None
        if predecessor["kind"] == PENDING_SLOT:
            slot = predecessor["slot"]
            if slot not in capture_like.active_slots:
                raise MalformedCaptureError(
                    f"row {row} pending slot {slot} is unscored padding "
                    f"(active slots {list(capture_like.active_slots)})"
                )
            if slot != own_slot:
                raise MalformedCaptureError(
                    f"row {row} pending slot {slot} belongs to another active "
                    f"chunk; the row's own lease slot is {own_slot!r}"
                )
        elif predecessor["kind"] == EARLIER_WAVE_ROW:
            earlier = predecessor["wave_row"]
            pred_chunk_ordinal, pred_j = capture_like.row_map[earlier]
            pred_chunk = capture_like.chunks[pred_chunk_ordinal]
            pred_position = pred_chunk["position"] + pred_j
            if pred_chunk is not chunk:
                raise MalformedCaptureError(
                    f"row {row} earlier-wave row {earlier} belongs to chunk "
                    f"{pred_chunk['index']!r}, not the selected chunk "
                    f"{chunk['index']!r}"
                )
            if pred_position != position - 1:
                raise MalformedCaptureError(
                    f"row {row} earlier-wave row {earlier} absolute position "
                    f"{pred_position} != prior absolute position {position - 1}"
                )

        completed = wave_row.get("completed_latent")
        if position % ratio == 0:
            if descriptor != DESCRIPTOR_SENTINEL:
                raise MalformedCaptureError(
                    f"row {row} at even position {position} carries "
                    f"descriptor {descriptor}, expected the sentinel"
                )
            if completed is not None:
                raise MalformedCaptureError(
                    f"row {row} at even position {position} carries a "
                    "completed_latent but completes no latent"
                )
            if positions[row] != 0:
                raise MalformedCaptureError(
                    f"row {row} positions padding {positions[row]} nonzero for "
                    "a row completing no latent"
                )
        else:
            if descriptor == DESCRIPTOR_SENTINEL:
                raise MalformedCaptureError(
                    f"row {row} at odd position {position} completes a latent "
                    "but carries the sentinel descriptor"
                )
            expected_first = position - (position % ratio)
            if not isinstance(completed, dict):
                raise MalformedCaptureError(
                    f"row {row} at odd position {position} has no "
                    "completed_latent"
                )
            if completed.get("first_token") != expected_first:
                raise MalformedCaptureError(
                    f"row {row} latent first_token {completed.get('first_token')!r} "
                    f"!= {expected_first} for position {position}"
                )
            if completed.get("logical_compressed_row") != expected_first // ratio:
                raise MalformedCaptureError(
                    f"row {row} logical_compressed_row "
                    f"{completed.get('logical_compressed_row')!r} != "
                    f"{expected_first // ratio}"
                )
            if completed.get("source_row") != row:
                raise MalformedCaptureError(
                    f"row {row} latent source_row {completed.get('source_row')!r} "
                    f"!= row {row}"
                )
            if completed.get("request_id") != chunk["request_id"]:
                raise MalformedCaptureError(
                    f"row {row} latent request_id "
                    f"{completed.get('request_id')!r} != chunk "
                    f"{chunk['request_id']!r}"
                )
            if positions[row] != expected_first:
                raise MalformedCaptureError(
                    f"row {row} positions value {positions[row]} != required "
                    f"completed first token {expected_first}"
                )
