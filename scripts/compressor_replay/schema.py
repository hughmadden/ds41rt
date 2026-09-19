"""Capture schema for the ratio-two compressor producer input trace.

Mirrors ``rust/crates/ds41rt-daemon/src/v41_compressor/input_trace.rs`` at
source commit 48f0c712 (schema 1, kind ``compressor-inputs``).  The device
descriptor bytes are authoritative; the staged host descriptors are evidence
only.  Malformed, missing or unsupported captures are rejected explicitly —
nothing is fabricated to make a capture load.

CPU-only: no torch, no CUDA, no native library.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

SCHEMA_VERSION = 1
CAPTURE_KIND = "compressor-inputs"

SOURCE_DIM = 5120
LATENT_DIM = 512
MAX_SLOTS = 16
DESCRIPTOR_SENTINEL = (1 << 64) - 1

# This harness targets the ratio-two producer; ratio one and anything else is
# rejected as unsupported rather than silently reinterpreted.
SUPPORTED_RATIO = 2
# Actual captures are expected as singleton and two-row decode batches.
SUPPORTED_ROW_COUNTS = (1, 2)

# Fixed per-row extents, derived from the trace planner's push_read calls.
ROW_BYTES = {
    "input": SOURCE_DIM * 2,              # BF16 [rows, 5120]
    "projected": LATENT_DIM * 4,          # FP32 [rows, 512] at ratio two
    "scores": LATENT_DIM * 4,             # FP32 [rows, 512] at ratio two
    "output": LATENT_DIM * 2,             # BF16 [rows, 512] (pre-pack input)
    "frequencies": 32 * 2 * 4,            # FP32 [rows, 32, 2]
    "positions": 8,                       # U64 [rows]
    "descriptors-device": 8,              # U64 [rows] (authoritative)
    "kv-values": 256,                     # packed FP4 rows x 256 bytes
    "kv-scales": 32,                      # E4M3 scale rows x 32 bytes
}
PENDING_ROW_BYTES = LATENT_DIM * 4        # FP32 [slot_count, 512]
WEIGHT_WKV_BYTES = LATENT_DIM * SOURCE_DIM * 2    # BF16 [512, 5120]
WEIGHT_WGATE_BYTES = LATENT_DIM * SOURCE_DIM * 2  # BF16 [512, 5120]
WEIGHT_NORM_BYTES = LATENT_DIM * 2                # BF16 [512]

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


class CaptureError(Exception):
    """Base error for capture loading/validation failures."""


class MissingFileError(CaptureError):
    """A manifest-referenced raw file is absent."""


class MalformedCaptureError(CaptureError):
    """Manifest content disagrees with the raw bytes or the schema invariants."""


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


# ---------------------------------------------------------------------------
# Predecessor resolution — exact port of resolve_predecessor in input_trace.rs.
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
    buffers: dict                     # name -> BufferRef
    device_descriptors: Tuple[int, ...]
    host_descriptors: Optional[Tuple[int, ...]]
    device_matches_host: Optional[bool]
    wave_rows: Tuple[dict, ...]
    chunks: Tuple[dict, ...]
    pending_slots: Tuple[dict, ...]
    weights_dir: Optional[Path] = None
    weights: dict = field(default_factory=dict)   # role -> BufferRef

    def read_buffer(self, name: str) -> bytes:
        ref = self.buffers.get(name)
        if ref is None:
            raise MissingFileError(f"capture {self.directory}: buffer {name!r} not captured")
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


# ---------------------------------------------------------------------------
# Loading and validation.
# ---------------------------------------------------------------------------

REQUIRED_RATIO2_BUFFERS = (
    "input", "projected", "scores", "output", "frequencies", "positions",
    "descriptors-device", "kv-values", "kv-scales", "pending-kv",
    "pending-scores",
)


def _find_manifest(directory: Path) -> Path:
    if not directory.is_dir():
        raise MissingFileError(f"capture directory not found: {directory}")
    manifests = sorted(directory.glob("layer*-compressor-inputs.json"))
    if not manifests:
        raise MissingFileError(
            f"no layer*-compressor-inputs.json under {directory}"
        )
    if len(manifests) > 1:
        raise MalformedCaptureError(
            f"{directory} holds {len(manifests)} compressor manifests; "
            "pass --manifest to select one explicitly"
        )
    return manifests[0]


def _expected_shape(name: str, rows: int, slot_count: int) -> Tuple[int, ...]:
    if name in ("pending-kv", "pending-scores"):
        return (slot_count, LATENT_DIM)
    if name == "frequencies":
        return (rows, 32, 2)
    if name in ("positions", "descriptors-device"):
        return (rows,)
    if name == "kv-values":
        return (rows, 256 * 2)
    if name == "kv-scales":
        return (rows, 32)
    if name in ("wkv-weight", "wgate-weight"):
        return (LATENT_DIM, SOURCE_DIM)
    if name == "norm-weight":
        return (LATENT_DIM,)
    return (rows, LATENT_DIM if name != "input" else SOURCE_DIM)


def _row_count_for(name: str, rows: int, slot_count: int) -> int:
    return slot_count if name in ("pending-kv", "pending-scores") else rows


def _validate_buffer_record(directory: Path, record: dict, rows: int,
                            slot_count: int) -> BufferRef:
    name = record.get("name")
    if name not in BUFFER_DTYPES:
        raise MalformedCaptureError(f"unknown buffer record {name!r}")
    if record.get("dtype") != BUFFER_DTYPES[name]:
        raise MalformedCaptureError(
            f"buffer {name!r} dtype {record.get('dtype')!r} != {BUFFER_DTYPES[name]!r}"
        )
    shape = tuple(record.get("shape") or ())
    if shape != _expected_shape(name, rows, slot_count):
        raise MalformedCaptureError(
            f"buffer {name!r} shape {shape} != expected "
            f"{_expected_shape(name, rows, slot_count)}"
        )
    expected_bytes = _row_count_for(name, rows, slot_count) * {
        **ROW_BYTES,
        "pending-kv": PENDING_ROW_BYTES,
        "pending-scores": PENDING_ROW_BYTES,
        "wkv-weight": WEIGHT_WKV_BYTES,
        "wgate-weight": WEIGHT_WGATE_BYTES,
        "norm-weight": WEIGHT_NORM_BYTES,
    }[name]
    if record.get("bytes") != expected_bytes:
        raise MalformedCaptureError(
            f"buffer {name!r} bytes {record.get('bytes')} != {expected_bytes}"
        )
    path = directory / record["file"]
    if not path.is_file():
        raise MissingFileError(f"capture buffer file missing: {path}")
    ref = BufferRef(name, path, record["dtype"], shape, expected_bytes,
                    sha256_file(path))
    return ref


def _u64_tuple(blob: bytes, what: str) -> Tuple[int, ...]:
    if len(blob) % 8:
        raise MalformedCaptureError(f"{what} byte length {len(blob)} is not a multiple of 8")
    return tuple(
        int.from_bytes(blob[i:i + 8], "little") for i in range(0, len(blob), 8)
    )


def validate_wave_geometry(capture_like) -> None:
    """Owner/lease adjacency and latent mapping checks shared by the loader
    and by synthetic-fixture tests.

    Enforces, from the actual device descriptors and chunk provenance:
      * even absolute positions carry the sentinel (no pooling predecessor);
      * odd absolute positions complete the latent [position-1, position];
      * earlier-wave predecessors are chronologically adjacent (row-1);
      * completed latents map first_token = position - (position % ratio) and
        logical_compressed_row = first_token // ratio (p41 -> 40 / 20).
    """
    ratio = capture_like.ratio
    for row in range(capture_like.rows):
        wave_row = capture_like.wave_rows[row]
        position = wave_row["absolute_position"]
        if position % ratio == 0:
            if wave_row["device_descriptor"] != DESCRIPTOR_SENTINEL:
                raise MalformedCaptureError(
                    f"row {row} at even position {position} carries descriptor "
                    f"{wave_row['device_descriptor']:#x}, expected the sentinel"
                )
        else:
            if wave_row["device_descriptor"] == DESCRIPTOR_SENTINEL:
                raise MalformedCaptureError(
                    f"row {row} at odd position {position} completes a latent "
                    "but carries the sentinel descriptor"
                )
        predecessor = resolve_predecessor(
            wave_row["device_descriptor"], row, capture_like.slot_count
        )
        if predecessor["kind"] == UNRESOLVED:
            raise MalformedCaptureError(
                f"row {row} descriptor {wave_row['device_descriptor']:#x} is "
                f"unresolvable (slot_count {capture_like.slot_count})"
            )
        if predecessor["kind"] == EARLIER_WAVE_ROW and predecessor["wave_row"] != row - 1:
            raise MalformedCaptureError(
                f"row {row} predecessor wave row {predecessor['wave_row']} is not "
                f"the chronologically adjacent row {row - 1}"
            )
        embedded = wave_row.get("predecessor")
        if embedded is not None and embedded != predecessor:
            raise MalformedCaptureError(
                f"row {row} manifest predecessor {embedded} disagrees with the "
                f"device descriptor resolution {predecessor}"
            )
        latent = wave_row.get("completed_latent")
        if latent is not None:
            expected_first = position - (position % ratio)
            if latent["first_token"] != expected_first:
                raise MalformedCaptureError(
                    f"row {row} latent first_token {latent['first_token']} != "
                    f"{expected_first} for position {position}"
                )
            if latent["logical_compressed_row"] != expected_first // ratio:
                raise MalformedCaptureError(
                    f"row {row} logical_compressed_row "
                    f"{latent['logical_compressed_row']} != "
                    f"{expected_first // ratio}"
                )


def load_capture(directory: Path, *, manifest_path: Optional[Path] = None,
                 require_weights: bool = True) -> Capture:
    """Load and fully validate one capture directory (CPU only)."""
    directory = Path(directory)
    manifest_path = Path(manifest_path) if manifest_path else _find_manifest(directory)
    if not manifest_path.is_file():
        raise MissingFileError(f"manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as error:
        raise MalformedCaptureError(f"manifest is not JSON: {error}") from error

    if manifest.get("schema") != SCHEMA_VERSION:
        raise UnsupportedCaptureError(
            f"manifest schema {manifest.get('schema')!r} != {SCHEMA_VERSION}"
        )
    if manifest.get("kind") != CAPTURE_KIND:
        raise UnsupportedCaptureError(
            f"manifest kind {manifest.get('kind')!r} != {CAPTURE_KIND!r}"
        )
    ratio = manifest.get("ratio")
    if ratio != SUPPORTED_RATIO:
        raise UnsupportedCaptureError(
            f"capture ratio {ratio!r}: this harness replays the ratio-two "
            "producer only"
        )
    rows = manifest.get("rows")
    if rows not in SUPPORTED_ROW_COUNTS:
        raise UnsupportedCaptureError(
            f"capture rows {rows!r}: expected a singleton or two-row decode "
            "batch"
        )
    slot_count = manifest.get("slot_count")
    if not isinstance(slot_count, int) or not (1 <= slot_count <= MAX_SLOTS):
        raise MalformedCaptureError(
            f"slot_count {slot_count!r} outside 1..{MAX_SLOTS}"
        )
    layer = manifest.get("layer")

    buffers_record = manifest.get("buffers") or {}
    pending = manifest.get("pending") or {}
    # Key every record by its own validated ``name``: the production JSON
    # keys differ (output_before_kv_pack, kv_values, ...), the record name
    # is the schema identity.
    records = {}
    for record in buffers_record.values():
        if record is not None:
            records[record["name"]] = record
    for name in ("kv", "scores"):
        record = pending.get(name)
        if record is not None:
            records[record["name"]] = record
    # Device descriptors are named by the descriptors section, not the
    # buffers section (device bytes authoritative, host staged for compare).
    descriptors_info = manifest.get("descriptors")
    if descriptors_info and descriptors_info.get("device_file"):
        records["descriptors-device"] = {
            "name": "descriptors-device",
            "file": descriptors_info["device_file"],
            "dtype": "uint64",
            "shape": [rows],
            "bytes": rows * 8,
        }
    missing = [name for name in REQUIRED_RATIO2_BUFFERS if name not in records]
    if missing:
        raise MalformedCaptureError(
            f"manifest is missing ratio-two buffers: {missing}"
        )

    buffers = {}
    for name, record in records.items():
        ref = _validate_buffer_record(directory, record, rows, slot_count)
        buffers[name] = ref

    device_descriptors = _u64_tuple(
        buffers["descriptors-device"].path.read_bytes(), "device descriptors"
    )
    if len(device_descriptors) != rows:
        raise MalformedCaptureError(
            f"device descriptor count {len(device_descriptors)} != rows {rows}"
        )

    host_descriptors = None
    device_matches_host = None
    host_path = directory / f"layer{layer}-compressor-descriptors-host.bin"
    if host_path.is_file():
        host_descriptors = _u64_tuple(host_path.read_bytes(), "host descriptors")
        if len(host_descriptors) != rows:
            raise MalformedCaptureError(
                f"host descriptor count {len(host_descriptors)} != rows {rows}"
            )
        device_matches_host = device_descriptors == host_descriptors
    if descriptors_info is not None and descriptors_info.get("device_matches_host") != device_matches_host:
        # Informational only: the device bytes stay authoritative.
        device_matches_host = device_descriptors == host_descriptors

    wave_rows = tuple(manifest.get("wave_rows") or ())
    if len(wave_rows) != rows:
        raise MalformedCaptureError(
            f"manifest wave_rows {len(wave_rows)} != rows {rows}"
        )
    chunks = tuple(manifest.get("chunks") or ())
    if not chunks:
        raise MalformedCaptureError("manifest has no chunks")

    capture = Capture(
        directory=directory,
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        layer=layer,
        ratio=ratio,
        rows=rows,
        slot_count=slot_count,
        buffers=buffers,
        device_descriptors=device_descriptors,
        host_descriptors=host_descriptors,
        device_matches_host=device_matches_host,
        wave_rows=wave_rows,
        chunks=chunks,
        pending_slots=tuple((pending.get("slots") or ())),
    )
    validate_wave_geometry(capture)

    weights_info = manifest.get("weights") or {}
    weights_dir_raw = weights_info.get("directory")
    if weights_dir_raw:
        weights_dir = Path(weights_dir_raw)
        if not weights_dir.is_absolute():
            weights_dir = directory / weights_dir
        capture.weights_dir = weights_dir
    if weights_info.get("already_written_for_root"):
        if weights_dir_raw is None:
            raise MalformedCaptureError(
                "manifest says weights already written but gives no directory"
            )
        # Tensor file names follow the planner convention.
        for role, name in (("wkv", "wkv-weight"), ("wgate", "wgate-weight"),
                           ("norm", "norm-weight")):
            record = {"name": name, "dtype": BUFFER_DTYPES[name],
                      "shape": list(_expected_shape(name, rows, slot_count)),
                      "bytes": {
                          "wkv-weight": WEIGHT_WKV_BYTES,
                          "wgate-weight": WEIGHT_WGATE_BYTES,
                          "norm-weight": WEIGHT_NORM_BYTES,
                      }[name],
                      "file": f"layer{layer}-compressor-{name}.bin"}
            capture.weights[role] = _validate_weight_record(weights_dir, record)
    else:
        by_filename = {t.get("file"): t for t in weights_info.get("tensors") or []}
        for role, name in (("wkv", "wkv-weight"), ("wgate", "wgate-weight"),
                           ("norm", "norm-weight")):
            tensor = by_filename.get(f"layer{layer}-compressor-{name}.bin")
            if tensor is None:
                continue
            capture.weights[role] = _validate_weight_record(
                capture.weights_dir or directory,
                {"name": name, "dtype": tensor.get("dtype"),
                 "shape": tensor.get("shape"),
                 "bytes": tensor.get("bytes"),
                 "file": tensor.get("file")},
            )
    if require_weights:
        for role in ("wkv", "wgate", "norm"):
            if role not in capture.weights:
                raise MissingFileError(
                    f"capture {directory} does not reference a {role} weight tensor"
                )
    return capture


def _validate_weight_record(weights_dir: Path, record: dict) -> BufferRef:
    name = record.get("name")
    if name not in BUFFER_DTYPES or not name.endswith("weight"):
        raise MalformedCaptureError(f"unknown weight record {name!r}")
    if record.get("dtype") != BUFFER_DTYPES[name]:
        raise MalformedCaptureError(f"weight {name!r} dtype mismatch")
    expected = {
        "wkv-weight": (WEIGHT_WKV_BYTES, (LATENT_DIM, SOURCE_DIM)),
        "wgate-weight": (WEIGHT_WGATE_BYTES, (LATENT_DIM, SOURCE_DIM)),
        "norm-weight": (WEIGHT_NORM_BYTES, (LATENT_DIM,)),
    }[name]
    if tuple(record.get("shape") or ()) != expected[1]:
        raise MalformedCaptureError(f"weight {name!r} shape mismatch")
    if record.get("bytes") != expected[0]:
        raise MalformedCaptureError(f"weight {name!r} byte size mismatch")
    path = weights_dir / record["file"]
    if not path.is_file():
        raise MissingFileError(f"weight file missing: {path}")
    return BufferRef(name, path, record["dtype"], expected[1], expected[0],
                     sha256_file(path))
