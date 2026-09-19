#!/usr/bin/env python3
"""Projection-only component replay for the layer-2 compressor.

This is a small, self-contained experiment.  It does *not* depend on the wider
compressor replay framework.  It validates the frozen activation fixture, builds
an exact case plan, and (only with ``--execute``) drives the three existing native
exports:

    ds41rt_v41_compressor_create
    ds41rt_v41_compressor_project
    ds41rt_v41_compressor_destroy

Everything before ``--execute`` is CPU-only and must never import torch.  The
native library's SHA-256 is verified *before* the shared object is loaded.

Design decisions that the task specification leaves open are recorded explicitly
in the emitted plan artifact (see ``provenance`` on every case) so the root owner
can review them.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Frozen constants
# ---------------------------------------------------------------------------

LAYER = 2
RATIO = 2
INPUT_ROW_BYTES = 10240            # BF16 [5120]
OUTPUT_ROW_BYTES = 2048            # FP32  [512]
WEIGHT_BYTES = 5242880             # BF16 [512, 5120]
WORKSPACE_BYTES = 4 * 1024 * 1024  # caller-owned 4 MiB
WORKSPACE_ALIGN = 256
REPEAT_PHASES = ("repeat1", "repeat2", "repeat3")
ALL_PHASES = ("warmup",) + REPEAT_PHASES

REFERENCE_DIR = "lane0-batch20-Full-1rows"
CANDIDATE_DIR = "lane0-batch81-Full-2rows"
MANIFEST_NAME = "layer2-compressor-inputs.json"
INPUT_FILE = "layer2-compressor-input.bin"
PROJECTED_FILE = "layer2-compressor-projected.bin"
SCORES_FILE = "layer2-compressor-scores.bin"
WEIGHTS_DIR = "projection-weights"
WKV_FILE = "layer2-compressor-wkv-weight.bin"
WGATE_FILE = "layer2-compressor-wgate-weight.bin"

# Case geometry for the counterfactual (CF) shape experiments.
# Every CF case carries exactly one real input row, placed at its target slot.
# Every other row is zero except the flip-neighbor row, which is the real row
# with a single byte flipped (see CF_NEIGHBOR_XOR_BYTE_OFFSET).
CF_TARGET_SLOTS: Dict[int, Tuple[int, ...]] = {2: (0, 1), 16: (0, 1, 7, 15)}
CF_NEIGHBOR_XOR_BYTE_OFFSET = 10238  # lowest bit of the last BF16 word
CF_NEIGHBOR_XOR_MASK = 1

# Output sub-directories.  Recorded repeats are the scored matrix; warmup
# downloads are retained separately and are never scored.
RECORDED_SUBDIR = "recorded"
WARMUP_SUBDIR = "warmup"

# Frozen root-proof hashes.  These identify the immutable fixture.
FREEZE: Dict[str, str] = {
    "reference_input_sha256":
        "503f93fcdff6129640371b870fd7360fb839651315ac1a2c7cd5ae0cc6391d15",
    "wkv_weight_sha256":
        "c4c6702055bfe1f8e95edc306eb9a796fdad8d894b59ff329e2d5325a719d044",
    "wgate_weight_sha256":
        "01826cbe5db945df69b3456c884cf1eccbe47840d91aacf7583640d06a25a36e",
    "reference_projected_sha256":
        "b6f3975b26580ea4394e44430fb016e0e0bf33f3605970552d26bb2afa667f41",
    "reference_scores_sha256":
        "39acafda1913029f21b78fb7073da76d3020f5edd8822ab01404ee07f64fc992",
    "candidate_projected_row_sha256":
        "14f10b70ed42ada6e1cb5b02d303711151d24d22236b695199b3d133de905d91",
    "candidate_scores_row_sha256":
        "5ae1783342a88a332c8dc81ddd95d54e503af9f124a02c5a845ccb2c11fd4de8",
    "native_library_sha256":
        "e0921fed65a37ee6679a3cfe84a96ee91365ae6cf7b6d529f4476cba3b333533",
}

NATIVE_SHA256_FROZEN = FREEZE["native_library_sha256"]


class ProjectionReplayError(Exception):
    """Base class for this experiment's failures."""


class FixtureError(ProjectionReplayError):
    """The activation fixture does not match the frozen identity."""


class ValidationError(ProjectionReplayError):
    """Bad CLI/native-library/plan input."""


class ExecutionError(ProjectionReplayError):
    """The native component could not be executed."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sydney_now() -> datetime.datetime:
    try:
        from zoneinfo import ZoneInfo
        return datetime.datetime.now(ZoneInfo("Australia/Sydney"))
    except Exception:  # pragma: no cover - zoneinfo always present on py>=3.9
        tz = datetime.timezone(datetime.timedelta(hours=10))
        return datetime.datetime.now(tz)


def sydney_now_iso() -> str:
    return sydney_now().isoformat()


def verify_expected_hash(actual: str, expected: str, label: str) -> None:
    """Raise FixtureError on mismatch; return None on match."""
    if actual != expected:
        raise FixtureError(
            f"{label}: expected sha256 {expected}, got {actual}")


def input_rows_all_equal(rows: Sequence[bytes], expected_sha256: str) -> bool:
    """True iff every captured input row hashes exactly to the frozen value."""
    if not rows:
        return False
    return all(sha256_bytes(r) == expected_sha256 for r in rows)


def split_rows(data: bytes, row_bytes: int) -> List[bytes]:
    if len(data) % row_bytes != 0:
        raise FixtureError(
            f"byte count {len(data)} is not a multiple of row size {row_bytes}")
    return [data[i:i + row_bytes] for i in range(0, len(data), row_bytes)]


def confined_path(root: str, *parts: str) -> str:
    """Resolve ``root/parts...`` and refuse any symlink escape outside root."""
    root_real = os.path.realpath(root)
    joined = os.path.join(root_real, *parts)
    if os.path.islink(joined):
        raise FixtureError(f"symlink not allowed: {joined}")
    real = os.path.realpath(joined)
    try:
        common = os.path.commonpath([root_real, real])
    except ValueError as exc:  # different drives / weird inputs
        raise FixtureError(f"path escapes fixture root: {joined}") from exc
    if common != root_real:
        raise FixtureError(f"path escapes fixture root: {joined} -> {real}")
    return real


def require_regular_file(path: str, label: str) -> None:
    if not os.path.isfile(path):
        raise FixtureError(f"{label}: not a regular file: {path}")
    if os.path.islink(path):
        raise FixtureError(f"{label}: symlink not allowed: {path}")


def validate_native_sha256_format(value: str) -> str:
    if not isinstance(value, str):
        raise ValidationError("native sha256 must be a string")
    text = value.strip().lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ValidationError(
            f"native sha256 must be 64 hex characters, got {value!r}")
    return text


def verify_native_library_sha256(path: str, expected_sha256: str) -> str:
    """Compute and check a native library file's hash *without loading it*."""
    expected = validate_native_sha256_format(expected_sha256)
    if not os.path.isfile(path):
        raise ValidationError(f"native library not found: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValidationError(
            f"native library sha256 mismatch: expected {expected}, got {actual}")
    return actual


# ---------------------------------------------------------------------------
# Native buffers (raw bytes only; no numeric casts anywhere)
# ---------------------------------------------------------------------------

CUDA_DEVICE = "cuda"


class DeviceBuffer:
    """A raw-byte buffer that lives on the CUDA device, allocated by torch.

    Production allocation is exactly ``torch.empty(nbytes, dtype=torch.uint8,
    device="cuda")`` and the pointer handed to native is ``tensor.data_ptr()``.
    Initial bytes are uploaded with ``torch.frombuffer(bytearray(data),
    dtype=torch.uint8)`` followed by ``tensor.copy_(source)`` (a synchronous
    host-to-device copy).  The bytes are never interpreted as floats, so BF16
    patterns round-trip unchanged.

    ``tobytes`` performs the CPU download ``tensor.cpu().numpy().tobytes()``;
    the caller is responsible for synchronizing the stream before calling it.
    """

    __slots__ = ("_torch", "_tensor", "_nbytes", "_alignment", "_host",
                 "_source", "label")

    def __init__(self, torch_module: Any, data: Optional[bytes] = None, *,
                 nbytes: Optional[int] = None, alignment: int = 1,
                 label: Optional[str] = None) -> None:
        if data is not None:
            if nbytes is not None and nbytes != len(data):
                raise ValueError("nbytes disagrees with data length")
            nbytes = len(data)
        if nbytes is None:
            raise ValueError("DeviceBuffer needs data or nbytes")
        if nbytes < 0:
            raise ValueError("nbytes must be non-negative")
        self._torch = torch_module
        self._nbytes = nbytes
        self._alignment = max(int(alignment), 1)
        self.label = label
        # Actual CUDA raw-byte allocation.  No CPU fallback.
        self._tensor = torch_module.empty(
            nbytes, dtype=torch_module.uint8, device=CUDA_DEVICE)
        self._host: Optional[bytearray] = None
        self._source: Any = None
        if data is not None:
            host = bytearray(data)
            source = torch_module.frombuffer(host, dtype=torch_module.uint8)
            self._tensor.copy_(source)
            # Keep both the host bytes and the host tensor alive: frombuffer
            # shares the bytearray's memory and copy_ is a synchronous copy.
            self._host = host
            self._source = source
        ptr = self.ptr
        if self._alignment > 1 and (ptr % self._alignment) != 0:
            raise ExecutionError(
                f"device pointer {ptr} is not {self._alignment}-byte aligned")

    @property
    def ptr(self) -> int:
        return int(self._tensor.data_ptr())

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def zero(self) -> None:
        """Zero a newly allocated output buffer (never inputs/weights)."""
        self._tensor.zero_()

    def tobytes(self) -> bytes:
        """Download after the caller has synchronized the CUDA stream."""
        return self._tensor.cpu().numpy().tobytes()


def case_input_nbytes(rows: int) -> int:
    return rows * INPUT_ROW_BYTES


def case_output_nbytes(rows: int) -> int:
    return rows * OUTPUT_ROW_BYTES


# ---------------------------------------------------------------------------
# Native binding
# ---------------------------------------------------------------------------

class CtypesNativeBinding:
    """Exact FFI binding for the three existing compressor exports.

    All pointer arguments use ``c_void_p`` and the raw buffers are passed
    through unchanged.  ``rows``/``ratio`` are ``c_int32`` and every export
    returns ``c_int32``.
    """

    def __init__(self, library_path: str) -> None:
        self.library_path = library_path
        self.lib = ctypes.CDLL(library_path)
        self._configure()

    def _configure(self) -> None:
        lib = self.lib
        lib.ds41rt_v41_compressor_create.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_void_p)]
        lib.ds41rt_v41_compressor_create.restype = ctypes.c_int32
        lib.ds41rt_v41_compressor_project.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p]
        lib.ds41rt_v41_compressor_project.restype = ctypes.c_int32
        lib.ds41rt_v41_compressor_destroy.argtypes = [ctypes.c_void_p]
        lib.ds41rt_v41_compressor_destroy.restype = ctypes.c_int32

    def create(self, workspace: "DeviceBuffer") -> Tuple[int, int]:
        handle = ctypes.c_void_p()
        rc = int(self.lib.ds41rt_v41_compressor_create(
            ctypes.c_void_p(workspace.ptr),
            ctypes.c_uint64(workspace.nbytes),
            ctypes.byref(handle)))
        return rc, (handle.value or 0)

    def project(self, handle: int, input_buf: "DeviceBuffer",
                weight_buf: "DeviceBuffer", output_buf: "DeviceBuffer",
                rows: int, ratio: int, stream: int) -> int:
        return int(self.lib.ds41rt_v41_compressor_project(
            ctypes.c_void_p(handle),
            ctypes.c_void_p(input_buf.ptr),
            ctypes.c_void_p(weight_buf.ptr),
            ctypes.c_void_p(output_buf.ptr),
            ctypes.c_int32(rows),
            ctypes.c_int32(ratio),
            ctypes.c_void_p(stream)))

    def destroy(self, handle: int) -> int:
        return int(self.lib.ds41rt_v41_compressor_destroy(
            ctypes.c_void_p(handle)))


def load_native_binding(library_path: str, expected_sha256: str) -> CtypesNativeBinding:
    """Hash-check the library *before* handing it to the dynamic loader."""
    verify_native_library_sha256(library_path, expected_sha256)
    return CtypesNativeBinding(library_path)


# ---------------------------------------------------------------------------
# Fixture validation
# ---------------------------------------------------------------------------

@dataclass
class RoleFixture:
    role: str
    directory: str
    manifest_path: str
    input_path: str
    projected_path: str
    scores_path: str
    rows: int
    input_bytes: bytes
    projected_bytes: bytes
    scores_bytes: bytes
    manifest: Dict[str, Any]


@dataclass
class FixtureData:
    root: str
    reference: RoleFixture
    candidate: RoleFixture
    wkv_path: str
    wgate_path: str
    wkv_bytes: bytes
    wgate_bytes: bytes
    real_input_row: bytes
    checks: Dict[str, Any] = field(default_factory=dict)


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise FixtureError(message)


def _check_buffer(manifest: Dict[str, Any], name: str, expected_file: str,
                  expected_dtype: str, rows: int, expected_bytes: int,
                  label: str) -> Dict[str, Any]:
    buffers = manifest.get("buffers")
    _expect(isinstance(buffers, dict), f"{label}: manifest has no buffers map")
    buf = buffers.get(name)
    _expect(isinstance(buf, dict), f"{label}: missing buffer {name!r}")
    _expect(buf.get("file") == expected_file,
            f"{label}: {name} file {buf.get('file')!r} != {expected_file!r}")
    _expect(buf.get("dtype") == expected_dtype,
            f"{label}: {name} dtype {buf.get('dtype')!r} != {expected_dtype!r}")
    _expect(buf.get("shape") == [rows, 5120 if name == "input" else 512],
            f"{label}: {name} shape {buf.get('shape')!r} != [{rows}, ...]")
    _expect(buf.get("bytes") == expected_bytes,
            f"{label}: {name} bytes {buf.get('bytes')!r} != {expected_bytes}")
    return buf


def _load_role(root: str, directory: str, expected_rows: int,
               label: str) -> RoleFixture:
    manifest_path = confined_path(root, directory, MANIFEST_NAME)
    require_regular_file(manifest_path, f"{label} manifest")
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    _expect(manifest.get("schema") == 1, f"{label}: unexpected manifest schema")
    _expect(manifest.get("kind") == "compressor-inputs",
            f"{label}: kind is not compressor-inputs")
    _expect(manifest.get("layer") == LAYER, f"{label}: layer != {LAYER}")
    _expect(manifest.get("ratio") == RATIO, f"{label}: ratio != {RATIO}")
    _expect(manifest.get("rows") == expected_rows,
            f"{label}: rows {manifest.get('rows')!r} != {expected_rows}")

    input_bytes = expected_rows * INPUT_ROW_BYTES
    output_bytes = expected_rows * OUTPUT_ROW_BYTES
    _check_buffer(manifest, "input", INPUT_FILE, "bfloat16",
                  expected_rows, input_bytes, label)
    _check_buffer(manifest, "projected", PROJECTED_FILE, "float32",
                  expected_rows, output_bytes, label)
    _check_buffer(manifest, "scores", SCORES_FILE, "float32",
                  expected_rows, output_bytes, label)

    input_path = confined_path(root, directory, INPUT_FILE)
    projected_path = confined_path(root, directory, PROJECTED_FILE)
    scores_path = confined_path(root, directory, SCORES_FILE)
    for path, what, size in (
            (input_path, "input", input_bytes),
            (projected_path, "projected", output_bytes),
            (scores_path, "scores", output_bytes)):
        require_regular_file(path, f"{label} {what}")
        actual = os.path.getsize(path)
        _expect(actual == size,
                f"{label} {what}: size {actual} != expected {size}")

    return RoleFixture(
        role=label, directory=directory, manifest_path=manifest_path,
        input_path=input_path, projected_path=projected_path,
        scores_path=scores_path, rows=expected_rows,
        input_bytes=open(input_path, "rb").read(),
        projected_bytes=open(projected_path, "rb").read(),
        scores_bytes=open(scores_path, "rb").read(),
        manifest=manifest)


def validate_fixture(activations_root: str) -> FixtureData:
    """Validate the frozen fixture and reject anything that differs."""
    if not os.path.isdir(activations_root):
        raise FixtureError(f"activations root is not a directory: {activations_root}")

    reference = _load_role(activations_root, REFERENCE_DIR, 1, "reference")
    candidate = _load_role(activations_root, CANDIDATE_DIR, 2, "candidate")

    # The manifest's remote weights directory is deliberately never followed.
    manifest_weights_dir = (reference.manifest.get("weights") or {}).get("directory")

    wkv_path = confined_path(activations_root, WEIGHTS_DIR, WKV_FILE)
    wgate_path = confined_path(activations_root, WEIGHTS_DIR, WGATE_FILE)
    for path, what in ((wkv_path, "wkv weight"), (wgate_path, "wgate weight")):
        require_regular_file(path, what)
        actual = os.path.getsize(path)
        _expect(actual == WEIGHT_BYTES,
                f"{what}: size {actual} != expected {WEIGHT_BYTES}")

    checks: Dict[str, Any] = {"hashes": {}}

    def record(label: str, actual: str, expected: str) -> None:
        verify_expected_hash(actual, expected, label)
        checks["hashes"][label] = {"expected": expected, "actual": actual,
                                   "ok": True}

    # Reference input / projected / scores are whole-file identities.
    record("reference_input", sha256_bytes(reference.input_bytes),
           FREEZE["reference_input_sha256"])
    record("reference_projected", sha256_bytes(reference.projected_bytes),
           FREEZE["reference_projected_sha256"])
    record("reference_scores", sha256_bytes(reference.scores_bytes),
           FREEZE["reference_scores_sha256"])

    # Weights are addressed explicitly under activations/projection-weights.
    wkv_bytes = open(wkv_path, "rb").read()
    wgate_bytes = open(wgate_path, "rb").read()
    record("wkv_weight", sha256_bytes(wkv_bytes), FREEZE["wkv_weight_sha256"])
    record("wgate_weight", sha256_bytes(wgate_bytes), FREEZE["wgate_weight_sha256"])

    # Candidate projected/scores must match per-row frozen hashes.
    for role, fixture, freeze_key in (
            ("candidate_projected", candidate.projected_bytes,
             "candidate_projected_row_sha256"),
            ("candidate_scores", candidate.scores_bytes,
             "candidate_scores_row_sha256")):
        rows = split_rows(fixture, OUTPUT_ROW_BYTES)
        for index, row in enumerate(rows):
            record(f"{role}_row{index}", sha256_bytes(row), FREEZE[freeze_key])

    # Every captured current input row (reference + candidate) must be equal.
    all_input_rows = split_rows(reference.input_bytes, INPUT_ROW_BYTES) + \
        split_rows(candidate.input_bytes, INPUT_ROW_BYTES)
    if not input_rows_all_equal(all_input_rows, FREEZE["reference_input_sha256"]):
        raise FixtureError(
            "captured current input rows are not all identical: different fixture")
    checks["captured_current_input_rows_all_equal"] = True
    checks["captured_input_row_count"] = len(all_input_rows)
    checks["manifest_weights_directory_ignored"] = manifest_weights_dir

    return FixtureData(
        root=os.path.realpath(activations_root),
        reference=reference, candidate=candidate,
        wkv_path=wkv_path, wgate_path=wgate_path,
        wkv_bytes=wkv_bytes, wgate_bytes=wgate_bytes,
        real_input_row=reference.input_bytes[:INPUT_ROW_BYTES],
        checks=checks)


# ---------------------------------------------------------------------------
# Case planning
# ---------------------------------------------------------------------------

def _projection_specs(wkv_sha: str, wgate_sha: str,
                      projected_expected: Optional[List[str]],
                      scores_expected: Optional[List[str]]) -> List[Dict[str, Any]]:
    return [
        {
            "role": "wkv",
            "weight_file": WKV_FILE,
            "weight_sha256": wkv_sha,
            "output_role": "projected",
            "expected_row_sha256": projected_expected,
            "oracle": "frozen_fixture_hashes" if projected_expected else None,
        },
        {
            "role": "wgate",
            "weight_file": WGATE_FILE,
            "weight_sha256": wgate_sha,
            "output_role": "scores",
            "expected_row_sha256": scores_expected,
            "oracle": "frozen_fixture_hashes" if scores_expected else None,
        },
    ]


def _baseline_cases(fixture: FixtureData) -> List[Dict[str, Any]]:
    wkv_sha = fixture.checks["hashes"]["wkv_weight"]["actual"]
    wgate_sha = fixture.checks["hashes"]["wgate_weight"]["actual"]
    return [
        {
            "id": "M1",
            "kind": "baseline",
            "source": "reference",
            "description": "reference lane0-batch20-Full-1rows",
            "rows": 1,
            "target_slot": 0,
            "target_slots": [0],
            "input_file": INPUT_FILE,
            "input_dir": REFERENCE_DIR,
            "input_sha256": sha256_bytes(fixture.reference.input_bytes),
            "input_row_sha256":
                [sha256_bytes(r) for r in
                 split_rows(fixture.reference.input_bytes, INPUT_ROW_BYTES)],
            "geometry": "direct",
            "variant": "baseline",
            "neighbor_slot": None,
            "neighbor_mutation": None,
            "provenance": {
                "input": f"{REFERENCE_DIR}/{INPUT_FILE}",
                "note": "exact recorded reference wave, rows=1",
            },
            "projections": _projection_specs(
                wkv_sha, wgate_sha,
                projected_expected=[FREEZE["reference_projected_sha256"]],
                scores_expected=[FREEZE["reference_scores_sha256"]]),
        },
        {
            "id": "M2",
            "kind": "baseline",
            "source": "candidate",
            "description": "candidate lane0-batch81-Full-2rows",
            "rows": 2,
            "target_slot": 0,
            "target_slots": [0, 1],
            "input_file": INPUT_FILE,
            "input_dir": CANDIDATE_DIR,
            "input_sha256": sha256_bytes(fixture.candidate.input_bytes),
            "input_row_sha256":
                [sha256_bytes(r) for r in
                 split_rows(fixture.candidate.input_bytes, INPUT_ROW_BYTES)],
            "geometry": "direct",
            "variant": "baseline",
            "neighbor_slot": None,
            "neighbor_mutation": None,
            "provenance": {
                "input": f"{CANDIDATE_DIR}/{INPUT_FILE}",
                "note": "exact recorded candidate wave, rows=2",
            },
            "projections": _projection_specs(
                wkv_sha, wgate_sha,
                projected_expected=[FREEZE["candidate_projected_row_sha256"]] * 2,
                scores_expected=[FREEZE["candidate_scores_row_sha256"]] * 2),
        },
    ]


def _cf_case(rows: int, target_slot: int, variant: str,
             fixture: FixtureData) -> Dict[str, Any]:
    """One CF shape probe with a single real row at ``target_slot``.

    All other rows are zero.  For ``flip-neighbor`` the real row is also copied
    to ``(target_slot + 1) % rows`` and byte 10238 is XORed with 1 there; the
    target row itself is never mutated.
    """
    neighbor_slot = (target_slot + 1) % rows
    if variant == "zero-neighbor":
        mutation = None
        neighbor = None
    else:
        mutation = {
            "byte_offset": CF_NEIGHBOR_XOR_BYTE_OFFSET,
            "xor_mask": CF_NEIGHBOR_XOR_MASK,
            "note": "lowest bit of the last BF16 word of the real input row",
        }
        neighbor = neighbor_slot
    case_id = f"CF-N{rows}-slot{target_slot}-{variant}"
    return {
        "id": case_id,
        "kind": "cf",
        "source": "derived",
        "description":
            f"shape probe rows={rows} target_slot={target_slot} "
            f"variant={variant}; not a baseline",
        "rows": rows,
        "target_slot": target_slot,
        "target_slots": [target_slot],
        "variant": variant,
        "neighbor_slot": neighbor,
        "neighbor_mutation": mutation,
        "fake_request": False,
        "input_file": None,
        "input_dir": None,
        "input_sha256": None,       # filled after materialization
        "input_row_sha256": None,
        "geometry": "one-real-row-plus-padding",
        "provenance": {
            "real_row_source": f"{REFERENCE_DIR}/{INPUT_FILE} row 0",
            "real_slot": target_slot,
            "target_slot": target_slot,
            "neighbor_slot": neighbor,
            "neighbor_mutation": mutation,
            "neighbor_is_real_request": False,
            "note": ("one real row at the target slot; all other rows are zero"
                     if variant == "zero-neighbor" else
                     "one real row at the target slot plus a mutated copy at "
                     f"slot {neighbor}; the target row is never mutated"),
            "cf_without_oracle": True,
        },
        "projections": _projection_specs(
            fixture.checks["hashes"]["wkv_weight"]["actual"],
            fixture.checks["hashes"]["wgate_weight"]["actual"],
            projected_expected=None, scores_expected=None),
    }


def _cf_cases(fixture: FixtureData) -> List[Dict[str, Any]]:
    cases = []
    for rows in sorted(CF_TARGET_SLOTS):
        for target_slot in CF_TARGET_SLOTS[rows]:
            for variant in ("zero-neighbor", "flip-neighbor"):
                cases.append(_cf_case(rows, target_slot, variant, fixture))
    return cases


def materialize_cf_input(case: Dict[str, Any], real_row: bytes) -> bytes:
    """Build the padded CF input for a case; never mutate the real source row."""
    rows = case["rows"]
    buffer = bytearray(case_input_nbytes(rows))
    target = case["target_slot"]
    buffer[target * INPUT_ROW_BYTES:(target + 1) * INPUT_ROW_BYTES] = real_row
    if case["variant"] == "flip-neighbor":
        neighbor = bytearray(real_row)
        neighbor[CF_NEIGHBOR_XOR_BYTE_OFFSET] ^= CF_NEIGHBOR_XOR_MASK
        slot = case["neighbor_slot"]
        buffer[slot * INPUT_ROW_BYTES:(slot + 1) * INPUT_ROW_BYTES] = bytes(neighbor)
    return bytes(buffer)


@dataclass
class BuiltPlan:
    plan: Dict[str, Any]
    case_inputs: Dict[str, bytes]
    weights: Dict[str, str]
    fixture: FixtureData


def build_plan(activations_root: str, *, native_library: Optional[str] = None,
               native_sha256: Optional[str] = None) -> BuiltPlan:
    fixture = validate_fixture(activations_root)
    native_provided = validate_native_sha256_format(native_sha256) \
        if native_sha256 else None

    cases = _baseline_cases(fixture) + _cf_cases(fixture)

    case_inputs: Dict[str, bytes] = {}
    for case in cases:
        if case["kind"] == "baseline":
            data = (fixture.reference.input_bytes if case["source"] == "reference"
                    else fixture.candidate.input_bytes)
        else:
            data = materialize_cf_input(case, fixture.real_input_row)
        # CF case metadata is completed only after materialization so the plan
        # records the exact bytes that would be projected.
        case["input_sha256"] = sha256_bytes(data)
        case["input_row_sha256"] = [sha256_bytes(r) for r in
                                    split_rows(data, INPUT_ROW_BYTES)]
        case_inputs[case["id"]] = data

    plan: Dict[str, Any] = {
        "schema": 1,
        "kind": "projection-replay-plan",
        "generated_sydney": sydney_now_iso(),
        "mode": "plan",
        "activations_root": fixture.root,
        "native_library": native_library,
        "native_sha256_provided": native_provided,
        "native_sha256_frozen": NATIVE_SHA256_FROZEN,
        "native_sha256_matches_frozen":
            (native_provided == NATIVE_SHA256_FROZEN) if native_provided else None,
        "native_file_hash_deferred_to_execute": native_provided is not None,
        "layer": LAYER,
        "ratio": RATIO,
        "byte_extents": {
            "input_row_bytes": INPUT_ROW_BYTES,
            "output_row_bytes": OUTPUT_ROW_BYTES,
            "weight_bytes": WEIGHT_BYTES,
            "workspace_bytes": WORKSPACE_BYTES,
            "workspace_alignment": WORKSPACE_ALIGN,
        },
        "fixture": {
            "checks": fixture.checks,
            "reference": {
                "directory": REFERENCE_DIR,
                "manifest": MANIFEST_NAME,
                "rows": fixture.reference.rows,
                "input_sha256": sha256_bytes(fixture.reference.input_bytes),
                "projected_sha256": sha256_bytes(fixture.reference.projected_bytes),
                "scores_sha256": sha256_bytes(fixture.reference.scores_bytes),
            },
            "candidate": {
                "directory": CANDIDATE_DIR,
                "manifest": MANIFEST_NAME,
                "rows": fixture.candidate.rows,
                "input_sha256": sha256_bytes(fixture.candidate.input_bytes),
                "projected_sha256": sha256_bytes(fixture.candidate.projected_bytes),
                "scores_sha256": sha256_bytes(fixture.candidate.scores_bytes),
            },
            "weights": {
                "directory": WEIGHTS_DIR,
                "wkv_file": WKV_FILE,
                "wgate_file": WGATE_FILE,
                "wkv_sha256": fixture.checks["hashes"]["wkv_weight"]["actual"],
                "wgate_sha256": fixture.checks["hashes"]["wgate_weight"]["actual"],
                "manifest_remote_directory_ignored":
                    fixture.checks.get("manifest_weights_directory_ignored"),
            },
        },
        "repeats": {"warmup": 1, "recorded": len(REPEAT_PHASES)},
        "cases": cases,
        "notes": [
            "Default mode is CPU validation/planning only; torch is imported "
            "solely in --execute.",
            "CF cases hold exactly one real input row at the target slot; all "
            "other rows are zero, except the flip-neighbor row.",
            "CF cases are shape probes with no oracle and never gate the "
            "component baseline pass.",
            "Each case/role has one warmup plus three recorded repeats; warmup "
            "downloads are retained separately and are never scored.",
            "weights.directory from the manifest is ignored; explicit files "
            "under activations/projection-weights are used.",
        ],
    }
    return BuiltPlan(plan=plan, case_inputs=case_inputs,
                     weights={"wkv": fixture.wkv_path, "wgate": fixture.wgate_path},
                     fixture=fixture)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def save_raw_output(output_dir: str, case_id: str, role: str, phase: str,
                    data: bytes) -> str:
    """Persist one downloaded output; recorded repeats and warmup are split."""
    subdir = WARMUP_SUBDIR if phase == "warmup" else RECORDED_SUBDIR
    raw_dir = os.path.join(output_dir, subdir)
    os.makedirs(raw_dir, exist_ok=True)
    path = os.path.join(raw_dir, f"{case_id}__{role}__{phase}.fp32")
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def _baseline_case_pass(case: Dict[str, Any],
                        calls: List[Dict[str, Any]]) -> bool:
    """Numerical gate: every recorded baseline row must match its oracle."""
    specs = {p["role"]: p for p in case["projections"]}
    for role, spec in specs.items():
        recorded = [c for c in calls
                    if c.get("op") == "project" and c.get("role") == role
                    and c.get("phase") in REPEAT_PHASES]
        if len(recorded) != len(REPEAT_PHASES):
            return False
        expected = spec.get("expected_row_sha256")
        if expected is None:
            return False
        for call in recorded:
            if call.get("rc") != 0:
                return False
            if call.get("row_sha256") != expected:
                return False
    return True


def _download_outputs(case_id: str,
                      project_records: List[Tuple[str, Dict[str, Any],
                                                  "DeviceBuffer", int]],
                      output_dir: str, save_raw: bool
                      ) -> Tuple[List[Optional[bytes]], List[Optional[str]]]:
    """Download every successful output, then persist.  Raises on failure.

    Every device read happens before any file is written, so a download
    exception cannot leave a partial matrix behind.
    """
    raws: List[Optional[bytes]] = []
    for _phase, _spec, out, rc in project_records:
        raws.append(None if rc != 0 else out.tobytes())
    saved: List[Optional[str]] = [None] * len(raws)
    if save_raw:
        written: List[str] = []
        try:
            for index, (phase, spec, _out, rc) in enumerate(project_records):
                if raws[index] is None:
                    continue
                saved[index] = save_raw_output(output_dir, case_id,
                                               spec["role"], phase, raws[index])
                written.append(saved[index])
        except Exception:
            for path in written:
                try:
                    os.remove(path)
                except OSError:
                    pass
            raise
    return raws, saved


def _run_case(case: Dict[str, Any], input_bytes: bytes,
              weight_buffers: Dict[str, "DeviceBuffer"],
              binding: Any, stream: int, sync_fn: Any, torch_module: Any,
              output_dir: str, save_raw: bool) -> Dict[str, Any]:
    """Run one case with explicit lifecycle cleanup.

    Ordering is create -> project -> synchronize -> download -> destroy.  The
    synchronize and destroy steps run even when project or download raised.
    """
    case_id = case["id"]
    rows = case["rows"]

    # Every buffer is a real device buffer allocated through the supplied torch
    # module.  No CPU fallback and no injectable allocator exists.
    workspace = DeviceBuffer(torch_module, nbytes=WORKSPACE_BYTES,
                             alignment=WORKSPACE_ALIGN, label="workspace")
    input_buf = DeviceBuffer(torch_module, data=input_bytes, label="input")
    # Retain every buffer for this case until after synchronization/destroy.
    retained: List[Any] = [workspace, input_buf]
    retained.extend(weight_buffers.values())

    calls: List[Dict[str, Any]] = []
    lifecycle_errors: List[Dict[str, Any]] = []
    project_records: List[Tuple[str, Dict[str, Any], DeviceBuffer, int]] = []

    create_rc: Optional[int] = None
    handle = 0
    destroy_rc: Optional[int] = None
    project_rejected = False
    sync_ok = False

    try:
        try:
            create_rc, handle = binding.create(workspace)
        except Exception as exc:  # lifecycle error, reported not raised
            lifecycle_errors.append({"phase": "create", "error": repr(exc)})
            create_rc = None
            handle = 0
        if create_rc is not None:
            calls.append({
                "op": "create", "phase": "create", "role": None,
                "rc": create_rc,
                "status": "ok" if create_rc == 0 else "ERROR",
            })

        if create_rc == 0:
            # Projection calls.  A rejected project stops all dependent work
            # for this case: no later project, no download, no scoring.
            try:
                for phase in ALL_PHASES:
                    for spec in case["projections"]:
                        out = DeviceBuffer(
                            torch_module, nbytes=case_output_nbytes(rows),
                            alignment=1, label=spec["role"])
                        # Zero only a newly allocated output.
                        out.zero()
                        retained.append(out)
                        rc = binding.project(
                            handle, input_buf, weight_buffers[spec["role"]],
                            out, rows, RATIO, stream)
                        project_records.append((phase, spec, out, rc))
                        if rc != 0:
                            project_rejected = True
                            break
                    if project_rejected:
                        break
            except Exception as exc:
                lifecycle_errors.append({"phase": "project", "error": repr(exc)})
                project_rejected = True

            # Synchronize before any download or destroy, even on failure.
            try:
                sync_fn()
                sync_ok = True
            except Exception as exc:
                lifecycle_errors.append(
                    {"phase": "synchronize", "error": repr(exc)})
                sync_ok = False

            project_calls: List[Dict[str, Any]] = []
            for phase, spec, _out, rc in project_records:
                project_calls.append({
                    "op": "project", "phase": phase, "role": spec["role"],
                    "output_role": spec["output_role"], "rc": rc,
                    "status": "ok" if rc == 0 else "ERROR",
                    "rows": rows, "ratio": RATIO, "stream": stream,
                    "scored": phase in REPEAT_PHASES, "saved": None,
                })
            calls.extend(project_calls)

            # Download only after a successful sync and only when no project
            # was rejected.  A sync failure suppresses the download.
            if sync_ok and not project_rejected:
                try:
                    raws, saved = _download_outputs(
                        case_id, project_records, output_dir, save_raw)
                    for call, raw, path in zip(project_calls, raws, saved):
                        if raw is None:
                            continue
                        call["sha256"] = sha256_bytes(raw)
                        call["row_sha256"] = [
                            sha256_bytes(
                                raw[i * OUTPUT_ROW_BYTES:(i + 1) * OUTPUT_ROW_BYTES])
                            for i in range(rows)]
                        call["saved"] = path
                except Exception as exc:
                    lifecycle_errors.append(
                        {"phase": "download", "error": repr(exc)})
                    for call in project_calls:
                        call["saved"] = None
                        call.pop("sha256", None)
                        call.pop("row_sha256", None)
    finally:
        # Destroy runs only after a successful create, and only after
        # synchronization/download.  It is itself reported, never raised.
        # Buffers stay referenced (`retained`, `project_records`) until here.
        if create_rc == 0 and handle:
            try:
                destroy_rc = binding.destroy(handle)
            except Exception as exc:
                lifecycle_errors.append(
                    {"phase": "destroy", "error": repr(exc)})
                destroy_rc = None
            calls.append({
                "op": "destroy", "phase": "destroy", "role": None,
                "rc": destroy_rc,
                "status": "ok" if destroy_rc == 0 else "ERROR",
            })

    lifecycle_success = (
        create_rc == 0
        and not project_rejected
        and sync_ok
        and not lifecycle_errors
        and destroy_rc == 0
    )
    all_calls_ok = lifecycle_success
    # Retain downloaded evidence even if destroy fails. Lifecycle failure keeps
    # the case unqualified; it must not erase the result that needs inspection.

    result: Dict[str, Any] = {
        "case": case_id,
        "kind": case["kind"],
        "rows": rows,
        "target_slot": case["target_slot"],
        "variant": case.get("variant"),
        "calls": calls,
        "input_sha256": sha256_bytes(input_bytes),
        "project_rejected": project_rejected,
        "synchronized": sync_ok,
        "destroy_rc": destroy_rc,
        "lifecycle_errors": lifecycle_errors,
        "lifecycle_success": lifecycle_success,
        "all_calls_ok": all_calls_ok,
        "aborted": not lifecycle_success,
    }
    if case["kind"] == "baseline":
        numerical = _baseline_case_pass(case, calls)
        result["numerical_baseline_pass"] = numerical
        # The baseline gate requires both a clean lifecycle and matching bytes.
        result["baseline_case_pass"] = bool(lifecycle_success and numerical)
        result["status"] = ("ok" if numerical else "mismatch") if lifecycle_success else "unscored"
        result["scored_rows"] = {"all_recorded_rows": True, "rows": rows}
    else:
        result["baseline_case_pass"] = None
        result["numerical_baseline_pass"] = None
        result["cf_without_oracle"] = True
        result["status"] = "recorded" if lifecycle_success else "unscored"
        # Only the explicit real target row is recorded for root analysis; the
        # neighbor/padding rows are still saved raw but are not scored.
        target_sha: Dict[str, Optional[str]] = {}
        for call in calls:
            if (call.get("op") == "project"
                    and call.get("phase") in REPEAT_PHASES
                    and call["role"] not in target_sha):
                row_shas = call.get("row_sha256")
                target_sha[call["role"]] = (
                    row_shas[case["target_slot"]] if row_shas else None)
        result["scored_target"] = {
            "slot": case["target_slot"],
            "rows_scored": 1,
            "projected_target_row_sha256": target_sha.get("wkv"),
            "scores_target_row_sha256": target_sha.get("wgate"),
        }
    return result


def summarize(plan: Dict[str, Any],
              case_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute the two *separate* gates; CF never contributes to baseline."""
    by_id = {r["case"]: r for r in case_results}
    all_ok = (len(case_results) == len(plan["cases"])
              and all(r.get("all_calls_ok", False) for r in case_results))
    baseline_ids = [c["id"] for c in plan["cases"] if c["kind"] == "baseline"]
    baseline_pass = bool(baseline_ids) and all(
        by_id.get(cid, {}).get("baseline_case_pass") is True for cid in baseline_ids)
    return {
        "all_calls_succeeded": all_ok,
        "component_baseline_pass": baseline_pass,
        "component_baseline_pass_scope": baseline_ids,
        "cf_cases_excluded_from_baseline_pass": [
            c["id"] for c in plan["cases"] if c["kind"] == "cf"],
        "no_overall_arithmetic_pass": True,
    }


def execute_plan(built: BuiltPlan, *, binding: Any, torch_module: Any,
                 output_dir: str, save_raw: bool = True) -> Dict[str, Any]:
    """Run every planned case through the existing native exports.

    The device buffers are always instantiated through ``torch_module`` by
    :class:`DeviceBuffer`; this entry point has no CPU path or allocator hook.
    """
    plan = built.plan
    stream = int(torch_module.cuda.current_stream().cuda_stream)
    sync_fn = torch_module.cuda.synchronize

    fixture_weight_bytes = {
        "wkv": built.fixture.wkv_bytes,
        "wgate": built.fixture.wgate_bytes,
    }
    weight_buffers = {
        role: DeviceBuffer(torch_module, data=fixture_weight_bytes[role],
                           label=role)
        for role in built.weights
    }

    case_results: List[Dict[str, Any]] = []
    exceptions: List[Dict[str, Any]] = []
    for case in plan["cases"]:
        try:
            result = _run_case(case, built.case_inputs[case["id"]],
                               weight_buffers, binding, stream, sync_fn,
                               torch_module, output_dir, save_raw)
        except Exception as exc:  # safety net; _run_case reports its own errors
            exceptions.append({"case": case["id"], "error": repr(exc)})
            result = {"case": case["id"], "kind": case["kind"],
                      "status": "exception", "calls": [],
                      "all_calls_ok": False,
                      "baseline_case_pass":
                          None if case["kind"] != "baseline" else False}
        case_results.append(result)

    summary = summarize(plan, case_results)
    return {
        "schema": 1,
        "kind": "projection-replay-results",
        "executed_sydney": sydney_now_iso(),
        "stream": stream,
        "cases": case_results,
        "summary": summary,
        "exceptions": exceptions,
        "native_library": plan.get("native_library"),
        "native_sha256_verified": plan.get("native_sha256_provided"),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Projection-only layer-2 compressor component replay.")
    parser.add_argument("--activations", required=True,
                        help="activation fixture root")
    parser.add_argument("--native-library", required=True,
                        help="path to the native shared object")
    parser.add_argument("--native-sha256", required=True,
                        help="expected sha256 of the native shared object")
    parser.add_argument("--output", required=True,
                        help="new output directory (must not exist)")
    parser.add_argument("--execute", action="store_true",
                        help="actually run the native component (default: plan only)")
    return parser.parse_args(argv)


def write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
        fh.write("\n")


def run(args: argparse.Namespace) -> int:
    if os.path.exists(args.output):
        raise ValidationError(f"output already exists (new only): {args.output}")
    native_sha = validate_native_sha256_format(args.native_sha256)

    built = build_plan(args.activations, native_library=args.native_library,
                       native_sha256=native_sha)
    built.plan["mode"] = "execute" if args.execute else "plan"

    os.makedirs(args.output, exist_ok=False)
    write_json(os.path.join(args.output, "plan.json"), built.plan)

    if not args.execute:
        print(f"plan written: {os.path.join(args.output, 'plan.json')}")
        print(f"cases: {len(built.plan['cases'])} "
              f"(baselines: {sum(1 for c in built.plan['cases'] if c['kind'] == 'baseline')}, "
              f"cf: {sum(1 for c in built.plan['cases'] if c['kind'] == 'cf')})")
        return 0

    # Execute: verify the native hash BEFORE loading, then lazily import torch.
    binding = load_native_binding(args.native_library, native_sha)
    import importlib
    torch_module = importlib.import_module("torch")
    results = execute_plan(built, binding=binding, torch_module=torch_module,
                           output_dir=args.output, save_raw=True)
    write_json(os.path.join(args.output, "results.json"), results)
    print(f"results written: {os.path.join(args.output, 'results.json')}")
    print(f"summary: {json.dumps(results['summary'], sort_keys=True)}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = parse_args(argv)
        return run(args)
    except ProjectionReplayError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
