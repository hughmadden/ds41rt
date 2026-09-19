"""Explicit ctypes bindings for the pinned native compressor/kv exports.

Signatures are mirrored from ``native/include/ds41rt_v41_compressor.h`` and
``native/include/ds41rt_v41_kv.h`` (pinned tree
``ds41rt-native-wob-m16-split2`` @ 18cd6a50; the repo-vendored copies are
byte-identical).  Every pointer parameter is a ``c_void_p`` carrying a raw
device address obtained from ``tensor.data_ptr()`` — the harness uploads
operands as uint8 and only *views* dtypes, so no typed ctypes pointer ever
aliases device memory.

Prior FFI pitfall (scripts/sparse_replay/runner.py): a metadata/host buffer
and the CUDA stream are both opaque pointers; passing the stream where an
operand buffer belongs is rejected by the native span checks.  The pool
signature here takes *six* operand pointers before rows/slots/stream; the
argument-role table below and the CPU stub tests pin each slot to its role.
"""

from __future__ import annotations

import ctypes as C
import hashlib
from dataclasses import dataclass
from pathlib import Path

# Header pin (overridable for tests); the repo vendors byte-identical copies.
PINNED_INCLUDE_DIR = Path(
    "/home/turq/dev/ds41rt-native-wob-m16-split2/native/include"
)
PINNED_TREE_COMMIT = "18cd6a50748cf103e40f9d18a810fe261bae3de1"

WORKSPACE_BYTES = 4 * 1024 * 1024
WORKSPACE_ALIGNMENT = 256

# ds41rt_v41_compressor_create(void* workspace, uint64_t bytes, void** handle)
CREATE_ARGTYPES = [C.c_void_p, C.c_uint64, C.POINTER(C.c_void_p)]
# ds41rt_v41_compressor_destroy(void* handle)
DESTROY_ARGTYPES = [C.c_void_p]
# ds41rt_v41_compressor_project(void* handle, const uint16_t* input,
#     const uint16_t* weight, void* output, int32_t rows, int32_t ratio,
#     void* stream)
PROJECT_ARGTYPES = (
    [C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_int32, C.c_int32,
     C.c_void_p]
)
# ds41rt_v41_compressor_pool(const float* kv, const float* scores,
#     const float* pending_kv, const float* pending_scores,
#     const uint64_t* predecessors, const uint16_t* norm_weight,
#     uint16_t* output, int32_t rows, int32_t slots, void* stream)
POOL_ARGTYPES = (
    [C.c_void_p] * 6 + [C.c_void_p, C.c_int32, C.c_int32, C.c_void_p]
)
# ds41rt_v41_compressed_kv_pack(const uint16_t* input, const float* frequencies,
#     uint8_t* values, uint8_t* scales, int32_t rows, void* stream)
PACK_ARGTYPES = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_int32,
                 C.c_void_p]
RESTYPE = C.c_int32


class FFIError(Exception):
    """Native library verification or binding failure."""


@dataclass(frozen=True)
class PoolRoles:
    """Argument roles for ds41rt_v41_compressor_pool, in positional order."""

    kv: int = 0            # FP32 current projected KV [rows, 512]
    scores: int = 1        # FP32 current gate scores [rows, 512]
    pending_kv: int = 2    # FP32 committed pending plane [slots, 512]
    pending_scores: int = 3
    predecessors: int = 4  # U64 device descriptors [rows] (NOT the stream)
    norm_weight: int = 5   # BF16 norm weight [512]
    output: int = 6        # BF16 pooled/normalized rows [rows, 512]
    rows: int = 7
    slots: int = 8
    stream: int = 9


@dataclass(frozen=True)
class ProjectRoles:
    handle: int = 0
    input: int = 1         # BF16 [rows, 5120]
    weight: int = 2        # BF16 [512, 5120]
    output: int = 3        # FP32 [rows, 512] at ratio 2
    rows: int = 4
    ratio: int = 5
    stream: int = 6


@dataclass(frozen=True)
class PackRoles:
    input: int = 0         # BF16 pooled rows [rows, 512]
    frequencies: int = 1   # FP32 [rows, 32, 2] or NULL
    values: int = 2        # packed FP4 out [rows, 256]
    scales: int = 3        # E4M3 scales out [rows, 32]
    rows: int = 4
    stream: int = 5


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass
class NativeBindings:
    """The five bound exports used by this harness (nothing else)."""

    create: object
    destroy: object
    project: object
    pack: object
    pool: object
    path: str = ""
    sha256: str = ""

    @classmethod
    def from_library(cls, lib, *, path: str = "", sha256: str = "") -> "NativeBindings":
        create = lib.ds41rt_v41_compressor_create
        create.argtypes = CREATE_ARGTYPES
        create.restype = RESTYPE
        destroy = lib.ds41rt_v41_compressor_destroy
        destroy.argtypes = DESTROY_ARGTYPES
        destroy.restype = RESTYPE
        project = lib.ds41rt_v41_compressor_project
        project.argtypes = PROJECT_ARGTYPES
        project.restype = RESTYPE
        pool = lib.ds41rt_v41_compressor_pool
        pool.argtypes = POOL_ARGTYPES
        pool.restype = RESTYPE
        pack = lib.ds41rt_v41_compressed_kv_pack
        pack.argtypes = PACK_ARGTYPES
        pack.restype = RESTYPE
        return cls(create=create, destroy=destroy, project=project,
                   pool=pool, pack=pack, path=path, sha256=sha256)


def verify_native_library(path: Path, expected_sha256: str) -> str:
    """Pin check: refuse to bind any library whose sha256 differs."""
    path = Path(path)
    if not path.is_file():
        raise FFIError(f"native library not found: {path}")
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise FFIError(
            f"native library sha256 mismatch for {path}: "
            f"expected {expected_sha256}, got {digest}"
        )
    return digest


def check_workspace_pointer(address: int) -> None:
    """The native create validates 4MiB/256-alignment itself; pre-checking on
    the host gives a clear error instead of an opaque cudaErrorInvalidValue."""
    if address <= 0 or address % WORKSPACE_ALIGNMENT != 0:
        raise FFIError(
            f"workspace pointer {address:#x} is not {WORKSPACE_ALIGNMENT}-aligned"
        )
