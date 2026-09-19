"""CPU contract tests for the native compressor/kv FFI bindings.

The exact ctypes argument table is checked against the pinned native headers
(18cd6a50, vendored byte-identically in this tree), and the launch helpers are
driven with fake tensors so the descriptor/stream argument slots cannot drift.
No torch, no CUDA, no library load.
"""

from __future__ import annotations

import ctypes as C
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compressor_replay import ffi  # noqa: E402
from compressor_replay import runner  # noqa: E402

import compressor_replay_fakes as fakes  # noqa: E402

NATIVE_INCLUDE = SCRIPT_DIR.parent / "native" / "include"
COMPRESSOR_HEADER = NATIVE_INCLUDE / "ds41rt_v41_compressor.h"
KV_HEADER = NATIVE_INCLUDE / "ds41rt_v41_kv.h"

STREAM = fakes.STREAM


class _Session:
    def __init__(self):
        self.handle = fakes.HANDLE
        self.stream = STREAM
        self.bindings = fakes.FakeBindings(fakes.FakeTorch())


def _header_params(path, name):
    text = re.sub(r"//[^\n]*", "", path.read_text())
    match = re.search(r"\b" + name + r"\s*\((.*?)\)\s*;", text, re.S)
    assert match, f"missing native prototype: {name}"
    params, current, depth = [], "", 0
    for char in match.group(1):
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        if char == "," and depth == 0:
            params.append(current.strip())
            current = ""
        else:
            current += char
    if current.strip():
        params.append(current.strip())
    return params


def test_ctypes_signatures_match_native_headers():
    create = _header_params(COMPRESSOR_HEADER, "ds41rt_v41_compressor_create")
    destroy = _header_params(COMPRESSOR_HEADER, "ds41rt_v41_compressor_destroy")
    project = _header_params(COMPRESSOR_HEADER, "ds41rt_v41_compressor_project")
    pool = _header_params(COMPRESSOR_HEADER, "ds41rt_v41_compressor_pool")
    pack = _header_params(KV_HEADER, "ds41rt_v41_compressed_kv_pack")

    assert len(create) == len(ffi.CREATE_ARGTYPES) == 3
    assert len(destroy) == len(ffi.DESTROY_ARGTYPES) == 1
    assert len(project) == len(ffi.PROJECT_ARGTYPES) == 7
    assert len(pool) == len(ffi.POOL_ARGTYPES) == 10
    assert len(pack) == len(ffi.PACK_ARGTYPES) == 6

    # The pool's predecessor descriptors are argument 5 (index 4) and the
    # stream is the final argument; these are the slots the old sparse-replay
    # defect crossed.
    assert "predecessors" in pool[4] and "stream" not in pool[4]
    assert "norm_weight" in pool[5]
    assert "output" in pool[6]
    assert "stream" in pool[9]
    assert ffi.POOL_ARGTYPES[4] == C.c_void_p
    assert ffi.POOL_ARGTYPES[9] == C.c_void_p
    assert ffi.PROJECT_ARGTYPES[0] == C.c_void_p
    assert ffi.PACK_ARGTYPES[4] == C.c_int32


def test_launch_project_exact_argument_order():
    session = _Session()
    torch = session.bindings.torch
    input_t = torch.empty(4)
    weight_t = torch.empty(4)
    output_t = torch.empty(4)
    rc = runner.launch_project(
        session, input_t=input_t, weight_t=weight_t, output_t=output_t,
        rows=2, ratio=2, stream=STREAM,
    )
    assert rc == 0
    name, args = session.bindings.calls[0]
    assert name == "project"
    assert args == [fakes.HANDLE, input_t.data_ptr(), weight_t.data_ptr(),
                    output_t.data_ptr(), 2, 2, STREAM]
    assert args[6] == STREAM
    assert args[1] != STREAM


def test_launch_pool_keeps_predecessors_and_stream_in_their_slots():
    session = _Session()
    torch = session.bindings.torch
    kv = torch.empty(4)
    scores = torch.empty(4)
    pending_kv = torch.empty(4)
    pending_scores = torch.empty(4)
    predecessors = torch.empty(4)
    norm = torch.empty(4)
    output = torch.empty(4)
    rc = runner.launch_pool(
        session, kv_t=kv, scores_t=scores, pending_kv_t=pending_kv,
        pending_scores_t=pending_scores, predecessors_t=predecessors,
        norm_t=norm, output_t=output, rows=2, slots=4, stream=STREAM,
    )
    assert rc == 0
    name, args = session.bindings.calls[0]
    assert name == "pool"
    assert args == [kv.data_ptr(), scores.data_ptr(), pending_kv.data_ptr(),
                    pending_scores.data_ptr(), predecessors.data_ptr(),
                    norm.data_ptr(), output.data_ptr(), 2, 4, STREAM]
    assert args[4] == predecessors.data_ptr()
    assert args[4] != STREAM
    assert args[9] == STREAM


def test_launch_pack_passes_null_frequencies():
    session = _Session()
    torch = session.bindings.torch
    input_t = torch.empty(4)
    values = torch.empty(4)
    scales = torch.empty(4)
    rc = runner.launch_pack(
        session, input_t=input_t, frequencies_t=None, values_t=values,
        scales_t=scales, rows=2, stream=STREAM,
    )
    assert rc == 0
    name, args = session.bindings.calls[0]
    assert name == "pack"
    assert args == [input_t.data_ptr(), None, values.data_ptr(),
                    scales.data_ptr(), 2, STREAM]


def test_fake_bindings_role_calls_record_native_rc():
    torch = fakes.FakeTorch()
    bindings = fakes.FakeBindings(torch, rcs=[7])
    assert bindings.pool(1, 2, 3) == 7
    assert bindings.pack(1, 2, 3) == 0  # exhausted queue falls back to 0


def test_workspace_pointer_alignment_check():
    ffi.check_workspace_pointer(0x100000)
    for bad in (0, 0x101):
        try:
            ffi.check_workspace_pointer(bad)
        except ffi.FFIError:
            continue
        raise AssertionError(f"pointer {bad:#x} should have been rejected")
