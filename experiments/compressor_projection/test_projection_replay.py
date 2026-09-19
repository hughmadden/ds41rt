#!/usr/bin/env python3
"""CPU stub tests for projection_replay.

No GPU, no network, no services and no real native load.  The production
DeviceBuffer is exercised through a FakeTorch that implements the exact torch
protocol (``empty``/``frombuffer``/``copy_``/``data_ptr``/``zero_``/``cpu``/
``numpy``/``tobytes``) with device/dtype assertions and a pointer-keyed
registry.  The fake native binding writes through that registry, so the
device-pointer plumbing is genuinely tested rather than simulated with
ctypes.  The real activation fixture is read-only and is never modified.
"""

from __future__ import annotations

import inspect
import ctypes
import json
import os
import runpy
import sys
from types import SimpleNamespace

import pytest

import projection_replay as pr


def test_real_ctypes_adapter_passes_exact_native_arguments(monkeypatch):
    calls = []

    class Export:
        def __init__(self, name):
            self.name = name

        def __call__(self, *args):
            if self.name == "create":
                ctypes.cast(args[2], ctypes.POINTER(ctypes.c_void_p))[0] = 0x12345678
                calls.append((self.name, args[0].value, args[1].value))
            else:
                calls.append((self.name, *(arg.value for arg in args)))
            return 0

    lib = SimpleNamespace(**{
        "ds41rt_v41_compressor_" + name: Export(name)
        for name in ["create", "project", "destroy"]})
    monkeypatch.setattr(pr.ctypes, "CDLL", lambda path: lib)
    binding = pr.CtypesNativeBinding("stub-only")
    workspace = SimpleNamespace(ptr=0x10000000, nbytes=4194304)
    rc, handle = binding.create(workspace)
    assert (rc, handle) == (0, 0x12345678)
    buffers = [SimpleNamespace(ptr=p) for p in [0x20000000, 0x30000000, 0x40000000]]
    assert binding.project(handle, *buffers, rows=16, ratio=2, stream=0x76543210) == 0
    assert binding.destroy(handle) == 0
    assert calls == [
        ("create", 0x10000000, 4194304),
        ("project", handle, 0x20000000, 0x30000000, 0x40000000, 16, 2, 0x76543210),
        ("destroy", handle)]
    assert lib.ds41rt_v41_compressor_project.argtypes == [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p]
    assert lib.ds41rt_v41_compressor_project.restype is ctypes.c_int32

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE_PATH = os.path.join(HERE, "projection_replay.py")
REAL_FIXTURE = ("/home/turq/.cache/afd-dsh-workers-20260919/"
                "flash-cap16-p2-compressor-diagnostic-01/activations")


# ---------------------------------------------------------------------------
# Fake torch: exact allocation/copy/data_ptr/cpu protocol
# ---------------------------------------------------------------------------

class FakeNumpyArray:
    def __init__(self, data: bytes) -> None:
        self._data = bytes(data)

    def tobytes(self) -> bytes:
        return self._data


class FakeCpuTensor:
    def __init__(self, data: bytes) -> None:
        self._data = bytes(data)

    def numpy(self) -> FakeNumpyArray:
        return FakeNumpyArray(self._data)


class FakeHostTensor:
    """Result of ``frombuffer``: host bytes shared with the source bytearray."""

    def __init__(self, buffer: bytearray) -> None:
        self._buffer = buffer

    def host_bytes(self) -> bytes:
        return bytes(self._buffer)


class FakeTensor:
    """Simulated device tensor; bytes live in FakeTorch.registry[ptr]."""

    def __init__(self, torch_module, ptr: int, nbytes: int) -> None:
        self._torch = torch_module
        self._ptr = int(ptr)
        self._nbytes = int(nbytes)

    def data_ptr(self) -> int:
        return self._ptr

    def copy_(self, source):
        data = source.host_bytes()
        assert len(data) == self._nbytes
        self._torch.registry[self._ptr] = bytearray(data)
        self._torch.events.append(("copy", self._ptr))
        return self

    def zero_(self):
        self._torch.registry[self._ptr] = bytearray(self._nbytes)
        self._torch.events.append(("zero", self._ptr))
        return self

    def cpu(self):
        self._torch.cpu_downloads.append(
            {"ptr": self._ptr, "sync_count": self._torch.cuda.sync_count})
        self._torch.events.append(("cpu", self._ptr))
        if self._torch.cpu_error is not None:
            raise self._torch.cpu_error
        return FakeCpuTensor(
            self._torch.registry.get(self._ptr, bytearray(self._nbytes)))


class FakeCuda:
    def __init__(self, owner, stream: int = 0x5EED) -> None:
        self._owner = owner
        self._stream = stream
        self.sync_count = 0
        self.sync_error = None

    def current_stream(self):
        return SimpleNamespace(cuda_stream=self._stream)

    def synchronize(self) -> None:
        self.sync_count += 1
        self._owner.events.append(("sync", self.sync_count))
        if self.sync_error is not None:
            raise self.sync_error


class FakeTorch:
    """Strict stand-in for torch that mirrors the raw-byte buffer protocol."""

    uint8 = "torch.uint8"

    def __init__(self, stream: int = 0x5EED) -> None:
        self.events = []
        self.cuda = FakeCuda(self, stream)
        self.registry = {}
        self.allocations = []
        self.cpu_downloads = []
        self.empty_calls = []
        self.frombuffer_calls = []
        self.cpu_error = None
        self._next_ptr = 0x10000000  # already 256-byte aligned

    def _alloc_ptr(self, nbytes: int) -> int:
        size = max(int(nbytes), 1)
        ptr = self._next_ptr
        self._next_ptr = (ptr + size + 255) & ~255
        return ptr

    def empty(self, nbytes, dtype=None, device=None):
        assert dtype == self.uint8, f"expected uint8, got {dtype!r}"
        assert device == "cuda", f"expected cuda, got {device!r}"
        ptr = self._alloc_ptr(nbytes)
        self.registry[ptr] = bytearray(int(nbytes))
        record = {"ptr": ptr, "nbytes": int(nbytes), "dtype": dtype,
                  "device": device}
        self.allocations.append(record)
        self.empty_calls.append((int(nbytes), dtype, device))
        self.events.append(("empty", ptr, int(nbytes)))
        return FakeTensor(self, ptr, nbytes)

    def frombuffer(self, buffer, dtype=None, count=-1, offset=0):
        assert dtype == self.uint8, f"expected uint8, got {dtype!r}"
        assert count == -1 and offset == 0
        self.frombuffer_calls.append(len(buffer))
        self.events.append(("frombuffer", len(buffer)))
        return FakeHostTensor(buffer)

    def peek(self, buf) -> bytes:
        return bytes(self.registry.get(buf.ptr, bytearray(buf.nbytes)))


# ---------------------------------------------------------------------------
# Fake native binding: writes through the device-pointer registry
# ---------------------------------------------------------------------------

class FakeNativeBinding:
    """Records every FFI-shaped call so tests can inspect roles and ordering."""

    def __init__(self, create_rc: int = 0, project_rcs=None,
                 destroy_rc: int = 0, torch_module=None, write_fill=None,
                 outputs_by_label=None, create_exc: bool = False,
                 destroy_exc: bool = False) -> None:
        self.create_rc = create_rc
        self.project_rcs = list(project_rcs or [])
        self.destroy_rc = destroy_rc
        self.torch = torch_module
        self.write_fill = write_fill
        self.outputs_by_label = dict(outputs_by_label or {})
        self.create_exc = create_exc
        self.destroy_exc = destroy_exc
        self.calls = []
        self._handle = 0x1000

    def _event(self, *event) -> None:
        if self.torch is not None:
            self.torch.events.append(tuple(event))

    def create(self, workspace):
        self.calls.append({"op": "create", "workspace": workspace})
        self._event("native_create")
        if self.create_exc:
            raise RuntimeError("create boom")
        return self.create_rc, (self._handle if self.create_rc == 0 else 0)

    def project(self, handle, input_buf, weight_buf, output_buf,
                rows, ratio, stream):
        rc = self.project_rcs.pop(0) if self.project_rcs else 0
        entry = {
            "op": "project", "handle": handle,
            "input_buf": input_buf, "weight_buf": weight_buf,
            "output_buf": output_buf,
            "rows": rows, "ratio": ratio, "stream": stream, "rc": rc,
        }
        if self.torch is not None:
            entry["input_at_entry"] = self.torch.peek(input_buf)
            entry["output_at_entry"] = self.torch.peek(output_buf)
        self.calls.append(entry)
        self._event("native_project", getattr(weight_buf, "label", None))
        if rc == 0 and self.torch is not None:
            self._write_output(output_buf, weight_buf)
        return rc

    def _write_output(self, output_buf, weight_buf) -> None:
        label = getattr(weight_buf, "label", None)
        data = self.outputs_by_label.get(label)
        if data is None and self.write_fill is not None:
            data = self.write_fill * output_buf.nbytes
        if data is None:
            return
        payload = bytearray(output_buf.nbytes)
        payload[:min(len(data), output_buf.nbytes)] = \
            data[:output_buf.nbytes]
        # The write lands in device memory keyed by the device pointer.
        self.torch.registry[output_buf.ptr] = payload

    def destroy(self, handle):
        self.calls.append({"op": "destroy", "handle": handle})
        self._event("native_destroy")
        if self.destroy_exc:
            raise RuntimeError("destroy boom")
        return self.destroy_rc

    # convenience
    def projects(self):
        return [c for c in self.calls if c["op"] == "project"]

    def ops(self):
        return [c["op"] for c in self.calls]


class RaisingProjectBinding(FakeNativeBinding):
    def project(self, *args, **kwargs):
        raise RuntimeError("native boom")


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def built():
    return pr.build_plan(REAL_FIXTURE, native_library="/nonexistent/libds41rt.so",
                         native_sha256=pr.NATIVE_SHA256_FROZEN)


def _case(built, case_id):
    return next(c for c in built.plan["cases"] if c["id"] == case_id)


def _single_case_built(built, case_id):
    case = _case(built, case_id)
    plan = dict(built.plan)
    plan["cases"] = [case]
    return pr.BuiltPlan(plan=plan,
                        case_inputs={case_id: built.case_inputs[case_id]},
                        weights=built.weights, fixture=built.fixture)


def _rows(data, row_bytes=pr.INPUT_ROW_BYTES):
    return [data[i:i + row_bytes] for i in range(0, len(data), row_bytes)]


def _raw_files(output_dir):
    found = []
    for sub in (pr.RECORDED_SUBDIR, pr.WARMUP_SUBDIR):
        directory = os.path.join(str(output_dir), sub)
        if os.path.isdir(directory):
            found.extend(os.listdir(directory))
    return sorted(found)


def _timeline(fake_torch):
    return [event[0] for event in fake_torch.events]


def _allocation_ptrs(fake_torch):
    return {record["ptr"] for record in fake_torch.allocations}


EXPECTED_CF_IDS = {
    "CF-N2-slot0-zero-neighbor", "CF-N2-slot0-flip-neighbor",
    "CF-N2-slot1-zero-neighbor", "CF-N2-slot1-flip-neighbor",
    "CF-N16-slot0-zero-neighbor", "CF-N16-slot0-flip-neighbor",
    "CF-N16-slot1-zero-neighbor", "CF-N16-slot1-flip-neighbor",
    "CF-N16-slot7-zero-neighbor", "CF-N16-slot7-flip-neighbor",
    "CF-N16-slot15-zero-neighbor", "CF-N16-slot15-flip-neighbor",
}


# ---------------------------------------------------------------------------
# Planning: 14 cases, baselines unchanged, CF geometry
# ---------------------------------------------------------------------------

def test_build_plan_has_exactly_fourteen_cases(built):
    cases = built.plan["cases"]
    ids = [c["id"] for c in cases]
    assert len(cases) == 14
    assert len(ids) == len(set(ids))
    assert set(ids) == {"M1", "M2"} | EXPECTED_CF_IDS
    assert sum(1 for c in cases if c["kind"] == "baseline") == 2
    assert sum(1 for c in cases if c["kind"] == "cf") == 12


def test_cf_ids_encode_n_slot_and_variant(built):
    for case in built.plan["cases"]:
        if case["kind"] != "cf":
            continue
        case_id = case["id"]
        assert f"N{case['rows']}" in case_id
        assert f"slot{case['target_slot']}" in case_id
        assert case["variant"] in case_id


def test_build_plan_has_exact_baselines(built):
    cases = {c["id"]: c for c in built.plan["cases"]}
    assert cases["M1"]["rows"] == 1 and cases["M2"]["rows"] == 2
    assert cases["M1"]["kind"] == "baseline" and cases["M2"]["kind"] == "baseline"
    assert cases["M1"]["input_sha256"] == pr.FREEZE["reference_input_sha256"]
    assert cases["M1"]["input_row_sha256"] == [pr.FREEZE["reference_input_sha256"]]
    assert cases["M2"]["input_row_sha256"] == \
        [pr.FREEZE["reference_input_sha256"]] * 2
    wkv, wgate = cases["M1"]["projections"]
    assert (wkv["role"], wkv["output_role"]) == ("wkv", "projected")
    assert (wgate["role"], wgate["output_role"]) == ("wgate", "scores")
    assert wkv["weight_sha256"] == pr.FREEZE["wkv_weight_sha256"]
    assert wgate["weight_sha256"] == pr.FREEZE["wgate_weight_sha256"]
    assert wkv["expected_row_sha256"] == [pr.FREEZE["reference_projected_sha256"]]
    assert wgate["expected_row_sha256"] == [pr.FREEZE["reference_scores_sha256"]]
    assert cases["M2"]["projections"][0]["expected_row_sha256"] == \
        [pr.FREEZE["candidate_projected_row_sha256"]] * 2
    assert cases["M2"]["projections"][1]["expected_row_sha256"] == \
        [pr.FREEZE["candidate_scores_row_sha256"]] * 2
    assert len(built.case_inputs["M1"]) == 1 * pr.INPUT_ROW_BYTES
    assert len(built.case_inputs["M2"]) == 2 * pr.INPUT_ROW_BYTES


def test_cf_cases_geometry_and_no_oracle(built):
    expected_targets = {2: (0, 1), 16: (0, 1, 7, 15)}
    for rows, slots in expected_targets.items():
        for slot in slots:
            for variant in ("zero-neighbor", "flip-neighbor"):
                case = _case(built, f"CF-N{rows}-slot{slot}-{variant}")
                assert case["kind"] == "cf"
                assert case["rows"] == rows
                assert case["target_slot"] == slot
                assert case["target_slots"] == [slot]
                assert case["fake_request"] is False
                for spec in case["projections"]:
                    assert spec["expected_row_sha256"] is None
                    assert spec["oracle"] is None
                if variant == "zero-neighbor":
                    assert case["neighbor_slot"] is None
                    assert case["neighbor_mutation"] is None
                else:
                    assert case["neighbor_slot"] == (slot + 1) % rows
                    assert case["neighbor_mutation"] == {
                        "byte_offset": pr.CF_NEIGHBOR_XOR_BYTE_OFFSET,
                        "xor_mask": pr.CF_NEIGHBOR_XOR_MASK,
                        "note": ("lowest bit of the last BF16 word of the "
                                 "real input row"),
                    }
                assert case["provenance"]["cf_without_oracle"] is True
                assert case["provenance"]["real_slot"] == slot


def test_cf_cases_drop_legacy_real_slot_fields(built):
    """The old 'slot1 replaces real' bookkeeping must be gone."""
    for case in built.plan["cases"]:
        if case["kind"] != "cf":
            continue
        assert "real_slots" not in case
        assert "neighbor_replaces_real_slot" not in case
        assert "real_slots" not in case["provenance"]
    source = open(MODULE_PATH, "r", encoding="utf-8").read()
    assert "real_slots" not in source
    assert "replaces_real" not in source
    assert "replaces" not in source.lower()


def test_warmup_calls_are_marked_unscored(built, tmp_path):
    sub = _single_case_built(built, "M1")
    fake = FakeTorch()
    binding = FakeNativeBinding(torch_module=fake, write_fill=b"\x66")
    results = pr.execute_plan(sub, binding=binding, torch_module=fake,
                              output_dir=str(tmp_path), save_raw=True)
    for call in results["cases"][0]["calls"]:
        if call["op"] != "project":
            continue
        assert call["scored"] == (call["phase"] in pr.REPEAT_PHASES)
    warmup = os.listdir(os.path.join(str(tmp_path), pr.WARMUP_SUBDIR))
    assert len(warmup) == 2  # two roles, one unscored warmup each


def test_cf_in_plan_has_materialized_input_hashes(built):
    for case in built.plan["cases"]:
        if case["kind"] != "cf":
            continue
        data = built.case_inputs[case["id"]]
        assert len(data) == case["rows"] * pr.INPUT_ROW_BYTES
        assert pr.sha256_bytes(data) == case["input_sha256"]
        assert case["input_row_sha256"] == \
            [pr.sha256_bytes(r) for r in _rows(data)]


# ---------------------------------------------------------------------------
# CF padding / neighbor mutation invariants
# ---------------------------------------------------------------------------

def test_cf_exactly_one_real_row_at_target_slot(built):
    """Every CF bytearray has exactly one real row, at its target slot."""
    real = built.fixture.real_input_row
    zero = bytes(pr.INPUT_ROW_BYTES)
    for case in built.plan["cases"]:
        if case["kind"] != "cf":
            continue
        rows = _rows(built.case_inputs[case["id"]])
        assert rows[case["target_slot"]] == real, case["id"]
        if case["variant"] == "zero-neighbor":
            assert [i for i, r in enumerate(rows) if r != zero] == \
                [case["target_slot"]], case["id"]


def test_cf_flip_neighbor_exact_point_mutation(built):
    real = built.fixture.real_input_row
    zero = bytes(pr.INPUT_ROW_BYTES)
    for case in built.plan["cases"]:
        if case["kind"] != "cf":
            continue
        rows = _rows(built.case_inputs[case["id"]])
        target = case["target_slot"]
        # The scored target row is never mutated.
        assert rows[target] == real, case["id"]
        if case["variant"] == "zero-neighbor":
            continue
        neighbor = case["neighbor_slot"]
        mutated = rows[neighbor]
        assert len(mutated) == pr.INPUT_ROW_BYTES
        diff = [i for i in range(len(real)) if mutated[i] != real[i]]
        assert diff == [pr.CF_NEIGHBOR_XOR_BYTE_OFFSET], case["id"]
        assert mutated[pr.CF_NEIGHBOR_XOR_BYTE_OFFSET] == \
            real[pr.CF_NEIGHBOR_XOR_BYTE_OFFSET] ^ 1
        # Every other row (including all N16 padding) is untouched zero.
        for slot, row in enumerate(rows):
            if slot == target or slot == neighbor:
                continue
            assert row == zero, (case["id"], slot)


def test_cf_neighbor_is_target_plus_one_mod_n(built):
    for case in built.plan["cases"]:
        if case["kind"] != "cf":
            continue
        if case["variant"] == "flip-neighbor":
            assert case["neighbor_slot"] == \
                (case["target_slot"] + 1) % case["rows"]
        else:
            assert case["neighbor_slot"] is None


def test_cf_n16_placements_cover_requested_slots(built):
    for slot in (0, 1, 7, 15):
        case = _case(built, f"CF-N16-slot{slot}-zero-neighbor")
        rows = _rows(built.case_inputs[case["id"]])
        nonzero = [i for i, r in enumerate(rows)
                   if r != bytes(pr.INPUT_ROW_BYTES)]
        assert nonzero == [slot]


# ---------------------------------------------------------------------------
# Device buffer: CUDA allocation protocol and raw bytes
# ---------------------------------------------------------------------------

def test_role_buffer_byte_extents():
    assert pr.case_input_nbytes(1) == 10240
    assert pr.case_input_nbytes(2) == 20480
    assert pr.case_input_nbytes(16) == 16 * 10240
    assert pr.case_output_nbytes(1) == 2048
    assert pr.case_output_nbytes(2) == 4096
    fake = FakeTorch()
    ws = pr.DeviceBuffer(fake, nbytes=pr.WORKSPACE_BYTES,
                         alignment=pr.WORKSPACE_ALIGN)
    assert ws.nbytes == 4 * 1024 * 1024
    assert ws.ptr % pr.WORKSPACE_ALIGN == 0
    weight = pr.DeviceBuffer(fake, data=b"x" * pr.WEIGHT_BYTES)
    assert weight.nbytes == 5242880
    assert pr.CUDA_DEVICE == "cuda"


def test_device_buffer_uploads_raw_bytes_via_cuda_protocol():
    fake = FakeTorch()
    payload = bytes(range(256)) * 4
    buf = pr.DeviceBuffer(fake, data=payload)
    assert buf.nbytes == len(payload)
    # Empty/document (device=cuda, uint8), frombuffer, then copy_.
    assert fake.empty_calls[0][1] == fake.uint8
    assert fake.empty_calls[0][2] == "cuda"
    assert fake.frombuffer_calls == [len(payload)]
    assert [e[0] for e in fake.events][:2] == ["empty", "frombuffer"]
    assert fake.registry[buf.ptr] == payload
    assert buf.tobytes() == payload
    # Source lifetime: the host bytearray is still referenced.
    assert buf._host is not None and bytes(buf._host) == payload
    assert buf._source is not None


def test_zero_only_touches_the_output_buffer():
    fake = FakeTorch()
    out = pr.DeviceBuffer(fake, nbytes=64)
    fake.registry[out.ptr][:] = b"\xaa" * 64
    out.zero()
    assert fake.registry[out.ptr] == bytes(64)
    # The zero event is recorded but no download happened.
    assert ("zero", out.ptr) in fake.events
    assert fake.cpu_downloads == []


def test_raw_bf16_unusual_bit_patterns_preserved():
    """NaN/inf/subnormal/negative-zero BF16 words survive byte-exactly."""
    unit = bytes([
        0x00, 0x80,  # negative zero
        0xFF, 0x7F,  # positive NaN
        0x01, 0x00,  # positive subnormal
        0xC0, 0x7F,  # positive quiet NaN
        0x01, 0x80,  # negative subnormal
        0xFF, 0xFF,  # negative NaN
        0x34, 0x12,  # ordinary value
    ])
    pattern = (unit * ((pr.INPUT_ROW_BYTES // len(unit)) + 1))[
        :pr.INPUT_ROW_BYTES]
    fake = FakeTorch()
    buf = pr.DeviceBuffer(fake, data=pattern)
    assert fake.registry[buf.ptr] == pattern
    assert buf.tobytes() == pattern
    # No float interpretation happened: raw bytes in, raw bytes out.
    assert bytes(fake.registry[buf.ptr]) == pattern


def test_fake_torch_is_strict_about_device_and_dtype():
    fake = FakeTorch()
    with pytest.raises(AssertionError):
        fake.empty(4, dtype=fake.uint8, device="cpu")
    with pytest.raises(AssertionError):
        fake.empty(4, dtype="torch.float32", device="cuda")


def test_simulated_device_addresses_are_distinct_and_aligned():
    fake = FakeTorch()
    buffers = [pr.DeviceBuffer(fake, nbytes=32) for _ in range(5)]
    ptrs = [b.ptr for b in buffers]
    assert len(set(ptrs)) == len(ptrs)
    assert all(p % pr.WORKSPACE_ALIGN == 0 for p in ptrs)


def test_production_has_no_ctypes_buffer_simulation():
    source = open(MODULE_PATH, "r", encoding="utf-8").read()
    assert "ctypes.memmove" not in source
    assert "ctypes.string_at" not in source
    assert not hasattr(pr, "NativeBuffer")
    assert "DeviceBuffer" in source
    assert 'device="cuda"' in source or "CUDA_DEVICE" in source


def test_execute_plan_has_no_injectable_allocator():
    params = inspect.signature(pr.execute_plan).parameters
    assert "torch_module" in params
    assert "buffer_factory" not in params
    assert "allocator" not in params
    run_params = inspect.signature(pr._run_case).parameters
    assert "torch_module" in run_params


# ---------------------------------------------------------------------------
# Execution: device pointers, zeroing, sync/download ordering
# ---------------------------------------------------------------------------

def test_execute_allocates_workspace_weights_inputs_outputs_on_device(built, tmp_path):
    sub = _single_case_built(built, "M1")
    fake = FakeTorch(stream=0xABCD)
    binding = FakeNativeBinding(torch_module=fake)
    result = pr.execute_plan(sub, binding=binding, torch_module=fake,
                             output_dir=str(tmp_path), save_raw=False)

    records = fake.allocations
    assert records and all(r["device"] == "cuda" for r in records)
    assert all(r["dtype"] == fake.uint8 for r in records)

    workspaces = [r for r in records if r["nbytes"] == pr.WORKSPACE_BYTES]
    weights = [r for r in records if r["nbytes"] == pr.WEIGHT_BYTES]
    inputs = [r for r in records
              if r["nbytes"] == pr.case_input_nbytes(1)]
    outputs = [r for r in records
               if r["nbytes"] == pr.case_output_nbytes(1)]
    assert len(workspaces) == 1
    assert workspaces[0]["ptr"] % pr.WORKSPACE_ALIGN == 0
    assert len(weights) == 2
    assert len(inputs) == 1
    assert len(outputs) == len(pr.ALL_PHASES) * 2

    ptrs = _allocation_ptrs(fake)
    for call in binding.projects():
        assert call["input_buf"].ptr in ptrs
        assert call["weight_buf"].ptr in ptrs
        assert call["output_buf"].ptr in ptrs
        assert len({call["input_buf"].ptr, call["weight_buf"].ptr,
                    call["output_buf"].ptr}) == 3
    assert result["cases"][0]["all_calls_ok"] is True


def test_outputs_zeroed_and_inputs_weights_untouched(built, tmp_path):
    sub = _single_case_built(built, "M1")
    fake = FakeTorch()
    binding = FakeNativeBinding(torch_module=fake)
    pr.execute_plan(sub, binding=binding, torch_module=fake,
                    output_dir=str(tmp_path), save_raw=False)
    input_bytes = built.case_inputs["M1"]
    for call in binding.projects():
        # Outputs are freshly zeroed before every native call.
        assert call["output_at_entry"] == bytes(pr.case_output_nbytes(1))
        # Inputs and weights are never zeroed and arrive byte-exactly.
        assert call["input_at_entry"] == input_bytes
        assert call["input_buf"].tobytes() == input_bytes
    wkv = [c for c in binding.projects()
           if c["weight_buf"].label == "wkv"]
    wgate = [c for c in binding.projects()
             if c["weight_buf"].label == "wgate"]
    assert len(wkv) == len(pr.ALL_PHASES)
    assert len(wgate) == len(pr.ALL_PHASES)
    for call in wkv:
        assert fake.peek(call["weight_buf"]) == built.fixture.wkv_bytes
    for call in wgate:
        assert fake.peek(call["weight_buf"]) == built.fixture.wgate_bytes


def test_downloads_occur_only_after_sync(built, tmp_path):
    sub = _single_case_built(built, "M1")
    fake = FakeTorch()
    binding = FakeNativeBinding(torch_module=fake, write_fill=b"\x11")
    pr.execute_plan(sub, binding=binding, torch_module=fake,
                    output_dir=str(tmp_path), save_raw=False)
    assert fake.cuda.sync_count == 1
    assert len(fake.cpu_downloads) == len(pr.ALL_PHASES) * 2
    assert all(d["sync_count"] >= 1 for d in fake.cpu_downloads)


def test_normal_order_create_sync_download_destroy(built, tmp_path):
    sub = _single_case_built(built, "M1")
    fake = FakeTorch()
    binding = FakeNativeBinding(torch_module=fake, write_fill=b"\x22")
    pr.execute_plan(sub, binding=binding, torch_module=fake,
                    output_dir=str(tmp_path), save_raw=False)
    timeline = _timeline(fake)
    assert timeline.index("native_create") < timeline.index("native_project")
    assert timeline.index("native_project") < timeline.index("sync")
    assert timeline.index("sync") < timeline.index("cpu")
    assert timeline.index("cpu") < timeline.index("native_destroy")
    assert timeline[-1] == "native_destroy"


def test_device_pointer_writes_are_visible_after_download(built, tmp_path):
    sub = _single_case_built(built, "M1")
    fake = FakeTorch()
    binding = FakeNativeBinding(torch_module=fake, write_fill=b"\x5a")
    pr.execute_plan(sub, binding=binding, torch_module=fake,
                    output_dir=str(tmp_path), save_raw=False)
    for call in binding.projects():
        assert call["output_buf"].tobytes() == b"\x5a" * pr.case_output_nbytes(1)


# ---------------------------------------------------------------------------
# Baselines: numerical gate vs lifecycle
# ---------------------------------------------------------------------------

def _baseline_outputs(built, case_id):
    if case_id == "M1":
        return {"wkv": built.fixture.reference.projected_bytes,
                "wgate": built.fixture.reference.scores_bytes}
    return {"wkv": built.fixture.candidate.projected_bytes,
            "wgate": built.fixture.candidate.scores_bytes}


def test_baseline_bytes_match_oracle_gives_pass(built, tmp_path):
    sub = _single_case_built(built, "M1")
    fake = FakeTorch()
    binding = FakeNativeBinding(torch_module=fake,
                                outputs_by_label=_baseline_outputs(built, "M1"))
    results = pr.execute_plan(sub, binding=binding, torch_module=fake,
                              output_dir=str(tmp_path), save_raw=False)
    case_result = results["cases"][0]
    assert case_result["numerical_baseline_pass"] is True
    assert case_result["baseline_case_pass"] is True
    assert results["summary"]["all_calls_succeeded"] is True
    assert results["summary"]["component_baseline_pass"] is True


def test_baseline_gate_requires_matching_bytes_not_just_rc(built, tmp_path):
    plan = dict(built.plan)
    plan["cases"] = [c for c in built.plan["cases"] if c["kind"] == "baseline"]
    sub = pr.BuiltPlan(plan=plan,
                       case_inputs={c["id"]: built.case_inputs[c["id"]]
                                    for c in plan["cases"]},
                       weights=built.weights, fixture=built.fixture)
    fake = FakeTorch()
    binding = FakeNativeBinding(torch_module=fake, write_fill=b"\x00")
    results = pr.execute_plan(sub, binding=binding, torch_module=fake,
                              output_dir=str(tmp_path), save_raw=False)
    assert results["summary"]["all_calls_succeeded"] is True
    assert results["summary"]["component_baseline_pass"] is False
    for case_result in results["cases"]:
        assert case_result["numerical_baseline_pass"] is False
        assert case_result["baseline_case_pass"] is False
        assert case_result["status"] == "mismatch"


def test_lifecycle_and_numerical_gates_are_distinct(built, tmp_path):
    # Bytes match, but destroy fails: lifecycle fails, numerical passes.
    sub = _single_case_built(built, "M1")
    fake = FakeTorch()
    binding = FakeNativeBinding(
        torch_module=fake, destroy_rc=9,
        outputs_by_label=_baseline_outputs(built, "M1"))
    results = pr.execute_plan(sub, binding=binding, torch_module=fake,
                              output_dir=str(tmp_path), save_raw=True)
    case_result = results["cases"][0]
    assert case_result["numerical_baseline_pass"] is True
    assert case_result["lifecycle_success"] is False
    assert case_result["baseline_case_pass"] is False
    assert case_result["all_calls_ok"] is False
    # Downloaded evidence survives cleanup failure, but the gate stays failed.
    assert len(_raw_files(tmp_path)) == 8


# ---------------------------------------------------------------------------
# Full matrix and CF target recording
# ---------------------------------------------------------------------------

def test_full_matrix_saves_every_role_and_repeat(built, tmp_path):
    fake = FakeTorch()
    binding = FakeNativeBinding(torch_module=fake, write_fill=b"\x33")
    results = pr.execute_plan(built, binding=binding, torch_module=fake,
                              output_dir=str(tmp_path), save_raw=True)
    recorded = sorted(os.listdir(os.path.join(str(tmp_path),
                                              pr.RECORDED_SUBDIR)))
    warmup = sorted(os.listdir(os.path.join(str(tmp_path),
                                            pr.WARMUP_SUBDIR)))
    n_cases = len(built.plan["cases"])
    assert len(recorded) == n_cases * 2 * len(pr.REPEAT_PHASES) == 84
    assert len(warmup) == n_cases * 2 * 1 == 28
    assert all("__warmup." in name or name.endswith("__warmup.fp32")
               for name in warmup)
    assert all("__repeat" in name for name in recorded)
    assert results["summary"]["all_calls_succeeded"] is True


def test_cf_records_target_slot_without_oracle(built, tmp_path):
    for case_id in ("CF-N2-slot1-zero-neighbor",
                    "CF-N16-slot15-flip-neighbor"):
        case = _case(built, case_id)
        sub = _single_case_built(built, case_id)
        fake = FakeTorch()
        binding = FakeNativeBinding(torch_module=fake, write_fill=b"\x44")
        results = pr.execute_plan(sub, binding=binding, torch_module=fake,
                                  output_dir=str(tmp_path / case_id),
                                  save_raw=False)
        case_result = results["cases"][0]
        assert case_result["cf_without_oracle"] is True
        assert case_result["baseline_case_pass"] is None
        assert case_result["status"] == "recorded"
        assert case_result["scored_target"]["slot"] == case["target_slot"]
        assert case_result["scored_target"]["rows_scored"] == 1
        assert case_result["scored_target"]["projected_target_row_sha256"] is not None
        assert case_result["scored_target"]["scores_target_row_sha256"] is not None


# ---------------------------------------------------------------------------
# Lifecycle failures: cleanup and no output
# ---------------------------------------------------------------------------

def test_failed_project_stops_dependents_and_suppresses_output(built, tmp_path):
    sub = _single_case_built(built, "M1")
    # warmup wkv succeeds, then warmup wgate is rejected (rc=7).
    fake = FakeTorch()
    binding = FakeNativeBinding(torch_module=fake, project_rcs=[0, 7],
                                write_fill=b"\x01")
    result = pr.execute_plan(sub, binding=binding, torch_module=fake,
                             output_dir=str(tmp_path), save_raw=True)
    projects = binding.projects()
    assert len(projects) == 2
    assert projects[0]["weight_buf"].label == "wkv"
    assert projects[1]["weight_buf"].label == "wgate"
    # No dependent project after the rejection.
    assert [c["weight_buf"].label for c in projects] == ["wkv", "wgate"]
    assert _raw_files(tmp_path) == []
    case_result = result["cases"][0]
    assert case_result["project_rejected"] is True
    assert case_result["aborted"] is True
    assert case_result["status"] == "unscored"
    assert case_result["baseline_case_pass"] is False
    # Synchronize happened before destroy even though project was rejected.
    timeline = _timeline(fake)
    assert timeline.index("sync") < timeline.index("native_destroy")
    assert binding.ops().count("destroy") == 1


def test_create_failure_skips_projects_download_and_destroy(built, tmp_path):
    sub = _single_case_built(built, "M1")
    fake = FakeTorch()
    binding = FakeNativeBinding(torch_module=fake, create_rc=13)
    result = pr.execute_plan(sub, binding=binding, torch_module=fake,
                             output_dir=str(tmp_path), save_raw=True)
    assert binding.ops() == ["create"]
    assert _raw_files(tmp_path) == []
    assert fake.cpu_downloads == []
    case_result = result["cases"][0]
    assert case_result["all_calls_ok"] is False
    assert case_result["lifecycle_success"] is False
    assert result["summary"]["all_calls_succeeded"] is False


def test_project_exception_still_syncs_and_destroys(built, tmp_path):
    sub = _single_case_built(built, "M1")
    fake = FakeTorch()
    binding = RaisingProjectBinding(torch_module=fake)
    result = pr.execute_plan(sub, binding=binding, torch_module=fake,
                             output_dir=str(tmp_path), save_raw=True)
    case_result = result["cases"][0]
    assert any(e["phase"] == "project"
               for e in case_result["lifecycle_errors"])
    assert "native boom" in json.dumps(case_result["lifecycle_errors"])
    assert _raw_files(tmp_path) == []
    timeline = _timeline(fake)
    assert timeline.index("native_create") < timeline.index("sync")
    assert timeline.index("sync") < timeline.index("native_destroy")
    assert binding.ops()[-1] == "destroy"


def test_sync_failure_suppresses_download(built, tmp_path):
    sub = _single_case_built(built, "M1")
    fake = FakeTorch()
    fake.cuda.sync_error = RuntimeError("sync boom")
    binding = FakeNativeBinding(torch_module=fake, write_fill=b"\x02")
    result = pr.execute_plan(sub, binding=binding, torch_module=fake,
                             output_dir=str(tmp_path), save_raw=True)
    case_result = result["cases"][0]
    assert fake.cuda.sync_count == 1
    assert fake.cpu_downloads == []
    assert _raw_files(tmp_path) == []
    assert any(e["phase"] == "synchronize"
               for e in case_result["lifecycle_errors"])
    assert case_result["synchronized"] is False
    assert binding.ops()[-1] == "destroy"


def test_download_exception_leaves_no_output(built, tmp_path):
    sub = _single_case_built(built, "M1")
    fake = FakeTorch()
    fake.cpu_error = RuntimeError("download boom")
    binding = FakeNativeBinding(torch_module=fake, write_fill=b"\x03")
    result = pr.execute_plan(sub, binding=binding, torch_module=fake,
                             output_dir=str(tmp_path), save_raw=True)
    case_result = result["cases"][0]
    assert any(e["phase"] == "download"
               for e in case_result["lifecycle_errors"])
    assert _raw_files(tmp_path) == []
    assert binding.ops()[-1] == "destroy"
    assert result["summary"]["all_calls_succeeded"] is False


def test_destroy_failure_is_reported_and_output_retained(built, tmp_path):
    for kwargs in ({"destroy_rc": 4}, {"destroy_exc": True}):
        sub = _single_case_built(built, "M1")
        fake = FakeTorch()
        binding = FakeNativeBinding(
            torch_module=fake, write_fill=b"\x04", **kwargs)
        result = pr.execute_plan(sub, binding=binding, torch_module=fake,
                                 output_dir=str(tmp_path / str(kwargs)),
                                 save_raw=True)
        case_result = result["cases"][0]
        assert case_result["lifecycle_success"] is False
        assert case_result["all_calls_ok"] is False
        assert len(_raw_files(tmp_path / str(kwargs))) == 8
        assert binding.ops()[-1] == "destroy"


def test_create_exception_is_reported_without_destroy(built, tmp_path):
    sub = _single_case_built(built, "M1")
    fake = FakeTorch()
    binding = FakeNativeBinding(torch_module=fake, create_exc=True)
    result = pr.execute_plan(sub, binding=binding, torch_module=fake,
                             output_dir=str(tmp_path), save_raw=True)
    case_result = result["cases"][0]
    assert any(e["phase"] == "create"
               for e in case_result["lifecycle_errors"])
    assert binding.ops() == ["create"]
    assert _raw_files(tmp_path) == []


# ---------------------------------------------------------------------------
# Gates: baseline vs CF (summarize directly)
# ---------------------------------------------------------------------------

def _minimal_plan(cases):
    return {"cases": cases}


def _case_result(case_id, kind, all_ok=True, baseline_pass=None):
    return {"case": case_id, "kind": kind, "all_calls_ok": all_ok,
            "baseline_case_pass": baseline_pass}


def test_baseline_gate_requires_pass_on_every_baseline():
    plan = _minimal_plan([
        {"id": "M1", "kind": "baseline"},
        {"id": "M2", "kind": "baseline"},
        {"id": "CF", "kind": "cf"},
    ])
    good = [_case_result("M1", "baseline", True, True),
            _case_result("M2", "baseline", True, True),
            _case_result("CF", "cf", True, None)]
    summary = pr.summarize(plan, good)
    assert summary["all_calls_succeeded"] is True
    assert summary["component_baseline_pass"] is True

    unscored = [_case_result("M1", "baseline", False, False),
                _case_result("M2", "baseline", True, True),
                _case_result("CF", "cf", True, None)]
    summary = pr.summarize(plan, unscored)
    assert summary["all_calls_succeeded"] is False
    assert summary["component_baseline_pass"] is False

    mismatched = [_case_result("M1", "baseline", True, True),
                  _case_result("M2", "baseline", True, False),
                  _case_result("CF", "cf", True, None)]
    summary = pr.summarize(plan, mismatched)
    assert summary["all_calls_succeeded"] is True
    assert summary["component_baseline_pass"] is False


def test_cf_cases_cannot_produce_baseline_pass():
    plan = _minimal_plan([{"id": "CF", "kind": "cf"}])
    results = [_case_result("CF", "cf", True, None)]
    summary = pr.summarize(plan, results)
    assert summary["all_calls_succeeded"] is True
    assert summary["component_baseline_pass"] is False
    assert summary["no_overall_arithmetic_pass"] is True


# ---------------------------------------------------------------------------
# Native hash handling and fixture hash enforcement
# ---------------------------------------------------------------------------

def test_native_sha256_format_and_deferred_file_hash():
    assert pr.validate_native_sha256_format(pr.NATIVE_SHA256_FROZEN) == \
        pr.NATIVE_SHA256_FROZEN
    assert pr.validate_native_sha256_format(pr.NATIVE_SHA256_FROZEN.upper()) == \
        pr.NATIVE_SHA256_FROZEN
    for bad in ("", "abc", "z" * 64, pr.NATIVE_SHA256_FROZEN + "0"):
        with pytest.raises(pr.ValidationError):
            pr.validate_native_sha256_format(bad)
    built = pr.build_plan(REAL_FIXTURE, native_library="/nonexistent/lib.so",
                          native_sha256=pr.NATIVE_SHA256_FROZEN)
    assert built.plan["native_file_hash_deferred_to_execute"] is True


def test_native_library_hash_checked_before_load(tmp_path):
    fake_lib = tmp_path / "libfake.so"
    fake_lib.write_bytes(b"not a shared object")
    with pytest.raises(pr.ValidationError):
        pr.load_native_binding(str(fake_lib), "0" * 64)
    actual = pr.sha256_bytes(fake_lib.read_bytes())
    with pytest.raises(OSError):
        pr.load_native_binding(str(fake_lib), actual)


def test_real_fixture_hash_enforcement_passes(built):
    checks = built.plan["fixture"]["checks"]
    assert checks["captured_current_input_rows_all_equal"] is True
    for label, entry in checks["hashes"].items():
        assert entry["ok"] is True
        assert entry["actual"] == entry["expected"]
    assert checks["hashes"]["candidate_projected_row0"]["actual"] == \
        pr.FREEZE["candidate_projected_row_sha256"]
    assert checks["hashes"]["candidate_projected_row1"]["actual"] == \
        pr.FREEZE["candidate_projected_row_sha256"]
    assert checks["hashes"]["candidate_scores_row0"]["actual"] == \
        pr.FREEZE["candidate_scores_row_sha256"]


def test_hash_helper_rejects_mismatch_and_unequal_rows():
    with pytest.raises(pr.FixtureError):
        pr.verify_expected_hash("00" * 32, "11" * 32, "unit")
    pr.verify_expected_hash("11" * 32, "11" * 32, "unit")

    same = [b"row-a", b"row-a"]
    assert pr.input_rows_all_equal(same, pr.sha256_bytes(b"row-a")) is True
    assert pr.input_rows_all_equal([b"row-a", b"row-b"],
                                   pr.sha256_bytes(b"row-a")) is False
    assert pr.input_rows_all_equal([], pr.sha256_bytes(b"row-a")) is False
    with pytest.raises(pr.FixtureError):
        pr.split_rows(b"12345", 2)


def _write_synthetic_fixture(root, input_fill=b"\x00"):
    root = str(root)
    for directory, rows in ((pr.REFERENCE_DIR, 1), (pr.CANDIDATE_DIR, 2)):
        d = os.path.join(root, directory)
        os.makedirs(d, exist_ok=True)
        manifest = {
            "schema": 1, "kind": "compressor-inputs", "layer": 2, "ratio": 2,
            "rows": rows,
            "buffers": {
                "input": {"name": "input", "file": pr.INPUT_FILE,
                          "dtype": "bfloat16", "shape": [rows, 5120],
                          "bytes": rows * pr.INPUT_ROW_BYTES},
                "projected": {"name": "projected", "file": pr.PROJECTED_FILE,
                              "dtype": "float32", "shape": [rows, 512],
                              "bytes": rows * pr.OUTPUT_ROW_BYTES},
                "scores": {"name": "scores", "file": pr.SCORES_FILE,
                           "dtype": "float32", "shape": [rows, 512],
                           "bytes": rows * pr.OUTPUT_ROW_BYTES},
            },
        }
        with open(os.path.join(d, pr.MANIFEST_NAME), "w", encoding="utf-8") as fh:
            json.dump(manifest, fh)
        open(os.path.join(d, pr.INPUT_FILE), "wb").write(
            input_fill * (rows * pr.INPUT_ROW_BYTES))
        open(os.path.join(d, pr.PROJECTED_FILE), "wb").write(
            b"\x00" * (rows * pr.OUTPUT_ROW_BYTES))
        open(os.path.join(d, pr.SCORES_FILE), "wb").write(
            b"\x00" * (rows * pr.OUTPUT_ROW_BYTES))
    wd = os.path.join(root, pr.WEIGHTS_DIR)
    os.makedirs(wd, exist_ok=True)
    open(os.path.join(wd, pr.WKV_FILE), "wb").write(b"\x00" * pr.WEIGHT_BYTES)
    open(os.path.join(wd, pr.WGATE_FILE), "wb").write(b"\x00" * pr.WEIGHT_BYTES)
    return root


def test_synthetic_fixture_with_correct_shape_is_rejected(tmp_path):
    root = _write_synthetic_fixture(tmp_path / "synthetic")
    with pytest.raises(pr.FixtureError):
        pr.validate_fixture(root)
    with pytest.raises(pr.FixtureError):
        pr.build_plan(root, native_library="x",
                      native_sha256=pr.NATIVE_SHA256_FROZEN)


# ---------------------------------------------------------------------------
# Default CLI on the real fixture: CPU-only, no torch
# ---------------------------------------------------------------------------

def test_default_cli_actual_fixture_plan_cpu_only(tmp_path):
    assert "torch" not in sys.modules
    out = tmp_path / "plan-out"
    argv = [
        "projection_replay.py",
        "--activations", REAL_FIXTURE,
        "--native-library", "/nonexistent/libds41rt.so",
        "--native-sha256", pr.NATIVE_SHA256_FROZEN,
        "--output", str(out),
    ]
    saved = sys.argv
    sys.argv = argv
    try:
        with pytest.raises(SystemExit) as excinfo:
            runpy.run_path(MODULE_PATH, run_name="__main__")
        assert excinfo.value.code == 0
    finally:
        sys.argv = saved

    assert "torch" not in sys.modules
    plan = json.loads((out / "plan.json").read_text())
    assert plan["mode"] == "plan"
    assert len(plan["cases"]) == 14
    assert plan["fixture"]["checks"]["captured_current_input_rows_all_equal"] is True
    assert not (out / "results.json").exists()


def test_output_is_new_only(tmp_path):
    existing = tmp_path / "already-there"
    existing.mkdir()
    with pytest.raises(pr.ValidationError):
        pr.run(SimpleNamespace(activations=REAL_FIXTURE,
                               native_library="/nonexistent/lib.so",
                               native_sha256=pr.NATIVE_SHA256_FROZEN,
                               output=str(existing), execute=False))
