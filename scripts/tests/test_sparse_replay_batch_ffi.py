"""CPU contract stubs for the sparse-replay batch FFI path (no torch, no CUDA).

The extracted ``sparse_replay.runner.launch_batch`` helper is the exact call
path the GPU runner uses.  These tests drive it with duck-typed fake buffers
and stub only ``torch.frombuffer`` and the device descriptor ``copy_`` so the
ordering contract is observable:

* the native validator receives the *device descriptor* address, never the
  stream handle;
* the host descriptors (exactly 120 raw bytes per row) are uploaded only after
  validation succeeds;
* the launch receives the device descriptor address and the stream in their
  correct argument slots;
* an rc-1 validation performs no upload and no launch.

The rest of the ctypes signature is checked against the local native header so
the argument order cannot drift from the ABI.
"""

from __future__ import annotations

import ctypes as C
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from sparse_replay.runner import (  # noqa: E402
    BATCH_ARGTYPES,
    BATCH_VALIDATE_ARGTYPES,
    BOUNDED_ARGTYPES,
    VIEW_BYTES,
    View,
    classify_case,
    launch_batch,
)

HEADER = (
    Path(__file__).resolve().parents[2]
    / "native"
    / "include"
    / "ds41rt_v41_sparse_attention.h"
)

# Distinct sentinel addresses: a bad call path would leak one where the other
# belongs, which is exactly the defect this contract guards against.
STREAM = 0x5EAD0
DESCRIPTOR_DEVICE = 0xD0D0D0


class FakeTensor:
    """Minimal duck-typed tensor: address and element count only."""

    def __init__(self, address, elements=1):
        self._address = address
        self._elements = elements

    def data_ptr(self):
        return self._address

    def numel(self):
        return self._elements


class FakeDescriptors(FakeTensor):
    """Fake device descriptor buffer recording the ``copy_`` upload."""

    def __init__(self, address, events):
        super().__init__(address, VIEW_BYTES)
        self.events = events
        self.copied = None

    def copy_(self, source, non_blocking=False):
        self.copied = source
        self.events.append(("copy", source, non_blocking))


class FakeTorch:
    """Stub exposing only ``frombuffer``/``uint8`` used by the helper."""

    uint8 = "uint8"

    def __init__(self, events):
        self.events = events

    def frombuffer(self, buffer, dtype=None):
        raw = bytes(buffer)
        self.events.append(("frombuffer", raw, dtype))
        return ("host-buffer", raw)


class FakeSession:
    """Native-session double recording validate/launch arguments and order."""

    def __init__(self, events, validate_rc=0, launch_rc=0):
        self.events = events
        self.torch = FakeTorch(events)
        self.validate_rc = validate_rc
        self.launch_rc = launch_rc
        self.validate_args = None
        self.batch_args = None

    def batch_validate(self, *args):
        self.events.append(("validate", args))
        self.validate_args = args
        return self.validate_rc

    def batch(self, *args):
        self.events.append(("batch", args))
        self.batch_args = args
        return self.launch_rc


def _host_views(rows):
    """A real ``(View * rows)`` host array; only geometry bytes matter here."""
    views = []
    for _ in range(rows):
        view = View()
        view.window_capacity = 128
        view.source_capacity = 20
        view.source_proposal_capacity = 3
        view.page_stride = 2
        view.compressed = 2
        views.append(view)
    return (View * rows)(*views)


def _call(session, rows=2, stream=STREAM, descriptor_device=DESCRIPTOR_DEVICE):
    host_views = _host_views(rows)
    descriptors = FakeDescriptors(descriptor_device, session.events)
    rc = launch_batch(
        session,
        query=FakeTensor(0x1000, 1),
        sink=FakeTensor(0x1100, 1),
        metadata=FakeTensor(0x1200, 1),
        selected=FakeTensor(0x1300, 1),
        output=FakeTensor(0x1400, 1),
        host_views=host_views,
        descriptors=descriptors,
        stream=stream,
        bounds=FakeTensor(0x1500, 1),
        scratch=FakeTensor(0x1600, 5),
        rows=rows,
        parts=10,
        compressed=2,
    )
    return rc, session, host_views, descriptors


def _event_names(session):
    return [event[0] for event in session.events]


# ---------------------------------------------------------------------------
# Argument placement.
# ---------------------------------------------------------------------------

def test_validator_receives_device_descriptor_address_not_stream():
    rc, session, host_views, _ = _call(FakeSession([]))
    assert rc == 0
    # batch_validate: ..., rows, host_views, device_views, begins, partial, ...
    args = session.validate_args
    assert len(args) == 13
    assert args[5] == 2
    assert isinstance(args[6], C.Array)
    assert C.addressof(args[6]) == C.addressof(host_views)
    assert args[7] == DESCRIPTOR_DEVICE
    assert args[7] != STREAM
    assert args[8] == 0x1500  # begins
    assert args[10] == 5 * 4  # scratch bytes
    assert args[11] == 10
    assert args[12] == 2


def test_launch_receives_stream_and_device_descriptor_in_proper_slots():
    rc, session, _, _ = _call(FakeSession([]))
    assert rc == 0
    args = session.batch_args
    assert len(args) == 12
    # batch: ..., rows, device_views, stream, begins, partial, parts, compressed
    assert args[6] == DESCRIPTOR_DEVICE
    assert args[7] == STREAM
    assert args[6] != STREAM
    assert args[8] == 0x1500
    assert args[10] == 10
    assert args[11] == 2


# ---------------------------------------------------------------------------
# Ordering and rejection.
# ---------------------------------------------------------------------------

def test_upload_happens_only_after_successful_validation():
    rc, session, host_views, descriptors = _call(FakeSession([]))
    assert rc == 0
    assert _event_names(session) == ["validate", "frombuffer", "copy", "batch"]
    # The upload is exactly the raw 120 bytes per host descriptor row.
    frombuffer = next(e for e in session.events if e[0] == "frombuffer")
    assert len(frombuffer[1]) == 2 * VIEW_BYTES == 240
    assert frombuffer[1] == bytes(memoryview(host_views))
    copy = next(e for e in session.events if e[0] == "copy")
    assert copy[2] is False  # non_blocking=False
    assert descriptors.copied == ("host-buffer", bytes(memoryview(host_views)))


def test_rejected_validation_prevents_upload_and_launch():
    rc, session, _, descriptors = _call(FakeSession([], validate_rc=1))
    assert rc == 1
    # Validation ran, against the device descriptor address, and nothing else.
    assert _event_names(session) == ["validate"]
    assert session.validate_args[7] == DESCRIPTOR_DEVICE
    assert session.batch_args is None
    assert descriptors.copied is None


def test_launch_rejection_is_returned_unchanged_after_upload():
    rc, session, _, descriptors = _call(FakeSession([], launch_rc=7))
    assert rc == 7
    assert _event_names(session) == ["validate", "frombuffer", "copy", "batch"]
    assert descriptors.copied is not None


def test_descriptor_row_width_is_raw_120_bytes():
    assert VIEW_BYTES == 120
    assert C.sizeof(View) == VIEW_BYTES
    host_views = _host_views(3)
    assert len(bytes(memoryview(host_views))) == 3 * VIEW_BYTES
    assert bytes(memoryview(host_views)) == b"".join(
        bytes(memoryview(view)) for view in host_views
    )


# ---------------------------------------------------------------------------
# ABI signature vs. the local native header.
# ---------------------------------------------------------------------------

def _header_params(name):
    text = re.sub(r"//[^\n]*", "", HEADER.read_text())
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


def test_ctypes_signatures_match_local_header():
    bounded = _header_params("ds41rt_v41_sparse_attention_bounded")
    validate = _header_params("ds41rt_v41_sparse_attention_batch_validate")
    batch = _header_params("ds41rt_v41_sparse_attention_batch")

    assert len(bounded) == len(BOUNDED_ARGTYPES) == 13
    assert len(validate) == len(BATCH_VALIDATE_ARGTYPES) == 13
    assert len(batch) == len(BATCH_ARGTYPES) == 12

    assert "view" in bounded[7] and "stream" in bounded[8]
    assert "host_views" in validate[6]
    assert "device_views" in validate[7]
    assert "stream" not in validate[7]
    assert "begins" in validate[8]
    assert "partial" in validate[9]
    assert "device_views" in batch[6]
    assert "stream" in batch[7]

    # ctypes pointer placements: host descriptor is a View pointer; the device
    # descriptor and stream are opaque pointers.
    assert BOUNDED_ARGTYPES[7] == C.POINTER(View)
    assert BATCH_VALIDATE_ARGTYPES[6] == C.POINTER(View)
    assert BATCH_VALIDATE_ARGTYPES[7] == C.c_void_p
    assert BATCH_ARGTYPES[6] == C.c_void_p
    assert BATCH_ARGTYPES[7] == C.c_void_p


# ---------------------------------------------------------------------------
# Rejected launches are unexecuted, never numerical mismatches.
# ---------------------------------------------------------------------------

def _record(executed, byte_exact):
    return {
        "executed": executed,
        "rows": [{"byte_exact": byte_exact}],
    }


def test_rc1_warmup_is_reported_unexecuted_not_numerical_mismatch():
    outcome = classify_case(1, [_record(False, False)])
    assert outcome == {
        "executed": False,
        "numerical_pass": False,
        "status": "unexecuted",
        "unscored": True,
    }


def test_rc1_repeat_is_reported_unexecuted_not_numerical_mismatch():
    outcome = classify_case(0, [_record(False, False)])
    assert outcome["status"] == "unexecuted"
    assert outcome["numerical_pass"] is False
    assert outcome["unscored"] is True


def test_executed_byte_exact_is_a_pass():
    outcome = classify_case(0, [_record(True, True), _record(True, True)])
    assert outcome == {
        "executed": True,
        "numerical_pass": True,
        "status": "pass",
        "unscored": False,
    }


def test_executed_byte_mismatch_is_a_numerical_mismatch():
    outcome = classify_case(0, [_record(True, True), _record(True, False)])
    assert outcome["status"] == "numerical_mismatch"
    assert outcome["numerical_pass"] is False
    assert outcome["unscored"] is False

