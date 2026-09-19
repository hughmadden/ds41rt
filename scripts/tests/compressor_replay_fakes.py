"""CPU fakes for the compressor replay executor paths (no torch, no CUDA).

``FakeTorch`` models just enough of the tensor surface used by
``compressor_replay.runner``: raw-byte buffers with a stable ``data_ptr``,
dtype/shape *views* that share storage, ``zero_``, ``copy_`` and a
``numpy().tobytes()`` download.  ``FakeBindings`` records every native call
argument and lets a test inject return codes and output writers.  Together they
let the real ``execute_experiment``/``run_experiments`` code paths run on CPU
while still observing ordering, aliasing and download suppression.
"""

from __future__ import annotations

import ctypes as C


class FakeNumpy:
    def __init__(self, memory):
        self._memory = memory

    def tobytes(self) -> bytes:
        return bytes(self._memory)


class FakeTensor:
    def __init__(self, torch, memory, address, dtype):
        self._torch = torch
        self.memory = memory
        self.address = address
        self.dtype = dtype

    def data_ptr(self) -> int:
        return self.address

    def zero_(self):
        self.memory[:] = bytes(len(self.memory))
        return self

    def view(self, arg):
        dtype = arg if isinstance(arg, str) else self.dtype
        return FakeTensor(self._torch, self.memory, self.address, dtype)

    def cpu(self):
        return self

    def numpy(self):
        return FakeNumpy(self.memory)

    def copy_(self, source, non_blocking=False):
        self.memory[:] = bytes(source.memory)
        self._torch.events.append(("copy_", non_blocking))
        return self

    def numel(self) -> int:
        return len(self.memory)

    def tobytes(self) -> bytes:
        return bytes(self.memory)


class _FakeCudaStream:
    def __init__(self, value):
        self.cuda_stream = value


class _FakeCuda:
    def __init__(self, torch, events):
        self._torch = torch
        self._events = events

    def set_device(self, device):
        self._events.append(("set_device", device))

    def synchronize(self):
        self._events.append(("synchronize",))

    def current_stream(self):
        return _FakeCudaStream(STREAM)


STREAM = 0x5EAD0
HANDLE = 0x11C0FFEE


class FakeTorch:
    uint8 = "uint8"
    bfloat16 = "bfloat16"
    float32 = "float32"
    uint64 = "uint64"

    def __init__(self):
        self.events = []
        self.pool = {}
        self._next = 0x10000000
        self.cuda = _FakeCuda(self, self.events)

    def _register(self, memory):
        address = self._next
        self._next += 0x10000  # keep every allocation 256-byte aligned
        self.pool[address] = memory
        return address

    def empty(self, size, dtype=None, device=None):
        if isinstance(size, (tuple, list)):
            count = 1
            for dim in size:
                count *= int(dim)
        else:
            count = int(size)
        memory = bytearray(count)
        address = self._register(memory)
        return FakeTensor(self, memory, address, dtype or "uint8")

    def frombuffer(self, buffer, dtype=None):
        memory = bytearray(bytes(buffer))
        address = self._register(memory)
        return FakeTensor(self, memory, address, dtype or "uint8")

    def memory_at(self, address) -> bytes:
        return bytes(self.pool[address])

    def write_at(self, address, data: bytes):
        self.pool[address][:len(data)] = data


class FakeBindings:
    """Records native calls; injectable return codes and output writers."""

    def __init__(self, torch, rcs=None, writer=None):
        self.torch = torch
        # ``None`` means every call returns 0.
        self.rcs = None if rcs is None else list(rcs)
        self.writer = writer
        self.calls = []
        self.created = []
        self.destroyed = []

    def _invoke(self, name, args):
        args = list(args)
        self.calls.append((name, args))
        if self.writer is not None:
            result = self.writer(name, args)
            if result is not None:
                return int(result)
        if self.rcs:
            return int(self.rcs.pop(0))
        return 0

    # Native ABI surface used by the runner.
    def create(self, workspace, size, handle_ref):
        self.created.append((workspace, size))
        self.torch.events.append(("create",))
        handle_ref._obj.value = HANDLE
        return self._invoke("create", [workspace, size, handle_ref])

    def destroy(self, handle):
        self.destroyed.append(handle)
        self.torch.events.append(("destroy",))
        return self._invoke("destroy", [handle])

    def project(self, *args):
        return self._invoke("project", args)

    def pool(self, *args):
        return self._invoke("pool", args)

    def pack(self, *args):
        return self._invoke("pack", args)

    # Convenience for tests.
    def call_names(self):
        return [name for name, _ in self.calls]
