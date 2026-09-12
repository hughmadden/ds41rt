"""ctypes bindings for official TP4 and local dSpark expert qualification."""

import ctypes as C
import torch

P = C.c_void_p
I = C.c_int32
U = C.c_uint32
L = C.c_uint64


class Info(C.Structure):
    _fields_ = (
        [
            (n, U)
            for n in [
                "abi_version",
                "role",
                "experts",
                "hidden_size",
                "logical_intermediate",
                "kernel_intermediate",
                "topk",
                "capacity_rows",
            ]
        ]
        + [("scratch_bytes", L)]
        + [
            (n, I)
            for n in [
                "max_rows",
                "rows_padded",
                "max_tasks",
                "max_phys_tiles",
                "max_active_clusters",
            ]
        ]
        + [("input_dtype", U)]
    )


class Launch(C.Structure):
    _fields_ = (
        [("tensors", P * 44)]
        + [
            (n, I)
            for n in [
                "num_tokens",
                "max_rows",
                "scatter_rows",
                "rows_padded",
                "max_tasks",
                "max_phys_tiles",
                "max_active_clusters",
            ]
        ]
        + [("stream", P)]
    )


assert C.sizeof(Info) == 64 and C.sizeof(Launch) == 392


def library(path):
    lib = C.CDLL(path)
    for name, args in {
        "ds41rt_v41_expert_info": [I, C.POINTER(Info)],
        "ds41rt_v41_expert_initialize": [I, C.POINTER(P)],
        "ds41rt_v41_expert_bind_scratch": [P, P, L, C.POINTER(P)],
        "ds41rt_v41_expert_initialize_scratch_async": [P, P, L, P],
        "ds41rt_v41_expert_launch": [P, C.POINTER(Launch)],
        "ds41rt_v41_pack_expert_async": [C.POINTER(P), C.POINTER(P), U, P],
        "ds41rt_v41_compact_routes_bf16_async": [P, P, U, P],
    }.items():
        fn = getattr(lib, name)
        fn.argtypes = args
        fn.restype = I
    return lib


def check(code):
    assert code == 0, code


class Native:
    def __init__(self, lib, capacity, weights, wire, ids, routing, *, coordinator=False):
        self.lib = lib
        self.info = info = Info()
        self.handle = P()
        check(lib.ds41rt_v41_expert_info(capacity, C.byref(info)))
        expected = ((0, 128, 5120, 2304, 2304, 3, capacity, 1)
                    if coordinator else (1, 384, 5120, 576, 640, 6, capacity, 7))
        assert (
            info.abi_version,
            info.role,
            info.experts,
            info.hidden_size,
            info.logical_intermediate,
            info.kernel_intermediate,
            info.topk,
            info.capacity_rows,
            info.input_dtype,
        ) == (info.abi_version, *expected)
        assert info.abi_version in (2, 3)
        self.token_accumulation = info.abi_version == 3
        if self.token_accumulation:
            query = lib.ds41rt_v41_expert_output_kind
            query.argtypes = [I, C.POINTER(U)]
            query.restype = I
            kind = U(99)
            check(query(capacity, C.byref(kind)))
            assert kind.value == 1
        check(lib.ds41rt_v41_expert_initialize(capacity, C.byref(self.handle)))
        self.storage = torch.empty(info.scratch_bytes, device="cuda", dtype=torch.uint8)
        self.args = Launch()
        slots = self.args.tensors
        check(
            lib.ds41rt_v41_expert_bind_scratch(
                self.handle, self.storage.data_ptr(), self.storage.numel(), slots
            )
        )
        w13, s13, w2, s2 = [x.data_ptr() for x in weights]
        for slot, pointer in [
            (22, w13),
            (23, s13),
            (24, w2),
            (25, s2),
            (26, s13),
            (27, s2),
            (28, slots[34]),
            (29, slots[34]),
            (30, w13),
            (31, s13),
            (32, w2),
            (33, s2),
            (38, slots[37]),
            (39, slots[37]),
            (0, wire.data_ptr()),
            (1, ids.data_ptr()),
            (2, routing.data_ptr()),
        ]:
            slots[slot] = pointer
        check(
            lib.ds41rt_v41_expert_initialize_scratch_async(
                self.handle,
                self.storage.data_ptr(),
                self.storage.numel(),
                torch.cuda.current_stream().cuda_stream,
            )
        )
        for name in [
            "max_rows",
            "rows_padded",
            "max_tasks",
            "max_phys_tiles",
            "max_active_clusters",
        ]:
            setattr(self.args, name, getattr(info, name))
        offset = slots[41] - self.storage.data_ptr()
        output_rows = capacity if self.token_accumulation else capacity * info.topk
        self.output = (
            self.storage[offset : offset + output_rows * 5120 * 4]
            .view(torch.float32)
            .reshape(output_rows, 5120)
        )

    def run(self, rows):
        self.args.num_tokens = rows
        self.args.scatter_rows = rows * self.info.topk
        self.args.stream = torch.cuda.current_stream().cuda_stream
        check(self.lib.ds41rt_v41_expert_launch(self.handle, C.byref(self.args)))
