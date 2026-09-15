#!/usr/bin/env python3
"""Check per-GPU TP2 module ownership before executing expert kernels."""
import argparse
import ctypes as C
import json
import torch
from _v41_expert_native import Info, P


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-lib", required=True)
    args = parser.parse_args()
    lib = C.CDLL(args.native_lib)
    initialize = lib.ds41rt_v41_tp2_expert_initialize
    initialize.argtypes = [C.c_int32, C.POINTER(P)]
    info_fn = lib.ds41rt_v41_tp2_expert_info
    info_fn.argtypes = [C.c_int32, C.POINTER(Info)]
    bind = lib.ds41rt_v41_tp2_expert_bind_scratch
    bind.argtypes = [P, P, C.c_uint64, C.POINTER(P)]
    init_scratch = lib.ds41rt_v41_tp2_expert_initialize_scratch_async
    init_scratch.argtypes = [P, P, C.c_uint64, P]
    assert torch.cuda.device_count() >= 2
    results = []
    for capacity in (1, 16):
        handles, infos = [P(), P()], [Info(), Info()]
        for device in (0, 1):
            with torch.cuda.device(device):
                assert info_fn(capacity, C.byref(infos[device])) == 0
                assert (infos[device].role, infos[device].logical_intermediate) == (3, 1152)
                assert initialize(capacity, C.byref(handles[device])) == 0
                again = P()
                assert initialize(capacity, C.byref(again)) == 0
                assert again.value == handles[device].value
        assert handles[0].value != handles[1].value
        for device in (0, 1):
            with torch.cuda.device(device):
                storage = torch.empty(infos[device].scratch_bytes, dtype=torch.uint8, device="cuda")
                slots = (P * 44)(*([16] * 44))
                assert bind(handles[device], storage.data_ptr(), storage.numel(), slots) == 0
                before = list(slots)
                assert bind(handles[1-device], storage.data_ptr(), storage.numel(), slots) != 0
                assert list(slots) == before
                stream = torch.cuda.current_stream().cuda_stream
                assert init_scratch(handles[device], storage.data_ptr(), storage.numel(), stream) == 0
                assert init_scratch(handles[1-device], storage.data_ptr(), storage.numel(), stream) != 0
                torch.cuda.synchronize()
        results.append(dict(capacity=capacity, distinct_handles=True, crossed_handles_rejected=True,
                            scratch_bytes=infos[0].scratch_bytes))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
