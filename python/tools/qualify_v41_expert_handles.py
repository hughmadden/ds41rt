#!/usr/bin/env python3
import ctypes as C
import json
import argparse
import torch
from _v41_expert_native import library, Info, P, check
parser = argparse.ArgumentParser(description="Check dSpark/local expert handle isolation in one native library")
parser.add_argument("--native-lib", required=True)
path = parser.parse_args().native_lib
ordinary, local = library(path), library(path, local=True)
results = []
for capacity in (1, 16, 80, 256, 1024, 4096):
    infos, handles = [Info(), Info()], [P(), P()]
    for lib, info, handle in zip((ordinary, local), infos, handles):
        check(lib.ds41rt_v41_expert_info(capacity, C.byref(info)))
        check(lib.ds41rt_v41_expert_initialize(capacity, C.byref(handle)))
    assert (infos[0].role, infos[0].experts, infos[0].topk) == (0, 128, 3)
    assert (infos[1].role, infos[1].experts, infos[1].topk) == (2, 384, 6)
    assert handles[0].value != handles[1].value
    scratch = torch.empty(max(i.scratch_bytes for i in infos), dtype=torch.uint8, device='cuda')
    for i, lib in enumerate((ordinary, local)):
        slots = (P * 44)(*([16] * 44))
        check(lib.ds41rt_v41_expert_bind_scratch(handles[i], scratch.data_ptr(), scratch.numel(), slots))
        before = list(slots)
        assert lib.ds41rt_v41_expert_bind_scratch(handles[1-i], scratch.data_ptr(), scratch.numel(), slots) != 0
        assert list(slots) == before
    results.append(dict(capacity=capacity, distinct_handles=True, crossed_handles_rejected_without_slot_mutation=True, draft_scratch=infos[0].scratch_bytes, local_scratch=infos[1].scratch_bytes))
print(json.dumps(results, indent=2))
