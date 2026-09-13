#!/usr/bin/env python3
"""Poison and reuse a single arena across ordered and atomic local variants."""
import argparse
import ctypes as C
import json
import torch
import _pinned_sparkinfer
from _v41_expert_native import Native, Info, library, check
from tests.moe.test_v41_grouped_slices import _check_grouped_slices

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--native-lib', required=True)
args = parser.parse_args()
lib = library(args.native_lib, local=True)
capacities = (1, 16, 80, 256, 1024, 4096)
sizes = []
for capacity in capacities:
    info = Info()
    check(lib.ds41rt_v41_expert_info(capacity, C.byref(info)))
    sizes.append(info.scratch_bytes)
arena = torch.empty(max(sizes), dtype=torch.uint8, device='cuda')
weights_full = None
owners = {}
records = []

def after_case(**case):
    global weights_full
    weights, wire, ids_cpu, routing = case['native_inputs']
    rows = case['rows']
    if weights_full is None:
        weights_full = []
        for weight in weights:
            full = torch.zeros(weight.numel()*3, dtype=weight.dtype, device='cuda')
            full[weight.numel()*2:].copy_(weight.reshape(-1))
            weights_full.append(full)
    ids_cpu = ids_cpu + 256
    # Ascend then descend on alternating input cases, returning from atomic
    # prefill variants to ordered decode without reinitializing shared scratch.
    order = capacities if len(records) % 2 == 0 else capacities[::-1]
    for capacity in order:
        if capacity < rows:
            continue
        if capacity not in owners:
            ids = torch.full((capacity,6), -1, dtype=torch.int32, device='cuda')
            rw = torch.zeros(capacity,6, device='cuda')
            shared = Native(lib, capacity, weights_full, wire, ids, rw, full_backbone=True, storage=arena)
            separate = Native(lib, capacity, weights_full, wire, ids, rw, full_backbone=True)
            shared.run(1); separate.run(1)
            owners[capacity] = shared, separate, ids, rw, {}
        shared, separate, ids, rw, graphs = owners[capacity]
        ids.fill_(-1); ids[:rows].copy_(ids_cpu)
        rw.zero_(); rw[:rows].copy_(routing)
        if rows not in graphs:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                shared.run(rows)
            graphs[rows] = graph
        separate.run(rows)
        n = rows if shared.token_accumulation else rows*6
        reference = separate.output[:n].clone()
        # Every scratch byte is hostile; no initialization or previous variant
        # output can supply missing data to this replay.
        arena.fill_(255)
        allocated = torch.cuda.memory_allocated()
        graphs[rows].replay()
        assert torch.cuda.memory_allocated() == allocated
        actual = shared.output[:n]
        assert torch.isfinite(actual).all() and actual.abs().max() > 0
        if shared.token_accumulation:
            rel = ((actual-reference).norm()/reference.norm()).item()
            assert rel < 1e-5, (case['case'], capacity, rel)
        else:
            assert torch.equal(actual, reference), (case['case'], capacity)
            rel = 0.0
            # Also compare against the independently launched grouped oracle.
            expected = torch.zeros_like(reference)
            partial = case['out'][:,:rows*6]
            summed = torch.zeros_like(partial[0])
            for plane in partial:
                summed += plane
            pairs = case['pair_gpu']
            expected[pairs[:,0]*6+pairs[:,1]] = summed
            assert torch.equal(actual, expected)
        records.append(dict(case=case['case'], rows=rows, capacity=capacity,
                            atomic=shared.token_accumulation, relative_l2=rel,
                            poisoned=True, allocation_free_replay=True))
for_run = after_case
_check_grouped_slices(192, after_case=for_run, n=2304, topk=6)
for shared, separate, ids, rw, graphs in owners.values():
    for graph in graphs.values():
        graph.reset()
assert set(owners) == set(capacities)
print('RESULT '+json.dumps(dict(arena_bytes=arena.numel(), separate_bytes=sum(sizes), cases=records)))
