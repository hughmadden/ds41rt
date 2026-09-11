#!/usr/bin/env python3
"""Qualify native direct-token expert output against the deployed ordered path.

Loads official TP-sharded weights and uses synthetic activations/routing. Includes
native grouping, compute and output layout checks; this is not API throughput.
"""
import argparse
import ctypes as C
import hashlib
import json
import statistics
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from _v41_expert_native import Native, library, check, P, I, U


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--baseline-lib', type=Path, required=True)
    p.add_argument('--candidate-lib', type=Path, required=True)
    p.add_argument('--layer', type=int, choices=range(40), default=0)
    p.add_argument('--rank', type=int, choices=range(4), default=0)
    p.add_argument('--experts', type=int, choices=[32, 384], default=384)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.manual_seed(413264 + a.rank)
    libs = {'baseline': library(str(a.baseline_lib)), 'candidate': library(str(a.candidate_lib))}
    token_cast = libs['candidate'].ds41rt_v41_compact_tokens_bf16_async
    token_cast.argtypes = [P, P, U, P]
    token_cast.restype = I
    query = libs['candidate'].ds41rt_v41_expert_output_kind
    query.argtypes = [I, C.POINTER(U)]
    query.restype = I
    invalid = U(99)
    assert query(3, C.byref(invalid)) != 0 and invalid.value == 99
    capacity, h = 1024, 5120
    weights = [torch.empty((384, size), dtype=torch.uint8, device='cuda')
               for size in [3276800, 204800, 1638400, 102400]]
    index = json.loads((a.snapshot / 'model.safetensors.index.json').read_text())['weight_map']
    with ExitStack() as stack:
        files = {}
        for expert in range(a.experts):
            sources = []
            for projection, suffix in [('w1','weight'), ('w3','weight'), ('w2','weight'),
                                       ('w1','scale'), ('w3','scale'), ('w2','scale')]:
                key = f'layers.{a.layer}.ffn.experts.{expert}.{projection}.{suffix}'
                shard = index[key]
                if shard not in files:
                    files[shard] = stack.enter_context(safe_open(a.snapshot / shard, framework='pt', device='cpu'))
                raw = files[shard].get_tensor(key).view(torch.uint8)
                if projection == 'w2':
                    width = 288 if suffix == 'weight' else 18
                    part = raw[:, a.rank * width:(a.rank + 1) * width]
                else:
                    part = raw[a.rank * 576:(a.rank + 1) * 576]
                sources.append(part.contiguous().cuda())
            check(libs['baseline'].ds41rt_v41_pack_expert_async(
                (P * 6)(*[x.data_ptr() for x in sources]),
                (P * 4)(*[x[expert].data_ptr() for x in weights]), 576,
                torch.cuda.current_stream().cuda_stream))
        torch.cuda.synchronize()
    wire = torch.empty((capacity, 5280), dtype=torch.uint8, device='cuda')
    ids = torch.empty((capacity, 6), dtype=torch.int32, device='cuda')
    routing = torch.empty((capacity, 6), device='cuda')
    owners = {name: {cap: Native(lib, cap, weights, wire, ids, routing)
                     for cap in [1, 80, 256, 1024]} for name, lib in libs.items()}
    for cap, owner in owners['candidate'].items():
        assert owner.token_accumulation == (cap >= 256)
    compact = torch.empty((capacity, h), dtype=torch.bfloat16, device='cuda')
    for ptr, dst, rows in [(0, compact.data_ptr(), 1), (wire.data_ptr(), 0, 1),
                          (wire.data_ptr()+1, compact.data_ptr(), 1),
                          (wire.data_ptr(), wire.data_ptr(), 1),
                          (wire.data_ptr(), compact.data_ptr(), 0),
                          (wire.data_ptr(), compact.data_ptr(), 4097),
                          (2**64-4, compact.data_ptr(), 1)]:
        assert token_cast(ptr, dst, rows, 0) != 0
    report = dict(layer=a.layer, rank=a.rank, experts=a.experts,
                  device=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
                  sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in [a.baseline_lib, a.candidate_lib, Path(__file__)]},
                  scratch_bytes={name: {cap: owner.info.scratch_bytes for cap, owner in bank.items()}
                                 for name, bank in owners.items()}, cases=[])
    cases = [(1, 'shared'), (6, 'shared'), (80, 'shared'), (81, 'shared'),
             (256, 'shared'), (1024, 'shared'), (1024, 'mixed'), (256, 'mixed'),
             (256, 'group_bound'), (256, 'zero'), (6, 'shared')]
    for rows, kind in cases:
        cap = 1 if rows == 1 else 80 if rows <= 80 else 256 if rows <= 256 else 1024
        graphs = {}
        wire.zero_()
        wire[:, h:].fill_(127)
        ids.copy_(torch.arange(6, device='cuda').expand(capacity, 6))
        routing.fill_(1/6)
        for name, bank in owners.items():
            owner = bank[cap]
            owner.run(rows)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                owner.run(rows)
            graphs[name] = graph
        metrics = []
        first = None
        for cycle in range(4):
            if cycle < 2:
                x = torch.randn((capacity, h), device='cuda').bfloat16().float().reshape(capacity, h//32, 32)
                if kind == 'zero':
                    x.zero_()
                exp = torch.ceil(torch.log2(x.abs().amax(-1).clamp_min(1e-4)/448))
                wire[:, :h].view(torch.uint32).copy_((x/torch.exp2(exp)[..., None]).to(torch.float8_e4m3fn).view(torch.uint32).reshape(capacity, h//4))
                wire[:, h:].copy_((exp+127).to(torch.uint8))
                chosen = torch.arange(6, device='cuda').expand(capacity, 6)
                if kind == 'mixed':
                    chosen = (torch.arange(capacity, device='cuda')[:, None]*7 + torch.arange(6, device='cuda')) % a.experts
                ids.copy_(chosen)
                if kind == 'group_bound':
                    ids.zero_()
                    ids.flatten()[:a.experts].copy_(torch.arange(a.experts, device='cuda'))
                routing.copy_(torch.rand_like(routing))
                routing.div_(routing.sum(-1, keepdim=True))
            for name, graph in graphs.items():
                owners[name][cap].output.fill_(12345)
                before = torch.cuda.memory_allocated()
                graph.replay()
                assert torch.cuda.memory_allocated() == before
            torch.cuda.synchronize()
            expected_routes = owners['baseline'][cap].output[:rows*6].reshape(rows, 6, h)
            expected = torch.zeros(rows, h, device='cuda')
            for j in range(6):
                expected.add_(expected_routes[:, j])
            candidate = owners['candidate'][cap]
            if candidate.token_accumulation:
                actual = candidate.output[:rows]
                check(token_cast(actual.data_ptr(), compact.data_ptr(), rows, torch.cuda.current_stream().cuda_stream))
                assert torch.equal(compact[:rows], actual.bfloat16())
            else:
                assert torch.equal(candidate.output[:rows*6], owners['baseline'][cap].output[:rows*6])
                actual = torch.zeros_like(expected)
                for j in range(6):
                    actual.add_(candidate.output[:rows*6].reshape(rows,6,h)[:,j])
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-5)
            assert (candidate.output[rows if candidate.token_accumulation else rows*6:] == 12345).all()
            if kind == 'zero':
                assert torch.count_nonzero(actual) == 0
            else:
                assert torch.count_nonzero(actual) > 0
            diff = actual - expected
            item = dict(cycle=cycle, max_abs=diff.abs().max().item(),
                        rel_l2=(diff.norm()/expected.norm().clamp_min(1e-30)).item(),
                        bf16_different=int((actual.bfloat16()!=expected.bfloat16()).sum()), elements=actual.numel())
            if cycle == 1:
                first = actual.clone()
            elif cycle > 1:
                item['repeat_max_abs'] = (actual-first).abs().max().item()
                item['repeat_bf16_different'] = int((actual.bfloat16()!=first.bfloat16()).sum())
            metrics.append(item)
        samples = {name: [] for name in graphs}
        for iteration in range(6):
            for name in (['baseline','candidate'] if iteration%2 else ['candidate','baseline']):
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(5):
                    graphs[name].replay()
                end.record()
                end.synchronize()
                samples[name].append(begin.elapsed_time(end)*1000/5)
        record = dict(rows=rows, kind=kind, capacity=cap, metrics=metrics,
                      graph_us=samples, median_us={k: statistics.median(v) for k,v in samples.items()})
        report['cases'].append(record)
        a.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(record), flush=True)
        for graph in graphs.values():
            graph.reset()


if __name__ == '__main__':
    main()
