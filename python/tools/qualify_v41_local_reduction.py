#!/usr/bin/env python3
"""Check the local routed/shared BF16 boundary and native argument rejection."""
import argparse
import ctypes as C
import json
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--native-lib', required=True)
args = parser.parse_args()
lib = C.CDLL(args.native_lib)
finish = lib.ds41rt_v41_finish_local_experts_async
finish.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_uint32, C.c_uint32, C.c_void_p]
finish.restype = C.c_int32
torch.manual_seed(4193)
records = []
for rows in (1, 17, 80, 256, 4096):
    for tokens in (False, True):
        routes = 1 if tokens else 6
        x = torch.randn(rows, routes, 5120, device='cuda') * 3
        shared = torch.randn(rows, 5120, dtype=torch.bfloat16, device='cuda')
        output = torch.empty_like(shared)
        def launch(destination=output, count=rows, kind=int(tokens), source=x.data_ptr(), shared_ptr=shared.data_ptr()):
            return finish(source, shared_ptr, destination.data_ptr(), count, kind, torch.cuda.current_stream().cuda_stream)
        assert launch() == 0
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            assert launch() == 0
        x.mul_(-0.75)
        shared.add_(0.125)
        expected = x[:, 0].clone()
        for route in range(1, routes):
            expected.add_(x[:, route])
        expected = (expected.bfloat16().float() + shared.float()).bfloat16()
        output.fill_(float('nan'))
        allocated = torch.cuda.memory_allocated()
        graph.replay()
        assert torch.cuda.memory_allocated() == allocated
        assert torch.equal(output, expected)
        # Shared may alias output exactly, but rejected calls must not mutate it.
        assert launch(destination=shared) == 0
        assert torch.equal(shared, expected)
        before = shared.clone()
        assert launch(destination=shared, count=0) != 0
        assert launch(destination=shared, count=4097) != 0
        assert launch(destination=shared, kind=2) != 0
        assert launch(destination=shared, source=x.data_ptr()+2) != 0
        assert launch(destination=shared, source=shared.data_ptr()) != 0
        assert launch(destination=shared, shared_ptr=shared.data_ptr()+2) != 0
        assert launch(destination=shared, source=(1 << 64)-4) != 0
        torch.cuda.synchronize()
        assert torch.equal(shared, before)
        graph.reset()
        records.append(dict(rows=rows, token_sums=tokens, exact=True, replay_allocation_free=True, rejected=7))
print(json.dumps(records, indent=2))
