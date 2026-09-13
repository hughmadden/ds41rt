#!/usr/bin/env python3
"""Compare two native RTX expert halves with the unsplit numerical oracle."""
import argparse
import ctypes as C
import json
import torch
import _pinned_sparkinfer
from _v41_expert_native import Native, P, check, library
from tests.moe.test_v41_expert_numerics import reference


def wire_input(x, wire):
    blocks = x.float().reshape(x.shape[0], -1, 32)
    scale = torch.exp2(torch.ceil(torch.log2(blocks.abs().amax(-1).clamp_min(1e-4) / 448)))
    wire[:, :5120].copy_((blocks / scale[..., None]).to(torch.float8_e4m3fn).view(torch.uint8).reshape(-1, 5120))
    wire[:, 5120:].copy_(scale.to(torch.float8_e8m0fnu).view(torch.uint8))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-lib", required=True)
    parser.add_argument("--rows", default="1,16", help="Comma-separated exported capacities")
    args = parser.parse_args()
    capacities = [int(value) for value in args.rows.split(",")]
    assert capacities and all(value in (1, 16, 80, 256, 1024, 4096) for value in capacities)
    assert torch.cuda.device_count() >= 2
    lib = library(args.native_lib, tp2=True)
    reduce = lib.ds41rt_v41_reduce_tp2_experts_async
    reduce.argtypes = [P, P, P, C.c_uint32, C.c_uint32, P]
    torch.manual_seed(41152)
    experts, hidden, width = 8, 5120, 2304
    weights, scales = {}, {}
    for name, shape in [("w1", (experts, width, hidden//2)),
                        ("w3", (experts, width, hidden//2)),
                        ("w2", (experts, hidden, width//2))]:
        weights[name] = torch.randint(0, 256, shape, dtype=torch.uint8)
        scales[name] = torch.randint(121, 124, (*shape[:-1], shape[-1]//16), dtype=torch.uint8)
    states = []
    for rank in (0, 1):
        with torch.cuda.device(rank):
            packed = [torch.empty((384, size), device="cuda", dtype=torch.uint8)
                      for size in (5898240, 368640, 2949120, 184320)]
            for expert in range(experts):
                source = []
                for table in (weights, scales):
                    for name in ("w1", "w3", "w2"):
                        value = table[name][expert]
                        axis = 1 if name == "w2" else 0
                        source.append(value.chunk(2, dim=axis)[rank].contiguous().cuda())
                check(lib.ds41rt_v41_pack_expert_async(
                    (P*6)(*[t.data_ptr() for t in source]),
                    (P*4)(*[t[expert].data_ptr() for t in packed]), 1152,
                    torch.cuda.current_stream().cuda_stream))
            torch.cuda.synchronize()
            states.append(packed)
    results = []
    for rows in capacities:
        x = torch.randn(rows, hidden).mul_(0.5).bfloat16()
        ids = torch.rand(rows, experts).topk(6, -1).indices.int()
        routing = torch.rand(rows, 6)
        routing = routing / routing.sum(-1, keepdim=True) * 1.5
        owners = []
        for rank in (0, 1):
            with torch.cuda.device(rank):
                wire = torch.empty((rows, 5280), device="cuda", dtype=torch.uint8)
                live_ids, live_routing = ids.cuda(), routing.cuda()
                wire_input(x.cuda(), wire)
                native = Native(lib, rows, states[rank], wire, live_ids, live_routing, tp2=True)
                native.run(rows)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                capture_stream = torch.cuda.Stream(device=rank)
                with torch.cuda.graph(graph, stream=capture_stream): native.run(rows)
                owners.append((native, graph, wire, live_ids, live_routing, capture_stream))
        for changed in (False, True):
            if changed:
                x = x.mul(-0.5)
                ids = (ids + 1) % experts
                routing = routing.flip(-1).contiguous()
            rank_outputs = []
            for rank, (native, graph, wire, live_ids, live_routing, _) in enumerate(owners):
                with torch.cuda.device(rank):
                    wire_input(x.cuda(), wire)
                    live_ids.copy_(ids)
                    live_routing.copy_(routing)
                    before = torch.cuda.memory_allocated()
                    graph.replay()
                    assert torch.cuda.memory_allocated() == before
                    rank_outputs.append(native.output.clone())
            with torch.cuda.device(0):
                token_sums = owners[0][0].token_accumulation
                assert owners[1][0].token_accumulation == token_sums
                peer = rank_outputs[1].to(device="cuda:0")
                result = torch.empty((rows, hidden), dtype=torch.bfloat16, device="cuda:0")
                check(reduce(rank_outputs[0].data_ptr(), peer.data_ptr(), result.data_ptr(),
                             rows, int(token_sums), torch.cuda.current_stream().cuda_stream))
                actual = result.float().cpu()
                expected = reference(x.cuda(), ids.cuda(), routing.cuda(),
                    {k: v.cuda() for k, v in weights.items()},
                    {k: v.cuda() for k, v in scales.items()}).cpu()
            assert torch.isfinite(actual).all() and actual.norm() > 0
            # Large prefill vectors need FP64 metric accumulation: FP32 CPU
            # reductions can otherwise report an impossible cosine above one.
            measured, oracle = actual.double(), expected.double()
            relative = ((measured-oracle).norm()/oracle.norm()).item()
            cosine = torch.nn.functional.cosine_similarity(measured.flatten(), oracle.flatten(), dim=0).item()
            assert relative < 0.01 and cosine > 0.9999, (rows, changed, relative, cosine)
            results.append(dict(rows=rows, changed=changed, token_sums=token_sums,
                                relative_l2=relative, cosine=cosine))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
