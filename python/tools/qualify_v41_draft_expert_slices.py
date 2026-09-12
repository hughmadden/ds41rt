#!/usr/bin/env python3
"""Official dSpark weights: native persistent vs full grouped slice path.

Synthetic BF16 inputs and controlled shared/dispersed routing; this is a
component comparison, not live model activation or API qualification.
"""
import argparse
import ctypes as C
import hashlib
import json
import statistics
import subprocess
from contextlib import ExitStack
from pathlib import Path
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from safetensors import safe_open
import _pinned_sparkinfer
from _v41_expert_native import Native, library, check, P, U
from b12x._lib.utils import current_cuda_stream
from b12x.moe._shared.kernels.w4a8_v41_slice import V41FusedSliceKernel
from b12x.moe._shared.kernels.v41_route_plan import V41RoutePlan, V41SliceReduce


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--snapshot', type=Path, required=True)
    ap.add_argument('--native-lib', type=Path, required=True)
    ap.add_argument('--stage', type=int, choices=range(3), default=0)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--no-timing', action='store_true')
    opt = ap.parse_args()
    torch.manual_seed(410123)
    capacity, experts, topk, h, n = 80, 128, 3, 5120, 2304
    records = []
    def emit(kind, **value):
        record = dict(kind=kind, **value)
        records.append(record)
        print(json.dumps(record), flush=True)
        opt.output.write_text(json.dumps(records, indent=2) + '\n')
    def gpu_state():
        return subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,pstate,clocks.sm,clocks.mem,clocks_event_reasons.active', '--format=csv'], text=True).strip()
    emit('gpu_start', state=gpu_state())
    emit('source', b12x=_pinned_sparkinfer.REVISION, stage=opt.stage,
         native_sha256=hashlib.sha256(opt.native_lib.read_bytes()).hexdigest(),
         script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         device=torch.cuda.get_device_name(), scope=__doc__)
    lib = library(str(opt.native_lib))
    for name, types in {
        'ds41rt_v41_expert_input_quant_initialize': [C.POINTER(P)],
        'ds41rt_v41_expert_input_quantize_async': [P, P, P, U, P],
        'ds41rt_v41_reduce_routes_async': [C.POINTER(P), P, P, U, U, U, P],
    }.items():
        fn = getattr(lib, name); fn.argtypes = types; fn.restype = C.c_int32
    quant = P(); check(lib.ds41rt_v41_expert_input_quant_initialize(C.byref(quant)))
    sizes = [n*h, n*h//16, n*h//2, n*h//32]
    weights = [torch.empty((experts, size), dtype=torch.uint8, device='cuda') for size in sizes]
    index = json.loads((opt.snapshot/'model.safetensors.index.json').read_text())['weight_map']
    with ExitStack() as stack:
        files = {}
        for expert in range(experts):
            sources = []
            for suffix in ('weight', 'scale'):
                for proj in ('w1', 'w3', 'w2'):
                    key = f'mtp.{opt.stage}.ffn.experts.{expert}.{proj}.{suffix}'
                    shard = index[key]
                    if shard not in files:
                        files[shard] = stack.enter_context(safe_open(opt.snapshot/shard, framework='pt', device='cpu'))
                    t = files[shard].get_tensor(key)
                    shape = ((h, n//2 if suffix == 'weight' else n//32) if proj == 'w2'
                             else (n, h//2 if suffix == 'weight' else h//32))
                    assert tuple(t.shape) == shape, (key, t.shape)
                    assert t.dtype == (torch.int8 if suffix == 'weight' else torch.float8_e8m0fnu), (key, t.dtype)
                    sources.append(t.view(torch.uint8).contiguous().cuda())
            check(lib.ds41rt_v41_pack_expert_async((P*6)(*[t.data_ptr() for t in sources]),
                (P*4)(*[t[expert].data_ptr() for t in weights]), n, torch.cuda.current_stream().cuda_stream))
        torch.cuda.synchronize()
    emit('loaded', experts=experts, bytes=sum(t.numel() for t in weights))
    x = torch.empty(capacity, h, dtype=torch.bfloat16, device='cuda')
    wire = torch.empty(capacity, 5280, dtype=torch.uint8, device='cuda')
    ids = torch.empty(capacity, topk, dtype=torch.int32, device='cuda')
    routing = torch.empty(capacity, topk, device='cuda')
    native = Native(lib, capacity, weights, x, ids, routing, coordinator=True)
    routes = capacity * topk
    meta = torch.empty(routes, 19, dtype=torch.int32, device='cuda')
    live = torch.empty(1, dtype=torch.int32, device='cuda')
    rw = torch.empty(routes, device='cuda')
    inverse = torch.empty(routes, dtype=torch.int32, device='cuda')
    buffers = [ids.flatten(), routing.flatten(), live,
               torch.empty(experts*routes, dtype=torch.int32, device='cuda'),
               torch.empty(experts, dtype=torch.int32, device='cuda'),
               torch.empty(experts, 2, dtype=torch.int32, device='cuda'), meta, rw, inverse]
    view = lambda t: from_dlpack(t, assumed_align=16)
    pa = list(map(view, buffers))
    planner = cute.compile(V41RoutePlan(capacity, experts, topk), *pa, current_cuda_stream())
    variants = {}
    for width in (64, 128, 192):
        partial = torch.empty(n//width, routes, h, device='cuda')
        out = torch.empty(routes, h, device='cuda')
        args = list(map(view, [wire[:, :h].view(torch.uint32), wire[:, h:],
                   *[t.view(torch.uint32).flatten() for t in weights], rw, partial]))
        fn = cute.compile(V41FusedSliceKernel(width, grouped=True, intermediate=n),
             *args, cutlass.Int32(capacity), current_cuda_stream(), pa[6], cutlass.Int32(routes))
        ra = [args[-1], view(out), pa[8], pa[2]]
        reducer = cute.compile(V41SliceReduce(width, capacity, topk, intermediate=n),
                              *ra, current_cuda_stream())
        variants[width] = (partial, out, args, fn, ra, reducer)
    shared = torch.zeros(capacity, h, device='cuda', dtype=torch.bfloat16)
    outputs = {arm: torch.empty_like(shared) for arm in ('native', 64, 128, 192)}
    def finish(route_output, out, rows):
        check(lib.ds41rt_v41_reduce_routes_async((P*4)(route_output.data_ptr(), None, None, None),
              shared.data_ptr(), out.data_ptr(), rows, 1, topk, torch.cuda.current_stream().cuda_stream))
    for case in ('shared', 'dispersed'):
        for rows in (5, 15, 40):
            live.fill_(rows)
            def populate(seed):
                torch.manual_seed(seed)
                x.normal_(0, .5)
                if case == 'shared':
                    ids.copy_((torch.arange(topk, device='cuda') + seed % 126).expand(capacity, topk))
                else:
                    ids.copy_((torch.arange(routes, device='cuda').reshape(capacity, topk)+seed) % experts)
                routing.uniform_(.1, 1)
                routing.mul_(1.5 / routing.sum(-1, keepdim=True))
            populate(41)
            graphs = {}
            for arm in outputs:
                def run():
                    if arm == 'native':
                        native.run(rows); result = native.output
                    else:
                        partial, result, args, fn, ra, reducer = variants[arm]
                        check(lib.ds41rt_v41_expert_input_quantize_async(quant, x.data_ptr(), wire.data_ptr(), rows,
                              torch.cuda.current_stream().cuda_stream))
                        planner(*pa, current_cuda_stream())
                        fn(*args, rows, current_cuda_stream(), pa[6], min(rows*topk, experts + max(rows*topk-experts, 0)//16))
                        reducer(*ra, current_cuda_stream(), rows)
                    finish(result, outputs[arm], rows)
                run(); torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph): run()
                graphs[arm] = graph
            # Changed input/IDs/routing must propagate through the same graphs.
            for seed in (42, 77):
                populate(seed)
                before = torch.cuda.memory_allocated()
                for output in outputs.values(): output.fill_(12345)
                for graph in graphs.values(): graph.replay()
                assert torch.cuda.memory_allocated() == before
                torch.cuda.synchronize()
                ref = outputs['native'][:rows].float()
                assert torch.isfinite(ref).all() and ref.norm() > 0
                for arm in variants:
                    actual = outputs[arm][:rows].float()
                    rel = ((actual-ref).norm()/ref.norm()).item()
                    cosine = torch.nn.functional.cosine_similarity(actual.flatten(), ref.flatten(), dim=0).item()
                    native_routes = native.output[:rows*topk]
                    candidate_routes = variants[arm][1][:rows*topk]
                    route_rel = ((candidate_routes-native_routes).norm()/native_routes.norm()).item()
                    assert torch.isfinite(candidate_routes).all() and route_rel < 1e-5
                    assert bool((outputs[arm][rows:] == 12345).all())
                    emit('check', route_rel_l2=route_rel, case=case, rows=rows, seed=seed, width=arm,
                         rel_l2=rel, cosine=cosine, mismatches=int((actual != ref).sum()))
                    assert torch.isfinite(actual).all() and rel < .0001 and cosine > .99999
            if not opt.no_timing:
                samples = {arm: [] for arm in graphs}
                for cycle in range(6):
                    order = list(graphs) if cycle % 2 == 0 else list(reversed(graphs))
                    for arm in order:
                        graph = graphs[arm]
                        for _ in range(5): graph.replay()
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(30): graph.replay()
                        end.record(); end.synchronize()
                        samples[arm].append(start.elapsed_time(end)*1000/30)
                emit('gpu_after_timing', case=case, rows=rows, state=gpu_state())
                emit('timing', case=case, rows=rows, samples_us=samples,
                     medians_us={str(k): statistics.median(v) for k,v in samples.items()})
            for graph in graphs.values(): graph.reset()

if __name__ == '__main__': main()
