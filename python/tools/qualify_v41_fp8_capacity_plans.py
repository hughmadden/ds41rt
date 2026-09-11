#!/usr/bin/env python3
"""Compare preloaded FP8 capacity plans on official weights and alternating live shapes."""

import argparse
import ctypes as C
import hashlib
import json
import statistics
import torch
from pathlib import Path
from safetensors import safe_open

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--native-lib", type=Path, required=True)
parser.add_argument("--snapshot", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--prefixes", nargs="+")
args = parser.parse_args()
torch.cuda.set_device(args.device)
torch.manual_seed(41410)
torch.backends.cuda.matmul.allow_tf32 = False
P, I32 = C.c_void_p, C.c_int32


class Info(C.Structure):
    _fields_ = [(n, C.c_uint32) for n in ["abi", "capacity", "k", "n"]] + [
        (n, C.c_uint64)
        for n in ["scratch", "values", "row_scales", "mma_scales", "weight_scales"]
    ]


native_sha256 = hashlib.sha256(args.native_lib.read_bytes()).hexdigest()
lib = C.CDLL(str(args.native_lib))


def bind(name, types):
    f = getattr(lib, name)
    f.argtypes = types
    f.restype = I32

    def call(*args):
        result = f(*args)
        assert result == 0, (name, result)

    return call


info_fn = bind("ds41rt_v41_fp8_matrix_info", [I32, I32, I32, C.POINTER(Info)])
init = bind("ds41rt_v41_fp8_matrix_initialize", [I32, I32, I32, C.POINTER(P)])
pack = bind("ds41rt_v41_fp8_matrix_pack_scales", [P, P, I32, I32, P])
scratch_init = bind("ds41rt_v41_fp8_initialize_scratch", [P, P, C.c_uint64, P, P])
linear = bind("ds41rt_v41_fp8_launch", [P, P, P, P, P, C.c_uint64, P, P, I32, P])
root = args.snapshot
weight_map = json.loads((root / "model.safetensors.index.json").read_text())[
    "weight_map"
]


def load(name):
    with safe_open(root / weight_map[name], framework="pt", device="cpu") as f:
        return f.get_tensor(name).cuda().contiguous()


prefixes = [
    'layers.0.attn.wq_a', 'layers.0.attn.wq_b',
    'layers.0.attn.wo_a', 'layers.0.attn.wo_b', 'layers.0.attn.wkv',
    'layers.0.ffn.shared_experts.w1', 'layers.0.ffn.shared_experts.w2',
    'layers.8.attn.indexer.wq_b', 'layers.1.engram.wkv', 'mtp.0.main_proj',
]
if args.prefixes:
    assert set(args.prefixes) <= set(prefixes), "unknown projection prefix"
    prefixes = args.prefixes
records = []
stream = torch.cuda.current_stream().cuda_stream
for prefix in prefixes:
    weight, scale = load(prefix + '.weight'), load(prefix + '.scale')
    n, k = (8192, 32768) if prefix.endswith('.wo_a') else weight.shape
    infos, handles, offsets = {}, {}, {}
    total = 0
    for capacity in [1, 16, 80, 4096]:
        info, handle = Info(), P()
        info_fn(capacity, k, n, C.byref(info))
        assert (info.abi, info.capacity, info.k, info.n) == (1, capacity, k, n)
        init(capacity, k, n, C.byref(handle))
        infos[capacity], handles[capacity] = info, handle
        offsets[capacity] = total + 256
        total += 512 + ((info.scratch + 255) // 256) * 256
    pool = torch.full((total,), 165, device='cuda', dtype=torch.uint8)
    alpha = torch.empty(1, device='cuda', dtype=torch.float32)
    packed = torch.empty(infos[4096].weight_scales, device='cuda', dtype=torch.uint8)
    pack(scale.data_ptr(), packed.data_ptr(), k, n, stream)
    for capacity in infos:
        scratch_init(handles[capacity], pool.data_ptr() + offsets[capacity], infos[capacity].scratch, alpha.data_ptr(), stream)
    baseline_scratch = torch.empty(infos[4096].scratch, device='cuda', dtype=torch.uint8)
    scratch_init(handles[4096], baseline_scratch.data_ptr(), baseline_scratch.numel(), alpha.data_ptr(), stream)
    x = torch.empty((4096, k), device='cuda', dtype=torch.bfloat16)
    output = torch.empty((4096, n), device='cuda', dtype=torch.bfloat16)
    baseline = torch.empty_like(output)
    torch.cuda.synchronize()
    for rows in [1, 6, 16, 17, 80, 81, 256, 1024, 4096, 6, 1]:
        selected = next(capacity for capacity in infos if rows <= capacity)
        def run(selected_arm):
            handle = handles[selected] if selected_arm else handles[4096]
            ptr = pool.data_ptr() + offsets[selected] if selected_arm else baseline_scratch.data_ptr()
            size = infos[selected].scratch if selected_arm else baseline_scratch.numel()
            dest = output if selected_arm else baseline
            linear(handle, x.data_ptr(), weight.data_ptr(), packed.data_ptr(), ptr, size, alpha.data_ptr(), dest.data_ptr(), rows, torch.cuda.current_stream().cuda_stream)
        graphs = []
        x.normal_()
        for arm in [False, True]:
            run(arm)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run(arm)
            graphs.append(graph)
        errors = []
        for cycle in range(2):
            x.normal_(std=1 if cycle == 0 else 0.25)
            output.fill_(123)
            before = torch.cuda.memory_allocated()
            for graph in graphs:
                graph.replay()
            assert torch.cuda.memory_allocated() == before
            torch.cuda.synchronize()
            ref, actual = baseline[:rows].float(), output[:rows].float()
            assert torch.isfinite(ref).all() and torch.isfinite(actual).all()
            assert ref.norm() > 0 and actual.norm() > 0
            error = (actual - ref).abs()
            rel_l2 = float(error.norm() / ref.norm())
            max_abs = float(error.max())
            # Two BF16 steps at the tensor's maximum magnitude, plus a tiny
            # absolute floor for cancellation. Also constrain aggregate error.
            step = 2.0 ** (int(torch.floor(torch.log2(ref.abs().max())).item()) - 7)
            assert rel_l2 <= 1e-3 and max_abs <= 2 * step + 1e-4, (prefix, rows, selected, rel_l2, max_abs)
            assert (output[rows:] == 123).all()
            for capacity in infos:
                start, size = offsets[capacity], infos[capacity].scratch
                end = start + ((size + 255) // 256) * 256
                assert (pool[start-256:start] == 165).all()
                assert (pool[start+size:end+256] == 165).all()
            errors.append(dict(relative_l2=rel_l2, max_abs=max_abs, changed_fraction=float((error != 0).float().mean())))
        samples = {'baseline': [], 'selected': []}
        if rows in [1, 6, 80, 4096]:
            for iteration in range(8):
                for arm in ([0, 1] if iteration % 2 == 0 else [1, 0]):
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record()
                    for _ in range(10):
                        graphs[arm].replay()
                    end.record(); end.synchronize()
                    samples[['baseline', 'selected'][arm]].append(begin.elapsed_time(end) * 100)
        record = dict(prefix=prefix, k=k, n=n, rows=rows, selected_capacity=selected, errors=errors,
                      graph_us=samples, median_us={key:statistics.median(values) for key,values in samples.items() if values},
                      total_disjoint_scratch_bytes=total, baseline_scratch_bytes=infos[4096].scratch)
        records.append(record)
        print(json.dumps({key:record[key] for key in ['prefix','rows','selected_capacity','errors','median_us']}), flush=True)
        args.output.write_text(json.dumps(dict(scope='Official weights with synthetic activations; preloaded native C ABI kernels and disjoint scratch; bounded numerical comparison to the existing capacity4096 plan, not a full-model quality gate or a Rust dispatch test.', native_sha256=native_sha256, seed=41410, prefixes=prefixes, records=records),indent=2)+'\n')
        for graph in graphs:
            graph.reset()
    del weight, scale, pool, alpha, packed, baseline_scratch, x, output, baseline
