import argparse
import ctypes as C
import json
import statistics
import torch
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Compare mHC small-batch projection dispatch against a native baseline"
)
parser.add_argument("--baseline", required=True)
parser.add_argument("--candidate", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--snapshot", type=Path)
args = parser.parse_args()
torch.manual_seed(731)
libs = [C.CDLL(args.baseline), C.CDLL(args.candidate)]
for lib in libs:
    lib.ds41rt_v41_hc_mixes.argtypes = [C.c_void_p] * 7 + [C.c_int32, C.c_void_p]
s = torch.cuda.Stream()
results = []
for rows in [1, 2, 6, 16, 17, 80, 256, 4096]:
    r = torch.empty((rows, 20480), device="cuda", dtype=torch.bfloat16)
    fn = torch.randn((24, 20480), device="cuda", dtype=torch.float32) * 0.01
    scale = torch.tensor([0.1, 0.1, 0.1], device="cuda")
    base = torch.randn(24, device="cuda")
    outputs = [
        [torch.empty((rows, n), device="cuda") for n in [4, 4, 16]] for _ in libs
    ]
    graphs = []
    r.normal_()
    torch.cuda.synchronize()
    for lib, out in zip(libs, outputs):
        graph = torch.cuda.CUDAGraph()
        ptrs = [t.data_ptr() for t in [r, fn, scale, base] + out]
        assert lib.ds41rt_v41_hc_mixes(*ptrs, rows, s.cuda_stream) == 0
        s.synchronize()
        with torch.cuda.graph(graph, stream=s):
            assert lib.ds41rt_v41_hc_mixes(*ptrs, rows, s.cuda_stream) == 0
        graphs.append(graph)
    for kind in ["random", "zero", "large", "small"]:
        with torch.cuda.stream(s):
            if kind == "zero":
                r.zero_()
            else:
                r.normal_().mul_({"random": 1, "large": 1000, "small": 1e-12}[kind])
            for graph in graphs:
                graph.replay()
        s.synchronize()
        for a, b in zip(*outputs):
            assert torch.equal(a.view(torch.int32), b.view(torch.int32)), (
                rows,
                kind,
                (a - b).abs().max().item(),
            )
    samples = [[], []]
    for i in range(24):
        for mode in [0, 1] if i % 2 else [1, 0]:
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            with torch.cuda.stream(s):
                start.record()
                for _ in range(10):
                    graphs[mode].replay()
                end.record()
            end.synchronize()
            if i >= 4:
                samples[mode].append(start.elapsed_time(end) * 100)
    item = dict(
        rows=rows,
        exact_cases=4,
        before_us=statistics.median(samples[0]),
        candidate_us=statistics.median(samples[1]),
    )
    results.append(item)
    print(json.dumps(item), flush=True)
real_checks = []
if args.snapshot:
    from safetensors import safe_open

    index = json.loads((args.snapshot / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    names = sorted(k for k in index if k.endswith(("hc_attn_fn", "hc_ffn_fn")))
    assert names, "checkpoint has no mHC weights"
    for name in names:
        tensors = []
        for key in [name, name[:-2] + "scale", name[:-2] + "base"]:
            with safe_open(
                args.snapshot / index[key], framework="pt", device="cpu"
            ) as f:
                tensors.append(f.get_tensor(key).float().cuda().contiguous())
        fn, scale, base = tensors
        assert fn.numel() == 24 * 20480 and scale.numel() == 3 and base.numel() == 24
        for rows in [1, 6, 16]:
            r = torch.randn((rows, 20480), device="cuda").bfloat16()
            outputs = [
                [torch.empty((rows, n), device="cuda") for n in [4, 4, 16]]
                for _ in libs
            ]
            torch.cuda.synchronize()
            for lib, out in zip(libs, outputs):
                assert (
                    lib.ds41rt_v41_hc_mixes(
                        *[t.data_ptr() for t in [r, fn, scale, base] + out],
                        rows,
                        s.cuda_stream,
                    )
                    == 0
                )
            s.synchronize()
            for a, b in zip(*outputs):
                assert torch.equal(a.view(torch.int32), b.view(torch.int32)), (
                    name,
                    rows,
                )
            real_checks.append(dict(name=name, rows=rows, exact=True))
    print(json.dumps(dict(real_weight_exact_checks=len(real_checks))), flush=True)
args.output.write_text(
    json.dumps(dict(synthetic=results, real_weights=real_checks), indent=2) + "\n"
)
