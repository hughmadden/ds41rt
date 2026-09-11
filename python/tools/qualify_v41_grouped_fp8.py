#!/usr/bin/env python3
"""Qualify native grouped FP8 WO-A against independently decoded operands."""

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
parser.add_argument("--all-layers", action="store_true")
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


lib = C.CDLL(str(args.native_lib))


def bind(name, types):
    f = getattr(lib, name)
    f.argtypes = types
    f.restype = I32

    def call(*args):
        result = f(*args)
        assert result == 0, (name, result)

    return call


create = bind("ds41rt_v41_grouped_output_create", [P, C.c_uint64, C.POINTER(P)])
group = bind("ds41rt_v41_grouped_output_launch", [P, P, P, P, I32, P])
destroy = bind("ds41rt_v41_grouped_output_destroy", [P])
dequant = bind("ds41rt_v41_grouped_output_dequant", [P, P, P, P])
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


stream = torch.cuda.Stream()
workspace = torch.empty(4 * 1024 * 1024, device="cuda", dtype=torch.uint8)
handle = P()
create(workspace.data_ptr(), workspace.numel(), C.byref(handle))
flush = torch.empty(256 * 1024 * 1024, device="cuda", dtype=torch.uint8)
results = []
prefixes = (
    [f"layers.{i}" for i in range(40)] + [f"mtp.{i}" for i in range(3)]
    if args.all_layers
    else ["layers.0"]
)
try:
    for prefix in prefixes:
        weight = load(prefix + ".attn.wo_a.weight")
        scale = load(prefix + ".attn.wo_a.scale")
        bf16 = torch.empty((8, 1024, 4096), device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize()
        dequant(
            weight.data_ptr(), scale.data_ptr(), bf16.data_ptr(), stream.cuda_stream
        )
        stream.synchronize()
        packed = torch.empty(1048576, device="cuda", dtype=torch.uint8)
        pack(scale.data_ptr(), packed.data_ptr(), 32768, 8192, stream.cuda_stream)
        stream.synchronize()
        cases = (
            [
                (cap, m)
                for cap in [1, 16, 80, 256, 1024, 4096]
                for m in sorted({1, min(cap, 6), cap})
            ]
            if prefix == "layers.0"
            else [(16, 6)]
        )
        for capacity, rows in cases:
            info = Info()
            kernel = P()
            info_fn(capacity, 32768, 8192, C.byref(info))
            init(capacity, 32768, 8192, C.byref(kernel))
            assert info.weight_scales == 1048576
            scratch = torch.empty(info.scratch, device="cuda", dtype=torch.uint8)
            alpha = torch.empty(1, device="cuda")
            x = (torch.randn((rows, 8, 4096), device="cuda") * 0.2).bfloat16()
            y = torch.empty((rows, 8, 1024), device="cuda", dtype=torch.bfloat16)
            baseline = torch.empty_like(y)
            torch.cuda.synchronize()
            scratch_init(
                kernel,
                scratch.data_ptr(),
                scratch.numel(),
                alpha.data_ptr(),
                stream.cuda_stream,
            )

            def candidate():
                linear(
                    kernel,
                    x.data_ptr(),
                    weight.data_ptr(),
                    packed.data_ptr(),
                    scratch.data_ptr(),
                    scratch.numel(),
                    alpha.data_ptr(),
                    y.data_ptr(),
                    rows,
                    stream.cuda_stream,
                )

            def before():
                group(
                    handle,
                    x.data_ptr(),
                    bf16.data_ptr(),
                    baseline.data_ptr(),
                    rows,
                    stream.cuda_stream,
                )

            candidate()
            before()
            stream.synchronize()
            graphs = []
            for fn in [before, candidate]:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    fn()
                graphs.append(graph)
            for pattern in ["random", "tiny", "zero"]:
                if pattern == "tiny":
                    x.mul_(1e-8)
                elif pattern == "zero":
                    x.zero_()
                torch.cuda.synchronize()
                for graph in graphs:
                    graph.replay()
                stream.synchronize()
                # Independent per-group FP32 GEMMs on per-32-column quantized inputs.
                blocks = x.float().reshape(rows, 8, 128, 32)
                amax = blocks.abs().amax(-1, keepdim=True)
                scales = torch.exp2(
                    torch.ceil(torch.log2(torch.where(amax > 0, amax / 448, 1.0)))
                )
                quantized = (blocks / scales).to(torch.float8_e4m3fn).float() * scales
                operand = quantized.reshape(rows, 8, 4096)
                oracle = torch.stack(
                    [operand[:, g] @ bf16[g].float().T for g in range(8)], dim=1
                ).bfloat16()
                torch.testing.assert_close(y, oracle, rtol=0.008, atol=0.002)
                if pattern == "zero":
                    assert torch.count_nonzero(y).item() == 0
                else:
                    assert torch.count_nonzero(y).item() > 0
                err = y.float() - baseline.float()
                result = dict(
                    layer=prefix,
                    capacity=capacity,
                    rows=rows,
                    pattern=pattern,
                    quantized_oracle_pass=True,
                    max_abs=err.abs().max().item(),
                    rel_l2=(err.norm() / baseline.float().norm()).item()
                    if pattern != "zero"
                    else 0,
                )
                if pattern == "random" and prefix == "layers.0":
                    samples = [[], []]
                    for iteration in range(24):
                        for index in [0, 1] if iteration % 2 else [1, 0]:
                            a = torch.cuda.Event(enable_timing=True)
                            b = torch.cuda.Event(enable_timing=True)
                            with torch.cuda.stream(stream):
                                flush.zero_()
                                a.record()
                                graphs[index].replay()
                                b.record()
                            b.synchronize()
                            if iteration >= 4:
                                samples[index].append(a.elapsed_time(b) * 1000)
                    result.update(
                        bf16_us=statistics.median(samples[0]),
                        fp8_us=statistics.median(samples[1]),
                        samples_us=samples,
                    )
                results.append(result)
                print(json.dumps(result), flush=True)
            del graphs
finally:
    stream.synchronize()
    destroy(handle)
args.output.write_text(
    json.dumps(
        dict(
            scope="Native component qualification; random inputs, not full-model quality. Alternating graph timings with 256 MiB zeroing before each launch; no clock admission.",
            native_sha256=hashlib.sha256(args.native_lib.read_bytes()).hexdigest(),
            qualifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            device=torch.cuda.get_device_name(args.device),
            results=results,
        ),
        indent=2,
    )
    + "\n"
)
