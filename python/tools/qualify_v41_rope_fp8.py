#!/usr/bin/env python3
import argparse
import ctypes as C
import hashlib
import json
import statistics
import torch
from pathlib import Path
from safetensors import safe_open

parser = argparse.ArgumentParser(
    description="Native fused inverse-RoPE/FP8 grouped projection qualification"
)
parser.add_argument("--native-lib", type=Path, required=True)
parser.add_argument("--snapshot", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--device", type=int, default=0)
args = parser.parse_args()
torch.cuda.set_device(args.device)
torch.manual_seed(41411)
P, I32 = C.c_void_p, C.c_int32


class Info(C.Structure):
    _fields_ = [(n, C.c_uint32) for n in ["abi", "capacity", "k", "n"]] + [
        (n, C.c_uint64)
        for n in ["scratch", "values", "row_scales", "mma_scales", "weight_scales"]
    ]


native = C.CDLL(str(args.native_lib))


def bind(name, types):
    f = getattr(native, name)
    f.argtypes = types
    f.restype = I32
    return f


rope = bind("ds41rt_v41_attention_rope", [P, P, P, I32, I32, I32, P])
info_fn = bind("ds41rt_v41_fp8_matrix_info", [I32, I32, I32, C.POINTER(Info)])
init = bind("ds41rt_v41_fp8_matrix_initialize", [I32, I32, I32, C.POINTER(P)])
pack = bind("ds41rt_v41_fp8_matrix_pack_scales", [P, P, I32, I32, P])
scratch_init = bind("ds41rt_v41_fp8_initialize_scratch", [P, P, C.c_uint64, P, P])
launch = bind(
    "ds41rt_v41_fp8_launch_rope", [P, P, P, P, P, P, C.c_uint64, P, P, I32, P]
)
weight_map = json.loads((args.snapshot / "model.safetensors.index.json").read_text())[
    "weight_map"
]


def load(name):
    with safe_open(args.snapshot / weight_map[name], framework="pt", device="cpu") as f:
        return f.get_tensor(name).cuda().contiguous()


weight = load("layers.0.attn.wo_a.weight")
scale = load("layers.0.attn.wo_a.scale")
packed = torch.empty(1048576, device="cuda", dtype=torch.uint8)
s = torch.cuda.Stream()
flush = torch.empty(256 * 1024 * 1024, device="cuda", dtype=torch.uint8)
results = []
torch.cuda.synchronize()
assert pack(scale.data_ptr(), packed.data_ptr(), 32768, 8192, s.cuda_stream) == 0
s.synchronize()
for rows in [1, 2, 6, 16, 80, 256, 1024, 4096]:
    capacity = 1 if rows == 1 else 16 if rows <= 16 else rows
    info = Info()
    kernel = P()
    assert info_fn(capacity, 32768, 8192, C.byref(info)) == 0
    assert init(capacity, 32768, 8192, C.byref(kernel)) == 0
    x = torch.randn((rows, 8, 4096), device="cuda", dtype=torch.bfloat16) * 0.2
    rotated = torch.empty_like(x)
    angles = torch.randn((rows, 32), device="cuda")
    freq = torch.stack([angles.cos(), angles.sin()], dim=-1).contiguous()
    scratch = [
        torch.empty(info.scratch, device="cuda", dtype=torch.uint8) for _ in range(2)
    ]
    output = [
        torch.empty((rows, 8, 1024), device="cuda", dtype=torch.bfloat16)
        for _ in range(2)
    ]
    alpha = torch.empty(1, device="cuda")
    torch.cuda.synchronize()
    for v in scratch:
        assert (
            scratch_init(
                kernel, v.data_ptr(), v.numel(), alpha.data_ptr(), s.cuda_stream
            )
            == 0
        )

    def before():
        assert (
            rope(
                x.data_ptr(),
                freq.data_ptr(),
                rotated.data_ptr(),
                rows,
                64,
                1,
                s.cuda_stream,
            )
            == 0
        )
        assert (
            launch(
                kernel,
                rotated.data_ptr(),
                None,
                weight.data_ptr(),
                packed.data_ptr(),
                scratch[0].data_ptr(),
                scratch[0].numel(),
                alpha.data_ptr(),
                output[0].data_ptr(),
                rows,
                s.cuda_stream,
            )
            == 0
        )

    def after():
        assert (
            launch(
                kernel,
                x.data_ptr(),
                freq.data_ptr(),
                weight.data_ptr(),
                packed.data_ptr(),
                scratch[1].data_ptr(),
                scratch[1].numel(),
                alpha.data_ptr(),
                output[1].data_ptr(),
                rows,
                s.cuda_stream,
            )
            == 0
        )

    # Invalid row counts, undersized scratch, and frequency alias reject before enqueue.
    for bad_rows, bad_bytes, bad_freq in [
        (0, info.scratch, freq.data_ptr()),
        (capacity + 1, info.scratch, freq.data_ptr()),
        (rows, info.scratch - 1, freq.data_ptr()),
        (rows, info.scratch, x.data_ptr()),
    ]:
        assert (
            launch(
                kernel,
                x.data_ptr(),
                bad_freq,
                weight.data_ptr(),
                packed.data_ptr(),
                scratch[1].data_ptr(),
                bad_bytes,
                alpha.data_ptr(),
                output[1].data_ptr(),
                bad_rows,
                s.cuda_stream,
            )
            != 0
        )
    graphs = []
    for fn in [before, after]:
        fn()
        s.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            fn()
        graphs.append(g)
    for case in ["random", "changed", "tiny", "zero"]:
        if case == "changed":
            x.normal_()
            freq.neg_()
        elif case == "tiny":
            x.mul_(1e-8)
        elif case == "zero":
            x.zero_()
        torch.cuda.synchronize()
        for g in graphs:
            g.replay()
        s.synchronize()
        assert torch.equal(output[0], output[1]), (rows, case, "BF16 output")
        for start, length in [
            (info.values, rows * 32768),
            (info.row_scales, rows * 1024),
            (info.mma_scales, 8 * ((rows + 127) // 128) * 16384),
        ]:
            assert torch.equal(
                scratch[0][start : start + length], scratch[1][start : start + length]
            ), (rows, case, "quantized operands", start)
        result = dict(rows=rows, case=case, exact=True)
        if case == "random":
            samples = [[], []]
            for it in range(24):
                for arm in [0, 1] if it % 2 else [1, 0]:
                    a = torch.cuda.Event(enable_timing=True)
                    b = torch.cuda.Event(enable_timing=True)
                    with torch.cuda.stream(s):
                        flush.zero_()
                        a.record()
                        graphs[arm].replay()
                        b.record()
                    b.synchronize()
                    if it >= 4:
                        samples[arm].append(a.elapsed_time(b) * 1000)
            result.update(
                before_us=statistics.median(samples[0]),
                after_us=statistics.median(samples[1]),
                samples_us=samples,
            )
        results.append(result)
        print(json.dumps(result), flush=True)
    del graphs
args.output.write_text(
    json.dumps(
        dict(
            scope="Native graph-replay component qualification; alternating 256 MiB flush timings without clock admission.",
            native_sha256=hashlib.sha256(args.native_lib.read_bytes()).hexdigest(),
            qualifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            results=results,
        ),
        indent=2,
    )
    + "\n"
)
