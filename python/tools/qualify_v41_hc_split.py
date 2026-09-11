#!/usr/bin/env python3
"""Qualify native split-K mHC coefficients, scratch boundaries, and graph replay."""

import argparse
import ctypes as C
import hashlib
import json
import statistics
from pathlib import Path

import torch
from safetensors import safe_open

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--native", type=Path, required=True)
parser.add_argument("--snapshot", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
torch.manual_seed(731)
lib = C.CDLL(str(args.native))
old = lib.ds41rt_v41_hc_mixes
old.argtypes = [C.c_void_p] * 7 + [C.c_int32, C.c_void_p]
new = lib.ds41rt_v41_hc_mixes_workspace
new.argtypes = [C.c_void_p] * 8 + [C.c_uint64, C.c_int32, C.c_void_p]
initialize = lib.ds41rt_v41_hc_project_initialize
initialize.argtypes = []
assert initialize() == 0
assert initialize() == 0
s = torch.cuda.Stream()
capacity = 4096
r = torch.empty((capacity, 20480), device="cuda", dtype=torch.bfloat16)
fn = torch.randn((24, 20480), device="cuda") * 0.01
scale = torch.tensor([0.1, 0.1, 0.1], device="cuda")
base = torch.randn(24, device="cuda")
outputs = [
    [torch.empty((capacity, n), device="cuda") for n in (4, 4, 16)] for _ in range(2)
]
# Match the serving allocation to be reused: collapsed BF16 hidden, including
# untouched tails beyond the projection's required scratch extent.
scratch = torch.empty((capacity, 5120), device="cuda", dtype=torch.bfloat16)
scratch_words = scratch.view(torch.int32).flatten()
graphs = {}
synthetic = []
real = []


def launch(mode, rows):
    ptrs = [t.data_ptr() for t in [r, fn, scale, base] + outputs[mode]]
    if mode:
        return new(*ptrs, scratch.data_ptr(), scratch.numel() * 2, rows, s.cuda_stream)
    return old(*ptrs, rows, s.cuda_stream)


def check(rows, pattern):
    with torch.cuda.stream(s):
        r[:rows].normal_().mul_(
            {"random": 1, "zero": 0, "small": 1e-12, "large": 1e3}[pattern]
        )
        scratch_words.fill_(0x12345678)
        for out in outputs:
            for t in out:
                t.fill_(float("nan"))
        for graph in graphs[rows]:
            graph.replay()
    s.synchronize()
    errors = []
    for a, b in zip(*outputs):
        assert torch.isfinite(a[:rows]).all() and torch.isfinite(b[:rows]).all()
        torch.testing.assert_close(a[:rows], b[:rows], rtol=2e-5, atol=2e-6)
        if rows > 16:
            assert torch.equal(a[:rows], b[:rows])
        assert torch.isnan(a[rows:]).all() and torch.isnan(b[rows:]).all()
        errors.append((a[:rows] - b[:rows]).abs().max().item())
    written = rows * 1536 // 4 if rows <= 16 else 0
    assert (scratch_words[written:] == 0x12345678).all()
    return max(errors)


for rows in (1, 2, 6, 16, 17, 80, 256, 4096):
    with torch.cuda.stream(s):
        r[:rows].normal_()
    s.synchronize()
    graphs[rows] = []
    for mode in (0, 1):
        assert launch(mode, rows) == 0
        s.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=s):
            assert launch(mode, rows) == 0
        graphs[rows].append(graph)
    errors = {p: check(rows, p) for p in ("random", "zero", "small", "large")}
    with torch.cuda.stream(s):
        r[:rows].normal_()
    s.synchronize()
    samples = [[], []]
    for round_id in range(24):
        for mode in (0, 1) if round_id % 2 else (1, 0):
            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            with torch.cuda.stream(s):
                start.record()
                for _ in range(20):
                    graphs[rows][mode].replay()
                end.record()
            end.synchronize()
            if round_id >= 4:
                samples[mode].append(start.elapsed_time(end) * 50)
    item = dict(
        rows=rows,
        max_abs=errors,
        median_us=[statistics.median(v) for v in samples],
        samples_us=samples,
    )
    synthetic.append(item)
    print(json.dumps({k: v for k, v in item.items() if k != "samples_us"}), flush=True)

# Fail before any GPU write: extent, alignment, overflow, and every scratch alias.
ptrs = [t.data_ptr() for t in [r, fn, scale, base] + outputs[1]]
with torch.cuda.stream(s):
    for t in outputs[1]:
        t.fill_(float("nan"))
s.synchronize()
bad = [
    (scratch.data_ptr(), 1535),
    (scratch.data_ptr() + 2, 1536),
    ((1 << 64) - 16, 1536),
]
bad += [(p, 1536) for p in ptrs]
for pointer, size in bad:
    assert new(*ptrs, pointer, size, 1, s.cuda_stream) != 0
for rows in (0, -1, 4097):
    assert new(*ptrs, scratch.data_ptr(), scratch.numel() * 2, rows, s.cuda_stream) != 0
s.synchronize()
assert all(torch.isnan(t).all() for t in outputs[1])

index = json.loads((args.snapshot / "model.safetensors.index.json").read_text())[
    "weight_map"
]
names = sorted(n for n in index if n.endswith(("hc_attn_fn", "hc_ffn_fn")))
assert len(names) == 86, len(names)
for name in names:
    for key, dst in zip(
        (name, name[:-2] + "scale", name[:-2] + "base"), (fn, scale, base)
    ):
        with safe_open(
            args.snapshot / index[key], framework="pt", device="cpu"
        ) as handle:
            with torch.cuda.stream(s):
                dst.copy_(handle.get_tensor(key))
    for rows in (1, 6, 16):
        for pattern in ("random", "small", "large"):
            real.append(
                dict(
                    weight=name,
                    rows=rows,
                    pattern=pattern,
                    max_abs=check(rows, pattern),
                )
            )
    print(name, "passed", flush=True)

props = torch.cuda.get_device_properties(torch.cuda.current_device())
result = dict(
    native_sha256=hashlib.sha256(args.native.read_bytes()).hexdigest(),
    device=str(props),
    synthetic=synthetic,
    real_weights=real,
    invalid_calls=len(bad) + 3,
    tolerance=dict(rtol=2e-5, atol=2e-6),
    timing_scope="Warm graph component timings; no clock/throttle admission or full-model speed claim.",
)
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(
    "PASS",
    len(real),
    "real cases; maximum error",
    max(x["max_abs"] for x in real),
    flush=True,
)
