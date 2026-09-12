#!/usr/bin/env python3
"""Byte-exact native attention regression and CUDA-graph microbenchmark."""
import argparse
import ctypes as C
import json
import torch
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Compare native sparse attention implementations across cache and masking paths"
)
parser.add_argument("--baseline", required=True)
parser.add_argument("--candidate", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--unaligned-values", action="store_true")
parser.add_argument("--window-only", action="store_true")
parser.add_argument("--fp4-source", action="store_true",
                    help="Candidate uses FP4 source rows; default baseline is the test-only decoded-BF16 reader")
parser.add_argument("--baseline-fp4-source", action="store_true",
                    help="With --fp4-source, compare two FP4 kernels using identical packed source buffers")
parser.add_argument("--window-begin", type=int,
                    help="Exercise bounded replay with this device lower bound (2049 tests invalid metadata)")
parser.add_argument("--repeats", type=int, default=20,
                    help="Graph timing repetitions; use 1 for memory-safety instrumentation")
parser.add_argument("--rows", type=int, nargs="+", default=[1, 2, 6, 16, 80])
parser.add_argument("--parts", type=int, choices=range(0, 11), default=0,
                    help="0 uses unsplit attention; 1–10 compare the same split count")
args = parser.parse_args()
if args.baseline_fp4_source and not args.fp4_source:
    parser.error("--baseline-fp4-source requires --fp4-source")
if not args.rows or min(args.rows) < 1 or max(args.rows) > 4096:
    parser.error("rows must be between 1 and 4096")
if args.repeats < 1:parser.error("repeats must be positive")
torch.cuda.set_device(args.device)


class View(C.Structure):
    _fields_ = [
        ("values", C.c_void_p * 4),
        ("scales", C.c_void_p * 4),
        ("window_end", C.c_void_p),
        ("pages", C.c_void_p),
        ("source_end", C.c_void_p),
        ("window_capacity", C.c_uint64),
        ("source_capacity", C.c_uint64),
        ("source_proposal_capacity", C.c_uint64),
        ("page_stride", C.c_uint32),
        ("compressed", C.c_uint32),
    ]


assert C.sizeof(View) == 120
libs = [C.CDLL(args.baseline), C.CDLL(args.candidate)]
for i,lib in enumerate(libs):
    assert lib.ds41rt_v41_sparse_attention_initialize() == 0
    fn = lib.ds41rt_v41_sparse_attention_bounded if args.window_begin is not None else (lib.ds41rt_v41_sparse_attention if args.parts==0 else lib.ds41rt_v41_sparse_attention_split)
    fn.argtypes = [C.c_void_p]*5+[C.c_int32,C.c_int32,C.POINTER(View),C.c_void_p]+([C.c_void_p,C.c_void_p,C.c_uint64,C.c_int32] if args.window_begin is not None else ([] if args.parts==0 else [C.c_void_p,C.c_uint64,C.c_int32]))
torch.manual_seed(731)
results = []
for rows in args.rows:
    q = (torch.randn(rows, 64, 512, device="cuda") * 0.2).bfloat16()
    sink = torch.randn(64, device="cuda")
    values = [
        torch.randn(n, 512, device="cuda").to(torch.float8_e4m3fn).view(torch.uint8)
        for n in [128, rows, 768, 3]
    ]
    if args.fp4_source:
        for i in [2,3]:
            values[i]=torch.randint(0,256,(values[i].shape[0],256),dtype=torch.uint8,device='cuda')
    if args.unaligned_values:
        shifted = []
        for value in values:
            storage = torch.empty(value.numel() + 1, dtype=torch.uint8, device="cuda")
            view = storage[1:].view_as(value)
            view.copy_(value)
            shifted.append(view)
        values = shifted
    scales = [
        torch.full((n, 16), 125, dtype=torch.uint8, device="cuda")
        for n in [128, rows, 768, 3]
    ]
    if args.fp4_source:
        for i in [2,3]:scales[i]=torch.full((values[i].shape[0],32),56,dtype=torch.uint8,device='cuda')
    decoded = [torch.empty((values[i].shape[0],512),dtype=torch.bfloat16,device='cuda') for i in [2,3]] if args.fp4_source else []
    def decode_source():
        lut=torch.tensor([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6],device='cuda')
        for i in [2,3]:
            nibbles=torch.stack([values[i]&15,values[i]>>4],dim=-1).reshape(-1,512).long()
            factor=scales[i].view(torch.float8_e4m3fn).float().repeat_interleave(16,dim=1)
            decoded[i-2].copy_((lut[nibbles]*factor).bfloat16())
    if args.fp4_source:decode_source()
    end = torch.tensor([2048], dtype=torch.uint64, device="cuda")
    source_end = torch.tensor([512], dtype=torch.uint64, device="cuda")
    pages = torch.tensor([1, 0], dtype=torch.int32, device="cuda")
    meta = torch.tensor(
        [[2048, 0, rows, 2048 + r, 0, 512, 512, 0, 0, 1] for r in range(rows)],
        dtype=torch.uint64,
        device="cuda",
    )
    selected = torch.full((rows, 512), -1, dtype=torch.int32, device="cuda")
    out = [torch.empty_like(q), torch.empty_like(q)]
    scratch=torch.empty((rows,max(args.parts,1),64,514),dtype=torch.float32,device="cuda")
    bounds=torch.full((rows,),args.window_begin or 0,dtype=torch.uint64,device='cuda')
    v = View(
        (C.c_void_p * 4)(*[x.data_ptr() for x in values]),
        (C.c_void_p * 4)(*[x.data_ptr() for x in scales]),
        end.data_ptr(),
        pages.data_ptr(),
        source_end.data_ptr(),
        rows,
        768,
        3,
        2,
        (2 if args.fp4_source else 1)*int(not args.window_only),
    )
    reference_view=View.from_buffer_copy(v)
    if args.fp4_source and not args.baseline_fp4_source:
        reference_view.compressed=int(not args.window_only)
        reference_view.values[2]=decoded[0].data_ptr()
        reference_view.values[3]=decoded[1].data_ptr()

    def launch(i):
        fn=libs[i].ds41rt_v41_sparse_attention_bounded if args.window_begin is not None else (libs[i].ds41rt_v41_sparse_attention if args.parts==0 else libs[i].ds41rt_v41_sparse_attention_split)
        tail=([bounds.data_ptr(),scratch.data_ptr() if args.parts else None,scratch.numel()*4 if args.parts else 0,max(args.parts,1)] if args.window_begin is not None else ([] if args.parts==0 else [scratch.data_ptr(),scratch.numel()*4,args.parts]))
        status = fn(
            q.data_ptr(),
            sink.data_ptr(),
            meta.data_ptr(),
            selected.data_ptr(),
            out[i].data_ptr(),
            rows,
            0,
            C.byref(reference_view if i==0 else v),
            torch.cuda.current_stream().cuda_stream,
            *tail,
        )
        assert status == 0, status

    graphs = []
    for i in range(2):
        launch(i)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            launch(i)
        graphs.append(g)
    for pattern in [
        "none",
        "first5",
        "late5",
        "gaps",
        "full",
        "stale",
        "private",
        "private_stride2",
        "scale_zero",
        "mixed_scales",
    ]:
        s = torch.full((rows, 512), -1, dtype=torch.int32, device="cuda")
        for scale in scales:
            scale.fill_(125)
        if args.fp4_source:
            for scale in scales[2:]:scale.fill_(56)
        meta.copy_(
            torch.tensor(
                [[2048, 0, rows, 2048 + r, 0, 512, 512, 0, 0, 1] for r in range(rows)],
                dtype=torch.uint64,
                device="cuda",
            )
        )
        if pattern in ["first5", "scale_zero", "mixed_scales"]:
            s[:, :5] = torch.arange(5, device="cuda")
        if pattern == "late5":
            s[:, 450:455] = torch.arange(5, device="cuda")
        if pattern == "gaps":
            s[:, ::64] = torch.arange(8, device="cuda")
        if pattern == "full":
            s[:] = torch.randperm(512, device="cuda")
        if pattern in ["private", "private_stride2"]:
            s[:, :2] = torch.tensor([512, 513], device="cuda", dtype=torch.int32)
            stride = 2 if pattern == "private_stride2" else 1
            origin = 0 if stride == 2 else 1
            meta.copy_(
                torch.tensor(
                    [
                        [2048, 0, rows, 2048 + r, 0, 514, 512, 2, origin, stride]
                        for r in range(rows)
                    ],
                    dtype=torch.uint64,
                    device="cuda",
                )
            )
        if pattern == "scale_zero":
            for scale in scales:
                scale.zero_()
        if pattern == "mixed_scales":
            for i,scale in enumerate(scales):
                scale.copy_(
                    torch.tensor(
                        ([0,1,56,77]*8 if args.fp4_source and i>=2 else [0,124,127,130]*4), dtype=torch.uint8, device="cuda"
                    ).expand_as(scale)
                )
        if args.fp4_source:decode_source()
        selected.copy_(s)
        end.fill_(2047 if pattern == "stale" else 2048)
        for g in graphs:
            g.replay()
        torch.cuda.synchronize()
        assert torch.isfinite(out[1]).all()
        error=(out[0].float()-out[1].float())
        bit_exact=torch.equal(out[0].view(torch.int16),out[1].view(torch.int16))
        assert bit_exact, (rows, pattern, args.parts)
        metrics=dict(max_abs=float(error.abs().max()),rel_l2=float(error.norm()/out[0].float().norm().clamp_min(1e-30)),different=int((out[0]!=out[1]).sum()),elements=out[0].numel())
        elapsed = []
        for g in graphs:
            a, b = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            a.record()
            for _ in range(args.repeats):
                g.replay()
            b.record()
            b.synchronize()
            elapsed.append(a.elapsed_time(b) * 1000 / args.repeats)
        results.append(
            dict(
                rows=rows,parts=args.parts,scope=("Identical packed FP4 source regression and graph timing" if args.baseline_fp4_source else "FP4 source versus independently decoded BF16 source with identical attention arithmetic" if args.fp4_source else "Byte-exact regression against baseline; graph microbenchmark, not independent reference math"),
                pattern=pattern,
                bit_exact=bit_exact,
                **metrics,
                unaligned_values=args.unaligned_values,
                window_only=args.window_only,
                fp4_source=args.fp4_source,
                baseline_fp4_source=args.baseline_fp4_source,
                window_begin=args.window_begin,
                timing_repeats=args.repeats,
                baseline_us=elapsed[0],
                candidate_us=elapsed[1],
            )
        )
print(json.dumps(results, indent=2))
args.output.write_text(json.dumps(results, indent=2) + "\n")
