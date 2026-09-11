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
parser.add_argument("--rows", type=int, nargs="+", default=[1, 2, 6, 16, 80])
args = parser.parse_args()
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
for lib in libs:
    assert lib.ds41rt_v41_sparse_attention_initialize() == 0
    lib.ds41rt_v41_sparse_attention.argtypes = [C.c_void_p] * 5 + [
        C.c_int32,
        C.c_int32,
        C.POINTER(View),
        C.c_void_p,
    ]
torch.manual_seed(731)
results = []
for rows in args.rows:
    q = (torch.randn(rows, 64, 512, device="cuda") * 0.2).bfloat16()
    sink = torch.randn(64, device="cuda")
    values = [
        torch.randn(n, 512, device="cuda").to(torch.float8_e4m3fn).view(torch.uint8)
        for n in [128, rows, 768, 3]
    ]
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
        int(not args.window_only),
    )

    def launch(i):
        status = libs[i].ds41rt_v41_sparse_attention(
            q.data_ptr(),
            sink.data_ptr(),
            meta.data_ptr(),
            selected.data_ptr(),
            out[i].data_ptr(),
            rows,
            0,
            C.byref(v),
            torch.cuda.current_stream().cuda_stream,
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
            for scale in scales:
                scale.copy_(
                    torch.tensor(
                        [0, 124, 127, 130] * 4, dtype=torch.uint8, device="cuda"
                    ).expand_as(scale)
                )
        selected.copy_(s)
        end.fill_(2047 if pattern == "stale" else 2048)
        for g in graphs:
            g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out[0].view(torch.int16), out[1].view(torch.int16)), (
            rows,
            pattern,
            (out[0].float() - out[1].float()).abs().max().item(),
        )
        elapsed = []
        for g in graphs:
            a, b = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            a.record()
            for _ in range(20):
                g.replay()
            b.record()
            b.synchronize()
            elapsed.append(a.elapsed_time(b) * 1000 / 20)
        results.append(
            dict(
                rows=rows,
                pattern=pattern,
                bit_exact=True,
                unaligned_values=args.unaligned_values,
                window_only=args.window_only,
                baseline_us=elapsed[0],
                candidate_us=elapsed[1],
            )
        )
print(json.dumps(results, indent=2))
args.output.write_text(json.dumps(results, indent=2) + "\n")
