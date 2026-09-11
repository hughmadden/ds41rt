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
parser.add_argument("--split-parts", type=int, choices=range(1, 11), default=None,
                    help="Qualify split rounding against independent FP32 attention; default requires bit equality")
parser.add_argument("--seed", type=int, default=731)
parser.add_argument("--query-scale", type=float, default=.2)
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
launches = []
for i, lib in enumerate(libs):
    assert lib.ds41rt_v41_sparse_attention_initialize() == 0
    split = i == 1 and args.split_parts is not None
    fn = lib.ds41rt_v41_sparse_attention_split if split else lib.ds41rt_v41_sparse_attention
    fn.argtypes = [C.c_void_p] * 5 + [C.c_int32, C.c_int32, C.POINTER(View), C.c_void_p]
    if split:
        fn.argtypes += [C.c_void_p, C.c_uint64, C.c_int32]
    launches.append(fn)
torch.manual_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = False
results = []
for rows in args.rows:
    q = (torch.randn(rows, 64, 512, device="cuda") * args.query_scale).bfloat16()
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
    scratch = None
    if args.split_parts is not None:
        storage = torch.full((rows * args.split_parts * 64 * 514 + 128,), 19.,
                             dtype=torch.float32, device="cuda")
        scratch = storage[64:-64]
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
        status = launches[i](
            q.data_ptr(),
            sink.data_ptr(),
            meta.data_ptr(),
            selected.data_ptr(),
            out[i].data_ptr(),
            rows,
            0,
            C.byref(v),
            torch.cuda.current_stream().cuda_stream,
            *([scratch.data_ptr(), scratch.numel() * 4, args.split_parts] if i == 1 and scratch is not None else []),
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
        for value in out:
            value.fill_(float("nan"))
        if scratch is not None:
            scratch.fill_(float("nan"))
        allocated = torch.cuda.memory_allocated()
        for g in graphs:
            g.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
        assert all(torch.isfinite(value).all() for value in out)
        bit_exact = torch.equal(out[0].view(torch.int16), out[1].view(torch.int16))
        error = out[0].float() - out[1].float()
        metrics = dict(max_abs=float(error.abs().max()),
                       relative_l2=float(error.norm() / out[0].float().norm().clamp_min(1e-30)))
        if args.split_parts is None:
            assert bit_exact, (rows, pattern, metrics)
        else:
            assert (storage[:64] == 19).all() and (storage[-64:] == 19).all()
        if args.split_parts is not None:
            # Independent address gathering and full FP32 softmax/PV, including sink.
            if pattern == "stale":
                oracle=torch.zeros_like(q,dtype=torch.float32)
            else:
                decoded=[(x.view(torch.float8_e4m3fn).float()*torch.exp2(sc.float()-127).repeat_interleave(32,-1)).bfloat16().float() for x,sc in zip(values,scales)]
                joined=torch.cat(decoded); offsets=[0,128,128+rows,128+rows+768]
                refs=[]; masks=[]; ph=pages.cpu().tolist(); mh=meta.cpu().tolist(); sh=selected.cpu().tolist()
                for row,m in enumerate(mh):
                    r=[]
                    for pos in range(m[3]-127,m[3]+1):
                        r.append(offsets[0]+pos%128 if pos<m[0] else offsets[1]+m[1]+pos-m[0])
                    if not args.window_only:
                        for id in sh[row]:
                            if id<0 or id>=m[5]: r.append(-1)
                            elif id<m[6]: r.append(offsets[2]+ph[id//256]*256+id%256)
                            else: r.append(offsets[3]+m[8]+(id-m[6])*m[9])
                    refs.append([max(i,0) for i in r]);masks.append([i>=0 for i in r])
                kv=joined[torch.tensor(refs,device="cuda")]
                mask=torch.tensor(masks,device="cuda")
                score=torch.bmm(q.float(),kv.transpose(1,2))*(512**-.5)
                score.masked_fill_(~mask[:,None,:],-torch.inf)
                logits=torch.cat([score,sink[None,:,None].expand(rows,-1,-1)],dim=-1)
                prob=logits.softmax(-1)[...,:score.shape[-1]]
                oracle=torch.bmm(prob,kv)
            errors=[]
            for value in out:
                e=value.float()-oracle
                errors.append(dict(relative_l2=float(e.norm()/oracle.norm().clamp_min(1e-30)),max_abs=float(e.abs().max())))
            # Fixed absolute accuracy bound plus non-regression against the
            # sequential BF16-probability kernel on the same FP32 oracle.
            assert errors[1]["relative_l2"] <= .003
            assert errors[1]["relative_l2"] <= errors[0]["relative_l2"] * 1.05 + 1e-5
            assert errors[1]["max_abs"] <= errors[0]["max_abs"] * 1.25 + 1e-5
            metrics["fp32_oracle_errors"] = errors
            metrics["prior_elementwise_tolerance_pass"] = bool(torch.isclose(out[1], out[0], rtol=.008, atol=.002).all())
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
                bit_exact=bit_exact,
                split_parts=args.split_parts,seed=args.seed,query_scale=args.query_scale,
                **metrics,
                unaligned_values=args.unaligned_values,
                window_only=args.window_only,
                baseline_us=elapsed[0],
                candidate_us=elapsed[1],
            )
        )
print(json.dumps(results, indent=2))
args.output.write_text(json.dumps(results, indent=2) + "\n")
