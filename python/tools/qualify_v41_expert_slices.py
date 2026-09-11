#!/usr/bin/env python3
"""Compare official TP4 expert slices with a deployed native FP8 consumer.

Inputs are two captured capacity-80 wire/ID/routing files. Route planning for
candidates runs outside timing; native timing includes its internal planner.
Only referenced expert shards are loaded, into full 384-expert GPU pools.
"""

import json, hashlib, statistics, time
from pathlib import Path
from contextlib import ExitStack
import torch, cutlass, cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass.cute.runtime import from_dlpack
from safetensors import safe_open
from _v41_expert_native import Native, library, check, P
import _pinned_sparkinfer
from b12x._lib.utils import current_cuda_stream
from b12x.moe._shared.kernels.w4a8_v41_slice import V41FusedSliceKernel
from tests.moe.test_v41_grouped_slices import _metadata


class Reduce:
    def __init__(self, width):
        self.slices = (576 + width - 1) // width

    @cute.jit
    def __call__(
        self,
        source: cute.Tensor,
        dest: cute.Tensor,
        pairs: cute.Tensor,
        routes: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(source, dest, pairs, routes).launch(
            grid=((routes * 5120 + 255) // 256, 1, 1), block=(256, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        source: cute.Tensor,
        dest: cute.Tensor,
        pairs: cute.Tensor,
        routes: cutlass.Int32,
    ):
        index = cutlass.Int64(cute.arch.block_idx()[0]) * 256 + cutlass.Int64(
            cute.arch.thread_idx()[0]
        )
        row = index // 5120
        col = index % 5120
        if row < routes:
            value = cutlass.Float32(0)
            for plane in cutlass.range_constexpr(self.slices):
                value += source[plane, row, col]
            dest[cutlass.Int64(pairs[row]), col] = value


def main():
    import argparse, inspect

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--native-lib", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Earlier Rust-owner route files; required files must match exactly",
    )
    parser.add_argument("--layer", type=int, choices=range(40), required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    if options.baseline and options.layer not in (0, 1):
        parser.error("Rust-owner baseline files cover layers zero and one")
    records = []

    def emit(kind, value):
        records.append(dict(kind=kind, **value))
        print(kind + " " + json.dumps(value), flush=True)

    hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [
            options.native_lib,
            Path(__file__),
            Path(inspect.getsourcefile(Native)),
            Path(inspect.getsourcefile(V41FusedSliceKernel)),
            options.snapshot / "model.safetensors.index.json",
            *[
                options.inputs / f"{case}-{name}.bin"
                for case in [0, 1]
                for name in ["hidden", "ids", "routing"]
            ],
        ]
    }
    emit(
        "SOURCE",
        dict(
            b12x_revision=_pinned_sparkinfer.REVISION,
            sha256=hashes,
            device=torch.cuda.get_device_name(),
            capability=torch.cuda.get_device_capability(),
        ),
    )
    snapshot = options.snapshot
    layer = options.layer
    lib = library(str(options.native_lib))
    cases = []
    for case in [0, 1]:
        fields = [
            torch.frombuffer(
                bytearray((options.inputs / f"{case}-{name}.bin").read_bytes()),
                dtype=dtype,
            )
            .clone()
            .reshape(80, width)
            for name, dtype, width in [
                ("hidden", torch.uint8, 5280),
                ("ids", torch.int32, 6),
                ("routing", torch.float32, 6),
            ]
        ]
        cases.append(fields)
    active = sorted(set(torch.cat([case[1].flatten() for case in cases]).tolist()))
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    sizes = [3276800, 204800, 1638400, 102400]
    weights = [
        torch.empty((384, size), device="cuda", dtype=torch.uint8) for size in sizes
    ]
    start = time.perf_counter()
    with ExitStack() as stack:
        files = {}
        for expert in active:
            sources = []
            for projection, suffix in [
                ("w1", "weight"),
                ("w3", "weight"),
                ("w2", "weight"),
                ("w1", "scale"),
                ("w3", "scale"),
                ("w2", "scale"),
            ]:
                key = f"layers.{layer}.ffn.experts.{expert}.{projection}.{suffix}"
                shard = index[key]
                if shard not in files:
                    files[shard] = stack.enter_context(
                        safe_open(snapshot / shard, framework="pt", device="cpu")
                    )
                tensor = files[shard].get_tensor(key)
                expected_shape = (
                    (5120, 1152 if suffix == "weight" else 72)
                    if projection == "w2"
                    else (2304, 2560 if suffix == "weight" else 160)
                )
                assert tuple(tensor.shape) == expected_shape, (key, tensor.shape)
                assert tensor.dtype == (
                    torch.int8 if suffix == "weight" else torch.float8_e8m0fnu
                ), (key, tensor.dtype)
                raw = tensor.view(torch.uint8)
                shard_tensor = (
                    raw[:576]
                    if projection != "w2"
                    else raw[:, : 288 if suffix == "weight" else 18]
                )
                sources.append(shard_tensor.contiguous().cuda())
            src = (P * 6)(*[x.data_ptr() for x in sources])
            dst = (P * 4)(*[x[expert].data_ptr() for x in weights])
            check(
                lib.ds41rt_v41_pack_expert_async(
                    src, dst, 576, torch.cuda.current_stream().cuda_stream
                )
            )
        torch.cuda.synchronize()
    emit(
        "LOAD",
        dict(
            layer=layer, active_experts=len(active), seconds=time.perf_counter() - start
        ),
    )
    wire = torch.empty((80, 5280), device="cuda", dtype=torch.uint8)
    ids = torch.empty((80, 6), device="cuda", dtype=torch.int32)
    routing = torch.empty((80, 6), device="cuda")
    qa = wire[:, :5120].view(torch.uint32)
    qs = wire[:, 5120:]
    packed = [x.view(torch.uint32).flatten() for x in weights]
    native = {
        capacity: Native(lib, capacity, weights, wire, ids, routing)
        for capacity in [1, 80]
    }
    variants = {}
    for capacity in [1, 16, 80]:
        metadata = torch.full((capacity * 6, 19), -1, device="cuda", dtype=torch.int32)
        pairmap = torch.empty(capacity * 6, device="cuda", dtype=torch.int32)
        rw = torch.empty(capacity * 6, device="cuda")
        mv = from_dlpack(metadata, assumed_align=16)
        pv = from_dlpack(pairmap, assumed_align=16)
        for width in [64, 128, 192]:
            output = torch.empty(
                ((576 + width - 1) // width, capacity * 6, 5120), device="cuda"
            )
            reduced = torch.empty((capacity * 6, 5120), device="cuda")
            args = [
                from_dlpack(x, assumed_align=16) for x in [qa, qs, *packed, rw, output]
            ]
            rv = from_dlpack(reduced, assumed_align=16)
            fn = cute.compile(
                V41FusedSliceKernel(width, grouped=True),
                *args,
                cutlass.Int32(80),
                current_cuda_stream(),
                mv,
                cutlass.Int32(capacity * 6),
            )
            reduce = cute.compile(
                Reduce(width),
                args[-1],
                rv,
                pv,
                cutlass.Int32(capacity * 6),
                current_cuda_stream(),
            )
            variants[capacity, width] = (
                metadata,
                pairmap,
                rw,
                mv,
                pv,
                output,
                reduced,
                rv,
                args,
                fn,
                reduce,
            )
    for case, (wire_cpu, ids_cpu, rw_cpu) in enumerate(cases):
        wire.copy_(wire_cpu)
        ids.copy_(ids_cpu)
        routing.copy_(rw_cpu)
        for rows in [1, 2, 6, 16, 80, 1]:
            baseline = native[1 if rows == 1 else 80]
            baseline.run(rows)
            torch.cuda.synchronize()
            base = baseline.output[: rows * 6].clone()
            assert torch.isfinite(base).all() and base.norm() > 0
            prior = (
                (options.baseline / f"v{layer}-c{case}-m{rows}-routes.bin")
                if options.baseline
                else None
            )
            if prior is not None and rows in (1, 16, 80):
                assert prior.is_file(), prior
                expected = (
                    torch.frombuffer(bytearray(prior.read_bytes()), dtype=torch.float32)
                    .reshape(rows * 6, 5120)
                    .cuda()
                )
                assert torch.equal(base, expected), (
                    "baseline mismatch",
                    case,
                    rows,
                    (base - expected).abs().max().item(),
                )
                emit(
                    "BASELINE",
                    dict(
                        layer=layer,
                        case=case,
                        rows=rows,
                        path=str(prior),
                        sha256=hashlib.sha256(prior.read_bytes()).hexdigest(),
                    ),
                )
            graphs = {}
            outputs = {"native": baseline.output}
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                baseline.run(rows)
            graphs["native"] = graph
            cap = 1 if rows == 1 else 16 if rows <= 16 else 80
            tasks, pairs = _metadata(ids_cpu[:rows])
            groups = len(tasks)
            routes = rows * 6
            metrics = {}
            for width in [64, 128, 192]:
                meta, pmap, rw, mv, pv, out, reduced, rv, args, fn, reduce = variants[
                    cap, width
                ]
                meta.fill_(-1)
                meta[:groups].copy_(tasks)
                pmap[:routes].copy_((pairs[:, 0] * 6 + pairs[:, 1]).int())
                rw[:routes].copy_(rw_cpu[pairs[:, 0], pairs[:, 1]])

                def run():
                    fn(*args, rows, current_cuda_stream(), mv, cap * 6)
                    reduce(args[-1], rv, pv, routes, current_cuda_stream())

                run()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
                before = torch.cuda.memory_allocated()
                graph.replay()
                assert before == torch.cuda.memory_allocated()
                candidate = reduced[:routes]
                diff = candidate - base
                rel = (diff.norm() / base.norm()).item()
                maxabs = diff.abs().max().item()
                # Compare current compact rank arithmetic as well as individual FP32 routes.
                ac = torch.empty(rows, 5120, device="cuda", dtype=torch.bfloat16)
                bc = torch.empty_like(ac)
                check(
                    lib.ds41rt_v41_compact_routes_bf16_async(
                        base.data_ptr(),
                        ac.data_ptr(),
                        rows,
                        torch.cuda.current_stream().cuda_stream,
                    )
                )
                check(
                    lib.ds41rt_v41_compact_routes_bf16_async(
                        candidate.data_ptr(),
                        bc.data_ptr(),
                        rows,
                        torch.cuda.current_stream().cuda_stream,
                    )
                )
                metrics[width] = dict(
                    route_rel_l2=rel,
                    route_max_abs=maxabs,
                    compact_mismatches=int((ac != bc).sum()),
                    compact_elements=ac.numel(),
                    compact_rel_l2=(
                        (ac.float() - bc.float()).norm() / ac.float().norm()
                    ).item(),
                )
                emit(
                    "CHECK",
                    dict(
                        layer=layer, case=case, rows=rows, width=width, **metrics[width]
                    ),
                )
                assert torch.isfinite(candidate).all() and rel < 1e-6, (width, rel)
                assert metrics[width]["compact_rel_l2"] < 1e-5, metrics[width]
                if width == 128:
                    assert torch.equal(candidate, base), metrics[width]
                graphs[str(width)] = graph
                outputs[str(width)] = reduced
            samples = {arm: [] for arm in graphs}
            for _ in range(5):
                for graph in graphs.values():
                    graph.replay()
            import itertools
            import random

            orders = list(itertools.permutations(graphs))
            random.Random(41).shuffle(orders)
            for order in orders:
                for arm in order:
                    begin, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    begin.record()
                    for _ in range(10):
                        graphs[arm].replay()
                    end.record()
                    end.synchronize()
                    samples[arm].append(begin.elapsed_time(end) * 1000 / 10)
            emit(
                "TIMING",
                dict(
                    layer=layer,
                    case=case,
                    rows=rows,
                    groups=groups,
                    capacity=cap,
                    graph_us=samples,
                    median_us={k: statistics.median(v) for k, v in samples.items()},
                    metrics=metrics,
                ),
            )
            for graph in graphs.values():
                graph.reset()
    options.output.write_text(
        json.dumps(dict(schema=1, records=records), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
