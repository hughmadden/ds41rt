#!/usr/bin/env python3
"""Qualify compact-return CUDA arithmetic, aliases and graph replay.

This compares the new operation with its explicit ordered arithmetic oracle.
It does not establish equivalence to per-route TP rounding or model quality.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-lib", type=Path, required=True)
    parser.add_argument("--real-planes", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lib = C.CDLL(str(args.native_lib.resolve()))
    pack = lib.ds41rt_v41_compact_routes_bf16_async
    pack.argtypes = [C.c_void_p, C.c_void_p, C.c_uint32, C.c_void_p]
    pack.restype = C.c_int32
    reduce = lib.ds41rt_v41_reduce_compact_bf16_async
    reduce.argtypes = [C.POINTER(C.c_void_p), C.c_void_p, C.c_void_p,
                       C.c_uint32, C.c_void_p]
    reduce.restype = C.c_int32
    torch.cuda.init()
    stream = torch.cuda.current_stream().cuda_stream
    torch.manual_seed(941)
    results = []

    def ordered_sum(values):
        result = values[0].float().clone()
        for value in values[1:]:
            result = result + value.float()
        return result

    def check(name, routes):
        rows = routes[0].shape[0]
        partials = [torch.empty((rows, 5120), dtype=torch.bfloat16, device="cuda")
                    for _ in range(4)]
        shared = torch.randn((rows, 5120), device="cuda").to(torch.bfloat16)
        out = torch.empty_like(shared)
        pointers = (C.c_void_p * 4)(*[x.data_ptr() for x in partials])

        def launch(with_shared=True):
            for source, destination in zip(routes, partials):
                assert pack(source.data_ptr(), destination.data_ptr(), rows, stream) == 0
            assert reduce(pointers, shared.data_ptr() if with_shared else None,
                          out.data_ptr(), rows, stream) == 0

        def verify(with_shared=True):
            expected_parts = [ordered_sum(list(x.unbind(1))).to(torch.bfloat16)
                              for x in routes]
            for actual, expected in zip(partials, expected_parts):
                assert torch.equal(actual, expected), f"{name}: compact partial mismatch"
            expected = ordered_sum(expected_parts)
            if with_shared:
                expected = expected + shared.float()
            expected = expected.to(torch.bfloat16)
            assert torch.isfinite(out).all() and out.abs().max() > 0
            assert torch.equal(out, expected), f"{name}: rank reduction mismatch"

        launch(False)
        verify(False)
        launch()
        verify()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            # graph capture uses its own current stream.
            capture_stream = torch.cuda.current_stream().cuda_stream
            for source, destination in zip(routes, partials):
                assert pack(source.data_ptr(), destination.data_ptr(), rows,
                            capture_stream) == 0
            assert reduce(pointers, shared.data_ptr(), out.data_ptr(), rows,
                          capture_stream) == 0
        for replay in range(2):
            routes[replay].add_(0.03125)
            shared.add_(0.125)
            out.fill_(float("nan"))
            graph.replay()
            verify()

        expected_alias = (ordered_sum(partials) + shared.float()).to(torch.bfloat16)
        assert reduce(pointers, shared.data_ptr(), shared.data_ptr(), rows, stream) == 0
        assert torch.equal(shared, expected_alias), f"{name}: exact shared alias mismatch"
        # Rejections occur before launch and must not poison the CUDA stream.
        assert pack(routes[0].data_ptr(), routes[0].data_ptr(), rows, stream) != 0
        assert pack(routes[0].data_ptr() + 1, out.data_ptr(), rows, stream) != 0
        assert pack(routes[0].data_ptr(), out.data_ptr(), 0, stream) != 0
        assert reduce(pointers, None, partials[0].data_ptr(), rows, stream) != 0
        assert reduce(pointers, shared.data_ptr() + 2, shared.data_ptr(), rows, stream) != 0
        assert reduce(pointers, None, out.data_ptr(), 0, stream) != 0
        null_rank = (C.c_void_p * 4)(None, *[x.data_ptr() for x in partials[1:]])
        assert reduce(null_rank, None, out.data_ptr(), rows, stream) != 0
        launch()
        verify()
        results.append({"case": name, "rows": rows, "exact": True,
                        "changed_input_graph_replays": 2, "alias_and_rejection_checks": True})

    for rows in (1, 16, 80, 129, 257):
        check(f"synthetic-{rows}", [torch.randn((rows, 6, 5120), device="cuda")
                                    for _ in range(4)])
    if args.real_planes:
        for cycle in range(2):
            planes = []
            for rank in range(4):
                path = args.real_planes / f"rank{rank}" / f"l0-c{cycle}-plane.bin"
                assert path.stat().st_size == 80 * 6 * 5120 * 4
                planes.append(torch.from_file(str(path), size=80 * 6 * 5120,
                                              dtype=torch.float32).reshape(80, 6, 5120).cuda())
            check(f"real-layer0-cycle{cycle}", planes)
    torch.cuda.synchronize()
    report = {"scope": __doc__, "device": torch.cuda.get_device_name(),
              "capability": torch.cuda.get_device_capability(),
              "device_uuid": str(torch.cuda.get_device_properties(0).uuid),
              "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
              "native_sha256": hashlib.sha256(args.native_lib.read_bytes()).hexdigest(),
              "cases": results}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
