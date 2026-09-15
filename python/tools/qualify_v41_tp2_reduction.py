#!/usr/bin/env python3
"""Verify FP32 TP2 route/token reduction and changed-data graph replay."""
import argparse
import ctypes as C
import json
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-lib", required=True)
    args = parser.parse_args()
    fn = C.CDLL(args.native_lib).ds41rt_v41_reduce_tp2_experts_async
    fn.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_uint32, C.c_uint32, C.c_void_p]
    results = []
    assert torch.cuda.device_count() >= 2
    for device in (0, 1):
        with torch.cuda.device(device):
            for rows in (1, 16, 4096):
                for routes in (1, 6):
                    a = torch.randn((rows, routes, 5120), device="cuda")
                    b = torch.randn_like(a)
                    storage = torch.full((rows * 5120 + 16,), 17, dtype=torch.bfloat16, device="cuda")
                    out = storage[:rows*5120].reshape(rows, 5120)
                    stream = torch.cuda.Stream(device=device)
                    stream.wait_stream(torch.cuda.current_stream())
                    def launch():
                        assert fn(a.data_ptr(), b.data_ptr(), out.data_ptr(), rows, int(routes == 1),
                                  torch.cuda.current_stream().cuda_stream) == 0
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream): launch()
                    for changed in (False, True):
                        if changed:
                            a.mul_(-0.5)
                            b.add_(0.25)
                        graph.replay()
                        expected = a[:, 0] + b[:, 0]
                        for route in range(1, routes): expected = expected + (a[:, route] + b[:, route])
                        assert torch.equal(out, expected.bfloat16())
                        assert bool((storage[rows*5120:] == 17).all())
                    assert fn(a.data_ptr(), b.data_ptr(), a.data_ptr(), rows, int(routes == 1),
                              torch.cuda.current_stream().cuda_stream) != 0
                    assert fn(a.data_ptr(), b.data_ptr(), out.data_ptr(), 4097, 0,
                              torch.cuda.current_stream().cuda_stream) != 0
                    results.append(dict(device=device, rows=rows, routes=routes, exact=True))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
