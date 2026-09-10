#!/usr/bin/env python3
"""Exercise the production engram CUDA ABI against the pinned upstream forward."""
import argparse
import ast
import ctypes
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--reference-dir', type=Path, required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--large-offset', action='store_true', help='exercise element offsets beyond 2^31 (about 14 GB VRAM)')
    args = parser.parse_args()
    source = args.reference_dir / 'inference/model.py'
    lock = json.loads((ROOT / 'docs/ds41-reference-lock.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == lock['files']['inference/model.py']
    cls = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.ClassDef) and node.name == 'Engram')
    forward = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'forward')
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[forward], type_ignores=[]), str(source), 'exec'), namespace)
    lib = ctypes.CDLL(str(args.library.resolve()))
    kernel = lib.ds41rt_cuda_engram_gate_bf16_async
    kernel.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int, ctypes.c_void_p]
    kernel.restype = ctypes.c_int
    torch.cuda.set_device(args.device)
    torch.manual_seed(4101)
    evidence = []
    dequant = lib.ds41rt_cuda_engram_dequant_bf16_async
    dequant.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int, ctypes.c_void_p]
    dequant.restype = ctypes.c_int
    for hash_rows in (1, 24, 384, 1920):
        weights = torch.arange(256, device='cuda', dtype=torch.int32).to(torch.uint8).repeat(hash_rows, 1)
        scales = torch.tensor([0, 1, 120, 127, 130, 200, 254, 255], device='cuda', dtype=torch.uint8).repeat(hash_rows, 1)
        unpacked = torch.empty((hash_rows, 256), device='cuda', dtype=torch.bfloat16)

        def launch_dequant():
            assert dequant(weights.data_ptr(), scales.data_ptr(), unpacked.data_ptr(), hash_rows,
                           torch.cuda.current_stream().cuda_stream) == 0

        def check_dequant():
            expected = (weights.view(torch.float8_e4m3fn).float().view(hash_rows, 8, 32) *
                        scales.view(torch.float8_e8m0fnu).float().unsqueeze(-1)).view(hash_rows, 256).bfloat16()
            torch.testing.assert_close(unpacked, expected, rtol=0, atol=0, equal_nan=True)
            finite = torch.isfinite(expected)
            assert torch.equal(unpacked.view(torch.int16)[finite], expected.view(torch.int16)[finite])

        launch_dequant()
        check_dequant()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): launch_dequant()
        weights.bitwise_xor_(3)
        scales.bitwise_xor_(1)
        allocated = torch.cuda.memory_allocated()
        graph.replay()
        assert torch.cuda.memory_allocated() == allocated
        check_dequant()
        evidence.append(dict(hash_rows=hash_rows, dequant_exact=True, graph_replay=True))
    for rows in (1, 16, 80, 256):
        x = torch.randn((1, rows, 4, 5120), device='cuda', dtype=torch.bfloat16)
        kv = torch.randn((1, rows, 25600), device='cuda', dtype=torch.bfloat16)
        qw = torch.randn((4, 5120), device='cuda', dtype=torch.bfloat16)
        kw = torch.randn_like(qw)
        mask = (torch.arange(rows, device='cuda') % 3 != 1).to(torch.uint8)
        out = torch.empty_like(x)
        context = SimpleNamespace(embed=lambda _: kv.unsqueeze(-2), wkv=lambda v: v,
                                  hc_mult=4, dim=5120, q_weight=qw, k_weight=kw,
                                  eps=1e-20, clamp_value=1e-6)

        def reference():
            return namespace['forward'](context, x, None, mask.bool().view(1, rows))

        def launch(destination=out):
            assert kernel(x.data_ptr(), kv.data_ptr(), qw.data_ptr(), kw.data_ptr(), mask.data_ptr(),
                          destination.data_ptr(), rows, torch.cuda.current_stream().cuda_stream) == 0

        def check(actual, expected):
            assert torch.isfinite(actual).all() and actual.abs().sum() > 0
            torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.00002)
            assert torch.equal(actual[:, mask == 0], x[:, mask == 0])

        launch()
        expected = reference()
        check(out, expected)
        maximum = (out.float() - expected.float()).abs().max().item()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3): launch()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): launch()
        # Replay must consume changed addresses' contents, with no capture-time constants.
        x.mul_(0.125)
        kv.mul_(4)
        allocated = torch.cuda.memory_allocated()
        graph.replay()
        assert torch.cuda.memory_allocated() == allocated
        check(out, reference())
        # Exact zero dot exercises signed-root clamping and norm epsilon.
        kv[..., :4 * 5120].zero_()
        allocated = torch.cuda.memory_allocated()
        graph.replay()
        assert torch.cuda.memory_allocated() == allocated
        check(out, reference())
        expected = reference()
        launch(x)
        torch.testing.assert_close(x, expected, rtol=0.008, atol=0.00002)
        evidence.append(dict(rows=rows, max_abs_error=maximum, graph_replay=True, inplace=True))
    if args.large_offset:
        rows = (1 << 31) // (4 * 5120) + 3
        high_x = torch.zeros((rows, 4, 5120), device='cuda', dtype=torch.bfloat16)
        high_kv = torch.empty((rows, 25600), device='cuda', dtype=torch.bfloat16)
        high_out = torch.empty_like(high_x)
        high_mask = torch.zeros(rows, device='cuda', dtype=torch.uint8)
        high_x[-1].normal_()
        high_kv[-1].normal_()
        high_mask[-1] = 1
        tail_context = SimpleNamespace(embed=lambda _: high_kv[-1:].view(1, 1, 1, 25600),
            wkv=lambda v: v, hc_mult=4, dim=5120, q_weight=qw, k_weight=kw, eps=1e-20, clamp_value=1e-6)
        tail_expected = namespace['forward'](tail_context, high_x[-1:].unsqueeze(0), None, None)
        assert kernel(high_x.data_ptr(), high_kv.data_ptr(), qw.data_ptr(), kw.data_ptr(),
            high_mask.data_ptr(), high_out.data_ptr(), rows, torch.cuda.current_stream().cuda_stream) == 0
        torch.testing.assert_close(high_out[-1:].unsqueeze(0), tail_expected, rtol=0.008, atol=0.00002)
        assert high_out[0].count_nonzero() == 0
        evidence.append(dict(rows=rows, high_element_offset=(rows - 1) * 4 * 5120, large_offset=True))
    assert kernel(None, None, None, None, None, None, 0, None) == 1
    torch.cuda.synchronize()
    print(json.dumps(dict(device=args.device, gpu=torch.cuda.get_device_name(), cases=evidence,
                          library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest())))


if __name__ == '__main__':
    main()
