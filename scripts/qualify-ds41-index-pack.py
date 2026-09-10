#!/usr/bin/env python3
"""Byte-exact native index packing versus the pinned official TileLang kernel."""
import argparse
import ctypes as C
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
import tilelang


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-lib', type=Path, required=True)
    parser.add_argument('--reference-dir', type=Path, required=True)
    parser.add_argument('--device', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    lock = json.loads((root / 'docs/ds41-reference-lock.json').read_text())
    source = args.reference_dir / 'inference/kernel.py'
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    assert source_hash == lock['files']['inference/kernel.py']
    assert tilelang.__version__ == '0.1.8'
    spec = importlib.util.spec_from_file_location('ds41_official_kernel', source)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    lib = C.CDLL(str(args.native_lib))
    pack = lib.ds41rt_v41_index_pack
    pack.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_int32, C.c_void_p]
    pack.restype = C.c_int32
    torch.cuda.set_device(args.device)
    torch.manual_seed(410128)
    stream = torch.cuda.Stream()
    results = []
    with torch.cuda.stream(stream), torch.no_grad():
        def check(x, name):
            rows = x.shape[0]
            # Surround each destination with sentinels, including unaligned byte outputs.
            storage = torch.full((rows * 64 + 2,), 0xa5, device='cuda', dtype=torch.uint8)
            scale_storage = torch.full((rows * 4 + 2,), 0xa5, device='cuda', dtype=torch.uint8)
            y = storage[1:-1].view(rows, 64)
            scales = scale_storage[1:-1].view(rows, 4)
            def launch():
                status = pack(x.data_ptr(), y.data_ptr(), scales.data_ptr(), rows, stream.cuda_stream)
                assert status == 0, status
            def compare():
                expected, expected_scales = reference.fp4_act_quant(x)
                torch.testing.assert_close(y, expected.view(torch.uint8), rtol=0, atol=0)
                torch.testing.assert_close(scales, expected_scales.view(torch.uint8), rtol=0, atol=0)
                assert storage[0].item() == storage[-1].item() == 0xa5
                assert scale_storage[0].item() == scale_storage[-1].item() == 0xa5
            original = x.clone()
            launch()
            compare()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                launch()
            x.neg_()
            graph.replay()
            compare()
            x.zero_()
            graph.replay()
            compare()
            assert torch.count_nonzero(y).item() == 0
            assert (scales == 1).all().item()  # floor scale 2^-126
            x.copy_(original)
            graph.replay()
            compare()
            assert torch.equal(x.view(torch.uint8), original.view(torch.uint8))
            for n in (0, -1, 131073):
                assert pack(x.data_ptr(), y.data_ptr(), scales.data_ptr(), n, stream.cuda_stream) != 0
            for a, b, c in ((0, y.data_ptr(), scales.data_ptr()),
                            (x.data_ptr() + 1, y.data_ptr(), scales.data_ptr()),
                            (x.data_ptr(), 0, scales.data_ptr()),
                            (x.data_ptr(), y.data_ptr(), 0),
                            (x.data_ptr(), x.data_ptr(), scales.data_ptr()),
                            (x.data_ptr(), y.data_ptr(), x.data_ptr()),
                            (x.data_ptr(), y.data_ptr(), y.data_ptr()),
                            ((1 << 64) - 2, y.data_ptr(), scales.data_ptr()),
                            (x.data_ptr(), (1 << 64) - 1, scales.data_ptr()),
                            (x.data_ptr(), y.data_ptr(), (1 << 64) - 1)):
                assert pack(a, b, c, rows, stream.cuda_stream) != 0
            results.append({'name': name, 'rows': rows, 'packed_bytes': rows * 64,
                            'scale_bytes': rows * 4, 'reference_bytes_exact': True,
                            'changed_graph_and_recovery_exact': True, 'guards': True})
            print(f'PASS {name} rows={rows}', flush=True)
        for rows in (1, 3, 16, 80, 255, 4096, 131072):
            x = torch.randn(rows, 128, device='cuda').bfloat16()
            check(x, 'random')
        # All finite BF16 bit patterns, including both zeros and subnormals.
        bits = torch.arange(65536, device='cuda', dtype=torch.int32)
        bits = bits[(bits & 0x7f80) != 0x7f80].to(torch.uint16)
        check(bits.view(torch.bfloat16).reshape(-1, 128).contiguous(), 'all_finite_bf16')
        # Each group has maximum 6 to fix its scale at one; include ties and both
        # BF16 neighbors of every positive/negative FP4 rounding midpoint.
        mid = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], device='cuda').bfloat16()
        mid = torch.cat((mid, -mid))
        values = torch.stack((torch.nextafter(mid, torch.full_like(mid, -float('inf'))),
                              mid, torch.nextafter(mid, torch.full_like(mid, float('inf')))), 1).flatten()
        groups = torch.zeros(44, 32, device='cuda', dtype=torch.bfloat16)
        groups[:, 0] = 6
        groups[:42, 1] = values
        groups[42:, 1] = -0.
        check(groups.reshape(-1, 128), 'rounding_midpoints_and_neighbors')
        stream.synchronize()
    args.output.write_text(json.dumps({'scope': 'FP4 E2M1 / E8M0 index encoding only',
        'device': args.device, 'gpu': torch.cuda.get_device_name(args.device),
        'torch': torch.__version__, 'tilelang': tilelang.__version__,
        'reference_kernel_sha256': source_hash,
        'native_library_sha256': hashlib.sha256(args.native_lib.read_bytes()).hexdigest(),
        'cases': results}, indent=2) + '\n')


if __name__ == '__main__':
    main()
