#!/usr/bin/env python3
"""Exact index-cache scatter, skipped-row and boundary checks on a native library."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-lib', type=Path, required=True)
    parser.add_argument('--device', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    torch.manual_seed(41256)
    lib = C.CDLL(str(args.native_lib))
    store = lib.ds41rt_v41_index_store
    store.argtypes = [C.c_void_p] * 5 + [C.c_int32, C.c_uint64, C.c_void_p]
    store.restype = C.c_int32
    stream = torch.cuda.Stream()
    results = []
    with torch.cuda.stream(stream), torch.no_grad():
        for rows, capacity in ((1, 1), (3, 257), (80, 1024), (4096, 8192), (4096, 16777216)):
            x = torch.randint(256, (rows, 64), device='cuda', dtype=torch.uint8)
            s = torch.randint(256, (rows, 4), device='cuda', dtype=torch.uint8)
            y = torch.full((capacity, 64), 0xa5, device='cuda', dtype=torch.uint8)
            ys = torch.full((capacity, 4), 0xa5, device='cuda', dtype=torch.uint8)
            # Include the highest physical row and reverse ordering; valid IDs are unique.
            ids = torch.arange(capacity - 1, capacity - rows - 1, -1, device='cuda', dtype=torch.int64)
            ids[1::3] = -1  # UINT64_MAX
            ids[2::3] = capacity
            expected, expected_scales = y.clone(), ys.clone()
            def launch():
                assert store(x.data_ptr(), s.data_ptr(), ids.data_ptr(), y.data_ptr(),
                             ys.data_ptr(), rows, capacity, stream.cuda_stream) == 0
            def compare():
                valid = (ids >= 0) & (ids < capacity)
                expected[ids[valid]] = x[valid]
                expected_scales[ids[valid]] = s[valid]
                assert torch.equal(y, expected)
                assert torch.equal(ys, expected_scales)
            launch()
            compare()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                launch()
            x.bitwise_xor_(0xff)
            s.bitwise_xor_(0xff)
            ids.copy_(torch.arange(rows, device='cuda', dtype=torch.int64))
            graph.replay()
            compare()
            ids.fill_(-1)
            graph.replay()
            compare()
            ptrs = [x.data_ptr(), s.data_ptr(), ids.data_ptr(), y.data_ptr(), ys.data_ptr()]
            for i in range(5):
                bad = ptrs.copy()
                bad[i] = 0
                assert store(*bad, rows, capacity, stream.cuda_stream) != 0
                bad[i] = (1 << 64) - 1
                assert store(*bad, rows, capacity, stream.cuda_stream) != 0
                for j in range(i):
                    bad = ptrs.copy()
                    bad[i] = ptrs[j]
                    assert store(*bad, rows, capacity, stream.cuda_stream) != 0
            bad = ptrs.copy()
            bad[2] += 1
            assert store(*bad, rows, capacity, stream.cuda_stream) != 0
            for n, c in ((0, capacity), (-1, capacity), (4097, capacity), (rows, 0), (rows, 16777217)):
                assert store(*ptrs, n, c, stream.cuda_stream) != 0
            compare()
            results.append({'rows': rows, 'capacity': capacity, 'full_cache_exact': True,
                            'changed_graph_exact': True, 'skips_and_guards': True})
            print(f'PASS rows={rows} capacity={capacity}', flush=True)
            del graph, x, s, y, ys, expected, expected_scales, ids
        stream.synchronize()
    args.output.write_text(json.dumps({'scope': 'Native index scatter only',
        'gpu': torch.cuda.get_device_name(args.device), 'device': args.device,
        'native_library_sha256': hashlib.sha256(args.native_lib.read_bytes()).hexdigest(),
        'cases': results}, indent=2) + '\n')


if __name__ == '__main__':
    main()
