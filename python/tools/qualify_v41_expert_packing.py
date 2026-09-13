#!/usr/bin/env python3
"""Compare native TP4/TP2/full expert packing with SparkInfer's tensor converter."""
import argparse
import ctypes as C
import json
import torch
import _pinned_sparkinfer
from b12x.moe.fused_moe._impl import (
    _logical_weight_to_w4a8_rp_inplace as pack_weight,
    _e8m0_scale_to_w4a8_sfb_inplace as pack_scale,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-lib", required=True)
    args = parser.parse_args()
    lib = C.CDLL(args.native_lib)
    ptr = C.c_void_p
    lib.ds41rt_v41_expert_packed_sizes.argtypes = [C.c_uint32, C.POINTER(C.c_uint64)]
    lib.ds41rt_v41_pack_expert_async.argtypes = [C.POINTER(ptr), C.POINTER(ptr), C.c_uint32, ptr]
    results = []
    for device in range(min(2, torch.cuda.device_count())):
        with torch.cuda.device(device):
            for n in (576, 1152, 2304):
                torch.manual_seed(4100 + n)
                h = 5120
                shapes = [(1, n, h // 2), (1, n, h // 2), (1, h, n // 2)]
                source = [torch.randint(0, 256, shape, device="cuda", dtype=torch.uint8)
                          for shape in shapes]
                # SparkInfer clamps E8M0 bytes above 247; compare its unmodified
                # domain because the native checkpoint packer preserves raw scales.
                source += [torch.randint(0, 248, (*s[:-1], s[-1] // 16), device="cuda", dtype=torch.uint8)
                           for s in shapes]
                sizes = (C.c_uint64 * 4)()
                assert lib.ds41rt_v41_expert_packed_sizes(n, sizes) == 0
                expected = [
                    pack_weight(torch.cat([source[1], source[0]], 1), size_k=h, size_n=2*n, gated_half_rows=n),
                    pack_scale(torch.cat([source[4], source[3]], 1), weight_E=1, rows=2*n, k_dim=h, gated_half_rows=n),
                    pack_weight(source[2].clone(), size_k=n, size_n=h),
                    pack_scale(source[5].clone(), weight_E=1, rows=h, k_dim=n),
                ]
                output = [torch.full((size + 32,), 205, device="cuda", dtype=torch.uint8) for size in sizes]
                assert lib.ds41rt_v41_pack_expert_async(
                    (ptr * 6)(*[s.data_ptr() for s in source]),
                    (ptr * 4)(*[s.data_ptr() for s in output]), n,
                    torch.cuda.current_stream().cuda_stream) == 0
                for actual, reference, size in zip(output, expected, sizes):
                    reference = reference.view(torch.uint8).flatten()
                    assert reference.numel() == size
                    assert torch.equal(actual[:size], reference), (device, n, size)
                    assert bool((actual[size:] == 205).all())
                results.append(dict(device=device, intermediate=n, packed_bytes=list(sizes), exact=True))
    assert results, "no CUDA devices tested"
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
