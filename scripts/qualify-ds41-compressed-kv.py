#!/usr/bin/env python3
"""Compare native compressed-KV packing with the pinned DeepSeek FP4 quantizer."""
import argparse
import ast
import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
import tvm_ffi


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-dir', type=Path, required=True)
    p.add_argument('--native-lib', type=Path, required=True)
    p.add_argument('--vectors-dir', type=Path, required=True)
    p.add_argument('--device', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    lock = json.loads((root / 'docs/ds41-reference-lock.json').read_text())
    for name in ('model.py', 'kernel.py'):
        assert hashlib.sha256((a.reference_dir / 'inference' / name).read_bytes()).hexdigest() == lock['files']['inference/' + name]
    source = a.reference_dir / 'inference/model.py'
    nodes = [n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'apply_rotary_emb']
    ns = {'torch': torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), ns)
    spec = importlib.util.spec_from_file_location('official_kernel', a.reference_dir / 'inference/kernel.py')
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    assert ref.tilelang.__version__ == '0.1.8'
    ref.fp4_quant_kernel.pass_configs = {**ref.fp4_quant_kernel.pass_configs, 'tir.disable_vectorize': True}
    ref.act_quant_kernel.pass_configs = {**ref.act_quant_kernel.pass_configs, 'tir.disable_vectorize': True}
    lib = ctypes.CDLL(str(a.native_lib.resolve()))
    ptr = ctypes.c_void_p
    for name in ('ds41rt_v41_compressed_kv_pack', 'ds41rt_v41_kv_pack'):
        fn = getattr(lib, name)
        fn.argtypes = [ptr, ptr, ptr, ptr, ctypes.c_int, ptr]
        fn.restype = ctypes.c_int
    store = lib.ds41rt_v41_compressed_kv_store
    store.argtypes = [ptr] * 5 + [ctypes.c_int, ctypes.c_uint64, ptr]
    store.restype = ctypes.c_int
    torch.cuda.set_device(a.device)
    stream = torch.cuda.Stream()
    results = []
    paths = sorted(a.vectors_dir.glob('c*-latent.bin'))
    assert paths
    with torch.cuda.stream(stream), tvm_ffi.use_torch_stream(), torch.no_grad():
        def check(name, x, freq=None):
            rows = x.numel() // 512
            x = x.reshape(1, rows, 512)
            values = torch.empty((rows, 256), dtype=torch.uint8, device='cuda')
            scales = torch.empty((rows, 32), dtype=torch.uint8, device='cuda')
            assert lib.ds41rt_v41_compressed_kv_pack(x.data_ptr(), None if freq is None else freq.data_ptr(), values.data_ptr(), scales.data_ptr(), rows, stream.cuda_stream) == 0
            expected = x.clone()
            if freq is not None:
                ns['apply_rotary_emb'](expected[..., -64:], torch.view_as_complex(freq))
            expected_scales = (expected.float().reshape(rows, 32, 16).abs().amax(-1).clamp_min(6 * 2**-9) / 6).to(torch.float8_e4m3fn)
            assert torch.equal(scales, expected_scales.view(torch.uint8)), name + ': scales'
            reference_inplace = expected.clone()
            ref.fp4_act_quant(reference_inplace, 16, True, scale_dtype=torch.float8_e4m3fn)
            magnitudes = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device='cuda')
            nibbles = torch.stack((values & 15, values >> 4), dim=-1).reshape(rows, 512)
            decoded = magnitudes[(nibbles & 7).long()] * torch.where(nibbles >= 8, -1., 1.)
            decoded = (decoded * expected_scales.float().repeat_interleave(16, dim=-1)).to(torch.bfloat16)
            assert torch.equal(decoded, reference_inplace.reshape(rows, 512)), name + ': decoded FP4'
            # The unchanged window pack remains byte-identical to its reference.
            wv = torch.empty((rows, 512), dtype=torch.uint8, device='cuda')
            ws = torch.empty((rows, 16), dtype=torch.uint8, device='cuda')
            assert lib.ds41rt_v41_kv_pack(x.data_ptr(), None if freq is None else freq.data_ptr(), wv.data_ptr(), ws.data_ptr(), rows, stream.cuda_stream) == 0
            wq, wscale = ref.act_quant(expected, 32, 'ue8m0', torch.float8_e8m0fnu)
            assert torch.equal(wv, wq.view(torch.uint8).reshape(rows, 512)), name + ': window values'
            assert torch.equal(ws, wscale.view(torch.uint8).reshape(rows, 16)), name + ': window scales'
            # Reverse scatter, skip one proposal, and preserve untouched sentinels.
            capacity = rows + 3
            destinations = torch.arange(rows - 1, -1, -1, dtype=torch.int64, device='cuda')
            destinations[0] = capacity
            cv = torch.full((capacity, 256), 165, dtype=torch.uint8, device='cuda')
            cs = torch.full((capacity, 32), 165, dtype=torch.uint8, device='cuda')
            ev, es = cv.clone(), cs.clone()
            ev[destinations[1:]] = values[1:]
            es[destinations[1:]] = scales[1:]
            assert store(values.data_ptr(), scales.data_ptr(), destinations.data_ptr(), cv.data_ptr(), cs.data_ptr(), rows, capacity, stream.cuda_stream) == 0
            assert torch.equal(cv, ev) and torch.equal(cs, es), name + ': scatter'
            results.append(dict(case=name, rows=rows, decoded_values_exact=True, scales_exact=True, window_exact=True, scatter_exact=True))
            print('PASS', name, rows, flush=True)

        for path in paths:
            prefix = path.name.split('-')[0]
            x = torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.bfloat16).cuda()
            freq = torch.frombuffer(bytearray((path.parent / (prefix + '-freq.bin')).read_bytes()), dtype=torch.float32).reshape(-1, 32, 2).cuda()
            check(prefix, x, freq)
        torch.manual_seed(41)
        for rows in (1, 35, 129, 4096):
            check(f'random-{rows}', torch.randn((rows, 512), device='cuda').to(torch.bfloat16))
        check('zeros', torch.zeros((1, 512), device='cuda', dtype=torch.bfloat16))
        # Sweep every finite BF16 value within the representable E4M3 scale range.
        bits = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).float()
        bits = bits[torch.isfinite(bits) & (bits.abs() <= 2688)]
        bits = torch.cat((bits, torch.zeros((-bits.numel()) % 512)))
        check('bf16-sweep', bits.to(device='cuda', dtype=torch.bfloat16))
        stream.synchronize()
    a.output.write_text(json.dumps(dict(scope='Compressed FP4 primitive only; serving integration pending', device=a.device, reference_revision=lock['revision'], compiler_overrides={'tir.disable_vectorize': True}, cases=results), indent=2) + '\n')


if __name__ == '__main__':
    main()
