#!/usr/bin/env python3
"""Export native V4.1 K32 activation quantization and 32x32-scale projections."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import _pinned_sparkinfer  # noqa: F401


def validate_abi(path: Path, label: str, kind: str) -> dict:
    header = path.read_text()
    pointers = ('source_ptr', 'values_ptr', 'scale_rows_ptr', 'scale_mma_ptr') if kind == 'quant' else (
        'a_ptr', 'b_ptr', 'sfa_ptr', 'sfb_ptr', 'c_ptr', 'quant_c_values_ptr',
        'quant_c_scale_rows_ptr', 'quant_c_scale_mma_ptr', 'alpha_ptr')
    scalars = ('m', 'grid_x') if kind == 'quant' else ('m',)
    stream = 'stream' if kind == 'quant' else 'current_stream'
    expected = [f'ds41rt_{label}_Kernel_Module_t *module']
    expected += [f'void *{name}' for name in pointers]
    expected += [f'int32_t {name}' for name in scalars] + [f'cudaStream_t {stream}']
    signature = re.search(r'static inline int32_t cute_dsl_ds41rt_' + re.escape(label) + r'_wrapper\(([^)]*)\)', header)
    if not signature or re.sub(r'\s+', '', signature[1]) != re.sub(r'\s+', '', ','.join(expected)):
        raise ValueError(f'unexpected generated ABI: {path}')
    symbols = re.findall(r'void (_mlir_\w+)\(void \*\*args, int32_t num_args\);', header)
    count = len(pointers) + len(scalars) + 2
    if len(symbols) != 1 or f'{symbols[0]}(args, {count});' not in header:
        raise ValueError(f'unexpected generated dispatch ABI: {path}')
    arguments = re.search(r'void \*args\[' + str(count) + r'\] = \{([^}]*)\}', header)
    expected_args = ','.join('&' + name for name in (*pointers, *scalars, stream, 'ret'))
    if not arguments or re.sub(r'\s+', '', arguments[1]) != expected_args:
        raise ValueError(f'unexpected generated argument order: {path}')
    return {'symbol': symbols[0], 'pointers': pointers, 'i32': scalars, 'stream': stream, 'argument_count': count}


def export(output: Path, rows: tuple[int, ...]) -> None:
    os.environ['SPARKINFER_COMPILE_DISK_CACHE'] = '0'
    os.environ['SPARKINFER_COMPILE_MEMORY_CACHE'] = '0'
    import torch
    from b12x._lib.dense_gemm import compile_dense_gemm_mxfp8_aot
    from b12x._lib.quant.mxfp8_rows import compile_mxfp8_rows_quant_aot
    from b12x.gemm._shared.block_fp8 import _block_fp8_linear_scratch_layout

    torch.cuda.init()
    device = torch.device('cuda', torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    if (props.major, props.minor) != (12, 0):
        raise ValueError('coordinator FP8 export requires native SM120')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'v41_fp8.json').unlink(missing_ok=True)
    manifest = {'schema': 1, 'role': 'coordinator', 'capability': [props.major, props.minor],
                'physical_sms': props.multi_processor_count, 'device': props.name,
                'projection': {'name': 'engram_wkv', 'n': 25600, 'k': 6144,
                               'weight_block': [32, 32], 'activation_block': 32},
                'sparkinfer_revision': json.loads((Path(__file__).resolve().parents[2] / 'third_party/sparkinfer.lock.json').read_text())['revision'],
                'variants': []}
    for capacity in rows:
        label = f'v41_engram_fp8_m{capacity}'
        quant = compile_mxfp8_rows_quant_aot(size_k=6144, scale_block_size=32, expected_m=capacity)
        quant.export_to_c(str(output), label + '_quant', 'ds41rt_' + label + '_quant')
        gemm = compile_dense_gemm_mxfp8_aot(size_m=capacity, size_n=25600, size_k=6144,
                                          expected_m=capacity, sfb_k_replicated=False, device=device)
        gemm.export_to_c(str(output), label + '_gemm', 'ds41rt_' + label + '_gemm')
        layout = _block_fp8_linear_scratch_layout(tokens=capacity, in_features=6144,
                                                out_features=25600, output_dtype=torch.bfloat16)
        manifest['variants'].append({'capacity': capacity, 'label': label,
            'activation_scratch_bytes': layout.nbytes,
            'activation_values_offset': layout.x_values_offset_bytes,
            'activation_row_scales_offset': layout.x_scale_rows_offset_bytes,
            'activation_mma_scales_offset': layout.x_scale_mma_offset_bytes,
            'activation_mma_scale_shape': list(layout.x_scale_mma_physical_shape),
            'split_k_slices': 1, 'output_dtype': 'BF16', 'output_bytes': capacity * 25600 * 2,
            'quant_abi': validate_abi(output / (label + '_quant.h'), label + '_quant', 'quant'),
            'gemm_abi': validate_abi(output / (label + '_gemm.h'), label + '_gemm', 'gemm')})
        print(f'exported {label}', flush=True)
    artifacts = {}
    for variant in manifest['variants']:
        for kind in ('quant', 'gemm'):
            for suffix in ('.h', '.o'):
                path = output / (variant['label'] + '_' + kind + suffix)
                artifacts[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest['artifacts'] = artifacts
    (output / 'v41_fp8.json').write_text(json.dumps(manifest, indent=2) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--rows', default='1,16,80,256,1024,4096')
    args = parser.parse_args()
    rows = tuple(int(value) for value in args.rows.split(','))
    if not rows or len(set(rows)) != len(rows) or any(value not in (1,16,80,256,1024,4096) for value in rows):
        parser.error('rows must be distinct native capacities: 1,16,80,256,1024,4096')
    export(args.output_dir, rows)


if __name__ == '__main__':
    main()
