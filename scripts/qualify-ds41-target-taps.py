#!/usr/bin/env python3
"""Compare target tap exports and optional dSpark main context with the pinned reference."""
import argparse
import ast
import hashlib
import importlib.util
import json
import struct
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference-dir', 'snapshot', 'input-dir', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', type=int, required=True)
    args = parser.parse_args()
    lock = json.loads((Path(__file__).resolve().parents[1] / 'docs/ds41-reference-lock.json').read_text())
    for name in ('model.py', 'kernel.py'):
        path = args.reference_dir / 'inference' / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == lock['files']['inference/' + name]
    source = args.reference_dir / 'inference/model.py'
    tree = ast.parse(source.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RMSNorm']
    namespace = dict(torch=torch, nn=torch.nn)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    spec = importlib.util.spec_from_file_location('ds41_official_kernel', args.reference_dir / 'inference/kernel.py')
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    assert reference.tilelang.__version__ == '0.1.8'
    reference.act_quant_kernel.pass_configs = {
        **reference.act_quant_kernel.pass_configs, 'tir.disable_vectorize': True,
    }
    torch.cuda.set_device(args.device)
    torch.set_default_dtype(torch.bfloat16)
    torch.backends.cuda.matmul.allow_tf32 = False
    index = json.loads((args.snapshot / 'model.safetensors.index.json').read_text())['weight_map']
    hashes, weights = {}, {}

    def weight(name, dtype):
        if name not in weights:
            with (args.snapshot / index[name]).open('rb') as file:
                length = struct.unpack('<Q', file.read(8))[0]
                entry = json.loads(file.read(length))[name]
                begin, end = entry['data_offsets']
                file.seek(8 + length + begin)
                raw = file.read(end - begin)
                assert len(raw) == end - begin
            hashes[name] = hashlib.sha256(raw).hexdigest()
            weights[name] = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(entry['shape']).cuda()
        return weights[name]

    def read(path):
        return torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.bfloat16)

    records = []
    with torch.inference_mode():
        for path in sorted(args.input_dir.glob('cycle*-tap-metadata.json')):
            metadata = json.loads(path.read_text())
            batch, rows = metadata['batch'], metadata['rows']
            expected = torch.cat([
                read(args.input_dir / f'batch{batch}-layer{layer}-tap-input.bin')
                .reshape(rows, 4, 5120).mean(dim=1) for layer in (37, 38, 39)
            ], dim=1)
            actual = read(args.input_dir / f'batch{batch}-taps.bin').reshape(rows, 15360)
            assert torch.isfinite(actual).all()
            assert torch.equal(actual.view(torch.int16), expected.view(torch.int16)), path
            record = dict(metadata=path.name, batch=batch, rows=rows, taps_byte_exact=True)
            main_path = args.input_dir / path.name.replace('-tap-metadata.json', '-main-context.bin')
            if main_path.exists():
                q, scales = reference.act_quant(expected.cuda().contiguous(), 32, 'ue8m0', torch.float8_e8m0fnu)
                projected = reference.fp8_gemm(q, scales,
                    weight('mtp.0.main_proj.weight', torch.float8_e4m3fn),
                    weight('mtp.0.main_proj.scale', torch.float8_e8m0fnu),
                    torch.float8_e8m0fnu, block_size=32)
                norm = namespace['RMSNorm'](5120, 1e-20).cuda()
                norm.weight.copy_(weight('mtp.0.main_norm.weight', torch.bfloat16))
                expected_main = norm(projected).double().cpu()
                actual_main = read(main_path).reshape(rows, 5120).double()
                assert torch.isfinite(actual_main).all()
                relative = float(torch.linalg.vector_norm(actual_main - expected_main) /
                                 torch.linalg.vector_norm(expected_main).clamp_min(1e-30))
                record['main_context'] = dict(relative_l2=relative,
                    max_abs=float((actual_main - expected_main).abs().max()),
                    cosine=float(torch.nn.functional.cosine_similarity(actual_main.flatten(), expected_main.flatten(), dim=0)))
                assert relative < 0.005, record
            records.append(record)
            print('PASS', record, flush=True)
    assert records, 'no target tap exports found'
    args.output.write_text(json.dumps(dict(scope='Target input tap means and optional real dSpark main projection/norm; not proposal or acceptance qualification.',
        reference_files=lock['files'], checkpoint_tensor_sha256=hashes, results=records), indent=2) + '\n')


if __name__ == '__main__':
    main()
