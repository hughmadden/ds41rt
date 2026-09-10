#!/usr/bin/env python3
"""Independently resolve owned attention dumps and check every output coordinate."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vectors-dir', type=Path, required=True)
    p.add_argument('--device', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    primitive = Path(__file__).with_name('qualify-ds41-sparse-attention.py')
    spec = importlib.util.spec_from_file_location('qualified_sparse_math', primitive)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    torch.cuda.set_device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    results = []
    with torch.device('cuda'), torch.no_grad():
        for layer in [0, 2, 7, 8, 14, 20, 24, 39]:
            root = a.vectors_dir / f'l{layer}'
            hashes = {}
            def load(name, dtype, columns):
                raw = (root / name).read_bytes()
                hashes[name] = hashlib.sha256(raw).hexdigest()
                return torch.frombuffer(bytearray(raw), dtype=dtype).reshape(-1, columns).cuda()
            def kv(name):
                v = load(name + '.bin', torch.uint8, 512).view(torch.float8_e4m3fn).float()
                s = load(name + '_scales.bin', torch.uint8, 16).float()
                return (v * torch.exp2(s - 127).repeat_interleave(32, -1)).bfloat16()
            q = load('query.bin', torch.bfloat16, 64 * 512).reshape(80, 64, 512)
            sink = load('sink.bin', torch.float32, 64).flatten()
            selected = load('selected.bin', torch.int32, 512).cpu().tolist() if layer >= 2 else None
            actual = load('out0.bin', torch.bfloat16, 64 * 512).reshape_as(q)
            requests = json.loads((root / 'requests.json').read_text())
            hashes['requests.json'] = hashlib.sha256((root / 'requests.json').read_bytes()).hexdigest()
            maximum = 0.
            row = 0
            for i, request in enumerate(requests):
                ring = kv(f'r{i}-ring')
                window = kv(f'r{i}-window')
                if selected is not None:
                    pool = kv(f'r{i}-pool')
                    source = kv(f'r{i}-source')
                    pages = load(f'r{i}-pages.bin', torch.uint32, 1).flatten().cpu().tolist()
                for m in request['metadata']:
                    vectors = []
                    valid = []
                    for pos in range(max(0, m[3] - 127), m[3] + 1):
                        vectors.append(ring[pos % 128] if pos < m[0] else window[m[1] + pos - m[0]])
                        valid.append(True)
                    while len(vectors) < 128:
                        vectors.append(torch.zeros(512, dtype=torch.bfloat16)); valid.append(False)
                    if selected is not None:
                        for index in selected[row]:
                            if index < 0 or index >= m[5]:
                                vectors.append(torch.zeros(512, dtype=torch.bfloat16)); valid.append(False)
                            elif index < m[6]:
                                vectors.append(pool[pages[index // 256] * 256 + index % 256]); valid.append(True)
                            else:
                                assert index - m[6] < m[7]
                                vectors.append(source[m[8] + (index - m[6]) * m[9]]); valid.append(True)
                    expected = reference.oracle(q[row:row + 1], torch.stack(vectors)[None],
                                                torch.tensor(valid, dtype=torch.bool)[None], sink)
                    torch.testing.assert_close(actual[row:row + 1], expected, rtol=.008, atol=.002)
                    maximum = max(maximum, (actual[row:row + 1].float() - expected.float()).abs().max().item())
                    row += 1
            assert row == 80
            for name in ['out1.bin', 'recovered.bin'] + (['resized.bin'] if layer == 0 else []):
                torch.testing.assert_close(load(name, torch.bfloat16, 64 * 512).reshape_as(actual), actual, rtol=0, atol=0)
            results.append(dict(layer=layer, requests=16, rows=row, max_abs=maximum,
                                replay_recovery_exact=True, payloads_sha256=hashes))
            print(f'PASS layer={layer} requests=16 rows={row} max_abs={maximum}', flush=True)
    a.output.write_text(json.dumps(dict(device=a.device, oracle_source_sha256=hashlib.sha256(primitive.read_bytes()).hexdigest(),
        scope='Real KV/index/selection/sink producers, finite query fixture; owned composition versus independent host addressing and qualified blockwise math',
        cases=results), indent=2) + '\n')


if __name__ == '__main__':
    main()
