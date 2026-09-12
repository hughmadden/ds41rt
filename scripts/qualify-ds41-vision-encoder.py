#!/usr/bin/env python3
"""Compare complete native vision vectors with the pinned official encoder.

Reports every saved stage, delimiters, warm reference time, and source hashes.
This is numerical component evidence, not image-answer or serving qualification.
Run native_full_vision_vectors first, separately for BF16 and FP32 attention.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from safetensors import safe_open
import torch
import torch.nn.functional as F


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', type=Path, default=Path('/tmp/ds41-reference'))
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--vectors', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--reference-spans', type=Path, help='optional directory for complete reference span vectors')
    parser.add_argument('--compare-reference-spans', type=Path, help='compare another reference backend using its saved complete spans')
    parser.add_argument('--reference-attention', choices=('math', 'flash'), default='math',
        help='math is unmodified pinned execution; flash adds a batch dimension for a rounding control')
    parser.add_argument('--max-relative-l2', type=float, default=0.03)
    parser.add_argument('--min-cosine', type=float, default=0.9995)
    args = parser.parse_args()
    if args.report.exists():
        raise ValueError('preserve evidence: report already exists')
    if args.reference_spans:
        args.reference_spans.mkdir(parents=True, exist_ok=False)
    source = args.reference_dir / 'inference/vision.py'
    lock = json.loads(Path('docs/ds41-reference-lock.json').read_text())
    assert digest(source) == lock['files']['inference/vision.py']
    spec = importlib.util.spec_from_file_location('pinned_vision', source)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    if args.reference_attention == 'flash':
        original_sdpa = F.scaled_dot_product_attention
        def flash_sdpa(q, k, v):
            from torch.nn.attention import sdpa_kernel, SDPBackend
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return original_sdpa(q[None], k[None], v[None])[0]
        F.scaled_dot_product_attention = flash_sdpa
    config = SimpleNamespace(vision_patch_size=14, vision_dim=1024, vision_n_heads=16,
        vision_inter_dim=2816, vision_rope_theta=10000, vision_downsample_ratio=3,
        vision_n_layers=32, dim=5120)
    torch.set_default_dtype(torch.bfloat16)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    with torch.device('cuda'):
        vit, aligner = reference.ViT(config).eval(), reference.Aligner(config).eval()
    index = args.model / 'model.safetensors.index.json'
    weight_map = json.loads(index.read_text())['weight_map']
    names = [k for k in weight_map if k.startswith(('vision.', 'aligner.')) or
        k in ('image_start', 'image_end', 'image_newline')]
    assert len(names) == 266
    weights = {}
    for shard in sorted({weight_map[k] for k in names}):
        with safe_open(args.model / shard, framework='pt', device='cpu') as file:
            for key in names:
                if weight_map[key] == shard:
                    weights[key] = file.get_tensor(key)
    vit.load_state_dict({k.removeprefix('vision.'): v for k, v in weights.items()
        if k.startswith('vision.')}, strict=True)
    aligner.load_state_dict({k.removeprefix('aligner.'): v for k, v in weights.items()
        if k.startswith('aligner.')}, strict=True)
    delimiters = {k: weights[k].cuda().reshape(1, 5120) for k in
        ('image_start', 'image_end', 'image_newline')}
    del weights
    manifest = json.loads((args.vectors / 'cases.json').read_text())
    report = dict(scope=__doc__, reference_revision=lock['revision'],
        reference_sha256=digest(source), reference_attention=args.reference_attention,
        checkpoint_index_sha256=digest(index),
        torch=torch.__version__, gpu=torch.cuda.get_device_name(),
        native=manifest, max_relative_l2=args.max_relative_l2, min_cosine=args.min_cosine,
        cases=[], passed=False)

    maps = Path('/proc/self/maps')
    report['loaded_cuda_libraries'] = sorted({line.split()[-1] for line in
        maps.read_text().splitlines() if '/libcublas' in line or '/libcudart' in line}) if maps.exists() else []

    def save():
        args.report.write_text(json.dumps(report, indent=2) + '\n')

    def compare(path, expected):
        raw = path.read_bytes()
        actual = torch.from_numpy(np.frombuffer(raw, dtype='<u2').copy()).view(torch.bfloat16)
        actual = actual.reshape(expected.shape).double().cpu()
        expected_raw = expected.detach().contiguous().cpu()
        expected = expected_raw.double()
        delta = actual - expected
        relative = float(delta.norm() / expected.norm().clamp_min(1e-30))
        cosine = float(F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0))
        finite = bool(actual.isfinite().all() and expected.isfinite().all())
        return dict(elements=actual.numel(), finite=finite, relative_l2=relative, cosine=cosine,
            max_absolute=float(delta.abs().max()), mean_absolute=float(delta.abs().mean()),
            identical_fraction=float((actual == expected).float().mean()),
            native_sha256=hashlib.sha256(raw).hexdigest(),
            reference_sha256=hashlib.sha256(expected_raw.view(torch.uint8).numpy().tobytes()).hexdigest(),
            passed=finite and relative <= args.max_relative_l2 and cosine >= args.min_cosine)

    with torch.inference_mode():
        for case in manifest['cases']:
            prefix, grid = case['prefix'], case['grid']
            h, w = grid['vit_height'], grid['vit_width']
            raw = (args.vectors / f'{prefix}-patches.bin').read_bytes()
            patches = torch.from_numpy(np.frombuffer(raw, dtype='<u2').copy()).view(torch.bfloat16)
            patches = patches.reshape(h*w, 3, 14, 14).cuda()
            traces = {t['name']: args.vectors / t['file'] for t in case['traces']}
            result = dict(prefix=prefix, grid=grid, stages={})
            report['cases'].append(result)

            def observe(name, tensor):
                if name in traces:
                    result['stages'][name] = compare(traces[name], tensor)
                    metric = result['stages'][name]
                    print(prefix, name, 'relL2', metric['relative_l2'], 'cos', metric['cosine'], flush=True)
                    save()

            x = vit.patch_embed(patches)
            observe('patch', x)
            with torch.device('cuda'):
                cos, sin = reference.get_vision_cos_sin(h, w, vit.rope_dim, vit.rope_theta)
            for layer, block in enumerate(vit.blocks):
                x = block(x, cos, sin)
                observe(f'block-{layer}', x)
            x = vit.norm(x)
            observe('norm', x)
            merged = F.pad(x.view(h, w, -1).permute(2, 0, 1), (0, -w % 3, 0, -h % 3))
            merged = F.unfold(merged.unsqueeze(0), 3, stride=3).squeeze(0).transpose(0, 1)
            observe('merged', merged)
            aligned = aligner(x, h, w)
            observe('aligned', aligned)
            lines = aligned.reshape(grid['llm_height'], grid['llm_width'], 5120)
            span = torch.cat([delimiters['image_start'], *[
                part for line in lines for part in (line, delimiters['image_newline'])],
                delimiters['image_end']])
            result['span'] = compare(args.vectors / f'{prefix}-span.bin', span)
            if args.reference_spans:
                (args.reference_spans / f'{prefix}-span.bin').write_bytes(
                    span.contiguous().cpu().view(torch.uint8).numpy().tobytes())
            if args.compare_reference_spans:
                result['reference_backend_difference'] = compare(
                    args.compare_reference_spans / f'{prefix}-span.bin', span)
            native_span = torch.from_numpy(np.frombuffer(
                (args.vectors / f'{prefix}-span.bin').read_bytes(), dtype='<u2').copy())
            native_span = native_span.view(torch.bfloat16).reshape(-1, 5120)
            positions = [0, span.shape[0]-1] + [1+i*(grid['llm_width']+1)+grid['llm_width']
                for i in range(grid['llm_height'])]
            result['delimiters_exact'] = torch.equal(native_span[positions], span.cpu()[positions])
            # Reuse loaded weights and position cache, no tracing/copies in warm timing.
            torch.cuda.synchronize()
            start = time.perf_counter()
            aligner(vit(patches, h, w), h, w)
            torch.cuda.synchronize()
            result['reference_encode_seconds'] = time.perf_counter() - start
            result['passed'] = result['span']['passed'] and result['delimiters_exact'] and all(
                v['passed'] for v in result['stages'].values())
            print(prefix, 'passed', result['passed'], 'reference seconds', result['reference_encode_seconds'], flush=True)
            save()
    report['passed'] = all(case['passed'] for case in report['cases'])
    save()
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
