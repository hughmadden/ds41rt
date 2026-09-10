#!/usr/bin/env python3
"""Audit pinned reference parameter names/geometry without allocating model weights.

This is a metadata audit, not numerical or GPU qualification; the reference's
WO-A weights are BF16 after checkpoint conversion and require separate handling.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    lock = json.loads((root / 'docs/ds41-reference-lock.json').read_text())
    reference = args.reference_dir.resolve()
    for name in ('inference/model.py', 'inference/config.json', 'inference/kernel.py',
                 'inference/vision.py', 'inference/engram.py', 'inference/image_processor.py',
                 'tokenizer.json', 'model.safetensors.index.json'):
        actual = hashlib.sha256((reference / name).read_bytes()).hexdigest()
        if actual != lock['files'][name]:
            raise ValueError(f'pinned source mismatch: {name}')
    sys.path.insert(0, str(reference / 'inference'))
    import torch
    from tokenizers import Tokenizer
    from model import ModelArgs, Transformer

    backend = Tokenizer.from_file(str(reference / 'tokenizer.json'))

    class Wrapped:
        backend_tokenizer = backend

        def __len__(self):
            return backend.get_vocab_size(with_added_tokens=True)

    model_args = ModelArgs(**json.loads((reference / 'inference/config.json').read_text()))
    prior_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device('meta'):
            model = Transformer(model_args, Wrapped())
    finally:
        torch.set_default_dtype(prior_dtype)
    parameters = {}
    for name, tensor in model.named_parameters():
        if tensor.device.type != 'meta':
            raise ValueError(f'reference unexpectedly allocated parameter {name}')
        parameters[name] = {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                            'nbytes': tensor.numel() * tensor.element_size()}
    index = json.loads((reference / 'model.safetensors.index.json').read_text())['weight_map']
    missing = sorted(set(parameters) - set(index))
    extra = sorted(set(index) - set(parameters))
    expected_extra = {f'layers.{i}.attn.wo_a.scale' for i in range(40)}
    expected_extra.update(f'mtp.{i}.attn.wo_a.scale' for i in range(3))
    if missing or set(extra) != expected_extra:
        raise ValueError(f'reference/checkpoint inventory mismatch: missing={missing}, extra={extra}')
    families = {
        'dspark_unshared': lambda n: n.startswith('mtp.'),
        'dspark_routed_experts': lambda n: n.startswith('mtp.') and '.ffn.experts.' in n,
        'vision_and_aligner': lambda n: n.startswith(('vision.', 'aligner.', 'image_')),
        'mapped_engram_tables': lambda n: '.engram.embed.' in n,
        'backbone_routed_experts': lambda n: n.startswith('layers.') and '.ffn.experts.' in n,
    }
    summary = {family: {'parameters': sum(predicate(n) for n in parameters),
                        'reference_bytes': sum(v['nbytes'] for n, v in parameters.items() if predicate(n))}
               for family, predicate in families.items()}
    result = {'revision': lock['revision'], 'reference_parameter_count': len(parameters),
              'checkpoint_tensor_count': len(index), 'checkpoint_only_wo_a_scales': extra,
              'families': summary, 'parameters': parameters}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'parameters'}, indent=2))


if __name__ == '__main__':
    main()
