#!/usr/bin/env python3
"""Compare common BF16 residual rows from two opt-in target activation traces."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def compare(left, right, common_rows):
    positions = [json.loads((p / 'positions.json').read_text()) for p in (left, right)]
    if common_rows < 1 or any(len(p) < common_rows for p in positions):
        raise ValueError('common row count exceeds trace')
    if positions[0][:common_rows] != positions[1][:common_rows]:
        raise ValueError('common token positions differ')
    files = [set(p.name for p in d.glob('layer*-residual.bin')) for d in (left, right)]
    if not files[0] or files[0] != files[1]:
        raise ValueError('trace layer coverage differs or is empty')
    records = []
    for name in sorted(files[0], key=lambda n: int(n.split('-')[0][5:])):
        raw = [(d / name).read_bytes() for d in (left, right)]
        if any(len(b) != len(p) * 40960 for b, p in zip(raw, positions)):
            raise ValueError('invalid BF16 residual geometry')
        values = [(np.frombuffer(b, dtype='<u2').astype(np.uint32) << 16).view(np.float32)
                  .reshape(-1, 20480)[:common_rows].astype(np.float64) for b in raw]
        if not all(np.isfinite(v).all() for v in values):
            raise ValueError('non-finite residual')
        delta = values[0] - values[1]
        rms = np.sqrt(np.mean(delta * delta, axis=1))
        reference_rms = np.sqrt(np.mean(values[0] * values[0], axis=1))
        records.append(dict(layer=int(name.split('-')[0][5:]),
            sha256=[hashlib.sha256(b).hexdigest() for b in raw],
            changed_elements=np.count_nonzero(delta, axis=1).tolist(),
            max_abs=np.max(abs(delta), axis=1).tolist(), rms=rms.tolist(),
            reference_rms=reference_rms.tolist()))
    return dict(scope=__doc__, positions=positions, common_rows=common_rows,
                first_differing_layer=next((r['layer'] for r in records if any(r['changed_elements'])), None),
                layers=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left', type=Path)
    parser.add_argument('right', type=Path)
    parser.add_argument('--common-rows', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.left, args.right, args.common_rows)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()
