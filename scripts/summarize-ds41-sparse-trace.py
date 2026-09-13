#!/usr/bin/env python3
"""Summarize attention timing/capture work between instrumented scheduler rounds."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics


def summarize(path, requests):
    rounds, pending, sparse_us = [], [], 0
    batched_shapes = Counter()
    for raw in path.read_text().splitlines():
        line = re.sub(r'\x1b\[[0-9;]*m', '', raw)
        fields = {k: int(v) for k, v in re.findall(r'\b(\w+)=(\d+)\b', line)}
        if 'sparse graph capture' in line:
            batch = 'batched=true' in line
            pending.append(dict(layer=fields['layer'], rows=fields['rows'], batched=batch))
            if batch:
                batched_shapes[(fields['layer'], fields['rows'])] += 1
        if 'target attention stages' in line:
            sparse_us += fields['sparse_us']
        if 'native scheduler round speculative=true' in line:
            if fields['requests'] == requests:
                rounds.append(dict(sparse_us=sparse_us, captures=pending, **fields))
            pending, sparse_us = [], 0
    assert rounds, f'no matching rounds in {path}'
    captured = [r for r in rounds if r['captures']]
    return dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), requests=requests,
        rounds=len(rounds), median_sparse_ms=statistics.median(r['sparse_us'] for r in rounds)/1000,
        median_verify_ms=statistics.median(r['verify_us'] for r in rounds)/1000,
        median_total_ms=statistics.median(r['total_us'] for r in rounds)/1000,
        median_proposed=statistics.median(r['proposed'] for r in rounds),
        median_emitted=statistics.median(r['emitted'] for r in rounds),
        rounds_with_captures=len(captured),
        median_sparse_ms_with_captures=statistics.median(r['sparse_us'] for r in captured)/1000 if captured else None,
        batched_shapes=[dict(layer=k[0], rows=k[1], captures=v) for k, v in sorted(batched_shapes.items())],
        raw_rounds=rounds)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before', type=Path, required=True)
    p.add_argument('--after', type=Path, required=True)
    p.add_argument('--requests', type=int, default=16)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert not args.output.exists()
    report = {arm: summarize(getattr(args, arm), args.requests) for arm in ['before', 'after']}
    report['limitations'] = ['Instrumented sequential runs; row counts, routing and emitted token counts may differ.',
        'Capture counts group layer/rows across both lanes; the trace does not identify individual wave ownership.',
        'Work is attributed to the following completed speculative scheduler round; initial rounds may include prefill.']
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({arm: {k:v for k,v in report[arm].items() if k not in ['batched_shapes', 'raw_rounds']}
                      for arm in ['before', 'after']}, indent=2))

if __name__ == '__main__':
    main()
