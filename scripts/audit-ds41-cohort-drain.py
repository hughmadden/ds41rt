#!/usr/bin/env python3
"""Attribute completed rounds to active-request counts within an explicit initial cohort."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re


def summarize(path, initial_requests):
    decisions, phases = [], defaultdict(Counter)
    for raw in path.read_text().splitlines():
        if not any(s in raw for s in ['native draft policy observation', 'native scheduler round']):
            continue
        line = re.sub(r'\x1b\[[0-9;]*m', '', raw)
        f = {k: int(v) for k, v in re.findall(r'\b(\w+)=(\d+)\b', line)}
        if 'native draft policy observation' in line:
            decisions.append(f)
        elif decisions:
            # Exclude the initial prefill/cold round, retain final drain rounds.
            if max(d['request_id'] for d in decisions) <= initial_requests and min(d['generated'] for d in decisions) > 1:
                phase = phases[f['requests']]
                phase['rounds'] += 1
                for key in ['total_us', 'verify_us', 'emitted', 'proposed']:
                    phase[key] += f[key]
            decisions = []
    assert phases, 'no qualifying cohort rounds'
    return dict(trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        phases=[dict(active_requests=n, **data, emitted_per_scheduler_second=data['emitted']*1e6/data['total_us'])
                for n, data in sorted(phases.items(), reverse=True)])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before', type=Path, required=True)
    p.add_argument('--after', type=Path, required=True)
    p.add_argument('--initial-requests', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or args.initial_requests < 1:
        p.error('output must be new and cohort size positive')
    report = {name: summarize(getattr(args, name), args.initial_requests) for name in ['before', 'after']}
    report['initial_requests'] = args.initial_requests
    report['limitations'] = ['Instrumented scheduler time, not HTTP aggregate throughput.',
        'Different schedules redistribute tokens and content across phases; phase TPS is not a paired counterfactual.',
        'Initial prefill round is excluded. Caller must supply matching workloads and explicit cohort size.']
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
