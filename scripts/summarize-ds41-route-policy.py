#!/usr/bin/env python3
"""Summarize observed native route reuse and instrumented complete FFN times."""
import argparse
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics


def parse(text):
    observations = []
    for line in re.sub(r'\x1b\[[0-9;]*m', '', text).splitlines():
        if 'native route policy observation' not in line:
            continue
        fields = dict(re.findall(r'(\w+)=(.*?)(?= \w+=|$)', line))
        owners = ast.literal_eval(fields['owners'])
        routes = json.loads(fields['route_ids'])
        rows = int(fields['rows'])
        if len(owners) != rows or len(routes) != 6 * rows:
            raise ValueError('route/owner geometry differs')
        if any(not 0 <= expert < 384 for expert in routes):
            raise ValueError('expert outside backbone vocabulary')
        unique = len(set(routes))
        if unique != int(fields['unique_experts']):
            raise ValueError('unique expert count differs')
        observations.append(dict(layer=int(fields['layer']), rows=rows, owners=owners,
            unique=unique, ffn_us=int(fields['ffn_us'])))
    return observations


def summarize(observations):
    groups = defaultdict(list)
    passes = defaultdict(dict)
    for r in observations:
        groups[r['rows']].append(r)
        key = tuple(tuple(owner) for owner in r['owners'])
        if r['layer'] in passes[key]:
            raise ValueError('duplicate layer for the same request/position batch')
        passes[key][r['layer']] = r
    def stats(values):
        return dict(minimum=min(values), median=statistics.median(values), maximum=max(values))
    return dict(scope=__doc__, observations=len(observations), by_rows=[dict(rows=rows,
        samples=len(records), unique_experts=stats([r['unique'] for r in records]),
        ffn_us=stats([r['ffn_us'] for r in records])) for rows, records in sorted(groups.items())],
        complete_layer_passes=[dict(rows=records[0]['rows'], requests=len({o[0] for o in key}),
            first_owner=key[0], mean_unique_experts=statistics.mean(r['unique'] for r in records.values()),
            summed_ffn_us=sum(r['ffn_us'] for r in records.values()))
            for key, records in passes.items() if set(records) == set(range(40))],
        incomplete_layer_passes=sum(set(records) != set(range(40)) for records in passes.values()),
        limitations='FFN times include routing, host dispatch, shared work and response completion. Summed layer times are not whole-pass latency when lanes overlap. Unique packed experts are not measured DRAM traffic; routes are observed after target execution, not known to the draft policy in advance.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('trace', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    raw = a.trace.read_bytes()
    observations = parse(raw.decode())
    if not observations:
        p.error('no route observations')
    report = summarize(observations)
    report['trace_sha256'] = hashlib.sha256(raw).hexdigest()
    with a.output.open('x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()
