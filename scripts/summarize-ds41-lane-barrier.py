#!/usr/bin/env python3
"""Summarize host-visible verifier join waits; not removable time or GPU idle time."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics


def summarize(path):
    text = re.sub(r'\x1b\[[0-9;]*m', '', path.read_text())
    groups = {}
    ignored_single_lane = 0
    for line in text.splitlines():
        if 'native scheduler round' not in line:
            continue
        fields = dict(re.findall(r'(\w+)=([\w.]+)', line))
        if 'lane0_join_wait_us' not in fields:
            raise ValueError('trace predates lane completion instrumentation')
        row = {k: int(fields[k]) for k in ['requests', 'lane0', 'lane1', 'emitted',
            'draft_us', 'verify_us', 'total_us', 'lane0_verify_done_us',
            'lane1_verify_done_us', 'lane0_join_wait_us', 'lane1_join_wait_us']}
        assert row['lane0'] + row['lane1'] == row['requests']
        if not row['lane0'] or not row['lane1']:
            ignored_single_lane += 1
            continue
        joins = [row[f'lane{i}_verify_done_us'] + row[f'lane{i}_join_wait_us'] for i in range(2)]
        assert joins[0] == joins[1] and joins[0] <= row['total_us']
        assert 0 <= row['verify_us'] <= joins[0] and row['emitted'] > 0
        row['max_join_wait_us'] = max(row['lane0_join_wait_us'], row['lane1_join_wait_us'])
        groups.setdefault(str(row['requests']), []).append(row)
    summaries = {}
    for concurrency, rows in groups.items():
        summary = {'rounds': len(rows), 'emitted_tokens': sum(r['emitted'] for r in rows)}
        for key in ['draft_us', 'verify_us', 'total_us', 'max_join_wait_us']:
            values = sorted(r[key] / 1000 for r in rows)
            summary[key.replace('_us', '_median_ms')] = statistics.median(values)
            summary[key.replace('_us', '_p95_ms')] = values[max(0, (95 * len(values) + 99) // 100 - 1)]
        summary['sum_max_join_wait_over_sum_round_time'] = sum(r['max_join_wait_us'] for r in rows) / sum(r['total_us'] for r in rows)
        summaries[concurrency] = summary
    return {'trace': str(path), 'trace_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'ignored_single_lane_rounds': ignored_single_lane, 'by_active_requests': summaries,
        'rounds': groups}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = {'scope': __doc__, 'traces': [summarize(p) for p in args.trace],
        'limitations': ['Instrumented host timing includes scheduling, synchronization and logging.',
            'Only completed rounds with both lanes active are included; request counts may decline during a batch.',
            'Join wait ends before commits and next draft generation; those barriers are additional.',
            'A finished lane waiting does not imply idle GPU/Sparks or predict an equivalent throughput gain.',
            'Medians include cold shape encounters; p95 can include first-use costs.']}
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
