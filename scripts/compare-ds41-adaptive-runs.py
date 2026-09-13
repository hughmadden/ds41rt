#!/usr/bin/env python3
"""Compare identical-input fixed/adaptive probes without claiming prose quality."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import statistics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arm in ['fixed', 'adaptive']:
        for kind in ['decode', 'mixed']:
            p.add_argument(f'--{arm}-{kind}', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('output must be new')
    paths = {k: v for k, v in vars(a).items() if k != 'output'}
    data = {k: json.loads(v.read_text()) for k, v in paths.items()}
    check = runpy.run_path(str(Path(__file__).with_name('release_throughput_checks.py')))['check_output']
    report = dict(scope=__doc__, source_sha256={k: hashlib.sha256(v.read_bytes()).hexdigest()
                  for k, v in paths.items()}, decode={}, mixed=[], lifecycle={})
    f, d = data['fixed_decode']['samples'], data['adaptive_decode']['samples']
    assert len(f) == len(d) and f
    pairs = []
    for x, y in zip(f, d):
        assert (x['case'], x['repeat'], x['request']) == (y['case'], y['repeat'], y['request'])
        pairs.append(dict(case=x['case'], repeat=x['repeat'], exact=x['content'] == y['content'],
                          fixed_checks=check(x['case'], x['content']),
                          adaptive_checks=check(y['case'], y['content'])))
    for case in dict.fromkeys(x['case'] for x in f):
        before = statistics.median(x['observed_decode_tokens_per_second'] for x in f if x['case'] == case)
        after = statistics.median(x['observed_decode_tokens_per_second'] for x in d if x['case'] == case)
        report['decode'][case] = dict(fixed_tps=before, adaptive_tps=after,
                                     adaptive_over_fixed=after / before)
    report['decode_pairs'] = pairs
    fb, db = data['fixed_mixed']['batches'], data['adaptive_mixed']['batches']
    assert len(fb) == len(db)
    for x, y in zip(fb, db):
        assert x['concurrency'] == y['concurrency'] and len(x['rows']) == len(y['rows'])
        pairs = []
        for i, (fx, dy) in enumerate(zip(x['rows'], y['rows'])):
            assert fx['request'] == dy['request'] and fx['case'] == dy['case']
            pairs.append(dict(index=i, case=fx['case'], exact=fx['result']['text'] == dy['result']['text'],
                fixed_checks=check(fx['case'], fx['result']['text']),
                adaptive_checks=check(dy['case'], dy['result']['text'])))
        report['mixed'].append(dict(concurrency=x['concurrency'], fixed_tps=x['aggregate_tps'],
            adaptive_tps=y['aggregate_tps'], adaptive_over_fixed=y['aggregate_tps'] / x['aggregate_tps'],
            pairs=pairs, scope='one cold mixed batch per arm, includes admission gaps'))
    for arm in ['fixed', 'adaptive']:
        source = data[arm + '_mixed']
        life = source['lifecycle']
        cold = life['needle_cold']['result']
        report['lifecycle'][arm] = dict(passed=source['passed'],
            cold_prompt_tokens=cold['usage']['prompt_tokens'],
            cold_ttft_seconds=cold['first_content_seconds'],
            cold_prompt_tokens_per_ttft_second=cold['usage']['prompt_tokens'] / cold['first_content_seconds'],
            warm_usage=life['needle_warm']['result']['usage'],
            retained_usage=life['retained_turn']['result']['usage'],
            cancellation_count=sum(x['cancel'] for x in life['cancellation_batch']),
            survivor_count=sum(not x['cancel'] for x in life['cancellation_batch']))
    report['limitations'] = ['Exploratory sequential arms, not a balanced release comparison.',
        'Decode has three samples per case; mixed traffic and long prompt have one sample per arm.',
        'Exact text equality is reported separately from named output checks; prose quality is not assessed.',
        'Cold prompt/TTFT is an HTTP proxy including overhead, not isolated prefill throughput.']
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(decode=report['decode'], mixed=[{k:v for k,v in x.items() if k!='pairs'}
                     for x in report['mixed']], lifecycle=report['lifecycle']), indent=2))


if __name__ == '__main__':
    main()
