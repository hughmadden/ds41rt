#!/usr/bin/env python3
"""Summarize matched private-verifier prefix trials; never report probe HTTP TPS."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert not args.output.exists()
    states, aggregate = [], defaultdict(list)
    for path in sorted(args.directory.glob('snapshot-*.json')):
        d = json.loads(path.read_text())
        assert len(d['samples']) == 108 and len(d['cache_sha256']) == 2
        baseline = d['samples'][0]['best']
        groups = defaultdict(list)
        for sample in d['samples']:
            if sample['lengths'] == [5, 5]:
                assert sample['best'] == baseline, 'full-prefix sentinel differs'
            if not sample['warmup'] and not sample['sentinel']:
                groups[tuple(sample['lengths'])].append(sample)
        assert len(groups) == 25 and all(len(v) == 3 for v in groups.values())
        shapes = []
        for lengths, samples in sorted(groups.items()):
            assert all(x['best'] == samples[0]['best'] for x in samples), 'repeat argmax differs'
            agreement = sum(a == b for sample in samples for lane in range(2)
                            for a, b in zip(sample['best'][lane], baseline[lane]))
            compared = sum(len(best) for sample in samples for best in sample['best'])
            unique = [sum(lane[0] for lane in sample['expert_work']) for sample in samples]
            assert all(x == unique[0] for x in unique), 'repeat expert union differs'
            predicted = 19864 + 803*(sum(lengths)+2) + 636*unique[0] + 3168
            timings = [sample['verify_us'] for sample in samples]
            aggregate[lengths].extend(timings)
            shapes.append(dict(lengths=lengths, verify_us=timings, median_verify_us=statistics.median(timings),
                actual_mean_unique_experts=unique[0], model_us_using_actual_experts=predicted,
                common_argmax_matches=agreement, common_argmax_compared=compared))
        states.append(dict(snapshot=path.name, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            request_ids=d['ids'], committed_ends=d['ends'], cache_sha256=d['cache_sha256'], shapes=shapes))
    assert states, 'no snapshots'
    diagonals = []
    for length in [1, 3, 5]:
        actual, predicted = [], []
        for state in states:
            shape = next(x for x in state['shapes'] if x['lengths'] == (length, length))
            full = next(x for x in state['shapes'] if x['lengths'] == (5, 5))
            actual.append(1-shape['median_verify_us']/full['median_verify_us'])
            predicted.append(1-shape['model_us_using_actual_experts']/full['model_us_using_actual_experts'])
        diagonals.append(dict(length=length, median_measured_saving_fraction=statistics.median(actual),
                              median_predicted_saving_fraction=statistics.median(predicted)))
    report = dict(scope=__doc__, snapshots=len(states), measured_trials=sum(len(v) for v in aggregate.values()),
        total_executions=sum(108 for _ in states),
        common_argmax_matches=sum(x['common_argmax_matches'] for s in states for x in s['shapes']),
        common_argmax_compared=sum(x['common_argmax_compared'] for s in states for x in s['shapes']),
        median_ms_matrix=[[statistics.median(aggregate[a,b])/1000 for b in range(1,6)] for a in range(1,6)],
        diagonal_savings=diagonals, states=states,
        limitations=['Two requests, one per lane, six nearby committed contexts from one code/fable pair.',
            'Three measured trials per shape, rotated/reversed sweep order after one warmup sweep.',
            'Times include full target verification, route capture and logit collection; exclude hashing, discard, draft generation and publication.',
            'Model comparison uses actual post-verification expert unions, not the online accepted-history forecast.',
            'Hashed committed windows, paged KV/index rows, page tables and published lengths; hidden compressor carry is not independently hashed.',
            'Full-prefix sentinels and common argmax are checked, not byte equality of all logits or a general quality claim.'])
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['states','limitations']}, indent=2))

if __name__ == '__main__':
    main()
