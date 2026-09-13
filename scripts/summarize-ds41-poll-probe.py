#!/usr/bin/env python3
"""Compare polling intervals within matched private-verifier contexts and prefix shapes."""
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
    states, aggregate, paired = [], defaultdict(list), defaultdict(list)
    matches = compared = 0
    for path in sorted(args.directory.glob('snapshot-*.json')):
        data = json.loads(path.read_text())
        assert len(data['samples']) == 68 and len(data['cache_sha256']) == 2
        reference = data['samples'][0]['best']
        groups = defaultdict(list)
        for sample in data['samples']:
            if sample['lengths'] == [5,5]:
                assert sample['best'] == reference, 'full-prefix sentinel differs'
            if sample['sentinel'] or sample['warmup']:
                continue
            groups[tuple(sample['lengths']), sample['poll_us']].append(sample)
            for lane, best in enumerate(sample['best']):
                matches += sum(x == y for x,y in zip(best, reference[lane]))
                compared += len(best)
        assert len(groups) == 15 and all(len(v) == 3 for v in groups.values())
        rows = []
        for (lengths, interval), samples in sorted(groups.items()):
            assert all(x['best'] == samples[0]['best'] and x['expert_work'] == samples[0]['expert_work'] for x in samples)
            control = groups[lengths, 250]
            assert samples[0]['best'] == control[0]['best'] and samples[0]['expert_work'] == control[0]['expert_work']
            change = statistics.median(x['verify_us'] for x in samples)/statistics.median(x['verify_us'] for x in control)-1
            if interval != 250:
                paired[interval].append(change)
            aggregate[lengths, interval].extend(samples)
            rows.append(dict(lengths=lengths, interval_us=interval,
                verify_us=[x['verify_us'] for x in samples], poll_counts=[x['poll_counts'] for x in samples],
                change_vs_250=change, expert_work=samples[0]['expert_work']))
        states.append(dict(file=path.name, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            committed_ends=data['ends'], cache_sha256=data['cache_sha256'], rows=rows))
    assert states and matches == compared
    summary = []
    for (lengths, interval), samples in sorted(aggregate.items()):
        summary.append(dict(lengths=lengths, interval_us=interval,
            median_verify_ms=statistics.median(x['verify_us'] for x in samples)/1000,
            median_polls=statistics.median(x['poll_counts'][0] for x in samples),
            median_yields=statistics.median(x['poll_counts'][1] for x in samples)))
    report = dict(scope=__doc__, snapshots=len(states), measured_trials=sum(len(v) for v in aggregate.values()),
        total_executions=len(states)*68, common_argmax_matches=matches, common_argmax_compared=compared,
        summary=summary, paired_changes={interval:dict(pairs=len(changes), median=statistics.median(changes),
            minimum=min(changes), maximum=max(changes)) for interval,changes in paired.items()}, states=states,
        limitations=['Two requests, one per lane; six nearby contexts from one code/fable pair.',
            'Three measured sweeps after warmup; interval/shape order rotates and reverses.',
            'Complete target verification and logit collection, with route capture and poll counters enabled in every arm.',
            'No draft, cache hashing, discard, or publication time; probe HTTP TPS is deliberately excluded.',
            'Cache hashes cover committed windows/paged KV/index/publication metadata, not hidden compressor carry.'])
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['states','limitations']},indent=2))

if __name__ == '__main__':
    main()
