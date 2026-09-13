#!/usr/bin/env python3
"""Verify lane issue/commit order and evidence of independent next-round issue."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def verify(path):
    inflight = [None, None]
    last = [0, 0]
    committed = [None, None]
    issues = commits = advances = 0
    lane_cohort_entries = [0, 0]
    examples = []
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        line = re.sub(r'\x1b\[[0-9;]*m', '', raw)
        issue = 'independent verifier issued' in line
        commit = 'independent verifier committed' in line
        if not issue and not commit:
            continue
        fields = dict(re.findall(r'(\w+)=(\d+)', line))
        lane, round_id = int(fields['lane']), int(fields['round_id'])
        assert lane in (0, 1) and round_id > 0
        event = {'line': number, 'text': line, 'round_id': round_id}
        if issue:
            assert inflight[lane] is None, 'lane reused before previous commit'
            if round_id == 1:
                last[lane] = 0
                committed[lane] = None
                lane_cohort_entries[lane] += 1
            assert round_id == last[lane] + 1, 'nonconsecutive lane rounds'
            assert 1 <= int(fields['requests']) <= 8
            peer = inflight[1-lane]
            if peer and committed[lane] and committed[lane]['line'] > peer['line']:
                advances += 1
                if len(examples) < 12:
                    examples.append({'peer_issue': peer, 'own_commit': committed[lane], 'own_next_issue': event})
            inflight[lane] = event
            issues += 1
        else:
            assert inflight[lane] and inflight[lane]['round_id'] == round_id, 'commit without matching issue'
            inflight[lane] = None
            last[lane] = round_id
            committed[lane] = event
            commits += 1
    assert issues == commits and issues > 0 and not any(inflight), 'trace ends with live or missing work'
    assert advances > 0, 'no demonstrated next issue while peer verification remains live'
    return {'scope': __doc__, 'trace': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'issues': issues, 'commits': commits, 'lane_cohort_entries': lane_cohort_entries,
        'next_issues_before_peer_commit': advances, 'examples': examples,
        'limitations': 'Host verifier issue/commit order; does not prove concurrent GPU execution or a throughput gain.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(verify(args.trace), indent=2) + '\n')


if __name__ == '__main__':
    main()
