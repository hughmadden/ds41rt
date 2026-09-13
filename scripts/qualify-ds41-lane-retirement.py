#!/usr/bin/env python3
"""Verify a short request retires without restarting its longer peer's lane."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import runpy
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trace-log', type=Path, help='Verify saved lane_schedule debug logs after the request phase')
    args = parser.parse_args()
    if args.trace_log:
        report = json.loads(args.output.read_text())
        assert report['responses_passed']
        lines = re.sub(r'\x1b\[[0-9;]*m', '', args.trace_log.read_text()).splitlines()
        def field(line, name):
            match = re.search(r'\b' + name + r'=(\d+)\b', line)
            return int(match[1]) if match else None
        retired = [i for i, line in enumerate(lines) if 'independent lane request retired' in line
                   and field(line, 'request_id') == 2]
        assert len(retired) == 1, 'fresh server must retire short request 2 exactly once'
        boundary = retired[0]
        assert field(lines[boundary], 'lane') == 1
        issued = [(i, field(line, 'round_id')) for i, line in enumerate(lines)
                  if 'independent verifier issued' in line and field(line, 'lane') == 0]
        before = [(i, r) for i, r in issued if i < boundary]
        after = [(i, r) for i, r in issued if i > boundary]
        assert before and after, 'peer must issue work before and after retirement'
        assert after[0][1] == before[-1][1] + 1, 'peer round sequence restarted at retirement'
        committed = [field(line, 'round_id') for line in lines[:boundary]
                     if 'independent verifier committed' in line and field(line, 'lane') == 0]
        report['trace'] = dict(path=str(args.trace_log), short_request_id=2, retired_lane=1,
            peer_lane=0, peer_round_before=before[-1][1], peer_round_after=after[0][1],
            peer_round_uncommitted_at_retirement=not committed or committed[-1] < before[-1][1])
        report['passed'] = True
    else:
        assert not args.output.exists(), 'refusing to overwrite request evidence'
        api = runpy.run_path(str(Path(__file__).with_name('qualify-ds41-native-api.py')))
        started = threading.Event()
        def request(count, callback=None):
            body = api['payload'](f'Count from 1 to {count}, separated by commas. Output only the sequence.', True)
            body['max_tokens'] = 640 if count == 200 else 80
            begin = time.perf_counter()
            result = api['stream_case'](args.base_url, body, on_first_content=callback)
            result.pop('events', None)
            assert [int(n) for n in re.findall(r'\d+', result['text'])] == list(range(1, count+1))
            return dict(request=body, start=begin, result=result)
        with ThreadPoolExecutor(max_workers=2) as pool:
            long = pool.submit(request, 200, started.set)
            assert started.wait(timeout=180), 'long request did not begin'
            short = pool.submit(request, 20)
            short, long = short.result(), long.result()
        assert short['start'] + short['result']['finish_seconds'] < long['start'] + long['result']['finish_seconds']
        report = dict(scope=__doc__, long=long, short=short, responses_passed=True, passed=False,
                      requires='fresh server with RUST_LOG=info,ds41rt::lane_schedule=debug; verify saved trace next')
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print('PASS retirement trace' if args.trace_log else 'PASS overlapping counting responses; trace verification pending')


if __name__ == '__main__':
    main()
