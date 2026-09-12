#!/usr/bin/env python3
"""Qualify C2 failure isolation with four FP4 source page groups and no retention.

Launch with --concurrency 2 --prefix-cache-entries 0 --kv-pool-size 1822720.
Two ~900-token prompts fit, but both 512-token completions cannot fit together.
A decoder capacity failure must retire one owner and let the other finish.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import runpy
import threading
import time
import urllib.error


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url', required=True)
    p.add_argument('--legacy-round-failure', action='store_true',
                   help='Record the preserved baseline, which fails both active requests on decode pressure')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('output already exists')
    api = runpy.run_path(str(Path(__file__).with_name('qualify-ds41-native-api.py')))
    report = dict(scope=__doc__, base_url=a.base_url, legacy_round_failure=a.legacy_round_failure,
                  cases=[], passed=False)

    def save():
        a.output.write_text(json.dumps(report, indent=2) + '\n')

    def run(body, barrier, first, peer, cancel):
        barrier.wait(timeout=30)
        result = dict(request=body, events=[], cancelled=False, done=False)
        started = time.monotonic()
        try:
            with api['open_request'](a.base_url, body) as response:
                result['status'] = response.status
                if not body['stream']:
                    result['response'] = json.load(response)
                    result['success'] = bool(result['response'].get('choices'))
                    return result
                content = ''
                for line in response:
                    if not line.startswith(b'data: '):
                        continue
                    data = line[6:].strip()
                    if data == b'[DONE]':
                        result['done'] = True
                        break
                    event = json.loads(data)
                    result['events'].append(event)
                    if event.get('error'):
                        result['error'] = event['error']
                    for choice in event.get('choices', []):
                        part = choice.get('delta', {}).get('content') or ''
                        if part:
                            content += part
                            first.set()
                            if cancel:
                                assert peer.wait(30), 'peer never entered decode before cancellation'
                                result['cancelled'] = True
                                result['text'] = content
                                result['success'] = False
                                return result
                        if choice.get('finish_reason'):
                            result['finish_reason'] = choice['finish_reason']
                result['text'] = content
                result['success'] = result['done'] and bool(result.get('finish_reason')) and 'error' not in result
                if 'error' in result:
                    assert not result['done'] and not result.get('finish_reason'), 'failed SSE emitted a success terminator'
                return result
        except urllib.error.HTTPError as error:
            result.update(status=error.code, error=json.loads(error.read()), success=False)
            return result
        finally:
            result['elapsed_seconds'] = time.monotonic() - started

    for name, filler, stream, cancel in [
        ('decode-json', 900, False, False),
        ('decode-sse', 900, True, False),
        ('admission-json', 1100, False, False),
        ('admission-sse', 1100, True, False),
        ('cancel-sse', 900, True, True),
    ]:
        case = dict(name=name, filler_tokens=filler, runs=[])
        report['cases'].append(case)
        save()
        barrier = threading.Barrier(2)
        first = [threading.Event(), threading.Event()]
        bodies = [dict(api['payload'](
            f'Request {i}. ' + ' amber' * filler
            + '\nIgnore the filler. Count from 1 to 1000, separated by commas. Output only the numbers.', stream),
            max_tokens=512) for i in range(2)]
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run, bodies[i], barrier, first[i], first[1-i], cancel and i == 0)
                       for i in range(2)]
            for future in futures:
                case['runs'].append(future.result())
                save()
        successful = [r for r in case['runs'] if r['success']]
        expected = 0 if a.legacy_round_failure and name.startswith('decode-') else 1
        assert len(successful) == expected, (name, 'unexpected survivors', len(successful), expected)
        for r in successful:
            if stream:
                assert r['finish_reason'] == 'length'
                usage = next(e['usage'] for e in reversed(r['events']) if e.get('usage'))
            else:
                assert r['response']['choices'][0]['finish_reason'] == 'length'
                usage = r['response']['usage']
            assert usage['completion_tokens'] == 512
        if cancel:
            assert case['runs'][0]['cancelled'] and case['runs'][1]['success']
        else:
            for r in case['runs']:
                if not r['success']:
                    if stream and 'error' not in r:
                        # A late SSE backend error terminates the HTTP body.
                        # Require a truncated stream without any success marker;
                        # JSON preserves the concrete pool cause.
                        assert r['events'] and not r['done'] and not r.get('finish_reason')
                    else:
                        assert 'pool exhausted' in json.dumps(r.get('error', {})), \
                            'failure was not pool pressure'
        # Recovery is a separate request after both prior owners have terminated.
        recovery = api['payload']('What is 2 + 2? Answer with just the number.')
        with api['open_request'](a.base_url, recovery) as response:
            case['recovery'] = json.load(response)
        assert case['recovery']['choices'][0]['message']['content'].strip() == '4'
        case['passed'] = True
        save()
        print('PASS', name, 'survivors', len(successful), flush=True)
    report['passed'] = True
    save()


if __name__ == '__main__':
    main()
