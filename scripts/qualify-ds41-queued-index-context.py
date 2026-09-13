#!/usr/bin/env python3
"""Check overlapping decode and retained index candidates with a shared 16K prompt."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import runpy
import threading
import time
from tokenizers import Tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output must be new')
    api = runpy.run_path(str(Path(__file__).with_name('qualify-ds41-native-api.py')))
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    ids = tokenizer.encode('alpha beta gamma delta epsilon zeta eta theta ' * 4000).ids[:16384]
    assert len(ids) == 16384
    prompt = ('Reference text:\n<reference>\n' + tokenizer.decode(ids) +
              '\n</reference>\nIgnore the reference text for this task. '
              'Count from 1 to 200, separated by commas. Output only the sequence.')
    body = api['payload'](prompt, True)
    body['max_tokens'] = 640
    started = threading.Event()
    def request(callback=None):
        begin = time.perf_counter()
        result = api['stream_case'](args.base_url, body, on_first_content=callback)
        result.pop('events', None)
        return dict(start=begin, result=result)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(request, started.set)
        assert started.wait(timeout=180), 'first request did not start decoding'
        second = pool.submit(request)
        rows = [first.result(), second.result()]
    exact = all([int(n) for n in re.findall(r'\d+', row['result']['text'])] == list(range(1, 201)) for row in rows)
    overlap = max(row['start'] + row['result']['first_content_seconds'] for row in rows) < min(
        row['start'] + row['result']['finish_seconds'] for row in rows)
    usage = rows[1]['result']['usage']
    cached = usage.get('prompt_cache_hit_tokens', usage.get('prompt_tokens_details', {}).get('cached_tokens', 0))
    report = dict(scope=__doc__, request=body, rows=rows, exact_sequences=exact,
                  decode_overlapped=overlap, second_cached_tokens=cached,
                  passed=exact and overlap and cached >= 16384)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    assert report['passed'], {k:v for k,v in report.items() if k not in ('rows','request')}
    print('PASS overlapping 16K-context counting, prompt reuse, retained index candidates')


if __name__ == '__main__':
    main()
