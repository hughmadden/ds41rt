#!/usr/bin/env python3
"""C16 divergent partial-prefix admission with shared retained global pages."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import runpy
import time
import uuid

API = runpy.run_path(str(Path(__file__).with_name('qualify-ds41-native-api.py')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prefix = f'Parallel session {uuid.uuid4().hex}. Inventory data is background; follow the final request.\n'
    prefix += ''.join(f'Inventory row {i}: item orchid, quantity {i % 7}.\n' for i in range(160))
    parent = API['payload'](prefix + '\nReply with only OK.', True)
    parent['max_tokens'] = 16
    record = dict(scope=__doc__, base_url=args.base_url, parent_request=parent,
                  parent=API['stream_case'](args.base_url, parent), results=[])
    assert record['parent']['text'].strip() == 'OK'

    def run(i):
        body = API['payload'](prefix + f'\nRequest {i}. Count from 1 to 20, separated by commas. Output only the sequence.', True)
        started = time.perf_counter()
        try:
            result = API['stream_case'](args.base_url, body)
            return dict(index=i, request=body, start=started, end=time.perf_counter(), result=result)
        except Exception as error:
            return dict(index=i, request=body, start=started, end=time.perf_counter(), error=repr(error))

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        for future in concurrent.futures.as_completed([pool.submit(run, i) for i in range(16)]):
            record['results'].append(future.result())
            args.output.write_text(json.dumps(record, indent=2) + '\n')
    for entry in record['results']:
        assert 'error' not in entry, entry
        result = entry['result']
        assert [s.strip() for s in result['text'].strip().split(',')] == [str(i) for i in range(1, 21)], entry['index']
        usage = result['usage']
        hit, miss = usage['prompt_cache_hit_tokens'], usage['prompt_cache_miss_tokens']
        assert hit > 128 and hit % 2 == 0 and miss >= 128, (entry['index'], usage)
        assert hit + miss == usage['prompt_tokens']
        assert usage['prompt_tokens_details']['cached_tokens'] == hit
    print('PASS C16 unique divergent suffixes: bounded replay, counting and cache accounting', flush=True)


if __name__ == '__main__':
    main()
