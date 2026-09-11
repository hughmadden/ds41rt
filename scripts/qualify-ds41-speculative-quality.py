#!/usr/bin/env python3
"""Paired live greedy quality checks; not a standardized model benchmark."""
import argparse
import json
from pathlib import Path
import runpy
import time

API = runpy.run_path(str(Path(__file__).with_name('qualify-ds41-native-api.py')))


def cases():
    records = [f'Record {i:03d}: label=item{i}; value={i * 17 + 3}.' for i in range(100)]
    records[47] = 'Record 047: label=ZEBRA; value=quartz-731.'
    return [
        dict(name='arithmetic', prompt='Compute (37 * 19) - 128. Output only the integer.', expected='575'),
        dict(name='signed_sort', prompt='Sort [12, -4, 7, 7, 0, -11] in ascending order. Preserve duplicates. Output only a JSON array.', expected_json=[-11, -4, 0, 7, 7, 12]),
        dict(name='extraction', prompt='Alice has 17 tickets and Bob has 23 tickets. Return only JSON with keys "total" (the total number of tickets) and "names" (the two names in order).', expected_json=dict(total=40, names=['Alice', 'Bob'])),
        dict(name='logic', prompt='A is before B. C is after B. D is before A. Output the resulting order as a JSON array of the four letters, with no other text.', expected_json=['D', 'A', 'B', 'C']),
        dict(name='unicode', prompt='請計算 18 加 27，再減去 9。只輸出阿拉伯數字答案。', expected='36'),
        dict(name='translation', prompt='Translate into Traditional Chinese: "The red door is open." Output only the translation.'),
        dict(name='explanation', prompt='Explain why a metal spoon feels colder than a wooden spoon in the same room. Use exactly two short sentences.'),
        dict(name='long_retrieval', prompt='Read these records and return only the value for label ZEBRA.\n' + '\n'.join(records), expected='quartz-731'),
    ]


def check(case, text):
    if 'expected' in case:
        return text.strip() == case['expected']
    if 'expected_json' in case:
        try:
            return json.loads(text) == case['expected_json']
        except ValueError:
            return False
    return None  # Open-ended outputs require inspection, not a fabricated score.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-url', default='http://127.0.0.1:18041')
    parser.add_argument('--speculative-url', default='http://127.0.0.1:18042')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    record = dict(scope='Eight paired single-client greedy cases. Six objective checks and two open-ended comparisons; not broad model quality, stochastic equivalence or release performance.', cases=[])
    for case in cases():
        pair = dict(case=case, runs={})
        # Sequential calls avoid contention on the shared Spark expert workers.
        for mode, base in [('target', args.target_url), ('speculative', args.speculative_url)]:
            body = API['payload'](case['prompt'], stream=True)
            start = time.perf_counter()
            try:
                result = API['stream_case'](base, body)
                result['objective_pass'] = check(case, result['text'])
                pair['runs'][mode] = result
                print(json.dumps(dict(case=case['name'], mode=mode, seconds=time.perf_counter()-start, objective_pass=result['objective_pass'], text=result['text']), ensure_ascii=False), flush=True)
            except Exception as error:
                pair['runs'][mode] = dict(error=str(error))
                print(json.dumps(dict(case=case['name'], mode=mode, error=str(error))), flush=True)
        runs = pair['runs']
        pair['text_equal'] = all('text' in runs[m] for m in runs) and runs['target']['text'] == runs['speculative']['text']
        pair['usage_equal'] = all('usage' in runs[m] for m in runs) and runs['target']['usage'] == runs['speculative']['usage']
        record['cases'].append(pair)
        args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    good = all(c['text_equal'] and c['usage_equal'] and all('error' not in r and r['objective_pass'] is not False for r in c['runs'].values()) for c in record['cases'])
    print(json.dumps(dict(all_pair_checks_pass=good, cases=len(record['cases']))), flush=True)
    raise SystemExit(0 if good else 1)


if __name__ == '__main__':
    main()
