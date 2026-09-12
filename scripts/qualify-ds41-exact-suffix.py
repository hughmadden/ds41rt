#!/usr/bin/env python3
"""Compare retained-turn large-suffix prefill with the previous cached API.

Each measured request follows its own completed parent. Unique session tags
avoid complete-prompt hits; AB/BA alternation limits shared-worker drift.
This focused short-context matrix is not the final release prefill matrix.
"""
import argparse
import json
from pathlib import Path
import runpy
import statistics
import uuid

API = runpy.run_path(str(Path(__file__).with_name('qualify-ds41-native-api.py')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-url', required=True)
    parser.add_argument('--candidate-url', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--suffix-rows', default='16,96,192,384,768,1536')
    parser.add_argument('--warm-each-shape', action='store_true',
                        help='Prime each suffix shape before its paired measurements.')
    args = parser.parse_args()
    record = dict(scope=__doc__, arguments=vars(args) | {'output': str(args.output)}, cases=[])

    def save():
        args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')

    def run(rows, iteration, warmup=False):
        tag = uuid.uuid4().hex
        prompt = f'Session {tag}. Remember the access code quartz731.\n'
        prompt += ''.join(f'Inventory row {i}: item orchid, quantity {i % 11}.\n' for i in range(160))
        prompt += 'Reply with only OK.'
        suffix = ''.join(f'Additional row {i}: verified and unchanged.\n' for i in range(rows))
        suffix += 'What is the access code? Reply with only the code.'
        case = dict(suffix_rows=rows, iteration=iteration, warmup=warmup, arms={})
        record['cases'].append(case)
        order = ['reference', 'candidate'] if iteration % 2 == 0 else ['candidate', 'reference']
        case['order'] = order
        for arm in order:
            url = getattr(args, arm + '_url')
            parent_body = API['payload'](prompt, True)
            parent_body['max_tokens'] = 8
            parent = API['stream_case'](url, parent_body)
            result = dict(parent_request=parent_body, parent=parent)
            case['arms'][arm] = result
            save()
            assert parent['text'].strip() == 'OK', (arm, parent['text'])
            body = dict(parent_body, max_tokens=32, messages=parent_body['messages'] + [
                dict(role='assistant', content=parent['text']), dict(role='user', content=suffix)])
            continued = API['stream_case'](url, body)
            result.update(request=body, continued=continued)
            save()
            assert continued['text'].strip() == 'quartz731', (arm, continued['text'])
            usage = continued['usage']
            assert usage['prompt_cache_hit_tokens'] >= parent['usage']['prompt_tokens']
            assert usage['prompt_cache_miss_tokens'] >= 128
        assert case['arms']['reference']['continued']['usage'] == case['arms']['candidate']['continued']['usage']
        print(json.dumps(dict(suffix_rows=rows, iteration=iteration, warmup=warmup,
            ttft={arm: result['continued']['first_content_seconds'] for arm, result in case['arms'].items()})), flush=True)

    run(96, 0, warmup=True)
    for rows in map(int, args.suffix_rows.split(',')):
        if args.warm_each_shape:
            run(rows, 0, warmup=True)
        for iteration in range(args.repeats):
            run(rows, iteration)
    record['summaries'] = []
    for rows in map(int, args.suffix_rows.split(',')):
        cases = [c for c in record['cases'] if c['suffix_rows'] == rows and not c['warmup']]
        medians = {arm: statistics.median(c['arms'][arm]['continued']['first_content_seconds'] for c in cases)
                   for arm in ['reference', 'candidate']}
        record['summaries'].append(dict(suffix_rows=rows, ttft_seconds=medians,
            speedup=medians['reference'] / medians['candidate'],
            prompt_tokens=[c['arms']['candidate']['continued']['usage']['prompt_tokens'] for c in cases],
            new_tokens=[c['arms']['candidate']['continued']['usage']['prompt_cache_miss_tokens'] for c in cases]))
    record['passed'] = True
    save()
    print('PASS retained-turn suffix answers, cache accounting and paired prefill measurements', flush=True)


if __name__ == '__main__':
    main()
