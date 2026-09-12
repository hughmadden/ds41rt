#!/usr/bin/env python3
"""Native complete-prompt and retained-turn reuse checks against an uncached API.

This is a focused qualification, not the release performance matrix.
"""
import argparse
import json
from pathlib import Path
import runpy
import uuid

API = runpy.run_path(str(Path(__file__).with_name('qualify-ds41-native-api.py')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--reference-url', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    record = {'scope': __doc__, 'candidate': args.base_url, 'reference': args.reference_url, 'cases': []}

    def run(name, body, base):
        result = API['stream_case'](base, body)
        record['cases'].append(dict(name=name, base=base, request=body, result=result))
        args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
        usage = result['usage']
        hit = usage['prompt_cache_hit_tokens']
        assert 0 <= hit <= usage['prompt_tokens']
        assert usage['prompt_cache_miss_tokens'] + hit == usage['prompt_tokens']
        assert usage['prompt_tokens_details']['cached_tokens'] == hit
        print(json.dumps(dict(name=name, base=base, cached=hit, prompt=usage['prompt_tokens'],
                              ttft=result['first_content_seconds'], text=result['text']), ensure_ascii=False), flush=True)
        return result

    def same(a, b):
        assert a['text'] == b['text'], (a['text'], b['text'])
        for key in ['prompt_tokens', 'completion_tokens', 'total_tokens']:
            assert a['usage'][key] == b['usage'][key], key

    tag = uuid.uuid4().hex
    body = API['payload'](f'Session {tag}. Count from 1 to 20, separated by commas. Output only the numbers.', True)
    baseline = run('count_reference', body, args.reference_url)
    cold = run('count_cold', body, args.base_url)
    warm = run('count_complete_hit', body, args.base_url)
    same(cold, baseline)
    same(warm, baseline)
    assert warm['usage']['prompt_cache_hit_tokens'] == warm['usage']['prompt_tokens']

    # Rendered conversation includes the prior assistant output. The cached
    # prompt/turn frontier must be restored before processing the new question.
    body = API['payload'](f'Session {tag}. Remember that the access code is quartz731. Reply only OK.', True)
    initial = run('turn_initial', body, args.base_url)
    followup = dict(body, messages=body['messages'] + [dict(role='assistant', content=initial['text']),
        dict(role='user', content='What is the access code? Output only the code.')])
    baseline = run('turn_reference', followup, args.reference_url)
    continued = run('turn_retained_prefix', followup, args.base_url)
    same(continued, baseline)
    assert continued['usage']['prompt_cache_hit_tokens'] >= initial['usage']['prompt_tokens']
    assert continued['usage']['prompt_cache_hit_tokens'] < continued['usage']['prompt_tokens']
    assert continued['text'].strip() == 'quartz731'
    again = run('turn_complete_hit', followup, args.base_url)
    same(again, continued)
    assert again['usage']['prompt_cache_hit_tokens'] == again['usage']['prompt_tokens']
    record['passed'] = True
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
