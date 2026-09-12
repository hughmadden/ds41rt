#!/usr/bin/env python3
"""Focused native bounded-prefix replay, branch isolation and cache accounting.

The replay policy is approximate. Objective answers and token accounting are
checked here; these cases do not establish full-forward numerical identity.
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
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--retained-entries', type=int, choices=[2,16], default=16,
                        help='Expected cache capacity; two entries evict the older parent tail.')
    args = parser.parse_args()
    tag = uuid.uuid4().hex
    header = f'Session {tag}. The access code is quartz731. The following inventory records do not change the access code.\n'
    records = ''.join(f'Inventory row {i}: shelf {i % 7}, item orchid, quantity {i % 11}.\n' for i in range(240))
    tail = ''.join(f'Additional row {i}: verified and unchanged.\n' for i in range(256))
    ask = '\nWhat is the access code? Reply with only the code.'
    cases = [('parent', header + records + '\nReply with only OK.', 'OK', 'cold'),
             ('divergent_suffix', header + records + ask, 'quartz731', 'partial'),
             ('divergent_exact_repeat', header + records + ask, 'quartz731', 'full'),
             ('shorter_branch', header + ''.join(records.splitlines(keepends=True)[:120]) + ask, 'quartz731', 'partial'),
             ('parent_still_intact', header + records + '\nReply with only OK.', 'OK', 'full'),
             ('large_uncached_suffix', header + records + tail + ask, 'quartz731', 'partial')]
    if args.retained_entries == 2:
        cases = [(name,prompt,answer,'partial' if name == 'parent_still_intact' else kind)
                 for name,prompt,answer,kind in cases]
    record = dict(scope=__doc__, retained_entries=args.retained_entries, candidate=args.base_url, reference=args.reference_url, cases=[])
    for name, prompt, answer, kind in cases:
        body = API['payload'](prompt, True)
        body['max_tokens'] = 32
        reference = API['stream_case'](args.reference_url, body)
        candidate = API['stream_case'](args.base_url, body)
        case = dict(name=name, expected_reuse=kind, expected_answer=answer, request=body,
                    reference=reference, candidate=candidate)
        record['cases'].append(case)
        args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
        assert reference['text'].strip() == candidate['text'].strip() == answer, name
        usage = candidate['usage']
        hit, miss, total = usage['prompt_cache_hit_tokens'], usage['prompt_cache_miss_tokens'], usage['prompt_tokens']
        assert hit + miss == total and usage['prompt_tokens_details']['cached_tokens'] == hit
        assert total == reference['usage']['prompt_tokens']
        if kind == 'cold':
            assert hit == 0, (name, usage)
        elif kind == 'partial':
            assert 0 < hit < total and hit % 2 == 0 and miss >= 128, (name, usage)
        else:
            assert hit == total, (name, usage)
        print(json.dumps(dict(name=name, prompt=total, cached=hit, uncached=miss,
            reference_ttft=reference['first_content_seconds'], candidate_ttft=candidate['first_content_seconds'])), flush=True)
    print('PASS bounded replay, shorter/divergent branches, retained parent and multi-chunk suffix', flush=True)


if __name__ == '__main__':
    main()
