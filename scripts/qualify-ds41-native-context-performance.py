#!/usr/bin/env python3
"""Paired C1 retained-prompt decode or cold-prefill latency; sequential arms.

Default first requests are recorded but not used to qualify prefill.
--cold-prefill uses fresh prompt prefixes for each pair and bounds cache hits.
Thinking is disabled for controlled counting throughput, not tool evaluation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import statistics
from tokenizers import Tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-url', required=True)
    parser.add_argument('--candidate-url', required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--context-file', type=Path, required=True)
    parser.add_argument('--context-tokens', type=int, nargs='+', default=[4096, 16384, 24576])
    parser.add_argument('--tokens', type=int, default=512)
    parser.add_argument('--pairs', type=int, default=4)
    parser.add_argument('--cold-prefill', action='store_true',
                        help='Use fresh prefixes per pair; measure first-content latency with at most 32 cached tokens')
    parser.add_argument('--label', required=True, help='Unique prompt namespace for this comparison')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists')
    if min(args.context_tokens) < 1 or args.tokens < 2 or args.pairs < 2:
        parser.error('positive context sizes, at least two output tokens and two pairs required')
    source = args.context_file.read_text()
    if not source.strip():
        parser.error('context file is empty')
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    text = source
    while True:
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        if len(ids) >= max(args.context_tokens):
            break
        text += text
    api = runpy.run_path(str(Path(__file__).with_name('qualify-ds41-native-api.py')))
    urls = dict(reference=args.reference_url, candidate=args.candidate_url)
    report = dict(scope=__doc__, urls=urls, label=args.label, cold_prefill=args.cold_prefill,
                  tokenizer_sha256=hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(source.encode()).hexdigest(), cases=[], passed=False)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + '\n')

    for count in args.context_tokens:
        prompt = (f'{args.label} {count}. Reference source follows.\n'
                  + tokenizer.decode(ids[:count], skip_special_tokens=False)
                  + '\nIgnore the reference source. Count from 1 to 1000, separated by commas. Output only the numbers.')
        body = dict(api['payload'](prompt, True), max_tokens=args.tokens)
        case = dict(filler_tokens=count, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                    request=body, first_requests={}, runs=[])
        report['cases'].append(case)
        save()
        for arm, url in urls.items():
            case['first_requests'][arm] = api['stream_case'](url, body)
            save()
        for index in range(args.pairs):
            pair = dict(order=['reference', 'candidate'] if index % 2 == 0 else ['candidate', 'reference'])
            if args.cold_prefill:
                nonce = hashlib.sha256(f'{args.label}:{count}:{index}'.encode()).hexdigest()
                pair['request'] = dict(body, messages=[dict(role='user', content=nonce + '\n' + prompt)])
            case['runs'].append(pair)
            for arm in pair['order']:
                pair[arm] = api['stream_case'](urls[arm], pair.get('request', body))
                save()
            ref, candidate = pair['reference'], pair['candidate']
            pair['text_equal'] = ref['text'] == candidate['text']
            pair['token_counts_equal'] = all(ref['usage'][k] == candidate['usage'][k]
                                            for k in ['prompt_tokens', 'completion_tokens'])
            pair['full_prompt_hits'] = all(pair[arm]['usage'].get('prompt_cache_hit_tokens') ==
                                           pair[arm]['usage']['prompt_tokens'] for arm in urls)
            pair['bounded_cold_hits'] = all(0 <= pair[arm]['usage'].get('prompt_cache_hit_tokens', -1) <= 32 for arm in urls)
            pair['first_content_ratio'] = candidate['first_content_seconds'] / ref['first_content_seconds']
            pair['decode_ratio'] = candidate['observed_decode_tokens_per_second'] / ref['observed_decode_tokens_per_second']
            save()
            assert pair['token_counts_equal'] and (pair['bounded_cold_hits'] if args.cold_prefill else pair['full_prompt_hits']), 'unequal work or unexpected cache reuse'
        case['median_decode_change_percent'] = 100 * (statistics.median(r['decode_ratio'] for r in case['runs']) - 1)
        case['median_first_content_change_percent'] = 100 * (statistics.median(r['first_content_ratio'] for r in case['runs']) - 1)
        save()
        print(args.label, count, {k: v for k, v in case.items() if k.startswith('median_')}, flush=True)
    # Complete measurements and reuse assertions; not a throughput acceptance threshold.
    report['passed'] = True
    save()


if __name__ == '__main__':
    main()
