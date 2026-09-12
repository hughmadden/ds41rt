#!/usr/bin/env python3
"""Controlled warm C1 comparison; sequential arms share the Spark workers.

Ordinary counting/code decode and whole-request latency, not release headlines.
Each workload primes both processes, then alternates AB/BA sample order.
"""
import argparse
import json
from pathlib import Path
import runpy
import statistics

API = runpy.run_path(str(Path(__file__).with_name('qualify-ds41-native-api.py')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-url', required=True)
    parser.add_argument('--candidate-url', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--runs', type=int, default=4)
    parser.add_argument('--tokens', type=int, default=256)
    parser.add_argument('--workload', choices=['counting', 'code'], action='append',
                        help='Select a workload; omitted runs both.')
    args = parser.parse_args()
    assert args.runs >= 2 and args.tokens >= 32
    record = dict(scope=__doc__, urls=dict(reference=args.reference_url, candidate=args.candidate_url),
                  workloads=[], summaries=[])
    prompts = dict(counting='Count from 1 to 200, separated by commas. Output only the sequence.',
        code='Write a complete Python 3 implementation of a thread-safe bounded LRU cache using OrderedDict and RLock. Include get, put, delete, clear, length, and unittest cases. Output only code.')
    for name, prompt in prompts.items():
        if args.workload and name not in args.workload:
            continue
        body = API['payload'](prompt, True)
        body['max_tokens'] = args.tokens
        workload = dict(name=name, request=body, warmups={}, runs=[])
        record['workloads'].append(workload)
        for arm, url in record['urls'].items():
            workload['warmups'][arm] = API['stream_case'](url, body)
        for sample in range(args.runs):
            pair = dict(sample=sample)
            for arm in (['reference', 'candidate'] if sample % 2 == 0 else ['candidate', 'reference']):
                pair[arm] = API['stream_case'](record['urls'][arm], body)
            pair['text_equal'] = pair['reference']['text'] == pair['candidate']['text']
            pair['token_counts_equal'] = all(pair['reference']['usage'][k] == pair['candidate']['usage'][k]
                                            for k in ['prompt_tokens', 'completion_tokens', 'total_tokens'])
            workload['runs'].append(pair)
            args.output.write_text(json.dumps(record, indent=2) + '\n')
            print(json.dumps(dict(workload=name, sample=sample, text_equal=pair['text_equal'],
                reference_tps=pair['reference']['observed_decode_tokens_per_second'],
                candidate_tps=pair['candidate']['observed_decode_tokens_per_second'])), flush=True)
        summary = dict(workload=name, arms={})
        for arm in record['urls']:
            summary['arms'][arm] = {key: statistics.median(p[arm][key] for p in workload['runs'])
                for key in ['observed_decode_tokens_per_second', 'first_content_seconds', 'finish_seconds']}
        summary['candidate_decode_ratio'] = summary['arms']['candidate']['observed_decode_tokens_per_second'] / summary['arms']['reference']['observed_decode_tokens_per_second']
        record['summaries'].append(summary)
        args.output.write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
