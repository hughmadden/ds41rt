#!/usr/bin/env python3
"""Focused fixed-input mixed traffic and cache/lifecycle probe, not release qualification."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import runpy
import threading
import time
from tokenizers import Tokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url', required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--nonce-seed', type=int, default=56001)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--concurrency', type=int, nargs='+', choices=range(1,17), default=[4,16],
                   help='Mixed batch concurrency levels, default 4 16')
    p.add_argument('--skip-lifecycle', action='store_true', help='Run only mixed traffic for focused diagnostics')
    p.add_argument('--ordered-admission', action='store_true',
                   help='Start each request after its predecessor emits content, controlling admission order while decode overlaps')
    args = p.parse_args()
    if args.output.exists():
        p.error('output must be new')
    here = Path(__file__).parent
    api = runpy.run_path(str(here / 'qualify-ds41-native-api.py'))
    bench = runpy.run_path(str(here / 'bench-ds41-release-decode.py'))
    corpus_path = here / 'fixtures/release-semantic-corpus.json'
    corpus = json.loads(corpus_path.read_text())
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    nonces = iter(bench['token_zero_nonces'](sum(args.concurrency)+20, args.nonce_seed, tokenizer))
    report = dict(scope=__doc__, nonce_seed=args.nonce_seed,
                  corpus_sha256=hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
                  tokenizer_sha256=hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
                  admission='ordered_first_content' if args.ordered_admission else 'simultaneous_unordered',
                  batches=[], lifecycle={}, passed=False)

    def save():
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')

    def request(body, cancel=False, on_first_content=None):
        start = time.perf_counter()
        result = api['stream_case'](args.base_url, body, cancel=cancel, on_first_content=on_first_content)
        result.pop('events', None)
        return dict(start=start, request=body, result=result)

    # Inputs match across arms. Simultaneous clients do not guarantee server
    # arrival order; ordered mode controls that variable without serializing decode.
    for concurrency in args.concurrency:
        bodies = []
        for i in range(concurrency):
            case = ['code', 'fable', 'topic'][i % 3]
            definition = corpus['cases'][case]
            body = api['payload'](next(nonces)['prefix'] + definition['prompt'], True)
            body['max_tokens'] = definition['max_tokens']
            bodies.append((case, body))
        barrier = threading.Barrier(concurrency)
        admitted = [threading.Event() for _ in range(concurrency+1)]
        admitted[0].set()
        failed = threading.Event()

        def work(item):
            index, (case, body) = item
            try:
                if args.ordered_admission:
                    if not admitted[index].wait(timeout=180) or failed.is_set():
                        raise RuntimeError('ordered admission predecessor failed or timed out')
                else:
                    barrier.wait(timeout=30)
                return dict(case=case, admission_index=index, **request(body,
                    on_first_content=admitted[index+1].set if args.ordered_admission else None))
            except BaseException:
                failed.set()
                for gate in admitted:
                    gate.set()
                raise

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            rows = list(pool.map(work, enumerate(bodies)))
        begin = min(r['start'] + r['result']['first_content_seconds'] for r in rows)
        end = max(r['start'] + r['result']['finish_seconds'] for r in rows)
        tokens = sum(r['result']['usage']['completion_tokens'] - 1 for r in rows)
        report['batches'].append(dict(concurrency=concurrency, rows=rows,
            aggregate_tps=tokens / (end - begin),
            span_definition='earliest first content through last finish, including admission gaps'))
        save()
        assert all(r['result']['text'].strip() for r in rows)
        print('MIXED', concurrency, round(tokens / (end - begin), 2), flush=True)

    if args.skip_lifecycle:
        report['lifecycle_skipped'] = True
        report['passed'] = True
        save()
        return

    # Exercise large->small graph transitions, prompt reuse, and a completed-turn
    # continuation. The needle is in the middle, not repeated in the question.
    filler = ''.join(f'Entry {i:05d}: storage record remains ordinary and unchanged.\n' for i in range(5000))
    ids = tokenizer.encode(filler, add_special_tokens=False).ids[:32768]
    prompt = (next(nonces)['prefix'] + tokenizer.decode(ids[:16384]) +
              '\nThe unique key SILVER_MAPLE has value BLUE-7319.\n' +
              tokenizer.decode(ids[16384:]) +
              '\nReturn only the value for SILVER_MAPLE. No other text.')
    body = api['payload'](prompt, True)
    body['max_tokens'] = 32
    cold = request(body)
    warm = request(body)
    continuation = dict(body, messages=body['messages'] + [
        dict(role='assistant', content=cold['result']['text']),
        dict(role='user', content='What is two plus two? Reply only 4.')])
    resumed = request(continuation)
    report['lifecycle'].update(needle_cold=cold, needle_warm=warm, retained_turn=resumed)
    save()
    assert cold['result']['text'].strip() == warm['result']['text'].strip() == 'BLUE-7319'
    assert warm['result']['usage']['prompt_cache_hit_tokens'] == warm['result']['usage']['prompt_tokens']
    assert resumed['result']['text'].strip() == '4'
    assert resumed['result']['usage']['prompt_cache_hit_tokens'] >= cold['result']['usage']['prompt_tokens']
    print('NEEDLE, PROMPT REUSE, RETAINED TURN PASS', flush=True)

    jobs = []
    for i in range(16):
        cancel = i % 2 == 0
        prompt = next(nonces)['prefix'] + ('Count from 1 to 1000.' if cancel else
            'Count from 1 to 20, separated by commas. Output only the numbers.')
        body = api['payload'](prompt, True)
        body['max_tokens'] = 2048 if cancel else 96
        jobs.append((body, cancel))
    barrier = threading.Barrier(16)

    def lifecycle_work(item):
        body, cancel = item
        barrier.wait(timeout=30)
        return dict(cancel=cancel, **request(body, cancel))

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        outcomes = list(pool.map(lifecycle_work, jobs))
    report['lifecycle']['cancellation_batch'] = outcomes
    save()
    for r in outcomes:
        if r['cancel']:
            assert r['result']['cancelled_after_content']
        else:
            assert [x.strip() for x in r['result']['text'].split(',')] == [str(x) for x in range(1, 21)]
    recovery = request(api['payload']('What is 2 + 2? Reply only 4.', True))
    report['lifecycle']['post_cancellation'] = recovery
    save()
    assert recovery['result']['text'].strip() == '4'
    report['passed'] = True
    save()
    print('CANCELLATION/SURVIVORS/RECOVERY PASS', flush=True)


if __name__ == '__main__':
    main()
