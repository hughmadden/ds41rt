#!/usr/bin/env python3
"""Sequential C1 prefill status workloads; effective rate includes API/TTFT overhead."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy

from tokenizers import Tokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--context-file', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--target-url', default='http://127.0.0.1:18041')
    p.add_argument('--speculative-url', default='http://127.0.0.1:18042')
    p.add_argument('--filler-tokens', type=int, nargs='+', default=[3950,16384])
    p.add_argument('--kinds', nargs='+', choices=['repeated','code'], default=['repeated','code'])
    a = p.parse_args()
    if not a.filler_tokens or min(a.filler_tokens) < 1:
        p.error('filler token counts must be positive')
    api = runpy.run_path(str(Path(__file__).with_name('qualify-ds41-native-api.py')))
    tokenizer = Tokenizer.from_file(str(a.tokenizer))
    context = a.context_file.read_text()
    if not context.strip():
        p.error('context file is empty')
    record = dict(scope='Sequential C1 generated counting after repeated filler or supplied code; no prefix-cache hits expected, not a broad quality/performance benchmark.',
                  tokenizer_sha256=hashlib.sha256(a.tokenizer.read_bytes()).hexdigest(),
                  context_sha256=hashlib.sha256(context.encode()).hexdigest(), results=[])
    for kind in a.kinds:
        text = ' amber' if kind == 'repeated' else context + '\n'
        # Tokenize repetitions together so boundary merges are included.
        while True:
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            if len(ids) >= max(a.filler_tokens):
                break
            text = text * 2
        for count in a.filler_tokens:
            prompt = tokenizer.decode(ids[:count], skip_special_tokens=False)
            prompt += '\nIgnore the filler above. Count from 1 to 20, separated by commas. Output only the numbers.'
            for mode, base in [('target',a.target_url),('speculative',a.speculative_url)]:
                body = api['payload'](prompt, stream=True)
                body['max_tokens'] = 64
                result = api['stream_case'](base, body)
                result.pop('events', None)
                result.update(mode=mode, kind=kind, filler_tokens=count,
                              prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
                result['effective_prompt_tokens_per_second'] = result['usage']['prompt_tokens']/result['first_content_seconds']
                record['results'].append(result)
                a.output.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
                print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
