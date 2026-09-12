#!/usr/bin/env python3
"""Separate native UTF-8/streaming correctness from model copying/arithmetic."""
import argparse
import json
from pathlib import Path
import runpy

API = runpy.run_path(str(Path(__file__).with_name('qualify-ds41-native-api.py')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'preserve existing evidence'
    cases = [
        ('literal-replacement', '�', 8, '�'),
        ('parrot-one-token', '🦜', 1, '�'),
        ('parrot-two-tokens', '🦜', 2, '�'),
        ('parrot-complete', '🦜', 8, '🦜'),
        ('trailing-replacement', 'A�', 8, 'A�'),
        ('traditional-chinese', '台北，世界！', 32, None),
        ('decomposed-accent', 'e\u0301', 16, None),
        ('emoji-zwj', '👩🏽\u200d💻', 24, None),
        ('mixed-scripts', 'café Ελληνικά العربية हिन्दी ไทย', 64, None),
    ]
    report = dict(base_url=args.base_url, cases=[], passed=False)

    def save():
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')

    for name, source, limit, required in cases:
        prompt = 'Output exactly this single character with no explanation or quotes: ' + source
        if len(source) > 1:
            prompt = 'Copy the following text exactly, with no explanation or quotes: ' + source
        # Keep this prompt identical across all parrot cutoffs and the original
        # live reproducer. Token limit changes must not alter the model input.
        body = API['payload'](prompt)
        body['max_tokens'] = limit
        with API['open_request'](args.base_url, body) as response:
            buffered = json.load(response)
        streamed = API['stream_case'](args.base_url, dict(body, stream=True,
                                      stream_options=dict(include_usage=True)))
        text = buffered['choices'][0]['message']['content']
        record = dict(name=name, request=body, source=source, buffered=buffered,
                      streamed=streamed, exact_echo=text == source)
        report['cases'].append(record)
        save()
        assert text == streamed['text'], (name, text, streamed['text'])
        assert all(buffered['usage'][key] == streamed['usage'][key]
                   for key in ['prompt_tokens', 'completion_tokens', 'total_tokens'])
        if required is not None:
            assert text == required, (name, text, required)
        print(json.dumps(dict(name=name, text=text, exact_echo=record['exact_echo']),
                         ensure_ascii=False), flush=True)

    # This was previously labeled a Unicode failure. Preserve exact model
    # answers separately; extra arithmetic working is not an encoding defect.
    arithmetic = []
    for thinking in [False, True]:
        body = API['payload']('請計算 18 加 27，再減去 9。只輸出阿拉伯數字答案。')
        body['max_tokens'] = 256
        if thinking:
            body.pop('thinking')
        with API['open_request'](args.base_url, body) as response:
            result = json.load(response)
        arithmetic.append(dict(default_thinking=thinking, response=result,
                               strict_answer=result['choices'][0]['message']['content'].strip() == '36'))
    report['arithmetic_instruction_following'] = arithmetic
    report['passed'] = True
    save()


if __name__ == '__main__':
    main()
