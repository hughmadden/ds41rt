#!/usr/bin/env python3
"""Focused C4 high-thinking tool/response grammar isolation under independent lanes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import runpy
import threading

API = runpy.run_path(str(Path(__file__).with_name('qualify-ds41-native-api.py')))


def run(base, request):
    if not request['stream']:
        with API['open_request'](base, request) as response:
            raw = json.load(response)
        choice = raw['choices'][0]
        return dict(raw=raw, message=choice['message'], finish=choice['finish_reason'], usage=raw['usage'])
    raw = dict(events=[], done=False, finish_reason=None, usage=None, text='')
    with API['open_request'](base, request) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            data = line[6:].strip()
            if data == b'[DONE]':
                raw['done'] = True
                break
            event = json.loads(data)
            assert not event.get('error'), event
            raw['events'].append(dict(event=event))
            if event.get('usage'):
                raw['usage'] = event['usage']
            for choice in event.get('choices', []):
                raw['text'] += choice.get('delta', {}).get('content') or ''
                if choice.get('finish_reason'):
                    raw['finish_reason'] = choice['finish_reason']
    assert raw['done'] and raw['finish_reason'] and raw['usage'], raw
    message = dict(content=raw['text'], reasoning_content='', tool_calls=[])
    calls = {}
    for event in raw['events']:
        for choice in event['event'].get('choices', []):
            delta = choice.get('delta', {})
            message['reasoning_content'] += delta.get('reasoning_content') or ''
            for piece in delta.get('tool_calls') or []:
                call = calls.setdefault(piece['index'], dict(function=dict(name='', arguments='')))
                for key in ['name', 'arguments']:
                    call['function'][key] += piece.get('function', {}).get(key) or ''
    message['tool_calls'] = [calls[i] for i in sorted(calls)]
    return dict(raw=raw, message=message, finish=raw['finish_reason'], usage=raw['usage'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'refusing to overwrite evidence'
    cases = []
    for i in range(4):
        expected = dict(slot=i, marker=f'lane-{i}-台北')
        schema = dict(type='object', properties={k: {'const': v} for k, v in expected.items()},
            required=list(expected), additionalProperties=False)
        tool = i < 2
        prompt = ('Call lookup exactly once with the required slot and marker from its schema.' if tool
            else 'Return exactly this JSON object: ' + json.dumps(expected, ensure_ascii=False)
                + '. The response schema enforces this same object.')
        request = dict(model=API['MODEL'], messages=[dict(role='user', content=prompt)],
            thinking=dict(type='enabled'), reasoning_effort='high', temperature=0,
            max_tokens=2048, stream=bool(i % 2))
        if request['stream']:
            request['stream_options'] = dict(include_usage=True)
        if tool:
            request.update(tools=[dict(type='function', function=dict(name='lookup', description='Record the requested object.',
                parameters=schema, strict=True))], tool_choice='required', parallel_tool_calls=False)
        else:
            request['response_format'] = dict(type='json_schema', json_schema=dict(name='lane', strict=True, schema=schema))
        cases.append(dict(request=request, schema=schema, expected=expected, tool=tool))
    report = dict(scope=__doc__, base_url=args.base_url, cases=cases, passed=False)
    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    save()
    barrier = threading.Barrier(4)
    def worker(case):
        barrier.wait()
        return run(args.base_url, case['request'])
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(worker, cases))
    for case, result in zip(cases, results):
        case['result'] = result
    save()
    for case in cases:
        result = case['result']; message = result['message']
        failure = dict(expected=case['expected'], finish=result['finish'],
            content=message.get('content'), tool_calls=message.get('tool_calls'),
            reasoning_tail=message.get('reasoning_content', '')[-500:])
        assert message.get('reasoning_content'), failure
        if case['tool']:
            assert result['finish'] == 'tool_calls', failure
            calls = message['tool_calls']; assert len(calls) == 1, failure
            assert calls[0]['function']['name'] == 'lookup', failure
            value = json.loads(calls[0]['function']['arguments'])
        else:
            assert result['finish'] == 'stop', failure
            assert not message.get('tool_calls'), failure
            value = json.loads(message['content'])
        assert isinstance(value, dict) and set(value) == {'slot', 'marker'}, failure
        assert type(value['slot']) is int and type(value['marker']) is str, failure
        assert value == case['expected'], failure
        case['passed'] = True
    report['passed'] = True; save()
    print('PASS four concurrent high-thinking tool/response constraints, JSON and SSE', flush=True)


if __name__ == '__main__':
    main()
