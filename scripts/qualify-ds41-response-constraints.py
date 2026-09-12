#!/usr/bin/env python3
"""Validate native response constraints, cache reselection, SSE and concurrent isolation."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import re
import runpy
import urllib.error
import jsonschema

API = runpy.run_path(str(Path(__file__).with_name('qualify-ds41-native-api.py')))
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--base-url', required=True)
p.add_argument('--concurrency', type=int, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
assert not a.output.exists()
r = dict(base_url=a.base_url, concurrency=a.concurrency, cases=[], passed=False)
def save(): a.output.write_text(json.dumps(r, ensure_ascii=False, indent=2) + '\n')
def body(schema, prompt='Reply with only DISALLOWED.', thinking=False, strict=True):
    result = API['payload'](prompt)
    result['max_tokens'] = 256
    result['response_format'] = dict(type='json_schema', json_schema=dict(name='qualification', strict=strict, schema=schema))
    if thinking: result.pop('thinking')
    return result
def call(request):
    with API['open_request'](a.base_url, request) as response: return json.load(response)
def check(name, request, schema=None, expected=None):
    record = dict(name=name, request=request)
    r['cases'].append(record); save()
    response = call(request); record['response'] = response; save()
    content = response['choices'][0]['message']['content']
    assert response['choices'][0]['finish_reason'] == 'stop', record
    if schema is not None: jsonschema.validate(json.loads(content), schema)
    if expected is not None: assert json.loads(content) == expected, record
    streamed = API['stream_case'](a.base_url, dict(request, stream=True, stream_options=dict(include_usage=True)))
    record['streamed'] = streamed; save()
    assert streamed['text'] == content, record
    thoughts = ''.join(choice.get('delta', {}).get('reasoning_content') or '' for event in streamed['events'] for choice in event['event'].get('choices', []))
    assert thoughts == (response['choices'][0]['message'].get('reasoning_content') or ''), record
    assert all(streamed['usage'][key] == response['usage'][key] for key in ['prompt_tokens','completion_tokens','total_tokens'])
    assert streamed['usage']['prompt_cache_hit_tokens'] == streamed['usage']['prompt_tokens']
    print('PASS', name, flush=True)
    return response

for value in ['allowed', 'changed', '台北 🦜']:
    schema = {'const': {'value': value}}
    response = check('same-prompt-' + value, body(schema), schema, {'value': value})
    if value != 'allowed': assert response['usage']['prompt_cache_hit_tokens'] == response['usage']['prompt_tokens']
first = r['cases'][0]
schema = {'const':{'value':'followup'}}
request = body(schema)
request['messages'] = first['request']['messages'] + [
    {'role':'assistant','content':first['response']['choices'][0]['message']['content']},
    {'role':'user','content':'Now reply only DISALLOWED again.'}]
response = check('retokenized-turn-new-schema', request, schema, {'value':'followup'})
assert response['usage']['prompt_cache_hit_tokens'] >= first['response']['usage']['prompt_tokens']
# A canonical single-token answer supplies a genuine exact completed frontier.
schema = {'const':42}
seed_request = body(schema, 'What is 6 times 7? Reply as a JSON number.')
seed = check('canonical-number', seed_request, schema, 42)
assert seed['usage']['completion_tokens'] == 2
schema = {'const':43}
request = body(schema)
request['messages'] = seed_request['messages'] + [
    {'role':'assistant','content':seed['choices'][0]['message']['content']},
    {'role':'user','content':'Now add one and reply as a JSON number.'}]
response = check('exact-completed-turn-new-schema', request, schema, 43)
assert response['usage']['prompt_cache_hit_tokens'] >= seed['usage']['total_tokens'] - 1
schema = {'type':'object','properties':{'x':{'const':1}},'required':['x']}
for strict in [False, True]:
    response = check('strict-' + str(strict), body(schema, 'Return JSON with x=1 and extra=2.', strict=strict), schema)
    value = json.loads(response['choices'][0]['message']['content'])
    if strict: assert value == {'x':1}
    else: assert value == {'x':1,'extra':2}
cases = [
    ('nested-ref', {'$defs':{'item':{'type':'object','properties':{'n':{'type':'integer','minimum':42,'maximum':42},'ok':{'const':True}},'required':['n','ok'],'additionalProperties':False}},'type':'array','items':{'$ref':'#/$defs/item'},'minItems':2,'maxItems':2}),
    ('nullable', {'anyOf':[{'type':'null'},{'const':'allowed'}]}),
    ('pattern-length', {'type':'string','pattern':'^[A-Z]{3}[0-9]{2}$','minLength':5,'maxLength':5}),
    ('tuple', {'type':'array','prefixItems':[{'const':True},{'const':None},{'const':'台北'}],'minItems':3,'maxItems':3,'items':False}),
]
for name, schema in cases: check(name, body(schema), schema)
schema = {'const': {'answer':42}}
response = check('default-high-thinking', body(schema, 'What is 6 times 7? Reply only DISALLOWED.', True), schema, {'answer':42})
assert response['choices'][0]['message'].get('reasoning_content')
request = API['payload']('Return a JSON object with x equal to 1.')
request['response_format'] = {'type':'json_object'}
check('json-object', request, {'type':'object'})
request = API['payload']('Reply only DISALLOWED.')
request['response_format'] = {'type':'regex', 'regex':'A[0-9]{2}'}
response = check('regex', request)
assert re.fullmatch('A[0-9]{2}', response['choices'][0]['message']['content'])
# Response schema still permits tool dispatch; strict argument enforcement is
# a separate gate. Required mode starts inside the prefilled DSML calls block.
for choice in ['auto', 'required', {'type':'function','function':{'name':'lookup'}}]:
    request = body({'const':{'answer':'done'}}, 'Call lookup with query cat. Do not answer directly.')
    request['tools'] = [{'type':'function','function':{'name':'lookup','description':'Look up the supplied query.',
        'parameters':{'type':'object','properties':{'query':{'type':'string'}},'required':['query'],'additionalProperties':False}}}]
    request['tool_choice'] = choice
    response = call(request)
    r.setdefault('combined_tool_dispatch',[]).append(dict(request=request,response=response)); save()
    assert response['choices'][0]['finish_reason'] == 'tool_calls'
    calls = response['choices'][0]['message']['tool_calls']
    assert len(calls) == 1 and calls[0]['function']['name'] == 'lookup'
    assert json.loads(calls[0]['function']['arguments']) == {'query':'cat'}
short = call(dict(body({'const':{'value':'long enough to need several tokens'}}), max_tokens=1))
r['length_limit'] = short; save(); assert short['choices'][0]['finish_reason'] == 'length'
# Every request uses exactly the same rendered prompt but a different schema.
requests = [body({'const':{'slot':i}}, 'Concurrent constraint selection: reply only DISALLOWED.') for i in range(a.concurrency)]
with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as pool: results = list(pool.map(call, requests))
r['concurrent'] = [dict(request=q,response=s) for q,s in zip(requests,results)]; save()
for i, response in enumerate(results): assert json.loads(response['choices'][0]['message']['content']) == {'slot':i}
cancel = dict(body({'type':'array','items':{'type':'integer'},'minItems':1000,'maxItems':1000}, 'Return a JSON array counting upward from 1.'),stream=True)
r['cancellation'] = API['stream_case'](a.base_url, cancel, cancel=True); save()
schema = {'const':{'recovered':True}}
check('after-cancellation', body(schema), schema, {'recovered':True})
for schema in [None, {'type':'not-a-type'}]:
    request = body(schema)
    try:
        call(request); raise AssertionError('invalid schema accepted')
    except urllib.error.HTTPError as error: assert error.code == 400
r['invalid_schema_status'] = 400
r['passed'] = True; save()
