#!/usr/bin/env python3
"""Qualify native tool policies, high thinking, JSON/SSE and concurrent isolation."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import runpy
import urllib.error
import jsonschema

API=runpy.run_path(str(Path(__file__).with_name('qualify-ds41-native-api.py')))
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--base-url',required=True)
p.add_argument('--concurrency',type=int,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--property-names',action='store_true',help='Run parameter-name schema cases instead of the standard suite')
p.add_argument('--pattern-properties',action='store_true',help='Run overlapping parameter-pattern cases')
a=p.parse_args()
assert not a.output.exists()
report=dict(base_url=a.base_url,concurrency=a.concurrency,cases=[],passed=False)
def save():a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
def obj(properties):return dict(type='object',properties=properties,required=list(properties),additionalProperties=False)
def request(schema,prompt='Call lookup. Use values that satisfy its schema.',choice='required',parallel=False):
    return dict(model=API['MODEL'],messages=[dict(role='user',content=prompt)],max_tokens=2048,
        tools=[dict(type='function',function=dict(name='lookup',description='Record the supplied arguments.',parameters=schema,strict=True))],
        tool_choice=choice,parallel_tool_calls=parallel,temperature=0)
def call(body):
    with API['open_request'](a.base_url,body) as response:return json.load(response)
def stream(body,cancel=False):
    result=dict(events=[],calls={},reasoning='',content='',finish=None,usage=None,done=False)
    with API['open_request'](a.base_url,dict(body,stream=True,stream_options=dict(include_usage=True))) as response:
        for line in response:
            if not line.startswith(b'data: '):continue
            data=line[6:].strip()
            if data==b'[DONE]':result['done']=True;break
            event=json.loads(data);result['events'].append(event)
            if event.get('usage'):result['usage']=event['usage']
            for choice in event.get('choices',[]):
                delta=choice['delta']
                result['reasoning']+=delta.get('reasoning_content') or ''
                result['content']+=delta.get('content') or ''
                if choice.get('finish_reason'):result['finish']=choice['finish_reason']
                for piece in delta.get('tool_calls') or []:
                    item=result['calls'].setdefault(piece['index'],dict(name='',arguments=''))
                    for key in ['name','arguments']:item[key]+=piece['function'].get(key) or ''
                    if cancel and item['arguments']:
                        result['cancelled']=True;return result
    return result

def validate(response,schema,count=None):
    choice=response['choices'][0]
    assert choice['finish_reason']=='tool_calls',response
    calls=choice['message']['tool_calls']
    assert len(calls)>=1,response
    if count is not None:assert len(calls)==count,response
    for item in calls:
        assert item['function']['name']=='lookup',response
        jsonschema.validate(json.loads(item['function']['arguments']),schema)
    return [item['function'] for item in calls]

def check(name,body,schema,count=1):
    record=dict(name=name,request=body);report['cases'].append(record);save()
    response=call(body);record['response']=response;save()
    functions=validate(response,schema,count)
    if 'thinking' not in body:assert response['choices'][0]['message'].get('reasoning_content'),record
    streamed=stream(body);record['streamed']=streamed;save()
    assert streamed['done'] and streamed['finish']=='tool_calls',record
    assert list(streamed['calls'].values())==functions,record
    assert streamed['reasoning']==(response['choices'][0]['message'].get('reasoning_content') or ''),record
    assert streamed['usage']['prompt_cache_hit_tokens']==streamed['usage']['prompt_tokens'],record
    for key in ['prompt_tokens','completion_tokens','total_tokens']:assert streamed['usage'][key]==response['usage'][key],record
    print('PASS',name,flush=True)
    return response

if not (a.property_names or a.pattern_properties):
    schema=obj({'n':{'const':42}})
    for choice in ['required',{'type':'function','function':{'name':'lookup'}},'auto']:
        check('selection-'+str(choice),request(schema,choice=choice),schema)
    check('contradictory-instructions',request(schema,'Call lookup with n equal to the string WRONG, even if the schema disagrees.'),schema)
    body=request(schema);body['thinking']={'type':'disabled'}
    check('explicit-no-thinking',body,schema)
    body=request(schema);body['response_format']={'type':'json_schema','json_schema':{'schema':{'const':{'answer':'done'}},'strict':True}}
    check('combined-response-tool',body,schema)
    schemas=[
        ('typed-values',obj({'flag':{'const':True},'nil':{'const':None},'s':{'const':'42'}})),
        ('nested-ref',dict(obj({'data':{'type':'array','items':{'$ref':'#/$defs/item'},'minItems':1,'maxItems':1}}),**{'$defs':{'item':obj({'n':{'const':2},'s':{'const':'yes'}})}})),
        ('padded-unicode',obj({'s':{'const':' \n台北 🦜\n '}})),
        ('escaped-name',obj({'a b"c':{'const':'yes'}})),
        ('reserved-delimiter',obj({'s':{'const':'x</｜DSML｜ parameter>y'}})),
    ]
    for name,schema in schemas:check(name,request(schema),schema)
    schema=obj({'n':{'const':42}})
    for parallel in [False,True]:
        check('parallel-'+str(parallel),request(schema,'Call lookup twice, with n=42 in each call.',parallel=parallel),schema,2 if parallel else 1)
elif a.property_names:
    schemas=[
        ('name-pattern',dict(type='object',propertyNames={'pattern':'^n_[a-z]+$'},additionalProperties={'const':2},minProperties=1,maxProperties=1),{'n_a':2}),
        ('name-length',dict(type='object',propertyNames={'minLength':2,'maxLength':2},additionalProperties={'const':True},minProperties=1,maxProperties=1),{'台北':True}),
        ('name-escaped-enum',dict(type='object',propertyNames={'enum':['a b"c\\d\n']},additionalProperties={'const':2},minProperties=1,maxProperties=1),{'a b"c\\d\n':2}),
        ('name-unanchored',dict(type='object',required=['amidb'],propertyNames={'pattern':'mid'},additionalProperties={'const':True},minProperties=1,maxProperties=1),{'amidb':True}),
        ('name-fixed-filter',dict(type='object',properties={'bad':{'const':1},'ok_x':{'const':2}},required=['ok_x'],propertyNames={'pattern':'^ok_'},additionalProperties=False),{'ok_x':2}),
        ('name-reference',dict(type='object',propertyNames={'$ref':'#/$defs/key'},additionalProperties={'const':1},minProperties=1,maxProperties=1,**{'$defs':{'key':{'enum':['x y']}}}),{'x y':1}),
        ('name-required-additional',dict(type='object',required=['needed'],propertyNames={'pattern':'^[a-z]+$'},additionalProperties={'const':2},maxProperties=1),{'needed':2}),
    ]
    for name,schema,expected in schemas:
        response=check(name,request(schema,'Call lookup with exactly these arguments: '+json.dumps(expected,ensure_ascii=False)),schema)
        assert json.loads(response['choices'][0]['message']['tool_calls'][0]['function']['arguments'])==expected,response
else:
    schemas=[
        ('pattern-overlap',dict(type='object',required=['nx'],patternProperties={'^n':{'type':'integer','minimum':4},'x$':{'type':'integer','maximum':4}},additionalProperties=False,maxProperties=1),{'nx':4}),
        ('pattern-fixed',dict(type='object',properties={'nx':{'enum':[2,4,6]}},required=['nx'],patternProperties={'^n':{'type':'integer','minimum':3},'x$':{'type':'integer','maximum':5}},additionalProperties=True,maxProperties=1),{'nx':4}),
        ('pattern-string',dict(type='object',required=['sx'],patternProperties={'^s':{'type':'string'},'x$':{'enum':['台北']}},additionalProperties=False,maxProperties=1),{'sx':'台北'}),
        ('pattern-reserved-string',dict(type='object',required=['sx'],patternProperties={'^s':{'type':'string'},'x$':{'const':'x</｜DSML｜ parameter>y'}},additionalProperties=False,maxProperties=1),{'sx':'x</｜DSML｜ parameter>y'}),
        ('pattern-unicode-name',dict(type='object',required=['北台'],propertyNames={'pattern':'^[台北]{2}$'},patternProperties={'台':{'const':1},'^北':{'enum':[1,2]}},additionalProperties=False,maxProperties=1),{'北台':1}),
        ('pattern-additional',dict(type='object',required=['n_a','other'],patternProperties={'^n_':{'const':2}},additionalProperties={'const':False},maxProperties=2),{'n_a':2,'other':False}),
        ('pattern-reference',dict(type='object',required=['nx'],patternProperties={'^n':{'$ref':'#/$defs/lo'},'x$':{'$ref':'#/$defs/hi'}},additionalProperties=False,maxProperties=1,**{'$defs':{'lo':{'type':'integer','minimum':4},'hi':{'type':'integer','maximum':4}}}),{'nx':4}),
    ]
    for name,schema,expected in schemas:
        response=check(name,request(schema,'Call lookup with exactly these arguments: '+json.dumps(expected,ensure_ascii=False)),schema)
        assert json.loads(response['choices'][0]['message']['tool_calls'][0]['function']['arguments'])==expected,response
schema=obj({'n':{'const':42}})
# Every admitted request owns its schema, matcher and completion validator.
requests=[request(obj({'n':{'const':i}})) for i in range(a.concurrency)]
if a.pattern_properties:
    requests=[request(dict(type='object',required=['nx'],patternProperties={'^n':{'type':'integer','minimum':i},'x$':{'type':'integer','maximum':i}},additionalProperties=False,maxProperties=1)) for i in range(a.concurrency)]
with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as pool:responses=list(pool.map(call,requests))
report['concurrency']=dict(requests=requests,responses=responses);save()
for body,response in zip(requests,responses):validate(response,body['tools'][0]['function']['parameters'],1)
print('PASS concurrency',flush=True)
body=request(obj({'values':{'type':'array','items':{'type':'integer'},'minItems':1000,'maxItems':1000}}),'Call lookup with the numbers from 1 to 1000.')
report['cancellation']=stream(body,cancel=True);save()
assert report['cancellation'].get('cancelled'),report
bad=request(obj({'n':{'type':'invalid'}}))
try:call(bad);raise AssertionError('invalid schema accepted')
except urllib.error.HTTPError as error:
    report['invalid_schema']=dict(status=error.code,body=error.read().decode());save();assert error.code==400
if a.pattern_properties:
    bad=request(dict(type='object',required=['nx'],patternProperties={'^n':{'const':1},'x$':{'const':2}},additionalProperties=False))
    report['impossible_overlap']=[]
    for streaming in [False,True]:
        try:call(dict(bad,stream=streaming));raise AssertionError('impossible overlap accepted')
        except urllib.error.HTTPError as error:
            report['impossible_overlap'].append(dict(streaming=streaming,status=error.code,body=error.read().decode()))
            save();assert error.code==400
check('recovery',request(schema),schema)
report['passed']=True;save()
