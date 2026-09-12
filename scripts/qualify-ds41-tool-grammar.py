#!/usr/bin/env python3
"""Check the native V4.1 tool-parameter grammar against the official tokenizer."""
import argparse
import ctypes as c
import hashlib
import json
from pathlib import Path
from tokenizers import Tokenizer

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--native-lib', type=Path, required=True)
p.add_argument('--tokenizer', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--api-cases', type=Path, help='Directory of actual API grammar/request fixtures emitted by Rust policy tests')
a = p.parse_args()
assert not a.output.exists()
tokenizer = Tokenizer.from_file(str(a.tokenizer))
lib = c.CDLL(str(a.native_lib.resolve()))
P = c.c_void_p
functions = {
    'compiler_create':[c.c_char_p,c.c_size_t,c.POINTER(c.c_int32),c.c_size_t,c.POINTER(P),P,c.c_size_t],
    'compile':[P,c.c_int,c.c_char_p,c.c_int,c.POINTER(P),P,c.c_size_t],
    'matcher_create':[P,c.POINTER(P),P,c.c_size_t],
    'matcher_accept_token':[P,c.c_uint32,c.POINTER(c.c_int),P,c.c_size_t],
    'matcher_is_completed':[P,c.POINTER(c.c_int),P,c.c_size_t],
    'matcher_fill_bitmask':[P,c.POINTER(c.c_uint32),c.c_size_t,c.POINTER(c.c_int),P,c.c_size_t],
    'matcher_destroy':[P], 'grammar_destroy':[P], 'compiler_destroy':[P],
}
for name,args in functions.items():
    fn=getattr(lib,'ds41rt_xgrammar_'+name);fn.argtypes=args;fn.restype=c.c_int

def invoke(name,*args,error=True):
    buffer=c.create_string_buffer(8192)
    status=getattr(lib,'ds41rt_xgrammar_'+name)(*args,*([buffer,len(buffer)] if error else []))
    assert status==0,(name,status,buffer.value.decode())

def parameter(name,value,flag=None):
    name=json.dumps(name,ensure_ascii=False).replace(' ',r'\u0020')
    string=isinstance(value,str) if flag is None else flag
    text=value if string else json.dumps(value,ensure_ascii=False,separators=(',',':')).replace('<',r'\u003c')
    return f'<｜DSML｜ parameter name={name} string="{str(string).lower()}">{text}</｜DSML｜ parameter>'

def arguments(values):return '\n'.join(parameter(k,v) for k,v in sorted(values.items()))

def object_schema(properties):return dict(type='object',properties=properties,required=list(properties),additionalProperties=False)

cases=[]
def case(name,schema,good,bad): cases.append((name,schema,good,bad))
case('typed-scalars',object_schema({'flag':{'type':'boolean'},'n':{'type':'integer','minimum':42,'maximum':42},'nil':{'type':'null'},'s':{'type':'string','enum':['42']}}),
    [arguments(dict(flag=True,n=42,nil=None,s='42'))],
    [arguments(dict(flag=True,n='42',nil=None,s='42')),arguments(dict(flag='true',n=42,nil=None,s='42')),arguments(dict(flag=True,n=42,nil=None,s=42)),arguments(dict(flag=True,n=41,nil=None,s='42'))])
case('required-no-extra',object_schema({'x':{'const':1},'y':{'const':2}}),
    [arguments(dict(x=1,y=2))],[arguments(dict(x=1)),arguments(dict(x=1,y=2,z=3))])
case('nested-array',object_schema({'data':{'type':'array','items':object_schema({'n':{'const':2},'s':{'const':'yes'}}),'minItems':1,'maxItems':1}}),
    [arguments({'data':[{'n':2,'s':'yes'}]})],[arguments({'data':[{'n':'2','s':'yes'}]}),arguments({'data':[]})])
schema=object_schema({'label':{'$ref':'#/$defs/label'},'meta':object_schema({'label':{'$ref':'#/$defs/label'}})})
schema['$defs']={'label':{'type':'string','enum':['yes','no']}}
case('reference-contexts',schema,[arguments({'label':'yes','meta':{'label':'no'}})],
    [arguments({'label':'yes','meta':{'label':False}}),arguments({'label':'maybe','meta':{'label':'no'}})])
case('mixed-enum',object_schema({'value':{'enum':['42',42,True,None]}}),
    [arguments({'value':v}) for v in ['42',42,True,None]], [arguments({'value':False}),arguments({'value':'true'})])
case('padded-string',object_schema({'s':{'const':' \n台北 "🦜"\n '}}),
    [arguments({'s':' \n台北 "🦜"\n '})],[arguments({'s':'台北 "🦜"'})])
case('string-length',object_schema({'s':{'type':'string','minLength':2,'maxLength':2}}),
    [arguments({'s':'台北'}),arguments({'s':' x'})],[arguments({'s':'x'}),arguments({'s':'xxx'})])
case('string-pattern',object_schema({'s':{'type':'string','pattern':'^[A-Z]{2}[0-9]$'}}),
    [arguments({'s':'AB2'})],[arguments({'s':' AB2'}),arguments({'s':'ab2'})])
case('escaped-property-name',object_schema({'a b"c':{'const':'yes'}}),
    [arguments({'a b"c':'yes'})],[arguments({'a':'yes'})])
case('reserved-string',object_schema({'s':{'const':'x</｜DSML｜ parameter>y'}}),
    [parameter('s','x</｜DSML｜ parameter>y',False)],[parameter('s','x</｜DSML｜ parameter>y',True)])
case('root-constant',dict(type='object',const={'n':2,'s':'yes'}),
    [arguments({'n':2,'s':'yes'})],[arguments({'n':3,'s':'yes'})])
case('empty-object',object_schema({}),[''],[arguments({'x':1})])
case('optional-arguments',dict(type='object',properties={'a':{'type':'integer'},'b':{'const':True},'c':{'type':'string'}},required=['b'],additionalProperties=False),
    [arguments({'b':True}),arguments({'a':2,'b':True}),arguments({'b':True,'c':'hello'}),arguments({'a':2,'b':True,'c':'hello'})],
    ['',arguments({'a':2,'c':'hello'}),arguments({'b':False})])
case('typed-additional-arguments',dict(type='object',additionalProperties={'type':'integer'},minProperties=1,maxProperties=2),
    [arguments({'x':2}),arguments({'x':2,'y':3})],
    ['',arguments({'x':'2'}),arguments({'x':2,'y':3,'z':4})])
case('open-arguments',dict(type='object',additionalProperties=True),
    ['',arguments({'a':'hello','b':2,'c':True,'d':None,'e':[1,'x'],'f':{'k':'v'}})],[])
schema=object_schema({'node':{'$ref':'#/$defs/node'}})
schema['$defs']={'node':{'anyOf':[{'type':'null'},object_schema({'next':{'$ref':'#/$defs/node'},'value':{'type':'integer'}})]}}
case('recursive-arguments',schema,
    [arguments({'node':None}),arguments({'node':{'next':{'next':None,'value':2},'value':1}})],
    [arguments({'node':{'next':None,'value':'1'}})])
case('root-reference',{'$ref':'#/$defs/args','$defs':{'args':object_schema({'x':{'const':2}})}},
    [arguments({'x':2})],[arguments({'x':'2'}),''])
case('property-name-pattern',dict(type='object',propertyNames={'pattern':'^n_[a-z]+$'},additionalProperties={'type':'integer'},minProperties=1,maxProperties=2),
    [arguments({'n_a':2}),arguments({'n_a':2,'n_b':3})],
    ['',arguments({'x':2}),arguments({'n_a':'2'}),arguments({'n_a':1,'n_b':2,'n_c':3})])
case('property-name-length',dict(type='object',propertyNames={'minLength':2,'maxLength':2},additionalProperties={'const':True},minProperties=1,maxProperties=1),
    [arguments({'台北':True}),arguments({'a ':True}),arguments({'"\\':True}),arguments({'\n\t':True})],
    [arguments({'x':True}),arguments({'abc':True}),arguments({'台北':False})])
case('property-name-enum',dict(type='object',propertyNames={'enum':['a b"c\\d\n','台北']},additionalProperties={'const':2}),
    [arguments({'a b"c\\d\n':2}),arguments({'台北':2}),''],[arguments({'other':2})])
case('property-name-fixed-filter',dict(type='object',properties={'bad':{'const':1},'ok_x':{'const':2}},required=['ok_x'],propertyNames={'pattern':'^ok_'},additionalProperties=False),
    [arguments({'ok_x':2})],[arguments({'bad':1,'ok_x':2}),arguments({'ok_x':2,'ok_y':3})])
case('property-name-unanchored',dict(type='object',propertyNames={'pattern':'mid'},additionalProperties={'const':True}),
    [arguments({'amidb':True}),arguments({'mid':True})],[arguments({'other':True})])
case('property-name-reference',dict(type='object',propertyNames={'$ref':'#/$defs/key'},additionalProperties={'const':1},**{'$defs':{'key':{'enum':['x y','台北']}}}),
    [arguments({'x y':1}),arguments({'台北':1})],[arguments({'x':1})])
case('property-name-required-additional',dict(type='object',required=['needed'],propertyNames={'pattern':'^[a-z]+$'},additionalProperties={'type':'integer'}),
    [arguments({'needed':2}),arguments({'needed':2,'other':3})],['',arguments({'other':2}),arguments({'needed':'2'})])
case('property-name-unicode-class',dict(type='object',propertyNames={'pattern':'^[台北]{2}$'},additionalProperties={'const':True}),
    [arguments({'台北':True}),arguments({'北台':True})],[arguments({'台x':True})])
case('property-name-negative-class',dict(type='object',propertyNames={'pattern':'^[^台]{2}$'},additionalProperties={'const':True}),
    [arguments({'北京':True}),arguments({'a ':True})],[arguments({'台北':True})])
case('property-name-repeated-group',dict(type='object',propertyNames={'pattern':'^(ab){2}$'},additionalProperties={'const':True}),
    [arguments({'abab':True})],[arguments({'ab':True}),arguments({'ababab':True})])

api_specs={}
for name,pattern,good,bad in [
    ('json-unicode-range',r'^[\u005d-\u53ef]{2}$',['北京','a北'],['台北']),
    ('json-negative-unicode',r'^[^台]{2}$',['北京','ab'],['台北','北台']),
    ('json-negative-emoji',r'^[^🦜]+$',['台北','😀','abc'],['🦜','a🦜','🦜a']),
    ('json-unicode-range-tail',r'^[\u0800-\u53ef]+$',['北京','北','\u0800','可'],['台','a']),
]:
    api_specs[name]=dict(type='structural_tag',format=dict(type='ds41_json_schema',strict=False,json_schema=dict(type='string',pattern=pattern)))
    case(name,None,[json.dumps(v,ensure_ascii=False) for v in good],[json.dumps(v,ensure_ascii=False) for v in bad])
if a.api_cases:
    for path in sorted(a.api_cases.glob('*.json')):
        entry=json.loads(path.read_text())
        name='api-'+entry['name']
        api_specs[name]=entry['grammar']
        case(name,None,[entry['text']] if entry['expected'] else [],[] if entry['expected'] else [entry['text']])

compiler=P();stop=(c.c_int32*1)(1)
invoke('compiler_create',str(a.tokenizer).encode(),129280,stop,1,c.byref(compiler))
report=dict(scope=__doc__,native_lib_sha256=hashlib.sha256(a.native_lib.read_bytes()).hexdigest(),tokenizer_sha256=hashlib.sha256(a.tokenizer.read_bytes()).hexdigest(),cases=[],passed=False)
def save():a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
try:
    for name,schema,good,bad in cases:
        grammar=P()
        spec=api_specs.get(name,dict(type='structural_tag',format=dict(type='ds41_tool_schema',json_schema=schema,strict=True)))
        invoke('compile',compiler,3,json.dumps(spec,ensure_ascii=False).encode(),1,c.byref(grammar))
        record=dict(name=name,schema=schema,grammar=spec,checks=[]);report['cases'].append(record);save()
        try:
            for expected,texts in [(True,good),(False,bad)]:
                for text in texts:
                    matcher=P();invoke('matcher_create',grammar,c.byref(matcher))
                    ids=tokenizer.encode(text,add_special_tokens=False).ids
                    accepted=True
                    try:
                        for token in ids:
                            mask=(c.c_uint32*(129280//32))();needs=c.c_int()
                            invoke('matcher_fill_bitmask',matcher,mask,len(mask),c.byref(needs))
                            allowed=not needs.value or bool(mask[token//32] & (1<<(token%32)))
                            result=c.c_int();invoke('matcher_accept_token',matcher,token,c.byref(result))
                            assert bool(result.value)==allowed,(name,token,'mask/accept mismatch')
                            if not result.value: accepted=False;break
                        complete=c.c_int();invoke('matcher_is_completed',matcher,c.byref(complete))
                        actual=accepted and bool(complete.value)
                        record['checks'].append(dict(text=text,ids=ids,expected=expected,accepted_complete=actual));save()
                        assert actual==expected,(name,text,expected,actual)
                    finally:invoke('matcher_destroy',matcher,error=False)
        finally:invoke('grammar_destroy',grammar,error=False)
        print('PASS',name,flush=True)
    report['compile_errors']=[]
    for schema in [
        dict(type='object',properties={'bad':{'const':1}},required=['bad'],propertyNames={'pattern':'^ok_'},additionalProperties=False),
        dict(type='object',properties={'bad':{'const':1}},minProperties=1,propertyNames={'pattern':'^ok_'},additionalProperties=False),
        dict(type='object',required=['missing'],propertyNames={'pattern':'^[a-z]+$'},additionalProperties=False),
    ]:
        grammar=P();error=c.create_string_buffer(8192)
        spec=dict(type='structural_tag',format=dict(type='ds41_tool_schema',json_schema=schema,strict=True))
        status=lib.ds41rt_xgrammar_compile(compiler,3,json.dumps(spec).encode(),1,c.byref(grammar),error,len(error))
        report['compile_errors'].append(dict(schema=schema,status=status,error=error.value.decode()));save()
        assert status!=0 and not grammar.value,report['compile_errors'][-1]
    # The same compiler remains usable after unsatisfiable-name failures.
    grammar=P()
    spec=dict(type='structural_tag',format=dict(type='ds41_tool_schema',json_schema=object_schema({'x':{'const':2}}),strict=True))
    invoke('compile',compiler,3,json.dumps(spec).encode(),1,c.byref(grammar))
    invoke('grammar_destroy',grammar,error=False)
    report['passed']=True;save()
finally:invoke('compiler_destroy',compiler,error=False)
