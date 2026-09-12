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

compiler=P();stop=(c.c_int32*1)(1)
invoke('compiler_create',str(a.tokenizer).encode(),129280,stop,1,c.byref(compiler))
report=dict(scope=__doc__,native_lib_sha256=hashlib.sha256(a.native_lib.read_bytes()).hexdigest(),tokenizer_sha256=hashlib.sha256(a.tokenizer.read_bytes()).hexdigest(),cases=[],passed=False)
def save():a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
try:
    for name,schema,good,bad in cases:
        grammar=P()
        spec=dict(type='structural_tag',format=dict(type='ds41_tool_schema',json_schema=schema,strict=True))
        invoke('compile',compiler,3,json.dumps(spec,ensure_ascii=False).encode(),1,c.byref(grammar))
        record=dict(name=name,schema=schema,checks=[]);report['cases'].append(record);save()
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
    report['passed']=True;save()
finally:invoke('compiler_destroy',compiler,error=False)
