#!/usr/bin/env python3
"""Live C2/C6/C16 counting, cancellation and replacement regression checks."""
import argparse,concurrent.futures,json,runpy,time
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--base-url', default='http://127.0.0.1:18042')
parser.add_argument('--output', type=Path, required=True)
args=parser.parse_args()
api=runpy.run_path(str(Path(__file__).with_name('qualify-ds41-native-api.py')))
base=args.base_url
records=[]
def body(i):
 b=api['payload'](f'Count from {i%4*200+1} to {(i%4+1)*200}, separated by commas. Output only the sequence.',True)
 b['max_tokens']=[32,64,128,256][i%4]
 return b
def run(i,cancel=False):
 start=time.perf_counter()
 r=api['stream_case'](base,body(i),cancel)
 return dict(index=i,start=start,end=time.perf_counter(),cancel=cancel,result=r)
expected={}
def same_result(actual, reference):
 # Cache reuse may change between cold, warm and concurrent admissions. Token
 # counts and output stay invariant; validate cache accounting independently.
 a,e=actual['usage'],reference['usage']
 hit=a.get('prompt_cache_hit_tokens',0)
 assert 0 <= hit <= a['prompt_tokens']
 assert a.get('prompt_cache_miss_tokens',a['prompt_tokens']-hit)+hit==a['prompt_tokens']
 assert a.get('prompt_tokens_details',{}).get('cached_tokens',hit)==hit
 return actual['text']==reference['text'] and all(a[k]==e[k] for k in ['prompt_tokens','completion_tokens','total_tokens'])
for i in range(4):expected[i]=run(i)['result']
for count in [2,6,16]:
 with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
  results=list(pool.map(run,range(count)))
 for r in results:
  e=expected[r['index']%4];o=r['result']
  assert same_result(o,e),(count,r['index'],'output differs')
 records.append(dict(concurrency=count,results=results))
 args.output.write_text(json.dumps(records,indent=2))
 print('PASS',count,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
 pending={pool.submit(run,i,i in [0,3,8]):i for i in range(16)}
 results=[];replacement=16
 while pending:
  done,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
  for f in done:
   pending.pop(f);results.append(f.result())
   if replacement<20:
    pending[pool.submit(run,replacement)]=replacement;replacement+=1
for r in results:
 if not r['cancel']:
  e=expected[r['index']%4];o=r['result'];assert same_result(o,e),('reuse',r['index'])
records.append(dict(concurrency=16,cancellation_and_replacement=True,results=results))
args.output.write_text(json.dumps(records,indent=2))
print('PASS cancellation and replacement',flush=True)
