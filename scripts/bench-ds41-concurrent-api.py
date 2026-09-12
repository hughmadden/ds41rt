#!/usr/bin/env python3
"""Concurrent warm counting decode through C16, including admission gaps."""
import argparse,concurrent.futures,json,runpy,time,statistics,uuid
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--base-url', default='http://127.0.0.1:18042')
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--concurrency', type=int, nargs='+', default=[1,2,4,8,16])
parser.add_argument('--repeats', type=int, default=3)
parser.add_argument('--label', default='release-concurrency')
args=parser.parse_args()
if args.repeats < 1 or any(c < 1 or c > 16 for c in args.concurrency):
 parser.error('repeats must be positive and concurrency must be 1..16')
if args.output.exists() or len(set(args.concurrency)) != len(args.concurrency):
 parser.error('output must be new and concurrency values unique')
api=runpy.run_path(str(Path(__file__).with_name('qualify-ds41-native-api.py')));records=[]
prompt=f'{args.label} {uuid.uuid4().hex}. Count from 1 to 200, separated by commas. Output only the sequence.'
def run(i):
 b=api['payload'](prompt,True);b['max_tokens']=640
 start=time.perf_counter();r=api['stream_case'](args.base_url,b)
 r.pop('events',None)
 return dict(start=start,result=r)
warmup=run(0)['result'];reference=[warmup['text'],{k:warmup['usage'][k] for k in ['prompt_tokens','completion_tokens','total_tokens']}]
assert [x.strip() for x in warmup['text'].split(',')]==[str(x) for x in range(1,201)],'counting warmup was incorrect'
for c in args.concurrency:
 for repeat in range(args.repeats):
  with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool:rows=list(pool.map(run,range(c)))
  for row in rows:
   usage=row['result']['usage'];pair=[row['result']['text'],{k:usage[k] for k in ['prompt_tokens','completion_tokens','total_tokens']}]
   assert pair==reference,(c,repeat,'counting output changed')
   assert usage['prompt_cache_hit_tokens']==usage['prompt_tokens'],(c,repeat,'prompt was not warm')
  # Inclusive span from earliest content to final finish, including admission gaps.
  begin=min(r['start']+r['result']['first_content_seconds'] for r in rows)
  end=max(r['start']+r['result']['finish_seconds'] for r in rows)
  agg=sum(r['result']['usage']['completion_tokens']-1 for r in rows)/(end-begin)
  record=dict(concurrency=c,repeat=repeat+1,aggregate_tps=agg,per_stream_mean=statistics.mean(r['result']['observed_decode_tokens_per_second'] for r in rows),rows=rows)
  records.append(record);args.output.write_text(json.dumps(records,indent=2))
  print(c,repeat+1,round(agg,2),round(record['per_stream_mean'],2),flush=True)
summaries=[]
for c in args.concurrency:
 values=[row['aggregate_tps'] for row in records if row['concurrency']==c]
 summaries.append(dict(concurrency=c,samples=len(values),median_aggregate_tps=statistics.median(values),min_aggregate_tps=min(values),max_aggregate_tps=max(values)))
args.output.write_text(json.dumps(dict(scope=__doc__,base_url=args.base_url,label=args.label,prompt=prompt,warmup=warmup,concurrency=args.concurrency,repeats=args.repeats,records=records,summaries=summaries,passed=True),indent=2)+'\n')
