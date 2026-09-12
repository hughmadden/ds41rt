#!/usr/bin/env python3
"""Concurrent counting decode measurement, including admission gaps."""
import argparse,concurrent.futures,json,runpy,time,statistics
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--base-url', default='http://127.0.0.1:18042')
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--concurrency', type=int, nargs='+', default=[1,2,6,16])
parser.add_argument('--repeats', type=int, default=2)
args=parser.parse_args()
if args.repeats < 1 or any(c < 1 or c > 16 for c in args.concurrency):
 parser.error('repeats must be positive and concurrency must be 1..16')
api=runpy.run_path(str(Path(__file__).with_name('qualify-ds41-native-api.py')));records=[];reference=None
def run(i):
 b=api['payload']('Count from 1 to 200, separated by commas. Output only the sequence.',True);b['max_tokens']=640
 start=time.perf_counter();r=api['stream_case'](args.base_url,b)
 return dict(start=start,result=r)
for c in args.concurrency:
 for repeat in range(args.repeats):
  with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool:rows=list(pool.map(run,range(c)))
  for row in rows:
   pair=[row['result']['text'],row['result']['usage']]
   if reference is None:reference=pair
   assert pair==reference,(c,repeat,'counting output changed')
  # Inclusive span from earliest content to final finish, including admission gaps.
  begin=min(r['start']+r['result']['first_content_seconds'] for r in rows)
  end=max(r['start']+r['result']['finish_seconds'] for r in rows)
  agg=sum(r['result']['usage']['completion_tokens']-1 for r in rows)/(end-begin)
  record=dict(concurrency=c,repeat=repeat,aggregate_tps=agg,per_stream_mean=statistics.mean(r['result']['observed_decode_tokens_per_second'] for r in rows),rows=rows)
  records.append(record);args.output.write_text(json.dumps(records,indent=2))
  print(c,repeat,round(agg,2),round(record['per_stream_mean'],2),flush=True)
