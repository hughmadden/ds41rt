#!/usr/bin/env python3
"""Summarize complete single-request, 40-layer timing rounds; stages overlap."""
import argparse
import re,json,statistics,hashlib
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--spec',type=Path,required=True)
parser.add_argument('--target',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
assert not args.output.exists(), 'refusing to overwrite existing evidence'
events=['target attention chain','target layer preparation','target query preparation','target attention stages','target collection','target experts','target layer']
out={}
for arm in ['spec','target']:
 p=getattr(args,arm)
 rounds=[];pending={e:[] for e in events}
 for raw in p.read_text().splitlines():
  line=re.sub(r'\x1b\[[0-9;]*m','',raw);fields={k:int(v) for k,v in re.findall(r'\b(\w+)=(\d+)\b',line)}
  for e in events:
   if e+' ' in line:
    pending[e].append(fields);break
  if 'native scheduler round ' in line:
   layers=pending['target layer']; wanted=6 if arm=='spec' else 1
   if fields['requests']==1 and len(layers)==40 and [r['layer'] for r in layers]==list(range(40)) and all(r['rows']==wanted for r in layers):
    sums={e:{k:sum(r.get(k,0) for r in rs) for k in rs[0] if k.endswith('_us')} for e,rs in pending.items() if rs}
    rounds.append({'scheduler':fields,'stages':sums})
   pending={e:[] for e in events}
 assert rounds
 report={'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'rounds':len(rounds),'scheduler_median_ms':{k:statistics.median(r['scheduler'][k] for r in rounds)/1000 for k in rounds[0]['scheduler'] if k.endswith('_us')},'stage_median_ms':{e:{k:statistics.median(r['stages'][e][k] for r in rounds)/1000 for k in rounds[0]['stages'][e]} for e in rounds[0]['stages']},'raw_rounds':rounds}
 out[arm]=report;print(arm,json.dumps({k:v for k,v in report.items() if k!='raw_rounds'},indent=2))
out['limitations']=['Host wall timings include synchronization, logging and device execution.',
 'Nested stages overlap: do not sum the entire table.',
 'Only consecutive complete 40-layer C1 rounds at one target row or six verifier rows; excludes prefill and partial rounds.',
 'Three nearby code-generation requests; no long-context or concurrency coverage.']
args.output.write_text(json.dumps(out,indent=2)+'\n')
