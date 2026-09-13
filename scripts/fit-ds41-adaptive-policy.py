#!/usr/bin/env python3
"""Fit exploratory adaptive costs from corrected native route/confidence traces."""
import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
from collections import defaultdict, deque
import numpy as np


def observations(path):
    hist={};pending=[];decisions=[];records=[];conf=[]
    for raw in path.open():
     if not any(s in raw for s in ['native route policy observation','native draft policy observation','native scheduler round']):continue
     line=re.sub(r'\x1b\[[0-9;]*m','',raw)
     f=dict(re.findall(r'(\w+)=(.*?)(?= \w+=|$)',line.strip()))
     if 'native route policy observation' in line:
      pending.append((int(f['layer']),ast.literal_eval(f['owners']),json.loads(f['route_ids'])))
     elif 'native draft policy observation' in line:
      decisions.append({k:int(f[k]) for k in ['request_id','lane','context_tokens','verifier_rows','accepted_inputs','matched_prefix','generated']}|{'confidence':json.loads(f['raw_confidence']),'terminal':f['eos']=='true' or f['length_limit']=='true','constrained':f['constrained']=='true'})
     else:
      if not decisions:pending=[];continue
      ds={d['request_id']:d for d in decisions}
      routes=[r for r in pending if all(i in ds and ds[i]['context_tokens']<=p<ds[i]['context_tokens']+ds[i]['verifier_rows'] for i,p in r[1])]
      actual=defaultdict(set);predicted=defaultdict(set);missing=0
      for d in decisions:
       if d['generated']>1 and not d['terminal'] and not d['constrained']:
        for j in range(min(5,d['verifier_rows']-1,d['matched_prefix']+1)):
         conf.append([d['request_id'],j,d['confidence'][j],int(d['matched_prefix']>j)])
       for l in range(40):
        h=hist.get((d['request_id'],l),[])
        if h:
         for ids in list(h)[-d['verifier_rows']:]:predicted[d['lane'],l].update(ids)
        else:missing+=1
      for l,owners,ids in routes:
       lane=ds[owners[0][0]]['lane']
       actual[lane,l].update(ids)
      if len(actual)==40*len({d['lane'] for d in decisions}):
       records.append({'rows':sum(d['verifier_rows'] for d in decisions),'lanes':len({d['lane'] for d in decisions}),'requests':len(ds),'unique':sum(len(s) for s in actual.values())/40,'history_unique':sum(len(s) for s in predicted.values())/40,'missing':missing,'verify_us':int(f['verify_us']),'draft_us':int(f['draft_us']),'ids':list(ds),'warm':all(d['generated']>6 for d in decisions),'terminal':any(d['terminal'] for d in decisions)})
      for l,owners,ids in routes:
       for row,(i,p) in enumerate(owners):
        if p<ds[i]['context_tokens']+ds[i]['accepted_inputs']:
         h=hist.setdefault((i,l),deque(maxlen=6));h.append(ids[row*6:row*6+6])
      pending=[];decisions=[]
    return records, conf

def fit(X,y):
 scale=np.linalg.norm(X,axis=0);z=X/scale;beta=np.zeros(X.shape[1]);w=np.ones(len(y))
 for outer in range(8):
  for _ in range(1000):
   for j in range(len(beta)):
    residual=y-z@beta+z[:,j]*beta[j]
    beta[j]=max(0,np.dot(w*z[:,j],residual)/np.dot(w*z[:,j],z[:,j]))
  residual=abs(y-z@beta);w=np.minimum(1,1500/np.maximum(residual,1))
 return beta/scale

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('trace',type=Path)
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--workload',default='unspecified',help='Description of the trace workload, recorded without inference')
 p.add_argument('--initial-mixed-requests',type=int,default=0,help='Optional known initial mixed-request count for a separate calibration group')
 a=p.parse_args()
 if a.initial_mixed_requests < 0: p.error('initial mixed request count must be nonnegative')
 records,confidence=observations(a.trace)
 r=[x for x in records if x['warm'] and not x['terminal'] and not x['missing']]
 if len(r)<100: p.error('insufficient complete warm observations')
 X=np.array([[1,x['rows'],x['unique'],x['lanes']-1] for x in r])
 P=np.array([[1,x['rows'],x['history_unique'],x['lanes']-1] for x in r])
 y=np.array([x['verify_us'] for x in r])
 if np.any(np.linalg.norm(X,axis=0)==0): p.error('missing cost feature coverage')
 hold=np.array([(i//10)%5==4 for i in range(len(r))])
 b=fit(X[~hold],y[~hold])
 error=abs(P[hold]@b-y[hold])/y[hold]
 c=np.array(confidence)
 A=np.column_stack((np.ones(len(c)),c[:,2]));labels=c[:,3]
 chold=c[:,0]%3==0;cb=np.array([0.,1.])
 for _ in range(30):
  probability=1/(1+np.exp(-np.clip(A[~chold]@cb,-30,30)))
  w=probability*(1-probability)
  cb-=np.linalg.solve((A[~chold].T*w)@A[~chold]+.1*np.eye(2),A[~chold].T@(probability-labels[~chold])+.1*cb)
 calibration=[]
 groups=[('all',chold)]
 if a.initial_mixed_requests:
  groups.append((f'initial_mixed_requests_1_to_{a.initial_mixed_requests}',chold&(c[:,0]<=a.initial_mixed_requests)))
 for name,mask in groups:
  if not mask.any():continue
  raw=1/(1+np.exp(-np.clip(A[mask,1],-30,30)))
  probability=1/(1+np.exp(-np.clip(A[mask]@cb,-30,30)))
  calibration.append(dict(group=name,samples=int(mask.sum()),raw_brier=float(np.mean((raw-labels[mask])**2)),fitted_brier=float(np.mean((probability-labels[mask])**2))))
 report=dict(scope=__doc__,workload=a.workload,trace_sha256=hashlib.sha256(a.trace.read_bytes()).hexdigest(),
  rounds=len(records),warm_rounds=len(r),cost_features=['intercept','total_rows','sum_of_lane_mean_unique_experts','extra_lane'],
  fitted_cost_us=fit(X,y).tolist(),cost_validation=dict(training_rounds=int((~hold).sum()),heldout_rounds=int(hold.sum()),
   coefficients=b.tolist(),median_absolute_relative_error=float(np.median(error)),p90_absolute_relative_error=float(np.quantile(error,.9))),
  route_history=dict(median_absolute_unique_error=float(np.median(abs(P[:,2]-X[:,2]))),p90_absolute_unique_error=float(np.quantile(abs(P[:,2]-X[:,2]),.9))),
  confidence=dict(conditional_labels=len(c),heldout_transform=cb.tolist(),validation=calibration),
  limitations='Exploratory same-workload validation: every fifth contiguous ten-round block for costs; request IDs divisible by three for confidence. Workload composition is caller-described. No independent-workload or shorter-prefix counterfactual accuracy claim. Route forecasts use only previously accepted inputs, not current target routes. Any initial mixed subgroup is explicitly supplied by the caller.')
 with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')

if __name__=='__main__':main()
