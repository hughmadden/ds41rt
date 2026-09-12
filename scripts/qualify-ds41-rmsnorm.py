#!/usr/bin/env python3
"""Check BF16 RMSNorm bit equality and measure CUDA-graph kernel latency."""
import argparse,ctypes,hashlib,json,statistics
from pathlib import Path
import torch
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--baseline', type=Path, required=True)
parser.add_argument('--candidate', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--race-fixture', action='store_true', help='Only four cached-kernel shapes; omit timing loops')
args=parser.parse_args()
paths=[str(args.baseline),str(args.candidate)]
libs=[ctypes.CDLL(p) for p in paths];fns=[]
for lib in libs:
 f=lib.ds41rt_cuda_ds4_rmsnorm_bf16_rne_async;f.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int,ctypes.c_int,ctypes.c_float,ctypes.c_void_p];f.restype=ctypes.c_int;fns.append(f)
stream=torch.cuda.Stream();torch.manual_seed(741)
def call(f,x,w,y,eps=1e-20):
 status=f(x.data_ptr(),w.data_ptr(),y.data_ptr(),x.shape[0],x.shape[1],eps,stream.cuda_stream)
 assert status==0,status
cases=[]
shapes=[(r,h) for r in [1,6,48,80] for h in [127,256,513,1280,4096,5120]]+[(r,5120) for r in [128,256,1024,2048,4096]]
if args.race_fixture:shapes=[(r,5120) for r in [1,6,80,2048]]
with torch.cuda.stream(stream):
 for rows,hidden in shapes:
  for scale in ([1.] if args.race_fixture else [0.,1e-4,1.,1e4]):
   x=(torch.randn((rows,hidden),device='cuda')*scale).bfloat16();w=torch.randn(hidden,device='cuda').bfloat16();a=torch.empty_like(x);b=torch.empty_like(x)
   call(fns[0],x,w,a);call(fns[1],x,w,b);stream.synchronize()
   assert torch.equal(a.view(torch.int16),b.view(torch.int16)),(rows,hidden,scale)
   cases.append([rows,hidden,scale])
 print('exact cases',len(cases),flush=True)
 times=[]
 for rows in ([] if args.race_fixture else [1,2,6,16,48,80,256,2048]):
  x=torch.randn((rows,5120),device='cuda',dtype=torch.bfloat16);w=torch.randn(5120,device='cuda',dtype=torch.bfloat16);y=torch.empty_like(x);stream.synchronize()
  graphs=[];iterations=100 if rows<=80 else 20
  for f in fns:
   call(f,x,w,y);stream.synchronize();g=torch.cuda.CUDAGraph()
   with torch.cuda.graph(g,stream=stream):
    for _ in range(iterations):call(f,x,w,y)
   graphs.append(g)
  samples=[[],[]]
  for repeat in range(9):
   for i in [repeat%2,1-repeat%2]:
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    start.record(stream);graphs[i].replay();end.record(stream);end.synchronize()
    samples[i].append(start.elapsed_time(end)*1000/iterations)
  record={'rows':rows,'baseline_us':statistics.median(samples[0]),'candidate_us':statistics.median(samples[1]),'samples_us':samples};times.append(record);print(record,flush=True)
args.output.write_text(json.dumps({'libraries':dict(zip(['baseline','candidate'],[{'path':p,'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in paths])), 'exact_cases':cases,'timings':times},indent=2)+'\n')
