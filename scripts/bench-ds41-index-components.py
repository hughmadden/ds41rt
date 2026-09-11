#!/usr/bin/env python3
"""Time native index scoring and top-512 separately using generated packed inputs.

Replay equality and output poisoning reject empty/wrong-stream graphs. This is
not a mathematical correctness oracle or end-to-end throughput benchmark.
"""
import argparse,ctypes as C,hashlib,json,statistics
from pathlib import Path
import torch
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--native-lib',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--device',type=int,default=0)
args=parser.parse_args()
torch.cuda.set_device(args.device)
lib=C.CDLL(str(args.native_lib))
score=lib.ds41rt_v41_index_scores_overlay
score.argtypes=[C.c_void_p]*12+[C.c_int32]*4+[C.c_uint64,C.c_uint64,C.c_void_p];score.restype=C.c_int32
top=lib.ds41rt_v41_index_top512
top.argtypes=[C.c_void_p]*4+[C.c_uint64,C.c_void_p]+[C.c_int32]*3+[C.c_void_p];top.restype=C.c_int32
torch.manual_seed(4132)
results=[]
for rows in [1,6,80,256,1024]:
 for width in [512,4096,16384]:
  capacity=16384
  q=torch.randint(256,(rows,32,64),device='cuda',dtype=torch.uint8)
  qs=torch.full((rows,32,4),124,device='cuda',dtype=torch.uint8)
  w=torch.rand((rows,32),device='cuda',dtype=torch.bfloat16)
  k=torch.randint(256,(capacity,64),device='cuda',dtype=torch.uint8)
  ks=torch.full((capacity,4),124,device='cuda',dtype=torch.uint8)
  pages=torch.arange(64,device='cuda',dtype=torch.int32)
  lengths=torch.tensor([capacity],device='cuda',dtype=torch.int64)
  meta=torch.tensor([0,capacity,capacity,0,0,1],device='cuda',dtype=torch.int64).expand(rows,6).contiguous()
  pos=torch.arange(width,device='cuda',dtype=torch.int64).expand(rows,width).contiguous()
  out=torch.empty((rows,width),device='cuda')
  prop=k[:1];ps=ks[:1]
  carry=torch.empty((rows,512),device='cuda',dtype=torch.int64)
  scratch=torch.empty((rows*((width+1023)//1024)*512*2,),device='cuda',dtype=torch.int64)
  selected=torch.empty((rows,512),device='cuda',dtype=torch.int32)
  def scoring():
   assert score(*[x.data_ptr() for x in [q,qs,w,k,ks,pages,lengths,meta,pos,out,prop,ps]],rows,width,1,64,capacity,1,torch.cuda.current_stream().cuda_stream)==0
  def selecting():
   assert top(out.data_ptr(),pos.data_ptr(),carry.data_ptr(),scratch.data_ptr(),scratch.numel()*8,selected.data_ptr(),rows,width,1,torch.cuda.current_stream().cuda_stream)==0
  record={'rows':rows,'width':width}
  for name,fn in [('score',scoring),('top512',selecting)]:
   fn();torch.cuda.synchronize()
   g=torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):fn()
   expected=(out if name=='score' else selected).clone()
   (out if name=='score' else selected).fill_(-12345)
   g.replay();torch.cuda.synchronize()
   torch.testing.assert_close(out if name=='score' else selected,expected,rtol=0,atol=0)
   assert torch.isfinite(out).all() and torch.count_nonzero(out)>0
   samples=[]
   for _ in range(5):
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(5):g.replay()
    b.record();b.synchronize();samples.append(a.elapsed_time(b)*200)
   record[name+'_us']=statistics.median(samples)
  results.append(record);print(record,flush=True)
  args.output.write_text(json.dumps(dict(
   scope='Synthetic all-valid committed-cache index scoring and top-512, separate captured kernels; no proposal reads, network, or API overhead.',
   device=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),
   native_sha256=hashlib.sha256(args.native_lib.read_bytes()).hexdigest(),
   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
   samples=5,replays_per_sample=5,results=results),indent=2)+'\n')
