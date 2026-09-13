"""Reproduce captured FP8 QB divergence under concurrent memory traffic.

Requires CUDA PyTorch, safetensors, the native FP8 library, checkpoint, and a
qb_pair capture directory. Retains every target result. This is a correctness
stress test, not a performance benchmark. --serial moves eviction onto the
target stream as a control; --eviction-only omits the competing GEMM.
"""
import argparse
import ctypes as C
import json
from pathlib import Path
import torch
from safetensors import safe_open
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('native_lib', type=Path)
parser.add_argument('snapshot', type=Path)
parser.add_argument('capture', type=Path)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--competitor-capacity', type=int, default=16, choices=[1, 16, 80, 256, 1024, 4096])
parser.add_argument('--evict-mib', type=int, default=256)
parser.add_argument('--steps', type=int, default=256)
parser.add_argument('--trials', type=int, default=32)
parser.add_argument('--serial', action='store_true')
parser.add_argument('--eviction-only', action='store_true')
parser.add_argument('--failure-output', type=Path)
args = parser.parse_args()
assert args.steps > 0 and args.trials > 0 and args.evict_mib >= 0
torch.cuda.set_device(args.device)
cap=args.capture; meta=json.loads((cap/'qb.json').read_text()); print(meta,flush=True)
model=args.snapshot
index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
def tensor(name):
 with safe_open(model/index[name],framework='pt',device='cpu') as f: return f.get_tensor(name)
def load(name,width): return torch.frombuffer(bytearray((cap/name).read_bytes()),dtype=torch.bfloat16).reshape(meta['rows'],width)
x=load('input.bf16',1280).cuda(); expected=load('expected.bf16',32768); observed=load('actual.bf16',32768)
w=tensor(f"layers.{meta['layer']}.attn.wq_b.weight").cuda(); scales=tensor(f"layers.{meta['layer']}.attn.wq_b.scale").cuda()
P=C.c_void_p; I=C.c_int32
class Info(C.Structure):
 _fields_=[(n,C.c_uint32) for n in ('abi','capacity','k','n')]+[(n,C.c_uint64) for n in ('scratch','values','row_scales','mma_scales','weight_scales')]
lib=C.CDLL(str(args.native_lib))
def bind(name,types):
 fn=getattr(lib,name);fn.argtypes=types;fn.restype=I
 def checked(*args):
  status=fn(*args)
  if status: raise RuntimeError(name,status)
 return checked
infofn=bind('ds41rt_v41_fp8_matrix_info',[I,I,I,C.POINTER(Info)])
init=bind('ds41rt_v41_fp8_matrix_initialize',[I,I,I,C.POINTER(P)])
init_scratch=bind('ds41rt_v41_fp8_initialize_scratch',[P,P,C.c_uint64,P,P])
pack=bind('ds41rt_v41_fp8_matrix_pack_scales',[P,P,I,I,P])
launch=bind('ds41rt_v41_fp8_launch',[P,P,P,P,P,C.c_uint64,P,P,I,P])

capacities=[1 if meta['rows']==1 else 16, args.competitor_capacity]
infos=[];handles=[]
for capacity in capacities:
 info=Info();infofn(capacity,1280,32768,C.byref(info));h=P();init(capacity,1280,32768,C.byref(h));infos.append(info);handles.append(h)
streams=[torch.cuda.Stream(),torch.cuda.Stream()]
packed=torch.empty(infos[0].weight_scales,device='cuda',dtype=torch.uint8)
a=[torch.empty(info.scratch,device='cuda',dtype=torch.uint8) for info in infos]
alpha=[torch.empty(1,device='cuda') for _ in streams]
steps=args.steps
y=(x[:1].float().repeat(capacities[1],1)*1.03).bfloat16()
retained=torch.empty((steps,meta['rows'],32768),device='cuda',dtype=torch.bfloat16)
other=torch.empty((capacities[1],32768),device='cuda',dtype=torch.bfloat16)
evict_mib=args.evict_mib
eviction=torch.zeros(evict_mib*1024*1024//4,device='cuda') if evict_mib else None
torch.cuda.synchronize()
pack(scales.data_ptr(),packed.data_ptr(),1280,32768,streams[0].cuda_stream)
for i,s in enumerate(streams):init_scratch(handles[i],a[i].data_ptr(),infos[i].scratch,alpha[i].data_ptr(),s.cuda_stream)
torch.cuda.synchronize()
def run(i,step):
 value=x if i==0 else y
 output=retained[step] if i==0 else other
 launch(handles[i],value.data_ptr(),w.data_ptr(),packed.data_ptr(),a[i].data_ptr(),infos[i].scratch,alpha[i].data_ptr(),output.data_ptr(),value.shape[0],streams[i].cuda_stream)
for i in range(2):run(i,0)
torch.cuda.synchronize()
assert torch.equal(retained[0].cpu(),expected),'standalone baseline differs from captured expected'
graphs=[]
for i in range(2):
 g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g,stream=streams[i]):
  for step in range(steps):
   if i==0 or not args.eviction_only:run(i,step)
   if i==(0 if args.serial else 1) and eviction is not None:eviction.add_(0.001)
 graphs.append(g)
print('captured mixed capacities',capacities,'steps',steps,flush=True)
for trial in range(args.trials):
 for i in range(1 if args.serial else 2):
  with torch.cuda.stream(streams[i]):graphs[i].replay()
 for s in streams:s.synchronize()
 result=retained.cpu()
 different=(result!=expected).reshape(steps,-1).any(dim=1)
 count=int(different.sum())
 print('trial',trial,'bad projections',count,flush=True)
 if count:
  first=int(different.nonzero()[0]);bad=result[first]
  print('bad_steps',different.nonzero().flatten().tolist(),flush=True)
  print('first',first,'changed',int((bad!=expected).sum()),'peak',float((bad.float()-expected.float()).abs().max()),flush=True)
  if args.failure_output:args.failure_output.write_bytes(bad.view(torch.uint8).numpy().tobytes())
  raise SystemExit(1)
print(f'PASS all {steps * args.trials} retained projections exact',flush=True)
