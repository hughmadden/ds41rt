#!/usr/bin/env python3
"""Check native router initialization, rejected inputs, and ABI alignment fallback."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-lib',type=Path,required=True)
    p.add_argument('--baseline-lib',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    torch.cuda.set_device(0);torch.manual_seed(4191)
    lib=C.CDLL(str(a.native_lib.resolve()))
    baseline_lib=C.CDLL(str(a.baseline_lib.resolve()))
    call=lib.ds41rt_v41_router;baseline=baseline_lib.ds41rt_v41_router
    call.argtypes=baseline.argtypes=[C.c_void_p]*8+[C.c_int32,C.c_int32,C.c_void_p]
    rows,experts=26,384
    x=torch.randn(rows,5120,device='cuda',dtype=torch.bfloat16)
    w=torch.randn(experts,5120,device='cuda',dtype=torch.bfloat16)
    bias=torch.randn(experts,device='cuda');mask=torch.zeros(rows,device='cuda',dtype=torch.uint8)
    scores=torch.empty(rows,experts,device='cuda');ids=torch.empty(rows,6,device='cuda',dtype=torch.int32);routing=torch.empty(rows,6,device='cuda')
    args=[t.data_ptr() for t in (x,w,bias,bias,mask,scores,ids,routing)]+[rows,experts,torch.cuda.current_stream().cuda_stream]
    # A launch must not implicitly load modules during graph preparation/replay.
    assert call(*args)==600  # cudaErrorNotReady
    assert lib.ds41rt_v41_router_initialize()==0
    assert lib.ds41rt_v41_router_initialize()==0
    checks=[]
    for slot,value in [(8,0),(8,-1),(8,4097),(9,127),(0,0),(1,0),(2,0),(3,0),(5,0),(6,0),(7,0),(5,x.data_ptr()),(5,(1<<64)-16)]:
        bad=args.copy();bad[slot]=value
        assert call(*bad)==1,(slot,value)  # cudaErrorInvalidValue
        checks.append(dict(argument=slot,value=value,status=1))
    assert call(*args)==0
    torch.cuda.synchronize()
    # Keep the legacy public ABI valid for BF16 buffers with only 2-byte alignment.
    storage=torch.empty(x.numel()+1,device='cuda',dtype=torch.bfloat16)
    unaligned=storage[1:].view_as(x);unaligned.copy_(x)
    assert unaligned.data_ptr()%16==2
    test=args.copy();test[0]=unaligned.data_ptr()
    assert call(*test)==0
    expected_scores=torch.empty_like(scores);expected_ids=torch.empty_like(ids);expected_routing=torch.empty_like(routing)
    reference=test.copy();reference[5:8]=[t.data_ptr() for t in (expected_scores,expected_ids,expected_routing)]
    assert baseline(*reference)==0
    torch.cuda.synchronize()
    assert torch.equal(scores,expected_scores)
    assert torch.equal(ids,expected_ids)
    assert torch.equal(routing,expected_routing)
    a.output.write_text(json.dumps(dict(passed=True,initialize_required=True,repeat_initialize=True,
        alignment_fallback_exact=True,rejected=checks,
        native_sha256=hashlib.sha256(a.native_lib.read_bytes()).hexdigest()),indent=2)+'\n')
    print('PASS initialization, 13 rejection cases, exact unaligned fallback',flush=True)


if __name__=='__main__':
    main()
