#!/usr/bin/env python3
"""Qualify target-layer stream means and packed tap slices on CUDA."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import torch

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-lib',type=Path,required=True)
    p.add_argument('--reference-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    reference=args.reference_root/'inference/model.py'
    lock=json.loads((Path(__file__).resolve().parents[1]/'docs/ds41-reference-lock.json').read_text())
    assert hashlib.sha256(reference.read_bytes()).hexdigest()==lock['files']['inference/model.py']
    assert 'main_hiddens.append(h.mean(dim=2))' in reference.read_text()
    lib=C.CDLL(str(args.native_lib));fn=lib.ds41rt_v41_dspark_tap
    fn.argtypes=[C.c_void_p,C.c_void_p,C.c_int32,C.c_int32,C.c_void_p];fn.restype=C.c_int32
    torch.manual_seed(413);stream=torch.cuda.Stream();results=[]
    with torch.cuda.stream(stream),torch.no_grad():
        for rows in [1,13,73,255,1001,4095,4096]:
            inputs=[torch.empty((1,rows,4,5120),device='cuda',dtype=torch.bfloat16) for _ in range(3)]
            output=torch.full((rows,3,5120),-123.,device='cuda',dtype=torch.bfloat16)
            def fill(case):
                for tap,x in enumerate(inputs):
                    x.copy_(torch.randn_like(x.float())*torch.exp2(torch.randint(-10,10,x.shape,device='cuda')))
                    x[:,::5].zero_()
                    # Cancellation stresses FP32 accumulation before BF16 rounding.
                    x[:,1::5,0].fill_(256);x[:,1::5,1].fill_(.5+tap*.25)
                    x[:,1::5,2].fill_(-256);x[:,1::5,3].fill_(case*.125)
            def launch(tap):assert fn(inputs[tap].data_ptr(),output.data_ptr(),rows,37+tap,stream.cuda_stream)==0
            fill(0)
            for tap in [2,0,1]:
                before=output.clone();launch(tap)
                expected=inputs[tap].mean(dim=2).squeeze(0)
                torch.testing.assert_close(output[:,tap],expected,rtol=0,atol=0)
                for other in set(range(3))-{tap}:assert torch.equal(output[:,other],before[:,other])
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):
                for tap in [2,0,1]:launch(tap)
            fill(1);graph.replay()
            expected=torch.stack([x.mean(dim=2).squeeze(0) for x in inputs],dim=1)
            torch.testing.assert_close(output,expected,rtol=0,atol=0)
            for layer in [36,40]:assert fn(inputs[0].data_ptr(),output.data_ptr(),rows,layer,stream.cuda_stream)!=0
            for bad in [0,4097]:assert fn(inputs[0].data_ptr(),output.data_ptr(),bad,37,stream.cuda_stream)!=0
            assert fn(inputs[0].data_ptr(),inputs[0].data_ptr(),rows,37,stream.cuda_stream)!=0
            results.append({'rows':rows,'tap_order':[39,37,38],'initial_exact':True,'changed_graph_exact':True,'other_slices_unchanged':True,'guards':True})
            print('PASS',rows,flush=True)
        stream.synchronize()
    args.output.write_text(json.dumps({'device':torch.cuda.get_device_name(0),'reference_revision':lock['revision'],'results':results,'native_sha256':hashlib.sha256(args.native_lib.read_bytes()).hexdigest(),'qualifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
if __name__=='__main__':main()
