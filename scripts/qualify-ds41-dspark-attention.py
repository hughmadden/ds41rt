#!/usr/bin/env python3
"""Native dSpark attention versus pinned blockwise reference math; no model load."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import torch


def oracle(query,ring,draft,sink,descriptors):
    outputs=[]
    for request,(slot,valid) in enumerate(descriptors):
        kv=torch.cat([ring[slot,:valid],draft[request]],dim=0).float()
        q=query[request].float()
        maximum=torch.full((5,64),-1e30,device=q.device)
        total=torch.zeros_like(maximum)
        acc=torch.zeros_like(q)
        for start in range(0,valid+5,64):
            values=kv[start:start+64]
            scores=(q@values.T)*(1/512**.5)
            new_max=torch.maximum(maximum,scores.amax(-1))
            factor=torch.exp(maximum-new_max)
            probability=torch.exp(scores-new_max[...,None])
            total=total*factor+probability.sum(-1)
            acc=acc*factor[...,None]+probability.bfloat16().float()@values
            maximum=new_max
        total+=torch.exp(sink-maximum)
        outputs.append((acc/total[...,None]).bfloat16())
    return torch.stack(outputs)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--native-lib',type=Path,required=True)
    p.add_argument('--reference-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',type=int,required=True)
    args=p.parse_args()
    lock=json.loads((Path(__file__).resolve().parents[1]/'docs/ds41-reference-lock.json').read_text())
    hashes={name:hashlib.sha256((args.reference_dir/name).read_bytes()).hexdigest() for name in ('inference/model.py','inference/kernel.py')}
    assert all(value==lock['files'][name] for name,value in hashes.items())
    torch.cuda.set_device(args.device);torch.manual_seed(41900+args.device)
    torch.backends.cuda.matmul.allow_tf32=False
    lib=C.CDLL(str(args.native_lib));P=C.c_void_p;I=C.c_int32
    init=lib.ds41rt_v41_dspark_attention_initialize;init.argtypes=[];init.restype=I;assert init()==0
    launch=lib.ds41rt_v41_dspark_attention;launch.argtypes=[P,P,P,P,P,P,I,I,P];launch.restype=I
    stream=torch.cuda.Stream();results=[]
    lengths=[2,58,59,60,63,64,65,122,123,124,127,128,3,31,96,128]
    with torch.cuda.stream(stream),torch.no_grad():
        for requests in (1,3,16):
            q=torch.empty((requests,5,64,512),device='cuda',dtype=torch.bfloat16)
            ring=torch.empty((16,128,512),device='cuda',dtype=torch.bfloat16)
            draft=torch.empty((requests,5,512),device='cuda',dtype=torch.bfloat16)
            sink=torch.empty(64,device='cuda')
            windows=torch.empty((requests,2),device='cuda',dtype=torch.int32)
            output=torch.empty_like(q)
            def populate(case):
                q.copy_((torch.randn_like(q.float())*(.3 if case==0 else 1.2)).bfloat16())
                draft.copy_((torch.randn_like(draft.float())*.4).bfloat16())
                sink.copy_(torch.linspace(-4,4,64,device='cuda')+case)
                ring.fill_(float('nan'))
                descriptors=[((r*7+case*3)%16,lengths[(r+case*5)%16]) for r in range(requests)]
                for slot,valid in descriptors:ring[slot,:valid].copy_((torch.randn((valid,512),device='cuda')*.4).bfloat16())
                windows.copy_(torch.tensor(descriptors,device='cuda',dtype=torch.int32))
                return descriptors
            def run():assert launch(q.data_ptr(),ring.data_ptr(),draft.data_ptr(),sink.data_ptr(),windows.data_ptr(),output.data_ptr(),requests,16,stream.cuda_stream)==0
            def compare(descriptors):
                expected=oracle(q,ring,draft,sink,descriptors)
                assert torch.isfinite(output).all().item()
                torch.testing.assert_close(output,expected,rtol=.008,atol=.002)
                return {'max_abs_error':(output.float()-expected.float()).abs().max().item(),'different_bf16_elements':(output!=expected).sum().item()}
            descriptors=populate(0);run();initial=compare(descriptors)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):run()
            descriptors=populate(1);graph.replay();changed=compare(descriptors)
            # Change only the sink: it contributes denominator mass with zero value.
            sink.fill_(1000);graph.replay();assert torch.count_nonzero(output).item()==0
            # Uniform scores, zero committed values and five distinct nonzero drafts:
            # every query must see ALL five, independent of its position.
            q.zero_();ring.zero_();sink.zero_()
            for k in range(5):draft[:,k].fill_(k+1)
            descriptors=[(r,128) for r in range(requests)]
            windows.copy_(torch.tensor(descriptors,device='cuda',dtype=torch.int32));graph.replay()
            expected=torch.full_like(output,15/134)
            torch.testing.assert_close(output,expected,rtol=0,atol=0)
            # With no committed prefix this kernel supports the private-only edge;
            # serving still uses prefill solely to seed caches before drafting.
            descriptors=[(r,0) for r in range(requests)]
            windows.copy_(torch.tensor(descriptors,device='cuda',dtype=torch.int32));graph.replay()
            torch.testing.assert_close(output,torch.full_like(output,2.5),rtol=0,atol=0)
            # A single nonzero value exposes the required BF16 probability cast:
            # bypassing it changes the final BF16 value, not merely an FP32 ulp.
            q.zero_();q[...,0]=1;draft.zero_()
            draft[:,1,0]=-16;draft[:,1,1]=1;draft[:,2:,0]=-512
            graph.replay()
            rounded=oracle(q,ring,draft,sink,descriptors)
            torch.testing.assert_close(output[...,1],rounded[...,1],rtol=0,atol=0)
            scores=torch.tensor([0.,-16.,-512.,-512.,-512.],device='cuda')*(1/512**.5)
            probabilities=torch.exp(scores)
            unrounded=(probabilities[1]/(probabilities.sum()+1)).bfloat16()
            assert torch.all(output[...,1]!=unrounded).item()
            windows[0,0]=16;graph.replay();assert torch.count_nonzero(output[0]).item()==0
            windows[0,0]=0;windows[0,1]=129;graph.replay();assert torch.count_nonzero(output[0]).item()==0
            assert launch(q.data_ptr(),ring.data_ptr(),draft.data_ptr(),sink.data_ptr(),windows.data_ptr(),q.data_ptr(),requests,16,stream.cuda_stream)!=0
            assert launch(q.data_ptr(),ring.data_ptr(),draft.data_ptr(),sink.data_ptr(),windows.data_ptr(),output.data_ptr(),0,16,stream.cuda_stream)!=0
            assert launch(q.data_ptr(),ring.data_ptr(),draft.data_ptr(),sink.data_ptr(),windows.data_ptr(),output.data_ptr(),requests,17,stream.cuda_stream)!=0
            stream.synchronize();del graph
            results.append({'requests':requests,'initial':initial,'changed_graph':changed,'poison_unused_ring':True,'all_five_drafts_exact':True,'private_only_exact':True,'probability_rounding_exact_counterfactual':True,'large_sink_zero':True,'invalid_descriptors_zero':True,'host_guards':True})
            print(f'PASS requests={requests}',flush=True)
    args.output.write_text(json.dumps({'scope':'Native BF16 dSpark QK/64-key online softmax/BF16 probability PV/sink; excludes owned stage integration','reference_hashes':hashes,'device':args.device,'name':torch.cuda.get_device_name(args.device),'native_library_sha256':hashlib.sha256(args.native_lib.read_bytes()).hexdigest(),'qualifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'results':results},indent=2)+'\n')
if __name__=='__main__':main()
