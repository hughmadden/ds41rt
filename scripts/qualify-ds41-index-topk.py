#!/usr/bin/env python3
"""Index top-512 selection, deterministic ties and bounded tiled accumulation."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-lib',type=Path,required=True)
    parser.add_argument('--device',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.cuda.set_device(args.device)
    torch.manual_seed(41512)
    lib=C.CDLL(str(args.native_lib))
    top=lib.ds41rt_v41_index_top512
    top.argtypes=[C.c_void_p]*4+[C.c_uint64,C.c_void_p]+[C.c_int32]*3+[C.c_void_p]
    top.restype=C.c_int32
    stream=torch.cuda.Stream()
    results=[]
    with torch.cuda.stream(stream),torch.no_grad():
        def expected(scores,positions):
            valid=(positions>=0)&(positions<1048576)&~torch.isnan(scores)&(scores>-float('inf'))
            masked=scores.masked_fill(~valid,-float('inf'))
            by_id=positions.argsort(dim=-1,stable=True)
            by_score=masked.gather(-1,by_id).argsort(dim=-1,descending=True,stable=True)
            ranks=by_id.gather(-1,by_score)[...,:512]
            ids=positions.gather(-1,ranks).masked_fill(~valid.gather(-1,ranks),1<<62).sort(-1).values
            ids=ids.masked_fill(ids==1<<62,-1).int()
            result=torch.full((scores.shape[0],512),-1,device='cuda',dtype=torch.int32)
            result[:,:ids.shape[1]]=ids
            return result
        for queries,count in ((1,1),(1,511),(1,512),(1,513),(3,1023),(16,1024),
                              (16,1025),(3,2049),(1,6145),(80,4095),(1,16384),(4096,3)):
            scores=torch.rand(queries,count,device='cuda').argsort(-1).float()-count//2
            positions=torch.rand(queries,count,device='cuda').argsort(-1)
            positions[:,7::79]=-1
            positions[:,11::83]=1048576
            scores[:,9::97]=float('nan')
            scores[:,5::101]=-float('inf')
            # Carry is deliberately dirty; reset must not read it.
            carry=torch.full((queries,512),-1,device='cuda',dtype=torch.int64)
            scratch=torch.empty(queries*((count+1023)//1024)*512*2,device='cuda',dtype=torch.int64)
            out=torch.empty(queries,512,device='cuda',dtype=torch.int32)
            def launch(reset=1):
                assert top(scores.data_ptr(),positions.data_ptr(),carry.data_ptr(),scratch.data_ptr(),
                           scratch.numel()*8,out.data_ptr(),queries,count,reset,stream.cuda_stream)==0
            launch()
            torch.testing.assert_close(out,expected(scores,positions),rtol=0,atol=0)
            # With distinct reachable scores, the selected set also matches torch.topk.
            valid=(positions>=0)&(positions<1048576)&~torch.isnan(scores)&(scores>-float('inf'))
            masked=scores.masked_fill(~valid,-float('inf'))
            idx=masked.topk(min(512,count),dim=-1,sorted=False).indices
            ref=positions.gather(-1,idx).masked_fill(~valid.gather(-1,idx),1<<62).sort(-1).values
            ref=ref.masked_fill(ref==1<<62,-1).int()
            assert torch.equal(out[:,:ref.shape[1]],ref)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):launch()
            # Extensive ties, both signed zeros, infinities and NaNs.
            scores.copy_(torch.floor(scores/17))
            scores[:,1::103]=float('inf')
            scores[:,2::107]=-0.
            scores[:,3::109]=0.
            graph.replay()
            torch.testing.assert_close(out,expected(scores,positions),rtol=0,atol=0)
            scores.fill_(-float('inf'))
            graph.replay()
            assert (out==-1).all().item() and (carry==0).all().item()
            ptrs=[scores.data_ptr(),positions.data_ptr(),carry.data_ptr(),scratch.data_ptr(),out.data_ptr()]
            def call(p=ptrs,n=count,q=queries,reset=1,size=None):
                return top(*p[:4],scratch.numel()*8 if size is None else size,p[4],q,n,reset,stream.cuda_stream)
            assert call(size=scratch.numel()*8-1)!=0
            for i in range(5):
                bad=ptrs.copy();bad[i]=0;assert call(bad)!=0
                bad[i]=(1<<64)-1;assert call(bad)!=0
                bad=ptrs.copy();bad[i]+=1;assert call(bad)!=0
                for j in range(i):
                    bad=ptrs.copy();bad[i]=ptrs[j];assert call(bad)!=0
            for n,q,r in ((0,queries,1),(16385,queries,1),(count,0,1),(count,4097,1),(count,queries,2)):
                assert call(n=n,q=q,reset=r)!=0
            assert (out==-1).all().item() and (carry==0).all().item()
            results.append({'queries':queries,'candidates':count,'torch_topk_untied_exact':True,
                            'deterministic_ties_exact':True,'changed_reset_graph':True,
                            'all_invalid_empty':True,'guards':True,'scratch_bytes':scratch.numel()*8})
            print(f'PASS queries={queries} candidates={count}',flush=True)
            del graph,scores,positions,carry,scratch,out
        # Entire official maximum history, streamed through 64 bounded tiles.
        count=1048576;tile=16384
        scores=(torch.randperm(count,device='cuda').float()//64-count//128)[None]
        positions=torch.randperm(count,device='cuda')[None]
        reference=expected(scores,positions)
        carry=torch.empty(1,512,device='cuda',dtype=torch.int64)
        scratch=torch.empty(16*512*2,device='cuda',dtype=torch.int64)
        out=torch.empty(1,512,device='cuda',dtype=torch.int32)
        def tiled(order):
            for i,t in enumerate(order):
                s=scores[:,t*tile:(t+1)*tile];p=positions[:,t*tile:(t+1)*tile]
                assert top(s.data_ptr(),p.data_ptr(),carry.data_ptr(),scratch.data_ptr(),
                           scratch.numel()*8,out.data_ptr(),1,tile,int(i==0),stream.cuda_stream)==0
        tiled(range(64))
        assert torch.equal(out,reference)
        forward_carry=carry.clone()
        tiled(reversed(range(64)))
        assert torch.equal(out,reference) and torch.equal(carry,forward_carry)
        # An uneven partition also changes scratch layout between calls, while
        # preserving the same query accumulator and full candidate set.
        for i,start in enumerate(range(0,count,8191)):
            s=scores[:,start:start+8191];p=positions[:,start:start+8191]
            assert top(s.data_ptr(),p.data_ptr(),carry.data_ptr(),scratch.data_ptr(),
                       scratch.numel()*8,out.data_ptr(),1,s.shape[1],int(i==0),stream.cuda_stream)==0
        assert torch.equal(out,reference) and torch.equal(carry,forward_carry)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):tiled(range(64))
        scores.neg_()
        graph.replay()
        assert torch.equal(out,expected(scores,positions))
        results.append({'queries':1,'history':count,'tile_candidates':tile,'tiles':64,
                        'forward_reverse_exact':True,'carry_order_independent':True,'uneven_partition_exact':True,
                        'changed_graph_exact':True,'scratch_bytes':scratch.numel()*8})
        print('PASS tiled history=1048576 forward/reverse/graph',flush=True)
        stream.synchronize()
    args.output.write_text(json.dumps({'scope':'Index top-512 selection only; deterministic lowest-position tie break',
        'gpu':torch.cuda.get_device_name(args.device),'device':args.device,
        'native_library_sha256':hashlib.sha256(args.native_lib.read_bytes()).hexdigest(),'cases':results},indent=2)+'\n')


if __name__=='__main__':main()
