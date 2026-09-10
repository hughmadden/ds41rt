#!/usr/bin/env python3
"""Official candidate-block masks versus tiled native max/top-2048/expansion."""
import argparse
import ast
import ctypes as C
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-lib',type=Path,required=True)
    parser.add_argument('--reference-dir',type=Path,required=True)
    parser.add_argument('--device',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    lock=json.loads((root/'docs/ds41-reference-lock.json').read_text())
    source=args.reference_dir/'inference/model.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==lock['files']['inference/model.py']
    config=args.reference_dir/'inference/config.json'
    assert hashlib.sha256(config.read_bytes()).hexdigest()==lock['files']['inference/config.json']
    cfg=json.loads(config.read_text())
    assert (cfg['candidate_topk_blocks'],cfg['candidate_block_size'])==(2048,8)
    node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='select_candidate_blocks')
    ns={'torch':torch,'F':F}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
    reference=ns['select_candidate_blocks']
    torch.cuda.set_device(args.device);torch.manual_seed(412048)
    lib=C.CDLL(str(args.native_lib))
    maximum=lib.ds41rt_v41_candidate_block_max
    maximum.argtypes=[C.c_void_p]*5+[C.c_int32]*2+[C.c_void_p];maximum.restype=C.c_int32
    top=lib.ds41rt_v41_index_top2048_blocks
    top.argtypes=[C.c_void_p]*4+[C.c_uint64,C.c_void_p]+[C.c_int32]*3+[C.c_void_p];top.restype=C.c_int32
    expand=lib.ds41rt_v41_candidate_expand
    expand.argtypes=[C.c_void_p]*3+[C.c_int32,C.c_void_p];expand.restype=C.c_int32
    stream=torch.cuda.Stream();results=[]
    with torch.cuda.stream(stream),torch.no_grad():
        def pipeline(scores,lengths,tile_width=16384):
            queries,width=scores.shape
            tiles=[]
            for start in range(0,width,tile_width):
                value=scores[:,start:start+tile_width].contiguous()
                first=torch.full((queries,),start,device='cuda',dtype=torch.int64)
                tiles.append((value,first))
            block_count=(min(width,tile_width)+7)//8
            maxima=torch.empty(queries,block_count,device='cuda')
            ids=torch.empty(queries,block_count,device='cuda',dtype=torch.int64)
            carry=torch.empty(queries,2048,device='cuda',dtype=torch.int64)
            scratch=torch.empty(queries*((block_count+4095)//4096)*2048*2,device='cuda',dtype=torch.int64)
            selected=torch.empty(queries,2048,device='cuda',dtype=torch.int32)
            positions=torch.empty(queries,16384,device='cuda',dtype=torch.int64)
            def launch(reverse=False):
                for i,(value,first) in enumerate(reversed(tiles) if reverse else tiles):
                    n=value.shape[1];b=(n+7)//8
                    assert maximum(value.data_ptr(),first.data_ptr(),lengths.data_ptr(),maxima.data_ptr(),ids.data_ptr(),queries,n,stream.cuda_stream)==0
                    assert top(maxima.data_ptr(),ids.data_ptr(),carry.data_ptr(),scratch.data_ptr(),scratch.numel()*8,
                               selected.data_ptr(),queries,b,int(i==0),stream.cuda_stream)==0
                assert expand(selected.data_ptr(),lengths.data_ptr(),positions.data_ptr(),queries,stream.cuda_stream)==0
            def compare():
                causal=scores.masked_fill(torch.arange(width,device='cuda')[None]>=lengths[:,None],-float('inf'))
                want=reference(causal,lengths[:,None],2048,8)
                want &= torch.arange(width,device='cuda')[None]<lengths[:,None]
                actual=torch.zeros_like(want)
                valid=positions>=0
                q=torch.arange(queries,device='cuda')[:,None].expand_as(positions)
                actual[q[valid],positions[valid]]=True
                assert torch.equal(actual,want)
                assert torch.equal(valid.sum(-1),actual.sum(-1)), 'duplicate expansion positions'
            launch();compare()
            original=selected.clone();original_carry=carry.clone()
            launch(True);compare()
            assert torch.equal(selected,original) and torch.equal(carry,original_carry)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):launch()
            # Keep maxima distinct while reversing the old block ordering.
            scores.neg_()
            for value,first in tiles:
                start=int(first[0].item())
                value.copy_(scores[:,start:start+value.shape[1]])
            graph.replay();compare()
            lengths.zero_();graph.replay()
            assert (positions==-1).all().item() and (selected==-1).all().item()
            # Invalid first positions and lengths fail closed before reading scores.
            value,first=tiles[0]
            for invalid in (1,-1,1048576):
                first.fill_(invalid)
                lengths.fill_(8)
                assert maximum(value.data_ptr(),first.data_ptr(),lengths.data_ptr(),maxima.data_ptr(),ids.data_ptr(),queries,value.shape[1],stream.cuda_stream)==0
                assert torch.isneginf(maxima.view(-1)[:queries*((value.shape[1]+7)//8)]).all().item()
            selected.fill_(131072);lengths.fill_(1048576)
            assert expand(selected.data_ptr(),lengths.data_ptr(),positions.data_ptr(),queries,stream.cuda_stream)==0
            assert (positions==-1).all().item()
            selected.zero_();lengths.fill_(1048577)
            assert expand(selected.data_ptr(),lengths.data_ptr(),positions.data_ptr(),queries,stream.cuda_stream)==0
            assert (positions==-1).all().item()
            # Native span/shape guards.
            ptrs=[value.data_ptr(),first.data_ptr(),lengths.data_ptr(),maxima.data_ptr(),ids.data_ptr()]
            for i in range(5):
                bad=ptrs.copy();bad[i]=0
                assert maximum(*bad,queries,value.shape[1],stream.cuda_stream)!=0
                bad[i]=(1<<64)-1
                assert maximum(*bad,queries,value.shape[1],stream.cuda_stream)!=0
                if i>=3:
                    for j in range(i):
                        bad=ptrs.copy();bad[i]=ptrs[j]
                        assert maximum(*bad,queries,value.shape[1],stream.cuda_stream)!=0
            assert maximum(*ptrs,queries,16385,stream.cuda_stream)!=0
            assert top(maxima.data_ptr(),ids.data_ptr(),carry.data_ptr(),scratch.data_ptr(),0,selected.data_ptr(),queries,block_count,1,stream.cuda_stream)!=0
            assert expand(selected.data_ptr(),lengths.data_ptr(),selected.data_ptr(),queries,stream.cuda_stream)!=0
            assert expand(selected.data_ptr(),lengths.data_ptr(),positions.data_ptr(),0,stream.cuda_stream)!=0
            results.append({'kind':'pipeline','queries':queries,'width':width,'tile_width':tile_width,
                            'reference_mask_exact':True,'reverse_carry_exact':True,'changed_graph_exact':True,
                            'empty_and_invalid_metadata':True,'guards':True})
            print(f'PASS pipeline queries={queries} width={width}',flush=True)
        for queries,width in ((1,1),(3,7),(16,8),(16,9),(80,127),(4096,1),(1,16383),(1,16384),(3,1048576)):
            blocks=(width+7)//8
            scores=torch.rand(queries,blocks,device='cuda').argsort(-1).float().repeat_interleave(8,-1)[:,:width].contiguous()
            lengths=torch.randint(0,width+1,(queries,),device='cuda',dtype=torch.int64)
            lengths[0]=width
            if queries>1:lengths[1]=0
            if queries>2:lengths[2]=max(1,width-5)
            pipeline(scores,lengths)
        # Direct top-2048 larger than one merge block, including odd merge counts.
        for count in (2047,2048,2049,4095,4096,4097,8193,16384):
            scores=torch.randperm(count,device='cuda').float()[None]
            ids=torch.randperm(count,device='cuda')[None]
            carry=torch.empty(1,2048,device='cuda',dtype=torch.int64)
            scratch=torch.empty(((count+4095)//4096)*2048*2,device='cuda',dtype=torch.int64)
            selected=torch.empty(1,2048,device='cuda',dtype=torch.int32)
            assert top(scores.data_ptr(),ids.data_ptr(),carry.data_ptr(),scratch.data_ptr(),scratch.numel()*8,selected.data_ptr(),1,count,1,stream.cuda_stream)==0
            chosen=ids.gather(-1,scores.topk(min(count,2048),-1).indices).sort(-1).values.int()
            assert torch.equal(selected[:,:chosen.shape[1]],chosen)
            assert (selected[:,chosen.shape[1]:]==-1).all().item()
            results.append({'kind':'top2048','candidates':count,'torch_topk_exact':True})
            print(f'PASS top2048 candidates={count}',flush=True)
        # Ties use the documented lowest-block-ID policy and always retain the newest block.
        width=20003;count=(width+7)//8
        scores=torch.zeros(1,count,device='cuda');scores[0,-1]=float('inf')
        ids=torch.arange(count,device='cuda',dtype=torch.int64)[None]
        carry=torch.empty(1,2048,device='cuda',dtype=torch.int64)
        scratch=torch.empty(4096,device='cuda',dtype=torch.int64)
        selected=torch.empty(1,2048,device='cuda',dtype=torch.int32)
        assert top(scores.data_ptr(),ids.data_ptr(),carry.data_ptr(),scratch.data_ptr(),scratch.numel()*8,selected.data_ptr(),1,count,1,stream.cuda_stream)==0
        wanted=torch.cat((torch.arange(2047,device='cuda',dtype=torch.int32),torch.tensor([count-1],device='cuda',dtype=torch.int32)))[None]
        assert torch.equal(selected,wanted)
        results.append({'kind':'ties','latest_pinned':True,'lowest_block_id':True})
        stream.synchronize()
    args.output.write_text(json.dumps({'scope':'Hierarchical candidate masks; deterministic lowest-block-ID ties',
        'gpu':torch.cuda.get_device_name(args.device),'device':args.device,
        'reference_model_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'native_library_sha256':hashlib.sha256(args.native_lib.read_bytes()).hexdigest(),'cases':results},indent=2)+'\n')


if __name__=='__main__':main()
