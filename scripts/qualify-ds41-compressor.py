#!/usr/bin/env python3
"""Official-weight CSA2 compressor projection/pooling/normalization qualification."""
import argparse
import ast
import ctypes as C
import hashlib
import json
import struct
from pathlib import Path
from types import SimpleNamespace
import torch


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--native-lib',type=Path,required=True)
    parser.add_argument('--reference-dir',type=Path,required=True)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',type=int,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    lock=json.loads((root/'docs/ds41-reference-lock.json').read_text())
    source=args.reference_dir/'inference/model.py';config_path=args.reference_dir/'inference/config.json'
    for p in [source,config_path]:assert hashlib.sha256(p.read_bytes()).hexdigest()==lock['files']['inference/'+p.name]
    config=json.loads(config_path.read_text());assert config['norm_eps']==1e-20
    ns={'torch':torch,'nn':torch.nn,'ModelArgs':SimpleNamespace,
        'Linear':lambda inp,out,dtype=None:torch.nn.Linear(inp,out,bias=False,dtype=dtype or torch.bfloat16)}
    nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name in ['Compressor','RMSNorm']]
    assert len(nodes)==2
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
    index=json.loads((args.snapshot/'model.safetensors.index.json').read_text())['weight_map']
    payload_hashes={}
    def load(name):
        path=args.snapshot/index[name]
        with path.open('rb') as f:
            length=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(length));entry=header[name]
            assert entry['dtype']=='BF16'
            a,b=entry['data_offsets'];f.seek(8+length+a);data=f.read(b-a);assert len(data)==b-a
        payload_hashes[name]=hashlib.sha256(data).hexdigest()
        return torch.frombuffer(bytearray(data),dtype=torch.bfloat16).reshape(entry['shape']).cuda()
    torch.cuda.set_device(args.device);torch.manual_seed(410233)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    lib=C.CDLL(str(args.native_lib));P=C.c_void_p;I=C.c_int32
    def bind(name,types):
        fn=getattr(lib,name);fn.argtypes=types;fn.restype=I;return fn
    create=bind('ds41rt_v41_compressor_create',[P,C.c_uint64,C.POINTER(P)])
    destroy=bind('ds41rt_v41_compressor_destroy',[P])
    project=bind('ds41rt_v41_compressor_project',[P,P,P,P,I,I,P])
    pool=bind('ds41rt_v41_compressor_pool',[P,P,P,P,P,P,P,I,I,P])
    norm=bind('ds41rt_v41_attention_norm',[P,P,P,P,I,I,P])
    ptr=lambda t:t.data_ptr()
    def check(status):assert status==0,status
    workspace=torch.empty(4*1024*1024,device='cuda',dtype=torch.uint8);handle=P()
    check(create(ptr(workspace),workspace.numel(),C.byref(handle)))
    stream=torch.cuda.Stream();results=[]
    try:
        with torch.cuda.stream(stream),torch.no_grad():
            pool_results=[]
            for rows in [1,16,80,4096]:
                slots=16
                kv=torch.randn((rows,512),device='cuda');score=torch.randn_like(kv)*80
                pk=torch.randn((slots,512),device='cuda');ps=torch.randn_like(pk)*80
                weight=(torch.randn(512,device='cuda')*.2+1).bfloat16();out=torch.empty((rows,512),device='cuda',dtype=torch.bfloat16)
                desc=torch.tensor([0 if row==0 else (slots+row-1 if row%2 else (1<<64)-1) for row in range(rows)],device='cuda',dtype=torch.uint64)
                def launch_pool():check(pool(ptr(kv),ptr(score),ptr(pk),ptr(ps),ptr(desc),ptr(weight),ptr(out),rows,slots,stream.cuda_stream))
                def pool_oracle():
                    indices=desc.long();valid=(indices>=0)&(indices<slots+torch.arange(rows,device='cuda'))
                    indices=indices.clamp(0,slots+rows-1)
                    pair=torch.stack([torch.cat([pk,kv])[indices],kv],1)
                    gates=torch.stack([torch.cat([ps,score])[indices],score],1).softmax(1)
                    pooled=(pair*gates).sum(1).bfloat16().float()
                    expected=(pooled*torch.rsqrt(pooled.square().mean(-1,keepdim=True)+config['norm_eps'])*weight.float()).bfloat16()
                    expected[~valid]=0
                    torch.testing.assert_close(out,expected,rtol=.008,atol=.002)
                    return (out.float()-expected.float()).abs().max().item()
                launch_pool();stream.synchronize();graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):launch_pool()
                errors=[]
                for case in range(3):
                    if case==1:kv.mul_(1e-10);pk.mul_(1e-10);score.zero_();ps.zero_()
                    if case==2:kv.zero_();pk.zero_()
                    graph.replay();errors.append(pool_oracle())
                # Inputs, including committed state, are untouched by a replay.
                snapshot=[kv.clone(),score.clone(),pk.clone(),ps.clone()]
                graph.replay()
                assert all(torch.equal(x,y) for x,y in zip([kv,score,pk,ps],snapshot))
                for invalid in [(1<<64)-1,1<<40,slots+rows]:
                    desc.fill_(invalid);graph.replay();assert torch.count_nonzero(out).item()==0
                assert pool(ptr(kv),ptr(score),ptr(pk),ptr(ps),ptr(desc),ptr(weight),ptr(kv),rows,slots,stream.cuda_stream)!=0
                assert pool(ptr(kv),ptr(score),ptr(pk),ptr(ps),ptr(desc),ptr(weight),ptr(out),rows,17,stream.cuda_stream)!=0
                stream.synchronize();graph.reset()
                pool_results.append({'rows':rows,'extreme_equal_tiny_zero_errors':errors,'invalid_descriptors_zero':True,'overlap_and_slot_guards':True})
            for layer_id in [2,8,14,20]:
                ratio=config['compress_ratios'][layer_id]
                weights={name:load(f'layers.{layer_id}.attn.compressor.{name}.weight') for name in (['wkv','wgate','norm'] if ratio==2 else ['wkv','norm'])}
                # Constructors get the actual model epsilon and precision modes.
                reference=ns['Compressor'](SimpleNamespace(**{**config,'max_batch_size':16}),layer_id).cuda()
                reference.norm.weight.data=weights['norm'].clone()
                reference.wkv.weight.copy_(weights['wkv'])
                if ratio==2:reference.wgate.weight.copy_(weights['wgate'])
                for requests in [1,3,16]:
                    for chunks in [[1]*9,[1,2,5,64,127]]:
                        length=sum(chunks)
                        x=(torch.randn((requests,length,5120),device='cuda')*.125).bfloat16()
                        # Small inputs detect epsilon and pre-normalization BF16 boundaries.
                        x[:,::13]*=1e-8
                        expected=reference(x,0).clone()
                        pending_kv=torch.zeros((requests,512),device='cuda')
                        pending_scores=torch.zeros_like(pending_kv)
                        start=0;max_error=0.;projection_error=0.;different=0
                        for size in chunks:
                            inp=x[:,start:start+size].contiguous().reshape(-1,5120);rows=inp.shape[0]
                            projected=torch.empty((rows,512),device='cuda',dtype=torch.float32 if ratio==2 else torch.bfloat16)
                            scores=torch.empty((rows,512),device='cuda')
                            output=torch.empty((rows,512),device='cuda',dtype=torch.bfloat16)
                            descriptors=[];emitted=[]
                            for r in range(requests):
                                for j in range(size):
                                    pos=start+j;row=r*size+j
                                    descriptors.append((r if j==0 else requests+row-1) if pos%2 else (1<<64)-1)
                                    if ratio==1 or pos%2:emitted.append((row,r,pos//ratio))
                            pred=torch.tensor(descriptors,device='cuda',dtype=torch.uint64)
                            before=[pending_kv.clone(),pending_scores.clone()]
                            def launch():
                                check(project(handle,ptr(inp),ptr(weights['wkv']),ptr(projected),rows,ratio,stream.cuda_stream))
                                if ratio==2:
                                    check(project(handle,ptr(inp),ptr(weights['wgate']),ptr(scores),rows,ratio,stream.cuda_stream))
                                    check(pool(ptr(projected),ptr(scores),ptr(pending_kv),ptr(pending_scores),ptr(pred),ptr(weights['norm']),ptr(output),rows,requests,stream.cuda_stream))
                                else:check(norm(ptr(projected),ptr(weights['norm']),None,ptr(output),rows,512,stream.cuda_stream))
                            launch();stream.synchronize();graph=torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph,stream=stream):launch()
                            graph.replay();first=output.clone();graph.replay()
                            assert torch.equal(output,first)
                            assert torch.equal(pending_kv,before[0]) and torch.equal(pending_scores,before[1])
                            # Compare FP32 dot to FP64 accumulation to separate GEMM reduction error.
                            dot=(inp.double() @ weights['wkv'].double().T).float()
                            if ratio==2:
                                torch.testing.assert_close(projected,dot,rtol=2e-5,atol=1e-4)
                                projection_error=max(projection_error,(projected-dot).abs().max().item())
                                gate_dot=(inp.double() @ weights['wgate'].double().T).float()
                                torch.testing.assert_close(scores,gate_dot,rtol=2e-5,atol=1e-4)
                                # Isolate pooling/norm from projection reduction differences.
                                for row,r,pos in emitted:
                                    old=pred[row].item()
                                    ka=pending_kv[old] if old<requests else projected[old-requests]
                                    sa=pending_scores[old] if old<requests else scores[old-requests]
                                    pooled=(torch.stack([ka,projected[row]])*torch.stack([sa,scores[row]]).softmax(0)).sum(0).bfloat16()
                                    oracle=reference.norm(pooled)
                                    torch.testing.assert_close(output[row],oracle,rtol=.008,atol=.002)
                            else:torch.testing.assert_close(projected,dot.bfloat16(),rtol=.008,atol=.002)
                            for row,r,pos in emitted:
                                oracle=expected[r,pos]
                                torch.testing.assert_close(output[row],oracle,rtol=.016,atol=.004)
                                max_error=max(max_error,(output[row].float()-oracle.float()).abs().max().item())
                                different+=(output[row]!=oracle).sum().item()
                            if ratio==2:
                                incomplete=[r*size+j for r in range(requests) for j in range(size) if (start+j)%2==0]
                                if incomplete:assert torch.count_nonzero(output[incomplete]).item()==0
                                # Explicit commit only after accepted execution, never during capture/replay.
                                if (start+size)%2:
                                    for r in range(requests):pending_kv[r].copy_(projected[(r+1)*size-1]);pending_scores[r].copy_(scores[(r+1)*size-1])
                                # Invalid device descriptors fail closed even on graph replay.
                                pred.fill_((1<<64)-1);graph.replay();assert torch.count_nonzero(output).item()==0
                            assert project(handle,ptr(inp),ptr(weights['wkv']),ptr(inp),rows,ratio,stream.cuda_stream)!=0
                            assert project(handle,ptr(inp),ptr(weights['wkv']),ptr(projected),0,ratio,stream.cuda_stream)!=0
                            stream.synchronize();graph.reset();start+=size
                        assert start==length
                        results.append({'layer':layer_id,'ratio':ratio,'requests':requests,'chunks':chunks,'reference_max_abs_error':max_error,'different_bf16_elements':different,'projection_fp64_max_abs_error':projection_error,'state_unchanged_until_commit':True,'graph_replay_exact':True})
                        print('PASS',results[-1],flush=True)
            stream.synchronize()
    finally:
        stream.synchronize();check(destroy(handle))
    assert len(results)==24 and len(pool_results)==4
    args.output.write_text(json.dumps({'scope':'Real checkpoint compressor weights with synthetic activations; native primitives, no Rust request state or serving','device':args.device,'name':torch.cuda.get_device_name(args.device),'reference_revision':lock['revision'],'norm_eps':config['norm_eps'],'checkpoint_payload_sha256':payload_hashes,'native_library_sha256':hashlib.sha256(args.native_lib.read_bytes()).hexdigest(),'qualifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'pool_results':pool_results,'results':results},indent=2)+'\n')

if __name__=='__main__':main()
