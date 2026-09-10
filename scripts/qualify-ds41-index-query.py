#!/usr/bin/env python3
"""Real-weight learned index query projection, fused rotary/FP4 and head scaling."""
import argparse,ast,ctypes as C,hashlib,importlib.util,json,math,struct
from functools import lru_cache
from pathlib import Path
import torch
import tvm_ffi

P,I=C.c_void_p,C.c_int32
class Info(C.Structure):
    _fields_=[(n,C.c_uint32) for n in ('abi','capacity','k','n')]+[(n,C.c_uint64) for n in ('scratch','values','row_scales','mma_scales','weight_scales')]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-lib',type=Path,required=True)
    parser.add_argument('--reference-dir',type=Path,required=True)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--device',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--vectors-dir',type=Path)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    lock=json.loads((root/'docs/ds41-reference-lock.json').read_text())
    for name in ('model.py','kernel.py','config.json'):
        assert hashlib.sha256((args.reference_dir/'inference'/name).read_bytes()).hexdigest()==lock['files']['inference/'+name]
    source=args.reference_dir/'inference/model.py'
    nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('apply_rotary_emb','precompute_freqs_cis')]
    ns={'torch':torch,'math':math,'lru_cache':lru_cache};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
    spec=importlib.util.spec_from_file_location('ds41_official_kernel',args.reference_dir/'inference/kernel.py')
    reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
    assert reference.tilelang.__version__=='0.1.8'
    # TileLang 0.1.8 vectorization aliases BF16 input/FP8 output shared
    # storage without a cross-warp barrier. Disable this compiler optimization
    # for the reference quantizer; retain its exact source and arithmetic.
    reference.act_quant_kernel.pass_configs={**reference.act_quant_kernel.pass_configs,
        'tir.disable_vectorize':True}
    torch.cuda.set_device(args.device);torch.manual_seed(41256);torch.set_default_dtype(torch.bfloat16)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_tf32=False
    lib=C.CDLL(str(args.native_lib))
    def bind(name,types):
        fn=getattr(lib,name);fn.argtypes=types;fn.restype=I
        def checked(*a):
            status=fn(*a);assert status==0,(name,status)
        return checked,fn
    info,_=bind('ds41rt_v41_fp8_matrix_info',[I,I,I,P]);init,_=bind('ds41rt_v41_fp8_matrix_initialize',[I,I,I,P])
    pack,_=bind('ds41rt_v41_fp8_matrix_pack_scales',[P,P,I,I,P])
    scratch_init,_=bind('ds41rt_v41_fp8_initialize_scratch',[P,P,C.c_uint64,P,P])
    launch,_=bind('ds41rt_v41_fp8_launch',[P,P,P,P,P,C.c_uint64,P,P,I,P])
    create,_=bind('ds41rt_v41_compressor_create',[P,C.c_uint64,P]);destroy,_=bind('ds41rt_v41_compressor_destroy',[P])
    heads,_=bind('ds41rt_v41_index_weights_project',[P,P,P,P,I,P])
    prepare,prepare_raw=bind('ds41rt_v41_index_query_prepare',[P,P,P,P,P,P,I,P])
    freq,_=bind('ds41rt_v41_backbone_frequencies',[P,P,I,I,P])
    index=json.loads((args.snapshot/'model.safetensors.index.json').read_text())['weight_map'];headers={};payloads={}
    def weight(name,dtype):
        path=args.snapshot/index[name]
        with path.open('rb') as f:
            if path not in headers:
                n=struct.unpack('<Q',f.read(8))[0];headers[path]=(n,json.loads(f.read(n)))
            n,h=headers[path];entry=h[name];a,b=entry['data_offsets'];f.seek(8+n+a);raw=f.read(b-a)
        payloads[name]=hashlib.sha256(raw).hexdigest()
        return torch.frombuffer(bytearray(raw),dtype=dtype).reshape(entry['shape']).cuda()
    def save(name,value):
        if args.vectors_dir:
            (args.vectors_dir/name).write_bytes(value.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    if args.vectors_dir:args.vectors_dir.mkdir(parents=True,exist_ok=True)
    stream=torch.cuda.Stream();results=[]
    with torch.device('cuda'),torch.cuda.stream(stream),tvm_ffi.use_torch_stream(),torch.no_grad():
        table=ns['precompute_freqs_cis'](64,1048576,65536,160000,16,32,1)
        workspace=torch.empty(4*1024*1024,dtype=torch.uint8);dense=P();create(workspace.data_ptr(),workspace.numel(),C.byref(dense))
        for layer in (2,8,14,20,24,28,32,36):
            w=weight(f'layers.{layer}.attn.indexer.wq_b.weight',torch.float8_e4m3fn)
            ws=weight(f'layers.{layer}.attn.indexer.wq_b.scale',torch.float8_e8m0fnu)
            wh=weight(f'layers.{layer}.attn.indexer.weights_proj.weight',torch.bfloat16)
            assert w.shape==(4096,1280) and ws.shape==(128,40) and wh.shape==(32,5120)
            packed_ws=torch.empty(4096*1280//32,dtype=torch.uint8);pack(ws.data_ptr(),packed_ws.data_ptr(),1280,4096,stream.cuda_stream)
            for capacity in (1,16,80,256,1024,4096):
                rows=capacity if capacity<=80 else capacity-1
                shape=Info();info(capacity,1280,4096,C.byref(shape));handle=P();init(capacity,1280,4096,C.byref(handle))
                scratch=torch.empty(shape.scratch,dtype=torch.uint8);alpha=torch.empty(1,dtype=torch.float32)
                scratch_init(handle,scratch.data_ptr(),scratch.numel(),alpha.data_ptr(),stream.cuda_stream)
                qr=torch.empty(rows,1280,dtype=torch.bfloat16);hidden=torch.empty(rows,5120,dtype=torch.bfloat16)
                positions=torch.empty(rows,dtype=torch.int64);f=torch.empty(rows,32,2,dtype=torch.float32)
                projected=torch.empty(rows,4096,dtype=torch.bfloat16);head=torch.empty(rows,32,dtype=torch.bfloat16)
                q=torch.empty(rows,32,64,dtype=torch.uint8);s=torch.empty(rows,32,4,dtype=torch.uint8);hw=torch.empty_like(head)
                def run():
                    freq(positions.data_ptr(),f.data_ptr(),rows,layer,stream.cuda_stream)
                    launch(handle,qr.data_ptr(),w.data_ptr(),packed_ws.data_ptr(),scratch.data_ptr(),scratch.numel(),alpha.data_ptr(),projected.data_ptr(),rows,stream.cuda_stream)
                    heads(dense,hidden.data_ptr(),wh.data_ptr(),head.data_ptr(),rows,stream.cuda_stream)
                    prepare(projected.data_ptr(),f.data_ptr(),head.data_ptr(),q.data_ptr(),s.data_ptr(),hw.data_ptr(),rows,stream.cuda_stream)
                def inputs():
                    qr.copy_(torch.randn(qr.shape,dtype=torch.float32)*.4);hidden.copy_(torch.randn(hidden.shape,dtype=torch.float32)*.3)
                    positions.copy_((torch.arange(rows,dtype=torch.int64)*7919+65535)%1048576)
                def check():
                    aq,ascale=reference.act_quant(qr,32,'ue8m0',torch.float8_e8m0fnu)
                    assert torch.isfinite(aq.float()).all().item() and torch.isfinite(ascale.float()).all().item()
                    expect=reference.fp8_gemm(aq,ascale,w,ws,torch.float8_e8m0fnu,block_size=32)
                    assert torch.isfinite(projected).all().item(), ('native projection nonfinite',layer,capacity,int((~torch.isfinite(projected)).sum()))
                    assert torch.isfinite(expect).all().item(), ('reference projection nonfinite',layer,capacity,int((~torch.isfinite(expect)).sum()))
                    torch.testing.assert_close(projected,expect,rtol=.008,atol=.002)
                    expect_head=F.linear(hidden,wh)
                    torch.testing.assert_close(head,expect_head,rtol=.008,atol=.002)
                    torch.testing.assert_close(f,torch.view_as_real(table[positions]),rtol=0,atol=0)
                    rotated=projected.reshape(rows,32,128).clone()
                    ns['apply_rotary_emb'](rotated[None,...,-64:],table[positions])
                    pq,ps=reference.fp4_act_quant(rotated)
                    assert torch.equal(q,pq.view(torch.uint8)) and torch.equal(s,ps.view(torch.uint8))
                    assert torch.equal(hw,head*(128**-.5*32**-.5))
                    assert torch.equal(scratch[shape.values:shape.values+qr.numel()],aq.view(torch.uint8).flatten())
                    assert torch.equal(scratch[shape.row_scales:shape.row_scales+ascale.numel()],ascale.view(torch.uint8).flatten())
                    return {'query_max_abs':(projected.float()-expect.float()).abs().max().item(),
                            'head_max_abs':(head.float()-expect_head.float()).abs().max().item(),
                            'fused_rope_fp4_bytes_exact':True,'head_scaling_exact':True,'activation_bytes_exact':True}
                inputs();run();initial=check()
                for name,value in [('qr',qr),('hidden',hidden),('positions',positions),('q',q),('s',s),('head',hw)]:save(f'l{layer}-m{capacity}-c0-{name}.bin',value)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):run()
                inputs();positions.add_(1);graph.replay();changed=check()
                for name,value in [('qr',qr),('hidden',hidden),('positions',positions),('q',q),('s',s),('head',hw)]:save(f'l{layer}-m{capacity}-c1-{name}.bin',value)
                qr.zero_();hidden.zero_();graph.replay();check()
                assert ((q & 0x77)==0).all().item() and (s==1).all().item() and (hw==0).all().item()
                ptrs=[projected.data_ptr(),f.data_ptr(),head.data_ptr(),q.data_ptr(),s.data_ptr(),hw.data_ptr()]
                assert prepare_raw(*ptrs,0,stream.cuda_stream)!=0
                for i in range(6):
                    bad=ptrs.copy();bad[i]=0;assert prepare_raw(*bad,rows,stream.cuda_stream)!=0
                    if i>=3:
                        for j in range(i):
                            bad=ptrs.copy();bad[i]=ptrs[j];assert prepare_raw(*bad,rows,stream.cuda_stream)!=0
                results.append({'layer':layer,'capacity':capacity,'rows':rows,'initial':initial,'changed_graph':changed,'zero_exact':True,'guards':True,'scratch_bytes':shape.scratch})
                print(f'PASS layer={layer} capacity={capacity}',flush=True)
                stream.synchronize();del graph
        destroy(dense);stream.synchronize()
    args.output.write_text(json.dumps({'scope':'Real-weight projections and reference-exact fused packing of native BF16 projection outputs',
        'reference_quantizer_compiler_overrides':{'tir.disable_vectorize':True},
        'reference_stream_binding':'tvm_ffi.use_torch_stream','reference_tvm_ffi_version':__import__('tvm_ffi').__version__,
        'device':args.device,'gpu':torch.cuda.get_device_name(args.device),'payloads_sha256':payloads,'cases':results,
        'native_library_sha256':hashlib.sha256(args.native_lib.read_bytes()).hexdigest()},indent=2)+'\n')


if __name__=='__main__':
    import torch.nn.functional as F
    main()
