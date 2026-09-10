#!/usr/bin/env python3
"""Fixed FP8 KV encoding versus official K32 quantization, plus sparse cache writes."""
import argparse,ast,ctypes as C,hashlib,importlib.util,json,math
from functools import lru_cache
from pathlib import Path
import torch
import tvm_ffi


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-lib',type=Path,required=True);p.add_argument('--reference-dir',type=Path,required=True)
    p.add_argument('--device',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    lock=json.loads((root/'docs/ds41-reference-lock.json').read_text())
    for name in ('model.py','kernel.py'):
        assert hashlib.sha256((a.reference_dir/'inference'/name).read_bytes()).hexdigest()==lock['files']['inference/'+name]
    spec=importlib.util.spec_from_file_location('ds41_official_kernel',a.reference_dir/'inference/kernel.py')
    ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    assert ref.tilelang.__version__=='0.1.8'
    # Required to avoid the TileLang 0.1.8 shared-memory aliasing race documented
    # in the learned-query qualification. The quantizer arithmetic is unchanged.
    ref.act_quant_kernel.pass_configs={**ref.act_quant_kernel.pass_configs,'tir.disable_vectorize':True}
    source=a.reference_dir/'inference/model.py'
    nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('apply_rotary_emb','precompute_freqs_cis')]
    ns={'torch':torch,'math':math,'lru_cache':lru_cache};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
    torch.cuda.set_device(a.device);torch.manual_seed(41512528)
    lib=C.CDLL(str(a.native_lib));pack=lib.ds41rt_v41_kv_pack;store=lib.ds41rt_v41_kv_store
    pack.argtypes=[C.c_void_p]*4+[C.c_int32,C.c_void_p];pack.restype=C.c_int32
    store.argtypes=[C.c_void_p]*5+[C.c_int32,C.c_uint64,C.c_void_p];store.restype=C.c_int32
    stream=torch.cuda.Stream();results=[]
    with torch.device('cuda'),torch.cuda.stream(stream),tvm_ffi.use_torch_stream(),torch.no_grad():
        table=ns['precompute_freqs_cis'](64,1048576,65536,160000,16,32,1)
        cases=[(f'random-{r}',torch.randn(r,512).bfloat16(),False) for r in (1,3,16,80,255,4096)]
        codes=torch.arange(65536,dtype=torch.int32);codes=codes[(codes&0x7f80)!=0x7f80].to(torch.uint16)
        all_bits=torch.zeros(65536,dtype=torch.uint16);all_bits[:codes.numel()]=codes
        cases.append(('all-finite-bf16',all_bits.view(torch.bfloat16).reshape(-1,512),False))
        for r in (1,80,4096):cases.append((f'rotary-{r}',torch.randn(r,512).bfloat16(),True))
        for name,x,rotary in cases:
            rows=x.shape[0];positions=(torch.arange(rows)*7919+65535)%1048576
            freq=torch.view_as_real(table[positions]).contiguous() if rotary else None
            ys=torch.full((rows*512+2,),0xa5,dtype=torch.uint8);ss=torch.full((rows*16+2,),0xa5,dtype=torch.uint8)
            y=ys[1:-1].reshape(rows,512);s=ss[1:-1].reshape(rows,16)
            def launch():assert pack(x.data_ptr(),freq.data_ptr() if rotary else 0,y.data_ptr(),s.data_ptr(),rows,stream.cuda_stream)==0
            def check():
                v=x.clone()
                if rotary:ns['apply_rotary_emb'](v[None,...,-64:],torch.view_as_complex(freq))
                q,scale=ref.act_quant(v,32,'ue8m0',torch.float8_e8m0fnu)
                assert torch.equal(y,q.view(torch.uint8)) and torch.equal(s,scale.view(torch.uint8)),name
                assert ys[0].item()==ys[-1].item()==ss[0].item()==ss[-1].item()==0xa5
            launch();check();g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g,stream=stream):launch()
            x.neg_()
            if rotary:freq.copy_(torch.view_as_real(table[(positions+1)%1048576]))
            g.replay();check();x.zero_();g.replay();check()
            assert ((y&127)==0).all().item() and (s==105).all().item()
            args=[x.data_ptr(),freq.data_ptr() if rotary else 0,y.data_ptr(),s.data_ptr(),rows,stream.cuda_stream]
            for i in (0,2,3):
                for value in (0,2**64-1):
                    bad=args.copy();bad[i]=value;assert pack(*bad)!=0
            for i in (2,3):
                bad=args.copy();bad[i]=x.data_ptr();assert pack(*bad)!=0
            assert pack(x.data_ptr(),0,y.data_ptr(),y.data_ptr(),rows,stream.cuda_stream)!=0
            for bad_rows in (0,4097):assert pack(x.data_ptr(),0,y.data_ptr(),s.data_ptr(),bad_rows,stream.cuda_stream)!=0
            results.append(dict(kind='pack',name=name,rows=rows,reference_bytes_exact=True,changed_graph_exact=True,zero_exact=True,guard_bytes=True))
            print(f'PASS pack {name}',flush=True);del g
        for rows,capacity in ((1,1),(80,8192),(4096,8192),(3,16777216)):
            values=torch.randint(256,(rows,512),dtype=torch.uint8);scales=torch.randint(256,(rows,16),dtype=torch.uint8)
            cache=torch.empty(capacity,512,dtype=torch.uint8);cs=torch.empty(capacity,16,dtype=torch.uint8)
            watch=torch.arange(capacity) if capacity<10000 else torch.tensor([0,1,capacity-2,capacity-1])
            cache.index_fill_(0,watch,0xa5);cs.index_fill_(0,watch,0xa5)
            expected=cache[watch].clone();expected_s=cs[watch].clone()
            dst=torch.arange(rows,dtype=torch.int64);dst[0]=capacity-1
            if rows>1:dst[-1]=-1
            if rows>2:dst[-2]=capacity
            def launch():assert store(values.data_ptr(),scales.data_ptr(),dst.data_ptr(),cache.data_ptr(),cs.data_ptr(),rows,capacity,stream.cuda_stream)==0
            def check():
                valid=(dst>=0)&(dst<capacity);d=dst[valid];src=torch.arange(rows)[valid]
                at=torch.searchsorted(watch,d).clamp(max=watch.numel()-1);keep=watch[at]==d
                expected[at[keep]]=values[src[keep]];expected_s[at[keep]]=scales[src[keep]]
                assert torch.equal(cache[watch],expected) and torch.equal(cs[watch],expected_s)
            launch();check();g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g,stream=stream):launch()
            values.bitwise_xor_(0xff);scales.bitwise_xor_(0x43);dst[0]=max(capacity-2,0);g.replay();check()
            dst.fill_(-1);g.replay();check()
            args=[values.data_ptr(),scales.data_ptr(),dst.data_ptr(),cache.data_ptr(),cs.data_ptr(),rows,capacity,stream.cuda_stream]
            for i in range(5):
                for value in (0,2**64-1):
                    bad=args.copy();bad[i]=value;assert store(*bad)!=0
                if i<3:
                    bad=args.copy();bad[3]=args[i];assert store(*bad)!=0
            for i,invalid in ((5,0),(5,4097),(6,0),(6,16777217)):
                bad=args.copy();bad[i]=invalid;assert store(*bad)!=0
            results.append(dict(kind='store',rows=rows,capacity=capacity,watched_rows=watch.numel(),changed_graph_exact=True,skip_exact=True,guards=True))
            print(f'PASS store rows={rows} capacity={capacity}',flush=True)
            del g,cache,cs,expected,expected_s
        stream.synchronize()
    a.output.write_text(json.dumps(dict(device=a.device,gpu=torch.cuda.get_device_name(a.device),native_library_sha256=hashlib.sha256(a.native_lib.read_bytes()).hexdigest(),reference_quantizer_compiler_overrides={'tir.disable_vectorize':True},cases=results),indent=2)+'\n')


if __name__=='__main__':main()
