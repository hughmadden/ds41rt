#!/usr/bin/env python3
"""Real-weight shifted backbone mHC boundaries against pinned projection/norm/rotary/FP8 code."""
import argparse,ast,hashlib,importlib.util,json,math,struct
from types import SimpleNamespace
from functools import lru_cache
from pathlib import Path
import torch
import tvm_ffi


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-dir',type=Path,required=True);p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--vectors-dir',type=Path,required=True);p.add_argument('--device',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];lock=json.loads((root/'docs/ds41-reference-lock.json').read_text())
    for name in ('model.py','kernel.py'):
        assert hashlib.sha256((a.reference_dir/'inference'/name).read_bytes()).hexdigest()==lock['files']['inference/'+name]
    source=a.reference_dir/'inference/model.py';names={'RMSNorm','apply_rotary_emb','precompute_freqs_cis'}
    nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    ns={'torch':torch,'nn':torch.nn,'F':torch.nn.functional,'math':math,'lru_cache':lru_cache};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
    spec=importlib.util.spec_from_file_location('ds41_official_kernel',a.reference_dir/'inference/kernel.py')
    ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref);assert ref.tilelang.__version__=='0.1.8'
    block=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='Block')
    for method in ('hc_mixes','hc_pre','hc_post'):
        node=next(n for n in block.body if isinstance(n,ast.FunctionDef) and n.name==method)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
    ns['hc_split_sinkhorn']=ref.hc_split_sinkhorn
    obj=SimpleNamespace(norm_eps=1e-20,hc_eps=1e-6,hc_mult=4,hc_sinkhorn_iters=20)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_device(a.device);torch.set_default_dtype(torch.bfloat16)
    stream=torch.cuda.Stream();index=json.loads((a.snapshot/'model.safetensors.index.json').read_text())['weight_map'];headers={};hashes={};results=[]
    def weight(name,dtype):
        path=a.snapshot/index[name]
        with path.open('rb') as f:
            if path not in headers:
                n=struct.unpack('<Q',f.read(8))[0];headers[path]=(n,json.loads(f.read(n)))
            n,h=headers[path];entry=h[name];begin,end=entry['data_offsets'];f.seek(8+n+begin);raw=f.read(end-begin)
        hashes[name]=hashlib.sha256(raw).hexdigest();return torch.frombuffer(bytearray(raw),dtype=dtype).reshape(entry['shape']).cuda()
    def load(prefix,name,dtype,shape):
        raw=(a.vectors_dir/f'{prefix}-{name}.bin').read_bytes()
        return torch.frombuffer(bytearray(raw),dtype=dtype).reshape(shape).cuda(),hashlib.sha256(raw).hexdigest()
    with torch.device('cuda'),torch.cuda.stream(stream),tvm_ffi.use_torch_stream(),torch.no_grad():
        for layer in range(40):
            for kind in ('attn','ffn'):
                fn=weight(f'layers.{layer}.hc_{kind}_fn',torch.float32)
                scale=weight(f'layers.{layer}.hc_{kind}_scale',torch.float32)
                base=weight(f'layers.{layer}.hc_{kind}_base',torch.float32)
                gamma=weight(f'layers.{layer}.{kind}_norm.weight',torch.bfloat16)
                norm=ns['RMSNorm'](5120,1e-20).cuda();norm.weight.copy_(gamma)
                for rows in ((1,16,80,4096) if layer in (0,20,39) else (80,)):
                    for case in (0,1):
                        prefix=f'l{layer}-{kind}-m{rows}-c{case}';payloads={};t={}
                        for name,dtype,shape in [('residual',torch.bfloat16,(1,rows,4,5120)),('incoming',torch.float32,(1,rows,4)),
                            ('sublayer',torch.bfloat16,(1,rows,5120)),('normalized',torch.bfloat16,(1,rows,5120)),
                            ('output',torch.bfloat16,(1,rows,4,5120)),('next_pre',torch.float32,(1,rows,4))]:
                            t[name],payloads[name]=load(prefix,name,dtype,shape)
                        pre,post,comb=ns['hc_mixes'](obj,t['residual'],fn,scale,base)
                        normalized=norm(ns['hc_pre'](obj,t['residual'],t['incoming']))
                        out=ns['hc_post'](obj,t['sublayer'],t['residual'],post,comb)
                        torch.testing.assert_close(t['next_pre'],pre,rtol=2e-5,atol=2e-5)
                        torch.testing.assert_close(t['normalized'],normalized,rtol=.008,atol=.002)
                        torch.testing.assert_close(t['output'],out,rtol=.008,atol=.002)
                        results.append(dict(layer=layer,kind=kind,rows=rows,case=case,
                            pre_max_abs=(t['next_pre']-pre).abs().max().item(),
                            norm_max_abs=(t['normalized'].float()-normalized.float()).abs().max().item(),
                            output_max_abs=(t['output'].float()-out.float()).abs().max().item(),payloads_sha256=payloads))
                        print(f'PASS {prefix} coefficients_norm_post_close=true',flush=True)
        stream.synchronize()
    a.output.write_text(json.dumps(dict(device=a.device,scope='Real backbone mHC coefficient, shifted collapse/norm and residual expansion against actual reference methods; attention/FFN results are fixture inputs',
        reference_compiler_overrides={},weight_payloads_sha256=hashes,cases=results),indent=2)+'\n')


if __name__=='__main__':main()
