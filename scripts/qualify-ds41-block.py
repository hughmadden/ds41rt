#!/usr/bin/env python3
"""Backbone attention block sequencing with an explicit identity FFN fixture against pinned projection/norm/rotary/FP8 code."""
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
        for layer in (0,2,7,8,14,20,24,39):
            params={}
            for kind in ('attn','ffn'):
                params[kind]=[weight(f'layers.{layer}.hc_{kind}_{suffix}',torch.float32) for suffix in ('fn','scale','base')]
                gamma=weight(f'layers.{layer}.{kind}_norm.weight',torch.bfloat16)
                norm=ns['RMSNorm'](5120,1e-20).cuda();norm.weight.copy_(gamma);params[kind].append(norm)
            prefix=f'l{layer}/block';payloads={};t={};rows=80
            for name,dtype,shape in [('residual',torch.bfloat16,(1,rows,4,5120)),('incoming',torch.float32,(1,rows,4)),
                ('attn_input',torch.bfloat16,(1,rows,5120)),('attn_output',torch.bfloat16,(1,rows,5120)),
                ('ffn_input',torch.bfloat16,(1,rows,5120)),('ffn_residual',torch.bfloat16,(1,rows,4,5120)),('ffn_pre',torch.float32,(1,rows,4)),('output',torch.bfloat16,(1,rows,4,5120)),
                ('next_pre',torch.float32,(1,rows,4))]:
                t[name],payloads[name]=load(prefix,name,dtype,shape)
            afn,ascale,abase,anorm=params['attn'];ffn,fscale,fbase,fnorm=params['ffn']
            apre,apost,acomb=ns['hc_mixes'](obj,t['residual'],afn,ascale,abase)
            attn_input=anorm(ns['hc_pre'](obj,t['residual'],t['incoming']))
            residual=ns['hc_post'](obj,t['attn_output'],t['residual'],apost,acomb)
            chained_pre,_,_=ns['hc_mixes'](obj,residual,ffn,fscale,fbase)
            # Check the attention boundary and its actual handoff first. The
            # downstream coefficient tolerance is defined for identical input,
            # so also record (without hiding) the compounded BF16 input effect.
            torch.testing.assert_close(t['ffn_residual'],residual,rtol=.008,atol=.002)
            torch.testing.assert_close(t['ffn_pre'],apre,rtol=2e-5,atol=2e-5)
            pre,post,comb=ns['hc_mixes'](obj,t['ffn_residual'],ffn,fscale,fbase)
            ffn_input=fnorm(ns['hc_pre'](obj,t['ffn_residual'],t['ffn_pre']))
            output=ns['hc_post'](obj,t['ffn_input'],t['ffn_residual'],post,comb)
            for name,expected in [('attn_input',attn_input),('ffn_input',ffn_input),('output',output)]:
                torch.testing.assert_close(t[name],expected,rtol=.008,atol=.002)
            torch.testing.assert_close(t['next_pre'],pre,rtol=2e-5,atol=2e-5)
            results.append(dict(layer=layer,rows=rows,pre_max_abs=(t['next_pre']-pre).abs().max().item(),
                attn_input_max_abs=(t['attn_input'].float()-attn_input.float()).abs().max().item(),
                handoff_residual_max_abs=(t['ffn_residual'].float()-residual.float()).abs().max().item(),
                handoff_pre_max_abs=(t['ffn_pre']-apre).abs().max().item(),
                chained_pre_max_abs=(t['next_pre']-chained_pre).abs().max().item(),
                chained_pre_outside_stage_tolerance=int((~torch.isclose(t['next_pre'],chained_pre,rtol=2e-5,atol=2e-5)).sum().item()),
                ffn_input_max_abs=(t['ffn_input'].float()-ffn_input.float()).abs().max().item(),
                output_max_abs=(t['output'].float()-output.float()).abs().max().item(),identity_ffn_fixture=True,payloads_sha256=payloads))
            print(f'PASS layer={layer} rows=80 shifted_block_close=true identity_ffn_fixture=true',flush=True)
        stream.synchronize()
    a.output.write_text(json.dumps(dict(device=a.device,scope='Real mHC sequencing around produced attention output; FFN result is explicitly the identity fixture, not routed/shared MoE',
        reference_compiler_overrides={},weight_payloads_sha256=hashes,cases=results),indent=2)+'\n')


if __name__=='__main__':main()
