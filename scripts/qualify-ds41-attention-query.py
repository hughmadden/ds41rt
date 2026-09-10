#!/usr/bin/env python3
"""Real-weight backbone low-rank attention queries against pinned projection/norm/rotary/FP8 code."""
import argparse,ast,hashlib,importlib.util,json,math,struct
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
    ns={'torch':torch,'nn':torch.nn,'math':math,'lru_cache':lru_cache};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
    spec=importlib.util.spec_from_file_location('ds41_official_kernel',a.reference_dir/'inference/kernel.py')
    ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref);assert ref.tilelang.__version__=='0.1.8'
    ref.act_quant_kernel.pass_configs={**ref.act_quant_kernel.pass_configs,'tir.disable_vectorize':True}
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
        plain=ns['precompute_freqs_cis'](64,1048576,0,10000,16,32,1)
        compressed=ns['precompute_freqs_cis'](64,1048576,65536,160000,16,32,1)
        for layer in range(40):
            wa=weight(f'layers.{layer}.attn.wq_a.weight',torch.float8_e4m3fn)
            sa=weight(f'layers.{layer}.attn.wq_a.scale',torch.float8_e8m0fnu)
            wb=weight(f'layers.{layer}.attn.wq_b.weight',torch.float8_e4m3fn)
            sb=weight(f'layers.{layer}.attn.wq_b.scale',torch.float8_e8m0fnu)
            gamma=weight(f'layers.{layer}.attn.q_norm.weight',torch.bfloat16)
            norm=ns['RMSNorm'](1280,1e-20).cuda();norm.weight.copy_(gamma)
            for rows in ((1,16,80,256,1024,4096) if layer in (0,1,20,39) else (80,)):
                for case in (0,1):
                    prefix=f'l{layer}-m{rows}-c{case}';payloads={};tensors={}
                    for name,dtype,shape in [('input',torch.bfloat16,(rows,5120)),('positions',torch.int64,(rows,)),
                        ('raw_rank',torch.bfloat16,(rows,1280)),('normalized_rank',torch.bfloat16,(rows,1280)),
                        ('projected',torch.bfloat16,(rows,64,512)),('rotated',torch.bfloat16,(rows,64,512)),
                        ('freq',torch.float32,(rows,32,2))]:
                        tensors[name],payloads[name]=load(prefix,name,dtype,shape)
                    aq,scale=ref.act_quant(tensors['input'],32,'ue8m0',torch.float8_e8m0fnu)
                    rank=ref.fp8_gemm(aq,scale,wa,sa,torch.float8_e8m0fnu,block_size=32)
                    torch.testing.assert_close(tensors['raw_rank'],rank,rtol=.008,atol=.002)
                    normalized=norm(tensors['raw_rank'])
                    torch.testing.assert_close(tensors['normalized_rank'],normalized,rtol=.008,atol=.002)
                    bq,bs=ref.act_quant(tensors['normalized_rank'],32,'ue8m0',torch.float8_e8m0fnu)
                    projected=ref.fp8_gemm(bq,bs,wb,sb,torch.float8_e8m0fnu,block_size=32).reshape(rows,64,512)
                    torch.testing.assert_close(tensors['projected'],projected,rtol=.008,atol=.002)
                    freq=(plain if layer<2 else compressed)[tensors['positions']]
                    torch.testing.assert_close(tensors['freq'],torch.view_as_real(freq),rtol=0,atol=0)
                    rotated=tensors['projected'].clone();ns['apply_rotary_emb'](rotated[None,...,-64:],freq)
                    torch.testing.assert_close(tensors['rotated'],rotated,rtol=0,atol=0)
                    results.append(dict(layer=layer,rows=rows,case=case,
                        rank_max_abs=(tensors['raw_rank'].float()-rank.float()).abs().max().item(),
                        norm_max_abs=(tensors['normalized_rank'].float()-normalized.float()).abs().max().item(),
                        projection_max_abs=(tensors['projected'].float()-projected.float()).abs().max().item(),
                        frequencies_exact=True,rotary_exact=True,payloads_sha256=payloads))
                    print(f'PASS {prefix} projections_norm_close=true rotary_exact=true',flush=True)
        stream.synchronize()
    a.output.write_text(json.dumps(dict(device=a.device,scope='Real backbone query projection stages compared on native BF16 intermediate inputs; rotary exact from native projected queries; end-to-end full-model qualification remains open',
        reference_quantizer_compiler_overrides={'tir.disable_vectorize':True},weight_payloads_sha256=hashes,cases=results),indent=2)+'\n')


if __name__=='__main__':main()
