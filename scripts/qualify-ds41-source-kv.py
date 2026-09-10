#!/usr/bin/env python3
"""Validate owned real-source KV bytes against official rotary and K32 FP8 arithmetic."""
import argparse,ast,hashlib,importlib.util,json
from pathlib import Path
import torch
import tvm_ffi


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-dir',type=Path,required=True);p.add_argument('--vectors-dir',type=Path,required=True)
    p.add_argument('--device',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    lock=json.loads((root/'docs/ds41-reference-lock.json').read_text())
    for name in ('model.py','kernel.py'):
        assert hashlib.sha256((a.reference_dir/'inference'/name).read_bytes()).hexdigest()==lock['files']['inference/'+name]
    source=a.reference_dir/'inference/model.py'
    nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='apply_rotary_emb']
    ns={'torch':torch};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
    spec=importlib.util.spec_from_file_location('ds41_official_kernel',a.reference_dir/'inference/kernel.py')
    ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    assert ref.tilelang.__version__=='0.1.8'
    ref.act_quant_kernel.pass_configs={**ref.act_quant_kernel.pass_configs,'tir.disable_vectorize':True}
    torch.cuda.set_device(a.device);stream=torch.cuda.Stream();results=[]
    paths=sorted(a.vectors_dir.glob('c*-latent.bin'),key=lambda p:int(p.name.split('-')[0][1:]));assert paths
    with torch.cuda.stream(stream),tvm_ffi.use_torch_stream(),torch.no_grad():
        for path in paths:
            prefix=path.name.split('-')[0];payloads={name:(a.vectors_dir/f'{prefix}-{name}.bin').read_bytes() for name in ('latent','freq','kv','scale')}
            rows=len(payloads['latent'])//1024;assert 1<=rows<=4096 and len(payloads['latent'])==rows*1024
            def tensor(name,dtype,shape):return torch.frombuffer(bytearray(payloads[name]),dtype=dtype).reshape(shape).cuda()
            x=tensor('latent',torch.bfloat16,(1,rows,512));freq=tensor('freq',torch.float32,(rows,32,2))
            ns['apply_rotary_emb'](x[...,-64:],torch.view_as_complex(freq))
            q,s=ref.act_quant(x,32,'ue8m0',torch.float8_e8m0fnu)
            assert torch.equal(q.view(torch.uint8).reshape(rows,512),tensor('kv',torch.uint8,(rows,512))),prefix
            assert torch.equal(s.view(torch.uint8).reshape(rows,16),tensor('scale',torch.uint8,(rows,16))),prefix
            results.append(dict(case=prefix,rows=rows,exact=True,payloads_sha256={n:hashlib.sha256(b).hexdigest() for n,b in payloads.items()}))
            print(f'PASS {prefix} rows={rows} kv_bytes_exact=true',flush=True)
        stream.synchronize()
    a.output.write_text(json.dumps(dict(scope='Owned real-source latents encoded with fixed K32 FP8 serving policy, not reference compressed FP4',device=a.device,reference_quantizer_compiler_overrides={'tir.disable_vectorize':True},cases=results),indent=2)+'\n')


if __name__=='__main__':main()
