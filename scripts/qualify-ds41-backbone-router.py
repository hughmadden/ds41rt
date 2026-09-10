#!/usr/bin/env python3
"""Real backbone text/vision router checks against the pinned official Gate."""
import argparse, ast, hashlib, json, struct
from pathlib import Path
from types import SimpleNamespace
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference-dir','snapshot','vectors-dir','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',type=int,required=True)
    p.add_argument('--layers',type=int,nargs='+');p.add_argument('--capacities',type=int,nargs='+');a=p.parse_args()
    root=Path(__file__).resolve().parents[1];lock=json.loads((root/'docs/ds41-reference-lock.json').read_text())
    source=a.reference_dir/'inference/model.py';assert hashlib.sha256(source.read_bytes()).hexdigest()==lock['files']['inference/model.py']
    tree=ast.parse(source.read_text());gate=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Gate')
    forward=next(n for n in gate.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
    linear=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='linear')
    ns={'torch':torch,'F':torch.nn.functional};exec(compile(ast.Module(body=[linear,forward],type_ignores=[]),str(source),'exec'),ns)
    torch.cuda.set_device(a.device);torch.backends.cuda.matmul.allow_tf32=False
    index=json.loads((a.snapshot/'model.safetensors.index.json').read_text())['weight_map'];headers={};hashes={};results=[]
    def weight(name,dtype):
        path=a.snapshot/index[name]
        with path.open('rb') as f:
            if path not in headers:
                n=struct.unpack('<Q',f.read(8))[0];headers[path]=(n,json.loads(f.read(n)))
            n,h=headers[path];e=h[name];begin,end=e['data_offsets'];f.seek(8+n+begin);raw=f.read(end-begin)
        hashes[name]=hashlib.sha256(raw).hexdigest();return torch.frombuffer(bytearray(raw),dtype=dtype).reshape(e['shape']).cuda()
    for layer in (a.layers or range(40)):
        w=weight(f'layers.{layer}.ffn.gate.weight',torch.bfloat16)
        b=weight(f'layers.{layer}.ffn.gate.bias',torch.float32);v=weight(f'layers.{layer}.ffn.gate.bias_vl',torch.float32)
        obj=SimpleNamespace(weight=w,bias=b,bias_vl=v,gate_temp=1.,score_func='sqrtsoftplus',topk=6,norm_topk_prob=True,route_scale=1.5)
        for rows in (a.capacities or ([1,16,80,256,1024,4096] if layer in (0,20,39) else [80])):
            for case in range(2):
                prefix=f'l{layer}-m{rows}-c{case}';t={};payloads={}
                for name,dtype,shape in [('input',torch.bfloat16,(rows,5120)),('mask',torch.uint8,(rows,)),('scores',torch.float32,(rows,384)),('ids',torch.int32,(rows,6)),('routing',torch.float32,(rows,6))]:
                    raw=(a.vectors_dir/f'{prefix}-{name}.bin').read_bytes();payloads[name]=hashlib.sha256(raw).hexdigest()
                    t[name]=torch.frombuffer(bytearray(raw),dtype=dtype).reshape(shape).cuda()
                expected,ids=ns['forward'](obj,t['input'],t['mask'].bool())
                scores=torch.nn.functional.softplus(torch.nn.functional.linear(t['input'].float(),w.float())).sqrt()
                torch.testing.assert_close(t['scores'],scores,rtol=2e-5,atol=2e-5)
                assert torch.equal(t['ids'].long(),ids),f'{prefix}: selected experts differ'
                torch.testing.assert_close(t['routing'],expected,rtol=2e-5,atol=2e-5)
                results.append(dict(layer=layer,rows=rows,case=case,selected_experts_exact=True,
                    scores_max_abs=(t['scores']-scores).abs().max().item(),routing_max_abs=(t['routing']-expected).abs().max().item(),payloads_sha256=payloads))
                print(f'PASS {prefix} exact_ids=true scores_weights_close=true',flush=True)
    a.output.write_text(json.dumps(dict(device=a.device,scope='Real-weight backbone router component; excludes routed expert execution and full model',weight_payloads_sha256=hashes,cases=results),indent=2)+'\n')


if __name__=='__main__':main()
