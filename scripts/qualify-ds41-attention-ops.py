#!/usr/bin/env python3
"""Pinned-reference native dSpark normalization/rotation checks; no checkpoint."""
import argparse
import ast
import ctypes as C
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--native-lib', type=Path, required=True)
    parser.add_argument('--reference-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', type=int, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    lock = json.loads((root/'docs/ds41-reference-lock.json').read_text())
    source = args.reference_dir/'inference/model.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == lock['files']['inference/model.py']
    names = {'RMSNorm', 'apply_rotary_emb', 'precompute_freqs_cis'}
    nodes = [n for n in ast.parse(source.read_text()).body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert len(nodes) == 3
    ns = {'torch': torch, 'nn': torch.nn, 'math': math, 'lru_cache': lru_cache}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), ns)
    lib = C.CDLL(str(args.native_lib))
    norm, rope = lib.ds41rt_v41_attention_norm, lib.ds41rt_v41_attention_rope
    P, I = C.c_void_p, C.c_int32
    norm.argtypes = [P,P,P,P,I,I,P]
    rope.argtypes = [P,P,P,I,I,I,P]
    kv=lib.ds41rt_v41_attention_kv
    kv.argtypes=[P,P,P,P,I,P]
    norm.restype = rope.restype = kv.restype = I
    torch.cuda.set_device(args.device)
    torch.manual_seed(41091 + args.device)
    stream = torch.cuda.Stream()
    results = []
    with torch.cuda.stream(stream), torch.no_grad():
        # Per-row arbitrary positions from the official sliding-window construction, including
        # nonconsecutive/repeated positions across independently scheduled requests.
        freq_table = ns['precompute_freqs_cis'](64, 262144, 0, 10000, 40, 32, 1).cuda()
        def frequencies(rows):
            ids = torch.randint(0, len(freq_table), (rows,), device='cuda')
            return torch.view_as_real(freq_table[ids]).contiguous()
        def apply(x, f, inverse=False):
            result=x.clone()
            ns['apply_rotary_emb'](result[None, ..., -64:], torch.view_as_complex(f), inverse)
            return result
        def compare(a,b):
            torch.testing.assert_close(a,b,rtol=.008,atol=.002)
            return {'max_abs_error': (a.float()-b.float()).abs().max().item(),
                    'different_bf16_elements': (a!=b).sum().item()}
        def check(status): assert status == 0, status
        for rows in (1,16,80,255,1023,4095):
            f=frequencies(rows)
            for dim in (512,1280,5120):
                x=(torch.randn((rows,dim),device='cuda')*.3).bfloat16()
                w=(torch.randn(dim,device='cuda')*.2+1).bfloat16()
                layer=ns['RMSNorm'](dim).cuda().bfloat16()
                layer.weight.copy_(w)
                y=torch.empty_like(x)
                for rotated in ([False,True] if dim==512 else [False]):
                    def launch(): check(norm(x.data_ptr(),w.data_ptr(),f.data_ptr() if rotated else None,y.data_ptr(),rows,dim,stream.cuda_stream))
                    launch()
                    expected=layer(x)
                    if rotated: expected=apply(expected,f)
                    initial=compare(y,expected)
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,stream=stream): launch()
                    x.copy_((torch.randn_like(x.float())*.7).bfloat16())
                    f.copy_(frequencies(rows))
                    graph.replay()
                    expected=layer(x)
                    if rotated: expected=apply(expected,f)
                    changed=compare(y,expected)
                    x.zero_();graph.replay();assert torch.count_nonzero(y).item()==0
                    x.fill_(1e-8);graph.replay()
                    expected=layer(x)
                    if rotated: expected=apply(expected,f)
                    tiny=compare(y,expected)
                    assert norm(x.data_ptr(),w.data_ptr(),None,x.data_ptr(),rows,dim,stream.cuda_stream)!=0
                    assert norm(x.data_ptr(),w.data_ptr(),None,y.data_ptr(),0,dim,stream.cuda_stream)!=0
                    assert norm(x.data_ptr(),w.data_ptr(),None,y.data_ptr(),rows,513,stream.cuda_stream)!=0
                    assert norm(x.data_ptr(),w.data_ptr(),f.data_ptr(),y.data_ptr(),rows,1280,stream.cuda_stream)!=0
                    results.append({'op':'norm_rope' if rotated else 'norm','rows':rows,'dim':dim,'initial':initial,'changed_graph':changed,'tiny':tiny,'zero_exact':True,'guards':True})
                    if rotated:
                        z=torch.empty_like(x)
                        def kv_launch(): check(kv(x.data_ptr(),w.data_ptr(),f.data_ptr(),z.data_ptr(),rows,stream.cuda_stream))
                        def quantized():
                            groups=y.float().reshape(rows,16,32)
                            amax=groups.abs().amax(-1).clamp_min(1e-4)
                            scales=torch.exp2(torch.ceil(torch.log2(amax/448)))
                            return ((groups/scales[...,None]).to(torch.float8_e4m3fn).float()*scales[...,None]).reshape(rows,512).bfloat16()
                        kv_launch();torch.testing.assert_close(z,quantized(),rtol=0,atol=0)
                        kv_graph=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(kv_graph,stream=stream): kv_launch()
                        x.copy_((torch.randn_like(x.float())*.9).bfloat16());f.copy_(frequencies(rows))
                        graph.replay();kv_graph.replay()
                        torch.testing.assert_close(z,quantized(),rtol=0,atol=0)
                        x.zero_();kv_graph.replay();assert torch.count_nonzero(z).item()==0
                        assert kv(x.data_ptr(),w.data_ptr(),f.data_ptr(),x.data_ptr(),rows,stream.cuda_stream)!=0
                        results.append({'op':'fused_kv','rows':rows,'tiny_exact':True,'changed_graph_exact':True,'zero_exact':True,'overlap_guard':True})
                        del kv_graph
                    del graph
            for heads in (1,64):
                x=torch.randn((rows,heads,512),device='cuda').bfloat16()
                y=torch.empty_like(x)
                for inverse in (False,True):
                    def launch(): check(rope(x.data_ptr(),f.data_ptr(),y.data_ptr(),rows,heads,int(inverse),stream.cuda_stream))
                    launch();expected=apply(x,f,inverse)
                    # No reduction: complex multiply follows the reference's FP32 operations.
                    torch.testing.assert_close(y,expected,rtol=0,atol=0)
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,stream=stream): launch()
                    x.copy_(torch.randn_like(x.float()).bfloat16()); f.copy_(frequencies(rows));graph.replay()
                    torch.testing.assert_close(y,apply(x,f,inverse),rtol=0,atol=0)
                    assert rope(x.data_ptr(),f.data_ptr(),x.data_ptr(),rows,heads,0,stream.cuda_stream)!=0
                    assert rope(x.data_ptr(),f.data_ptr(),y.data_ptr(),rows,2,0,stream.cuda_stream)!=0
                    assert rope(x.data_ptr(),f.data_ptr(),y.data_ptr(),rows,heads,2,stream.cuda_stream)!=0
                    results.append({'op':'rope','rows':rows,'heads':heads,'inverse':inverse,'initial_exact':True,'changed_graph_exact':True,'guards':True})
                    del graph
            print(f'PASS rows={rows}', flush=True)
        stream.synchronize()
    evidence={'scope':'Native dSpark norm and rotary operations; excludes frequency ownership, full attention and serving',
              'device':args.device,'device_name':torch.cuda.get_device_name(args.device),
              'reference_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'native_library_sha256':hashlib.sha256(args.native_lib.read_bytes()).hexdigest(),
              'qualifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'results':results}
    args.output.write_text(json.dumps(evidence,indent=2)+'\n')

if __name__=='__main__': main()
