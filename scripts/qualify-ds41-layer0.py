#!/usr/bin/env python3
"""Complete layer-0 numerical reference from token IDs to final residual and pre-mix."""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
import struct
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import torch
import tvm_ffi


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('reference-dir', 'snapshot', 'inputs', 'rank-dir', 'final-dir', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', type=int, required=True)
    a = p.parse_args()
    lock = json.loads((Path(__file__).resolve().parents[1] / 'docs/ds41-reference-lock.json').read_text())
    for name in ('model.py', 'kernel.py'):
        assert hashlib.sha256((a.reference_dir/'inference'/name).read_bytes()).hexdigest() == lock['files']['inference/'+name]
    source = a.reference_dir/'inference/model.py'
    tree = ast.parse(source.read_text())
    names = {'RMSNorm', 'apply_rotary_emb', 'precompute_freqs_cis', 'get_window_topk_idxs', 'make_identity_pre_mix', 'linear'}
    ns = dict(torch=torch, nn=torch.nn, F=torch.nn.functional, math=math, lru_cache=lru_cache)
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), ns)
    block = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Block')
    for name in ('hc_mixes', 'hc_pre', 'hc_post'):
        node = next(n for n in block.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), ns)
    gate = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Gate')
    node = next(n for n in gate.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), ns)
    gate_forward = ns['forward']
    expert_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Expert')
    node = next(n for n in expert_class.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), ns)
    expert_forward = ns['forward']
    moe_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MoE')
    node = next(n for n in moe_class.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), ns)
    moe_forward = ns['forward']; ns['world_size'] = 1
    spec = importlib.util.spec_from_file_location('ds41_official_kernel', a.reference_dir/'inference/kernel.py')
    ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)
    assert ref.tilelang.__version__ == '0.1.8'
    ref.act_quant_kernel.pass_configs = {**ref.act_quant_kernel.pass_configs, 'tir.disable_vectorize': True}
    ns['hc_split_sinkhorn'] = ref.hc_split_sinkhorn
    obj = SimpleNamespace(norm_eps=1e-20, hc_eps=1e-6, hc_mult=4, hc_sinkhorn_iters=20)
    torch.cuda.set_device(a.device); torch.set_default_dtype(torch.bfloat16)
    torch.backends.cuda.matmul.allow_tf32 = False
    index = json.loads((a.snapshot/'model.safetensors.index.json').read_text())['weight_map']
    headers = {}; hashes = {}; payloads = {}; results = []
    @lru_cache(None)
    def weight(name, dtype):
        path = a.snapshot/index[name]
        with path.open('rb') as f:
            if path not in headers:
                n = struct.unpack('<Q', f.read(8))[0]; headers[path] = (n, json.loads(f.read(n)))
            n, header = headers[path]; entry = header[name]; begin, end = entry['data_offsets']
            f.seek(8+n+begin); raw = f.read(end-begin)
        hashes[name] = hashlib.sha256(raw).hexdigest()
        return torch.frombuffer(bytearray(raw), dtype=dtype).reshape(entry['shape']).cuda()
    def fp8(x, name):
        q, s = ref.act_quant(x.contiguous(), 32, 'ue8m0', torch.float8_e8m0fnu)
        return ref.fp8_gemm(q, s, weight(name+'.weight', torch.float8_e4m3fn), weight(name+'.scale', torch.float8_e8m0fnu), torch.float8_e8m0fnu, block_size=32)
    def norm(x, name):
        module = ns['RMSNorm'](x.shape[-1], 1e-20).cuda()
        module.weight.copy_(weight(name, torch.bfloat16)); return module(x)
    def fp4(x, name):
        q, scales = ref.act_quant(x.contiguous(),32,'ue8m0',torch.float8_e8m0fnu)
        return ref.fp4_gemm(q,scales,weight(name+'.weight',torch.float4_e2m1fn_x2),weight(name+'.scale',torch.float8_e8m0fnu),torch.float8_e8m0fnu,act_block_size=32)
    def expert(prefix, linear):
        return SimpleNamespace(w1=lambda x:linear(x,prefix+'.w1'),w3=lambda x:linear(x,prefix+'.w3'),w2=lambda x:linear(x,prefix+'.w2'),swiglu_limit=10.)
    def metrics(actual, expected):
        x,y=actual.float(),expected.float()
        assert torch.isfinite(x).all() and torch.isfinite(y).all()
        relative=((x-y).norm()/y.norm()).item()
        cosine=torch.nn.functional.cosine_similarity(x.flatten(),y.flatten(),dim=0).item()
        assert relative < .01 and cosine > .9999, (relative,cosine)
        return dict(relative_l2=relative,cosine=cosine,max_abs=(x-y).abs().max().item())
    stream = torch.cuda.Stream()
    with torch.device('cuda'), torch.cuda.stream(stream), tvm_ffi.use_torch_stream(), torch.no_grad():
        frequencies = ns['precompute_freqs_cis'](64, 5, 0, 10000, 16, 32, 1)
        table = weight('embed.weight', torch.bfloat16)
        mixing = [weight('layers.0.hc_attn_'+suffix, torch.float32) for suffix in ('fn', 'scale', 'base')]
        wa = weight('layers.0.attn.wo_a.weight', torch.float8_e4m3fn)
        sa = weight('layers.0.attn.wo_a.scale', torch.float8_e8m0fnu)
        grouped_weight = (wa.float()*sa.float().repeat_interleave(32,0).repeat_interleave(32,1)).bfloat16().reshape(8,1024,4096)
        gate = SimpleNamespace(weight=weight('layers.0.ffn.gate.weight', torch.bfloat16), bias=weight('layers.0.ffn.gate.bias', torch.float32), bias_vl=None, gate_temp=1., score_func='sqrtsoftplus', topk=6, norm_topk_prob=True, route_scale=1.5)
        for case in json.loads(a.inputs.read_text()):
            number = case['case']; tokens = torch.tensor(case['tokens'], dtype=torch.long).reshape(16,5)
            assert case['positions'] == list(range(5))*16
            raw = (a.rank_dir/f'rank0/l0-c{number}-request.bin').read_bytes()
            payloads[f'case{number}-request'] = hashlib.sha256(raw).hexdigest()
            assert raw[:8] == b'DS41RTE3' and struct.unpack_from('<I',raw,12)[0] == 96
            assert struct.unpack_from('<III',raw,32) == (0,80,5120)
            assert len(raw) == 96+3200+5760+819200
            for rank in range(1,4): assert (a.rank_dir/f'rank{rank}/l0-c{number}-request.bin').read_bytes() == raw
            actual = torch.frombuffer(bytearray(raw[-819200:]), dtype=torch.bfloat16).reshape(16,5,5120).cuda()
            residual = table[tokens].unsqueeze(2).expand(-1,-1,4,-1).contiguous()
            incoming = ns['make_identity_pre_mix'](residual,4)
            pre, post, comb = ns['hc_mixes'](obj,residual,*mixing)
            x = norm(ns['hc_pre'](obj,residual,incoming),'layers.0.attn_norm.weight')
            qr = norm(fp8(x,'layers.0.attn.wq_a'),'layers.0.attn.q_norm.weight')
            q = fp8(qr,'layers.0.attn.wq_b').reshape(16,5,64,512)
            ns['apply_rotary_emb'](q[..., -64:],frequencies)
            kv = norm(fp8(x,'layers.0.attn.wkv'),'layers.0.attn.kv_norm.weight')
            ns['apply_rotary_emb'](kv[..., -64:],frequencies)
            ref.act_quant(kv,32,'ue8m0',torch.float8_e8m0fnu,True)
            # Heads are independent. The reference's 64-head shared allocation
            # exceeds SM120 limits; run its unchanged kernel in 16-head groups.
            sink = weight('layers.0.attn.attn_sink',torch.float32)
            indices = ns['get_window_topk_idxs'](128,16,5,0)
            attention = torch.cat([ref.sparse_attn(q[:,:,h:h+16].contiguous(),kv,sink[h:h+16].contiguous(),indices,512**-.5) for h in range(0,64,16)],dim=2)
            ns['apply_rotary_emb'](attention[..., -64:],frequencies,True)
            grouped = torch.einsum('bsgd,grd->bsgr',attention.reshape(16,5,8,4096),grouped_weight)
            projected = fp8(grouped.flatten(2),'layers.0.attn.wo_b')
            after_attention = ns['hc_post'](obj,projected,residual,post,comb)
            expected = norm(ns['hc_pre'](obj,after_attention,pre),'layers.0.ffn_norm.weight')
            assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
            relative = ((actual.float()-expected.float()).norm()/expected.float().norm()).item()
            cosine = torch.nn.functional.cosine_similarity(actual.float().flatten(),expected.float().flatten(),dim=0).item()
            assert relative < .01 and cosine > .9999, (relative,cosine)
            route_weights, ids = gate_forward(gate,actual.reshape(80,5120))
            entries = [struct.unpack_from('<IIf',raw,96+3200+i*12) for i in range(480)]
            assert all(e[0] == i//6 for i,e in enumerate(entries))
            wire_ids = torch.tensor([e[1] for e in entries]).reshape(80,6)
            wire_weights = torch.tensor([e[2] for e in entries],dtype=torch.float32).reshape(80,6)
            assert torch.equal(ids,wire_ids), 'recorded expert selections differ from official Gate on actual FFN input'
            torch.testing.assert_close(route_weights,wire_weights,rtol=2e-5,atol=2e-5)
            result = dict(case=number,upstream_relative_l2=relative,upstream_cosine=cosine,upstream_max_abs=(actual.float()-expected.float()).abs().max().item(),router_ids_exact=True,router_weights_max_abs=(route_weights-wire_weights).abs().max().item())
            reference_route_weights, reference_ids = gate_forward(gate,expected.reshape(80,5120))
            active = sorted(set(reference_ids.flatten().tolist()))
            experts = [None]*384
            for e in active:
                implementation = expert(f'layers.0.ffn.experts.{e}',fp4)
                experts[e] = lambda x, w, implementation=implementation: expert_forward(implementation,x,w)
            shared = expert('layers.0.ffn.shared_experts',fp8)
            moe = SimpleNamespace(dim=5120,gate=lambda x,mask:gate_forward(gate,x,mask),n_routed_experts=384,experts_start_idx=0,experts_end_idx=384,experts=experts,shared_experts=lambda x:expert_forward(shared,x))
            # This path consumes only reference-produced hidden rows and routes.
            reference_ffn = moe_forward(moe,expected,None)
            ffn_mixing = [weight('layers.0.hc_ffn_'+suffix,torch.float32) for suffix in ('fn','scale','base')]
            next_pre, fpost, fcomb = ns['hc_mixes'](obj,after_attention,*ffn_mixing)
            final = ns['hc_post'](obj,reference_ffn,after_attention,fpost,fcomb)
            for label,dtype,shape,target in [('residual',torch.bfloat16,(16,5,4,5120),final),('pre',torch.float32,(16,5,4),next_pre)]:
                path=a.final_dir/f'layer0-c{number}-{label}.bin'; raw_output=path.read_bytes()
                payloads[f'case{number}-{label}']=hashlib.sha256(raw_output).hexdigest()
                native=torch.frombuffer(bytearray(raw_output),dtype=dtype).reshape(shape).cuda()
                result['final_'+label]=metrics(native,target)
            result['reference_active_experts']=len(active)
            result['upstream_rounding_changed_route_slots']=int((reference_ids!=wire_ids).sum().item())
            results.append(result); print('PASS',result,flush=True)
        stream.synchronize()
    a.output.write_text(json.dumps(dict(scope='Complete layer0 from token embeddings through attention, routed/shared FFN and final mHC using reference-produced hidden rows and routing throughout; additional official Gate check on native wire hidden rows',reference_attention_head_group=16,reference_quantizer_compiler_overrides={'tir.disable_vectorize':True},cases=results,weight_payloads_sha256=hashes,payloads_sha256=payloads,inputs_sha256=hashlib.sha256(a.inputs.read_bytes()).hexdigest(),qualifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')

if __name__ == '__main__': main()
