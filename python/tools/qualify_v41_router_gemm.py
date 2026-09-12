#!/usr/bin/env python3
"""Qualify runtime-row CuTe BF16 routers on official weights before serving integration."""
import argparse
import ctypes as C
import hashlib
import json
import statistics
from pathlib import Path
import torch
import cutlass
import cutlass.cute as cute
from safetensors import safe_open
import _pinned_sparkinfer  # noqa: F401  # verifies and prepends the pinned fork
from b12x._lib.utils import make_ptr, current_cuda_stream
from b12x._lib.runtime_control import freeze_kernel_resolution
from b12x.moe._shared.v41_router import compile_v41_router_scores_aot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--native-lib', type=Path, required=True)
    p.add_argument('--serving-lib', type=Path, help='Qualify the native serving dispatch instead of the Python launch')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--prefixes', nargs='+')
    p.add_argument('--rows', nargs='+', type=int, default=[1, 6, 8, 16, 26, 80, 256, 1024, 2048, 4096, 6])
    a = p.parse_args()
    torch.cuda.set_device(a.device)
    torch.manual_seed(731)
    torch.backends.cuda.matmul.allow_tf32 = False
    lib = C.CDLL(str(a.native_lib.resolve()))
    old = lib.ds41rt_v41_router
    old.argtypes = [C.c_void_p]*8 + [C.c_int32, C.c_int32, C.c_void_p]
    select = lib.ds41rt_v41_router_select_logits
    select.argtypes = [C.c_void_p]*6 + [C.c_int32, C.c_int32, C.c_void_p]
    serving = None
    if a.serving_lib:
        serving_lib = C.CDLL(str(a.serving_lib.resolve()))
        initialize = serving_lib.ds41rt_v41_router_initialize
        initialize.restype = C.c_int32
        assert initialize() == 0
        assert initialize() == 0
        serving = serving_lib.ds41rt_v41_router
        serving.argtypes = old.argtypes
    compiled = {n: compile_v41_router_scores_aot(experts=n) for n in (128, 384)}
    # Compile both model geometries once. Subsequent live rows only change launch args.
    freeze_kernel_resolution('V4.1 router qualification')
    index = json.loads((a.snapshot/'model.safetensors.index.json').read_text())['weight_map']
    hashes = {}
    def load(name):
        with safe_open(a.snapshot/index[name], framework='pt', device='cpu') as f:
            value = f.get_tensor(name).contiguous()
        hashes[name] = hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()
        return value.cuda()
    prefixes = a.prefixes or [f'layers.{i}.ffn.gate' for i in range(40)] + [f'mtp.{i}.ffn.gate' for i in range(3)]
    records = []
    result = dict(scope='Component qualification; random BF16 activations, official gate weights; excludes full-model quality and API performance',
                  device=torch.cuda.get_device_name(), device_properties=str(torch.cuda.get_device_properties(a.device)),
                  native_sha256=hashlib.sha256(a.native_lib.read_bytes()).hexdigest(),
                  serving_sha256=hashlib.sha256(a.serving_lib.read_bytes()).hexdigest() if a.serving_lib else None,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  weight_sha256=hashes, cases=records)
    def save():
        a.output.write_text(json.dumps(result, indent=2)+'\n')
    for prefix in prefixes:
        w = load(prefix+'.weight'); b = load(prefix+'.bias')
        bv = load(prefix+'.bias_vl') if prefix+'.bias_vl' in index else b
        n, k = w.shape; topk = 6 if n == 384 else 3
        for rows in a.rows:
            assert 1 <= rows <= 4096
            # Exact-sized input tests TMA tail bounds; guarded output tests stores.
            x = torch.randn(rows,k,device='cuda',dtype=torch.bfloat16)
            mask = torch.randint(0,2,(rows,),device='cuda',dtype=torch.uint8)
            scores = torch.empty(rows,n,device='cuda')
            guarded = torch.full((rows*n+128,),12345.,device='cuda')
            candidate_scores = guarded[64:-64].view(rows,n)
            ids = torch.empty(rows,topk,device='cuda',dtype=torch.int32)
            candidate_ids = torch.empty_like(ids)
            routing = torch.empty(rows,topk,device='cuda'); candidate_routing = torch.empty_like(routing)
            ptr = lambda t,d: make_ptr(d,t.data_ptr(),cute.AddressSpace.gmem,assumed_align=16)
            xp,wp,zp = ptr(x,cutlass.BFloat16),ptr(w,cutlass.BFloat16),ptr(candidate_scores,cutlass.Float32)
            def baseline():
                assert old(*[t.data_ptr() for t in (x,w,b,bv,mask,scores,ids,routing)],rows,n,torch.cuda.current_stream().cuda_stream)==0
            def candidate():
                if serving is not None:
                    assert serving(*[t.data_ptr() for t in (x,w,b,bv,mask,candidate_scores,candidate_ids,candidate_routing)],rows,n,torch.cuda.current_stream().cuda_stream)==0
                    return
                compiled[n](xp,wp,zp,cutlass.Int32(rows),current_cuda_stream())
                assert select(*[t.data_ptr() for t in (candidate_scores,b,bv,mask,candidate_ids,candidate_routing)],rows,n,torch.cuda.current_stream().cuda_stream)==0
            baseline(); candidate(); torch.cuda.synchronize()
            graphs=[]
            for fn in (baseline,candidate):
                g=torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    for _ in range(10): fn()
                graphs.append(g)
            record=dict(prefix=prefix,rows=rows,experts=n,mutations=[])
            records.append(record)
            for mutation in range(2):
                if mutation:
                    x.normal_(std=0.25); mask.bitwise_xor_(1)
                candidate_scores.fill_(float('nan')); candidate_ids.fill_(-1); candidate_routing.fill_(float('nan'))
                before=torch.cuda.memory_allocated()
                for g in graphs: g.replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_allocated()==before
                assert bool((guarded[:64]==12345).all()) and bool((guarded[-64:]==12345).all())
                ref=torch.nn.functional.softplus(x.float()@w.float().T).sqrt()
                correction=torch.where(mask.bool()[:,None],bv[None,:],b[None,:])
                # Stable sort implements the native lower-ID tie break.
                ref_ids=torch.argsort(ref+correction,descending=True,stable=True)[:,:topk]
                ref_routing=ref.gather(1,ref_ids)
                ref_routing=ref_routing/(ref_routing.sum(-1,keepdim=True)+1e-20)*1.5
                score_close=torch.allclose(candidate_scores,ref,rtol=2e-5,atol=2e-5)
                ids_exact=torch.equal(candidate_ids.long(),ref_ids)
                route_close=torch.allclose(candidate_routing,ref_routing,rtol=2e-5,atol=2e-5)
                candidate_order=torch.argsort(candidate_ids,dim=1)
                reference_order=torch.argsort(ref_ids,dim=1)
                same_set=torch.equal(candidate_ids.gather(1,candidate_order).long(),ref_ids.gather(1,reference_order))
                weights_by_id_close=same_set and torch.allclose(
                    candidate_routing.gather(1,candidate_order),ref_routing.gather(1,reference_order),rtol=2e-5,atol=2e-5)
                record['mutations'].append(dict(mutation=mutation,score_close=score_close,
                    ids_exact_oracle=ids_exact,ids_exact_baseline=torch.equal(candidate_ids,ids),
                    expert_set_exact=same_set,weights_by_id_close=weights_by_id_close,
                    baseline_ids_exact_oracle=torch.equal(ids.long(),ref_ids),route_close=route_close,
                    max_abs_score=(candidate_scores-ref).abs().max().item(),
                    oracle_id_differences=int((candidate_ids!=ref_ids).sum()),
                    baseline_id_differences=int((candidate_ids!=ids).sum())))
                if serving is not None:
                    from b12x.moe._shared.v41_router import v41_router_gemm_min_rows
                    if rows >= v41_router_gemm_min_rows(experts=n):
                        py_scores=torch.empty_like(candidate_scores)
                        py_ids=torch.empty_like(candidate_ids)
                        py_routing=torch.empty_like(candidate_routing)
                        compiled[n](xp,wp,ptr(py_scores,cutlass.Float32),cutlass.Int32(rows),current_cuda_stream())
                        assert select(*[t.data_ptr() for t in (py_scores,b,bv,mask,py_ids,py_routing)],rows,n,torch.cuda.current_stream().cuda_stream)==0
                        assert torch.equal(candidate_scores,py_scores)
                        assert torch.equal(candidate_ids,py_ids)
                        assert torch.equal(candidate_routing,py_routing)
                    else:
                        assert torch.equal(candidate_scores,scores)
                        assert torch.equal(candidate_ids,ids)
                        assert torch.equal(candidate_routing,routing)
                    record['mutations'][-1]['native_dispatch_exact']=True
                if not ids_exact:
                    changed=(candidate_ids!=ref_ids).any(dim=1).nonzero().flatten()
                    precise=torch.nn.functional.softplus(x[changed].double()@w.double().T).sqrt()
                    corrected=precise+correction[changed].double()
                    ordered=torch.argsort(corrected,descending=True,stable=True)
                    record['mutations'][-1]['changed_rows']=[dict(
                        row=int(row),baseline=ids[row].tolist(),candidate=candidate_ids[row].tolist(),
                        fp32_oracle=ref_ids[row].tolist(),fp64_oracle=ordered[j,:topk].tolist(),
                        same_expert_set=bool(torch.equal(ids[row].sort().values,candidate_ids[row].sort().values)),
                        fp64_selection_margin=float(corrected[j,ordered[j,topk-1]]-corrected[j,ordered[j,topk]]),
                        baseline_fp64_max_abs=float((scores[row].double()-precise[j]).abs().max()),
                        candidate_fp64_max_abs=float((candidate_scores[row].double()-precise[j]).abs().max()),
                    ) for j,row in enumerate(changed)]
                save()
            times=[[],[]]
            for repeat in range(6):
                for arm in ([0,1] if repeat%2 else [1,0]):
                    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    begin.record();graphs[arm].replay();end.record();end.synchronize()
                    times[arm].append(begin.elapsed_time(end)*100)
            record.update(samples_us=times,baseline_us=statistics.median(times[0]),candidate_us=statistics.median(times[1]))
            save()
            print(json.dumps({k:v for k,v in record.items() if k!='samples_us'}),flush=True)
    # Expert order is diagnostic: the FFN sums (expert output * weight).
    # Require exactly the same expert set and compare each weight by expert ID.
    # This does not permit replacing a nearly tied boundary expert.
    result['passed']=all(m['score_close'] and m['expert_set_exact'] and m['weights_by_id_close'] for r in records for m in r['mutations'])
    save()
    if not result['passed']:
        raise SystemExit('Router oracle gate failed; inspect recorded mismatches before rollout')


if __name__=='__main__':
    main()
