#!/usr/bin/env python3
"""Probe dense AOT tile/split choices against native quantization/GEMM on official weights."""
import argparse
import ctypes as C
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

import _pinned_sparkinfer  # noqa: F401
import torch
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from safetensors import safe_open
import b12x._lib.dense_gemm as dense
from b12x import freeze_kernel_resolution, unfreeze_kernel_resolution
from b12x._lib.utils import make_ptr
from b12x._lib.quant.mxfp8_rows import compile_mxfp8_rows_quant_aot, mxfp8_rows_quant_aot_grid


class Info(C.Structure):
    _fields_ = [(n, C.c_uint32) for n in ['abi','capacity','k','n']] + [
        (n, C.c_uint64) for n in ['scratch','values','row_scales','mma_scales','weight_scales']]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-lib', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--projection',choices=['q_b','q_a','kv'],default='q_b')
    args = parser.parse_args()
    assert not args.output.exists(), 'refusing to overwrite evidence'
    torch.manual_seed(41413)
    torch.backends.cuda.matmul.allow_tf32 = False
    lib = C.CDLL(str(args.native_lib))
    P, I = C.c_void_p, C.c_int32
    def bind(name, types):
        fn = getattr(lib,name); fn.argtypes = types; fn.restype = I
        def call(*values):
            status = fn(*values)
            assert status == 0, (name,status)
        return call
    info_fn = bind('ds41rt_v41_fp8_matrix_info',[I,I,I,C.POINTER(Info)])
    init = bind('ds41rt_v41_fp8_matrix_initialize',[I,I,I,C.POINTER(P)])
    pack = bind('ds41rt_v41_fp8_matrix_pack_scales',[P,P,I,I,P])
    storage = bind('ds41rt_v41_fp8_initialize_scratch',[P,P,C.c_uint64,P,P])
    linear = bind('ds41rt_v41_fp8_launch',[P,P,P,P,P,C.c_uint64,P,P,I,P])
    reduce = bind('ds41rt_v41_fp8_reduce_splits',[P,P,I,I,I,P])
    def stream(): return torch.cuda.current_stream().cuda_stream
    def ptr(address,dtype): return make_ptr(dtype,address,cute.AddressSpace.gmem,assumed_align=16)
    mapping = json.loads((args.snapshot/'model.safetensors.index.json').read_text())['weight_map']
    def load(name):
        with safe_open(args.snapshot/mapping[name],framework='pt',device='cpu') as f:
            return f.get_tensor(name).cuda().contiguous()
    prefix = 'layers.0.attn.' + {'q_b':'wq_b','q_a':'wq_a','kv':'wkv'}[args.projection]
    weight = load(prefix+'.weight')
    scale = load(prefix+'.scale')
    n,k = weight.shape
    assert (n,k)=={'q_b':(32768,1280),'q_a':(1280,5120),'kv':(512,5120)}[args.projection]
    configs = [((16,64),1),((16,128),1),((32,64),1),((32,128),1)] if args.projection=='q_b' else [((16,64),1),((64,64),1),((16,64),2),((16,64),4),((32,64),4)]
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    report = {'scope':__doc__,'native_sha256':hashlib.sha256(args.native_lib.read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'sparkinfer_lock':json.loads((Path(__file__).resolve().parents[2]/'third_party/sparkinfer.lock.json').read_text()),
        'projection':args.projection,'configs':configs,'shape':[n,k],'sms':sms,'eviction_bytes':256<<20,'cases':[],'hardware':[]}
    def save(): args.output.write_text(json.dumps(report,indent=2)+'\n')
    gpu_uuid = str(torch.cuda.get_device_properties(0).uuid)
    if not gpu_uuid.startswith('GPU-'): gpu_uuid = 'GPU-' + gpu_uuid
    def hardware():
        return subprocess.check_output(['nvidia-smi','--id='+gpu_uuid,
            '--query-gpu=uuid,pstate,power.limit,power.draw,clocks.sm,clocks.mem,clocks_event_reasons.active',
            '--format=csv,noheader'],text=True).strip()
    report['hardware'].append(hardware())
    plans = {}
    selector = dense._select_default_dense_gemm_plan
    policy = dense._dense_gemm_policy_for
    # Compile every static capacity/config before correctness or timing.
    for capacity in [1,16]:
        quant = compile_mxfp8_rows_quant_aot(size_k=k,expected_m=capacity,amax_floor=1e-4)
        for tile,wanted_slices in configs:
            dense._select_default_dense_gemm_plan = lambda *a, _tile=tile, **kw: replace(selector(*a,**kw),mma_tiler_mn=_tile)
            dense._dense_gemm_policy_for = lambda *a, _slices=wanted_slices, **kw: replace(policy(*a,**kw),split_k_slices=_slices)
            try:
                compiled,slices = dense.compile_dense_gemm_mxfp8_aot(size_m=capacity,size_n=n,size_k=k,
                    expected_m=capacity,return_split_k_metadata=True)
                assert slices==wanted_slices
                plans[capacity,tile,slices] = (quant,compiled)
                print('COMPILED',capacity,tile,slices,flush=True)
            finally:
                dense._select_default_dense_gemm_plan = selector
                dense._dense_gemm_policy_for = policy
    freeze_kernel_resolution('all dense probe plans compiled before live row changes')
    eviction = torch.zeros(256<<20,dtype=torch.uint8,device='cuda')
    original_weight = weight.view(torch.uint8).clone()
    try:
        for capacity in [1,16]:
            info,handle = Info(),P()
            info_fn(capacity,k,n,C.byref(info)); init(capacity,k,n,C.byref(handle))
            packed = torch.empty(info.weight_scales,dtype=torch.uint8,device='cuda')
            pack(scale.data_ptr(),packed.data_ptr(),k,n,stream())
            alpha = torch.empty(1,device='cuda')
            scratch = torch.empty(info.scratch,dtype=torch.uint8,device='cuda')
            storage(handle,scratch.data_ptr(),scratch.numel(),alpha.data_ptr(),stream())
            x = torch.empty((capacity,k),dtype=torch.bfloat16,device='cuda')
            outputs = [torch.empty((capacity,n),dtype=torch.bfloat16,device='cuda') for _ in range(len(configs)+1)]
            arenas = [torch.empty_like(scratch) for _ in configs]
            for arena in arenas: storage(handle,arena.data_ptr(),arena.numel(),alpha.data_ptr(),stream())
            partials = [torch.empty((slices,capacity,n),dtype=torch.float32,device='cuda') for _,slices in configs]
            for rows in ([1] if capacity==1 else [1,5,6,16]):
                def native():
                    linear(handle,x.data_ptr(),weight.data_ptr(),packed.data_ptr(),scratch.data_ptr(),scratch.numel(),
                           alpha.data_ptr(),outputs[0].data_ptr(),rows,stream())
                runs = [native]; names = ['native']
                for index,(tile,slices) in enumerate(configs):
                    quant,gemm = plans[capacity,tile,slices]; arena = arenas[index]; output = outputs[index+1]; partial = partials[index]
                    def run(quant=quant,gemm=gemm,arena=arena,output=output,partial=partial,slices=slices):
                        address = arena.data_ptr()
                        quant(ptr(x.data_ptr(),cutlass.BFloat16),ptr(address+info.values,cutlass.Uint8),
                              ptr(address+info.row_scales,cutlass.Uint8),ptr(address+info.mma_scales,cutlass.Uint8),
                              rows,mxfp8_rows_quant_aot_grid(size_k=k,rows=rows,expected_m=capacity,sm_count=sms),cuda.CUstream(stream()))
                        gemm(ptr(address+info.values,cutlass.Float8E4M3FN),ptr(weight.data_ptr(),cutlass.Float8E4M3FN),
                             ptr(address+info.mma_scales,cutlass.Float8E8M0FNU),ptr(packed.data_ptr(),cutlass.Float8E8M0FNU),
                             ptr(partial.data_ptr() if slices>1 else output.data_ptr(),cutlass.Float32 if slices>1 else cutlass.BFloat16),ptr(address+info.values,cutlass.Float8E4M3FN),
                             ptr(address+info.row_scales,cutlass.Float8E8M0FNU),ptr(address+info.mma_scales,cutlass.Float8E8M0FNU),
                             ptr(alpha.data_ptr(),cutlass.Float32),rows,cuda.CUstream(stream()))
                        if slices>1: reduce(partial.data_ptr(),output.data_ptr(),rows,n,slices,stream())
                    runs.append(run); names.append(f'{tile[0]}x{tile[1]}_s{slices}')
                graphs = []; x.normal_()
                for run in runs:
                    run(); graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph): run()
                    graphs.append(graph)
                errors = []
                for mutation in range(2):
                    x.normal_(std=1 if mutation==0 else 0.25)
                    for output in outputs: output.fill_(123)
                    for partial in partials: partial.fill_(float('nan'))
                    allocated = torch.cuda.memory_allocated()
                    for graph in graphs: graph.replay()
                    assert allocated==torch.cuda.memory_allocated()
                    torch.cuda.synchronize()
                    reference = outputs[0][:rows].float()
                    assert torch.isfinite(reference).all() and reference.norm()>0
                    for name,output in zip(names[1:],outputs[1:]):
                        actual = output[:rows].float(); error = actual-reference
                        rel = float(error.norm()/reference.norm()); absolute = float(error.abs().max())
                        assert torch.isfinite(actual).all() and actual.norm()>0 and (output[rows:]==123).all()
                        step = 2**(int(torch.floor(torch.log2(reference.abs().max())))-7)
                        assert rel<=1e-3 and absolute<=2*step+1e-4,(capacity,rows,name,rel,absolute)
                        errors.append({'mutation':mutation,'tile':name,'relative_l2':rel,'max_abs':absolute})
                samples = {kind:{name:[] for name in names} for kind in ['warm','evicted']}
                for repeat in range(6):
                    order = list(range(len(graphs))); order = order[repeat%len(graphs):]+order[:repeat%len(graphs)]
                    if repeat%2: order.reverse()
                    for index in order:
                        graph = graphs[index]
                        for kind in samples:
                            timings = []
                            for _ in range(12):
                                if kind=='evicted': eviction.add_(1)
                                start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                                start.record(); graph.replay(); end.record()
                                timings.append((start,end))
                            timings[-1][1].synchronize()
                            samples[kind][names[index]].append(statistics.mean(a.elapsed_time(b)*1000 for a,b in timings))
                assert torch.equal(weight.view(torch.uint8),original_weight)
                case = {'capacity':capacity,'rows':rows,'errors':errors,'samples_us':samples,
                        'median_us':{kind:{name:statistics.median(values) for name,values in arm.items()} for kind,arm in samples.items()}}
                report['cases'].append(case)
                report['hardware'].append(hardware())
                save(); print('PASS',capacity,rows,json.dumps(case['median_us']),flush=True)
                for graph in graphs: graph.reset()
    finally: unfreeze_kernel_resolution()
    report['limitations']=['Official weights with synthetic finite activations; bounded numerical comparison to native output, not full-model quality.',
        'Component graph timing includes quantization, GEMM and any native FP32-plane reduction; eviction work is outside events.',
        '256 MiB read/write eviction is a cache-pressure proxy, not proof of full cache invalidation.',
        'No serving performance or fixed-clock qualification claim.']
    save()


if __name__=='__main__': main()
