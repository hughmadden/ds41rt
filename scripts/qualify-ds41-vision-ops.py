#!/usr/bin/env python3
"""Independent CUDA vision operation checks, including tiled attention and guards.

Uses full tensors and FP64 error metrics; run under Compute Sanitizer as well.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise ValueError('preserve evidence: report already exists')
    lib = C.CDLL(str(args.library.resolve()))
    signatures = {
        'create': [C.c_void_p, C.c_uint64, C.POINTER(C.c_void_p)],
        'destroy': [C.c_void_p],
        'linear': [C.c_void_p]*6+[C.c_int]*3+[C.c_void_p],
        'norm': [C.c_void_p]*3+[C.c_int,C.c_void_p],
        'elementwise': [C.c_void_p]*3+[C.c_int]*3+[C.c_void_p],
        'rope': [C.c_void_p]*5+[C.c_int]*2+[C.c_void_p],
        'attention': [C.c_void_p]*8+[C.c_int]*2+[C.c_void_p],
        'merge': [C.c_void_p]*2+[C.c_int]*2+[C.c_void_p],
        'span': [C.c_void_p]*5+[C.c_int]*2+[C.c_void_p],
    }
    functions = {}
    for name, types in signatures.items():
        fn = getattr(lib, 'ds41rt_v41_vision_'+name)
        fn.argtypes, fn.restype = types, C.c_int
        functions[name] = fn
    torch.manual_seed(841)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    stream = torch.cuda.current_stream().cuda_stream
    workspace = torch.empty(4<<20, dtype=torch.uint8, device='cuda')
    handle = C.c_void_p()
    assert functions['create'](workspace.data_ptr(), workspace.numel(), C.byref(handle)) == 0
    report = dict(scope=__doc__, library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
        torch=torch.__version__, gpu=torch.cuda.get_device_name(), cases=[], guards=[], passed=False)

    maps = Path('/proc/self/maps')
    report['loaded_cuda_libraries'] = sorted({line.split()[-1] for line in
        maps.read_text().splitlines() if '/libcublas' in line or '/libcudart' in line}) if maps.exists() else []

    def rand(*shape):
        return torch.randn(*shape, device='cuda', dtype=torch.bfloat16)

    def empty(*shape, dtype=torch.bfloat16):
        return torch.empty(*shape, device='cuda', dtype=dtype)

    def call(name, *values):
        values = [v.data_ptr() if isinstance(v, torch.Tensor) else v for v in values]
        status = functions[name](*values, stream)
        assert status == 0, (name, status)
        torch.cuda.synchronize()

    def check(name, actual, expected, tolerance=0.0001, exact=False):
        a, b = actual.double(), expected.double()
        relative = float((a-b).norm()/b.norm().clamp_min(1e-30))
        same = torch.equal(actual, expected)
        passed = bool(a.isfinite().all() and b.isfinite().all()) and (same if exact else relative <= tolerance)
        report['cases'].append(dict(name=name, relative_l2=relative, max_absolute=float((a-b).abs().max()),
            exact=same, tolerance=tolerance, passed=passed))
        print(name, relative, passed, flush=True)

    def reject(name, operation, *values):
        values = [v.data_ptr() if isinstance(v, torch.Tensor) else v for v in values]
        status = functions[operation](*values, stream)
        report['guards'].append(dict(name=name, status=status, passed=status != 0))
        assert status != 0, name

    try:
        for rows in (1, 35, 129, 257):
            x, weight, bias = rand(rows, 588), rand(1024, 588), rand(1024)
            scratch, y = empty(rows, 1024, dtype=torch.float32), empty(rows, 1024)
            call('linear', handle, x, weight, bias, scratch, y, rows, 588, 1024)
            check(f'linear-bias-{rows}', y, (x.float() @ weight.float().T + bias.float()).bfloat16())
            call('linear', handle, x, weight, None, scratch, y, rows, 588, 1024)
            check(f'linear-{rows}', y, F.linear(x, weight))
            x, weight = rand(rows, 1024), rand(1024)
            call('norm', x, weight, y, rows)
            normalized = (x.float()*torch.rsqrt(x.float().square().mean(-1, keepdim=True)+1e-6)*weight.float()).bfloat16()
            check(f'norm-{rows}', y, normalized, 0.0002)
            other = rand(rows, 1024)
            call('elementwise', x, other, y, rows, 1024, 0)
            check(f'add-{rows}', y, x+other, exact=True)
            expected = x+other
            call('elementwise', x, other, x, rows, 1024, 0)
            check(f'add-in-place-{rows}', x, expected, exact=True)
            gates, activated = rand(rows, 5632), empty(rows, 2816)
            call('elementwise', gates, None, activated, rows, 2816, 1)
            gate, up = gates.chunk(2, -1)
            check(f'swiglu-{rows}', activated, F.silu(gate)*up, exact=True)
            g = rand(rows, 5120)
            expected = F.gelu(g)
            call('elementwise', g, None, g, rows, 5120, 2)
            check(f'gelu-in-place-{rows}', g, expected, 0.0002)
            qkv = rand(rows, 3072)
            q,k,v = empty(rows, 1024),empty(rows, 1024),empty(rows, 1024)
            inv = 1.0/(10000.0**(torch.arange(16,device='cuda',dtype=torch.float32)/16))
            call('rope', qkv, inv, q, k, v, 1, rows)
            position = torch.stack((torch.zeros(rows,device='cuda'),torch.arange(rows,device='cuda')), -1)
            frequencies = (position[..., None]*inv).flatten(1).unsqueeze(1)
            def rotate(t):
                x1,x2 = t.reshape(rows,16,64).float().chunk(2,-1)
                cos,sin = frequencies.cos(),frequencies.sin()
                return torch.cat((x1*cos-x2*sin,x2*cos+x1*sin),-1).reshape(rows,1024).bfloat16()
            qr,kr,vr = qkv.chunk(3,-1)
            check(f'rope-q-{rows}',q,rotate(qr),0.0002)
            check(f'rope-k-{rows}',k,rotate(kr),0.0002)
            check(f'rope-v-{rows}',v,vr,exact=True)
            scores = empty(16*128*rows,dtype=torch.float32)
            values,output,y = empty(rows,1024,dtype=torch.float32),empty(rows,1024,dtype=torch.float32),empty(rows,1024)
            for scale in (1, 16):
                q,k,v = rand(rows,1024)*scale,rand(rows,1024)*scale,rand(rows,1024)
                qh,kh,vh = [t.view(rows,16,64).transpose(0,1) for t in (q,k,v)]
                probabilities = (qh.float() @ kh.float().transpose(-1,-2)*0.125).softmax(-1)
                for fp32 in (0,1):
                    call('attention',handle,q,k,v,scores,values,output,y,rows,fp32)
                    probs = probabilities if fp32 else probabilities.bfloat16().float()
                    expected = (probs @ vh.float()).transpose(0,1).reshape(rows,1024).bfloat16()
                    check(f'attention-{rows}-scale{scale}-fp32{fp32}',y,expected,0.0004)
                if rows == 35:
                    reject('attention-alias','attention',handle,q,k,v,scores,values,output,q,rows,0)
                    reject('attention-mode','attention',handle,q,k,v,scores,values,output,y,rows,2)
            reject(f'norm-partial-overlap-{rows}','norm',x,weight,x.data_ptr()+2,rows)
        for h,w in ((1,1),(5,7),(39,39),(3,3063)):
            x = rand(h*w,1024)
            mh,mw = (h+2)//3,(w+2)//3
            y = empty(mh*mw,9216)
            call('merge',x,y,h,w)
            padded = F.pad(x.view(h,w,-1).permute(2,0,1),(0,-w%3,0,-h%3))
            expected = F.unfold(padded[None],3,stride=3).squeeze(0).T
            check(f'merge-{h}x{w}',y,expected,exact=True)
            features,start,newline,end = rand(mh*mw,5120),rand(1,5120),rand(1,5120),rand(1,5120)
            span = empty(mh*(mw+1)+2,5120)
            call('span',features,start,newline,end,span,mh,mw)
            expected = torch.cat([start,*[part for row in features.view(mh,mw,5120) for part in (row,newline)],end])
            check(f'span-{h}x{w}',span,expected,exact=True)
            reject(f'merge-overflow-{h}x{w}','merge',x,y,2**31-1,w)
        report['passed'] = all(c['passed'] for c in report['cases']+report['guards'])
    finally:
        torch.cuda.synchronize()
        assert functions['destroy'](handle) == 0
        args.report.write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
