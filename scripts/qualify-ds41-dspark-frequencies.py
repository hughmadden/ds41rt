#!/usr/bin/env python3
"""Compare native dSpark frequencies with the pinned reference on CUDA."""
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
    parser=argparse.ArgumentParser()
    parser.add_argument('--native-lib',type=Path,required=True)
    parser.add_argument('--reference-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    source=args.reference_root/'inference/model.py'
    lock=json.loads((Path(__file__).resolve().parents[1]/'docs/ds41-reference-lock.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest()==lock['files']['inference/model.py']
    nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='precompute_freqs_cis']
    ns={'torch':torch,'math':math,'lru_cache':lru_cache}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
    lib=C.CDLL(str(args.native_lib));fn=lib.ds41rt_v41_dspark_frequencies
    fn.argtypes=[C.c_void_p,C.c_void_p,C.c_int32,C.c_void_p];fn.restype=C.c_int32
    stream=torch.cuda.Stream()
    results=[]
    with torch.device('cuda'),torch.cuda.stream(stream):
        # Model construction in the official generator is inside torch.device(cuda).
        table=torch.view_as_real(ns['precompute_freqs_cis'](64,1048576,0,10000,40,32,1))
        for rows in [1,16,80,255,1023,4095,4096]:
            positions=torch.arange(rows,dtype=torch.int64)
            output=torch.empty((rows,32,2),dtype=torch.float32)
            def launch():assert fn(positions.data_ptr(),output.data_ptr(),rows,stream.cuda_stream)==0
            launch()
            torch.testing.assert_close(output,table[positions],rtol=0,atol=2e-7)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):launch()
            maximum=0.;changed=0
            batches=range(0,1048576,4096) if rows==4096 else [1048576-rows]
            for start in batches:
                positions.copy_(torch.arange(start,start+rows,dtype=torch.int64))
                graph.replay()
                expected=table[positions]
                error=(output-expected).abs().max().item()
                maximum=max(maximum,error);changed+=int((output!=expected).sum())
                torch.testing.assert_close(output,expected,rtol=0,atol=2e-7)
            assert fn(positions.data_ptr(),positions.data_ptr(),rows,stream.cuda_stream)!=0
            assert fn(positions.data_ptr(),output.data_ptr(),0,stream.cuda_stream)!=0
            assert fn(positions.data_ptr(),output.data_ptr(),4097,stream.cuda_stream)!=0
            results.append({'rows':rows,'replay_batches':len(batches),'max_abs':maximum,'different_fp32':changed})
            print('PASS',results[-1],flush=True)
        stream.synchronize()
    args.output.write_text(json.dumps({'reference_revision':lock['revision'],'device':torch.cuda.get_device_name(0),'results':results,'native_sha256':hashlib.sha256(args.native_lib.read_bytes()).hexdigest(),'qualifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
if __name__=='__main__':main()
