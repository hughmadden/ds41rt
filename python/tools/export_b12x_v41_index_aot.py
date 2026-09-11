#!/usr/bin/env python3
"""Export the pointer-only V4.1 FP4 overlay scorer; reject unexpected C ABIs."""
import argparse, hashlib, json, re
from pathlib import Path
import _pinned_sparkinfer  # noqa: F401
import cutlass.cute as cute
from cutlass import Uint8, Uint32, Int32, Int64, BFloat16, Float32
from b12x._lib.compiler import compile as compile_kernel, KernelCompileSpec
from b12x._lib.utils import make_ptr, current_cuda_stream
from b12x.attention.dsa_indexer._v41_overlay import V41OverlayScore

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output-dir',type=Path,required=True)
a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
types=[Uint8,Uint8,BFloat16,Uint8,Uint8,Uint32,Int64,Int64,Int64,Float32,Uint8,Uint8]
raw=compile_kernel(V41OverlayScore(),
    *[make_ptr(t,16,cute.AddressSpace.gmem,assumed_align=1) for t in types],
    Int32(1),Int32(1),Int32(1),Int32(1),Int64(256),Int64(1),current_cuda_stream(),
    compile_spec=KernelCompileSpec.from_key('attention.indexer.v41_overlay',1,()))
raw.export_to_c(str(a.output_dir),'v41_index_score','ds41rt_v41_index_score')
h=(a.output_dir/'v41_index_score.h').read_text()
pointers='q qs weights keys ks pages lengths metadata positions output proposals ps'.split()
i32='rows width slots stride'.split();i64='capacity proposal_capacity'.split()
expected=['ds41rt_v41_index_score_Kernel_Module_t *module']
expected += ['void *'+x for x in pointers]+['int32_t '+x for x in i32]+['int64_t '+x for x in i64]+['cudaStream_t stream']
sig=re.search(r'static inline int32_t cute_dsl_ds41rt_v41_index_score_wrapper\(([^)]*)\)',h)
normalize=lambda x:re.sub(r'\s+','',x)
if not sig or normalize(sig[1])!=normalize(','.join(expected)):
    raise ValueError('unexpected V4.1 index generated signature')
args=re.search(r'void \*args\[20\] = \{([^}]*)\}',h)
if not args or normalize(args[1])!=','.join('&'+x for x in pointers+i32+i64+['stream','ret']):
    raise ValueError('unexpected V4.1 index generated dispatch order')
manifest=dict(pointers=pointers,i32=i32,i64=i64,artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.output_dir/'v41_index_score.h',a.output_dir/'v41_index_score.o']})
(a.output_dir/'v41_index_score.json').write_text(json.dumps(manifest,indent=2)+'\n')
