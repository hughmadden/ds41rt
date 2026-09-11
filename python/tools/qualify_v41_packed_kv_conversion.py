#!/usr/bin/env python3
"""Exhaustively compare the native packed FP8-pair helper with its scalar fallback."""
import argparse
import ctypes as C
import hashlib
import json
import subprocess
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[2]
    source=root/'native/cuda/kernels/v41_sparse_attention.cu'
    probe=a.output_dir/'probe.cu'
    probe.write_text('#include "'+str(source)+'"\n'+r'''
__global__ void pairs(uint32_t* packed,uint32_t* scalar) {
  const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
  const uint16_t code=uint16_t(i);
  const uint32_t exponent=i>>16;
  const uint32_t factor=exponent?(exponent<<7):0x40;
#if defined(__CUDA_ARCH_SPECIFIC__) && __CUDA_ARCH_SPECIFIC__ == 1200
  packed[i]=packed_fp8_pair(code,factor|(factor<<16));
#elif defined(__CUDA_ARCH__)
#error "This probe must compile the SM120 architecture-specific helper"
#endif
  const float scale=exponent==0?0x1p-127f:__uint_as_float(exponent<<23);
  uint32_t reference=0;
  for(int j=0;j<2;++j) {
    __nv_fp8_e4m3 f;f.__x=uint8_t(code>>(j*8));
    const auto value=__float2bfloat16_rn(__fmul_rn(float(f),scale));
    reference|=uint32_t(__bfloat16_as_ushort(value))<<(j*16);
  }
  scalar[i]=reference;
}
extern "C" int test_pairs(uint32_t* packed,uint32_t* scalar,void* stream) {
  pairs<<<65536,256,0,(cudaStream_t)stream>>>(packed,scalar);
  return cudaGetLastError();
}
''')
    library=a.output_dir/'probe.so'
    subprocess.run(['nvcc','-O3','-std=c++17','-gencode','arch=compute_120a,code=sm_120a',
        '--shared','-Xcompiler=-fPIC','-I'+str(root/'native/include'),str(probe),'-o',str(library)],check=True)
    lib=C.CDLL(str(library.resolve()));fn=lib.test_pairs
    fn.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p]
    out=[torch.empty(1<<24,device='cuda',dtype=torch.int32) for _ in range(2)]
    assert fn(*[t.data_ptr() for t in out],torch.cuda.current_stream().cuda_stream)==0
    torch.cuda.synchronize()
    bits=[t.view(torch.int16).int().bitwise_and(65535) for t in out]
    nan=[((t&0x7f80)==0x7f80)&((t&0x7f)!=0) for t in bits]
    different=bits[0]!=bits[1]
    finite_mismatches=int((different & ~(nan[0]&nan[1])).sum())
    assert finite_mismatches==0,finite_mismatches
    record=dict(passed=True,pairs=1<<24,scalar_values=1<<25,
        all_e4m3_pairs=True,all_256_scale_bytes=True,finite_and_infinite_bit_mismatches=finite_mismatches,
        nan_payload_differences=int((different & nan[0]&nan[1]).sum()),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        probe_sha256=hashlib.sha256(library.read_bytes()).hexdigest())
    (a.output_dir/'result.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)


if __name__=='__main__':
    main()
