#!/usr/bin/env python3
"""Generate an isolated CUDA probe; does not alter the serving kernel or ABI."""
import argparse
from pathlib import Path
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', required=True, type=Path)
args = parser.parse_args()
assert not args.output.exists()
p = Path(__file__).resolve().parents[1] / 'native/cuda/kernels/v41_sparse_attention.cu'
s = p.read_text()
assert s.count('template<bool Split,int Groups=1,bool SourceFP4=false>') == 1
assert s.count('ds41rt_v41_sparse_kv_t v,float* partial,\n    const uint64_t* window_begins) {') == 1
s = s.replace('template<bool Split,int Groups=1,bool SourceFP4=false>', 'template<bool Split,int Groups=1,bool SourceFP4=false,bool Batched=false>')
s=s.replace('ds41rt_v41_sparse_kv_t v,float* partial,\n    const uint64_t* window_begins) {', 'ds41rt_v41_sparse_kv_t uniform_view,float* partial,\n    const uint64_t* window_begins,const ds41rt_v41_sparse_kv_t* row_views=nullptr) {\n  const auto& v=Batched?row_views[blockIdx.x]:uniform_view;')
assert 'const auto& v=Batched' in s
s+='''
// Isolated probe ABI: caller supplies validated device descriptors and disjoint
// live storage. Not part of the serving ABI; no host descriptor validation.
extern "C" int32_t ds41rt_probe_sparse_batch_initialize() {
  auto status=cudaFuncSetAttribute(attend<true,1,false,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,kSharedBytes);
  if(status!=cudaSuccess)return status;
  return cudaFuncSetAttribute(attend<true,1,true,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,kSharedBytes);
}
template<bool FP4> static int32_t probe_batch(const uint16_t* query,const float* sink,
    const uint64_t* metadata,const int32_t* selected,uint16_t* output,int32_t rows,
    const ds41rt_v41_sparse_kv_t* views,void* stream,const uint64_t* begins,
    float* partial,int32_t parts) {
  ds41rt_v41_sparse_kv_t unused{};
  attend<true,1,FP4,true><<<dim3(rows,4,parts),128,kSharedBytes,reinterpret_cast<cudaStream_t>(stream)>>>(
    reinterpret_cast<const __nv_bfloat16*>(query),sink,metadata,selected,
    reinterpret_cast<__nv_bfloat16*>(output),0,unused,partial,begins,views);
  auto status=cudaGetLastError();
  if(status!=cudaSuccess)return status;
  merge<<<dim3(rows,64),256,0,reinterpret_cast<cudaStream_t>(stream)>>>(
    partial,sink,reinterpret_cast<__nv_bfloat16*>(output),parts);
  return cudaGetLastError();
}
extern "C" int32_t ds41rt_probe_sparse_batch(const uint16_t* query,const float* sink,
    const uint64_t* metadata,const int32_t* selected,uint16_t* output,int32_t rows,
    const ds41rt_v41_sparse_kv_t* views,void* stream,const uint64_t* begins,
    float* partial,int32_t parts,int32_t fp4) {
  if(rows<1 || rows>48 || parts<1 || parts>10)return cudaErrorInvalidValue;
  return fp4?probe_batch<true>(query,sink,metadata,selected,output,rows,views,stream,begins,partial,parts):
    probe_batch<false>(query,sink,metadata,selected,output,rows,views,stream,begins,partial,parts);
}
'''
args.output.write_text(s)
