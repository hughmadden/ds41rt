#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "ds41rt_v41_hc.h"
namespace {
__global__ void pre_kernel(const __nv_bfloat16* residual, const float* pre,
    __nv_bfloat16* output, uint64_t values) {
  const uint64_t i = uint64_t(blockIdx.x)*blockDim.x + threadIdx.x;
  if (i >= values) return;
  const uint64_t row = i/5120, col = i%5120;
  float sum = 0;
#pragma unroll
  for (int copy = 0; copy < 4; ++copy)
    sum = __fadd_rn(sum, __fmul_rn(pre[row*4+copy], __bfloat162float(residual[(row*4+copy)*5120+col])));
  output[i] = __float2bfloat16_rn(sum);
}
__global__ void post_kernel(const __nv_bfloat16* sublayer, const __nv_bfloat16* residual,
    const float* post, const float* comb, __nv_bfloat16* output, uint64_t values) {
  const uint64_t i = uint64_t(blockIdx.x)*blockDim.x + threadIdx.x;
  if (i >= values) return;
  const uint64_t row = i/(4*5120), destination = (i/5120)%4, col = i%5120;
  float sum = 0;
#pragma unroll
  for (int source = 0; source < 4; ++source)
    sum = __fadd_rn(sum, __fmul_rn(comb[row*16+source*4+destination],
        __bfloat162float(residual[(row*4+source)*5120+col])));
  const float expansion = __fmul_rn(post[row*4+destination], __bfloat162float(sublayer[row*5120+col]));
  output[i] = __float2bfloat16_rn(__fadd_rn(expansion, sum));
}
bool valid(const void* pointer, uint64_t bytes, uint64_t alignment) {
  const auto address = reinterpret_cast<uintptr_t>(pointer);
  return address && address%alignment == 0 && address <= UINTPTR_MAX-bytes;
}
bool disjoint(const void* a, uint64_t na, const void* b, uint64_t nb) {
  const auto x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
  return x+na <= y || y+nb <= x;
}
}
extern "C" int32_t ds41rt_v41_hc_pre(const uint16_t* residual, const float* pre,
    uint16_t* collapsed, int32_t rows, void* stream) {
  if (rows < 1 || rows > 4096) return cudaErrorInvalidValue;
  const uint64_t r = uint64_t(rows)*40960, p = uint64_t(rows)*16, c = uint64_t(rows)*10240;
  if (!valid(residual,r,2) || !valid(pre,p,4) || !valid(collapsed,c,2) ||
      !disjoint(residual,r,collapsed,c) || !disjoint(pre,p,collapsed,c)) return cudaErrorInvalidValue;
  pre_kernel<<<(c/2+255)/256,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(residual),pre,
      reinterpret_cast<__nv_bfloat16*>(collapsed),c/2);
  return cudaGetLastError();
}
extern "C" int32_t ds41rt_v41_hc_post(const uint16_t* sublayer, const uint16_t* residual,
    const float* post, const float* comb, uint16_t* output, int32_t rows, void* stream) {
  if (rows < 1 || rows > 4096) return cudaErrorInvalidValue;
  const uint64_t r = uint64_t(rows)*40960, p = uint64_t(rows)*16, c = uint64_t(rows)*10240;
  const void* inputs[] = {sublayer,residual,post,comb};
  const uint64_t sizes[] = {c,r,p,p*4};
  if (!valid(output,r,2)) return cudaErrorInvalidValue;
  for (int i=0;i<4;++i)
    if (!valid(inputs[i],sizes[i],i<2?2:4) || !disjoint(inputs[i],sizes[i],output,r)) return cudaErrorInvalidValue;
  post_kernel<<<(r/2+255)/256,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(sublayer),reinterpret_cast<const __nv_bfloat16*>(residual),
      post,comb,reinterpret_cast<__nv_bfloat16*>(output),r/2);
  return cudaGetLastError();
}
