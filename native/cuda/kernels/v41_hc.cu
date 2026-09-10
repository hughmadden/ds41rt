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

namespace {
__device__ float warp_sum(float value) {
  for (int offset=16;offset;offset>>=1) value += __shfl_down_sync(0xffffffffu,value,offset);
  return value;
}
__global__ void mixes_kernel(const __nv_bfloat16* residual, const float* fn,
    const float* scale, const float* base, float* pre, float* post, float* comb) {
  const uint64_t row=blockIdx.x;
  const int lane=threadIdx.x%32, warp=threadIdx.x/32;
  float dot[3]={0,0,0}, square=0;
  for (int col=lane;col<20480;col+=32) {
    const float x=__bfloat162float(residual[row*20480+col]);
    if (warp==0) square=fmaf(x,x,square);
#pragma unroll
    for(int j=0;j<3;++j) dot[j]=fmaf(x,fn[uint64_t(warp+j*8)*20480+col],dot[j]);
  }
  __shared__ float projected[24], inverse;
  if(warp==0) {
    square=warp_sum(square);
    if(lane==0) inverse=rsqrtf(square/20480.0f+1e-20f);
  }
#pragma unroll
  for(int j=0;j<3;++j) {
    dot[j]=warp_sum(dot[j]);
    if(lane==0) projected[warp+j*8]=dot[j];
  }
  __syncthreads();
  if(threadIdx.x<4) {
    const int j=threadIdx.x;
    const float a=fmaf(projected[j]*inverse,scale[0],base[j]);
    const float b=fmaf(projected[j+4]*inverse,scale[1],base[j+4]);
    pre[row*4+j]=1.0f/(1.0f+expf(-a))+1e-6f;
    post[row*4+j]=2.0f/(1.0f+expf(-b));
  }
  if(warp==0) {
    float value=lane<16 ? fmaf(projected[8+lane]*inverse,scale[2],base[8+lane]) : 0;
    float maximum=fmaxf(value,__shfl_xor_sync(0xffffffffu,value,1));
    maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffffu,maximum,2));
    value=expf(value-maximum);
    float sum=value+__shfl_xor_sync(0xffffffffu,value,1);
    sum+=__shfl_xor_sync(0xffffffffu,sum,2);
    value=value/sum+1e-6f;
    sum=value+__shfl_xor_sync(0xffffffffu,value,4);
    sum+=__shfl_xor_sync(0xffffffffu,sum,8);
    value/=sum+1e-6f;
    for(int iteration=1;iteration<20;++iteration) {
      sum=value+__shfl_xor_sync(0xffffffffu,value,1);
      sum+=__shfl_xor_sync(0xffffffffu,sum,2);
      value/=sum+1e-6f;
      sum=value+__shfl_xor_sync(0xffffffffu,value,4);
      sum+=__shfl_xor_sync(0xffffffffu,sum,8);
      value/=sum+1e-6f;
    }
    if(lane<16) comb[row*16+lane]=value;
  }
}
}
extern "C" int32_t ds41rt_v41_hc_mixes(const uint16_t* residual, const float* fn,
    const float* scale, const float* base, float* pre, float* post, float* comb,
    int32_t rows, void* stream) {
  if(rows<1 || rows>4096) return cudaErrorInvalidValue;
  const void* pointers[]={residual,fn,scale,base,pre,post,comb};
  const uint64_t small=uint64_t(rows)*16;
  const uint64_t sizes[]={uint64_t(rows)*40960,24*20480*4,12,96,small,small,small*4};
  for(int i=0;i<7;++i) if(!valid(pointers[i],sizes[i],i==0?2:4)) return cudaErrorInvalidValue;
  for(int i=4;i<7;++i) for(int j=0;j<i;++j)
    if(!disjoint(pointers[i],sizes[i],pointers[j],sizes[j])) return cudaErrorInvalidValue;
  mixes_kernel<<<rows,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(residual),fn,scale,base,pre,post,comb);
  return cudaGetLastError();
}
