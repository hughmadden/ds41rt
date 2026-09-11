#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <stdint.h>
#include "ds41rt_v41_router.h"
namespace {
__global__ void score_kernel(const __nv_bfloat16* hidden,const __nv_bfloat16* weight,float* scores,int experts) {
  const uint64_t row=blockIdx.y,expert=blockIdx.x;
  const int tid=threadIdx.x;
  float sum=0;
  for(int col=tid;col<5120;col+=256)
    sum=fmaf(__bfloat162float(hidden[row*5120+col]),__bfloat162float(weight[expert*5120+col]),sum);
  __shared__ float partial[256];partial[tid]=sum;__syncthreads();
  for(int stride=128;stride;stride>>=1) {if(tid<stride) partial[tid]+=partial[tid+stride];__syncthreads();}
  if(tid==0) {
    const float logit=partial[0];
    scores[row*experts+expert]=sqrtf(logit>20.0f?logit:log1pf(expf(logit)));
  }
}
template<bool TransformLogits = false>
__global__ void select_kernel(float* scores,const float* bias,const float* bias_vl,
    const uint8_t* image_mask,uint32_t* ids,float* routing,int experts,int topk) {
  const uint64_t row=blockIdx.x;const int tid=threadIdx.x;
  if constexpr (TransformLogits) {
    if(tid<experts) {
      const float logit=scores[row*experts+tid];
      scores[row*experts+tid]=sqrtf(logit>20.0f?logit:log1pf(expf(logit)));
    }
    __syncthreads();
  }
  const float* correction=image_mask && image_mask[row]?bias_vl:bias;
  float candidate=tid<experts?scores[row*experts+tid]+correction[tid]:-CUDART_INF_F;
  __shared__ float values[512],selected[6];__shared__ uint32_t indices[512];
  for(int rank=0;rank<topk;++rank) {
    values[tid]=candidate;indices[tid]=tid<experts?tid:UINT32_MAX;__syncthreads();
    for(int stride=256;stride;stride>>=1) {
      if(tid<stride && (values[tid+stride]>values[tid] ||
          (values[tid+stride]==values[tid] && indices[tid+stride]<indices[tid]))) {
        values[tid]=values[tid+stride];indices[tid]=indices[tid+stride];
      }
      __syncthreads();
    }
    const uint32_t winner=indices[0];
    if(tid==0) {ids[row*topk+rank]=winner;selected[rank]=scores[row*experts+winner];}
    if(tid==winner) candidate=-CUDART_INF_F;
    __syncthreads();
  }
  if(tid<topk) {
    float total=0;for(int j=0;j<topk;++j) total+=selected[j];
    routing[row*topk+tid]=(selected[tid]/(total+1e-20f))*1.5f;
  }
}
bool span(const void* p,uint64_t n,int alignment) {
  auto a=reinterpret_cast<uintptr_t>(p);return a && a%alignment==0 && a<=UINTPTR_MAX-n;
}
bool disjoint(const void* a,uint64_t n,const void* b,uint64_t m) {
  auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);return x+n<=y || y+m<=x;
}
}
extern "C" int32_t ds41rt_v41_router(const uint16_t* hidden,const uint16_t* weight,
    const float* bias,const float* bias_vl,const uint8_t* image_mask,float* scores,
    uint32_t* ids,float* routing,int32_t rows,int32_t experts,void* stream) {
  if(rows<1 || rows>4096 || (experts!=128 && experts!=384)) return cudaErrorInvalidValue;
  const int topk=experts==128?3:6;
  const void* p[]={hidden,weight,bias,bias_vl,image_mask,scores,ids,routing};
  const uint64_t n[]={uint64_t(rows)*10240,uint64_t(experts)*10240,uint64_t(experts)*4,
      image_mask?uint64_t(experts)*4:0,image_mask?uint64_t(rows):0,uint64_t(rows)*experts*4,
      uint64_t(rows)*topk*4,uint64_t(rows)*topk*4};
  for(int i=0;i<8;++i) if(n[i] && !span(p[i],n[i],i<2?2:i==4?1:4)) return cudaErrorInvalidValue;
  for(int i=5;i<8;++i) for(int j=0;j<i;++j) if(n[j] && !disjoint(p[i],n[i],p[j],n[j])) return cudaErrorInvalidValue;
  auto s=reinterpret_cast<cudaStream_t>(stream);
  score_kernel<<<dim3(experts,rows),256,0,s>>>(reinterpret_cast<const __nv_bfloat16*>(hidden),
      reinterpret_cast<const __nv_bfloat16*>(weight),scores,experts);
  auto status=cudaGetLastError();if(status!=cudaSuccess) return status;
  select_kernel<false><<<rows,512,0,s>>>(scores,bias,bias_vl,image_mask,ids,routing,experts,topk);
  return cudaGetLastError();
}
extern "C" int32_t ds41rt_v41_router_select_logits(float* scores,const float* bias,
    const float* bias_vl,const uint8_t* image_mask,uint32_t* ids,float* routing,
    int32_t rows,int32_t experts,void* stream) {
  if(rows<1 || rows>4096 || (experts!=128 && experts!=384)) return cudaErrorInvalidValue;
  const int topk=experts==128?3:6;
  const void* p[]={bias,bias_vl,image_mask,scores,ids,routing};
  const uint64_t n[]={uint64_t(experts)*4,image_mask?uint64_t(experts)*4:0,
      image_mask?uint64_t(rows):0,uint64_t(rows)*experts*4,
      uint64_t(rows)*topk*4,uint64_t(rows)*topk*4};
  for(int i=0;i<6;++i) if(n[i] && !span(p[i],n[i],i==2?1:4)) return cudaErrorInvalidValue;
  for(int i=3;i<6;++i) for(int j=0;j<i;++j)
    if(n[j] && !disjoint(p[i],n[i],p[j],n[j])) return cudaErrorInvalidValue;
  select_kernel<true><<<rows,512,0,reinterpret_cast<cudaStream_t>(stream)>>>(
      scores,bias,bias_vl,image_mask,ids,routing,experts,topk);
  return cudaGetLastError();
}
