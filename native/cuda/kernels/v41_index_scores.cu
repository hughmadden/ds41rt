#include <cuda_runtime.h>
#include <math_constants.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "ds41rt_v41_index_scores.h"
namespace {
bool valid(const void* p,uint64_t bytes,uint64_t alignment) {
  const auto a=reinterpret_cast<uintptr_t>(p);
  return a && a%alignment==0 && a<=UINTPTR_MAX-bytes;
}
bool disjoint(const void* a,uint64_t na,const void* b,uint64_t nb) {
  const auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);
  return x+na<=y || y+nb<=x;
}
__device__ float bf(float x) {return __bfloat162float(__float2bfloat16_rn(x));}
__device__ float value(uint8_t packed,int high,uint8_t scale) {
  const unsigned code=(packed>>(high*4))&15;
  const float level=code%8<4?float(code%8)*0.5f:
      (code%8==4?2.f:(code%8==5?3.f:(code%8==6?4.f:6.f)));
  const float s=scale==0?0x1p-127f:__uint_as_float(uint32_t(scale)<<23);
  return bf(__fmul_rn(code&8?-level:level,s));
}
__device__ float warp_sum(float x) {
  #pragma unroll
  for(int d=16;d;d>>=1)x=__fadd_rn(x,__shfl_down_sync(0xffffffffu,x,d));
  return x;
}
template<bool Overlay>
__global__ void scores_kernel(const uint8_t* q,const uint8_t* qs,const __nv_bfloat16* weights,
    const uint8_t* keys,const uint8_t* ks,const uint32_t* pages,const uint64_t* lengths,
    const uint64_t* metadata,const uint64_t* positions,float* output,int candidates,
    int slots,int stride,uint64_t capacity,const uint8_t* proposals,const uint8_t* proposal_scales,
    uint64_t proposal_capacity) {
  const uint64_t row=blockIdx.y,index=row*candidates+blockIdx.x;
  constexpr int width=Overlay?6:2;
  const uint64_t slot=metadata[row*width],causal=metadata[row*width+1],pos=positions[index];
  if(slot>=uint64_t(slots) || pos>=causal) {
    if(threadIdx.x==0)output[index]=-CUDART_INF_F;return;
  }
  const uint64_t committed=lengths[slot];
  uint64_t physical=0;
  const uint8_t* values=keys;const uint8_t* scales=ks;
  if constexpr(Overlay) {
    const uint64_t start=metadata[row*6+2],count=metadata[row*6+3],offset=metadata[row*6+4],step=metadata[row*6+5];
    // A proposal is an append-only view, never an overwrite of committed keys.
    // Validate the full descriptor before reading either source. Use subtraction
    // bounds to avoid wrapping attacker/stale U64 metadata.
    if(start!=committed || start>1048576 || count>1048576-start ||
        (step!=1 && step!=2) || offset>proposal_capacity ||
        (count && (offset>=proposal_capacity || count-1>(proposal_capacity-1-offset)/step))) {
      if(threadIdx.x==0)output[index]=-CUDART_INF_F;return;
    }
    if(pos>=committed) {
      if(pos-committed>=count) {if(threadIdx.x==0)output[index]=-CUDART_INF_F;return;}
      physical=offset+(pos-committed)*step;values=proposals;scales=proposal_scales;
    }
  }
  if(pos<committed) {
    if(pos>=uint64_t(stride)*256) {if(threadIdx.x==0)output[index]=-CUDART_INF_F;return;}
    physical=uint64_t(pages[slot*stride+pos/256])*256+pos%256;
    if(physical>=capacity) {if(threadIdx.x==0)output[index]=-CUDART_INF_F;return;}
  } else if constexpr(!Overlay) {
    if(threadIdx.x==0)output[index]=-CUDART_INF_F;return;
  }
  const int lane=threadIdx.x%32,warp=threadIdx.x/32;
  float key[4];
  #pragma unroll
  for(int j=0;j<4;++j) {
    const int col=lane+j*32;
    key[j]=value(values[physical*64+col/2],col%2,scales[physical*4+j]);
  }
  __shared__ float partial[32];
  #pragma unroll
  for(int i=0;i<4;++i) {
    const int head=warp+i*8;
    const uint64_t qb=(row*32+head)*64,sb=(row*32+head)*4;
    float dot=0;
    #pragma unroll
    for(int j=0;j<4;++j) {
      const int col=lane+j*32;
      const float query=value(q[qb+col/2],col%2,qs[sb+j]);
      dot=__fadd_rn(dot,__fmul_rn(query,key[j]));
    }
    dot=warp_sum(dot);
    if(lane==0) {
      const float rounded=bf(dot);
      partial[head]=bf(__fmul_rn(rounded<0.f?0.f:rounded,__bfloat162float(weights[row*32+head])));
    }
  }
  __syncthreads();
  if(warp==0) {
    const float sum=warp_sum(partial[lane]);
    if(lane==0)output[index]=bf(sum);
  }
}
}
extern "C" int32_t ds41rt_v41_index_scores(const uint8_t* q,const uint8_t* qs,const uint16_t* weights,
    const uint8_t* keys,const uint8_t* ks,const uint32_t* pages,const uint64_t* lengths,
    const uint64_t* metadata,const uint64_t* positions,float* output,
    int32_t queries,int32_t candidates,int32_t slots,int32_t stride,uint64_t capacity,void* stream) {
  if(queries<1 || queries>4096 || candidates<1 || candidates>16384 || slots<1 || slots>16 ||
      stride<1 || stride>4096 || capacity<1 || capacity>16777216ull)return cudaErrorInvalidValue;
  const uint64_t rows=queries,count=rows*candidates,out=count*4;
  if(!valid(output,out,4))return cudaErrorInvalidValue;
  const void* ptrs[]={q,qs,weights,keys,ks,pages,lengths,metadata,positions};
  const uint64_t bytes[]={rows*2048,rows*128,rows*64,capacity*64,capacity*4,
      uint64_t(slots)*stride*4,uint64_t(slots)*8,rows*16,count*8};
  const int align[]={1,1,2,1,1,4,8,8,8};
  for(int i=0;i<9;++i)if(!valid(ptrs[i],bytes[i],align[i]) || !disjoint(ptrs[i],bytes[i],output,out))return cudaErrorInvalidValue;
  scores_kernel<false><<<dim3(candidates,queries),256,0,reinterpret_cast<cudaStream_t>(stream)>>>(
      q,qs,reinterpret_cast<const __nv_bfloat16*>(weights),keys,ks,pages,lengths,
      metadata,positions,output,candidates,slots,stride,capacity,nullptr,nullptr,0);
  return cudaGetLastError();
}

extern "C" int32_t ds41rt_v41_index_scores_overlay(const uint8_t* q,const uint8_t* qs,const uint16_t* weights,
    const uint8_t* keys,const uint8_t* ks,const uint32_t* pages,const uint64_t* lengths,
    const uint64_t* metadata,const uint64_t* positions,float* output,
    const uint8_t* proposals,const uint8_t* proposal_scales,
    int32_t queries,int32_t candidates,int32_t slots,int32_t stride,uint64_t capacity,
    uint64_t proposal_capacity,void* stream) {
  if(queries<1 || queries>4096 || candidates<1 || candidates>16384 || slots<1 || slots>16 ||
      stride<1 || stride>4096 || capacity<1 || capacity>16777216ull ||
      proposal_capacity<1 || proposal_capacity>4096)return cudaErrorInvalidValue;
  const uint64_t rows=queries,count=rows*candidates,out=count*4;
  if(!valid(output,out,4))return cudaErrorInvalidValue;
  const void* ptrs[]={q,qs,weights,keys,ks,pages,lengths,metadata,positions,proposals,proposal_scales};
  const uint64_t bytes[]={rows*2048,rows*128,rows*64,capacity*64,capacity*4,
      uint64_t(slots)*stride*4,uint64_t(slots)*8,rows*48,count*8,proposal_capacity*64,proposal_capacity*4};
  const int align[]={1,1,2,1,1,4,8,8,8,1,1};
  for(int i=0;i<11;++i)if(!valid(ptrs[i],bytes[i],align[i]) || !disjoint(ptrs[i],bytes[i],output,out))return cudaErrorInvalidValue;
  scores_kernel<true><<<dim3(candidates,queries),256,0,reinterpret_cast<cudaStream_t>(stream)>>>(
      q,qs,reinterpret_cast<const __nv_bfloat16*>(weights),keys,ks,pages,lengths,
      metadata,positions,output,candidates,slots,stride,capacity,proposals,proposal_scales,proposal_capacity);
  return cudaGetLastError();
}
