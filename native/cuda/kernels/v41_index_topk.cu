#include <cuda_runtime.h>
#include <math_constants.h>
#include <cub/block/block_radix_sort.cuh>
#include <stdint.h>
#include "ds41rt_v41_index_topk.h"
namespace {
bool valid(const void* p,uint64_t n,uint64_t alignment) {
  const auto a=reinterpret_cast<uintptr_t>(p);
  return a && a%alignment==0 && a<=UINTPTR_MAX-n;
}
bool disjoint(const void* a,uint64_t na,const void* b,uint64_t nb) {
  const auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);
  return x+na<=y || y+nb<=x;
}
__device__ uint64_t encode(float score,uint64_t pos) {
  if(pos>=1048576ull || isnan(score) || score==-CUDART_INF_F)return 0;
  // Treat both signed zeros as equal scores.
  const uint32_t bits=score==0.f?0u:__float_as_uint(score);
  const uint32_t ordered=bits&0x80000000u?~bits:bits^0x80000000u;
  return (uint64_t(ordered)<<32)|uint32_t(~uint32_t(pos));
}
using Sort=cub::BlockRadixSort<uint64_t,256,4>;
__global__ void tiles(const float* scores,const uint64_t* positions,uint64_t* out,int count,int blocks) {
  __shared__ Sort::TempStorage storage;
  uint64_t keys[4];const uint64_t row=blockIdx.y;
  #pragma unroll
  for(int i=0;i<4;++i) {
    const int col=blockIdx.x*1024+threadIdx.x*4+i;
    keys[i]=col<count?encode(scores[row*count+col],positions[row*count+col]):0;
  }
  Sort(storage).SortDescending(keys);
  #pragma unroll
  for(int i=0;i<4;++i) {
    const int rank=threadIdx.x*4+i;
    if(rank<512)out[(row*blocks+blockIdx.x)*512+rank]=keys[i];
  }
}
__global__ void merge(const uint64_t* in,uint64_t* out,int in_blocks,int out_blocks) {
  __shared__ Sort::TempStorage storage;
  uint64_t keys[4];const uint64_t row=blockIdx.y;
  #pragma unroll
  for(int i=0;i<4;++i) {
    const int col=blockIdx.x*1024+threadIdx.x*4+i;
    keys[i]=col<in_blocks*512?in[row*in_blocks*512+col]:0;
  }
  Sort(storage).SortDescending(keys);
  #pragma unroll
  for(int i=0;i<4;++i) {
    const int rank=threadIdx.x*4+i;
    if(rank<512)out[(row*out_blocks+blockIdx.x)*512+rank]=keys[i];
  }
}
__global__ void finish(const uint64_t* selected,uint64_t* carry,int32_t* output,int reset) {
  __shared__ Sort::TempStorage storage;
  uint64_t keys[4];const uint64_t row=blockIdx.x;
  #pragma unroll
  for(int i=0;i<4;++i) {
    const int col=threadIdx.x*4+i;
    keys[i]=col<512?selected[row*512+col]:(reset?0:carry[row*512+col-512]);
  }
  Sort(storage).SortDescending(keys);
  #pragma unroll
  for(int i=0;i<4;++i) {
    const int rank=threadIdx.x*4+i;
    if(rank<512)carry[row*512+rank]=keys[i];
    // Rank-sort the selected logical positions with the same storage. Unselected
    // entries are UINT64_MAX, so only the first 512 outputs need publishing.
    keys[i]=rank<512 && keys[i]!=0?uint32_t(~uint32_t(keys[i])):UINT64_MAX;
  }
  __syncthreads();
  Sort(storage).Sort(keys);
  #pragma unroll
  for(int i=0;i<4;++i) {
    const int rank=threadIdx.x*4+i;
    if(rank<512)output[row*512+rank]=keys[i]==UINT64_MAX?-1:int32_t(keys[i]);
  }
}
}
extern "C" int32_t ds41rt_v41_index_top512(const float* scores,const uint64_t* positions,
    uint64_t* carry,void* scratch,uint64_t scratch_bytes,int32_t* output,
    int32_t queries,int32_t candidates,int32_t reset,void* stream) {
  if(queries<1 || queries>4096 || candidates<1 || candidates>16384 || (reset!=0 && reset!=1))return cudaErrorInvalidValue;
  const int blocks=(candidates+1023)/1024;
  const uint64_t count=uint64_t(queries)*candidates,half=uint64_t(queries)*blocks*512;
  const uint64_t bytes[]={count*4,count*8,uint64_t(queries)*4096,half*16,uint64_t(queries)*2048};
  const void* ptrs[]={scores,positions,carry,scratch,output};
  const int alignment[]={4,8,8,8,4};
  if(scratch_bytes<bytes[3])return cudaErrorInvalidValue;
  for(int i=0;i<5;++i) {
    if(!valid(ptrs[i],bytes[i],alignment[i]))return cudaErrorInvalidValue;
    for(int j=0;j<i;++j)if(!disjoint(ptrs[i],bytes[i],ptrs[j],bytes[j]))return cudaErrorInvalidValue;
  }
  auto cuda_stream=reinterpret_cast<cudaStream_t>(stream);
  auto* a=static_cast<uint64_t*>(scratch);auto* b=a+half;
  tiles<<<dim3(blocks,queries),256,0,cuda_stream>>>(scores,positions,a,candidates,blocks);
  auto status=cudaGetLastError();if(status!=cudaSuccess)return status;
  for(int current=blocks;current>1;) {
    const int next=(current+1)/2;
    merge<<<dim3(next,queries),256,0,cuda_stream>>>(a,b,current,next);
    status=cudaGetLastError();if(status!=cudaSuccess)return status;
    auto* tmp=a;a=b;b=tmp;current=next;
  }
  finish<<<queries,256,0,cuda_stream>>>(a,carry,output,reset);
  return cudaGetLastError();
}
