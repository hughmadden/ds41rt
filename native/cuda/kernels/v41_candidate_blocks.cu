#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>
#include "ds41rt_v41_candidate_blocks.h"
namespace {
bool valid(const void* p,uint64_t n,int align) {
  auto a=reinterpret_cast<uintptr_t>(p);return a && a%align==0 && a<=UINTPTR_MAX-n;
}
bool disjoint(const void* a,uint64_t na,const void* b,uint64_t nb) {
  auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);
  return x+na<=y || y+nb<=x;
}
__global__ void tile_positions(uint64_t* positions,uint64_t* first,int width,uint64_t begin) {
  const uint64_t col=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x,row=blockIdx.y;
  if(col==0)first[row]=begin;
  if(col<uint64_t(width))positions[row*width+col]=begin+col<1048576?begin+col:UINT64_MAX;
}
__global__ void block_max(const float* scores,const uint64_t* first,const uint64_t* lengths,
    float* maxima,uint64_t* ids,int width,int blocks) {
  const uint64_t row=blockIdx.y,block=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(block>=uint64_t(blocks))return;
  const uint64_t begin=first[row],end=lengths[row],out=row*blocks+block;
  float maximum=-CUDART_INF_F;uint64_t id=UINT64_MAX;
  if(begin<1048576 && begin%8==0 && end<=1048576) {
    const uint64_t pos=begin+block*8;
    if(pos<end && pos<1048576) {
      id=pos/8;
      #pragma unroll
      for(int j=0;j<8;++j) {
        const uint64_t col=block*8+j;
        if(col<uint64_t(width) && pos+j<end && pos+j<1048576)
          maximum=fmaxf(maximum,scores[row*width+col]);
      }
      if(id==(end-1)/8)maximum=CUDART_INF_F;
    }
  }
  maxima[out]=maximum;ids[out]=id;
}
__global__ void expand(const int32_t* blocks,const uint64_t* lengths,uint64_t* out) {
  const uint64_t row=blockIdx.y,col=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  const int32_t block=blocks[row*2048+col/8];const uint64_t end=lengths[row];
  const uint64_t pos=uint64_t(uint32_t(block))*8+col%8;
  out[row*16384+col]=block>=0 && block<131072 && end<=1048576 && pos<end?pos:UINT64_MAX;
}
}
extern "C" int32_t ds41rt_v41_candidate_block_max(const float* scores,const uint64_t* first,
    const uint64_t* lengths,float* maxima,uint64_t* ids,int32_t queries,int32_t width,void* stream) {
  if(queries<1 || queries>4096 || width<1 || width>16384)return cudaErrorInvalidValue;
  const int blocks=(width+7)/8;const uint64_t q=queries;
  const void* ptrs[]={scores,first,lengths,maxima,ids};
  const uint64_t sizes[]={q*width*4,q*8,q*8,q*blocks*4,q*blocks*8};
  const int align[]={4,8,8,4,8};
  for(int i=0;i<5;++i) {
    if(!valid(ptrs[i],sizes[i],align[i]))return cudaErrorInvalidValue;
    if(i>=3)for(int j=0;j<i;++j)if(!disjoint(ptrs[i],sizes[i],ptrs[j],sizes[j]))return cudaErrorInvalidValue;
  }
  block_max<<<dim3((blocks+255)/256,queries),256,0,reinterpret_cast<cudaStream_t>(stream)>>>(
      scores,first,lengths,maxima,ids,width,blocks);
  return cudaGetLastError();
}
extern "C" int32_t ds41rt_v41_candidate_expand(const int32_t* blocks,const uint64_t* lengths,
    uint64_t* positions,int32_t queries,void* stream) {
  if(queries<1 || queries>4096)return cudaErrorInvalidValue;
  const uint64_t q=queries;
  if(!valid(blocks,q*8192,4)||!valid(lengths,q*8,8)||!valid(positions,q*131072,8)||
      !disjoint(positions,q*131072,blocks,q*8192)||!disjoint(positions,q*131072,lengths,q*8))return cudaErrorInvalidValue;
  expand<<<dim3(64,queries),256,0,reinterpret_cast<cudaStream_t>(stream)>>>(blocks,lengths,positions);
  return cudaGetLastError();
}

extern "C" int32_t ds41rt_v41_candidate_tile(uint64_t* positions,uint64_t* first,
    int32_t queries,int32_t width,uint64_t begin,void* stream) {
  if(queries<1 || queries>4096 || width<1 || width>16384 || begin>=1048576 || begin%8)
    return cudaErrorInvalidValue;
  const uint64_t bytes=uint64_t(queries)*width*8,starts=uint64_t(queries)*8;
  if(!valid(positions,bytes,8) || !valid(first,starts,8) || !disjoint(positions,bytes,first,starts))
    return cudaErrorInvalidValue;
  tile_positions<<<dim3((width+255)/256,queries),256,0,reinterpret_cast<cudaStream_t>(stream)>>>(
      positions,first,width,begin);
  return cudaGetLastError();
}
