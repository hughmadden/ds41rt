#include "ds41rt_v41_sparse_attention.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>
#include <stdint.h>
#include <math_constants.h>
namespace {
using namespace nvcuda;
constexpr int kSharedBytes=65536+32768+2048+192+512;

__device__ float warp_max(float x) {
  for(int n=16;n;n>>=1)x=fmaxf(x,__shfl_xor_sync(0xffffffffu,x,n));
  return x;
}
__device__ float warp_sum(float x) {
  for(int n=16;n;n>>=1)x=__fadd_rn(x,__shfl_xor_sync(0xffffffffu,x,n));
  return x;
}
// Top two bits distinguish ring, private window, paged source and private source.
// Invalid rows never dereference a value or scale pointer.
__device__ uint64_t locate(ds41rt_v41_sparse_kv_t v,const uint64_t* m,
    const int32_t* selected,int key,int width) {
  if(key<width) {
    const uint64_t begin=m[3]+1>128?m[3]+1-128:0,pos=begin+key;
    if(pos>m[3])return UINT64_MAX;
    return pos<m[0]?pos%128:((1ull<<62)|(m[1]+pos-m[0]));
  }
  const int32_t id=selected[key-width];
  if(id<0 || uint64_t(id)>=m[5])return UINT64_MAX;
  const uint64_t pos=uint64_t(id);
  if(pos>=m[6]) {
    if(pos-m[6]>=m[7])return UINT64_MAX;
    return (3ull<<62)|(m[8]+(pos-m[6])*m[9]);
  }
  if(pos>=uint64_t(v.page_stride)*256)return UINT64_MAX;
  const uint64_t physical=uint64_t(v.pages[pos/256])*256+pos%256;
  return physical<v.source_capacity?((2ull<<62)|physical):UINT64_MAX;
}
__global__ void attend(const __nv_bfloat16* query,const float* sink,
    const uint64_t* metadata,const int32_t* selected,__nv_bfloat16* output,
    int width,ds41rt_v41_sparse_kv_t v) {
  const int row=blockIdx.x,group=blockIdx.y;
  const int tid=threadIdx.x,warp=tid/32,lane=tid%32;
  const uint64_t base=(uint64_t(row)*64+group*16)*512;
  const uint64_t* m=metadata+uint64_t(row)*10;
  bool valid=m[0]==*v.window_end && m[0]<=1048576 && m[2]>0 &&
    m[2]<=1048576-m[0] && m[1]<=v.window_proposal_capacity &&
    m[2]<=v.window_proposal_capacity-m[1] && m[3]>=m[0] && m[3]-m[0]<m[2] &&
    uint64_t(width)>=(m[3]+1<128?m[3]+1:128);
  if(valid && v.compressed)valid=m[4]==0 && m[6]==*v.source_end && m[6]<=1048576 &&
    m[7]<=1048576-m[6] && m[5]<=m[6]+m[7] && (m[9]==1 || m[9]==2) &&
    m[8]<=v.source_proposal_capacity && (!m[7] || (m[8]<v.source_proposal_capacity &&
      m[7]-1<=(v.source_proposal_capacity-1-m[8])/m[9]));
  if(!valid) {
    for(int i=tid;i<16*512;i+=128)output[base+i]=__float2bfloat16(0);
    return;
  }
  extern __shared__ __align__(32) unsigned char memory[];
  auto* kv=reinterpret_cast<__nv_bfloat16*>(memory);
  auto* scratch=reinterpret_cast<float*>(memory+65536);
  auto* probability=reinterpret_cast<__nv_bfloat16*>(memory+65536+32768);
  auto* maximum=reinterpret_cast<float*>(memory+65536+32768+2048);
  float* sum=maximum+16;float* rescale=sum+16;
  if(tid<16){maximum[tid]=-1e30f;sum[tid]=0;}
  wmma::fragment<wmma::accumulator,16,16,16,float> acc[8];
#pragma unroll
  for(int t=0;t<8;++t)wmma::fill_fragment(acc[t],0.0f);
  auto* refs=reinterpret_cast<uint64_t*>(memory+65536+32768+2048+192);
  const int count=width+(v.compressed?512:0);
  for(int start=0;start<count;start+=64) {
    if(tid<64)refs[tid]=start+tid<count?locate(v,m,
      v.compressed?selected+uint64_t(row)*512:nullptr,start+tid,width):UINT64_MAX;
    __syncthreads();
    for(int i=tid;i<64*512;i+=128) {
      const uint64_t ref=refs[i/512];const int col=i%512;
      __nv_bfloat16 value=__float2bfloat16(0);
      if(ref!=UINT64_MAX) {
        const int tag=ref>>62;const uint64_t physical=ref&((1ull<<62)-1);
        __nv_fp8_e4m3 f;f.__x=v.values[tag][physical*512+col];
        const uint8_t exponent=v.scales[tag][physical*16+col/32];
        // E8M0 exponent zero is 2^-127, unlike an IEEE zero exponent field.
        const float scale=exponent==0?0x1p-127f:__uint_as_float(uint32_t(exponent)<<23);
        value=__float2bfloat16_rn(__fmul_rn(float(f),scale));
      }
      kv[i]=value;
    }
    __syncthreads();
    if(warp==0) {
      wmma::fragment<wmma::accumulator,16,16,16,float> scores[4];
#pragma unroll
      for(int t=0;t<4;++t)wmma::fill_fragment(scores[t],0.0f);
      for(int k=0;k<512;k+=16) {
        wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a;
        wmma::load_matrix_sync(a,query+base+k,512);
#pragma unroll
        for(int t=0;t<4;++t) {
          wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> b;
          wmma::load_matrix_sync(b,kv+t*16*512+k,512);
          wmma::mma_sync(scores[t],a,b,scores[t]);
        }
      }
#pragma unroll
      for(int t=0;t<4;++t)wmma::store_matrix_sync(scratch+t*16,scores[t],64,wmma::mem_row_major);
    }
    __syncthreads();
    for(int h=warp;h<16;h+=4) {
      const float a=refs[lane]!=UINT64_MAX?scratch[h*64+lane]*0.04419417382415922f:-CUDART_INF_F;
      const float b=refs[lane+32]!=UINT64_MAX?scratch[h*64+lane+32]*0.04419417382415922f:-CUDART_INF_F;
      const float prev=maximum[h],m=fmaxf(prev,warp_max(fmaxf(a,b)));
      const float scale=expf(prev-m),pa=expf(a-m),pb=expf(b-m);
      const float total=warp_sum(pa+pb);
      probability[h*64+lane]=__float2bfloat16_rn(pa);
      probability[h*64+lane+32]=__float2bfloat16_rn(pb);
      if(lane==0){maximum[h]=m;rescale[h]=scale;sum[h]=fmaf(sum[h],scale,total);}
    }
    __syncthreads();
    // WMMA accumulator lane ownership is opaque: rescale through shared storage
    // instead of depending on an undocumented fragment-to-head mapping.
#pragma unroll
    for(int t=0;t<8;++t)wmma::store_matrix_sync(scratch+warp*128+t*16,acc[t],512,wmma::mem_row_major);
    __syncthreads();
    for(int i=tid;i<16*512;i+=128)scratch[i]*=rescale[i/512];
    __syncthreads();
#pragma unroll
    for(int t=0;t<8;++t)wmma::load_matrix_sync(acc[t],scratch+warp*128+t*16,512,wmma::mem_row_major);
    for(int k=0;k<64;k+=16) {
      wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> p;
      wmma::load_matrix_sync(p,probability+k,64);
#pragma unroll
      for(int t=0;t<8;++t) {
        wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> v;
        wmma::load_matrix_sync(v,kv+k*512+warp*128+t*16,512);
        wmma::mma_sync(acc[t],p,v,acc[t]);
      }
    }
    __syncthreads();
  }
#pragma unroll
  for(int t=0;t<8;++t)wmma::store_matrix_sync(scratch+warp*128+t*16,acc[t],512,wmma::mem_row_major);
  if(tid<16)sum[tid]+=expf(sink[group*16+tid]-maximum[tid]);
  __syncthreads();
  for(int i=tid;i<16*512;i+=128)output[base+i]=__float2bfloat16_rn(scratch[i]/sum[i/512]);
}
bool span(const void* ptr,uint64_t n,uint32_t a) {
  const auto p=reinterpret_cast<uintptr_t>(ptr);return p&&p%a==0&&p<=UINTPTR_MAX-n;
}
bool disjoint(const void* a,uint64_t n,const void* b,uint64_t m) {
  const auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);return x+n<=y||y+m<=x;
}
}
extern "C" int32_t ds41rt_v41_sparse_attention_initialize(void) {
  return cudaFuncSetAttribute(attend,cudaFuncAttributeMaxDynamicSharedMemorySize,kSharedBytes);
}
extern "C" int32_t ds41rt_v41_sparse_attention(const uint16_t* query,const float* sink,
    const uint64_t* metadata,const int32_t* selected,uint16_t* output,int32_t rows,
    int32_t window_width,const ds41rt_v41_sparse_kv_t* view,void* stream) {
  if(!view || rows<1 || rows>4096 || window_width<1 || window_width>128)return cudaErrorInvalidValue;
  const auto v=*view;
  if(v.compressed>1 || v.window_proposal_capacity<1 || v.window_proposal_capacity>4096 ||
    (v.compressed && (v.source_capacity<1 || v.source_capacity>16777216ull ||
     v.source_proposal_capacity<1 || v.source_proposal_capacity>4096 ||
     v.page_stride<1 || v.page_stride>4096)))return cudaErrorInvalidValue;
  const uint64_t q=uint64_t(rows)*64*512*2;
  if(!span(output,q,32))return cudaErrorInvalidValue;
  const void* inputs[]={query,sink,metadata,v.window_end,selected,v.pages,v.source_end};
  const uint64_t sizes[]={q,256,uint64_t(rows)*80,8,uint64_t(rows)*2048,uint64_t(v.page_stride)*4,8};
  const uint32_t align[]={32,4,8,8,4,4,8};
  for(int i=0;i<(v.compressed?7:4);++i)
    if(!span(inputs[i],sizes[i],align[i]) || !disjoint(inputs[i],sizes[i],output,q))return cudaErrorInvalidValue;
  const uint64_t capacity[]={128,v.window_proposal_capacity,v.source_capacity,v.source_proposal_capacity};
  for(int i=0;i<(v.compressed?4:2);++i) {
    if(!span(v.values[i],capacity[i]*512,1) || !span(v.scales[i],capacity[i]*16,1) ||
      !disjoint(v.values[i],capacity[i]*512,output,q) || !disjoint(v.scales[i],capacity[i]*16,output,q))return cudaErrorInvalidValue;
  }
  attend<<<dim3(rows,4),128,kSharedBytes,reinterpret_cast<cudaStream_t>(stream)>>>(
    reinterpret_cast<const __nv_bfloat16*>(query),sink,metadata,selected,
    reinterpret_cast<__nv_bfloat16*>(output),window_width,v);
  return cudaGetLastError();
}
