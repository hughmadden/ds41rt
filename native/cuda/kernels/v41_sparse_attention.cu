#include "ds41rt_v41_sparse_attention.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>
#include <stdint.h>
#include <math_constants.h>
namespace {
using namespace nvcuda;
#if defined(__CUDA_ARCH_SPECIFIC__) && __CUDA_ARCH_SPECIFIC__ == 1200
__device__ __forceinline__ uint32_t packed_fp8_pair(uint16_t input,uint32_t factors) {
  uint32_t output;
  asm("{ .reg .b32 values; cvt.rn.bf16x2.e4m3x2 values, %1; mul.bf16x2 %0, values, %2; }"
      : "=r"(output) : "h"(input), "r"(factors));
  return output;
}

#endif
// Pad shared rows to distribute WMMA traffic across memory banks.
constexpr int kKvStride=520, kOutputStride=516, kProbabilityStride=80;
constexpr int kKvBytes=64*kKvStride*2, kOutputBytes=16*kOutputStride*4;
constexpr int kSharedBytes=kKvBytes+kOutputBytes+192+512;

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
__device__ __forceinline__ uint64_t locate(const ds41rt_v41_sparse_kv_t& v,const uint64_t* m,
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
// Grid-constant descriptor avoids a per-thread copy for dynamic source indexing.
template<bool Split>
__global__ void attend(const __nv_bfloat16* query,const float* sink,
    const uint64_t* metadata,const int32_t* selected,__nv_bfloat16* output,
    int width,const __grid_constant__ ds41rt_v41_sparse_kv_t v,float* partial) {
  // Zero width requests the exact former host maximum for this contiguous,
  // ascending request. Reading metadata keeps decode graph arguments stable.
  if(width==0) {
    const uint64_t last=metadata[(uint64_t(gridDim.x)-1)*10+3];
    width=last<127?int(last+1):128;
  }
  const int row=blockIdx.x,group=blockIdx.y;
  const uint64_t partial_base=((uint64_t(row)*gridDim.z+blockIdx.z)*64+group*16)*514;
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
    if constexpr(Split) {
      for(int i=tid;i<16*514;i+=128)partial[partial_base+i]=i%514==513?-1.0f:0.0f;
    } else {
      for(int i=tid;i<16*512;i+=128)output[base+i]=__float2bfloat16(0);
    }
    return;
  }
  extern __shared__ __align__(32) unsigned char memory[];
  auto* kv=reinterpret_cast<__nv_bfloat16*>(memory);
  auto* scratch=reinterpret_cast<float*>(memory+kKvBytes);
  // Scores, accumulator rescaling and PV probabilities have disjoint lifetimes.
  auto* probability=reinterpret_cast<__nv_bfloat16*>(scratch);
  auto* maximum=reinterpret_cast<float*>(memory+kKvBytes+kOutputBytes);
  float* sum=maximum+16;float* rescale=sum+16;
  if(tid<16){maximum[tid]=-1e30f;sum[tid]=0;}
  wmma::fragment<wmma::accumulator,16,16,16,float> acc[8];
#pragma unroll
  for(int t=0;t<8;++t)wmma::fill_fragment(acc[t],0.0f);
  auto* refs=reinterpret_cast<uint64_t*>(memory+kKvBytes+kOutputBytes+192);
  const int count=width+(v.compressed?512:0);
  // The first valid tile starts from zero accumulators; no rescale is needed.
  bool empty=true;
  const int tiles=(count+63)/64, per_part=Split?(tiles+gridDim.z-1)/gridDim.z:tiles;
  const int first=Split?blockIdx.z*per_part*64:0;
  const int last=Split?min(count,int(blockIdx.z+1)*per_part*64):count;
  for(int start=first;start<last;start+=64) {
    if(tid<64)refs[tid]=start+tid<count?locate(v,m,
      v.compressed?selected+uint64_t(row)*512:nullptr,start+tid,width):UINT64_MAX;
    // An entirely masked tile contributes zero probability and leaves both
    // online-softmax state and accumulators unchanged. Vote over the same
    // resolved references used below; valid entries may occur after empty tiles.
    if(!__syncthreads_or(tid<64 && refs[tid]!=UINT64_MAX))continue;
    // A warp stages four contiguous FP8 values per lane, reusing each
    // resolved key pointer. Preserve unaligned byte-addressed ABI inputs.
    for(int key=warp;key<64;key+=4) {
      const uint64_t ref=refs[key];
      for(int block=0;block<4;++block) {
        const int col=block*128+lane*4;
        uint64_t packed=0;
        if(ref!=UINT64_MAX) {
          const int tag=ref>>62;const uint64_t physical=ref&((1ull<<62)-1);
          const uint8_t* source=v.values[tag]+physical*512+col;
          uint32_t bytes;
          if((reinterpret_cast<uintptr_t>(source)&3)==0)
            bytes=*reinterpret_cast<const uint32_t*>(source);
          else bytes=uint32_t(source[0])|(uint32_t(source[1])<<8)|
              (uint32_t(source[2])<<16)|(uint32_t(source[3])<<24);
          const uint8_t exponent=v.scales[tag][physical*16+col/32];
#if defined(__CUDA_ARCH_SPECIFIC__) && __CUDA_ARCH_SPECIFIC__ == 1200
          // Match the existing scale decoding, including zero and 255.
          const uint32_t factor=exponent?(uint32_t(exponent)<<7):0x40;
          const uint32_t factors=factor|(factor<<16);
          packed=uint64_t(packed_fp8_pair(uint16_t(bytes),factors))|
              (uint64_t(packed_fp8_pair(uint16_t(bytes>>16),factors))<<32);
#else
          const float scale=exponent==0?0x1p-127f:__uint_as_float(uint32_t(exponent)<<23);
#pragma unroll
          for(int j=0;j<4;++j) {
            __nv_fp8_e4m3 f;f.__x=uint8_t(bytes>>(j*8));
            const auto value=__float2bfloat16_rn(__fmul_rn(float(f),scale));
            packed|=uint64_t(__bfloat16_as_ushort(value))<<(j*16);
          }
#endif
        }
        *reinterpret_cast<uint64_t*>(kv+key*kKvStride+col)=packed;
      }
    }
    __syncthreads();
    // Each warp computes one 16-key score tile in the original K order.
    {
      wmma::fragment<wmma::accumulator,16,16,16,float> scores;
      wmma::fill_fragment(scores,0.0f);
      for(int k=0;k<512;k+=16) {
        wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a;
        wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> b;
        wmma::load_matrix_sync(a,query+base+k,512);
        wmma::load_matrix_sync(b,kv+warp*16*kKvStride+k,kKvStride);
        wmma::mma_sync(scores,a,b,scores);
      }
      wmma::store_matrix_sync(scratch+warp*16,scores,64,wmma::mem_row_major);
    }
    __syncthreads();
    // Explicit carries avoid a dynamically indexed array spilling to local memory.
    __nv_bfloat16 p0a{},p0b{},p1a{},p1b{},p2a{},p2b{},p3a{},p3b{};
    for(int h=warp;h<16;h+=4) {
      const float a=refs[lane]!=UINT64_MAX?scratch[h*64+lane]*0.04419417382415922f:-CUDART_INF_F;
      const float b=refs[lane+32]!=UINT64_MAX?scratch[h*64+lane+32]*0.04419417382415922f:-CUDART_INF_F;
      const float prev=maximum[h],m=fmaxf(prev,warp_max(fmaxf(a,b)));
      const float scale=expf(prev-m),pa=expf(a-m),pb=expf(b-m);
      const float total=warp_sum(pa+pb);
      const auto a16=__float2bfloat16_rn(pa),b16=__float2bfloat16_rn(pb);
      if(h<4){p0a=a16;p0b=b16;}
      else if(h<8){p1a=a16;p1b=b16;}
      else if(h<12){p2a=a16;p2b=b16;}
      else {p3a=a16;p3b=b16;}
      if(lane==0){maximum[h]=m;rescale[h]=scale;sum[h]=fmaf(sum[h],scale,total);}
    }
    __syncthreads();
    if(!empty) {
      // WMMA accumulator lane ownership is opaque: rescale through shared storage
      // instead of depending on an undocumented fragment-to-head mapping.
#pragma unroll
      for(int t=0;t<8;++t)wmma::store_matrix_sync(scratch+warp*128+t*16,acc[t],kOutputStride,wmma::mem_row_major);
      __syncthreads();
      for(int i=tid;i<16*512;i+=128)scratch[(i/512)*kOutputStride+i%512]*=rescale[i/512];
      __syncthreads();
#pragma unroll
      for(int t=0;t<8;++t)wmma::load_matrix_sync(acc[t],scratch+warp*128+t*16,kOutputStride,wmma::mem_row_major);
      __syncthreads();
    }
    empty=false;
    // Accumulators are ready before probabilities reuse scratch.
    probability[warp*kProbabilityStride+lane]=p0a;
    probability[warp*kProbabilityStride+lane+32]=p0b;
    probability[(warp+4)*kProbabilityStride+lane]=p1a;
    probability[(warp+4)*kProbabilityStride+lane+32]=p1b;
    probability[(warp+8)*kProbabilityStride+lane]=p2a;
    probability[(warp+8)*kProbabilityStride+lane+32]=p2b;
    probability[(warp+12)*kProbabilityStride+lane]=p3a;
    probability[(warp+12)*kProbabilityStride+lane+32]=p3b;
    __syncthreads();
    for(int k=0;k<64;k+=16) {
      wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> p;
      wmma::load_matrix_sync(p,probability+k,kProbabilityStride);
#pragma unroll
      for(int t=0;t<8;++t) {
        wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> v;
        wmma::load_matrix_sync(v,kv+k*kKvStride+warp*128+t*16,kKvStride);
        wmma::mma_sync(acc[t],p,v,acc[t]);
      }
    }
    __syncthreads();
  }
#pragma unroll
  for(int t=0;t<8;++t)wmma::store_matrix_sync(scratch+warp*128+t*16,acc[t],kOutputStride,wmma::mem_row_major);
  if(tid<16) {
    if constexpr(Split) {
      partial[partial_base+tid*514+512]=maximum[tid];
      partial[partial_base+tid*514+513]=sum[tid];
    } else sum[tid]+=expf(sink[group*16+tid]-maximum[tid]);
  }
  __syncthreads();
  for(int i=tid;i<16*512;i+=128) {
    if constexpr(Split)partial[partial_base+(i/512)*514+i%512]=scratch[(i/512)*kOutputStride+i%512];
    else output[base+i]=__float2bfloat16_rn(scratch[(i/512)*kOutputStride+i%512]/sum[i/512]);
  }
}
__global__ void merge(const float* partial,const float* sink,__nv_bfloat16* output,int parts) {
  const int row=blockIdx.x,head=blockIdx.y,col=threadIdx.x;
  const uint64_t b=(uint64_t(row)*parts*64+head)*514;
  const uint64_t dest=(uint64_t(row)*64+head)*512;
  // Invalid proposal metadata must yield zero even for sinks whose exponential
  // underflows. A negative normalizer is reserved for this whole-row failure.
  if(partial[b+513]<0) {
    output[dest+col]=__float2bfloat16(0);
    output[dest+col+256]=__float2bfloat16(0);
    return;
  }
  float maximum=-1e30f;
  for(int p=0;p<parts;++p)maximum=fmaxf(maximum,partial[b+uint64_t(p)*64*514+512]);
  float sum=0,a=0,c=0;
  for(int p=0;p<parts;++p) {
    const uint64_t i=b+uint64_t(p)*64*514;
    const float factor=expf(partial[i+512]-maximum);
    sum=fmaf(partial[i+513],factor,sum);
    a=fmaf(partial[i+col],factor,a);
    c=fmaf(partial[i+col+256],factor,c);
  }
  sum+=expf(sink[head]-maximum);
  output[dest+col]=__float2bfloat16_rn(a/sum);
  output[dest+col+256]=__float2bfloat16_rn(c/sum);
}
bool span(const void* ptr,uint64_t n,uint32_t a) {
  const auto p=reinterpret_cast<uintptr_t>(ptr);return p&&p%a==0&&p<=UINTPTR_MAX-n;
}
bool disjoint(const void* a,uint64_t n,const void* b,uint64_t m) {
  const auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);return x+n<=y||y+m<=x;
}
}
extern "C" int32_t ds41rt_v41_sparse_attention_initialize(void) {
  const auto status=cudaFuncSetAttribute(attend<false>,cudaFuncAttributeMaxDynamicSharedMemorySize,kSharedBytes);
  if(status!=cudaSuccess)return status;
  return cudaFuncSetAttribute(attend<true>,cudaFuncAttributeMaxDynamicSharedMemorySize,kSharedBytes);
}
static int32_t launch_attention(const uint16_t* query,const float* sink,
    const uint64_t* metadata,const int32_t* selected,uint16_t* output,int32_t rows,
    int32_t window_width,const ds41rt_v41_sparse_kv_t* view,void* stream,float* partial,uint64_t scratch_bytes,int parts) {
  if(!view || rows<1 || rows>4096 || window_width<0 || window_width>128)return cudaErrorInvalidValue;
  const auto v=*view;
  if(v.compressed>1 || v.window_proposal_capacity<1 || v.window_proposal_capacity>4096 ||
    (v.compressed && (v.source_capacity<1 || v.source_capacity>16777216ull ||
     v.source_proposal_capacity<1 || v.source_proposal_capacity>4096 ||
     v.page_stride<1 || v.page_stride>4096)))return cudaErrorInvalidValue;
  if(parts<1 || parts>10)return cudaErrorInvalidValue;
  const uint64_t required=uint64_t(rows)*parts*64*514*sizeof(float);
  const uint64_t q=uint64_t(rows)*64*512*2;
  if(partial && (scratch_bytes<required || !span(partial,required,4) ||
      !disjoint(partial,required,output,q)))return cudaErrorInvalidValue;
  if(!span(output,q,32))return cudaErrorInvalidValue;
  const void* inputs[]={query,sink,metadata,v.window_end,selected,v.pages,v.source_end};
  const uint64_t sizes[]={q,256,uint64_t(rows)*80,8,uint64_t(rows)*2048,uint64_t(v.page_stride)*4,8};
  const uint32_t align[]={32,4,8,8,4,4,8};
  for(int i=0;i<(v.compressed?7:4);++i)
    if(!span(inputs[i],sizes[i],align[i]) || !disjoint(inputs[i],sizes[i],output,q) ||
      (partial && !disjoint(inputs[i],sizes[i],partial,required)))return cudaErrorInvalidValue;
  const uint64_t capacity[]={128,v.window_proposal_capacity,v.source_capacity,v.source_proposal_capacity};
  for(int i=0;i<(v.compressed?4:2);++i) {
    if(!span(v.values[i],capacity[i]*512,1) || !span(v.scales[i],capacity[i]*16,1) ||
      !disjoint(v.values[i],capacity[i]*512,output,q) || !disjoint(v.scales[i],capacity[i]*16,output,q) ||
      (partial && (!disjoint(v.values[i],capacity[i]*512,partial,required) ||
                   !disjoint(v.scales[i],capacity[i]*16,partial,required))))return cudaErrorInvalidValue;
  }
  if(partial) {
    attend<true><<<dim3(rows,4,parts),128,kSharedBytes,reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(query),sink,metadata,selected,
      reinterpret_cast<__nv_bfloat16*>(output),window_width,v,partial);
    const auto status=cudaGetLastError();
    if(status!=cudaSuccess)return status;
    merge<<<dim3(rows,64),256,0,reinterpret_cast<cudaStream_t>(stream)>>>(
      partial,sink,reinterpret_cast<__nv_bfloat16*>(output),parts);
  } else {
    attend<false><<<dim3(rows,4),128,kSharedBytes,reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(query),sink,metadata,selected,
      reinterpret_cast<__nv_bfloat16*>(output),window_width,v,nullptr);
  }
  return cudaGetLastError();
}
extern "C" int32_t ds41rt_v41_sparse_attention(const uint16_t* query,const float* sink,
    const uint64_t* metadata,const int32_t* selected,uint16_t* output,int32_t rows,
    int32_t window_width,const ds41rt_v41_sparse_kv_t* view,void* stream) {
  return launch_attention(query,sink,metadata,selected,output,rows,window_width,view,stream,nullptr,0,1);
}
extern "C" int32_t ds41rt_v41_sparse_attention_split(const uint16_t* query,const float* sink,
    const uint64_t* metadata,const int32_t* selected,uint16_t* output,int32_t rows,
    int32_t window_width,const ds41rt_v41_sparse_kv_t* view,void* stream,
    float* partial,uint64_t scratch_bytes,int32_t parts) {
  if(!partial)return cudaErrorInvalidValue;
  return launch_attention(query,sink,metadata,selected,output,rows,window_width,view,stream,partial,scratch_bytes,parts);
}
