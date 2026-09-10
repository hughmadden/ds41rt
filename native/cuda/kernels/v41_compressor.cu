#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cublas_v2.h>
#include <stdint.h>
#include <new>
#include "ds41rt_v41_compressor.h"
namespace {
constexpr uint64_t workspace_bytes=4*1024*1024;
bool valid(const void* p,uint64_t bytes,uint64_t alignment) {
  const auto a=reinterpret_cast<uintptr_t>(p);
  return a && a%alignment==0 && a<=UINTPTR_MAX-bytes;
}
bool disjoint(const void* a,uint64_t na,const void* b,uint64_t nb) {
  const auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);
  return x+na<=y || y+nb<=x;
}
int32_t bs(cublasStatus_t s){return s==CUBLAS_STATUS_SUCCESS?0:-int32_t(s);}
struct Handle {cublasHandle_t blas;void* workspace;int device;};
__global__ void index_store_kernel(const uint8_t* packed,const uint8_t* scales,
    const uint64_t* destinations,uint8_t* cache,uint8_t* cache_scales,uint64_t capacity) {
  const uint64_t row=blockIdx.x,dst=destinations[row];
  if(dst>=capacity)return;
  const int t=threadIdx.x;
  if(t<64)cache[dst*64+t]=packed[row*64+t];
  if(t<4)cache_scales[dst*4+t]=scales[row*4+t];
}
__global__ void index_pack_kernel(const __nv_bfloat16* input,uint8_t* packed,
    uint8_t* scales,uint64_t groups) {
  const uint64_t pair=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  const uint64_t group=pair/16;
  if(group>=groups)return;
  const float a=__bfloat162float(input[pair*2]);
  const float b=__bfloat162float(input[pair*2+1]);
  float maximum=fmaxf(fabsf(a),fabsf(b));
  const unsigned mask=__activemask();
  #pragma unroll
  for(int delta=8;delta;delta>>=1)
    maximum=fmaxf(maximum,__shfl_xor_sync(mask,maximum,delta,16));
  maximum=fmaxf(maximum,6.0f*0x1p-126f);
  const uint32_t bits=__float_as_uint(__fmul_rn(maximum,1.0f/6.0f));
  const uint32_t exponent=(bits>>23)+((bits&0x7fffffu)!=0);
  const float scale=__uint_as_float(exponent<<23);
  if(threadIdx.x%16==0)scales[group]=uint8_t(exponent);
  // Division by an exact power of two; the native conversion is saturating RNE.
  packed[pair]=__nv_cvt_float2_to_fp4x2(
      make_float2(__fdiv_rn(a,scale),__fdiv_rn(b,scale)),__NV_E2M1,cudaRoundNearest);
}
__device__ float sum_warp(float x) {
  for(int step=16;step;step>>=1)x=__fadd_rn(x,__shfl_down_sync(0xffffffffu,x,step));
  return x;
}
__global__ void pool_kernel(const float* kv,const float* scores,const float* pending_kv,
    const float* pending_scores,const uint64_t* predecessors,const __nv_bfloat16* weight,
    __nv_bfloat16* output,int slots) {
  const uint64_t row=blockIdx.x,previous=predecessors[row],base=row*512;
  const int t=threadIdx.x;
  // A source in the projection batch must precede this row. UINT64_MAX is the
  // normal no-output sentinel; all other invalid descriptors also fail closed.
  if(previous>=uint64_t(slots)+row) {
    output[base+t]=__float2bfloat16_rn(0);
    output[base+t+256]=__float2bfloat16_rn(0);
    return;
  }
  const float* old_kv=previous<uint64_t(slots)?pending_kv+previous*512:kv+(previous-slots)*512;
  const float* old_scores=previous<uint64_t(slots)?pending_scores+previous*512:scores+(previous-slots)*512;
  float pooled[2],square=0;
  #pragma unroll
  for(int j=0;j<2;++j) {
    const int col=t+j*256;
    const float a=old_scores[col],b=scores[base+col],maximum=fmaxf(a,b);
    const float ea=expf(a-maximum),eb=expf(b-maximum),sum=__fadd_rn(ea,eb);
    const float wa=__fdiv_rn(ea,sum),wb=__fdiv_rn(eb,sum);
    const float value=__fadd_rn(__fmul_rn(old_kv[col],wa),__fmul_rn(kv[base+col],wb));
    pooled[j]=__bfloat162float(__float2bfloat16_rn(value));
    square=__fadd_rn(square,__fmul_rn(pooled[j],pooled[j]));
  }
  __shared__ float partial[8],inverse;
  square=sum_warp(square);
  if(t%32==0)partial[t/32]=square;
  __syncthreads();
  if(t<32) {
    square=sum_warp(t<8?partial[t]:0);
    if(t==0)inverse=rsqrtf(__fadd_rn(square/512.0f,1e-20f));
  }
  __syncthreads();
  #pragma unroll
  for(int j=0;j<2;++j) {
    const int col=t+j*256;
    output[base+col]=__float2bfloat16_rn(__fmul_rn(__fmul_rn(pooled[j],inverse),__bfloat162float(weight[col])));
  }
}
}
extern "C" int32_t ds41rt_v41_compressor_create(void* workspace,uint64_t bytes,void** output) {
  if(!output)return cudaErrorInvalidValue;
  *output=nullptr;
  if(bytes<workspace_bytes || !valid(workspace,workspace_bytes,256))return cudaErrorInvalidValue;
  auto* h=new(std::nothrow)Handle{};
  if(!h)return cudaErrorMemoryAllocation;
  auto cuda=cudaGetDevice(&h->device);
  if(cuda!=cudaSuccess){delete h;return cuda;}
  auto status=cublasCreate(&h->blas);
  if(status!=CUBLAS_STATUS_SUCCESS){delete h;return bs(status);}
  status=cublasSetMathMode(h->blas,CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
  if(status!=CUBLAS_STATUS_SUCCESS){cublasDestroy(h->blas);delete h;return bs(status);}
  h->workspace=workspace;*output=h;return 0;
}
extern "C" int32_t ds41rt_v41_compressor_destroy(void* opaque) {
  if(!opaque)return cudaErrorInvalidValue;
  auto* h=static_cast<Handle*>(opaque);auto status=cublasDestroy(h->blas);delete h;return bs(status);
}
static int32_t project(void* opaque,const uint16_t* input,
    const uint16_t* weight,void* output,int32_t rows,int32_t ratio,void* stream,int n,int k) {
  if(!opaque || rows<1 || rows>4096 || (ratio!=1 && ratio!=2))return cudaErrorInvalidValue;
  auto* h=static_cast<Handle*>(opaque);int device;auto cuda=cudaGetDevice(&device);
  if(cuda!=cudaSuccess)return cuda;
  if(device!=h->device)return cudaErrorInvalidDevice;
  const uint64_t x=uint64_t(rows)*k*2,w=uint64_t(n)*k*2,o=uint64_t(rows)*n*(ratio==2?4:2);
  if(!valid(input,x,2)||!valid(weight,w,2)||!valid(output,o,ratio==2?4:2)||
      !disjoint(input,x,output,o)||!disjoint(weight,w,output,o)||
      !disjoint(h->workspace,workspace_bytes,input,x)||!disjoint(h->workspace,workspace_bytes,weight,w)||
      !disjoint(h->workspace,workspace_bytes,output,o))return cudaErrorInvalidValue;
  auto status=cublasSetStream(h->blas,reinterpret_cast<cudaStream_t>(stream));
  if(status!=CUBLAS_STATUS_SUCCESS)return bs(status);
  status=cublasSetWorkspace(h->blas,h->workspace,workspace_bytes);
  if(status!=CUBLAS_STATUS_SUCCESS)return bs(status);
  const float alpha=1,beta=0;
  return bs(cublasGemmEx(h->blas,CUBLAS_OP_T,CUBLAS_OP_N,n,rows,k,&alpha,
      weight,CUDA_R_16BF,k,input,CUDA_R_16BF,k,&beta,output,ratio==2?CUDA_R_32F:CUDA_R_16BF,
      n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
extern "C" int32_t ds41rt_v41_compressor_project(void* handle,const uint16_t* input,
    const uint16_t* weight,void* output,int32_t rows,int32_t ratio,void* stream) {
  return project(handle,input,weight,output,rows,ratio,stream,512,5120);
}
extern "C" int32_t ds41rt_v41_index_key_project(void* handle,const uint16_t* input,
    const uint16_t* weight,uint16_t* output,int32_t rows,void* stream) {
  return project(handle,input,weight,output,rows,1,stream,128,512);
}
extern "C" int32_t ds41rt_v41_index_pack(const uint16_t* input,uint8_t* packed,
    uint8_t* scales,int32_t rows,void* stream) {
  if(rows<1 || rows>131072)return cudaErrorInvalidValue;
  const uint64_t x=uint64_t(rows)*256,p=uint64_t(rows)*64,s=uint64_t(rows)*4;
  if(!valid(input,x,2)||!valid(packed,p,1)||!valid(scales,s,1)||
      !disjoint(input,x,packed,p)||!disjoint(input,x,scales,s)||
      !disjoint(packed,p,scales,s))return cudaErrorInvalidValue;
  index_pack_kernel<<<(p+255)/256,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(input),packed,scales,uint64_t(rows)*4);
  return cudaGetLastError();
}
extern "C" int32_t ds41rt_v41_index_store(const uint8_t* packed,const uint8_t* scales,
    const uint64_t* destinations,uint8_t* cache,uint8_t* cache_scales,
    int32_t rows,uint64_t capacity,void* stream) {
  if(rows<1 || rows>4096 || capacity<1 || capacity>16*1048576ull)return cudaErrorInvalidValue;
  const void* ptrs[]={packed,scales,destinations,cache,cache_scales};
  const uint64_t bytes[]={uint64_t(rows)*64,uint64_t(rows)*4,uint64_t(rows)*8,capacity*64,capacity*4};
  for(int i=0;i<5;++i) {
    if(!valid(ptrs[i],bytes[i],i==2?8:1))return cudaErrorInvalidValue;
    for(int j=0;j<i;++j)if(!disjoint(ptrs[i],bytes[i],ptrs[j],bytes[j]))return cudaErrorInvalidValue;
  }
  index_store_kernel<<<rows,64,0,reinterpret_cast<cudaStream_t>(stream)>>>(
      packed,scales,destinations,cache,cache_scales,capacity);
  return cudaGetLastError();
}
extern "C" int32_t ds41rt_v41_compressor_pool(const float* kv,const float* scores,
    const float* pending_kv,const float* pending_scores,const uint64_t* predecessors,
    const uint16_t* norm_weight,uint16_t* output,int32_t rows,int32_t slots,void* stream) {
  if(rows<1 || rows>4096 || slots<1 || slots>16)return cudaErrorInvalidValue;
  const void* inputs[]={kv,scores,pending_kv,pending_scores,predecessors,norm_weight};
  const uint64_t sizes[]={uint64_t(rows)*2048,uint64_t(rows)*2048,uint64_t(slots)*2048,
      uint64_t(slots)*2048,uint64_t(rows)*8,1024};
  const uint64_t o=uint64_t(rows)*1024;
  if(!valid(output,o,2))return cudaErrorInvalidValue;
  for(int i=0;i<6;++i)if(!valid(inputs[i],sizes[i],i==4?8:(i==5?2:4))||!disjoint(inputs[i],sizes[i],output,o))return cudaErrorInvalidValue;
  pool_kernel<<<rows,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(kv,scores,pending_kv,pending_scores,
      predecessors,reinterpret_cast<const __nv_bfloat16*>(norm_weight),reinterpret_cast<__nv_bfloat16*>(output),slots);
  return cudaGetLastError();
}
