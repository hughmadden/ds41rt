#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include "ds41rt_v41_attention_ops.h"
namespace {
bool valid(const void* p, uint64_t bytes, uint64_t align) {
  const auto x = reinterpret_cast<uintptr_t>(p);
  return x && x % align == 0 && x <= UINTPTR_MAX - bytes;
}
bool disjoint(const void* a, uint64_t na, const void* b, uint64_t nb) {
  const auto x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
  return x + na <= y || y + nb <= x;
}
__device__ float sum_warp(float x) {
  for (int s=16;s;s>>=1) x=__fadd_rn(x,__shfl_down_sync(0xffffffffu,x,s));
  return x;
}
// Preserve the pinned CUDA complex multiply contraction order before BF16 RNE.
__device__ void rotate(float x, float y, float c, float s, __nv_bfloat16* out) {
  out[0]=__float2bfloat16_rn(fmaf(x,c,-__fmul_rn(y,s)));
  out[1]=__float2bfloat16_rn(fmaf(y,c,__fmul_rn(x,s)));
}
template<int D, bool Quantize=false> __global__ void norm_kernel(const __nv_bfloat16* input,
    const __nv_bfloat16* weight, const float* freq, __nv_bfloat16* output) {
  const uint64_t row=blockIdx.x, base=row*D;
  const int t=threadIdx.x;
  float sum=0;
  for (int i=t;i<D;i+=256) {
    float x=__bfloat162float(input[base+i]);
    sum=__fadd_rn(sum,__fmul_rn(x,x));
  }
  __shared__ float partial[8], inverse;
  sum=sum_warp(sum);
  if (t%32==0) partial[t/32]=sum;
  __syncthreads();
  if (t<32) {
    sum=sum_warp(t<8?partial[t]:0);
    if (t==0) inverse=rsqrtf(__fadd_rn(sum/float(D),1e-20f));
  }
  __syncthreads();
  // Each thread owns a pair; normalization rounds BEFORE optional rotation.
  for (int i=t*2;i<D;i+=512) {
    const auto a=__float2bfloat16_rn(__fmul_rn(__fmul_rn(__bfloat162float(input[base+i]),inverse),__bfloat162float(weight[i])));
    const auto b=__float2bfloat16_rn(__fmul_rn(__fmul_rn(__bfloat162float(input[base+i+1]),inverse),__bfloat162float(weight[i+1])));
    __nv_bfloat16 pair[2]={a,b};
    if ((D==128 || D==512) && freq && i>=D-64) {
      const uint64_t f=row*64+i-(D-64);
      rotate(__bfloat162float(a),__bfloat162float(b),freq[f],freq[f+1],pair);
    }
    if constexpr (Quantize) {
      static_assert(D==512);
      const float x=__bfloat162float(pair[0]),y=__bfloat162float(pair[1]);
      float maximum=fmaxf(fabsf(x),fabsf(y));
      // Sixteen adjacent pairs are one official K32 activation group.
      for(int offset=8;offset;offset>>=1)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffffu,maximum,offset,16));
      const uint32_t bits=__float_as_uint(fmaxf(maximum,1e-4f)*(1.0f/448.0f));
      const int exponent=int((bits>>23)&255)-127+((bits&0x7fffff)!=0);
      const float scale=ldexpf(1.0f,exponent);
      pair[0]=__float2bfloat16_rn(float(__nv_fp8_e4m3(x/scale))*scale);
      pair[1]=__float2bfloat16_rn(float(__nv_fp8_e4m3(y/scale))*scale);
    }
    output[base+i]=pair[0];output[base+i+1]=pair[1];
  }
}
// Each thread owns one complex pair and loads both values before writing it.
// Exact in-place rotation is safe; partially overlapping vectors are rejected.
__global__ void rope_kernel(const __nv_bfloat16* input, const float* freq,
    __nv_bfloat16* output, int heads, int inverse) {
  const uint64_t vector=blockIdx.x, base=vector*512;
  const int i=threadIdx.x*2;
  if(i<448) {output[base+i]=input[base+i]; output[base+i+1]=input[base+i+1];}
  else {
    const uint64_t f=(vector/heads)*64+i-448;
    rotate(__bfloat162float(input[base+i]),__bfloat162float(input[base+i+1]),
        freq[f],inverse?-freq[f+1]:freq[f+1],output+base+i);
  }
}
}
extern "C" int32_t ds41rt_v41_attention_norm(const uint16_t* input, const uint16_t* weight,
    const float* freq, uint16_t* output, int32_t rows, int32_t dim, void* stream) {
  if (rows<1 || rows>4096 || (dim!=128 && dim!=512 && dim!=1280 && dim!=5120) || (freq && dim!=128 && dim!=512)) return cudaErrorInvalidValue;
  const uint64_t bytes=uint64_t(rows)*dim*2, w=uint64_t(dim)*2, f=uint64_t(rows)*256;
  if (!valid(input,bytes,2) || !valid(weight,w,2) || !valid(output,bytes,2) ||
      !disjoint(input,bytes,output,bytes) || !disjoint(weight,w,output,bytes) ||
      (freq && (!valid(freq,f,4) || !disjoint(freq,f,output,bytes)))) return cudaErrorInvalidValue;
#define LAUNCH(D) norm_kernel<D><<<rows,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(reinterpret_cast<const __nv_bfloat16*>(input),reinterpret_cast<const __nv_bfloat16*>(weight),freq,reinterpret_cast<__nv_bfloat16*>(output))
  if(dim==128) {LAUNCH(128);} else if(dim==512) {LAUNCH(512);} else if(dim==1280) {LAUNCH(1280);} else {LAUNCH(5120);}
#undef LAUNCH
  return cudaGetLastError();
}
extern "C" int32_t ds41rt_v41_attention_rope(const uint16_t* input, const float* freq,
    uint16_t* output, int32_t rows, int32_t heads, int32_t inverse, void* stream) {
  if(rows<1 || rows>4096 || (heads!=1 && heads!=64) || (inverse!=0 && inverse!=1)) return cudaErrorInvalidValue;
  const uint64_t bytes=uint64_t(rows)*heads*1024, f=uint64_t(rows)*256;
  if (!valid(input,bytes,2) || !valid(freq,f,4) || !valid(output,bytes,2) ||
      (input != output && !disjoint(input,bytes,output,bytes)) || !disjoint(freq,f,output,bytes)) return cudaErrorInvalidValue;
  rope_kernel<<<rows*heads,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(input),freq,reinterpret_cast<__nv_bfloat16*>(output),heads,inverse);
  return cudaGetLastError();
}

#include <cublas_v2.h>
#include <new>
namespace {
constexpr uint64_t kGroupedWorkspace = 4*1024*1024;
struct GroupedHandle { cublasHandle_t blas; void* workspace; int device; };
int32_t blas_status(cublasStatus_t s) { return s==CUBLAS_STATUS_SUCCESS ? 0 : -int32_t(s); }
}
extern "C" int32_t ds41rt_v41_grouped_output_create(void* workspace, uint64_t bytes, void** out) {
  if (!out) return cudaErrorInvalidValue;
  *out=nullptr;
  if (bytes<kGroupedWorkspace || !valid(workspace,kGroupedWorkspace,256)) return cudaErrorInvalidValue;
  auto* h=new(std::nothrow) GroupedHandle{};
  if (!h) return cudaErrorMemoryAllocation;
  auto status=cudaGetDevice(&h->device);
  if(status!=cudaSuccess) {delete h;return status;}
  auto result=cublasCreate(&h->blas);
  if(result!=CUBLAS_STATUS_SUCCESS) {delete h;return blas_status(result);}
  // Prevent reduced-precision intermediate reductions before the final BF16 output.
  result=cublasSetMathMode(h->blas,CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
  if(result!=CUBLAS_STATUS_SUCCESS) {cublasDestroy(h->blas);delete h;return blas_status(result);}
  h->workspace=workspace;*out=h;return 0;
}
extern "C" int32_t ds41rt_v41_grouped_output_destroy(void* opaque) {
  if(!opaque) return cudaErrorInvalidValue;
  auto* h=static_cast<GroupedHandle*>(opaque);
  auto status=cublasDestroy(h->blas);delete h;return blas_status(status);
}
extern "C" int32_t ds41rt_v41_grouped_output_launch(void* opaque, const uint16_t* input,
    const uint16_t* weight, uint16_t* output, int32_t rows, void* stream) {
  if(!opaque || rows<1 || rows>4096) return cudaErrorInvalidValue;
  auto* h=static_cast<GroupedHandle*>(opaque);
  int device; auto status=cudaGetDevice(&device);
  if(status!=cudaSuccess) return status;
  if(device!=h->device) return cudaErrorInvalidDevice;
  const uint64_t a=uint64_t(rows)*8*4096*2,w=uint64_t(8)*1024*4096*2,c=uint64_t(rows)*8*1024*2;
  if(!valid(input,a,2) || !valid(weight,w,2) || !valid(output,c,2) ||
      !disjoint(input,a,output,c) || !disjoint(weight,w,output,c) ||
      !disjoint(h->workspace,kGroupedWorkspace,input,a) ||
      !disjoint(h->workspace,kGroupedWorkspace,weight,w) ||
      !disjoint(h->workspace,kGroupedWorkspace,output,c)) return cudaErrorInvalidValue;
  auto result=cublasSetStream(h->blas,reinterpret_cast<cudaStream_t>(stream));
  if(result!=CUBLAS_STATUS_SUCCESS) return blas_status(result);
  // SetStream resets the workspace; restore caller-owned storage for every launch.
  result=cublasSetWorkspace(h->blas,h->workspace,kGroupedWorkspace);
  if(result!=CUBLAS_STATUS_SUCCESS) return blas_status(result);
  const float alpha=1,beta=0;
  // Column-major W^T [1024,4096] times strided token columns; group slices
  // interleave within each row but share no elements. No transpose/copy kernels.
  return blas_status(cublasGemmStridedBatchedEx(h->blas,CUBLAS_OP_T,CUBLAS_OP_N,
      1024,rows,4096,&alpha,weight,CUDA_R_16BF,4096,1024LL*4096,
      input,CUDA_R_16BF,8*4096,4096,&beta,output,CUDA_R_16BF,8*1024,1024,
      8,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

#include <cuda_fp8.h>
namespace {
__global__ void grouped_dequant_kernel(const uint8_t* input, const uint8_t* scales,
    __nv_bfloat16* output) {
  const uint64_t i=uint64_t(blockIdx.x)*256+threadIdx.x;
  if(i>=uint64_t(8192)*4096) return;
  __nv_fp8_e4m3 value;value.__x=input[i];
  const auto scale=scales[(i/4096/32)*128+(i%4096)/32];
  output[i]=__float2bfloat16_rn(float(value)*exp2f(int(scale)-127));
}
}
extern "C" int32_t ds41rt_v41_grouped_output_dequant(const uint8_t* input,
    const uint8_t* scales, uint16_t* output, void* stream) {
  constexpr uint64_t w=uint64_t(8192)*4096,s=w/1024,o=w*2;
  if(!valid(input,w,1)||!valid(scales,s,1)||!valid(output,o,2)||
      !disjoint(input,w,output,o)||!disjoint(scales,s,output,o)) return cudaErrorInvalidValue;
  grouped_dequant_kernel<<<w/256,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(
      input,scales,reinterpret_cast<__nv_bfloat16*>(output));
  return cudaGetLastError();
}

extern "C" int32_t ds41rt_v41_attention_kv(const uint16_t* input,const uint16_t* weight,
    const float* freq,uint16_t* output,int32_t rows,void* stream) {
  if(rows<1||rows>4096)return cudaErrorInvalidValue;
  const uint64_t b=uint64_t(rows)*1024,f=uint64_t(rows)*256;
  if(!valid(input,b,2)||!valid(weight,1024,2)||!valid(freq,f,4)||!valid(output,b,2)||
      !disjoint(input,b,output,b)||!disjoint(weight,1024,output,b)||!disjoint(freq,f,output,b))return cudaErrorInvalidValue;
  norm_kernel<512,true><<<rows,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(input),reinterpret_cast<const __nv_bfloat16*>(weight),freq,reinterpret_cast<__nv_bfloat16*>(output));
  return cudaGetLastError();
}

namespace {
__global__ void dspark_frequencies_kernel(const uint64_t* positions, float* output) {
  const uint64_t row=blockIdx.x;
  const int pair=threadIdx.x;
  // Match CUDA reference FP32 pow, reciprocal, outer product and polar;
  // dSpark pure window attention has base 10000 and no YaRN.
  const float inverse=__fdiv_rn(1.0f,powf(10000.0f,float(pair)/32.0f));
  const float phase=__fmul_rn(__ull2float_rn(positions[row]),inverse);
  output[row*64+pair*2]=cosf(phase);
  output[row*64+pair*2+1]=sinf(phase);
}
}
extern "C" int32_t ds41rt_v41_dspark_frequencies(const uint64_t* positions,
    float* output, int32_t rows, void* stream) {
  if(rows<1 || rows>4096) return cudaErrorInvalidValue;
  const uint64_t p=uint64_t(rows)*8, o=uint64_t(rows)*256;
  if(!valid(positions,p,8) || !valid(output,o,4) || !disjoint(positions,p,output,o))
    return cudaErrorInvalidValue;
  dspark_frequencies_kernel<<<rows,32,0,reinterpret_cast<cudaStream_t>(stream)>>>(positions,output);
  return cudaGetLastError();
}

namespace {
__global__ void dspark_tap_kernel(const __nv_bfloat16* input,
    __nv_bfloat16* output, uint64_t elements, int tap) {
  const uint64_t i=uint64_t(blockIdx.x)*256+threadIdx.x;
  if(i>=elements) return;
  const uint64_t row=i/5120,column=i%5120,base=row*20480+column;
  float sum=0.0f;
  #pragma unroll
  for(int stream=0;stream<4;++stream)
    sum=__fadd_rn(sum,__bfloat162float(input[base+uint64_t(stream)*5120]));
  output[row*15360+uint64_t(tap)*5120+column]=__float2bfloat16_rn(__fmul_rn(sum,0.25f));
}
}
extern "C" int32_t ds41rt_v41_dspark_tap(const uint16_t* input,
    uint16_t* output, int32_t rows, int32_t layer, void* stream) {
  if(rows<1 || rows>4096 || layer<37 || layer>39) return cudaErrorInvalidValue;
  const uint64_t in=uint64_t(rows)*40960,out=uint64_t(rows)*30720;
  if(!valid(input,in,2) || !valid(output,out,2) || !disjoint(input,in,output,out))
    return cudaErrorInvalidValue;
  const uint64_t elements=uint64_t(rows)*5120;
  dspark_tap_kernel<<<(elements+255)/256,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(input),reinterpret_cast<__nv_bfloat16*>(output),elements,layer-37);
  return cudaGetLastError();
}

namespace {
template<bool Draft>
__global__ void embed_kernel(const __nv_bfloat16* table, const int32_t* tokens,
    __nv_bfloat16* residual, float* pre) {
  const uint64_t row=blockIdx.x,request=Draft?row/5:row;
  const int32_t seed=tokens[request];
  const bool ok=seed>=0 && seed<129280;
  const int32_t token=Draft && row%5!=0?128799:seed;
  for(int col=threadIdx.x;col<5120;col+=256) {
    const auto value=ok?table[uint64_t(token)*5120+col]:__float2bfloat16_rn(0);
    #pragma unroll
    for(int hc=0;hc<4;++hc)residual[row*20480+uint64_t(hc)*5120+col]=value;
  }
  if(threadIdx.x<4)pre[row*4+threadIdx.x]=ok && threadIdx.x==0?1.0f:0.0f;
}
}
extern "C" int32_t ds41rt_v41_dspark_embed(const uint16_t* table, const int32_t* tokens,
    uint16_t* residual, float* pre, int32_t requests, void* stream) {
  if(requests<1 || requests>16)return cudaErrorInvalidValue;
  const uint64_t w=uint64_t(129280)*5120*2,t=uint64_t(requests)*4,
      r=uint64_t(requests)*5*40960,p=uint64_t(requests)*5*16;
  if(!valid(table,w,2)||!valid(tokens,t,4)||!valid(residual,r,2)||!valid(pre,p,4)||
      !disjoint(table,w,residual,r)||!disjoint(table,w,pre,p)||
      !disjoint(tokens,t,residual,r)||!disjoint(tokens,t,pre,p)||!disjoint(residual,r,pre,p))
    return cudaErrorInvalidValue;
  embed_kernel<true><<<requests*5,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(table),tokens,reinterpret_cast<__nv_bfloat16*>(residual),pre);
  return cudaGetLastError();
}

extern "C" int32_t ds41rt_v41_target_embed(const uint16_t* table, const int32_t* tokens,
    uint16_t* residual, float* pre, int32_t rows, void* stream) {
  if(rows<1 || rows>4096)return cudaErrorInvalidValue;
  const uint64_t w=uint64_t(129280)*5120*2,t=uint64_t(rows)*4,
      r=uint64_t(rows)*40960,p=uint64_t(rows)*16;
  if(!valid(table,w,2)||!valid(tokens,t,4)||!valid(residual,r,2)||!valid(pre,p,4)||
      !disjoint(table,w,residual,r)||!disjoint(table,w,pre,p)||
      !disjoint(tokens,t,residual,r)||!disjoint(tokens,t,pre,p)||!disjoint(residual,r,pre,p))
    return cudaErrorInvalidValue;
  embed_kernel<false><<<rows,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(table),tokens,reinterpret_cast<__nv_bfloat16*>(residual),pre);
  return cudaGetLastError();
}

namespace {
__global__ void dspark_terminal_layout_kernel(const __nv_bfloat16* residual,
    const float* pre, __nv_bfloat16* output, float* output_pre, int requests) {
  const uint64_t row=blockIdx.x;
  const uint64_t source=(row%requests)*5+row/requests;
  for(int col=threadIdx.x;col<20480;col+=256)output[row*20480+col]=residual[source*20480+col];
  if(threadIdx.x<4)output_pre[row*4+threadIdx.x]=pre[source*4+threadIdx.x];
}
}
extern "C" int32_t ds41rt_v41_dspark_terminal_layout(const uint16_t* residual,
    const float* pre, uint16_t* output, float* output_pre, int32_t requests, void* stream) {
  if(requests<1 || requests>16)return cudaErrorInvalidValue;
  const uint64_t r=uint64_t(requests)*5*40960,p=uint64_t(requests)*5*16;
  if(!valid(residual,r,2)||!valid(pre,p,4)||!valid(output,r,2)||!valid(output_pre,p,4)||
      !disjoint(residual,r,output,r)||!disjoint(residual,r,output_pre,p)||
      !disjoint(pre,p,output,r)||!disjoint(pre,p,output_pre,p)||!disjoint(output,r,output_pre,p))
    return cudaErrorInvalidValue;
  dspark_terminal_layout_kernel<<<requests*5,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(residual),pre,reinterpret_cast<__nv_bfloat16*>(output),output_pre,requests);
  return cudaGetLastError();
}

namespace {
__global__ void backbone_frequencies_kernel(const uint64_t* positions,float* output,bool compressed) {
  const uint64_t row=blockIdx.x;const int pair=threadIdx.x;
  float inverse=__fdiv_rn(1.0f,powf(compressed?160000.0f:10000.0f,float(pair)/32.0f));
  if(compressed) {
    // Official YaRN: original=65536, factor=16, beta_fast=32, beta_slow=1.
    // The reference's FP64 host corrected_dim floors/ceils to 15 and 25.
    // Tensor/scalar division in the pinned CUDA reference uses reciprocal multiply.
    const float ramp=fminf(1.0f,fmaxf(0.0f,__fmul_rn(float(pair)-15.0f,0.1f)));
    const float smooth=__fsub_rn(1.0f,ramp);
    inverse=__fadd_rn(__fmul_rn(__fdiv_rn(inverse,16.0f),__fsub_rn(1.0f,smooth)),__fmul_rn(inverse,smooth));
  }
  const float phase=__fmul_rn(__ull2float_rn(positions[row]),inverse);
  output[row*64+pair*2]=cosf(phase);output[row*64+pair*2+1]=sinf(phase);
}
}
extern "C" int32_t ds41rt_v41_backbone_frequencies(const uint64_t* positions,
    float* output,int32_t rows,int32_t layer,void* stream) {
  if(rows<1 || rows>4096 || layer<0 || layer>=40)return cudaErrorInvalidValue;
  const uint64_t p=uint64_t(rows)*8,o=uint64_t(rows)*256;
  if(!valid(positions,p,8)||!valid(output,o,4)||!disjoint(positions,p,output,o))return cudaErrorInvalidValue;
  backbone_frequencies_kernel<<<rows,32,0,reinterpret_cast<cudaStream_t>(stream)>>>(positions,output,layer>=2);
  return cudaGetLastError();
}
