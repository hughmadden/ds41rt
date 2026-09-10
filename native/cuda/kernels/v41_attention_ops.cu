#include <cuda_runtime.h>
#include <cuda_bf16.h>
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
template<int D> __global__ void norm_kernel(const __nv_bfloat16* input,
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
    if (t==0) inverse=rsqrtf(__fadd_rn(sum/float(D),1e-6f));
  }
  __syncthreads();
  // Each thread owns a pair; normalization rounds BEFORE optional rotation.
  for (int i=t*2;i<D;i+=512) {
    const auto a=__float2bfloat16_rn(__fmul_rn(__fmul_rn(__bfloat162float(input[base+i]),inverse),__bfloat162float(weight[i])));
    const auto b=__float2bfloat16_rn(__fmul_rn(__fmul_rn(__bfloat162float(input[base+i+1]),inverse),__bfloat162float(weight[i+1])));
    if (D==512 && freq && i>=448) {
      const uint64_t f=row*64+i-448;
      rotate(__bfloat162float(a),__bfloat162float(b),freq[f],freq[f+1],output+base+i);
    } else {output[base+i]=a; output[base+i+1]=b;}
  }
}
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
  if (rows<1 || rows>4096 || (dim!=512 && dim!=1280 && dim!=5120) || (freq && dim!=512)) return cudaErrorInvalidValue;
  const uint64_t bytes=uint64_t(rows)*dim*2, w=uint64_t(dim)*2, f=uint64_t(rows)*256;
  if (!valid(input,bytes,2) || !valid(weight,w,2) || !valid(output,bytes,2) ||
      !disjoint(input,bytes,output,bytes) || !disjoint(weight,w,output,bytes) ||
      (freq && (!valid(freq,f,4) || !disjoint(freq,f,output,bytes)))) return cudaErrorInvalidValue;
#define LAUNCH(D) norm_kernel<D><<<rows,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(reinterpret_cast<const __nv_bfloat16*>(input),reinterpret_cast<const __nv_bfloat16*>(weight),freq,reinterpret_cast<__nv_bfloat16*>(output))
  if(dim==512) {LAUNCH(512);} else if(dim==1280) {LAUNCH(1280);} else {LAUNCH(5120);}
#undef LAUNCH
  return cudaGetLastError();
}
extern "C" int32_t ds41rt_v41_attention_rope(const uint16_t* input, const float* freq,
    uint16_t* output, int32_t rows, int32_t heads, int32_t inverse, void* stream) {
  if(rows<1 || rows>4096 || (heads!=1 && heads!=64) || (inverse!=0 && inverse!=1)) return cudaErrorInvalidValue;
  const uint64_t bytes=uint64_t(rows)*heads*1024, f=uint64_t(rows)*256;
  if (!valid(input,bytes,2) || !valid(freq,f,4) || !valid(output,bytes,2) ||
      !disjoint(input,bytes,output,bytes) || !disjoint(freq,f,output,bytes)) return cudaErrorInvalidValue;
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
