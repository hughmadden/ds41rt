#include "ds41rt_v41_vision.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <cublas_v2.h>
#include <stdint.h>
#include <new>

namespace {
using B = __nv_bfloat16;
constexpr uint64_t workspace_bytes=4*1024*1024;
constexpr int max_rows=9216, heads=16, channels=1024, tile=128;
struct Handle { cublasHandle_t blas; void* workspace; int device; };
int status(cublasStatus_t s) { return s==CUBLAS_STATUS_SUCCESS?0:-int(s); }
bool valid(const void* p,uint64_t bytes,int alignment) {
  auto address=reinterpret_cast<uintptr_t>(p);
  return address && address%alignment==0 && address<=UINTPTR_MAX-bytes;
}
bool apart(const void* a,uint64_t na,const void* b,uint64_t nb) {
  auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);
  return x+na<=y || y+nb<=x;
}
int bind(Handle* h,void* stream) {
  int device;auto result=cudaGetDevice(&device);if(result)return result;
  if(device!=h->device)return cudaErrorInvalidDevice;
  auto s=cublasSetStream(h->blas,reinterpret_cast<cudaStream_t>(stream));if(s)return status(s);
  return status(cublasSetWorkspace(h->blas,h->workspace,workspace_bytes));
}
__device__ float sum(float x) {
  for(int d=16;d;d>>=1)x=__fadd_rn(x,__shfl_down_sync(0xffffffff,x,d));return x;
}
__global__ void bias_cast(const float* x,const B* bias,B* y,int width,uint64_t count) {
  auto i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<count)y[i]=__float2bfloat16_rn(__fadd_rn(x[i],__bfloat162float(bias[i%width])));
}
__global__ void norm_kernel(const B* x,const B* w,B* y) {
  const int t=threadIdx.x;const uint64_t base=uint64_t(blockIdx.x)*1024;
  float s=0;for(int c=t;c<1024;c+=256) {float v=__bfloat162float(x[base+c]);s=__fadd_rn(s,__fmul_rn(v,v));}
  __shared__ float partial[8],inverse;
  s=sum(s);if(t%32==0)partial[t/32]=s;__syncthreads();
  if(t<32) {s=sum(t<8?partial[t]:0);if(t==0)inverse=rsqrtf(__fadd_rn(s/1024.0f,1e-6f));}
  __syncthreads();
  for(int c=t;c<1024;c+=256)y[base+c]=__float2bfloat16_rn(__fmul_rn(
      __fmul_rn(__bfloat162float(x[base+c]),inverse),__bfloat162float(w[c])));
}
__global__ void element_kernel(const B* x,const B* other,B* y,uint64_t count,int width,int mode) {
  auto i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=count)return;
  float value;
  if(mode==1) {
    const uint64_t base=(i/width)*2*width+i%width;
    const float gate=__bfloat162float(x[base]);
    const float activated=__bfloat162float(__float2bfloat16_rn(gate/(1.0f+expf(-gate))));
    value=__fmul_rn(activated,__bfloat162float(x[base+width]));
  } else {
    value=__bfloat162float(x[i]);
    if(mode==0)value=__fadd_rn(value,__bfloat162float(other[i]));
    else value=__fmul_rn(__fmul_rn(value,0.5f),__fadd_rn(1.0f,erff(__fmul_rn(value,0.7071067811865476f))));
  }
  y[i]=__float2bfloat16_rn(value);
}
__global__ void rope_kernel(const B* x,const float* inv,B* q,B* k,B* v,int width) {
  const uint64_t row=blockIdx.x;
  for(int col=threadIdx.x;col<1024;col+=256) {
    const int local=col%64,freq=local%32;
    const int position=freq<16?row/width:row%width;
    const float angle=__fmul_rn(float(position),inv[freq%16]);
    float sine,cosine;sincosf(angle,&sine,&cosine);
    const int other=local<32?col+32:col-32;
    const float sign=local<32?-1.0f:1.0f;
    const uint64_t base=row*3072,dst=row*1024+col;
    q[dst]=__float2bfloat16_rn(__fadd_rn(__fmul_rn(__bfloat162float(x[base+col]),cosine),
        __fmul_rn(sign,__fmul_rn(__bfloat162float(x[base+other]),sine))));
    k[dst]=__float2bfloat16_rn(__fadd_rn(__fmul_rn(__bfloat162float(x[base+1024+col]),cosine),
        __fmul_rn(sign,__fmul_rn(__bfloat162float(x[base+1024+other]),sine))));
    v[dst]=x[base+2048+col];
  }
}
__global__ void to_float(const B* x,float* y,uint64_t count) {
  auto i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<count)y[i]=__bfloat162float(x[i]);
}
__global__ void to_bf16(const float* x,B* y,uint64_t count) {
  auto i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<count)y[i]=__float2bfloat16_rn(x[i]);
}
__global__ void softmax_kernel(float* scores,B* probabilities,int rows) {
  const int t=threadIdx.x;const uint64_t base=uint64_t(blockIdx.x)*rows;
  __shared__ float partial[8],maximum,inverse;
  float m=-CUDART_INF_F;for(int i=t;i<rows;i+=256)m=fmaxf(m,scores[base+i]);
  for(int d=16;d;d>>=1)m=fmaxf(m,__shfl_down_sync(0xffffffff,m,d));
  if(t%32==0)partial[t/32]=m;__syncthreads();
  if(t<32) {m=t<8?partial[t]:-CUDART_INF_F;for(int d=16;d;d>>=1)m=fmaxf(m,__shfl_down_sync(0xffffffff,m,d));if(t==0)maximum=m;}
  __syncthreads();
  float s=0;for(int i=t;i<rows;i+=256)s+=expf(scores[base+i]-maximum);
  s=sum(s);if(t%32==0)partial[t/32]=s;__syncthreads();
  if(t<32) {s=sum(t<8?partial[t]:0);if(t==0)inverse=1.0f/s;}
  __syncthreads();
  for(int i=t;i<rows;i+=256) {
    const float probability=expf(scores[base+i]-maximum)*inverse;
    if(probabilities)probabilities[base+i]=__float2bfloat16_rn(probability);
    else scores[base+i]=probability;
  }
}
__global__ void merge_kernel(const B* x,B* y,int height,int width,int merged_width) {
  const uint64_t row=blockIdx.x;
  for(int col=threadIdx.x;col<9216;col+=256) {
    const int channel=col/9,h=(row/merged_width)*3+(col/3)%3,w=(row%merged_width)*3+col%3;
    y[row*9216+col]=(h<height && w<width)?x[(uint64_t(h)*width+w)*1024+channel]:__float2bfloat16_rn(0);
  }
}
__global__ void image_embed_kernel(const B* features,const uint32_t* indices,B* residual,int rows) {
  const int image=blockIdx.x;const uint32_t row=indices[image];
  if(row>=uint32_t(rows))return;
  for(int col=threadIdx.x;col<5120;col+=256) {
    const B value=features[uint64_t(image)*5120+col];
    for(int hc=0;hc<4;++hc)residual[(uint64_t(row)*4+hc)*5120+col]=value;
  }
}
__global__ void span_kernel(const B* features,const B* start,const B* newline,const B* end,B* y,int width,int tokens) {
  const int row=blockIdx.x;
  const B* source=row==0?start:(row==tokens-1?end:((row-1)%(width+1)==width?newline:
      features+uint64_t((row-1)/(width+1)*width+(row-1)%(width+1))*5120));
  for(int col=threadIdx.x;col<5120;col+=256)y[uint64_t(row)*5120+col]=source[col];
}
}

extern "C" int32_t ds41rt_v41_vision_create(void* workspace,uint64_t bytes,void** output) {
  if(!output)return cudaErrorInvalidValue;*output=nullptr;
  if(bytes<workspace_bytes || !valid(workspace,workspace_bytes,256))return cudaErrorInvalidValue;
  auto* h=new(std::nothrow)Handle{};if(!h)return cudaErrorMemoryAllocation;
  auto result=cudaGetDevice(&h->device);if(result){delete h;return result;}
  auto s=cublasCreate(&h->blas);if(s){delete h;return status(s);}
  s=cublasSetMathMode(h->blas,CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
  if(s){cublasDestroy(h->blas);delete h;return status(s);}
  h->workspace=workspace;*output=h;return 0;
}
extern "C" int32_t ds41rt_v41_vision_destroy(void* opaque) {
  if(!opaque)return cudaErrorInvalidValue;auto* h=static_cast<Handle*>(opaque);
  auto s=cublasDestroy(h->blas);delete h;return status(s);
}
extern "C" int32_t ds41rt_v41_vision_linear(void* opaque,const uint16_t* x,const uint16_t* w,
    const uint16_t* bias,float* scratch,uint16_t* y,int rows,int in,int out,void* stream) {
  if(!opaque || rows<1 || rows>max_rows || in<1 || in>9216 || out<1 || out>5632)return cudaErrorInvalidValue;
  auto* h=static_cast<Handle*>(opaque);
  const uint64_t xb=uint64_t(rows)*in*2,wb=uint64_t(out)*in*2,yb=uint64_t(rows)*out*2;
  const void* p[]={x,w,y,h->workspace,bias,scratch};const uint64_t sizes[]={xb,wb,yb,workspace_bytes,uint64_t(out)*2,yb*2};
  for(int i=0;i<(bias?6:4);++i) {
    if(!valid(p[i],sizes[i],i==5?4:2))return cudaErrorInvalidValue;
    for(int j=0;j<i;++j)if(!apart(p[i],sizes[i],p[j],sizes[j]))return cudaErrorInvalidValue;
  }
  auto result=bind(h,stream);if(result)return result;
  const float alpha=1,beta=0;
  auto s=cublasGemmEx(h->blas,CUBLAS_OP_T,CUBLAS_OP_N,out,rows,in,&alpha,w,CUDA_R_16BF,in,
      x,CUDA_R_16BF,in,&beta,bias?static_cast<void*>(scratch):y,bias?CUDA_R_32F:CUDA_R_16BF,
      out,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  if(s)return status(s);
  if(bias)bias_cast<<<(uint64_t(rows)*out+255)/256,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(scratch,
      reinterpret_cast<const B*>(bias),reinterpret_cast<B*>(y),out,uint64_t(rows)*out);
  return cudaGetLastError();
}
extern "C" int32_t ds41rt_v41_vision_norm(const uint16_t* x,const uint16_t* weight,uint16_t* y,int rows,void* stream) {
  const uint64_t bytes=uint64_t(rows)*2048;
  if(rows<1 || rows>max_rows || !valid(x,bytes,2)||!valid(weight,2048,2)||!valid(y,bytes,2)||
      !apart(x,bytes,y,bytes)||!apart(weight,2048,y,bytes))return cudaErrorInvalidValue;
  norm_kernel<<<rows,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(reinterpret_cast<const B*>(x),reinterpret_cast<const B*>(weight),reinterpret_cast<B*>(y));
  return cudaGetLastError();
}
extern "C" int32_t ds41rt_v41_vision_elementwise(const uint16_t* x,const uint16_t* other,uint16_t* y,int rows,int width,int mode,void* stream) {
  if(rows<1 || rows>max_rows || mode<0 || mode>2 || width!=(mode==0?1024:(mode==1?2816:5120)))return cudaErrorInvalidValue;
  const uint64_t bytes=uint64_t(rows)*width*2,xb=bytes*(mode==1?2:1);
  if(!valid(x,xb,2)||!valid(y,bytes,2)||(!apart(x,xb,y,bytes) && !(mode!=1 && x==y)))return cudaErrorInvalidValue;
  if(mode==0 && (!valid(other,bytes,2)||(!apart(other,bytes,y,bytes) && other!=y)))return cudaErrorInvalidValue;
  element_kernel<<<(uint64_t(rows)*width+255)/256,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const B*>(x),reinterpret_cast<const B*>(other),reinterpret_cast<B*>(y),uint64_t(rows)*width,width,mode);
  return cudaGetLastError();
}
extern "C" int32_t ds41rt_v41_vision_rope(const uint16_t* qkv,const float* inv,uint16_t* q,uint16_t* k,uint16_t* v,int height,int width,void* stream) {
  const int64_t rows=int64_t(height)*width;if(height<1 || width<1 || rows>max_rows)return cudaErrorInvalidValue;
  const uint64_t bytes=rows*2048;const void* p[]={qkv,inv,q,k,v};const uint64_t sizes[]={bytes*3,64,bytes,bytes,bytes};
  for(int i=0;i<5;++i){if(!valid(p[i],sizes[i],i==1?4:2))return cudaErrorInvalidValue;for(int j=0;j<i;++j)if(!apart(p[i],sizes[i],p[j],sizes[j]))return cudaErrorInvalidValue;}
  rope_kernel<<<rows,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(reinterpret_cast<const B*>(qkv),inv,
      reinterpret_cast<B*>(q),reinterpret_cast<B*>(k),reinterpret_cast<B*>(v),width);return cudaGetLastError();
}
extern "C" int32_t ds41rt_v41_vision_attention(void* opaque,const uint16_t* q,const uint16_t* k,
    const uint16_t* v,float* scores,float* values,float* output,uint16_t* y,int rows,int fp32_probabilities,void* stream) {
  if(!opaque || rows<1 || rows>max_rows || (fp32_probabilities!=0 && fp32_probabilities!=1))return cudaErrorInvalidValue;
  auto* h=static_cast<Handle*>(opaque);const uint64_t bytes=uint64_t(rows)*2048,sb=uint64_t(rows)*tile*heads*4;
  const void* p[]={q,k,v,scores,values,output,y,h->workspace};const uint64_t sizes[]={bytes,bytes,bytes,sb,bytes*2,bytes*2,bytes,workspace_bytes};
  for(int i=0;i<8;++i){if(!valid(p[i],sizes[i],i>=3 && i<=5?4:2))return cudaErrorInvalidValue;for(int j=0;j<i;++j)if(!apart(p[i],sizes[i],p[j],sizes[j]))return cudaErrorInvalidValue;}
  auto result=bind(h,stream);if(result)return result;
  const float scale=0.125f,one=1,zero=0;
  if(fp32_probabilities) {
    to_float<<<(uint64_t(rows)*1024+255)/256,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(reinterpret_cast<const B*>(v),values,uint64_t(rows)*1024);
    auto cast_status=cudaGetLastError();if(cast_status)return cast_status;
  }
  for(int start=0;start<rows;start+=tile) {
    const int live=rows-start<tile?rows-start:tile;const long long stride=static_cast<long long>(rows)*live;
    auto s=cublasGemmStridedBatchedEx(h->blas,CUBLAS_OP_T,CUBLAS_OP_N,rows,live,64,&scale,
        k,CUDA_R_16BF,1024,64,q+uint64_t(start)*1024,CUDA_R_16BF,1024,64,&zero,
        scores,CUDA_R_32F,rows,stride,heads,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if(s)return status(s);
    softmax_kernel<<<heads*live,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(scores,fp32_probabilities?nullptr:reinterpret_cast<B*>(values),rows);
    auto e=cudaGetLastError();if(e)return e;
    if(fp32_probabilities) {
      s=cublasGemmStridedBatchedEx(h->blas,CUBLAS_OP_N,CUBLAS_OP_N,64,live,rows,&one,
          values,CUDA_R_32F,1024,64,scores,CUDA_R_32F,rows,stride,&zero,
          output+uint64_t(start)*1024,CUDA_R_32F,1024,64,heads,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);
    } else {
      // Reuse value scratch for BF16 probabilities; both occupy rows*4096 bytes.
      s=cublasGemmStridedBatchedEx(h->blas,CUBLAS_OP_N,CUBLAS_OP_N,64,live,rows,&one,
          v,CUDA_R_16BF,1024,64,values,CUDA_R_16BF,rows,stride,&zero,
          y+uint64_t(start)*1024,CUDA_R_16BF,1024,64,heads,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    if(s)return status(s);
  }
  if(fp32_probabilities)to_bf16<<<(uint64_t(rows)*1024+255)/256,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(output,reinterpret_cast<B*>(y),uint64_t(rows)*1024);
  return cudaGetLastError();
}
extern "C" int32_t ds41rt_v41_vision_merge(const uint16_t* x,uint16_t* y,int height,int width,void* stream) {
  if(height<1 || width<1 || height>max_rows || width>max_rows)return cudaErrorInvalidValue;
  const int64_t rows=int64_t(height)*width,merged=int64_t((height+2)/3)*((width+2)/3);
  if(rows>max_rows || merged>1024)return cudaErrorInvalidValue;
  if(!valid(x,rows*2048,2)||!valid(y,merged*18432,2)||!apart(x,rows*2048,y,merged*18432))return cudaErrorInvalidValue;
  merge_kernel<<<merged,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(reinterpret_cast<const B*>(x),reinterpret_cast<B*>(y),height,width,(width+2)/3);return cudaGetLastError();
}
extern "C" int32_t ds41rt_v41_vision_span(const uint16_t* features,const uint16_t* start,const uint16_t* newline,const uint16_t* end,uint16_t* y,int height,int width,void* stream) {
  if(height<1 || width<1 || height>1024 || width>1024)return cudaErrorInvalidValue;
  const int tokens=height*(width+1)+2;if(tokens>1024)return cudaErrorInvalidValue;
  const uint64_t out=uint64_t(tokens)*10240;
  const void* p[]={features,start,newline,end};const uint64_t sizes[]={uint64_t(height)*width*10240,10240,10240,10240};
  if(!valid(y,out,2))return cudaErrorInvalidValue;
  for(int i=0;i<4;++i)if(!valid(p[i],sizes[i],2)||!apart(p[i],sizes[i],y,out))return cudaErrorInvalidValue;
  span_kernel<<<tokens,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(reinterpret_cast<const B*>(features),reinterpret_cast<const B*>(start),
      reinterpret_cast<const B*>(newline),reinterpret_cast<const B*>(end),reinterpret_cast<B*>(y),width,tokens);return cudaGetLastError();
}

extern "C" int32_t ds41rt_v41_vision_embed(const uint16_t* features,const uint32_t* indices,
    uint16_t* residual,int image_rows,int rows,void* stream) {
  if(image_rows<1 || rows<1 || image_rows>rows || rows>4096)return cudaErrorInvalidValue;
  const uint64_t fb=uint64_t(image_rows)*10240,ib=uint64_t(image_rows)*4,rb=uint64_t(rows)*40960;
  if(!valid(features,fb,2)||!valid(indices,ib,4)||!valid(residual,rb,2)||
      !apart(features,fb,residual,rb)||!apart(indices,ib,residual,rb))return cudaErrorInvalidValue;
  image_embed_kernel<<<image_rows,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const B*>(features),indices,reinterpret_cast<B*>(residual),rows);
  return cudaGetLastError();
}
