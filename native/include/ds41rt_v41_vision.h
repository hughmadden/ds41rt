#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Serialized handle with caller-owned 256-byte aligned 4 MiB BLAS workspace.
// All image operations use BF16 storage, FP32 accumulation, and at most 9216
// patches. Inputs/output/workspaces must remain live until the stream drains.
int32_t ds41rt_v41_vision_create(void* workspace,uint64_t bytes,void** handle);
int32_t ds41rt_v41_vision_destroy(void* handle);
// X[rows,in] W[out,in] -> Y[rows,out]. Optional BF16 bias[out] is added to
// FP32 GEMM output before the single BF16 rounding; scratch needs rows*out*4.
// Without bias scratch is unused. Input/weight/output/scratch must be disjoint.
int32_t ds41rt_v41_vision_linear(void* handle,const uint16_t* x,const uint16_t* w,
    const uint16_t* bias,float* scratch,uint16_t* y,int rows,int in,int out,void* stream);
int32_t ds41rt_v41_vision_norm(const uint16_t* x,const uint16_t* weight,
    uint16_t* y,int rows,void* stream);
// mode 0: add x+other, width=1024; 1: BF16 silu(gate)*up with distinct
// intermediate BF16 rounding, x[rows,2*2816] -> y[rows,2816];
// 2: exact-erf GELU, width=5120. Add/GELU may be exactly in-place.
int32_t ds41rt_v41_vision_elementwise(const uint16_t* x,const uint16_t* other,
    uint16_t* y,int rows,int width,int mode,void* stream);
// QKV[rows,3072] -> Q,K,V[rows,1024]. Split-half 2D RoPE, H then W;
// inv_freq is FP32[16], geometry height*width == rows. Outputs disjoint.
int32_t ds41rt_v41_vision_rope(const uint16_t* qkv,const float* inv_freq,
    uint16_t* q,uint16_t* k,uint16_t* v,int height,int width,void* stream);
// Bidirectional per-image attention with 16 heads of width 64. Tiles at most
// 128 queries: score/probability scratch 16*128*rows FP32 (in-place softmax),
// value/output scratch rows*1024 FP32 each. Default fp32_probabilities=0 uses
// BF16 probabilities and Tensor Core PV with FP32 accumulation; value scratch
// holds those probabilities. fp32_probabilities=1 is a numerical control using
// FP32 probabilities and PV. All buffer spans are disjoint.
int32_t ds41rt_v41_vision_attention(void* handle,const uint16_t* q,const uint16_t* k,
    const uint16_t* v,float* scores,float* values,float* output,uint16_t* y,int rows,int fp32_probabilities,void* stream);
// Right/bottom zero-padding and channel-major 3x3 unfold, not pixel shuffle.
int32_t ds41rt_v41_vision_merge(const uint16_t* x,uint16_t* y,int height,int width,void* stream);
// Aligned [llm_h*llm_w,5120] -> complete delimiter/newline span [tokens,5120].
int32_t ds41rt_v41_vision_span(const uint16_t* features,const uint16_t* start,
    const uint16_t* newline,const uint16_t* end,uint16_t* y,int llm_h,int llm_w,void* stream);
// Replace selected target rows with BF16 image features, replicated across all
// four mHC lanes. Features[image_rows,5120], indices[image_rows], residual[rows,4,5120].
// Indices must be unique and less than rows; buffers remain live through stream completion.
int32_t ds41rt_v41_vision_embed(const uint16_t* features,const uint32_t* indices,
    uint16_t* residual,int image_rows,int rows,void* stream);
#ifdef __cplusplus
}
#endif
