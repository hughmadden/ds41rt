#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Asynchronous allocation-free operations; buffers are on the stream device,
// initialized and live through completion/replay. Output is disjoint from inputs.
// BF16 [rows,dim] input/output and [dim] norm weight, eps=1e-6.
// dim is 512,1280,5120. Optional FP32 complex frequencies [rows,32,2]
// rotate the final 64 coordinates AFTER BF16 normalization rounding (dim=512 only).
int32_t ds41rt_v41_attention_norm(const uint16_t* input, const uint16_t* weight,
    const float* frequencies, uint16_t* output, int32_t rows, int32_t dim, void* stream);
// BF16 [rows,heads,512], heads=1 or64; FP32 complex frequencies [rows,32,2].
// Adjacent pairs in the final64 dimensions; inverse=0 or1 (conjugate frequencies).
int32_t ds41rt_v41_attention_rope(const uint16_t* input, const float* frequencies,
    uint16_t* output, int32_t rows, int32_t heads, int32_t inverse, void* stream);
// Grouped BF16 wo_a: [rows,8,4096] x [8,1024,4096] -> [rows,8,1024].
// Caller owns a distinct 256-byte-aligned workspace of at least 4 MiB, which
// stays live through handle destruction. One handle per exclusively used stream.
// Warm launch before capture; drain launches/graphs before destroy/free.
int32_t ds41rt_v41_grouped_output_create(void* workspace, uint64_t bytes, void** handle);
int32_t ds41rt_v41_grouped_output_destroy(void* handle);
int32_t ds41rt_v41_grouped_output_launch(void* handle, const uint16_t* input,
    const uint16_t* weight, uint16_t* output, int32_t rows, void* stream);
// Official FP8 [8192,4096] and UE8 [256,128] -> resident BF16 [8192,4096].
int32_t ds41rt_v41_grouped_output_dequant(const uint8_t* weight,
    const uint8_t* scales, uint16_t* output, void* stream);
// Fused KV norm -> BF16 -> RoPE -> BF16 -> K32 UE8 FP8 quantize/dequantize.
// Shapes as attention_norm dim512, mandatory frequencies; preserves both rounds.
int32_t ds41rt_v41_attention_kv(const uint16_t* input,const uint16_t* weight,
    const float* frequencies,uint16_t* output,int32_t rows,void* stream);
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
// U64 absolute positions [rows] -> FP32 complex frequencies [rows,32,2].
// dSpark fixed RoPE64/base10000/no-YaRN; output is disjoint from positions.
int32_t ds41rt_v41_dspark_frequencies(const uint64_t* positions,
    float* output, int32_t rows, void* stream);
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
// BF16 [rows,4,5120] attention-input stream mean -> the layer's slice of
// BF16 [rows,15360], layer=37/38/39; the other two slices stay unchanged.
int32_t ds41rt_v41_dspark_tap(const uint16_t* input, uint16_t* output,
    int32_t rows, int32_t layer, void* stream);
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
// Shared BF16 [129280,5120] table and I32 [requests] seeds -> BF16
// [requests,5,4,5120] residual and FP32 [requests,5,4] one-hot pre-mix.
// Invalid device seed IDs zero that request's complete output; hosts validate IDs.
int32_t ds41rt_v41_dspark_embed(const uint16_t* table, const int32_t* tokens,
    uint16_t* residual, float* pre, int32_t requests, void* stream);
#ifdef __cplusplus
}
#endif
