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
#ifdef __cplusplus
}
#endif
