#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Finite BF16 [rows,512] -> E4M3 [rows,512], E8M0 [rows,16], K32 groups.
// Optional complex FP32 frequencies [rows,32,2] rotate the final 64 coordinates,
// rounded to BF16 before quantization. Rotated values must remain finite.
// Scale: ceil-power-of-two(max(amax,1e-4)/448). Outputs are disjoint from inputs.
int32_t ds41rt_v41_kv_pack(const uint16_t* input,const float* frequencies,
    uint8_t* values,uint8_t* scales,int32_t rows,void* stream);
// Scatter accepted rows into persistent KV. U64 destinations [rows] must be
// unique among valid destinations; IDs >=capacity skip. No cache metadata is
// published here. The caller serializes writes and publishes after draining.
int32_t ds41rt_v41_kv_store(const uint8_t* values,const uint8_t* scales,
    const uint64_t* destinations,uint8_t* cache,uint8_t* cache_scales,
    int32_t rows,uint64_t capacity,void* stream);
#ifdef __cplusplus
}
#endif
