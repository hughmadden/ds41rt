#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Caller-owned, 256-byte-aligned 4MiB workspace. Warm before capture; one handle
// per serialized wave. Keep workspace/weights/buffers live through graph teardown.
int32_t ds41rt_v41_compressor_create(void* workspace, uint64_t bytes, void** handle);
int32_t ds41rt_v41_compressor_destroy(void* handle);
// BF16 [rows,5120] x BF16 [512,5120]. ratio=2: FP32 [rows,512] output;
// ratio=1: BF16 [rows,512] output. FP32 accumulation, no reduced-precision reduction.
int32_t ds41rt_v41_compressor_project(void* handle, const uint16_t* input,
    const uint16_t* weight, void* output, int32_t rows, int32_t ratio, void* stream);
// Shared compressor handle/workspace, serialized on the same stream.
// BF16 unrotated latent [rows,512] x BF16 index weight [128,512] -> BF16 [rows,128].
int32_t ds41rt_v41_index_key_project(void* handle,const uint16_t* input,
    const uint16_t* weight,uint16_t* output,int32_t rows,void* stream);
// Finite BF16 index vectors [rows,128] -> E2M1 packed bytes [rows,64] and
// E8M0 scale bytes [rows,4], group size 32. Adjacent even column is low nibble.
// All three spans must be disjoint. No allocation or persistent cache mutation.
// rows <= 131072 also supports flattening 4096 queries with 32 index heads.
int32_t ds41rt_v41_index_pack(const uint16_t* input,uint8_t* packed,
    uint8_t* scales,int32_t rows,void* stream);
// Scatter proposal rows into physical index-cache rows. U64 destinations [rows]
// must be unique among valid entries; >= capacity (including UINT64_MAX) skips.
// packed/scales have 64/4 bytes per row. All input/output spans are disjoint.
int32_t ds41rt_v41_index_store(const uint8_t* packed,const uint8_t* scales,
    const uint64_t* destinations,uint8_t* cache,uint8_t* cache_scales,
    int32_t rows,uint64_t capacity,void* stream);
// FP32 projected KV/scores [rows,512], committed pending KV/scores [slots,512],
// U64 predecessor descriptors [rows], BF16 norm weight [512] -> BF16 [rows,512].
// UINT64_MAX: incomplete group, emit zero. Otherwise [0,slots) reads committed
// pending state; [slots,slots+row) reads an earlier row of these projections.
// Descriptors must encode chronological adjacent pairs, never reuse request state
// across leases. Invalid predecessor values emit zero without out-of-bounds reads.
// Does not mutate projected or committed state; commit accepted prefixes separately.
// Softmax pooling rounds to BF16 BEFORE RMS normalization (epsilon 1e-20).
int32_t ds41rt_v41_compressor_pool(const float* kv, const float* scores,
    const float* pending_kv, const float* pending_scores, const uint64_t* predecessors,
    const uint16_t* norm_weight, uint16_t* output, int32_t rows, int32_t slots, void* stream);
#ifdef __cplusplus
}
#endif
