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
