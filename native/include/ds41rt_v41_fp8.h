#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* All entry points return raw CUDA status; caller owns storage and stream ordering.
 * Weights remain contiguous FP8 [25600,6144]; native scales are UE8M0 [800,192]. */
typedef struct {
  uint32_t abi_version, capacity_rows, input_dim, output_dim;
  uint64_t scratch_bytes, values_offset, row_scales_offset, mma_scales_offset;
  uint64_t packed_weight_scale_bytes;
} ds41rt_v41_fp8_info_t;
/* Matrix entry points additionally cover shared FFN [2304,5120] and [5120,2304].
 * Shapes are explicit (K input, N output); only compiled capacities are admitted. */
int32_t ds41rt_v41_fp8_matrix_info(int32_t capacity, int32_t k, int32_t n, ds41rt_v41_fp8_info_t* out);
int32_t ds41rt_v41_fp8_matrix_initialize(int32_t capacity, int32_t k, int32_t n, void** out);
int32_t ds41rt_v41_fp8_matrix_pack_scales(const uint8_t* source, uint8_t* destination,
    int32_t k, int32_t n, void* stream);
/* BF16 [rows,2304] inputs/output; FP32 asymmetric limit-10 SwiGLU, BF16 RNE output. */
int32_t ds41rt_v41_shared_swiglu(const uint16_t* gate, const uint16_t* up,
    uint16_t* output, int32_t rows, void* stream);
int32_t ds41rt_v41_fp8_info(int32_t capacity, ds41rt_v41_fp8_info_t* out);
int32_t ds41rt_v41_fp8_initialize(int32_t capacity, void** out);
/* Scratch is initialized once before capture/replay; alpha is a separate float. */
int32_t ds41rt_v41_fp8_initialize_scratch(void* kernel, void* scratch, uint64_t bytes,
    float* alpha, void* stream);
/* Source 153600 bytes, destination 4915200 bytes, disjoint CUDA allocations. */
int32_t ds41rt_v41_fp8_pack_scales(const uint8_t* source, uint8_t* destination, void* stream);
/* Buffers must be aligned and disjoint, sized for rows (scratch for capacity),
 * and live through stream completion; weights/scales/alpha may be shared read-only.
 * Source/output are BF16. No allocation, module loading, or synchronization occurs. */
int32_t ds41rt_v41_fp8_launch(void* kernel, const uint16_t* source, const uint8_t* weight,
    const uint8_t* packed_scales, void* scratch, uint64_t scratch_bytes,
    const float* alpha, uint16_t* output, int32_t rows, void* stream);
#ifdef __cplusplus
}
#endif
