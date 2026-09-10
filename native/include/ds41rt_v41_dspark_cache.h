#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
  uint64_t position;
  uint32_t source_row, token_count, slot, reserved;
} ds41rt_v41_kv_write_t;
/* Input is normalized, RoPE-applied BF16 [source_rows,512]; output is the
 * reference's FP8-quantized/dequantized BF16 ring [slots,128,512].
 * Exactly 16 device descriptors; token_count=0 disables a descriptor.
 * Active descriptors must have distinct slots, valid source spans, and no
 * position+token_count overflow. Invalid device descriptors do not write.
 * Source, descriptors and ring are disjoint and live through completion.
 * Returns raw CUDA status. No allocation or synchronization. */
int32_t ds41rt_v41_dspark_cache_write(const uint16_t* source,
    const ds41rt_v41_kv_write_t* writes, uint16_t* ring,
    int32_t source_rows, int32_t slots, void* stream);
#ifdef __cplusplus
}
#endif
