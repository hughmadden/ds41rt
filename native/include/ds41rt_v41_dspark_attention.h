#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// One descriptor per request, max16. valid_rows counts physical ring prefix
// entries (0..128); slot selects one of the request-owned committed rings.
// Draft KV is separate: all five draft positions are visible to all five queries.
typedef struct ds41rt_v41_attention_window { uint32_t slot; uint32_t valid_rows; } ds41rt_v41_attention_window_t;
// Initialize launch attributes on the active device before capture.
int32_t ds41rt_v41_dspark_attention_initialize(void);
// BF16 query/output [requests,5,64,512], packed FP8 ring [slots,128,528],
// draft [requests,5,512]; FP32 sink [64], descriptors [requests].
// Ring rows contain 512 E4M3 bytes followed by 16 E8M0 K32 scales.
// Private draft KV is normalized, rotated and FP8 quantized/dequantized BF16.
// All inputs finite where attended, output disjoint; same stream device and
// stable live buffers through completion/replay. No allocation or synchronization.
int32_t ds41rt_v41_dspark_attention_fp8(const uint16_t* query, const uint8_t* ring,
    const uint16_t* draft, const float* sink, const ds41rt_v41_attention_window_t* windows,
    uint16_t* output, int32_t requests, int32_t slots, void* stream);
// Width-specialized variants: width is 5 or 7. Replace every draft dimension
// above by width. Initialize the chosen specialization before graph capture.
int32_t ds41rt_v41_dspark_attention_initialize_width(int32_t width);
int32_t ds41rt_v41_dspark_attention_fp8_width(const uint16_t* query, const uint8_t* ring,
    const uint16_t* draft, const float* sink, const ds41rt_v41_attention_window_t* windows,
    uint16_t* output, int32_t requests, int32_t slots, int32_t width, void* stream);
#ifdef __cplusplus
}
#endif
