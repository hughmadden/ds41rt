#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Native V4.1 mHC=4, hidden=5120; raw CUDA status, no allocation or sync.
// BF16 residual [rows,4,5120], FP32 pre/post [rows,4], comb [rows,4,4].
// Comb is [source,destination], matching sum(dim=2) in the reference.
// BF16 collapsed/sublayer [rows,5120]; outputs must be disjoint from inputs.
int32_t ds41rt_v41_hc_pre(const uint16_t* residual, const float* pre,
    uint16_t* collapsed, int32_t rows, void* stream);
int32_t ds41rt_v41_hc_post(const uint16_t* sublayer, const uint16_t* residual,
    const float* post, const float* comb, uint16_t* output, int32_t rows, void* stream);
#ifdef __cplusplus
}
#endif
