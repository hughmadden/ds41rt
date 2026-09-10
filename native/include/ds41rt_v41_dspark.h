#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// BF16 hidden [rows,5120], Markov embeddings [rows,256], checkpoint weight
// [5376]; FP32 raw projection output [rows], without sigmoid. Raw CUDA status.
// Inputs must be initialized on this stream's device and remain live through
// completion; output must be disjoint from every input. No allocation or sync.
int32_t ds41rt_v41_dspark_confidence(const uint16_t* hidden,
    const uint16_t* markov, const uint16_t* weight, float* output,
    int32_t rows, void* stream);
#ifdef __cplusplus
}
#endif
