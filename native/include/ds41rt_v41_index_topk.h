#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Merge FP32 scores/U64 logical positions [queries,candidates] into an opaque
// U64 carry [queries,512]; reset=1 starts a new search without reading old carry.
// Positions must be unique across this tile AND all previously merged tiles.
// Invalid IDs >= 1048576, NaN scores and -inf scores are ignored. +inf is allowed.
// Descending score, then ascending logical position breaks ties deterministically.
// Output I32 [queries,512] is in ascending position order, trailing -1 padding.
// Carry is request/wave-owned state, not persistent model cache. Keep it private
// to one selection and reset when its query bindings change.
// Scratch bytes = queries * ceil(candidates/1024) * 512 * 8 * 2.
// All five buffer spans are disjoint; current stream device owns each span.
int32_t ds41rt_v41_index_top512(const float* scores,const uint64_t* positions,
    uint64_t* carry,void* scratch,uint64_t scratch_bytes,int32_t* output,
    int32_t queries,int32_t candidates,int32_t reset,void* stream);
#ifdef __cplusplus
}
#endif
