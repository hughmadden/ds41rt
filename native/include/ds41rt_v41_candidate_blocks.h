#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Contiguous logical score tile [queries,width], U64 first positions [queries]
// (must align to 8), U64 causal lengths [queries]. Produces ceil(width/8) block
// maxima and U64 block IDs per query. Bounds are [0,1048576]; malformed metadata
// emits empty blocks. The newest reachable block receives +inf even if its
// score is -inf. Tiles partition blocks; only the final tile may end mid-block.
// Scores are finite or -inf. NaNs are ignored. Outputs are disjoint from inputs.
int32_t ds41rt_v41_candidate_block_max(const float* scores,const uint64_t* first,
    const uint64_t* lengths,float* maxima,uint64_t* ids,int32_t queries,int32_t width,void* stream);
// Sorted I32 selected blocks [queries,2048], U64 causal lengths [queries] ->
// U64 logical row candidates [queries,16384]. Unreachable/invalid rows use MAX.
int32_t ds41rt_v41_candidate_expand(const int32_t* blocks,const uint64_t* lengths,
    uint64_t* positions,int32_t queries,void* stream);
#ifdef __cplusplus
}
#endif
