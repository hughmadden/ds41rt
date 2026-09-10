#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Q: E2M1 [queries,32,64] / E8M0 [queries,32,4]; weights BF16 [queries,32].
// Keys: E2M1 [capacity,64] / E8M0 [capacity,4]. Pages U32 [slots,stride],
// committed lengths U64 [slots], query metadata U64 [queries,2]=(slot,causal_rows).
// Candidate logical row IDs U64 [queries,candidates]; UINT64_MAX skips.
// Output FP32 [queries,candidates], with BF16 rounding at dot, weighted product
// and final head sum, matching reference BF16 einsum/relu/multiply/sum semantics.
// Invalid or unreachable addressing yields -inf without reading a key/query.
// Finite inputs/scales and disjoint output are required. No allocation or mutation
// of cache state. Caller validates request leases and serializes cache publication.
int32_t ds41rt_v41_index_scores(const uint8_t* q,const uint8_t* qs,const uint16_t* weights,
    const uint8_t* keys,const uint8_t* ks,const uint32_t* pages,const uint64_t* lengths,
    const uint64_t* metadata,const uint64_t* positions,float* output,
    int32_t queries,int32_t candidates,int32_t slots,int32_t stride,uint64_t capacity,void* stream);
// Append-only proposal view. Metadata [queries,6] is
// (slot,causal_rows,committed_start,proposal_count,proposal_offset,proposal_step).
// Proposal step is 1 or 2, preserving completed rows in ratio-one/two waves.
// Proposal values/scales [proposal_capacity,64/4] remain immutable and uncommitted.
// The descriptor must match the slot's committed length. Malformed descriptors
// mask the entire query; causal and physical bounds are checked before reads.
// Proposal rows may extend beyond the currently allocated committed page table.
// Caller validates proposal owner/version/row mapping against the request lease.
int32_t ds41rt_v41_index_scores_overlay(const uint8_t* q,const uint8_t* qs,const uint16_t* weights,
    const uint8_t* keys,const uint8_t* ks,const uint32_t* pages,const uint64_t* lengths,
    const uint64_t* metadata,const uint64_t* positions,float* output,
    const uint8_t* proposals,const uint8_t* proposal_scales,
    int32_t queries,int32_t candidates,int32_t slots,int32_t stride,uint64_t capacity,
    uint64_t proposal_capacity,void* stream);
#ifdef __cplusplus
}
#endif
