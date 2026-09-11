#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Host launch descriptor, copied by value into the graph. Arrays select:
// 0: one request's 128-row ring; 1: private window wave;
// 2: shared paged compressed pool; 3: private compressor wave.
typedef struct ds41rt_v41_sparse_kv {
  const uint8_t* values[4];
  const uint8_t* scales[4];
  const uint64_t* window_end;
  const uint32_t* pages;
  const uint64_t* source_end;
  uint64_t window_proposal_capacity;
  uint64_t source_capacity;
  uint64_t source_proposal_capacity;
  uint32_t page_stride;
  uint32_t compressed;
} ds41rt_v41_sparse_kv_t;
int32_t ds41rt_v41_sparse_attention_initialize(void);
// BF16 query/output [rows,64,512], FP32 sink [64], selected I32 [rows,512].
// U64 metadata [rows,10] concatenates WindowProposal::metadata (4 fields) and
// IndexProposal::metadata (6 fields), with source slot narrowed to zero.
// Window keys are oldest first, then padding to window_width, then 512 selected
// source IDs. Negative, noncausal or unmapped source IDs are masked. Malformed
// proposal metadata yields zero for the whole query. Compressed=0 ignores source
// pointers/metadata/IDs. Width must cover every query's causal window (max128).
// Width 0 derives that maximum from the last metadata row; the caller must
// provide ascending request positions. Positive widths retain explicit control.
// Values E4M3 and scales E8M0/K32 must dequantize to finite BF16; queries/sinks
// finite. Same stream device, live immutable inputs, disjoint output through
// completion/replay. Caller guarantees lease/snapshot/request correspondence,
// selected-ID uniqueness and causal source counts. No allocation/synchronization.
int32_t ds41rt_v41_sparse_attention(const uint16_t* query,const float* sink,
    const uint64_t* metadata,const int32_t* selected,uint16_t* output,int32_t rows,
    int32_t window_width,const ds41rt_v41_sparse_kv_t* view,void* stream);
// Key partitions retain FP32 partial numerators/maxima/normalizers, then merge
// into BF16 output. Rounding differs from sequential online softmax. Caller
// owns disjoint scratch [rows,parts,64,514] FP32 through completion/replay;
// every element is overwritten, so no scratch initialization is required.
// Parts 1..10; no allocation, module resolution or synchronization at launch.
int32_t ds41rt_v41_sparse_attention_split(const uint16_t* query,const float* sink,
    const uint64_t* metadata,const int32_t* selected,uint16_t* output,int32_t rows,
    int32_t window_width,const ds41rt_v41_sparse_kv_t* view,void* stream,
    float* partial,uint64_t scratch_bytes,int32_t parts);
#ifdef __cplusplus
}
#endif
