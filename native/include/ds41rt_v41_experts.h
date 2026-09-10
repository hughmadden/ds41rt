#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define DS41RT_V41_EXPERT_POINTERS 44
/* Pointer slots follow the exported dynamic W4A8 C ABI; see the manifest. */
typedef struct ds41rt_v41_expert_launch_t {
  void* tensors[DS41RT_V41_EXPERT_POINTERS];
  int32_t num_tokens;
  int32_t max_rows;
  int32_t scatter_rows;
  int32_t rows_padded;
  int32_t max_tasks;
  int32_t max_phys_tiles;
  int32_t max_active_clusters;
  void* stream;
} ds41rt_v41_expert_launch_t;

typedef struct ds41rt_v41_expert_info_t {
  uint32_t abi_version;
  uint32_t role; /* 0: coordinator dSpark; 1: Spark backbone shard */
  uint32_t experts;
  uint32_t hidden_size;
  uint32_t logical_intermediate;
  uint32_t kernel_intermediate;
  uint32_t topk;
  uint32_t capacity_rows;
  uint64_t scratch_bytes;
  int32_t max_rows;
  int32_t rows_padded;
  int32_t max_tasks;
  int32_t max_phys_tiles;
  int32_t max_active_clusters;
} ds41rt_v41_expert_info_t;

/* These functions return CUDA runtime error codes (zero is success).
 * Initialize before graph capture; each variant binds to its first CUDA device.
 * Kernel handles borrow the library and must outlive every launch/graph replay.
 * Callers own CUDA buffers, capacity, initialization and stream ordering. */
int32_t ds41rt_v41_expert_info(int32_t capacity, ds41rt_v41_expert_info_t* out);
int32_t ds41rt_v41_expert_initialize(int32_t capacity, void** out_kernel);
int32_t ds41rt_v41_expert_launch(void* kernel, const ds41rt_v41_expert_launch_t* args);
/* Reduce contiguous FP32 [rows,topk,5120] route planes into BF16 [rows,5120].
 * Supported geometries: ranks=1/topk=3 (RTX dSpark), ranks=4/topk=6 (backbone).
 * planes is a host array of device pointers; unused slots must be null.
 * Sum TP ranks before rounding each route to BF16, then sum routes in FP32,
 * add optional BF16 shared output and round once to BF16.
 * Output may equal shared exactly, but must not overlap any route plane or
 * partially overlap shared; all device storage must outlive stream completion.
 * No allocation or synchronization; returns a CUDA runtime error code. */
int32_t ds41rt_v41_reduce_routes_async(const float* const planes[4],
    const uint16_t* shared, uint16_t* output, uint32_t rows,
    uint32_t ranks, uint32_t topk, void* stream);
#ifdef __cplusplus
}
#endif
