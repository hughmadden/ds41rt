#include "common.h"

#include "ds4_pro_spark_moe_aot_config.h"
#include "ds4_pro_shared_down_m1.h"
#include "ds4_pro_shared_down_m2048.h"
#include "ds4_pro_shared_down_m8.h"
#include "ds4_pro_shared_activated_quant_m2048.h"
#include "ds4_pro_shared_input_quant_m2048.h"
#include "ds4_pro_shared_up_m1.h"
#include "ds4_pro_shared_up_m2048.h"
#include "ds4_pro_shared_up_m8.h"
#include "ds4_pro_tp4_exl3_k2_decode_m1.h"
#include "ds4_pro_tp4_exl3_k2_prefill_m1024_topk6.h"
#include "ds4_pro_tp4_exl3_k2_prefill_m128_topk6.h"
#include "ds4_pro_tp4_exl3_k2_prefill_m16_topk6.h"
#include "ds4_pro_tp4_exl3_k2_prefill_m2048_topk6.h"
#include "ds4_pro_tp4_exl3_k2_prefill_m256_topk6.h"
#include "ds4_pro_tp4_exl3_k2_prefill_m2_topk6.h"
#include "ds4_pro_tp4_exl3_k2_prefill_m32_topk6.h"
#include "ds4_pro_tp4_exl3_k2_prefill_m4_topk6.h"
#include "ds4_pro_tp4_exl3_k2_prefill_m512_topk6.h"
#include "ds4_pro_tp4_exl3_k2_prefill_m64_topk6.h"
#include "ds4_pro_tp4_exl3_k2_prefill_m8_topk6.h"
#include "ds4_pro_tp4_exl3_k2_topk6_sum.h"

#include <mutex>
#include <cuda_bf16.h>

namespace {

constexpr size_t kHidden = DS4RT_DS4_PRO_HIDDEN_SIZE;
constexpr size_t kIntermediate = DS4RT_DS4_PRO_TP_INTERMEDIATE_SIZE;
constexpr size_t kExperts = DS4RT_DS4_PRO_NUM_EXPERTS;
constexpr size_t kTopK = DS4RT_DS4_PRO_TOP_K;
constexpr size_t kW13Rows = 2 * kIntermediate;
constexpr size_t kPrefillMaxRows = DS4RT_DS4_PRO_PREFILL_MAX_ROWS;
constexpr size_t kPrefillMaxPackedRouteSlots =
    DS4RT_DS4_PRO_EXL3_K2_PREFILL_M2048_TOPK6_PACKED_ROUTE_SLOTS;
constexpr size_t kPrefillMaxRouteBlocks =
    DS4RT_DS4_PRO_EXL3_K2_PREFILL_M2048_TOPK6_MAX_M_BLOCKS;
constexpr size_t kPrefillRouteBlockRows = 32;
constexpr size_t kExl3Bits = 2;
constexpr size_t kExl3TrellisWords = 16 * kExl3Bits;
constexpr size_t kExl3W13Bytes = 2 * kExperts * (kHidden / 16) *
                                 (kIntermediate / 16) * kExl3TrellisWords *
                                 sizeof(int16_t);
constexpr size_t kExl3W2Bytes = kExperts * (kIntermediate / 16) *
                                (kHidden / 16) * kExl3TrellisWords *
                                sizeof(int16_t);
constexpr size_t kExl3HiddenRotationBytes =
    kExperts * kHidden * sizeof(uint16_t);
constexpr size_t kExl3IntermediateRotationBytes =
    kExperts * 3 * kIntermediate * sizeof(uint16_t);

// packed_gemm_scratch_elements(size_n, route_slots, block_m, sms=48).
// Decode uses 48 route slots and block-M 8 (including the small-M x2 factor).
// The existing prefill arena is deliberately conservative relative to the
// block-M 32, 48-SM cap of 1,572,864 elements.
constexpr size_t kDecodeFc1ScratchElements = 147'456;
constexpr size_t kDecodeFc2ScratchElements = 688'128;
constexpr size_t kPrefillScratchElements = 3'145'728;
constexpr size_t kWorkspaceElements = 48 * 4 + 2;
constexpr size_t kSharedIntermediate =
    DS4RT_DS4_PRO_SHARED_INTERMEDIATE_SIZE;
constexpr size_t kSharedChunkRows = 8;
constexpr size_t kSharedWeightBytes = kHidden * kSharedIntermediate;
constexpr size_t kSharedScaleMmaBytes =
    (kHidden / 128) * (kSharedIntermediate / 128) * 512;
constexpr size_t kMxfp8ScaleVector = 32;
constexpr size_t kMxfp8ScaleTileRows = 128;
constexpr size_t kMxfp8ScaleTileK = 128;
constexpr size_t kMxfp8ScaleTileBytes = 512;
constexpr size_t kMxfp8QuantThreads = 256;
constexpr size_t kMxfp8QuantWarps = kMxfp8QuantThreads / 32;
constexpr size_t kMxfp8QuantCtasPerSm = 4;

ds4rt_ds4_pro_tp4_exl3_k2_decode_m1_Kernel_Module_t pro_decode_module;
ds4rt_ds4_pro_tp4_exl3_k2_topk6_sum_Kernel_Module_t pro_topk6_sum_module;
ds4rt_ds4_pro_shared_up_m1_Kernel_Module_t pro_shared_up_m1_module;
ds4rt_ds4_pro_shared_up_m8_Kernel_Module_t pro_shared_up_m8_module;
ds4rt_ds4_pro_shared_up_m2048_Kernel_Module_t pro_shared_up_m2048_module;
ds4rt_ds4_pro_shared_down_m1_Kernel_Module_t pro_shared_down_m1_module;
ds4rt_ds4_pro_shared_down_m8_Kernel_Module_t pro_shared_down_m8_module;
ds4rt_ds4_pro_shared_down_m2048_Kernel_Module_t pro_shared_down_m2048_module;
ds4rt_ds4_pro_shared_input_quant_m2048_Kernel_Module_t
    pro_shared_input_quant_m2048_module;
ds4rt_ds4_pro_shared_activated_quant_m2048_Kernel_Module_t
    pro_shared_activated_quant_m2048_module;
#define DS4RT_DEFINE_PRO_PREFILL_MODULE(M)                                     \
  ds4rt_ds4_pro_tp4_exl3_k2_prefill_m##M##_topk6_Kernel_Module_t               \
      pro_prefill_m##M##_module;
DS4RT_DEFINE_PRO_PREFILL_MODULE(2)
DS4RT_DEFINE_PRO_PREFILL_MODULE(4)
DS4RT_DEFINE_PRO_PREFILL_MODULE(8)
DS4RT_DEFINE_PRO_PREFILL_MODULE(16)
DS4RT_DEFINE_PRO_PREFILL_MODULE(32)
DS4RT_DEFINE_PRO_PREFILL_MODULE(64)
DS4RT_DEFINE_PRO_PREFILL_MODULE(128)
DS4RT_DEFINE_PRO_PREFILL_MODULE(256)
DS4RT_DEFINE_PRO_PREFILL_MODULE(512)
DS4RT_DEFINE_PRO_PREFILL_MODULE(1024)
DS4RT_DEFINE_PRO_PREFILL_MODULE(2048)
#undef DS4RT_DEFINE_PRO_PREFILL_MODULE

std::once_flag pro_module_init_once;
ds4rt_status_t pro_module_init_status = DS4RT_STATUS_OK;

bool has_bytes(ds4rt_device_buffer_t buffer, size_t required) {
  return buffer.ptr != nullptr && buffer.bytes >= required;
}

bool is_aligned(ds4rt_device_buffer_t buffer, size_t alignment) {
  return reinterpret_cast<uintptr_t>(buffer.ptr) % alignment == 0;
}

void initialize_pro_modules_on_current_device() {
  int32_t device_id = 0;
  cudaError_t result = cudaGetDevice(&device_id);
  if (result != cudaSuccess) {
    ds4rt_set_last_error_message(cudaGetErrorString(result));
    pro_module_init_status = DS4RT_STATUS_INTERNAL_ERROR;
    return;
  }

#define DS4RT_INIT_PRO_MODULE(prefix, module_value)                            \
  do {                                                                         \
    cudaLibrary_t *library = &(module_value).module;                           \
    result = cudaSuccess;                                                      \
    struct {                                                                   \
      cudaLibrary_t **library;                                                 \
      cudaError_t *result;                                                     \
    } init_args = {&library, &result};                                         \
    _mlir_##prefix##_cuda_init(reinterpret_cast<void **>(&init_args));         \
    if (result != cudaSuccess) {                                               \
      ds4rt_set_last_error_message(cudaGetErrorString(result));                \
      pro_module_init_status = DS4RT_STATUS_INTERNAL_ERROR;                    \
      return;                                                                  \
    }                                                                          \
    struct {                                                                   \
      cudaLibrary_t **library;                                                 \
      int32_t *device_id;                                                      \
      cudaError_t *result;                                                     \
    } load_args = {&library, &device_id, &result};                             \
    _mlir_##prefix##_cuda_load_to_device(                                      \
        reinterpret_cast<void **>(&load_args));                                \
    if (result != cudaSuccess) {                                               \
      ds4rt_set_last_error_message(cudaGetErrorString(result));                \
      pro_module_init_status = DS4RT_STATUS_INTERNAL_ERROR;                    \
      return;                                                                  \
    }                                                                          \
  } while (false)

  DS4RT_INIT_PRO_MODULE(ds4rt_ds4_pro_tp4_exl3_k2_decode_m1, pro_decode_module);
  DS4RT_INIT_PRO_MODULE(ds4rt_ds4_pro_tp4_exl3_k2_topk6_sum,
                        pro_topk6_sum_module);
  DS4RT_INIT_PRO_MODULE(ds4rt_ds4_pro_shared_up_m1,
                        pro_shared_up_m1_module);
  DS4RT_INIT_PRO_MODULE(ds4rt_ds4_pro_shared_up_m8,
                        pro_shared_up_m8_module);
  DS4RT_INIT_PRO_MODULE(ds4rt_ds4_pro_shared_up_m2048,
                        pro_shared_up_m2048_module);
  DS4RT_INIT_PRO_MODULE(ds4rt_ds4_pro_shared_down_m1,
                        pro_shared_down_m1_module);
  DS4RT_INIT_PRO_MODULE(ds4rt_ds4_pro_shared_down_m8,
                        pro_shared_down_m8_module);
  DS4RT_INIT_PRO_MODULE(ds4rt_ds4_pro_shared_down_m2048,
                        pro_shared_down_m2048_module);
  DS4RT_INIT_PRO_MODULE(ds4rt_ds4_pro_shared_input_quant_m2048,
                        pro_shared_input_quant_m2048_module);
  DS4RT_INIT_PRO_MODULE(ds4rt_ds4_pro_shared_activated_quant_m2048,
                        pro_shared_activated_quant_m2048_module);
#define DS4RT_INIT_PRO_PREFILL_MODULE(M)                                       \
  DS4RT_INIT_PRO_MODULE(ds4rt_ds4_pro_tp4_exl3_k2_prefill_m##M##_topk6,        \
                        pro_prefill_m##M##_module)
  DS4RT_INIT_PRO_PREFILL_MODULE(2);
  DS4RT_INIT_PRO_PREFILL_MODULE(4);
  DS4RT_INIT_PRO_PREFILL_MODULE(8);
  DS4RT_INIT_PRO_PREFILL_MODULE(16);
  DS4RT_INIT_PRO_PREFILL_MODULE(32);
  DS4RT_INIT_PRO_PREFILL_MODULE(64);
  DS4RT_INIT_PRO_PREFILL_MODULE(128);
  DS4RT_INIT_PRO_PREFILL_MODULE(256);
  DS4RT_INIT_PRO_PREFILL_MODULE(512);
  DS4RT_INIT_PRO_PREFILL_MODULE(1024);
  DS4RT_INIT_PRO_PREFILL_MODULE(2048);
#undef DS4RT_INIT_PRO_PREFILL_MODULE
#undef DS4RT_INIT_PRO_MODULE
}

bool buffers_share_device(
    const ds4rt_ds4_pro_spark_exl3_k2_moe_buffers_t *buffers) {
  const int device = buffers->input.device_id;
#define DS4RT_PRO_SAME_DEVICE(field) buffers->field.device_id == device
  const bool same =
      DS4RT_PRO_SAME_DEVICE(w13_trellis) && DS4RT_PRO_SAME_DEVICE(w2_trellis) &&
      DS4RT_PRO_SAME_DEVICE(gate_suh) && DS4RT_PRO_SAME_DEVICE(up_suh) &&
      DS4RT_PRO_SAME_DEVICE(intermediate_rotations) &&
      DS4RT_PRO_SAME_DEVICE(down_svh) && DS4RT_PRO_SAME_DEVICE(expert_map) &&
      DS4RT_PRO_SAME_DEVICE(dummy_scale) &&
      DS4RT_PRO_SAME_DEVICE(trellis_lut) &&
      DS4RT_PRO_SAME_DEVICE(global_scale) && DS4RT_PRO_SAME_DEVICE(topk_ids) &&
      DS4RT_PRO_SAME_DEVICE(topk_weights) &&
      DS4RT_PRO_SAME_DEVICE(rotation_gate) &&
      DS4RT_PRO_SAME_DEVICE(rotation_up) && DS4RT_PRO_SAME_DEVICE(fc1_output) &&
      DS4RT_PRO_SAME_DEVICE(activated) &&
      DS4RT_PRO_SAME_DEVICE(routed_output) &&
      DS4RT_PRO_SAME_DEVICE(output_f32) && DS4RT_PRO_SAME_DEVICE(output_bf16) &&
      DS4RT_PRO_SAME_DEVICE(packed_route_indices) &&
      DS4RT_PRO_SAME_DEVICE(block_expert_ids) &&
      DS4RT_PRO_SAME_DEVICE(packed_route_count) &&
      DS4RT_PRO_SAME_DEVICE(expert_counts) &&
      DS4RT_PRO_SAME_DEVICE(expert_offsets) &&
      DS4RT_PRO_SAME_DEVICE(fc1_scratch) &&
      DS4RT_PRO_SAME_DEVICE(fc2_scratch) && DS4RT_PRO_SAME_DEVICE(workspace);
#undef DS4RT_PRO_SAME_DEVICE
  return same;
}

ds4rt_status_t
validate_buffers(const ds4rt_ds4_pro_spark_exl3_k2_moe_buffers_t *buffers,
                 size_t capacity_rows) {
  if (buffers == nullptr || capacity_rows == 0 ||
      capacity_rows > kPrefillMaxRows) {
    return DS4RT_STATUS_INVALID_ARGUMENT;
  }
  const bool decode = capacity_rows == 1;
  const size_t routed_rows = capacity_rows * kTopK;
  const size_t packed_route_elements =
      decode ? DS4RT_DS4_PRO_EXL3_K2_DECODE_M1_PACKED_ROUTE_SLOTS
             : kPrefillMaxPackedRouteSlots;
  const size_t route_block_elements =
      decode ? DS4RT_DS4_PRO_EXL3_K2_DECODE_M1_MAX_M_BLOCKS
             : kPrefillMaxRouteBlocks;
  const size_t fc1_scratch_elements =
      decode ? kDecodeFc1ScratchElements : kPrefillScratchElements;
  const size_t fc2_scratch_elements =
      decode ? kDecodeFc2ScratchElements : kPrefillScratchElements;
  const bool valid =
      has_bytes(buffers->input, capacity_rows * kHidden * sizeof(uint16_t)) &&
      has_bytes(buffers->w13_trellis, kExl3W13Bytes) &&
      has_bytes(buffers->w2_trellis, kExl3W2Bytes) &&
      has_bytes(buffers->gate_suh, kExl3HiddenRotationBytes) &&
      has_bytes(buffers->up_suh, kExl3HiddenRotationBytes) &&
      has_bytes(buffers->intermediate_rotations,
                kExl3IntermediateRotationBytes) &&
      has_bytes(buffers->down_svh, kExl3HiddenRotationBytes) &&
      has_bytes(buffers->expert_map, kExperts * sizeof(int32_t)) &&
      has_bytes(buffers->dummy_scale, sizeof(float)) &&
      has_bytes(buffers->trellis_lut, 1 << 12) &&
      has_bytes(buffers->global_scale, kExperts * sizeof(float)) &&
      has_bytes(buffers->topk_ids, routed_rows * sizeof(int32_t)) &&
      has_bytes(buffers->topk_weights, routed_rows * sizeof(float)) &&
      has_bytes(buffers->rotation_gate,
                routed_rows * kHidden * sizeof(uint16_t)) &&
      has_bytes(buffers->rotation_up,
                routed_rows * kHidden * sizeof(uint16_t)) &&
      has_bytes(buffers->fc1_output,
                routed_rows * kW13Rows * sizeof(uint16_t)) &&
      has_bytes(buffers->activated,
                routed_rows * kIntermediate * sizeof(uint16_t)) &&
      has_bytes(buffers->routed_output,
                routed_rows * kHidden * sizeof(uint16_t)) &&
      has_bytes(buffers->output_f32, capacity_rows * kHidden * sizeof(float)) &&
      has_bytes(buffers->output_bf16,
                capacity_rows * kHidden * sizeof(uint16_t)) &&
      has_bytes(buffers->packed_route_indices,
                packed_route_elements * sizeof(int32_t)) &&
      has_bytes(buffers->block_expert_ids,
                route_block_elements * sizeof(int32_t)) &&
      has_bytes(buffers->packed_route_count, sizeof(int32_t)) &&
      has_bytes(buffers->expert_counts, kExperts * sizeof(int32_t)) &&
      has_bytes(buffers->expert_offsets, (kExperts + 1) * sizeof(int32_t)) &&
      has_bytes(buffers->fc1_scratch, fc1_scratch_elements * sizeof(float)) &&
      has_bytes(buffers->fc2_scratch, fc2_scratch_elements * sizeof(float)) &&
      has_bytes(buffers->workspace, kWorkspaceElements * sizeof(int32_t));
  if (!valid) {
    return DS4RT_STATUS_BUFFER_TOO_SMALL;
  }
  const bool aligned =
      is_aligned(buffers->input, 16) && is_aligned(buffers->w13_trellis, 16) &&
      is_aligned(buffers->w2_trellis, 16) &&
      is_aligned(buffers->gate_suh, 16) && is_aligned(buffers->up_suh, 16) &&
      is_aligned(buffers->intermediate_rotations, 16) &&
      is_aligned(buffers->down_svh, 16) &&
      is_aligned(buffers->expert_map, 16) &&
      is_aligned(buffers->dummy_scale, 16) &&
      is_aligned(buffers->trellis_lut, 16) &&
      is_aligned(buffers->global_scale, 16) &&
      is_aligned(buffers->topk_ids, 16) &&
      is_aligned(buffers->topk_weights, 16) &&
      is_aligned(buffers->rotation_gate, 16) &&
      is_aligned(buffers->rotation_up, 16) &&
      is_aligned(buffers->fc1_output, 16) &&
      is_aligned(buffers->activated, 16) &&
      is_aligned(buffers->routed_output, 16) &&
      is_aligned(buffers->output_f32, 16) &&
      is_aligned(buffers->output_bf16, 16) &&
      is_aligned(buffers->packed_route_indices, 16) &&
      is_aligned(buffers->block_expert_ids, 16) &&
      is_aligned(buffers->fc1_scratch, 16) &&
      is_aligned(buffers->fc2_scratch, 16) &&
      is_aligned(buffers->workspace, 16);
  return aligned && buffers_share_device(buffers)
             ? DS4RT_STATUS_OK
             : DS4RT_STATUS_INVALID_ARGUMENT;
}

bool shared_buffers_share_device(
    const ds4rt_ds4_flash_shared_expert_fp8_buffers_t *buffers) {
  const int device = buffers->input.device_id;
#define DS4RT_PRO_SHARED_SAME_DEVICE(field) buffers->field.device_id == device
  const bool same =
      DS4RT_PRO_SHARED_SAME_DEVICE(w1_weight) &&
      DS4RT_PRO_SHARED_SAME_DEVICE(w1_scale_mma) &&
      DS4RT_PRO_SHARED_SAME_DEVICE(w3_weight) &&
      DS4RT_PRO_SHARED_SAME_DEVICE(w3_scale_mma) &&
      DS4RT_PRO_SHARED_SAME_DEVICE(w2_weight) &&
      DS4RT_PRO_SHARED_SAME_DEVICE(w2_scale_mma) &&
      DS4RT_PRO_SHARED_SAME_DEVICE(gate) &&
      DS4RT_PRO_SHARED_SAME_DEVICE(up) &&
      DS4RT_PRO_SHARED_SAME_DEVICE(activated) &&
      DS4RT_PRO_SHARED_SAME_DEVICE(output) &&
      DS4RT_PRO_SHARED_SAME_DEVICE(alpha);
#undef DS4RT_PRO_SHARED_SAME_DEVICE
  return same;
}

ds4rt_status_t validate_shared_buffers(
    const ds4rt_ds4_flash_shared_expert_fp8_buffers_t *buffers, size_t rows) {
  if (buffers == nullptr || rows == 0 || rows > kPrefillMaxRows) {
    return DS4RT_STATUS_INVALID_ARGUMENT;
  }
  const bool valid =
      has_bytes(buffers->input, rows * kHidden * sizeof(uint16_t)) &&
      has_bytes(buffers->w1_weight, kSharedWeightBytes) &&
      has_bytes(buffers->w1_scale_mma, kSharedScaleMmaBytes) &&
      has_bytes(buffers->w3_weight, kSharedWeightBytes) &&
      has_bytes(buffers->w3_scale_mma, kSharedScaleMmaBytes) &&
      has_bytes(buffers->w2_weight, kSharedWeightBytes) &&
      has_bytes(buffers->w2_scale_mma, kSharedScaleMmaBytes) &&
      has_bytes(buffers->gate,
                kSharedChunkRows * kSharedIntermediate * sizeof(uint16_t)) &&
      has_bytes(buffers->up,
                kSharedChunkRows * kSharedIntermediate * sizeof(uint16_t)) &&
      has_bytes(buffers->activated,
                kSharedChunkRows * kSharedIntermediate * sizeof(uint16_t)) &&
      has_bytes(buffers->output, rows * kHidden * sizeof(uint16_t)) &&
      has_bytes(buffers->alpha, sizeof(float));
  if (!valid) {
    return DS4RT_STATUS_BUFFER_TOO_SMALL;
  }
  return shared_buffers_share_device(buffers)
             ? DS4RT_STATUS_OK
             : DS4RT_STATUS_INVALID_ARGUMENT;
}

size_t mxfp8_scale_mma_bytes(size_t rows, size_t size_k) {
  const size_t row_tiles =
      (rows + kMxfp8ScaleTileRows - 1) / kMxfp8ScaleTileRows;
  const size_t k_tiles =
      (size_k + kMxfp8ScaleTileK - 1) / kMxfp8ScaleTileK;
  return row_tiles * k_tiles * kMxfp8ScaleTileBytes;
}

bool pro_prefill_buffers_share_device(
    const ds4rt_ds4_pro_shared_expert_fp8_prefill_buffers_t *buffers) {
  const int device = buffers->base.input.device_id;
#define DS4RT_PRO_PREFILL_SAME_DEVICE(field)                                  \
  buffers->field.device_id == device
  const bool same =
      DS4RT_PRO_PREFILL_SAME_DEVICE(input_q_values) &&
      DS4RT_PRO_PREFILL_SAME_DEVICE(input_q_scale_rows) &&
      DS4RT_PRO_PREFILL_SAME_DEVICE(input_q_scale_mma) &&
      DS4RT_PRO_PREFILL_SAME_DEVICE(activated_q_values) &&
      DS4RT_PRO_PREFILL_SAME_DEVICE(activated_q_scale_rows) &&
      DS4RT_PRO_PREFILL_SAME_DEVICE(activated_q_scale_mma);
#undef DS4RT_PRO_PREFILL_SAME_DEVICE
  return same && shared_buffers_share_device(&buffers->base);
}

ds4rt_status_t validate_pro_prefill_buffers(
    const ds4rt_ds4_pro_shared_expert_fp8_prefill_buffers_t *buffers,
    size_t rows) {
  if (buffers == nullptr || rows <= kSharedChunkRows ||
      rows > kPrefillMaxRows) {
    return DS4RT_STATUS_INVALID_ARGUMENT;
  }
  const ds4rt_status_t base_valid = validate_shared_buffers(&buffers->base, rows);
  if (base_valid != DS4RT_STATUS_OK) {
    return base_valid;
  }
  const size_t shared_values = rows * kSharedIntermediate;
  const size_t input_values = rows * kHidden;
  const bool valid =
      has_bytes(buffers->base.gate, shared_values * sizeof(uint16_t)) &&
      has_bytes(buffers->base.up, shared_values * sizeof(uint16_t)) &&
      has_bytes(buffers->base.activated,
                shared_values * sizeof(uint16_t)) &&
      has_bytes(buffers->input_q_values, input_values) &&
      has_bytes(buffers->input_q_scale_rows,
                input_values / kMxfp8ScaleVector) &&
      has_bytes(buffers->input_q_scale_mma,
                mxfp8_scale_mma_bytes(rows, kHidden)) &&
      has_bytes(buffers->activated_q_values, shared_values) &&
      has_bytes(buffers->activated_q_scale_rows,
                shared_values / kMxfp8ScaleVector) &&
      has_bytes(buffers->activated_q_scale_mma,
                mxfp8_scale_mma_bytes(rows, kSharedIntermediate));
  if (!valid) {
    return DS4RT_STATUS_BUFFER_TOO_SMALL;
  }
  const bool aligned =
      is_aligned(buffers->input_q_values, 16) &&
      is_aligned(buffers->input_q_scale_rows, 16) &&
      is_aligned(buffers->input_q_scale_mma, 16) &&
      is_aligned(buffers->activated_q_values, 16) &&
      is_aligned(buffers->activated_q_scale_rows, 16) &&
      is_aligned(buffers->activated_q_scale_mma, 16);
  return aligned && pro_prefill_buffers_share_device(buffers)
             ? DS4RT_STATUS_OK
             : DS4RT_STATUS_INVALID_ARGUMENT;
}

int32_t mxfp8_quant_grid_x(size_t rows, size_t size_k) {
  int device_id = 0;
  if (cudaGetDevice(&device_id) != cudaSuccess) {
    return 0;
  }
  cudaDeviceProp properties{};
  if (cudaGetDeviceProperties(&properties, device_id) != cudaSuccess) {
    return 0;
  }
  const size_t tasks = rows * (size_k / 128);
  const size_t natural_grid =
      (tasks + kMxfp8QuantWarps - 1) / kMxfp8QuantWarps;
  const size_t grid_cap =
      static_cast<size_t>(properties.multiProcessorCount) *
      kMxfp8QuantCtasPerSm;
  return static_cast<int32_t>(natural_grid < grid_cap ? natural_grid
                                                      : grid_cap);
}

__global__ void pro_shared_swiglu_kernel(const uint16_t *gate,
                                         const uint16_t *up,
                                         uint16_t *activated,
                                         size_t values) {
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= values) {
    return;
  }
  float gate_value = bf16_to_f32(gate[index]);
  float up_value = bf16_to_f32(up[index]);
  gate_value = fminf(gate_value, DS4RT_DS4_PRO_SWIGLU_LIMIT);
  up_value = fminf(fmaxf(up_value, -DS4RT_DS4_PRO_SWIGLU_LIMIT),
                   DS4RT_DS4_PRO_SWIGLU_LIMIT);
  const float silu = gate_value / (1.0f + expf(-gate_value));
  reinterpret_cast<__nv_bfloat16 *>(activated)[index] =
      __float2bfloat16_rn(silu * up_value);
}

#define DS4RT_DEFINE_PRO_SHARED_LINEAR_LAUNCH(NAME, M)                         \
  int launch_pro_shared_##NAME##_m##M(                                        \
      void *source, void *weight, void *scale, void *output, void *alpha,      \
      int32_t rows, cudaStream_t stream) {                                     \
    return cute_dsl_ds4rt_ds4_pro_shared_##NAME##_m##M##_wrapper(              \
        &pro_shared_##NAME##_m##M##_module, source, source, weight, source,    \
        scale, output, alpha, rows, stream);                                   \
  }
DS4RT_DEFINE_PRO_SHARED_LINEAR_LAUNCH(up, 1)
DS4RT_DEFINE_PRO_SHARED_LINEAR_LAUNCH(up, 8)
DS4RT_DEFINE_PRO_SHARED_LINEAR_LAUNCH(down, 1)
DS4RT_DEFINE_PRO_SHARED_LINEAR_LAUNCH(down, 8)
#undef DS4RT_DEFINE_PRO_SHARED_LINEAR_LAUNCH

int launch_pro_shared_input_quant_m2048(
    void *source, void *values, void *scale_rows, void *scale_mma,
    int32_t rows, int32_t grid_x, cudaStream_t stream) {
  return cute_dsl_ds4rt_ds4_pro_shared_input_quant_m2048_wrapper(
      &pro_shared_input_quant_m2048_module, source, values, scale_rows,
      scale_mma, rows, grid_x, stream);
}

int launch_pro_shared_activated_quant_m2048(
    void *source, void *values, void *scale_rows, void *scale_mma,
    int32_t rows, int32_t grid_x, cudaStream_t stream) {
  return cute_dsl_ds4rt_ds4_pro_shared_activated_quant_m2048_wrapper(
      &pro_shared_activated_quant_m2048_module, source, values, scale_rows,
      scale_mma, rows, grid_x, stream);
}

#define DS4RT_DEFINE_PRO_SHARED_PREFILL_LINEAR_LAUNCH(NAME)                   \
  int launch_pro_shared_##NAME##_m2048(                                      \
      void *values, void *scale_mma, void *weight, void *weight_scale,        \
      void *output, void *alpha, int32_t rows, cudaStream_t stream) {         \
    return cute_dsl_ds4rt_ds4_pro_shared_##NAME##_m2048_wrapper(              \
        &pro_shared_##NAME##_m2048_module, values, weight, scale_mma,         \
        weight_scale, output, alpha, alpha, alpha, alpha, rows, stream);      \
  }
DS4RT_DEFINE_PRO_SHARED_PREFILL_LINEAR_LAUNCH(up)
DS4RT_DEFINE_PRO_SHARED_PREFILL_LINEAR_LAUNCH(down)
#undef DS4RT_DEFINE_PRO_SHARED_PREFILL_LINEAR_LAUNCH

__global__ void count_routes_kernel(const int32_t *topk_ids,
                                    int32_t *expert_counts,
                                    size_t live_routes) {
  const size_t route =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (route >= live_routes) {
    return;
  }
  const int32_t expert = topk_ids[route];
  if (expert >= 0 && expert < static_cast<int32_t>(kExperts)) {
    atomicAdd(expert_counts + expert, 1);
  }
}

__global__ void
prefix_routes_kernel(const int32_t *expert_counts, int32_t *expert_offsets,
                     int32_t *packed_route_indices, int32_t *block_expert_ids,
                     int32_t *packed_route_count, int32_t route_sentinel) {
  for (size_t route = threadIdx.x; route < kPrefillMaxPackedRouteSlots;
       route += blockDim.x) {
    packed_route_indices[route] = route_sentinel;
  }
  for (size_t block = threadIdx.x; block < kPrefillMaxRouteBlocks;
       block += blockDim.x) {
    block_expert_ids[block] = -1;
  }
  __syncthreads();

  if (threadIdx.x != 0) {
    return;
  }
  int32_t packed_offset = 0;
  for (int32_t expert = 0; expert < static_cast<int32_t>(kExperts); ++expert) {
    expert_offsets[expert] = packed_offset;
    const int32_t count = expert_counts[expert];
    const int32_t blocks =
        (count + static_cast<int32_t>(kPrefillRouteBlockRows) - 1) /
        static_cast<int32_t>(kPrefillRouteBlockRows);
    for (int32_t block = 0; block < blocks; ++block) {
      block_expert_ids[packed_offset /
                           static_cast<int32_t>(kPrefillRouteBlockRows) +
                       block] = expert;
    }
    packed_offset += blocks * static_cast<int32_t>(kPrefillRouteBlockRows);
  }
  expert_offsets[kExperts] = packed_offset;
  packed_route_count[0] = packed_offset;
}

__global__ void scatter_routes_kernel(const int32_t *topk_ids,
                                      int32_t *expert_offsets,
                                      int32_t *packed_route_indices,
                                      size_t live_routes) {
  const size_t route =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (route >= live_routes) {
    return;
  }
  const int32_t expert = topk_ids[route];
  if (expert < 0 || expert >= static_cast<int32_t>(kExperts)) {
    return;
  }
  const int32_t destination = atomicAdd(expert_offsets + expert, 1);
  packed_route_indices[destination] = static_cast<int32_t>(route);
}

ds4rt_status_t
pack_routes(const ds4rt_ds4_pro_spark_exl3_k2_moe_buffers_t *buffers,
            size_t rows, cudaStream_t stream) {
  cudaError_t error = cudaMemsetAsync(buffers->expert_counts.ptr, 0,
                                      kExperts * sizeof(int32_t), stream);
  if (error != cudaSuccess) {
    return status_from_cuda(error);
  }
  constexpr size_t threads = 256;
  const size_t live_routes = rows * kTopK;
  const size_t route_blocks = (live_routes + threads - 1) / threads;
  count_routes_kernel<<<static_cast<unsigned int>(route_blocks), threads, 0,
                        stream>>>(
      static_cast<const int32_t *>(buffers->topk_ids.ptr),
      static_cast<int32_t *>(buffers->expert_counts.ptr), live_routes);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    return status_from_cuda(error);
  }
  prefix_routes_kernel<<<1, threads, 0, stream>>>(
      static_cast<const int32_t *>(buffers->expert_counts.ptr),
      static_cast<int32_t *>(buffers->expert_offsets.ptr),
      static_cast<int32_t *>(buffers->packed_route_indices.ptr),
      static_cast<int32_t *>(buffers->block_expert_ids.ptr),
      static_cast<int32_t *>(buffers->packed_route_count.ptr),
      static_cast<int32_t>(live_routes));
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    return status_from_cuda(error);
  }
  scatter_routes_kernel<<<static_cast<unsigned int>(route_blocks), threads, 0,
                          stream>>>(
      static_cast<const int32_t *>(buffers->topk_ids.ptr),
      static_cast<int32_t *>(buffers->expert_offsets.ptr),
      static_cast<int32_t *>(buffers->packed_route_indices.ptr), live_routes);
  return status_from_cuda(cudaGetLastError());
}

int launch_topk6_sum(const ds4rt_ds4_pro_spark_exl3_k2_moe_buffers_t *buffers,
                     size_t active_m, cudaStream_t stream) {
  return cute_dsl_ds4rt_ds4_pro_tp4_exl3_k2_topk6_sum_wrapper(
      &pro_topk6_sum_module, buffers->routed_output.ptr,
      buffers->output_f32.ptr, buffers->topk_weights.ptr, buffers->topk_ids.ptr,
      buffers->expert_map.ptr, buffers->down_svh.ptr,
      static_cast<int32_t>(kExperts), static_cast<int32_t>(kExperts),
      static_cast<int32_t>(active_m), stream);
}

int launch_decode(const ds4rt_ds4_pro_spark_exl3_k2_moe_buffers_t *buffers,
                  cudaStream_t stream) {
  ds4rt_ds4_pro_tp4_exl3_k2_decode_m1_Tensor_fc1_bf16_flat_t fc1{
      buffers->fc1_output.ptr};
  ds4rt_ds4_pro_tp4_exl3_k2_decode_m1_Tensor_activated_bf16_flat_t activated{
      buffers->activated.ptr};
  ds4rt_ds4_pro_tp4_exl3_k2_decode_m1_Tensor_fc2_bf16_flat_t routed_output{
      buffers->routed_output.ptr};
  ds4rt_ds4_pro_tp4_exl3_k2_decode_m1_Tensor_packed_route_indices_t routes{
      buffers->topk_ids.ptr};
  ds4rt_ds4_pro_tp4_exl3_k2_decode_m1_Tensor_block_expert_ids_t block_experts{
      buffers->block_expert_ids.ptr};
  ds4rt_ds4_pro_tp4_exl3_k2_decode_m1_Tensor_packed_route_count_t route_count{
      buffers->packed_route_count.ptr};
  ds4rt_ds4_pro_tp4_exl3_k2_decode_m1_Tensor_activation_amax_flat_t
      activation_amax{buffers->global_scale.ptr};
  ds4rt_ds4_pro_tp4_exl3_k2_decode_m1_Tensor_fc1_c_tmp_f32_flat_t fc1_scratch{
      buffers->fc1_scratch.ptr};
  ds4rt_ds4_pro_tp4_exl3_k2_decode_m1_Tensor_fc2_c_tmp_f32_flat_t fc2_scratch{
      buffers->fc2_scratch.ptr};
  ds4rt_ds4_pro_tp4_exl3_k2_decode_m1_Tensor_locks_i32_flat_t workspace{
      buffers->workspace.ptr};
  return cute_dsl_ds4rt_ds4_pro_tp4_exl3_k2_decode_m1_wrapper(
      &pro_decode_module, buffers->rotation_gate.ptr, buffers->rotation_up.ptr,
      buffers->input.ptr, buffers->w13_trellis.ptr, buffers->w2_trellis.ptr,
      static_cast<int64_t>(buffers->w13_trellis.bytes / sizeof(int32_t)),
      static_cast<int64_t>(buffers->w2_trellis.bytes / sizeof(int32_t)),
      &fc1, &activated, &routed_output, buffers->dummy_scale.ptr,
      buffers->dummy_scale.ptr, buffers->global_scale.ptr,
      buffers->global_scale.ptr, &routes, &block_experts, &route_count,
      &activation_amax, 0, buffers->topk_weights.ptr, &fc1_scratch,
      &fc2_scratch, &workspace, buffers->intermediate_rotations.ptr,
      buffers->gate_suh.ptr, buffers->up_suh.ptr, buffers->expert_map.ptr,
      buffers->trellis_lut.ptr, buffers->trellis_lut.ptr,
      static_cast<int32_t>(kExperts), static_cast<int32_t>(kExperts), 1,
      DS4RT_DS4_PRO_EXL3_K2_DECODE_M1_GRID_X, stream);
}

using ProPrefillLaunchFn = int (*)(
    const ds4rt_ds4_pro_spark_exl3_k2_moe_buffers_t *, size_t, cudaStream_t);

#define DS4RT_DEFINE_PRO_PREFILL_LAUNCH(M)                                       \
  int launch_prefill_m##M(                                                       \
      const ds4rt_ds4_pro_spark_exl3_k2_moe_buffers_t *buffers,                  \
      size_t active_m, cudaStream_t stream) {                                    \
    ds4rt_ds4_pro_tp4_exl3_k2_prefill_m##M##_topk6_Tensor_fc1_bf16_flat_t fc1{   \
        buffers->fc1_output.ptr};                                                \
    ds4rt_ds4_pro_tp4_exl3_k2_prefill_m##M##_topk6_Tensor_activated_bf16_flat_t  \
        activated{buffers->activated.ptr};                                       \
    ds4rt_ds4_pro_tp4_exl3_k2_prefill_m##M##_topk6_Tensor_fc2_bf16_flat_t        \
        routed_output{buffers->routed_output.ptr};                               \
    ds4rt_ds4_pro_tp4_exl3_k2_prefill_m##M##_topk6_Tensor_packed_route_indices_t \
        routes{buffers->packed_route_indices.ptr};                               \
    ds4rt_ds4_pro_tp4_exl3_k2_prefill_m##M##_topk6_Tensor_block_expert_ids_t     \
        block_experts{buffers->block_expert_ids.ptr};                            \
    ds4rt_ds4_pro_tp4_exl3_k2_prefill_m##M##_topk6_Tensor_packed_route_count_t   \
        route_count{buffers->packed_route_count.ptr};                            \
    ds4rt_ds4_pro_tp4_exl3_k2_prefill_m##M##_topk6_Tensor_activation_amax_flat_t \
        activation_amax{buffers->global_scale.ptr};                              \
    ds4rt_ds4_pro_tp4_exl3_k2_prefill_m##M##_topk6_Tensor_fc1_c_tmp_f32_flat_t   \
        fc1_scratch{buffers->fc1_scratch.ptr};                                   \
    ds4rt_ds4_pro_tp4_exl3_k2_prefill_m##M##_topk6_Tensor_fc2_c_tmp_f32_flat_t   \
        fc2_scratch{buffers->fc2_scratch.ptr};                                   \
    ds4rt_ds4_pro_tp4_exl3_k2_prefill_m##M##_topk6_Tensor_locks_i32_flat_t       \
        workspace{buffers->workspace.ptr};                                       \
    return cute_dsl_ds4rt_ds4_pro_tp4_exl3_k2_prefill_m##M##_topk6_wrapper(      \
        &pro_prefill_m##M##_module, buffers->rotation_gate.ptr,                  \
        buffers->rotation_up.ptr, buffers->input.ptr,                            \
        buffers->w13_trellis.ptr, buffers->w2_trellis.ptr,                      \
        static_cast<int64_t>(buffers->w13_trellis.bytes / sizeof(int32_t)),       \
        static_cast<int64_t>(buffers->w2_trellis.bytes / sizeof(int32_t)),        \
        &fc1, &activated, &routed_output, buffers->dummy_scale.ptr,              \
        buffers->dummy_scale.ptr,                                                \
        buffers->global_scale.ptr, buffers->global_scale.ptr, &routes,           \
        &block_experts, &route_count, &activation_amax, 0,                       \
        buffers->topk_weights.ptr, &fc1_scratch, &fc2_scratch, &workspace,       \
        buffers->intermediate_rotations.ptr, buffers->gate_suh.ptr,              \
        buffers->up_suh.ptr, buffers->expert_map.ptr,                            \
        buffers->trellis_lut.ptr, buffers->trellis_lut.ptr,                      \
        static_cast<int32_t>(kExperts), static_cast<int32_t>(kExperts),          \
        static_cast<int32_t>(active_m),                                          \
        DS4RT_DS4_PRO_EXL3_K2_PREFILL_M##M##_TOPK6_GRID_X, stream);              \
  }
DS4RT_DEFINE_PRO_PREFILL_LAUNCH(2)
DS4RT_DEFINE_PRO_PREFILL_LAUNCH(4)
DS4RT_DEFINE_PRO_PREFILL_LAUNCH(8)
DS4RT_DEFINE_PRO_PREFILL_LAUNCH(16)
DS4RT_DEFINE_PRO_PREFILL_LAUNCH(32)
DS4RT_DEFINE_PRO_PREFILL_LAUNCH(64)
DS4RT_DEFINE_PRO_PREFILL_LAUNCH(128)
DS4RT_DEFINE_PRO_PREFILL_LAUNCH(256)
DS4RT_DEFINE_PRO_PREFILL_LAUNCH(512)
DS4RT_DEFINE_PRO_PREFILL_LAUNCH(1024)
DS4RT_DEFINE_PRO_PREFILL_LAUNCH(2048)
#undef DS4RT_DEFINE_PRO_PREFILL_LAUNCH

ProPrefillLaunchFn prefill_launcher(size_t capacity_rows) {
  switch (capacity_rows) {
  case 2:
    return &launch_prefill_m2;
  case 4:
    return &launch_prefill_m4;
  case 8:
    return &launch_prefill_m8;
  case 16:
    return &launch_prefill_m16;
  case 32:
    return &launch_prefill_m32;
  case 64:
    return &launch_prefill_m64;
  case 128:
    return &launch_prefill_m128;
  case 256:
    return &launch_prefill_m256;
  case 512:
    return &launch_prefill_m512;
  case 1024:
    return &launch_prefill_m1024;
  case 2048:
    return &launch_prefill_m2048;
  default:
    return nullptr;
  }
}

} // namespace

extern "C" ds4rt_status_t
ds4rt_cuda_ds4_pro_spark_aot_available(int *out_available) {
  if (out_available == nullptr) {
    return DS4RT_STATUS_INVALID_ARGUMENT;
  }
  *out_available = 1;
  return DS4RT_STATUS_OK;
}

extern "C" ds4rt_status_t ds4rt_cuda_ds4_pro_spark_aot_init(void) {
  std::call_once(pro_module_init_once,
                 initialize_pro_modules_on_current_device);
  return pro_module_init_status;
}

extern "C" ds4rt_status_t
ds4rt_cuda_ds4_pro_shared_expert_fp8_bf16_async(
    const ds4rt_ds4_flash_shared_expert_fp8_buffers_t *buffers, size_t rows,
    void *cuda_stream) {
  const ds4rt_status_t valid = validate_shared_buffers(buffers, rows);
  if (valid != DS4RT_STATUS_OK) {
    return valid;
  }
  const ds4rt_status_t initialized = ds4rt_cuda_ds4_pro_spark_aot_init();
  if (initialized != DS4RT_STATUS_OK) {
    return initialized;
  }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  auto *input = static_cast<uint16_t *>(buffers->input.ptr);
  auto *output = static_cast<uint16_t *>(buffers->output.ptr);
  for (size_t row_offset = 0; row_offset < rows;
       row_offset += kSharedChunkRows) {
    const size_t remaining_rows = rows - row_offset;
    const size_t chunk_rows = remaining_rows < kSharedChunkRows
                                  ? remaining_rows
                                  : kSharedChunkRows;
    void *chunk_input = input + row_offset * kHidden;
    void *chunk_output = output + row_offset * kHidden;
    int gate_status = 0;
    int up_status = 0;
    int down_status = 0;
    if (rows == 1) {
      gate_status = launch_pro_shared_up_m1(
          chunk_input, buffers->w1_weight.ptr, buffers->w1_scale_mma.ptr,
          buffers->gate.ptr, buffers->alpha.ptr, 1, stream);
      up_status = launch_pro_shared_up_m1(
          chunk_input, buffers->w3_weight.ptr, buffers->w3_scale_mma.ptr,
          buffers->up.ptr, buffers->alpha.ptr, 1, stream);
    } else {
      gate_status = launch_pro_shared_up_m8(
          chunk_input, buffers->w1_weight.ptr, buffers->w1_scale_mma.ptr,
          buffers->gate.ptr, buffers->alpha.ptr,
          static_cast<int32_t>(chunk_rows), stream);
      up_status = launch_pro_shared_up_m8(
          chunk_input, buffers->w3_weight.ptr, buffers->w3_scale_mma.ptr,
          buffers->up.ptr, buffers->alpha.ptr,
          static_cast<int32_t>(chunk_rows), stream);
    }
    if (gate_status != 0 || up_status != 0) {
      ds4rt_set_last_error_message(
          "DeepSeek-V4-Pro shared expert up-projection AOT launch failed");
      return DS4RT_STATUS_INTERNAL_ERROR;
    }
    constexpr size_t kThreads = 256;
    const size_t values = chunk_rows * kSharedIntermediate;
    const size_t blocks = (values + kThreads - 1) / kThreads;
    pro_shared_swiglu_kernel<<<static_cast<unsigned int>(blocks), kThreads, 0,
                               stream>>>(
        static_cast<const uint16_t *>(buffers->gate.ptr),
        static_cast<const uint16_t *>(buffers->up.ptr),
        static_cast<uint16_t *>(buffers->activated.ptr), values);
    if (rows == 1) {
      down_status = launch_pro_shared_down_m1(
          buffers->activated.ptr, buffers->w2_weight.ptr,
          buffers->w2_scale_mma.ptr, chunk_output, buffers->alpha.ptr, 1,
          stream);
    } else {
      down_status = launch_pro_shared_down_m8(
          buffers->activated.ptr, buffers->w2_weight.ptr,
          buffers->w2_scale_mma.ptr, chunk_output, buffers->alpha.ptr,
          static_cast<int32_t>(chunk_rows), stream);
    }
    if (down_status != 0) {
      ds4rt_set_last_error_message(
          "DeepSeek-V4-Pro shared expert down-projection AOT launch failed");
      return DS4RT_STATUS_INTERNAL_ERROR;
    }
  }
  return status_from_cuda(cudaGetLastError());
}

extern "C" ds4rt_status_t
ds4rt_cuda_ds4_pro_shared_expert_fp8_prefill_bf16_async(
    const ds4rt_ds4_pro_shared_expert_fp8_prefill_buffers_t *buffers,
    size_t rows, void *cuda_stream) {
  const ds4rt_status_t valid = validate_pro_prefill_buffers(buffers, rows);
  if (valid != DS4RT_STATUS_OK) {
    return valid;
  }
  const ds4rt_status_t initialized = ds4rt_cuda_ds4_pro_spark_aot_init();
  if (initialized != DS4RT_STATUS_OK) {
    return initialized;
  }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const int32_t live_rows = static_cast<int32_t>(rows);
  const int32_t input_grid = mxfp8_quant_grid_x(rows, kHidden);
  const int32_t activated_grid =
      mxfp8_quant_grid_x(rows, kSharedIntermediate);
  if (input_grid <= 0 || activated_grid <= 0) {
    ds4rt_set_last_error_message(
        "DeepSeek-V4-Pro shared expert could not resolve quantizer grid");
    return DS4RT_STATUS_INTERNAL_ERROR;
  }

  int status = launch_pro_shared_input_quant_m2048(
      buffers->base.input.ptr, buffers->input_q_values.ptr,
      buffers->input_q_scale_rows.ptr, buffers->input_q_scale_mma.ptr,
      live_rows, input_grid, stream);
  if (status != 0) {
    ds4rt_set_last_error_message(
        "DeepSeek-V4-Pro shared expert input quantization AOT launch failed");
    return DS4RT_STATUS_INTERNAL_ERROR;
  }
  status = launch_pro_shared_up_m2048(
      buffers->input_q_values.ptr, buffers->input_q_scale_mma.ptr,
      buffers->base.w1_weight.ptr, buffers->base.w1_scale_mma.ptr,
      buffers->base.gate.ptr, buffers->base.alpha.ptr, live_rows, stream);
  status |= launch_pro_shared_up_m2048(
      buffers->input_q_values.ptr, buffers->input_q_scale_mma.ptr,
      buffers->base.w3_weight.ptr, buffers->base.w3_scale_mma.ptr,
      buffers->base.up.ptr, buffers->base.alpha.ptr, live_rows, stream);
  if (status != 0) {
    ds4rt_set_last_error_message(
        "DeepSeek-V4-Pro shared expert full-width up-projection AOT launch failed");
    return DS4RT_STATUS_INTERNAL_ERROR;
  }

  constexpr size_t kThreads = 256;
  const size_t values = rows * kSharedIntermediate;
  const size_t blocks = (values + kThreads - 1) / kThreads;
  pro_shared_swiglu_kernel<<<static_cast<unsigned int>(blocks), kThreads, 0,
                             stream>>>(
      static_cast<const uint16_t *>(buffers->base.gate.ptr),
      static_cast<const uint16_t *>(buffers->base.up.ptr),
      static_cast<uint16_t *>(buffers->base.activated.ptr), values);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return status_from_cuda(error);
  }

  status = launch_pro_shared_activated_quant_m2048(
      buffers->base.activated.ptr, buffers->activated_q_values.ptr,
      buffers->activated_q_scale_rows.ptr,
      buffers->activated_q_scale_mma.ptr, live_rows, activated_grid, stream);
  if (status != 0) {
    ds4rt_set_last_error_message(
        "DeepSeek-V4-Pro shared expert activation quantization AOT launch failed");
    return DS4RT_STATUS_INTERNAL_ERROR;
  }
  status = launch_pro_shared_down_m2048(
      buffers->activated_q_values.ptr, buffers->activated_q_scale_mma.ptr,
      buffers->base.w2_weight.ptr, buffers->base.w2_scale_mma.ptr,
      buffers->base.output.ptr, buffers->base.alpha.ptr, live_rows, stream);
  if (status != 0) {
    ds4rt_set_last_error_message(
        "DeepSeek-V4-Pro shared expert full-width down-projection AOT launch failed");
    return DS4RT_STATUS_INTERNAL_ERROR;
  }
  return status_from_cuda(cudaGetLastError());
}

extern "C" ds4rt_status_t ds4rt_cuda_ds4_pro_spark_exl3_k2_decode_m1_async(
    const ds4rt_ds4_pro_spark_exl3_k2_moe_buffers_t *buffers,
    void *cuda_stream) {
  const ds4rt_status_t valid = validate_buffers(buffers, 1);
  if (valid != DS4RT_STATUS_OK) {
    return valid;
  }
  const ds4rt_status_t initialized = ds4rt_cuda_ds4_pro_spark_aot_init();
  if (initialized != DS4RT_STATUS_OK) {
    return initialized;
  }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  cudaError_t error = cudaMemsetAsync(
      buffers->workspace.ptr, 0, kWorkspaceElements * sizeof(int32_t), stream);
  if (error != cudaSuccess) {
    return status_from_cuda(error);
  }
  if (launch_decode(buffers, stream) != 0) {
    ds4rt_set_last_error_message(
        "DeepSeek-V4-Pro Spark TP4 EXL3 K2 decode M1 AOT launch failed");
    return DS4RT_STATUS_INTERNAL_ERROR;
  }
  if (launch_topk6_sum(buffers, 1, stream) != 0) {
    ds4rt_set_last_error_message(
        "DeepSeek-V4-Pro Spark TP4 EXL3 K2 decode sum launch failed");
    return DS4RT_STATUS_INTERNAL_ERROR;
  }
  return ds4rt_cuda_f32_to_bf16_async(
      static_cast<const float *>(buffers->output_f32.ptr),
      static_cast<uint16_t *>(buffers->output_bf16.ptr), kHidden, cuda_stream);
}

extern "C" ds4rt_status_t ds4rt_cuda_ds4_pro_spark_exl3_k2_prefill_topk6_async(
    const ds4rt_ds4_pro_spark_exl3_k2_moe_buffers_t *buffers, size_t rows,
    void *cuda_stream) {
  size_t capacity_rows = 2;
  while (capacity_rows < rows && capacity_rows < kPrefillMaxRows) {
    capacity_rows *= 2;
  }
  if (rows < 2 || rows > capacity_rows) {
    return DS4RT_STATUS_INVALID_ARGUMENT;
  }
  const ProPrefillLaunchFn launcher = prefill_launcher(capacity_rows);
  if (launcher == nullptr) {
    return DS4RT_STATUS_INVALID_ARGUMENT;
  }
  const ds4rt_status_t valid = validate_buffers(buffers, capacity_rows);
  if (valid != DS4RT_STATUS_OK) {
    return valid;
  }
  const ds4rt_status_t initialized = ds4rt_cuda_ds4_pro_spark_aot_init();
  if (initialized != DS4RT_STATUS_OK) {
    return initialized;
  }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  ds4rt_status_t status = pack_routes(buffers, rows, stream);
  if (status != DS4RT_STATUS_OK) {
    return status;
  }
  cudaError_t error = cudaMemsetAsync(
      buffers->workspace.ptr, 0, kWorkspaceElements * sizeof(int32_t), stream);
  if (error != cudaSuccess) {
    return status_from_cuda(error);
  }
  if (launcher(buffers, rows, stream) != 0) {
    ds4rt_set_last_error_message(
        "DeepSeek-V4-Pro Spark TP4 EXL3 K2 prefill AOT launch failed");
    return DS4RT_STATUS_INTERNAL_ERROR;
  }
  if (launch_topk6_sum(buffers, rows, stream) != 0) {
    ds4rt_set_last_error_message(
        "DeepSeek-V4-Pro Spark TP4 EXL3 K2 prefill sum launch failed");
    return DS4RT_STATUS_INTERNAL_ERROR;
  }
  return ds4rt_cuda_f32_to_bf16_async(
      static_cast<const float *>(buffers->output_f32.ptr),
      static_cast<uint16_t *>(buffers->output_bf16.ptr), rows * kHidden,
      cuda_stream);
}
