#include "common.h"

#include "ds4_flash_spark_moe_aot_config.h"
#include "ds4_flash_shared_down_m1.h"
#include "ds4_flash_shared_down_m8.h"
#include "ds4_flash_shared_up_m1.h"
#include "ds4_flash_shared_up_m8.h"
#include "ds4_flash_tp4_exl3_k2_decode_m1.h"
#include "ds4_flash_tp4_exl3_k2_decode_m4_direct_m4.h"
#include "ds4_flash_tp4_exl3_k2_decode_m8_direct_m6.h"
#include "ds4_flash_tp4_exl3_k2_prefill_m1024_topk6.h"
#include "ds4_flash_tp4_exl3_k2_prefill_m128_topk6.h"
#include "ds4_flash_tp4_exl3_k2_prefill_m16_topk6.h"
#include "ds4_flash_tp4_exl3_k2_prefill_m2048_topk6.h"
#include "ds4_flash_tp4_exl3_k2_prefill_m256_topk6.h"
#include "ds4_flash_tp4_exl3_k2_prefill_m2_topk6.h"
#include "ds4_flash_tp4_exl3_k2_prefill_m32_topk6.h"
#include "ds4_flash_tp4_exl3_k2_prefill_m4_topk6.h"
#include "ds4_flash_tp4_exl3_k2_prefill_m512_topk6.h"
#include "ds4_flash_tp4_exl3_k2_prefill_m64_topk6.h"
#include "ds4_flash_tp4_exl3_k2_prefill_m8_topk6.h"
#include "ds4_flash_tp4_exl3_k2_topk6_sum.h"
#include "ds4_flash_tp4_exl3_k3_decode_m1.h"
#include "ds4_flash_tp4_exl3_k3_decode_m4_direct_m4.h"
#include "ds4_flash_tp4_exl3_k3_decode_m8_direct_m6.h"
#include "ds4_flash_tp4_exl3_k3_prefill_m1024_topk6.h"
#include "ds4_flash_tp4_exl3_k3_prefill_m128_topk6.h"
#include "ds4_flash_tp4_exl3_k3_prefill_m16_topk6.h"
#include "ds4_flash_tp4_exl3_k3_prefill_m2048_topk6.h"
#include "ds4_flash_tp4_exl3_k3_prefill_m256_topk6.h"
#include "ds4_flash_tp4_exl3_k3_prefill_m2_topk6.h"
#include "ds4_flash_tp4_exl3_k3_prefill_m32_topk6.h"
#include "ds4_flash_tp4_exl3_k3_prefill_m4_topk6.h"
#include "ds4_flash_tp4_exl3_k3_prefill_m512_topk6.h"
#include "ds4_flash_tp4_exl3_k3_prefill_m64_topk6.h"
#include "ds4_flash_tp4_exl3_k3_prefill_m8_topk6.h"
#include "ds4_flash_tp4_exl3_mixed_k2_k3_m1.h"
#include "ds4_flash_tp4_exl3_mixed_k2_k3_m2.h"
#include "ds4_flash_tp4_exl3_mixed_k2_k3_m3.h"
#include "ds4_flash_tp4_exl3_mixed_k2_k3_m4.h"
#include "ds4_flash_tp4_exl3_mixed_k2_k3_m5.h"
#include "ds4_flash_tp4_exl3_mixed_k2_k3_m6.h"
#include "ds4_flash_tp4_exl3_mixed_k2_k3_m7.h"
#include "ds4_flash_tp4_exl3_mixed_k2_k3_m8.h"
#include "ds4_flash_tp4_exl3_mixed_k2_k3_m9.h"
#include "ds4_flash_tp4_exl3_mixed_k2_k3_m10.h"
#include "ds4_flash_tp4_exl3_mixed_k2_k3_m11.h"
#include "ds4_flash_tp4_exl3_mixed_k2_k3_m12.h"
#include "ds4_flash_tp4_exl3_mixed_k2_k3_m16.h"
#include "ds4_flash_tp4_exl3_mixed_k2_k3_m32.h"
#include "ds4_flash_tp4_exl3_mixed_k2_k3_m64.h"
#include "ds4_flash_tp4_exl3_mixed_k2_k3_m128.h"
#include "ds4_flash_tp4_exl3_mixed_k2_k3_m256.h"
#include "ds4_flash_tp4_exl3_mixed_k2_k3_m512.h"
#include "ds4_flash_tp4_exl3_mixed_k2_k3_m1024.h"
#include "ds4_flash_tp4_exl3_mixed_k2_k3_m2048.h"
#include "ds4_flash_tp4_w4a16_decode_m1_fused_sum.h"
#include "ds4_flash_tp4_w4a16_prefill_m1024_topk6.h"
#include "ds4_flash_tp4_w4a16_prefill_m128_topk6.h"
#include "ds4_flash_tp4_w4a16_prefill_m16_topk6.h"
#include "ds4_flash_tp4_w4a16_prefill_m2048_topk6.h"
#include "ds4_flash_tp4_w4a16_prefill_m256_topk6.h"
#include "ds4_flash_tp4_w4a16_prefill_m2_topk6.h"
#include "ds4_flash_tp4_w4a16_prefill_m32_topk6.h"
#include "ds4_flash_tp4_w4a16_prefill_m4_topk6.h"
#include "ds4_flash_tp4_w4a16_prefill_m512_topk6.h"
#include "ds4_flash_tp4_w4a16_prefill_m64_topk6.h"
#include "ds4_flash_tp4_w4a16_prefill_m8_topk6.h"

#include <algorithm>
#include <mutex>
#include <cuda_bf16.h>

namespace {

using FlashKernel =
    ds41rt_ds4_flash_tp4_w4a16_decode_m1_fused_sum_Kernel_Module_t;
using FlashFc1 =
    ds41rt_ds4_flash_tp4_w4a16_decode_m1_fused_sum_Tensor_fc1_bf16_flat_t;
using FlashActivated =
    ds41rt_ds4_flash_tp4_w4a16_decode_m1_fused_sum_Tensor_activated_bf16_flat_t;
using FlashFc2 =
    ds41rt_ds4_flash_tp4_w4a16_decode_m1_fused_sum_Tensor_fc2_bf16_flat_t;
using FlashRoutes =
    ds41rt_ds4_flash_tp4_w4a16_decode_m1_fused_sum_Tensor_packed_route_indices_t;
using FlashBlockExperts =
    ds41rt_ds4_flash_tp4_w4a16_decode_m1_fused_sum_Tensor_block_expert_ids_t;
using FlashRouteCount =
    ds41rt_ds4_flash_tp4_w4a16_decode_m1_fused_sum_Tensor_packed_route_count_t;
using FlashActivationAmax =
    ds41rt_ds4_flash_tp4_w4a16_decode_m1_fused_sum_Tensor_activation_amax_flat_t;
using FlashFc1Scratch =
    ds41rt_ds4_flash_tp4_w4a16_decode_m1_fused_sum_Tensor_fc1_c_tmp_f32_flat_t;
using FlashFc2Scratch =
    ds41rt_ds4_flash_tp4_w4a16_decode_m1_fused_sum_Tensor_fc2_c_tmp_f32_flat_t;
using FlashLocks =
    ds41rt_ds4_flash_tp4_w4a16_decode_m1_fused_sum_Tensor_locks_i32_flat_t;

constexpr size_t kHidden = DS41RT_DS4_FLASH_HIDDEN_SIZE;
constexpr size_t kIntermediate = DS41RT_DS4_FLASH_TP_INTERMEDIATE_SIZE;
constexpr size_t kExperts = DS41RT_DS4_FLASH_NUM_EXPERTS;
constexpr size_t kTopK = DS41RT_DS4_FLASH_TOP_K;
constexpr size_t kW13Rows = 2 * kIntermediate;
constexpr size_t kW13WeightBytes = kExperts * kW13Rows * kHidden / 2;
constexpr size_t kW2WeightBytes = kExperts * kHidden * kIntermediate / 2;
constexpr size_t kW13ScaleBytes = kExperts * kW13Rows * (kHidden / 32);
constexpr size_t kW2ScaleBytes = kExperts * kHidden * (kIntermediate / 32);
constexpr size_t kFc1Elements = kTopK * kW13Rows;
constexpr size_t kActivatedElements = kTopK * kIntermediate;
constexpr size_t kFc1ScratchElements = 98'304;
constexpr size_t kFc2ScratchElements = 393'216;
constexpr size_t kLockElements = 48 * 4 + 2;
constexpr size_t kPrefillMaxRows = DS41RT_DS4_FLASH_PREFILL_MAX_ROWS;
constexpr size_t kPrefillMaxPackedRouteSlots =
    DS41RT_DS4_FLASH_W4A16_PREFILL_M2048_TOPK6_PACKED_ROUTE_SLOTS;
constexpr size_t kPrefillMaxRouteBlocks =
    DS41RT_DS4_FLASH_W4A16_PREFILL_M2048_TOPK6_MAX_M_BLOCKS;
constexpr size_t kPrefillRouteBlockRows = 32;
constexpr size_t kPrefillScratchElements = 3'145'728;
constexpr size_t kExl3K2Bits = 2;
constexpr size_t kExl3K3Bits = 3;
constexpr size_t exl3_w13_bytes(size_t bits) {
  return 2 * kExperts * (kHidden / 16) * (kIntermediate / 16) *
         (16 * bits) * sizeof(int16_t);
}
constexpr size_t exl3_w2_bytes(size_t bits) {
  return kExperts * (kIntermediate / 16) * (kHidden / 16) * (16 * bits) *
         sizeof(int16_t);
}
constexpr size_t kExl3HiddenRotationBytes =
    kExperts * kHidden * sizeof(uint16_t);
constexpr size_t kExl3IntermediateRotationBytes =
    kExperts * 3 * kIntermediate * sizeof(uint16_t);
constexpr size_t kExl3TrellisLutBytes = 1 << 12;
constexpr size_t kExl3WorkspaceElements = 48 * 4 + 2;
constexpr size_t kSharedIntermediate =
    DS41RT_DS4_FLASH_SHARED_INTERMEDIATE_SIZE;
constexpr size_t kSharedChunkRows = 8;
constexpr size_t kSharedWeightBytes = kHidden * kSharedIntermediate;
constexpr size_t kSharedScaleMmaBytes =
    (kHidden / 128) * (kSharedIntermediate / 128) * 512;

FlashKernel flash_decode_module;
ds41rt_ds4_flash_tp4_exl3_k2_decode_m1_Kernel_Module_t
    flash_exl3_decode_module;
ds41rt_ds4_flash_tp4_exl3_k2_decode_m4_direct_m4_Kernel_Module_t
    flash_exl3_decode_m4_direct_m4_module;
ds41rt_ds4_flash_tp4_exl3_k2_decode_m8_direct_m6_Kernel_Module_t
    flash_exl3_decode_m8_direct_m6_module;
ds41rt_ds4_flash_tp4_exl3_k3_decode_m1_Kernel_Module_t
    flash_exl3_k3_decode_module;
ds41rt_ds4_flash_tp4_exl3_k3_decode_m4_direct_m4_Kernel_Module_t
    flash_exl3_k3_decode_m4_direct_m4_module;
ds41rt_ds4_flash_tp4_exl3_k3_decode_m8_direct_m6_Kernel_Module_t
    flash_exl3_k3_decode_m8_direct_m6_module;
ds41rt_ds4_flash_tp4_exl3_k2_topk6_sum_Kernel_Module_t
    flash_exl3_topk6_sum_module;
ds41rt_ds4_flash_shared_up_m1_Kernel_Module_t flash_shared_up_m1_module;
ds41rt_ds4_flash_shared_up_m8_Kernel_Module_t flash_shared_up_m8_module;
ds41rt_ds4_flash_shared_down_m1_Kernel_Module_t flash_shared_down_m1_module;
ds41rt_ds4_flash_shared_down_m8_Kernel_Module_t flash_shared_down_m8_module;
#define DS41RT_DEFINE_FLASH_PREFILL_MODULE(M)                                   \
  ds41rt_ds4_flash_tp4_w4a16_prefill_m##M##_topk6_Kernel_Module_t               \
      flash_prefill_m##M##_module;
DS41RT_DEFINE_FLASH_PREFILL_MODULE(2)
DS41RT_DEFINE_FLASH_PREFILL_MODULE(4)
DS41RT_DEFINE_FLASH_PREFILL_MODULE(8)
DS41RT_DEFINE_FLASH_PREFILL_MODULE(16)
DS41RT_DEFINE_FLASH_PREFILL_MODULE(32)
DS41RT_DEFINE_FLASH_PREFILL_MODULE(64)
DS41RT_DEFINE_FLASH_PREFILL_MODULE(128)
DS41RT_DEFINE_FLASH_PREFILL_MODULE(256)
DS41RT_DEFINE_FLASH_PREFILL_MODULE(512)
DS41RT_DEFINE_FLASH_PREFILL_MODULE(1024)
DS41RT_DEFINE_FLASH_PREFILL_MODULE(2048)
#undef DS41RT_DEFINE_FLASH_PREFILL_MODULE
#define DS41RT_DEFINE_FLASH_EXL3_PREFILL_MODULE(M)                             \
  ds41rt_ds4_flash_tp4_exl3_k2_prefill_m##M##_topk6_Kernel_Module_t            \
      flash_exl3_prefill_m##M##_module;
DS41RT_DEFINE_FLASH_EXL3_PREFILL_MODULE(2)
DS41RT_DEFINE_FLASH_EXL3_PREFILL_MODULE(4)
DS41RT_DEFINE_FLASH_EXL3_PREFILL_MODULE(8)
DS41RT_DEFINE_FLASH_EXL3_PREFILL_MODULE(16)
DS41RT_DEFINE_FLASH_EXL3_PREFILL_MODULE(32)
DS41RT_DEFINE_FLASH_EXL3_PREFILL_MODULE(64)
DS41RT_DEFINE_FLASH_EXL3_PREFILL_MODULE(128)
DS41RT_DEFINE_FLASH_EXL3_PREFILL_MODULE(256)
DS41RT_DEFINE_FLASH_EXL3_PREFILL_MODULE(512)
DS41RT_DEFINE_FLASH_EXL3_PREFILL_MODULE(1024)
DS41RT_DEFINE_FLASH_EXL3_PREFILL_MODULE(2048)
#undef DS41RT_DEFINE_FLASH_EXL3_PREFILL_MODULE
#define DS41RT_DEFINE_FLASH_EXL3_K3_PREFILL_MODULE(M)                          \
  ds41rt_ds4_flash_tp4_exl3_k3_prefill_m##M##_topk6_Kernel_Module_t           \
      flash_exl3_k3_prefill_m##M##_module;
DS41RT_DEFINE_FLASH_EXL3_K3_PREFILL_MODULE(2)
DS41RT_DEFINE_FLASH_EXL3_K3_PREFILL_MODULE(4)
DS41RT_DEFINE_FLASH_EXL3_K3_PREFILL_MODULE(8)
DS41RT_DEFINE_FLASH_EXL3_K3_PREFILL_MODULE(16)
DS41RT_DEFINE_FLASH_EXL3_K3_PREFILL_MODULE(32)
DS41RT_DEFINE_FLASH_EXL3_K3_PREFILL_MODULE(64)
DS41RT_DEFINE_FLASH_EXL3_K3_PREFILL_MODULE(128)
DS41RT_DEFINE_FLASH_EXL3_K3_PREFILL_MODULE(256)
DS41RT_DEFINE_FLASH_EXL3_K3_PREFILL_MODULE(512)
DS41RT_DEFINE_FLASH_EXL3_K3_PREFILL_MODULE(1024)
DS41RT_DEFINE_FLASH_EXL3_K3_PREFILL_MODULE(2048)
#undef DS41RT_DEFINE_FLASH_EXL3_K3_PREFILL_MODULE
#define DS41RT_DEFINE_FLASH_EXL3_MIXED_MODULE(M)                               \
  ds41rt_ds4_flash_tp4_exl3_mixed_k2_k3_m##M##_Kernel_Module_t                \
      flash_exl3_mixed_m##M##_module;
DS41RT_DEFINE_FLASH_EXL3_MIXED_MODULE(1)
DS41RT_DEFINE_FLASH_EXL3_MIXED_MODULE(2)
DS41RT_DEFINE_FLASH_EXL3_MIXED_MODULE(3)
DS41RT_DEFINE_FLASH_EXL3_MIXED_MODULE(4)
DS41RT_DEFINE_FLASH_EXL3_MIXED_MODULE(5)
DS41RT_DEFINE_FLASH_EXL3_MIXED_MODULE(6)
DS41RT_DEFINE_FLASH_EXL3_MIXED_MODULE(7)
DS41RT_DEFINE_FLASH_EXL3_MIXED_MODULE(8)
DS41RT_DEFINE_FLASH_EXL3_MIXED_MODULE(9)
DS41RT_DEFINE_FLASH_EXL3_MIXED_MODULE(10)
DS41RT_DEFINE_FLASH_EXL3_MIXED_MODULE(11)
DS41RT_DEFINE_FLASH_EXL3_MIXED_MODULE(12)
DS41RT_DEFINE_FLASH_EXL3_MIXED_MODULE(16)
DS41RT_DEFINE_FLASH_EXL3_MIXED_MODULE(32)
DS41RT_DEFINE_FLASH_EXL3_MIXED_MODULE(64)
DS41RT_DEFINE_FLASH_EXL3_MIXED_MODULE(128)
DS41RT_DEFINE_FLASH_EXL3_MIXED_MODULE(256)
DS41RT_DEFINE_FLASH_EXL3_MIXED_MODULE(512)
DS41RT_DEFINE_FLASH_EXL3_MIXED_MODULE(1024)
DS41RT_DEFINE_FLASH_EXL3_MIXED_MODULE(2048)
#undef DS41RT_DEFINE_FLASH_EXL3_MIXED_MODULE
std::once_flag flash_module_init_once;
ds41rt_status_t flash_module_init_status = DS41RT_STATUS_OK;

bool has_bytes(ds41rt_device_buffer_t buffer, size_t required) {
  return buffer.ptr != nullptr && buffer.bytes >= required;
}

void initialize_flash_module_on_current_device() {
  int32_t device_id = 0;
  cudaError_t result = cudaGetDevice(&device_id);
  if (result != cudaSuccess) {
    ds41rt_set_last_error_message(cudaGetErrorString(result));
    flash_module_init_status = DS41RT_STATUS_INTERNAL_ERROR;
    return;
  }

#define DS41RT_INIT_FLASH_MODULE(prefix, module_value)                          \
  do {                                                                         \
    cudaLibrary_t *library = &(module_value).module;                           \
    result = cudaSuccess;                                                      \
    struct {                                                                   \
      cudaLibrary_t **library;                                                 \
      cudaError_t *result;                                                     \
    } init_args = {&library, &result};                                         \
    _mlir_##prefix##_cuda_init(reinterpret_cast<void **>(&init_args));         \
    if (result != cudaSuccess) {                                               \
      ds41rt_set_last_error_message(cudaGetErrorString(result));                \
      flash_module_init_status = DS41RT_STATUS_INTERNAL_ERROR;                  \
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
      ds41rt_set_last_error_message(cudaGetErrorString(result));                \
      flash_module_init_status = DS41RT_STATUS_INTERNAL_ERROR;                  \
      return;                                                                  \
    }                                                                          \
  } while (false)

  DS41RT_INIT_FLASH_MODULE(ds41rt_ds4_flash_tp4_w4a16_decode_m1_fused_sum,
                          flash_decode_module);
  DS41RT_INIT_FLASH_MODULE(ds41rt_ds4_flash_tp4_exl3_k2_decode_m1,
                          flash_exl3_decode_module);
  DS41RT_INIT_FLASH_MODULE(ds41rt_ds4_flash_tp4_exl3_k2_decode_m4_direct_m4,
                          flash_exl3_decode_m4_direct_m4_module);
  DS41RT_INIT_FLASH_MODULE(ds41rt_ds4_flash_tp4_exl3_k2_decode_m8_direct_m6,
                          flash_exl3_decode_m8_direct_m6_module);
  DS41RT_INIT_FLASH_MODULE(ds41rt_ds4_flash_tp4_exl3_k3_decode_m1,
                          flash_exl3_k3_decode_module);
  DS41RT_INIT_FLASH_MODULE(ds41rt_ds4_flash_tp4_exl3_k3_decode_m4_direct_m4,
                          flash_exl3_k3_decode_m4_direct_m4_module);
  DS41RT_INIT_FLASH_MODULE(ds41rt_ds4_flash_tp4_exl3_k3_decode_m8_direct_m6,
                          flash_exl3_k3_decode_m8_direct_m6_module);
  DS41RT_INIT_FLASH_MODULE(ds41rt_ds4_flash_tp4_exl3_k2_topk6_sum,
                          flash_exl3_topk6_sum_module);
  DS41RT_INIT_FLASH_MODULE(ds41rt_ds4_flash_shared_up_m1,
                          flash_shared_up_m1_module);
  DS41RT_INIT_FLASH_MODULE(ds41rt_ds4_flash_shared_up_m8,
                          flash_shared_up_m8_module);
  DS41RT_INIT_FLASH_MODULE(ds41rt_ds4_flash_shared_down_m1,
                          flash_shared_down_m1_module);
  DS41RT_INIT_FLASH_MODULE(ds41rt_ds4_flash_shared_down_m8,
                          flash_shared_down_m8_module);
#define DS41RT_INIT_FLASH_PREFILL_MODULE(M)                                     \
  DS41RT_INIT_FLASH_MODULE(ds41rt_ds4_flash_tp4_w4a16_prefill_m##M##_topk6,      \
                          flash_prefill_m##M##_module)
  DS41RT_INIT_FLASH_PREFILL_MODULE(2);
  DS41RT_INIT_FLASH_PREFILL_MODULE(4);
  DS41RT_INIT_FLASH_PREFILL_MODULE(8);
  DS41RT_INIT_FLASH_PREFILL_MODULE(16);
  DS41RT_INIT_FLASH_PREFILL_MODULE(32);
  DS41RT_INIT_FLASH_PREFILL_MODULE(64);
  DS41RT_INIT_FLASH_PREFILL_MODULE(128);
  DS41RT_INIT_FLASH_PREFILL_MODULE(256);
  DS41RT_INIT_FLASH_PREFILL_MODULE(512);
  DS41RT_INIT_FLASH_PREFILL_MODULE(1024);
  DS41RT_INIT_FLASH_PREFILL_MODULE(2048);
#undef DS41RT_INIT_FLASH_PREFILL_MODULE
#define DS41RT_INIT_FLASH_EXL3_PREFILL_MODULE(M)                               \
  DS41RT_INIT_FLASH_MODULE(ds41rt_ds4_flash_tp4_exl3_k2_prefill_m##M##_topk6,   \
                          flash_exl3_prefill_m##M##_module)
  DS41RT_INIT_FLASH_EXL3_PREFILL_MODULE(2);
  DS41RT_INIT_FLASH_EXL3_PREFILL_MODULE(4);
  DS41RT_INIT_FLASH_EXL3_PREFILL_MODULE(8);
  DS41RT_INIT_FLASH_EXL3_PREFILL_MODULE(16);
  DS41RT_INIT_FLASH_EXL3_PREFILL_MODULE(32);
  DS41RT_INIT_FLASH_EXL3_PREFILL_MODULE(64);
  DS41RT_INIT_FLASH_EXL3_PREFILL_MODULE(128);
  DS41RT_INIT_FLASH_EXL3_PREFILL_MODULE(256);
  DS41RT_INIT_FLASH_EXL3_PREFILL_MODULE(512);
  DS41RT_INIT_FLASH_EXL3_PREFILL_MODULE(1024);
  DS41RT_INIT_FLASH_EXL3_PREFILL_MODULE(2048);
#undef DS41RT_INIT_FLASH_EXL3_PREFILL_MODULE
#define DS41RT_INIT_FLASH_EXL3_K3_PREFILL_MODULE(M)                            \
  DS41RT_INIT_FLASH_MODULE(ds41rt_ds4_flash_tp4_exl3_k3_prefill_m##M##_topk6,   \
                          flash_exl3_k3_prefill_m##M##_module)
  DS41RT_INIT_FLASH_EXL3_K3_PREFILL_MODULE(2);
  DS41RT_INIT_FLASH_EXL3_K3_PREFILL_MODULE(4);
  DS41RT_INIT_FLASH_EXL3_K3_PREFILL_MODULE(8);
  DS41RT_INIT_FLASH_EXL3_K3_PREFILL_MODULE(16);
  DS41RT_INIT_FLASH_EXL3_K3_PREFILL_MODULE(32);
  DS41RT_INIT_FLASH_EXL3_K3_PREFILL_MODULE(64);
  DS41RT_INIT_FLASH_EXL3_K3_PREFILL_MODULE(128);
  DS41RT_INIT_FLASH_EXL3_K3_PREFILL_MODULE(256);
  DS41RT_INIT_FLASH_EXL3_K3_PREFILL_MODULE(512);
  DS41RT_INIT_FLASH_EXL3_K3_PREFILL_MODULE(1024);
  DS41RT_INIT_FLASH_EXL3_K3_PREFILL_MODULE(2048);
#undef DS41RT_INIT_FLASH_EXL3_K3_PREFILL_MODULE
#define DS41RT_INIT_FLASH_EXL3_MIXED_MODULE(M)                                \
  DS41RT_INIT_FLASH_MODULE(ds41rt_ds4_flash_tp4_exl3_mixed_k2_k3_m##M,         \
                          flash_exl3_mixed_m##M##_module)
  DS41RT_INIT_FLASH_EXL3_MIXED_MODULE(1);
  DS41RT_INIT_FLASH_EXL3_MIXED_MODULE(2);
  DS41RT_INIT_FLASH_EXL3_MIXED_MODULE(3);
  DS41RT_INIT_FLASH_EXL3_MIXED_MODULE(4);
  DS41RT_INIT_FLASH_EXL3_MIXED_MODULE(5);
  DS41RT_INIT_FLASH_EXL3_MIXED_MODULE(6);
  DS41RT_INIT_FLASH_EXL3_MIXED_MODULE(7);
  DS41RT_INIT_FLASH_EXL3_MIXED_MODULE(8);
  DS41RT_INIT_FLASH_EXL3_MIXED_MODULE(9);
  DS41RT_INIT_FLASH_EXL3_MIXED_MODULE(10);
  DS41RT_INIT_FLASH_EXL3_MIXED_MODULE(11);
  DS41RT_INIT_FLASH_EXL3_MIXED_MODULE(12);
  DS41RT_INIT_FLASH_EXL3_MIXED_MODULE(16);
  DS41RT_INIT_FLASH_EXL3_MIXED_MODULE(32);
  DS41RT_INIT_FLASH_EXL3_MIXED_MODULE(64);
  DS41RT_INIT_FLASH_EXL3_MIXED_MODULE(128);
  DS41RT_INIT_FLASH_EXL3_MIXED_MODULE(256);
  DS41RT_INIT_FLASH_EXL3_MIXED_MODULE(512);
  DS41RT_INIT_FLASH_EXL3_MIXED_MODULE(1024);
  DS41RT_INIT_FLASH_EXL3_MIXED_MODULE(2048);
#undef DS41RT_INIT_FLASH_EXL3_MIXED_MODULE
#undef DS41RT_INIT_FLASH_MODULE
}

bool is_aligned(ds41rt_device_buffer_t buffer, size_t alignment) {
  return reinterpret_cast<uintptr_t>(buffer.ptr) % alignment == 0;
}

bool exl3_buffers_share_device(
    const ds41rt_ds4_flash_spark_exl3_k2_moe_buffers_t *buffers) {
  const int device = buffers->input.device_id;
#define DS41RT_EXL3_SAME_DEVICE(field) buffers->field.device_id == device
  const bool same =
      DS41RT_EXL3_SAME_DEVICE(w13_trellis) &&
      DS41RT_EXL3_SAME_DEVICE(w2_trellis) &&
      DS41RT_EXL3_SAME_DEVICE(gate_suh) && DS41RT_EXL3_SAME_DEVICE(up_suh) &&
      DS41RT_EXL3_SAME_DEVICE(intermediate_rotations) &&
      DS41RT_EXL3_SAME_DEVICE(down_svh) &&
      DS41RT_EXL3_SAME_DEVICE(expert_map) &&
      DS41RT_EXL3_SAME_DEVICE(dummy_scale) &&
      DS41RT_EXL3_SAME_DEVICE(trellis_lut) &&
      DS41RT_EXL3_SAME_DEVICE(global_scale) &&
      DS41RT_EXL3_SAME_DEVICE(topk_ids) &&
      DS41RT_EXL3_SAME_DEVICE(topk_weights) &&
      DS41RT_EXL3_SAME_DEVICE(rotation_gate) &&
      DS41RT_EXL3_SAME_DEVICE(rotation_up) &&
      DS41RT_EXL3_SAME_DEVICE(fc1_output) &&
      DS41RT_EXL3_SAME_DEVICE(activated) &&
      DS41RT_EXL3_SAME_DEVICE(routed_output) &&
      DS41RT_EXL3_SAME_DEVICE(output_f32) &&
      DS41RT_EXL3_SAME_DEVICE(output_bf16) &&
      DS41RT_EXL3_SAME_DEVICE(packed_route_indices) &&
      DS41RT_EXL3_SAME_DEVICE(block_expert_ids) &&
      DS41RT_EXL3_SAME_DEVICE(packed_route_count) &&
      DS41RT_EXL3_SAME_DEVICE(expert_counts) &&
      DS41RT_EXL3_SAME_DEVICE(expert_offsets) &&
      DS41RT_EXL3_SAME_DEVICE(fc1_scratch) &&
      DS41RT_EXL3_SAME_DEVICE(fc2_scratch) &&
      DS41RT_EXL3_SAME_DEVICE(workspace);
#undef DS41RT_EXL3_SAME_DEVICE
  return same;
}

ds41rt_status_t validate_exl3_buffers(
    const ds41rt_ds4_flash_spark_exl3_k2_moe_buffers_t *buffers,
    size_t capacity_rows, size_t trellis_bits) {
  if (buffers == nullptr || capacity_rows == 0 ||
      capacity_rows > kPrefillMaxRows) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  const size_t routed_rows = capacity_rows * kTopK;
  const bool decode = capacity_rows == 1;
  const size_t packed_route_elements =
      decode ? kTopK : kPrefillMaxPackedRouteSlots;
  const size_t route_block_elements = decode ? 1 : kPrefillMaxRouteBlocks;
  const size_t fc1_scratch_elements =
      decode ? kFc1ScratchElements : kPrefillScratchElements;
  const size_t fc2_scratch_elements =
      decode ? kFc2ScratchElements : kPrefillScratchElements;
  const bool valid =
      has_bytes(buffers->input,
                capacity_rows * kHidden * sizeof(uint16_t)) &&
      has_bytes(buffers->w13_trellis, exl3_w13_bytes(trellis_bits)) &&
      has_bytes(buffers->w2_trellis, exl3_w2_bytes(trellis_bits)) &&
      has_bytes(buffers->gate_suh, kExl3HiddenRotationBytes) &&
      has_bytes(buffers->up_suh, kExl3HiddenRotationBytes) &&
      has_bytes(buffers->intermediate_rotations,
                kExl3IntermediateRotationBytes) &&
      has_bytes(buffers->down_svh, kExl3HiddenRotationBytes) &&
      has_bytes(buffers->expert_map, kExperts * sizeof(int32_t)) &&
      has_bytes(buffers->dummy_scale, 4) &&
      has_bytes(buffers->trellis_lut, kExl3TrellisLutBytes) &&
      has_bytes(buffers->global_scale, kExperts * sizeof(float)) &&
      has_bytes(buffers->topk_ids, routed_rows * sizeof(int32_t)) &&
      has_bytes(buffers->topk_weights, routed_rows * sizeof(float)) &&
      has_bytes(buffers->rotation_gate,
                routed_rows * kHidden * sizeof(uint16_t)) &&
      has_bytes(buffers->rotation_up,
                routed_rows * kHidden * sizeof(uint16_t)) &&
      has_bytes(buffers->fc1_output,
                routed_rows * 2 * kIntermediate * sizeof(uint16_t)) &&
      has_bytes(buffers->activated,
                routed_rows * kIntermediate * sizeof(uint16_t)) &&
      has_bytes(buffers->routed_output,
                routed_rows * kHidden * sizeof(uint16_t)) &&
      has_bytes(buffers->output_f32,
                capacity_rows * kHidden * sizeof(float)) &&
      has_bytes(buffers->output_bf16,
                capacity_rows * kHidden * sizeof(uint16_t)) &&
      has_bytes(buffers->packed_route_indices,
                packed_route_elements * sizeof(int32_t)) &&
      has_bytes(buffers->block_expert_ids,
                route_block_elements * sizeof(int32_t)) &&
      has_bytes(buffers->packed_route_count, sizeof(int32_t)) &&
      has_bytes(buffers->expert_counts, kExperts * sizeof(int32_t)) &&
      has_bytes(buffers->expert_offsets, (kExperts + 1) * sizeof(int32_t)) &&
      has_bytes(buffers->fc1_scratch,
                fc1_scratch_elements * sizeof(float)) &&
      has_bytes(buffers->fc2_scratch,
                fc2_scratch_elements * sizeof(float)) &&
      has_bytes(buffers->workspace,
                kExl3WorkspaceElements * sizeof(int32_t));
  if (!valid) {
    return DS41RT_STATUS_BUFFER_TOO_SMALL;
  }
  const bool aligned =
      is_aligned(buffers->input, 16) &&
      is_aligned(buffers->w13_trellis, 16) &&
      is_aligned(buffers->w2_trellis, 16) &&
      is_aligned(buffers->gate_suh, 16) &&
      is_aligned(buffers->up_suh, 16) &&
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
  if (!aligned || !exl3_buffers_share_device(buffers)) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  return DS41RT_STATUS_OK;
}

ds41rt_status_t
validate_buffers(const ds41rt_ds4_flash_spark_w4a16_moe_buffers_t *buffers) {
  if (buffers == nullptr) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  const bool valid =
      has_bytes(buffers->input, kHidden * sizeof(uint16_t)) &&
      has_bytes(buffers->w13_weight, kW13WeightBytes) &&
      has_bytes(buffers->w2_weight, kW2WeightBytes) &&
      has_bytes(buffers->fc1_output, kFc1Elements * sizeof(uint16_t)) &&
      has_bytes(buffers->activated, kActivatedElements * sizeof(uint16_t)) &&
      has_bytes(buffers->output, kHidden * sizeof(uint16_t)) &&
      has_bytes(buffers->w13_scale, kW13ScaleBytes) &&
      has_bytes(buffers->w2_scale, kW2ScaleBytes) &&
      has_bytes(buffers->w13_global_scale, kExperts * sizeof(float)) &&
      has_bytes(buffers->w2_global_scale, kExperts * sizeof(float)) &&
      has_bytes(buffers->packed_route_indices, kTopK * sizeof(int32_t)) &&
      has_bytes(buffers->block_expert_ids, sizeof(int32_t)) &&
      has_bytes(buffers->packed_route_count, sizeof(int32_t)) &&
      has_bytes(buffers->topk_weights, kTopK * sizeof(float)) &&
      has_bytes(buffers->fc1_scratch, kFc1ScratchElements * sizeof(float)) &&
      has_bytes(buffers->fc2_scratch, kFc2ScratchElements * sizeof(float)) &&
      has_bytes(buffers->locks, kLockElements * sizeof(int32_t));
  return valid ? DS41RT_STATUS_OK : DS41RT_STATUS_BUFFER_TOO_SMALL;
}

ds41rt_status_t validate_prefill_buffers(
    const ds41rt_ds4_flash_spark_w4a16_moe_buffers_t *buffers,
    size_t capacity_rows) {
  if (buffers == nullptr || capacity_rows < 2 ||
      capacity_rows > kPrefillMaxRows) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  const size_t routed_rows = capacity_rows * kTopK;
  const bool valid =
      has_bytes(buffers->input, capacity_rows * kHidden * sizeof(uint16_t)) &&
      has_bytes(buffers->w13_weight, kW13WeightBytes) &&
      has_bytes(buffers->w2_weight, kW2WeightBytes) &&
      has_bytes(buffers->fc1_output,
                routed_rows * kW13Rows * sizeof(uint16_t)) &&
      has_bytes(buffers->activated,
                routed_rows * kIntermediate * sizeof(uint16_t)) &&
      has_bytes(buffers->routed_output,
                routed_rows * kHidden * sizeof(uint16_t)) &&
      has_bytes(buffers->output, capacity_rows * kHidden * sizeof(uint16_t)) &&
      has_bytes(buffers->w13_scale, kW13ScaleBytes) &&
      has_bytes(buffers->w2_scale, kW2ScaleBytes) &&
      has_bytes(buffers->w13_global_scale, kExperts * sizeof(float)) &&
      has_bytes(buffers->w2_global_scale, kExperts * sizeof(float)) &&
      has_bytes(buffers->packed_route_indices,
                kPrefillMaxPackedRouteSlots * sizeof(int32_t)) &&
      has_bytes(buffers->block_expert_ids,
                kPrefillMaxRouteBlocks * sizeof(int32_t)) &&
      has_bytes(buffers->packed_route_count, sizeof(int32_t)) &&
      has_bytes(buffers->topk_weights, routed_rows * sizeof(float)) &&
      has_bytes(buffers->fc1_scratch,
                kPrefillScratchElements * sizeof(float)) &&
      has_bytes(buffers->fc2_scratch,
                kPrefillScratchElements * sizeof(float)) &&
      has_bytes(buffers->locks, kLockElements * sizeof(int32_t));
  return valid ? DS41RT_STATUS_OK : DS41RT_STATUS_BUFFER_TOO_SMALL;
}

ds41rt_status_t validate_route_pack_buffers(
    const ds41rt_ds4_flash_route_pack_buffers_t *buffers, size_t rows) {
  if (buffers == nullptr || rows == 0 || rows > kPrefillMaxRows) {
    ds41rt_set_last_error_message(
        "DeepSeek-V4-Flash route pack received invalid buffers or row count");
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  const size_t live_routes = rows * kTopK;
  const bool decode = rows == 1;
  const size_t packed_route_elements =
      decode ? kTopK : kPrefillMaxPackedRouteSlots;
  const size_t route_block_elements =
      decode ? kTopK : kPrefillMaxRouteBlocks;
  const bool valid =
      has_bytes(buffers->topk_ids, live_routes * sizeof(int32_t)) &&
      has_bytes(buffers->packed_route_indices,
                packed_route_elements * sizeof(int32_t)) &&
      has_bytes(buffers->block_expert_ids,
                route_block_elements * sizeof(int32_t)) &&
      has_bytes(buffers->packed_route_count, sizeof(int32_t)) &&
      has_bytes(buffers->expert_counts, kExperts * sizeof(int32_t)) &&
      has_bytes(buffers->expert_offsets, (kExperts + 1) * sizeof(int32_t));
  if (!valid) {
    ds41rt_set_last_error_message(
        "DeepSeek-V4-Flash route pack received an undersized buffer");
    return DS41RT_STATUS_BUFFER_TOO_SMALL;
  }
  const int32_t device_id = buffers->topk_ids.device_id;
#define DS41RT_REQUIRE_ROUTE_PACK_DEVICE(FIELD)                                \
  if (buffers->FIELD.device_id != device_id) {                               \
    ds41rt_set_last_error_message(                                             \
        "DeepSeek-V4-Flash route pack device mismatch: " #FIELD);           \
    return DS41RT_STATUS_INVALID_ARGUMENT;                                     \
  }
  DS41RT_REQUIRE_ROUTE_PACK_DEVICE(packed_route_indices);
  DS41RT_REQUIRE_ROUTE_PACK_DEVICE(block_expert_ids);
  DS41RT_REQUIRE_ROUTE_PACK_DEVICE(packed_route_count);
  DS41RT_REQUIRE_ROUTE_PACK_DEVICE(expert_counts);
  DS41RT_REQUIRE_ROUTE_PACK_DEVICE(expert_offsets);
#undef DS41RT_REQUIRE_ROUTE_PACK_DEVICE
  return DS41RT_STATUS_OK;
}

ds41rt_status_t validate_shared_buffers(
    const ds41rt_ds4_flash_shared_expert_fp8_buffers_t *buffers, size_t rows) {
  if (buffers == nullptr || rows == 0 || rows > kPrefillMaxRows) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
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
  return valid ? DS41RT_STATUS_OK : DS41RT_STATUS_BUFFER_TOO_SMALL;
}

__global__ void sum_flash_topk6_bf16_kernel(const uint16_t *routed,
                                            uint16_t *output, size_t rows) {
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t values = rows * kHidden;
  if (index >= values) {
    return;
  }
  const size_t row = index / kHidden;
  const size_t col = index % kHidden;
  float sum = 0.0f;
#pragma unroll
  for (size_t route = 0; route < kTopK; ++route) {
    sum += bf16_to_f32(routed[(row * kTopK + route) * kHidden + col]);
  }
  output[index] = f32_to_bf16(sum);
}

__global__ void count_flash_topk6_routes_kernel(const int32_t *topk_ids,
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

__global__ void prefix_flash_topk6_routes_kernel(
    const int32_t *expert_counts, int32_t *expert_offsets,
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
    packed_offset +=
        blocks * static_cast<int32_t>(kPrefillRouteBlockRows);
  }
  expert_offsets[kExperts] = packed_offset;
  packed_route_count[0] = packed_offset;
}

__global__ void scatter_flash_topk6_routes_kernel(
    const int32_t *topk_ids, int32_t *expert_offsets,
    int32_t *packed_route_indices, size_t live_routes) {
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

__global__ void flash_shared_swiglu_kernel(const uint16_t *gate,
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
  gate_value = fminf(gate_value, DS41RT_DS4_FLASH_SWIGLU_LIMIT);
  up_value = fminf(fmaxf(up_value, -DS41RT_DS4_FLASH_SWIGLU_LIMIT),
                   DS41RT_DS4_FLASH_SWIGLU_LIMIT);
  const float silu = gate_value / (1.0f + expf(-gate_value));
  reinterpret_cast<__nv_bfloat16 *>(activated)[index] =
      __float2bfloat16_rn(silu * up_value);
}

#define DS41RT_DEFINE_SHARED_LINEAR_LAUNCH(NAME, M)                             \
  int launch_shared_##NAME##_m##M(                                            \
      void *source, void *weight, void *scale, void *output, void *alpha,      \
      int32_t rows, cudaStream_t stream) {                                     \
    return cute_dsl_ds41rt_ds4_flash_shared_##NAME##_m##M##_wrapper(            \
        &flash_shared_##NAME##_m##M##_module, source, source, weight, source,  \
        scale, output, alpha, rows, stream);                                   \
  }
DS41RT_DEFINE_SHARED_LINEAR_LAUNCH(up, 1)
DS41RT_DEFINE_SHARED_LINEAR_LAUNCH(up, 8)
DS41RT_DEFINE_SHARED_LINEAR_LAUNCH(down, 1)
DS41RT_DEFINE_SHARED_LINEAR_LAUNCH(down, 8)
#undef DS41RT_DEFINE_SHARED_LINEAR_LAUNCH

using FlashPrefillLaunchFn = int (*)(
    const ds41rt_ds4_flash_spark_w4a16_moe_buffers_t *, size_t, cudaStream_t);

#define DS41RT_DEFINE_FLASH_PREFILL_LAUNCH(M)                                     \
  int launch_flash_prefill_m##M(                                                 \
      const ds41rt_ds4_flash_spark_w4a16_moe_buffers_t *buffers,                  \
      size_t active_m, cudaStream_t stream) {                                    \
    ds41rt_ds4_flash_tp4_w4a16_prefill_m##M##_topk6_Tensor_fc1_bf16_flat_t fc1{   \
        buffers->fc1_output.ptr};                                                \
    ds41rt_ds4_flash_tp4_w4a16_prefill_m##M##_topk6_Tensor_activated_bf16_flat_t  \
        activated{buffers->activated.ptr};                                       \
    ds41rt_ds4_flash_tp4_w4a16_prefill_m##M##_topk6_Tensor_fc2_bf16_flat_t        \
        routed_output{buffers->routed_output.ptr};                               \
    ds41rt_ds4_flash_tp4_w4a16_prefill_m##M##_topk6_Tensor_packed_route_indices_t \
        routes{buffers->packed_route_indices.ptr};                               \
    ds41rt_ds4_flash_tp4_w4a16_prefill_m##M##_topk6_Tensor_block_expert_ids_t     \
        block_experts{buffers->block_expert_ids.ptr};                            \
    ds41rt_ds4_flash_tp4_w4a16_prefill_m##M##_topk6_Tensor_packed_route_count_t   \
        route_count{buffers->packed_route_count.ptr};                            \
    ds41rt_ds4_flash_tp4_w4a16_prefill_m##M##_topk6_Tensor_activation_amax_flat_t \
        activation_amax{buffers->w13_global_scale.ptr};                          \
    ds41rt_ds4_flash_tp4_w4a16_prefill_m##M##_topk6_Tensor_fc1_c_tmp_f32_flat_t   \
        fc1_scratch{buffers->fc1_scratch.ptr};                                   \
    ds41rt_ds4_flash_tp4_w4a16_prefill_m##M##_topk6_Tensor_fc2_c_tmp_f32_flat_t   \
        fc2_scratch{buffers->fc2_scratch.ptr};                                   \
    ds41rt_ds4_flash_tp4_w4a16_prefill_m##M##_topk6_Tensor_locks_i32_flat_t       \
        locks{buffers->locks.ptr};                                               \
    return cute_dsl_ds41rt_ds4_flash_tp4_w4a16_prefill_m##M##_topk6_wrapper(      \
        &flash_prefill_m##M##_module, buffers->input.ptr, buffers->input.ptr,    \
        buffers->input.ptr, buffers->w13_weight.ptr, buffers->w2_weight.ptr,     \
        static_cast<int64_t>(buffers->w13_weight.bytes / sizeof(int32_t)),       \
        static_cast<int64_t>(buffers->w2_weight.bytes / sizeof(int32_t)),        \
        &fc1, &activated, &routed_output, buffers->w13_scale.ptr,                \
        buffers->w2_scale.ptr, buffers->w13_global_scale.ptr,                    \
        buffers->w2_global_scale.ptr, &routes, &block_experts, &route_count,     \
        &activation_amax, 0, buffers->topk_weights.ptr, &fc1_scratch,            \
        &fc2_scratch, &locks, buffers->w13_global_scale.ptr,                     \
        buffers->w13_global_scale.ptr, buffers->w13_global_scale.ptr,            \
        buffers->packed_route_indices.ptr, buffers->w13_scale.ptr,               \
        buffers->w13_scale.ptr, static_cast<int32_t>(kExperts), 0,               \
        static_cast<int32_t>(active_m),                                          \
        DS41RT_DS4_FLASH_W4A16_PREFILL_M##M##_TOPK6_GRID_X, stream);              \
  }
DS41RT_DEFINE_FLASH_PREFILL_LAUNCH(2)
DS41RT_DEFINE_FLASH_PREFILL_LAUNCH(4)
DS41RT_DEFINE_FLASH_PREFILL_LAUNCH(8)
DS41RT_DEFINE_FLASH_PREFILL_LAUNCH(16)
DS41RT_DEFINE_FLASH_PREFILL_LAUNCH(32)
DS41RT_DEFINE_FLASH_PREFILL_LAUNCH(64)
DS41RT_DEFINE_FLASH_PREFILL_LAUNCH(128)
DS41RT_DEFINE_FLASH_PREFILL_LAUNCH(256)
DS41RT_DEFINE_FLASH_PREFILL_LAUNCH(512)
DS41RT_DEFINE_FLASH_PREFILL_LAUNCH(1024)
DS41RT_DEFINE_FLASH_PREFILL_LAUNCH(2048)
#undef DS41RT_DEFINE_FLASH_PREFILL_LAUNCH

FlashPrefillLaunchFn flash_prefill_launcher(size_t capacity_rows) {
  switch (capacity_rows) {
  case 2:
    return &launch_flash_prefill_m2;
  case 4:
    return &launch_flash_prefill_m4;
  case 8:
    return &launch_flash_prefill_m8;
  case 16:
    return &launch_flash_prefill_m16;
  case 32:
    return &launch_flash_prefill_m32;
  case 64:
    return &launch_flash_prefill_m64;
  case 128:
    return &launch_flash_prefill_m128;
  case 256:
    return &launch_flash_prefill_m256;
  case 512:
    return &launch_flash_prefill_m512;
  case 1024:
    return &launch_flash_prefill_m1024;
  case 2048:
    return &launch_flash_prefill_m2048;
  default:
    return nullptr;
  }
}

int launch_flash_exl3_topk6_sum(
    const ds41rt_ds4_flash_spark_exl3_k2_moe_buffers_t *buffers,
    size_t active_m, cudaStream_t stream) {
  return cute_dsl_ds41rt_ds4_flash_tp4_exl3_k2_topk6_sum_wrapper(
      &flash_exl3_topk6_sum_module, buffers->routed_output.ptr,
      buffers->output_f32.ptr, buffers->topk_weights.ptr,
      buffers->topk_ids.ptr, buffers->expert_map.ptr, buffers->down_svh.ptr,
      static_cast<int32_t>(kExperts), static_cast<int32_t>(kExperts),
      static_cast<int32_t>(active_m), stream);
}

int launch_flash_exl3_decode(
    const ds41rt_ds4_flash_spark_exl3_k2_moe_buffers_t *buffers,
    cudaStream_t stream) {
  ds41rt_ds4_flash_tp4_exl3_k2_decode_m1_Tensor_fc1_bf16_flat_t fc1{
      buffers->fc1_output.ptr};
  ds41rt_ds4_flash_tp4_exl3_k2_decode_m1_Tensor_activated_bf16_flat_t
      activated{buffers->activated.ptr};
  ds41rt_ds4_flash_tp4_exl3_k2_decode_m1_Tensor_fc2_bf16_flat_t routed_output{
      buffers->routed_output.ptr};
  ds41rt_ds4_flash_tp4_exl3_k2_decode_m1_Tensor_packed_route_indices_t routes{
      buffers->topk_ids.ptr};
  ds41rt_ds4_flash_tp4_exl3_k2_decode_m1_Tensor_block_expert_ids_t
      block_experts{buffers->block_expert_ids.ptr};
  ds41rt_ds4_flash_tp4_exl3_k2_decode_m1_Tensor_packed_route_count_t
      route_count{buffers->packed_route_count.ptr};
  ds41rt_ds4_flash_tp4_exl3_k2_decode_m1_Tensor_activation_amax_flat_t
      activation_amax{buffers->global_scale.ptr};
  ds41rt_ds4_flash_tp4_exl3_k2_decode_m1_Tensor_fc1_c_tmp_f32_flat_t
      fc1_scratch{buffers->fc1_scratch.ptr};
  ds41rt_ds4_flash_tp4_exl3_k2_decode_m1_Tensor_fc2_c_tmp_f32_flat_t
      fc2_scratch{buffers->fc2_scratch.ptr};
  ds41rt_ds4_flash_tp4_exl3_k2_decode_m1_Tensor_locks_i32_flat_t workspace{
      buffers->workspace.ptr};
  return cute_dsl_ds41rt_ds4_flash_tp4_exl3_k2_decode_m1_wrapper(
      &flash_exl3_decode_module, buffers->rotation_gate.ptr,
      buffers->rotation_up.ptr, buffers->input.ptr, buffers->w13_trellis.ptr,
      buffers->w2_trellis.ptr,
      static_cast<int64_t>(buffers->w13_trellis.bytes / sizeof(int32_t)),
      static_cast<int64_t>(buffers->w2_trellis.bytes / sizeof(int32_t)), &fc1,
      &activated, &routed_output,
      buffers->dummy_scale.ptr, buffers->dummy_scale.ptr,
      buffers->global_scale.ptr, buffers->global_scale.ptr, &routes,
      &block_experts, &route_count, &activation_amax, 0,
      buffers->topk_weights.ptr, &fc1_scratch, &fc2_scratch, &workspace,
      buffers->intermediate_rotations.ptr, buffers->gate_suh.ptr,
      buffers->up_suh.ptr, buffers->expert_map.ptr,
      buffers->trellis_lut.ptr, buffers->trellis_lut.ptr,
      static_cast<int32_t>(kExperts), static_cast<int32_t>(kExperts), 1,
      DS41RT_DS4_FLASH_EXL3_K2_DECODE_M1_GRID_X, stream);
}

int launch_flash_exl3_decode_m8_direct_m6(
    const ds41rt_ds4_flash_spark_exl3_k2_moe_buffers_t *buffers,
    size_t active_m, cudaStream_t stream) {
  ds41rt_ds4_flash_tp4_exl3_k2_decode_m8_direct_m6_Tensor_fc1_bf16_flat_t fc1{
      buffers->fc1_output.ptr};
  ds41rt_ds4_flash_tp4_exl3_k2_decode_m8_direct_m6_Tensor_activated_bf16_flat_t
      activated{buffers->activated.ptr};
  ds41rt_ds4_flash_tp4_exl3_k2_decode_m8_direct_m6_Tensor_fc2_bf16_flat_t
      routed_output{buffers->routed_output.ptr};
  ds41rt_ds4_flash_tp4_exl3_k2_decode_m8_direct_m6_Tensor_packed_route_indices_t
      routes{buffers->topk_ids.ptr};
  ds41rt_ds4_flash_tp4_exl3_k2_decode_m8_direct_m6_Tensor_block_expert_ids_t
      block_experts{buffers->block_expert_ids.ptr};
  ds41rt_ds4_flash_tp4_exl3_k2_decode_m8_direct_m6_Tensor_packed_route_count_t
      route_count{buffers->packed_route_count.ptr};
  ds41rt_ds4_flash_tp4_exl3_k2_decode_m8_direct_m6_Tensor_activation_amax_flat_t
      activation_amax{buffers->global_scale.ptr};
  ds41rt_ds4_flash_tp4_exl3_k2_decode_m8_direct_m6_Tensor_fc1_c_tmp_f32_flat_t
      fc1_scratch{buffers->fc1_scratch.ptr};
  ds41rt_ds4_flash_tp4_exl3_k2_decode_m8_direct_m6_Tensor_fc2_c_tmp_f32_flat_t
      fc2_scratch{buffers->fc2_scratch.ptr};
  ds41rt_ds4_flash_tp4_exl3_k2_decode_m8_direct_m6_Tensor_locks_i32_flat_t
      workspace{buffers->workspace.ptr};
  return cute_dsl_ds41rt_ds4_flash_tp4_exl3_k2_decode_m8_direct_m6_wrapper(
      &flash_exl3_decode_m8_direct_m6_module, buffers->rotation_gate.ptr,
      buffers->rotation_up.ptr, buffers->input.ptr, buffers->w13_trellis.ptr,
      buffers->w2_trellis.ptr,
      static_cast<int64_t>(buffers->w13_trellis.bytes / sizeof(int32_t)),
      static_cast<int64_t>(buffers->w2_trellis.bytes / sizeof(int32_t)), &fc1,
      &activated, &routed_output,
      buffers->dummy_scale.ptr, buffers->dummy_scale.ptr,
      buffers->global_scale.ptr, buffers->global_scale.ptr, &routes,
      &block_experts, &route_count, &activation_amax, 0,
      buffers->topk_weights.ptr, &fc1_scratch, &fc2_scratch, &workspace,
      buffers->intermediate_rotations.ptr, buffers->gate_suh.ptr,
      buffers->up_suh.ptr, buffers->expert_map.ptr,
      buffers->trellis_lut.ptr, buffers->trellis_lut.ptr,
      static_cast<int32_t>(kExperts), static_cast<int32_t>(kExperts),
      static_cast<int32_t>(active_m),
      DS41RT_DS4_FLASH_EXL3_K2_DECODE_M8_DIRECT_M6_GRID_X, stream);
}

int launch_flash_exl3_decode_m4_direct_m4(
    const ds41rt_ds4_flash_spark_exl3_k2_moe_buffers_t *buffers,
    size_t active_m, cudaStream_t stream) {
  ds41rt_ds4_flash_tp4_exl3_k2_decode_m4_direct_m4_Tensor_fc1_bf16_flat_t fc1{
      buffers->fc1_output.ptr};
  ds41rt_ds4_flash_tp4_exl3_k2_decode_m4_direct_m4_Tensor_activated_bf16_flat_t
      activated{buffers->activated.ptr};
  ds41rt_ds4_flash_tp4_exl3_k2_decode_m4_direct_m4_Tensor_fc2_bf16_flat_t
      routed_output{buffers->routed_output.ptr};
  ds41rt_ds4_flash_tp4_exl3_k2_decode_m4_direct_m4_Tensor_packed_route_indices_t
      routes{buffers->topk_ids.ptr};
  ds41rt_ds4_flash_tp4_exl3_k2_decode_m4_direct_m4_Tensor_block_expert_ids_t
      block_experts{buffers->block_expert_ids.ptr};
  ds41rt_ds4_flash_tp4_exl3_k2_decode_m4_direct_m4_Tensor_packed_route_count_t
      route_count{buffers->packed_route_count.ptr};
  ds41rt_ds4_flash_tp4_exl3_k2_decode_m4_direct_m4_Tensor_activation_amax_flat_t
      activation_amax{buffers->global_scale.ptr};
  ds41rt_ds4_flash_tp4_exl3_k2_decode_m4_direct_m4_Tensor_fc1_c_tmp_f32_flat_t
      fc1_scratch{buffers->fc1_scratch.ptr};
  ds41rt_ds4_flash_tp4_exl3_k2_decode_m4_direct_m4_Tensor_fc2_c_tmp_f32_flat_t
      fc2_scratch{buffers->fc2_scratch.ptr};
  ds41rt_ds4_flash_tp4_exl3_k2_decode_m4_direct_m4_Tensor_locks_i32_flat_t
      workspace{buffers->workspace.ptr};
  return cute_dsl_ds41rt_ds4_flash_tp4_exl3_k2_decode_m4_direct_m4_wrapper(
      &flash_exl3_decode_m4_direct_m4_module, buffers->rotation_gate.ptr,
      buffers->rotation_up.ptr, buffers->input.ptr, buffers->w13_trellis.ptr,
      buffers->w2_trellis.ptr,
      static_cast<int64_t>(buffers->w13_trellis.bytes / sizeof(int32_t)),
      static_cast<int64_t>(buffers->w2_trellis.bytes / sizeof(int32_t)), &fc1,
      &activated, &routed_output,
      buffers->dummy_scale.ptr, buffers->dummy_scale.ptr,
      buffers->global_scale.ptr, buffers->global_scale.ptr, &routes,
      &block_experts, &route_count, &activation_amax, 0,
      buffers->topk_weights.ptr, &fc1_scratch, &fc2_scratch, &workspace,
      buffers->intermediate_rotations.ptr, buffers->gate_suh.ptr,
      buffers->up_suh.ptr, buffers->expert_map.ptr,
      buffers->trellis_lut.ptr, buffers->trellis_lut.ptr,
      static_cast<int32_t>(kExperts), static_cast<int32_t>(kExperts),
      static_cast<int32_t>(active_m),
      DS41RT_DS4_FLASH_EXL3_K2_DECODE_M4_DIRECT_M4_GRID_X, stream);
}

using FlashExl3PrefillLaunchFn = int (*)(
    const ds41rt_ds4_flash_spark_exl3_k2_moe_buffers_t *, size_t,
    cudaStream_t);

#define DS41RT_DEFINE_FLASH_EXL3_PREFILL_LAUNCH(M)                               \
  int launch_flash_exl3_prefill_m##M(                                           \
      const ds41rt_ds4_flash_spark_exl3_k2_moe_buffers_t *buffers,               \
      size_t active_m, cudaStream_t stream) {                                    \
    ds41rt_ds4_flash_tp4_exl3_k2_prefill_m##M##_topk6_Tensor_fc1_bf16_flat_t     \
        fc1{buffers->fc1_output.ptr};                                            \
    ds41rt_ds4_flash_tp4_exl3_k2_prefill_m##M##_topk6_Tensor_activated_bf16_flat_t \
        activated{buffers->activated.ptr};                                       \
    ds41rt_ds4_flash_tp4_exl3_k2_prefill_m##M##_topk6_Tensor_fc2_bf16_flat_t     \
        routed_output{buffers->routed_output.ptr};                               \
    ds41rt_ds4_flash_tp4_exl3_k2_prefill_m##M##_topk6_Tensor_packed_route_indices_t \
        routes{buffers->packed_route_indices.ptr};                               \
    ds41rt_ds4_flash_tp4_exl3_k2_prefill_m##M##_topk6_Tensor_block_expert_ids_t  \
        block_experts{buffers->block_expert_ids.ptr};                            \
    ds41rt_ds4_flash_tp4_exl3_k2_prefill_m##M##_topk6_Tensor_packed_route_count_t \
        route_count{buffers->packed_route_count.ptr};                            \
    ds41rt_ds4_flash_tp4_exl3_k2_prefill_m##M##_topk6_Tensor_activation_amax_flat_t \
        activation_amax{buffers->global_scale.ptr};                              \
    ds41rt_ds4_flash_tp4_exl3_k2_prefill_m##M##_topk6_Tensor_fc1_c_tmp_f32_flat_t \
        fc1_scratch{buffers->fc1_scratch.ptr};                                   \
    ds41rt_ds4_flash_tp4_exl3_k2_prefill_m##M##_topk6_Tensor_fc2_c_tmp_f32_flat_t \
        fc2_scratch{buffers->fc2_scratch.ptr};                                   \
    ds41rt_ds4_flash_tp4_exl3_k2_prefill_m##M##_topk6_Tensor_locks_i32_flat_t    \
        workspace{buffers->workspace.ptr};                                       \
    return cute_dsl_ds41rt_ds4_flash_tp4_exl3_k2_prefill_m##M##_topk6_wrapper(   \
        &flash_exl3_prefill_m##M##_module, buffers->rotation_gate.ptr,           \
        buffers->rotation_up.ptr, buffers->input.ptr, buffers->w13_trellis.ptr,  \
        buffers->w2_trellis.ptr,                                                 \
        static_cast<int64_t>(buffers->w13_trellis.bytes / sizeof(int32_t)),       \
        static_cast<int64_t>(buffers->w2_trellis.bytes / sizeof(int32_t)),        \
        &fc1, &activated, &routed_output,                                         \
        buffers->dummy_scale.ptr, buffers->dummy_scale.ptr,                      \
        buffers->global_scale.ptr, buffers->global_scale.ptr, &routes,           \
        &block_experts, &route_count, &activation_amax, 0,                       \
        buffers->topk_weights.ptr, &fc1_scratch, &fc2_scratch, &workspace,       \
        buffers->intermediate_rotations.ptr, buffers->gate_suh.ptr,              \
        buffers->up_suh.ptr, buffers->expert_map.ptr,                            \
        buffers->trellis_lut.ptr, buffers->trellis_lut.ptr,                      \
        static_cast<int32_t>(kExperts), static_cast<int32_t>(kExperts),          \
        static_cast<int32_t>(active_m),                                          \
        DS41RT_DS4_FLASH_EXL3_K2_PREFILL_M##M##_TOPK6_GRID_X, stream);            \
  }
DS41RT_DEFINE_FLASH_EXL3_PREFILL_LAUNCH(2)
DS41RT_DEFINE_FLASH_EXL3_PREFILL_LAUNCH(4)
DS41RT_DEFINE_FLASH_EXL3_PREFILL_LAUNCH(8)
DS41RT_DEFINE_FLASH_EXL3_PREFILL_LAUNCH(16)
DS41RT_DEFINE_FLASH_EXL3_PREFILL_LAUNCH(32)
DS41RT_DEFINE_FLASH_EXL3_PREFILL_LAUNCH(64)
DS41RT_DEFINE_FLASH_EXL3_PREFILL_LAUNCH(128)
DS41RT_DEFINE_FLASH_EXL3_PREFILL_LAUNCH(256)
DS41RT_DEFINE_FLASH_EXL3_PREFILL_LAUNCH(512)
DS41RT_DEFINE_FLASH_EXL3_PREFILL_LAUNCH(1024)
DS41RT_DEFINE_FLASH_EXL3_PREFILL_LAUNCH(2048)
#undef DS41RT_DEFINE_FLASH_EXL3_PREFILL_LAUNCH

FlashExl3PrefillLaunchFn flash_exl3_prefill_launcher(size_t capacity_rows) {
  switch (capacity_rows) {
  case 2:
    return &launch_flash_exl3_prefill_m2;
  case 4:
    return &launch_flash_exl3_prefill_m4;
  case 8:
    return &launch_flash_exl3_prefill_m8;
  case 16:
    return &launch_flash_exl3_prefill_m16;
  case 32:
    return &launch_flash_exl3_prefill_m32;
  case 64:
    return &launch_flash_exl3_prefill_m64;
  case 128:
    return &launch_flash_exl3_prefill_m128;
  case 256:
    return &launch_flash_exl3_prefill_m256;
  case 512:
    return &launch_flash_exl3_prefill_m512;
  case 1024:
    return &launch_flash_exl3_prefill_m1024;
  case 2048:
    return &launch_flash_exl3_prefill_m2048;
  default:
    return nullptr;
  }
}

int launch_flash_exl3_k3_decode(
    const ds41rt_ds4_flash_spark_exl3_k2_moe_buffers_t *buffers,
    cudaStream_t stream) {
  ds41rt_ds4_flash_tp4_exl3_k3_decode_m1_Tensor_fc1_bf16_flat_t fc1{
      buffers->fc1_output.ptr};
  ds41rt_ds4_flash_tp4_exl3_k3_decode_m1_Tensor_activated_bf16_flat_t
      activated{buffers->activated.ptr};
  ds41rt_ds4_flash_tp4_exl3_k3_decode_m1_Tensor_fc2_bf16_flat_t routed_output{
      buffers->routed_output.ptr};
  ds41rt_ds4_flash_tp4_exl3_k3_decode_m1_Tensor_packed_route_indices_t routes{
      buffers->topk_ids.ptr};
  ds41rt_ds4_flash_tp4_exl3_k3_decode_m1_Tensor_block_expert_ids_t block_experts{
      buffers->block_expert_ids.ptr};
  ds41rt_ds4_flash_tp4_exl3_k3_decode_m1_Tensor_packed_route_count_t route_count{
      buffers->packed_route_count.ptr};
  ds41rt_ds4_flash_tp4_exl3_k3_decode_m1_Tensor_activation_amax_flat_t
      activation_amax{buffers->global_scale.ptr};
  ds41rt_ds4_flash_tp4_exl3_k3_decode_m1_Tensor_fc1_c_tmp_f32_flat_t fc1_scratch{
      buffers->fc1_scratch.ptr};
  ds41rt_ds4_flash_tp4_exl3_k3_decode_m1_Tensor_fc2_c_tmp_f32_flat_t fc2_scratch{
      buffers->fc2_scratch.ptr};
  ds41rt_ds4_flash_tp4_exl3_k3_decode_m1_Tensor_locks_i32_flat_t workspace{
      buffers->workspace.ptr};
  return cute_dsl_ds41rt_ds4_flash_tp4_exl3_k3_decode_m1_wrapper(
      &flash_exl3_k3_decode_module, buffers->rotation_gate.ptr,
      buffers->rotation_up.ptr, buffers->input.ptr, buffers->w13_trellis.ptr,
      buffers->w2_trellis.ptr,
      static_cast<int64_t>(buffers->w13_trellis.bytes / sizeof(int32_t)),
      static_cast<int64_t>(buffers->w2_trellis.bytes / sizeof(int32_t)), &fc1,
      &activated, &routed_output,
      buffers->dummy_scale.ptr, buffers->dummy_scale.ptr,
      buffers->global_scale.ptr, buffers->global_scale.ptr, &routes,
      &block_experts, &route_count, &activation_amax, 0,
      buffers->topk_weights.ptr, &fc1_scratch, &fc2_scratch, &workspace,
      buffers->intermediate_rotations.ptr, buffers->gate_suh.ptr,
      buffers->up_suh.ptr, buffers->expert_map.ptr, buffers->trellis_lut.ptr,
      buffers->trellis_lut.ptr, static_cast<int32_t>(kExperts),
      static_cast<int32_t>(kExperts), 1,
      DS41RT_DS4_FLASH_EXL3_K3_DECODE_M1_GRID_X, stream);
}

#define DS41RT_DEFINE_FLASH_EXL3_K3_DIRECT_LAUNCH(CAPACITY, LABEL)              \
  int launch_flash_exl3_k3_##LABEL(                                           \
      const ds41rt_ds4_flash_spark_exl3_k2_moe_buffers_t *buffers,             \
      size_t active_m, cudaStream_t stream) {                                  \
    ds41rt_ds4_flash_tp4_exl3_k3_##LABEL##_Tensor_fc1_bf16_flat_t fc1{         \
        buffers->fc1_output.ptr};                                              \
    ds41rt_ds4_flash_tp4_exl3_k3_##LABEL##_Tensor_activated_bf16_flat_t        \
        activated{buffers->activated.ptr};                                     \
    ds41rt_ds4_flash_tp4_exl3_k3_##LABEL##_Tensor_fc2_bf16_flat_t              \
        routed_output{buffers->routed_output.ptr};                             \
    ds41rt_ds4_flash_tp4_exl3_k3_##LABEL##_Tensor_packed_route_indices_t       \
        routes{buffers->topk_ids.ptr};                                         \
    ds41rt_ds4_flash_tp4_exl3_k3_##LABEL##_Tensor_block_expert_ids_t           \
        block_experts{buffers->block_expert_ids.ptr};                          \
    ds41rt_ds4_flash_tp4_exl3_k3_##LABEL##_Tensor_packed_route_count_t         \
        route_count{buffers->packed_route_count.ptr};                          \
    ds41rt_ds4_flash_tp4_exl3_k3_##LABEL##_Tensor_activation_amax_flat_t       \
        activation_amax{buffers->global_scale.ptr};                            \
    ds41rt_ds4_flash_tp4_exl3_k3_##LABEL##_Tensor_fc1_c_tmp_f32_flat_t         \
        fc1_scratch{buffers->fc1_scratch.ptr};                                 \
    ds41rt_ds4_flash_tp4_exl3_k3_##LABEL##_Tensor_fc2_c_tmp_f32_flat_t         \
        fc2_scratch{buffers->fc2_scratch.ptr};                                 \
    ds41rt_ds4_flash_tp4_exl3_k3_##LABEL##_Tensor_locks_i32_flat_t workspace{  \
        buffers->workspace.ptr};                                               \
    return cute_dsl_ds41rt_ds4_flash_tp4_exl3_k3_##LABEL##_wrapper(            \
        &flash_exl3_k3_##LABEL##_module, buffers->rotation_gate.ptr,           \
        buffers->rotation_up.ptr, buffers->input.ptr, buffers->w13_trellis.ptr,\
        buffers->w2_trellis.ptr,                                               \
        static_cast<int64_t>(buffers->w13_trellis.bytes / sizeof(int32_t)),     \
        static_cast<int64_t>(buffers->w2_trellis.bytes / sizeof(int32_t)),      \
        &fc1, &activated, &routed_output,                                       \
        buffers->dummy_scale.ptr, buffers->dummy_scale.ptr,                    \
        buffers->global_scale.ptr, buffers->global_scale.ptr, &routes,         \
        &block_experts, &route_count, &activation_amax, 0,                     \
        buffers->topk_weights.ptr, &fc1_scratch, &fc2_scratch, &workspace,     \
        buffers->intermediate_rotations.ptr, buffers->gate_suh.ptr,            \
        buffers->up_suh.ptr, buffers->expert_map.ptr, buffers->trellis_lut.ptr,\
        buffers->trellis_lut.ptr, static_cast<int32_t>(kExperts),              \
        static_cast<int32_t>(kExperts), static_cast<int32_t>(active_m),        \
        DS41RT_DS4_FLASH_EXL3_K3_##CAPACITY##_GRID_X, stream);                  \
  }
DS41RT_DEFINE_FLASH_EXL3_K3_DIRECT_LAUNCH(DECODE_M4_DIRECT_M4,
                                         decode_m4_direct_m4)
DS41RT_DEFINE_FLASH_EXL3_K3_DIRECT_LAUNCH(DECODE_M8_DIRECT_M6,
                                         decode_m8_direct_m6)
#undef DS41RT_DEFINE_FLASH_EXL3_K3_DIRECT_LAUNCH

#define DS41RT_DEFINE_FLASH_EXL3_K3_PREFILL_LAUNCH(M)                           \
  int launch_flash_exl3_k3_prefill_m##M(                                      \
      const ds41rt_ds4_flash_spark_exl3_k2_moe_buffers_t *buffers,             \
      size_t active_m, cudaStream_t stream) {                                  \
    ds41rt_ds4_flash_tp4_exl3_k3_prefill_m##M##_topk6_Tensor_fc1_bf16_flat_t   \
        fc1{buffers->fc1_output.ptr};                                          \
    ds41rt_ds4_flash_tp4_exl3_k3_prefill_m##M##_topk6_Tensor_activated_bf16_flat_t \
        activated{buffers->activated.ptr};                                     \
    ds41rt_ds4_flash_tp4_exl3_k3_prefill_m##M##_topk6_Tensor_fc2_bf16_flat_t   \
        routed_output{buffers->routed_output.ptr};                             \
    ds41rt_ds4_flash_tp4_exl3_k3_prefill_m##M##_topk6_Tensor_packed_route_indices_t \
        routes{buffers->packed_route_indices.ptr};                             \
    ds41rt_ds4_flash_tp4_exl3_k3_prefill_m##M##_topk6_Tensor_block_expert_ids_t \
        block_experts{buffers->block_expert_ids.ptr};                          \
    ds41rt_ds4_flash_tp4_exl3_k3_prefill_m##M##_topk6_Tensor_packed_route_count_t \
        route_count{buffers->packed_route_count.ptr};                          \
    ds41rt_ds4_flash_tp4_exl3_k3_prefill_m##M##_topk6_Tensor_activation_amax_flat_t \
        activation_amax{buffers->global_scale.ptr};                            \
    ds41rt_ds4_flash_tp4_exl3_k3_prefill_m##M##_topk6_Tensor_fc1_c_tmp_f32_flat_t \
        fc1_scratch{buffers->fc1_scratch.ptr};                                 \
    ds41rt_ds4_flash_tp4_exl3_k3_prefill_m##M##_topk6_Tensor_fc2_c_tmp_f32_flat_t \
        fc2_scratch{buffers->fc2_scratch.ptr};                                 \
    ds41rt_ds4_flash_tp4_exl3_k3_prefill_m##M##_topk6_Tensor_locks_i32_flat_t  \
        workspace{buffers->workspace.ptr};                                     \
    return cute_dsl_ds41rt_ds4_flash_tp4_exl3_k3_prefill_m##M##_topk6_wrapper( \
        &flash_exl3_k3_prefill_m##M##_module, buffers->rotation_gate.ptr,      \
        buffers->rotation_up.ptr, buffers->input.ptr, buffers->w13_trellis.ptr,\
        buffers->w2_trellis.ptr,                                               \
        static_cast<int64_t>(buffers->w13_trellis.bytes / sizeof(int32_t)),     \
        static_cast<int64_t>(buffers->w2_trellis.bytes / sizeof(int32_t)),      \
        &fc1, &activated, &routed_output,                                       \
        buffers->dummy_scale.ptr, buffers->dummy_scale.ptr,                    \
        buffers->global_scale.ptr, buffers->global_scale.ptr, &routes,         \
        &block_experts, &route_count, &activation_amax, 0,                     \
        buffers->topk_weights.ptr, &fc1_scratch, &fc2_scratch, &workspace,     \
        buffers->intermediate_rotations.ptr, buffers->gate_suh.ptr,            \
        buffers->up_suh.ptr, buffers->expert_map.ptr, buffers->trellis_lut.ptr,\
        buffers->trellis_lut.ptr, static_cast<int32_t>(kExperts),              \
        static_cast<int32_t>(kExperts), static_cast<int32_t>(active_m),        \
        DS41RT_DS4_FLASH_EXL3_K3_PREFILL_M##M##_TOPK6_GRID_X, stream);          \
  }
DS41RT_DEFINE_FLASH_EXL3_K3_PREFILL_LAUNCH(2)
DS41RT_DEFINE_FLASH_EXL3_K3_PREFILL_LAUNCH(4)
DS41RT_DEFINE_FLASH_EXL3_K3_PREFILL_LAUNCH(8)
DS41RT_DEFINE_FLASH_EXL3_K3_PREFILL_LAUNCH(16)
DS41RT_DEFINE_FLASH_EXL3_K3_PREFILL_LAUNCH(32)
DS41RT_DEFINE_FLASH_EXL3_K3_PREFILL_LAUNCH(64)
DS41RT_DEFINE_FLASH_EXL3_K3_PREFILL_LAUNCH(128)
DS41RT_DEFINE_FLASH_EXL3_K3_PREFILL_LAUNCH(256)
DS41RT_DEFINE_FLASH_EXL3_K3_PREFILL_LAUNCH(512)
DS41RT_DEFINE_FLASH_EXL3_K3_PREFILL_LAUNCH(1024)
DS41RT_DEFINE_FLASH_EXL3_K3_PREFILL_LAUNCH(2048)
#undef DS41RT_DEFINE_FLASH_EXL3_K3_PREFILL_LAUNCH

FlashExl3PrefillLaunchFn flash_exl3_k3_prefill_launcher(size_t capacity_rows) {
  switch (capacity_rows) {
  case 2: return &launch_flash_exl3_k3_prefill_m2;
  case 4: return &launch_flash_exl3_k3_prefill_m4;
  case 8: return &launch_flash_exl3_k3_prefill_m8;
  case 16: return &launch_flash_exl3_k3_prefill_m16;
  case 32: return &launch_flash_exl3_k3_prefill_m32;
  case 64: return &launch_flash_exl3_k3_prefill_m64;
  case 128: return &launch_flash_exl3_k3_prefill_m128;
  case 256: return &launch_flash_exl3_k3_prefill_m256;
  case 512: return &launch_flash_exl3_k3_prefill_m512;
  case 1024: return &launch_flash_exl3_k3_prefill_m1024;
  case 2048: return &launch_flash_exl3_k3_prefill_m2048;
  default: return nullptr;
  }
}

__global__ void map_flash_mixed_topk6_kernel(
    const int32_t *global_ids, const int32_t *global_to_combined,
    int32_t *combined_ids, size_t routes) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < routes) {
    const int32_t global_id = global_ids[index];
    combined_ids[index] =
        global_id >= 0 && global_id < static_cast<int32_t>(kExperts)
            ? global_to_combined[global_id]
            : -1;
  }
}

#define DS41RT_DEFINE_FLASH_EXL3_MIXED_LAUNCH(M)                              \
  int launch_flash_exl3_mixed_m##M(                                         \
      const ds41rt_ds4_flash_spark_exl3_mixed_moe_buffers_t *buffers,        \
      int32_t tier0_slots, int32_t tier1_slots,                              \
      int32_t tier0_gate_experts, int32_t tier1_gate_experts,                \
      int32_t tier0_up_experts, int32_t tier1_up_experts,                    \
      int32_t tier0_down_experts, int32_t tier1_down_experts,                \
      int32_t active_m,                                                       \
      cudaStream_t stream) {                                                 \
    ds41rt_ds4_flash_tp4_exl3_mixed_k2_k3_m##M##_Tensor_rotation_gate_t      \
        rotation_gate{buffers->rotation_gate.ptr};                           \
    ds41rt_ds4_flash_tp4_exl3_mixed_k2_k3_m##M##_Tensor_rotation_up_t        \
        rotation_up{buffers->rotation_up.ptr};                               \
    ds41rt_ds4_flash_tp4_exl3_mixed_k2_k3_m##M##_Tensor_fc1_t fc1{           \
        buffers->fc1_output.ptr};                                            \
    ds41rt_ds4_flash_tp4_exl3_mixed_k2_k3_m##M##_Tensor_activated_t          \
        activated{buffers->activated.ptr};                                   \
    ds41rt_ds4_flash_tp4_exl3_mixed_k2_k3_m##M##_Tensor_fc2_t fc2{           \
        buffers->routed_output.ptr};                                         \
    ds41rt_ds4_flash_tp4_exl3_mixed_k2_k3_m##M##_Tensor_packed_route_indices_t \
        routes{DS41RT_DS4_FLASH_EXL3_MIXED_K2_K3_M##M##_DIRECT_TOPK           \
                   ? buffers->mapped_topk_ids.ptr                             \
                   : buffers->packed_route_indices.ptr};                      \
    ds41rt_ds4_flash_tp4_exl3_mixed_k2_k3_m##M##_Tensor_block_expert_ids_t   \
        block_experts{buffers->block_expert_ids.ptr};                         \
    ds41rt_ds4_flash_tp4_exl3_mixed_k2_k3_m##M##_Tensor_packed_route_count_t \
        route_count{buffers->packed_route_count.ptr};                         \
    ds41rt_ds4_flash_tp4_exl3_mixed_k2_k3_m##M##_Tensor_fc1_scratch_t        \
        fc1_scratch{buffers->fc1_scratch.ptr};                                \
    ds41rt_ds4_flash_tp4_exl3_mixed_k2_k3_m##M##_Tensor_fc2_scratch_t        \
        fc2_scratch{buffers->fc2_scratch.ptr};                                \
    ds41rt_ds4_flash_tp4_exl3_mixed_k2_k3_m##M##_Tensor_workspace_t          \
        workspace{buffers->workspace.ptr};                                    \
    return cute_dsl_ds41rt_ds4_flash_tp4_exl3_mixed_k2_k3_m##M##_wrapper(    \
        &flash_exl3_mixed_m##M##_module, buffers->input.ptr, &rotation_gate, \
        &rotation_up, buffers->tier0_w13_trellis.ptr,                         \
        buffers->tier0_w2_trellis.ptr, buffers->dummy_scale.ptr,             \
        buffers->dummy_scale.ptr, buffers->tier0_global_scale.ptr,           \
        buffers->tier0_global_scale.ptr, buffers->tier1_w13_trellis.ptr,     \
        buffers->tier1_w2_trellis.ptr, buffers->dummy_scale.ptr,             \
        buffers->dummy_scale.ptr, buffers->tier1_global_scale.ptr,           \
        buffers->tier1_global_scale.ptr, &fc1, &activated, &fc2, &routes,    \
        &block_experts, &route_count, buffers->descriptor_map.ptr,            \
        buffers->topk_weights.ptr, &fc1_scratch, &fc2_scratch, &workspace,   \
        buffers->intermediate_rotations.ptr, buffers->gate_suh.ptr,           \
        buffers->up_suh.ptr, buffers->trellis_lut.ptr, tier0_slots,          \
        tier1_slots, tier0_down_experts, tier1_down_experts, active_m,       \
        DS41RT_DS4_FLASH_EXL3_MIXED_K2_K3_M##M##_GRID_X, stream,              \
        tier0_gate_experts, tier1_gate_experts, tier0_up_experts,            \
        tier1_up_experts);                                                    \
  }
DS41RT_DEFINE_FLASH_EXL3_MIXED_LAUNCH(1)
DS41RT_DEFINE_FLASH_EXL3_MIXED_LAUNCH(2)
DS41RT_DEFINE_FLASH_EXL3_MIXED_LAUNCH(3)
DS41RT_DEFINE_FLASH_EXL3_MIXED_LAUNCH(4)
DS41RT_DEFINE_FLASH_EXL3_MIXED_LAUNCH(5)
DS41RT_DEFINE_FLASH_EXL3_MIXED_LAUNCH(6)
DS41RT_DEFINE_FLASH_EXL3_MIXED_LAUNCH(7)
DS41RT_DEFINE_FLASH_EXL3_MIXED_LAUNCH(8)
DS41RT_DEFINE_FLASH_EXL3_MIXED_LAUNCH(9)
DS41RT_DEFINE_FLASH_EXL3_MIXED_LAUNCH(10)
DS41RT_DEFINE_FLASH_EXL3_MIXED_LAUNCH(11)
DS41RT_DEFINE_FLASH_EXL3_MIXED_LAUNCH(12)
DS41RT_DEFINE_FLASH_EXL3_MIXED_LAUNCH(16)
DS41RT_DEFINE_FLASH_EXL3_MIXED_LAUNCH(32)
DS41RT_DEFINE_FLASH_EXL3_MIXED_LAUNCH(64)
DS41RT_DEFINE_FLASH_EXL3_MIXED_LAUNCH(128)
DS41RT_DEFINE_FLASH_EXL3_MIXED_LAUNCH(256)
DS41RT_DEFINE_FLASH_EXL3_MIXED_LAUNCH(512)
DS41RT_DEFINE_FLASH_EXL3_MIXED_LAUNCH(1024)
DS41RT_DEFINE_FLASH_EXL3_MIXED_LAUNCH(2048)
#undef DS41RT_DEFINE_FLASH_EXL3_MIXED_LAUNCH

using FlashExl3MixedLaunchFn = int (*)(
    const ds41rt_ds4_flash_spark_exl3_mixed_moe_buffers_t *, int32_t, int32_t,
    int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t,
    cudaStream_t);

FlashExl3MixedLaunchFn flash_exl3_mixed_launcher(size_t capacity_rows) {
  switch (capacity_rows) {
  case 1: return &launch_flash_exl3_mixed_m1;
  case 2: return &launch_flash_exl3_mixed_m2;
  case 3: return &launch_flash_exl3_mixed_m3;
  case 4: return &launch_flash_exl3_mixed_m4;
  case 5: return &launch_flash_exl3_mixed_m5;
  case 6: return &launch_flash_exl3_mixed_m6;
  case 7: return &launch_flash_exl3_mixed_m7;
  case 8: return &launch_flash_exl3_mixed_m8;
  case 9: return &launch_flash_exl3_mixed_m9;
  case 10: return &launch_flash_exl3_mixed_m10;
  case 11: return &launch_flash_exl3_mixed_m11;
  case 12: return &launch_flash_exl3_mixed_m12;
  case 16: return &launch_flash_exl3_mixed_m16;
  case 32: return &launch_flash_exl3_mixed_m32;
  case 64: return &launch_flash_exl3_mixed_m64;
  case 128: return &launch_flash_exl3_mixed_m128;
  case 256: return &launch_flash_exl3_mixed_m256;
  case 512: return &launch_flash_exl3_mixed_m512;
  case 1024: return &launch_flash_exl3_mixed_m1024;
  case 2048: return &launch_flash_exl3_mixed_m2048;
  default: return nullptr;
  }
}

} // namespace

extern "C" ds41rt_status_t
ds41rt_cuda_ds4_flash_spark_aot_available(int *out_available) {
  if (out_available == nullptr) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  *out_available = 1;
  return DS41RT_STATUS_OK;
}

extern "C" ds41rt_status_t ds41rt_cuda_ds4_flash_spark_aot_init(void) {
  std::call_once(flash_module_init_once,
                 initialize_flash_module_on_current_device);
  return flash_module_init_status;
}

extern "C" ds41rt_status_t
ds41rt_cuda_ds4_flash_shared_expert_fp8_bf16_async(
    const ds41rt_ds4_flash_shared_expert_fp8_buffers_t *buffers, size_t rows,
    void *cuda_stream) {
  const ds41rt_status_t valid = validate_shared_buffers(buffers, rows);
  if (valid != DS41RT_STATUS_OK) {
    return valid;
  }
  const ds41rt_status_t initialized = ds41rt_cuda_ds4_flash_spark_aot_init();
  if (initialized != DS41RT_STATUS_OK) {
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
      gate_status = launch_shared_up_m1(
          chunk_input, buffers->w1_weight.ptr, buffers->w1_scale_mma.ptr,
          buffers->gate.ptr, buffers->alpha.ptr, 1, stream);
      up_status = launch_shared_up_m1(
          chunk_input, buffers->w3_weight.ptr, buffers->w3_scale_mma.ptr,
          buffers->up.ptr, buffers->alpha.ptr, 1, stream);
    } else {
      gate_status = launch_shared_up_m8(
          chunk_input, buffers->w1_weight.ptr, buffers->w1_scale_mma.ptr,
          buffers->gate.ptr, buffers->alpha.ptr,
          static_cast<int32_t>(chunk_rows), stream);
      up_status = launch_shared_up_m8(
          chunk_input, buffers->w3_weight.ptr, buffers->w3_scale_mma.ptr,
          buffers->up.ptr, buffers->alpha.ptr,
          static_cast<int32_t>(chunk_rows), stream);
    }
    if (gate_status != 0 || up_status != 0) {
      ds41rt_set_last_error_message(
          "DeepSeek-V4-Flash shared expert up-projection AOT launch failed");
      return DS41RT_STATUS_INTERNAL_ERROR;
    }
    constexpr size_t kThreads = 256;
    const size_t values = chunk_rows * kSharedIntermediate;
    const size_t blocks = (values + kThreads - 1) / kThreads;
    flash_shared_swiglu_kernel<<<static_cast<unsigned int>(blocks), kThreads,
                                 0, stream>>>(
        static_cast<const uint16_t *>(buffers->gate.ptr),
        static_cast<const uint16_t *>(buffers->up.ptr),
        static_cast<uint16_t *>(buffers->activated.ptr), values);
    if (rows == 1) {
      down_status = launch_shared_down_m1(
          buffers->activated.ptr, buffers->w2_weight.ptr,
          buffers->w2_scale_mma.ptr, chunk_output, buffers->alpha.ptr, 1,
          stream);
    } else {
      down_status = launch_shared_down_m8(
          buffers->activated.ptr, buffers->w2_weight.ptr,
          buffers->w2_scale_mma.ptr, chunk_output, buffers->alpha.ptr,
          static_cast<int32_t>(chunk_rows), stream);
    }
    if (down_status != 0) {
      ds41rt_set_last_error_message(
          "DeepSeek-V4-Flash shared expert down-projection AOT launch failed");
      return DS41RT_STATUS_INTERNAL_ERROR;
    }
  }
  return status_from_cuda(cudaGetLastError());
}

extern "C" ds41rt_status_t ds41rt_cuda_ds4_flash_spark_w4a16_decode_m1_bf16_async(
    const ds41rt_ds4_flash_spark_w4a16_moe_buffers_t *buffers,
    void *cuda_stream) {
  const ds41rt_status_t valid = validate_buffers(buffers);
  if (valid != DS41RT_STATUS_OK) {
    return valid;
  }
  const ds41rt_status_t initialized = ds41rt_cuda_ds4_flash_spark_aot_init();
  if (initialized != DS41RT_STATUS_OK) {
    return initialized;
  }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  cudaError_t error = cudaMemsetAsync(buffers->locks.ptr, 0,
                                      kLockElements * sizeof(int32_t), stream);
  if (error != cudaSuccess) {
    return status_from_cuda(error);
  }

  FlashFc1 fc1{buffers->fc1_output.ptr};
  FlashActivated activated{buffers->activated.ptr};
  FlashFc2 output{buffers->output.ptr};
  FlashRoutes routes{buffers->packed_route_indices.ptr};
  FlashBlockExperts block_experts{buffers->block_expert_ids.ptr};
  FlashRouteCount route_count{buffers->packed_route_count.ptr};
  FlashActivationAmax activation_amax{buffers->w13_global_scale.ptr};
  FlashFc1Scratch fc1_scratch{buffers->fc1_scratch.ptr};
  FlashFc2Scratch fc2_scratch{buffers->fc2_scratch.ptr};
  FlashLocks locks{buffers->locks.ptr};
  const int launch_status =
      cute_dsl_ds41rt_ds4_flash_tp4_w4a16_decode_m1_fused_sum_wrapper(
          &flash_decode_module, buffers->input.ptr, buffers->input.ptr,
          buffers->input.ptr, buffers->w13_weight.ptr, buffers->w2_weight.ptr,
          static_cast<int64_t>(buffers->w13_weight.bytes / sizeof(int32_t)),
          static_cast<int64_t>(buffers->w2_weight.bytes / sizeof(int32_t)),
          &fc1, &activated, &output, buffers->w13_scale.ptr,
          buffers->w2_scale.ptr, buffers->w13_global_scale.ptr,
          buffers->w2_global_scale.ptr, &routes, &block_experts, &route_count,
          &activation_amax, 0, buffers->topk_weights.ptr, &fc1_scratch,
          &fc2_scratch, &locks, buffers->w13_global_scale.ptr,
          buffers->w13_global_scale.ptr, buffers->w13_global_scale.ptr,
          buffers->packed_route_indices.ptr, buffers->w13_scale.ptr,
          buffers->w13_scale.ptr, static_cast<int32_t>(kExperts), 0,
          1, DS41RT_DS4_FLASH_W4A16_DECODE_M1_FUSED_SUM_GRID_X, stream);
  if (launch_status != 0) {
    ds41rt_set_last_error_message(
        "DeepSeek-V4-Flash Spark TP4 W4A16 decode M1 AOT launch failed");
    return DS41RT_STATUS_INTERNAL_ERROR;
  }
  return status_from_cuda(cudaGetLastError());
}

extern "C" ds41rt_status_t ds41rt_cuda_ds4_flash_pack_topk6_routes_async(
    const ds41rt_ds4_flash_route_pack_buffers_t *buffers, size_t rows,
    void *cuda_stream) {
  const ds41rt_status_t valid = validate_route_pack_buffers(buffers, rows);
  if (valid != DS41RT_STATUS_OK) {
    return valid;
  }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  cudaError_t error = cudaMemsetAsync(buffers->expert_counts.ptr, 0,
                                      kExperts * sizeof(int32_t), stream);
  if (error != cudaSuccess) {
    return status_from_cuda(error);
  }
  constexpr size_t threads = 256;
  const size_t live_routes = rows * kTopK;
  const size_t route_blocks = (live_routes + threads - 1) / threads;
  count_flash_topk6_routes_kernel<<<static_cast<unsigned int>(route_blocks),
                                    threads, 0, stream>>>(
      static_cast<const int32_t *>(buffers->topk_ids.ptr),
      static_cast<int32_t *>(buffers->expert_counts.ptr), live_routes);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    return status_from_cuda(error);
  }
  prefix_flash_topk6_routes_kernel<<<1, threads, 0, stream>>>(
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
  scatter_flash_topk6_routes_kernel<<<static_cast<unsigned int>(route_blocks),
                                      threads, 0, stream>>>(
      static_cast<const int32_t *>(buffers->topk_ids.ptr),
      static_cast<int32_t *>(buffers->expert_offsets.ptr),
      static_cast<int32_t *>(buffers->packed_route_indices.ptr), live_routes);
  return status_from_cuda(cudaGetLastError());
}

extern "C" ds41rt_status_t
ds41rt_cuda_ds4_flash_spark_w4a16_prefill_topk6_bf16_async(
    const ds41rt_ds4_flash_spark_w4a16_moe_buffers_t *buffers, size_t rows,
    void *cuda_stream) {
  size_t capacity_rows = 2;
  while (capacity_rows < rows && capacity_rows < kPrefillMaxRows) {
    capacity_rows *= 2;
  }
  if (rows < 2 || rows > capacity_rows) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  const FlashPrefillLaunchFn launcher = flash_prefill_launcher(capacity_rows);
  if (launcher == nullptr) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  const ds41rt_status_t valid = validate_prefill_buffers(buffers, capacity_rows);
  if (valid != DS41RT_STATUS_OK) {
    return valid;
  }
  const ds41rt_status_t initialized = ds41rt_cuda_ds4_flash_spark_aot_init();
  if (initialized != DS41RT_STATUS_OK) {
    return initialized;
  }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  cudaError_t error = cudaMemsetAsync(buffers->locks.ptr, 0,
                                      kLockElements * sizeof(int32_t), stream);
  if (error != cudaSuccess) {
    return status_from_cuda(error);
  }
  const int launch_status = launcher(buffers, rows, stream);
  if (launch_status != 0) {
    ds41rt_set_last_error_message(
        "DeepSeek-V4-Flash Spark TP4 W4A16 prefill top-k=6 AOT launch failed");
    return DS41RT_STATUS_INTERNAL_ERROR;
  }
  constexpr size_t threads = 256;
  const size_t values = rows * kHidden;
  const size_t blocks = (values + threads - 1) / threads;
  sum_flash_topk6_bf16_kernel<<<static_cast<unsigned int>(blocks), threads, 0,
                                stream>>>(
      static_cast<const uint16_t *>(buffers->routed_output.ptr),
      static_cast<uint16_t *>(buffers->output.ptr), rows);
  return status_from_cuda(cudaGetLastError());
}

extern "C" ds41rt_status_t
ds41rt_cuda_ds4_flash_spark_exl3_k2_decode_m1_async(
    const ds41rt_ds4_flash_spark_exl3_k2_moe_buffers_t *buffers,
    void *cuda_stream) {
  const ds41rt_status_t valid =
      validate_exl3_buffers(buffers, 1, kExl3K2Bits);
  if (valid != DS41RT_STATUS_OK) {
    return valid;
  }
  const ds41rt_status_t initialized = ds41rt_cuda_ds4_flash_spark_aot_init();
  if (initialized != DS41RT_STATUS_OK) {
    return initialized;
  }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  cudaError_t error =
      cudaMemsetAsync(buffers->workspace.ptr, 0,
                      kExl3WorkspaceElements * sizeof(int32_t), stream);
  if (error != cudaSuccess) {
    return status_from_cuda(error);
  }
  if (launch_flash_exl3_decode(buffers, stream) != 0) {
    ds41rt_set_last_error_message(
        "DeepSeek-V4-Flash Spark TP4 EXL3 K2 decode M1 AOT launch failed");
    return DS41RT_STATUS_INTERNAL_ERROR;
  }
  if (launch_flash_exl3_topk6_sum(buffers, 1, stream) != 0) {
    ds41rt_set_last_error_message(
        "DeepSeek-V4-Flash Spark TP4 EXL3 K2 decode sum launch failed");
    return DS41RT_STATUS_INTERNAL_ERROR;
  }
  return ds41rt_cuda_f32_to_bf16_async(
      static_cast<const float *>(buffers->output_f32.ptr),
      static_cast<uint16_t *>(buffers->output_bf16.ptr), kHidden,
      cuda_stream);
}

extern "C" ds41rt_status_t
ds41rt_cuda_ds4_flash_spark_exl3_k2_prefill_topk6_async(
    const ds41rt_ds4_flash_spark_exl3_k2_moe_buffers_t *buffers, size_t rows,
    void *cuda_stream) {
  size_t capacity_rows = 2;
  while (capacity_rows < rows && capacity_rows < kPrefillMaxRows) {
    capacity_rows *= 2;
  }
  if (rows < 2 || rows > capacity_rows) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  const bool direct_m4 =
      rows >= DS41RT_DS4_FLASH_EXL3_K2_DIRECT_M4_MIN_ACTIVE && rows <= 4;
  const bool direct_m6 = rows == 6;
  const bool direct = direct_m4 || direct_m6;
  const FlashExl3PrefillLaunchFn launcher = direct
      ? nullptr
      : flash_exl3_prefill_launcher(capacity_rows);
  if (!direct && launcher == nullptr) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  const ds41rt_status_t valid =
      validate_exl3_buffers(buffers, capacity_rows, kExl3K2Bits);
  if (valid != DS41RT_STATUS_OK) {
    return valid;
  }
  const ds41rt_status_t initialized = ds41rt_cuda_ds4_flash_spark_aot_init();
  if (initialized != DS41RT_STATUS_OK) {
    return initialized;
  }

  if (!direct) {
    const ds41rt_ds4_flash_route_pack_buffers_t route_buffers{
        buffers->topk_ids,
        buffers->packed_route_indices,
        buffers->block_expert_ids,
        buffers->packed_route_count,
        buffers->expert_counts,
        buffers->expert_offsets,
    };
    ds41rt_status_t status = ds41rt_cuda_ds4_flash_pack_topk6_routes_async(
        &route_buffers, rows, cuda_stream);
    if (status != DS41RT_STATUS_OK) {
      return status;
    }
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  cudaError_t error =
      cudaMemsetAsync(buffers->workspace.ptr, 0,
                      kExl3WorkspaceElements * sizeof(int32_t), stream);
  if (error != cudaSuccess) {
    return status_from_cuda(error);
  }
  int launch_status = 0;
  if (direct_m4) {
    launch_status = launch_flash_exl3_decode_m4_direct_m4(buffers, rows, stream);
  } else if (direct_m6) {
    launch_status = launch_flash_exl3_decode_m8_direct_m6(buffers, rows, stream);
  } else {
    launch_status = launcher(buffers, rows, stream);
  }
  if (launch_status != 0) {
    ds41rt_set_last_error_message(
        "DeepSeek-V4-Flash Spark TP4 EXL3 K2 prefill AOT launch failed");
    return DS41RT_STATUS_INTERNAL_ERROR;
  }
  if (launch_flash_exl3_topk6_sum(buffers, rows, stream) != 0) {
    ds41rt_set_last_error_message(
        "DeepSeek-V4-Flash Spark TP4 EXL3 K2 prefill sum launch failed");
    return DS41RT_STATUS_INTERNAL_ERROR;
  }
  return ds41rt_cuda_f32_to_bf16_async(
      static_cast<const float *>(buffers->output_f32.ptr),
      static_cast<uint16_t *>(buffers->output_bf16.ptr), rows * kHidden,
      cuda_stream);
}

extern "C" ds41rt_status_t
ds41rt_cuda_ds4_flash_spark_exl3_k3_decode_m1_async(
    const ds41rt_ds4_flash_spark_exl3_k2_moe_buffers_t *buffers,
    void *cuda_stream) {
  const ds41rt_status_t valid =
      validate_exl3_buffers(buffers, 1, kExl3K3Bits);
  if (valid != DS41RT_STATUS_OK) {
    return valid;
  }
  const ds41rt_status_t initialized = ds41rt_cuda_ds4_flash_spark_aot_init();
  if (initialized != DS41RT_STATUS_OK) {
    return initialized;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  cudaError_t error =
      cudaMemsetAsync(buffers->workspace.ptr, 0,
                      kExl3WorkspaceElements * sizeof(int32_t), stream);
  if (error != cudaSuccess) {
    return status_from_cuda(error);
  }
  if (launch_flash_exl3_k3_decode(buffers, stream) != 0) {
    ds41rt_set_last_error_message(
        "DeepSeek-V4-Flash Spark TP4 EXL3 K3 decode M1 AOT launch failed");
    return DS41RT_STATUS_INTERNAL_ERROR;
  }
  if (launch_flash_exl3_topk6_sum(buffers, 1, stream) != 0) {
    ds41rt_set_last_error_message(
        "DeepSeek-V4-Flash Spark TP4 EXL3 K3 decode sum launch failed");
    return DS41RT_STATUS_INTERNAL_ERROR;
  }
  return ds41rt_cuda_f32_to_bf16_async(
      static_cast<const float *>(buffers->output_f32.ptr),
      static_cast<uint16_t *>(buffers->output_bf16.ptr), kHidden,
      cuda_stream);
}

extern "C" ds41rt_status_t
ds41rt_cuda_ds4_flash_spark_exl3_k3_prefill_topk6_async(
    const ds41rt_ds4_flash_spark_exl3_k2_moe_buffers_t *buffers, size_t rows,
    void *cuda_stream) {
  size_t capacity_rows = 2;
  while (capacity_rows < rows && capacity_rows < kPrefillMaxRows) {
    capacity_rows *= 2;
  }
  if (rows < 2 || rows > capacity_rows) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  const bool direct_m4 =
      rows >= DS41RT_DS4_FLASH_EXL3_K2_DIRECT_M4_MIN_ACTIVE && rows <= 4;
  const bool direct_m6 = rows == 6;
  const bool direct = direct_m4 || direct_m6;
  const FlashExl3PrefillLaunchFn launcher =
      direct ? nullptr : flash_exl3_k3_prefill_launcher(capacity_rows);
  if (!direct && launcher == nullptr) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  const ds41rt_status_t valid =
      validate_exl3_buffers(buffers, capacity_rows, kExl3K3Bits);
  if (valid != DS41RT_STATUS_OK) {
    return valid;
  }
  const ds41rt_status_t initialized = ds41rt_cuda_ds4_flash_spark_aot_init();
  if (initialized != DS41RT_STATUS_OK) {
    return initialized;
  }
  if (!direct) {
    const ds41rt_ds4_flash_route_pack_buffers_t route_buffers{
        buffers->topk_ids,
        buffers->packed_route_indices,
        buffers->block_expert_ids,
        buffers->packed_route_count,
        buffers->expert_counts,
        buffers->expert_offsets,
    };
    ds41rt_status_t status = ds41rt_cuda_ds4_flash_pack_topk6_routes_async(
        &route_buffers, rows, cuda_stream);
    if (status != DS41RT_STATUS_OK) {
      return status;
    }
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  cudaError_t error =
      cudaMemsetAsync(buffers->workspace.ptr, 0,
                      kExl3WorkspaceElements * sizeof(int32_t), stream);
  if (error != cudaSuccess) {
    return status_from_cuda(error);
  }
  int launch_status = 0;
  if (direct_m4) {
    launch_status =
        launch_flash_exl3_k3_decode_m4_direct_m4(buffers, rows, stream);
  } else if (direct_m6) {
    launch_status =
        launch_flash_exl3_k3_decode_m8_direct_m6(buffers, rows, stream);
  } else {
    launch_status = launcher(buffers, rows, stream);
  }
  if (launch_status != 0) {
    ds41rt_set_last_error_message(
        "DeepSeek-V4-Flash Spark TP4 EXL3 K3 prefill AOT launch failed");
    return DS41RT_STATUS_INTERNAL_ERROR;
  }
  if (launch_flash_exl3_topk6_sum(buffers, rows, stream) != 0) {
    ds41rt_set_last_error_message(
        "DeepSeek-V4-Flash Spark TP4 EXL3 K3 prefill sum launch failed");
    return DS41RT_STATUS_INTERNAL_ERROR;
  }
  return ds41rt_cuda_f32_to_bf16_async(
      static_cast<const float *>(buffers->output_f32.ptr),
      static_cast<uint16_t *>(buffers->output_bf16.ptr), rows * kHidden,
      cuda_stream);
}

extern "C" ds41rt_status_t
ds41rt_cuda_ds4_flash_spark_exl3_mixed_k2_k3_async(
    const ds41rt_ds4_flash_spark_exl3_mixed_moe_buffers_t *buffers,
    size_t tier0_slots, size_t tier1_slots,
    size_t tier0_gate_experts, size_t tier1_gate_experts,
    size_t tier0_up_experts, size_t tier1_up_experts,
    size_t tier0_down_experts, size_t tier1_down_experts, size_t rows,
    void *cuda_stream) {
  const size_t total_slots = tier0_slots + tier1_slots;
  if (buffers == nullptr || tier0_slots == 0 || tier1_slots == 0 ||
      tier0_slots > kExperts || tier1_slots > kExperts ||
      total_slots < kExperts ||
      tier0_gate_experts == 0 || tier1_gate_experts == 0 ||
      tier0_up_experts == 0 || tier1_up_experts == 0 ||
      tier0_down_experts == 0 || tier1_down_experts == 0 ||
      tier0_gate_experts + tier1_gate_experts != kExperts ||
      tier0_up_experts + tier1_up_experts != kExperts ||
      tier0_down_experts + tier1_down_experts != kExperts ||
      tier0_slots != std::max(tier0_gate_experts, tier0_up_experts) ||
      tier1_slots != std::max(tier1_gate_experts, tier1_up_experts) ||
      rows == 0 || rows > kPrefillMaxRows) {
    ds41rt_set_last_error_message(
        "DeepSeek-V4-Flash mixed K2/K3 launch received invalid projection tiers or rows");
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  size_t capacity_rows = rows <= 12 ? rows : 16;
  while (capacity_rows < rows) capacity_rows *= 2;
  const bool decode = capacity_rows == 1;
  const size_t packed_route_elements =
      decode ? kTopK : kPrefillMaxPackedRouteSlots;
  const size_t route_block_elements =
      decode ? kTopK : kPrefillMaxRouteBlocks;
  const size_t fc1_scratch_elements =
      decode ? kFc1ScratchElements : kPrefillScratchElements;
  const size_t fc2_scratch_elements =
      decode ? kFc2ScratchElements : kPrefillScratchElements;
  const size_t tier0_projection =
      kHidden * kIntermediate * kExl3K2Bits / 8;
  const size_t tier1_projection =
      kHidden * kIntermediate * kExl3K3Bits / 8;
  const size_t routed_rows = capacity_rows * kTopK;
#define DS41RT_REQUIRE_MIXED_BYTES(FIELD, REQUIRED)                            \
  do {                                                                       \
    if (!has_bytes(buffers->FIELD, (REQUIRED))) {                            \
      ds41rt_set_last_error_message(                                           \
          "DeepSeek-V4-Flash mixed K2/K3 undersized buffer: " #FIELD);       \
      return DS41RT_STATUS_BUFFER_TOO_SMALL;                                  \
    }                                                                        \
  } while (false)
  DS41RT_REQUIRE_MIXED_BYTES(input,
                            capacity_rows * kHidden * sizeof(uint16_t));
  DS41RT_REQUIRE_MIXED_BYTES(
      tier0_w13_trellis,
      (tier0_gate_experts + tier0_up_experts) * tier0_projection);
  DS41RT_REQUIRE_MIXED_BYTES(tier0_w2_trellis,
                            tier0_down_experts * tier0_projection);
  DS41RT_REQUIRE_MIXED_BYTES(
      tier1_w13_trellis,
      (tier1_gate_experts + tier1_up_experts) * tier1_projection);
  DS41RT_REQUIRE_MIXED_BYTES(tier1_w2_trellis,
                            tier1_down_experts * tier1_projection);
  DS41RT_REQUIRE_MIXED_BYTES(dummy_scale, 16);
  DS41RT_REQUIRE_MIXED_BYTES(tier0_global_scale,
                            std::max(tier0_slots, tier0_down_experts) *
                                sizeof(float));
  DS41RT_REQUIRE_MIXED_BYTES(tier1_global_scale,
                            std::max(tier1_slots, tier1_down_experts) *
                                sizeof(float));
  DS41RT_REQUIRE_MIXED_BYTES(gate_suh,
                            total_slots * kHidden * sizeof(uint16_t));
  DS41RT_REQUIRE_MIXED_BYTES(up_suh,
                            total_slots * kHidden * sizeof(uint16_t));
  DS41RT_REQUIRE_MIXED_BYTES(intermediate_rotations,
                            total_slots * 3 * kIntermediate *
                                sizeof(uint16_t));
  DS41RT_REQUIRE_MIXED_BYTES(down_svh,
                            total_slots * kHidden * sizeof(uint16_t));
  DS41RT_REQUIRE_MIXED_BYTES(trellis_lut, kExl3TrellisLutBytes);
  DS41RT_REQUIRE_MIXED_BYTES(global_to_combined,
                            kExperts * sizeof(int32_t));
  DS41RT_REQUIRE_MIXED_BYTES(descriptor_map,
                            3 * total_slots * sizeof(int32_t));
  DS41RT_REQUIRE_MIXED_BYTES(topk_ids, routed_rows * sizeof(int32_t));
  DS41RT_REQUIRE_MIXED_BYTES(mapped_topk_ids,
                            routed_rows * sizeof(int32_t));
  DS41RT_REQUIRE_MIXED_BYTES(topk_weights, routed_rows * sizeof(float));
  DS41RT_REQUIRE_MIXED_BYTES(rotation_gate,
                            routed_rows * kHidden * sizeof(uint16_t));
  DS41RT_REQUIRE_MIXED_BYTES(rotation_up,
                            routed_rows * kHidden * sizeof(uint16_t));
  DS41RT_REQUIRE_MIXED_BYTES(
      fc1_output, routed_rows * 2 * kIntermediate * sizeof(uint16_t));
  DS41RT_REQUIRE_MIXED_BYTES(
      activated, routed_rows * kIntermediate * sizeof(uint16_t));
  DS41RT_REQUIRE_MIXED_BYTES(routed_output,
                            routed_rows * kHidden * sizeof(uint16_t));
  DS41RT_REQUIRE_MIXED_BYTES(output_f32,
                            capacity_rows * kHidden * sizeof(float));
  DS41RT_REQUIRE_MIXED_BYTES(output_bf16,
                            capacity_rows * kHidden * sizeof(uint16_t));
  DS41RT_REQUIRE_MIXED_BYTES(packed_route_indices,
                            packed_route_elements * sizeof(int32_t));
  DS41RT_REQUIRE_MIXED_BYTES(block_expert_ids,
                            route_block_elements * sizeof(int32_t));
  DS41RT_REQUIRE_MIXED_BYTES(packed_route_count, sizeof(int32_t));
  DS41RT_REQUIRE_MIXED_BYTES(expert_counts, kExperts * sizeof(int32_t));
  DS41RT_REQUIRE_MIXED_BYTES(expert_offsets,
                            (kExperts + 1) * sizeof(int32_t));
  DS41RT_REQUIRE_MIXED_BYTES(fc1_scratch,
                            fc1_scratch_elements * sizeof(float));
  DS41RT_REQUIRE_MIXED_BYTES(fc2_scratch,
                            fc2_scratch_elements * sizeof(float));
  DS41RT_REQUIRE_MIXED_BYTES(workspace,
                            kExl3WorkspaceElements * sizeof(int32_t));
#undef DS41RT_REQUIRE_MIXED_BYTES
  const ds41rt_status_t initialized = ds41rt_cuda_ds4_flash_spark_aot_init();
  if (initialized != DS41RT_STATUS_OK) return initialized;
  const FlashExl3MixedLaunchFn launcher =
      flash_exl3_mixed_launcher(capacity_rows);
  if (launcher == nullptr) {
    ds41rt_set_last_error_message(
        "DeepSeek-V4-Flash mixed K2/K3 launch has no row-capacity kernel");
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const size_t live_routes = rows * kTopK;
  constexpr size_t threads = 256;
  const size_t blocks = (live_routes + threads - 1) / threads;
  map_flash_mixed_topk6_kernel<<<static_cast<unsigned int>(blocks), threads, 0,
                                 stream>>>(
      static_cast<const int32_t *>(buffers->topk_ids.ptr),
      static_cast<const int32_t *>(buffers->global_to_combined.ptr),
      static_cast<int32_t *>(buffers->mapped_topk_ids.ptr), live_routes);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) return status_from_cuda(error);
  const bool direct_topk = capacity_rows <= 12;
  if (!direct_topk) {
    const ds41rt_ds4_flash_route_pack_buffers_t route_buffers{
        buffers->mapped_topk_ids,
        buffers->packed_route_indices,
        buffers->block_expert_ids,
        buffers->packed_route_count,
        buffers->expert_counts,
        buffers->expert_offsets,
    };
    ds41rt_status_t status = ds41rt_cuda_ds4_flash_pack_topk6_routes_async(
        &route_buffers, rows, cuda_stream);
    if (status != DS41RT_STATUS_OK) return status;
  }
  error = cudaMemsetAsync(buffers->workspace.ptr, 0,
                          kExl3WorkspaceElements * sizeof(int32_t), stream);
  if (error != cudaSuccess) return status_from_cuda(error);
  if (launcher(buffers, static_cast<int32_t>(tier0_slots),
               static_cast<int32_t>(tier1_slots),
               static_cast<int32_t>(tier0_gate_experts),
               static_cast<int32_t>(tier1_gate_experts),
               static_cast<int32_t>(tier0_up_experts),
               static_cast<int32_t>(tier1_up_experts),
               static_cast<int32_t>(tier0_down_experts),
               static_cast<int32_t>(tier1_down_experts),
               static_cast<int32_t>(rows), stream) != 0) {
    ds41rt_set_last_error_message(
        "DeepSeek-V4-Flash Spark TP4 mixed K2/K3 EXL3 launch failed");
    return DS41RT_STATUS_INTERNAL_ERROR;
  }
  ds41rt_ds4_flash_spark_exl3_k2_moe_buffers_t sum_buffers{};
  sum_buffers.routed_output = buffers->routed_output;
  sum_buffers.output_f32 = buffers->output_f32;
  sum_buffers.topk_weights = buffers->topk_weights;
  sum_buffers.topk_ids = buffers->topk_ids;
  sum_buffers.expert_map = buffers->global_to_combined;
  sum_buffers.down_svh = buffers->down_svh;
  if (launch_flash_exl3_topk6_sum(&sum_buffers, rows, stream) != 0) {
    ds41rt_set_last_error_message(
        "DeepSeek-V4-Flash Spark TP4 mixed K2/K3 EXL3 sum failed");
    return DS41RT_STATUS_INTERNAL_ERROR;
  }
  return ds41rt_cuda_f32_to_bf16_async(
      static_cast<const float *>(buffers->output_f32.ptr),
      static_cast<uint16_t *>(buffers->output_bf16.ptr), rows * kHidden,
      cuda_stream);
}
