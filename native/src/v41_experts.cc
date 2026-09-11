#include "ds41rt_v41_experts.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstddef>
#include <mutex>
#include "v41_expert_variants.h"
#include "v41_input_quant_dispatch.h"

static_assert(sizeof(ds41rt_v41_expert_info_t) == 64);
static_assert(sizeof(ds41rt_v41_expert_launch_t) == 392);
static_assert(offsetof(ds41rt_v41_expert_launch_t, stream) == 384);

namespace {
using ModuleFn = void (*)(void**);
using LaunchFn = void (*)(void**, int32_t);
struct Variant {
  ds41rt_v41_expert_info_t info;
  ModuleFn initialize;
  ModuleFn load;
  LaunchFn launch;
  uint64_t scratch_offsets[DS41RT_V41_EXPERT_POINTERS];
  cudaLibrary_t library = nullptr;
  int device = -1;
  ~Variant() { if (library) cudaLibraryUnload(library); }
};
Variant variants[] = {DS41RT_V41_VARIANTS};
std::mutex initialization_mutex;
Variant* by_handle(void* handle) {
  for (auto& variant : variants) if (&variant == handle) return &variant;
  return nullptr;
}
bool valid_scratch(Variant* variant, void* storage, uint64_t bytes) {
  return variant && variant->device >= 0 && storage &&
    reinterpret_cast<uintptr_t>(storage) % 16 == 0 && bytes >= variant->info.scratch_bytes &&
    reinterpret_cast<uintptr_t>(storage) <= UINTPTR_MAX - variant->info.scratch_bytes;
}
Variant* by_capacity(int32_t capacity) {
  for (auto& variant : variants)
    if (variant.info.capacity_rows == static_cast<uint32_t>(capacity)) return &variant;
  return nullptr;
}
}

extern "C" int32_t ds41rt_v41_initialize_scratch_storage_async(
    void*, uint64_t, uint64_t, uint64_t, uint32_t, void*);

extern "C" int32_t ds41rt_v41_expert_bind_scratch(void* kernel, void* storage,
    uint64_t bytes, void* tensors[DS41RT_V41_EXPERT_POINTERS]) {
  auto* variant = by_handle(kernel);
  if (!valid_scratch(variant, storage, bytes) || !tensors) return cudaErrorInvalidValue;
  for (int slot = 0; slot < DS41RT_V41_EXPERT_POINTERS; ++slot)
    if (variant->scratch_offsets[slot] != UINT64_MAX)
      tensors[slot] = static_cast<char*>(storage) + variant->scratch_offsets[slot];
  return cudaSuccess;
}

extern "C" int32_t ds41rt_v41_expert_initialize_scratch_async(void* kernel,
    void* storage, uint64_t bytes, void* stream) {
  auto* variant = by_handle(kernel);
  if (!valid_scratch(variant, storage, bytes)) return cudaErrorInvalidValue;
  int device = -1;
  auto status = cudaGetDevice(&device);
  if (status != cudaSuccess) return status;
  if (device != variant->device) return cudaErrorInvalidDevice;
  return ds41rt_v41_initialize_scratch_storage_async(storage, variant->info.scratch_bytes,
      variant->scratch_offsets[37], variant->scratch_offsets[40],
      variant->info.experts, stream);
}

extern "C" int32_t ds41rt_v41_expert_info(int32_t capacity, ds41rt_v41_expert_info_t* out) {
  auto* variant = by_capacity(capacity);
  if (!variant || !out) return cudaErrorInvalidValue;
  *out = variant->info;
  return cudaSuccess;
}

extern "C" int32_t ds41rt_v41_expert_initialize(int32_t capacity, void** out) {
  if (!out) return cudaErrorInvalidValue;
  *out = nullptr;
  auto* variant = by_capacity(capacity);
  if (!variant) return cudaErrorInvalidValue;
  int device = -1, major = 0, minor = 0, sms = 0;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) return status;
  status = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  if (status != cudaSuccess) return status;
  status = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
  if (status != cudaSuccess) return status;
  status = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  if (status != cudaSuccess) return status;
  if (major != 12 || minor != DS41RT_V41_CC_MINOR || sms != DS41RT_V41_SMS)
    return cudaErrorInvalidDevice;
  std::lock_guard<std::mutex> lock(initialization_mutex);
  if (variant->device >= 0) {
    if (variant->device != device) return cudaErrorInvalidDevice;
    *out = variant;
    return cudaSuccess;
  }
  auto* library_ptr = &variant->library;
  void* init_args[] = {&library_ptr, &status};
  variant->initialize(init_args);
  if (status != cudaSuccess) {
    if (variant->library) cudaLibraryUnload(variant->library);
    variant->library = nullptr;
    return status;
  }
  void* load_args[] = {&library_ptr, &device, &status};
  variant->load(load_args);
  if (status != cudaSuccess) {
    cudaLibraryUnload(variant->library);
    variant->library = nullptr;
    return status;
  }
  variant->device = device;
  *out = variant;
  return cudaSuccess;
}

extern "C" int32_t ds41rt_v41_expert_launch(void* kernel, const ds41rt_v41_expert_launch_t* args) {
  Variant* variant = nullptr;
  for (auto& candidate : variants) if (&candidate == kernel) variant = &candidate;
  if (!variant || !args || variant->device < 0) return cudaErrorInvalidValue;
  const auto& info = variant->info;
  if (args->num_tokens <= 0 || static_cast<uint32_t>(args->num_tokens) > info.capacity_rows ||
      args->scatter_rows != args->num_tokens * static_cast<int32_t>(info.topk) ||
      args->max_rows != info.max_rows || args->rows_padded != info.rows_padded ||
      args->max_tasks != info.max_tasks || args->max_phys_tiles != info.max_phys_tiles ||
      args->max_active_clusters <= 0 || args->max_active_clusters > 2 * DS41RT_V41_SMS)
    return cudaErrorInvalidValue;
  for (auto* pointer : args->tensors) if (!pointer) return cudaErrorInvalidValue;
  int device = -1;
  auto status = cudaGetDevice(&device);
  if (status != cudaSuccess) return status;
  if (device != variant->device) return cudaErrorInvalidDevice;
  void* pointers[44];
  std::copy(args->tensors, args->tensors + 44, pointers);
  int32_t scalars[] = {args->num_tokens, args->max_rows, args->scatter_rows,
    args->rows_padded, args->max_tasks, args->max_phys_tiles, args->max_active_clusters};
  void* stream = args->stream;
  int32_t result = 0;
  void* parameters[53];
  for (int i = 0; i < 44; ++i) parameters[i] = &pointers[i];
  for (int i = 0; i < 7; ++i) parameters[44 + i] = &scalars[i];
  parameters[51] = &stream;
  parameters[52] = &result;
  variant->launch(parameters, 53);
  return result;
}

namespace {
struct InputQuantModule {
  cudaLibrary_t library = nullptr;
  int device = -1;
  ~InputQuantModule() { if (library) cudaLibraryUnload(library); }
} input_quant;
}
extern "C" int32_t ds41rt_v41_expert_input_quant_initialize(void** out) {
  if (!out) return cudaErrorInvalidValue;
  *out = nullptr;
  int device, major, minor, sms;
  auto status = cudaGetDevice(&device); if (status) return status;
  status = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device); if (status) return status;
  status = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device); if (status) return status;
  status = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device); if (status) return status;
  if (major != 12 || minor != DS41RT_V41_CC_MINOR || sms != DS41RT_V41_SMS)
    return cudaErrorInvalidDevice;
  std::lock_guard<std::mutex> lock(initialization_mutex);
  if (input_quant.device >= 0) {
    if (input_quant.device != device) return cudaErrorInvalidDevice;
    *out = &input_quant;
    return cudaSuccess;
  }
  auto* library = &input_quant.library;
  void* init[] = {&library, &status};
  _mlir_ds41rt_v41_expert_input_quant_cuda_init(init);
  if (!status) {
    void* load[] = {&library, &device, &status};
    _mlir_ds41rt_v41_expert_input_quant_cuda_load_to_device(load);
  }
  if (status) {
    if (input_quant.library) cudaLibraryUnload(input_quant.library);
    input_quant.library = nullptr;
    return status;
  }
  input_quant.device = device;
  *out = &input_quant;
  return cudaSuccess;
}
extern "C" int32_t ds41rt_v41_expert_input_quantize_async(void* kernel,
    const uint16_t* input, uint8_t* output, uint32_t rows, void* stream) {
  if (kernel != &input_quant || input_quant.device < 0 || !input || !output ||
      rows == 0 || rows > 4096) return cudaErrorInvalidValue;
  const auto a = reinterpret_cast<uintptr_t>(input), b = reinterpret_cast<uintptr_t>(output);
  const uint64_t an = uint64_t(rows) * 10240, bn = uint64_t(rows) * 5280;
  if (a % 16 || b % 16 || a > UINTPTR_MAX-an || b > UINTPTR_MAX-bn ||
      (a <= b ? b-a < an : a-b < bn)) return cudaErrorInvalidValue;
  int device;
  auto status = cudaGetDevice(&device); if (status) return status;
  if (device != input_quant.device) return cudaErrorInvalidDevice;
  void* source = const_cast<uint16_t*>(input);
  void* values = output;
  void* scales = output + 5120;
  void* unused_mma = output; // wire specialization does not write MMA scales
  int32_t m = rows, grid = ds41rt_v41_input_quant_grids[rows-1];
  int32_t result = 0;
  void* args[] = {&source, &values, &scales, &unused_mma, &m, &grid, &stream, &result};
  DS41RT_V41_INPUT_QUANT_ENTRY(args, 8);
  return result;
}
