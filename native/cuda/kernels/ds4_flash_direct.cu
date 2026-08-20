#include "common.h"

#include <limits>

namespace {

bool buffer_has_bytes(ds4rt_device_buffer_t buffer, size_t required) {
  return buffer.ptr != nullptr && buffer.bytes >= required;
}

__global__ void pack_ds4_flash_w4a16_weight_kernel(
    const uint8_t* source, uint32_t* destination, size_t size_k,
    size_t source_size_k, size_t source_start_k, size_t size_n,
    size_t row_rotation) {
  const size_t output_index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t k_tiles = size_k / 16;
  const size_t n_tiles = size_n / 64;
  const size_t output_words = k_tiles * n_tiles * 128;
  if (output_index >= output_words) {
    return;
  }
  const size_t packed_position = output_index % 128;
  const size_t tile_index = output_index / 128;
  const size_t n_tile = tile_index % n_tiles;
  const size_t k_tile = tile_index / n_tiles;
  const size_t thread_group = packed_position / 4;
  const size_t warp_column = packed_position % 4;
  const size_t tensor_column = thread_group / 4;
  const size_t tensor_row = (thread_group % 4) * 2;
  constexpr int kElementOffsets[4] = {0, 1, 8, 9};
  constexpr int kPackOrder[8] = {0, 2, 4, 6, 1, 3, 5, 7};
  uint32_t result = 0;
  for (int slot = 0; slot < 8; ++slot) {
    const int source_slot = kPackOrder[slot];
    const int element_slot = source_slot & 3;
    const size_t element = tensor_row + kElementOffsets[element_slot];
    const size_t k_half = element / 8;
    const size_t nibble = element % 8;
    const size_t column_base = warp_column * 16 + tensor_column;
    const size_t packed_row =
        n_tile * 64 + column_base + (source_slot >= 4 ? 8 : 0);
    const size_t source_row = (packed_row + row_rotation) % size_n;
    const size_t source_word = source_start_k / 8 + k_tile * 2 + k_half;
    const uint32_t word = reinterpret_cast<const uint32_t*>(source)[
        source_row * (source_size_k / 8) + source_word];
    result |= ((word >> (nibble * 4)) & 0x0fU) << (slot * 4);
  }
  destination[output_index] = result;
}

__global__ void pack_ds4_flash_e8m0_k32_scale_kernel(
    const uint8_t* source, uint8_t* destination, size_t size_k,
    size_t source_size_k, size_t source_start_k, size_t size_n,
    size_t row_rotation) {
  const size_t output_index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t values = (size_k / 32) * size_n;
  if (output_index >= values) {
    return;
  }

  const size_t k_block = output_index / size_n;
  const size_t output_row = output_index % size_n;
  // SparkInfer first transposes source [N,K/32] to [K/32,N], applies its
  // 8x8 scale permutation within every 64 rows, and finally swaps lanes 1/2
  // in every four-byte group.  Compute the inverse source index directly so
  // this transformation remains a single streaming GPU pass.
  constexpr int kSwapFour[4] = {0, 2, 1, 3};
  const size_t swapped =
      (output_row & ~size_t{3}) + kSwapFour[output_row & size_t{3}];
  const size_t group_base = (swapped / 64) * 64;
  const size_t group_offset = swapped % 64;
  const size_t permuted_row =
      group_base + group_offset / 8 + 8 * (group_offset % 8);
  const size_t source_row = (permuted_row + row_rotation) % size_n;
  const size_t source_index =
      source_row * (source_size_k / 32) + source_start_k / 32 + k_block;
  const uint8_t value = source[source_index];
  destination[output_index] = value > 247 ? 247 : value;
}

__global__ void pack_ds4_flash_fp8_block_scale_mma_kernel(
    const uint8_t* source, uint8_t* destination, size_t k_tiles,
    size_t destination_bytes) {
  const size_t output_index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (output_index >= destination_bytes) {
    return;
  }
  constexpr size_t kScaleAtomBytes = 32 * 4 * 4;
  const size_t tile_index = output_index / kScaleAtomBytes;
  const size_t n_tile = tile_index / k_tiles;
  const size_t k_tile = tile_index % k_tiles;
  destination[output_index] = source[n_tile * k_tiles + k_tile];
}

ds4rt_status_t pack_weight_async(
    ds4rt_device_buffer_t source, ds4rt_device_buffer_t destination,
    size_t size_k, size_t source_size_k, size_t source_start_k, size_t size_n,
    size_t row_rotation, void* cuda_stream) {
  if (size_k == 0 || source_size_k == 0 || size_n == 0 || size_k % 16 != 0 ||
      source_size_k % 16 != 0 || source_start_k % 16 != 0 ||
      source_start_k > source_size_k ||
      size_k > source_size_k - source_start_k || size_n % 64 != 0 ||
      row_rotation >= size_n ||
      size_n > std::numeric_limits<size_t>::max() / source_size_k ||
      size_n > std::numeric_limits<size_t>::max() / size_k) {
    return DS4RT_STATUS_INVALID_ARGUMENT;
  }
  const size_t source_bytes = size_n * source_size_k / 2;
  const size_t destination_bytes = size_n * size_k / 2;
  if (!buffer_has_bytes(source, source_bytes) ||
      !buffer_has_bytes(destination, destination_bytes)) {
    return DS4RT_STATUS_BUFFER_TOO_SMALL;
  }
  const size_t words = destination_bytes / sizeof(uint32_t);
  constexpr size_t kThreads = 256;
  const size_t blocks = (words + kThreads - 1) / kThreads;
  pack_ds4_flash_w4a16_weight_kernel<<<
      static_cast<unsigned int>(blocks), kThreads, 0,
      reinterpret_cast<cudaStream_t>(cuda_stream)>>>(
      static_cast<const uint8_t*>(source.ptr),
      static_cast<uint32_t*>(destination.ptr), size_k, source_size_k,
      source_start_k, size_n, row_rotation);
  return status_from_cuda(cudaGetLastError());
}

ds4rt_status_t pack_e8m0_scale_async(
    ds4rt_device_buffer_t source, ds4rt_device_buffer_t destination,
    size_t size_k, size_t source_size_k, size_t source_start_k, size_t size_n,
    size_t row_rotation, void* cuda_stream) {
  if (size_k == 0 || source_size_k == 0 || size_n == 0 || size_k % 32 != 0 ||
      source_size_k % 32 != 0 || source_start_k % 32 != 0 ||
      source_start_k > source_size_k ||
      size_k > source_size_k - source_start_k || size_n % 64 != 0 ||
      row_rotation >= size_n ||
      size_n > std::numeric_limits<size_t>::max() / (source_size_k / 32) ||
      size_n > std::numeric_limits<size_t>::max() / (size_k / 32)) {
    return DS4RT_STATUS_INVALID_ARGUMENT;
  }
  const size_t source_bytes = size_n * (source_size_k / 32);
  const size_t destination_bytes = size_n * (size_k / 32);
  if (!buffer_has_bytes(source, source_bytes) ||
      !buffer_has_bytes(destination, destination_bytes)) {
    return DS4RT_STATUS_BUFFER_TOO_SMALL;
  }
  constexpr size_t kThreads = 256;
  const size_t blocks = (destination_bytes + kThreads - 1) / kThreads;
  pack_ds4_flash_e8m0_k32_scale_kernel<<<
      static_cast<unsigned int>(blocks), kThreads, 0,
      reinterpret_cast<cudaStream_t>(cuda_stream)>>>(
      static_cast<const uint8_t*>(source.ptr),
      static_cast<uint8_t*>(destination.ptr), size_k, source_size_k,
      source_start_k, size_n, row_rotation);
  return status_from_cuda(cudaGetLastError());
}

}  // namespace

extern "C" ds4rt_status_t ds4rt_cuda_ds4_flash_w4a16_pack_weight_async(
    ds4rt_device_buffer_t source, ds4rt_device_buffer_t destination,
    size_t size_k, size_t size_n, size_t row_rotation, void* cuda_stream) {
  return pack_weight_async(source, destination, size_k, size_k, 0, size_n,
                           row_rotation, cuda_stream);
}

extern "C" ds4rt_status_t
ds4rt_cuda_ds4_flash_w4a16_pack_weight_strided_async(
    ds4rt_device_buffer_t source, ds4rt_device_buffer_t destination,
    size_t size_k, size_t source_size_k, size_t source_start_k, size_t size_n,
    size_t row_rotation, void* cuda_stream) {
  return pack_weight_async(source, destination, size_k, source_size_k,
                           source_start_k, size_n, row_rotation, cuda_stream);
}

extern "C" ds4rt_status_t
ds4rt_cuda_ds4_flash_w4a16_pack_e8m0_scale_async(
    ds4rt_device_buffer_t source, ds4rt_device_buffer_t destination,
    size_t size_k, size_t size_n, size_t row_rotation, void* cuda_stream) {
  return pack_e8m0_scale_async(source, destination, size_k, size_k, 0, size_n,
                              row_rotation, cuda_stream);
}

extern "C" ds4rt_status_t
ds4rt_cuda_ds4_flash_w4a16_pack_e8m0_scale_strided_async(
    ds4rt_device_buffer_t source, ds4rt_device_buffer_t destination,
    size_t size_k, size_t source_size_k, size_t source_start_k, size_t size_n,
    size_t row_rotation, void* cuda_stream) {
  return pack_e8m0_scale_async(source, destination, size_k, source_size_k,
                              source_start_k, size_n, row_rotation, cuda_stream);
}

extern "C" ds4rt_status_t
ds4rt_cuda_ds4_flash_fp8_pack_block_scale_mma_async(
    ds4rt_device_buffer_t source, ds4rt_device_buffer_t destination,
    size_t size_n, size_t size_k, void* cuda_stream) {
  if (size_n == 0 || size_k == 0 || size_n % 128 != 0 ||
      size_k % 128 != 0 ||
      size_n > std::numeric_limits<size_t>::max() / size_k) {
    return DS4RT_STATUS_INVALID_ARGUMENT;
  }
  const size_t n_tiles = size_n / 128;
  const size_t k_tiles = size_k / 128;
  const size_t source_bytes = n_tiles * k_tiles;
  constexpr size_t kScaleAtomBytes = 32 * 4 * 4;
  if (source_bytes >
      std::numeric_limits<size_t>::max() / kScaleAtomBytes) {
    return DS4RT_STATUS_INVALID_ARGUMENT;
  }
  const size_t destination_bytes = source_bytes * kScaleAtomBytes;
  if (!buffer_has_bytes(source, source_bytes) ||
      !buffer_has_bytes(destination, destination_bytes)) {
    return DS4RT_STATUS_BUFFER_TOO_SMALL;
  }
  constexpr size_t kThreads = 256;
  const size_t blocks = (destination_bytes + kThreads - 1) / kThreads;
  pack_ds4_flash_fp8_block_scale_mma_kernel<<<
      static_cast<unsigned int>(blocks), kThreads, 0,
      reinterpret_cast<cudaStream_t>(cuda_stream)>>>(
      static_cast<const uint8_t*>(source.ptr),
      static_cast<uint8_t*>(destination.ptr), k_tiles, destination_bytes);
  return status_from_cuda(cudaGetLastError());
}
