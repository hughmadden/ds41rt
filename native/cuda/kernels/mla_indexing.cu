#include "common.h"

#if defined(__GNUC__)
#define DS4RT_MLA_INDEXING_EXPORT __attribute__((visibility("default")))
#else
#define DS4RT_MLA_INDEXING_EXPORT
#endif

namespace {

// Transactional compatibility-KV geometry. DeepSeek target-attention storage
// owns a separate 256-token physical-page ABI.
constexpr size_t kGenericKvPageSize = 64;
constexpr size_t kBf16ValuesPerVector = sizeof(uint4) / sizeof(uint16_t);

__global__ void transpose_rows_heads_bf16_kernel(
    const uint4* input, uint4* output, size_t rows, size_t heads,
    size_t width_vectors) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = rows * heads * width_vectors;
  if (index >= total) {
    return;
  }
  const size_t vector_col = index % width_vectors;
  const size_t row_head = index / width_vectors;
  const size_t head = row_head % heads;
  const size_t row = row_head / heads;
  output[(head * rows + row) * width_vectors + vector_col] = input[index];
}

__global__ void transpose_heads_rows_bf16_kernel(
    const uint4* input, uint4* output, size_t rows, size_t heads,
    size_t width_vectors) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = rows * heads * width_vectors;
  if (index >= total) {
    return;
  }
  const size_t vector_col = index % width_vectors;
  const size_t head_row = index / width_vectors;
  const size_t row = head_row % rows;
  const size_t head = head_row / rows;
  output[(row * heads + head) * width_vectors + vector_col] = input[index];
}

__global__ void mla_compose_absorbed_query_bf16_kernel(
    const uint4* latent_heads_rows, const uint4* rope_rows_heads,
    uint4* output_rows_heads, size_t rows, size_t heads,
    size_t latent_vectors, size_t rope_vectors) {
  const size_t output_vectors = latent_vectors + rope_vectors;
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = rows * heads * output_vectors;
  if (index >= total) {
    return;
  }
  const size_t vector_col = index % output_vectors;
  const size_t row_head = index / output_vectors;
  const size_t head = row_head % heads;
  const size_t row = row_head / heads;
  if (vector_col < latent_vectors) {
    output_rows_heads[index] =
        latent_heads_rows[(head * rows + row) * latent_vectors + vector_col];
  } else {
    output_rows_heads[index] =
        rope_rows_heads[(row * heads + head) * rope_vectors +
                        vector_col - latent_vectors];
  }
}

__global__ void generic_kv_page_table_init_kernel(
    int32_t* page_table, size_t query_rows, size_t page_table_width,
    int32_t base_offset) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = query_rows * page_table_width;
  if (index < total) {
    page_table[index] =
        base_offset + static_cast<int32_t>(index % page_table_width);
  }
}

__global__ void generic_kv_page_table_init_offsets_kernel(
    int32_t* page_table, const int32_t* row_offsets, size_t query_rows,
    size_t page_table_width) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = query_rows * page_table_width;
  if (index < total) {
    const size_t row = index / page_table_width;
    const size_t column = index % page_table_width;
    page_table[index] = row_offsets[row] + static_cast<int32_t>(column);
  }
}

__global__ void generic_kv_page_table_expand_indices_kernel(
    int32_t* output_indices, const uint32_t* physical_pages,
    size_t query_rows, size_t output_width, size_t active_tokens) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = query_rows * output_width;
  if (index >= total) {
    return;
  }
  const size_t logical_token = index % output_width;
  if (logical_token >= active_tokens) {
    output_indices[index] = 0;
    return;
  }
  const size_t page_index = logical_token / kGenericKvPageSize;
  const size_t page_token = logical_token % kGenericKvPageSize;
  output_indices[index] = static_cast<int32_t>(
      static_cast<size_t>(physical_pages[page_index]) * kGenericKvPageSize +
      page_token);
}

ds4rt_status_t validate_transpose_rows_heads_bf16_args(
    const uint16_t* input, const uint16_t* output, size_t rows,
    size_t heads, size_t width) {
  if (input == nullptr || output == nullptr || rows == 0 || heads == 0 ||
      width == 0 || width % kBf16ValuesPerVector != 0 ||
      reinterpret_cast<uintptr_t>(input) % alignof(uint4) != 0 ||
      reinterpret_cast<uintptr_t>(output) % alignof(uint4) != 0) {
    return DS4RT_STATUS_INVALID_ARGUMENT;
  }
  size_t values = 0;
  return checked_mul(rows, heads, &values) && checked_mul(values, width, &values)
             ? DS4RT_STATUS_OK
             : DS4RT_STATUS_INVALID_ARGUMENT;
}

ds4rt_status_t validate_mla_compose_absorbed_query_bf16_args(
    const uint16_t* latent_heads_rows, const uint16_t* rope_rows_heads,
    const uint16_t* output_rows_heads, size_t rows, size_t heads,
    size_t latent_width, size_t rope_width) {
  if (latent_heads_rows == nullptr || rope_rows_heads == nullptr ||
      output_rows_heads == nullptr || rows == 0 || heads == 0 ||
      latent_width == 0 || rope_width == 0 ||
      latent_width % kBf16ValuesPerVector != 0 ||
      rope_width % kBf16ValuesPerVector != 0 ||
      reinterpret_cast<uintptr_t>(latent_heads_rows) % alignof(uint4) != 0 ||
      reinterpret_cast<uintptr_t>(rope_rows_heads) % alignof(uint4) != 0 ||
      reinterpret_cast<uintptr_t>(output_rows_heads) % alignof(uint4) != 0) {
    return DS4RT_STATUS_INVALID_ARGUMENT;
  }
  size_t values = 0;
  size_t output_width = 0;
  return checked_add(latent_width, rope_width, &output_width) &&
                 checked_mul(rows, heads, &values) &&
                 checked_mul(values, output_width, &values)
             ? DS4RT_STATUS_OK
             : DS4RT_STATUS_INVALID_ARGUMENT;
}

ds4rt_status_t validate_generic_kv_page_table_init_args(
    const int32_t* page_table, size_t query_rows, size_t page_table_width) {
  size_t entries = 0;
  if (page_table == nullptr || query_rows == 0 || page_table_width == 0 ||
      page_table_width > static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
      !checked_mul(query_rows, page_table_width, &entries)) {
    return DS4RT_STATUS_INVALID_ARGUMENT;
  }
  return DS4RT_STATUS_OK;
}

ds4rt_status_t validate_generic_kv_page_table_base_offset(
    size_t page_table_width, size_t base_offset) {
  size_t end = 0;
  if (!checked_add(base_offset, page_table_width, &end) ||
      end > static_cast<size_t>(std::numeric_limits<int32_t>::max()) + 1) {
    return DS4RT_STATUS_INVALID_ARGUMENT;
  }
  return DS4RT_STATUS_OK;
}

ds4rt_status_t validate_generic_kv_page_table_expand_indices_args(
    const int32_t* output_indices, const uint32_t* physical_pages,
    size_t query_rows, size_t output_width, size_t active_tokens) {
  size_t entries = 0;
  if (output_indices == nullptr || physical_pages == nullptr ||
      query_rows == 0 || output_width == 0 || active_tokens == 0 ||
      active_tokens > output_width ||
      !checked_mul(query_rows, output_width, &entries) ||
      active_tokens >
          static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    return DS4RT_STATUS_INVALID_ARGUMENT;
  }
  return DS4RT_STATUS_OK;
}

extern "C" DS4RT_MLA_INDEXING_EXPORT ds4rt_status_t
ds4rt_cuda_transpose_rows_heads_bf16_async(
    const uint16_t* input, uint16_t* output, size_t rows, size_t heads,
    size_t width, void* cuda_stream) {
  const ds4rt_status_t valid = validate_transpose_rows_heads_bf16_args(
      input, output, rows, heads, width);
  if (valid != DS4RT_STATUS_OK) {
    return valid;
  }
  const size_t total_vectors = rows * heads * (width / kBf16ValuesPerVector);
  constexpr int threads = 256;
  const size_t blocks = (total_vectors + threads - 1) / threads;
  if (blocks > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return DS4RT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  transpose_rows_heads_bf16_kernel<<<static_cast<int>(blocks), threads, 0, stream>>>(
      reinterpret_cast<const uint4*>(input), reinterpret_cast<uint4*>(output),
      rows, heads, width / kBf16ValuesPerVector);
  return status_from_cuda(cudaGetLastError());
}

extern "C" DS4RT_MLA_INDEXING_EXPORT ds4rt_status_t
ds4rt_cuda_transpose_rows_heads_bf16(
    const uint16_t* input, uint16_t* output, size_t rows, size_t heads,
    size_t width) {
  const ds4rt_status_t status = ds4rt_cuda_transpose_rows_heads_bf16_async(
      input, output, rows, heads, width, nullptr);
  if (status != DS4RT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" DS4RT_MLA_INDEXING_EXPORT ds4rt_status_t
ds4rt_cuda_transpose_heads_rows_bf16_async(
    const uint16_t* input, uint16_t* output, size_t rows, size_t heads,
    size_t width, void* cuda_stream) {
  const ds4rt_status_t valid = validate_transpose_rows_heads_bf16_args(
      input, output, rows, heads, width);
  if (valid != DS4RT_STATUS_OK) {
    return valid;
  }
  const size_t total_vectors = rows * heads * (width / kBf16ValuesPerVector);
  constexpr int threads = 256;
  const size_t blocks = (total_vectors + threads - 1) / threads;
  if (blocks > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return DS4RT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  transpose_heads_rows_bf16_kernel<<<static_cast<int>(blocks), threads, 0, stream>>>(
      reinterpret_cast<const uint4*>(input), reinterpret_cast<uint4*>(output),
      rows, heads, width / kBf16ValuesPerVector);
  return status_from_cuda(cudaGetLastError());
}

extern "C" DS4RT_MLA_INDEXING_EXPORT ds4rt_status_t
ds4rt_cuda_transpose_heads_rows_bf16(
    const uint16_t* input, uint16_t* output, size_t rows, size_t heads,
    size_t width) {
  const ds4rt_status_t status = ds4rt_cuda_transpose_heads_rows_bf16_async(
      input, output, rows, heads, width, nullptr);
  if (status != DS4RT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" DS4RT_MLA_INDEXING_EXPORT ds4rt_status_t
ds4rt_cuda_mla_compose_absorbed_query_bf16_async(
    const uint16_t* latent_heads_rows, const uint16_t* rope_rows_heads,
    uint16_t* output_rows_heads, size_t rows, size_t heads,
    size_t latent_width, size_t rope_width, void* cuda_stream) {
  const ds4rt_status_t valid = validate_mla_compose_absorbed_query_bf16_args(
      latent_heads_rows, rope_rows_heads, output_rows_heads, rows, heads,
      latent_width, rope_width);
  if (valid != DS4RT_STATUS_OK) {
    return valid;
  }
  const size_t output_vectors =
      (latent_width + rope_width) / kBf16ValuesPerVector;
  const size_t total_vectors = rows * heads * output_vectors;
  constexpr int threads = 256;
  const size_t blocks = (total_vectors + threads - 1) / threads;
  if (blocks > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return DS4RT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  mla_compose_absorbed_query_bf16_kernel<<<static_cast<int>(blocks), threads, 0, stream>>>(
      reinterpret_cast<const uint4*>(latent_heads_rows),
      reinterpret_cast<const uint4*>(rope_rows_heads),
      reinterpret_cast<uint4*>(output_rows_heads), rows, heads,
      latent_width / kBf16ValuesPerVector,
      rope_width / kBf16ValuesPerVector);
  return status_from_cuda(cudaGetLastError());
}

extern "C" DS4RT_MLA_INDEXING_EXPORT ds4rt_status_t
ds4rt_cuda_mla_compose_absorbed_query_bf16(
    const uint16_t* latent_heads_rows, const uint16_t* rope_rows_heads,
    uint16_t* output_rows_heads, size_t rows, size_t heads,
    size_t latent_width, size_t rope_width) {
  const ds4rt_status_t status = ds4rt_cuda_mla_compose_absorbed_query_bf16_async(
      latent_heads_rows, rope_rows_heads, output_rows_heads, rows, heads,
      latent_width, rope_width, nullptr);
  if (status != DS4RT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" DS4RT_MLA_INDEXING_EXPORT ds4rt_status_t
ds4rt_cuda_generic_kv_page_table_init_async(
    int32_t* page_table, size_t query_rows, size_t page_table_width,
    void* cuda_stream) {
  return ds4rt_cuda_generic_kv_page_table_init_base_async(
      page_table, query_rows, page_table_width, 0, cuda_stream);
}

extern "C" DS4RT_MLA_INDEXING_EXPORT ds4rt_status_t
ds4rt_cuda_generic_kv_page_table_init_base_async(
    int32_t* page_table, size_t query_rows, size_t page_table_width,
    size_t base_offset, void* cuda_stream) {
  const ds4rt_status_t valid = validate_generic_kv_page_table_init_args(
      page_table, query_rows, page_table_width);
  if (valid != DS4RT_STATUS_OK) {
    return valid;
  }
  const ds4rt_status_t valid_base =
      validate_generic_kv_page_table_base_offset(page_table_width, base_offset);
  if (valid_base != DS4RT_STATUS_OK) {
    return valid_base;
  }
  const size_t entries = query_rows * page_table_width;
  constexpr int threads = 256;
  const size_t blocks = (entries + threads - 1) / threads;
  if (blocks > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return DS4RT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  generic_kv_page_table_init_kernel<<<static_cast<int>(blocks), threads, 0, stream>>>(
      page_table, query_rows, page_table_width,
      static_cast<int32_t>(base_offset));
  return status_from_cuda(cudaGetLastError());
}

extern "C" DS4RT_MLA_INDEXING_EXPORT ds4rt_status_t
ds4rt_cuda_generic_kv_page_table_init(
    int32_t* page_table, size_t query_rows, size_t page_table_width) {
  const ds4rt_status_t status = ds4rt_cuda_generic_kv_page_table_init_async(
      page_table, query_rows, page_table_width, nullptr);
  if (status != DS4RT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" DS4RT_MLA_INDEXING_EXPORT ds4rt_status_t
ds4rt_cuda_generic_kv_page_table_init_base(
    int32_t* page_table, size_t query_rows, size_t page_table_width,
    size_t base_offset) {
  const ds4rt_status_t status = ds4rt_cuda_generic_kv_page_table_init_base_async(
      page_table, query_rows, page_table_width, base_offset, nullptr);
  if (status != DS4RT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" DS4RT_MLA_INDEXING_EXPORT ds4rt_status_t
ds4rt_cuda_generic_kv_page_table_init_offsets_async(
    int32_t* page_table, const int32_t* row_offsets, size_t query_rows,
    size_t page_table_width, void* cuda_stream) {
  const ds4rt_status_t valid = validate_generic_kv_page_table_init_args(
      page_table, query_rows, page_table_width);
  if (valid != DS4RT_STATUS_OK) {
    return valid;
  }
  if (row_offsets == nullptr) {
    return DS4RT_STATUS_INVALID_ARGUMENT;
  }
  const size_t entries = query_rows * page_table_width;
  constexpr int threads = 256;
  const size_t blocks = (entries + threads - 1) / threads;
  if (blocks > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return DS4RT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  generic_kv_page_table_init_offsets_kernel<<<static_cast<int>(blocks), threads,
                                                0, stream>>>(
      page_table, row_offsets, query_rows, page_table_width);
  return status_from_cuda(cudaGetLastError());
}

extern "C" DS4RT_MLA_INDEXING_EXPORT ds4rt_status_t
ds4rt_cuda_generic_kv_page_table_init_offsets(
    int32_t* page_table, const int32_t* row_offsets, size_t query_rows,
    size_t page_table_width) {
  const ds4rt_status_t status =
      ds4rt_cuda_generic_kv_page_table_init_offsets_async(
          page_table, row_offsets, query_rows, page_table_width, nullptr);
  if (status != DS4RT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" DS4RT_MLA_INDEXING_EXPORT ds4rt_status_t
ds4rt_cuda_generic_kv_page_table_expand_indices_async(
    int32_t* output_indices, const uint32_t* physical_pages,
    size_t query_rows, size_t output_width, size_t active_tokens,
    void* cuda_stream) {
  const ds4rt_status_t valid =
      validate_generic_kv_page_table_expand_indices_args(
          output_indices, physical_pages, query_rows, output_width,
          active_tokens);
  if (valid != DS4RT_STATUS_OK) {
    return valid;
  }
  const size_t entries = query_rows * output_width;
  constexpr int threads = 256;
  const size_t blocks = (entries + threads - 1) / threads;
  if (blocks > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return DS4RT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  generic_kv_page_table_expand_indices_kernel<<<
      static_cast<int>(blocks), threads, 0, stream>>>(
      output_indices, physical_pages, query_rows, output_width, active_tokens);
  return status_from_cuda(cudaGetLastError());
}

extern "C" DS4RT_MLA_INDEXING_EXPORT ds4rt_status_t
ds4rt_cuda_generic_kv_page_table_expand_indices(
    int32_t* output_indices, const uint32_t* physical_pages,
    size_t query_rows, size_t output_width, size_t active_tokens) {
  const ds4rt_status_t status =
      ds4rt_cuda_generic_kv_page_table_expand_indices_async(
          output_indices, physical_pages, query_rows, output_width,
          active_tokens, nullptr);
  if (status != DS4RT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

}  // namespace
