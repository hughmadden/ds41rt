#include <cuda_runtime.h>
#include <stdint.h>
#include <cuda_bf16.h>
#include "ds41rt_v41_fp8.h"
namespace {
__global__ void pack_scales(const uint8_t* source, uint8_t* output, uint64_t bytes, uint32_t scale_k) {
  const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= bytes) return;
  const uint64_t k4 = i % 4, row4 = (i / 4) % 4;
  const uint64_t tile = i / 512, nk = tile / (scale_k / 4), kk = tile % (scale_k / 4);
  output[i] = source[(nk * 4 + row4) * scale_k + kk * 4 + k4];
}
__global__ void alpha_one(float* alpha) { *alpha = 1.0f; }
__global__ void shared_swiglu(const __nv_bfloat16* gate, const __nv_bfloat16* up,
    __nv_bfloat16* output, uint64_t count) {
  const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const float g = fminf(__bfloat162float(gate[i]), 10.0f);
  const float u = fminf(fmaxf(__bfloat162float(up[i]), -10.0f), 10.0f);
  output[i] = __float2bfloat16_rn((g / (1.0f + expf(-g))) * u);
}
}
extern "C" int32_t ds41rt_v41_shared_swiglu(const uint16_t* gate, const uint16_t* up,
    uint16_t* output, int32_t rows, void* stream) {
  if (rows < 1 || rows > 4096) return cudaErrorInvalidValue;
  const uint64_t count = uint64_t(rows) * 2304, bytes = count * 2;
  const uintptr_t g = reinterpret_cast<uintptr_t>(gate), u = reinterpret_cast<uintptr_t>(up),
      o = reinterpret_cast<uintptr_t>(output);
  if (!g || !u || !o || (g | u | o) % 2 || g > UINTPTR_MAX - bytes ||
      u > UINTPTR_MAX - bytes || o > UINTPTR_MAX - bytes ||
      (o < g + bytes && g < o + bytes) || (o < u + bytes && u < o + bytes))
    return cudaErrorInvalidValue;
  shared_swiglu<<<(count + 255) / 256, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(gate), reinterpret_cast<const __nv_bfloat16*>(up),
      reinterpret_cast<__nv_bfloat16*>(output), count);
  return cudaGetLastError();
}
extern "C" int32_t ds41rt_v41_fp8_matrix_pack_scales(
    const uint8_t* source, uint8_t* destination, int32_t k, int32_t n, void* stream) {
  if (!((k == 6144 && n == 25600) || (k == 5120 && n == 2304) || (k == 2304 && n == 5120)))
    return cudaErrorInvalidValue;
  const uint64_t src_bytes = uint64_t(k) * n / 1024, dst_bytes = src_bytes * 32;
  auto a = reinterpret_cast<uintptr_t>(source), b = reinterpret_cast<uintptr_t>(destination);
  if (!a || !b || a > UINTPTR_MAX - src_bytes || b > UINTPTR_MAX - dst_bytes ||
      (a < b + dst_bytes && b < a + src_bytes)) return cudaErrorInvalidValue;
  pack_scales<<<(dst_bytes + 255) / 256, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      source, destination, dst_bytes, k / 32);
  return cudaGetLastError();
}
extern "C" int32_t ds41rt_v41_fp8_pack_scales(const uint8_t* source, uint8_t* destination, void* stream) {
  return ds41rt_v41_fp8_matrix_pack_scales(source, destination, 6144, 25600, stream);
}
extern "C" int32_t ds41rt_v41_fp8_initialize_storage(void* scratch, uint64_t bytes, float* alpha, void* stream) {
  auto status = cudaMemsetAsync(scratch, 0, bytes, reinterpret_cast<cudaStream_t>(stream));
  if (status != cudaSuccess) return status;
  alpha_one<<<1, 1, 0, reinterpret_cast<cudaStream_t>(stream)>>>(alpha);
  return cudaGetLastError();
}
