#include <cuda_runtime.h>
#include <stdint.h>
#include "ds41rt_v41_fp8.h"
namespace {
__global__ void pack_scales(const uint8_t* source, uint8_t* output) {
  const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= 4915200) return;
  const uint64_t k4 = i % 4, row4 = (i / 4) % 4;
  const uint64_t tile = i / 512, nk = tile / 48, kk = tile % 48;
  output[i] = source[(nk * 4 + row4) * 192 + kk * 4 + k4];
}
__global__ void alpha_one(float* alpha) { *alpha = 1.0f; }
}
extern "C" int32_t ds41rt_v41_fp8_pack_scales(const uint8_t* source, uint8_t* destination, void* stream) {
  auto a = reinterpret_cast<uintptr_t>(source), b = reinterpret_cast<uintptr_t>(destination);
  if (!a || !b || a > UINTPTR_MAX - 153600 || b > UINTPTR_MAX - 4915200 ||
      (a < b + 4915200 && b < a + 153600)) return cudaErrorInvalidValue;
  pack_scales<<<19200, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(source, destination);
  return cudaGetLastError();
}
extern "C" int32_t ds41rt_v41_fp8_initialize_storage(void* scratch, uint64_t bytes, float* alpha, void* stream) {
  auto status = cudaMemsetAsync(scratch, 0, bytes, reinterpret_cast<cudaStream_t>(stream));
  if (status != cudaSuccess) return status;
  alpha_one<<<1, 1, 0, reinterpret_cast<cudaStream_t>(stream)>>>(alpha);
  return cudaGetLastError();
}
