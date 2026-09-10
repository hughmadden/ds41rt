#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "ds41rt_v41_dspark.h"
namespace {
__device__ float warp_sum(float value) {
  for (int offset = 16; offset; offset >>= 1)
    value += __shfl_down_sync(0xffffffffu, value, offset);
  return value;
}
// Fuse concatenation and BF16 -> FP32 promotion into the projection. The
// reference promotes checkpoint confidence weights and both inputs to FP32.
__global__ void confidence(const __nv_bfloat16* hidden,
    const __nv_bfloat16* markov, const __nv_bfloat16* weight, float* output) {
  const uint64_t row = blockIdx.x;
  const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
  float sum = 0;
  for (int col = tid; col < 5376; col += 256) {
    const float x = __bfloat162float(col < 5120 ? hidden[row * 5120 + col]
        : markov[row * 256 + col - 5120]);
    sum = fmaf(x, __bfloat162float(weight[col]), sum);
  }
  sum = warp_sum(sum);
  __shared__ float partials[8];
  if (lane == 0) partials[warp] = sum;
  __syncthreads();
  if (warp == 0) {
    sum = warp_sum(lane < 8 ? partials[lane] : 0);
    if (lane == 0) output[row] = sum;
  }
}
bool span(const void* p, uint64_t bytes, uint64_t alignment) {
  const auto address = reinterpret_cast<uintptr_t>(p);
  return address && address % alignment == 0 && address <= UINTPTR_MAX - bytes;
}
bool disjoint(const void* a, uint64_t na, const void* b, uint64_t nb) {
  const auto x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
  return x + na <= y || y + nb <= x;
}
}
extern "C" int32_t ds41rt_v41_dspark_confidence(const uint16_t* hidden,
    const uint16_t* markov, const uint16_t* weight, float* output,
    int32_t rows, void* stream) {
  if (rows <= 0 || rows > 4096) return cudaErrorInvalidValue;
  const uint64_t h = uint64_t(rows) * 10240, m = uint64_t(rows) * 512;
  const uint64_t w = 10752, o = uint64_t(rows) * 4;
  if (!span(hidden, h, 2) || !span(markov, m, 2) || !span(weight, w, 2) ||
      !span(output, o, 4) || !disjoint(hidden, h, output, o) ||
      !disjoint(markov, m, output, o) || !disjoint(weight, w, output, o))
    return cudaErrorInvalidValue;
  confidence<<<rows, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(hidden),
      reinterpret_cast<const __nv_bfloat16*>(markov),
      reinterpret_cast<const __nv_bfloat16*>(weight), output);
  return cudaGetLastError();
}
