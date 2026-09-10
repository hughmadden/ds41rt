#include "ds41rt_v41_experts.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace {
constexpr uint64_t hidden = 5120;
template<int Ranks, int TopK>
__global__ void reduce_routes(const float* p0, const float* p1,
    const float* p2, const float* p3, const __nv_bfloat16* shared,
    __nv_bfloat16* output, uint64_t count) {
  const float* planes[] = {p0, p1, p2, p3};
  for (uint64_t offset = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       offset < count; offset += uint64_t(gridDim.x) * blockDim.x) {
    const uint64_t row = offset / hidden, col = offset % hidden;
    float total = 0;
#pragma unroll
    for (int route = 0; route < TopK; ++route) {
      const uint64_t index = (row * TopK + route) * hidden + col;
      float value = planes[0][index];
#pragma unroll
      for (int rank = 1; rank < Ranks; ++rank)
        value = __fadd_rn(value, planes[rank][index]);
      total = __fadd_rn(total, __bfloat162float(__float2bfloat16_rn(value)));
    }
    if (shared) total = __fadd_rn(total, __bfloat162float(shared[offset]));
    output[offset] = __float2bfloat16_rn(total);
  }
}

bool overlaps(const void* a, uint64_t a_bytes, const void* b, uint64_t b_bytes) {
  const auto av = reinterpret_cast<uintptr_t>(a);
  const auto bv = reinterpret_cast<uintptr_t>(b);
  // Subtraction avoids overflow at the upper end of the address space.
  return av <= bv ? bv - av < a_bytes : av - bv < b_bytes;
}
}

extern "C" int32_t ds41rt_v41_reduce_routes_async(const float* const planes[4],
    const uint16_t* shared, uint16_t* output, uint32_t rows,
    uint32_t ranks, uint32_t topk, void* stream) {
  if (!planes || !output || !rows ||
      !((ranks == 1 && topk == 3) || (ranks == 4 && topk == 6)))
    return cudaErrorInvalidValue;
  const uint64_t count = uint64_t(rows) * hidden;
  for (uint32_t rank = 0; rank < 4; ++rank) {
    if (rank >= ranks) {
      if (planes[rank]) return cudaErrorInvalidValue;
    } else if (!planes[rank] || reinterpret_cast<uintptr_t>(planes[rank]) % 4 ||
        overlaps(output, count * 2, planes[rank], count * topk * 4)) {
      return cudaErrorInvalidValue;
    }
  }
  if (reinterpret_cast<uintptr_t>(output) % 2 ||
      (shared && (reinterpret_cast<uintptr_t>(shared) % 2 ||
        (shared != output && overlaps(output, count * 2, shared, count * 2)))))
    return cudaErrorInvalidValue;
  const unsigned blocks = static_cast<unsigned>(count / 256 < 4096 ? count / 256 : 4096);
  auto cuda_stream = static_cast<cudaStream_t>(stream);
  if (ranks == 1)
    reduce_routes<1, 3><<<blocks, 256, 0, cuda_stream>>>(planes[0], nullptr,
        nullptr, nullptr, reinterpret_cast<const __nv_bfloat16*>(shared),
        reinterpret_cast<__nv_bfloat16*>(output), count);
  else
    reduce_routes<4, 6><<<blocks, 256, 0, cuda_stream>>>(planes[0], planes[1],
        planes[2], planes[3], reinterpret_cast<const __nv_bfloat16*>(shared),
        reinterpret_cast<__nv_bfloat16*>(output), count);
  return cudaGetLastError();
}
