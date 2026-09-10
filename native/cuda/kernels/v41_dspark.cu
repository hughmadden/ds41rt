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

#include <cublas_v2.h>
#include <new>
namespace {
struct MarkovHandle { cublasHandle_t blas; int device; void* workspace; };
constexpr uint64_t kMarkovWorkspace = 4 * 1024 * 1024;
int32_t blas_status(cublasStatus_t status) { return status == CUBLAS_STATUS_SUCCESS ? 0 : -int32_t(status); }
}
extern "C" int32_t ds41rt_v41_markov_create(void* workspace, uint64_t bytes, void** output) {
  if (!output) return cudaErrorInvalidValue;
  *output = nullptr;
  if (bytes < kMarkovWorkspace || !span(workspace, kMarkovWorkspace, 256)) return cudaErrorInvalidValue;
  auto* handle = new (std::nothrow) MarkovHandle{};
  if (!handle) return cudaErrorMemoryAllocation;
  auto status = cudaGetDevice(&handle->device);
  if (status != cudaSuccess) { delete handle; return status; }
  auto created = cublasCreate(&handle->blas);
  if (created != CUBLAS_STATUS_SUCCESS) { delete handle; return blas_status(created); }
  handle->workspace = workspace;
  *output = handle;
  return 0;
}
extern "C" int32_t ds41rt_v41_markov_destroy(void* opaque) {
  if (!opaque) return cudaErrorInvalidValue;
  auto* handle = static_cast<MarkovHandle*>(opaque);
  const auto status = cublasDestroy(handle->blas);
  delete handle;
  return blas_status(status);
}
extern "C" int32_t ds41rt_v41_markov_launch(void* opaque, const uint16_t* embedding,
    const uint16_t* weight, float* logits, int32_t rows, void* stream) {
  if (!opaque || rows < 1 || rows > 16) return cudaErrorInvalidValue;
  auto* handle = static_cast<MarkovHandle*>(opaque);
  int device;
  auto status = cudaGetDevice(&device);
  if (status != cudaSuccess) return status;
  if (device != handle->device) return cudaErrorInvalidDevice;
  const uint64_t e = uint64_t(rows) * 512, w = uint64_t(129280) * 512;
  const uint64_t o = uint64_t(rows) * 129280 * 4;
  if (!span(embedding, e, 2) || !span(weight, w, 2) || !span(logits, o, 4) ||
      !disjoint(logits, o, embedding, e) || !disjoint(logits, o, weight, w) ||
      !disjoint(handle->workspace, kMarkovWorkspace, embedding, e) ||
      !disjoint(handle->workspace, kMarkovWorkspace, weight, w) ||
      !disjoint(handle->workspace, kMarkovWorkspace, logits, o)) return cudaErrorInvalidValue;
  auto result = cublasSetStream(handle->blas, reinterpret_cast<cudaStream_t>(stream));
  if (result != CUBLAS_STATUS_SUCCESS) return blas_status(result);
  // SetStream resets the workspace: rebind this wave's owned storage afterward.
  result = cublasSetWorkspace(handle->blas, handle->workspace, kMarkovWorkspace);
  if (result != CUBLAS_STATUS_SUCCESS) return blas_status(result);
  const float alpha = 1, beta = 0;
  return blas_status(cublasGemmEx(handle->blas, CUBLAS_OP_T, CUBLAS_OP_N,
      129280, rows, 256, &alpha, weight, CUDA_R_16BF, 256,
      embedding, CUDA_R_16BF, 256, &beta, logits, CUDA_R_32F, 129280,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
