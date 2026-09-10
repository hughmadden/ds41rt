#include "ds41rt_v41_fp8.h"
#include <cuda_runtime.h>
#include <mutex>
#include "v41_fp8_variants.h"
static_assert(sizeof(ds41rt_v41_fp8_info_t) == 56);
namespace {
using ModuleFn = void (*)(void**);
using LaunchFn = void (*)(void**, int32_t);
struct Module {
  ModuleFn initialize, load;
  LaunchFn launch;
  cudaLibrary_t library = nullptr;
  void reset() { if (library) cudaLibraryUnload(library); library = nullptr; }
  ~Module() { reset(); }
};
struct Variant {
  ds41rt_v41_fp8_info_t info;
  Module quant, gemm;
  const uint32_t* grids;
  int device = -1;
};
Variant variants[] = {DS41RT_V41_FP8_VARIANTS};
std::mutex mutex;
Variant* capacity(int rows, int k, int n) { for (auto& v : variants) if (int(v.info.capacity_rows) == rows && int(v.info.input_dim) == k && int(v.info.output_dim) == n) return &v; return nullptr; }
Variant* handle(void* p) { for (auto& v : variants) if (&v == p) return &v; return nullptr; }
int load(Module& m, int device) {
  auto* ptr = &m.library;
  int status = 0;
  void* init[] = {&ptr, &status}; m.initialize(init);
  if (!status) { void* args[] = {&ptr, &device, &status}; m.load(args); }
  if (status) m.reset();
  return status;
}
bool span(const void* p, uint64_t bytes, uintptr_t& start, uintptr_t& end) {
  start = reinterpret_cast<uintptr_t>(p);
  if (!start || start % 16 || start > UINTPTR_MAX - bytes) return false;
  end = start + bytes; return true;
}
int device_matches(Variant* v) {
  if (!v || v->device < 0) return cudaErrorInvalidValue;
  int device = -1; auto status = cudaGetDevice(&device);
  return status ? int(status) : (device == v->device ? 0 : int(cudaErrorInvalidDevice));
}
}
extern "C" int32_t ds41rt_v41_fp8_initialize_storage(void*, uint64_t, float*, void*);
extern "C" int32_t ds41rt_v41_fp8_matrix_info(int32_t rows, int32_t k, int32_t n, ds41rt_v41_fp8_info_t* out) {
  auto* v = capacity(rows, k, n); if (!v || !out) return cudaErrorInvalidValue;
  *out = v->info; return 0;
}
extern "C" int32_t ds41rt_v41_fp8_matrix_initialize(int32_t rows, int32_t k, int32_t n, void** out) {
  if (!out) return cudaErrorInvalidValue; *out = nullptr;
  auto* v = capacity(rows, k, n); if (!v) return cudaErrorInvalidValue;
  int device, major, minor, sms;
  auto status = cudaGetDevice(&device); if (status) return status;
  status = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device); if (status) return status;
  status = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device); if (status) return status;
  status = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device); if (status) return status;
  if (major != 12 || minor != 0 || sms != DS41RT_V41_FP8_SMS) return cudaErrorInvalidDevice;
  std::lock_guard<std::mutex> lock(mutex);
  if (v->device >= 0) {
    if (v->device != device) return cudaErrorInvalidDevice;
  } else {
    int result = load(v->quant, device);
    if (!result) result = load(v->gemm, device);
    if (result) { v->quant.reset(); v->gemm.reset(); return result; }
    v->device = device;
  }
  *out = v; return 0;
}
extern "C" int32_t ds41rt_v41_fp8_initialize_scratch(void* kernel, void* scratch, uint64_t bytes, float* alpha, void* stream) {
  auto* v = handle(kernel); int status = device_matches(v); if (status) return status;
  uintptr_t a,b,c,d;
  if (bytes < v->info.scratch_bytes || !span(scratch, v->info.scratch_bytes,a,b) ||
      !span(alpha,4,c,d) || (a<d && c<b)) return cudaErrorInvalidValue;
  return ds41rt_v41_fp8_initialize_storage(scratch, v->info.scratch_bytes, alpha, stream);
}
extern "C" int32_t ds41rt_v41_fp8_launch(void* kernel, const uint16_t* source, const uint8_t* weight,
    const uint8_t* packed_scales, void* scratch, uint64_t bytes, const float* alpha,
    uint16_t* output, int32_t rows, void* stream) {
  auto* v = handle(kernel); int status = device_matches(v); if (status) return status;
  if (rows <= 0 || uint32_t(rows) > v->info.capacity_rows || bytes < v->info.scratch_bytes) return cudaErrorInvalidValue;
  const void* buffers[] = {source,weight,packed_scales,scratch,alpha,output};
  uint64_t sizes[] = {uint64_t(rows)*v->info.input_dim*2,uint64_t(v->info.output_dim)*v->info.input_dim,v->info.packed_weight_scale_bytes,v->info.scratch_bytes,4,uint64_t(rows)*v->info.output_dim*2};
  uintptr_t starts[6], ends[6];
  for (int i=0;i<6;++i) {
    if (!span(buffers[i],sizes[i],starts[i],ends[i])) return cudaErrorInvalidValue;
    for (int j=0;j<i;++j) if (starts[i]<ends[j] && starts[j]<ends[i]) return cudaErrorInvalidValue;
  }
  void* x = const_cast<uint16_t*>(source);
  void* a = static_cast<char*>(scratch)+v->info.values_offset;
  void* sr = static_cast<char*>(scratch)+v->info.row_scales_offset;
  void* sm = static_cast<char*>(scratch)+v->info.mma_scales_offset;
  int grid = v->grids[rows-1];
  void* quant_args[] = {&x,&a,&sr,&sm,&rows,&grid,&stream,&status};
  v->quant.launch(quant_args,8); if (status) return status;
  void* w=const_cast<uint8_t*>(weight), *s=const_cast<uint8_t*>(packed_scales), *c=output, *one=const_cast<float*>(alpha);
  // Quantized-output slots are compile-time inactive for this BF16 projection.
  void* gemm_args[] = {&a,&w,&sm,&s,&c,&c,&c,&c,&one,&rows,&stream,&status};
  v->gemm.launch(gemm_args,12); return status;
}

// Existing engram entry points retain their explicit geometry.
extern "C" int32_t ds41rt_v41_fp8_info(int32_t rows, ds41rt_v41_fp8_info_t* out) {
  return ds41rt_v41_fp8_matrix_info(rows, 6144, 25600, out);
}
extern "C" int32_t ds41rt_v41_fp8_initialize(int32_t rows, void** out) {
  return ds41rt_v41_fp8_matrix_initialize(rows, 6144, 25600, out);
}
