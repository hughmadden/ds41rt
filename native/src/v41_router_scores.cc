#include <atomic>
#include <mutex>
#include "ds41rt_v41_router.h"
#include "v41_router_e128.h"
#include "v41_router_e384.h"

namespace {
ds41rt_v41_router_e128_Kernel_Module_t small{};
ds41rt_v41_router_e384_Kernel_Module_t large{};
std::atomic<int> loaded_device{-1};
std::mutex initialization;
using ModuleFn = void (*)(void**);
int load(cudaLibrary_t& library, ModuleFn initialize, ModuleFn load_device, int device) {
  auto* ptr=&library;
  int status=0;
  void* init[]={&ptr,&status}; initialize(init);
  if(!status) {void* args[]={&ptr,&device,&status};load_device(args);}
  return status;
}
}

extern "C" int32_t ds41rt_v41_router_initialize() {
  int device=-1;
  auto status=cudaGetDevice(&device);if(status)return status;
  if(loaded_device.load(std::memory_order_acquire)==device)return 0;
  std::lock_guard<std::mutex> lock(initialization);
  if(loaded_device.load()>=0)return loaded_device.load()==device?0:cudaErrorInvalidDevice;
  cudaStreamCaptureStatus capture;
  status=cudaStreamIsCapturing(cudaStreamPerThread,&capture);if(status)return status;
  if(capture!=cudaStreamCaptureStatusNone)return cudaErrorStreamCaptureUnsupported;
  int major=0,minor=0;
  status=cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device);if(status)return status;
  status=cudaDeviceGetAttribute(&minor,cudaDevAttrComputeCapabilityMinor,device);if(status)return status;
  if(major!=12 || minor!=0)return cudaErrorInvalidDevice;
  int result=load(small.module,_mlir_ds41rt_v41_router_e128_cuda_init,
      _mlir_ds41rt_v41_router_e128_cuda_load_to_device,device);
  if(!result)result=load(large.module,_mlir_ds41rt_v41_router_e384_cuda_init,
      _mlir_ds41rt_v41_router_e384_cuda_load_to_device,device);
  if(result) {
    if(small.module)cudaLibraryUnload(small.module);
    if(large.module)cudaLibraryUnload(large.module);
    small.module=nullptr;large.module=nullptr;
    return result;
  }
  loaded_device.store(device,std::memory_order_release);
  return 0;
}

// Internal projection entry: full buffer/alias validation is in v41_router.cu.
// Never initialize modules or allocate during launch/capture.
extern "C" int32_t ds41rt_v41_router_scores_aot(const uint16_t* input,
    const uint16_t* weight,float* logits,int32_t rows,int32_t experts,void* stream) {
  if(rows<1 || rows>4096 || (experts!=128 && experts!=384))return cudaErrorInvalidValue;
  int device=-1;
  auto status=cudaGetDevice(&device);if(status)return status;
  int ready=loaded_device.load(std::memory_order_acquire);
  if(ready<0)return cudaErrorNotReady;
  if(ready!=device)return cudaErrorInvalidDevice;
  if(experts==128)return cute_dsl_ds41rt_v41_router_e128_wrapper(
      &small,(void*)input,(void*)weight,logits,rows,(cudaStream_t)stream);
  return cute_dsl_ds41rt_v41_router_e384_wrapper(
      &large,(void*)input,(void*)weight,logits,rows,(cudaStream_t)stream);
}
