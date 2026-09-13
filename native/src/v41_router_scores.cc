#include <atomic>
#include <mutex>
#include "ds41rt_v41_router.h"
#include "v41_router_e128.h"
#include "v41_router_e384.h"

namespace {
ds41rt_v41_router_e128_Kernel_Module_t small{};
ds41rt_v41_router_e384_Kernel_Module_t large{};
std::atomic<int> loaded_device[2]{{-1},{-1}};
std::mutex initialization;
using ModuleFn = void (*)(void**);
int load(cudaLibrary_t& library, ModuleFn initialize, ModuleFn load_device, int device) {
  auto* ptr=&library;
  int status=0;
  const bool existing=library!=nullptr;
  void* init[]={&ptr,&status}; if(!existing)initialize(init);
  if(!status) {void* args[]={&ptr,&device,&status};load_device(args);}
  if(status && !existing) { if(library)cudaLibraryUnload(library);library=nullptr; }
  return status;
}
}

extern "C" int32_t ds41rt_v41_router_initialize() {
  int device=-1;
  auto status=cudaGetDevice(&device);if(status)return status;
  for(auto& owner:loaded_device)if(owner.load(std::memory_order_acquire)==device)return 0;
  std::lock_guard<std::mutex> lock(initialization);
  int slot=-1;
  for(int i=0;i<2;++i) {
    if(loaded_device[i].load()==device)return 0;
    if(slot<0 && loaded_device[i].load()<0)slot=i;
  }
  if(slot<0)return cudaErrorInvalidDevice;
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
  if(result)return result;
  loaded_device[slot].store(device,std::memory_order_release);
  return 0;
}

// Internal projection entry: full buffer/alias validation is in v41_router.cu.
// Never initialize modules or allocate during launch/capture.
extern "C" int32_t ds41rt_v41_router_scores_aot(const uint16_t* input,
    const uint16_t* weight,float* logits,int32_t rows,int32_t experts,void* stream) {
  if(rows<1 || rows>4096 || (experts!=128 && experts!=384))return cudaErrorInvalidValue;
  int device=-1;
  auto status=cudaGetDevice(&device);if(status)return status;
  bool ready=false,initialized=false;
  for(auto& owner:loaded_device) {
    const int id=owner.load(std::memory_order_acquire);
    initialized|=id>=0;
    if(id==device) { ready=true;break; }
  }
  if(!ready)return initialized?cudaErrorInvalidDevice:cudaErrorNotReady;
  if(experts==128)return cute_dsl_ds41rt_v41_router_e128_wrapper(
      &small,(void*)input,(void*)weight,logits,rows,(cudaStream_t)stream);
  return cute_dsl_ds41rt_v41_router_e384_wrapper(
      &large,(void*)input,(void*)weight,logits,rows,(cudaStream_t)stream);
}
