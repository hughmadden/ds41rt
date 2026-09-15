#include <atomic>
#include <mutex>
#include "v41_index_score.h"
namespace {
ds41rt_v41_index_score_Kernel_Module_t module{};
std::atomic<int> loaded_device[2]{{-1},{-1}};
std::mutex initialization;
}
extern "C" int32_t ds41rt_v41_index_scores_initialize() {
  int device=-1;
  auto status=cudaGetDevice(&device); if(status)return status;
  for(auto& owner:loaded_device)if(owner.load(std::memory_order_acquire)==device)return 0;
  std::lock_guard<std::mutex> lock(initialization);
  int slot=-1;
  for(int i=0;i<2;++i) {
    if(loaded_device[i].load()==device)return 0;
    if(slot<0 && loaded_device[i].load()<0)slot=i;
  }
  if(slot<0)return cudaErrorInvalidDevice;
  auto* ptr=&module.module;
  // Generated launch symbols are process-global. Configure the same CUDA
  // library on both devices rather than overwriting them with a second library.
  const bool existing=module.module!=nullptr;
  void* init[]={&ptr,&status};
  if(!existing)_mlir_ds41rt_v41_index_score_cuda_init(init);
  if(!status) {void* load[]={&ptr,&device,&status}; _mlir_ds41rt_v41_index_score_cuda_load_to_device(load);}
  if(status) {
    if(!existing) {if(module.module)cudaLibraryUnload(module.module);module.module=nullptr;}
    return status;
  }
  loaded_device[slot].store(device,std::memory_order_release);
  return 0;
}
extern "C" int32_t ds41rt_v41_index_scores_overlay_aot(
    const uint8_t* q,const uint8_t* qs,const uint16_t* weights,const uint8_t* keys,
    const uint8_t* ks,const uint32_t* pages,const uint64_t* lengths,
    const uint64_t* metadata,const uint64_t* positions,float* output,
    const uint8_t* proposals,const uint8_t* ps,int32_t rows,int32_t width,
    int32_t slots,int32_t stride,uint64_t capacity,uint64_t proposal_capacity,void* stream) {
  // Normal serving initializes during planning. Direct ABI users must prewarm
  // outside graph capture, as for the existing native projection kernels.
  auto status=ds41rt_v41_index_scores_initialize();if(status)return status;
  return cute_dsl_ds41rt_v41_index_score_wrapper(&module,
      (void*)q,(void*)qs,(void*)weights,(void*)keys,(void*)ks,(void*)pages,
      (void*)lengths,(void*)metadata,(void*)positions,output,(void*)proposals,(void*)ps,
      rows,width,slots,stride,int64_t(capacity),int64_t(proposal_capacity),(cudaStream_t)stream);
}
