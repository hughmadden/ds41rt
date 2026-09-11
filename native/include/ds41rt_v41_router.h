#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Official sqrtsoftplus routing: BF16 hidden [rows,5120], weight [experts,5120],
// FP32 biases [experts]; optional byte image mask [rows] selects bias_vl.
// FP32 scores workspace [rows,experts], u32 IDs and FP32 weights [rows,topk].
// experts=128/topk=3 or experts=384/topk=6; outputs disjoint from all inputs.
int32_t ds41rt_v41_router(const uint16_t* hidden,const uint16_t* weight,
    const float* bias,const float* bias_vl,const uint8_t* image_mask,float* scores,
    uint32_t* ids,float* routing,int32_t rows,int32_t experts,void* stream);
// Convert caller-projected FP32 logits in place to sqrtsoftplus scores and
// select/normalize with the same contract as ds41rt_v41_router.
int32_t ds41rt_v41_router_select_logits(float* scores,const float* bias,
    const float* bias_vl,const uint8_t* image_mask,uint32_t* ids,float* routing,
    int32_t rows,int32_t experts,void* stream);
#ifdef __cplusplus
}
#endif
