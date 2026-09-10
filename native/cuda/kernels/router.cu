#include "common.h"

#include <cub/cub.cuh>

namespace {

constexpr size_t kDs4FlashHiddenDim = 4096;
constexpr size_t kDs4FlashExperts = 256;
constexpr size_t kDs4FlashTopK = 6;
constexpr size_t kDs4FlashVocabSize = 129280;
constexpr float kDs4FlashRoutedScalingFactor = 1.5f;
constexpr size_t kDs4FlashHybridCandidates = 12;
constexpr size_t kDs4ProHiddenDim = 7168;
constexpr size_t kDs4ProExperts = 384;
constexpr size_t kDs4ProTopK = 6;
constexpr size_t kDs4ProVocabSize = 129280;
constexpr float kDs4ProRoutedScalingFactor = 2.5f;
constexpr size_t kDs4ProExactRouterMaxRows = 256;

__device__ float sqrt_softplus_f32(float value) {
  const float softplus = fmaxf(value, 0.0f) + log1pf(expf(-fabsf(value)));
  return sqrtf(softplus);
}

template <size_t HiddenDim, size_t Experts>
__global__ void ds4_router_bf16_score_workspace_kernel(
    const uint16_t *hidden, const uint16_t *router_weight, float *raw_scores,
    size_t rows) {
  __shared__ float scratch[kBlock];
  const size_t expert = blockIdx.x;
  const size_t row = blockIdx.y;
  if (row >= rows || expert >= Experts) {
    return;
  }
  const size_t tid = threadIdx.x;
  const uint16_t *row_hidden = hidden + row * HiddenDim;
  const uint16_t *weight_row = router_weight + expert * HiddenDim;
  float partial = 0.0f;
  for (size_t col = tid; col < HiddenDim; col += blockDim.x) {
    partial = fmaf(bf16_to_f32(row_hidden[col]), bf16_to_f32(weight_row[col]),
                   partial);
  }
  scratch[tid] = partial;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < static_cast<size_t>(stride)) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }
  if (tid != 0) {
    return;
  }

  const float score = sqrt_softplus_f32(scratch[0]);
  raw_scores[row * Experts + expert] = score;
}

template <size_t Experts>
__global__ void ds4_router_logits_to_scores_kernel(float *raw_scores,
                                                   size_t rows) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
  const size_t values = rows * Experts;
  if (index < values) {
    raw_scores[index] = sqrt_softplus_f32(raw_scores[index]);
  }
}

// Score production remains fully parallel over (row, expert). Selection is a
// separate, stream-ordered pass so no block observes another block's ordinary
// global writes through an ad-hoc atomic lock. Scanning experts in ascending
// order also gives equal corrected scores a stable expert-ID tie break.
template <size_t Experts, size_t TopK>
__global__ void ds4_router_select_workspace_kernel(
    const float *raw_scores, const float *correction_bias,
    uint32_t *topk_indices, float *topk_scores, float *topk_weights,
    size_t rows, float routed_scaling_factor) {
  const size_t row = blockIdx.x;
  if (row >= rows || threadIdx.x != 0) {
    return;
  }

  float best_scores[TopK];
  float best_corrected[TopK];
  uint32_t best_indices[TopK];
  for (size_t rank = 0; rank < TopK; ++rank) {
    best_scores[rank] = 0.0f;
    best_corrected[rank] = -CUDART_INF_F;
    best_indices[rank] = 0;
  }
  const float *row_scores = raw_scores + row * Experts;
  for (size_t expert = 0; expert < Experts; ++expert) {
    const float raw_score = row_scores[expert];
    const float score = isfinite(raw_score) ? raw_score : 0.0f;
    const float raw_corrected = score + correction_bias[expert];
    const float corrected = isfinite(raw_score) && isfinite(raw_corrected)
                                ? raw_corrected
                                : -CUDART_INF_F;
    for (size_t rank = 0; rank < TopK; ++rank) {
      if (corrected > best_corrected[rank] ||
          (corrected == best_corrected[rank] &&
           static_cast<uint32_t>(expert) < best_indices[rank])) {
        for (size_t shift = TopK - 1; shift > rank; --shift) {
          best_indices[shift] = best_indices[shift - 1];
          best_scores[shift] = best_scores[shift - 1];
          best_corrected[shift] = best_corrected[shift - 1];
        }
        best_indices[rank] = static_cast<uint32_t>(expert);
        best_scores[rank] = score;
        best_corrected[rank] = corrected;
        break;
      }
    }
  }

  float score_sum = 0.0f;
  for (size_t rank = 0; rank < TopK; ++rank) {
    score_sum += best_scores[rank];
  }
  score_sum = fmaxf(score_sum, 1.0e-12f);
  const size_t out_offset = row * TopK;
  for (size_t rank = 0; rank < TopK; ++rank) {
    topk_indices[out_offset + rank] = best_indices[rank];
    topk_scores[out_offset + rank] = best_scores[rank];
    topk_weights[out_offset + rank] =
        best_scores[rank] / score_sum * routed_scaling_factor;
  }
}

__global__ void ds4_flash_router_shortlist_kernel(
    float *approximate_logits, const float *correction_bias, size_t rows) {
  const size_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const size_t expert = threadIdx.x;
  const float logit = approximate_logits[row * kDs4FlashExperts + expert];
  const float score = isfinite(logit) ? sqrt_softplus_f32(logit) : 0.0f;
  const float raw_corrected = score + correction_bias[expert];
  float corrected[1] = {isfinite(logit) && isfinite(raw_corrected)
                            ? raw_corrected
                            : -CUDART_INF_F};
  uint32_t expert_id[1] = {static_cast<uint32_t>(expert)};
  using BlockSort =
      cub::BlockRadixSort<float, kBlock, 1, uint32_t>;
  __shared__ typename BlockSort::TempStorage sort_scratch;
  BlockSort(sort_scratch).SortDescending(corrected, expert_id);
  if (expert < kDs4FlashHybridCandidates) {
    uint32_t *candidate_indices =
        reinterpret_cast<uint32_t *>(approximate_logits +
                                     row * kDs4FlashExperts);
    candidate_indices[expert] = expert_id[0];
  }
}

__global__ void ds4_flash_router_refine_candidate_scores_kernel(
    const uint16_t *hidden, const uint16_t *router_weight,
    float *approximate_logits, size_t rows) {
  __shared__ float scratch[kBlock];
  const size_t candidate = blockIdx.x;
  const size_t row = blockIdx.y;
  if (row >= rows || candidate >= kDs4FlashHybridCandidates) {
    return;
  }
  const size_t tid = threadIdx.x;
  const float *row_scratch =
      approximate_logits + row * kDs4FlashExperts;
  const uint32_t *candidate_indices =
      reinterpret_cast<const uint32_t *>(row_scratch);
  const uint32_t expert = candidate_indices[candidate];
  const uint16_t *row_hidden = hidden + row * kDs4FlashHiddenDim;
  const uint16_t *weight_row =
      router_weight + static_cast<size_t>(expert) * kDs4FlashHiddenDim;
  float partial = 0.0f;
  for (size_t col = tid; col < kDs4FlashHiddenDim; col += blockDim.x) {
    partial = fmaf(bf16_to_f32(row_hidden[col]), bf16_to_f32(weight_row[col]),
                   partial);
  }
  scratch[tid] = partial;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < static_cast<size_t>(stride)) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    approximate_logits[row * kDs4FlashExperts +
                       kDs4FlashHybridCandidates + candidate] =
        sqrt_softplus_f32(scratch[0]);
  }
}

__global__ void ds4_flash_router_select_refined_kernel(
    const float *approximate_logits, const float *correction_bias,
    uint32_t *topk_indices, float *topk_scores, float *topk_weights,
    size_t rows) {
  const size_t row = blockIdx.x;
  if (row >= rows || threadIdx.x != 0) {
    return;
  }

  float best_scores[kDs4FlashTopK];
  float best_corrected[kDs4FlashTopK];
  uint32_t best_indices[kDs4FlashTopK];
  for (size_t rank = 0; rank < kDs4FlashTopK; ++rank) {
    best_scores[rank] = 0.0f;
    best_corrected[rank] = -CUDART_INF_F;
    best_indices[rank] = 0;
  }
  const float *row_scratch =
      approximate_logits + row * kDs4FlashExperts;
  const uint32_t *candidate_indices =
      reinterpret_cast<const uint32_t *>(row_scratch);
  const float *candidate_scores =
      row_scratch + kDs4FlashHybridCandidates;
  for (size_t candidate = 0; candidate < kDs4FlashHybridCandidates;
       ++candidate) {
    const uint32_t expert = candidate_indices[candidate];
    const float raw_score = candidate_scores[candidate];
    const float score = isfinite(raw_score) ? raw_score : 0.0f;
    const float raw_corrected = score + correction_bias[expert];
    const float corrected = isfinite(raw_score) && isfinite(raw_corrected)
                                ? raw_corrected
                                : -CUDART_INF_F;
    for (size_t rank = 0; rank < kDs4FlashTopK; ++rank) {
      if (corrected > best_corrected[rank] ||
          (corrected == best_corrected[rank] && expert < best_indices[rank])) {
        for (size_t shift = kDs4FlashTopK - 1; shift > rank; --shift) {
          best_indices[shift] = best_indices[shift - 1];
          best_scores[shift] = best_scores[shift - 1];
          best_corrected[shift] = best_corrected[shift - 1];
        }
        best_indices[rank] = expert;
        best_scores[rank] = score;
        best_corrected[rank] = corrected;
        break;
      }
    }
  }

  float score_sum = 0.0f;
  for (size_t rank = 0; rank < kDs4FlashTopK; ++rank) {
    score_sum += best_scores[rank];
  }
  score_sum = fmaxf(score_sum, 1.0e-12f);
  const size_t output = row * kDs4FlashTopK;
  for (size_t rank = 0; rank < kDs4FlashTopK; ++rank) {
    topk_indices[output + rank] = best_indices[rank];
    topk_scores[output + rank] = best_scores[rank];
    topk_weights[output + rank] =
        best_scores[rank] / score_sum * kDs4FlashRoutedScalingFactor;
  }
}

__global__ void ds4_flash_hash_router_bf16_score_kernel(
    const uint16_t *hidden, const uint16_t *router_weight,
    const int64_t *token_to_experts, const int64_t *token_ids,
    uint32_t *topk_indices, float *topk_scores, size_t rows) {
  __shared__ float scratch[kBlock];
  const size_t rank = blockIdx.x;
  const size_t row = blockIdx.y;
  if (row >= rows || rank >= kDs4FlashTopK) {
    return;
  }
  const size_t tid = threadIdx.x;
  const int64_t token_id = token_ids[row];
  const bool valid_token =
      token_id >= 0 && token_id < static_cast<int64_t>(kDs4FlashVocabSize);
  const int64_t expert_id =
      valid_token ? token_to_experts[token_id * kDs4FlashTopK + rank]
                  : int64_t{-1};
  const bool valid_expert =
      expert_id >= 0 && expert_id < static_cast<int64_t>(kDs4FlashExperts);
  float partial = 0.0f;
  if (valid_expert) {
    const uint16_t *row_hidden = hidden + row * kDs4FlashHiddenDim;
    const uint16_t *weight_row = router_weight + expert_id * kDs4FlashHiddenDim;
    for (size_t col = tid; col < kDs4FlashHiddenDim; col += blockDim.x) {
      partial = fmaf(bf16_to_f32(row_hidden[col]), bf16_to_f32(weight_row[col]),
                     partial);
    }
  }
  scratch[tid] = partial;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < static_cast<size_t>(stride)) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    const size_t output = row * kDs4FlashTopK + rank;
    const float raw_score = valid_expert ? sqrt_softplus_f32(scratch[0]) : 0.0f;
    topk_indices[output] = valid_expert ? static_cast<uint32_t>(expert_id) : 0;
    topk_scores[output] = isfinite(raw_score) ? raw_score : 0.0f;
  }
}

__global__ void ds4_flash_router_finalize_kernel(uint32_t *topk_indices,
                                                 float *topk_scores,
                                                 float *topk_weights,
                                                 size_t rows) {
  const size_t row = blockIdx.x;
  if (row >= rows || threadIdx.x != 0) {
    return;
  }
  const size_t offset = row * kDs4FlashTopK;
  float score_sum = 0.0f;
  for (size_t rank = 0; rank < kDs4FlashTopK; ++rank) {
    const float score = topk_scores[offset + rank];
    score_sum += isfinite(score) ? score : 0.0f;
  }
  score_sum = fmaxf(score_sum, 1.0e-12f);
  for (size_t rank = 0; rank < kDs4FlashTopK; ++rank) {
    topk_indices[offset + rank] &= ~kRouterTopKLockBit;
    const float score = topk_scores[offset + rank];
    topk_scores[offset + rank] = isfinite(score) ? score : 0.0f;
    topk_weights[offset + rank] =
        topk_scores[offset + rank] / score_sum * kDs4FlashRoutedScalingFactor;
  }
}

__global__ void ds4_pro_hash_router_bf16_score_kernel(
    const uint16_t *hidden, const uint16_t *router_weight,
    const int64_t *token_to_experts, const int64_t *token_ids,
    uint32_t *topk_indices, float *topk_scores, size_t rows) {
  __shared__ float scratch[kBlock];
  const size_t rank = blockIdx.x;
  const size_t row = blockIdx.y;
  if (row >= rows || rank >= kDs4ProTopK) {
    return;
  }
  const size_t tid = threadIdx.x;
  const int64_t token_id = token_ids[row];
  const bool valid_token =
      token_id >= 0 && token_id < static_cast<int64_t>(kDs4ProVocabSize);
  const int64_t expert_id =
      valid_token ? token_to_experts[token_id * kDs4ProTopK + rank]
                  : int64_t{-1};
  const bool valid_expert =
      expert_id >= 0 && expert_id < static_cast<int64_t>(kDs4ProExperts);
  float partial = 0.0f;
  if (valid_expert) {
    const uint16_t *row_hidden = hidden + row * kDs4ProHiddenDim;
    const uint16_t *weight_row = router_weight + expert_id * kDs4ProHiddenDim;
    for (size_t col = tid; col < kDs4ProHiddenDim; col += blockDim.x) {
      partial = fmaf(bf16_to_f32(row_hidden[col]), bf16_to_f32(weight_row[col]),
                     partial);
    }
  }
  scratch[tid] = partial;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < static_cast<size_t>(stride)) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    const size_t output = row * kDs4ProTopK + rank;
    const float raw_score = valid_expert ? sqrt_softplus_f32(scratch[0]) : 0.0f;
    topk_indices[output] = valid_expert ? static_cast<uint32_t>(expert_id) : 0;
    topk_scores[output] = isfinite(raw_score) ? raw_score : 0.0f;
  }
}

__global__ void ds4_pro_router_finalize_kernel(uint32_t *topk_indices,
                                               float *topk_scores,
                                               float *topk_weights,
                                               size_t rows) {
  const size_t row = blockIdx.x;
  if (row >= rows || threadIdx.x != 0) {
    return;
  }
  const size_t offset = row * kDs4ProTopK;
  float score_sum = 0.0f;
  for (size_t rank = 0; rank < kDs4ProTopK; ++rank) {
    const float score = topk_scores[offset + rank];
    score_sum += isfinite(score) ? score : 0.0f;
  }
  score_sum = fmaxf(score_sum, 1.0e-12f);
  for (size_t rank = 0; rank < kDs4ProTopK; ++rank) {
    topk_indices[offset + rank] &= ~kRouterTopKLockBit;
    const float score = topk_scores[offset + rank];
    topk_scores[offset + rank] = isfinite(score) ? score : 0.0f;
    topk_weights[offset + rank] =
        topk_scores[offset + rank] / score_sum * kDs4ProRoutedScalingFactor;
  }
}

__global__ void router_topk_f32_kernel(const float* hidden, const float* router_weight,
                                       const float* correction_bias, uint32_t* topk_indices,
                                       float* topk_scores, float* topk_weights, size_t rows,
                                       size_t hidden_dim, size_t experts, size_t top_k) {
  const size_t row = blockIdx.x;
  if (row >= rows || threadIdx.x != 0) {
    return;
  }

  float best_scores[kMaxRouterTopK];
  float best_corrected[kMaxRouterTopK];
  uint32_t best_indices[kMaxRouterTopK];
  for (size_t rank = 0; rank < top_k; ++rank) {
    best_scores[rank] = 0.0f;
    best_corrected[rank] = -CUDART_INF_F;
    best_indices[rank] = 0;
  }

  const float* row_hidden = hidden + row * hidden_dim;
  for (size_t expert = 0; expert < experts; ++expert) {
    const float* weight_row = router_weight + expert * hidden_dim;
    float logit = 0.0f;
    for (size_t col = 0; col < hidden_dim; ++col) {
      logit += row_hidden[col] * weight_row[col];
    }
    const float raw_score = sigmoid_f32(logit);
    const float score = isfinite(raw_score) ? raw_score : 0.0f;
    const float raw_corrected = score + correction_bias[expert];
    const float corrected = isfinite(raw_score) && isfinite(raw_corrected) ? raw_corrected
                                                                            : -CUDART_INF_F;
    for (size_t rank = 0; rank < top_k; ++rank) {
      if (corrected > best_corrected[rank]) {
        for (size_t shift = top_k - 1; shift > rank; --shift) {
          best_corrected[shift] = best_corrected[shift - 1];
          best_scores[shift] = best_scores[shift - 1];
          best_indices[shift] = best_indices[shift - 1];
        }
        best_corrected[rank] = corrected;
        best_scores[rank] = score;
        best_indices[rank] = static_cast<uint32_t>(expert);
        break;
      }
    }
  }

  float score_sum = 0.0f;
  for (size_t rank = 0; rank < top_k; ++rank) {
    score_sum += best_scores[rank];
  }
  score_sum = fmaxf(score_sum, 1.0e-12f);

  const size_t out_offset = row * top_k;
  for (size_t rank = 0; rank < top_k; ++rank) {
    topk_indices[out_offset + rank] = best_indices[rank];
    topk_scores[out_offset + rank] = best_scores[rank];
    topk_weights[out_offset + rank] = best_scores[rank] / score_sum * kGlm52RoutedScalingFactor;
  }
}

__global__ void router_topk_bf16_init_kernel(uint32_t* topk_indices, float* topk_scores,
                                             float* topk_weights, size_t rows, size_t top_k) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = rows * top_k;
  if (idx >= total) {
    return;
  }
  topk_indices[idx] = 0;
  topk_scores[idx] = 0.0f;
  topk_weights[idx] = -CUDART_INF_F;
}

__global__ void router_topk_bf16_score_kernel(const uint16_t* hidden,
                                              const uint16_t* router_weight,
                                              const float* correction_bias,
                                              uint32_t* topk_indices, float* topk_scores,
                                              float* topk_weights, size_t rows,
                                              size_t hidden_dim, size_t experts,
                                              size_t top_k) {
  __shared__ float scratch[kBlock];
  const size_t expert = blockIdx.x;
  const size_t row = blockIdx.y;
  if (row >= rows || expert >= experts) {
    return;
  }
  const size_t tid = threadIdx.x;
  const uint16_t* row_hidden = hidden + row * hidden_dim;
  const uint16_t* weight_row = router_weight + expert * hidden_dim;
  float partial = 0.0f;
  for (size_t col = tid; col < hidden_dim; col += blockDim.x) {
    partial = fmaf(bf16_to_f32(row_hidden[col]), bf16_to_f32(weight_row[col]), partial);
  }
  scratch[tid] = partial;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < static_cast<size_t>(stride)) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }

  if (tid != 0) {
    return;
  }

  const float raw_score = sigmoid_f32(scratch[0]);
  const float score = isfinite(raw_score) ? raw_score : 0.0f;
  const float raw_corrected = score + correction_bias[expert];
  const float corrected = isfinite(raw_score) && isfinite(raw_corrected) ? raw_corrected
                                                                          : -CUDART_INF_F;
  const size_t out_offset = row * top_k;
  uint32_t* lock_word = topk_indices + out_offset;

  while (true) {
    const uint32_t current = atomicAdd(lock_word, 0);
    if ((current & kRouterTopKLockBit) != 0) {
      continue;
    }
    if (atomicCAS(lock_word, current, current | kRouterTopKLockBit) == current) {
      break;
    }
  }

  for (size_t rank = 0; rank < top_k; ++rank) {
    uint32_t current_index = topk_indices[out_offset + rank];
    if (rank == 0) {
      current_index &= ~kRouterTopKLockBit;
    }
    const float current_corrected = topk_weights[out_offset + rank];
    if (corrected > current_corrected ||
        (corrected == current_corrected && static_cast<uint32_t>(expert) < current_index)) {
      for (size_t shift = top_k - 1; shift > rank; --shift) {
        uint32_t shifted_index = topk_indices[out_offset + shift - 1];
        if (shift == 1) {
          shifted_index &= ~kRouterTopKLockBit;
        }
        topk_indices[out_offset + shift] = shifted_index;
        topk_scores[out_offset + shift] = topk_scores[out_offset + shift - 1];
        topk_weights[out_offset + shift] = topk_weights[out_offset + shift - 1];
      }
      topk_indices[out_offset + rank] =
          static_cast<uint32_t>(expert) | (rank == 0 ? kRouterTopKLockBit : 0);
      topk_scores[out_offset + rank] = score;
      topk_weights[out_offset + rank] = corrected;
      break;
    }
  }
  __threadfence();
  atomicAnd(lock_word, ~kRouterTopKLockBit);
}

__global__ void router_topk_bf16_finalize_kernel(uint32_t* topk_indices, float* topk_scores,
                                                 float* topk_weights, size_t rows,
                                                 size_t top_k) {
  const size_t row = blockIdx.x;
  if (row >= rows || threadIdx.x != 0) {
    return;
  }
  const size_t out_offset = row * top_k;
  float score_sum = 0.0f;
  for (size_t rank = 0; rank < top_k; ++rank) {
    const float score = topk_scores[out_offset + rank];
    score_sum += isfinite(score) ? score : 0.0f;
  }
  score_sum = fmaxf(score_sum, 1.0e-12f);

  for (size_t rank = 0; rank < top_k; ++rank) {
    topk_indices[out_offset + rank] &= ~kRouterTopKLockBit;
    const float score = topk_scores[out_offset + rank];
    topk_scores[out_offset + rank] = isfinite(score) ? score : 0.0f;
    topk_weights[out_offset + rank] =
        topk_scores[out_offset + rank] / score_sum * kGlm52RoutedScalingFactor;
  }
}

__global__ void router_topk_bf16_cub_fill_indices_offsets_kernel(uint32_t* indices,
                                                                 int* segment_offsets,
                                                                 size_t rows, size_t experts) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = rows * experts;
  if (idx < total) {
    indices[idx] = static_cast<uint32_t>(idx % experts);
  }
  if (idx <= rows) {
    segment_offsets[idx] = static_cast<int>(idx * experts);
  }
}

__global__ void router_topk_bf16_cub_score_kernel(const uint16_t* hidden,
                                                  const uint16_t* router_weight,
                                                  const float* correction_bias,
                                                  float* corrected_scores, size_t rows,
                                                  size_t hidden_dim, size_t experts) {
  __shared__ float scratch[kBlock];
  const size_t expert = blockIdx.x;
  const size_t row = blockIdx.y;
  if (row >= rows || expert >= experts) {
    return;
  }
  const size_t tid = threadIdx.x;
  const uint16_t* row_hidden = hidden + row * hidden_dim;
  const uint16_t* weight_row = router_weight + expert * hidden_dim;
  float partial = 0.0f;
  for (size_t col = tid; col < hidden_dim; col += blockDim.x) {
    partial = fmaf(bf16_to_f32(row_hidden[col]), bf16_to_f32(weight_row[col]), partial);
  }
  scratch[tid] = partial;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < static_cast<size_t>(stride)) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    const float raw_score = sigmoid_f32(scratch[0]);
    const float score = isfinite(raw_score) ? raw_score : 0.0f;
    const float raw_corrected = score + correction_bias[expert];
    corrected_scores[row * experts + expert] =
        isfinite(raw_score) && isfinite(raw_corrected) ? raw_corrected : -CUDART_INF_F;
  }
}

__global__ void router_topk_bf16_cub_finalize_kernel(
    const float* sorted_corrected_scores, const uint32_t* sorted_indices,
    const float* correction_bias, uint32_t* topk_indices, float* topk_scores,
    float* topk_weights, size_t rows, size_t experts, size_t top_k) {
  const size_t row = blockIdx.x;
  if (row >= rows || threadIdx.x != 0) {
    return;
  }
  const size_t sorted_offset = row * experts;
  const size_t out_offset = row * top_k;
  float score_sum = 0.0f;
  for (size_t rank = 0; rank < top_k; ++rank) {
    const uint32_t expert = sorted_indices[sorted_offset + rank];
    const float raw_score =
        sorted_corrected_scores[sorted_offset + rank] - correction_bias[expert];
    const float score = isfinite(raw_score) ? raw_score : 0.0f;
    topk_indices[out_offset + rank] = expert;
    topk_scores[out_offset + rank] = score;
    score_sum += score;
  }
  score_sum = fmaxf(score_sum, 1.0e-12f);
  for (size_t rank = 0; rank < top_k; ++rank) {
    topk_weights[out_offset + rank] =
        topk_scores[out_offset + rank] / score_sum * kGlm52RoutedScalingFactor;
  }
}

ds41rt_status_t validate_router_topk_args(const float* hidden, const float* router_weight,
                                         const float* correction_bias,
                                         const uint32_t* topk_indices, const float* topk_scores,
                                         const float* topk_weights, size_t rows,
                                         size_t hidden_dim, size_t experts, size_t top_k) {
  if (hidden == nullptr || router_weight == nullptr || correction_bias == nullptr ||
      topk_indices == nullptr || topk_scores == nullptr || topk_weights == nullptr) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  if (rows == 0 || hidden_dim == 0 || experts == 0 || top_k == 0 || top_k > experts ||
      top_k > kMaxRouterTopK) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  size_t ignored = 0;
  if (!checked_mul(rows, hidden_dim, &ignored) ||
      !checked_mul(experts, hidden_dim, &ignored) ||
      !checked_mul(rows, top_k, &ignored) ||
      rows > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  return DS41RT_STATUS_OK;
}

ds41rt_status_t validate_router_topk_bf16_args(
    const uint16_t* hidden, const uint16_t* router_weight, const float* correction_bias,
    const uint32_t* topk_indices, const float* topk_scores, const float* topk_weights,
    size_t rows, size_t hidden_dim, size_t experts, size_t top_k) {
  if (hidden == nullptr || router_weight == nullptr || correction_bias == nullptr ||
      topk_indices == nullptr || topk_scores == nullptr || topk_weights == nullptr) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  if (rows == 0 || hidden_dim == 0 || experts == 0 || top_k == 0 || top_k > experts ||
      top_k > kMaxRouterTopK ||
      experts > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  size_t ignored = 0;
  if (!checked_mul(rows, hidden_dim, &ignored) ||
      !checked_mul(experts, hidden_dim, &ignored) ||
      !checked_mul(rows, top_k, &ignored) ||
      rows > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  return DS41RT_STATUS_OK;
}

ds41rt_status_t validate_router_topk_bf16_cub_args(
    const uint16_t* hidden, const uint16_t* router_weight, const float* correction_bias,
    float* corrected_scores, float* sorted_corrected_scores, uint32_t* unsorted_indices,
    uint32_t* sorted_indices, int* segment_offsets, uint32_t* topk_indices, float* topk_scores,
    float* topk_weights, void* cub_temp_storage, size_t cub_temp_storage_bytes, size_t rows,
    size_t hidden_dim, size_t experts, size_t top_k) {
  const ds41rt_status_t valid = validate_router_topk_bf16_args(
      hidden, router_weight, correction_bias, topk_indices, topk_scores, topk_weights, rows,
      hidden_dim, experts, top_k);
  if (valid != DS41RT_STATUS_OK) {
    return valid;
  }
  if (corrected_scores == nullptr || sorted_corrected_scores == nullptr ||
      unsorted_indices == nullptr || sorted_indices == nullptr || segment_offsets == nullptr ||
      cub_temp_storage == nullptr || cub_temp_storage_bytes == 0) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  size_t score_values = 0;
  if (!checked_mul(rows, experts, &score_values) ||
      score_values > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      rows > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  return DS41RT_STATUS_OK;
}

ds41rt_status_t validate_bf16_graph_router_topk_buffers(
    ds41rt_device_buffer_t hidden, ds41rt_device_buffer_t router_weight,
    ds41rt_device_buffer_t correction_bias, ds41rt_device_buffer_t topk_indices,
    ds41rt_device_buffer_t topk_scores, ds41rt_device_buffer_t topk_weights, size_t rows,
    size_t hidden_dim, size_t experts, size_t top_k) {
  const ds41rt_status_t valid = validate_router_topk_bf16_args(
      static_cast<const uint16_t*>(hidden.ptr), static_cast<const uint16_t*>(router_weight.ptr),
      static_cast<const float*>(correction_bias.ptr), static_cast<const uint32_t*>(topk_indices.ptr),
      static_cast<const float*>(topk_scores.ptr), static_cast<const float*>(topk_weights.ptr),
      rows, hidden_dim, experts, top_k);
  if (valid != DS41RT_STATUS_OK) {
    return valid;
  }
  size_t hidden_values = 0;
  size_t weight_values = 0;
  size_t topk_values = 0;
  if (!checked_mul(rows, hidden_dim, &hidden_values) ||
      !checked_mul(experts, hidden_dim, &weight_values) ||
      !checked_mul(rows, top_k, &topk_values)) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  size_t hidden_bytes = 0;
  size_t weight_bytes = 0;
  size_t bias_bytes = 0;
  size_t index_bytes = 0;
  size_t score_bytes = 0;
  if (!checked_mul(hidden_values, sizeof(uint16_t), &hidden_bytes) ||
      !checked_mul(weight_values, sizeof(uint16_t), &weight_bytes) ||
      !checked_mul(experts, sizeof(float), &bias_bytes) ||
      !checked_mul(topk_values, sizeof(uint32_t), &index_bytes) ||
      !checked_mul(topk_values, sizeof(float), &score_bytes)) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  if (hidden.bytes < hidden_bytes || router_weight.bytes < weight_bytes ||
      correction_bias.bytes < bias_bytes || topk_indices.bytes < index_bytes ||
      topk_scores.bytes < score_bytes || topk_weights.bytes < score_bytes) {
    return DS41RT_STATUS_BUFFER_TOO_SMALL;
  }
  return DS41RT_STATUS_OK;
}

}  // namespace

extern "C" ds41rt_status_t ds41rt_cuda_graph_update_router_topk_bf16_node(
    void* cuda_graph, void* cuda_graph_exec, size_t kernel_node_index,
    ds41rt_device_buffer_t hidden, ds41rt_device_buffer_t router_weight,
    ds41rt_device_buffer_t correction_bias, ds41rt_device_buffer_t topk_indices,
    ds41rt_device_buffer_t topk_scores, ds41rt_device_buffer_t topk_weights, size_t rows,
    size_t hidden_dim, size_t experts, size_t top_k) {
  if (cuda_graph == nullptr || cuda_graph_exec == nullptr) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  const ds41rt_status_t valid = validate_bf16_graph_router_topk_buffers(
      hidden, router_weight, correction_bias, topk_indices, topk_scores, topk_weights, rows,
      hidden_dim, experts, top_k);
  if (valid != DS41RT_STATUS_OK) {
    return valid;
  }

  const uint16_t* hidden_ptr = static_cast<const uint16_t*>(hidden.ptr);
  const uint16_t* router_weight_ptr = static_cast<const uint16_t*>(router_weight.ptr);
  const float* correction_bias_ptr = static_cast<const float*>(correction_bias.ptr);
  uint32_t* topk_indices_ptr = static_cast<uint32_t*>(topk_indices.ptr);
  float* topk_scores_ptr = static_cast<float*>(topk_scores.ptr);
  float* topk_weights_ptr = static_cast<float*>(topk_weights.ptr);
  void* init_args[] = {
      &topk_indices_ptr,
      &topk_scores_ptr,
      &topk_weights_ptr,
      &rows,
      &top_k,
  };
  void* score_args[] = {
      &hidden_ptr,
      &router_weight_ptr,
      &correction_bias_ptr,
      &topk_indices_ptr,
      &topk_scores_ptr,
      &topk_weights_ptr,
      &rows,
      &hidden_dim,
      &experts,
      &top_k,
  };
  void* finalize_args[] = {
      &topk_indices_ptr,
      &topk_scores_ptr,
      &topk_weights_ptr,
      &rows,
      &top_k,
  };

  const size_t topk_values = rows * top_k;
  const size_t init_blocks = (topk_values - 1) / kBlock + 1;
  if (init_blocks > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }

  auto update_kernel_node = [&](size_t node_index, void* expected_func,
                                cudaKernelNodeParams* params) -> ds41rt_status_t {
    cudaGraphNode_t node = nullptr;
    const ds41rt_status_t node_status =
        find_kernel_node_by_index(cuda_graph, node_index, &node);
    if (node_status != DS41RT_STATUS_OK) {
      return node_status;
    }
    cudaKernelNodeParams existing = {};
    cudaError_t err = cudaGraphKernelNodeGetParams(node, &existing);
    if (err != cudaSuccess) {
      return DS41RT_STATUS_INTERNAL_ERROR;
    }
    if (existing.func != expected_func) {
      return DS41RT_STATUS_INVALID_ARGUMENT;
    }
    err = cudaGraphKernelNodeSetParams(node, params);
    if (err != cudaSuccess) {
      return DS41RT_STATUS_INTERNAL_ERROR;
    }
    err = cudaGraphExecKernelNodeSetParams(reinterpret_cast<cudaGraphExec_t>(cuda_graph_exec),
                                           node, params);
    if (err != cudaSuccess) {
      return DS41RT_STATUS_INTERNAL_ERROR;
    }
    return DS41RT_STATUS_OK;
  };

  cudaKernelNodeParams init_params = {};
  init_params.func = reinterpret_cast<void*>(router_topk_bf16_init_kernel);
  init_params.gridDim = dim3(static_cast<unsigned int>(init_blocks), 1, 1);
  init_params.blockDim = dim3(kBlock, 1, 1);
  init_params.sharedMemBytes = 0;
  init_params.kernelParams = init_args;
  init_params.extra = nullptr;

  cudaKernelNodeParams score_params = {};
  score_params.func = reinterpret_cast<void*>(router_topk_bf16_score_kernel);
  score_params.gridDim =
      dim3(static_cast<unsigned int>(experts), static_cast<unsigned int>(rows), 1);
  score_params.blockDim = dim3(kBlock, 1, 1);
  score_params.sharedMemBytes = 0;
  score_params.kernelParams = score_args;
  score_params.extra = nullptr;

  cudaKernelNodeParams finalize_params = {};
  finalize_params.func = reinterpret_cast<void*>(router_topk_bf16_finalize_kernel);
  finalize_params.gridDim = dim3(static_cast<unsigned int>(rows), 1, 1);
  finalize_params.blockDim = dim3(1, 1, 1);
  finalize_params.sharedMemBytes = 0;
  finalize_params.kernelParams = finalize_args;
  finalize_params.extra = nullptr;

  ds41rt_status_t status = update_kernel_node(
      kernel_node_index, reinterpret_cast<void*>(router_topk_bf16_init_kernel), &init_params);
  if (status != DS41RT_STATUS_OK) {
    return status;
  }
  status = update_kernel_node(kernel_node_index + 1,
                              reinterpret_cast<void*>(router_topk_bf16_score_kernel),
                              &score_params);
  if (status != DS41RT_STATUS_OK) {
    return status;
  }
  status = update_kernel_node(kernel_node_index + 2,
                              reinterpret_cast<void*>(router_topk_bf16_finalize_kernel),
                              &finalize_params);
  if (status != DS41RT_STATUS_OK) {
    return status;
  }
  return DS41RT_STATUS_OK;
}

extern "C" ds41rt_status_t ds41rt_cuda_router_topk_f32_async(
    const float* hidden, const float* router_weight, const float* correction_bias,
    uint32_t* topk_indices, float* topk_scores, float* topk_weights, size_t rows,
    size_t hidden_dim, size_t experts, size_t top_k, void* cuda_stream) {
  const ds41rt_status_t valid =
      validate_router_topk_args(hidden, router_weight, correction_bias, topk_indices, topk_scores,
                                topk_weights, rows, hidden_dim, experts, top_k);
  if (valid != DS41RT_STATUS_OK) {
    return valid;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  router_topk_f32_kernel<<<static_cast<int>(rows), 1, 0, stream>>>(
      hidden, router_weight, correction_bias, topk_indices, topk_scores, topk_weights, rows,
      hidden_dim, experts, top_k);
  return status_from_cuda(cudaGetLastError());
}

extern "C" ds41rt_status_t ds41rt_cuda_router_topk_f32(
    const float* hidden, const float* router_weight, const float* correction_bias,
    uint32_t* topk_indices, float* topk_scores, float* topk_weights, size_t rows,
    size_t hidden_dim, size_t experts, size_t top_k) {
  const ds41rt_status_t status =
      ds41rt_cuda_router_topk_f32_async(hidden, router_weight, correction_bias, topk_indices,
                                       topk_scores, topk_weights, rows, hidden_dim, experts, top_k,
                                       nullptr);
  if (status != DS41RT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" ds41rt_status_t ds41rt_cuda_router_topk_bf16_async(
    const uint16_t* hidden, const uint16_t* router_weight, const float* correction_bias,
    uint32_t* topk_indices, float* topk_scores, float* topk_weights, size_t rows,
    size_t hidden_dim, size_t experts, size_t top_k, void* cuda_stream) {
  const ds41rt_status_t valid = validate_router_topk_bf16_args(
      hidden, router_weight, correction_bias, topk_indices, topk_scores, topk_weights, rows,
      hidden_dim, experts, top_k);
  if (valid != DS41RT_STATUS_OK) {
    return valid;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const size_t topk_values = rows * top_k;
  const size_t init_blocks = (topk_values - 1) / kBlock + 1;
  if (init_blocks > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  router_topk_bf16_init_kernel<<<static_cast<unsigned int>(init_blocks), kBlock, 0, stream>>>(
      topk_indices, topk_scores, topk_weights, rows, top_k);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return status_from_cuda(err);
  }
  router_topk_bf16_score_kernel<<<
      dim3(static_cast<unsigned int>(experts), static_cast<unsigned int>(rows), 1), kBlock, 0,
      stream>>>(hidden, router_weight, correction_bias, topk_indices, topk_scores, topk_weights,
                rows, hidden_dim, experts, top_k);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    return status_from_cuda(err);
  }
  router_topk_bf16_finalize_kernel<<<static_cast<unsigned int>(rows), 1, 0, stream>>>(
      topk_indices, topk_scores, topk_weights, rows, top_k);
  return status_from_cuda(cudaGetLastError());
}

extern "C" ds41rt_status_t ds41rt_cuda_router_topk_bf16(
    const uint16_t* hidden, const uint16_t* router_weight, const float* correction_bias,
    uint32_t* topk_indices, float* topk_scores, float* topk_weights, size_t rows,
    size_t hidden_dim, size_t experts, size_t top_k) {
  const ds41rt_status_t status =
      ds41rt_cuda_router_topk_bf16_async(hidden, router_weight, correction_bias, topk_indices,
                                        topk_scores, topk_weights, rows, hidden_dim, experts,
                                        top_k, nullptr);
  if (status != DS41RT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" ds41rt_status_t ds41rt_cuda_ds4_flash_router_topk_bf16_async(
    const uint16_t *hidden, const uint16_t *router_weight,
    const float *correction_bias, const int64_t *token_to_experts,
    const int64_t *token_ids, uint32_t *topk_indices, float *topk_scores,
    float *topk_weights, size_t rows, int hash_routing, void *cuda_stream) {
  if (hidden == nullptr || router_weight == nullptr ||
      topk_indices == nullptr || topk_scores == nullptr ||
      topk_weights == nullptr || rows == 0 ||
      rows > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  if ((hash_routing == 0 && correction_bias == nullptr) ||
      (hash_routing != 0 &&
       (token_to_experts == nullptr || token_ids == nullptr))) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  size_t ignored = 0;
  if (!checked_mul(rows, kDs4FlashHiddenDim, &ignored) ||
      !checked_mul(rows, kDs4FlashTopK + kDs4FlashExperts, &ignored)) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  if (hash_routing != 0) {
    ds4_flash_hash_router_bf16_score_kernel<<<
        dim3(kDs4FlashTopK, static_cast<unsigned int>(rows), 1), kBlock, 0,
        stream>>>(hidden, router_weight, token_to_experts, token_ids,
                  topk_indices, topk_scores, rows);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      return status_from_cuda(err);
    }
    ds4_flash_router_finalize_kernel<<<static_cast<unsigned int>(rows), 1, 0,
                                       stream>>>(topk_indices, topk_scores,
                                                 topk_weights, rows);
  } else {
    float *raw_scores = topk_scores + rows * kDs4FlashTopK;
    ds4_router_bf16_score_workspace_kernel<kDs4FlashHiddenDim,
                                            kDs4FlashExperts><<<
        dim3(kDs4FlashExperts, static_cast<unsigned int>(rows), 1), kBlock, 0,
        stream>>>(hidden, router_weight, raw_scores, rows);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      return status_from_cuda(err);
    }
    ds4_router_select_workspace_kernel<kDs4FlashExperts, kDs4FlashTopK>
        <<<static_cast<unsigned int>(rows), 1, 0, stream>>>(
            raw_scores, correction_bias, topk_indices, topk_scores,
            topk_weights, rows, kDs4FlashRoutedScalingFactor);
  }
  return status_from_cuda(cudaGetLastError());
}

extern "C" ds41rt_status_t ds41rt_cuda_ds4_flash_router_topk_bf16(
    const uint16_t *hidden, const uint16_t *router_weight,
    const float *correction_bias, const int64_t *token_to_experts,
    const int64_t *token_ids, uint32_t *topk_indices, float *topk_scores,
    float *topk_weights, size_t rows, int hash_routing) {
  const ds41rt_status_t status = ds41rt_cuda_ds4_flash_router_topk_bf16_async(
      hidden, router_weight, correction_bias, token_to_experts, token_ids,
      topk_indices, topk_scores, topk_weights, rows, hash_routing, nullptr);
  if (status != DS41RT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" ds41rt_status_t ds41rt_cuda_ds4_flash_router_refine_topk_bf16_async(
    const uint16_t *hidden, const uint16_t *router_weight,
    const float *correction_bias, float *approximate_logits,
    uint32_t *topk_indices, float *topk_scores, float *topk_weights, size_t rows,
    void *cuda_stream) {
  if (hidden == nullptr || router_weight == nullptr ||
      correction_bias == nullptr || approximate_logits == nullptr ||
      topk_indices == nullptr || topk_scores == nullptr ||
      topk_weights == nullptr || rows == 0 ||
      rows > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  size_t ignored = 0;
  if (!checked_mul(rows, kDs4FlashHiddenDim, &ignored) ||
      !checked_mul(rows, kDs4FlashExperts, &ignored) ||
      !checked_mul(rows, kDs4FlashHybridCandidates, &ignored) ||
      !checked_mul(rows, kDs4FlashTopK, &ignored)) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  ds4_flash_router_shortlist_kernel<<<static_cast<unsigned int>(rows), kBlock,
                                      0, stream>>>(
      approximate_logits, correction_bias, rows);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return status_from_cuda(err);
  }
  ds4_flash_router_refine_candidate_scores_kernel<<<
      dim3(kDs4FlashHybridCandidates, static_cast<unsigned int>(rows), 1),
      kBlock, 0, stream>>>(hidden, router_weight, approximate_logits, rows);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    return status_from_cuda(err);
  }
  ds4_flash_router_select_refined_kernel<<<static_cast<unsigned int>(rows), 1,
                                           0, stream>>>(
      approximate_logits, correction_bias, topk_indices, topk_scores,
      topk_weights, rows);
  return status_from_cuda(cudaGetLastError());
}

extern "C" ds41rt_status_t ds41rt_cuda_ds4_pro_router_topk_bf16_async(
    const uint16_t *hidden, const uint16_t *router_weight,
    const float *correction_bias, const int64_t *token_to_experts,
    const int64_t *token_ids, uint32_t *topk_indices, float *topk_scores,
    float *topk_weights, size_t rows, int hash_routing, void *cuda_stream) {
  if (hidden == nullptr || router_weight == nullptr || topk_indices == nullptr ||
      topk_scores == nullptr || topk_weights == nullptr || rows == 0 ||
      rows > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  if ((hash_routing == 0 && correction_bias == nullptr) ||
      (hash_routing != 0 &&
       (token_to_experts == nullptr || token_ids == nullptr))) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  size_t ignored = 0;
  if (!checked_mul(rows, kDs4ProHiddenDim, &ignored) ||
      !checked_mul(rows, kDs4ProTopK + kDs4ProExperts, &ignored)) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  if (hash_routing != 0) {
    ds4_pro_hash_router_bf16_score_kernel<<<
        dim3(kDs4ProTopK, static_cast<unsigned int>(rows), 1), kBlock, 0,
        stream>>>(hidden, router_weight, token_to_experts, token_ids,
                  topk_indices, topk_scores, rows);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      return status_from_cuda(err);
    }
    ds4_pro_router_finalize_kernel<<<static_cast<unsigned int>(rows), 1, 0,
                                     stream>>>(topk_indices, topk_scores,
                                               topk_weights, rows);
  } else {
    float *raw_scores = topk_scores + rows * kDs4ProTopK;
    // Preserve the qualified reduction order for decode and ordinary small
    // prompts. Large prefill chunks use the tensor-core BF16/FP32 projection;
    // converting its logits in place avoids the one-block-per-(row, expert)
    // dot product without perturbing the latency/quality-sensitive path.
    if (rows > kDs4ProExactRouterMaxRows) {
      const ds41rt_status_t score_status =
          ds41rt_cuda_linear_bf16_f32_cublas_async(
              hidden, router_weight, raw_scores, rows, kDs4ProHiddenDim,
              kDs4ProExperts, cuda_stream);
      if (score_status != DS41RT_STATUS_OK) {
        return score_status;
      }
      const size_t values = rows * kDs4ProExperts;
      const size_t blocks = (values + kBlock - 1) / kBlock;
      if (blocks >
          static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        return DS41RT_STATUS_INVALID_ARGUMENT;
      }
      ds4_router_logits_to_scores_kernel<kDs4ProExperts>
          <<<static_cast<unsigned int>(blocks), kBlock, 0, stream>>>(
              raw_scores, rows);
    } else {
      ds4_router_bf16_score_workspace_kernel<kDs4ProHiddenDim,
                                              kDs4ProExperts><<<
          dim3(kDs4ProExperts, static_cast<unsigned int>(rows), 1), kBlock, 0,
          stream>>>(hidden, router_weight, raw_scores, rows);
    }
    const cudaError_t score_error = cudaGetLastError();
    if (score_error != cudaSuccess) {
      return status_from_cuda(score_error);
    }
    ds4_router_select_workspace_kernel<kDs4ProExperts, kDs4ProTopK>
        <<<static_cast<unsigned int>(rows), 1, 0, stream>>>(
            raw_scores, correction_bias, topk_indices, topk_scores,
            topk_weights, rows, kDs4ProRoutedScalingFactor);
  }
  return status_from_cuda(cudaGetLastError());
}

extern "C" ds41rt_status_t ds41rt_cuda_ds4_pro_router_topk_bf16(
    const uint16_t *hidden, const uint16_t *router_weight,
    const float *correction_bias, const int64_t *token_to_experts,
    const int64_t *token_ids, uint32_t *topk_indices, float *topk_scores,
    float *topk_weights, size_t rows, int hash_routing) {
  const ds41rt_status_t status = ds41rt_cuda_ds4_pro_router_topk_bf16_async(
      hidden, router_weight, correction_bias, token_to_experts, token_ids,
      topk_indices, topk_scores, topk_weights, rows, hash_routing, nullptr);
  if (status != DS41RT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" ds41rt_status_t ds41rt_cuda_router_topk_bf16_cub_async(
    const uint16_t* hidden, const uint16_t* router_weight, const float* correction_bias,
    float* corrected_scores, float* sorted_corrected_scores, uint32_t* unsorted_indices,
    uint32_t* sorted_indices, int* segment_offsets, uint32_t* topk_indices, float* topk_scores,
    float* topk_weights, void* cub_temp_storage, size_t cub_temp_storage_bytes, size_t rows,
    size_t hidden_dim, size_t experts, size_t top_k, void* cuda_stream) {
  const ds41rt_status_t valid = validate_router_topk_bf16_cub_args(
      hidden, router_weight, correction_bias, corrected_scores, sorted_corrected_scores,
      unsorted_indices, sorted_indices, segment_offsets, topk_indices, topk_scores, topk_weights,
      cub_temp_storage, cub_temp_storage_bytes, rows, hidden_dim, experts, top_k);
  if (valid != DS41RT_STATUS_OK) {
    return valid;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const size_t score_values = rows * experts;
  const size_t fill_values = std::max(score_values, rows + 1);
  const size_t fill_blocks = (fill_values - 1) / kBlock + 1;
  if (fill_blocks > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
    return DS41RT_STATUS_INVALID_ARGUMENT;
  }
  router_topk_bf16_cub_fill_indices_offsets_kernel<<<
      static_cast<unsigned int>(fill_blocks), kBlock, 0, stream>>>(unsorted_indices,
                                                                   segment_offsets, rows, experts);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return status_from_cuda(err);
  }
  router_topk_bf16_cub_score_kernel<<<
      dim3(static_cast<unsigned int>(experts), static_cast<unsigned int>(rows), 1), kBlock, 0,
      stream>>>(hidden, router_weight, correction_bias, corrected_scores, rows, hidden_dim,
                experts);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    return status_from_cuda(err);
  }
  err = cub::DeviceSegmentedRadixSort::SortPairsDescending(
      cub_temp_storage, cub_temp_storage_bytes, corrected_scores, sorted_corrected_scores,
      unsorted_indices, sorted_indices, static_cast<int>(score_values), static_cast<int>(rows),
      segment_offsets, segment_offsets + 1, 0, sizeof(float) * 8, stream);
  if (err != cudaSuccess) {
    return status_from_cuda(err);
  }
  router_topk_bf16_cub_finalize_kernel<<<static_cast<unsigned int>(rows), 1, 0, stream>>>(
      sorted_corrected_scores, sorted_indices, correction_bias, topk_indices, topk_scores,
      topk_weights, rows, experts, top_k);
  return status_from_cuda(cudaGetLastError());
}

extern "C" ds41rt_status_t ds41rt_cuda_router_topk_bf16_cub(
    const uint16_t* hidden, const uint16_t* router_weight, const float* correction_bias,
    float* corrected_scores, float* sorted_corrected_scores, uint32_t* unsorted_indices,
    uint32_t* sorted_indices, int* segment_offsets, uint32_t* topk_indices, float* topk_scores,
    float* topk_weights, void* cub_temp_storage, size_t cub_temp_storage_bytes, size_t rows,
    size_t hidden_dim, size_t experts, size_t top_k) {
  const ds41rt_status_t status = ds41rt_cuda_router_topk_bf16_cub_async(
      hidden, router_weight, correction_bias, corrected_scores, sorted_corrected_scores,
      unsorted_indices, sorted_indices, segment_offsets, topk_indices, topk_scores, topk_weights,
      cub_temp_storage, cub_temp_storage_bytes, rows, hidden_dim, experts, top_k, nullptr);
  if (status != DS41RT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}
