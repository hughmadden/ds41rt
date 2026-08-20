#!/usr/bin/env bash
set -Eeuo pipefail

tp_size=${TP_SIZE:-1}
case "${tp_size}" in
  1)
    default_gpu_ids=0
    allreduce=nccl
    ;;
  2)
    default_gpu_ids=0,1
    allreduce=b12x
    ;;
  *)
    echo "TP_SIZE must be 1 or 2; got '${tp_size}'" >&2
    exit 2
    ;;
esac
gpu_ids=${GPU_IDS:-${default_gpu_ids}}
port=${PORT:-8000}
kv_cache_dtype=${KV_CACHE_DTYPE:-nvfp4_ds_mla}

image=${K2_IMAGE:-deepseek-v4-flash-0731-exl3-k2-spark:amd64-release}
container=${CONTAINER_NAME:-ds4-k2-rtx-tp${tp_size}}
hf_cache_host=${HF_CACHE_HOST:-${HOME}/.cache/huggingface}
cache_host=${JIT_CACHE_HOST:-${HOME}/.cache/ds4-bench/k2-rtx-tp${tp_size}}
revision=dff9afc6f5fe50a890590f7b6d5339ceaf5ba51e
model_rel=hub/models--wrldsuksgo2mars--DeepSeek-V4-Flash-0731-EXL3-K2-calibrated-v0/snapshots/${revision}

test -d "${hf_cache_host}/${model_rel}" || {
  echo "missing K2 model snapshot: ${hf_cache_host}/${model_rel}" >&2
  exit 2
}
mkdir -p "${cache_host}"

if docker container inspect "${container}" >/dev/null 2>&1; then
  docker rm -f "${container}" >/dev/null
fi

docker run -d \
  --name "${container}" \
  --gpus all \
  --network host \
  --ipc host \
  --shm-size 32g \
  --ulimit memlock=-1:-1 \
  --ulimit nofile=1048576:1048576 \
  --entrypoint /opt/recipe/scripts/entrypoint.sh \
  -v "${hf_cache_host}:/models/huggingface:ro" \
  -v "${cache_host}:/cache" \
  -e CUDA_VISIBLE_DEVICES="${gpu_ids}" \
  -e MODEL_PATH="/models/huggingface/${model_rel}" \
  -e SPEC_MODEL_PATH="/models/huggingface/${model_rel}" \
  -e SERVED_MODEL_NAME=deepseek-v4-flash-0731-k2 \
  -e PORT="${port}" \
  -e MODE=dspark \
  -e DSPARK_TOKENS=5 \
  -e TP_SIZE="${tp_size}" \
  -e DCP_SIZE=1 \
  -e ALLREDUCE_MODE="${ALLREDUCE_MODE:-${allreduce}}" \
  -e CUTE_DSL_ARCH=sm_120a \
  -e MAX_MODEL_LEN=131072 \
  -e MAX_NUM_SEQS=4 \
  -e MAX_NUM_BATCHED_TOKENS=4096 \
  -e MAX_CUDAGRAPH_CAPTURE_SIZE=24 \
  -e CUDAGRAPH_CAPTURE_SIZES=1,2,4,6,12,24 \
  -e GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.975}" \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-0}" \
  -e KV_CACHE_DTYPE="${kv_cache_dtype}" \
  -e PREFIX_CACHE=0 \
  "${image}"

echo "${container} launched; follow startup with: docker logs -f ${container}"
