#!/usr/bin/env bash
set -Eeuo pipefail

image=${NATIVE_IMAGE:-voipmonitor/vllm@sha256:fdde59fed7f9fc12f9fd5ef1b3b3ea8d5097bf10ebad54b348497102c3a83f82}
kv_cache_dtype=${KV_CACHE_DTYPE:-fp8}
container=${CONTAINER_NAME:-ds4-native-rtx-tp2}
hf_cache_host=${HF_CACHE_HOST:-${HOME}/.cache/huggingface}
cache_host=${JIT_CACHE_HOST:-${HOME}/.cache/ds4-bench/native-rtx-tp2}
revision=9e165c30e2704aec5d9d593cce3eebd58bbef1cb
model_rel=hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/${revision}

test -d "${hf_cache_host}/${model_rel}" || {
  echo "missing native model snapshot: ${hf_cache_host}/${model_rel}" >&2
  exit 2
}
mkdir -p "${cache_host}" "${cache_host}/tmp" "${cache_host}/native-l2"

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
  --ulimit stack=67108864:67108864 \
  --entrypoint /usr/local/bin/serve-ds4-flash.sh \
  -v "${hf_cache_host}:/root/.cache/huggingface:ro" \
  -v "${cache_host}:/cache" \
  -v "${cache_host}/tmp:/container-tmp" \
  -v "${cache_host}/native-l2:/native-l2" \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e MODEL_PATH="/root/.cache/huggingface/${model_rel}" \
  -e SPEC_MODEL_PATH="/root/.cache/huggingface/${model_rel}" \
  -e SERVED_MODEL_NAME=deepseek-v4-flash-0731-native \
  -e PORT=8000 \
  -e MODE=dspark \
  -e BACKEND=b12x-a8 \
  -e TP_SIZE=2 \
  -e DCP_SIZE=1 \
  -e ALLREDUCE_MODE=auto \
  -e DSPARK_DEPTH_MODE=fixed \
  -e DSPARK_TOKENS=5 \
  -e MAX_NUM_SEQS=4 \
  -e MAX_MODEL_LEN=131072 \
  -e MAX_NUM_BATCHED_TOKENS=8192 \
  -e GRAPH=auto \
  -e GPU_MEMORY_UTILIZATION=0.975 \
  -e KV_CACHE_DTYPE="${kv_cache_dtype}" \
  -e EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:---default-chat-template-kwargs.thinking=false}" \
  -e LOAD_FORMAT=instanttensor \
  -e INSTANTTENSOR_BACKEND=BUFFERED \
  -e PREFIX_CACHE=0 \
  -e PYTHONHASHSEED=0 \
  "${image}"

echo "${container} launched; follow startup with: docker logs -f ${container}"
