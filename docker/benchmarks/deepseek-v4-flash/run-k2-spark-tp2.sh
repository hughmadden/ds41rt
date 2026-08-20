#!/usr/bin/env bash
set -Eeuo pipefail

head_host=${HEAD_HOST:-kiwi}
worker_host=${WORKER_HOST:-dodo}
head_ip=${HEAD_IP:-10.55.0.4}
worker_ip=${WORKER_IP:-10.55.0.2}
image=${K2_SPARK_IMAGE:-deepseek-v4-flash-0731-exl3-k2-spark:arm64-tp}
kv_cache_dtype=${KV_CACHE_DTYPE:-fp8}
model_revision=dff9afc6f5fe50a890590f7b6d5339ceaf5ba51e
model_root=/home/tj/Developer/deepseek-v4-flash-0731-exl3-k2-spark/data
model_rel=huggingface/hub/models--wrldsuksgo2mars--DeepSeek-V4-Flash-0731-EXL3-K2-calibrated-v0/snapshots/${model_revision}
head_container=ds4-k2-spark-tp2-head
worker_container=ds4-k2-spark-tp2-worker

remote() {
  local host=$1 remote_command
  shift
  printf -v remote_command '%q ' "$@"
  ssh -o BatchMode=yes "${host}" "${remote_command}"
}

remove_if_present() {
  local host=$1 container=$2
  if remote "${host}" docker container inspect "${container}" >/dev/null 2>&1; then
    remote "${host}" docker rm -f "${container}" >/dev/null
  fi
}

common_args=(
  --gpus all
  --network host
  --ipc host
  --ulimit memlock=-1:-1
  --ulimit nofile=1048576:1048576
  --entrypoint bash
  -v "${model_root}:/models:ro"
  -e CUDA_VISIBLE_DEVICES=0
  -e MODEL_PATH="/models/${model_rel}"
  -e SPEC_MODEL_PATH="/models/${model_rel}"
  -e SERVED_MODEL_NAME=deepseek-v4-flash-0731-k2-spark-tp2
  -e MODE=dspark
  -e DSPARK_TOKENS=5
  -e TP_SIZE=2
  -e DCP_SIZE=1
  -e ALLREDUCE_MODE=nccl
  -e CUTE_DSL_ARCH=sm_121a
  -e MAX_MODEL_LEN=131072
  -e MAX_NUM_SEQS=4
  -e MAX_NUM_BATCHED_TOKENS=4096
  -e MAX_CUDAGRAPH_CAPTURE_SIZE=6
  -e CUDAGRAPH_CAPTURE_SIZES=6
  -e GPU_MEMORY_UTILIZATION=0.85
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
  -e KV_CACHE_DTYPE="${kv_cache_dtype}"
  -e PREFIX_CACHE=0
  -e EXTRA_VLLM_ARGS="--distributed-executor-backend ray"
  -e NCCL_IB_DISABLE=0
  -e NCCL_IB_HCA="=rocep1s0f0,roceP2p1s0f0"
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0
  -e GLOO_SOCKET_IFNAME=enp1s0f0np0
  -e RAY_memory_monitor_refresh_ms=0
  -e RAY_USAGE_STATS_ENABLED=0
  -e HF_HUB_OFFLINE=1
  -e TRANSFORMERS_OFFLINE=1
)
ray_cli=(/opt/runtime-venv/bin/python -m ray.scripts.scripts)

remove_if_present "${head_host}" "${head_container}"
remove_if_present "${worker_host}" "${worker_container}"
remote "${head_host}" mkdir -p /home/tj/.cache/ds4-bench/k2-spark-tp2
remote "${worker_host}" mkdir -p /home/tj/.cache/ds4-bench/k2-spark-tp2

remote "${head_host}" docker run -d \
  --name "${head_container}" \
  "${common_args[@]}" \
  -v /home/tj/.cache/ds4-bench/k2-spark-tp2:/cache \
  -e VLLM_HOST_IP="${head_ip}" \
  "${image}" -lc \
  "exec /opt/runtime-venv/bin/python -m ray.scripts.scripts start --head --node-ip-address=${head_ip} --port=6379 --include-dashboard=false --block" \
  >/dev/null

for _ in $(seq 1 30); do
  if remote "${head_host}" docker exec "${head_container}" \
      "${ray_cli[@]}" status >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

remote "${worker_host}" docker run -d \
  --name "${worker_container}" \
  "${common_args[@]}" \
  -v /home/tj/.cache/ds4-bench/k2-spark-tp2:/cache \
  -e VLLM_HOST_IP="${worker_ip}" \
  "${image}" -lc \
  "exec /opt/runtime-venv/bin/python -m ray.scripts.scripts start --address=${head_ip}:6379 --node-ip-address=${worker_ip} --block" \
  >/dev/null

for _ in $(seq 1 60); do
  if remote "${head_host}" docker exec "${head_container}" \
      "${ray_cli[@]}" status 2>/dev/null \
      | grep -q '/2.0 GPU'; then
    break
  fi
  sleep 1
done
remote "${head_host}" docker exec "${head_container}" "${ray_cli[@]}" status

remote "${head_host}" docker exec -d "${head_container}" bash -lc \
  'exec /opt/recipe/scripts/entrypoint.sh > /cache/server.log 2>&1'

echo "TP2 K2 cluster launched. Follow startup with:"
echo "  ssh ${head_host} tail -f /home/tj/.cache/ds4-bench/k2-spark-tp2/server.log"
