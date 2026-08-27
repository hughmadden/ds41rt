#!/usr/bin/env bash
set -Eeuo pipefail

if (( $# != 2 )); then
  echo "usage: $0 LABEL SERVED_MODEL_NAME" >&2
  exit 2
fi

label=$1
model=$2
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
result_dir=${RESULT_DIR:-${repo_root}/reports/performance/spark-v-rtx-k2-native}
benchy=${LLAMA_BENCHY:-${HOME}/.local/share/uv/tools/tool-eval-bench/bin/llama-benchy}
tokenizer=${TOKENIZER:-${HOME}/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/9e165c30e2704aec5d9d593cce3eebd58bbef1cb/tokenizer.json}
base_url=${BASE_URL:-http://127.0.0.1:8000/v1}
extra_body_args=(--extra-body "temperature=0,enable_thinking=false")
if [[ -n "${EXTRA_BODY:-}" ]]; then
  extra_body_args+=(--extra-body "${EXTRA_BODY}")
fi
warmup_args=()
if [[ "${BENCH_NO_WARMUP:-0}" == "1" ]]; then
  # Use this only after an explicit serving warmup. It makes a metrics-proxy
  # capture contain exactly the 63 measured requests in the 3x3x3 matrix.
  warmup_args=(--no-warmup)
fi

test -x "${benchy}" || { echo "llama-benchy not found: ${benchy}" >&2; exit 2; }
test -f "${tokenizer}" || { echo "tokenizer not found: ${tokenizer}" >&2; exit 2; }
mkdir -p "${result_dir}"

"${benchy}" \
  --base-url "${base_url}" \
  --model "${model}" \
  --served-model-name "${model}" \
  --tokenizer "${tokenizer}" \
  --pp 2048 \
  --tg 128 \
  --exact-tg \
  --depth 0 4096 8192 \
  --concurrency 1 2 4 \
  --runs 3 \
  "${warmup_args[@]}" \
  --no-cache \
  --latency-mode generation \
  --skip-coherence \
  --no-adapt-prompt \
  "${extra_body_args[@]}" \
  --format json \
  --save-result "${result_dir}/${label}.json"
