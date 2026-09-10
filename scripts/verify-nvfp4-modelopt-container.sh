#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="${DS41RT_NVFP4_MODEL_OPT_IMAGE:-nvcr.io/nvidia/pytorch:26.05-py3}"
fixture="${DS41RT_NVFP4_MODEL_OPT_FIXTURE:-tests/fixtures/nvfp4/real_tensor_decode.json}"
output="${DS41RT_NVFP4_MODEL_OPT_OUTPUT:-tests/fixtures/nvfp4/modelopt_reference.json}"
gpus="${DS41RT_NVFP4_MODEL_OPT_DOCKER_GPUS:-all}"
device="${DS41RT_NVFP4_MODEL_OPT_DEVICE:-cuda}"

docker_args=(
  run --rm -i
  --ipc=host
  --ulimit memlock=-1
  --ulimit stack=67108864
  -v "$repo_root:/workspace/ds41rt"
  -w /workspace/ds41rt
  -e "DS41RT_NVFP4_MODEL_OPT_IMAGE=$image"
  -e "DS41RT_NVFP4_MODEL_OPT_DEVICE=$device"
  --entrypoint python
)
if [ -n "$gpus" ] && [ "$gpus" != "none" ]; then
  docker_args+=(--gpus "$gpus")
fi

docker "${docker_args[@]}" "$image" \
  python/tools/verify_nvfp4_modelopt_real_tensor_decode_fixture.py \
    --fixture "$fixture" \
    --output "$output" \
    --device "$device"
