#!/bin/bash
source .venv/bin/activate

# CUDA host setup — RTX PRO 6000 Blackwell, sm_120
export CUDA_HOME="/usr/local/cuda-13.3"
export CUDA_PATH="/usr/local/cuda-13.3"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

echo "CUDA: $(nvcc --version 2>&1 | head -1)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"

codex --dangerously-bypass-approvals-and-sandbox resume
