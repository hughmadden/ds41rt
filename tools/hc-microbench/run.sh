#!/usr/bin/env bash
# run.sh — build and run hc_microbench, teeing all output to a timestamped
# markdown file. Read-only with respect to the rest of the machine: it never
# touches containers, the driver, or other processes (no docker, no
# nvidia-smi -r, no kill). The only nvidia-smi use is a query for the PCIe
# link generation/width, which CUDA itself cannot report.
set -euo pipefail

cd "$(dirname "$0")"

NVCC="$(command -v nvcc || true)"
if [[ -z "$NVCC" ]]; then
  NVCC="$(ls /usr/local/cuda*/bin/nvcc 2>/dev/null | sort | tail -n1 || true)"
fi
if [[ -z "$NVCC" ]]; then
  echo "error: no nvcc found (looked on PATH and in /usr/local/cuda*/bin)" >&2
  exit 1
fi

# Build: when a GPU is visible, target its native arch; otherwise fall back to
# sm_120 (RTX 5090 / Blackwell). (-arch=native without a visible GPU silently
# degrades to a default arch, which would not match the target card.)
if nvidia-smi >/dev/null 2>&1; then
  "$NVCC" -O2 -arch=native -o hc_microbench hc_microbench.cu
else
  echo "note: no visible GPU at build time; using -arch=sm_120" >&2
  "$NVCC" -O2 -arch=sm_120 -o hc_microbench hc_microbench.cu
fi

TS="$(date +%Y%m%d-%H%M%S)"
OUT="hc-microbench-${TS}.md"

{
  echo "# hc_microbench run ${TS}"
  echo
  echo '```'
  # nvidia-smi prints its driver-failure banner to stdout (and the banner
  # itself contains commas), so accept only lines whose gen and width fields
  # are numeric.
  smi_out="$(nvidia-smi --query-gpu=name,pcie.link.gen.current,pcie.link.width.current \
             --format=csv,noheader 2>/dev/null \
             | awk -F, 'NF >= 2 && $(NF-1) ~ /^[0-9]+$/ && $NF ~ /^[0-9]+$/' || true)"
  if [[ -n "$smi_out" ]]; then
    printf '%s\n' "$smi_out"
  else
    echo "nvidia-smi query unavailable; PCIe link info missing"
  fi
  echo '```'
  echo
  ./hc_microbench "$@"
} | tee "$OUT"

echo "wrote $OUT" >&2
