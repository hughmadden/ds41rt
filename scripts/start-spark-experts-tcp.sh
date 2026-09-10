#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: scripts/start-spark-experts-tcp.sh [--hosts CSV] [--mode real|synthetic] [--catalog PATH] [--loadplan-dir DIR] [--dry-run]

Starts persistent Spark-hosted ProtocolV2 TCP expert daemons. Source/dev real
mode uses the native NVFP4 binary protocol path with CUDA, model catalog, and
per-host loadplans. Prebuilt release mode infers the catalog and placement from
the selected model and stable spark-0..3 role. The underlying Spark launcher
stages the repo and distributes the existing Spark image over the configured
Spark image copy path; it only rebuilds the image when
DS41RT_SPARK_BUILD_IMAGE=1 is set.

Environment:
  DS41RT_SPARK_HOSTS                 default: ostrich,dodo,emu,kiwi
  DS41RT_PHASE0_SPARK_EXPERT_MODE    real or synthetic; default: real
  DS41RT_SPARK_PREBUILT              use release artifacts and inferred placement; default: 0
  DS41RT_PHASE0_SPARK_CATALOG        default: .ds41rt-cache/model-artifacts/diagnostic/model_catalog.json
  DS41RT_PHASE0_SPARK_LOADPLAN_DIR   default: .ds41rt-cache/model-artifacts/diagnostic
  DS41RT_SPARK_EXPERT_REAL_LAYER     default: all for real start-only serving
  DS41RT_REAL_FULL_NVFP4_ROUTE_CUDA_GRAPHS
                                      forward to Spark expert containers; default: 1 in real mode
  DS41RT_B12X_SPARK_AOT               build direct SparkInfer AOT kernels; default: 1 in real mode
  DS41RT_B12X_SPARK_GROUPED_DECODE    grouped TP4 M=1/topk=8 decode; default: 1
  DS41RT_B12X_SPARK_W4A16_M1_FUSED_SUM
                                      atomic M=1 top-k accumulation; default: 1 in real mode
  DS41RT_REAL_FULL_NVFP4_ROUTE_CUDA_EVENT_TIMING
                                      forward to Spark expert containers; default: 0
  DS41RT_SPARK_GPU_RUNTIME           nvidia or manual; default: nvidia
  DS41RT_SPARK_EXPERT_TRANSPORT      tcp or verbs-host; default: tcp
  DS41RT_PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES
                                      concurrent RDMA/NCCL request lanes in 1..8; default: 2
  DS41RT_REAL_FULL_PROTOCOL_V2_PACKED_DIRECT_MAX_ROWS
                                      non-reduced packed expert rows in 8..16; default: 16
  DS41RT_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP
                                      auto-discovered per Spark from RDMA netdev IPs
  DS41RT_EXPERT_INTERMEDIATE_SHARDS    strict real serving requires 4
  DS41RT_EXPERT_INTERMEDIATE_REDUCTION defaults to spark-rdma for verbs-host,
                                      coordinator for TCP
EOF
  exit 0
fi

hosts="${DS41RT_SPARK_HOSTS:-ostrich,dodo,emu,kiwi}"
mode="${DS41RT_PHASE0_SPARK_EXPERT_MODE:-real}"
catalog="${DS41RT_PHASE0_SPARK_CATALOG:-${CATALOG:-.ds41rt-cache/model-artifacts/diagnostic/model_catalog.json}}"
loadplan_dir="${DS41RT_PHASE0_SPARK_LOADPLAN_DIR:-.ds41rt-cache/model-artifacts/diagnostic}"
prebuilt="${DS41RT_SPARK_PREBUILT:-0}"
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hosts)
      hosts="${2:?--hosts requires a comma-separated host list}"
      shift 2
      ;;
    --mode)
      mode="${2:?--mode requires real or synthetic}"
      shift 2
      ;;
    --catalog)
      catalog="${2:?--catalog requires a path}"
      shift 2
      ;;
    --loadplan-dir)
      loadplan_dir="${2:?--loadplan-dir requires a path}"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

case "$mode" in
  real|synthetic) ;;
  *)
    echo "--mode must be real or synthetic, got: $mode" >&2
    exit 2
    ;;
esac

if [[ -z "$hosts" ]]; then
  echo "--hosts must not be empty" >&2
  exit 2
fi

if [[ "$mode" == "real" && "$prebuilt" != "1" ]]; then
  test -f "$catalog" || {
    echo "catalog not found: $catalog" >&2
    exit 2
  }
  IFS=',' read -r -a host_items <<< "$hosts"
  for host in "${host_items[@]}"; do
    host="$(echo "$host" | xargs)"
    [[ -n "$host" ]] || continue
    test -f "${loadplan_dir}/loadplan.${host}.json" || {
      echo "loadplan not found for $host: ${loadplan_dir}/loadplan.${host}.json" >&2
      exit 2
    }
  done
  export DS41RT_SPARK_EXPERT_REAL_LAYER="${DS41RT_SPARK_EXPERT_REAL_LAYER:-all}"
fi

if [[ "$mode" == "real" ]]; then
  [[ "${DS41RT_EXPERT_INTERMEDIATE_SHARDS:-4}" == "4" ]] || {
    echo "strict real serving requires DS41RT_EXPERT_INTERMEDIATE_SHARDS=4" >&2
    exit 2
  }
  export DS41RT_EXPERT_INTERMEDIATE_SHARDS=4
  if [[ -z "${DS41RT_EXPERT_INTERMEDIATE_REDUCTION:-}" ]]; then
    if [[ "${DS41RT_SPARK_EXPERT_TRANSPORT:-tcp}" == "verbs-host" ]]; then
      export DS41RT_EXPERT_INTERMEDIATE_REDUCTION=spark-rdma
    else
      export DS41RT_EXPERT_INTERMEDIATE_REDUCTION=coordinator
    fi
  fi
fi

export DS41RT_SPARK_HOSTS="$hosts"
export DS41RT_PHASE0_SPARK_EXPERT_MODE="$mode"
export DS41RT_PHASE0_SPARK_CATALOG="$catalog"
export DS41RT_PHASE0_SPARK_LOADPLAN_DIR="$loadplan_dir"
export DS41RT_SPARK_KEEP_EXPERTS="${DS41RT_SPARK_KEEP_EXPERTS:-1}"
export DS41RT_PHASE0_SPARK_SKIP_BENCH=1

if [[ "$dry_run" == "1" ]]; then
  printf 'DS41RT_SPARK_HOSTS=%s\n' "$DS41RT_SPARK_HOSTS"
  printf 'DS41RT_PHASE0_SPARK_EXPERT_MODE=%s\n' "$DS41RT_PHASE0_SPARK_EXPERT_MODE"
  printf 'DS41RT_SPARK_PREBUILT=%s\n' "$prebuilt"
  printf 'DS41RT_PHASE0_SPARK_CATALOG=%s\n' "$DS41RT_PHASE0_SPARK_CATALOG"
  printf 'DS41RT_PHASE0_SPARK_LOADPLAN_DIR=%s\n' "$DS41RT_PHASE0_SPARK_LOADPLAN_DIR"
  printf 'DS41RT_SPARK_KEEP_EXPERTS=%s\n' "$DS41RT_SPARK_KEEP_EXPERTS"
  printf 'DS41RT_PHASE0_SPARK_SKIP_BENCH=%s\n' "$DS41RT_PHASE0_SPARK_SKIP_BENCH"
  if [[ "${DS41RT_SPARK_EXPERT_REAL_LAYER+x}" ]]; then
    printf 'DS41RT_SPARK_EXPERT_REAL_LAYER=%s\n' "$DS41RT_SPARK_EXPERT_REAL_LAYER"
  fi
  if [[ "${DS41RT_REAL_FULL_NVFP4_ROUTE_CUDA_GRAPHS+x}" ]]; then
    printf 'DS41RT_REAL_FULL_NVFP4_ROUTE_CUDA_GRAPHS=%s\n' "$DS41RT_REAL_FULL_NVFP4_ROUTE_CUDA_GRAPHS"
  elif [[ "$mode" == "real" ]]; then
    printf 'DS41RT_REAL_FULL_NVFP4_ROUTE_CUDA_GRAPHS=1\n'
  else
    printf 'DS41RT_REAL_FULL_NVFP4_ROUTE_CUDA_GRAPHS=0\n'
  fi
  if [[ "${DS41RT_B12X_SPARK_AOT+x}" ]]; then
    printf 'DS41RT_B12X_SPARK_AOT=%s\n' "$DS41RT_B12X_SPARK_AOT"
  elif [[ "$mode" == "real" ]]; then
    printf 'DS41RT_B12X_SPARK_AOT=1\n'
  else
    printf 'DS41RT_B12X_SPARK_AOT=0\n'
  fi
  printf 'DS41RT_B12X_SPARK_GROUPED_DECODE=%s\n' "${DS41RT_B12X_SPARK_GROUPED_DECODE:-1}"
  if [[ "${DS41RT_SERVE_PROFILE+x}" ]]; then
    printf 'DS41RT_SERVE_PROFILE=%s\n' "$DS41RT_SERVE_PROFILE"
  fi
  if [[ "${DS41RT_B12X_SPARK_W4A16_M1_FUSED_SUM+x}" ]]; then
    printf 'DS41RT_B12X_SPARK_W4A16_M1_FUSED_SUM=%s\n' "$DS41RT_B12X_SPARK_W4A16_M1_FUSED_SUM"
  elif [[ "$mode" == "real" ]]; then
    printf 'DS41RT_B12X_SPARK_W4A16_M1_FUSED_SUM=1\n'
  else
    printf 'DS41RT_B12X_SPARK_W4A16_M1_FUSED_SUM=0\n'
  fi
  printf 'DS41RT_REAL_FULL_NVFP4_ROUTE_CUDA_EVENT_TIMING=%s\n' "${DS41RT_REAL_FULL_NVFP4_ROUTE_CUDA_EVENT_TIMING:-0}"
  printf 'DS41RT_PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES=%s\n' "${DS41RT_PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES:-2}"
  printf 'DS41RT_REAL_FULL_PROTOCOL_V2_PACKED_DIRECT_MAX_ROWS=%s\n' "${DS41RT_REAL_FULL_PROTOCOL_V2_PACKED_DIRECT_MAX_ROWS:-2048}"
  printf 'DS41RT_SPARK_GPU_RUNTIME=%s\n' "${DS41RT_SPARK_GPU_RUNTIME:-nvidia}"
  if [[ "${DS41RT_SPARK_EXPERT_TRANSPORT:-tcp}" == "verbs-host" ]]; then
    printf 'DS41RT_SPARK_EXPERT_TRANSPORT=%s\n' "$DS41RT_SPARK_EXPERT_TRANSPORT"
  elif [[ "${DS41RT_SPARK_EXPERT_TRANSPORT+x}" ]]; then
    printf 'DS41RT_SPARK_EXPERT_TRANSPORT=%s\n' "$DS41RT_SPARK_EXPERT_TRANSPORT"
  fi
  printf 'exec scripts/phase0-spark-tcp-bench.sh\n'
  exit 0
fi

exec scripts/phase0-spark-tcp-bench.sh
