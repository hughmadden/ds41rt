#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$repo_root/scripts/release-common.sh"

usage() {
  cat <<'EOF'
Usage: ./run.sh [--profile FILE] [--restart] [--dry-run]
       ./run.sh --wip [--wip-slot NAME] [--profile FILE] [--restart] [--dry-run]
                [--allow-development-unqualified-exl3]

Uses ds4rt.config beside this script by default. Despite the option name,
--profile FILE selects an entire alternate configuration file.

--restart  gracefully restarts the selected serving stack; WIP launches retain
           exact fingerprint-matched resident experts when safe
--dry-run  performs configuration/image/model/resource checks without mutation
--wip      runs a named slot inside persistent development containers
--allow-development-unqualified-exl3
           WIP-only explicit override for a manifest-marked unqualified EXL3 snapshot
EOF
}

for run_arg in "$@"; do
  if [[ "$run_arg" == --wip ]]; then
    exec "$repo_root/scripts/run-wip.sh" "$@"
  fi
done

config="$repo_root/ds4rt.config"
restart=0
dry_run=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile|--config)
      config="${2:?$1 requires a configuration file}"
      shift 2
      ;;
    --restart)
      restart=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      release_die "unknown run argument: $1"
      ;;
  esac
done

release_load_config "$config"
release_need docker
release_need ssh
release_need rsync
release_need jq
release_need curl
release_need ss
release_need nvidia-smi
release_need sha256sum
release_need python3

if ((restart && dry_run)); then
  release_die "--restart and --dry-run are mutually exclusive"
fi

docker info >/dev/null 2>&1 || release_die "local Docker daemon is unavailable"
if ((restart)); then
  # Switching back from the persistent WIP lane stops only its DS4RT
  # processes. The development containers, build caches, and slots survive.
  release_stop_wip_services || release_die "failed to stop one or more WIP services"
fi
docker image inspect "$COORDINATOR_DOCKER_INFERENCE" >/dev/null 2>&1 ||
  release_die "coordinator image is missing: $COORDINATOR_DOCKER_INFERENCE (run ./build.sh)"
sparkinfer_commit="$(
  python3 "$repo_root/scripts/verify-sparkinfer-source.py" \
    --source "$repo_root/third_party/sparkinfer" \
    --lock "$repo_root/third_party/sparkinfer.lock.json" \
    --print-revision
)"
coordinator_sparkinfer_commit="$(
  docker image inspect -f '{{index .Config.Labels "io.ds4rt.sparkinfer.revision"}}' \
    "$COORDINATOR_DOCKER_INFERENCE"
)"
[[ "$coordinator_sparkinfer_commit" == "$sparkinfer_commit" ]] ||
  release_die "coordinator image uses SparkInfer $coordinator_sparkinfer_commit; expected $sparkinfer_commit (run ./build.sh)"
coordinator_engine_commit="$(
  docker image inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' \
    "$COORDINATOR_DOCKER_INFERENCE"
)"
case "$coordinator_engine_commit" in
  ""|"<no value>"|unknown|unknown-*)
    release_die "coordinator image has no concrete DS4RT engine revision (run ./build.sh)"
    ;;
esac

hosts_csv="$(release_hosts_csv)"
lane_a_csv="$(release_lane_a_csv)"
lane_b_csv="$(release_lane_b_csv)"
expert_hosts_csv="$(release_expert_hosts_csv)"
coordinator_container="$RELEASE_COORDINATOR_CONTAINER_NAME"
spark_container_prefix="$RELEASE_SPARK_CONTAINER_PREFIX"
state_dir="$repo_root/.ds4rt-release"
mkdir -p "$state_dir"
hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$hf_home"
release_resolve_local_model_revision "$hf_home"
release_resolve_coordinator_gpu_identity

echo "== checking SSH, Docker images, and current containers =="
running_release_sparks=0
stale_release_sparks=0
running_legacy_sparks=0
for host in "$SPARK_0_HOST" "$SPARK_1_HOST" "$SPARK_2_HOST" "$SPARK_3_HOST"; do
  release_container="${spark_container_prefix}-${host}-${EXPERT_PORT}"
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" \
    "docker info >/dev/null && docker image inspect '$SPARK_EXPERT_DOCKER_INFERENCE' >/dev/null" ||
    release_die "$host is unreachable or lacks image $SPARK_EXPERT_DOCKER_INFERENCE"
  remote_sparkinfer_commit="$(
    ssh -o BatchMode=yes "$host" \
      "docker image inspect -f '{{index .Config.Labels \"io.ds4rt.sparkinfer.revision\"}}' '$SPARK_EXPERT_DOCKER_INFERENCE'"
  )"
  [[ "$remote_sparkinfer_commit" == "$sparkinfer_commit" ]] ||
    release_die "$host Spark image uses SparkInfer $remote_sparkinfer_commit; expected $sparkinfer_commit (run ./build.sh)"
  remote_engine_commit="$(
    ssh -o BatchMode=yes "$host" \
      "docker image inspect -f '{{index .Config.Labels \"org.opencontainers.image.revision\"}}' '$SPARK_EXPERT_DOCKER_INFERENCE'"
  )"
  [[ "$remote_engine_commit" == "$coordinator_engine_commit" ]] ||
    release_die "$host Spark image uses DS4RT engine $remote_engine_commit; coordinator uses $coordinator_engine_commit (run ./build.sh)"
  if ssh -o BatchMode=yes "$host" "docker container inspect '$release_container' >/dev/null 2>&1"; then
    image_state="$(
      ssh -o BatchMode=yes "$host" bash -s -- \
        "$release_container" "$SPARK_EXPERT_DOCKER_INFERENCE" <<'REMOTE'
container="$1"
image="$2"
if [ "$(docker inspect -f '{{.State.Running}}' "$container")" != true ]; then
  echo stopped
  exit 0
fi
container_image="$(docker inspect -f '{{.Image}}' "$container")"
selected_image="$(docker image inspect -f '{{.Id}}' "$image")"
if [ "$container_image" = "$selected_image" ]; then
  echo current
else
  echo stale
fi
REMOTE
    )"
    case "$image_state" in
      current)
        echo "  $host: current release expert container already running ($release_container)"
        ((running_release_sparks += 1))
        ;;
      stopped)
        echo "  $host: stopped release expert container exists ($release_container)"
        ((stale_release_sparks += 1))
        ;;
      *)
        echo "  $host: release expert container uses a stale image ($release_container)"
        ((stale_release_sparks += 1))
        ;;
    esac
  else
    echo "  $host: image ready; no current release expert container"
  fi
  if ssh -o BatchMode=yes "$host" "docker ps --format '{{.Names}}' | grep -Eq '^ds4rt-phase0-tcp-expertd-${host}-${EXPERT_PORT}$'" >/dev/null; then
    echo "  $host: legacy expert container currently occupies the GPU"
    ((running_legacy_sparks += 1))
  fi
done

release_coordinator_running=0
stale_release_coordinator=0
host_api_running=0
if docker container inspect "$coordinator_container" >/dev/null 2>&1; then
  if [[ "$(docker inspect -f '{{.State.Running}}' "$coordinator_container")" != true ]]; then
    echo "  coordinator: stopped release API container exists ($coordinator_container)"
    stale_release_coordinator=1
  else
    coordinator_image="$(docker inspect -f '{{.Image}}' "$coordinator_container")"
    selected_coordinator_image="$(docker image inspect -f '{{.Id}}' "$COORDINATOR_DOCKER_INFERENCE")"
    if [[ "$coordinator_image" == "$selected_coordinator_image" ]]; then
      echo "  coordinator: current release API container already running ($coordinator_container)"
      release_coordinator_running=1
    else
      echo "  coordinator: release API container uses a stale image ($coordinator_container)"
      stale_release_coordinator=1
    fi
  fi
fi
if ss -ltnp "sport = :${ADDR##*:}" 2>/dev/null | grep -q ds4rt &&
  ! docker ps --format '{{.Names}}' | grep -Fx "$coordinator_container" >/dev/null; then
  echo "  coordinator: host ds4rt API process currently occupies ${ADDR##*:}"
  host_api_running=1
elif ((release_coordinator_running == 0 && stale_release_coordinator == 0)); then
  echo "  coordinator: image ready; no API process running"
fi

profile_args=(
  --repo-root "$repo_root"
  --profile "$PROFILE"
  --model-id "$MODEL_ID"
  --model-variant "$MODEL_VARIANT"
  --expert-format "$EXPERT_FORMAT"
  --dspark "$DSPARK"
  --dspark-draft-policy "$DSPARK_DRAFT_POLICY"
  --coordinator-gpu "$COORDINATOR_GPU"
  --coordinator-gpu-uuid "$RELEASE_COORDINATOR_GPU_UUID"
  --coordinator-gpu-pci-bus-id "$RELEASE_COORDINATOR_GPU_PCI_BUS_ID"
  --headroom-gib "$COORDINATOR_GPU_HEADROOM_GIB"
  --concurrency "$CONCURRENCY"
  --spark-reduction-min-rows "$SPARK_REDUCTION_MIN_ROWS"
  --dry-run
)
[[ -z "$KV_POOL_TOKENS" ]] || profile_args+=(--kv-pool-tokens "$KV_POOL_TOKENS")
[[ -z "$MAX_CONTEXT_TOKENS" ]] || profile_args+=(--max-context-tokens "$MAX_CONTEXT_TOKENS")
[[ -z "$MAX_OUTPUT_TOKENS" ]] || profile_args+=(--max-output-tokens "$MAX_OUTPUT_TOKENS")

resolve_profile() {
  docker run --rm \
    --gpus device="$RELEASE_COORDINATOR_GPU_UUID" \
    --net=host \
    -v "$repo_root:$repo_root:ro" \
    -v "$hf_home:$hf_home:ro" \
    -v "$hf_home:/root/.cache/huggingface:ro" \
    -e HF_HOME="$hf_home" \
    "$COORDINATOR_DOCKER_INFERENCE" \
    python3 /opt/ds4rt/python/tools/resolve_serve_profile.py "${profile_args[@]}"
}

resolved_json="$state_dir/resolved-profile.json"
resolve_profile >"$resolved_json"
jq -e . "$resolved_json" >/dev/null || release_die "profile resolver returned invalid JSON"
resolved_dspark_draft_policy="$(
  jq -er '.dspark_draft_policy | select(type == "string")' "$resolved_json"
)" || release_die "profile resolver did not report its dSpark draft policy"
[[ "$resolved_dspark_draft_policy" == "$DSPARK_DRAFT_POLICY" ]] ||
  release_die "profile resolver draft policy mismatch: requested $DSPARK_DRAFT_POLICY, resolved $resolved_dspark_draft_policy"
resolved_fixed_drafts="$(
  jq -r '.environment.DS4RT_REAL_FULL_DSPARK_FIXED_DRAFTS // ""' "$resolved_json"
)"
if [[ "$DSPARK:$DSPARK_DRAFT_POLICY" == on:full ]]; then
  [[ "$resolved_fixed_drafts" == 5 ]] ||
    release_die "full dSpark policy did not resolve exactly five proposals"
else
  [[ -z "$resolved_fixed_drafts" ]] ||
    release_die "disabled/adaptive dSpark unexpectedly resolved a fixed proposal width"
fi

blockers="$(jq -r '.blockers[]?' "$resolved_json")"
[[ -z "$blockers" ]] || release_die "profile blockers:\n$blockers"
deployment_fingerprint="$(
  {
    jq -S . "$resolved_json"
    printf '%s\n' \
      "$ADDR" "$EXPERT_PORT" \
      "$coordinator_engine_commit" \
      "$RELEASE_MODEL_REVISION" \
      "$SPARKINFER_EXL3" \
      "$SPARK_0_HOST" "$SPARK_0_LANE_A" "$SPARK_0_LANE_B" \
      "$SPARK_1_HOST" "$SPARK_1_LANE_A" "$SPARK_1_LANE_B" \
      "$SPARK_2_HOST" "$SPARK_2_LANE_A" "$SPARK_2_LANE_B" \
      "$SPARK_3_HOST" "$SPARK_3_LANE_A" "$SPARK_3_LANE_B"
  } | sha256sum | awk '{print $1}'
)"

check_model_cache_local() {
  local model_id="$1"
  local revision="${2:-}"
  local root="$hf_home/hub/models--${model_id//\//--}"
  if [[ -z "$revision" && -f "$root/refs/main" ]]; then
    revision="$(<"$root/refs/main")"
  fi
  [[ -n "$revision" && -d "$root/snapshots/$revision" ]] ||
    release_die "coordinator model snapshot is missing: $model_id${revision:+@$revision} under $hf_home"
  if find "$root/snapshots/$revision" -xtype l -print -quit | grep -q .; then
    release_die "coordinator model snapshot has unresolved blobs: $model_id@$revision"
  fi
  if [[ "$EXPERT_FORMAT" == exl3 ]]; then
    python3 "$repo_root/python/tools/validate_ds4_staged_snapshot.py" \
      --checkpoint "$root/snapshots/$revision" \
      --model-id "$model_id" \
      --revision "$revision" \
      --startup-contract-only >/dev/null ||
      release_die "coordinator EXL3 snapshot is not a qualified immutable publication: $model_id@$revision"
  fi
}

check_model_cache_remote() {
  local host="$1"
  local model_id="$2"
  local revision="${3:-}"
  ssh -o BatchMode=yes "$host" bash -s -- \
    "$model_id" "$revision" "$EXPERT_FORMAT" "$SPARK_EXPERT_DOCKER_INFERENCE" <<'REMOTE'
set -euo pipefail
model_id="$1"
revision="$2"
expert_format="$3"
image="$4"
hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
root="$hf_home/hub/models--${model_id//\//--}"
if [[ -z "$revision" ]]; then
  test -f "$root/refs/main"
  revision="$(<"$root/refs/main")"
fi
test -n "$revision"
test -d "$root/snapshots/$revision"
if find "$root/snapshots/$revision" -xtype l -print -quit | grep -q .; then
  exit 1
fi
if [[ "$expert_format" == exl3 ]]; then
  docker run --rm \
    -v "$hf_home:$hf_home:ro" \
    -v "$hf_home:/root/.cache/huggingface:ro" \
    -e HF_HOME="$hf_home" \
    "$image" \
    python3 /opt/ds4rt/python/tools/validate_ds4_staged_snapshot.py \
      --checkpoint "$root/snapshots/$revision" \
      --model-id "$model_id" \
      --revision "$revision" \
      --startup-contract-only >/dev/null
fi
REMOTE
}

echo "== checking model snapshots =="
check_model_cache_local "$RELEASE_MODEL_ID" "$RELEASE_MODEL_REVISION"
for host in "$SPARK_0_HOST" "$SPARK_1_HOST" "$SPARK_2_HOST" "$SPARK_3_HOST"; do
  check_model_cache_remote "$host" "$RELEASE_MODEL_ID" "$RELEASE_MODEL_REVISION" ||
    release_die "$host is missing the selected text model: $RELEASE_MODEL_ID"
done
echo "  selected model snapshots are complete"

services_active=$((
  release_coordinator_running + stale_release_coordinator + host_api_running +
    running_release_sparks + stale_release_sparks + running_legacy_sparks
))
if ((services_active)); then
  if ((!restart)); then
    deployment_matches=0
    if ((release_coordinator_running == 1 &&
      running_release_sparks == 4 &&
      stale_release_coordinator == 0 &&
      stale_release_sparks == 0 &&
      running_legacy_sparks == 0 &&
      host_api_running == 0)); then
      coordinator_fingerprint="$(
        docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$coordinator_container" |
          sed -n 's/^DS4RT_RELEASE_CONFIG_SHA256=//p'
      )"
      spark_fingerprint_matches=1
      for host in "$SPARK_0_HOST" "$SPARK_1_HOST" "$SPARK_2_HOST" "$SPARK_3_HOST"; do
        release_container="${spark_container_prefix}-${host}-${EXPERT_PORT}"
        remote_fingerprint="$(
          ssh -o BatchMode=yes "$host" \
            "docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' '$release_container'" |
            sed -n 's/^DS4RT_RELEASE_CONFIG_SHA256=//p'
        )"
        [[ "$remote_fingerprint" == "$deployment_fingerprint" ]] ||
          spark_fingerprint_matches=0
      done
      if [[ "$coordinator_fingerprint" == "$deployment_fingerprint" ]] &&
        ((spark_fingerprint_matches)); then
        deployment_matches=1
      fi
    fi
    if ((deployment_matches)) &&
      release_api_advertises_model \
        "http://127.0.0.1:${ADDR##*:}" "$RELEASE_MODEL_ID"; then
      echo "All five release services already match the selected images and configuration."
      exit 0
    fi
    release_die "partial, legacy, stale, or configuration-mismatched service state is active; use --restart"
  fi
  echo "== stopping existing API and expert containers =="
  release_stop_services "$coordinator_container" "$spark_container_prefix"
fi

check_local_resources() {
  local available_kib gpu_line total_mib free_mib
  local min_gpu_mib=$((80 * 1024))
  available_kib="$(awk '/MemAvailable:/{print $2}' /proc/meminfo)"
  ((available_kib >= 8 * 1024 * 1024)) ||
    release_die "coordinator has less than 8 GiB available system memory"
  # Query the pinned physical GPU explicitly. On a multi-GPU coordinator,
  # `nvidia-smi | head` can SIGPIPE under pipefail and abort the launch.
  gpu_line="$(
    nvidia-smi --id="$RELEASE_COORDINATOR_GPU_PCI_BUS_ID" \
      --query-gpu=memory.total,memory.free --format=csv,noheader,nounits
  )"
  IFS=, read -r total_mib free_mib <<<"$gpu_line"
  total_mib="$(release_trim "$total_mib")"
  free_mib="$(release_trim "$free_mib")"
  ((free_mib >= min_gpu_mib)) ||
    release_die "coordinator GPU has only ${free_mib} MiB free of ${total_mib} MiB; at least ${min_gpu_mib} MiB (80 GiB) is required"
  echo "  coordinator: RAM $((available_kib / 1024)) MiB available; GPU ${free_mib}/${total_mib} MiB free"
}

check_remote_resources() {
  local host="$1"
  local min_unified_gib="$2"
  ssh -o BatchMode=yes "$host" bash -s -- \
    "$SPARK_EXPERT_DOCKER_INFERENCE" "$min_unified_gib" <<'REMOTE'
set -euo pipefail
image="$1"
min_unified_gib="$2"
min_unified_kib=$((min_unified_gib * 1024 * 1024))
min_unified_mib=$((min_unified_gib * 1024))
available_kib="$(awk '/MemAvailable:/{print $2}' /proc/meminfo)"
if ((available_kib < min_unified_kib)); then
  echo "only $((available_kib / 1024)) MiB unified system/GPU memory is available; at least ${min_unified_mib} MiB (${min_unified_gib} GiB) is required" >&2
  exit 2
fi
gpu_line="$(
  docker run --rm --gpus all "$image" \
    nvidia-smi --id=0 --query-gpu=memory.total,memory.free --format=csv,noheader,nounits
)"
IFS=, read -r total_mib free_mib <<<"$gpu_line"
total_mib="${total_mib//[[:space:]]/}"
free_mib="${free_mib//[[:space:]]/}"
if [[ "$total_mib" =~ ^[0-9]+$ && "$free_mib" =~ ^[0-9]+$ ]] &&
  ((free_mib < min_unified_mib)); then
  echo "GPU has only ${free_mib} MiB free of ${total_mib} MiB; at least ${min_unified_mib} MiB (${min_unified_gib} GiB) is required" >&2
  exit 2
fi
if [[ "$total_mib" =~ ^[0-9]+$ && "$free_mib" =~ ^[0-9]+$ ]]; then
  echo "RAM $((available_kib / 1024)) MiB available; GPU ${free_mib}/${total_mib} MiB free"
else
  echo "unified system/GPU memory $((available_kib / 1024)) MiB available"
fi
REMOTE
}

echo "== checking launch headroom =="
check_local_resources
spark_min_unified_gib=105
for host in "$SPARK_0_HOST" "$SPARK_1_HOST" "$SPARK_2_HOST" "$SPARK_3_HOST"; do
  remote_resources=""
  if ! remote_resources="$(check_remote_resources "$host" "$spark_min_unified_gib")"; then
    release_die "$host failed launch headroom check; refusing to start any release containers"
  fi
  echo "  $host: $remote_resources"
done
echo "  SparkInfer EXL3 mode: $SPARKINFER_EXL3"
echo "  coordinator GPU: $RELEASE_COORDINATOR_GPU_UUID ($RELEASE_COORDINATOR_GPU_PCI_BUS_ID)"
if ((dry_run)); then
  echo "Dry-run checks passed."
  exit 0
fi

echo "== starting fresh prebuilt Spark expert containers =="
eval "$(
  jq -r '.environment | to_entries[] | "\(.key)=\(.value | @sh); export \(.key)"' \
    "$resolved_json"
)"
export DS4RT_SPARK_HOSTS="$hosts_csv"
export DS4RT_REAL_FULL_SERVE_EXPERT_HOSTS="$expert_hosts_csv"
export DS4RT_SPARK_IMAGE="$SPARK_EXPERT_DOCKER_INFERENCE"
export DS4RT_SPARK_PREBUILT=1
export DS4RT_MODEL_REVISION="$RELEASE_MODEL_REVISION"
# Release images already contain the verified binary, native library, Python
# sources, and pinned SparkInfer tree.  Staging the mutable checkout is both
# unnecessary and unsafe: an old root-owned build tree can make rsync fail
# before the prebuilt container is even launched.
export DS4RT_SPARK_SKIP_STAGE=1
export DS4RT_RELEASE_CONFIG_SHA256="$deployment_fingerprint"
export DS4RT_SPARK_CONTAINER_PREFIX="$spark_container_prefix"
export DS4RT_SPARK_EXPERT_PORT="$EXPERT_PORT"
export DS4RT_SPARK_EXPERT_TRANSPORT=verbs-host
export DS4RT_SPARK_KEEP_EXPERTS=1
export DS4RT_SPARK_EXPERT_REAL_LAYER=all
export DS4RT_PHASE0_SPARK_SKIP_BENCH=1
export DS4RT_EXPERT_INTERMEDIATE_RDMA_PEERS="$lane_a_csv"
if [[ -n "$lane_b_csv" ]]; then
  export DS4RT_EXPERT_INTERMEDIATE_RDMA_ADDITIONAL_PEERS="$lane_b_csv"
else
  unset DS4RT_EXPERT_INTERMEDIATE_RDMA_ADDITIONAL_PEERS || true
fi
"$repo_root/scripts/phase0-spark-tcp-bench.sh" &
spark_start_pid=$!

env_file="$state_dir/coordinator.env"
jq -r '.environment | to_entries[] | "\(.key)=\(.value)"' "$resolved_json" >"$env_file"
if [[ -n "${DS4RT_REAL_FULL_DSPARK_PROFILE_AT_STARTUP:-}" ]]; then
  echo "DS4RT_REAL_FULL_DSPARK_PROFILE_AT_STARTUP=$DS4RT_REAL_FULL_DSPARK_PROFILE_AT_STARTUP" \
    >>"$env_file"
fi
for benchmark_env_name in \
  DS4RT_REAL_FULL_SERVE_PREFIX_PREFILL_PROBE \
  DS4RT_REAL_FULL_SERVE_PREFIX_PREFILL_PROBE_PREFIX_ROWS \
  DS4RT_REAL_FULL_SERVE_PREFIX_PREFILL_PROBE_NEW_ROWS \
  DS4RT_REAL_FULL_SERVE_PREFIX_PREFILL_PROBE_REPEATS; do
  if [[ -v "$benchmark_env_name" ]]; then
    printf '%s=%s\n' "$benchmark_env_name" "${!benchmark_env_name}" >>"$env_file"
  fi
done
coordinator_image_id="$(docker image inspect -f '{{.Id}}' "$COORDINATOR_DOCKER_INFERENCE")"
{
  echo "ADDR=$ADDR"
  echo "DS4RT_REAL_FULL_SERVE_EXPERT_HOSTS=$expert_hosts_csv"
  echo "DS4RT_SPARK_HOSTS=$hosts_csv"
  echo "DS4RT_SPARK_EXPERT_PORT=$EXPERT_PORT"
  echo "DS4RT_SPARKINFER_EXL3=$SPARKINFER_EXL3"
  echo "DS4RT_MODEL_REVISION=$RELEASE_MODEL_REVISION"
  echo "DS4RT_REAL_FULL_SERVE_START_EXPERTS=0"
  echo "DS4RT_REAL_FULL_SERVE_BUILD_DAEMON=0"
  echo "DS4RT_REAL_FULL_SERVE_BUILD_NATIVE=0"
  echo "DS4RT_REAL_FULL_SERVE_REQUIRE_CUDA=1"
  echo "DS4RT_REAL_FULL_SERVE_EXPERT_WARMUP_STATUS_FILE=/tmp/ds4rt-expert-warmup.status"
  echo "DS4RT_BIN=/opt/ds4rt/bin/ds4rt"
  echo "DS4RT_NATIVE_LIB=/opt/ds4rt/lib/libds4rt_native.so"
  echo "DS4RT_ENGINE_COMMIT=$(docker image inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$COORDINATOR_DOCKER_INFERENCE")"
  echo "DS4RT_RELEASE_CONFIG_SHA256=$deployment_fingerprint"
  echo "DS4RT_KERNEL_CACHE_BASE=/var/cache/ds4rt/kernels"
  echo "DS4RT_KERNEL_CACHE_ENVIRONMENT_ID=$coordinator_image_id"
  echo "DS4RT_RUNTIME_CATALOG_CACHE_DIR=/var/cache/ds4rt/catalogs"
} >>"$env_file"

mkdir -p "$state_dir/kernel-cache" "$state_dir/catalog-cache"
docker_args=(
  run -d
  --name "$coordinator_container"
  --restart no
  --gpus device="$RELEASE_COORDINATOR_GPU_UUID"
  --net=host
  --ipc=host
  --ulimit memlock=-1:-1
  --cap-add IPC_LOCK
  --env-file "$env_file"
  --workdir "$repo_root"
  -v "$repo_root:$repo_root:ro"
  -v "$hf_home:$hf_home:ro"
  -v "$hf_home:/root/.cache/huggingface:ro"
  -v "$state_dir/kernel-cache:/var/cache/ds4rt/kernels"
  -v "$state_dir/catalog-cache:/var/cache/ds4rt/catalogs"
  -e HF_HOME="$hf_home"
)
if [[ -e /dev/infiniband ]]; then
  docker_args+=(--device=/dev/infiniband)
fi
docker_args+=(
  "$COORDINATOR_DOCKER_INFERENCE"
  /opt/ds4rt/scripts/real-full-tcp-serve.sh
)

echo "== starting coordinator container =="
if ! docker "${docker_args[@]}" >/dev/null; then
  kill "$spark_start_pid" >/dev/null 2>&1 || true
  wait "$spark_start_pid" >/dev/null 2>&1 || true
  release_die "failed to start coordinator container"
fi

if ! wait "$spark_start_pid"; then
  docker logs --tail 200 "$coordinator_container" >&2 || true
  docker rm -f "$coordinator_container" >/dev/null 2>&1 || true
  release_die "one or more Spark experts failed during parallel startup"
fi

ready_timeout_seconds="${DS4RT_RELEASE_READY_TIMEOUT_SECONDS:-900}"
[[ "$ready_timeout_seconds" =~ ^[1-9][0-9]*$ ]] ||
  release_die "DS4RT_RELEASE_READY_TIMEOUT_SECONDS must be a positive integer"
deadline=$((SECONDS + ready_timeout_seconds))
until release_api_advertises_model \
  "http://127.0.0.1:${ADDR##*:}" "$RELEASE_MODEL_ID"; do
  coordinator_state="$(
    docker inspect -f '{{.State.Status}} {{.RestartCount}}' \
      "$coordinator_container" 2>/dev/null || true
  )"
  if [[ "$coordinator_state" != "running 0" ]]; then
    docker logs --tail 200 "$coordinator_container" >&2 || true
    release_die "coordinator container became unhealthy during startup (state: ${coordinator_state:-missing})"
  fi
  ((SECONDS < deadline)) || {
    docker logs --tail 200 "$coordinator_container" >&2 || true
    release_die "API did not become ready within $ready_timeout_seconds seconds"
  }
  sleep 0.25
done

curl -fsS "http://127.0.0.1:${ADDR##*:}/v1/models" >"$state_dir/models.json"
release_validate_model_list_file "$state_dir/models.json" "$RELEASE_MODEL_ID"
echo "DS4RT release server is ready at http://127.0.0.1:${ADDR##*:}/v1/"
echo "  profile:     $PROFILE"
echo "  model:       $RELEASE_MODEL_ID"
echo "  variant:     $MODEL_VARIANT"
echo "  experts:     $EXPERT_FORMAT"
if [[ "$DSPARK" == on ]]; then
  echo "  dSpark:      on ($DSPARK_DRAFT_POLICY proposals)"
else
  echo "  dSpark:      off"
fi
echo "  EXL3 kernel: $SPARKINFER_EXL3"
echo "  concurrency: $CONCURRENCY"
echo "  containers:  $coordinator_container + four $spark_container_prefix experts"
