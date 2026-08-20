# DS4RT developer index

> This is a dense orientation and command index for people and coding agents
> changing DS4RT. Run commands from the repository root, inspect the worktree
> before editing, preserve unrelated changes, and use focused gates before
> consuming the five-machine test cluster.

## Stable release contract

- The topology is one x86_64 `sm_120` RTX PRO 6000 coordinator plus four ARM64
  `sm_121` DGX Spark expert hosts. Worker count is an architecture constant.
- Runtime rank identities are `spark-0` through `spark-3`; `SPARK_N_HOST`
  values are SSH/deployment names only.
- The coordinator owns API, scheduling, residuals, attention, physical KV,
  mHC, routers, shared experts, LM head, sampling, and dSpark control.
- Every Spark owns one quarter of every routed expert's intermediate
  dimension. All ranks receive the same top-6 routes; DS4RT is not expert
  parallel.
- The v1 model is the pinned public Pro EXL3 K2 artifact in `ds4rt.config`.
  Pro never falls back to native experts or an unqualified quantization recipe.
- Release serving uses host networking, Protocol V2, native verbs transport,
  GPU-resident experts, and startup-captured graph shapes.
- Generated files belong under ignored `.ds4rt-cache/`, `.ds4rt-release/`,
  `.ds4rt-release-image/`, `.ds4rt-wip/`, `dist/`, `rust/target/`, or
  `native/build*`, not in source control.

Read [`architecture.md`](architecture.md) before changing model ownership,
routes, caches, transport, graph lifecycles, or admission.

## Repository map

| Path | Responsibility |
| --- | --- |
| `rust/crates/ds4rt-daemon` | Binary, role startup, real model execution and scheduling |
| `rust/crates/ds4rt-api` | OpenAI-compatible routes, request/response semantics |
| `rust/crates/ds4rt-core` | Model/runtime contracts and shared types |
| `rust/crates/ds4rt-loader` | Snapshot catalog, classification, load planning |
| `rust/crates/ds4rt-transport` | Protocol V2 and remote expert transport |
| `rust/crates/ds4rt-ffi` | Native library loading and FFI boundary |
| `native/` | CUDA/C++ kernels, AOT bindings, RDMA and XGrammar integration |
| `python/reference/` | Reference operators, capture modules, profile resolver |
| `python/tools/` | Inspection, validation, benchmarks and release probes |
| `scripts/` | Build, deployment, smoke, provenance and operational helpers |
| `quantization/` | Reproducible offline EXL3 conversion tooling |
| `third_party/` | Verified Git submodules plus content-digest locks |
| `docker/` | Development, release and quantization image definitions |

## Session setup

| Command | Use |
| --- | --- |
| `git status --short --branch` | Establish existing changes before editing. |
| `git submodule status --recursive` | Check every dependency pin and nested checkout. |
| `just --list` | List the maintained command surface. |
| `source .venv/bin/activate` | Enter the optional host Python environment. |
| `export PATH=/usr/local/cuda/bin:$PATH` | Make the coordinator CUDA toolkit visible. |
| `export DS4RT_PYTHON="$PWD/.venv/bin/python"` | Select the embedded Python interpreter. |
| `export PYO3_PYTHON="$DS4RT_PYTHON"` | Bind PyO3 to the same interpreter. |
| `just doctor-host` | Inspect local GPU, CUDA, Docker, RDMA, routes and cache. |
| `just doctor-hosts` | Run the expert-host doctor over the configured Sparks. |
| `./run.sh --dry-run` | Validate the complete deployment without mutation. |

For a direct binary or focused Cargo test, derive the Python library directory
instead of hard-coding a patch version:

```bash
DS4RT_PYTHON_LIBDIR="$(
  "${DS4RT_PYTHON:-python3}" -c \
    'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")'
)"
export LD_LIBRARY_PATH="$DS4RT_PYTHON_LIBDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Do not export `PYTHONHOME`; it breaks embedded interpreter discovery.

## Dependency provenance

Initialize and verify all four public source pins before a release-shaped
build:

```bash
git submodule update --init --recursive
python3 scripts/verify-sparkinfer-source.py \
  --source third_party/sparkinfer \
  --lock third_party/sparkinfer.lock.json
python3 scripts/verify-xgrammar-source.py \
  --source third_party/xgrammar \
  --lock third_party/xgrammar.lock.json
python3 scripts/verify-exllamav3-source.py \
  --source third_party/exllamav3 \
  --lock third_party/exllamav3.lock.json
python3 scripts/verify-gptqmodel-source.py \
  --source third_party/gptqmodel \
  --lock third_party/gptqmodel.lock.json
```

Each lock binds repository URL, revision, and a deterministic source-tree
digest. Update a submodule and its lock together; never build from an
independently installed or modified copy.

## Static build checks

| Command | Use |
| --- | --- |
| `cargo fmt --manifest-path rust/Cargo.toml --all --check` | Check Rust formatting. |
| `cargo check --manifest-path rust/Cargo.toml --workspace` | Type-check every Rust crate. |
| `just build-rust` | Build the Rust workspace in the configured host environment. |
| `python3 -m compileall -q python/reference python/tools quantization` | Catch Python syntax failures. |
| `bash -n build.sh run.sh push-containers.sh stop.sh wip.sh scripts/*.sh docker/*.sh` | Syntax-check shell entrypoints. |
| `git diff --check` | Reject whitespace errors and conflict markers. |
| `./build.sh` | Build and distribute both release roles. |

A direct release daemon build is:

```bash
PYO3_PYTHON="${PYO3_PYTHON:-$PWD/.venv/bin/python}" \
  cargo build --manifest-path rust/Cargo.toml -p ds4rt-daemon --release
```

## Test ladder

| Command | Use |
| --- | --- |
| `just test-rust-fast` | Normal fast Rust gate with stale native discovery disabled. |
| `just test-rust` | Complete Rust workspace test suite. |
| `cargo test --manifest-path rust/Cargo.toml -p CRATE FILTER -- --nocapture` | One focused Rust test. |
| `just test-python` | All Python reference/oracle tests through `uv`. |
| `uv run --project python pytest PATH_OR_TEST -q` | One focused Python test. |
| `just test-native` | CPU-only native library and Rust FFI boundary. |
| `just build-native-coordinator-test` | Production-shaped coordinator CUDA/RDMA/AOT build. |
| `just test-rust-cuda-graphs` | Native CTest plus CUDA graph integration. |
| `just test-rust-phase0a-focused` | Focused real-path graph, KV and daemon tests. |
| `just test-native-rdma` | Clean RDMA native build check. |
| `just test-smoke` | Host doctor, Rust build, and full Rust tests. |

When a focused CUDA-backed daemon test needs native symbols, select the current
library explicitly:

```bash
DS4RT_NATIVE_LIB="$PWD/native/build-cuda-rdma-coordinator-aot/libds4rt_native.so" \
  cargo test --manifest-path rust/Cargo.toml \
  -p ds4rt-daemon TEST_FILTER -- --test-threads=1
```

An FFI symbol failure often means a stale ignored native library was loaded.

## Development containers and WIP slots

| Command | Use |
| --- | --- |
| `just docker-build-coordinator` | Build the amd64/`sm_120` development image. |
| `just docker-build-spark` | Build/stage the ARM64/`sm_121` development image. |
| `just docker-shell-coordinator bash` | Open a GPU-enabled coordinator container. |
| `just docker-shell-spark bash` | Open a GPU/RDMA-enabled Spark container. |
| `./wip.sh --slot NAME` | Incrementally build one fingerprinted slot on all roles. |
| `./run.sh --wip --wip-slot NAME` | Launch the exact WIP slot. |
| `./run.sh --wip --wip-slot NAME --restart` | Restart while retaining matched resident experts when safe. |

Use WIP slots for iteration. Use `./build.sh` only when the source snapshot is
ready for a production-shaped, distributable build. Slot identity includes
the source/dirty digest, rendered configuration, architecture, pinned sources,
and artifact hashes; do not bypass a mismatch.

## Release build and runtime

`./build.sh` builds amd64 locally and arm64 natively on `SPARK_0_HOST`. It then
copies the expert image to the remaining hosts and verifies engine revision,
SparkInfer revision, and provenance on every role. It requires at least 60 GiB
of free build space locally and on the seed Spark.

| Command | Use |
| --- | --- |
| `./build.sh` | Clean source build, artifact export, expert distribution. |
| `./run.sh --dry-run` | Images, SSH, model snapshots, capacity, profiles and active-state audit. |
| `./run.sh` | Start a clean five-container deployment. |
| `./run.sh --restart` | Replace a stale or configuration-mismatched deployment. |
| `scripts/api-smoke.sh` | Basic OpenAI API contract probe. |
| `scripts/api-constrained-smoke.sh` | Tool/JSON constrained-decoding probe. |
| `scripts/api-prefill-smoke.sh` | Timed prefill probe. |
| `./stop.sh` | Gracefully stop the selected release and WIP services. |

`run.sh` fails closed on a missing/mismatched image, model revision, EXL3
qualification, Spark source identity, GPU identity, capacity contract,
transport readiness, numerical probe, or API model list.

## Benchmarking

Release performance must be end-to-end and correctness-qualified. Record the
source revision, model revision, profile, dSpark policy, concurrency, prompt
shape, warmup/repeat count, and every sample. A timed request must report full
attention, numeric progression, and zero request-time graph captures.

Maintained harnesses include:

- `python/tools/bench_real_full_mtp_acceptance.py` for the seven-case adaptive
  semantic blend and draft acceptance;
- `python/tools/bench_real_full_repeat_decode.py` for the explicitly
  low-entropy repeated-word decode ceiling;
- `python/tools/bench_real_full_prefill_curve.py` for exact-row fresh prefill;
- the startup retained-prefix probe, enabled through
  `DS4RT_REAL_FULL_SERVE_PREFIX_PREFILL_PROBE=1`, for physical
  compressor/KV reuse (set `DS4RT_RELEASE_READY_TIMEOUT_SECONDS` high enough
  for the requested matrix);
- `python/tools/bench_real_full_concurrency.py` and
  `scripts/bench-real-full-mixed-concurrency.py` for concurrent decode;
- `python/tools/bench_real_full_long_context_session.py` for retained-state
  context growth; and
- Tool Eval Bench through the OpenAI-compatible endpoint for semantic and
  structured-output quality.

Keep raw generated runs under ignored cache/output directories. Commit a
compact measurement contract and summary only after the run passes its gates;
the v1 public format is demonstrated in [`benchmarks/pro-v1.json`](benchmarks/pro-v1.json).

## Publishing containers

The two release images must come from the same clean source revision and pass
an actual five-host run before publication:

```bash
./build.sh
./run.sh --restart
scripts/api-smoke.sh \
  "http://127.0.0.1:8000" \
  "wrldsuksgo2mars/DeepSeek-V4-Pro-0813-EXL3-K2-calibrated-v1"
./push-containers.sh v1
```

`push-containers.sh` publishes both `v1` and `latest`. The OCI source label in
`docker/Dockerfile.release` records this repository on both GHCR packages.
After a package's first publication, connect it to this repository and set its
visibility to public in GitHub's package settings. Do not retag mismatched
coordinator/expert roles or publish from a WIP slot.

## Change invariants

- Derive tensor catalogs and model facts from safetensors/config metadata.
- Preserve checkpoint operation order for MLA, compressors/indexers, mHC,
  routing, shared experts, routed experts, dSpark, and the LM head.
- Keep the four-rank global-route TP4 contract at every target/dSpark block.
- Keep physical attention state coordinator-local and scheduler-owned.
- Treat CUDA graph pointers, shapes, streams, and workspace offsets as stable
  bindings; prepare production shapes before readiness.
- Preserve deterministic source/model/image provenance and fail closed when an
  identity cannot be established.
- Do not put model weights, generated catalogs, build artifacts, benchmark
  databases, credentials, or site-local configuration into Git.
