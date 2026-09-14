# DS41RT

DS41RT serves the official [DeepSeek V4.1 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) checkpoint across one or two RTX PRO 6000 Blackwell coordinator GPUs and four DGX Spark expert workers. It combines native target execution, local dSpark speculative decoding, token-level prefix reuse, tools, constrained output, and vision in one OpenAI-compatible service.

All reported RTX measurements use an enforced **400 W power limit** and **standard 14,001 MHz maximum memory speed with no memory overclock**. The loaded memory clock reached 13,365 MHz; the RTX driver was 595.91.07. The four GB10 workers used driver 580.159.03.

[![DS41RT native execution across RTX coordinators and four expert workers](docs/native-path-execution.svg)](docs/native-path-execution.svg)

## Performance

The standard launcher now selects two feasible peer-connected RTX cards automatically and falls back to one. Two-RTX mode hosts all twenty encoder expert layers TP2 on the RTX pair; one-RTX mode uses bottom-up local placement and the new placement-aware adaptive K5 default. Throughput uses temperature zero and thinking disabled. Every cell below has three fresh samples from the same release image and matched prompts.

**Headline results.** Median throughput, the qualified cache and placement policy, memory use, and clean startup.

| Measurement | 1 RTX | 2 RTX | 2 RTX change |
|---|---:|---:|---:|
| Best median prefill | 7,878 tok/s (0 + 32K) | **8,454 tok/s** (0 + 16K) | +7.3% |
| Counting target-only decode | 44.95 tok/s | **48.41 tok/s** | +7.7% |
| Counting dSpark decode | 156.08 tok/s | **181.31 tok/s** | +16.2% |
| Weighted eight-type target-only decode | 43.04 tok/s | **46.14 tok/s** | +7.2% |
| Weighted eight-type dSpark decode | 76.72 tok/s | **79.33 tok/s** | +3.4% |
| C16 code aggregate | 992.78 tok/s | **1,181.49 tok/s** | +19.0% |
| C16 topic aggregate | 514.96 tok/s | **596.14 tok/s** | +15.8% |
| C16 counting aggregate | 1,161.12 tok/s | **1,333.57 tok/s** | +14.9% |
| C16 mixed aggregate | 196.46 tok/s | **309.06 tok/s** | +57.3% |
| RTX routed-expert placement | Layers 0–4, full width | Encoder 0–19, TP2 | — |
| Spark expert residency / configured budget | 40 layers / 100 GiB each | 20 layers / 100 GiB each | — |
| Global FP4 source pool | 16.681 GB / 18,710,016 logical tokens + 32,768 private-tail | 13.094 GB / 14,680,064 logical tokens + 32,768 private-tail | — |
| Exact prompt / completed-turn retention | 24 / 24 entries | 24 / 24 entries | — |
| Loaded RTX memory after readiness | 95,080 MiB | 95,338 / 95,578 MiB | — |
| Standard restart to API readiness | 58.05 s | **46.00 s** | −20.8% |

The weighted direct-corpus result is the main real-content decode headline. Counting is retained as a low-entropy throughput reference. The mixed workload has different prompt and output lengths from the same-code workload and should be compared only against its matching layout.

**Eight content types and counting.** Three samples per local mode. The official Flash column is the preserved one-request reference and was not called again. Counting is outside the weighted score. Every local response completed; code, math, and JSON objective checks passed, while open prose was intentionally left unscored.

| Case | 1 RTX target | 1 RTX dSpark | 2 RTX target | 2 RTX dSpark | Official Flash |
|---|---:|---:|---:|---:|---:|
| Code | 44.90 | 127.62 | 48.10 | **149.07** | 345.90 |
| Math | 41.93 | 119.56 | 47.76 | **135.72** | 285.33 |
| Fable | 41.31 | 51.88 | 44.39 | **52.64** | 123.63 |
| Hello | 41.58 | 69.89 | 45.40 | **84.69** | 141.10 |
| Topic | 42.48 | 71.86 | 45.27 | **75.28** | 169.24 |
| Natural JSON | 44.04 | 102.23 | 47.11 | **121.76** | 175.33 |
| Schema JSON | 43.65 | 104.65 | 46.53 | **107.63** | HTTP 400 |
| Multilingual | 42.69 | 66.27 | 46.07 | **66.31** | 183.61 |
| Counting 1–200 | 44.95 | 156.08 | 48.41 | **181.31** | 427.29 |

The median weighted target-to-dSpark gain is 78.3% on one RTX and 72.0% on two RTX cards. Two RTX improves target-only weighted throughput by 7.2% and dSpark weighted throughput by 3.4% on this prompt set.

**One-RTX prefill matrix.** Median effective prefill tokens per second after one shape warmup. Each measured request uses a unique marker; retained rows verify the exact parent cache hit.

| Retained base | +1K | +2K | +4K | +8K | +16K | +32K |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3,084 | 4,078 | 6,709 | 7,064 | 7,726 | 7,878 |
| 32K | 2,513 | 3,442 | 5,261 | 6,073 | 6,619 | 6,894 |
| 64K | 2,326 | 3,214 | 4,845 | 5,590 | 6,116 | 6,322 |
| 128K | 2,007 | 2,813 | 4,129 | 4,842 | 5,308 | 5,514 |
| 256K | 1,507 | 2,192 | 3,111 | 3,711 | 4,108 | 4,284 |

**Two-RTX prefill matrix.** The same prompts and token shapes with all twenty encoder expert layers hosted TP2 on the RTX pair.

| Retained base | +1K | +2K | +4K | +8K | +16K | +32K |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4,850 | 6,471 | 7,982 | 8,331 | 8,454 | 8,348 |
| 32K | 3,736 | 5,068 | 5,633 | 6,434 | 6,937 | 7,101 |
| 64K | 3,318 | 4,543 | 5,066 | 5,832 | 6,316 | 6,459 |
| 128K | 2,748 | 3,790 | 4,258 | 4,949 | 5,329 | 5,483 |
| 256K | 1,923 | 2,731 | 3,213 | 3,729 | 4,052 | 4,198 |

Two RTX gains 57–59% on cold +1K/+2K prefills and 18–19% on cold +4K/+8K. At deep retained context with long suffixes, attention dominates; the +16K and +32K cells are close, including small measured losses at 256K retained context.

**Decode over retained context.** Three samples for each of eight content types at every base size. All 240 requests completed with verified exact retained-prefix reuse, and every assessed objective passed.

| Retained base | 1 RTX weighted dSpark | 2 RTX weighted dSpark | 1 RTX completed/reused | 2 RTX completed/reused |
|---:|---:|---:|---:|---:|
| 0 | 76.75 | **93.58** | 24/24 | 24/24 |
| 32K | 71.97 | **85.82** | 24/24 | 24/24 |
| 64K | 70.73 | **86.82** | 24/24 | 24/24 |
| 128K | 68.32 | **84.20** | 24/24 | 24/24 |
| 256K | 67.39 | **78.08** | 24/24 | 24/24 |

**Concurrency scaling.** Each cell is the median of three warm runs. Aggregate timing spans earliest first content through final completion and includes admission gaps. Code and topic are the useful serving curves; counting is the low-entropy ceiling.

| Concurrency | 1 RTX counting | 2 RTX counting | 1 RTX code | 2 RTX code | 1 RTX topic | 2 RTX topic |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 152.65 | **186.83** | 124.18 | **148.83** | 71.93 | **81.13** |
| 2 | 246.29 | **294.15** | 196.77 | **231.63** | 122.43 | **131.94** |
| 4 | 437.02 | **524.04** | 355.92 | **412.60** | 222.72 | **242.58** |
| 8 | 693.73 | **808.47** | 586.58 | **712.09** | 358.31 | **415.86** |
| 16 | 1,161.12 | **1,333.57** | 992.78 | **1,181.49** | 514.96 | **596.14** |

**Mixed traffic.** The fixed code/fable/topic mix uses simultaneous admission and nonce seed 56001. Ranges are retained because request order and output mix create visible noise.

| Concurrency | 1 RTX median (range) | 2 RTX median (range) |
|---:|---:|---:|
| 1 | 113.35 (109.40–128.59) | **151.26 (150.21–154.21)** |
| 2 | 80.83 (72.82–104.85) | **141.54 (123.44–143.40)** |
| 4 | 118.20 (104.83–152.12) | **181.74 (173.78–192.51)** |
| 8 | 124.14 (110.56–170.85) | **203.05 (195.39–216.16)** |
| 16 | 196.46 (195.53–197.69) | **309.06 (300.63–311.28)** |

**Memory and startup.** The loaded measurements were taken immediately after API readiness. The clean release build packaged both local and TP2 expert interfaces and distributed the rebuilt Spark image to all four workers.

| Resource | 1 RTX | 2 RTX |
|---|---:|---:|
| Loaded RTX memory | 95,080 MiB | 95,338 / 95,578 MiB |
| Free RTX memory after readiness | 2,171 MiB | 1,913 / 1,670 MiB |
| Runtime headroom policy | 2 GiB | 800 MiB per GPU |
| Global FP4 source pool | 16,681,077,760 bytes / 18,710,016 logical + 32,768 tail tokens | 13,094,420,480 bytes / 14,680,064 logical + 32,768 tail tokens |
| Standard restart to readiness | 58.05 s | 46.00 s |

The focused dual lifecycle check passed a 32K needle, cold and exact-warm prompt reuse, a retained continuation, eight simultaneous cancellations with eight surviving peers, and post-cancellation recovery. The full qualification suite was intentionally not repeated.

The [Phase 2 performance report](docs/phase2-release-performance.md) records methodology, memory, startup, and qualification scope. [Machine-readable results](docs/phase2-release-performance.json) preserve exact summaries, ranges, controls, and raw-evidence hashes; the [release evidence archive](docs/evidence/phase2-release-performance.tar.gz) contains every raw sample. The one-shot official API comparison is retained as a clearly labeled prior reference.

## Getting started

The release topology requires:

- one Linux amd64 host with one or two RTX PRO 6000 Blackwell 96 GB GPUs;
- four ARM64 DGX Spark hosts reachable over passwordless SSH;
- Docker with the NVIDIA Container Toolkit on all five hosts;
- IP connectivity for the expert fabric and `/dev/infiniband` access for the qualified RoCE path;
- the pinned model snapshot in the same `HF_HOME` layout on every host.

Weights are mounted read-only from the host Hugging Face cache and are not included in the repository or images.

Clone the source and initialize its pinned dependencies:

```bash
git clone --recurse-submodules https://github.com/tpurtell/ds41rt.git
cd ds41rt
git submodule update --init --recursive
```

Edit [`ds41rt.config`](ds41rt.config) for the deployment. At minimum, verify the four `SPARK_*_HOST` names and `SPARK_*_LANE_A` addresses. Configure all four `LANE_B` values for the qualified secondary rail or leave all four empty. Pin `COORDINATOR_GPU_UUID` and `COORDINATOR_GPU_PCI_BUS_ID` on multi-GPU hosts. Keep `MODEL_REVISION` synchronized with the snapshot installed on every machine.

To use the published images, pull the coordinator image locally and the Spark image on each worker:

```bash
docker pull ghcr.io/tpurtell/ds41rt-coordinator:v2
for host in ostrich dodo emu kiwi; do
  ssh "$host" docker pull ghcr.io/tpurtell/ds41rt-spark-expert:v2
done
./run.sh --dry-run
./run.sh
```

To build from the checked-out source instead, run:

```bash
./build.sh
./run.sh --dry-run
./run.sh
```

`build.sh` compiles amd64 coordinator artifacts locally, compiles ARM64 expert artifacts natively on the first Spark, verifies source and dependency identity, and distributes the Spark image to all configured workers. A subset such as `./build.sh --spark-hosts ostrich,dodo` limits build and distribution only; serving still needs all four ranks.

The standard launch listens on port **8000**. Replace an existing deployment with `./run.sh --restart`; stop it with `./stop.sh`. A basic request is:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-ai/DeepSeek-V4.1-Flash",
    "messages": [{"role": "user", "content": "Write a CUDA haiku."}],
    "stream": false
  }'
```

Thinking is enabled at **high** effort when a request omits thinking controls. Pass `"reasoning_effort":"low"`, `"high"`, or `"max"` to select an effort, `"reasoning_effort":"none"` to turn it off, or `"thinking":{"type":"disabled"}` to disable it explicitly. Responses keep reasoning and final content separate.

## Runtime options

Command-line values override [`ds41rt.config`](ds41rt.config) for one launch:

| Option | Default | Purpose |
|---|---:|---|
| `--listen HOST:PORT` | `0.0.0.0:8000` | API bind address |
| `--rtx-gpus auto\|1\|2` | `auto` | Select two feasible peer GPUs automatically, or force a layout |
| `--concurrency N` | `16` | Active requests, 1–16 |
| `--kv-pool-size SIZE` | automatic | Exact global KV/index pool; B, MB, GB, MiB, or GiB |
| `--memory-reservation SIZE` | device plan | Total GPU occupancy ceiling as bytes or a percentage |
| `--prefix-cache-entries N` | `24` | Independent limits for retained completed turns and prompt snapshots |
| `--max-context-tokens N` | `1,048,576` | Per-request context maximum |
| `--max-output-tokens N` | `393,216` | Model output maximum |
| `--prefill-batch-tokens N` | `2,048` | Prefill step size, 80–4,096 |
| `--dspark` / `--no-dspark` | enabled | Enable or disable speculative decoding |
| `--restart` | off | Replace the running five-host deployment |
| `--dry-run` | off | Validate configuration, images, hosts, model, and devices without starting services |

Automatic RTX selection checks physical UUIDs, bidirectional peer reads, the
requested cache or memory ceiling, and available memory. With `--restart`, it
adds back only memory owned by the coordinator container being replaced; other
GPU processes still count against feasibility. The chosen UUID order is passed
through unchanged as logical RTX0/RTX1. Dual mode starts Spark routed experts at
layer 20, while single mode keeps all 40 layers on every Spark.

With no explicit pool setting, single-RTX mode uses a **16.681 GB global pool
for 18,710,016 tokens plus 32,768 private-tail tokens**. Dual-RTX mode targets a
**13.094 GB global pool with 14,712,832 source-token positions**, including
32,768 tail/COW positions. The 24 completed-turn and prompt-snapshot limits
remain independent of the aggregate token budget. `--kv-pool-size` selects the
exact global pool; `--memory-reservation` caps total planned device occupancy;
when both are present, the exact pool must fit under the ceiling. Smaller values
are useful for side-by-side development servers:

```bash
./run.sh --listen 0.0.0.0:18000 --concurrency 2 \
  --max-context-tokens 65536 --max-output-tokens 8192 \
  --kv-pool-size 2GiB --prefix-cache-entries 3 --no-dspark
```

The API supports incremental SSE, cancellation, tools, parallel tool calls, JSON and supported JSON Schema constraints, and up to sixteen images in a prompt. Image input accepts data URLs and bounded HTTP/HTTPS URLs. See the [thinking](docs/release-v1-thinking.md), [tool serving](docs/release-v1-tool-serving.md), [output constraints](docs/release-v1-response-constraints.md), and [vision](docs/release-v1-vision-serving.md) qualification records.

## Engineering

Single-RTX dSpark defaults to adaptive K5 with placement-aware costs calibrated
for the 400 W RTX and four Spark configuration. The binary embeds the profile
and selects costs from the expert layers actually installed on each backend.
Native serving accepts `--dspark-draft-limit 7` for K7;
`DS41RT_ADAPTIVE_COST_PROFILE=legacy` restores the previous cost formula.
The [development comparison](docs/phase2-adaptive-verification.md#single-rtx-default-comparison)
records the policy tradeoffs; the release performance tables above measure the selected K5 default.

The coordinator owns attention, mHC residuals, embeddings, mapped Engram lookup, routers, shared experts, vision, all three dSpark stages, the vocabulary head, sampling, cache ownership, and the API. Dual mode splits this work by dependency across the RTX pair, uses TP2 for all shared experts and encoder routed experts, and partitions vocabulary rows for deterministic parallel greedy selection. Four Sparks retain only decoder routed experts in dual mode and all routed experts in single mode.

Compressed global KV uses FP4 E2M1 values with group-16 E4M3 scales. The 128-token sliding windows remain FP8, and the independent selection index uses its own FP4 format. A token radix shares immutable pages, uses copy-on-write for divergent suffixes, and restores retained target/dSpark state. Partial matches replay no more than the final 128 encoder tokens; exact hits can reuse saved first-token logits. The default keeps 24 completed turns and 24 prompt snapshots under LRU eviction.

The [engineering report](docs/ENGINEERING.md) covers the final kernels, execution lanes, cache transactions, prefix policy, transport, dSpark, vision, constrained decoding, memory planning, and startup path. [`architecture.md`](architecture.md) gives the compact ownership contract.

## Qualification

The clean Phase 2 candidate passed the scoped release qualification:

- matched one/two-RTX target-only and dSpark throughput across eight content types and warm exact counting;
- fresh one/two-RTX prefill matrices covering 30 base/suffix cells and three measured samples per cell;
- retained-prefix decode at 0, 32K, 64K, 128K, and 256K, with exact reuse in all 240 requests;
- warm counting, code, and topic at C1, C2, C4, C8, and C16, plus three mixed-traffic sweeps per layout;
- a focused dual lifecycle check covering a 32K needle, exact prompt reuse, retained continuation, cancellation, survivors, and recovery;
- a clean five-host image build, automatic one/two-RTX selection, replacement accounting, and standard launch on port 8000.

The one-shot official API comparison is a preserved prior measurement. The full vision, tool, and agentic suites were intentionally not repeated; their detailed earlier evidence remains available for the unchanged serving interfaces. The [Phase 2 plan and evidence log](docs/phase2-dual-rtx.md) separates fresh coverage from inherited checks.

## RDMA tools for DGX Spark and RoCE PCs

Move model weights, checkpoints, datasets, and container images between your local AI machines with [Local AI Tap](https://github.com/tpurtell/local-ai-tap):

- **`rdmasync`** — rsync-style file synchronization with RDMA bulk transfers.
- **`rdmapipe`** — stream command output over RDMA into a remote command, using SSH for authentication and orchestration.

Native **ARM64 and AMD64 binary bottles** are available for Linux, including DGX Spark. Homebrew installs the dependencies automatically.

**Install on both endpoints**, with [Homebrew](https://brew.sh/) already installed:

```bash
brew tap tpurtell/local-ai https://github.com/tpurtell/local-ai-tap.git

if brew commands | grep -qx trust; then
  brew trust --tap tpurtell/local-ai
fi

brew install tpurtell/local-ai/rdmapipe tpurtell/local-ai/rdmasync
```

Replace `spark` below with your machine’s SSH hostname or alias.

**Copy model files over RDMA:**

```bash
rdmasync -a --rdma=required \
  --rsync-path=/home/linuxbrew/.linuxbrew/bin/rdmasync \
  ./models/ spark:~/models/
```

**Stream an ARM64 container image directly into a Spark:**

```bash
docker image save --platform linux/arm64 my-ai-image:latest |
  rdmapipe \
    --remote-path=/home/linuxbrew/.linuxbrew/bin/rdmapipe \
    spark -- docker image load
```

The Docker example requires an ARM64 image locally and Docker access on both machines.

Use these tools on a trusted, configured RDMA/RoCE fabric with working Linux drivers and SSH access. Bulk RDMA traffic is not encrypted.

[Installation guide and documentation →](https://github.com/tpurtell/local-ai-tap#readme)

## Thanks

DS41RT builds on the official DeepSeek V4.1 Flash model and reference implementation, NVIDIA CUDA and DGX Spark, [SparkInfer](https://github.com/efeslab/SparkInfer), [XGrammar](https://github.com/mlc-ai/xgrammar), Hugging Face, and the open-source Rust, Python, and CUDA ecosystems.

DS41RT is available under the [MIT License](LICENSE). Dependency licenses and notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
