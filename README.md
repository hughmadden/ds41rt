# DS41RT

DS41RT serves the official [DeepSeek V4.1 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) checkpoint across one RTX PRO 6000 Blackwell coordinator and four DGX Spark expert workers. It combines native target execution, local dSpark speculative decoding, token-level prefix reuse, tools, constrained output, and vision in one OpenAI-compatible service.

All reported RTX measurements use an enforced **400 W power limit** and **standard 14,001 MHz maximum memory speed with no memory overclock**. The loaded memory clock reached 13,365 MHz; the RTX driver was 595.91.07. The four GB10 workers used driver 580.159.03.

[![DS41RT native execution across one coordinator and four expert workers](docs/native-path-execution.svg)](docs/native-path-execution.svg)

## Performance

The corrected standard `v1` build uses architectural FP4 compressed KV, the standard C16 launch, and the qualified fused Spark expert kernels. Local target-only and dSpark workloads ran sequentially. Throughput tests use temperature zero and thinking disabled; tool evaluation uses high thinking. The earlier release regression is fixed: standard-build prefill rose from 2,668 to 7,743 tok/s at the headline cell, and warm counting decode rose from 127.70 to 150.51 tok/s.

**Headline results.** Median throughput, the 24-context cache policy, memory use, and measured clean-launch time.

| Measurement | Result |
|---|---:|
| Best median prefill, 0 base + 32K new | **7,743.47 tok/s** |
| Best observed prefill sample | **8,023.26 tok/s** |
| Low-entropy target-only decode, counting 1–200 warm median | 43.08 tok/s |
| Low-entropy dSpark decode, counting 1–200 warm median | **150.51 tok/s** |
| Weighted eight-type target-only median | 41.42 tok/s |
| Weighted eight-type dSpark median | **70.43 tok/s** |
| dSpark gain on weighted mix | 70.04% |
| C16 aggregate warm decode median | **742.91 tok/s** |
| Benchmark architectural cache (prior default) | 20.93 GiB for 25,165,824 tokens total (24 × 1,048,576) |
| Exact prompt / completed-turn retention | 24 / 24 entries |
| Warmed C16 coordinator process | 64.23 GiB |
| Clean standard dSpark launch readiness | 56.77 s |

**Eight content types and counting.** Three local samples per mode and the preserved one-request official reference. Completion counts report serving success; open-ended prose is unscored. The official schema request returned HTTP 400, so it has no full weighted aggregate.

| Case | Target tok/s | dSpark tok/s | Official Flash tok/s (one request) | Target completed | dSpark completed | Official completed |
|---|---:|---:|---:|---:|---:|---:|
| Code | 42.62 | 113.73 | 345.90 | 3/3 | 3/3 | 1/1 |
| Math | 41.08 | 109.54 | 285.33 | 3/3 | 3/3 | 1/1 |
| Fable | 40.42 | 47.10 | 123.63 | 3/3 | 3/3 | 1/1 |
| Hello | 39.96 | 57.14 | 141.10 | 3/3 | 3/3 | 1/1 |
| Topic | 41.24 | 63.48 | 169.24 | 3/3 | 3/3 | 1/1 |
| Natural JSON | 42.34 | 93.24 | 175.33 | 3/3 | 3/3 | 1/1 |
| Schema JSON | 41.89 | 85.78 | HTTP 400 | 3/3 | 3/3 | 0/1 (HTTP 400) |
| Multilingual | 40.99 | 62.70 | 183.61 | 3/3 | 3/3 | 1/1 |
| Counting 1–200 | **43.08** | **150.51** | **427.29** | 3/3 warm | 3/3 warm | 1/1 |

Counting is outside the weighted score. Local values are three warm exact-sequence samples after one prime. The official API was called once and was not rerun.

**Prefill.** Median new prompt tokens/s above each retained base, target-only; three timed samples per cell after one warmup.

| Retained base | +1K | +2K | +4K | +8K | +16K | +32K |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2,818 | 3,757 | 6,922 | 7,414 | 7,660 | **7,743** |
| 32K | 2,440 | 3,319 | 6,055 | 6,736 | 7,086 | 7,119 |
| 64K | 2,251 | 3,112 | 5,557 | 6,255 | 6,594 | 6,768 |
| 128K | 1,942 | 2,724 | 4,724 | 5,432 | 5,785 | 5,961 |
| 256K | 1,477 | 2,148 | 3,565 | 4,123 | 4,444 | 4,612 |

**Decode over retained context.** Weighted dSpark tokens/s across eight content types, with three samples per type and verified prefix reuse at each base.

| Retained base | Weighted dSpark tok/s | Completed with verified cache reuse |
|---:|---:|---:|
| 0 | 73.43 | 24/24 |
| 32K | 68.08 | 24/24 |
| 64K | 64.82 | 24/24 |
| 128K | 62.71 | 24/24 |
| 256K | 61.27 | 24/24 |

**Concurrency scaling.** Three samples per concurrency with a fully cached prompt and exact 599-token counting output. Aggregate timing includes scheduler admission gaps.

| Concurrency | Median aggregate tok/s | Range | Scale vs C1 |
|---:|---:|---:|---:|
| 1 | 149.16 | 148.89–149.24 | 1.00× |
| 2 | 188.47 | 187.82–188.66 | 1.26× |
| 4 | 325.90 | 323.81–326.08 | 2.18× |
| 8 | 467.10 | 463.93–467.14 | 3.13× |
| 16 | **742.91** | 735.48–745.69 | **4.98×** |

The [performance report](docs/release-v1-performance.md) provides methodology, artifact identities, per-case retained-context results, memory, startup, needle retrieval, and agentic results. [Machine-readable results](docs/release-v1-performance.json) and [raw evidence](docs/evidence/native-release-performance.tar.gz) preserve samples, inputs, outputs, cache counters, hardware state, and errors.

## Getting started

The release topology requires:

- one Linux amd64 host with an RTX PRO 6000 Blackwell 96 GB GPU;
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
docker pull ghcr.io/tpurtell/ds41rt-coordinator:v1
for host in ostrich dodo emu kiwi; do
  ssh "$host" docker pull ghcr.io/tpurtell/ds41rt-spark-expert:v1
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

With no explicit pool setting, the planner starts from sixteen maximum-context active requests plus two additional maximum-context equivalents, then trades enough global KV pages for preallocated snapshot storage. At default C16 with dSpark, this gives a **16.681 GB global pool for 18,710,016 tokens plus private-tail headroom**, with 139.4 MiB of snapshot arenas. The 24 completed-turn and prompt-snapshot limits remain independent of this aggregate token budget. The performance tables above retain the measured prior 24-context configuration until the new placement is qualified. `--kv-pool-size` selects the exact global pool; `--memory-reservation` caps total planned device occupancy; when both are present, the exact pool must fit under the ceiling. Smaller values are useful for side-by-side development servers:

```bash
./run.sh --listen 0.0.0.0:18000 --concurrency 2 \
  --max-context-tokens 65536 --max-output-tokens 8192 \
  --kv-pool-size 2GiB --prefix-cache-entries 3 --no-dspark
```

The API supports incremental SSE, cancellation, tools, parallel tool calls, JSON and supported JSON Schema constraints, and up to sixteen images in a prompt. Image input accepts data URLs and bounded HTTP/HTTPS URLs. See the [thinking](docs/release-v1-thinking.md), [tool serving](docs/release-v1-tool-serving.md), [output constraints](docs/release-v1-response-constraints.md), and [vision](docs/release-v1-vision-serving.md) qualification records.

## Engineering

The coordinator owns attention, mHC residuals, embeddings, mapped Engram lookup, routers, shared experts, vision, all three dSpark stages, the vocabulary head, sampling, cache ownership, and the API. Four Sparks hold tensor-parallel slices of the backbone routed experts and return reduced expert contributions over persistent transport buffers.

Compressed global KV uses FP4 E2M1 values with group-16 E4M3 scales. The 128-token sliding windows remain FP8, and the independent selection index uses its own FP4 format. A token radix shares immutable pages, uses copy-on-write for divergent suffixes, and restores retained target/dSpark state. Partial matches replay no more than the final 128 encoder tokens; exact hits can reuse saved first-token logits. The default keeps 24 completed turns and 24 prompt snapshots under LRU eviction.

The [engineering report](docs/ENGINEERING.md) covers the final kernels, execution lanes, cache transactions, prefix policy, transport, dSpark, vision, constrained decoding, memory planning, and startup path. [`architecture.md`](architecture.md) gives the compact ownership contract.

## Qualification

The final candidate passed:

- cold, partial, exact, divergent, multi-chunk, cancellation, C16, and 24-turn prefix-cache cases;
- 1.04M-token needle retrieval in target and dSpark modes, both fresh and exact-reuse;
- native vision through sixteen images, including identity changes, reorder, reuse, cancellation, and concurrency;
- Unicode/token-boundary streaming, tools, JSON Schema constraints, and recovery paths;
- three C16 high-thinking tool-eval campaigns scoring 153, 159, and 156 of 176 points;
- a clean five-host image build and standard `run.sh` launch;
- a dsh coding run that generated and repaired a [playable single-file WebGL Frogger](https://tpurtell.github.io/ds41rt/frogger.html).

The [release checklist](docs/release-v1-checklist.md) links each detailed report and its preserved evidence.

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
