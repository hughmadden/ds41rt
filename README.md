# DS41RT

DS41RT serves the official [DeepSeek V4.1 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) checkpoint across one RTX PRO 6000 Blackwell coordinator and four DGX Spark expert workers. It combines native target execution, local dSpark speculative decoding, token-level prefix reuse, tools, constrained output, and vision in one OpenAI-compatible service.

All reported RTX measurements use an enforced **400 W power limit** and **standard 14,001 MHz maximum memory speed with no memory overclock**. The loaded memory clock reached 13,365 MHz; the RTX driver was 595.91.07. The four GB10 workers used driver 580.159.03.

[![DS41RT native execution across one coordinator and four expert workers](docs/native-path-execution.svg)](docs/native-path-execution.svg)

## Performance

The release candidate uses architectural FP4 compressed KV and the standard C16 launch. Local target-only and dSpark workloads ran sequentially. Throughput tests use temperature zero and thinking disabled; tool evaluation uses high thinking.

**Performance qualification is reopened:** development-to-release prefill regression remains unresolved. These candidate measurements do not establish parity with the earlier ~7K prefill and ~145 counting decode results.

**Headline results.** Median throughput, cache size, memory use, and startup time for the measured release candidate.

| Measurement | Result |
|---|---:|
| Best median prefill, 0 base + 32K new | 2,667.95 tok/s |
| Best observed prefill sample | 2,738.47 tok/s |
| Low-entropy dSpark decode, counting 1–200 warm median | 127.70 tok/s |
| Weighted eight-type target-only median | 37.89 tok/s |
| Weighted eight-type dSpark median | 60.68 tok/s |
| dSpark gain on weighted mix | 60.16% |
| C16 aggregate warm decode median | 683.70 tok/s |
| Default architectural cache | 20.93 GiB |
| Warmed C16 coordinator process | 64.05 GiB |
| Clean standard-launch readiness | 55–56 s |

**Eight content types and counting.** Local eight-case results use five samples per mode; official Flash uses one request per case. Quality columns count passed checks. Official JSON Schema returned HTTP 400, so no full official weighted score is available.

| Case | Target tok/s | dSpark tok/s | Official Flash tok/s (one request) | Target quality | dSpark quality | Official quality |
|---|---:|---:|---:|---:|---:|---:|
| Code | 38.35 | 99.57 | 345.90 | 5/5 | 5/5 | 1/1 |
| Math | 37.79 | 96.52 | 285.33 | 5/5 | 5/5 | 1/1 |
| Fable | 37.89 | 38.32 | 123.63 | 1/5 | 0/5 | 0/1 |
| Hello | 37.64 | 50.00 | 141.10 | 5/5 | 5/5 | 1/1 |
| Topic | 37.66 | 56.82 | 169.24 | 2/5 | 1/5 | 0/1 |
| Natural JSON | 38.05 | 81.74 | 175.33 | 5/5 | 5/5 | 1/1 |
| Schema JSON | 37.78 | 70.74 | HTTP 400 | 5/5 | 5/5 | N/A |
| Multilingual | 37.58 | 57.20 | 183.61 | 5/5 | 5/5 | 1/1 |
| Counting 1–200 | Not measured | **127.70** | **427.29** | N/A | 3/3 warm | 1/1 |

Counting is outside the weighted score: local dSpark uses three warm samples; official Flash uses one fresh request. Both pass the 1–200 sequence check. Official timings include network streaming on unknown provider hardware and use provider-reported token counts.

**Prefill.** Median new prompt tokens/s above each retained base, target-only; two timed samples per cell after one warmup.

| Retained base | +1K | +2K | +4K | +8K | +16K | +32K |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1,472 | 1,815 | 2,490 | 2,233 | 2,573 | **2,668** |
| 32K | 1,384 | 1,719 | 2,440 | 2,595 | 2,545 | 2,654 |
| 64K | 1,308 | 1,649 | 2,388 | 2,568 | 2,539 | 2,647 |
| 128K | 1,209 | 1,535 | 2,256 | 2,507 | 2,594 | 2,598 |
| 256K | 989 | 1,346 | 2,001 | 1,902 | 2,326 | 2,264 |

**Decode over retained context.** Weighted dSpark tokens/s across eight content types, with two samples per type and verified prefix reuse at each base.

| Retained base | Weighted dSpark tok/s | Quality passes |
|---:|---:|---:|
| 0 | 59.59 | 13/16 |
| 32K | 53.75 | 15/16 |
| 64K | 55.06 | 15/16 |
| 128K | 54.04 | 12/16 |
| 256K | 51.87 | 13/16 |

**Concurrency scaling.** Three samples per concurrency with a fully cached prompt and exact 599-token counting output. Aggregate timing includes scheduler admission gaps.

| Concurrency | Median aggregate tok/s | Range | Scale vs C1 |
|---:|---:|---:|---:|
| 1 | 124.81 | 123.20–124.98 | 1.00× |
| 2 | 176.58 | 174.10–176.86 | 1.41× |
| 4 | 288.89 | 288.72–289.02 | 2.31× |
| 8 | 416.34 | 411.79–416.79 | 3.34× |
| 16 | **683.70** | 683.64–689.58 | **5.48×** |

The [performance report](docs/release-v1-performance.md) provides methodology, artifact identities, quality misses, and the memory, startup, needle-retrieval, and agentic results. [Machine-readable results](docs/release-v1-performance.json) and [raw evidence](docs/evidence/native-release-performance.tar.gz) preserve samples, inputs, outputs, cache counters, hardware state, and errors.

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
docker pull ghcr.io/tpurtell/ds41rt-coordinator:v3
for host in ostrich dodo emu kiwi; do
  ssh "$host" docker pull ghcr.io/tpurtell/ds41rt-spark-expert:v3
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

With no explicit pool setting, the planner reserves source capacity for sixteen maximum-context active requests plus eight additional maximum-context equivalents. `--kv-pool-size` selects the exact global pool; `--memory-reservation` caps total planned device occupancy; when both are present, the exact pool must fit under the ceiling. Smaller values are useful for side-by-side development servers:

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

## Thanks

DS41RT builds on the official DeepSeek V4.1 Flash model and reference implementation, NVIDIA CUDA and DGX Spark, [SparkInfer](https://github.com/efeslab/SparkInfer), [XGrammar](https://github.com/mlc-ai/xgrammar), Hugging Face, and the open-source Rust, Python, and CUDA ecosystems.

DS41RT is available under the [MIT License](LICENSE). Dependency licenses and notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
