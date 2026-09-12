# DS41RT

DS41RT serves the official [DeepSeek V4.1 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) checkpoint across one RTX PRO 6000 Blackwell coordinator and four DGX Spark expert workers. It combines native target execution, local dSpark speculative decoding, token-level prefix reuse, tools, constrained output, and vision in one OpenAI-compatible service.

All reported RTX measurements use an enforced **400 W power limit** and **standard 14,001 MHz maximum memory speed with no memory overclock**. The loaded memory clock reached 13,365 MHz; the RTX driver was 595.91.07. The four GB10 workers used driver 580.159.03.

[![DS41RT native execution across one coordinator and four expert workers](docs/native-path-execution.svg)](docs/native-path-execution.svg)

## Performance

The release candidate uses corrected architectural FP4 compressed KV and the standard C16 launch. Target-only and dSpark workloads ran sequentially across the shared workers.

| Measurement | Result |
|---|---:|
| Best median prefill, 0 base + 32K new | **2,668 prompt tok/s** |
| Low-entropy dSpark decode, counting 1–200 | **127.70 tok/s** |
| Weighted eight-type target decode | 37.89 tok/s |
| Weighted eight-type dSpark decode | **60.68 tok/s** |
| dSpark gain on the weighted mix | 60.16% |
| C16 aggregate warm decode | **683.70 tok/s** |
| Default architectural cache | 20.93 GiB |
| Warmed coordinator process | 64.05 GiB |
| Clean standard-launch readiness | 55–56 s |

The [performance report](docs/release-v1-performance.md) contains every sample, the 30-cell prefill matrix, retained-context decode through 256K, concurrency scaling, memory, startup, 1.04M-token needle retrieval, and three high-thinking tool-eval runs. The [machine-readable results](docs/release-v1-performance.json) and [raw evidence](docs/evidence/native-release-performance.tar.gz) preserve inputs, outputs, cache counters, hardware state, and errors.

The [eight-case comparison table](docs/release-v1-performance.md#eight-content-types-and-low-entropy-decode)
also includes one official `deepseek-flash` API reference run. Seven cases
completed; JSON Schema returned HTTP 400.

**Performance qualification is reopened:** development-to-release prefill
regression remains unresolved. The published candidate figures above do not
establish parity with the earlier ~7K prefill and ~145 counting decode results.

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
