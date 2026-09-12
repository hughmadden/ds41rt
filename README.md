# DS41RT

DS41RT is an attention–FFN-disaggregated engine for the official
[DeepSeek V4.1 Flash checkpoint](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
on one RTX PRO 6000 Blackwell coordinator and four DGX Spark workers.

Qualification measurements use a **400 W RTX power limit and standard memory
speed (no memory overclock)**.

[![DS41RT native execution across one coordinator and four expert workers](docs/native-path-execution.svg)](docs/native-path-execution.svg)

Native target-only and dSpark serving are under first-release qualification.
The [release checklist](docs/release-v1-checklist.md) tracks the remaining
performance, correctness, deployment and publication gates.

## Execution design

The coordinator owns the causal encoder/decoder attention, vision encoder,
engram projections and gating, shared experts, routing, cache, sampling, API,
and all three dSpark stages including their experts.
The four Sparks execute intermediate-dimension tensor-parallel slices of the
backbone's routed experts through the existing AFD transport.

Engram embedding tables and scales remain memory-mapped on the host, with
bounded background prefetch driven by known token IDs and gathered rows staged
for GPU execution.
The serving target is 16 concurrent requests with alternating waves.

See [architecture.md](architecture.md) for ownership and execution contracts and
[the architecture audit](docs/ds41-architecture-audit.md) for reference details.

## Implementation and qualification

[TO_SHIP_V1.md](TO_SHIP_V1.md) tracks implementation, full-checkpoint bring-up,
correctness, performance and release on this host and all four Sparks.
The checkpoint is available on every host; there is no separate modelless phase.
[TO_DELETE_SCAFFOLDING.md](TO_DELETE_SCAFFOLDING.md) tracks temporary committed
code to remove when replacement coverage exists.

Implemented engram pieces include exact tokenizer compression and hashes,
transactional request history, mapped tables, bounded page prefetch, native FP8
row dequantization, and fused CUDA gating.
Qualification evidence lives in [the build record](docs/ds41-build-qualification.md)
and [the engram CUDA record](docs/ds41-engram-cuda-qualification.json).

## Build tooling

The build scripts use native x86-64 and ARM64 builds, preserve source provenance,
and generate source manifests automatically for dirty checkouts.
To restrict a build to the available Spark hosts:

```bash
./build.sh --spark-hosts ostrich,dodo
```

This restricts build and image distribution targets; it does not change the
four-rank serving topology.
Full container and launch qualification remains in `TO_SHIP_V1.md`.
Use `./build.sh --help`, `./wip.sh --help`, and `./run.sh --help` for script options.

The model download is separate from builds. Set `HF_HOME` to the host checkpoint
cache when using a location other than the Hugging Face default.
No weights are bundled with the repository or containers.

## Serving scope

Native requests default to thinking enabled at **high** effort. Set
`reasoning_effort` to override the effort, or use `"thinking":{"type":"disabled"}`
to disable reasoning explicitly. The pinned V4.1 encoder maps high to 75/100.
See [reasoning controls and qualification](docs/release-v1-thinking.md).

`serve-native` defaults to a 1,048,576-token context and a 393,216-token output
maximum. Omitted output limits use the maximum; each request clamps to its
remaining context. Use `--max-context-tokens` and `--max-output-tokens` for smaller
development launches. See [limit behavior and verification](docs/release-v1-model-limits.md).

Compressed source KV uses FP4 E2M1 with group-16 E4M3 scales. Sliding-window KV
uses FP8 E4M3 with group-32 E8M0 scales; the independent index remains FP4.
See [cache integration and qualification](docs/release-v1-compressed-serving.md).

Native serving defaults to 16 concurrent requests, global KV capacity for 24
configured-context equivalents, and 24 retained completed turns. A separate
24-entry prompt cache preserves fast repeats. `--concurrency`,
`--prefix-cache-entries`, `--kv-pool-size` and `--memory-reservation` override these
settings. See [pool controls](docs/release-v1-pool.md) and
[completed-turn retention](docs/release-v1-retained-turns.md). Standard `run.sh`
wiring remains under qualification.

The migration retains the OpenAI-compatible API, streaming, reasoning controls,
tool calls, JSON/JSON Schema constraints, admission, continuous batching,
prefix reuse, cancellation, metrics, health/readiness, and restart tooling.
Native `serve-native` now accepts up to sixteen images in both target-only and
dSpark modes, including image-aware prefix reuse. See the
[vision serving qualification](docs/release-v1-vision-serving.md).
Standard launch-script integration and the remaining release gates are still open.

Final release throughput numbers remain pending. Development comparisons are
linked from the release checklist.

DS41RT is released under the [MIT License](LICENSE); dependencies retain their
own terms as recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
