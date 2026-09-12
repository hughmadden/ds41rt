# DS41RT

DS41RT is an attention–FFN-disaggregated engine being adapted for the official
[DeepSeek V4.1 Flash checkpoint](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
on one RTX PRO 6000 Blackwell coordinator and four DGX Spark workers.

Native target-only and dSpark serving are under release qualification. Vision,
long-context prefix replay, deployment and the full performance report remain
release gates. The engine uses the official checkpoint's native representations.

[![DS41RT native execution across one coordinator and four expert workers](docs/native-path-execution.svg)](docs/native-path-execution.svg)

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

Qualification hardware uses a **400 W RTX power limit and standard memory
speed** (no memory overclock). The current driver is 595.91.07; its reported
maximum memory clock is 14,001 MHz. Final results will include clocks observed
under load and the enforced power limit for each campaign.

## Implementation and qualification

[TO_SHIP_V1.md](TO_SHIP_V1.md) tracks implementation, full-checkpoint bring-up,
correctness, performance and release on this host and all four Sparks.
The checkpoint is available on every host; there is no separate modelless phase.
[TO_DELETE_SCAFFOLDING.md](TO_DELETE_SCAFFOLDING.md) tracks temporary committed
code to remove when replacement coverage exists.

Performance targets are 90 tokens/s target-only decode, approximately 270 tokens/s
with optimized dSpark, and 8,000 tokens/s prefill for prompts of 8K tokens or more;
these are unqualified targets, not current performance claims.

Implemented engram pieces include exact tokenizer compression and hashes,
transactional request history, mapped tables, bounded page prefetch, native FP8
row dequantization, and fused CUDA gating.
They still need full execution-path integration.
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

The migration retains the OpenAI-compatible API, streaming, reasoning controls,
tool calls, JSON/JSON Schema constraints, admission, continuous batching,
prefix reuse, cancellation, metrics, health/readiness, and restart tooling.
V4.1 native vision and the dSpark-disabled `-full` control are part of the release
scope and must pass the full-model serving gates.

There are no qualified V4.1 end-to-end throughput claims yet.

DS41RT is released under the [MIT License](LICENSE); dependencies retain their
own terms as recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
