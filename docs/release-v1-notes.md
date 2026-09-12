# DS41RT v3 release notes

DS41RT v3 is the first public release of the DeepSeek V4.1 runtime. It serves the official DeepSeek V4.1 Flash checkpoint on one RTX PRO 6000 Blackwell coordinator with four DGX Spark expert workers, combining native target inference with local dSpark speculative decoding and agent-oriented cache reuse.

## Key features

- OpenAI-compatible chat completions with streaming, cancellation, usage and cache counters, high-effort thinking by default, tool calls, parallel tools, JSON, and supported JSON Schema constraints.
- Native DeepSeek V4.1 execution with 40 target layers, four tensor-parallel Spark expert ranks, two coordinator execution lanes, and up to sixteen concurrent requests.
- Architectural FP4 compressed global KV, FP8 128-token sliding windows, an independent FP4 learned index, and an exact configurable pool planner.
- Token-radix prefix sharing with compression-boundary reuse, at most 128 tokens of SWA replay, copy-on-write branches, and independent 24-entry prompt and completed-turn caches.
- Three-stage dSpark speculative decoding on the coordinator, reaching 60.68 tok/s on the weighted eight-type mix and 128.03 tok/s on the low-entropy release case.
- Native vision for up to sixteen images per prompt, including content-aware prefix identity, partial replay, cancellation, and bounded preprocessing.
- Official maximums of 1,048,576 context tokens and 393,216 output tokens, with configurable concurrency, retention, prefill, KV pool, and total GPU memory ceilings.
- Reproducible native amd64 coordinator and ARM64 Spark images with pinned model, SparkInfer, XGrammar, source, binary, and license provenance.

The release candidate reaches 2,668 prompt tok/s at its best median prefill cell and 683.70 aggregate decode tok/s at C16. Measurements use a 400 W RTX power limit and standard memory speed with no memory overclock.

Read the [README](https://github.com/tpurtell/ds41rt/blob/v3/README.md), [engineering report](https://github.com/tpurtell/ds41rt/blob/v3/docs/ENGINEERING.md), [performance report](https://github.com/tpurtell/ds41rt/blob/v3/docs/release-v1-performance.md), and [completed release checklist](https://github.com/tpurtell/ds41rt/blob/main/docs/release-v1-checklist.md). The agentic qualification also produced a [playable single-file WebGL Frogger](https://tpurtell.github.io/ds41rt/frogger.html).

Container images:

- `ghcr.io/tpurtell/ds41rt-coordinator:v3` (`linux/amd64`), digest `sha256:0d8a29160924dc62694d65f46e5101bf39071fb28e7611344489dde416bfe950`
- `ghcr.io/tpurtell/ds41rt-spark-expert:v3` (`linux/arm64`), digest `sha256:672f82a1a99872cdc8014811b99c0967e0955c8c3e1e29b91bd48e7b06d3566d`

The v3 and `latest` tags are identical for each role. See the [container qualification](https://github.com/tpurtell/ds41rt/blob/v3/docs/release-v1-build-run.md) for platform manifests and pull-by-digest verification.
