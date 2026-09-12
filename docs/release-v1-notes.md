# DS41RT v1 release notes

DS41RT v1 is the first public release of the DeepSeek V4.1 runtime. It serves the official DeepSeek V4.1 Flash checkpoint on one RTX PRO 6000 Blackwell coordinator with four DGX Spark expert workers, combining native target inference with local dSpark speculative decoding and agent-oriented cache reuse.

## Key features

- OpenAI-compatible chat completions with streaming, cancellation, usage and cache counters, high-effort thinking by default, tool calls, parallel tools, JSON, and supported JSON Schema constraints.
- Native DeepSeek V4.1 execution with 40 target layers, four tensor-parallel Spark expert ranks, two coordinator execution lanes, and up to sixteen concurrent requests.
- Architectural FP4 compressed global KV, FP8 128-token sliding windows, an independent FP4 learned index, and an exact configurable pool planner.
- Token-radix prefix sharing with compression-boundary reuse, at most 128 tokens of SWA replay, copy-on-write branches, and independent 24-entry prompt and completed-turn caches.
- Three-stage dSpark speculative decoding reaching 70.43 tok/s on the weighted eight-type mix, 150.51 tok/s on exact counting, and 742.91 aggregate tok/s at C16.
- Qualified fused Spark expert kernels that restore 7,743 prompt tok/s median at the headline 32K prefill cell.
- Native vision for up to sixteen images per prompt, including content-aware prefix identity, partial replay, cancellation, and bounded preprocessing.
- Official maximums of 1,048,576 context tokens and 393,216 output tokens, with configurable concurrency, retention, prefill, KV pool, and total GPU memory ceilings.
- Reproducible native amd64 coordinator and ARM64 Spark images with pinned model, SparkInfer, XGrammar, source, binary, and license provenance.

Measurements use a 400 W RTX power limit and standard 14,001 MHz maximum memory speed with no memory overclock. Read the [README](https://github.com/tpurtell/ds41rt/blob/v1/README.md), [engineering report](https://github.com/tpurtell/ds41rt/blob/v1/docs/ENGINEERING.md), [performance report](https://github.com/tpurtell/ds41rt/blob/v1/docs/release-v1-performance.md), and [completed release checklist](https://github.com/tpurtell/ds41rt/blob/v1/docs/release-v1-checklist.md). The agentic qualification also produced a [playable single-file WebGL Frogger](https://tpurtell.github.io/ds41rt/frogger.html).

Container images:

- `ghcr.io/tpurtell/ds41rt-coordinator:v1` (`linux/amd64`), digest `sha256:67f2954e18f69b39f8fbb68164f7d9e2b8f4c4b9e3242281ecfcb8afec8552e9`
- `ghcr.io/tpurtell/ds41rt-spark-expert:v1` (`linux/arm64`), digest `sha256:1f1bff295a1d112c8a2fb80918b5abcafcf10eec4936717e234d3d727635a0be`

The `v1` and `latest` tags are identical for each role. See the [container qualification](https://github.com/tpurtell/ds41rt/blob/v1/docs/release-v1-build-run.md) for manifests, source labels, and verification evidence.
