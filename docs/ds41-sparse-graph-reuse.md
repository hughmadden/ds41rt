# Decode timing and sparse-attention graph reuse

Opt-in `RUST_LOG=info,ds41rt::timing=debug` records wall time for request preparation, the target pass, host logit sampling, commit, cache production, index selection, attention/output/FFN preparation, routing, dispatch, shared FFN and collection/reduction. The component methods drain their CUDA work where required by their existing contracts. These are serialized host-observed intervals, not an independent CUDA-event breakdown. Layer totals omit between-layer query preparation and the final head, which remain included in the target-pass time.

The initial trace showed approximately 171 ms in a single-row target pass, 5.5 ms in host sampling and 0.6 ms in commit. Median expert collection was 2.77 ms per layer and attention/output/FFN preparation was 0.93 ms per layer. Expert collection includes remote execution, response delivery, host-to-device copies and native reduction; it does not isolate those costs.

## Change

Sparse attention had one captured graph shared by all 40 layers. Each layer transition replaced it. The owner now retains one graph per layer and still compares all existing buffer pointers, sizes, selected storage and launch geometry before replay. Replacing a layer's shape destroys only that graph; complete teardown drains and attempts destruction of every retained graph.

Retaining graphs alone did not improve the short-prompt measurement: the explicit causal-window width still changed every token below position 128. Width zero now selects a dynamic mode in the existing native kernel, computing `min(last_request_position + 1, 128)` from the last metadata row. The caller validates ascending positions and launches each request separately. This produces the same width as the former host maximum, preserving key order and floating-point reduction grouping. Positive widths retain the original explicit-width API. Window/source metadata continues to be copied before replay.

The retained full-target test now runs three default cycles: 16 five-token prefills and two subsequent single-token decode batches. All three FP32 vocabulary-logit arrays (8,273,920 bytes each) and greedy-ID arrays are byte-exact against a run made before either graph change. Both accepted-prefix commits and the further decode commit pass. This is regression evidence against the prior native implementation; it does not independently qualify every layer's arithmetic.

## Measurements

With layer retention but explicit widths, the repeated short-prompt API runs measured 5.57–5.62 TPS. With dynamic width and retention they measured 6.04–6.10 TPS (the first run was 5.99 TPS). The instrumented median single-row step fell from 178.64 ms to 165.15 ms; median attention/output/FFN preparation fell from 944 to 612.5 microseconds per layer. Expert collection remained approximately 2.82 ms per layer and is the next profiling priority. These are debug builds, serialized greedy text, TCP and a tiny counting prompt, not release benchmarks or target-speed qualification.

The API qualifier still returns exact `4` and the 1-to-20 sequence, requested SSE usage and completion, rejects nonzero temperature and recovers after a client disconnect. Trace counts and raw event/timing evidence are in [the record](ds41-sparse-graph-reuse.json). Original single-slot runs did not emit graph-capture events, so their capture count is unknown rather than zero.

The native API now handles SIGTERM/Ctrl-C through graceful HTTP shutdown and joins the CUDA-owning thread. An idle development container stopped successfully with exit zero in under 0.5 seconds. In-flight shutdown and broader lifecycle behavior still need tests.

## Remaining work

Break expert collection into Spark execution and transfer/reduction costs, check capacity-1/16/80 expert kernel behavior for small live batches, remove host sampling overhead, and measure optimized builds before proceeding with broader kernel and scheduling work. Concurrency-16 API batching, dSpark, vision, long context, startup optimization and release launcher migration remain open.
