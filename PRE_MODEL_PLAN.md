# Pre-model implementation and qualification

Goal 1 ends when the official checkpoint can be loaded and executed by the qualified full stack, without requiring the ongoing model download or access to emu and kiwi.

- [x] Locate and pin the official configuration, reference implementation, and technical report at revision `df42c109f1defefcbfcedbe7d905718a12266e40`.
- [ ] Complete the architecture and tensor-layout audit with explicit reference-to-production mappings.
- [ ] Repair and qualify build, WIP, run, restart, stop, and container export flows without requiring a manually prepared source manifest.
- [x] Rename the engine, packages, configuration, container names, and deployment paths from ds4 to ds41.
- [ ] Remove obsolete model execution paths, alternate weight formats, quantizers, dependencies, optimization notes, and inapplicable benchmarks while preserving serving features.
- [ ] Implement strict official-checkpoint configuration, tensor inventory, representation validation, and memory-budgeted loading from the alternate Hugging Face cache.
- [ ] Implement CED encoder/decoder execution with shared KV ownership, chunked prefill, exact replay, and explicitly qualified bounded replay.
- [ ] Implement CSA2 ratio-2 and ratio-1 cache production, FP8 window KV, FP4 compressed KV, and source/reindex/reuse modes.
- [ ] Implement hierarchical candidate-block selection and sparse top-512 attention with causal visibility and large-pool addressing checks.
- [ ] Implement and fuse single-pass mHC, normalization, RoPE, projections, and modality-aware routing against the official numerical reference.
- [x] Implement deterministic engram tokenizer compression, prime layouts, hashes, image barriers, and per-request history.
- [ ] Memory-map native engram weights and scales with bounded asynchronous prefetch, deduplication, staging, and request-safe cancellation.
- [x] Add CUDA engram row dequantization and fused gate/residual kernels with native and Rust entry points qualified on both RTX GPUs.
- [ ] Wire early engram prefetch for decode, speculative verification, and batched prefill with rollback-safe history.
- [ ] Implement native FP4 backbone expert tensor parallelism through the AFD transport with appropriate SparkInfer/b12x kernels.
- [ ] Implement native image processing, ViT, aligner, image-span embeddings, and vision routing through the serving API.
- [ ] Place all dSpark stages and experts on the coordinator RTX and include their native weights, caches, and concurrency-16 workspace in its memory budget.
- [ ] Implement all three dSpark stages, Markov and confidence heads, proposal verification, sampling, acceptance, and state rollback.
- [ ] Extend admission, scheduling, graph shapes, cache ownership, and cancellation to 16 concurrent requests with alternating waves.
- [ ] Qualify native numerical paths and CUDA graph replay on both local RTX GPUs and on ostrich and dodo without using emu or kiwi.
- [ ] Qualify the integrated modelless stack with deterministic nonzero fixtures, transport, serving regressions, and allocation/capacity checks.
- [ ] Record required temporary fixtures in TO_DELETE_SCAFFOLDING.md and preserve reusable qualification tooling.
- [ ] Commit and push engine changes and required SparkInfer/b12x changes to their fork main branches with matching dependency pins.
- [ ] Audit every pre-model requirement and document exact bring-up commands, remaining checkpoint-dependent gates, and hardware evidence for handoff.

Progress evidence: `docs/ds41-build-qualification.md` records the rename and partial-fleet build checks; full serving launch and integrated GPU qualification remain open, with successful partial-fleet container evidence recorded below.

Engram progress: the loader now provides mapped native table/scale access, bounded cancellable background page advice, and eager-load rejection, with scheduler integration, device staging, and hardware performance qualification still pending.

Addressing evidence: `scripts/qualify-ds41-engram.py` matches the pinned official token map and 128 hash/acceptance batches across 16 request histories, while 119 core and 58 loader tests pass.

Cleanup progress: obsolete Pro benchmark data, plots, deployment claims, and architecture diagrams were removed, while alternate-format execution and quantization tooling still await replacement/removal.

FP8 progress: native 32x32 checkpoint packing, independent K32 scales, and exact two/four-way FP32 reduction pass on both RTX GPUs and ostrich/dodo, with variable-M Spark prefill and preplanned split workspace still open as recorded in `docs/ds41-fp8-qualification.md`.

Checkpoint contract progress: a strict typed reader validates every official nested text, vision, engram, dSpark, and quantization field against the pinned configuration, with 60 loader tests passing and integration into the replacement tensor catalog and execution path still pending.

Container evidence: `./build.sh --spark-hosts ostrich,dodo` now completes from a dirty checkout with an automatic manifest, native coordinator/Spark builds, artifact exports, and distribution to both available Sparks, while full run/restart/stop qualification remains open.

Native loader progress: the 96,085-tensor checkpoint contract now validates physical storage and exposes bounded Spark TP4 staging, RTX-only dSpark staging, and host-only engram mapping/prefetch, with native storage budgets recorded and execution workspace budgeting/integration still pending.

Native expert progress: `silu_v41` now preserves official projection rounding and intermediate router weighting with FP32 per-route accumulation, passing six native/graph cases on both RTX GPUs and ostrich/dodo, while TP-global rounding and runtime integration remain open in `docs/ds41-expert-qualification.md`.

TP arithmetic progress: native scratch-backed FP32 route outputs and TP4-before-BF16 reduction now match full-width experts on both RTX GPUs and ostrich/dodo, while FP32 routing/payload transport and runtime wiring remain open in `docs/ds41-tp4-qualification.json`.
