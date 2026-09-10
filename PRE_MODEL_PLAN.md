# Pre-model implementation and qualification

Goal 1 ends when the official checkpoint can be loaded and executed by the qualified full stack, without requiring the ongoing model download or access to emu and kiwi.

- [x] Locate and pin the official configuration, reference implementation, and technical report at revision `df42c109f1defefcbfcedbe7d905718a12266e40`.
- [ ] Complete the architecture and tensor-layout audit with explicit reference-to-production mappings.
- [ ] Repair and qualify build, WIP, run, restart, stop, and container export flows without requiring a manually prepared source manifest.
- [ ] Rename the engine, packages, configuration, container names, and deployment paths from ds4 to ds41.
- [ ] Remove obsolete model execution paths, alternate weight formats, quantizers, dependencies, optimization notes, and inapplicable benchmarks while preserving serving features.
- [ ] Implement strict official-checkpoint configuration, tensor inventory, representation validation, and memory-budgeted loading from the alternate Hugging Face cache.
- [ ] Implement CED encoder/decoder execution with shared KV ownership, chunked prefill, exact replay, and explicitly qualified bounded replay.
- [ ] Implement CSA2 ratio-2 and ratio-1 cache production, FP8 window KV, FP4 compressed KV, and source/reindex/reuse modes.
- [ ] Implement hierarchical candidate-block selection and sparse top-512 attention with causal visibility and large-pool addressing checks.
- [ ] Implement and fuse single-pass mHC, normalization, RoPE, projections, and modality-aware routing against the official numerical reference.
- [ ] Implement deterministic engram tokenizer compression, prime layouts, hashes, image barriers, and per-request history.
- [ ] Memory-map native engram weights and scales with bounded asynchronous prefetch, deduplication, staging, and request-safe cancellation.
- [ ] Wire early engram prefetch for decode, speculative verification, and batched prefill with rollback-safe history.
- [ ] Implement native FP4 backbone and dSpark expert tensor parallelism through the AFD transport with appropriate SparkInfer/b12x kernels.
- [ ] Implement native image processing, ViT, aligner, image-span embeddings, and vision routing through the serving API.
- [ ] Budget and qualify RTX-resident dSpark against remote-expert placement, including concurrency-16 workspace and coordinator contention.
- [ ] Implement all three dSpark stages, Markov and confidence heads, proposal verification, sampling, acceptance, and state rollback.
- [ ] Extend admission, scheduling, graph shapes, cache ownership, and cancellation to 16 concurrent requests with alternating waves.
- [ ] Qualify native numerical paths and CUDA graph replay on both local RTX GPUs and on ostrich and dodo without using emu or kiwi.
- [ ] Qualify the integrated modelless stack with deterministic nonzero fixtures, transport, serving regressions, and allocation/capacity checks.
- [ ] Record required temporary fixtures in TO_DELETE_SCAFFOLDING.md and preserve reusable qualification tooling.
- [ ] Commit and push engine changes and required SparkInfer/b12x changes to their fork main branches with matching dependency pins.
- [ ] Audit every pre-model requirement and document exact bring-up commands, remaining checkpoint-dependent gates, and hardware evidence for handoff.
