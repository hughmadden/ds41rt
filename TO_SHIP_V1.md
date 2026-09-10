# Ship DS41RT V1

This is the single implementation, full-checkpoint bring-up and release plan; it supersedes PRE_MODEL_PLAN.md and POST_MODEL_PLAN.md, whose history remains in Git.
The official checkpoint is available on the coordinator and all four Sparks, so component fixtures and checkpoint-backed work can proceed in whichever order removes dependencies fastest.
Completion means a working, qualified release with measured performance, not component-only qualification.

## Required release contract

- [ ] Serve only the official DeepSeek-V4.1-Flash checkpoint on one RTX coordinator and four Spark TP4 backbone-expert workers.
- [ ] Keep native vision, all three dSpark stages and their experts, shared experts, routing, attention, sampling and API execution on the RTX.
- [x] Remove balanced/long/accuracy serving profiles and require FP8-style serving KV.
- [ ] Preserve streaming, tools, structured constraints, cancellation, admission, error handling, prefix reuse and restart readiness.
- [ ] Support concurrency 1 through 16 with request-safe state and qualified alternating-wave scheduling.
- [ ] Measure at least 90 generated tokens/s for target-only decode with dSpark disabled under a documented workload.
- [ ] Measure approximately 270 generated tokens/s with optimized dSpark under documented sampling, acceptance and prompt conditions.
- [ ] Measure at least 8,000 prefill tokens/s for prompts of 8,192 tokens or longer under a documented batching policy.
- [ ] Publish latency, throughput, acceptance, memory, context length and concurrency together so performance results are reproducible.

## Checkpoint, loading and build

- [x] Pin the architecture/reference audit to official revision `df42c109f1defefcbfcedbe7d905718a12266e40`.
- [x] Validate the completed local `dba1be0a40aa45a94ad051997016db3960a90277` snapshot with the production 96,085-tensor catalog reader.
- [x] Confirm checkpoint headers and required configuration, tokenizer and vision assets agree across all five hosts and record any differences from the reference lock.
- [ ] Validate checkpoint payload representations through selected real-weight component execution before full-model loading.
- [ ] Read and stage only each node's required tensors or TP slices with explicit native packed-storage and workspace budgets.
- [ ] Measure useful weight bytes, physical read bytes and load time separately and optimize loading when it falls behind measured Spark NVMe throughput near 5 GB/s.
- [ ] Measure graph setup separately from weight loading and prune graph work if additional startup exceeds 75 seconds.
- [ ] Qualify build, WIP, run, restart, stop, source-manifest generation, artifact export and deployment across all five hosts.
- [ ] Replace inherited release defaults and deployment gates with the official V4.1 execution path and matching ds41 images.
- [x] Use the b12x fork's master branch and preserve matching root/submodule pins.

## Model execution and state

- [ ] Complete the architecture and tensor-layout audit with explicit reference-to-production mappings.
- [ ] Wire CED encoder/decoder execution, shared KV ownership, chunked prefill and exact replay with separately qualified bounded replay.
- [ ] Implement CSA2 ratio-2 and ratio-1 production and source/reindex/reuse modes under the FP8-only serving KV contract.
- [ ] Wire hierarchical candidate-block selection and causal sparse top-512 attention with large-pool addressing checks.
- [ ] Compose reference-qualified mHC, normalization, RoPE, projections and modality-aware routing into backbone execution.
- [ ] Connect native FP4 backbone TP4 experts, FP32 route transport and correctly ordered reduction to the production AFD scheduler.
- [x] Implement deterministic engram tokenizer compression, prime layouts, hashes, image barriers and request histories.
- [ ] Wire mapped engram tables/scales, bounded asynchronous prefetch, deduplicated gathers, staging and cancellation into execution.
- [ ] Trigger engram work early for decode, verification and batched prefill and commit accepted-prefix history without stale-wave reuse.
- [ ] Implement native image preprocessing, ViT, aligner, image-span embeddings and vision routing through the API.
- [x] Qualify owned dSpark main projection, independent committed KV producers and draft attention with native rotary frequencies on both RTX GPUs.
- [x] Implement and qualify allocation-free decoder stream-mean gathering directly into the owned dSpark main input on both RTX GPUs.
- [ ] Gather dSpark taps from target-layer attention-input stream means after engram updates in layer order 37/38/39.
- [ ] Compose the complete three-stage dSpark attention/mHC/FFN sequence with embedding, Markov, confidence and terminal sampling.
- [ ] Wire proposal verification, acceptance, RNG ownership, cancellation and rollback across all caches and engram histories.
- [ ] Complete concurrency-16 admission, mixed prefill/decode/verification scheduling, graph capacity management and alternating-wave overlap.

## Integrated qualification and release

- [ ] Validate real-checkpoint text logits and generation against the official reference with dSpark disabled first.
- [ ] Validate dSpark proposal distributions, confidence policy, acceptance, rollback and generated-output correctness.
- [ ] Validate single-image, multiple-image, interleaved text and multi-turn vision requests.
- [ ] Validate long contexts, prefix reuse and exact/bounded CED replay across multi-turn conversations.
- [ ] Exercise serving regressions under mixed 16-request text, vision, prefill, decode and verification traffic.
- [ ] Compare alternating waves with alternatives on the complete four-Spark AFD topology.
- [ ] Tune fusions, shapes, expert reduction, engram prefetch and memory budgets using measured full-stack bottlenecks.
- [ ] Run sustained load, cancellation, recovery and restart tests before publishing release claims.
- [ ] Remove obsolete V4 model execution, alternate formats, quantizers, dependencies, optimization notes and inapplicable benchmarks while preserving serving features.
- [ ] Remove tracked temporary scaffolding once replacement coverage exists and retain useful small component fixtures during development.
- [ ] Finalize release commands, images, dependency pins, correctness/performance evidence and pushed commits after every required gate passes.

## Current evidence and next dependency

The native component records in `docs/ds41-*-qualification.*` establish their stated local scopes; they do not establish full-model readiness or the throughput targets above.
The latest dSpark work is recorded in `docs/ds41-dspark-frequencies-qualification.md`, and full-backbone attention, vision and serving integration remain incomplete.
The next dSpark dependency is target-layer tap gathering followed by complete stage and verification wiring, while the now-complete checkpoint allows selected real-weight validation and loader measurements in parallel with that work.
`TO_DELETE_SCAFFOLDING.md` tracks temporary files committed to the repository, not a requirement to delete useful external test data now.

Fleet checkpoint evidence is recorded in `docs/ds41-checkpoint-inventory.json` and `docs/ds41-checkpoint-inventory.md`; all weights still require execution qualification, and payload hashes were not scanned.

The tap gather and composed main-context evidence is in `docs/ds41-dspark-taps-qualification.md`; production target-layer call sites remain unwired.
