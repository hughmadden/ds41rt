# Ship DS41RT V1

This is the single implementation, full-checkpoint bring-up and release plan; it supersedes PRE_MODEL_PLAN.md and POST_MODEL_PLAN.md, whose history remains in Git.
The official checkpoint is available on the coordinator and all four Sparks, so component fixtures and checkpoint-backed work can proceed in whichever order removes dependencies fastest.
Completion means a working, qualified release with measured performance, not component-only qualification.

## Required release contract

- [ ] Serve only the official DeepSeek-V4.1-Flash checkpoint on one RTX coordinator and four Spark TP4 backbone-expert workers.
- [ ] Keep native vision, all three dSpark stages and their experts, shared experts, routing, attention, sampling and API execution on the RTX.
- [x] Remove balanced/long/accuracy serving profiles and enforce FP8 target KV in launch settings.
- [x] Store dSpark committed windows as packed E4M3/E8M0 K32, completing the fixed FP8 persistent-cache representation in the V4.1 components.
- [ ] Requalify the packed dSpark windows and complete draft graphs on RTX. Spark cache-byte/ownership and attention checks pass; RTX driver repair is still required.
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
- [x] Validate checkpoint payload representations through selected real-weight component execution before full-model loading.
- [ ] Read and stage only each node's required tensors or TP slices with explicit native packed-storage and workspace budgets.
- [x] Add bounded expert read-ahead and verify real TP staging bytes with a reusable cold/warm I/O probe on ostrich and the coordinator.
- [ ] Measure useful weight bytes, physical read bytes and load time separately and optimize loading when it falls behind measured Spark NVMe throughput near 5 GB/s.
- [ ] Measure graph setup separately from weight loading and prune graph work if additional startup exceeds 75 seconds.
- [ ] Qualify build, WIP, run, restart, stop, source-manifest generation, artifact export and deployment across all five hosts.
- [ ] Replace inherited release defaults and deployment gates with the official V4.1 execution path and matching ds41 images.
- [x] Use the b12x fork's master branch and preserve matching root/submodule pins.

## Model execution and state

- [ ] Complete the architecture and tensor-layout audit with explicit reference-to-production mappings.
- [ ] Wire CED encoder/decoder execution, shared KV ownership, chunked prefill and exact replay with separately qualified bounded replay.
- [x] Implement native ratio-two compressor projection/pooling and ratio-one projection/norm; qualify all four real-weight sources on both RTX GPUs.
- [x] Add and qualify compressor request leases, partial-group ownership, captured proposal execution and accepted-prefix commit across two competing waves.
- [ ] Implement CSA2 ratio-2 and ratio-1 production and source/reindex/reuse modes under the FP8-only serving KV contract.
- [x] Qualify official backbone RoPE/YaRN through 1,048,576 positions and generate completed-latent frequencies inside compressor graphs.
- [x] Project and normalize/rotate real-weight index keys inside the source graph; qualify all four sources and 16-request prefill/acceptance replay.
- [x] Pack FP4/E8M0 index proposals inside source graphs and match official quantizer bytes on both RTX GPUs.
- [x] Own paged persistent index storage and commit only accepted complete rows, with pool exhaustion/reuse qualification.
- [x] Publish device index page tables and committed row counts after accepted writes, with release/reuse qualification.
- [x] Score explicit index candidates from paged committed history with reference BF16 rounding and causal bounds.
- [x] Select top-512 over bounded score tiles with deterministic ties and full-history partition/replay qualification.
- [x] Implement tiled eight-row candidate maxima, newest-block pinning, top-2048 blocks and causal expansion.
- [x] Produce real-weight learned index queries and scaled head weights in captured RTX waves; qualify all eight producer layers and shared-weight ownership.
- [x] Score causal, strided index proposals without cache mutation and expose lease/version-checked compressor views.
- [x] Compose learned queries, causal views and captured selection with exact proposal-snapshot candidate sharing across index layers.
- [x] Persist fixed K32 FP8 compressed-source KV with paired index/KV page ownership and accepted-prefix publication.
- [x] Own real-weight FP8 backbone window production, private proposals and accepted ring writes across all 40 layers and 16 requests.
- [x] Implement direct FP8 window/paged-source sparse attention with private overlays, reference arithmetic and maximum-pool addressing qualification.
- [x] Compose owned sparse attention with request/window/source views and exact selection snapshots across source, reuse and later index layers.
- [x] Produce real backbone low-rank attention queries in shared-weight captured waves and qualify all 40 layers.
- [x] Produce real backbone inverse-rotary/grouped/FP8 attention output projections with shared weights across all 40 layers.
- [x] Connect real query, index selection, sparse attention and output stages with checked execution identities and token handoffs.
- [x] Share shifted mHC ownership between backbone and dSpark, qualify real attention/FFN parameters across all 40 layers, and retain exact dSpark stage regression.
- [x] Sequence backbone attention and FFN mHC phases with exact query-result binding; qualify real attention composition and explicitly bounded identity-FFN fixtures.
- [x] Share native FP8 shared-expert execution with dSpark, load all 40 backbone shared experts and qualify captured waves plus block-bound normalized-input handoffs.
- [x] Own real backbone text/vision routing, bind canonical TP4 requests and shared results to block execution, and qualify analytical transport/reduction handoffs.
- [x] Execute generated routes against real checkpoint layers 0/20/39 on all four Sparks, qualify every selected expert and RTX shared reduction, and finish the block mHC boundary.
- [ ] Wire hierarchical candidate-block selection and causal sparse top-512 attention with large-pool addressing checks.
- [x] Correct RMS epsilon to the official 1e-20 separately from mHC mixing epsilon 1e-6, and requalify primitive/stage/draft paths.
- [ ] Compose reference-qualified mHC, normalization, RoPE, projections and modality-aware routing into backbone execution.
- [x] Add adjacent-block preparation with predecessor identity, engram ordering guards and prepared dSpark tap inputs; all 39 transition guards/copies and distinct decoder tap placement pass on Spark with controlled producer values.
- [x] Reuse each lane's two mHC workspaces across all 40 layers; two real-weight passes at 1/80/4096 rows match fresh owners on Spark. Fix and regression-check synchronous D2D completion before nonblocking consumers. See `docs/ds41-reusable-block-qualification.md`.
- [x] Add query/output/shared-FFN buffer rebinding with a bounded, exact-weight-owner graph cache. The daemon builds and the retained CUDA cache test passes 240 captures/480 replays across two lanes and all 40 layers on Spark; see `docs/ds41-layer-workspace-reuse.md`.
- [ ] Qualify rebound query/output/shared-FFN arithmetic and identities against fresh real-weight owners on RTX, including layer revisits, changed shapes and two lanes.
- [ ] Assemble reusable mHC/query/output/shared-FFN and layer-independent sparse/index workspaces into actual alternating execution lanes, and account for all weights, caches and other owners before full-model allocation. The five audited workspace groups once per 4096-row lane total 2.75 GiB; this excludes other storage and does not establish full-model fit.
- [x] Add the owned backbone lane for mHC/query/sparse/output/FFN sequencing, a shared 40-layer weight owner and aggregate allocation gates. Its production build and CPU-only official-checkpoint budget checks pass; see `docs/ds41-backbone-lane.md`.
- [ ] Execute the assembled lane on RTX, qualify numerical/state recovery and two-lane independence, and compose the remaining request-cache/index/router owners. Separate TP4 dispatch/gather so routed execution can overlap shared FFN work.
- [ ] Execute the chained block/engram/prepared-query and owned dSpark tap paths on RTX, then connect them to scheduler-owned gather histories.
- [x] Initialize target text residuals and identity pre-mix from the embedding table shared with dSpark, retaining token/position metadata and copying into layer-0 inputs; exact checkpoint comparisons and owner guards pass on Spark.
- [ ] Qualify embedding-to-attention execution on RTX and integrate image embedding replacement before residual expansion.
- [ ] Qualify the bound final target head on RTX and wire selected prefill/decode/verification rows into sampling. The owner builds and its real-weight numerical checks pass on Spark; final RTX and dSpark projection requalification remain open after the cuBLAS accumulation correction.
- [ ] Connect native FP4 backbone TP4 experts, FP32 route transport and correctly ordered reduction to the production AFD scheduler.
- [x] Implement deterministic engram tokenizer compression, prime layouts, hashes, image barriers and request histories.
- [ ] Wire mapped engram tables/scales, bounded asynchronous prefetch, deduplicated gathers, staging and cancellation into execution.
- [ ] Trigger engram work early for decode, verification and batched prefill and commit accepted-prefix history without stale-wave reuse.
- [ ] Implement native image preprocessing, ViT, aligner, image-span embeddings and vision routing through the API.
- [x] Qualify owned dSpark main projection, independent committed KV producers and draft attention with native rotary frequencies on both RTX GPUs.
- [x] Implement and qualify allocation-free decoder stream-mean gathering directly into the owned dSpark main input on both RTX GPUs.
- [ ] Gather dSpark taps from target-layer attention-input stream means after engram updates in layer order 37/38/39.
- [x] Compose and qualify each complete dSpark attention/mHC/shared-and-routed-FFN stage in one captured graph on both RTX GPUs.
- [x] Chain all three dSpark transformer stages in one graph with validated cross-stage cache bindings and exact residual/pre-mix handoff.
- [x] Initialize dSpark seed/noise embeddings from the shared coordinator table inside the three-stage graph; qualify changed-token replay on both RTX GPUs.
- [x] Compose and qualify the complete three-stage dSpark attention/mHC/FFN proposal graph with embedding, Markov, confidence and terminal sampling on both RTX GPUs.
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
- [x] Remove the offline V4 quantization pipeline, converter images/operations, GPTQModel and exllamav3 source dependencies, and old Docker comparison benchmarks.
- [ ] Remove the remaining inherited V4 execution and alternate-format runtime paths, model-specific tools and tests while preserving serving features.
- [ ] Remove tracked temporary scaffolding once replacement coverage exists and retain useful small component fixtures during development.
- [ ] Finalize release commands, images, dependency pins, correctness/performance evidence and pushed commits after every required gate passes.

## Current evidence and integration gaps

Component records establish their stated scopes, not full-model readiness or the performance targets above. Historical records remain under `docs/ds41-*`; use the checklist for current completion status.

| Area | Evidence | Remaining integration or qualification |
| --- | --- | --- |
| Official checkpoint on all hosts | [Inventory](docs/ds41-checkpoint-inventory.md) | Complete runtime load and execution; inventory is not an all-payload hash audit |
| b12x master and native TP4 | [Master review](docs/ds41-b12x-master-review.md), [real four-Spark FFNs](docs/ds41-real-tp4-qualification.md) | All-layer startup, cold I/O and full-model scheduling |
| Engram | [Pipeline](docs/ds41-engram-pipeline.md), [CUDA record](docs/ds41-engram-cuda-qualification.json) | Early prefetch, request histories and residual updates in the model driver |
| FP8 backbone cache and attention | [KV ownership](docs/ds41-kv-qualification.md), [window](docs/ds41-window-qualification.md), [attention handoffs](docs/ds41-attention-handoff-qualification.md) | CED execution/replay and accepted-prefix scheduling across layers |
| Backbone mHC and FFNs | [Block sequencing](docs/ds41-block-qualification.md), [shared experts](docs/ds41-backbone-shared-qualification.md), [routing](docs/ds41-backbone-router-qualification.md), [real TP4](docs/ds41-real-tp4-qualification.md) | Full 40-layer driver and numerical qualification |
| Between-block preparation | [Transition checks](docs/ds41-block-transition-qualification.md) | Chained engram/query execution and owned dSpark tap validation on RTX; scheduler gather association |
| Target initialization | [Embedding and input handoff](docs/ds41-target-embedding-qualification.md) | RTX embedding-to-attention execution and vision replacement |
| Final target head | [Partial qualification](docs/ds41-target-head-qualification.md) | Final RTX checks after accumulation correction; sampling integration |
| dSpark | [Draft graph](docs/ds41-dspark-draft-qualification.md), [packed cache](docs/ds41-dspark-fp8-cache-qualification.md) | RTX requalification after cache/head changes, target taps, verification, acceptance and rollback |
| Vision, serving and performance | Checklist above | Full native vision path, concurrency 16, API/restart gates and measured throughput |

RTX validation currently requires resolving the host NVIDIA kernel/library version mismatch; Spark work and CPU/build checks can continue. The official checkpoint is available, so use the smallest relevant real-weight fixture or integration run for each change. `TO_DELETE_SCAFFOLDING.md` tracks temporary committed code, not a requirement to delete useful external test data.
