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
- [ ] Wire hierarchical candidate-block selection and causal sparse top-512 attention with large-pool addressing checks.
- [x] Correct RMS epsilon to the official 1e-20 separately from mHC mixing epsilon 1e-6, and requalify primitive/stage/draft paths.
- [ ] Compose reference-qualified mHC, normalization, RoPE, projections and modality-aware routing into backbone execution.
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
- [ ] Remove obsolete V4 model execution, alternate formats, quantizers, dependencies, optimization notes and inapplicable benchmarks while preserving serving features.
- [ ] Remove tracked temporary scaffolding once replacement coverage exists and retain useful small component fixtures during development.
- [ ] Finalize release commands, images, dependency pins, correctness/performance evidence and pushed commits after every required gate passes.

## Current evidence and next dependency

The native component records in `docs/ds41-*-qualification.*` establish their stated local scopes; they do not establish full-model readiness or the throughput targets above.
The latest dSpark work is recorded in `docs/ds41-dspark-draft-qualification.md`, and full-backbone attention, vision and serving integration remain incomplete.
The next dSpark dependency is target-layer/scheduler wiring and target verification, while the now-complete checkpoint allows selected real-weight validation and loader measurements in parallel with that work.
`TO_DELETE_SCAFFOLDING.md` tracks temporary files committed to the repository, not a requirement to delete useful external test data now.

Fleet checkpoint evidence is recorded in `docs/ds41-checkpoint-inventory.json` and `docs/ds41-checkpoint-inventory.md`; all weights still require execution qualification, and payload hashes were not scanned.

The tap gather and composed main-context evidence is in `docs/ds41-dspark-taps-qualification.md`; production target-layer call sites remain unwired.

Checkpoint-backed staging and the remaining read-amplification bottleneck are recorded in `docs/ds41-expert-load-probe.md`; these host reads do not establish real-weight GPU numerical or startup qualification.

Complete-stage composition and its remaining integration limits are recorded in `docs/ds41-dspark-stage-qualification.md`.

Three-stage composition is qualified in `docs/ds41-dspark-chain-qualification.md`; shared embedding integration is qualified in `docs/ds41-dspark-embedding-qualification.md`, and the complete proposal graph is qualified in `docs/ds41-dspark-draft-qualification.md`.

The official normalization contract and corrected regression evidence are in `docs/ds41-normalization-correction.md`; earlier default-epsilon normalization evidence is superseded.

Native backbone compressor evidence is in `docs/ds41-compressor-qualification.md`; request-owned partial-group state and accepted-prefix commit are qualified in `docs/ds41-compressor-owner-qualification.md`, with index/cache producers next.

Backbone rotary frequencies and their compressor handoff are qualified in `docs/ds41-backbone-frequencies-qualification.md`; index-key projection is qualified in `docs/ds41-index-key-qualification.md`, with packed cache producers next.

Index proposal encoding is qualified in `docs/ds41-index-pack-qualification.md`; persistent index/KV cache writes and candidate selection remain next.

Persistent index ownership and accepted writes are qualified in `docs/ds41-index-cache-qualification.md`; candidate selection and persistent FP8 KV remain open.

Device index page-table publication is qualified in `docs/ds41-index-table-qualification.md`; sparse scoring/selection consumers remain unwired.

Native paged index scoring is qualified in `docs/ds41-index-scores-qualification.md`; query production, proposal overlays and hierarchical/top-512 selection remain open.

Bounded tiled top-512 selection is qualified in `docs/ds41-index-topk-qualification.md`; hierarchical candidate blocks and production selection ownership remain open.

Hierarchical candidate selection is qualified in `docs/ds41-candidate-blocks-qualification.md`; learned query production, candidate sharing and proposal overlays remain open.

Learned index queries are qualified in `docs/ds41-index-query-qualification.md`; request-owned selection composition, candidate sharing and proposal overlays remain open.

Causal index proposals and borrowed compressor views are qualified in `docs/ds41-index-overlay-qualification.md`; complete query/selection ownership and shared candidate scheduling remain open.

Owned query/selection composition and candidate sharing are qualified in `docs/ds41-index-selection-qualification.md`; persistent FP8 KV, sparse attention and backbone scheduling remain open.

Fixed FP8 source KV and paired index/KV commit are qualified in `docs/ds41-kv-qualification.md`; window ownership and sparse attention consumption remain open.

Fixed FP8 backbone windows are qualified in `docs/ds41-window-qualification.md`; sparse attention consumption, dSpark storage conversion and backbone/scheduler integration remain open.

Native direct FP8 sparse attention is qualified in `docs/ds41-sparse-attention-qualification.md`; owned composition with selection/window/source views and backbone query/output production remain open.

Owned cache/selection attention composition is qualified in `docs/ds41-attention-owner-qualification.md`; real backbone attention query and output production remain open.

Real backbone query production is qualified in `docs/ds41-attention-query-qualification.md`; query/index/attention handoff and inverse rotary/grouped output production remain open.

Real backbone output production is qualified in `docs/ds41-backbone-attention-output-qualification.md`; complete query/index/attention/output composition remains open.

Real query/index/attention/output handoffs are qualified in `docs/ds41-attention-handoff-qualification.md`; backbone mHC/CED/engram/routing composition and scheduler integration remain open.

Shared mHC ownership and real backbone boundary arithmetic are qualified in `docs/ds41-backbone-hc-qualification.md`; attention/FFN orchestration and scheduler integration remain open.

Backbone phase sequencing is recorded in `docs/ds41-block-qualification.md`, including observed chained BF16 sensitivity; actual shared/routed FFN integration and full-model numerical qualification remain open.
