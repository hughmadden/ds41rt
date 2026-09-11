# Optimization parity audit: native V4.1 versus GLMRT and DS4RT

The native V4.1 path retained the TP4 model split, packed resident weights and several component graphs, but bypassed substantial serving optimizations already present in the older engines. The largest concrete regression is returning six FP32 route vectors from every Spark instead of reducing to hidden width before transport. Changing TCP to verbs alone cannot fix that payload problem.

This is a source audit of the serving-critical paths, plus a bounded arithmetic experiment. It is not a claim that every old kernel has been benchmarked or that the missing features below have been restored. Inherited implementations elsewhere in this repository do not count as integrated unless `serve-native` actually reaches them.

## Source baseline

Reviewed on 2026-09-11:

| Checkout | Revision |
| --- | --- |
| `../glmrt-release` | `dc6d9b8e1600e001cb1d4228bd911f4df8091f99` |
| `../ds4rt-pro-rtx-4spark` | `e95211997eda79523864a66467a4b0160c7feeda` |
| Native V4.1 implementation | `f9fdeb4` |
| Current b12x `master` | `aeb1d8c18e99809797c76af6fbfaca1273c36ff6` |

The sibling links below require these adjacent checkouts. DS4's [architecture](../../ds4rt-pro-rtx-4spark/architecture.md) establishes the TP4, reduction, graph and readiness contract. GLM's [developer contract](../../glmrt-release/DEVELOPER.md) explicitly records verbs-host, four qualified execution/QP lanes, resident weights, direct packed attention and no timed graph capture.

## Wire comparison

All figures below use **V4.1's** hidden width 5120, six routes and forty backbone layers, so model-size differences do not distort the comparison. Payload only: framing, row maps, routing metadata, inter-Spark traffic and request traffic are excluded.

| Return design | Bytes/token/layer arriving at RTX, all ranks | Bytes/token across 40 layers | Reduction versus current |
| --- | ---: | ---: | ---: |
| Current: four ranks × six FP32 route vectors | 491,520 | 19,660,800 | 1× |
| Compact FP32: four hidden-width partials | 81,920 | 3,276,800 | 6× |
| Old small-batch design: four BF16 hidden-width partials | 40,960 | 1,638,400 | 12× |
| Combined BF16 result, potentially returned as disjoint row shards | 10,240 | 409,600 | 48× |

The earlier 48× comparison was against a single combined BF16 result. Against the old **small-batch** four-partial return, the current regression is **12×**. Wider old batches used Spark-side reduction; their actual output dtype depended on configuration, so the final row is a BF16 design comparison, not a claim that all old runs returned BF16.

The activation input already is BF16 hidden width: 10,240 bytes/token per rank, or 40,960 bytes/token across four ranks, plus routes and metadata. The output route axis is the avoidable expansion.

At 8k input tokens/s, current return traffic alone needs 157.29 GB/s. Four compact BF16 partials need 13.11 GB/s; combined BF16 needs 3.28 GB/s. Replicated activation requests additionally need 13.11 GB/s in the outbound direction. These are directional payload budgets, not throughput predictions. Spark reduction must also fit the Spark-to-Spark links and cannot be treated as free. See the measured [return bottleneck](ds41-prefill-return-bottleneck.md).

## Differential inventory

Status describes the active native V4.1 path, not whether similarly named code exists in the tree.

| Optimization and old evidence | Native V4.1 finding | Required disposition |
| --- | --- | --- |
| Strict TP4, one global route list, every rank owns a quarter of every expert: DS4 architecture, routed expert tensor parallelism | **Retained.** [Request validation](../rust/crates/ds41rt-transport/src/v41_expert.rs) requires six canonical routes and replicated BF16 inputs. | Keep TP4; do not replace it with whole-expert placement. |
| Hidden-width small-batch BF16 partials: DS4 [scheduler protocol](../../ds4rt-pro-rtx-4spark/rust/crates/ds4rt-daemon/src/commands/real_full/scheduler/protocol_v2.rs), `real_full_scheduler_ds4_tp4_try_start_verbs_raw` | **Missing.** Native request/response geometry hardcodes `5120 * 6` FP32 output. | P0: Spark-local route accumulation and compact BF16 return, qualified for changed arithmetic. FP32-only compaction is an intermediate comparison, not parity. |
| Wide row-sharded reduction, qualified crossover 16 rows: DS4 architecture and [config](../../ds4rt-pro-rtx-4spark/ds4rt.config) | **Missing.** Native parser rejects legacy reduction/compression/stream flags. | P1: connect a native reduction plan and return completed row shards. Measure a new crossover; do not assume 16 is optimal for V4.1. |
| Fused reduction/packing and cached accumulators: DS4 [route execution](../../ds4rt-pro-rtx-4spark/rust/crates/ds4rt-daemon/src/commands/real_full/sparse_mlp/route.rs), `fused_fp8_reduction_eligible`, `pack_streamed_completion_rows` | **Missing from native distributed returns.** Existing expert scratch is reused, but route planes are copied out in full. | Fuse route accumulation/output packing where profitable. Qualify BF16 first; FP8 wire is a separate numerical/performance decision from FP8 KV. Do not transplant W4/EXL3 kernel predicates. |
| Persistent verbs-host transport and mapped rings: GLM [executor](../../glmrt-release/rust/crates/glmrt-daemon/src/commands/real_full/expert_probe/protocol_v2_executor.rs), `VerbsHostMappedRdmaRing`; DS4 scheduler protocol | **Missing.** Native uses [TCP](../rust/crates/ds41rt-transport/src/v41_expert/tcp.rs), preserving sockets within a job and resetting them at each admission. | P1: reuse transport mechanisms with native identities, geometry and startup negotiation. Keep reconnect/failure behavior explicit. |
| Borrowed recyclable pinned responses and asynchronous staging: GLM executor `emit_borrowed_output`; DS4 scheduler protocol `reduce_verbs_chunks_async_owned` and chunk-segment reduction | **Missing end to end.** [Spark execution](../rust/crates/ds41rt-daemon/src/v41_experts/execution.rs) synchronizes and copies full results to `Vec`; [RTX collection](../rust/crates/ds41rt-daemon/src/v41_experts/coordinator.rs) uses blocking H2D. | Own registered/pinned payloads through GPU completion and reuse them. Avoid an extra pageable-to-pinned copy. The previous extra-staging experiment regressed end-to-end time and was reverted. |
| Streamed completion/reduction: DS4 scheduler `dispatch_reduced_identity_payload_streaming`; GLM executor emitted borrowed output | **Partial.** Native TCP chunks transport a result only after full GPU execution and D2H; chunk framing is not GPU/communication pipelining. | P1: emit completed row ranges, preserve disjoint ownership, and reduce ready ranges without requiring unrelated rows to finish. Qualify reorder, duplication, cancellation and buffer reuse. |
| Four execution/QP lanes: GLM developer contract; transport/CPU-affinity constructors in both schedulers' `execution/progression.rs` | **Missing from active service.** [Service](../rust/crates/ds41rt-daemon/src/v41_experts/service.rs) has one owning CUDA worker and a bounded queue. | Integrate independently owned wave buffers and transport lanes; measure CPU affinity/polling policy. Queue depth 16 is not sixteen concurrently executing requests. |
| Optional second Spark rail: DS4 architecture and release configuration | **Not active in native transport.** Only the four primary addresses are used by the current API. | P1: map actual links and NIC locality, configure all four secondary addresses consistently, measure each rail and shared bottlenecks. |
| Nonblocking expert dispatch before coordinator shared work: DS4 scheduler `try_start_verbs_raw` | **Retained structurally.** [Backbone lane](../rust/crates/ds41rt-daemon/src/v41_backbone_lane.rs) dispatches before shared execution. | Preserve this ordering; measure actual overlap. Sequential API admission still prevents overlap between requests. |
| Two coordinator graph segments surrounding the expert barrier, startup binding/prewarm: DS4 architecture | **Partial.** Native component graphs and bounded reuse exist; the entire block is not integrated into the old two-segment execution contract, and new shapes can capture on request execution. | P2: compose barrier-bounded graphs with stable owners; prewarm admitted shapes. Count captures during measured requests and require zero for qualified shapes. |
| Packed resident GPU weights, cached workspaces, direct packed attention: both architecture contracts | **Retained in native components.** FP4 backbone experts, FP8 persistent KV, native attention and reusable workspaces are implemented. Expert read-ahead reduced measured load time to 35.7–37.8 s/rank. | Preserve [loading improvements](ds41-parallel-expert-loading.md) and [workspace reuse](ds41-layer-workspace-reuse.md). Benchmark new shapes; do not blindly enable expert graphs whose prior probe showed no benefit. |
| Rolling bounded admissions and shape-aware chunking: GLM [execution](../../glmrt-release/rust/crates/glmrt-daemon/src/commands/real_full/scheduler/execution.rs), `MAX_ACTIVE_BOUNDED_LONG_PREFILL_CHUNKS`; both engines' `entry.rs` default to 2048 with smaller policies | **Missing in live API.** [Native serving](../rust/crates/ds41rt-daemon/src/v41_native_serve.rs) processes one job at a time and chunks prompts at 80 rows. | P2: capacity-budgeted larger prefill plus mixed decode/verify admission and alternating waves. Six verification rows × sixteen requests needs at least 96 rows. GLM's sixteen active prefill chunks are not evidence of sixteen concurrent user requests. |
| Shared cache pool reservations and prefix reuse: DS4 architecture and [prefix cache](../../ds4rt-pro-rtx-4spark/rust/crates/ds4rt-daemon/src/commands/real_full/prefix_cache.rs) | **Missing in API.** Current admission uses slot zero, reports zero prefix hits, releases after each job and limits prompt plus output to 4096. | P2: attach pool admission and prefix ownership to V4.1 cache/engram/dSpark state. Qualify hits, eviction, cancellation and long-context capacity together. |
| Confidence-driven adaptive dSpark width 1–5: DS4 architecture | **Missing policy.** Native [speculative serving](../rust/crates/ds41rt-daemon/src/v41_native_serve/speculative.rs) proposes five when possible, truncated only by remaining output budget. | P2: calibrate confidence and verification cost on this model. Retain all dSpark computation on RTX as requested. |
| Fingerprint-matched resident expert reuse, graph readiness and deployment identity: DS4 [WIP launcher](../../ds4rt-pro-rtx-4spark/scripts/run-wip.sh) and architecture readiness gates | **Not integrated into native release launch.** Current live service uses manually frozen binaries; inherited [release defaults](../scripts/release-common.sh) still include old Pro/EXL3 assumptions. | P0/P3: native release build/run, model and artifact identity, graph warmup, readiness, resident reuse, graceful shutdown and restart. Dirty-checkout manifest generation already exists in [build.sh](../build.sh), but does not establish a working native release. |
| Reproducible warmed API measurement, capture counters and separated decode/speculative/prefill metrics: GLM developer contract and both benchmark tools | **Partial.** Native has paired API quality and short counting measurements, with explicit scope. | P0 onward: preserve those gates and add bytes/link, GPU/transport overlap, C1–C16, fresh/cached long prefill, acceptance and sustained-load distributions. Old headline TPS numbers are not this model's baseline. |

## Expert packing, fusion and large-prefill reuse

The user's load-time GPU packing and streamed GEMM work is a required part of this audit, not an optional follow-up to transport.

**GPU packing is retained.** DS4's `sparse_mlp/route.rs` uploads bounded staging and calls `cuda_ds4_flash_w4a16_pack_weight_async` / `cuda_ds4_flash_w4a16_pack_e8m0_scale_async`; GLM calls `cuda_b12x_w4a16_pack_weight_strided_async` / `cuda_b12x_w4a16_pack_scale_strided_async`. These are GPU layout transformations, not CPU repacking during inference. Their W4A16/trellis representations are not automatically the correct representation for V4.1's FP4/K32 checkpoint.

Current [native loading](../rust/crates/ds41rt-daemon/src/v41_experts.rs) also uploads pinned staging and calls the GPU packer once per expert. The [packing kernel](../native/cuda/kernels/v41_expert_pack.cu) creates N256/K128 lane-major weight and E8M0 scale arrays consumed directly by b12x W4A8. W13 stores up/W3 first and gate/W1 second, padded independently. Spark intermediate width 576 pads to 640; W2 and scales use matching padded storage. Four packed buffers remain resident; the logical device staging is discarded after loading. Current loading synchronizes after each expert, so further upload/packing overlap is a measurable startup opportunity, not an inference hot-path fix.

**The FFN intermediate is fused in the active V4.1 kernel.** The [AOT exporter](../python/tools/export_b12x_v41_experts_aot.py) selects `w4a8_mx`, `fp4_e8m0_k32`, `silu_v41`, `w13_layout="w13"` and `w4a8_repacked=True`. In b12x's [dynamic kernel](../third_party/sparkinfer/b12x/moe/_shared/kernels/dynamic.py), the V4.1 constructor explicitly rejects `materialize_intermediate`. The task runs FC1 → activation/quantization → FC2, with producer warps streaming weights. A scratch field named `materialized_intermediate` exists and is allocated by the general planner, but that name is **not evidence of active intermediate writes** in this specialization. Remove unused allocation only with ABI/workspace qualification.

**There still is global traffic beyond weight reads.** The dynamic kernel routes and packs token activations into expert-major global scratch, maintains row/task metadata, and stores FP32 per-route output. The latter is the large transport regression discussed above. Do not describe the entire path as having no materialization at any stage merely because FC1-to-FC2 stays fused. Likewise the older W4A16 family contains `intermediate_cache13` / `intermediate_cache2` paths: whether an old specialization avoids those writes must be established from its selected launch, not inferred from “fused” in its name.

**Large-prefill reuse is not yet qualified and has a concrete restriction.** b12x's [tile planner](../third_party/sparkinfer/b12x/moe/fused_moe/_impl.py), `_select_dynamic_tile_mn`, normally considers routed rows per expert. The V4.1 branch selects M16 up to `16 * num_experts` routed rows and M32 above that, specifically to retain its fused BF16 boundary. It does not select the generic M64/M128 materialized regimes. With 384 experts/top-6 this means M16 through planned capacity 1024 tokens and M32 above it. Current live 80-token chunks average only 1.25 routed rows/expert across all 384 experts; an 8192-token batch averages 128. These are arithmetic averages, not assumptions of uniform routing. Expert-major grouping enables reuse within tiles, but larger batches can still reread weights across multiple tiles. The AOT integration also calls the private dynamic exporter directly; policy-selected alternative backends and shared-input tactics are not established as integrated.

Add the following kernel gates before claiming parity or peak bandwidth:

- Record the exact packed layout, dtype, physical padding, selected kernel/tile/producer plan and resident allocation for each planned capacity, with no hot-path reformatting or compilation.
- Capture real routing histograms and compare rows per active expert, tile occupancy, skew and repeated expert-weight reads at decode, verification and large prefill sizes. Preserve b12x ownership of plan-time policy rather than adding daemon-side tile heuristics.
- Profile actual DRAM reads/writes, cache hits, shared-memory traffic, register spills, MMA activity and stalls for the selected fused plan. Separate useful checkpoint bytes from padded or repeatedly fetched bytes. A claimed 95% bandwidth figure needs measured traffic and a stated device-bandwidth denominator; it is not demonstrated by source inspection or API throughput.
- Evaluate compact output/fused route accumulation and wider **fused** prefill work without dropping V4.1 activation, routing-weight and BF16-boundary semantics. Do not enable the generic materialized path merely to obtain wider tiles.
- Measure isolated expert execution and complete four-Spark/API execution on the same workload. More token sharing changes arithmetic intensity; high DRAM utilization alone is not the acceptance criterion. Require lower elapsed time with qualified outputs.

## Arithmetic decision and bounded evidence

The current [native reducer](../native/cuda/kernels/v41_route_reduce.cu) sums TP ranks **per route**, rounds that route to BF16, sums routes in FP32, then adds the shared expert and rounds to BF16. Summing routes locally before TP reduction changes this rounding order. Compact FP32 return also changes it; retaining FP32 on the wire does not preserve the current algorithm.

Two saved real layer-0 batches, each 80 rows across four Sparks, were evaluated with explicit ordered CPU FP32 sums and BF16 conversions. The comparison excludes the shared expert and later layers:

| Candidate versus current routed result | Relative L2, two batches | Maximum absolute difference | Differing final BF16 elements |
| --- | --- | --- | --- |
| Local FP32 route sum, FP32 rank return | 0.002530 / 0.002528 | 0.0078125 | 36.54% / 37.06% |
| Local FP32 route sum, BF16 rank return | 0.003007 / 0.003006 | 0.0078125 | 45.43% / 45.78% |

Both candidates remained finite. These results quantify the arithmetic change; they do **not** qualify either candidate for model quality. [Machine-readable observations](ds41-compact-return-audit.json) record the input artifacts and exact metrics. Require selected real layers, final-logit comparisons and paired live quality/acceptance results before replacing the serving format. Keep route-plane output in a diagnostic/reference path if needed, rather than requiring it for normal serving.

If preserving per-route rounding becomes necessary, reduction can instead occur among Sparks before route accumulation. That alternative moves the large intermediate traffic onto the Spark fabric; account for that traffic explicitly rather than claiming a free 48× system-wide saving.

## Implementation order and acceptance gates

1. **P0: compact BF16 return and a usable native launch path.** Add explicit response geometry/version agreement on both ends, local accumulation/packing, rank-ordered coordinator reduction, bounded buffers and numerical gates above. Verify stale/mixed-worker rejection and actual response byte counts. Finish the source-manifest/build/run integration against the official checkpoint. Do not silently reinterpret the current FP32 route-plane frames.
2. **P1: communication pipeline.** Reuse persistent verbs, registered payload ownership, row-sharded reduction, streaming and available rails. Compare small/wide crossover on the real fabric. Measure Spark GEMM, local reduction, D2H, wire, H2D and coordinator finish separately and end to end. Fusing compact output into the expert epilogue is an improvement candidate after correctness, not an assumed win.
3. **P2: scheduling and launch overhead.** Compose and prewarm graph segments; integrate real C1–C16 alternating execution; increase prefill capacity with explicit memory budgets; restore cache-aware admission/prefix reuse; tune adaptive dSpark. The current full-vocabulary FP32 D2H and CPU greedy scan is an additional native improvement candidate: move selection/verification to RTX with deterministic ties and failure checks. It is not marked as an established old-runtime parity item here.
4. **P3: release and sustained qualification.** Identity-checked resident worker reuse, numerical transport startup probes, zero timed captures, shutdown/restart recovery, long-context/vision/API feature coverage, and fresh plus retained-prefix measurements. Report quality, latency, acceptance and memory with throughput. Only measured benefits graduate into defaults.

## Intentional exclusions

Do not restore serving profiles, BF16/NVFP4 persistent KV options, old EXL3/GPTQ conversion paths, old model-specific MLA/compression assumptions, GLM DFlash2/MTP architecture, or Spark placement of the old Pro dSpark experts. V4.1 uses its official native checkpoint, fixed FP8 persistent KV, its own attention/engram/vision architecture, and all dSpark stages and experts on RTX. Reuse transport, ownership, scheduling and measurement mechanisms; adapt their model-facing contracts.

## Clarification: old packed batches and internal decode work

The optimized packed W4A16 paths in both pinned sibling engines already formed contiguous expert runs and submitted an entire batch through one b12x call. In GLM, `plan_packed_topk8_prefill_flat_with_block_rows` counts routes per expert, computes padded expert offsets and fills `packed_route_indices` plus `block_expert_ids`; DS4's `plan_packed_w4a16_topk8_prefill_flat` does the corresponding work. Their optimized prefill branches call `cuda_b12x_spark_w4a16_prefill_topk8_nvfp4_async` (or the direct FP8 response variant). This does not assert that every fallback path or every backend uses one CUDA kernel internally.

Both planners bypass that packing for single-row direct top-k execution. Their W4A16 GEMM scheduler computes `global_mn_tiles = route_blocks * n_tiles`, distributing output tiles as well as route blocks across the GPU. A single batched submission must not be confused with one compute task per expert. The current V4.1 deterministic fused task instead owns every intermediate slice and output column for its expert/M tile, which supplies only six compute tasks for one-row/top-six decode.

The V4.1 [output-split and direct-routing prototypes](ds41-expert-cost-profile.md) address those internal scheduling differences while retaining its BF16/FP8/router-weight boundaries and shared-memory intermediate. They do not restore a missing batch-submission mechanism, nor establish that all old W4A16 arithmetic or intermediate-storage strategies are appropriate for V4.1. Preserve expert-grouped runs and weight reuse when tuning larger prefill separately.
