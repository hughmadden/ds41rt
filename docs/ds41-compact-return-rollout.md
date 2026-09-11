# Compact-return development rollout

The compact BF16 response implementation at `182a3fe` is running on all four Sparks and both development APIs. Return payload is 12× smaller. Prefill improves on the measured short-context workload; target-only decode is essentially unchanged. Numerical equivalence is **not** established: full-model logits change noticeably despite passing the bounded text checks below.

## Deployment and checks

All workers use matching frozen executable and SM121 native-library hashes. Their forty layer loads total 36.5–37.9 seconds per rank (excluding other startup work). RTX uses a separately built SM120 library and frozen daemon. The previous binaries/libraries remain available for rollback. Current containers are `ds41-compact-target-worker` on ostrich/dodo/emu/kiwi, `ds41-native-api-dev` on raptor GPU0/port 18041 and `ds41-native-spec-api-dev` on GPU1/port 18042. Both APIs share the four workers, so qualification calls run sequentially.

- Replaying both saved real layer-0 requests on all four ranks yields compact BF16 partials exactly equal to ordered local summation of their old FP32 route outputs: eight comparisons, zero differing elements.
- The real full-target fixture executes 80 prefill rows for sixteen requests, then two sixteen-row decode passes, with finite logits and successful cache/engram commits. It passes its state checks; that is not proof of unchanged numerical output.
- Eight fresh before/after API cases retain identical text and usage in both modes. Target and dSpark also match each other. Six objective checks pass; translation and explanation remain bounded open-ended comparisons.
- JSON/SSE arithmetic, three counting streams per mode, cancellation recovery and unsupported-sampling rejection pass.

## Measured performance

| Measurement | Previous return format | Compact BF16 |
| --- | ---: | ---: |
| Target counting, three short greedy streams | Previously recorded 10.1–10.6 TPS | 10.19 / 10.11 / 10.47 TPS |
| dSpark counting, three short greedy streams | Previously recorded 40.6–43.2 TPS | 46.49 / 47.39 / 46.81 TPS |
| 1,363-token retrieval, target finish time | Fresh paired baseline 9.331 s | 6.269 s |
| Same retrieval, dSpark finish time | Fresh paired baseline 8.887 s | 5.920 s |
| Median 80-row target step | Earlier profile about 493 ms | 323.758 ms |
| Median 80-row collection H2D call time | Earlier profile 1,645 µs | 155 µs |
| Median 80-row remaining receive time | Earlier profile 7,900 µs | 5,317.5 µs |

The counting comparison uses the previous recorded baseline rather than a newly interleaved A/B run. The retrieval before/after calls were rerun for this rollout. These are C1 API workloads with fixed 80-row prefill chunks and short contexts, not the 8k-token prefill or C16 release benchmarks.

Current target collection medians are based on 680 eighty-row and 10,560 one-row layer collections. At one row, median receive is 1,409.5 µs, H2D 21 µs and reduction 6 µs; median complete target step is 97.190 ms. At eighty rows, median expert-boundary collection is 5,487.5 µs per layer. Receive includes waiting for Spark execution, response preparation and network delivery; it is **not** an isolated kernel or link time. Separate those costs before choosing the next expert optimization. Compacting payload alone does not resolve the remaining boundary latency.

## Numerical limitation

The sixteen-request synthetic fixture uses arbitrary token sequences. Its first, same-input prefill pass changes final logits by relative L2 0.5473, with seven of sixteen top-1 selections unchanged. Subsequent decode inputs diverge, so comparing later cycles' logits as if their histories were identical would be invalid; those cycles are excluded from the same-input comparison.

The real arithmetic chat prompt keeps target token 22 (`4`) followed by EOS 1. Its corresponding final-logit comparisons have relative L2 0.1461 and 0.2218, with cosine 0.9894 and 0.9752. These differences are larger than the layer-0 routed-result difference from the component audit. Exact rank replay and passing text checks narrow the investigation, but do not explain or excuse the full-model drift. Broader same-input, layerwise and independent-reference qualification remains required. Do not label this change mathematically equivalent or generally quality-qualified.

[Machine-readable results, artifact hashes, load times, quality outputs and phase timings](ds41-compact-return-rollout.json). Raw development observations remain in `/tmp/ds41-compact-*` on raptor, including the full GPU/test/server logs. The [component record](ds41-compact-return-component.md) documents arithmetic and protocol checks; this record supersedes its not-yet-deployed status.

## Next priorities

Keep expert execution as the first tuning focus, with separate budgets for packed-weight reads, expert GEMM, compact output, host staging, network, coordinator input preparation and final reduction. Qualify larger planned prefill capacities, real routing occupancy and wider fused execution before interpreting peak-bandwidth claims. Restore persistent verbs, streaming reduction and effective overlap from the older engines where applicable.

For the later attention pass, current [window ownership](../rust/crates/ds41rt-daemon/src/v41_window.rs) already allocates 128 FP8 entries plus K32 scales per layer/request slot at initialization. Forty layers and sixteen slots require 43,258,880 bytes, about 41.3 MiB, including per-slot counters. This excludes proposals, attention workspaces and compressed-source storage. [Backbone cache ownership](../rust/crates/ds41rt-daemon/src/v41_backbone_cache.rs) constructs four compressed sources at layers 2/8/14/20 and shares source references with consumers. Audit proposal/workspace allocation, source/index reuse, gather/layout overhead and decode/prefill kernels separately. Add model-specific CuTeDSL functionality in b12x when it is needed rather than constraining V4.1 to old attention kernels; keep plan-time policy in b12x and qualification tied to real memory traffic and outputs.
