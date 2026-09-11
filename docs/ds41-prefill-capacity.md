# Prefill capacity and staging

Native serving accepts `--prefill-batch-tokens`; the existing AOT capacities for this path are 80, 256, 1024 and 4096. The compatibility default remains 80. The setting sizes target embedding, backbone/query/index/compressor execution, Engram staging and gates, target taps, transport and dSpark prefill state together. The head still processes only request-final rows. All Spark peers must support the selected capacity. Frame limits now allow up to 64 MiB; actual registered rings are sized by the existing negotiated request/response geometry, not unconditionally to that limit.

The four development Sparks now use the production `expertd-native` command with capacity 1024. Larger worker plans retain both the capacity-one decode kernel and a separately preallocated capacity-80 state for multirow decode. These states share immutable weights and input buffers, rebind together across layers, and retain their own scratch addresses. Their memory is included in the startup budget. Large requests use the full capacity state.

The router request arena and coordinator rank-upload arena now cover the entire planned batch capacity. That allows captured pinned router downloads and asynchronous pinned rank uploads for prefill, using the same existing stream/drain contracts as decode. Allocation remains at startup. Fixed FP8 request windows and the context-provisioned compressed pools are unchanged.

## Qualification

Both x86-64 coordinator and aarch64 worker release binaries build. The extended live RoCE fixture exercises rows 1, 6, 16, 80, 256, 1024, 6 and 1. All 24 completed waves have finite/nonzero rank planes and exact replay after eight abandoned dispatches and eight injected response-sink failures. Set `DS41RT_LIVE_ROCE_CAPACITY=1024` to reproduce this fixture against idle matching workers. It does not prove broad model numerical accuracy or concurrency.

Before the full-capacity pinned staging extension, both API smoke suites passed and all eight paired quality cases preserved text and token usage against the previous deployment. The inherited Unicode formatting failure and 7/8 cross-mode text agreement remain.

## Initial capacity sweep

One sequential C1 greedy counting request per context and mode, preceded by repeated `amber` filler; 59 generated tokens, no prefix-cache hits. This is a synthetic status workload. Effective prefill is prompt tokens divided by time to first content and includes tokenization/API/first-output overhead. It is not isolated GPU throughput. Runs are sequential and do not establish small percentage differences statistically.

At 16,411 prompt tokens, the previous 80-row deployment measured 668.8 target / 674.1 speculative effective prefill tokens/s. The initial 256-row trial measured 831.9 / 837.5, and 1024 measured 868.3 / 872.8. Retaining the small worker workspace measured 868.2 / 851.4 at 1024, with target decode 25.28 and speculative decode 89.91 tokens/s. The corresponding 256 trial measured around 837 speculative prefill and 89.25 speculative decode.

Short counting on the same new binaries/workers measured 35.56–35.78 target and 106.22–107.07 speculative tokens/s at coordinator capacity 80, versus 34.98–35.68 and 104.81–105.85 at 1024. This is much closer than comparison with the older deployment's short counting numbers. The older deployment gap is not explained solely by coordinator capacity and remains an open diagnostic.

## Stage attribution before wider pinned staging

An instrumented target run at 1024-row capacity completed the same 16,411-token prompt with 19.196 s to first output. Seventeen prefill steps accounted for 19.083 s. Across 680 layer executions, the expert stage took 10.962 s, index selection 3.607 s, and attention including projection/FFN preparation 3.833 s (sparse attention alone 3.288 s). Within the expert stage: routing/request preparation 1.371 s, dispatch 1.447 s, shared FFN 0.169 s, collection 7.972 s. Collection included 1.402 s of rank upload callbacks and 6.535 s of response waiting/progress. These are nested measurements: do not add parent and child totals, and response waiting includes remote expert compute rather than isolated network latency.

The fused expert slice implementation still groups sorted runs into M16 tiles. Larger batches create sharing but do not eliminate weight rereads between these tiles. A wider fused M tile is a concrete next experiment, alongside sparse index selection/attention. Neither unique weight bytes divided by elapsed time nor this host trace measures actual DRAM bandwidth.

Raw command arrays, immutable binaries, build logs, API responses, capacity sweep and traces live under `/tmp/ds41-prefill`. Workers use the unchanged `/tmp/ds41-slice-artifacts/cmake` native library; coordinators use the unchanged `/tmp/ds41-mhc-split/cmake` library. No b12x source or native kernel changes are included in this rollout.

## Full-capacity pinned staging result

The extended GPU fixtures pass 332 real-weight router comparisons and eight upload/reduction shapes (1, 6, 80, 81, 256, 1024, 4096, 6), with poisoned/reused arenas, exact output checks and error/drop drains. Observed Spark traces confirm the selected capacities: row 1 uses capacity 1, rows 6–80 use capacity 80, and larger rows use capacity 1024.

The final pinned-buffer coordinator measured 18.726 s target / 18.692 s speculative time to first content for the 16,411-token prompt: 876.4 / 878.0 effective prefill tokens/s. Decode was 24.44 / 89.70 tokens/s. The small difference from the preceding 1024-row trial is not an established standalone pinning speedup. Relative to the earlier 80-row deployment, the combined rollout reduces observed 16k TTFT by about 24% and improves effective prefill throughput by about 30%; it remains far below both the next 2–3k checkpoint and the original 8k release target.

Selected coordinator SHA256: `890267cea2ddcb667e337f0c73bccbfde012d649fed8fed6febf395de0ddd7c6`. Worker SHA256: `f60eb7c952dcd79b936419006608eac61d228d2bdc61d2f12f0f875463cdb1cd`. The coordinator includes the subsequent host-staging extension; the worker is the frozen production build with the small expert workspace. Current containers are `ds41-prefill-pinned-{target,spec}-api-dev` and `ds41-prefill-final-worker` on each Spark. The prior context/worker deployments remain stopped for rollback. The actual startup command arrays are `/tmp/ds41-prefill/create-pinned-commands.json` and `worker-final-create-commands.json`.

[Recorded measurements and stage totals](ds41-prefill-capacity.json).

Final pinned deployment: both API smoke suites pass, including streaming, cancellation/recovery and unsupported-sampling rejection. All eight paired quality cases preserve the preceding baseline text and token usage. The inherited strict-format failure and 7/8 cross-mode agreement remain; no broad quality claim is made.
