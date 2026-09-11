# Keep healthy RoCE sessions across API requests

The serving owner previously reset all four QPs before every API admission. A warm 16k code-context profile still spent 488 ms reconnecting on its first expert dispatch. Admission now invalidates the previous output while retaining completed sessions. Request IDs and model/cache leases remain fresh. Errors from generation reset sessions, including cancellation between completed model steps; the existing pending-transport drop reset still handles abandoned dispatches and receive failures. Capacity growth retains the existing session-fit/reconnect behavior.

The candidate trace confirms first dispatch at 536.798 ms on startup and 0.920 ms on the next request. Startup and recovery still incur connection setup; successful repeated requests avoid it. This does not implement concurrent scheduling or prefix reuse.

## Validation and status measurements

The actual production daemon builds. All eight paired quality cases preserve preceding text and usage across varying request lengths. The inherited Unicode formatting failure and 7/8 cross-mode agreement remain; broad model quality is still unqualified. Both quiet APIs pass JSON, three streaming counting requests, cancellation followed by recovery, and unsupported-temperature rejection. Warm counting TTFT changes from 0.540–0.578 s to 0.091–0.093 s for target and 0.567–0.578 s to 0.093–0.094 s for dSpark. Short decode measures 35.59–36.09 target and 110.51–113.17 dSpark tok/s; these are status runs, not a statistical decode-speedup claim.

With stage logging enabled, two consecutive 16,410-token code-context requests measure 15.310/14.372 s to first content, or 1072/1142 effective prefill tok/s. The previous warm profile measured 14.992 s (1095 tok/s) but still reconnected. Same frozen prompt hashes, 59 completion tokens and counting text are retained. The per-dispatch trace establishes the removed setup cost more directly than noisy whole-request timing.

## Current prefill costs

The preceding deployed daemon was profiled on one warm, sequential C1 code-context request. Seventeen prefill steps produce 680 layer calls. Nested totals include 14.929 s of target steps, 10.560 s of expert phases and 3.032 s of sparse attention. Within the expert phase: router/request preparation 1.106 s, dispatch 1.194 s (including the 0.488 s reconnect), shared FFN 0.155 s and collection 8.102 s. Collection includes 1.013 s of host upload calls, 6.943 s of receive/progress and 0.139 s of reduction/stream drain. Do not sum parents with children or interpret receive/progress as pure network latency.

All four Spark logs have matching layer/row sequences for 680 prefill calls. Summing the largest GPU kernel interval across ranks for each corresponding call yields 4.920 s. The analogous maximum complete worker interval sums to 5.340 s. These exclude coordinator and network work; separate metric maxima need not occur on the same rank and must not be added as an exact decomposition.

For the 640 full 1024-row calls, median active experts are 268 and median maximum rows for one expert are 485. On ostrich, experts with more than 16 rows receive 83.58% of routes, while accounting for 29.86% of active-expert observations. Median GPU kernel time is 7.359 ms, with 0.123 ms compaction, 0.331 ms upload and 0.185 ms download. The other ranks are similar. This is substantial, skewed sharing; a uniform mixed-route fixture is insufficient to characterize the workload. Unique resident weight bytes are not actual DRAM traffic, and near-peak memory bandwidth remains unmeasured.

Next work should address coordinator/response handling and use the observed expert sharing in kernel experiments. The existing path copies pinned response payloads into another pinned upload arena; removing that copy requires retaining frame ownership through the CUDA stream drain, including on errors. Sparse attention remains another significant target. The original 8k prefill and 90/270 decode targets, nearer 2–3k prefill checkpoint and C16 scheduling remain open.

## Artifacts

Daemon SHA256: `b32065d1a29ad80e175f7a578674975584cee7a68312592fb66c22ac9fe9934f`. Native library remains `07db52dec294d18ee98e44b553a7c442a43cdabb65ff8103690572fc4eeaa42b`, b12x e5343cfb. Quiet APIs are `ds41-admission-{target,spec}-api-dev`, ports 18041/18042, batch 1024. The four capacity-4096 Spark workers retain their previous artifacts. Preceding and instrumented APIs remain stopped for rollback.

Frozen binaries, launch arrays, raw traces and API results are under `/tmp/ds41-admission-qp`; the preceding profile and four Spark logs are under `/tmp/ds41-current-profile`. [The evidence record](ds41-admission-qp-reuse.json) retains stage totals, expert distributions and API results. `scripts/bench-ds41-prefill-api.py --modes target` now supports targeted profiling without an extra speculative request; its default still measures both modes.
