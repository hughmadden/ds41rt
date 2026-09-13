# Queued draft setup and cold-chain warmup

Each draft lane stages seed IDs and sampling metadata in owned pinned host
storage, then queues these and the three stages' descriptors/positions on its
existing chain stream. The two eight-request lanes add **384 bytes of pinned
host staging**, with no extra GPU allocation or stream. Existing descriptor
staging is reused. Validation checks the entire RNG batch before reserving ranges;
failed or cancelled attempts consume their reserved ranges, while warmup/replay
reuse the same draws.

Cold warmup now becomes a pending phase. The poller waits through stream queries,
captures without suspension after completion, then queues graph replay and compact
token/confidence download. An armed owner retains all three cache read reservations
before the first upload and drains on errors, cancellation or unwinding. Normal
successful completion disarms the guard without synchronizing. Graph presence is
owned by the chain, replacing the serving layer's redundant captured-count set.
Direct C1 execution remains available.

The real-weight 16-request target fixture passes cold/warm queued replay and
cancellation/reuse across three cycles, including temperature 0.7. Tokens, full
logits and confidence match eager execution byte-for-byte. Full target/cache/Engram
commit and accepted-prefix ring checks also pass. The final release build includes
a follow-up preserving invalidation on failed direct uploads; that error-only
follow-up was compiled and included in the serving checks; its failure branch
was not injected, and the GPU fixture was not rerun. Both test and release builds pass.

Focused serving checks pass C1 code three times, C2/C8/C16 mixed and lifecycle
checks, four high-thinking constraint checks and concurrent 16K indexed-context
counting. C1 samples are **120.65, 130.32, 130.16 tok/s** (median 130.16), retaining
the slower first sample without assigning a cause. Mixed results are **93.81,
152.98, 163.69 tok/s** at C2/C8/C16. These are unpaired follow-up observations,
not a new full curve or release qualification; the original controls and losses
remain recorded. Standard serving was restored after all completed checks.

## CUDA audit

The same first-four-second, all-16-active audit uses a 14 GiB KV pool for profiler
headroom and five RTX layers. The first request finishes 7.14 seconds after the
capture request, beyond the analysis window. This is instrumented API wall time,
not throughput evidence.

| First four seconds | Projection/head checkpoint | Queued draft |
| --- | ---: | ---: |
| Synchronous memcpy calls | 432 | 0 |
| Stream-synchronize calls | 3,404 | 3,448 |
| Time in stream-synchronize APIs | 100.94 ms | 82.35 ms |
| Longest stream-synchronize call | 2.47 ms | 2.41 ms |
| Graph instantiation calls | 5,733 | 6,311 |
| Graph instantiation API time | 252.98 ms | 275.18 ms |

The synchronous uploads are gone from this window. Thousands of remaining
stream-synchronize calls still need removal or an ownership-based explanation
before the loop is considered complete. A separate call-stack diagnostic locates
active-lane calls in `NativeTp4Wave::dispatch_ffn` (the initial stream drain) and
`NativePendingFfn::finish` through `reduce_planes` (the GPU reduction drain).
These are the next changes. The diagnostic is not performance evidence; shutdown
destructor stacks after the workload are excluded from active-decode attribution.

[Fixture, serving and API evidence](phase1-queued-draft.json).
Raw reports/scripts: `/home/tj/.cache/ds41rt-experiments/queued-draft`.
Qualified README/report release tables are unchanged.
