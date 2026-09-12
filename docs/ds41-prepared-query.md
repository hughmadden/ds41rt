# Combined mHC/query graph experiment

The standalone combined graph is **not selected**. Capturing attention-input mHC
preparation and query projection together preserved values, but did not establish
a worthwhile end-to-end gain. Production source and the running RMSNorm APIs
remain unchanged. The candidate source patch and frozen daemon are retained under
`/tmp/ds41-prepared-query` for a future larger graph experiment.

## Scope and correctness

The candidate put mHC mixing, residual collapse and RMSNorm inside the query
CUDA graph. It retained one shape per layer and keyed the additional graph cache
by query weight owner, row count, unique mHC owner and mHC weight pointers. Owner
changes destroyed the drained graph before recapture; successful replay restored
mHC host state only after completion. Ordinary query graphs remained separate.

The real-weight query fixture passed 56 bit-exact cases across layers 0, 14 and
39, row counts 1–256, changed inputs, replay, and replacement of the mHC owner
with poisoned old buffers. The full-model two-lane fixture passed, including
C2/C6/C16 request ownership, retirement, migration and draft execution. Candidate
API lifecycle and distinct-sequence C2/C6/C16 cancellation/replacement checks
also passed. Eight paired target-only prompts preserved text and token usage;
the inherited Unicode answer-format failure persists. These checks do not
establish broad model quality.

## Performance

Each long comparison ran sequentially on the same GPU for that mode, counting
599 output tokens after 16k code context. Baseline and candidate used the same
selected native library and four Spark workers. Raw values, hashes and scope are
in [the result record](ds41-prepared-query.json).

| Measurement | Baseline | Candidate |
|---|---:|---:|
| Target C1, first pair | 38.55 TPS | 39.17 TPS |
| Target C1, warm pair | 39.92 TPS | 39.58 TPS |
| dSpark C1, last two of three pairs | 121.24 TPS | 121.82 TPS |
| Short-prompt dSpark C6, initial aggregate | 314.23 TPS | 303.01 TPS |
| Short-prompt dSpark C16, initial aggregate | 613.57 TPS | 611.54 TPS |

C6 repeats in candidate/baseline/baseline/candidate order measured
297.64 / 312.20 / 314.04 / 315.65 aggregate TPS. The candidate was restarted
before these repeats. Its final warm result recovered, but these samples do not
establish either a consistent gain or a persistent steady-state regression.
Aggregate timing includes admission gaps. Short-prompt concurrency results are
not comparable to the long-context C1 measurements as identical workloads.

Cold target prefill was 3,925 versus 2,292 tokens/s; the second pair was 6,885
versus 7,093 tokens/s. The initial candidate slowdown is retained in the record;
its cause was not isolated. Warm dSpark prefill remained around 7.1–7.2k tokens/s.

The small dSpark C1 change, inconclusive target C1 result and variable C6 result
do not justify retaining an additional owner-sensitive graph cache by itself.
A future graph change should remove more host boundaries and be checked against
both cold membership changes and warm concurrency. This experiment does not
reject larger graph capture or kernel fusion.
