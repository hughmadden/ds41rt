# Concurrent scheduler selected on the normal APIs

Source `ef37942` is selected on ports 18041 (target-only) and 18042 (dSpark).
The frozen daemon is `/tmp/ds41-scheduler-rollout/daemon`, SHA-256
`016f6073a19af3ccec3fe39085a80b9977ba502e5d0cc2faa51889e30461778c`.
Native kernels and the four registered-output Spark workers are unchanged.
Containers are `ds41-scheduler-live-target-api-dev` and
`ds41-scheduler-live-spec-api-dev`. The preceding stream2048 containers remain
stopped for rollback. Both selected APIs pass the native lifecycle qualifier,
including cancellation recovery.

## Paired long-context regression check

All paired output text and token usage match. Prefill uses 16k code or repeated
text and three samples per side; means below use the last two. Long decode uses
two 599-token counting responses after 16k code context. Both baseline and
candidate were measured sequentially against the same Spark workers.

| Measurement | Baseline | Scheduler |
|---|---:|---:|
| Target code prefill, tokens/s | 6,835 | 7,010 |
| Target repeated prefill, tokens/s | 8,043 | 8,009 |
| dSpark code prefill, tokens/s | 7,013 | 6,987 |
| dSpark repeated prefill, tokens/s | 7,777 | 8,112 |
| Target long decode, tokens/s | 38.72 | 38.92 |
| dSpark long decode, tokens/s | 119.92 | 117.77 |

The dSpark long-decode mean is about 1.8% lower in this comparison; this rollout
is selected for concurrent serving, not a demonstrated C1 latency improvement.
The separate short-prompt counting result reaches 605.72 aggregate tokens/s at
C16; see [concurrent serving](ds41-concurrent-serving.md) for its scope and limits.

Raw paired records, creation commands and post-selection checks are in
`/tmp/ds41-scheduler-rollout`. The [measurement summary](ds41-scheduler-rollout.json)
contains the exact means and daemon identity. Mixed prefill scheduling,
independent lane progress, slow-client isolation and broad/sustained quality
remain open, as do the 90/270 TPS C1 targets.

## Real expert workload behind the C1 profile

The earlier 16k counting profile was correlated with all four workers' timestamped
execution records, restricted to the last request's decode interval. Each rank
provides roughly 23.9k one-row samples and 4k six-row samples; malformed/interleaved
log lines are excluded. These are real routed batches, not a fixed six-expert
fixture.

| Per-rank mean | Target one row | dSpark six rows |
|---|---:|---:|
| Active experts | 6 | 22.35–22.37 |
| Expert kernel | 156.6–157.6 µs | 500.1–509.5 µs |
| Output compaction | 4.1 µs | 6.2–6.4 µs |
| Worker total | 180.5–181.6 µs | 528.4–539.4 µs |
| Unique-weight rate | 199–200 GB/s | 229–233 GB/s |

The last row divides instrumented unique expert-weight bytes by kernel duration;
it is not a hardware DRAM counter and excludes repeat tile reads/cache effects.
Most of the six-row expert wait is real computation and weight access. The older
small fixed-expert fixture cannot be subtracted from this workload to infer
transport overhead. RTX query/attention work and synchronization remain major
decode optimization targets.

## RTX synchronization evidence

A separate three-second target-only Nsight Systems capture is retained at
`/tmp/ds41-nsys/target.nsys-rep`, with CSV summaries and SQLite export alongside it.
It records 118,586 `cudaStreamSynchronize` calls, 28,637 `cudaGraphLaunch` calls
and 54,343 blocking `cudaMemcpy` calls. The interval contains 109 logit downloads,
giving approximately 1,088 synchronizations and 263 graph launches per decoded
token. Stream synchronization accounts for 1.59 seconds of host API duration,
but that includes waiting for useful device execution and is not a removable
overhead estimate.

This capture uses graph-level tracing, so the direct-kernel summary does not
attribute all work inside graphs. It supports reducing inter-operation waits and
combining execution stages as the next investigation; it does not establish a
kernel-specific speedup or a complete GPU utilization estimate. The profiler's
bounded container lifetime truncated that diagnostic prompt; API correctness is
established by the separate full-response qualifiers above.
