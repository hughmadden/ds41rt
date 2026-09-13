# Cooperative attention completion

Retained on top of accepted chained queries (`b12bb3d`). Sparse attention,
output projection and mHC/FFN preparation are submitted together, then the lane
waits cooperatively before publishing normalized FFN input. `PendingLaneFfn`
retains the lane's consumers; drop and error paths drain before resetting them.
The enclosing unsafe preparation contract retains admitted cache slots, cache
producers and index storage until completion. Shared request/cache borrows end
before the wait. No new GPU allocation or stream is introduced.

The ordinary direct path remains available. Cold sparse graph setup is still
synchronous, as are cache production and index selection; this is one step
toward the complete asynchronous loop, not a completion claim.

## Verification

The distributed fixture drops queued layer-0 attention without polling, reuses
the same producers/cache batch/lane, completes attention and remote experts, and
advances through Engram into layer 1. Eight captured residual/pre/Engram/query
buffers match the direct baseline byte-for-byte. Existing reserved and paired
encoder checks also pass. These checks cover cancellation/reuse and numerical
behavior at the tested shapes, not every possible interleaving.

The first fixture attempt selected a fallback native library for the transport
and failed before attention execution. Setting `DS41RT_NATIVE_LIB` to the same
CUDA/RDMA library as `DS41RT_LAYER0_LIBRARY` resolved that environment problem;
both baseline and candidate fixture runs then passed.

C1 code matches exactly in all three repeats. Both serving arms pass needle,
prompt reuse, retained-turn, cancellation/survivor/recovery and high-thinking
constrained-output checks. Test and release builds pass.

## Initial serving pair

Adaptive dSpark, independent lanes, ordered admission; control then candidate.
Five RTX resident layers, unchanged KV pool, RTX 400 W and standard memory speed.
No build overlaps inference. One mixed batch per point; these are diagnostic
observations, not updated release tables.

| Metric | Chained-query baseline | Queued attention |
| --- | ---: | ---: |
| C1 code, median tok/s | 129.86 | 129.74 |
| Mixed C2, tok/s | 92.33 | 93.03 |
| Mixed C4, tok/s | 119.45 | 122.83 |
| Mixed C8, tok/s | 161.10 | 157.86 |
| Mixed C16, tok/s | 175.53 | 140.43 |
| HTTP startup readiness, seconds | 5.323 | 6.326 |
| Peak observed GPU memory, MiB | 96,876 | 96,878 |

Readiness is polled once per second. Internal target-weight loading is
1.777 → 1.850 seconds and local-expert loading is 1.696 → 1.702 seconds; the
HTTP observation alone does not isolate a one-second loading regression.
Comparing Docker's start timestamp to the application's ready log gives
5.176 → 5.266 seconds in the first pair and 5.198 → 5.255 seconds in the wider
sweep. The observed difference is 57–90 ms, not the whole polling interval.

The headline C16 loss is strongly sensitive to one completion tail: request 14
(topic) grows from 264 to 341 tokens, with its own decode duration increasing
from 14.69 to 19.54 seconds. The full measurement span grows from 18.316 to
23.136 seconds; the six code requests have nearly unchanged decode durations.
An **offline diagnostic** excluding request 14 from both arms gives spans of
17.594/17.640 seconds and 167.78/164.91 tok/s. This is not a new workload run or
a replacement score, and it does not establish the cause of the prose difference.
It shows why the full-batch scalar is not evidence of a uniform 20% slowdown.

The [broader C2–C16 sweep](phase1-queued-attention-curve.md), with candidate first
and control second, shows −0.7% to +4.1% changes through C14 and positive C15/C16
results. Together with the completion-tail diagnostic and passing functional
checks, this supports retaining the change toward the complete asynchronous
path. Preserve both observations; no uniform speedup is claimed.

[Initial evidence](phase1-queued-attention.json). Raw runs and frozen binaries:
`/home/tj/.cache/ds41rt-experiments/queued-attention`.
