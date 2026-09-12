# Fewer copies and waits between backbone blocks

FFN post-mixing now reads the completed TP4 reduction buffer directly. Previously,
each layer copied that result into a separate mHC scratch buffer first. Advancing
to the next layer now submits the residual and pre-state transfers on one stream
and waits once, instead of using two independently completed copies.

The arithmetic and output representation are unchanged. Result ownership remains
borrowed until post-mixing completes. Successful and partially failed enqueue
paths drain the stream before state or allocations can be reused. Input extent,
device and binding checks remain in place.

A full 40-layer target or verification stack removes 118 blocking copy calls:
40 disappear, and 78 become asynchronous copies. It removes 79 explicit stream
waits: the 40 result-copy waits and one of the two waits at each of 39 transitions.
The result-copy traffic falls by 409,600 bytes per input row per full stack.

## Measurements and qualification

Each mode uses three paired 599-token counting responses after 16k code context,
with baseline and candidate on the same RTX GPU. Every output and token usage
matches. Exact samples are in [the measurement record](ds41-block-boundaries.json).

| Decode tokens/s | Baseline | Candidate |
|---|---:|---:|
| Target, all three samples | 38.63 | 39.36 |
| Target, last two samples | 38.93 | 39.51 |
| dSpark, all three samples | 119.33 | 119.50 |
| dSpark, last two samples | 118.77 | 120.44 |

This is a small target-only improvement and a modest warm dSpark result; the
all-sample dSpark mean is essentially unchanged. These samples do not establish a
large or statistically precise speedup. An initial dSpark comparison used different
RTX devices; it is retained under `/tmp/ds41-boundary/speculative-long.json` but
is not used for the table.

The full-model C2/C6/C16 execution fixture passes, including serial/overlap logit
and draft-token equality, retirement and migration. The speculative API lifecycle
check and concurrent distinct-sequence cancellation/replacement qualifier pass.
Artifacts are under `/tmp/ds41-boundary`. Native kernels and Spark workers are
unchanged; this is coordinator sequencing work.

All eight paired quality outputs and token usages also match; both versions retain
the inherited Unicode-format failure. The frozen daemon
`/tmp/ds41-boundary/daemon` is now selected on 18041/18042, in containers
`ds41-boundary-live-target-api-dev` and `ds41-boundary-live-spec-api-dev`.
Its SHA-256 is `42e91194af2610140ec7fd4f7a42d021683cf36095a50a3c0cf7592d91e54dc8`.
Both selected APIs pass the lifecycle qualifier. The preceding scheduler
containers remain stopped for rollback; creation commands and live qualification
records are in `/tmp/ds41-boundary`. The subsequent
[RMSNorm rollout](ds41-rmsnorm.md) retains this daemon and selects a newer native
library on the normal APIs.
