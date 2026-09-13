# Accumulated asynchronous counting comparison

The first matched pair finds a concurrent throughput loss despite identical
counting output, token usage and fully warm prompt-cache hits. Do not dismiss
this as varying prose length or claim the release regression check has passed.
A fresh reversed-order pair at C4/C8/C16 is pending.

Both arms use adaptive dSpark, independent lanes, five RTX resident layers,
400 W power limit and standard memory clocks. Control is frozen e9c07ae;
candidate is 2957772, without the rejected combined attention graph. Each row
is the median of three samples. The prompt nonce is identical across arms.

| Concurrency | Control tok/s | Asynchronous tok/s | Change |
|---|---:|---:|---:|
| 1 | 150.94 | 150.59 | −0.23% |
| 2 | 225.27 | 223.62 | −0.73% |
| 4 | 407.07 | 369.15 | −9.31% |
| 8 | 572.16 | 538.49 | −5.88% |
| 16 | 932.35 | 880.42 | −5.57% |

Each sample includes the span from earliest first content to last completion,
including admission gaps. All responses exactly count 1–200; output and token
usage also match across arms. Three samples within a process do not establish
independence from startup, routing history or run order, hence the focused reverse
pair before assigning a cause.

[Evidence](phase1-async-counting.json) contains commands, artifact hashes, medians
and ranges. Raw records retain every response and timing. This comparison is a
regression investigation, not completed v2 release qualification.
