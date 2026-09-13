# Yielding decode receives after asynchronous GPU completion

Keep the change to yield after every unsuccessful decode/MTP receive poll.
The previous 50 µs spin can delay the peer lane's short GPU completions now that
more component operations yield. Prefill, benchmark and mixed-prefill waves retain
250 µs; ready receives complete immediately. No GPU storage or arithmetic changes.

Weighted eight-type performance is flat: 80.77 → 80.75 tok/s, three repeats per
case, all checks pass. Exact-output counting recovers versus the asynchronous
50 µs baseline: C4 approximately 365–369 → 385, C8 535–538 → 584, C16
877–880 → 938 tok/s. Against the older e9c07ae control, C4 remains below 405–407;
this does not explain all of the accumulated cost.

| Ordered mixed concurrency | 50 µs tok/s | Yield each failed poll tok/s |
|---|---:|---:|
| 2 | 90.45 | 99.46 |
| 6 | 133.12 | 141.56 |
| 8 | 152.30 | 161.16 |
| 16 | 179.56 | 164.56 |

Keep the C16 loss visible. This mixed pair does not establish a uniform speedup;
prose lengths and trajectories vary, and earlier fresh C16 controls span a wide
range. Weighted and real-workload results take priority over counting alone.
The controlled change supports receive spinning as a source of scheduling cost,
but timing logs include overlapping lane work and do not prove exclusive causality.

Both arms use adaptive independent lanes, five RTX layers, the same native
library, 400 W and standard memory clocks. Candidate lifecycle, cancellation,
constraints and indexed-context checks pass. The existing transport test verifies
decode polling and preservation of prefill/mixed-wave behavior; release build passes.

[Evidence](phase1-async-poll-yield.json) preserves commands, identities and summaries;
full responses and logs remain in the raw directory. Revisit combined attention
graph fusion against this scheduler before final clean v2 qualification, because
host polling can affect the benefit of changed GPU submission/completion boundaries.
