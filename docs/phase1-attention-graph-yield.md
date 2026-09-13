# Combined attention graph with yielding receives

Keep the combined attention/projection/mHC graph on the zero-spin decode
scheduler. The weighted eight-type score improves 80.75 → 81.46 tok/s (+0.88%),
with every case median higher; code improves 128.24 → 130.39. Each case has three
samples. This supersedes the decision to omit the graph under the earlier 50 µs
receive-polling scheduler; retain that [earlier comparison](phase1-attention-graph-revisit.md).

| Ordered mixed concurrency | Separate graphs tok/s | Combined graph tok/s |
|---|---:|---:|
| 2 | 99.46 | 94.75 |
| 6 | 141.56 | 144.43 |
| 8 | 161.16 | 163.46 |
| 16 | 164.56 | 176.89 |

The C2 loss remains part of the evidence. These fresh mixed runs use the same
ordered admission and nonce seed, but prose trajectories/lengths can vary; they
do not establish a uniform speedup. The baseline is the immediately preceding
zero-spin candidate comparison, not the old graph test with 50 µs receive spinning.

Both arms use adaptive independent dSpark lanes, five RTX expert layers, identical
native kernels, 400 W and standard memory clocks. The graph preserves arithmetic
and GPU allocations, retains captured consumers through cancellation, yields during
cold warmup and does not suspend inside capture. Its implementation is the same
five-file patch previously verified with 16 exact fixture outputs at 32/80 rows.
The rebuilt combination passes mixed lifecycle, cancellation/recovery, constraints
and indexed-context checks. Previously measured absence of blocking CUDA calls
applies to the exercised graph path; this is not a claim about all CPU allocations.

[Evidence](phase1-attention-graph-yield.json) preserves commands, identities,
weighted samples and mixed rates. Proceed to the clean v2 build and scoped
qualification; do not repeat the excluded prefill matrix or full suite.
