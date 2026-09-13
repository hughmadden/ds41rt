# Current C1 profile after decode polling changes

One RTX PRO 6000 Blackwell at **400 W, standard memory speed (no overclock)**,
four resident Sparks, official weights, source `df60262`. The frozen serving
binary/native library are the `after` artifacts in
[the polling evidence](phase1-poll-quantum.json). No serving implementation changed
for this profile. The uninstrumented references remain approximately **122 tok/s
speculative code and 42.8 tok/s target-only**.

## Complete host timing

Three code requests per mode, temperature zero, thinking disabled, matching
nonces, C16 admission, 24 retained snapshots and prefill batch 2,048. Only
single-request rounds with exactly 40 ordered layers and six verifier rows or
one target row are included: 123 speculative and 636 target rounds. Initial
prefill and partial rounds are excluded. Values are medians of each round's
summed host wall timings; **nested stages overlap and must not all be added**.

| Stage, ms per complete round | Speculative | Target-only |
| --- | ---: | ---: |
| Scheduler total | 42.65 | 23.48 |
| Draft generation | 2.87 | 0 |
| Complete target verification | 39.33 | 23.05 |
| Query preparation | 3.25 | 3.08 |
| Sparse attention | 1.80 | 1.83 |
| Attention output projection | 2.73 | 2.81 |
| FFN normalization/preparation | 1.28 | 1.26 |
| Routing | 1.50 | 1.38 |
| Shared expert | 2.26 | 2.08 |
| Remaining response receive wait | 20.32 | 5.52 |
| Receive-slot GPU upload | 0.16 | 0.20 |
| Expert reduction | 0.34 | 0.21 |
| Adjacent-layer advance copies/drain | 0.41 | 0.39 |

The receive wait includes useful remote computation and transport, after the
shared expert has run; it is not a removable polling-overhead estimate. Upload
alone is too small to explain the remaining C1 gap. Draft optimization alone
also cannot close that gap. Query and output projection remain substantial RTX
work; merely removing the adjacent-layer copy drain has a small upper bound.

## GPU attribution

Nsight Systems 2026.1.3 used explicit interactive start after one warmup and stop
after one complete code response, CUDA graph tracing at node granularity, no
CPU sampling or context-switch tracing. Unlike the host table, these captures
include the entire profiled request, including request startup. The target arm
uses a new nonce; the speculative arm reuses its 54-token prompt from warmup.
Both complete responses pass Python structure checks. The speculative benchmark
reports `passed=false` because its cold-cache gate sees those 54 cached tokens;
that is not an output failure. Profiled throughput is not promoted to benchmarks.

| Captured GPU kernel work | Speculative request | Target-only request |
| --- | ---: | ---: |
| All kernel duration summed, ms | 633.04 | 2,475.65 |
| Dense FP8 kernel duration, ms | 276.63 | 1,182.90 |
| Sparse attention kernel duration, ms | 54.97 | 293.96 |
| Draft attention kernel duration, ms | 27.42 | — |
| Stream synchronizations, calls | 36,338 | 183,464 |
| Graph launches, calls | 11,256 | 56,497 |
| Graph instantiations, calls | 336 | 2,222 |

The primary BF16 vocabulary kernel averages about 0.88 ms per call in both
captures. Draft attention averages 0.218 ms over 126 calls, with longer-window
calls around 0.242 ms. Its current implementation launches only 20 CTAs at C1,
uses about 100 KB shared memory per CTA, and processes up to three 64-key chunks
sequentially. Key partitioning is a concrete smaller experiment, but its total
contribution is modest compared with target execution and remote experts.

Graph instantiations persist despite warmed requests; they need attribution to
specific owners before changing graph retention. The target capture contains a
new prefill shape as well as steady decode, so its total cannot be described as
steady-state recaptures alone. Synchronization API time includes useful device
execution and profiler overhead. Summed GPU kernel time is not wall latency or
an estimate of removable overhead.

The next architectural investigation should join a larger query/attention/
projection/FFN-preparation region across completion boundaries, with explicit
stream dependency and lifetime contracts, while separately investigating the
six-row Spark cost. The 90/270 tok/s targets remain open.

## Evidence and reproduction

[JSON evidence](phase1-current-c1-profile.json) preserves source/report hashes,
all kernel/API summary rows, host timing medians and response checks. Recompute
the host table with `scripts/summarize-ds41-c1-timing.py --spec TRACE --target TRACE
--output NEW_JSON`. Raw host traces and the runner are in
`/tmp/ds41-phase1-c1-current`; the speculative GPU report is in
`/tmp/ds41-phase1-c1-nsys-interactive`, and target report/working runner are in
`/tmp/ds41-phase1-c1-nsys-target` and `/tmp/ds41-phase1-c1-nsys/target.py`.

The initial duration-based capture missed inference; its container lifetime also
interrupted a target response. It is excluded. Interactive launch must run as a
background process while start/stop run independently, and report export must
finish before stopping the container. A first synchronous-launch runner timed
out after the manually completed speculative capture; its finally block restored
serving. The corrected target runner completed normally and restored serving.
These are profiling harness issues, not accepted runtime changes. No full release
qualification suite was rerun.
