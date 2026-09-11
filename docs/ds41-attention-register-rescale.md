# Attention accumulator rescaling

Ported the qualified WMMA prototype into the current sparse-attention kernel,
retaining decoder-window bounds and all current ABI validation. Online-softmax
rescaling now loads a repeated per-head multiplier through the accumulator
fragment layout and multiplies each accumulator element in registers. Previously
each non-first valid KV tile stored and reloaded the full 16×512 FP32 accumulator
through shared memory to apply that multiplier. The new multiplier occupies only
16×16 FP32 values in shared memory. KV gathering, softmax order, probabilities,
final output storage and split merging keep their existing arithmetic.

The native library rebuilt with the pinned b12x revision. Only this CUDA source
was recompiled; the daemon and worker artifacts are unchanged. Resource inspection
shows ordinary attention registers decrease from 130 to 126 and split attention
from 166 to 128, without spills. Dynamic shared-memory allocation is unchanged;
these counts alone do not establish an occupancy gain.

## Kernel qualification

All 34 reference cases pass: ordinary/private attention, committed compressed
sources, and bounded decoder replay with two and ten key partitions. The harness
checks independent blockwise math for every row and the pinned TileLang reference
for the first min(rows,8) rows, including changed graph metadata and invalid
bounds/spans. Cases include 4096-row prefill. This is the same qualified numerical
tolerance as the prior implementation, not a new claim of exact reference math.

All 160 direct old/new comparisons are byte-identical. The reusable harness is
`scripts/compare-ds41-sparse-attention.py`. They cover 128/1024/2048/
4096-row prefill; 1/2/6/16-row split decode; unaligned FP8 values; window-only KV;
and masked, private, stale and varying-scale patterns. The full compressed pattern
measures roughly 19–20% less kernel time in this microbenchmark. For 2048 rows,
it drops from about 9718 to 7754 us. Timing uses CUDA graphs and twenty launches
per measurement; raw observations are not a statistical guarantee or a measured
DRAM-bandwidth claim.

## Serving comparison

Both binaries use the selected independent-chunk scheduler. Each library runs
in isolation with the other API pair stopped. No concurrent GPU work or compilation
runs during measurements. Eight sequential C1 requests contain 16,410 code or
16,411 repeated-text prompt tokens and generate 59 counting tokens. All prompt
hashes, text and usage match, with no prefix cache hits. The table reports the
second request of each kind/mode; rates are tok/s, decode shown baseline/candidate.

| Workload / mode | Baseline prefill | Register rescale prefill | Change | Short decode |
|---|---:|---:|---:|---:|
| code / target | 3746.36 | 4182.53 | +11.6% | 37.46 / 37.21 |
| code / speculative | 3813.19 | 4262.81 | +11.8% | 111.58 / 100.30 |
| repeated / target | 4090.13 | 4498.75 | +10.0% | 37.32 / 37.28 |
| repeated / speculative | 4099.03 | 4592.25 | +12.0% | 113.80 / 113.08 |

Both API lifecycle checks pass, including cancellation and recovery. All eight
quality cases retain each mode's previous text and usage. The inherited Unicode
format failure and differing open-ended explanation remain, so the strict quality
script still exits 1. A longer counting decode comparison uses 599 generated tokens after the same
16k code context. Text, prompt hashes and usage match for both modes. Target
throughput is 37.79 versus 37.64 tok/s, and dSpark is 119.02 versus 119.33 tok/s.
This does not show a substantial sustained counting-decode regression. The short
code dSpark sample still varies, and category-based evaluation remains open.

The library is selected on ports 18041/18042 in
`ds41-attention2048-live-target-api-dev` and
`ds41-attention2048-live-spec-api-dev`. The prior wavefront containers are stopped
and retained for rollback. The daemon is unchanged from the wavefront deployment.

Raw commands, frozen library, resource listings, reference checks and comparisons
are under `/tmp/ds41-attention-integrated`. [Portable evidence](ds41-attention-register-rescale.json)
contains the results and artifact hashes. The old prototype predates bounded CED
replay, so its complete source was not substituted for the current implementation.

Next prefill investigations remain adjacent-query KV reuse versus cache behavior,
chunk-pair/worker transfer overlap, and expert grouping under the resulting
schedule. Wider concurrency, category-based decode and release quality gates
remain open.
