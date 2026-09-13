# Narrow FP8 projection split-K candidate

One RTX PRO 6000 Blackwell at **400 W, standard memory speed** (13,365 MHz
loaded), four unchanged Sparks, five complete local expert layers, 18 ×
1,048,576-token KV capacity and 24 retained snapshots. No builds overlapped
serving measurements. The standard service was restored after the experiment.

The candidate is published as `bae6e5cf` on our SparkInfer fork's `master`.
**The ds41rt source pin remains unchanged pending startup and C16 follow-up.**
The change selects four FP32 split-K planes and a 16×64 tile for native query-A
and KV projections at capacities 1 and 16 on the 188-SM device. Larger
capacities, other geometries and grouped projections retain existing plans.
Official FP8 weights and K32 activation quantization are preserved; the existing
native reducer sums the planes before one BF16 conversion.

In the captured component pipeline, including quantization and reduction, six
live rows under cache pressure improved query-A from 18.43 to 12.29 µs and KV
from 16.39 to 10.24 µs. All 100 component comparisons passed bounded native-output
checks with poisoned tails, fixed allocations and frozen kernel resolution.
The full native export changed only four small-capacity plans. Its capacity
qualification passed 44 comparisons across live-row transitions through 4096,
with worst relative L2 difference 1.51e-5 against the unchanged large plan.
These checks compare native implementations, not an independent mathematical
oracle or a broad model-quality suite.

Serving used separate ABBA sequences for adaptive dSpark and target-only, three
no-thinking code samples per arm. The table pools six samples per implementation;
prefill excludes each arm's warmup.

| Metric | Baseline | Candidate |
|---|---:|---:|
| C1 dSpark code, tok/s | 122.12 | 127.51 |
| C1 target-only code, tok/s | 43.52 | 44.00 |
| 32K fresh prefill, tok/s | 7,711 | 7,752 |
| C4 mixed aggregate, first / second arm, tok/s | 118.84 / 118.38 | 124.82 / 125.08 |
| C16 mixed aggregate, first / second arm, tok/s | 178.00 / 189.72 | 169.45 / 190.46 |
| dSpark startup, first / second arm, seconds | 6.32 / 7.32 | 10.38 / 8.34 |
| Target startup, first / second arm, seconds | 9.36 / 5.31 | 8.34 / 7.33 |

All dSpark code responses had identical hashes. Target baseline's third sample
in each arm differed from the candidate; all code structure checks passed.
Mixed response lengths changed, so aggregate rates do not represent identical
token sequences. Mixed completion checks establish serving completion and
nonempty output, not semantic quality. The first baseline/candidate pair also
passed retained-context needle, reuse, cancellation and recovery checks. All
prefill samples answered `7` with 32,768 new tokens and no cached tokens.

The C1 dSpark gain repeats, but C16's first candidate arm was lower and the
startup samples show a possible regression. Neither is dismissed as noise.
Before adopting the pin, compare matched warm C16 work and inspect startup
phases. Added plan scratch totals only 487,424 bytes per complete query-plan
owner, but that alone cannot establish unchanged loading speed.

[Evidence](phase1-narrow-fp8-split.json) preserves component/native checks,
serving commands, artifact identities, individual code/prefill samples and mixed
batch rates. Raw service logs and responses remain in
`/tmp/ds41-dense-split-serving`; native build and capacity evidence are in
`/tmp/ds41-dense-split-native`. Reproduce component probes with
`python/tools/bench_v41_dense_plans.py --projection q_a` or `--projection kv`
and explicit snapshot, native-library and output arguments.
