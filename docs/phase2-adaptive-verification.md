# Adaptive verification comparison

2026-09-14. Source review uses the published image's pinned vLLM commit
`66c293578412417476f842c1da5805d3a3d959a8`, rather than an unpinned branch.

| Decision | Published vLLM implementation | Current ds41rt |
|---|---|---|
| Objective | Expected emitted tokens divided by draft plus verification cost | Same objective, with estimated serving overhead |
| Confidence | Previous published confidence for CPU budget; current confidence for GPU assignment | Current draft confidence |
| Cost | Profiled draft/verification tables, including compatible padded graph shapes | Legacy default; experimental per-backend fit using installed layer placement |
| Selection | Scan total draft budgets, then assign slots by cumulative survival probability | Greedy suffix-removal search over requests in one lane |
| Per-request length | Can differ; zero drafts permitted | Can differ; voluntarily retains at least one draft |
| Expert sharing | No route-reuse term in this selector | Forecast from the last six accepted target rows across forty layers |
| Coordination | TP ranks share confidence; asynchronous double-buffered CPU copies | Independent lane decisions; no cross-lane selection |

Source: [adaptive_verification.py at the image's pinned commit](https://github.com/local-inference-lab/vllm/blob/66c293578412417476f842c1da5805d3a3d959a8/vllm/v1/worker/gpu/spec_decode/adaptive_verification.py).
Its [graph-cost correction](https://github.com/local-inference-lab/vllm/pull/748)
addresses sparse profiles that otherwise make larger verification shapes appear
free. The image uses K7 with adaptive trimming, not fixed K7.

## Current cost fit needs recalibration

`DraftRuntime::select_prefixes` currently estimates microseconds as:

```
measured_draft_us + 1000 + 19864
    + 803 * total_verification_rows
    + 636 * mean_unique_experts
```

The fixed terms originate in `7fc10a26`; lane-local selection was retained in
`8d976950`. There is no GPU-count or RTX-resident-layer-count input. The expert
forecast averages all forty layers equally. That cannot represent the newly
measured difference between RTX encoder and Spark decoder costs. This is evidence
that the model is stale, not proof that its selected lengths are always wrong.

The policy searches all suffix-removal steps, including temporary predicted rate
losses, then trims only if its best shape beats the full prefix by more than 2%.
Missing accepted-route history keeps full prefixes. Confidence-cutoff and
reuse-adjusted cutoff modes are separate experimental options, not the default.

Four high-reasoning Sieve requests (one warmup plus three measured) logged 824
eligible decisions: 476 retained five drafts, 88 retained four, 176 retained
three and 84 retained two. Thus 42.2% of eligible decisions trimmed. Warm requests
measured approximately 120.53 tok/s median with policy logging enabled, consistent
with the 120.25 tok/s result without it. All emitted 802 tokens and stopped.

The fixed-K5 control measured 121.22 tok/s median across three warm requests,
also 802 tokens with natural stops. Its roughly 0.8% advantage over the 120.25
baseline is too small to establish a meaningful gain from separate short runs.
This does not explain the external Sieve gap. Policy logging was disabled for
the fixed control. [Raw summaries and artifact hashes](phase2-adaptive-verification.json).

## Experiment order

1. Record fixed-K5 control against the existing adaptive Sieve measurement.
2. Support seven proposals end to end: draft attention layout and embedding,
   stage/terminal storage, output extents, RNG ranges, route history and target
   verification. Merely widening the CLI limit is insufficient.
3. Calibrate timing for the current single- and dual-RTX layouts before evaluating
   adaptive K7. Separate encoder
   and decoder expert costs, preserve measured shape discontinuities, and include
   request count. Check predictions against held-out code and mixed workloads.
4. Compare fixed K7 and adaptive K7 with the same prompts, output checks and
   measurement windows; preserve independent lanes.

The one-RTX configuration also needs a fresh calibration: its current partial
RTX residency is not an input to the existing cost formula. No new policy default
is justified by the external throughput headline alone.

### Placement-aware calibration

The replacement cost model should sum layer costs using the **actual installed
expert placement**, rather than infer placement from GPU count or encoder/decoder
names. Its execution classes are Spark TP4, local RTX, and RTX TP2. Shared-expert
TP width is a separate feature: Spark routed experts can run alongside RTX TP2
shared experts. Hardware and topology identify a calibration profile; layer
placement determines which profile entries contribute to a candidate prefix.
Measured batch-shape boundaries and within-layer expert sharing must remain
visible. Concurrent shared/routed work must be priced by its combined elapsed
stage cost, not by adding overlapping GPU times.

The route forecast now exposes distinct experts per layer, preserving sharing
within each lane and charging independent lanes separately. The original mean
API and serving cost coefficients remain in use pending calibration.

`RUST_LOG=info,ds41rt::cost_model=debug` enables calibration records. Each layer
records its globally unique cache batch ID, layer, verification rows, installed
routed backend, shared TP width, distinct experts, 16-route group count, and
elapsed production/index/attention/expert/finish stages. Round records use the
same batch ID and add lane, request count, preparation time and total verification
time, so interleaved lane records can be joined without assuming log order.
These are elapsed serving-stage measurements, including scheduling and transport,
not isolated GPU kernel durations. The 16-route group count is a workload feature,
not an assertion that every backend uses that tile size.

Fixed-prefix runs also enable the existing adaptive route-capture and small-shape
graph paths under this logging target. This deliberately measures the path that
adaptive selection uses, without trimming prefixes. Instrumented throughput is
not a substitute for the final uninstrumented comparison. Normal logging does
not enable this extra capture, and no startup profiling pass or cross-lane wait
was added.

Validation: the optimized daemon build passed, and route-forecast tests passed
(9 passed, 1 ignored), including distinct layer counts and independent-lane
charging. A fixed-K5 serving smoke and C1/C2/C4/C8/C16 code sweep produced 1,374
complete round records / 54,960 layer records across both lanes. Every round
joined to exactly forty layers with matching row counts, correct installed
backends and TP width, and consistent elapsed stage totals. All 93 code checks
passed. This validates the instrumentation; it does not validate a replacement
cost fit or adaptive K7. Artifacts are under
`~/.cache/ds41rt-experiments/phase2-planner/placement-cost-*`.

## K7 implementation progress

Native draft attention, embedding and request/position transpose now have
compile-time K5/K7 specializations with width-selecting entry points. Legacy K5
symbols remain available. Rust FFI owners select the width before capture and
validate the corresponding buffer extents; default constructors retain K5 and
do not require the new symbols. No allocation or synchronization was added to
these kernel launch paths.

`v41_dspark_width_selftest.py` passed on both RTX cards for K5/K7 and request counts
1/3/8/16: FP32 attention reference, graph replay with changed inputs, exact layout
transpose, invalid geometry/alias rejection, and bitwise K5 legacy parity. Maximum
attention absolute error was 0.002562 (the native path rounds probabilities to
BF16). `v41_dspark_peer_embedding_selftest.py` passed both widths with GPU0's sole
embedding table and two independently replayed GPU1 graphs, including changed
and invalid seeds. The Rust FFI and daemon compile checks passed.

K7 is now connected through workspace sizing, terminal output extents, route
history, RNG reservations and target verification. The [initial fixed-K7
experiment](phase2-k7-experiment.md) records serving results and lifecycle checks.
The serving default remains K5. Placement-aware adaptive K5/K7 comparisons are now in progress; see below.


## Experimental placement profile and K7 batching

The optional `DS41RT_ADAPTIVE_COST_PROFILE` JSON file now supplies nonnegative
cost coefficients for each installed routed/shared backend combination. Startup
maps all forty layers from the actual transport installation: local RTX,
RTX TP2, or Spark TP4, with shared-expert TP width recorded separately. Moving
an expert layer therefore changes that layer's price. A separate per-round term
covers the one- or two-RTX non-expert path. The profile loads before serving;
selection uses lane-local route history and introduces no cross-lane wait.
Leaving the variable unset retains the legacy cost model. The measured
[experimental profile](phase2-adaptive-cost-profile.json) and
[fit report with trace hashes](phase2-adaptive-cost-fit.json) are included for
reproduction; the profile is not a recommended serving default yet.

`scripts/fit-ds41-placement-costs.py` fits odd fixed widths (K1/K3/K5/K7), holding
out even widths (K2/K4/K6). Code and mixed traffic both contribute training
observations. The initial code-only fit underpredicted mixed traffic, so it was
not adopted. Coefficients price the combined elapsed routed/shared expert stage,
not the sum of overlapping GPU durations. These are short-context measurements
on the current RTX PRO 6000 / four-Spark system at 400 W RTX power limits and
standard memory speed; they are not universal hardware coefficients.

The current affine candidate's held-out errors, using **observed** expert routes:

| Layout / workload | Legacy median / p90 error | Candidate median / p90 error |
|---|---:|---:|
| 1 RTX code | 8.5% / 14.3% | 4.5% / 16.8% |
| 1 RTX mixed | 20.5% / 35.3% | 6.0% / 26.2% |
| 2 RTX code | 7.8% / 26.2% | 4.2% / 16.0% |
| 2 RTX mixed | 7.6% / 26.5% | 7.6% / 19.9% |

The one-RTX K6/K7 calibration still uses the earlier 48-row sparse batching path
and must be refreshed before qualifying that layout at K7. In a separate dual
adaptive-K7 trace using **forecast** routes, code median/p90 error was
7.0%/24.1%; mixed was 13.8%/33.9%. The legacy formula on those same forecast
inputs gave 8.6%/25.8% and 15.0%/30.0%, respectively. Mixed forecast tail error
remains a limitation. Better observed-route fitting alone does not prove better
prefix decisions.

Sparse attention now supports batches through 64 rows, covering eight requests
with seven drafts plus an anchor each. K7 reserves the additional scratch before
KV planning; K5 retains its original 48-row allocations. This adds 20.08 MiB per
attention wave, or 40.16 MiB per GPU with two lanes. Both GPUs passed 264
byte-exact comparisons each against the bounded per-request path, including
changed graph descriptors, FP8 SWA, FP4/FP8 sources, and 48/56/60/64-row batches.
Reservation checks verified unchanged K5 accounting and exact K7 allocation
increments. The existing decode graph bank also retains all shapes through 64.

With corrected batching, instrumented fixed-K7 C16 code rose from approximately
1,145 to 1,180 aggregate tok/s; fixed K6 rose from 1,202 to 1,242. This is a
roughly 3% improvement in those wide code cases, not a general throughput claim.
The matched, uninstrumented three-run comparison of legacy K5, placement K5,
and placement K7 is the next serving gate. K7 currently has about 14.32 GB of
compressed-cache allocation representing 16,094,720 raw source positions, below
the approved 16×1,048,576-token target; K5 keeps its approved capacity. K7 is
therefore still experimental regardless of short-context throughput.

Raw artifacts: `~/.cache/ds41rt-experiments/phase2-planner/` under
`cost-calibration-v{1,2}-*`, `cost-calibration-v2-combined-{fit,profile}.json`,
`cost-adaptive-v2-rtx2-k7/forecast-validation.json`, and `sparse64-*`.


## Matched uninstrumented dual-RTX comparison

All arms used the same optimized daemon/native library, prompts, nonce seed,
and C1/C2/C4/C8/C16 order. Each arm ran three code samples per concurrency and
three mixed sweeps on one server process (the final K7 sweep failed). The table gives medians in
aggregate tok/s; the [raw summary](phase2-adaptive-serving-comparison.json)
includes individual samples and artifact hashes. Power limits were 400 W per
RTX with standard memory speed. These are focused experiments, not release tables.

| Workload / concurrency | Legacy adaptive K5 | Placement adaptive K5 | Placement adaptive K7 |
|---|---:|---:|---:|
| code C1 | 149.1 | 148.8 | 155.6 |
| code C2 | 248.2 | 240.1 | 257.4 |
| code C4 | 437.3 | 440.9 | 465.4 |
| code C8 | 714.9 | 714.3 | 737.6 |
| code C16 | 1155.1 | 1157.2 | 1176.1 |
| mixed C1 | 152.0 | 152.4 | 140.8 |
| mixed C2 | 143.7 | 146.8 | 118.2 |
| mixed C4 | 174.1 | 174.9 | 158.2 |
| mixed C8 | 194.1 | 213.4 | 191.5 |
| mixed C16 | 293.0 | 298.0 | failed / incomplete |

Placement K5 improves mixed C8 by approximately 10% in these medians, with code
mostly unchanged (C2 is about 3% lower). K7 improves code C1 by about 4% and C4
by about 6%, but its later mixed sweeps are weaker. Its third C16 mixed batch
failed the streaming gate. The original helper did not retain the failed SSE
response, and the server log contains no corresponding error, so the cause is
not yet established. The benchmark now records failed requests and incomplete
SSE events to distinguish empty stops from transport or usage failures. A repeat
of the same sequence is in progress. Neither the K7 mixed result nor the new
cost model is promoted to a serving default.
