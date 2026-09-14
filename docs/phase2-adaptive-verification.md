# Adaptive verification comparison

2026-09-14. Source review uses the published image's pinned vLLM commit
`66c293578412417476f842c1da5805d3a3d959a8`, rather than an unpinned branch.

| Decision | Published vLLM implementation | Current ds41rt |
|---|---|---|
| Objective | Expected emitted tokens divided by draft plus verification cost | Same objective, with estimated serving overhead |
| Confidence | Previous published confidence for CPU budget; current confidence for GPU assignment | Current draft confidence |
| Cost | Profiled draft/verification tables, including compatible padded graph shapes | Fixed coefficients plus predicted distinct experts |
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
3. Compare fixed K7 and adaptive K7 with the same prompts, output checks and
   measurement windows; preserve independent lanes.
4. Calibrate timing for the current single- and dual-RTX layouts. Separate encoder
   and decoder expert costs, preserve measured shape discontinuities, and include
   request count. Check predictions against held-out code and mixed workloads.

The one-RTX configuration also needs a fresh calibration: its current partial
RTX residency is not an input to the existing cost formula. No new policy default
is justified by the external throughput headline alone.

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

These tests cover native components, not end-to-end K7 generation. Serving still
uses K5 until draft workspace sizing, terminal output extents, route history,
RNG reservations and target verification are updated and checked. No K7 throughput
result or serving default change is claimed.
