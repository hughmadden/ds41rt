# Phase 1: uninstrumented adaptive comparison

The current adaptive candidate improves the selected C1 workload but fails the
mixed-C16 performance gate. It remains off by default. This is a focused probe,
not another full release qualification or proof of the Phase 1 throughput goals.

Both arms use the same frozen sparse-retention binary and corrected FP4 native
library, `RUST_LOG=warn`, RTX 400 W and standard memory speed (13365 MHz under
load, 14001 MHz maximum), and the same requests. Fixed mode uses five proposals;
adaptive mode selects one through five. Standard serving was restored afterward.

| C1 case, three samples | Fixed tok/s | Adaptive tok/s |
|---|---:|---:|
| Code | 119.61 | 121.58 |
| Fable | 45.21 | 52.52 |
| Topic | 62.49 | 65.52 |

All nine C1 outputs matched exactly, using held-out nonce seed 55001. Both arms
passed applicable named checks. Prose quality was not automatically assessed.

| Mixed traffic | Fixed tok/s | Adaptive tok/s | Exact paired outputs |
|---|---:|---:|---:|
| C4 | 101.34 | 100.85 | 4/4 |
| C16 | 167.84 | 144.70 | 6/16 |

Mixed batches cycle code, fable and topic with distinct first-content-token
nonces, seed 56001. All calls begin behind a barrier. Aggregate timing spans the
earliest first content through the final finish, including admission gaps.
Each arm has one batch per concurrency; output lengths differ at C16. All
applicable Python structure checks pass, but this does not qualify broad quality
or explain the ten nonidentical outputs.

A second fixed control measured C16 at 167.29 tok/s with 15/16 outputs matching
the first fixed run. An instrumented adaptive diagnostic measured 146.31 tok/s.
Scheduling variation exists but does not explain away the repeated performance
loss. The diagnostic selector costs 650 µs median / 698 µs maximum at C16,
versus 164.9 ms median whole verification. It trimmed in all 37 C16 decisions
with sufficient route history. Selector CPU work is not the dominant loss.

## Cache and lifecycle checks

Both arms retrieved the middle needle from a 32,815-token prompt, then reused
all 32,815 prompt tokens on an exact repeat. A completed-turn continuation
reused 32,821 tokens and returned the expected answer. Both passed eight
cancellations interleaved with eight exact-counting survivors at C16, followed
by a successful new request.

Cold needle TTFT was 6.09 seconds fixed and 6.00 seconds adaptive. The corresponding
prompt/TTFT proxies are 5,390 and 5,470 tok/s; these include HTTP and first-token
overhead, are single samples, and do not qualify overall prefill performance.

## Next: improve the cost representation

The mixed trace has a poor fit to the original linear row/unique-expert model.
Refitting its 228 eligible observations drives the nonnegative linear row term
to zero, while held-out cost error remains 10.3% median and 27.1% p90. This is
evidence to investigate the representation, not justification to replace the
coefficients with this same-workload fit.

Actual routing distinguishes unique experts from 16-row expert work groups.
Across complete C16 rounds, the sum of lane means has about 1.1 extra groups
beyond unique experts in mixed traffic, versus 7.25 in the earlier counting
trace. Counting repeats rows on relatively few experts; heterogeneous traffic
spreads them across many experts. Forecasting both group count and unique
experts can represent a distinction that a universal linear row term misses.
This is a supported next hypothesis, not proof that it explains the full loss.
Preserve explicit total-row costs and qualify kernel-shape transitions as well.

Do not fix the gate by simply disabling high-concurrency adaptivity or raising
its threshold. Extend the bounded forecast/cost model, validate predictions on
new workload splits, and compare serving behavior again. Greedy divergence at
C16 also remains an open numerical/lifecycle investigation.

[Raw comparison](phase1-adaptive-uninstrumented.json) and
[mixed cost diagnostic](phase1-adaptive-mixed-cost.json) retain the results.
`scripts/bench-ds41-adaptive-mixed.py` reproduces the focused workload;
`scripts/compare-ds41-adaptive-runs.py` checks input pairing and named output
checks without rescoring prose. Full HTTP records and diagnostic trace are in
`/tmp/ds41-phase1-uninstrumented*` and `/tmp/ds41-phase1-mixed-diagnostic*`.
