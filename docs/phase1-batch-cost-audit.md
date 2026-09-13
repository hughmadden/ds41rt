# Batched-path cost and confidence audit

One RTX PRO 6000 Blackwell at **400 W and standard memory speed** (configured
maximum 14,001 MHz, no memory overclock), plus the four unchanged Sparks.
These are exploratory probes, not release qualification. **The coefficient
candidate was rejected and the serving source restored to the previous policy.**

## Confidence is already close on this corpus

A fresh fixed-five-proposal trace exposes suffix acceptance without the adaptive
selector censoring low-confidence suffixes. Labels are conditional: position j is
observed only if every earlier proposal matched. Initial warmup, terminal and
constrained decisions are excluded. The first four request IDs are the initial
C4 cohort, including its drain; later IDs are the C16 cohort. These are explicit
cohorts, not inferred prompt categories.

| Held-out cohort | Conditional labels | Raw expected emitted / observed | Raw Brier | Cross-cohort fitted Brier |
| --- | ---: | ---: | ---: | ---: |
| Initial C4 requests | 760 | 3.3154 / 3.3158 | 0.113019 | 0.112925 |
| Subsequent C16 requests | 2,961 | 2.9881 / 2.9718 | 0.124564 | 0.124490 |

The logistic transform is trained on the other cohort. Its intercept/slope are
−0.0048/0.9791 for the C4 holdout and 0.0080/0.9521 for the C16 holdout: both are
close to the current identity transform. There is no evidence here to justify
changing raw confidence. Mean calibration is not proof of per-decision accuracy,
and both cohorts share the same code/fable/topic corpus.

## Cost mismatch and candidate

On the current adaptive trace, warm two-request rounds have a median six total
verifier rows. The installed model predicts median verification of 44.15 ms;
measured verification is 57.87 ms. Median signed relative error is −25.3%.
This is the largest mismatch among the inspected 1/2/4/16-request groups.

Fitting the current fixed and adaptive traces, with every fifth contiguous
10-round block held out independently in each trace, produced this candidate:

| Cost coefficient, µs | Existing | Candidate (rounded) |
| --- | ---: | ---: |
| Intercept | 19,864 | 25,152 |
| Total verifier rows | 803 | 969 |
| Sum of per-lane mean unique experts | 636 | 382 |
| Additional lane | 3,168 | 12,273 |

Measured draft cost and the existing 1,000 µs allowance are added separately;
confidence, bounded joint suffix search, minimum prefix and 2% decision margin
were unchanged. An extra expert-group feature fitted to zero on these traces.
This does not prove expert-group counts are unimportant on other workloads.

The fit used 354 training rounds and held out 38 fixed / 40 adaptive rounds.
Accepted-history median/p90 relative errors were 2.7%/10.1% and 6.1%/13.7%.
Leave-trace-out errors were worse: 4.4%/15.2% and 7.6%/25.1%. On the inspected
two-request adaptive group the candidate predicts 53.03 ms, reducing median
underprediction to 8.6%. These are observational errors, not measured savings
from trimming the same draft batch.

## Serving rejected the global refit

Three no-thinking C1 runs per case, then one C4/C16 mixed and lifecycle probe,
with the same deterministic prefixes and frozen native library as the prior
batched-adaptive milestone:

| Workload | Existing adaptive | Candidate adaptive |
| --- | ---: | ---: |
| C1 code tok/s | 122.06 | 120.91 |
| C1 fable tok/s | 52.61 | 51.71 |
| C1 topic tok/s | 65.83 | 64.40 |
| Mixed C4 aggregate tok/s | 102.19 | 101.45 |
| Mixed C16 aggregate tok/s | 182.80 | 188.61 |

C16 improved 3.2%, but C4 did not improve and every C1 median fell 0.9–2.2%.
This is not enough evidence to promote a global coefficient replacement. The
candidate is retained only as [a reproduction patch](phase1-batch-cost-candidate.patch).
C1 outputs matched 9/9, C4 matched 4/4, and C16 matched 5/16; prose quality was
not scored. The candidate passed the 32K needle, prompt reuse, retained turn,
eight cancellations/eight counting survivors and recovery checks. Standard
serving was restored and the original-policy binary rebuilt afterward.

## The C4 aggregate gap needs stronger confirmation

The fresh **instrumented** fixed trace measured C4 100.71 tok/s, versus 100.67
for the earlier instrumented adaptive trace. That comparison does not reproduce
the earlier uninstrumented 109.21 versus 102.19 gap. Do not attribute the whole
aggregate difference to the two-request cost error.

The initial four requests' scheduler phases, excluding the first prefill round,
show where further paired measurement is useful:

| Active requests | Fixed rounds / emitted tokens / scheduler tok/s | Adaptive rounds / emitted tokens / scheduler tok/s |
| --- | ---: | ---: |
| 4 | 41 / 591 / 149.37 | 41 / 566 / 163.07 |
| 3 | — | 1 / 8 / 88.22 |
| 2 | 23 / 108 / 68.63 | 27 / 113 / 62.02 |
| 1 | 45 / 134 / 70.24 | 54 / 146 / 70.48 |

Both schedules emit 833 tokens after that initial round; their total measured
scheduler time is about 7.44 versus 7.46 seconds. Adaptation improves the initial
four-request phase but loses efficiency in the two-request tail. Different
schedules redistribute work and content across phases, so these phase rates
are not paired counterfactuals. The next useful measurement is fixed/shorter
prefix verification on matched small two-lane work, plus a repeated warm C4
comparison, rather than another confidence transform or global coefficient fit.

[The audit data](phase1-batch-cost-audit.json) records source hashes, calibration,
cost validation, phase accounting and paired serving checks. Reproduce with
`scripts/audit-ds41-draft-confidence.py`, `scripts/compare-ds41-route-cost-models.py`,
and `scripts/audit-ds41-cohort-drain.py`. Raw traces and runners remain in
`/tmp/ds41-phase1-batch-fixed-trace` and `/tmp/ds41-phase1-batch-cost`.
