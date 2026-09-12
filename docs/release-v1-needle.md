# Long-context needle retrieval

The corrected FP4 source-cache candidate retrieves deterministic needles from
fresh 32K, 128K, 512K, and 1.04M-token source contexts in both target-only and
dSpark modes. The needle placements cover 10%, 50%, and 90% of the source. All
eight cold requests return the exact code with zero cached prompt tokens. An
immediate repeat of each request also returns the exact code and reports the
entire prompt as cached.

| Source context | Needle position | Actual prompt tokens | Target cold / exact | dSpark cold / exact |
| ---: | ---: | ---: | ---: | ---: |
| 32,768 | 10% | 32,878 | 6.06 s / 1.59 s | 7.14 s / 1.01 s |
| 131,072 | 50% | 131,181 | 20.86 s / 2.07 s | 19.52 s / 1.05 s |
| 524,288 | 90% | 524,396 | 111.56 s / 3.28 s | 112.46 s / 1.71 s |
| 1,040,000 | 50% | 1,040,110 | 322.57 s / 5.19 s | 323.55 s / 3.20 s |

The harness enables thinking with high reasoning effort, uses temperature zero,
and requires the corrected FP4 system fingerprints. It derives each needle and
fresh-prefix discriminator deterministically, streams the response, requires an
exact answer, and validates usage accounting. The timings are single functional
observations rather than performance qualification.

The initial full invocation correctly rejected an unintended cache hit caused by
the preceding smoke test sharing the repeated source prefix. The harness now
adds a deterministic discriminator before the source. The completed matrix then
records zero cold-cache reuse. The failed invocation, smoke test, and completed
matrix remain identified in the [machine-readable summary](release-v1-needle.json).
The [compressed raw matrix](evidence/native-fp4-needle.json.gz) preserves every
request result, usage record, answer, fingerprint, and timing.
