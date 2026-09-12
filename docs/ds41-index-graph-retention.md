# Per-layer index graph retention experiment

The combined September 12 candidate is **not selected**. It retained index
selection graphs by layer and prepared each lane's Engram work immediately after
its draft proposals, before proposing the other lane. Full-model correctness and
recovery checks passed, but the concurrent throughput comparison was worse than
the selected coordinator-stream-staging baseline.

Same-mode counting after 16k code context produced identical text and usage:

| Mode | Baseline TPS | Combined candidate TPS |
| --- | ---: | ---: |
| Target pair 1 | 41.68 | 41.83 |
| Target pair 2 | 41.84 | 41.88 |
| dSpark pair 1 | 125.07 | 124.83 |
| dSpark pair 2 | 124.39 | 124.93 |
| dSpark pair 3 | 123.73 | 124.93 |

Subsequent short-counting concurrent runs, in baseline/candidate/candidate/baseline
order, measured:

| Run | C6 aggregate TPS | C16 aggregate TPS |
| --- | ---: | ---: |
| Baseline 1 | 319.55 | 620.49 |
| Candidate 1 | 305.80 | 618.50 |
| Candidate 2 | 311.85 | 611.06 |
| Baseline 2 | 317.94 | 619.44 |

Each request generated the same 599 counting tokens. The first candidate run used
a freshly started process; its second run still lagged both baseline runs. These
are bounded comparisons, not category-wide quality or performance claims.

The full-model alternating-lane fixture passed in 11.29 seconds. Native API and
C2/C6/C16 cancellation/replacement checks passed. Eight same-mode target quality
cases preserved all text and usage; the objective Unicode check still fails in
both baseline and candidate.

The graph change used one retained entry per layer, validated the existing full
pointer/shape fingerprint at every replay, and invalidated published outputs at
batch boundaries without destroying all executable graphs. A separate timed C6
trace still captured 978 row-18 graphs over 100 full two-lane rounds. Candidate
width changes continue to invalidate entries; per-layer retention alone does not
remove that repeated setup.

The graph-retention source has been restored to baseline. Its complete tested
combined patch and daemon are preserved under `/tmp/ds41-index-graphs`, including
`candidate.patch`, for later investigation. Earlier Engram submission is being
tested independently; the combined result does not establish which change caused
the concurrency difference. Neither this result nor the standalone I/O tests
selects a new serving backend.

[Measurements and artifact identity](ds41-index-graph-retention.json) record the
frozen daemon and qualifications. The selected live APIs remain coordinator stream
staging with the previously qualified native RMSNorm library.
