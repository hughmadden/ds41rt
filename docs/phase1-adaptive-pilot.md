# Experimental adaptive serving pilot

`serve-native --dspark --dspark-adaptive` now connects the bounded joint selector
to native serving. It remains **off by default**. This pilot has not established
a serving speedup, and A4 is not complete.

The runtime captures already-downloaded router IDs only when adaptive mode is
enabled. After a successful target/draft commit, it records only accepted input
rows, retaining the last six per request and layer. Request release drops this
history. Forecasts union recent route sets within each lane; separately executing
lanes each pay their union cost. Missing history retains fixed draft lengths.
Confidence uses the reference sigmoid. The fitted transform slightly worsened
held-out Brier error, so it was not adopted.

The preliminary verification cost in microseconds is approximately
`19864 + 803*total_rows + 636*sum_of_lane_mean_unique_experts + 3168*extra_lane`.
The selector also includes measured draft time and a 1 ms overhead allowance,
and requires over 2% predicted improvement. These are exploratory coefficients,
not a universal hardware model. Same-workload held-out blocks using prior routing
history had 2.7% median and 6.6% p90 absolute relative cost error. Homogeneous
counting dominates this trace; heterogeneous traffic and alternate shorter
prefixes need further validation. `scripts/fit-ds41-adaptive-policy.py` reproduces
the fit and reports its split assumptions and limitations.

Both lanes generate proposals before joint selection and Engram preparation in
adaptive mode. Fixed mode keeps its existing early lane-zero Engram overlap.
Grammar-constrained batches currently retain fixed lengths because their
conditional acceptance has not been calibrated. Confidence is downloaded once
per draft lane in addition to the token download; transfer consolidation remains
an optimization to measure.

## Pilot findings

The first instrumented pilot allowed zero proposals, entering the separate M1
numerical path. Six of nine paired outputs matched fixed-five serving. All three
divergent requests had earlier taken an anchor-only step. Restricting voluntary
selection to **one through five proposals** restored **9/9 exact paired outputs**
on the same requests. Applicable objective checks pass in both pilots. This
controlled result supports the M1-path hypothesis; it is not a root-cause proof
for every operator in that path. Normal output-budget anchor-only steps remain.

Full C1 forecast and selection took **6 µs median, 15 µs maximum** in the bounded
pilot. It selected shorter prefixes in 505 of 679 observed decisions. The first
pilot also completed exact counting at C1/C4/C16, but that does not qualify the
final bounded policy or heterogeneous concurrency.

The current graph owners repeatedly replace a captured row shape when the
policy changes lengths. The first pilot shows this verification timing pattern:

| Verifier rows | Same shape as previous round | Changed shape |
|---|---:|---:|
| 2 | 26.95 ms | 38.44 ms |
| 3 | 30.17 ms | 41.79 ms |
| 4 | 33.00 ms | 45.15 ms |
| 6 | 39.48 ms | 51.45 ms |

These are instrumented medians, not matched operator-level attribution. The
roughly 12 ms transition penalty is consistent with the inspected graph
replacement paths and explains why a steady-shape cost fit cannot establish
pilot speedup. The bounded pilot measured code/fable/topic at 115.3/44.1/56.2
tok/s with debug instrumentation; these must not be presented as uninstrumented
before/after gains.

## Next gate: bounded graph reuse

Audit and retain qualified small row shapes with exact weight, buffer and state
owner identity. Relevant owners include `LayerGraphs` users (query, output,
shared expert, router and index query), window/compressor graphs, sparse attention
fingerprints and the vocabulary head. Do not reuse captures based on row count
alone where arguments include mutable owners or pointers. Bound retained small
shapes to the two-lane geometry, keep large-prefill retention bounded, and verify
replay after shape changes, cancellation and destruction. Measure captures,
complete verification latency, and graph memory before enabling adaptivity by
default. All original Phase 1 throughput and prefill requirements remain open.

[Evidence and hashes](phase1-adaptive-pilot.json) preserve the failed first pilot,
the bounded follow-up and calibration results. The standard v1 coordinator was
restored after both disposable runs.
