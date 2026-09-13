# Phase 1: bounded expert-work forecast

The core now exposes an optional forecast of both unique experts and the number
of 16-route work groups. The serving policy still uses its existing forecast and
coefficients; this change adds no work to its current selection path.

The new forecast copies only accepted route history. It preserves immutable
snapshots, keeps lanes separate, and enforces the scheduler's eight-request
limit per lane. Seven nine-bit counters fit in each word. Even duplicate route
IDs can contribute at most 288 counts to one expert per lane, below the field
limit. An incremental evaluator applies only suffix rows that changed between
candidate evaluations, adjusting unique/group totals at zero and 16-row
boundaries. It supports arbitrary candidate order and restoration.

The initial full-rebuild implementation cost 10.66 ms median for synthetic C16
forecast plus joint search. Incremental evaluation reduced this to 1.02 ms
median, 1.65 ms maximum, over 25 optimized-build iterations with 905 cost
queries each. This is a host diagnostic, not a serving throughput improvement.
No per-token GPU launch or allocation is introduced.

## Model evidence

Offline trace parsing now retains route multiplicities, excludes insufficient
accepted history, and rejects duplicate/malformed layer observations. The new
comparison tool evaluates the original row/unique model against a model with
an additional work-group term. Both retain explicit row and lane features.

On block-held-out portions of the existing mixed trace, adding extra work groups
improves median accepted-history cost error from 14.25% to 11.02%; p90 moves from
37.04% to 34.55%. It does not improve leave-trace-out mixed error (14.43% median,
44.85% p90). These results justify retaining the feature for investigation,
not replacing serving coefficients or claiming the C16 regression is fixed.

There is another material confounder. Of 41 complete mixed C16 rounds, 34 have
sparse graph captures. Their median sparse-attention time is 36.61 ms versus
23.80 ms in the seven rounds without captures. The counting trace has 99 C16
rounds without captures at 24.38 ms. These subsets differ in row counts, routing
and warmup, so their differences are not controlled causal measurements.
Whole verification is not faster in the mixed no-capture subset. Nevertheless,
recapture and request-layout variability need direct qualification before
another serving-policy comparison. A group-only fit is insufficient.

## Validation and reproduction

Seventeen focused core tests pass (the explicit host benchmark is separately
opt-in). Tests compare the evaluator to scalar route counts across all layers,
lanes and changing prefixes; cover packed-field/group boundaries and maximum
duplicate counts; reject overfull lanes and duplicate requests; and verify
accepted-history retention, release, and immutable snapshots. Three Python tests
check rejected-suffix exclusion, insufficient history, and duplicate-layer input.
Daemon compilation also passes. No server or worker binary is replaced.

```sh
cargo test --manifest-path rust/Cargo.toml -p ds41rt-core dspark_
cargo test --release --manifest-path rust/Cargo.toml -p ds41rt-core \
  work_forecast_joint_search_cost_probe -- --ignored --nocapture
.venv/bin/python scripts/test-ds41-route-observations.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/compare-ds41-route-cost-models.py \
  --trace /tmp/ds41-phase1-sparse-graphs-5-trace.log \
  --trace /tmp/ds41-phase1-mixed-diagnostic-1-trace.log \
  --output /tmp/new-group-comparison.json
```

[Raw diagnostic results](phase1-expert-group-cost.json) record the model splits,
coefficients, trace hashes, recapture observations, and host benchmark.
