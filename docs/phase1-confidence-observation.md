# Native confidence observation

Enable `RUST_LOG=warn,ds41rt::timing=debug,ds41rt::draft_policy=debug`
on a disposable coordinator to observe the current fixed-length proposal path.
`native draft policy observation` records request identity, generated-token
offset, context length, lane and lane rows, raw five-position confidence,
matched prefix, actual committed inputs, constraints and termination flags.
`native scheduler round` supplies draft, prepare, verification and total time.

The raw confidence is a logit: the reference uses sigmoid to obtain conditional
confidence. Calibration must evaluate position j only when preceding proposals
matched. Later target predictions are conditioned on rejected draft history and
are not independent acceptance labels. Fully matching proposals are censored
beyond their verified length; EOS, budget-limited and constrained observations
must be identified separately. Verification timing covers both lanes together,
not an independently measured cost for each request or lane.

This opt-in trace performs one additional small confidence download per active
draft lane. It does not change proposal selection or cache commits. Normal
serving does not download confidence. Instrumented timings are attribution data;
compare performance with logging disabled. This is observation infrastructure,
not the adaptive verification policy or a claim of speedup.

Confidence records are retained by request identity across both lane proposals,
replaced at the next proposal for that request, and removed on release. Short
history and output-budget anchor-only paths clear any previous observation.

The initial fresh code baseline is recorded in `phase1-code-baseline.json`:
120.76 tok/s median, with all three code checks passing and zero cache hits.
Further baseline coverage and live confidence calibration remain outstanding.

## First live observation

The optimized observation binary ran against the published v1 native library
and its four resident Spark workers. All eight mixed-content requests completed
and passed applicable objective checks. The trace contains 332 scheduler rounds:
median instrumented draft time 2.865 ms, verification 38.180 ms, total 41.487 ms.
These medians include graph first use and cannot be subtracted to obtain an exact
overhead breakdown. This initial attribution points to verification as the larger
phase; it is not a new uninstrumented throughput result.

Excluding constrained, EOS and output-limit rounds, and censoring positions
after a mismatch, the first three positions have 316/226/155 observations.
Their observed conditional acceptance is 71.5%/68.6%/68.4%, compared with mean
sigmoid confidence 74.7%/68.8%/67.9%. Positions four and five have only 106 and
83 observations. This preliminary sample supports further calibration, not a
validated confidence policy. It is conditioned on the fixed-five proposal
trajectory and does not simulate all alternative shorter-policy frontiers.

Raw local records: `/tmp/ds41-phase1-confidence-corpus.json` and
`/tmp/ds41-phase1-observation.log`. Concurrent observation is the next step.

## Fixed-length cost controls

The native binary accepts `--dspark-draft-limit 1..5`, default 5. The control
retains the emitted anchor, generates the same five-position draft graph, and
verifies only the selected prefix (also bounded by remaining output budget).
It measures the cost of changing verifier rows without changing draft generation.
This is a fixed-policy comparison control, not adaptive selection.

`scripts/summarize-ds41-native-policy.py TRACE --output SUMMARY` groups complete
scheduler times by request count and total verifier rows, and reports per-position
confidence bins with conditional acceptance. It excludes constrained and terminal
observations from calibration and never labels unverified suffix positions.
The timings include instrumentation and first-use effects. They do not constitute
a counterfactual shorter-policy trajectory or a held-out throughput prediction.
