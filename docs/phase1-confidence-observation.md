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

## Greedy divergence diagnostics

Add `ds41rt::logit_trace=debug` to record each verifier input, selected token,
and the two highest unconstrained target logits per row. Records include the
request identity, generated-token offset, committed context, and accepted input
count. Only rows through the accepted frontier describe the emitted trajectory;
rows after rejection use a different history. Constrained selections may differ
from the unconstrained top two.

The scan uses scores already downloaded for normal target selection and adds no
device transfers. It runs only when this separate trace target is enabled. It is
intended to compare the first divergence at identical token histories, not to
justify treating arbitrary output differences as harmless numerical drift.

Fresh-start repeats of the first code, fable, and topic requests reproduce each
fixed-length arm's earlier output exactly (six of six). The trace reconstructs
the streamed content, and paired requests are identical. The first divergence
has the same top-two candidates in both arms:

| Case | Output token offset (zero-based) | Limit 1 winner / margin | Limit 5 winner / margin |
|---|---:|---|---|
| Code | 35 | ` them` / 0.00754 | ` sorted` / 0.06940 |
| Fable | 2 | ` mango` / 0.58017 | ` lush` / 2.31431 |
| Topic | 1 | ` are` / 0.28997 | `'s` / 0.08312 |

Code is a close decision, but the fable difference cannot be characterized as a
tiny tie. Fable and topic diverge in the first verification pass, before any
length-dependent target commit or rollback. This narrows their investigation to
the proposed batch execution and its inputs; it does not establish the source
of the numerical difference or rule out a cache-read/masking defect. Next,
compare intermediate activations at the same first-pass rows across shapes.
Adaptive selection remains disabled pending this investigation.

[Paired scores and trace hashes](phase1-logit-comparison.json) preserve the
observations. Local raw files are `/tmp/ds41-phase1-logits-{1,5}.json` and
`/tmp/ds41-phase1-logits-{1,5}-trace.log`. Terminal matched EOS can commit one
more input than it emits: reconstruct output from emitted tokens (or stop at
EOS), never blindly interpret accepted-input count as emitted-token count.
The standard coordinator was restored after the probe.

## Layer-boundary activation tracing

For a disposable coordinator, enable `ds41rt::activation_trace=debug` with
`DS41RT_ACTIVATION_TRACE_POSITION` set to the first query position to inspect
and `DS41RT_ACTIVATION_TRACE_DIR` set to a writable diagnostic directory.
Matching passes write per-layer completed residual/pre buffers and token-position
metadata into batch/stage/row-count subdirectories. Existing buffer files are
never overwritten. This adds synchronous downloads and file writes; do not use
its timings as production throughput. Normal serving performs neither operation.

`scripts/compare-ds41-layer-residuals.py LEFT RIGHT --common-rows 2 --output JSON`
checks matching positions, BF16 geometry, finite values and layer coverage before
reporting per-row exact differences, RMS differences and maximum errors. Matching
positions alone do not prove matching token history; pair identical requests and
use the verification log to establish the shared prefix.
