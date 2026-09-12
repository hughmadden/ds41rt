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
