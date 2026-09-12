# Phase 1: single-RTX performance

Scope: one RTX PRO 6000 Blackwell coordinator and four DGX Spark workers,
using the official V4.1 Flash weights. Target 90 tok/s target-only and
270 tok/s C1 dSpark code generation, preserve prefill, and avoid substantial
high-concurrency throughput regressions. Commit and push incremental work on
dev. Phase 2 dual-RTX execution is forward-looking context, not this goal.

## Required adaptive verification (A4)

- Capture current uninstrumented code, mixed-content, prefill and concurrency
  baselines with fixed inputs, cache state, warmups and every timed sample.
- Profile complete draft, verification, commit and transport phases; distinguish
  instrumented attribution from uninstrumented throughput.
- Observe raw draft confidence, per-position acceptance, live verifier rows,
  actual expert reuse and complete verification cost. Confidence already exists
  in the draft graph; serving currently downloads only token IDs.
- Fit and validate acceptance calibration and a cost model. Actual target
  routes are unavailable before verification; measure prediction error for any
  expert-count estimate instead of assuming future routes are known.
- Select request prefix lengths jointly across each batch. Reconsider removable
  suffixes against updated total rows, expected accepted output and cost. Keep
  the emitted anchor and account for bucket transitions and both execution lanes.
- Use bounded host work or a compact GPU operation, avoiding per-token launches
  and new serial completion barriers. Retain a fixed-length comparison mode.
- Verify greedy output equivalence, constraints, output limits, cancellation,
  rollback, retained-prefix ownership and changing concurrency. Validate policy
  benefit on held-out workloads, including low-acceptance prose and C16.

## Profile-directed investigations

A1: larger queued dependency regions around remote FFN boundaries with explicit
buffer lifetime and publication states. Success is lower complete verification
latency, stable graph capture and correct cancellation, not fewer sync calls alone.

A2: pipeline Spark weight/scale loads across one, two and three stages while
sweeping slice widths. Preserve fused FC1/SwiGLU/FC2. Use real routing at target
M1, speculative C1, concurrent verification and prefill; inspect generated code,
transactions, stalls, occupancy and end-to-end impact. Investigate direct six-route
M1 only where it does not sacrifice expert reuse.

A3: profile RTX projections, vocabulary head plus sampling, and draft attention
at actual small-row and concurrent shapes. Preserve weight precision and
activation contracts; test context-dependent key partitioning with correct sink
and softmax merging for draft attention.

A5: measure receive/upload/publication overhead before endpoint changes. Evaluate
mapped receive consumption or narrower events only with correct visibility and
buffer lifetimes. Treat prefill microtile publication as a separate experiment.

## Completion evidence

Record source, dependency and artifact identities, official weights, 400 W power
limit, standard memory clocks, input shapes, confidence/acceptance, cache state,
warmups, sample distributions, memory and captures. Retain numerical and lifecycle
checks appropriate to each change. Do not infer completion from operator speedups
or a single easy decoding workload. Document achieved end-to-end rates and any
remaining gap to the requested targets.

Phase 2 may split encoder/decoder ownership across two RTX GPUs with TP2 shared
and encoder routed experts, partitioned KV sources and vocabulary head. Avoid
unnecessary single-device assumptions in new interfaces, but do not implement it
as part of Phase 1.
