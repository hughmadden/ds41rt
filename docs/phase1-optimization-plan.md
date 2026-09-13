# Phase 1: single-RTX performance

Scope: one RTX PRO 6000 Blackwell coordinator and four DGX Spark workers,
using the official V4.1 Flash weights. Target 90 tok/s target-only and
270 tok/s C1 dSpark code generation, preserve prefill, and avoid substantial
high-concurrency throughput regressions. Commit and push incremental work on
dev. Milestone commit messages should include approximate before → after values
for the metrics already measured, labeling instrumented probes and allowing a
summary across several commits. Do not run extra tests solely for commit-message
numbers. Diagnostic-only work should not imply a measured speedup. Phase 2 dual-RTX execution is forward-looking context, not this goal.

## V2 completion sequence

The current release objective is to finish removing cross-lane dependencies,
except new-request admission and prefill, then revisit plausible C1 improvements
whose concurrent results were mixed or unfavorable. Use adaptive dSpark with
independent decisions for all new comparisons. The historical throughput targets
above remain performance targets, not evidence that they have been achieved.

Remaining synchronization work, based on the current serving source:

- Queued dSpark main-context/cache commit and target SWA publication are implemented
  and have focused numerical/lifecycle evidence. Draft replay also has separate
  lane workspaces. The scoped v2 performance comparison remains pending.
- Compressed-cache publication is now queued with producer-owned upload storage
  and page/slot reservations. Disjoint plans can publish out of order; partial
  errors drain before rollback. Numerical, ownership and cache-recovery tests pass.
- Completed/cancelled requests now retire within their owning lane, without a
  peer drain. Snapshot arenas remove allocation/free calls from retention;
  [snapshot copies now complete cooperatively](phase1-queued-snapshots.md), with
  source-slot ownership retained until completion or drained abort.
- [Constrained/full-frontier logits downloads](phase1-queued-logits.md) now use
  cooperative transfers with head-owned pinned storage and preserve full retained
  scores for future grammars.
- [Attention/projection/mHC completion](phase1-queued-attention.md) is now cooperative
  with a retained lane owner and drained cancellation. Numerical/lifecycle checks
  pass; the complete C2–C16 curve supports retention despite an initial C16
  scalar dominated by a longer prose completion tail.
- [Queued cache production](phase1-queued-cache-production.md) is a development
  checkpoint: numerical/lifecycle checks pass, but median C2–C14 is about 3%
  lower. It is not release-qualified. Preserve frozen e9c07ae for the combined
  producer/index comparison; investigate a common GPU submission/completion
  boundary rather than accumulating more host waits.
- Finish index selection, graph capture/rebind/eviction,
  and remaining component waits on the
  shared host thread. Retain coordinated fatal-error cleanup and admission/prefill drains.

The [separate cooperative embedding/handoff candidate](phase1-cooperative-embedding.md)
was rejected after a consistent 2.3% C8 decline across three pairs. Its patch and
exact-byte checks are archived; serving remains on the prior implementation.
The subsequent [chained embedding/query change](phase1-chained-query.md)
was initially rejected after three C16 losses, then **accepted by the user**
after the [complete C2–C16 curve](phase1-chained-query-curve.md) reversed the C16
result (132.30 → 162.05 tok/s). It is now integrated into serving source. Keep
both sets of evidence: the initial gate does not establish a general regression,
and the wider sweep does not establish a uniform speedup. Intermediate batches
change nonce consumption and warmup history.

When a narrow performance gate is unclear but a change advances the required
larger design, investigate a broader workload/concurrency curve before rejecting
it. Distinguish workload-sensitive variation from a repeatable implementation
problem; preserve correctness and performance evidence without abandoning the
required asynchronous design. Next finish adjacent blocking cache/index/attention
work identified in the [wait audit](phase1-lane-wait-audit.md).
Remaining blocking decode waits on the shared host thread are unfinished work,
even when they synchronize only one lane’s stream; removing the scheduler’s
round barrier alone is insufficient.

After these changes, first revisit the combined attention graph, which previously
showed small target/adaptive C1 gains but a C16 decline. Rank other archived
candidates by actual C1 serving evidence; component-only Spark FC2 gains require
serving integration and are not established end-to-end wins. Local expert graphs
previously showed no C1 gain and have lower priority. Preserve rejected-candidate
records and record new measurements separately.
Use ordered admission for mixed-workload comparisons: simultaneous client threads
can change initial lane groupings and expert sharing even with identical prompts.
Record admission mode and timestamps; keep unordered arrivals for lifecycle/race
coverage rather than treating them as controlled grouping comparisons.

For v2, rerun only the benchmark tables in the main README **excluding the prefill
matrix**, and update matching performance-report tables with the same data.
Rerun tool calling at high concurrency with thinking enabled/high; do not rerun
the full qualification suite. Add RTX resident layer count to the headline
table, retaining power limit, standard memory speed, and KV bytes/token capacity.
Publish the v2 GitHub release and matching serving container with concise release
notes listing optimizations that demonstrated an impact. Verify the clean build
and standard run path. A measured memory requirement may reduce the KV pool by
the necessary amount; it need not remain a multiple of one million tokens.

## Bottom-up RTX layer placement

Use excess RTX memory for complete routed-expert layers, starting at layer 0.
Phase 1 runs those layers on the single RTX; leave TP2 weight splitting and the
other dual-device changes to Phase 2. Select placement at startup after budgeting
mandatory weights, both execution lanes, vision, dSpark, KV, load staging and
peak runtime workspace. Preserve explicit user memory and KV overrides.

The default aggregate source pool starts from 18 maximum-context equivalents at
C16, trading snapshot arena bytes for global pages. With dSpark it now provides
18,710,016 tokens plus private tails in a 16.681 GB global pool, with 24 retained
completed-turn and prompt-snapshot entries and 139.4 MiB of snapshot arenas.
Use remaining memory aggressively, but determine headroom from measured runtime
and loading peaks rather than adopting an unverified occupancy percentage.
Whole-layer granularity can leave space that cannot accommodate another layer.

Keep loading extremely fast. Measure cold and warm startup against the existing
configuration, preserve bounded parallel reads and packing, and allow Sparks to
omit locally owned layers once placement and worker compatibility are validated.
Report the chosen RTX layer range, actual Spark memory budget, and KV bytes with
the represented token capacity in final configuration headlines. These are
qualification requirements, not claims about the current deployment.

Qualify numerical behavior, tool/needle checks, retained-context lifecycle,
C1 target/dSpark, prefill and concurrency before promoting local placement.
Once placement works, profile longer RTX chains through local experts and across
layers using A1 ownership rules; do not combine that experiment with the initial
placement comparison.

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
  suffixes against updated total rows, expected accepted output and cost within
  each lane. Keep the emitted anchor and account for bucket transitions. Lanes
  must advance independently; no joint cross-lane selection or decode-round join.
- Use bounded host work or a compact GPU operation, avoiding per-token launches
  and new serial completion barriers. Retain a fixed-length comparison mode.
- Verify greedy output equivalence, constraints, output limits, cancellation,
  rollback, retained-prefix ownership and changing concurrency. Validate policy
  benefit on held-out workloads, including low-acceptance prose and C16.

## Profile-directed investigations

A1: larger queued dependency regions around remote FFN boundaries with explicit
buffer lifetime and publication states. Success is lower complete verification
latency, stable graph capture and correct cancellation, not fewer sync calls alone.

The joint decode path is now removed. Unconstrained decode now uses
[GPU top-1 and compact transfers](phase1-compact-head-selection.md), with
cooperative per-lane head completion. Continue through remaining blocking
completion boundaries: constrained logits download and target/draft-cache commit.
[Shared draft replay now completes cooperatively](phase1-async-draft-completion.md).
The [two lane-local draft workspaces](phase1-split-draft-workspaces.md) now share
weights and use forty-row storage for eight requests each.
[Queued dSpark commit](phase1-queued-draft-commit.md) prepares and writes accepted
KV from each lane's producer; [queued SWA commit](phase1-queued-window-commit.md)
removes the sequential window waits. [Queued compressed-source publication](phase1-queued-source-commit.md)
claims pages until completion/rollback, preserves shared-tail copy ownership and
uses producer-owned upload staging. [Lane-local retirement](phase1-lane-retirement.md)
now releases completed requests without draining the peer, and avoids redundant
device length clears. [Snapshot arenas](phase1-snapshot-arenas.md) now remove CUDA
allocation/free calls from retention, trading their bounded storage for default
global KV pages. [Queued snapshot copies](phase1-queued-snapshots.md) now retain
source ownership while polling per-lane streams and publish only after both cache
owners complete. Next finish the remaining blocking transfer and component-wait audit.
Constrained and retained-frontier logits transfers are now cooperative as well;
continue with token upload and the embedding/query handoff.
Queue work and
poll completion while retaining exclusive buffer ownership; cancellation must
drain before releasing storage. Keep admission at complete pass boundaries and
expert sharing within each lane. Measure complete C2–C16 serving, preserve C1 and
prefill, and revisit earlier concurrency experiments only when the changed
completion path provides a concrete reason.

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

The [combined cache/index checkpoint](phase1-queued-index.md) now overlaps index
projection with cache production and polls selection with guarded ownership.
Against preserved e9c07ae: C1 130.18 → 128.43 tok/s, median C2–C14 −0.34%,
C16 150.25 → 131.40. Functional checks pass; performance remains unresolved.
Keep that control for remaining FFN/layer-completion work and final qualification.
