# Twenty-four retained completed turns

Native serving defaults to 24 completed-turn snapshots, independently of the
configured concurrency. This keeps sixteen recently active conversations and
eight spare turns eligible for exact continuation. `--prefix-cache-entries N`
accepts 0 through 128; zero disables reuse.

Complete-prompt repeats have a separate bank with the same limit. A request's
prompt snapshot therefore cannot consume its completed-turn slot. Each bank
uses LRU eviction, replacing an identical frontier without adding an entry.
Lookup selects the snapshot that skips the most execution, accounting for
compression alignment and bounded replay. Completed turns win ties.

Under shared source-page pressure, prompt snapshots are evicted first, followed
by completed turns. Retention is a maximum, not a guarantee that all histories
fit an explicitly reduced pool. Shared pages remain alive until their final
owner releases them. Snapshot eviction frees retained SWAs, compressor/history
state and optional dSpark windows as well as its page references.

## Memory accounting

The global-pool default remains `(concurrency + 8)` configured-context
equivalents, plus `concurrency + 2 × retention` spare page groups for active
copy-on-write and retained partial tails. With concurrency 16 and retention 24,
this adds 64 spare groups. In the final compressed-cache format, one group
occupies 455,680 bytes.

At the default 1,048,576-token context, the source page counts become
49,216 / 49,216 / 49,216 / 98,432, totaling 22,426,746,880 global index/KV bytes.
Together with FP8 SWA, page tables and compressor tails, the eager cache is
22,471,251,456 bytes. Historical FP8 measurements in the original JSON retain
their original sizes; the [final memory report](release-v1-memory.md) supersedes
them for release sizing.

Each target snapshot also allocates 2,720,064 bytes for forty SWA windows and
four compressor tails. The two full 24-entry banks use 130,563,072 target-tail
bytes; dSpark adds up to 202,752 bytes per snapshot for its three retained
windows. These allocations use the existing 2 GiB runtime headroom and are
separate from the global-pool byte option. At the maximum 128-entry setting,
both banks' target and draft tails total at most 748,240,896 bytes.

## Qualification

The live retention campaign seeds 24 distinct completed turns, repeats every
prompt with a complete cache hit, and resumes all 24 turns. Followups are
cancelled after first content so they do not add completed turns. Inserting
turn 25 then evicts the oldest while the second-oldest retains exactly its
previous frontier. Target-only runs at C16 and dSpark at C2 both pass, proving
that retention is independent of active concurrency.

A separate launch with `--prefix-cache-entries 2` passes the same two-turn
retention/eviction contract. The updated full default pool boots and serves a
short arithmetic request in both target-only and dSpark modes at C16, confirming
the logged 24-turn/24-prompt limits and source-page allocation. This establishes
allocation and short serving behavior; the 1M needle remains a separate gate.

For every resumed turn, cached tokens include all assistant content. Target
decoding leaves EOS as an unevaluated final anchor. dSpark may commit EOS when
it matched a draft input; a correction or bonus EOS remains unevaluated. The
collector accepts precisely these verifier-defined frontiers and rejects a
prompt-only hit. Its original target-only frontier assumption failed on dSpark;
the original responses, logs and assertion are preserved, and validation of
those same requests passes with the corrected contract.

The pure retention test covers independent banks, prompt churn, LRU eviction,
pressure priority and disabled reuse. Sizing and CLI tests cover the new default
and configurable limit. These focused checks do not replace the final retained
long-context matrix, 1M needle or full release qualification.

Complete-prompt, agentic and divergent partial-prefix API checks match the
preceding build's output and token counts in both modes. Target C2/C6/C16 and
dSpark C2 also pass counting, cancellation and replacement checks.

Four alternating warm A/B pairs per workload show median decode differences
of +0.40% counting / +0.70% code for target-only and -0.36% / -0.14% for dSpark.
All paired outputs and token counts match. These small differences provide no
evidence of a material regression or improvement; they are short C1 diagnostics,
not release throughput claims. Both arms share the current driver and power
configuration, and all inference runs serially across the four Spark workers.
Candidate and reference use the same context/concurrency but different pool
budgets: the new context-scaled default versus the prior explicit 1 GiB pool.

The RTX cards report driver 595.91.07 and enforced 400 W limits. The user reports
standard memory speed with no overclock; loaded telemetry shows 13,365 MHz,
with a driver-reported maximum of 14,001 MHz. Historical results before the
driver/power-cap reset are not used as the control for this comparison.

[Evidence metadata](release-v1-retained-turns.json) records source and artifact
hashes, commands, complete responses, logs and hardware telemetry.
