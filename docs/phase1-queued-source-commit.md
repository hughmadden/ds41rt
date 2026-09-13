# Queued compressed-cache commit

Independent decode now queues compressed-source writes together with SWA and
dSpark writes, then polls their completion while allowing the peer lane to run.
Each compressor producer owns its metadata upload staging, accepted proposal,
page reservation, and pending transaction. Direct C1/prefill commits share the
same scatter/carry logic and retain synchronous completion.

Page plans now claim free physical pages immediately after validating every
participant. Disjoint plans can coexist and publish in either order. Cancellation
returns only that plan's claimed pages, and failed validation leaves the free pool
unchanged. Slot reservations prevent overlapping append, snapshot, or release;
request/history release checks these guards before removing ownership. Existing
source references protect shared tails until their queued copies finish.

After GPU completion, publication transfers the page claims into request page
tables and advances source frontiers and versions. Partial enqueue failures drain
before dropping the plan and revoking its requests. Wave reset rejects a pending
commit, and destruction drains before returning pages or staging. All four-plane
index/KV writes and the ratio-two carried-token copies use the existing math.

The upload buffer adds `capacity * 4 + 256` bytes of pinned host memory per
compressor wave. This change adds no GPU workspace and does not reduce the KV pool.
The old shared source upload helper remains for existing fixtures; serving commits
use producer-owned staging.

## Validation

CPU ownership checks pass. Nine focused compressor CUDA tests pass, including:

- Two queued producers sharing a state, for ratios one and two, compared exactly
  against direct index/KV bytes and carried-token state. They publish in reverse
  order and abort one request while preserving the peer.
- Disjoint page plans, pool exhaustion without mutation, out-of-order publication,
  rejected conflicting access, and rollback restoring the pool without duplicates.
- Prefix truncation, copy-on-write, shared writers, nonwriters, eviction and odd
  carry preservation, including the highest physical-page addressing fixture.

The all-cache GPU transaction fixture also passes with both SWA and compressed
sources queued in its partial/zero acceptance cycles. Its direct wraparound,
prefix/encoder replay, late source-failure revocation, all-component cleanup and
reclaimed-page recovery checks remain passing. Cargo check, test compilation and
release build pass. GPU tests use GPU zero and the serving native library; the
larger all-cache fixture runs with standard serving stopped.

Request retirement still asks both lanes to drain, and snapshot copies and some
host work remain blocking. This change completes queued cache publication, not
the entire v2 synchronization objective.

## Serving observations

One RTX PRO 6000 Blackwell at **400 W and standard memory speed**, four unchanged
Sparks, five RTX resident expert layers, 18 × 1,048,576-token KV capacity, and 24
retained snapshots. Both versions use adaptive independent lanes and the same
native library. Control: `5447319`. No builds overlap timed inference.

| Workload | Queued SWA/dSpark | + Queued compressed sources |
|---|---:|---:|
| C1 code median, tok/s (three samples) | 129.30 | 129.90 |
| C2 mixed, tok/s | 92.82 | 92.38 |
| C4 mixed, initial unordered batch, tok/s | 122.37 | 115.51 |
| C8 mixed, tok/s | 163.83 | 168.47 |
| C16 mixed, tok/s | 199.64 | 198.86 |
| C4 ordered admission median, tok/s (three batches) | 130.32 | 130.20 |
| Initial startup to API ready, seconds | 5.310 | 5.327 |
| Initial peak observed GPU memory, MiB | 96,998 | 97,092 |

All three C1 outputs match exactly. Both arms pass needle retrieval, prompt/turn
reuse, eight cancellations with eight survivors, recovery, and four concurrent
high-thinking constraint checks. Mixed samples assess named code structure, not
general prose quality. Peak process memory includes runtime/graph allocations
and differs from explicit device workspace accounting. The prefill matrix and
full release qualification were not rerun.

### Admission-order confound

The initial C4 decline prompted two additional unordered pairs, 116.96 → 115.95
and 147.77 → 135.64 tok/s. First-content order differed between versions. The
inferred initial lane groupings put both code requests together in the faster
controls and paired code with prose in the corresponding candidates. These are
inferences from admission timing, not instrumented lane assignments.

Simultaneous client threads do not guarantee server arrival order, so fixed input
content alone does not control expert-sharing groups. The probe now offers
`--ordered-admission`: each request starts after its predecessor emits its first
content, while existing requests continue decoding. The record labels the
admission mode and retains per-request start/first-content timestamps. Every
adjacent pair in all six ordered batches satisfies that ordering check.

The ordered C4 pairs are 118.41 → 116.50, 133.66 → 133.85, and 130.32 → 130.20
tok/s. They do not reproduce the larger unordered declines. Candidate ran first
in this ordered comparison; these observations support retaining the ownership
change without claiming a throughput gain. Admission gaps remain included in the
measurement. Use ordered admission for subsequent mixed-workload comparisons;
retain the unordered mode for arrival-race/lifecycle coverage.

All original and follow-up results remain in the [evidence record](phase1-queued-source-commit.json).
Standard serving was restored after every run. Experiment containers were removed
after their logs were saved. No published release container changed.
