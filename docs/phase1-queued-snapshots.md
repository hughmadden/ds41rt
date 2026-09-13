# Cooperative retirement snapshots

Independent retirement now queues backbone and draft snapshot copies, releases
the shared host borrows, and polls completion while allowing the peer lane to run.
Each lane has separate copy streams. Copies use the existing bounded arenas and
preserve the same packed SWA, compressor carry and draft-ring bytes.

Backbone pending state belongs to its cache owner. A pending snapshot rejects
planning, mutation, duplicate retention and release of its source request, while
identity queries remain valid for guarded cleanup. Draft snapshots hold read
reservations on their source slots and retain their request identity until all
three stages finish. Other request slots remain usable.

The retained entry is published only after target and draft completion. Failed
enqueue/poll paths drain both owners before releasing pending storage or the
request. Owner destruction drains copy streams before dropping their source and
destination storage. Two pending entries can publish in either order; insertion
rechecks the bank limit after another lane's intervening insertion. The existing
two spare arena slots bound that overlap.

C1 and prompt retention retain synchronous wrappers around the same copy logic.
Admission/prefill still drains both lanes. This change adds eight copy streams and
host ownership state, with no new device workspace or KV-pool reduction. Snapshot
copies no longer impose a host completion wait during independent retirement.

## Validation

The all-cache GPU fixture queues two disjoint snapshots, publishes in reverse
order, aborts a replacement copy and commits a peer request while the first
snapshot remains reserved. Conflicting release, planning, duplicate queue and
wrong-owner polling are rejected. Exact restored state, odd compressor carry,
wrapped windows, partial encoder replay, immutable retained futures and all-owner
failure recovery pass.

The draft GPU fixture compares queued and direct snapshot bytes at short,
wrapped and million-token frontiers. It exercises reverse publication, rejected
source writes/releases, peer survival and abort cleanup. The existing draft
slot-reuse fixture also passes. Test compilation and release build pass.

The live trace verifies queued snapshot publication before the short request
retires. Its peer has an uncommitted round 10 and subsequently issues round 11,
without restarting. Both counting outputs are exact. No peer round event occurred
inside this very short snapshot interval, so the trace is an ownership/ordering
check rather than a measurement of useful compute overlap.

## Serving observations

One RTX PRO 6000 Blackwell at **400 W and standard memory speed**, four unchanged
Sparks, five RTX resident expert layers, adaptive independent dSpark, 24 retained
turns, and a **16.681 GB global KV pool for 18,710,016 tokens plus private tails**.
Both arms use the same default pool and native library. Control is `8ae0e15`;
candidate ran first, with no builds during inference.

| Workload | Synchronous snapshots | Queued snapshots |
|---|---:|---:|
| C1 code median, tok/s (three samples) | 129.95 | 130.01 |
| C2 mixed, tok/s | 92.15 | 93.33 |
| C4 mixed, tok/s | 114.37 | 115.04 |
| C8 mixed, tok/s | 153.54 | 157.04 |
| C16 mixed, tok/s | 173.36 | 169.94 |
| Startup to API ready, seconds | 5.310 | 5.299 |
| Peak observed GPU memory, MiB | 96,884 | 96,904 |

All three C1 outputs match exactly. Mixed batches use ordered first-content
admission and include admission gaps; these single batches do not demonstrate a
uniform throughput gain. C1 is unchanged, C2–C8 slightly higher and C16 2.0% lower.
Both arms pass needle retrieval, prompt/turn reuse, eight cancellations with eight
survivors, recovery and four concurrent high-thinking constraint checks. The
prefill matrix and full release qualification were not rerun.

The [evidence record](phase1-queued-snapshots.json) preserves artifact identities,
outputs, timing, GPU test logs and trace checks. Standard serving was restored and
stopped experiment containers removed after saving their logs.

Remaining audit targets include constrained/full-frontier logits downloads,
token upload and embedding handoff, and graph capture/rebind or other component
waits on the shared host thread. This milestone does not claim the entire serving
loop is asynchronous.
