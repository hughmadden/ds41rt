# Lane-local request retirement

Each independent lane now retires its completed or cancelled requests after its
own work finishes, without asking its peer to drain or restarting the peer's round
sequence. Admission and prefill still drain both lanes. Surviving requests stay
in their existing lanes until the next admission boundary.

Retirement always attempts both target and draft cleanup. Released SWA/source
leases are revoked in host ownership without uploading redundant zero lengths;
begin/reset installs clean device metadata before a replacement lease can be
used. Slot guards still reject release while that slot has pending writes.
This adds no GPU workspace and keeps the existing KV capacity.

Snapshot retention still allocates and copies synchronously. Removing the explicit
retirement join does not yet make retention cooperative; that remains follow-up
work, along with remaining blocking logits transfers.

## Validation

The live counting trace shows request 2 retiring in lane 1 while lane 0's round 10
is issued but uncommitted. Lane 0 next issues round 11, without restarting its
cohort. Both counting outputs are exact and the short request finishes first.
The reusable check is `scripts/qualify-ds41-lane-retirement.py` and requires a
fresh server with lane schedule debug logging.

The all-cache GPU fixture passes retained-prefix, partial replay, zero acceptance,
slot reuse, late failure revocation and recovery checks across all 44 cache owners.
Focused SWA/source tests confirm release rejects old leases, leaves device lengths
untouched, and resets them before replacement use; disjoint source reservations
still publish out of order and roll back correctly. Cargo check, test compilation
and release build pass.

## Serving observations

One RTX PRO 6000 Blackwell at **400 W and standard memory speed**, four unchanged
Sparks, five RTX resident expert layers, 18 × 1,048,576-token KV capacity and 24
retained turn snapshots. Control is `22d472d`; both arms use adaptive independent
lanes and the same native library. Candidate ran first; no builds overlap inference.

| Workload | Joint retirement | Lane-local retirement |
|---|---:|---:|
| C1 code median, tok/s (three samples) | 130.50 | 130.48 |
| C2 mixed, tok/s | 91.90 | 92.88 |
| C4 mixed, tok/s | 114.26 | 113.96 |
| C8 mixed, tok/s | 155.38 | 154.51 |
| C16 mixed, tok/s | 153.85 | 168.55 |
| Startup to API ready, seconds | 5.307 | 5.301 |
| Peak observed GPU memory, MiB | 97,142 | 97,102 |

Mixed batches use ordered first-content admission and include admission gaps.
These are single batch observations, not a demonstrated uniform throughput gain.
C1 is unchanged; C2–C8 remain within about 1%, with a 9.6% improvement in this C16
sample. Both arms pass mixed code-structure checks, needle retrieval, prompt/turn
reuse, eight cancellations with eight survivors, recovery and four concurrent
high-thinking constraint checks. The prefill matrix and full release qualification
were not rerun.

The [evidence record](phase1-lane-retirement.json) preserves outputs, timing,
artifact identities, GPU test logs and the trace proof. Standard serving was
restored and stopped experiment containers removed after saving their logs.
