# Queued SWA commit

Independent decode lanes queue their accepted SWA writes before waiting for GPU
completion. Each layer keeps its existing producer stream, private KV output and
pinned metadata. The scheduler polls all queued windows together with its own
dSpark commit and yields while they run. This replaces the sequence of up to forty
launch-and-wait pairs. It adds no device workspace and preserves existing KV math.

A pending write owns a per-request slot reservation, its original proposal,
accepted counts, destination metadata and new frontiers. Host identity/frontier
queries retain the committed state; device views, new producers, snapshots,
release and slot reuse reject the reserved slots. Disjoint slots remain available
to the peer lane. Publication validates the owner, generations, versions and
accepted counts, then advances the frontier only after completion. Errors drain
all queued windows before revoking requests. Request release checks reservations
before removing history ownership. Reset rejects a pending commit; destruction
drains before releasing its reservations and storage.

Direct C1 and prefill calls retain synchronous completion. The same descriptor and
scatter logic is shared with the queued path. Compressed-source publication still
uses its original synchronous commits and shared pinned upload storage. Retirement,
snapshots and constrained-logit host work also remain in the v2 dependency audit.
This milestone does not claim that every cross-lane dependency has been removed.

## Validation

- A CPU ownership test checks that releasing one write reservation preserves the
  peer's reservation.
- A GPU test compares queued and direct SWA values/scales exactly, with different
  accepted lengths on two independent streams. It checks unpublished views,
  rejected release and changed acceptance, separate frontier publication, and
  aborting one transaction while keeping the peer usable.
- The existing all-cache GPU fixture now queues two of its acceptance cycles
  across all forty windows. It checks rejection of premature bank release,
  partial/zero acceptance and cache bytes. Its existing direct wraparound,
  retained-prefix/encoder replay, late failure, all-component revocation and
  recovery checks also pass.
- Cargo check, release build and test compilation pass. GPU tests ran on GPU zero
  with the serving native library and official weights. The larger cache fixture
  ran while the service was stopped; no build overlapped serving measurements.

## Serving observations

One RTX PRO 6000 Blackwell at **400 W and standard memory speed**, four unchanged
Sparks, five RTX expert layers, 18 × 1,048,576-token KV capacity, and 24 retained
snapshots. Both arms use adaptive independent lanes and the same native library.
The control is `45883d1`; candidate first, then control, each in a fresh process.

| Workload | Queued dSpark only | Queued dSpark + SWA |
|---|---:|---:|
| C1 code median, tok/s (three samples) | 129.61 | 130.48 |
| C2 mixed, tok/s | 92.31 | 93.46 |
| C4 mixed, tok/s | 119.07 | 115.54 |
| C8 mixed, tok/s | 163.08 | 167.13 |
| C16 mixed, tok/s | 177.62 | 202.19 |
| Startup to API ready, seconds | 5.322 | 5.307 |
| Peak observed GPU memory, MiB | 97,004 | 97,028 |

All three C1 outputs match exactly. Mixed traffic is one cold batch per concurrency
and includes admission gaps. C16 improves 13.8% in this pair, C4 declines 3.0%, and
the other rates differ by about 1–2.5%; this does not establish a uniform gain.
Concurrent outputs can differ with changed batching; named code structure checks
pass, but prose quality is not assessed. The later scoped v2 table runs remain the
release performance evidence.

Both versions pass needle retrieval, exact prompt/turn reuse, eight cancellations
with eight survivors, recovery, and four concurrent high-thinking constraint
checks covering tools, response schemas, JSON and SSE. No prefill matrix or full
release suite was rerun. The standard service was restored, and stopped experiment
containers were removed after saving their logs. Samples, artifacts, commands and
GPU test output are in the [evidence record](phase1-queued-window-commit.json).
