# Remaining lane waits

Initial source audit at `d92caad` with the chained embedding/query candidate.
That change passed functional checks, was initially rejected for three C16 losses,
then accepted after the broader C2–C16 sweep contradicted a general C16 regression.
The query change is now integrated. This remains a source audit, not a latency
measurement or a claim that the complete loop is asynchronous.

## Scheduler and ownership

`v41_native_serve/scheduler/independent.rs` runs two lane futures in a single
`tokio::join!`. That join encloses the serving cohort, not each decode round.
Each lane proposes, selects verification lengths, verifies, commits and retires
its own requests. Its normal round does not wait for the peer's round counter.
The common drain is requested for admission/prefill or errors; error handling
allows the peer to finish owned GPU/RDMA work before shared cleanup.

Draft replay, main-context projection, cache publication, retained snapshot
copies and logits transfers have lane-owned in-flight storage. Shared host
owners are borrowed for synchronous submission/publication, with borrows released
before cooperative waits. Device streams are created with
`cudaStreamNonBlocking` in `native/src/ds41rt_native.cc`.

## Remaining host-thread stalls

A wait on one lane's stream is still a stall for the other lane's host future
when both run on this thread. Absence of a round barrier does not prove the loop
is fully asynchronous.

| Path | Current behavior | Next consideration |
| --- | --- | --- |
| Token upload, embedding and mHC/query | Candidate writes embeddings directly into block inputs and waits cooperatively after the query chain; subsequent queries also wait cooperatively | Accepted after broader sweep; finish adjacent blocking work |
| Cache production and index selection | [Production and index now poll together](phase1-queued-index.md), with projection overlapping production and guarded selection completion | Combined checkpoint versus e9c07ae: C1 −1.3%, median C2–C14 −0.34%, C16 −12.5%; costs remain unresolved |
| Attention through FFN input preparation | [Queued completion](phase1-queued-attention.md) retains lane consumers and drains before cancellation/reuse, with no shared bank borrow across the wait | Implemented; [cold sparse/projection warmup now yields](phase1-queued-sparse.md) |
| Router, shared FFN and local experts | [Cooperative router/shared/local completion](phase1-queued-ffn.md) retains inputs and drains cancellation; route exports keep their existing owned staging | Implemented, including cold router/shared warmup; remaining final mHC and advance follow below |
| FFN completion and layer advance | [Queued layer completion](phase1-queued-layer.md) chains final mHC and next-input copies, waits once, and marks copies ready for advance | Implemented with exact output/copy and cancellation/reuse checks; high-concurrency cost remains unresolved |
| Engram gate at layers 1 and 14 | [Owned gather leases leave the shared request borrow](phase1-queued-layer.md); pinned H2D/dequantization and gate/residual completion now wait cooperatively | Implemented with direct output parity, cold/warm and cancellation/reuse checks |
| dSpark taps at layers 37–39 | `TargetTapWave::capture_cooperative` retains prepared input and waits cooperatively before the next query | Implemented; batch/order check and full serving checks pass |
| Spark dispatch and result reduction | [Completed-owner dispatch check and cooperative TP reduction](phase1-queued-tp.md) retain received frames and shared input | GPU parity, pending cancellation and live receive-lifetime checks pass; dynamic capture has no remaining blocking CUDA calls |
| Weight rebind | Query/projection/shared/router/mHC owners require already-complete streams through nonblocking queries | Fixture cancellation/reuse checks pass; aggregate dynamic wait counts are recorded separately |
| Cold graph setup and eviction | LayerGraphs retains small shapes per layer; insertion/eviction requires completed launches, and capture remains synchronous | Never suspend between begin/end capture; containing owners must drain before replacing or destroying captured storage |

The [combined attention graph revisit](phase1-attention-graph-revisit.md) is complete.
The adapted candidate passes correctness and cancellation checks, but three pairs
show only a small C1 benefit and uncertain concurrent impact. Keep it out of the
release; both the original and adapted patches and measurements remain archived.

## Historical chained-query evidence

Raw evidence: `/home/tj/.cache/ds41rt-experiments/chained-query`.
The first pair reports C1 code 129.83 → 129.84 tok/s, C8 152.14 → 159.20,
and C16 181.77 → 162.87. The two follow-up C16 pairs also decline: 167.66 → 158.38 and
165.10 → 159.84 tok/s. See [the accepted change and both sets of evidence](phase1-chained-query.md). Exact embedding/image and 56 real-weight query cases,
cache reuse, cancellation/recovery and high-thinking constrained checks pass.

## Dynamic audit follow-up

[The CUDA trace](phase1-queued-sparse.md) finds 37,046 → 3,404 stream-synchronize
calls and 981 → 101 ms API time in matched four-second C16 windows. Cold sparse,
projection and head warmup now yield; completed-owner rebind/eviction checks no
longer synchronize. This is measured host API time, not a throughput gain.

[Draft setup and cold-chain warmup now poll](phase1-queued-draft.md), with owned
pinned seed/sampling/descriptor staging and retained cache reservations. Synchronous
memcpy calls disappear from the four-second trace. Remaining stream-synchronize
API time is 82 ms across 3,448 calls. A separate stack capture confirms active-lane
waits in TP dispatch and post-receive TP reduction; teardown stacks are excluded.

[TP dispatch and reduction now complete without blocking the owner thread](phase1-queued-tp.md).
The next capture contains zero stream/device synchronizations, synchronous memcpy,
CUDA allocations or CUDA frees throughout 12 seconds of mixed decode. The first
six seconds retain all 16 requests; later time includes changing shapes and request
completion. This is evidence for the exercised serving path, not for shutdown,
error cleanup or excluded admission/prefill. Graph instantiation still consumes
host CPU/driver time and is a performance concern, not a remaining GPU wait.

[RoCE decode receive polling now yields on every unsuccessful poll](phase1-async-poll-yield.md). It waits for its own four results, without a peer-lane
round barrier. Those required within-lane data dependencies remain.

## Remaining completion order

1. The fresh accumulated pair against e9c07ae is complete: C1 129.77 → 129.48,
   median C2–C14 +0.39%, C6 −7.1%, C16 −9.8%. Retain every earlier curve and
   investigate the repeated C6 loss and high-concurrency uncertainty. CUDA wait
   removal alone is not release performance qualification.
2. The combined-attention revisit is complete and omitted. Run matched fixed-output
   counting against e9c07ae to assess accumulated asynchronous changes without
   varying prose output lengths.
3. Run the scoped README/report qualification excluding the prefill matrix, plus
   high-concurrency thinking/high tool calling; prepare and verify the v2 release
   and matching container with the requested headline and release-note updates.

The [final graph recheck after removing decode receive spinning](phase1-attention-graph-yield.md)
supports integration. Earlier omission notes describe the 50 µs scheduler; the
selected release candidate includes the graph and zero-spin cooperative receives.
