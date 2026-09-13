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
| Cache production and index selection | `BackboneExecution::prepare_layer_with_cache` produces window/source data and calls index selection synchronously, under a short shared cache borrow | Separate enqueue/completion/publication with explicit cache lifetimes before introducing an await |
| Attention through FFN input preparation | `SparseAttentionWave::execute_query_then` queues sparse attention and its projection/mHC continuation, then synchronizes the stream | Cooperative completion must retain query, cache, selection and consumer owners without a shared bank borrow across suspension |
| FFN completion and layer advance | `complete_layer` finishes mHC; `BackboneBlockWave::advance` copies residual/pre values and synchronizes before taps/Engram/query consume them | Preserve those producer dependencies when moving completion to a cooperative wait |
| Weight rebind | Query/projection/shared/router owners synchronize their own streams before rebinding | Establish already-complete ownership before removing redundant waits; a source search alone cannot establish their measured cost |
| Cold graph setup and eviction | LayerGraphs retains small shapes per layer; insertion/eviction requires completed launches, and capture remains synchronous | Never suspend between begin/end capture; containing owners must drain before replacing or destroying captured storage |

The combined attention graph archived in
`phase1-attention-graph-candidate.patch` still applies cleanly to the current
worktree. It is a later performance experiment, not evidence that these host
waits have disappeared. Keep its original rejected results intact and record the
new independent-lane comparison separately.

## Current candidate evidence

Raw evidence: `/home/tj/.cache/ds41rt-experiments/chained-query`.
The first pair reports C1 code 129.83 → 129.84 tok/s, C8 152.14 → 159.20,
and C16 181.77 → 162.87. The two follow-up C16 pairs also decline: 167.66 → 158.38 and
165.10 → 159.84 tok/s. See [the accepted change and both sets of evidence](phase1-chained-query.md). Exact embedding/image and 56 real-weight query cases,
cache reuse, cancellation/recovery and high-thinking constrained checks pass.
