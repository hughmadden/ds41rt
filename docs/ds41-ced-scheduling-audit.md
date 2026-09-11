# Scheduling after bounded CED deployment

Source audit at `7063f4b`, using the second request of the deployed CED trace.
This records implementation constraints and measured priorities; it does not
claim that overlapping prefill is implemented or that a scheduling speedup has
been measured.

## Measured opportunity

The [CED profile](ds41-ced-serving.json) records 5,454,785 us in encoder steps plus
decoder replay, of which 2,547,638 us is expert response waiting/handling. Simply
subtracting that entire interval leaves 2,907,147 us, or 5,644.7 prompt tok/s for
16,410 tokens. This is an optimistic accounting scenario, not a hardware bound:
these are nested host timers, future overlap can change resource contention, and
response processing still costs work. Scheduling alone cannot be assumed to
reach the 8k goal. Sparse attention accounts for 1,618,788 us; it remains a useful
kernel target alongside scheduler integration.

## Where serialization actually lives

- `v41_native_serve.rs::prefill` completes and commits each encoder chunk before
  preparing the next. The serving loop also admits one request at slot zero and
  runs it to completion before taking another queued request.
- `v41_target_pass.rs::execute_phase` executes all twenty encoder layers per
  chunk. One pass owns mutable block/query/projection/shared/router/index/cache
  producer workspaces. Sharing immutable weights between independent passes is
  already supported, but the native API constructs only one pass.
- `v41_experts/coordinator.rs::dispatch_ffn` already returns pending RoCE work;
  `NativePendingFfn::finish` collects and reduces it. `LaneFfn::execute_tp4` does
  shared FFN between dispatch and collection. There is no need to invent a TCP
  queue or replace the transport to obtain this split.
- `ds41rt-transport/src/v41_expert/roce.rs::receive_owned` progresses all four QPs
  and yields after a bounded 250-us polling quantum. Other runnable futures can
  progress on the same CUDA-owning thread. A future scheduler should measure
  polling fairness with two active waves; the current serial trace cannot do so.
- The decisive same-prompt restriction is `BackboneCache::plan_stage`: a batch
  starts at the one request-wide committed position. `committed_end` checks all
  phase window owners and all source owners against that position.
  `validate_batch` also requires the same request version and position.
  `commit` publishes all twenty encoder windows and four sources together after
  the complete chunk, then increments that one version. A second chunk cannot
  currently start at the next token position while the first is partway through
  its layers. Removing validation would admit stale or nonexistent KV.

## Required dependency and ownership change

For encoder task `(chunk, layer)`, attention may start only after the previous
layer's FFN for this chunk has completed and the previous chunk's required KV at
this layer has been published. The previous chunk's FFN at the same layer does
not need to finish before this attention starts. That is the useful overlap.

The old GLM `execute_scheduler_bounded_long_prefill_wavefront` distinguishes
started and finished tasks and allows up to sixteen resident chunks. Its
previous-chunk `started` condition assumes attention/cache production has already
completed at dispatch. DS41 needs to express that publication explicitly:

1. Track per-layer window and per-source encoder production frontiers separately
   from whole-chunk completion. Keep admission identity and generation checks;
   whole-request history must not pretend that later layers have completed.
2. Publish prompt-only KV after its current attention consumers finish, before
   remote FFN completion. Preserve ratio-two compressor pending groups across
   chunk boundaries. Publication is irreversible for that admission; cancellation
   or a later failure must revoke the request rather than reuse a partial history.
3. Retain each resident chunk's residual/pre-state and learned selections across
   interleaving. Selection bindings include source proposal identity; merely
   retaining numeric top-k IDs is insufficient with the current validation.
   Appending future global KV must preserve the causal snapshot seen by an older
   chunk. Window-ring overwrites require completed readers, not just issued work.
4. Keep Engram token-history preparation ordered independently of layer execution.
   Its current transaction advances with whole-chunk cache commit. Input tokens
   are already known, but preparing several chunks from the same unadvanced
   history would hash incorrect positions/context.
5. Bound resident chunk storage and pending transport owners; immutable model
   weights can remain shared. Do not equate sixteen request slots with sixteen
   full-capacity copies of all execution workspaces.
6. Publish final encoder completion only when all encoder chunks and source 20
   are complete. Then run the existing bounded decoder replay/dSpark seed.

Qualification must cover odd chunk sizes (compressor carry), window wrap, source
and index reuse, chunk/slot reuse, cancellation during pending remote work, and
subsequent admission recovery. Compare sequential and scheduled CED outputs with
identical math and prompts before measuring scheduling gain. C16 request mixing
is a separate live gate even if one prompt's chunks overlap successfully.

## Correction to the recent optimization checklist

| Suggested item | Current source evidence |
|---|---|
| Keep `wo_a` FP8 / grouped GEMM | Already implemented in `v41_attention_output.rs`; `grouped.launch_rope` consumes original FP8 weights/scales and fuses inverse rotary preparation. |
| BF16 LM-head weights | Already implemented by `v41_tensors.rs::VocabularyHead`, shared with dSpark. FP32 output logits do not imply FP32 weight residency. |
| Fused router/top-6 | Native router deployed and measured; see the router rollout record. |
| mHC/norm/residual fusion | Partial integration: `v41_block.rs::begin_ffn` enqueues attention finish, state transfer and FFN begin on one stream with one drain. This is not proof that all tiny kernels are fused. |
| Avoid local-index materialization/KV concatenation | Direct bounded-window/global-cache attention is deployed. Explicit adjacent-query KV tile sharing remains untested. |
| Host Engram gather/dequant/projection | Background mapped gathers and native FP8 projection plus residual gate exist; gather, unpack/dequant and projection are not one fused operation. Warm between-layer Engram interval is only 42,141 us in this trace. |
| Sparse indexer/top-k fusion | Native score and top-k update kernels exist as separate calls in `v41_index_selection.rs`; broad fusion remains a candidate. Layer index intervals total 50,133 us here. |
| Compressor prefill fusion/precision | Native projection, score projection, pooling/norm and index packing exist; they remain separate launches in `v41_compressor.rs`. Producer intervals total 24,802 us here, excluding boundary source work not logged as a normal layer. |

The small index/Engram/producer intervals in this warm trace favor scheduler and
sparse-attention work first. They do not settle cold page-fault behavior, decode
launch overhead, or accuracy tradeoffs of changing compressor precision.

## Implemented prerequisite: committed encoder source views

`CompressorState::committed_proposal` now accepts all four source layers, using
ratio two for layers 2/8/14 and ratio one for layer 20. Metadata exposes only
`floor((query_position + 1) / ratio)` causal rows, zero private overlay rows, and
`floor(committed_tokens / ratio)` physical cache rows. Thus an incomplete final
ratio-two group is never exposed as a complete latent. A caller-retained snapshot
identity can be reused for older query ranges after later appends, while the
borrow contract still requires existing consumers to drain before mutation.

The real-weight cache transaction test covers all sixteen requests and all four
sources after 64-token and 65-token appends, including per-query metadata, odd
carry, stable binding identities, range rejection and zero-snapshot rejection.
Existing full-prefix, decoder replay and late-failure recovery checks also run.
Raw build and GPU logs are `/tmp/ds41-ced-bounds/encoder-source-{build,gpu}.log`.
This removes the decoder-only restriction from the committed-view primitive;
per-layer publication, retained index consumers and native scheduling remain to
be integrated. The running APIs still use the previous frozen CED artifacts.

## Implemented prerequisite: per-owner encoder publication

The cache bank now supports `publish_encoder_window` and
`publish_encoder_source`. Each commits the full prompt proposal for that owner
and records the batch identity, physical end and published-owner mask separately
from logical request completion. Final chunk commit skips already-published
owners and advances request completion only once. Mixed early/deferred window
commits are supported; publishing any owner requires full acceptance afterward.
Publication failure revokes every participating admission, including owners
already written. No new GPU allocation is introduced.

While publication is pending, validation checks physical histories against the
recorded per-owner ends, rejects competing batch identities and duplicate writes,
and preserves the prior logical request position. The ordinary planner still
rejects a second chunk of the same request until completion: multi-chunk
reservations/frontiers and Engram ordering remain the next scheduler work. This
primitive does not by itself implement overlapping chunks.

Attention views can consume a published source without its original private
producer. Encoder batches now reserve a committed-source snapshot as decoder
replay batches already did; later layers sharing that source retain the same
binding and ratio-correct causal metadata. Existing window readers must drain
before publication overwrites their ring; the caller contract remains explicit.

The expanded real-weight GPU transaction test checks sixteen requests through
64+65 encoder rows and decoder replay, mixed/all early window publication, all
four sources, layer-3/layer-5 committed-source reuse, stale/duplicate/partial
acceptance rejection, and both ordinary and early-publication late source
exhaustion with complete admission recovery. Build/GPU logs are
`/tmp/ds41-ced-bounds/publication-{build,gpu}.log`. Native serving has not been
switched to early publication yet; no new API throughput result is claimed.

## Implemented prerequisite: Engram preparation lookahead

`EngramHistory::prefill_cursor` creates a private preparation frontier containing
only the request identity, generation, position and three preceding compressed
IDs. `EngramPrefillCursor::advance_full` advances that frontier across known prompt
batches without accepting anything into the actual request history. Its borrowed
history works with the existing token-map and pipeline preparation interfaces.
A scheduler must retain each in-flight batch's starting cursor for I/O validation
while advancing its main preparation cursor, then commit completed batches to the
actual request history in order. Cursor storage is constant-size; bounded batch
and gather ownership remain scheduler responsibilities.

Engram batches now record their starting position and preceding context as well
as owner/generation. This matters because an alternative predecessor could have
the same length and generation but different lookback tokens. Such a successor,
or one whose predecessor was only partly accepted, must not enter history.
Ordinary speculative accepted-prefix behavior remains supported; dependent
lookahead batches must be discarded after divergence.

Four core Engram tests pass. New coverage compares all hashes against a continuous
prompt for chunk widths 1/3/64/65/128/2048 with image barriers, verifies that
preparation does not advance accepted history, then commits in order and compares
the next token. It also rejects out-of-order, foreign, stale, partial-predecessor
and equal-length divergent-predecessor batches. Results are in
`/tmp/ds41-ced-bounds/engram-lookahead-tests.log`. This is preparation state support;
serving does not yet schedule several chunks concurrently.

## Implemented prerequisite: bounded encoder reservations

`BackboneCache::reserve_encoder` now reserves contiguous prompt chunks before
execution, capped at sixteen outstanding chunks per request. Planning validates
all participants and the prompt extent before changing any reservation queue.
Reservations retain admission identity and their exact token range; completing an
earlier chunk no longer invalidates a later reserved chunk merely because the
request completion version advanced.

Physical KV validation follows the latest published reservation for each owner,
while logical completion stays at the oldest unfinished chunk. Source/window
producer validation still requires the expected physical starting position.
Reserved chunks must publish all twenty windows and four sources before final
completion; final completion is ordered and requires full acceptance. Request
release revokes all queued chunks. The ordinary planner remains exclusive with
reservations, preserving existing decode/speculative transaction behavior.

The real-weight GPU test now reserves 64+65 rows for all sixteen requests before
executing either chunk, publishes both chunks layer by layer, then finishes them
in order. It verifies that logical completion remains zero throughout publication,
advances to 64 then 129, and allows decoder replay only after completion. It also
checks out-of-order/incomplete completion rejection, successor validity after
predecessor completion, stale-batch release rejection, and the sixteen-chunk cap.
The full cache test passes in 4.24 s, including existing ordinary and early-write
failure/recovery cases. Logs are `/tmp/ds41-ced-bounds/reservation-{build,gpu}.log`.

This supersedes the single-pending-chunk restriction described in the earlier
publication section. Integration with request-owned Engram cursors, retained
chunk execution state, task scheduling and the API remains unfinished. The
running CED artifacts and measured throughput are unchanged.

## Implemented integration: request reservations and mapped Engram I/O

`Requests::reserve_encoder` now combines cache reservations with the private
Engram preparation frontier. It verifies the cache and Engram preparation
positions, prepares hashes/prefetch/gather, and advances the live preparation
cursor only after all cache reservations succeed. Capacity/extent rejection
cancels the newly prepared I/O and leaves both frontiers unchanged. Ordinary
preparation cannot be mixed into this reserved encoder sequence.

Each reserved `RequestBatch` retains its starting cursor. Engram upload polling
uses these retained histories rather than the still-unadvanced accepted request
history. Acceptance validation and commit continue using the actual request
history, so later chunks cannot commit early. Decoder replay clears the encoder
preparation cursor after the phase transition succeeds; release discards it with
the admission.

The real request integration GPU test passes in 5.82 s: sixteen admissions with
two queued five-token chunks each, exact hash comparison against a continuous
prompt including image barriers, mapped gather/upload for both Engram layers,
rejected over-extent preparation without frontier movement, out-of-order
acceptance rejection, and release invalidation of both queued batches. Existing
ordinary cancellation, incomplete-pass failure and admission recovery also pass.
Raw logs: `/tmp/ds41-ced-bounds/request-reservation-{build,gpu}.log`.

Cache and mapped-I/O preparation are now connected at the request layer. Retained
execution lanes/index state and scheduling these batches through the backbone
remain outstanding; neither API has been switched to this path yet.

## Implemented integration: split backbone layer execution

`BackboneExecution::prepare_layer` performs cache production, index selection and
attention, returning `PreparedLayer` with a borrowed FFN lane but no cache-bank
or index borrow. Its async `execute` dispatches routed experts, computes the
shared contribution and collects TP4 results. `complete_layer` validates the
batch/phase/layer identity before final mHC and progress publication. Dropping
prepared or completed work leaves execution progress invalid until restart.
The existing sequential `execute_layer` delegates to these same three operations.

The distributed real layer-zero test passes on RTX plus four existing RoCE Spark
workers for two changed-input cycles of eighty rows across sixteen requests.
It covers embedding, attention, TP4 experts/reduction, mHC, next-layer mapped
Engram preparation and rejection/recovery of an incomplete model commit. Two
pure execution-progress tests also pass; the optional allocation-plan test was
skipped because its environment was not enabled. Build/state/distributed logs
are `/tmp/ds41-ced-bounds/layer-split-{build,state,gpu}.log`.

The initial distributed launch failed before inference because the transport
loader needed `DS41RT_NATIVE_LIB`; rerunning with the existing frozen native
library path passed. No GPU or worker restart was needed. Early encoder KV
publication and multi-chunk task scheduling still need to be connected to this
split execution interface. These are component results, not a new serving
throughput measurement.

## Implemented integration: reserved layer publication and execution

`Requests::prepare_encoder_layer` now connects reserved request validation to
`BackboneExecution::prepare_encoder_layer`. Source KV is committed immediately
after production, before indexing/attention, so the first learned selection and
subsequent consumers share the committed snapshot identity. Window KV is
published only after synchronous attention consumers drain. The returned
`PreparedLayer` borrows its lane, not the request/cache bank; another chunk can
be reserved while that FFN owner remains alive. Ordinary execution uses the same
preparation body with publication disabled.

The expanded real distributed test passes in 5.59 s on RTX plus four RoCE Sparks.
It executes reserved layers 0–3 over eighty rows/sixteen requests, covering
mapped Engram, ratio-two source publication at layer 2, learned-index reuse at
layer 3, and all four remote expert calls. A successor is reserved while layer
zero's FFN owner is live, accepted history remains at zero, and incomplete-pass
failure revokes both chunks. This proves interface ownership and actual reserved
layer execution, not simultaneous execution of the two chunks or full-model
numerical equivalence. Logs: `/tmp/ds41-ced-bounds/published-layer-{build,gpu}.log`.

The remaining integration is retaining/advancing independent chunk execution
state, scheduling ready tasks, publishing the source-20 boundary, and handing the
completed encoder suffix to decoder replay. Serving still uses the frozen serial
CED deployment until that path is qualified end to end.

## Implemented integration: complete reserved encoder transactions

`Requests::publish_encoder_boundary` connects the prepared layer-20 query to
source-20 production/publication. Execution marks the encoder source complete
only after publication succeeds, allowing the ordinary combined cache/Engram
commit to finish a fully published reserved chunk.

The distributed test now completes two queued five-token chunks across all
sixteen requests, executing all twenty encoder layers per chunk through the real
RTX and four RoCE Spark workers. Both Engram layers, every learned encoder index,
ratio-two carry across the odd chunk boundary and the ratio-one source-20 boundary
execute. After the first combined commit, the successor remains valid and accepted
history is five; after the second it is ten and decoder replay initialization
succeeds for every request. Ordinary incomplete-pass rejection remains covered.
The test passes in 5.94 s; logs are
`/tmp/ds41-ced-bounds/reserved-encoder-{build,gpu}.log`.

These chunks are queued ahead but still executed sequentially in this fixture.
This qualifies the complete reserved encoder transaction and handoff readiness,
not alternating execution, decoder output quality or serving performance. The
next integration is the chunk task loop and retained independent execution state.

## Implemented integration: paired encoder task execution

`Requests::execute_encoder_pair_layer` now runs two ordered chunks with independent
lane/index/execution owners and TP4 RoCE connection sets. It prepares the first
attention, then cooperatively polls its remote FFN alongside preparation and
execution of the second chunk. The first request's published KV is therefore
available before the second attention, while request history remains unaccepted.
Both outputs complete before advancing the pair to the next layer. An enclosing
RAII guard revokes both batches on returned error or future cancellation.

The distributed fixture runs this loop through all twenty encoder layers for
sixteen requests, two five-token chunks each. Both final residual and pre-state
byte arrays exactly match an earlier sequential reserved-encoder run with the
same token inputs and math. Both source-20 boundaries then publish, combined
commits advance to ten tokens, and decoder replay initialization succeeds. The
full test passes in 6.31 s. Logs:
`/tmp/ds41-ced-bounds/paired-encoder-{build,gpu}.log`.

This is actual paired task execution, not just reservation metadata. It is still
a small component fixture with a barrier between layer pairs; the test does not
measure GPU/NIC overlap duration, prove a throughput gain, or qualify cancellation
at every in-flight transport stage. Native serving must instantiate the additional
owners and route prefill through the pair loop, retain/capture the encoder suffix,
and qualify large prompts plus target/dSpark decode before rollout. Wider
wavefront scheduling and C16 API admission remain separate work.

## Serving follow-up: overlap the following query

Serving now constructs both passes and RoCE waves and runs reserved encoder pairs
through the final suffix/source-20 handoff. The next change moves the following
chunk's lane advance, Engram gate and query preparation into the interval after
the leading expert dispatch, and polls the leading FFN first deterministically.
The pair still waits at each layer boundary. The distributed sixteen-request
fixture remains byte-identical to sequential at both chunk outputs (6.39 s), and
both API lifecycle checks plus the prior eight per-mode quality outputs survive.

The isolated 16k comparison observes roughly 2–4% additional code prefill and
under 2% repeated-text prefill throughput. The earlier code decode slowdown is
not stable across runs, with first-round dSpark acceptance an observed contributor.
Serial CED remains selected. See [query overlap evidence](ds41-paired-query.md);
worker queue/transfer overlap, wider scheduling, and attention integration remain
open. The user's category-based decode comparison is queued after this prefill
work, without treating counting output as a comparable category benchmark.

## Independent progress selected for serving

`TargetPass` now runs both encoder chunk futures through all twenty layers, with
per-layer notification of leading KV publication. Expert completion is local to
each chunk instead of a shared layer barrier. Cache/history borrows remain
synchronous on the CUDA owner, and the enclosing guard revokes the participating admission
on cancellation. The pair still joins before suffix/source-20 handoff.

A real-model serving-method fixture matches serial suffix bytes for 65+65 and
79+1 rows, including odd compressor carry and the retained-window boundary. It
also cancels a suspended pair, verifies revocation and reuses both execution
owners successfully. API output/lifecycle qualification and isolated 16k plus
599-token decode comparisons support selecting this version on 18041/18042.
See [encoder progress evidence](ds41-encoder-wavefront.md) for rates, artifacts
and the still-open quality, wider scheduling and C16 requirements.
