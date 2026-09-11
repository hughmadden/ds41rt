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
