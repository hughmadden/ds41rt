# Missing CED prefill and inherited scheduling gaps

Historical audit: the missing CED execution described below was fixed by the
[bounded CED rollout](ds41-ced-serving.md). Scheduling remains open; see the
[post-CED dependency audit](ds41-ced-scheduling-audit.md) for current constraints.

The deployed native path does not obtain DeepSeek V4.1's advertised 8B prefill
compute benefit. It runs all forty backbone layers for every prompt token. This
is a more important performance gap than the attention fragment prototype tested
in the same turn.

## Authoritative evidence

- `rust/crates/ds41rt-daemon/src/v41_native_serve.rs`: `run_job` iterates prompt
  chunks and completes `step` on each before starting the next.
- `rust/crates/ds41rt-daemon/src/v41_target_pass.rs`: `execute` unconditionally
  loops `for layer in 0..40` for every batch; every iteration reaches distributed
  backbone execution. This is actual full decoder work, not just decoder cache
  projection.
- The [current profile](ds41-packed-prefill-profile.json) records 360 layer
  executions across nine prefill chunks. The first eight contain 2048 live rows
  each and the last contains 26.
- The pinned official report in `/tmp/ds41-reference/report.txt`, sections 2.2 and
  3.2.2, specifies encoder-only full-prompt processing, decoder global KV projected
  from final encoder states, and bounded decoder replay over the final window.
  Its README states 8B active parameters during prefill and 16B during decode.
- The simple official Python `Transformer.forward` also loops over all layers.
  Reproducing that loop is insufficient to implement the report's efficient CED
  serving algorithm. The runtime followed that full-forward structure too far.

For 16,410 tokens, current block work is `16410 * 40 = 656400` token-layer blocks.
The intended bounded path is `16410 * 20 + 128 * 20 = 330760`, plus decoder global
cache projection/packing and other overhead. This is a block-work calculation,
not a guaranteed twofold throughput improvement or a physical bandwidth estimate.

Bounded replay is intentionally approximate. The report explicitly distinguishes
it from exact decoder SWA reconstruction, which needs a much larger dependency
suffix (described as decoder depth times window size). A truncated 128-row decoder
must use truncated SWA bounds; it cannot read uninitialized earlier decoder windows.
The report describes post-training adaptation and small quality impact, but this
runtime still needs its own quality and acceptance evidence.

## Comparison with older engines

The reviewed GLM checkout at `../glmrt-release` contains
`execute_scheduler_bounded_long_prefill_wavefront` in
`rust/crates/glmrt-daemon/src/commands/real_full/scheduler/execution.rs`.
It tracks started/finished layer-by-chunk tasks, permits up to sixteen resident
chunks, issues pending sparse dispatches, and preserves causal/chunk and dSpark
checkpoint dependencies while progressing other ready work. Its dispatch is split
into start and finish operations. Native V4.1's complete-step chunk loop does not
reach that machinery. Inherited code elsewhere in this tree does not establish
native integration.

The older DS4 configuration selects Spark-side reduction for batches of at least
sixteen rows (`SPARK_REDUCTION_MIN_ROWS=16`). Native V4.1 still returns four compact
rank partials and reduces them on RTX. This difference is real; whether restoring
Spark reduction wins on this fabric requires a new crossover measurement.

Native `v41_experts/execution.rs::execute_host_chunks` calls
`execute_host_request` to finish the entire request before emitting response chunks.
That path launches compute and compaction, synchronizes, copies the whole output
to host, then calls the chunk sink. It provides framed transport streaming, but
not overlap of unfinished expert computation with early output transfer.

Expert-major ordering, M16 grouping, resident GPU-packed FP4 weights, fused expert
intermediates, compact BF16 returns and persistent RoCE are already present. The
missing scheduling does not mean those optimizations were all lost. Conversely,
small kernel gains do not close the missing CED/wavefront execution contract.

## Implementation priority and gates

1. Split full-prompt encoder progression from bounded decoder progression. Publish
   decoder compressed KV/index source state from all encoder rows, retaining the
   final encoder suffix needed for decoder replay. Respect mHC preparation and
   source-layer projections rather than treating raw encoder output as ready KV.
2. Give encoder/global-source progress and decoder window replay explicit state
   and bounds. Do not fake committed decoder history or publish nonexistent rows.
   Rebuild decoder windows and dSpark taps/prefix state from the final 128 rows,
   then select the final prompt logit. Preserve cancellation and restart ownership.
3. Compare against independently implemented bounded reference execution as well
   as full-forward quality; cover short prompts, window/chunk boundaries, partial
   compressor groups, long prompts and subsequent target/dSpark decode. Report
   changed outputs and acceptance, not a false bit-exact full-forward gate.
4. Restore overlapping ready prefill work and bounded request waves using the old
   dependency scheduler as a guide, adapted to CED source/index/window lifetimes.
   Then evaluate earlier output streaming and Spark reduction independently.

## Deferred attention prototype

A standalone native prototype replaces the online-softmax accumulator shared
round trip with a 16-by-16 scale fragment loaded through the same WMMA accumulator
layout. Matching fragment elements are multiplied in registers, without encoding
an undocumented lane map. Seventy cases across ten cache/masking patterns and
rows 1/6/16/80/256/1024/4096 are bit-exact against the deployed packed library.
Six alternating timing samples give roughly 9% kernel speedup on full selection:
M1024 4170.8 to 3826.2 microseconds; M4096 16861.7 to 15424.2 microseconds.
There are no spills. Small-row split qualification, final artifact integration
and API validation are not complete, so it is not deployed. Source and detailed
results are preserved in [the evidence record](ds41-attention-fragment-prototype.json)
and `/tmp/ds41-rescale-fragment`. Both selected APIs remain unchanged.
