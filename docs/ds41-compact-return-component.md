# Compact BF16 expert returns: implementation and component qualification

The candidate native backbone path now sums each Spark's six local FP32 route vectors into one BF16 hidden-width partial before D2H and transport. RTX sums the four BF16 partials in rank order using FP32, adds the BF16 shared expert and rounds once to BF16. All three dSpark stages retain their existing RTX execution and reduction.

This implements the small-batch return design identified by the [optimization audit](ds41-optimization-parity-audit.md). It does not yet implement wide Spark-side reduction, verbs transport, direct compact expert epilogues or larger prefill scheduling.

## Protocol and ownership

- Revision-3 frames explicitly request and echo `EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16` (bit 16, separate from the collective-part count bits). The native request parser requires this flag and the collector requires it on responses. Previous workers reject the unknown flag; previous request frames are rejected by new workers. Rollout must update all four workers and the coordinator together.
- Response geometry is BF16 `[rows,5120]`, stride 10,240 bytes, instead of FP32 `[rows,6,5120]`, stride 122,880 bytes. That is 12× less return payload: 40,960 bytes/token/layer across four ranks instead of 491,520.
- Spark wave owners allocate a reusable BF16 compact output. Host exchange storage and coordinator rank buffers use the compact extent. At capacity 80, the four coordinator partial buffers use 3,276,800 bytes instead of 39,321,600 bytes. GPU route scratch still exists; removing its materialization is a subsequent kernel optimization.
- Input activations and route IDs/weights retain their previous representation and ordering. Frame identity, rank identity, duplicate/reorder checks, cancellation and stream draining remain enforced.
- The separate native entry points allocate nothing and support graph capture. The original per-route reducer and device route views remain available for numerical diagnostics. New FFI symbols load only when the compact reducer is constructed, leaving diagnostic/dSpark consumers independent.

## Qualification

The [reusable GPU qualifier](../scripts/qualify-ds41-compact-reduction.py) compares the new operation against explicit ordered FP32 sums and BF16 conversions. On RTX it covers synthetic 1/16/80/129/257-row cases and two saved real layer-0 batches of 80 rows. On Spark it covers the five synthetic sizes. Every case passes exact partial/final equality, optional shared output, exact shared/output aliasing, two changed-input graph replays, invalid overlap/alignment/zero-row/null-rank rejection, and successful execution after rejection. The 257-row case also exercises the capped grid's stride loop.

[Raw component results and library/device identities](ds41-compact-return-component.json).

The transport crate passes 148 tests with one ignored test, including complete/chunked native responses, identity/geometry/format-flag rejection, cancellation, reconnection and persistent within-wave dispatch. The release daemon builds. These tests do not claim that a full model using the new arithmetic produces identical logits to the old model path: local route accumulation changes the rounding order, as quantified separately in the audit.

Reproduction with the coordinator development image uses:

```sh
python scripts/qualify-ds41-compact-reduction.py \
  --native-lib /candidate/libds41rt_native.so \
  --real-planes /planes \
  --output /output/compact-gpu.json
```

`/planes` contains the saved `rank{0..3}/l0-c{0,1}-plane.bin` FP32 route outputs; omit `--real-planes` for the synthetic-only Spark checks. The caller exposes the intended CUDA device and mounts the matching native candidate library. The qualifier performs no model loading or network calls.

## Rollout gate still open

At this checkpoint the running APIs/workers still use their frozen previous binaries and libraries. Candidate native libraries were built separately at `/tmp/ds41-compact-cmake` on raptor and ostrich. The changed native response contract is not yet deployed to those services.

Next: build/freeze matching Spark workers, update all four workers and the coordinator with isolated artifacts, run selected real FFN/full-target comparisons and the paired live target/dSpark quality suite, then measure decode, acceptance, prefill, response bytes and phase timings. Retain the old frozen deployment for rollback. Do not infer an end-to-end speedup from the byte ratio or component exactness, and do not call the wire/performance release gates complete before these checks.
