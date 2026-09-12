# Register-cached backbone RMSNorm

The official 5,120-wide BF16 RMSNorm now keeps each thread's input stripe in
registers across the reduction. It avoids the second input read and unrolls the
fixed-width loops. The reduction preserves the descending-stride FP32 addition
tree, using warp zero to combine the published partials and shuffles for the last
five levels. Block-wide barriers fall from nine to two. Other widths retain the
generic input loops with the same reduction tree.

The compiled specialization uses 36 registers and no local memory or stack
storage. The input cache therefore does not spill intermediates into VRAM.

## Evidence

`scripts/qualify-ds41-rmsnorm.py` passes 116 bit-exact BF16 comparisons against
the preceding native library, covering small/large row counts, odd and model
widths, zero inputs and four magnitude scales. Compute Sanitizer racecheck reports
zero hazards on the specialization at 1, 6, 80 and 2,048 rows.

Warm CUDA-graph microbenchmarks, with identical inputs and alternating library
order, measured these median kernel durations:

| Rows | Previous | Candidate |
|---|---:|---:|
| 1 | 6.72 µs | 3.77 µs |
| 6 | 7.23 µs | 4.04 µs |
| 48 | 7.23 µs | 4.09 µs |
| 2,048 | 16.66 µs | 10.77 µs |

The full-model effect is much smaller. Paired 599-token counting responses after
16k code context, on the same GPU for each mode, preserve output and token usage.
Target-only averages 39.73→39.78 TPS across two pairs, effectively unchanged.
The last two of three dSpark pairs average 120.55→121.91 TPS, about 1.1% higher.
Exact measurements are in [the result record](ds41-rmsnorm.json); these small
samples do not establish statistically precise end-to-end gains.

The speculative lifecycle and C2/C6/C16 distinct-sequence cancellation/replacement
checks pass. Eight target-only comparison prompts preserve text and usage; the
inherited Unicode-format failure remains, so strict quality qualification is
still incomplete. Both selected normal APIs pass post-deployment lifecycle checks.

## Selected artifacts

The frozen native directory is `/tmp/ds41-rmsnorm/selected-native`, library SHA-256
`d56c464ce91bf68ca349838fb7854ce273512c4ba80f6ecd139f9a4994446f4b`.
Containers `ds41-rmsnorm-live-target-api-dev` and
`ds41-rmsnorm-live-spec-api-dev` serve 18041/18042. The daemon remains
`/tmp/ds41-boundary/daemon`; Spark workers and the pinned b12x revision are unchanged.
The preceding boundary containers are stopped for rollback. Raw qualification,
racecheck, benchmark and creation records are under `/tmp/ds41-rmsnorm`.

This improves a recurring kernel; it does not remove the coordinator's graph
launches or host synchronization points. Combining mHC input preparation with
the query graph remains a separate, larger scheduling investigation.
