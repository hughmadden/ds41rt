# Parallel mHC projections for small batches

For one through sixteen token rows, mHC mixing now exposes twenty-four independent projection CTAs per row. Each warp retains the original lane-wise sequential FMA and shuffle reduction order. It computes the same residual normalization and stores each normalized projection temporarily into the existing pre/post/comb output buffers. A second kernel consumes all twenty-four values before writing the sigmoid and twenty-iteration Sinkhorn results. No allocation, scratch expansion, ABI change or host synchronization is introduced. Stream completion still governs output visibility.

The projection loop uses an unroll factor of thirty-two to expose independent loads while retaining accumulation order. Simply distributing the original loop saved only a few microseconds; distributing it with unrolled loads roughly halves small-batch latency. Unrolling the original single-CTA implementation alone did not materially help. Above sixteen rows the original implementation remains selected because it already has row parallelism and the new path regressed larger batches.

Full native-library GPU graph measurements:

| Token rows | Before µs | Selected µs |
|---:|---:|---:|
| 1 | 32.87 | 16.43 |
| 2 | 32.85 | 16.46 |
| 6 | 34.88 | 16.44 |
| 16 | 34.88 | 16.63 |
| 17 | 34.92 | 34.89 |
| 80 | 43.09 | 43.17 |
| 256 | 69.72 | 69.72 |
| 4096 | 945.07 | 944.90 |

These are alternating CUDA-event measurements around ten graph replays, with twenty measured samples per path after warmup. They are not release throughput or controlled-clock measurements. One row denotes one token's mHC residual, not one of its six routed experts.

`python/tools/qualify_v41_hc_parallel.py` compares the original and selected full libraries. All thirty-two synthetic graph-replay cases pass FP32 bit equality: eight row counts spanning the dispatch boundary and random, zero, large and small residuals. All eighty-six official backbone/dSpark mHC weight sets also pass at rows 1/6/16, yielding 258 exact real-weight comparisons. Native and CUDA CTest both pass; the qualification script passes Ruff.

Both deployed APIs pass streaming counting/usage, JSON arithmetic, cancellation recovery and unsupported sampling rejection. All eight paired quality cases retain exact text and usage against the pinned-router deployment, with all six objective checks passing. These bounded comparisons do not close older broad numerical-quality gates.

Live FFN preparation median falls from 96 to 57 µs for one-row target calls and from 97 to 54 µs for six-row verification calls. The live stage includes additional boundary work; it is not the isolated kernel interval. Three fresh target counting streams measured 25.17/25.22/25.20 TPS before and 26.71/26.96/26.78 after. dSpark measured 85.26/90.35/90.32 before and 91.72/95.24/95.29 after. Modes ran sequentially, share the same Spark workers, and have no clock/throttle admission. These are short diagnostic improvements, not evidence of the 90/270 TPS goals.

Running coordinators are `ds41-hc-parallel-target-api-dev` and `ds41-hc-parallel-spec-api-dev`, ports 18041/18042. Their executable remains `/tmp/ds41-router-pinned-artifacts/daemon`; native library is `/tmp/ds41-hc-parallel-artifacts/cmake/libds41rt_native.so`. The four local-QP Spark workers and RoCE inference path are unchanged. Prior pinned-router coordinator containers are stopped and retained.

[Exact comparisons, timing data, API outputs, commands and hashes](ds41-hc-small-batch-parallel.json) retain the evidence. Full logs are under `/tmp/ds41-hc-parallel` and `/tmp/ds41-hc-parallel-rollout`. Sparse attention, projection, transport staging/client handling, C16/alternating waves, long-context/prefill and release cleanup remain open.
