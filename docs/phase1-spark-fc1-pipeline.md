# Phase 1: Spark FC1 pipeline probe

The current instrumented C1 trace attributes median six-row verification of
40.25 ms, including 20.28 ms waiting in receive/collection versus 0.16 ms of
upload. These nested host intervals are not additive GPU kernel attribution;
they motivate inspecting the Spark path, not assuming all waiting is compute.

An isolated CuTe prototype pipelines FC1 weight/scales over two or three shared
slots, preserving FC2, fused activation, quantization and ordered reduction.
It was tested against the original kernel at widths 64/128/192. All 54
variant/case combinations passed the independent oracle, changed input/route
checks, inactive-output poison checks and stable-allocation graph replay.
Each staged variant was bit-identical to the original at the same width.

On ostrich's GB10, width 192 remains the best observed configuration for mixed
six-row routing (22 distinct experts). Whole-tile FC1 buffering loses:

| Workload | Original 192, µs | Two stages, µs | Three stages, µs |
|---|---:|---:|---:|
| One row / six experts | 163.0 | 162.8 | 170.1 |
| Mixed six rows / 22 experts | 441.9 | 485.1 | 506.9 |
| Shared 16 rows | 227.5 | 233.5 | 239.2 |
| Shared 80 rows | 442.7 | 570.6 | 573.3 |

These are balanced interleaved CUDA-event timings including ordered reduction,
with nine samples of 20 replays each. They exclude route planning, input encoding,
transport and serving. Synthetic weights and representative routing were used;
this is not official-weight or prefill serving qualification. No formal
clock/throttle gate was applied, so these are exploratory measurements.

Nsight Compute basic profiling independently supports the occupancy explanation:

| Width-192 mixed-six kernel | Original | Two FC1 stages |
|---|---:|---:|
| Shared memory per block, decimal KB | 42.880 | 68.992 |
| Registers per thread | 160 | 196 |
| Shared-memory block limit per SM | 2 | 1 |
| Grid waves per SM | 2 | 4 |
| Achieved active warp occupancy | 11.54% | 8.39% |
| Instrumented kernel time, µs | 499.94 | 566.53 |

The prototype is **not integrated**. A smaller staging unit is the next useful
investigation: gate/up halves separately, or narrower K tiles, rather than
allocating two whole gate-plus-up tiles. FC2 double buffering can also reuse
space sized for both FC1 halves. Preserve accumulation and quantization order.
The default coordinator and all four resident workers remain unchanged.

## Reproduction

The [patch](phase1-spark-fc1-pipeline.patch) adds only isolated prototype and
benchmark modules to a frozen sparkinfer checkout. It does not modify the
production kernel. Apply it to source revision
`7299b3b92e70d539b2c0a63aaadce36932ceef4d` with `patch -p1`, mount that tree at
`/source`, and use `PYTHONPATH=/source` in `ds41rt-spark-expert-dev:latest`:

```sh
python /source/benchmarks/benchmark_pipeline.py
ncu --nvtx --nvtx-include 'profile/' --set basic -o /audit/pipeline-profile \
  python /source/benchmarks/profile_pipeline.py
```

The [raw summary](phase1-spark-fc1-pipeline.json) records all timing samples,
correctness results, source hashes, device UUID and image identity. Full logs
and the Nsight report remain under `/tmp/ds41-phase1-spark-pipeline` on ostrich.
