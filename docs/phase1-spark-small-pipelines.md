# Phase 1: smaller Spark staging units

Following the [whole-tile occupancy regression](phase1-spark-fc1-pipeline.md),
three isolated CuTe variants preserved the fused arithmetic and were compared
with the original kernel using changed inputs, expert mappings, live counts,
poisoned inactive output, and fixed-allocation graph replay:

- FC1 gate/up halves staged separately within the original shared weight space.
- FC2 output tiles double buffered, reusing the existing FC1 weight space.
- FC2 double buffering with asynchronous raw scale-word copies, followed by
  scale unpacking from shared memory. This removes synchronous global scale
  gathers from the prefetch path, including cross-word shifts at width 192.

All three passed six cases at widths 64/128/192, including bit-exact equality
with the original kernel at each width. The full fixture also checks an
independent numerical oracle. No variant is integrated into serving.

Width-192 medians, microseconds, original → candidate within each interleaved run:

| Variant | One row | Mixed six / 22 experts | Shared 16 | Shared 80 |
|---|---:|---:|---:|---:|
| FC1 halves | 159.5 → 187.9 | 462.5 → 481.6 | 227.3 → 239.5 | 445.6 → 455.1 |
| FC2 tiles | 165.0 → 156.5 | 455.9 → 457.6 | 233.2 → 224.8 | 453.8 → 448.8 |
| FC2 async scales | 136.6 → 120.9 | 449.6 → 451.2 | 236.5 → 215.1 | 445.1 → 449.8 |

Each median has nine rotated/reversed samples of 20 graph replays, including
ordered slice reduction. Do not compare absolute values across separate runs:
there was no formal clock/throttle gate. These synthetic component probes exclude
encoding, planning, transport and serving. Shared-80 is not a prefill serving gate.

## Instruction-level evidence

Nsight full profiling of original versus FC2 tile buffering confirms that the
smaller buffers preserve the shared-memory block limit of two per SM: allocated
shared memory is 42.880 versus 43.392 decimal KB, and registers are 160 versus 162.
Instrumented kernel times are 513.38 versus 508.99 µs, distinct from event timings.

The original's long-scoreboard stalls occupy about 79.3% of the average cycles
between issued instructions. SASS sampling puts major waits at the FC1
`DEPBAR`/barrier, FC1 MMA consumers, and FC2 scale-word shifts after global loads.
This motivated the asynchronous-scale variant. Both profiled kernels fetch
about 3.242 million system-memory fill sectors; adding stages does not reduce
weight traffic. The async-scale variant helps cases with six distinct experts in
this probe but does not establish a mixed-six gain. Neither headline stall
percentages nor profiler speedup estimates are predictions of serving gains.

The next investigation should return to full target/draft verification costs,
with actual routing and uninstrumented comparisons. Do not keep increasing
staging depth without new evidence. Coordinator and resident worker binaries
remain unchanged, and the standard API is healthy.

## Reproduction

Use the frozen source, GPU and dev-image setup from the previous probe. Before
applying one selected patch, copy the original source file:

```sh
cp b12x/moe/_shared/kernels/w4a8_v41_slice.py \
   b12x/moe/_shared/kernels/w4a8_v41_pipeline.py
```

Apply one of [FC1 halves](phase1-spark-half.patch),
[FC2 tiles](phase1-spark-fc2.patch), or
[FC2 asynchronous scales](phase1-spark-fc2-async.patch) with `patch -p1`.
Run `/source/benchmarks/benchmark_small_pipeline.py` with `PYTHONPATH=/source`.
Each patch targets only the copied prototype and adds a benchmark; it does not
modify the production source. The previous probe's profile script also works
with these two-stage variants; FC2 used `--set full` and a kernel-name filter
`regex:.*w4a8_v41.*` inside the `profile/` NVTX range.

[Raw samples and metrics](phase1-spark-small-pipelines.json) retain source hashes.
Full Nsight reports, SASS sampling, and logs are under
`/tmp/ds41-phase1-spark-fc2` on ostrich; the other probe directories use suffixes
`half` and `fc2-async`.
