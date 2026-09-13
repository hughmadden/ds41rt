# Dense projection profile and query-B tile probe

One RTX PRO 6000 Blackwell, **400 W power limit, standard memory speed**
(13,365 MHz observed), four unchanged Sparks, and five local routed-expert
layers. The serving profile uses the 18 × 1,048,576-token KV configuration with
24 retained snapshots. No builds overlapped serving measurements. Production
kernels remain unchanged after this probe.

Fresh Nsight Systems traces of one warmed no-thinking code request in each mode
identify the following recurrent captured dense kernels. These are instrumented
GPU intervals, excluding quantization and reduction, not serving throughput.
The speculative column combines backbone and draft calls of the same geometry.
One-off prefill nodes are excluded using repeated graph-node signatures.

| Projection | Target median µs/call | Speculative median µs/call |
|---|---:|---:|
| Query-B, N32768 K1280 | 28.51 | 29.57 |
| Output-B, N5120 K8192 | 28.89 | 29.54 |
| Grouped output-A, eight N1024 K4096 groups | 22.62 | 22.98 |
| Query-A, N1280 K5120 | 9.92 | 12.70 |
| KV, N512 K5120 | 9.02 | 11.87 |
| Shared gate/up, N2304 K5120 | 11.07 | 13.57 |

The profiles produced identical code text and passed the structure checks.
They do not establish a speedup. The signature map in
`scripts/summarize-ds41-dense-profile.py` is specific to this native artifact;
changed exports require reviewing the map.

Query-B already uses a 16×64 tile at capacity 1 and 32×128 at capacity 16.
The probe compiled four tiles—16×64, 16×128, 32×64 and 32×128—at both static
capacities through the existing AOT factory. All eight plans were compiled
before timing. Native K32 quantization and contiguous official layer-0 FP8
weights/scales were retained. Candidate graphs include quantization and GEMM,
and are compared with the existing native C ABI pipeline.

The best apparent warm-cache candidate, 16×128, loses that advantage when each
measurement follows a 256 MiB read/write eviction pass:

| Live rows / capacity | Warm native → 16×128, µs | Eviction native → 16×128, µs |
|---|---:|---:|
| 1 / 1 | 19.06 → 13.72 | 33.96 → 34.13 |
| 5 / 16 | 18.47 → 14.35 | 34.56 → 34.30 |
| 6 / 16 | 18.36 → 14.56 | 34.73 → 34.39 |
| 16 / 16 | 15.82 → 11.62 | 35.39 → 35.24 |

Each entry is the median of six rotated/reversed samples, each averaging 12
event intervals. Eviction runs outside those intervals. It is a cache-pressure
proxy, not proof that every cache line was invalidated. Hardware snapshots stayed
at P1 with no active throttle reasons, memory at 13,365 MHz, and SM clocks from
2,707 to 2,805 MHz. Clocks were not locked, and this is not formal release
performance qualification. The small eviction-case differences do not justify
integration or a serving speedup claim.

All 40 candidate comparisons were bit-exact against native output: two input
mutations for every tile at live rows 1 under capacity 1, and 1/5/6/16 under
capacity 16. Outputs were finite/nonzero, inactive output tails remained poisoned,
weights remained byte-identical, and replay allocated no Torch memory. Kernel
resolution was frozen while live row counts changed. This is a native-output
comparison with synthetic activations, not a separate mathematical oracle or
full-model quality qualification.

The next structural experiment is split-K for narrow query-A and KV projections:
their recurrent target launches expose only 20 and 8 CTAs respectively on 188
SMs. Query-B already exposes a full SM grid, and output-B already uses two-way
split-K at one row. Extra parallelism must be evaluated with its reduction and
quantization costs, followed by complete serving validation if promising.

Both profiling and component runners restored the standard service. Initial
probe failures were a missing CuTe runtime library path and a telemetry UUID
format; both were corrected before the complete recorded run.
[Evidence](phase1-dense-query-profile.json) includes raw timings, identities,
commands, hashes and raw artifact locations. Reproduce the component probe with
`python/tools/bench_v41_dense_plans.py` using its explicit native-library,
snapshot and output arguments, GPU0 selection and the CuTe runtime library path.
