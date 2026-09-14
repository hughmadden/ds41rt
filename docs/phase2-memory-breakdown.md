# Measured dual-RTX memory breakdown

Measured 2026-09-14 using the serving binary (not the test binary), after startup
and before request-time graph warmup. Settings: 2048-row prefill, C16 admission,
dSpark enabled, 24 prompt and 24 turn snapshots plus two in-flight snapshot slots,
20/20 attention placement, all 20 encoder routed and all 40 shared layers TP2.
Both GPUs have 400 W power limits. This measurement includes the current lane-local
index scratch reuse and smaller second-lane decoder index/tap buffers. It is an intermediate configuration, not release qualification.

Values below are **GiB (2^30 bytes)** measured as CUDA-reported occupancy changes
around component construction. Thus they include allocation rounding and any CUDA
resources loaded with that component, rather than just tensor-file payload sizes.
The tiny head-normalization allocation did not change rounded occupancy separately.

| Component | RTX0 GiB | RTX1 GiB |
|---|---:|---:|
| Encoder routed experts, layers 0–19, TP2 halves | 67.266 | 67.266 |
| Shared experts, layers 0–39, TP2 halves | 0.703 | 0.703 |
| Attention, mHC and router weights (0–19 / 20–39) | 2.600 | 2.600 |
| SWA/compressor projection weights | 0.113 | 0.084 |
| Index-query weights | 0.018 | 0.029 |
| GPU Engram weights, layers 1 and 14 on RTX0 | 0.305 | 0.000 |
| Embedding table on RTX0 | 1.234 | 0.000 |
| Vocabulary head, half on each card | 0.617 | 0.617 |
| dSpark weights on RTX1 | 0.000 | 7.432 |
| Vision, aligner and markers including workspace, RTX0 | 1.592 | 0.000 |
| Target execution workspaces, both independent lanes | 8.951 | 6.340 |
| TP2 expert/shared and transport workspaces, both lanes | 1.521 | 1.756 |
| dSpark runtime/workspaces including split vocabulary | 0.154 | 1.041 |
| Prompt/turn snapshot arenas | 0.127 | 0.012 |
| FP4 compressed KV + index + FP8 SWA and cache state | 8.398 | 5.598 |
| CUDA contexts and peer access | 0.546 | 0.556 |
| **Total occupied at readiness** | **94.146** | **94.032** |
| Unallocated, including 800 MiB runtime headroom | 0.824 | 0.935 |
| **CUDA-visible total** | **94.970** | **94.967** |

The global compressed/index pool is **14,960,885,760 bytes (14.96 GB / 13.933 GiB)**,
representing **16,809,984 source-token positions including COW/tail allowance**.
Source ownership remains 2/8/14 on RTX0 and 20 on RTX1; SWA follows attention
layers 0–19 and 20–39 respectively. CUDA-visible capacity excludes device-reserved
memory, so it differs from the nominal capacity shown by `nvidia-smi`.

The original weight estimates were fairly close. The major missing cost was the
execution storage: two target lanes, TP2 transfer/output/scratch, dSpark state and
projection/sampling storage, vision workspace, and CUDA overhead. On RTX1 the two
target lanes plus TP2 and dSpark workspaces total about 9.14 GiB, in addition to
the 7.43 GiB dSpark weights already anticipated by the plan. Vision's old ~0.904 GiB
estimate covered weights; the measured 1.592 GiB includes its workspace.

The previous 4.82 GB pool figure came from the test binary before index-scratch
reuse. Test-only trace buffers added about 0.285 GiB per GPU (0.57 GiB across both),
which reduced the pool further because RTX1 was limiting. These numbers supersede
that fixture-based capacity estimate; further memory optimization is still needed
to reach the intended aggregate context budget.

Both encoder lanes retain full 2048-row backbone/query, compressor and TP2
expert buffers. Encoder chunks also produce compressed source 20 on RTX1; shrinking
all second-lane RTX1 buffers to 80 rows was invalid and failed long prefill. Only
its decoder index selection and target taps now use 80-row buffers, sufficient for
48 verification rows. Source-20 encoder production does not run index selection.
The first lane retains full capacity for decoder replay and cached continuation.

Compared with full-size second-lane index/tap buffers, fixed RTX1 occupancy fell
by approximately 0.885 GiB and the pool grew from 9.74 GB / 10.94M positions to
12.11 GB / 13.61M positions. That intermediate configuration was limited by RTX1, with approximately
0.43 GiB surplus on RTX0 beyond its then-2-GiB runtime reserve. Embeddings remain on RTX0 beside
layer 0. A dSpark TP2 split needs a separate performance assessment and would move
much more memory than the current imbalance warrants.

Validation: the actual HTTP server completed a 3013-token arithmetic prompt and a
concurrent code request, exact-prefix reuse with 3013 cached prompt tokens, a
retained conversation continuation, and 16 concurrent code requests. These check
serving/reuse, not throughput, constrained tool output or vision quality. The
32-case distributed correctness/cancellation fixture, single-RTX target fixture
(including C16), and 27 serving unit tests also passed.

Source measurements are retained in
`~/.cache/ds41rt-experiments/phase2-planner/lane-capacity-server.log` and
`lane-capacity-api.json` in the same directory. The preceding full-size measurement
is in `memory-audit-server.log` and `memory-audit-api.json`.

The approved default now targets sixteen full 1,048,576-token contexts plus 64
page groups for partial tails/COW, while preserving 24 prompt and 24 turn snapshot
slots. Dual-RTX runtime headroom is 800 MiB per GPU; single-RTX headroom is unchanged.
The table reflects the larger pool allocated by the optimized serving binary,
with measurements in `dual-release-server.log`. Warmed C16 counting and same-code
requests have completed; a chart-description vision request, 32K needle, retained continuation and cancellation
checks also passed. Broader peak-allocation checks remain pending.
This is pool capacity, not a claim that sixteen simultaneous 1M-token requests
have yet been exercised. Retained snapshots share the global pool.
