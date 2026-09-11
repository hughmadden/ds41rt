# Small-row FP8 plans alongside prefill capacity

The RTX runtime now preloads FP8 plans for capacities 1, 16, 80 and the full reserved capacity, retaining only the distinct capacities that fit. A launch selects the smallest preloaded plan covering its live rows. This applies to backbone projections, grouped WO-A, shared FFN, index query, Engram and dSpark projections. Weight packing and the exact-capacity native ABI stay unchanged.

Each plan has its own 256-byte-aligned scratch slice, initialized before graph capture. Disjoint slices prevent one layout from corrupting another layout's padding or scales. Selection performs no allocation or module resolution. Checked 64-bit layout arithmetic and the CPU-only budget gate account for the extra scratch. For the six backbone projection geometries, capacity 4096 adds 7,778,304 bytes relative to the old full-capacity scratch.

## Measured API behavior

Frozen code context and repeated `amber` filler, 16,410/16,411 actual prompt tokens, greedy counting from 1 to 20 (59 completion tokens), sequential C1 with no prefix-cache hits. Prefill rate is prompt tokens divided by time to first content, including API overhead. The table reports the second request in each pair; all first requests are retained in the JSON.

| Version / batch | Code prefill tok/s | Repeated prefill tok/s | Code decode target / dSpark tok/s | Repeated decode target / dSpark tok/s |
|---|---:|---:|---:|---:|
| Prior / 1024 | 1235 / 1243 | 1525 / 1533 | 25.91 / 93.60 | 25.83 / 94.66 |
| Prior / 2048 | 1333 / 1337 | 1501 / 1512 | 23.80 / 82.31 | 23.87 / 90.66 |
| Small plans / 1024 | 1238 / 1248 | 1531 / 1548 | 26.71 / 85.37 | 26.66 / 96.29 |
| Small plans / 2048 | 1342 / 1327 | 1513 / 1514 | 26.57 / 85.92 | 26.60 / 96.17 |

Prefill entries show target / dSpark. Prior measurements are preserved in `ds41-intermediate-prefill-batches.json`. All sixteen new prefill outputs, usage records and prompt hashes match their corresponding prior 1024 controls. Larger reservation no longer causes the target's previous ~8% decode regression. There is no claim of a large prefill speedup from these small-row plans.

Code-context dSpark measurements remain variable: new 1024's first request was 95.40 tok/s versus 85.37 on its second. A separate controlled old/new 1024 trial with `ds41rt::timing=debug`, two identical code requests each and GPU0 quiet, gives warm rates 91.98 versus 92.93 tok/s. Both warm requests take ten verification steps and accept 49 of 50 proposed tokens. Their summed speculative-step intervals are 630,481 versus 624,041 microseconds; draft intervals are 73,377 versus 73,608. These observations do not establish the cause of the unprofiled variation, nor justify a special-case numeric policy for the main projection.

## Qualification and limits

- Release daemon build and daemon check pass; the FFI scratch-layout unit covers alignment, offsets past 2 GiB, overflow and invalid geometry/capacity.
- CPU-only official backbone budget test passes without GPU devices. Stale expected weight/workspace totals from earlier changes were corrected; full-capacity 4096 workspace is 2,964,176,908 bytes and resident lane weights 6,896,423,360 bytes.
- Native real-weight probe covers ten FP8 geometries, eleven live-row transitions each, two input mutations, poisoned output tails, scratch guards and graph replay without Torch allocation. 216/220 comparisons are bit-exact against the old 4096 plan. Worst relative L2 is 0.0001451, max absolute difference 0.0625. Nonexact cases are one WO-B M1 mutation and three main-projection M1 mutations. The bounded gate allows relative L2 <= 0.001 and at most two BF16 steps at the reference tensor maximum plus 0.0001. This is not a mathematical reference or broad model-quality test.
- Native probe used two successful subsets after fixing graph-stream capture and recognizing the stored WO-A tensor is two-dimensional. The combined artifact contains all 110 cases. Later script metadata/ABI checks were not followed by another full run. Actual Rust dispatch is exercised by the API trials, not by the ctypes probe.
- Both 1024 and 2048 APIs pass JSON/streaming, cancellation/recovery and unsupported-sampling lifecycle checks. Short 2048 counting measures about 38 target and 117 dSpark tok/s.
- Eight paired quality cases are identical between new 1024 and 2048. Seven preserve prior outputs/usage. The target Unicode arithmetic response gains one blank line and now matches dSpark; the answer stays 36. Both modes still fail the inherited strict-only-integer requirement (5/6 objective checks), so the quality command returns1. Pair equality is now 8/8, not a broad quality qualification.

Warm component graph medians improve substantially (M6 main projection 129.0 to 36.9 microseconds, WO-B 76.1 to 22.6, Engram 214.2 to 114.8). Cache residency is uncontrolled; these are not full-model speedups.

The live trial uses batch 2048, reservation 4096, on ports 18041/18042 with `ds41-fp8plans2048-{target,spec}-api-dev`. Frozen daemon SHA256 is `00fb072c122eb9ae1c623e137a52ca57e1c50b636b3684a76a9a61a375a2d43c`; RTX native remains `07db52dec294d18ee98e44b553a7c442a43cdabb65ff8103690572fc4eeaa42b`. All four RoCE workers remain unchanged. Prior 1024 containers are retained stopped for rollback. This does not change the CLI default batch size. Launch arrays and selected raw measurements are in the companion JSON; full logs are under `/tmp/ds41-fp8-plans`.

At 16k, observed prefill remains around 1.3k code and 1.5k repeated tok/s, below the proposed 2–3k checkpoint. The prior warm 2048 profile attributes 7.812s to expert phases (5.952s collection) and 3.212s to sparse attention within 12.368s overall. Matched remote GPU work totals 3.499s, excluding tail/decode. These nested intervals are not additive and collection includes remote compute. Attention and coordinator/transport overhead remain material priorities; sparse C1 work distribution and the remaining registered-receive-to-pinned-frame copy need separate profiling and qualification.
