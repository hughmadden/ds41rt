# DS41RT Phase 2 release performance

All RTX measurements use an enforced 400 W power limit and the standard
14,001 MHz maximum memory speed with no memory overclock. Loaded work ran at a
13,365 MHz memory clock. The RTX driver is 595.91.07; the four GB10 workers use
580.159.03. Results were collected on 2026-09-14 from release image engine
`f652dc848e24335242ae245d92f18f7d73cd0f39`, with three measured samples per
cell. Throughput uses temperature zero and thinking disabled.

## Headline results

| Measurement | 1 RTX | 2 RTX | 2 RTX change |
|---|---:|---:|---:|
| Best median prefill | 7,878 tok/s (0 + 32K) | **8,454 tok/s** (0 + 16K) | +7.3% |
| Counting target-only decode | 44.95 tok/s | **48.41 tok/s** | +7.7% |
| Counting dSpark decode | 156.08 tok/s | **181.31 tok/s** | +16.2% |
| Weighted eight-type target-only decode | 43.04 tok/s | **46.14 tok/s** | +7.2% |
| Weighted eight-type dSpark decode | 76.72 tok/s | **79.33 tok/s** | +3.4% |
| C16 code aggregate | 992.78 tok/s | **1,181.49 tok/s** | +19.0% |
| C16 topic aggregate | 514.96 tok/s | **596.14 tok/s** | +15.8% |
| C16 counting aggregate | 1,161.12 tok/s | **1,333.57 tok/s** | +14.9% |
| C16 mixed aggregate | 196.46 tok/s | **309.06 tok/s** | +57.3% |
| RTX routed-expert placement | Layers 0–4, full width | Encoder 0–19, TP2 | — |
| Spark expert residency / configured budget | 40 layers / 100 GiB each | 20 layers / 100 GiB each | — |
| Global FP4 source pool | 16.681 GB / 18,710,016 logical tokens + 32,768 private-tail | 13.094 GB / 14,680,064 logical tokens + 32,768 private-tail | — |
| Exact prompt / completed-turn retention | 24 / 24 entries | 24 / 24 entries | — |
| Loaded RTX memory after readiness | 95,080 MiB | 95,338 / 95,578 MiB | — |
| Standard restart to API readiness | 58.05 s | **46.00 s** | −20.8% |

The weighted direct-corpus result is the main real-content decode headline.
Counting is retained as a low-entropy throughput reference. The mixed workload
has different prompt and output lengths from the same-code workload and should
be compared only against its matching layout.

## Eight content types and counting

Three samples per local mode. The official Flash column is the preserved
one-request reference and was not called again. Counting is outside the weighted
score. Every local response completed; code, math, and JSON objective checks
passed, while open prose was intentionally left unscored.

| Case | 1 RTX target | 1 RTX dSpark | 2 RTX target | 2 RTX dSpark | Official Flash |
|---|---:|---:|---:|---:|---:|
| Code | 44.90 | 127.62 | 48.10 | **149.07** | 345.90 |
| Math | 41.93 | 119.56 | 47.76 | **135.72** | 285.33 |
| Fable | 41.31 | 51.88 | 44.39 | **52.64** | 123.63 |
| Hello | 41.58 | 69.89 | 45.40 | **84.69** | 141.10 |
| Topic | 42.48 | 71.86 | 45.27 | **75.28** | 169.24 |
| Natural JSON | 44.04 | 102.23 | 47.11 | **121.76** | 175.33 |
| Schema JSON | 43.65 | 104.65 | 46.53 | **107.63** | HTTP 400 |
| Multilingual | 42.69 | 66.27 | 46.07 | **66.31** | 183.61 |
| Counting 1–200 | 44.95 | 156.08 | 48.41 | **181.31** | 427.29 |

The median weighted target-to-dSpark gain is 78.3% on one RTX and 72.0% on two
RTX cards. Two RTX improves target-only weighted throughput by 7.2% and dSpark
weighted throughput by 3.4% on this prompt set.

## One-RTX prefill matrix

Median effective prefill tokens per second after one shape warmup. Each measured
request uses a unique marker; retained rows verify the exact parent cache hit.

| Retained base | +1K | +2K | +4K | +8K | +16K | +32K |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3,084 | 4,078 | 6,709 | 7,064 | 7,726 | 7,878 |
| 32K | 2,513 | 3,442 | 5,261 | 6,073 | 6,619 | 6,894 |
| 64K | 2,326 | 3,214 | 4,845 | 5,590 | 6,116 | 6,322 |
| 128K | 2,007 | 2,813 | 4,129 | 4,842 | 5,308 | 5,514 |
| 256K | 1,507 | 2,192 | 3,111 | 3,711 | 4,108 | 4,284 |

## Two-RTX prefill matrix

The same prompts and token shapes with all twenty encoder expert layers hosted
TP2 on the RTX pair.

| Retained base | +1K | +2K | +4K | +8K | +16K | +32K |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4,850 | 6,471 | 7,982 | 8,331 | 8,454 | 8,348 |
| 32K | 3,736 | 5,068 | 5,633 | 6,434 | 6,937 | 7,101 |
| 64K | 3,318 | 4,543 | 5,066 | 5,832 | 6,316 | 6,459 |
| 128K | 2,748 | 3,790 | 4,258 | 4,949 | 5,329 | 5,483 |
| 256K | 1,923 | 2,731 | 3,213 | 3,729 | 4,052 | 4,198 |

Two RTX gains 57–59% on cold +1K/+2K prefills and 18–19% on cold +4K/+8K.
At deep retained context with long suffixes, attention dominates; the +16K and
+32K cells are close, including small measured losses at 256K retained context.

## Decode over retained context

Three samples for each of eight content types at every base size. All 240
requests completed with verified exact retained-prefix reuse, and every assessed
objective passed.

| Retained base | 1 RTX weighted dSpark | 2 RTX weighted dSpark | 1 RTX completed/reused | 2 RTX completed/reused |
|---:|---:|---:|---:|---:|
| 0 | 76.75 | **93.58** | 24/24 | 24/24 |
| 32K | 71.97 | **85.82** | 24/24 | 24/24 |
| 64K | 70.73 | **86.82** | 24/24 | 24/24 |
| 128K | 68.32 | **84.20** | 24/24 | 24/24 |
| 256K | 67.39 | **78.08** | 24/24 | 24/24 |

## Concurrency scaling

Each cell is the median of three warm runs. Aggregate timing spans earliest
first content through final completion and includes admission gaps. Code and
topic are the useful serving curves; counting is the low-entropy ceiling.

| Concurrency | 1 RTX counting | 2 RTX counting | 1 RTX code | 2 RTX code | 1 RTX topic | 2 RTX topic |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 152.65 | **186.83** | 124.18 | **148.83** | 71.93 | **81.13** |
| 2 | 246.29 | **294.15** | 196.77 | **231.63** | 122.43 | **131.94** |
| 4 | 437.02 | **524.04** | 355.92 | **412.60** | 222.72 | **242.58** |
| 8 | 693.73 | **808.47** | 586.58 | **712.09** | 358.31 | **415.86** |
| 16 | 1,161.12 | **1,333.57** | 992.78 | **1,181.49** | 514.96 | **596.14** |

## Mixed traffic

The fixed code/fable/topic mix uses simultaneous admission and nonce seed 56001.
Ranges are retained because request order and output mix create visible noise.

| Concurrency | 1 RTX median (range) | 2 RTX median (range) |
|---:|---:|---:|
| 1 | 113.35 (109.40–128.59) | **151.26 (150.21–154.21)** |
| 2 | 80.83 (72.82–104.85) | **141.54 (123.44–143.40)** |
| 4 | 118.20 (104.83–152.12) | **181.74 (173.78–192.51)** |
| 8 | 124.14 (110.56–170.85) | **203.05 (195.39–216.16)** |
| 16 | 196.46 (195.53–197.69) | **309.06 (300.63–311.28)** |

## Memory, startup, and scope

| Resource | 1 RTX | 2 RTX |
|---|---:|---:|
| Loaded RTX memory | 95,080 MiB | 95,338 / 95,578 MiB |
| Free RTX memory after readiness | 2,171 MiB | 1,913 / 1,670 MiB |
| Runtime headroom policy | 2 GiB | 800 MiB per GPU |
| Global FP4 source pool | 16,681,077,760 bytes / 18,710,016 logical + 32,768 tail tokens | 13,094,420,480 bytes / 14,680,064 logical + 32,768 tail tokens |
| Standard restart to readiness | 58.05 s | 46.00 s |

The clean release build compiled and packaged both local and TP2 expert
interfaces, verified provenance and checksums, and distributed the rebuilt Spark
image to all four workers. The focused dual lifecycle check passed a 32K needle,
cold and exact-warm prompt reuse, a retained continuation, eight simultaneous
cancellations with eight surviving peers, and post-cancellation recovery. The
full qualification suite was intentionally not repeated.

The compact [machine-readable report](phase2-release-performance.json) records
exact medians, ranges, controls, artifact sizes, and SHA-256 hashes. Raw artifacts
are preserved in the [release evidence archive](evidence/phase2-release-performance.tar.gz)
(`279770583bee5cc7478b4107580415ec3d2751613b1aad281dd023fdc0b6e5ac`).
