# DS41RT v2 performance report

All RTX measurements use a 400 W power limit and standard 14,001 MHz maximum memory speed with no memory overclock. The standard topology is one RTX PRO 6000 Blackwell coordinator and four DGX Spark workers. Throughput uses temperature zero and thinking disabled; tool evaluation uses thinking enabled at high effort and C16.

## Headline results

| Measurement | Result |
|---|---:|
| Best median prefill, 0 base + 32K new (preserved v1) | **7,743.47 tok/s** |
| Low-entropy target-only decode, counting 1–200 warm median | 45.08 tok/s |
| Low-entropy dSpark decode, counting 1–200 warm median | **155.91 tok/s** |
| Weighted eight-type target-only median | 44.29 tok/s |
| Weighted eight-type dSpark median | **79.80 tok/s** |
| dSpark gain on weighted mix | 80.17% |
| C16 aggregate warm decode median | **934.05 tok/s** |
| Standard RTX routed-expert placement | **5 layers (0–4)** |
| Spark expert residency / configured budget | 40 layers per worker / 100 GiB |
| Global FP4 source pool | 16.681 GB for 18,710,016 logical tokens (+32,768 private-tail tokens) |
| Exact prompt / completed-turn retention | 24 / 24 entries |
| Peak observed coordinator GPU memory used | 96,950 MiB |
| Clean build / standard dSpark launch | 304.85 s / 57.76 s |

## Eight content types and counting

Three local samples per mode. The official Flash column is the preserved one-request v1 reference and was not called again. Counting is outside the weighted score.

| Case | Target tok/s | dSpark tok/s | Official Flash tok/s | Target completed | dSpark completed | Official completed |
|---|---:|---:|---:|---:|---:|---:|
| Code | 44.87 | 126.21 | 345.90 | 3/3 | 3/3 | 1/1 |
| Math | 44.61 | 135.07 | 285.33 | 3/3 | 3/3 | 1/1 |
| Fable | 43.90 | 54.31 | 123.63 | 3/3 | 3/3 | 1/1 |
| Hello | 42.48 | 77.55 | 141.10 | 3/3 | 3/3 | 1/1 |
| Topic | 44.93 | 72.06 | 169.24 | 3/3 | 3/3 | 1/1 |
| Natural JSON | 44.30 | 92.42 | 175.33 | 3/3 | 3/3 | 1/1 |
| Schema JSON | 44.07 | 94.09 | HTTP 400 | 3/3 | 3/3 | 0/1 (HTTP 400) |
| Multilingual | 43.69 | 71.33 | 183.61 | 3/3 | 3/3 | 1/1 |
| Counting 1–200 | **45.08** | **155.91** | **427.29** | 3/3 warm | 3/3 warm | 1/1 |

## Prefill matrix

Preserved v1 target-only measurements; the prefill matrix was intentionally excluded from the scoped v2 rerun.

| Retained base | +1K | +2K | +4K | +8K | +16K | +32K |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2,818 | 3,757 | 6,922 | 7,414 | 7,660 | 7,743 |
| 32K | 2,440 | 3,319 | 6,055 | 6,736 | 7,086 | 7,119 |
| 64K | 2,251 | 3,112 | 5,557 | 6,255 | 6,594 | 6,768 |
| 128K | 1,942 | 2,724 | 4,724 | 5,432 | 5,785 | 5,961 |
| 256K | 1,477 | 2,148 | 3,565 | 4,123 | 4,444 | 4,612 |

## Decode over retained context

Three samples per each of eight content types, with verified exact retained-prefix reuse.

| Retained base | Weighted dSpark tok/s | Completed with verified cache reuse |
|---:|---:|---:|
| 0 | 77.97 | 24/24 |
| 32K | 71.60 | 24/24 |
| 64K | 71.97 | 24/24 |
| 128K | 70.99 | 24/24 |
| 256K | 66.82 | 24/24 |

## Concurrency scaling

Three exact 599-token counting samples per concurrency after one fully cached prime; aggregate timing includes admission gaps.

| Concurrency | Median aggregate tok/s | Range | Scale vs C1 |
|---:|---:|---:|---:|
| 1 | 151.12 | 150.21–151.73 | 1.00× |
| 2 | 238.85 | 230.28–239.94 | 1.58× |
| 4 | 393.38 | 382.04–420.01 | 2.60× |
| 8 | 582.56 | 577.00–586.01 | 3.85× |
| 16 | 934.05 | 932.59–936.37 | 6.18× |

## High-thinking tool evaluation

Three hard-mode campaigns use C16, thinking enabled, high reasoning effort, temperature zero, a 900-second timeout, and the normal output policy.

| Run | Basic | Hard | Total | Pass / partial / fail |
|---:|---:|---:|---:|---:|
| 1 | 122/138 | 33/38 | 155/176 | 71 / 13 / 4 |
| 2 | 118/138 | 35/38 | 153/176 | 69 / 15 / 4 |
| 3 | 122/138 | 36/38 | 158/176 | 73 / 12 / 3 |

## Memory, startup, and scope

The clean build completed in 304.85 seconds. The final standard dSpark launch reached port 8000 in 57.76 seconds. Automatic placement loaded layers 0–4 on the RTX; every Spark retained all 40 TP expert layers under its configured 100 GiB device budget. Peak observed coordinator GPU memory use was 96,950 MiB.

V2 reran only the tables present in the main README, excluding prefill, plus the requested tool campaign. The v1 prefill matrix and one-shot official API comparison remain clearly labeled prior measurements. The full needle, vision, cache, and agentic artifact suites were not repeated; their v1 evidence remains applicable to unchanged interfaces but is not represented as fresh v2 qualification.

Machine-readable results are in [release-v2-performance.json](release-v2-performance.json). The [release evidence archive](https://github.com/tpurtell/ds41rt/releases/download/v2/ds41rt-v2-qualification-evidence.tar.gz) preserves raw requests, responses, traces, build and launch logs, memory samples, image labels, publication manifests, and hashes.
