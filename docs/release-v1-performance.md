# Release v1 performance report

DS41RT serves DeepSeek V4.1 Flash on one RTX PRO 6000 Blackwell workstation GPU plus four DGX Spark expert workers. The clean-built corrected-FP4 candidate reaches **2,668 prompt tok/s** at the best median prefill cell, **128.03 tok/s** single-request low-entropy dSpark decode, **60.68 tok/s** across the weighted eight-type dSpark corpus, and **683.70 aggregate tok/s at C16**. Its default C16 cache is 20.93 GiB and the warmed coordinator process uses 64.05 GiB.

All release measurements use an enforced **400 W RTX power limit** and **standard memory speed**: 14,001 MHz maximum, with 13,365 MHz observed under load. The RTX driver is 595.91.07. The four NVIDIA GB10 workers use driver 580.159.03. Target-only and dSpark arms ran sequentially across the shared workers. Sampling used temperature 0 and disabled thinking for throughput; the agentic tool evaluation used thinking enabled at high effort.

The exact clean-built artifacts are coordinator image `sha256:a13258e92dd25ddb889bd31bb77c8813c7881868b103ceec0c140e30e893c213` and Spark image `sha256:2f5d328a14f1a52a0d3b2356b041415d744635f3f3557e41403246429984093b`. They contain engine revision `d5fb015d00aa7b302e3ce179b097482da678db76`, model revision `dba1be0a40aa45a94ad051997016db3960a90277`, SparkInfer `7299b3b92e70d539b2c0a63aaadce36932ceef4d`, and XGrammar `557becfb64c503ae9c04344b0047661f43f44320`.

## Headline results

| Measurement | Result |
|---|---:|
| Best median prefill, 0 base + 32K new | 2,667.95 tok/s |
| Best observed prefill sample | 2,738.47 tok/s |
| Low-entropy dSpark decode, Orchid median | 128.03 tok/s |
| Weighted eight-type target-only median | 37.89 tok/s |
| Weighted eight-type dSpark median | 60.68 tok/s |
| dSpark gain on weighted mix | 60.16% |
| C16 aggregate warm decode median | 683.70 tok/s |
| Default architectural cache | 20.93 GiB |
| Warmed C16 coordinator process | 64.05 GiB |
| Clean standard-launch readiness | 55–56 s |

## Eight content types and low-entropy decode

Each arm ran five repetitions with unique one-token Unicode nonces to prevent unintended complete-prefix reuse. The eight-type weighted score gives the two structured JSON cases weight 0.5 and the other six cases weight 1.0, matching the preserved GLMRT corpus contract. Decode timing starts after first content. All 90 requests completed without server errors and reported zero cache hits.

| Case | Target tok/s | dSpark tok/s | Target quality | dSpark quality |
|---|---:|---:|---:|---:|
| Code | 38.35 | 99.57 | 5/5 | 5/5 |
| Math | 37.79 | 96.52 | 5/5 | 5/5 |
| Fable | 37.89 | 38.32 | 1/5 | 0/5 |
| Hello | 37.64 | 50.00 | 5/5 | 5/5 |
| Topic | 37.66 | 56.82 | 2/5 | 1/5 |
| Natural JSON | 38.05 | 81.74 | 5/5 | 5/5 |
| Schema JSON | 37.78 | 70.74 | 5/5 | 5/5 |
| Multilingual | 37.58 | 57.20 | 5/5 | 5/5 |
| Orchid, low entropy | 38.53 | **128.03** | 0/5 | 0/5 |

The target weighted repetitions were 37.82, 37.62, 37.89, 37.96, and 37.93 tok/s. dSpark produced 62.68, 60.68, 59.04, 61.91, and 59.12 tok/s. The strict quality contract passed 33/40 target and 31/40 dSpark weighted samples. Preserved misses are mostly exact word-count failures in the fable case and omission of a required literal in the topic case. Orchid is a throughput diagnostic and generated 99 words instead of the contract length, so its 0/5 quality result is retained rather than hidden.

## Prefill matrix

The target-only matrix measures only new prompt tokens above an exactly retained base. Each of the 30 cells has one excluded warmup and two timed samples. All 90 requests used exact requested token counts, every timed request had the expected cache frontier, and all used the corrected FP4 fingerprint.

Median effective prompt tok/s:

| Retained base | +1K | +2K | +4K | +8K | +16K | +32K |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1,472 | 1,815 | 2,490 | 2,233 | 2,573 | **2,668** |
| 32K | 1,384 | 1,719 | 2,440 | 2,595 | 2,545 | 2,654 |
| 64K | 1,308 | 1,649 | 2,388 | 2,568 | 2,539 | 2,647 |
| 128K | 1,209 | 1,535 | 2,256 | 2,507 | 2,594 | 2,598 |
| 256K | 989 | 1,346 | 2,001 | 1,902 | 2,326 | 2,264 |

The 0/+8K and 256K/+8K cells show wider two-sample spreads, preserved in the raw data. Headline values use cell medians; the maximum individual sample was 2,738.47 tok/s.

## Decode over retained context

The dSpark retained-context matrix uses two samples for every content type at each base. Every request proves the exact retained frontier through API cache counters before decode.

| Retained base | Weighted dSpark tok/s | Quality passes |
|---:|---:|---:|
| 0 | 59.59 | 13/16 |
| 32K | 53.75 | 15/16 |
| 64K | 55.06 | 15/16 |
| 128K | 54.04 | 12/16 |
| 256K | 51.87 | 13/16 |

At 256K, median case rates remain 86.66 code, 89.59 math, 34.92 fable, 56.55 hello, 49.08 topic, 65.03 natural JSON, 67.09 schema JSON, and 49.76 multilingual tok/s. The machine-readable report contains every cell, min/max range, cache count, output, and quality result.

## Concurrency scaling

A deterministic 599-token counting completion was warmed to a complete prompt hit, then run three times at each concurrency. Aggregate timing spans earliest first content through latest completion and therefore includes scheduler admission gaps.

| Concurrency | Median aggregate tok/s | Range | Scale vs C1 |
|---:|---:|---:|---:|
| 1 | 124.81 | 123.20–124.98 | 1.00× |
| 2 | 176.58 | 174.10–176.86 | 1.41× |
| 4 | 288.89 | 288.72–289.02 | 2.31× |
| 8 | 416.34 | 411.79–416.79 | 3.34× |
| 16 | **683.70** | 683.64–689.58 | **5.48×** |

Every stream returned the exact same 1–200 sequence, prompt/completion counts, and complete cache hit.

## Corrected FP4 cache, long context, and agentic quality

Matched four-pair comparisons against the preserved FP8 baseline show retained target decode changes from -0.18% to +0.37% and retained dSpark changes from -0.72% to +2.84% across 32K–256K. Cold first-content time improves 6.55% at 256K in both modes; the other cells remain within about ±2%. A direct dSpark counting followup improves 2.80%. The result meets the no-regression gate while reducing compressed source storage to FP4. Full methodology is in [the FP4 performance record](release-v1-fp4-performance.json).

Needle retrieval passes fresh and exact-reuse prompts at 32K, 128K, 512K, and 1.04M source tokens in both modes. At 1.04M, cold completion takes 322.57 s target-only and 323.55 s dSpark; exact reuse takes 5.19 s and 3.20 s. See [the needle report](release-v1-needle.md).

Three C16 tool-eval-bench hard campaigns run with thinking enabled, high effort, temperature 0, and eight extra turns. Scores are 119/34/153, 123/36/159, and 120/36/156 for basic/hard/total points. All 264 scenarios finish without serving failures. See [the tool-eval report](release-v1-tool-eval-final.md).

## Memory and startup

The default C16/24-turn cache reserves 22,471,251,456 bytes: 16.13 GB FP4 values, 2.02 GB E4M3 group-16 scales, 4.03 GB independent FP4 index keys, 0.25 GB index scales, 43.26 MB FP8 SWA, and small ownership metadata. The full warmed coordinator process uses 65,584 MiB. [Memory accounting](release-v1-memory.md) breaks down weights, execution arenas, vision, dSpark, retained tails, graph/runtime growth, and headroom.

A clean five-host build takes 374.44 s. The standard `run.sh` reaches port 8000 in 56.48 s on first launch and 55.15 s after restoration, with target weight loading taking about 1.77 s. See [the build/run report](release-v1-build-run.md). Phase-level startup I/O, transform, exchange, and graph accounting is reported separately.

Raw outputs, every timed sample, warmups, quality misses, launch logs, and hardware state are preserved in [`evidence/native-release-performance.tar.gz`](evidence/native-release-performance.tar.gz). The complete structured summary is [`release-v1-performance.json`](release-v1-performance.json).
