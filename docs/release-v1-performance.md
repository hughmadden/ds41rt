# DS41RT v1 performance report

DS41RT serves DeepSeek V4.1 Flash on one RTX PRO 6000 Blackwell workstation GPU plus four DGX Spark expert workers. The corrected standard build reaches **7,743 prompt tok/s** at its best median prefill cell, **150.51 tok/s** warm counting dSpark decode, **70.43 tok/s** across the weighted eight-type dSpark corpus, and **742.91 aggregate tok/s at C16**. Its default C16 cache reserves **20.93 GiB for 25,165,824 tokens total** (24 × 1,048,576-token contexts: sixteen active plus eight spare), and the warmed coordinator process uses 64.23 GiB.

All measurements use an enforced **400 W RTX power limit** and **standard memory speed with no overclock**: 14,001 MHz maximum and 13,365 MHz observed under load. The RTX driver is 595.91.07. The four NVIDIA GB10 workers use driver 580.159.03. Target-only and dSpark arms ran sequentially across the shared workers. Throughput sampling used temperature 0 and disabled thinking; the agentic tool evaluation used thinking enabled at high effort.

The exact published artifacts are coordinator index `sha256:67f2954e18f69b39f8fbb68164f7d9e2b8f4c4b9e3242281ecfcb8afec8552e9` and Spark image `sha256:1f1bff295a1d112c8a2fb80918b5abcafcf10eec4936717e234d3d727635a0be`. They were built from revision `9ea5c96468da690fe7dd01471d4fa2fb8555a606` with model revision `dba1be0a40aa45a94ad051997016db3960a90277`, SparkInfer `7299b3b92e70d539b2c0a63aaadce36932ceef4d`, and XGrammar `557becfb64c503ae9c04344b0047661f43f44320`.

The earlier 2.6K prefill regression came from the standard Spark AOT export omitting the qualified fused-slice width map and direct token-accumulation defaults. The corrected standard exporter selects width 64 at capacity 1, width 192 at capacities 16–4,096, and direct token accumulation from capacity 256. Seventeen independent numerical comparisons pass with maximum absolute difference `2.288818359375e-05`. The full serving rerun below restores and exceeds the earlier 6.5K–7.5K prefill and roughly 145 tok/s counting measurements.

## Headline results

| Measurement | Result |
|---|---:|
| Best median prefill, 0 base + 32K new | **7,743.47 tok/s** |
| Best observed prefill sample | **8,023.26 tok/s** |
| Low-entropy target-only decode, counting 1–200 warm median | 43.08 tok/s |
| Low-entropy dSpark decode, counting 1–200 warm median | **150.51 tok/s** |
| Weighted eight-type target-only median | 41.42 tok/s |
| Weighted eight-type dSpark median | **70.43 tok/s** |
| dSpark gain on weighted mix | 70.04% |
| C16 aggregate warm decode median | **742.91 tok/s** |
| Default architectural cache | 20.93 GiB for 25,165,824 tokens total (24 × 1,048,576) |
| Exact prompt / completed-turn retention | 24 / 24 entries |
| Warmed C16 coordinator process | 64.23 GiB |
| Clean standard dSpark launch readiness | 56.77 s |

## Eight content types and low-entropy decode

Local values are medians of three requests per mode. Completion columns count finished serving requests. Named arithmetic, Python-structure, and JSON objectives are recorded separately; fable, greeting, explanation, and multilingual prose are not reduced to brittle word-count or keyword scores.

| Case | Target tok/s | dSpark tok/s | Official Flash tok/s (one request) | Target completed | dSpark completed | Official completed |
|---|---:|---:|---:|---:|---:|---:|
| Code | 42.62 | 113.73 | 345.90 | 3/3 | 3/3 | 1/1 |
| Math | 41.08 | 109.54 | 285.33 | 3/3 | 3/3 | 1/1 |
| Fable | 40.42 | 47.10 | 123.63 | 3/3 | 3/3 | 1/1 |
| Hello | 39.96 | 57.14 | 141.10 | 3/3 | 3/3 | 1/1 |
| Topic | 41.24 | 63.48 | 169.24 | 3/3 | 3/3 | 1/1 |
| Natural JSON | 42.34 | 93.24 | 175.33 | 3/3 | 3/3 | 1/1 |
| Schema JSON | 41.89 | 85.78 | HTTP 400 | 3/3 | 3/3 | 0/1 (HTTP 400) |
| Multilingual | 40.99 | 62.70 | 183.61 | 3/3 | 3/3 | 1/1 |
| Counting 1–200 | **43.08** | **150.51** | **427.29** | 3/3 warm | 3/3 warm | 1/1 |

The local eight-case runs use unique one-token Unicode nonces to prevent unintended complete-prefix reuse. Their weighted repetitions are 40.23, 41.42, and 41.65 tok/s target-only and 70.43, 68.94, and 71.81 tok/s with dSpark. All 48 requests completed.

The official `deepseek-flash` column remains the requested one-time sequential API reference. It used the prior fable wording, temperature zero, disabled thinking, provider-reported token counts, and client timing after first content. Network streaming and unknown provider hardware make it a reference rather than a controlled hardware comparison. JSON Schema returned HTTP 400 and was not retried or downgraded, so there is no complete official weighted aggregate. [Official raw reference](release-official-flash-reference.json) preserves requests, outputs, timings, fingerprint, and failure details.

Counting is outside the eight-case weighted score. Each local value is the median of three warm requests after one prime; all six warm requests reuse 21 prompt tokens, produce 599 completion tokens, and pass the exact ordered 1–200 sequence check. The official value is one fresh request with zero cache hits and 598 provider-reported completion tokens. [Official counting evidence](release-official-counting-reference.json) retains that request.

## Prefill matrix

The target-only matrix measures new prompt tokens above an exactly retained base. Each of the 30 cells has one excluded warmup and three timed samples. All 120 requests used exact requested token counts and every timed retained request reported the expected cache frontier.

| Retained base | +1K | +2K | +4K | +8K | +16K | +32K |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2,818 | 3,757 | 6,922 | 7,414 | 7,660 | **7,743** |
| 32K | 2,440 | 3,319 | 6,055 | 6,736 | 7,086 | 7,119 |
| 64K | 2,251 | 3,112 | 5,557 | 6,255 | 6,594 | 6,768 |
| 128K | 1,942 | 2,724 | 4,724 | 5,432 | 5,785 | 5,961 |
| 256K | 1,477 | 2,148 | 3,565 | 4,123 | 4,444 | 4,612 |

The best cell's three measured samples are 7,743.47, 8,023.26, and 7,641.34 tok/s. Headline values use cell medians; no sample is omitted.

## Decode over retained context

The dSpark matrix uses three samples for every content type at each base. Every request completed and proves its retained frontier through API cache counters.

| Retained base | Weighted dSpark tok/s | Completed with verified cache reuse |
|---:|---:|---:|
| 0 | 73.43 | 24/24 |
| 32K | 68.08 | 24/24 |
| 64K | 64.82 | 24/24 |
| 128K | 62.71 | 24/24 |
| 256K | 61.27 | 24/24 |

At 256K, median case rates are 94.99 code, 87.46 math, 40.35 fable, 56.33 hello, 58.64 topic, 72.02 natural JSON, 70.53 schema JSON, and 59.67 multilingual tok/s. These changes with context length reflect the full retained-attention workload; they are not quality scores.

## Concurrency scaling

A deterministic 599-token counting completion was warmed, then run three times at each concurrency. Aggregate timing spans earliest first content through latest completion and includes scheduler admission gaps. Every stream returned the exact same 1–200 sequence, prompt/completion counts, and complete cache hit.

| Concurrency | Median aggregate tok/s | Range | Scale vs C1 |
|---:|---:|---:|---:|
| 1 | 149.16 | 148.89–149.24 | 1.00× |
| 2 | 188.47 | 187.82–188.66 | 1.26× |
| 4 | 325.90 | 323.81–326.08 | 2.18× |
| 8 | 467.10 | 463.93–467.14 | 3.13× |
| 16 | **742.91** | 735.48–745.69 | **4.98×** |

## Corrected FP4 cache, long context, and agentic quality

Matched four-pair comparisons against the preserved FP8 baseline show retained target decode changes from -0.18% to +0.37% and retained dSpark changes from -0.72% to +2.84% across 32K–256K. Cold first-content time improves 6.55% at 256K in both modes; the other cells remain within about ±2%. A direct dSpark counting followup improves 2.80%. The result meets the no-regression gate while reducing compressed source storage to FP4. Full methodology is in [the FP4 performance record](release-v1-fp4-performance.json).

Needle retrieval passes fresh and exact-reuse prompts at 32K, 128K, 512K, and 1.04M source tokens in both modes. At 1.04M, cold completion takes 322.57 s target-only and 323.55 s dSpark; exact reuse takes 5.19 s and 3.20 s. See [the needle report](release-v1-needle.md).

Three C16 tool-eval-bench hard campaigns run with thinking enabled, high effort, temperature 0, and eight extra turns. Scores are 119/34/153, 123/36/159, and 120/36/156 for basic/hard/total points. All 264 scenarios finish without serving failures. See [the tool-eval report](release-v1-tool-eval-final.md).

## Agentic artifact generation

DeepSeek Harness 0.1.5-rc.1 used this release service with thinking enabled at high effort to create a self-contained WebGL Frogger game. The passing run took 433.17 seconds, 15 model steps, and 16 tool calls. The agent repaired two moving-platform defects and passed its 20/20 logic simulation; independent checks passed 20/20 plus JavaScript parsing, and Chrome rendered the file through WebGL without errors. [Play Frogger](https://tpurtell.github.io/ds41rt/frogger.html) or read the [complete execution report](release-v1-frogger.md).

## Memory and startup

The default planner reserves source capacity for sixteen active maximum-context requests plus eight spare equivalents: 25,165,824 tokens total. The resulting C16/24-turn cache reserves 22,471,251,456 bytes: 16.13 GB FP4 values, 2.02 GB E4M3 group-16 scales, 4.03 GB independent FP4 index keys, 0.25 GB index scales, 43.26 MB FP8 SWA, and small ownership metadata. Prompt snapshots and completed turns each retain up to 24 entries. The warmed coordinator process uses 65,772 MiB. [Memory accounting](release-v1-memory.md) breaks down weights, execution arenas, vision, dSpark, retained tails, graph/runtime growth, and headroom.

The clean standard `build.sh` completed in 268.82 s, including native Spark construction and distribution to all four workers. Target-only `run.sh` reached port 8000 in 53.88 s; the final standard dSpark launch reached it in 56.77 s. See the [startup](release-v1-startup.md) and [build/run](release-v1-build-run.md) reports.

Raw local outputs, every timed sample and warmup, exact prompts, cache counters, launch/build logs, hardware state, kernel numerical results, lifecycle checks, and registry manifests are preserved in [`evidence/native-release-performance.tar.gz`](evidence/native-release-performance.tar.gz). The complete structured summary is [`release-v1-performance.json`](release-v1-performance.json).
