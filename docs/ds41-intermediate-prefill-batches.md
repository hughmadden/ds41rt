# Intermediate prefill batches and capacity-dependent decode regression

Fixed startup for intermediate `serve-native --prefill-batch-tokens` values such as 2048. The CLI previously accepted every value from 80 through 4096, but forwarded that value as an exact AOT capacity; 2048 failed with `native V4.1 FP8 info failed with CUDA status 1` after loading weights. Startup now reserves the smallest supported capacity in 80/256/1024/4096 and passes the original requested batch size to prompt chunking. Planning and allocations remain outside inference replay.

Both live APIs are restored to batch 1024 with the fixed daemon. Larger batches improve code prefill but regress decode, so they are not selected defaults. No GPU kernel, expert packing or precision change is deployed.

## API comparison

Identical frozen code context and repeated `amber` filler, with 16,410 and 16,411 actual prompt tokens respectively. Each arm makes two requests per kind; the table uses the second request. The first code request includes startup/warmup costs and remains in the JSON. These are sequential C1 status workloads, not a broad performance benchmark.

| Mode / batch | Code prefill tok/s | Repeated prefill tok/s | Decode after code tok/s |
|---|---:|---:|---:|
| Target / 1024 | 1235 | 1525 | 25.91 |
| Target / 4096 | 1337 | 1427 | 23.82 |
| Target / 2048 | 1333 | 1501 | 23.80 |
| dSpark / 1024 | 1243 | 1533 | 93.60 |
| dSpark / 2048 | 1337 | 1512 | 82.31 |

Target code TTFT improves from 13.287 to 12.313 seconds at batch 2048, and dSpark from 13.207 to 12.272 seconds. Repeated prefill regresses about 1–2%, while target decode regresses about 8%. dSpark decode after repeated text is 90.66 versus 94.66 tok/s, less severe than the code case but still adverse. All twelve candidate prefill outputs, prompt hashes and usage records match their corresponding controls. The larger-batch prefill gain does not justify treating this as an overall serving improvement.

The daemon is `/tmp/ds41-prefill-revisit/daemon`, SHA256 `5a165c59030685cf24db1ffc559ea686888bfc3ad995100ef8fcae34c615ab54`. RTX native remains `07db52dec294d18ee98e44b553a7c442a43cdabb65ff8103690572fc4eeaa42b`; Spark native remains `ca4d7174e0b83eba9d1e06933c61e4a3290e6c48ed1bd4de909a97e275d03628`. Current containers are `ds41-planned1024-target-api-dev` and `ds41-planned1024-spec-api-dev`, ports 18041/18042. Exact final launch arrays are in the companion JSON and `/tmp/ds41-prefill-revisit/api-create-commands.json`. The 2048/4096 and profiling containers are stopped.

## Qualification

- Capacity boundary/error unit test passes for 80, 81, 256, 257, 1024, 1025, 2048, 4096 and invalid bounds. The release daemon build succeeds.
- Both actual 2048 APIs start, execute four 16k requests each, and pass JSON/streaming, cancellation/recovery and unsupported-sampling lifecycle checks.
- Eight paired quality cases preserve both modes' text and usage against the retained-upload baseline. The inherited Unicode-format failure and 7/8 pair equality remain; the quality command returns 1, so this is not a claim of full quality qualification.
- Every Spark logs exactly 2,560 full 2048-row executions across those eight requests, each with `kernel_capacity=4096`. Layer and routing-statistic sequences agree across ranks. This verifies live chunking independently of the CLI and latency measurements.
- Both restored 1024 APIs also pass lifecycle checks. Their short counting runs measure 36.80–37.28 target and 114.73–116.60 dSpark tokens/s; no speedup is attributed to the startup-only fix.

Raw build/test logs, benchmarks, quality/lifecycle results, worker logs and profiles are under `/tmp/ds41-prefill-revisit`. The source change does not alter the CLI's default batch size. It makes accepted intermediate values usable and keeps the startup reservation distinct from the requested batch size.

## Where the extra decode time appears

The fresh 2048 profile and prior 1024 retained-upload profile each contain 4,640 one-row layer observations from two 16k code requests. Mean host-observed stage microseconds:

| Stage | Capacity 1024 | Capacity 4096, batch 2048 |
|---|---:|---:|
| Attention output projection | 79.4 | 119.8 |
| Layer begin/query preparation | 97.5 | 115.3 |
| Shared FFN submission/execution | 71.3 | 86.7 |
| Sparse attention | 351.8 | 352.0 |
| Complete target step | 39,917 | 42,627 |

These nested intervals are not additive. Sparse attention stays effectively unchanged while projection/query/shared paths grow. Source inspection shows the affected wave constructors retain an FP8 kernel selected from the full prefill capacity and reuse it for small live row counts. The next change should preplan appropriate small-row kernels and scratch at startup, then select among those existing plans during decode without compilation or allocation. The profile supports prioritizing that issue; a new implementation still needs GPU and API qualification.

The warm 2048 code-prefill profile totals 12.368 seconds: expert phases 7.812 seconds, including 5.952 seconds collection, and sparse attention 3.212 seconds. Matching the 320 full-row calls across all four ranks gives 3.499 seconds summed slowest-rank GPU kernel intervals and 3.925 seconds complete worker intervals, excluding the final 26-row tail and decode. Host timestamps across Sparks are not synchronized; pairing uses ordinal/layer/routing agreement. These measurements leave substantial coordinator/transport and attention work after the expert-kernel experiments.
