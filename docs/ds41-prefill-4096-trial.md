# 4096-token prefill batch trial

The 4096-token coordinator batch is rejected: it slows the repeated-text workload and provides no improvement on the frozen code context. Serving returns to 1024 tokens per prefill step. Both configurations use the same daemon/native artifacts and the same four capacity-4096 workers, so this comparison changes coordinator batching only.

## End-to-end measurements

Sequential C1 streaming API calls, greedy counting from 1 to 20 after context, 59 completion tokens in the 16k cases, no prefix-cache hits. Effective prefill is actual prompt tokens divided by time to first content; decode excludes that initial latency. These are bounded status measurements, not a representative corpus or clock-controlled statistical benchmark.

| 16k context | Mode | 1024 batch prefill tok/s | 4096 batch prefill tok/s | 1024 / 4096 TTFT, s | 1024 / 4096 decode tok/s |
|---|---|---:|---:|---|---|
| Repeated text | Target | 1287.6 | 1110.8 | 12.745 / 14.775 | 25.10 / 24.39 |
| Repeated text | dSpark | 1289.5 | 1121.2 | 12.727 / 14.637 | 93.49 / 90.88 |
| Code, warmed control | Target | 1065.0 | 1036.0 | 15.408 / 15.840 | 25.89 / 24.08 |
| Code, warmed control | dSpark | 1074.2 | 1051.8 | 15.276 / 15.602 | 94.06 / 81.34 |

Repeated text has 16,411 prompt tokens; code has 16,410. Initial code controls measured 1035.4/1068.1 prefill tok/s before the warm repeat. All four shorter, approximately 4k cases also slow with 4096 batching. All 16k benchmark text and usage are unchanged. The 4k code dSpark candidate omits spaces after commas, producing 40 completion tokens versus 59; its shorter decode measurement is therefore not directly comparable. Seven of eight benchmark outputs retain exact text/usage. The eight paired quality cases retain preceding text/usage, including the inherited strict Unicode formatting failure and 7/8 cross-mode agreement.

`scripts/bench-ds41-prefill-api.py` records tokenizer, context and prompt hashes and runs workloads sequentially. The frozen context combines sparse-attention CUDA, native serving Rust, and expert-execution Rust; exact source identities and results are in [the evidence record](ds41-prefill-4096-trial.json). Raw context, launch arrays and logs are under `/tmp/ds41-prefill4096`. Use the same frozen context for comparisons; regenerating it after source changes changes the workload.

## Component and transport qualification

`python/tools/qualify_v41_token_accumulation.py --capacity 4096` passes fifteen cases on official layer 39/rank 0/all 384 experts on ostrich. Activations and routes are generated. The existing FP32 gate remains rtol 2e-6 / atol 2e-5. At 4096 rows, shared/mixed/group-bound expert graph timings are 12.42/12.55/19.25 ms with atomic token output, versus 24.20/28.53/30.99 ms with ordered intermediate output. This establishes component behavior, not an end-to-end batching gain.

The separate ignored `local_qps_atomic_prefill_cancel_and_recover_live` test passes through all four actual RoCE workers at rows 1/6/16/80/256/1024/4096/6/1: 27 completed calls, nine abandoned dispatches and nine injected sink failures. Protocol identity and row/rank checks remain strict; outputs through 80 rows remain byte-exact. Larger atomic outputs allow at most 0.1% changed values, a per-value bound of 2e-5 plus max absolute value/128, and relative L2 at most 5e-4. This run observed zero changed values; atomic summation is not generally deterministic. A unit test checks that the gate rejects corruption, broad changes, NaNs and extent mismatches. The existing strict replay fixture is unchanged.

## Deployment and next work

Restored APIs: `ds41-attention-pad-{target,spec}-api-dev`, ports 18041/18042, batch 1024. Workers: `ds41-prefill4096-worker` on ostrich/dodo/emu/kiwi, capacity 4096. Capacity-1024 `ds41-direct-worker` containers remain stopped for rollback. Candidate 4096 APIs are stopped. Both restored API smoke suites pass JSON/streaming, cancellation/recovery and unsupported-temperature rejection.

Coordinator daemon SHA256: `6c49103accaaf91e3ca4096830e3d550baa4ef0356895ff2ea214c26c457bbba`; native: `07db52dec294d18ee98e44b553a7c442a43cdabb65ff8103690572fc4eeaa42b`, b12x e5343cfb. Spark daemon: `5f25e73ae472b1a1f3edb83ec98196d3896a4acf443f316200fa69f2394fee81`; native: `ca4d7174e0b83eba9d1e06933c61e4a3290e6c48ed1bd4de909a97e275d03628`, b12x 6e868d97. No production binary changes in this trial.

The nearer 2–3k prefill checkpoint remains open. Active parameter count and compact KV do not establish whole-pipeline throughput. The existing stage trace still points to expert/coordinator execution and sparse attention; a fresh trace is needed to attribute the larger-batch regression. Coordinator request encoding and retention also deserve inspection before further batch increases. Actual weight DRAM bandwidth, broader quality and concurrency/context scaling remain unqualified.
