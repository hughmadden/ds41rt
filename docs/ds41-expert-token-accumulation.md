# Direct FP32 expert token accumulation

The large-prefill expert kernel now writes FP32 token sums directly, avoiding the materialized slice and route output planes. At capacity 1024 those planes occupied 360 MiB and 120 MiB respectively; their replacement is a 20 MiB token output. Other planner scratch remains. FP4 resident weights, GPU weight packing, FP8 wire inputs, BF16 rank returns and RTX rank-ordered reduction retain their existing formats.

The selected kernel remains M16, N64 at capacity 1 and N192 at larger capacities. M32/M64 experiments passed component correctness but did not give a consistent performance win; their implementation was removed. The pipeline also bounds compute and reduction launches by live rows rather than maximum capacity, with a proven worst-case expert-group bound tested against concentrated and distributed routes.

## Arithmetic and ABI

Capacity 1/16/80 variants retain the ordered route reduction and ABI 2. Capacity 256/1024/4096 variants use ABI 3, explicitly advertising FP32 token output. The worker selects the matching BF16 cast at planning/execution. Older clients reject ABI 3; new clients remain compatible with old ABI 2 libraries. The export option is `--atomic-min-capacity 256`; CMake uses `-DDS41RT_V41_EXPERT_ATOMIC_MIN_CAPACITY=256` with `-DDS41RT_V41_EXPERT_SLICE_WIDTH=1:64,16:192,80:192,256:192,1024:192,4096:192`.

The large variants use GPU FP32 atomic additions. Their summation order is unspecified, with subnormal flushing in the atomic instruction. They can differ between identical replays and cross BF16 rounding boundaries. This is an intentional numerical change, not bit-exact equivalence. Small ordered variants remain exact in the qualification fixtures.

## Qualification

`python/tools/qualify_v41_token_accumulation.py` compares both native libraries using all 384 official experts for layer 0/rank 0 and layer 39/ranks 1–3 on ostrich. Eleven cases cover rows 1, 6, 80, 81, 256 and 1024, shared/mixed routes, worst-case group counts, zero activations, input rebinding, poisoned outputs, untouched tails and graph replay without allocator growth. Invalid token-cast arguments are rejected. Later-layer runs require `rtol=2e-6, atol=2e-5`; the initial layer-0 run used a looser gate. Maximum observed absolute error was 4.58e-5 and relative L2 error 9.21e-8. At most 0.00390% of BF16 elements changed in an individual case. These checks do not qualify all model layers or end-to-end logits.

RTX and Spark both passed the ten token-accumulation/route-plan component tests; an additional worst-group-bound fixture passed on RTX. The Spark native build passed both CTests, and production daemons built for x86_64 and aarch64. The live RoCE fixture passed exact replay, abandoned-dispatch and sink-failure recovery for rows 1/6/16/80/6/1 against all four new workers. Large atomic-output transport replay is covered by the native numerical fixture rather than claiming byte-exact live recovery at those sizes.

For 1024 mixed rows, late-layer native graph medians improved from 12.20–12.39 ms to 8.17–8.24 ms; shared routing improved from 6.02–6.07 ms to 3.23–3.33 ms. These include planning, expert computation and FP32 accumulation, but exclude the separate final BF16 compaction and network. Layer-0 timing ran alongside a CPU build and is retained as diagnostic evidence, not the primary timing claim. These are component timings, not API throughput or measured DRAM bandwidth.

All eight paired API quality cases preserve baseline text and token usage. The inherited Unicode format failure and 7/8 cross-mode agreement remain. Broad quality and numerical qualification remain open.

## Deployment

b12x master: `6e868d9793565dff41ebcd47b35f5326fcccad7b`. New worker/native artifacts are frozen under `/tmp/ds41-expert-direct` on all four Sparks. Worker SHA256: `5f25e73ae472b1a1f3edb83ec98196d3896a4acf443f316200fa69f2394fee81`; native SHA256: `ca4d7174e0b83eba9d1e06933c61e4a3290e6c48ed1bd4de909a97e275d03628`. Containers are `ds41-direct-worker` and `ds41-direct-{target,spec}-api-dev`. The RTX native library remains the prior split-mHC artifact, SHA256 `cf202558ee672d1c34a273e9c62d09b1ca8f6bad1367f1cae2c63ceedc5f3560`; coordinator daemon SHA256 `21125c64edccffe48f5f3e6f030518dd9ddf77beb82d0b3478c51198e62374a6`. Source pin and built native identity are recorded separately.

Exact startup arrays, raw measurements and immutable binaries are under `/tmp/ds41-expert-direct`; full Spark build sources/artifacts are under `/tmp/ds41-direct-source` and `/tmp/ds41-direct-artifacts` on ostrich. Prior pinned-prefill API and worker containers remain stopped for rollback.

[Component measurements](ds41-expert-token-accumulation.json).

## Live API measurement

Sequential C1 greedy requests use the same repeated `amber` filler and “Count from 1 to 20, separated by commas. Output only the numbers.” suffix as the pinned-prefill baseline. Each produced 59 completion tokens, with zero prefix-cache hits. Effective prefill is prompt tokens divided by time to first content, including API/tokenization/first-output overhead; decode is 58 intervals from first content to finish.

| Prompt tokens | Mode | First content (s) | Effective prefill (tokens/s) | Decode (tokens/s) |
|---:|---|---:|---:|---:|
| 3,977 | Target | 4.186 | 950.1 | 24.38 |
| 3,977 | dSpark | 4.100 | 970.0 | 92.14 |
| 16,411 | Target | 16.479 | 995.9 | 24.67 |
| 16,411 | dSpark | 16.246 | 1010.2 | 90.18 |

The preceding pinned-prefill 16k result was 18.726/18.692 seconds and 876.4/878.0 effective tokens/s. This rollout improves observed throughput by approximately 14–15%, with similar long-context decode. These are single sequential measurements, not confidence intervals or a representative prompt corpus. Large-context prefill throughput does not fall between these two sizes in this workload, but broader context/decode scaling remains unqualified. The next 2–3k prefill checkpoint and original 8k release goal remain open. Sparse indexing/attention and coordinator dispatch/collection are the next profile targets.

Both API smoke suites pass streaming, cancellation/recovery and unsupported-sampling rejection. Both context sizes in both modes preserve the baseline counting text and token usage.
