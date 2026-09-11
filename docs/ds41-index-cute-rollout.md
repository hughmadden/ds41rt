# Tiled CuTe V4.1 index scoring

The coordinator now scores packed FP4 index queries and paged/proposed keys with BF16 tensor cores. The old scalar path launched one 256-thread block per query/candidate pair. The new CuTe kernel scores 16 candidates per warp, 64 per 128-thread block, reusing decoded inputs in registers. It retains the existing FP32 score buffer and deterministic top-k stage; no cache concatenation or new persistent scratch is introduced.

All live rows, candidate widths, page strides and capacities are runtime arguments. Pool and row offsets are Int64. The kernel validates committed pages, causal visibility, append-only proposal start/count/offset/step and physical capacity. Entirely invalid warps write negative infinity without loading query/key data. Native host pointer/overlap/dimension validation is unchanged. The native overlay entry point uses the exported scorer in coordinator FP8 AOT builds; the committed-history compatibility function and non-coordinator builds retain the scalar implementation. Rust prewarms the optional AOT initializer before capture; older libraries remain usable.

Dots round to BF16 before ReLU, weighted head products round to BF16, and the head sum uses the original reduction tree before BF16 rounding. Tensor-core dot accumulation changes the internal FP32 dot order, so mathematical equality beyond the measured input corpus is not assumed. The measured scores were exact against both the native oracle and pinned reference expressions.

## Qualification

- Seven b12x GPU tests pass: six shape cases with seven input/metadata mutations each, poisoned graph replay, fixed pointers, no replay allocator growth, stale/invalid pages and descriptors, causal masks, strided proposal tails and unsigned U64 causal bounds. The seventh case parks live keys beyond the signed-32-bit byte-offset boundary in a mostly uninitialized pool.
- The actual exported native library passes all eight cases in `scripts/qualify-ds41-index-overlay.py`, extracting the scoring expressions from the pinned reference model. Queries span 1–4096, candidates 3–16384, and committed capacity reaches 16,777,216 rows. These are generated packed inputs, not full-model logit comparisons. Proposal/committed score values, masks, mutations, graph replay and guards match exactly.
- The CMake build passes both native/CUDA self-tests. The production x86 daemon builds successfully.
- All eight paired API quality cases preserve preceding text and token usage. The inherited Unicode formatting failure and 7/8 cross-mode agreement remain. Both 4k and 16k counting runs preserve baseline text and token usage. This is bounded greedy quality evidence, not a broad model-quality benchmark.

The early prototype's invalid-page handling bug was fixed before export. The final kernel also preserves unsigned causal-bound semantics. Initial standalone tensor timings are diagnostic; the table below uses the actual exported native library and the durable component benchmark.

## Measurements

On RTX PRO 6000, the all-valid committed-cache synthetic benchmark measured:

| Queries × candidates | Previous scalar score | Exported CuTe score | Top-512 with new library |
|---|---:|---:|---:|
| 1024 × 512 | 2.082 ms | 0.0495 ms | 0.130 ms |
| 1024 × 4096 | 16.855 ms | 0.3775 ms | 0.358 ms |
| 1024 × 16384 | 68.923 ms | 1.5895 ms | 1.264 ms |

Each component number is a median of five samples of five graph replays with poisoned-output replay checks. Baseline and candidate were measured in separate runs on the same GPU, not a clock-controlled interleaved release benchmark. Scores and top-k are timed separately. These measurements include neither the complete index producer nor the API, and do not measure DRAM bandwidth.

The sequential C1 greedy API workload is repeated `amber` filler followed by counting 1–20, 59 completion tokens, zero prefix-cache hits. Effective prefill divides prompt tokens by time to first content, including tokenization/API/first-output overhead. Decode measures 58 intervals from first content to finish.

| Prompt tokens | Mode | First content | Effective prefill | Decode |
|---:|---|---:|---:|---:|
| 3977 | Target | 4.020 s | 989.4 tokens/s | 25.46 tokens/s |
| 3977 | dSpark | 3.922 s | 1014.0 tokens/s | 92.72 tokens/s |
| 16411 | Target | 13.066 s | 1256.0 tokens/s | 25.44 tokens/s |
| 16411 | dSpark | 12.922 s | 1270.0 tokens/s | 93.95 tokens/s |

The preceding 16k deployment measured 16.479/16.246 seconds and 995.9/1010.2 effective tokens/s. Observed 16k throughput improves about 26%; this is a single sequential status workload, not confidence intervals or a representative corpus. The next 2–3k checkpoint and original 8k prefill target remain open. Sparse attention and expert/coordinator overhead remain the largest targets from the prior stage trace; a fresh trace should determine the next implementation.

## Deployment and reproduction

b12x master: `e5343cfb6874fecb9fe012a8c56e79ffdfca831f` (new kernel in `3f76b539`, unsigned-bound fix in `e5343cfb`). Selected native library SHA256: `83a9cd22319518952b03004566326077255a8ff5a0e94ce94510407a7d5fb4fd`. Coordinator daemon SHA256: `6c49103accaaf91e3ca4096830e3d550baa4ef0356895ff2ea214c26c457bbba`.

Selected APIs are `ds41-index-cute-{target,spec}-api-dev` on ports 18041/18042, with capacity 1024 and context limit 32768. They mount the frozen `/tmp/ds41-index-aot/daemon` and `/tmp/ds41-index-aot/cmake`. The four `ds41-direct-worker` expert services retain their preceding artifacts and RoCE transport. Prior direct-accumulation APIs remain stopped for rollback.

The scorer export is included in `DS41RT_ENABLE_V41_FP8_AOT` and uses the pinned b12x source. `python/tools/export_b12x_v41_index_aot.py` rejects unexpected generated C signatures or dispatch argument order. Raw build logs, manifest, command arrays, qualification and measurements are in `/tmp/ds41-index-aot`; earlier standalone diagnostics are in `/tmp/ds41-index-profile`.

[Recorded component and API evidence](ds41-index-cute-rollout.json).

Both API smoke suites pass streaming, cancellation/recovery and unsupported-sampling rejection.
