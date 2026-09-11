# Grouped slice compute: 192 favors the mixed decode fixture

b12x `8e46d1c` extends the fused 64/128/192 expert-slice kernel to multiple expert groups in one launch. Runtime metadata selects each expert, up to sixteen input rows and a disjoint output route span. Longer expert runs use multiple groups. Input views read the 5,280-byte FP8 K32 wire-row layout directly. Weight-pool offsets use 64-bit arithmetic.

Nonpositive group row counts skip inactive capacity slots. The qualification graph always launches capacity sixty-four, and the same graph replays across different live expert counts, input-row mappings and expert IDs. No live count enters a compile key. This is still a private compute path: fixture code prepares route metadata, and serving does not yet dispatch it.

## Correctness

Six tests pass on each of RTX SM120 and Spark SM121, covering the existing single-expert path and the new grouped path. Each width's grouped test uses one graph across one row, shared M2, mixed six-row routing, shared sixteen rows, shared eighty rows, and return to one row. The eighty-row run splits six expert runs into thirty M16 groups. Expert IDs, row order and routing weights change after capture; unused output capacity remains poisoned and replay allocations stay stable.

All eighteen grouped oracle comparisons per device pass. Maximum relative L2 is `2.705e-6` on RTX and `8.928e-8` on Spark. These are synthetic fixtures with a thirty-two-expert weight pool, not official checkpoint qualification or a proof of identical FP32 accumulation across widths.

## Spark diagnostic

Each comparison uses the same encoded inputs, weights and metadata across all three widths. Timing includes fused FC1/activation/FC2 and an ordered CuTe FP32 slice reduction. Route planning, input encoding, transport, compact rank return and coordinator execution are excluded. Staging remains synchronous.

| Workload | Active groups | Width 64 µs | Width 128 µs | Width 192 µs |
|---|---:|---:|---:|---:|
| One row, six experts | 6 | **110.13** | 142.79 | 126.98 |
| Two rows, six shared experts | 6 | **140.71** | 169.99 | 150.98 |
| Six rows, twenty-two experts | 22 | 543.91 | 508.40 | **456.77** |
| Sixteen rows, six shared experts | 6 | 284.95 | 255.75 | **225.68** |
| Eighty rows, six shared experts | 30 | 1103.14 | 783.97 | **452.32** |
| Return to one row | 6 | **108.02** | 165.21 | 141.78 |

These are medians of twelve samples per arm, each twenty warm graph replays. Ordering covers all six permutations twice. The mixed case has twelve M1, seven M2, two M3 and one M4 experts; its active-expert count matches the live six-row median, not the complete observed distribution. The eighty-row case deliberately stresses sharing and is not a large-prefill throughput qualification.

Width 192 wins the mixed decode fixture by about 10% over 128; width 64 wins the small low-expert-count cases. This supports workload-dependent compute tiles. It does not yet choose a resident layout: all variants retain the padded-640 weight allocation. It also does not establish a serving speedup: there is no same-fixture comparison with the deployed kernel, per-arm clock/throttle admission, cache-pressure test or DRAM-counter measurement. Asynchronous staging may change the ordering.

The reusable benchmark is `benchmarks/benchmark_v41_grouped_slices.py` in b12x. It checks the oracle and reduction before timing, and prints source hashes and raw samples. [Qualification logs, source hashes and timing samples](ds41-expert-grouped-slices.json) are retained here.

Next, compare the deployed path on the same fixture, qualify useful variants against official weights, then add native route preparation/dispatch and asynchronous staging. Include scratch and reduction costs throughout. Serving retains its previous kernels; the temporary worker/API shutdown has been restored.
