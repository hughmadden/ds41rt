# Skewed expert sharing and GB10 cache counters

The live code-context profile has median 268 active experts per 1024-row batch, a hottest expert at 485 rows, and 83.58% of routes in expert batches larger than sixteen rows. Uniform mixed routing does not represent that workload. The native expert qualifier now includes skewed 1024/4096-row cases and reports complete expert histograms and M16 group counts.

The generated fixture samples six distinct experts per token without replacement. Sixty-four hot experts receive nominal probability mass 0.84 with rank weights proportional to rank^-0.70; the remaining 208 experts share mass 0.16. Sampling without replacement changes the realized fractions. This approximates observed sharing, not the actual tokens, activations or routing from an API request. Existing shared, uniform, boundary and zero cases remain.

| Shape | Active experts | Hottest expert rows | Routes in batches >16 | M16 groups |
|---|---:|---:|---:|---:|
| Generated 1024 rows | 270 | 450 | 82.42% | 551 |
| Generated 4096 rows | 272 | 1932 | 97.65% | 1676 |

Both cases use official layer 39/rank 0/all 384 expert weights on ostrich and the deployed capacity-4096 native state, with M16/N192 and atomic token output. `--force-largest-capacity` matches the worker's large state even at 1024 live rows. Correctness comparisons retain rtol 2e-6 / atol 2e-5 against the prior ordered native path, with poisoned outputs, graph mutation/replay, finite/nonzero results, bounds checks and no replay allocation. All runs pass; maximum observed absolute difference is 1.53e-5 and relative L2 is below 8.4e-8. These remain bounded component comparisons, not full-model reference quality.

## Native graph timings

At 1024 skewed rows, the current atomic path initially measures 8.304 ms versus 12.040 ms ordered. A later unprofiled repeat measures 7.590 versus 11.170 ms. At 4096 rows it measures 15.922 versus 28.005 ms. Timings alternate arms, six samples of five replays each. Clocks/caches are uncontrolled; the initial spread is retained rather than silently discarded. The ordered arm is an older deployed implementation used as a numerical comparator, so this is not a newly achieved serving speedup. Full inference artifacts are unchanged.

## Hardware counters

Nsight Compute 2026.1.1 exposes no direct `dram__bytes_read`, `dram__bytes_write` or `dram__throughput` metrics on this GB10. Collection succeeds with available L2/system-memory counters, using a container with SYS_ADMIN capability. An NVTX range selects one qualified candidate replay. Cache flushing and clock control are disabled, so these are diagnostic warm-workload observations, not a formal peak-bandwidth benchmark.

| Fused compute kernel | 1024 rows | 4096 rows |
|---|---:|---:|
| Kernel duration | 7.461 ms | 14.837 ms |
| L2 fills from system memory | 1.402 GB | 1.897 GB |
| System-fill bytes / duration | 187.9 GB/s | 127.8 GB/s |
| L2 sector access volume | 3.455 GB | 11.329 GB |
| L2 sector access rate | 463.1 GB/s | 763.5 GB/s |
| Reported L2 hit rate | 59.43% | 83.22% |
| Reported SM throughput | 12.20% | 19.07% |
| Registers/thread | 162 | 162 |

Volumes use decimal GB and the [documented 32-byte sector size](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#quantities). System fills include all relevant kernel accesses, not just weights, and do not directly measure physical DRAM bandwidth or percentage of its peak. Tiny planner-kernel hit-rate counters include inconsistent ratios above 100% in this uncontrolled collection; the table selects only the long fused kernel and does not use those tiny-kernel ratios.

The larger batch has approximately 3.3 times the L2 access volume but only 1.35 times the system-fill volume and twice the fused-kernel time. This supports substantial cache reuse and argues against treating every expert group as a fresh external-memory read. It does not establish which internal pipeline is the sole bottleneck. Further optimization should assess repeated L2 traffic, register pressure and larger M tiles on the skewed workload. Hypothetically changing grouping alone from M16 to M32 would reduce group counts from 551 to 395 and 1676 to 875; that is a metadata calculation, not a measured kernel speedup. Any new tile still needs actual compiled resource, numerical, decode and API evidence.

## Reproduction and artifacts

Use the actual capacity-4096 candidate and ordered baseline libraries with `python/tools/qualify_v41_token_accumulation.py --layer 39 --rank 0 --capacity 4096 --force-largest-capacity --only-case 1024:skew` (or `4096:skew`), plus the required snapshot/library/output paths. `--nvtx-candidate` emits one range named `v41_1024_skew` or `v41_4096_skew` after qualification and ordinary graph timings.

Run Nsight with `--nvtx --nvtx-include v41_1024_skew/ --cache-control none --clock-control none` and the metrics listed in [the evidence record](ds41-expert-skew-counters.json). Export the report, then use `ncu --import REPORT --page raw --csv` and `scripts/summarize-ds41-expert-counters.py RAW.csv --output SUMMARY.json`. The parser checks units and finite nonnegative values; real exports and ns/us/ms/s conversion plus malformed-value rejection were exercised.

Raw reports, CSV exports, qualifier logs and hardware snapshot are under `/tmp/ds41-expert-skew` on ostrich; selected evidence is copied to the same directory locally. The native candidate SHA256 remains `ca4d7174e0b83eba9d1e06933c61e4a3290e6c48ed1bd4de909a97e275d03628` (b12x 6e868d97), and ordered baseline `872591810ab2d9f3ed1715c13d8207b3d715aabd7bf466bd33a661b64a5c29ad`. GPU UUID is `GPU-a7c503f0-bb16-4333-4f44-92e331dbcf2e`. Source hashes and raw timings are retained. This turn changes diagnostic tooling only; the qualified retained-response APIs and four expert workers remain running with their preceding binaries.
