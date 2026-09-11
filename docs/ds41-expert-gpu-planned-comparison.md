# Official expert comparison including GPU planning

The candidate now captures stable GPU expert grouping, fused expert-slice compute and inverse-map FP32 slice reduction in one CUDA graph. CPU metadata is used only as a correctness oracle. The deployed native library retains its own internal planner. Both arms exclude input encoding, compact BF16 return, transport and coordinator execution.

b12x `72a476b` supplies the planner and ordered inverse-map reducer. Seven GPU tests pass on each of RTX SM120 and Spark SM121. Planner cases cover capacities 1/16/80/4096 and changed live counts, duplicate/invalid routes, empty batches and stale capacity. Reducer cases cover all three widths, invalid routes producing zero, changed mappings/data and untouched unused output capacity. These are replay tests using a device live-row scalar, fixed allocations and the same compiled callable/graph within each specialization.

The official comparison uses the same rank-zero layers 0/1, two captured inputs and actual deployed native library as the [earlier compute-only comparison](ds41-expert-official-slice-comparison.md). All 72 candidate route/compact checks pass, with width 128 exactly matching native FP32 output. GPU metadata and inverse maps also match their CPU oracle exactly. The 16 earlier Rust-owner baseline comparisons remain exact. The tiny width 64/192 differences do not close the separate whole-model compact-return quality gate.

Timing uses all 24 arm permutations, ten graph replays per sample, after correctness checks. Raw samples, numerical metrics and source/native/input hashes are retained in [the evidence file](ds41-expert-gpu-planned-comparison.json). These remain diagnostic measurements without per-arm clock/throttle admission or DRAM counters, and establish no API throughput improvement. Serving binaries remain unchanged.

Median ranges across both layers and input sets, in microseconds:

| Request rows | Native | Width 64 | Width 128 | Width 192 |
|---|---:|---:|---:|---:|
| 1 | 204–209 | **149–153** | 178–187 | 168–171 |
| 2 | 448–449 | **253–292** | 277–316 | 259–294 |
| 6 | 678–903 | 583–631 | 574–643 | **518–576** |
| 16 | 1451–1726 | 1209–1363 | 1234–1484 | **1094–1260** |
| 80 | 4356–4545 | 3733–4016 | 3709–3968 | **3173–3435** |

The width ordering survives inclusion of GPU planning on this corpus. Request rows are not expert-local M: the six-row cases contain 24/27 active expert groups, and the eighty-row cases contain 145/156 groups. Maximum route relative L2 is 1.113e-8 for width 64 and 9.235e-9 for width 192; compact BF16 mismatches total two and one respectively across 2,170,880 compared elements per width, including repeated one-row checks.

Next: export and bind this path natively, qualify every TP rank and paired live model behavior, and profile staging and planner costs. Select around speculative-decode expert-local M and active-expert counts; prefill remains a separate throughput gate. The simple three-launch planner and its experts × route-capacity scratch are a starting implementation, not a tuned prefill planner.
