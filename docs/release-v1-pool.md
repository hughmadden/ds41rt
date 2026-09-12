# Native pool sizing and concurrency

`serve-native` defaults to sixteen active requests and compressed-source space
for another eight configured-context equivalents. At the default 1,048,576-token
context, the four pools reserve 24 full contexts plus 32 spare page groups for
partial tails and copy-on-write. This is a shared pool, not 24 independent copies
of each request. Shared prefix pages retain references until their owners expire.

Each physical row contains 64 index bytes, four index-scale bytes, 512 KV bytes
and sixteen KV-scale bytes. A page contains 256 rows (152,576 bytes). The source
layers compress at ratios 2/2/2/1, so one allocation group contains pages in a
1/1/1/2 ratio and occupies 762,880 bytes.

| Default allocation | Bytes |
| --- | ---: |
| 24 full-context equivalents | 37,497,077,760 |
| 32 spare page groups | 24,412,160 |
| Global index/KV pool | 37,521,489,920 |
| Entire active backbone cache, including windows, carry and page tables | 37,565,994,496 |
| Additional runtime headroom | 2,147,483,648 |

The global pool is approximately 34.95 GiB. Default physical page counts are
49,184 / 49,184 / 49,184 / 98,368. This exceeds the former 65,536-page cap on the
ratio-one source. Rust and CUDA producers/consumers now accept up to 262,144
physical pages (67,108,864 rows) per source. Per-request logical context remains
bounded to 1,048,576 tokens.

## Controls

- `--concurrency N` accepts 1 through 16 and bounds active admission across both
  execution lanes. The HTTP queue has the same capacity. Backbone slot storage
  and draft workspaces follow the configured concurrency.
- `--kv-pool-size SIZE` selects an explicit global index/KV byte budget. Accepted
  units are B, MB, GB, MiB and GiB; a number without a suffix means bytes. The
  engine rounds down to whole allocation groups and logs the actual byte count.
- `--memory-reservation SIZE` accepts those byte units or a percentage such as
  `90%`. It sets a total-device occupancy ceiling. After loading both target
  lanes, transport and optional draft allocations, the engine sizes the cache
  from the remaining budget, keeping 2 GiB for future runtime allocations.
- With both options, the explicit global pool is used and the total-device
  ceiling is checked. With neither option, the context-based default is used.
- `--max-context-tokens` controls each request's maximum and scales the default
  pool. Explicit pool budgets may be smaller than the default; they must still
  cover at least one page per active owner and the spare groups. Such budgets
  reduce aggregate capacity and may exhaust under long concurrent requests.

The percentage applies to total device memory, including memory already occupied
by other processes. It is a startup sizing policy, not an enforced CUDA process
quota. Cache sizes exceeding available memory or the selected ceiling fail
startup; the required default is not silently reduced. Reported cache bytes
include active SWAs and page tables; the explicit global pool option does not.

For example, a smaller development server can use:

```text
serve-native ... --concurrency 2 --max-context-tokens 32768 \
  --kv-pool-size 1GiB --memory-reservation 95%
```

Standard `run.sh` parameter wiring remains a release gate. The existing retained
snapshot accounting also needs the user's new 24-turn default: prompt and
completed-turn frontiers must not silently count as two separate turns. The
current C2 qualification below uses the preceding two-snapshot retention policy.

## Verification

The high-address GPU test allocates the expanded source capacity and writes the
highest physical page through both native index and KV scatter kernels. It checks
row 67,108,863, whose KV byte offset is 34,359,737,856. Both direct and overlay
index scorers read it correctly, sparse attention matches an independent numeric
expectation, and a divergent retained prefix copies all four planes to a private
low page without changing the high-page original. Compute Sanitizer reports zero
memory errors. The existing all-cache commit/rollback test also passes.

Pure sizing tests cover default capacity, smaller contexts/concurrency, explicit
byte budgets, decimal and binary units, percentages, rounding, overflow and
insufficient memory. CLI tests cover valid controls and invalid concurrency.

Live target serving with an explicit 1 GiB pool passes exact and partial reuse,
C2/C6/C16 counting, cancellation and replacement. Live dSpark at C2 passes the
same applicable checks. The original partial-prefix test expected the older
parent's full snapshot to survive; C2 correctly evicted it and reused a shorter
prefix instead. The original assertion failure is preserved. A follow-up with
an explicit two-entry expectation passes and verifies the same answer.

The full default allocation boots and serves an arithmetic request in both
target-only and dSpark modes on GPU 0. An 80 GiB total-device budget also boots
and serves in target-only mode. The latter allocates 48,408,550,400 global bytes;
its occupied-before + cache + runtime-headroom sum remains below the ceiling,
with less than one allocation group's space left. These are allocation and
short-inference checks, not a substitute for the 1M needle or retained-context
release benchmarks.

Four alternating warm A/B pairs per workload compare against the preceding
artifact under the same current driver/power setting. Text and token counts
match. Median decode changes are +0.24% counting / +0.54% code for target-only,
and +0.10% / +0.25% for dSpark. This focused comparison shows no regression;
it does not establish a throughput improvement or replace the final suite.
First-use prefill samples are preserved separately and are slower on the newly
built native artifact; final startup/warmup and prefill qualification remain open.

The RTX cards report driver 595.91.07 and an enforced 400 W power limit. Memory
uses standard speed, without a memory overclock; a loaded sample reports
13,365 MHz while the driver reports a 14,001 MHz maximum. Raw telemetry accompanies
the evidence. Historical results before the user's driver/power-cap reset are
not a controlled comparison to this campaign.

[Evidence metadata](release-v1-pool.json) identifies source, binaries, launch
commands, raw logs, complete API responses and sampled hardware state.
