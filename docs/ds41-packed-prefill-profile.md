# Prefill profile after packed KV conversion

This historical profile predates the [bounded CED rollout](ds41-ced-serving.md).

The then-current packed-conversion native runtime and router daemon were profiled on
the target API with two sequential 16,410-token code prompts, 59 generated tokens,
2048-token chunks and four unchanged TP4 RoCE workers. The temporary instrumented
API was stopped and the selected target API restored; the speculative API was
unchanged. This uses host stage timers and transport logging, so it is diagnostic
and does not replace the uninstrumented throughput results.

Second-request totals over nine prefill chunks (eight of 2048, one of 26):

| Stage | Summed time |
|---|---:|
| Whole prefill steps | 10.376 s |
| Expert response wait/handling | 4.883 s |
| Sparse attention | 2.949 s |
| Expert dispatch/enqueue | 0.615 s |
| Router/request preparation | 0.237 s |
| Return reduction/final stream drain | 0.210 s |
| Return upload enqueue | 0.004 s |
| Shared FFN | 0.144 s |

These are nested host intervals: shared FFN overlaps remote execution, response
wait includes remote compute and client handling, and dispatch is not NIC send
completion. Do not add this table to infer independent kernel time or wire latency.
Full stage totals are in [the evidence JSON](ds41-packed-prefill-profile.json).
Raw logs, exact instrumented container creation and timings remain under
`/tmp/ds41-packed-profile`. Frozen runtime identity is recorded in the
[packed-conversion report](ds41-attention-packed.json).

The router is now a smaller contributor; sparse attention remains almost three
seconds and remote expert response collection almost five. The next attention
candidate should address within-query overhead before assuming adjacent-query
sharing is necessary. In particular, online-softmax accumulator rescaling currently
stores and reloads a 16-by-512 FP32 plane through shared memory between key tiles.
A multiplier fragment loaded through the same documented WMMA layout could rescale
accumulator elements without depending on undocumented lane ownership. This is a
candidate for qualification, not an implemented optimization or measured speedup.

## Expert batch size and reuse interpretation

Each Spark receives all 2048 token rows per full layer request. TP4 divides expert
matrix dimensions, not the token batch. Top-6 routing produces 12,288 assignments;
uniform routing over 384 experts would average 32 rows per expert. The allocated
worker capacity is 4096, but only live rows are scheduled.

Sorted expert runs are partitioned into M16 groups in the selected M16/N192 fused
kernel. Explicit within-tile weight reuse is up to sixteen rows; a 32-row expert
uses two groups. Larger or skewed runs can also reuse weights through cache across
groups. Average rows per expert therefore is not the same as rows per compute tile.

The existing isolated skewed fixtures measure 7.461 ms at 1024 tokens and 14.837 ms
at 4096 tokens. Their system-memory fills divided by time are 187.9 and 127.8 GB/s
per Spark, while L2 access rates are 463.1 and 763.5 GB/s. These include other kernel
traffic and do not establish pure weight DRAM bandwidth or percentage of peak.
They are historical same-worker-artifact component observations, not new counters
for live 2048-token chunks. See the [counter evidence](ds41-expert-skew-counters.md).

Consequently the current 1.6–1.8k end-to-end prefill rate cannot by itself determine
expert weight bandwidth. The lower external traffic rate at 4096 is compatible
with stronger cache reuse; it does not by itself mean worse expert efficiency.
