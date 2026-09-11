# Shared KV across attention head groups

All 64 query heads use the same latent KV vectors. The previous kernel launched
four independent blocks per query, each handling 16 heads and loading/dequantizing
the same keys. The new prefill paths share one decoded KV tile across two or four
head groups in a single block. Per-head online-softmax and accumulation order are
unchanged. This is reuse within a query; adjacent-query reuse remains unmeasured.

The register-rescaling change makes each group's in-loop scratch small enough to
fit together. Once the last PV completes, the KV tile's shared storage is reused
for output staging. Four groups drain in two synchronized pairs, so all 64 heads'
outputs never need simultaneous shared storage. No intermediate VRAM allocation
or request-time allocation is introduced.

| Path | Head groups/block | Threads | Blocks/query | Dynamic shared bytes |
|---|---:|---:|---:|---:|
| Fewer than 128 rows, or split calls | 1 | 128 | 4 per key partition | 100288 |
| Unsplit 128–255 rows | 2 | 256 | 2 | 75648 |
| Unsplit 256–4096 rows | 4 | 512 | 1 | 84224 |

The four-group kernel uses a 512-thread launch bound and compiles at 128 registers
without spills in the prototype. It has sixteen active warps per block instead
of four. Thresholds keep adequate block count for prefill; split decode remains
on the one-group implementation and can be evaluated separately later.

## Qualification

The integrated library passes all 34 existing reference cases, including ordinary
and committed compressed caches and bounded replay with two/ten key partitions.
The reference harness checks independent blockwise math for every row and pinned
TileLang reference results for the first min(rows,8) rows.

All 220 direct old/new comparisons are byte-identical, including 127/128/129 and
255/256/257 dispatch boundaries, 1024/2048/4096-row prefill, split decode,
unaligned values on both new paths, window-only input, varying scales and stale
metadata. CUDA racecheck reports zero hazards in full-KV fixtures at 128 and 256
rows, exercising both shared-memory output paths.

The initial two-group prototype passed 40 exact comparisons and roughly halved
large compressed-attention time. Four groups won a further 50 exact comparisons
against that prototype. The 2048-row full-compressed case falls from approximately
7.76 ms in the selected baseline to 1.95 ms in the four-group prototype. Integrated
regression/timing records are in the evidence JSON. These are kernel measurements,
not a claim of a fourfold end-to-end gain or measured DRAM traffic.

## Isolated 16k serving comparison

Both versions use the independent-chunk scheduler and 2048-token chunks. Only one
API pair is resident at a time; no concurrent GPU work or compilation runs during
measurement. Eight sequential C1 requests contain 16,410 code or 16,411 repeated
prompt tokens and generate 59 counting tokens. All prompt hashes, output text and
usage match, with no prefix cache hits. The table uses the second request of each
kind/mode. Rates are tok/s.

| Workload / mode | Baseline prefill | Head reuse prefill | Change |
|---|---:|---:|---:|
| code / target | 4177.98 | 5320.59 | +27.3% |
| code / speculative | 4118.01 | 5276.37 | +28.1% |
| repeated / target | 4536.98 | 6563.79 | +44.7% |
| repeated / speculative | 4433.79 | 6459.57 | +45.7% |

The separate 599-token code-context decode comparison preserves text and usage:
target throughput is 37.87 versus 38.02 tok/s, dSpark is 118.78 versus 117.97.
These are close observations on one counting workload, not broad decode metrics.

Both API lifecycle checks pass, including cancellation/recovery. All eight quality
cases preserve the previous text and usage for each mode. The inherited Unicode
format failure and differing open-ended explanation remain, so the strict quality
script still exits 1. Release-quality and broader model-equivalence gates remain
open.

## Batch-size follow-up and selection

Retesting 4096-token chunks with the new kernel still loses: warm code prefill is
4409/4595 tok/s and repeated text 5343/5451, versus approximately 5.3k/6.5k at 2048.
The default chunk size stays 2048. This rejects that batch-size configuration;
it does not reject larger expert M groups within the selected batch. The reason
for the larger-batch loss still needs profiling, including the unpaired leading
chunk and worker execution/transfer schedule.

The new library is selected on ports 18041/18042 in
`ds41-head-reuse2048-live-target-api-dev` and
`ds41-head-reuse2048-live-spec-api-dev`. The prior attention containers are stopped
and retained for rollback. The daemon and Spark worker artifacts are unchanged.
Raw source experiments, exact commands, frozen library and logs are under
`/tmp/ds41-head-reuse`; [portable evidence](ds41-attention-head-reuse.json) records
the hashes, checks and results.

Next: use a fresh prefill profile to choose expert grouping/ordering or remaining
chunk-pair/transfer overlap work. Adjacent-query KV reuse is still unmeasured;
its priority should reflect attention's reduced cost. The 8k prefill, 90/270 decode,
C16 and release-quality goals remain open.
