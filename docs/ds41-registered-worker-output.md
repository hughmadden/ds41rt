# Compact expert output directly into registered response slots

The native worker previously compacted its final rank result in device memory,
copied it to a reusable host plane, then copied that plane into the transport's
registered send slot. The persistent RoCE transport already posted sends without
waiting for the final completion; it could start the next request while the NIC
sent the preceding response. The missing connection was the GPU-accessible slot
already supplied to the executor callback.

The worker now compacts its final BF16 rank plane directly into that slot. Once
the CUDA stream is synchronized, the transport writes the response header and
posts the send. Its existing send-ring completion accounting prevents reuse while
the NIC still owns the slot. There is no intermediate download, CPU plane copy,
or transport-side device copy for a complete fitting response. No expert matrix,
precision, kernel, routing, reduction order or inference protocol changes.

A response that cannot fit its negotiated slot, or requests a debug checksum,
uses the existing host/chunked path. Admission still bounds registered storage;
this change adds no per-request allocation. The existing fallback host plane and
device compaction buffer remain allocated for those cases.

## Qualification

The real rank-zero Spark fixture loads one official expert layer and exercises
1, 16, 80, 256, 2048 and 4096 rows. For every shape it compares registered output
with ordinary device compaction of the *same* FP32 expert result. This avoids
confounding the comparison with a second atomic execution's summation order.
All bytes match. An extended 1.90-second rerun also exercises the host fallback
for every shape, compares its downloaded bytes with its own device result, and
requires nonzero outputs. Prefix/suffix guards survive, and too-small slots and checksummed
requests return to the fallback path before execution. The fixture passes in
1.38 seconds. Ten native transport tests also pass; the two live-only tests are
not counted in that unit result.

## Baseline evidence

A rank-zero diagnostic sample of 4,957 complete 2,048-row transport timing lines
has median send/staging 0.850 ms and send-completion polling 0.002 ms. The matching
expert sample has median download 0.358 ms, compaction 0.315 ms and kernel 9.90 ms.
These mixed-workload, per-field medians are not an additive critical-path profile.
They identify avoidable copies without attributing network latency to a final
completion wait that is absent from the implementation.

## Four-worker and API results

The live four-Spark RoCE fixture passes repeated dispatch, abandoned dispatch,
sink failure and recovery through 4,096 rows. Ordered small-row responses remain
exact; the existing bounded atomic replay gate applies to large prefill. All
12 benchmark texts/usages and both 599-token counting outputs match baseline.
Both API lifecycle checks pass, and all eight quality-case texts/usages per mode
are preserved. The inherited Unicode-format failure and target/speculative
explanation difference remain; strict quality qualification is still open.

Means of the last two of three 16k requests, with the coordinator and kernels held
fixed, are:

| Effective prefill tok/s | Host-copy worker | Registered-output worker |
| --- | ---: | ---: |
| Target, code | 6,460 | 6,964 |
| dSpark, code | 6,379 | 6,941 |
| Target, repeated text | 8,078 | 8,096 |
| dSpark, repeated text | 7,675 | 7,966 |

The separate 599-token decode sample changes from 39.02 to 38.00 target tok/s and
119.85 to 118.04 dSpark tok/s. This small-sample run establishes no decode gain;
retain the measured decrease rather than calling the rates identical. These are
counting workloads, not the external category benchmark or C16 throughput.

Across all four workers, each has 2,240 complete 2,048-row timing samples: every
response uses one direct device frame, median download is zero, and median send
posting is 0–1 microseconds at the logger's resolution. Median expert kernels
remain 9.83–9.94 ms. Registered-output compaction takes 0.411–0.422 ms, versus
0.315 ms in the earlier rank-zero device-output sample, trading about 0.1 ms for
removal of the much larger host copies. This is not a measurement of wire latency.

Selected workers are `ds41-mapped-worker` on ostrich, dodo, emu and kiwi. The frozen
worker is `/tmp/ds41-mapped/worker`; native libraries remain
`/tmp/ds41-expert-direct/cmake/libds41rt_native.so`, built from b12x `6e868d97`.
The repository b12x pin remains `e53fd5a1`; no native library was rebuilt here.
The unchanged continuous-encoder APIs remain on ports 18041/18042. Launch arrays,
logs, comparisons and hash/smoke checks are under `/tmp/ds41-mapped`; previous
`ds41-prefill4096-worker` containers remain stopped for rollback. Exact hashes and
measurements are in [the result record](ds41-registered-worker-output.json).

## Remaining scope

The expert kernel and compaction still finish before the complete response is
posted. This change does not make partially accumulated token rows available
mid-kernel, or add Spark-side cross-rank reduction. Those require separate
measurement and numerical/ownership qualification. Same-prompt coordinator lanes
already continue across chunk boundaries; mixed-request concurrency-16 serving
and broad bounded-decoder-replay quality remain open.
