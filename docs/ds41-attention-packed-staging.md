# Packed KV staging and parallel score computation

Sparse attention now loads four consecutive FP8 values per lane and reuses the resolved cache-row pointer while staging that row. It performs the same E4M3 conversion, E8M0/K32 scaling and BF16 rounding, then writes four packed BF16 values into shared memory. Byte-addressed unaligned inputs retain a scalar load fallback. Invalid references still avoid all KV dereferences and stage zero. Cache layout, tile dimensions, masking and shared-memory footprint are unchanged; no new allocation or VRAM intermediate is introduced.

The four existing warps also compute one sixteen-key QK tile each, preserving each score's K accumulation order. Previously one warp computed all four score tiles. Online softmax, probability rounding, accumulator rescaling and PV computation remain unchanged. Standalone comparisons tested score parallelism alone, packed staging alone and their combination: the combination performed best.

Full native-library measurements:

| Rows / selected source keys | Before µs | Packed + parallel µs |
|---|---:|---:|
| 1 / first 5 | 147.27 | 61.52 |
| 1 / full 512 | 576.39 | 221.44 |
| 6 / first 5 | 151.68 | 62.85 |
| 6 / full 512 | 593.42 | 231.63 |
| 80 / first 5 | 281.14 | 120.94 |
| 80 / full 512 | 1141.53 | 450.61 |
| 4096 / first 5 | 13641.38 | 5993.14 |
| 4096 / full 512 | 49877.84 | 19731.53 |

These CUDA-event timings surround twenty graph replays per implementation, with baseline measured first; they have no clock admission or alternating-order statistical qualification. They are kernel timings, not end-to-end prefill throughput. Unaligned values and window-only attention also improve in the recorded checks.

The existing `python/tools/qualify_v41_sparse_skip.py` now additionally covers private compressed-source slots with strides one and two, zero/mixed scale exponents, unaligned value buffers, window-only mode and selectable row counts. All 180 full-library cases pass BF16 bit equality: fifty aligned, fifty unaligned, fifty window-only/unaligned, and thirty at rows 128/256/4096. Each captured graph is reused while metadata, selected IDs and scale values change. Cases retain empty tiles, late valid tiles, gaps, full selections and stale metadata. Native/CUDA CTest both pass; the script passes Ruff.

Both live APIs pass streaming counting/usage, JSON arithmetic, cancellation recovery and unsupported sampling rejection. All eight paired quality cases preserve exact text and usage against the pinned-rank-upload deployment; all six objective checks pass. This does not close earlier broad numerical-quality gates.

Live sparse-attention median falls 182→78 µs for one-row target calls and 182→81 µs for six-row verification. Projection and FFN preparation remain roughly 93–94 and 57 µs respectively. Three fresh sequential counting streams measured target TPS 27.39/27.46/27.34 before and 30.55/30.35/30.31 after. dSpark measured 92.28/95.77/95.47 before and 98.11/100.83/101.49 after. Modes ran sequentially against the same Spark workers. These instrumented short runs exclude prompt processing and first-token latency, and do not establish the 90/270/8000 TPS targets.

Running coordinators are `ds41-attention-packed-target-api-dev` and `ds41-attention-packed-spec-api-dev`, ports 18041/18042. Executable remains `/tmp/ds41-plane-upload-artifacts/daemon`; native library is `/tmp/ds41-attention-packed-artifacts/cmake/libds41rt_native.so`. The four local-QP Spark workers and RoCE inference path are unchanged. Prior pinned-upload coordinators are stopped and retained.

[Variant comparisons, exact-output checks, live outputs, commands and hashes](ds41-attention-packed-staging.json) retain the evidence. Full records are under `/tmp/ds41-attention-parallel` and `/tmp/ds41-attention-packed-rollout`. Projection, expert/client costs, C16 and alternating waves, full-context quality/prefill and official-only release cleanup remain open.
