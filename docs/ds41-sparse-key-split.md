# Sparse attention key partitions for decode

The sequential sparse kernel walks up to ten 64-key tiles using four CTAs per query. The RTX serving path now distributes those tiles across ten key partitions for compressed attention and two for window-only attention when one request has at most 16 live rows. Each partition retains FP32 numerator, maximum and normalizer; a second kernel combines them with the sink and writes BF16 output. Larger requests keep sequential attention. Both paths skip the first valid tile's unnecessary rescale of zero accumulators.

The new additive C ABI takes explicit caller-owned scratch and partition count. Scratch spans must cover `[rows, parts, 64, 514]` FP32 and be disjoint from every input and output. Pointer arithmetic uses 64-bit offsets. One arena is allocated per attention wave at startup, sized for `min(capacity,16)` rows and ten partitions: maximum 21,053,440 bytes. All scratch elements are overwritten, including empty partitions and invalid-metadata sentinels. Consecutive requests reuse it on one stream after the preceding merge. Graph replay performs no allocation, initialization or module resolution. The original C ABI remains available; a new daemon requires the new native library.

## Live results

Baseline is `3b7dad7`, batch 2048/reservation 4096, with small-row FP8 plans already enabled. Both versions use the same frozen code/repeated filler, 16,410/16,411 actual prompt tokens, no prefix-cache hits, and a greedy request to count 1–20 (59 completion tokens). Each arm has two requests per context; these are the second requests. Prefill is effective prompt throughput including time to first content.

| Context / mode | Before decode tok/s | After decode tok/s | Before prefill tok/s | After prefill tok/s |
|---|---:|---:|---:|---:|
| Code / target | 26.57 | 38.52 | 1342 | 1341 |
| Code / dSpark | 85.92 | 113.28 | 1327 | 1337 |
| Repeated / target | 26.60 | 38.92 | 1513 | 1512 |
| Repeated / dSpark | 96.17 | 114.56 | 1514 | 1524 |

Target decode improves about 45%; code-context dSpark improves 32%, repeated-context dSpark 19%. All eight benchmark texts, usage records and prompt hashes match the baseline. Short counting remains about 38–39 target and 115–119 dSpark tok/s. The large-context target penalty is substantially reduced; this is not a claim of universal context-independent performance or the 90/270 tok/s goal being reached. Prefill remains below the 2–3k intermediate checkpoint and 8k release target.

Both APIs pass JSON/streaming, cancellation/recovery and unsupported-sampling checks. Seven of eight quality cases preserve both modes' text/usage. Both Unicode arithmetic responses now omit the prior extra blank line, with the numeric answer still 36 and paired outputs equal. The inherited strict-only-integer failure remains: 5/6 objective checks, 8/8 pair equality, quality command exit 1. These are bounded quality checks, not broad model qualification.

## Numerical and memory qualification

Key splitting changes BF16 probability rounding; it is not bit-exact to sequential online softmax. An initial mixed-scale comparison failed the old pairwise `rtol=.008, atol=.002` gate at 22/32768 elements, with the largest failing difference 0.00439453125. This failure is recorded here and in the companion JSON; the final tool retains per-case diagnostics rather than describing an unchanged numerical result.

An independent address gather plus full FP32 QK/softmax/PV (including sink, TF32 disabled) shows slightly lower relative L2 error for the candidate across all original 40 stress cases. The qualification tool now measures both implementations against that oracle. The split gate requires candidate relative L2 <= .003, relative L2 <= 1.05 times baseline plus 1e-5, and max absolute error <= 1.25 times baseline plus 1e-5. The old elementwise check remains a reported diagnostic. The unchanged sequential path still requires bit equality.

Final component checks cover 150 split cases: aligned and unaligned compressed values, window-only attention, plus a fresh seed 2026 with query scale 1 instead of .2. They cover rows 1/2/6/16, changing selections, empty/late/gapped tiles, permuted pages, private proposals with strides 1/2, zero/mixed scales and stale metadata. Every split case meets the FP32 gate; none has larger relative L2 error than its corresponding baseline. Worst candidate relative L2 against FP32 is 0.002307981. Fifteen mixed-scale cases fail the prior pairwise elementwise diagnostic. Forty large-row sequential cases (80/256/1024/4096) remain bit-exact. Outputs and scratch are poisoned before replay, scratch guards survive, and Torch allocated memory stays unchanged across replay.

The separate official-reference harness retains its existing elementwise tolerances. It passes seven geometry/history cases, comparing every row to an independent 64-key online-softmax transcription and the first min(rows,8) rows/all 64 heads to pinned TileLang. Small rows exercise the split ABI; 80/4096 rows exercise sequential attention. Query/page/metadata changes, invalid descriptors, output guards, undersized/aliased scratch, invalid partition counts and width-zero ascending positions pass. Invalid metadata also produces zero with sink=-1000, using a negative-normalizer sentinel rather than relying on a non-underflowing sink exponential. The large-page test passes exact analytic results for an 8GiB value pool, recycled high pages and an invalid page. Reference tooling requires TileLang 0.1.8 with the separate TVM-FFI 0.1.6 overlay; the initial missing-package/incompatible-FFI attempts did not test kernel correctness.

Release Rust/native builds, both native selftests and the CPU-only official lane budget gate pass. Lane workspace totals are 2,301,981 / 80,606,556 / 2,985,230,348 bytes at capacities 1/80/4096; resident weight accounting remains 6,896,423,360 bytes.

Representative component graph times for full compressed selection are 212→26 microseconds at one row, 217→45 at six rows and 217→86 at 16 rows. Window-only one-row attention is 46→25 microseconds. These are warm component measurements, not full-model speedups. Output-column partitioning was also tested and rejected: the 256-column variant only modestly improved small rows, while narrower variants duplicated too much QK work and regressed wider batches. Skipping rescale when all factors equal one added little beyond the selected first-tile skip.

## Current bottlenecks and deployment

The final two-request 16k code profile contains 4,640 one-row layer observations. Mean sparse attention is 55.7 microseconds, compared with roughly 352 in the prior profile. Current means are 85.7 for query preparation, 77.6 output projection, 34.8 FFN preparation, 271.8 the complete expert phase, and 178.6 collection within that phase (166.3 receive, 4.5 upload, 5.7 reduction). Timing logs add overhead; complete profiled target steps average 27.24 ms. These nested intervals are not additive. Host submission/synchronization and expert turnaround now matter substantially for decode.

Warm code prefill remains 12.265 s in the profile: 3.189 s sparse attention and 7.729 s expert phases, including 5.845 s collection. The split path deliberately does not duplicate large-prefill work. Improving the large-row attention kernel and remaining coordinator/transport overhead are still necessary for the prefill goal.

Live containers are `ds41-splitattention2048-{target,spec}-api-dev` on 18041/18042. Frozen daemon SHA256 is `975d65d80192aa781d4d999e05994bcd0aa22d63e1218b1a6d184623c944e97b`; native is `0aac3fd644b25369baa068947dba326d07134bad897a4912d8d484afc0678385`. All four RoCE expert workers remain unchanged. Prior FP8-plan containers are retained stopped for rollback; the profile container is stopped. CLI batch defaults are unchanged. Launch arrays, summarized evidence and hashes are in the companion JSON; full raw artifacts are in `/tmp/ds41-attention-rescale`.

Reproduce component checks with `python/tools/qualify_v41_sparse_skip.py --baseline OLD.so --candidate NEW.so --split-parts 10 --rows 1 2 6 16 --output result.json`. Window-only uses `--split-parts 2 --window-only`; unaligned uses `--unaligned-values`; the held-out arm uses `--seed 2026 --query-scale 1`. Run `scripts/qualify-ds41-sparse-attention.py` with the pinned reference directory, `--split-parts 10` and then `--large-only` for the cache-address gate. The exact live benchmark uses `scripts/bench-ds41-prefill-api.py` with `--filler-tokens 16384 16384 --kinds code repeated` and the frozen context file from the prior prefill run.
