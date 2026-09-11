# Skip entirely masked sparse-attention tiles

The coordinator's previous attention-phase timer included sparse attention, output projection, and FFN preparation. New stage tracing measured one-row medians of 379, 93, and 116 µs respectively in the short API probe. Window-only layers 0/1 used roughly 120 µs of sparse attention; compressed-source layers used 368–407 µs. The native sparse kernel iterated over the window plus all 512 selected-entry slots, performing WMMA and online-softmax work even on tiles whose resolved references were all invalid.

The kernel now uses a CTA-wide vote after resolving a 64-entry tile. If every reference is invalid, it skips that tile's loads, WMMA operations, and softmax update. This is a `continue`, not a truncation: later valid tiles are still processed. Entirely masked tiles contribute zero probability and preserve the existing accumulator/softmax state. No selection count is moved into a compile or graph key.

## Numerical and live checks

`python/tools/qualify_v41_sparse_skip.py` compares the baseline and candidate native libraries through the actual CUDA ABI. Thirty replay cases at 1/2/6/16/80 query rows are BF16 bit-exact, covering no valid selections, five early selections, five late selections after empty tiles, separated valid entries, all 512 selections, and stale metadata. The same graphs replay after input selection/metadata changes. A standalone candidate was checked first, then the complete CMake-built native library was checked again. Both native/CUDA CTest pass for the complete build.

Both live APIs pass arithmetic JSON, counting streams/usage, cancellation recovery and unsupported-sampling rejection. All eight paired quality prompts preserve text and usage exactly against the local-QP-owner deployment, and target/dSpark agree with each other. This does not close broader numerical quality gates inherited from earlier compact-return changes.

| Diagnostic median | Before | After |
|---|---:|---:|
| Target decode TPS | 20.56 | 24.48 |
| dSpark decode TPS | 79.40 | 88.00 |
| Target sparse attention stage | 379 µs | 180 µs |
| Target output projection | 93 µs | 91 µs |
| Target FFN preparation | 116 µs | 113 µs |

The API measurements use three sequential same-prompt streams per mode, one client, greedy output, excluding prompt/first-token time. The stage baseline comes from an instrumented target-only run; the preceding deployment supplied the before TPS. These are diagnostic measurements without clock/throttle admission, not a controlled broad benchmark. The full-native kernel probe with one row and five valid selections measured about 342→147 µs; full selections remained bit-exact and did not regress in that small probe. This is not evidence of 8k-prompt throughput or optimized fully populated attention.

## Deployment and next work

The running APIs are now `ds41-sparse-target-api-dev` (18041) and `ds41-sparse-spec-api-dev` (18042). The four `ds41-local-qp-worker` Spark containers and their expert libraries are unchanged. New coordinator native SHA256: `999d8b20ff691198bcff46a338064b162f829a194dec2f67a6f0b9ac443ae5e7`. The stage-tracing coordinator executable SHA256 is `6e5636728be366a7f106fb80f8a332471fa5e52d5d6011c374b36850a4f38e97`. Both are under `/tmp/ds41-sparse-skip-artifacts`. Prior API containers/artifacts remain stopped and intact.

Sparse attention still uses the existing native WMMA implementation. Fully populated selection work, projection/FFN preparation, synchronous stage copies and low decode occupancy need further tuning, including lower-level b12x attention support where useful. C16, alternating waves, long-context quality and prefill targets remain open.

[API results, exact-output comparisons, all thirty numerical cases, stage medians, artifact hashes and creation commands](ds41-sparse-empty-tiles.json) are retained. Full event/log records are in `/tmp/ds41-sparse-skip-rollout`; initial stage tracing and kernel checks are in `/tmp/ds41-attention-stage`.
