# Chain mHC preparation into the attention query stream

The preparation trace exposed work outside `target layer`: layer advance,
Engram integration, tap capture, and mHC/query preparation. The one-row
baseline layer-begin median was 117 us; six-row verification was 120 us.
Nested query timers attributed roughly 49–50 us to mHC/norm, 6 us to the
normalized-input copy, and 60–61 us to query execution. Nested medians must
not be added to other stage totals.

`AttentionQueryWave::execute_tokens_prepared` uploads token positions before
calling the producer. The backbone mHC producer enqueues its existing
mix/pre/norm operations on the query stream, writing the norm result directly
into the stable query input. The query graph follows on that stream and drains
before publishing its output. This removes one synchronous D2D copy and the
intermediate mHC stream drain. The producer remains outside the captured query
graph; graph ownership, weights, row-shape replacement, and numerical kernels
are unchanged.

If the producer or later query operation fails after submitting work, the
stream drains before the producer borrow is released, and query output remains
unpublished. Existing callers with already completed inputs retain the ordinary
`execute_tokens` entry point. This change neither frees the legacy mHC norm
buffer nor claims to fuse mHC into a single kernel.

## Validation and measurements

- Release daemon build passed.
- The ignored integration test `v41_query_preparation` uses official layer
  0/14/39 weights, row sequence 1/6/16/80/256/6/1, changed inputs and positions,
  and both graph capture/replacement and replay. All 42 cases preserve hidden,
  rank, normalized rank, projection, RoPE, and frequency bytes exactly. Injected
  errors after enqueueing mHC are followed by successful reuse and exact output.
- Target and speculative API smoke, streaming, cancellation/recovery, and
  unsupported-sampling checks passed. All eight quality cases preserve the
  preceding build's text and usage. The inherited Unicode formatting failure
  remains: 5/6 strict objective checks, with the quality script exiting 1.
- Layer-begin median: 117 -> 109 us for one row; 120 -> 113 us for six rows.
  Candidate query-preparation total: 107/110 us respectively.
- Three counting streams: target median 31.01 -> 31.38 tokens/s; speculative
  102.39 -> 103.00 tokens/s. These small changes do not establish an end-to-end
  throughput gain. The counting prompt is predictable and not representative
  quality/performance qualification. Component logs also include setup and
  quality work; no clock-admitted paired microbenchmark is claimed.

To rerun the real-weight test, supply `DS41RT_NATIVE_LIB` and
`DS41RT_V41_SNAPSHOT`, select the physical GPU with `CUDA_VISIBLE_DEVICES`, and
run:

```sh
PYO3_PYTHON=/home/tj/.local/bin/python3.12 cargo test --release \
  --manifest-path rust/Cargo.toml -p ds41rt-daemon \
  --test v41_query_preparation -- --ignored --nocapture
```

Raw baseline artifacts: `/tmp/ds41-query-trace`. Candidate API/timing artifacts:
`/tmp/ds41-query-chain`. Component fixture and integration-test logs:
`/tmp/ds41-query-chain-fixture`. Both APIs use the merged b12x native library
whose hash is recorded in `ds41-b12x-upstream-merge.json`.
