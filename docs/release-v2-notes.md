# DS41RT v2

V2 makes the two dSpark decode lanes independent after shared admission and
prefill, moves the normal serving loop to cooperative GPU completion, and uses
the RTX memory left after cache planning for complete routed-expert layers from
layer 0 upward. It keeps the official DeepSeek V4.1 Flash checkpoint and the
OpenAI-compatible serving surface from v1.

## Highlights

- Adaptive dSpark selects verification lengths per lane and per request. The
  former joint decision path and cross-lane cost term are removed.
- Normal decode completes without CUDA stream/device synchronization,
  synchronous CUDA copies, or CUDA allocation/free calls in the measured
  12-second mixed-serving trace. Admission, prefill, cancellation cleanup, and
  required within-lane data dependencies retain their explicit boundaries.
- Decode receive polling yields whenever a result is not ready, allowing the
  peer lane's short GPU work to progress. Combined attention, projection, and
  mHC graphs reduce submission overhead on this scheduler.
- The default memory plan keeps a 16.681 GB FP4 compressed source pool for
  18,710,016 logical tokens plus private tails, 24 completed-turn snapshots,
  and 24 prompt snapshots. SWA remains FP8 and the independent index remains
  FP4.
- Automatic bottom-up RTX placement loads five complete routed-expert layers
  in the standard dSpark configuration. Target-only serving can use the memory
  released by dSpark for a sixth layer.
- Unconstrained greedy decoding selects top-1 on the GPU and transfers 8 bytes
  per row instead of the full 517,120-byte vocabulary logits. Constrained
  decoding and diagnostics retain full logits when required.
- Cache production, index selection, attention, router/shared/local experts,
  TP dispatch/reduction, draft replay/commit, snapshot copies, logits, and lane
  retirement use owned in-flight state and cooperative completion.
- The clean five-host build now packages the required local RTX expert kernels,
  verifies their ABI, and publishes matching amd64 coordinator and arm64 Spark
  images under the v2 tag.

## Demonstrated impact

- Weighted dSpark throughput across the eight real content types increased from
  70.43 tok/s in v1 to 79.80 tok/s in the clean v2 image, a 13.3% gain. Every
  content type completed all three samples.
- Warm C16 aggregate decode increased from 742.91 to 934.05 tok/s, a 25.7%
  gain, while C1 remained stable at 151.12 tok/s.
- Weighted decode over a verified 256K retained prefix increased from 61.27 to
  66.82 tok/s. All 120 retained-context requests verified exact cache reuse.
- Removing unsuccessful-poll spin made the weighted mix effectively flat
  (80.77 to 80.75 tok/s in the development comparison) while raising the
  counting C16 diagnostic from about 880 to 938 tok/s. On top of that scheduler,
  the combined attention graph raised weighted throughput from 80.75 to 81.46
  tok/s in its matched development comparison.
- The clean standard configuration starts in 57.76 seconds, automatically keeps
  five complete routed-expert layers on the RTX, and retains an 18,710,016-token
  global FP4 pool plus 32,768 private-tail tokens.

Final release throughput, retained-context, concurrency, memory, startup, and
three-run high-thinking tool-eval results are recorded in the
[v2 performance report](release-v2-performance.md). The v1 prefill matrix and
official API reference are retained as prior measurements because those tests
were intentionally excluded from the scoped v2 rerun.
