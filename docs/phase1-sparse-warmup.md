# Phase 1: avoid repeated sparse warmup execution

Adaptive sparse graph misses previously executed the whole workload as warmup,
then captured and replayed it. A changed request layout therefore repeated GPU
work even when all involved kernel variants were already initialized.

The wave now remembers the eight native dispatch variants: split or one/two/four
unsplit head groups, for each cache format. Adaptive mode warms each variant
once after a successful synchronized execution. Pointer or grid changes alone
do not repeat that warmup. Existing fingerprint validation, capture, replay and
stream draining remain. Fixed-policy behavior is unchanged. Warmed status belongs
to the wave and its retained library/device; graph eviction does not unload code.

| Three-run C1 medians | Previous adaptive tok/s | New adaptive tok/s |
|---|---:|---:|
| Code | 121.58 | 122.22 |
| Fable | 52.52 | 53.00 |
| Topic | 65.52 | 66.41 |

Uninstrumented mixed C16 improved from 144.70 to 151.37 tok/s; C4 from 100.85 to
101.90. A matching instrumented diagnostic measured C16 at 153.61 versus the
previous 146.31 tok/s. These are sequential exploratory samples, not balanced
release qualification. The fixed-five mixed-C16 reference remains about 168
tok/s, so the adaptive concurrency gate is still open and adaptive stays off
by default.

The instrumented C16 sparse-attention median fell from 35.38 to 27.62 ms and
whole verification from 164.88 to 160.73 ms. Captures remained in 36/42 rounds
versus 34/41 before; the optimization removes duplicate warmup execution, not
the changing-layout captures themselves. Routing and row choices differ between
runs, so these phase medians are not isolated causal estimates.

Validation: all nine C1 outputs and four C4 outputs match the previous adaptive
probe. C16 has 6/16 exact matches; its broader numerical divergence remains open.
Both versions pass the 32K middle needle, full prompt reuse, completed-turn
continuation, eight cancellations with eight surviving counting requests, and
post-cancellation recovery. Two distinct 191-token prompts additionally exercise
the two-group unsplit variant and both return the expected answer. The short
and 32K probes exercise split, single-group and four-group variants. Release
build and test compilation pass. Standard serving is restored.

RTX remains at 400 W with standard memory speed (13365 MHz under load, 14001 MHz
maximum); native weights, FP4 global KV and FP8 SWA are unchanged. The
[raw summary](phase1-sparse-warmup.json) records artifact hashes and results.

Next, investigate a batched sparse launch with per-row device descriptors for
request buffers, preserving each query's causal metadata and arithmetic. That
could key retained graphs by bounded total-row shapes instead of every request
layout, reduce launch overhead, and address remaining captures. It needs native
ABI, scratch-capacity, ownership, numerical and lifecycle qualification before
serving integration; it is not part of this change.
