# Chain attention post and FFN preparation on one stream

Backbone FFN preparation now passes the completed attention projection directly to attention mHC post, copies the resulting residual and next-pre coefficients asynchronously, and enqueues FFN mixing/pre/normalization on that same stream. It removes the projection-to-temporary copy and the intermediate host synchronization. One final drain remains and runs even after partial enqueue failure, before either mHC owner can be reset or rebound. Existing buffers and arithmetic order are preserved.

The existing block fixture compared the old and new paths across layers 0/2/7/8/14/20/24/39, with eighty rows spanning sixteen requests. Forty FFN-input, residual, pre-coefficient and final-output arrays—59,002,880 bytes—match exactly. Replay, stale-query and origin guards pass. The daemon and its test target compile.

Both APIs pass JSON arithmetic, streaming counting/usage, cancellation recovery and unsupported-sampling rejection. All eight paired quality cases retain exact text and usage relative to the masked-tile deployment. This remains bounded quality evidence, not closure of the earlier full-model numerical gates.

Median FFN preparation fell from 113 to **95 µs** for one-row target calls and from 114 to **96 µs** for six-row speculative calls. Three short sequential counting streams measured target TPS **24.66, 24.91, 24.92**, versus **24.51, 24.48, 24.42** before; dSpark measured **89.04, 89.02, 89.97**, versus **88.29, 87.55, 88.00**. These small rate changes are diagnostic and have no clock/throttle admission or statistical-significance claim. They exclude prompt processing and first-token latency.

Running APIs are `ds41-hc-chain-target-api-dev` and `ds41-hc-chain-spec-api-dev` on ports 18041/18042. The new executable is `/tmp/ds41-hc-chain-artifacts/daemon`; the native library remains the masked-tile build at `/tmp/ds41-sparse-skip-artifacts/cmake`. All four `ds41-local-qp-worker` Spark workers are unchanged. Previous API containers remain stopped and retained.

The mHC mixing kernel still launches one block per row and computes all twenty-four projections within that block. For single-row decode this restricts its execution to one SM; kernel-level timing and a more parallel implementation are next. Broader attention, C16, alternating waves, long-context quality and prefill targets remain open.

[Exact tensor comparisons, live outputs, stage timings, executable hash and container commands](ds41-hc-boundary-chain.json) are retained. Full fixtures are under `/tmp/ds41-hc-chain-fixture` and live logs under `/tmp/ds41-hc-chain-rollout`.
