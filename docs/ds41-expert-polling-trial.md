# Bounded GPU-thread polling: rejected placement

An isolated worker compared the same binary with zero, 1,000 and 2,000 microseconds of active polling on its incoming work queue before returning to blocking receive. Only ostrich changed; all four workers kept the mixed-capacity native expert library. The zero-polling default preserves the sleeping queue. The experiment bounded every idle spin window and kept existing queue capacity, ordering, response backpressure, disconnect and CUDA ownership semantics.

**This polling placement is rejected.** It did not improve end-to-end API performance or TCP server time and substantially increased CPU consumption. The experimental setting/helper are removed from repository source. Request ID, layer and row fields on worker service timing are retained because they allow matching queue and TCP traces reliably.

| Poll window | Requests caught polling | Median queue | Median service | Median server execution | Matched server remainder | Avg. process CPU cores |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0% | 105.5 µs | 277 µs | 512 µs | 99 µs | 0.197 |
| 1,000 µs | 0.10% | 97 µs | 280 µs | 541 µs | 120 µs | 0.603 |
| 2,000 µs | 52.47% | 1 µs | 196 µs | 544 µs | 301 µs | 0.935 |

Each row matches 7,080 one-row requests by ID. The server remainder is calculated per matched request as TCP server execution minus GPU-thread queue wait and service duration, then summarized; it is not the difference of aggregate medians. It includes uninstrumented submission/response scheduling and validation work, with possible overlap. It does not uniquely identify OS scheduling or prove CPU contention. The larger window successfully reduced queue wait, but time elsewhere offset that benefit.

Target-only counting TPS, three sequential runs per trial:

- Zero: 11.51, 11.45, 11.31.
- 1,000 µs: 11.20, 10.84, 11.86.
- 2,000 µs: 11.05, 11.08, 11.11.

All trials pass JSON arithmetic, streaming output, cancellation recovery and unsupported-sampling rejection. Counting text and usage match exactly across trials. These are ordered single-host experiments with extra tracing, not randomized multi-host release benchmarks. CPU core equivalents come from process CPU-time deltas divided by elapsed time over the entire qualification, including startup graph work. No clock/throttle admission or host power/network settings were changed.

Three standalone Rust tests passed for the experimental queue helper: FIFO/disconnect, zero-window blocking behavior, and a delayed sender waking the receiver after the window. Both the experimental worker and final trace-only source compiled in release mode on Spark. The isolated package initially needed dependency-cache population and lock regeneration; all three trials used the same resulting executable and lock, whose hashes are retained.

## Resulting state and next action

The normal `ds41-slice-target-worker` is restored on ostrich, with the prior diagnostic logging settings. Temporary polling containers were removed after their logs were collected. The other three workers never changed. Both APIs were restarted after readiness and returned `ready` in smoke checks. The mixed expert kernels remain live; the previously recorded explanation wording divergence remains open.

The next structural experiment should reduce handoffs between the I/O loop and GPU owner, or use a persistent registered/RDMA path with deliberate completion polling. These results reject spinning on this queue at this point in the current architecture; they do not rule out active polling in a different I/O design.

[Matched summaries, CPU samples, API results, launch commands and experimental source](ds41-expert-polling-trial.json) are retained for reproduction. Full API event/worker logs remain in `/tmp/ds41-poll-trial`, with hashes in the evidence. The polling implementation is preserved there as evidence rather than as an unused runtime option. Final repository behavior only adds correlation fields to service timing; it still calls blocking `requests.recv()`.
