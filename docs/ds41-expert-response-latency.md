# Locating latency outside the expert kernels

The live slice rollout's coordinator trace places most expert latency in response collection. For one-row target execution, median routing/dispatch/shared times are 49/11/52 microseconds; collection is 1,188 microseconds. Within collection, shared copying, return uploads and reduction are 5/21/6 microseconds, while waiting/processing responses is 1,153 microseconds. Six-row verification shows routing/dispatch/shared at 59/18/54 microseconds and response wait at 1,581 microseconds. These are aggregate phase medians, not additive per-request critical-path measurements.

Additional existing timing was enabled on **ostrich only**: `ds41rt::timing=debug` appended to RUST_LOG and `DS41RT_PROTOCOL_V2_TCP_TIMING=1`. The other three workers remain at the rollout's logging settings. Candidate library and all executable/weight bindings are unchanged. Ostrich was reloaded, both APIs restarted after readiness, and the target streaming/cancellation qualification passed.

Across 7,320 traced worker calls:

| Measurement | Median | p90 |
|---|---:|---:|
| GPU-thread queue wait | 95 µs | 219 µs |
| Service work, excluding response emission | 259 µs | 325 µs |
| Response emission | 5 µs | 9 µs |
| TCP server execution span | 466 µs | 735 µs |
| TCP response write | 20 µs | 48 µs |
| GPU expert kernel | 173 µs | 188 µs |
| Host expert execution span | 225 µs | 263 µs |

The server execution span includes queueing, execution and async response delivery. It is not solely GPU execution. The queue and server logs cannot yet be joined by a shared request identifier, so differences between these medians are suggestive rather than an exact decomposition. Extra logging may itself affect timing. The remaining coordinator wait may also reflect the slowest of four peers, network/IRQ scheduling and coordinator polling.

## Network diagnostic

Route lookup confirms traffic from `10.55.0.12` to `10.55.0.1` uses `enp1s0np0`; ostrich returns over `enp1s0f0np0`. Both NICs report adaptive RX/TX coalescing enabled, eight-usec RX/TX settings, and no GRO flush or deferred IRQ timeout. Coordinator cpu0 reports the `powersave` governor and ostrich cpu0 `performance`; these labels alone do not establish actual clock or idle-state behavior. No settings were changed. Noninteractive sudo was unavailable on both hosts, but the read-only and application-level probes could proceed.

Eight 100-ms-spaced pings had 0.226/0.778/1.075 ms minimum/mean/maximum RTT. Forty 10-ms-spaced pings had 0.027/0.320/0.981 ms RTT. ICMP results do not isolate the runtime or predict its four-peer TCP latency.

A bounded Python TCP echo server on ostrich and persistent TCP_NODELAY client on the coordinator measured 200 samples per case after twenty warmups:

| Request/reply bytes | Client pause | Median RTT | p90 |
|---|---:|---:|---:|
| 8/8 | none | 91 µs | 126 µs |
| 8/8 | 1 ms | 500 µs | 911 µs |
| 5280/10240 | none | 132 µs | 209 µs |
| 5280/10240 | 1 ms | 139 µs | 197 µs |
| 31680/61440 | none | 211 µs | 219 µs |
| 31680/61440 | 1 ms | 136 µs | 430 µs |

These include Python scheduling and copies and test only one peer. They do not include ProtocolV2 validation, GPU execution, the worker queue or four-peer tail latency. Case-order and power-state effects are uncontrolled. They nevertheless show that a decode-sized TCP exchange can complete far below the observed live response wait, while small idle exchanges can suffer substantial variability. The temporary echo server exited normally after the client closed; no probe listener remains.

## Next work

Measure each request's network receipt, GPU queue/service completion and response delivery with a shared identifier, and account for the slowest peer. Test bounded active polling or a direct native I/O/compute handoff against the existing sleeping queue; retain cancellation/backpressure behavior and measure CPU cost. In parallel, investigate the NIC/CPU wake-up contribution through a controlled experiment. Persistent registered/RDMA transport remains the larger structural target. Avoid assuming that eliminating the few measured copy microseconds can close a millisecond-scale response gap.

[Raw samples, phase summaries, traced target API results, probe source and exact traced-worker command](ds41-expert-response-latency.json) are retained. Full worker trace remains `/tmp/ds41-latency/worker.log` on the coordinator. This is profiling evidence, not a new speedup, network tuning result, or resolution of the rollout's output divergence.
