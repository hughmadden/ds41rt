# Direct native expert submission

TCP previously created a blocking execution task for every request, which then submitted to the native service's GPU-owning thread and waited on a second response queue. The native service now submits directly from the socket runtime to that existing thread. Requests remain bounded at 16 and responses at one owned chunk. CUDA allocations and execution remain on their owning thread; slow consumers still apply bounded backpressure.

`ProtocolV2ExpertExecutor::submit_tcp_chunks` is optional. Existing synchronous executors retain the blocking adapter. Both paths use the same response validation before socket writes: frame size, request/placement/layer identity, response shape and final-chunk ordering. A channel closing before its final frame or reporting an execution error fails the connection. Dropping the receiver cancels delivery; it does not interrupt an in-flight CUDA launch.

## Measurements before full deployment

The one-layer real-weight fixture used the recorded layer-0 routing and input from the existing distributed qualification. Before the change, median GPU-thread queue wait was 94.5 µs and worker time was 505 µs, while the TCP server's non-write interval was 1.185 ms. With direct submission, corresponding medians were 98, 494 and 707 µs. These intervals include host scheduling and are not pure GPU or network measurements.

Three paired single-peer runs compared the resident full worker to the temporary direct-submission layer-0 worker:

| Repeat | Existing worker, µs | Direct worker, µs |
| --- | ---: | ---: |
| 1 | 1559.532 | 1066.432 |
| 2 | 1249.116 | 1166.732 |
| 3 | 1227.173 | 897.528 |

Each entry is a median of 120 requests after 40 warmups. All returned FP32 route payloads were byte-identical across the two services and repeated inputs. Runs were sequential and the services differed in resident layer count, so this is supporting latency evidence rather than a controlled full-model speedup claim. The earlier one-layer/full-layer comparison found similar latency before this change. The temporary service was stopped before full deployment.

## Validation

The transport suite passes 148 tests; one CUDA-dependent verbs test remains ignored. New tests exercise direct submission without invoking the blocking adapter, persistent connection reuse, missing final completion, mismatched request identity and explicit worker errors. The production release daemon and the ARM release worker fixture both build successfully. The full workers use the production service/transport through an external CLI shim; the shim is not a release packaging qualification.

## Full five-host deployment

All four resident workers now use an ARM release build with direct submission and queue/load instrumentation. The coordinator also uses a release build. The same API qualifier returned `4` for the arithmetic prompt and the exact comma-separated sequence 1 through 20 on all three streams, with normal usage/completion. Disconnect recovery and nonzero-temperature rejection passed.

| Stream | First content, seconds | Decode tokens/s |
| --- | ---: | ---: |
| 1 | 0.253729 | 10.019948 |
| 2 | 0.238798 | 10.395924 |
| 3 | 0.244285 | 10.166078 |

The earlier release coordinator with development Spark workers measured 7.798–7.984 tokens/s. This rollout combines optimized Spark builds and direct submission, so the full-model gain cannot be attributed exclusively to queue removal. Both runs used the fabric and the same short-context greedy workload. Median single-row target step time is 97.321 ms; median collection receive time excluding GPU uploads is 1.352 ms per layer. These remain development measurements, far below the requested targets and without dSpark or concurrent scheduling.

Summed per-layer loading took 118.293, 115.483, 119.453 and 113.067 seconds for ranks 0–3. Each rank retained 80,216,064,000 bytes of expert weights. Process counters reported approximately 144.4 GB of read syscall bytes and 146–157 GB of storage reads per worker; these counters include startup and readahead and are not a cold NVMe bandwidth benchmark. Setup from the final layer-loaded event to readiness took 22–24 ms. Loading still needs optimization; there is no large post-weight graph setup cost in this path.

The upgraded development API remains at `127.0.0.1:18041`, using fabric peers `10.55.0.1` through `.4` on port 19441. [Complete API events, probe timings, loading records and binary hashes](ds41-native-direct-submission.json).
