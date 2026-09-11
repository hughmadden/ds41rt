# Native TP4 collection and fabric measurements

The development API now connects to Spark workers at `10.55.0.1` through `.4`, port 19441. The RTX host's fabric address is `10.55.0.12` (400 Gb/s interface); the four Spark primary fabric interfaces report 200 Gb/s. The earlier `172.22.2.x` endpoints used 10 Gb/s management Ethernet. Link rates are interface reports, not measured application bandwidth.

Three real counting streams on the fabric returned the expected sequence with normal usage/completion, at 6.054, 6.247 and 6.373 decode tokens/s. Disconnect recovery also passed. These use the development daemon, one serial request, greedy target-only decode and short context. The metric and workload are defined in [the API record](ds41-native-api.md). They do not establish the requested 90/270/8000 TPS targets.

## Optimized coordinator baseline

Building the same daemon source with `cargo build --release` and restarting only the RTX API container produced 7.929, 7.798 and 7.984 decode tokens/s. The repeated counting streams had first-content latencies of 0.319 and 0.323 seconds. The qualifier passed expected text, usage, completion, disconnect recovery and unsupported-temperature rejection. All four Spark workers remained unchanged development binaries. This isolates the coordinator build profile operationally, though sequential runs can still vary. The release daemon is the current development endpoint; these results remain short-context target-only measurements, not release qualification.

## Locating the collection overhead

The new `ds41rt::timing` / `target collection` event reports shared device copy, summed response uploads, receive time excluding those copies, and native reduction. A 520-layer-call, single-row trace on the management network had medians of 6, 45, 2680 and 9 microseconds respectively. Receive time includes waiting and protocol work; it is not a pure network latency measurement.

An external fixture on ostrich loaded the real layer-0 rank-0 expert weights and executed one synthetic finite input row with six routes. After warmup, 100 samples per capacity gave:

| Allocated row capacity | Kernel median, µs | Host exchange median, µs |
| --- | ---: | ---: |
| 1 | 382.807 | 408.822 |
| 16 | 393.703 | 418.646 |
| 80 | 392.231 | 418.998 |

Host exchange includes binding the layer, three input uploads, synchronized GPU execution, output allocation and download of 122880 bytes. Output partials were finite and byte-identical across capacities. This warm, single-layer probe does not support reducing workspace capacity as a meaningful decode optimization, nor does it measure full-model weight locality.

A separate coordinator probe replayed one recorded real input row to all four resident all-layer workers. It validated response sizes and consumed bytes without uploading them to a coordinator GPU. Each mode used 40 warmups and 120 measured round trips:

| Client build / network | Repeated layer 0, µs | Rotating all 40 layers, µs |
| --- | ---: | ---: |
| Development / management | 2730.741 | 2780.436 |
| Development / fabric | 2419.674 | 2399.243 |
| Release / fabric | 2148.456 | 2068.707 |

The workers remained the same development binaries throughout. The release-client result is not a release-worker measurement. Runs were sequential, not a randomized benchmark, so small differences may include runtime variability. The remaining roughly two-millisecond round trip warrants worker service/serialization and scheduling measurements before transport redesign. Reusing the same input at different layers tests latency, not mathematical correctness.

[Machine-readable probe results and complete fabric API record](ds41-expert-collection-timing.json). Temporary probe source and build outputs remain outside the repository.
