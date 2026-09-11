# Rejected mHC preparation graph cache

The query-stream chain in `bf72ed0` still launches mHC mixes, collapse, and
normalization separately before the query graph. A trial replaced those host
launches with an mHC-owned graph, retaining at most one row shape/destination
per weight binding and at most 40 bindings. Replays ran on the query stream;
weight rebinding, graph destruction, and error drains were explicit.

The release build and 56 real-weight cases passed. The test revisited layers
0/14/39/0, changed row counts and positions, switched between two query
destinations, poisoned inputs after computing the oracle, and injected producer
errors before reuse. Target/speculative API checks also preserved all eight
baseline texts/usages, including the existing strict formatting failure.

The trial did not establish a useful performance gain:

| Metric | Query chain | mHC graph cache |
| --- | ---: | ---: |
| One-row query preparation median | 107 us | 106 us |
| Six-row query preparation median | 110 us | 105 us |
| Target counting median | 31.38 tokens/s | 31.27 tokens/s |
| Speculative counting median | 103.00 tokens/s | 103.31 tokens/s |

These were sequential development runs with three short, predictable counting
streams, not a clock-admitted balanced benchmark. Small differences are not
evidence of a throughput improvement. The runtime graph cache was removed and
both APIs restored to the qualified `bf72ed0` query chain. The stronger
real-weight regression test remains, using ordinary streamed preparation.

This result does not prove that wider graph composition cannot help. It does
show that adding this intermediate graph owner is not presently justified.
Prioritize measured kernel execution and pipeline overlap for the next change.

Exact trial source, daemon, timing and API artifacts are under
`/tmp/ds41-hc-query-graph`. The adjacent JSON records artifact hashes and raw
counting samples. GPU test/build logs are under `/tmp/ds41-query-chain-fixture`.
