# Pair full prefill chunks before the tail

The 16,410-token prompt previously ran one ordinary 2,048-token encoder chunk,
then three full pairs and a final 2,048+26 pair. After KV head sharing, the warm
code trace spent 547 ms on the initial unpaired chunk and 496 ms on the uneven
last pair. At 4,096 rows those phases cost 1,153 and 918 ms. These are host
wall-clock intervals; nested expert, attention and collection timings overlap
and must not be added as GPU kernel costs. The trace is retained in
`/tmp/ds41-tail/profile-summary.json`.

The serving scheduler now pairs full chunks from the start and runs the remaining
26-token chunk alone. Reserved single-chunk execution shares the independent
encoder loop with paired execution, preserves ordered source/window publication,
and revokes its admission on cancellation or failure. Short single-chunk prompts
retain their ordinary path. This changes pairing, not token chunk boundaries,
expert tiling, native kernels, decoder replay length or transport precision.

The real RTX/four-Spark TargetPass fixture compares serial execution with pairs
and optional tails: 65+65, 79+1, 65+65+1 and 64+64+27. All retained residual/pre
bytes match; ordered commits and decoder-replay readiness pass. The existing
cancelled-pair test also passes before reusing the same owners. The release daemon
and test binary build successfully.

The first warm API run reaches 6,510 target / 6,393 speculative code-prefill tok/s
and 7,407 / 7,823 repeated-text tok/s at roughly 16k prompt tokens. The preceding
selected-kernel run measured 5,321 / 5,276 and 6,564 / 6,460 respectively. These are
single warm samples per mode and workload, not statistical peak estimates. All
eight benchmark outputs/usages and eight quality-case outputs/usages per mode
are preserved. Both lifecycle checks pass. The inherited Unicode-format failure
and differing target/speculative explanation remain; this does not close broad
quality or bounded-decoder-replay approximation qualification.

The separate 599-token counting comparison preserves text and usage in both
modes: target 37.71 → 38.00 tok/s, dSpark 119.95 → 118.42 tok/s. This is a
regression check, not a decode speedup claim. The selected daemon is frozen at
`/tmp/ds41-tail/daemon`, with the unchanged head-sharing native library at
`/tmp/ds41-head-reuse/cmake/libds41rt_native.so`. Exact hashes, API results and
profile phases are in [the result record](ds41-prefill-tail.json).

Selected development APIs are `ds41-tail2048-live-target-api-dev` on port 18041
and `ds41-tail2048-live-spec-api-dev` on port 18042; exact launch arrays and
post-rollout smoke responses are under `/tmp/ds41-tail`. Workers and b12x pin
remain unchanged. The previous head-sharing APIs are retained stopped for rollback.

## Remaining prefill investigations

- Measure the remaining pair-boundary idle time and Spark response compute,
  compaction and transfer overlap before changing pipeline depth.
- Revisit expert ordering and larger within-block groups after those scheduling
  costs are controlled. Existing M32/M64 losses reject those implementations;
  M16 remains selected until a realistic routed workload shows a better choice.
- Use the new attention profile to decide whether adjacent-query KV sharing is
  worth its synchronization/storage cost. Head sharing is deployed; cross-query
  sharing is still unmeasured.
- Recheck sparse indexer/top-k, Engram and compressor fusion only where the live
  profile shows material remaining cost. Concurrency-16 scheduling and broad
  prefill correctness remain separate open gates.

The user-supplied four-Spark dSpark category table is preserved in
[the benchmark record](ds41-external-dspark-benchmark.json). Its “ours” label is
from the supplied comparison, not this runtime. Finish the planned prefill work
before broad decode tuning; then use matching category prompts and concurrency,
separately reporting acceptance, draft/verification cost, stream decode and
wall-clock aggregate throughput. Counting alone is not that comparison.
