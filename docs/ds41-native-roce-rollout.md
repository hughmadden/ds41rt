# Native model rollout over RoCE

The target and dSpark APIs now run real DeepSeek-V4.1-Flash inference through the existing persistent RoCE QPs to four Spark expert workers. Native inference has no TCP payload fallback. TCP remains the endpoint bootstrap/control channel and the client-facing HTTP API transport.

Running containers:

- RTX: `ds41-roce-target-api-dev` on localhost 18041, `ds41-roce-spec-api-dev` on localhost 18042.
- Each Spark: `ds41-roce-target-worker`, all forty backbone expert layers ready, capacity eighty.

The worker native library is unchanged from the mixed-slice rollout: width 64 for capacity one, width 192 for capacity eighty, SHA256 `872591810ab2d9f3ed1715c13d8207b3d715aabd7bf466bd33a661b64a5c29ad`. The new worker executable SHA256 is `f0277dba54ede1178426ad81a61617e79887481a8eed2f9278d9099d842f511d` on all four hosts. The rebuilt coordinator executable is `07e377fa7078d654664f41c25a33278c5e46bd6dcc30230b0688bdb744abbdeb`; its full CUDA/AOT/RDMA native library is `fb922d87f2755d5f933ff069b026e3aaae81f7e154ab871698e1bc66f43ba14d`. b12x remains pinned to `c1b2b8f6410808cd9c7d28a5b0de28dd4dd8c17e`.

The frozen previous coordinator library had RDMA disabled. Its replacement was rebuilt with RDMA enabled and all configured AOT exports regenerated; native and CUDA CTest both passed. The first build invocation omitted GPU access required by the exporter and failed. Re-running the build container with GPU access completed successfully. The temporary worker package lock also needed refresh; the cached offline build succeeded. These are resolved build failures, not model failures.

## API checks and diagnostic throughput

Both modes passed JSON arithmetic, streamed counting with usage, client cancellation and subsequent recovery, and rejection of unsupported sampling. Eight paired quality prompts preserve exact text and usage relative to the prior mixed-slice deployment in both modes. All six objective checks pass, and target/dSpark text and usage agree on all eight cases. This does not close the earlier compact-return numerical discrepancy or establish broad model quality.

Three sequential streams per mode, one client, same short counting prompt, greedy output. Decode rate excludes the first completion token/prompt work and includes EOS/HTTP overhead. The first run warms. Historical TCP results are from the previous rollout; this is not a clock-controlled or isolated transport-only A/B because binaries and the coordinator native build also changed.

| Mode | Previous TCP TPS | RoCE TPS | Median change |
|---|---|---|---:|
| Target | 12.02, 11.02, 11.64 | 14.11, 13.60, 15.06 | +21.2% |
| dSpark | 49.02, 49.85, 48.26 | 59.57, 64.97, 65.90 | +32.5% |

Coordinator phase medians across qualification traffic: one-row expert phase 839 µs and attention 579 µs; six-row expert phase 1166.5 µs and attention 598 µs. Previous corresponding expert phases were approximately 1308/1783 µs. Aggregated phase medians are not an exact additive critical-path decomposition.

## Remaining work and artifacts

Worker request/frame copying, host-to-device input staging, host response staging, coordinator input D2H, response H2D and execution-thread handoffs remain. Registered mapped slots are available in the verbs implementation but the native GPU worker does not yet consume them directly. No GPU-direct, C16, long-context, prefill-throughput or release-target claim follows from this rollout.

[API results, paired comparisons, phase summaries, worker hashes and exact container commands](ds41-native-roce-rollout.json) are retained. Full logs are in `/tmp/ds41-roce-rollout`; artifacts are `/tmp/ds41-roce-rtx-artifacts`, `/tmp/ds41-roce-worker`, and each Spark's unchanged `/tmp/ds41-slice-artifacts/cmake`. Old TCP API/worker containers remain stopped; restoring them would violate the current inference transport requirement. The RoCE containers and artifacts should be used for subsequent qualification.
