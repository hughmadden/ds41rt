# Native greedy speculative API

`serve-native --dspark` now loads all dSpark weights, main-context producers, three FP8 windows and the captured draft chain on the coordinator RTX. The shared embedding and vocabulary owners are borrowed from the target. Omitting `--dspark` retains target-only generation for comparison.

Prefill publishes target-derived context through the combined target/engram/dSpark commit. Decode generates five private drafts, verifies the anchor and drafts with the target, commits the matching input prefix and streams only verified output. Correction and bonus tokens become the next pending anchor. EOS stops emission; the remaining output budget bounds verification and prevents context-limit overshoot. Each completed, disconnected or failed request releases its dSpark admissions. Cancellation is checked between GPU passes and before publication; it does not interrupt a running kernel.

The API uses fingerprints `ds41rt-native-fp8-kv-dspark` and `ds41rt-native-fp8-kv` to identify its mode. The development endpoints are `127.0.0.1:18042` for dSpark and `127.0.0.1:18041` for target-only. Add `--dspark` to the existing native launch command to enable speculative serving. The service remains serial, greedy, text-only and limited to 4096 prompt-plus-output tokens.

## Live qualification

Both modes pass arithmetic, counting, JSON/SSE completion and usage, disconnect recovery and unsupported-temperature checks. Output limits 1, 2, 5, 6, 7 and 13 produce identical text and exact completion counts in both modes, with `length` finishes. Both also succeed after six idle seconds.

Testing exposed stale TCP connections: Spark's fallback server closes connections after five idle seconds. Native serving now discards idle sockets before each admission, while retaining connections across steps within a request. No partially delivered expert wave is replayed. This fixes the observed between-request failure; it is not exhaustive transport-fault qualification.

Three short counting streams with dSpark measured 40.565, 43.188 and 42.454 completion tokens/s, with first content in 0.258, 0.255 and 0.248 seconds. The measurement includes EOS and HTTP overhead and excludes time through first content. The same final build without dSpark measured 10.112, 10.295, 10.611 tokens/s on this workload. The easy counting workload has high acceptance; these results do not establish broad quality or the 90/270 tokens/s and 8k prefill targets. Qualification and comparison requests run sequentially against the same four Spark workers; the two coordinator modes occupy separate RTX GPUs.

The idle process memory observation was 20,764 MiB for dSpark and 12,522 MiB for target-only. It establishes fit for this tested serial configuration, not the final concurrent/long-context budget. The release build and three native TCP dispatch/cancellation regression tests pass.

Concurrent acceptance and scheduling, stochastic sampling, confidence policy, broader quality/performance evaluation, long context, vision and release build/run migration remain open.

[Complete API events, target comparison, output-limit and idle checks](ds41-speculative-api.json).
