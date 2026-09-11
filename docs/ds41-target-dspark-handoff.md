# Real target-to-dSpark context handoff

The target pass now owns a bounded tap buffer and captures the four-stream residual mean before attention at layers 37, 38 and 39, matching the pinned reference's `main_hiddens` construction. The driver reaches each tap after the layer's required engram work and drains the tap read before advancing attention. At capacity 80 the BF16 `[rows,15360]` buffer uses 2,457,600 bytes.

Tap readiness requires all three layers in order for the same cache-batch identity. Outputs retain flattened request IDs, absolute positions and row kinds. The target pass exposes taps only while that batch is ready; commit, discard and failed/cancelled execution prevent subsequent access. These are private proposed inputs, not committed dSpark history.

`DsparkMainContext::execute_input` accepts the completed tap buffer and its flattened positions, validates extent/device, stages them into owned storage and executes the real main projection, normalization and three stage KV producers. Invalid input clears readiness before returning an error. The target API captures taps, but still performs target-only generation; it does not yet invoke this producer or generate/accept drafts.

## Real checkpoint validation

The full target fixture ran sixteen requests through an 80-row prefill and two 16-row decode passes on RTX GPU 1 with the four resident Spark workers. The test exports pre-attention residuals only in test builds. For each pass:

- Tap output is byte-identical to PyTorch's mean over the four BF16 streams, concatenated in layer order 37/38/39.
- The full FP32 target logit arrays are byte-identical to the saved pre-tap baseline.
- Request IDs, positions and batch association match the target cache batch; post-commit tap access fails.
- Real dSpark main-context output and all three BF16 KV proposal planes are finite. Invalid input during reuse clears previous readiness before a valid retry.

All three dSpark weight sets were resident on the same RTX as the target fixture, using 8,148,701,064 bytes; loading took 0.973 seconds in the recorded warm run. This is actual residency evidence, not a cold-start benchmark or a complete speculative-serving memory budget.

`scripts/qualify-ds41-target-taps.py` checks reference source hashes, uses the pinned official FP8 activation quantization/GEMM and RMSNorm with the real main projection/norm weights, and compares the normalized context:

| Input rows | Relative L2 error | Maximum absolute error |
| --- | ---: | ---: |
| 80 | 0.0001435685 | 0.0009765625 |
| 16 | 0.0001147297 | 0.0009765625 |
| 16 | 0.0001126501 | 0.0009765625 |

Reference execution uses TileLang 0.1.8 with activation-kernel vectorization disabled, as in the earlier SM120 reference checks. Metrics accumulate in FP64. The tolerance is relative L2 below 0.005. This independently checks the main projection/norm; finite KV planes alone are not independent KV numerical qualification.

The production release daemon builds. The tap-order/batch guard test passes, as does the real-model fixture. The restarted API passed arithmetic/counting, JSON/SSE usage/completion, disconnect recovery and unsupported-temperature checks. Three target-only decode observations were 10.998, 11.110 and 10.885 tokens/s, with first content in 0.250, 0.241 and 0.235 seconds. These short-context observations do not establish a speedup or the performance targets.

[Reference results, row metadata, output hashes, finite-plane checks and complete API events](ds41-target-dspark-handoff.json).

## Remaining integration

The three dSpark caches now have [request-bound accepted-prefix publication](ds41-dspark-prefix-publication.md), qualified separately from the target transaction. A [combined commit path](ds41-dspark-combined-commit.md) now coordinates publication with target-cache and engram commits, revoking admissions on partial failure. Proposal generation, confidence policy, verification, acceptance, RNG ownership, cancellation/rollback and concurrent scheduling are not enabled by this change. The development API remains target-only at `127.0.0.1:18041`.
