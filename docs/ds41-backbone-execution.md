# Per-layer backbone execution owner

`CacheProducerWeights` loads the official 40 window projections, four compressed-source producers and 40 FP32 attention sinks under one aggregate device budget. `BackboneExecution` owns all 44 producer workspaces and retains each layer's private proposal until acceptance. Each alternating pass needs independent mutable workspaces; immutable weights can be shared.

`execute_layer` consumes the backbone lane's already-completed query. It produces the window and, at layers 2/8/14/20, compressed KV/index proposals; obtains the bank's matching attention views; runs learned index selection at all eight index producers; executes sparse attention/output projection/FFN normalization; dispatches TP4 routed experts; executes the shared expert; collects and reduces routed results; and completes FFN mHC. It owns the matching checkpoint attention sink and derives expert request metadata from the cache batch.

The caller still initializes embedding/query state, performs engram updates and dSpark tap reads between layers, advances the lane and starts the next query. Layer 39's completed output remains accessible through `BackboneLane::output` for the target head. This owner does not implement the full request scheduler, target sampling, vision input or dSpark acceptance.

Each planned cache batch now has a unique identity even when its request/position lists equal another plan. The execution owner requires one identity across all 40 layers. It invalidates pass progress before work starts and only publishes completion after FFN mHC finishes. Errors or cancellation after polling require restart. `commit` requires exactly one complete 40-layer pass and consumes that readiness before committing accepted prefixes to the cache bank. Failed or duplicate commits require restart. The enclosing scheduler still determines acceptance and coordinates engram/dSpark histories; this guard is not a complete model transaction.

## Validation

- Production daemon build passed: `/tmp/ds41-backbone-execution-build.log`.
- Retained `pass_commit_requires_all_layers_and_one_batch` unit test passed: rejects incomplete passes, unfinished steps, foreign batches, skipped layers and repeated commits. This tests the progress guard, not induced GPU/transport cancellation.
- CPU-only official-catalog/native-AOT test passed: all cache-producer weights and sinks total **145,517,568 bytes**; one byte short rejects before payload loading/allocation, and invalid capacities reject.
- The actual Spark CUDA cache-bank lifecycle test passed again for two cycles of 16 requests, now also checking distinct identities for equivalent plans.

| Capacity | All 44 producer device workspaces |
| ---: | ---: |
| 1 | 18,429,768 |
| 80 | 81,507,616 |
| 4096 | 3,314,188,448 |

Workspace totals exclude persistent caches, host staging, backbone/index/transport workspaces, engram, vision and dSpark. These figures do not establish full-model fit.

The combined layer method has **not** executed on GPU. Real query/cache/index/attention/TP4/shared/mHC numerical composition, async cancellation across its owners and full accepted-prefix commit still need qualification. RTX execution remains unavailable under the current driver/library mismatch. Prior component qualifications do not prove this combined path. The source metadata and narrower test results are recorded in [the evidence file](ds41-backbone-execution.json).

## Subsequent distributed execution

The combined layer-0 method now passes two real distributed batches on RTX plus four Sparks, including embedding/query/window/attention/shared/routed/mHC execution and partial-pass commit rejection. See [the layer-0 integration record](ds41-layer0-integration.md). This establishes execution of that path; independent assembled-layer numerical comparison, compressed/index/engram dependencies and the full 40-layer driver remain open.
