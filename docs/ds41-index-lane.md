# Owned index execution lane

`IndexLaneWeights` owns the eight official learned index-query producers at layers 2, 8, 14, 20, 24, 28, 32 and 36. Each execution lane borrows those weights, owns one reusable query workspace, and owns two selection workspaces. Query rebinding retains an exact-weight-owner capture per producer; outputs are unpublished on rebind, streams drain before switching weights, and all retained graphs are destroyed before their weights can be released.

The first selection workspace runs layers 2/8/14/20 and retains layer 20's candidate blocks. The second runs layers 24/28/32/36 using those retained candidates. This avoids overwriting the candidate source during later reindexing. Every producer must execute in order. Failure or an invalid transition makes the owner require restart; restart clears selections and starts at layer 2. Intermediate attention layers reuse the nearest producer's selection after checking exact source proposal bindings and row positions from `CacheAttention`.

`BackboneLane::select_index` feeds the lane's completed query into this owner. `attention_indexed_ffn` validates its retained selection against the current cache view and calls the existing sparse-attention/output/mHC handoff. Layers 0/1 use window attention without index selection. The caller still supplies the matching checkpoint sink, owns the actual request-to-query association, advances layers and handles the full batch transaction.

## Allocation evidence

A CPU-only container loaded the production official-checkpoint catalog and native SM120 AOT metadata, without exposing a GPU. The retained `official_index_lane_budgets_reject_before_allocation` test passed: all eight weight owners total **45,916,160 bytes**, insufficient weight/workspace budgets reject before allocation, missing weight owners reject, and capacities 0, 4097 and `u32::MAX` reject.

| Capacity | Query workspace | Each selection workspace | Total device workspace |
| ---: | ---: | ---: | ---: |
| 1 | 4,226,060 | 415,808 | 5,057,676 |
| 80 | 6,190,724 | 33,264,640 | 72,720,004 |
| 4096 | 106,266,628 | 1,703,149,568 | 3,512,565,764 |

These totals exclude persistent caches, host staging, other backbone owners, vision, engram and dSpark. They do not establish full-model fit. Reusing query storage avoids seven additional 106,266,628-byte workspaces at capacity 4096.

The production daemon builds with the final backbone handoffs. **This combined index owner and its rebound query projections have not been GPU-qualified.** The RTX driver/library mismatch remains unresolved; the Spark native library lacks the required SM120 FP8 projection AOT. Existing primitive and graph-cache qualifications do not prove this composition correct. Required follow-up includes real-weight layer revisits, source-20 candidate retention across all reindex layers, changed shapes, two-lane independence, stale-batch/restart recovery and numerical comparison through sparse attention.

Logs: `/tmp/ds41-index-lane-budget-test.log`, `/tmp/ds41-index-lane-final-build.log`. The budget test lives in the production module; `/tmp/ds41-lane-ffn` is only its external build harness. See [evidence record](ds41-index-lane.json).
