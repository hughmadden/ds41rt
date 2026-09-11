# Request ownership and engram handoff

`Requests` owns backbone-cache admission and engram history under one cache lease. Preparing a batch derives cache positions, ordered token IDs and image barriers together, then starts both mapped-table prefetch/gather jobs before embedding. Text entry rejects image batches until vision replacement exists. The model driver can poll the required engram layer on its CUDA thread, apply the gate, and only then begin its query.

Acceptance validates every engram history and count before attempting device-cache publication. Only a complete 40-layer execution may commit. An execution/publication error cancels the batch and revokes participating cache and history admissions. Invalid acceptance counts fail preflight without consuming the batch. Cancelling a proposal leaves request admission intact; releasing or failed publication invalidates its generation. dSpark acceptance remains the enclosing scheduler's responsibility.

The chained test exposed a readiness cycle: engram polling initially asked for query-ready input while the block was awaiting engram. `pending_engram` now exposes only layer/position metadata in that state. The existing prepared-input guard still prevents queries or taps from consuming the block before its engram gate completes.

## Actual execution evidence

- The retained request-owner test passed on RTX in 3.67 seconds. It uses the official tokenizer and both mapped engram tables, 16 admitted requests and 80 rows with image barriers; runs gather/upload/dequant; checks masks; rejects cancelled access and excessive acceptance; rejects an incomplete pass and readmits all 16 requests with fresh generations and empty histories. It does not compare gathered payload arithmetic independently.
- The retained distributed layer test now uses `Requests::prepare` and `begin_text`, runs actual layer-0 attention/shared/mHC on RTX and real routed experts on four Sparks, then advances into layer 1, polls mapped engram work, applies its real gate and executes its real query. Both changed 80-row batches pass in a 6.15-second test. Gated residuals change and all inspected BF16 values are finite. Layer/position association and incomplete-pass revocation pass.
- All four saved layer-0 residual/pre outputs are byte-identical to the earlier independently qualified results. This carries the layer-0 regression forward; it does not independently qualify the new layer-1 arithmetic.
- All five distributed processes exited successfully. The production daemon build passed. The loader's three focused engram tests passed before the chained run.

Inputs retain the deterministic token formula in the distributed test: `(i * 7919 + cycle * 113 + 17) % 129280`, with 16 contiguous five-token requests at positions 0 through 4. Spark workers load real layer-0 TP shards and handle both requests over persistent connections. Execution uses the isolated matching RTX driver libraries documented in [RTX testing restoration](ds41-rtx-testing-restored.md).

The per-batch 0.377/0.292-second observations include the tested layer handoff and host assertions but exclude much of initialization. They are not full-model throughput measurements. Artifact hashes and logs are in [the evidence record](ds41-request-engram-integration.json).

## Still required

Independent layer-1 gate/query reference comparison, layer-1 attention/FFN, compressed/index integration across later layers, all 40 layers, successful cache/engram accepted-prefix publication, dSpark transactions, vision and live API generation remain open. No end-to-end quality or performance result is claimed.
