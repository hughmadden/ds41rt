# Target text execution pass

`TargetPass` owns the mutable embedding, backbone lane, index lane, all cache producers, engram upload/two gates and target-head workspace for one pass. Immutable weight owners are borrowed and can be shared between alternating waves. TP4 transport is supplied by the enclosing executor. All CUDA work must be polled on its owning thread.

`execute` validates the request batch and selected head rows, starts text embedding, and calls the combined layer executor for all 40 layers. Between layers it advances the residual/pre state, polls bounded asynchronous engram work at layers 1 and 14 with a timeout, applies each gate and begins the prepared query. Only a completed layer-39 output can enter the target head. Output row selection supports at most 80 distinct rows in flattened batch order, for prefill-last/decode or verification use.

The owner rejects reuse until commit or explicit discard. Before execution it enters a non-publishable state; dropping or failing the async execution cancels the request batch and mapped I/O through a drop guard. Completed logits are associated with the exact cache-batch identity. Commit requires that identity and delegates accepted-prefix publication to `Requests`, coupling engram history and all backbone caches. Failed commit consumes readiness; discard resets the execution owners before reuse. Acceptance decisions and sampling remain external.

## Evidence and limits

The production daemon build passed (`/tmp/ds41-target-pass-build.log`, 10.72 seconds). The retained CPU progress-guard test rejects idle/incomplete, foreign-batch and consumed readiness and reuse of a pending pass (`/tmp/ds41-target-pass-guard.log`). This test does not inject asynchronous GPU or transport cancellation.

**The complete 40-layer pass has not run on GPU.** Existing distributed layer-0/reference and layer-1 handoff results qualify those narrower paths only. Next validation must use all-layer real Spark workers and run prefill, accepted-prefix commit and a subsequent decode through this owner. It must check compressed/index behavior, final logits and history positions before connecting the path to a live API.

This is the target-only text path. Vision replacement, dSpark tap capture/acceptance, sampling, the API adapter, full request scheduling and alternating-wave performance are still required. No end-to-end quality, startup or throughput claim is made.
