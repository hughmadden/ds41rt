# TP4 dispatch and collection

`V41Tp4Tcp::dispatch` validates the canonical native request, writes it concurrently to all four peers and returns a `V41Tp4Pending` guard. The guard owns the four sockets until `receive` has validated complete, ordered rank coverage. It borrows the transport exclusively so another wave cannot overtake its responses. The existing `execute` method delegates to these two operations.

The underlying persistent client similarly separates `dispatch_chunks` from response collection. An in-flight socket stays inside the guard/future, outside the client's idle connection slot. Dropping dispatch, an uncollected guard or an incomplete receive closes that socket. Only a fully validated final response restores a persistent connection. An abandoned wave is not automatically replayed. This discards responses; it does not interrupt expert GPU kernels that have already started.

`NativeTp4Wave::dispatch_ffn` invalidates prior output, checks the bound request and returns `NativePendingFfn` after all four writes. The caller can execute the lane's shared FFN before calling `finish`. Completion validates exact shared/request identity and geometry, copies the shared result, collects bounded route chunks into the four existing planes and runs the unchanged native reduction. It publishes output only after reduction drains. Cancellation or failure leaves output unpublished. The combined `execute_ffn` remains available and rejects invalid shared bindings before dispatch.

The pending native guard borrows every reduction buffer and its publication state. Frame copies and reduction stay on the caller's polling thread. No background CUDA task, extra device workspace or unbounded response buffering was introduced. Reduction now drains its stream even when launching reports an error.

## Retained transport tests

Three localhost TP4 tests exercise real TCP connections:

- Workers withhold every response until dispatch returns, proving the dispatch call does not wait for collection. Two changed requests then complete on the same four persistent connections.
- Workers send responses that are deliberately left unread. Dropping the dispatched guard closes all four sockets; the next changed request connects afresh and receives no stale data.
- Workers send one valid partial chunk per rank and withhold final chunks. Cancellation after at least one destination callback closes incomplete streams; the next wave reconnects and receives complete, distinct results.

The full transport test suite passes **146 tests**, with one existing ignored test. The production daemon builds successfully.

## Real four-Spark execution

The new coordinator path ran on ostrich's GB10, with real expert computation on ostrich/dodo/emu/kiwi. For each of layers 0, 20 and 39, the workers loaded that layer's official checkpoint TP slices and executed both previously qualified 80-row request payloads. Each rank returned seven-row chunks. Both cases used one persistent connection per rank within each layer run.

After dispatch, the coordinator uploaded the corresponding previously qualified real shared-FFN output, then called the new pending completion method. All **24 rank planes**, **24 received request frames** and **six combined reductions** match the [earlier real TP4 qualification](ds41-real-tp4-qualification.md) exactly. Worker and coordinator fixture processes all exited successfully; none were left waiting.

The external fixture reconstructs typed bindings around the recorded router/shared payloads. It recomputes routed experts and native reduction, but does not rerun routing, the shared FP8 projection or the assembled backbone lane. The coordinator was co-located with rank zero on Spark because RTX still has a driver/userspace mismatch. These results validate dispatch ordering, persistent transport, complete rank delivery and reduction consistency. They do not measure shared-compute overlap, production RTX performance, end-to-end generation or the 90/270-tps targets.

Source/artifact hashes, exact comparisons, raw fixture timings and process records are in [the machine-readable evidence](ds41-tp4-dispatch-qualification.json). Scheduler wiring, real RTX shared-FFN overlap, alternating lanes, request cancellation/rollback and production transport/performance qualification remain open.
