# Callable native encoder stream

The scoped target now exposes `NativeTarget::stream_prefill`. It calls the
retained `execute_encoder_stream` directly: two independently owned contexts
alternate prompt chunks, and a successor's layer attention waits on its
predecessor's cache publication rather than on its remote expert response.
The retained stream keeps suffix capture and cache commits in prompt order.
After the complete encoder, the facade replays layers20–39 over only the final
128 rows and borrows the resulting FP32 device logits. It does not call the
native HTTP server, sample tokens, copy a full vocabulary to the host, or load
another model/cache.

`StreamInput` owns exact token IDs, a fresh native request lease, the requested
chunk size, and increasing absolute prompt rows selected for logits. Selected
rows must lie in the final window; at most48 rows fit the current compact head.
The requested chunk size is80 through configured `batch_tokens`; larger rounded
AOT storage does not increase that requested chunk budget. Inputs and
the existing context/admission state are validated before any cache mutation.

```rust,ignore
let lease = target.bank().admit(0, request_id)?;
let runtime = target.runtime();
let result = runtime.block_on(target.stream_prefill(StreamInput {
    request: lease, tokens, chunk_rows: 1024, selected: vec![prompt_end - 1],
}, &|| !cancelled.load(Ordering::Acquire)))?;
let logits = result.logits()?;
// External CUDA consumers retain this borrow until their completion event.
drop(logits);
let accepted_end = result.commit()?;
```

The returned `StreamingResult` borrows the whole native target. Both ordinary
context jobs must be idle before entry, and neither context can be reused while
its output survives. This is the explicit mode switch required by the C ABI's
two-lane actors: drain their jobs/result leases, end their borrowing scope, run
the stream, then re-enter the lane actors after sampling and publication.

Commit consumes the result and completes the entire decoder replay, leaving the
native request in Full phase for subsequent decode. Partial accepted counts do
not apply to prompt prefill. After encoder admission begins, cancel, an error, or dropping an unfinished
future drains both lanes and revokes the entire fresh admission. Validation or a
pre-dispatch Rust callback rejection preserves an unstarted admission; the C ABI
explicitly revokes it when acknowledging cancellation before execute. Encoder chunks already
publish internal source state; that state must never be advertised as a complete
model cache hit. A canceled streaming request has a stale lease and requires a
new admission. A drain failure poisons the owner and prohibits further execution.

The retained suffix adds at most128×40,976bytes (about5MiB) of transient CUDA
storage to the existing constructor allocation. It remains owned until final
replay consumers drain. This must be included in integration memory accounting;
the requested source-pool budget is not total model/cache/workspace memory.

CPU tests replace only the stream device operations and run the actual facade
lifecycle. They cover exact final-window selection, chunk boundaries, invalid
inputs before mutation, errors at begin/encoder/replay/commit, cancellation at
publication fences, dropping pending encoder/replay futures, and ready-result
cancel/drop without double release. A Rust compile-fail test protects the
borrowed logits from premature cancellation. They do not measure GPU overlap or
prove kernel arithmetic. The parent ran `abc4529f` GPU probes on 19 September
2026: short streaming results matched the full-target logits and continuation
hashes; a370-token prompt with80-row chunks and final128-row replay passed, as
did in-flight and ready-result cancellation with source credits restored.
These development-profile probes do not qualify throughput.

This first callable seam accepts fresh text prompts only. Prefix restoration,
continuation CED, images, dSpark anchors and simultaneous unrelated decode during
an encoder stream remain explicit integration work. The [C ABI](afd-native-target-abi.md) now exposes `encoder_stream` with the
exclusive mode switch and consumer fence; the vLLM binding must opt into that
phase. The new C ABI mode needs its own live qualification. Presence of the Rust
method alone is not a serving or performance claim.
