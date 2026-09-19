# Native target C ABI 1

This disabled serving backend calls the retained native model directly. It loads
coordinator weights once, constructs two native execution contexts and owns one
physical `Requests → BackboneCache → SourceCache` bank. It never constructs the
schema-1 `CacheCommands` prototype, a Python backbone, a duplicate KV allocator,
a native HTTP server or a native sampler.

The exported declarations and caller safety contract are in
[`ds41rt_target.h`](../rust/crates/ds41rt-daemon/include/ds41rt_target.h).
Build offline with `cargo build --offline -p ds41rt-daemon --lib` from `rust/`;
the development shared library is `target/debug/libds41rt_daemon.so`. No new
dependencies or lockfile changes are needed.

## Boundary and ownership

`create(config_json)` immediately returns an opaque numeric registry handle.
Initialization has command ID zero. One dedicated CUDA owner thread enters
`with_target` and keeps the native library, immutable weights, cache, transport
and both `TargetPass` objects within its constructor callback. After initialization,
`command(handle, json)` copies input and admits it to a bounded channel. Its
returned command ID is a receipt for queue admission; the JSON reply determines
whether the operation succeeded. At most 32 uncollected commands/replies exist.

`result(handle, command_id, buffer, capacity)` returns NOT_READY until complete.
BUFFER_TOO_SMALL reports the required size without consuming the reply. OK copies
and consumes it; subsequent reads are STALE. Config and reply JSON are capped at
65536 bytes; command JSON is capped at 16777216 bytes, with no NUL terminator.
This fits the full 1048576-token prompt without a separate upload protocol. Replies contain `command_id`, `ok`, and
`result`, or `command_id`, `ok:false`, `code` and `error`. Handles are never raw
Rust pointers; destroyed handles and foreign/old tickets cannot address resources.

Two local asynchronous lane actors poll retained execution futures on the
constructing thread. A per-client mutex protects only command/reply metadata;
it is not held across native execution, network waits or consumer completion.
Each lane owns one batch. The actual Rust `Logits` borrow stays in that lane's
result scope; it is not transmuted, made Send or escaped as a borrowed Rust object.
The exported CUDA pointer remains valid because the actor retains that borrow.

After `acquire_logits`, the caller may read the immutable FP32 device rows on
the indicated device. Producer writes have completed before READY. Submit all
sampler reads to one retained CUDA stream, then call `release_logits` with its
handle. Native records an owned event after those reads and makes its private
wait stream depend on that event. Only native completion of that wait stream
allows the release ACK. The caller keeps its stream alive and submits no further
reads after requesting release. A timeout leaves the original command pending;
resume collecting that command, never invent an early completion or free storage.

Held/draining leases block commit, cancel, request release and shutdown. One lane's
held lease does not stop the other lane. A completed lease cannot be acquired
again. Invalid accepted counts preserve the completed output; execution/publication
failures expose no new accepted extent. Cancellation drops the scoped future,
drains native transport/private work, then releases its scheduling lease.

Shutdown requires idle lanes and all other replies collected. Its ACK is emitted
only after the constructor scope has dropped native resources. `destroy` then
removes the registry entry and joins the finished owner. BUSY preserves resources;
there is no finalizer that frees outstanding GPU readers.

## Commands and replies

Native requests are `{owner,slot,generation}` and tickets are `{request,lane,id}`.
`owner` is a nonzero fresh worker-incarnation nonce. All command objects are
strict; unknown fields and unsupported work phases are rejected.

| Command | Successful `result` |
| --- | --- |
| initialization, ID 0 | `{state:"initialized",bank}` |
| `info {request?}` | `{state:"info",bank,request,committed_end}`; last two fields null if omitted |
| `admit {slot,request_id}` | `{state:"admitted",request,committed_end:0,bank}` |
| `can_prepare {work:[{request,tokens}]}` | `{state:"capacity",can_prepare:bool,bank}` |
| `submit {lane,request,expected_committed_end,work}` | `{state:"prepared",ticket}` |
| `execute {ticket}` | `{state:"executing",ticket}` |
| `poll {ticket}` | `{state,ticket,error}` |
| `acquire_logits {ticket}` | `{state:"leased",ticket,lease,device_pointer,device_id,bytes,rows,vocabulary:129280,dtype:"float32",selected,positions}` |
| `release_logits {ticket,lease,consumer_stream:"0x..."}` | `{state:"consumed",ticket,lease}` after the native fence |
| `commit {ticket,accepted}` | `{state:"committed",ticket,committed_end,bank}` |
| `cancel {ticket}` | `{state:"cancelled",ticket,bank}` |
| `release {request}` | `{state:"released",request,bank}` |
| `shutdown` | `{state:"closed"}` after resource teardown |

Full-target work is `{phase:"full_target",tokens:[u32],selected:[usize],
kind:"prefill"|"decode",placement:u64}`. Tokens must be below 129280; selected
rows are strictly increasing and bounded by the compact head capacity (48).
The expected committed frontier is checked against the actual bank before
preparation. Poll states are prepared, executing, ready, leased, consumer_draining,
consumed, committed, cancelled and failed; terminal tickets remain queryable until
that lane admits its next batch. `device_pointer` and stream handles use hex strings
so no JSON number conversion can truncate pointers.

Encoder-stream work is `{phase:"encoder_stream",tokens:[u32],chunk_rows:u32,
selected:[usize]}` with outer `lane:0` and `expected_committed_end:0`. It requires
a fresh admission and both normal contexts idle. The prompt has 1–1048576 tokens,
bounded by the configured model context. Chunk size is 80 through the requested
`batch_tokens`, not any larger AOT-rounded storage capacity. Select 1–48 strictly
increasing absolute rows from the prompt's final 128 rows (or all rows for shorter
prompts). Descriptor `selected` stays absolute; native replay-relative selection
metadata is normalized without copying the device logits.

The controller ends and joins its idle normal lane actors before entering the
stream scope, releasing their split borrows. The native outer loop then calls
retained `NativeTarget::stream_prefill`, borrowing the whole target through encoder
chunks, final-window decoder replay, and the shared consumer-fence result scope.
No other target work or bank command enters until that scope ends; these commands
return BUSY. Polling/cancellation and the standard logits handshake continue.
Normal actors and their retained fences resume afterward, without reloading weights,
allocating another bank or resetting ticket identities.

Stream commit accepts the entire prompt length only. Partial counts are INVALID
and preserve ready/consumed results. Stream cancellation before execute, during
encoder/replay, after failure or after result consumption revokes the entire
admission; its ACK adds `revoked:true`. Discard both ticket and request lease;
never issue a subsequent release for that revoked request. This differs from
normal full-target cancellation, which preserves the request's previous accepted
frontier. The native facade may revoke while reporting execution failure; the
caller retains its request identity until the explicit cancellation receipt. A
new admission receives a new generation. Held/draining external logits still
block cancellation; no timeout bypasses the consumer fence.

Bank snapshots contain `owner`, `capacity_rows`, four `source_page_capacity`,
four `source_pages_free`, `source_payload_bytes` and `cache_bytes`. Every snapshot
reads the attached real bank. The required constructor field
`source_pool_budget_bytes` is the paired compressed-source payload budget,
rounded down to native page groups. Fixed windows and metadata are additional:
`cache_bytes` is the total allocation plan. The retained first GPU probe requested
536870912 bytes, planned 536791040 source-payload bytes and 542270160 total cache
bytes. Capacity is the constructor's actual AOT-rounded row count.

## Capacity query and remaining integration limits

`can_prepare` accepts 1–16 unique idle requests and each request's future append
tokens, including zero; the resulting end must fit the configured model context.
This token count may exceed one execution batch. It calls actual
`BackboneCache::check_append_capacity`, which uses the retained native source-page
reservation/COW rules and then releases those temporary reservations. Only typed
`SourcePoolExhausted` becomes false; stale handles, active work, invalid history
or context bounds remain errors. It does not mirror page arithmetic in Python.

The query retains no reservation. A synchronous scheduler can check all remaining
logical grants together, serialize admission/dispatch, and dispatch both lanes
before allowing unrelated allocation. Arbitrary concurrent unrelated admission
needs an actual native reservation/group-admission extension. This packet makes
no guarantee that a standalone capacity query reserves future pages.

Full-target and encoder-stream/final-decoder C ABI work are implemented. The
retained streaming Rust facade passed GPU smoke tests; the new exclusive C ABI
mode still needs its own live external-consumer qualification. No separate native cache
initialization stage, vLLM prefix-cache mapping, multimodal/dSpark commands or
serving activation is implemented here. No full-vocabulary host-copy operation
is exposed; the existing Rust diagnostic probe is separate from serving.

## Validation

CPU ABI tests use the production C entrypoints, opaque registry, bounded command
channel, controller and two scoped lane actors. Only model/device/fence completion
is stubbed. The fake cache uses retained `SourcePages` reservations, shared prefix
readers, typed exhaustion and physical page-credit release. Tests cover independent
lane completion, delayed consumer fences, cancellation/release/shutdown exclusion,
owned input, failed execution without output publication, old/foreign tickets,
reply consumption, bounded admission, nondestructive buffer-capacity errors,
aggregate capacity exhaustion and corrected accepted-count retries. Native Rust
lease/streaming tests and compile-fail lifetime checks run alongside them.

Validation for this packet: 48 `native_executor` tests passed, including 16 C ABI
cases; five rustdoc lifetime checks passed. Offline checks cover the library,
binaries and examples. The development cdylib builds with the existing lockfile.
The actual package Python client loaded the cdylib, received the asynchronous
missing-native-library initialization failure and explicitly destroyed the exited
owner on CPU; it did not load a CUDA library or contact expert peers.

On 19 September 2026 at 09:57 AEST the campaign parent ran package `dff621b`
against the full-target C ABI at `0dad275c` on the fleet: both contexts matched
full-vocabulary prefill/decode golden hashes, Torch views shared native logits
storage, real consumer streams completed through native event ACKs, held-result
commit guards worked, canceling the second decode left ends `[11,10]`, source
credits returned, and close completed. This qualifies that bounded full-target
external-consumer path. The new encoder-stream C ABI mode still needs a live
probe; its preceding retained Rust streaming facade already passed GPU tests.
No throughput or general vLLM serving claim is made from these correctness probes.

Streaming CPU cases use the actual mode switch, controller, result-scope fence,
and native `SourcePages` in the fake device. They prove occupied-context rejection,
full-target → stream → decode reentry, cancellation before execute/during encode/
during replay, failed-stage revocation receipts, source-credit restoration,
delayed external consumer exclusion, absolute selected-row metadata, non-destructive
partial-commit rejection, invalid-input admission preservation, and a full
1048576-token command through the C boundary. These complement the retained
`StreamSession` lifecycle tests; they do not simulate GPU throughput.
