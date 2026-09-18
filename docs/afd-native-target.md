# Scoped retained target backend

This packet makes the retained native target callable from the Rust library,
without starting its HTTP server or replacing vLLM scheduling and sampling.
Serving integration remains disabled. The retained native Rust probe at
`c6e7bafa` passed two-context GPU prefill, one-token decode and cancellation;
all 129280 logits were byte-identical between contexts and source credits
returned after release. This is a correctness smoke, not a throughput result.
The new C ABI has separate CPU lifetime tests; its GPU/Python consumer path
still needs live qualification.

`native_executor::target::with_target(config, callback)` loads the native library
and coordinator weights using the original one-RTX constructor, then supplies
two real `TargetPass` / `NativeTp4Wave` contexts and one `Requests` owner. The
native CLI calls that same extracted `v41_native_serve::with_components`
constructor. Its weight-loading, workspace sizing, memory planning and native
kernel/transport implementations have not been copied or replaced.

All native owners stay on the constructing thread's stack for the callback's
duration. Higher-ranked callback lifetimes prevent returning a context, bank or
logits borrow after its weights/library have been destroyed. The `Rc` ownership
prevents moving this scope to another thread. Two execution futures can progress
on its current-thread Tokio runtime without holding a mutable bank borrow across
expert/network waits; this uses retained `TargetPass::execute_shared` directly.

The caller must delegate vLLM's physical coordinator-weight and KV allocation
before invoking this factory. The factory is the sole allocator for that native
model instance. It does not attach to existing PyTorch weights or allow vLLM to
allocate another physical KV bank. Its startup allocations are the retained
weights, two context workspaces, two transports, Engram staging and native cache;
the retained constructor also includes the original vision workspace. These
allocations are planned before KV sizing. The explicit target configuration
disables local routed layers, dSpark and prefix/snapshot retention for this first
binding; those are not yet integrated through this interface.

## Callable contract

`TargetConfig` contains an externally minted nonzero `owner` incarnation nonce,
snapshot and native-library paths, four peer addresses, requested `batch_tokens`,
`max_context_tokens`, request `slots` and explicit `cache_bytes`. Despite the
legacy Rust field name, that input is the paired compressed-source payload
budget (`NativeServeArgs.kv_pool_size`), rounded down to native page groups.
Fixed windows and cache metadata are additional. `CacheInfo.source_payload_bytes`
reports actual paired payload; `CacheInfo.cache_bytes` reports the total native
cache allocation plan. The first GPU probe requested 536870912 bytes and reported
536791040 payload bytes / 542270160 total cache bytes. The nonce must
not repeat across worker restarts. The actual BackboneCache owner is bound to
this nonce before its first admission. The CLI's existing
automatic owner allocation is unchanged.

Capacity uses the original constructor's AOT rounding (80 requested rows require
capacity 256 because the retained encoder suffix reservation covers 128 rows).
The compact vocabulary head has capacity 48 in this target-only configuration.
Request tokens and selected rows remain owned until publication or cancellation.

| API | Contract |
| --- | --- |
| `target.bank().admit(slot, request_id)` | Calls actual `Requests::admit`, returning its native `{owner, slot, generation}` lease. |
| `target.split()` | Returns the current-thread runtime, bank handle and two distinct mutable contexts. |
| `context.submit(TargetInput)` | Claims one request/context lease before preparing; owns tokens, strictly increasing selected row indices, source kind and expert-placement identity. Returns `Ticket {request, lane, id}`. |
| `context.execute(ticket).await` | Calls retained full target execution. Incomplete/failed futures publish neither a result nor committed cache extent. |
| `context.logits(ticket)` | Borrows the real compact FP32 device logits and selected row/position metadata. No logits allocation or host copy. |
| `context.download_logits(ticket, rows).await` | Explicit diagnostic D2H copy through the retained downloader; indices address compact output rows. |
| `context.commit(ticket, accepted)` | After logits consumers drain, calls actual `TargetPass::commit` → `Requests::commit` → native window/source/Engram publication; returns authoritative committed extent. Errors poison the integration scope. |
| `context.cancel(ticket)` | After dropping its execution future/output borrows, synchronizes transport, aborts any queued cache publication and discards native private work before releasing the scheduling lease. |
| `bank.release(request)` | Refuses live target batches/readers and delegates native request release, preserving generation validation. |
| `bank.can_prepare(&[(request, tokens)])` | Delegates the actual aggregate append-capacity check. Only typed pool exhaustion yields false; no reservation remains after the query. Requests must be idle and unique; tokens may describe the entire remaining context. |
| `bank.committed_end(request)` / `bank.info()` | Reads actual native cache extent, actual source capacities/free credits, and the constructor's physical byte plan. |

`Logits::device_buffer()` is unsafe because a raw CUDA descriptor cannot express
the external stream's lifetime in Rust. Its caller must retain the logits lease
until every external asynchronous consumer drains. Normal Rust callers cannot
commit, cancel or reuse that context while holding its logits borrow. Another
context can still finish and publish a different request. Dropping a context
cancels and drains its remaining batch. Dropping an execution future alone keeps
the batch claimed until explicit cancellation or context drop; it cannot be
silently reused.

The bank attachment is executable: `NativeBank` borrows the actual `Requests`,
whose `BackboneCache` owns real `SourceCache` allocations and the canonical
`SourcePages`. It owns no alternate page table, free list, committed positions or
prefix dictionary. Only the two scheduling tickets are tracked outside the bank.
Never construct the schema-1 `CacheCommands` metadata prototype beside it.

Full target execution is the C ABI's current bound. The separate
[streaming facade](afd-native-streaming.md) now exposes retained encoder chunks
and final-decoder replay in Rust and passed bounded GPU probes. The actor still
needs a quiescent mode switch to call it while holding both contexts. Prefix
snapshots, image inputs and dSpark transactions are not bound here. The full-target
interface rejects encoder/replay-stage requests. No native sampler is exposed;
the vLLM adapter must consume logits using its existing sampling semantics.

## Dynamic binding

[ABI 1](afd-native-target-abi.md) exposes this constructor through a dependency-free
C shared library. One CUDA owner thread enters `with_target`; two local lane
actors keep their execution futures and borrowed results inside that callback.
The schema-1 `CacheCommands` prototype is not instantiated. All ACK credits and
committed extents come from the actual native bank.

The C ABI covers asynchronous initialization, owned-token submit, execute/poll,
logits leases, native-observed CUDA consumer completion, accepted publication,
cancellation, release and shutdown. No future becomes `Send`, self-referential
or leaked to static lifetime. Python/vLLM serving integration and the C ABI's
live GPU consumer validation remain separate gates.

## CPU proof

The target suite uses the same `TargetContext` lifecycle as the native driver,
with only device computation/transport completion replaced. Its fake physical
rows use retained `SourcePages::reserve/destination/apply`, `SourcePrefix`, COW
tail copies, page references and generations. Tests prove independent lane
progress, borrowed-output request retention, exact accepted extent, failure
without publication, canceled-future drain before reuse, stale/foreign tickets,
owned inputs, admission before preparation, shared-tail preservation and
physical page credits returning only after the final prefix reader drops.

Three compile-fail tests prove same-context mutation cannot invalidate borrowed
logits, a native scope cannot escape its constructor, and it cannot become Send.
An external-crate `no_run` example also compiles a complete two-request
submit/execute/borrow/commit/release sequence. It is never run on the CPU host.
The constructor also rejects invalid geometry before loading any native library.
These are CPU ownership/contract proofs, not GPU or throughput measurements.

Validation: 25 unit cases passed across the target and schema-1 command suites,
plus the existing constructor capacity regression. Four rustdoc checks passed
(three intentional compile failures and one positive external consumer). The
library and CLI pass offline checks and the development-profile library builds
against the existing lockfile; no dependency or lockfile changes are included.

```sh
cd rust
cargo check --offline -p ds41rt-daemon --lib --bins
cargo build --offline -p ds41rt-daemon --lib
cargo test --offline -p ds41rt-daemon --lib native_executor::
cargo test --offline -p ds41rt-daemon --doc native_executor::target
cargo test --offline -p ds41rt-daemon --lib prefill_capacity_tests
```

## Standalone first-GPU smoke

`examples/native_target_probe.rs` calls this library directly. It starts no
server and contains no PyTorch model. Its default token IDs are the campaign's
thinking=false tokenize result for `Reply with exactly: APPLE`:
`[0,128803,19905,418,9045,28,56684,4392,128804,128822]`. Tokenization is external
to the target backend; the probe neither guesses a template nor invokes a new
tokenizer.

```sh
cd rust
cargo test --offline -p ds41rt-daemon --example native_target_probe
cargo build --offline -p ds41rt-daemon --example native_target_probe
target/debug/examples/native_target_probe \
  --snapshot "$NATIVE_TARGET_SNAPSHOT" \
  --native-lib "$NATIVE_TARGET_LIBRARY" \
  --peers "$NATIVE_TARGET_FOUR_PEERS" \
  --owner "$NATIVE_TARGET_FRESH_OWNER" \
  --cache-bytes 536870912 --batch-tokens 80 --steps 1
```

Only the final command loads the model or touches expert peers. Run it in the
parent-scheduled fleet window after releasing the coordinator GPU and confirming
the retained native library and expert-worker artifacts. Build/run provenance
must record this source SHA, build profile, native-library hashes and worker
identities. The current local artifact is a development-profile correctness
probe; its observed times must not be used as a throughput qualification.

Stdout is flushed JSON lines: `start`, `initialized`, `logits`, `published`, and
`complete`; native diagnostics go to stderr. Two distinct requests execute on
the two retained contexts. The probe checks non-aliased device-logit buffers,
selected row/position bindings, finite full-vocabulary FP32 rows, matching greedy
token IDs, authoritative accepted positions, stale leases after release, and
restored actual native source-page credits. It reports each output SHA256 and
inter-context relative L2/max error; bitwise equality is reported, not assumed.
Host diagnostic argmax is only a smoke oracle and is not installed as a serving
sampler. A differing greedy token fails before accepted publication.

`--steps 2` through `16` add bounded one-token decode forwards using the previous
host argmax token. `--cancel-ready` cancels the second request's last completed
batch after logits inspection, proving no accepted advancement for that batch
while the first request still publishes. Both requests are then released and
their source credits checked. Default execution is one prefill forward, with no
warmup or graph-performance claim. The two example tests check host-logit
geometry/finiteness/tie handling and exact default token/step bounds on CPU.

The campaign parent ran the development-profile `c6e7bafa` native probe on
19 September 2026 AEST: two full-target contexts produced byte-identical logits
for the 10-token APPLE input, then for the next greedy decode token. Greedy IDs
were `[21992,4392]`; canceling the second ready decode left accepted extents
`[11,10]`, and releasing both requests restored all source credits. Evidence is
retained in the recipes campaign `native-transplant-20260919/receipts/` under
`native-target-probe-c6e7bafa-*`. This does not establish model quality or speed.
