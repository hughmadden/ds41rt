# Native executor command seam (Packet A, schema 1)

This is a disabled integration component. It does not load weights, allocate GPU
KV, launch an executor or start a server. The daemon's existing module tree is
now a reusable Rust library; its CLI still calls the same command dispatcher.
`native_executor` exposes the ownership/publication seam. The real native
SourceCache and this seam use one canonical SourcePages implementation extracted
from the original reserve/apply/refcount/copy-on-write code, not two cache rules.

An executor incarnation supplies a nonzero owner ID which must never repeat
across worker restarts. Every envelope is `{owner, epoch, command_id, command}`.
Commands are ordered; `epoch` must equal the last acknowledged epoch. Success
increments epoch and returns `{command_id, epoch, result, snapshot}`. An exact
retry of the most recent successful envelope returns the same acknowledgement
without replaying mutation. Older commands, changed retries, wrong owners and
stale epochs fail without mutation. The sender must serialize commands or
resynchronize from the last acknowledgement. This is a typed Rust/serde schema,
not a new network transport or a stable C ABI yet.

The serde command encoding uses an `op` tag, for example:

```json
{"owner":17,"epoch":0,"command_id":1,"command":{"op":"admit","slot":0,"request_id":100}}
```

Results use a `state` tag. `failed` includes `draining`: a failed start may have
enqueued writes and reports `draining:true`, retaining the transaction/pages
until abort/drain completes. A drained write failure reports `draining:false`.
Neither form advances the request's committed extent. Reader guards retain the
actual device storage owner even if they outlive the facade itself.

Request handles serialize native CacheLease `{owner, slot, generation}`.
Transaction handles include owner, monotonic ID, lane and request handle; two
lanes may progress independently. No raw pointer crosses the command boundary.

| Command | Meaning |
| --- | --- |
| `admit {slot, request_id}` | Reserve a fresh request generation, no computed prefix. |
| `begin {request, lane, rows}` | Own one bounded proposed batch/context; admission is not cache publication. |
| `reserve_publish {transaction, accepted}` | After native readers drain, reserve physical source pages for accepted rows. All four reservations succeed or all roll back. Start backend writes against these destinations. |
| `publish {transaction}` | Return pending until backend writes complete; then atomically apply native metadata and advance acknowledged committed extent. |
| `abort {transaction}` | Cancel writes and retain reservations until backend drain; then return pages and release the context. |
| `release {request}` | Revoke new use immediately; defer physical free while readers or transactions survive. |
| `drain {request}` | Finish canceled work and free a revoked request only after all consumers drain. |
| `retain_source_prefix {request}` | Retain immutable source-page references at the committed extent. |
| `restore_source_prefix {prefix, slot, request_id}` | Attach retained source references to a fresh request generation; appending a shared tail uses native COW. |
| `drop_source_prefix {prefix}` / `reset_source_prefixes` | Evict prefix references; active requests continue to own their pages. |

Source-prefix handles are deliberately not complete model checkpoints. Window
ring snapshots, compressor carry, Engram history, dSpark state and payload writes
must be supplied by the native execution/checkpoint integration before vLLM may
advertise a reusable whole-model prefix hit. Packet B may test these primitive
reference rules, but cannot promote a source-prefix acknowledgement to a model
cache hit. Similarly, overlapping encoder chunks of one request still require
the retained per-layer publication algorithm; this initial seam admits disjoint
request transactions on its two contexts and refuses a second transaction for
the same request.

The backend owns completion evidence. External commands cannot set a ready bit
or claim that GPU readers drained. A native-only reader guard holds physical
references through cancellation. The fake backend used by CPU tests controls
write completion, errors and cancellation explicitly. Publication is separate
from completion and free, and no failed write publishes metadata.

The snapshot reports fixed physical source-page capacities/free credits, live
request/transaction/prefix/reader counts and canonical cache byte extents.
Storage bytes are a plan, not an allocation of a second KV pool. Future native
bindings must attach this ledger to the actual SourceCache storage and report
weights, execution scratch and snapshots separately before vLLM reserves VRAM.
Current construction is metadata-only and cannot be selected by the serving
launcher. Sampling and kernel behavior are unchanged.

Future attachment must move or borrow the canonical SourcePages already owned
by the live SourceCache; constructing this metadata owner beside an independent
live bank would violate the one-ledger contract. No such attachment is present
yet. The generic device completion boundary and reader storage leases avoid
`'static` leaks or self-referential owners; its Rc-based ownership stays on the
native computing thread.

Construction validates the backend's physical source-page capacity against the
ledger and accepts an explicit bounded source-prefix capacity (0 disables
retention, maximum 4096). Outstanding transactions are bounded to two, requests
to the native 16-slot limit, and rows to a configured maximum of 4096. Command
replay retains only the last successful acknowledgement.

## Implementation and CPU validation

* `rust/crates/ds41rt-daemon/src/lib.rs` owns the retained module tree and exposes
  the facade; `main.rs` calls the same CLI dispatcher.
* `src/native_executor.rs` defines the schema, completion trait, bounded command
  owner and request/read/prefix lifecycles; `src/native_executor/tests.rs` uses
  controlled fake device writes against native physical row destinations.
* `src/v41_compressor/source_cache/metadata.rs` contains the extracted native
  SourcePages operations. Real SourceCache delegates to it. The existing
  PagePool/PageReservation/SourcePrefix rules remain at their original paths.
* `src/v41_backbone_cache.rs` exports the existing CacheLease identity and shares
  its owner/slot/generation validator with the command facade.

19 CPU tests passed: 14 facade cases, two direct SourcePages cases, and three
retained prefix-ownership/reservation/geometry cases. No selected case requires
a GPU or silently skips for a missing native library. The library and CLI check
offline against the existing lockfile; no dependency or lockfile changes were
needed. Compilation retains the repository's existing warning-heavy native
module tree; no unrelated warning cleanup is included.

```sh
cd rust
cargo check --offline -p ds41rt-daemon --lib --bins
cargo test --offline -p ds41rt-daemon --lib native_executor::tests
cargo test --offline -p ds41rt-daemon --lib v41_compressor::source_cache::metadata::tests
cargo test --offline -p ds41rt-daemon --lib v41_compressor::source_cache::ownership::tests
cargo test --offline -p ds41rt-daemon --lib rollback_returns_only_its_pages_and_preserves_peer_reservation
cargo test --offline -p ds41rt-daemon --lib context_geometry_tests
```

The cases prove delayed and out-of-order independent lane completion, no partial
publication, fourth-source reservation rollback, retained native shared-tail
COW and page generations, accepted-only source extents, stale/foreign identities,
epoch/retry behavior, bounded prefix metadata, physical-capacity agreement,
cancel/start failure, reader-held release, storage lifetime after facade drop,
and repeated lane/slot reuse with restored page credits. They do not prove
native GPU kernels, full checkpoint replay, encoder chunk concurrency within
one prompt, or end-to-end serving parity.
