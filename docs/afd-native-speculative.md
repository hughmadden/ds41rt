# Retained native dSpark through the target facade

The target facade can optionally construct the original dSpark runtime and return
owned draft proposals for external target verification. The default is disabled.
It uses the same coordinator constructor, shared immutable weights, two target
contexts, one physical target cache and the runtime's three request-indexed draft
windows. It does not run the Rust HTTP scheduler or its sampler.

This packet has CPU ownership/ABI proof and offline builds. Its enabled native GPU
path, target numerical parity, vLLM rejection-sampler integration and speculative
throughput still require separate live qualification. Previous target-only and
encoder-stream GPU passes do not establish those claims.

## Construction and memory

`TargetConfig.dspark: Option<DsparkConfig>` and the equivalent optional C JSON
`dspark` field are absent/null for target-only operation. Enabled configuration is
`{draft_limit:1..5, adaptive:bool, confidence_cutoff:null|p}`. A cutoff must be
finite and in `(0,1]`; adaptive and confidence-only policies are exclusive.
This facade requires at least two request slots when enabled, matching the retained
constructor's two independent draft-chain allocation rule. The five-wide generated
chain is retained; the configured policy selects up to that many draft tokens.

`v41_native_serve::with_components` loads draft weights once and allocates two
`DsparkChain` scratch/stream owners, lane-local main-projection workspaces and three
`DsparkWindow` banks before choosing the source pool. There is no Python draft
model, extra target weight load, alternate target page ledger or copied kernel.
The internal `NativeDraft` trait erases only the borrowed draft-weight lifetime;
its methods delegate to the original `DraftRuntime`.

The existing source-pool budget continues to describe paired compressed-source
payload. `CacheInfo.cache_bytes` describes the native target cache plan, not total
model/process GPU use; optional draft weights and workspaces are additional startup
allocations. Native draft-ring sizing remains `DsparkWindow::device_bytes`: per
stage, `slots * V41DsparkCache::SLOT_BYTES + source_rows * 1024 + 384`, plus native
projection/chain workspaces. No new memory estimates replace that allocator.

## Proposal and sampling contract

Rust callers use:

```text
context.submit_speculative(SpeculativeInput {
  request, expected_committed_end, anchor,
  remaining_output_tokens, placement
}).await -> SpeculativeProposal { ticket, tokens, draft_us }
```

The anchor has already been emitted and has not yet been processed as a target
input. The returned token array contains that anchor followed by zero to five
greedy draft tokens. The request must be generation-valid, in full-target phase,
and have a nonempty committed prefix. Its actual three draft-window frontiers
must match the actual target frontier. One request/lane is claimed before polling.
The proposed extent is bounded by the supplied envelope, draft limit and remaining
model context. A short history or a one-row envelope uses the retained anchor-only
fallback. Each returned row is selected for full-vocabulary target verification;
selected indices are `0..M`, positions are `committed_end..committed_end+M`.
The target owns its own token copy, so the caller cannot modify verification input.

The draft chain uses temperature zero and does not expose a probability distribution.
For vLLM rejection sampling this is a deterministic proposal: `draft_probs=None`,
with probability one on the proposed token. The existing vLLM sampler must apply
target temperature, penalties, filtering and rejection/correction. The native
`verify_dspark_greedy` helper is reused only as a CPU acceptance oracle here; it is
not installed as a replacement sampler. The native confidence/adaptive selectors
may shorten K, using the existing confidence and accepted-route history.

After ordinary `execute` and the existing borrowed-logits consumer fence, commit
receives the accepted **input** prefix: anchor plus accepted draft rows. A correction
or bonus token remains the next unprocessed anchor. The runtime's existing
`DraftRuntime::commit` projects target taps and calls
`TargetPass::commit_with_dspark`, publishing the same accepted prefix to all three
draft windows, target windows/source pages and Engram state. Rejected private rows
never become committed cache. Full-target cancellation preserves the prior joint
frontier. Publication failures follow the retained native revocation rules and
poison this facade; they do not advertise partial success.

Every ordinary full-target commit also seeds/advances dSpark when enabled, including
prefill and anchor-only decode. Taps are the BF16 target inputs of layers 37, 38 and
39, in that order: `rows * 15360` values. Streaming encoder chunks publish source
state internally but do not seed draft state. The final decoder replay commits its
last 128 rows (or the complete shorter prompt) through the same joint path, seeding
the draft rings at the absolute prompt end. The retained ring rule permits a fresh
nonzero-position seed only when it fills the 128-row window. Stream cancellation
revokes the whole target and draft admission after both contexts drain.

The first binding retains synchronous joint main projection/publication using the
original main workspace zero. Proposals have independent lane scratch/readers,
but this commit may stall the owner thread. The original queued lane-local commit
methods remain available for a later measured optimization; this packet does not
claim their overlap or speculative throughput.

## C command ownership and cancellation

The ABI remains version one; declarations and exact fields are in
[`ds41rt_target.h`](../rust/crates/ds41rt-daemon/include/ds41rt_target.h).
`submit_speculative` carries `lane`, `request`, `expected_committed_end`, `anchor`,
`remaining_output_tokens` and `placement`. Its original command ID owns pending
proposal work. The prepared reply is delayed until proposal completion and contains
`ticket`, `tokens`, `draft_us`, `bank`, `committed_end`, `draft_committed_end`.
No target ticket or borrowed device output escapes before it is actually prepared.

`cancel_proposal {command_id: original}` can interrupt that pending command. The
scoped future drops and drains actual chain readers before cancellation is ACKed.
If cancellation wins, its reply is `proposal_cancelled`, with the original command
ID, request and unchanged target/draft ends. The original reply becomes FAILED(6,
`native proposal cancelled`) and must also be collected. If completion wins, the
cancellation reply is `proposal_completed` with the actual ticket; the original
prepared reply remains intact. Collect it and then cancel its ticket normally.
Repeated or replaced identities cannot cancel a new proposal. Timeouts preserve
both command identities and any native ownership until their replies are resolved.

`info(request)`, commit and nonrevoking cancel return authoritative
`draft_committed_end`: null if disabled; logical zero when enabled but all three
windows are unseeded; otherwise the matched native window end. Stream commit also
returns it. This is a readback of the actual windows, not a second frontier ledger.
The existing CUDA logits lease/event protocol is unchanged. A result held by an
external consumer still blocks commit, cancellation, release and shutdown.

## CPU proof and remaining live checks

Tests execute the production `TargetContext`, C registry/channel, two lane actors,
proposal cancellation and result fences. The fake device uses retained
`SourcePages` reservations/publication, canonical dSpark `SlotAccess` readers and
writers, and the same `append_end` rule used by native ring writes. Only projection,
GPU copying and completion are simulated. Existing slot reservation tests remain
on that same extracted implementation; no kernel or codec was copied.

Coverage includes a pending draft while the peer verifies/commits; reader drain
before release/generation reuse; caller token mutation; every greedy mismatch
position and accepted source extent; ready cancellation and anchor-only fallback;
stale/foreign/busy and malformed work; fail-closed drain errors; final-window seed
rules; delayed consumer exclusion; both proposal-cancel race outcomes; and repeated
concurrent completion/cancel arbitration without lost tickets or readers.

Run from `rust/` with the existing lockfile:

```sh
cargo test --offline -p ds41rt-daemon --lib native_executor -- --test-threads=1
cargo test --offline -p ds41rt-daemon --lib v41_dspark_cache::reservation_tests
cargo test --offline -p ds41rt-core dspark_verify
cargo test --offline -p ds41rt-daemon --doc
cargo check --offline -p ds41rt-daemon --lib --bins --examples
cargo build --offline -p ds41rt-daemon --lib
```

The dedicated native owner thread remains mandatory; no borrowed target, draft or
logits object becomes `Send` or escapes the constructor. Before enabling serving,
run a bounded real-GPU proposal/verification/commit probe, including long-prefill
seed, cancellation, both contexts and native credits. Then qualify the actual
vLLM rejection sampler against target-only output and measure matched speculative
throughput. General prefix persistence and multimodal proposal paths remain outside
this packet.

Validation on 19 September 2026 AEST: 61 `native_executor` tests passed (including
21 C ABI cases), two retained dSpark reservation tests passed (two CUDA tests
remained ignored), two retained greedy verifier tests passed, and five rustdoc
lifetime tests passed. Offline library/binary/example checks and the development
cdylib build passed without dependency changes. Package client `9bf0a2b6` loaded
the compiled library with explicit dSpark configuration, consumed the expected
asynchronous missing-library failure and destroyed the exited owner on CPU.
That last check used no native CUDA library or expert peers.
