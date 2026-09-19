# Native multi-request target passes — 19 September 2026 AEST

The singleton facade advanced at most two requests per paired native pass. At
concurrency four, vLLM consequently required two full scheduling steps to advance
all four requests once. The retained Rust scheduler groups all requests assigned
to a lane into one `RequestBatch`; its two passes can advance four requests in the
same wave. Round-robin fairness in the old adapter did not remove this difference.
The observed latency gap remains a measured system result, not a claim that every
millisecond has been attributed to this mechanism.

This additive seam restores actual native grouping. `TargetContext::submit_batch`
and `submit_speculative_batch` own 1–8 unique members per lane. Their `NativeDriver`
prepares one retained `Requests::prepare(&work)`, invokes one
`TargetPass::execute_shared` over concatenated inputs, and exposes the compact
selected FP32 rows as one Rust borrow. No kernel, codec, physical cache allocator,
draft chain, or sampler is duplicated. vLLM still selects requests and performs
its existing target/rejection sampling.

The C contract is in `rust/crates/ds41rt-daemon/include/ds41rt_target.h`. A group
uses the existing opaque ticket shape; `ticket.request` is the first manifest
member, while native scheduling ownership covers **every** member. The prepared
manifest fixes member order, owned tokens, local selected indices, input offsets,
and compact-output offsets. Descriptor positions come from the actual native
request positions. `commit_batch` accepts one input-prefix count per member,
including zero; a single-member group still requires that command. Mixed dSpark
K0 and K>0 envelopes share a verification pass with `MtpVerify` metadata. The old
singleton speculative metadata is also corrected from `Decode` to `MtpVerify`.
Both kinds share the same current target/cache and expert execution paths; RoCE
polling explicitly treats them alike. This metadata correction is not an
explanation or fix for the separately observed cross-shape logits mismatch.

Target and draft publication use the retained lane-local queued transaction:
`begin_queued_commit`, `TargetPass::enqueue_cache_commit`, completion polling and
`finish_queued_commit`. A waiting commit holds its entire group while allowing
the peer lane and controller to run. The result's single native CUDA consumer
fence must drain first. Dropping a Rust commit future leaves a failed job requiring
cancellation/drain; enqueue/publication failure revokes every participating
admission, including zero-accepted members, and poisons the owner. There is no
partial-success ACK. All-zero acceptance retires only temporary pass state and
keeps both target/draft frontiers unchanged. Ordinary cancellation before
publication preserves every prior accepted prefix.

Grouped draft generation calls the retained batched `poll_propose` and existing
confidence/adaptive prefix policy. One command owns every proposal window reader;
command-ID cancellation drains them before replying. If preparation wins the
race, cancellation recovers the same ticket and the original manifest reply stays
collectible. Target and speculative groups share the same actor, logits lease,
consumer fence, and publication handling. Encoder streaming continues to reserve
both contexts and uses the qualified singleton whole-prompt path.

Capacity remains the existing explicit whole-step contract: the scheduler asks
the actual native bank about the aggregate chosen/future work before dispatch;
that query does not retain reservations. These changes do not permit arbitrary
uncoordinated producers to race unrelated capacity checks. Each group respects
the native batch capacity and 48 selected-row head bound. Weights and the physical
cache bank are still constructed once. Per-group host metadata is bounded by
8 members and the existing input/selected capacities; no extra GPU weights, KV
bank or context is allocated.

CPU contract tests use the production C ABI registry, owner thread and actors.
Only device computation/completion is fake. Canonical `SourcePages` aggregate
reservations/publication and `SlotAccess` draft read/write reservations are real.
The tests count two pass calls for four requests, check unequal row/selection
maps, joint acceptance, held-result lifetime, delayed commit with peer progress,
invalid admission/vector rollback, failure/revocation, K0 mixed groups and both
proposal-cancellation races. These are structural and ownership proofs, not a
GPU throughput or numerical-parity claim. The parent campaign separately owns
GPU correctness, the cross-shape numerical failure investigation, and matched
concurrency benchmarks; old 508f1909 artifacts remain retained for comparison.

Validation for this packet (offline, CPU only): 72 native-executor tests passed;
2 retained draft reservation tests passed with 4 GPU-only tests ignored; the
retained flattened draft-prefix test passed; 4 retained source-page tests passed
with 6 GPU-only tests ignored; all 5 library doctests passed. The grouped commit
future-drop test specifically proves that source-page and draft-window write
reservations survive the dropped future until the context explicitly drains.
