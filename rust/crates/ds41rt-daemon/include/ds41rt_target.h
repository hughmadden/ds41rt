#ifndef DS41RT_TARGET_H
#define DS41RT_TARGET_H
#include <stddef.h>
#include <stdint.h>
#define DS41RT_TARGET_MAX_CONFIG_BYTES ((size_t)65536)
#define DS41RT_TARGET_MAX_REPLY_BYTES ((size_t)65536)
#define DS41RT_TARGET_MAX_COMMAND_BYTES ((size_t)16777216)
#ifdef __cplusplus
extern "C" {
#endif
/* ABI 1. All calls are thread-safe. Handles/command IDs are opaque, never pointers.
 * Inputs are copied before return. Config/reply maximum: 65536 bytes.
 * Command maximum: 16777216 bytes, allowing a whole 1048576-token stream prompt.
 * Status: 0 OK, 1 INVALID, 2 BUSY, 3 STALE, 4 NOT_READY,
 *         5 BUFFER_TOO_SMALL, 6 FAILED, 8 INTERNAL.
 * create starts one native CUDA owner thread. Its initialization reply has ID 0.
 * command returns only queue admission; the asynchronous JSON reply determines
 * whether the operation succeeded. At most 32 uncollected replies are retained.
 * result copies and consumes a reply on OK. BUFFER_TOO_SMALL writes required
 * length and leaves the reply intact. Bytes are UTF-8 JSON, without a terminator.
 * destroy succeeds only after shutdown/initialization failure has finished and
 * the native resource owner has exited. BUSY preserves the handle and resources.
 * Unknown, foreign and destroyed handles/consumed commands return STALE.
 *
 * Required config: owner, snapshot, native_lib, peers[4], batch_tokens,
 * max_context_tokens, slots, source_pool_budget_bytes. owner is a nonzero fresh
 * incarnation nonce, never reused across process restarts. source budget means
 * paired source payload, rounded down to native page groups; reported cache_bytes
 * additionally includes fixed windows and metadata. No second KV bank is created.
 * Optional dspark:null|{draft_limit:1..5,adaptive:bool,confidence_cutoff:null|p}.
 * Absent/null disables draft allocations. p must be finite in (0,1]; adaptive
 * and confidence-only policies are exclusive. Enabled draft requires slots>=2.
 * Draft weights/windows are constructed once by the same retained constructor.
 *
 * Commands (strict JSON object with op):
 * info {request?: {owner,slot,generation}}
 * admit {slot,request_id}
 * can_prepare {work:[{request,tokens:u32}]}: 1..16 unique idle requests;
 * tokens is a future append budget (zero allowed), bounded by model context,
 * independent of per-call row capacity. Reply {state:"capacity",can_prepare:bool,
 * bank:{owner,capacity_rows,source_page_capacity[4],source_pages_free[4],
 * source_payload_bytes,cache_bytes}}. Delegates the real native aggregate append
 * check. It releases temporary native reservations before replying; it does NOT
 * reserve future grants. Only native page exhaustion becomes false. Stale,
 * writing/active, duplicate and invalid requests remain errors. The scheduler
 * must serialize admission/dispatch or add actual native group reservations.
 * submit {lane,request,expected_committed_end,work:{phase:"full_target",
 *         tokens:[u32],selected:[usize],kind:"prefill"|"decode",placement:u64}}
 * Streaming submit uses lane:0, expected_committed_end:0 and
 * work:{phase:"encoder_stream",tokens:[u32],chunk_rows:u32,selected:[usize]}.
 * It requires a fresh admission and both normal contexts idle. chunk_rows is
 * 80..configured batch_tokens (not rounded AOT capacity); selected has 1..48
 * increasing absolute rows in [max(0,prompt_length-128),prompt_length).
 * The whole prompt is 1..max_context_tokens, at most 1048576 tokens.
 * Both contexts are held through encoder chunks, final decoder replay and result
 * consumption. Bank commands/new work are BUSY while this mode is active.
 * Streaming commit requires accepted == whole prompt length. Partial accepted
 * counts are INVALID and preserve the result. Stream cancel (also before execute
 * and after failure) adds revoked:true to its ACK: the entire admission is gone,
 * so discard its request lease and do not release/reuse it. Normal actors resume
 * only after completion/revocation; full_target cancellation is unchanged.
 * submit_speculative {lane,request,expected_committed_end,anchor:u32,
 *                     remaining_output_tokens:usize,placement:u64}
 * Requires enabled dSpark and a committed full-target prefix. anchor is already
 * emitted but not yet processed as input. remaining_output_tokens is 1..1048576;
 * proposals are bounded by that envelope, draft_limit+1 and remaining context.
 * Retained drafts are greedy (q(proposed)=1); no draft probabilities are invented.
 * The original command remains pending while the native chain runs. Reply:
 * {state:"prepared",ticket,tokens:[anchor,drafts...],draft_us,bank,
 *  committed_end,draft_committed_end}. Every actual token row is selected for
 * target verification; logits positions are expected_committed_end+[0..M-1].
 * cancel_proposal {command_id:original_submit_speculative_command}
 * If cancellation wins, readers drain before ACK {state:"proposal_cancelled",
 * proposal_command,request,committed_end,draft_committed_end,bank}; original reply
 * is FAILED(6,"native proposal cancelled") and must also be collected. If native
 * preparation already completed, ACK {state:"proposal_completed",proposal_command,
 * ticket,committed_end,draft_committed_end,bank}; collect original prepared reply
 * then cancel its ticket normally. The completion race never discards the ticket.
 * Full-target/stream commits seed draft windows when enabled. Verification commit
 * accepted counts INPUT rows (anchor+accepted draft prefix), not a correction or
 * bonus token that remains the next unprocessed anchor. All target/draft/Engram
 * publication uses the retained joint native transaction. Ordinary cancel keeps
 * that prior accepted prefix. info(request), commit and nonrevoking cancel ACKs
 * report draft_committed_end from all three native windows: null if disabled,
 * logical zero if enabled but unseeded. Otherwise it must match committed_end.
 * No native sampler replaces vLLM's target sampling/rejection policy.
 *
 * Additive grouped decode commands (singleton commands remain unchanged):
 * submit_batch {lane,placement,members:[{request,expected_committed_end,
 *                tokens:[u32],selected:[usize],kind:"decode"}]}
 * submit_speculative_batch {lane,placement,members:[{request,
 *                expected_committed_end,anchor,remaining_output_tokens}]}
 * Both support 1..8 unique requests in the given canonical order. Full-target
 * grouping is decode only; encoder-stream/prefill use singleton commands. Each
 * group calls ONE retained native TargetPass; member rows are concatenated.
 * Aggregate input rows <= native capacity_rows; aggregate selected rows <=48.
 * Grouped proposals allow different envelopes including K0 (remaining=1).
 * Prepared reply {state:"prepared",ticket,bank,draft_us,members:[{request,
 * committed_end,draft_committed_end,tokens,selected,input_offset,output_offset}]}.
 * selected is local to each member; offsets map concatenated input rows and
 * compact selected output rows. Descriptor.selected adds input_offset; descriptor
 * positions use that member's actual committed_end plus its local selected row.
 * ticket remains {request:first_member,lane,id}; this is opaque identity of the
 * WHOLE group, not permission to release or mutate another manifest member.
 * All members stay claimed through execution, output consumers and publication.
 * execute/poll/acquire_logits/release_logits/cancel operate on that whole ticket.
 * There is one native logits borrow and one CUDA consumer fence for the group.
 * commit_batch {ticket,accepted:[u32]} is mandatory even for one-member groups.
 * The vector must match the canonical member order/count; each entry is an input
 * prefix 0..member.tokens.length. Invalid vectors preserve the ready result.
 * Reply {state:"committed",ticket,bank,members:[{request,committed_end,
 * draft_committed_end}]}. Native lane-local queued target/draft publication drains
 * before this ACK; peer lane/controller continue while this commit is pending.
 * Group cancel reply {state:"cancelled",ticket,bank,members:[{request,
 * revoked:false,committed_end,draft_committed_end}]} retains every prior prefix.
 * Publication failure is FAILED, poisons the owner and drains/revokes the whole
 * group (including zero-accepted members); it never reports partial success.
 * cancel_proposal also accepts a grouped proposal command. Both race ACKs use
 * members:[{request,committed_end,draft_committed_end}] instead of singleton
 * request/frontier fields; proposal_completed includes the recovered ticket.
 * The original command reply must still be collected. No grouped operation
 * creates another physical cache bank or changes whole-step capacity admission.
 *
 * execute {ticket}, poll {ticket}, acquire_logits {ticket}
 * release_logits {ticket,lease:u64,consumer_stream:"0x..."}
 * commit {ticket,accepted:u32}, cancel {ticket}, release {request}, shutdown {}
 * ticket is the exact {request,lane,id} returned by submit. Phase-tagged work
 * supports full_target and encoder_stream; unsupported phases are rejected.
 * A reply is {command_id,ok,result} or {command_id,ok:false,code,error}.
 *
 * Producer writes are complete before ready/acquire replies.
 * acquire_logits returns an explicit lease plus device_pointer (hex string),
 * device_id, bytes, rows, vocabulary=129280, dtype="float32", selected/positions.
 * The real Rust logits borrow stays alive in its lane actor until release finishes.
 * No host copy. The descriptor may only be read on its CUDA device; no mutation.
 * release_logits must be called AFTER all consumer work has been submitted to
 * the supplied CUDA stream. Native records its own event on that stream and polls
 * a private wait stream. The caller keeps its stream alive until the reply, and
 * launches no further reads of the result after submitting release_logits.
 * No external event ownership or caller-supplied ready flag is trusted.
 * Commit/cancel/release/shutdown return BUSY while a result lease is held/draining.
 * A timeout never revokes the lease or frees the storage. Retry/drain explicitly.
 * shutdown requires idle lanes and no other uncollected commands; cancel/drain
 * active work and release result leases first. Close never discards GPU readers.
 *
 * Pointers supplied as buffers must be valid for their stated extent. CUDA stream
 * values must be live in this process and on the descriptor's device. Misuse of
 * raw CUDA handles is outside memory safety guarantees of this C interface.
 */
uint32_t ds41rt_target_abi_version(void);
int32_t ds41rt_target_create(const uint8_t *config, size_t length, uint64_t *handle);
int32_t ds41rt_target_command(uint64_t handle, const uint8_t *json, size_t length, uint64_t *command_id);
int32_t ds41rt_target_result(uint64_t handle, uint64_t command_id, uint8_t *json, size_t capacity, size_t *written);
int32_t ds41rt_target_destroy(uint64_t handle);
#ifdef __cplusplus
}
#endif
#endif
