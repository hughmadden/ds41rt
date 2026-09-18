#ifndef DS41RT_TARGET_H
#define DS41RT_TARGET_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* ABI 1. All calls are thread-safe. Handles/command IDs are opaque, never pointers.
 * Inputs are copied before return. Maximum config/command/reply JSON length: 65536.
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
 * Config (all required): owner, snapshot, native_lib, peers[4], batch_tokens,
 * max_context_tokens, slots, source_pool_budget_bytes. owner is a nonzero fresh
 * incarnation nonce, never reused across process restarts. source budget means
 * paired source payload, rounded down to native page groups; reported cache_bytes
 * additionally includes fixed windows and metadata. No second KV bank is created.
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
 * execute {ticket}, poll {ticket}, acquire_logits {ticket}
 * release_logits {ticket,lease:u64,consumer_stream:"0x..."}
 * commit {ticket,accepted:u32}, cancel {ticket}, release {request}, shutdown {}
 * ticket is the exact {request,lane,id} returned by submit. Phase-tagged work
 * allows later encoder/stream/replay descriptors; unsupported phases are rejected.
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
