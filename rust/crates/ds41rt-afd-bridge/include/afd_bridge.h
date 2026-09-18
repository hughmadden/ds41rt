#ifndef DS41RT_AFD_BRIDGE_H
#define DS41RT_AFD_BRIDGE_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct afd_bridge_handle afd_bridge_handle;
enum afd_bridge_status {
  AFD_OK = 0, AFD_INVALID = 1, AFD_BUSY = 2, AFD_STALE = 3,
  AFD_NOT_READY = 4, AFD_BUFFER_TOO_SMALL = 5, AFD_FAILED = 6,
  AFD_CANCELLED = 7, AFD_INTERNAL = 8
};
enum afd_bridge_state {
  AFD_IDLE = 0, AFD_PENDING = 1, AFD_READY = 2,
  AFD_STATE_FAILED = 3, AFD_STATE_CANCELLED = 4
};

/* ABI 1. Config JSON (UTF-8, no trailing NUL required):
 * {"transport":"tcp", "peers":["127.0.0.1:1", ... exactly 4],
 *  "executors":[1,2,3,4], "capacity_rows":1024, "lanes":2,
 *  "timeout_ms":1000, "max_frame_bytes":67108864, "device":0}
 * transport, peers, executors and capacity_rows are required. Defaults for the
 * other fields are shown. lanes=1 or 2, capacity_rows=1..4096.
 * RoCE loads the existing DS41RT_NATIVE_LIB and host-device-map environment;
 * this must be configured before create. Creation initializes local owners;
 * network bootstrap remains lazy, as in the native Rust client.
 * All methods except close may be called concurrently. close requires exclusive
 * ownership: no simultaneous ABI calls and no subsequent use of its pointer.
 * Input buffers need remain valid only through the call. No callbacks or Rust
 * allocations cross the ABI. Every output pointer must be writable and aligned.
 */
uint32_t afd_bridge_abi_version(void);
int32_t afd_bridge_create(const uint8_t *config_json, size_t len,
                          afd_bridge_handle **out);

/* Copies and validates one canonical native ProtocolV2 request. Native codec,
 * top-6/5120/FP8-K32 or BF16 contract and four executor identities are reused.
 * One pending or uncollected READY result per lane; excess admission is BUSY.
 * BUSY is checked before request parsing/copying, including for malformed input.
 * FAILED/CANCELLED lanes may accept a new generation. Tickets are unique and
 * monotonic within a bridge, including across lanes; every operation rejects a
 * foreign-lane ticket or a ticket superseded by a later submit on its lane.
 */
int32_t afd_bridge_submit(afd_bridge_handle *, uint32_t lane,
                          const uint8_t *frame, size_t len, uint64_t *ticket);
int32_t afd_bridge_poll(afd_bridge_handle *, uint32_t lane, uint64_t ticket,
                        uint32_t *state);

/* READY output: rank-major contiguous BF16 [4,rows,5120], little endian.
 * Required byte count is always written for READY. An insufficient buffer
 * returns BUFFER_TOO_SMALL without consuming the result; NULL/0 is a size query.
 * Successful collect changes the lane to IDLE; a second collect is NOT_READY.
 */
int32_t afd_bridge_collect(afd_bridge_handle *, uint32_t lane, uint64_t ticket,
                           uint8_t *output, size_t capacity, size_t *written);

/* Asynchronous cancellation: poll until CANCELLED or another terminal state.
 * CANCELLED is published only after the operation drops its borrowed pending
 * transport and resets connections. READY can be discarded immediately.
 * Request cancellation does not claim to stop already executing remote GPU work.
 */
int32_t afd_bridge_cancel(afd_bridge_handle *, uint32_t lane, uint64_t ticket);

/* Copies a terminal operation's diagnostic as UTF-8 (no NUL). Size query works
 * like collect and never consumes the diagnostic. ticket=0 returns the last
 * synchronous API error for this handle; lane is then ignored. Creation errors
 * return status only because no handle exists yet.
 */
int32_t afd_bridge_error(afd_bridge_handle *, uint32_t lane, uint64_t ticket,
                         uint8_t *output, size_t capacity, size_t *written);
/* Narrow allocation diagnostic: retained rank-result capacity and number of
 * capacity increases in this lane. Counts do not include codec/input buffers. */
int32_t afd_bridge_buffer_stats(afd_bridge_handle *, uint32_t lane,
                                size_t *capacity_bytes, uint64_t *growths);
int32_t afd_bridge_close(afd_bridge_handle *);
#ifdef __cplusplus
}
#endif
#endif
