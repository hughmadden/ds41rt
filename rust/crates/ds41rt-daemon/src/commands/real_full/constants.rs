pub(super) const REAL_DS4_FULL_BLOCKER: &str = "real-ds4-full is not runnable yet: native DeepSeek V4 attention, complete scheduler-driven residual execution, DSpark, and terminal sampling have not all passed readiness gates";

pub(super) const REAL_FULL_PREFLIGHT_DECODE_ROWS: usize = 1;
pub(super) const REAL_FULL_PREFLIGHT_MTP_ROWS: usize = 8;
pub(super) const REAL_FULL_PREFLIGHT_MTP_ACCEPTED_ROWS: usize = 4;
pub(super) const REAL_FULL_PREFLIGHT_PREFILL_ROWS: usize = 512;
pub(super) const REAL_FULL_PREFLIGHT_PREFILL_TOKEN_START: u64 = 512;
pub(super) const REAL_FULL_PREFLIGHT_DECODE_POSITION: u64 = 1024;
pub(super) const REAL_FULL_PREFLIGHT_MTP_TOKEN_START: u64 =
    REAL_FULL_PREFLIGHT_DECODE_POSITION + REAL_FULL_PREFLIGHT_DECODE_ROWS as u64;
pub(super) const REAL_FULL_PREFLIGHT_KV_RESERVATION_ID: u64 = 52;
pub(super) const REAL_FULL_PREFLIGHT_REQUEST_ID: &str = "real-full-preflight";
pub(super) const REAL_FULL_PREFLIGHT_SEQUENCE_ID: &str = "real-full-preflight-sequence";

// Target mHC is per-active-row transformer state, not attention history. The
// qualified layer-major scheduler retains at most 8K rows. Above that point,
// the bounded wavefront admits at most sixteen prefill chunks at once. Keep
// these values shared by the scheduler and the target-storage planner so a
// throughput change cannot silently invalidate the GPU arena bound.
pub(super) const REAL_FULL_LAYER_MAJOR_PREFILL_MAX_ROWS: usize = 8_192;
pub(super) const REAL_FULL_BOUNDED_PREFILL_MAX_ACTIVE_CHUNKS: usize = 16;
// One authoritative decode row plus the largest jointly issued proposal
// window admitted by the target attention path.
pub(super) const REAL_FULL_TARGET_AUXILIARY_MAX_ROWS: usize = 16;
