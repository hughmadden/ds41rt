//! Disk persistence for DS41RT retained snapshots — host-only crate.
//!
//! Design: `dsv41-flash-tp4-engram/research/afd-persistence-design.md` in the recipes repo.
//! A retained snapshot in the engine (`v41_native_serve/prefix.rs::Saved`) is: shared refcounted
//! source pages in four compressor pools (890 B/token: 5 pages × 256 rows × 356 B per 512-token
//! group), a fixed 2.72 MB backbone tail in a snapshot arena, 203 KB of dSpark rings, and one
//! logit row. This crate persists those objects to local NVMe with the write-behind rules that
//! made the vLLM tier no-tax: store on eviction, never on the request path; an unpressured engine
//! writes nothing; stale entries flush on decode-free steps; misses cost a memory lookup.
//!
//! Every module declares an interface (trait + types + the production semantics it mirrors) and
//! the packets add stub implementations plus functional, performance and concurrency suites.
//! Nothing here links a native library; the engine binding lands in Phase 4 on the daemon side.

// The shared test fixture under `tests/common/` names this crate as `ds41rt_persist`; in the unit
// test build the crate is not otherwise reachable by its own name, so alias it there.
#[cfg(test)]
extern crate self as ds41rt_persist;

pub mod arena;
pub mod copy;
pub mod events;
pub mod metrics;
pub mod object;
pub mod pool;
pub mod restore;
pub mod sched;
pub mod sink;
pub mod storage;
pub mod store;

/// Bytes per compressed source row: 68 B packed index + scales, 256 B FP4 KV values, 32 B scales.
pub const SOURCE_ROW_BYTES: usize = 68 + 256 + 32;
/// Rows per page in every compressor source pool.
pub const PAGE_ROWS: usize = 256;
/// Bytes per page.
pub const PAGE_BYTES: usize = PAGE_ROWS * SOURCE_ROW_BYTES;
/// Pages per 512-token group across the four compressors (three at ratio 2, one at ratio 1).
pub const PAGES_PER_GROUP: usize = 5;
/// Bytes per 512-token group.
pub const GROUP_BYTES: usize = PAGES_PER_GROUP * PAGE_BYTES;
/// Tokens per group.
pub const GROUP_TOKENS: usize = 512;
/// Backbone tail: 40 window rings of 128 × 528 B + 8 B end, plus 4 compressor carries of 4,096 B.
pub const WINDOW_PREFIX_BYTES: usize = 128 * 528 + 8;
pub const COMPRESSOR_PREFIX_BYTES: usize = 4096;
pub const TAIL_BYTES: usize = 40 * WINDOW_PREFIX_BYTES + 4 * COMPRESSOR_PREFIX_BYTES;
/// dSpark rings: three stages of 128 × 528 B.
pub const DRAFT_BYTES: usize = 3 * 128 * 528;
/// Maximum context, and therefore the largest prefix an object can describe.
pub const MAX_CONTEXT_TOKENS: u64 = 1_048_576;
