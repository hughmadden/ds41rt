//! Restore (Phase 3): longest-prefix lookup on miss, materialise, attach.
//!
//! Materialise = reserve pages in each compressor pool, write the page bytes, apply; take an arena
//! slot and copy the tail; rebuild the prefix handles; then the engine's own restore branches
//! (exact / encoder continuation / encoder-prefix replay) run unchanged. Every byte is checksum-
//! verified before it lands; a failure releases everything it reserved.
#[derive(Clone, Copy, Debug)]
pub struct RestoreConfig {
    pub max_parallel: usize,
    pub prefer_restore_over_store: bool,
}
