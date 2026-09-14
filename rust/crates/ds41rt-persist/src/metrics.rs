//! Counters every phase exposes; the fleet bench asserts on them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Counters {
    pub stores: u64,
    pub store_bytes: u64,
    pub store_declined_pressure: u64,
    pub store_declined_gate: u64,
    pub restores: u64,
    pub restore_bytes: u64,
    pub restore_cancelled: u64,
    pub lookups: u64,
    pub lookup_hits: u64,
    pub lookup_misses_fast: u64,
    pub idle_flush_candidates: u64,
    pub idle_flush_enqueued: u64,
    pub evictions_disk: u64,
    pub checksum_failures: u64,
}
