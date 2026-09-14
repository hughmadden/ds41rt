//! Counters and gauges, exported under `host_cache` whether or not the cache is on. Counters are
//! monotonic; gauges are the current state. Single-threaded: the daemon publishes a `Snapshot`
//! per tick for its metrics endpoint.
use serde::Serialize;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Snapshot {
    pub stores_issued: u64,
    pub stores_completed: u64,
    pub stores_failed: u64,
    pub stores_skipped: u64,
    pub store_bytes: u64,
    pub pages_copied: u64,
    pub pages_shared: u64,
    pub lookups: u64,
    pub host_hits: u64,
    pub restores: u64,
    pub restore_bytes: u64,
    pub restore_timeouts: u64,
    pub restore_failures: u64,
    /// Latency histogram of completed restores in nanoseconds: bounds `RESTORE_BUCKETS_NS`.
    pub restore_latency_buckets: [u64; RESTORE_BUCKETS_NS.len() + 1],
    pub restore_latency_sum_ns: u64,
    pub evict_waits: u64,
    pub evict_wait_ns: u64,
    pub evict_drops_uncached: u64,
    pub host_evictions: u64,
    pub host_evicted_bytes: u64,
    pub resident_snapshots: u64,
    pub bytes_used: u64,
    pub quota_bytes: u64,
}

/// Upper bounds of the restore latency buckets (1 ms … 1 s), plus an overflow bucket.
pub const RESTORE_BUCKETS_NS: [u64; 7] = [
    1_000_000,
    5_000_000,
    10_000_000,
    50_000_000,
    100_000_000,
    500_000_000,
    1_000_000_000,
];

/// The live counters. `record_restore` places a latency in its bucket; everything else is a
/// plain increment so the facade's code reads as what it counts.
#[derive(Clone, Debug, Default)]
pub struct Metrics {
    snapshot: Snapshot,
}

impl Metrics {
    pub fn snapshot(&self) -> Snapshot {
        self.snapshot
    }
    pub fn get_mut(&mut self) -> &mut Snapshot {
        &mut self.snapshot
    }
    pub fn record_restore(&mut self, latency_ns: u64, bytes: u64) {
        let s = &mut self.snapshot;
        s.restores += 1;
        s.restore_bytes += bytes;
        s.restore_latency_sum_ns += latency_ns;
        let bucket = RESTORE_BUCKETS_NS
            .iter()
            .position(|&bound| latency_ns <= bound)
            .unwrap_or(RESTORE_BUCKETS_NS.len());
        s.restore_latency_buckets[bucket] += 1;
    }
}
