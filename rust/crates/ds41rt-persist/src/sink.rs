//! Write-behind sink (Phase 2): hold-on-evict, background drain, pressure gate, stale flush.
//!
//! Rules carried from the vLLM tier: nothing is stored on the request path; an evicted snapshot is
//! held (its pages and arena slot kept alive) until its copy drains; under pool exhaustion the sink
//! releases holds rather than stall admission; while free groups exceed the watermark the sink
//! declines new holds; stale entries flush in place on decode-free steps only.
#[derive(Clone, Copy, Debug)]
pub struct SinkConfig {
    pub hold_cap: usize,
    pub drain_per_step: usize,
    pub free_groups_watermark: usize,
    pub stale_seconds: u64,
    pub flush_per_scan: usize,
}
