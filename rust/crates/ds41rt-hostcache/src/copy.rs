//! Copy engine (packet HC-3): the only thing the cache asks of the GPU. Two streams (store and
//! restore) carry byte copies between device ranges and pinned host ranges; events mark
//! positions on a stream and are polled or waited on with a budget. The daemon implements it
//! over CUDA; [`StubCopyEngine`] implements it on a virtual clock with a bandwidth and latency
//! model and fake memories, so the suites can check both timing and content.
//!
//! Ordering contract: copies on one stream complete in issue order; the two streams are
//! independent; an event completes when every copy issued on its stream before `record` has
//! completed. A copy reads its source when it *executes*, not when it is issued: the caller must
//! keep source memory alive and unchanged until the event after it completes (the stub models
//! this by copying at completion time, so a violated hold shows up as wrong bytes).
use crate::pool::{HostRange, PinnedMemory};
use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum Stream {
    Store,
    Restore,
}

/// A device byte range: an address the engine understands (a device pointer under CUDA, an
/// offset into the fake device under the stub) and a length.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct DeviceRange {
    pub addr: u64,
    pub bytes: usize,
}

/// A position on a stream. Valid until `completed` returns true or `wait` succeeds; querying a
/// stale event is a logic error.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct Event(pub u64);

pub trait CopyEngine: PinnedMemory {
    fn d2h(&mut self, stream: Stream, src: DeviceRange, dst: HostRange) -> anyhow::Result<()>;
    fn h2d(&mut self, stream: Stream, src: HostRange, dst: DeviceRange) -> anyhow::Result<()>;
    fn record(&mut self, stream: Stream) -> anyhow::Result<Event>;
    /// Non-blocking: has everything before `event` completed?
    fn completed(&mut self, event: Event) -> anyhow::Result<bool>;
    /// Block up to `budget_ns`; true if the event completed within the budget.
    fn wait(&mut self, event: Event, budget_ns: u64) -> anyhow::Result<bool>;
    /// The clock every budget is measured against (monotonic; virtual under the stub).
    fn now_ns(&self) -> u64;
}

/// Bandwidths and latencies the stub models. Defaults are the design's assumptions (25 GB/s
/// each way, 10 µs per copy), replaced by measurements from the fleet when they exist.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct CopyModel {
    pub d2h_bytes_per_ns: f64,
    pub h2d_bytes_per_ns: f64,
    pub per_copy_latency_ns: u64,
}

impl Default for CopyModel {
    fn default() -> Self {
        Self {
            d2h_bytes_per_ns: 25.0,
            h2d_bytes_per_ns: 25.0,
            per_copy_latency_ns: 10_000,
        }
    }
}

/// A fault the stub can arm for the next matching operation (exactly once).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CopyFault {
    /// The next `d2h`/`h2d` issue fails.
    IssueFails(Stream),
    /// The next event on the stream never completes (models a wedged stream); `wait` times out.
    StreamStalls(Stream),
}

/// The virtual-clock engine. Fake device memory is a flat byte array of `device_bytes`; fake
/// host chunks are allocated on demand up to `host_bytes`. `advance` moves the clock and
/// executes every copy whose completion time has passed, in stream order, moving bytes between
/// the fake memories; `wait` advances the clock itself up to the budget.
pub struct StubCopyEngine {
    _private: (),
}

impl StubCopyEngine {
    pub fn new(model: CopyModel, device_bytes: usize, host_bytes: usize) -> Self {
        let _ = (model, device_bytes, host_bytes);
        unimplemented!("HC-3")
    }
    pub fn advance(&mut self, nanos: u64) {
        let _ = nanos;
        unimplemented!("HC-3")
    }
    pub fn write_device(&mut self, range: DeviceRange, bytes: &[u8]) {
        let _ = (range, bytes);
        unimplemented!("HC-3")
    }
    pub fn read_device(&self, range: DeviceRange) -> Vec<u8> {
        let _ = range;
        unimplemented!("HC-3")
    }
    pub fn read_host(&self, range: HostRange) -> Vec<u8> {
        let _ = range;
        unimplemented!("HC-3")
    }
    pub fn inject(&mut self, fault: CopyFault) {
        let _ = fault;
        unimplemented!("HC-3")
    }
    /// Copies issued but not yet executed, per stream (for the suites' invariants).
    pub fn pending(&self, stream: Stream) -> usize {
        let _ = stream;
        unimplemented!("HC-3")
    }
}

impl PinnedMemory for StubCopyEngine {
    fn allocate_chunk(&mut self, bytes: usize) -> anyhow::Result<crate::pool::HostChunk> {
        let _ = bytes;
        unimplemented!("HC-3")
    }
    fn release_chunk(&mut self, chunk: crate::pool::HostChunk) -> anyhow::Result<()> {
        let _ = chunk;
        unimplemented!("HC-3")
    }
}

impl CopyEngine for StubCopyEngine {
    fn d2h(&mut self, stream: Stream, src: DeviceRange, dst: HostRange) -> anyhow::Result<()> {
        let _ = (stream, src, dst);
        unimplemented!("HC-3")
    }
    fn h2d(&mut self, stream: Stream, src: HostRange, dst: DeviceRange) -> anyhow::Result<()> {
        let _ = (stream, src, dst);
        unimplemented!("HC-3")
    }
    fn record(&mut self, stream: Stream) -> anyhow::Result<Event> {
        let _ = stream;
        unimplemented!("HC-3")
    }
    fn completed(&mut self, event: Event) -> anyhow::Result<bool> {
        let _ = event;
        unimplemented!("HC-3")
    }
    fn wait(&mut self, event: Event, budget_ns: u64) -> anyhow::Result<bool> {
        let _ = (event, budget_ns);
        unimplemented!("HC-3")
    }
    fn now_ns(&self) -> u64 {
        unimplemented!("HC-3")
    }
}
