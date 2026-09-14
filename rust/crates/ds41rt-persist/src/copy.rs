//! Copy engine: the one boundary between host memory and device memory.
//!
//! Production (Phase 4) wraps `ds41rt_ffi` streams: `copy_h2d`, `copy_d2h`, `copy_d2d_async`,
//! `cuda_stream_query` / `cuda_stream_synchronize`, with **pinned, engine-owned host staging
//! buffers** (`HostAllocation`) on the host side of every transfer. The stub must keep the same
//! contract: copies enqueued on a stream complete only when the stream is polled or synchronised,
//! in enqueue order per stream, independently across streams; a host buffer or device buffer with
//! a pending copy on any stream may not be freed or read until that stream has drained; an aborted
//! stream drains before its buffers may be reused. Host bytes are read and written only through
//! `HostBuf` handles, never through borrowed slices, because completion happens after the enqueue
//! call returns. The stub carries a bandwidth/latency model so performance suites can run at
//! measured PCIe rates (RTX 5090 host↔device ≈ 25 GB/s each way; device↔device ≈ 1 TB/s).
use anyhow::Result;

/// Opaque device buffer handle. In the stub it indexes host memory owned by the engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceBuf {
    pub id: u64,
    pub offset: usize,
    pub bytes: usize,
}
impl DeviceBuf {
    pub fn slice(self, offset: usize, bytes: usize) -> DeviceBuf {
        debug_assert!(offset + bytes <= self.bytes);
        DeviceBuf { id: self.id, offset: self.offset + offset, bytes }
    }
}

/// Opaque stream handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Stream(pub u64);

/// Opaque pinned host staging buffer handle, owned by the engine (production: `HostAllocation`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HostBuf {
    pub id: u64,
    pub offset: usize,
    pub bytes: usize,
}
impl HostBuf {
    pub fn slice(self, offset: usize, bytes: usize) -> HostBuf {
        debug_assert!(offset + bytes <= self.bytes);
        HostBuf { id: self.id, offset: self.offset + offset, bytes }
    }
}

pub trait CopyEngine {
    /// Allocate a device buffer of `bytes`.
    fn alloc(&mut self, bytes: usize) -> Result<DeviceBuf>;
    /// Free a device buffer. Fails while any stream has a pending copy touching it.
    fn free(&mut self, buf: DeviceBuf) -> Result<()>;
    /// Allocate a pinned host staging buffer of `bytes`, zero-initialised.
    fn alloc_host(&mut self, bytes: usize) -> Result<HostBuf>;
    /// Free a host buffer. Fails while any stream has a pending copy touching it.
    fn free_host(&mut self, buf: HostBuf) -> Result<()>;
    /// Write into a host buffer synchronously (the producer side of an upload). Fails while a
    /// pending copy touches it.
    fn host_write(&mut self, dst: HostBuf, src: &[u8]) -> Result<()>;
    /// Read a host buffer synchronously (the consumer side of a download). Fails while a pending
    /// copy touches it; after completion it returns exactly the delivered bytes.
    fn host_read(&mut self, src: HostBuf) -> Result<Vec<u8>>;
    fn stream_create(&mut self) -> Result<Stream>;
    /// Enqueue host→device from a host buffer.
    fn h2d(&mut self, dst: DeviceBuf, src: HostBuf, stream: Stream) -> Result<()>;
    /// Enqueue device→host into a host buffer; the bytes are visible through `host_read` only
    /// once the stream has completed the copy.
    fn d2h(&mut self, dst: HostBuf, src: DeviceBuf, stream: Stream) -> Result<()>;
    /// Enqueue device→device.
    fn d2d(&mut self, dst: DeviceBuf, src: DeviceBuf, bytes: usize, stream: Stream) -> Result<()>;
    /// Non-blocking: has every enqueued copy on `stream` completed?
    fn query(&mut self, stream: Stream) -> Result<bool>;
    /// Blocking drain of `stream`.
    fn synchronize(&mut self, stream: Stream) -> Result<()>;
    /// Advance the stub's virtual clock by `nanos`; production ignores it.
    fn advance(&mut self, nanos: u64);
    /// Read device bytes for verification in tests (production: a synchronous d2h through a
    /// scratch host buffer). Shows only completed copies.
    fn read_back(&mut self, buf: DeviceBuf) -> Result<Vec<u8>>;
}

/// Bandwidth/latency model for the stub, in bytes per second and nanoseconds of fixed cost.
#[derive(Clone, Copy, Debug)]
pub struct CopyModel {
    pub h2d_bps: u64,
    pub d2h_bps: u64,
    pub d2d_bps: u64,
    pub launch_ns: u64,
}
impl CopyModel {
    /// RTX 5090 on PCIe 5.0 x16, conservative measured-class figures.
    pub const RTX5090: CopyModel =
        CopyModel { h2d_bps: 25_000_000_000, d2h_bps: 25_000_000_000, d2d_bps: 1_000_000_000_000, launch_ns: 5_000 };
    /// Instant copies for functional tests.
    pub const INSTANT: CopyModel = CopyModel { h2d_bps: u64::MAX, d2h_bps: u64::MAX, d2d_bps: u64::MAX, launch_ns: 0 };
}
