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
use crate::pool::{HostChunk, HostRange, PinnedMemory};
use anyhow::{anyhow, bail, Result};
use serde::Serialize;
use std::collections::BTreeMap;

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

/// Which way a copy moves bytes; selects the bandwidth the model charges.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Direction {
    D2h,
    H2d,
}

/// One end of a copy: a device range or a pinned host range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MemRange {
    Device(DeviceRange),
    Host(HostRange),
}

impl MemRange {
    /// The length of the range, whichever memory it names.
    fn bytes(self) -> usize {
        match self {
            MemRange::Device(range) => range.bytes,
            MemRange::Host(range) => range.bytes,
        }
    }
}

/// A copy waiting for its completion time on the virtual clock. `seq` breaks ties between the
/// two streams so execution order is total and deterministic.
struct PendingCopy {
    stream: Stream,
    direction: Direction,
    src: MemRange,
    dst: MemRange,
    completion_ns: u64,
    seq: u64,
}

/// The virtual-clock engine. Fake device memory is a flat byte array of `device_bytes`; fake
/// host chunks are allocated on demand up to `host_bytes`. `advance` moves the clock and
/// executes every copy whose completion time has passed, in stream order, moving bytes between
/// the fake memories; `wait` advances the clock itself up to the budget.
pub struct StubCopyEngine {
    model: CopyModel,
    device: Vec<u8>,
    host_bytes: usize,
    host_used: usize,
    chunks: Vec<Option<Vec<u8>>>,
    now_ns: u64,
    /// Completion time of the last copy issued on each stream, or zero before the first.
    last_completion: [u64; 2],
    /// Pending copies keyed by `(completion_ns, seq)`, so the first entry is the next to execute.
    pending: BTreeMap<(u64, u64), PendingCopy>,
    pending_count: [usize; 2],
    /// Completion time of each recorded event; `None` is a stalled event that never completes.
    events: Vec<Option<u64>>,
    issue_fault: [bool; 2],
    stall_fault: [bool; 2],
    seq: u64,
}

impl StubCopyEngine {
    /// A stub over `device_bytes` of fake device memory and `host_bytes` of fake pinned host
    /// memory, with the clock at zero and nothing in flight.
    pub fn new(model: CopyModel, device_bytes: usize, host_bytes: usize) -> Self {
        Self {
            model,
            device: vec![0; device_bytes],
            host_bytes,
            host_used: 0,
            chunks: Vec::new(),
            now_ns: 0,
            last_completion: [0; 2],
            pending: BTreeMap::new(),
            pending_count: [0; 2],
            events: Vec::new(),
            issue_fault: [false; 2],
            stall_fault: [false; 2],
            seq: 0,
        }
    }

    /// Move the virtual clock forward by `nanos` and execute every copy whose completion time
    /// has been reached, in completion order. The clock never moves backwards.
    pub fn advance(&mut self, nanos: u64) {
        self.now_ns = self.now_ns.saturating_add(nanos);
        self.execute_due();
    }

    /// Overwrite `range` of the fake device memory with `bytes`; panics if the range is out of
    /// bounds or its length disagrees with `bytes`.
    pub fn write_device(&mut self, range: DeviceRange, bytes: &[u8]) {
        assert_eq!(range.bytes, bytes.len(), "device write length mismatch");
        let start = range.addr as usize;
        let Some(end) = start.checked_add(range.bytes) else {
            panic!("device write range overflows");
        };
        assert!(end <= self.device.len(), "device write out of bounds");
        self.device[start..end].copy_from_slice(bytes);
    }

    /// The bytes currently in `range` of the fake device memory; panics if out of bounds.
    pub fn read_device(&self, range: DeviceRange) -> Vec<u8> {
        let start = range.addr as usize;
        let Some(end) = start.checked_add(range.bytes) else {
            panic!("device read range overflows");
        };
        assert!(end <= self.device.len(), "device read out of bounds");
        self.device[start..end].to_vec()
    }

    /// The bytes currently in `range` of a fake host chunk; panics if the chunk is not allocated
    /// or the range is out of bounds.
    pub fn read_host(&self, range: HostRange) -> Vec<u8> {
        let Some(chunk) = self
            .chunks
            .get(range.chunk as usize)
            .and_then(Option::as_ref)
        else {
            panic!("host chunk {} is not allocated", range.chunk);
        };
        let Some(end) = range.offset.checked_add(range.bytes) else {
            panic!("host read range overflows");
        };
        assert!(end <= chunk.len(), "host read out of bounds");
        chunk[range.offset..end].to_vec()
    }

    /// Arm `fault`; it fires on the next matching operation and is then disarmed.
    pub fn inject(&mut self, fault: CopyFault) {
        match fault {
            CopyFault::IssueFails(stream) => self.issue_fault[stream_index(stream)] = true,
            CopyFault::StreamStalls(stream) => self.stall_fault[stream_index(stream)] = true,
        }
    }

    /// Copies issued but not yet executed, per stream (for the suites' invariants).
    pub fn pending(&self, stream: Stream) -> usize {
        self.pending_count[stream_index(stream)]
    }

    /// Queue one copy: charge the model, advance the stream's tail, and remember the source and
    /// destination so the bytes move when the copy executes.
    fn issue(
        &mut self,
        stream: Stream,
        direction: Direction,
        src: MemRange,
        dst: MemRange,
    ) -> Result<()> {
        let index = stream_index(stream);
        if self.issue_fault[index] {
            self.issue_fault[index] = false;
            bail!("injected issue fault on {stream:?}");
        }
        if src.bytes() != dst.bytes() {
            bail!(
                "copy length mismatch: source {} bytes, destination {} bytes",
                src.bytes(),
                dst.bytes()
            );
        }
        self.check_range(src)?;
        self.check_range(dst)?;
        let rate = match direction {
            Direction::D2h => self.model.d2h_bytes_per_ns,
            Direction::H2d => self.model.h2d_bytes_per_ns,
        };
        let start = self.last_completion[index].max(self.now_ns);
        let completion_ns = start
            .saturating_add(self.model.per_copy_latency_ns)
            .saturating_add(transfer_ns(src.bytes(), rate));
        self.last_completion[index] = completion_ns;
        self.pending_count[index] += 1;
        self.seq += 1;
        self.pending.insert(
            (completion_ns, self.seq),
            PendingCopy {
                stream,
                direction,
                src,
                dst,
                completion_ns,
                seq: self.seq,
            },
        );
        Ok(())
    }

    /// Reject a range that names memory the stub does not have.
    fn check_range(&self, range: MemRange) -> Result<()> {
        match range {
            MemRange::Device(r) => {
                let end = (r.addr as usize)
                    .checked_add(r.bytes)
                    .ok_or_else(|| anyhow!("device range overflows"))?;
                if end > self.device.len() {
                    bail!(
                        "device range {}..{} exceeds {} bytes",
                        r.addr,
                        end,
                        self.device.len()
                    );
                }
            }
            MemRange::Host(r) => {
                let chunk = self
                    .chunks
                    .get(r.chunk as usize)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| anyhow!("host chunk {} is not allocated", r.chunk))?;
                let end = r
                    .offset
                    .checked_add(r.bytes)
                    .ok_or_else(|| anyhow!("host range overflows"))?;
                if end > chunk.len() {
                    bail!(
                        "host range {}..{} exceeds chunk {} of {} bytes",
                        r.offset,
                        end,
                        r.chunk,
                        chunk.len()
                    );
                }
            }
        }
        Ok(())
    }

    /// Execute every pending copy whose completion time the clock has reached, earliest first.
    fn execute_due(&mut self) {
        while let Some((_, copy)) = self.pending.pop_first() {
            if copy.completion_ns > self.now_ns {
                self.pending.insert((copy.completion_ns, copy.seq), copy);
                break;
            }
            self.pending_count[stream_index(copy.stream)] -= 1;
            self.execute(copy);
        }
    }

    /// Move one copy's bytes at execution time. A chunk released before execution is skipped:
    /// the caller broke the hold contract, and the stub has nowhere to write.
    fn execute(&mut self, copy: PendingCopy) {
        match copy.direction {
            Direction::D2h => {
                let (MemRange::Device(src), MemRange::Host(dst)) = (copy.src, copy.dst) else {
                    return;
                };
                let Some(chunk) = self
                    .chunks
                    .get_mut(dst.chunk as usize)
                    .and_then(Option::as_mut)
                else {
                    return;
                };
                let n = src.bytes;
                let start = src.addr as usize;
                chunk[dst.offset..dst.offset + n].copy_from_slice(&self.device[start..start + n]);
            }
            Direction::H2d => {
                let (MemRange::Host(src), MemRange::Device(dst)) = (copy.src, copy.dst) else {
                    return;
                };
                let Some(chunk) = self.chunks.get(src.chunk as usize).and_then(Option::as_ref)
                else {
                    return;
                };
                let n = src.bytes;
                let start = dst.addr as usize;
                self.device[start..start + n].copy_from_slice(&chunk[src.offset..src.offset + n]);
            }
        }
    }

    /// The completion time of a recorded event, or `None` for a stalled one.
    fn event_completion(&self, event: Event) -> Result<Option<u64>> {
        self.events
            .get(event.0 as usize)
            .copied()
            .ok_or_else(|| anyhow!("unknown event {}", event.0))
    }
}

impl PinnedMemory for StubCopyEngine {
    fn allocate_chunk(&mut self, bytes: usize) -> anyhow::Result<crate::pool::HostChunk> {
        let used = self
            .host_used
            .checked_add(bytes)
            .ok_or_else(|| anyhow!("host allocation overflows"))?;
        if used > self.host_bytes {
            bail!(
                "host memory exhausted: {} of {} bytes in use, {} requested",
                self.host_used,
                self.host_bytes,
                bytes
            );
        }
        let id = u32::try_from(self.chunks.len()).map_err(|_| anyhow!("too many host chunks"))?;
        self.chunks.push(Some(vec![0; bytes]));
        self.host_used = used;
        Ok(HostChunk { id, bytes })
    }

    fn release_chunk(&mut self, chunk: crate::pool::HostChunk) -> anyhow::Result<()> {
        let slot = self
            .chunks
            .get_mut(chunk.id as usize)
            .ok_or_else(|| anyhow!("unknown host chunk {}", chunk.id))?;
        let bytes = slot
            .take()
            .ok_or_else(|| anyhow!("host chunk {} already released", chunk.id))?;
        self.host_used -= bytes.len();
        Ok(())
    }
}

impl CopyEngine for StubCopyEngine {
    fn d2h(&mut self, stream: Stream, src: DeviceRange, dst: HostRange) -> anyhow::Result<()> {
        self.issue(
            stream,
            Direction::D2h,
            MemRange::Device(src),
            MemRange::Host(dst),
        )
    }

    fn h2d(&mut self, stream: Stream, src: HostRange, dst: DeviceRange) -> anyhow::Result<()> {
        self.issue(
            stream,
            Direction::H2d,
            MemRange::Host(src),
            MemRange::Device(dst),
        )
    }

    fn record(&mut self, stream: Stream) -> anyhow::Result<Event> {
        let index = stream_index(stream);
        let completion = if self.stall_fault[index] {
            self.stall_fault[index] = false;
            None
        } else {
            Some(self.last_completion[index])
        };
        let id = self.events.len() as u64;
        self.events.push(completion);
        Ok(Event(id))
    }

    fn completed(&mut self, event: Event) -> anyhow::Result<bool> {
        Ok(match self.event_completion(event)? {
            Some(completion) => self.now_ns >= completion,
            None => false,
        })
    }

    fn wait(&mut self, event: Event, budget_ns: u64) -> anyhow::Result<bool> {
        let completion = self.event_completion(event)?;
        let target = match completion {
            Some(completion) => completion.min(self.now_ns.saturating_add(budget_ns)),
            None => self.now_ns.saturating_add(budget_ns),
        };
        if target > self.now_ns {
            self.now_ns = target;
            self.execute_due();
        }
        Ok(matches!(completion, Some(completion) if self.now_ns >= completion))
    }

    fn now_ns(&self) -> u64 {
        self.now_ns
    }
}

/// The slot a stream occupies in the per-stream arrays.
fn stream_index(stream: Stream) -> usize {
    match stream {
        Stream::Store => 0,
        Stream::Restore => 1,
    }
}

/// Nanoseconds to move `bytes` at `bytes_per_ns`, rounded up so a non-empty copy always costs
/// at least one nanosecond.
fn transfer_ns(bytes: usize, bytes_per_ns: f64) -> u64 {
    (bytes as f64 / bytes_per_ns).ceil() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transfer_rounds_up() {
        assert_eq!(transfer_ns(0, 25.0), 0);
        assert_eq!(transfer_ns(1, 25.0), 1);
        assert_eq!(transfer_ns(25, 25.0), 1);
        assert_eq!(transfer_ns(26, 25.0), 2);
        assert_eq!(transfer_ns(crate::PAGE_BYTES, 25.0), 3_646);
    }

    #[test]
    fn stream_slots_are_distinct() {
        assert_ne!(stream_index(Stream::Store), stream_index(Stream::Restore));
    }

    #[test]
    fn default_model_page_cost_is_latency_plus_transfer() {
        let model = CopyModel::default();
        let cost =
            model.per_copy_latency_ns + transfer_ns(crate::PAGE_BYTES, model.d2h_bytes_per_ns);
        assert_eq!(cost, 13_646);
    }

    #[test]
    fn mem_range_reports_its_length() {
        assert_eq!(
            MemRange::Device(DeviceRange { addr: 7, bytes: 11 }).bytes(),
            11
        );
        assert_eq!(
            MemRange::Host(HostRange {
                chunk: 0,
                offset: 3,
                bytes: 5
            })
            .bytes(),
            5
        );
    }
}
