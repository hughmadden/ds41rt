//! The cache facade (packet HC-5): the five calls the engine's scheduler makes, all on its own
//! thread, none of which blocks except the two whose whole purpose is to wait a bounded time
//! (`before_device_evict`, `restore`).
//!
//! Life of a snapshot: the engine retains it → `store` plans slabs and enqueues device→host
//! copies on the store stream, returning a ticket → the engine keeps the device snapshot alive
//! while the ticket is pending → `tick` reports completion and the snapshot becomes
//! lookup-visible → the engine may evict it from the device (`before_device_evict` confirms the
//! copy is done or waits within budget) → a later device-bank miss consults `lookup` → on a hit
//! the engine reserves device memory and calls `restore`, which copies host→device into the
//! engine's destinations and waits within budget → the engine applies its reservation and
//! inserts the rebuilt snapshot into its bank.
//!
//! With `StoreMode::OnEvict`, `store` only records the snapshot and the copy is issued by
//! `before_device_evict`, which then waits within the copy budget.
use crate::config::Config;
use crate::copy::{CopyEngine, DeviceRange};
use crate::metrics::{Metrics, Snapshot as MetricsSnapshot};
use crate::pool::Layout;
use crate::snapshot::{DevicePageId, Hit, Key, SnapshotMeta};
use crate::COMPRESSORS;
use serde::Serialize;

/// One device page as the engine addresses it: identity plus where its bytes are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DevicePage {
    pub id: DevicePageId,
    pub range: DeviceRange,
}

/// A retained snapshot as it sits on the device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceSnapshot {
    pub meta: SnapshotMeta,
    pub pages: [Vec<DevicePage>; COMPRESSORS],
    pub tail: DeviceRange,
    pub draft: Option<DeviceRange>,
    pub scores: DeviceRange,
}

/// Where a restore writes: the engine's reserved destinations, in the same shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RestoreTarget {
    pub pages: [Vec<DeviceRange>; COMPRESSORS],
    pub tail: DeviceRange,
    pub draft: Option<DeviceRange>,
    pub scores: DeviceRange,
}

/// Handle for a store in flight. The engine keeps the device snapshot alive (it is retained
/// anyway) until `tick` lists the ticket as completed or failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct StoreTicket(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum SkipReason {
    TooSmall,
    TooLarge,
    KindOff,
    Exhausted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum StoreOutcome {
    Issued(StoreTicket),
    /// `OnEvict` mode: recorded, nothing copied yet.
    Deferred(StoreTicket),
    Skipped(SkipReason),
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TickReport {
    pub completed: Vec<StoreTicket>,
    pub failed: Vec<StoreTicket>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum EvictDecision {
    /// The host copy is complete: the device snapshot may be dropped.
    Clean,
    /// Waited `ns` within the copy budget and the copy completed.
    WaitedClean { ns: u64 },
    /// The budget ran out or the copy failed: the snapshot leaves the device uncached.
    DroppedUncached,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum RestoreOutcome {
    /// Every part is on the device; the engine applies its reservation.
    Done {
        ns: u64,
        bytes: u64,
    },
    /// The restore budget ran out; the engine cancels its reservation and prefills.
    TimedOut,
    Failed,
}

/// The cache. Invariants: `metrics().bytes_used <= quota`; a snapshot with a restore in flight
/// is never evicted; every ticket is reported exactly once; a `lookup` hit is a `Retention` hit
/// over the resident snapshots; when disabled every call is a no-op returning the neutral value.
pub struct HostCache<E: CopyEngine> {
    _engine: E,
    _private: (),
}

impl<E: CopyEngine> HostCache<E> {
    /// Validates `config`, allocates the pinned pool through `engine` (nothing when disabled).
    pub fn new(config: Config, layout: Layout, engine: E) -> anyhow::Result<Self> {
        let _ = (config, layout, engine);
        unimplemented!("HC-5")
    }
    pub fn enabled(&self) -> bool {
        unimplemented!("HC-5")
    }
    pub fn store(&mut self, snapshot: &DeviceSnapshot) -> StoreOutcome {
        let _ = snapshot;
        unimplemented!("HC-5")
    }
    /// Poll the store stream; completed stores become lookup-visible.
    pub fn tick(&mut self) -> TickReport {
        unimplemented!("HC-5")
    }
    /// The engine is about to evict this device snapshot. With a pending ticket, wait within the
    /// copy budget (issuing the copy first under `OnEvict`); without one, `Clean`.
    pub fn before_device_evict(
        &mut self,
        ticket: Option<StoreTicket>,
        snapshot: &DeviceSnapshot,
    ) -> EvictDecision {
        let _ = (ticket, snapshot);
        unimplemented!("HC-5")
    }
    pub fn device_page_freed(&mut self, id: DevicePageId) {
        let _ = id;
        unimplemented!("HC-5")
    }
    /// The engine's reuse rule over host snapshots; `None` falls through to prefill.
    pub fn lookup(&mut self, tokens: &[u32]) -> Option<Hit> {
        let _ = tokens;
        unimplemented!("HC-5")
    }
    /// Pin, copy every part into `target`, wait within the restore budget, unpin.
    pub fn restore(&mut self, key: Key, target: &RestoreTarget) -> RestoreOutcome {
        let _ = (key, target);
        unimplemented!("HC-5")
    }
    pub fn metrics(&self) -> MetricsSnapshot {
        unimplemented!("HC-5")
    }
    pub fn config(&self) -> &Config {
        unimplemented!("HC-5")
    }
    pub fn engine_mut(&mut self) -> &mut E {
        unimplemented!("HC-5")
    }
    #[doc(hidden)]
    pub fn metrics_mut(&mut self) -> &mut Metrics {
        unimplemented!("HC-5")
    }
}
