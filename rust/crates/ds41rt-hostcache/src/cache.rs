//! The cache facade (packet HC-5): the five calls the engine's scheduler makes, all on its own
//! thread, none of which blocks except the two whose whole purpose is to wait a bounded time
//! (`before_device_evict`, `restore`).
//!
//! The engine attaches a payload `P` to every store (its host-side descriptors: image keys,
//! Engram history, logits, window and compressor metadata); the cache hands it back on a hit
//! and drops it when the snapshot is evicted, so the engine keeps no side table.
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
use crate::config::{Config, StoreMode};
use crate::copy::{CopyEngine, DeviceRange, Event, Stream};
use crate::metrics::{Metrics, Snapshot as MetricsSnapshot};
use crate::pool::{HostRange, Layout, SlabPool};
use crate::snapshot::{
    DevicePageId, Hit, HostSnapshot, Key, PageRef, SnapshotMeta, Snapshots, StorePlan,
};
use crate::{SnapshotKind, COMPRESSORS};
use serde::Serialize;
use std::collections::HashMap;

/// One device page as the engine addresses it: identity plus where its bytes are. The engine
/// keeps a page's rows in several device buffers (packed index, index scales, KV values, KV
/// scales), so a page is a list of segments whose lengths sum to the layout's page size; the
/// host slab holds them concatenated in this order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DevicePage {
    pub id: DevicePageId,
    pub segments: Vec<DeviceRange>,
}

/// A retained snapshot as it sits on the device. Tail and draft are segment lists like pages
/// (the draft is three dSpark rings of varying length); each list's bytes must fit its slab and
/// is stored concatenated. `scores` is empty when the layout's scores class is zero.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceSnapshot {
    pub meta: SnapshotMeta,
    pub pages: [Vec<DevicePage>; COMPRESSORS],
    pub tail: Vec<DeviceRange>,
    pub draft: Option<Vec<DeviceRange>>,
    pub scores: Vec<DeviceRange>,
}

/// Where a restore writes: the engine's fresh destinations, in the same shape and with the same
/// segment lengths as the stored snapshot. Pages carry their new device identities so the cache
/// records them as shared after a successful restore (a later store of the same snapshot then
/// copies nothing).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RestoreTarget {
    pub pages: [Vec<DevicePage>; COMPRESSORS],
    pub tail: Vec<DeviceRange>,
    pub draft: Option<Vec<DeviceRange>>,
    pub scores: Vec<DeviceRange>,
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
    /// A part's segments do not fit the slab class the layout gives it.
    Malformed,
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
pub struct HostCache<E: CopyEngine, P> {
    config: Config,
    layout: Layout,
    engine: E,
    /// `None` exactly when the cache is disabled, so no pool is ever allocated through the
    /// engine for a disabled cache.
    snapshots: Option<Snapshots>,
    metrics: Metrics,
    /// The engine's payload per resident snapshot, dropped with the snapshot.
    payloads: HashMap<Key, P>,
    /// Stores in flight, in issue order; `tick` polls the issued ones in that order.
    pending: Vec<PendingStore<P>>,
    next_ticket: u64,
}

/// Where a store in flight is in its life.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PendingState {
    /// Copies are on the store stream; `event` completes when they land.
    Issued { event: Event },
    /// `OnEvict`: the plan is recorded and the copies are issued by `before_device_evict`.
    Deferred,
    /// An issue failed; `tick` reports the ticket and aborts the plan.
    Failed,
}

/// A store in flight: its plan, the device ranges it must copy (kept only while deferred), the
/// payload that becomes resident at commit, the bytes it copies, and its state.
struct PendingStore<P> {
    ticket: StoreTicket,
    plan: StorePlan,
    parts: Option<StoreParts>,
    payload: P,
    bytes: u64,
    state: PendingState,
}

/// The device ranges a store copies, captured at store time so `OnEvict` can issue them later.
struct StoreParts {
    /// Segments per copied page, in `StorePlan::copies` order.
    pages: Vec<Vec<DeviceRange>>,
    tail: Vec<DeviceRange>,
    draft: Option<Vec<DeviceRange>>,
    scores: Vec<DeviceRange>,
}

impl StoreParts {
    /// Capture the ranges `plan` copies from `snapshot`; shared pages are not in `plan.copies`
    /// and so are not captured.
    fn capture(plan: &StorePlan, snapshot: &DeviceSnapshot) -> Self {
        Self {
            pages: plan
                .copies
                .iter()
                .map(|&(compressor, index, _, _)| {
                    snapshot.pages[compressor as usize][index as usize]
                        .segments
                        .clone()
                })
                .collect(),
            tail: snapshot.tail.clone(),
            draft: snapshot.draft.clone(),
            scores: snapshot.scores.clone(),
        }
    }

    /// The bytes this store copies to host.
    fn bytes(&self) -> u64 {
        let pages: usize = self
            .pages
            .iter()
            .map(|segments| segment_bytes(segments))
            .sum();
        let draft = self.draft.as_deref().map_or(0, segment_bytes);
        (pages + segment_bytes(&self.tail) + draft + segment_bytes(&self.scores)) as u64
    }
}

/// A restore's copy list: the host range of every part and the target segments it feeds, plus
/// the `(device identity, host page)` pairs to register as shared on success.
struct RestorePlan<'a> {
    pages: Vec<(HostRange, &'a [DeviceRange])>,
    tail: (HostRange, &'a [DeviceRange]),
    draft: Option<(HostRange, &'a [DeviceRange])>,
    scores: Option<(HostRange, &'a [DeviceRange])>,
    shared: Vec<(DevicePageId, PageRef)>,
    bytes: u64,
}

impl<'a> RestorePlan<'a> {
    /// Match `target` against the resident `snapshot`; `None` when a part's shape or size does
    /// not mirror the stored one. Invariant: every returned host range names a live slab.
    fn build(
        snapshots: &Snapshots,
        snapshot: &HostSnapshot,
        target: &'a RestoreTarget,
        layout: &Layout,
    ) -> Option<Self> {
        let mut pages = Vec::new();
        let mut shared = Vec::new();
        let mut bytes = 0u64;
        for (compressor, list) in snapshot.pages.iter().enumerate() {
            let targets = target.pages.get(compressor)?;
            if targets.len() != list.len() {
                return None;
            }
            for (page, device_page) in list.iter().zip(targets) {
                let host = snapshots.page_location(*page)?;
                let total = segment_bytes(&device_page.segments);
                if total > layout.page {
                    return None;
                }
                bytes += total as u64;
                pages.push((host, device_page.segments.as_slice()));
                shared.push((device_page.id, *page));
            }
        }
        let tail_total = segment_bytes(&target.tail);
        if tail_total > layout.tail {
            return None;
        }
        bytes += tail_total as u64;
        let draft = match (snapshot.draft, target.draft.as_deref()) {
            (None, None) | (None, Some([])) => None,
            (Some(slab), Some(segments)) => {
                let total = segment_bytes(segments);
                if total > layout.draft {
                    return None;
                }
                bytes += total as u64;
                Some((snapshots.location(slab), segments))
            }
            _ => return None,
        };
        let scores = match (snapshot.scores, target.scores.as_slice()) {
            (None, []) => None,
            (Some(slab), segments) => {
                let total = segment_bytes(segments);
                if total > layout.scores {
                    return None;
                }
                bytes += total as u64;
                Some((snapshots.location(slab), segments))
            }
            _ => return None,
        };
        Some(Self {
            pages,
            tail: (snapshots.location(snapshot.tail), target.tail.as_slice()),
            draft,
            scores,
            shared,
            bytes,
        })
    }
}

impl<E: CopyEngine, P> HostCache<E, P> {
    /// Validates `config`, allocates the pinned pool through `engine` (nothing when disabled).
    pub fn new(config: Config, layout: Layout, engine: E) -> anyhow::Result<Self> {
        config.validate()?;
        let mut engine = engine;
        let snapshots = if config.enabled() {
            let chunk_bytes = usize::try_from(config.chunk_bytes)
                .map_err(|_| anyhow::anyhow!("chunk_bytes exceeds usize"))?;
            let pool = SlabPool::new(config.bytes, chunk_bytes, layout, &mut engine)?;
            Some(Snapshots::new(pool))
        } else {
            None
        };
        let mut cache = Self {
            config,
            layout,
            engine,
            snapshots,
            metrics: Metrics::default(),
            payloads: HashMap::new(),
            pending: Vec::new(),
            next_ticket: 0,
        };
        cache.refresh_gauges();
        Ok(cache)
    }

    pub fn enabled(&self) -> bool {
        self.config.enabled()
    }

    /// Plan and issue the store; `payload` travels with the snapshot until it is evicted.
    pub fn store(&mut self, snapshot: &DeviceSnapshot, payload: P) -> StoreOutcome {
        if !self.enabled() {
            return StoreOutcome::Skipped(SkipReason::KindOff);
        }
        if let Some(reason) = self.skip_reason(snapshot) {
            self.metrics.get_mut().stores_skipped += 1;
            return StoreOutcome::Skipped(reason);
        }
        let device_pages: [Vec<DevicePageId>; COMPRESSORS] =
            std::array::from_fn(|c| snapshot.pages[c].iter().map(|page| page.id).collect());
        let plan = match self.snapshots.as_mut() {
            Some(snapshots) => match snapshots.plan_store(snapshot.meta.clone(), &device_pages) {
                Ok(plan) => plan,
                Err(_) => {
                    self.metrics.get_mut().stores_skipped += 1;
                    return StoreOutcome::Skipped(SkipReason::Exhausted);
                }
            },
            None => return StoreOutcome::Skipped(SkipReason::KindOff),
        };
        let ticket = StoreTicket(self.next_ticket);
        self.next_ticket += 1;
        self.metrics.get_mut().stores_issued += 1;
        let (state, parts, bytes) = match self.config.store {
            StoreMode::OnRetain => {
                let parts = StoreParts::capture(&plan, snapshot);
                let bytes = parts.bytes();
                let state = match self.issue_store(&plan, &parts) {
                    Ok(event) => PendingState::Issued { event },
                    Err(_) => PendingState::Failed,
                };
                (state, None, bytes)
            }
            StoreMode::OnEvict => {
                let parts = StoreParts::capture(&plan, snapshot);
                let bytes = parts.bytes();
                (PendingState::Deferred, Some(parts), bytes)
            }
        };
        self.pending.push(PendingStore {
            ticket,
            plan,
            parts,
            payload,
            bytes,
            state,
        });
        self.refresh_gauges();
        match state {
            PendingState::Deferred => StoreOutcome::Deferred(ticket),
            _ => StoreOutcome::Issued(ticket),
        }
    }

    /// The engine's payload for a resident snapshot.
    pub fn payload(&self, key: Key) -> Option<&P> {
        self.payloads.get(&key)
    }

    /// Poll the store stream; completed stores become lookup-visible.
    pub fn tick(&mut self) -> TickReport {
        let mut report = TickReport::default();
        if !self.enabled() {
            return report;
        }
        let now = self.engine.now_ns();
        let mut index = 0;
        while index < self.pending.len() {
            match self.pending[index].state {
                PendingState::Deferred => index += 1,
                PendingState::Failed => {
                    let pending = self.pending.remove(index);
                    let ticket = pending.ticket;
                    self.abort_pending(pending);
                    report.failed.push(ticket);
                }
                PendingState::Issued { event } => match self.engine.completed(event) {
                    Ok(true) => {
                        let pending = self.pending.remove(index);
                        let ticket = pending.ticket;
                        self.commit_pending(pending, now);
                        report.completed.push(ticket);
                    }
                    Ok(false) => break,
                    Err(_) => {
                        let pending = self.pending.remove(index);
                        let ticket = pending.ticket;
                        self.abort_pending(pending);
                        report.failed.push(ticket);
                    }
                },
            }
        }
        self.refresh_gauges();
        report
    }

    /// The engine is about to drop a device snapshot. With a pending ticket, wait within the
    /// copy budget (under `OnEvict`, issue the copy first from the plan recorded at store time,
    /// whose device ranges are still valid because the snapshot is still alive); without a
    /// ticket, `Clean`.
    pub fn before_device_evict(&mut self, ticket: Option<StoreTicket>) -> EvictDecision {
        let decision = self.decide_evict(ticket);
        self.refresh_gauges();
        decision
    }

    /// The body of [`before_device_evict`](Self::before_device_evict), so the gauges are
    /// refreshed on every exit.
    fn decide_evict(&mut self, ticket: Option<StoreTicket>) -> EvictDecision {
        if !self.enabled() {
            return EvictDecision::Clean;
        }
        let Some(ticket) = ticket else {
            return EvictDecision::Clean;
        };
        let Some(index) = self
            .pending
            .iter()
            .position(|pending| pending.ticket == ticket)
        else {
            return EvictDecision::Clean;
        };
        if matches!(self.pending[index].state, PendingState::Deferred) {
            let mut pending = self.pending.remove(index);
            let Some(parts) = pending.parts.take() else {
                self.abort_pending(pending);
                self.metrics.get_mut().evict_drops_uncached += 1;
                return EvictDecision::DroppedUncached;
            };
            match self.issue_store(&pending.plan, &parts) {
                Ok(event) => {
                    pending.state = PendingState::Issued { event };
                    self.pending.insert(index, pending);
                }
                Err(_) => {
                    self.abort_pending(pending);
                    self.metrics.get_mut().evict_drops_uncached += 1;
                    return EvictDecision::DroppedUncached;
                }
            }
        }
        let PendingState::Issued { event } = self.pending[index].state else {
            let pending = self.pending.remove(index);
            self.abort_pending(pending);
            self.metrics.get_mut().evict_drops_uncached += 1;
            return EvictDecision::DroppedUncached;
        };
        let start = self.engine.now_ns();
        let completed = self
            .engine
            .wait(event, self.config.copy_budget_ns)
            .unwrap_or(false);
        let ns = self.engine.now_ns().saturating_sub(start);
        let metrics = self.metrics.get_mut();
        metrics.evict_waits += 1;
        metrics.evict_wait_ns += ns;
        if completed {
            let pending = self.pending.remove(index);
            let now = self.engine.now_ns();
            self.commit_pending(pending, now);
            EvictDecision::WaitedClean { ns }
        } else {
            let pending = self.pending.remove(index);
            self.abort_pending(pending);
            self.metrics.get_mut().evict_drops_uncached += 1;
            EvictDecision::DroppedUncached
        }
    }

    /// The key-space tokens of a resident snapshot (the sequence its radix entry is keyed by).
    pub fn snapshot_tokens(&self, key: Key) -> Option<&[u32]> {
        self.snapshots
            .as_ref()
            .and_then(|snapshots| snapshots.get(key))
            .map(|snapshot| snapshot.meta.tokens.as_slice())
    }

    pub fn device_page_freed(&mut self, id: DevicePageId) {
        if let Some(snapshots) = self.snapshots.as_mut() {
            snapshots.device_page_freed(id);
        }
    }

    /// The engine's reuse rule over host snapshots; `None` falls through to prefill.
    pub fn lookup(&mut self, tokens: &[u32]) -> Option<Hit> {
        if !self.enabled() {
            return None;
        }
        self.metrics.get_mut().lookups += 1;
        let now = self.engine.now_ns();
        let hit = self
            .snapshots
            .as_mut()
            .and_then(|snapshots| snapshots.lookup(tokens, now));
        if hit.is_some() {
            self.metrics.get_mut().host_hits += 1;
        }
        hit
    }

    /// Pin, copy every part into `target`, wait within the restore budget, unpin. On `TimedOut`
    /// the copies may still land later, so the engine must not reuse `target` until it has
    /// synchronized the restore stream itself; the host snapshot stays resident either way.
    pub fn restore(&mut self, key: Key, target: &RestoreTarget) -> RestoreOutcome {
        if !self.enabled() {
            return RestoreOutcome::Failed;
        }
        let plan = {
            let Some(snapshots) = self.snapshots.as_ref() else {
                return RestoreOutcome::Failed;
            };
            let Some(snapshot) = snapshots.get(key) else {
                return RestoreOutcome::Failed;
            };
            match RestorePlan::build(snapshots, snapshot, target, &self.layout) {
                Some(plan) => plan,
                None => {
                    self.metrics.get_mut().restore_failures += 1;
                    return RestoreOutcome::Failed;
                }
            }
        };
        if let Some(snapshots) = self.snapshots.as_mut() {
            snapshots.pin(key);
        }
        let start = self.engine.now_ns();
        let event = match self.issue_restore(&plan) {
            Ok(event) => event,
            Err(_) => {
                self.unpin(key);
                self.metrics.get_mut().restore_failures += 1;
                return RestoreOutcome::Failed;
            }
        };
        let completed = self
            .engine
            .wait(event, self.config.restore_budget_ns)
            .unwrap_or(false);
        let ns = self.engine.now_ns().saturating_sub(start);
        self.unpin(key);
        if completed {
            if let Some(snapshots) = self.snapshots.as_mut() {
                for &(id, page) in &plan.shared {
                    snapshots.register_device_page(id, page);
                }
            }
            self.metrics.record_restore(ns, plan.bytes);
            RestoreOutcome::Done {
                ns,
                bytes: plan.bytes,
            }
        } else {
            self.metrics.get_mut().restore_timeouts += 1;
            RestoreOutcome::TimedOut
        }
    }

    pub fn metrics(&self) -> MetricsSnapshot {
        self.metrics.snapshot()
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn engine_mut(&mut self) -> &mut E {
        &mut self.engine
    }

    #[doc(hidden)]
    pub fn metrics_mut(&mut self) -> &mut Metrics {
        &mut self.metrics
    }

    /// Why `snapshot` is not cached, or `None` when it is.
    fn skip_reason(&self, snapshot: &DeviceSnapshot) -> Option<SkipReason> {
        let tokens = snapshot.meta.tokens.len();
        if tokens < self.config.min_tokens as usize {
            return Some(SkipReason::TooSmall);
        }
        if tokens > self.config.max_tokens as usize {
            return Some(SkipReason::TooLarge);
        }
        if !self.kind_enabled(snapshot.meta.kind) {
            return Some(SkipReason::KindOff);
        }
        if !self.parts_fit(snapshot) {
            return Some(SkipReason::Malformed);
        }
        None
    }

    /// Whether the configured kinds include `kind`.
    fn kind_enabled(&self, kind: SnapshotKind) -> bool {
        match kind {
            SnapshotKind::Prompt => self.config.kinds.prompt,
            SnapshotKind::Turn => self.config.kinds.turn,
        }
    }

    /// Whether every part's segments fit the slab class the layout gives it.
    fn parts_fit(&self, snapshot: &DeviceSnapshot) -> bool {
        let fits = |segments: &[DeviceRange], class: usize| segment_bytes(segments) <= class;
        snapshot.pages.iter().all(|list| {
            list.iter()
                .all(|page| fits(&page.segments, self.layout.page))
        }) && fits(&snapshot.tail, self.layout.tail)
            && snapshot
                .draft
                .as_deref()
                .is_none_or(|draft| fits(draft, self.layout.draft))
            && fits(&snapshot.scores, self.layout.scores)
    }

    /// Issue every copy `plan` owes on the store stream and record its event.
    fn issue_store(&mut self, plan: &StorePlan, parts: &StoreParts) -> anyhow::Result<Event> {
        let Some(snapshots) = self.snapshots.as_ref() else {
            anyhow::bail!("cache is disabled");
        };
        for (index, &(_, _, _, slab)) in plan.copies.iter().enumerate() {
            copy_out(
                &mut self.engine,
                &parts.pages[index],
                snapshots.location(slab),
            )?;
        }
        copy_out(&mut self.engine, &parts.tail, snapshots.location(plan.tail))?;
        if let (Some(slab), Some(segments)) = (plan.draft, parts.draft.as_deref()) {
            copy_out(&mut self.engine, segments, snapshots.location(slab))?;
        }
        if let (Some(slab), segments) = (plan.scores, parts.scores.as_slice()) {
            copy_out(&mut self.engine, segments, snapshots.location(slab))?;
        }
        self.engine.record(Stream::Store)
    }

    /// Issue every copy a restore owes on the restore stream and record its event.
    fn issue_restore(&mut self, plan: &RestorePlan) -> anyhow::Result<Event> {
        for (host, segments) in &plan.pages {
            copy_in(&mut self.engine, segments, *host)?;
        }
        copy_in(&mut self.engine, plan.tail.1, plan.tail.0)?;
        if let Some((host, segments)) = plan.draft {
            copy_in(&mut self.engine, segments, host)?;
        }
        if let Some((host, segments)) = plan.scores {
            copy_in(&mut self.engine, segments, host)?;
        }
        self.engine.record(Stream::Restore)
    }

    /// Make a completed store resident, drop the payload it replaced, and bring the pool under
    /// quota. Invariant: the payload is resident exactly while its snapshot is.
    fn commit_pending(&mut self, pending: PendingStore<P>, now: u64) {
        let PendingStore {
            plan,
            payload,
            bytes,
            ..
        } = pending;
        let copied = plan.copies.len() as u64;
        let total: u64 = plan.pages.iter().map(|list| list.len() as u64).sum();
        let before = self.snapshots.as_ref().map_or(0, Snapshots::len);
        let key = match self.snapshots.as_mut() {
            Some(snapshots) => snapshots.commit_store(plan, now),
            None => return,
        };
        let after = self.snapshots.as_ref().map_or(0, Snapshots::len);
        let replaced = before + 1 - after;
        if replaced > 0 {
            let Self {
                payloads,
                snapshots,
                ..
            } = self;
            if let Some(snapshots) = snapshots.as_ref() {
                payloads.retain(|key, _| snapshots.get(*key).is_some());
            }
        }
        self.payloads.insert(key, payload);
        let metrics = self.metrics.get_mut();
        metrics.stores_completed += 1;
        metrics.stores_replaced += replaced as u64;
        metrics.pages_copied += copied;
        metrics.pages_shared += total - copied;
        metrics.store_bytes += bytes;
        self.evict_to_quota();
    }

    /// Release a failed store's plan and count it.
    fn abort_pending(&mut self, pending: PendingStore<P>) {
        if let Some(snapshots) = self.snapshots.as_mut() {
            snapshots.abort_store(pending.plan);
        }
        self.metrics.get_mut().stores_failed += 1;
    }

    /// Evict in the engine's order until the pool is under `quota`, dropping the payloads of the
    /// evicted snapshots.
    fn evict_to(&mut self, quota: u64) {
        let Some(snapshots) = self.snapshots.as_mut() else {
            return;
        };
        let (evicted, freed) = snapshots.evict_to(quota);
        if evicted.is_empty() {
            return;
        }
        for key in &evicted {
            self.payloads.remove(key);
        }
        let metrics = self.metrics.get_mut();
        metrics.host_evictions += evicted.len() as u64;
        metrics.host_evicted_bytes += freed;
    }

    /// The quota a commit evicts to: two chunks below the pool's quota, so the next plan has
    /// headroom in every class. Without the headroom `evict_to(config.bytes)` could never evict,
    /// because the pool guarantees `bytes_used <= config.bytes`.
    fn evict_to_quota(&mut self) {
        let quota = self
            .config
            .bytes
            .saturating_sub(2 * self.config.chunk_bytes);
        self.evict_to(quota);
    }

    /// Drop one pin from `key`; an unknown key is a no-op.
    fn unpin(&mut self, key: Key) {
        if let Some(snapshots) = self.snapshots.as_mut() {
            snapshots.unpin(key);
        }
    }

    /// Publish the current pool state as the metrics gauges.
    fn refresh_gauges(&mut self) {
        let (resident, bytes) = match self.snapshots.as_ref() {
            Some(snapshots) => (snapshots.len() as u64, snapshots.bytes_used()),
            None => (0, 0),
        };
        let quota = self.config.bytes;
        let metrics = self.metrics.get_mut();
        metrics.resident_snapshots = resident;
        metrics.bytes_used = bytes;
        metrics.quota_bytes = quota;
    }
}

/// The bytes of a segment list.
fn segment_bytes(segments: &[DeviceRange]) -> usize {
    segments.iter().map(|segment| segment.bytes).sum()
}

/// Copy `segments` into `dst` on the store stream, concatenated in order.
fn copy_out<E: CopyEngine>(
    engine: &mut E,
    segments: &[DeviceRange],
    dst: HostRange,
) -> anyhow::Result<()> {
    let mut offset = 0;
    for segment in segments {
        engine.d2h(
            Stream::Store,
            *segment,
            HostRange {
                chunk: dst.chunk,
                offset: dst.offset + offset,
                bytes: segment.bytes,
            },
        )?;
        offset += segment.bytes;
    }
    Ok(())
}

/// Copy `segments` out of `src` on the restore stream, concatenated in order.
fn copy_in<E: CopyEngine>(
    engine: &mut E,
    segments: &[DeviceRange],
    src: HostRange,
) -> anyhow::Result<()> {
    let mut offset = 0;
    for segment in segments {
        engine.h2d(
            Stream::Restore,
            HostRange {
                chunk: src.chunk,
                offset: src.offset + offset,
                bytes: segment.bytes,
            },
            *segment,
        )?;
        offset += segment.bytes;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pool::testing::layout;
    use crate::snapshot::testing::{id, meta, pages, snapshots};
    use crate::SnapshotKind;

    #[test]
    fn segment_bytes_sums_lengths() {
        assert_eq!(segment_bytes(&[]), 0);
        assert_eq!(
            segment_bytes(&[
                DeviceRange { addr: 0, bytes: 3 },
                DeviceRange { addr: 3, bytes: 4 },
            ]),
            7
        );
    }

    #[test]
    fn restore_plan_matches_the_stored_shape() {
        let mut store = snapshots(1 << 30);
        let plan = store
            .plan_store(meta(SnapshotKind::Turn, &[1, 2], false), &pages(&[id(1)]))
            .expect("plan");
        let key = store.commit_store(plan, 0);
        let snapshot = store.get(key).expect("snapshot");
        let layout = layout();
        let host = store.page_location(snapshot.pages[0][0]).expect("page");
        let target = RestoreTarget {
            pages: std::array::from_fn(|compressor| {
                if compressor == 0 {
                    vec![DevicePage {
                        id: id(9),
                        segments: vec![DeviceRange {
                            addr: 0,
                            bytes: layout.page,
                        }],
                    }]
                } else {
                    Vec::new()
                }
            }),
            tail: vec![DeviceRange {
                addr: 0,
                bytes: layout.tail,
            }],
            draft: None,
            scores: vec![],
        };
        let plan = RestorePlan::build(&store, snapshot, &target, &layout).expect("plan");
        assert_eq!(plan.pages.len(), 1);
        assert_eq!(plan.pages[0].0, host);
        assert_eq!(plan.bytes, (layout.page + layout.tail) as u64);
        assert_eq!(plan.shared, vec![(id(9), snapshot.pages[0][0])]);

        // A target with the wrong page count is rejected.
        let mut short = target.clone();
        short.pages[0].clear();
        assert!(RestorePlan::build(&store, snapshot, &short, &layout).is_none());
        // A target whose page does not fit the class is rejected.
        let mut oversized = target.clone();
        oversized.pages[0][0].segments = vec![DeviceRange {
            addr: 0,
            bytes: layout.page + 1,
        }];
        assert!(RestorePlan::build(&store, snapshot, &oversized, &layout).is_none());
    }
}
