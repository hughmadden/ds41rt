//! Disabled native executor integration seam. No model/server/GPU factory is exposed.
//! Physical source ownership is the same SourcePages used by native SourceCache.
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::{cell::Cell, collections::BTreeMap, rc::Rc};
use crate::v41_backbone_cache::BackboneCache;
use crate::v41_compressor::source_cache::{IndexPlan, SourcePages, SourcePrefix};
pub use crate::v41_backbone_cache::CacheLease as RequestHandle;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Transaction {
    pub owner: u64,
    pub id: u64,
    pub lane: usize,
    pub request: RequestHandle,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrefixHandle { pub owner: u64, pub id: u64 }
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Command {
    Admit { slot: usize, request_id: u64 },
    Begin { request: RequestHandle, lane: usize, rows: u32 },
    ReservePublish { transaction: Transaction, accepted: u32 },
    Publish { transaction: Transaction },
    Abort { transaction: Transaction },
    Release { request: RequestHandle },
    Drain { request: RequestHandle },
    RetainSourcePrefix { request: RequestHandle },
    RestoreSourcePrefix { prefix: PrefixHandle, slot: usize, request_id: u64 },
    DropSourcePrefix { prefix: PrefixHandle },
    ResetSourcePrefixes,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Envelope {
    pub owner: u64,
    pub epoch: u64,
    pub command_id: u64,
    pub command: Command,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum Outcome {
    Admitted { request: RequestHandle },
    Begun { transaction: Transaction },
    Reserved,
    Pending,
    Published { committed_end: u64 },
    Aborted,
    Closing,
    Released,
    SourcePrefix { prefix: PrefixHandle, committed_end: u64 },
    PrefixesDropped,
    Failed { reason: String, draining: bool },
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestSnapshot {
    pub request: RequestHandle,
    pub request_id: u64,
    pub committed_end: u64,
    pub prepared_end: u64,
    pub readers: usize,
    pub closing: bool,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Snapshot {
    pub owner: u64,
    pub epoch: u64,
    pub source_page_capacity: [usize; 4],
    pub source_pages_free: [usize; 4],
    /// Canonical physical cache plan, not an allocation or a measured process peak.
    pub cache_bytes: usize,
    pub transactions: usize,
    pub source_prefixes: usize,
    pub source_prefix_capacity: usize,
    pub requests: Vec<RequestSnapshot>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Acknowledgement {
    pub command_id: u64,
    pub epoch: u64,
    pub result: Outcome,
    pub snapshot: Snapshot,
}

/// Copy shared tails before any accepted writes. Destinations are native
/// physical source rows (page * 256 + offset), never vLLM block IDs.
#[derive(Clone, Debug)]
pub struct SourceWrites {
    pub source: usize,
    pub first_row: usize,
    pub destinations: Vec<u64>,
    pub tail_copies: Vec<(u32, u32)>,
}
#[derive(Debug)]
pub enum Completion { Pending, Complete, Failed(String) }
/// Native completion evidence belongs to the executor, not external commands.
///
/// # Safety
/// A Complete/Failed return must mean all reads/writes using the transaction's
/// buffers have drained. `drain` must make that true before returning, including
/// drop/error paths. Storage may not be released while a transaction can use it.
/// `retain_storage` must retain the actual physical allocation owner.
/// No implementation is selected by the serving backend in this milestone.
pub unsafe trait CacheDevice {
    type Storage: Clone;
    fn source_page_capacity(&self) -> [usize; 4];
    /// Keep the allocation owner alive when a read lease outlives this facade.
    fn retain_storage(&self) -> Self::Storage;
    fn start(&mut self, transaction: Transaction, writes: Vec<SourceWrites>) -> Result<()>;
    fn poll(&mut self, transaction: Transaction) -> Completion;
    fn cancel(&mut self, transaction: Transaction);
    fn drain(&mut self, transaction: Transaction);
}
struct Request {
    id: u64,
    end: u64,
    closing: bool,
    readers: Rc<Cell<usize>>,
}
struct Batch {
    handle: Transaction,
    proposed: u32,
    accepted: Option<u32>,
    plans: Vec<IndexPlan>,
    started: bool,
    aborting: bool,
}
struct Prefix { end: u64, sources: Vec<SourcePrefix> }
/// Internal executor read lease, intentionally absent from the command schema.
/// It retains the physical source references even after logical cancellation.
pub struct CacheReader<S> {
    count: Rc<Cell<usize>>,
    _sources: Vec<SourcePrefix>,
    _storage: S,
}
impl<S> Drop for CacheReader<S> {
    fn drop(&mut self) { self.count.set(self.count.get() - 1); }
}
/// Metadata owner for the integration contract. One set of SourcePages governs
/// reservations, prefix references and free credits; no mirrored allocator.
pub struct CacheCommands<D: CacheDevice> {
    device: D,
    owner: u64,
    epoch: u64,
    next_id: u64,
    capacity_rows: u32,
    page_capacity: [usize; 4],
    cache_bytes: usize,
    sources: [SourcePages; 4],
    generations: Vec<u64>,
    requests: Vec<Option<Request>>,
    batches: [Option<Batch>; 2],
    prefixes: BTreeMap<u64, Prefix>,
    prefix_capacity: usize,
    last: Option<(Envelope, Acknowledgement)>,
}
impl<D: CacheDevice> CacheCommands<D> {
    /// `owner` is a nonzero, externally minted executor-incarnation ID. It must
    /// not repeat across worker restarts. This creates metadata only.
    pub fn new(owner: u64, slots: usize, source_pages: [usize; 4],
        capacity_rows: u32, prefix_capacity: usize, device: D) -> Result<Self> {
        ensure!(owner != 0 && (1..=4096).contains(&capacity_rows), "invalid owner/capacity");
        ensure!(prefix_capacity <= 4096, "source prefix capacity exceeds bound");
        ensure!(device.source_page_capacity() == source_pages, "physical source capacity differs from ledger");
        let cache_bytes = BackboneCache::device_bytes(slots, source_pages)?;
        let sources = source_pages.into_iter().map(|pages| SourcePages::new(pages, slots))
            .collect::<Result<Vec<_>>>()?.try_into().ok().expect("four source owners");
        Ok(Self { device, owner, epoch: 0, next_id: 1, capacity_rows,
            page_capacity: source_pages, cache_bytes, sources, generations: vec![0; slots],
            requests: (0..slots).map(|_| None).collect(), batches: [None, None],
            prefixes: BTreeMap::new(), prefix_capacity, last: None })
    }
    pub fn snapshot(&self) -> Snapshot {
        Snapshot { owner: self.owner, epoch: self.epoch,
            source_page_capacity: self.page_capacity,
            source_pages_free: std::array::from_fn(|s| self.sources[s].free_pages()),
            cache_bytes: self.cache_bytes, transactions: self.batches.iter().flatten().count(),
            source_prefixes: self.prefixes.len(), source_prefix_capacity: self.prefix_capacity,
            requests: self.requests.iter().enumerate()
                .filter_map(|(slot, r)| r.as_ref().map(|r| {
                    let proposed = self.batches.iter().flatten().find(|b| b.handle.request.slot() == slot)
                        .map_or(0, |b| b.proposed as u64);
                    RequestSnapshot { request: RequestHandle::new(self.owner, slot, self.generations[slot]),
                        request_id: r.id, committed_end: r.end, prepared_end: r.end + proposed,
                        readers: r.readers.get(), closing: r.closing }
                })).collect() }
    }
    fn request(&self, handle: RequestHandle) -> Result<&Request> {
        ensure!(handle.matches(self.owner, &self.generations), "foreign or stale request lease");
        self.requests[handle.slot()].as_ref().context("request released")
    }
    fn live(&self, handle: RequestHandle) -> Result<&Request> {
        let request = self.request(handle)?;
        ensure!(!request.closing, "request closing");
        Ok(request)
    }
    fn batch(&self, handle: Transaction) -> Result<&Batch> {
        self.request(handle.request)?;
        let batch = self.batches.get(handle.lane).and_then(Option::as_ref).context("transaction absent")?;
        ensure!(batch.handle == handle, "foreign or stale transaction");
        Ok(batch)
    }
    fn idle_slot(&self, slot: usize, request_id: u64) -> Result<()> {
        ensure!(slot < self.requests.len() && self.requests[slot].is_none(), "request slot unavailable");
        ensure!(!self.requests.iter().flatten().any(|r| r.id == request_id), "duplicate request ID");
        self.generations[slot].checked_add(1).context("request generation exhausted")?;
        Ok(())
    }
    fn admit(&mut self, slot: usize, request_id: u64) -> Result<RequestHandle> {
        self.idle_slot(slot, request_id)?;
        self.generations[slot] += 1;
        self.requests[slot] = Some(Request { id: request_id, end: 0, closing: false,
            readers: Rc::new(Cell::new(0)) });
        Ok(RequestHandle::new(self.owner, slot, self.generations[slot]))
    }
    /// Called by the executor on the CUDA owner, not by the scheduler's wire input.
    pub fn read(&self, request: RequestHandle) -> Result<CacheReader<D::Storage>> {
        let r = self.live(request)?;
        let sources = self.sources.iter().map(|source| source.retain_prefix(request.slot(),
            source.committed_rows(request.slot())?)).collect::<Result<Vec<_>>>()?;
        let count = Rc::clone(&r.readers);
        count.set(count.get().checked_add(1).context("reader count exhausted")?);
        Ok(CacheReader { count, _sources: sources, _storage: self.device.retain_storage() })
    }
    pub fn execute(&mut self, envelope: Envelope) -> Result<Acknowledgement> {
        if let Some((previous, acknowledgement)) = &self.last {
            if *previous == envelope { return Ok(acknowledgement.clone()); }
            ensure!(envelope.command_id > previous.command_id, "stale or changed command retry");
        }
        ensure!(envelope.owner == self.owner && envelope.epoch == self.epoch,
            "foreign owner or stale cache epoch");
        ensure!(envelope.command_id != 0, "zero command ID");
        let epoch = self.epoch.checked_add(1).context("cache epoch exhausted")?;
        let result = self.apply(&envelope.command)?;
        self.epoch = epoch;
        let ack = Acknowledgement { command_id: envelope.command_id, epoch,
            result, snapshot: self.snapshot() };
        self.last = Some((envelope, ack.clone()));
        Ok(ack)
    }
    fn apply(&mut self, command: &Command) -> Result<Outcome> {
        match *command {
            Command::Admit { slot, request_id } => Ok(Outcome::Admitted { request: self.admit(slot, request_id)? }),
            Command::Begin { request, lane, rows } => {
                let r = self.live(request)?;
                ensure!(lane < 2 && self.batches[lane].is_none(), "compute lane busy or invalid");
                ensure!(!self.batches.iter().flatten().any(|b| b.handle.request == request),
                    "request already has a transaction");
                ensure!((1..=self.capacity_rows).contains(&rows) && r.end + rows as u64 <= 1048576,
                    "batch extent exceeds native geometry");
                let id = self.next_id;
                let next = id.checked_add(1).context("transaction IDs exhausted")?;
                let transaction = Transaction { owner: self.owner, id, lane, request };
                self.batches[lane] = Some(Batch { handle: transaction, proposed: rows, accepted: None,
                    plans: vec![], started: false, aborting: false });
                self.next_id = next;
                Ok(Outcome::Begun { transaction })
            }
            Command::ReservePublish { transaction, accepted } => self.reserve_publish(transaction, accepted),
            Command::Publish { transaction } => {
                let b = self.batch(transaction)?;
                ensure!(!b.aborting && b.accepted.is_some(), "publication not reserved or aborted");
                self.live(transaction.request)?;
                match self.device.poll(transaction) {
                    Completion::Pending => Ok(Outcome::Pending),
                    Completion::Failed(reason) => {
                        self.batches[transaction.lane].take();
                        Ok(Outcome::Failed { reason, draining: false })
                    }
                    Completion::Complete => {
                        let b = self.batches[transaction.lane].take().unwrap();
                        for (source, plan) in self.sources.iter_mut().zip(b.plans) { source.apply(plan); }
                        let request = self.requests[transaction.request.slot()].as_mut().unwrap();
                        request.end += b.accepted.unwrap() as u64;
                        Ok(Outcome::Published { committed_end: request.end })
                    }
                }
            }
            Command::Abort { transaction } => {
                self.batch(transaction)?;
                self.cancel(transaction);
                Ok(if self.finish_abort(transaction) { Outcome::Aborted } else { Outcome::Pending })
            }
            Command::Release { request } => {
                self.request(request)?;
                self.requests[request.slot()].as_mut().unwrap().closing = true;
                let handles: Vec<_> = self.batches.iter().flatten().filter(|b| b.handle.request == request)
                    .map(|b| b.handle).collect();
                for transaction in handles { self.cancel(transaction); }
                self.drain_request(request)
            }
            Command::Drain { request } => {
                ensure!(self.request(request)?.closing, "drain requires released request");
                self.drain_request(request)
            }
            Command::RetainSourcePrefix { request } => {
                let end = self.live(request)?.end;
                ensure!(self.prefixes.len() < self.prefix_capacity, "source prefix capacity exhausted");
                ensure!(!self.batches.iter().flatten().any(|b| b.handle.request == request),
                    "prefix retention requires committed transaction");
                let next = self.next_id.checked_add(1).context("prefix IDs exhausted")?;
                let sources = self.sources.iter().map(|source| source.retain_prefix(request.slot(),
                    source.committed_rows(request.slot())?)).collect::<Result<Vec<_>>>()?;
                let prefix = PrefixHandle { owner: self.owner, id: self.next_id };
                self.prefixes.insert(prefix.id, Prefix { end, sources });
                self.next_id = next;
                Ok(Outcome::SourcePrefix { prefix, committed_end: end })
            }
            Command::RestoreSourcePrefix { prefix, slot, request_id } => {
                self.idle_slot(slot, request_id)?;
                ensure!(prefix.owner == self.owner, "foreign prefix owner");
                let p = self.prefixes.get(&prefix.id).context("source prefix expired")?;
                for (source, retained) in self.sources.iter().zip(&p.sources) {
                    source.validate_restore(slot, retained)?;
                }
                let end = p.end;
                for (source, retained) in self.sources.iter_mut().zip(&p.sources) {
                    source.restore_prefix(slot, retained)?;
                }
                let request = self.admit(slot, request_id)?;
                self.requests[slot].as_mut().unwrap().end = end;
                Ok(Outcome::Admitted { request })
            }
            Command::DropSourcePrefix { prefix } => {
                ensure!(prefix.owner == self.owner && self.prefixes.contains_key(&prefix.id), "foreign or expired prefix");
                self.prefixes.remove(&prefix.id);
                Ok(Outcome::PrefixesDropped)
            }
            Command::ResetSourcePrefixes => { self.prefixes.clear(); Ok(Outcome::PrefixesDropped) }
        }
    }
    fn reserve_publish(&mut self, transaction: Transaction, accepted: u32) -> Result<Outcome> {
        let b = self.batch(transaction)?;
        let r = self.live(transaction.request)?;
        ensure!(b.accepted.is_none() && !b.aborting && accepted <= b.proposed,
            "invalid or duplicate accepted extent");
        ensure!(r.readers.get() == 0, "cache readers have not drained");
        let end = r.end + accepted as u64;
        let slot = transaction.request.slot();
        let mut plans = Vec::with_capacity(4);
        let mut writes = Vec::with_capacity(4);
        for (index, source) in self.sources.iter().enumerate() {
            let ratio = if index == 3 { 1 } else { 2 };
            let old = source.committed_rows(slot)?;
            let new = end as usize / ratio;
            let plan = source.reserve(&[(slot, old, new)])?;
            let destinations = (old..new).map(|row| source.destination(&plan, slot, row))
                .collect::<Result<Vec<_>>>()?;
            writes.push(SourceWrites { source: index, first_row: old, destinations,
                tail_copies: plan.tail_copies() });
            plans.push(plan);
        }
        let b = self.batches[transaction.lane].as_mut().unwrap();
        b.accepted = Some(accepted);
        b.plans = plans;
        b.started = true;
        if let Err(error) = self.device.start(transaction, writes) {
            // Start may have enqueued work. Keep reservations until explicit drain.
            b.aborting = true;
            self.device.cancel(transaction);
            return Ok(Outcome::Failed { reason: error.to_string(), draining: true });
        }
        Ok(Outcome::Reserved)
    }
    fn cancel(&mut self, transaction: Transaction) {
        let b = self.batches[transaction.lane].as_mut().unwrap();
        if !b.aborting {
            b.aborting = true;
            if b.started { self.device.cancel(transaction); }
        }
    }
    fn finish_abort(&mut self, transaction: Transaction) -> bool {
        let b = self.batches[transaction.lane].as_ref().unwrap();
        if b.started && matches!(self.device.poll(transaction), Completion::Pending) { return false; }
        self.batches[transaction.lane].take(); // Native PageReservation drop returns only its pages.
        true
    }
    fn drain_request(&mut self, request: RequestHandle) -> Result<Outcome> {
        let handles: Vec<_> = self.batches.iter().flatten().filter(|b| b.handle.request == request)
            .map(|b| b.handle).collect();
        for transaction in handles {
            if !self.finish_abort(transaction) { return Ok(Outcome::Closing); }
        }
        if self.request(request)?.readers.get() != 0 { return Ok(Outcome::Closing); }
        for source in &mut self.sources { source.release(request.slot())?; }
        self.requests[request.slot()] = None;
        Ok(Outcome::Released)
    }
}
impl<D: CacheDevice> Drop for CacheCommands<D> {
    fn drop(&mut self) {
        for batch in self.batches.iter().flatten() {
            if batch.started { self.device.cancel(batch.handle); self.device.drain(batch.handle); }
        }
        // Drop reservations before request ownership, then prefix references.
        self.batches = [None, None];
        for slot in 0..self.requests.len() {
            for source in &mut self.sources { source.release(slot).expect("drained source slot"); }
        }
    }
}

#[cfg(test)]
mod tests;
