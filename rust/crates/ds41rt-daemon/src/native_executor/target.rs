//! Scoped target execution over the retained runtime. No HTTP or sampler.
//! A lane owns its batch until accepted publication or synchronous cancellation;
//! output borrows prevent either operation while another consumer holds logits.
use super::RequestHandle;
use anyhow::{ensure, Context as _, Result};
use ds41rt_ffi::Ds41rtDeviceBuffer;
pub use ds41rt_transport::ExpertV2SourceKind as SourceKind;
use std::{cell::{Cell, RefCell}, rc::Rc};

mod native;
pub use native::{with_target, NativeBank, NativeDriver, NativeTarget, TargetConfig, CacheInfo};
pub use native::{StreamInput, StreamingResult};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ticket { pub request: RequestHandle, pub lane: usize, pub id: u64 }

/// Owned input. These are internal batch identities, not HTTP request IDs.
pub struct TargetInput {
    pub request: RequestHandle,
    pub tokens: Vec<u32>,
    pub selected: Vec<usize>,
    pub kind: SourceKind,
    pub placement: u64,
}

/// Selected FP32 vocabulary rows. The device descriptor is deliberately unsafe
/// to export: all external CUDA consumers must drain before this borrow ends.
pub struct Logits<'a> {
    pub rows: usize,
    pub selected: &'a [usize],
    pub positions: &'a [u64],
    device: Option<Ds41rtDeviceBuffer>,
    host: Option<&'a [f32]>,
}
impl<'a> Logits<'a> {
    pub fn host(&self) -> Option<&'a [f32]> { self.host }
    /// # Safety
    /// The descriptor may not be retained or used after this borrow. Drain all
    /// asynchronous CUDA consumers before dropping it; it does not own storage.
    pub unsafe fn device_buffer(&self) -> Result<Ds41rtDeviceBuffer> {
        self.device.context("logits are not on a native device")
    }
}

/// Private scheduling leases only: physical pages and committed positions belong
/// exclusively to the attached bank. No cached copy of either is kept here.
#[derive(Default)]
struct Active {
    next: Cell<u64>,
    tickets: RefCell<Vec<Ticket>>,
    poisoned: Cell<bool>,
}
impl Active {
    fn healthy(&self) -> Result<()> { ensure!(!self.poisoned.get(), "target owner poisoned"); Ok(()) }
    fn idle(&self, request: RequestHandle) -> Result<()> {
        self.healthy()?;
        ensure!(!self.tickets.borrow().iter().any(|t| t.request == request), "request has a live target batch");
        Ok(())
    }
    fn claim(&self, request: RequestHandle, lane: usize) -> Result<Ticket> {
        self.idle(request)?;
        ensure!(!self.tickets.borrow().iter().any(|t| t.lane == lane), "target lane busy");
        let id = self.next.get().checked_add(1).context("target ticket exhausted")?;
        self.next.set(id);
        let ticket = Ticket { request, lane, id };
        self.tickets.borrow_mut().push(ticket);
        Ok(ticket)
    }
    fn finish(&self, ticket: Ticket) { self.tickets.borrow_mut().retain(|t| *t != ticket); }
}

/// Implemented by the retained native lane and CPU test device. It is not a
/// second engine: numerical work and cache publication stay behind this seam.
///
/// # Safety
/// `execute` must publish outputs only on completed success. Dropping its future
/// must revoke pending outputs; `drain` must finish every device/transport read
/// and write before returning (including errors). `commit` must publish only the
/// accepted rows in the authoritative bank. Logits must borrow real lane storage.
/// Implementations must remain on their constructing thread.
#[allow(async_fn_in_trait)]
pub unsafe trait TargetDriver {
    type Batch;
    fn validate(&self, input: &TargetInput) -> Result<()>;
    fn prepare(&mut self, input: &TargetInput) -> Result<Self::Batch>;
    async fn execute(&mut self, batch: &mut Self::Batch, input: &TargetInput) -> Result<()>;
    fn logits<'a>(&'a self, batch: &'a Self::Batch) -> Result<Logits<'a>>;
    fn commit(&mut self, batch: &mut Self::Batch, accepted: u32) -> Result<u64>;
    fn drain(&mut self, batch: &mut Self::Batch) -> Result<()>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase { Prepared, Executing, Ready, Failed }
struct Job<B> { ticket: Ticket, input: TargetInput, batch: B, phase: Phase }

/// One independently progressing native context. `Rc` intentionally makes it
/// neither Send nor Sync. Dropping an execution future leaves a cancellable job;
/// no result or accepted rows become visible from that incomplete execution.
pub struct TargetContext<D: TargetDriver> {
    driver: D,
    active: Rc<Active>,
    lane: usize,
    capacity: usize,
    head_capacity: usize,
    job: Option<Job<D::Batch>>,
}
impl<D: TargetDriver> TargetContext<D> {
    fn new(driver: D, active: Rc<Active>, lane: usize, capacity: usize, head_capacity: usize) -> Self {
        Self { driver, active, lane, capacity, head_capacity, job: None }
    }
    pub fn submit(&mut self, input: TargetInput) -> Result<Ticket> {
        self.active.healthy()?;
        ensure!(self.job.is_none(), "target lane busy");
        ensure!(!input.tokens.is_empty() && input.tokens.len() <= self.capacity, "invalid target row count");
        ensure!(!input.selected.is_empty() && input.selected.len() <= self.head_capacity
            && input.selected.iter().all(|&row| row < input.tokens.len()), "invalid selected logits rows");
        ensure!(input.selected.windows(2).all(|w| w[0] < w[1]), "selected logits rows must increase");
        self.driver.validate(&input)?;
        let ticket = self.active.claim(input.request, self.lane)?;
        let batch = match self.driver.prepare(&input) {
            Ok(batch) => batch,
            Err(error) => { self.active.finish(ticket); return Err(error); }
        };
        self.job = Some(Job { ticket, input, batch, phase: Phase::Prepared });
        Ok(ticket)
    }
    fn check(&self, ticket: Ticket) -> Result<()> {
        self.active.healthy()?;
        ensure!(self.job.as_ref().is_some_and(|job| job.ticket == ticket), "foreign or stale target ticket");
        Ok(())
    }
    pub async fn execute(&mut self, ticket: Ticket) -> Result<()> {
        self.check(ticket)?;
        let job = self.job.as_mut().unwrap();
        ensure!(job.phase == Phase::Prepared, "target batch already executed");
        job.phase = Phase::Executing;
        let result = self.driver.execute(&mut job.batch, &job.input).await;
        job.phase = if result.is_ok() { Phase::Ready } else { Phase::Failed };
        result
    }
    /// A same-lane mutation cannot invalidate a borrowed output.
    /// ```compile_fail,E0502
    /// use ds41rt_daemon::native_executor::target::{TargetContext, TargetDriver, Ticket};
    /// fn premature_reuse<D: TargetDriver>(context: &mut TargetContext<D>, ticket: Ticket) {
    ///     let logits = context.logits(ticket).unwrap();
    ///     context.cancel(ticket).unwrap();
    ///     drop(logits);
    /// }
    /// ```
    pub fn logits(&self, ticket: Ticket) -> Result<Logits<'_>> {
        self.check(ticket)?;
        let job = self.job.as_ref().unwrap();
        ensure!(job.phase == Phase::Ready, "target logits not complete");
        self.driver.logits(&job.batch)
    }
    pub fn commit(&mut self, ticket: Ticket, accepted: u32) -> Result<u64> {
        self.check(ticket)?;
        let job = self.job.as_mut().unwrap();
        ensure!(job.phase == Phase::Ready, "target execution not complete");
        ensure!(accepted as usize <= job.input.tokens.len(), "accepted rows exceed target batch");
        // Mark before publication so a failure can never expose old logits.
        job.phase = Phase::Failed;
        let end = match self.driver.commit(&mut job.batch, accepted) {
            Ok(end) => end,
            Err(error) => { self.active.poisoned.set(true); return Err(error); }
        };
        self.job = None;
        self.active.finish(ticket);
        Ok(end)
    }
    /// Drop the execution future and all logits borrows before calling. This
    /// synchronous boundary drains native consumers before releasing the lease.
    pub fn cancel(&mut self, ticket: Ticket) -> Result<()> {
        ensure!(self.job.as_ref().is_some_and(|job| job.ticket == ticket), "foreign or stale target ticket");
        self.cancel_current()
    }
    fn cancel_current(&mut self) -> Result<()> {
        let Some(mut job) = self.job.take() else { return Ok(()); };
        let result = self.driver.drain(&mut job.batch);
        if result.is_err() { self.active.poisoned.set(true); }
        self.active.finish(job.ticket);
        result
    }
}
impl<D: TargetDriver> Drop for TargetContext<D> {
    fn drop(&mut self) {
        if let Err(error) = self.cancel_current() { tracing::error!(%error, "native target context drain failed"); }
    }
}

#[cfg(test)]
mod tests;
