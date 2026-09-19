//! Retained two-lane encoder stream followed by one final-window decoder replay.
//! A fresh admission is either fully published or revoked after both lanes drain.
use super::*;
use crate::v41_backbone_cache::CacheWork;
use crate::v41_block::EncoderSuffix;

pub struct StreamInput {
    pub request: RequestHandle,
    pub tokens: Vec<u32>,
    pub chunk_rows: usize,
    /// Absolute prompt rows; only the retained final 128 rows have decoder logits.
    pub selected: Vec<usize>,
}

impl StreamInput {
    pub(crate) fn validate(&self, chunk_rows: usize, max_context: u64) -> Result<()> {
        StreamPlan::new(self, chunk_rows, max_context).map(|_| ())
    }
}

struct StreamPlan {
    request: RequestHandle,
    end: usize,
    start: usize,
    chunk_rows: usize,
    selected: Vec<usize>,
}
impl StreamPlan {
    fn new(input: &StreamInput, capacity: usize, max_context: u64) -> Result<Self> {
        let end = input.tokens.len();
        ensure!(end > 0 && end as u64 <= max_context && end <= 1048576,
            "invalid streaming prompt extent");
        ensure!((80..=capacity).contains(&input.chunk_rows), "invalid streaming chunk size");
        ensure!(input.tokens.iter().all(|&id| id < 129280), "streaming token outside vocabulary");
        let start = end.saturating_sub(128);
        ensure!(!input.selected.is_empty() && input.selected.len() <= 48
            && input.selected.iter().all(|&row| start <= row && row < end)
            && input.selected.windows(2).all(|w| w[0] < w[1]),
            "streaming logits must select increasing final-window rows");
        Ok(Self { request: input.request, end, start, chunk_rows: input.chunk_rows,
            selected: input.selected.iter().map(|row| row - start).collect() })
    }
}

// Only computation is replaceable in the CPU tests. This lifecycle, including
// cancellation after a dropped future, is shared by the real native driver.
trait StreamDriver {
    fn begin(&mut self, plan: &StreamPlan) -> Result<()>;
    async fn encode(&mut self, plan: &StreamPlan, tokens: &[u32], keep_running: &dyn Fn() -> bool) -> Result<()>;
    async fn replay(&mut self, plan: &StreamPlan) -> Result<()>;
    fn logits(&self) -> Result<Logits<'_>>;
    fn commit(&mut self, plan: &StreamPlan) -> Result<u64>;
    fn revoke(&mut self, plan: &StreamPlan) -> Result<()>;
}

struct StreamSession<D: StreamDriver> {
    driver: D,
    plan: StreamPlan,
    tokens: Vec<u32>,
    ready: bool,
    armed: bool,
}
impl<D: StreamDriver> StreamSession<D> {
    async fn execute(&mut self, keep_running: &dyn Fn() -> bool) -> Result<()> {
        ensure!(keep_running(), "stream canceled before dispatch");
        self.driver.begin(&self.plan)?;
        self.driver.encode(&self.plan, &self.tokens, keep_running).await?;
        ensure!(keep_running(), "stream canceled before decoder replay");
        self.driver.replay(&self.plan).await?;
        ensure!(keep_running(), "stream canceled before output publication");
        self.ready = true;
        Ok(())
    }
    fn logits(&self) -> Result<Logits<'_>> {
        ensure!(self.ready, "streaming logits not ready");
        self.driver.logits()
    }
    fn commit(&mut self) -> Result<u64> {
        ensure!(self.ready && self.armed, "streaming result not publishable");
        self.ready = false;
        let end = self.driver.commit(&self.plan)?;
        ensure!(end == self.plan.end as u64, "native streaming accepted extent differs");
        self.armed = false;
        Ok(end)
    }
    fn cancel(&mut self) -> Result<()> {
        self.ready = false;
        if !self.armed { return Ok(()); }
        // Do not run the same failed teardown again from Drop.
        self.armed = false;
        self.driver.revoke(&self.plan)
    }
}
impl<D: StreamDriver> Drop for StreamSession<D> {
    fn drop(&mut self) {
        if let Err(error) = self.cancel() { tracing::error!(%error, "revoking native streaming admission"); }
    }
}

struct NativeStreamDriver<'r, 's, 'w, 'a> {
    target: &'r mut NativeTarget<'s, 'w, 'a>,
    suffix: Option<EncoderSuffix<'a>>,
    replay_batch: Option<RequestBatch>,
    begun: bool,
    draft_id: u64,
}
impl StreamDriver for NativeStreamDriver<'_, '_, '_, '_> {
    fn begin(&mut self, plan: &StreamPlan) -> Result<()> {
        let lib = self.target.bank.library();
        self.suffix = Some(EncoderSuffix::new(lib, plan.end as u64,
            EncoderSuffix::device_bytes(plan.end as u64)?)?);
        self.target.bank.requests.borrow_mut().begin_encoder(plan.request, plan.end as u64)?;
        self.begun = true;
        Ok(())
    }
    async fn encode(&mut self, plan: &StreamPlan, tokens: &[u32], keep_running: &dyn Fn() -> bool) -> Result<()> {
        let [first, second] = &mut self.target.contexts;
        let chunks = tokens.chunks(plan.chunk_rows).collect::<Vec<_>>();
        // Both contexts are exclusively borrowed for this whole operation; no
        // independent request can enter either one until the result is consumed.
        unsafe { first.driver.pass.execute_encoder_stream(second.driver.pass,
            &mut self.target.bank.requests.borrow_mut(), plan.request, &chunks,
            [first.driver.transport, second.driver.transport], self.suffix.as_mut().unwrap(),
            keep_running).await }
    }
    async fn replay(&mut self, plan: &StreamPlan) -> Result<()> {
        let mut requests = self.target.bank.requests.borrow_mut();
        let start = requests.begin_decoder_replay(plan.request)?;
        ensure!(start == plan.start as u64, "native decoder replay start differs");
        self.replay_batch = Some(requests.prepare_replay(&[CacheWork {
            lease: plan.request, tokens: (plan.end - plan.start) as u32, kind: SourceKind::Prefill,
        }])?);
        let first = &mut self.target.contexts[0].driver;
        let suffix = self.suffix.as_ref().unwrap().output()?;
        unsafe { first.pass.execute_replay(&requests, self.replay_batch.as_mut().unwrap(),
            first.transport, 0, &plan.selected, &suffix).await?; }
        Ok(())
    }
    fn logits(&self) -> Result<Logits<'_>> {
        let output = self.target.contexts[0].driver.pass.output(self.replay_batch.as_ref().unwrap())?;
        Ok(Logits { rows: output.rows, selected: output.selected_rows,
            positions: output.token_positions, device: Some(output.logits), host: None })
    }
    fn commit(&mut self, plan: &StreamPlan) -> Result<u64> {
        let mut requests = self.target.bank.requests.borrow_mut();
        let pass = &mut self.target.contexts[0].driver.pass;
        let batch = self.replay_batch.as_mut().unwrap();
        if let Some(draft) = self.target.bank.draft.borrow_mut().as_deref_mut() {
            // Only the retained final-window decoder taps seed dSpark. Encoder
            // source publication is not an accepted draft/model prefix.
            draft.commit(pass, &mut requests, batch, (plan.end - plan.start) as u32, false)?;
        } else { pass.commit(&mut requests, batch, &[(plan.end - plan.start) as u32])?; }
        ensure!(requests.cache().stage(plan.request)? == CacheStage::Full,
            "native decoder replay did not return to full phase");
        requests.cache().committed_end(plan.request)
    }
    fn revoke(&mut self, plan: &StreamPlan) -> Result<()> {
        if !self.begun { return Ok(()); }
        let [first, second] = &mut self.target.contexts;
        // Evaluate every drain, including when an earlier one fails. A failed
        // drain poisons this owner; unrelated work must never resume on it.
        let a = first.driver.transport.synchronize();
        let b = second.driver.transport.synchronize();
        let mut requests = self.target.bank.requests.borrow_mut();
        let c = first.driver.pass.abort_cache_commit(&mut requests);
        let d = second.driver.pass.abort_cache_commit(&mut requests);
        let e = if let Some(batch) = &mut self.replay_batch { first.driver.pass.discard(batch) } else { Ok(()) };
        let drained = a.and(b).and(c).and(d).and(e);
        if drained.is_err() {
            self.target.bank.active.poisoned.set(true);
            return drained;
        }
        let draft = self.target.bank.draft.borrow_mut().as_deref_mut()
            .map(|draft| draft.release(self.draft_id)).transpose();
        let released = requests.release_if_present(plan.request).and(draft.map(|_| ()));
        if released.is_err() { self.target.bank.active.poisoned.set(true); }
        released
    }
}

/// Both contexts and the result's CUDA storage remain borrowed until sampling
/// drains and this result is committed. Drop/cancel revokes the fresh admission.
/// ```compile_fail
/// use ds41rt_daemon::native_executor::target::StreamingResult;
/// fn premature_reuse(result: StreamingResult<'_, '_, '_, '_>) {
///     let logits = result.logits().unwrap();
///     result.cancel().unwrap();
///     drop(logits);
/// }
/// ```
pub struct StreamingResult<'r, 's, 'w, 'a> {
    session: StreamSession<NativeStreamDriver<'r, 's, 'w, 'a>>,
}
impl StreamingResult<'_, '_, '_, '_> {
    pub fn logits(&self) -> Result<Logits<'_>> { self.session.logits() }
    pub fn commit(mut self) -> Result<u64> { self.session.commit() }
    pub fn cancel(mut self) -> Result<()> { self.session.cancel() }
    /// Explicit correctness diagnostic; serving should borrow CUDA logits.
    pub async fn download_logits(&mut self) -> Result<Vec<u8>> {
        ensure!(self.session.ready, "streaming logits not ready");
        let rows = (0..self.session.plan.selected.len()).collect::<Vec<_>>();
        let driver = &mut self.session.driver;
        driver.target.contexts[0].driver.pass.download_logits(
            driver.replay_batch.as_ref().unwrap(), &rows).await
    }
}
impl<'s, 'w, 'a> NativeTarget<'s, 'w, 'a> {
    /// Runtime borrowed from the constructor, usable while this target itself
    /// is mutably borrowed by a scoped streaming future.
    pub fn runtime(&self) -> &'s tokio::runtime::Runtime { self.runtime }

    pub(crate) fn validate_stream(&self, input: &StreamInput) -> Result<()> {
        self.bank.active.healthy()?;
        ensure!(self.contexts.iter().all(|lane| lane.job.is_none()), "native contexts have outstanding work");
        self.bank.active.idle(input.request)?;
        input.validate(self.stream_chunk_rows, self.contexts[0].driver.max_context)?;
        let requests = self.bank.requests.borrow();
        ensure!(requests.cache().stage(input.request)? == CacheStage::Full
            && requests.cache().committed_end(input.request)? == 0,
            "streaming prefill requires a fresh admission");
        Ok(())
    }
    /// Command identity only. Whole-target borrowing owns both contexts during
    /// streaming; this does not invent another physical cache transaction.
    pub(crate) fn stream_identity(&self, input: &StreamInput) -> Result<Ticket> {
        self.validate_stream(input)?;
        self.bank.active.identity(input.request, 0)
    }
    pub(crate) fn revoke_stream_admission(&self, request: RequestHandle) -> Result<()> {
        self.bank.active.healthy()?;
        let id = self.bank.requests.borrow().cache().request_id(request).ok();
        let draft = id.map(|id| self.bank.draft.borrow_mut().as_deref_mut()
            .map(|draft| draft.release(id)).transpose()).transpose();
        let released = self.bank.requests.borrow_mut().release_if_present(request).and(draft.map(|_| ()));
        if released.is_err() { self.bank.active.poisoned.set(true); }
        released
    }

    /// Fresh-prompt CED: retained two-context encoder overlap, then one decoder
    /// replay of the final 128 rows. No sampler, HTTP or copied model/cache.
    /// The callback is checked at native chunk admission and publication fences.
    pub async fn stream_prefill<'r>(&'r mut self, input: StreamInput,
        keep_running: &dyn Fn() -> bool) -> Result<StreamingResult<'r, 's, 'w, 'a>> {
        self.validate_stream(&input)?;
        let plan = StreamPlan::new(&input, self.stream_chunk_rows, self.contexts[0].driver.max_context)?;
        let draft_id = self.bank.requests.borrow().cache().request_id(input.request)?;
        let mut session = StreamSession { driver: NativeStreamDriver { target: self,
            suffix: None, replay_batch: None, begun: false, draft_id }, plan, tokens: input.tokens,
            ready: false, armed: true };
        session.execute(keep_running).await?;
        Ok(StreamingResult { session })
    }
}

#[cfg(test)]
#[path = "stream_tests.rs"]
mod tests;
