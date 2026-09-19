//! Borrow the retained runtime without exposing its immutable-weight lifetime.
//! This object only erases a lifetime; no algorithm, cache or GPU owner is copied.
use super::*;
use crate::v41_native_serve::speculative::DraftRuntime;

pub(super) type SharedDraft<'s, 'a> = Rc<RefCell<Option<&'s mut dyn NativeDraft<'a>>>>;
pub(super) trait NativeDraft<'a> {
    fn admit(&mut self, id: u64) -> Result<()>;
    fn release(&mut self, id: u64) -> Result<()>;
    fn validate_position(&self, id: u64, end: u64) -> Result<()>;
    fn committed_end(&self, id: u64) -> Result<Option<u64>>;
    fn capture_routes(&self) -> bool;
    fn poll(
        &mut self,
        lane: usize,
        id: u64,
        input: &SpeculativeInput,
    ) -> Result<Option<DraftTokens>>;
    fn poll_batch(&mut self, lane: usize, inputs: &[(u64, SpeculativeInput)]) -> Result<Option<DraftBatch>>;
    fn cancel(&mut self, lane: usize) -> Result<()>;
    fn begin_commit(&mut self, lane: usize, pass: &TargetPass<'_, 'a>, requests: &Requests<'a>,
        batch: &RequestBatch, accepted: &[u32]) -> Result<()>;
    fn poll_commit(&self, lane: usize) -> Result<bool>;
    fn finish_commit(&mut self, lane: usize, pass: &mut TargetPass<'_, 'a>, requests: &mut Requests<'a>,
        batch: &mut RequestBatch, accepted: &[u32]) -> Result<()>;
    fn abort_commit(&mut self, lane: usize, requests: &mut Requests<'a>, batch: &mut RequestBatch) -> Result<()>;
    fn commit(
        &mut self,
        pass: &mut TargetPass<'_, 'a>,
        requests: &mut Requests<'a>,
        batch: &mut RequestBatch,
        accepted: u32,
        observe_routes: bool,
    ) -> Result<()>;
}
impl<'a> NativeDraft<'a> for DraftRuntime<'_, 'a> {
    fn admit(&mut self, id: u64) -> Result<()> {
        DraftRuntime::admit(self, id)
    }
    fn release(&mut self, id: u64) -> Result<()> {
        DraftRuntime::release(self, id)
    }
    fn validate_position(&self, id: u64, end: u64) -> Result<()> {
        DraftRuntime::validate_position(self, id, end)
    }
    fn committed_end(&self, id: u64) -> Result<Option<u64>> {
        DraftRuntime::committed_end(self, id)
    }
    fn capture_routes(&self) -> bool {
        DraftRuntime::capture_routes(self)
    }
    fn poll(
        &mut self,
        lane: usize,
        id: u64,
        input: &SpeculativeInput,
    ) -> Result<Option<DraftTokens>> {
        Ok(NativeDraft::poll_batch(self, lane, &[(id, *input)])?.map(|mut batch|
            DraftTokens { tokens: batch.tokens.remove(0), draft_us: batch.draft_us }))
    }
    fn poll_batch(&mut self, lane: usize, inputs: &[(u64, SpeculativeInput)]) -> Result<Option<DraftBatch>> {
        let seeds = inputs.iter().map(|(id, input)| (*id, input.anchor, input.expected_committed_end,
            input.remaining_output_tokens)).collect::<Vec<_>>();
        let Some((mut tokens, draft_us)) = self.poll_propose(lane, &seeds)? else { return Ok(None); };
        ensure!(tokens.len() == inputs.len(), "native draft result count differs");
        let candidates = inputs.iter().zip(&mut tokens).map(|((id, _), tokens)| {
            ensure!(!tokens.is_empty(), "native draft omitted anchor");
            let maximum = self.confidence_prefix(*id, tokens.len() - 1)?;
            tokens.truncate(maximum + 1); Ok((*id, lane, maximum))
        }).collect::<Result<Vec<_>>>()?;
        if self.adaptive_enabled() && candidates.iter().any(|(_, _, n)| *n > 0) {
            if let Some(lengths) = self.select_prefixes(&candidates, draft_us)? {
                ensure!(lengths.len() == tokens.len(), "native adaptive member count differs");
                for (tokens, length) in tokens.iter_mut().zip(lengths) {
                    ensure!(length < tokens.len(), "native adaptive result exceeds proposal");
                    tokens.truncate(length + 1);
                }
            }
        }
        Ok(Some(DraftBatch { tokens, draft_us }))
    }
    fn begin_commit(&mut self, lane: usize, pass: &TargetPass<'_, 'a>, requests: &Requests<'a>,
        batch: &RequestBatch, accepted: &[u32]) -> Result<()> {
        self.begin_queued_commit(lane, pass, requests, batch, accepted)
    }
    fn poll_commit(&self, lane: usize) -> Result<bool> { self.poll_queued_commit(lane) }
    fn finish_commit(&mut self, lane: usize, pass: &mut TargetPass<'_, 'a>, requests: &mut Requests<'a>,
        batch: &mut RequestBatch, accepted: &[u32]) -> Result<()> {
        let ids = batch.cache()?.request_ids().to_vec();
        let rows = batch.cache()?.window_chunks(0)?;
        // Observe while the retained queued transaction still owns its member
        // IDs. Any policy failure can then drain/revoke the whole target/draft
        // group; a poisoned owner cannot reuse this uncommitted policy history.
        if self.capture_routes() {
            let mut offset = 0;
            for ((id, rows), &accepted) in ids.into_iter().zip(rows).zip(accepted) {
                self.observe_accepted_routes(id, offset, accepted as usize, pass.captured_routes())?;
                offset += rows.tokens as usize;
            }
        }
        self.finish_queued_commit(lane, pass, requests, batch, accepted)
    }
    fn abort_commit(&mut self, lane: usize, requests: &mut Requests<'a>, batch: &mut RequestBatch) -> Result<()> {
        self.abort_queued_commit(lane, requests, batch)
    }
    fn cancel(&mut self, lane: usize) -> Result<()> {
        self.cancel_propose(lane)
    }
    fn commit(
        &mut self,
        pass: &mut TargetPass<'_, 'a>,
        requests: &mut Requests<'a>,
        batch: &mut RequestBatch,
        accepted: u32,
        observe_routes: bool,
    ) -> Result<()> {
        let ids = batch.cache()?.request_ids();
        ensure!(
            ids.len() == 1,
            "native speculative commit requires one request"
        );
        let id = ids[0];
        // Synchronous retained joint publication. No target or draft suffix is
        // made visible before the external sampler supplies this accepted count.
        DraftRuntime::commit(self, pass, requests, batch, accepted)?;
        if observe_routes && self.capture_routes() {
            self.observe_accepted_routes(id, 0, accepted as usize, pass.captured_routes())?;
        }
        Ok(())
    }
}

// Safety: every pending native chain owns the canonical draft-window read
// reservations. Cancel drains its producer stream before dropping those guards.
unsafe impl SpeculativeDriver for NativeDriver<'_, '_, '_> {
    fn validate_proposal(&self, input: &SpeculativeInput) -> Result<()> {
        let requests = self.requests.borrow();
        let end = requests.cache().committed_end(input.request)?;
        ensure!(
            end > 0 && end == input.expected_committed_end,
            "speculative target frontier differs"
        );
        ensure!(end < self.max_context, "target context bound exceeded");
        ensure!(
            requests.cache().stage(input.request)? == CacheStage::Full,
            "draft requires full-target stage"
        );
        let id = requests.cache().request_id(input.request)?;
        self.draft
            .borrow()
            .as_deref()
            .context("native dSpark disabled")?
            .validate_position(id, end)
    }
    fn poll_proposal(&mut self, input: &SpeculativeInput) -> Result<Option<DraftTokens>> {
        let id = self.requests.borrow().cache().request_id(input.request)?;
        let mut bounded = *input;
        bounded.remaining_output_tokens = bounded
            .remaining_output_tokens
            .min((self.max_context - input.expected_committed_end) as usize);
        self.draft
            .borrow_mut()
            .as_deref_mut()
            .context("native dSpark disabled")?
            .poll(self.lane, id, &bounded)
    }
    fn poll_proposal_batch(&mut self, inputs: &[SpeculativeInput]) -> Result<Option<DraftBatch>> {
        let bounded = inputs.iter().map(|input| {
            let id = self.requests.borrow().cache().request_id(input.request)?;
            let mut input = *input;
            input.remaining_output_tokens = input.remaining_output_tokens.min((self.max_context - input.expected_committed_end) as usize);
            Ok((id, input))
        }).collect::<Result<Vec<_>>>()?;
        self.draft.borrow_mut().as_deref_mut().context("native dSpark disabled")?.poll_batch(self.lane, &bounded)
    }
    fn cancel_proposal(&mut self) -> Result<()> {
        self.draft
            .borrow_mut()
            .as_deref_mut()
            .context("native dSpark disabled")?
            .cancel(self.lane)
    }
}
