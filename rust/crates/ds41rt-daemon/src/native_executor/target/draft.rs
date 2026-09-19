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
    fn cancel(&mut self, lane: usize) -> Result<()>;
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
        let Some((mut proposals, draft_us)) = self.poll_propose(
            lane,
            &[(
                id,
                input.anchor,
                input.expected_committed_end,
                input.remaining_output_tokens,
            )],
        )?
        else {
            return Ok(None);
        };
        ensure!(proposals.len() == 1, "native draft result count differs");
        let mut tokens = proposals.remove(0);
        ensure!(!tokens.is_empty(), "native draft omitted anchor");
        let maximum = self.confidence_prefix(id, tokens.len() - 1)?;
        tokens.truncate(maximum + 1);
        if self.adaptive_enabled() && maximum > 0 {
            if let Some(lengths) = self.select_prefixes(&[(id, lane, maximum)], draft_us)? {
                ensure!(
                    lengths.len() == 1 && lengths[0] <= maximum,
                    "native adaptive result differs"
                );
                tokens.truncate(lengths[0] + 1);
            }
        }
        Ok(Some(DraftTokens { tokens, draft_us }))
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
    fn cancel_proposal(&mut self) -> Result<()> {
        self.draft
            .borrow_mut()
            .as_deref_mut()
            .context("native dSpark disabled")?
            .cancel(self.lane)
    }
}
