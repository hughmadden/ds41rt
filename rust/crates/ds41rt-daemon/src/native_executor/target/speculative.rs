//! Native draft proposals become owned target batches before caller verification.
use super::*;

#[derive(Clone, Copy, Debug)]
pub struct SpeculativeInput {
    pub request: RequestHandle,
    pub expected_committed_end: u64,
    /// Already emitted token, not yet evaluated as a target input.
    pub anchor: u32,
    pub remaining_output_tokens: usize,
    pub placement: u64,
}
#[derive(Debug)]
pub struct DraftTokens {
    pub tokens: Vec<u32>,
    pub draft_us: u64,
}
/// The target owns its own copy of these rows; callers cannot alter verification.
#[derive(Debug)]
pub struct SpeculativeProposal {
    pub ticket: Ticket,
    pub tokens: Vec<u32>,
    pub draft_us: u64,
}

/// # Safety
/// Polling owns lane-local native draft scratch and reserved window readers.
/// Cancellation must drain before those readers and scratch can be reused;
/// errors must retain unsafe resources and poison the enclosing owner.
#[allow(async_fn_in_trait)]
pub unsafe trait SpeculativeDriver: TargetDriver {
    fn validate_proposal(&self, input: &SpeculativeInput) -> Result<()>;
    fn poll_proposal(&mut self, input: &SpeculativeInput) -> Result<Option<DraftTokens>>;
    fn cancel_proposal(&mut self) -> Result<()>;
    fn poll_proposal_batch(&mut self, inputs: &[SpeculativeInput]) -> Result<Option<DraftBatch>> {
        ensure!(inputs.len() == 1, "driver does not support grouped drafts");
        Ok(self.poll_proposal(&inputs[0])?.map(|p| DraftBatch { tokens: vec![p.tokens], draft_us: p.draft_us }))
    }
}
pub(super) struct ProposalGuard<'a, D: SpeculativeDriver> {
    pub(super) driver: &'a mut D,
    pub(super) active: Rc<Active>,
    pub(super) ticket: Ticket,
    pub(super) armed: bool,
}
impl<D: SpeculativeDriver> Drop for ProposalGuard<'_, D> {
    fn drop(&mut self) {
        if self.armed {
            if let Err(error) = self.driver.cancel_proposal() {
                self.active.poisoned.set(true);
                tracing::error!(%error,"draining cancelled native draft");
            }
            self.active.finish(self.ticket);
        }
    }
}
impl<D: SpeculativeDriver> TargetContext<D> {
    /// Claim the request/lane before polling the retained draft. Dropping this
    /// future drains proposal readers; success leaves an ordinary prepared target
    /// batch whose complete logits are verified by the caller's existing sampler.
    pub async fn submit_speculative(
        &mut self,
        input: SpeculativeInput,
    ) -> Result<SpeculativeProposal> {
        self.active.healthy()?;
        ensure!(self.job.is_none(), "target lane busy");
        ensure!(
            input.anchor < 129280
                && input.remaining_output_tokens > 0
                && input.remaining_output_tokens <= 1048576,
            "invalid speculative input"
        );
        self.driver.validate_proposal(&input)?;
        let ticket = self.active.claim(input.request, self.lane)?;
        let mut guard = ProposalGuard {
            driver: &mut self.driver,
            active: self.active.clone(),
            ticket,
            armed: true,
        };
        let proposed = loop {
            if let Some(proposed) = guard.driver.poll_proposal(&input)? {
                break proposed;
            }
            tokio::task::yield_now().await;
        };
        ensure!(
            !proposed.tokens.is_empty()
                && proposed.tokens.len() <= 6
                && proposed.tokens.len() <= input.remaining_output_tokens
                && proposed.tokens[0] == input.anchor,
            "invalid native draft proposal extent or anchor"
        );
        let selected = (0..proposed.tokens.len()).collect();
        let target = TargetInput {
            request: input.request,
            tokens: proposed.tokens.clone(),
            selected,
            kind: SourceKind::MtpVerify,
            placement: input.placement,
        };
        target.validate(self.capacity, self.head_capacity)?;
        guard.driver.validate(&target)?;
        let batch = guard.driver.prepare(&target)?;
        guard.armed = false;
        drop(guard);
        self.job = Some(Job {
            ticket,
            inputs: vec![target],
            grouped: false,
            batch,
            phase: Phase::Prepared,
        });
        Ok(SpeculativeProposal {
            ticket,
            tokens: proposed.tokens,
            draft_us: proposed.draft_us,
        })
    }
}
