//! One retained TargetPass per lane, with canonical per-request row ownership.
use super::*;

#[derive(Clone, Debug, serde::Serialize)]
pub struct BatchMember {
    pub request: RequestHandle,
    pub tokens: Vec<u32>,
    /// Indices local to this member, before concatenation into the native pass.
    pub selected: Vec<usize>,
    pub input_offset: usize,
    pub output_offset: usize,
}
#[derive(Debug)]
pub struct DraftBatch {
    pub tokens: Vec<Vec<u32>>,
    pub draft_us: u64,
}
#[derive(Debug)]
pub struct BatchProposal {
    pub ticket: Ticket,
    pub members: Vec<BatchMember>,
    pub draft_us: u64,
}

fn validate_inputs(inputs: &[TargetInput], capacity: usize, head_capacity: usize) -> Result<()> {
    ensure!(
        !inputs.is_empty() && inputs.len() <= 8,
        "invalid target member count"
    );
    let mut rows = 0usize;
    let mut selected = 0usize;
    for (i, input) in inputs.iter().enumerate() {
        input.validate(capacity, head_capacity)?;
        ensure!(
            !inputs[..i]
                .iter()
                .any(|other| other.request == input.request),
            "duplicate target request"
        );
        ensure!(
            input.placement == inputs[0].placement,
            "target batch placement differs"
        );
        ensure!(
            input.kind == inputs[0].kind
                && matches!(input.kind, SourceKind::Decode | SourceKind::MtpVerify),
            "grouped target requires homogeneous decode work"
        );
        rows = rows
            .checked_add(input.tokens.len())
            .context("target row count overflow")?;
        selected = selected
            .checked_add(input.selected.len())
            .context("target head count overflow")?;
    }
    ensure!(
        rows <= capacity && selected <= head_capacity,
        "aggregate target rows exceed lane capacity"
    );
    Ok(())
}
fn manifest(inputs: &[TargetInput]) -> Vec<BatchMember> {
    let (mut input_offset, mut output_offset) = (0, 0);
    inputs
        .iter()
        .map(|input| {
            let member = BatchMember {
                request: input.request,
                tokens: input.tokens.clone(),
                selected: input.selected.clone(),
                input_offset,
                output_offset,
            };
            input_offset += input.tokens.len();
            output_offset += input.selected.len();
            member
        })
        .collect()
}
impl<D: TargetDriver> TargetContext<D> {
    pub fn submit_batch(&mut self, inputs: Vec<TargetInput>) -> Result<Ticket> {
        self.active.healthy()?;
        ensure!(self.job.is_none(), "target lane busy");
        validate_inputs(&inputs, self.capacity, self.head_capacity)?;
        for input in &inputs {
            self.driver.validate(input)?;
        }
        let ticket = self.active.claim_many(
            &inputs.iter().map(|i| i.request).collect::<Vec<_>>(),
            self.lane,
        )?;
        let batch = match self.driver.prepare_batch(&inputs) {
            Ok(batch) => batch,
            Err(error) => {
                self.active.finish(ticket);
                return Err(error);
            }
        };
        self.job = Some(Job {
            ticket,
            inputs,
            grouped: true,
            batch,
            phase: Phase::Prepared,
        });
        Ok(ticket)
    }
    pub fn batch_members(&self, ticket: Ticket) -> Result<Vec<BatchMember>> {
        self.check(ticket)?;
        let job = self.job.as_ref().unwrap();
        ensure!(job.grouped, "singleton target has no grouped manifest");
        Ok(manifest(&job.inputs))
    }
    /// The future owns the lane until the actual native target/draft publication
    /// completes. Dropping it leaves a failed job which must be drained/cancelled;
    /// no partial success or accepted count is reported to the scheduler.
    pub async fn commit_batch(&mut self, ticket: Ticket, accepted: &[u32]) -> Result<Vec<u64>> {
        self.check(ticket)?;
        let job = self.job.as_mut().unwrap();
        ensure!(
            job.grouped && job.phase == Phase::Ready,
            "grouped target execution not complete"
        );
        ensure!(
            accepted.len() == job.inputs.len()
                && accepted
                    .iter()
                    .zip(&job.inputs)
                    .all(|(&n, input)| n as usize <= input.tokens.len()),
            "accepted vector differs from target members"
        );
        job.phase = Phase::Failed;
        let result = self.driver.commit_batch(&mut job.batch, accepted).await;
        let ends = match result {
            Ok(ends) if ends.len() == accepted.len() => ends,
            Ok(_) => {
                self.active.poisoned.set(true);
                anyhow::bail!("native publication count differs");
            }
            Err(error) => {
                self.active.poisoned.set(true);
                return Err(error);
            }
        };
        self.job = None;
        self.active.finish(ticket);
        Ok(ends)
    }
}
impl<D: SpeculativeDriver> TargetContext<D> {
    pub async fn submit_speculative_batch(
        &mut self,
        inputs: Vec<SpeculativeInput>,
    ) -> Result<BatchProposal> {
        self.active.healthy()?;
        ensure!(self.job.is_none(), "target lane busy");
        ensure!(
            !inputs.is_empty() && inputs.len() <= 8,
            "invalid draft member count"
        );
        for input in &inputs {
            ensure!(
                input.anchor < 129280 && (1..=1048576).contains(&input.remaining_output_tokens),
                "invalid speculative input"
            );
            ensure!(
                input.placement == inputs[0].placement,
                "draft batch placement differs"
            );
            self.driver.validate_proposal(input)?;
        }
        let ticket = self.active.claim_many(
            &inputs.iter().map(|i| i.request).collect::<Vec<_>>(),
            self.lane,
        )?;
        let mut guard = speculative::ProposalGuard {
            driver: &mut self.driver,
            active: self.active.clone(),
            ticket,
            armed: true,
        };
        let proposed = loop {
            if let Some(proposed) = guard.driver.poll_proposal_batch(&inputs)? {
                break proposed;
            }
            tokio::task::yield_now().await;
        };
        ensure!(
            proposed.tokens.len() == inputs.len(),
            "native draft member count differs"
        );
        let targets = inputs
            .iter()
            .zip(proposed.tokens)
            .map(|(input, tokens)| {
                ensure!(
                    !tokens.is_empty()
                        && tokens.len() <= 6
                        && tokens.len() <= input.remaining_output_tokens
                        && tokens[0] == input.anchor,
                    "invalid native draft extent or anchor"
                );
                Ok(TargetInput {
                    request: input.request,
                    selected: (0..tokens.len()).collect(),
                    tokens,
                    kind: SourceKind::MtpVerify,
                    placement: input.placement,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        validate_inputs(&targets, self.capacity, self.head_capacity)?;
        for target in &targets {
            guard.driver.validate(target)?;
        }
        let batch = guard.driver.prepare_batch(&targets)?;
        guard.armed = false;
        drop(guard);
        let members = manifest(&targets);
        self.job = Some(Job {
            ticket,
            inputs: targets,
            grouped: true,
            batch,
            phase: Phase::Prepared,
        });
        Ok(BatchProposal {
            ticket,
            members,
            draft_us: proposed.draft_us,
        })
    }
}
