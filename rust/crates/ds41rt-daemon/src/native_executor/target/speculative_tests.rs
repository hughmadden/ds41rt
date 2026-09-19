use super::*;
use crate::v41_dspark_cache::{
    access::{ReadReservation, SlotAccess},
    append_end,
};

#[derive(Default)]
struct DraftControl {
    ready: [bool; 2],
    fail: [bool; 2],
    bad_anchor: bool,
    bad_token: bool,
    fail_drain: bool,
    polls: [usize; 2],
    drains: [usize; 2],
    ends: [[Option<u64>; 2]; 3],
}
struct DraftDevice {
    target: Device,
    access: Rc<[SlotAccess; 3]>,
    state: Rc<RefCell<DraftControl>>,
    readers: Option<Vec<ReadReservation>>,
}
unsafe impl TargetDriver for DraftDevice {
    type Batch = Batch;
    fn validate(&self, input: &TargetInput) -> Result<()> {
        self.target.validate(input)
    }
    fn prepare(&mut self, input: &TargetInput) -> Result<Batch> {
        self.target.prepare(input)
    }
    async fn execute(&mut self, batch: &mut Batch, input: &TargetInput) -> Result<()> {
        self.target.execute(batch, input).await
    }
    fn logits<'a>(&'a self, batch: &'a Batch) -> Result<Logits<'a>> {
        self.target.logits(batch)
    }
    fn commit(&mut self, batch: &mut Batch, accepted: u32) -> Result<u64> {
        let slot = batch.request.slot();
        let mask = std::array::from_fn(|i| i == slot);
        // The actual native read/write reservation algorithm excludes pending
        // proposals while allowing publication into the peer's disjoint slots.
        let _writers = self
            .access
            .iter()
            .map(|a| a.reserve_write(mask))
            .collect::<Result<Vec<_>>>()?;
        let ends = self.state.borrow().ends;
        let next = ends
            .iter()
            .map(|ends| append_end(ends[slot], batch.old, accepted))
            .collect::<Result<Vec<_>>>()?;
        // SourcePages reservations/publication are the retained physical bank;
        // only projection and copying into the three draft rings are simulated.
        let end = self.target.commit(batch, accepted)?;
        for (stage, next) in next.into_iter().enumerate() {
            self.state.borrow_mut().ends[stage][slot] = Some(next);
        }
        Ok(end)
    }
    fn drain(&mut self, batch: &mut Batch) -> Result<()> {
        self.target.drain(batch)
    }
}
unsafe impl SpeculativeDriver for DraftDevice {
    fn validate_proposal(&self, input: &SpeculativeInput) -> Result<()> {
        let end = self.target.bank.borrow().end(input.request)?;
        ensure!(
            end > 0 && end == input.expected_committed_end,
            "stale speculative frontier"
        );
        ensure!(
            self.state
                .borrow()
                .ends
                .iter()
                .all(|s| s[input.request.slot()] == Some(end)),
            "draft frontier differs"
        );
        Ok(())
    }
    fn poll_proposal(&mut self, input: &SpeculativeInput) -> Result<Option<DraftTokens>> {
        let lane = self.target.lane;
        self.state.borrow_mut().polls[lane] += 1;
        if self.readers.is_none() {
            let mask = std::array::from_fn(|i| i == input.request.slot());
            self.readers = Some(
                self.access
                    .iter()
                    .map(|a| a.reserve(mask))
                    .collect::<Result<Vec<_>>>()?,
            );
        }
        let state = self.state.borrow();
        if !state.ready[lane] {
            return Ok(None);
        }
        ensure!(!state.fail[lane], "controlled draft failure");
        let mut tokens = (0..input.remaining_output_tokens.min(6))
            .map(|i| input.anchor + i as u32)
            .collect::<Vec<_>>();
        if state.bad_anchor {
            tokens[0] += 1;
        }
        if state.bad_token {
            *tokens.last_mut().unwrap() = 129280;
        }
        self.readers = None; // simulated completion; actual code queries CUDA
        Ok(Some(DraftTokens {
            tokens,
            draft_us: 7,
        }))
    }
    fn cancel_proposal(&mut self) -> Result<()> {
        self.state.borrow_mut().drains[self.target.lane] += 1;
        ensure!(
            !self.state.borrow().fail_drain,
            "controlled CUDA drain failure"
        );
        self.readers = None;
        Ok(())
    }
}
struct DraftFixture {
    lanes: [TargetContext<DraftDevice>; 2],
    bank: Rc<RefCell<Bank>>,
    active: Rc<Active>,
    control: Rc<RefCell<Control>>,
    state: Rc<RefCell<DraftControl>>,
    access: Rc<[SlotAccess; 3]>,
    requests: [RequestHandle; 2],
}
fn draft_fixture() -> Result<DraftFixture> {
    let f = fixture();
    let access = Rc::new(std::array::from_fn(|_| SlotAccess::default()));
    let state = Rc::new(RefCell::new(DraftControl::default()));
    let lanes = std::array::from_fn(|lane| {
        TargetContext::new(
            DraftDevice {
                target: Device {
                    bank: f.bank.clone(),
                    control: f.control.clone(),
                    lane,
                    output: vec![],
                    selected: vec![],
                    positions: vec![],
                },
                access: access.clone(),
                state: state.clone(),
                readers: None,
            },
            f.active.clone(),
            lane,
            1024,
            48,
        )
    });
    let mut f = DraftFixture {
        lanes,
        bank: f.bank,
        active: f.active,
        control: f.control,
        requests: f.requests,
        access,
        state,
    };
    f.control.borrow_mut().ready = [true; 2];
    // Seed through ordinary full-target commit, not a fabricated draft-only seed.
    for lane in 0..2 {
        let ticket = f.lanes[lane].submit(input(f.requests[lane], 4))?;
        ready(f.lanes[lane].execute(ticket))?;
        f.lanes[lane].commit(ticket, 4)?;
    }
    Ok(f)
}
fn ready<T>(future: impl Future<Output = Result<T>>) -> Result<T> {
    match poll_once(pin!(future).as_mut()) {
        Poll::Ready(result) => result,
        Poll::Pending => panic!("expected ready"),
    }
}
fn proposal(request: RequestHandle) -> SpeculativeInput {
    SpeculativeInput {
        request,
        expected_committed_end: 4,
        anchor: 10,
        remaining_output_tokens: 6,
        placement: 1,
    }
}

#[test]
fn peer_verifies_and_commits_while_first_native_draft_readers_are_pending() -> Result<()> {
    let mut f = draft_fixture()?;
    let [first, second] = &mut f.lanes;
    let mut waiting = pin!(first.submit_speculative(proposal(f.requests[0])));
    assert!(poll_once(waiting.as_mut()).is_pending());
    for slots in f.access.iter() {
        assert!(slots.writable(0).is_err());
        slots.writable(1)?;
    }
    assert!(f
        .bank
        .borrow_mut()
        .release(&f.active, f.requests[0])
        .is_err());
    f.state.borrow_mut().ready[1] = true;
    let proposed = ready(second.submit_speculative(proposal(f.requests[1])))?;
    ready(second.execute(proposed.ticket))?;
    assert_eq!(second.logits(proposed.ticket)?.selected, [0, 1, 2, 3, 4, 5]);
    assert_eq!(second.commit(proposed.ticket, 3)?, 7);
    assert!(f
        .state
        .borrow()
        .ends
        .iter()
        .all(|ends| *ends == [Some(4), Some(7)]));
    assert_eq!(f.bank.borrow().pages[3].committed_rows(1)?, 7);
    Ok(())
}

#[test]
fn dropped_proposal_drains_readers_before_request_release_and_reuse() -> Result<()> {
    let mut f = draft_fixture()?;
    {
        let mut pending = pin!(f.lanes[0].submit_speculative(proposal(f.requests[0])));
        assert!(poll_once(pending.as_mut()).is_pending());
        assert!(f.access[0].writable(0).is_err());
    }
    assert_eq!(f.state.borrow().drains, [1, 0]);
    for access in f.access.iter() {
        access.writable(0)?;
    }
    f.bank.borrow_mut().release(&f.active, f.requests[0])?;
    let next = f.bank.borrow_mut().admit(0);
    assert_ne!(next.generation(), f.requests[0].generation());
    assert!(ready(f.lanes[0].submit_speculative(proposal(f.requests[0]))).is_err());
    assert_eq!(f.state.borrow().polls, [1, 0]);
    Ok(())
}

#[test]
fn external_prefix_decision_publishes_only_accepted_inputs_at_every_mismatch() -> Result<()> {
    for mismatch in 0..6 {
        let mut f = draft_fixture()?;
        f.state.borrow_mut().ready[0] = true;
        let mut proposed = ready(f.lanes[0].submit_speculative(proposal(f.requests[0])))?;
        let original = proposed.tokens.clone();
        proposed.tokens.fill(99); // caller copy cannot alter owned target rows
        ready(f.lanes[0].execute(proposed.ticket))?;
        assert_eq!(f.control.borrow().executed[0], original);
        assert_eq!(
            f.lanes[0].logits(proposed.ticket)?.positions,
            [4, 5, 6, 7, 8, 9]
        );
        let mut next = [11, 12, 13, 14, 15, 16];
        next[mismatch] = 99;
        let decision = ds41rt_core::verify_dspark_greedy(&original, &next, 1, 100).unwrap();
        assert_eq!(decision.accepted_inputs, mismatch as u32 + 1);
        let end = f.lanes[0].commit(proposed.ticket, decision.accepted_inputs)?;
        assert_eq!(end, 5 + mismatch as u64);
        assert!(f
            .state
            .borrow()
            .ends
            .iter()
            .all(|ends| ends[0] == Some(end)));
        assert_eq!(f.bank.borrow().pages[3].committed_rows(0)?, end as usize);
        assert!(f.lanes[0].commit(proposed.ticket, 1).is_err());
    }
    Ok(())
}

#[test]
fn ready_cancel_keeps_joint_frontier_and_short_budget_uses_anchor_only() -> Result<()> {
    let mut f = draft_fixture()?;
    f.state.borrow_mut().ready[0] = true;
    let proposed = ready(f.lanes[0].submit_speculative(proposal(f.requests[0])))?;
    ready(f.lanes[0].execute(proposed.ticket))?;
    f.lanes[0].cancel(proposed.ticket)?;
    assert_eq!(f.bank.borrow().end(f.requests[0])?, 4);
    assert!(f.state.borrow().ends.iter().all(|ends| ends[0] == Some(4)));
    let mut short = proposal(f.requests[0]);
    short.remaining_output_tokens = 1;
    let proposed = ready(f.lanes[0].submit_speculative(short))?;
    assert_eq!(proposed.tokens, [10]);
    ready(f.lanes[0].execute(proposed.ticket))?;
    assert_eq!(f.lanes[0].commit(proposed.ticket, 1)?, 5);
    Ok(())
}

#[test]
fn stale_busy_and_malformed_proposals_cannot_prepare_or_publish() -> Result<()> {
    let mut f = draft_fixture()?;
    let mut stale = proposal(f.requests[0]);
    stale.expected_committed_end = 3;
    assert!(ready(f.lanes[0].submit_speculative(stale)).is_err());
    let ticket = f.lanes[0].submit(input(f.requests[0], 1))?;
    assert!(ready(f.lanes[0].submit_speculative(proposal(f.requests[1]))).is_err());
    assert!(ready(f.lanes[1].submit_speculative(proposal(f.requests[0]))).is_err());
    assert_eq!(f.state.borrow().polls, [0, 0]);
    f.lanes[0].cancel(ticket)?;
    f.state.borrow_mut().ready[0] = true;
    for failure in 0..3 {
        {
            let mut state = f.state.borrow_mut();
            state.bad_anchor = failure == 0;
            state.bad_token = failure == 1;
            state.fail[0] = failure == 2;
        }
        assert!(ready(f.lanes[0].submit_speculative(proposal(f.requests[0]))).is_err());
        assert!(f.lanes[0].job.is_none());
        f.active.idle(f.requests[0])?;
        for access in f.access.iter() {
            access.writable(0)?;
        }
    }
    assert_eq!(f.bank.borrow().end(f.requests[0])?, 4);
    assert_eq!(f.control.borrow().prepared, [2, 1]);
    Ok(())
}

#[test]
fn failed_proposal_drain_poisons_owner_and_retains_readers() -> Result<()> {
    let mut f = draft_fixture()?;
    {
        let mut pending = pin!(f.lanes[0].submit_speculative(proposal(f.requests[0])));
        assert!(poll_once(pending.as_mut()).is_pending());
        f.state.borrow_mut().fail_drain = true;
    }
    assert!(f.active.healthy().is_err());
    assert!(f.access[0].writable(0).is_err());
    assert!(f
        .bank
        .borrow_mut()
        .release(&f.active, f.requests[0])
        .is_err());
    assert!(f.lanes[1].submit(input(f.requests[1], 1)).is_err());
    Ok(())
}

#[test]
fn retained_window_seed_and_append_rules_cover_final_decoder_replay() -> Result<()> {
    assert_eq!(append_end(None, 0, 10)?, 10);
    assert_eq!(append_end(None, 242, 128)?, 370);
    assert!(append_end(None, 242, 127).is_err());
    assert_eq!(append_end(Some(370), 370, 3)?, 373);
    assert!(append_end(Some(370), 371, 3).is_err());
    assert!(append_end(Some(370), 369, 3).is_err());
    assert!(append_end(Some(u64::MAX), u64::MAX, 1).is_err());
    Ok(())
}

#[test]
fn speculative_wire_kind_matches_retained_native_verification_even_for_k0() -> Result<()> {
    let mut f = draft_fixture()?;
    f.state.borrow_mut().ready[0] = true;
    for rows in [1, 4] {
        let mut input = proposal(f.requests[0]); input.remaining_output_tokens = rows;
        let proposed = ready(f.lanes[0].submit_speculative(input))?;
        assert_eq!(f.lanes[0].job.as_ref().unwrap().inputs[0].kind, SourceKind::MtpVerify);
        f.lanes[0].cancel(proposed.ticket)?;
    }
    Ok(())
}
