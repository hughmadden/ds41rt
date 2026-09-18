use super::*;
use crate::v41_compressor::source_cache::{SourcePages, SourcePrefix};
use std::{future::{poll_fn, Future}, pin::pin, task::{Context, Poll}};

// Only computation and completion are fake. Reservations, physical destinations,
// shared prefix references, COW tails and page generations use retained native code.
struct Bank {
    pages: [SourcePages; 4],
    values: [Vec<u64>; 4],
    generations: [u64; 2],
    ends: [Option<u64>; 2],
}
impl Bank {
    fn new() -> Self {
        Self { pages: std::array::from_fn(|_| SourcePages::new(4, 2).unwrap()),
            values: std::array::from_fn(|_| vec![0; 4 * 256]), generations: [0; 2], ends: [None; 2] }
    }
    fn admit(&mut self, slot: usize) -> RequestHandle {
        assert!(self.ends[slot].is_none());
        self.generations[slot] += 1;
        self.ends[slot] = Some(0);
        RequestHandle::new(77, slot, self.generations[slot])
    }
    fn end(&self, request: RequestHandle) -> Result<u64> {
        ensure!(request.matches(77, &self.generations), "foreign/stale request");
        self.ends[request.slot()].context("released request")
    }
    fn release(&mut self, active: &Active, request: RequestHandle) -> Result<()> {
        active.idle(request)?;
        self.end(request)?;
        for source in &mut self.pages { source.release(request.slot())?; }
        self.ends[request.slot()] = None;
        Ok(())
    }
    fn retain(&self, request: RequestHandle) -> Vec<SourcePrefix> {
        self.pages.iter().map(|s| s.retain_prefix(request.slot(), s.committed_rows(request.slot()).unwrap()).unwrap()).collect()
    }
}
#[derive(Default)]
struct Control {
    ready: [bool; 2], fail: [bool; 2], inflight: [bool; 2],
    prepared: [usize; 2], drained: [usize; 2], executed: [Vec<u32>; 2],
    fail_commit: bool,
}
struct Device {
    bank: Rc<RefCell<Bank>>, control: Rc<RefCell<Control>>, lane: usize,
    output: Vec<f32>, selected: Vec<usize>, positions: Vec<u64>,
}
struct Batch { request: RequestHandle, old: u64, reads: Vec<SourcePrefix> }
struct Pending { control: Rc<RefCell<Control>>, lane: usize }
impl Drop for Pending { fn drop(&mut self) { self.control.borrow_mut().inflight[self.lane] = false; } }
unsafe impl TargetDriver for Device {
    type Batch = Batch;
    fn validate(&self, input: &TargetInput) -> Result<()> { self.bank.borrow().end(input.request)?; Ok(()) }
    fn prepare(&mut self, input: &TargetInput) -> Result<Batch> {
        self.control.borrow_mut().prepared[self.lane] += 1;
        let bank = self.bank.borrow();
        Ok(Batch { request: input.request, old: bank.end(input.request)?, reads: bank.retain(input.request) })
    }
    async fn execute(&mut self, batch: &mut Batch, input: &TargetInput) -> Result<()> {
        self.control.borrow_mut().inflight[self.lane] = true;
        let _pending = Pending { control: self.control.clone(), lane: self.lane };
        poll_fn(|_| if self.control.borrow().ready[self.lane] { Poll::Ready(()) } else { Poll::Pending }).await;
        ensure!(!self.control.borrow().fail[self.lane], "controlled rank failure");
        // Read the real native source references only after the simulated wait.
        assert_eq!(batch.reads.len(), 4);
        self.control.borrow_mut().executed[self.lane] = input.tokens.clone();
        self.selected = input.selected.clone();
        self.positions = input.selected.iter().map(|&row| batch.old + row as u64).collect();
        self.output = input.selected.iter().map(|&row| input.tokens[row] as f32).collect();
        Ok(())
    }
    fn logits<'a>(&'a self, _: &'a Batch) -> Result<Logits<'a>> {
        Ok(Logits { rows: self.output.len(), selected: &self.selected,
            positions: &self.positions, device: None, host: Some(&self.output) })
    }
    fn commit(&mut self, batch: &mut Batch, accepted: u32) -> Result<u64> {
        assert!(!self.control.borrow().inflight[self.lane]);
        // Real target consumers have drained before publishing accepted writes.
        batch.reads.clear();
        let mut bank = self.bank.borrow_mut();
        let slot = batch.request.slot();
        let end = bank.end(batch.request)? + accepted as u64;
        let plans = (0..4).map(|s| {
            let ratio = if s == 3 { 1 } else { 2 };
            let old = bank.pages[s].committed_rows(slot)?;
            bank.pages[s].reserve(&[(slot, old, end as usize / ratio)])
        }).collect::<Result<Vec<_>>>()?;
        ensure!(!self.control.borrow().fail_commit, "controlled publication failure");
        for (s, plan) in plans.into_iter().enumerate() {
            for (from, to) in plan.tail_copies() {
                bank.values[s].copy_within(from as usize * 256..(from as usize + 1) * 256, to as usize * 256);
            }
            let ratio = if s == 3 { 1 } else { 2 };
            let old = bank.pages[s].committed_rows(slot)?;
            for row in old..end as usize / ratio {
                let dst = bank.pages[s].destination(&plan, slot, row)? as usize;
                bank.values[s][dst] = (slot as u64 + 1) * 10000 + row as u64;
            }
            bank.pages[s].apply(plan);
        }
        bank.ends[slot] = Some(end);
        Ok(end)
    }
    fn drain(&mut self, batch: &mut Batch) -> Result<()> {
        assert!(!self.control.borrow().inflight[self.lane], "execution future still borrows device");
        self.control.borrow_mut().drained[self.lane] += 1;
        self.output.clear();
        batch.reads.clear();
        Ok(())
    }
}
struct Fixture {
    lanes: [TargetContext<Device>; 2], bank: Rc<RefCell<Bank>>,
    control: Rc<RefCell<Control>>, active: Rc<Active>, requests: [RequestHandle; 2],
}
fn fixture() -> Fixture {
    let bank = Rc::new(RefCell::new(Bank::new()));
    let requests = { let mut bank = bank.borrow_mut(); [bank.admit(0), bank.admit(1)] };
    let control = Rc::new(RefCell::new(Control::default()));
    let active = Rc::new(Active::default());
    let lanes = std::array::from_fn(|lane| TargetContext::new(Device {
        bank: bank.clone(), control: control.clone(), lane, output: vec![], selected: vec![], positions: vec![],
    }, active.clone(), lane, 1024, 48));
    Fixture { lanes, bank, control, active, requests }
}
fn input(request: RequestHandle, rows: usize) -> TargetInput {
    TargetInput { request, tokens: (0..rows as u32).collect(), selected: vec![rows - 1],
        kind: SourceKind::Prefill, placement: 1 }
}
fn poll_once<F: Future>(future: std::pin::Pin<&mut F>) -> Poll<F::Output> {
    future.poll(&mut Context::from_waker(std::task::Waker::noop()))
}
fn complete(lane: &mut TargetContext<Device>, ticket: Ticket) -> Result<()> {
    let mut future = pin!(lane.execute(ticket));
    match poll_once(future.as_mut()) { Poll::Ready(result) => result, Poll::Pending => panic!("device not complete") }
}

#[test]
fn independent_context_completes_and_commits_while_peer_waits() -> Result<()> {
    let mut f = fixture();
    let [first, second] = &mut f.lanes;
    let a = first.submit(input(f.requests[0], 4))?;
    let b = second.submit(input(f.requests[1], 6))?;
    let mut waiting = pin!(first.execute(a));
    assert!(poll_once(waiting.as_mut()).is_pending());
    assert!(f.bank.borrow_mut().release(&f.active, f.requests[0]).is_err());
    f.control.borrow_mut().ready[1] = true;
    complete(second, b)?;
    assert_eq!(second.logits(b)?.host(), Some(&[5.0][..]));
    assert_eq!(second.commit(b, 3)?, 3);
    assert_eq!(f.bank.borrow().end(f.requests[0])?, 0);
    assert_eq!(f.bank.borrow().pages[3].committed_rows(1)?, 3);
    f.control.borrow_mut().ready[0] = true;
    assert!(poll_once(waiting.as_mut()).is_ready());
    Ok(())
}

#[test]
fn borrowed_logits_keep_request_live_but_do_not_block_other_context() -> Result<()> {
    let mut f = fixture(); f.control.borrow_mut().ready = [true; 2];
    let [first, second] = &mut f.lanes;
    let a = first.submit(input(f.requests[0], 3))?; complete(first, a)?;
    let logits = first.logits(a)?;
    let b = second.submit(input(f.requests[1], 4))?; complete(second, b)?;
    assert_eq!(second.commit(b, 4)?, 4);
    assert!(f.bank.borrow_mut().release(&f.active, f.requests[0]).is_err());
    assert_eq!(logits.host(), Some(&[2.0][..]));
    assert_eq!(logits.positions, [2]);
    drop(logits);
    first.cancel(a)?;
    f.bank.borrow_mut().release(&f.active, f.requests[0])?;
    Ok(())
}

#[test]
fn cancellation_after_pending_future_drains_before_release_and_generation_reuse() -> Result<()> {
    let mut f = fixture();
    let ticket = f.lanes[0].submit(input(f.requests[0], 4))?;
    {
        let mut future = pin!(f.lanes[0].execute(ticket));
        assert!(poll_once(future.as_mut()).is_pending());
    }
    assert!(f.lanes[0].logits(ticket).is_err());
    assert!(f.lanes[0].commit(ticket, 1).is_err());
    f.lanes[0].cancel(ticket)?;
    assert_eq!(f.control.borrow().drained, [1, 0]);
    f.bank.borrow_mut().release(&f.active, f.requests[0])?;
    let fresh = f.bank.borrow_mut().admit(0);
    assert_ne!(fresh.generation(), f.requests[0].generation());
    assert!(f.lanes[0].submit(input(f.requests[0], 1)).is_err());
    let next = f.lanes[0].submit(input(fresh, 1))?;
    assert!(f.lanes[0].cancel(ticket).is_err());
    f.lanes[0].cancel(next)?;
    assert!(f.bank.borrow().pages.iter().all(|p| p.free_pages() == 4));
    Ok(())
}

#[test]
fn failed_execution_publishes_neither_logits_nor_cache_and_is_cancellable() -> Result<()> {
    let mut f = fixture(); f.control.borrow_mut().ready[0] = true;
    f.control.borrow_mut().fail[0] = true;
    let ticket = f.lanes[0].submit(input(f.requests[0], 6))?;
    assert!(complete(&mut f.lanes[0], ticket).is_err());
    assert!(f.lanes[0].logits(ticket).is_err());
    assert!(f.lanes[0].commit(ticket, 3).is_err());
    assert_eq!(f.bank.borrow().end(f.requests[0])?, 0);
    f.lanes[0].cancel(ticket)?;
    assert!(f.bank.borrow().pages.iter().all(|p| p.free_pages() == 4));
    Ok(())
}

#[test]
fn busy_and_foreign_tickets_do_not_prepare_or_mutate_other_request() -> Result<()> {
    let mut f = fixture();
    let ticket = f.lanes[0].submit(input(f.requests[0], 1))?;
    assert!(f.lanes[0].submit(input(f.requests[1], 1)).is_err());
    assert!(f.lanes[1].submit(input(f.requests[0], 1)).is_err());
    assert!(f.lanes[1].cancel(ticket).is_err());
    assert_eq!(f.control.borrow().prepared, [1, 0]);
    assert!(f.lanes[1].submit(input(RequestHandle::new(999, 1, 1), 1)).is_err());
    f.lanes[0].cancel(ticket)?;
    Ok(())
}

#[test]
fn inputs_are_owned_until_execution_and_invalid_selection_never_prepares() -> Result<()> {
    let mut f = fixture();
    let mut source = vec![8, 9, 10];
    let mut request = input(f.requests[0], 3); request.tokens = source.clone();
    let ticket = f.lanes[0].submit(request)?;
    source.fill(0); f.control.borrow_mut().ready[0] = true;
    complete(&mut f.lanes[0], ticket)?;
    assert_eq!(f.control.borrow().executed[0], [8, 9, 10]);
    assert!(f.lanes[0].commit(ticket, 4).is_err());
    assert_eq!(f.lanes[0].logits(ticket)?.host(), Some(&[10.0][..]));
    f.lanes[0].cancel(ticket)?;
    for selected in [vec![], vec![3], vec![1, 1], vec![2, 1]] {
        let mut request = input(f.requests[0], 3); request.selected = selected;
        assert!(f.lanes[0].submit(request).is_err());
    }
    assert_eq!(f.control.borrow().prepared, [1, 0]);
    Ok(())
}

#[test]
fn native_page_reservations_rollback_on_failed_publication() -> Result<()> {
    let mut f = fixture(); f.control.borrow_mut().ready[0] = true;
    let ticket = f.lanes[0].submit(input(f.requests[0], 257))?;
    complete(&mut f.lanes[0], ticket)?;
    f.control.borrow_mut().fail_commit = true;
    assert!(f.lanes[0].commit(ticket, 257).is_err());
    assert!(f.lanes[0].logits(ticket).is_err());
    assert!(f.lanes[1].submit(input(f.requests[1], 1)).is_err());
    assert_eq!(f.bank.borrow().end(f.requests[0])?, 0);
    assert!(f.bank.borrow().pages.iter().all(|p| p.free_pages() == 4));
    f.lanes[0].cancel(ticket)?;
    assert!(f.lanes[0].submit(input(f.requests[0], 1)).is_err());
    Ok(())
}

#[test]
fn shared_native_prefix_tail_survives_peer_append_and_delayed_release() -> Result<()> {
    let mut f = fixture(); f.control.borrow_mut().ready = [true; 2];
    let ticket = f.lanes[0].submit(input(f.requests[0], 126))?;
    complete(&mut f.lanes[0], ticket)?; f.lanes[0].commit(ticket, 126)?;
    let prefix = f.bank.borrow().retain(f.requests[0]);
    {
        let mut bank = f.bank.borrow_mut();
        for s in 0..4 { bank.pages[s].restore_prefix(1, &prefix[s])?; }
        bank.ends[1] = Some(126);
    }
    let before = f.bank.borrow().values[3].clone();
    let original = prefix[3].pages()[0] as usize;
    let ticket = f.lanes[1].submit(input(f.requests[1], 2))?;
    complete(&mut f.lanes[1], ticket)?; f.lanes[1].commit(ticket, 2)?;
    assert_eq!(&f.bank.borrow().values[3][original * 256..(original + 1) * 256],
        &before[original * 256..(original + 1) * 256]);
    assert_eq!(f.bank.borrow().end(f.requests[0])?, 126);
    assert_eq!(f.bank.borrow().end(f.requests[1])?, 128);
    f.bank.borrow_mut().release(&f.active, f.requests[0])?;
    f.bank.borrow_mut().release(&f.active, f.requests[1])?;
    assert!(f.bank.borrow().pages.iter().all(|p| p.free_pages() == 3));
    drop(prefix);
    assert!(f.bank.borrow().pages.iter().all(|p| p.free_pages() == 4));
    Ok(())
}

#[test]
fn dropping_context_drains_and_releases_its_batch_without_freeing_shared_bank() -> Result<()> {
    let mut f = fixture();
    f.lanes[0].submit(input(f.requests[0], 4))?;
    f.lanes[1].submit(input(f.requests[1], 4))?;
    drop(f.lanes);
    assert_eq!(f.control.borrow().drained, [1, 1]);
    assert!(f.active.tickets.borrow().is_empty());
    assert_eq!(f.bank.borrow().end(f.requests[0])?, 0);
    f.bank.borrow_mut().release(&f.active, f.requests[0])?;
    f.bank.borrow_mut().release(&f.active, f.requests[1])?;
    Ok(())
}
