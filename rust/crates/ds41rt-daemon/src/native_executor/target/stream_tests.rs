use super::*;
use std::{future::{poll_fn, Future}, pin::pin, task::{Context, Poll, Waker}};

#[derive(Default)]
struct State {
    events: Vec<&'static str>, begun: bool, ready: bool, published: Option<u64>,
    fail: Option<&'static str>, pending_encode: bool, pending_replay: bool,
    chunks: Vec<usize>, replay_rows: usize, selected: Vec<usize>,
}
struct Fake {
    state: Rc<RefCell<State>>, values: Vec<f32>, selected: Vec<usize>, positions: Vec<u64>,
}
impl StreamDriver for Fake {
    fn begin(&mut self, _: &StreamPlan) -> Result<()> {
        let mut s = self.state.borrow_mut();
        s.events.push("begin");
        ensure!(s.fail != Some("begin"), "begin failed");
        s.begun = true;
        Ok(())
    }
    async fn encode(&mut self, p: &StreamPlan, tokens: &[u32], keep: &dyn Fn() -> bool) -> Result<()> {
        {
            let mut s = self.state.borrow_mut();
            s.events.push("encode"); s.chunks = tokens.chunks(p.chunk_rows).map(<[u32]>::len).collect();
        }
        poll_fn(|_| if self.state.borrow().pending_encode { Poll::Pending } else { Poll::Ready(()) }).await;
        ensure!(keep() && self.state.borrow().fail != Some("encode"), "encoder failed");
        Ok(())
    }
    async fn replay(&mut self, p: &StreamPlan) -> Result<()> {
        {
            let mut s = self.state.borrow_mut();
            s.events.push("replay"); s.replay_rows = p.end - p.start; s.selected = p.selected.clone();
        }
        poll_fn(|_| if self.state.borrow().pending_replay { Poll::Pending } else { Poll::Ready(()) }).await;
        ensure!(self.state.borrow().fail != Some("replay"), "replay failed");
        self.selected = p.selected.clone();
        self.positions = p.selected.iter().map(|&r| (p.start+r) as u64).collect();
        self.values = self.positions.iter().map(|&r| r as f32).collect();
        self.state.borrow_mut().ready = true;
        Ok(())
    }
    fn logits(&self) -> Result<Logits<'_>> {
        ensure!(self.state.borrow().ready, "device output incomplete");
        Ok(Logits { rows: self.values.len(), selected: &self.selected, positions: &self.positions,
            device: None, host: Some(&self.values) })
    }
    fn commit(&mut self, p: &StreamPlan) -> Result<u64> {
        let mut s = self.state.borrow_mut(); s.events.push("commit");
        ensure!(s.fail != Some("commit"), "commit failed");
        s.published = Some(p.end as u64);
        Ok(p.end as u64)
    }
    fn revoke(&mut self, _: &StreamPlan) -> Result<()> {
        let mut s = self.state.borrow_mut();
        if s.begun { s.events.extend(["drain-both", "revoke"]); s.published = None; s.begun = false; }
        s.ready = false;
        ensure!(s.fail != Some("revoke"), "drain failed");
        Ok(())
    }
}
fn input(rows: usize, chunk: usize) -> StreamInput {
    StreamInput { request: RequestHandle::new(71, 0, 1), tokens: vec![3; rows],
        chunk_rows: chunk, selected: vec![rows.saturating_sub(1)] }
}
fn session(rows: usize) -> (StreamSession<Fake>, Rc<RefCell<State>>) {
    let i = input(rows, 80);
    let state = Rc::new(RefCell::new(State::default()));
    (StreamSession { plan: StreamPlan::new(&i, 256, 4096).unwrap(), tokens: i.tokens,
        ready: false, armed: true, driver: Fake { state: state.clone(),
            values: Vec::new(), selected: Vec::new(), positions: Vec::new() } }, state)
}
fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread().build().unwrap()
}

#[test]
fn final_window_selection_and_chunk_crossings_are_exact() -> Result<()> {
    for rows in [1, 79, 80, 81, 127, 128, 129, 257, 2049] {
        let (mut s, state) = session(rows);
        assert!(s.logits().is_err());
        runtime().block_on(s.execute(&|| true))?;
        let output = s.logits()?;
        assert_eq!(output.positions, [rows as u64-1]);
        assert_eq!(output.selected, [rows.min(128)-1]);
        assert_eq!(output.host().unwrap(), [rows as f32-1.0]);
        drop(output);
        assert_eq!(s.commit()?, rows as u64);
        drop(s);
        let state = state.borrow();
        assert_eq!(state.events, ["begin", "encode", "replay", "commit"]);
        assert_eq!(state.chunks.iter().sum::<usize>(), rows);
        assert!(state.chunks.iter().all(|&n| n <= 80));
        assert_eq!(state.replay_rows, rows.min(128));
        assert_eq!(state.published, Some(rows as u64));
    }
    Ok(())
}

#[test]
fn input_validation_precedes_any_cache_or_device_mutation() {
    for bad in [input(0, 80), input(4097, 80), input(3, 79), input(3, 257)] {
        assert!(StreamPlan::new(&bad, 256, 4096).is_err());
    }
    let mut bad = input(257, 80);
    for selection in [vec![], vec![128], vec![257], vec![256,255], vec![256,256], (129..178).collect()] {
        bad.selected = selection;
        assert!(StreamPlan::new(&bad,256,4096).is_err());
    }
    bad.selected = vec![129,256];
    assert_eq!(StreamPlan::new(&bad,256,4096).unwrap().selected, [0,127]);
    bad.tokens[0] = 129280;
    assert!(StreamPlan::new(&bad,256,4096).is_err());
}

#[test]
fn cancellation_and_failures_never_publish_partial_encoder_cache() {
    for failure in ["begin", "encode", "replay", "commit"] {
        let (mut s, state) = session(257);
        state.borrow_mut().fail = Some(failure);
        let executed = runtime().block_on(s.execute(&|| true));
        if failure == "commit" { executed.unwrap(); assert!(s.commit().is_err()); }
        else { assert!(executed.is_err()); assert!(s.logits().is_err()); }
        drop(s);
        let state = state.borrow();
        assert_eq!(state.published, None);
        if failure != "begin" { assert!(state.events.ends_with(&["drain-both","revoke"])); }
        if failure == "encode" { assert!(!state.events.contains(&"replay")); }
    }
}

#[test]
fn dropping_future_in_either_stage_drains_before_admission_release() {
    for encode in [true,false] {
        let (mut s,state) = session(257);
        state.borrow_mut().pending_encode = encode;
        state.borrow_mut().pending_replay = !encode;
        {
            let mut pending = pin!(s.execute(&|| true));
            assert!(pending.as_mut().poll(&mut Context::from_waker(Waker::noop())).is_pending());
        }
        assert!(s.logits().is_err());
        drop(s);
        assert!(state.borrow().events.ends_with(&["drain-both","revoke"]));
        assert_eq!(state.borrow().published,None);
    }
}

#[test]
fn ready_result_cancel_and_drop_revoke_once_and_pre_dispatch_cancel_is_empty() -> Result<()> {
    for explicit in [true,false] {
        let (mut s,state) = session(257);
        runtime().block_on(s.execute(&|| true))?;
        if explicit { s.cancel()?; assert!(s.logits().is_err()); }
        drop(s);
        assert_eq!(state.borrow().events.iter().filter(|&&x| x=="revoke").count(),1);
        assert_eq!(state.borrow().published,None);
    }
    let (mut s,state) = session(257);
    assert!(runtime().block_on(s.execute(&|| false)).is_err());
    drop(s);
    assert!(state.borrow().events.is_empty());
    Ok(())
}

#[test]
fn cancellation_fence_after_encoder_suppresses_decoder() {
    let (mut s,state) = session(257);
    let checks = Cell::new(0);
    let keep = || { let n=checks.get(); checks.set(n+1); n<2 };
    assert!(runtime().block_on(s.execute(&keep)).is_err());
    drop(s);
    assert_eq!(state.borrow().events,["begin","encode","drain-both","revoke"]);
}
