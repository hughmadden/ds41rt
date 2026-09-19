//! CPU ABI contract tests use the production channel, registry and lane actors.
use super::*;
use crate::v41_compressor::source_cache::{SourcePages, SourcePrefix};
use crate::v41_dspark_cache::access::{SlotAccess, ReadReservation};
use std::{
    sync::atomic::{AtomicBool, AtomicUsize},
    time::{Duration, Instant},
};

struct Control {
    execute: [AtomicBool; 2],
    consumer: [AtomicBool; 2],
    fail: [AtomicBool; 2],
    records: [AtomicUsize; 2],
    drains: [AtomicUsize; 2],
    dropped: AtomicUsize,
    stream_stage: AtomicUsize,
    proposal_ready: [AtomicBool; 2],
    proposal_readers: [AtomicUsize; 2],
    proposal_drains: [AtomicUsize; 2],
}
impl Control {
    fn new() -> Self {
        Self {
            execute: std::array::from_fn(|_| AtomicBool::new(true)),
            consumer: std::array::from_fn(|_| AtomicBool::new(true)),
            fail: Default::default(),
            records: Default::default(),
            drains: Default::default(),
            dropped: AtomicUsize::new(0),
            stream_stage: AtomicUsize::new(0),
            proposal_ready: std::array::from_fn(|_| AtomicBool::new(true)),
            proposal_readers: Default::default(), proposal_drains: Default::default(),
        }
    }
}
struct Physical {
    owner: u64,
    pages: [SourcePages; 4],
    ends: [Option<u64>; 2],
    generations: [u64; 2],
}
#[derive(Clone)]
struct FakeBank {
    physical: Rc<RefCell<Physical>>,
    active: Rc<Active>,
    draft_access: Rc<[SlotAccess; 3]>,
}
impl actor::Bank for FakeBank {
    fn info(&self) -> Value {
        let p = self.physical.borrow();
        json!({"owner":p.owner,"capacity_rows":16,
        "source_page_capacity":[8,8,8,8],"source_pages_free":p.pages.iter().map(SourcePages::free_pages).collect::<Vec<_>>(),
        "source_payload_bytes":8*4*256*356,"cache_bytes":8*4*256*356+4096})
    }
    fn admit(&self, slot: usize, _id: u64) -> Result<RequestHandle> {
        let mut p = self.physical.borrow_mut();
        ensure!(slot < 2 && p.ends[slot].is_none(), "slot busy");
        p.generations[slot] += 1;
        p.ends[slot] = Some(0);
        Ok(RequestHandle::new(p.owner, slot, p.generations[slot]))
    }
    fn end(&self, r: RequestHandle) -> Result<u64> {
        let p = self.physical.borrow();
        ensure!(r.matches(p.owner, &p.generations), "stale request");
        p.ends[r.slot()].context("released request")
    }
    fn draft_end(&self, r: RequestHandle) -> Result<Option<u64>> {
        // Simulated projection/frontier. Reader ownership is the real native
        // SlotAccess implementation; physical source pages are SourcePages.
        self.end(r).map(Some)
    }
    fn can_prepare(&self, work: &[(RequestHandle, u32)]) -> Result<bool> {
        ensure!(!work.is_empty() && work.len() <= 16, "invalid query count");
        for (i, &(r, tokens)) in work.iter().enumerate() {
            self.active.idle(r)?;
            let end = self.end(r)?;
            ensure!(end + u64::from(tokens) <= 1048576, "context exceeded");
            ensure!(
                !work[..i].iter().any(|&(other, _)| other == r),
                "duplicate query"
            );
        }
        let p = self.physical.borrow();
        for s in 0..4 {
            let work = work
                .iter()
                .map(|&(r, tokens)| {
                    let end = p.ends[r.slot()].unwrap() as usize;
                    (
                        r.slot(),
                        end / if s == 3 { 1 } else { 2 },
                        (end + tokens as usize) / if s == 3 { 1 } else { 2 },
                    )
                })
                .collect::<Vec<_>>();
            match p.pages[s].reserve(&work) {
                Ok(_) => (),
                Err(error)
                    if error.is::<crate::v41_compressor::source_cache::SourcePoolExhausted>() =>
                {
                    return Ok(false)
                }
                Err(error) => return Err(error),
            }
        }
        Ok(true)
    }
    fn release(&self, r: RequestHandle) -> Result<()> {
        self.active.idle(r)?;
        self.end(r)?;
        for access in self.draft_access.iter() { access.writable(r.slot())?; }
        let mut p = self.physical.borrow_mut();
        for source in &mut p.pages {
            source.release(r.slot())?;
        }
        p.ends[r.slot()] = None;
        Ok(())
    }
}
struct FakeBatch {
    request: RequestHandle,
    selected: Vec<usize>,
    positions: Vec<u64>,
    reads: Vec<SourcePrefix>,
}
struct Driver {
    bank: FakeBank,
    control: Arc<Control>,
    lane: usize,
    output: Vec<f32>,
    draft_readers: Option<Vec<ReadReservation>>,
}
impl Drop for Driver {
    fn drop(&mut self) {
        self.control.dropped.fetch_add(1, Ordering::SeqCst);
    }
}
unsafe impl TargetDriver for Driver {
    type Batch = FakeBatch;
    fn validate(&self, input: &TargetInput) -> Result<()> {
        actor::Bank::end(&self.bank, input.request)?;
        Ok(())
    }
    fn prepare(&mut self, input: &TargetInput) -> Result<FakeBatch> {
        let end = actor::Bank::end(&self.bank, input.request)?;
        let p = self.bank.physical.borrow();
        let reads = p
            .pages
            .iter()
            .map(|s| {
                s.retain_prefix(
                    input.request.slot(),
                    s.committed_rows(input.request.slot())?,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(FakeBatch {
            request: input.request,
            selected: input.selected.clone(),
            positions: input.selected.iter().map(|&r| end + r as u64).collect(),
            reads,
        })
    }
    async fn execute(&mut self, _batch: &mut FakeBatch, input: &TargetInput) -> Result<()> {
        while !self.control.execute[self.lane].load(Ordering::SeqCst) {
            tokio::task::yield_now().await;
        }
        ensure!(
            !self.control.fail[self.lane].load(Ordering::SeqCst),
            "controlled native execution failure"
        );
        self.output = input
            .selected
            .iter()
            .flat_map(|&r| std::iter::repeat_n(input.tokens[r] as f32, 129280))
            .collect();
        Ok(())
    }
    fn logits<'a>(&'a self, batch: &'a FakeBatch) -> Result<Logits<'a>> {
        Ok(Logits {
            rows: batch.selected.len(),
            selected: &batch.selected,
            positions: &batch.positions,
            device: None,
            host: Some(&self.output),
        })
    }
    fn commit(&mut self, batch: &mut FakeBatch, accepted: u32) -> Result<u64> {
        batch.reads.clear();
        let end = actor::Bank::end(&self.bank, batch.request)? + accepted as u64;
        let mut p = self.bank.physical.borrow_mut();
        let slot = batch.request.slot();
        let plans = (0..4)
            .map(|s| {
                p.pages[s].reserve(&[(
                    slot,
                    p.pages[s].committed_rows(slot)?,
                    end as usize / if s == 3 { 1 } else { 2 },
                )])
            })
            .collect::<Result<Vec<_>>>()?;
        for (s, plan) in plans.into_iter().enumerate() {
            p.pages[s].apply(plan);
        }
        p.ends[slot] = Some(end);
        Ok(end)
    }
    fn drain(&mut self, batch: &mut FakeBatch) -> Result<()> {
        batch.reads.clear();
        self.output.clear();
        self.control.drains[self.lane].fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}
unsafe impl SpeculativeDriver for Driver {
    fn validate_proposal(&self, input: &SpeculativeInput) -> Result<()> {
        let end = actor::Bank::end(&self.bank, input.request)?;
        ensure!(end > 0 && end == input.expected_committed_end, "invalid speculative frontier");
        Ok(())
    }
    fn poll_proposal(&mut self, input: &SpeculativeInput) -> Result<Option<DraftTokens>> {
        if self.draft_readers.is_none() {
            let mask = std::array::from_fn(|i| i == input.request.slot());
            self.draft_readers = Some(self.bank.draft_access.iter().map(|a|a.reserve(mask)).collect::<Result<Vec<_>>>()?);
            self.control.proposal_readers[self.lane].store(3, Ordering::SeqCst);
        }
        if !self.control.proposal_ready[self.lane].load(Ordering::SeqCst) { return Ok(None); }
        self.draft_readers = None;
        self.control.proposal_readers[self.lane].store(0, Ordering::SeqCst);
        Ok(Some(DraftTokens { tokens: (0..input.remaining_output_tokens.min(4))
            .map(|i|input.anchor + i as u32).collect(), draft_us: 11 }))
    }
    fn cancel_proposal(&mut self) -> Result<()> {
        self.draft_readers = None;
        self.control.proposal_readers[self.lane].store(0, Ordering::SeqCst);
        self.control.proposal_drains[self.lane].fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}
struct FakeFence {
    control: Arc<Control>,
    lane: usize,
}
impl actor::Fence for FakeFence {
    fn record(&mut self, _stream: usize) -> Result<()> {
        self.control.records[self.lane].fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    fn ready(&self) -> Result<bool> {
        Ok(self.control.consumer[self.lane].load(Ordering::SeqCst))
    }
}
struct Fixture {
    handle: u64,
    client: Arc<Client>,
    control: Arc<Control>,
}
fn fixture() -> Fixture {
    static OWNER: AtomicU64 = AtomicU64::new(1000);
    let owner = OWNER.fetch_add(1, Ordering::SeqCst);
    let (client, mut receive) = Client::pair();
    let control = Arc::new(Control::new());
    let (c, signals) = (client.clone(), control.clone());
    let thread = std::thread::spawn(move || {
        let result = (|| -> Result<u64> {
            let active = Rc::new(Active::default());
            let bank = FakeBank {
                physical: Rc::new(RefCell::new(Physical {
                    owner,
                    pages: std::array::from_fn(|_| SourcePages::new(8, 2).unwrap()),
                    ends: [None; 2],
                    generations: [0; 2],
                })),
                active: active.clone(),
                draft_access: Rc::new(std::array::from_fn(|_| SlotAccess::default())),
            };
            let mut contexts = std::array::from_fn::<_, 2, _>(|lane| {
                TargetContext::new(
                    Driver {
                        bank: bank.clone(),
                        control: signals.clone(),
                        lane,
                        output: vec![], draft_readers: None,
                    },
                    active.clone(),
                    lane,
                    16,
                    4,
                )
            });
            let mut fences = std::array::from_fn(|lane| FakeFence {
                control: signals.clone(),
                lane,
            });
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?;
            c.state.lock().unwrap().initialized = true;
            c.reply(
                0,
                Ok(json!({"state":"initialized","bank":actor::Bank::info(&bank)})),
            );
            loop {
                let next = {
                    let [first, second] = &mut contexts;
                    runtime.block_on(actor::run(
                        &bank,
                        [first, second],
                        &mut fences,
                        c.clone(),
                        &mut receive,
                    ))?
                };
                match next {
                    actor::Exit::Shutdown(id) => return Ok(id),
                    actor::Exit::Stream(message) => {
                        let mut target = FakeStream {
                            bank: &bank,
                            control: &signals,
                        };
                        runtime.block_on(stream::run(
                            &mut target,
                            message,
                            &mut fences[0],
                            c.clone(),
                            &mut receive,
                        ))?;
                    }
                }
            }
        })();
        c.finish(result);
    });
    *client.thread.lock().unwrap() = Some(thread);
    let handle = register(client.clone()).unwrap();
    let f = Fixture {
        handle,
        client,
        control,
    };
    assert!(collect(handle, 0)["ok"].as_bool().unwrap());
    f
}
fn submit(handle: u64, value: Value) -> u64 {
    let bytes = serde_json::to_vec(&value).unwrap();
    let mut id = 0;
    assert_eq!(
        unsafe { ds41rt_target_command(handle, bytes.as_ptr(), bytes.len(), &mut id) },
        OK
    );
    id
}
fn collect(handle: u64, id: u64) -> Value {
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut n = 0;
    loop {
        let status = unsafe { ds41rt_target_result(handle, id, ptr::null_mut(), 0, &mut n) };
        if status == NOT_READY {
            assert!(Instant::now() < deadline, "reply {id} timed out");
            std::thread::yield_now();
            continue;
        }
        assert_eq!(status, BUFFER_TOO_SMALL, "reply {id}");
        break;
    }
    assert!(n <= 65536);
    let mut bytes = vec![0; n];
    assert_eq!(
        unsafe { ds41rt_target_result(handle, id, bytes.as_mut_ptr(), bytes.len(), &mut n) },
        OK
    );
    serde_json::from_slice(&bytes).unwrap()
}
fn rpc(handle: u64, value: Value) -> Value {
    let id = submit(handle, value);
    let ack = collect(handle, id);
    assert_eq!(ack["ok"], true, "{ack}");
    ack["result"].clone()
}
fn error(handle: u64, value: Value, code: i32) {
    let id = submit(handle, value);
    let ack = collect(handle, id);
    assert_eq!(ack["ok"], false, "{ack}");
    assert_eq!(ack["code"], code, "{ack}");
}
fn admit(f: &Fixture, slot: usize) -> Value {
    rpc(
        f.handle,
        json!({"op":"admit","slot":slot,"request_id":100+slot}),
    )["request"]
        .clone()
}
fn prepared(f: &Fixture, lane: usize, request: &Value, tokens: &[u32]) -> Value {
    rpc(f.handle,json!({"op":"submit","lane":lane,"request":request,"expected_committed_end":0,
        "work":{"phase":"full_target","tokens":tokens,"selected":[tokens.len()-1],"kind":"prefill","placement":1}}))["ticket"].clone()
}
fn state(f: &Fixture, ticket: &Value, wanted: &str) {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let result = rpc(f.handle, json!({"op":"poll","ticket":ticket}));
        if result["state"] == wanted {
            return;
        }
        assert!(Instant::now() < deadline, "state stalled: {result}");
        std::thread::yield_now();
    }
}
fn execute(f: &Fixture, ticket: &Value) {
    rpc(f.handle, json!({"op":"execute","ticket":ticket}));
    state(f, ticket, "ready");
}
fn close(f: Fixture) {
    let result = rpc(f.handle, json!({"op":"shutdown"}));
    assert_eq!(result["state"], "closed");
    assert_eq!(f.control.dropped.load(Ordering::SeqCst), 2);
    assert_eq!(ds41rt_target_destroy(f.handle), OK);
    assert_eq!(ds41rt_target_destroy(f.handle), STALE);
}

#[test]
fn two_lane_actor_completion_is_independent_and_pages_publish_only_on_commit() {
    let f = fixture();
    f.control.execute[0].store(false, Ordering::SeqCst);
    let a = admit(&f, 0);
    let b = admit(&f, 1);
    let x = prepared(&f, 0, &a, &[1, 2]);
    let y = prepared(&f, 1, &b, &[3, 4]);
    rpc(f.handle, json!({"op":"execute","ticket":x}));
    execute(&f, &y);
    assert_eq!(
        rpc(f.handle, json!({"op":"info","request":b}))["committed_end"],
        0
    );
    assert_eq!(
        rpc(f.handle, json!({"op":"commit","ticket":y,"accepted":1}))["committed_end"],
        1
    );
    state(&f, &x, "executing");
    rpc(f.handle, json!({"op":"cancel","ticket":x}));
    rpc(f.handle, json!({"op":"release","request":a}));
    rpc(f.handle, json!({"op":"release","request":b}));
    assert_eq!(
        rpc(f.handle, json!({"op":"info"}))["bank"]["source_pages_free"],
        json!([8, 8, 8, 8])
    );
    close(f);
}

#[test]
fn result_borrow_survives_delayed_consumer_and_blocks_cancel_free_reuse_shutdown() {
    let f = fixture();
    f.control.consumer[0].store(false, Ordering::SeqCst);
    let a = admit(&f, 0);
    let b = admit(&f, 1);
    let x = prepared(&f, 0, &a, &[17, 42]);
    execute(&f, &x);
    let lease = rpc(f.handle, json!({"op":"acquire_logits","ticket":x}));
    let address = usize::from_str_radix(
        lease["device_pointer"]
            .as_str()
            .unwrap()
            .trim_start_matches("0x"),
        16,
    )
    .unwrap();
    assert_eq!(unsafe { *(address as *const f32) }, 42.0);
    error(f.handle, json!({"op":"cancel","ticket":x}), BUSY);
    error(f.handle, json!({"op":"release","request":a}), BUSY);
    error(f.handle, json!({"op":"shutdown"}), BUSY);
    let pending = submit(
        f.handle,
        json!({"op":"release_logits","ticket":x,"lease":lease["lease"],"consumer_stream":"0x1234"}),
    );
    state(&f, &x, "consumer_draining");
    let mut written = 0;
    assert_eq!(
        unsafe { ds41rt_target_result(f.handle, pending, ptr::null_mut(), 0, &mut written) },
        NOT_READY
    );
    assert_eq!(ds41rt_target_destroy(f.handle), BUSY);
    let y = prepared(&f, 1, &b, &[8]);
    execute(&f, &y);
    rpc(f.handle, json!({"op":"commit","ticket":y,"accepted":1}));
    assert_eq!(unsafe { *(address as *const f32) }, 42.0);
    assert_eq!(f.control.drains[0].load(Ordering::SeqCst), 0);
    f.control.consumer[0].store(true, Ordering::SeqCst);
    assert_eq!(collect(f.handle, pending)["result"]["state"], "consumed");
    assert_eq!(f.control.records[0].load(Ordering::SeqCst), 1);
    error(f.handle, json!({"op":"acquire_logits","ticket":x}), STALE);
    rpc(f.handle, json!({"op":"commit","ticket":x,"accepted":2}));
    close(f);
}

#[test]
fn foreign_tickets_generation_reuse_and_consumed_commands_are_rejected() {
    let f = fixture();
    let other = fixture();
    let a = admit(&f, 0);
    let b = admit(&other, 0);
    let x = prepared(&f, 0, &a, &[1]);
    let y = prepared(&other, 0, &b, &[2]);
    error(f.handle, json!({"op":"execute","ticket":y}), STALE);
    rpc(f.handle, json!({"op":"cancel","ticket":x}));
    rpc(f.handle, json!({"op":"release","request":a}));
    let fresh = admit(&f, 0);
    assert_ne!(a["generation"], fresh["generation"]);
    let next = prepared(&f, 0, &fresh, &[3]);
    error(f.handle, json!({"op":"cancel","ticket":x}), STALE);
    let id = submit(f.handle, json!({"op":"cancel","ticket":next}));
    collect(f.handle, id);
    let mut n = 0;
    assert_eq!(
        unsafe { ds41rt_target_result(f.handle, id, ptr::null_mut(), 0, &mut n) },
        STALE
    );
    rpc(other.handle, json!({"op":"cancel","ticket":y}));
    close(other);
    close(f);
}

#[test]
fn failed_execution_has_no_acquirable_output_or_committed_pages() {
    let f = fixture();
    f.control.fail[0].store(true, Ordering::SeqCst);
    let a = admit(&f, 0);
    let x = prepared(&f, 0, &a, &[9]);
    rpc(f.handle, json!({"op":"execute","ticket":x}));
    state(&f, &x, "failed");
    error(
        f.handle,
        json!({"op":"acquire_logits","ticket":x}),
        NOT_READY,
    );
    assert_eq!(
        rpc(f.handle, json!({"op":"info","request":a}))["committed_end"],
        0
    );
    rpc(f.handle, json!({"op":"cancel","ticket":x}));
    assert_eq!(
        rpc(f.handle, json!({"op":"info"}))["bank"]["source_pages_free"],
        json!([8, 8, 8, 8])
    );
    close(f);
}

#[test]
fn queue_bounds_and_reply_capacity_are_nondestructive() {
    let f = fixture();
    let mut ids = vec![];
    for _ in 0..LIMIT {
        ids.push(submit(f.handle, json!({"op":"info"})));
    }
    let malformed = b"not JSON";
    let mut id = 0;
    assert_eq!(
        unsafe { ds41rt_target_command(f.handle, malformed.as_ptr(), malformed.len(), &mut id) },
        BUSY
    );
    let mut required = 0;
    let deadline = Instant::now() + Duration::from_secs(5);
    while unsafe { ds41rt_target_result(f.handle, ids[0], ptr::null_mut(), 0, &mut required) }
        == NOT_READY
    {
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
    let mut too_small = vec![0; required - 1];
    assert_eq!(
        unsafe {
            ds41rt_target_result(
                f.handle,
                ids[0],
                too_small.as_mut_ptr(),
                too_small.len(),
                &mut required,
            )
        },
        BUFFER_TOO_SMALL
    );
    for id in ids {
        assert_eq!(collect(f.handle, id)["ok"], true);
    }
    assert!(f.client.state.lock().unwrap().replies.is_empty());
    close(f);
}

#[test]
fn input_json_is_owned_and_wrong_frontier_and_result_lease_fail_closed() {
    let f = fixture();
    let a = admit(&f, 0);
    error(
        f.handle,
        json!({"op":"submit","lane":0,"request":a,"expected_committed_end":9,
        "work":{"phase":"full_target","tokens":[1],"selected":[0],"kind":"prefill","placement":1}}),
        STALE,
    );
    let mut encoded=serde_json::to_vec(&json!({"op":"submit","lane":0,"request":a,"expected_committed_end":0,
        "work":{"phase":"full_target","tokens":[23],"selected":[0],"kind":"prefill","placement":1}})).unwrap();
    let mut id = 0;
    assert_eq!(
        unsafe { ds41rt_target_command(f.handle, encoded.as_ptr(), encoded.len(), &mut id) },
        OK
    );
    encoded.fill(b'x');
    let x = collect(f.handle, id)["result"]["ticket"].clone();
    execute(&f, &x);
    let lease = rpc(f.handle, json!({"op":"acquire_logits","ticket":x}));
    error(
        f.handle,
        json!({"op":"release_logits","ticket":x,"lease":9999,"consumer_stream":"0x1"}),
        STALE,
    );
    error(
        f.handle,
        json!({"op":"release_logits","ticket":x,"lease":lease["lease"],"consumer_stream":"bad"}),
        INVALID,
    );
    error(
        f.handle,
        json!({"op":"commit","ticket":x,"accepted":1}),
        BUSY,
    );
    rpc(
        f.handle,
        json!({"op":"release_logits","ticket":x,"lease":lease["lease"],"consumer_stream":"0x1"}),
    );
    rpc(f.handle, json!({"op":"cancel","ticket":x}));
    close(f);
}

#[test]
fn invalid_create_and_unknown_handles_never_load_native_model() {
    let bytes=br#"{"owner":0,"snapshot":"none","native_lib":"none","peers":["127.0.0.1:1","127.0.0.1:2","127.0.0.1:3","127.0.0.1:4"],"batch_tokens":80,"max_context_tokens":2048,"slots":2,"source_pool_budget_bytes":100}"#;
    let mut handle = 123;
    assert_eq!(
        unsafe { ds41rt_target_create(bytes.as_ptr(), bytes.len(), &mut handle) },
        INVALID
    );
    assert_eq!(handle, 0);
    assert_eq!(ds41rt_target_destroy(u64::MAX), STALE);
}

#[test]
fn capacity_query_uses_native_reservation_rules_without_retaining_pages() {
    let f = fixture();
    let a = admit(&f, 0);
    let b = admit(&f, 1);
    let before = rpc(f.handle, json!({"op":"info"}))["bank"].clone();
    // Each alone fits 8 ratio-one pages, together require 12; no query reserves.
    for request in [&a, &b] {
        assert_eq!(
            rpc(
                f.handle,
                json!({"op":"can_prepare","work":[{"request":request,"tokens":1536}]})
            )["can_prepare"],
            true
        );
    }
    assert_eq!(
        rpc(
            f.handle,
            json!({"op":"can_prepare","work":[{"request":a,"tokens":1536},{"request":b,"tokens":1536}]})
        )["can_prepare"],
        false
    );
    assert_eq!(rpc(f.handle, json!({"op":"info"}))["bank"], before);
    error(
        f.handle,
        json!({"op":"can_prepare","work":[{"request":a,"tokens":1},{"request":a,"tokens":1}]}),
        FAILED,
    );
    let x = prepared(&f, 0, &a, &[8]);
    error(
        f.handle,
        json!({"op":"can_prepare","work":[{"request":a,"tokens":1}]}),
        BUSY,
    );
    rpc(f.handle, json!({"op":"cancel","ticket":x}));
    rpc(f.handle, json!({"op":"release","request":a}));
    error(
        f.handle,
        json!({"op":"can_prepare","work":[{"request":a,"tokens":1048576}]}),
        FAILED,
    );
    close(f);
}

#[test]
fn invalid_accepted_count_preserves_ready_output_for_corrected_commit() {
    let f = fixture();
    let a = admit(&f, 0);
    let x = prepared(&f, 0, &a, &[8]);
    execute(&f, &x);
    error(
        f.handle,
        json!({"op":"commit","ticket":x,"accepted":2}),
        INVALID,
    );
    state(&f, &x, "ready");
    assert_eq!(f.control.drains[0].load(Ordering::SeqCst), 0);
    assert_eq!(
        rpc(f.handle, json!({"op":"commit","ticket":x,"accepted":1}))["committed_end"],
        1
    );
    close(f);
}

#[test]
fn simultaneous_callers_cannot_prepare_two_batches_on_one_lane() {
    let f = fixture();
    let a = admit(&f, 0);
    let b = admit(&f, 1);
    let barrier = Arc::new(std::sync::Barrier::new(3));
    let workers = [a, b].map(|request| {
        let barrier = barrier.clone();
        let handle = f.handle;
        std::thread::spawn(move || {
            barrier.wait();
            submit(
                handle,
                json!({"op":"submit","lane":0,
            "request":request,"expected_committed_end":0,"work":{"phase":"full_target",
            "tokens":[42],"selected":[0],"kind":"prefill","placement":1}}),
            )
        })
    });
    barrier.wait();
    let replies = workers.map(|worker| collect(f.handle, worker.join().unwrap()));
    assert_eq!(replies.iter().filter(|r| r["ok"] == true).count(), 1);
    assert_eq!(replies.iter().filter(|r| r["code"] == BUSY).count(), 1);
    let winner = replies.iter().find(|r| r["ok"] == true).unwrap()["result"]["ticket"].clone();
    rpc(f.handle, json!({"op":"cancel","ticket":winner}));
    close(f);
}

struct FakeStream<'a> {
    bank: &'a FakeBank,
    control: &'a Arc<Control>,
}
struct StreamGuard<'a> {
    bank: &'a FakeBank,
    control: &'a Arc<Control>,
    request: RequestHandle,
    reads: Vec<SourcePrefix>,
    armed: bool,
}
impl Drop for StreamGuard<'_> {
    fn drop(&mut self) {
        self.reads.clear();
        if self.armed {
            for lane in 0..2 {
                self.control.drains[lane].fetch_add(1, Ordering::SeqCst);
            }
            actor::Bank::release(self.bank, self.request).unwrap();
        }
    }
}
struct FakeReady<'a> {
    guard: StreamGuard<'a>,
    rows: usize,
    values: Vec<f32>,
    selected: Vec<usize>,
    positions: Vec<u64>,
}
impl stream::Output for FakeReady<'_> {
    fn logits(&self) -> Result<Logits<'_>> {
        Ok(Logits {
            rows: self.selected.len(),
            selected: &self.selected,
            positions: &self.positions,
            device: None,
            host: Some(&self.values),
        })
    }
    fn commit(mut self) -> Result<u64> {
        self.guard.reads.clear();
        let mut p = self.guard.bank.physical.borrow_mut();
        let slot = self.guard.request.slot();
        let plans = (0..4)
            .map(|s| {
                p.pages[s].reserve(&[(
                    slot,
                    p.pages[s].committed_rows(slot)?,
                    self.rows / if s == 3 { 1 } else { 2 },
                )])
            })
            .collect::<Result<Vec<_>>>()?;
        for (s, plan) in plans.into_iter().enumerate() {
            p.pages[s].apply(plan);
        }
        p.ends[slot] = Some(self.rows as u64);
        self.guard.armed = false;
        Ok(self.rows as u64)
    }
    fn cancel(self) -> Result<()> {
        drop(self);
        Ok(())
    }
}
impl stream::Backend for FakeStream<'_> {
    type Ready<'a>
        = FakeReady<'a>
    where
        Self: 'a;
    fn identity(&self, input: &StreamInput) -> Result<Ticket> {
        input.validate(80, 1048576)?;
        self.bank.active.idle(input.request)?;
        ensure!(
            actor::Bank::end(self.bank, input.request)? == 0,
            "stream requires fresh request"
        );
        self.bank.active.identity(input.request, 0)
    }
    async fn execute<'a>(&'a mut self, input: StreamInput) -> Result<Self::Ready<'a>> {
        // Emulate one internally published encoder source row with the retained
        // physical page algorithm. This is not an accepted full-model frontier.
        let slot = input.request.slot();
        {
            let mut p = self.bank.physical.borrow_mut();
            let plan = p.pages[3].reserve(&[(slot, 0, 1)])?;
            p.pages[3].apply(plan);
            p.ends[slot] = Some(1);
        }
        let reads = {
            let p = self.bank.physical.borrow();
            p.pages
                .iter()
                .map(|s| s.retain_prefix(slot, s.committed_rows(slot)?))
                .collect::<Result<Vec<_>>>()?
        };
        let guard = StreamGuard {
            bank: self.bank,
            control: self.control,
            request: input.request,
            reads,
            armed: true,
        };
        self.control.stream_stage.store(1, Ordering::SeqCst);
        while !self.control.execute[0].load(Ordering::SeqCst) {
            tokio::task::yield_now().await;
        }
        ensure!(
            !self.control.fail[0].load(Ordering::SeqCst),
            "controlled encoder failure"
        );
        self.control.stream_stage.store(2, Ordering::SeqCst);
        while !self.control.execute[1].load(Ordering::SeqCst) {
            tokio::task::yield_now().await;
        }
        ensure!(
            !self.control.fail[1].load(Ordering::SeqCst),
            "controlled replay failure"
        );
        let start = input.tokens.len().saturating_sub(128);
        let values = input
            .selected
            .iter()
            .flat_map(|&r| std::iter::repeat_n(input.tokens[r] as f32, 129280))
            .collect();
        self.control.stream_stage.store(3, Ordering::SeqCst);
        Ok(FakeReady {
            guard,
            rows: input.tokens.len(),
            values,
            selected: input.selected.iter().map(|r| r - start).collect(),
            positions: input.selected.iter().map(|&r| r as u64).collect(),
        })
    }
    fn revoke(&self, request: RequestHandle) -> Result<()> {
        if actor::Bank::end(self.bank, request).is_ok() {
            actor::Bank::release(self.bank, request)?;
        }
        Ok(())
    }
    fn info(&self) -> Value {
        actor::Bank::info(self.bank)
    }
}
fn stream_work(request: &Value, rows: usize) -> Value {
    json!({"op":"submit","lane":0,"request":request,
    "expected_committed_end":0,"work":{"phase":"encoder_stream","tokens":vec![42;rows],"chunk_rows":80,"selected":[rows-1]}})
}
fn stage(f: &Fixture, wanted: usize) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while f.control.stream_stage.load(Ordering::SeqCst) != wanted {
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
}

#[test]
fn streaming_mode_rejects_occupied_context_then_returns_to_full_target_after_commit() {
    let f = fixture();
    let a = admit(&f, 0);
    let b = admit(&f, 1);
    let normal = prepared(&f, 1, &b, &[3]);
    error(f.handle, stream_work(&a, 370), BUSY);
    rpc(f.handle, json!({"op":"cancel","ticket":normal}));
    let x = rpc(f.handle, stream_work(&a, 370))["ticket"].clone();
    error(
        f.handle,
        json!({"op":"submit","lane":1,"request":b,"expected_committed_end":0,
        "work":{"phase":"full_target","tokens":[3],"selected":[0],"kind":"prefill","placement":1}}),
        BUSY,
    );
    execute(&f, &x);
    error(
        f.handle,
        json!({"op":"commit","ticket":x,"accepted":128}),
        INVALID,
    );
    assert_eq!(
        rpc(f.handle, json!({"op":"commit","ticket":x,"accepted":370}))["committed_end"],
        370
    );
    let y = rpc(
        f.handle,
        json!({"op":"submit","lane":0,"request":a,"expected_committed_end":370,
        "work":{"phase":"full_target","tokens":[8],"selected":[0],"kind":"decode","placement":1}}),
    )["ticket"]
        .clone();
    assert_ne!(x["id"], y["id"]);
    execute(&f, &y);
    assert_eq!(
        rpc(f.handle, json!({"op":"commit","ticket":y,"accepted":1}))["committed_end"],
        371
    );
    close(f);
}
#[test]
fn streaming_cancellation_before_execute_encoder_and_replay_revokes_generation_and_credits() {
    for checkpoint in [0, 1, 2] {
        let f = fixture();
        let a = admit(&f, 0);
        let before = rpc(f.handle, json!({"op":"info"}))["bank"].clone();
        if checkpoint > 0 {
            f.control.execute[checkpoint - 1].store(false, Ordering::SeqCst);
        }
        let x = rpc(f.handle, stream_work(&a, 370))["ticket"].clone();
        if checkpoint > 0 {
            rpc(f.handle, json!({"op":"execute","ticket":x}));
            stage(&f, checkpoint);
        }
        assert_eq!(
            rpc(f.handle, json!({"op":"cancel","ticket":x}))["revoked"],
            true
        );
        assert_eq!(rpc(f.handle, json!({"op":"info"}))["bank"], before);
        error(f.handle, json!({"op":"release","request":a}), FAILED);
        let fresh = admit(&f, 0);
        assert_ne!(fresh["generation"], a["generation"]);
        let y = prepared(&f, 1, &fresh, &[4]);
        rpc(f.handle, json!({"op":"cancel","ticket":y}));
        close(f);
    }
}
#[test]
fn streaming_logits_fence_normalizes_selection_and_retains_both_contexts_until_cancel() {
    let f = fixture();
    let a = admit(&f, 0);
    let x = rpc(f.handle, stream_work(&a, 370))["ticket"].clone();
    execute(&f, &x);
    let lease = rpc(f.handle, json!({"op":"acquire_logits","ticket":x}));
    assert_eq!(lease["selected"], json!([369]));
    assert_eq!(lease["positions"], json!([369]));
    error(f.handle, json!({"op":"cancel","ticket":x}), BUSY);
    error(
        f.handle,
        json!({"op":"admit","slot":1,"request_id":88}),
        BUSY,
    );
    f.control.consumer[0].store(false, Ordering::SeqCst);
    let release = submit(
        f.handle,
        json!({"op":"release_logits","ticket":x,"lease":lease["lease"],"consumer_stream":"0x1234"}),
    );
    state(&f, &x, "consumer_draining");
    assert_eq!(f.control.drains[0].load(Ordering::SeqCst), 0);
    f.control.consumer[0].store(true, Ordering::SeqCst);
    collect(f.handle, release);
    error(
        f.handle,
        json!({"op":"commit","ticket":x,"accepted":128}),
        INVALID,
    );
    state(&f, &x, "consumed");
    assert_eq!(
        rpc(f.handle, json!({"op":"cancel","ticket":x}))["revoked"],
        true
    );
    assert_eq!(
        rpc(f.handle, json!({"op":"info"}))["bank"]["source_pages_free"],
        json!([8, 8, 8, 8])
    );
    close(f);
}
#[test]
fn failed_encoder_and_replay_require_revoked_cancellation_receipt_before_reentry() {
    for checkpoint in 0..2 {
        let f = fixture();
        let a = admit(&f, 0);
        f.control.fail[checkpoint].store(true, Ordering::SeqCst);
        let x = rpc(f.handle, stream_work(&a, 370))["ticket"].clone();
        rpc(f.handle, json!({"op":"execute","ticket":x}));
        state(&f, &x, "failed");
        error(
            f.handle,
            json!({"op":"acquire_logits","ticket":x}),
            NOT_READY,
        );
        assert_eq!(
            rpc(f.handle, json!({"op":"cancel","ticket":x}))["revoked"],
            true
        );
        assert_eq!(
            rpc(f.handle, json!({"op":"info"}))["bank"]["source_pages_free"],
            json!([8, 8, 8, 8])
        );
        let fresh = admit(&f, 0);
        let y = prepared(&f, 0, &fresh, &[1]);
        rpc(f.handle, json!({"op":"cancel","ticket":y}));
        close(f);
    }
}
#[test]
fn whole_million_token_stream_fits_command_bound_without_increasing_reply_or_config_bound() {
    let f = fixture();
    let a = admit(&f, 0);
    let mut work = stream_work(&a, 1048576);
    work["work"]["tokens"] = json!(vec![129279u32; 1048576]);
    let encoded = serde_json::to_vec(&work).unwrap();
    assert!(encoded.len() > MAX_JSON && encoded.len() < MAX_COMMAND);
    let id = submit(f.handle, work);
    let x = collect(f.handle, id)["result"]["ticket"].clone();
    assert_eq!(
        rpc(f.handle, json!({"op":"cancel","ticket":x}))["revoked"],
        true
    );
    let mut handle = 0;
    assert_eq!(
        unsafe { ds41rt_target_create(encoded.as_ptr(), encoded.len(), &mut handle) },
        INVALID
    );
    close(f);
}

#[test]
fn invalid_stream_input_preserves_fresh_admission_and_normal_mode_reentry() {
    let f = fixture();
    let a = admit(&f, 0);
    for field in ["chunk_rows", "selected", "tokens"] {
        let mut work = stream_work(&a, 370);
        work["work"][field] = match field {
            "chunk_rows" => json!(81), // AOT storage may cover more; requested budget does not.
            "selected" => json!([241]), // First permitted row is 242.
            _ => json!([129280]),
        };
        error(f.handle, work, FAILED);
        assert_eq!(
            rpc(f.handle, json!({"op":"info","request":a}))["committed_end"],
            0
        );
    }
    let mut wrong_lane = stream_work(&a, 370);
    wrong_lane["lane"] = json!(1);
    error(f.handle, wrong_lane, INVALID);
    let x = prepared(&f, 1, &a, &[8]);
    execute(&f, &x);
    rpc(f.handle, json!({"op":"commit","ticket":x,"accepted":1}));
    error(f.handle, stream_work(&a, 370), FAILED);
    assert_eq!(
        rpc(f.handle, json!({"op":"info","request":a}))["committed_end"],
        1
    );
    close(f);
}

#[path = "proposal_tests.rs"]
mod proposal_tests;
