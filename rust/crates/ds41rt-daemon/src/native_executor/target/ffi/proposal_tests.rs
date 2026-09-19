use super::*;

fn seed(f: &Fixture, lane: usize) -> Value {
    let request = admit(f, lane);
    let ticket = prepared(f, lane, &request, &[2, 3, 4, 5]);
    execute(f, &ticket);
    rpc(
        f.handle,
        json!({"op":"commit","ticket":ticket,"accepted":4}),
    );
    request
}
fn propose(f: &Fixture, lane: usize, request: &Value) -> u64 {
    submit(
        f.handle,
        json!({"op":"submit_speculative","lane":lane,"request":request,
        "expected_committed_end":4,"anchor":10,"remaining_output_tokens":4,"placement":1}),
    )
}
pub(super) fn wait_readers(f: &Fixture, lane: usize) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while f.control.proposal_readers[lane].load(Ordering::SeqCst) != 3 {
        assert!(
            Instant::now() < deadline,
            "native proposal readers not reserved"
        );
        std::thread::yield_now();
    }
}

#[test]
fn pending_native_proposal_can_cancel_by_command_while_peer_commits() {
    let f = fixture();
    let a = seed(&f, 0);
    let b = seed(&f, 1);
    f.control.proposal_ready[0].store(false, Ordering::SeqCst);
    let pending = propose(&f, 0, &a);
    wait_readers(&f, 0);
    let mut needed = 0;
    assert_eq!(
        unsafe { ds41rt_target_result(f.handle, pending, ptr::null_mut(), 0, &mut needed) },
        NOT_READY
    );
    error(f.handle, json!({"op":"release","request":a}), BUSY);
    let duplicate = propose(&f, 0, &b);
    assert_eq!(collect(f.handle, duplicate)["code"], BUSY);
    let other = collect(f.handle, propose(&f, 1, &b));
    assert_eq!(other["result"]["tokens"], json!([10, 11, 12, 13]));
    let ticket = &other["result"]["ticket"];
    execute(&f, ticket);
    let committed = rpc(
        f.handle,
        json!({"op":"commit","ticket":ticket,"accepted":2}),
    );
    assert_eq!(committed["committed_end"], 6);
    assert_eq!(committed["draft_committed_end"], 6);
    let canceled = rpc(
        f.handle,
        json!({"op":"cancel_proposal","command_id":pending}),
    );
    assert_eq!(canceled["state"], "proposal_cancelled");
    assert_eq!(canceled["committed_end"], 4);
    assert_eq!(canceled["draft_committed_end"], 4);
    let original = collect(f.handle, pending);
    assert_eq!(original["ok"], false);
    assert_eq!(original["code"], FAILED);
    assert_eq!(f.control.proposal_readers[0].load(Ordering::SeqCst), 0);
    assert_eq!(f.control.proposal_drains[0].load(Ordering::SeqCst), 1);
    rpc(f.handle, json!({"op":"release","request":a}));
    rpc(f.handle, json!({"op":"release","request":b}));
    close(f);
}

#[test]
fn completed_proposal_cancel_race_retains_original_reply_and_real_result_lease() {
    let f = fixture();
    let request = seed(&f, 0);
    let original = propose(&f, 0, &request);
    let deadline = Instant::now() + Duration::from_secs(5);
    while f.client.state.lock().unwrap().slots[0].phase != "prepared" {
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
    // Cancellation arrives after native completion but before the caller has
    // collected the original command. It must recover, not lose, that ticket.
    let race = rpc(
        f.handle,
        json!({"op":"cancel_proposal","command_id":original}),
    );
    assert_eq!(race["state"], "proposal_completed");
    let prepared = collect(f.handle, original);
    assert_eq!(prepared["result"]["ticket"], race["ticket"]);
    assert_eq!(prepared["result"]["tokens"], json!([10, 11, 12, 13]));
    let ticket = &race["ticket"];
    execute(&f, ticket);
    let lease = rpc(f.handle, json!({"op":"acquire_logits","ticket":ticket}));
    assert_eq!(lease["selected"], json!([0, 1, 2, 3]));
    assert_eq!(lease["positions"], json!([4, 5, 6, 7]));
    f.control.consumer[0].store(false, Ordering::SeqCst);
    let draining = submit(
        f.handle,
        json!({"op":"release_logits","ticket":ticket,
        "lease":lease["lease"],"consumer_stream":"0x123"}),
    );
    state(&f, ticket, "consumer_draining");
    error(f.handle, json!({"op":"cancel","ticket":ticket}), BUSY);
    f.control.consumer[0].store(true, Ordering::SeqCst);
    assert_eq!(collect(f.handle, draining)["ok"], true);
    let canceled = rpc(f.handle, json!({"op":"cancel","ticket":ticket}));
    assert_eq!(canceled["committed_end"], 4);
    assert_eq!(canceled["draft_committed_end"], 4);
    rpc(f.handle, json!({"op":"release","request":request}));
    close(f);
}

#[test]
fn ready_transition_and_command_cancel_never_lose_a_ticket_or_reader() {
    for _ in 0..16 {
        let f = fixture();
        let request = seed(&f, 0);
        f.control.proposal_ready[0].store(false, Ordering::SeqCst);
        let original = propose(&f, 0, &request);
        wait_readers(&f, 0);
        let cancellation = submit(
            f.handle,
            json!({"op":"cancel_proposal","command_id":original}),
        );
        f.control.proposal_ready[0].store(true, Ordering::SeqCst);
        let canceled = collect(f.handle, cancellation);
        assert_eq!(canceled["ok"], true, "{canceled}");
        let original_reply = collect(f.handle, original);
        match canceled["result"]["state"].as_str().unwrap() {
            "proposal_cancelled" => assert_eq!(original_reply["ok"], false),
            "proposal_completed" => {
                assert_eq!(original_reply["ok"], true);
                let ticket = &original_reply["result"]["ticket"];
                assert_eq!(*ticket, canceled["result"]["ticket"]);
                rpc(f.handle, json!({"op":"cancel","ticket":ticket}));
            }
            phase => panic!("unexpected phase {phase}"),
        }
        assert_eq!(f.control.proposal_readers[0].load(Ordering::SeqCst), 0);
        rpc(f.handle, json!({"op":"release","request":request}));
        close(f);
    }
}

#[test]
fn cancelled_command_cannot_cancel_a_new_generation_or_replaced_proposal() {
    let f = fixture();
    let request = seed(&f, 0);
    f.control.proposal_ready[0].store(false, Ordering::SeqCst);
    let original = propose(&f, 0, &request);
    wait_readers(&f, 0);
    rpc(
        f.handle,
        json!({"op":"cancel_proposal","command_id":original}),
    );
    assert_eq!(collect(f.handle, original)["ok"], false);
    f.control.proposal_ready[0].store(true, Ordering::SeqCst);
    let next = collect(f.handle, propose(&f, 0, &request));
    assert_eq!(next["ok"], true);
    error(
        f.handle,
        json!({"op":"cancel_proposal","command_id":original}),
        STALE,
    );
    error(
        f.handle,
        json!({"op":"cancel_proposal","command_id":u64::MAX}),
        STALE,
    );
    rpc(
        f.handle,
        json!({"op":"cancel","ticket":next["result"]["ticket"]}),
    );
    rpc(f.handle, json!({"op":"release","request":request}));
    close(f);
}

#[test]
fn config_requires_explicit_valid_native_draft_and_default_remains_disabled() {
    let base = json!({"owner":1,"snapshot":"missing","native_lib":"missing","peers":vec!["127.0.0.1:1";4],
        "batch_tokens":80,"max_context_tokens":8192,"slots":2,"source_pool_budget_bytes":1024});
    let config: Config = serde_json::from_value(base.clone()).unwrap();
    assert!(config.dspark.is_none());
    config.validate().unwrap();
    for draft in [
        json!({"draft_limit":0,"adaptive":false,"confidence_cutoff":null}),
        json!({"draft_limit":6,"adaptive":false,"confidence_cutoff":null}),
        json!({"draft_limit":5,"adaptive":true,"confidence_cutoff":0.5}),
        json!({"draft_limit":5,"adaptive":false,"confidence_cutoff":0.0}),
    ] {
        let mut value = base.clone();
        value["dspark"] = draft;
        assert!(serde_json::from_value::<Config>(value)
            .unwrap()
            .validate()
            .is_err());
    }
    let mut valid = base;
    valid["dspark"] = json!({"draft_limit":5,"adaptive":true,"confidence_cutoff":null});
    serde_json::from_value::<Config>(valid.clone())
        .unwrap()
        .validate()
        .unwrap();
    valid["slots"] = json!(1);
    assert!(serde_json::from_value::<Config>(valid)
        .unwrap()
        .validate()
        .is_err());
}
