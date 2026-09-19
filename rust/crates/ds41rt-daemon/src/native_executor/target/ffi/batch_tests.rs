use super::*;

fn member(request: &Value, end: u64, tokens: &[u32], selected: &[usize]) -> Value {
    json!({"request":request,"expected_committed_end":end,"tokens":tokens,"selected":selected,"kind":"decode"})
}
fn group(f: &Fixture, lane: usize, members: Vec<Value>) -> Value {
    rpc(
        f.handle,
        json!({"op":"submit_batch","lane":lane,"placement":1,"members":members}),
    )
}
fn commit(f: &Fixture, ticket: &Value, accepted: &[u32]) -> Value {
    rpc(
        f.handle,
        json!({"op":"commit_batch","ticket":ticket,"accepted":accepted}),
    )
}
fn release_all(f: &Fixture, requests: &[Value]) {
    for request in requests {
        rpc(f.handle, json!({"op":"release","request":request}));
    }
    assert_eq!(
        rpc(f.handle, json!({"op":"info"}))["bank"]["source_pages_free"],
        json!([8, 8, 8, 8])
    );
}
fn seeded(f: &Fixture) -> Vec<Value> {
    (0..4)
        .map(|slot| {
            let request = admit(f, slot);
            let ticket = prepared(f, slot % 2, &request, &[2, 3, 4, 5]);
            execute(f, &ticket);
            rpc(
                f.handle,
                json!({"op":"commit","ticket":ticket,"accepted":4}),
            );
            request
        })
        .collect()
}
fn draft(f: &Fixture, lane: usize, members: &[(&Value, usize)]) -> u64 {
    submit(
        f.handle,
        json!({"op":"submit_speculative_batch","lane":lane,"placement":1,
        "members":members.iter().map(|(r,n)| json!({"request":r,"expected_committed_end":4,
            "anchor":10,"remaining_output_tokens":n})).collect::<Vec<_>>()}),
    )
}

#[test]
fn four_requests_advance_in_two_real_group_driver_calls() {
    let f = fixture_with_head(48);
    let requests: Vec<_> = (0..4).map(|slot| admit(&f, slot)).collect();
    let x = group(
        &f,
        0,
        vec![
            member(&requests[0], 0, &[11], &[0]),
            member(&requests[2], 0, &[22], &[0]),
        ],
    );
    let y = group(
        &f,
        1,
        vec![
            member(&requests[1], 0, &[33], &[0]),
            member(&requests[3], 0, &[44], &[0]),
        ],
    );
    assert_eq!(x["members"][1]["request"], requests[2]);
    assert_eq!(x["members"][1]["input_offset"], 1);
    assert_eq!(x["members"][1]["output_offset"], 1);
    execute(&f, &x["ticket"]);
    execute(&f, &y["ticket"]);
    for lane in 0..2 {
        assert_eq!(f.control.passes[lane].load(Ordering::SeqCst), 1);
        assert_eq!(f.control.grouped_members[lane].load(Ordering::SeqCst), 2);
    }
    for prepared in [&x, &y] {
        let end = commit(&f, &prepared["ticket"], &[1, 1]);
        assert_eq!(end["members"][0]["committed_end"], 1);
        assert_eq!(end["members"][1]["committed_end"], 1);
    }
    release_all(&f, &requests);
    close(f);
}

#[test]
fn unequal_member_rows_have_compact_outputs_and_one_whole_group_fence() {
    let f = fixture_with_head(48);
    let requests = seeded(&f);
    let x = group(
        &f,
        0,
        vec![
            member(&requests[0], 4, &[10, 11, 12], &[0, 2]),
            member(&requests[2], 4, &[20, 21], &[1]),
        ],
    );
    assert_eq!(x["members"][1]["input_offset"], 3);
    assert_eq!(x["members"][1]["output_offset"], 2);
    execute(&f, &x["ticket"]);
    let output = rpc(
        f.handle,
        json!({"op":"acquire_logits","ticket":x["ticket"]}),
    );
    assert_eq!(output["selected"], json!([0, 2, 4]));
    assert_eq!(output["positions"], json!([4, 6, 5]));
    let ptr = usize::from_str_radix(
        output["device_pointer"]
            .as_str()
            .unwrap()
            .trim_start_matches("0x"),
        16,
    )
    .unwrap() as *const f32;
    assert_eq!(
        unsafe { [*ptr, *ptr.add(129280), *ptr.add(2 * 129280)] },
        [10., 12., 21.]
    );
    for request in [&requests[0], &requests[2]] {
        error(f.handle, json!({"op":"release","request":request}), BUSY);
    }
    error(
        f.handle,
        json!({"op":"commit_batch","ticket":x["ticket"],"accepted":[1,1]}),
        BUSY,
    );
    f.control.consumer[0].store(false, Ordering::SeqCst);
    let drain = submit(
        f.handle,
        json!({"op":"release_logits","ticket":x["ticket"],"lease":output["lease"],"consumer_stream":"0x1"}),
    );
    state(&f, &x["ticket"], "consumer_draining");
    let y = group(
        &f,
        1,
        vec![
            member(&requests[1], 4, &[30], &[0]),
            member(&requests[3], 4, &[40], &[0]),
        ],
    );
    execute(&f, &y["ticket"]);
    commit(&f, &y["ticket"], &[1, 1]);
    assert_eq!(unsafe { *ptr.add(2 * 129280) }, 21.);
    f.control.consumer[0].store(true, Ordering::SeqCst);
    assert_eq!(collect(f.handle, drain)["ok"], true);
    let committed = commit(&f, &x["ticket"], &[2, 0]);
    assert_eq!(committed["members"][0]["committed_end"], 6);
    assert_eq!(committed["members"][1]["committed_end"], 4);
    error(
        f.handle,
        json!({"op":"acquire_logits","ticket":x["ticket"]}),
        STALE,
    );
    release_all(&f, &requests);
    close(f);
}

#[test]
fn grouped_commit_waits_without_global_lock_or_partial_publication() {
    let f = fixture_with_head(48);
    let requests = seeded(&f);
    let x = group(
        &f,
        0,
        vec![
            member(&requests[0], 4, &[10], &[0]),
            member(&requests[2], 4, &[20], &[0]),
        ],
    );
    let y = group(
        &f,
        1,
        vec![
            member(&requests[1], 4, &[30], &[0]),
            member(&requests[3], 4, &[40], &[0]),
        ],
    );
    execute(&f, &x["ticket"]);
    execute(&f, &y["ticket"]);
    f.control.commit_ready[0].store(false, Ordering::SeqCst);
    let pending = submit(
        f.handle,
        json!({"op":"commit_batch","ticket":x["ticket"],"accepted":[1,0]}),
    );
    let deadline = Instant::now() + Duration::from_secs(5);
    while !f.control.commit_started[0].load(Ordering::SeqCst) {
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
    for request in [&requests[0], &requests[2]] {
        assert_eq!(
            rpc(f.handle, json!({"op":"info","request":request}))["committed_end"],
            4
        );
        error(f.handle, json!({"op":"release","request":request}), BUSY);
    }
    error(f.handle, json!({"op":"cancel","ticket":x["ticket"]}), BUSY);
    commit(&f, &y["ticket"], &[1, 1]);
    f.control.commit_ready[0].store(true, Ordering::SeqCst);
    let reply = collect(f.handle, pending);
    assert_eq!(reply["ok"], true);
    assert_eq!(reply["result"]["members"][0]["committed_end"], 5);
    assert_eq!(reply["result"]["members"][1]["committed_end"], 4);
    release_all(&f, &requests);
    close(f);
}

#[test]
fn invalid_batch_and_acceptance_are_atomic_and_nondestructive() {
    let f = fixture_with_head(48);
    let a = admit(&f, 0);
    let b = admit(&f, 1);
    let submit_value =
        |members| json!({"op":"submit_batch","lane":0,"placement":1,"members":members});
    error(
        f.handle,
        submit_value(vec![member(&a, 0, &[1], &[0]), member(&a, 0, &[2], &[0])]),
        INVALID,
    );
    error(
        f.handle,
        submit_value(vec![member(&a, 0, &[1], &[0]), member(&b, 1, &[2], &[0])]),
        STALE,
    );
    error(
        f.handle,
        submit_value(vec![
            member(&a, 0, &[1; 10], &[0]),
            member(&b, 0, &[2; 10], &[0]),
        ]),
        FAILED,
    );
    let ticket = group(
        &f,
        0,
        vec![member(&a, 0, &[1, 2], &[1]), member(&b, 0, &[3], &[0])],
    )["ticket"]
        .clone();
    error(
        f.handle,
        json!({"op":"submit","lane":1,"request":b,"expected_committed_end":0,
        "work":{"phase":"full_target","tokens":[1],"selected":[0],"kind":"decode","placement":1}}),
        BUSY,
    );
    execute(&f, &ticket);
    error(
        f.handle,
        json!({"op":"commit","ticket":ticket,"accepted":1}),
        INVALID,
    );
    for accepted in [vec![1], vec![3, 1], vec![1, 1, 1]] {
        error(
            f.handle,
            json!({"op":"commit_batch","ticket":ticket,"accepted":accepted}),
            INVALID,
        );
    }
    let end = commit(&f, &ticket, &[0, 0]);
    assert_eq!(end["members"][0]["committed_end"], 0);
    assert_eq!(end["members"][1]["committed_end"], 0);
    release_all(&f, &[a, b]);
    close(f);
}

#[test]
fn failed_execution_cancels_all_members_without_suffix_publication() {
    let f = fixture_with_head(48);
    let requests = seeded(&f);
    let x = group(
        &f,
        0,
        vec![
            member(&requests[0], 4, &[10], &[0]),
            member(&requests[2], 4, &[20], &[0]),
        ],
    );
    f.control.fail[0].store(true, Ordering::SeqCst);
    rpc(f.handle, json!({"op":"execute","ticket":x["ticket"]}));
    state(&f, &x["ticket"], "failed");
    error(
        f.handle,
        json!({"op":"acquire_logits","ticket":x["ticket"]}),
        NOT_READY,
    );
    let cancelled = rpc(f.handle, json!({"op":"cancel","ticket":x["ticket"]}));
    for row in cancelled["members"].as_array().unwrap() {
        assert_eq!(row["revoked"], false);
        assert_eq!(row["committed_end"], 4);
    }
    release_all(&f, &requests);
    close(f);
}

#[test]
fn joint_publication_failure_revokes_all_including_zero_accepted_member() {
    let f = fixture_with_head(48);
    let a = admit(&f, 0);
    let b = admit(&f, 1);
    let x = group(
        &f,
        0,
        vec![member(&a, 0, &[10], &[0]), member(&b, 0, &[20], &[0])],
    );
    execute(&f, &x["ticket"]);
    f.control.fail_commit[0].store(true, Ordering::SeqCst);
    error(
        f.handle,
        json!({"op":"commit_batch","ticket":x["ticket"],"accepted":[1,0]}),
        FAILED,
    );
    for request in [&a, &b] {
        error(f.handle, json!({"op":"info","request":request}), FAILED);
    }
    assert_eq!(
        rpc(f.handle, json!({"op":"info"}))["bank"]["source_pages_free"],
        json!([8, 8, 8, 8])
    );
    close(f);
}

#[test]
fn mixed_k0_and_speculative_members_use_one_pass_and_independent_accepted_prefixes() {
    let f = fixture_with_head(48);
    let requests = seeded(&f);
    f.control.passes[0].store(0, Ordering::SeqCst);
    let ack = collect(
        f.handle,
        draft(&f, 0, &[(&requests[0], 4), (&requests[2], 1)]),
    );
    assert_eq!(ack["ok"], true);
    let result = &ack["result"];
    assert_eq!(result["members"][0]["tokens"], json!([10, 11, 12, 13]));
    assert_eq!(result["members"][1]["tokens"], json!([10]));
    assert_eq!(result["members"][1]["output_offset"], 4);
    execute(&f, &result["ticket"]);
    assert_eq!(f.control.passes[0].load(Ordering::SeqCst), 1);
    let end = commit(&f, &result["ticket"], &[2, 1]);
    assert_eq!(end["members"][0]["committed_end"], 6);
    assert_eq!(end["members"][1]["committed_end"], 5);
    for m in end["members"].as_array().unwrap() {
        assert_eq!(m["committed_end"], m["draft_committed_end"]);
    }
    release_all(&f, &requests);
    close(f);
}

#[test]
fn pending_group_proposal_cancellation_drains_every_native_reader() {
    let f = fixture_with_head(48);
    let requests = seeded(&f);
    f.control.proposal_ready[0].store(false, Ordering::SeqCst);
    let original = draft(&f, 0, &[(&requests[0], 3), (&requests[2], 1)]);
    proposal_tests::wait_readers(&f, 0);
    for request in [&requests[0], &requests[2]] {
        error(f.handle, json!({"op":"release","request":request}), BUSY);
    }
    let peer = collect(
        f.handle,
        draft(&f, 1, &[(&requests[1], 2), (&requests[3], 2)]),
    );
    execute(&f, &peer["result"]["ticket"]);
    commit(&f, &peer["result"]["ticket"], &[1, 2]);
    let cancelled = rpc(
        f.handle,
        json!({"op":"cancel_proposal","command_id":original}),
    );
    assert_eq!(cancelled["state"], "proposal_cancelled");
    assert_eq!(cancelled["members"].as_array().unwrap().len(), 2);
    assert_eq!(collect(f.handle, original)["code"], FAILED);
    assert_eq!(f.control.proposal_readers[0].load(Ordering::SeqCst), 0);
    assert_eq!(f.control.proposal_drains[0].load(Ordering::SeqCst), 1);
    release_all(&f, &requests);
    close(f);
}

#[test]
fn group_proposal_ready_cancel_race_preserves_manifest_ticket_and_all_leases() {
    for _ in 0..16 {
        let f = fixture_with_head(48);
        let requests = seeded(&f);
        f.control.proposal_ready[0].store(false, Ordering::SeqCst);
        let original = draft(&f, 0, &[(&requests[0], 3), (&requests[2], 1)]);
        proposal_tests::wait_readers(&f, 0);
        let cancellation = submit(
            f.handle,
            json!({"op":"cancel_proposal","command_id":original}),
        );
        f.control.proposal_ready[0].store(true, Ordering::SeqCst);
        let cancelled = collect(f.handle, cancellation);
        assert_eq!(cancelled["ok"], true);
        let reply = collect(f.handle, original);
        match cancelled["result"]["state"].as_str().unwrap() {
            "proposal_cancelled" => assert_eq!(reply["code"], FAILED),
            "proposal_completed" => {
                assert_eq!(reply["ok"], true);
                assert_eq!(reply["result"]["ticket"], cancelled["result"]["ticket"]);
                assert_eq!(reply["result"]["members"].as_array().unwrap().len(), 2);
                let cancelled = rpc(
                    f.handle,
                    json!({"op":"cancel","ticket":reply["result"]["ticket"]}),
                );
                assert_eq!(cancelled["members"].as_array().unwrap().len(), 2);
            }
            phase => panic!("unexpected cancellation phase {phase}"),
        }
        release_all(&f, &requests);
        close(f);
    }
}

#[test]
fn dropped_group_commit_retains_native_reservations_until_explicit_drain() -> Result<()> {
    use actor::Bank as _;
    use std::{
        future::Future,
        pin::pin,
        task::{Context, Poll, Waker},
    };
    let active = Rc::new(Active::default());
    let bank = FakeBank {
        physical: Rc::new(RefCell::new(Physical {
            owner: 11,
            pages: std::array::from_fn(|_| SourcePages::new(8, 4).unwrap()),
            ends: [None; 4],
            generations: [0; 4],
        })),
        active: active.clone(),
        draft_access: Rc::new(std::array::from_fn(|_| SlotAccess::default())),
    };
    let requests = [bank.admit(0, 1)?, bank.admit(1, 2)?];
    let control = Arc::new(Control::new());
    let mut context = TargetContext::new(
        Driver {
            bank: bank.clone(),
            control: control.clone(),
            lane: 0,
            output: vec![],
            draft_readers: None,
        },
        active,
        0,
        16,
        48,
    );
    let ticket = context.submit_batch(
        requests
            .iter()
            .map(|&request| TargetInput {
                request,
                tokens: vec![10, 11],
                selected: vec![1],
                kind: SourceKind::Decode,
                placement: 0,
            })
            .collect(),
    )?;
    let mut cx = Context::from_waker(Waker::noop());
    assert!(matches!(
        pin!(context.execute(ticket)).as_mut().poll(&mut cx),
        Poll::Ready(Ok(()))
    ));
    control.commit_ready[0].store(false, Ordering::SeqCst);
    {
        let mut commit = pin!(context.commit_batch(ticket, &[2, 0]));
        assert!(commit.as_mut().poll(&mut cx).is_pending());
    }
    // Future drop cannot advertise old logits or release either live member.
    assert!(context.logits(ticket).is_err());
    for request in requests {
        assert!(bank.release(request).is_err());
        assert_eq!(bank.end(request)?, 0);
    }
    assert!(bank.draft_access[0].writable(0).is_err());
    // Native draft commits do not write zero-accepted members, whose public
    // lease is nevertheless protected by the group until drain.
    bank.draft_access[0].writable(1)?;
    assert!(bank.physical.borrow().pages[3].free_pages() < 8);
    context.cancel(ticket)?;
    assert_eq!(control.drains[0].load(Ordering::SeqCst), 1);
    for request in requests {
        assert!(bank.end(request).is_err());
        bank.draft_access[0].writable(request.slot())?;
    }
    assert_eq!(bank.info()["source_pages_free"], json!([8, 8, 8, 8]));
    Ok(())
}
