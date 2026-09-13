//! Lane-local rounds; shared owners are borrowed only during synchronous work.
use super::*;
use std::cell::{Cell, RefCell};

pub(super) fn run<'w, 'a>(lib: &'a NativeLibrary, runtime: &tokio::runtime::Runtime,
    first: &mut TargetPass<'w, 'a>, second: &mut TargetPass<'w, 'a>, requests: &mut Requests<'a>,
    first_transport: &mut NativeTp4Wave<'a>, second_transport: &mut NativeTp4Wave<'a>,
    active: &mut [Option<Active<'a>>], draft: Option<&mut DraftRuntime<'_, 'a>>,
    prefixes: &mut PrefixCache<'a>, receive: &mpsc::Receiver<NativeRequest>,
) -> Result<()> {
    let requests = RefCell::new(requests);
    let active = RefCell::new(active);
    let draft = RefCell::new(draft);
    let prefixes = RefCell::new(prefixes);
    let drain = Cell::new(false);
    // Do not cancel the peer future on error: it may own queued CUDA/RDMA work.
    let results = runtime.block_on(async { tokio::join!(
        lane(0, lib, first, first_transport, &requests, &active, &draft, &prefixes, receive, &drain),
        lane(1, lib, second, second_transport, &requests, &active, &draft, &prefixes, receive, &drain),
    ) });
    results.0?; results.1?;
    Ok(())
}

async fn lane<'w, 'a>(lane: usize, lib: &'a NativeLibrary, pass: &mut TargetPass<'w, 'a>,
    transport: &mut NativeTp4Wave<'a>, requests: &RefCell<&mut Requests<'a>>,
    active: &RefCell<&mut [Option<Active<'a>>]>, draft: &RefCell<Option<&mut DraftRuntime<'_, 'a>>>,
    prefixes: &RefCell<&mut PrefixCache<'a>>, receive: &mpsc::Receiver<NativeRequest>, drain: &Cell<bool>,
) -> Result<()> {
    let result = async {
        let mut round_id = 0u64;
        loop {
            if drain.get() { return Ok(()); }
            // Prefill uses both execution lanes. Retire/admit/migrate only after
            // draining the current stacks, never merely to align decode rounds.
            let members: Vec<_> = {
                let active = active.borrow();
                if active.iter().flatten().any(|r| r.finished || r.job.events.is_closed())
                    || (active.iter().any(Option::is_none) && !receive.is_empty()) {
                    drain.set(true); return Ok(());
                }
                active.iter().enumerate().filter_map(|(slot, r)|
                    r.as_ref().filter(|r| r.lane == lane).map(|_| slot)).collect()
            };
            if members.is_empty() { return Ok(()); }
            ensure!(members.len() <= 8, "independent lane exceeds eight requests");
            let started = Instant::now();
            // Reserve target capacity and snapshot this lane's seeds, then release
            // all bank borrows before waiting on the shared draft workspace.
            let seeds = {
                let active = active.borrow();
                let mut requests = requests.borrow_mut();
                let speculative = draft.borrow().is_some();
                let capacity: Vec<_> = members.iter().map(|&slot| {
                    let r = active[slot].as_ref().unwrap();
                    (r.lease, if speculative { (r.job.max_tokens-r.generated).min(6) as u32 } else { 1 })
                }).collect();
                prefixes.borrow_mut().make_room(&mut requests, &capacity)?;
                members.iter().map(|&slot| {
                    let r = active[slot].as_ref().unwrap();
                    Ok((r.id, r.anchor, requests.cache().committed_end(r.lease)?, r.job.max_tokens-r.generated))
                }).collect::<Result<Vec<_>>>()?
            };
            let (mut inputs, draft_us) = loop {
                let proposed = {
                    let mut draft = draft.borrow_mut();
                    if let Some(draft) = draft.as_deref_mut() { draft.poll_propose(lane, &seeds)? }
                    else { Some((seeds.iter().map(|r| vec![r.1]).collect(), 0)) }
                };
                if let Some(inputs) = proposed { break inputs; }
                // Even when a peer requests a cohort drain, finish our pending
                // draft and transaction before retirement can recycle its slots.
                tokio::task::yield_now().await;
            };
            let (inputs, mut batch, capture_routes) = {
                let active = active.borrow();
                let mut requests = requests.borrow_mut();
                let draft = draft.borrow();
                for (&slot, input) in members.iter().zip(&mut inputs) {
                    let r = active[slot].as_ref().unwrap();
                    if let Some(constraint) = &r.constraint { constraint.truncate_proposal(input)?; }
                    else if let Some(draft) = draft.as_deref() {
                        input.truncate(draft.confidence_prefix(r.id, input.len()-1)? + 1);
                    }
                }
                if members.iter().all(|&slot| active[slot].as_ref().unwrap().constraint.is_none()) {
                    if let Some(draft) = draft.as_deref().filter(|d| d.reuse_enabled() || d.adaptive_enabled()) {
                        let candidates: Vec<_> = members.iter().zip(&inputs).map(|(&slot, input)|
                            (active[slot].as_ref().unwrap().id, lane, input.len()-1)).collect();
                        // Only this lane contributes proposals, route unions and
                        // draft time. The existing cost model's cross-lane term
                        // is zero for this single-lane forecast.
                        let lengths = if draft.adaptive_enabled() {
                            draft.select_prefixes(&candidates, draft_us)?
                        } else { draft.select_reuse_prefixes(&candidates)? };
                        if let Some(lengths) = lengths {
                            for (input, length) in inputs.iter_mut().zip(lengths) { input.truncate(length+1); }
                        }
                    }
                }
                let batch = prepare_decode_lane(&mut requests, &active, &members, &inputs, draft.is_some())?;
                (inputs, Some(batch), draft.as_deref().is_some_and(DraftRuntime::capture_routes))
            };
            let prepared_us = started.elapsed().as_micros() as u64;
            round_id += 1;
            tracing::debug!(target: "ds41rt::lane_schedule", lane, round_id, requests=members.len(),
                "independent verifier issued");
            pass.set_route_capture(capture_routes);
            let operation: Result<()> = async {
                let current = batch.as_mut().unwrap();
                let selected: Vec<_> = (0..current.cache()?.positions().len()).collect();
                let compact = !tracing::enabled!(target: "ds41rt::logit_trace", tracing::Level::DEBUG)
                    && members.iter().all(|&slot| active.borrow()[slot].as_ref().unwrap().constraint.is_none());
                let next = if compact {
                    BatchScores::from_greedy(unsafe {
                        pass.execute_shared_greedy(requests, current, transport, 0, &selected).await?
                    })?
                } else {
                    let logits = unsafe { pass.execute_shared(requests, current, transport, 0, &selected).await? };
                    let mut bytes = vec![0; logits.logits.bytes];
                    lib.copy_d2h(&mut bytes, logits.logits)?;
                    BatchScores::new(bytes)?
                };
                let verify_us = started.elapsed().as_micros() as u64 - prepared_us;
                let (accepted, emitted, emissions) = commit_lane(lib, lane, pass, &mut requests.borrow_mut(),
                    &mut active.borrow_mut(), &members, &inputs, &mut batch, &next,
                    draft.borrow_mut().as_deref_mut(), capture_routes, verify_us)?;
                tracing::debug!(target: "ds41rt::lane_schedule", lane, round_id,
                    "independent verifier committed");
                for (&slot, tokens) in members.iter().zip(emissions) {
                    let sender = active.borrow()[slot].as_ref().unwrap().job.events.clone();
                    let delivered: Result<()> = async {
                        for token in tokens {
                            let (chunks, finished) = {
                                let mut active = active.borrow_mut();
                                let request = active[slot].as_mut().unwrap();
                                (request.emit_one(token)?, request.finished)
                            };
                            for chunk in chunks.into_iter().flatten() { sender.send(Ok(chunk)).await?; }
                            if finished { break; }
                        }
                        Ok(())
                    }.await;
                    if let Err(error) = delivered {
                        active.borrow_mut()[slot].as_mut().unwrap().finished = true;
                        let _ = sender.send(Err(format!("{error:#}").into())).await;
                    }
                }
                tracing::debug!(target: "ds41rt::timing", lane, requests=members.len(),
                    proposed=inputs.iter().map(|r| r.len()-1).sum::<usize>(), accepted, emitted,
                    prepared_us, verify_us, total_us=started.elapsed().as_micros() as u64,
                    "native independent lane round");
                Ok(())
            }.await;
            pass.set_route_capture(false);
            if let Some(batch) = &mut batch {
                // Successful commits relinquish ownership. Failed execution or
                // commit must drain/discard this lane before the outer reset.
                pass.discard(batch)?;
            }
            operation?;
            // Give already-ready remote completions an opportunity to run before
            // queuing another draft on the shared RTX.
            tokio::task::yield_now().await;
        }
    }.await;
    if result.is_err() { drain.set(true); }
    result
}
