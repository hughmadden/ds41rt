use super::*;

pub(super) trait Bank {
    fn info(&self) -> Value;
    fn admit(&self, slot: usize, id: u64) -> Result<RequestHandle>;
    fn end(&self, request: RequestHandle) -> Result<u64>;
    fn draft_end(&self, _request: RequestHandle) -> Result<Option<u64>> { Ok(None) }
    fn release(&self, request: RequestHandle) -> Result<()>;
    fn can_prepare(&self, work: &[(RequestHandle, u32)]) -> Result<bool>;
}
/// Native owns both the completion event and its private wait stream. A CPU
/// implementation controls completion explicitly without accepting ready claims.
pub(super) trait Fence {
    fn record(&mut self, consumer_stream: usize) -> Result<()>;
    fn ready(&self) -> Result<bool>;
}
pub(super) enum LaneMessage {
    Command(Message),
    Stop,
}
pub(super) fn active(slot: &Slot) -> bool {
    slot.pending
        || matches!(
            slot.phase,
            "prepared"
                | "executing"
                | "ready"
                | "leased"
                | "consumer_draining"
                | "consumed"
                | "failed"
        )
}
pub(super) fn reserve(client: &Client, message: &Message) -> Api<usize> {
    let lane = message
        .command
        .lane()
        .ok_or_else(|| fail(INVALID, "not a lane command"))?;
    let mut shared = client
        .state
        .lock()
        .map_err(|_| fail(INTERNAL, "client state poisoned"))?;
    let slot = &shared.slots[lane];
    if slot.pending {
        return Err(fail(BUSY, "lane command pending"));
    }
    let submissions = match &message.command {
        Command::Submit { request, .. } | Command::SubmitSpeculative { request, .. } => Some(vec![*request]),
        Command::SubmitBatch { members, .. } => Some(members.iter().map(|m| m.request).collect()),
        Command::SubmitSpeculativeBatch { members, .. } => Some(members.iter().map(|m| m.request).collect()),
        _ => None,
    };
    if let Some(members) = submissions {
        if members.is_empty() || members.len() > 8 || members.iter().enumerate().any(|(i, r)| members[..i].contains(r)) {
            return Err(fail(INVALID, "invalid or duplicate batch members"));
        }
        if active(slot) || shared.slots.iter().any(|s| active(s) && members.iter().any(|r| s.members.contains(r))) {
            return Err(fail(BUSY, "lane or request is already owned"));
        }
        let proposing = matches!(message.command, Command::SubmitSpeculative { .. } | Command::SubmitSpeculativeBatch { .. });
        shared.slots[lane] = Slot {
            request: Some(members[0]), members,
            grouped: matches!(message.command, Command::SubmitBatch { .. } | Command::SubmitSpeculativeBatch { .. }),
            pending: true,
            phase: if proposing { "proposing" } else { "" },
            proposal_command: proposing.then_some(message.id),
            ..Default::default()
        };
    } else {
        if slot.ticket != message.command.ticket() {
            return Err(fail(STALE, "foreign or stale target ticket"));
        }
        let allowed = match &message.command {
            Command::Execute { .. } => slot.phase == "prepared",
            Command::AcquireLogits { .. } => slot.phase == "ready",
            Command::ReleaseLogits { .. } => slot.phase == "leased",
            Command::Commit { .. } | Command::CommitBatch { .. } => matches!(slot.phase, "ready" | "consumed"),
            Command::Cancel { .. } => matches!(
                slot.phase,
                "prepared" | "executing" | "ready" | "consumed" | "failed"
            ),
            _ => false,
        };
        if !allowed {
            let code = if matches!(slot.phase, "leased" | "consumer_draining") {
                BUSY
            } else if matches!(slot.phase, "committed" | "cancelled" | "consumed") {
                STALE
            } else {
                NOT_READY
            };
            return Err(fail(
                code,
                format!("operation unavailable in {}", slot.phase),
            ));
        }
        shared.slots[lane].pending = true;
    }
    Ok(lane)
}
pub(super) fn clear_pending(client: &Client, lane: usize) {
    client.state.lock().expect("client state").slots[lane].pending = false;
}

pub(super) enum Exit {
    Shutdown(u64),
    Stream(Message),
}

pub(super) async fn run<B: Bank, D: SpeculativeDriver, F: Fence>(
    bank: &B,
    contexts: [&mut TargetContext<D>; 2],
    fences: &mut [F; 2],
    client: Arc<Client>,
    receive: &mut mpsc::Receiver<Message>,
) -> Result<Exit> {
    let (tx0, rx0) = mpsc::channel(LIMIT);
    let (tx1, rx1) = mpsc::channel(LIMIT);
    let [first, second] = contexts;
    let [fence0, fence1] = fences;
    let (controller, (), ()) = tokio::join!(
        control(bank, client.clone(), receive, [tx0, tx1]),
        lane(bank, first, fence0, client.clone(), rx0),
        lane(bank, second, fence1, client, rx1)
    );
    controller
}

async fn control<B: Bank>(
    bank: &B,
    client: Arc<Client>,
    receive: &mut mpsc::Receiver<Message>,
    lanes: [mpsc::Sender<LaneMessage>; 2],
) -> Result<Exit> {
    while let Some(message) = receive.recv().await {
        let id = message.id;
        match &message.command {
            Command::Submit {
                lane,
                work: Work::EncoderStream { .. },
                ..
            } => {
                if *lane != 0 {
                    client.reply(id, Err(fail(INVALID, "encoder stream must use lane zero")));
                } else if client
                    .state
                    .lock()
                    .expect("client state")
                    .slots
                    .iter()
                    .any(active)
                {
                    client.reply(
                        id,
                        Err(fail(BUSY, "encoder stream requires both contexts idle")),
                    );
                } else {
                    for lane in &lanes {
                        let _ = lane.send(LaneMessage::Stop).await;
                    }
                    return Ok(Exit::Stream(message));
                }
            }
            Command::CancelProposal { command_id } => {
                let lane = {
                    let state = client.state.lock().expect("client state");
                    state.slots.iter().position(|slot| slot.proposal_command == Some(*command_id)
                        && slot.phase == "proposing")
                };
                if let Some(lane) = lane {
                    if lanes[lane].send(LaneMessage::Command(message)).await.is_err() {
                        client.reply(id, Err(fail(FAILED, "native proposal lane stopped")));
                    }
                } else { proposal::completed(bank, &client, id, *command_id); }
            }
            Command::Poll { ticket } => client.reply(id, client.poll(*ticket)),
            Command::Info { request } => {
                let end = request
                    .map(|r| bank.end(r))
                    .transpose()
                    .map_err(native_error);
                let draft = request.map(|r| bank.draft_end(r)).transpose().map_err(native_error);
                client.reply(id, end.and_then(|end| draft.map(|draft| {
                    json!({"state":"info","bank":bank.info(),"request":request,
                        "committed_end":end,"draft_committed_end":draft.flatten()})
                })));

            }
            Command::CanPrepare { work } => {
                if client
                    .state
                    .lock()
                    .expect("client state")
                    .slots
                    .iter()
                    .any(|slot| {
                        active(slot) && work.iter().any(|w| slot.members.contains(&w.request))
                    })
                {
                    client.reply(
                        id,
                        Err(fail(BUSY, "capacity query request has active target work")),
                    );
                    continue;
                }
                let pairs = work
                    .iter()
                    .map(|w| (w.request, w.tokens))
                    .collect::<Vec<_>>();
                client.reply(id, bank.can_prepare(&pairs).map_err(native_error)
                    .map(|can_prepare|json!({"state":"capacity","can_prepare":can_prepare,"bank":bank.info()})));
            }
            Command::Admit { slot, request_id } => {
                let result=bank.admit(*slot,*request_id).map_err(native_error)
                    .map(|request|json!({"state":"admitted","request":request,"committed_end":0,"bank":bank.info()}));
                client.reply(id, result);
            }
            Command::Release { request } => {
                let busy = client
                    .state
                    .lock()
                    .expect("client state")
                    .slots
                    .iter()
                    .any(|s| active(s) && s.members.contains(request));
                client.reply(
                    id,
                    if busy {
                        Err(fail(BUSY, "request has an active target or result lease"))
                    } else {
                        bank.release(*request).map_err(native_error).map(
                            |_| json!({"state":"released","request":request,"bank":bank.info()}),
                        )
                    },
                );
            }
            Command::Shutdown => {
                if client
                    .state
                    .lock()
                    .expect("client state")
                    .slots
                    .iter()
                    .any(active)
                {
                    client.state.lock().expect("client state").closing = false;
                    client.reply(
                        id,
                        Err(fail(
                            BUSY,
                            "cancel work and drain all result leases before shutdown",
                        )),
                    );
                } else {
                    for lane in &lanes {
                        let _ = lane.send(LaneMessage::Stop).await;
                    }
                    return Ok(Exit::Shutdown(id)); // final ACK is sent only after with_target drops native owners
                }
            }
            _ => match reserve(&client, &message) {
                Err(error) => client.reply(id, Err(error)),
                Ok(index) => {
                    if lanes[index]
                        .send(LaneMessage::Command(message))
                        .await
                        .is_err()
                    {
                        clear_pending(&client, index);
                        client.reply(id, Err(fail(FAILED, "native lane stopped")));
                    }
                }
            },
        }
    }
    anyhow::bail!("native command channel closed without shutdown")
}

pub(super) enum Finish {
    Commit(Message, u32),
    CommitBatch(Message, Vec<u32>),
    Cancel(Message),
}

/// The descriptor escapes only while this function retains the actual Rust
/// Logits borrow. No mutable context operation is possible inside this scope.
pub(super) enum Acceptance {
    Single(std::ops::RangeInclusive<usize>),
    Batch(Vec<usize>),
}
pub(super) async fn result_scope<F: Fence>(
    output: Logits<'_>,
    ticket: Ticket,
    acceptance: Acceptance,
    fence: &mut F,
    client: &Client,
    receive: &mut mpsc::Receiver<LaneMessage>,
) -> Finish {
    let mut lease = None;
    let mut consumed = false;
    client.update(ticket, "ready", None);
    loop {
        let Some(LaneMessage::Command(message)) = receive.recv().await else {
            // Stop cannot be queued for a live result. Preserve the borrow and
            // wait if a caller violates that internal shutdown invariant.
            std::future::pending::<()>().await;
            unreachable!();
        };
        let id = message.id;
        match &message.command {
            Command::AcquireLogits { .. } if lease.is_none() && !consumed => {
                let descriptor = if let Some(buffer) = output.device {
                    json!({"device_pointer":format!("0x{:x}",buffer.ptr as usize),
                        "device_id":buffer.device_id,"bytes":buffer.bytes})
                } else {
                    let host = output.host.expect("test logits storage");
                    json!({"device_pointer":format!("0x{:x}",host.as_ptr() as usize),
                        "device_id":-1,"bytes":std::mem::size_of_val(host)})
                };
                lease = Some(id);
                client.update(ticket, "leased", None);
                let mut reply = json!({"state":"leased","ticket":ticket,"lease":id,"rows":output.rows,
                    "vocabulary":129280,"dtype":"float32","selected":output.selected,"positions":output.positions});
                for (key, value) in descriptor.as_object().unwrap() {
                    reply[key] = value.clone();
                }
                client.reply(id, Ok(reply));
            }
            Command::ReleaseLogits {
                lease: expected,
                consumer_stream,
                ..
            } if lease == Some(*expected) => {
                let stream = consumer_stream
                    .strip_prefix("0x")
                    .filter(|s| !s.is_empty())
                    .and_then(|s| usize::from_str_radix(s, 16).ok());
                let record = stream
                    .ok_or_else(|| {
                        fail(INVALID, "consumer_stream must be a hexadecimal CUDA handle")
                    })
                    .and_then(|stream| fence.record(stream).map_err(native_error));
                if let Err(error) = record {
                    client.update(ticket, "leased", None);
                    client.reply(id, Err(error));
                    continue;
                }
                client.update(ticket, "consumer_draining", None);
                // No other mutable operation is admitted for this lane while
                // draining. The other lane and controller continue to be polled.
                let complete = loop {
                    match fence.ready() {
                        Ok(true) => break Ok(()),
                        Err(error) => break Err(native_error(error)),
                        Ok(false) => tokio::task::yield_now().await,
                    }
                };
                if let Err(error) = complete {
                    client.update(ticket, "leased", None);
                    client.reply(id, Err(error));
                    continue;
                }
                consumed = true;
                lease = None;
                client.update(ticket, "consumed", None);
                client.reply(
                    id,
                    Ok(json!({"state":"consumed","ticket":ticket,"lease":expected})),
                );
            }
            Command::Commit { accepted, .. } if lease.is_none() => {
                if !matches!(&acceptance, Acceptance::Single(range) if range.contains(&(*accepted as usize))) {
                    clear_pending(client, ticket.lane);
                    client.reply(
                        id,
                        Err(fail(INVALID, "accepted rows outside work contract")),
                    );
                    continue;
                }
                let accepted = *accepted;
                return Finish::Commit(message, accepted);
            }
            Command::CommitBatch { accepted, .. } if lease.is_none() => {
                if !matches!(&acceptance, Acceptance::Batch(rows) if rows.len() == accepted.len()
                    && rows.iter().zip(accepted).all(|(&rows, &n)| n as usize <= rows)) {
                    clear_pending(client, ticket.lane);
                    client.reply(id, Err(fail(INVALID, "accepted vector outside batch contract"))); continue;
                }
                let accepted = accepted.clone(); return Finish::CommitBatch(message, accepted);
            }
            Command::Cancel { .. } if lease.is_none() => return Finish::Cancel(message),
            _ => {
                clear_pending(client, ticket.lane);
                client.reply(id, Err(fail(STALE, "wrong result lease or operation")));
            }
        }
    }
}

async fn lane<B: Bank, D: SpeculativeDriver, F: Fence>(
    bank: &B,
    context: &mut TargetContext<D>,
    fence: &mut F,
    client: Arc<Client>,
    mut receive: mpsc::Receiver<LaneMessage>,
) {
    while let Some(LaneMessage::Command(message)) = receive.recv().await {
        let id = message.id;
        let prepared = match message.command {
            Command::SubmitBatch { placement, members, .. } => batch::prepare(bank, context, placement, members)
                .map(|ticket| (ticket, None)),
            Command::SubmitSpeculativeBatch { placement, members, .. } => {
                let inputs = members.into_iter().map(|m| SpeculativeInput { request: m.request,
                    expected_committed_end: m.expected_committed_end, anchor: m.anchor,
                    remaining_output_tokens: m.remaining_output_tokens, placement }).collect();
                match proposal::prepare_batch(bank, context, &client, &mut receive, id, inputs).await {
                    Some(result) => result.map(|p| (p.ticket, Some((vec![], p.draft_us)))),
                    None => continue,
                }
            }
            Command::Submit { request, expected_committed_end, work, .. } =>
                prepare_full(bank, context, request, expected_committed_end, work)
                    .map(|ticket| (ticket, None)),
            Command::SubmitSpeculative { request, expected_committed_end, anchor,
                remaining_output_tokens, placement, .. } => {
                match proposal::prepare(bank, context, &client, &mut receive, id,
                    SpeculativeInput { request, expected_committed_end, anchor,
                        remaining_output_tokens, placement }).await {
                    Some(result) => result.map(|p| (p.ticket, Some((p.tokens, p.draft_us)))),
                    None => continue,
                }
            }
            Command::CancelProposal { command_id } => {
                proposal::completed(bank, &client, id, command_id); continue;
            }
            _ => {
                client.reply(id, Err(fail(STALE, "target lane has no live batch"))); continue;
            }
        };
        let (ticket, proposed) = match prepared {
            Ok(value) => value,
            Err(error) => {
                client.state.lock().expect("client state").slots[context.lane] = Slot::default();
                client.reply(id, Err(error)); continue;
            }
        };
        client.update(ticket, "prepared", None);
        let mut reply = json!({"state":"prepared","ticket":ticket});
        if context.job.as_ref().unwrap().grouped {
            match batch::prepared(bank, context, ticket, proposed.as_ref().map_or(0, |(_, us)| *us)) {
                Ok(prepared) => reply = prepared,
                Err(error) => {
                    let _ = context.cancel(ticket); client.update(ticket, "cancelled", None);
                    client.reply(id, Err(error)); continue;
                }
            }
        } else if let Some((tokens, draft_us)) = proposed {
            match proposal::frontiers(bank, ticket.request) {
                Ok((end, draft)) => {
                    reply["tokens"] = json!(tokens); reply["draft_us"] = json!(draft_us);
                    reply["bank"] = bank.info(); reply["committed_end"] = json!(end);
                    reply["draft_committed_end"] = json!(draft);
                }
                Err(error) => {
                    let _ = context.cancel(ticket); client.update(ticket, "cancelled", None);
                    client.reply(id, Err(error)); continue;
                }
            }
        }
        client.reply(id, Ok(reply));
        // A cancellation can have been routed while the scoped proposal became
        // ready. Return its ticket without consuming the original prepared ACK.
        let start = loop {
            match receive.recv().await {
                Some(LaneMessage::Command(message)) => {
                    if let Command::CancelProposal { command_id } = message.command {
                        proposal::completed(bank, &client, message.id, command_id);
                    } else { break Some(LaneMessage::Command(message)); }
                }
                other => break other,
            }
        };
        let Some(LaneMessage::Command(start)) = start else {
            let _ = context.cancel(ticket);
            return;
        };
        if matches!(start.command, Command::Cancel { .. }) {
            finish(bank, context, &client, ticket, Finish::Cancel(start)).await;
            continue;
        }
        if !matches!(start.command, Command::Execute { .. }) {
            clear_pending(&client, ticket.lane);
            client.reply(
                start.id,
                Err(fail(INVALID, "prepared target requires execute or cancel")),
            );
            let _ = context.cancel(ticket);
            client.update(ticket, "cancelled", None);
            continue;
        }
        client.update(ticket, "executing", None);
        client.reply(start.id, Ok(json!({"state":"executing","ticket":ticket})));
        let outcome = {
            let future = context.execute(ticket);
            tokio::pin!(future);
            tokio::select! {
                result=&mut future=>Ok(result),
                message=receive.recv()=>Err(message),
            }
        }; // dropped execution future drains retained cancellation guards
        let executed = match outcome {
            Err(Some(LaneMessage::Command(cancel))) => {
                finish(bank, context, &client, ticket, Finish::Cancel(cancel)).await;
                continue;
            }
            Err(_) => {
                let _ = context.cancel(ticket);
                return;
            }
            Ok(result) => result,
        };
        if let Err(error) = executed {
            client.update(ticket, "failed", Some(format!("{error:#}")));
            if let Some(LaneMessage::Command(cancel)) = receive.recv().await {
                finish(bank, context, &client, ticket, Finish::Cancel(cancel)).await;
            } else {
                let _ = context.cancel(ticket);
                return;
            }
            continue;
        }
        let job = context.job.as_ref().expect("ready job");
        let acceptance = if job.grouped { Acceptance::Batch(job.inputs.iter().map(|i| i.tokens.len()).collect()) }
            else { Acceptance::Single(0..=job.inputs[0].tokens.len()) };
        let output = match context.logits(ticket) {
            Ok(output) => output,
            Err(error) => {
                let reason = format!("{error:#}");
                let _ = context.cancel(ticket);
                client.update(ticket, "cancelled", Some(reason));
                continue;
            }
        };
        let action = result_scope(
            output,
            ticket,
            acceptance,
            fence,
            &client,
            &mut receive,
        )
        .await;
        finish(bank, context, &client, ticket, action).await;
    }
}

async fn finish<B: Bank, D: TargetDriver>(
    bank: &B,
    context: &mut TargetContext<D>,
    client: &Client,
    ticket: Ticket,
    action: Finish,
) {
    let members = context.job.as_ref().map(|job| job.inputs.iter().map(|i| i.request).collect::<Vec<_>>()).unwrap_or_default();
    let grouped = context.job.as_ref().is_some_and(|job| job.grouped);
    match action {
        Finish::CommitBatch(message, accepted) => match context.commit_batch(ticket, &accepted).await {
            Ok(_) => {
                client.update(ticket, "committed", None);
                client.reply(message.id, batch::frontiers(bank, &members, false).map(|members|
                    json!({"state":"committed","ticket":ticket,"members":members,"bank":bank.info()})));
            }
            Err(error) => {
                let reason = format!("{error:#}");
                let _ = context.cancel(ticket);
                client.update(ticket, "cancelled", Some(reason.clone()));
                client.reply(message.id, Err(fail(FAILED, reason)));
            }
        },
        Finish::Commit(message, accepted) => match context.commit(ticket, accepted) {
            Ok(end) => {
                client.update(ticket, "committed", None);
                let result = bank.draft_end(ticket.request).map_err(native_error).map(|draft|
                    json!({"state":"committed","ticket":ticket,"committed_end":end,
                        "draft_committed_end":draft,"bank":bank.info()}));
                client.reply(message.id, result);
            }
            Err(error) => {
                let reason = format!("{error:#}");
                let _ = context.cancel(ticket);
                client.update(ticket, "cancelled", Some(reason.clone()));
                client.reply(message.id, Err(fail(FAILED, reason)));
            }
        },
        Finish::Cancel(message) => match context.cancel(ticket) {
            Ok(()) => {
                client.update(ticket, "cancelled", None);
                client.reply(
                    message.id,
                    if grouped {
                        batch::frontiers(bank, &members, true).map(|members|
                            json!({"state":"cancelled","ticket":ticket,"members":members,"bank":bank.info()}))
                    } else { proposal::frontiers(bank, ticket.request).map(|(end, draft)|
                        json!({"state":"cancelled","ticket":ticket,"committed_end":end,
                            "draft_committed_end":draft,"bank":bank.info()})) },
                );
            }
            Err(error) => {
                let reason = format!("{error:#}");
                client.update(ticket, "cancelled", Some(reason.clone()));
                client.reply(message.id, Err(fail(FAILED, reason)));
            }
        },
    }
}

fn prepare_full<B: Bank, D: TargetDriver>(bank: &B, context: &mut TargetContext<D>,
    request: RequestHandle, expected_committed_end: u64, work: Work) -> Api<Ticket> {
        let Work::FullTarget {
            tokens,
            selected,
            kind,
            placement,
        } = work
        else {
            unreachable!("stream work switches out of lane actors");
        };
        (|| -> Api<Ticket> {
            let end = bank.end(request).map_err(native_error)?;
            if end != expected_committed_end {
                return Err(fail(STALE, "native committed frontier differs from grant"));
            }
            context
                .submit(TargetInput {
                    request,
                    tokens,
                    selected,
                    placement,
                    kind: match kind {
                        Kind::Prefill => SourceKind::Prefill,
                        Kind::Decode => SourceKind::Decode,
                    },
                })
                .map_err(native_error)
        })()

}
