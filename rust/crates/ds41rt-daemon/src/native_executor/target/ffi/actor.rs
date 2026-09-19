use super::*;

pub(super) trait Bank {
    fn info(&self) -> Value;
    fn admit(&self, slot: usize, id: u64) -> Result<RequestHandle>;
    fn end(&self, request: RequestHandle) -> Result<u64>;
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
    if let Command::Submit { request, .. } = &message.command {
        if active(slot)
            || shared
                .slots
                .iter()
                .any(|s| active(s) && s.request == Some(*request))
        {
            return Err(fail(BUSY, "lane or request is already owned"));
        }
        shared.slots[lane] = Slot {
            request: Some(*request),
            pending: true,
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
            Command::Commit { .. } => matches!(slot.phase, "ready" | "consumed"),
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

pub(super) async fn run<B: Bank, D: TargetDriver, F: Fence>(
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
            Command::Poll { ticket } => client.reply(id, client.poll(*ticket)),
            Command::Info { request } => {
                let end = request
                    .map(|r| bank.end(r))
                    .transpose()
                    .map_err(native_error);
                client.reply(
                    id,
                    end.map(|end| {
                        json!({"state":"info","bank":bank.info(),
                    "request":request,"committed_end":end})
                    }),
                );
            }
            Command::CanPrepare { work } => {
                if client
                    .state
                    .lock()
                    .expect("client state")
                    .slots
                    .iter()
                    .any(|slot| {
                        active(slot) && work.iter().any(|w| Some(w.request) == slot.request)
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
                    .any(|s| active(s) && s.request == Some(*request));
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
    Cancel(Message),
}

/// The descriptor escapes only while this function retains the actual Rust
/// Logits borrow. No mutable context operation is possible inside this scope.
pub(super) async fn result_scope<F: Fence>(
    output: Logits<'_>,
    ticket: Ticket,
    accepted_range: std::ops::RangeInclusive<usize>,
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
                if !accepted_range.contains(&(*accepted as usize)) {
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
            Command::Cancel { .. } if lease.is_none() => return Finish::Cancel(message),
            _ => {
                clear_pending(client, ticket.lane);
                client.reply(id, Err(fail(STALE, "wrong result lease or operation")));
            }
        }
    }
}

async fn lane<B: Bank, D: TargetDriver, F: Fence>(
    bank: &B,
    context: &mut TargetContext<D>,
    fence: &mut F,
    client: Arc<Client>,
    mut receive: mpsc::Receiver<LaneMessage>,
) {
    while let Some(LaneMessage::Command(message)) = receive.recv().await {
        let id = message.id;
        let Command::Submit {
            request,
            expected_committed_end,
            work,
            ..
        } = message.command
        else {
            if let Some(ticket) = message.command.ticket() {
                clear_pending(&client, ticket.lane);
            }
            client.reply(id, Err(fail(STALE, "target lane has no live batch")));
            continue;
        };
        let Work::FullTarget {
            tokens,
            selected,
            kind,
            placement,
        } = work
        else {
            unreachable!("stream work switches out of lane actors");
        };
        let prepared = (|| -> Api<Ticket> {
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
        })();
        let ticket = match prepared {
            Ok(ticket) => ticket,
            Err(error) => {
                client.state.lock().expect("client state").slots[context.lane] = Slot::default();
                client.reply(id, Err(error));
                continue;
            }
        };
        client.update(ticket, "prepared", None);
        client.reply(id, Ok(json!({"state":"prepared","ticket":ticket})));
        let Some(LaneMessage::Command(start)) = receive.recv().await else {
            let _ = context.cancel(ticket);
            return;
        };
        if matches!(start.command, Command::Cancel { .. }) {
            finish(bank, context, &client, ticket, Finish::Cancel(start));
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
                finish(bank, context, &client, ticket, Finish::Cancel(cancel));
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
                finish(bank, context, &client, ticket, Finish::Cancel(cancel));
            } else {
                let _ = context.cancel(ticket);
                return;
            }
            continue;
        }
        let proposed_rows = context.job.as_ref().expect("ready job").input.tokens.len();
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
            0..=proposed_rows,
            fence,
            &client,
            &mut receive,
        )
        .await;
        finish(bank, context, &client, ticket, action);
    }
}

fn finish<B: Bank, D: TargetDriver>(
    bank: &B,
    context: &mut TargetContext<D>,
    client: &Client,
    ticket: Ticket,
    action: Finish,
) {
    match action {
        Finish::Commit(message, accepted) => match context.commit(ticket, accepted) {
            Ok(end) => {
                client.update(ticket, "committed", None);
                client.reply(message.id,
                Ok(json!({"state":"committed","ticket":ticket,"committed_end":end,"bank":bank.info()})));
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
                    Ok(json!({"state":"cancelled","ticket":ticket,"bank":bank.info()})),
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
