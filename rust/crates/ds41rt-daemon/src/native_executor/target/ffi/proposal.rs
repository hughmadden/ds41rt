//! Scoped draft futures own readers until completion or command-ID cancellation.
use super::*;
use actor::{Bank, LaneMessage};

pub(super) fn frontiers<B: Bank>(bank: &B, request: RequestHandle) -> Api<(u64, Option<u64>)> {
    let end = bank.end(request).map_err(native_error)?;
    let draft = bank.draft_end(request).map_err(native_error)?;
    if draft.is_some_and(|draft| draft != end) {
        return Err(fail(FAILED, "native target and draft frontiers differ"));
    }
    Ok((end, draft))
}

pub(super) fn completed<B: Bank>(bank: &B, client: &Client, id: u64, original: u64) {
    let found = {
        let state = client.state.lock().expect("client state");
        state
            .slots
            .iter()
            .find(|slot| slot.proposal_command == Some(original))
            .map(|slot| (slot.ticket, slot.request, slot.phase, slot.members.clone(), slot.grouped))
    };
    let result = match found {
        Some((ticket, _, phase, members, true)) if ticket.is_some() || phase == "proposal_cancelled" => {
            batch::frontiers(bank, &members, false).map(|members| json!({
                "state":if ticket.is_some() { "proposal_completed" } else { "proposal_cancelled" },
                "proposal_command":original,"ticket":ticket,"members":members,"bank":bank.info()}))
        }
        Some((Some(ticket), _, _, _, false)) => frontiers(bank, ticket.request).map(|(end, draft)| {
            json!({"state":"proposal_completed","proposal_command":original,"ticket":ticket,
                "committed_end":end,"draft_committed_end":draft,"bank":bank.info()})
        }),
        Some((None, Some(request), "proposal_cancelled", _, false)) => {
            frontiers(bank, request).map(|(end, draft)| {
                json!({"state":"proposal_cancelled","proposal_command":original,"request":request,
                "committed_end":end,"draft_committed_end":draft,"bank":bank.info()})
            })
        }
        _ => Err(fail(STALE, "unknown or replaced proposal command")),
    };
    client.reply(id, result);
}

/// None means the original and cancellation replies were both handled. The
/// pending future cannot escape this computing-thread scope or its reader guards.
pub(super) async fn prepare<B: Bank, D: SpeculativeDriver>(
    bank: &B,
    context: &mut TargetContext<D>,
    client: &Client,
    receive: &mut mpsc::Receiver<LaneMessage>,
    id: u64,
    input: SpeculativeInput,
) -> Option<Api<SpeculativeProposal>> {
    match bank.end(input.request) {
        Ok(end) if end == input.expected_committed_end => (),
        Ok(_) => {
            return Some(Err(fail(
                STALE,
                "native committed frontier differs from proposal",
            )))
        }
        Err(error) => return Some(Err(native_error(error))),
    }
    let active = context.active.clone();
    let lane = context.lane;
    wait(bank, &active, lane, client, receive, id, vec![input.request], false,
        context.submit_speculative(input)).await
}

pub(super) async fn prepare_batch<B: Bank, D: SpeculativeDriver>(
    bank: &B, context: &mut TargetContext<D>, client: &Client,
    receive: &mut mpsc::Receiver<LaneMessage>, id: u64, inputs: Vec<SpeculativeInput>,
) -> Option<Api<BatchProposal>> {
    for input in &inputs {
        match bank.end(input.request) {
            Ok(end) if end == input.expected_committed_end => (),
            Ok(_) => return Some(Err(fail(STALE, "native batch frontier differs from proposal"))),
            Err(error) => return Some(Err(native_error(error))),
        }
    }
    let active = context.active.clone();
    let lane = context.lane;
    let requests = inputs.iter().map(|i| i.request).collect();
    wait(bank, &active, lane, client, receive, id, requests, true,
        context.submit_speculative_batch(inputs)).await
}

async fn wait<B: Bank, T>(bank: &B, active: &Active, lane: usize, client: &Client,
    receive: &mut mpsc::Receiver<LaneMessage>, id: u64, members: Vec<RequestHandle>, grouped: bool,
    future: impl std::future::Future<Output = Result<T>>,
) -> Option<Api<T>> {
    let result = {
        tokio::pin!(future);
        tokio::select! {
            result = &mut future => Ok(result),
            cancel = receive.recv() => Err(cancel),
        }
    }; // cancellation drops/drains the actual proposal before either ACK
    match result {
        Ok(result) => Some(result.map_err(native_error)),
        Err(Some(LaneMessage::Command(cancel))) => {
            let valid = matches!(cancel.command, Command::CancelProposal { command_id } if command_id == id);
            let drained = active.healthy().map_err(native_error);
            if !valid || drained.is_err() {
                let error = drained.err().unwrap_or_else(|| fail(INTERNAL, "unexpected proposal interruption"));
                client.reply(id, Err(error.clone())); client.reply(cancel.id, Err(error)); return None;
            }
            client.state.lock().expect("client state").slots[lane] = Slot {
                request: members.first().copied(), members, grouped,
                phase: "proposal_cancelled", proposal_command: Some(id), ..Default::default()
            };
            client.reply(id, Err(fail(FAILED, "native proposal cancelled")));
            completed(bank, client, cancel.id, id); None
        }
        Err(_) => { client.reply(id, Err(fail(FAILED, "native proposal channel closed"))); None }
    }
}
