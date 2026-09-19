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
            .map(|slot| (slot.ticket, slot.request, slot.phase))
    };
    let result = match found {
        Some((Some(ticket), _, _)) => frontiers(bank, ticket.request).map(|(end, draft)| {
            json!({"state":"proposal_completed","proposal_command":original,"ticket":ticket,
                "committed_end":end,"draft_committed_end":draft,"bank":bank.info()})
        }),
        Some((None, Some(request), "proposal_cancelled")) => {
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
    let result = {
        let proposal = context.submit_speculative(input);
        tokio::pin!(proposal);
        tokio::select! {
            result = &mut proposal => Ok(result),
            cancel = receive.recv() => Err(cancel),
        }
    }; // cancellation drops/drains the actual proposal before either ACK
    match result {
        Ok(result) => Some(result.map_err(native_error)),
        Err(Some(LaneMessage::Command(cancel))) => {
            let valid = matches!(cancel.command, Command::CancelProposal { command_id } if command_id == id);
            let drained = context.active.healthy().map_err(native_error);
            if !valid || drained.is_err() {
                // Fail closed; a failed native drain retains readers and poisons
                // the shared native owner, so do not advertise cancellation.
                let error = drained
                    .err()
                    .unwrap_or_else(|| fail(INTERNAL, "unexpected proposal interruption"));
                client.reply(id, Err(error.clone()));
                client.reply(cancel.id, Err(error));
                return None;
            }
            client.state.lock().expect("client state").slots[context.lane] = Slot {
                request: Some(input.request),
                phase: "proposal_cancelled",
                proposal_command: Some(id),
                ..Default::default()
            };
            client.reply(id, Err(fail(FAILED, "native proposal cancelled")));
            completed(bank, client, cancel.id, id);
            None
        }
        Err(_) => {
            client.reply(id, Err(fail(FAILED, "native proposal channel closed")));
            None
        }
    }
}
