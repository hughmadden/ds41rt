//! Exclusive whole-target mode. Reuses normal ticket/result/fence machinery.
use super::*;
use actor::{Fence, Finish, LaneMessage};
use tokio::sync::oneshot;

pub(super) trait Output {
    fn logits(&self) -> Result<Logits<'_>>;
    fn commit(self) -> Result<u64>;
    fn cancel(self) -> Result<()>;
}
/// The real implementation returns StreamingResult borrowing the whole target.
/// CPU tests replace only this device-facing implementation, not the actor.
pub(super) trait Backend {
    type Ready<'a>: Output
    where
        Self: 'a;
    fn identity(&self, input: &StreamInput) -> Result<Ticket>;
    async fn execute<'a>(&'a mut self, input: StreamInput) -> Result<Self::Ready<'a>>;
    fn revoke(&self, request: RequestHandle) -> Result<()>;
    fn info(&self) -> Value;
    fn draft_end(&self, _request: RequestHandle) -> Result<Option<u64>> { Ok(None) }
}

pub(super) async fn run<B: Backend, F: Fence>(
    backend: &mut B,
    message: Message,
    fence: &mut F,
    client: Arc<Client>,
    receive: &mut mpsc::Receiver<Message>,
) -> Result<()> {
    let id = message.id;
    let Command::Submit {
        request,
        expected_committed_end,
        work:
            Work::EncoderStream {
                tokens,
                chunk_rows,
                selected,
            },
        ..
    } = message.command
    else {
        anyhow::bail!("expected encoder stream mode switch");
    };
    if expected_committed_end != 0 {
        client.reply(
            id,
            Err(fail(INVALID, "encoder stream requires a fresh frontier")),
        );
        return Ok(());
    }
    let input = StreamInput {
        request,
        tokens,
        chunk_rows,
        selected,
    };
    let ticket = match backend.identity(&input) {
        Ok(ticket) => ticket,
        Err(error) => {
            client.reply(id, Err(native_error(error)));
            return Ok(());
        }
    };
    client.update(ticket, "prepared", None);
    client.reply(id, Ok(json!({"state":"prepared","ticket":ticket})));
    let (commands, mut routed) = mpsc::channel(LIMIT);
    let (finished, done) = oneshot::channel();
    let (result, ()) = tokio::join!(
        async {
            let result = job(backend, input, ticket, fence, &client, &mut routed).await;
            let _ = finished.send(());
            result
        },
        control(&client, ticket, receive, commands, done)
    );
    result
}

async fn control(
    client: &Client,
    ticket: Ticket,
    receive: &mut mpsc::Receiver<Message>,
    commands: mpsc::Sender<LaneMessage>,
    mut done: oneshot::Receiver<()>,
) {
    loop {
        let message = tokio::select! { biased;
            _=&mut done=>return,
            message=receive.recv()=>match message {Some(message)=>message,None=>return},
        };
        let id = message.id;
        if let Command::Poll { ticket } = message.command {
            client.reply(id, client.poll(ticket));
        } else if message.command.ticket().is_some() {
            if message.command.ticket() != Some(ticket) {
                client.reply(
                    id,
                    Err(fail(STALE, "stream owns both contexts; unrelated ticket")),
                );
            } else {
                match actor::reserve(client, &message) {
                    Ok(_) => {
                        let _ = commands.send(LaneMessage::Command(message)).await;
                    }
                    Err(error) => client.reply(id, Err(error)),
                }
            }
        } else {
            if matches!(message.command, Command::Shutdown) {
                client.state.lock().expect("client state").closing = false;
            }
            // No bank access while retained encode/replay may hold its RefCell
            // borrow. Never substitute a cached snapshot for a real bank read.
            client.reply(
                id,
                Err(fail(
                    BUSY,
                    "encoder stream owns both contexts and the native bank",
                )),
            );
        }
    }
}

enum Outcome {
    Commit(Message, Result<u64>),
    Cancel(Message, Result<()>),
    Failed(String),
}
async fn job<B: Backend, F: Fence>(
    backend: &mut B,
    input: StreamInput,
    ticket: Ticket,
    fence: &mut F,
    client: &Client,
    receive: &mut mpsc::Receiver<LaneMessage>,
) -> Result<()> {
    let rows = input.tokens.len();
    let selected = input.selected.clone();
    let Some(LaneMessage::Command(start)) = receive.recv().await else {
        backend.revoke(ticket.request)?;
        anyhow::bail!("stream command channel stopped");
    };
    if matches!(start.command, Command::Cancel { .. }) {
        backend.revoke(ticket.request)?;
        cancelled(client, backend, ticket, start.id);
        return Ok(());
    }
    ensure!(
        matches!(start.command, Command::Execute { .. }),
        "stream requires execute or cancel"
    );
    client.update(ticket, "executing", None);
    client.reply(start.id, Ok(json!({"state":"executing","ticket":ticket})));
    let outcome = {
        let executed = {
            let future = backend.execute(input);
            tokio::pin!(future);
            tokio::select! {
                ready=&mut future=>Ok(ready),
                cancel=receive.recv()=>Err(cancel),
            }
        }; // cancelled future drains the retained streaming session
        match executed {
            Err(Some(LaneMessage::Command(cancel))) => Outcome::Cancel(cancel, Ok(())),
            Err(_) => Outcome::Failed("stream command channel stopped".into()),
            Ok(Err(error)) => Outcome::Failed(format!("{error:#}")),
            Ok(Ok(output)) => {
                let action = {
                    let mut logits = output.logits()?;
                    // Native replay metadata is relative to the final window;
                    // wire selection is the caller's absolute prompt selection.
                    logits.selected = &selected;
                    actor::result_scope(logits, ticket, actor::Acceptance::Single(rows..=rows), fence, client, receive).await
                };
                match action {
                    Finish::Commit(message, _) => Outcome::Commit(message, output.commit()),
                    Finish::Cancel(message) => Outcome::Cancel(message, output.cancel()),
                    Finish::CommitBatch(..) => unreachable!("stream result rejects grouped commits"),
                }
            }
        }
    }; // no Ready/logits/future borrow survives into bank operations below
    match outcome {
        Outcome::Commit(message, Ok(end)) => {
            let draft_end = backend.draft_end(ticket.request)?;
            ensure!(draft_end.is_none_or(|draft| draft == end), "streaming draft frontier differs");
            client.update(ticket, "committed", None);
            client.reply(
                message.id,
                Ok(json!({"state":"committed","ticket":ticket,
                "committed_end":end,"draft_committed_end":draft_end,"bank":backend.info()})),
            );
        }
        Outcome::Commit(message, Err(error)) => {
            backend.revoke(ticket.request)?;
            client.update(ticket, "failed", Some(format!("{error:#}")));
            client.reply(message.id, Err(native_error(error)));
            await_cancel(backend, client, ticket, receive).await?;
        }
        Outcome::Cancel(message, drained) => {
            drained?;
            backend.revoke(ticket.request)?;
            cancelled(client, backend, ticket, message.id);
        }
        Outcome::Failed(reason) => {
            backend.revoke(ticket.request)?;
            client.update(ticket, "failed", Some(reason));
            await_cancel(backend, client, ticket, receive).await?;
        }
    }
    Ok(())
}
async fn await_cancel<B: Backend>(
    backend: &B,
    client: &Client,
    ticket: Ticket,
    receive: &mut mpsc::Receiver<LaneMessage>,
) -> Result<()> {
    let Some(LaneMessage::Command(message)) = receive.recv().await else {
        anyhow::bail!("stream cancellation receipt channel stopped");
    };
    ensure!(
        matches!(message.command, Command::Cancel { .. }),
        "failed stream requires cancellation receipt"
    );
    cancelled(client, backend, ticket, message.id);
    Ok(())
}
fn cancelled<B: Backend>(client: &Client, backend: &B, ticket: Ticket, id: u64) {
    client.update(ticket, "cancelled", None);
    client.reply(
        id,
        Ok(json!({"state":"cancelled","ticket":ticket,"revoked":true,"bank":backend.info()})),
    );
}
