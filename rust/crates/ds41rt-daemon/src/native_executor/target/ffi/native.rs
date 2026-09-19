use super::*;
use ds41rt_ffi::NativeLibrary;
use std::ffi::c_void;

impl actor::Bank for NativeBank<'_, '_> {
    fn info(&self) -> Value {
        serde_json::to_value(NativeBank::info(self)).expect("native cache info serializes")
    }
    fn admit(&self, slot: usize, id: u64) -> Result<RequestHandle> {
        NativeBank::admit(self, slot, id)
    }
    fn end(&self, request: RequestHandle) -> Result<u64> {
        self.committed_end(request)
    }
    fn release(&self, request: RequestHandle) -> Result<()> {
        NativeBank::release(self, request)
    }
    fn can_prepare(&self, work: &[(RequestHandle, u32)]) -> Result<bool> {
        NativeBank::can_prepare(self, work)
    }
}

/// Reused native-owned event + private wait stream. Recording on the external
/// consumer stream orders its prior reads; querying our wait stream avoids
/// polling unrelated work submitted later on the consumer's stream.
struct ConsumerFence<'a> {
    lib: &'a NativeLibrary,
    stream: *mut c_void,
    event: *mut c_void,
    recorded: Cell<Option<usize>>,
}
impl<'a> ConsumerFence<'a> {
    fn new(lib: &'a NativeLibrary) -> Result<Self> {
        let stream = lib.cuda_stream_create()?;
        let event = match lib.cuda_event_create() {
            Ok(event) => event,
            Err(error) => {
                let _ = unsafe { lib.cuda_stream_destroy(stream) };
                return Err(error);
            }
        };
        Ok(Self {
            lib,
            stream,
            event,
            recorded: Cell::new(None),
        })
    }
}
impl actor::Fence for ConsumerFence<'_> {
    fn record(&mut self, consumer: usize) -> Result<()> {
        if let Some(previous) = self.recorded.get() {
            ensure!(
                previous == consumer,
                "failed consumer fence must resume the same stream"
            );
        } else {
            unsafe {
                self.lib
                    .cuda_event_record(self.event, consumer as *mut c_void)?;
            }
            self.recorded.set(Some(consumer));
        }
        unsafe { self.lib.cuda_stream_wait_event(self.stream, self.event) }
    }
    fn ready(&self) -> Result<bool> {
        let ready = unsafe { self.lib.cuda_stream_query(self.stream) }?;
        if ready {
            self.recorded.set(None);
        }
        Ok(ready)
    }
}
impl Drop for ConsumerFence<'_> {
    fn drop(&mut self) {
        if self.recorded.get().is_some() {
            if let Err(error) = unsafe { self.lib.cuda_event_synchronize(self.event) } {
                tracing::error!(%error,"draining native target consumer event");
            }
        }
        let _ = unsafe { self.lib.cuda_stream_destroy(self.stream) };
        let _ = unsafe { self.lib.cuda_event_destroy(self.event) };
    }
}

pub(super) fn run(
    config: Config,
    client: Arc<Client>,
    receive: mpsc::Receiver<Message>,
) -> Result<u64> {
    with_target(
        TargetConfig {
            owner: config.owner,
            snapshot: config.snapshot,
            native_lib: config.native_lib,
            peers: config.peers,
            batch_tokens: config.batch_tokens,
            max_context_tokens: config.max_context_tokens,
            slots: config.slots,
            cache_bytes: config.source_pool_budget_bytes,
        },
        |mut target| {
            let runtime = target.runtime();
            let mut fences = [
                ConsumerFence::new(target.bank().library())?,
                ConsumerFence::new(target.bank().library())?,
            ];
            client.state.lock().expect("client state").initialized = true;
            client.reply(
                0,
                Ok(json!({"state":"initialized","bank":actor::Bank::info(target.bank())})),
            );
            let mut receive = receive;
            loop {
                let next = {
                    let (_, bank, contexts) = target.split();
                    runtime.block_on(actor::run(
                        bank,
                        contexts,
                        &mut fences,
                        client.clone(),
                        &mut receive,
                    ))?
                };
                match next {
                    actor::Exit::Shutdown(id) => return Ok(id),
                    actor::Exit::Stream(message) => runtime.block_on(stream::run(
                        &mut target,
                        message,
                        &mut fences[0],
                        client.clone(),
                        &mut receive,
                    ))?,
                }
            }
        },
    )
}

impl stream::Output for StreamingResult<'_, '_, '_, '_> {
    fn logits(&self) -> Result<Logits<'_>> {
        StreamingResult::logits(self)
    }
    fn commit(self) -> Result<u64> {
        StreamingResult::commit(self)
    }
    fn cancel(self) -> Result<()> {
        StreamingResult::cancel(self)
    }
}
impl<'s, 'w, 'a> stream::Backend for NativeTarget<'s, 'w, 'a> {
    type Ready<'r>
        = StreamingResult<'r, 's, 'w, 'a>
    where
        Self: 'r;
    fn identity(&self, input: &StreamInput) -> Result<Ticket> {
        self.stream_identity(input)
    }
    async fn execute<'r>(&'r mut self, input: StreamInput) -> Result<Self::Ready<'r>> {
        self.stream_prefill(input, &|| true).await
    }
    fn revoke(&self, request: RequestHandle) -> Result<()> {
        self.revoke_stream_admission(request)
    }
    fn info(&self) -> Value {
        actor::Bank::info(self.bank())
    }
}
