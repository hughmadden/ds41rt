//! Thread-safe C boundary. Native resources never leave their scoped owner.
use super::*;
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    panic::{catch_unwind, AssertUnwindSafe},
    ptr, slice,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex, OnceLock,
    },
    thread::JoinHandle,
};
use tokio::sync::mpsc;
mod actor;
mod native;
mod stream;

const OK: i32 = 0;
const INVALID: i32 = 1;
const BUSY: i32 = 2;
const STALE: i32 = 3;
const NOT_READY: i32 = 4;
const BUFFER_TOO_SMALL: i32 = 5;
const FAILED: i32 = 6;
const INTERNAL: i32 = 8;
const LIMIT: usize = 32;
const MAX_JSON: usize = 65536;
const MAX_COMMAND: usize = 16 * 1024 * 1024;
type Error = (i32, String);
type Api<T> = std::result::Result<T, Error>;
fn fail(code: i32, message: impl Into<String>) -> Error {
    (code, message.into())
}
fn native_error(error: anyhow::Error) -> Error {
    fail(FAILED, format!("{error:#}"))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    owner: u64,
    snapshot: std::path::PathBuf,
    native_lib: std::path::PathBuf,
    peers: [std::net::SocketAddr; 4],
    batch_tokens: u32,
    max_context_tokens: u32,
    slots: u32,
    source_pool_budget_bytes: usize,
}
impl Config {
    fn validate(&self) -> Api<()> {
        if self.owner == 0
            || !(80..=4096).contains(&self.batch_tokens)
            || !(1..=1048576).contains(&self.max_context_tokens)
            || !(1..=16).contains(&self.slots)
            || self.source_pool_budget_bytes == 0
        {
            return Err(fail(INVALID, "invalid native target geometry"));
        }
        Ok(())
    }
}
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Kind {
    Prefill,
    Decode,
}
#[derive(Deserialize)]
#[serde(tag = "phase", rename_all = "snake_case", deny_unknown_fields)]
enum Work {
    FullTarget {
        tokens: Vec<u32>,
        selected: Vec<usize>,
        kind: Kind,
        placement: u64,
    },
    EncoderStream {
        tokens: Vec<u32>,
        chunk_rows: usize,
        selected: Vec<usize>,
    },
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Append {
    request: RequestHandle,
    tokens: u32,
}
#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
enum Command {
    Info {
        request: Option<RequestHandle>,
    },
    Admit {
        slot: usize,
        request_id: u64,
    },
    CanPrepare {
        work: Vec<Append>,
    },
    Submit {
        lane: usize,
        request: RequestHandle,
        expected_committed_end: u64,
        work: Work,
    },
    Execute {
        ticket: Ticket,
    },
    Poll {
        ticket: Ticket,
    },
    AcquireLogits {
        ticket: Ticket,
    },
    ReleaseLogits {
        ticket: Ticket,
        lease: u64,
        consumer_stream: String,
    },
    Commit {
        ticket: Ticket,
        accepted: u32,
    },
    Cancel {
        ticket: Ticket,
    },
    Release {
        request: RequestHandle,
    },
    Shutdown,
}
impl Command {
    fn lane(&self) -> Option<usize> {
        match self {
            Self::Submit { lane, .. } => Some(*lane),
            Self::Execute { ticket }
            | Self::Poll { ticket }
            | Self::AcquireLogits { ticket }
            | Self::ReleaseLogits { ticket, .. }
            | Self::Commit { ticket, .. }
            | Self::Cancel { ticket } => Some(ticket.lane),
            _ => None,
        }
    }
    fn ticket(&self) -> Option<Ticket> {
        match self {
            Self::Execute { ticket }
            | Self::Poll { ticket }
            | Self::AcquireLogits { ticket }
            | Self::ReleaseLogits { ticket, .. }
            | Self::Commit { ticket, .. }
            | Self::Cancel { ticket } => Some(*ticket),
            _ => None,
        }
    }
}
struct Message {
    id: u64,
    command: Command,
}
#[derive(Default)]
struct Slot {
    ticket: Option<Ticket>,
    request: Option<RequestHandle>,
    phase: &'static str,
    error: Option<String>,
    pending: bool,
}
struct Shared {
    replies: BTreeMap<u64, Option<Vec<u8>>>,
    next_id: u64,
    slots: [Slot; 2],
    initialized: bool,
    closing: bool,
    closed: bool,
}
struct Client {
    state: Mutex<Shared>,
    send: mpsc::Sender<Message>,
    thread: Mutex<Option<JoinHandle<()>>>,
}
impl Client {
    fn pair() -> (Arc<Self>, mpsc::Receiver<Message>) {
        let (send, receive) = mpsc::channel(LIMIT);
        (
            Arc::new(Self {
                state: Mutex::new(Shared {
                    replies: BTreeMap::from([(0, None)]),
                    next_id: 1,
                    slots: Default::default(),
                    initialized: false,
                    closing: false,
                    closed: false,
                }),
                send,
                thread: Mutex::new(None),
            }),
            receive,
        )
    }
    fn reply(&self, id: u64, result: Api<Value>) {
        let value = match result {
            Ok(result) => json!({"command_id":id,"ok":true,"result":result}),
            Err((code, error)) => json!({"command_id":id,"ok":false,"code":code,
                "error":error.chars().take(4096).collect::<String>()}),
        };
        if let Ok(mut shared) = self.state.lock() {
            if let Some(reply) = shared.replies.get_mut(&id) {
                *reply = Some(serde_json::to_vec(&value).expect("JSON value serializes"));
            }
        }
    }
    fn update(&self, ticket: Ticket, phase: &'static str, error: Option<String>) {
        let mut shared = self.state.lock().expect("native client lock");
        let error = error.map(|s| s.chars().take(4096).collect());
        shared.slots[ticket.lane] = Slot {
            ticket: Some(ticket),
            request: Some(ticket.request),
            phase,
            error,
            pending: false,
        };
    }
    fn poll(&self, ticket: Ticket) -> Api<Value> {
        let state = self
            .state
            .lock()
            .map_err(|_| fail(INTERNAL, "client state poisoned"))?;
        let slot = state
            .slots
            .get(ticket.lane)
            .ok_or_else(|| fail(INVALID, "invalid lane"))?;
        if slot.ticket != Some(ticket) {
            return Err(fail(STALE, "foreign or stale target ticket"));
        }
        Ok(json!({"state":slot.phase,"ticket":ticket,"error":slot.error}))
    }
    fn enqueue(&self, bytes: &[u8]) -> Api<u64> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| fail(INTERNAL, "client state poisoned"))?;
        if state.closed {
            return Err(fail(STALE, "target owner closed"));
        }
        if !state.initialized {
            return Err(fail(NOT_READY, "target initialization pending"));
        }
        if state.closing || state.replies.len() >= LIMIT {
            return Err(fail(BUSY, "target queue full or closing"));
        }
        let command: Command =
            serde_json::from_slice(bytes).map_err(|e| fail(INVALID, e.to_string()))?;
        if command.lane().is_some_and(|lane| lane >= 2) {
            return Err(fail(INVALID, "invalid target lane"));
        }
        if matches!(command, Command::Shutdown) {
            if !state.replies.is_empty() {
                return Err(fail(
                    BUSY,
                    "collect outstanding command replies before shutdown",
                ));
            }
            state.closing = true;
        }
        let id = state.next_id;
        state.next_id = id
            .checked_add(1)
            .ok_or_else(|| fail(INTERNAL, "command ID exhausted"))?;
        state.replies.insert(id, None);
        if self.send.try_send(Message { id, command }).is_err() {
            state.replies.remove(&id);
            state.closing = false;
            return Err(fail(BUSY, "native owner channel unavailable"));
        }
        Ok(id)
    }
    fn finish(&self, result: Result<u64>) {
        self.state.lock().expect("client lock").closed = true;
        match result {
            Ok(id) => self.reply(id, Ok(json!({"state":"closed"}))),
            Err(error) => {
                let ids = self
                    .state
                    .lock()
                    .expect("client lock")
                    .replies
                    .iter()
                    .filter_map(|(&id, r)| r.is_none().then_some(id))
                    .collect::<Vec<_>>();
                for id in ids {
                    self.reply(
                        id,
                        Err(fail(FAILED, format!("native owner stopped: {error:#}"))),
                    );
                }
            }
        }
    }
}
static NEXT_HANDLE: AtomicU64 = AtomicU64::new(1);
fn registry() -> &'static Mutex<BTreeMap<u64, Arc<Client>>> {
    static REGISTRY: OnceLock<Mutex<BTreeMap<u64, Arc<Client>>>> = OnceLock::new();
    REGISTRY.get_or_init(Default::default)
}
fn client(handle: u64) -> Api<Arc<Client>> {
    registry()
        .lock()
        .map_err(|_| fail(INTERNAL, "registry poisoned"))?
        .get(&handle)
        .cloned()
        .ok_or_else(|| fail(STALE, "unknown or destroyed target handle"))
}
fn register(client: Arc<Client>) -> Api<u64> {
    let handle = NEXT_HANDLE
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
        .map_err(|_| fail(INTERNAL, "target handles exhausted"))?;
    registry()
        .lock()
        .map_err(|_| fail(INTERNAL, "registry poisoned"))?
        .insert(handle, client);
    Ok(handle)
}
fn guarded(action: impl FnOnce() -> Api<()>) -> i32 {
    match catch_unwind(AssertUnwindSafe(action)) {
        Ok(Ok(())) => OK,
        Ok(Err((code, _))) => code,
        Err(_) => INTERNAL,
    }
}
unsafe fn bytes<'a>(p: *const u8, len: usize, limit: usize) -> Api<&'a [u8]> {
    if p.is_null() || len == 0 || len > limit {
        return Err(fail(INVALID, "invalid input buffer"));
    }
    Ok(unsafe { slice::from_raw_parts(p, len) })
}
#[no_mangle]
pub extern "C" fn ds41rt_target_abi_version() -> u32 {
    1
}
/// # Safety
/// All pointers obey include/ds41rt_target.h.
#[no_mangle]
pub unsafe extern "C" fn ds41rt_target_create(p: *const u8, len: usize, out: *mut u64) -> i32 {
    guarded(|| {
        if out.is_null() {
            return Err(fail(INVALID, "null handle output"));
        }
        unsafe {
            *out = 0;
        }
        let config: Config = serde_json::from_slice(unsafe { bytes(p, len, MAX_JSON)? })
            .map_err(|e| fail(INVALID, e.to_string()))?;
        config.validate()?;
        let (c, receive) = Client::pair();
        let worker = c.clone();
        let thread = std::thread::Builder::new()
            .name("ds41rt-target-owner".into())
            .spawn(move || {
                let result = native::run(config, worker.clone(), receive);
                worker.finish(result);
            })
            .map_err(|e| fail(FAILED, e.to_string()))?;
        *c.thread.lock().map_err(|_| fail(INTERNAL, "thread lock"))? = Some(thread);
        unsafe {
            *out = register(c)?;
        }
        Ok(())
    })
}
/// # Safety
/// All pointers and CUDA handles obey include/ds41rt_target.h.
#[no_mangle]
pub unsafe extern "C" fn ds41rt_target_command(
    handle: u64,
    p: *const u8,
    len: usize,
    out: *mut u64,
) -> i32 {
    guarded(|| {
        if out.is_null() {
            return Err(fail(INVALID, "null command output"));
        }
        unsafe {
            *out = 0;
        }
        let id = client(handle)?.enqueue(unsafe { bytes(p, len, MAX_COMMAND)? })?;
        unsafe {
            *out = id;
        }
        Ok(())
    })
}
/// # Safety
/// All pointers obey include/ds41rt_target.h.
#[no_mangle]
pub unsafe extern "C" fn ds41rt_target_result(
    handle: u64,
    id: u64,
    out: *mut u8,
    capacity: usize,
    written: *mut usize,
) -> i32 {
    guarded(|| {
        if written.is_null() {
            return Err(fail(INVALID, "null size output"));
        }
        unsafe {
            *written = 0;
        }
        let client = client(handle)?;
        let mut state = client
            .state
            .lock()
            .map_err(|_| fail(INTERNAL, "client state poisoned"))?;
        let result = state
            .replies
            .get(&id)
            .ok_or_else(|| fail(STALE, "unknown or consumed command"))?
            .as_ref()
            .ok_or_else(|| fail(NOT_READY, "native reply pending"))?;
        unsafe {
            *written = result.len();
        }
        if capacity < result.len() {
            return Err(fail(BUFFER_TOO_SMALL, "reply capacity insufficient"));
        }
        if out.is_null() {
            return Err(fail(INVALID, "null reply output"));
        }
        unsafe {
            ptr::copy_nonoverlapping(result.as_ptr(), out, result.len());
        }
        state.replies.remove(&id);
        Ok(())
    })
}
#[no_mangle]
pub extern "C" fn ds41rt_target_destroy(handle: u64) -> i32 {
    guarded(|| {
        let c = client(handle)?;
        if !c
            .state
            .lock()
            .map_err(|_| fail(INTERNAL, "client state poisoned"))?
            .closed
        {
            return Err(fail(BUSY, "native owner has not shut down"));
        }
        if registry()
            .lock()
            .map_err(|_| fail(INTERNAL, "registry poisoned"))?
            .remove(&handle)
            .is_none()
        {
            return Err(fail(STALE, "target already destroyed"));
        }
        if let Some(thread) = c
            .thread
            .lock()
            .map_err(|_| fail(INTERNAL, "thread lock"))?
            .take()
        {
            thread
                .join()
                .map_err(|_| fail(FAILED, "native owner panicked"))?;
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests;
