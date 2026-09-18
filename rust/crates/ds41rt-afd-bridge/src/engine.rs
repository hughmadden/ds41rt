use anyhow::{ensure, Context, Result};
use ds41rt_transport::{
    v41_expert::{V41BackboneRequest, V41Tp4Roce, V41Tp4Tcp, V41_PARTIAL_ROW_BYTES},
    ExpertProtocolV2Request, TcpTransportConfig,
};
use serde::Deserialize;
use std::{
    net::SocketAddr,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex,
    },
    thread::{self, JoinHandle},
    time::Duration,
};
use tokio::sync::{mpsc, Notify};

pub const IDLE: u32 = 0;
pub const PENDING: u32 = 1;
pub const READY: u32 = 2;
pub const FAILED: u32 = 3;
pub const CANCELLED: u32 = 4;

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub transport: String,
    pub peers: [SocketAddr; 4],
    pub executors: [u64; 4],
    pub capacity_rows: u32,
    #[serde(default = "default_lanes")]
    pub lanes: usize,
    #[serde(default = "default_timeout")]
    pub timeout_ms: u64,
    #[serde(default = "default_frame")]
    pub max_frame_bytes: usize,
    #[serde(default)]
    pub device: i32,
}
fn default_lanes() -> usize {
    2
}
fn default_timeout() -> u64 {
    1000
}
fn default_frame() -> usize {
    64 * 1024 * 1024
}

impl Config {
    fn validate(&self) -> Result<()> {
        ensure!(
            matches!(self.transport.as_str(), "tcp" | "roce"),
            "transport must be tcp or roce"
        );
        ensure!((1..=2).contains(&self.lanes), "lanes must be one or two");
        ensure!(
            (1..=4096).contains(&self.capacity_rows),
            "capacity_rows must be 1..4096"
        );
        ensure!(
            self.timeout_ms > 0 && self.timeout_ms <= 120_000,
            "timeout_ms must be 1..120000"
        );
        ensure!(self.device >= 0, "device must be nonnegative");
        // Delegate geometry, frame budget, endpoint and identity checks to the
        // same constructors used by native serving; this does not connect.
        let _ = V41Tp4Tcp::new(
            self.peers,
            self.executors,
            self.capacity_rows,
            self.wire_config(),
        )?;
        Ok(())
    }
    fn wire_config(&self) -> TcpTransportConfig {
        TcpTransportConfig {
            timeout: Duration::from_millis(self.timeout_ms),
            max_frame_bytes: self.max_frame_bytes,
        }
    }
}

pub struct Slot {
    pub ticket: u64,
    pub state: u32,
    pub output: Vec<u8>,
    pub error: String,
    pub cancel: Option<Arc<AtomicBool>>,
    pub buffer_capacity: usize,
    pub buffer_growths: u64,
}
impl Default for Slot {
    fn default() -> Self {
        Self {
            ticket: 0,
            state: IDLE,
            output: Vec::new(),
            error: String::new(),
            cancel: None,
            buffer_capacity: 0,
            buffer_growths: 0,
        }
    }
}
struct Job {
    ticket: u64,
    request: ExpertProtocolV2Request,
    cancel: Arc<AtomicBool>,
    output: Vec<u8>,
}

pub struct Bridge {
    pub config: Config,
    pub slots: Arc<Vec<Mutex<Slot>>>,
    pub last_error: Mutex<String>,
    senders: Vec<mpsc::Sender<Job>>,
    signals: Vec<Arc<Notify>>,
    shutdown: Arc<AtomicBool>,
    next_ticket: AtomicU64,
    worker: Option<JoinHandle<()>>,
}

impl Bridge {
    pub fn new(config: Config) -> Result<Self> {
        config.validate()?;
        let slots = Arc::new(
            (0..config.lanes)
                .map(|_| Mutex::new(Slot::default()))
                .collect::<Vec<_>>(),
        );
        let signals = (0..config.lanes)
            .map(|_| Arc::new(Notify::new()))
            .collect::<Vec<_>>();
        let shutdown = Arc::new(AtomicBool::new(false));
        let mut senders = Vec::new();
        let mut receivers = Vec::new();
        for _ in 0..config.lanes {
            let (tx, rx) = mpsc::channel(1);
            senders.push(tx);
            receivers.push(rx);
        }
        let worker_config = config.clone();
        let worker_slots = Arc::clone(&slots);
        let worker_signals = signals.clone();
        let worker_shutdown = Arc::clone(&shutdown);
        let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(1);
        let worker = thread::Builder::new()
            .name("afd-native-owner".into())
            .spawn(move || {
                let started = (|| -> Result<_> {
                    // QP and native GPU resources are created on their sole owner
                    // thread. TCP never loads a CUDA/RDMA library.
                    let native = if worker_config.transport == "roce" {
                        let path = std::env::var("DS41RT_NATIVE_LIB")
                            .context("RoCE requires DS41RT_NATIVE_LIB")?;
                        let lib = unsafe { ds41rt_ffi::NativeLibrary::load(path)? };
                        lib.cuda_set_device(worker_config.device)?;
                        Some(lib)
                    } else {
                        None
                    };
                    let runtime = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()?;
                    let transports = (0..worker_config.lanes)
                        .map(|_| Transport::new(&worker_config))
                        .collect::<Result<Vec<_>>>()?;
                    Ok((native, runtime, transports))
                })();
                let (_native, runtime, transports) = match started {
                    Ok(started) => {
                        let _ = ready_tx.send(Ok(()));
                        started
                    }
                    Err(error) => {
                        let _ = ready_tx.send(Err(error.to_string()));
                        return;
                    }
                };
                let local = tokio::task::LocalSet::new();
                runtime.block_on(local.run_until(async move {
                    let mut tasks = Vec::new();
                    for (index, (receiver, transport)) in
                        receivers.into_iter().zip(transports).enumerate()
                    {
                        tasks.push(tokio::task::spawn_local(run_lane(
                            index,
                            transport,
                            receiver,
                            Arc::clone(&worker_slots),
                            Arc::clone(&worker_signals[index]),
                            Arc::clone(&worker_shutdown),
                            Duration::from_millis(worker_config.timeout_ms),
                        )));
                    }
                    for task in tasks {
                        let _ = task.await;
                    }
                }));
            })?;
        match ready_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                config,
                slots,
                last_error: Mutex::new(String::new()),
                senders,
                signals,
                shutdown,
                next_ticket: AtomicU64::new(1),
                worker: Some(worker),
            }),
            error => {
                let _ = worker.join();
                anyhow::bail!("native owner initialization failed: {error:?}")
            }
        }
    }

    pub fn submit(&self, lane: usize, frame: &[u8]) -> std::result::Result<u64, (i32, String)> {
        let slot = self
            .slots
            .get(lane)
            .ok_or((crate::INVALID, "lane out of range".into()))?;
        if frame.len() > self.config.max_frame_bytes {
            return Err((crate::INVALID, "request exceeds max_frame_bytes".into()));
        }
        // Borrowed validation rejects malformed extents/rows before the owned
        // codec allocates. No independent ProtocolV2 implementation exists here.
        V41BackboneRequest::parse(frame, self.config.capacity_rows)
            .map_err(|e| (crate::INVALID, format!("invalid native request: {e:#}")))?;
        let request = ExpertProtocolV2Request::decode(frame)
            .map_err(|e| (crate::INVALID, format!("invalid request frame: {e:#}")))?;
        let mut slot = slot
            .lock()
            .map_err(|_| (crate::INTERNAL, "lane mutex poisoned".into()))?;
        if matches!(slot.state, PENDING | READY) {
            return Err((crate::BUSY, "lane has pending or uncollected output".into()));
        }
        let ticket = self
            .next_ticket
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| (crate::INTERNAL, "ticket exhausted".into()))?;
        let cancel = Arc::new(AtomicBool::new(false));
        let output = std::mem::take(&mut slot.output);
        if let Err(error) = self.senders[lane].try_send(Job {
            ticket,
            request,
            cancel: Arc::clone(&cancel),
            output,
        }) {
            slot.output = error.into_inner().output;
            return Err((crate::INTERNAL, "native owner unavailable".into()));
        }
        slot.ticket = ticket;
        slot.state = PENDING;
        slot.error.clear();
        slot.cancel = Some(cancel);
        Ok(ticket)
    }

    pub fn cancel(&self, lane: usize, ticket: u64) -> std::result::Result<(), (i32, String)> {
        let slot = self
            .slots
            .get(lane)
            .ok_or((crate::INVALID, "lane out of range".into()))?;
        let mut slot = slot
            .lock()
            .map_err(|_| (crate::INTERNAL, "lane mutex poisoned".into()))?;
        if ticket == 0 || slot.ticket != ticket {
            return Err((crate::STALE, "ticket is not current".into()));
        }
        match slot.state {
            PENDING => {
                slot.cancel
                    .as_ref()
                    .expect("pending cancellation owner")
                    .store(true, Ordering::Release);
                self.signals[lane].notify_one();
            }
            READY => {
                slot.state = CANCELLED;
                slot.error = "uncollected result discarded".into();
            }
            _ => {}
        }
        Ok(())
    }
}
impl Drop for Bridge {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        for signal in &self.signals {
            signal.notify_one();
        }
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

enum Transport {
    Tcp(V41Tp4Tcp),
    Roce(V41Tp4Roce),
}
impl Transport {
    fn new(c: &Config) -> Result<Self> {
        Ok(match c.transport.as_str() {
            "tcp" => Self::Tcp(V41Tp4Tcp::new(
                c.peers,
                c.executors,
                c.capacity_rows,
                c.wire_config(),
            )?),
            "roce" => Self::Roce(V41Tp4Roce::new(
                c.peers,
                c.executors,
                c.capacity_rows,
                c.wire_config(),
            )?),
            _ => unreachable!("validated transport"),
        })
    }
    fn reset(&mut self) {
        match self {
            Self::Tcp(t) => t.reset_connections(),
            Self::Roce(t) => t.reset_connections(),
        }
    }
    async fn exchange(
        &mut self,
        request: &ExpertProtocolV2Request,
        output: &mut Vec<u8>,
    ) -> Result<()> {
        let plane_bytes = request.header.row_count as usize * V41_PARTIAL_ROW_BYTES as usize;
        ensure!(
            output.len() == plane_bytes * 4,
            "rank buffer does not match request"
        );
        let mut copy = |rank: usize, first: u32, bytes: &[u8]| -> Result<()> {
            let start = rank * plane_bytes + first as usize * V41_PARTIAL_ROW_BYTES as usize;
            let end = start
                .checked_add(bytes.len())
                .context("response extent overflow")?;
            ensure!(
                rank < 4 && end <= (rank + 1) * plane_bytes,
                "response exceeds rank plane"
            );
            output[start..end].copy_from_slice(bytes);
            Ok(())
        };
        match self {
            Self::Tcp(t) => t.dispatch(request).await?.receive(&mut copy).await?,
            Self::Roce(t) => t.dispatch(request).await?.receive(&mut copy).await?,
        }
        Ok(())
    }
}

async fn wait_cancel(signal: &Notify, cancel: &AtomicBool, shutdown: &AtomicBool) {
    loop {
        if cancel.load(Ordering::Acquire) || shutdown.load(Ordering::Acquire) {
            return;
        }
        signal.notified().await;
    }
}
async fn run_lane(
    index: usize,
    mut transport: Transport,
    mut receiver: mpsc::Receiver<Job>,
    slots: Arc<Vec<Mutex<Slot>>>,
    signal: Arc<Notify>,
    shutdown: Arc<AtomicBool>,
    timeout: Duration,
) {
    loop {
        if shutdown.load(Ordering::Acquire) {
            break;
        }
        let mut job = tokio::select! {
            biased;
            _ = signal.notified() => { continue; }
            job = receiver.recv() => match job { Some(job) => job, None => break },
        };
        // Recycle this buffer after collect, cancellation or failure. Reserve
        // exactly the required growth so geometric Vec growth cannot exceed
        // the configured row cap. Equal/shrinking shapes allocate nothing.
        let required = job.request.header.row_count as usize * V41_PARTIAL_ROW_BYTES as usize * 4;
        let grew = required > job.output.capacity();
        if grew {
            job.output.reserve_exact(required - job.output.len());
        }
        job.output.resize(required, 0);
        if let Ok(mut slot) = slots[index].lock() {
            slot.buffer_capacity = job.output.capacity();
            slot.buffer_growths += u64::from(grew);
        }
        // The async future owns no self-referential handle. This task owns the
        // transport and request; their ordinary borrows end before reset or
        // terminal publication. Both lanes run on the same native owner thread.
        let result = tokio::select! {
            biased;
            _ = wait_cancel(&signal, &job.cancel, &shutdown) => None,
            result = tokio::time::timeout(timeout, transport.exchange(&job.request, &mut job.output)) => Some(result),
        };
        let cancelled = job.cancel.load(Ordering::Acquire) || shutdown.load(Ordering::Acquire);
        let (state, error) = match result {
            _ if cancelled => (
                CANCELLED,
                "operation cancelled after transport drain/reset".into(),
            ),
            Some(Ok(Ok(()))) => (READY, String::new()),
            Some(Ok(Err(error))) => (FAILED, format!("transport failed: {error:#}")),
            Some(Err(_)) => (FAILED, "operation deadline expired".into()),
            None => (CANCELLED, "operation cancelled".into()),
        };
        if state != READY {
            transport.reset();
        }
        if let Ok(mut slot) = slots[index].lock() {
            if slot.ticket == job.ticket {
                slot.output = job.output;
                // cancel may race completion after the outer check. Discard a
                // completed result rather than publishing cancelled work READY.
                if job.cancel.load(Ordering::Acquire) && state == READY {
                    transport.reset();
                    slot.state = CANCELLED;
                    slot.error = "completed result cancelled".into();
                } else {
                    slot.state = state;
                    slot.error = error;
                }
                slot.cancel = None;
            }
        }
    }
}
