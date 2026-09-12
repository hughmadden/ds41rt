//! Bounded background gathers with reusable storage and request-owned cancellation.
use crate::{EngramBatchStaging, EngramGatherView, EngramTable};
use anyhow::{ensure, Context, Result};
use ds41rt_core::{EngramBatch, ENGRAM_ROWS};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Arc, Mutex,
};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Recorded only when the runtime's timing trace is enabled at submission.
pub struct EngramGatherTiming {
    pub queued: Duration,
    pub gather: Duration,
    pub completed: Instant,
    /// Fault/I/O counts are -1 when the OS counter query was unavailable.
    pub minor_faults: i64,
    pub major_faults: i64,
    pub input_blocks: i64,
}

fn thread_usage() -> Option<libc::rusage> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: getrusage initializes the complete output on success.
    if unsafe { libc::getrusage(libc::RUSAGE_THREAD, usage.as_mut_ptr()) } == 0 {
        Some(unsafe { usage.assume_init() })
    } else {
        None
    }
}

/// Holds a staging slot until its synchronous GPU upload has consumed the view.
/// Dropping a ready result returns the slot to the pool without blocking.
pub struct EngramGatherLease {
    staging: Option<EngramBatchStaging>,
    recycler: mpsc::SyncSender<EngramBatchStaging>,
    batches: Vec<Arc<EngramBatch>>,
    timing: Option<EngramGatherTiming>,
}
impl EngramGatherLease {
    pub fn view(&self) -> Result<EngramGatherView<'_>> {
        self.staging
            .as_ref()
            .context("engram gather lease is empty")?
            .view()
    }
    /// Exact immutable batches represented by the concatenated output rows.
    pub fn batches(&self) -> &[Arc<EngramBatch>] {
        &self.batches
    }
    pub fn timing(&self) -> Option<&EngramGatherTiming> {
        self.timing.as_ref()
    }
}
impl Drop for EngramGatherLease {
    fn drop(&mut self) {
        if let Some(staging) = self.staging.take() {
            // Full/disconnected means the pool is no longer usable; dropping is safe.
            let _ = self.recycler.try_send(staging);
        }
    }
}

pub enum EngramGatherPoll {
    Pending,
    Cancelled,
    Ready(EngramGatherLease),
}
/// Independent of scheduler slot reuse; drop cancels queued/in-progress work.
/// An in-progress OS page fault cannot be interrupted, but its result is discarded.
pub struct EngramGatherTicket {
    cancelled: Arc<AtomicBool>,
    completion: Option<mpsc::Receiver<Result<Option<EngramGatherLease>>>>,
    consumed: bool,
}
impl EngramGatherTicket {
    pub fn cancel(&mut self) {
        self.cancelled.store(true, Ordering::Release);
        self.completion.take();
    }
    /// Nonblocking polling for a CUDA or scheduler thread; consume at most once.
    pub fn poll(&mut self) -> Result<EngramGatherPoll> {
        ensure!(!self.consumed, "engram gather completion already consumed");
        if self.cancelled.load(Ordering::Acquire) {
            self.consumed = true;
            return Ok(EngramGatherPoll::Cancelled);
        }
        match self
            .completion
            .as_ref()
            .context("engram completion receiver is closed")?
            .try_recv()
        {
            Ok(result) => {
                self.consumed = true;
                self.completion.take();
                Ok(match result? {
                    Some(lease) => EngramGatherPoll::Ready(lease),
                    None => EngramGatherPoll::Cancelled,
                })
            }
            Err(mpsc::TryRecvError::Empty) => Ok(EngramGatherPoll::Pending),
            Err(mpsc::TryRecvError::Disconnected) => {
                self.consumed = true;
                self.completion.take();
                anyhow::bail!("engram gather worker stopped before completion")
            }
        }
    }
}
impl Drop for EngramGatherTicket {
    fn drop(&mut self) {
        self.cancel();
    }
}
struct Job {
    submitted: Option<Instant>,
    table: Arc<EngramTable>,
    layer: usize,
    lease: EngramGatherLease,
    cancelled: Arc<AtomicBool>,
    completion: mpsc::SyncSender<Result<Option<EngramGatherLease>>>,
}

pub struct EngramGatherer {
    sender: Option<mpsc::SyncSender<Job>>,
    worker: Option<JoinHandle<()>>,
    pool: Mutex<mpsc::Receiver<EngramBatchStaging>>,
    recycler: mpsc::SyncSender<EngramBatchStaging>,
    capacity: usize,
    shutdown: Arc<AtomicBool>,
}
impl EngramGatherer {
    /// The slot pool bounds queued, in-flight and completed-but-unconsumed storage.
    pub fn new(slots: usize, capacity: usize, staging_budget: usize) -> Result<Self> {
        ensure!(
            slots > 0 && slots <= 32,
            "engram gather pool requires 1..32 slots"
        );
        let bytes = EngramBatchStaging::storage_bytes(capacity)?
            .checked_mul(slots)
            .context("engram gather pool budget overflow")?;
        ensure!(
            bytes <= staging_budget,
            "engram gather pool exceeds host staging budget"
        );
        let (recycler, pool) = mpsc::sync_channel(slots);
        for _ in 0..slots {
            recycler
                .try_send(EngramBatchStaging::new(capacity)?)
                .map_err(|_| anyhow::anyhow!("initializing engram staging pool failed"))?;
        }
        let (sender, jobs) = mpsc::sync_channel::<Job>(slots);
        let shutdown = Arc::new(AtomicBool::new(false));
        let stopping = Arc::clone(&shutdown);
        let worker = thread::Builder::new()
            .name("engram-gather".into())
            .spawn(move || {
                while let Ok(mut job) = jobs.recv() {
                    if stopping.load(Ordering::Acquire) || job.cancelled.load(Ordering::Acquire) {
                        let _ = job.completion.send(Ok(None));
                        continue;
                    }
                    let batches: Vec<_> = job.lease.batches.iter().map(Arc::as_ref).collect();
                    let before = job.submitted.and_then(|_| thread_usage());
                    let started = job.submitted.map(|_| Instant::now());
                    let result = job
                        .lease
                        .staging
                        .as_mut()
                        .expect("owned gather slot")
                        .gather(&job.table, &batches, job.layer)
                        .map(|_| ());
                    if let (Some(submitted), Some(started)) = (job.submitted, started) {
                        let completed = Instant::now();
                        let after = thread_usage();
                        let counts = before
                            .zip(after)
                            .map(|(a, b)| {
                                [
                                    b.ru_minflt - a.ru_minflt,
                                    b.ru_majflt - a.ru_majflt,
                                    b.ru_inblock - a.ru_inblock,
                                ]
                            })
                            .unwrap_or([-1; 3]);
                        job.lease.timing = Some(EngramGatherTiming {
                            queued: started.duration_since(submitted),
                            gather: completed.duration_since(started),
                            completed,
                            minor_faults: counts[0],
                            major_faults: counts[1],
                            input_blocks: counts[2],
                        });
                    }
                    let result = if stopping.load(Ordering::Acquire)
                        || job.cancelled.load(Ordering::Acquire)
                    {
                        Ok(None)
                    } else {
                        result.map(|()| Some(job.lease))
                    };
                    let _ = job.completion.send(result);
                }
            })
            .context("starting engram gather worker")?;
        Ok(Self {
            sender: Some(sender),
            worker: Some(worker),
            pool: Mutex::new(pool),
            recycler,
            capacity,
            shutdown,
        })
    }
    /// Submit early when decode/prefill/verification token hashes become available.
    /// None is bounded backpressure; it performs no mapped reads or blocking wait.
    pub fn try_submit(
        &self,
        table: Arc<EngramTable>,
        batches: &[Arc<EngramBatch>],
        layer: usize,
    ) -> Result<Option<EngramGatherTicket>> {
        ensure!(
            !batches.is_empty() && batches.len() <= 16,
            "invalid engram request batch count"
        );
        let rows = batches.iter().try_fold(0usize, |sum, batch| {
            sum.checked_add(batch.hashes().len())
                .context("engram batch length overflow")
        })?;
        ensure!(
            rows > 0 && rows <= self.capacity,
            "engram gather exceeds row capacity"
        );
        ensure!(
            table.weights().rows() == *ENGRAM_ROWS.get(layer).context("invalid engram layer")?,
            "engram gather table belongs to another layer"
        );
        let staging = match self.pool.try_lock() {
            Ok(pool) => match pool.try_recv() {
                Ok(staging) => staging,
                Err(mpsc::TryRecvError::Empty) => return Ok(None),
                Err(mpsc::TryRecvError::Disconnected) => {
                    anyhow::bail!("engram staging pool stopped")
                }
            },
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => {
                anyhow::bail!("engram staging pool poisoned")
            }
        };
        let cancelled = Arc::new(AtomicBool::new(false));
        let (completion, receive) = mpsc::sync_channel(1);
        let job = Job {
            submitted: tracing::enabled!(target: "ds41rt::timing", tracing::Level::DEBUG)
                .then(Instant::now),
            table,
            layer,
            lease: EngramGatherLease {
                staging: Some(staging),
                recycler: self.recycler.clone(),
                batches: batches.to_vec(),
                timing: None,
            },
            cancelled: Arc::clone(&cancelled),
            completion,
        };
        match self
            .sender
            .as_ref()
            .context("engram gatherer stopped")?
            .try_send(job)
        {
            Ok(()) => Ok(Some(EngramGatherTicket {
                cancelled,
                completion: Some(receive),
                consumed: false,
            })),
            Err(mpsc::TrySendError::Full(_)) => Ok(None),
            Err(mpsc::TrySendError::Disconnected(_)) => {
                anyhow::bail!("engram gather worker stopped")
            }
        }
    }
}
impl Drop for EngramGatherer {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        self.sender.take();
        if let Some(worker) = self.worker.take() {
            if worker.join().is_err() {
                eprintln!("engram gather worker panicked during shutdown");
            }
        }
    }
}
