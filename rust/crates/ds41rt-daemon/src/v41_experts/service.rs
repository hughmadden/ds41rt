//! Native GPU resources remain on one owning thread; transport sees bounded queues.
use super::{ExpertLayer, ExpertWeights, HostExpertExchange};
use anyhow::{bail, ensure, Context, Result};
use ds41rt_ffi::NativeLibrary;
use ds41rt_loader::{read_official_v41_catalog, OFFICIAL_V41_MODEL_ID};
use ds41rt_transport::{
    v41_expert::V41BackboneRequest, ExpertProtocolV2RequestView, ExpertProtocolV2Response,
    ExpertProtocolV2ResponseRef, ProtocolV2ExpertExecutor,
};
use std::{path::PathBuf, sync::mpsc, thread};

pub(crate) async fn run(args: crate::cli::NativeExpertDaemonArgs) -> Result<()> {
    let config = NativeExpertServiceConfig {
        library: args.native_lib,
        snapshot: args.snapshot,
        rank: args.rank as usize,
        capacity: args.capacity,
        device_budget: args.device_budget_bytes,
        max_frame_bytes: args.max_frame_bytes,
    };
    let service = tokio::task::spawn_blocking(move || NativeExpertService::start(config))
        .await
        .context("native expert startup task failed")??;
    tracing::info!(
        rank = args.rank,
        capacity = args.capacity,
        layers = 40,
        "native V4.1 expert worker ready"
    );
    ds41rt_transport::serve_protocol_v2_tcp_with_executor(
        &args.listen,
        std::sync::Arc::new(service),
    )
    .await
}

pub(crate) struct NativeExpertServiceConfig {
    pub library: PathBuf,
    pub snapshot: PathBuf,
    pub rank: usize,
    pub capacity: u32,
    pub device_budget: usize,
    pub max_frame_bytes: usize,
}

struct Work {
    frame: Vec<u8>,
    responses: mpsc::SyncSender<WorkerResponse>,
}
enum WorkerResponse {
    Chunk(ExpertProtocolV2Response),
    Done(Result<()>),
}

pub(crate) struct NativeExpertService {
    requests: Option<mpsc::SyncSender<Work>>,
    worker: Option<thread::JoinHandle<Result<()>>>,
    capacity: u32,
    max_frame_bytes: usize,
}
impl NativeExpertService {
    /// Start with one visible CUDA device; all native allocations use logical GPU 0.
    /// Returns only after all forty backbone layers and shared wave workspace load.
    pub fn start(config: NativeExpertServiceConfig) -> Result<Self> {
        ensure!(config.rank < 4, "native expert rank must be 0..3");
        ensure!(
            matches!(config.capacity, 1 | 16 | 80 | 256 | 1024 | 4096),
            "unsupported native expert capacity"
        );
        ensure!(
            config.max_frame_bytes >= 128 + 122880 + 4,
            "native response frame budget cannot fit a token row"
        );
        ensure!(
            config.max_frame_bytes <= 64 * 1024 * 1024,
            "native frame budget exceeds TCP service limit"
        );
        let (requests, receive) = mpsc::sync_channel(16);
        let (ready, readiness) = mpsc::sync_channel(1);
        let capacity = config.capacity;
        let max_frame_bytes = config.max_frame_bytes;
        let worker = thread::Builder::new()
            .name(format!("v41-expert-rank-{}", config.rank))
            .spawn(move || run_worker(config, receive, ready))
            .context("starting native expert worker")?;
        let service = Self {
            requests: Some(requests),
            worker: Some(worker),
            capacity,
            max_frame_bytes,
        };
        readiness
            .recv()
            .context("native expert worker stopped during startup")??;
        Ok(service)
    }
}
impl Drop for NativeExpertService {
    fn drop(&mut self) {
        self.requests.take();
        if let Some(worker) = self.worker.take() {
            match worker.join() {
                Ok(Ok(())) => {}
                Ok(Err(error)) => tracing::error!(%error, "native expert worker stopped"),
                Err(_) => tracing::error!("native expert worker panicked"),
            }
        }
    }
}
impl ProtocolV2ExpertExecutor for NativeExpertService {
    fn name(&self) -> &'static str {
        "v41-native-tp4"
    }
    fn tcp_response_chunks(&self) -> bool {
        true
    }
    fn execute(&self, _: &ExpertProtocolV2RequestView<'_>) -> Result<ExpertProtocolV2Response> {
        bail!("native TP4 execution requires the bounded streaming interface")
    }
    fn execute_streaming(
        &self,
        request: &ExpertProtocolV2RequestView<'_>,
        emit: &mut dyn FnMut(ExpertProtocolV2ResponseRef<'_>) -> Result<()>,
    ) -> Result<()> {
        ensure!(
            request.frame_bytes().len() <= self.max_frame_bytes,
            "native request exceeds admitted frame size"
        );
        V41BackboneRequest::parse(request.frame_bytes(), self.capacity)?;
        let (responses, receive) = mpsc::sync_channel(1);
        self.requests
            .as_ref()
            .context("native expert service stopped")?
            .try_send(Work {
                frame: request.frame_bytes().to_vec(),
                responses,
            })
            .map_err(|error| match error {
                mpsc::TrySendError::Full(_) => {
                    anyhow::anyhow!("native expert request queue is full")
                }
                mpsc::TrySendError::Disconnected(_) => {
                    anyhow::anyhow!("native expert worker stopped")
                }
            })?;
        loop {
            match receive
                .recv()
                .context("native expert worker stopped before completion")?
            {
                WorkerResponse::Chunk(response) => emit(response.as_borrowed())?,
                WorkerResponse::Done(result) => return result,
            }
        }
    }
    // Native TP identity is the configured rank plus one, not a hash of a shared name.
    fn execute_streaming_with_identity(
        &self,
        request: &ExpertProtocolV2RequestView<'_>,
        emit: &mut dyn FnMut(ExpertProtocolV2ResponseRef<'_>) -> Result<()>,
    ) -> Result<()> {
        self.execute_streaming(request, emit)
    }
    fn execute_streaming_device_payload(
        &self,
        _: &ExpertProtocolV2RequestView<'_>,
        _: ds41rt_transport::ProtocolV2RequestDevicePayload,
        _: &mut dyn FnMut(ds41rt_transport::ProtocolV2ExecutorResponseRef<'_>) -> Result<()>,
    ) -> Result<()> {
        bail!("native expert service device ingress is not wired; use TCP host ingress")
    }
}

fn run_worker(
    config: NativeExpertServiceConfig,
    requests: mpsc::Receiver<Work>,
    ready: mpsc::SyncSender<Result<()>>,
) -> Result<()> {
    // The library and every GPU owner are constructed and dropped on this thread.
    let library = match unsafe { NativeLibrary::load(&config.library) } {
        Ok(library) => library,
        Err(error) => {
            let _ = ready.send(Err(error));
            return Ok(());
        }
    };
    let initialize = || -> Result<_> {
        let catalog = read_official_v41_catalog(OFFICIAL_V41_MODEL_ID, &config.snapshot)?;
        let mut resident = 0usize;
        let mut staging = 0usize;
        for layer in 0..40 {
            let plan = ExpertWeights::plan(
                &library,
                &catalog,
                ExpertLayer::Backbone {
                    layer,
                    rank: config.rank,
                },
            )?;
            resident = resident
                .checked_add(plan.resident_bytes)
                .context("resident budget overflow")?;
            staging = staging.max(plan.device_staging_bytes);
        }
        ensure!(
            resident
                .checked_add(staging)
                .context("loading budget overflow")?
                <= config.device_budget,
            "native TP weights and staging exceed device budget"
        );
        let workspace = ExpertWeights::plan_execution(&library, config.capacity)?.total()?;
        ensure!(
            resident
                .checked_add(workspace)
                .context("execution budget overflow")?
                <= config.device_budget,
            "native TP weights and execution workspace exceed device budget"
        );
        let mut weights = Vec::with_capacity(40);
        let mut remaining = config.device_budget;
        for layer in 0..40 {
            let weight = ExpertWeights::load(
                &library,
                &catalog,
                ExpertLayer::Backbone {
                    layer,
                    rank: config.rank,
                },
                remaining,
            )?;
            remaining = remaining
                .checked_sub(weight.budget().resident_bytes)
                .context("resident budget exhausted")?;
            weights.push(weight);
        }
        Ok((weights, remaining))
    };
    let (weights, remaining) = match initialize() {
        Ok(state) => state,
        Err(error) => {
            let _ = ready.send(Err(error));
            return Ok(());
        }
    };
    let mut execution = match weights[0].execution(config.capacity, remaining) {
        Ok(execution) => execution,
        Err(error) => {
            let _ = ready.send(Err(error));
            return Ok(());
        }
    };
    let mut exchange = HostExpertExchange::new(config.capacity)?;
    let mut row_indices = vec![0; config.capacity as usize];
    ready
        .send(Ok(()))
        .map_err(|_| anyhow::anyhow!("native startup caller disconnected"))?;
    while let Ok(work) = requests.recv() {
        let mut disconnected = false;
        let result = (|| -> Result<()> {
            let request = V41BackboneRequest::parse(&work.frame, config.capacity)?;
            execution.bind_layer(&weights[request.layer() as usize])?;
            execution.execute_host_chunks(
                &request,
                config.rank as u64 + 1,
                &mut exchange,
                &mut row_indices,
                config.max_frame_bytes,
                |response| {
                    if work
                        .responses
                        .send(WorkerResponse::Chunk(response.to_owned()?))
                        .is_err()
                    {
                        disconnected = true;
                        bail!("native response consumer disconnected");
                    }
                    Ok(())
                },
            )
        })();
        let failed = result.is_err();
        let _ = work.responses.send(WorkerResponse::Done(result));
        // A disconnected socket does not poison GPU execution, but a native error
        // may leave CUDA state unusable; stop rather than executing queued waves.
        if failed && !disconnected {
            bail!("native expert execution failed; worker stopped");
        }
    }
    Ok(())
}
