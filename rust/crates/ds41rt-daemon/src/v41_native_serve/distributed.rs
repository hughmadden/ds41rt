//! Production owners for the 20/20 attention split and TP2 encoder experts.
use super::*;
use crate::v41_backbone_cache::CachePlacement;
use crate::v41_backbone_execution::DistributedExecution;
use crate::v41_backbone_shared::tp2::Weights as SharedWeights;
use crate::v41_engram::placement::{PlacedEngram, PlacedEngramWeights};
use crate::v41_experts::{tp2::RankWeights, tp2_ffn, ExpertLayer, ExpertWeights};
use crate::v41_memory::device::Device;
use crate::v41_target_head::distributed_target::DistributedTargetHead;
use crate::v41_target_pass::{DistributedTargetPass, TargetTapWave};
use std::rc::Rc;

pub(super) fn worker(args: crate::cli::NativeServeArgs, mut receive: mpsc::Receiver<NativeRequest>,
    ready: &mut Option<oneshot::Sender<std::result::Result<(), String>>>) -> Result<()> {
    ensure!(matches!(args.rtx_expert_layers, memory::LocalLayers::Auto | memory::LocalLayers::Count(20)),
        "two RTX serving requires all 20 encoder expert layers");
    let capacity = prefill_capacity(args.prefill_batch_tokens)?;
    let lib = unsafe { NativeLibrary::load(&args.native_lib)? };
    lib.cuda_set_device(0)?;
    let devices = [Device { library: &lib, id: 0 }, Device { library: &lib, id: 1 }];
    for device in devices { device.run(|| lib.cuda_enable_peer(1 - device.id))?; }
    let memory_checkpoint = |stage: &str| -> Result<()> {
        let occupied = devices.map(|d| d.run(|| { let (free, total) = lib.cuda_memory_info()?; Ok(total - free) }))
            .into_iter().collect::<Result<Vec<_>>>()?;
        tracing::info!(stage, occupied_bytes=?occupied, "dual RTX startup memory");
        Ok(())
    };
    let catalog = ds41rt_loader::read_official_v41_catalog(ds41rt_loader::OFFICIAL_V41_MODEL_ID, &args.snapshot)?;
    let map = CachePlacement::encoder_decoder();
    let started = Instant::now();
    // Sum resident storage plus the largest transient loading excess.
    let budgets = (0..20).map(|layer| ExpertWeights::plan(&lib, &catalog,
        ExpertLayer::BackboneTp2 { layer, rank: 0 })).collect::<Result<Vec<_>>>()?;
    let resident = budgets.iter().try_fold(0usize, |sum, b| sum.checked_add(b.resident_bytes)
        .context("encoder resident budget overflow"))?;
    let transient = budgets.iter().map(|b| Ok(b.peak_device_bytes()? - b.resident_bytes))
        .collect::<Result<Vec<_>>>()?.into_iter().max().unwrap_or(0);
    let rank_budget = resident.checked_add(transient).context("encoder load budget overflow")?;
    let weights = BackboneLaneWeights::load_distributed(
        &lib,
        &catalog,
        map,
        BackboneLaneWeights::distributed_device_bytes(&lib, &catalog, map)?,
        16 << 20,
    )?;
    let producers = CacheProducerWeights::load_distributed(
        &lib,
        &catalog,
        map,
        CacheProducerWeights::distributed_device_bytes(&lib, &catalog, map)?,
        16 << 20,
    )?;
    let iw = IndexLaneWeights::load_distributed(
        &lib,
        &catalog,
        map,
        IndexLaneWeights::distributed_device_bytes(&lib, &catalog, map)?,
        16 << 20,
    )?;
    let ew = PlacedEngramWeights::load(
        &lib,
        &catalog,
        map,
        PlacedEngramWeights::device_bytes(&lib, &catalog, map)?,
        16 << 20,
    )?;
    let names = ["embed.weight".to_string()];
    let table = devices[0].own(|| {
        NativeRtxTensors::load(
            &lib,
            &catalog,
            &names,
            NativeRtxTensors::plan(&catalog, &names)?,
            16 << 20,
        )
    })?;
    let vocab = [
        devices[0].own(|| crate::v41_tensors::VocabularyShard::load(&lib, &catalog, 0..64640, 1 << 30, 16 << 20))?,
        devices[1].own(|| crate::v41_tensors::VocabularyShard::load(&lib, &catalog, 64640..129280, 1 << 30, 16 << 20))?,
    ];
    let hw = devices[1].own(|| {
        TargetHeadWeights::load(
            &lib,
            &catalog,
            TargetHeadWeights::device_bytes(&catalog)?,
            16 << 20,
        )
    })?;
    eprintln!("loading all 20 encoder expert layers as TP2");
    let routed = [
        Rc::new(RankWeights::load(devices[0], &catalog, 20, rank_budget)?),
        Rc::new(RankWeights::load(devices[1], &catalog, 20, rank_budget)?),
    ];
    let shared: [Rc<Vec<SharedWeights<'_>>>; 2] = [0, 1]
        .map(|r| {
            (0..40)
                .map(|l| {
                    SharedWeights::load(
                        devices[r],
                        &catalog,
                        l,
                        SharedWeights::load_peak_device_bytes(),
                    )
                })
                .collect::<Result<Vec<_>>>()
                .map(Rc::new)
        })
        .into_iter()
        .collect::<Result<Vec<_>>>()?
        .try_into()
        .ok()
        .expect("two ranks");
    memory_checkpoint("weights")?;
    tracing::info!(capacity, backbone_lane_bytes=BackboneLane::placed_workspace_bytes(&lib, capacity)?,
        producer_bytes=?crate::v41_backbone_execution::PlacedProducerWaves::device_bytes(&lib, map, capacity)?,
        "dual RTX per-lane workspace plan");
    let make_pass = || -> Result<DistributedTargetPass<'_, '_>> {
        let lanes = [
            BackboneLane::new_on_device(
                &weights,
                capacity,
                BackboneLane::placed_workspace_bytes(&lib, capacity)?,
                0,
            )?,
            BackboneLane::new_on_device(
                &weights,
                capacity,
                BackboneLane::placed_workspace_bytes(&lib, capacity)?,
                1,
            )?,
        ];
        let indices = [
            Some(IndexLane::new_on_device(
                &iw,
                capacity,
                IndexLane::placed_workspace_bytes(&lib, map, capacity, 0)?
                    .iter()
                    .sum(),
                0,
            )?),
            Some(IndexLane::new_on_device(
                &iw,
                capacity,
                IndexLane::placed_workspace_bytes(&lib, map, capacity, 1)?
                    .iter()
                    .sum(),
                1,
            )?),
        ];
        DistributedTargetPass::new(
            map,
            devices[0].own(|| {
                TargetEmbeddingWave::new(
                    &lib,
                    &table,
                    capacity as usize,
                    TargetEmbeddingWave::device_bytes(capacity as usize)?,
                )
            })?,
            lanes,
            indices,
            DistributedExecution::new(
                &producers,
                capacity,
                crate::v41_backbone_execution::PlacedProducerWaves::device_bytes(
                    &lib, map, capacity,
                )?,
            )?,
            PlacedEngram::new(&ew, capacity as usize, PlacedEngram::device_bytes(&lib, map, capacity as usize)?)?,
            DistributedTargetHead::new(devices, &hw, [&vocab[0], &vocab[1]], 48,
                DistributedTargetHead::device_bytes(48, 64640)?)?,
            devices[1]
                .own(|| TargetTapWave::new(&lib, capacity as usize, TargetTapWave::device_bytes(capacity as usize)?))?,
            Duration::from_secs(120),
        )
    };
    let mut pass = make_pass().context("constructing first distributed target lane")?;
    memory_checkpoint("first target lane")?;
    let mut second = make_pass().context("constructing second distributed target lane")?;
    memory_checkpoint("second target lane")?;
    let make_transport = || {
        let mut transport = devices[1].own(|| NativeTp4Wave::new(&lib,
            V41Tp4Roce::new(args.peers.clone().try_into().map_err(|_| anyhow::anyhow!("four Spark peers required"))?,
                [1, 2, 3, 4], capacity, TcpTransportConfig { timeout: Duration::from_secs(120),
                    max_frame_bytes: 64 << 20 })?, NativeTp4Wave::device_bytes(capacity)?))?;
        transport.install_tp2(tp2_ffn::Wave::new(routed.clone(), shared.clone(), 20, capacity)?)?;
        Ok::<_, anyhow::Error>(transport)
    };
    let mut transport = make_transport()?;
    let mut second_transport = make_transport()?;
    memory_checkpoint("TP2 transports")?;
    let draft_weights = if args.dspark {
        Some(devices[1].own(|| crate::v41_experts::dspark::DsparkWeights::load_serving(&lib, &catalog,
            capacity, args.concurrency, 32 << 30, 16 << 20))?)
    } else { None };
    let mut draft = draft_weights.as_ref().map(|weights| DraftRuntime::with_distributed_requests(
        devices, weights, &table, [&vocab[0], &vocab[1]], capacity, args.concurrency)).transpose()?;
    if let Some(draft) = &mut draft {
        draft.set_draft_limit(args.dspark_draft_limit)?;
        draft.set_adaptive(args.adaptive_dspark());
        draft.set_confidence_cutoff(args.dspark_confidence_cutoff);
        draft.set_reuse_floor(args.dspark_reuse_floor)?;
    }
    memory_checkpoint("draft runtime")?;
    // Vision and target snapshot copies use GPU0. These allocations precede KV sizing.
    let mut vision = crate::v41_vision::VisionRuntime::new(&lib, &catalog, 9216,
        crate::v41_vision::VisionRuntime::device_bytes(&catalog, 9216)?)?;
    memory_checkpoint("vision")?;
    ensure!(args.prefix_cache_entries <= 128, "invalid retained-turn limit");
    let snapshot_slots = if args.prefix_cache_entries == 0 { 0 } else { 2 * args.prefix_cache_entries as usize + 2 };
    let target_prefix_pool = (snapshot_slots > 0).then(|| crate::v41_memory::SnapshotPool::new(
        &lib, crate::v41_backbone_cache::BackbonePrefix::device_bytes(), snapshot_slots)).transpose()?;
    let draft_snapshot_bytes = draft.as_mut().map(|d| d.reserve_prefixes(snapshot_slots)).transpose()?.unwrap_or(0);
    let snapshot_bytes = target_prefix_pool.as_ref().map_or(0, crate::v41_memory::SnapshotPool::device_bytes) + draft_snapshot_bytes;
    let memory = [devices[0].run(|| lib.cuda_memory_info())?, devices[1].run(|| lib.cuda_memory_info())?];
    let pool = memory::distributed::PoolPlan::new(map, args.concurrency as usize,
        args.max_context_tokens as usize, args.prefix_cache_entries as usize, snapshot_bytes,
        args.kv_pool_size, args.memory_reservation, memory)?;
    tracing::info!(source_pages=?pool.pages, global_bytes=pool.global_bytes, cache_bytes=?pool.cache_bytes,
        occupied_before=?pool.occupied_before, reservation_bytes=?pool.reservation_bytes,
        unused_bytes=?pool.unused_bytes, desired_groups=pool.desired_groups, snapshot_slots, snapshot_bytes,
        encoder_layers=20, "dual RTX cache reservation after fixed allocations");
    let token_map = ds41rt_loader::EngramTokenMap::from_file(&args.snapshot.join("tokenizer.json"))?;
    let rows = capacity as usize;
    let pipeline = unsafe { ds41rt_loader::EngramPipeline::new(&catalog, token_map, rows,
        2 * ds41rt_core::ENGRAM_LAYERS.len(), rows * 64 * 1024)? };
    let mut requests = Requests::new_distributed(&lib, pipeline, args.concurrency as usize,
        pool.pages, map, pool.cache_bytes)?;
    if let Some(pool) = target_prefix_pool { requests.install_prefix_pool(pool)?; }
    let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
    tracing::info!(elapsed_ms=started.elapsed().as_millis(), "dual RTX serving owners ready");
    ready.take().context("startup readiness missing")?.send(Ok(()))
        .map_err(|_| anyhow::anyhow!("API startup cancelled"))?;
    scheduler::serve(&lib, &args, &runtime, &mut receive, &mut pass, &mut second, &mut requests,
        &mut transport, &mut second_transport, draft.as_mut().map(|d| d.get_mut()), &mut vision)
}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires DS41RT_NATIVE_LIB, DS41RT_SNAPSHOT, DS41RT_DUAL_PEERS, two GPUs and live Sparks"]
    fn distributed_worker_serves_text_requests() -> Result<()> {
        use clap::{Args, FromArgMatches};
        let _ = tracing_subscriber::fmt().with_env_filter("info").try_init();
        let snapshot = std::env::var("DS41RT_SNAPSHOT")?;
        let native = std::env::var("DS41RT_NATIVE_LIB")?;
        let peers = std::env::var("DS41RT_DUAL_PEERS")?;
        let batch = std::env::var("DS41RT_WORKER_PREFILL").unwrap_or_else(|_| "80".into());
        let matches = crate::cli::NativeServeArgs::augment_args(clap::Command::new("fixture"))
            .try_get_matches_from(["fixture", "--snapshot", &snapshot, "--native-lib", &native,
                "--peers", &peers, "--dspark", "--rtx-gpus", "2", "--prefill-batch-tokens", &batch])?;
        let args = crate::cli::NativeServeArgs::from_arg_matches(&matches)?;
        let (send, receive) = mpsc::channel(16);
        let mut outputs = Vec::new();
        for text in ["What is two plus two? Answer briefly.", "Write a Python function that adds two numbers.",
            "What is two plus two? Answer briefly."] {
            let (events, output) = mpsc::channel(64);
            send.blocking_send(NativeRequest { prompt: format!("<｜begin▁of▁sentence｜><｜User｜>{text}<｜Assistant｜></think>"),
                constraint: None, images: Vec::new(), max_tokens: 8, events })?;
            outputs.push(output);
        }
        drop(send);
        let (ready, mut readiness) = oneshot::channel();
        super::super::worker(args, receive, &mut Some(ready))?;
        readiness.try_recv()?.map_err(anyhow::Error::msg)?;
        for (i, mut output) in outputs.into_iter().enumerate() {
            let mut ready = 0;
            let mut finish = 0;
            let mut response = String::new();
            while let Ok(chunk) = output.try_recv() {
                match chunk {
                    Ok(InferenceChunk::Ready { prompt_usage, .. }) => {
                        ready += 1;
                        if i == 2 { ensure!(prompt_usage.prompt_cache_hit_tokens == prompt_usage.prompt_tokens,
                            "repeated worker prompt did not restore the exact prefix"); }
                    },
                    Ok(InferenceChunk::Text { content, .. }) => response.push_str(&content),
                    Ok(InferenceChunk::Finish { .. }) => finish += 1,
                    Err(error) => anyhow::bail!("worker request {i} failed: {error:?}"),
                    _ => (),
                }
            }
            ensure!(ready == 1 && finish == 1 && !response.is_empty(), "worker request {i} did not complete");
            eprintln!("PASS distributed worker request {i}: {response:?}");
        }
        Ok(())
    }
}
