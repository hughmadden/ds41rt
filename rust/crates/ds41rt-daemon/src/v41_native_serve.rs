mod speculative;
use crate::v41_backbone_cache::BackboneCache;
use crate::v41_backbone_execution::BackboneExecution;
use crate::v41_backbone_execution::CacheProducerWeights;
use crate::v41_backbone_lane::BackboneLane;
use crate::v41_backbone_lane::BackboneLaneWeights;
use crate::v41_engram::{
    layer::{EngramGate, EngramLayerWeights},
    EngramDeviceRows,
};
use crate::v41_experts::coordinator::NativeTp4Wave;
use crate::v41_index_lane::IndexLane;
use crate::v41_index_lane::IndexLaneWeights;
use crate::v41_requests::{RequestTokens, Requests};
use crate::v41_target_embedding::TargetEmbeddingWave;
use crate::v41_target_head::{TargetHeadWave, TargetHeadWeights};
use crate::v41_target_pass::TargetPass;
use crate::v41_tensors::{NativeRtxTensors, VocabularyHead};
use anyhow::Context;
use anyhow::{ensure, Result};
use ds41rt_api::native_v41::{InferenceChunk, InferenceFinishReason, NativeRequest, PromptUsage};
use ds41rt_ffi::NativeLibrary;
use ds41rt_transport::v41_expert::V41Tp4Roce;
use ds41rt_transport::{ExpertV2SourceKind, TcpTransportConfig};
use speculative::DraftRuntime;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};

pub(crate) async fn run(args: crate::cli::NativeServeArgs) -> Result<()> {
    ensure!(args.peers.len() == 4, "four Spark peers required");
    let listen = args.listen.clone();
    let (send, receive) = mpsc::channel(16);
    let (ready, readiness) = oneshot::channel();
    let worker_thread = std::thread::Builder::new()
        .name("v41-target-cuda".into())
        .spawn(move || {
            let mut ready = Some(ready);
            let result = worker(args, receive, &mut ready);
            if let Some(ready) = ready.take() {
                let _ = ready.send(Err(result
                    .as_ref()
                    .err()
                    .map(|e| format!("{e:#}"))
                    .unwrap_or_else(|| "worker stopped during startup".into())));
            }
            if let Err(error) = result {
                tracing::error!(%error,"native target worker stopped");
            }
        })?;
    readiness
        .await
        .context("native target startup stopped")?
        .map_err(anyhow::Error::msg)?;
    let listener = tokio::net::TcpListener::bind(&listen).await?;
    tracing::info!(%listen,"native V4.1 target API ready");
    axum::serve(listener, ds41rt_api::native_v41::router(send))
        .with_graceful_shutdown(async {
            let mut term =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                    .expect("install SIGTERM handler");
            tokio::select! { _ = tokio::signal::ctrl_c() => {}, _ = term.recv() => {} }
        })
        .await?;
    tokio::task::spawn_blocking(move || worker_thread.join())
        .await?
        .map_err(|_| anyhow::anyhow!("native CUDA worker panicked during shutdown"))?;
    Ok(())
}
// Reserve a supported AOT capacity once; live prefill chunks retain the user's
// requested size. All backbone/draft workspaces and transport share this bound.
fn prefill_capacity(batch_tokens: u32) -> Result<u32> {
    anyhow::ensure!(
        (80..=4096).contains(&batch_tokens),
        "prefill batch must be in 80..=4096"
    );
    [80, 256, 1024, 4096]
        .into_iter()
        .find(|&capacity| capacity >= batch_tokens.max(128))
        .context("no prefill capacity covers the requested batch")
}

#[cfg(test)]
mod prefill_capacity_tests {
    use super::prefill_capacity;

    #[test]
    fn intermediate_batches_use_covering_preallocated_capacity() {
        for (batch, expected) in [
            (80, 256),
            (81, 256),
            (256, 256),
            (257, 1024),
            (1024, 1024),
            (1025, 4096),
            (2048, 4096),
            (4096, 4096),
        ] {
            assert_eq!(prefill_capacity(batch).unwrap(), expected);
        }
        for invalid in [0, 79, 4097, u32::MAX] {
            assert!(prefill_capacity(invalid).is_err());
        }
    }
}

fn worker(
    args: crate::cli::NativeServeArgs,
    mut receive: mpsc::Receiver<NativeRequest>,
    ready: &mut Option<oneshot::Sender<std::result::Result<(), String>>>,
) -> Result<()> {
    let capacity = prefill_capacity(args.prefill_batch_tokens)?;
    let rows = capacity as usize;
    let lib = unsafe { NativeLibrary::load(&args.native_lib)? };
    let catalog = ds41rt_loader::read_official_v41_catalog(
        ds41rt_loader::OFFICIAL_V41_MODEL_ID,
        &args.snapshot,
    )?;
    let start = Instant::now();
    let weights = BackboneLaneWeights::load(
        &lib,
        &catalog,
        BackboneLaneWeights::device_bytes(&lib, &catalog)?,
        16 * 1024 * 1024,
    )?;
    let producers = CacheProducerWeights::load(
        &lib,
        &catalog,
        CacheProducerWeights::device_bytes(&lib, &catalog)?,
        16 * 1024 * 1024,
    )?;
    let index_weights = IndexLaneWeights::load(
        &lib,
        &catalog,
        IndexLaneWeights::device_bytes(&lib, &catalog)?,
        16 * 1024 * 1024,
    )?;
    let names = ["embed.weight".to_string()];
    let table = NativeRtxTensors::load(
        &lib,
        &catalog,
        &names,
        NativeRtxTensors::plan(&catalog, &names)?,
        16 * 1024 * 1024,
    )?;
    eprintln!(
        "native target backbone/index/embedding weights loaded in {:.3}s",
        start.elapsed().as_secs_f64()
    );
    let embedding =
        TargetEmbeddingWave::new(&lib, &table, rows, TargetEmbeddingWave::device_bytes(rows)?)?;
    let lane = BackboneLane::new(
        &weights,
        capacity,
        BackboneLane::workspace_bytes(&lib, capacity)?.into_iter().sum(),
    )?;
    let index = IndexLane::new(
        &index_weights,
        capacity,
        IndexLane::workspace_bytes(&lib, capacity)?.into_iter().sum(),
    )?;
    let execution = BackboneExecution::new(
        &producers,
        capacity,
        BackboneExecution::workspace_bytes(&lib, capacity)?,
    )?;
    let map = ds41rt_loader::EngramTokenMap::from_file(&&args.snapshot.join("tokenizer.json"))?;
    let pipeline =
        unsafe { ds41rt_loader::EngramPipeline::new(&catalog, map, rows, 2, rows * 64 * 1024)? };
    let source_pages = BackboneCache::pages_for_context(16, args.max_context_tokens as usize)?;
    let mut requests = Requests::new(
        &lib,
        pipeline,
        16,
        source_pages,
        BackboneCache::device_bytes(16, source_pages)?,
    )?;
    let engram_weights = [0, 1]
        .map(|i| EngramLayerWeights::load(&lib, &catalog, i, 256 * 1024 * 1024, 16 * 1024 * 1024))
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    let gates = [
        EngramGate::new(&engram_weights[0], rows, 1024 * 1024 * 1024)?,
        EngramGate::new(&engram_weights[1], rows, 1024 * 1024 * 1024)?,
    ];
    let upload = EngramDeviceRows::new(&lib, rows, EngramDeviceRows::device_bytes(rows)?)?;
    let vocabulary = VocabularyHead::load(
        &lib,
        &catalog,
        VocabularyHead::plan(&catalog)?,
        16 * 1024 * 1024,
    )?;
    let head_weights = TargetHeadWeights::load(
        &lib,
        &catalog,
        TargetHeadWeights::device_bytes(&catalog)?,
        16 * 1024 * 1024,
    )?;
    let head = head_weights.wave(&vocabulary, 16, TargetHeadWave::device_bytes(16)?)?;
    let mut pass = TargetPass::new(
        embedding,
        lane,
        index,
        execution,
        upload,
        gates,
        head,
        crate::v41_target_pass::TargetTapWave::new(
            &lib,
            rows,
            crate::v41_target_pass::TargetTapWave::device_bytes(rows)?,
        )?,
        Duration::from_secs(120),
    )?;
    let roce = V41Tp4Roce::new(
        args.peers
            .clone()
            .try_into()
            .map_err(|_| anyhow::anyhow!("four Spark peers required"))?,
        [1, 2, 3, 4],
        capacity,
        TcpTransportConfig {
            timeout: Duration::from_secs(120),
            max_frame_bytes: 64 * 1024 * 1024,
        },
    )?;
    let mut transport = NativeTp4Wave::new(&lib, roce, NativeTp4Wave::device_bytes(capacity)?)?;
    let mut prefill_pass = TargetPass::new(
        TargetEmbeddingWave::new(&lib, &table, rows, TargetEmbeddingWave::device_bytes(rows)?)?,
        BackboneLane::new(&weights, capacity, BackboneLane::workspace_bytes(&lib, capacity)?.into_iter().sum())?,
        IndexLane::new(&index_weights, capacity, IndexLane::workspace_bytes(&lib, capacity)?.into_iter().sum())?,
        BackboneExecution::new(&producers, capacity, BackboneExecution::workspace_bytes(&lib, capacity)?)?,
        EngramDeviceRows::new(&lib, rows, EngramDeviceRows::device_bytes(rows)?)?,
        [EngramGate::new(&engram_weights[0], rows, 1024 * 1024 * 1024)?,
         EngramGate::new(&engram_weights[1], rows, 1024 * 1024 * 1024)?],
        head_weights.wave(&vocabulary, 16, TargetHeadWave::device_bytes(16)?)?,
        crate::v41_target_pass::TargetTapWave::new(&lib, rows, crate::v41_target_pass::TargetTapWave::device_bytes(rows)?)?,
        Duration::from_secs(120),
    )?;
    let prefill_roce = V41Tp4Roce::new(args.peers.clone().try_into()
        .map_err(|_| anyhow::anyhow!("four Spark peers required"))?, [1, 2, 3, 4], capacity,
        TcpTransportConfig { timeout: Duration::from_secs(120), max_frame_bytes: 64 * 1024 * 1024 })?;
    let mut prefill_transport = NativeTp4Wave::new(&lib, prefill_roce, NativeTp4Wave::device_bytes(capacity)?)?;
    let draft_weights = if args.dspark {
        Some(crate::v41_experts::dspark::DsparkWeights::load(
            &lib,
            &catalog,
            capacity,
            1,
            32 * 1024 * 1024 * 1024,
            16 * 1024 * 1024,
        )?)
    } else {
        None
    };
    let mut draft = draft_weights
        .as_ref()
        .map(|weights| DraftRuntime::new(&lib, weights, &table, &vocabulary, capacity))
        .transpose()?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    ready
        .take()
        .context("startup readiness missing")?
        .send(Ok(()))
        .map_err(|_| anyhow::anyhow!("API startup cancelled"))?;
    let mut id = 0u64;
    while let Some(job) = receive.blocking_recv() {
        if job.events.is_closed() {
            continue;
        }
        id = id.checked_add(1).context("request ID exhausted")?;
        // Keep healthy QPs across admissions; request/output ownership is fresh.
        transport.begin_request();
        prefill_transport.begin_request();
        let lease = requests.admit(0, id)?;
        if let Some(draft) = &mut draft {
            draft.admit(id)?;
        }
        let result = generate(
            &lib,
            &args.snapshot,
            args.max_context_tokens as usize,
            args.prefill_batch_tokens as usize,
            &runtime,
            &mut pass,
            &mut prefill_pass,
            &mut prefill_transport,
            &mut requests,
            &mut transport,
            lease,
            &job,
            draft.as_mut(),
        );
        let cleanup = if requests.cache().request_id(lease).is_ok() {
            requests.release(lease)
        } else {
            Ok(())
        };
        if let Some(draft) = &mut draft {
            draft.release()?;
        }
        if let Err(error) = result {
            // Also reset failures outside a pending transport borrow (for example
            // client cancellation between completed model steps).
            transport.reset_connections();
            prefill_transport.reset_connections();
            let _ = job.events.blocking_send(Err(format!("{error:#}")));
        }
        cleanup?;
    }
    Ok(())
}
fn generate<'w, 'a>(
    lib: &'a NativeLibrary,
    snapshot: &std::path::Path,
    max_context_tokens: usize,
    prefill_batch_tokens: usize,
    runtime: &tokio::runtime::Runtime,
    pass: &mut TargetPass<'w, 'a>,
    prefill_pass: &mut TargetPass<'w, 'a>,
    prefill_transport: &mut NativeTp4Wave<'a>,
    requests: &mut Requests<'a>,
    transport: &mut NativeTp4Wave<'a>,
    lease: crate::v41_backbone_cache::CacheLease,
    job: &NativeRequest,
    mut draft: Option<&mut DraftRuntime<'_, 'a>>,
) -> Result<()> {
    let prompt = ds41rt_loader::encode_tokenizer_text(snapshot, &job.prompt, false)?.token_ids;
    ensure!(
        !prompt.is_empty() && prompt.len().checked_add(job.max_tokens).is_some_and(|total| total <= max_context_tokens),
        "native text request exceeds {max_context_tokens}-token context limit"
    );
    let mut decoder = ds41rt_loader::streaming_token_decoder(snapshot, false)?;
    job.events.blocking_send(Ok(InferenceChunk::Ready {
        system_fingerprint: Some(
            if draft.is_some() {
                "ds41rt-native-fp8-kv-dspark"
            } else {
                "ds41rt-native-fp8-kv"
            }
            .into(),
        ),
        prompt_usage: PromptUsage {
            prompt_tokens: prompt.len(),
            prompt_cache_hit_tokens: 0,
        },
    }))?;
    let mut next = prefill(lib, runtime, pass, prefill_pass, requests, transport, prefill_transport, lease, &prompt,
        prefill_batch_tokens, job, draft.as_deref_mut())?;
    let mut buffered = 0usize;
    let mut pending = std::collections::VecDeque::new();
    for generated in 0..job.max_tokens {
        ensure!(!job.events.is_closed(), "client disconnected");
        buffered += 1;
        if next == 1 {
            job.events.blocking_send(Ok(InferenceChunk::Text {
                content: String::new(),
                content_tokens: buffered,
            }))?;
            job.events.blocking_send(Ok(InferenceChunk::Finish {
                finish_reason: InferenceFinishReason::Stop,
            }))?;
            return Ok(());
        }
        if let Some(content) = decoder.step(next)? {
            job.events.blocking_send(Ok(InferenceChunk::Text {
                content,
                content_tokens: buffered,
            }))?;
            buffered = 0;
        }
        if generated + 1 < job.max_tokens {
            if pending.is_empty() {
                if let Some(draft) = draft.as_deref_mut() {
                    pending.extend(draft.verify(
                        lib,
                        runtime,
                        pass,
                        requests,
                        transport,
                        lease,
                        next,
                        job.max_tokens - generated - 1,
                        job,
                    )?);
                } else {
                    pending.push_back(step(
                        lib,
                        runtime,
                        pass,
                        requests,
                        transport,
                        lease,
                        &[next],
                        ExpertV2SourceKind::Decode,
                        job,
                        None,
                    )?);
                }
            }
            next = pending
                .pop_front()
                .context("generation produced no next token")?;
        }
    }
    if buffered > 0 {
        job.events.blocking_send(Ok(InferenceChunk::Text {
            content: String::new(),
            content_tokens: buffered,
        }))?;
    }
    job.events.blocking_send(Ok(InferenceChunk::Finish {
        finish_reason: InferenceFinishReason::Length,
    }))?;
    Ok(())
}
fn prefill<'w, 'a>(lib: &'a NativeLibrary, runtime: &tokio::runtime::Runtime,
    pass: &mut TargetPass<'w, 'a>, other: &mut TargetPass<'w, 'a>, requests: &mut Requests<'a>,
    transport: &mut NativeTp4Wave<'a>, other_transport: &mut NativeTp4Wave<'a>,
    lease: crate::v41_backbone_cache::CacheLease, tokens: &[u32], chunk_rows: usize,
    job: &NativeRequest, draft: Option<&mut DraftRuntime<'_, 'a>>) -> Result<u32> {
    use crate::v41_block::EncoderSuffix;
    use crate::v41_backbone_cache::CacheWork;
    let end = tokens.len() as u64;
    let mut suffix = EncoderSuffix::new(lib, end, EncoderSuffix::device_bytes(end)?)?;
    requests.begin_encoder(lease, end)?;
    let mut chunks = tokens.chunks(chunk_rows);
    // Keep the ordinary path for short prompts; pair full chunks first otherwise.
    if chunks.len() == 1 {
        let chunk = chunks.next().expect("one chunk");
        ensure!(!job.events.is_closed(), "client disconnected");
        let mut batch = requests.prepare(&[RequestTokens { lease, tokens: chunk,
            image_mask: None, kind: ExpertV2SourceKind::Prefill }])?;
        let started = Instant::now();
        let result = (|| -> Result<()> {
            runtime.block_on(unsafe { pass.execute_encoder(requests, &mut batch, transport, 0, &mut suffix) })?;
            ensure!(!job.events.is_closed(), "client disconnected");
            pass.commit(requests, &mut batch, &[chunk.len() as u32])
        })();
        if result.is_err() { pass.discard(&mut batch)?; }
        result?;
        tracing::debug!(target: "ds41rt::timing", rows=chunk.len(), total_us=started.elapsed().as_micros() as u64, "target encoder step");
    }
    if chunks.len() != 0 {
        let chunks: Vec<_> = chunks.collect();
        let started = Instant::now();
        runtime.block_on(unsafe { pass.execute_encoder_stream(other, requests, lease, &chunks,
            [transport, other_transport], &mut suffix, &|| !job.events.is_closed()) })?;
        tracing::debug!(target: "ds41rt::timing", rows=tokens.len(),
            total_us=started.elapsed().as_micros() as u64, "target encoder stream");
    }
    ensure!(!job.events.is_closed(), "client disconnected");
    let start = requests.begin_decoder_replay(lease)?;
    let rows = (end - start) as u32;
    let mut batch = requests.prepare_replay(&[CacheWork { lease, tokens: rows, kind: ExpertV2SourceKind::Prefill }])?;
    let started = Instant::now();
    let result = (|| -> Result<u32> {
        let encoder = suffix.output()?;
        let logits = runtime.block_on(unsafe { pass.execute_replay(requests, &mut batch,
            transport, 0, &[rows as usize - 1], &encoder) })?;
        let mut bytes = vec![0; logits.logits.bytes]; lib.copy_d2h(&mut bytes, logits.logits)?;
        let mut best = (0u32, f32::NEG_INFINITY);
        for (i, b) in bytes.chunks_exact(4).enumerate() {
            let value = f32::from_ne_bytes(b.try_into().unwrap());
            ensure!(value.is_finite(), "non-finite target logit");
            if value > best.1 { best = (i as u32, value); }
        }
        ensure!(!job.events.is_closed(), "client disconnected");
        if let Some(draft) = draft { draft.commit(pass, requests, &mut batch, rows)?; }
        else { pass.commit(requests, &mut batch, &[rows])?; }
        Ok(best.0)
    })();
    if result.is_err() { pass.discard(&mut batch)?; }
    tracing::debug!(target: "ds41rt::timing", rows, total_us=started.elapsed().as_micros() as u64, "target decoder replay");
    result
}

fn step<'a>(
    lib: &'a NativeLibrary,
    runtime: &tokio::runtime::Runtime,
    pass: &mut TargetPass<'_, 'a>,
    requests: &mut Requests<'a>,
    transport: &mut NativeTp4Wave<'a>,
    lease: crate::v41_backbone_cache::CacheLease,
    tokens: &[u32],
    kind: ExpertV2SourceKind,
    job: &NativeRequest,
    draft: Option<&mut DraftRuntime<'_, 'a>>,
) -> Result<u32> {
    ensure!(!job.events.is_closed(), "client disconnected");
    let timing = Instant::now();
    let mut batch = requests.prepare(&[RequestTokens {
        lease,
        tokens,
        image_mask: None,
        kind,
    }])?;
    let prepared_us = timing.elapsed().as_micros() as u64;
    let result = (|| -> Result<u32> {
        let logits = runtime.block_on(unsafe {
            pass.execute(requests, &mut batch, transport, 0, &[tokens.len() - 1])
        })?;
        let executed_us = timing.elapsed().as_micros() as u64;
        let mut bytes = vec![0; logits.logits.bytes];
        lib.copy_d2h(&mut bytes, logits.logits)?;
        let mut best = (0u32, f32::NEG_INFINITY);
        for (i, b) in bytes.chunks_exact(4).enumerate() {
            let v = f32::from_ne_bytes(b.try_into().unwrap());
            ensure!(v.is_finite(), "non-finite target logit");
            if v > best.1 {
                best = (i as u32, v);
            }
        }
        ensure!(!job.events.is_closed(), "client disconnected");
        let sampled_us = timing.elapsed().as_micros() as u64;
        if let Some(draft) = draft {
            draft.commit(pass, requests, &mut batch, tokens.len() as u32)?;
        } else {
            pass.commit(requests, &mut batch, &[tokens.len() as u32])?;
        }
        tracing::debug!(target: "ds41rt::timing", rows=tokens.len(), prepared_us, execute_us=executed_us-prepared_us, sample_us=sampled_us-executed_us, commit_us=timing.elapsed().as_micros() as u64-sampled_us, total_us=timing.elapsed().as_micros() as u64, "target step");
        Ok(best.0)
    })();
    if result.is_err() {
        pass.discard(&mut batch)?;
    }
    result
}
