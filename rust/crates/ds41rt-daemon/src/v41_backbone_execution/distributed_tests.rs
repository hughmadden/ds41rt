use super::*;
use crate::v41_backbone_lane::BackboneLaneWeights;
use crate::v41_engram::{
    layer::{EngramGate, EngramLayerWeights},
    EngramDeviceRows,
};
use crate::v41_index_lane::IndexLaneWeights;
use crate::v41_requests::{RequestTokens, Requests};
use crate::v41_target_embedding::TargetEmbeddingWave;
use ds41rt_transport::v41_expert::V41Tp4Roce;
use ds41rt_transport::{ExpertV2SourceKind, TcpTransportConfig};
use std::{
    net::SocketAddr,
    time::{Duration, Instant},
};

#[test]
fn real_layer_zero_executes_embedding_attention_tp4_and_mhc() -> Result<()> {
    let Some(path) = std::env::var_os("DS41RT_LAYER0_LIBRARY") else {
        eprintln!("skip distributed layer test: DS41RT_LAYER0_LIBRARY unset");
        return Ok(());
    };
    let model = std::env::var_os("DS41RT_LAYER0_MODEL").context("DS41RT_LAYER0_MODEL required")?;
    let peers: [SocketAddr; 4] = std::env::var("DS41RT_LAYER0_PEERS")?
        .split(',')
        .map(str::parse)
        .collect::<std::result::Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| anyhow::anyhow!("four test peers required"))?;
    let lib = unsafe { NativeLibrary::load(path)? };
    let catalog = ds41rt_loader::read_official_v41_catalog(
        ds41rt_loader::OFFICIAL_V41_MODEL_ID,
        std::path::Path::new(&model),
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
        "layer fixture official coordinator weights loaded in {:.3}s",
        start.elapsed().as_secs_f64()
    );
    let mut embedding =
        TargetEmbeddingWave::new(&lib, &table, 80, TargetEmbeddingWave::device_bytes(80)?)?;
    let mut lane = BackboneLane::new(
        &weights,
        80,
        BackboneLane::workspace_bytes(&lib, 80)?.into_iter().sum(),
    )?;
    let mut index = IndexLane::new(
        &index_weights,
        80,
        IndexLane::workspace_bytes(&lib, 80)?.into_iter().sum(),
    )?;
    let mut execution = BackboneExecution::new(
        &producers,
        80,
        BackboneExecution::workspace_bytes(&lib, 80)?,
    )?;
    let map = ds41rt_loader::EngramTokenMap::from_file(
        &std::path::Path::new(&model).join("tokenizer.json"),
    )?;
    let pipeline =
        unsafe { ds41rt_loader::EngramPipeline::new(&catalog, map, 80, 2, 8 * 1024 * 1024)? };
    let mut requests = Requests::new(
        &lib,
        pipeline,
        16,
        [16; 4],
        BackboneCache::device_bytes(16, [16; 4])?,
    )?;
    let engram_weights =
        EngramLayerWeights::load(&lib, &catalog, 0, 256 * 1024 * 1024, 16 * 1024 * 1024)?;
    let engram14_weights = EngramLayerWeights::load(&lib, &catalog, 1, 256 * 1024 * 1024, 16 * 1024 * 1024)?;
    let mut gate = EngramGate::new(&engram_weights, 80, 128 * 1024 * 1024)?;
    let mut gate14 = EngramGate::new(&engram14_weights, 80, 128 * 1024 * 1024)?;
    let mut upload = EngramDeviceRows::new(&lib, 80, EngramDeviceRows::device_bytes(80)?)?;
    let roce = V41Tp4Roce::new(
        peers,
        [1, 2, 3, 4],
        80,
        TcpTransportConfig {
            timeout: Duration::from_secs(120),
            max_frame_bytes: 2 * 1024 * 1024,
        },
    )?;
    let mut transport = NativeTp4Wave::new(&lib, roce, NativeTp4Wave::device_bytes(80)?)?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let mut previous = None;
    for cycle in 0..2 {
        execution.restart();
        lane.restart()?;
        index.restart()?;
        let leases = (0..16)
            .map(|slot| requests.admit(slot, 1000 + slot as u64))
            .collect::<Result<Vec<_>>>()?;
        let tokens = (0..80u32)
            .map(|i| (i * 7919 + cycle * 113 + 17) % 129280)
            .collect::<Vec<_>>();
        let work = leases
            .iter()
            .zip(tokens.chunks_exact(5))
            .map(|(&lease, tokens)| RequestTokens {
                lease,
                tokens,
                image_mask: None,
                kind: ExpertV2SourceKind::Prefill,
            })
            .collect::<Vec<_>>();
        if cycle == 1 {
            for &lease in &leases { requests.begin_encoder(lease, 10)?; }
            execution.restart_for(CacheStage::Encoder);
        }
        let mut batch = if cycle == 1 { requests.reserve_encoder(&work)? } else { requests.prepare(&work)? };
        let positions = batch.cache()?.positions();
        unsafe {
            requests.begin_text(&batch, &mut embedding, &mut lane)?;
        }
        let start = Instant::now();
        let mut successor = None;
        if cycle == 1 {
            let prepared = unsafe { requests.prepare_encoder_layer(&batch, &mut execution, &mut lane, &mut index)? };
            // The prepared FFN borrows the lane, but no request/cache owner.
            successor = Some(requests.reserve_encoder(&work)?);
            for &lease in &leases { assert_eq!(requests.cache().committed_end(lease)?, 0); }
            let completed = runtime.block_on(unsafe { prepared.execute(&mut transport, 0, batch.image_mask()) })?;
            unsafe { execution.complete_layer(batch.cache()?, &mut lane, completed)?; }
        } else {
        runtime.block_on(unsafe {
            execution.execute_layer(
                requests.cache(),
                batch.cache()?,
                &mut lane,
                &mut index,
                &mut transport,
                0,
                batch.image_mask(),
            )
        })?;
        }
        let output = lane.output()?;
        assert_eq!(output.layer, 0);
        assert_eq!(output.tokens, positions);
        let mut residual = vec![0; output.residual.bytes];
        lib.copy_d2h(&mut residual, output.residual)?;
        let mut pre = vec![0; output.pre.bytes];
        lib.copy_d2h(&mut pre, output.pre)?;
        assert!(residual
            .chunks_exact(2)
            .all(|b| u16::from_ne_bytes(b.try_into().unwrap()) & 0x7f80 != 0x7f80));
        assert!(pre
            .chunks_exact(4)
            .all(|b| f32::from_ne_bytes(b.try_into().unwrap()).is_finite()));
        if let Some(old) = &previous {
            assert_ne!(
                old, &residual,
                "changed token input reused stale layer output"
            );
        }
        if let Some(dir) = std::env::var_os("DS41RT_LAYER0_OUTPUT") {
            let dir = std::path::PathBuf::from(dir);
            std::fs::create_dir_all(&dir)?;
            std::fs::write(dir.join(format!("layer0-c{cycle}-residual.bin")), &residual)?;
            std::fs::write(dir.join(format!("layer0-c{cycle}-pre.bin")), &pre)?;
        }
        previous = Some(residual);
        lane.advance()?;
        let deadline = Instant::now() + Duration::from_secs(120);
        while !unsafe { requests.poll_engram(&mut batch, &mut upload, &mut gate, &mut lane)? } {
            ensure!(Instant::now() < deadline, "layer-1 engram timed out");
            std::thread::sleep(Duration::from_millis(1));
        }
        let prepared = lane.prepared_input()?;
        assert_eq!(prepared.layer, 1);
        assert_eq!(prepared.tokens, positions);
        let mut gated = vec![0; prepared.residual.bytes];
        lib.copy_d2h(&mut gated, prepared.residual)?;
        assert_ne!(
            gated,
            *previous.as_ref().unwrap(),
            "engram did not update real residual"
        );
        let query = unsafe { lane.begin_prepared()? };
        assert_eq!(query.layer, 1);
        assert_eq!(query.tokens()?, positions);
        let mut rotated = vec![0; query.rotated.bytes];
        lib.copy_d2h(&mut rotated, query.rotated)?;
        for bytes in [&gated, &rotated] {
            assert!(bytes
                .chunks_exact(2)
                .all(|b| u16::from_ne_bytes(b.try_into().unwrap()) & 0x7f80 != 0x7f80));
        }
        if let Some(dir) = std::env::var_os("DS41RT_LAYER0_OUTPUT") {
            let dir = std::path::PathBuf::from(dir);
            std::fs::write(dir.join(format!("layer1-c{cycle}-gated.bin")), &gated)?;
            std::fs::write(dir.join(format!("layer1-c{cycle}-query.bin")), &rotated)?;
        }
        if cycle == 1 {
            for layer in 1..20 {
                if layer != 1 {
                    lane.advance()?;
                    if layer == 14 {
                        let deadline = Instant::now() + Duration::from_secs(120);
                        while !unsafe { requests.poll_engram(&mut batch, &mut upload, &mut gate14, &mut lane)? } {
                            ensure!(Instant::now() < deadline, "encoder layer-14 engram timed out");
                            std::thread::sleep(Duration::from_millis(1));
                        }
                    }
                    unsafe { lane.begin_prepared()?; }
                }
                let prepared = unsafe { requests.prepare_encoder_layer(&batch, &mut execution, &mut lane, &mut index)? };
                let completed = runtime.block_on(unsafe { prepared.execute(&mut transport, 0, batch.image_mask()) })?;
                unsafe { execution.complete_layer(batch.cache()?, &mut lane, completed)?; }
                assert_eq!(lane.output()?.layer, layer);
            }
            lane.advance()?;
            unsafe {
                lane.begin_prepared()?;
                requests.publish_encoder_boundary(&batch, &mut execution, &lane)?;
            }
            requests.commit(&mut batch, &mut execution, &[5; 16])?;
            let mut successor = successor.take().context("queued encoder successor missing")?;
            requests.validate(&successor)?;
            for &lease in &leases { assert_eq!(requests.cache().committed_end(lease)?, 5); }
            execution.restart_for(CacheStage::Encoder);
            lane.restart()?; index.restart()?;
            unsafe { requests.begin_text(&successor, &mut embedding, &mut lane)?; }
            for layer in 0..20 {
                if layer != 0 {
                    lane.advance()?;
                    if layer == 1 || layer == 14 {
                        let current_gate = if layer == 1 { &mut gate } else { &mut gate14 };
                        let deadline = Instant::now() + Duration::from_secs(120);
                        while !unsafe { requests.poll_engram(&mut successor, &mut upload, current_gate, &mut lane)? } {
                            ensure!(Instant::now() < deadline, "successor engram timed out");
                            std::thread::sleep(Duration::from_millis(1));
                        }
                    }
                    unsafe { lane.begin_prepared()?; }
                }
                let prepared = unsafe { requests.prepare_encoder_layer(&successor, &mut execution, &mut lane, &mut index)? };
                let completed = runtime.block_on(unsafe { prepared.execute(&mut transport, 0, successor.image_mask()) })?;
                unsafe { execution.complete_layer(successor.cache()?, &mut lane, completed)?; }
            }
            lane.advance()?;
            unsafe {
                lane.begin_prepared()?;
                requests.publish_encoder_boundary(&successor, &mut execution, &lane)?;
            }
            requests.commit(&mut successor, &mut execution, &[5; 16])?;
            for &lease in &leases {
                assert_eq!(requests.cache().committed_end(lease)?, 10);
                assert_eq!(requests.begin_decoder_replay(lease)?, 0);
                requests.release(lease)?;
            }
            eprintln!("PASS full reserved encoder: 16 requests, two queued chunks, all 20 layers, both Engrams, odd compressor carry, source-20 boundary and decoder readiness");
            continue;

        }
        assert!(
            requests
                .commit(&mut batch, &mut execution, &[5; 16])
                .is_err(),
            "partial model pass committed"
        );
        for &lease in &leases {
            assert!(requests.cache().committed_end(lease).is_err());
        }
        if let Some(successor) = successor { assert!(requests.validate(&successor).is_err()); }

        eprintln!("PASS cycle={cycle} rows=80 requests=16 actual distributed layer0 -> mapped layer1 engram -> prepared query; incomplete commit revoked request histories; elapsed={:.3}s", start.elapsed().as_secs_f64());
    }
    Ok(())
}
