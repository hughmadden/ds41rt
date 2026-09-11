use super::*;
use crate::v41_backbone_cache::CacheWork;
use crate::v41_backbone_lane::BackboneLaneWeights;
use crate::v41_index_lane::IndexLaneWeights;
use crate::v41_target_embedding::TargetEmbeddingWave;
use ds41rt_transport::v41_expert::V41Tp4Tcp;
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
    let mut bank =
        BackboneCache::new(&lib, 16, [16; 4], BackboneCache::device_bytes(16, [16; 4])?)?;
    let leases = (0..16)
        .map(|slot| bank.begin_request(slot, 1000 + slot as u64))
        .collect::<Result<Vec<_>>>()?;
    let tcp = V41Tp4Tcp::new(
        peers,
        [1, 2, 3, 4],
        80,
        TcpTransportConfig {
            timeout: Duration::from_secs(120),
            max_frame_bytes: 2 * 1024 * 1024,
        },
    )?;
    let mut transport = NativeTp4Wave::new(&lib, tcp, NativeTp4Wave::device_bytes(80)?)?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let mut previous = None;
    for cycle in 0..2 {
        execution.restart();
        lane.restart()?;
        index.restart()?;
        let work = leases
            .iter()
            .map(|&lease| CacheWork {
                lease,
                tokens: 5,
                kind: ExpertV2SourceKind::Prefill,
            })
            .collect::<Vec<_>>();
        let batch = bank.plan(&work)?;
        let positions = batch.positions();
        let tokens = (0..80u32)
            .map(|i| (i * 7919 + cycle * 113 + 17) % 129280)
            .collect::<Vec<_>>();
        let embedded = embedding.execute(&tokens, &positions)?;
        unsafe {
            lane.begin_embedded(&embedded)?;
        }
        let start = Instant::now();
        runtime.block_on(unsafe {
            execution.execute_layer(
                &bank,
                &batch,
                &mut lane,
                &mut index,
                &mut transport,
                0,
                &[0; 80],
            )
        })?;
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
        assert!(
            execution.commit(&mut bank, &batch, &[5; 16]).is_err(),
            "partial model pass committed"
        );
        for &lease in &leases {
            assert_eq!(bank.committed_end(lease)?, 0);
        }
        eprintln!("PASS layer0 cycle={cycle} rows=80 requests=16 actual embedding/query/window/attention/shared/TP4/mHC, finite changed output, partial-pass commit rejected, elapsed={:.3}s",start.elapsed().as_secs_f64());
    }
    bank.release(&leases)?;
    Ok(())
}
