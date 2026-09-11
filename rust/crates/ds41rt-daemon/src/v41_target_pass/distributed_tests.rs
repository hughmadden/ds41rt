use super::*;
use crate::v41_backbone_cache::BackboneCache;
use crate::v41_backbone_execution::CacheProducerWeights;
use crate::v41_backbone_lane::BackboneLaneWeights;
use crate::v41_engram::{
    layer::{EngramGate, EngramLayerWeights},
    EngramDeviceRows,
};
use crate::v41_index_lane::IndexLaneWeights;
use crate::v41_requests::{RequestTokens, Requests};
use crate::v41_target_embedding::TargetEmbeddingWave;
use crate::v41_target_head::{TargetHeadWave, TargetHeadWeights};
use crate::v41_tensors::{NativeRtxTensors, VocabularyHead};
use anyhow::Context;
use ds41rt_ffi::NativeLibrary;
use ds41rt_transport::v41_expert::V41Tp4Tcp;
use ds41rt_transport::{ExpertV2SourceKind, TcpTransportConfig};
use std::{
    net::SocketAddr,
    time::{Duration, Instant},
};

#[test]
fn real_target_prefill_commit_and_decode() -> Result<()> {
    let Some(path) = std::env::var_os("DS41RT_TARGET_PASS_LIBRARY") else {
        eprintln!("skip distributed layer test: DS41RT_TARGET_PASS_LIBRARY unset");
        return Ok(());
    };
    let model = std::env::var_os("DS41RT_TARGET_PASS_MODEL")
        .context("DS41RT_TARGET_PASS_MODEL required")?;
    let peers: [SocketAddr; 4] = std::env::var("DS41RT_TARGET_PASS_PEERS")?
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
    let embedding =
        TargetEmbeddingWave::new(&lib, &table, 80, TargetEmbeddingWave::device_bytes(80)?)?;
    let lane = BackboneLane::new(
        &weights,
        80,
        BackboneLane::workspace_bytes(&lib, 80)?.into_iter().sum(),
    )?;
    let index = IndexLane::new(
        &index_weights,
        80,
        IndexLane::workspace_bytes(&lib, 80)?.into_iter().sum(),
    )?;
    let execution = BackboneExecution::new(
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
    let engram_weights = [0, 1]
        .map(|i| EngramLayerWeights::load(&lib, &catalog, i, 256 * 1024 * 1024, 16 * 1024 * 1024))
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    let gates = [
        EngramGate::new(&engram_weights[0], 80, 128 * 1024 * 1024)?,
        EngramGate::new(&engram_weights[1], 80, 128 * 1024 * 1024)?,
    ];
    let upload = EngramDeviceRows::new(&lib, 80, EngramDeviceRows::device_bytes(80)?)?;
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
        Duration::from_secs(120),
    )?;
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
    let case = std::env::var_os("DS41RT_TARGET_PASS_CASE")
        .map(|path| -> Result<serde_json::Value> {
            Ok(serde_json::from_slice(&std::fs::read(path)?)?)
        })
        .transpose()?;
    let request_count = if case.is_some() { 1 } else { 16 };
    let mut tokens: Vec<u32> = match &case {
        Some(case) => serde_json::from_value(case["token_ids"].clone())?,
        None => (0..80u32).map(|i| (i * 7919 + 17) % 129280).collect(),
    };
    ensure!(
        !tokens.is_empty() && tokens.len() <= 80,
        "fixture input must fit 80 rows"
    );
    let prompt_rows = tokens.len() / request_count;
    let cycles = case
        .as_ref()
        .and_then(|c| c["max_new_tokens"].as_u64())
        .unwrap_or(2) as usize;
    ensure!(
        (1..=128).contains(&cycles),
        "fixture generation must be 1..128 tokens"
    );
    let eos = case.as_ref().and_then(|c| c["eos_id"].as_u64());
    let leases = (0..request_count)
        .map(|slot| requests.admit(slot, 1000 + slot as u64))
        .collect::<Result<Vec<_>>>()?;
    let mut committed = 0u64;
    for cycle in 0..cycles {
        let count = if cycle == 0 { prompt_rows } else { 1 };
        let kind = if cycle == 0 {
            ExpertV2SourceKind::Prefill
        } else {
            ExpertV2SourceKind::Decode
        };
        let work = leases
            .iter()
            .zip(tokens.chunks_exact(count))
            .map(|(&lease, tokens)| RequestTokens {
                lease,
                tokens,
                image_mask: None,
                kind,
            })
            .collect::<Vec<_>>();
        let mut batch = requests.prepare(&work)?;
        let selected = (0..request_count)
            .map(|i| (i + 1) * count - 1)
            .collect::<Vec<_>>();
        let start = Instant::now();
        let logits = runtime.block_on(unsafe {
            pass.execute(&requests, &mut batch, &mut transport, 0, &selected)
        })?;
        assert_eq!(logits.rows, request_count);
        assert_eq!(logits.selected_rows, selected);
        assert!(logits
            .token_positions
            .iter()
            .all(|&p| p == committed + count as u64 - 1));
        let mut bytes = vec![0; logits.logits.bytes];
        lib.copy_d2h(&mut bytes, logits.logits)?;
        let values = bytes
            .chunks_exact(4)
            .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert!(values.iter().all(|v| v.is_finite()));
        tokens = values
            .chunks_exact(129280)
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0 as u32
            })
            .collect();
        if let Some(dir) = std::env::var_os("DS41RT_TARGET_PASS_OUTPUT") {
            let dir = std::path::PathBuf::from(dir);
            std::fs::create_dir_all(&dir)?;
            std::fs::write(dir.join(format!("cycle{cycle}-logits.bin")), &bytes)?;
            std::fs::write(
                dir.join(format!("cycle{cycle}-greedy.json")),
                serde_json::to_vec(&tokens)?,
            )?;
        }
        pass.commit(
            &mut requests,
            &mut batch,
            &vec![count as u32; request_count],
        )?;
        committed += count as u64;
        assert!(pass.output(&batch).is_err());
        for &lease in &leases {
            assert_eq!(requests.cache().committed_end(lease)?, committed);
        }
        eprintln!("PASS full target cycle={cycle} requests={request_count} input_rows={} finite_logits=true cache_and_engram_commit=true seconds={:.3}",count*request_count,start.elapsed().as_secs_f64());
        if request_count == 1 && eos.is_some_and(|eos| u64::from(tokens[0]) == eos) {
            break;
        }
    }
    for lease in leases {
        requests.release(lease)?;
    }
    Ok(())
}
