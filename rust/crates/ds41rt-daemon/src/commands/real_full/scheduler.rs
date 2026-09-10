use anyhow::{Context, Result};
use ds41rt_core::{
    DType, DecodeStep, ExpertBatch, GraphBucket, LayerWave, LayerWaveMode, ModelFacts,
    MtpVerifyBlock, PositionId, PrefillChunk, Priority,
};

use super::constants::{
    REAL_FULL_PREFLIGHT_DECODE_POSITION, REAL_FULL_PREFLIGHT_DECODE_ROWS,
    REAL_FULL_PREFLIGHT_KV_RESERVATION_ID, REAL_FULL_PREFLIGHT_MTP_ROWS,
    REAL_FULL_PREFLIGHT_MTP_TOKEN_START, REAL_FULL_PREFLIGHT_PREFILL_ROWS,
    REAL_FULL_PREFLIGHT_PREFILL_TOKEN_START, REAL_FULL_PREFLIGHT_REQUEST_ID,
    REAL_FULL_PREFLIGHT_SEQUENCE_ID,
};
use super::types::{
    RealFullExpertBatchDryRun, RealFullLayerSchedulerDryRun, RealFullSchedulerDryRun,
    RealFullWaveDryRun,
};
#[cfg(test)]
pub(super) use execution::real_full_scheduler_execution_for_shape;
pub(super) use execution::{
    load_real_full_kv_snapshot,
    real_full_scheduler_execution_for_batched_shapes_with_shared_sparse_tcp_and_state_device_hidden,
    real_full_scheduler_execution_for_shape_with_shared_sparse_tcp_and_state_device_hidden,
    save_real_full_kv_snapshot, scheduler_prefill_chunk_count_for_rows, RealFullKvSnapshot,
    RealFullSchedulerBatchedInput, RealFullSchedulerDeviceExecution,
    RealFullSchedulerDsparkTp4PendingDispatch, RealFullSchedulerExecutionShape,
    RealFullSchedulerExecutionState, RealFullSchedulerNativeTargetContext,
    RealFullSchedulerNativeTargetIdentity, RealFullSchedulerSparseDispatchTransport,
    RealFullSchedulerSparseTcpDispatchWorker, RealFullSchedulerTargetHiddenTaps,
};
use protocol_v2::real_full_protocol_v2_batch_probe;
pub(super) use protocol_v2::RealFullSchedulerSparseTcpDispatchProbe;

mod execution;
mod protocol_v2;

pub(super) fn real_full_scheduler_dry_run(
    catalog_hash: &str,
    facts: &ModelFacts,
) -> Result<RealFullSchedulerDryRun> {
    anyhow::ensure!(
        facts.model_type == "deepseek_v4",
        "real-full scheduler dry-run requires a DeepSeek V4 catalog, got {}",
        facts.model_type,
    );
    let placement_version = format!("catalog-{}", &catalog_hash[..16]);
    let graph_bucket = GraphBucket::new(
        REAL_FULL_PREFLIGHT_PREFILL_ROWS
            + REAL_FULL_PREFLIGHT_MTP_ROWS
            + REAL_FULL_PREFLIGHT_DECODE_ROWS,
    );
    let quantization_recipe = facts.quantization_recipe.clone();
    let mut layer_dry_runs = Vec::with_capacity(facts.num_hidden_layers);
    let mut decode_prefix_read_layers = 0_usize;
    let mut decode_kv_write_layers = 0_usize;
    let mut prefill_prefix_read_layers = 0_usize;
    let mut prefill_kv_write_layers = 0_usize;
    let mut mtp_prefix_read_layers = 0_usize;
    let mut mtp_tentative_write_records = 0_usize;
    let mut sparse_expert_batches = 0_usize;
    let mut rows_per_sparse_expert_batch = 0_usize;
    let mut routes_per_sparse_expert_batch = 0_usize;
    let mut protocol_v2_batch_probe = None;

    for layer_id in 0..facts.num_hidden_layers {
        let decode = real_full_decode_wave(layer_id, &placement_version, facts);
        let prefill = real_full_prefill_wave(layer_id, &placement_version, facts);
        let mtp_verify = real_full_mtp_wave(layer_id, &placement_version, facts);

        decode_prefix_read_layers += usize::from(!decode.kv_reads.is_empty());
        decode_kv_write_layers += usize::from(!decode.kv_writes.is_empty());
        prefill_prefix_read_layers += usize::from(!prefill.kv_reads.is_empty());
        prefill_kv_write_layers += usize::from(!prefill.kv_writes.is_empty());
        mtp_prefix_read_layers += usize::from(!mtp_verify.kv_reads.is_empty());
        mtp_tentative_write_records += mtp_verify.tentative_kv_writes.len();

        let expert_batch = if layer_id >= facts.first_k_dense_replace {
            let mut batch = ExpertBatch::from_wave_with_envelope(
                &prefill,
                DType::Bf16,
                quantization_recipe.clone(),
                graph_bucket,
            )
            .with_context(|| format!("building prefill ExpertBatch for layer {layer_id}"))?;
            batch
                .try_append_wave(&mtp_verify, DType::Bf16, quantization_recipe.clone())
                .with_context(|| format!("appending MTP ExpertBatch rows for layer {layer_id}"))?;
            batch
                .try_append_wave(&decode, DType::Bf16, quantization_recipe.clone())
                .with_context(|| {
                    format!("appending decode ExpertBatch rows for layer {layer_id}")
                })?;
            sparse_expert_batches += 1;
            rows_per_sparse_expert_batch = batch.num_rows();
            routes_per_sparse_expert_batch = batch.route_count();
            if protocol_v2_batch_probe.is_none() {
                protocol_v2_batch_probe =
                    Some(real_full_protocol_v2_batch_probe(layer_id, &batch)?);
            }
            Some(RealFullExpertBatchDryRun {
                rows: batch.num_rows(),
                routes: batch.route_count(),
                graph_bucket_rows: batch.graph_bucket.row_capacity,
                source_modes: vec!["prefill_chunk", "mtp_verify", "decode_step"],
                hidden_dim: batch.hidden_dim,
                hidden_bytes_per_row: batch.hidden_bytes_per_row,
            })
        } else {
            None
        };

        layer_dry_runs.push(RealFullLayerSchedulerDryRun {
            layer_id,
            layer_kind: real_full_layer_kind(layer_id, facts.first_k_dense_replace),
            decode: real_full_wave_dry_run(&decode),
            prefill: real_full_wave_dry_run(&prefill),
            mtp_verify: real_full_wave_dry_run(&mtp_verify),
            expert_batch,
        });
    }

    Ok(RealFullSchedulerDryRun {
        status: "dry-run-only",
        scope: "construct catalog-shaped LayerWave and mixed ExpertBatch records for DeepSeek V4",
        placement_version,
        request_id: REAL_FULL_PREFLIGHT_REQUEST_ID,
        sequence_id: REAL_FULL_PREFLIGHT_SEQUENCE_ID,
        kv_reservation_id: REAL_FULL_PREFLIGHT_KV_RESERVATION_ID,
        total_layerwaves: facts.num_hidden_layers * 3,
        decode_layerwaves: facts.num_hidden_layers,
        prefill_layerwaves: facts.num_hidden_layers,
        mtp_verify_layerwaves: facts.num_hidden_layers,
        dense_coordinator_layers: facts.first_k_dense_replace,
        sparse_expert_batches,
        graph_bucket_rows: graph_bucket.row_capacity,
        rows_per_sparse_expert_batch,
        routes_per_sparse_expert_batch,
        decode_prefix_read_layers,
        decode_kv_write_layers,
        prefill_prefix_read_layers,
        prefill_kv_write_layers,
        mtp_prefix_read_layers,
        mtp_tentative_write_records,
        protocol_v2_batch_probe: protocol_v2_batch_probe
            .expect("real-full scheduler dry-run has at least one sparse layer"),
        layer_dry_runs,
    })
}

fn real_full_decode_wave(
    layer_id: usize,
    placement_version: &str,
    facts: &ModelFacts,
) -> LayerWave {
    LayerWave::decode_with_model(
        DecodeStep::new(
            REAL_FULL_PREFLIGHT_REQUEST_ID,
            REAL_FULL_PREFLIGHT_SEQUENCE_ID,
            layer_id as u32,
            PositionId(REAL_FULL_PREFLIGHT_DECODE_POSITION),
            Some(REAL_FULL_PREFLIGHT_KV_RESERVATION_ID),
            Priority(0),
            placement_version.to_owned(),
        ),
        facts,
    )
}

fn real_full_prefill_wave(
    layer_id: usize,
    placement_version: &str,
    facts: &ModelFacts,
) -> LayerWave {
    LayerWave::prefill_with_model(
        PrefillChunk::new_with_model(
            REAL_FULL_PREFLIGHT_REQUEST_ID,
            REAL_FULL_PREFLIGHT_SEQUENCE_ID,
            layer_id as u32,
            PositionId(REAL_FULL_PREFLIGHT_PREFILL_TOKEN_START),
            REAL_FULL_PREFLIGHT_PREFILL_ROWS,
            REAL_FULL_PREFLIGHT_KV_RESERVATION_ID,
            Priority(1),
            GraphBucket::new(REAL_FULL_PREFLIGHT_PREFILL_ROWS),
            placement_version.to_owned(),
            facts,
        ),
        facts,
    )
}

fn real_full_mtp_wave(layer_id: usize, placement_version: &str, facts: &ModelFacts) -> LayerWave {
    LayerWave::mtp_verify_with_model(
        MtpVerifyBlock::new(
            REAL_FULL_PREFLIGHT_REQUEST_ID,
            REAL_FULL_PREFLIGHT_SEQUENCE_ID,
            layer_id as u32,
            PositionId(REAL_FULL_PREFLIGHT_MTP_TOKEN_START),
            REAL_FULL_PREFLIGHT_MTP_ROWS,
            Some(REAL_FULL_PREFLIGHT_KV_RESERVATION_ID),
            Priority(0),
            GraphBucket::new(REAL_FULL_PREFLIGHT_MTP_ROWS),
            placement_version.to_owned(),
        ),
        facts,
    )
}

fn real_full_wave_dry_run(wave: &LayerWave) -> RealFullWaveDryRun {
    RealFullWaveDryRun {
        mode: real_full_wave_mode_label(wave.mode),
        rows: wave.num_rows(),
        graph_bucket_rows: wave.graph_bucket.row_capacity,
        payload_bytes: wave.payload_bytes_per_direction(),
        kv_reads: wave.kv_reads.len(),
        kv_writes: wave.kv_writes.len(),
        tentative_kv_writes: wave.tentative_kv_writes.len(),
    }
}

fn real_full_wave_mode_label(mode: LayerWaveMode) -> &'static str {
    match mode {
        LayerWaveMode::Decode => "decode",
        LayerWaveMode::Prefill => "prefill",
        LayerWaveMode::MtpVerify => "mtp_verify",
        LayerWaveMode::Benchmark => "benchmark",
    }
}

fn real_full_layer_kind(layer_id: usize, first_sparse_layer: usize) -> &'static str {
    if layer_id < first_sparse_layer {
        "dense-mlp"
    } else {
        "sparse-routed-moe"
    }
}

#[cfg(test)]
mod tests {
    use super::real_full_scheduler_dry_run;
    use crate::commands::real_full::constants::{
        REAL_FULL_PREFLIGHT_DECODE_ROWS, REAL_FULL_PREFLIGHT_MTP_ROWS,
        REAL_FULL_PREFLIGHT_PREFILL_ROWS,
    };
    use ds41rt_core::{
        ModelFacts, DS4_PRO_HIDDEN_SIZE, DS4_PRO_MOE_INTERMEDIATE_SIZE, DS4_PRO_NUM_HIDDEN_LAYERS,
        DS4_PRO_ROUTED_EXPERTS,
    };
    use ds41rt_loader::DEEPSEEK_V4_EXL3_RECIPE;

    #[test]
    fn scheduler_dry_run_uses_pro_catalog_geometry() {
        let mut facts = ModelFacts::default();
        facts.hidden_size = DS4_PRO_HIDDEN_SIZE;
        facts.num_hidden_layers = DS4_PRO_NUM_HIDDEN_LAYERS;
        facts.routed_experts = DS4_PRO_ROUTED_EXPERTS;
        facts.moe_intermediate_size = DS4_PRO_MOE_INTERMEDIATE_SIZE;
        facts.quantization_recipe = DEEPSEEK_V4_EXL3_RECIPE.to_owned();

        let report = real_full_scheduler_dry_run(&"a".repeat(64), &facts).unwrap();
        let rows = REAL_FULL_PREFLIGHT_PREFILL_ROWS
            + REAL_FULL_PREFLIGHT_MTP_ROWS
            + REAL_FULL_PREFLIGHT_DECODE_ROWS;
        assert_eq!(report.decode_layerwaves, DS4_PRO_NUM_HIDDEN_LAYERS);
        assert_eq!(report.prefill_layerwaves, DS4_PRO_NUM_HIDDEN_LAYERS);
        assert_eq!(report.mtp_verify_layerwaves, DS4_PRO_NUM_HIDDEN_LAYERS);
        assert_eq!(report.sparse_expert_batches, DS4_PRO_NUM_HIDDEN_LAYERS);
        assert_eq!(report.rows_per_sparse_expert_batch, rows);
        assert_eq!(report.routes_per_sparse_expert_batch, rows * facts.top_k);
        assert_eq!(
            report.protocol_v2_batch_probe.hidden_dim,
            DS4_PRO_HIDDEN_SIZE
        );
        assert_eq!(
            report.protocol_v2_batch_probe.hidden_bytes_per_row,
            DS4_PRO_HIDDEN_SIZE * std::mem::size_of::<u16>(),
        );
        assert!(report.layer_dry_runs.iter().all(|layer| {
            layer
                .expert_batch
                .as_ref()
                .is_some_and(|batch| batch.hidden_dim == DS4_PRO_HIDDEN_SIZE)
        }));
    }
}
