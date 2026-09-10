use ds41rt_core::DS4_FLASH_NUM_HIDDEN_LAYERS;

const FLASH_INDEXER_LAYERS: usize = 21;

use super::super::super::constants::{
    REAL_FULL_PREFLIGHT_MTP_ACCEPTED_ROWS, REAL_FULL_PREFLIGHT_MTP_ROWS,
};
use super::super::super::coordinator_kernels::coordinator_cuda_reference_kernels_enabled;
use super::super::super::types::RealDs4FullPreflightReport;

pub(super) fn assert_kv_attention_report(report: &RealDs4FullPreflightReport) {
    assert_eq!(
        report.kv_backing_store_dry_run.layer_count,
        DS4_FLASH_NUM_HIDDEN_LAYERS
    );
    assert_eq!(
        report.kv_backing_store_dry_run.bytes_per_model_token,
        49_408
    );
    assert_eq!(
        report.kv_backing_store_dry_run.dsa_layer_bytes_per_token,
        1_280
    );
    assert_eq!(
        report
            .kv_backing_store_dry_run
            .non_dsa_layer_bytes_per_token,
        1_024
    );
    assert_eq!(
        report.kv_backing_store_dry_run.backed_prefill_writes,
        DS4_FLASH_NUM_HIDDEN_LAYERS
    );
    assert_eq!(
        report.kv_backing_store_dry_run.backed_decode_writes,
        DS4_FLASH_NUM_HIDDEN_LAYERS
    );
    assert_eq!(
        report.kv_backing_store_dry_run.committed_mtp_writes,
        DS4_FLASH_NUM_HIDDEN_LAYERS * REAL_FULL_PREFLIGHT_MTP_ACCEPTED_ROWS
    );
    assert_eq!(
        report.kv_backing_store_dry_run.discarded_mtp_writes,
        DS4_FLASH_NUM_HIDDEN_LAYERS
            * (REAL_FULL_PREFLIGHT_MTP_ROWS - REAL_FULL_PREFLIGHT_MTP_ACCEPTED_ROWS)
    );
    assert_eq!(
        report.kv_backing_store_dry_run.backed_bytes_after_discard,
        25_543_936
    );
    assert_eq!(
        report
            .kv_backing_store_dry_run
            .visible_layer0_blocks_at_decode,
        1
    );
    assert_eq!(
        report.attention_kv_io_dry_run.prefix_prefill_wave_writes,
        DS4_FLASH_NUM_HIDDEN_LAYERS
    );
    assert_eq!(
        report
            .attention_kv_io_dry_run
            .later_prefill_prefix_read_blocks,
        DS4_FLASH_NUM_HIDDEN_LAYERS
    );
    assert_eq!(
        report.attention_kv_io_dry_run.decode_prefix_read_blocks,
        DS4_FLASH_NUM_HIDDEN_LAYERS * 2
    );
    assert_eq!(
        report.attention_kv_io_dry_run.mtp_prefix_read_blocks,
        DS4_FLASH_NUM_HIDDEN_LAYERS * 3
    );
    assert_eq!(
        report.attention_kv_io_dry_run.mtp_tentative_wave_writes,
        DS4_FLASH_NUM_HIDDEN_LAYERS * REAL_FULL_PREFLIGHT_MTP_ROWS
    );
    assert_eq!(report.attention_kv_io_dry_run.layer0_decode_read_blocks, 2);
    assert_eq!(report.attention_kv_io_dry_run.layer0_mtp_read_blocks, 3);
    assert!(
        report
            .attention_kv_io_dry_run
            .layerwave_payload_count_mismatch_guard
    );
    assert_attention_device_kv_status(
        report.attention_kv_io_dry_run.device_kv_status,
        report.attention_kv_io_dry_run.uses_device_kv_cache,
    );
    if report.attention_kv_io_dry_run.uses_device_kv_cache {
        assert_eq!(
            report.attention_kv_io_dry_run.device_kv_status,
            "cuda-kv-cache-live-scheduler"
        );
        assert_eq!(
            report.attention_kv_io_dry_run.device_kv_writes,
            DS4_FLASH_NUM_HIDDEN_LAYERS * (3 + REAL_FULL_PREFLIGHT_MTP_ROWS)
        );
        assert_eq!(
            report.attention_kv_io_dry_run.device_kv_reads,
            DS4_FLASH_NUM_HIDDEN_LAYERS * 6
        );
        assert!(report.attention_kv_io_dry_run.device_kv_bytes > 0);
    } else {
        assert_eq!(report.attention_kv_io_dry_run.device_kv_writes, 0);
        assert_eq!(report.attention_kv_io_dry_run.device_kv_reads, 0);
        assert_eq!(report.attention_kv_io_dry_run.device_kv_bytes, 0);
    }
    assert_eq!(
        report.attention_kv_binding_dry_run.attention_layers,
        DS4_FLASH_NUM_HIDDEN_LAYERS
    );
    assert_eq!(
        report.attention_kv_binding_dry_run.attention_tensors,
        DS4_FLASH_NUM_HIDDEN_LAYERS * 13
    );
    assert_eq!(
        report
            .attention_kv_binding_dry_run
            .common_layers_with_required_tensors,
        DS4_FLASH_NUM_HIDDEN_LAYERS
    );
    assert_eq!(
        report.attention_kv_binding_dry_run.dsa_indexer_layers,
        FLASH_INDEXER_LAYERS
    );
    assert_eq!(
        report
            .attention_kv_binding_dry_run
            .dsa_indexer_layers_with_required_tensors,
        FLASH_INDEXER_LAYERS
    );
    assert_eq!(report.attention_kv_binding_dry_run.non_dsa_layers, 22);
    assert_eq!(
        report
            .attention_kv_binding_dry_run
            .non_dsa_layers_without_indexer_tensors,
        22
    );
    assert!(
        report
            .attention_kv_binding_dry_run
            .catalog_dsa_indexer_layers_match_kv_config
    );
    assert_eq!(
        report.attention_kv_binding_dry_run.kv_layer_bytes_sum,
        49_408
    );
    assert_eq!(
        report
            .attention_kv_binding_dry_run
            .dsa_layer_bytes_per_token,
        1_280
    );
    assert_eq!(
        report
            .attention_kv_binding_dry_run
            .non_dsa_layer_bytes_per_token,
        1_024
    );
    assert!(
        report
            .attention_kv_binding_dry_run
            .all_attention_layers_bound_to_kv
    );
    assert_eq!(
        report.attention_kv_binding_dry_run.kv_io_layer_count,
        DS4_FLASH_NUM_HIDDEN_LAYERS
    );
    assert_eq!(
        report.attention_kv_binding_dry_run.kv_io_prefill_writes,
        DS4_FLASH_NUM_HIDDEN_LAYERS * 2
    );
    assert_eq!(
        report.attention_kv_binding_dry_run.kv_io_decode_writes,
        DS4_FLASH_NUM_HIDDEN_LAYERS
    );
    assert!(report.attention_kv_binding_dry_run.kv_io_prefix_read_blocks > 0);
}

fn assert_attention_device_kv_status(status: &str, uses_device_kv_cache: bool) {
    if coordinator_cuda_reference_kernels_enabled() {
        assert_eq!(status, "cuda-kv-cache-live-scheduler");
        assert!(uses_device_kv_cache);
    } else {
        assert!(matches!(
            status,
            "cuda-kv-cache-live-scheduler" | "cuda-kv-cache-unavailable" | "cuda-kv-cache-error"
        ));
    }
}
