use ds4rt_core::{DS4_FLASH_NUM_HIDDEN_LAYERS, DS4_FLASH_ROUTED_EXPERTS, DS4_FLASH_TOP_K};

const DS4_FLASH_FIRST_K_DENSE_REPLACE: usize = 0;

use super::super::super::constants::{
    REAL_FULL_PREFLIGHT_DECODE_ROWS, REAL_FULL_PREFLIGHT_MTP_ROWS,
    REAL_FULL_PREFLIGHT_PREFILL_ROWS, REAL_FULL_PREFLIGHT_PREFILL_TOKEN_START,
};
use super::super::super::scheduler::scheduler_prefill_chunk_count_for_rows;
use super::super::super::types::RealDs4FullPreflightReport;

pub(super) fn assert_expert_execution_report(report: &RealDs4FullPreflightReport) {
    let expected_prefill_chunks = scheduler_prefill_chunk_count_for_rows(
        REAL_FULL_PREFLIGHT_PREFILL_TOKEN_START as usize + REAL_FULL_PREFLIGHT_PREFILL_ROWS,
        REAL_FULL_PREFLIGHT_PREFILL_ROWS,
        REAL_FULL_PREFLIGHT_DECODE_ROWS + REAL_FULL_PREFLIGHT_MTP_ROWS,
    );
    assert_eq!(
        report.expert_execution_dry_run.sparse_layers,
        DS4_FLASH_NUM_HIDDEN_LAYERS - DS4_FLASH_FIRST_K_DENSE_REPLACE
    );
    assert_eq!(
        report.expert_execution_dry_run.expected_routed_experts,
        (DS4_FLASH_NUM_HIDDEN_LAYERS - DS4_FLASH_FIRST_K_DENSE_REPLACE) * DS4_FLASH_ROUTED_EXPERTS
    );
    assert_eq!(
        report.expert_execution_dry_run.fully_covered_experts,
        report.expert_execution_dry_run.expected_routed_experts
    );
    assert_eq!(
        report.expert_execution_dry_run.routed_weight_tensors,
        report.expert_execution_dry_run.expected_routed_experts * 3
    );
    assert_eq!(
        report
            .expert_execution_dry_run
            .routed_quant_metadata_tensors,
        report.expert_execution_dry_run.expected_routed_experts * 9
    );
    assert_eq!(
        report
            .expert_execution_dry_run
            .planned_sparse_expert_batches,
        (DS4_FLASH_NUM_HIDDEN_LAYERS - DS4_FLASH_FIRST_K_DENSE_REPLACE) * expected_prefill_chunks
    );
    assert_eq!(
        report.expert_execution_dry_run.planned_expert_batch_rows,
        44_419
    );
    assert_eq!(
        report.expert_execution_dry_run.planned_route_entries,
        266_514
    );
    assert_eq!(
        report.expert_execution_dry_run.planned_expert_source_modes,
        ["prefill_chunk", "decode_step", "mtp_verify"]
    );
    assert_eq!(
        report.expert_execution_dry_run.planned_prefill_expert_rows,
        (DS4_FLASH_NUM_HIDDEN_LAYERS - DS4_FLASH_FIRST_K_DENSE_REPLACE)
            * (REAL_FULL_PREFLIGHT_PREFILL_TOKEN_START as usize + REAL_FULL_PREFLIGHT_PREFILL_ROWS)
    );
    assert_eq!(
        report.expert_execution_dry_run.planned_decode_expert_rows,
        (DS4_FLASH_NUM_HIDDEN_LAYERS - DS4_FLASH_FIRST_K_DENSE_REPLACE)
            * REAL_FULL_PREFLIGHT_DECODE_ROWS
    );
    assert_eq!(
        report
            .expert_execution_dry_run
            .planned_mtp_verify_expert_rows,
        (DS4_FLASH_NUM_HIDDEN_LAYERS - DS4_FLASH_FIRST_K_DENSE_REPLACE)
            * REAL_FULL_PREFLIGHT_MTP_ROWS
    );
    assert_eq!(
        report
            .expert_execution_dry_run
            .planned_prefill_route_entries,
        report.expert_execution_dry_run.planned_prefill_expert_rows * DS4_FLASH_TOP_K
    );
    assert_eq!(
        report.expert_execution_dry_run.planned_decode_route_entries,
        report.expert_execution_dry_run.planned_decode_expert_rows * DS4_FLASH_TOP_K
    );
    assert_eq!(
        report
            .expert_execution_dry_run
            .planned_mtp_verify_route_entries,
        report
            .expert_execution_dry_run
            .planned_mtp_verify_expert_rows
            * DS4_FLASH_TOP_K
    );
    assert!(report.expert_execution_dry_run.planned_source_modes_covered);
    assert!(
        report
            .expert_execution_dry_run
            .planned_route_entries_match_source_rows
    );
    assert_eq!(report.expert_execution_dry_run.owner_partitions.len(), 4);
    assert!(report
        .expert_execution_dry_run
        .owner_partitions
        .iter()
        .all(|partition| partition.routed_experts == 2_752));
    assert!(
        report
            .expert_execution_dry_run
            .all_sparse_layers_have_all_experts
    );
    assert!(
        report
            .expert_execution_dry_run
            .all_experts_have_quant_metadata
    );
    assert_eq!(
        report
            .expert_execution_dry_run
            .real_nvfp4_numeric_probe
            .status,
        "not-run"
    );
    assert_eq!(
        report
            .expert_execution_dry_run
            .real_nvfp4_numeric_probe
            .opt_in_env,
        "not-applicable"
    );
    assert_eq!(
        report
            .expert_execution_dry_run
            .real_nvfp4_numeric_probe
            .row_mode,
        "superseded"
    );
    assert!(report
        .expert_execution_dry_run
        .real_nvfp4_numeric_probe
        .skipped_reason
        .as_deref()
        .unwrap_or_default()
        .contains("removed one-off probe"));
    assert_eq!(
        report
            .expert_execution_dry_run
            .real_nvfp4_numeric_probe
            .route_count,
        0
    );
    assert!(
        !report
            .expert_execution_dry_run
            .real_nvfp4_numeric_probe
            .uses_real_router
    );
    assert_eq!(
        report
            .expert_execution_dry_run
            .real_nvfp4_numeric_probe
            .residual_prefix_values,
        0
    );
    assert!(
        !report
            .expert_execution_dry_run
            .real_nvfp4_numeric_probe
            .applies_mlp_residual_prefix
    );
    assert!(
        !report
            .expert_execution_dry_run
            .real_nvfp4_numeric_probe
            .passed
    );
    assert!(
        !report
            .expert_execution_dry_run
            .real_nvfp4_numeric_probe
            .covers_all_sparse_layers
    );
    assert_eq!(
        report
            .expert_execution_dry_run
            .real_nvfp4_all_layer_probe
            .status,
        "not-run"
    );
    assert_eq!(
        report
            .expert_execution_dry_run
            .real_nvfp4_all_layer_probe
            .opt_in_env,
        "not-applicable"
    );
    assert!(report
        .expert_execution_dry_run
        .real_nvfp4_all_layer_probe
        .skipped_reason
        .as_deref()
        .unwrap_or_default()
        .contains("removed one-off probe"));
    assert_eq!(
        report
            .expert_execution_dry_run
            .real_nvfp4_all_layer_probe
            .layers_executed,
        0
    );
    assert!(
        !report
            .expert_execution_dry_run
            .real_nvfp4_all_layer_probe
            .covers_all_sparse_layers
    );
    assert!(
        !report
            .expert_execution_dry_run
            .real_nvfp4_all_layer_probe
            .covers_full_top_k
    );
    assert_eq!(
        report
            .expert_execution_dry_run
            .real_nvfp4_residual_chain_probe
            .status,
        "not-run"
    );
    assert_eq!(
        report
            .expert_execution_dry_run
            .real_nvfp4_residual_chain_probe
            .opt_in_env,
        "not-applicable"
    );
    assert!(report
        .expert_execution_dry_run
        .real_nvfp4_residual_chain_probe
        .skipped_reason
        .as_deref()
        .unwrap_or_default()
        .contains("removed one-off probe"));
    assert_eq!(
        report
            .expert_execution_dry_run
            .real_nvfp4_residual_chain_probe
            .layers_executed,
        0
    );
    assert!(
        !report
            .expert_execution_dry_run
            .real_nvfp4_residual_chain_probe
            .covers_all_sparse_layers
    );
    let serialized_expert_report =
        serde_json::to_value(&report.expert_execution_dry_run).expect("serializing expert report");
    assert!(serialized_expert_report
        .get("real_sparse_mlp_shared_chain_probe")
        .is_none());
    assert_eq!(
        report
            .expert_execution_dry_run
            .real_nvfp4_scheduler_rows_probe
            .status,
        "not-run"
    );
    assert_eq!(
        report
            .expert_execution_dry_run
            .real_nvfp4_scheduler_rows_probe
            .opt_in_env,
        "not-applicable"
    );
    assert_eq!(
        report
            .expert_execution_dry_run
            .real_nvfp4_scheduler_rows_probe
            .row_mode,
        "superseded"
    );
    assert_eq!(
        report
            .expert_execution_dry_run
            .real_nvfp4_scheduler_rows_probe
            .output_rows,
        0
    );
    assert!(
        !report
            .expert_execution_dry_run
            .real_nvfp4_scheduler_rows_probe
            .covers_full_output_rows
    );
    assert_eq!(
        report
            .expert_execution_dry_run
            .real_nvfp4_scheduler_rows_probe
            .source_rows_executed,
        0
    );
    assert_eq!(
        report
            .expert_execution_dry_run
            .real_nvfp4_scheduler_rows_probe
            .planned_source_rows,
        REAL_FULL_PREFLIGHT_PREFILL_ROWS
            + REAL_FULL_PREFLIGHT_MTP_ROWS
            + REAL_FULL_PREFLIGHT_DECODE_ROWS
    );
    assert_eq!(
        report
            .expert_execution_dry_run
            .real_nvfp4_scheduler_rows_probe
            .planned_decode_source_rows,
        REAL_FULL_PREFLIGHT_DECODE_ROWS
    );
    assert_eq!(
        report
            .expert_execution_dry_run
            .real_nvfp4_scheduler_rows_probe
            .planned_prefill_source_rows,
        REAL_FULL_PREFLIGHT_PREFILL_ROWS
    );
    assert_eq!(
        report
            .expert_execution_dry_run
            .real_nvfp4_scheduler_rows_probe
            .planned_mtp_verify_source_rows,
        REAL_FULL_PREFLIGHT_MTP_ROWS
    );
    assert_eq!(
        report
            .expert_execution_dry_run
            .real_nvfp4_scheduler_rows_probe
            .executed_decode_source_rows,
        0
    );
    assert_eq!(
        report
            .expert_execution_dry_run
            .real_nvfp4_scheduler_rows_probe
            .executed_prefill_source_rows,
        0
    );
    assert_eq!(
        report
            .expert_execution_dry_run
            .real_nvfp4_scheduler_rows_probe
            .executed_mtp_verify_source_rows,
        0
    );
    assert_eq!(
        report
            .expert_execution_dry_run
            .real_nvfp4_scheduler_rows_probe
            .planned_compatible_batch_rows,
        REAL_FULL_PREFLIGHT_PREFILL_ROWS
            + REAL_FULL_PREFLIGHT_MTP_ROWS
            + REAL_FULL_PREFLIGHT_DECODE_ROWS
    );
    assert!(
        !report
            .expert_execution_dry_run
            .real_nvfp4_scheduler_rows_probe
            .covers_scheduler_source_modes
    );
    assert!(
        !report
            .expert_execution_dry_run
            .real_nvfp4_scheduler_rows_probe
            .passed
    );
    assert!(report
        .expert_execution_dry_run
        .real_nvfp4_scheduler_rows_probe
        .skipped_reason
        .as_deref()
        .is_some_and(|reason| reason.contains("removed one-off probe")));
    assert!(
        !report
            .expert_execution_dry_run
            .numeric_execution_implemented
    );
}
