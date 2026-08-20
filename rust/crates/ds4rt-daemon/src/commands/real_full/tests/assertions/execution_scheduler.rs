use ds4rt_core::{
    DS4_EXPERT_TP_WORLD_SIZE, DS4_FLASH_HIDDEN_BF16_BYTES, DS4_FLASH_NUM_HIDDEN_LAYERS,
    DS4_FLASH_TOP_K,
};

use super::super::super::constants::{
    REAL_FULL_PREFLIGHT_DECODE_ROWS, REAL_FULL_PREFLIGHT_MTP_ROWS,
    REAL_FULL_PREFLIGHT_PREFILL_ROWS, REAL_FULL_PREFLIGHT_PREFILL_TOKEN_START,
};
use super::super::super::scheduler::scheduler_prefill_chunk_count_for_rows;
use super::super::super::types::RealDs4FullPreflightReport;

pub(super) fn assert_execution_scheduler_report(report: &RealDs4FullPreflightReport) {
    assert_eq!(report.status, "blocked");
    assert_eq!(report.expected_facts.model_id, report.model_facts.model_id);
    assert_eq!(report.expected_facts.variant, report.model_facts.variant);
    assert_eq!(
        report.expected_facts.hidden_size,
        report.model_facts.hidden_size
    );
    assert_eq!(
        report.expected_facts.num_hidden_layers,
        report.model_facts.num_hidden_layers
    );
    assert_eq!(
        report.expected_facts.routed_experts,
        report.model_facts.routed_experts
    );
    assert_eq!(report.expected_facts.top_k, report.model_facts.top_k);
    assert_eq!(
        report.expected_facts.quantization_recipe,
        report.model_facts.quantization_recipe
    );

    assert_eq!(
        report
            .full_model_tensor_coverage
            .hidden_layers_with_any_tensor,
        DS4_FLASH_NUM_HIDDEN_LAYERS
    );
    assert_eq!(
        report
            .full_model_tensor_coverage
            .sparse_layers_with_routed_experts,
        DS4_FLASH_NUM_HIDDEN_LAYERS
    );
    assert_requirement(report, "full_tensor_catalog_coverage", true);
    assert_requirement(report, "full_model_execution_plan_available", true);
    assert_requirement(
        report,
        "coordinator_startup_resident_preload_plan_available",
        true,
    );
    assert_requirement(report, "full_model_scheduler_dry_run_available", true);
    assert_requirement(
        report,
        "full_model_admitted_scheduler_execution_dry_run_available",
        false,
    );
    assert_requirement(
        report,
        "scheduler_numeric_progression_self_test_available",
        false,
    );

    let resident = &report.coordinator_resident_preload;
    assert_eq!(resident.status, "planned");
    assert!(resident.startup_required);
    assert!(resident.uses_named_resident_buffers);
    assert_eq!(resident.loaded_tensor_bytes, 0);
    assert!(resident.selected_tensor_count > 0);
    assert!(resident.selected_tensor_bytes > 0);
    assert_eq!(resident.role_counts.get("Embedding").copied(), Some(1));
    assert_eq!(resident.role_counts.get("LmHead").copied(), Some(1));
    assert_eq!(resident.role_counts.get("RoutedExpert"), None);
    assert!(resident.skipped_routed_expert_tensors > 0);

    let scheduler = &report.scheduler_dry_run;
    let mixed_rows = REAL_FULL_PREFLIGHT_PREFILL_ROWS
        + REAL_FULL_PREFLIGHT_DECODE_ROWS
        + REAL_FULL_PREFLIGHT_MTP_ROWS;
    assert_eq!(scheduler.total_layerwaves, DS4_FLASH_NUM_HIDDEN_LAYERS * 3);
    assert_eq!(scheduler.sparse_expert_batches, DS4_FLASH_NUM_HIDDEN_LAYERS);
    assert_eq!(scheduler.rows_per_sparse_expert_batch, mixed_rows);
    assert_eq!(
        scheduler.routes_per_sparse_expert_batch,
        mixed_rows * DS4_FLASH_TOP_K
    );
    assert_eq!(
        scheduler.protocol_v2_batch_probe.host_batches,
        DS4_EXPERT_TP_WORLD_SIZE
    );
    assert_eq!(
        scheduler.protocol_v2_batch_probe.host_batch_rows,
        mixed_rows * DS4_EXPERT_TP_WORLD_SIZE
    );
    assert_eq!(
        scheduler.protocol_v2_batch_probe.host_request_payload_bytes,
        scheduler.protocol_v2_batch_probe.host_batch_rows * DS4_FLASH_HIDDEN_BF16_BYTES
    );
    assert!(
        scheduler
            .protocol_v2_batch_probe
            .host_batch_routes_match_global
    );
    assert!(
        scheduler
            .protocol_v2_batch_probe
            .host_batch_graph_counts_valid
    );
    assert!(scheduler.protocol_v2_batch_probe.host_wire_envelopes_valid);
    assert!(scheduler.protocol_v2_batch_probe.passed);

    let execution = &report.scheduler_execution_dry_run;
    let prefill_tokens =
        REAL_FULL_PREFLIGHT_PREFILL_TOKEN_START as usize + REAL_FULL_PREFLIGHT_PREFILL_ROWS;
    let prefill_chunks = scheduler_prefill_chunk_count_for_rows(
        prefill_tokens,
        REAL_FULL_PREFLIGHT_PREFILL_ROWS,
        REAL_FULL_PREFLIGHT_DECODE_ROWS + REAL_FULL_PREFLIGHT_MTP_ROWS,
    );
    let planned_rows = DS4_FLASH_NUM_HIDDEN_LAYERS
        * (prefill_tokens + REAL_FULL_PREFLIGHT_DECODE_ROWS + REAL_FULL_PREFLIGHT_MTP_ROWS);
    assert_eq!(
        execution.status,
        "not-run-preflight-uses-structural-deepseek-plan"
    );
    assert_eq!(execution.request_prefill_tokens, prefill_tokens);
    assert_eq!(execution.request_prefill_chunks, prefill_chunks);
    assert_eq!(
        execution.sparse_expert_batches,
        DS4_FLASH_NUM_HIDDEN_LAYERS * prefill_chunks
    );
    assert_eq!(execution.sparse_expert_batch_rows, planned_rows);
    assert_eq!(
        execution.sparse_expert_batch_routes,
        planned_rows * DS4_FLASH_TOP_K
    );
    assert_eq!(execution.iterations, 0);
    assert_eq!(execution.selected_layerwaves, 0);
    assert_eq!(execution.device_kv_status, "not-run");
    assert_eq!(execution.device_attention_status, "not-run");
    assert!(!execution.uses_device_kv_cache);
    assert!(!execution.uses_device_kv_attention);
    assert!(!execution.full_context_device_attention_complete);
    assert!(!execution.layer_order_verified);

    let progression = &execution.numeric_progression_self_test;
    assert_eq!(progression.status, "not-run");
    assert_eq!(progression.residual_dtype, "bf16");
    assert_eq!(
        progression.source_modes,
        ["prefill", "decode", "dspark-target-verify"]
    );
    assert!(!progression.passed);

    let terminal = &execution.terminal_lm_head_sample;
    assert_eq!(terminal.status, "not-run");
    assert!(!terminal.uses_final_decode_device_hidden);
    assert!(!terminal.passed);
    assert!(terminal.blocker.is_some());
}

fn assert_requirement(report: &RealDs4FullPreflightReport, name: &str, passed: bool) {
    let requirement = report
        .requirements
        .iter()
        .find(|requirement| requirement.name == name)
        .unwrap_or_else(|| panic!("missing preflight requirement {name}"));
    assert_eq!(
        requirement.passed, passed,
        "unexpected result for {name}: {}",
        requirement.evidence
    );
}
