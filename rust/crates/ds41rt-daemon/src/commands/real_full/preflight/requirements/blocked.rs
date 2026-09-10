use super::super::super::types::*;

pub(super) struct BlockedRuntimeRequirementInputs<'a> {
    pub(super) attention_kv_binding_available: bool,
    pub(super) attention_kv_binding_dry_run: &'a RealFullAttentionKvBindingDryRun,
    pub(super) scheduler_real_tensor_catalog_available: bool,
    pub(super) expert_execution_dry_run: &'a RealFullExpertExecutionDryRun,
    pub(super) scheduler_execution_dry_run: &'a RealFullSchedulerExecutionDryRun,
    pub(super) residual_stream_dry_run: &'a RealFullResidualStreamDryRun,
    pub(super) sampling_dry_run: &'a RealFullSamplingDryRun,
    pub(super) sampling_real_lm_head_probe: &'a RealFullSamplingRealLmHeadProbe,
}

pub(super) fn blocked_runtime_requirements(
    inputs: BlockedRuntimeRequirementInputs<'_>,
) -> Vec<RealFullRequirement> {
    let BlockedRuntimeRequirementInputs {
        attention_kv_binding_available,
        attention_kv_binding_dry_run,
        scheduler_real_tensor_catalog_available,
        expert_execution_dry_run,
        scheduler_execution_dry_run,
        residual_stream_dry_run,
        sampling_dry_run,
        sampling_real_lm_head_probe,
    } = inputs;

    vec![
        RealFullRequirement {
            name: "full_residual_stream_execution",
            passed: false,
            evidence: format!(
                "DeepSeek V4 catalog-ordered residual plan covers {} layers, {} attention adds, {} MLP adds, and {} terminal rows; request scheduler dry-run status={} native_attention_status={} native_attention_launches={} uses_device_kv_attention={} full_context_device_attention_complete={} numeric_progression_passed={} terminal_sample_status={} terminal_sample_passed={}",
                residual_stream_dry_run.layer_count,
                residual_stream_dry_run.attention_residual_adds,
                residual_stream_dry_run.mlp_residual_adds,
                residual_stream_dry_run.terminal_rows,
                scheduler_execution_dry_run.status,
                scheduler_execution_dry_run.device_attention_status,
                scheduler_execution_dry_run.device_attention_launches,
                scheduler_execution_dry_run.uses_device_kv_attention,
                scheduler_execution_dry_run.full_context_device_attention_complete,
                scheduler_execution_dry_run.numeric_progression_self_test.passed,
                scheduler_execution_dry_run.terminal_lm_head_sample.status,
                scheduler_execution_dry_run.terminal_lm_head_sample.passed,
            ),
            blocker: Some(REAL_FULL_RESIDUAL_COMPLETION_BLOCKER),
        },
        RealFullRequirement {
            name: "real_attention_kv_backing_storage",
            passed: attention_kv_binding_available,
            evidence: format!(
                "DeepSeek V4 attention/KV binding covers {} layers, {} attention tensors, {} KV bytes/token, {} KV I/O layers, and {} prefix-read blocks",
                attention_kv_binding_dry_run.attention_layers,
                attention_kv_binding_dry_run.attention_tensors,
                attention_kv_binding_dry_run.kv_layer_bytes_sum,
                attention_kv_binding_dry_run.kv_io_layer_count,
                attention_kv_binding_dry_run.kv_io_prefix_read_blocks,
            ),
            blocker: (!attention_kv_binding_available)
                .then_some("DeepSeek V4 attention tensors are not fully bound to the selected KV plan"),
        },
        RealFullRequirement {
            name: "all_layer_real_nvfp4_expert_execution",
            passed: false,
            evidence: format!(
                "source-artifact expert dry-run covers {}/{} routed experts across {} sparse layers with {} planned route entries; this source-format diagnostic does not admit the EXL3 serving artifact",
                expert_execution_dry_run.fully_covered_experts,
                expert_execution_dry_run.expected_routed_experts,
                expert_execution_dry_run.sparse_layers,
                expert_execution_dry_run.planned_route_entries,
            ),
            blocker: Some(
                "live serving requires the selected artifact format to execute every routed row through strict TP4 workers",
            ),
        },
        RealFullRequirement {
            name: "scheduler_integration_for_real_full",
            passed: false,
            evidence: format!(
                "DeepSeek V4 scheduler dry-run has real_tensor_catalog={} layer_order_verified={} sparse_host_batch_sets={} sparse_host_batches={} sparse_rows={} sparse_routes={} routes_match_global={} graph_counts_valid={} wire_envelopes_valid={} coordinator_graphs={}/{} native_attention_complete={} terminal_sample_passed={}",
                scheduler_real_tensor_catalog_available,
                scheduler_execution_dry_run.layer_order_verified,
                scheduler_execution_dry_run.sparse_expert_host_batch_sets,
                scheduler_execution_dry_run.sparse_expert_host_batches,
                scheduler_execution_dry_run.sparse_expert_host_batch_rows,
                scheduler_execution_dry_run.sparse_expert_host_batch_routes,
                scheduler_execution_dry_run.sparse_expert_host_batch_routes_match_global,
                scheduler_execution_dry_run.sparse_expert_host_batch_graph_counts_valid,
                scheduler_execution_dry_run.sparse_expert_host_wire_envelopes_valid,
                scheduler_execution_dry_run.request_coordinator_graph_captured_graphs,
                scheduler_execution_dry_run.request_coordinator_graph_launches,
                scheduler_execution_dry_run.full_context_device_attention_complete,
                scheduler_execution_dry_run.terminal_lm_head_sample.passed,
            ),
            blocker: Some(
                "startup dry-run evidence must not substitute for a live DeepSeek V4 request lifecycle",
            ),
        },
        RealFullRequirement {
            name: "full_vocab_sampling",
            passed: false,
            evidence: format!(
                "lm_head dry-run covers vocabulary={} rows={} chunks={} guarded_probe_status={} guarded_probe_passed={} logits={} request_terminal_status={} request_terminal_passed={}",
                sampling_dry_run.vocab_size,
                sampling_dry_run.sampled_rows,
                sampling_dry_run.chunk_count,
                sampling_real_lm_head_probe.status,
                sampling_real_lm_head_probe.passed,
                sampling_real_lm_head_probe.logits_evaluated,
                scheduler_execution_dry_run.terminal_lm_head_sample.status,
                scheduler_execution_dry_run.terminal_lm_head_sample.passed,
            ),
            blocker: Some(
                "live terminal sampling must consume the final DeepSeek V4 request residual",
            ),
        },
    ]
}
