use std::collections::{BTreeMap, BTreeSet};

use ds41rt_core::{owner_for_expert, ModelFacts, PlacementPolicy, TensorCatalog, TensorRole};

use super::constants::{
    REAL_FULL_PREFLIGHT_DECODE_ROWS, REAL_FULL_PREFLIGHT_MTP_ROWS,
    REAL_FULL_PREFLIGHT_PREFILL_ROWS, REAL_FULL_PREFLIGHT_PREFILL_TOKEN_START,
};
use super::scheduler::scheduler_prefill_chunk_count_for_rows;
use super::types::{
    RealFullExpertAllLayerNvfp4Probe, RealFullExpertExecutionDryRun, RealFullExpertOwnerPartition,
    RealFullExpertRealNvfp4Probe, RealFullExpertResidualChainNvfp4Probe,
    RealFullExpertSchedulerRowsNvfp4Probe,
};

#[derive(Debug, Default, Clone, Copy)]
struct ExpertTensorStats {
    weight_tensors: usize,
    quant_metadata_tensors: usize,
    weight_bytes: u64,
    quant_metadata_bytes: u64,
}

#[derive(Debug, Default)]
struct OwnerPartitionStats {
    sparse_layers: BTreeSet<usize>,
    routed_experts: usize,
    routed_weight_tensors: usize,
    routed_quant_metadata_tensors: usize,
    routed_weight_bytes: u64,
    routed_quant_metadata_bytes: u64,
}

pub(super) fn real_full_expert_execution_dry_run(
    catalog: &TensorCatalog,
    expert_hosts: &[String],
) -> RealFullExpertExecutionDryRun {
    let facts = &catalog.facts;
    let sparse_layers = facts.num_hidden_layers - facts.first_k_dense_replace;
    let expected_routed_experts = sparse_layers * facts.routed_experts;
    let mut stats_by_expert = BTreeMap::<(usize, usize), ExpertTensorStats>::new();
    let mut covered_sparse_layers = BTreeSet::<usize>::new();

    for tensor in &catalog.tensors {
        if tensor.role != TensorRole::RoutedExpert {
            continue;
        }
        let Some(layer_id) = tensor.layer_id.map(|layer_id| layer_id as usize) else {
            continue;
        };
        let Some(expert_id) = tensor.expert_id.map(|expert_id| expert_id as usize) else {
            continue;
        };
        if !(facts.first_k_dense_replace..facts.num_hidden_layers).contains(&layer_id)
            || expert_id >= facts.routed_experts
        {
            continue;
        }

        covered_sparse_layers.insert(layer_id);
        let stats = stats_by_expert.entry((layer_id, expert_id)).or_default();
        if tensor.is_quantization_metadata {
            stats.quant_metadata_tensors += 1;
            stats.quant_metadata_bytes += tensor.byte_length;
        } else {
            stats.weight_tensors += 1;
            stats.weight_bytes += tensor.byte_length;
        }
    }

    let mut experts_with_any_tensor = 0_usize;
    let mut experts_with_weight_tensors = 0_usize;
    let mut experts_with_quant_metadata = 0_usize;
    let mut fully_covered_experts = 0_usize;
    let mut routed_weight_tensors = 0_usize;
    let mut routed_quant_metadata_tensors = 0_usize;
    let mut routed_weight_bytes = 0_u64;
    let mut routed_quant_metadata_bytes = 0_u64;
    let mut owner_stats = BTreeMap::<String, OwnerPartitionStats>::new();

    for layer_id in facts.first_k_dense_replace..facts.num_hidden_layers {
        for expert_id in 0..facts.routed_experts {
            let stats = stats_by_expert
                .get(&(layer_id, expert_id))
                .copied()
                .unwrap_or_default();
            let has_weight = stats.weight_tensors > 0;
            let has_quant_metadata = stats.quant_metadata_tensors > 0;

            experts_with_any_tensor += usize::from(has_weight || has_quant_metadata);
            experts_with_weight_tensors += usize::from(has_weight);
            experts_with_quant_metadata += usize::from(has_quant_metadata);
            fully_covered_experts += usize::from(has_weight && has_quant_metadata);
            routed_weight_tensors += stats.weight_tensors;
            routed_quant_metadata_tensors += stats.quant_metadata_tensors;
            routed_weight_bytes += stats.weight_bytes;
            routed_quant_metadata_bytes += stats.quant_metadata_bytes;

            let owner = owner_for_expert(
                layer_id,
                expert_id,
                catalog.facts.routed_experts,
                expert_hosts,
                PlacementPolicy::Modulo,
            )
            .unwrap_or_else(|| "<unassigned>".to_owned());
            let owner = owner_stats.entry(owner).or_default();
            owner.sparse_layers.insert(layer_id);
            owner.routed_experts += 1;
            owner.routed_weight_tensors += stats.weight_tensors;
            owner.routed_quant_metadata_tensors += stats.quant_metadata_tensors;
            owner.routed_weight_bytes += stats.weight_bytes;
            owner.routed_quant_metadata_bytes += stats.quant_metadata_bytes;
        }
    }

    let mut owner_partitions = Vec::new();
    for host in expert_hosts {
        if let Some(stats) = owner_stats.remove(host) {
            owner_partitions.push(owner_partition(host.clone(), stats));
        }
    }
    owner_partitions.extend(
        owner_stats
            .into_iter()
            .map(|(owner, stats)| owner_partition(owner, stats)),
    );
    let real_nvfp4_numeric_probe = superseded_real_nvfp4_numeric_probe(facts);
    let real_nvfp4_all_layer_probe = superseded_real_nvfp4_all_layer_probe(facts);
    let real_nvfp4_residual_chain_probe = superseded_real_nvfp4_residual_chain_probe(facts);
    let planned_compatible_batch_rows = REAL_FULL_PREFLIGHT_COMPATIBLE_BATCH_ROWS;
    let real_nvfp4_scheduler_rows_probe =
        superseded_real_nvfp4_scheduler_rows_probe(planned_compatible_batch_rows, facts);
    let prefill_rows_per_sparse_layer =
        REAL_FULL_PREFLIGHT_PREFILL_TOKEN_START as usize + REAL_FULL_PREFLIGHT_PREFILL_ROWS;
    let planned_prefill_expert_rows = sparse_layers * prefill_rows_per_sparse_layer;
    let planned_decode_expert_rows = sparse_layers * REAL_FULL_PREFLIGHT_DECODE_ROWS;
    let planned_mtp_verify_expert_rows = sparse_layers * REAL_FULL_PREFLIGHT_MTP_ROWS;
    let planned_prefill_route_entries = planned_prefill_expert_rows * facts.top_k;
    let planned_decode_route_entries = planned_decode_expert_rows * facts.top_k;
    let planned_mtp_verify_route_entries = planned_mtp_verify_expert_rows * facts.top_k;
    let planned_source_row_sum =
        planned_prefill_expert_rows + planned_decode_expert_rows + planned_mtp_verify_expert_rows;
    let planned_source_route_sum = planned_prefill_route_entries
        + planned_decode_route_entries
        + planned_mtp_verify_route_entries;
    let planned_sparse_expert_batches = sparse_layers
        * scheduler_prefill_chunk_count_for_rows(
            prefill_rows_per_sparse_layer,
            REAL_FULL_PREFLIGHT_PREFILL_ROWS,
            REAL_FULL_PREFLIGHT_DECODE_ROWS + REAL_FULL_PREFLIGHT_MTP_ROWS,
        );

    RealFullExpertExecutionDryRun {
        status: "dry-run-only",
        scope: "verify all target-layer routed expert tensors, TP=4 serving geometry, ingestion-source balance, and scheduler ExpertBatch route volume",
        sparse_layers,
        routed_experts_per_layer: facts.routed_experts,
        expected_routed_experts,
        catalog_experts_with_any_tensor: experts_with_any_tensor,
        experts_with_weight_tensors,
        experts_with_quant_metadata,
        fully_covered_experts,
        missing_weight_experts: expected_routed_experts - experts_with_weight_tensors,
        missing_quant_metadata_experts: expected_routed_experts - experts_with_quant_metadata,
        covered_sparse_layers: covered_sparse_layers.len(),
        routed_weight_tensors,
        routed_quant_metadata_tensors,
        routed_weight_bytes,
        routed_quant_metadata_bytes,
        placement_policy:
            "strict-tp4-replicated-experts; modulo-ingestion-and-reduction-owner-metadata",
        owner_partitions,
        planned_sparse_expert_batches,
        planned_expert_batch_rows: planned_source_row_sum,
        planned_route_entries: planned_source_route_sum,
        planned_expert_source_modes: ["prefill_chunk", "decode_step", "mtp_verify"],
        planned_prefill_expert_rows,
        planned_decode_expert_rows,
        planned_mtp_verify_expert_rows,
        planned_prefill_route_entries,
        planned_decode_route_entries,
        planned_mtp_verify_route_entries,
        planned_source_modes_covered: planned_prefill_expert_rows > 0
            && planned_decode_expert_rows > 0
            && planned_mtp_verify_expert_rows > 0,
        planned_route_entries_match_source_rows: true,
        hidden_dim: facts.hidden_size,
        hidden_bytes_per_row: facts.hidden_size * std::mem::size_of::<u16>(),
        logical_hidden_row_bytes: planned_source_row_sum
            * facts.hidden_size * std::mem::size_of::<u16>(),
        logical_partial_output_row_bytes: planned_source_row_sum
            * facts.hidden_size * std::mem::size_of::<u16>(),
        route_entries_per_row: facts.top_k,
        max_touched_hosts_per_batch: expert_hosts.len(),
        all_sparse_layers_have_all_experts: covered_sparse_layers.len() == sparse_layers
            && fully_covered_experts == expected_routed_experts,
        all_experts_have_weight_tensors: experts_with_weight_tensors == expected_routed_experts,
        all_experts_have_quant_metadata: experts_with_quant_metadata == expected_routed_experts,
        real_nvfp4_numeric_probe,
        real_nvfp4_all_layer_probe,
        real_nvfp4_residual_chain_probe,
        real_nvfp4_scheduler_rows_probe,
        numeric_execution_implemented: false,
    }
}

const REAL_FULL_PREFLIGHT_COMPATIBLE_BATCH_ROWS: usize =
    super::constants::REAL_FULL_PREFLIGHT_PREFILL_ROWS
        + super::constants::REAL_FULL_PREFLIGHT_MTP_ROWS
        + super::constants::REAL_FULL_PREFLIGHT_DECODE_ROWS;

fn owner_partition(owner: String, stats: OwnerPartitionStats) -> RealFullExpertOwnerPartition {
    RealFullExpertOwnerPartition {
        owner,
        sparse_layers: stats.sparse_layers.len(),
        routed_experts: stats.routed_experts,
        routed_weight_tensors: stats.routed_weight_tensors,
        routed_quant_metadata_tensors: stats.routed_quant_metadata_tensors,
        routed_weight_bytes: stats.routed_weight_bytes,
        routed_quant_metadata_bytes: stats.routed_quant_metadata_bytes,
    }
}

fn superseded_real_nvfp4_numeric_probe(facts: &ModelFacts) -> RealFullExpertRealNvfp4Probe {
    RealFullExpertRealNvfp4Probe {
        status: "not-run",
        scope: "legacy single-row NVFP4 routed expert probe superseded by the layer-ordered execution stepper and live ProtocolV2 scheduler-row evidence",
        opt_in_env: "not-applicable",
        hidden_source: "not-run",
        quant_recipe: facts.quantization_recipe.clone(),
        layer_id: facts.first_k_dense_replace,
        expert_id: 0,
        hidden_dim: facts.hidden_size,
        top_k: facts.top_k,
        route_count: 0,
        router_weight_bytes_read: 0,
        router_bias_bytes_read: 0,
        row_mode: "superseded",
        intermediate_rows: 0,
        output_rows: 0,
        covers_full_intermediate_rows: false,
        covers_full_output_rows: false,
        weight_tensors_read: 0,
        quant_metadata_tensors_read: 0,
        weight_bytes_read: 0,
        quant_metadata_bytes_read: 0,
        gate_checksum: None,
        up_checksum: None,
        activation_checksum: None,
        output_checksum: None,
        output_l2_norm: None,
        first_output: None,
        last_output: None,
        reduced_output_checksum: None,
        reduced_output_l2_norm: None,
        reduced_first_output: None,
        reduced_last_output: None,
        residual_prefix_values: 0,
        residual_before_checksum: None,
        residual_delta_checksum: None,
        residual_after_checksum: None,
        residual_first_before: None,
        residual_first_after: None,
        residual_last_before: None,
        residual_last_after: None,
        routes: Vec::new(),
        uses_real_nvfp4_weights: false,
        uses_real_router: false,
        applies_mlp_residual_prefix: false,
        uses_full_model_residual: false,
        covers_full_top_k: false,
        covers_all_sparse_layers: false,
        covers_all_experts: false,
        passed: false,
        skipped_reason: Some(
            "removed one-off probe; use layer-ordered execution stepper and real ProtocolV2 scheduler-row coverage".to_owned(),
        ),
    }
}

fn superseded_real_nvfp4_all_layer_probe(facts: &ModelFacts) -> RealFullExpertAllLayerNvfp4Probe {
    RealFullExpertAllLayerNvfp4Probe {
        status: "not-run",
        scope: "legacy all-sparse-layer NVFP4 top-k probe superseded by the layer-ordered execution stepper and live ProtocolV2 scheduler-row evidence",
        opt_in_env: "not-applicable",
        hidden_source: "not-run",
        quant_recipe: facts.quantization_recipe.clone(),
        sparse_layers: facts.num_hidden_layers - facts.first_k_dense_replace,
        layers_executed: 0,
        top_k_per_layer: facts.top_k,
        routes_executed: 0,
        intermediate_rows: 0,
        output_rows: 0,
        router_weight_bytes_read: 0,
        router_bias_bytes_read: 0,
        weight_tensors_read: 0,
        quant_metadata_tensors_read: 0,
        weight_bytes_read: 0,
        quant_metadata_bytes_read: 0,
        output_checksum: None,
        output_l2_norm: None,
        first_layer_id: None,
        last_layer_id: None,
        first_expert_id: None,
        last_expert_id: None,
        layer_summaries: Vec::new(),
        uses_real_nvfp4_weights: false,
        uses_real_router: false,
        covers_all_sparse_layers: false,
        covers_full_top_k: false,
        covers_all_experts: false,
        uses_full_model_residual: false,
        passed: false,
        skipped_reason: Some(
            "removed one-off probe; use layer-ordered execution stepper and real ProtocolV2 scheduler-row coverage".to_owned(),
        ),
    }
}

fn superseded_real_nvfp4_residual_chain_probe(
    facts: &ModelFacts,
) -> RealFullExpertResidualChainNvfp4Probe {
    RealFullExpertResidualChainNvfp4Probe {
        status: "not-run",
        scope: "legacy sparse NVFP4 residual-chain probe superseded by the layer-ordered execution stepper and live ProtocolV2 scheduler-row evidence",
        opt_in_env: "not-applicable",
        hidden_source: "not-run",
        quant_recipe: facts.quantization_recipe.clone(),
        sparse_layers: facts.num_hidden_layers - facts.first_k_dense_replace,
        layers_executed: 0,
        top_k_per_layer: facts.top_k,
        routes_executed: 0,
        intermediate_rows: 0,
        output_rows: 0,
        residual_prefix_values: 0,
        residual_adds: 0,
        router_weight_bytes_read: 0,
        router_bias_bytes_read: 0,
        weight_tensors_read: 0,
        quant_metadata_tensors_read: 0,
        weight_bytes_read: 0,
        quant_metadata_bytes_read: 0,
        output_checksum: None,
        output_l2_norm: None,
        initial_residual_checksum: None,
        residual_delta_checksum: None,
        final_residual_checksum: None,
        first_residual_before: None,
        first_residual_after: None,
        last_residual_before: None,
        last_residual_after: None,
        first_layer_id: None,
        last_layer_id: None,
        first_expert_id: None,
        last_expert_id: None,
        layer_summaries: Vec::new(),
        uses_real_nvfp4_weights: false,
        uses_real_router: false,
        applies_sparse_mlp_residual_chain: false,
        includes_attention: false,
        includes_dense_layers: false,
        includes_shared_expert: false,
        covers_all_sparse_layers: false,
        covers_full_top_k: false,
        uses_full_model_residual: false,
        passed: false,
        skipped_reason: Some(
            "removed one-off probe; use layer-ordered execution stepper and real ProtocolV2 scheduler-row coverage".to_owned(),
        ),
    }
}

fn superseded_real_nvfp4_scheduler_rows_probe(
    planned_compatible_batch_rows: usize,
    facts: &ModelFacts,
) -> RealFullExpertSchedulerRowsNvfp4Probe {
    RealFullExpertSchedulerRowsNvfp4Probe {
        status: "not-run",
        scope: "legacy mixed scheduler-row NVFP4 probe superseded by the layer-ordered execution stepper and live ProtocolV2 expert-daemon coverage",
        opt_in_env: "not-applicable",
        hidden_source: "not-run",
        quant_recipe: facts.quantization_recipe.clone(),
        row_mode: "superseded",
        source_modes: Vec::new(),
        source_rows_executed: 0,
        planned_compatible_batch_rows,
        planned_source_rows: planned_compatible_batch_rows,
        planned_decode_source_rows: REAL_FULL_PREFLIGHT_DECODE_ROWS,
        planned_prefill_source_rows: REAL_FULL_PREFLIGHT_PREFILL_ROWS,
        planned_mtp_verify_source_rows: REAL_FULL_PREFLIGHT_MTP_ROWS,
        executed_decode_source_rows: 0,
        executed_prefill_source_rows: 0,
        executed_mtp_verify_source_rows: 0,
        source_row_window_start: 0,
        source_row_window_end: 0,
        uses_source_row_window: false,
        uses_full_scheduler_rows: false,
        covers_all_scheduler_rows: false,
        sparse_layers: facts.num_hidden_layers - facts.first_k_dense_replace,
        layers_executed: 0,
        top_k_per_layer: facts.top_k,
        routes_executed: 0,
        intermediate_rows: 0,
        output_rows: 0,
        covers_full_output_rows: false,
        residual_prefix_values_per_row: 0,
        residual_adds: 0,
        router_weight_bytes_read: 0,
        router_bias_bytes_read: 0,
        weight_tensors_read: 0,
        quant_metadata_tensors_read: 0,
        weight_bytes_read: 0,
        quant_metadata_bytes_read: 0,
        router_cache_entries: 0,
        router_cache_hits: 0,
        router_tensor_loads: 0,
        route_projection_cache_entries: 0,
        route_projection_cache_hits: 0,
        route_projection_loads: 0,
        requested_parallelism: 0,
        worker_count: 0,
        row_chunks: 0,
        output_checksum: None,
        output_l2_norm: None,
        initial_residual_checksum: None,
        residual_delta_checksum: None,
        final_residual_checksum: None,
        first_layer_id: None,
        last_layer_id: None,
        row_summaries: Vec::new(),
        uses_real_nvfp4_weights: false,
        uses_real_router: false,
        applies_sparse_mlp_residual_chain: false,
        covers_all_sparse_layers: false,
        covers_full_top_k: false,
        covers_decode_source: false,
        covers_prefill_source: false,
        covers_mtp_verify_source: false,
        covers_scheduler_source_modes: false,
        includes_attention: false,
        includes_dense_layers: false,
        includes_shared_expert: false,
        uses_full_model_residual: false,
        passed: false,
        skipped_reason: Some(
            "removed one-off probe; use layer-ordered execution stepper, request scheduler sparse TCP dispatch, and live ProtocolV2 expert-daemon coverage".to_owned(),
        ),
    }
}
