use ds41rt_core::ModelFacts;
use ds41rt_transport::{
    EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN, EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN,
    EXPERT_PROTOCOL_V2_ROUTE_ENTRY_LEN, EXPERT_PROTOCOL_V2_ROW_DESCRIPTOR_LEN,
};

use super::constants::{
    REAL_FULL_PREFLIGHT_DECODE_ROWS, REAL_FULL_PREFLIGHT_MTP_ROWS, REAL_FULL_PREFLIGHT_PREFILL_ROWS,
};
use super::types::{
    RealFullAttentionStagePlan, RealFullExecutionPlan, RealFullExecutionStageCounts,
    RealFullKvSemanticsPlan, RealFullLayerExecutionPlan, RealFullMlpStagePlan,
    RealFullProtocolPayloadPlan, RealFullSchedulerContract,
};

pub(super) fn real_full_execution_plan(
    expert_hosts: &[String],
    kv_bytes_per_token: usize,
    facts: &ModelFacts,
) -> RealFullExecutionPlan {
    let hidden_bytes_per_row = facts.hidden_size * std::mem::size_of::<u16>();
    let sparse_layer_count = facts.num_hidden_layers - facts.first_k_dense_replace;
    // Routed experts are strict TP, not EP: every rank receives every row and
    // the complete global route list. "Touched" therefore means all attached
    // ranks even when an ingestion/reduction owner is recorded separately.
    let max_touched_expert_hosts = expert_hosts.len();
    let decode_one_way_bytes = REAL_FULL_PREFLIGHT_DECODE_ROWS * hidden_bytes_per_row;
    let mtp_one_way_bytes = REAL_FULL_PREFLIGHT_MTP_ROWS * hidden_bytes_per_row;
    let prefill_one_way_bytes = REAL_FULL_PREFLIGHT_PREFILL_ROWS * hidden_bytes_per_row;
    let full_sparse_roundtrip =
        |one_way_bytes: usize| sparse_layer_count * max_touched_expert_hosts * one_way_bytes * 2;
    let decode_routes = REAL_FULL_PREFLIGHT_DECODE_ROWS * facts.top_k;
    let prefill_routes = REAL_FULL_PREFLIGHT_PREFILL_ROWS * facts.top_k;
    let mtp_routes = REAL_FULL_PREFLIGHT_MTP_ROWS * facts.top_k;
    let decode_routes_per_host = decode_routes;
    let prefill_routes_per_host = prefill_routes;
    let mtp_routes_per_host = mtp_routes;
    let decode_request_wire = protocol_v2_request_wire_bytes(
        REAL_FULL_PREFLIGHT_DECODE_ROWS,
        decode_routes_per_host,
        decode_one_way_bytes,
    );
    let decode_response_wire = protocol_v2_response_wire_bytes(decode_one_way_bytes);
    let prefill_request_wire = protocol_v2_request_wire_bytes(
        REAL_FULL_PREFLIGHT_PREFILL_ROWS,
        prefill_routes_per_host,
        prefill_one_way_bytes,
    );
    let prefill_response_wire = protocol_v2_response_wire_bytes(prefill_one_way_bytes);
    let mtp_request_wire = protocol_v2_request_wire_bytes(
        REAL_FULL_PREFLIGHT_MTP_ROWS,
        mtp_routes_per_host,
        mtp_one_way_bytes,
    );
    let mtp_response_wire = protocol_v2_response_wire_bytes(mtp_one_way_bytes);
    let full_sparse_wire_roundtrip = |request_wire: usize, response_wire: usize| {
        sparse_layer_count * max_touched_expert_hosts * (request_wire + response_wire)
    };

    RealFullExecutionPlan {
        status: "contract-only",
        scope: "decode-prefill-dspark native DeepSeek V4 runtime plan",
        hidden_dim: facts.hidden_size,
        hidden_dtype: "bf16",
        hidden_bytes_per_row,
        decode_rows: REAL_FULL_PREFLIGHT_DECODE_ROWS,
        mtp_verify_rows: REAL_FULL_PREFLIGHT_MTP_ROWS,
        prefill_chunk_rows: REAL_FULL_PREFLIGHT_PREFILL_ROWS,
        layer_count: facts.num_hidden_layers,
        dense_layer_count: facts.first_k_dense_replace,
        sparse_layer_count,
        stage_counts: RealFullExecutionStageCounts {
            attention_layers: facts.num_hidden_layers,
            compressed_kv_layers: facts.num_hidden_layers,
            dense_mlp_layers: facts.first_k_dense_replace,
            sparse_moe_layers: sparse_layer_count,
            remote_expert_exchange_layers: sparse_layer_count,
            residual_add_boundaries: facts.num_hidden_layers * 2,
        },
        protocol_payloads: RealFullProtocolPayloadPlan {
            protocol: "ExpertProtocolV2",
            request_header_bytes: EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN,
            response_header_bytes: EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN,
            row_descriptor_bytes: EXPERT_PROTOCOL_V2_ROW_DESCRIPTOR_LEN,
            route_entry_bytes: EXPERT_PROTOCOL_V2_ROUTE_ENTRY_LEN,
            sparse_decode_roundtrips_per_token: sparse_layer_count,
            sparse_prefill_roundtrips_per_chunk: sparse_layer_count,
            max_touched_expert_hosts,
            routes_per_decode_sparse_layer: decode_routes,
            routes_per_prefill_sparse_layer: prefill_routes,
            routes_per_mtp_sparse_layer: mtp_routes,
            routes_per_decode_touched_host: decode_routes_per_host,
            routes_per_prefill_touched_host: prefill_routes_per_host,
            routes_per_mtp_touched_host: mtp_routes_per_host,
            decode_logical_request_bytes_per_touched_host: decode_one_way_bytes,
            decode_logical_response_bytes_per_touched_host: decode_one_way_bytes,
            prefill_logical_request_bytes_per_touched_host: prefill_one_way_bytes,
            prefill_logical_response_bytes_per_touched_host: prefill_one_way_bytes,
            mtp_logical_request_bytes_per_touched_host: mtp_one_way_bytes,
            mtp_logical_response_bytes_per_touched_host: mtp_one_way_bytes,
            decode_wire_request_bytes_per_touched_host: decode_request_wire,
            decode_wire_response_bytes_per_touched_host: decode_response_wire,
            prefill_wire_request_bytes_per_touched_host: prefill_request_wire,
            prefill_wire_response_bytes_per_touched_host: prefill_response_wire,
            mtp_wire_request_bytes_per_touched_host: mtp_request_wire,
            mtp_wire_response_bytes_per_touched_host: mtp_response_wire,
            decode_full_sparse_roundtrip_logical_bytes: full_sparse_roundtrip(decode_one_way_bytes),
            prefill_full_sparse_roundtrip_logical_bytes: full_sparse_roundtrip(
                prefill_one_way_bytes,
            ),
            mtp_full_sparse_roundtrip_logical_bytes: full_sparse_roundtrip(mtp_one_way_bytes),
            decode_full_sparse_roundtrip_wire_bytes: full_sparse_wire_roundtrip(
                decode_request_wire,
                decode_response_wire,
            ),
            prefill_full_sparse_roundtrip_wire_bytes: full_sparse_wire_roundtrip(
                prefill_request_wire,
                prefill_response_wire,
            ),
            mtp_full_sparse_roundtrip_wire_bytes: full_sparse_wire_roundtrip(
                mtp_request_wire,
                mtp_response_wire,
            ),
        },
        kv_semantics: RealFullKvSemanticsPlan {
            layout: "deepseek-v4-compressed-bf16",
            bytes_per_token: kv_bytes_per_token,
            decode_reads_prefix_when_position_gt_zero: true,
            decode_writes_committed_current_token: true,
            prefill_chunk_zero_reads_prefix: false,
            prefill_later_chunks_read_prefix: true,
            prefill_writes_committed_chunk_range: true,
            mtp_reads_accepted_prefix: true,
            mtp_writes_tentative_draft_range: true,
            mtp_commits_only_accepted_prefix: true,
        },
        scheduler_contract: RealFullSchedulerContract {
            layer_order_is_strict: true,
            modes: ["decode", "prefill", "mtp_verify"],
            dense_layers_run_on_coordinator: facts.first_k_dense_replace,
            sparse_layers_require_expert_batches: sparse_layer_count,
            expert_batch_can_mix_compatible_sources: true,
            sampling_after_layer_count: facts.num_hidden_layers,
        },
        terminal_stages: vec!["final_norm", "lm_head", "full_vocab_sampling"],
        layers: (0..facts.num_hidden_layers)
            .map(|layer_id| real_full_layer_execution_plan(layer_id, facts))
            .collect(),
    }
}

fn protocol_v2_request_wire_bytes(
    row_count: usize,
    route_count: usize,
    payload_bytes: usize,
) -> usize {
    EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN
        + row_count * EXPERT_PROTOCOL_V2_ROW_DESCRIPTOR_LEN
        + route_count * EXPERT_PROTOCOL_V2_ROUTE_ENTRY_LEN
        + payload_bytes
}

fn protocol_v2_response_wire_bytes(payload_bytes: usize) -> usize {
    EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN + payload_bytes
}

fn real_full_layer_execution_plan(
    layer_id: usize,
    facts: &ModelFacts,
) -> RealFullLayerExecutionPlan {
    let is_dense = layer_id < facts.first_k_dense_replace;
    RealFullLayerExecutionPlan {
        layer_id,
        layer_kind: if is_dense {
            "dense-mlp"
        } else {
            "sparse-routed-moe"
        },
        attention: RealFullAttentionStagePlan {
            input_norm: true,
            qkv_projection: true,
            compressed_kv_read_write: true,
            attention_output_projection: true,
        },
        mlp: RealFullMlpStagePlan {
            post_attention_norm: true,
            dense_mlp_on_coordinator: is_dense,
            router_on_coordinator: !is_dense,
            routed_nvfp4_expert_exchange: !is_dense,
            shared_expert_on_coordinator: !is_dense,
            routes_per_row: if is_dense { 0 } else { facts.top_k },
        },
        residual_adds: 2,
    }
}

#[cfg(test)]
mod tests {
    use super::real_full_execution_plan;
    use crate::commands::real_full::constants::{
        REAL_FULL_PREFLIGHT_DECODE_ROWS, REAL_FULL_PREFLIGHT_MTP_ROWS,
        REAL_FULL_PREFLIGHT_PREFILL_ROWS,
    };
    use ds41rt_core::{
        ModelFacts, DS4_PRO_HIDDEN_SIZE, DS4_PRO_MOE_INTERMEDIATE_SIZE, DS4_PRO_NUM_HIDDEN_LAYERS,
        DS4_PRO_ROUTED_EXPERTS,
    };

    fn tp4_hosts() -> Vec<String> {
        ["ostrich", "dodo", "emu", "kiwi"]
            .into_iter()
            .map(str::to_owned)
            .collect()
    }

    #[test]
    fn execution_plan_uses_catalog_geometry_and_replicates_routes_to_every_tp_rank() {
        let hosts = tp4_hosts();
        let flash = ModelFacts::default();
        let flash_plan = real_full_execution_plan(&hosts, 1_024, &flash);
        assert_eq!(flash_plan.hidden_dim, flash.hidden_size);
        assert_eq!(flash_plan.layer_count, flash.num_hidden_layers);
        assert_eq!(flash_plan.protocol_payloads.max_touched_expert_hosts, 4);
        assert_eq!(
            flash_plan.protocol_payloads.routes_per_decode_touched_host,
            REAL_FULL_PREFLIGHT_DECODE_ROWS * flash.top_k,
        );
        assert_eq!(
            flash_plan.protocol_payloads.routes_per_prefill_touched_host,
            REAL_FULL_PREFLIGHT_PREFILL_ROWS * flash.top_k,
        );
        assert_eq!(
            flash_plan.protocol_payloads.routes_per_mtp_touched_host,
            REAL_FULL_PREFLIGHT_MTP_ROWS * flash.top_k,
        );

        let mut pro = flash;
        pro.hidden_size = DS4_PRO_HIDDEN_SIZE;
        pro.num_hidden_layers = DS4_PRO_NUM_HIDDEN_LAYERS;
        pro.routed_experts = DS4_PRO_ROUTED_EXPERTS;
        pro.moe_intermediate_size = DS4_PRO_MOE_INTERMEDIATE_SIZE;
        let pro_plan = real_full_execution_plan(&hosts, 2_048, &pro);
        assert_eq!(pro_plan.hidden_dim, DS4_PRO_HIDDEN_SIZE);
        assert_eq!(pro_plan.layer_count, DS4_PRO_NUM_HIDDEN_LAYERS);
        assert_eq!(pro_plan.sparse_layer_count, DS4_PRO_NUM_HIDDEN_LAYERS);
        assert_eq!(
            pro_plan
                .protocol_payloads
                .decode_logical_request_bytes_per_touched_host,
            DS4_PRO_HIDDEN_SIZE * std::mem::size_of::<u16>(),
        );
        assert_eq!(
            pro_plan.protocol_payloads.routes_per_decode_touched_host,
            REAL_FULL_PREFLIGHT_DECODE_ROWS * pro.top_k,
        );
    }
}
