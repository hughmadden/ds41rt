use anyhow::{Context, Result};
use bytes::Bytes;
use ds4rt_core::{
    admit_layerwaves_for_iteration, plan_prefill_chunks, DType, DecodeStep, ExpertBatch,
    ExpertBatchRoute, ExpertGraphBufferContract, ExpertHostBatchSet, ExpertOwnerLookup,
    GraphBucket, KvCacheBackingStore, KvCacheConfig, LayerId, LayerWave, LayerWaveMode, ModelFacts,
    MtpVerifyBlock, PositionId, PrefillChunkPolicy, Priority, RowSourceKind,
    DS4_EXPERT_TP_WORLD_SIZE, DS4_FLASH_HIDDEN_SIZE, DS4_FLASH_ROUTED_EXPERTS, DS4_FLASH_TOP_K,
    DS4_PRO_HIDDEN_SIZE, DS4_PRO_ROUTED_EXPERTS, DS4_PRO_TOP_K, EXPERT_HOSTS,
};
use ds4rt_ffi::{Ds4rtDeviceBuffer, Ds4rtHostBuffer};
use ds4rt_loader::{is_deepseek_v4_exl3_recipe, DEEPSEEK_V4_EXL3_RECIPE};
use ds4rt_transport::{
    expert_protocol_v2_compact_id, protocol_v2_synthetic_response,
    tcp_protocol_v2_host_batch_set_bf16_dispatch,
    tcp_protocol_v2_host_batch_set_bf16_payload_dispatch,
    verbs_host_protocol_v2_host_batch_set_bf16_dispatch,
    verbs_host_protocol_v2_host_batch_set_bf16_payload_dispatch,
    verbs_host_protocol_v2_host_batch_set_bf16_payload_dispatch_structural_stats,
    ExpertProtocolV2FrameBuffer, ExpertProtocolV2Request, ExpertProtocolV2Response,
    ExpertProtocolV2RouteEntry, ExpertV2Dtype, TcpProtocolV2HostBatchSetBf16PayloadDispatch,
    TcpProtocolV2HostBatchSetDispatch, TcpProtocolV2HostBatchSetDispatchStats,
    TcpProtocolV2HostBatchSetPersistentClient, TcpProtocolV2HostBatchTarget, TcpTransportConfig,
    VerbsHostProtocolV2HostBatchSetBf16PayloadChunk, VerbsHostProtocolV2HostBatchSetPayloadStart,
    VerbsHostProtocolV2HostBatchSetPersistentClient,
    VerbsHostProtocolV2ReducedIdentityPayloadPending,
    VerbsHostProtocolV2ReducedIdentityPayloadStart, EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN,
    EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN, EXPERT_PROTOCOL_V2_ROUTE_ENTRY_LEN,
    EXPERT_PROTOCOL_V2_ROW_DESCRIPTOR_LEN,
};
use std::{env, sync::OnceLock, time::Duration};

use super::super::constants::REAL_FULL_PREFLIGHT_KV_RESERVATION_ID;
use super::super::coordinator_kernels::{
    DeviceBf16Output, Ds4FlashTp4CoordinatorReductionArena, Ds4FlashTp4HostPartialChunk,
    Ds4FlashTp4ReductionDeviceBuffers,
};
use super::super::expert_probe::REAL_NVFP4_PROTOCOL_V2_EXECUTOR;
use super::super::intermediate_sharding::{
    spark_expert_reduction_dispatch_for_rows, SparkExpertReductionDispatch,
};
use super::super::sparse_mlp::router::ScoredRoute;
use super::super::types::RealFullSchedulerProtocolV2BatchProbe;
use super::execution::RealFullSchedulerExecutionShape;

const REAL_FULL_PROTOCOL_V2_TIMEOUT_MS_ENV: &str = "DS4RT_REAL_FULL_PROTOCOL_V2_TIMEOUT_MS";
const REAL_FULL_MOE_RESPONSE_DTYPE_ENV: &str = "DS4RT_REAL_FULL_MOE_RESPONSE_DTYPE";
const REAL_FULL_MOE_OWNER_RESPONSE_DTYPE_ENV: &str = "DS4RT_REAL_FULL_MOE_OWNER_RESPONSE_DTYPE";
fn parse_moe_response_dtype(name: &str, value: &str) -> std::result::Result<ExpertV2Dtype, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "bf16" => Ok(ExpertV2Dtype::Bf16),
        "fp8" | "fp8-e4m3" | "fp8-e4m3-row-scaled" => {
            Ok(ExpertV2Dtype::Fp8E4m3RowScaled)
        }
        "nvfp4" | "nvfp4-e2m1" | "nvfp4-e2m1-fp8-e4m3" => {
            Ok(ExpertV2Dtype::Nvfp4E2m1Fp8E4m3)
        }
        _ => Err(format!(
            "unsupported {name} value {value}; expected bf16, fp8-e4m3-row-scaled, or nvfp4-e2m1-fp8-e4m3"
        )),
    }
}

pub(super) fn real_full_moe_response_dtype() -> Result<ExpertV2Dtype> {
    static RESPONSE_DTYPE: OnceLock<std::result::Result<ExpertV2Dtype, String>> = OnceLock::new();
    RESPONSE_DTYPE
        .get_or_init(|| {
            let value =
                env::var(REAL_FULL_MOE_RESPONSE_DTYPE_ENV).unwrap_or_else(|_| "bf16".to_owned());
            parse_moe_response_dtype(REAL_FULL_MOE_RESPONSE_DTYPE_ENV, &value)
        })
        .clone()
        .map_err(anyhow::Error::msg)
}

fn real_full_moe_owner_response_dtype() -> Result<ExpertV2Dtype> {
    static RESPONSE_DTYPE: OnceLock<std::result::Result<ExpertV2Dtype, String>> = OnceLock::new();
    RESPONSE_DTYPE
        .get_or_init(|| match env::var(REAL_FULL_MOE_OWNER_RESPONSE_DTYPE_ENV) {
            Ok(value) => parse_moe_response_dtype(REAL_FULL_MOE_OWNER_RESPONSE_DTYPE_ENV, &value),
            Err(_) => real_full_moe_response_dtype().map_err(|error| error.to_string()),
        })
        .clone()
        .map_err(anyhow::Error::msg)
}

fn real_full_moe_response_dtype_for_owner_fanout(owner_fanout: bool) -> Result<ExpertV2Dtype> {
    if owner_fanout {
        real_full_moe_owner_response_dtype()
    } else {
        real_full_moe_response_dtype()
    }
}

fn spark_expert_reduction_dispatch_for_batch(
    batch: &ExpertBatch,
) -> Result<Option<SparkExpertReductionDispatch>> {
    spark_expert_reduction_dispatch_for_rows(batch.num_rows())
}

pub(super) fn real_full_moe_response_dtype_for_batch(batch: &ExpertBatch) -> Result<ExpertV2Dtype> {
    let reduction = spark_expert_reduction_dispatch_for_batch(batch)?;
    real_full_moe_response_dtype_for_owner_fanout(
        reduction.is_some_and(|reduction| reduction.owner_fanout),
    )
}

pub(in crate::commands::real_full) fn real_full_protocol_v2_transport_config(
) -> Result<TcpTransportConfig> {
    let mut config = TcpTransportConfig::default();
    match env::var(REAL_FULL_PROTOCOL_V2_TIMEOUT_MS_ENV) {
        Ok(value) if !value.trim().is_empty() => {
            let timeout_ms = value.trim().parse::<u64>().with_context(|| {
                format!("invalid {REAL_FULL_PROTOCOL_V2_TIMEOUT_MS_ENV} value {value}")
            })?;
            anyhow::ensure!(
                timeout_ms > 0,
                "{REAL_FULL_PROTOCOL_V2_TIMEOUT_MS_ENV} must be positive"
            );
            config.timeout = Duration::from_millis(timeout_ms);
        }
        _ => {}
    }
    Ok(config)
}

pub(super) fn real_full_protocol_v2_batch_probe(
    layer_id: usize,
    batch: &ExpertBatch,
) -> Result<RealFullSchedulerProtocolV2BatchProbe> {
    let core_routes = core_routes_for_batch(batch);
    let routes = protocol_routes_for_core_routes(&core_routes);
    let hidden_payload = hidden_payload_for_batch(batch);
    let request = ExpertProtocolV2Request::from_expert_batch(
        REAL_FULL_PREFLIGHT_KV_RESERVATION_ID,
        batch,
        routes,
        hidden_payload,
    )?;
    let request_stats = request.wire_stats();
    let response = protocol_v2_synthetic_response(&request)?;
    let response_stats = response.wire_stats();
    let request_frame = request_frame_probe(&request)?;
    let response_frame = response_frame_probe(&response)?;
    let reconstruction = response_reconstruction_probe(batch, &response)?;
    let host_partition =
        host_partition_probe(batch, &core_routes, HostWireEnvelopeMode::ConstructRequests)?;
    let passed = request_frame.decoded_matches
        && response_frame.decoded_matches
        && request_frame.stable_allocation
        && response_frame.stable_allocation
        && reconstruction.row_order_matches
        && reconstruction.payload_matches
        && host_partition.routes_match_global
        && host_partition.graph_counts_valid
        && host_partition.wire_envelopes_valid
        && request_stats.logical_payload_bytes == batch.num_rows() * batch.hidden_bytes_per_row
        && response_stats.logical_payload_bytes == batch.num_rows() * batch.hidden_bytes_per_row;

    Ok(RealFullSchedulerProtocolV2BatchProbe {
        status: "encoded-and-reconstructed-mixed-expert-batch-protocol-v2",
        scope: "encode the first real-full sparse-layer mixed ExpertBatch as binary ExpertProtocolV2 request/response frames and reconstruct row-ordered partial outputs",
        layer_id,
        rows: batch.num_rows(),
        routes: batch.route_count(),
        source_modes: vec!["prefill_chunk", "mtp_verify", "decode_step"],
        hidden_dim: batch.hidden_dim,
        hidden_bytes_per_row: batch.hidden_bytes_per_row,
        hidden_payload_bytes: batch.num_rows() * batch.hidden_bytes_per_row,
        request_wire_bytes: request_frame.wire_bytes,
        response_wire_bytes: response_frame.wire_bytes,
        request_logical_payload_bytes: request_stats.logical_payload_bytes,
        response_logical_payload_bytes: response_stats.logical_payload_bytes,
        request_frame_buffer_capacity_bytes: request_frame.capacity_bytes,
        response_frame_buffer_capacity_bytes: response_frame.capacity_bytes,
        request_frame_buffer_stable: request_frame.stable_allocation,
        response_frame_buffer_stable: response_frame.stable_allocation,
        decoded_request_matches: request_frame.decoded_matches,
        decoded_response_matches: response_frame.decoded_matches,
        reconstructed_response_rows: reconstruction.rows,
        reconstructed_response_payload_bytes: reconstruction.payload_bytes,
        reconstructed_response_row_order_matches: reconstruction.row_order_matches,
        reconstructed_response_payload_matches: reconstruction.payload_matches,
        host_batches: host_partition.host_batches,
        host_batch_rows: host_partition.rows,
        host_batch_routes: host_partition.routes,
        host_batch_expert_tiles: host_partition.expert_tiles,
        host_batch_routes_match_global: host_partition.routes_match_global,
        host_batch_graph_counts_valid: host_partition.graph_counts_valid,
        host_request_frames: host_partition.request_frames,
        host_request_rows: host_partition.request_rows,
        host_request_routes: host_partition.request_routes,
        host_request_payload_bytes: host_partition.request_payload_bytes,
        host_request_wire_bytes: host_partition.request_wire_bytes,
        host_response_frames: host_partition.response_frames,
        host_response_rows: host_partition.response_rows,
        host_response_payload_bytes: host_partition.response_payload_bytes,
        host_response_wire_bytes: host_partition.response_wire_bytes,
        host_wire_envelopes_valid: host_partition.wire_envelopes_valid,
        passed,
    })
}

fn core_routes_for_batch(batch: &ExpertBatch) -> Vec<ExpertBatchRoute> {
    batch
        .rows
        .iter()
        .enumerate()
        .flat_map(|(row_index, row)| {
            (0..row.route_count).map(move |route| ExpertBatchRoute {
                row_index,
                expert_id: (row_index * row.route_count + route) % batch.routed_experts,
                gate_weight: 1.0 / row.route_count as f32,
            })
        })
        .collect()
}

fn protocol_routes_for_core_routes(routes: &[ExpertBatchRoute]) -> Vec<ExpertProtocolV2RouteEntry> {
    routes
        .iter()
        .map(|route| ExpertProtocolV2RouteEntry {
            row_index: route.row_index as u32,
            expert_id: route.expert_id as u32,
            gate_weight: route.gate_weight,
        })
        .collect()
}

fn hidden_payload_for_batch(batch: &ExpertBatch) -> Vec<u8> {
    let mut payload = vec![0_u8; batch.num_rows() * batch.hidden_bytes_per_row];
    for (row_index, row) in batch.rows.iter().enumerate() {
        let start = row_index * batch.hidden_bytes_per_row;
        payload[start..start + 8].copy_from_slice(&row.row_id.to_le_bytes());
        payload[start + 8..start + 16].copy_from_slice(&row.token_position.0.to_le_bytes());
        payload[start + 16..start + 24].copy_from_slice(&(row.route_offset as u64).to_le_bytes());
        payload[start + 24] = row_source_marker(row.source_kind);
        let marker = 1.0_f32
            + row_source_marker(row.source_kind) as f32 * 0.125
            + (row_index % 17) as f32 * 0.00390625;
        if batch.hidden_bytes_per_row >= 2 {
            let marker_offset = 32.min(batch.hidden_bytes_per_row - 2);
            payload[start + marker_offset..start + marker_offset + 2]
                .copy_from_slice(&((marker.to_bits() >> 16) as u16).to_le_bytes());
        }
    }
    payload
}

fn row_source_marker(source_kind: RowSourceKind) -> u8 {
    match source_kind {
        RowSourceKind::DecodeStep => 1,
        RowSourceKind::PrefillChunk => 2,
        RowSourceKind::MtpVerifyBlock => 3,
        RowSourceKind::Benchmark => 4,
    }
}

struct ProtocolV2FrameProbe {
    wire_bytes: usize,
    capacity_bytes: usize,
    stable_allocation: bool,
    decoded_matches: bool,
}

struct ProtocolV2ReconstructionProbe {
    rows: usize,
    payload_bytes: usize,
    row_order_matches: bool,
    payload_matches: bool,
}

struct HostPartitionProbe {
    host_batches: usize,
    rows: usize,
    routes: usize,
    expert_tiles: usize,
    routes_match_global: bool,
    graph_counts_valid: bool,
    request_frames: usize,
    request_rows: usize,
    request_routes: usize,
    request_payload_bytes: usize,
    request_wire_bytes: usize,
    response_frames: usize,
    response_rows: usize,
    response_payload_bytes: usize,
    response_wire_bytes: usize,
    wire_envelopes_valid: bool,
}

#[derive(Debug, Clone, Copy)]
enum HostWireEnvelopeMode {
    ConstructRequests,
    CountOnly,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct RealFullSchedulerHostBatchPartitionProbe {
    pub(super) host_batch_sets: usize,
    pub(super) host_batches: usize,
    pub(super) rows: usize,
    pub(super) routes: usize,
    pub(super) expert_tiles: usize,
    pub(super) routes_match_global: bool,
    pub(super) graph_counts_valid: bool,
    pub(super) request_frames: usize,
    pub(super) request_rows: usize,
    pub(super) request_routes: usize,
    pub(super) request_payload_bytes: usize,
    pub(super) request_wire_bytes: usize,
    pub(super) response_frames: usize,
    pub(super) response_rows: usize,
    pub(super) response_payload_bytes: usize,
    pub(super) response_wire_bytes: usize,
    pub(super) wire_envelopes_valid: bool,
}

pub(super) fn real_full_scheduler_host_batch_partition_probe(
    batch: &ExpertBatch,
) -> Result<RealFullSchedulerHostBatchPartitionProbe> {
    let core_routes = core_routes_for_batch(batch);
    let host_partition =
        host_partition_probe(batch, &core_routes, HostWireEnvelopeMode::CountOnly)?;
    Ok(RealFullSchedulerHostBatchPartitionProbe {
        host_batch_sets: 1,
        host_batches: host_partition.host_batches,
        rows: host_partition.rows,
        routes: host_partition.routes,
        expert_tiles: host_partition.expert_tiles,
        routes_match_global: host_partition.routes_match_global,
        graph_counts_valid: host_partition.graph_counts_valid,
        request_frames: host_partition.request_frames,
        request_rows: host_partition.request_rows,
        request_routes: host_partition.request_routes,
        request_payload_bytes: host_partition.request_payload_bytes,
        request_wire_bytes: host_partition.request_wire_bytes,
        response_frames: host_partition.response_frames,
        response_rows: host_partition.response_rows,
        response_payload_bytes: host_partition.response_payload_bytes,
        response_wire_bytes: host_partition.response_wire_bytes,
        wire_envelopes_valid: host_partition.wire_envelopes_valid,
    })
}

// Covered by the full-size TCP regression and used by request-shaped TCP serving.
#[allow(dead_code)]
pub(in crate::commands::real_full) async fn real_full_scheduler_host_batch_set_tcp_dispatch(
    batch: &ExpertBatch,
    targets: &[TcpProtocolV2HostBatchTarget],
    owner_lookup: Option<&ExpertOwnerLookup>,
    request_id_base: u64,
) -> Result<TcpProtocolV2HostBatchSetDispatch> {
    let core_routes = core_routes_for_batch(batch);
    let hidden_payload = hidden_payload_for_batch(batch);
    real_full_scheduler_host_batch_set_tcp_dispatch_with_payload(
        batch,
        &core_routes,
        &hidden_payload,
        targets,
        owner_lookup,
        request_id_base,
    )
    .await
}

pub(in crate::commands::real_full) fn scored_routes_for_scheduler_batch(
    batch: &ExpertBatch,
    row_routes: &[Vec<ScoredRoute>],
) -> Result<Vec<ExpertBatchRoute>> {
    anyhow::ensure!(
        row_routes.len() == batch.num_rows(),
        "scheduler scored-route row count {} did not match batch rows {}",
        row_routes.len(),
        batch.num_rows()
    );
    let mut routes = Vec::with_capacity(batch.route_count());
    for (row_index, (row, scored_routes)) in batch.rows.iter().zip(row_routes.iter()).enumerate() {
        anyhow::ensure!(
            scored_routes.len() == row.route_count,
            "scheduler scored-route count for row {row_index} was {}, expected {}",
            scored_routes.len(),
            row.route_count
        );
        for route in scored_routes {
            routes.push(ExpertBatchRoute {
                row_index,
                expert_id: route.expert_id,
                gate_weight: route.normalized_weight,
            });
        }
    }
    anyhow::ensure!(
        routes.len() == batch.route_count(),
        "scheduler scored routes produced {} routes, expected {}",
        routes.len(),
        batch.route_count()
    );
    Ok(routes)
}

pub(in crate::commands::real_full) async fn real_full_scheduler_host_batch_set_tcp_dispatch_with_payload(
    batch: &ExpertBatch,
    routes: &[ExpertBatchRoute],
    global_hidden_payload: &[u8],
    targets: &[TcpProtocolV2HostBatchTarget],
    owner_lookup: Option<&ExpertOwnerLookup>,
    request_id_base: u64,
) -> Result<TcpProtocolV2HostBatchSetDispatch> {
    anyhow::ensure!(
        global_hidden_payload.len() == batch.num_rows() * batch.hidden_bytes_per_row,
        "scheduler TCP dispatch hidden payload bytes {} did not match batch rows {} * row bytes {}",
        global_hidden_payload.len(),
        batch.num_rows(),
        batch.hidden_bytes_per_row
    );
    let set = scheduler_host_batch_set(batch, routes, owner_lookup)?;
    tcp_protocol_v2_host_batch_set_bf16_dispatch(
        &set,
        global_hidden_payload,
        targets,
        request_id_base,
        real_full_protocol_v2_transport_config()?,
    )
    .await
}

pub(in crate::commands::real_full) async fn real_full_scheduler_host_batch_set_verbs_host_dispatch_with_payload(
    batch: &ExpertBatch,
    routes: &[ExpertBatchRoute],
    global_hidden_payload: &[u8],
    targets: &[TcpProtocolV2HostBatchTarget],
    owner_lookup: Option<&ExpertOwnerLookup>,
    request_id_base: u64,
) -> Result<TcpProtocolV2HostBatchSetDispatch> {
    anyhow::ensure!(
        global_hidden_payload.len() == batch.num_rows() * batch.hidden_bytes_per_row,
        "scheduler verbs-host dispatch hidden payload bytes {} did not match batch rows {} * row bytes {}",
        global_hidden_payload.len(),
        batch.num_rows(),
        batch.hidden_bytes_per_row
    );
    let set = scheduler_host_batch_set(batch, routes, owner_lookup)?;
    verbs_host_protocol_v2_host_batch_set_bf16_dispatch(
        &set,
        global_hidden_payload,
        targets,
        request_id_base,
        real_full_protocol_v2_transport_config()?,
    )
    .await
}

pub(in crate::commands::real_full) async fn real_full_scheduler_host_batch_set_tcp_dispatch_with_payload_persistent(
    client: &mut TcpProtocolV2HostBatchSetPersistentClient,
    batch: &ExpertBatch,
    routes: &[ExpertBatchRoute],
    global_hidden_payload: &[u8],
    owner_lookup: Option<&ExpertOwnerLookup>,
    request_id_base: u64,
) -> Result<TcpProtocolV2HostBatchSetDispatch> {
    anyhow::ensure!(
        global_hidden_payload.len() == batch.num_rows() * batch.hidden_bytes_per_row,
        "scheduler persistent TCP dispatch hidden payload bytes {} did not match batch rows {} * row bytes {}",
        global_hidden_payload.len(),
        batch.num_rows(),
        batch.hidden_bytes_per_row
    );
    let set = scheduler_host_batch_set(batch, routes, owner_lookup)?;
    client
        .dispatch_bf16(&set, global_hidden_payload, request_id_base)
        .await
}

pub(in crate::commands::real_full) async fn real_full_scheduler_host_batch_set_verbs_host_dispatch_with_payload_persistent(
    client: &VerbsHostProtocolV2HostBatchSetPersistentClient,
    batch: &ExpertBatch,
    routes: &[ExpertBatchRoute],
    global_hidden_payload: &[u8],
    owner_lookup: Option<&ExpertOwnerLookup>,
    request_id_base: u64,
) -> Result<TcpProtocolV2HostBatchSetDispatch> {
    anyhow::ensure!(
        global_hidden_payload.len() == batch.num_rows() * batch.hidden_bytes_per_row,
        "scheduler persistent verbs-host dispatch hidden payload bytes {} did not match batch rows {} * row bytes {}",
        global_hidden_payload.len(),
        batch.num_rows(),
        batch.hidden_bytes_per_row
    );
    let set = scheduler_host_batch_set(batch, routes, owner_lookup)?;
    client
        .dispatch_bf16(&set, global_hidden_payload, request_id_base)
        .await
}

pub(in crate::commands::real_full) async fn real_full_scheduler_host_batch_set_tcp_bf16_payload_dispatch_with_payload_persistent(
    client: &mut TcpProtocolV2HostBatchSetPersistentClient,
    batch: &ExpertBatch,
    routes: &[ExpertBatchRoute],
    global_hidden_payload: &[u8],
    owner_lookup: Option<&ExpertOwnerLookup>,
    request_id_base: u64,
) -> Result<TcpProtocolV2HostBatchSetBf16PayloadDispatch> {
    anyhow::ensure!(
        global_hidden_payload.len() == batch.num_rows() * batch.hidden_bytes_per_row,
        "scheduler persistent TCP BF16 payload dispatch hidden payload bytes {} did not match batch rows {} * row bytes {}",
        global_hidden_payload.len(),
        batch.num_rows(),
        batch.hidden_bytes_per_row
    );
    let set = scheduler_host_batch_set(batch, routes, owner_lookup)?;
    client
        .dispatch_bf16_payload(&set, global_hidden_payload, request_id_base)
        .await
}

pub(in crate::commands::real_full) async fn real_full_scheduler_host_batch_set_verbs_host_bf16_payload_dispatch_with_payload_persistent(
    client: &VerbsHostProtocolV2HostBatchSetPersistentClient,
    batch: &ExpertBatch,
    routes: &[ExpertBatchRoute],
    global_hidden_payload: &[u8],
    owner_lookup: Option<&ExpertOwnerLookup>,
    request_id_base: u64,
) -> Result<TcpProtocolV2HostBatchSetBf16PayloadDispatch> {
    anyhow::ensure!(
        global_hidden_payload.len() == batch.num_rows() * batch.hidden_bytes_per_row,
        "scheduler persistent verbs-host BF16 payload dispatch hidden payload bytes {} did not match batch rows {} * row bytes {}",
        global_hidden_payload.len(),
        batch.num_rows(),
        batch.hidden_bytes_per_row
    );
    let set = scheduler_host_batch_set(batch, routes, owner_lookup)?;
    client
        .dispatch_bf16_payload(&set, global_hidden_payload, request_id_base)
        .await
}

pub(in crate::commands::real_full) async fn real_full_scheduler_host_batch_set_verbs_host_bf16_payload_dispatch_with_payload_persistent_streaming(
    client: &VerbsHostProtocolV2HostBatchSetPersistentClient,
    batch: &ExpertBatch,
    routes: &[ExpertBatchRoute],
    global_hidden_payload: &[u8],
    owner_lookup: Option<&ExpertOwnerLookup>,
    request_id_base: u64,
    chunk_tx: std::sync::mpsc::Sender<VerbsHostProtocolV2HostBatchSetBf16PayloadChunk>,
) -> Result<TcpProtocolV2HostBatchSetDispatchStats> {
    anyhow::ensure!(
        global_hidden_payload.len() == batch.num_rows() * batch.hidden_bytes_per_row,
        "scheduler persistent streaming verbs-host BF16 payload dispatch hidden payload bytes {} did not match batch rows {} * row bytes {}",
        global_hidden_payload.len(),
        batch.num_rows(),
        batch.hidden_bytes_per_row
    );
    let set = scheduler_host_batch_set(batch, routes, owner_lookup)?;
    let reduction = spark_expert_reduction_dispatch_for_batch(batch)?;
    let response_dtype = real_full_moe_response_dtype_for_batch(batch)?;
    let reduced_root_host_index = reduction.map(|reduction| reduction.root_rank);
    let owner_fanout = reduction.is_some_and(|reduction| reduction.owner_fanout);
    let row_sharded_reduction = reduction.is_some_and(|reduction| reduction.row_sharded);
    client
        .dispatch_bf16_payload_streaming(
            &set,
            global_hidden_payload,
            request_id_base,
            response_dtype,
            reduced_root_host_index,
            owner_fanout,
            row_sharded_reduction,
            chunk_tx,
        )
        .await
}

pub(in crate::commands::real_full) async fn real_full_scheduler_verbs_host_direct_owner_payload_dispatch_with_payload_persistent_streaming(
    client: &VerbsHostProtocolV2HostBatchSetPersistentClient,
    batch: &ExpertBatch,
    routes: &[ExpertBatchRoute],
    global_hidden_payload: &[u8],
    request_id: u64,
    chunk_tx: &std::sync::mpsc::Sender<VerbsHostProtocolV2HostBatchSetBf16PayloadChunk>,
) -> Result<Option<TcpProtocolV2HostBatchSetDispatchStats>> {
    let Some(reduction) = spark_expert_reduction_dispatch_for_batch(batch)? else {
        return Ok(None);
    };
    if !reduction.owner_fanout {
        return Ok(None);
    }
    let hidden_payload_bytes = batch
        .num_rows()
        .checked_mul(batch.hidden_bytes_per_row)
        .context("direct Spark-owner hidden payload byte count overflow")?;
    anyhow::ensure!(
        global_hidden_payload.len() == hidden_payload_bytes,
        "direct Spark-owner hidden payload bytes {} did not match {} rows of {} bytes",
        global_hidden_payload.len(),
        batch.num_rows(),
        batch.hidden_bytes_per_row
    );
    let response_dtype = real_full_moe_owner_response_dtype()?;
    let request = ExpertProtocolV2Request::from_expert_batch(
        request_id,
        batch,
        protocol_routes_for_core_routes(routes),
        global_hidden_payload.to_vec(),
    )?;
    let request = match response_dtype {
        ExpertV2Dtype::Fp8E4m3RowScaled => request.with_fp8_e4m3_row_scaled_response(),
        ExpertV2Dtype::Nvfp4E2m1Fp8E4m3 => request.with_nvfp4_e2m1_fp8_e4m3_response(),
        _ => request,
    }
    .with_spark_reduction();
    let logical_routes = scheduler_dispatched_route_count(batch.route_count())?;
    client
        .dispatch_reduced_identity_payload_streaming(
            request,
            response_dtype,
            reduction.root_rank,
            logical_routes,
            chunk_tx.clone(),
        )
        .await
        .map(Some)
}

pub(in crate::commands::real_full) enum DirectOwnerPayloadDispatchStart {
    Started(VerbsHostProtocolV2ReducedIdentityPayloadPending),
    Unavailable(Vec<u8>),
}

pub(in crate::commands::real_full) fn real_full_scheduler_verbs_host_try_start_direct_owner_payload_dispatch(
    client: &VerbsHostProtocolV2HostBatchSetPersistentClient,
    batch: &ExpertBatch,
    routes: &[ExpertBatchRoute],
    global_hidden_payload: Vec<u8>,
    request_id: u64,
) -> Result<DirectOwnerPayloadDispatchStart> {
    let reduction = spark_expert_reduction_dispatch_for_batch(batch)?;
    if reduction.is_none() && batch.num_rows() == 1 && scheduler_route_replication_factor()? > 1 {
        let response_dtype = real_full_moe_response_dtype_for_batch(batch)?;
        let request = ExpertProtocolV2Request::from_expert_batch(
            request_id,
            batch,
            protocol_routes_for_core_routes(routes),
            global_hidden_payload,
        )?;
        return Ok(
            match client.try_start_replicated_one_row_payload(request, response_dtype)? {
                VerbsHostProtocolV2HostBatchSetPayloadStart::Started(pending) => {
                    DirectOwnerPayloadDispatchStart::Started(pending)
                }
                VerbsHostProtocolV2HostBatchSetPayloadStart::Busy(payload) => {
                    DirectOwnerPayloadDispatchStart::Unavailable(payload)
                }
            },
        );
    }
    let Some(reduction) = reduction else {
        return Ok(DirectOwnerPayloadDispatchStart::Unavailable(
            global_hidden_payload,
        ));
    };
    if !reduction.owner_fanout {
        return Ok(DirectOwnerPayloadDispatchStart::Unavailable(
            global_hidden_payload,
        ));
    }
    let hidden_payload_bytes = batch
        .num_rows()
        .checked_mul(batch.hidden_bytes_per_row)
        .context("direct Spark-owner hidden payload byte count overflow")?;
    anyhow::ensure!(
        global_hidden_payload.len() == hidden_payload_bytes,
        "direct Spark-owner hidden payload bytes {} did not match {} rows of {} bytes",
        global_hidden_payload.len(),
        batch.num_rows(),
        batch.hidden_bytes_per_row
    );
    let response_dtype = real_full_moe_owner_response_dtype()?;
    let request = ExpertProtocolV2Request::from_expert_batch(
        request_id,
        batch,
        protocol_routes_for_core_routes(routes),
        global_hidden_payload,
    )?;
    let request = match response_dtype {
        ExpertV2Dtype::Fp8E4m3RowScaled => request.with_fp8_e4m3_row_scaled_response(),
        ExpertV2Dtype::Nvfp4E2m1Fp8E4m3 => request.with_nvfp4_e2m1_fp8_e4m3_response(),
        _ => request,
    }
    .with_spark_reduction();
    let logical_routes = scheduler_dispatched_route_count(batch.route_count())?;
    Ok(
        match client.try_start_reduced_identity_payload(
            request,
            response_dtype,
            reduction.root_rank,
            logical_routes,
        )? {
            VerbsHostProtocolV2ReducedIdentityPayloadStart::Started(pending) => {
                DirectOwnerPayloadDispatchStart::Started(pending)
            }
            VerbsHostProtocolV2ReducedIdentityPayloadStart::Busy(request) => {
                DirectOwnerPayloadDispatchStart::Unavailable(request.hidden_payload.to_vec())
            }
        },
    )
}

pub(in crate::commands::real_full) async fn real_full_scheduler_host_batch_set_verbs_host_bf16_payload_dispatch_with_payload(
    batch: &ExpertBatch,
    routes: &[ExpertBatchRoute],
    global_hidden_payload: &[u8],
    targets: &[TcpProtocolV2HostBatchTarget],
    owner_lookup: Option<&ExpertOwnerLookup>,
    request_id_base: u64,
) -> Result<TcpProtocolV2HostBatchSetBf16PayloadDispatch> {
    anyhow::ensure!(
        global_hidden_payload.len() == batch.num_rows() * batch.hidden_bytes_per_row,
        "scheduler verbs-host BF16 payload dispatch hidden payload bytes {} did not match batch rows {} * row bytes {}",
        global_hidden_payload.len(),
        batch.num_rows(),
        batch.hidden_bytes_per_row
    );
    let set = scheduler_host_batch_set(batch, routes, owner_lookup)?;
    verbs_host_protocol_v2_host_batch_set_bf16_payload_dispatch(
        &set,
        global_hidden_payload,
        targets,
        request_id_base,
        real_full_protocol_v2_transport_config()?,
    )
    .await
}

pub(in crate::commands::real_full) async fn real_full_scheduler_host_batch_set_tcp_bf16_payload_dispatch_with_payload_persistent_structural_stats(
    client: &mut TcpProtocolV2HostBatchSetPersistentClient,
    batch: &ExpertBatch,
    routes: &[ExpertBatchRoute],
    global_hidden_payload: &[u8],
    owner_lookup: Option<&ExpertOwnerLookup>,
    request_id_base: u64,
) -> Result<TcpProtocolV2HostBatchSetBf16PayloadDispatch> {
    anyhow::ensure!(
        global_hidden_payload.len() == batch.num_rows() * batch.hidden_bytes_per_row,
        "scheduler persistent TCP BF16 payload dispatch hidden payload bytes {} did not match batch rows {} * row bytes {}",
        global_hidden_payload.len(),
        batch.num_rows(),
        batch.hidden_bytes_per_row
    );
    let set = scheduler_host_batch_set(batch, routes, owner_lookup)?;
    client
        .dispatch_bf16_payload_structural_stats(&set, global_hidden_payload, request_id_base)
        .await
}

pub(in crate::commands::real_full) async fn real_full_scheduler_host_batch_set_verbs_host_bf16_payload_dispatch_with_payload_persistent_structural_stats(
    client: &VerbsHostProtocolV2HostBatchSetPersistentClient,
    batch: &ExpertBatch,
    routes: &[ExpertBatchRoute],
    global_hidden_payload: &[u8],
    owner_lookup: Option<&ExpertOwnerLookup>,
    request_id_base: u64,
) -> Result<TcpProtocolV2HostBatchSetBf16PayloadDispatch> {
    anyhow::ensure!(
        global_hidden_payload.len() == batch.num_rows() * batch.hidden_bytes_per_row,
        "scheduler persistent verbs-host BF16 payload structural dispatch hidden payload bytes {} did not match batch rows {} * row bytes {}",
        global_hidden_payload.len(),
        batch.num_rows(),
        batch.hidden_bytes_per_row
    );
    let set = scheduler_host_batch_set(batch, routes, owner_lookup)?;
    client
        .dispatch_bf16_payload_structural_stats(&set, global_hidden_payload, request_id_base)
        .await
}

pub(in crate::commands::real_full) async fn real_full_scheduler_host_batch_set_verbs_host_bf16_payload_dispatch_with_payload_structural_stats(
    batch: &ExpertBatch,
    routes: &[ExpertBatchRoute],
    global_hidden_payload: &[u8],
    targets: &[TcpProtocolV2HostBatchTarget],
    owner_lookup: Option<&ExpertOwnerLookup>,
    request_id_base: u64,
) -> Result<TcpProtocolV2HostBatchSetBf16PayloadDispatch> {
    anyhow::ensure!(
        global_hidden_payload.len() == batch.num_rows() * batch.hidden_bytes_per_row,
        "scheduler verbs-host BF16 payload structural dispatch hidden payload bytes {} did not match batch rows {} * row bytes {}",
        global_hidden_payload.len(),
        batch.num_rows(),
        batch.hidden_bytes_per_row
    );
    let set = scheduler_host_batch_set(batch, routes, owner_lookup)?;
    verbs_host_protocol_v2_host_batch_set_bf16_payload_dispatch_structural_stats(
        &set,
        global_hidden_payload,
        targets,
        request_id_base,
        real_full_protocol_v2_transport_config()?,
    )
    .await
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(in crate::commands::real_full) struct RealFullSchedulerSparseTcpDispatchProbe {
    pub(in crate::commands::real_full) status: &'static str,
    pub(in crate::commands::real_full) scope: &'static str,
    pub(in crate::commands::real_full) sparse_layers: usize,
    pub(in crate::commands::real_full) scheduler_iterations_per_sparse_layer: usize,
    pub(in crate::commands::real_full) sparse_batches: usize,
    pub(in crate::commands::real_full) host_batches: usize,
    pub(in crate::commands::real_full) global_rows: usize,
    pub(in crate::commands::real_full) host_rows: usize,
    pub(in crate::commands::real_full) routes: usize,
    pub(in crate::commands::real_full) request_wire_bytes: usize,
    pub(in crate::commands::real_full) response_wire_bytes: usize,
    pub(in crate::commands::real_full) output_values: usize,
    pub(in crate::commands::real_full) output_finite_values: usize,
    pub(in crate::commands::real_full) output_nonzero_values: usize,
    pub(in crate::commands::real_full) output_checksum: f64,
    pub(in crate::commands::real_full) expected_real_executor_id: u64,
    pub(in crate::commands::real_full) response_executor_ids_observed: usize,
    pub(in crate::commands::real_full) real_executor_responses: usize,
    pub(in crate::commands::real_full) non_real_executor_responses: usize,
    pub(in crate::commands::real_full) all_responses_real_nvfp4: bool,
    pub(in crate::commands::real_full) passed: bool,
}

// Covered by request-shaped TCP fanout tests and used by TCP serving diagnostics.
#[allow(dead_code)]
pub(in crate::commands::real_full) async fn real_full_scheduler_sparse_tcp_dispatch_for_shape(
    shape: &RealFullSchedulerExecutionShape,
    targets: &[TcpProtocolV2HostBatchTarget],
    owner_lookup: Option<&ExpertOwnerLookup>,
    request_id_base: u64,
) -> Result<RealFullSchedulerSparseTcpDispatchProbe> {
    anyhow::ensure!(
        !targets.is_empty(),
        "real-full scheduler sparse TCP dispatch requires at least one target"
    );
    anyhow::ensure!(
        shape.prefill_tokens > 0,
        "real-full scheduler sparse TCP dispatch requires at least one prefill token"
    );
    anyhow::ensure!(
        shape.prefill_chunk_tokens > 0,
        "real-full scheduler sparse TCP dispatch requires a nonzero prefill chunk size"
    );
    anyhow::ensure!(
        shape.decode_rows > 0,
        "real-full scheduler sparse TCP dispatch requires at least one decode row"
    );

    let mut store =
        KvCacheBackingStore::new(KvCacheConfig::glm52_phase0(shape.reservation_tokens()));
    let reservation_id = store.reserve(shape.sequence_id.as_str(), shape.reservation_tokens())?;
    let policy = PrefillChunkPolicy {
        chunk_tokens: shape.prefill_chunk_tokens,
        max_prefill_tokens_per_iteration: shape.prefill_chunk_tokens,
        max_active_prefill_chunks: 1,
        decode_priority: true,
    };
    let graph_bucket =
        GraphBucket::new(shape.prefill_chunk_tokens + shape.decode_rows + shape.mtp_rows);
    let facts = ModelFacts::default();
    let quantization_recipe = facts.quantization_recipe.clone();
    let prefill_chunks = shape.prefill_tokens.div_ceil(shape.prefill_chunk_tokens);
    let scheduler_iterations_per_sparse_layer = prefill_chunks;
    let expected_real_executor_id = expert_protocol_v2_compact_id(REAL_NVFP4_PROTOCOL_V2_EXECUTOR);
    let mut probe = RealFullSchedulerSparseTcpDispatchProbe {
        status: "not-run",
        scope: "dispatch request-shaped admitted sparse scheduler batches through ProtocolV2 TCP host-batch-set fanout",
        sparse_layers: facts.num_hidden_layers - facts.first_k_dense_replace,
        scheduler_iterations_per_sparse_layer,
        sparse_batches: 0,
        host_batches: 0,
        global_rows: 0,
        host_rows: 0,
        routes: 0,
        request_wire_bytes: 0,
        response_wire_bytes: 0,
        output_values: 0,
        output_finite_values: 0,
        output_nonzero_values: 0,
        output_checksum: 0.0,
        expected_real_executor_id,
        response_executor_ids_observed: 0,
        real_executor_responses: 0,
        non_real_executor_responses: 0,
        all_responses_real_nvfp4: false,
        passed: false,
    };

    for layer_id in facts.first_k_dense_replace..facts.num_hidden_layers {
        let layer = LayerId(layer_id as u32);
        let mut decode_mtp = (0..shape.decode_rows)
            .map(|decode_offset| {
                LayerWave::decode(DecodeStep::new(
                    shape.request_id.as_str(),
                    shape.sequence_id.as_str(),
                    layer,
                    PositionId((shape.prefix_tokens + shape.prefill_tokens + decode_offset) as u64),
                    Some(reservation_id),
                    Priority(0),
                    shape.placement_version.as_str(),
                ))
            })
            .collect::<Vec<_>>();
        if shape.mtp_rows > 0 {
            decode_mtp.push(LayerWave::mtp_verify(MtpVerifyBlock::new(
                shape.request_id.as_str(),
                shape.sequence_id.as_str(),
                layer,
                PositionId((shape.prefix_tokens + shape.prefill_tokens + shape.decode_rows) as u64),
                shape.mtp_rows,
                Some(reservation_id),
                Priority(0),
                GraphBucket::new(shape.mtp_rows),
                shape.placement_version.as_str(),
            )));
        }
        let planned_prefill_chunks = plan_prefill_chunks(
            shape.request_id.as_str(),
            shape.sequence_id.as_str(),
            layer.0,
            shape.prefill_tokens,
            reservation_id,
            Priority(1),
            &policy,
            shape.placement_version.as_str(),
        );
        let planned_prefill_chunk_count = planned_prefill_chunks.len();
        for (chunk_index, prefill_chunk) in planned_prefill_chunks.into_iter().enumerate() {
            let final_prefill_chunk = chunk_index + 1 == planned_prefill_chunk_count;
            let mut waves =
                Vec::with_capacity(1 + usize::from(final_prefill_chunk) * decode_mtp.len());
            waves.push(LayerWave::prefill(prefill_chunk));
            if final_prefill_chunk {
                waves.append(&mut decode_mtp);
            }
            let mut iteration_policy = policy.clone();
            if final_prefill_chunk {
                iteration_policy.decode_priority = false;
            }
            let admission = admit_layerwaves_for_iteration(waves, &iteration_policy);
            anyhow::ensure!(
                admission.deferred.is_empty(),
                "combined prefill/decode/MTP sparse TCP dispatch unexpectedly deferred waves"
            );
            dispatch_admitted_sparse_batch_over_tcp(
                &admission.selected,
                graph_bucket,
                &quantization_recipe,
                targets,
                owner_lookup,
                request_id_base + probe.sparse_batches as u64 * 16,
                &mut probe,
            )
            .await
            .with_context(|| {
                format!(
                    "dispatching admitted prefill sparse batch final={final_prefill_chunk} for layer {layer_id}"
                )
            })?;
        }
        anyhow::ensure!(
            decode_mtp.is_empty(),
            "decode/MTP waves were not folded into the final prefill chunk"
        );
    }

    let expected_sparse_batches = (facts.num_hidden_layers - facts.first_k_dense_replace)
        * scheduler_iterations_per_sparse_layer;
    probe.passed = probe.sparse_batches == expected_sparse_batches
        && probe.host_batches > 0
        && probe.global_rows > 0
        && probe.host_rows > 0
        && probe.routes == probe.global_rows * facts.top_k * scheduler_route_replication_factor()?
        && probe.request_wire_bytes > 0
        && probe.response_wire_bytes > 0
        && probe.output_values == probe.global_rows * facts.hidden_size
        && probe.output_finite_values == probe.output_values
        && probe.output_nonzero_values > 0
        && probe.output_checksum.is_finite()
        && probe.response_executor_ids_observed == probe.host_batches;
    probe.all_responses_real_nvfp4 = probe.host_batches > 0
        && probe.response_executor_ids_observed == probe.host_batches
        && probe.real_executor_responses == probe.host_batches
        && probe.non_real_executor_responses == 0;
    probe.status = if probe.passed {
        "request-shaped-sparse-tcp-dispatch-passed"
    } else {
        "request-shaped-sparse-tcp-dispatch-blocked"
    };
    Ok(probe)
}

async fn dispatch_admitted_sparse_batch_over_tcp(
    selected: &[LayerWave],
    graph_bucket: GraphBucket,
    quantization_recipe: &str,
    targets: &[TcpProtocolV2HostBatchTarget],
    owner_lookup: Option<&ExpertOwnerLookup>,
    request_id_base: u64,
    probe: &mut RealFullSchedulerSparseTcpDispatchProbe,
) -> Result<()> {
    let Some(first_wave) = selected.first() else {
        return Ok(());
    };
    let mut batch = ExpertBatch::bf16_from_wave_with_envelope(
        first_wave,
        quantization_recipe.to_owned(),
        graph_bucket,
    )?;
    for wave in &selected[1..] {
        batch.try_append_wave(wave, DType::Bf16, quantization_recipe.to_owned())?;
    }
    let dispatch = real_full_scheduler_host_batch_set_tcp_dispatch(
        &batch,
        targets,
        owner_lookup,
        request_id_base,
    )
    .await?;
    anyhow::ensure!(
        dispatch.stats.global_rows == batch.num_rows(),
        "sparse TCP dispatch global rows {} did not match batch rows {}",
        dispatch.stats.global_rows,
        batch.num_rows()
    );
    anyhow::ensure!(
        dispatch.stats.routes == scheduler_dispatched_route_count(batch.route_count())?,
        "sparse TCP dispatch routes {} did not match batch routes {}",
        dispatch.stats.routes,
        batch.route_count()
    );
    probe.sparse_batches += 1;
    probe.host_batches += dispatch.stats.hosts;
    probe.global_rows += dispatch.stats.global_rows;
    probe.host_rows += dispatch.stats.host_rows;
    probe.routes += dispatch.stats.routes;
    probe.request_wire_bytes += dispatch.stats.request_wire_bytes;
    probe.response_wire_bytes += dispatch.stats.response_wire_bytes;
    probe.output_values += dispatch.stats.output_values;
    probe.output_finite_values += dispatch
        .accumulation
        .values
        .iter()
        .filter(|value| value.is_finite())
        .count();
    probe.output_nonzero_values += dispatch
        .accumulation
        .values
        .iter()
        .filter(|value| **value != 0.0)
        .count();
    probe.output_checksum += dispatch.stats.output_checksum;
    probe.response_executor_ids_observed += dispatch.stats.response_executor_ids.len();
    let real_responses = dispatch
        .stats
        .response_executor_ids
        .iter()
        .filter(|executor_id| **executor_id == probe.expected_real_executor_id)
        .count();
    probe.real_executor_responses += real_responses;
    probe.non_real_executor_responses += dispatch
        .stats
        .response_executor_ids
        .len()
        .saturating_sub(real_responses);
    Ok(())
}

fn request_frame_probe(request: &ExpertProtocolV2Request) -> Result<ProtocolV2FrameProbe> {
    let expected_wire_bytes = request.wire_stats().wire_bytes;
    let mut buffer = ExpertProtocolV2FrameBuffer::with_capacity(expected_wire_bytes);
    let (first_ptr, first_wire_bytes, first_decoded_matches) = {
        let frame = buffer.encode_request(request)?;
        (
            frame.as_ptr(),
            frame.len(),
            ExpertProtocolV2Request::decode(frame)? == *request,
        )
    };
    let first_capacity = buffer.capacity();
    let (second_ptr, second_wire_bytes, second_decoded_matches) = {
        let frame = buffer.encode_request(request)?;
        (
            frame.as_ptr(),
            frame.len(),
            ExpertProtocolV2Request::decode(frame)? == *request,
        )
    };
    Ok(ProtocolV2FrameProbe {
        wire_bytes: second_wire_bytes,
        capacity_bytes: buffer.capacity(),
        stable_allocation: first_ptr == second_ptr
            && first_capacity == buffer.capacity()
            && first_wire_bytes == expected_wire_bytes
            && second_wire_bytes == expected_wire_bytes,
        decoded_matches: first_decoded_matches && second_decoded_matches,
    })
}

fn host_partition_probe(
    batch: &ExpertBatch,
    routes: &[ExpertBatchRoute],
    wire_mode: HostWireEnvelopeMode,
) -> Result<HostPartitionProbe> {
    let set = scheduler_host_batch_set(batch, routes, None)?;
    let global_hidden_payload = match wire_mode {
        HostWireEnvelopeMode::ConstructRequests => Some(hidden_payload_for_batch(batch)),
        HostWireEnvelopeMode::CountOnly => None,
    };
    let graph_contract =
        graph_buffer_contract_for_batch(batch, batch.layer_id, LayerWaveMode::Prefill)?;
    let mut host_batches = 0_usize;
    let mut rows = 0_usize;
    let mut route_count = 0_usize;
    let mut expert_tiles = 0_usize;
    let mut graph_counts_valid = true;
    let mut request_frames = 0_usize;
    let mut request_rows = 0_usize;
    let mut request_routes = 0_usize;
    let mut request_payload_bytes = 0_usize;
    let mut request_wire_bytes = 0_usize;
    let mut response_frames = 0_usize;
    let mut response_rows = 0_usize;
    let mut response_payload_bytes = 0_usize;
    let mut response_wire_bytes = 0_usize;
    let mut wire_envelopes_valid = true;

    for (host_index, host_batch) in set.batches.iter().enumerate() {
        let counts = graph_contract.active_counts_for_host_batch(host_batch)?;
        host_batches += 1;
        rows += counts.rows;
        route_count += counts.routes;
        expert_tiles += counts.expert_tiles;
        graph_counts_valid &= counts.rows == host_batch.num_rows()
            && counts.routes == host_batch.route_count()
            && counts.expert_tiles <= graph_contract.workspace.max_expert_tiles;

        let expected_payload_bytes = host_batch.num_rows() * host_batch.hidden_bytes_per_row;
        let (
            request_row_entries,
            request_route_entries,
            request_logical_payload_bytes,
            request_wire_frame_bytes,
        ) = match wire_mode {
            HostWireEnvelopeMode::ConstructRequests => {
                let compact_hidden = host_batch.compact_hidden_payload(
                    global_hidden_payload
                        .as_ref()
                        .context("construct request mode missing global hidden payload")?,
                    set.global_row_count,
                )?;
                let request = ExpertProtocolV2Request::from_expert_host_batch(
                    REAL_FULL_PREFLIGHT_KV_RESERVATION_ID + host_index as u64,
                    host_batch,
                    compact_hidden,
                )?;
                let request_stats = request.wire_stats();
                (
                    request.rows.len(),
                    request.routes.len(),
                    request_stats.logical_payload_bytes,
                    request_stats.wire_bytes,
                )
            }
            HostWireEnvelopeMode::CountOnly => (
                host_batch.num_rows(),
                host_batch.route_count(),
                expected_payload_bytes,
                EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN
                    + host_batch.num_rows() * EXPERT_PROTOCOL_V2_ROW_DESCRIPTOR_LEN
                    + host_batch.route_count() * EXPERT_PROTOCOL_V2_ROUTE_ENTRY_LEN
                    + expected_payload_bytes,
            ),
        };
        let response_logical_payload_bytes = expected_payload_bytes;
        let response_wire_frame_bytes =
            EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN + response_logical_payload_bytes;

        request_frames += 1;
        request_rows += request_row_entries;
        request_routes += request_route_entries;
        request_payload_bytes += request_logical_payload_bytes;
        request_wire_bytes += request_wire_frame_bytes;
        response_frames += 1;
        response_rows += host_batch.num_rows();
        response_payload_bytes += response_logical_payload_bytes;
        response_wire_bytes += response_wire_frame_bytes;
        wire_envelopes_valid &= request_row_entries == host_batch.num_rows()
            && request_route_entries == host_batch.route_count()
            && request_logical_payload_bytes == expected_payload_bytes
            && request_wire_frame_bytes > request_logical_payload_bytes
            && response_logical_payload_bytes == expected_payload_bytes
            && response_wire_frame_bytes > response_logical_payload_bytes;
    }

    Ok(HostPartitionProbe {
        host_batches,
        rows,
        routes: route_count,
        expert_tiles,
        routes_match_global: route_count == scheduler_dispatched_route_count(batch.route_count())?,
        graph_counts_valid,
        request_frames,
        request_rows,
        request_routes,
        request_payload_bytes,
        request_wire_bytes,
        response_frames,
        response_rows,
        response_payload_bytes,
        response_wire_bytes,
        wire_envelopes_valid: wire_envelopes_valid
            && request_frames == host_batches
            && response_frames == host_batches
            && request_rows == rows
            && response_rows == rows
            && request_routes == route_count
            && request_payload_bytes == rows * batch.hidden_bytes_per_row
            && response_payload_bytes == rows * batch.hidden_bytes_per_row,
    })
}

fn graph_buffer_contract_for_batch(
    batch: &ExpertBatch,
    layer_id: LayerId,
    mode: LayerWaveMode,
) -> Result<ExpertGraphBufferContract> {
    let top_k = batch
        .rows
        .first()
        .map(|row| row.route_count)
        .context("expert graph contract requires at least one batch row")?;
    anyhow::ensure!(
        top_k > 0 && batch.rows.iter().all(|row| row.route_count == top_k),
        "expert graph contract requires one nonzero route count across all rows"
    );
    let mut facts = ModelFacts::default();
    facts.hidden_size = batch.hidden_dim;
    facts.routed_experts = batch.routed_experts;
    facts.top_k = top_k;
    facts.quantization_recipe = batch.quantization_recipe.clone();
    ExpertGraphBufferContract::for_model(
        &facts,
        layer_id,
        mode,
        batch.graph_bucket,
        batch.hidden_dtype.clone(),
        batch.quantization_recipe.clone(),
    )
    .map_err(Into::into)
}

fn scheduler_host_batch_set(
    batch: &ExpertBatch,
    routes: &[ExpertBatchRoute],
    _owner_lookup: Option<&ExpertOwnerLookup>,
) -> Result<ExpertHostBatchSet> {
    match ds4_tp4_scheduler_host_batch_set(batch, routes) {
        Ok(set) => Ok(set),
        // Small synthetic protocol fixtures deliberately do not carry a real
        // Flash/Pro shape, but must still exercise replicated TP4 transport.
        #[cfg(test)]
        Err(_) => Ok(ExpertHostBatchSet::tp4_from_expert_batch(
            batch,
            routes,
            &scheduler_expert_hosts(),
        )?),
        #[cfg(not(test))]
        Err(error) => Err(error),
    }
}

/// Build the fixed DS4 expert transport envelope. This boundary cannot select
/// owner/EP partitioning:
/// every row and every one of its six global routes is replicated to all four
/// Spark ranks so each rank can evaluate its quarter-width expert slabs.
#[cfg_attr(not(test), allow(dead_code))]
fn ds4_tp4_scheduler_host_batch_set(
    batch: &ExpertBatch,
    routes: &[ExpertBatchRoute],
) -> Result<ExpertHostBatchSet> {
    let expected_top_k = if batch.hidden_dim == DS4_FLASH_HIDDEN_SIZE
        && batch.routed_experts == DS4_FLASH_ROUTED_EXPERTS
        && (batch.quantization_recipe == "deepseek_v4_native_fp4_fp8_mixed_v1"
            || is_deepseek_v4_exl3_recipe(&batch.quantization_recipe))
    {
        DS4_FLASH_TOP_K
    } else if batch.hidden_dim == DS4_PRO_HIDDEN_SIZE
        && batch.routed_experts == DS4_PRO_ROUTED_EXPERTS
        && is_deepseek_v4_exl3_recipe(&batch.quantization_recipe)
    {
        DS4_PRO_TOP_K
    } else {
        anyhow::bail!(
            "DS4 TP4 transport does not support hidden={} experts={} recipe={:?}",
            batch.hidden_dim,
            batch.routed_experts,
            batch.quantization_recipe
        );
    };
    let expected_hidden_row_bytes = match &batch.hidden_dtype {
        DType::Bf16 => batch.hidden_dim * std::mem::size_of::<u16>(),
        DType::F4 => ExpertV2Dtype::Nvfp4E2m1Fp8E4m3.row_bytes(batch.hidden_dim)?,
        dtype => {
            anyhow::bail!("DS4 TP4 transport requires BF16 or NVFP4 hidden rows, got {dtype:?}")
        }
    };
    anyhow::ensure!(
        batch.hidden_bytes_per_row == expected_hidden_row_bytes,
        "DS4 TP4 transport requires compact {:?} rows, got hidden={} row_bytes={} expected={expected_hidden_row_bytes}",
        batch.hidden_dtype,
        batch.hidden_dim,
        batch.hidden_bytes_per_row,
    );
    anyhow::ensure!(
        batch
            .rows
            .iter()
            .all(|row| row.route_count == expected_top_k),
        "DS4 TP4 transport requires exactly {expected_top_k} global routes per row"
    );
    anyhow::ensure!(
        routes.len() == batch.num_rows() * expected_top_k,
        "DS4 TP4 transport route count {} did not match rows {} * top-k {expected_top_k}",
        routes.len(),
        batch.num_rows()
    );
    let hosts = scheduler_expert_hosts();
    anyhow::ensure!(
        hosts.len() == DS4_EXPERT_TP_WORLD_SIZE,
        "DS4 Flash TP4 transport requires {DS4_EXPERT_TP_WORLD_SIZE} Spark hosts, got {}",
        hosts.len()
    );
    let set = ExpertHostBatchSet::tp4_from_expert_batch(batch, routes, &hosts)?;
    anyhow::ensure!(
        set.num_hosts() == DS4_EXPERT_TP_WORLD_SIZE
            && set.host_row_count() == batch.num_rows() * DS4_EXPERT_TP_WORLD_SIZE
            && set.route_count() == routes.len() * DS4_EXPERT_TP_WORLD_SIZE,
        "DS4 Flash TP4 transport did not replicate the full batch to all four ranks"
    );
    for (rank, (host_batch, expected_host)) in set.batches.iter().zip(&hosts).enumerate() {
        anyhow::ensure!(
            host_batch.host == *expected_host
                && host_batch.num_rows() == batch.num_rows()
                && host_batch.routes == routes,
            "DS4 Flash TP4 rank {rank} did not receive the identical global route batch"
        );
    }
    Ok(set)
}

/// Inactive strict-TP4 TCP qualification boundary. It returns the four raw
/// BF16 hidden-width rank partials in stable Spark-rank order; it intentionally
/// performs no host accumulation or Spark-side reduction so the coordinator's
/// fixed FP32 reduction graph remains the sole merge point.
#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) async fn real_full_scheduler_ds4_flash_tp4_tcp_bf16_partials(
    batch: &ExpertBatch,
    routes: &[ExpertBatchRoute],
    global_hidden_payload: &[u8],
    targets: &[TcpProtocolV2HostBatchTarget],
    request_id_base: u64,
) -> Result<TcpProtocolV2HostBatchSetBf16PayloadDispatch> {
    let set = ds4_tp4_scheduler_host_batch_set(batch, routes)?;
    anyhow::ensure!(
        targets.len() == DS4_EXPERT_TP_WORLD_SIZE,
        "DS4 Flash TP4 transport requires {DS4_EXPERT_TP_WORLD_SIZE} targets, got {}",
        targets.len()
    );
    for host in set.touched_hosts() {
        let matches = targets.iter().filter(|target| target.host == host).count();
        anyhow::ensure!(
            matches == 1,
            "DS4 Flash TP4 transport requires exactly one target for {host}, got {matches}"
        );
    }
    let dispatch = tcp_protocol_v2_host_batch_set_bf16_payload_dispatch(
        &set,
        global_hidden_payload,
        targets,
        request_id_base,
        real_full_protocol_v2_transport_config()?,
    )
    .await?;
    validate_ds4_flash_tp4_tcp_bf16_partials(batch, routes, &dispatch)?;
    Ok(dispatch)
}

/// Persistent-client form used by the live scheduler worker. The envelope is
/// still built by the strict TP4 boundary, so connection reuse cannot turn
/// native Flash dispatch back into owner/EP partitioning.
pub(in crate::commands::real_full) async fn real_full_scheduler_ds4_flash_tp4_tcp_bf16_partials_persistent(
    client: &mut TcpProtocolV2HostBatchSetPersistentClient,
    batch: &ExpertBatch,
    routes: &[ExpertBatchRoute],
    global_hidden_payload: &[u8],
    targets: &[TcpProtocolV2HostBatchTarget],
    request_id_base: u64,
) -> Result<TcpProtocolV2HostBatchSetBf16PayloadDispatch> {
    let set = ds4_tp4_scheduler_host_batch_set(batch, routes)?;
    anyhow::ensure!(
        targets.len() == DS4_EXPERT_TP_WORLD_SIZE,
        "DS4 Flash TP4 transport requires {DS4_EXPERT_TP_WORLD_SIZE} targets, got {}",
        targets.len()
    );
    for host in set.touched_hosts() {
        let matches = targets.iter().filter(|target| target.host == host).count();
        anyhow::ensure!(
            matches == 1,
            "DS4 Flash TP4 transport requires exactly one target for {host}, got {matches}"
        );
    }
    let dispatch = client
        .dispatch_bf16_payload(&set, global_hidden_payload, request_id_base)
        .await?;
    validate_ds4_flash_tp4_tcp_bf16_partials(batch, routes, &dispatch)?;
    Ok(dispatch)
}

fn validate_ds4_flash_tp4_tcp_bf16_partials(
    batch: &ExpertBatch,
    routes: &[ExpertBatchRoute],
    dispatch: &TcpProtocolV2HostBatchSetBf16PayloadDispatch,
) -> Result<()> {
    let rank_payload_bytes = batch
        .num_rows()
        .checked_mul(batch.hidden_bytes_per_row)
        .context("DS4 Flash TP4 rank response byte count overflow")?;
    let expected_rows = (0..batch.num_rows()).collect::<Vec<_>>();
    anyhow::ensure!(
        dispatch.stats.hosts == DS4_EXPERT_TP_WORLD_SIZE
            && dispatch.stats.host_rows == batch.num_rows() * DS4_EXPERT_TP_WORLD_SIZE
            && dispatch.stats.routes == routes.len() * DS4_EXPERT_TP_WORLD_SIZE
            && dispatch.stats.contribution_counts
                == vec![DS4_EXPERT_TP_WORLD_SIZE; batch.num_rows()]
            && dispatch.partial_outputs_bf16_by_host.len() == DS4_EXPERT_TP_WORLD_SIZE
            && dispatch
                .partial_outputs_bf16_by_host
                .iter()
                .all(|payload| payload.len() == rank_payload_bytes)
            && dispatch.global_row_indices_by_host.len() == DS4_EXPERT_TP_WORLD_SIZE
            && dispatch
                .global_row_indices_by_host
                .iter()
                .all(|rows| rows == &expected_rows),
        "DS4 Flash TP4 transport did not return four complete hidden-width rank partials"
    );
    Ok(())
}

/// Strict TP4 verbs path. Every Spark receives the identical global route
/// batch. The row-count policy decides whether the returned BF16 chunks are
/// four raw rank partials or the completed rows from a Spark-side reduction.
#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) async fn real_full_scheduler_ds4_flash_tp4_verbs_host_bf16_partial_chunks(
    client: &VerbsHostProtocolV2HostBatchSetPersistentClient,
    batch: &ExpertBatch,
    routes: &[ExpertBatchRoute],
    global_hidden_payload: Bytes,
    request_id_base: u64,
    force_raw_rank_partials: bool,
    chunk_tx: std::sync::mpsc::Sender<VerbsHostProtocolV2HostBatchSetBf16PayloadChunk>,
) -> Result<TcpProtocolV2HostBatchSetDispatchStats> {
    let set = ds4_tp4_scheduler_host_batch_set(batch, routes)?;
    anyhow::ensure!(
        global_hidden_payload.len() == batch.num_rows() * batch.hidden_bytes_per_row,
        "DS4 Flash TP4 verbs hidden payload bytes {} did not match rows {} * row bytes {}",
        global_hidden_payload.len(),
        batch.num_rows(),
        batch.hidden_bytes_per_row
    );
    let reduction = if force_raw_rank_partials {
        None
    } else {
        spark_expert_reduction_dispatch_for_batch(batch)?
    };
    let reduced_root_host_index = reduction.map(|reduction| reduction.root_rank);
    let owner_fanout = reduction.is_some_and(|reduction| reduction.owner_fanout);
    let row_sharded_reduction = reduction.is_some_and(|reduction| reduction.row_sharded);
    // Honor Protocol V2's chunked-ingress policy for large prefill frames.
    // The Spark reassembles logical rows before one joint native TP4 launch;
    // chunking transport ingress never serializes expert compute.
    let stats = client
        .dispatch_bf16_payload_streaming_shared(
            &set,
            global_hidden_payload,
            request_id_base,
            ExpertV2Dtype::Bf16,
            reduced_root_host_index,
            owner_fanout,
            row_sharded_reduction,
            chunk_tx,
        )
        .await?;
    let expected_response_hosts = if owner_fanout {
        1
    } else {
        DS4_EXPERT_TP_WORLD_SIZE
    };
    let expected_host_rows = if owner_fanout {
        batch.num_rows()
    } else {
        batch.num_rows() * DS4_EXPERT_TP_WORLD_SIZE
    };
    anyhow::ensure!(
        stats.hosts == expected_response_hosts
            && stats.global_rows == batch.num_rows()
            && stats.host_rows == expected_host_rows
            && stats.routes == routes.len() * DS4_EXPERT_TP_WORLD_SIZE
            && stats.output_dim == batch.hidden_dim
            && stats.output_values == batch.num_rows() * batch.hidden_dim
            && stats.response_executor_ids.len() == expected_response_hosts
            && stats.contribution_counts.is_empty(),
        "DS4 Flash TP4 verbs dispatch stats do not match the selected reduction topology"
    );
    Ok(stats)
}

/// Nonblocking one-row launch for the same strict TP4 contract. This preserves
/// the GLMRT overlap point: Spark work is enqueued before coordinator shared
/// expert work, and the pending handle is joined only at the reduction barrier.
#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) fn real_full_scheduler_ds4_tp4_try_start_verbs_raw(
    client: &VerbsHostProtocolV2HostBatchSetPersistentClient,
    batch: &ExpertBatch,
    routes: &[ExpertBatchRoute],
    global_hidden_payload: Vec<u8>,
    request_id_base: u64,
) -> Result<VerbsHostProtocolV2HostBatchSetPayloadStart> {
    anyhow::ensure!(
        batch.num_rows() == 1,
        "DS4 TP4 direct verbs raw dispatch requires one row, got {}",
        batch.num_rows()
    );
    let set = ds4_tp4_scheduler_host_batch_set(batch, routes)?;
    let expected_hidden_bytes = batch
        .num_rows()
        .checked_mul(batch.hidden_bytes_per_row)
        .context("DS4 TP4 direct verbs raw hidden byte count overflow")?;
    anyhow::ensure!(
        global_hidden_payload.len() == expected_hidden_bytes,
        "DS4 TP4 direct verbs raw hidden payload bytes {} did not match {} rows of {} bytes",
        global_hidden_payload.len(),
        batch.num_rows(),
        batch.hidden_bytes_per_row
    );
    client.try_start_host_batch_set_payload(
        set,
        global_hidden_payload,
        request_id_base,
        ExpertV2Dtype::Bf16,
    )
}

#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) fn real_full_scheduler_ds4_tp4_finish_verbs_raw(
    pending: VerbsHostProtocolV2ReducedIdentityPayloadPending,
    rows: usize,
    logical_routes: usize,
    hidden_size: usize,
) -> Result<(
    TcpProtocolV2HostBatchSetDispatchStats,
    Vec<VerbsHostProtocolV2HostBatchSetBf16PayloadChunk>,
)> {
    let (stats, chunks) = pending
        .finish()
        .context("finishing strict DS4 TP4 direct verbs raw dispatch")?;
    anyhow::ensure!(
        rows == 1 && logical_routes > 0 && hidden_size > 0,
        "DS4 TP4 direct verbs raw completion requires one row and nonzero routes and hidden size"
    );
    let expected_host_rows = rows
        .checked_mul(DS4_EXPERT_TP_WORLD_SIZE)
        .context("DS4 TP4 direct verbs raw host row count overflow")?;
    let expected_routes = logical_routes
        .checked_mul(DS4_EXPERT_TP_WORLD_SIZE)
        .context("DS4 TP4 direct verbs raw route count overflow")?;
    let expected_output_values = rows
        .checked_mul(hidden_size)
        .context("DS4 TP4 direct verbs raw output value count overflow")?;
    anyhow::ensure!(
        stats.hosts == DS4_EXPERT_TP_WORLD_SIZE
            && stats.global_rows == rows
            && stats.host_rows == expected_host_rows
            && stats.routes == expected_routes
            && stats.output_dim == hidden_size
            && stats.output_values == expected_output_values
            && stats.response_executor_ids.len() == DS4_EXPERT_TP_WORLD_SIZE
            && stats.contribution_counts.is_empty(),
        "DS4 TP4 direct verbs dispatch did not return four raw rank streams"
    );
    ds4_tp4_verbs_chunk_views(&chunks, rows, hidden_size)?;
    Ok((stats, chunks))
}

/// Bridge a completed strict-TP4 TCP barrier into the coordinator's fixed GPU0
/// receive pointers. This remains outside CUDA capture and does not reduce the
/// rank payloads; the caller launches the ordered coordinator reduction next.
#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) fn real_full_scheduler_ds4_flash_tp4_stage_tcp_partials(
    arena: &mut Ds4FlashTp4CoordinatorReductionArena,
    dispatch: &TcpProtocolV2HostBatchSetBf16PayloadDispatch,
    rows: usize,
    shared_delta: Ds4rtDeviceBuffer,
) -> Result<Ds4FlashTp4ReductionDeviceBuffers> {
    let hidden_size = arena.hidden_size();
    anyhow::ensure!(
        dispatch.stats.hosts == DS4_EXPERT_TP_WORLD_SIZE
            && dispatch.stats.global_rows == rows
            && dispatch.stats.host_rows == rows * DS4_EXPERT_TP_WORLD_SIZE
            && dispatch.stats.output_dim == hidden_size
            && dispatch.stats.output_values == rows * hidden_size
            && dispatch.stats.contribution_counts == vec![DS4_EXPERT_TP_WORLD_SIZE; rows],
        "DS4 Flash TP4 TCP dispatch stats do not describe four complete hidden-width rank partials"
    );
    arena.stage_host_partials(
        rows,
        &dispatch.partial_outputs_bf16_by_host,
        &dispatch.global_row_indices_by_host,
        shared_delta,
    )
}

fn ds4_tp4_verbs_chunk_views<'a>(
    chunks: &'a [VerbsHostProtocolV2HostBatchSetBf16PayloadChunk],
    rows: usize,
    hidden_size: usize,
) -> Result<Vec<Ds4FlashTp4HostPartialChunk<'a>>> {
    anyhow::ensure!(
        !chunks.is_empty(),
        "DS4 TP4 verbs barrier returned no response chunks"
    );
    let row_bytes = hidden_size * std::mem::size_of::<u16>();
    let mut completed_rows = vec![false; rows];
    let mut views = Vec::with_capacity(chunks.len());
    for (chunk_index, chunk) in chunks.iter().enumerate() {
        anyhow::ensure!(
            chunk.host_index < DS4_EXPERT_TP_WORLD_SIZE,
            "DS4 TP4 verbs chunk {chunk_index} has invalid rank {}",
            chunk.host_index
        );
        anyhow::ensure!(
            chunk.output_dtype == ExpertV2Dtype::Bf16 && chunk.output_row_stride_bytes == row_bytes,
            "DS4 TP4 verbs chunk {chunk_index} rank {} must be hidden-width BF16",
            chunk.host_index
        );
        anyhow::ensure!(
            chunk.partial_output.as_ref().len() == chunk.global_row_indices.len() * row_bytes,
            "DS4 TP4 verbs chunk {chunk_index} rank {} payload bytes do not match its row map",
            chunk.host_index
        );
        for &row in &chunk.completed_global_row_indices {
            let completed = completed_rows
                .get_mut(row)
                .with_context(|| format!("DS4 TP4 verbs chunk completed out-of-range row {row}"))?;
            anyhow::ensure!(
                !*completed,
                "DS4 TP4 verbs barrier completed row {row} more than once"
            );
            *completed = true;
        }
        views.push(Ds4FlashTp4HostPartialChunk {
            rank: chunk.host_index,
            payload_bf16: chunk.partial_output.as_ref(),
            pinned_payload: chunk.partial_output.pinned_host_buffer(),
            global_row_indices: &chunk.global_row_indices,
        });
    }
    if let Some(row) = completed_rows.iter().position(|completed| !*completed) {
        anyhow::bail!("DS4 TP4 verbs barrier did not complete row {row}");
    }
    Ok(views)
}

/// Land striped verbs chunks into the same four fixed GPU0 rank pointers used
/// by TCP, then complete the ordered reduction before the recyclable response
/// frames leave this borrow.
#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) fn real_full_scheduler_ds4_flash_tp4_reduce_verbs_chunks_synchronized(
    arena: &mut Ds4FlashTp4CoordinatorReductionArena,
    chunks: &[VerbsHostProtocolV2HostBatchSetBf16PayloadChunk],
    rows: usize,
    shared_delta: Ds4rtDeviceBuffer,
) -> Result<Ds4rtDeviceBuffer> {
    let views = ds4_tp4_verbs_chunk_views(chunks, rows, arena.hidden_size())?;
    arena.stage_host_partial_chunks_and_reduce_synchronized(rows, &views, shared_delta)
}

/// Preserve the raw-rank BF16 order while handing reduction completion to the
/// native target stream instead of synchronizing the coordinator host.  The
/// owned chunks remain attached to the output until its CUDA event completes.
pub(in crate::commands::real_full) fn real_full_scheduler_ds4_flash_tp4_reduce_verbs_chunks_async_owned(
    arena: &mut Ds4FlashTp4CoordinatorReductionArena,
    chunks: Vec<VerbsHostProtocolV2HostBatchSetBf16PayloadChunk>,
    rows: usize,
    shared_delta: Ds4rtDeviceBuffer,
) -> Result<DeviceBf16Output> {
    let views = ds4_tp4_verbs_chunk_views(&chunks, rows, arena.hidden_size())?;
    let mut output =
        arena.stage_host_partial_chunks_and_reduce_async_owned(rows, &views, shared_delta)?;
    output.retain_host_dependency_until_ready(chunks);
    Ok(output)
}

struct Ds4Tp4SegmentChunk<'a> {
    rank: usize,
    payload_bf16: &'a [u8],
    pinned_payload: Option<Ds4rtHostBuffer>,
    global_row_indices: Vec<usize>,
}

fn ds4_tp4_verbs_chunk_segments<'a>(
    chunks: &'a [VerbsHostProtocolV2HostBatchSetBf16PayloadChunk],
    segment_row_start: usize,
    rows: usize,
    hidden_size: usize,
) -> Result<Vec<Ds4Tp4SegmentChunk<'a>>> {
    anyhow::ensure!(rows > 0, "DS4 TP4 verbs segment cannot be empty");
    let segment_row_end = segment_row_start
        .checked_add(rows)
        .context("DS4 TP4 verbs segment row range overflow")?;
    let row_bytes = hidden_size * std::mem::size_of::<u16>();
    let mut completed_rows = vec![false; rows];
    let mut segments = Vec::new();
    for (chunk_index, chunk) in chunks.iter().enumerate() {
        anyhow::ensure!(
            chunk.host_index < DS4_EXPERT_TP_WORLD_SIZE,
            "DS4 TP4 verbs chunk {chunk_index} has invalid rank {}",
            chunk.host_index
        );
        anyhow::ensure!(
            chunk.output_dtype == ExpertV2Dtype::Bf16 && chunk.output_row_stride_bytes == row_bytes,
            "DS4 TP4 verbs chunk {chunk_index} rank {} must be hidden-width BF16",
            chunk.host_index
        );
        anyhow::ensure!(
            chunk.partial_output.as_ref().len() == chunk.global_row_indices.len() * row_bytes,
            "DS4 TP4 verbs chunk {chunk_index} rank {} payload bytes do not match its row map",
            chunk.host_index
        );
        for &global_row in &chunk.completed_global_row_indices {
            if !(segment_row_start..segment_row_end).contains(&global_row) {
                continue;
            }
            let local_row = global_row - segment_row_start;
            let completed = &mut completed_rows[local_row];
            anyhow::ensure!(
                !*completed,
                "DS4 TP4 verbs segment completed row {local_row} more than once"
            );
            *completed = true;
        }
        let full_pinned = chunk.partial_output.pinned_host_buffer();
        if let Some(pinned) = full_pinned {
            anyhow::ensure!(
                !pinned.ptr.is_null() && pinned.bytes == chunk.partial_output.as_ref().len(),
                "DS4 TP4 verbs chunk {chunk_index} pinned view is inconsistent"
            );
        }
        let mut local_start = 0_usize;
        while local_start < chunk.global_row_indices.len() {
            while local_start < chunk.global_row_indices.len()
                && !(segment_row_start..segment_row_end)
                    .contains(&chunk.global_row_indices[local_start])
            {
                local_start += 1;
            }
            if local_start == chunk.global_row_indices.len() {
                break;
            }
            let mut local_end = local_start + 1;
            while local_end < chunk.global_row_indices.len()
                && (segment_row_start..segment_row_end)
                    .contains(&chunk.global_row_indices[local_end])
            {
                local_end += 1;
            }
            let byte_start = local_start
                .checked_mul(row_bytes)
                .context("DS4 TP4 verbs segment payload offset overflow")?;
            let byte_end = local_end
                .checked_mul(row_bytes)
                .context("DS4 TP4 verbs segment payload end overflow")?;
            let payload_bf16 = &chunk.partial_output.as_ref()[byte_start..byte_end];
            let pinned_payload = full_pinned.map(|pinned| Ds4rtHostBuffer {
                ptr: unsafe { pinned.ptr.cast::<u8>().add(byte_start).cast() },
                bytes: payload_bf16.len(),
                flags: pinned.flags,
            });
            let global_row_indices = chunk.global_row_indices[local_start..local_end]
                .iter()
                .map(|global_row| global_row - segment_row_start)
                .collect();
            segments.push(Ds4Tp4SegmentChunk {
                rank: chunk.host_index,
                payload_bf16,
                pinned_payload,
                global_row_indices,
            });
            local_start = local_end;
        }
    }
    if let Some(row) = completed_rows.iter().position(|completed| !*completed) {
        anyhow::bail!("DS4 TP4 verbs segment did not complete row {row}");
    }
    Ok(segments)
}

/// Reduce one request-local row segment directly from a jointly issued raw
/// TP4 verbs barrier. Payloads remain borrowed from the recyclable pinned
/// response frames through the arena synchronization; only tiny row maps are
/// rebased per member.
pub(in crate::commands::real_full) fn real_full_scheduler_ds4_flash_tp4_reduce_verbs_chunk_segment_synchronized(
    arena: &mut Ds4FlashTp4CoordinatorReductionArena,
    chunks: &[VerbsHostProtocolV2HostBatchSetBf16PayloadChunk],
    segment_row_start: usize,
    rows: usize,
    shared_delta: Ds4rtDeviceBuffer,
) -> Result<Ds4rtDeviceBuffer> {
    let segments =
        ds4_tp4_verbs_chunk_segments(chunks, segment_row_start, rows, arena.hidden_size())?;
    let views = segments
        .iter()
        .map(|segment| Ds4FlashTp4HostPartialChunk {
            rank: segment.rank,
            payload_bf16: segment.payload_bf16,
            pinned_payload: segment.pinned_payload,
            global_row_indices: &segment.global_row_indices,
        })
        .collect::<Vec<_>>();
    arena.stage_host_partial_chunks_and_reduce_synchronized(rows, &views, shared_delta)
}

pub(super) fn scheduler_route_replication_factor() -> Result<usize> {
    anyhow::ensure!(
        EXPERT_HOSTS.len() == DS4_EXPERT_TP_WORLD_SIZE,
        "DS4 expert TP4 requires {DS4_EXPERT_TP_WORLD_SIZE} fixed Spark hosts, got {}",
        EXPERT_HOSTS.len()
    );
    Ok(DS4_EXPERT_TP_WORLD_SIZE)
}

pub(super) fn scheduler_dispatched_route_count(logical_routes: usize) -> Result<usize> {
    logical_routes
        .checked_mul(scheduler_route_replication_factor()?)
        .context("scheduler dispatched route count overflow")
}

fn scheduler_expert_hosts() -> Vec<String> {
    EXPERT_HOSTS.iter().map(|host| (*host).to_owned()).collect()
}

fn response_reconstruction_probe(
    batch: &ExpertBatch,
    response: &ExpertProtocolV2Response,
) -> Result<ProtocolV2ReconstructionProbe> {
    let mut buffer = ExpertProtocolV2FrameBuffer::with_capacity(response.wire_stats().wire_bytes);
    let decoded = {
        let frame = buffer.encode_response(response)?;
        ExpertProtocolV2Response::decode(frame)?
    };
    let mut chunks = decoded
        .partial_output_payload
        .chunks_exact(batch.hidden_bytes_per_row);
    let row_payloads = chunks.by_ref().collect::<Vec<_>>();
    let no_remainder = chunks.remainder().is_empty();
    let reconstructed = batch.reconstruct_partial_outputs(&row_payloads)?;

    let row_order_matches = reconstructed.len() == batch.num_rows()
        && reconstructed
            .iter()
            .zip(batch.rows.iter())
            .all(|((actual_row, _), expected_row)| actual_row == expected_row);
    let payload_matches = no_remainder
        && reconstructed
            .iter()
            .zip(row_payloads.iter())
            .all(|((_, payload), encoded)| {
                payload.len() == batch.hidden_bytes_per_row && *payload == *encoded
            });

    Ok(ProtocolV2ReconstructionProbe {
        rows: reconstructed.len(),
        payload_bytes: row_payloads.len() * batch.hidden_bytes_per_row,
        row_order_matches,
        payload_matches,
    })
}

fn response_frame_probe(response: &ExpertProtocolV2Response) -> Result<ProtocolV2FrameProbe> {
    let expected_wire_bytes = response.wire_stats().wire_bytes;
    let mut buffer = ExpertProtocolV2FrameBuffer::with_capacity(expected_wire_bytes);
    let (first_ptr, first_wire_bytes, first_decoded_matches) = {
        let frame = buffer.encode_response(response)?;
        (
            frame.as_ptr(),
            frame.len(),
            ExpertProtocolV2Response::decode(frame)? == *response,
        )
    };
    let first_capacity = buffer.capacity();
    let (second_ptr, second_wire_bytes, second_decoded_matches) = {
        let frame = buffer.encode_response(response)?;
        (
            frame.as_ptr(),
            frame.len(),
            ExpertProtocolV2Response::decode(frame)? == *response,
        )
    };
    Ok(ProtocolV2FrameProbe {
        wire_bytes: second_wire_bytes,
        capacity_bytes: buffer.capacity(),
        stable_allocation: first_ptr == second_ptr
            && first_capacity == buffer.capacity()
            && first_wire_bytes == expected_wire_bytes
            && second_wire_bytes == expected_wire_bytes,
        decoded_matches: first_decoded_matches && second_decoded_matches,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        core_routes_for_batch, ds4_tp4_scheduler_host_batch_set, ds4_tp4_verbs_chunk_segments,
        ds4_tp4_verbs_chunk_views, parse_moe_response_dtype, real_full_protocol_v2_batch_probe,
        real_full_scheduler_ds4_flash_tp4_tcp_bf16_partials,
        real_full_scheduler_host_batch_set_tcp_dispatch,
        real_full_scheduler_host_batch_set_tcp_dispatch_with_payload,
        real_full_scheduler_sparse_tcp_dispatch_for_shape, scheduler_host_batch_set,
        scored_routes_for_scheduler_batch, DEEPSEEK_V4_EXL3_RECIPE,
    };
    use crate::cli::ExpertDaemonArgs;
    use crate::commands::expertd::run_expertd;
    use crate::commands::model_artifacts::read_expert_owner_lookup;
    use crate::commands::real_full::constants::{
        REAL_FULL_PREFLIGHT_DECODE_ROWS, REAL_FULL_PREFLIGHT_MTP_ROWS,
        REAL_FULL_PREFLIGHT_PREFILL_ROWS,
    };
    use crate::commands::real_full::expert_probe::{
        RealNvfp4ProtocolV2Executor, REAL_NVFP4_PROTOCOL_V2_EXECUTOR,
    };
    use crate::commands::real_full::sampling::RealFullLmHeadSamplingOptions;
    use crate::commands::real_full::scheduler::RealFullSchedulerExecutionShape;
    use crate::commands::real_full::sparse_mlp::router::ScoredRoute;
    use anyhow::Result;
    use ds4rt_core::{
        DType, ExpertBatch, ExpertHostBatchSet, ExpertOwnerLookup, GraphBucket, ModelFacts,
        RowSourceKind, TensorCatalog, DS4_EXPERT_TP_WORLD_SIZE, DS4_FLASH_HIDDEN_BF16_BYTES,
        DS4_FLASH_HIDDEN_SIZE, DS4_FLASH_NUM_HIDDEN_LAYERS, DS4_FLASH_ROUTED_EXPERTS,
        DS4_FLASH_TOP_K, DS4_PRO_HIDDEN_SIZE, DS4_PRO_ROUTED_EXPERTS, EXPERT_HOSTS,
        GLM52_FIRST_K_DENSE_REPLACE, GLM52_NUM_HIDDEN_LAYERS,
    };
    use ds4rt_loader::DEEPSEEK_V4_EXL3_RECIPE_V2;
    use ds4rt_transport::{
        expert_protocol_v2_compact_id, serve_protocol_v2_tcp_listener_with_executor,
        tcp_protocol_v2_host_batch_set_bf16_dispatch, ExpertProtocolV2RequestView,
        ExpertProtocolV2Response, ExpertV2Dtype, ProtocolV2ExpertExecutor, SyntheticRouteExecutor,
        TcpProtocolV2HostBatchTarget, TcpTransportConfig,
        VerbsHostProtocolV2HostBatchSetBf16PayloadChunk, VerbsHostProtocolV2ResponsePayload,
        PROTOCOL_V2_SYNTHETIC_ROUTE_EXECUTOR,
    };
    use std::{
        fs::File,
        net::{SocketAddr, TcpListener as StdTcpListener},
        path::{Path, PathBuf},
        sync::Arc,
    };
    use tokio::{
        net::{TcpListener, TcpStream},
        time::{sleep, Duration},
    };

    struct RankedSyntheticExecutor {
        name: &'static str,
    }

    impl ProtocolV2ExpertExecutor for RankedSyntheticExecutor {
        fn name(&self) -> &'static str {
            self.name
        }

        fn execute(
            &self,
            request: &ExpertProtocolV2RequestView<'_>,
        ) -> Result<ExpertProtocolV2Response> {
            ProtocolV2ExpertExecutor::execute(&SyntheticRouteExecutor, request)
        }
    }

    #[test]
    fn moe_response_dtype_parser_supports_adaptive_owner_codecs() {
        assert_eq!(
            parse_moe_response_dtype("test", "bf16").unwrap(),
            ExpertV2Dtype::Bf16
        );
        assert_eq!(
            parse_moe_response_dtype("test", "fp8").unwrap(),
            ExpertV2Dtype::Fp8E4m3RowScaled
        );
        assert!(parse_moe_response_dtype("test", "f32").is_err());
    }

    #[test]
    fn scheduler_host_batch_set_replicates_full_first_sparse_batch() -> Result<()> {
        let batch = first_sparse_scheduler_batch()?;
        let routes = core_routes_for_batch(&batch);

        let set = scheduler_host_batch_set(&batch, &routes, None)?;

        assert_eq!(
            batch.layer_id.0 as usize,
            ModelFacts::default().first_k_dense_replace
        );
        assert_eq!(batch.num_rows(), 521);
        assert_eq!(batch.hidden_dim, DS4_FLASH_HIDDEN_SIZE);
        assert_eq!(routes.len(), batch.num_rows() * DS4_FLASH_TOP_K);
        assert_eq!(set.num_hosts(), 4);
        assert_eq!(set.global_row_count, batch.num_rows());
        assert_eq!(set.host_row_count(), 2_084);
        assert_eq!(set.route_count(), routes.len() * DS4_EXPERT_TP_WORLD_SIZE);
        set.reconstruction_plan
            .validate_for_batches(&set.batches, set.global_row_count)?;

        let probe = real_full_protocol_v2_batch_probe(batch.layer_id.0 as usize, &batch)?;
        assert!(probe.passed);
        assert_eq!(probe.host_batch_rows, set.host_row_count());
        assert_eq!(probe.host_batch_routes, set.route_count());
        assert_eq!(probe.host_request_frames, set.num_hosts());
        assert_eq!(probe.host_request_rows, set.host_row_count());
        assert_eq!(probe.host_request_routes, set.route_count());
        assert_eq!(
            probe.host_request_payload_bytes,
            set.host_row_count() * DS4_FLASH_HIDDEN_BF16_BYTES
        );
        assert!(probe.host_request_wire_bytes > probe.host_request_payload_bytes);
        assert_eq!(probe.host_response_frames, set.num_hosts());
        assert_eq!(probe.host_response_rows, set.host_row_count());
        assert_eq!(
            probe.host_response_payload_bytes,
            set.host_row_count() * DS4_FLASH_HIDDEN_BF16_BYTES
        );
        assert!(probe.host_response_wire_bytes > probe.host_response_payload_bytes);
        assert!(probe.host_wire_envelopes_valid);
        eprintln!(
            "scheduler_host_batch_set_tp4_full_first_sparse_batch layer={} rows={} routes={} hidden_dim={} hosts={} host_rows={} host_routes={} request_wire_bytes={} response_wire_bytes={} host_request_wire_bytes={} host_response_wire_bytes={}",
            batch.layer_id.0,
            batch.num_rows(),
            routes.len(),
            batch.hidden_dim,
            set.num_hosts(),
            set.host_row_count(),
            set.route_count(),
            probe.request_wire_bytes,
            probe.response_wire_bytes,
            probe.host_request_wire_bytes,
            probe.host_response_wire_bytes
        );
        Ok(())
    }

    #[test]
    fn ds4_flash_tp4_transport_ignores_ep_owner_lookup_and_replicates_every_route() -> Result<()> {
        let mut batch = reduced_mixed_scheduler_batch()?;
        batch.hidden_dim = DS4_FLASH_HIDDEN_SIZE;
        batch.hidden_bytes_per_row = DS4_FLASH_HIDDEN_BF16_BYTES;
        let routes = core_routes_for_batch(&batch);
        let owner_lookup =
            ExpertOwnerLookup::from_pairs((0..DS4_FLASH_ROUTED_EXPERTS).map(|expert_id| {
                (
                    (batch.layer_id.0 as usize, expert_id),
                    EXPERT_HOSTS[0].to_owned(),
                )
            }));

        let set = scheduler_host_batch_set(&batch, &routes, Some(&owner_lookup))?;

        assert_eq!(set.num_hosts(), 4);
        assert_eq!(set.host_row_count(), batch.num_rows() * 4);
        assert_eq!(set.route_count(), routes.len() * 4);
        assert_eq!(
            set.touched_hosts().collect::<Vec<_>>(),
            EXPERT_HOSTS.to_vec()
        );
        assert!(set.batches.iter().all(|host_batch| {
            host_batch.num_rows() == batch.num_rows() && host_batch.routes == routes
        }));
        assert!(set
            .batches
            .iter()
            .flat_map(|host_batch| &host_batch.routes)
            .all(|route| route.expert_id < DS4_FLASH_ROUTED_EXPERTS));
        Ok(())
    }

    #[test]
    fn ds4_flash_tp4_transport_accepts_compact_nvfp4_ingress() -> Result<()> {
        let mut batch = reduced_mixed_scheduler_batch()?;
        batch.hidden_dim = DS4_FLASH_HIDDEN_SIZE;
        batch.hidden_bytes_per_row =
            ExpertV2Dtype::Nvfp4E2m1Fp8E4m3.row_bytes(DS4_FLASH_HIDDEN_SIZE)?;
        batch.hidden_dtype = DType::F4;
        let routes = core_routes_for_batch(&batch);

        let set = ds4_tp4_scheduler_host_batch_set(&batch, &routes)?;

        assert_eq!(set.num_hosts(), DS4_EXPERT_TP_WORLD_SIZE);
        assert_eq!(
            set.host_row_count(),
            batch.num_rows() * DS4_EXPERT_TP_WORLD_SIZE
        );
        assert!(set.batches.iter().all(|host_batch| {
            host_batch.hidden_dtype == DType::F4
                && host_batch.hidden_bytes_per_row == batch.hidden_bytes_per_row
                && host_batch.routes == routes
        }));
        Ok(())
    }

    #[test]
    fn ds4_pro_tp4_transport_replicates_every_route_without_owner_lookup() -> Result<()> {
        let mut batch = reduced_mixed_scheduler_batch()?;
        batch.hidden_dim = DS4_PRO_HIDDEN_SIZE;
        batch.hidden_bytes_per_row = DS4_PRO_HIDDEN_SIZE * std::mem::size_of::<u16>();
        batch.routed_experts = DS4_PRO_ROUTED_EXPERTS;
        batch.quantization_recipe = DEEPSEEK_V4_EXL3_RECIPE.to_owned();
        let routes = core_routes_for_batch(&batch);

        let set = ds4_tp4_scheduler_host_batch_set(&batch, &routes)?;

        assert_eq!(set.num_hosts(), 4);
        assert_eq!(set.host_row_count(), batch.num_rows() * 4);
        assert_eq!(set.route_count(), routes.len() * 4);
        assert_eq!(
            set.touched_hosts().collect::<Vec<_>>(),
            EXPERT_HOSTS.to_vec()
        );
        assert!(set.batches.iter().all(|host_batch| {
            host_batch.num_rows() == batch.num_rows() && host_batch.routes == routes
        }));
        assert!(set
            .batches
            .iter()
            .flat_map(|host_batch| &host_batch.routes)
            .all(|route| route.expert_id < DS4_PRO_ROUTED_EXPERTS));

        batch.quantization_recipe = "deepseek_v4_native_fp4_fp8_mixed_v1".to_owned();
        let error = ds4_tp4_scheduler_host_batch_set(&batch, &routes).unwrap_err();
        assert!(error.to_string().contains("does not support"));
        Ok(())
    }

    #[test]
    fn historical_exl3_flash_transport_remains_strict_tp4() -> Result<()> {
        let mut batch = reduced_mixed_scheduler_batch()?;
        batch.hidden_dim = DS4_FLASH_HIDDEN_SIZE;
        batch.hidden_bytes_per_row = DS4_FLASH_HIDDEN_BF16_BYTES;
        batch.quantization_recipe = DEEPSEEK_V4_EXL3_RECIPE_V2.to_owned();
        let routes = core_routes_for_batch(&batch);

        let set = ds4_tp4_scheduler_host_batch_set(&batch, &routes)?;

        assert_eq!(set.num_hosts(), 4);
        assert_eq!(set.host_row_count(), batch.num_rows() * 4);
        assert_eq!(set.route_count(), routes.len() * 4);
        Ok(())
    }

    #[test]
    fn ds4_flash_tp4_verbs_chunks_keep_rank_identity_across_rail_order() -> Result<()> {
        let rows = 3_usize;
        let row_bytes = DS4_FLASH_HIDDEN_BF16_BYTES;
        let chunk = |rank: usize, global_rows: Vec<usize>, completed_rows: Vec<usize>, fill: u8| {
            VerbsHostProtocolV2HostBatchSetBf16PayloadChunk {
                host_index: rank,
                partial_output: VerbsHostProtocolV2ResponsePayload::from_owned(vec![
                    fill;
                    global_rows
                        .len()
                        * row_bytes
                ]),
                output_dtype: ExpertV2Dtype::Bf16,
                output_row_stride_bytes: row_bytes,
                global_row_indices: global_rows,
                completed_global_row_indices: completed_rows,
            }
        };
        let mut chunks = vec![
            chunk(2, vec![2, 0], vec![], 20),
            chunk(0, vec![1], vec![], 1),
            chunk(3, vec![1], vec![], 31),
            chunk(1, vec![2, 0], vec![], 10),
            chunk(0, vec![2, 0], vec![], 0),
            chunk(2, vec![1], vec![], 21),
            chunk(1, vec![1], vec![], 11),
            chunk(3, vec![2, 0], vec![2, 0, 1], 30),
        ];
        {
            let views = ds4_tp4_verbs_chunk_views(&chunks, rows, DS4_FLASH_HIDDEN_SIZE)?;
            assert_eq!(
                views.iter().map(|view| view.rank).collect::<Vec<_>>(),
                vec![2, 0, 3, 1, 0, 2, 1, 3]
            );
        }
        let segments = ds4_tp4_verbs_chunk_segments(&chunks, 1, 2, DS4_FLASH_HIDDEN_SIZE)?;
        let mut rows_by_rank = vec![Vec::new(); DS4_EXPERT_TP_WORLD_SIZE];
        let mut segment_payload_bytes = 0_usize;
        for segment in &segments {
            rows_by_rank[segment.rank].extend_from_slice(&segment.global_row_indices);
            segment_payload_bytes += segment.payload_bf16.len();
        }
        for rank_rows in &mut rows_by_rank {
            rank_rows.sort_unstable();
            assert_eq!(rank_rows, &[0, 1]);
        }
        assert_eq!(segment_payload_bytes, 4 * 2 * row_bytes);

        chunks[0].output_dtype = ExpertV2Dtype::Fp8E4m3RowScaled;
        let error = ds4_tp4_verbs_chunk_views(&chunks, rows, DS4_FLASH_HIDDEN_SIZE).unwrap_err();
        assert!(error.to_string().contains("must be hidden-width BF16"));
        chunks[0].output_dtype = ExpertV2Dtype::Bf16;

        chunks[6].completed_global_row_indices = vec![0];
        let error = ds4_tp4_verbs_chunk_views(&chunks, rows, DS4_FLASH_HIDDEN_SIZE).unwrap_err();
        assert!(error.to_string().contains("completed row 0 more than once"));
        chunks[6].completed_global_row_indices.clear();

        chunks[7].completed_global_row_indices.pop();
        let error = ds4_tp4_verbs_chunk_views(&chunks, rows, DS4_FLASH_HIDDEN_SIZE).unwrap_err();
        assert!(error.to_string().contains("did not complete row 1"));
        Ok(())
    }

    #[test]
    fn ds4_pro_tp4_verbs_chunks_accept_pro_hidden_width() -> Result<()> {
        let row_bytes = DS4_PRO_HIDDEN_SIZE * std::mem::size_of::<u16>();
        let chunks = (0..DS4_EXPERT_TP_WORLD_SIZE)
            .map(|rank| VerbsHostProtocolV2HostBatchSetBf16PayloadChunk {
                host_index: rank,
                partial_output: VerbsHostProtocolV2ResponsePayload::from_owned(vec![
                    rank as u8;
                    row_bytes
                ]),
                output_dtype: ExpertV2Dtype::Bf16,
                output_row_stride_bytes: row_bytes,
                global_row_indices: vec![0],
                completed_global_row_indices: if rank + 1 == DS4_EXPERT_TP_WORLD_SIZE {
                    vec![0]
                } else {
                    Vec::new()
                },
            })
            .collect::<Vec<_>>();

        let views = ds4_tp4_verbs_chunk_views(&chunks, 1, DS4_PRO_HIDDEN_SIZE)?;
        assert_eq!(views.len(), DS4_EXPERT_TP_WORLD_SIZE);
        assert!(views
            .iter()
            .enumerate()
            .all(|(rank, view)| view.rank == rank && view.payload_bf16.len() == row_bytes));

        let segments = ds4_tp4_verbs_chunk_segments(&chunks, 0, 1, DS4_PRO_HIDDEN_SIZE)?;
        assert_eq!(segments.len(), DS4_EXPERT_TP_WORLD_SIZE);
        assert!(segments
            .iter()
            .enumerate()
            .all(|(rank, segment)| segment.rank == rank
                && segment.payload_bf16.len() == row_bytes
                && segment.global_row_indices == [0]));
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn ds4_flash_tp4_tcp_barrier_returns_four_raw_hidden_width_partials() -> Result<()> {
        let mut batch = reduced_mixed_scheduler_batch()?;
        batch.hidden_dim = DS4_FLASH_HIDDEN_SIZE;
        batch.hidden_bytes_per_row = DS4_FLASH_HIDDEN_BF16_BYTES;
        let routes = core_routes_for_batch(&batch);
        let hidden = numeric_bf16_hidden_payload(batch.num_rows(), batch.hidden_dim);
        let mut targets = Vec::new();
        let mut servers = Vec::new();
        let rank_executor_names = [
            "ds4-flash-tp4-test-rank-0",
            "ds4-flash-tp4-test-rank-1",
            "ds4-flash-tp4-test-rank-2",
            "ds4-flash-tp4-test-rank-3",
        ];
        for (rank, host) in EXPERT_HOSTS.into_iter().enumerate() {
            let listener = TcpListener::bind("127.0.0.1:0").await?;
            let addr = listener.local_addr()?;
            let executor_name = rank_executor_names[rank];
            servers.push(tokio::spawn(async move {
                let _ = serve_protocol_v2_tcp_listener_with_executor(
                    listener,
                    Arc::new(RankedSyntheticExecutor {
                        name: executor_name,
                    }),
                )
                .await;
            }));
            targets.push(TcpProtocolV2HostBatchTarget {
                host: host.to_owned(),
                addr,
            });
        }
        // Target discovery order is not rank order; the host-batch envelope
        // remains the stable rank-0..3 source of returned partial ordering.
        targets.reverse();

        let dispatch = real_full_scheduler_ds4_flash_tp4_tcp_bf16_partials(
            &batch, &routes, &hidden, &targets, 72_000,
        )
        .await?;

        assert_eq!(dispatch.stats.hosts, 4);
        assert_eq!(dispatch.stats.global_rows, batch.num_rows());
        assert_eq!(dispatch.stats.host_rows, batch.num_rows() * 4);
        assert_eq!(dispatch.stats.routes, routes.len() * 4);
        assert_eq!(dispatch.stats.output_dim, DS4_FLASH_HIDDEN_SIZE);
        assert_eq!(
            dispatch.stats.contribution_counts,
            vec![4; batch.num_rows()]
        );
        assert_eq!(dispatch.partial_outputs_bf16_by_host.len(), 4);
        assert!(dispatch
            .partial_outputs_bf16_by_host
            .iter()
            .all(|payload| payload.len() == batch.num_rows() * DS4_FLASH_HIDDEN_BF16_BYTES));
        assert!(dispatch
            .global_row_indices_by_host
            .iter()
            .all(|rows| rows == &(0..batch.num_rows()).collect::<Vec<_>>()));
        assert_eq!(
            dispatch.stats.response_executor_ids,
            rank_executor_names.map(expert_protocol_v2_compact_id)
        );
        assert!(dispatch.stats.request_wire_bytes > hidden.len() * 4);
        assert!(dispatch.stats.response_wire_bytes > hidden.len() * 4);

        for server in servers {
            server.abort();
        }
        Ok(())
    }

    #[tokio::test]
    async fn scheduler_mixed_batch_dispatches_host_batch_set_over_protocol_v2_tcp() -> Result<()> {
        let batch = reduced_mixed_scheduler_batch()?;
        let routes = core_routes_for_batch(&batch);
        let set = scheduler_host_batch_set(&batch, &routes, None)?;
        let touched_hosts = set.touched_hosts().map(str::to_owned).collect::<Vec<_>>();
        let mut targets = Vec::new();
        let mut servers = Vec::new();

        for host in &touched_hosts {
            let listener = TcpListener::bind("127.0.0.1:0").await?;
            let addr = listener.local_addr()?;
            servers.push(tokio::spawn(async move {
                let _ = serve_protocol_v2_tcp_listener_with_executor(
                    listener,
                    Arc::new(SyntheticRouteExecutor),
                )
                .await;
            }));
            targets.push(TcpProtocolV2HostBatchTarget {
                host: host.clone(),
                addr,
            });
        }

        let dispatch = tcp_protocol_v2_host_batch_set_bf16_dispatch(
            &set,
            &numeric_bf16_hidden_payload(batch.num_rows(), batch.hidden_dim),
            &targets,
            487,
            TcpTransportConfig::default(),
        )
        .await?;

        assert_eq!(
            batch
                .rows
                .iter()
                .map(|row| row.source_kind)
                .collect::<Vec<_>>(),
            vec![
                RowSourceKind::PrefillChunk,
                RowSourceKind::MtpVerifyBlock,
                RowSourceKind::DecodeStep,
            ]
        );
        assert_eq!(set.num_hosts(), 4);
        assert_eq!(set.global_row_count, 3);
        assert_eq!(set.host_row_count(), 12);
        assert_eq!(
            set.route_count(),
            3 * DS4_FLASH_TOP_K * DS4_EXPERT_TP_WORLD_SIZE
        );
        assert_eq!(dispatch.stats.hosts, set.num_hosts());
        assert_eq!(dispatch.stats.global_rows, set.global_row_count);
        assert_eq!(dispatch.stats.host_rows, set.host_row_count());
        assert_eq!(dispatch.stats.routes, set.route_count());
        assert_eq!(dispatch.stats.output_dim, batch.hidden_dim);
        assert_eq!(
            dispatch.stats.output_values,
            batch.num_rows() * batch.hidden_dim
        );
        assert_eq!(dispatch.stats.contribution_counts, vec![4, 4, 4]);
        assert!(dispatch.stats.request_wire_bytes > 0);
        assert!(dispatch.stats.response_wire_bytes > 0);
        assert!(dispatch
            .accumulation
            .values
            .iter()
            .all(|value| value.is_finite()));
        assert!(dispatch
            .accumulation
            .values
            .iter()
            .any(|value| *value != 0.0));
        eprintln!(
            "scheduler_mixed_batch_host_batch_set_tcp_dispatch helper=tcp_protocol_v2_host_batch_set_bf16_dispatch executor=protocol-v2-synthetic-route-dependent-executor hosts={} global_rows={} host_rows={} routes={} hidden_dim={} output_values={} contribution_counts={:?} request_wire_bytes={} response_wire_bytes={} output_checksum={}",
            dispatch.stats.hosts,
            dispatch.stats.global_rows,
            dispatch.stats.host_rows,
            dispatch.stats.routes,
            dispatch.stats.output_dim,
            dispatch.stats.output_values,
            dispatch.stats.contribution_counts,
            dispatch.stats.request_wire_bytes,
            dispatch.stats.response_wire_bytes,
            dispatch.stats.output_checksum
        );

        for server in servers {
            server.abort();
        }
        Ok(())
    }

    #[tokio::test]
    async fn scheduler_tcp_dispatch_uses_router_routes_and_supplied_hidden_payload() -> Result<()> {
        let batch = reduced_mixed_scheduler_batch()?;
        let scored_routes = scored_routes_for_reduced_batch(&batch);
        let routes = scored_routes_for_scheduler_batch(&batch, &scored_routes)?;
        let set = scheduler_host_batch_set(&batch, &routes, None)?;
        let touched_hosts = set.touched_hosts().map(str::to_owned).collect::<Vec<_>>();
        let mut targets = Vec::new();
        let mut servers = Vec::new();

        for host in &touched_hosts {
            let listener = TcpListener::bind("127.0.0.1:0").await?;
            let addr = listener.local_addr()?;
            servers.push(tokio::spawn(async move {
                let _ = serve_protocol_v2_tcp_listener_with_executor(
                    listener,
                    Arc::new(SyntheticRouteExecutor),
                )
                .await;
            }));
            targets.push(TcpProtocolV2HostBatchTarget {
                host: host.clone(),
                addr,
            });
        }

        let hidden_a = numeric_bf16_hidden_payload(batch.num_rows(), batch.hidden_dim);
        let hidden_b = shifted_numeric_bf16_hidden_payload(batch.num_rows(), batch.hidden_dim);
        let dispatch_a = real_full_scheduler_host_batch_set_tcp_dispatch_with_payload(
            &batch, &routes, &hidden_a, &targets, None, 50_000,
        )
        .await?;
        let dispatch_b = real_full_scheduler_host_batch_set_tcp_dispatch_with_payload(
            &batch, &routes, &hidden_b, &targets, None, 50_100,
        )
        .await?;

        assert_eq!(routes.len(), batch.route_count());
        assert_eq!(routes[0].expert_id, scored_routes[0][0].expert_id);
        assert_eq!(routes[0].gate_weight, scored_routes[0][0].normalized_weight);
        assert_eq!(dispatch_a.stats.hosts, set.num_hosts());
        assert_eq!(dispatch_a.stats.global_rows, batch.num_rows());
        assert_eq!(dispatch_a.stats.host_rows, set.host_row_count());
        assert_eq!(
            dispatch_a.stats.routes,
            routes.len() * DS4_EXPERT_TP_WORLD_SIZE
        );
        assert_eq!(dispatch_a.stats.contribution_counts, vec![4, 4, 4]);
        assert_eq!(
            dispatch_b.stats.contribution_counts,
            dispatch_a.stats.contribution_counts
        );
        assert!(dispatch_a.stats.output_checksum.is_finite());
        assert!(dispatch_b.stats.output_checksum.is_finite());
        assert!(
            (dispatch_a.stats.output_checksum - dispatch_b.stats.output_checksum).abs() > 1.0e-6,
            "TCP dispatch output checksum must depend on supplied normalized hidden payload"
        );

        eprintln!(
            "scheduler_tcp_dispatch_real_payload helper=real_full_scheduler_host_batch_set_tcp_dispatch_with_payload routes={} hosts={} hidden_dim={} checksum_a={} checksum_b={}",
            routes.len(),
            dispatch_a.stats.hosts,
            dispatch_a.stats.output_dim,
            dispatch_a.stats.output_checksum,
            dispatch_b.stats.output_checksum
        );

        for server in servers {
            server.abort();
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn scheduler_full_first_sparse_batch_dispatches_host_batch_set_over_protocol_v2_tcp(
    ) -> Result<()> {
        let batch = first_sparse_scheduler_batch()?;
        let routes = core_routes_for_batch(&batch);
        let set = scheduler_host_batch_set(&batch, &routes, None)?;
        let touched_hosts = set.touched_hosts().map(str::to_owned).collect::<Vec<_>>();
        let mut targets = Vec::new();
        let mut servers = Vec::new();

        for host in &touched_hosts {
            let listener = TcpListener::bind("127.0.0.1:0").await?;
            let addr = listener.local_addr()?;
            servers.push(tokio::spawn(async move {
                let _ = serve_protocol_v2_tcp_listener_with_executor(
                    listener,
                    Arc::new(SyntheticRouteExecutor),
                )
                .await;
            }));
            targets.push(TcpProtocolV2HostBatchTarget {
                host: host.clone(),
                addr,
            });
        }

        let dispatch =
            real_full_scheduler_host_batch_set_tcp_dispatch(&batch, &targets, None, 489).await?;

        assert_eq!(
            batch.layer_id.0 as usize,
            ModelFacts::default().first_k_dense_replace
        );
        assert_eq!(batch.num_rows(), 521);
        assert_eq!(batch.hidden_dim, DS4_FLASH_HIDDEN_SIZE);
        assert_eq!(set.num_hosts(), 4);
        assert_eq!(set.host_row_count(), 2_084);
        assert_eq!(
            set.route_count(),
            batch.num_rows() * DS4_FLASH_TOP_K * DS4_EXPERT_TP_WORLD_SIZE
        );
        assert_eq!(dispatch.stats.hosts, set.num_hosts());
        assert_eq!(dispatch.stats.global_rows, set.global_row_count);
        assert_eq!(dispatch.stats.host_rows, set.host_row_count());
        assert_eq!(dispatch.stats.routes, set.route_count());
        assert_eq!(dispatch.stats.output_dim, batch.hidden_dim);
        assert_eq!(
            dispatch.stats.output_values,
            batch.num_rows() * batch.hidden_dim
        );
        assert_eq!(
            dispatch.stats.contribution_counts,
            vec![4; batch.num_rows()]
        );
        assert_eq!(dispatch.stats.response_executor_ids.len(), set.num_hosts());
        assert!(
            dispatch.stats.request_wire_bytes > set.host_row_count() * DS4_FLASH_HIDDEN_BF16_BYTES
        );
        assert!(
            dispatch.stats.response_wire_bytes > set.host_row_count() * DS4_FLASH_HIDDEN_BF16_BYTES
        );
        assert!(dispatch
            .accumulation
            .values
            .iter()
            .all(|value| value.is_finite()));
        assert!(dispatch
            .accumulation
            .values
            .iter()
            .any(|value| *value != 0.0));
        eprintln!(
            "scheduler_full_first_sparse_batch_host_batch_set_tcp_dispatch helper=real_full_scheduler_host_batch_set_tcp_dispatch executor=protocol-v2-synthetic-route-dependent-executor hosts={} global_rows={} host_rows={} routes={} hidden_dim={} output_values={} contribution_counts_len={} request_wire_bytes={} response_wire_bytes={} output_checksum={}",
            dispatch.stats.hosts,
            dispatch.stats.global_rows,
            dispatch.stats.host_rows,
            dispatch.stats.routes,
            dispatch.stats.output_dim,
            dispatch.stats.output_values,
            dispatch.stats.contribution_counts.len(),
            dispatch.stats.request_wire_bytes,
            dispatch.stats.response_wire_bytes,
            dispatch.stats.output_checksum
        );

        for server in servers {
            server.abort();
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn scheduler_request_shape_dispatches_all_sparse_batches_over_protocol_v2_tcp(
    ) -> Result<()> {
        let hosts = EXPERT_HOSTS
            .iter()
            .map(|host| (*host).to_owned())
            .collect::<Vec<_>>();
        let mut targets = Vec::new();
        let mut servers = Vec::new();

        for host in &hosts {
            let listener = TcpListener::bind("127.0.0.1:0").await?;
            let addr = listener.local_addr()?;
            servers.push(tokio::spawn(async move {
                let _ = serve_protocol_v2_tcp_listener_with_executor(
                    listener,
                    Arc::new(SyntheticRouteExecutor),
                )
                .await;
            }));
            targets.push(TcpProtocolV2HostBatchTarget {
                host: host.clone(),
                addr,
            });
        }

        let shape = RealFullSchedulerExecutionShape {
            request_id: "scheduler-request-shape-sparse-tcp".to_owned(),
            sequence_id: "scheduler-request-shape-sparse-tcp-sequence".to_owned(),
            placement_version: "scheduler-request-shape-sparse-tcp-placement".to_owned(),
            prefix_tokens: 0,
            prefill_tokens: 4,
            prefill_chunk_tokens: 2,
            decode_rows: 1,
            mtp_rows: 2,
            mtp_accepted_rows: 1,
            prefill_token_ids: None,
            decode_token_ids: None,
            lm_head_sampling: RealFullLmHeadSamplingOptions::diagnostic(),
        };
        let probe =
            real_full_scheduler_sparse_tcp_dispatch_for_shape(&shape, &targets, None, 50_000)
                .await?;

        let sparse_layers = DS4_FLASH_NUM_HIDDEN_LAYERS;
        let expected_sparse_batches = sparse_layers * 2;
        let expected_global_rows = sparse_layers * 7;
        assert!(probe.passed);
        assert_eq!(probe.status, "request-shaped-sparse-tcp-dispatch-passed");
        assert!(probe.scope.contains("ProtocolV2 TCP host-batch-set fanout"));
        assert_eq!(probe.sparse_layers, sparse_layers);
        assert_eq!(probe.scheduler_iterations_per_sparse_layer, 2);
        assert_eq!(probe.sparse_batches, expected_sparse_batches);
        assert_eq!(
            probe.host_batches,
            expected_sparse_batches * EXPERT_HOSTS.len()
        );
        assert_eq!(probe.global_rows, expected_global_rows);
        assert_eq!(probe.host_rows, expected_global_rows * EXPERT_HOSTS.len());
        assert_eq!(
            probe.routes,
            expected_global_rows * DS4_FLASH_TOP_K * DS4_EXPERT_TP_WORLD_SIZE
        );
        assert_eq!(
            probe.output_values,
            expected_global_rows * DS4_FLASH_HIDDEN_SIZE
        );
        assert_eq!(probe.output_finite_values, probe.output_values);
        assert!(probe.output_nonzero_values > 0);
        assert!(probe.output_checksum.is_finite());
        assert!(probe.request_wire_bytes > probe.host_rows * DS4_FLASH_HIDDEN_BF16_BYTES);
        assert!(probe.response_wire_bytes > probe.host_rows * DS4_FLASH_HIDDEN_BF16_BYTES);
        assert_eq!(
            probe.expected_real_executor_id,
            expert_protocol_v2_compact_id(REAL_NVFP4_PROTOCOL_V2_EXECUTOR)
        );
        assert_eq!(probe.response_executor_ids_observed, probe.host_batches);
        assert_eq!(probe.real_executor_responses, 0);
        assert_eq!(probe.non_real_executor_responses, probe.host_batches);
        assert!(!probe.all_responses_real_nvfp4);
        assert_ne!(
            probe.expected_real_executor_id,
            expert_protocol_v2_compact_id(PROTOCOL_V2_SYNTHETIC_ROUTE_EXECUTOR)
        );
        eprintln!(
            "scheduler_request_shape_all_sparse_tcp_dispatch status={} sparse_layers={} sparse_batches={} global_rows={} host_rows={} routes={} request_wire_bytes={} response_wire_bytes={} output_values={} output_nonzero_values={} output_checksum={} response_executor_ids_observed={} real_executor_responses={} all_responses_real_nvfp4={}",
            probe.status,
            probe.sparse_layers,
            probe.sparse_batches,
            probe.global_rows,
            probe.host_rows,
            probe.routes,
            probe.request_wire_bytes,
            probe.response_wire_bytes,
            probe.output_values,
            probe.output_nonzero_values,
            probe.output_checksum,
            probe.response_executor_ids_observed,
            probe.real_executor_responses,
            probe.all_responses_real_nvfp4
        );

        for server in servers {
            server.abort();
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn scheduler_request_shape_sparse_dispatch_ignores_ep_owner_lookup() -> Result<()> {
        let owner_host = "spark-0";
        let owner_lookup = all_sparse_experts_owned_by(owner_host);
        let mut targets = Vec::new();
        let mut servers = Vec::new();
        for host in EXPERT_HOSTS {
            let listener = TcpListener::bind("127.0.0.1:0").await?;
            let addr = listener.local_addr()?;
            servers.push(tokio::spawn(async move {
                let _ = serve_protocol_v2_tcp_listener_with_executor(
                    listener,
                    Arc::new(SyntheticRouteExecutor),
                )
                .await;
            }));
            targets.push(TcpProtocolV2HostBatchTarget {
                host: host.to_owned(),
                addr,
            });
        }
        let shape = RealFullSchedulerExecutionShape {
            request_id: "scheduler-owner-lookup-sparse-tcp".to_owned(),
            sequence_id: "scheduler-owner-lookup-sparse-tcp-sequence".to_owned(),
            placement_version: "scheduler-owner-lookup-sparse-tcp-placement".to_owned(),
            prefix_tokens: 0,
            prefill_tokens: 4,
            prefill_chunk_tokens: 2,
            decode_rows: 1,
            mtp_rows: 2,
            mtp_accepted_rows: 1,
            prefill_token_ids: None,
            decode_token_ids: None,
            lm_head_sampling: RealFullLmHeadSamplingOptions::diagnostic(),
        };
        let probe = real_full_scheduler_sparse_tcp_dispatch_for_shape(
            &shape,
            &targets,
            Some(&owner_lookup),
            60_000,
        )
        .await?;

        let sparse_layers = DS4_FLASH_NUM_HIDDEN_LAYERS;
        let expected_sparse_batches = sparse_layers * 2;
        let expected_global_rows = sparse_layers * 7;
        assert!(probe.passed);
        assert_eq!(probe.sparse_batches, expected_sparse_batches);
        assert_eq!(
            probe.host_batches,
            expected_sparse_batches * DS4_EXPERT_TP_WORLD_SIZE
        );
        assert_eq!(probe.global_rows, expected_global_rows);
        assert_eq!(
            probe.host_rows,
            expected_global_rows * DS4_EXPERT_TP_WORLD_SIZE
        );
        assert_eq!(
            probe.routes,
            expected_global_rows * DS4_FLASH_TOP_K * DS4_EXPERT_TP_WORLD_SIZE
        );
        assert_eq!(probe.response_executor_ids_observed, probe.host_batches);
        assert_eq!(probe.real_executor_responses, 0);
        assert_eq!(probe.non_real_executor_responses, probe.host_batches);
        assert!(!probe.all_responses_real_nvfp4);

        for server in servers {
            server.abort();
        }
        Ok(())
    }

    #[tokio::test]
    async fn scheduler_mixed_batch_dispatches_real_nvfp4_host_batch_set_when_available(
    ) -> Result<()> {
        let Some(catalog) = load_real_catalog_or_skip() else {
            return Ok(());
        };
        let Some(owner_lookup) = load_full_owner_lookup_or_skip() else {
            return Ok(());
        };
        let batch = reduced_mixed_scheduler_batch()?;
        let routes = core_routes_for_batch(&batch);
        let hosts = EXPERT_HOSTS
            .iter()
            .map(|host| (*host).to_owned())
            .collect::<Vec<_>>();
        let set = ExpertHostBatchSet::from_expert_batch_with_owner_lookup(
            &batch,
            &routes,
            &hosts,
            &owner_lookup,
        )?;
        let touched_hosts = set.touched_hosts().map(str::to_owned).collect::<Vec<_>>();
        let mut targets = Vec::new();
        let mut servers = Vec::new();

        for host in &touched_hosts {
            let listener = TcpListener::bind("127.0.0.1:0").await?;
            let addr = listener.local_addr()?;
            let executor =
                RealNvfp4ProtocolV2Executor::new(catalog.clone(), Some(3), Some(host.clone()))
                    .with_owner_lookup(owner_lookup.clone());
            servers.push(tokio::spawn(async move {
                let _ = serve_protocol_v2_tcp_listener_with_executor(listener, Arc::new(executor))
                    .await;
            }));
            targets.push(TcpProtocolV2HostBatchTarget {
                host: host.clone(),
                addr,
            });
        }

        let dispatch = tcp_protocol_v2_host_batch_set_bf16_dispatch(
            &set,
            &numeric_bf16_hidden_payload(batch.num_rows(), batch.hidden_dim),
            &targets,
            488,
            TcpTransportConfig::default(),
        )
        .await?;

        assert_eq!(touched_hosts, hosts);
        assert_eq!(dispatch.stats.hosts, 4);
        assert_eq!(dispatch.stats.global_rows, 3);
        assert_eq!(dispatch.stats.host_rows, 12);
        assert_eq!(dispatch.stats.routes, 3 * DS4_FLASH_TOP_K);
        assert_eq!(dispatch.stats.output_dim, batch.hidden_dim);
        assert_eq!(dispatch.stats.output_values, 48);
        assert_eq!(dispatch.stats.contribution_counts, vec![4, 4, 4]);
        assert!(dispatch.stats.request_wire_bytes > 0);
        assert!(dispatch.stats.response_wire_bytes > 0);
        assert_eq!(
            dispatch.stats.response_executor_ids,
            vec![
                expert_protocol_v2_compact_id(REAL_NVFP4_PROTOCOL_V2_EXECUTOR);
                dispatch.stats.hosts
            ]
        );
        assert!(dispatch
            .accumulation
            .values
            .iter()
            .all(|value| value.is_finite()));
        assert!(dispatch
            .accumulation
            .values
            .iter()
            .any(|value| *value != 0.0));
        eprintln!(
            "scheduler_mixed_batch_real_nvfp4_host_batch_set_tcp_dispatch executor=protocol-v2-real-nvfp4-checkpoint-executor helper=tcp_protocol_v2_host_batch_set_bf16_dispatch hosts={} host_names={:?} global_rows={} host_rows={} routes={} hidden_dim={} output_values={} contribution_counts={:?} request_wire_bytes={} response_wire_bytes={} output_checksum={}",
            dispatch.stats.hosts,
            touched_hosts,
            dispatch.stats.global_rows,
            dispatch.stats.host_rows,
            dispatch.stats.routes,
            dispatch.stats.output_dim,
            dispatch.stats.output_values,
            dispatch.stats.contribution_counts,
            dispatch.stats.request_wire_bytes,
            dispatch.stats.response_wire_bytes,
            dispatch.stats.output_checksum
        );

        for server in servers {
            server.abort();
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    #[ignore = "starts four real-expertd checkpoint daemons after startup resident preload; run explicitly for live scheduler ProtocolV2 coverage"]
    async fn scheduler_mixed_batch_dispatches_real_nvfp4_through_expertd_entrypoints_when_available(
    ) -> Result<()> {
        let Some(loadplan_path) = load_full_loadplan_path_or_skip() else {
            return Ok(());
        };
        let Some(owner_lookup) = load_full_owner_lookup_or_skip() else {
            return Ok(());
        };
        let catalog_path = real_catalog_path();
        let batch = reduced_mixed_scheduler_batch()?;
        let routes = core_routes_for_batch(&batch);
        let hosts = EXPERT_HOSTS
            .iter()
            .map(|host| (*host).to_owned())
            .collect::<Vec<_>>();
        let set = ExpertHostBatchSet::from_expert_batch_with_owner_lookup(
            &batch,
            &routes,
            &hosts,
            &owner_lookup,
        )?;
        let touched_hosts = set.touched_hosts().map(str::to_owned).collect::<Vec<_>>();
        assert_eq!(touched_hosts, hosts);
        let mut targets = Vec::new();
        let mut servers = Vec::new();

        for host in &touched_hosts {
            let addr = unused_loopback_addr();
            let args = ExpertDaemonArgs {
                synthetic_weights: false,
                preflight_only: false,
                transport: "tcp".to_owned(),
                listen: addr.to_string(),
                loadplan: Some(loadplan_path.clone()),
                catalog: Some(catalog_path.clone()),
                model_id: ds4rt_core::DEFAULT_MODEL_ID.to_owned(),
                real_layer: Some(3),
                role_hostname: Some(host.clone()),
            };
            servers.push(tokio::spawn(async move { run_expertd(args).await }));
            targets.push(TcpProtocolV2HostBatchTarget {
                host: host.clone(),
                addr,
            });
        }
        for target in &targets {
            wait_for_expertd_tcp_listener(target.addr).await;
        }

        let dispatch = tcp_protocol_v2_host_batch_set_bf16_dispatch(
            &set,
            &numeric_bf16_hidden_payload(batch.num_rows(), batch.hidden_dim),
            &targets,
            506,
            TcpTransportConfig::default(),
        )
        .await?;

        assert_eq!(
            batch
                .rows
                .iter()
                .map(|row| row.source_kind)
                .collect::<Vec<_>>(),
            vec![
                RowSourceKind::PrefillChunk,
                RowSourceKind::MtpVerifyBlock,
                RowSourceKind::DecodeStep,
            ]
        );
        assert_eq!(dispatch.stats.hosts, 4);
        assert_eq!(dispatch.stats.global_rows, 3);
        assert_eq!(dispatch.stats.host_rows, 12);
        assert_eq!(dispatch.stats.routes, 3 * DS4_FLASH_TOP_K);
        assert_eq!(dispatch.stats.output_dim, batch.hidden_dim);
        assert_eq!(dispatch.stats.output_values, 48);
        assert_eq!(dispatch.stats.contribution_counts, vec![4, 4, 4]);
        assert!(dispatch.stats.request_wire_bytes > 0);
        assert!(dispatch.stats.response_wire_bytes > 0);
        assert!(dispatch
            .accumulation
            .values
            .iter()
            .all(|value| value.is_finite()));
        assert!(dispatch
            .accumulation
            .values
            .iter()
            .any(|value| *value != 0.0));
        eprintln!(
            "scheduler_mixed_batch_real_nvfp4_expertd_entrypoint_dispatch daemon=run_expertd executor=protocol-v2-real-nvfp4-checkpoint-executor hosts={} host_names={:?} global_rows={} host_rows={} routes={} hidden_dim={} output_values={} contribution_counts={:?} request_wire_bytes={} response_wire_bytes={} output_checksum={}",
            dispatch.stats.hosts,
            touched_hosts,
            dispatch.stats.global_rows,
            dispatch.stats.host_rows,
            dispatch.stats.routes,
            dispatch.stats.output_dim,
            dispatch.stats.output_values,
            dispatch.stats.contribution_counts,
            dispatch.stats.request_wire_bytes,
            dispatch.stats.response_wire_bytes,
            dispatch.stats.output_checksum
        );

        for server in servers {
            server.abort();
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    #[ignore = "starts real-expertd checkpoint daemons for multiple layers; run explicitly for live scheduler ProtocolV2 coverage"]
    async fn scheduler_mixed_batch_dispatches_multiple_real_layers_through_unpinned_expertd_entrypoints_when_available(
    ) -> Result<()> {
        let Some(loadplan_path) = load_full_loadplan_path_or_skip() else {
            return Ok(());
        };
        let Some(owner_lookup) = load_full_owner_lookup_or_skip() else {
            return Ok(());
        };
        let catalog_path = real_catalog_path();
        let hosts = EXPERT_HOSTS
            .iter()
            .map(|host| (*host).to_owned())
            .collect::<Vec<_>>();
        let layer_ids = [GLM52_FIRST_K_DENSE_REPLACE, GLM52_NUM_HIDDEN_LAYERS - 1];
        let mut total_global_rows = 0;
        let mut total_host_rows = 0;
        let mut total_routes = 0;
        let mut total_output_values = 0;
        let mut total_request_wire_bytes = 0;
        let mut total_response_wire_bytes = 0;
        let mut output_checksums = Vec::new();

        for layer_id in layer_ids {
            let mut targets = Vec::new();
            let mut servers = Vec::new();

            for host in &hosts {
                let addr = unused_loopback_addr();
                let args = ExpertDaemonArgs {
                    synthetic_weights: false,
                    preflight_only: false,
                    transport: "tcp".to_owned(),
                    listen: addr.to_string(),
                    loadplan: Some(loadplan_path.clone()),
                    catalog: Some(catalog_path.clone()),
                    model_id: ds4rt_core::DEFAULT_MODEL_ID.to_owned(),
                    real_layer: Some(layer_id as u32),
                    role_hostname: Some(host.clone()),
                };
                servers.push(tokio::spawn(async move { run_expertd(args).await }));
                targets.push(TcpProtocolV2HostBatchTarget {
                    host: host.clone(),
                    addr,
                });
            }
            for target in &targets {
                wait_for_expertd_tcp_listener(target.addr).await;
            }

            let batch = reduced_mixed_scheduler_batch_for_layer(layer_id)?;
            let routes = core_routes_for_batch(&batch);
            let set = ExpertHostBatchSet::from_expert_batch_with_owner_lookup(
                &batch,
                &routes,
                &hosts,
                &owner_lookup,
            )?;
            assert_eq!(
                set.touched_hosts().map(str::to_owned).collect::<Vec<_>>(),
                hosts
            );

            let dispatch = tcp_protocol_v2_host_batch_set_bf16_dispatch(
                &set,
                &numeric_bf16_hidden_payload(batch.num_rows(), batch.hidden_dim),
                &targets,
                507 + layer_id as u64,
                TcpTransportConfig::default(),
            )
            .await?;

            assert_eq!(batch.layer_id.0 as usize, layer_id);
            assert_eq!(
                batch
                    .rows
                    .iter()
                    .map(|row| row.source_kind)
                    .collect::<Vec<_>>(),
                vec![
                    RowSourceKind::PrefillChunk,
                    RowSourceKind::MtpVerifyBlock,
                    RowSourceKind::DecodeStep,
                ]
            );
            assert_eq!(dispatch.stats.hosts, 4);
            assert_eq!(dispatch.stats.global_rows, 3);
            assert_eq!(dispatch.stats.host_rows, 12);
            assert_eq!(dispatch.stats.routes, 3 * DS4_FLASH_TOP_K);
            assert_eq!(dispatch.stats.output_dim, batch.hidden_dim);
            assert_eq!(dispatch.stats.output_values, 48);
            assert_eq!(dispatch.stats.contribution_counts, vec![4, 4, 4]);
            assert!(dispatch.stats.request_wire_bytes > 0);
            assert!(dispatch.stats.response_wire_bytes > 0);
            assert!(dispatch
                .accumulation
                .values
                .iter()
                .all(|value| value.is_finite()));
            assert!(dispatch
                .accumulation
                .values
                .iter()
                .any(|value| *value != 0.0));

            total_global_rows += dispatch.stats.global_rows;
            total_host_rows += dispatch.stats.host_rows;
            total_routes += dispatch.stats.routes;
            total_output_values += dispatch.stats.output_values;
            total_request_wire_bytes += dispatch.stats.request_wire_bytes;
            total_response_wire_bytes += dispatch.stats.response_wire_bytes;
            output_checksums.push(dispatch.stats.output_checksum);

            for server in servers {
                server.abort();
            }
        }

        eprintln!(
            "scheduler_mixed_batch_multi_layer_real_nvfp4_expertd_entrypoint_dispatch daemon=run_expertd executor=protocol-v2-real-nvfp4-checkpoint-executor serving_layer_filter=per-layer hosts={} host_names={:?} layers={:?} total_global_rows={} total_host_rows={} total_routes={} hidden_dim=16 total_output_values={} contribution_counts_per_layer=[4,4,4] total_request_wire_bytes={} total_response_wire_bytes={} output_checksums={:?}",
            hosts.len(),
            hosts,
            layer_ids,
            total_global_rows,
            total_host_rows,
            total_routes,
            total_output_values,
            total_request_wire_bytes,
            total_response_wire_bytes,
            output_checksums
        );
        Ok(())
    }

    fn first_sparse_scheduler_batch() -> Result<ExpertBatch> {
        sparse_scheduler_batch_for_layer(ModelFacts::default().first_k_dense_replace)
    }

    fn sparse_scheduler_batch_for_layer(layer_id: usize) -> Result<ExpertBatch> {
        let placement_version = "scheduler-protocol-v2-test";
        let facts = ModelFacts::default();
        let graph_bucket = GraphBucket::new(
            REAL_FULL_PREFLIGHT_PREFILL_ROWS
                + REAL_FULL_PREFLIGHT_MTP_ROWS
                + REAL_FULL_PREFLIGHT_DECODE_ROWS,
        );
        let quantization_recipe = facts.quantization_recipe.clone();
        let prefill = super::super::real_full_prefill_wave(layer_id, placement_version, &facts);
        let mtp_verify = super::super::real_full_mtp_wave(layer_id, placement_version, &facts);
        let decode = super::super::real_full_decode_wave(layer_id, placement_version, &facts);
        let mut batch = ExpertBatch::bf16_from_wave_with_envelope(
            &prefill,
            quantization_recipe.clone(),
            graph_bucket,
        )?;
        batch.try_append_wave(&mtp_verify, DType::Bf16, quantization_recipe.clone())?;
        batch.try_append_wave(&decode, DType::Bf16, quantization_recipe)?;
        Ok(batch)
    }

    fn reduced_mixed_scheduler_batch() -> Result<ExpertBatch> {
        reduced_mixed_scheduler_batch_for_layer(ModelFacts::default().first_k_dense_replace)
    }

    fn reduced_mixed_scheduler_batch_for_layer(layer_id: usize) -> Result<ExpertBatch> {
        let full = sparse_scheduler_batch_for_layer(layer_id)?;
        let selected = [
            0,
            REAL_FULL_PREFLIGHT_PREFILL_ROWS,
            REAL_FULL_PREFLIGHT_PREFILL_ROWS + REAL_FULL_PREFLIGHT_MTP_ROWS,
        ];
        let mut rows = Vec::with_capacity(selected.len());
        for (row_index, full_index) in selected.into_iter().enumerate() {
            let mut row = full.rows[full_index].clone();
            row.row_id = row_index as u64;
            row.route_offset = row_index * row.route_count;
            rows.push(row);
        }

        Ok(ExpertBatch {
            layer_id: full.layer_id,
            placement_version: full.placement_version,
            hidden_dim: 16,
            hidden_bytes_per_row: 32,
            hidden_dtype: full.hidden_dtype,
            graph_bucket: GraphBucket::new(rows.len()),
            routed_experts: full.routed_experts,
            quantization_recipe: full.quantization_recipe,
            rows,
        })
    }

    fn numeric_bf16_hidden_payload(rows: usize, hidden_dim: usize) -> Vec<u8> {
        let mut payload = Vec::with_capacity(rows * hidden_dim * 2);
        for row in 0..rows {
            for col in 0..hidden_dim {
                let value = (((row * 17 + col) % 31) as f32 - 15.0) / 32.0;
                payload.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
            }
        }
        payload
    }

    fn shifted_numeric_bf16_hidden_payload(rows: usize, hidden_dim: usize) -> Vec<u8> {
        let mut payload = Vec::with_capacity(rows * hidden_dim * 2);
        for row in 0..rows {
            for col in 0..hidden_dim {
                let value = (((row * 29 + col * 3 + 7) % 43) as f32 - 21.0) / 24.0;
                payload.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
            }
        }
        payload
    }

    fn scored_routes_for_reduced_batch(batch: &ExpertBatch) -> Vec<Vec<ScoredRoute>> {
        batch
            .rows
            .iter()
            .enumerate()
            .map(|(row_index, row)| {
                (0..row.route_count)
                    .map(|route_index| {
                        let expert_id =
                            (row_index * row.route_count + route_index) % batch.routed_experts;
                        let normalized_weight = (route_index + 1) as f32 / row.route_count as f32;
                        ScoredRoute {
                            expert_id,
                            score: normalized_weight,
                            corrected_score: normalized_weight,
                            normalized_weight,
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn load_real_catalog_or_skip() -> Option<TensorCatalog> {
        if !crate::commands::real_full::tests::fixture::real_checkpoint_tests_enabled() {
            crate::commands::real_full::tests::fixture::real_checkpoint_tests_skip_message(
                "scheduler ProtocolV2 real catalog",
            );
            return None;
        }
        let catalog_path = real_catalog_path();
        let Ok(file) = File::open(&catalog_path) else {
            eprintln!("skipped: missing {}", catalog_path.display());
            return None;
        };
        let catalog: TensorCatalog =
            serde_json::from_reader(file).expect("parsing real GLM catalog fixture");
        if !Path::new(&catalog.snapshot_path).exists() {
            eprintln!("skipped: missing snapshot {}", catalog.snapshot_path);
            return None;
        }
        Some(catalog)
    }

    fn all_sparse_experts_owned_by(host: &str) -> ExpertOwnerLookup {
        let mut pairs = Vec::new();
        let facts = ModelFacts::default();
        for layer_id in facts.first_k_dense_replace..facts.num_hidden_layers {
            for expert_id in 0..facts.routed_experts {
                pairs.push(((layer_id, expert_id), host.to_owned()));
            }
        }
        ExpertOwnerLookup::from_pairs(pairs)
    }

    fn real_catalog_path() -> PathBuf {
        repo_root().join(".ds4rt-cache/model-artifacts/diagnostic/model_catalog.json")
    }

    fn load_full_owner_lookup_or_skip() -> Option<ExpertOwnerLookup> {
        let loadplan_path = load_full_loadplan_path_or_skip()?;
        Some(read_expert_owner_lookup(&loadplan_path).expect("parsing full GLM loadplan fixture"))
    }

    fn load_full_loadplan_path_or_skip() -> Option<PathBuf> {
        if !crate::commands::real_full::tests::fixture::real_checkpoint_tests_enabled() {
            crate::commands::real_full::tests::fixture::real_checkpoint_tests_skip_message(
                "scheduler ProtocolV2 full loadplan",
            );
            return None;
        }
        let loadplan_path =
            repo_root().join(".ds4rt-cache/model-artifacts/diagnostic/loadplan.json");
        if !loadplan_path.exists() {
            eprintln!("skipped: missing {}", loadplan_path.display());
            return None;
        }
        Some(loadplan_path)
    }

    fn unused_loopback_addr() -> SocketAddr {
        let listener = StdTcpListener::bind("127.0.0.1:0").unwrap();
        listener.local_addr().unwrap()
    }

    async fn wait_for_expertd_tcp_listener(addr: SocketAddr) {
        for _ in 0..24_000 {
            if TcpStream::connect(addr).await.is_ok() {
                return;
            }
            sleep(Duration::from_millis(10)).await;
        }
        panic!("timed out waiting for expertd TCP listener at {addr}");
    }

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
    }
}
