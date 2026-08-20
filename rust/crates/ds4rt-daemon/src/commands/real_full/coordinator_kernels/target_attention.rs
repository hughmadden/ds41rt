use super::*;
use crate::commands::real_full::constants::{
    REAL_FULL_BOUNDED_PREFILL_MAX_ACTIVE_CHUNKS, REAL_FULL_LAYER_MAJOR_PREFILL_MAX_ROWS,
    REAL_FULL_TARGET_AUXILIARY_MAX_ROWS,
};
use crate::python_graph_capture::{
    launch_python_graph_capture, query_python_usize_during_startup, PythonDeviceBufferArg,
    PythonGraphCaptureLaunch, PythonKernelArg, PythonUsizeQuery,
};
use ds4rt_core::{
    DType, DeepseekV4AttentionPlan, DeepseekV4KvCacheFormat, DeepseekV4PhysicalKvPlan, ModelFacts,
    ModelVariant, TensorCatalog, DS4_INDEX_HEADS, DS4_INDEX_HEAD_DIM,
};
use ds4rt_loader::read_tensor_bytes_into;

const TARGET_ATTENTION_CAPTURE_MODULE: &str = "deepseek_v4_attention_layer_capture";
const TARGET_ATTENTION_ARENA_NBYTES_FUNCTION: &str = "deepseek_v4_attention_layer_arena_nbytes";
const TARGET_C4_SELECTOR_SCRATCH_NBYTES_FUNCTION: &str = "deepseek_v4_c4_selector_scratch_nbytes";
const TARGET_SLIDING_PREPARE_FUNCTION: &str = "prepare_deepseek_v4_sliding_attention_layer";
const TARGET_SLIDING_CAPTURE_FUNCTION: &str = "capture_deepseek_v4_sliding_attention_layer";
const TARGET_C4_PREFILL_PREPARE_FUNCTION: &str = "prepare_deepseek_v4_c4_prefill_attention_layer";
const TARGET_C4_PREFILL_CAPTURE_FUNCTION: &str = "capture_deepseek_v4_c4_prefill_attention_layer";
const TARGET_C4_CONTINUATION_PREPARE_FUNCTION: &str =
    "prepare_deepseek_v4_c4_continuation_attention_layer";
const TARGET_C4_CONTINUATION_CAPTURE_FUNCTION: &str =
    "capture_deepseek_v4_c4_continuation_attention_layer";
const TARGET_C128_PREFILL_PREPARE_FUNCTION: &str =
    "prepare_deepseek_v4_c128_prefill_attention_layer";
const TARGET_C128_PREFILL_CAPTURE_FUNCTION: &str =
    "capture_deepseek_v4_c128_prefill_attention_layer";
const TARGET_C128_CONTINUATION_PREPARE_FUNCTION: &str =
    "prepare_deepseek_v4_c128_continuation_attention_layer";
const TARGET_C128_CONTINUATION_CAPTURE_FUNCTION: &str =
    "capture_deepseek_v4_c128_continuation_attention_layer";
const TARGET_ATTENTION_TIMING_ENV: &str = "DS4RT_REAL_FULL_ATTENTION_CUDA_TIMING";
const TARGET_MHC_CAPTURE_MODULE: &str = "deepseek_v4_mhc_capture";
const TARGET_MHC_SCRATCH_NBYTES_FUNCTION: &str = "deepseek_v4_mhc_scratch_nbytes";
const TARGET_MHC_ENTRY_PREPARE_FUNCTION: &str = "prepare_deepseek_v4_mhc_entry";
const TARGET_MHC_ENTRY_CAPTURE_FUNCTION: &str = "capture_deepseek_v4_mhc_entry";
const TARGET_MHC_POST_PRE_PREPARE_FUNCTION: &str = "prepare_deepseek_v4_mhc_post_pre";
const TARGET_MHC_POST_PRE_CAPTURE_FUNCTION: &str = "capture_deepseek_v4_mhc_post_pre";
const TARGET_MHC_TERMINAL_PREPARE_FUNCTION: &str = "prepare_deepseek_v4_mhc_terminal";
const TARGET_MHC_TERMINAL_CAPTURE_FUNCTION: &str = "capture_deepseek_v4_mhc_terminal";
const TARGET_CAPTURE_ALIGNMENT: usize = 1_024;
const TARGET_HC_MULT: usize = 4;
const TARGET_HC_MIXES: usize = 24;
const TARGET_SWA_WIDTH: usize = 128;
const TARGET_ROPE_WIDTH: usize = 64;
// SparkInfer's SM120 C128 prefill kernel consumes selection metadata in one
// 64-entry tile even when the physical context contains fewer completed
// 128-token blocks. Keep this synchronized with
// DS4_SM120_PREFILL_SELECTION_TILE in deepseek_v4_attention_capture.py.
const TARGET_C128_PREFILL_SELECTION_TILE: usize = 64;
const TARGET_COMPRESSOR_CARRY_ROPE_SLOTS: usize = 1;
const TARGET_C128_BLOCKS_PER_SOURCE_PAGE: usize = 2;
const TARGET_EXPERT_TP: usize = 4;
const TARGET_LAYER0_HC_ATTN_FN: &str = "layers.0.hc_attn_fn";
const TARGET_LAYER0_HC_ATTN_FN_LANE_SUM: &str =
    "ds4rt#deepseek-v4-target-layer0-hc-attn-fn-lane-sum";
const TARGET_LAYER0_HC_ATTN_SCALE: &str = "layers.0.hc_attn_scale";
const TARGET_LAYER0_HC_ATTN_BASE: &str = "layers.0.hc_attn_base";
const TARGET_LAYER0_ATTN_NORM: &str = "layers.0.attn_norm.weight";
const TARGET_GRAPH_ROWS: [usize; 23] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 64, 128, 256, 512, 1_024, 2_048,
];

fn target_attention_timing_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env::var(TARGET_ATTENTION_TIMING_ENV)
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false)
    })
}

fn target_attention_elapsed_ms(start: Option<Instant>) -> f64 {
    start
        .map(|start| start.elapsed().as_secs_f64() * 1_000.0)
        .unwrap_or(0.0)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::commands::real_full) struct DeepseekV4TargetDeviceRegion {
    pub(in crate::commands::real_full) offset: usize,
    pub(in crate::commands::real_full) bytes: usize,
}

impl DeepseekV4TargetDeviceRegion {
    fn end(self) -> usize {
        self.offset + self.bytes
    }
}

/// Stable target-transformer storage shared by the serving execution lanes.
///
/// Each lane owns one composite SparkInfer workspace reused by every layer.
/// Physical KV uses the shared global token-space. mHC is lane-local but is
/// bounded by the scheduler's active row frontier: it survives strict-TP4
/// expert handoffs only while that source segment traverses the transformer;
/// it is not attention history. Compressor state remains incremental per
/// active sequence. This is a planner contract until scheduler replay binds
/// the regions below.
#[derive(Clone, Debug, PartialEq)]
pub(in crate::commands::real_full) struct DeepseekV4TargetDeviceStoragePlan {
    pub(in crate::commands::real_full) variant: &'static str,
    pub(in crate::commands::real_full) cache_format: &'static str,
    pub(in crate::commands::real_full) hidden: usize,
    pub(in crate::commands::real_full) rope_theta: f32,
    pub(in crate::commands::real_full) compress_rope_theta: f32,
    pub(in crate::commands::real_full) main_rope_inv_freq: [f32; TARGET_ROPE_WIDTH / 2],
    pub(in crate::commands::real_full) compress_rope_inv_freq: [f32; TARGET_ROPE_WIDTH / 2],
    pub(in crate::commands::real_full) target_layers: usize,
    pub(in crate::commands::real_full) execution_lanes: usize,
    pub(in crate::commands::real_full) max_graph_rows: usize,
    pub(in crate::commands::real_full) max_sequence_rows: usize,
    pub(in crate::commands::real_full) hc_active_rows_per_lane: usize,
    pub(in crate::commands::real_full) physical_pool_rows: usize,
    pub(in crate::commands::real_full) source_page_tokens: usize,
    pub(in crate::commands::real_full) source_pages: usize,
    pub(in crate::commands::real_full) max_sequence_pages: usize,
    pub(in crate::commands::real_full) expert_tensor_parallel: usize,
    pub(in crate::commands::real_full) expert_parallel: bool,
    pub(in crate::commands::real_full) lane_storage: DeepseekV4TargetDeviceRegion,
    pub(in crate::commands::real_full) lane_stride_bytes: usize,
    pub(in crate::commands::real_full) workspace: DeepseekV4TargetDeviceRegion,
    pub(in crate::commands::real_full) selector_scratch: DeepseekV4TargetDeviceRegion,
    pub(in crate::commands::real_full) entry_mhc_scratch: DeepseekV4TargetDeviceRegion,
    pub(in crate::commands::real_full) hidden_input: DeepseekV4TargetDeviceRegion,
    pub(in crate::commands::real_full) hidden_work: DeepseekV4TargetDeviceRegion,
    pub(in crate::commands::real_full) hidden_output: DeepseekV4TargetDeviceRegion,
    pub(in crate::commands::real_full) positions: DeepseekV4TargetDeviceRegion,
    pub(in crate::commands::real_full) main_slots: DeepseekV4TargetDeviceRegion,
    pub(in crate::commands::real_full) cos_sin: DeepseekV4TargetDeviceRegion,
    pub(in crate::commands::real_full) selected_indices: DeepseekV4TargetDeviceRegion,
    pub(in crate::commands::real_full) selected_lengths: DeepseekV4TargetDeviceRegion,
    pub(in crate::commands::real_full) active_groups: DeepseekV4TargetDeviceRegion,
    pub(in crate::commands::real_full) group_source_starts: DeepseekV4TargetDeviceRegion,
    pub(in crate::commands::real_full) group_sequence_slots: DeepseekV4TargetDeviceRegion,
    pub(in crate::commands::real_full) group_rope_positions: DeepseekV4TargetDeviceRegion,
    pub(in crate::commands::real_full) compressed_slots: DeepseekV4TargetDeviceRegion,
    pub(in crate::commands::real_full) active_sequences: DeepseekV4TargetDeviceRegion,
    pub(in crate::commands::real_full) sequence_offsets: DeepseekV4TargetDeviceRegion,
    pub(in crate::commands::real_full) sequence_start_positions: DeepseekV4TargetDeviceRegion,
    pub(in crate::commands::real_full) state_sequence_ids: DeepseekV4TargetDeviceRegion,
    pub(in crate::commands::real_full) real_page_table: DeepseekV4TargetDeviceRegion,
    pub(in crate::commands::real_full) index_cache_seqlens: DeepseekV4TargetDeviceRegion,
    pub(in crate::commands::real_full) c128_indexed_indices: DeepseekV4TargetDeviceRegion,
    pub(in crate::commands::real_full) c128_indexed_lengths: DeepseekV4TargetDeviceRegion,
    pub(in crate::commands::real_full) residual_stage_ping: DeepseekV4TargetDeviceRegion,
    pub(in crate::commands::real_full) residual_stage_pong: DeepseekV4TargetDeviceRegion,
    pub(in crate::commands::real_full) post_stage_ping: DeepseekV4TargetDeviceRegion,
    pub(in crate::commands::real_full) post_stage_pong: DeepseekV4TargetDeviceRegion,
    pub(in crate::commands::real_full) comb_stage_ping: DeepseekV4TargetDeviceRegion,
    pub(in crate::commands::real_full) comb_stage_pong: DeepseekV4TargetDeviceRegion,
    pub(in crate::commands::real_full) residual_lane_bytes: usize,
    pub(in crate::commands::real_full) post_lane_bytes: usize,
    pub(in crate::commands::real_full) comb_lane_bytes: usize,
    pub(in crate::commands::real_full) residual_ping: DeepseekV4TargetDeviceRegion,
    pub(in crate::commands::real_full) residual_pong: DeepseekV4TargetDeviceRegion,
    pub(in crate::commands::real_full) post_ping: DeepseekV4TargetDeviceRegion,
    pub(in crate::commands::real_full) post_pong: DeepseekV4TargetDeviceRegion,
    pub(in crate::commands::real_full) comb_ping: DeepseekV4TargetDeviceRegion,
    pub(in crate::commands::real_full) comb_pong: DeepseekV4TargetDeviceRegion,
    pub(in crate::commands::real_full) physical_kv: DeepseekV4TargetDeviceRegion,
    pub(in crate::commands::real_full) compressor_state: DeepseekV4TargetDeviceRegion,
    pub(in crate::commands::real_full) total_bytes: usize,
    pub(in crate::commands::real_full) status: &'static str,
    physical_plan: DeepseekV4PhysicalKvPlan,
}

pub(in crate::commands::real_full) fn plan_deepseek_v4_target_device_storage(
    facts: &ModelFacts,
    execution_lanes: usize,
    max_graph_rows: usize,
    max_sequence_rows: usize,
    physical_pool_rows: usize,
) -> Result<DeepseekV4TargetDeviceStoragePlan> {
    plan_deepseek_v4_target_device_storage_with_cache_format(
        facts,
        execution_lanes,
        max_graph_rows,
        max_sequence_rows,
        physical_pool_rows,
        DeepseekV4KvCacheFormat::Fp8Ue8m0,
    )
}

pub(in crate::commands::real_full) fn plan_deepseek_v4_target_device_storage_with_cache_format(
    facts: &ModelFacts,
    execution_lanes: usize,
    max_graph_rows: usize,
    max_sequence_rows: usize,
    physical_pool_rows: usize,
    cache_format: DeepseekV4KvCacheFormat,
) -> Result<DeepseekV4TargetDeviceStoragePlan> {
    anyhow::ensure!(
        execution_lanes > 0,
        "target device storage requires at least one execution lane"
    );
    anyhow::ensure!(
        facts.model_type == "deepseek_v4",
        "target device storage requires model_type deepseek_v4, got {:?}",
        facts.model_type
    );
    anyhow::ensure!(
        max_graph_rows > 0 && max_graph_rows <= max_sequence_rows,
        "target device storage graph rows {max_graph_rows} must be within sequence rows {max_sequence_rows}"
    );
    anyhow::ensure!(
        physical_pool_rows >= max_sequence_rows,
        "target physical pool rows {physical_pool_rows} must cover sequence rows {max_sequence_rows}"
    );
    let variant = match facts.variant {
        ModelVariant::Flash => "flash",
        ModelVariant::Pro => "pro",
        variant => {
            anyhow::bail!("target device storage supports DeepSeek V4 Flash/Pro, got {variant:?}")
        }
    };
    let attention = DeepseekV4AttentionPlan::from_model_facts(facts)
        .context("planning native DeepSeek V4 target attention geometry")?;
    attention
        .validate_sparkinfer_sm120_contract()
        .context("validating native DeepSeek V4 target SparkInfer geometry")?;
    let physical = DeepseekV4PhysicalKvPlan::for_model_with_format(
        facts,
        physical_pool_rows,
        false,
        cache_format,
    )
    .context("planning native DeepSeek V4 physical target KV")?;
    anyhow::ensure!(
        physical.layers.len() == attention.target_layer_count
            && physical.source_page_count
                == physical_pool_rows.div_ceil(physical.source_page_tokens),
        "native target physical KV lost its target-layer/page geometry"
    );
    let max_sequence_pages = max_sequence_rows.div_ceil(physical.source_page_tokens);
    anyhow::ensure!(
        max_sequence_pages > 0 && max_sequence_pages <= physical.source_page_count,
        "native target sequence pages {max_sequence_pages} exceed physical cache pages {}",
        physical.source_page_count
    );

    let workspace_bytes = query_target_attention_arena_bytes(
        variant,
        max_graph_rows,
        physical.source_page_count,
        max_sequence_pages,
        cache_format.kernel_label(),
    )?;
    let selector_scratch_bytes = query_target_c4_selector_scratch_bytes(
        variant,
        max_graph_rows,
        physical.source_page_count,
        max_sequence_pages,
    )?;
    let entry_mhc_scratch_bytes = query_target_mhc_scratch_bytes(variant, max_graph_rows)?;
    anyhow::ensure!(
        workspace_bytes >= entry_mhc_scratch_bytes,
        "target composite workspace {workspace_bytes} is smaller than entry mHC scratch {entry_mhc_scratch_bytes}"
    );

    let graph_hidden_bytes = checked_bytes(
        &[max_graph_rows, facts.hidden_size],
        std::mem::size_of::<u16>(),
        "target graph hidden",
    )?;
    let graph_u32_bytes = checked_bytes(
        &[max_graph_rows],
        std::mem::size_of::<u32>(),
        "target graph metadata",
    )?;
    let cos_sin_bytes = checked_bytes(
        &[
            max_graph_rows
                .checked_add(TARGET_COMPRESSOR_CARRY_ROPE_SLOTS)
                .context("target graph RoPE row capacity overflow")?,
            TARGET_ROPE_WIDTH,
        ],
        std::mem::size_of::<f32>(),
        "target graph cos/sin",
    )?;
    let selected_indices_bytes = checked_bytes(
        &[max_graph_rows, TARGET_SWA_WIDTH],
        std::mem::size_of::<i32>(),
        "target graph sliding selection",
    )?;
    let hc_active_rows_per_lane =
        target_hc_active_rows_per_lane(max_graph_rows, max_sequence_rows)?;
    let residual_bytes = checked_bytes(
        &[hc_active_rows_per_lane, TARGET_HC_MULT, facts.hidden_size],
        std::mem::size_of::<u16>(),
        "target persistent HC residual",
    )?;
    let post_bytes = checked_bytes(
        &[hc_active_rows_per_lane, TARGET_HC_MULT],
        std::mem::size_of::<f32>(),
        "target persistent HC post mix",
    )?;
    let comb_bytes = checked_bytes(
        &[hc_active_rows_per_lane, TARGET_HC_MULT, TARGET_HC_MULT],
        std::mem::size_of::<f32>(),
        "target persistent HC combination mix",
    )?;

    let mut lane_cursor = 0usize;
    let workspace = push_region(&mut lane_cursor, workspace_bytes, "target workspace")?;
    let selector_scratch = push_region(
        &mut lane_cursor,
        selector_scratch_bytes,
        "target C4 selector scratch",
    )?;
    // Entry runs immediately before the layer-0 composite graph, so it reuses
    // the beginning of the same fixed workspace instead of reserving another
    // 13-23 MiB per lane.
    let entry_mhc_scratch = DeepseekV4TargetDeviceRegion {
        offset: workspace.offset,
        bytes: entry_mhc_scratch_bytes,
    };
    let hidden_input = push_region(&mut lane_cursor, graph_hidden_bytes, "target hidden input")?;
    let hidden_work = push_region(&mut lane_cursor, graph_hidden_bytes, "target hidden work")?;
    let hidden_output = push_region(&mut lane_cursor, graph_hidden_bytes, "target hidden output")?;
    let positions = push_region(&mut lane_cursor, graph_u32_bytes, "target positions")?;
    let main_slots = push_region(&mut lane_cursor, graph_u32_bytes, "target main slots")?;
    let cos_sin = push_region(&mut lane_cursor, cos_sin_bytes, "target cos/sin")?;
    let selected_indices = push_region(
        &mut lane_cursor,
        selected_indices_bytes,
        "target selected indices",
    )?;
    let selected_lengths =
        push_region(&mut lane_cursor, graph_u32_bytes, "target selected lengths")?;
    // The C4 continuation path can complete one carried group in addition to
    // the groups wholly contained in this graph chunk.
    let group_capacity = max_graph_rows
        .div_ceil(4)
        .checked_add(1)
        .context("target compressor group capacity overflow")?;
    let group_u32_bytes = checked_bytes(
        &[group_capacity],
        std::mem::size_of::<u32>(),
        "target compressor group metadata",
    )?;
    let active_groups = push_region(
        &mut lane_cursor,
        graph_u32_bytes,
        "target active compressor groups",
    )?;
    let group_source_starts = push_region(
        &mut lane_cursor,
        group_u32_bytes,
        "target compressor group source starts",
    )?;
    let group_sequence_slots = push_region(
        &mut lane_cursor,
        group_u32_bytes,
        "target compressor group sequence slots",
    )?;
    let group_rope_positions = push_region(
        &mut lane_cursor,
        group_u32_bytes,
        "target compressor group RoPE positions",
    )?;
    let compressed_slots = push_region(
        &mut lane_cursor,
        group_u32_bytes,
        "target compressed KV slots",
    )?;
    let active_sequences = push_region(
        &mut lane_cursor,
        graph_u32_bytes,
        "target active compressor sequences",
    )?;
    let sequence_offsets = push_region(
        &mut lane_cursor,
        checked_bytes(
            &[max_graph_rows + 1],
            std::mem::size_of::<u32>(),
            "target compressor sequence offsets",
        )?,
        "target compressor sequence offsets",
    )?;
    let sequence_start_positions = push_region(
        &mut lane_cursor,
        graph_u32_bytes,
        "target compressor sequence starts",
    )?;
    let state_sequence_ids = push_region(
        &mut lane_cursor,
        graph_u32_bytes,
        "target compressor state sequence IDs",
    )?;
    let real_page_table = push_region(
        &mut lane_cursor,
        checked_bytes(
            &[max_sequence_pages],
            std::mem::size_of::<u32>(),
            "target C4 shared real page table",
        )?,
        "target C4 shared real page table",
    )?;
    let index_cache_seqlens = push_region(
        &mut lane_cursor,
        graph_u32_bytes,
        "target C4 index cache lengths",
    )?;
    anyhow::ensure!(
        physical.source_page_tokens == 256,
        "target C128 metadata requires 256-token source pages, got {}",
        physical.source_page_tokens
    );
    let c128_indexed_width = target_c128_indexed_width(max_sequence_pages)?;
    let c128_indexed_indices = push_region(
        &mut lane_cursor,
        checked_bytes(
            &[c128_indexed_width],
            std::mem::size_of::<i32>(),
            "target C128 shared indexed indices",
        )?,
        "target C128 shared indexed indices",
    )?;
    let c128_indexed_lengths = push_region(
        &mut lane_cursor,
        graph_u32_bytes,
        "target C128 indexed lengths",
    )?;
    let stage_residual_bytes = checked_bytes(
        &[max_graph_rows, TARGET_HC_MULT, facts.hidden_size],
        std::mem::size_of::<u16>(),
        "target graph HC residual staging",
    )?;
    let stage_post_bytes = checked_bytes(
        &[max_graph_rows, TARGET_HC_MULT],
        std::mem::size_of::<f32>(),
        "target graph HC post staging",
    )?;
    let stage_comb_bytes = checked_bytes(
        &[max_graph_rows, TARGET_HC_MULT, TARGET_HC_MULT],
        std::mem::size_of::<f32>(),
        "target graph HC combination staging",
    )?;
    let residual_stage_ping = push_region(
        &mut lane_cursor,
        stage_residual_bytes,
        "target residual stage ping",
    )?;
    let residual_stage_pong = push_region(
        &mut lane_cursor,
        stage_residual_bytes,
        "target residual stage pong",
    )?;
    let post_stage_ping =
        push_region(&mut lane_cursor, stage_post_bytes, "target post stage ping")?;
    let post_stage_pong =
        push_region(&mut lane_cursor, stage_post_bytes, "target post stage pong")?;
    let comb_stage_ping =
        push_region(&mut lane_cursor, stage_comb_bytes, "target comb stage ping")?;
    let comb_stage_pong =
        push_region(&mut lane_cursor, stage_comb_bytes, "target comb stage pong")?;
    let lane_stride_bytes = align_up(lane_cursor, TARGET_CAPTURE_ALIGNMENT)
        .context("aligning target execution-lane stride")?;
    let lane_storage_bytes = lane_stride_bytes
        .checked_mul(execution_lanes)
        .context("target execution-lane storage bytes overflow")?;
    let lane_storage = DeepseekV4TargetDeviceRegion {
        offset: 0,
        bytes: lane_storage_bytes,
    };
    let mut cursor = lane_storage_bytes;
    let residual_plane_bytes = residual_bytes
        .checked_mul(execution_lanes)
        .context("target all-lane HC residual bytes overflow")?;
    let post_plane_bytes = post_bytes
        .checked_mul(execution_lanes)
        .context("target all-lane HC post bytes overflow")?;
    let comb_plane_bytes = comb_bytes
        .checked_mul(execution_lanes)
        .context("target all-lane HC combination bytes overflow")?;
    let residual_ping = push_region(&mut cursor, residual_plane_bytes, "target residual ping")?;
    let residual_pong = push_region(&mut cursor, residual_plane_bytes, "target residual pong")?;
    let post_ping = push_region(&mut cursor, post_plane_bytes, "target post ping")?;
    let post_pong = push_region(&mut cursor, post_plane_bytes, "target post pong")?;
    let comb_ping = push_region(&mut cursor, comb_plane_bytes, "target combination ping")?;
    let comb_pong = push_region(&mut cursor, comb_plane_bytes, "target combination pong")?;
    let physical_kv = push_region(&mut cursor, physical.persistent_bytes, "target physical KV")?;
    let compressor_state = push_region(
        &mut cursor,
        physical
            .compressor_state_bytes_per_sequence
            .checked_mul(execution_lanes)
            .context("target per-lane compressor state bytes overflow")?,
        "target compressor state",
    )?;
    let total_bytes = align_up(cursor, TARGET_CAPTURE_ALIGNMENT)
        .context("aligning target device storage total bytes")?;

    let plan = DeepseekV4TargetDeviceStoragePlan {
        variant,
        cache_format: cache_format.kernel_label(),
        hidden: facts.hidden_size,
        rope_theta: facts.rope_theta,
        compress_rope_theta: facts.compress_rope_theta,
        main_rope_inv_freq: target_default_rope_inv_freq(facts.rope_theta, facts.qk_rope_head_dim)?,
        compress_rope_inv_freq: target_yarn_rope_inv_freq(
            facts.compress_rope_theta,
            facts.qk_rope_head_dim,
            facts.rope_scaling_factor,
            facts.original_max_position_embeddings,
            facts.rope_beta_fast,
            facts.rope_beta_slow,
        )?,
        target_layers: attention.target_layer_count,
        execution_lanes,
        max_graph_rows,
        max_sequence_rows,
        hc_active_rows_per_lane,
        physical_pool_rows,
        source_page_tokens: physical.source_page_tokens,
        source_pages: physical.source_page_count,
        max_sequence_pages,
        expert_tensor_parallel: TARGET_EXPERT_TP,
        expert_parallel: false,
        lane_storage,
        lane_stride_bytes,
        workspace,
        selector_scratch,
        entry_mhc_scratch,
        hidden_input,
        hidden_work,
        hidden_output,
        positions,
        main_slots,
        cos_sin,
        selected_indices,
        selected_lengths,
        active_groups,
        group_source_starts,
        group_sequence_slots,
        group_rope_positions,
        compressed_slots,
        active_sequences,
        sequence_offsets,
        sequence_start_positions,
        state_sequence_ids,
        real_page_table,
        index_cache_seqlens,
        c128_indexed_indices,
        c128_indexed_lengths,
        residual_stage_ping,
        residual_stage_pong,
        post_stage_ping,
        post_stage_pong,
        comb_stage_ping,
        comb_stage_pong,
        residual_lane_bytes: residual_bytes,
        post_lane_bytes: post_bytes,
        comb_lane_bytes: comb_bytes,
        residual_ping,
        residual_pong,
        post_ping,
        post_pong,
        comb_ping,
        comb_pong,
        physical_kv,
        compressor_state,
        total_bytes,
        status: "planned-native-target-capture-not-active",
        physical_plan: physical,
    };
    validate_target_device_storage_plan(&plan)?;
    Ok(plan)
}

/// Owns the fixed GPU0 allocation and one capture stream per execution lane.
/// Graph sets are added to this owner as each C0/C4/C128 lifecycle is activated.
pub(in crate::commands::real_full) struct DeepseekV4TargetDeviceStorage {
    library: &'static NativeLibrary,
    plan: DeepseekV4TargetDeviceStoragePlan,
    arena: Ds4rtDeviceBuffer,
    streams: Vec<CoordinatorCudaStream>,
    entry_graphs: Vec<DeepseekV4TargetEntryGraph>,
    sliding_graphs: Vec<DeepseekV4TargetSlidingGraph>,
    c4_graphs: Vec<DeepseekV4TargetC4Graph>,
    c128_graphs: Vec<DeepseekV4TargetC128Graph>,
    post_pre_graphs: Vec<DeepseekV4TargetPostPreGraph>,
    terminal_graphs: Vec<DeepseekV4TargetTerminalGraph>,
    hc_lanes: Vec<DeepseekV4TargetHcLaneState>,
    metadata_staging: Vec<ReusableHostBuffer>,
    metadata_copy_events: Vec<Arc<CoordinatorCudaEvent>>,
    metadata_copy_in_flight: Vec<bool>,
    replay_metadata_cache:
        Vec<HashMap<DeepseekV4TargetReplayMetadataKey, DeepseekV4TargetReplayMetadata>>,
    output_ready_events: Vec<Vec<Arc<CoordinatorCudaEvent>>>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct DeepseekV4TargetHcSegmentKey {
    logical_row_start: usize,
    rows: usize,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum DeepseekV4TargetReplayMetadataFamily {
    C0,
    C4,
    C128,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct DeepseekV4TargetReplayMetadataKey {
    family: DeepseekV4TargetReplayMetadataFamily,
    logical_row_start: usize,
    rows: usize,
}

enum DeepseekV4TargetReplayMetadata {
    C0(DeepseekV4TargetPreparedReplayMetadata),
    C4(DeepseekV4TargetPreparedReplayMetadata),
    C128(DeepseekV4TargetPreparedReplayMetadata),
}

struct DeepseekV4TargetPreparedReplayMetadataEntry {
    destination: DeepseekV4TargetDeviceRegion,
    source_offset: usize,
    bytes: usize,
    label: &'static str,
}

struct DeepseekV4TargetPreparedReplayMetadata {
    staging: ReusableHostBuffer,
    entries: Vec<DeepseekV4TargetPreparedReplayMetadataEntry>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DeepseekV4TargetHcAllocation {
    row_offset: usize,
    rows: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DeepseekV4TargetHcFreeRange {
    row_offset: usize,
    rows: usize,
}

#[derive(Debug)]
struct DeepseekV4TargetHcLaneState {
    capacity_rows: usize,
    allocations: HashMap<DeepseekV4TargetHcSegmentKey, DeepseekV4TargetHcAllocation>,
    free_ranges: Vec<DeepseekV4TargetHcFreeRange>,
}

impl DeepseekV4TargetHcLaneState {
    fn new(capacity_rows: usize) -> Result<Self> {
        anyhow::ensure!(
            capacity_rows > 0,
            "target mHC lane requires nonzero row capacity"
        );
        Ok(Self {
            capacity_rows,
            allocations: HashMap::new(),
            free_ranges: vec![DeepseekV4TargetHcFreeRange {
                row_offset: 0,
                rows: capacity_rows,
            }],
        })
    }

    fn reset(&mut self) {
        self.allocations.clear();
        self.free_ranges.clear();
        self.free_ranges.push(DeepseekV4TargetHcFreeRange {
            row_offset: 0,
            rows: self.capacity_rows,
        });
    }

    fn allocate(
        &mut self,
        key: DeepseekV4TargetHcSegmentKey,
    ) -> Result<DeepseekV4TargetHcAllocation> {
        anyhow::ensure!(
            key.rows > 0 && key.rows <= self.capacity_rows,
            "target mHC segment rows {} exceed lane capacity {}",
            key.rows,
            self.capacity_rows
        );
        anyhow::ensure!(
            !self.allocations.contains_key(&key),
            "target mHC segment already exists for logical rows {}..+{}",
            key.logical_row_start,
            key.rows
        );
        let range_index = self
            .free_ranges
            .iter()
            .position(|range| range.rows >= key.rows)
            .with_context(|| {
                let free_rows = self.free_ranges.iter().map(|range| range.rows).sum::<usize>();
                format!(
                    "target mHC active frontier exhausted: requested={} free={} capacity={} active_segments={}",
                    key.rows,
                    free_rows,
                    self.capacity_rows,
                    self.allocations.len()
                )
            })?;
        let range = self.free_ranges[range_index];
        let allocation = DeepseekV4TargetHcAllocation {
            row_offset: range.row_offset,
            rows: key.rows,
        };
        if range.rows == key.rows {
            self.free_ranges.remove(range_index);
        } else {
            self.free_ranges[range_index].row_offset += key.rows;
            self.free_ranges[range_index].rows -= key.rows;
        }
        self.allocations.insert(key, allocation);
        Ok(allocation)
    }

    fn resolve(&self, key: DeepseekV4TargetHcSegmentKey) -> Result<DeepseekV4TargetHcAllocation> {
        if let Some(allocation) = self.allocations.get(&key).copied() {
            return Ok(allocation);
        }
        let logical_end = key
            .logical_row_start
            .checked_add(key.rows)
            .context("target mHC segment logical end overflow")?;
        let mut segments = self
            .allocations
            .iter()
            .filter(|(candidate, _)| {
                candidate.logical_row_start >= key.logical_row_start
                    && candidate
                        .logical_row_start
                        .checked_add(candidate.rows)
                        .is_some_and(|end| end <= logical_end)
            })
            .map(|(candidate, allocation)| (*candidate, *allocation))
            .collect::<Vec<_>>();
        segments.sort_unstable_by_key(|(candidate, _)| candidate.logical_row_start);
        let first = segments.first().copied().with_context(|| {
            format!(
                "target mHC segment is missing for logical rows {}..+{}",
                key.logical_row_start, key.rows
            )
        })?;
        let mut next_logical = key.logical_row_start;
        let mut next_physical = first.1.row_offset;
        for (candidate, allocation) in &segments {
            anyhow::ensure!(
                candidate.logical_row_start == next_logical
                    && allocation.row_offset == next_physical,
                "target mHC segments do not form one contiguous logical/physical range for rows {}..+{}",
                key.logical_row_start,
                key.rows
            );
            next_logical += candidate.rows;
            next_physical += allocation.rows;
        }
        anyhow::ensure!(
            next_logical == logical_end,
            "target mHC segments cover logical rows {}..{}, expected through {logical_end}",
            key.logical_row_start,
            next_logical
        );
        Ok(DeepseekV4TargetHcAllocation {
            row_offset: first.1.row_offset,
            rows: key.rows,
        })
    }

    fn release(
        &mut self,
        key: DeepseekV4TargetHcSegmentKey,
    ) -> Result<DeepseekV4TargetHcAllocation> {
        let allocation = self.resolve(key).with_context(|| {
            format!(
                "target mHC segment cannot be released for logical rows {}..+{}",
                key.logical_row_start, key.rows
            )
        })?;
        let logical_end = key
            .logical_row_start
            .checked_add(key.rows)
            .context("target mHC release logical end overflow")?;
        let release_keys = self
            .allocations
            .keys()
            .filter(|candidate| {
                candidate.logical_row_start >= key.logical_row_start
                    && candidate
                        .logical_row_start
                        .checked_add(candidate.rows)
                        .is_some_and(|end| end <= logical_end)
            })
            .copied()
            .collect::<Vec<_>>();
        anyhow::ensure!(
            !release_keys.is_empty(),
            "target mHC release resolved no physical allocations"
        );
        for release_key in release_keys {
            let released = self
                .allocations
                .remove(&release_key)
                .context("resolved target mHC allocation disappeared during release")?;
            self.free_ranges.push(DeepseekV4TargetHcFreeRange {
                row_offset: released.row_offset,
                rows: released.rows,
            });
        }
        self.free_ranges
            .sort_unstable_by_key(|range| range.row_offset);
        let mut merged = Vec::<DeepseekV4TargetHcFreeRange>::with_capacity(self.free_ranges.len());
        for range in self.free_ranges.drain(..) {
            if let Some(previous) = merged.last_mut() {
                if previous.row_offset + previous.rows == range.row_offset {
                    previous.rows += range.rows;
                    continue;
                }
                anyhow::ensure!(
                    previous.row_offset + previous.rows < range.row_offset,
                    "target mHC free ranges overlap"
                );
            }
            merged.push(range);
        }
        self.free_ranges = merged;
        Ok(allocation)
    }
}

struct DeepseekV4TargetEntryGraph {
    lane: usize,
    rows: usize,
    graph: CoordinatorCudaCapturedGraph,
}

struct DeepseekV4TargetSlidingGraph {
    layer_id: usize,
    lane: usize,
    rows: usize,
    graph: CoordinatorCudaCapturedGraph,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DeepseekV4TargetC4Lifecycle {
    Prefill,
    Continuation,
}

struct DeepseekV4TargetC4Graph {
    layer_id: usize,
    lane: usize,
    rows: usize,
    lifecycle: DeepseekV4TargetC4Lifecycle,
    graph: CoordinatorCudaCapturedGraph,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DeepseekV4TargetC128Lifecycle {
    Prefill,
    Continuation,
}

struct DeepseekV4TargetC128Graph {
    layer_id: usize,
    lane: usize,
    rows: usize,
    lifecycle: DeepseekV4TargetC128Lifecycle,
    graph: CoordinatorCudaCapturedGraph,
}

struct DeepseekV4TargetPostPreGraph {
    completed_layer_id: usize,
    lane: usize,
    rows: usize,
    captures_aux_hidden: bool,
    graph: CoordinatorCudaCapturedGraph,
}

struct DeepseekV4TargetTerminalGraph {
    lane: usize,
    rows: usize,
    graph: CoordinatorCudaCapturedGraph,
}

unsafe impl Send for DeepseekV4TargetDeviceStorage {}

impl DeepseekV4TargetDeviceStorage {
    pub(in crate::commands::real_full) fn new(
        plan: DeepseekV4TargetDeviceStoragePlan,
    ) -> Result<Self> {
        validate_target_device_storage_plan(&plan)?;
        let library = cuda_native_library()?;
        let mut streams = Vec::with_capacity(plan.execution_lanes);
        for lane in 0..plan.execution_lanes {
            streams.push(
                CoordinatorCudaStream::create(library).with_context(|| {
                    format!("creating native target capture stream lane {lane}")
                })?,
            );
        }
        let mut arena = library
            .alloc_device_buffer(plan.total_bytes)
            .context("allocating native target execution storage")?;
        if arena.ptr.is_null() || arena.bytes < plan.total_bytes || arena.device_id != 0 {
            let details = format!(
                "ptr={:?} bytes={} device={} expected_bytes={}",
                arena.ptr, arena.bytes, arena.device_id, plan.total_bytes
            );
            if !arena.ptr.is_null() {
                let _ = library.free_device_buffer(&mut arena);
            }
            anyhow::bail!("native target storage lost its GPU0 allocation contract: {details}");
        }
        // C4's shared-paged scorer can speculatively gather fixed graph chunks
        // beyond a row's live index-cache length before the later K bound masks
        // their logits.  Initialize the untouched pitched tails to physical
        // page zero once; replay uploads overwrite every semantically live
        // prefix and never expose an uninitialized page ID to address math.
        for lane in 0..plan.execution_lanes {
            let offset = lane
                .checked_mul(plan.lane_stride_bytes)
                .and_then(|base| base.checked_add(plan.real_page_table.offset))
                .context("target C4 page-table initialization offset overflow")?;
            let page_table = device_buffer_byte_view(
                arena,
                offset,
                plan.real_page_table.bytes,
                "target C4 initial page table",
            )?;
            library
                .cuda_zero_bytes(page_table, page_table.bytes)
                .with_context(|| format!("initializing target C4 page-table lane {lane}"))?;
        }
        let hc_lanes = (0..plan.execution_lanes)
            .map(|_| DeepseekV4TargetHcLaneState::new(plan.hc_active_rows_per_lane))
            .collect::<Result<Vec<_>>>()?;
        let metadata_staging = (0..plan.execution_lanes)
            .map(|_| ReusableHostBuffer::default())
            .collect();
        let metadata_copy_events = (0..plan.execution_lanes)
            .map(|lane| {
                CoordinatorCudaEvent::create(library)
                    .map(Arc::new)
                    .with_context(|| {
                        format!("creating native target metadata copy event lane {lane}")
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let metadata_copy_in_flight = vec![false; plan.execution_lanes];
        let replay_metadata_cache = (0..plan.execution_lanes).map(|_| HashMap::new()).collect();
        let output_ready_events = (0..plan.execution_lanes).map(|_| Vec::new()).collect();
        Ok(Self {
            library,
            plan,
            arena,
            streams,
            entry_graphs: Vec::new(),
            sliding_graphs: Vec::new(),
            c4_graphs: Vec::new(),
            c128_graphs: Vec::new(),
            post_pre_graphs: Vec::new(),
            terminal_graphs: Vec::new(),
            hc_lanes,
            metadata_staging,
            metadata_copy_events,
            metadata_copy_in_flight,
            replay_metadata_cache,
            output_ready_events,
        })
    }

    pub(in crate::commands::real_full) fn plan(&self) -> &DeepseekV4TargetDeviceStoragePlan {
        &self.plan
    }

    #[allow(dead_code)]
    pub(in crate::commands::real_full) fn stream_ptr(&self, lane: usize) -> Result<*mut c_void> {
        self.streams
            .get(lane)
            .map(CoordinatorCudaStream::as_ptr)
            .with_context(|| {
                format!(
                    "native target stream lane {lane} exceeds {} execution lanes",
                    self.plan.execution_lanes
                )
            })
    }

    fn copy_output_async(
        &mut self,
        lane: usize,
        source: Ds4rtDeviceBuffer,
        rows: usize,
        label: &'static str,
    ) -> Result<DeviceBf16Output> {
        let expected_bytes =
            validate_device_bf16_template_buffer(source, rows, self.plan.hidden, label)?;
        let mut output = device_bf16_output_uninitialized(
            rows,
            self.plan.hidden,
            CUDA_REFERENCE_DEVICE_BF16_TEMPLATE_COPY_BACKEND,
            label,
        )?;
        let stream = self.stream_ptr(lane)?;
        unsafe {
            self.library
                .copy_d2d_async(output.buffer(), source, expected_bytes, stream)
                .with_context(|| format!("copying native target output for {label}"))?;
        }
        let events = self
            .output_ready_events
            .get_mut(lane)
            .with_context(|| format!("native target output event lane {lane} is unavailable"))?;
        let ready_event =
            if let Some(event) = events.iter().find(|event| Arc::strong_count(event) == 1) {
                Arc::clone(event)
            } else {
                let event = Arc::new(CoordinatorCudaEvent::create(self.library)?);
                events.push(Arc::clone(&event));
                event
            };
        ready_event
            .record(stream)
            .with_context(|| format!("recording native target output for {label}"))?;
        output.set_ready_event(ready_event);
        Ok(output)
    }

    #[allow(dead_code)]
    fn lane_region(
        &self,
        lane: usize,
        region: DeepseekV4TargetDeviceRegion,
        label: &'static str,
    ) -> Result<Ds4rtDeviceBuffer> {
        anyhow::ensure!(
            lane < self.plan.execution_lanes && region.end() <= self.plan.lane_stride_bytes,
            "native target {label} lane/region exceeds the fixed lane layout"
        );
        let offset = lane
            .checked_mul(self.plan.lane_stride_bytes)
            .and_then(|base| base.checked_add(region.offset))
            .context("native target lane-region offset overflow")?;
        device_buffer_byte_view(self.arena, offset, region.bytes, label)
    }

    #[allow(dead_code)]
    fn hc_lane_region(
        &self,
        lane: usize,
        plane: DeepseekV4TargetDeviceRegion,
        lane_bytes: usize,
        label: &'static str,
    ) -> Result<Ds4rtDeviceBuffer> {
        anyhow::ensure!(
            lane < self.plan.execution_lanes
                && plane.bytes == lane_bytes * self.plan.execution_lanes,
            "native target {label} lost its per-lane HC plane contract"
        );
        let offset = plane
            .offset
            .checked_add(
                lane.checked_mul(lane_bytes)
                    .context("native target HC lane offset overflow")?,
            )
            .context("native target HC plane offset overflow")?;
        device_buffer_byte_view(self.arena, offset, lane_bytes, label)
    }

    pub(in crate::commands::real_full) fn reset_hc_lane(&mut self, lane: usize) -> Result<()> {
        // Prepared replay metadata is immutable CUDA-pinned source memory.
        // A lane reset must not free it while an asynchronous H2D copy can
        // still be reading it.
        self.streams
            .get(lane)
            .with_context(|| format!("native target replay metadata lane {lane} is unavailable"))?
            .synchronize()
            .context("synchronizing native target replay metadata lane reset")?;
        let state = self.hc_lanes.get_mut(lane).with_context(|| {
            format!(
                "native target mHC lane {lane} exceeds {} execution lanes",
                self.plan.execution_lanes
            )
        })?;
        state.reset();
        self.replay_metadata_cache
            .get_mut(lane)
            .with_context(|| {
                format!(
                    "native target replay metadata lane {lane} exceeds {} execution lanes",
                    self.plan.execution_lanes
                )
            })?
            .clear();
        Ok(())
    }

    pub(in crate::commands::real_full) fn allocate_fused_hc_segments(
        &mut self,
        lane: usize,
        segments: &[(usize, usize)],
    ) -> Result<()> {
        anyhow::ensure!(
            segments.len() >= 2 && segments.iter().all(|(_, rows)| *rows > 0),
            "fused target mHC allocation requires at least two non-empty segments"
        );
        let mut allocated = Vec::with_capacity(segments.len());
        for &(logical_row_start, rows) in segments {
            if let Err(error) = self.allocate_hc_segment(lane, logical_row_start, rows) {
                for &(rollback_start, rollback_rows) in allocated.iter().rev() {
                    let _ = self.release_hc_segment(lane, rollback_start, rollback_rows);
                }
                return Err(error.context("allocating fused target mHC segments"));
            }
            allocated.push((logical_row_start, rows));
        }
        let logical_row_start = segments[0].0;
        let total_rows = segments.iter().try_fold(0_usize, |total, (_, rows)| {
            total
                .checked_add(*rows)
                .context("fused target mHC row count overflow")
        })?;
        if let Err(error) = self
            .hc_segment_row_offset(lane, logical_row_start, total_rows)
            .context("validating fused target mHC segment contiguity")
        {
            for &(rollback_start, rollback_rows) in allocated.iter().rev() {
                let _ = self.release_hc_segment(lane, rollback_start, rollback_rows);
            }
            return Err(error);
        }
        Ok(())
    }

    pub(in crate::commands::real_full) fn release_fused_hc_segments(
        &mut self,
        lane: usize,
        segments: &[(usize, usize)],
    ) -> Result<()> {
        for &(logical_row_start, rows) in segments {
            self.release_hc_segment(lane, logical_row_start, rows)?;
        }
        Ok(())
    }

    fn hc_segment_key(
        &self,
        logical_row_start: usize,
        rows: usize,
    ) -> Result<DeepseekV4TargetHcSegmentKey> {
        anyhow::ensure!(
            rows > 0
                && logical_row_start
                    .checked_add(rows)
                    .is_some_and(|end| end <= self.plan.max_sequence_rows),
            "native target mHC logical rows {logical_row_start}..+{rows} exceed sequence geometry {}",
            self.plan.max_sequence_rows
        );
        Ok(DeepseekV4TargetHcSegmentKey {
            logical_row_start,
            rows,
        })
    }

    fn allocate_hc_segment(
        &mut self,
        lane: usize,
        logical_row_start: usize,
        rows: usize,
    ) -> Result<()> {
        let key = self.hc_segment_key(logical_row_start, rows)?;
        self.hc_lanes
            .get_mut(lane)
            .with_context(|| {
                format!(
                    "native target mHC lane {lane} exceeds {} execution lanes",
                    self.plan.execution_lanes
                )
            })?
            .allocate(key)?;
        Ok(())
    }

    fn hc_segment_row_offset(
        &self,
        lane: usize,
        logical_row_start: usize,
        rows: usize,
    ) -> Result<usize> {
        let key = self.hc_segment_key(logical_row_start, rows)?;
        Ok(self
            .hc_lanes
            .get(lane)
            .with_context(|| {
                format!(
                    "native target mHC lane {lane} exceeds {} execution lanes",
                    self.plan.execution_lanes
                )
            })?
            .resolve(key)?
            .row_offset)
    }

    fn release_hc_segment(
        &mut self,
        lane: usize,
        logical_row_start: usize,
        rows: usize,
    ) -> Result<()> {
        let key = self.hc_segment_key(logical_row_start, rows)?;
        let logical_end = logical_row_start
            .checked_add(rows)
            .context("native target replay metadata release range overflow")?;
        // Prepared metadata is immutable CUDA-pinned source memory. Normal
        // terminal release arrives after run_terminal has synchronized this
        // lane; retain the explicit fence here so error rollback can never
        // free an image while an asynchronous H2D copy still reads it.
        self.streams
            .get(lane)
            .with_context(|| format!("native target replay metadata lane {lane} is unavailable"))?
            .synchronize()
            .context("synchronizing native target replay metadata release")?;
        self.hc_lanes
            .get_mut(lane)
            .with_context(|| {
                format!(
                    "native target mHC lane {lane} exceeds {} execution lanes",
                    self.plan.execution_lanes
                )
            })?
            .release(key)?;
        self.replay_metadata_cache
            .get_mut(lane)
            .with_context(|| {
                format!(
                    "native target replay metadata lane {lane} exceeds {} execution lanes",
                    self.plan.execution_lanes
                )
            })?
            .retain(|metadata_key, _| {
                let metadata_end = metadata_key
                    .logical_row_start
                    .saturating_add(metadata_key.rows);
                metadata_end <= logical_row_start || metadata_key.logical_row_start >= logical_end
            });
        Ok(())
    }

    #[allow(dead_code)]
    fn physical_main_layer(&self, layer_id: usize) -> Result<Ds4rtDeviceBuffer> {
        use ds4rt_core::DeepseekV4KvRegionKind;

        self.physical_layer_region(layer_id, DeepseekV4KvRegionKind::Main)
    }

    fn physical_layer_region(
        &self,
        layer_id: usize,
        kind: ds4rt_core::DeepseekV4KvRegionKind,
    ) -> Result<Ds4rtDeviceBuffer> {
        let layer =
            self.plan.physical_plan.layer(layer_id).with_context(|| {
                format!("native target physical KV layer {layer_id} is missing")
            })?;
        let region = layer.region(kind).with_context(|| {
            format!("native target physical {kind:?} KV layer {layer_id} is missing")
        })?;
        let offset = self
            .plan
            .physical_kv
            .offset
            .checked_add(region.offset_bytes)
            .context("native target physical layer-region offset overflow")?;
        device_buffer_byte_view(
            self.arena,
            offset,
            region.length_bytes,
            "native target physical KV layer region",
        )
    }

    pub(in crate::commands::real_full) fn copy_physical_boundary_page(
        &self,
        source_page: u32,
        destination_page: u32,
        valid_source_tokens: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            source_page != destination_page,
            "native target boundary copy source and destination page are both {source_page}"
        );
        let source_page = source_page as usize;
        let destination_page = destination_page as usize;
        let copy = self
            .plan
            .physical_plan
            .boundary_copy_plan(valid_source_tokens)
            .context("planning native target radix boundary copy")?;
        for layer in copy.layers {
            for span in layer.spans {
                anyhow::ensure!(
                    source_page < span.page_count && destination_page < span.page_count,
                    "native target boundary pages source={source_page} destination={destination_page} exceed {} {:?} pages for layer {}",
                    span.page_count,
                    span.region,
                    layer.logical_layer_id,
                );
                let page_offset = |page: usize| -> Result<usize> {
                    self.plan
                        .physical_kv
                        .offset
                        .checked_add(span.region_offset_bytes)
                        .and_then(|offset| {
                            page.checked_mul(span.bytes_per_page)
                                .and_then(|page_offset| offset.checked_add(page_offset))
                        })
                        .and_then(|offset| offset.checked_add(span.offset_within_page_bytes))
                        .context("native target boundary-copy offset overflow")
                };
                let source = device_buffer_byte_view(
                    self.arena,
                    page_offset(source_page)?,
                    span.length_bytes,
                    "native target boundary-copy source",
                )?;
                let destination = device_buffer_byte_view(
                    self.arena,
                    page_offset(destination_page)?,
                    span.length_bytes,
                    "native target boundary-copy destination",
                )?;
                self.library
                    .copy_d2d(destination, source, span.length_bytes)
                    .with_context(|| {
                        format!(
                            "copying native target layer {} {:?} {:?} boundary plane",
                            layer.logical_layer_id, span.region, span.plane,
                        )
                    })?;
            }
        }
        Ok(())
    }

    fn compressor_layer_state(
        &self,
        lane: usize,
        layer_id: usize,
    ) -> Result<DeepseekV4TargetC4StateBuffers> {
        anyhow::ensure!(
            lane < self.plan.execution_lanes,
            "native target compressor-state lane {lane} exceeds {} lanes",
            self.plan.execution_lanes
        );
        let layer = self
            .plan
            .physical_plan
            .layer(layer_id)
            .with_context(|| format!("native target compressor layer {layer_id} is missing"))?;
        anyhow::ensure!(
            layer.compress_ratio == 4 && layer.compressor_state_bytes_per_sequence == 163_840,
            "native target layer {layer_id} is not a Flash C4 compressor-state layer"
        );
        let layer_prefix = self
            .plan
            .physical_plan
            .layers
            .iter()
            .take(layer_id)
            .try_fold(0_usize, |bytes, layer| {
                bytes
                    .checked_add(layer.compressor_state_bytes_per_sequence)
                    .context("native target compressor-state layer prefix overflow")
            })?;
        let lane_offset = lane
            .checked_mul(self.plan.physical_plan.compressor_state_bytes_per_sequence)
            .context("native target compressor-state lane offset overflow")?;
        let base = self
            .plan
            .compressor_state
            .offset
            .checked_add(lane_offset)
            .and_then(|offset| offset.checked_add(layer_prefix))
            .context("native target compressor-state base overflow")?;
        let state_rows = 16;
        let main_bytes = state_rows * 1_024 * std::mem::size_of::<f32>();
        let index_bytes = state_rows * 256 * std::mem::size_of::<f32>();
        let view = |offset: usize, bytes: usize, label: &'static str| {
            device_buffer_byte_view(self.arena, base + offset, bytes, label)
        };
        Ok(DeepseekV4TargetC4StateBuffers {
            main_kv: view(0, main_bytes, "native target C4 main KV state")?,
            main_score: view(main_bytes, main_bytes, "native target C4 main score state")?,
            index_kv: view(
                2 * main_bytes,
                index_bytes,
                "native target C4 index KV state",
            )?,
            index_score: view(
                2 * main_bytes + index_bytes,
                index_bytes,
                "native target C4 index score state",
            )?,
        })
    }

    fn c128_compressor_layer_state(
        &self,
        lane: usize,
        layer_id: usize,
    ) -> Result<DeepseekV4TargetC128StateBuffers> {
        anyhow::ensure!(
            lane < self.plan.execution_lanes,
            "native target C128 compressor-state lane {lane} exceeds {} lanes",
            self.plan.execution_lanes
        );
        let layer = self.plan.physical_plan.layer(layer_id).with_context(|| {
            format!("native target C128 compressor layer {layer_id} is missing")
        })?;
        anyhow::ensure!(
            layer.compress_ratio == 128 && layer.compressor_state_bytes_per_sequence == 1_048_576,
            "native target layer {layer_id} is not a Flash C128 compressor-state layer"
        );
        let layer_prefix = self
            .plan
            .physical_plan
            .layers
            .iter()
            .take(layer_id)
            .try_fold(0_usize, |bytes, layer| {
                bytes
                    .checked_add(layer.compressor_state_bytes_per_sequence)
                    .context("native target C128 compressor-state layer prefix overflow")
            })?;
        let lane_offset = lane
            .checked_mul(self.plan.physical_plan.compressor_state_bytes_per_sequence)
            .context("native target C128 compressor-state lane offset overflow")?;
        let base = self
            .plan
            .compressor_state
            .offset
            .checked_add(lane_offset)
            .and_then(|offset| offset.checked_add(layer_prefix))
            .context("native target C128 compressor-state base overflow")?;
        let state_bytes = 256 * 512 * std::mem::size_of::<f32>();
        Ok(DeepseekV4TargetC128StateBuffers {
            main_kv: device_buffer_byte_view(
                self.arena,
                base,
                state_bytes,
                "native target C128 main KV state",
            )?,
            main_score: device_buffer_byte_view(
                self.arena,
                base + state_bytes,
                state_bytes,
                "native target C128 main score state",
            )?,
        })
    }

    pub(in crate::commands::real_full) fn prepare_entry_graphs(
        &mut self,
        catalog: &TensorCatalog,
        rms_eps: f32,
        hc_eps: f32,
        sinkhorn_iters: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            rms_eps.is_finite()
                && rms_eps > 0.0
                && hc_eps.is_finite()
                && hc_eps > 0.0
                && sinkhorn_iters > 0,
            "native target entry mHC parameters must be finite and positive"
        );
        if !self.entry_graphs.is_empty() {
            anyhow::ensure!(
                self.entry_graphs_prepared(),
                "native target entry graph set is partially initialized"
            );
            return Ok(());
        }
        preload_target_layer0_hc_attn_lane_sum(catalog, self.plan.hidden)?;
        let weights = target_layer0_entry_weights(self.plan.hidden)?;
        for lane in 0..self.plan.execution_lanes {
            let stream = self.streams[lane].as_ptr();
            for rows in TARGET_GRAPH_ROWS
                .into_iter()
                .filter(|rows| *rows <= self.plan.max_graph_rows)
            {
                let buffers = self.entry_python_buffers(lane, weights)?;
                let kwargs = [
                    ("variant", PythonKernelArg::Str(self.plan.variant)),
                    ("rows", PythonKernelArg::Usize(rows)),
                    ("max_rows", PythonKernelArg::Usize(self.plan.max_graph_rows)),
                    ("rms_eps", PythonKernelArg::F64(rms_eps.into())),
                    ("hc_eps", PythonKernelArg::F64(hc_eps.into())),
                    ("sinkhorn_iters", PythonKernelArg::Usize(sinkhorn_iters)),
                    ("norm_eps", PythonKernelArg::F64(rms_eps.into())),
                ];
                launch_python_graph_capture(PythonGraphCaptureLaunch {
                    module: TARGET_MHC_CAPTURE_MODULE,
                    function: TARGET_MHC_ENTRY_PREPARE_FUNCTION,
                    cuda_stream: stream,
                    buffers: &buffers,
                    kwargs: &kwargs,
                })
                .with_context(|| {
                    format!("preparing native target entry lane {lane} bucket {rows}")
                })?;
                self.streams[lane].synchronize().with_context(|| {
                    format!("synchronizing native target entry lane {lane} bucket {rows}")
                })?;
                unsafe {
                    self.library
                        .cuda_graph_begin_capture(stream)
                        .with_context(|| {
                            format!(
                                "beginning native target entry lane {lane} bucket {rows} capture"
                            )
                        })?;
                }
                let captured = launch_python_graph_capture(PythonGraphCaptureLaunch {
                    module: TARGET_MHC_CAPTURE_MODULE,
                    function: TARGET_MHC_ENTRY_CAPTURE_FUNCTION,
                    cuda_stream: stream,
                    buffers: &buffers,
                    kwargs: &kwargs,
                });
                if let Err(error) = captured {
                    if let Ok(capture) = unsafe { self.library.cuda_graph_end_capture(stream) } {
                        let _ = unsafe { self.library.cuda_graph_exec_destroy(capture) };
                    }
                    return Err(error).with_context(|| {
                        format!("capturing native target entry lane {lane} bucket {rows}")
                    });
                }
                let capture = unsafe {
                    self.library
                        .cuda_graph_end_capture_retained(stream)
                        .with_context(|| {
                            format!("ending native target entry lane {lane} bucket {rows} capture")
                        })?
                };
                let graph = CoordinatorCudaCapturedGraph::new(self.library, capture)?;
                graph.validate_before_launch()?;
                anyhow::ensure!(
                    graph.memcpy_node_count == 0,
                    "native target entry lane {lane} bucket {rows} unexpectedly captured memcpy nodes"
                );
                self.entry_graphs
                    .push(DeepseekV4TargetEntryGraph { lane, rows, graph });
            }
        }
        Ok(())
    }

    pub(in crate::commands::real_full) fn entry_graphs_prepared(&self) -> bool {
        let buckets = TARGET_GRAPH_ROWS
            .into_iter()
            .filter(|rows| *rows <= self.plan.max_graph_rows)
            .collect::<Vec<_>>();
        self.entry_graphs.len() == self.plan.execution_lanes * buckets.len()
            && self
                .entry_graphs
                .iter()
                .zip(
                    (0..self.plan.execution_lanes)
                        .flat_map(|lane| buckets.iter().copied().map(move |rows| (lane, rows))),
                )
                .all(|(graph, expected)| {
                    (graph.lane, graph.rows) == expected && !graph.graph.as_ptr().is_null()
                })
    }

    fn entry_python_buffers(
        &self,
        lane: usize,
        weights: DeepseekV4TargetEntryWeights,
    ) -> Result<[PythonDeviceBufferArg<'static>; 10]> {
        Ok([
            target_python_device_buffer(
                "scratch",
                self.lane_region(lane, self.plan.entry_mhc_scratch, "entry mHC scratch")?,
            ),
            target_python_device_buffer(
                "residual_input",
                self.lane_region(lane, self.plan.hidden_input, "entry hidden input")?,
            ),
            target_python_device_buffer(
                "normalized_output",
                self.lane_region(lane, self.plan.hidden_work, "entry normalized output")?,
            ),
            target_python_device_buffer(
                "residual_out",
                self.lane_region(
                    lane,
                    self.plan.residual_stage_ping,
                    "entry residual staging output",
                )?,
            ),
            target_python_device_buffer(
                "post_out",
                self.lane_region(lane, self.plan.post_stage_ping, "entry post staging output")?,
            ),
            target_python_device_buffer(
                "comb_out",
                self.lane_region(lane, self.plan.comb_stage_ping, "entry comb staging output")?,
            ),
            target_python_device_buffer("hc_fn", weights.hc_fn),
            target_python_device_buffer("hc_scale", weights.hc_scale),
            target_python_device_buffer("hc_base", weights.hc_base),
            target_python_device_buffer("norm_weight", weights.norm_weight),
        ])
    }

    pub(in crate::commands::real_full) fn prepare_sliding_attention_graphs(
        &mut self,
        catalog: &TensorCatalog,
        producer_eps: f32,
        rms_eps: f32,
        hc_eps: f32,
        sinkhorn_iters: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            producer_eps.is_finite()
                && producer_eps > 0.0
                && rms_eps.is_finite()
                && rms_eps > 0.0
                && hc_eps.is_finite()
                && hc_eps > 0.0
                && sinkhorn_iters > 0,
            "native target sliding-attention parameters must be finite and positive"
        );
        let attention = DeepseekV4AttentionPlan::from_model_facts(&catalog.facts)
            .context("planning native target sliding-attention graph geometry")?;
        let sliding_layers = attention
            .target_layers()
            .iter()
            .filter(|layer| layer.compress_ratio == 0)
            .map(|layer| layer.logical_layer_id)
            .collect::<Vec<_>>();
        if sliding_layers.is_empty() {
            anyhow::ensure!(
                self.sliding_graphs.is_empty(),
                "native target sliding-attention graph set exists for a topology with no C0 target layers"
            );
            return Ok(());
        }
        if !self.sliding_graphs.is_empty() {
            anyhow::ensure!(
                self.sliding_graphs_prepared(&sliding_layers),
                "native target sliding-attention graph set is partially initialized"
            );
            return Ok(());
        }
        let buckets = target_graph_row_buckets(self.plan.max_graph_rows);
        for lane in 0..self.plan.execution_lanes {
            self.initialize_sliding_capture_metadata(lane, &self.plan.main_rope_inv_freq)?;
        }
        for layer_id in sliding_layers.iter().copied() {
            let weights = target_sliding_attention_weights(&attention, layer_id)?;
            for lane in 0..self.plan.execution_lanes {
                let stream = self.streams[lane].as_ptr();
                for rows in buckets.iter().copied() {
                    let buffers = self.sliding_attention_python_buffers(lane, layer_id, weights)?;
                    let kwargs = [
                        ("variant", PythonKernelArg::Str(self.plan.variant)),
                        ("mode", PythonKernelArg::Str("extend")),
                        ("rows", PythonKernelArg::Usize(rows)),
                        ("max_rows", PythonKernelArg::Usize(self.plan.max_graph_rows)),
                        (
                            "source_pages",
                            PythonKernelArg::Usize(self.plan.source_pages),
                        ),
                        (
                            "max_page_table_width",
                            PythonKernelArg::Usize(self.plan.max_sequence_pages),
                        ),
                        (
                            "max_positions",
                            PythonKernelArg::Usize(self.plan.max_graph_rows),
                        ),
                        ("swa_width", PythonKernelArg::Usize(TARGET_SWA_WIDTH)),
                        ("cache_format", PythonKernelArg::Str(self.plan.cache_format)),
                        ("producer_eps", PythonKernelArg::F64(producer_eps.into())),
                        ("rms_eps", PythonKernelArg::F64(rms_eps.into())),
                        ("hc_eps", PythonKernelArg::F64(hc_eps.into())),
                        ("sinkhorn_iters", PythonKernelArg::Usize(sinkhorn_iters)),
                        ("norm_eps", PythonKernelArg::F64(rms_eps.into())),
                    ];
                    launch_python_graph_capture(PythonGraphCaptureLaunch {
                        module: TARGET_ATTENTION_CAPTURE_MODULE,
                        function: TARGET_SLIDING_PREPARE_FUNCTION,
                        cuda_stream: stream,
                        buffers: &buffers,
                        kwargs: &kwargs,
                    })
                    .with_context(|| {
                        format!(
                            "preparing native target C0 layer {layer_id} lane {lane} bucket {rows}"
                        )
                    })?;
                    self.streams[lane].synchronize().with_context(|| {
                        format!(
                            "synchronizing native target C0 layer {layer_id} lane {lane} bucket {rows}"
                        )
                    })?;
                    unsafe {
                        self.library.cuda_graph_begin_capture(stream).with_context(|| {
                            format!(
                                "beginning native target C0 layer {layer_id} lane {lane} bucket {rows} capture"
                            )
                        })?;
                    }
                    let captured = launch_python_graph_capture(PythonGraphCaptureLaunch {
                        module: TARGET_ATTENTION_CAPTURE_MODULE,
                        function: TARGET_SLIDING_CAPTURE_FUNCTION,
                        cuda_stream: stream,
                        buffers: &buffers,
                        kwargs: &kwargs,
                    });
                    if let Err(error) = captured {
                        if let Ok(capture) = unsafe { self.library.cuda_graph_end_capture(stream) }
                        {
                            let _ = unsafe { self.library.cuda_graph_exec_destroy(capture) };
                        }
                        return Err(error).with_context(|| {
                            format!(
                                "capturing native target C0 layer {layer_id} lane {lane} bucket {rows}"
                            )
                        });
                    }
                    let capture = unsafe {
                        self.library
                            .cuda_graph_end_capture_retained(stream)
                            .with_context(|| {
                                format!(
                                    "ending native target C0 layer {layer_id} lane {lane} bucket {rows} capture"
                                )
                            })?
                    };
                    let graph = CoordinatorCudaCapturedGraph::new(self.library, capture)?;
                    graph.validate_before_launch()?;
                    anyhow::ensure!(
                        graph.memcpy_node_count == 0,
                        "native target C0 layer {layer_id} lane {lane} bucket {rows} unexpectedly captured memcpy nodes"
                    );
                    self.sliding_graphs.push(DeepseekV4TargetSlidingGraph {
                        layer_id,
                        lane,
                        rows,
                        graph,
                    });
                }
            }
        }
        Ok(())
    }

    fn sliding_graphs_prepared(&self, sliding_layers: &[usize]) -> bool {
        let buckets = target_graph_row_buckets(self.plan.max_graph_rows);
        self.sliding_graphs.len()
            == sliding_layers.len() * self.plan.execution_lanes * buckets.len()
            && self
                .sliding_graphs
                .iter()
                .zip(sliding_layers.iter().copied().flat_map(|layer_id| {
                    (0..self.plan.execution_lanes).flat_map({
                        let buckets = buckets.clone();
                        move |lane| {
                            buckets
                                .clone()
                                .into_iter()
                                .map(move |rows| (layer_id, lane, rows))
                        }
                    })
                }))
                .all(|(graph, expected)| {
                    (graph.layer_id, graph.lane, graph.rows) == expected
                        && !graph.graph.as_ptr().is_null()
                })
    }

    fn initialize_sliding_capture_metadata(
        &self,
        lane: usize,
        rope_inv_freq: &[f32; TARGET_ROPE_WIDTH / 2],
    ) -> Result<()> {
        anyhow::ensure!(
            rope_inv_freq
                .iter()
                .all(|value| value.is_finite() && *value > 0.0),
            "native target capture RoPE frequencies must be finite and positive"
        );
        let rows = self.plan.max_graph_rows;
        let positions = (0..rows).map(|row| row as u32).collect::<Vec<_>>();
        let main_slots = positions.clone();
        let mut cos_sin = vec![0.0_f32; rows * TARGET_ROPE_WIDTH];
        let pairs = TARGET_ROPE_WIDTH / 2;
        for row in 0..rows {
            for pair in 0..pairs {
                let angle = row as f32 * rope_inv_freq[pair];
                let (sin, cos) = angle.sin_cos();
                cos_sin[row * TARGET_ROPE_WIDTH + pair] = cos;
                cos_sin[row * TARGET_ROPE_WIDTH + pairs + pair] = sin;
            }
        }
        let mut selected_indices = vec![0_i32; rows * TARGET_SWA_WIDTH];
        let mut selected_lengths = vec![0_i32; rows];
        for row in 0..rows {
            let first = (row + 1).saturating_sub(TARGET_SWA_WIDTH);
            let length = row + 1 - first;
            selected_lengths[row] = length as i32;
            for (column, slot) in (first..=row).enumerate() {
                selected_indices[row * TARGET_SWA_WIDTH + column] = slot as i32;
            }
        }
        self.library.copy_h2d(
            self.lane_region(lane, self.plan.positions, "capture positions")?,
            u32_host_bytes(&positions),
        )?;
        self.library.copy_h2d(
            self.lane_region(lane, self.plan.main_slots, "capture main slots")?,
            u32_host_bytes(&main_slots),
        )?;
        self.library.copy_h2d(
            self.lane_region(lane, self.plan.cos_sin, "capture cos/sin")?,
            f32_host_bytes(&cos_sin),
        )?;
        self.library.copy_h2d(
            self.lane_region(lane, self.plan.selected_indices, "capture selected indices")?,
            i32_host_bytes(&selected_indices),
        )?;
        self.library.copy_h2d(
            self.lane_region(lane, self.plan.selected_lengths, "capture selected lengths")?,
            i32_host_bytes(&selected_lengths),
        )?;
        Ok(())
    }

    fn sliding_attention_python_buffers(
        &self,
        lane: usize,
        layer_id: usize,
        weights: DeepseekV4TargetSlidingWeights,
    ) -> Result<[PythonDeviceBufferArg<'static>; 32]> {
        Ok([
            target_python_device_buffer(
                "workspace",
                self.lane_region(lane, self.plan.workspace, "C0 workspace")?,
            ),
            target_python_device_buffer(
                "hidden_states",
                self.lane_region(lane, self.plan.hidden_work, "C0 normalized attention input")?,
            ),
            target_python_device_buffer(
                "normalized_output",
                self.lane_region(lane, self.plan.hidden_output, "C0 normalized FFN output")?,
            ),
            target_python_device_buffer(
                "positions",
                self.lane_region(lane, self.plan.positions, "C0 positions")?,
            ),
            target_python_device_buffer(
                "main_slots",
                self.lane_region(lane, self.plan.main_slots, "C0 main slots")?,
            ),
            target_python_device_buffer(
                "cos_sin_cache",
                self.lane_region(lane, self.plan.cos_sin, "C0 cos/sin")?,
            ),
            target_python_device_buffer("main_kv_cache", self.physical_main_layer(layer_id)?),
            target_python_device_buffer(
                "selected_indices",
                self.lane_region(lane, self.plan.selected_indices, "C0 selected indices")?,
            ),
            target_python_device_buffer(
                "selected_lengths",
                self.lane_region(lane, self.plan.selected_lengths, "C0 selected lengths")?,
            ),
            target_python_device_buffer(
                "residual",
                self.lane_region(lane, self.plan.residual_stage_ping, "C0 residual input")?,
            ),
            target_python_device_buffer(
                "prev_post",
                self.lane_region(lane, self.plan.post_stage_ping, "C0 post input")?,
            ),
            target_python_device_buffer(
                "prev_comb",
                self.lane_region(lane, self.plan.comb_stage_ping, "C0 combination input")?,
            ),
            target_python_device_buffer(
                "residual_out",
                self.lane_region(lane, self.plan.residual_stage_pong, "C0 residual output")?,
            ),
            target_python_device_buffer(
                "post_out",
                self.lane_region(lane, self.plan.post_stage_pong, "C0 post output")?,
            ),
            target_python_device_buffer(
                "comb_out",
                self.lane_region(lane, self.plan.comb_stage_pong, "C0 combination output")?,
            ),
            target_python_device_buffer("wq_a_weight", weights.wq_a_weight),
            target_python_device_buffer("wq_a_scale", weights.wq_a_scale),
            target_python_device_buffer("wq_b_weight", weights.wq_b_weight),
            target_python_device_buffer("wq_b_scale", weights.wq_b_scale),
            target_python_device_buffer("wkv_weight", weights.wkv_weight),
            target_python_device_buffer("wkv_scale", weights.wkv_scale),
            target_python_device_buffer("q_norm_weight", weights.q_norm_weight),
            target_python_device_buffer("kv_norm_weight", weights.kv_norm_weight),
            target_python_device_buffer("wo_a_weight", weights.wo_a_weight),
            target_python_device_buffer("wo_a_scale", weights.wo_a_scale),
            target_python_device_buffer("wo_b_weight", weights.wo_b_weight),
            target_python_device_buffer("wo_b_scale", weights.wo_b_scale),
            target_python_device_buffer("attn_sink", weights.attn_sink),
            target_python_device_buffer("hc_fn", weights.hc_fn),
            target_python_device_buffer("hc_scale", weights.hc_scale),
            target_python_device_buffer("hc_base", weights.hc_base),
            target_python_device_buffer("norm_weight", weights.norm_weight),
        ])
    }

    #[allow(clippy::too_many_arguments)]
    fn capture_compressed_attention_graph(
        &self,
        family: &'static str,
        lifecycle: &'static str,
        layer_id: usize,
        lane: usize,
        rows: usize,
        prepare_function: &'static str,
        capture_function: &'static str,
        prepare: bool,
        buffers: &[PythonDeviceBufferArg<'static>],
        kwargs: &[(&'static str, PythonKernelArg<'static>)],
    ) -> Result<CoordinatorCudaCapturedGraph> {
        let stream = self.streams[lane].as_ptr();
        let label = || {
            format!("native target {family} {lifecycle} layer {layer_id} lane {lane} bucket {rows}")
        };
        if prepare {
            launch_python_graph_capture(PythonGraphCaptureLaunch {
                module: TARGET_ATTENTION_CAPTURE_MODULE,
                function: prepare_function,
                cuda_stream: stream,
                buffers,
                kwargs,
            })
            .with_context(|| format!("preparing {}", label()))?;
        }
        self.streams[lane]
            .synchronize()
            .with_context(|| format!("synchronizing {}", label()))?;
        unsafe {
            self.library
                .cuda_graph_begin_capture(stream)
                .with_context(|| format!("beginning {} capture", label()))?;
        }
        let captured = launch_python_graph_capture(PythonGraphCaptureLaunch {
            module: TARGET_ATTENTION_CAPTURE_MODULE,
            function: capture_function,
            cuda_stream: stream,
            buffers,
            kwargs,
        });
        if let Err(error) = captured {
            if let Ok(capture) = unsafe { self.library.cuda_graph_end_capture(stream) } {
                let _ = unsafe { self.library.cuda_graph_exec_destroy(capture) };
            }
            return Err(error).with_context(|| format!("capturing {}", label()));
        }
        let capture = unsafe {
            self.library
                .cuda_graph_end_capture_retained(stream)
                .with_context(|| format!("ending {} capture", label()))?
        };
        let graph = CoordinatorCudaCapturedGraph::new(self.library, capture)?;
        graph.validate_before_launch()?;
        anyhow::ensure!(
            graph.memcpy_node_count == 0,
            "{} unexpectedly captured memcpy nodes",
            label()
        );
        Ok(graph)
    }

    pub(in crate::commands::real_full) fn prepare_c4_attention_graphs(
        &mut self,
        catalog: &TensorCatalog,
        producer_eps: f32,
        rms_eps: f32,
        hc_eps: f32,
        sinkhorn_iters: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            producer_eps.is_finite()
                && producer_eps > 0.0
                && rms_eps.is_finite()
                && rms_eps > 0.0
                && hc_eps.is_finite()
                && hc_eps > 0.0
                && sinkhorn_iters > 0,
            "native target C4 parameters must be finite and positive"
        );
        let attention = DeepseekV4AttentionPlan::from_model_facts(&catalog.facts)
            .context("planning native target C4 graph geometry")?;
        let layer_ids = target_connected_compressed_layer_ids(&attention, 4);
        anyhow::ensure!(
            !layer_ids.is_empty(),
            "native target graph set has no connected C4 layer"
        );
        if !self.c4_graphs.is_empty() {
            anyhow::ensure!(
                self.c4_graphs_prepared(&layer_ids),
                "native target C4 graph set is partially initialized"
            );
            return Ok(());
        }
        let buckets = target_graph_row_buckets(self.plan.max_graph_rows);
        let compress_rope_inv_freq = self.plan.compress_rope_inv_freq;
        for (layer_index, layer_id) in layer_ids.iter().copied().enumerate() {
            let weights = target_c4_attention_weights(&attention, layer_id)?;
            for lane in 0..self.plan.execution_lanes {
                for lifecycle in [
                    DeepseekV4TargetC4Lifecycle::Prefill,
                    DeepseekV4TargetC4Lifecycle::Continuation,
                ] {
                    for rows in buckets.iter().copied() {
                        let prepare = lane == 0
                            && (layer_index == 0
                                || (lifecycle == DeepseekV4TargetC4Lifecycle::Prefill
                                    && rows == buckets[0]));
                        // Capture records kernels without executing them. Only the
                        // prepare replay consumes the seeded metadata; requests upload
                        // their own metadata before every serving graph launch.
                        if prepare {
                            self.initialize_c4_capture_metadata(
                                lane,
                                rows,
                                lifecycle,
                                &compress_rope_inv_freq,
                            )?;
                        }
                        let buffers = self.c4_attention_python_buffers(lane, layer_id, weights)?;
                        let kwargs = [
                            ("variant", PythonKernelArg::Str(self.plan.variant)),
                            ("rows", PythonKernelArg::Usize(rows)),
                            ("max_rows", PythonKernelArg::Usize(self.plan.max_graph_rows)),
                            (
                                "source_pages",
                                PythonKernelArg::Usize(self.plan.source_pages),
                            ),
                            (
                                "max_page_table_width",
                                PythonKernelArg::Usize(self.plan.max_sequence_pages),
                            ),
                            (
                                "max_positions",
                                PythonKernelArg::Usize(self.plan.max_graph_rows),
                            ),
                            ("swa_width", PythonKernelArg::Usize(TARGET_SWA_WIDTH)),
                            ("cache_format", PythonKernelArg::Str(self.plan.cache_format)),
                            ("producer_eps", PythonKernelArg::F64(producer_eps.into())),
                            ("compressor_eps", PythonKernelArg::F64(rms_eps.into())),
                            ("rms_eps", PythonKernelArg::F64(rms_eps.into())),
                            ("hc_eps", PythonKernelArg::F64(hc_eps.into())),
                            ("sinkhorn_iters", PythonKernelArg::Usize(sinkhorn_iters)),
                            ("norm_eps", PythonKernelArg::F64(rms_eps.into())),
                        ];
                        let (lifecycle_label, prepare_function, capture_function) = match lifecycle
                        {
                            DeepseekV4TargetC4Lifecycle::Prefill => (
                                "prefill",
                                TARGET_C4_PREFILL_PREPARE_FUNCTION,
                                TARGET_C4_PREFILL_CAPTURE_FUNCTION,
                            ),
                            DeepseekV4TargetC4Lifecycle::Continuation => (
                                "continuation",
                                TARGET_C4_CONTINUATION_PREPARE_FUNCTION,
                                TARGET_C4_CONTINUATION_CAPTURE_FUNCTION,
                            ),
                        };
                        let graph = self.capture_compressed_attention_graph(
                            "C4",
                            lifecycle_label,
                            layer_id,
                            lane,
                            rows,
                            prepare_function,
                            capture_function,
                            prepare,
                            &buffers,
                            &kwargs,
                        )?;
                        self.c4_graphs.push(DeepseekV4TargetC4Graph {
                            layer_id,
                            lane,
                            rows,
                            lifecycle,
                            graph,
                        });
                    }
                }
            }
        }
        Ok(())
    }

    fn c4_graphs_prepared(&self, layer_ids: &[usize]) -> bool {
        let buckets = target_graph_row_buckets(self.plan.max_graph_rows);
        self.c4_graphs.len() == layer_ids.len() * self.plan.execution_lanes * 2 * buckets.len()
            && self
                .c4_graphs
                .iter()
                .zip(layer_ids.iter().copied().flat_map(|layer_id| {
                    (0..self.plan.execution_lanes).flat_map({
                        let buckets = buckets.clone();
                        move |lane| {
                            [
                                DeepseekV4TargetC4Lifecycle::Prefill,
                                DeepseekV4TargetC4Lifecycle::Continuation,
                            ]
                            .into_iter()
                            .flat_map({
                                let buckets = buckets.clone();
                                move |lifecycle| {
                                    buckets
                                        .clone()
                                        .into_iter()
                                        .map(move |rows| (layer_id, lane, lifecycle, rows))
                                }
                            })
                        }
                    })
                }))
                .all(|(graph, expected)| {
                    (graph.layer_id, graph.lane, graph.lifecycle, graph.rows) == expected
                        && !graph.graph.as_ptr().is_null()
                })
    }

    fn c4_attention_python_buffers(
        &self,
        lane: usize,
        layer_id: usize,
        weights: DeepseekV4TargetC4Weights,
    ) -> Result<Vec<PythonDeviceBufferArg<'static>>> {
        use ds4rt_core::DeepseekV4KvRegionKind;

        let state = self.compressor_layer_state(lane, layer_id)?;
        Ok(vec![
            target_python_device_buffer(
                "workspace",
                self.lane_region(lane, self.plan.workspace, "C4 workspace")?,
            ),
            target_python_device_buffer(
                "selector_scratch",
                self.lane_region(lane, self.plan.selector_scratch, "C4 selector scratch")?,
            ),
            target_python_device_buffer(
                "hidden_states",
                self.lane_region(lane, self.plan.hidden_work, "C4 normalized attention input")?,
            ),
            target_python_device_buffer(
                "normalized_output",
                self.lane_region(lane, self.plan.hidden_output, "C4 normalized FFN output")?,
            ),
            target_python_device_buffer(
                "positions",
                self.lane_region(lane, self.plan.positions, "C4 positions")?,
            ),
            target_python_device_buffer(
                "main_slots",
                self.lane_region(lane, self.plan.main_slots, "C4 main slots")?,
            ),
            target_python_device_buffer(
                "cos_sin_cache",
                self.lane_region(lane, self.plan.cos_sin, "C4 cos/sin")?,
            ),
            target_python_device_buffer("main_kv_cache", self.physical_main_layer(layer_id)?),
            target_python_device_buffer(
                "swa_indices",
                self.lane_region(lane, self.plan.selected_indices, "C4 SWA indices")?,
            ),
            target_python_device_buffer(
                "swa_lengths",
                self.lane_region(lane, self.plan.selected_lengths, "C4 SWA lengths")?,
            ),
            target_python_device_buffer(
                "active_groups",
                self.lane_region(lane, self.plan.active_groups, "C4 active groups")?,
            ),
            target_python_device_buffer(
                "group_source_starts",
                self.lane_region(
                    lane,
                    self.plan.group_source_starts,
                    "C4 group source starts",
                )?,
            ),
            target_python_device_buffer(
                "group_sequence_slots",
                self.lane_region(
                    lane,
                    self.plan.group_sequence_slots,
                    "C4 group sequence slots",
                )?,
            ),
            target_python_device_buffer(
                "group_rope_positions",
                self.lane_region(
                    lane,
                    self.plan.group_rope_positions,
                    "C4 group RoPE positions",
                )?,
            ),
            target_python_device_buffer(
                "compressed_slots",
                self.lane_region(lane, self.plan.compressed_slots, "C4 compressed slots")?,
            ),
            target_python_device_buffer(
                "active_sequences",
                self.lane_region(lane, self.plan.active_sequences, "C4 active sequences")?,
            ),
            target_python_device_buffer(
                "sequence_offsets",
                self.lane_region(lane, self.plan.sequence_offsets, "C4 sequence offsets")?,
            ),
            target_python_device_buffer(
                "sequence_start_positions",
                self.lane_region(
                    lane,
                    self.plan.sequence_start_positions,
                    "C4 sequence start positions",
                )?,
            ),
            target_python_device_buffer(
                "state_sequence_ids",
                self.lane_region(lane, self.plan.state_sequence_ids, "C4 state sequence IDs")?,
            ),
            target_python_device_buffer(
                "compressed_main_cache",
                self.physical_layer_region(layer_id, DeepseekV4KvRegionKind::Compressed)?,
            ),
            target_python_device_buffer("main_kv_state", state.main_kv),
            target_python_device_buffer("main_score_state", state.main_score),
            target_python_device_buffer(
                "index_cache",
                self.physical_layer_region(layer_id, DeepseekV4KvRegionKind::Indexer)?,
            ),
            target_python_device_buffer("index_kv_state", state.index_kv),
            target_python_device_buffer("index_score_state", state.index_score),
            target_python_device_buffer(
                "real_page_table",
                self.lane_region(lane, self.plan.real_page_table, "C4 real page table")?,
            ),
            target_python_device_buffer(
                "index_cache_seqlens",
                self.lane_region(
                    lane,
                    self.plan.index_cache_seqlens,
                    "C4 index cache lengths",
                )?,
            ),
            target_python_device_buffer(
                "residual",
                self.lane_region(lane, self.plan.residual_stage_ping, "C4 residual input")?,
            ),
            target_python_device_buffer(
                "prev_post",
                self.lane_region(lane, self.plan.post_stage_ping, "C4 post input")?,
            ),
            target_python_device_buffer(
                "prev_comb",
                self.lane_region(lane, self.plan.comb_stage_ping, "C4 combination input")?,
            ),
            target_python_device_buffer(
                "residual_out",
                self.lane_region(lane, self.plan.residual_stage_pong, "C4 residual output")?,
            ),
            target_python_device_buffer(
                "post_out",
                self.lane_region(lane, self.plan.post_stage_pong, "C4 post output")?,
            ),
            target_python_device_buffer(
                "comb_out",
                self.lane_region(lane, self.plan.comb_stage_pong, "C4 combination output")?,
            ),
            target_python_device_buffer("wq_a_weight", weights.common.wq_a_weight),
            target_python_device_buffer("wq_a_scale", weights.common.wq_a_scale),
            target_python_device_buffer("wq_b_weight", weights.common.wq_b_weight),
            target_python_device_buffer("wq_b_scale", weights.common.wq_b_scale),
            target_python_device_buffer("wkv_weight", weights.common.wkv_weight),
            target_python_device_buffer("wkv_scale", weights.common.wkv_scale),
            target_python_device_buffer("q_norm_weight", weights.common.q_norm_weight),
            target_python_device_buffer("kv_norm_weight", weights.common.kv_norm_weight),
            target_python_device_buffer("compressor_main_wkv", weights.main_wkv),
            target_python_device_buffer("compressor_main_wgate", weights.main_wgate),
            target_python_device_buffer("compressor_main_ape", weights.main_ape),
            target_python_device_buffer("compressor_main_norm", weights.main_norm),
            target_python_device_buffer("compressor_index_wkv", weights.index_wkv),
            target_python_device_buffer("compressor_index_wgate", weights.index_wgate),
            target_python_device_buffer("compressor_index_ape", weights.index_ape),
            target_python_device_buffer("compressor_index_norm", weights.index_norm),
            target_python_device_buffer("indexer_wq_weight", weights.indexer_wq_weight),
            target_python_device_buffer("indexer_wq_scale", weights.indexer_wq_scale),
            target_python_device_buffer(
                "indexer_weights_projection",
                weights.indexer_weights_projection,
            ),
            target_python_device_buffer("wo_a_weight", weights.common.wo_a_weight),
            target_python_device_buffer("wo_a_scale", weights.common.wo_a_scale),
            target_python_device_buffer("wo_b_weight", weights.common.wo_b_weight),
            target_python_device_buffer("wo_b_scale", weights.common.wo_b_scale),
            target_python_device_buffer("attn_sink", weights.common.attn_sink),
            target_python_device_buffer("hc_fn", weights.common.hc_fn),
            target_python_device_buffer("hc_scale", weights.common.hc_scale),
            target_python_device_buffer("hc_base", weights.common.hc_base),
            target_python_device_buffer("norm_weight", weights.common.norm_weight),
        ])
    }

    fn initialize_c4_capture_metadata(
        &mut self,
        lane: usize,
        rows: usize,
        lifecycle: DeepseekV4TargetC4Lifecycle,
        rope_inv_freq: &[f32; TARGET_ROPE_WIDTH / 2],
    ) -> Result<()> {
        let logical_row_start = match lifecycle {
            DeepseekV4TargetC4Lifecycle::Prefill => 0,
            DeepseekV4TargetC4Lifecycle::Continuation => 4,
        };
        let physical_slots = (0..logical_row_start + rows)
            .map(|slot| u32::try_from(slot).context("C4 capture slot exceeds u32"))
            .collect::<Result<Vec<_>>>()?;
        let metadata = target_c4_replay_metadata(
            logical_row_start,
            rows,
            &physical_slots,
            rope_inv_freq,
            self.plan.max_graph_rows,
            self.plan.max_sequence_pages,
        )?;
        self.upload_c4_metadata(lane, &metadata)
    }

    pub(in crate::commands::real_full) fn prepare_c128_attention_graphs(
        &mut self,
        catalog: &TensorCatalog,
        producer_eps: f32,
        rms_eps: f32,
        hc_eps: f32,
        sinkhorn_iters: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            producer_eps.is_finite()
                && producer_eps > 0.0
                && rms_eps.is_finite()
                && rms_eps > 0.0
                && hc_eps.is_finite()
                && hc_eps > 0.0
                && sinkhorn_iters > 0,
            "native target C128 parameters must be finite and positive"
        );
        let attention = DeepseekV4AttentionPlan::from_model_facts(&catalog.facts)
            .context("planning native target C128 graph geometry")?;
        let layer_ids = target_connected_compressed_layer_ids(&attention, 128);
        anyhow::ensure!(
            !layer_ids.is_empty(),
            "native target graph set has no connected C128 layer"
        );
        if !self.c128_graphs.is_empty() {
            anyhow::ensure!(
                self.c128_graphs_prepared(&layer_ids),
                "native target C128 graph set is partially initialized"
            );
            return Ok(());
        }
        let buckets = target_graph_row_buckets(self.plan.max_graph_rows);
        let compress_rope_inv_freq = self.plan.compress_rope_inv_freq;
        for (layer_index, layer_id) in layer_ids.iter().copied().enumerate() {
            let weights = target_c128_attention_weights(&attention, layer_id)?;
            for lane in 0..self.plan.execution_lanes {
                for lifecycle in [
                    DeepseekV4TargetC128Lifecycle::Prefill,
                    DeepseekV4TargetC128Lifecycle::Continuation,
                ] {
                    for rows in buckets.iter().copied() {
                        let prepare = lane == 0
                            && (layer_index == 0
                                || (lifecycle == DeepseekV4TargetC128Lifecycle::Prefill
                                    && rows == buckets[0]));
                        // Capture records kernels without executing them. Only the
                        // prepare replay consumes the seeded metadata; requests upload
                        // their own metadata before every serving graph launch.
                        if prepare {
                            self.initialize_c128_capture_metadata(
                                lane,
                                rows,
                                lifecycle,
                                &compress_rope_inv_freq,
                            )?;
                        }
                        let buffers =
                            self.c128_attention_python_buffers(lane, layer_id, weights)?;
                        let kwargs = [
                            ("variant", PythonKernelArg::Str(self.plan.variant)),
                            ("rows", PythonKernelArg::Usize(rows)),
                            ("max_rows", PythonKernelArg::Usize(self.plan.max_graph_rows)),
                            (
                                "source_pages",
                                PythonKernelArg::Usize(self.plan.source_pages),
                            ),
                            (
                                "max_page_table_width",
                                PythonKernelArg::Usize(self.plan.max_sequence_pages),
                            ),
                            (
                                "max_positions",
                                PythonKernelArg::Usize(self.plan.max_graph_rows),
                            ),
                            ("swa_width", PythonKernelArg::Usize(TARGET_SWA_WIDTH)),
                            ("cache_format", PythonKernelArg::Str(self.plan.cache_format)),
                            ("producer_eps", PythonKernelArg::F64(producer_eps.into())),
                            ("compressor_eps", PythonKernelArg::F64(rms_eps.into())),
                            ("rms_eps", PythonKernelArg::F64(rms_eps.into())),
                            ("hc_eps", PythonKernelArg::F64(hc_eps.into())),
                            ("sinkhorn_iters", PythonKernelArg::Usize(sinkhorn_iters)),
                            ("norm_eps", PythonKernelArg::F64(rms_eps.into())),
                        ];
                        let (lifecycle_label, prepare_function, capture_function) = match lifecycle
                        {
                            DeepseekV4TargetC128Lifecycle::Prefill => (
                                "prefill",
                                TARGET_C128_PREFILL_PREPARE_FUNCTION,
                                TARGET_C128_PREFILL_CAPTURE_FUNCTION,
                            ),
                            DeepseekV4TargetC128Lifecycle::Continuation => (
                                "continuation",
                                TARGET_C128_CONTINUATION_PREPARE_FUNCTION,
                                TARGET_C128_CONTINUATION_CAPTURE_FUNCTION,
                            ),
                        };
                        let graph = self.capture_compressed_attention_graph(
                            "C128",
                            lifecycle_label,
                            layer_id,
                            lane,
                            rows,
                            prepare_function,
                            capture_function,
                            prepare,
                            &buffers,
                            &kwargs,
                        )?;
                        self.c128_graphs.push(DeepseekV4TargetC128Graph {
                            layer_id,
                            lane,
                            rows,
                            lifecycle,
                            graph,
                        });
                    }
                }
            }
        }
        Ok(())
    }

    fn c128_graphs_prepared(&self, layer_ids: &[usize]) -> bool {
        let buckets = target_graph_row_buckets(self.plan.max_graph_rows);
        self.c128_graphs.len() == layer_ids.len() * self.plan.execution_lanes * 2 * buckets.len()
            && self
                .c128_graphs
                .iter()
                .zip(layer_ids.iter().copied().flat_map(|layer_id| {
                    (0..self.plan.execution_lanes).flat_map({
                        let buckets = buckets.clone();
                        move |lane| {
                            [
                                DeepseekV4TargetC128Lifecycle::Prefill,
                                DeepseekV4TargetC128Lifecycle::Continuation,
                            ]
                            .into_iter()
                            .flat_map({
                                let buckets = buckets.clone();
                                move |lifecycle| {
                                    buckets
                                        .clone()
                                        .into_iter()
                                        .map(move |rows| (layer_id, lane, lifecycle, rows))
                                }
                            })
                        }
                    })
                }))
                .all(|(graph, expected)| {
                    (graph.layer_id, graph.lane, graph.lifecycle, graph.rows) == expected
                        && !graph.graph.as_ptr().is_null()
                })
    }

    fn c128_attention_python_buffers(
        &self,
        lane: usize,
        layer_id: usize,
        weights: DeepseekV4TargetC128Weights,
    ) -> Result<Vec<PythonDeviceBufferArg<'static>>> {
        use ds4rt_core::DeepseekV4KvRegionKind;

        let state = self.c128_compressor_layer_state(lane, layer_id)?;
        Ok(vec![
            target_python_device_buffer(
                "workspace",
                self.lane_region(lane, self.plan.workspace, "C128 workspace")?,
            ),
            target_python_device_buffer(
                "hidden_states",
                self.lane_region(
                    lane,
                    self.plan.hidden_work,
                    "C128 normalized attention input",
                )?,
            ),
            target_python_device_buffer(
                "normalized_output",
                self.lane_region(lane, self.plan.hidden_output, "C128 normalized FFN output")?,
            ),
            target_python_device_buffer(
                "positions",
                self.lane_region(lane, self.plan.positions, "C128 positions")?,
            ),
            target_python_device_buffer(
                "main_slots",
                self.lane_region(lane, self.plan.main_slots, "C128 main slots")?,
            ),
            target_python_device_buffer(
                "cos_sin_cache",
                self.lane_region(lane, self.plan.cos_sin, "C128 cos/sin")?,
            ),
            target_python_device_buffer("main_kv_cache", self.physical_main_layer(layer_id)?),
            target_python_device_buffer(
                "swa_indices",
                self.lane_region(lane, self.plan.selected_indices, "C128 SWA indices")?,
            ),
            target_python_device_buffer(
                "swa_lengths",
                self.lane_region(lane, self.plan.selected_lengths, "C128 SWA lengths")?,
            ),
            target_python_device_buffer(
                "active_groups",
                self.lane_region(lane, self.plan.active_groups, "C128 active groups")?,
            ),
            target_python_device_buffer(
                "group_source_starts",
                self.lane_region(
                    lane,
                    self.plan.group_source_starts,
                    "C128 group source starts",
                )?,
            ),
            target_python_device_buffer(
                "group_sequence_slots",
                self.lane_region(
                    lane,
                    self.plan.group_sequence_slots,
                    "C128 group sequence slots",
                )?,
            ),
            target_python_device_buffer(
                "group_rope_positions",
                self.lane_region(
                    lane,
                    self.plan.group_rope_positions,
                    "C128 group RoPE positions",
                )?,
            ),
            target_python_device_buffer(
                "compressed_slots",
                self.lane_region(lane, self.plan.compressed_slots, "C128 compressed slots")?,
            ),
            target_python_device_buffer(
                "active_sequences",
                self.lane_region(lane, self.plan.active_sequences, "C128 active sequences")?,
            ),
            target_python_device_buffer(
                "sequence_offsets",
                self.lane_region(lane, self.plan.sequence_offsets, "C128 sequence offsets")?,
            ),
            target_python_device_buffer(
                "sequence_start_positions",
                self.lane_region(
                    lane,
                    self.plan.sequence_start_positions,
                    "C128 sequence start positions",
                )?,
            ),
            target_python_device_buffer(
                "state_sequence_ids",
                self.lane_region(
                    lane,
                    self.plan.state_sequence_ids,
                    "C128 state sequence IDs",
                )?,
            ),
            target_python_device_buffer(
                "compressed_main_cache",
                self.physical_layer_region(layer_id, DeepseekV4KvRegionKind::Compressed)?,
            ),
            target_python_device_buffer("main_kv_state", state.main_kv),
            target_python_device_buffer("main_score_state", state.main_score),
            target_python_device_buffer(
                "indexed_indices",
                self.lane_region(lane, self.plan.c128_indexed_indices, "C128 indexed indices")?,
            ),
            target_python_device_buffer(
                "indexed_lengths",
                self.lane_region(lane, self.plan.c128_indexed_lengths, "C128 indexed lengths")?,
            ),
            target_python_device_buffer(
                "residual",
                self.lane_region(lane, self.plan.residual_stage_ping, "C128 residual input")?,
            ),
            target_python_device_buffer(
                "prev_post",
                self.lane_region(lane, self.plan.post_stage_ping, "C128 post input")?,
            ),
            target_python_device_buffer(
                "prev_comb",
                self.lane_region(lane, self.plan.comb_stage_ping, "C128 combination input")?,
            ),
            target_python_device_buffer(
                "residual_out",
                self.lane_region(lane, self.plan.residual_stage_pong, "C128 residual output")?,
            ),
            target_python_device_buffer(
                "post_out",
                self.lane_region(lane, self.plan.post_stage_pong, "C128 post output")?,
            ),
            target_python_device_buffer(
                "comb_out",
                self.lane_region(lane, self.plan.comb_stage_pong, "C128 combination output")?,
            ),
            target_python_device_buffer("wq_a_weight", weights.common.wq_a_weight),
            target_python_device_buffer("wq_a_scale", weights.common.wq_a_scale),
            target_python_device_buffer("wq_b_weight", weights.common.wq_b_weight),
            target_python_device_buffer("wq_b_scale", weights.common.wq_b_scale),
            target_python_device_buffer("wkv_weight", weights.common.wkv_weight),
            target_python_device_buffer("wkv_scale", weights.common.wkv_scale),
            target_python_device_buffer("q_norm_weight", weights.common.q_norm_weight),
            target_python_device_buffer("kv_norm_weight", weights.common.kv_norm_weight),
            target_python_device_buffer("compressor_main_wkv", weights.main_wkv),
            target_python_device_buffer("compressor_main_wgate", weights.main_wgate),
            target_python_device_buffer("compressor_main_ape", weights.main_ape),
            target_python_device_buffer("compressor_main_norm", weights.main_norm),
            target_python_device_buffer("wo_a_weight", weights.common.wo_a_weight),
            target_python_device_buffer("wo_a_scale", weights.common.wo_a_scale),
            target_python_device_buffer("wo_b_weight", weights.common.wo_b_weight),
            target_python_device_buffer("wo_b_scale", weights.common.wo_b_scale),
            target_python_device_buffer("attn_sink", weights.common.attn_sink),
            target_python_device_buffer("hc_fn", weights.common.hc_fn),
            target_python_device_buffer("hc_scale", weights.common.hc_scale),
            target_python_device_buffer("hc_base", weights.common.hc_base),
            target_python_device_buffer("norm_weight", weights.common.norm_weight),
        ])
    }

    fn initialize_c128_capture_metadata(
        &mut self,
        lane: usize,
        rows: usize,
        lifecycle: DeepseekV4TargetC128Lifecycle,
        rope_inv_freq: &[f32; TARGET_ROPE_WIDTH / 2],
    ) -> Result<()> {
        let logical_row_start = match lifecycle {
            DeepseekV4TargetC128Lifecycle::Prefill => 0,
            DeepseekV4TargetC128Lifecycle::Continuation => 128,
        };
        let physical_slots = (0..logical_row_start + rows)
            .map(|slot| u32::try_from(slot).context("C128 capture slot exceeds u32"))
            .collect::<Result<Vec<_>>>()?;
        let metadata = target_c128_replay_metadata(
            logical_row_start,
            rows,
            &physical_slots,
            rope_inv_freq,
            self.plan.max_graph_rows,
            self.plan.max_sequence_pages,
        )?;
        self.upload_c128_metadata(lane, &metadata)
    }

    pub(in crate::commands::real_full) fn prepare_post_pre_graphs(
        &mut self,
        catalog: &TensorCatalog,
        rms_eps: f32,
        hc_eps: f32,
        sinkhorn_iters: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            rms_eps.is_finite()
                && rms_eps > 0.0
                && hc_eps.is_finite()
                && hc_eps > 0.0
                && sinkhorn_iters > 0,
            "native target post/pre parameters must be finite and positive"
        );
        let attention = DeepseekV4AttentionPlan::from_model_facts(&catalog.facts)
            .context("planning native target post/pre graph boundaries")?;
        let completed_layers = target_connected_post_pre_layer_ids(&attention);
        anyhow::ensure!(
            !completed_layers.is_empty(),
            "native target graph set has no connected post/pre boundaries"
        );
        if !self.post_pre_graphs.is_empty() {
            anyhow::ensure!(
                self.post_pre_graphs_prepared(&completed_layers),
                "native target post/pre graph set is partially initialized"
            );
            return Ok(());
        }
        let buckets = target_graph_row_buckets(self.plan.max_graph_rows);
        let first_regular_layer = completed_layers
            .iter()
            .copied()
            .find(|layer_id| !catalog.facts.dspark_target_layer_ids.contains(layer_id));
        let first_aux_layer = completed_layers
            .iter()
            .copied()
            .find(|layer_id| catalog.facts.dspark_target_layer_ids.contains(layer_id));
        for completed_layer_id in completed_layers.iter().copied() {
            let weights = target_post_pre_weights(self.plan.hidden, completed_layer_id + 1)?;
            let captures_aux_hidden = catalog
                .facts
                .dspark_target_layer_ids
                .contains(&completed_layer_id);
            for lane in 0..self.plan.execution_lanes {
                let stream = self.streams[lane].as_ptr();
                for rows in buckets.iter().copied() {
                    let buffers = self.post_pre_python_buffers(lane, weights)?;
                    let kwargs = [
                        ("variant", PythonKernelArg::Str(self.plan.variant)),
                        ("rows", PythonKernelArg::Usize(rows)),
                        ("max_rows", PythonKernelArg::Usize(self.plan.max_graph_rows)),
                        ("rms_eps", PythonKernelArg::F64(rms_eps.into())),
                        ("hc_eps", PythonKernelArg::F64(hc_eps.into())),
                        ("sinkhorn_iters", PythonKernelArg::Usize(sinkhorn_iters)),
                        ("norm_eps", PythonKernelArg::F64(rms_eps.into())),
                        (
                            "capture_aux_hidden",
                            PythonKernelArg::Bool(captures_aux_hidden),
                        ),
                    ];
                    if Some(completed_layer_id)
                        == if captures_aux_hidden {
                            first_aux_layer
                        } else {
                            first_regular_layer
                        }
                    {
                        launch_python_graph_capture(PythonGraphCaptureLaunch {
                            module: TARGET_MHC_CAPTURE_MODULE,
                            function: TARGET_MHC_POST_PRE_PREPARE_FUNCTION,
                            cuda_stream: stream,
                            buffers: &buffers,
                            kwargs: &kwargs,
                        })
                        .with_context(|| {
                            format!(
                                "preparing native target layer {completed_layer_id} post/pre lane {lane} bucket {rows}"
                            )
                        })?;
                    }
                    self.streams[lane].synchronize().with_context(|| {
                        format!(
                            "synchronizing native target layer {completed_layer_id} post/pre lane {lane} bucket {rows}"
                        )
                    })?;
                    unsafe {
                        self.library.cuda_graph_begin_capture(stream).with_context(|| {
                            format!(
                                "beginning native target layer {completed_layer_id} post/pre lane {lane} bucket {rows} capture"
                            )
                        })?;
                    }
                    let captured = launch_python_graph_capture(PythonGraphCaptureLaunch {
                        module: TARGET_MHC_CAPTURE_MODULE,
                        function: TARGET_MHC_POST_PRE_CAPTURE_FUNCTION,
                        cuda_stream: stream,
                        buffers: &buffers,
                        kwargs: &kwargs,
                    });
                    if let Err(error) = captured {
                        if let Ok(capture) = unsafe { self.library.cuda_graph_end_capture(stream) }
                        {
                            let _ = unsafe { self.library.cuda_graph_exec_destroy(capture) };
                        }
                        return Err(error).with_context(|| {
                            format!(
                                "capturing native target layer {completed_layer_id} post/pre lane {lane} bucket {rows}"
                            )
                        });
                    }
                    let capture = unsafe {
                        self.library
                            .cuda_graph_end_capture_retained(stream)
                            .with_context(|| {
                                format!(
                                    "ending native target layer {completed_layer_id} post/pre lane {lane} bucket {rows} capture"
                                )
                            })?
                    };
                    let graph = CoordinatorCudaCapturedGraph::new(self.library, capture)?;
                    graph.validate_before_launch()?;
                    anyhow::ensure!(
                        graph.memcpy_node_count == 0,
                        "native target layer {completed_layer_id} post/pre lane {lane} bucket {rows} unexpectedly captured memcpy nodes"
                    );
                    self.post_pre_graphs.push(DeepseekV4TargetPostPreGraph {
                        completed_layer_id,
                        lane,
                        rows,
                        captures_aux_hidden,
                        graph,
                    });
                }
            }
        }
        Ok(())
    }

    fn post_pre_graphs_prepared(&self, completed_layers: &[usize]) -> bool {
        let buckets = target_graph_row_buckets(self.plan.max_graph_rows);
        self.post_pre_graphs.len()
            == completed_layers.len() * self.plan.execution_lanes * buckets.len()
            && self
                .post_pre_graphs
                .iter()
                .zip(
                    completed_layers
                        .iter()
                        .copied()
                        .flat_map(|completed_layer_id| {
                            (0..self.plan.execution_lanes).flat_map({
                                let buckets = buckets.clone();
                                move |lane| {
                                    buckets
                                        .clone()
                                        .into_iter()
                                        .map(move |rows| (completed_layer_id, lane, rows))
                                }
                            })
                        }),
                )
                .all(|(graph, expected)| {
                    (graph.completed_layer_id, graph.lane, graph.rows) == expected
                        && !graph.graph.as_ptr().is_null()
                })
    }

    fn post_pre_python_buffers(
        &self,
        lane: usize,
        weights: DeepseekV4TargetPostPreWeights,
    ) -> Result<[PythonDeviceBufferArg<'static>; 14]> {
        Ok([
            target_python_device_buffer(
                "scratch",
                self.lane_region(lane, self.plan.entry_mhc_scratch, "post/pre mHC scratch")?,
            ),
            target_python_device_buffer(
                "delta",
                self.lane_region(lane, self.plan.hidden_input, "post/pre TP4 delta")?,
            ),
            target_python_device_buffer(
                "residual",
                self.lane_region(
                    lane,
                    self.plan.residual_stage_pong,
                    "post/pre residual input",
                )?,
            ),
            target_python_device_buffer(
                "prev_post",
                self.lane_region(lane, self.plan.post_stage_pong, "post/pre post input")?,
            ),
            target_python_device_buffer(
                "prev_comb",
                self.lane_region(
                    lane,
                    self.plan.comb_stage_pong,
                    "post/pre combination input",
                )?,
            ),
            target_python_device_buffer(
                "normalized_output",
                self.lane_region(
                    lane,
                    self.plan.hidden_work,
                    "post/pre normalized attention output",
                )?,
            ),
            target_python_device_buffer(
                "residual_out",
                self.lane_region(
                    lane,
                    self.plan.residual_stage_ping,
                    "post/pre residual output",
                )?,
            ),
            target_python_device_buffer(
                "post_out",
                self.lane_region(lane, self.plan.post_stage_ping, "post/pre post output")?,
            ),
            target_python_device_buffer(
                "comb_out",
                self.lane_region(
                    lane,
                    self.plan.comb_stage_ping,
                    "post/pre combination output",
                )?,
            ),
            target_python_device_buffer(
                "aux_hidden_output",
                self.lane_region(
                    lane,
                    self.plan.hidden_output,
                    "post/pre raw mHC auxiliary hidden output",
                )?,
            ),
            target_python_device_buffer("hc_fn", weights.hc_fn),
            target_python_device_buffer("hc_scale", weights.hc_scale),
            target_python_device_buffer("hc_base", weights.hc_base),
            target_python_device_buffer("norm_weight", weights.norm_weight),
        ])
    }

    pub(in crate::commands::real_full) fn prepare_terminal_graphs(
        &mut self,
        rms_eps: f32,
        hc_eps: f32,
    ) -> Result<()> {
        anyhow::ensure!(
            rms_eps.is_finite() && rms_eps > 0.0 && hc_eps.is_finite() && hc_eps > 0.0,
            "native target terminal parameters must be finite and positive"
        );
        if !self.terminal_graphs.is_empty() {
            anyhow::ensure!(
                self.terminal_graphs_prepared(),
                "native target terminal graph set is partially initialized"
            );
            return Ok(());
        }
        let weights = target_terminal_weights(self.plan.hidden)?;
        let buckets = target_graph_row_buckets(self.plan.max_graph_rows);
        for lane in 0..self.plan.execution_lanes {
            let stream = self.streams[lane].as_ptr();
            for rows in buckets.iter().copied() {
                let buffers = self.terminal_python_buffers(lane, weights)?;
                let kwargs = [
                    ("variant", PythonKernelArg::Str(self.plan.variant)),
                    ("rows", PythonKernelArg::Usize(rows)),
                    ("max_rows", PythonKernelArg::Usize(self.plan.max_graph_rows)),
                    ("rms_eps", PythonKernelArg::F64(rms_eps.into())),
                    ("hc_eps", PythonKernelArg::F64(hc_eps.into())),
                    ("norm_eps", PythonKernelArg::F64(rms_eps.into())),
                ];
                launch_python_graph_capture(PythonGraphCaptureLaunch {
                    module: TARGET_MHC_CAPTURE_MODULE,
                    function: TARGET_MHC_TERMINAL_PREPARE_FUNCTION,
                    cuda_stream: stream,
                    buffers: &buffers,
                    kwargs: &kwargs,
                })
                .with_context(|| {
                    format!("preparing native target terminal lane {lane} bucket {rows}")
                })?;
                self.streams[lane].synchronize().with_context(|| {
                    format!("synchronizing native target terminal lane {lane} bucket {rows}")
                })?;
                unsafe {
                    self.library
                        .cuda_graph_begin_capture(stream)
                        .with_context(|| {
                            format!(
                            "beginning native target terminal lane {lane} bucket {rows} capture"
                        )
                        })?;
                }
                let captured = launch_python_graph_capture(PythonGraphCaptureLaunch {
                    module: TARGET_MHC_CAPTURE_MODULE,
                    function: TARGET_MHC_TERMINAL_CAPTURE_FUNCTION,
                    cuda_stream: stream,
                    buffers: &buffers,
                    kwargs: &kwargs,
                });
                if let Err(error) = captured {
                    if let Ok(capture) = unsafe { self.library.cuda_graph_end_capture(stream) } {
                        let _ = unsafe { self.library.cuda_graph_exec_destroy(capture) };
                    }
                    return Err(error).with_context(|| {
                        format!("capturing native target terminal lane {lane} bucket {rows}")
                    });
                }
                let capture = unsafe {
                    self.library
                        .cuda_graph_end_capture_retained(stream)
                        .with_context(|| {
                            format!(
                                "ending native target terminal lane {lane} bucket {rows} capture"
                            )
                        })?
                };
                let graph = CoordinatorCudaCapturedGraph::new(self.library, capture)?;
                graph.validate_before_launch()?;
                anyhow::ensure!(
                    graph.memcpy_node_count == 0,
                    "native target terminal lane {lane} bucket {rows} unexpectedly captured memcpy nodes"
                );
                self.terminal_graphs
                    .push(DeepseekV4TargetTerminalGraph { lane, rows, graph });
            }
        }
        Ok(())
    }

    fn terminal_graphs_prepared(&self) -> bool {
        let buckets = target_graph_row_buckets(self.plan.max_graph_rows);
        self.terminal_graphs.len() == self.plan.execution_lanes * buckets.len()
            && self
                .terminal_graphs
                .iter()
                .zip((0..self.plan.execution_lanes).flat_map({
                    let buckets = buckets.clone();
                    move |lane| buckets.clone().into_iter().map(move |rows| (lane, rows))
                }))
                .all(|(graph, expected)| {
                    (graph.lane, graph.rows) == expected && !graph.graph.as_ptr().is_null()
                })
    }

    fn terminal_python_buffers(
        &self,
        lane: usize,
        weights: DeepseekV4TargetTerminalWeights,
    ) -> Result<[PythonDeviceBufferArg<'static>; 13]> {
        Ok([
            target_python_device_buffer(
                "scratch",
                self.lane_region(lane, self.plan.entry_mhc_scratch, "terminal mHC scratch")?,
            ),
            target_python_device_buffer(
                "delta",
                self.lane_region(lane, self.plan.hidden_input, "terminal TP4 delta")?,
            ),
            target_python_device_buffer(
                "residual",
                self.lane_region(
                    lane,
                    self.plan.residual_stage_pong,
                    "terminal residual input",
                )?,
            ),
            target_python_device_buffer(
                "prev_post",
                self.lane_region(lane, self.plan.post_stage_pong, "terminal post input")?,
            ),
            target_python_device_buffer(
                "prev_comb",
                self.lane_region(
                    lane,
                    self.plan.comb_stage_pong,
                    "terminal combination input",
                )?,
            ),
            target_python_device_buffer(
                "terminal_residual",
                self.lane_region(
                    lane,
                    self.plan.residual_stage_ping,
                    "terminal residual output",
                )?,
            ),
            target_python_device_buffer(
                "collapsed_output",
                self.lane_region(lane, self.plan.hidden_output, "terminal collapsed output")?,
            ),
            target_python_device_buffer(
                "normalized_output",
                self.lane_region(lane, self.plan.hidden_work, "terminal normalized output")?,
            ),
            // The terminal graph has consumed its FFN delta before producing
            // the dSpark tap, so the input region is safe scratch for the raw
            // four-lane mean without growing the target arena.
            target_python_device_buffer(
                "aux_hidden_output",
                self.lane_region(
                    lane,
                    self.plan.hidden_input,
                    "terminal raw mHC auxiliary hidden output",
                )?,
            ),
            target_python_device_buffer("hc_head_fn", weights.hc_head_fn),
            target_python_device_buffer("hc_head_scale", weights.hc_head_scale),
            target_python_device_buffer("hc_head_base", weights.hc_head_base),
            target_python_device_buffer("norm_weight", weights.norm_weight),
        ])
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::commands::real_full) fn run_attention(
        &mut self,
        lane: usize,
        layer_id: usize,
        logical_row_start: usize,
        rows: usize,
        hidden: Ds4rtDeviceBuffer,
        hidden_ready_event: Option<&CoordinatorCudaEvent>,
        physical_slots: &[u32],
        rope_theta: f32,
    ) -> Result<DeviceBf16Output> {
        let compress_ratio = self
            .plan
            .physical_plan
            .layer(layer_id)
            .with_context(|| format!("native target attention layer {layer_id} is missing"))?
            .compress_ratio;
        anyhow::ensure!(
            rope_theta.to_bits() == self.plan.rope_theta.to_bits(),
            "native target scheduler base RoPE theta {rope_theta} differs from planned {}",
            self.plan.rope_theta
        );
        let layer_rope_inv_freq = *target_attention_rope_inv_freq(
            compress_ratio,
            &self.plan.main_rope_inv_freq,
            &self.plan.compress_rope_inv_freq,
        )?;
        let new_segment = layer_id == 0;
        let allocate_segment = new_segment
            && self
                .hc_segment_row_offset(lane, logical_row_start, rows)
                .is_err();
        if allocate_segment {
            self.allocate_hc_segment(lane, logical_row_start, rows)?;
        } else {
            self.hc_segment_row_offset(lane, logical_row_start, rows)?;
        }
        let result = match compress_ratio {
            0 => self.run_c0_attention(
                lane,
                layer_id,
                logical_row_start,
                rows,
                hidden,
                hidden_ready_event,
                physical_slots,
                &layer_rope_inv_freq,
            ),
            4 => self.run_c4_attention(
                lane,
                layer_id,
                logical_row_start,
                rows,
                hidden,
                hidden_ready_event,
                physical_slots,
                &layer_rope_inv_freq,
            ),
            128 => self.run_c128_attention(
                lane,
                layer_id,
                logical_row_start,
                rows,
                hidden,
                hidden_ready_event,
                physical_slots,
                &layer_rope_inv_freq,
            ),
            ratio => Err(anyhow::anyhow!(
                "native target attention layer {layer_id} has unsupported compression ratio {ratio}"
            )),
        };
        if result.is_err() && allocate_segment {
            let _ = self.release_hc_segment(lane, logical_row_start, rows);
        }
        result
    }

    fn stage_attention_input(
        &mut self,
        lane: usize,
        layer_id: usize,
        logical_row_start: usize,
        rows: usize,
        hidden: Ds4rtDeviceBuffer,
        stream: *mut std::ffi::c_void,
    ) -> Result<()> {
        let hidden_bytes = rows
            .checked_mul(self.plan.hidden)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("native target attention replay hidden bytes overflow")?;
        let graph_hidden = if layer_id == 0 {
            self.lane_region(lane, self.plan.hidden_input, "target entry replay input")?
        } else {
            self.lane_region(lane, self.plan.hidden_work, "target attention replay input")?
        };
        unsafe {
            self.library
                .copy_d2d_async(graph_hidden, hidden, hidden_bytes, stream)
                .context("copying target attention replay hidden into fixed graph input")?;
        }
        if layer_id == 0 {
            let entry = self
                .entry_graphs
                .iter()
                .find(|graph| graph.lane == lane && graph.rows == rows)
                .context("native target entry replay has no exact captured graph")?;
            unsafe {
                self.library
                    .cuda_graph_launch(entry.graph.as_ptr(), stream)
                    .context("launching native target entry graph")?;
            }
        } else {
            self.copy_persistent_hc_to_stage(lane, logical_row_start, rows, false)?;
        }
        Ok(())
    }

    #[allow(dead_code, clippy::too_many_arguments)]
    pub(in crate::commands::real_full) fn run_c0_attention(
        &mut self,
        lane: usize,
        layer_id: usize,
        logical_row_start: usize,
        rows: usize,
        hidden: Ds4rtDeviceBuffer,
        hidden_ready_event: Option<&CoordinatorCudaEvent>,
        physical_slots: &[u32],
        rope_inv_freq: &[f32; TARGET_ROPE_WIDTH / 2],
    ) -> Result<DeviceBf16Output> {
        let timing_enabled = target_attention_timing_enabled();
        let total_started = timing_enabled.then(Instant::now);
        anyhow::ensure!(
            rows > 0
                && rows <= self.plan.max_graph_rows
                && TARGET_GRAPH_ROWS.contains(&rows)
                && logical_row_start
                    .checked_add(rows)
                    .is_some_and(|end| end <= self.plan.max_sequence_rows),
            "native target C0 replay rows {logical_row_start}..+{rows} are outside captured sequence geometry"
        );
        anyhow::ensure!(
            physical_slots.len() >= logical_row_start + rows,
            "native target C0 replay physical slots {} do not cover logical rows through {}",
            physical_slots.len(),
            logical_row_start + rows
        );
        let metadata_build_started = timing_enabled.then(Instant::now);
        let metadata_key = DeepseekV4TargetReplayMetadataKey {
            family: DeepseekV4TargetReplayMetadataFamily::C0,
            logical_row_start,
            rows,
        };
        let (metadata, metadata_cache_hit) = match self
            .replay_metadata_cache
            .get_mut(lane)
            .with_context(|| format!("native target C0 metadata lane {lane} is unavailable"))?
            .remove(&metadata_key)
        {
            Some(DeepseekV4TargetReplayMetadata::C0(metadata)) => (metadata, true),
            Some(_) => anyhow::bail!("native target C0 metadata cache family mismatch"),
            None => {
                let metadata = target_c0_replay_metadata(
                    logical_row_start,
                    rows,
                    physical_slots,
                    rope_inv_freq,
                )?;
                (self.prepare_c0_replay_metadata(&metadata)?, false)
            }
        };
        let metadata_build_ms = target_attention_elapsed_ms(metadata_build_started);
        let metadata_upload_started = timing_enabled.then(Instant::now);
        let upload_result = self.upload_prepared_replay_metadata_async(lane, &metadata);
        self.replay_metadata_cache[lane]
            .insert(metadata_key, DeepseekV4TargetReplayMetadata::C0(metadata));
        upload_result?;
        let metadata_upload_ms = target_attention_elapsed_ms(metadata_upload_started);
        let input_queue_started = timing_enabled.then(Instant::now);
        let stream = self.stream_ptr(lane)?;
        if let Some(event) = hidden_ready_event {
            event
                .wait_on_stream(stream)
                .context("waiting for native target C0 hidden input")?;
        }
        self.stage_attention_input(lane, layer_id, logical_row_start, rows, hidden, stream)?;
        let input_queue_ms = target_attention_elapsed_ms(input_queue_started);
        let attention = self
            .sliding_graphs
            .iter()
            .find(|graph| graph.layer_id == layer_id && graph.lane == lane && graph.rows == rows)
            .with_context(|| {
                format!(
                    "native target C0 layer {layer_id} lane {lane} rows {rows} has no captured graph"
                )
            })?;
        let graph_queue_started = timing_enabled.then(Instant::now);
        unsafe {
            self.library
                .cuda_graph_launch(attention.graph.as_ptr(), stream)
                .with_context(|| format!("launching native target C0 layer {layer_id} graph"))?;
        }
        let graph_queue_ms = target_attention_elapsed_ms(graph_queue_started);
        let output_queue_started = timing_enabled.then(Instant::now);
        self.copy_stage_hc_to_persistent(lane, logical_row_start, rows, true)?;
        let output_queue_ms = target_attention_elapsed_ms(output_queue_started);
        let output_copy_started = timing_enabled.then(Instant::now);
        let output = self.copy_output_async(
            lane,
            self.lane_region(lane, self.plan.hidden_output, "target C0 dispatch hidden")?,
            rows,
            "native target C0 normalized TP4 dispatch hidden",
        )?;
        let output_copy_ms = target_attention_elapsed_ms(output_copy_started);
        if timing_enabled {
            eprintln!(
                "real_full_native_target_attention_timing family=c0 layer_id={layer_id} lane={lane} logical_row_start={logical_row_start} rows={rows} metadata_cache_hit={metadata_cache_hit} metadata_build_ms={metadata_build_ms:.3} metadata_upload_ms={metadata_upload_ms:.3} input_queue_ms={input_queue_ms:.3} graph_queue_ms={graph_queue_ms:.3} output_queue_ms={output_queue_ms:.3} synchronize_ms=0.000 output_copy_ms={output_copy_ms:.3} total_ms={:.3}",
                target_attention_elapsed_ms(total_started),
            );
        }
        Ok(output)
    }

    #[allow(dead_code)]
    pub(in crate::commands::real_full) fn run_post_pre(
        &mut self,
        lane: usize,
        completed_layer_id: usize,
        logical_row_start: usize,
        rows: usize,
        ffn_delta: Ds4rtDeviceBuffer,
    ) -> Result<(DeviceBf16Output, Option<DeviceBf16Output>)> {
        anyhow::ensure!(
            rows > 0
                && TARGET_GRAPH_ROWS.contains(&rows)
                && logical_row_start
                    .checked_add(rows)
                    .is_some_and(|end| end <= self.plan.max_sequence_rows),
            "native target post/pre replay rows {logical_row_start}..+{rows} are outside captured sequence geometry"
        );
        let stream = self.stream_ptr(lane)?;
        let delta_bytes = rows
            .checked_mul(self.plan.hidden)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("native target post/pre delta bytes overflow")?;
        unsafe {
            self.library
                .copy_d2d_async(
                    self.lane_region(lane, self.plan.hidden_input, "target post/pre delta")?,
                    ffn_delta,
                    delta_bytes,
                    stream,
                )
                .context("copying completed TP4 delta into target post/pre graph")?;
        }
        self.copy_persistent_hc_to_stage(lane, logical_row_start, rows, true)?;
        let graph = self
            .post_pre_graphs
            .iter()
            .find(|graph| {
                graph.completed_layer_id == completed_layer_id
                    && graph.lane == lane
                    && graph.rows == rows
            })
            .with_context(|| {
                format!(
                    "native target layer {completed_layer_id} post/pre lane {lane} rows {rows} has no captured graph"
                )
            })?;
        let captures_aux_hidden = graph.captures_aux_hidden;
        unsafe {
            self.library
                .cuda_graph_launch(graph.graph.as_ptr(), stream)
                .with_context(|| {
                    format!("launching native target layer {completed_layer_id} post/pre graph")
                })?;
        }
        self.copy_stage_hc_to_persistent(lane, logical_row_start, rows, false)?;
        let normalized = self.copy_output_async(
            lane,
            self.lane_region(
                lane,
                self.plan.hidden_work,
                "target next-attention normalized hidden",
            )?,
            rows,
            "native target post/pre normalized attention hidden",
        )?;
        let aux_hidden = if captures_aux_hidden {
            Some(self.copy_output_async(
                lane,
                self.lane_region(
                    lane,
                    self.plan.hidden_output,
                    "target post/pre raw mHC auxiliary hidden",
                )?,
                rows,
                "native target post/pre raw mHC auxiliary hidden",
            )?)
        } else {
            None
        };
        Ok((normalized, aux_hidden))
    }

    pub(in crate::commands::real_full) fn run_terminal(
        &mut self,
        lane: usize,
        logical_row_start: usize,
        rows: usize,
        ffn_delta: Ds4rtDeviceBuffer,
    ) -> Result<(DeviceBf16Output, DeviceBf16Output)> {
        anyhow::ensure!(
            rows > 0
                && TARGET_GRAPH_ROWS.contains(&rows)
                && logical_row_start
                    .checked_add(rows)
                    .is_some_and(|end| end <= self.plan.max_sequence_rows),
            "native target terminal replay rows {logical_row_start}..+{rows} are outside captured sequence geometry"
        );
        let stream = self.stream_ptr(lane)?;
        let delta_bytes = rows
            .checked_mul(self.plan.hidden)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("native target terminal delta bytes overflow")?;
        unsafe {
            self.library
                .copy_d2d_async(
                    self.lane_region(lane, self.plan.hidden_input, "target terminal delta")?,
                    ffn_delta,
                    delta_bytes,
                    stream,
                )
                .context("copying completed TP4 delta into target terminal graph")?;
        }
        self.copy_persistent_hc_to_stage(lane, logical_row_start, rows, true)?;
        let graph = self
            .terminal_graphs
            .iter()
            .find(|graph| graph.lane == lane && graph.rows == rows)
            .with_context(|| {
                format!("native target terminal lane {lane} rows {rows} has no captured graph")
            })?;
        unsafe {
            self.library
                .cuda_graph_launch(graph.graph.as_ptr(), stream)
                .context("launching native target terminal graph")?;
        }
        self.streams[lane]
            .synchronize()
            .context("synchronizing native target terminal replay")?;
        // Keep the raw collapsed state as the scheduler handoff so the current
        // shared sampler remains its single final-norm owner. The graph also
        // writes the fused normalized sibling into hidden_work for the later
        // terminal sampling-fusion milestone.
        let output = device_bf16_output_from_device_template_buffer(
            self.lane_region(
                lane,
                self.plan.hidden_output,
                "target terminal collapsed hidden",
            )?,
            rows,
            self.plan.hidden,
            "native target terminal collapsed hidden",
        );
        let aux_hidden = device_bf16_output_from_device_template_buffer(
            self.lane_region(
                lane,
                self.plan.hidden_input,
                "target terminal raw mHC auxiliary hidden",
            )?,
            rows,
            self.plan.hidden,
            "native target terminal raw mHC auxiliary hidden",
        );
        let release = self.release_hc_segment(lane, logical_row_start, rows);
        match (output, aux_hidden, release) {
            (Ok(output), Ok(aux_hidden), Ok(())) => Ok((output, aux_hidden)),
            (Err(error), _, Ok(())) | (_, Err(error), Ok(())) => Err(error),
            (Ok(_), Ok(_), Err(error)) => Err(error),
            (Err(error), _, Err(release_error)) | (_, Err(error), Err(release_error)) => Err(error
                .context(format!(
                    "target terminal also failed to release its mHC segment: {release_error:#}"
                ))),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::commands::real_full) fn run_c4_attention(
        &mut self,
        lane: usize,
        layer_id: usize,
        logical_row_start: usize,
        rows: usize,
        hidden: Ds4rtDeviceBuffer,
        hidden_ready_event: Option<&CoordinatorCudaEvent>,
        physical_slots: &[u32],
        rope_inv_freq: &[f32; TARGET_ROPE_WIDTH / 2],
    ) -> Result<DeviceBf16Output> {
        let timing_enabled = target_attention_timing_enabled();
        let total_started = timing_enabled.then(Instant::now);
        anyhow::ensure!(
            rows > 0
                && rows <= self.plan.max_graph_rows
                && TARGET_GRAPH_ROWS.contains(&rows)
                && logical_row_start
                    .checked_add(rows)
                    .is_some_and(|end| end <= self.plan.max_sequence_rows),
            "native target C4 replay rows {logical_row_start}..+{rows} are outside captured sequence geometry"
        );
        let layer = self
            .plan
            .physical_plan
            .layer(layer_id)
            .with_context(|| format!("native target C4 layer {layer_id} is missing"))?;
        anyhow::ensure!(
            layer.compress_ratio == 4,
            "native target layer {layer_id} is not a C4 layer"
        );
        let metadata_build_started = timing_enabled.then(Instant::now);
        let metadata_key = DeepseekV4TargetReplayMetadataKey {
            family: DeepseekV4TargetReplayMetadataFamily::C4,
            logical_row_start,
            rows,
        };
        let (metadata, metadata_cache_hit) = match self
            .replay_metadata_cache
            .get_mut(lane)
            .with_context(|| format!("native target C4 metadata lane {lane} is unavailable"))?
            .remove(&metadata_key)
        {
            Some(DeepseekV4TargetReplayMetadata::C4(metadata)) => (metadata, true),
            Some(_) => anyhow::bail!("native target C4 metadata cache family mismatch"),
            None => {
                let metadata = target_c4_replay_metadata(
                    logical_row_start,
                    rows,
                    physical_slots,
                    rope_inv_freq,
                    self.plan.max_graph_rows,
                    self.plan.max_sequence_pages,
                )?;
                (self.prepare_c4_replay_metadata(&metadata)?, false)
            }
        };
        let metadata_build_ms = target_attention_elapsed_ms(metadata_build_started);
        let metadata_upload_started = timing_enabled.then(Instant::now);
        let upload_result = self.upload_prepared_replay_metadata_async(lane, &metadata);
        self.replay_metadata_cache[lane]
            .insert(metadata_key, DeepseekV4TargetReplayMetadata::C4(metadata));
        upload_result?;
        let metadata_upload_ms = target_attention_elapsed_ms(metadata_upload_started);
        let input_queue_started = timing_enabled.then(Instant::now);
        let stream = self.stream_ptr(lane)?;
        if let Some(event) = hidden_ready_event {
            event
                .wait_on_stream(stream)
                .context("waiting for native target C4 hidden input")?;
        }
        self.stage_attention_input(lane, layer_id, logical_row_start, rows, hidden, stream)?;
        let input_queue_ms = target_attention_elapsed_ms(input_queue_started);
        let lifecycle = if logical_row_start == 0 {
            DeepseekV4TargetC4Lifecycle::Prefill
        } else {
            DeepseekV4TargetC4Lifecycle::Continuation
        };
        let graph = self
            .c4_graphs
            .iter()
            .find(|graph| {
                graph.layer_id == layer_id
                    && graph.lane == lane
                    && graph.rows == rows
                    && graph.lifecycle == lifecycle
            })
            .with_context(|| {
                format!(
                    "native target C4 {lifecycle:?} layer {layer_id} lane {lane} rows {rows} has no captured graph"
                )
            })?;
        let graph_queue_started = timing_enabled.then(Instant::now);
        unsafe {
            self.library
                .cuda_graph_launch(graph.graph.as_ptr(), stream)
                .with_context(|| {
                    format!("launching native target C4 {lifecycle:?} layer {layer_id} graph")
                })?;
        }
        let graph_queue_ms = target_attention_elapsed_ms(graph_queue_started);
        let output_queue_started = timing_enabled.then(Instant::now);
        self.copy_stage_hc_to_persistent(lane, logical_row_start, rows, true)?;
        let output_queue_ms = target_attention_elapsed_ms(output_queue_started);
        let output_copy_started = timing_enabled.then(Instant::now);
        let output = self.copy_output_async(
            lane,
            self.lane_region(lane, self.plan.hidden_output, "target C4 dispatch hidden")?,
            rows,
            "native target C4 normalized TP4 dispatch hidden",
        )?;
        let output_copy_ms = target_attention_elapsed_ms(output_copy_started);
        if timing_enabled {
            eprintln!(
                "real_full_native_target_attention_timing family=c4 layer_id={layer_id} lane={lane} logical_row_start={logical_row_start} rows={rows} lifecycle={lifecycle:?} metadata_cache_hit={metadata_cache_hit} metadata_build_ms={metadata_build_ms:.3} metadata_upload_ms={metadata_upload_ms:.3} input_queue_ms={input_queue_ms:.3} graph_queue_ms={graph_queue_ms:.3} output_queue_ms={output_queue_ms:.3} synchronize_ms=0.000 output_copy_ms={output_copy_ms:.3} total_ms={:.3}",
                target_attention_elapsed_ms(total_started),
            );
        }
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::commands::real_full) fn run_c128_attention(
        &mut self,
        lane: usize,
        layer_id: usize,
        logical_row_start: usize,
        rows: usize,
        hidden: Ds4rtDeviceBuffer,
        hidden_ready_event: Option<&CoordinatorCudaEvent>,
        physical_slots: &[u32],
        rope_inv_freq: &[f32; TARGET_ROPE_WIDTH / 2],
    ) -> Result<DeviceBf16Output> {
        let timing_enabled = target_attention_timing_enabled();
        let total_started = timing_enabled.then(Instant::now);
        anyhow::ensure!(
            rows > 0
                && rows <= self.plan.max_graph_rows
                && TARGET_GRAPH_ROWS.contains(&rows)
                && logical_row_start
                    .checked_add(rows)
                    .is_some_and(|end| end <= self.plan.max_sequence_rows),
            "native target C128 replay rows {logical_row_start}..+{rows} are outside captured sequence geometry"
        );
        let layer = self
            .plan
            .physical_plan
            .layer(layer_id)
            .with_context(|| format!("native target C128 layer {layer_id} is missing"))?;
        anyhow::ensure!(
            layer.compress_ratio == 128,
            "native target layer {layer_id} is not a C128 layer"
        );
        let metadata_build_started = timing_enabled.then(Instant::now);
        let metadata_key = DeepseekV4TargetReplayMetadataKey {
            family: DeepseekV4TargetReplayMetadataFamily::C128,
            logical_row_start,
            rows,
        };
        let (metadata, metadata_cache_hit) = match self
            .replay_metadata_cache
            .get_mut(lane)
            .with_context(|| format!("native target C128 metadata lane {lane} is unavailable"))?
            .remove(&metadata_key)
        {
            Some(DeepseekV4TargetReplayMetadata::C128(metadata)) => (metadata, true),
            Some(_) => anyhow::bail!("native target C128 metadata cache family mismatch"),
            None => {
                let metadata = target_c128_replay_metadata(
                    logical_row_start,
                    rows,
                    physical_slots,
                    rope_inv_freq,
                    self.plan.max_graph_rows,
                    self.plan.max_sequence_pages,
                )?;
                (self.prepare_c128_replay_metadata(&metadata)?, false)
            }
        };
        let metadata_build_ms = target_attention_elapsed_ms(metadata_build_started);
        let metadata_upload_started = timing_enabled.then(Instant::now);
        let upload_result = self.upload_prepared_replay_metadata_async(lane, &metadata);
        self.replay_metadata_cache[lane]
            .insert(metadata_key, DeepseekV4TargetReplayMetadata::C128(metadata));
        upload_result?;
        let metadata_upload_ms = target_attention_elapsed_ms(metadata_upload_started);
        let input_queue_started = timing_enabled.then(Instant::now);
        let stream = self.stream_ptr(lane)?;
        if let Some(event) = hidden_ready_event {
            event
                .wait_on_stream(stream)
                .context("waiting for native target C128 hidden input")?;
        }
        self.stage_attention_input(lane, layer_id, logical_row_start, rows, hidden, stream)?;
        let input_queue_ms = target_attention_elapsed_ms(input_queue_started);
        let lifecycle = if logical_row_start == 0 {
            DeepseekV4TargetC128Lifecycle::Prefill
        } else {
            DeepseekV4TargetC128Lifecycle::Continuation
        };
        let graph = self
            .c128_graphs
            .iter()
            .find(|graph| {
                graph.layer_id == layer_id
                    && graph.lane == lane
                    && graph.rows == rows
                    && graph.lifecycle == lifecycle
            })
            .with_context(|| {
                format!(
                    "native target C128 {lifecycle:?} layer {layer_id} lane {lane} rows {rows} has no captured graph"
                )
            })?;
        let graph_queue_started = timing_enabled.then(Instant::now);
        unsafe {
            self.library
                .cuda_graph_launch(graph.graph.as_ptr(), stream)
                .with_context(|| {
                    format!("launching native target C128 {lifecycle:?} layer {layer_id} graph")
                })?;
        }
        let graph_queue_ms = target_attention_elapsed_ms(graph_queue_started);
        let output_queue_started = timing_enabled.then(Instant::now);
        self.copy_stage_hc_to_persistent(lane, logical_row_start, rows, true)?;
        let output_queue_ms = target_attention_elapsed_ms(output_queue_started);
        let output_copy_started = timing_enabled.then(Instant::now);
        let output = self.copy_output_async(
            lane,
            self.lane_region(lane, self.plan.hidden_output, "target C128 dispatch hidden")?,
            rows,
            "native target C128 normalized TP4 dispatch hidden",
        )?;
        let output_copy_ms = target_attention_elapsed_ms(output_copy_started);
        if timing_enabled {
            eprintln!(
                "real_full_native_target_attention_timing family=c128 layer_id={layer_id} lane={lane} logical_row_start={logical_row_start} rows={rows} lifecycle={lifecycle:?} metadata_cache_hit={metadata_cache_hit} metadata_build_ms={metadata_build_ms:.3} metadata_upload_ms={metadata_upload_ms:.3} input_queue_ms={input_queue_ms:.3} graph_queue_ms={graph_queue_ms:.3} output_queue_ms={output_queue_ms:.3} synchronize_ms=0.000 output_copy_ms={output_copy_ms:.3} total_ms={:.3}",
                target_attention_elapsed_ms(total_started),
            );
        }
        Ok(output)
    }

    fn upload_c0_replay_metadata(
        &mut self,
        lane: usize,
        logical_row_start: usize,
        rows: usize,
        physical_slots: &[u32],
        rope_inv_freq: &[f32; TARGET_ROPE_WIDTH / 2],
    ) -> Result<()> {
        let metadata =
            target_c0_replay_metadata(logical_row_start, rows, physical_slots, rope_inv_freq)?;
        self.upload_c0_replay_metadata_value(lane, &metadata)
    }

    fn upload_c0_replay_metadata_value(
        &mut self,
        lane: usize,
        metadata: &DeepseekV4TargetC0ReplayMetadata,
    ) -> Result<()> {
        let entries = self.c0_replay_metadata_entries(metadata);
        self.upload_replay_metadata_async(lane, entries)
    }

    fn prepare_c0_replay_metadata(
        &self,
        metadata: &DeepseekV4TargetC0ReplayMetadata,
    ) -> Result<DeepseekV4TargetPreparedReplayMetadata> {
        let entries = self.c0_replay_metadata_entries(metadata);
        self.prepare_replay_metadata(entries)
    }

    fn c0_replay_metadata_entries<'a>(
        &self,
        metadata: &'a DeepseekV4TargetC0ReplayMetadata,
    ) -> Vec<(DeepseekV4TargetDeviceRegion, &'a [u8], &'static str)> {
        vec![
            (
                self.plan.positions,
                u32_host_bytes(&metadata.positions),
                "target replay positions",
            ),
            (
                self.plan.main_slots,
                u32_host_bytes(&metadata.main_slots),
                "target replay main slots",
            ),
            (
                self.plan.cos_sin,
                f32_host_bytes(&metadata.cos_sin),
                "target replay cos/sin",
            ),
            (
                self.plan.selected_indices,
                i32_host_bytes(&metadata.selected_indices),
                "target replay selected indices",
            ),
            (
                self.plan.selected_lengths,
                i32_host_bytes(&metadata.selected_lengths),
                "target replay selected lengths",
            ),
        ]
    }

    fn upload_c4_metadata(
        &mut self,
        lane: usize,
        metadata: &DeepseekV4TargetC4ReplayMetadata,
    ) -> Result<()> {
        let entries = self.c4_replay_metadata_entries(metadata)?;
        self.upload_replay_metadata_async(lane, entries)
    }

    fn prepare_c4_replay_metadata(
        &self,
        metadata: &DeepseekV4TargetC4ReplayMetadata,
    ) -> Result<DeepseekV4TargetPreparedReplayMetadata> {
        let entries = self.c4_replay_metadata_entries(metadata)?;
        self.prepare_replay_metadata(entries)
    }

    fn c4_replay_metadata_entries<'a>(
        &self,
        metadata: &'a DeepseekV4TargetC4ReplayMetadata,
    ) -> Result<Vec<(DeepseekV4TargetDeviceRegion, &'a [u8], &'static str)>> {
        let active_groups = std::slice::from_ref(&metadata.active_groups);
        const ACTIVE_SEQUENCES: [u32; 1] = [1_u32];
        let mut entries = Vec::with_capacity(17);
        entries.extend([
            (
                self.plan.positions,
                u32_host_bytes(&metadata.positions),
                "target C4 positions",
            ),
            (
                self.plan.main_slots,
                u32_host_bytes(&metadata.main_slots),
                "target C4 main slots",
            ),
            (
                self.plan.group_source_starts,
                u32_host_bytes(&metadata.group_source_starts),
                "target C4 group source starts",
            ),
            (
                self.plan.group_sequence_slots,
                u32_host_bytes(&metadata.group_sequence_slots),
                "target C4 group sequence slots",
            ),
            (
                self.plan.group_rope_positions,
                u32_host_bytes(&metadata.group_rope_positions),
                "target C4 group RoPE positions",
            ),
            (
                self.plan.compressed_slots,
                u32_host_bytes(&metadata.compressed_slots),
                "target C4 compressed slots",
            ),
            (
                self.plan.sequence_offsets,
                u32_host_bytes(&metadata.sequence_offsets),
                "target C4 sequence offsets",
            ),
            (
                self.plan.sequence_start_positions,
                u32_host_bytes(&metadata.sequence_start_positions),
                "target C4 sequence start positions",
            ),
            (
                self.plan.state_sequence_ids,
                u32_host_bytes(&metadata.state_sequence_ids),
                "target C4 state sequence IDs",
            ),
        ]);
        entries.extend([
            (
                self.plan.selected_indices,
                i32_host_bytes(&metadata.swa_indices),
                "target C4 SWA indices",
            ),
            (
                self.plan.selected_lengths,
                i32_host_bytes(&metadata.swa_lengths),
                "target C4 SWA lengths",
            ),
            (
                self.plan.index_cache_seqlens,
                i32_host_bytes(&metadata.index_cache_seqlens),
                "target C4 index cache lengths",
            ),
            (
                self.plan.real_page_table,
                i32_host_bytes(&metadata.real_page_table),
                "target C4 shared real page table",
            ),
        ]);
        let active_cos_sin_values = metadata
            .positions
            .len()
            .checked_mul(TARGET_ROPE_WIDTH)
            .context("target C4 active cos/sin values overflow")?;
        entries.push((
            self.plan.cos_sin,
            f32_host_bytes(&metadata.cos_sin[..active_cos_sin_values]),
            "target C4 active cos/sin",
        ));
        if metadata
            .group_rope_positions
            .iter()
            .any(|position| *position as usize >= self.plan.max_graph_rows)
        {
            let carry_value_start = self
                .plan
                .max_graph_rows
                .checked_mul(TARGET_ROPE_WIDTH)
                .context("target C4 carry cos/sin offset overflow")?;
            let carry_value_end = carry_value_start
                .checked_add(TARGET_ROPE_WIDTH)
                .context("target C4 carry cos/sin extent overflow")?;
            entries.push((
                DeepseekV4TargetDeviceRegion {
                    offset: self
                        .plan
                        .cos_sin
                        .offset
                        .checked_add(carry_value_start * std::mem::size_of::<f32>())
                        .context("target C4 carry cos/sin region overflow")?,
                    bytes: TARGET_ROPE_WIDTH * std::mem::size_of::<f32>(),
                },
                f32_host_bytes(&metadata.cos_sin[carry_value_start..carry_value_end]),
                "target C4 carried cos/sin",
            ));
        }
        entries.extend([
            (
                self.plan.active_groups,
                u32_host_bytes(active_groups),
                "target C4 active groups",
            ),
            (
                self.plan.active_sequences,
                u32_host_bytes(&ACTIVE_SEQUENCES),
                "target C4 active sequences",
            ),
        ]);
        Ok(entries)
    }

    fn upload_c128_metadata(
        &mut self,
        lane: usize,
        metadata: &DeepseekV4TargetC128ReplayMetadata,
    ) -> Result<()> {
        let entries = self.c128_replay_metadata_entries(metadata)?;
        self.upload_replay_metadata_async(lane, entries)
    }

    fn prepare_c128_replay_metadata(
        &self,
        metadata: &DeepseekV4TargetC128ReplayMetadata,
    ) -> Result<DeepseekV4TargetPreparedReplayMetadata> {
        let entries = self.c128_replay_metadata_entries(metadata)?;
        self.prepare_replay_metadata(entries)
    }

    fn c128_replay_metadata_entries<'a>(
        &self,
        metadata: &'a DeepseekV4TargetC128ReplayMetadata,
    ) -> Result<Vec<(DeepseekV4TargetDeviceRegion, &'a [u8], &'static str)>> {
        let active_groups = std::slice::from_ref(&metadata.active_groups);
        const ACTIVE_SEQUENCES: [u32; 1] = [1_u32];
        let mut entries = Vec::with_capacity(17);
        entries.extend([
            (
                self.plan.positions,
                u32_host_bytes(&metadata.positions),
                "target C128 positions",
            ),
            (
                self.plan.main_slots,
                u32_host_bytes(&metadata.main_slots),
                "target C128 main slots",
            ),
            (
                self.plan.group_source_starts,
                u32_host_bytes(&metadata.group_source_starts),
                "target C128 group source starts",
            ),
            (
                self.plan.group_sequence_slots,
                u32_host_bytes(&metadata.group_sequence_slots),
                "target C128 group sequence slots",
            ),
            (
                self.plan.group_rope_positions,
                u32_host_bytes(&metadata.group_rope_positions),
                "target C128 group RoPE positions",
            ),
            (
                self.plan.compressed_slots,
                u32_host_bytes(&metadata.compressed_slots),
                "target C128 compressed slots",
            ),
            (
                self.plan.sequence_offsets,
                u32_host_bytes(&metadata.sequence_offsets),
                "target C128 sequence offsets",
            ),
            (
                self.plan.sequence_start_positions,
                u32_host_bytes(&metadata.sequence_start_positions),
                "target C128 sequence start positions",
            ),
            (
                self.plan.state_sequence_ids,
                u32_host_bytes(&metadata.state_sequence_ids),
                "target C128 state sequence IDs",
            ),
        ]);
        entries.extend([
            (
                self.plan.selected_indices,
                i32_host_bytes(&metadata.swa_indices),
                "target C128 SWA indices",
            ),
            (
                self.plan.selected_lengths,
                i32_host_bytes(&metadata.swa_lengths),
                "target C128 SWA lengths",
            ),
            (
                self.plan.c128_indexed_lengths,
                i32_host_bytes(&metadata.indexed_lengths),
                "target C128 indexed lengths",
            ),
            (
                self.plan.c128_indexed_indices,
                i32_host_bytes(&metadata.indexed_indices),
                "target C128 shared indexed indices",
            ),
        ]);
        let active_cos_sin_values = metadata
            .positions
            .len()
            .checked_mul(TARGET_ROPE_WIDTH)
            .context("target C128 active cos/sin values overflow")?;
        entries.push((
            self.plan.cos_sin,
            f32_host_bytes(&metadata.cos_sin[..active_cos_sin_values]),
            "target C128 active cos/sin",
        ));
        if metadata
            .group_rope_positions
            .iter()
            .any(|position| *position as usize >= self.plan.max_graph_rows)
        {
            let carry_value_start = self
                .plan
                .max_graph_rows
                .checked_mul(TARGET_ROPE_WIDTH)
                .context("target C128 carry cos/sin offset overflow")?;
            let carry_value_end = carry_value_start
                .checked_add(TARGET_ROPE_WIDTH)
                .context("target C128 carry cos/sin extent overflow")?;
            entries.push((
                DeepseekV4TargetDeviceRegion {
                    offset: self
                        .plan
                        .cos_sin
                        .offset
                        .checked_add(carry_value_start * std::mem::size_of::<f32>())
                        .context("target C128 carry cos/sin region overflow")?,
                    bytes: TARGET_ROPE_WIDTH * std::mem::size_of::<f32>(),
                },
                f32_host_bytes(&metadata.cos_sin[carry_value_start..carry_value_end]),
                "target C128 carried cos/sin",
            ));
        }
        entries.extend([
            (
                self.plan.active_groups,
                u32_host_bytes(active_groups),
                "target C128 active groups",
            ),
            (
                self.plan.active_sequences,
                u32_host_bytes(&ACTIVE_SEQUENCES),
                "target C128 active sequences",
            ),
        ]);
        Ok(entries)
    }

    fn upload_replay_metadata_async(
        &mut self,
        lane: usize,
        entries: Vec<(DeepseekV4TargetDeviceRegion, &[u8], &'static str)>,
    ) -> Result<()> {
        let copy_event =
            Arc::clone(self.metadata_copy_events.get(lane).with_context(|| {
                format!("native target metadata event lane {lane} is unavailable")
            })?);
        let copy_in_flight = *self
            .metadata_copy_in_flight
            .get(lane)
            .with_context(|| format!("native target metadata state lane {lane} is unavailable"))?;
        if copy_in_flight {
            // The pinned staging image is mutable host memory. CUDA stream
            // ordering protects the destination arena but cannot stop the CPU
            // from overwriting this source while an earlier H2D DMA still
            // reads it. Fence only the metadata copies, not the attention
            // graph that follows them.
            copy_event
                .synchronize()
                .with_context(|| format!("waiting to reuse native target metadata lane {lane}"))?;
            self.metadata_copy_in_flight[lane] = false;
        }
        let total_bytes = entries.iter().try_fold(0_usize, |total, (_, bytes, _)| {
            total
                .checked_add(bytes.len())
                .context("native target replay metadata staging size overflow")
        })?;
        anyhow::ensure!(
            total_bytes > 0,
            "native target replay metadata staging image is empty"
        );
        let destinations = entries
            .iter()
            .map(|(region, bytes, label)| {
                anyhow::ensure!(
                    bytes.len() <= region.bytes,
                    "native target replay metadata {label} has {} bytes, exceeding region capacity {}",
                    bytes.len(),
                    region.bytes
                );
                self.lane_region(lane, *region, label)
            })
            .collect::<Result<Vec<_>>>()?;
        let stream = self.stream_ptr(lane)?;
        let staging = self
            .metadata_staging
            .get_mut(lane)
            .with_context(|| format!("native target metadata lane {lane} is unavailable"))?;
        staging.ensure_capacity(
            self.library,
            total_bytes,
            "native target replay metadata staging",
        )?;
        let staging_buffer = staging.buffer;
        let mut offset = 0_usize;
        let mut batch_destinations = Vec::with_capacity(entries.len());
        let mut batch_sources = Vec::with_capacity(entries.len());
        let mut batch_bytes = Vec::with_capacity(entries.len());
        for ((_, bytes, _label), destination) in entries.iter().zip(destinations) {
            if bytes.is_empty() {
                continue;
            }
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    staging_buffer.ptr.cast::<u8>().add(offset),
                    bytes.len(),
                );
            }
            let source = Ds4rtHostBuffer {
                ptr: unsafe { staging_buffer.ptr.cast::<u8>().add(offset).cast() },
                bytes: bytes.len(),
                flags: staging_buffer.flags,
            };
            batch_destinations.push(destination);
            batch_sources.push(source);
            batch_bytes.push(bytes.len());
            offset += bytes.len();
        }
        if !batch_destinations.is_empty() {
            unsafe {
                self.library.copy_host_buffers_h2d_batch_async(
                    &batch_destinations,
                    &batch_sources,
                    &batch_bytes,
                    stream,
                )
            }
            .context("uploading native target replay metadata batch")?;
        }
        debug_assert_eq!(offset, total_bytes);
        copy_event
            .record(stream)
            .with_context(|| format!("recording native target metadata copy lane {lane}"))?;
        self.metadata_copy_in_flight[lane] = true;
        Ok(())
    }

    fn prepare_replay_metadata(
        &self,
        entries: Vec<(DeepseekV4TargetDeviceRegion, &[u8], &'static str)>,
    ) -> Result<DeepseekV4TargetPreparedReplayMetadata> {
        let total_bytes = entries.iter().try_fold(0_usize, |total, (_, bytes, _)| {
            total
                .checked_add(bytes.len())
                .context("native target prepared replay metadata size overflow")
        })?;
        anyhow::ensure!(
            total_bytes > 0,
            "native target prepared replay metadata image is empty"
        );
        let mut staging = ReusableHostBuffer::default();
        staging.ensure_capacity(
            self.library,
            total_bytes,
            "native target prepared replay metadata",
        )?;
        let mut offset = 0_usize;
        let mut prepared_entries = Vec::with_capacity(entries.len());
        for (destination, bytes, label) in entries {
            anyhow::ensure!(
                bytes.len() <= destination.bytes,
                "native target prepared replay metadata {label} has {} bytes, exceeding region capacity {}",
                bytes.len(),
                destination.bytes
            );
            if bytes.is_empty() {
                continue;
            }
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    staging.buffer.ptr.cast::<u8>().add(offset),
                    bytes.len(),
                );
            }
            prepared_entries.push(DeepseekV4TargetPreparedReplayMetadataEntry {
                destination,
                source_offset: offset,
                bytes: bytes.len(),
                label,
            });
            offset += bytes.len();
        }
        debug_assert_eq!(offset, total_bytes);
        Ok(DeepseekV4TargetPreparedReplayMetadata {
            staging,
            entries: prepared_entries,
        })
    }

    fn upload_prepared_replay_metadata_async(
        &self,
        lane: usize,
        metadata: &DeepseekV4TargetPreparedReplayMetadata,
    ) -> Result<()> {
        anyhow::ensure!(
            !metadata.entries.is_empty() && !metadata.staging.buffer.ptr.is_null(),
            "native target prepared replay metadata is empty"
        );
        let stream = self.stream_ptr(lane)?;
        let mut destinations = Vec::with_capacity(metadata.entries.len());
        let mut sources = Vec::with_capacity(metadata.entries.len());
        let mut bytes = Vec::with_capacity(metadata.entries.len());
        for entry in &metadata.entries {
            destinations.push(self.lane_region(lane, entry.destination, entry.label)?);
            sources.push(Ds4rtHostBuffer {
                ptr: unsafe {
                    metadata
                        .staging
                        .buffer
                        .ptr
                        .cast::<u8>()
                        .add(entry.source_offset)
                        .cast()
                },
                bytes: entry.bytes,
                flags: metadata.staging.buffer.flags,
            });
            bytes.push(entry.bytes);
        }
        unsafe {
            self.library
                .copy_host_buffers_h2d_batch_async(&destinations, &sources, &bytes, stream)
        }
        .context("uploading prepared native target replay metadata")
    }

    fn copy_persistent_hc_to_stage(
        &self,
        lane: usize,
        logical_row_start: usize,
        rows: usize,
        pong: bool,
    ) -> Result<()> {
        let stream = self.stream_ptr(lane)?;
        let active_row_start = self.hc_segment_row_offset(lane, logical_row_start, rows)?;
        for (plane, lane_bytes, stage, row_bytes, label) in self.hc_copy_regions(pong)? {
            let persistent = self.hc_lane_region(lane, plane, lane_bytes, label)?;
            let bytes = rows
                .checked_mul(row_bytes)
                .context("target persistent HC restore bytes overflow")?;
            let source = device_buffer_byte_view(
                persistent,
                active_row_start
                    .checked_mul(row_bytes)
                    .context("target persistent HC restore offset overflow")?,
                bytes,
                label,
            )?;
            let destination = self.lane_region(lane, stage, label)?;
            unsafe {
                self.library
                    .copy_d2d_async(destination, source, bytes, stream)?;
            }
        }
        Ok(())
    }

    fn copy_stage_hc_to_persistent(
        &self,
        lane: usize,
        logical_row_start: usize,
        rows: usize,
        pong: bool,
    ) -> Result<()> {
        let stream = self.stream_ptr(lane)?;
        let active_row_start = self.hc_segment_row_offset(lane, logical_row_start, rows)?;
        for (plane, lane_bytes, stage, row_bytes, label) in self.hc_copy_regions(pong)? {
            let persistent = self.hc_lane_region(lane, plane, lane_bytes, label)?;
            let bytes = rows
                .checked_mul(row_bytes)
                .context("target persistent HC save bytes overflow")?;
            let destination = device_buffer_byte_view(
                persistent,
                active_row_start
                    .checked_mul(row_bytes)
                    .context("target persistent HC save offset overflow")?,
                bytes,
                label,
            )?;
            let source = self.lane_region(lane, stage, label)?;
            unsafe {
                self.library
                    .copy_d2d_async(destination, source, bytes, stream)?;
            }
        }
        Ok(())
    }

    fn hc_copy_regions(
        &self,
        pong: bool,
    ) -> Result<
        [(
            DeepseekV4TargetDeviceRegion,
            usize,
            DeepseekV4TargetDeviceRegion,
            usize,
            &'static str,
        ); 3],
    > {
        let residual_row_bytes = TARGET_HC_MULT
            .checked_mul(self.plan.hidden)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("target HC residual row bytes overflow")?;
        let post_row_bytes = TARGET_HC_MULT * std::mem::size_of::<f32>();
        let comb_row_bytes = TARGET_HC_MULT * TARGET_HC_MULT * std::mem::size_of::<f32>();
        Ok(if pong {
            [
                (
                    self.plan.residual_pong,
                    self.plan.residual_lane_bytes,
                    self.plan.residual_stage_pong,
                    residual_row_bytes,
                    "target HC residual pong",
                ),
                (
                    self.plan.post_pong,
                    self.plan.post_lane_bytes,
                    self.plan.post_stage_pong,
                    post_row_bytes,
                    "target HC post pong",
                ),
                (
                    self.plan.comb_pong,
                    self.plan.comb_lane_bytes,
                    self.plan.comb_stage_pong,
                    comb_row_bytes,
                    "target HC combination pong",
                ),
            ]
        } else {
            [
                (
                    self.plan.residual_ping,
                    self.plan.residual_lane_bytes,
                    self.plan.residual_stage_ping,
                    residual_row_bytes,
                    "target HC residual ping",
                ),
                (
                    self.plan.post_ping,
                    self.plan.post_lane_bytes,
                    self.plan.post_stage_ping,
                    post_row_bytes,
                    "target HC post ping",
                ),
                (
                    self.plan.comb_ping,
                    self.plan.comb_lane_bytes,
                    self.plan.comb_stage_ping,
                    comb_row_bytes,
                    "target HC combination ping",
                ),
            ]
        })
    }

    pub(in crate::commands::real_full) fn smoke_replay_connected_attention(
        &mut self,
        rope_theta: f32,
    ) -> Result<()> {
        let lane = 0;
        let rows = 1;
        let attention_graph_prepared = |layer_id: usize| {
            self.plan
                .physical_plan
                .layer(layer_id)
                .is_some_and(|layer| match layer.compress_ratio {
                    0 => self.sliding_graphs.iter().any(|graph| {
                        graph.layer_id == layer_id && graph.lane == lane && graph.rows == rows
                    }),
                    4 => self.c4_graphs.iter().any(|graph| {
                        graph.layer_id == layer_id
                            && graph.lane == lane
                            && graph.rows == rows
                            && graph.lifecycle == DeepseekV4TargetC4Lifecycle::Prefill
                    }),
                    128 => self.c128_graphs.iter().any(|graph| {
                        graph.layer_id == layer_id
                            && graph.lane == lane
                            && graph.rows == rows
                            && graph.lifecycle == DeepseekV4TargetC128Lifecycle::Prefill
                    }),
                    _ => false,
                })
        };
        anyhow::ensure!(
            self.entry_graphs_prepared()
                && (0..4).all(attention_graph_prepared)
                && (0..4).all(|completed_layer_id| {
                    self.post_pre_graphs.iter().any(|graph| {
                        graph.completed_layer_id == completed_layer_id
                            && graph.lane == lane
                            && graph.rows == rows
                    })
                })
                && self.terminal_graphs_prepared(),
            "native target smoke replay requires the first four scheduled attention lifecycles"
        );
        let source =
            self.lane_region(lane, self.plan.hidden_output, "target smoke source hidden")?;
        let physical_slots = [0_u32];
        self.reset_hc_lane(lane)?;
        let layer0 =
            self.run_attention(lane, 0, 0, rows, source, None, &physical_slots, rope_theta)?;
        let layer1_input = self.run_post_pre(lane, 0, 0, rows, layer0.buffer())?.0;
        let layer1 = self.run_attention(
            lane,
            1,
            0,
            rows,
            layer1_input.buffer(),
            None,
            &physical_slots,
            rope_theta,
        )?;
        let layer2_input = self.run_post_pre(lane, 1, 0, rows, layer1.buffer())?.0;
        let layer2 = self.run_attention(
            lane,
            2,
            0,
            rows,
            layer2_input.buffer(),
            None,
            &physical_slots,
            rope_theta,
        )?;
        let layer3_input = self.run_post_pre(lane, 2, 0, rows, layer2.buffer())?.0;
        let layer3 = self.run_attention(
            lane,
            3,
            0,
            rows,
            layer3_input.buffer(),
            None,
            &physical_slots,
            rope_theta,
        )?;
        let layer4_input = self.run_post_pre(lane, 3, 0, rows, layer3.buffer())?.0;
        let terminal = self.run_terminal(lane, 0, rows, layer3.buffer())?.0;
        anyhow::ensure!(
            [
                layer0,
                layer1_input,
                layer1,
                layer2_input,
                layer2,
                layer3_input,
                layer3,
                layer4_input,
                terminal,
            ]
            .iter()
            .all(|hidden| hidden.rows == rows && hidden.values_per_row == self.plan.hidden),
            "native target attention smoke replay lost its fixed hidden shape"
        );
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct DeepseekV4TargetEntryWeights {
    hc_fn: Ds4rtDeviceBuffer,
    hc_scale: Ds4rtDeviceBuffer,
    hc_base: Ds4rtDeviceBuffer,
    norm_weight: Ds4rtDeviceBuffer,
}

#[derive(Clone, Copy)]
struct DeepseekV4TargetSlidingWeights {
    wq_a_weight: Ds4rtDeviceBuffer,
    wq_a_scale: Ds4rtDeviceBuffer,
    wq_b_weight: Ds4rtDeviceBuffer,
    wq_b_scale: Ds4rtDeviceBuffer,
    wkv_weight: Ds4rtDeviceBuffer,
    wkv_scale: Ds4rtDeviceBuffer,
    q_norm_weight: Ds4rtDeviceBuffer,
    kv_norm_weight: Ds4rtDeviceBuffer,
    wo_a_weight: Ds4rtDeviceBuffer,
    wo_a_scale: Ds4rtDeviceBuffer,
    wo_b_weight: Ds4rtDeviceBuffer,
    wo_b_scale: Ds4rtDeviceBuffer,
    attn_sink: Ds4rtDeviceBuffer,
    hc_fn: Ds4rtDeviceBuffer,
    hc_scale: Ds4rtDeviceBuffer,
    hc_base: Ds4rtDeviceBuffer,
    norm_weight: Ds4rtDeviceBuffer,
}

#[derive(Clone, Copy)]
struct DeepseekV4TargetC4Weights {
    common: DeepseekV4TargetSlidingWeights,
    main_wkv: Ds4rtDeviceBuffer,
    main_wgate: Ds4rtDeviceBuffer,
    main_ape: Ds4rtDeviceBuffer,
    main_norm: Ds4rtDeviceBuffer,
    index_wkv: Ds4rtDeviceBuffer,
    index_wgate: Ds4rtDeviceBuffer,
    index_ape: Ds4rtDeviceBuffer,
    index_norm: Ds4rtDeviceBuffer,
    indexer_wq_weight: Ds4rtDeviceBuffer,
    indexer_wq_scale: Ds4rtDeviceBuffer,
    indexer_weights_projection: Ds4rtDeviceBuffer,
}

#[derive(Clone, Copy)]
struct DeepseekV4TargetC128Weights {
    common: DeepseekV4TargetSlidingWeights,
    main_wkv: Ds4rtDeviceBuffer,
    main_wgate: Ds4rtDeviceBuffer,
    main_ape: Ds4rtDeviceBuffer,
    main_norm: Ds4rtDeviceBuffer,
}

#[derive(Clone, Copy)]
struct DeepseekV4TargetC4StateBuffers {
    main_kv: Ds4rtDeviceBuffer,
    main_score: Ds4rtDeviceBuffer,
    index_kv: Ds4rtDeviceBuffer,
    index_score: Ds4rtDeviceBuffer,
}

#[derive(Clone, Copy)]
struct DeepseekV4TargetC128StateBuffers {
    main_kv: Ds4rtDeviceBuffer,
    main_score: Ds4rtDeviceBuffer,
}

#[derive(Clone, Copy)]
struct DeepseekV4TargetPostPreWeights {
    hc_fn: Ds4rtDeviceBuffer,
    hc_scale: Ds4rtDeviceBuffer,
    hc_base: Ds4rtDeviceBuffer,
    norm_weight: Ds4rtDeviceBuffer,
}

#[derive(Clone, Copy)]
struct DeepseekV4TargetTerminalWeights {
    hc_head_fn: Ds4rtDeviceBuffer,
    hc_head_scale: Ds4rtDeviceBuffer,
    hc_head_base: Ds4rtDeviceBuffer,
    norm_weight: Ds4rtDeviceBuffer,
}

struct DeepseekV4TargetC0ReplayMetadata {
    positions: Vec<u32>,
    main_slots: Vec<u32>,
    cos_sin: Vec<f32>,
    selected_indices: Vec<i32>,
    selected_lengths: Vec<i32>,
}

struct DeepseekV4TargetC4ReplayMetadata {
    positions: Vec<u32>,
    main_slots: Vec<u32>,
    cos_sin: Vec<f32>,
    swa_indices: Vec<i32>,
    swa_lengths: Vec<i32>,
    active_groups: u32,
    group_source_starts: Vec<u32>,
    group_sequence_slots: Vec<u32>,
    group_rope_positions: Vec<u32>,
    compressed_slots: Vec<u32>,
    sequence_offsets: Vec<u32>,
    sequence_start_positions: Vec<u32>,
    state_sequence_ids: Vec<u32>,
    real_page_table: Vec<i32>,
    real_page_stride: usize,
    index_cache_seqlens: Vec<i32>,
}

struct DeepseekV4TargetC128ReplayMetadata {
    positions: Vec<u32>,
    main_slots: Vec<u32>,
    cos_sin: Vec<f32>,
    swa_indices: Vec<i32>,
    swa_lengths: Vec<i32>,
    active_groups: u32,
    group_source_starts: Vec<u32>,
    group_sequence_slots: Vec<u32>,
    group_rope_positions: Vec<u32>,
    compressed_slots: Vec<u32>,
    sequence_offsets: Vec<u32>,
    sequence_start_positions: Vec<u32>,
    state_sequence_ids: Vec<u32>,
    indexed_indices: Vec<i32>,
    indexed_stride: usize,
    indexed_lengths: Vec<i32>,
}

impl Drop for DeepseekV4TargetDeviceStorage {
    fn drop(&mut self) {
        for stream in &self.streams {
            let _ = stream.synchronize();
        }
        if !self.arena.ptr.is_null() {
            let _ = self.library.free_device_buffer(&mut self.arena);
        }
    }
}

fn target_layer0_entry_weights(hidden: usize) -> Result<DeepseekV4TargetEntryWeights> {
    let hc_fn_bytes = TARGET_HC_MIXES
        .checked_mul(hidden)
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .context("native target layer-zero HC lane-sum bytes overflow")?;
    let norm_bytes = hidden
        .checked_mul(std::mem::size_of::<u16>())
        .context("native target layer-zero attention norm bytes overflow")?;
    Ok(DeepseekV4TargetEntryWeights {
        hc_fn: preloaded_resident_weight_device_buffer(
            TARGET_LAYER0_HC_ATTN_FN_LANE_SUM,
            hc_fn_bytes,
        )?,
        hc_scale: preloaded_resident_weight_device_buffer(
            TARGET_LAYER0_HC_ATTN_SCALE,
            3 * std::mem::size_of::<f32>(),
        )?,
        hc_base: preloaded_resident_weight_device_buffer(
            TARGET_LAYER0_HC_ATTN_BASE,
            TARGET_HC_MIXES * std::mem::size_of::<f32>(),
        )?,
        norm_weight: preloaded_resident_weight_device_buffer(TARGET_LAYER0_ATTN_NORM, norm_bytes)?,
    })
}

fn target_sliding_attention_weights(
    attention: &DeepseekV4AttentionPlan,
    layer_id: usize,
) -> Result<DeepseekV4TargetSlidingWeights> {
    let layer = attention
        .layer(layer_id)
        .with_context(|| format!("native target attention layer {layer_id} is missing"))?;
    anyhow::ensure!(
        layer.logical_layer_id < attention.target_layer_count,
        "native target layer {layer_id} is not a target attention layer"
    );
    let geometry = &attention.geometry;
    let hidden = geometry.hidden_size;
    let query_width = geometry
        .attention_heads
        .checked_mul(geometry.head_dim)
        .context("native target query width overflow")?;
    let output_group_width = query_width
        .checked_div(geometry.o_groups)
        .context("native target output group count must be positive")?;
    let output_projected_width = geometry
        .o_groups
        .checked_mul(geometry.o_lora_rank)
        .context("native target output projected width overflow")?;
    let matrix_bytes = |rows: usize, columns: usize, label: &'static str| {
        rows.checked_mul(columns)
            .with_context(|| format!("{label} bytes overflow"))
    };
    let scale_bytes = |rows: usize, columns: usize, label: &'static str| {
        rows.div_ceil(128)
            .checked_mul(columns.div_ceil(128))
            .with_context(|| format!("{label} scale bytes overflow"))
    };
    let prefix = format!("layers.{layer_id}");
    let resident = |suffix: &str, bytes: usize| {
        preloaded_resident_weight_device_buffer(&format!("{prefix}.{suffix}"), bytes)
    };
    Ok(DeepseekV4TargetSlidingWeights {
        wq_a_weight: resident(
            "attn.wq_a.weight",
            matrix_bytes(geometry.q_lora_rank, hidden, "target wq_a")?,
        )?,
        wq_a_scale: resident(
            "attn.wq_a.scale",
            scale_bytes(geometry.q_lora_rank, hidden, "target wq_a")?,
        )?,
        wq_b_weight: resident(
            "attn.wq_b.weight",
            matrix_bytes(query_width, geometry.q_lora_rank, "target wq_b")?,
        )?,
        wq_b_scale: resident(
            "attn.wq_b.scale",
            scale_bytes(query_width, geometry.q_lora_rank, "target wq_b")?,
        )?,
        wkv_weight: resident(
            "attn.wkv.weight",
            matrix_bytes(geometry.head_dim, hidden, "target wkv")?,
        )?,
        wkv_scale: resident(
            "attn.wkv.scale",
            scale_bytes(geometry.head_dim, hidden, "target wkv")?,
        )?,
        q_norm_weight: resident(
            "attn.q_norm.weight",
            geometry
                .q_lora_rank
                .checked_mul(std::mem::size_of::<u16>())
                .context("target Q norm bytes overflow")?,
        )?,
        kv_norm_weight: resident(
            "attn.kv_norm.weight",
            geometry
                .head_dim
                .checked_mul(std::mem::size_of::<u16>())
                .context("target KV norm bytes overflow")?,
        )?,
        wo_a_weight: resident(
            "attn.wo_a.weight",
            matrix_bytes(output_projected_width, output_group_width, "target wo_a")?,
        )?,
        wo_a_scale: resident(
            "attn.wo_a.scale",
            scale_bytes(output_projected_width, output_group_width, "target wo_a")?,
        )?,
        wo_b_weight: resident(
            "attn.wo_b.weight",
            matrix_bytes(hidden, output_projected_width, "target wo_b")?,
        )?,
        wo_b_scale: resident(
            "attn.wo_b.scale",
            scale_bytes(hidden, output_projected_width, "target wo_b")?,
        )?,
        attn_sink: resident(
            "attn.attn_sink",
            geometry
                .attention_heads
                .checked_mul(std::mem::size_of::<f32>())
                .context("target attention sink bytes overflow")?,
        )?,
        hc_fn: resident(
            "hc_ffn_fn",
            TARGET_HC_MIXES
                .checked_mul(TARGET_HC_MULT)
                .and_then(|values| values.checked_mul(hidden))
                .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
                .context("target FFN HC function bytes overflow")?,
        )?,
        hc_scale: resident("hc_ffn_scale", 3 * std::mem::size_of::<f32>())?,
        hc_base: resident("hc_ffn_base", TARGET_HC_MIXES * std::mem::size_of::<f32>())?,
        norm_weight: resident(
            "ffn_norm.weight",
            hidden
                .checked_mul(std::mem::size_of::<u16>())
                .context("target FFN norm bytes overflow")?,
        )?,
    })
}

fn target_c4_attention_weights(
    attention: &DeepseekV4AttentionPlan,
    layer_id: usize,
) -> Result<DeepseekV4TargetC4Weights> {
    let layer = attention
        .layer(layer_id)
        .with_context(|| format!("native target C4 layer {layer_id} is missing"))?;
    anyhow::ensure!(
        layer.logical_layer_id < attention.target_layer_count && layer.compress_ratio == 4,
        "native target layer {layer_id} is not a target C4 layer"
    );
    let hidden = attention.geometry.hidden_size;
    let q_rank = attention.geometry.q_lora_rank;
    let prefix = format!("layers.{layer_id}.attn");
    let resident = |suffix: &str, bytes: usize| {
        preloaded_resident_weight_device_buffer(&format!("{prefix}.{suffix}"), bytes)
    };
    let matrix_bf16_bytes = |rows: usize, columns: usize, label: &'static str| {
        rows.checked_mul(columns)
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<u16>()))
            .with_context(|| format!("{label} bytes overflow"))
    };
    Ok(DeepseekV4TargetC4Weights {
        common: target_sliding_attention_weights(attention, layer_id)?,
        main_wkv: resident(
            "compressor.wkv.weight",
            matrix_bf16_bytes(1_024, hidden, "target C4 main WKV")?,
        )?,
        main_wgate: resident(
            "compressor.wgate.weight",
            matrix_bf16_bytes(1_024, hidden, "target C4 main gate")?,
        )?,
        main_ape: resident("compressor.ape", 4 * 1_024 * std::mem::size_of::<f32>())?,
        main_norm: resident("compressor.norm.weight", 512 * std::mem::size_of::<u16>())?,
        index_wkv: resident(
            "indexer.compressor.wkv.weight",
            matrix_bf16_bytes(256, hidden, "target C4 index WKV")?,
        )?,
        index_wgate: resident(
            "indexer.compressor.wgate.weight",
            matrix_bf16_bytes(256, hidden, "target C4 index gate")?,
        )?,
        index_ape: resident(
            "indexer.compressor.ape",
            4 * 256 * std::mem::size_of::<f32>(),
        )?,
        index_norm: resident(
            "indexer.compressor.norm.weight",
            128 * std::mem::size_of::<u16>(),
        )?,
        indexer_wq_weight: resident(
            "indexer.wq_b.weight",
            DS4_INDEX_HEADS
                .checked_mul(DS4_INDEX_HEAD_DIM)
                .and_then(|rows| rows.checked_mul(q_rank))
                .context("target C4 indexer WQ bytes overflow")?,
        )?,
        indexer_wq_scale: resident(
            "indexer.wq_b.scale",
            (DS4_INDEX_HEADS * DS4_INDEX_HEAD_DIM)
                .div_ceil(128)
                .checked_mul(q_rank.div_ceil(128))
                .context("target C4 indexer WQ scale bytes overflow")?,
        )?,
        indexer_weights_projection: resident(
            "indexer.weights_proj.weight",
            matrix_bf16_bytes(DS4_INDEX_HEADS, hidden, "target C4 indexer weights")?,
        )?,
    })
}

fn target_c128_attention_weights(
    attention: &DeepseekV4AttentionPlan,
    layer_id: usize,
) -> Result<DeepseekV4TargetC128Weights> {
    let layer = attention
        .layer(layer_id)
        .with_context(|| format!("native target C128 layer {layer_id} is missing"))?;
    anyhow::ensure!(
        layer.logical_layer_id < attention.target_layer_count && layer.compress_ratio == 128,
        "native target layer {layer_id} is not a target C128 layer"
    );
    let hidden = attention.geometry.hidden_size;
    let prefix = format!("layers.{layer_id}.attn");
    let resident = |suffix: &str, bytes: usize| {
        preloaded_resident_weight_device_buffer(&format!("{prefix}.{suffix}"), bytes)
    };
    let matrix_bf16_bytes = |rows: usize, columns: usize, label: &'static str| {
        rows.checked_mul(columns)
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<u16>()))
            .with_context(|| format!("{label} bytes overflow"))
    };
    Ok(DeepseekV4TargetC128Weights {
        common: target_sliding_attention_weights(attention, layer_id)?,
        main_wkv: resident(
            "compressor.wkv.weight",
            matrix_bf16_bytes(512, hidden, "target C128 main WKV")?,
        )?,
        main_wgate: resident(
            "compressor.wgate.weight",
            matrix_bf16_bytes(512, hidden, "target C128 main gate")?,
        )?,
        main_ape: resident("compressor.ape", 128 * 512 * std::mem::size_of::<f32>())?,
        main_norm: resident("compressor.norm.weight", 512 * std::mem::size_of::<u16>())?,
    })
}

fn target_post_pre_weights(
    hidden: usize,
    next_layer_id: usize,
) -> Result<DeepseekV4TargetPostPreWeights> {
    let hc_fn_bytes = TARGET_HC_MIXES
        .checked_mul(TARGET_HC_MULT)
        .and_then(|values| values.checked_mul(hidden))
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .context("native target next-attention HC function bytes overflow")?;
    let norm_bytes = hidden
        .checked_mul(std::mem::size_of::<u16>())
        .context("native target next-attention norm bytes overflow")?;
    let prefix = format!("layers.{next_layer_id}");
    let resident = |suffix: &str, bytes: usize| {
        preloaded_resident_weight_device_buffer(&format!("{prefix}.{suffix}"), bytes)
    };
    Ok(DeepseekV4TargetPostPreWeights {
        hc_fn: resident("hc_attn_fn", hc_fn_bytes)?,
        hc_scale: resident("hc_attn_scale", 3 * std::mem::size_of::<f32>())?,
        hc_base: resident("hc_attn_base", TARGET_HC_MIXES * std::mem::size_of::<f32>())?,
        norm_weight: resident("attn_norm.weight", norm_bytes)?,
    })
}

fn target_terminal_weights(hidden: usize) -> Result<DeepseekV4TargetTerminalWeights> {
    let hc_head_fn_bytes = TARGET_HC_MULT
        .checked_mul(TARGET_HC_MULT)
        .and_then(|values| values.checked_mul(hidden))
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .context("native target terminal HC head function bytes overflow")?;
    let norm_bytes = hidden
        .checked_mul(std::mem::size_of::<u16>())
        .context("native target terminal norm bytes overflow")?;
    Ok(DeepseekV4TargetTerminalWeights {
        hc_head_fn: preloaded_resident_weight_device_buffer("hc_head_fn", hc_head_fn_bytes)?,
        hc_head_scale: preloaded_resident_weight_device_buffer(
            "hc_head_scale",
            std::mem::size_of::<f32>(),
        )?,
        hc_head_base: preloaded_resident_weight_device_buffer(
            "hc_head_base",
            TARGET_HC_MULT * std::mem::size_of::<f32>(),
        )?,
        norm_weight: preloaded_resident_weight_device_buffer("norm.weight", norm_bytes)?,
    })
}

fn target_graph_row_buckets(max_graph_rows: usize) -> Vec<usize> {
    TARGET_GRAPH_ROWS
        .into_iter()
        .filter(|rows| *rows <= max_graph_rows)
        .collect()
}

fn target_default_rope_inv_freq(
    rope_theta: f32,
    rope_width: usize,
) -> Result<[f32; TARGET_ROPE_WIDTH / 2]> {
    anyhow::ensure!(
        rope_theta.is_finite() && rope_theta > 0.0 && rope_width == TARGET_ROPE_WIDTH,
        "native target attention requires finite positive RoPE theta and width {TARGET_ROPE_WIDTH}"
    );
    Ok(std::array::from_fn(|pair| {
        rope_theta.powf(-2.0 * pair as f32 / rope_width as f32)
    }))
}

fn target_yarn_rope_inv_freq(
    rope_theta: f32,
    rope_width: usize,
    factor: f32,
    original_max_position_embeddings: usize,
    beta_fast: f32,
    beta_slow: f32,
) -> Result<[f32; TARGET_ROPE_WIDTH / 2]> {
    anyhow::ensure!(
        rope_theta.is_finite()
            && rope_theta > 0.0
            && rope_width == TARGET_ROPE_WIDTH
            && factor.is_finite()
            && factor > 0.0
            && original_max_position_embeddings > 0
            && beta_fast.is_finite()
            && beta_fast > 0.0
            && beta_slow.is_finite()
            && beta_slow > 0.0
            && beta_fast >= beta_slow,
        "native target attention has invalid compressed YaRN geometry"
    );
    let correction_dim = |rotations: f32| {
        rope_width as f64
            * ((original_max_position_embeddings as f64
                / (rotations as f64 * 2.0 * std::f64::consts::PI))
                .ln())
            / (2.0 * (rope_theta as f64).ln())
    };
    // Transformers' YaRN implementation defaults to truncating the correction
    // range. Keep the same floor/ceil and [0, rope_width - 1] clamp so the
    // host-generated frequency table is the checkpoint's exact contract.
    let low = correction_dim(beta_fast)
        .floor()
        .clamp(0.0, (rope_width - 1) as f64) as f32;
    let high = correction_dim(beta_slow)
        .ceil()
        .clamp(0.0, (rope_width - 1) as f64) as f32;
    let ramp_denominator = if low == high { 0.001 } else { high - low };
    Ok(std::array::from_fn(|pair| {
        let pos_freq = rope_theta.powf(2.0 * pair as f32 / rope_width as f32);
        let inv_freq_extrapolation = 1.0 / pos_freq;
        let inv_freq_interpolation = 1.0 / (factor * pos_freq);
        let ramp = ((pair as f32 - low) / ramp_denominator).clamp(0.0, 1.0);
        inv_freq_interpolation * ramp + inv_freq_extrapolation * (1.0 - ramp)
    }))
}

fn target_attention_rope_inv_freq<'a>(
    compress_ratio: usize,
    main: &'a [f32; TARGET_ROPE_WIDTH / 2],
    compress: &'a [f32; TARGET_ROPE_WIDTH / 2],
) -> Result<&'a [f32; TARGET_ROPE_WIDTH / 2]> {
    anyhow::ensure!(
        main.iter()
            .chain(compress)
            .all(|value| value.is_finite() && *value > 0.0),
        "native target attention requires finite positive base/compressed RoPE frequencies"
    );
    match compress_ratio {
        0 => Ok(main),
        4 | 128 => Ok(compress),
        ratio => anyhow::bail!("native target attention has unsupported compression ratio {ratio}"),
    }
}

fn target_connected_compressed_layer_ids(
    attention: &DeepseekV4AttentionPlan,
    compress_ratio: usize,
) -> Vec<usize> {
    attention
        .target_layers()
        .iter()
        .filter(|layer| layer.compress_ratio == compress_ratio)
        .map(|layer| layer.logical_layer_id)
        .collect()
}

fn target_connected_post_pre_layer_ids(attention: &DeepseekV4AttentionPlan) -> Vec<usize> {
    attention
        .target_layers()
        .iter()
        .filter(|layer| layer.logical_layer_id + 1 < attention.target_layer_count)
        .map(|layer| layer.logical_layer_id)
        .collect()
}

fn target_c0_replay_metadata(
    logical_row_start: usize,
    rows: usize,
    physical_slots: &[u32],
    rope_inv_freq: &[f32; TARGET_ROPE_WIDTH / 2],
) -> Result<DeepseekV4TargetC0ReplayMetadata> {
    anyhow::ensure!(
        rows > 0
            && rope_inv_freq
                .iter()
                .all(|value| value.is_finite() && *value > 0.0)
            && physical_slots.len() >= logical_row_start + rows,
        "target C0 replay metadata has invalid rows, RoPE frequencies, or physical-slot extent"
    );
    let positions = (0..rows).map(|row| row as u32).collect::<Vec<_>>();
    let main_slots = physical_slots[logical_row_start..logical_row_start + rows].to_vec();
    let pairs = TARGET_ROPE_WIDTH / 2;
    let mut cos_sin = vec![0.0_f32; rows * TARGET_ROPE_WIDTH];
    let mut selected_indices = vec![0_i32; rows * TARGET_SWA_WIDTH];
    let mut selected_lengths = vec![0_i32; rows];
    for row in 0..rows {
        let logical_position = logical_row_start + row;
        for pair in 0..pairs {
            let angle = logical_position as f32 * rope_inv_freq[pair];
            let (sin, cos) = angle.sin_cos();
            cos_sin[row * TARGET_ROPE_WIDTH + pair] = cos;
            cos_sin[row * TARGET_ROPE_WIDTH + pairs + pair] = sin;
        }
        let first = (logical_position + 1).saturating_sub(TARGET_SWA_WIDTH);
        let length = logical_position + 1 - first;
        selected_lengths[row] =
            i32::try_from(length).context("target C0 SWA length exceeds i32")?;
        for (column, slot) in physical_slots[first..=logical_position]
            .iter()
            .copied()
            .enumerate()
        {
            selected_indices[row * TARGET_SWA_WIDTH + column] =
                i32::try_from(slot).context("target C0 physical selection slot exceeds i32")?;
        }
    }
    Ok(DeepseekV4TargetC0ReplayMetadata {
        positions,
        main_slots,
        cos_sin,
        selected_indices,
        selected_lengths,
    })
}

fn target_c4_replay_metadata(
    logical_row_start: usize,
    rows: usize,
    physical_slots: &[u32],
    rope_inv_freq: &[f32; TARGET_ROPE_WIDTH / 2],
    max_graph_rows: usize,
    source_pages: usize,
) -> Result<DeepseekV4TargetC4ReplayMetadata> {
    let logical_end = logical_row_start
        .checked_add(rows)
        .context("target C4 logical row range overflow")?;
    anyhow::ensure!(
        rows > 0
            && rows <= max_graph_rows
            && source_pages > 0
            && rope_inv_freq
                .iter()
                .all(|value| value.is_finite() && *value > 0.0)
            && physical_slots.len() >= logical_end,
        "target C4 replay metadata has invalid rows, pages, RoPE frequencies, or physical slots"
    );
    let positions = (0..rows)
        .map(|row| u32::try_from(row).context("target C4 local position exceeds u32"))
        .collect::<Result<Vec<_>>>()?;
    let main_slots = physical_slots[logical_row_start..logical_end].to_vec();
    let pairs = TARGET_ROPE_WIDTH / 2;
    let cos_sin_rows = max_graph_rows
        .checked_add(TARGET_COMPRESSOR_CARRY_ROPE_SLOTS)
        .context("target C4 RoPE row capacity overflow")?;
    let mut cos_sin = vec![0.0_f32; cos_sin_rows * TARGET_ROPE_WIDTH];
    let write_rope = |cos_sin: &mut [f32], slot: usize, logical_position: usize| {
        for pair in 0..pairs {
            let angle = logical_position as f32 * rope_inv_freq[pair];
            let (sin, cos) = angle.sin_cos();
            cos_sin[slot * TARGET_ROPE_WIDTH + pair] = cos;
            cos_sin[slot * TARGET_ROPE_WIDTH + pairs + pair] = sin;
        }
    };
    for row in 0..rows {
        write_rope(&mut cos_sin, row, logical_row_start + row);
    }

    let mut swa_indices = vec![0_i32; rows * TARGET_SWA_WIDTH];
    let mut swa_lengths = vec![0_i32; rows];
    let mut index_cache_seqlens = vec![0_i32; rows];
    for row in 0..rows {
        let logical_position = logical_row_start + row;
        let first = (logical_position + 1).saturating_sub(TARGET_SWA_WIDTH);
        let length = logical_position + 1 - first;
        swa_lengths[row] = i32::try_from(length).context("target C4 SWA length exceeds i32")?;
        index_cache_seqlens[row] = i32::try_from((logical_position + 1) / 4)
            .context("target C4 index-cache length exceeds i32")?;
        for (column, slot) in physical_slots[first..=logical_position]
            .iter()
            .copied()
            .enumerate()
        {
            swa_indices[row * TARGET_SWA_WIDTH + column] =
                i32::try_from(slot).context("target C4 SWA physical slot exceeds i32")?;
        }
    }

    let first_completion = logical_row_start + (4 - logical_row_start % 4) - 1;
    let completion_positions = if first_completion < logical_end {
        (first_completion..logical_end)
            .step_by(4)
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let mut group_source_starts = Vec::with_capacity(completion_positions.len());
    let mut group_sequence_slots = Vec::with_capacity(completion_positions.len());
    let mut group_rope_positions = Vec::with_capacity(completion_positions.len());
    let mut compressed_slots = Vec::with_capacity(completion_positions.len());
    let mut extra_rope_slots = 0_usize;
    for completion in completion_positions.iter().copied() {
        let group_start = completion - 3;
        group_source_starts.push(
            u32::try_from(group_start).context("target C4 group source position exceeds u32")?,
        );
        group_sequence_slots.push(0);
        let rope_slot = if group_start >= logical_row_start {
            group_start - logical_row_start
        } else {
            let slot = max_graph_rows + extra_rope_slots;
            extra_rope_slots += 1;
            anyhow::ensure!(
                slot < cos_sin_rows,
                "target C4 carried group has no fixed RoPE metadata slot"
            );
            write_rope(&mut cos_sin, slot, group_start);
            slot
        };
        group_rope_positions
            .push(u32::try_from(rope_slot).context("target C4 group RoPE slot exceeds u32")?);
        let source_slot = usize::try_from(physical_slots[completion])
            .context("target C4 source physical slot exceeds usize")?;
        let compressed_slot = source_slot / 256 * 64 + source_slot % 256 / 4;
        compressed_slots.push(
            u32::try_from(compressed_slot)
                .context("target C4 compressed physical slot exceeds u32")?,
        );
    }

    // The captured graph retains the physical-pool width as its destination
    // stride.  Only pages containing completed C4 groups are visible through
    // index_cache_seqlens, so materialize the prefix this replay can reach.
    let real_page_stride = (logical_end / 4).div_ceil(64).max(1);
    anyhow::ensure!(
        real_page_stride <= source_pages,
        "target C4 active page-table width {real_page_stride} exceeds physical capacity {source_pages}"
    );
    let page_ids = (0..real_page_stride)
        .map(|page| {
            let source_index = page * 256;
            let physical_page = physical_slots
                .get(source_index)
                .copied()
                .unwrap_or_default()
                / 256;
            i32::try_from(physical_page).context("target C4 physical page exceeds i32")
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(DeepseekV4TargetC4ReplayMetadata {
        positions,
        main_slots,
        cos_sin,
        swa_indices,
        swa_lengths,
        active_groups: u32::try_from(completion_positions.len())
            .context("target C4 active group count exceeds u32")?,
        group_source_starts,
        group_sequence_slots,
        group_rope_positions,
        compressed_slots,
        sequence_offsets: vec![
            0,
            u32::try_from(rows).context("target C4 sequence row count exceeds u32")?,
        ],
        sequence_start_positions: vec![
            u32::try_from(logical_row_start).context("target C4 sequence start exceeds u32")?
        ],
        state_sequence_ids: vec![0],
        real_page_table: page_ids,
        real_page_stride,
        index_cache_seqlens,
    })
}

fn target_c128_replay_metadata(
    logical_row_start: usize,
    rows: usize,
    physical_slots: &[u32],
    rope_inv_freq: &[f32; TARGET_ROPE_WIDTH / 2],
    max_graph_rows: usize,
    source_pages: usize,
) -> Result<DeepseekV4TargetC128ReplayMetadata> {
    let logical_end = logical_row_start
        .checked_add(rows)
        .context("target C128 logical row range overflow")?;
    anyhow::ensure!(
        rows > 0
            && rows <= max_graph_rows
            && source_pages > 0
            && rope_inv_freq
                .iter()
                .all(|value| value.is_finite() && *value > 0.0)
            && physical_slots.len() >= logical_end,
        "target C128 replay metadata has invalid rows, pages, RoPE frequencies, or physical slots"
    );
    let positions = (0..rows)
        .map(|row| u32::try_from(row).context("target C128 local position exceeds u32"))
        .collect::<Result<Vec<_>>>()?;
    let main_slots = physical_slots[logical_row_start..logical_end].to_vec();
    let pairs = TARGET_ROPE_WIDTH / 2;
    let cos_sin_rows = max_graph_rows
        .checked_add(TARGET_COMPRESSOR_CARRY_ROPE_SLOTS)
        .context("target C128 RoPE row capacity overflow")?;
    let mut cos_sin = vec![0.0_f32; cos_sin_rows * TARGET_ROPE_WIDTH];
    let write_rope = |cos_sin: &mut [f32], slot: usize, logical_position: usize| {
        for pair in 0..pairs {
            let angle = logical_position as f32 * rope_inv_freq[pair];
            let (sin, cos) = angle.sin_cos();
            cos_sin[slot * TARGET_ROPE_WIDTH + pair] = cos;
            cos_sin[slot * TARGET_ROPE_WIDTH + pairs + pair] = sin;
        }
    };
    for row in 0..rows {
        write_rope(&mut cos_sin, row, logical_row_start + row);
    }

    let mut swa_indices = vec![0_i32; rows * TARGET_SWA_WIDTH];
    let mut swa_lengths = vec![0_i32; rows];
    let indexed_capacity = target_c128_indexed_width(source_pages)?;
    // The completed-block list is identical for every row; only the causal
    // prefix in indexed_lengths varies. Keep one shared list and bind it with
    // row stride zero instead of constructing and uploading rows copies.
    let indexed_stride = (logical_end / 128)
        .max(TARGET_C128_PREFILL_SELECTION_TILE)
        .div_ceil(TARGET_C128_PREFILL_SELECTION_TILE)
        .saturating_mul(TARGET_C128_PREFILL_SELECTION_TILE)
        .min(indexed_capacity);
    let mut indexed_indices = vec![0_i32; indexed_stride];
    let mut indexed_lengths = vec![0_i32; rows];
    for row in 0..rows {
        let logical_position = logical_row_start + row;
        let first = (logical_position + 1).saturating_sub(TARGET_SWA_WIDTH);
        let swa_length = logical_position + 1 - first;
        swa_lengths[row] =
            i32::try_from(swa_length).context("target C128 SWA length exceeds i32")?;
        for (column, slot) in physical_slots[first..=logical_position]
            .iter()
            .copied()
            .enumerate()
        {
            swa_indices[row * TARGET_SWA_WIDTH + column] =
                i32::try_from(slot).context("target C128 SWA physical slot exceeds i32")?;
        }

        let indexed_length = (logical_position + 1) / 128;
        anyhow::ensure!(
            indexed_length <= indexed_stride,
            "target C128 indexed length {indexed_length} exceeds packed stride {indexed_stride}"
        );
        indexed_lengths[row] =
            i32::try_from(indexed_length).context("target C128 indexed length exceeds i32")?;
    }
    let indexed_length = logical_end / 128;
    for block in 0..indexed_length {
        let completion = block * 128 + 127;
        let source_slot = usize::try_from(physical_slots[completion])
            .context("target C128 indexed source slot exceeds usize")?;
        indexed_indices[block] = i32::try_from(source_slot / 128)
            .context("target C128 indexed physical slot exceeds i32")?;
    }

    let first_completion = logical_row_start + (128 - logical_row_start % 128) - 1;
    let completion_positions = if first_completion < logical_end {
        (first_completion..logical_end)
            .step_by(128)
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let mut group_source_starts = Vec::with_capacity(completion_positions.len());
    let mut group_sequence_slots = Vec::with_capacity(completion_positions.len());
    let mut group_rope_positions = Vec::with_capacity(completion_positions.len());
    let mut compressed_slots = Vec::with_capacity(completion_positions.len());
    let mut extra_rope_slots = 0_usize;
    for completion in completion_positions.iter().copied() {
        let group_start = completion - 127;
        group_source_starts.push(
            u32::try_from(group_start).context("target C128 group source position exceeds u32")?,
        );
        group_sequence_slots.push(0);
        let rope_slot = if group_start >= logical_row_start {
            group_start - logical_row_start
        } else {
            let slot = max_graph_rows + extra_rope_slots;
            extra_rope_slots += 1;
            anyhow::ensure!(
                slot < cos_sin_rows,
                "target C128 carried group has no fixed RoPE metadata slot"
            );
            write_rope(&mut cos_sin, slot, group_start);
            slot
        };
        group_rope_positions
            .push(u32::try_from(rope_slot).context("target C128 group RoPE slot exceeds u32")?);
        let source_slot = usize::try_from(physical_slots[completion])
            .context("target C128 source physical slot exceeds usize")?;
        compressed_slots.push(
            u32::try_from(source_slot / 128)
                .context("target C128 compressed physical slot exceeds u32")?,
        );
    }

    Ok(DeepseekV4TargetC128ReplayMetadata {
        positions,
        main_slots,
        cos_sin,
        swa_indices,
        swa_lengths,
        active_groups: u32::try_from(completion_positions.len())
            .context("target C128 active group count exceeds u32")?,
        group_source_starts,
        group_sequence_slots,
        group_rope_positions,
        compressed_slots,
        sequence_offsets: vec![
            0,
            u32::try_from(rows).context("target C128 sequence row count exceeds u32")?,
        ],
        sequence_start_positions: vec![
            u32::try_from(logical_row_start).context("target C128 sequence start exceeds u32")?
        ],
        state_sequence_ids: vec![0],
        indexed_indices,
        indexed_stride,
        indexed_lengths,
    })
}

fn target_c128_indexed_width(source_pages: usize) -> Result<usize> {
    Ok(source_pages
        .checked_mul(TARGET_C128_BLOCKS_PER_SOURCE_PAGE)
        .context("target C128 indexed width overflow")?
        .max(TARGET_C128_PREFILL_SELECTION_TILE))
}

fn native_little_endian_host_bytes<T>(values: &[T]) -> &[u8] {
    assert!(
        cfg!(target_endian = "little"),
        "native target metadata byte views require a little-endian host"
    );
    // Primitive numeric slices contain initialized bytes without padding, and
    // byte slices may alias any initialized representation. The returned view
    // retains the input lifetime and is copied into pinned staging before the
    // input can be mutated or dropped.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn u32_host_bytes(values: &[u32]) -> &[u8] {
    native_little_endian_host_bytes(values)
}

fn i32_host_bytes(values: &[i32]) -> &[u8] {
    native_little_endian_host_bytes(values)
}

fn f32_host_bytes(values: &[f32]) -> &[u8] {
    native_little_endian_host_bytes(values)
}

fn preload_target_layer0_hc_attn_lane_sum(catalog: &TensorCatalog, hidden: usize) -> Result<()> {
    let source_columns = TARGET_HC_MULT
        .checked_mul(hidden)
        .context("native target layer-zero HC source width overflow")?;
    let source_bytes = TARGET_HC_MIXES
        .checked_mul(source_columns)
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .context("native target layer-zero HC source bytes overflow")?;
    let reduced_bytes = TARGET_HC_MIXES
        .checked_mul(hidden)
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .context("native target layer-zero HC lane-sum bytes overflow")?;
    let mut source = vec![0_u8; source_bytes];
    let summary = read_tensor_bytes_into(catalog, TARGET_LAYER0_HC_ATTN_FN, &mut source)
        .context("reading native target layer-zero attention HC function")?;
    anyhow::ensure!(
        summary.dtype == DType::F32
            && summary.shape == [TARGET_HC_MIXES, source_columns]
            && summary.bytes_read as usize == source_bytes,
        "native target layer-zero attention HC function must be F32 [{TARGET_HC_MIXES}, {source_columns}]"
    );
    preload_resident_weight_from_host_staging(
        TARGET_LAYER0_HC_ATTN_FN_LANE_SUM,
        reduced_bytes,
        "native target layer-zero lane-summed attention HC function",
        |destination| sum_repeated_hc_input_lanes(&source, destination, hidden),
    )
    .context("preloading native target layer-zero lane-summed attention HC function")
}

fn sum_repeated_hc_input_lanes(source: &[u8], destination: &mut [u8], hidden: usize) -> Result<()> {
    let source_columns = TARGET_HC_MULT
        .checked_mul(hidden)
        .context("target HC lane-sum source width overflow")?;
    let expected_source_bytes = TARGET_HC_MIXES
        .checked_mul(source_columns)
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .context("target HC lane-sum source bytes overflow")?;
    let expected_destination_bytes = TARGET_HC_MIXES
        .checked_mul(hidden)
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .context("target HC lane-sum destination bytes overflow")?;
    anyhow::ensure!(
        hidden > 0
            && source.len() == expected_source_bytes
            && destination.len() == expected_destination_bytes,
        "target HC lane-sum buffers do not match 24 x 4 x {hidden} F32"
    );
    for mix in 0..TARGET_HC_MIXES {
        for column in 0..hidden {
            let mut sum = 0.0_f32;
            for lane in 0..TARGET_HC_MULT {
                let source_index = mix * source_columns + lane * hidden + column;
                let source_offset = source_index * std::mem::size_of::<f32>();
                sum += f32::from_le_bytes(
                    source[source_offset..source_offset + std::mem::size_of::<f32>()]
                        .try_into()
                        .expect("validated target HC f32 source slice"),
                );
            }
            let destination_index = mix * hidden + column;
            let destination_offset = destination_index * std::mem::size_of::<f32>();
            destination[destination_offset..destination_offset + std::mem::size_of::<f32>()]
                .copy_from_slice(&sum.to_le_bytes());
        }
    }
    Ok(())
}

fn target_python_device_buffer(
    name: &'static str,
    buffer: Ds4rtDeviceBuffer,
) -> PythonDeviceBufferArg<'static> {
    PythonDeviceBufferArg {
        name,
        ptr: buffer.ptr,
        bytes: buffer.bytes,
        device_id: buffer.device_id,
        flags: buffer.flags,
    }
}

fn query_target_attention_arena_bytes(
    variant: &str,
    max_rows: usize,
    source_pages: usize,
    max_page_table_width: usize,
    cache_format: &str,
) -> Result<usize> {
    [("extend", 0_usize), ("extend", 4), ("extend", 128)]
        .into_iter()
        .try_fold(0_usize, |largest, (mode, compression)| {
            let kwargs = [
                ("variant", PythonKernelArg::Str(variant)),
                ("mode", PythonKernelArg::Str(mode)),
                ("compression", PythonKernelArg::Usize(compression)),
                ("max_rows", PythonKernelArg::Usize(max_rows)),
                ("source_pages", PythonKernelArg::Usize(source_pages)),
                (
                    "max_page_table_width",
                    PythonKernelArg::Usize(max_page_table_width),
                ),
                ("max_positions", PythonKernelArg::Usize(max_rows)),
                ("swa_width", PythonKernelArg::Usize(TARGET_SWA_WIDTH)),
                ("cache_format", PythonKernelArg::Str(cache_format)),
            ];
            let bytes = query_python_usize_during_startup(PythonUsizeQuery {
                module: TARGET_ATTENTION_CAPTURE_MODULE,
                function: TARGET_ATTENTION_ARENA_NBYTES_FUNCTION,
                kwargs: &kwargs,
            })?;
            Ok(largest.max(bytes))
        })
}

fn query_target_c4_selector_scratch_bytes(
    variant: &str,
    max_rows: usize,
    source_pages: usize,
    max_page_table_width: usize,
) -> Result<usize> {
    let kwargs = [
        ("variant", PythonKernelArg::Str(variant)),
        ("mode", PythonKernelArg::Str("prefill")),
        ("max_rows", PythonKernelArg::Usize(max_rows)),
        ("source_pages", PythonKernelArg::Usize(source_pages)),
        (
            "max_page_table_width",
            PythonKernelArg::Usize(max_page_table_width),
        ),
    ];
    query_python_usize_during_startup(PythonUsizeQuery {
        module: TARGET_ATTENTION_CAPTURE_MODULE,
        function: TARGET_C4_SELECTOR_SCRATCH_NBYTES_FUNCTION,
        kwargs: &kwargs,
    })
}

fn query_target_mhc_scratch_bytes(variant: &str, max_rows: usize) -> Result<usize> {
    let kwargs = [
        ("variant", PythonKernelArg::Str(variant)),
        ("max_rows", PythonKernelArg::Usize(max_rows)),
    ];
    query_python_usize_during_startup(PythonUsizeQuery {
        module: TARGET_MHC_CAPTURE_MODULE,
        function: TARGET_MHC_SCRATCH_NBYTES_FUNCTION,
        kwargs: &kwargs,
    })
}

fn push_region(
    cursor: &mut usize,
    bytes: usize,
    label: &'static str,
) -> Result<DeepseekV4TargetDeviceRegion> {
    anyhow::ensure!(bytes > 0, "{label} must reserve nonzero bytes");
    let offset =
        align_up(*cursor, TARGET_CAPTURE_ALIGNMENT).with_context(|| format!("aligning {label}"))?;
    *cursor = offset
        .checked_add(bytes)
        .with_context(|| format!("advancing {label}"))?;
    Ok(DeepseekV4TargetDeviceRegion { offset, bytes })
}

fn align_up(value: usize, alignment: usize) -> Option<usize> {
    debug_assert!(alignment.is_power_of_two());
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
}

fn checked_bytes(shape: &[usize], element_bytes: usize, label: &'static str) -> Result<usize> {
    shape
        .iter()
        .try_fold(element_bytes, |bytes, dim| bytes.checked_mul(*dim))
        .with_context(|| format!("{label} byte count overflow"))
}

fn target_hc_active_rows_per_lane(
    max_graph_rows: usize,
    max_sequence_rows: usize,
) -> Result<usize> {
    let bounded_wavefront_rows = REAL_FULL_BOUNDED_PREFILL_MAX_ACTIVE_CHUNKS
        .checked_mul(max_graph_rows)
        .and_then(|rows| rows.checked_add(REAL_FULL_TARGET_AUXILIARY_MAX_ROWS))
        .context("target mHC bounded-wavefront row capacity overflow")?;
    Ok(REAL_FULL_LAYER_MAJOR_PREFILL_MAX_ROWS
        .max(bounded_wavefront_rows)
        .min(max_sequence_rows))
}

fn validate_target_device_storage_plan(plan: &DeepseekV4TargetDeviceStoragePlan) -> Result<()> {
    anyhow::ensure!(
        plan.expert_tensor_parallel == 4 && !plan.expert_parallel,
        "target storage must preserve strict TP4 rather than expert parallelism"
    );
    anyhow::ensure!(
        plan.rope_theta.is_finite()
            && plan.rope_theta > 0.0
            && plan.compress_rope_theta.is_finite()
            && plan.compress_rope_theta > 0.0
            && plan
                .main_rope_inv_freq
                .iter()
                .chain(&plan.compress_rope_inv_freq)
                .all(|value| value.is_finite() && *value > 0.0),
        "target storage requires finite positive base/compressed RoPE geometry"
    );
    anyhow::ensure!(
        plan.physical_pool_rows >= plan.max_sequence_rows
            && plan.cache_format == plan.physical_plan.cache_format.kernel_label()
            && plan.source_page_tokens == plan.physical_plan.source_page_tokens
            && plan.source_pages == plan.physical_plan.source_page_count
            && plan.source_pages == plan.physical_pool_rows.div_ceil(plan.source_page_tokens)
            && plan.max_sequence_pages == plan.max_sequence_rows.div_ceil(plan.source_page_tokens)
            && plan.max_sequence_pages <= plan.source_pages,
        "target storage lost its independent sequence/pool page geometry"
    );
    anyhow::ensure!(
        plan.real_page_table.bytes
            == checked_bytes(
                &[plan.max_sequence_pages],
                std::mem::size_of::<u32>(),
                "target validated C4 shared page table",
            )?
            && plan.c128_indexed_indices.bytes
                == checked_bytes(
                    &[target_c128_indexed_width(plan.max_sequence_pages)?],
                    std::mem::size_of::<i32>(),
                    "target validated C128 shared indices",
                )?,
        "target graph metadata is not bounded by per-sequence pages"
    );
    anyhow::ensure!(
        plan.execution_lanes > 0
            && plan.lane_storage.offset == 0
            && plan.lane_storage.bytes == plan.lane_stride_bytes * plan.execution_lanes,
        "target per-lane workspace storage lost its fixed stride"
    );
    anyhow::ensure!(
        plan.hc_active_rows_per_lane
            == target_hc_active_rows_per_lane(plan.max_graph_rows, plan.max_sequence_rows)?
            && plan.hc_active_rows_per_lane > 0
            && plan.hc_active_rows_per_lane <= plan.max_sequence_rows,
        "target mHC storage lost its scheduler-bounded active row geometry"
    );
    anyhow::ensure!(
        plan.entry_mhc_scratch.offset == plan.workspace.offset
            && plan.entry_mhc_scratch.end() <= plan.workspace.end(),
        "target entry mHC scratch does not reuse the composite workspace"
    );
    let lane_regions = [
        plan.workspace,
        plan.selector_scratch,
        plan.hidden_input,
        plan.hidden_work,
        plan.hidden_output,
        plan.positions,
        plan.main_slots,
        plan.cos_sin,
        plan.selected_indices,
        plan.selected_lengths,
        plan.active_groups,
        plan.group_source_starts,
        plan.group_sequence_slots,
        plan.group_rope_positions,
        plan.compressed_slots,
        plan.active_sequences,
        plan.sequence_offsets,
        plan.sequence_start_positions,
        plan.state_sequence_ids,
        plan.real_page_table,
        plan.index_cache_seqlens,
        plan.c128_indexed_indices,
        plan.c128_indexed_lengths,
        plan.residual_stage_ping,
        plan.residual_stage_pong,
        plan.post_stage_ping,
        plan.post_stage_pong,
        plan.comb_stage_ping,
        plan.comb_stage_pong,
    ];
    for (index, region) in lane_regions.iter().enumerate() {
        anyhow::ensure!(
            region.offset % TARGET_CAPTURE_ALIGNMENT == 0
                && region.bytes > 0
                && region.end() <= plan.lane_stride_bytes,
            "target lane region {index} exceeds or loses alignment in its fixed stride"
        );
        if let Some(next) = lane_regions.get(index + 1) {
            anyhow::ensure!(
                region.end() <= next.offset,
                "target lane regions {index} and {} overlap",
                index + 1
            );
        }
    }
    let shared_regions = [
        plan.residual_ping,
        plan.residual_pong,
        plan.post_ping,
        plan.post_pong,
        plan.comb_ping,
        plan.comb_pong,
        plan.physical_kv,
        plan.compressor_state,
    ];
    anyhow::ensure!(
        plan.residual_ping.bytes == plan.residual_lane_bytes * plan.execution_lanes
            && plan.residual_pong.bytes == plan.residual_lane_bytes * plan.execution_lanes
            && plan.post_ping.bytes == plan.post_lane_bytes * plan.execution_lanes
            && plan.post_pong.bytes == plan.post_lane_bytes * plan.execution_lanes
            && plan.comb_ping.bytes == plan.comb_lane_bytes * plan.execution_lanes
            && plan.comb_pong.bytes == plan.comb_lane_bytes * plan.execution_lanes,
        "target per-lane HC planes lost their fixed execution-lane stride"
    );
    anyhow::ensure!(
        plan.residual_lane_bytes
            == checked_bytes(
                &[plan.hc_active_rows_per_lane, TARGET_HC_MULT, plan.hidden],
                std::mem::size_of::<u16>(),
                "target validated HC residual",
            )?
            && plan.post_lane_bytes
                == checked_bytes(
                    &[plan.hc_active_rows_per_lane, TARGET_HC_MULT],
                    std::mem::size_of::<f32>(),
                    "target validated HC post",
                )?
            && plan.comb_lane_bytes
                == checked_bytes(
                    &[plan.hc_active_rows_per_lane, TARGET_HC_MULT, TARGET_HC_MULT,],
                    std::mem::size_of::<f32>(),
                    "target validated HC combination",
                )?,
        "target mHC planes are not sized from active rows"
    );
    for (index, region) in shared_regions.iter().enumerate() {
        anyhow::ensure!(
            region.offset % TARGET_CAPTURE_ALIGNMENT == 0
                && region.bytes > 0
                && region.end() <= plan.total_bytes,
            "target shared region {index} exceeds or loses alignment in its fixed arena"
        );
        if index == 0 {
            anyhow::ensure!(
                plan.lane_storage.end() <= region.offset,
                "target shared state overlaps per-lane graph storage"
            );
        }
        if let Some(next) = shared_regions.get(index + 1) {
            anyhow::ensure!(
                region.end() <= next.offset,
                "target shared regions {index} and {} overlap",
                index + 1
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flash_target_storage_keeps_one_workspace_and_persistent_hc_kv() -> Result<()> {
        let facts = ModelFacts::default();
        let plan = plan_deepseek_v4_target_device_storage(&facts, 4, 2_048, 16_384, 16_384)?;
        assert_eq!(plan.variant, "flash");
        assert_eq!(plan.hidden, 4_096);
        assert_eq!(plan.rope_theta, 10_000.0);
        assert_eq!(plan.compress_rope_theta, 160_000.0);
        assert_eq!(plan.main_rope_inv_freq[0], 1.0);
        assert!((plan.compress_rope_inv_freq[24] - 0.000_019_531_253).abs() < 1.0e-11);
        assert_eq!(plan.target_layers, 43);
        assert_eq!(plan.execution_lanes, 4);
        assert_eq!(plan.hc_active_rows_per_lane, 16_384);
        assert_eq!(plan.physical_pool_rows, 16_384);
        assert_eq!(plan.source_page_tokens, 256);
        assert_eq!(plan.source_pages, 64);
        assert_eq!(plan.max_sequence_pages, 64);
        assert_eq!(plan.workspace.bytes, 645_925_888);
        assert_eq!(plan.selector_scratch.bytes, 42_501_120);
        assert_eq!(plan.entry_mhc_scratch.bytes, 13_107_200);
        assert_eq!(plan.entry_mhc_scratch.offset, plan.workspace.offset);
        assert_eq!(plan.lane_storage.bytes, plan.lane_stride_bytes * 4);
        assert_eq!(
            plan.residual_stage_ping.bytes,
            2_048 * TARGET_HC_MULT * 4_096 * std::mem::size_of::<u16>()
        );
        assert_eq!(
            plan.residual_stage_ping.bytes,
            plan.residual_stage_pong.bytes
        );
        assert_eq!(plan.expert_tensor_parallel, 4);
        assert!(!plan.expert_parallel);
        assert_eq!(
            plan.residual_lane_bytes,
            16_384 * TARGET_HC_MULT * 4_096 * std::mem::size_of::<u16>()
        );
        assert_eq!(plan.residual_ping.bytes, plan.residual_lane_bytes * 4);
        assert_eq!(plan.residual_ping.bytes, plan.residual_pong.bytes);
        assert_eq!(plan.active_groups.bytes, 2_048 * 4);
        assert_eq!(plan.sequence_offsets.bytes, (2_048 + 1) * 4);
        assert_eq!(plan.real_page_table.bytes, 64 * 4);
        assert_eq!(plan.c128_indexed_indices.bytes, 128 * 4);
        assert_eq!(plan.c128_indexed_lengths.bytes, 2_048 * 4);
        assert!(plan.physical_kv.bytes > 0);
        assert!(plan.compressor_state.bytes > 0);
        assert_eq!(plan.status, "planned-native-target-capture-not-active");
        validate_target_device_storage_plan(&plan)
    }

    #[test]
    fn flash_native_nvfp4_target_storage_uses_exact_compact_physical_pages() -> Result<()> {
        let facts = ModelFacts::default();
        let plan = plan_deepseek_v4_target_device_storage_with_cache_format(
            &facts,
            4,
            2_048,
            16_384,
            16_384,
            DeepseekV4KvCacheFormat::Nvfp4,
        )?;

        assert_eq!(plan.cache_format, "nvfp4");
        assert_eq!(
            plan.physical_plan.cache_format,
            DeepseekV4KvCacheFormat::Nvfp4
        );
        assert_eq!(plan.source_pages, 64);
        assert_eq!(plan.physical_kv.bytes, 64 * 5_530_752);
        assert_eq!(
            plan.physical_plan
                .layer(0)
                .and_then(|layer| layer.region(ds4rt_core::DeepseekV4KvRegionKind::Main))
                .map(|region| region.bytes_per_page),
            Some(110_592),
        );
        validate_target_device_storage_plan(&plan)
    }

    #[test]
    fn target_graph_metadata_is_bounded_by_sequence_not_shared_pool() -> Result<()> {
        let facts = ModelFacts::default();
        let one_context = plan_deepseek_v4_target_device_storage(&facts, 4, 2_048, 16_384, 16_384)?;
        let shared_pool = plan_deepseek_v4_target_device_storage(&facts, 4, 2_048, 16_384, 32_768)?;
        assert_eq!(one_context.max_sequence_pages, 64);
        assert_eq!(shared_pool.max_sequence_pages, 64);
        assert_eq!(shared_pool.source_pages, 128);
        assert_eq!(shared_pool.workspace.bytes, one_context.workspace.bytes);
        assert_eq!(
            shared_pool.selector_scratch.bytes,
            one_context.selector_scratch.bytes
        );
        assert_eq!(
            shared_pool.real_page_table.bytes,
            one_context.real_page_table.bytes
        );
        assert_eq!(
            shared_pool.c128_indexed_indices.bytes,
            one_context.c128_indexed_indices.bytes
        );
        assert!(shared_pool.physical_kv.bytes > one_context.physical_kv.bytes);
        validate_target_device_storage_plan(&shared_pool)
    }

    #[test]
    fn flash_target_mhc_is_bounded_by_active_frontier_not_context() -> Result<()> {
        let facts = ModelFacts::default();
        let medium = plan_deepseek_v4_target_device_storage(&facts, 4, 2_048, 131_072, 131_072)?;
        let long = plan_deepseek_v4_target_device_storage(&facts, 4, 2_048, 400_000, 400_000)?;
        let expected_active_rows = REAL_FULL_BOUNDED_PREFILL_MAX_ACTIVE_CHUNKS * 2_048
            + REAL_FULL_TARGET_AUXILIARY_MAX_ROWS;
        assert_eq!(medium.hc_active_rows_per_lane, expected_active_rows);
        assert_eq!(long.hc_active_rows_per_lane, expected_active_rows);
        assert_eq!(medium.residual_lane_bytes, long.residual_lane_bytes);
        assert_eq!(medium.post_lane_bytes, long.post_lane_bytes);
        assert_eq!(medium.comb_lane_bytes, long.comb_lane_bytes);
        let persistent_hc_bytes = |plan: &DeepseekV4TargetDeviceStoragePlan| {
            plan.residual_ping.bytes * 2 + plan.post_ping.bytes * 2 + plan.comb_ping.bytes * 2
        };
        assert_eq!(persistent_hc_bytes(&medium), persistent_hc_bytes(&long));
        assert!(long.physical_kv.bytes > medium.physical_kv.bytes);
        assert!(long.total_bytes > medium.total_bytes);
        Ok(())
    }

    #[test]
    fn target_mhc_lane_allocator_reuses_and_coalesces_variable_segments() -> Result<()> {
        let mut lane = DeepseekV4TargetHcLaneState::new(10)?;
        let first = DeepseekV4TargetHcSegmentKey {
            logical_row_start: 100,
            rows: 4,
        };
        let second = DeepseekV4TargetHcSegmentKey {
            logical_row_start: 200,
            rows: 3,
        };
        let third = DeepseekV4TargetHcSegmentKey {
            logical_row_start: 300,
            rows: 2,
        };
        assert_eq!(lane.allocate(first)?.row_offset, 0);
        assert_eq!(lane.allocate(second)?.row_offset, 4);
        lane.release(first)?;
        assert_eq!(lane.allocate(third)?.row_offset, 0);
        lane.release(second)?;
        lane.release(third)?;
        assert!(lane.allocations.is_empty());
        assert_eq!(
            lane.free_ranges,
            vec![DeepseekV4TargetHcFreeRange {
                row_offset: 0,
                rows: 10,
            }]
        );
        Ok(())
    }

    #[test]
    fn target_mhc_lane_resolves_contiguous_fused_segments() -> Result<()> {
        let mut lane = DeepseekV4TargetHcLaneState::new(10)?;
        let decode = DeepseekV4TargetHcSegmentKey {
            logical_row_start: 100,
            rows: 1,
        };
        let proposals = DeepseekV4TargetHcSegmentKey {
            logical_row_start: 101,
            rows: 4,
        };
        lane.allocate(decode)?;
        lane.allocate(proposals)?;
        assert_eq!(
            lane.resolve(DeepseekV4TargetHcSegmentKey {
                logical_row_start: 100,
                rows: 5,
            })?,
            DeepseekV4TargetHcAllocation {
                row_offset: 0,
                rows: 5,
            }
        );
        lane.release(decode)?;
        lane.release(proposals)?;
        Ok(())
    }

    #[test]
    fn target_mhc_lane_releases_contiguous_fused_segments() -> Result<()> {
        let mut lane = DeepseekV4TargetHcLaneState::new(10)?;
        let decode = DeepseekV4TargetHcSegmentKey {
            logical_row_start: 100,
            rows: 1,
        };
        let proposals = DeepseekV4TargetHcSegmentKey {
            logical_row_start: 101,
            rows: 4,
        };
        lane.allocate(decode)?;
        lane.allocate(proposals)?;
        assert_eq!(
            lane.release(DeepseekV4TargetHcSegmentKey {
                logical_row_start: 100,
                rows: 5,
            })?,
            DeepseekV4TargetHcAllocation {
                row_offset: 0,
                rows: 5,
            }
        );
        assert!(lane.allocations.is_empty());
        assert_eq!(
            lane.free_ranges,
            vec![DeepseekV4TargetHcFreeRange {
                row_offset: 0,
                rows: 10,
            }]
        );
        Ok(())
    }

    #[test]
    fn target_mhc_lane_allocator_fails_closed_when_frontier_is_exhausted() -> Result<()> {
        let mut lane = DeepseekV4TargetHcLaneState::new(4)?;
        lane.allocate(DeepseekV4TargetHcSegmentKey {
            logical_row_start: 0,
            rows: 4,
        })?;
        let error = lane
            .allocate(DeepseekV4TargetHcSegmentKey {
                logical_row_start: 4,
                rows: 1,
            })
            .unwrap_err();
        assert!(error.to_string().contains("active frontier exhausted"));
        Ok(())
    }

    #[test]
    fn target_storage_rejects_graph_rows_wider_than_sequence() {
        let error =
            plan_deepseek_v4_target_device_storage(&ModelFacts::default(), 4, 2_048, 1_024, 1_024)
                .unwrap_err();
        assert!(error
            .to_string()
            .contains("graph rows 2048 must be within sequence rows 1024"));
    }

    #[test]
    fn flash_smoke_target_storage_pads_c128_indices_to_sm120_prefill_tile() -> Result<()> {
        let plan =
            plan_deepseek_v4_target_device_storage(&ModelFacts::default(), 4, 2_048, 4_096, 4_096)?;
        assert_eq!(plan.source_pages, 16);
        assert_eq!(plan.max_sequence_pages, 16);
        assert_eq!(
            plan.c128_indexed_indices.bytes,
            TARGET_C128_PREFILL_SELECTION_TILE * std::mem::size_of::<i32>()
        );
        validate_target_device_storage_plan(&plan)
    }

    #[test]
    fn flash_connected_graph_layers_cover_target_and_terminal_boundary() -> Result<()> {
        let attention = DeepseekV4AttentionPlan::from_model_facts(&ModelFacts::default())?;
        assert_eq!(
            target_connected_compressed_layer_ids(&attention, 4),
            (2..=42).step_by(2).collect::<Vec<_>>()
        );
        assert_eq!(
            target_connected_compressed_layer_ids(&attention, 128),
            (3..=41).step_by(2).collect::<Vec<_>>()
        );
        assert_eq!(
            target_connected_post_pre_layer_ids(&attention),
            (0..=41).collect::<Vec<_>>()
        );
        assert_eq!(attention.target_layer_count, 43);
        Ok(())
    }

    #[test]
    fn pro_connected_graph_layers_cover_model_target_and_terminal_boundary() -> Result<()> {
        let mut facts = ModelFacts::default();
        facts.variant = ModelVariant::Pro;
        facts.hidden_size = ds4rt_core::DS4_PRO_HIDDEN_SIZE;
        facts.num_hidden_layers = ds4rt_core::DS4_PRO_NUM_HIDDEN_LAYERS;
        facts.attention_heads = 128;
        facts.q_lora_rank = ds4rt_core::DS4_PRO_Q_LORA_RANK;
        facts.o_groups = 16;
        facts.compress_ratios = ds4rt_core::DS4_PRO_COMPRESS_RATIOS.to_vec();
        facts.dspark_markov_rank = ds4rt_core::DS4_PRO_DSPARK_MARKOV_RANK;
        facts.dspark_target_layer_ids = vec![58, 59, 60];

        let attention = DeepseekV4AttentionPlan::from_model_facts(&facts)?;
        assert!(attention
            .target_layers()
            .iter()
            .all(|layer| layer.compress_ratio != 0));
        assert!(attention
            .dspark_layers()
            .iter()
            .all(|layer| layer.compress_ratio == 0));
        assert_eq!(
            target_connected_compressed_layer_ids(&attention, 4),
            (2..=60).step_by(2).collect::<Vec<_>>()
        );
        assert_eq!(
            target_connected_compressed_layer_ids(&attention, 128),
            [0, 1]
                .into_iter()
                .chain((3..=59).step_by(2))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            target_connected_post_pre_layer_ids(&attention),
            (0..=59).collect::<Vec<_>>()
        );
        assert_eq!(attention.target_layer_count, 61);
        Ok(())
    }

    #[test]
    fn c4_replay_metadata_tracks_initial_compressed_groups_and_causal_lengths() -> Result<()> {
        let slots = (0..8).collect::<Vec<_>>();
        let rope = target_default_rope_inv_freq(10_000.0, TARGET_ROPE_WIDTH)?;
        let metadata = target_c4_replay_metadata(0, 8, &slots, &rope, 16, 1)?;
        assert_eq!(metadata.positions, (0..8).collect::<Vec<_>>());
        assert_eq!(metadata.main_slots, slots);
        assert_eq!(metadata.active_groups, 2);
        assert_eq!(metadata.group_source_starts, vec![0, 4]);
        assert_eq!(metadata.group_rope_positions, vec![0, 4]);
        assert_eq!(metadata.compressed_slots, vec![0, 1]);
        assert_eq!(metadata.sequence_offsets, vec![0, 8]);
        assert_eq!(metadata.sequence_start_positions, vec![0]);
        assert_eq!(metadata.index_cache_seqlens, vec![0, 0, 0, 1, 1, 1, 1, 2]);
        assert_eq!(metadata.real_page_stride, 1);
        assert_eq!(metadata.real_page_table, vec![0]);
        Ok(())
    }

    #[test]
    fn c4_replay_metadata_carries_an_overlapping_group_into_continuation() -> Result<()> {
        let slots = (512..520).collect::<Vec<_>>();
        let rope = target_default_rope_inv_freq(10_000.0, TARGET_ROPE_WIDTH)?;
        let metadata = target_c4_replay_metadata(6, 2, &slots, &rope, 16, 1)?;
        assert_eq!(metadata.positions, vec![0, 1]);
        assert_eq!(metadata.main_slots, vec![518, 519]);
        assert_eq!(metadata.active_groups, 1);
        assert_eq!(metadata.group_source_starts, vec![4]);
        assert_eq!(metadata.group_sequence_slots, vec![0]);
        assert_eq!(metadata.group_rope_positions, vec![16]);
        assert_eq!(metadata.compressed_slots, vec![129]);
        assert_eq!(metadata.sequence_start_positions, vec![6]);
        assert_eq!(metadata.index_cache_seqlens, vec![1, 2]);
        assert_eq!(metadata.real_page_stride, 1);
        assert_eq!(metadata.real_page_table, vec![2]);
        Ok(())
    }

    #[test]
    fn c4_replay_metadata_packs_only_reachable_physical_pages() -> Result<()> {
        let slots = (0..4_096).collect::<Vec<_>>();
        let rope = target_default_rope_inv_freq(10_000.0, TARGET_ROPE_WIDTH)?;
        let metadata = target_c4_replay_metadata(3_072, 1_024, &slots, &rope, 1_024, 6_250)?;
        assert_eq!(metadata.real_page_stride, 16);
        assert_eq!(metadata.real_page_table.len(), 16);
        assert_eq!(metadata.index_cache_seqlens[0], 768);
        assert_eq!(metadata.index_cache_seqlens[1_023], 1_024);
        assert_eq!(
            &metadata.real_page_table[..16],
            &(0..16).collect::<Vec<_>>()
        );
        Ok(())
    }

    #[test]
    fn c4_replay_metadata_preserves_noncontiguous_source_page_ids() -> Result<()> {
        let physical_slots = (0..256_u32)
            .map(|offset| 5 * 256 + offset)
            .chain((0..256_u32).map(|offset| 2 * 256 + offset))
            .collect::<Vec<_>>();
        let rope = target_default_rope_inv_freq(10_000.0, TARGET_ROPE_WIDTH)?;
        let metadata = target_c4_replay_metadata(252, 8, &physical_slots, &rope, 16, 2)?;
        assert_eq!(metadata.main_slots[0], 5 * 256 + 252);
        assert_eq!(metadata.main_slots[4], 2 * 256);
        assert_eq!(metadata.compressed_slots, vec![5 * 64 + 63, 2 * 64]);
        assert_eq!(metadata.real_page_table, [5, 2]);
        Ok(())
    }

    #[test]
    fn c4_replay_metadata_reserves_a_carry_rope_slot_for_a_full_chunk() -> Result<()> {
        let logical_row_start = 32_769_usize;
        let rows = 2_048_usize;
        let slots = (0..logical_row_start + rows)
            .map(u32::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let rope = target_default_rope_inv_freq(10_000.0, TARGET_ROPE_WIDTH)?;
        let source_pages = (logical_row_start + rows).div_ceil(256);
        let metadata =
            target_c4_replay_metadata(logical_row_start, rows, &slots, &rope, rows, source_pages)?;
        assert_eq!(metadata.group_source_starts[0], 32_768);
        assert_eq!(metadata.group_rope_positions[0], rows as u32);
        assert_eq!(
            metadata.cos_sin[rows * TARGET_ROPE_WIDTH],
            32_768.0_f32.cos()
        );
        Ok(())
    }

    #[test]
    fn c128_replay_metadata_tracks_initial_compressed_group_and_causal_lengths() -> Result<()> {
        let slots = (0..128).collect::<Vec<_>>();
        let rope = target_default_rope_inv_freq(10_000.0, TARGET_ROPE_WIDTH)?;
        let metadata = target_c128_replay_metadata(0, 128, &slots, &rope, 128, 1)?;
        assert_eq!(metadata.positions, (0..128).collect::<Vec<_>>());
        assert_eq!(metadata.main_slots, slots);
        assert_eq!(metadata.active_groups, 1);
        assert_eq!(metadata.group_source_starts, vec![0]);
        assert_eq!(metadata.group_rope_positions, vec![0]);
        assert_eq!(metadata.compressed_slots, vec![0]);
        assert_eq!(metadata.sequence_offsets, vec![0, 128]);
        assert_eq!(metadata.sequence_start_positions, vec![0]);
        assert_eq!(&metadata.indexed_lengths[..127], &[0; 127]);
        assert_eq!(metadata.indexed_lengths[127], 1);
        assert_eq!(metadata.indexed_stride, TARGET_C128_PREFILL_SELECTION_TILE);
        assert_eq!(
            metadata.indexed_indices.len(),
            TARGET_C128_PREFILL_SELECTION_TILE
        );
        assert_eq!(metadata.indexed_indices[0], 0);
        Ok(())
    }

    #[test]
    fn c128_replay_metadata_packs_only_the_active_index_tiles() -> Result<()> {
        let slots = (0..128).collect::<Vec<_>>();
        let rope = target_default_rope_inv_freq(10_000.0, TARGET_ROPE_WIDTH)?;
        let metadata = target_c128_replay_metadata(0, 128, &slots, &rope, 128, 128)?;
        assert_eq!(target_c128_indexed_width(128)?, 256);
        assert_eq!(metadata.indexed_stride, TARGET_C128_PREFILL_SELECTION_TILE);
        assert_eq!(
            metadata.indexed_indices.len(),
            TARGET_C128_PREFILL_SELECTION_TILE
        );
        assert_eq!(metadata.indexed_lengths[127], 1);
        Ok(())
    }

    #[test]
    fn c128_replay_metadata_expands_packed_stride_at_tile_boundary() -> Result<()> {
        let slots = (0..8_320).collect::<Vec<_>>();
        let rope = target_default_rope_inv_freq(10_000.0, TARGET_ROPE_WIDTH)?;
        let metadata = target_c128_replay_metadata(8_192, 128, &slots, &rope, 128, 128)?;
        assert_eq!(target_c128_indexed_width(128)?, 256);
        assert_eq!(
            metadata.indexed_stride,
            2 * TARGET_C128_PREFILL_SELECTION_TILE
        );
        assert_eq!(metadata.indexed_indices.len(), 128);
        assert_eq!(metadata.indexed_lengths[0], 64);
        assert_eq!(metadata.indexed_lengths[127], 65);
        assert_eq!(
            &metadata.indexed_indices[..64],
            &(0..64).collect::<Vec<_>>()
        );
        assert_eq!(
            &metadata.indexed_indices[..65],
            &(0..65).collect::<Vec<_>>()
        );
        Ok(())
    }

    #[test]
    fn c128_replay_metadata_carries_an_overlapping_group_into_continuation() -> Result<()> {
        let slots = (512..642).collect::<Vec<_>>();
        let rope = target_default_rope_inv_freq(10_000.0, TARGET_ROPE_WIDTH)?;
        let metadata = target_c128_replay_metadata(126, 4, &slots, &rope, 16, 1)?;
        assert_eq!(metadata.positions, vec![0, 1, 2, 3]);
        assert_eq!(metadata.main_slots, vec![638, 639, 640, 641]);
        assert_eq!(metadata.active_groups, 1);
        assert_eq!(metadata.group_source_starts, vec![0]);
        assert_eq!(metadata.group_sequence_slots, vec![0]);
        assert_eq!(metadata.group_rope_positions, vec![16]);
        assert_eq!(metadata.compressed_slots, vec![4]);
        assert_eq!(metadata.sequence_start_positions, vec![126]);
        assert_eq!(metadata.indexed_lengths, vec![0, 1, 1, 1]);
        assert_eq!(metadata.indexed_indices[0], 4);
        Ok(())
    }

    #[test]
    fn c128_replay_metadata_separates_global_group_position_from_local_rope_slot() -> Result<()> {
        let slots = (0..4_096).collect::<Vec<_>>();
        let rope = target_default_rope_inv_freq(10_000.0, TARGET_ROPE_WIDTH)?;
        let metadata = target_c128_replay_metadata(2_048, 128, &slots, &rope, 128, 16)?;
        assert_eq!(metadata.active_groups, 1);
        assert_eq!(metadata.group_source_starts, vec![2_048]);
        assert_eq!(metadata.group_rope_positions, vec![0]);
        assert_eq!(metadata.sequence_start_positions, vec![2_048]);
        assert_eq!(metadata.compressed_slots, vec![16]);
        Ok(())
    }

    #[test]
    fn c128_replay_metadata_reserves_a_carry_rope_slot_for_a_full_chunk() -> Result<()> {
        let logical_row_start = 32_769_usize;
        let rows = 2_048_usize;
        let slots = (0..logical_row_start + rows)
            .map(u32::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let rope = target_default_rope_inv_freq(10_000.0, TARGET_ROPE_WIDTH)?;
        let source_pages = (logical_row_start + rows).div_ceil(256);
        let metadata = target_c128_replay_metadata(
            logical_row_start,
            rows,
            &slots,
            &rope,
            rows,
            source_pages,
        )?;
        assert_eq!(metadata.group_source_starts[0], 32_768);
        assert_eq!(metadata.group_rope_positions[0], rows as u32);
        assert_eq!(
            metadata.cos_sin[rows * TARGET_ROPE_WIDTH],
            32_768.0_f32.cos()
        );
        Ok(())
    }

    #[test]
    fn c128_replay_metadata_preserves_noncontiguous_source_page_ids() -> Result<()> {
        let physical_slots = (0..256_u32)
            .map(|offset| 5 * 256 + offset)
            .chain((0..256_u32).map(|offset| 2 * 256 + offset))
            .collect::<Vec<_>>();
        let rope = target_default_rope_inv_freq(10_000.0, TARGET_ROPE_WIDTH)?;
        let metadata = target_c128_replay_metadata(252, 8, &physical_slots, &rope, 16, 2)?;
        assert_eq!(metadata.main_slots[0], 5 * 256 + 252);
        assert_eq!(metadata.main_slots[4], 2 * 256);
        assert_eq!(metadata.compressed_slots, vec![5 * 2 + 1]);
        assert_eq!(metadata.indexed_lengths, vec![1, 1, 1, 2, 2, 2, 2, 2]);
        assert_eq!(&metadata.indexed_indices[..2], &[5 * 2, 5 * 2 + 1]);
        Ok(())
    }

    #[test]
    fn entry_hc_lane_sum_collapses_repeated_input_width() -> Result<()> {
        let hidden = 2;
        let source = (0..TARGET_HC_MIXES)
            .flat_map(|mix| {
                (0..TARGET_HC_MULT).flat_map(move |lane| {
                    (0..hidden).map(move |column| (100 * mix + 10 * lane + column) as f32)
                })
            })
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let mut destination = vec![0_u8; TARGET_HC_MIXES * hidden * 4];
        sum_repeated_hc_input_lanes(&source, &mut destination, hidden)?;
        let values = destination
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
            .collect::<Vec<_>>();
        for mix in 0..TARGET_HC_MIXES {
            for column in 0..hidden {
                assert_eq!(
                    values[mix * hidden + column],
                    400.0 * mix as f32 + 60.0 + 4.0 * column as f32
                );
            }
        }
        Ok(())
    }

    #[test]
    fn c0_replay_metadata_keeps_absolute_rope_and_physical_swa_slots() -> Result<()> {
        let physical_slots = (0..140_u32)
            .map(|slot| 10_000 + slot * 3)
            .collect::<Vec<_>>();
        let rope = target_default_rope_inv_freq(10_000.0, TARGET_ROPE_WIDTH)?;
        let metadata = target_c0_replay_metadata(128, 3, &physical_slots, &rope)?;
        assert_eq!(metadata.positions, [0, 1, 2]);
        assert_eq!(
            metadata.main_slots,
            [
                physical_slots[128],
                physical_slots[129],
                physical_slots[130]
            ]
        );
        assert_eq!(metadata.selected_lengths, [128, 128, 128]);
        assert_eq!(metadata.selected_indices[0], physical_slots[1] as i32);
        assert_eq!(metadata.selected_indices[127], physical_slots[128] as i32);
        assert_eq!(
            metadata.selected_indices[TARGET_SWA_WIDTH],
            physical_slots[2] as i32
        );
        assert_eq!(metadata.cos_sin[0], 128.0_f32.cos());
        assert_eq!(metadata.cos_sin[TARGET_ROPE_WIDTH / 2], 128.0_f32.sin());
        Ok(())
    }

    #[test]
    fn native_metadata_byte_views_match_explicit_little_endian_encoding() {
        let unsigned = [0x0123_4567_u32, 0x89ab_cdef];
        let signed = [-1_i32, 0x1020_3040];
        let floats = [0.0_f32, -1.5, f32::from_bits(0x7f7f_ffff)];
        assert_eq!(
            u32_host_bytes(&unsigned),
            unsigned
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            i32_host_bytes(&signed),
            signed
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            f32_host_bytes(&floats),
            floats
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn target_graph_rows_keep_every_joint_verify_width() {
        assert!((1..=16).all(|rows| TARGET_GRAPH_ROWS.contains(&rows)));
        assert_eq!(target_graph_row_buckets(64).last(), Some(&64));
        assert!(!target_graph_row_buckets(64).contains(&128));
    }

    #[test]
    fn target_attention_selects_compressed_yarn_rope_for_c4_and_c128() -> Result<()> {
        let main = target_default_rope_inv_freq(10_000.0, TARGET_ROPE_WIDTH)?;
        let compress =
            target_yarn_rope_inv_freq(160_000.0, TARGET_ROPE_WIDTH, 16.0, 65_536, 32.0, 1.0)?;
        assert_eq!(target_attention_rope_inv_freq(0, &main, &compress)?, &main);
        assert_eq!(
            target_attention_rope_inv_freq(4, &main, &compress)?,
            &compress
        );
        assert_eq!(
            target_attention_rope_inv_freq(128, &main, &compress)?,
            &compress
        );
        // Reference values from Transformers' DeepseekV4RotaryEmbedding.
        assert!((compress[15] - 0.003_635_538_5).abs() < 1.0e-9);
        assert!((compress[16] - 0.002_265_625).abs() < 1.0e-9);
        assert!((compress[24] - 0.000_019_531_253).abs() < 1.0e-11);
        assert!((compress[31] - 0.000_000_568_052_94).abs() < 1.0e-12);
        Ok(())
    }
}
