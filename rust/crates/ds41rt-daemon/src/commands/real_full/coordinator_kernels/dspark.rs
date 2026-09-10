use super::*;
use crate::python_graph_capture::{
    launch_python_graph_capture, query_python_usize_during_startup, PythonDeviceBufferArg,
    PythonGraphCaptureLaunch, PythonKernelArg, PythonUsizeQuery,
};
use ds41rt_core::{DType, DeepseekV4KvCacheFormat, TensorCatalog, DS4_KV_NVFP4_BYTES_PER_ROW};
use ds41rt_loader::read_tensor_bytes_into;

const DSPARK_CAPTURE_MODULE: &str = "deepseek_v4_dspark_capture";
const DSPARK_ARENA_NBYTES_FUNCTION: &str = "deepseek_v4_dspark_arena_nbytes";
const DSPARK_ARENA_REGION_OFFSET_FUNCTION: &str = "deepseek_v4_dspark_arena_region_offset";
const DSPARK_ARENA_REGION_NBYTES_FUNCTION: &str = "deepseek_v4_dspark_arena_region_nbytes";
const DSPARK_ENTRY_BUFFER_OFFSET_FUNCTION: &str = "deepseek_v4_dspark_entry_buffer_offset";
const DSPARK_ENTRY_BUFFER_NBYTES_FUNCTION: &str = "deepseek_v4_dspark_entry_buffer_nbytes";
const DSPARK_PROMPT_BUFFER_OFFSET_FUNCTION: &str = "deepseek_v4_dspark_prompt_buffer_offset";
const DSPARK_PROMPT_BUFFER_NBYTES_FUNCTION: &str = "deepseek_v4_dspark_prompt_buffer_nbytes";
const DSPARK_PROPOSAL_BUFFER_OFFSET_FUNCTION: &str = "deepseek_v4_dspark_proposal_buffer_offset";
const DSPARK_PROPOSAL_BUFFER_NBYTES_FUNCTION: &str = "deepseek_v4_dspark_proposal_buffer_nbytes";
const DSPARK_BLOCK_BUFFER_OFFSET_FUNCTION: &str = "deepseek_v4_dspark_block_buffer_offset";
const DSPARK_BLOCK_BUFFER_NBYTES_FUNCTION: &str = "deepseek_v4_dspark_block_buffer_nbytes";
const DSPARK_PERSISTENT_KV_NBYTES_FUNCTION: &str = "deepseek_v4_dspark_persistent_kv_nbytes";
const DSPARK_BLOCKS: usize = 3;
const DSPARK_FP8_PACKED_PAGE_BYTES: usize = 149_760;
const DSPARK_PROJECTED_TARGET_MAIN_REGION: &str = "projected_target_main";
const DSPARK_REUSED_BLOCK_WORKSPACE_REGION: &str = "reused_sparse_block_workspace";
const DSPARK_REUSED_KV_PRODUCER_SCRATCH_REGION: &str = "reused_target_main_kv_producer_scratch";
const DSPARK_TARGET_TAP_CONCAT_BUFFER: &str = "target_tap_concat";
const DSPARK_PROJECTION_SCRATCH_BUFFER: &str = "projection_scratch";
const DSPARK_PROMPT_POSITIONS_BUFFER: &str = "positions";
const DSPARK_PROMPT_MAIN_SLOTS_BUFFER: &str = "main_slots";
const DSPARK_PROMPT_COS_SIN_BUFFER: &str = "cos_sin_cache";
const DSPARK_PROPOSAL_TOKEN_IDS_BUFFER: &str = "draft_input_token_ids";
const DSPARK_PROPOSAL_RESIDUAL_PING_BUFFER: &str = "residual_ping";
const DSPARK_PROPOSAL_RESIDUAL_PONG_BUFFER: &str = "residual_pong";
const DSPARK_PROPOSAL_COLLAPSED_HIDDEN_BUFFER: &str = "collapsed_hidden";
const DSPARK_PROPOSAL_NORMALIZED_HIDDEN_REGION: &str = "proposal_normalized_hidden";
const DSPARK_TERMINAL_COMPACT_NORMALIZED_REGION: &str = "terminal_compact_normalized_hidden";
const DSPARK_TERMINAL_SHARED_LOGITS_REGION: &str = "shared_lm_logits";
const DSPARK_TERMINAL_MARKOV_LOGITS_REGION: &str = "terminal_markov_logits";
const DSPARK_TERMINAL_MARKOV_EMBEDDINGS_REGION: &str = "markov_embeddings";
const DSPARK_TERMINAL_CONFIDENCE_REGION: &str = "confidence";
const DSPARK_TERMINAL_ACTIVE_SLOTS_REGION: &str = "terminal_active_slot_ids";
const DSPARK_TERMINAL_ANCHORS_REGION: &str = "terminal_anchor_token_ids";
const DSPARK_TERMINAL_OUTPUT_TOKENS_REGION: &str = "terminal_output_token_ids";
const DSPARK_PROPOSAL_POSITIONS_BUFFER: &str = "positions";
const DSPARK_PROPOSAL_MAIN_SLOTS_BUFFER: &str = "main_slots";
const DSPARK_PROPOSAL_COS_SIN_BUFFER: &str = "cos_sin_cache";
const DSPARK_PROPOSAL_POST_PING_BUFFER: &str = "post_ping";
const DSPARK_PROPOSAL_COMB_PING_BUFFER: &str = "comb_ping";
const DSPARK_PROPOSAL_POST_PONG_BUFFER: &str = "post_pong";
const DSPARK_PROPOSAL_COMB_PONG_BUFFER: &str = "comb_pong";
const DSPARK_PROPOSAL_SELECTED_INDICES_BUFFER: &str = "selected_indices";
const DSPARK_PROPOSAL_SELECTED_LENGTHS_BUFFER: &str = "selected_lengths";
const DSPARK_BLOCK_ROUTE_INDICES_BUFFER: &str = "route_indices";
const DSPARK_BLOCK_ROUTE_SCORES_BUFFER: &str = "route_scores";
const DSPARK_BLOCK_ROUTE_WEIGHTS_BUFFER: &str = "route_weights";
const DSPARK_BLOCK_SHARED_GATE_BUFFER: &str = "shared_gate";
const DSPARK_BLOCK_SHARED_UP_BUFFER: &str = "shared_up";
const DSPARK_BLOCK_SHARED_ACTIVATED_BUFFER: &str = "shared_activated";
const DSPARK_BLOCK_SHARED_DELTA_BUFFER: &str = "shared_delta";
const DSPARK_BLOCK_REDUCTION_F32_BUFFER: &str = "reduction_f32";
const DSPARK_BLOCK_FFN_DELTA_BUFFER: &str = "ffn_delta";
const DSPARK_ENTRY_ALIGNMENT: usize = 1_024;
const DSPARK_MAIN_PROJ_WEIGHT: &str = "mtp.0.main_proj.weight";
const DSPARK_MAIN_PROJ_SCALE: &str = "mtp.0.main_proj.scale";
const DSPARK_MAIN_NORM_WEIGHT: &str = "mtp.0.main_norm.weight";
const DSPARK_ENTRY_PREPARE_FUNCTION: &str = "prepare_deepseek_v4_dspark_entry_projection";
const DSPARK_ENTRY_CAPTURE_FUNCTION: &str = "capture_deepseek_v4_dspark_entry_projection";
const DSPARK_PROMPT_PREPARE_FUNCTION: &str = "prepare_deepseek_v4_dspark_prompt_prime";
const DSPARK_PROMPT_CAPTURE_FUNCTION: &str = "capture_deepseek_v4_dspark_prompt_prime";
const DSPARK_PROPOSAL_ENTRY_PREPARE_FUNCTION: &str = "prepare_deepseek_v4_dspark_proposal_entry";
const DSPARK_PROPOSAL_ENTRY_CAPTURE_FUNCTION: &str = "capture_deepseek_v4_dspark_proposal_entry";
const DSPARK_BLOCK_ATTN_PREPARE_FUNCTION: &str = "prepare_deepseek_v4_dspark_block_attention";
const DSPARK_BLOCK_ATTN_CAPTURE_FUNCTION: &str = "capture_deepseek_v4_dspark_block_attention";
const DSPARK_BLOCK_POST_PREPARE_FUNCTION: &str = "prepare_deepseek_v4_dspark_block_post_dispatch";
const DSPARK_BLOCK_POST_CAPTURE_FUNCTION: &str = "capture_deepseek_v4_dspark_block_post_dispatch";
const DSPARK_TERMINAL_PREPARE_FUNCTION: &str = "prepare_deepseek_v4_dspark_terminal_collapse";
const DSPARK_TERMINAL_CAPTURE_FUNCTION: &str = "capture_deepseek_v4_dspark_terminal_collapse";
const DSPARK_ENTRY_GRAPH_ROWS: [usize; 9] = [1, 16, 32, 64, 128, 256, 512, 1_024, 2_048];
const DSPARK_KV_WIDTH: usize = 512;
const DSPARK_PROPOSAL_ROWS_PER_REQUEST: usize = 5;
const DSPARK_PROPOSAL_HC_MULT: usize = 4;
const DSPARK_PROPOSAL_SELECTION_WIDTH: usize = 133;
const DSPARK_EMBED_WEIGHT: &str = "embed.weight";
const DSPARK_BLOCK0_HC_ATTN_FN: &str = "mtp.0.hc_attn_fn";
const DSPARK_BLOCK0_HC_ATTN_FN_LANE_SUM: &str =
    "ds41rt#deepseek-v4-dspark-block0-hc-attn-fn-lane-sum";
const DSPARK_BLOCK0_HC_ATTN_SCALE: &str = "mtp.0.hc_attn_scale";
const DSPARK_BLOCK0_HC_ATTN_BASE: &str = "mtp.0.hc_attn_base";
const DSPARK_BLOCK0_ATTN_NORM: &str = "mtp.0.attn_norm.weight";

fn dspark_joint_request_range_byte_span(
    active_request_range: std::ops::Range<usize>,
    max_batch: usize,
    hidden: usize,
) -> Result<std::ops::Range<usize>> {
    anyhow::ensure!(
        active_request_range.start < active_request_range.end
            && active_request_range.end <= max_batch,
        "integrated dSpark joint expert range {:?} exceeds max batch {max_batch}",
        active_request_range,
    );
    let request_bytes = DSPARK_PROPOSAL_ROWS_PER_REQUEST
        .checked_mul(hidden)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("integrated dSpark joint expert request byte count overflow")?;
    let start = active_request_range
        .start
        .checked_mul(request_bytes)
        .context("integrated dSpark joint expert buffer offset overflow")?;
    let bytes = active_request_range
        .len()
        .checked_mul(request_bytes)
        .context("integrated dSpark joint expert buffer byte count overflow")?;
    let end = start
        .checked_add(bytes)
        .context("integrated dSpark joint expert buffer end overflow")?;
    Ok(start..end)
}
const DSPARK_TERMINAL_HC_FN: &str = "mtp.2.hc_head_fn";
const DSPARK_TERMINAL_HC_SCALE: &str = "mtp.2.hc_head_scale";
const DSPARK_TERMINAL_HC_BASE: &str = "mtp.2.hc_head_base";
const DSPARK_TERMINAL_NORM: &str = "mtp.2.norm.weight";
const DSPARK_SHARED_HEAD: &str = "head.weight";
const DSPARK_TERMINAL_MARKOV_W1: &str = "mtp.2.markov_head.markov_w1.weight";
const DSPARK_TERMINAL_MARKOV_W2: &str = "mtp.2.markov_head.markov_w2.weight";
const DSPARK_TERMINAL_CONFIDENCE: &str = "mtp.2.confidence_head.proj.weight";
const DSPARK_HC_MIXES: usize = 24;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::commands::real_full) struct DeepseekV4DsparkEntryDeviceStoragePlan {
    projected_target_main_offset: usize,
    pub(in crate::commands::real_full) projected_target_main_bytes: usize,
    block_workspace_offset: usize,
    pub(in crate::commands::real_full) block_workspace_bytes: usize,
    target_tap_concat_offset: usize,
    pub(in crate::commands::real_full) target_tap_concat_bytes: usize,
    projection_scratch_offset: usize,
    pub(in crate::commands::real_full) projection_scratch_bytes: usize,
    prompt_positions_offset: usize,
    pub(in crate::commands::real_full) prompt_positions_bytes: usize,
    prompt_main_slots_offset: usize,
    pub(in crate::commands::real_full) prompt_main_slots_bytes: usize,
    prompt_cos_sin_offset: usize,
    pub(in crate::commands::real_full) prompt_cos_sin_bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::commands::real_full) struct DeepseekV4DsparkProposalDeviceStoragePlan {
    draft_token_ids_offset: usize,
    pub(in crate::commands::real_full) draft_token_ids_bytes: usize,
    residual_ping_offset: usize,
    pub(in crate::commands::real_full) residual_ping_bytes: usize,
    residual_pong_offset: usize,
    pub(in crate::commands::real_full) residual_pong_bytes: usize,
    collapsed_hidden_offset: usize,
    pub(in crate::commands::real_full) collapsed_hidden_bytes: usize,
    normalized_hidden_offset: usize,
    pub(in crate::commands::real_full) normalized_hidden_bytes: usize,
    positions_offset: usize,
    pub(in crate::commands::real_full) positions_bytes: usize,
    main_slots_offset: usize,
    pub(in crate::commands::real_full) main_slots_bytes: usize,
    cos_sin_offset: usize,
    pub(in crate::commands::real_full) cos_sin_bytes: usize,
    post_ping_offset: usize,
    pub(in crate::commands::real_full) post_ping_bytes: usize,
    comb_ping_offset: usize,
    pub(in crate::commands::real_full) comb_ping_bytes: usize,
    post_pong_offset: usize,
    pub(in crate::commands::real_full) post_pong_bytes: usize,
    comb_pong_offset: usize,
    pub(in crate::commands::real_full) comb_pong_bytes: usize,
    selected_indices_offset: usize,
    pub(in crate::commands::real_full) selected_indices_bytes: usize,
    selected_lengths_offset: usize,
    pub(in crate::commands::real_full) selected_lengths_bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::commands::real_full) struct DeepseekV4DsparkTerminalDeviceStoragePlan {
    compact_normalized_hidden_offset: usize,
    pub(in crate::commands::real_full) compact_normalized_hidden_bytes: usize,
    shared_logits_offset: usize,
    pub(in crate::commands::real_full) shared_logits_bytes: usize,
    markov_logits_offset: usize,
    pub(in crate::commands::real_full) markov_logits_bytes: usize,
    markov_embeddings_offset: usize,
    pub(in crate::commands::real_full) markov_embeddings_bytes: usize,
    confidence_offset: usize,
    pub(in crate::commands::real_full) confidence_bytes: usize,
    active_slot_ids_offset: usize,
    pub(in crate::commands::real_full) active_slot_ids_bytes: usize,
    anchor_token_ids_offset: usize,
    pub(in crate::commands::real_full) anchor_token_ids_bytes: usize,
    output_token_ids_offset: usize,
    pub(in crate::commands::real_full) output_token_ids_bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::commands::real_full) struct DeepseekV4DsparkBlockDeviceStoragePlan {
    route_indices_offset: usize,
    route_indices_bytes: usize,
    route_scores_offset: usize,
    route_scores_bytes: usize,
    route_weights_offset: usize,
    route_weights_bytes: usize,
    shared_gate_offset: usize,
    shared_gate_bytes: usize,
    shared_up_offset: usize,
    shared_up_bytes: usize,
    shared_activated_offset: usize,
    shared_activated_bytes: usize,
    shared_delta_offset: usize,
    shared_delta_bytes: usize,
    reduction_f32_offset: usize,
    reduction_f32_bytes: usize,
    ffn_delta_offset: usize,
    ffn_delta_bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::commands::real_full) struct DeepseekV4DsparkDeviceStoragePlan {
    variant: &'static str,
    cache_format: &'static str,
    packed_page_bytes: usize,
    hidden: usize,
    vocab: usize,
    markov_rank: usize,
    pub(in crate::commands::real_full) max_batch: usize,
    pub(in crate::commands::real_full) max_main_rows: usize,
    pub(in crate::commands::real_full) arena_bytes: usize,
    pub(in crate::commands::real_full) persistent_kv_bytes: usize,
    kv_producer_scratch_offset: usize,
    kv_producer_scratch_bytes: usize,
    pub(in crate::commands::real_full) entry: DeepseekV4DsparkEntryDeviceStoragePlan,
    pub(in crate::commands::real_full) proposal: DeepseekV4DsparkProposalDeviceStoragePlan,
    pub(in crate::commands::real_full) terminal: DeepseekV4DsparkTerminalDeviceStoragePlan,
    block: DeepseekV4DsparkBlockDeviceStoragePlan,
}

pub(in crate::commands::real_full) fn plan_deepseek_v4_dspark_device_storage(
    variant: &str,
    max_batch: usize,
    max_main_rows: usize,
) -> Result<DeepseekV4DsparkDeviceStoragePlan> {
    plan_deepseek_v4_dspark_device_storage_with_cache_format(
        variant,
        max_batch,
        max_main_rows,
        DeepseekV4KvCacheFormat::Fp8Ue8m0,
    )
}

pub(in crate::commands::real_full) fn plan_deepseek_v4_dspark_device_storage_with_cache_format(
    variant: &str,
    max_batch: usize,
    max_main_rows: usize,
    cache_format: DeepseekV4KvCacheFormat,
) -> Result<DeepseekV4DsparkDeviceStoragePlan> {
    let (variant, hidden, vocab, markov_rank) = match variant {
        "flash" => ("flash", 4_096usize, 129_280usize, 256usize),
        "pro" => ("pro", 7_168usize, 129_280usize, 512usize),
        _ => anyhow::bail!(
            "DeepSeek V4 dSpark device storage variant must be flash or pro, got {variant:?}"
        ),
    };
    anyhow::ensure!(
        max_batch > 0,
        "dSpark device storage max_batch must be positive"
    );
    anyhow::ensure!(
        max_main_rows > 0,
        "dSpark device storage max_main_rows must be positive"
    );
    let arena_bytes = query_dspark_usize(
        variant,
        max_batch,
        max_main_rows,
        DSPARK_ARENA_NBYTES_FUNCTION,
    )?;
    let persistent_kwargs = [
        ("variant", PythonKernelArg::Str(variant)),
        ("max_batch", PythonKernelArg::Usize(max_batch)),
        ("max_main_rows", PythonKernelArg::Usize(max_main_rows)),
        (
            "cache_format",
            PythonKernelArg::Str(cache_format.kernel_label()),
        ),
    ];
    let persistent_kv_bytes = query_python_usize_during_startup(PythonUsizeQuery {
        module: DSPARK_CAPTURE_MODULE,
        function: DSPARK_PERSISTENT_KV_NBYTES_FUNCTION,
        kwargs: &persistent_kwargs,
    })?;
    let projected_target_main_offset = query_dspark_named_usize(
        variant,
        max_batch,
        max_main_rows,
        DSPARK_ARENA_REGION_OFFSET_FUNCTION,
        "region",
        DSPARK_PROJECTED_TARGET_MAIN_REGION,
    )?;
    let projected_target_main_bytes = query_dspark_named_usize(
        variant,
        max_batch,
        max_main_rows,
        DSPARK_ARENA_REGION_NBYTES_FUNCTION,
        "region",
        DSPARK_PROJECTED_TARGET_MAIN_REGION,
    )?;
    let block_workspace_offset = query_dspark_named_usize(
        variant,
        max_batch,
        max_main_rows,
        DSPARK_ARENA_REGION_OFFSET_FUNCTION,
        "region",
        DSPARK_REUSED_BLOCK_WORKSPACE_REGION,
    )?;
    let block_workspace_bytes = query_dspark_named_usize(
        variant,
        max_batch,
        max_main_rows,
        DSPARK_ARENA_REGION_NBYTES_FUNCTION,
        "region",
        DSPARK_REUSED_BLOCK_WORKSPACE_REGION,
    )?;
    let kv_producer_scratch_offset = query_dspark_named_usize(
        variant,
        max_batch,
        max_main_rows,
        DSPARK_ARENA_REGION_OFFSET_FUNCTION,
        "region",
        DSPARK_REUSED_KV_PRODUCER_SCRATCH_REGION,
    )?;
    let kv_producer_scratch_bytes = query_dspark_named_usize(
        variant,
        max_batch,
        max_main_rows,
        DSPARK_ARENA_REGION_NBYTES_FUNCTION,
        "region",
        DSPARK_REUSED_KV_PRODUCER_SCRATCH_REGION,
    )?;
    let target_tap_concat_offset = query_dspark_named_usize(
        variant,
        max_batch,
        max_main_rows,
        DSPARK_ENTRY_BUFFER_OFFSET_FUNCTION,
        "buffer",
        DSPARK_TARGET_TAP_CONCAT_BUFFER,
    )?;
    let target_tap_concat_bytes = query_dspark_named_usize(
        variant,
        max_batch,
        max_main_rows,
        DSPARK_ENTRY_BUFFER_NBYTES_FUNCTION,
        "buffer",
        DSPARK_TARGET_TAP_CONCAT_BUFFER,
    )?;
    let projection_scratch_offset = query_dspark_named_usize(
        variant,
        max_batch,
        max_main_rows,
        DSPARK_ENTRY_BUFFER_OFFSET_FUNCTION,
        "buffer",
        DSPARK_PROJECTION_SCRATCH_BUFFER,
    )?;
    let projection_scratch_bytes = query_dspark_named_usize(
        variant,
        max_batch,
        max_main_rows,
        DSPARK_ENTRY_BUFFER_NBYTES_FUNCTION,
        "buffer",
        DSPARK_PROJECTION_SCRATCH_BUFFER,
    )?;
    let prompt_positions_offset = query_dspark_named_usize(
        variant,
        max_batch,
        max_main_rows,
        DSPARK_PROMPT_BUFFER_OFFSET_FUNCTION,
        "buffer",
        DSPARK_PROMPT_POSITIONS_BUFFER,
    )?;
    let prompt_positions_bytes = query_dspark_named_usize(
        variant,
        max_batch,
        max_main_rows,
        DSPARK_PROMPT_BUFFER_NBYTES_FUNCTION,
        "buffer",
        DSPARK_PROMPT_POSITIONS_BUFFER,
    )?;
    let prompt_main_slots_offset = query_dspark_named_usize(
        variant,
        max_batch,
        max_main_rows,
        DSPARK_PROMPT_BUFFER_OFFSET_FUNCTION,
        "buffer",
        DSPARK_PROMPT_MAIN_SLOTS_BUFFER,
    )?;
    let prompt_main_slots_bytes = query_dspark_named_usize(
        variant,
        max_batch,
        max_main_rows,
        DSPARK_PROMPT_BUFFER_NBYTES_FUNCTION,
        "buffer",
        DSPARK_PROMPT_MAIN_SLOTS_BUFFER,
    )?;
    let prompt_cos_sin_offset = query_dspark_named_usize(
        variant,
        max_batch,
        max_main_rows,
        DSPARK_PROMPT_BUFFER_OFFSET_FUNCTION,
        "buffer",
        DSPARK_PROMPT_COS_SIN_BUFFER,
    )?;
    let prompt_cos_sin_bytes = query_dspark_named_usize(
        variant,
        max_batch,
        max_main_rows,
        DSPARK_PROMPT_BUFFER_NBYTES_FUNCTION,
        "buffer",
        DSPARK_PROMPT_COS_SIN_BUFFER,
    )?;
    let proposal_value = |function, buffer| {
        query_dspark_named_usize(
            variant,
            max_batch,
            max_main_rows,
            function,
            "buffer",
            buffer,
        )
    };
    let draft_token_ids_offset = proposal_value(
        DSPARK_PROPOSAL_BUFFER_OFFSET_FUNCTION,
        DSPARK_PROPOSAL_TOKEN_IDS_BUFFER,
    )?;
    let draft_token_ids_bytes = proposal_value(
        DSPARK_PROPOSAL_BUFFER_NBYTES_FUNCTION,
        DSPARK_PROPOSAL_TOKEN_IDS_BUFFER,
    )?;
    let residual_ping_offset = proposal_value(
        DSPARK_PROPOSAL_BUFFER_OFFSET_FUNCTION,
        DSPARK_PROPOSAL_RESIDUAL_PING_BUFFER,
    )?;
    let residual_ping_bytes = proposal_value(
        DSPARK_PROPOSAL_BUFFER_NBYTES_FUNCTION,
        DSPARK_PROPOSAL_RESIDUAL_PING_BUFFER,
    )?;
    let residual_pong_offset = proposal_value(
        DSPARK_PROPOSAL_BUFFER_OFFSET_FUNCTION,
        DSPARK_PROPOSAL_RESIDUAL_PONG_BUFFER,
    )?;
    let residual_pong_bytes = proposal_value(
        DSPARK_PROPOSAL_BUFFER_NBYTES_FUNCTION,
        DSPARK_PROPOSAL_RESIDUAL_PONG_BUFFER,
    )?;
    let collapsed_hidden_offset = proposal_value(
        DSPARK_PROPOSAL_BUFFER_OFFSET_FUNCTION,
        DSPARK_PROPOSAL_COLLAPSED_HIDDEN_BUFFER,
    )?;
    let collapsed_hidden_bytes = proposal_value(
        DSPARK_PROPOSAL_BUFFER_NBYTES_FUNCTION,
        DSPARK_PROPOSAL_COLLAPSED_HIDDEN_BUFFER,
    )?;
    let arena_region_value = |function, region| {
        query_dspark_named_usize(
            variant,
            max_batch,
            max_main_rows,
            function,
            "region",
            region,
        )
    };
    let normalized_hidden_offset = arena_region_value(
        DSPARK_ARENA_REGION_OFFSET_FUNCTION,
        DSPARK_PROPOSAL_NORMALIZED_HIDDEN_REGION,
    )?;
    let normalized_hidden_bytes = arena_region_value(
        DSPARK_ARENA_REGION_NBYTES_FUNCTION,
        DSPARK_PROPOSAL_NORMALIZED_HIDDEN_REGION,
    )?;
    let terminal_region = |region| -> Result<(usize, usize)> {
        Ok((
            arena_region_value(DSPARK_ARENA_REGION_OFFSET_FUNCTION, region)?,
            arena_region_value(DSPARK_ARENA_REGION_NBYTES_FUNCTION, region)?,
        ))
    };
    let (compact_normalized_hidden_offset, compact_normalized_hidden_bytes) =
        terminal_region(DSPARK_TERMINAL_COMPACT_NORMALIZED_REGION)?;
    let (shared_logits_offset, shared_logits_bytes) =
        terminal_region(DSPARK_TERMINAL_SHARED_LOGITS_REGION)?;
    let (markov_logits_offset, markov_logits_bytes) =
        terminal_region(DSPARK_TERMINAL_MARKOV_LOGITS_REGION)?;
    let (markov_embeddings_offset, markov_embeddings_bytes) =
        terminal_region(DSPARK_TERMINAL_MARKOV_EMBEDDINGS_REGION)?;
    let (confidence_offset, confidence_bytes) = terminal_region(DSPARK_TERMINAL_CONFIDENCE_REGION)?;
    let (active_slot_ids_offset, active_slot_ids_bytes) =
        terminal_region(DSPARK_TERMINAL_ACTIVE_SLOTS_REGION)?;
    let (anchor_token_ids_offset, anchor_token_ids_bytes) =
        terminal_region(DSPARK_TERMINAL_ANCHORS_REGION)?;
    let (output_token_ids_offset, output_token_ids_bytes) =
        terminal_region(DSPARK_TERMINAL_OUTPUT_TOKENS_REGION)?;
    let positions_offset = proposal_value(
        DSPARK_PROPOSAL_BUFFER_OFFSET_FUNCTION,
        DSPARK_PROPOSAL_POSITIONS_BUFFER,
    )?;
    let positions_bytes = proposal_value(
        DSPARK_PROPOSAL_BUFFER_NBYTES_FUNCTION,
        DSPARK_PROPOSAL_POSITIONS_BUFFER,
    )?;
    let main_slots_offset = proposal_value(
        DSPARK_PROPOSAL_BUFFER_OFFSET_FUNCTION,
        DSPARK_PROPOSAL_MAIN_SLOTS_BUFFER,
    )?;
    let main_slots_bytes = proposal_value(
        DSPARK_PROPOSAL_BUFFER_NBYTES_FUNCTION,
        DSPARK_PROPOSAL_MAIN_SLOTS_BUFFER,
    )?;
    let cos_sin_offset = proposal_value(
        DSPARK_PROPOSAL_BUFFER_OFFSET_FUNCTION,
        DSPARK_PROPOSAL_COS_SIN_BUFFER,
    )?;
    let cos_sin_bytes = proposal_value(
        DSPARK_PROPOSAL_BUFFER_NBYTES_FUNCTION,
        DSPARK_PROPOSAL_COS_SIN_BUFFER,
    )?;
    let post_ping_offset = proposal_value(
        DSPARK_PROPOSAL_BUFFER_OFFSET_FUNCTION,
        DSPARK_PROPOSAL_POST_PING_BUFFER,
    )?;
    let post_ping_bytes = proposal_value(
        DSPARK_PROPOSAL_BUFFER_NBYTES_FUNCTION,
        DSPARK_PROPOSAL_POST_PING_BUFFER,
    )?;
    let comb_ping_offset = proposal_value(
        DSPARK_PROPOSAL_BUFFER_OFFSET_FUNCTION,
        DSPARK_PROPOSAL_COMB_PING_BUFFER,
    )?;
    let comb_ping_bytes = proposal_value(
        DSPARK_PROPOSAL_BUFFER_NBYTES_FUNCTION,
        DSPARK_PROPOSAL_COMB_PING_BUFFER,
    )?;
    let post_pong_offset = proposal_value(
        DSPARK_PROPOSAL_BUFFER_OFFSET_FUNCTION,
        DSPARK_PROPOSAL_POST_PONG_BUFFER,
    )?;
    let post_pong_bytes = proposal_value(
        DSPARK_PROPOSAL_BUFFER_NBYTES_FUNCTION,
        DSPARK_PROPOSAL_POST_PONG_BUFFER,
    )?;
    let comb_pong_offset = proposal_value(
        DSPARK_PROPOSAL_BUFFER_OFFSET_FUNCTION,
        DSPARK_PROPOSAL_COMB_PONG_BUFFER,
    )?;
    let comb_pong_bytes = proposal_value(
        DSPARK_PROPOSAL_BUFFER_NBYTES_FUNCTION,
        DSPARK_PROPOSAL_COMB_PONG_BUFFER,
    )?;
    let selected_indices_offset = proposal_value(
        DSPARK_PROPOSAL_BUFFER_OFFSET_FUNCTION,
        DSPARK_PROPOSAL_SELECTED_INDICES_BUFFER,
    )?;
    let selected_indices_bytes = proposal_value(
        DSPARK_PROPOSAL_BUFFER_NBYTES_FUNCTION,
        DSPARK_PROPOSAL_SELECTED_INDICES_BUFFER,
    )?;
    let selected_lengths_offset = proposal_value(
        DSPARK_PROPOSAL_BUFFER_OFFSET_FUNCTION,
        DSPARK_PROPOSAL_SELECTED_LENGTHS_BUFFER,
    )?;
    let selected_lengths_bytes = proposal_value(
        DSPARK_PROPOSAL_BUFFER_NBYTES_FUNCTION,
        DSPARK_PROPOSAL_SELECTED_LENGTHS_BUFFER,
    )?;
    let block_value = |function, buffer| {
        query_dspark_named_usize(
            variant,
            max_batch,
            max_main_rows,
            function,
            "buffer",
            buffer,
        )
    };
    let route_indices_offset = block_value(
        DSPARK_BLOCK_BUFFER_OFFSET_FUNCTION,
        DSPARK_BLOCK_ROUTE_INDICES_BUFFER,
    )?;
    let route_indices_bytes = block_value(
        DSPARK_BLOCK_BUFFER_NBYTES_FUNCTION,
        DSPARK_BLOCK_ROUTE_INDICES_BUFFER,
    )?;
    let route_scores_offset = block_value(
        DSPARK_BLOCK_BUFFER_OFFSET_FUNCTION,
        DSPARK_BLOCK_ROUTE_SCORES_BUFFER,
    )?;
    let route_scores_bytes = block_value(
        DSPARK_BLOCK_BUFFER_NBYTES_FUNCTION,
        DSPARK_BLOCK_ROUTE_SCORES_BUFFER,
    )?;
    let route_weights_offset = block_value(
        DSPARK_BLOCK_BUFFER_OFFSET_FUNCTION,
        DSPARK_BLOCK_ROUTE_WEIGHTS_BUFFER,
    )?;
    let route_weights_bytes = block_value(
        DSPARK_BLOCK_BUFFER_NBYTES_FUNCTION,
        DSPARK_BLOCK_ROUTE_WEIGHTS_BUFFER,
    )?;
    let shared_gate_offset = block_value(
        DSPARK_BLOCK_BUFFER_OFFSET_FUNCTION,
        DSPARK_BLOCK_SHARED_GATE_BUFFER,
    )?;
    let shared_gate_bytes = block_value(
        DSPARK_BLOCK_BUFFER_NBYTES_FUNCTION,
        DSPARK_BLOCK_SHARED_GATE_BUFFER,
    )?;
    let shared_up_offset = block_value(
        DSPARK_BLOCK_BUFFER_OFFSET_FUNCTION,
        DSPARK_BLOCK_SHARED_UP_BUFFER,
    )?;
    let shared_up_bytes = block_value(
        DSPARK_BLOCK_BUFFER_NBYTES_FUNCTION,
        DSPARK_BLOCK_SHARED_UP_BUFFER,
    )?;
    let shared_activated_offset = block_value(
        DSPARK_BLOCK_BUFFER_OFFSET_FUNCTION,
        DSPARK_BLOCK_SHARED_ACTIVATED_BUFFER,
    )?;
    let shared_activated_bytes = block_value(
        DSPARK_BLOCK_BUFFER_NBYTES_FUNCTION,
        DSPARK_BLOCK_SHARED_ACTIVATED_BUFFER,
    )?;
    let shared_delta_offset = block_value(
        DSPARK_BLOCK_BUFFER_OFFSET_FUNCTION,
        DSPARK_BLOCK_SHARED_DELTA_BUFFER,
    )?;
    let shared_delta_bytes = block_value(
        DSPARK_BLOCK_BUFFER_NBYTES_FUNCTION,
        DSPARK_BLOCK_SHARED_DELTA_BUFFER,
    )?;
    let reduction_f32_offset = block_value(
        DSPARK_BLOCK_BUFFER_OFFSET_FUNCTION,
        DSPARK_BLOCK_REDUCTION_F32_BUFFER,
    )?;
    let reduction_f32_bytes = block_value(
        DSPARK_BLOCK_BUFFER_NBYTES_FUNCTION,
        DSPARK_BLOCK_REDUCTION_F32_BUFFER,
    )?;
    let ffn_delta_offset = block_value(
        DSPARK_BLOCK_BUFFER_OFFSET_FUNCTION,
        DSPARK_BLOCK_FFN_DELTA_BUFFER,
    )?;
    let ffn_delta_bytes = block_value(
        DSPARK_BLOCK_BUFFER_NBYTES_FUNCTION,
        DSPARK_BLOCK_FFN_DELTA_BUFFER,
    )?;
    let packed_page_bytes = match cache_format {
        DeepseekV4KvCacheFormat::Fp8Ue8m0 => DSPARK_FP8_PACKED_PAGE_BYTES,
        DeepseekV4KvCacheFormat::Nvfp4 => 256 * DS4_KV_NVFP4_BYTES_PER_ROW,
    };
    let expected_kv_bytes = DSPARK_BLOCKS
        .checked_mul(max_batch)
        .and_then(|pages| pages.checked_mul(packed_page_bytes))
        .context("DeepSeek V4 dSpark persistent KV byte count overflow")?;
    anyhow::ensure!(
        persistent_kv_bytes == expected_kv_bytes,
        "DeepSeek V4 dSpark Python plan returned {persistent_kv_bytes} persistent KV bytes, expected {expected_kv_bytes}"
    );
    anyhow::ensure!(arena_bytes > 0, "DeepSeek V4 dSpark arena plan is empty");
    let expected_projected_bytes = max_main_rows
        .checked_mul(hidden)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("DeepSeek V4 dSpark projected-main byte count overflow")?;
    let expected_concat_bytes = expected_projected_bytes
        .checked_mul(3)
        .context("DeepSeek V4 dSpark target-tap concat byte count overflow")?;
    anyhow::ensure!(
        projected_target_main_bytes == expected_projected_bytes,
        "DeepSeek V4 dSpark projected-main region has {projected_target_main_bytes} bytes, expected {expected_projected_bytes}"
    );
    anyhow::ensure!(
        target_tap_concat_bytes == expected_concat_bytes,
        "DeepSeek V4 dSpark target-tap concat has {target_tap_concat_bytes} bytes, expected {expected_concat_bytes}"
    );
    anyhow::ensure!(
        target_tap_concat_offset % DSPARK_ENTRY_ALIGNMENT == 0
            && projection_scratch_offset % DSPARK_ENTRY_ALIGNMENT == 0,
        "DeepSeek V4 dSpark entry buffers lost {DSPARK_ENTRY_ALIGNMENT}-byte alignment"
    );
    let concat_end = target_tap_concat_offset
        .checked_add(target_tap_concat_bytes)
        .context("DeepSeek V4 dSpark target-tap concat range overflow")?;
    let projection_end = projection_scratch_offset
        .checked_add(projection_scratch_bytes)
        .context("DeepSeek V4 dSpark projection scratch range overflow")?;
    anyhow::ensure!(
        projection_scratch_bytes > 0
            && projection_scratch_offset >= concat_end
            && projection_end <= block_workspace_bytes,
        "DeepSeek V4 dSpark entry scratch does not fit the reused block workspace"
    );
    let expected_prompt_positions_bytes = max_main_rows
        .checked_mul(std::mem::size_of::<u32>())
        .context("DeepSeek V4 dSpark prompt-position byte count overflow")?;
    let expected_prompt_cos_sin_bytes = max_main_rows
        .checked_mul(64)
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .context("DeepSeek V4 dSpark prompt cos/sin byte count overflow")?;
    let prompt_cos_sin_end = prompt_cos_sin_offset
        .checked_add(prompt_cos_sin_bytes)
        .context("DeepSeek V4 dSpark prompt metadata range overflow")?;
    anyhow::ensure!(
        prompt_positions_bytes == expected_prompt_positions_bytes
            && prompt_main_slots_bytes == expected_prompt_positions_bytes
            && prompt_cos_sin_bytes == expected_prompt_cos_sin_bytes
            && prompt_positions_offset % DSPARK_ENTRY_ALIGNMENT == 0
            && prompt_main_slots_offset % DSPARK_ENTRY_ALIGNMENT == 0
            && prompt_cos_sin_offset % DSPARK_ENTRY_ALIGNMENT == 0
            && prompt_cos_sin_end <= target_tap_concat_bytes,
        "DeepSeek V4 dSpark prompt metadata lost its target-concat reuse contract"
    );
    anyhow::ensure!(
        projected_target_main_offset
            .checked_add(projected_target_main_bytes)
            .is_some_and(|end| end <= arena_bytes)
            && block_workspace_offset
                .checked_add(block_workspace_bytes)
                .is_some_and(|end| end <= arena_bytes)
            && kv_producer_scratch_bytes > 0
            && kv_producer_scratch_offset
                .checked_add(kv_producer_scratch_bytes)
                .is_some_and(|end| end <= arena_bytes),
        "DeepSeek V4 dSpark entry regions exceed the startup arena"
    );
    let proposal_rows = max_batch
        .checked_mul(DSPARK_PROPOSAL_ROWS_PER_REQUEST)
        .context("DeepSeek V4 dSpark proposal row count overflow")?;
    let expected_u32_rows = proposal_rows
        .checked_mul(std::mem::size_of::<u32>())
        .context("DeepSeek V4 dSpark proposal metadata bytes overflow")?;
    let expected_residual_bytes = proposal_rows
        .checked_mul(DSPARK_PROPOSAL_HC_MULT)
        .and_then(|values| values.checked_mul(hidden))
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("DeepSeek V4 dSpark proposal residual bytes overflow")?;
    let expected_collapsed_hidden_bytes = proposal_rows
        .checked_mul(hidden)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("DeepSeek V4 dSpark proposal collapsed-hidden bytes overflow")?;
    let expected_shared_logits_bytes = proposal_rows
        .checked_mul(vocab)
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .context("DeepSeek V4 dSpark shared-logits bytes overflow")?;
    let expected_markov_logits_bytes = max_batch
        .checked_mul(vocab)
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .context("DeepSeek V4 dSpark Markov-logits bytes overflow")?;
    let expected_markov_embeddings_bytes = proposal_rows
        .checked_mul(markov_rank)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("DeepSeek V4 dSpark Markov-embedding bytes overflow")?;
    let expected_output_token_bytes = max_batch
        .checked_mul(DSPARK_PROPOSAL_ROWS_PER_REQUEST + 1)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u32>()))
        .context("DeepSeek V4 dSpark terminal output-token bytes overflow")?;
    let expected_post_bytes = proposal_rows
        .checked_mul(DSPARK_PROPOSAL_HC_MULT)
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .context("DeepSeek V4 dSpark proposal post-mix bytes overflow")?;
    let expected_comb_bytes = proposal_rows
        .checked_mul(DSPARK_PROPOSAL_HC_MULT)
        .and_then(|values| values.checked_mul(DSPARK_PROPOSAL_HC_MULT))
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .context("DeepSeek V4 dSpark proposal combination-mix bytes overflow")?;
    let expected_cos_sin_bytes = proposal_rows
        .checked_mul(64)
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .context("DeepSeek V4 dSpark proposal cos/sin bytes overflow")?;
    let expected_selected_indices_bytes = proposal_rows
        .checked_mul(DSPARK_PROPOSAL_SELECTION_WIDTH)
        .and_then(|values| values.checked_mul(std::mem::size_of::<i32>()))
        .context("DeepSeek V4 dSpark proposal selection bytes overflow")?;
    anyhow::ensure!(
        draft_token_ids_bytes == expected_u32_rows
            && positions_bytes == expected_u32_rows
            && main_slots_bytes == expected_u32_rows
            && selected_lengths_bytes == expected_u32_rows
            && residual_ping_bytes == expected_residual_bytes
            && residual_pong_bytes == expected_residual_bytes
            && collapsed_hidden_bytes == expected_collapsed_hidden_bytes
            && normalized_hidden_bytes == expected_collapsed_hidden_bytes
            && cos_sin_bytes == expected_cos_sin_bytes
            && post_ping_bytes == expected_post_bytes
            && post_pong_bytes == expected_post_bytes
            && comb_ping_bytes == expected_comb_bytes
            && comb_pong_bytes == expected_comb_bytes
            && selected_indices_bytes == expected_selected_indices_bytes,
        "DeepSeek V4 dSpark proposal buffer sizes disagree with the fixed five-row native ABI"
    );
    anyhow::ensure!(
        compact_normalized_hidden_bytes == expected_collapsed_hidden_bytes
            && shared_logits_bytes == expected_shared_logits_bytes
            && markov_logits_bytes == expected_markov_logits_bytes
            && markov_embeddings_bytes == expected_markov_embeddings_bytes
            && confidence_bytes == expected_u32_rows
            && active_slot_ids_bytes == max_batch * std::mem::size_of::<u32>()
            && anchor_token_ids_bytes == max_batch * std::mem::size_of::<u32>()
            && output_token_ids_bytes == expected_output_token_bytes,
        "DeepSeek V4 dSpark terminal buffers disagree with the joint batch ABI"
    );
    let expected_route_bytes = proposal_rows
        .checked_mul(6)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u32>()))
        .context("DeepSeek V4 dSpark route buffer bytes overflow")?;
    let routed_experts = if variant == "flash" { 256 } else { 384 };
    let expected_route_score_bytes = proposal_rows
        .checked_mul(6 + routed_experts)
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .context("DeepSeek V4 dSpark route score workspace bytes overflow")?;
    let shared_intermediate = if variant == "flash" { 2_048 } else { 3_072 };
    let expected_shared_workspace_bytes = 8usize
        .checked_mul(shared_intermediate)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("DeepSeek V4 dSpark shared workspace bytes overflow")?;
    anyhow::ensure!(
        route_indices_bytes == expected_route_bytes
            && route_scores_bytes == expected_route_score_bytes
            && route_weights_bytes == expected_route_bytes
            && shared_gate_bytes == expected_shared_workspace_bytes
            && shared_up_bytes == expected_shared_workspace_bytes
            && shared_activated_bytes == expected_shared_workspace_bytes
            && shared_delta_bytes == expected_collapsed_hidden_bytes
            && reduction_f32_bytes == expected_collapsed_hidden_bytes * 2
            && ffn_delta_bytes == expected_collapsed_hidden_bytes,
        "DeepSeek V4 dSpark sparse workspace disagrees with its global-route/shared-expert ABI"
    );
    for (name, offset, bytes) in [
        ("route indices", route_indices_offset, route_indices_bytes),
        ("route scores", route_scores_offset, route_scores_bytes),
        ("route weights", route_weights_offset, route_weights_bytes),
        ("shared gate", shared_gate_offset, shared_gate_bytes),
        ("shared up", shared_up_offset, shared_up_bytes),
        (
            "shared activated",
            shared_activated_offset,
            shared_activated_bytes,
        ),
        ("shared delta", shared_delta_offset, shared_delta_bytes),
        ("reduction f32", reduction_f32_offset, reduction_f32_bytes),
        ("FFN delta", ffn_delta_offset, ffn_delta_bytes),
    ] {
        anyhow::ensure!(
            offset % DSPARK_ENTRY_ALIGNMENT == 0
                && offset
                    .checked_add(bytes)
                    .is_some_and(|end| end <= block_workspace_bytes),
            "DeepSeek V4 dSpark block {name} view exceeds or loses alignment in the reused workspace"
        );
    }
    for (name, offset, bytes) in [
        (
            "draft token ids",
            draft_token_ids_offset,
            draft_token_ids_bytes,
        ),
        ("residual ping", residual_ping_offset, residual_ping_bytes),
        ("residual pong", residual_pong_offset, residual_pong_bytes),
        (
            "collapsed hidden",
            collapsed_hidden_offset,
            collapsed_hidden_bytes,
        ),
        (
            "normalized hidden",
            normalized_hidden_offset,
            normalized_hidden_bytes,
        ),
        ("positions", positions_offset, positions_bytes),
        ("main slots", main_slots_offset, main_slots_bytes),
        ("cos/sin", cos_sin_offset, cos_sin_bytes),
        ("post ping", post_ping_offset, post_ping_bytes),
        ("comb ping", comb_ping_offset, comb_ping_bytes),
        ("post pong", post_pong_offset, post_pong_bytes),
        ("comb pong", comb_pong_offset, comb_pong_bytes),
        (
            "selected indices",
            selected_indices_offset,
            selected_indices_bytes,
        ),
        (
            "selected lengths",
            selected_lengths_offset,
            selected_lengths_bytes,
        ),
    ] {
        anyhow::ensure!(
            offset % 16 == 0
                && offset
                    .checked_add(bytes)
                    .is_some_and(|end| end <= arena_bytes),
            "DeepSeek V4 dSpark proposal {name} view exceeds or loses alignment in the startup arena"
        );
    }
    for (name, offset, bytes) in [
        (
            "compact normalized hidden",
            compact_normalized_hidden_offset,
            compact_normalized_hidden_bytes,
        ),
        ("shared logits", shared_logits_offset, shared_logits_bytes),
        ("Markov logits", markov_logits_offset, markov_logits_bytes),
        (
            "Markov embeddings",
            markov_embeddings_offset,
            markov_embeddings_bytes,
        ),
        ("confidence", confidence_offset, confidence_bytes),
        (
            "active slot IDs",
            active_slot_ids_offset,
            active_slot_ids_bytes,
        ),
        (
            "anchor token IDs",
            anchor_token_ids_offset,
            anchor_token_ids_bytes,
        ),
        (
            "output token IDs",
            output_token_ids_offset,
            output_token_ids_bytes,
        ),
    ] {
        anyhow::ensure!(
            offset % 16 == 0
                && offset
                    .checked_add(bytes)
                    .is_some_and(|end| end <= arena_bytes),
            "DeepSeek V4 dSpark terminal {name} view exceeds or loses alignment in the startup arena"
        );
    }
    let entry = DeepseekV4DsparkEntryDeviceStoragePlan {
        projected_target_main_offset,
        projected_target_main_bytes,
        block_workspace_offset,
        block_workspace_bytes,
        target_tap_concat_offset,
        target_tap_concat_bytes,
        projection_scratch_offset,
        projection_scratch_bytes,
        prompt_positions_offset,
        prompt_positions_bytes,
        prompt_main_slots_offset,
        prompt_main_slots_bytes,
        prompt_cos_sin_offset,
        prompt_cos_sin_bytes,
    };
    let proposal = DeepseekV4DsparkProposalDeviceStoragePlan {
        draft_token_ids_offset,
        draft_token_ids_bytes,
        residual_ping_offset,
        residual_ping_bytes,
        residual_pong_offset,
        residual_pong_bytes,
        collapsed_hidden_offset,
        collapsed_hidden_bytes,
        normalized_hidden_offset,
        normalized_hidden_bytes,
        positions_offset,
        positions_bytes,
        main_slots_offset,
        main_slots_bytes,
        cos_sin_offset,
        cos_sin_bytes,
        post_ping_offset,
        post_ping_bytes,
        comb_ping_offset,
        comb_ping_bytes,
        post_pong_offset,
        post_pong_bytes,
        comb_pong_offset,
        comb_pong_bytes,
        selected_indices_offset,
        selected_indices_bytes,
        selected_lengths_offset,
        selected_lengths_bytes,
    };
    let terminal = DeepseekV4DsparkTerminalDeviceStoragePlan {
        compact_normalized_hidden_offset,
        compact_normalized_hidden_bytes,
        shared_logits_offset,
        shared_logits_bytes,
        markov_logits_offset,
        markov_logits_bytes,
        markov_embeddings_offset,
        markov_embeddings_bytes,
        confidence_offset,
        confidence_bytes,
        active_slot_ids_offset,
        active_slot_ids_bytes,
        anchor_token_ids_offset,
        anchor_token_ids_bytes,
        output_token_ids_offset,
        output_token_ids_bytes,
    };
    let block = DeepseekV4DsparkBlockDeviceStoragePlan {
        route_indices_offset,
        route_indices_bytes,
        route_scores_offset,
        route_scores_bytes,
        route_weights_offset,
        route_weights_bytes,
        shared_gate_offset,
        shared_gate_bytes,
        shared_up_offset,
        shared_up_bytes,
        shared_activated_offset,
        shared_activated_bytes,
        shared_delta_offset,
        shared_delta_bytes,
        reduction_f32_offset,
        reduction_f32_bytes,
        ffn_delta_offset,
        ffn_delta_bytes,
    };
    Ok(DeepseekV4DsparkDeviceStoragePlan {
        variant,
        cache_format: cache_format.kernel_label(),
        packed_page_bytes,
        hidden,
        vocab,
        markov_rank,
        max_batch,
        max_main_rows,
        arena_bytes,
        persistent_kv_bytes,
        kv_producer_scratch_offset,
        kv_producer_scratch_bytes,
        entry,
        proposal,
        terminal,
        block,
    })
}

fn query_dspark_usize(
    variant: &str,
    max_batch: usize,
    max_main_rows: usize,
    function: &str,
) -> Result<usize> {
    let kwargs = [
        ("variant", PythonKernelArg::Str(variant)),
        ("max_batch", PythonKernelArg::Usize(max_batch)),
        ("max_main_rows", PythonKernelArg::Usize(max_main_rows)),
    ];
    query_python_usize_during_startup(PythonUsizeQuery {
        module: DSPARK_CAPTURE_MODULE,
        function,
        kwargs: &kwargs,
    })
}

fn query_dspark_named_usize(
    variant: &str,
    max_batch: usize,
    max_main_rows: usize,
    function: &str,
    name: &str,
    value: &str,
) -> Result<usize> {
    let kwargs = [
        ("variant", PythonKernelArg::Str(variant)),
        ("max_batch", PythonKernelArg::Usize(max_batch)),
        ("max_main_rows", PythonKernelArg::Usize(max_main_rows)),
        (name, PythonKernelArg::Str(value)),
    ];
    query_python_usize_during_startup(PythonUsizeQuery {
        module: DSPARK_CAPTURE_MODULE,
        function,
        kwargs: &kwargs,
    })
}

#[derive(Clone, Copy, Debug)]
pub(in crate::commands::real_full) struct DeepseekV4DsparkEntryDeviceBuffers {
    pub(in crate::commands::real_full) projected_target_main: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) block_workspace: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) target_tap_concat: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) projection_scratch: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) prompt_positions: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) prompt_main_slots: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) prompt_cos_sin: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) kv_producer_scratch: Ds41rtDeviceBuffer,
}

#[derive(Clone, Copy, Debug)]
pub(in crate::commands::real_full) struct DeepseekV4DsparkProposalDeviceBuffers {
    pub(in crate::commands::real_full) draft_token_ids: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) residual_ping: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) residual_pong: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) collapsed_hidden: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) normalized_hidden: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) positions: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) main_slots: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) cos_sin: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) post_ping: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) comb_ping: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) post_pong: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) comb_pong: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) selected_indices: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) selected_lengths: Ds41rtDeviceBuffer,
}

#[derive(Clone, Copy, Debug)]
pub(in crate::commands::real_full) struct DeepseekV4DsparkTerminalDeviceBuffers {
    pub(in crate::commands::real_full) compact_normalized_hidden: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) shared_logits: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) markov_logits: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) markov_embeddings: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) confidence: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) active_slot_ids: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) anchor_token_ids: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) output_token_ids: Ds41rtDeviceBuffer,
}

#[derive(Clone, Copy, Debug)]
pub(in crate::commands::real_full) struct DeepseekV4DsparkBlockPreDispatchBuffers {
    pub(in crate::commands::real_full) dispatch_hidden: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) sparse: Ds4FlashSparseBlockPreDispatchDeviceBuffers,
    pub(in crate::commands::real_full) ffn_delta: Ds41rtDeviceBuffer,
}

/// Host routing envelope plus stable GPU0 reduction pointers for one native
/// five-row dSpark block. The checkpoint's `mtp.N` prefix never crosses this
/// boundary: transport sees a normal strict-TP4 sparse layer with one global
/// route image replicated to all four ranks.
pub(in crate::commands::real_full) struct DeepseekV4DsparkTp4DispatchInput {
    pub(in crate::commands::real_full) rows: usize,
    pub(in crate::commands::real_full) hidden_bf16: Vec<u8>,
    pub(in crate::commands::real_full) route_indices: Vec<u32>,
    pub(in crate::commands::real_full) route_weights: Vec<f32>,
    pub(in crate::commands::real_full) shared_delta: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) ffn_delta: Ds41rtDeviceBuffer,
}

struct DeepseekV4DsparkEntryGraph {
    rows: usize,
    graph: CoordinatorCudaCapturedGraph,
}

struct DeepseekV4DsparkPromptPrimeGraph {
    block_index: usize,
    rows: usize,
    graph: CoordinatorCudaCapturedGraph,
}

struct DeepseekV4DsparkProposalEntryGraph {
    request_slot: usize,
    graph: CoordinatorCudaCapturedGraph,
}

struct DeepseekV4DsparkBlockAttentionGraph {
    block_index: usize,
    request_slot: usize,
    graph: CoordinatorCudaCapturedGraph,
}

struct DeepseekV4DsparkBlockPostDispatchGraph {
    block_index: usize,
    request_slot: usize,
    graph: CoordinatorCudaCapturedGraph,
}

struct DeepseekV4DsparkTerminalCollapseGraph {
    request_slot: usize,
    graph: CoordinatorCudaCapturedGraph,
}

struct DeepseekV4DsparkTerminalHeadGraph {
    request_bucket: usize,
    graph: CoordinatorCudaCapturedGraph,
}

#[derive(Clone, Copy, Debug)]
struct DeepseekV4DsparkBlockAttentionWeights {
    wq_a_weight: Ds41rtDeviceBuffer,
    wq_a_scale: Ds41rtDeviceBuffer,
    wq_b_weight: Ds41rtDeviceBuffer,
    wq_b_scale: Ds41rtDeviceBuffer,
    wkv_weight: Ds41rtDeviceBuffer,
    wkv_scale: Ds41rtDeviceBuffer,
    q_norm_weight: Ds41rtDeviceBuffer,
    kv_norm_weight: Ds41rtDeviceBuffer,
    wo_a_weight: Ds41rtDeviceBuffer,
    wo_a_scale: Ds41rtDeviceBuffer,
    wo_b_weight: Ds41rtDeviceBuffer,
    wo_b_scale: Ds41rtDeviceBuffer,
    attn_sink: Ds41rtDeviceBuffer,
    hc_fn: Ds41rtDeviceBuffer,
    hc_scale: Ds41rtDeviceBuffer,
    hc_base: Ds41rtDeviceBuffer,
    norm_weight: Ds41rtDeviceBuffer,
}

#[derive(Clone, Copy, Debug)]
struct DeepseekV4DsparkBlockPostDispatchWeights {
    next_hc_fn: Ds41rtDeviceBuffer,
    next_hc_scale: Ds41rtDeviceBuffer,
    next_hc_base: Ds41rtDeviceBuffer,
    next_norm_weight: Ds41rtDeviceBuffer,
}

#[derive(Clone, Copy, Debug)]
struct DeepseekV4DsparkTerminalCollapseWeights {
    hc_fn: Ds41rtDeviceBuffer,
    hc_scale: Ds41rtDeviceBuffer,
    hc_base: Ds41rtDeviceBuffer,
    norm_weight: Ds41rtDeviceBuffer,
}

#[derive(Clone, Copy, Debug)]
struct DeepseekV4DsparkTerminalHeadWeights {
    shared_head: Ds41rtDeviceBuffer,
    markov_w1: Ds41rtDeviceBuffer,
    markov_w2: Ds41rtDeviceBuffer,
    confidence: Ds41rtDeviceBuffer,
}

#[derive(Debug)]
pub(in crate::commands::real_full) struct DeepseekV4DsparkTerminalBatchOutput {
    pub(in crate::commands::real_full) active_requests: usize,
    /// Request-major five-token proposal rows.
    pub(in crate::commands::real_full) proposal_token_ids: Vec<u32>,
    /// Request-major five-token conditional probabilities.
    pub(in crate::commands::real_full) conditional_confidence: Vec<f32>,
}

#[derive(Clone, Copy, Debug)]
pub(in crate::commands::real_full) struct DeepseekV4DsparkProjectedMainChunk {
    pub(in crate::commands::real_full) target_row_start: usize,
    pub(in crate::commands::real_full) rows: usize,
    pub(in crate::commands::real_full) graph_rows: usize,
    pub(in crate::commands::real_full) projected_target_main: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) stream: *mut c_void,
}

fn dspark_terminal_request_buckets(max_batch: usize) -> Vec<usize> {
    let mut buckets = Vec::new();
    let mut width = 1usize;
    while width < max_batch {
        buckets.push(width);
        width = width.saturating_mul(2);
        if width == usize::MAX {
            break;
        }
    }
    if buckets.last().copied() != Some(max_batch) {
        buckets.push(max_batch);
    }
    buckets
}

/// Stable GPU0 pointers for integrated dSpark replay. The scheduler wraps this
/// owner in a mutex because one arena is deliberately reused across all three
/// blocks and low-concurrency production requests.
pub(in crate::commands::real_full) struct DeepseekV4DsparkDeviceStorage {
    library: &'static NativeLibrary,
    plan: DeepseekV4DsparkDeviceStoragePlan,
    stream: CoordinatorCudaStream,
    arena: Ds41rtDeviceBuffer,
    persistent_kv: Ds41rtDeviceBuffer,
    entry_graphs: Vec<DeepseekV4DsparkEntryGraph>,
    prompt_prime_graphs: Vec<DeepseekV4DsparkPromptPrimeGraph>,
    proposal_entry_graphs: Vec<DeepseekV4DsparkProposalEntryGraph>,
    block_attention_graphs: Vec<DeepseekV4DsparkBlockAttentionGraph>,
    block_post_dispatch_graphs: Vec<DeepseekV4DsparkBlockPostDispatchGraph>,
    terminal_collapse_graphs: Vec<DeepseekV4DsparkTerminalCollapseGraph>,
    terminal_head_graphs: Vec<DeepseekV4DsparkTerminalHeadGraph>,
    terminal_active_slots_host: Vec<u32>,
    terminal_anchors_host: Vec<u32>,
    terminal_output_tokens_host: Vec<u32>,
    terminal_confidence_host: Vec<f32>,
    block_sparse_weights: Vec<Ds4FlashSparseLayerResidentWeights>,
}

unsafe impl Send for DeepseekV4DsparkDeviceStorage {}

impl DeepseekV4DsparkDeviceStorage {
    pub(in crate::commands::real_full) fn new(
        plan: DeepseekV4DsparkDeviceStoragePlan,
    ) -> Result<Self> {
        let library = cuda_native_library()?;
        let stream = CoordinatorCudaStream::create(library)
            .context("creating integrated dSpark coordinator stream")?;
        let arena = library
            .alloc_device_buffer(plan.arena_bytes)
            .context("allocating integrated dSpark execution arena")?;
        let persistent_kv = match library.alloc_device_buffer(plan.persistent_kv_bytes) {
            Ok(buffer) => buffer,
            Err(error) => {
                let mut arena = arena;
                let _ = library.free_device_buffer(&mut arena);
                return Err(error).context("allocating integrated dSpark packed KV pages");
            }
        };
        if arena.ptr.is_null()
            || persistent_kv.ptr.is_null()
            || arena.bytes < plan.arena_bytes
            || persistent_kv.bytes < plan.persistent_kv_bytes
            || arena.device_id != 0
            || persistent_kv.device_id != 0
        {
            let details = format!(
                "arena_ptr={:?} arena_bytes={} arena_device={} kv_ptr={:?} kv_bytes={} kv_device={}",
                arena.ptr,
                arena.bytes,
                arena.device_id,
                persistent_kv.ptr,
                persistent_kv.bytes,
                persistent_kv.device_id,
            );
            let mut arena = arena;
            let mut persistent_kv = persistent_kv;
            let _ = library.free_device_buffer(&mut persistent_kv);
            let _ = library.free_device_buffer(&mut arena);
            anyhow::bail!("integrated dSpark storage lost its GPU0 allocation contract: {details}");
        }
        Ok(Self {
            library,
            plan,
            stream,
            arena,
            persistent_kv,
            entry_graphs: Vec::new(),
            prompt_prime_graphs: Vec::new(),
            proposal_entry_graphs: Vec::new(),
            block_attention_graphs: Vec::new(),
            block_post_dispatch_graphs: Vec::new(),
            terminal_collapse_graphs: Vec::new(),
            terminal_head_graphs: Vec::new(),
            terminal_active_slots_host: vec![0; plan.max_batch],
            terminal_anchors_host: vec![0; plan.max_batch],
            terminal_output_tokens_host: vec![
                0;
                plan.max_batch
                    * (DSPARK_PROPOSAL_ROWS_PER_REQUEST + 1)
            ],
            terminal_confidence_host: vec![0.0; plan.max_batch * DSPARK_PROPOSAL_ROWS_PER_REQUEST],
            block_sparse_weights: Vec::new(),
        })
    }

    pub(in crate::commands::real_full) fn plan(&self) -> DeepseekV4DsparkDeviceStoragePlan {
        self.plan
    }

    pub(in crate::commands::real_full) fn arena(&self) -> Ds41rtDeviceBuffer {
        self.arena
    }

    pub(in crate::commands::real_full) fn persistent_kv(&self) -> Ds41rtDeviceBuffer {
        self.persistent_kv
    }

    fn persistent_kv_request_page(
        &self,
        block_index: usize,
        request_slot: usize,
    ) -> Result<Ds41rtDeviceBuffer> {
        anyhow::ensure!(
            request_slot < self.plan.max_batch,
            "integrated dSpark request slot {request_slot} exceeds batch {}",
            self.plan.max_batch,
        );
        let page_offset = request_slot
            .checked_mul(self.plan.packed_page_bytes)
            .context("integrated dSpark request-page offset overflow")?;
        device_buffer_byte_view(
            self.persistent_kv_block(block_index)?,
            page_offset,
            self.plan.packed_page_bytes,
            "integrated dSpark request-local packed KV page",
        )
    }

    pub(in crate::commands::real_full) fn snapshot_request_kv(
        &self,
        request_slot: usize,
    ) -> Result<Vec<u8>> {
        self.stream
            .synchronize()
            .context("synchronizing integrated dSpark KV before snapshot")?;
        let bytes = DSPARK_BLOCKS
            .checked_mul(self.plan.packed_page_bytes)
            .context("integrated dSpark request snapshot byte count overflow")?;
        let mut snapshot = vec![0_u8; bytes];
        for block_index in 0..DSPARK_BLOCKS {
            let offset = block_index * self.plan.packed_page_bytes;
            self.library
                .copy_d2h(
                    &mut snapshot[offset..offset + self.plan.packed_page_bytes],
                    self.persistent_kv_request_page(block_index, request_slot)?,
                )
                .with_context(|| {
                    format!(
                        "copying integrated dSpark block {block_index} request slot {request_slot} KV to host"
                    )
                })?;
        }
        Ok(snapshot)
    }

    pub(in crate::commands::real_full) fn restore_request_kv(
        &self,
        request_slot: usize,
        snapshot: &[u8],
    ) -> Result<()> {
        let expected_bytes = DSPARK_BLOCKS
            .checked_mul(self.plan.packed_page_bytes)
            .context("integrated dSpark request restore byte count overflow")?;
        anyhow::ensure!(
            snapshot.len() == expected_bytes,
            "integrated dSpark request snapshot has {} bytes, expected {expected_bytes}",
            snapshot.len(),
        );
        self.stream
            .synchronize()
            .context("synchronizing integrated dSpark KV before restore")?;
        for block_index in 0..DSPARK_BLOCKS {
            let offset = block_index * self.plan.packed_page_bytes;
            self.library
                .copy_h2d(
                    self.persistent_kv_request_page(block_index, request_slot)?,
                    &snapshot[offset..offset + self.plan.packed_page_bytes],
                )
                .with_context(|| {
                    format!(
                        "restoring integrated dSpark block {block_index} request slot {request_slot} KV from host"
                    )
                })?;
        }
        Ok(())
    }

    pub(in crate::commands::real_full) fn entry_buffers(
        &self,
    ) -> Result<DeepseekV4DsparkEntryDeviceBuffers> {
        let plan = self.plan.entry;
        let projected_target_main = device_buffer_byte_view(
            self.arena,
            plan.projected_target_main_offset,
            plan.projected_target_main_bytes,
            "integrated dSpark projected target main",
        )?;
        let block_workspace = device_buffer_byte_view(
            self.arena,
            plan.block_workspace_offset,
            plan.block_workspace_bytes,
            "integrated dSpark reused block workspace",
        )?;
        let target_tap_concat = device_buffer_byte_view(
            block_workspace,
            plan.target_tap_concat_offset,
            plan.target_tap_concat_bytes,
            "integrated dSpark target-tap concat",
        )?;
        let projection_scratch = device_buffer_byte_view(
            block_workspace,
            plan.projection_scratch_offset,
            plan.projection_scratch_bytes,
            "integrated dSpark entry projection scratch",
        )?;
        let prompt_positions = device_buffer_byte_view(
            target_tap_concat,
            plan.prompt_positions_offset,
            plan.prompt_positions_bytes,
            "integrated dSpark prompt positions",
        )?;
        let prompt_main_slots = device_buffer_byte_view(
            target_tap_concat,
            plan.prompt_main_slots_offset,
            plan.prompt_main_slots_bytes,
            "integrated dSpark prompt main slots",
        )?;
        let prompt_cos_sin = device_buffer_byte_view(
            target_tap_concat,
            plan.prompt_cos_sin_offset,
            plan.prompt_cos_sin_bytes,
            "integrated dSpark prompt local cos/sin cache",
        )?;
        let kv_producer_scratch = device_buffer_byte_view(
            self.arena,
            self.plan.kv_producer_scratch_offset,
            self.plan.kv_producer_scratch_bytes,
            "integrated dSpark target-main KV producer scratch",
        )?;
        Ok(DeepseekV4DsparkEntryDeviceBuffers {
            projected_target_main,
            block_workspace,
            target_tap_concat,
            projection_scratch,
            prompt_positions,
            prompt_main_slots,
            prompt_cos_sin,
            kv_producer_scratch,
        })
    }

    pub(in crate::commands::real_full) fn proposal_buffers(
        &self,
    ) -> Result<DeepseekV4DsparkProposalDeviceBuffers> {
        let plan = self.plan.proposal;
        Ok(DeepseekV4DsparkProposalDeviceBuffers {
            draft_token_ids: device_buffer_byte_view(
                self.arena,
                plan.draft_token_ids_offset,
                plan.draft_token_ids_bytes,
                "integrated dSpark proposal token ids",
            )?,
            residual_ping: device_buffer_byte_view(
                self.arena,
                plan.residual_ping_offset,
                plan.residual_ping_bytes,
                "integrated dSpark proposal residual ping",
            )?,
            residual_pong: device_buffer_byte_view(
                self.arena,
                plan.residual_pong_offset,
                plan.residual_pong_bytes,
                "integrated dSpark proposal residual pong",
            )?,
            collapsed_hidden: device_buffer_byte_view(
                self.arena,
                plan.collapsed_hidden_offset,
                plan.collapsed_hidden_bytes,
                "integrated dSpark proposal collapsed hidden",
            )?,
            normalized_hidden: device_buffer_byte_view(
                self.arena,
                plan.normalized_hidden_offset,
                plan.normalized_hidden_bytes,
                "integrated dSpark proposal normalized hidden",
            )?,
            positions: device_buffer_byte_view(
                self.arena,
                plan.positions_offset,
                plan.positions_bytes,
                "integrated dSpark proposal local positions",
            )?,
            main_slots: device_buffer_byte_view(
                self.arena,
                plan.main_slots_offset,
                plan.main_slots_bytes,
                "integrated dSpark proposal KV slots",
            )?,
            cos_sin: device_buffer_byte_view(
                self.arena,
                plan.cos_sin_offset,
                plan.cos_sin_bytes,
                "integrated dSpark proposal cos/sin cache",
            )?,
            post_ping: device_buffer_byte_view(
                self.arena,
                plan.post_ping_offset,
                plan.post_ping_bytes,
                "integrated dSpark proposal post ping",
            )?,
            comb_ping: device_buffer_byte_view(
                self.arena,
                plan.comb_ping_offset,
                plan.comb_ping_bytes,
                "integrated dSpark proposal combination ping",
            )?,
            post_pong: device_buffer_byte_view(
                self.arena,
                plan.post_pong_offset,
                plan.post_pong_bytes,
                "integrated dSpark proposal post pong",
            )?,
            comb_pong: device_buffer_byte_view(
                self.arena,
                plan.comb_pong_offset,
                plan.comb_pong_bytes,
                "integrated dSpark proposal combination pong",
            )?,
            selected_indices: device_buffer_byte_view(
                self.arena,
                plan.selected_indices_offset,
                plan.selected_indices_bytes,
                "integrated dSpark proposal selected indices",
            )?,
            selected_lengths: device_buffer_byte_view(
                self.arena,
                plan.selected_lengths_offset,
                plan.selected_lengths_bytes,
                "integrated dSpark proposal selected lengths",
            )?,
        })
    }

    /// Return exact five-row views for one request slot. The execution arena
    /// is shared, but captured graphs must never sweep uninitialized rows from
    /// unrelated request slots through dSpark's block-zero mHC pre-mix.
    pub(in crate::commands::real_full) fn proposal_request_buffers(
        &self,
        request_slot: usize,
    ) -> Result<DeepseekV4DsparkProposalDeviceBuffers> {
        anyhow::ensure!(
            request_slot < self.plan.max_batch,
            "integrated dSpark proposal request slot {request_slot} exceeds max batch {}",
            self.plan.max_batch
        );
        let full = self.proposal_buffers()?;
        let view = |buffer: Ds41rtDeviceBuffer, label: &'static str| -> Result<Ds41rtDeviceBuffer> {
            anyhow::ensure!(
                buffer.bytes % self.plan.max_batch == 0,
                "integrated dSpark {label} bytes {} do not divide across {} request slots",
                buffer.bytes,
                self.plan.max_batch
            );
            let bytes = buffer.bytes / self.plan.max_batch;
            let offset = request_slot
                .checked_mul(bytes)
                .context("integrated dSpark request-local proposal offset overflow")?;
            device_buffer_byte_view(buffer, offset, bytes, label)
        };
        Ok(DeepseekV4DsparkProposalDeviceBuffers {
            draft_token_ids: view(full.draft_token_ids, "dSpark request token ids")?,
            residual_ping: view(full.residual_ping, "dSpark request residual ping")?,
            residual_pong: view(full.residual_pong, "dSpark request residual pong")?,
            collapsed_hidden: view(full.collapsed_hidden, "dSpark request collapsed hidden")?,
            normalized_hidden: view(full.normalized_hidden, "dSpark request normalized hidden")?,
            positions: view(full.positions, "dSpark request positions")?,
            main_slots: view(full.main_slots, "dSpark request main slots")?,
            cos_sin: view(full.cos_sin, "dSpark request cos/sin")?,
            post_ping: view(full.post_ping, "dSpark request post ping")?,
            comb_ping: view(full.comb_ping, "dSpark request combination ping")?,
            post_pong: view(full.post_pong, "dSpark request post pong")?,
            comb_pong: view(full.comb_pong, "dSpark request combination pong")?,
            selected_indices: view(full.selected_indices, "dSpark request selected indices")?,
            selected_lengths: view(full.selected_lengths, "dSpark request selected lengths")?,
        })
    }

    pub(in crate::commands::real_full) fn terminal_buffers(
        &self,
    ) -> Result<DeepseekV4DsparkTerminalDeviceBuffers> {
        let plan = self.plan.terminal;
        let view = |offset, bytes, label| device_buffer_byte_view(self.arena, offset, bytes, label);
        Ok(DeepseekV4DsparkTerminalDeviceBuffers {
            compact_normalized_hidden: view(
                plan.compact_normalized_hidden_offset,
                plan.compact_normalized_hidden_bytes,
                "integrated dSpark compact normalized hidden",
            )?,
            shared_logits: view(
                plan.shared_logits_offset,
                plan.shared_logits_bytes,
                "integrated dSpark shared LM logits",
            )?,
            markov_logits: view(
                plan.markov_logits_offset,
                plan.markov_logits_bytes,
                "integrated dSpark Markov logits",
            )?,
            markov_embeddings: view(
                plan.markov_embeddings_offset,
                plan.markov_embeddings_bytes,
                "integrated dSpark Markov embeddings",
            )?,
            confidence: view(
                plan.confidence_offset,
                plan.confidence_bytes,
                "integrated dSpark conditional confidence",
            )?,
            active_slot_ids: view(
                plan.active_slot_ids_offset,
                plan.active_slot_ids_bytes,
                "integrated dSpark active slot IDs",
            )?,
            anchor_token_ids: view(
                plan.anchor_token_ids_offset,
                plan.anchor_token_ids_bytes,
                "integrated dSpark anchor token IDs",
            )?,
            output_token_ids: view(
                plan.output_token_ids_offset,
                plan.output_token_ids_bytes,
                "integrated dSpark output token IDs",
            )?,
        })
    }

    fn block_pre_dispatch_buffers(
        &self,
        request_slot: usize,
    ) -> Result<DeepseekV4DsparkBlockPreDispatchBuffers> {
        let proposal = self.proposal_request_buffers(request_slot)?;
        let workspace = self.entry_buffers()?.block_workspace;
        let plan = self.plan.block;
        let route_bytes = DSPARK_PROPOSAL_ROWS_PER_REQUEST
            .checked_mul(6)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u32>()))
            .context("integrated dSpark request route bytes overflow")?;
        let routed_experts = if self.plan.variant == "flash" {
            256
        } else {
            384
        };
        let route_score_workspace_bytes = DSPARK_PROPOSAL_ROWS_PER_REQUEST
            .checked_mul(6 + routed_experts)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .context("integrated dSpark request route score workspace bytes overflow")?;
        let delta_bytes = DSPARK_PROPOSAL_ROWS_PER_REQUEST
            .checked_mul(self.plan.hidden)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("integrated dSpark request shared-delta bytes overflow")?;
        anyhow::ensure!(
            route_bytes <= plan.route_indices_bytes
                && route_score_workspace_bytes <= plan.route_scores_bytes
                && route_bytes <= plan.route_weights_bytes
                && delta_bytes <= plan.shared_delta_bytes
                && delta_bytes <= plan.ffn_delta_bytes
                && plan.reduction_f32_bytes >= delta_bytes * 2
                && plan.reduction_f32_offset < plan.ffn_delta_offset,
            "integrated dSpark request-local sparse views exceed the planned workspace"
        );
        let view = |offset: usize, bytes: usize, label: &'static str| {
            device_buffer_byte_view(workspace, offset, bytes, label)
        };
        Ok(DeepseekV4DsparkBlockPreDispatchBuffers {
            dispatch_hidden: proposal.collapsed_hidden,
            sparse: Ds4FlashSparseBlockPreDispatchDeviceBuffers {
                router: Ds4FlashRouterTopKDeviceBuffers {
                    indices: view(
                        plan.route_indices_offset,
                        route_bytes,
                        "integrated dSpark global route indices",
                    )?,
                    scores: view(
                        plan.route_scores_offset,
                        route_score_workspace_bytes,
                        "integrated dSpark global route scores",
                    )?,
                    weights: view(
                        plan.route_weights_offset,
                        route_bytes,
                        "integrated dSpark global route weights",
                    )?,
                },
                shared: Ds4FlashSharedExpertDeviceBuffers {
                    gate: view(
                        plan.shared_gate_offset,
                        plan.shared_gate_bytes,
                        "integrated dSpark shared-expert gate workspace",
                    )?,
                    up: view(
                        plan.shared_up_offset,
                        plan.shared_up_bytes,
                        "integrated dSpark shared-expert up workspace",
                    )?,
                    activated: view(
                        plan.shared_activated_offset,
                        plan.shared_activated_bytes,
                        "integrated dSpark shared-expert activation workspace",
                    )?,
                    output: view(
                        plan.shared_delta_offset,
                        delta_bytes,
                        "integrated dSpark shared-expert delta",
                    )?,
                },
            },
            ffn_delta: view(
                plan.ffn_delta_offset,
                delta_bytes,
                "integrated dSpark reduced FFN delta",
            )?,
        })
    }

    pub(in crate::commands::real_full) fn stream_ptr(&self) -> *mut c_void {
        self.stream.as_ptr()
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::commands::real_full) fn prepare_proposal_inputs(
        &mut self,
        anchor_token_id: usize,
        noise_token_id: usize,
        request_slot: usize,
        cache_window_start: usize,
        main_context_end: usize,
        vocab_size: usize,
        rope_theta: f32,
    ) -> Result<DeepseekV4DsparkProposalDeviceBuffers> {
        anyhow::ensure!(
            request_slot < self.plan.max_batch,
            "integrated dSpark proposal request slot {request_slot} exceeds max batch {}",
            self.plan.max_batch
        );
        anyhow::ensure!(
            anchor_token_id < vocab_size && noise_token_id < vocab_size,
            "integrated dSpark proposal token ids {anchor_token_id}/{noise_token_id} exceed vocabulary {vocab_size}"
        );
        anyhow::ensure!(
            cache_window_start < main_context_end
                && main_context_end - cache_window_start <= 128,
            "integrated dSpark proposal target window must contain 1..=128 tokens, got {cache_window_start}..{main_context_end}"
        );
        let embedding_bytes = vocab_size
            .checked_mul(self.plan.hidden)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("integrated dSpark embedding byte count overflow")?;
        let embedding =
            preloaded_resident_weight_device_buffer(DSPARK_EMBED_WEIGHT, embedding_bytes)
                .context("resolving integrated dSpark shared target embedding")?;
        // The native preparation kernels receive request-local output
        // pointers. request_slot is used only to encode physical KV indices;
        // the kernels deliberately do not stride these arena pointers.
        let proposal = self.proposal_request_buffers(request_slot)?;
        anyhow::ensure!(
            [
                embedding.device_id,
                proposal.draft_token_ids.device_id,
                proposal.residual_ping.device_id,
                proposal.positions.device_id,
                proposal.main_slots.device_id,
                proposal.cos_sin.device_id,
                proposal.selected_indices.device_id,
                proposal.selected_lengths.device_id,
            ]
            .into_iter()
            .all(|device| device == 0),
            "integrated dSpark proposal preparation requires every weight and arena view on GPU0"
        );
        unsafe {
            self.library.cuda_ds4_dspark_prepare_proposal_async(
                embedding,
                proposal.draft_token_ids,
                proposal.residual_ping,
                proposal.positions,
                proposal.main_slots,
                proposal.cos_sin,
                proposal.selected_indices,
                proposal.selected_lengths,
                anchor_token_id,
                noise_token_id,
                request_slot,
                cache_window_start,
                main_context_end,
                vocab_size,
                self.plan.hidden,
                rope_theta,
                self.stream.as_ptr(),
            )
        }
        .context("launching integrated dSpark proposal input preparation")?;
        self.stream
            .synchronize()
            .context("synchronizing integrated dSpark proposal input preparation")?;
        Ok(proposal)
    }

    fn persistent_kv_block(&self, block_index: usize) -> Result<Ds41rtDeviceBuffer> {
        anyhow::ensure!(
            block_index < DSPARK_BLOCKS,
            "integrated dSpark KV block index {block_index} is out of range"
        );
        let block_bytes = self
            .plan
            .max_batch
            .checked_mul(self.plan.packed_page_bytes)
            .context("integrated dSpark KV block byte count overflow")?;
        let block_offset = block_index
            .checked_mul(block_bytes)
            .context("integrated dSpark KV block offset overflow")?;
        device_buffer_byte_view(
            self.persistent_kv,
            block_offset,
            block_bytes,
            "integrated dSpark block-local packed KV pages",
        )
    }

    fn block_attention_weights(
        &self,
        block_index: usize,
    ) -> Result<DeepseekV4DsparkBlockAttentionWeights> {
        anyhow::ensure!(
            block_index < DSPARK_BLOCKS,
            "integrated dSpark attention block index {block_index} is out of range"
        );
        let (q_rank, heads, output_groups) = dspark_block_attention_geometry(self.plan.variant)?;
        let hidden = self.plan.hidden;
        let head_dim = DSPARK_KV_WIDTH;
        let query_width = heads
            .checked_mul(head_dim)
            .context("integrated dSpark attention query width overflow")?;
        let output_rank = 1_024usize;
        let output_group_width = heads
            .checked_div(output_groups)
            .and_then(|heads_per_group| heads_per_group.checked_mul(head_dim))
            .context("integrated dSpark attention output group width overflow")?;
        let output_projected_width = output_groups
            .checked_mul(output_rank)
            .context("integrated dSpark attention output rank width overflow")?;
        let bytes = |rows: usize, cols: usize| {
            rows.checked_mul(cols)
                .context("integrated dSpark attention tensor bytes overflow")
        };
        let scales = |rows: usize, cols: usize| {
            rows.div_ceil(128)
                .checked_mul(cols.div_ceil(128))
                .context("integrated dSpark attention scale bytes overflow")
        };
        let prefix = format!("mtp.{block_index}");
        let resident = |suffix: &str, expected_bytes: usize| {
            preloaded_resident_weight_device_buffer(&format!("{prefix}.{suffix}"), expected_bytes)
        };
        Ok(DeepseekV4DsparkBlockAttentionWeights {
            wq_a_weight: resident("attn.wq_a.weight", bytes(q_rank, hidden)?)?,
            wq_a_scale: resident("attn.wq_a.scale", scales(q_rank, hidden)?)?,
            wq_b_weight: resident("attn.wq_b.weight", bytes(query_width, q_rank)?)?,
            wq_b_scale: resident("attn.wq_b.scale", scales(query_width, q_rank)?)?,
            wkv_weight: resident("attn.wkv.weight", bytes(head_dim, hidden)?)?,
            wkv_scale: resident("attn.wkv.scale", scales(head_dim, hidden)?)?,
            q_norm_weight: resident(
                "attn.q_norm.weight",
                q_rank
                    .checked_mul(std::mem::size_of::<u16>())
                    .context("integrated dSpark attention Q norm bytes overflow")?,
            )?,
            kv_norm_weight: resident(
                "attn.kv_norm.weight",
                head_dim
                    .checked_mul(std::mem::size_of::<u16>())
                    .context("integrated dSpark attention KV norm bytes overflow")?,
            )?,
            wo_a_weight: resident(
                "attn.wo_a.weight",
                bytes(output_projected_width, output_group_width)?,
            )?,
            wo_a_scale: resident(
                "attn.wo_a.scale",
                scales(output_projected_width, output_group_width)?,
            )?,
            wo_b_weight: resident("attn.wo_b.weight", bytes(hidden, output_projected_width)?)?,
            wo_b_scale: resident("attn.wo_b.scale", scales(hidden, output_projected_width)?)?,
            attn_sink: resident(
                "attn.attn_sink",
                heads
                    .checked_mul(std::mem::size_of::<f32>())
                    .context("integrated dSpark attention sink bytes overflow")?,
            )?,
            hc_fn: resident(
                "hc_ffn_fn",
                DSPARK_HC_MIXES
                    .checked_mul(DSPARK_PROPOSAL_HC_MULT)
                    .and_then(|values| values.checked_mul(hidden))
                    .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
                    .context("integrated dSpark FFN HC function bytes overflow")?,
            )?,
            hc_scale: resident("hc_ffn_scale", 3 * std::mem::size_of::<f32>())?,
            hc_base: resident("hc_ffn_base", DSPARK_HC_MIXES * std::mem::size_of::<f32>())?,
            norm_weight: resident(
                "ffn_norm.weight",
                hidden
                    .checked_mul(std::mem::size_of::<u16>())
                    .context("integrated dSpark FFN norm bytes overflow")?,
            )?,
        })
    }

    fn block_post_dispatch_weights(
        &self,
        block_index: usize,
    ) -> Result<DeepseekV4DsparkBlockPostDispatchWeights> {
        anyhow::ensure!(
            block_index + 1 < DSPARK_BLOCKS,
            "integrated dSpark post-dispatch block index {block_index} has no next attention block"
        );
        let hidden = self.plan.hidden;
        let hc_fn_bytes = DSPARK_HC_MIXES
            .checked_mul(DSPARK_PROPOSAL_HC_MULT)
            .and_then(|values| values.checked_mul(hidden))
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .context("integrated dSpark next-attention HC function bytes overflow")?;
        let next_prefix = format!("mtp.{}", block_index + 1);
        let resident = |suffix: &str, expected_bytes: usize| {
            preloaded_resident_weight_device_buffer(
                &format!("{next_prefix}.{suffix}"),
                expected_bytes,
            )
        };
        Ok(DeepseekV4DsparkBlockPostDispatchWeights {
            next_hc_fn: resident("hc_attn_fn", hc_fn_bytes)?,
            next_hc_scale: resident("hc_attn_scale", 3 * std::mem::size_of::<f32>())?,
            next_hc_base: resident("hc_attn_base", DSPARK_HC_MIXES * std::mem::size_of::<f32>())?,
            next_norm_weight: resident(
                "attn_norm.weight",
                hidden
                    .checked_mul(std::mem::size_of::<u16>())
                    .context("integrated dSpark next-attention norm bytes overflow")?,
            )?,
        })
    }

    fn terminal_collapse_weights(&self) -> Result<DeepseekV4DsparkTerminalCollapseWeights> {
        let hidden = self.plan.hidden;
        let hc_fn_bytes = DSPARK_PROPOSAL_HC_MULT
            .checked_mul(DSPARK_PROPOSAL_HC_MULT)
            .and_then(|values| values.checked_mul(hidden))
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .context("integrated dSpark terminal HC function bytes overflow")?;
        Ok(DeepseekV4DsparkTerminalCollapseWeights {
            hc_fn: preloaded_resident_weight_device_buffer(DSPARK_TERMINAL_HC_FN, hc_fn_bytes)?,
            hc_scale: preloaded_resident_weight_device_buffer(
                DSPARK_TERMINAL_HC_SCALE,
                std::mem::size_of::<f32>(),
            )?,
            hc_base: preloaded_resident_weight_device_buffer(
                DSPARK_TERMINAL_HC_BASE,
                DSPARK_PROPOSAL_HC_MULT * std::mem::size_of::<f32>(),
            )?,
            norm_weight: preloaded_resident_weight_device_buffer(
                DSPARK_TERMINAL_NORM,
                hidden
                    .checked_mul(std::mem::size_of::<u16>())
                    .context("integrated dSpark terminal norm bytes overflow")?,
            )?,
        })
    }

    fn terminal_head_weights(&self) -> Result<DeepseekV4DsparkTerminalHeadWeights> {
        let head_bytes = self
            .plan
            .vocab
            .checked_mul(self.plan.hidden)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("integrated dSpark shared-head bytes overflow")?;
        let markov_bytes = self
            .plan
            .vocab
            .checked_mul(self.plan.markov_rank)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("integrated dSpark Markov-head bytes overflow")?;
        let confidence_bytes = self
            .plan
            .hidden
            .checked_add(self.plan.markov_rank)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("integrated dSpark confidence-head bytes overflow")?;
        Ok(DeepseekV4DsparkTerminalHeadWeights {
            shared_head: preloaded_resident_weight_device_buffer(DSPARK_SHARED_HEAD, head_bytes)?,
            markov_w1: preloaded_resident_weight_device_buffer(
                DSPARK_TERMINAL_MARKOV_W1,
                markov_bytes,
            )?,
            markov_w2: preloaded_resident_weight_device_buffer(
                DSPARK_TERMINAL_MARKOV_W2,
                markov_bytes,
            )?,
            confidence: preloaded_resident_weight_device_buffer(
                DSPARK_TERMINAL_CONFIDENCE,
                confidence_bytes,
            )?,
        })
    }

    pub(in crate::commands::real_full) fn prepare_entry_projection_graphs(
        &mut self,
        norm_eps: f32,
    ) -> Result<()> {
        anyhow::ensure!(
            norm_eps.is_finite() && norm_eps > 0.0,
            "integrated dSpark entry RMSNorm epsilon must be finite and positive, got {norm_eps}"
        );
        if !self.entry_graphs.is_empty() {
            anyhow::ensure!(
                self.entry_graphs.len() == DSPARK_ENTRY_GRAPH_ROWS.len(),
                "integrated dSpark entry graph set is partially initialized"
            );
            return Ok(());
        }
        let hidden = self.plan.hidden;
        let projection_width = hidden
            .checked_mul(3)
            .context("integrated dSpark entry projection width overflow")?;
        let weight_bytes = hidden
            .checked_mul(projection_width)
            .context("integrated dSpark main projection weight bytes overflow")?;
        let scale_bytes = hidden
            .div_ceil(128)
            .checked_mul(projection_width.div_ceil(128))
            .context("integrated dSpark main projection scale bytes overflow")?;
        let norm_bytes = hidden
            .checked_mul(std::mem::size_of::<u16>())
            .context("integrated dSpark main norm bytes overflow")?;
        let main_proj_weight =
            preloaded_resident_weight_device_buffer(DSPARK_MAIN_PROJ_WEIGHT, weight_bytes)?;
        let main_proj_scale =
            preloaded_resident_weight_device_buffer(DSPARK_MAIN_PROJ_SCALE, scale_bytes)?;
        let main_norm_weight =
            preloaded_resident_weight_device_buffer(DSPARK_MAIN_NORM_WEIGHT, norm_bytes)?;
        let entry = self.entry_buffers()?;
        anyhow::ensure!(
            [
                main_proj_weight.device_id,
                main_proj_scale.device_id,
                main_norm_weight.device_id,
                entry.target_tap_concat.device_id,
                entry.projection_scratch.device_id,
                entry.projected_target_main.device_id,
            ]
            .into_iter()
            .all(|device| device == 0),
            "integrated dSpark entry projection requires every weight and arena view on GPU0"
        );
        let stream = self.stream.as_ptr();
        for rows in DSPARK_ENTRY_GRAPH_ROWS {
            anyhow::ensure!(
                rows <= self.plan.max_main_rows,
                "integrated dSpark entry graph bucket {rows} exceeds max rows {}",
                self.plan.max_main_rows
            );
            let buffers = dspark_entry_python_buffers(entry, main_proj_weight, main_proj_scale);
            let kwargs = [
                ("variant", PythonKernelArg::Str(self.plan.variant)),
                ("rows", PythonKernelArg::Usize(rows)),
                ("max_rows", PythonKernelArg::Usize(self.plan.max_main_rows)),
                ("max_batch", PythonKernelArg::Usize(self.plan.max_batch)),
            ];
            launch_python_graph_capture(PythonGraphCaptureLaunch {
                module: DSPARK_CAPTURE_MODULE,
                function: DSPARK_ENTRY_PREPARE_FUNCTION,
                cuda_stream: stream,
                buffers: &buffers,
                kwargs: &kwargs,
            })
            .with_context(|| format!("preparing integrated dSpark entry bucket {rows}"))?;
            self.stream
                .synchronize()
                .with_context(|| format!("synchronizing prepared dSpark entry bucket {rows}"))?;
            unsafe {
                self.library
                    .cuda_graph_begin_capture(stream)
                    .with_context(|| format!("beginning dSpark entry bucket {rows} capture"))?;
            }
            let captured = launch_python_graph_capture(PythonGraphCaptureLaunch {
                module: DSPARK_CAPTURE_MODULE,
                function: DSPARK_ENTRY_CAPTURE_FUNCTION,
                cuda_stream: stream,
                buffers: &buffers,
                kwargs: &kwargs,
            })
            .and_then(|()| unsafe {
                self.library.cuda_ds4_rmsnorm_bf16_rne_async(
                    entry.projected_target_main,
                    main_norm_weight,
                    entry.projected_target_main,
                    rows as i32,
                    hidden as i32,
                    norm_eps,
                    stream,
                )
            });
            if let Err(error) = captured {
                if let Ok(capture) = unsafe { self.library.cuda_graph_end_capture(stream) } {
                    let _ = unsafe { self.library.cuda_graph_exec_destroy(capture) };
                }
                return Err(error)
                    .with_context(|| format!("capturing integrated dSpark entry bucket {rows}"));
            }
            let capture = unsafe {
                self.library
                    .cuda_graph_end_capture_retained(stream)
                    .with_context(|| format!("ending dSpark entry bucket {rows} capture"))?
            };
            let graph = CoordinatorCudaCapturedGraph::new(self.library, capture)?;
            graph.validate_before_launch()?;
            anyhow::ensure!(
                graph.memcpy_node_count == 0,
                "integrated dSpark entry graph bucket {rows} unexpectedly captured memcpy nodes"
            );
            self.entry_graphs
                .push(DeepseekV4DsparkEntryGraph { rows, graph });
        }
        Ok(())
    }

    pub(in crate::commands::real_full) fn entry_projection_graphs_prepared(&self) -> bool {
        self.entry_graphs.len() == DSPARK_ENTRY_GRAPH_ROWS.len()
            && self
                .entry_graphs
                .iter()
                .zip(DSPARK_ENTRY_GRAPH_ROWS)
                .all(|(graph, rows)| graph.rows == rows)
    }

    pub(in crate::commands::real_full) fn prepare_prompt_prime_graphs(
        &mut self,
        producer_eps: f32,
        rope_theta: f32,
    ) -> Result<()> {
        anyhow::ensure!(
            producer_eps.is_finite() && producer_eps > 0.0,
            "integrated dSpark KV producer epsilon must be finite and positive, got {producer_eps}"
        );
        anyhow::ensure!(
            rope_theta.is_finite() && rope_theta > 0.0,
            "integrated dSpark RoPE theta must be finite and positive, got {rope_theta}"
        );
        let expected_graphs = DSPARK_BLOCKS
            .checked_mul(DSPARK_ENTRY_GRAPH_ROWS.len())
            .context("integrated dSpark prompt graph count overflow")?;
        if !self.prompt_prime_graphs.is_empty() {
            anyhow::ensure!(
                self.prompt_prime_graphs.len() == expected_graphs
                    && self.prompt_prime_graphs_prepared(),
                "integrated dSpark prompt-prime graph set is partially initialized"
            );
            return Ok(());
        }
        let wkv_weight_bytes = DSPARK_KV_WIDTH
            .checked_mul(self.plan.hidden)
            .context("integrated dSpark WKV weight byte count overflow")?;
        let wkv_scale_bytes = DSPARK_KV_WIDTH
            .div_ceil(128)
            .checked_mul(self.plan.hidden.div_ceil(128))
            .context("integrated dSpark WKV scale byte count overflow")?;
        let kv_norm_bytes = DSPARK_KV_WIDTH
            .checked_mul(std::mem::size_of::<u16>())
            .context("integrated dSpark KV norm byte count overflow")?;
        let entry = self.entry_buffers()?;
        let stream = self.stream.as_ptr();
        for block_index in 0..DSPARK_BLOCKS {
            let prefix = format!("mtp.{block_index}.attn");
            let wkv_weight = preloaded_resident_weight_device_buffer(
                &format!("{prefix}.wkv.weight"),
                wkv_weight_bytes,
            )?;
            let wkv_scale = preloaded_resident_weight_device_buffer(
                &format!("{prefix}.wkv.scale"),
                wkv_scale_bytes,
            )?;
            let kv_norm_weight = preloaded_resident_weight_device_buffer(
                &format!("{prefix}.kv_norm.weight"),
                kv_norm_bytes,
            )?;
            let main_kv_cache = self.persistent_kv_block(block_index)?;
            anyhow::ensure!(
                [
                    entry.projected_target_main.device_id,
                    entry.prompt_positions.device_id,
                    entry.prompt_main_slots.device_id,
                    entry.prompt_cos_sin.device_id,
                    entry.kv_producer_scratch.device_id,
                    wkv_weight.device_id,
                    wkv_scale.device_id,
                    kv_norm_weight.device_id,
                    main_kv_cache.device_id,
                ]
                .into_iter()
                .all(|device| device == 0),
                "integrated dSpark prompt-prime graph requires every weight and arena view on GPU0"
            );
            for rows in DSPARK_ENTRY_GRAPH_ROWS {
                anyhow::ensure!(
                    rows <= self.plan.max_main_rows,
                    "integrated dSpark prompt graph bucket {rows} exceeds max rows {}",
                    self.plan.max_main_rows
                );
                unsafe {
                    self.library.cuda_ds4_dspark_prompt_metadata_async(
                        entry.prompt_positions,
                        entry.prompt_main_slots,
                        entry.prompt_cos_sin,
                        0,
                        0,
                        rows,
                        rope_theta,
                        stream,
                    )
                }
                .with_context(|| {
                    format!(
                        "preparing dSpark prompt metadata for block {block_index} bucket {rows}"
                    )
                })?;
                let buffers = dspark_prompt_python_buffers(
                    entry,
                    main_kv_cache,
                    wkv_weight,
                    wkv_scale,
                    kv_norm_weight,
                );
                let kwargs = [
                    ("variant", PythonKernelArg::Str(self.plan.variant)),
                    ("block_index", PythonKernelArg::Usize(block_index)),
                    ("rows", PythonKernelArg::Usize(rows)),
                    ("max_rows", PythonKernelArg::Usize(self.plan.max_main_rows)),
                    ("max_batch", PythonKernelArg::Usize(self.plan.max_batch)),
                    ("cache_format", PythonKernelArg::Str(self.plan.cache_format)),
                    ("producer_eps", PythonKernelArg::F64(producer_eps.into())),
                ];
                launch_python_graph_capture(PythonGraphCaptureLaunch {
                    module: DSPARK_CAPTURE_MODULE,
                    function: DSPARK_PROMPT_PREPARE_FUNCTION,
                    cuda_stream: stream,
                    buffers: &buffers,
                    kwargs: &kwargs,
                })
                .with_context(|| {
                    format!("preparing integrated dSpark prompt block {block_index} bucket {rows}")
                })?;
                self.stream.synchronize().with_context(|| {
                    format!(
                        "synchronizing prepared dSpark prompt block {block_index} bucket {rows}"
                    )
                })?;
                unsafe {
                    self.library
                        .cuda_graph_begin_capture(stream)
                        .with_context(|| {
                            format!(
                                "beginning dSpark prompt block {block_index} bucket {rows} capture"
                            )
                        })?;
                }
                let captured = launch_python_graph_capture(PythonGraphCaptureLaunch {
                    module: DSPARK_CAPTURE_MODULE,
                    function: DSPARK_PROMPT_CAPTURE_FUNCTION,
                    cuda_stream: stream,
                    buffers: &buffers,
                    kwargs: &kwargs,
                });
                if let Err(error) = captured {
                    if let Ok(capture) = unsafe { self.library.cuda_graph_end_capture(stream) } {
                        let _ = unsafe { self.library.cuda_graph_exec_destroy(capture) };
                    }
                    return Err(error).with_context(|| {
                        format!(
                            "capturing integrated dSpark prompt block {block_index} bucket {rows}"
                        )
                    });
                }
                let capture = unsafe {
                    self.library
                        .cuda_graph_end_capture_retained(stream)
                        .with_context(|| {
                            format!(
                                "ending dSpark prompt block {block_index} bucket {rows} capture"
                            )
                        })?
                };
                let graph = CoordinatorCudaCapturedGraph::new(self.library, capture)?;
                graph.validate_before_launch()?;
                anyhow::ensure!(
                    graph.memcpy_node_count == 0,
                    "integrated dSpark prompt block {block_index} bucket {rows} unexpectedly captured memcpy nodes"
                );
                self.prompt_prime_graphs
                    .push(DeepseekV4DsparkPromptPrimeGraph {
                        block_index,
                        rows,
                        graph,
                    });
            }
        }
        Ok(())
    }

    pub(in crate::commands::real_full) fn prompt_prime_graphs_prepared(&self) -> bool {
        self.prompt_prime_graphs.len() == DSPARK_BLOCKS * DSPARK_ENTRY_GRAPH_ROWS.len()
            && self
                .prompt_prime_graphs
                .iter()
                .zip((0..DSPARK_BLOCKS).flat_map(|block| {
                    DSPARK_ENTRY_GRAPH_ROWS
                        .into_iter()
                        .map(move |rows| (block, rows))
                }))
                .all(|(graph, expected)| (graph.block_index, graph.rows) == expected)
    }

    pub(in crate::commands::real_full) fn prepare_proposal_entry_graphs(
        &mut self,
        catalog: &TensorCatalog,
        rms_eps: f32,
        hc_eps: f32,
        sinkhorn_iters: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            rms_eps.is_finite() && rms_eps > 0.0 && hc_eps.is_finite() && hc_eps > 0.0,
            "integrated dSpark proposal-entry epsilons must be finite and positive"
        );
        anyhow::ensure!(
            sinkhorn_iters > 0,
            "integrated dSpark proposal-entry Sinkhorn iterations must be positive"
        );
        if !self.proposal_entry_graphs.is_empty() {
            anyhow::ensure!(
                self.proposal_entry_graphs_prepared(),
                "integrated dSpark proposal-entry graph set is partially initialized"
            );
            return Ok(());
        }
        let hidden = self.plan.hidden;
        let hc_fn_bytes = DSPARK_HC_MIXES
            .checked_mul(hidden)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .context("integrated dSpark proposal-entry HC function bytes overflow")?;
        let hc_scale_bytes = 3 * std::mem::size_of::<f32>();
        let hc_base_bytes = DSPARK_HC_MIXES * std::mem::size_of::<f32>();
        let norm_bytes = hidden
            .checked_mul(std::mem::size_of::<u16>())
            .context("integrated dSpark proposal-entry norm bytes overflow")?;
        preload_dspark_block0_hc_attn_lane_sum(catalog, hidden)?;
        let hc_fn = preloaded_resident_weight_device_buffer(
            DSPARK_BLOCK0_HC_ATTN_FN_LANE_SUM,
            hc_fn_bytes,
        )?;
        let hc_scale =
            preloaded_resident_weight_device_buffer(DSPARK_BLOCK0_HC_ATTN_SCALE, hc_scale_bytes)?;
        let hc_base =
            preloaded_resident_weight_device_buffer(DSPARK_BLOCK0_HC_ATTN_BASE, hc_base_bytes)?;
        let norm_weight =
            preloaded_resident_weight_device_buffer(DSPARK_BLOCK0_ATTN_NORM, norm_bytes)?;
        let entry = self.entry_buffers()?;
        let workspace = entry.block_workspace;
        let proposal_embedding_bytes = DSPARK_PROPOSAL_ROWS_PER_REQUEST
            .checked_mul(hidden)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("integrated dSpark proposal-entry embedding bytes overflow")?;
        let proposal_embedding = device_buffer_byte_view(
            entry.projected_target_main,
            0,
            proposal_embedding_bytes,
            "integrated dSpark proposal-entry base embeddings",
        )?;
        let stream = self.stream.as_ptr();
        for request_slot in 0..self.plan.max_batch {
            let proposal = self.proposal_request_buffers(request_slot)?;
            anyhow::ensure!(
                [
                    workspace.device_id,
                    proposal_embedding.device_id,
                    proposal.residual_pong.device_id,
                    proposal.collapsed_hidden.device_id,
                    proposal.post_ping.device_id,
                    proposal.comb_ping.device_id,
                    hc_fn.device_id,
                    hc_scale.device_id,
                    hc_base.device_id,
                    norm_weight.device_id,
                ]
                .into_iter()
                .all(|device| device == 0),
                "integrated dSpark proposal-entry graph requires every weight and arena view on GPU0"
            );
            let buffers = dspark_proposal_entry_python_buffers(
                workspace,
                proposal_embedding,
                proposal,
                hc_fn,
                hc_scale,
                hc_base,
                norm_weight,
            );
            let kwargs = [
                ("variant", PythonKernelArg::Str(self.plan.variant)),
                ("max_batch", PythonKernelArg::Usize(self.plan.max_batch)),
                (
                    "max_main_rows",
                    PythonKernelArg::Usize(self.plan.max_main_rows),
                ),
                ("rms_eps", PythonKernelArg::F64(rms_eps.into())),
                ("hc_eps", PythonKernelArg::F64(hc_eps.into())),
                ("sinkhorn_iters", PythonKernelArg::Usize(sinkhorn_iters)),
                ("norm_eps", PythonKernelArg::F64(rms_eps.into())),
            ];
            launch_python_graph_capture(PythonGraphCaptureLaunch {
                module: DSPARK_CAPTURE_MODULE,
                function: DSPARK_PROPOSAL_ENTRY_PREPARE_FUNCTION,
                cuda_stream: stream,
                buffers: &buffers,
                kwargs: &kwargs,
            })
            .with_context(|| {
                format!("preparing integrated dSpark proposal-entry request slot {request_slot}")
            })?;
            self.stream.synchronize().with_context(|| {
                format!("synchronizing prepared dSpark proposal-entry slot {request_slot}")
            })?;
            unsafe {
                self.library
                    .cuda_graph_begin_capture(stream)
                    .with_context(|| {
                        format!("beginning dSpark proposal-entry slot {request_slot} capture")
                    })?;
            }
            let captured = launch_python_graph_capture(PythonGraphCaptureLaunch {
                module: DSPARK_CAPTURE_MODULE,
                function: DSPARK_PROPOSAL_ENTRY_CAPTURE_FUNCTION,
                cuda_stream: stream,
                buffers: &buffers,
                kwargs: &kwargs,
            });
            if let Err(error) = captured {
                if let Ok(capture) = unsafe { self.library.cuda_graph_end_capture(stream) } {
                    let _ = unsafe { self.library.cuda_graph_exec_destroy(capture) };
                }
                return Err(error).with_context(|| {
                    format!("capturing integrated dSpark proposal-entry slot {request_slot}")
                });
            }
            let capture = unsafe {
                self.library
                    .cuda_graph_end_capture_retained(stream)
                    .with_context(|| {
                        format!("ending dSpark proposal-entry slot {request_slot} capture")
                    })?
            };
            let graph = CoordinatorCudaCapturedGraph::new(self.library, capture)?;
            graph.validate_before_launch()?;
            anyhow::ensure!(
                graph.memcpy_node_count == 0,
                "integrated dSpark proposal-entry slot {request_slot} unexpectedly captured memcpy nodes"
            );
            self.proposal_entry_graphs
                .push(DeepseekV4DsparkProposalEntryGraph {
                    request_slot,
                    graph,
                });
        }
        Ok(())
    }

    pub(in crate::commands::real_full) fn proposal_entry_graphs_prepared(&self) -> bool {
        self.proposal_entry_graphs.len() == self.plan.max_batch
            && self
                .proposal_entry_graphs
                .iter()
                .enumerate()
                .all(|(slot, graph)| graph.request_slot == slot)
    }

    pub(in crate::commands::real_full) fn run_proposal_entry(
        &mut self,
        request_slot: usize,
    ) -> Result<DeepseekV4DsparkProposalDeviceBuffers> {
        anyhow::ensure!(
            self.proposal_entry_graphs_prepared(),
            "integrated dSpark proposal-entry graphs are not prepared"
        );
        let proposal = self.proposal_request_buffers(request_slot)?;
        let hidden_row_bytes = self
            .plan
            .hidden
            .checked_mul(std::mem::size_of::<u16>())
            .context("integrated dSpark proposal-entry hidden row bytes overflow")?;
        let entry = self.entry_buffers()?;
        unsafe {
            self.library.copy_d2d_2d_async(
                entry.projected_target_main,
                hidden_row_bytes,
                proposal.residual_ping,
                DSPARK_PROPOSAL_HC_MULT * hidden_row_bytes,
                hidden_row_bytes,
                DSPARK_PROPOSAL_ROWS_PER_REQUEST,
                self.stream.as_ptr(),
            )
        }
        .context("gathering one base embedding from each repeated dSpark HC row")?;
        let graph = self
            .proposal_entry_graphs
            .iter()
            .find(|graph| graph.request_slot == request_slot)
            .with_context(|| {
                format!("integrated dSpark proposal-entry graph slot {request_slot} is missing")
            })?;
        graph.graph.validate_before_launch()?;
        unsafe {
            self.library
                .cuda_graph_launch(graph.graph.as_ptr(), self.stream.as_ptr())
        }
        .with_context(|| {
            format!("launching integrated dSpark proposal-entry graph slot {request_slot}")
        })?;
        self.stream.synchronize().with_context(|| {
            format!("synchronizing integrated dSpark proposal-entry slot {request_slot}")
        })?;
        Ok(proposal)
    }

    pub(in crate::commands::real_full) fn prepare_block_attention_graphs(
        &mut self,
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
                && hc_eps > 0.0,
            "integrated dSpark block-attention epsilons must be finite and positive"
        );
        anyhow::ensure!(
            sinkhorn_iters > 0,
            "integrated dSpark block-attention Sinkhorn iterations must be positive"
        );
        let expected_graphs = DSPARK_BLOCKS
            .checked_mul(self.plan.max_batch)
            .context("integrated dSpark block-attention graph count overflow")?;
        if !self.block_attention_graphs.is_empty() {
            anyhow::ensure!(
                self.block_attention_graphs.len() == expected_graphs
                    && self.block_attention_graphs_prepared(),
                "integrated dSpark block-attention graph set is partially initialized"
            );
            return Ok(());
        }
        let workspace = self.entry_buffers()?.block_workspace;
        let stream = self.stream.as_ptr();
        for block_index in 0..DSPARK_BLOCKS {
            let weights = self.block_attention_weights(block_index)?;
            let main_kv_cache = self.persistent_kv_block(block_index)?;
            anyhow::ensure!(
                [
                    workspace.device_id,
                    main_kv_cache.device_id,
                    weights.wq_a_weight.device_id,
                    weights.wq_a_scale.device_id,
                    weights.wq_b_weight.device_id,
                    weights.wq_b_scale.device_id,
                    weights.wkv_weight.device_id,
                    weights.wkv_scale.device_id,
                    weights.q_norm_weight.device_id,
                    weights.kv_norm_weight.device_id,
                    weights.wo_a_weight.device_id,
                    weights.wo_a_scale.device_id,
                    weights.wo_b_weight.device_id,
                    weights.wo_b_scale.device_id,
                    weights.attn_sink.device_id,
                    weights.hc_fn.device_id,
                    weights.hc_scale.device_id,
                    weights.hc_base.device_id,
                    weights.norm_weight.device_id,
                ]
                .into_iter()
                .all(|device| device == 0),
                "integrated dSpark block-attention graph requires every weight and persistent view on GPU0"
            );
            for request_slot in 0..self.plan.max_batch {
                let proposal = self.proposal_request_buffers(request_slot)?;
                let buffers = dspark_block_attention_python_buffers(
                    workspace,
                    main_kv_cache,
                    proposal,
                    weights,
                );
                let kwargs = [
                    ("variant", PythonKernelArg::Str(self.plan.variant)),
                    ("block_index", PythonKernelArg::Usize(block_index)),
                    ("max_batch", PythonKernelArg::Usize(self.plan.max_batch)),
                    (
                        "max_main_rows",
                        PythonKernelArg::Usize(self.plan.max_main_rows),
                    ),
                    ("cache_format", PythonKernelArg::Str(self.plan.cache_format)),
                    ("producer_eps", PythonKernelArg::F64(producer_eps.into())),
                    ("rms_eps", PythonKernelArg::F64(rms_eps.into())),
                    ("hc_eps", PythonKernelArg::F64(hc_eps.into())),
                    ("sinkhorn_iters", PythonKernelArg::Usize(sinkhorn_iters)),
                    ("norm_eps", PythonKernelArg::F64(rms_eps.into())),
                ];
                launch_python_graph_capture(PythonGraphCaptureLaunch {
                    module: DSPARK_CAPTURE_MODULE,
                    function: DSPARK_BLOCK_ATTN_PREPARE_FUNCTION,
                    cuda_stream: stream,
                    buffers: &buffers,
                    kwargs: &kwargs,
                })
                .with_context(|| {
                    format!(
                        "preparing integrated dSpark block {block_index} attention request slot {request_slot}"
                    )
                })?;
                self.stream.synchronize().with_context(|| {
                    format!(
                        "synchronizing prepared dSpark block {block_index} attention slot {request_slot}"
                    )
                })?;
                unsafe {
                    self.library
                        .cuda_graph_begin_capture(stream)
                        .with_context(|| {
                            format!(
                                "beginning dSpark block {block_index} attention slot {request_slot} capture"
                            )
                        })?;
                }
                let captured = launch_python_graph_capture(PythonGraphCaptureLaunch {
                    module: DSPARK_CAPTURE_MODULE,
                    function: DSPARK_BLOCK_ATTN_CAPTURE_FUNCTION,
                    cuda_stream: stream,
                    buffers: &buffers,
                    kwargs: &kwargs,
                });
                if let Err(error) = captured {
                    if let Ok(capture) = unsafe { self.library.cuda_graph_end_capture(stream) } {
                        let _ = unsafe { self.library.cuda_graph_exec_destroy(capture) };
                    }
                    return Err(error).with_context(|| {
                        format!(
                            "capturing integrated dSpark block {block_index} attention slot {request_slot}"
                        )
                    });
                }
                let capture = unsafe {
                    self.library
                        .cuda_graph_end_capture_retained(stream)
                        .with_context(|| {
                            format!(
                                "ending dSpark block {block_index} attention slot {request_slot} capture"
                            )
                        })?
                };
                let graph = CoordinatorCudaCapturedGraph::new(self.library, capture)?;
                graph.validate_before_launch()?;
                anyhow::ensure!(
                    graph.memcpy_node_count == 0,
                    "integrated dSpark block {block_index} attention slot {request_slot} unexpectedly captured memcpy nodes"
                );
                self.block_attention_graphs
                    .push(DeepseekV4DsparkBlockAttentionGraph {
                        block_index,
                        request_slot,
                        graph,
                    });
            }
        }
        Ok(())
    }

    pub(in crate::commands::real_full) fn block_attention_graphs_prepared(&self) -> bool {
        self.block_attention_graphs.len() == DSPARK_BLOCKS * self.plan.max_batch
            && self
                .block_attention_graphs
                .iter()
                .zip((0..DSPARK_BLOCKS).flat_map(|block_index| {
                    (0..self.plan.max_batch).map(move |request_slot| (block_index, request_slot))
                }))
                .all(|(graph, expected)| (graph.block_index, graph.request_slot) == expected)
    }

    pub(in crate::commands::real_full) fn run_block_attention(
        &mut self,
        block_index: usize,
        request_slot: usize,
    ) -> Result<DeepseekV4DsparkProposalDeviceBuffers> {
        anyhow::ensure!(
            self.block_attention_graphs_prepared(),
            "integrated dSpark block-attention graphs are not prepared"
        );
        let proposal = self.proposal_request_buffers(request_slot)?;
        let graph = self
            .block_attention_graphs
            .iter()
            .find(|graph| {
                graph.block_index == block_index && graph.request_slot == request_slot
            })
            .with_context(|| {
                format!(
                    "integrated dSpark block {block_index} attention graph slot {request_slot} is missing"
                )
            })?;
        graph.graph.validate_before_launch()?;
        unsafe {
            self.library
                .cuda_graph_launch(graph.graph.as_ptr(), self.stream.as_ptr())
        }
        .with_context(|| {
            format!(
                "launching integrated dSpark block {block_index} attention graph slot {request_slot}"
            )
        })?;
        self.stream.synchronize().with_context(|| {
            format!(
                "synchronizing integrated dSpark block {block_index} attention slot {request_slot}"
            )
        })?;
        Ok(proposal)
    }

    pub(in crate::commands::real_full) fn prepare_block_post_dispatch_graphs(
        &mut self,
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
            "integrated dSpark post-dispatch graph parameters must be positive"
        );
        let expected_graphs = (DSPARK_BLOCKS - 1)
            .checked_mul(self.plan.max_batch)
            .context("integrated dSpark post-dispatch graph count overflow")?;
        if !self.block_post_dispatch_graphs.is_empty() {
            anyhow::ensure!(
                self.block_post_dispatch_graphs.len() == expected_graphs
                    && self.block_post_dispatch_graphs_prepared(),
                "integrated dSpark post-dispatch graph set is partially initialized"
            );
            return Ok(());
        }
        let workspace = self.entry_buffers()?.block_workspace;
        let stream = self.stream.as_ptr();
        for block_index in 0..DSPARK_BLOCKS - 1 {
            let weights = self.block_post_dispatch_weights(block_index)?;
            anyhow::ensure!(
                [
                    workspace.device_id,
                    weights.next_hc_fn.device_id,
                    weights.next_hc_scale.device_id,
                    weights.next_hc_base.device_id,
                    weights.next_norm_weight.device_id,
                ]
                .into_iter()
                .all(|device| device == 0),
                "integrated dSpark post-dispatch graph requires every weight and arena view on GPU0"
            );
            for request_slot in 0..self.plan.max_batch {
                let proposal = self.proposal_request_buffers(request_slot)?;
                let pre_dispatch = self.block_pre_dispatch_buffers(request_slot)?;
                let buffers = dspark_block_post_dispatch_python_buffers(
                    workspace,
                    pre_dispatch.ffn_delta,
                    proposal,
                    weights,
                );
                let kwargs = [
                    ("variant", PythonKernelArg::Str(self.plan.variant)),
                    ("block_index", PythonKernelArg::Usize(block_index)),
                    ("max_batch", PythonKernelArg::Usize(self.plan.max_batch)),
                    (
                        "max_main_rows",
                        PythonKernelArg::Usize(self.plan.max_main_rows),
                    ),
                    ("rms_eps", PythonKernelArg::F64(rms_eps.into())),
                    ("hc_eps", PythonKernelArg::F64(hc_eps.into())),
                    ("sinkhorn_iters", PythonKernelArg::Usize(sinkhorn_iters)),
                    ("norm_eps", PythonKernelArg::F64(rms_eps.into())),
                ];
                launch_python_graph_capture(PythonGraphCaptureLaunch {
                    module: DSPARK_CAPTURE_MODULE,
                    function: DSPARK_BLOCK_POST_PREPARE_FUNCTION,
                    cuda_stream: stream,
                    buffers: &buffers,
                    kwargs: &kwargs,
                })
                .with_context(|| {
                    format!(
                        "preparing integrated dSpark block {block_index} post-dispatch request slot {request_slot}"
                    )
                })?;
                self.stream.synchronize().with_context(|| {
                    format!(
                        "synchronizing prepared dSpark block {block_index} post-dispatch slot {request_slot}"
                    )
                })?;
                unsafe {
                    self.library
                        .cuda_graph_begin_capture(stream)
                        .with_context(|| {
                            format!(
                                "beginning dSpark block {block_index} post-dispatch slot {request_slot} capture"
                            )
                        })?;
                }
                let captured = launch_python_graph_capture(PythonGraphCaptureLaunch {
                    module: DSPARK_CAPTURE_MODULE,
                    function: DSPARK_BLOCK_POST_CAPTURE_FUNCTION,
                    cuda_stream: stream,
                    buffers: &buffers,
                    kwargs: &kwargs,
                });
                if let Err(error) = captured {
                    if let Ok(capture) = unsafe { self.library.cuda_graph_end_capture(stream) } {
                        let _ = unsafe { self.library.cuda_graph_exec_destroy(capture) };
                    }
                    return Err(error).with_context(|| {
                        format!(
                            "capturing integrated dSpark block {block_index} post-dispatch slot {request_slot}"
                        )
                    });
                }
                let capture = unsafe {
                    self.library
                        .cuda_graph_end_capture_retained(stream)
                        .with_context(|| {
                            format!(
                                "ending dSpark block {block_index} post-dispatch slot {request_slot} capture"
                            )
                        })?
                };
                let graph = CoordinatorCudaCapturedGraph::new(self.library, capture)?;
                graph.validate_before_launch()?;
                anyhow::ensure!(
                    graph.memcpy_node_count == 0,
                    "integrated dSpark block {block_index} post-dispatch slot {request_slot} unexpectedly captured memcpy nodes"
                );
                self.block_post_dispatch_graphs
                    .push(DeepseekV4DsparkBlockPostDispatchGraph {
                        block_index,
                        request_slot,
                        graph,
                    });
            }
        }
        Ok(())
    }

    pub(in crate::commands::real_full) fn block_post_dispatch_graphs_prepared(&self) -> bool {
        self.block_post_dispatch_graphs.len() == (DSPARK_BLOCKS - 1) * self.plan.max_batch
            && self
                .block_post_dispatch_graphs
                .iter()
                .zip((0..DSPARK_BLOCKS - 1).flat_map(|block_index| {
                    (0..self.plan.max_batch).map(move |request_slot| (block_index, request_slot))
                }))
                .all(|(graph, expected)| (graph.block_index, graph.request_slot) == expected)
    }

    pub(in crate::commands::real_full) fn run_block_post_dispatch(
        &mut self,
        block_index: usize,
        request_slot: usize,
    ) -> Result<DeepseekV4DsparkProposalDeviceBuffers> {
        anyhow::ensure!(
            self.block_post_dispatch_graphs_prepared() && block_index + 1 < DSPARK_BLOCKS,
            "integrated dSpark inter-block post-dispatch graphs are not prepared"
        );
        let proposal = self.proposal_request_buffers(request_slot)?;
        let graph = self
            .block_post_dispatch_graphs
            .iter()
            .find(|graph| {
                graph.block_index == block_index && graph.request_slot == request_slot
            })
            .with_context(|| {
                format!(
                    "integrated dSpark block {block_index} post-dispatch graph slot {request_slot} is missing"
                )
            })?;
        graph.graph.validate_before_launch()?;
        unsafe {
            self.library
                .cuda_graph_launch(graph.graph.as_ptr(), self.stream.as_ptr())
        }
        .with_context(|| {
            format!(
                "launching integrated dSpark block {block_index} post-dispatch graph slot {request_slot}"
            )
        })?;
        self.stream.synchronize().with_context(|| {
            format!(
                "synchronizing integrated dSpark block {block_index} post-dispatch slot {request_slot}"
            )
        })?;
        Ok(proposal)
    }

    pub(in crate::commands::real_full) fn prepare_terminal_collapse_graphs(
        &mut self,
        rms_eps: f32,
        hc_eps: f32,
    ) -> Result<()> {
        anyhow::ensure!(
            rms_eps.is_finite() && rms_eps > 0.0 && hc_eps.is_finite() && hc_eps > 0.0,
            "integrated dSpark terminal collapse epsilons must be finite and positive"
        );
        if !self.terminal_collapse_graphs.is_empty() {
            anyhow::ensure!(
                self.terminal_collapse_graphs_prepared(),
                "integrated dSpark terminal collapse graph set is partially initialized"
            );
            return Ok(());
        }
        let workspace = self.entry_buffers()?.block_workspace;
        let weights = self.terminal_collapse_weights()?;
        anyhow::ensure!(
            [
                workspace.device_id,
                weights.hc_fn.device_id,
                weights.hc_scale.device_id,
                weights.hc_base.device_id,
                weights.norm_weight.device_id,
            ]
            .into_iter()
            .all(|device| device == 0),
            "integrated dSpark terminal collapse requires every weight and arena view on GPU0"
        );
        let stream = self.stream.as_ptr();
        for request_slot in 0..self.plan.max_batch {
            let proposal = self.proposal_request_buffers(request_slot)?;
            let pre_dispatch = self.block_pre_dispatch_buffers(request_slot)?;
            let buffers = dspark_terminal_collapse_python_buffers(
                workspace,
                pre_dispatch.ffn_delta,
                proposal,
                weights,
            );
            let kwargs = [
                ("variant", PythonKernelArg::Str(self.plan.variant)),
                ("max_batch", PythonKernelArg::Usize(self.plan.max_batch)),
                (
                    "max_main_rows",
                    PythonKernelArg::Usize(self.plan.max_main_rows),
                ),
                ("rms_eps", PythonKernelArg::F64(rms_eps.into())),
                ("hc_eps", PythonKernelArg::F64(hc_eps.into())),
                ("norm_eps", PythonKernelArg::F64(rms_eps.into())),
            ];
            launch_python_graph_capture(PythonGraphCaptureLaunch {
                module: DSPARK_CAPTURE_MODULE,
                function: DSPARK_TERMINAL_PREPARE_FUNCTION,
                cuda_stream: stream,
                buffers: &buffers,
                kwargs: &kwargs,
            })
            .with_context(|| {
                format!("preparing integrated dSpark terminal collapse slot {request_slot}")
            })?;
            self.stream.synchronize().with_context(|| {
                format!("synchronizing prepared dSpark terminal collapse slot {request_slot}")
            })?;
            unsafe {
                self.library
                    .cuda_graph_begin_capture(stream)
                    .with_context(|| {
                        format!("beginning dSpark terminal collapse slot {request_slot} capture")
                    })?;
            }
            let captured = launch_python_graph_capture(PythonGraphCaptureLaunch {
                module: DSPARK_CAPTURE_MODULE,
                function: DSPARK_TERMINAL_CAPTURE_FUNCTION,
                cuda_stream: stream,
                buffers: &buffers,
                kwargs: &kwargs,
            });
            if let Err(error) = captured {
                if let Ok(capture) = unsafe { self.library.cuda_graph_end_capture(stream) } {
                    let _ = unsafe { self.library.cuda_graph_exec_destroy(capture) };
                }
                return Err(error).with_context(|| {
                    format!("capturing integrated dSpark terminal collapse slot {request_slot}")
                });
            }
            let capture = unsafe {
                self.library
                    .cuda_graph_end_capture_retained(stream)
                    .with_context(|| {
                        format!("ending dSpark terminal collapse slot {request_slot} capture")
                    })?
            };
            let graph = CoordinatorCudaCapturedGraph::new(self.library, capture)?;
            graph.validate_before_launch()?;
            anyhow::ensure!(
                graph.memcpy_node_count == 0,
                "integrated dSpark terminal collapse slot {request_slot} unexpectedly captured memcpy nodes"
            );
            self.terminal_collapse_graphs
                .push(DeepseekV4DsparkTerminalCollapseGraph {
                    request_slot,
                    graph,
                });
        }
        Ok(())
    }

    pub(in crate::commands::real_full) fn terminal_collapse_graphs_prepared(&self) -> bool {
        self.terminal_collapse_graphs.len() == self.plan.max_batch
            && self
                .terminal_collapse_graphs
                .iter()
                .enumerate()
                .all(|(request_slot, graph)| graph.request_slot == request_slot)
    }

    pub(in crate::commands::real_full) fn run_terminal_collapse(
        &mut self,
        request_slot: usize,
    ) -> Result<DeepseekV4DsparkProposalDeviceBuffers> {
        anyhow::ensure!(
            self.terminal_collapse_graphs_prepared(),
            "integrated dSpark terminal collapse graphs are not prepared"
        );
        let proposal = self.proposal_request_buffers(request_slot)?;
        let graph = self
            .terminal_collapse_graphs
            .iter()
            .find(|graph| graph.request_slot == request_slot)
            .with_context(|| {
                format!("integrated dSpark terminal collapse graph slot {request_slot} is missing")
            })?;
        graph.graph.validate_before_launch()?;
        unsafe {
            self.library
                .cuda_graph_launch(graph.graph.as_ptr(), self.stream.as_ptr())
        }
        .with_context(|| {
            format!("launching integrated dSpark terminal collapse graph slot {request_slot}")
        })?;
        self.stream.synchronize().with_context(|| {
            format!("synchronizing integrated dSpark terminal collapse slot {request_slot}")
        })?;
        Ok(proposal)
    }

    fn launch_terminal_head_native(
        &self,
        request_bucket: usize,
        weights: DeepseekV4DsparkTerminalHeadWeights,
    ) -> Result<()> {
        let proposal = self.proposal_buffers()?;
        let terminal = self.terminal_buffers()?;
        unsafe {
            self.library.cuda_dspark_terminal_greedy_bf16_async(
                proposal.normalized_hidden,
                proposal.collapsed_hidden,
                weights.shared_head,
                weights.markov_w1,
                weights.markov_w2,
                weights.confidence,
                terminal.active_slot_ids,
                terminal.anchor_token_ids,
                terminal.compact_normalized_hidden,
                terminal.shared_logits,
                terminal.markov_embeddings,
                terminal.markov_logits,
                terminal.output_token_ids,
                terminal.confidence,
                request_bucket,
                self.plan.max_batch,
                self.plan.hidden,
                self.plan.vocab,
                self.plan.markov_rank,
                self.stream.as_ptr(),
            )
        }
        .with_context(|| {
            format!("launching integrated dSpark joint terminal head bucket {request_bucket}")
        })
    }

    pub(in crate::commands::real_full) fn prepare_terminal_head_graphs(&mut self) -> Result<()> {
        if !self.terminal_head_graphs.is_empty() {
            anyhow::ensure!(
                self.terminal_head_graphs_prepared(),
                "integrated dSpark terminal head graph set is partially initialized"
            );
            return Ok(());
        }
        let weights = self.terminal_head_weights()?;
        let proposal = self.proposal_buffers()?;
        let terminal = self.terminal_buffers()?;
        anyhow::ensure!(
            [
                proposal.normalized_hidden.device_id,
                proposal.collapsed_hidden.device_id,
                terminal.compact_normalized_hidden.device_id,
                terminal.shared_logits.device_id,
                terminal.markov_logits.device_id,
                terminal.markov_embeddings.device_id,
                terminal.confidence.device_id,
                terminal.active_slot_ids.device_id,
                terminal.anchor_token_ids.device_id,
                terminal.output_token_ids.device_id,
                weights.shared_head.device_id,
                weights.markov_w1.device_id,
                weights.markov_w2.device_id,
                weights.confidence.device_id,
            ]
            .into_iter()
            .all(|device| device == 0),
            "integrated dSpark joint terminal head requires all storage and weights on GPU0"
        );
        let stream = self.stream.as_ptr();
        for request_bucket in dspark_terminal_request_buckets(self.plan.max_batch) {
            for request in 0..request_bucket {
                self.terminal_active_slots_host[request] = request as u32;
                self.terminal_anchors_host[request] = 0;
            }
            self.library.copy_h2d(
                terminal.active_slot_ids,
                u32_bytes(&self.terminal_active_slots_host[..request_bucket]),
            )?;
            self.library.copy_h2d(
                terminal.anchor_token_ids,
                u32_bytes(&self.terminal_anchors_host[..request_bucket]),
            )?;
            // Warm the thread-local cuBLAS handle and exact bucket shape before
            // stream capture. The arena contents are intentionally unspecified
            // during startup; only pointer and launch topology are retained.
            self.launch_terminal_head_native(request_bucket, weights)?;
            self.stream.synchronize().with_context(|| {
                format!("warming integrated dSpark terminal bucket {request_bucket}")
            })?;
            unsafe {
                self.library
                    .cuda_graph_begin_capture(stream)
                    .with_context(|| {
                        format!("beginning dSpark terminal bucket {request_bucket} capture")
                    })?;
            }
            let captured = self.launch_terminal_head_native(request_bucket, weights);
            if let Err(error) = captured {
                if let Ok(capture) = unsafe { self.library.cuda_graph_end_capture(stream) } {
                    let _ = unsafe { self.library.cuda_graph_exec_destroy(capture) };
                }
                return Err(error).with_context(|| {
                    format!("capturing integrated dSpark terminal bucket {request_bucket}")
                });
            }
            let capture = unsafe {
                self.library
                    .cuda_graph_end_capture_retained(stream)
                    .with_context(|| {
                        format!("ending dSpark terminal bucket {request_bucket} capture")
                    })?
            };
            let graph = CoordinatorCudaCapturedGraph::new(self.library, capture)?;
            graph.validate_before_launch()?;
            anyhow::ensure!(
                graph.memcpy_node_count == 0 && graph.memset_node_count == 0,
                "integrated dSpark terminal bucket {request_bucket} captured host-transfer nodes"
            );
            self.terminal_head_graphs
                .push(DeepseekV4DsparkTerminalHeadGraph {
                    request_bucket,
                    graph,
                });
        }
        Ok(())
    }

    pub(in crate::commands::real_full) fn terminal_head_graphs_prepared(&self) -> bool {
        let expected = dspark_terminal_request_buckets(self.plan.max_batch);
        self.terminal_head_graphs.len() == expected.len()
            && self
                .terminal_head_graphs
                .iter()
                .zip(expected)
                .all(|(graph, request_bucket)| graph.request_bucket == request_bucket)
    }

    pub(in crate::commands::real_full) fn run_terminal_head_batch(
        &mut self,
        active_slots: &[usize],
        anchor_token_ids: &[usize],
    ) -> Result<DeepseekV4DsparkTerminalBatchOutput> {
        anyhow::ensure!(
            self.terminal_head_graphs_prepared(),
            "integrated dSpark terminal head graphs are not prepared"
        );
        anyhow::ensure!(
            !active_slots.is_empty()
                && active_slots.len() == anchor_token_ids.len()
                && active_slots.len() <= self.plan.max_batch,
            "integrated dSpark terminal batch slots/anchors must have one to {} matching entries",
            self.plan.max_batch
        );
        for (request, (&slot, &anchor)) in active_slots.iter().zip(anchor_token_ids).enumerate() {
            anyhow::ensure!(
                slot < self.plan.max_batch && anchor < self.plan.vocab,
                "integrated dSpark terminal request {request} has slot {slot} or anchor {anchor} outside geometry {}/{}",
                self.plan.max_batch,
                self.plan.vocab
            );
            anyhow::ensure!(
                !active_slots[..request].contains(&slot),
                "integrated dSpark terminal active slot {slot} is duplicated"
            );
        }
        let graph_index = self
            .terminal_head_graphs
            .iter()
            .position(|graph| graph.request_bucket >= active_slots.len())
            .context("integrated dSpark terminal batch has no captured width bucket")?;
        let request_bucket = self.terminal_head_graphs[graph_index].request_bucket;
        let pad_slot = active_slots[0];
        let pad_anchor = anchor_token_ids[0];
        for request in 0..request_bucket {
            let slot = active_slots.get(request).copied().unwrap_or(pad_slot);
            let anchor = anchor_token_ids.get(request).copied().unwrap_or(pad_anchor);
            self.terminal_active_slots_host[request] =
                u32::try_from(slot).context("integrated dSpark terminal slot exceeds u32")?;
            self.terminal_anchors_host[request] =
                u32::try_from(anchor).context("integrated dSpark terminal anchor exceeds u32")?;
        }
        let terminal = self.terminal_buffers()?;
        let stream = self.stream.as_ptr();
        unsafe {
            self.library.copy_h2d_async(
                terminal.active_slot_ids,
                u32_bytes(&self.terminal_active_slots_host[..request_bucket]),
                stream,
            )?;
            self.library.copy_h2d_async(
                terminal.anchor_token_ids,
                u32_bytes(&self.terminal_anchors_host[..request_bucket]),
                stream,
            )?;
            self.library.cuda_graph_launch(
                self.terminal_head_graphs[graph_index].graph.as_ptr(),
                stream,
            )?;
        }
        let output_values = request_bucket
            .checked_mul(DSPARK_PROPOSAL_ROWS_PER_REQUEST + 1)
            .context("integrated dSpark terminal output readback count overflow")?;
        let confidence_values = request_bucket
            .checked_mul(DSPARK_PROPOSAL_ROWS_PER_REQUEST)
            .context("integrated dSpark confidence readback count overflow")?;
        let output_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                self.terminal_output_tokens_host.as_mut_ptr().cast::<u8>(),
                output_values * std::mem::size_of::<u32>(),
            )
        };
        let confidence_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                self.terminal_confidence_host.as_mut_ptr().cast::<u8>(),
                confidence_values * std::mem::size_of::<f32>(),
            )
        };
        unsafe {
            self.library
                .copy_d2h_async(output_bytes, terminal.output_token_ids, stream)?;
            self.library
                .copy_d2h_async(confidence_bytes, terminal.confidence, stream)?;
        }
        self.stream
            .synchronize()
            .context("synchronizing integrated dSpark terminal batch readback")?;
        let active_requests = active_slots.len();
        let mut proposal_token_ids =
            Vec::with_capacity(active_requests * DSPARK_PROPOSAL_ROWS_PER_REQUEST);
        let mut conditional_confidence =
            Vec::with_capacity(active_requests * DSPARK_PROPOSAL_ROWS_PER_REQUEST);
        for request in 0..active_requests {
            for position in 0..DSPARK_PROPOSAL_ROWS_PER_REQUEST {
                proposal_token_ids.push(
                    self.terminal_output_tokens_host[(position + 1) * request_bucket + request],
                );
                conditional_confidence
                    .push(self.terminal_confidence_host[position * request_bucket + request]);
            }
        }
        Ok(DeepseekV4DsparkTerminalBatchOutput {
            active_requests,
            proposal_token_ids,
            conditional_confidence,
        })
    }

    pub(in crate::commands::real_full) fn prepare_block_sparse_pre_dispatch(
        &mut self,
        catalog: &TensorCatalog,
    ) -> Result<()> {
        anyhow::ensure!(
            matches!(
                (self.plan.variant, catalog.facts.variant),
                ("flash", ds41rt_core::ModelVariant::Flash) | ("pro", ds41rt_core::ModelVariant::Pro)
            ) && self.plan.hidden == catalog.facts.hidden_size,
            "integrated dSpark {:?}/{} sparse profile does not match its {}-wide device plan",
            catalog.facts.variant,
            catalog.facts.hidden_size,
            self.plan.hidden,
        );
        if !self.block_sparse_weights.is_empty() {
            anyhow::ensure!(
                self.block_sparse_weights.len() == DSPARK_BLOCKS,
                "integrated dSpark sparse pre-dispatch weights are partially initialized"
            );
            return Ok(());
        }
        for block_index in 0..DSPARK_BLOCKS {
            let logical_layer_id = catalog
                .facts
                .num_hidden_layers
                .checked_add(block_index)
                .context("integrated dSpark logical sparse layer ID overflow")?;
            let weights = preload_ds4_sparse_layer_resident_weights(catalog, logical_layer_id)
                .with_context(|| {
                    format!(
                    "preloading integrated dSpark block {block_index} sparse coordinator weights"
                )
                })?;
            anyhow::ensure!(
                !weights.hash_routing,
                "integrated dSpark block {block_index} must use learned global routing"
            );
            self.block_sparse_weights.push(weights);
        }
        Ok(())
    }

    pub(in crate::commands::real_full) fn block_sparse_pre_dispatch_prepared(&self) -> bool {
        self.block_sparse_weights.len() == DSPARK_BLOCKS
            && self
                .block_sparse_weights
                .iter()
                .all(|weights| !weights.hash_routing)
    }

    pub(in crate::commands::real_full) fn run_block_sparse_pre_dispatch(
        &mut self,
        block_index: usize,
        request_slot: usize,
    ) -> Result<DeepseekV4DsparkBlockPreDispatchBuffers> {
        anyhow::ensure!(
            self.block_attention_graphs_prepared() && self.block_sparse_pre_dispatch_prepared(),
            "integrated dSpark sparse pre-dispatch is not prepared"
        );
        let weights = self
            .block_sparse_weights
            .get(block_index)
            .with_context(|| format!("integrated dSpark sparse block {block_index} is missing"))?;
        let buffers = self.block_pre_dispatch_buffers(request_slot)?;
        ds4_sparse_block_pre_dispatch_fixed_input_into(
            weights,
            buffers.dispatch_hidden,
            DSPARK_PROPOSAL_ROWS_PER_REQUEST,
            None,
            buffers.sparse,
            self.stream.as_ptr(),
        )
        .with_context(|| {
            format!("running integrated dSpark block {block_index} global router and shared expert")
        })?;
        self.stream.synchronize().with_context(|| {
            format!("synchronizing integrated dSpark block {block_index} sparse pre-dispatch")
        })?;
        Ok(buffers)
    }

    pub(in crate::commands::real_full) fn read_block_tp4_dispatch_input(
        &self,
        buffers: DeepseekV4DsparkBlockPreDispatchBuffers,
        routed_experts: usize,
    ) -> Result<DeepseekV4DsparkTp4DispatchInput> {
        let exact_geometry = matches!(
            (self.plan.variant, self.plan.hidden, routed_experts),
            ("flash", 4_096, 256) | ("pro", 7_168, 384)
        );
        anyhow::ensure!(
            exact_geometry,
            "integrated dSpark TP4 dispatch requires exact Flash 4096/256 or Pro 7168/384 geometry; got {}/{}/{}",
            self.plan.variant,
            self.plan.hidden,
            routed_experts,
        );
        let rows = DSPARK_PROPOSAL_ROWS_PER_REQUEST;
        let hidden_bytes = rows
            .checked_mul(self.plan.hidden)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("integrated dSpark dispatch hidden byte count overflow")?;
        let route_values = rows
            .checked_mul(6)
            .context("integrated dSpark dispatch route count overflow")?;
        let route_bytes = route_values
            .checked_mul(std::mem::size_of::<u32>())
            .context("integrated dSpark dispatch route byte count overflow")?;
        anyhow::ensure!(
            buffers.dispatch_hidden.bytes == hidden_bytes
                && buffers.sparse.router.indices.bytes == route_bytes
                && buffers.sparse.router.weights.bytes == route_bytes
                && buffers.sparse.shared.output.bytes == hidden_bytes
                && buffers.ffn_delta.bytes == hidden_bytes
                && [
                    buffers.dispatch_hidden.device_id,
                    buffers.sparse.router.indices.device_id,
                    buffers.sparse.router.weights.device_id,
                    buffers.sparse.shared.output.device_id,
                    buffers.ffn_delta.device_id,
                ]
                .into_iter()
                .all(|device| device == 0),
            "integrated dSpark TP4 dispatch buffers lost the fixed five-row GPU0 ABI"
        );
        let mut hidden_bf16 = vec![0_u8; hidden_bytes];
        let mut route_index_bytes = vec![0_u8; route_bytes];
        let mut route_weight_bytes = vec![0_u8; route_bytes];
        self.library
            .copy_d2h(&mut hidden_bf16, buffers.dispatch_hidden)
            .context("reading integrated dSpark normalized dispatch hidden")?;
        self.library
            .copy_d2h(&mut route_index_bytes, buffers.sparse.router.indices)
            .context("reading integrated dSpark global route indices")?;
        self.library
            .copy_d2h(&mut route_weight_bytes, buffers.sparse.router.weights)
            .context("reading integrated dSpark global route weights")?;
        let route_indices = route_index_bytes
            .chunks_exact(std::mem::size_of::<u32>())
            .map(|bytes| u32::from_ne_bytes(bytes.try_into().expect("u32 route chunk")))
            .collect::<Vec<_>>();
        let route_weights = route_weight_bytes
            .chunks_exact(std::mem::size_of::<f32>())
            .map(|bytes| f32::from_ne_bytes(bytes.try_into().expect("f32 route chunk")))
            .collect::<Vec<_>>();
        anyhow::ensure!(
            route_indices.len() == route_values
                && route_weights.len() == route_values
                && route_indices
                    .iter()
                    .all(|expert| (*expert as usize) < routed_experts)
                && route_weights
                    .iter()
                    .all(|weight| weight.is_finite() && *weight >= 0.0),
            "integrated dSpark router produced an invalid global top-6 route image"
        );
        for (row_index, row_routes) in route_indices.chunks_exact(6).enumerate() {
            for (route_offset, expert) in row_routes.iter().enumerate() {
                anyhow::ensure!(
                    !row_routes[..route_offset].contains(expert),
                    "integrated dSpark router repeated expert {expert} in row {row_index}"
                );
            }
            let expected_scale = if self.plan.variant == "pro" { 2.5 } else { 1.5 };
            let weight_sum = route_weights[row_index * 6..(row_index + 1) * 6]
                .iter()
                .sum::<f32>();
            anyhow::ensure!(
                (weight_sum - expected_scale).abs() <= 1.0e-4,
                "integrated dSpark {} router row {row_index} weights sum to {weight_sum}, expected {expected_scale}",
                self.plan.variant,
            );
        }
        Ok(DeepseekV4DsparkTp4DispatchInput {
            rows,
            hidden_bf16,
            route_indices,
            route_weights,
            shared_delta: buffers.sparse.shared.output,
            ffn_delta: buffers.ffn_delta,
        })
    }

    /// Retain one request's shared-expert result in the dead-before-terminal
    /// slot-major normalization arena. This lets every active request finish
    /// routing before one joint strict-TP4 dispatch is issued for the block.
    pub(in crate::commands::real_full) fn stage_joint_block_shared_delta(
        &self,
        active_request_index: usize,
        input: &DeepseekV4DsparkTp4DispatchInput,
    ) -> Result<()> {
        anyhow::ensure!(
            input.rows == DSPARK_PROPOSAL_ROWS_PER_REQUEST
                && active_request_index < self.plan.max_batch,
            "integrated dSpark joint shared-delta slot {active_request_index} has {} rows for max batch {}",
            input.rows,
            self.plan.max_batch,
        );
        let request_bytes = input
            .rows
            .checked_mul(self.plan.hidden)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("integrated dSpark joint shared-delta byte count overflow")?;
        let joint = self.proposal_buffers()?.normalized_hidden;
        let destination = device_buffer_byte_view(
            joint,
            active_request_index
                .checked_mul(request_bytes)
                .context("integrated dSpark joint shared-delta offset overflow")?,
            request_bytes,
            "integrated dSpark joint shared-expert delta slot",
        )?;
        self.library
            .copy_d2d(destination, input.shared_delta, request_bytes)
            .context("retaining integrated dSpark joint shared-expert delta")
    }

    /// Return disjoint slot-major views for one jointly issued request cohort.
    /// C4 uses two of these ranges concurrently so the second pair's
    /// coordinator graphs can run while the first pair is executing on the
    /// Sparks. Keeping the range in the device view prevents either pair from
    /// overwriting the other's shared-expert input or reduced routed delta.
    pub(in crate::commands::real_full) fn joint_block_tp4_dispatch_buffers_for_range(
        &self,
        active_request_range: std::ops::Range<usize>,
    ) -> Result<(Ds41rtDeviceBuffer, Ds41rtDeviceBuffer)> {
        let span = dspark_joint_request_range_byte_span(
            active_request_range,
            self.plan.max_batch,
            self.plan.hidden,
        )?;
        let proposal = self.proposal_buffers()?;
        let terminal = self.terminal_buffers()?;
        Ok((
            device_buffer_byte_view(
                proposal.normalized_hidden,
                span.start,
                span.len(),
                "integrated dSpark joint shared-expert batch",
            )?,
            device_buffer_byte_view(
                terminal.compact_normalized_hidden,
                span.start,
                span.len(),
                "integrated dSpark joint reduced FFN batch",
            )?,
        ))
    }

    fn load_joint_block_ffn_delta(
        &self,
        active_request_index: usize,
        request_slot: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            active_request_index < self.plan.max_batch && request_slot < self.plan.max_batch,
            "integrated dSpark joint FFN source {active_request_index} or slot {request_slot} exceeds max batch {}",
            self.plan.max_batch,
        );
        let request_bytes = DSPARK_PROPOSAL_ROWS_PER_REQUEST
            .checked_mul(self.plan.hidden)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("integrated dSpark joint FFN request bytes overflow")?;
        let joint = self.terminal_buffers()?.compact_normalized_hidden;
        let source = device_buffer_byte_view(
            joint,
            active_request_index
                .checked_mul(request_bytes)
                .context("integrated dSpark joint FFN source offset overflow")?,
            request_bytes,
            "integrated dSpark joint reduced FFN request",
        )?;
        let destination = self.block_pre_dispatch_buffers(request_slot)?.ffn_delta;
        self.library
            .copy_d2d(destination, source, request_bytes)
            .context("restoring integrated dSpark request FFN delta after joint dispatch")
    }

    pub(in crate::commands::real_full) fn run_block_post_dispatch_from_joint(
        &mut self,
        block_index: usize,
        active_request_index: usize,
        request_slot: usize,
    ) -> Result<DeepseekV4DsparkProposalDeviceBuffers> {
        self.load_joint_block_ffn_delta(active_request_index, request_slot)?;
        self.run_block_post_dispatch(block_index, request_slot)
    }

    pub(in crate::commands::real_full) fn run_terminal_collapse_from_joint(
        &mut self,
        active_request_index: usize,
        request_slot: usize,
    ) -> Result<DeepseekV4DsparkProposalDeviceBuffers> {
        self.load_joint_block_ffn_delta(active_request_index, request_slot)?;
        self.run_terminal_collapse(request_slot)
    }

    pub(in crate::commands::real_full) fn project_and_prime_target_taps_in_chunks(
        &mut self,
        target_hidden_taps: [&DeviceBf16Output; 3],
        target_row_start: usize,
        committed_rows: usize,
        absolute_context_start: usize,
        request_slot: usize,
        rope_theta: f32,
        mut consume: impl FnMut(DeepseekV4DsparkProjectedMainChunk) -> Result<()>,
    ) -> Result<()> {
        anyhow::ensure!(
            self.entry_projection_graphs_prepared() && self.prompt_prime_graphs_prepared(),
            "integrated dSpark entry and prompt-prime graphs are not prepared"
        );
        anyhow::ensure!(
            committed_rows > 0,
            "integrated dSpark entry projection requires at least one committed row"
        );
        anyhow::ensure!(
            request_slot < self.plan.max_batch,
            "integrated dSpark request slot {request_slot} exceeds max batch {}",
            self.plan.max_batch
        );
        anyhow::ensure!(
            rope_theta.is_finite() && rope_theta > 0.0,
            "integrated dSpark RoPE theta must be finite and positive, got {rope_theta}"
        );
        let target_row_end = target_row_start
            .checked_add(committed_rows)
            .context("integrated dSpark target row range overflow")?;
        let hidden_row_bytes = self
            .plan
            .hidden
            .checked_mul(std::mem::size_of::<u16>())
            .context("integrated dSpark hidden row byte count overflow")?;
        let concat_row_bytes = hidden_row_bytes
            .checked_mul(target_hidden_taps.len())
            .context("integrated dSpark target-tap concat row byte count overflow")?;
        for (tap_index, tap) in target_hidden_taps.iter().enumerate() {
            anyhow::ensure!(
                tap.rows >= target_row_end
                    && tap.values_per_row == self.plan.hidden
                    && tap.buffer().device_id == 0,
                "integrated dSpark target tap {tap_index} must cover rows {target_row_start}..{target_row_end} at width {} on GPU0, got {}x{} on device {}",
                self.plan.hidden,
                tap.rows,
                tap.values_per_row,
                tap.buffer().device_id,
            );
        }
        let entry = self.entry_buffers()?;
        let stream = self.stream.as_ptr();
        let execution = (|| -> Result<()> {
            for (tap_index, tap) in target_hidden_taps.iter().enumerate() {
                tap.wait_ready_on_stream(stream).with_context(|| {
                    format!("waiting for integrated dSpark target tap {tap_index}")
                })?;
            }
            let mut consumed_rows = 0usize;
            while consumed_rows < committed_rows {
                let rows = dspark_entry_chunk_rows(
                    (committed_rows - consumed_rows).min(self.plan.max_main_rows),
                )?;
                let graph_rows = rows;
                let source_row_start = target_row_start
                    .checked_add(consumed_rows)
                    .context("integrated dSpark source row offset overflow")?;
                let source_offset = source_row_start
                    .checked_mul(hidden_row_bytes)
                    .context("integrated dSpark source byte offset overflow")?;
                for (tap_index, tap) in target_hidden_taps.iter().enumerate() {
                    let source_view_bytes = tap
                        .buffer()
                        .bytes
                        .checked_sub(source_offset)
                        .context("integrated dSpark target tap source offset exceeds buffer")?;
                    let source = device_buffer_byte_view(
                        tap.buffer(),
                        source_offset,
                        source_view_bytes,
                        "integrated dSpark target tap source",
                    )?;
                    let destination_offset = tap_index
                        .checked_mul(hidden_row_bytes)
                        .context("integrated dSpark concat column offset overflow")?;
                    let destination_view_bytes = entry
                        .target_tap_concat
                        .bytes
                        .checked_sub(destination_offset)
                        .context("integrated dSpark concat column offset exceeds target buffer")?;
                    let destination = device_buffer_byte_view(
                        entry.target_tap_concat,
                        destination_offset,
                        destination_view_bytes,
                        "integrated dSpark target-tap concat destination",
                    )?;
                    unsafe {
                        self.library.copy_d2d_2d_async(
                            destination,
                            concat_row_bytes,
                            source,
                            hidden_row_bytes,
                            hidden_row_bytes,
                            rows,
                            stream,
                        )
                    }
                    .with_context(|| {
                        format!(
                            "staging integrated dSpark target tap {tap_index} rows {source_row_start}..{}",
                            source_row_start + rows
                        )
                    })?;
                }
                let graph = self
                    .entry_graphs
                    .iter()
                    .find(|graph| graph.rows == graph_rows)
                    .with_context(|| {
                        format!("integrated dSpark entry graph bucket {graph_rows} is missing")
                    })?;
                graph.graph.validate_before_launch()?;
                unsafe {
                    self.library
                        .cuda_graph_launch(graph.graph.as_ptr(), stream)
                }
                .with_context(|| {
                    format!(
                        "launching integrated dSpark entry graph bucket {graph_rows} for {rows} rows"
                    )
                })?;
                let absolute_position_start = absolute_context_start
                    .checked_add(consumed_rows)
                    .context("integrated dSpark absolute prompt position overflow")?;
                unsafe {
                    self.library.cuda_ds4_dspark_prompt_metadata_async(
                        entry.prompt_positions,
                        entry.prompt_main_slots,
                        entry.prompt_cos_sin,
                        absolute_position_start,
                        request_slot,
                        rows,
                        rope_theta,
                        stream,
                    )
                }
                .with_context(|| {
                    format!(
                        "launching dSpark prompt metadata for positions {absolute_position_start}..{}",
                        absolute_position_start + rows
                    )
                })?;
                for block_index in 0..DSPARK_BLOCKS {
                    let graph = self
                        .prompt_prime_graphs
                        .iter()
                        .find(|graph| graph.block_index == block_index && graph.rows == rows)
                        .with_context(|| {
                            format!(
                                "integrated dSpark prompt graph block {block_index} bucket {rows} is missing"
                            )
                        })?;
                    graph.graph.validate_before_launch()?;
                    unsafe {
                        self.library
                            .cuda_graph_launch(graph.graph.as_ptr(), stream)
                    }
                    .with_context(|| {
                        format!(
                            "launching integrated dSpark prompt graph block {block_index} bucket {rows}"
                        )
                    })?;
                }
                let projected_bytes = rows
                    .checked_mul(hidden_row_bytes)
                    .context("integrated dSpark projected chunk byte count overflow")?;
                consume(DeepseekV4DsparkProjectedMainChunk {
                    target_row_start: source_row_start,
                    rows,
                    graph_rows,
                    projected_target_main: device_buffer_byte_view(
                        entry.projected_target_main,
                        0,
                        projected_bytes,
                        "integrated dSpark projected target-main chunk",
                    )?,
                    stream,
                })?;
                consumed_rows = consumed_rows
                    .checked_add(rows)
                    .context("integrated dSpark consumed row count overflow")?;
            }
            Ok(())
        })();
        let synchronized = self
            .stream
            .synchronize()
            .context("synchronizing integrated dSpark entry projection chunks");
        match (execution, synchronized) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(error), Ok(())) => Err(error),
            (Ok(()), Err(error)) => Err(error),
            (Err(error), Err(sync_error)) => Err(error.context(format!(
                "synchronizing the failed integrated dSpark entry projection also failed: {sync_error:#}"
            ))),
        }
    }
}

fn preload_dspark_block0_hc_attn_lane_sum(catalog: &TensorCatalog, hidden: usize) -> Result<()> {
    let source_columns = DSPARK_PROPOSAL_HC_MULT
        .checked_mul(hidden)
        .context("integrated dSpark block-zero HC source width overflow")?;
    let source_values = DSPARK_HC_MIXES
        .checked_mul(source_columns)
        .context("integrated dSpark block-zero HC source values overflow")?;
    let source_bytes = source_values
        .checked_mul(std::mem::size_of::<f32>())
        .context("integrated dSpark block-zero HC source bytes overflow")?;
    let reduced_values = DSPARK_HC_MIXES
        .checked_mul(hidden)
        .context("integrated dSpark block-zero HC lane-sum values overflow")?;
    let reduced_bytes = reduced_values
        .checked_mul(std::mem::size_of::<f32>())
        .context("integrated dSpark block-zero HC lane-sum bytes overflow")?;
    let mut source = vec![0_u8; source_bytes];
    let summary = read_tensor_bytes_into(catalog, DSPARK_BLOCK0_HC_ATTN_FN, &mut source)
        .context("reading integrated dSpark block-zero attention HC function")?;
    anyhow::ensure!(
        summary.dtype == DType::F32
            && summary.shape == [DSPARK_HC_MIXES, source_columns]
            && summary.bytes_read as usize == source_bytes,
        "integrated dSpark block-zero attention HC function must be F32 [{DSPARK_HC_MIXES}, {source_columns}]"
    );
    preload_resident_weight_from_host_staging(
        DSPARK_BLOCK0_HC_ATTN_FN_LANE_SUM,
        reduced_bytes,
        "integrated dSpark block-zero lane-summed attention HC function",
        |staging| dspark_sum_repeated_hc_input_lanes(&source, staging, hidden),
    )
    .context("preloading integrated dSpark block-zero lane-summed attention HC function")
}

fn dspark_sum_repeated_hc_input_lanes(
    source: &[u8],
    destination: &mut [u8],
    hidden: usize,
) -> Result<()> {
    let source_columns = DSPARK_PROPOSAL_HC_MULT
        .checked_mul(hidden)
        .context("dSpark HC lane-sum source width overflow")?;
    let expected_source_bytes = DSPARK_HC_MIXES
        .checked_mul(source_columns)
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .context("dSpark HC lane-sum source bytes overflow")?;
    let expected_destination_bytes = DSPARK_HC_MIXES
        .checked_mul(hidden)
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .context("dSpark HC lane-sum destination bytes overflow")?;
    anyhow::ensure!(
        hidden > 0
            && source.len() == expected_source_bytes
            && destination.len() == expected_destination_bytes,
        "dSpark HC lane-sum buffers do not match 24 x 4 x {hidden} F32"
    );
    for mix in 0..DSPARK_HC_MIXES {
        for column in 0..hidden {
            let mut sum = 0.0_f32;
            for lane in 0..DSPARK_PROPOSAL_HC_MULT {
                let source_index = mix * source_columns + lane * hidden + column;
                let source_offset = source_index * std::mem::size_of::<f32>();
                sum += f32::from_le_bytes(
                    source[source_offset..source_offset + 4]
                        .try_into()
                        .expect("validated f32 source slice"),
                );
            }
            let destination_index = mix * hidden + column;
            let destination_offset = destination_index * std::mem::size_of::<f32>();
            destination[destination_offset..destination_offset + 4]
                .copy_from_slice(&sum.to_le_bytes());
        }
    }
    Ok(())
}

fn dspark_entry_chunk_rows(remaining: usize) -> Result<usize> {
    anyhow::ensure!(
        remaining > 0,
        "dSpark entry chunk remaining row count must be positive"
    );
    DSPARK_ENTRY_GRAPH_ROWS
        .into_iter()
        .rev()
        .find(|bucket| *bucket <= remaining)
        .context("dSpark entry graph set has no exact chunk bucket")
}

fn dspark_entry_python_buffers(
    entry: DeepseekV4DsparkEntryDeviceBuffers,
    main_proj_weight: Ds41rtDeviceBuffer,
    main_proj_scale: Ds41rtDeviceBuffer,
) -> [PythonDeviceBufferArg<'static>; 5] {
    [
        python_device_buffer("target_tap_concat", entry.target_tap_concat),
        python_device_buffer("main_proj_weight", main_proj_weight),
        python_device_buffer("main_proj_scale", main_proj_scale),
        python_device_buffer("projection_scratch", entry.projection_scratch),
        python_device_buffer("projected_target_main", entry.projected_target_main),
    ]
}

fn dspark_prompt_python_buffers(
    entry: DeepseekV4DsparkEntryDeviceBuffers,
    main_kv_cache: Ds41rtDeviceBuffer,
    wkv_weight: Ds41rtDeviceBuffer,
    wkv_scale: Ds41rtDeviceBuffer,
    kv_norm_weight: Ds41rtDeviceBuffer,
) -> [PythonDeviceBufferArg<'static>; 9] {
    [
        python_device_buffer("projected_target_main", entry.projected_target_main),
        python_device_buffer("positions", entry.prompt_positions),
        python_device_buffer("main_slots", entry.prompt_main_slots),
        python_device_buffer("cos_sin_cache", entry.prompt_cos_sin),
        python_device_buffer("main_kv_cache", main_kv_cache),
        python_device_buffer("producer_scratch", entry.kv_producer_scratch),
        python_device_buffer("wkv_weight", wkv_weight),
        python_device_buffer("wkv_scale", wkv_scale),
        python_device_buffer("kv_norm_weight", kv_norm_weight),
    ]
}

fn dspark_proposal_entry_python_buffers(
    workspace: Ds41rtDeviceBuffer,
    proposal_embedding: Ds41rtDeviceBuffer,
    proposal: DeepseekV4DsparkProposalDeviceBuffers,
    hc_fn: Ds41rtDeviceBuffer,
    hc_scale: Ds41rtDeviceBuffer,
    hc_base: Ds41rtDeviceBuffer,
    norm_weight: Ds41rtDeviceBuffer,
) -> [PythonDeviceBufferArg<'static>; 10] {
    [
        python_device_buffer("scratch", workspace),
        python_device_buffer("residual_input", proposal_embedding),
        python_device_buffer("residual_output", proposal.residual_pong),
        python_device_buffer("normalized_output", proposal.collapsed_hidden),
        python_device_buffer("post_output", proposal.post_ping),
        python_device_buffer("comb_output", proposal.comb_ping),
        python_device_buffer("hc_fn", hc_fn),
        python_device_buffer("hc_scale", hc_scale),
        python_device_buffer("hc_base", hc_base),
        python_device_buffer("norm_weight", norm_weight),
    ]
}

fn dspark_block_attention_python_buffers(
    workspace: Ds41rtDeviceBuffer,
    main_kv_cache: Ds41rtDeviceBuffer,
    proposal: DeepseekV4DsparkProposalDeviceBuffers,
    weights: DeepseekV4DsparkBlockAttentionWeights,
) -> [PythonDeviceBufferArg<'static>; 31] {
    [
        python_device_buffer("workspace", workspace),
        python_device_buffer("hidden_states", proposal.collapsed_hidden),
        python_device_buffer("positions", proposal.positions),
        python_device_buffer("main_slots", proposal.main_slots),
        python_device_buffer("cos_sin_cache", proposal.cos_sin),
        python_device_buffer("main_kv_cache", main_kv_cache),
        python_device_buffer("selected_indices", proposal.selected_indices),
        python_device_buffer("selected_lengths", proposal.selected_lengths),
        python_device_buffer("residual", proposal.residual_pong),
        python_device_buffer("prev_post", proposal.post_ping),
        python_device_buffer("prev_comb", proposal.comb_ping),
        python_device_buffer("residual_out", proposal.residual_ping),
        python_device_buffer("post_out", proposal.post_pong),
        python_device_buffer("comb_out", proposal.comb_pong),
        python_device_buffer("wq_a_weight", weights.wq_a_weight),
        python_device_buffer("wq_a_scale", weights.wq_a_scale),
        python_device_buffer("wq_b_weight", weights.wq_b_weight),
        python_device_buffer("wq_b_scale", weights.wq_b_scale),
        python_device_buffer("wkv_weight", weights.wkv_weight),
        python_device_buffer("wkv_scale", weights.wkv_scale),
        python_device_buffer("q_norm_weight", weights.q_norm_weight),
        python_device_buffer("kv_norm_weight", weights.kv_norm_weight),
        python_device_buffer("wo_a_weight", weights.wo_a_weight),
        python_device_buffer("wo_a_scale", weights.wo_a_scale),
        python_device_buffer("wo_b_weight", weights.wo_b_weight),
        python_device_buffer("wo_b_scale", weights.wo_b_scale),
        python_device_buffer("attn_sink", weights.attn_sink),
        python_device_buffer("hc_fn", weights.hc_fn),
        python_device_buffer("hc_scale", weights.hc_scale),
        python_device_buffer("hc_base", weights.hc_base),
        python_device_buffer("norm_weight", weights.norm_weight),
    ]
}

fn dspark_block_post_dispatch_python_buffers(
    workspace: Ds41rtDeviceBuffer,
    ffn_delta: Ds41rtDeviceBuffer,
    proposal: DeepseekV4DsparkProposalDeviceBuffers,
    weights: DeepseekV4DsparkBlockPostDispatchWeights,
) -> [PythonDeviceBufferArg<'static>; 13] {
    [
        python_device_buffer("workspace", workspace),
        python_device_buffer("ffn_delta", ffn_delta),
        python_device_buffer("residual", proposal.residual_ping),
        python_device_buffer("prev_post", proposal.post_pong),
        python_device_buffer("prev_comb", proposal.comb_pong),
        python_device_buffer("residual_out", proposal.residual_pong),
        python_device_buffer("hidden_states", proposal.collapsed_hidden),
        python_device_buffer("post_out", proposal.post_ping),
        python_device_buffer("comb_out", proposal.comb_ping),
        python_device_buffer("next_hc_fn", weights.next_hc_fn),
        python_device_buffer("next_hc_scale", weights.next_hc_scale),
        python_device_buffer("next_hc_base", weights.next_hc_base),
        python_device_buffer("next_norm_weight", weights.next_norm_weight),
    ]
}

fn dspark_terminal_collapse_python_buffers(
    workspace: Ds41rtDeviceBuffer,
    ffn_delta: Ds41rtDeviceBuffer,
    proposal: DeepseekV4DsparkProposalDeviceBuffers,
    weights: DeepseekV4DsparkTerminalCollapseWeights,
) -> [PythonDeviceBufferArg<'static>; 12] {
    [
        python_device_buffer("workspace", workspace),
        python_device_buffer("ffn_delta", ffn_delta),
        python_device_buffer("residual", proposal.residual_ping),
        python_device_buffer("prev_post", proposal.post_pong),
        python_device_buffer("prev_comb", proposal.comb_pong),
        python_device_buffer("terminal_residual", proposal.residual_pong),
        python_device_buffer("collapsed_hidden", proposal.collapsed_hidden),
        python_device_buffer("normalized_hidden", proposal.normalized_hidden),
        python_device_buffer("hc_head_fn", weights.hc_fn),
        python_device_buffer("hc_head_scale", weights.hc_scale),
        python_device_buffer("hc_head_base", weights.hc_base),
        python_device_buffer("norm_weight", weights.norm_weight),
    ]
}

fn dspark_block_attention_geometry(variant: &str) -> Result<(usize, usize, usize)> {
    match variant {
        "flash" => Ok((1_024, 64, 8)),
        "pro" => Ok((1_536, 128, 16)),
        other => anyhow::bail!(
            "integrated dSpark block-attention variant must be flash or pro, got {other:?}"
        ),
    }
}

fn python_device_buffer(
    name: &'static str,
    buffer: Ds41rtDeviceBuffer,
) -> PythonDeviceBufferArg<'static> {
    PythonDeviceBufferArg {
        name,
        ptr: buffer.ptr,
        bytes: buffer.bytes,
        device_id: buffer.device_id,
        flags: buffer.flags,
    }
}

impl Drop for DeepseekV4DsparkDeviceStorage {
    fn drop(&mut self) {
        let _ = self.library.free_device_buffer(&mut self.persistent_kv);
        let _ = self.library.free_device_buffer(&mut self.arena);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entry_chunk_rows_selects_largest_exact_bucket() {
        for (remaining, expected) in [
            (1, 1),
            (2, 1),
            (15, 1),
            (16, 16),
            (17, 16),
            (129, 128),
            (2_047, 1_024),
            (2_048, 2_048),
            (4_097, 2_048),
        ] {
            assert_eq!(dspark_entry_chunk_rows(remaining).unwrap(), expected);
        }
        assert!(dspark_entry_chunk_rows(0).is_err());
    }

    #[test]
    fn long_entry_projection_is_chunkable_without_rejecting_total_rows() {
        let mut remaining = 4_097usize;
        let mut chunks = Vec::new();
        while remaining > 0 {
            let rows = dspark_entry_chunk_rows(remaining.min(2_048)).unwrap();
            chunks.push(rows);
            remaining -= rows;
        }
        assert_eq!(chunks, vec![2_048, 2_048, 1]);
    }

    #[test]
    fn terminal_graph_buckets_preserve_joint_batch_widths() {
        assert_eq!(dspark_terminal_request_buckets(1), vec![1]);
        assert_eq!(dspark_terminal_request_buckets(16), vec![1, 2, 4, 8, 16]);
        assert_eq!(dspark_terminal_request_buckets(13), vec![1, 2, 4, 8, 13]);
    }

    #[test]
    fn c4_joint_pair_buffers_are_adjacent_and_disjoint() -> Result<()> {
        let hidden = 4_096;
        let first = dspark_joint_request_range_byte_span(0..2, 4, hidden)?;
        let second = dspark_joint_request_range_byte_span(2..4, 4, hidden)?;
        assert_eq!(first.start, 0);
        assert_eq!(first.end, second.start);
        assert_eq!(first.len(), 2 * 5 * hidden * std::mem::size_of::<u16>());
        assert_eq!(second.len(), first.len());
        assert!(dspark_joint_request_range_byte_span(2..2, 4, hidden).is_err());
        assert!(dspark_joint_request_range_byte_span(3..5, 4, hidden).is_err());
        Ok(())
    }

    #[test]
    fn repeated_hc_input_lanes_are_summed_once_for_proposal_entry() -> Result<()> {
        let hidden = 2usize;
        let mut source = Vec::new();
        for mix in 0..DSPARK_HC_MIXES {
            for lane in 0..DSPARK_PROPOSAL_HC_MULT {
                for column in 0..hidden {
                    source.extend_from_slice(
                        &((mix * 100 + lane * 10 + column) as f32).to_le_bytes(),
                    );
                }
            }
        }
        let mut destination = vec![0_u8; DSPARK_HC_MIXES * hidden * 4];
        dspark_sum_repeated_hc_input_lanes(&source, &mut destination, hidden)?;
        for mix in 0..DSPARK_HC_MIXES {
            for column in 0..hidden {
                let offset = (mix * hidden + column) * 4;
                let actual = f32::from_le_bytes(
                    destination[offset..offset + 4]
                        .try_into()
                        .expect("four-byte test slice"),
                );
                let expected = 4.0 * (mix * 100 + column) as f32 + 60.0;
                assert_eq!(actual, expected);
            }
        }
        Ok(())
    }

    #[test]
    fn flash_proposal_storage_plan_matches_python_arena_contract() -> Result<()> {
        let plan = plan_deepseek_v4_dspark_device_storage("flash", 16, 2_048)?;
        assert_eq!(plan.arena_bytes, 168_077_824);
        assert_eq!(plan.persistent_kv_bytes, 3 * 16 * 149_760);
        assert_eq!(plan.proposal.draft_token_ids_offset, 27_787_264);
        assert_eq!(plan.proposal.draft_token_ids_bytes, 320);
        assert_eq!(plan.proposal.residual_ping_offset, 27_788_032);
        assert_eq!(plan.proposal.residual_ping_bytes, 2_621_440);
        assert_eq!(plan.proposal.residual_pong_offset, 30_409_472);
        assert_eq!(plan.proposal.residual_pong_bytes, 2_621_440);
        assert_eq!(plan.proposal.collapsed_hidden_offset, 33_030_912);
        assert_eq!(plan.proposal.collapsed_hidden_bytes, 655_360);
        assert_eq!(plan.proposal.normalized_hidden_offset, 33_686_272);
        assert_eq!(plan.proposal.normalized_hidden_bytes, 655_360);
        assert_eq!(plan.terminal.compact_normalized_hidden_offset, 34_341_632);
        assert_eq!(plan.terminal.compact_normalized_hidden_bytes, 655_360);
        assert_eq!(plan.terminal.shared_logits_offset, 34_996_992);
        assert_eq!(plan.terminal.shared_logits_bytes, 41_369_600);
        assert_eq!(plan.terminal.markov_logits_offset, 76_366_592);
        assert_eq!(plan.terminal.markov_logits_bytes, 8_273_920);
        assert_eq!(plan.terminal.markov_embeddings_offset, 84_640_512);
        assert_eq!(plan.terminal.markov_embeddings_bytes, 40_960);
        assert_eq!(plan.terminal.confidence_offset, 84_681_472);
        assert_eq!(plan.terminal.confidence_bytes, 320);
        assert_eq!(plan.terminal.active_slot_ids_offset, 84_681_984);
        assert_eq!(plan.terminal.active_slot_ids_bytes, 64);
        assert_eq!(plan.terminal.anchor_token_ids_offset, 84_682_240);
        assert_eq!(plan.terminal.anchor_token_ids_bytes, 64);
        assert_eq!(plan.terminal.output_token_ids_offset, 84_682_496);
        assert_eq!(plan.terminal.output_token_ids_bytes, 384);
        assert_eq!(plan.proposal.positions_offset, 84_640_512);
        assert_eq!(plan.proposal.positions_bytes, 320);
        assert_eq!(plan.proposal.main_slots_offset, 84_640_832);
        assert_eq!(plan.proposal.main_slots_bytes, 320);
        assert_eq!(plan.proposal.cos_sin_offset, 84_641_152);
        assert_eq!(plan.proposal.cos_sin_bytes, 20_480);
        assert_eq!(plan.proposal.post_ping_offset, 84_661_632);
        assert_eq!(plan.proposal.post_ping_bytes, 1_280);
        assert_eq!(plan.proposal.comb_ping_offset, 84_662_912);
        assert_eq!(plan.proposal.comb_ping_bytes, 5_120);
        assert_eq!(plan.proposal.post_pong_offset, 84_668_032);
        assert_eq!(plan.proposal.post_pong_bytes, 1_280);
        assert_eq!(plan.proposal.comb_pong_offset, 84_669_312);
        assert_eq!(plan.proposal.comb_pong_bytes, 5_120);
        assert_eq!(plan.proposal.selected_indices_offset, 84_683_008);
        assert_eq!(plan.proposal.selected_indices_bytes, 42_560);
        assert_eq!(plan.proposal.selected_lengths_offset, 84_725_760);
        assert_eq!(plan.proposal.selected_lengths_bytes, 320);
        assert_eq!(plan.block.route_indices_offset, 80_543_744);
        assert_eq!(plan.block.route_indices_bytes, 1_920);
        assert_eq!(plan.block.route_scores_offset, 80_545_792);
        assert_eq!(plan.block.route_scores_bytes, 83_840);
        assert_eq!(plan.block.route_weights_offset, 80_629_760);
        assert_eq!(plan.block.route_weights_bytes, 1_920);
        assert_eq!(plan.block.shared_gate_offset, 80_631_808);
        assert_eq!(plan.block.shared_gate_bytes, 32_768);
        assert_eq!(plan.block.shared_up_offset, 80_664_576);
        assert_eq!(plan.block.shared_up_bytes, 32_768);
        assert_eq!(plan.block.shared_activated_offset, 80_697_344);
        assert_eq!(plan.block.shared_activated_bytes, 32_768);
        assert_eq!(plan.block.shared_delta_offset, 80_730_112);
        assert_eq!(plan.block.shared_delta_bytes, 655_360);
        assert_eq!(plan.block.reduction_f32_offset, 81_385_472);
        assert_eq!(plan.block.reduction_f32_bytes, 1_310_720);
        assert_eq!(plan.block.ffn_delta_offset, 82_696_192);
        assert_eq!(plan.block.ffn_delta_bytes, 655_360);
        Ok(())
    }

    #[test]
    fn flash_native_nvfp4_dspark_storage_uses_exact_request_local_pages() -> Result<()> {
        let plan = plan_deepseek_v4_dspark_device_storage_with_cache_format(
            "flash",
            16,
            2_048,
            DeepseekV4KvCacheFormat::Nvfp4,
        )?;

        assert_eq!(plan.cache_format, "nvfp4");
        assert_eq!(plan.packed_page_bytes, 110_592);
        assert_eq!(plan.persistent_kv_bytes, DSPARK_BLOCKS * 16 * 110_592);
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn request_local_packed_kv_snapshot_roundtrips_all_three_blocks() -> Result<()> {
        let plan = plan_deepseek_v4_dspark_device_storage("flash", 16, 2_048)?;
        let storage = DeepseekV4DsparkDeviceStorage::new(plan)?;
        let request_bytes = DSPARK_BLOCKS * plan.packed_page_bytes;
        let expected = (0..request_bytes)
            .map(|index| ((index * 31 + 17) % 251) as u8)
            .collect::<Vec<_>>();
        storage.restore_request_kv(7, &expected)?;
        assert_eq!(storage.snapshot_request_kv(7)?, expected);
        Ok(())
    }

    #[test]
    fn pro_proposal_storage_plan_preserves_five_row_joint_sparse_geometry() -> Result<()> {
        let max_batch = 16_usize;
        let proposal_rows = max_batch * DSPARK_PROPOSAL_ROWS_PER_REQUEST;
        let plan = plan_deepseek_v4_dspark_device_storage("pro", max_batch, 2_048)?;
        assert_eq!(plan.variant, "pro");
        assert_eq!(plan.hidden, 7_168);
        assert_eq!(plan.vocab, 129_280);
        assert_eq!(plan.markov_rank, 512);
        assert_eq!(
            plan.proposal.collapsed_hidden_bytes,
            proposal_rows * plan.hidden * std::mem::size_of::<u16>()
        );
        assert_eq!(
            plan.proposal.normalized_hidden_bytes,
            proposal_rows * plan.hidden * std::mem::size_of::<u16>()
        );
        assert_eq!(
            plan.block.route_indices_bytes,
            proposal_rows * 6 * std::mem::size_of::<u32>()
        );
        assert_eq!(
            plan.block.route_scores_bytes,
            proposal_rows * (6 + 384) * std::mem::size_of::<f32>()
        );
        assert_eq!(
            plan.block.route_weights_bytes,
            plan.block.route_indices_bytes
        );
        assert_eq!(
            plan.block.shared_gate_bytes,
            8 * 3_072 * std::mem::size_of::<u16>()
        );
        assert_eq!(plan.block.shared_up_bytes, plan.block.shared_gate_bytes);
        assert_eq!(
            plan.block.shared_activated_bytes,
            plan.block.shared_gate_bytes
        );
        assert_eq!(
            plan.block.shared_delta_bytes,
            proposal_rows * plan.hidden * std::mem::size_of::<u16>()
        );
        assert_eq!(plan.block.ffn_delta_bytes, plan.block.shared_delta_bytes);
        assert_eq!(
            plan.block.reduction_f32_bytes,
            proposal_rows * plan.hidden * std::mem::size_of::<f32>()
        );
        assert!(plan.arena_bytes > plan.block.ffn_delta_offset + plan.block.ffn_delta_bytes);
        Ok(())
    }
}
