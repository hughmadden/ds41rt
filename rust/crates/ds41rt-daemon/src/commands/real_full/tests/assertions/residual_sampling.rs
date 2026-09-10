use ds41rt_core::{DS4_FLASH_HIDDEN_SIZE, DS4_FLASH_NUM_HIDDEN_LAYERS};

use super::super::super::constants::{
    REAL_FULL_PREFLIGHT_DECODE_ROWS, REAL_FULL_PREFLIGHT_MTP_ROWS, REAL_FULL_PREFLIGHT_PREFILL_ROWS,
};
use super::super::super::types::RealDs4FullPreflightReport;

pub(super) fn assert_residual_sampling_report(report: &RealDs4FullPreflightReport) {
    let residual = &report.residual_stream_dry_run;
    assert_eq!(residual.status, "catalog-ordered-deepseek-v4-residual-plan");
    assert!(residual.scope.contains("DeepSeek V4"));
    assert_eq!(residual.layer_count, DS4_FLASH_NUM_HIDDEN_LAYERS);
    assert_eq!(
        residual.row_count,
        REAL_FULL_PREFLIGHT_PREFILL_ROWS
            + REAL_FULL_PREFLIGHT_MTP_ROWS
            + REAL_FULL_PREFLIGHT_DECODE_ROWS
    );
    assert_eq!(residual.hidden_dim, DS4_FLASH_HIDDEN_SIZE);
    assert_eq!(residual.hidden_bytes_per_row, DS4_FLASH_HIDDEN_SIZE * 2);
    assert_eq!(residual.dense_layers, 0);
    assert_eq!(residual.sparse_layers, DS4_FLASH_NUM_HIDDEN_LAYERS);
    assert_eq!(residual.remote_sparse_layers, DS4_FLASH_NUM_HIDDEN_LAYERS);
    assert_eq!(
        residual.attention_residual_adds,
        DS4_FLASH_NUM_HIDDEN_LAYERS
    );
    assert_eq!(residual.mlp_residual_adds, DS4_FLASH_NUM_HIDDEN_LAYERS);
    assert_eq!(
        residual.total_residual_adds,
        DS4_FLASH_NUM_HIDDEN_LAYERS * 2
    );
    assert_eq!(
        residual.terminal_stages,
        ["final_norm", "lm_head", "full_vocab_sampling"]
    );
    assert!(residual.layer_order_verified);
    assert_eq!(residual.layer_dry_runs.len(), DS4_FLASH_NUM_HIDDEN_LAYERS);
    for (layer_id, layer) in residual.layer_dry_runs.iter().enumerate() {
        assert_eq!(layer.layer_id, layer_id);
        assert_eq!(layer.layer_kind, "sparse-routed-moe");
        assert!(layer.attention_input_norm);
        assert!(layer.attention_output_projection);
        assert!(layer.attention_residual_boundary);
        assert!(layer.post_attention_norm);
        assert!(layer.routed_expert_exchange);
        assert!(layer.mlp_residual_boundary);
        assert!(layer.layer_order_verified);
    }

    let kernel = &residual.numeric_kernel_self_test;
    assert!(kernel.passed);
    assert_eq!(kernel.layers, 3);
    assert_eq!(kernel.residual_adds, 6);
    assert_eq!(kernel.values_updated, 48);
    assert!((kernel.final_checksum - 12.0).abs() < 1.0e-6);

    let sampling = &report.sampling_dry_run;
    assert_eq!(sampling.lm_head_tensor, "lm_head.weight");
    assert_eq!(sampling.vocab_size, 154_880);
    assert_eq!(sampling.hidden_dim, DS4_FLASH_HIDDEN_SIZE);
    assert_eq!(sampling.sampled_rows, REAL_FULL_PREFLIGHT_DECODE_ROWS);
    assert_eq!(sampling.chunk_rows, 1_024);
    assert_eq!(sampling.chunk_count, 152);
    assert_eq!(sampling.final_chunk_rows, 256);
    assert_eq!(sampling.lm_head_bytes, 1_268_776_960);
    assert_eq!(sampling.logical_lm_head_read_bytes, 1_268_776_960);
    assert_eq!(sampling.dot_products, 154_880);
    assert_eq!(sampling.multiply_accumulate_ops, 634_388_480);
    assert!(sampling.covers_full_vocabulary);
    assert!(sampling.requires_numeric_logits);

    let chunk = &sampling.real_lm_head_default_chunk_probe;
    assert_eq!(chunk.status, "error");
    assert_eq!(chunk.hidden_dim, DS4_FLASH_HIDDEN_SIZE);
    assert_eq!(chunk.vocab_size, 154_880);
    assert_eq!(chunk.chunk_rows, 1_024);
    assert!(!chunk.passed);
    assert!(chunk.error.is_some());

    let guarded = &sampling.real_lm_head_full_vocab_probe;
    assert_eq!(guarded.status, "not-run");
    assert!(!guarded.passed);
    assert!(!guarded.uses_full_model_residual);
    assert_eq!(guarded.residual_prefix_values, 0);
    assert_eq!(guarded.residual_after_checksum, None);
    assert_eq!(guarded.opt_in_env, "DS41RT_REAL_FULL_SCORE_LM_HEAD");
}
