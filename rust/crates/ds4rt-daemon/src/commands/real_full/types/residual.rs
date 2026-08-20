use serde::Serialize;

pub(in crate::commands::real_full) const REAL_FULL_RESIDUAL_COMPLETION_BLOCKER: &str =
    "live DeepSeek V4 request execution must pass native target-attention, strict TP4 expert, residual, and terminal sampling gates";

#[derive(Debug, Serialize)]
pub(in crate::commands::real_full) struct RealFullResidualStreamDryRun {
    pub(in crate::commands::real_full) status: &'static str,
    pub(in crate::commands::real_full) scope: &'static str,
    pub(in crate::commands::real_full) layer_count: usize,
    pub(in crate::commands::real_full) row_count: usize,
    pub(in crate::commands::real_full) hidden_dim: usize,
    pub(in crate::commands::real_full) hidden_bytes_per_row: usize,
    pub(in crate::commands::real_full) residual_state_bytes: usize,
    pub(in crate::commands::real_full) dense_layers: usize,
    pub(in crate::commands::real_full) sparse_layers: usize,
    pub(in crate::commands::real_full) remote_sparse_layers: usize,
    pub(in crate::commands::real_full) attention_residual_adds: usize,
    pub(in crate::commands::real_full) mlp_residual_adds: usize,
    pub(in crate::commands::real_full) total_residual_adds: usize,
    pub(in crate::commands::real_full) terminal_rows: usize,
    pub(in crate::commands::real_full) terminal_stages: Vec<&'static str>,
    pub(in crate::commands::real_full) final_residual_state_hash: u64,
    pub(in crate::commands::real_full) numeric_kernel_self_test: RealFullResidualKernelSelfTest,
    pub(in crate::commands::real_full) layer_order_verified: bool,
    pub(in crate::commands::real_full) layer_dry_runs: Vec<RealFullResidualLayerDryRun>,
}

#[derive(Debug, Serialize)]
pub(in crate::commands::real_full) struct RealFullResidualKernelSelfTest {
    pub(in crate::commands::real_full) status: &'static str,
    pub(in crate::commands::real_full) scope: &'static str,
    pub(in crate::commands::real_full) layers: usize,
    pub(in crate::commands::real_full) rows: usize,
    pub(in crate::commands::real_full) hidden_dim: usize,
    pub(in crate::commands::real_full) residual_adds: usize,
    pub(in crate::commands::real_full) values_updated: usize,
    pub(in crate::commands::real_full) final_checksum: f32,
    pub(in crate::commands::real_full) expected_checksum: f32,
    pub(in crate::commands::real_full) first_value: f32,
    pub(in crate::commands::real_full) expected_first_value: f32,
    pub(in crate::commands::real_full) last_value: f32,
    pub(in crate::commands::real_full) expected_last_value: f32,
    pub(in crate::commands::real_full) passed: bool,
}

#[derive(Debug, Serialize)]
pub(in crate::commands::real_full) struct RealFullResidualLayerDryRun {
    pub(in crate::commands::real_full) layer_id: usize,
    pub(in crate::commands::real_full) layer_kind: &'static str,
    pub(in crate::commands::real_full) input_rows: usize,
    pub(in crate::commands::real_full) residual_state_bytes: usize,
    pub(in crate::commands::real_full) attention_input_norm: bool,
    pub(in crate::commands::real_full) attention_output_projection: bool,
    pub(in crate::commands::real_full) attention_residual_boundary: bool,
    pub(in crate::commands::real_full) post_attention_norm: bool,
    pub(in crate::commands::real_full) mlp_kind: &'static str,
    pub(in crate::commands::real_full) routed_expert_exchange: bool,
    pub(in crate::commands::real_full) mlp_residual_boundary: bool,
    pub(in crate::commands::real_full) output_rows: usize,
    pub(in crate::commands::real_full) residual_state_hash: u64,
    pub(in crate::commands::real_full) layer_order_verified: bool,
}
