use crate::{
    DEFAULT_MODEL_ID, DS4_COMPRESS_ROPE_THETA, DS4_DSPARK_BLOCK_SIZE, DS4_DSPARK_NOISE_TOKEN_ID,
    DS4_FLASH_COMPRESS_RATIOS, DS4_FLASH_DSPARK_MARKOV_RANK, DS4_FLASH_HIDDEN_SIZE,
    DS4_FLASH_MOE_INTERMEDIATE_SIZE, DS4_FLASH_NUM_HIDDEN_LAYERS, DS4_FLASH_Q_LORA_RANK,
    DS4_FLASH_ROUTED_EXPERTS, DS4_FLASH_TOP_K, DS4_HC_EPS, DS4_HC_MULT, DS4_HC_SINKHORN_ITERS,
    DS4_HEAD_DIM, DS4_INDEX_HEADS, DS4_INDEX_HEAD_DIM, DS4_KV_HEADS, DS4_NUM_HASH_LAYERS,
    DS4_NUM_SHARED_EXPERTS, DS4_ORIGINAL_MAX_POSITION_EMBEDDINGS, DS4_O_LORA_RANK,
    DS4_QK_ROPE_HEAD_DIM, DS4_ROPE_BETA_FAST, DS4_ROPE_BETA_SLOW, DS4_ROPE_SCALING_FACTOR,
    DS4_ROPE_THETA, DS4_SLIDING_WINDOW,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TensorRole {
    RoutedExpert,
    SharedExpert,
    Attention,
    AttentionCompressor,
    AttentionIndexer,
    Router,
    HyperConnection,
    Norm,
    Embedding,
    LmHead,
    DenseMlp,
    /// DeepSeek's integrated dSpark blocks. Checkpoints serialize these under
    /// `mtp.*`, but the role is architectural and deliberately does not reuse
    /// GLM's recurrent-MTP classification.
    Dspark,
    Mtp,
    Quantization,
    Config,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DType {
    Bf16,
    F16,
    F32,
    F8E4M3,
    F8E5M2,
    F8E8M0,
    I8,
    I16,
    I32,
    I64,
    U8,
    F4,
    Unknown(String),
}

impl DType {
    pub fn from_safetensors(value: &str) -> Self {
        match value {
            "BF16" => DType::Bf16,
            "F16" => DType::F16,
            "F32" => DType::F32,
            "F8_E4M3" => DType::F8E4M3,
            "F8_E5M2" => DType::F8E5M2,
            "F8_E8M0" => DType::F8E8M0,
            "I8" => DType::I8,
            "I16" => DType::I16,
            "I32" => DType::I32,
            "I64" => DType::I64,
            "U8" => DType::U8,
            "F4" => DType::F4,
            other => DType::Unknown(other.to_owned()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelVariant {
    Flash,
    Pro,
    Custom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AttentionKind {
    Sliding,
    CompressedSparse,
    HeavilyCompressed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelFacts {
    pub model_id: String,
    pub model_type: String,
    pub variant: ModelVariant,
    pub hidden_size: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    pub num_hidden_layers: usize,
    pub num_hash_layers: usize,
    /// Compatibility field for the imported engine. DeepSeek V4 has no dense
    /// bootstrap layers; its first layers are hash-routed MoE layers.
    pub first_k_dense_replace: usize,
    pub routed_experts: usize,
    pub top_k: usize,
    pub moe_intermediate_size: usize,
    pub shared_experts: usize,
    pub vocab_size: usize,
    pub attention_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub q_lora_rank: usize,
    pub o_lora_rank: usize,
    pub o_groups: usize,
    pub qk_rope_head_dim: usize,
    pub index_head_dim: usize,
    pub index_heads: usize,
    pub index_top_k: usize,
    pub sliding_window: usize,
    pub max_position_embeddings: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_compress_rope_theta")]
    pub compress_rope_theta: f32,
    #[serde(default = "default_rope_scaling_factor")]
    pub rope_scaling_factor: f32,
    #[serde(default = "default_original_max_position_embeddings")]
    pub original_max_position_embeddings: usize,
    #[serde(default = "default_rope_beta_fast")]
    pub rope_beta_fast: f32,
    #[serde(default = "default_rope_beta_slow")]
    pub rope_beta_slow: f32,
    pub hyper_connection_multiplier: usize,
    pub hyper_connection_sinkhorn_iters: usize,
    #[serde(default = "default_hyper_connection_eps")]
    pub hyper_connection_eps: f32,
    pub compress_ratios: Vec<usize>,
    pub routed_scaling_factor: f32,
    pub scoring_function: String,
    pub topk_method: String,
    pub swiglu_limit: f32,
    pub expert_dtype: String,
    pub dspark_block_size: usize,
    pub dspark_markov_rank: usize,
    #[serde(default = "default_dspark_noise_token_id")]
    pub dspark_noise_token_id: usize,
    pub dspark_target_layer_ids: Vec<usize>,
    pub quantization_recipe: String,
}

impl ModelFacts {
    pub fn attention_kind(&self, layer_id: usize) -> Option<AttentionKind> {
        self.compress_ratios
            .get(layer_id)
            .copied()
            .and_then(|ratio| match ratio {
                0 => Some(AttentionKind::Sliding),
                4 => Some(AttentionKind::CompressedSparse),
                128 => Some(AttentionKind::HeavilyCompressed),
                _ => None,
            })
    }

    pub fn total_transformer_blocks(&self) -> usize {
        self.num_hidden_layers + self.dspark_target_layer_ids.len()
    }
}

impl Default for ModelFacts {
    fn default() -> Self {
        Self {
            model_id: DEFAULT_MODEL_ID.to_owned(),
            model_type: "deepseek_v4".to_owned(),
            variant: ModelVariant::Flash,
            hidden_size: DS4_FLASH_HIDDEN_SIZE,
            rms_norm_eps: 1.0e-6,
            num_hidden_layers: DS4_FLASH_NUM_HIDDEN_LAYERS,
            num_hash_layers: DS4_NUM_HASH_LAYERS,
            first_k_dense_replace: 0,
            routed_experts: DS4_FLASH_ROUTED_EXPERTS,
            top_k: DS4_FLASH_TOP_K,
            moe_intermediate_size: DS4_FLASH_MOE_INTERMEDIATE_SIZE,
            shared_experts: DS4_NUM_SHARED_EXPERTS,
            vocab_size: 129_280,
            attention_heads: 64,
            kv_heads: DS4_KV_HEADS,
            head_dim: DS4_HEAD_DIM,
            q_lora_rank: DS4_FLASH_Q_LORA_RANK,
            o_lora_rank: DS4_O_LORA_RANK,
            o_groups: 8,
            qk_rope_head_dim: DS4_QK_ROPE_HEAD_DIM,
            index_head_dim: DS4_INDEX_HEAD_DIM,
            index_heads: DS4_INDEX_HEADS,
            index_top_k: 512,
            sliding_window: DS4_SLIDING_WINDOW,
            max_position_embeddings: 1_048_576,
            rope_theta: DS4_ROPE_THETA,
            compress_rope_theta: DS4_COMPRESS_ROPE_THETA,
            rope_scaling_factor: DS4_ROPE_SCALING_FACTOR,
            original_max_position_embeddings: DS4_ORIGINAL_MAX_POSITION_EMBEDDINGS,
            rope_beta_fast: DS4_ROPE_BETA_FAST,
            rope_beta_slow: DS4_ROPE_BETA_SLOW,
            hyper_connection_multiplier: DS4_HC_MULT,
            hyper_connection_sinkhorn_iters: DS4_HC_SINKHORN_ITERS,
            hyper_connection_eps: DS4_HC_EPS,
            compress_ratios: DS4_FLASH_COMPRESS_RATIOS.to_vec(),
            routed_scaling_factor: 1.5,
            scoring_function: "sqrtsoftplus".to_owned(),
            topk_method: "noaux_tc".to_owned(),
            swiglu_limit: 10.0,
            expert_dtype: "fp4".to_owned(),
            dspark_block_size: DS4_DSPARK_BLOCK_SIZE,
            dspark_markov_rank: DS4_FLASH_DSPARK_MARKOV_RANK,
            dspark_noise_token_id: DS4_DSPARK_NOISE_TOKEN_ID,
            dspark_target_layer_ids: vec![40, 41, 42],
            quantization_recipe: "deepseek_v4_native_fp4_fp8_mixed_v1".to_owned(),
        }
    }
}

fn default_rms_norm_eps() -> f32 {
    1.0e-6
}

fn default_rope_theta() -> f32 {
    DS4_ROPE_THETA
}

fn default_compress_rope_theta() -> f32 {
    DS4_COMPRESS_ROPE_THETA
}

fn default_rope_scaling_factor() -> f32 {
    DS4_ROPE_SCALING_FACTOR
}

fn default_original_max_position_embeddings() -> usize {
    DS4_ORIGINAL_MAX_POSITION_EMBEDDINGS
}

fn default_rope_beta_fast() -> f32 {
    DS4_ROPE_BETA_FAST
}

fn default_rope_beta_slow() -> f32 {
    DS4_ROPE_BETA_SLOW
}

fn default_hyper_connection_eps() -> f32 {
    DS4_HC_EPS
}

fn default_dspark_noise_token_id() -> usize {
    DS4_DSPARK_NOISE_TOKEN_ID
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInfo {
    pub name: String,
    pub file: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub byte_offset: u64,
    pub byte_length: u64,
    pub role: TensorRole,
    pub layer_id: Option<u32>,
    pub expert_id: Option<u32>,
    pub is_quantization_metadata: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorCatalog {
    pub model_id: String,
    pub snapshot_path: String,
    pub facts: ModelFacts,
    pub tensors: Vec<TensorInfo>,
}

impl TensorCatalog {
    pub fn summary_by_role(&self) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for tensor in &self.tensors {
            *counts.entry(format!("{:?}", tensor.role)).or_insert(0) += 1;
        }
        counts
    }

    pub fn content_hash(&self) -> String {
        let encoded = serde_json::to_vec(self).expect("serializing catalog cannot fail");
        let digest = Sha256::digest(encoded);
        format!("{digest:x}")
    }
}
