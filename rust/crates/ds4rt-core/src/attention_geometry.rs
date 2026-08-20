use serde::{Deserialize, Serialize};

use crate::{AttentionKind, Ds4rtError, ModelFacts, ModelVariant};

/// Immutable DeepSeek V4 attention dimensions shared by target and dSpark
/// blocks. Per-layer compression behavior lives in [`DeepseekV4AttentionLayerPlan`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeepseekV4AttentionGeometry {
    pub variant: ModelVariant,
    pub hidden_size: usize,
    pub attention_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub nope_head_dim: usize,
    pub rope_head_dim: usize,
    pub q_lora_rank: usize,
    pub o_lora_rank: usize,
    pub o_groups: usize,
    pub index_head_dim: usize,
    pub index_heads: usize,
    pub index_top_k: usize,
    pub sliding_window: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
    pub compress_rope_theta: f32,
    pub rope_scaling_factor: f32,
    pub original_max_position_embeddings: usize,
    pub rope_beta_fast: f32,
    pub rope_beta_slow: f32,
    pub hyper_connection_multiplier: usize,
    pub hyper_connection_sinkhorn_iters: usize,
    pub hyper_connection_eps: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DeepseekV4AttentionLayerSource {
    Target {
        layer_id: usize,
    },
    Dspark {
        block_index: usize,
        target_layer_id: usize,
    },
}

impl DeepseekV4AttentionLayerSource {
    /// Checkpoint namespace for the complete transformer block.
    ///
    /// DeepSeek serializes integrated dSpark blocks under `mtp.*`; this is a
    /// storage prefix only and does not imply GLM-style recurrent MTP.
    pub fn checkpoint_block_prefix(self) -> String {
        match self {
            Self::Target { layer_id } => format!("layers.{layer_id}"),
            Self::Dspark { block_index, .. } => format!("mtp.{block_index}"),
        }
    }

    pub fn target_layer_id(self) -> usize {
        match self {
            Self::Target { layer_id } => layer_id,
            Self::Dspark {
                target_layer_id, ..
            } => target_layer_id,
        }
    }

    pub fn is_dspark(self) -> bool {
        matches!(self, Self::Dspark { .. })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DeepseekV4CompressedSelection {
    None,
    LearnedIndexer { top_k: usize },
    AllCompletedBlocks,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeepseekV4AttentionLayerPlan {
    /// Dense logical ID used by scheduler/KV ownership. dSpark blocks follow
    /// the target layers in checkpoint order.
    pub logical_layer_id: usize,
    pub source: DeepseekV4AttentionLayerSource,
    pub kind: AttentionKind,
    pub compress_ratio: usize,
    pub compressed_selection: DeepseekV4CompressedSelection,
    pub sliding_window: usize,
}

impl DeepseekV4AttentionLayerPlan {
    pub fn uses_compressor(&self) -> bool {
        self.compress_ratio > 0
    }

    pub fn uses_indexer(&self) -> bool {
        matches!(
            self.compressed_selection,
            DeepseekV4CompressedSelection::LearnedIndexer { .. }
        )
    }

    pub fn uses_yarn(&self) -> bool {
        self.uses_compressor()
    }

    /// Position-indexed compressor ring. One history window is followed by an
    /// equally sized speculative guard, so rejected suffixes cannot overwrite
    /// accepted compressor history. Sliding-only layers have no state.
    pub fn compressor_state_rows(&self) -> usize {
        match self.compress_ratio {
            0 => 0,
            4 => 4 * self.compress_ratio,
            ratio => 2 * ratio,
        }
    }

    pub fn completed_compressed_blocks(&self, logical_tokens: usize) -> usize {
        if self.compress_ratio == 0 {
            0
        } else {
            logical_tokens / self.compress_ratio
        }
    }

    /// Persistent attention cache slots, excluding the compressor's FP32
    /// incremental state. The sliding ring is always allocated at full width,
    /// matching the native reference implementation.
    pub fn persistent_kv_slots(&self, max_logical_tokens: usize) -> usize {
        self.sliding_window + self.completed_compressed_blocks(max_logical_tokens)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeepseekV4AttentionPlan {
    pub geometry: DeepseekV4AttentionGeometry,
    pub target_layer_count: usize,
    pub dspark_layer_count: usize,
    pub layers: Vec<DeepseekV4AttentionLayerPlan>,
}

impl DeepseekV4AttentionPlan {
    pub fn from_model_facts(facts: &ModelFacts) -> Result<Self, Ds4rtError> {
        validate_model_facts(facts)?;

        let total_layers = facts.total_transformer_blocks();
        let mut layers = Vec::with_capacity(total_layers);
        for logical_layer_id in 0..total_layers {
            let source = if logical_layer_id < facts.num_hidden_layers {
                DeepseekV4AttentionLayerSource::Target {
                    layer_id: logical_layer_id,
                }
            } else {
                let block_index = logical_layer_id - facts.num_hidden_layers;
                DeepseekV4AttentionLayerSource::Dspark {
                    block_index,
                    target_layer_id: facts.dspark_target_layer_ids[block_index],
                }
            };
            let compress_ratio = facts.compress_ratios[logical_layer_id];
            let (kind, compressed_selection) = match compress_ratio {
                0 => (AttentionKind::Sliding, DeepseekV4CompressedSelection::None),
                4 => (
                    AttentionKind::CompressedSparse,
                    DeepseekV4CompressedSelection::LearnedIndexer {
                        top_k: facts.index_top_k,
                    },
                ),
                128 => (
                    AttentionKind::HeavilyCompressed,
                    DeepseekV4CompressedSelection::AllCompletedBlocks,
                ),
                _ => unreachable!("compression schedule was validated"),
            };
            layers.push(DeepseekV4AttentionLayerPlan {
                logical_layer_id,
                source,
                kind,
                compress_ratio,
                compressed_selection,
                sliding_window: facts.sliding_window,
            });
        }

        Ok(Self {
            geometry: DeepseekV4AttentionGeometry {
                variant: facts.variant,
                hidden_size: facts.hidden_size,
                attention_heads: facts.attention_heads,
                kv_heads: facts.kv_heads,
                head_dim: facts.head_dim,
                nope_head_dim: facts.head_dim - facts.qk_rope_head_dim,
                rope_head_dim: facts.qk_rope_head_dim,
                q_lora_rank: facts.q_lora_rank,
                o_lora_rank: facts.o_lora_rank,
                o_groups: facts.o_groups,
                index_head_dim: facts.index_head_dim,
                index_heads: facts.index_heads,
                index_top_k: facts.index_top_k,
                sliding_window: facts.sliding_window,
                max_position_embeddings: facts.max_position_embeddings,
                rope_theta: facts.rope_theta,
                compress_rope_theta: facts.compress_rope_theta,
                rope_scaling_factor: facts.rope_scaling_factor,
                original_max_position_embeddings: facts.original_max_position_embeddings,
                rope_beta_fast: facts.rope_beta_fast,
                rope_beta_slow: facts.rope_beta_slow,
                hyper_connection_multiplier: facts.hyper_connection_multiplier,
                hyper_connection_sinkhorn_iters: facts.hyper_connection_sinkhorn_iters,
                hyper_connection_eps: facts.hyper_connection_eps,
            },
            target_layer_count: facts.num_hidden_layers,
            dspark_layer_count: facts.dspark_target_layer_ids.len(),
            layers,
        })
    }

    pub fn layer(&self, logical_layer_id: usize) -> Option<&DeepseekV4AttentionLayerPlan> {
        self.layers.get(logical_layer_id)
    }

    pub fn target_layers(&self) -> &[DeepseekV4AttentionLayerPlan] {
        &self.layers[..self.target_layer_count]
    }

    pub fn dspark_layers(&self) -> &[DeepseekV4AttentionLayerPlan] {
        &self.layers[self.target_layer_count..]
    }

    /// Fail closed on the dimensions implemented by SparkInfer's unified DSV4
    /// sparse-MLA and mHC kernels. Flash and the current Pro preview both pass.
    pub fn validate_sparkinfer_sm120_contract(&self) -> Result<(), Ds4rtError> {
        let geometry = &self.geometry;
        ensure_geometry(
            geometry.kv_heads == 1,
            format!(
                "SparkInfer DSV4 requires one KV head, got {}",
                geometry.kv_heads
            ),
        )?;
        ensure_geometry(
            geometry.head_dim == 512
                && geometry.nope_head_dim == 448
                && geometry.rope_head_dim == 64,
            format!(
                "SparkInfer DSV4 requires head/nope/rope 512/448/64, got {}/{}/{}",
                geometry.head_dim, geometry.nope_head_dim, geometry.rope_head_dim
            ),
        )?;
        ensure_geometry(
            geometry.attention_heads % 16 == 0,
            format!(
                "SparkInfer DSV4 requires attention heads divisible by 16, got {}",
                geometry.attention_heads
            ),
        )?;
        ensure_geometry(
            geometry.index_head_dim == 128 && geometry.index_heads == 64,
            format!(
                "SparkInfer DSV4 indexer requires 64x128 heads, got {}x{}",
                geometry.index_heads, geometry.index_head_dim
            ),
        )?;
        ensure_geometry(
            geometry.hyper_connection_multiplier == 4
                && geometry.hyper_connection_sinkhorn_iters == 20,
            format!(
                "SparkInfer mHC requires multiplier/iterations 4/20, got {}/{}",
                geometry.hyper_connection_multiplier, geometry.hyper_connection_sinkhorn_iters
            ),
        )?;
        ensure_geometry(
            matches!(geometry.hidden_size, 4096 | 7168),
            format!(
                "SparkInfer mHC supports hidden size 4096 or 7168, got {}",
                geometry.hidden_size
            ),
        )?;
        Ok(())
    }
}

fn validate_model_facts(facts: &ModelFacts) -> Result<(), Ds4rtError> {
    ensure_geometry(
        facts.model_type == "deepseek_v4",
        format!(
            "expected model_type deepseek_v4, got {:?}",
            facts.model_type
        ),
    )?;
    ensure_geometry(facts.num_hidden_layers > 0, "model has no target layers")?;
    ensure_geometry(
        facts.compress_ratios.len() == facts.total_transformer_blocks(),
        format!(
            "compression schedule has {} entries, expected {} target+dSpark blocks",
            facts.compress_ratios.len(),
            facts.total_transformer_blocks()
        ),
    )?;
    ensure_geometry(
        facts
            .compress_ratios
            .iter()
            .all(|ratio| matches!(ratio, 0 | 4 | 128)),
        format!(
            "compression schedule contains an unsupported ratio: {:?}",
            facts.compress_ratios
        ),
    )?;
    ensure_geometry(
        facts
            .dspark_target_layer_ids
            .iter()
            .all(|layer_id| *layer_id < facts.num_hidden_layers),
        format!(
            "dSpark target layers {:?} exceed {} target layers",
            facts.dspark_target_layer_ids, facts.num_hidden_layers
        ),
    )?;
    ensure_geometry(
        facts.hidden_size > 0
            && facts.attention_heads > 0
            && facts.kv_heads > 0
            && facts.q_lora_rank > 0
            && facts.o_lora_rank > 0,
        "attention projection dimensions must be positive",
    )?;
    ensure_geometry(
        facts.qk_rope_head_dim > 0 && facts.qk_rope_head_dim < facts.head_dim,
        format!(
            "partial-RoPE width {} must be smaller than head width {}",
            facts.qk_rope_head_dim, facts.head_dim
        ),
    )?;
    ensure_geometry(
        facts.o_groups > 0 && facts.attention_heads % facts.o_groups == 0,
        format!(
            "attention heads {} are not divisible by O groups {}",
            facts.attention_heads, facts.o_groups
        ),
    )?;
    ensure_geometry(
        facts.index_heads > 0
            && facts.index_head_dim >= facts.qk_rope_head_dim
            && facts.index_top_k > 0,
        "indexer dimensions and top-k must cover the partial-RoPE width",
    )?;
    ensure_geometry(
        facts.sliding_window > 0 && facts.max_position_embeddings >= facts.sliding_window,
        "sliding window must be positive and fit the model context",
    )?;
    ensure_geometry(
        facts.original_max_position_embeddings > 0
            && facts.original_max_position_embeddings <= facts.max_position_embeddings,
        "YaRN original context must be positive and no larger than model context",
    )?;
    ensure_geometry(
        positive_finite(facts.rms_norm_eps)
            && positive_finite(facts.rope_theta)
            && positive_finite(facts.compress_rope_theta)
            && positive_finite(facts.rope_scaling_factor)
            && positive_finite(facts.rope_beta_fast)
            && positive_finite(facts.rope_beta_slow)
            && positive_finite(facts.hyper_connection_eps),
        "normalization, RoPE/YaRN, and mHC floating-point parameters must be finite and positive",
    )?;
    ensure_geometry(
        facts.rope_beta_fast >= facts.rope_beta_slow,
        format!(
            "YaRN beta_fast {} is smaller than beta_slow {}",
            facts.rope_beta_fast, facts.rope_beta_slow
        ),
    )?;
    ensure_geometry(
        facts.hyper_connection_multiplier > 0 && facts.hyper_connection_sinkhorn_iters > 0,
        "mHC multiplier and Sinkhorn iterations must be positive",
    )?;
    if !facts.dspark_target_layer_ids.is_empty() {
        ensure_geometry(
            facts.dspark_block_size > 0 && facts.dspark_markov_rank > 0,
            "dSpark block size and Markov rank must be positive",
        )?;
        ensure_geometry(
            facts.dspark_noise_token_id < facts.vocab_size,
            format!(
                "dSpark noise token {} exceeds vocabulary size {}",
                facts.dspark_noise_token_id, facts.vocab_size
            ),
        )?;
    }
    Ok(())
}

fn positive_finite(value: f32) -> bool {
    value.is_finite() && value > 0.0
}

fn ensure_geometry(condition: bool, reason: impl Into<String>) -> Result<(), Ds4rtError> {
    if condition {
        Ok(())
    } else {
        Err(Ds4rtError::InvalidDeepseekV4AttentionGeometry {
            reason: reason.into(),
        })
    }
}
