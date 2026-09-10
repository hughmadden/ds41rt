//! Strict contract for the pinned official checkpoint; no legacy defaults.
use anyhow::{ensure, Context, Result};
use serde::Deserialize;
use serde_json::Value;
use std::{fs::File, io::Read, path::Path};

pub const OFFICIAL_V41_MODEL_ID: &str = "deepseek-ai/DeepSeek-V4.1-Flash";
pub const OFFICIAL_V41_REVISION: &str = "df42c109f1defefcbfcedbe7d905718a12266e40";
const OFFICIAL_CONFIG: &str = include_str!("official-v41-config.json");
const MAX_CONFIG_BYTES: u64 = 1024 * 1024;

/// Constructed only after every official architecture and quantization field agrees.
#[derive(Debug, Clone)]
pub struct OfficialV41Config(V41CheckpointConfig);

impl OfficialV41Config {
    pub fn from_json(model_id: &str, bytes: &[u8]) -> Result<Self> {
        ensure!(
            model_id == OFFICIAL_V41_MODEL_ID,
            "only {OFFICIAL_V41_MODEL_ID} is supported, got {model_id}"
        );
        ensure!(
            bytes.len() as u64 <= MAX_CONFIG_BYTES,
            "V4.1 config exceeds one MiB"
        );
        let actual: Value =
            serde_json::from_slice(bytes).context("parsing official V4.1 config")?;
        let expected: Value =
            serde_json::from_str(OFFICIAL_CONFIG).expect("embedded official config");
        validate_fields("config", &expected, &actual)?;
        Ok(Self(
            serde_json::from_value(actual).context("decoding official V4.1 config")?,
        ))
    }

    pub fn architectures(&self) -> &[String] {
        &self.0.architectures
    }
    pub fn model_type(&self) -> &str {
        &self.0.model_type
    }
    pub fn dtype(&self) -> &str {
        &self.0.dtype
    }
    pub fn transformers_version(&self) -> &str {
        &self.0.transformers_version
    }
    pub fn target_compress_ratios(&self) -> &[usize] {
        &self.0.text_config.compress_ratios[..self.0.text_config.num_hidden_layers]
    }
    pub fn dspark_compress_ratios(&self) -> &[usize] {
        &self.0.text_config.compress_ratios[self.0.text_config.num_hidden_layers..]
    }
    pub fn text(&self) -> &V41TextConfig {
        &self.0.text_config
    }
    pub fn vision(&self) -> &V41VisionConfig {
        &self.0.vision_config
    }
    pub fn quantization(&self) -> &V41QuantizationConfig {
        &self.0.quantization_config
    }
    pub fn image_token_id(&self) -> usize {
        self.0.image_token_id
    }
    pub fn bos_token_id(&self) -> usize {
        self.0.bos_token_id
    }
    pub fn eos_token_id(&self) -> usize {
        self.0.eos_token_id
    }
    pub fn pad_token_id(&self) -> usize {
        self.0.pad_token_id
    }
}

pub fn read_official_v41_config(model_id: &str, snapshot: &Path) -> Result<OfficialV41Config> {
    let path = snapshot.join("config.json");
    let mut bytes = Vec::new();
    File::open(&path)
        .with_context(|| format!("opening {}", path.display()))?
        .take(MAX_CONFIG_BYTES + 1)
        .read_to_end(&mut bytes)?;
    OfficialV41Config::from_json(model_id, &bytes)
}

fn validate_fields(path: &str, expected: &Value, actual: &Value) -> Result<()> {
    if let Some(fields) = expected.as_object() {
        let supplied = actual
            .as_object()
            .with_context(|| format!("{path} must be an object"))?;
        for (key, wanted) in fields {
            let name = format!("{path}.{key}");
            let found = supplied
                .get(key)
                .with_context(|| format!("missing {name}"))?;
            // Library version is provenance metadata, not model execution geometry.
            if name == "config.transformers_version" {
                ensure!(found.is_string(), "{name} must be a string");
            } else {
                validate_fields(&name, wanted, found)?;
            }
        }
        for key in supplied.keys() {
            ensure!(fields.contains_key(key), "unsupported {path}.{key}");
        }
    } else {
        ensure!(
            expected == actual,
            "unsupported {path}: expected {expected}, got {actual}"
        );
    }
    Ok(())
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct V41QuantizationConfig {
    pub quant_method: String,
    pub activation_scheme: String,
    pub weight_block_size: Vec<usize>,
    pub scale_fmt: String,
    pub expert_dtype: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct V41RopeScaling {
    pub rope_type: String,
    pub factor: usize,
    pub beta_fast: usize,
    pub beta_slow: usize,
    pub original_max_position_embeddings: usize,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct V41TextConfig {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub moe_intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub q_lora_rank: usize,
    pub o_lora_rank: usize,
    pub o_groups: usize,
    pub hidden_act: String,
    pub swiglu_limit: f64,
    pub rms_norm_eps: f64,
    pub attention_bias: bool,
    pub attention_dropout: f64,
    pub initializer_range: f64,
    pub use_cache: bool,
    pub tie_word_embeddings: bool,
    pub max_position_embeddings: usize,
    pub rope_theta: usize,
    pub rope_scaling: V41RopeScaling,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    pub scoring_func: String,
    pub topk_method: String,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f64,
    pub sliding_window: usize,
    pub compress_ratios: Vec<usize>,
    pub compress_rope_theta: usize,
    pub kv_source_layer_ids: Vec<usize>,
    pub index_source_layer_ids: Vec<usize>,
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    pub candidate_source_layer_id: usize,
    pub candidate_topk_blocks: usize,
    pub candidate_block_size: usize,
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    pub hc_eps: f64,
    pub engram_layer_ids: Vec<usize>,
    pub engram_num_embeddings: Vec<usize>,
    pub engram_max_ngram_size: usize,
    pub engram_vocab_size: usize,
    pub engram_n_heads: usize,
    pub engram_head_dim: usize,
    pub engram_pad_token_id: usize,
    pub engram_compressed_vocab_size: usize,
    pub num_nextn_predict_layers: usize,
    pub dspark_block_size: usize,
    pub dspark_noise_token_id: usize,
    pub dspark_target_layer_ids: Vec<usize>,
    pub dspark_markov_rank: usize,
    pub dspark_n_routed_experts: usize,
    pub dspark_num_experts_per_tok: usize,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct V41VisionConfig {
    pub model_type: String,
    pub num_hidden_layers: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub patch_size: usize,
    pub rope_theta: usize,
    pub downsample_ratio: usize,
    pub max_image_tokens: usize,
    pub min_pixels: usize,
    pub max_wh_ratio: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct V41CheckpointConfig {
    pub architectures: Vec<String>,
    pub model_type: String,
    pub dtype: String,
    pub transformers_version: String,
    pub bos_token_id: usize,
    pub eos_token_id: usize,
    pub pad_token_id: usize,
    pub image_token_id: usize,
    pub quantization_config: V41QuantizationConfig,
    pub text_config: V41TextConfig,
    pub vision_config: V41VisionConfig,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_nested_architecture_has_distinct_dspark_and_ced_geometry() {
        let config =
            OfficialV41Config::from_json(OFFICIAL_V41_MODEL_ID, OFFICIAL_CONFIG.as_bytes())
                .unwrap();
        let text = config.text();
        assert_eq!((text.n_routed_experts, text.num_experts_per_tok), (384, 6));
        assert_eq!(
            (
                text.dspark_n_routed_experts,
                text.dspark_num_experts_per_tok
            ),
            (128, 3)
        );
        assert_eq!(text.dspark_target_layer_ids, [37, 38, 39]);
        assert_eq!(text.kv_source_layer_ids, [2, 8, 14, 20]);
        assert_eq!(&text.compress_ratios[..2], [0, 0]);
        assert!(text.compress_ratios[2..20].iter().all(|v| *v == 2));
        assert!(config.target_compress_ratios()[20..]
            .iter()
            .all(|v| *v == 1));
        assert_eq!(config.dspark_compress_ratios(), [0, 0, 0]);
        assert_eq!(config.quantization().weight_block_size, [32, 32]);
        assert_eq!(config.vision().patch_size, 14);
    }

    #[test]
    fn rejects_legacy_or_incomplete_architecture_instead_of_defaulting() {
        for (pointer, replacement) in [
            ("/model_type", serde_json::json!("deepseek_v4")),
            (
                "/quantization_config/weight_block_size",
                serde_json::json!([128, 128]),
            ),
            (
                "/text_config/dspark_n_routed_experts",
                serde_json::json!(384),
            ),
            (
                "/text_config/engram_num_embeddings",
                serde_json::json!([384006168]),
            ),
            ("/vision_config/num_hidden_layers", serde_json::json!(24)),
        ] {
            let mut value: Value = serde_json::from_str(OFFICIAL_CONFIG).unwrap();
            *value.pointer_mut(pointer).unwrap() = replacement;
            assert!(
                OfficialV41Config::from_json(
                    OFFICIAL_V41_MODEL_ID,
                    &serde_json::to_vec(&value).unwrap()
                )
                .is_err(),
                "{pointer}"
            );
        }
        let mut value: Value = serde_json::from_str(OFFICIAL_CONFIG).unwrap();
        value["text_config"]
            .as_object_mut()
            .unwrap()
            .remove("candidate_source_layer_id");
        let error = OfficialV41Config::from_json(
            OFFICIAL_V41_MODEL_ID,
            &serde_json::to_vec(&value).unwrap(),
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("config.text_config.candidate_source_layer_id"));
        assert!(OfficialV41Config::from_json(
            "deepseek-ai/DeepSeek-V4-Flash",
            OFFICIAL_CONFIG.as_bytes()
        )
        .is_err());
    }
}
