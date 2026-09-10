use anyhow::{Context, Result};
use ds41rt_core::{DType, ModelFacts, TensorCatalog, TensorRole};
use std::collections::{BTreeMap, BTreeSet};

use crate::{is_deepseek_v4_exl3_recipe, native_deepseek_v4_attention_tensor_specs};

const NATIVE_DSPARK_FP8_BLOCK: usize = 128;
const NATIVE_FP4_EXPERT_RECIPE: &str = "deepseek_v4_native_fp4_fp8_mixed_v1";
const NATIVE_FP4_TENSORS_PER_EXPERT: usize = 6;
const EXL3_TENSORS_PER_EXPERT: usize = 12;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeDeepseekV4DsparkTensorSpec {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub logical_layer_id: usize,
    pub is_quantization_metadata: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeDeepseekV4DsparkCatalogSummary {
    pub blocks: usize,
    pub target_taps: usize,
    pub proposal_tokens: usize,
    pub coordinator_tensors: usize,
    pub routed_expert_tensors: usize,
    pub coordinator_tensor_bytes: u64,
}

/// Derive the complete coordinator-side contract for DeepSeek V4's integrated
/// dSpark blocks. Routed expert tensors are deliberately counted here and
/// structurally checked by the selected native FP4 or EXL3 expert contract;
/// they use the same strict TP4 serving path as target-model experts.
pub fn native_deepseek_v4_dspark_tensor_specs(
    facts: &ModelFacts,
) -> Result<Vec<NativeDeepseekV4DsparkTensorSpec>> {
    validate_dspark_facts(facts)?;

    let hidden = facts.hidden_size;
    let hc_mult = facts.hyper_connection_multiplier;
    let hc_width = hc_mult
        .checked_mul(hidden)
        .context("native dSpark hyper-connection width overflow")?;
    let hc_mix = hc_mult
        .checked_add(2)
        .and_then(|value| value.checked_mul(hc_mult))
        .context("native dSpark hyper-connection mix width overflow")?;
    let shared_intermediate = facts
        .shared_experts
        .checked_mul(facts.moe_intermediate_size)
        .context("native dSpark shared-expert width overflow")?;
    let block_count = facts.dspark_target_layer_ids.len();
    let final_block = block_count - 1;
    let mut specs = native_deepseek_v4_attention_tensor_specs(facts)?
        .into_iter()
        .filter(|spec| spec.role == TensorRole::Dspark)
        .map(|spec| NativeDeepseekV4DsparkTensorSpec {
            name: spec.name,
            dtype: spec.dtype,
            shape: spec.shape,
            logical_layer_id: spec.logical_layer_id,
            is_quantization_metadata: spec.is_quantization_metadata,
        })
        .collect::<Vec<_>>();

    for block in 0..block_count {
        let prefix = format!("mtp.{block}");
        let logical_layer_id = facts.num_hidden_layers + block;
        push(
            &mut specs,
            &prefix,
            "attn_norm.weight",
            DType::Bf16,
            vec![hidden],
            logical_layer_id,
            false,
        );
        push(
            &mut specs,
            &prefix,
            "ffn_norm.weight",
            DType::Bf16,
            vec![hidden],
            logical_layer_id,
            false,
        );
        push(
            &mut specs,
            &prefix,
            "ffn.gate.weight",
            DType::Bf16,
            vec![facts.routed_experts, hidden],
            logical_layer_id,
            false,
        );
        push(
            &mut specs,
            &prefix,
            "ffn.gate.bias",
            DType::F32,
            vec![facts.routed_experts],
            logical_layer_id,
            false,
        );
        push_fp8_projection(
            &mut specs,
            &prefix,
            "ffn.shared_experts.w1",
            shared_intermediate,
            hidden,
            logical_layer_id,
        );
        push_fp8_projection(
            &mut specs,
            &prefix,
            "ffn.shared_experts.w2",
            hidden,
            shared_intermediate,
            logical_layer_id,
        );
        push_fp8_projection(
            &mut specs,
            &prefix,
            "ffn.shared_experts.w3",
            shared_intermediate,
            hidden,
            logical_layer_id,
        );
        for stem in ["hc_attn", "hc_ffn"] {
            push(
                &mut specs,
                &prefix,
                &format!("{stem}_fn"),
                DType::F32,
                vec![hc_mix, hc_width],
                logical_layer_id,
                false,
            );
            push(
                &mut specs,
                &prefix,
                &format!("{stem}_base"),
                DType::F32,
                vec![hc_mix],
                logical_layer_id,
                false,
            );
            push(
                &mut specs,
                &prefix,
                &format!("{stem}_scale"),
                DType::F32,
                vec![3],
                logical_layer_id,
                false,
            );
        }

        if block == 0 {
            push_fp8_projection(
                &mut specs,
                &prefix,
                "main_proj",
                hidden,
                hidden
                    .checked_mul(facts.dspark_target_layer_ids.len())
                    .context("native dSpark target-tap projection width overflow")?,
                logical_layer_id,
            );
            push(
                &mut specs,
                &prefix,
                "main_norm.weight",
                DType::Bf16,
                vec![hidden],
                logical_layer_id,
                false,
            );
        }

        if block == final_block {
            push(
                &mut specs,
                &prefix,
                "norm.weight",
                DType::Bf16,
                vec![hidden],
                logical_layer_id,
                false,
            );
            for stem in [
                "markov_head.markov_w1.weight",
                "markov_head.markov_w2.weight",
            ] {
                push(
                    &mut specs,
                    &prefix,
                    stem,
                    DType::Bf16,
                    vec![facts.vocab_size, facts.dspark_markov_rank],
                    logical_layer_id,
                    false,
                );
            }
            push(
                &mut specs,
                &prefix,
                "confidence_head.proj.weight",
                DType::Bf16,
                vec![1, hidden + facts.dspark_markov_rank],
                logical_layer_id,
                false,
            );
            push(
                &mut specs,
                &prefix,
                "hc_head_fn",
                DType::F32,
                vec![hc_mult, hc_width],
                logical_layer_id,
                false,
            );
            push(
                &mut specs,
                &prefix,
                "hc_head_base",
                DType::F32,
                vec![hc_mult],
                logical_layer_id,
                false,
            );
            push(
                &mut specs,
                &prefix,
                "hc_head_scale",
                DType::F32,
                vec![1],
                logical_layer_id,
                false,
            );
        }
    }

    Ok(specs)
}

pub fn validate_native_deepseek_v4_dspark_catalog(
    catalog: &TensorCatalog,
) -> Result<NativeDeepseekV4DsparkCatalogSummary> {
    let specs = native_deepseek_v4_dspark_tensor_specs(&catalog.facts)?;
    let expected_names = specs
        .iter()
        .map(|spec| spec.name.as_str())
        .collect::<BTreeSet<_>>();
    let actual = catalog
        .tensors
        .iter()
        .filter(|tensor| tensor.name.starts_with("mtp.") && tensor.role == TensorRole::Dspark)
        .map(|tensor| (tensor.name.as_str(), tensor))
        .collect::<BTreeMap<_, _>>();
    let actual_names = actual.keys().copied().collect::<BTreeSet<_>>();
    let missing = expected_names
        .difference(&actual_names)
        .copied()
        .collect::<Vec<_>>();
    let unexpected = actual_names
        .difference(&expected_names)
        .copied()
        .collect::<Vec<_>>();
    anyhow::ensure!(
        missing.is_empty() && unexpected.is_empty(),
        "native dSpark coordinator tensor set mismatch: missing={missing:?} unexpected={unexpected:?}"
    );

    let mut coordinator_tensor_bytes = 0_u64;
    for spec in &specs {
        let tensor = actual
            .get(spec.name.as_str())
            .with_context(|| format!("missing native dSpark tensor {}", spec.name))?;
        anyhow::ensure!(
            tensor.dtype == spec.dtype && tensor.shape == spec.shape,
            "native dSpark tensor {} has {:?} {:?}, expected {:?} {:?}",
            spec.name,
            tensor.dtype,
            tensor.shape,
            spec.dtype,
            spec.shape,
        );
        anyhow::ensure!(
            tensor.layer_id == Some(spec.logical_layer_id as u32),
            "native dSpark tensor {} has logical layer {:?}, expected {}",
            spec.name,
            tensor.layer_id,
            spec.logical_layer_id,
        );
        anyhow::ensure!(
            tensor.is_quantization_metadata == spec.is_quantization_metadata,
            "native dSpark tensor {} quantization-metadata classification changed",
            spec.name,
        );
        coordinator_tensor_bytes = coordinator_tensor_bytes
            .checked_add(tensor.byte_length)
            .context("native dSpark coordinator tensor byte total overflow")?;
    }

    let routed_expert_tensors = catalog
        .tensors
        .iter()
        .filter(|tensor| tensor.name.starts_with("mtp.") && tensor.role == TensorRole::RoutedExpert)
        .count();
    let tensors_per_expert = if catalog.facts.quantization_recipe == NATIVE_FP4_EXPERT_RECIPE {
        NATIVE_FP4_TENSORS_PER_EXPERT
    } else if is_deepseek_v4_exl3_recipe(&catalog.facts.quantization_recipe) {
        EXL3_TENSORS_PER_EXPERT
    } else {
        anyhow::bail!(
            "native dSpark does not support routed-expert quantization recipe {:?}",
            catalog.facts.quantization_recipe
        )
    };
    let expected_routed_expert_tensors = catalog
        .facts
        .dspark_target_layer_ids
        .len()
        .checked_mul(catalog.facts.routed_experts)
        .and_then(|value| value.checked_mul(tensors_per_expert))
        .context("native dSpark routed-expert tensor count overflow")?;
    anyhow::ensure!(
        routed_expert_tensors == expected_routed_expert_tensors,
        "native dSpark has {routed_expert_tensors} routed-expert tensors, expected {expected_routed_expert_tensors}"
    );

    Ok(NativeDeepseekV4DsparkCatalogSummary {
        blocks: catalog.facts.dspark_target_layer_ids.len(),
        target_taps: catalog.facts.dspark_target_layer_ids.len(),
        proposal_tokens: catalog.facts.dspark_block_size,
        coordinator_tensors: specs.len(),
        routed_expert_tensors,
        coordinator_tensor_bytes,
    })
}

fn validate_dspark_facts(facts: &ModelFacts) -> Result<()> {
    anyhow::ensure!(
        facts.model_type == "deepseek_v4",
        "native dSpark validation requires model_type=deepseek_v4"
    );
    anyhow::ensure!(
        !facts.dspark_target_layer_ids.is_empty(),
        "native dSpark requires at least one integrated mtp block and target tap"
    );
    anyhow::ensure!(
        facts
            .dspark_target_layer_ids
            .iter()
            .all(|layer_id| *layer_id < facts.num_hidden_layers),
        "native dSpark target taps {:?} exceed the {}-layer target",
        facts.dspark_target_layer_ids,
        facts.num_hidden_layers,
    );
    anyhow::ensure!(
        facts.dspark_block_size > 0,
        "native dSpark proposal block size is zero"
    );
    anyhow::ensure!(
        facts.dspark_noise_token_id < facts.vocab_size,
        "native dSpark noise token {} exceeds vocabulary {}",
        facts.dspark_noise_token_id,
        facts.vocab_size,
    );
    anyhow::ensure!(
        facts.dspark_markov_rank > 0,
        "native dSpark Markov rank is zero"
    );
    anyhow::ensure!(
        facts.hidden_size % NATIVE_DSPARK_FP8_BLOCK == 0
            && facts.moe_intermediate_size % NATIVE_DSPARK_FP8_BLOCK == 0,
        "native dSpark FP8 dimensions must be divisible by {NATIVE_DSPARK_FP8_BLOCK}"
    );
    Ok(())
}

fn push(
    specs: &mut Vec<NativeDeepseekV4DsparkTensorSpec>,
    prefix: &str,
    suffix: &str,
    dtype: DType,
    shape: Vec<usize>,
    logical_layer_id: usize,
    is_quantization_metadata: bool,
) {
    specs.push(NativeDeepseekV4DsparkTensorSpec {
        name: format!("{prefix}.{suffix}"),
        dtype,
        shape,
        logical_layer_id,
        is_quantization_metadata,
    });
}

fn push_fp8_projection(
    specs: &mut Vec<NativeDeepseekV4DsparkTensorSpec>,
    prefix: &str,
    stem: &str,
    rows: usize,
    columns: usize,
    logical_layer_id: usize,
) {
    push(
        specs,
        prefix,
        &format!("{stem}.weight"),
        DType::F8E4M3,
        vec![rows, columns],
        logical_layer_id,
        false,
    );
    push(
        specs,
        prefix,
        &format!("{stem}.scale"),
        DType::F8E8M0,
        vec![
            rows.div_ceil(NATIVE_DSPARK_FP8_BLOCK),
            columns.div_ceil(NATIVE_DSPARK_FP8_BLOCK),
        ],
        logical_layer_id,
        true,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use ds41rt_core::TensorInfo;

    fn synthetic_catalog() -> TensorCatalog {
        synthetic_catalog_for_recipe(NATIVE_FP4_EXPERT_RECIPE)
    }

    fn synthetic_catalog_for_recipe(recipe: &str) -> TensorCatalog {
        let mut facts = ModelFacts::default();
        facts.quantization_recipe = recipe.to_owned();
        let mut tensors = native_deepseek_v4_dspark_tensor_specs(&facts)
            .unwrap()
            .into_iter()
            .map(|spec| TensorInfo {
                name: spec.name,
                file: "model.safetensors".to_owned(),
                dtype: spec.dtype,
                shape: spec.shape,
                byte_offset: 0,
                byte_length: 1,
                role: TensorRole::Dspark,
                layer_id: Some(spec.logical_layer_id as u32),
                expert_id: None,
                is_quantization_metadata: spec.is_quantization_metadata,
            })
            .collect::<Vec<_>>();
        for block in 0..facts.dspark_target_layer_ids.len() {
            for expert in 0..facts.routed_experts {
                for projection in ["w1", "w2", "w3"] {
                    let suffixes: &[&str] = if is_deepseek_v4_exl3_recipe(recipe) {
                        &["trellis", "suh", "svh", "mcg"]
                    } else {
                        &["weight", "scale"]
                    };
                    for suffix in suffixes {
                        tensors.push(TensorInfo {
                            name: format!("mtp.{block}.ffn.experts.{expert}.{projection}.{suffix}"),
                            file: "model.safetensors".to_owned(),
                            dtype: match *suffix {
                                "weight" => DType::I8,
                                "scale" => DType::F8E8M0,
                                "trellis" => DType::I16,
                                "suh" | "svh" => DType::F16,
                                "mcg" => DType::I32,
                                _ => unreachable!(),
                            },
                            shape: vec![1],
                            byte_offset: 0,
                            byte_length: 1,
                            role: TensorRole::RoutedExpert,
                            layer_id: Some((facts.num_hidden_layers + block) as u32),
                            expert_id: Some(expert as u32),
                            is_quantization_metadata: !matches!(*suffix, "weight" | "trellis"),
                        });
                    }
                }
            }
        }
        TensorCatalog {
            model_id: facts.model_id.clone(),
            snapshot_path: "/model".to_owned(),
            facts,
            tensors,
        }
    }

    #[test]
    fn flash_contract_is_three_native_blocks_with_five_proposals() {
        let catalog = synthetic_catalog();
        let summary = validate_native_deepseek_v4_dspark_catalog(&catalog).unwrap();
        assert_eq!(summary.blocks, 3);
        assert_eq!(summary.target_taps, 3);
        assert_eq!(summary.proposal_tokens, 5);
        assert_eq!(summary.coordinator_tensors, 97);
        assert_eq!(summary.routed_expert_tensors, 4_608);
    }

    #[test]
    fn flash_contract_accepts_exl3_routed_experts() {
        let catalog = synthetic_catalog_for_recipe(crate::DEEPSEEK_V4_EXL3_RECIPE);
        let summary = validate_native_deepseek_v4_dspark_catalog(&catalog).unwrap();
        assert_eq!(summary.blocks, 3);
        assert_eq!(summary.proposal_tokens, 5);
        assert_eq!(summary.routed_expert_tensors, 9_216);
    }

    #[test]
    fn contract_rejects_incomplete_exl3_routed_experts() {
        let mut catalog = synthetic_catalog_for_recipe(crate::DEEPSEEK_V4_EXL3_RECIPE);
        catalog
            .tensors
            .retain(|tensor| tensor.name != "mtp.2.ffn.experts.255.w3.mcg");
        let error = validate_native_deepseek_v4_dspark_catalog(&catalog)
            .unwrap_err()
            .to_string();
        assert!(error.contains("9215 routed-expert tensors, expected 9216"));
    }

    #[test]
    fn contract_rejects_incomplete_or_unexpected_mtp_namespaces() {
        let mut catalog = synthetic_catalog();
        catalog
            .tensors
            .retain(|tensor| tensor.name != "mtp.2.confidence_head.proj.weight");
        let error = validate_native_deepseek_v4_dspark_catalog(&catalog)
            .unwrap_err()
            .to_string();
        assert!(error.contains("confidence_head.proj.weight"));

        let mut catalog = synthetic_catalog();
        catalog.tensors.push(TensorInfo {
            name: "mtp.3.unexpected_auxiliary.weight".to_owned(),
            file: "unexpected.safetensors".to_owned(),
            dtype: DType::Bf16,
            shape: vec![1],
            byte_offset: 0,
            byte_length: 2,
            role: TensorRole::Dspark,
            layer_id: Some(46),
            expert_id: None,
            is_quantization_metadata: false,
        });
        let error = validate_native_deepseek_v4_dspark_catalog(&catalog)
            .unwrap_err()
            .to_string();
        assert!(error.contains("unexpected_auxiliary"));
    }
}
