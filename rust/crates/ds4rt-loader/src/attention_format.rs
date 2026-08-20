use anyhow::{Context, Result};
use ds4rt_core::{
    DType, DeepseekV4AttentionLayerSource, DeepseekV4AttentionPlan, ModelFacts, TensorCatalog,
    TensorInfo, TensorRole,
};
use std::collections::{BTreeMap, BTreeSet};

pub const NATIVE_ATTENTION_FP8_BLOCK: usize = 128;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeDeepseekV4AttentionTensorFamily {
    Main,
    Compressor,
    Indexer,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeDeepseekV4AttentionTensorSpec {
    pub name: String,
    pub family: NativeDeepseekV4AttentionTensorFamily,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub role: TensorRole,
    pub logical_layer_id: usize,
    pub is_quantization_metadata: bool,
}

impl NativeDeepseekV4AttentionTensorSpec {
    pub fn byte_length(&self) -> Result<u64> {
        let dtype_bytes = match self.dtype {
            DType::Bf16 | DType::F16 | DType::I16 => 2_u64,
            DType::F32 | DType::I32 => 4,
            DType::I64 => 8,
            DType::F8E4M3 | DType::F8E5M2 | DType::F8E8M0 | DType::I8 | DType::U8 => 1,
            DType::F4 => {
                anyhow::bail!("native attention contract does not use packed F4 tensors")
            }
            DType::Unknown(ref dtype) => {
                anyhow::bail!("native attention contract has unknown dtype {dtype:?}")
            }
        };
        let elements = self.shape.iter().try_fold(1_u64, |size, dim| {
            size.checked_mul(*dim as u64)
                .context("native attention tensor element count overflow")
        })?;
        elements
            .checked_mul(dtype_bytes)
            .context("native attention tensor byte count overflow")
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeDeepseekV4AttentionCatalogSummary {
    pub target_blocks: usize,
    pub dspark_blocks: usize,
    pub sliding_blocks: usize,
    pub c4_blocks: usize,
    pub c128_blocks: usize,
    pub main_tensors: usize,
    pub compressor_tensors: usize,
    pub indexer_tensors: usize,
    pub tensor_bytes: u64,
}

/// Derive the exact checkpoint tensor contract consumed by native DSV4
/// attention. This describes all target and dSpark blocks, but deliberately
/// excludes mHC and the surrounding residual path.
pub fn native_deepseek_v4_attention_tensor_specs(
    facts: &ModelFacts,
) -> Result<Vec<NativeDeepseekV4AttentionTensorSpec>> {
    anyhow::ensure!(
        facts.model_type == "deepseek_v4",
        "native attention validation requires model_type=deepseek_v4"
    );
    let attention = DeepseekV4AttentionPlan::from_model_facts(facts)
        .context("building native DeepSeek V4 attention geometry")?;
    attention
        .validate_sparkinfer_sm120_contract()
        .context("validating native DeepSeek V4 SparkInfer geometry")?;
    let geometry = &attention.geometry;
    let query_width = geometry
        .attention_heads
        .checked_mul(geometry.head_dim)
        .context("native attention query width overflow")?;
    let grouped_attention_width = query_width
        .checked_div(geometry.o_groups)
        .context("native attention output group count must be positive")?;
    let grouped_output_rank = geometry
        .o_groups
        .checked_mul(geometry.o_lora_rank)
        .context("native attention grouped output rank overflow")?;

    let mut specs = Vec::new();
    for layer in &attention.layers {
        let role = match layer.source {
            DeepseekV4AttentionLayerSource::Target { .. } => TensorRole::Attention,
            DeepseekV4AttentionLayerSource::Dspark { .. } => TensorRole::Dspark,
        };
        let prefix = format!("{}.attn", layer.source.checkpoint_block_prefix());
        push_spec(
            &mut specs,
            &prefix,
            "attn_sink",
            NativeDeepseekV4AttentionTensorFamily::Main,
            DType::F32,
            vec![geometry.attention_heads],
            role.clone(),
            layer.logical_layer_id,
            false,
        );
        push_spec(
            &mut specs,
            &prefix,
            "kv_norm.weight",
            NativeDeepseekV4AttentionTensorFamily::Main,
            DType::Bf16,
            vec![geometry.head_dim],
            role.clone(),
            layer.logical_layer_id,
            false,
        );
        push_spec(
            &mut specs,
            &prefix,
            "q_norm.weight",
            NativeDeepseekV4AttentionTensorFamily::Main,
            DType::Bf16,
            vec![geometry.q_lora_rank],
            role.clone(),
            layer.logical_layer_id,
            false,
        );
        push_fp8_projection(
            &mut specs,
            &prefix,
            "wkv",
            NativeDeepseekV4AttentionTensorFamily::Main,
            geometry.head_dim,
            geometry.hidden_size,
            role.clone(),
            layer.logical_layer_id,
        )?;
        push_fp8_projection(
            &mut specs,
            &prefix,
            "wo_a",
            NativeDeepseekV4AttentionTensorFamily::Main,
            grouped_output_rank,
            grouped_attention_width,
            role.clone(),
            layer.logical_layer_id,
        )?;
        push_fp8_projection(
            &mut specs,
            &prefix,
            "wo_b",
            NativeDeepseekV4AttentionTensorFamily::Main,
            geometry.hidden_size,
            grouped_output_rank,
            role.clone(),
            layer.logical_layer_id,
        )?;
        push_fp8_projection(
            &mut specs,
            &prefix,
            "wq_a",
            NativeDeepseekV4AttentionTensorFamily::Main,
            geometry.q_lora_rank,
            geometry.hidden_size,
            role.clone(),
            layer.logical_layer_id,
        )?;
        push_fp8_projection(
            &mut specs,
            &prefix,
            "wq_b",
            NativeDeepseekV4AttentionTensorFamily::Main,
            query_width,
            geometry.q_lora_rank,
            role,
            layer.logical_layer_id,
        )?;

        if layer.uses_compressor() {
            push_compressor_specs(
                &mut specs,
                &prefix,
                "compressor",
                NativeDeepseekV4AttentionTensorFamily::Compressor,
                TensorRole::AttentionCompressor,
                layer.logical_layer_id,
                layer.compress_ratio,
                geometry.head_dim,
                geometry.hidden_size,
            )?;
        }
        if layer.uses_indexer() {
            let indexer_prefix = format!("{prefix}.indexer");
            push_compressor_specs(
                &mut specs,
                &indexer_prefix,
                "compressor",
                NativeDeepseekV4AttentionTensorFamily::Indexer,
                TensorRole::AttentionIndexer,
                layer.logical_layer_id,
                layer.compress_ratio,
                geometry.index_head_dim,
                geometry.hidden_size,
            )?;
            push_spec(
                &mut specs,
                &indexer_prefix,
                "weights_proj.weight",
                NativeDeepseekV4AttentionTensorFamily::Indexer,
                DType::Bf16,
                vec![geometry.index_heads, geometry.hidden_size],
                TensorRole::AttentionIndexer,
                layer.logical_layer_id,
                false,
            );
            let index_query_width = geometry
                .index_heads
                .checked_mul(geometry.index_head_dim)
                .context("native attention index query width overflow")?;
            push_fp8_projection(
                &mut specs,
                &indexer_prefix,
                "wq_b",
                NativeDeepseekV4AttentionTensorFamily::Indexer,
                index_query_width,
                geometry.q_lora_rank,
                TensorRole::AttentionIndexer,
                layer.logical_layer_id,
            )?;
        }
    }
    Ok(specs)
}

pub fn validate_native_deepseek_v4_attention_catalog(
    catalog: &TensorCatalog,
) -> Result<NativeDeepseekV4AttentionCatalogSummary> {
    let specs = native_deepseek_v4_attention_tensor_specs(&catalog.facts)?;
    let expected_by_name = specs
        .iter()
        .map(|spec| (spec.name.as_str(), spec))
        .collect::<BTreeMap<_, _>>();
    anyhow::ensure!(
        expected_by_name.len() == specs.len(),
        "native attention tensor contract generated duplicate names"
    );
    let actual_by_name = catalog
        .tensors
        .iter()
        .map(|tensor| (tensor.name.as_str(), tensor))
        .collect::<BTreeMap<_, _>>();
    let expected_names = expected_by_name.keys().copied().collect::<BTreeSet<_>>();
    let actual_names = actual_by_name
        .keys()
        .copied()
        .filter(|name| native_attention_tensor_name(name))
        .collect::<BTreeSet<_>>();
    let actual_tensor_count = catalog
        .tensors
        .iter()
        .filter(|tensor| native_attention_tensor_name(&tensor.name))
        .count();
    anyhow::ensure!(
        actual_tensor_count == actual_names.len(),
        "native DeepSeek V4 attention catalog contains duplicate tensor names"
    );
    anyhow::ensure!(
        actual_names == expected_names,
        "native DeepSeek V4 attention tensor set mismatch: expected {} tensors, found {}; missing={:?}; unexpected={:?}",
        expected_names.len(),
        actual_names.len(),
        expected_names
            .difference(&actual_names)
            .take(8)
            .collect::<Vec<_>>(),
        actual_names
            .difference(&expected_names)
            .take(8)
            .collect::<Vec<_>>()
    );

    let mut tensor_bytes = 0_u64;
    for spec in &specs {
        let tensor = actual_by_name
            .get(spec.name.as_str())
            .with_context(|| format!("native attention tensor {} not found", spec.name))?;
        validate_tensor(tensor, spec)?;
        tensor_bytes = tensor_bytes
            .checked_add(tensor.byte_length)
            .context("native attention catalog byte total overflow")?;
    }

    let attention = DeepseekV4AttentionPlan::from_model_facts(&catalog.facts)
        .context("rebuilding native DeepSeek V4 attention summary")?;
    Ok(NativeDeepseekV4AttentionCatalogSummary {
        target_blocks: attention.target_layer_count,
        dspark_blocks: attention.dspark_layer_count,
        sliding_blocks: attention
            .layers
            .iter()
            .filter(|layer| layer.compress_ratio == 0)
            .count(),
        c4_blocks: attention
            .layers
            .iter()
            .filter(|layer| layer.compress_ratio == 4)
            .count(),
        c128_blocks: attention
            .layers
            .iter()
            .filter(|layer| layer.compress_ratio == 128)
            .count(),
        main_tensors: specs
            .iter()
            .filter(|spec| spec.family == NativeDeepseekV4AttentionTensorFamily::Main)
            .count(),
        compressor_tensors: specs
            .iter()
            .filter(|spec| spec.family == NativeDeepseekV4AttentionTensorFamily::Compressor)
            .count(),
        indexer_tensors: specs
            .iter()
            .filter(|spec| spec.family == NativeDeepseekV4AttentionTensorFamily::Indexer)
            .count(),
        tensor_bytes,
    })
}

fn push_fp8_projection(
    specs: &mut Vec<NativeDeepseekV4AttentionTensorSpec>,
    prefix: &str,
    stem: &str,
    family: NativeDeepseekV4AttentionTensorFamily,
    output_features: usize,
    input_features: usize,
    role: TensorRole,
    logical_layer_id: usize,
) -> Result<()> {
    anyhow::ensure!(
        output_features % NATIVE_ATTENTION_FP8_BLOCK == 0
            && input_features % NATIVE_ATTENTION_FP8_BLOCK == 0,
        "native attention {stem} shape {output_features}x{input_features} must be divisible by FP8 block {NATIVE_ATTENTION_FP8_BLOCK}"
    );
    push_spec(
        specs,
        prefix,
        &format!("{stem}.scale"),
        family,
        DType::F8E8M0,
        vec![
            output_features / NATIVE_ATTENTION_FP8_BLOCK,
            input_features / NATIVE_ATTENTION_FP8_BLOCK,
        ],
        role.clone(),
        logical_layer_id,
        true,
    );
    push_spec(
        specs,
        prefix,
        &format!("{stem}.weight"),
        family,
        DType::F8E4M3,
        vec![output_features, input_features],
        role,
        logical_layer_id,
        false,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn push_compressor_specs(
    specs: &mut Vec<NativeDeepseekV4AttentionTensorSpec>,
    prefix: &str,
    stem: &str,
    family: NativeDeepseekV4AttentionTensorFamily,
    role: TensorRole,
    logical_layer_id: usize,
    compress_ratio: usize,
    head_dim: usize,
    hidden_size: usize,
) -> Result<()> {
    let coefficient: usize = if compress_ratio == 4 { 2 } else { 1 };
    let projected_width = coefficient
        .checked_mul(head_dim)
        .context("native attention compressor width overflow")?;
    let compressor_prefix = format!("{prefix}.{stem}");
    push_spec(
        specs,
        &compressor_prefix,
        "ape",
        family,
        DType::F32,
        vec![compress_ratio, projected_width],
        role.clone(),
        logical_layer_id,
        false,
    );
    push_spec(
        specs,
        &compressor_prefix,
        "norm.weight",
        family,
        DType::Bf16,
        vec![head_dim],
        role.clone(),
        logical_layer_id,
        false,
    );
    for projection in ["wgate.weight", "wkv.weight"] {
        push_spec(
            specs,
            &compressor_prefix,
            projection,
            family,
            DType::Bf16,
            vec![projected_width, hidden_size],
            role.clone(),
            logical_layer_id,
            false,
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn push_spec(
    specs: &mut Vec<NativeDeepseekV4AttentionTensorSpec>,
    prefix: &str,
    suffix: &str,
    family: NativeDeepseekV4AttentionTensorFamily,
    dtype: DType,
    shape: Vec<usize>,
    role: TensorRole,
    logical_layer_id: usize,
    is_quantization_metadata: bool,
) {
    specs.push(NativeDeepseekV4AttentionTensorSpec {
        name: format!("{prefix}.{suffix}"),
        family,
        dtype,
        shape,
        role,
        logical_layer_id,
        is_quantization_metadata,
    });
}

fn native_attention_tensor_name(name: &str) -> bool {
    (name.starts_with("layers.") || name.starts_with("mtp.")) && name.contains(".attn.")
}

fn validate_tensor(tensor: &TensorInfo, spec: &NativeDeepseekV4AttentionTensorSpec) -> Result<()> {
    anyhow::ensure!(
        tensor.dtype == spec.dtype,
        "tensor {} has dtype {:?}, expected {:?}",
        tensor.name,
        tensor.dtype,
        spec.dtype
    );
    anyhow::ensure!(
        tensor.shape == spec.shape,
        "tensor {} has shape {:?}, expected {:?}",
        tensor.name,
        tensor.shape,
        spec.shape
    );
    anyhow::ensure!(
        tensor.role == spec.role
            && tensor.layer_id == Some(spec.logical_layer_id as u32)
            && tensor.expert_id.is_none(),
        "tensor {} has inconsistent attention identity role={:?} layer={:?} expert={:?}; expected role={:?} layer={}",
        tensor.name,
        tensor.role,
        tensor.layer_id,
        tensor.expert_id,
        spec.role,
        spec.logical_layer_id
    );
    anyhow::ensure!(
        tensor.is_quantization_metadata == spec.is_quantization_metadata,
        "tensor {} quantization metadata flag is {}, expected {}",
        tensor.name,
        tensor.is_quantization_metadata,
        spec.is_quantization_metadata
    );
    let expected_bytes = spec.byte_length()?;
    anyhow::ensure!(
        tensor.byte_length == expected_bytes,
        "tensor {} records {} bytes, expected {}",
        tensor.name,
        tensor.byte_length,
        expected_bytes
    );
    Ok(())
}
