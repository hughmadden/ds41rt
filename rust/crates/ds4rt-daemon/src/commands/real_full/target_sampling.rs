use anyhow::{Context, Result};
use ds4rt_core::{DType, TensorCatalog, TensorRole};
use ds4rt_ffi::DS4RT_CUDA_SAMPLE_TOPK_MAX_K;

use super::constraint::RealFullConstraintMasks;
use super::coordinator_kernels::{
    concat_device_bf16_row_batches_async, device_bf16_output_from_device_template_buffer,
    device_buffer_byte_view, rmsnorm_hidden_bf16_preloaded_resident_weight_device_input_output,
    DeviceBf16Output,
};
use super::embedding::real_full_embedding_device_hidden_for_tokens;
use super::sampling::{
    score_real_lm_head_full_vocab_for_device_hidden_rows,
    score_real_lm_head_full_vocab_for_device_hidden_rows_constrained,
    score_real_lm_head_full_vocab_for_device_hidden_rows_with_options,
    RealFullLmHeadSamplingOptions, RealLmHeadBatchScoreForHidden,
};

const DEEPSEEK_V4_TARGET_FINAL_NORM_WEIGHT: &str = "norm.weight";

pub(in crate::commands::real_full) fn real_full_target_token_samples(
    catalog: &TensorCatalog,
    target_hidden: &DeviceBf16Output,
    suffix_rows: usize,
) -> Result<RealLmHeadBatchScoreForHidden> {
    let normalized = real_full_normalized_target_suffix(catalog, target_hidden, suffix_rows)?;
    score_real_lm_head_full_vocab_for_device_hidden_rows(catalog, &normalized)
        .context("scoring real-full target verification rows")
}

pub(in crate::commands::real_full) fn real_full_target_token_samples_with_options(
    catalog: &TensorCatalog,
    target_hidden: &DeviceBf16Output,
    suffix_rows: usize,
    sampler_options: RealFullLmHeadSamplingOptions,
    random_uniforms: &[f32],
) -> Result<RealLmHeadBatchScoreForHidden> {
    anyhow::ensure!(
        random_uniforms.len() == suffix_rows,
        "real-full target sampler received {} random uniforms for {suffix_rows} suffix rows",
        random_uniforms.len()
    );
    let normalized = real_full_normalized_target_suffix(catalog, target_hidden, suffix_rows)?;
    score_real_lm_head_full_vocab_for_device_hidden_rows_with_options(
        catalog,
        &normalized,
        sampler_options,
        random_uniforms,
        false,
    )
    .context("scoring sampled real-full target verification rows")
}

pub(in crate::commands::real_full) fn real_full_target_token_samples_constrained(
    catalog: &TensorCatalog,
    target_hidden: &DeviceBf16Output,
    suffix_rows: usize,
    sampler_options: RealFullLmHeadSamplingOptions,
    random_uniforms: &[f32],
    masks: &RealFullConstraintMasks,
) -> Result<RealLmHeadBatchScoreForHidden> {
    anyhow::ensure!(
        random_uniforms.len() == suffix_rows,
        "constrained target sampler received {} random uniforms for {suffix_rows} suffix rows",
        random_uniforms.len()
    );
    anyhow::ensure!(
        masks.rows == suffix_rows,
        "constrained target sampler received {} mask rows for {suffix_rows} suffix rows",
        masks.rows
    );
    let normalized = real_full_normalized_target_suffix(catalog, target_hidden, suffix_rows)?;
    score_real_lm_head_full_vocab_for_device_hidden_rows_constrained(
        catalog,
        &normalized,
        sampler_options,
        random_uniforms,
        &masks.words,
    )
    .context("scoring constrained real-full target verification rows")
}

fn real_full_normalized_target_suffix(
    catalog: &TensorCatalog,
    target_hidden: &DeviceBf16Output,
    suffix_rows: usize,
) -> Result<DeviceBf16Output> {
    let hidden_size = catalog.facts.hidden_size;
    anyhow::ensure!(
        suffix_rows > 0
            && suffix_rows <= target_hidden.rows
            && hidden_size > 0
            && target_hidden.values_per_row == hidden_size,
        "real-full target sampling suffix {} is invalid for model hidden size {} and hidden {}x{}",
        suffix_rows,
        hidden_size,
        target_hidden.rows,
        target_hidden.values_per_row
    );
    validate_target_final_norm(catalog)?;
    let row_bytes = hidden_size
        .checked_mul(std::mem::size_of::<u16>())
        .context("target hidden row byte count overflow")?;
    let suffix_bytes = suffix_rows
        .checked_mul(row_bytes)
        .context("target hidden suffix byte count overflow")?;
    let offset_bytes = (target_hidden.rows - suffix_rows)
        .checked_mul(row_bytes)
        .context("target hidden suffix offset overflow")?;
    let view = device_buffer_byte_view(
        target_hidden.buffer(),
        offset_bytes,
        suffix_bytes,
        "real-full target sample suffix",
    )?;
    let hidden = device_bf16_output_from_device_template_buffer(
        view,
        suffix_rows,
        hidden_size,
        "real-full target sample suffix",
    )?;
    rmsnorm_hidden_bf16_preloaded_resident_weight_device_input_output(
        DEEPSEEK_V4_TARGET_FINAL_NORM_WEIGHT,
        hidden.buffer(),
        suffix_rows,
        hidden_size,
        catalog.facts.rms_norm_eps,
    )
    .context("normalizing DeepSeek V4 target verification rows")
}

fn validate_target_final_norm(catalog: &TensorCatalog) -> Result<()> {
    let tensor = catalog
        .tensors
        .iter()
        .find(|tensor| tensor.name == DEEPSEEK_V4_TARGET_FINAL_NORM_WEIGHT)
        .context("DeepSeek V4 target final norm.weight is missing from the catalog")?;
    anyhow::ensure!(
        tensor.dtype == DType::Bf16
            && tensor.role == TensorRole::Norm
            && tensor.shape == [catalog.facts.hidden_size]
            && tensor.layer_id.is_none()
            && tensor.expert_id.is_none()
            && !tensor.is_quantization_metadata,
        "DeepSeek V4 target final norm metadata is invalid: dtype={:?} role={:?} shape={:?} layer={:?} expert={:?} quantization_metadata={}",
        tensor.dtype,
        tensor.role,
        tensor.shape,
        tensor.layer_id,
        tensor.expert_id,
        tensor.is_quantization_metadata
    );
    Ok(())
}

pub(in crate::commands::real_full) fn prewarm_real_full_paired_target_token_sample_rows(
    catalog: &TensorCatalog,
    min_rows: usize,
    max_rows: usize,
) -> Result<()> {
    anyhow::ensure!(
        min_rows >= 3 && min_rows <= max_rows,
        "invalid paired target-sample prewarm row range {min_rows}..={max_rows}"
    );
    let token_ids = vec![0; max_rows];
    let hidden = real_full_embedding_device_hidden_for_tokens(catalog, &token_ids)
        .context("creating target-sample prewarm hidden rows")?
        .context("target-sample prewarm requires device-resident embeddings")?;
    for rows in (min_rows..=max_rows).rev() {
        let rows_a = 16.min(rows - 2);
        let rows_b = rows - rows_a;
        let hidden_a = real_full_device_hidden_rows(&hidden.device_hidden, 0, rows_a)
            .context("slicing paired target-sample prewarm rows A")?;
        let hidden_b = real_full_device_hidden_rows(&hidden.device_hidden, rows_a, rows_b)
            .context("slicing paired target-sample prewarm rows B")?;
        real_full_target_token_samples_pair(catalog, &hidden_a, rows_a, &hidden_b, rows_b)
            .with_context(|| format!("prewarming paired target-sample graph for {rows} rows"))?;
    }
    Ok(())
}

pub(in crate::commands::real_full) fn prewarm_real_full_target_sampler_capacity(
    catalog: &TensorCatalog,
    max_rows: usize,
) -> Result<()> {
    anyhow::ensure!(
        max_rows > 0,
        "target sampler capacity prewarm requires nonzero rows"
    );
    let token_ids = vec![0; max_rows];
    let hidden = real_full_embedding_device_hidden_for_tokens(catalog, &token_ids)
        .context("creating target sampler capacity prewarm hidden rows")?
        .context("target sampler capacity prewarm requires device-resident embeddings")?;
    let options = RealFullLmHeadSamplingOptions {
        random_uniform: 0.5,
        temperature: 1.0,
        top_k: DS4RT_CUDA_SAMPLE_TOPK_MAX_K,
        top_p: 0.95,
    };
    for rows in [32, 16, 8, 1].into_iter().filter(|rows| *rows <= max_rows) {
        let target_hidden = real_full_device_hidden_rows(&hidden.device_hidden, 0, rows)
            .with_context(|| format!("slicing target sampler capacity rows={rows}"))?;
        let normalized = real_full_normalized_target_suffix(catalog, &target_hidden, rows)
            .with_context(|| format!("normalizing target sampler capacity rows={rows}"))?;
        let random_uniforms = vec![options.random_uniform; rows];
        score_real_lm_head_full_vocab_for_device_hidden_rows_with_options(
            catalog,
            &normalized,
            options,
            &random_uniforms,
            true,
        )
        .with_context(|| {
            format!(
                "prewarming target sampler capacity rows={rows} top_k={}",
                options.top_k
            )
        })?;
    }
    Ok(())
}

pub(in crate::commands::real_full) fn real_full_target_token_samples_pair(
    catalog: &TensorCatalog,
    target_hidden_a: &DeviceBf16Output,
    suffix_rows_a: usize,
    target_hidden_b: &DeviceBf16Output,
    suffix_rows_b: usize,
) -> Result<(RealLmHeadBatchScoreForHidden, RealLmHeadBatchScoreForHidden)> {
    let suffix_a = real_full_device_hidden_rows(
        target_hidden_a,
        target_hidden_a
            .rows
            .checked_sub(suffix_rows_a)
            .context("paired target sample A suffix exceeds retained rows")?,
        suffix_rows_a,
    )
    .context("slicing paired target sample A suffix")?;
    let suffix_b = real_full_device_hidden_rows(
        target_hidden_b,
        target_hidden_b
            .rows
            .checked_sub(suffix_rows_b)
            .context("paired target sample B suffix exceeds retained rows")?,
        suffix_rows_b,
    )
    .context("slicing paired target sample B suffix")?;
    anyhow::ensure!(
        suffix_a.values_per_row == catalog.facts.hidden_size
            && suffix_b.values_per_row == catalog.facts.hidden_size,
        "paired target hidden widths {}/{} do not match model hidden size {}",
        suffix_a.values_per_row,
        suffix_b.values_per_row,
        catalog.facts.hidden_size
    );
    validate_target_final_norm(catalog)?;
    let combined = concat_device_bf16_row_batches_async(
        &[&suffix_a, &suffix_b],
        "paired target sample hidden rows",
    )
    .context("concatenating paired target sample rows")?;
    let normalized = rmsnorm_hidden_bf16_preloaded_resident_weight_device_input_output(
        DEEPSEEK_V4_TARGET_FINAL_NORM_WEIGHT,
        combined.buffer(),
        combined.rows,
        catalog.facts.hidden_size,
        catalog.facts.rms_norm_eps,
    )
    .context("normalizing paired target verification rows")?;
    let mut samples = score_real_lm_head_full_vocab_for_device_hidden_rows(catalog, &normalized)
        .context("scoring paired target verification rows")?;
    anyhow::ensure!(
        samples.top_token_ids.len() == suffix_rows_a + suffix_rows_b
            && samples.sampled_token_ids.len() == suffix_rows_a + suffix_rows_b,
        "paired target scoring returned top/sample rows {}/{} for {} expected rows",
        samples.top_token_ids.len(),
        samples.sampled_token_ids.len(),
        suffix_rows_a + suffix_rows_b,
    );
    let top_token_ids_b = samples.top_token_ids.split_off(suffix_rows_a);
    let top_logits_b = samples.top_logits.split_off(suffix_rows_a);
    let sampled_token_ids_b = samples.sampled_token_ids.split_off(suffix_rows_a);
    let samples_b = RealLmHeadBatchScoreForHidden {
        vocab_size: samples.vocab_size,
        top_token_ids: top_token_ids_b,
        top_logits: top_logits_b,
        sampled_token_ids: sampled_token_ids_b,
        sample_top_k: samples.sample_top_k,
        sample_top_p: samples.sample_top_p,
        argmax_kernel_backend: samples.argmax_kernel_backend,
        sampler_kernel_backend: samples.sampler_kernel_backend,
    };
    Ok((samples, samples_b))
}

pub(in crate::commands::real_full) fn real_full_device_hidden_row(
    hidden: &DeviceBf16Output,
    row_index: usize,
) -> Result<DeviceBf16Output> {
    real_full_device_hidden_rows(hidden, row_index, 1)
}

pub(in crate::commands::real_full) fn real_full_device_hidden_rows(
    hidden: &DeviceBf16Output,
    row_start: usize,
    rows: usize,
) -> Result<DeviceBf16Output> {
    anyhow::ensure!(
        rows > 0
            && hidden.values_per_row > 0
            && row_start
                .checked_add(rows)
                .is_some_and(|row_end| row_end <= hidden.rows),
        "real-full hidden rows {}+{} are invalid for hidden {}x{}",
        row_start,
        rows,
        hidden.rows,
        hidden.values_per_row
    );
    let row_bytes = hidden
        .values_per_row
        .checked_mul(std::mem::size_of::<u16>())
        .context("hidden row byte count overflow")?;
    let offset_bytes = row_start
        .checked_mul(row_bytes)
        .context("hidden row byte offset overflow")?;
    let length_bytes = rows
        .checked_mul(row_bytes)
        .context("hidden row byte length overflow")?;
    let view = device_buffer_byte_view(
        hidden.buffer(),
        offset_bytes,
        length_bytes,
        "real-full hidden rows",
    )?;
    device_bf16_output_from_device_template_buffer(
        view,
        rows,
        hidden.values_per_row,
        "real-full hidden rows",
    )
}

pub(in crate::commands::real_full) fn real_full_last_device_hidden_row(
    hidden: &DeviceBf16Output,
) -> Result<DeviceBf16Output> {
    anyhow::ensure!(hidden.rows > 0, "real-full hidden tail requires a row");
    real_full_device_hidden_row(hidden, hidden.rows - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ds4rt_core::{ModelFacts, TensorInfo, DEFAULT_MODEL_ID};

    fn catalog_with_final_norm(name: &str, shape: Vec<usize>) -> TensorCatalog {
        TensorCatalog {
            model_id: DEFAULT_MODEL_ID.to_owned(),
            snapshot_path: "/tmp/deepseek-v4".to_owned(),
            facts: ModelFacts::default(),
            tensors: vec![TensorInfo {
                name: name.to_owned(),
                file: "model.safetensors".to_owned(),
                dtype: DType::Bf16,
                shape,
                byte_offset: 0,
                byte_length: 2,
                role: TensorRole::Norm,
                layer_id: None,
                expert_id: None,
                is_quantization_metadata: false,
            }],
        }
    }

    #[test]
    fn target_final_norm_accepts_deepseek_v4_root_name_and_shape() {
        let catalog = catalog_with_final_norm(
            DEEPSEEK_V4_TARGET_FINAL_NORM_WEIGHT,
            vec![ModelFacts::default().hidden_size],
        );
        validate_target_final_norm(&catalog).unwrap();
    }

    #[test]
    fn target_final_norm_rejects_glm_model_prefix() {
        let catalog =
            catalog_with_final_norm("model.norm.weight", vec![ModelFacts::default().hidden_size]);
        assert!(validate_target_final_norm(&catalog).is_err());
    }
}
