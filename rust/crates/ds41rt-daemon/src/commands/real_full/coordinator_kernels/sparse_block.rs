use super::*;
use anyhow::{Context, Result};
use ds41rt_core::{
    DType, DeepseekV4AttentionLayerSource, DeepseekV4AttentionPlan, ModelFacts, ModelVariant,
    TensorCatalog, DS4_FLASH_HIDDEN_SIZE, DS4_FLASH_MOE_INTERMEDIATE_SIZE,
    DS4_FLASH_ROUTED_EXPERTS, DS4_FLASH_TOP_K, DS4_NUM_SHARED_EXPERTS, DS4_PRO_HIDDEN_SIZE,
    DS4_PRO_MOE_INTERMEDIATE_SIZE, DS4_PRO_ROUTED_EXPERTS, DS4_PRO_TOP_K,
};
use ds41rt_ffi::DS41RT_HOST_BUFFER_FLAG_PINNED;
use ds41rt_loader::{read_tensor_bytes_into, DEEPSEEK_V4_EXL3_RECIPE};

const DS4_FLASH_HIDDEN: usize = 4_096;
const DS4_FLASH_TP: usize = 4;
const DS4_FLASH_MAX_ROWS: usize = 2_048;
const DS4_VOCAB: usize = 129_280;
const DS4_TP4_ASYNC_OWNED_REDUCTION_BACKEND: &str = "deepseek-v4-tp4-async-owned-bf16-reduction";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Ds4SparseProfile {
    variant: ModelVariant,
    hidden: usize,
    routed_experts: usize,
    top_k: usize,
    shared_intermediate: usize,
}

impl Ds4SparseProfile {
    fn from_facts(facts: &ModelFacts) -> Result<Self> {
        anyhow::ensure!(
            facts.model_type == "deepseek_v4",
            "DeepSeek V4 sparse coordinator requires model_type deepseek_v4, got {:?}",
            facts.model_type
        );
        let profile = match facts.variant {
            ModelVariant::Flash => Self {
                variant: ModelVariant::Flash,
                hidden: DS4_FLASH_HIDDEN_SIZE,
                routed_experts: DS4_FLASH_ROUTED_EXPERTS,
                top_k: DS4_FLASH_TOP_K,
                shared_intermediate: DS4_FLASH_MOE_INTERMEDIATE_SIZE,
            },
            ModelVariant::Pro => Self {
                variant: ModelVariant::Pro,
                hidden: DS4_PRO_HIDDEN_SIZE,
                routed_experts: DS4_PRO_ROUTED_EXPERTS,
                top_k: DS4_PRO_TOP_K,
                shared_intermediate: DS4_PRO_MOE_INTERMEDIATE_SIZE,
            },
            ModelVariant::Custom => {
                anyhow::bail!("custom DeepSeek V4 sparse coordinator geometry is unsupported")
            }
        };
        anyhow::ensure!(
            facts.hidden_size == profile.hidden
                && facts.routed_experts == profile.routed_experts
                && facts.top_k == profile.top_k
                && facts.moe_intermediate_size == profile.shared_intermediate
                && facts.shared_experts == DS4_NUM_SHARED_EXPERTS
                && facts.vocab_size == DS4_VOCAB,
            "DeepSeek V4 {:?} sparse coordinator requires hidden={}/experts={}/top-k={}/shared={}x{}/vocab={}; got hidden={}/experts={}/top-k={}/shared={}x{}/vocab={}",
            profile.variant,
            profile.hidden,
            profile.routed_experts,
            profile.top_k,
            DS4_NUM_SHARED_EXPERTS,
            profile.shared_intermediate,
            DS4_VOCAB,
            facts.hidden_size,
            facts.routed_experts,
            facts.top_k,
            facts.shared_experts,
            facts.moe_intermediate_size,
            facts.vocab_size,
        );
        let expected_scale = match profile.variant {
            ModelVariant::Flash => 1.5_f32,
            ModelVariant::Pro => 2.5_f32,
            ModelVariant::Custom => unreachable!("custom profile rejected above"),
        };
        anyhow::ensure!(
            facts.scoring_function == "sqrtsoftplus"
                && facts.topk_method == "noaux_tc"
                && facts.routed_scaling_factor == expected_scale,
            "DeepSeek V4 {:?} router requires sqrtsoftplus/noaux_tc scaling={expected_scale}; got {}/{}/{}",
            profile.variant,
            facts.scoring_function,
            facts.topk_method,
            facts.routed_scaling_factor,
        );
        if profile.variant == ModelVariant::Pro {
            anyhow::ensure!(
                facts.quantization_recipe == DEEPSEEK_V4_EXL3_RECIPE,
                "DeepSeek V4 Pro sparse coordinator requires calibrated recipe {DEEPSEEK_V4_EXL3_RECIPE}, got {}",
                facts.quantization_recipe
            );
        }
        Ok(profile)
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) struct Ds4FlashSparseLayerResidentWeights {
    variant: ModelVariant,
    hidden: usize,
    shared_intermediate: usize,
    pub(in crate::commands::real_full) ffn_norm_name: String,
    pub(in crate::commands::real_full) router_weight_name: String,
    pub(in crate::commands::real_full) router_aux_name: String,
    pub(in crate::commands::real_full) hash_routing: bool,
    pub(in crate::commands::real_full) shared_w1_weight_name: String,
    pub(in crate::commands::real_full) shared_w1_scale_name: String,
    pub(in crate::commands::real_full) shared_w3_weight_name: String,
    pub(in crate::commands::real_full) shared_w3_scale_name: String,
    pub(in crate::commands::real_full) shared_w2_weight_name: String,
    pub(in crate::commands::real_full) shared_w2_scale_name: String,
}

#[derive(Clone, Copy, Debug)]
#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) struct Ds4FlashSparseBlockPreDispatchDeviceBuffers {
    pub(in crate::commands::real_full) router: Ds4FlashRouterTopKDeviceBuffers,
    pub(in crate::commands::real_full) shared: Ds4FlashSharedExpertDeviceBuffers,
}

/// Coordinator graph segment one: produce one global top-6 route buffer and
/// the coordinator-owned shared-expert delta from the same normalized hidden.
/// The route indices and weights are then fanned out to every TP rank across
/// the transport barrier; no expert ownership map is applied here.
#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) fn ds4_flash_sparse_block_pre_dispatch_into(
    weights: &Ds4FlashSparseLayerResidentWeights,
    hidden: &DeviceBf16Output,
    token_ids: Option<Ds41rtDeviceBuffer>,
    buffers: Ds4FlashSparseBlockPreDispatchDeviceBuffers,
    cuda_stream: *mut c_void,
) -> Result<()> {
    ds4_flash_router_topk_bf16_preloaded_device_input_into(
        &weights.router_weight_name,
        &weights.router_aux_name,
        hidden,
        token_ids,
        weights.hash_routing,
        buffers.router,
        cuda_stream,
    )
    .context("launching DeepSeek V4 Flash global router before TP4 dispatch")?;
    ds4_flash_shared_expert_fp8_bf16_device_output_into(
        &weights.shared_w1_weight_name,
        &weights.shared_w1_scale_name,
        &weights.shared_w3_weight_name,
        &weights.shared_w3_scale_name,
        &weights.shared_w2_weight_name,
        &weights.shared_w2_scale_name,
        hidden,
        hidden.rows,
        buffers.shared,
        cuda_stream,
    )
    .context("launching DeepSeek V4 Flash shared expert before TP4 dispatch")
}

/// Arena-backed form of graph segment one for integrated dSpark. Attention
/// has already produced five normalized global rows on this same stream. The
/// router writes one global route image for all four TP ranks, while the
/// coordinator computes the shared-expert delta into fixed workspace views.
#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) fn ds4_flash_sparse_block_pre_dispatch_fixed_input_into(
    weights: &Ds4FlashSparseLayerResidentWeights,
    hidden: Ds41rtDeviceBuffer,
    rows: usize,
    token_ids: Option<Ds41rtDeviceBuffer>,
    buffers: Ds4FlashSparseBlockPreDispatchDeviceBuffers,
    cuda_stream: *mut c_void,
) -> Result<()> {
    anyhow::ensure!(
        weights.variant == ModelVariant::Flash,
        "Flash sparse pre-dispatch received {:?} resident weights",
        weights.variant
    );
    ds4_sparse_block_pre_dispatch_fixed_input_into(
        weights,
        hidden,
        rows,
        token_ids,
        buffers,
        cuda_stream,
    )
}

/// Profile-driven fixed-input sparse pre-dispatch used by integrated dSpark.
/// Both variants retain the same coordinator route image and shared-expert
/// launch; only their exact model geometry selects a native kernel family.
#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) fn ds4_sparse_block_pre_dispatch_fixed_input_into(
    weights: &Ds4FlashSparseLayerResidentWeights,
    hidden: Ds41rtDeviceBuffer,
    rows: usize,
    token_ids: Option<Ds41rtDeviceBuffer>,
    buffers: Ds4FlashSparseBlockPreDispatchDeviceBuffers,
    cuda_stream: *mut c_void,
) -> Result<()> {
    match weights.variant {
        ModelVariant::Flash => ds4_flash_router_topk_bf16_preloaded_fixed_device_input_into(
            &weights.router_weight_name,
            &weights.router_aux_name,
            hidden,
            rows,
            token_ids,
            weights.hash_routing,
            buffers.router,
            cuda_stream,
        ),
        ModelVariant::Pro => ds4_pro_router_topk_bf16_preloaded_fixed_device_input_into(
            &weights.router_weight_name,
            &weights.router_aux_name,
            hidden,
            rows,
            token_ids,
            weights.hash_routing,
            buffers.router,
            cuda_stream,
        ),
        ModelVariant::Custom => anyhow::bail!("custom DeepSeek V4 router is unsupported"),
    }
    .context("launching DeepSeek V4 global router from fixed dSpark input")?;
    ds4_shared_expert_fp8_bf16_preloaded_device_input_into_for_geometry(
        &weights.shared_w1_weight_name,
        &weights.shared_w1_scale_name,
        &weights.shared_w3_weight_name,
        &weights.shared_w3_scale_name,
        &weights.shared_w2_weight_name,
        &weights.shared_w2_scale_name,
        hidden,
        rows,
        weights.hidden,
        weights.shared_intermediate,
        buffers.shared,
        cuda_stream,
    )
    .context("launching DeepSeek V4 shared expert from fixed dSpark input")
}

/// Stage the global router and shared expert for one real Flash sparse layer.
/// Expert TP-rank slabs have independent residency and are loaded on Sparks.
#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) fn preload_ds4_flash_sparse_layer_resident_weights(
    catalog: &TensorCatalog,
    layer_id: usize,
) -> Result<Ds4FlashSparseLayerResidentWeights> {
    let weights = preload_ds4_sparse_layer_resident_weights(catalog, layer_id)?;
    anyhow::ensure!(
        weights.variant == ModelVariant::Flash,
        "Flash sparse preload received {:?} model facts",
        weights.variant
    );
    Ok(weights)
}

#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) fn preload_ds4_sparse_layer_resident_weights(
    catalog: &TensorCatalog,
    layer_id: usize,
) -> Result<Ds4FlashSparseLayerResidentWeights> {
    let profile = Ds4SparseProfile::from_facts(&catalog.facts)?;
    let weights = ds4_sparse_layer_resident_weight_names(&catalog.facts, layer_id)?;
    let hash_routing = weights.hash_routing;
    preload_ds4_sparse_block_tensor(
        catalog,
        &weights.router_weight_name,
        DType::Bf16,
        &[profile.routed_experts, profile.hidden],
        std::mem::size_of::<u16>(),
    )?;
    if hash_routing {
        preload_ds4_sparse_block_tensor(
            catalog,
            &weights.router_aux_name,
            DType::I64,
            &[DS4_VOCAB, profile.top_k],
            std::mem::size_of::<i64>(),
        )?;
    } else {
        preload_ds4_sparse_block_tensor(
            catalog,
            &weights.router_aux_name,
            DType::F32,
            &[profile.routed_experts],
            std::mem::size_of::<f32>(),
        )?;
    }
    let specs = [
        (
            weights.shared_w1_weight_name.as_str(),
            DType::F8E4M3,
            [profile.shared_intermediate, profile.hidden],
        ),
        (
            weights.shared_w1_scale_name.as_str(),
            DType::F8E8M0,
            [profile.shared_intermediate / 128, profile.hidden / 128],
        ),
        (
            weights.shared_w3_weight_name.as_str(),
            DType::F8E4M3,
            [profile.shared_intermediate, profile.hidden],
        ),
        (
            weights.shared_w3_scale_name.as_str(),
            DType::F8E8M0,
            [profile.shared_intermediate / 128, profile.hidden / 128],
        ),
        (
            weights.shared_w2_weight_name.as_str(),
            DType::F8E4M3,
            [profile.hidden, profile.shared_intermediate],
        ),
        (
            weights.shared_w2_scale_name.as_str(),
            DType::F8E8M0,
            [profile.hidden / 128, profile.shared_intermediate / 128],
        ),
    ];
    for (name, dtype, shape) in specs {
        preload_ds4_sparse_block_tensor(catalog, name, dtype, &shape, 1)?;
    }
    preload_ds4_shared_expert_fp8_runtime_for_geometry(
        profile.hidden,
        profile.shared_intermediate,
    )?;
    preload_ds4_flash_fp8_block_scale_mma(
        &weights.shared_w1_scale_name,
        profile.shared_intermediate,
        profile.hidden,
    )?;
    preload_ds4_flash_fp8_block_scale_mma(
        &weights.shared_w3_scale_name,
        profile.shared_intermediate,
        profile.hidden,
    )?;
    preload_ds4_flash_fp8_block_scale_mma(
        &weights.shared_w2_scale_name,
        profile.hidden,
        profile.shared_intermediate,
    )?;
    Ok(weights)
}

pub(in crate::commands::real_full) fn ds4_flash_sparse_layer_resident_weight_names(
    facts: &ModelFacts,
    layer_id: usize,
) -> Result<Ds4FlashSparseLayerResidentWeights> {
    let weights = ds4_sparse_layer_resident_weight_names(facts, layer_id)?;
    anyhow::ensure!(
        weights.variant == ModelVariant::Flash,
        "Flash sparse names received {:?} model facts",
        weights.variant
    );
    Ok(weights)
}

pub(in crate::commands::real_full) fn ds4_sparse_layer_resident_weight_names(
    facts: &ModelFacts,
    layer_id: usize,
) -> Result<Ds4FlashSparseLayerResidentWeights> {
    let profile = Ds4SparseProfile::from_facts(facts)?;
    anyhow::ensure!(
        layer_id < facts.total_transformer_blocks(),
        "DeepSeek V4 sparse layer {layer_id} exceeds checkpoint depth {}",
        facts.total_transformer_blocks()
    );
    let attention_plan = DeepseekV4AttentionPlan::from_model_facts(facts)
        .context("resolving DeepSeek V4 sparse-block checkpoint namespace")?;
    let source = attention_plan
        .layer(layer_id)
        .with_context(|| format!("DeepSeek V4 sparse block {layer_id} is not in the model plan"))?
        .source;
    let block_prefix = source.checkpoint_block_prefix();
    let hash_routing = matches!(
        source,
        DeepseekV4AttentionLayerSource::Target { layer_id }
            if layer_id < facts.num_hash_layers
    );
    let ffn_prefix = format!("{block_prefix}.ffn");
    Ok(Ds4FlashSparseLayerResidentWeights {
        variant: profile.variant,
        hidden: profile.hidden,
        shared_intermediate: profile.shared_intermediate,
        ffn_norm_name: format!("{block_prefix}.ffn_norm.weight"),
        router_weight_name: format!("{ffn_prefix}.gate.weight"),
        router_aux_name: if hash_routing {
            format!("{ffn_prefix}.gate.tid2eid")
        } else {
            format!("{ffn_prefix}.gate.bias")
        },
        hash_routing,
        shared_w1_weight_name: format!("{ffn_prefix}.shared_experts.w1.weight"),
        shared_w1_scale_name: format!("{ffn_prefix}.shared_experts.w1.scale"),
        shared_w3_weight_name: format!("{ffn_prefix}.shared_experts.w3.weight"),
        shared_w3_scale_name: format!("{ffn_prefix}.shared_experts.w3.scale"),
        shared_w2_weight_name: format!("{ffn_prefix}.shared_experts.w2.weight"),
        shared_w2_scale_name: format!("{ffn_prefix}.shared_experts.w2.scale"),
    })
}

fn preload_ds4_sparse_block_tensor(
    catalog: &TensorCatalog,
    name: &str,
    dtype: DType,
    shape: &[usize],
    bytes_per_value: usize,
) -> Result<()> {
    let info = catalog
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .with_context(|| format!("DeepSeek V4 tensor {name} is missing"))?;
    anyhow::ensure!(
        info.dtype == dtype && info.shape == shape,
        "DeepSeek V4 tensor {name} expected {dtype:?} {shape:?}, got {:?} {:?}",
        info.dtype,
        info.shape
    );
    let bytes = shape
        .iter()
        .try_fold(bytes_per_value, |bytes, &dimension| {
            bytes.checked_mul(dimension)
        })
        .context("DeepSeek V4 sparse tensor shape overflows usize")?;
    anyhow::ensure!(
        info.byte_length as usize == bytes,
        "DeepSeek V4 tensor {name} has {} bytes, expected {bytes}",
        info.byte_length
    );
    preload_resident_weight_from_host_staging(
        name,
        bytes,
        "DeepSeek V4 sparse block resident tensor",
        |staging| {
            let summary = read_tensor_bytes_into(catalog, name, staging)
                .with_context(|| format!("reading DeepSeek V4 tensor {name}"))?;
            anyhow::ensure!(
                summary.dtype == dtype
                    && summary.shape == shape
                    && summary.bytes_read as usize == bytes,
                "DeepSeek V4 tensor {name} changed during resident preload"
            );
            Ok(())
        },
    )
}

/// Fixed coordinator receive buffers and outputs for the post-dispatch half
/// of a DeepSeek V4 Flash sparse block. Each Spark returns a full hidden-width
/// BF16 partial because TP=4 partitions expert intermediate width, not experts.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) struct Ds4FlashTp4ReductionDeviceBuffers {
    pub(in crate::commands::real_full) shared_delta: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) rank_partials: [Ds41rtDeviceBuffer; DS4_FLASH_TP],
    pub(in crate::commands::real_full) reduction_f32: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) ffn_delta: Ds41rtDeviceBuffer,
}

#[derive(Clone, Copy, Debug)]
#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) struct Ds4FlashTp4HostPartialChunk<'a> {
    pub(in crate::commands::real_full) rank: usize,
    pub(in crate::commands::real_full) payload_bf16: &'a [u8],
    pub(in crate::commands::real_full) pinned_payload: Option<Ds41rtHostBuffer>,
    pub(in crate::commands::real_full) global_row_indices: &'a [usize],
}

/// Long-lived coordinator receive arena for the strict TP=4 transport barrier.
/// TCP qualification stages through four independent pinned buffers; a future
/// verbs/GPU-direct path can target the same four device pointers directly.
#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) struct Ds4FlashTp4CoordinatorReductionArena {
    library: &'static NativeLibrary,
    reduction_stream: CoordinatorCudaStream,
    max_rows: usize,
    hidden_size: usize,
    rank_partials: [OwnedCoordinatorDeviceBuffer; DS4_FLASH_TP],
    reduction_f32: OwnedCoordinatorDeviceBuffer,
    ffn_delta: OwnedCoordinatorDeviceBuffer,
    rank_host_staging: [ReusableHostBuffer; DS4_FLASH_TP],
}

// The shared scheduler wraps the arena in a mutex. Its CUDA allocations,
// pinned buffers, and stream are not thread-affine, and every stage/reduction
// mutation is serialized by that mutex.
unsafe impl Send for Ds4FlashTp4CoordinatorReductionArena {}

#[cfg_attr(not(test), allow(dead_code))]
impl Ds4FlashTp4CoordinatorReductionArena {
    pub(in crate::commands::real_full) fn new(max_rows: usize) -> Result<Self> {
        Self::new_for_hidden_size(max_rows, DS4_FLASH_HIDDEN)
    }

    pub(in crate::commands::real_full) fn new_for_hidden_size(
        max_rows: usize,
        hidden_size: usize,
    ) -> Result<Self> {
        anyhow::ensure!(
            (1..=DS4_FLASH_MAX_ROWS).contains(&max_rows),
            "DeepSeek V4 TP4 coordinator arena rows {max_rows} are outside 1..={DS4_FLASH_MAX_ROWS}"
        );
        anyhow::ensure!(
            hidden_size > 0,
            "DeepSeek V4 TP4 coordinator arena hidden size must be positive"
        );
        let library = cuda_native_library()?;
        let reduction_stream = CoordinatorCudaStream::create(library)
            .context("creating DeepSeek V4 TP4 coordinator reduction stream")?;
        let values = max_rows
            .checked_mul(hidden_size)
            .context("DeepSeek V4 TP4 coordinator arena shape overflows usize")?;
        let bf16_bytes = values
            .checked_mul(std::mem::size_of::<u16>())
            .context("DeepSeek V4 Flash TP4 coordinator arena BF16 bytes overflow usize")?;
        let f32_bytes = values
            .checked_mul(std::mem::size_of::<f32>())
            .context("DeepSeek V4 Flash TP4 coordinator arena FP32 bytes overflow usize")?;
        let rank_partials = [
            OwnedCoordinatorDeviceBuffer::new(library, bf16_bytes, "DS4 Flash TP4 rank-0 receive")?,
            OwnedCoordinatorDeviceBuffer::new(library, bf16_bytes, "DS4 Flash TP4 rank-1 receive")?,
            OwnedCoordinatorDeviceBuffer::new(library, bf16_bytes, "DS4 Flash TP4 rank-2 receive")?,
            OwnedCoordinatorDeviceBuffer::new(library, bf16_bytes, "DS4 Flash TP4 rank-3 receive")?,
        ];
        let reduction_f32 = OwnedCoordinatorDeviceBuffer::new(
            library,
            f32_bytes,
            "DS4 Flash TP4 ordered FP32 reduction",
        )?;
        let ffn_delta = OwnedCoordinatorDeviceBuffer::new(
            library,
            bf16_bytes,
            "DS4 Flash TP4 reduced BF16 delta",
        )?;
        let mut rank_host_staging = std::array::from_fn(|_| ReusableHostBuffer::default());
        for (rank, staging) in rank_host_staging.iter_mut().enumerate() {
            let label = [
                "DS4 Flash TP4 rank-0 host staging",
                "DS4 Flash TP4 rank-1 host staging",
                "DS4 Flash TP4 rank-2 host staging",
                "DS4 Flash TP4 rank-3 host staging",
            ][rank];
            staging.ensure_capacity(library, bf16_bytes, label)?;
        }
        let arena = Self {
            library,
            reduction_stream,
            max_rows,
            hidden_size,
            rank_partials,
            reduction_f32,
            ffn_delta,
            rank_host_staging,
        };
        anyhow::ensure!(
            arena
                .rank_partials
                .iter()
                .chain(std::iter::once(&arena.reduction_f32))
                .chain(std::iter::once(&arena.ffn_delta))
                .all(|buffer| buffer.buffer.device_id == 0),
            "DeepSeek V4 Flash TP4 coordinator arena must be allocated on GPU0"
        );
        Ok(arena)
    }

    pub(in crate::commands::real_full) fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Preserve the asynchronous shared-expert producer/consumer dependency
    /// without forcing a host synchronization before the TP4 reduction.
    pub(in crate::commands::real_full) fn wait_device_input_ready(
        &self,
        input: &DeviceBf16Output,
    ) -> Result<()> {
        input
            .wait_ready_on_stream(self.reduction_stream.as_ptr())
            .context("waiting for DeepSeek V4 TP4 coordinator reduction input")
    }

    /// Launch the fixed ordered reduction on the arena's persistent stream and
    /// establish a host-visible completion boundary before another scheduler
    /// lane can reuse the receive pointers.
    pub(in crate::commands::real_full) fn reduce_synchronized(
        &self,
        rows: usize,
        buffers: Ds4FlashTp4ReductionDeviceBuffers,
    ) -> Result<Ds41rtDeviceBuffer> {
        ds4_tp4_reduce_bf16_device_inputs_into_for_hidden_size(
            rows,
            self.hidden_size,
            buffers,
            self.reduction_stream.as_ptr(),
        )?;
        self.reduction_stream
            .synchronize()
            .context("synchronizing DeepSeek V4 Flash TP4 coordinator reduction")?;
        Ok(buffers.ffn_delta)
    }

    /// Copy four stable-rank TCP payloads outside CUDA capture. No scatter or
    /// reduction occurs here: every payload must already contain every row.
    pub(in crate::commands::real_full) fn stage_host_partials(
        &mut self,
        rows: usize,
        partial_outputs_bf16_by_rank: &[impl AsRef<[u8]>],
        global_row_indices_by_rank: &[impl AsRef<[usize]>],
        shared_delta: Ds41rtDeviceBuffer,
    ) -> Result<Ds4FlashTp4ReductionDeviceBuffers> {
        anyhow::ensure!(
            rows <= self.max_rows,
            "DeepSeek V4 Flash TP4 receive rows {rows} exceed fixed arena capacity {}",
            self.max_rows
        );
        let rank_bytes = validate_ds4_tp4_host_partial_layout_for_hidden_size(
            rows,
            self.hidden_size,
            partial_outputs_bf16_by_rank,
            global_row_indices_by_rank,
        )?;
        let buffers = self.reduction_device_buffers(rows, shared_delta)?;
        for rank in 0..DS4_FLASH_TP {
            let payload = partial_outputs_bf16_by_rank[rank].as_ref();
            let staging = self.rank_host_staging[rank].buffer;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    payload.as_ptr(),
                    staging.ptr.cast::<u8>(),
                    rank_bytes,
                );
            }
            self.library
                .copy_host_buffer_h2d(buffers.rank_partials[rank], staging, rank_bytes)
                .with_context(|| {
                    format!("staging DeepSeek V4 Flash Spark rank {rank} partial on GPU0")
                })?;
        }
        Ok(buffers)
    }

    /// Reconstruct striped/multi-rail verbs responses by explicit Spark rank.
    /// Every `(rank, global_row)` pair must appear exactly once; the completed
    /// pinned rank images are copied into the same fixed GPU0 receive pointers
    /// used by the unstriped TCP qualification path.
    fn stage_host_partial_chunks(
        &mut self,
        rows: usize,
        chunks: &[Ds4FlashTp4HostPartialChunk<'_>],
        shared_delta: Ds41rtDeviceBuffer,
    ) -> Result<Ds4FlashTp4ReductionDeviceBuffers> {
        anyhow::ensure!(
            rows <= self.max_rows,
            "DeepSeek V4 Flash TP4 chunk rows {rows} exceed fixed arena capacity {}",
            self.max_rows
        );
        let rank_bytes =
            validate_ds4_tp4_host_partial_chunks_for_hidden_size(rows, self.hidden_size, chunks)?;
        if chunks.iter().all(|chunk| chunk.pinned_payload.is_some()) {
            return self.stage_pinned_host_partial_chunks(rows, chunks, shared_delta);
        }
        let row_bytes = self.hidden_size * std::mem::size_of::<u16>();
        let buffers = self.reduction_device_buffers(rows, shared_delta)?;
        for chunk in chunks {
            let staging = self.rank_host_staging[chunk.rank].buffer;
            for (local_row, &global_row) in chunk.global_row_indices.iter().enumerate() {
                let src_offset = local_row * row_bytes;
                let dst_offset = global_row * row_bytes;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        chunk.payload_bf16.as_ptr().add(src_offset),
                        staging.ptr.cast::<u8>().add(dst_offset),
                        row_bytes,
                    );
                }
            }
        }
        for rank in 0..DS4_FLASH_TP {
            self.library
                .copy_host_buffer_h2d(
                    buffers.rank_partials[rank],
                    self.rank_host_staging[rank].buffer,
                    rank_bytes,
                )
                .with_context(|| {
                    format!(
                        "staging reconstructed DeepSeek V4 Flash Spark rank {rank} partial on GPU0"
                    )
                })?;
        }
        Ok(buffers)
    }

    /// Keep the response-frame borrow alive through the one synchronization
    /// that completes both direct H2D landing and the ordered reduction.
    pub(in crate::commands::real_full) fn stage_host_partial_chunks_and_reduce_synchronized(
        &mut self,
        rows: usize,
        chunks: &[Ds4FlashTp4HostPartialChunk<'_>],
        shared_delta: Ds41rtDeviceBuffer,
    ) -> Result<Ds41rtDeviceBuffer> {
        let buffers = self.stage_host_partial_chunks(rows, chunks, shared_delta)?;
        self.reduce_synchronized(rows, buffers)
    }

    /// Land pinned verbs responses and enqueue the ordered BF16 reduction into
    /// an owned output.  The caller retains the response frames on that output,
    /// so neither their pinned storage nor the shared-expert input can be
    /// recycled before the recorded reduction event completes.
    pub(in crate::commands::real_full) fn stage_host_partial_chunks_and_reduce_async_owned(
        &mut self,
        rows: usize,
        chunks: &[Ds4FlashTp4HostPartialChunk<'_>],
        shared_delta: Ds41rtDeviceBuffer,
    ) -> Result<DeviceBf16Output> {
        let result = (|| {
            let mut output = device_bf16_output_uninitialized(
                rows,
                self.hidden_size,
                DS4_TP4_ASYNC_OWNED_REDUCTION_BACKEND,
                "DeepSeek V4 TP4 asynchronous reduced FFN delta",
            )?;
            let mut buffers = self.stage_host_partial_chunks(rows, chunks, shared_delta)?;
            buffers.ffn_delta = output.buffer();
            ds4_tp4_reduce_bf16_device_inputs_into_for_hidden_size(
                rows,
                self.hidden_size,
                buffers,
                self.reduction_stream.as_ptr(),
            )?;
            let ready_event = Arc::new(CoordinatorCudaEvent::create(self.library)?);
            ready_event
                .record(self.reduction_stream.as_ptr())
                .context("recording asynchronous DeepSeek V4 TP4 reduction")?;
            output.set_ready_event(ready_event);
            Ok(output)
        })();
        if result.is_err() {
            // A failed enqueue must not let the caller recycle pinned response
            // frames while an earlier H2D operation can still reference them.
            let _ = self.reduction_stream.synchronize();
        }
        result
    }

    /// Land recyclable verbs response payloads directly in their fixed GPU0
    /// rank images. Copies share the arena reduction stream, so multi-rail
    /// chunks can be enqueued without an intermediate host reconstruction or
    /// per-copy synchronization; `reduce_synchronized` is the single lifetime
    /// barrier before the response frames are recycled.
    fn stage_pinned_host_partial_chunks(
        &mut self,
        rows: usize,
        chunks: &[Ds4FlashTp4HostPartialChunk<'_>],
        shared_delta: Ds41rtDeviceBuffer,
    ) -> Result<Ds4FlashTp4ReductionDeviceBuffers> {
        let row_bytes = self.hidden_size * std::mem::size_of::<u16>();
        let buffers = self.reduction_device_buffers(rows, shared_delta)?;
        let stage_result = (|| {
            for chunk in chunks {
                let pinned = chunk.pinned_payload.context(
                    "DeepSeek V4 Flash TP4 direct verbs landing requires pinned payloads",
                )?;
                anyhow::ensure!(
                    !pinned.ptr.is_null()
                        && pinned.bytes == chunk.payload_bf16.len()
                        && pinned.flags & DS41RT_HOST_BUFFER_FLAG_PINNED != 0,
                    "DeepSeek V4 Flash TP4 rank {} direct verbs payload is not an exact pinned view",
                    chunk.rank
                );
                let mut run_start = 0_usize;
                while run_start < chunk.global_row_indices.len() {
                    let mut run_end = run_start + 1;
                    while run_end < chunk.global_row_indices.len()
                        && chunk.global_row_indices[run_end]
                            == chunk.global_row_indices[run_end - 1] + 1
                    {
                        run_end += 1;
                    }
                    let src_offset = run_start
                        .checked_mul(row_bytes)
                        .context("DS4 Flash TP4 pinned source offset overflow")?;
                    let dst_offset = chunk.global_row_indices[run_start]
                        .checked_mul(row_bytes)
                        .context("DS4 Flash TP4 pinned destination offset overflow")?;
                    let copy_bytes = (run_end - run_start)
                        .checked_mul(row_bytes)
                        .context("DS4 Flash TP4 pinned copy bytes overflow")?;
                    let src = Ds41rtHostBuffer {
                        ptr: unsafe { pinned.ptr.cast::<u8>().add(src_offset).cast() },
                        bytes: copy_bytes,
                        flags: pinned.flags,
                    };
                    let dst = device_buffer_byte_view(
                        buffers.rank_partials[chunk.rank],
                        dst_offset,
                        copy_bytes,
                        "DS4 Flash TP4 direct verbs rank destination",
                    )?;
                    unsafe {
                        self.library.copy_host_buffer_h2d_async(
                            dst,
                            src,
                            copy_bytes,
                            self.reduction_stream.as_ptr(),
                        )
                    }
                    .with_context(|| {
                        format!(
                            "landing DeepSeek V4 Flash Spark rank {} rows {}..{} directly on GPU0",
                            chunk.rank,
                            chunk.global_row_indices[run_start],
                            chunk.global_row_indices[run_end - 1] + 1
                        )
                    })?;
                    run_start = run_end;
                }
            }
            Ok(())
        })();
        if let Err(error) = stage_result {
            let _ = self.reduction_stream.synchronize();
            return Err(error);
        }
        Ok(buffers)
    }

    fn reduction_device_buffers(
        &self,
        rows: usize,
        shared_delta: Ds41rtDeviceBuffer,
    ) -> Result<Ds4FlashTp4ReductionDeviceBuffers> {
        let buffers = Ds4FlashTp4ReductionDeviceBuffers {
            shared_delta,
            rank_partials: self.rank_partials.each_ref().map(|buffer| buffer.buffer),
            reduction_f32: self.reduction_f32.buffer,
            ffn_delta: self.ffn_delta.buffer,
        };
        validate_ds4_tp4_reduction_buffer_layout_for_hidden_size(rows, self.hidden_size, buffers)?;
        Ok(buffers)
    }
}

/// Reduce shared,S0,S1,S2,S3 in that exact FP32 order and cast once to BF16.
/// Transport has completed before this boundary and all pointers are owned by
/// the caller, so the operation is allocation-free and CUDA-graph safe.
#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) fn ds4_flash_tp4_reduce_bf16_device_inputs_into(
    rows: usize,
    buffers: Ds4FlashTp4ReductionDeviceBuffers,
    cuda_stream: *mut c_void,
) -> Result<()> {
    ds4_tp4_reduce_bf16_device_inputs_into_for_hidden_size(
        rows,
        DS4_FLASH_HIDDEN,
        buffers,
        cuda_stream,
    )
}

fn ds4_tp4_reduce_bf16_device_inputs_into_for_hidden_size(
    rows: usize,
    hidden_size: usize,
    buffers: Ds4FlashTp4ReductionDeviceBuffers,
    cuda_stream: *mut c_void,
) -> Result<()> {
    let values =
        validate_ds4_tp4_reduction_buffer_layout_for_hidden_size(rows, hidden_size, buffers)?;
    let library = cuda_native_library()?;
    unsafe {
        library
            .cuda_zero_f32_async(buffers.reduction_f32, values, cuda_stream)
            .context("zeroing DeepSeek V4 Flash TP4 FP32 reduction")?;
        for (source_index, source) in std::iter::once(buffers.shared_delta)
            .chain(buffers.rank_partials)
            .enumerate()
        {
            library
                .cuda_accumulate_bf16_to_f32_async(
                    source,
                    buffers.reduction_f32,
                    values,
                    cuda_stream,
                )
                .with_context(|| {
                    let source = if source_index == 0 {
                        "shared".to_owned()
                    } else {
                        format!("Spark rank {}", source_index - 1)
                    };
                    format!("accumulating DeepSeek V4 Flash {source} delta in FP32")
                })?;
        }
        library
            .cuda_f32_to_bf16_async(
                buffers.reduction_f32,
                buffers.ffn_delta,
                values,
                cuda_stream,
            )
            .context("casting DeepSeek V4 Flash TP4 reduction once to BF16")?;
    }
    Ok(())
}

fn validate_ds4_flash_tp4_reduction_buffer_layout(
    rows: usize,
    buffers: Ds4FlashTp4ReductionDeviceBuffers,
) -> Result<usize> {
    validate_ds4_tp4_reduction_buffer_layout_for_hidden_size(rows, DS4_FLASH_HIDDEN, buffers)
}

fn validate_ds4_tp4_reduction_buffer_layout_for_hidden_size(
    rows: usize,
    hidden_size: usize,
    buffers: Ds4FlashTp4ReductionDeviceBuffers,
) -> Result<usize> {
    anyhow::ensure!(
        (1..=DS4_FLASH_MAX_ROWS).contains(&rows),
        "DeepSeek V4 Flash TP4 reduction rows {rows} are outside 1..={DS4_FLASH_MAX_ROWS}"
    );
    anyhow::ensure!(
        hidden_size > 0,
        "DeepSeek V4 TP4 reduction hidden size must be positive"
    );
    let values = rows
        .checked_mul(hidden_size)
        .context("DeepSeek V4 Flash TP4 reduction shape overflows usize")?;
    let bf16_bytes = values
        .checked_mul(std::mem::size_of::<u16>())
        .context("DeepSeek V4 Flash TP4 BF16 byte count overflows usize")?;
    let f32_bytes = values
        .checked_mul(std::mem::size_of::<f32>())
        .context("DeepSeek V4 Flash TP4 FP32 byte count overflows usize")?;
    let named = [
        ("shared_delta", buffers.shared_delta, bf16_bytes),
        ("rank_0", buffers.rank_partials[0], bf16_bytes),
        ("rank_1", buffers.rank_partials[1], bf16_bytes),
        ("rank_2", buffers.rank_partials[2], bf16_bytes),
        ("rank_3", buffers.rank_partials[3], bf16_bytes),
        ("reduction_f32", buffers.reduction_f32, f32_bytes),
        ("ffn_delta", buffers.ffn_delta, bf16_bytes),
    ];
    let device_id = buffers.shared_delta.device_id;
    anyhow::ensure!(
        device_id == 0,
        "DeepSeek V4 Flash TP4 reduction must reside on coordinator GPU0, got device {device_id}"
    );
    for (name, buffer, bytes) in named {
        device_buffer_byte_view(buffer, 0, bytes, name)?;
        anyhow::ensure!(
            buffer.device_id == device_id,
            "DeepSeek V4 Flash TP4 reduction {name} must be on coordinator device {device_id}"
        );
    }
    for left in 0..named.len() {
        for right in left + 1..named.len() {
            anyhow::ensure!(
                !ds4_flash_reduction_ranges_overlap(
                    named[left].1,
                    named[left].2,
                    named[right].1,
                    named[right].2,
                )?,
                "DeepSeek V4 Flash TP4 reduction {} and {} buffers overlap",
                named[left].0,
                named[right].0,
            );
        }
    }
    Ok(values)
}

fn validate_ds4_flash_tp4_host_partial_layout(
    rows: usize,
    partial_outputs_bf16_by_rank: &[impl AsRef<[u8]>],
    global_row_indices_by_rank: &[impl AsRef<[usize]>],
) -> Result<usize> {
    validate_ds4_tp4_host_partial_layout_for_hidden_size(
        rows,
        DS4_FLASH_HIDDEN,
        partial_outputs_bf16_by_rank,
        global_row_indices_by_rank,
    )
}

fn validate_ds4_tp4_host_partial_layout_for_hidden_size(
    rows: usize,
    hidden_size: usize,
    partial_outputs_bf16_by_rank: &[impl AsRef<[u8]>],
    global_row_indices_by_rank: &[impl AsRef<[usize]>],
) -> Result<usize> {
    anyhow::ensure!(
        (1..=DS4_FLASH_MAX_ROWS).contains(&rows),
        "DeepSeek V4 Flash TP4 host partial rows {rows} are outside 1..={DS4_FLASH_MAX_ROWS}"
    );
    anyhow::ensure!(
        hidden_size > 0,
        "DeepSeek V4 TP4 host partial hidden size must be positive"
    );
    anyhow::ensure!(
        partial_outputs_bf16_by_rank.len() == DS4_FLASH_TP
            && global_row_indices_by_rank.len() == DS4_FLASH_TP,
        "DeepSeek V4 Flash TP4 host partials require exactly {DS4_FLASH_TP} rank payloads and row maps, got payloads={} row_maps={}",
        partial_outputs_bf16_by_rank.len(),
        global_row_indices_by_rank.len()
    );
    let rank_bytes = rows
        .checked_mul(hidden_size)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("DeepSeek V4 Flash TP4 host partial byte count overflows usize")?;
    for rank in 0..DS4_FLASH_TP {
        let payload = partial_outputs_bf16_by_rank[rank].as_ref();
        let row_indices = global_row_indices_by_rank[rank].as_ref();
        anyhow::ensure!(
            payload.len() == rank_bytes,
            "DeepSeek V4 Flash TP4 rank {rank} partial expected {rank_bytes} BF16 bytes, got {}",
            payload.len()
        );
        anyhow::ensure!(
            row_indices.len() == rows
                && row_indices
                    .iter()
                    .enumerate()
                    .all(|(row, &global_row)| row == global_row),
            "DeepSeek V4 Flash TP4 rank {rank} must return the complete identity row map 0..{rows}"
        );
    }
    Ok(rank_bytes)
}

fn validate_ds4_flash_tp4_host_partial_chunks(
    rows: usize,
    chunks: &[Ds4FlashTp4HostPartialChunk<'_>],
) -> Result<usize> {
    validate_ds4_tp4_host_partial_chunks_for_hidden_size(rows, DS4_FLASH_HIDDEN, chunks)
}

fn validate_ds4_tp4_host_partial_chunks_for_hidden_size(
    rows: usize,
    hidden_size: usize,
    chunks: &[Ds4FlashTp4HostPartialChunk<'_>],
) -> Result<usize> {
    anyhow::ensure!(
        (1..=DS4_FLASH_MAX_ROWS).contains(&rows),
        "DeepSeek V4 Flash TP4 host chunk rows {rows} are outside 1..={DS4_FLASH_MAX_ROWS}"
    );
    anyhow::ensure!(
        hidden_size > 0,
        "DeepSeek V4 TP4 host chunk hidden size must be positive"
    );
    anyhow::ensure!(
        !chunks.is_empty(),
        "DeepSeek V4 Flash TP4 host chunks are empty"
    );
    let row_bytes = hidden_size
        .checked_mul(std::mem::size_of::<u16>())
        .context("DeepSeek V4 Flash TP4 host chunk row bytes overflow usize")?;
    let rank_bytes = rows
        .checked_mul(row_bytes)
        .context("DeepSeek V4 Flash TP4 host chunk rank bytes overflow usize")?;
    let mut seen_rows_by_rank = vec![vec![false; rows]; DS4_FLASH_TP];
    for (chunk_index, chunk) in chunks.iter().enumerate() {
        anyhow::ensure!(
            chunk.rank < DS4_FLASH_TP,
            "DeepSeek V4 Flash TP4 host chunk {chunk_index} has invalid rank {}",
            chunk.rank
        );
        anyhow::ensure!(
            !chunk.global_row_indices.is_empty(),
            "DeepSeek V4 Flash TP4 host chunk {chunk_index} for rank {} has no rows",
            chunk.rank
        );
        let expected_bytes = chunk
            .global_row_indices
            .len()
            .checked_mul(row_bytes)
            .context("DeepSeek V4 Flash TP4 host chunk payload bytes overflow usize")?;
        anyhow::ensure!(
            chunk.payload_bf16.len() == expected_bytes,
            "DeepSeek V4 Flash TP4 host chunk {chunk_index} rank {} expected {expected_bytes} BF16 bytes, got {}",
            chunk.rank,
            chunk.payload_bf16.len()
        );
        for &global_row in chunk.global_row_indices {
            let seen = seen_rows_by_rank[chunk.rank]
                .get_mut(global_row)
                .with_context(|| {
                    format!(
                        "DeepSeek V4 Flash TP4 host chunk {chunk_index} rank {} row {global_row} exceeds {rows} rows",
                        chunk.rank
                    )
                })?;
            anyhow::ensure!(
                !*seen,
                "DeepSeek V4 Flash TP4 host chunks repeat rank {} row {global_row}",
                chunk.rank
            );
            *seen = true;
        }
    }
    for (rank, seen_rows) in seen_rows_by_rank.iter().enumerate() {
        if let Some(row) = seen_rows.iter().position(|seen| !*seen) {
            anyhow::bail!("DeepSeek V4 Flash TP4 host chunks are missing rank {rank} row {row}");
        }
    }
    Ok(rank_bytes)
}

fn ds4_flash_reduction_ranges_overlap(
    left: Ds41rtDeviceBuffer,
    left_bytes: usize,
    right: Ds41rtDeviceBuffer,
    right_bytes: usize,
) -> Result<bool> {
    let left_start = left.ptr as usize;
    let right_start = right.ptr as usize;
    let left_end = left_start
        .checked_add(left_bytes)
        .context("DeepSeek V4 Flash TP4 left buffer address overflows usize")?;
    let right_end = right_start
        .checked_add(right_bytes)
        .context("DeepSeek V4 Flash TP4 right buffer address overflows usize")?;
    Ok(left_start < right_end && right_start < left_end)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::real_full::intermediate_sharding::ExpertIntermediateShard;
    use crate::commands::real_full::sparse_mlp::route::{
        ds4_flash_spark_decode_m1_preloaded_device_input_into,
        ds4_flash_spark_prefill_topk6_preloaded_device_input_into,
        execute_native_flash_route_bf16_cached, preload_native_flash_layer_cuda_cache_local_tp4,
        Ds4FlashSparkDecodeM1DeviceBuffers, Ds4FlashSparkPrefillDeviceBuffers, RouteTensorCache,
        DS4_FLASH_DECODE_M1_ACTIVATED_BYTES, DS4_FLASH_DECODE_M1_FC1_BYTES,
        DS4_FLASH_DECODE_M1_FC1_SCRATCH_BYTES, DS4_FLASH_DECODE_M1_FC2_SCRATCH_BYTES,
        DS4_FLASH_DECODE_M1_HIDDEN_BYTES, DS4_FLASH_DECODE_M1_LOCK_BYTES,
        DS4_FLASH_DECODE_M1_OUTPUT_F32_BYTES, DS4_FLASH_DECODE_M1_ROUTED_OUTPUT_BYTES,
        DS4_FLASH_PREFILL_ACTIVATED_BYTES, DS4_FLASH_PREFILL_BLOCK_EXPERT_BYTES,
        DS4_FLASH_PREFILL_COMPUTE_SCRATCH_BYTES, DS4_FLASH_PREFILL_EXPERT_COUNT_BYTES,
        DS4_FLASH_PREFILL_EXPERT_OFFSET_BYTES, DS4_FLASH_PREFILL_FC1_BYTES,
        DS4_FLASH_PREFILL_HIDDEN_BYTES, DS4_FLASH_PREFILL_OUTPUT_F32_BYTES,
        DS4_FLASH_PREFILL_PACKED_ROUTE_BYTES, DS4_FLASH_PREFILL_ROUTED_OUTPUT_BYTES,
        DS4_FLASH_PREFILL_TOPK_ID_BYTES, DS4_FLASH_PREFILL_TOPK_WEIGHT_BYTES,
    };
    use crate::commands::real_full::sparse_mlp::router::{
        score_real_router_routes_bf16_cached_device_input, RouterTensorCache,
    };
    use ds41rt_loader::build_catalog_for_snapshot;
    use std::path::Path;

    #[test]
    fn sparse_weight_names_distinguish_target_layers_from_native_dspark_blocks() {
        let facts = ModelFacts::default();
        let target = ds4_flash_sparse_layer_resident_weight_names(&facts, 2).unwrap();
        assert!(target.hash_routing);
        assert_eq!(target.router_weight_name, "layers.2.ffn.gate.weight");
        assert_eq!(target.router_aux_name, "layers.2.ffn.gate.tid2eid");

        let dspark =
            ds4_flash_sparse_layer_resident_weight_names(&facts, facts.num_hidden_layers + 1)
                .unwrap();
        assert!(!dspark.hash_routing);
        assert_eq!(dspark.ffn_norm_name, "mtp.1.ffn_norm.weight");
        assert_eq!(dspark.router_weight_name, "mtp.1.ffn.gate.weight");
        assert_eq!(dspark.router_aux_name, "mtp.1.ffn.gate.bias");
        assert_eq!(
            dspark.shared_w2_scale_name,
            "mtp.1.ffn.shared_experts.w2.scale"
        );
    }

    fn pro_facts() -> ModelFacts {
        let mut facts = ModelFacts::default();
        facts.variant = ModelVariant::Pro;
        facts.hidden_size = DS4_PRO_HIDDEN_SIZE;
        facts.num_hidden_layers = ds41rt_core::DS4_PRO_NUM_HIDDEN_LAYERS;
        facts.routed_experts = DS4_PRO_ROUTED_EXPERTS;
        facts.top_k = DS4_PRO_TOP_K;
        facts.moe_intermediate_size = DS4_PRO_MOE_INTERMEDIATE_SIZE;
        facts.attention_heads = 128;
        facts.kv_heads = 1;
        facts.head_dim = 512;
        facts.q_lora_rank = ds41rt_core::DS4_PRO_Q_LORA_RANK;
        facts.o_groups = 16;
        facts.index_top_k = 1_024;
        facts.compress_ratios = ds41rt_core::DS4_PRO_COMPRESS_RATIOS.to_vec();
        facts.routed_scaling_factor = 2.5;
        facts.dspark_markov_rank = ds41rt_core::DS4_PRO_DSPARK_MARKOV_RANK;
        facts.dspark_target_layer_ids = vec![58, 59, 60];
        facts.quantization_recipe = DEEPSEEK_V4_EXL3_RECIPE.to_owned();
        facts
    }

    #[test]
    fn pro_sparse_names_preserve_one_global_router_and_mtp_namespace() {
        let facts = pro_facts();
        let target = ds4_sparse_layer_resident_weight_names(&facts, 2).unwrap();
        assert_eq!(target.variant, ModelVariant::Pro);
        assert_eq!(target.hidden, DS4_PRO_HIDDEN_SIZE);
        assert!(target.hash_routing);
        assert_eq!(target.router_weight_name, "layers.2.ffn.gate.weight");
        assert_eq!(target.router_aux_name, "layers.2.ffn.gate.tid2eid");

        let dspark =
            ds4_sparse_layer_resident_weight_names(&facts, facts.num_hidden_layers + 2).unwrap();
        assert!(!dspark.hash_routing);
        assert_eq!(dspark.ffn_norm_name, "mtp.2.ffn_norm.weight");
        assert_eq!(dspark.router_weight_name, "mtp.2.ffn.gate.weight");
        assert_eq!(dspark.router_aux_name, "mtp.2.ffn.gate.bias");
        assert_eq!(dspark.shared_intermediate, DS4_PRO_MOE_INTERMEDIATE_SIZE);
    }

    #[test]
    fn pro_sparse_profile_fails_closed_without_calibrated_exl3() {
        let mut facts = pro_facts();
        facts.quantization_recipe = "deepseek_v4_native_fp4_fp8_mixed_v1".to_owned();
        let error = Ds4SparseProfile::from_facts(&facts).unwrap_err();
        assert!(error.to_string().contains(DEEPSEEK_V4_EXL3_RECIPE));
    }

    fn buffer(ptr: usize, bytes: usize, device_id: i32) -> Ds41rtDeviceBuffer {
        Ds41rtDeviceBuffer {
            ptr: ptr as *mut c_void,
            bytes,
            device_id,
            flags: 0,
        }
    }

    fn assert_bf16_vectors_close(label: &str, actual: &[u8], expected: &[u8]) {
        let actual = bf16_values_to_f32(actual);
        let expected = bf16_values_to_f32(expected);
        let dot = actual
            .iter()
            .zip(&expected)
            .map(|(left, right)| left * right)
            .sum::<f32>();
        let actual_norm = actual.iter().map(|value| value * value).sum::<f32>();
        let expected_norm = expected.iter().map(|value| value * value).sum::<f32>();
        let error_norm = actual
            .iter()
            .zip(&expected)
            .map(|(left, right)| (left - right) * (left - right))
            .sum::<f32>();
        let cosine = dot / (actual_norm.sqrt() * expected_norm.sqrt());
        let relative_l2 = (error_norm / expected_norm).sqrt();
        assert!(
            cosine >= 0.999_98 && relative_l2 <= 0.006,
            "{label} differs from its TP oracle: cosine={cosine:.9} relative_l2={relative_l2:.9}"
        );
    }

    fn valid_buffers_for_hidden(
        rows: usize,
        hidden_size: usize,
    ) -> Ds4FlashTp4ReductionDeviceBuffers {
        let values = rows * hidden_size;
        let bf16_bytes = values * std::mem::size_of::<u16>();
        let f32_bytes = values * std::mem::size_of::<f32>();
        Ds4FlashTp4ReductionDeviceBuffers {
            shared_delta: buffer(0x1000_0000, bf16_bytes, 0),
            rank_partials: [
                buffer(0x2000_0000, bf16_bytes, 0),
                buffer(0x3000_0000, bf16_bytes, 0),
                buffer(0x4000_0000, bf16_bytes, 0),
                buffer(0x5000_0000, bf16_bytes, 0),
            ],
            reduction_f32: buffer(0x6000_0000, f32_bytes, 0),
            ffn_delta: buffer(0x7000_0000, bf16_bytes, 0),
        }
    }

    fn valid_buffers(rows: usize) -> Ds4FlashTp4ReductionDeviceBuffers {
        valid_buffers_for_hidden(rows, DS4_FLASH_HIDDEN)
    }

    #[test]
    fn flash_tp4_reduction_accepts_five_fixed_hidden_width_inputs() {
        let rows = 7;
        assert_eq!(
            validate_ds4_flash_tp4_reduction_buffer_layout(rows, valid_buffers(rows)).unwrap(),
            rows * DS4_FLASH_HIDDEN
        );
    }

    #[test]
    fn pro_tp4_reduction_accepts_five_fixed_7168_wide_inputs() {
        const DS4_PRO_HIDDEN: usize = 7_168;
        let rows = 7;
        assert_eq!(
            validate_ds4_tp4_reduction_buffer_layout_for_hidden_size(
                rows,
                DS4_PRO_HIDDEN,
                valid_buffers_for_hidden(rows, DS4_PRO_HIDDEN),
            )
            .unwrap(),
            rows * DS4_PRO_HIDDEN
        );
        let error = validate_ds4_tp4_reduction_buffer_layout_for_hidden_size(
            rows,
            0,
            valid_buffers_for_hidden(rows, DS4_PRO_HIDDEN),
        )
        .unwrap_err();
        assert!(error.to_string().contains("hidden size must be positive"));
    }

    #[test]
    fn flash_tp4_host_partials_require_four_complete_identity_row_payloads() {
        let rows = 3_usize;
        let rank_bytes = rows * DS4_FLASH_HIDDEN * std::mem::size_of::<u16>();
        let partials = (0..DS4_FLASH_TP)
            .map(|rank| vec![rank as u8; rank_bytes])
            .collect::<Vec<_>>();
        let row_maps = vec![(0..rows).collect::<Vec<_>>(); DS4_FLASH_TP];
        assert_eq!(
            validate_ds4_flash_tp4_host_partial_layout(rows, &partials, &row_maps).unwrap(),
            rank_bytes
        );

        let missing_rank = &partials[..DS4_FLASH_TP - 1];
        let error =
            validate_ds4_flash_tp4_host_partial_layout(rows, missing_rank, &row_maps).unwrap_err();
        assert!(error.to_string().contains("exactly 4 rank payloads"));

        let mut scattered_rows = row_maps.clone();
        scattered_rows[2].swap(0, 1);
        let error = validate_ds4_flash_tp4_host_partial_layout(rows, &partials, &scattered_rows)
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("rank 2 must return the complete identity row map"));

        let mut short_payload = partials.clone();
        short_payload[3].pop();
        let error = validate_ds4_flash_tp4_host_partial_layout(rows, &short_payload, &row_maps)
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("rank 3 partial expected 24576 BF16 bytes"));
    }

    #[test]
    fn flash_tp4_host_chunks_preserve_rank_and_cover_every_row_once() {
        let rows = 3_usize;
        let row_bytes = DS4_FLASH_HIDDEN * std::mem::size_of::<u16>();
        let ranks = [3_usize, 0, 2, 1, 3, 0, 2, 1];
        let row_maps = [
            vec![2, 0],
            vec![2, 0],
            vec![2, 0],
            vec![2, 0],
            vec![1],
            vec![1],
            vec![1],
            vec![1],
        ];
        let payloads = row_maps
            .iter()
            .enumerate()
            .map(|(chunk, row_map)| vec![chunk as u8; row_map.len() * row_bytes])
            .collect::<Vec<_>>();
        let chunks = (0..ranks.len())
            .map(|chunk| Ds4FlashTp4HostPartialChunk {
                rank: ranks[chunk],
                payload_bf16: &payloads[chunk],
                pinned_payload: None,
                global_row_indices: &row_maps[chunk],
            })
            .collect::<Vec<_>>();
        assert_eq!(
            validate_ds4_flash_tp4_host_partial_chunks(rows, &chunks).unwrap(),
            rows * row_bytes
        );

        let error = validate_ds4_flash_tp4_host_partial_chunks(rows, &chunks[..chunks.len() - 1])
            .unwrap_err();
        assert!(error.to_string().contains("missing rank 1 row 1"));

        let mut duplicate = chunks.clone();
        duplicate[7] = duplicate[3];
        let error = validate_ds4_flash_tp4_host_partial_chunks(rows, &duplicate).unwrap_err();
        assert!(error.to_string().contains("repeat rank 1 row 2"));

        let mut invalid_rank = chunks.clone();
        invalid_rank[0].rank = DS4_FLASH_TP;
        let error = validate_ds4_flash_tp4_host_partial_chunks(rows, &invalid_rank).unwrap_err();
        assert!(error.to_string().contains("invalid rank 4"));
    }

    #[test]
    fn flash_tp4_reduction_rejects_ep_like_aliasing_and_wrong_device() {
        let mut aliased = valid_buffers(1);
        aliased.rank_partials[3] = aliased.rank_partials[0];
        let error = validate_ds4_flash_tp4_reduction_buffer_layout(1, aliased).unwrap_err();
        assert!(error
            .to_string()
            .contains("rank_0 and rank_3 buffers overlap"));

        let mut wrong_device = valid_buffers(1);
        wrong_device.rank_partials[2].device_id = 1;
        let error = validate_ds4_flash_tp4_reduction_buffer_layout(1, wrong_device).unwrap_err();
        assert!(error
            .to_string()
            .contains("rank_2 must be on coordinator device 0"));

        let mut wrong_coordinator = valid_buffers(1);
        wrong_coordinator.shared_delta.device_id = 1;
        let error =
            validate_ds4_flash_tp4_reduction_buffer_layout(1, wrong_coordinator).unwrap_err();
        assert!(error
            .to_string()
            .contains("must reside on coordinator GPU0"));
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn flash_tp4_host_barrier_stages_into_fixed_gpu0_graph_inputs() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(true);
        let library = cuda_native_library().expect("loading CUDA native library");
        let rows = 3_usize;
        let values = rows * DS4_FLASH_HIDDEN;
        let bf16_bytes = values * std::mem::size_of::<u16>();
        let mut shared_delta = library
            .alloc_device_buffer(bf16_bytes)
            .expect("allocating TP4 shared delta");
        assert_eq!(shared_delta.device_id, 0);
        let shared_payload = f32_values_to_bf16_bytes(
            &(0..values)
                .map(|index| ((index * 11 % 193) as f32 - 96.0) / 1024.0)
                .collect::<Vec<_>>(),
        );
        library
            .copy_h2d(shared_delta, &shared_payload)
            .expect("uploading TP4 shared delta");
        let mut arena = Ds4FlashTp4CoordinatorReductionArena::new(7)
            .expect("allocating fixed TP4 coordinator arena");
        let row_maps = vec![(0..rows).collect::<Vec<_>>(); DS4_FLASH_TP];
        let make_payloads = |revision: usize| {
            (0..DS4_FLASH_TP)
                .map(|rank| {
                    f32_values_to_bf16_bytes(
                        &(0..values)
                            .map(|index| {
                                let modulus = 211 + rank * 16 + revision * 8;
                                ((index * (13 + rank + revision) % modulus) as f32
                                    - (modulus / 2) as f32)
                                    / (768.0 + rank as f32 * 64.0)
                            })
                            .collect::<Vec<_>>(),
                    )
                })
                .collect::<Vec<_>>()
        };
        let expected_delta = |partials: &[Vec<u8>]| {
            let mut expected = bf16_values_to_f32(&shared_payload);
            for partial in partials {
                for (dst, src) in expected.iter_mut().zip(bf16_values_to_f32(partial)) {
                    *dst += src;
                }
            }
            f32_values_to_bf16_bytes(&expected)
        };
        let first_payloads = make_payloads(0);
        let buffers = arena
            .stage_host_partials(rows, &first_payloads, &row_maps, shared_delta)
            .expect("staging first four TP4 rank payloads");
        let fixed_rank_pointers = buffers.rank_partials.map(|buffer| buffer.ptr);
        for (rank, (buffer, expected)) in buffers
            .rank_partials
            .iter()
            .copied()
            .zip(&first_payloads)
            .enumerate()
        {
            let mut actual = vec![0_u8; bf16_bytes];
            library
                .copy_d2h(&mut actual, buffer)
                .unwrap_or_else(|error| panic!("reading staged rank {rank}: {error:#}"));
            assert_eq!(&actual, expected);
        }
        let reduced = arena
            .reduce_synchronized(rows, buffers)
            .expect("reducing on the persistent TP4 arena stream");
        let mut first_actual = vec![0_u8; bf16_bytes];
        library
            .copy_d2h(&mut first_actual, reduced)
            .expect("reading persistent-stream TP4 reduced delta");
        assert_eq!(first_actual, expected_delta(&first_payloads));
        let residual = device_bf16_output_from_bf16_bytes(
            &vec![0_u8; bf16_bytes],
            rows,
            DS4_FLASH_HIDDEN,
            "TP4 arena synchronized residual handoff test",
        )
        .expect("uploading TP4 arena residual");
        let residual_output = residual_add_bf16_device_input_synchronized_delta_view_device_output(
            &residual, reduced,
        )
        .expect("applying synchronized TP4 arena delta to residual");
        assert_eq!(
            residual_output
                .copy_to_host_bytes()
                .expect("reading TP4 arena residual output"),
            first_actual
        );
        let stream = library
            .cuda_stream_create()
            .expect("creating TP4 receive reduction stream");
        unsafe {
            library
                .cuda_graph_begin_capture(stream)
                .expect("beginning TP4 receive reduction capture");
        }
        ds4_flash_tp4_reduce_bf16_device_inputs_into(rows, buffers, stream)
            .expect("capturing TP4 receive reduction");
        let capture = unsafe {
            library
                .cuda_graph_end_capture_retained(stream)
                .expect("ending TP4 receive reduction capture")
        };
        assert_eq!(capture.memcpy_node_count, 0);

        let second_payloads = make_payloads(1);
        let striped_row_maps = [
            vec![2, 0],
            vec![1],
            vec![2, 0],
            vec![1],
            vec![2, 0],
            vec![1],
            vec![2, 0],
            vec![1],
        ];
        let striped_ranks = [2_usize, 2, 0, 0, 3, 3, 1, 1];
        let striped_payloads = striped_ranks
            .iter()
            .zip(&striped_row_maps)
            .map(|(&rank, row_map)| {
                let mut payload = Vec::with_capacity(row_map.len() * DS4_FLASH_HIDDEN * 2);
                for &row in row_map {
                    let start = row * DS4_FLASH_HIDDEN * 2;
                    payload.extend_from_slice(
                        &second_payloads[rank][start..start + DS4_FLASH_HIDDEN * 2],
                    );
                }
                payload
            })
            .collect::<Vec<_>>();
        let striped_chunks = (0..striped_ranks.len())
            .map(|chunk| Ds4FlashTp4HostPartialChunk {
                rank: striped_ranks[chunk],
                payload_bf16: &striped_payloads[chunk],
                pinned_payload: None,
                global_row_indices: &striped_row_maps[chunk],
            })
            .collect::<Vec<_>>();
        let second_buffers = arena
            .stage_host_partial_chunks(rows, &striped_chunks, shared_delta)
            .expect("reconstructing striped TP4 rank payloads");
        assert_eq!(
            second_buffers.rank_partials.map(|buffer| buffer.ptr),
            fixed_rank_pointers
        );
        unsafe {
            library
                .cuda_graph_launch(capture.graph_exec, stream)
                .expect("launching TP4 reduction with refreshed receive buffers");
            library
                .cuda_stream_synchronize(stream)
                .expect("synchronizing TP4 receive reduction");
        }
        let mut actual = vec![0_u8; bf16_bytes];
        library
            .copy_d2h(&mut actual, second_buffers.ffn_delta)
            .expect("reading TP4 reduced delta");
        assert_eq!(actual, expected_delta(&second_payloads));

        let mut pinned_payloads = striped_payloads
            .iter()
            .map(|payload| {
                let host = library
                    .alloc_host_buffer(payload.len())
                    .expect("allocating pinned verbs response payload");
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        payload.as_ptr(),
                        host.ptr.cast::<u8>(),
                        payload.len(),
                    );
                }
                host
            })
            .collect::<Vec<_>>();
        let pinned_chunks = (0..striped_ranks.len())
            .map(|chunk| Ds4FlashTp4HostPartialChunk {
                rank: striped_ranks[chunk],
                payload_bf16: &striped_payloads[chunk],
                pinned_payload: Some(Ds41rtHostBuffer {
                    ptr: pinned_payloads[chunk].ptr,
                    bytes: striped_payloads[chunk].len(),
                    flags: pinned_payloads[chunk].flags,
                }),
                global_row_indices: &striped_row_maps[chunk],
            })
            .collect::<Vec<_>>();
        let pinned_reduced = arena
            .stage_host_partial_chunks_and_reduce_synchronized(rows, &pinned_chunks, shared_delta)
            .expect("landing and reducing pinned striped TP4 payloads on GPU0");
        let mut pinned_actual = vec![0_u8; bf16_bytes];
        library
            .copy_d2h(&mut pinned_actual, pinned_reduced)
            .expect("reading directly landed TP4 reduced delta");
        assert_eq!(pinned_actual, expected_delta(&second_payloads));
        let async_reduced = arena
            .stage_host_partial_chunks_and_reduce_async_owned(rows, &pinned_chunks, shared_delta)
            .expect("landing pinned TP4 payloads into an asynchronous owned reduction");
        assert_eq!(
            async_reduced
                .copy_to_host_bytes()
                .expect("reading asynchronous owned TP4 reduction"),
            expected_delta(&second_payloads)
        );
        for payload in &mut pinned_payloads {
            library
                .free_host_buffer(payload)
                .expect("freeing pinned verbs response payload");
        }

        unsafe {
            library
                .cuda_graph_destroy(capture.graph)
                .expect("destroying TP4 receive reduction graph");
            library
                .cuda_graph_exec_destroy(capture.graph_exec)
                .expect("destroying TP4 receive reduction graph exec");
            library
                .cuda_stream_destroy(stream)
                .expect("destroying TP4 receive reduction stream");
        }
        library
            .free_device_buffer(&mut shared_delta)
            .expect("freeing TP4 shared delta");
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn flash_tp4_reduction_is_fixed_pointer_and_graph_safe_on_gpu0() {
        let library = cuda_native_library().expect("loading CUDA native library");
        let rows = 7_usize;
        let values = rows * DS4_FLASH_HIDDEN;
        let bf16_bytes = values * std::mem::size_of::<u16>();
        let f32_bytes = values * std::mem::size_of::<f32>();
        let mut sources = (0..=DS4_FLASH_TP)
            .map(|source| {
                library
                    .alloc_device_buffer(bf16_bytes)
                    .unwrap_or_else(|error| panic!("allocating TP4 source {source}: {error:#}"))
            })
            .collect::<Vec<_>>();
        let mut reduction_f32 = library
            .alloc_device_buffer(f32_bytes)
            .expect("allocating TP4 FP32 reduction");
        let mut ffn_delta = library
            .alloc_device_buffer(bf16_bytes)
            .expect("allocating TP4 BF16 delta");
        let buffers = Ds4FlashTp4ReductionDeviceBuffers {
            shared_delta: sources[0],
            rank_partials: [sources[1], sources[2], sources[3], sources[4]],
            reduction_f32,
            ffn_delta,
        };
        let fixed_pointers = [
            buffers.shared_delta.ptr,
            buffers.rank_partials[0].ptr,
            buffers.rank_partials[1].ptr,
            buffers.rank_partials[2].ptr,
            buffers.rank_partials[3].ptr,
            buffers.reduction_f32.ptr,
            buffers.ffn_delta.ptr,
        ];
        let source_payloads = (0..=DS4_FLASH_TP)
            .map(|source| {
                f32_values_to_bf16_bytes(
                    &(0..values)
                        .map(|index| {
                            let modulus = 257 + source * 32;
                            ((index * (17 + source * 2) % modulus) as f32 - (modulus / 2) as f32)
                                / (512.0 + source as f32 * 128.0)
                        })
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();
        for (buffer, payload) in sources.iter().copied().zip(&source_payloads) {
            library
                .copy_h2d(buffer, payload)
                .expect("uploading fixed TP4 reduction source");
        }
        let expected = || {
            let mut expected = vec![0.0_f32; values];
            for payload in &source_payloads {
                for (dst, src) in expected.iter_mut().zip(bf16_values_to_f32(payload)) {
                    *dst += src;
                }
            }
            expected
        };
        let assert_outputs = |wanted: &[f32]| {
            let mut reduction_bytes = vec![0_u8; f32_bytes];
            library
                .copy_d2h(&mut reduction_bytes, buffers.reduction_f32)
                .expect("copying TP4 FP32 reduction");
            let reduction = reduction_bytes
                .chunks_exact(std::mem::size_of::<f32>())
                .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("four-byte f32")))
                .collect::<Vec<_>>();
            assert_eq!(reduction, wanted);
            let mut delta = vec![0_u8; bf16_bytes];
            library
                .copy_d2h(&mut delta, buffers.ffn_delta)
                .expect("copying TP4 BF16 delta");
            assert_eq!(delta, f32_values_to_bf16_bytes(wanted));
        };
        let stream = library
            .cuda_stream_create()
            .expect("creating TP4 reduction graph stream");
        ds4_flash_tp4_reduce_bf16_device_inputs_into(rows, buffers, stream)
            .expect("launching eager TP4 reduction");
        unsafe {
            library
                .cuda_stream_synchronize(stream)
                .expect("synchronizing eager TP4 reduction");
        }
        assert_outputs(&expected());

        unsafe {
            library
                .cuda_graph_begin_capture(stream)
                .expect("beginning TP4 reduction graph capture");
        }
        ds4_flash_tp4_reduce_bf16_device_inputs_into(rows, buffers, stream)
            .expect("capturing TP4 reduction");
        let capture = unsafe {
            library
                .cuda_graph_end_capture_retained(stream)
                .expect("ending TP4 reduction graph capture")
        };
        assert!(capture.kernel_node_count >= 6);
        assert!(capture.memset_node_count >= 1);
        assert_eq!(capture.memcpy_node_count, 0);
        for _ in 0..3 {
            unsafe {
                library
                    .cuda_graph_launch(capture.graph_exec, stream)
                    .expect("replaying TP4 reduction graph");
            }
        }
        unsafe {
            library
                .cuda_stream_synchronize(stream)
                .expect("synchronizing TP4 reduction graph replays");
        }
        assert_outputs(&expected());
        assert_eq!(
            fixed_pointers,
            [
                buffers.shared_delta.ptr,
                buffers.rank_partials[0].ptr,
                buffers.rank_partials[1].ptr,
                buffers.rank_partials[2].ptr,
                buffers.rank_partials[3].ptr,
                buffers.reduction_f32.ptr,
                buffers.ffn_delta.ptr,
            ]
        );
        unsafe {
            library
                .cuda_graph_destroy(capture.graph)
                .expect("destroying TP4 reduction graph");
            library
                .cuda_graph_exec_destroy(capture.graph_exec)
                .expect("destroying TP4 reduction graph exec");
            library
                .cuda_stream_destroy(stream)
                .expect("destroying TP4 reduction graph stream");
        }
        library
            .free_device_buffer(&mut ffn_delta)
            .expect("freeing TP4 BF16 delta");
        library
            .free_device_buffer(&mut reduction_f32)
            .expect("freeing TP4 FP32 reduction");
        for (source, buffer) in sources.iter_mut().enumerate().rev() {
            library
                .free_device_buffer(buffer)
                .unwrap_or_else(|error| panic!("freeing TP4 source {source}: {error:#}"));
        }
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn pro_tp4_arena_reduces_shared_then_four_rank_partials_on_gpu0() {
        const DS4_PRO_HIDDEN: usize = 7_168;
        let library = cuda_native_library().expect("loading CUDA native library");
        let rows = 2_usize;
        let values = rows * DS4_PRO_HIDDEN;
        let bf16_bytes = values * std::mem::size_of::<u16>();
        let mut shared_delta = library
            .alloc_device_buffer(bf16_bytes)
            .expect("allocating Pro TP4 shared delta");
        let shared_payload = f32_values_to_bf16_bytes(
            &(0..values)
                .map(|index| ((index * 7 % 127) as f32 - 63.0) / 512.0)
                .collect::<Vec<_>>(),
        );
        library
            .copy_h2d(shared_delta, &shared_payload)
            .expect("uploading Pro TP4 shared delta");
        let partials = (0..DS4_FLASH_TP)
            .map(|rank| {
                f32_values_to_bf16_bytes(
                    &(0..values)
                        .map(|index| {
                            let modulus = 149 + rank * 16;
                            ((index * (11 + rank) % modulus) as f32 - (modulus / 2) as f32)
                                / (640.0 + rank as f32 * 64.0)
                        })
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();
        let row_maps = vec![(0..rows).collect::<Vec<_>>(); DS4_FLASH_TP];
        let mut expected = bf16_values_to_f32(&shared_payload);
        for partial in &partials {
            for (dst, src) in expected.iter_mut().zip(bf16_values_to_f32(partial)) {
                *dst += src;
            }
        }

        let mut arena =
            Ds4FlashTp4CoordinatorReductionArena::new_for_hidden_size(rows, DS4_PRO_HIDDEN)
                .expect("allocating Pro TP4 coordinator arena");
        assert_eq!(arena.hidden_size(), DS4_PRO_HIDDEN);
        let buffers = arena
            .stage_host_partials(rows, &partials, &row_maps, shared_delta)
            .expect("staging four Pro TP4 rank payloads");
        let reduced = arena
            .reduce_synchronized(rows, buffers)
            .expect("reducing Pro TP4 payloads in fixed order");
        let mut actual = vec![0_u8; bf16_bytes];
        library
            .copy_d2h(&mut actual, reduced)
            .expect("reading Pro TP4 reduced delta");
        assert_eq!(actual, f32_values_to_bf16_bytes(&expected));
        assert!(bf16_values_to_f32(&actual)
            .iter()
            .all(|value| value.is_finite()));

        library
            .free_device_buffer(&mut shared_delta)
            .expect("freeing Pro TP4 shared delta");
    }

    #[test]
    #[ignore = "requires a real Flash checkpoint and CUDA"]
    fn real_flash_hash_sparse_block_pre_dispatch_is_graph_safe_on_gpu0() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(true);
        let snapshot = std::env::var("DS41RT_DS4_FLASH_SNAPSHOT")
            .expect("DS41RT_DS4_FLASH_SNAPSHOT must name the real Flash snapshot");
        let catalog =
            build_catalog_for_snapshot("deepseek-ai/DeepSeek-V4-Flash-0731", Path::new(&snapshot))
                .expect("building Flash checkpoint catalog");
        let layer_id = 0_usize;
        let rows = 3_usize;
        let hidden_values = (0..rows * DS4_FLASH_HIDDEN)
            .map(|index| ((index * 17 % 257) as f32 - 128.0) / 1024.0)
            .collect::<Vec<_>>();
        let hidden_bf16 = f32_values_to_bf16_bytes(&hidden_values);
        let hidden = device_bf16_output_from_bf16_bytes(
            &hidden_bf16,
            rows,
            DS4_FLASH_HIDDEN,
            "real Flash hash sparse-block hidden",
        )
        .expect("uploading real Flash hash sparse-block hidden");
        let token_ids = [0_usize, 17, 129_279];
        let weights = preload_ds4_flash_sparse_layer_resident_weights(&catalog, layer_id)
            .expect("preloading hash sparse-block coordinator weights");
        assert!(weights.hash_routing);

        let mut router_cache = RouterTensorCache::default();
        let router_oracle = score_real_router_routes_bf16_cached_device_input(
            &catalog,
            layer_id,
            &hidden,
            None,
            Some(&token_ids),
            DS4_FLASH_HIDDEN,
            6,
            &mut router_cache,
        )
        .expect("executing real Flash hash-router oracle");
        let route_ids = router_oracle
            .row_routes
            .iter()
            .flatten()
            .map(|route| route.expert_id as u32)
            .collect::<Vec<_>>();
        let route_weights = router_oracle
            .row_routes
            .iter()
            .flatten()
            .map(|route| route.normalized_weight)
            .collect::<Vec<_>>();
        let shared_oracle = ds4_flash_shared_expert_fp8_bf16_device_output(
            &weights.shared_w1_weight_name,
            &weights.shared_w1_scale_name,
            &weights.shared_w3_weight_name,
            &weights.shared_w3_scale_name,
            &weights.shared_w2_weight_name,
            &weights.shared_w2_scale_name,
            &hidden,
            rows,
        )
        .expect("executing real Flash hash-layer shared-expert oracle");
        let shared_oracle_bytes = shared_oracle
            .copy_to_host_bytes()
            .expect("copying real Flash hash-layer shared-expert oracle");

        let library = cuda_native_library().expect("loading hash sparse-block CUDA library");
        let route_bytes = rows * 6 * std::mem::size_of::<u32>();
        let hidden_bytes = rows * DS4_FLASH_HIDDEN * std::mem::size_of::<u16>();
        let shared_workspace_bytes = 8 * 2_048 * std::mem::size_of::<u16>();
        let mut allocations = vec![
            library
                .alloc_device_buffer(rows * std::mem::size_of::<i64>())
                .expect("allocating hash sparse-block token IDs"),
            library
                .alloc_device_buffer(route_bytes)
                .expect("allocating hash sparse-block route indices"),
            library
                .alloc_device_buffer(route_bytes)
                .expect("allocating hash sparse-block route scores"),
            library
                .alloc_device_buffer(route_bytes)
                .expect("allocating hash sparse-block route weights"),
            library
                .alloc_device_buffer(shared_workspace_bytes)
                .expect("allocating hash sparse-block shared gate"),
            library
                .alloc_device_buffer(shared_workspace_bytes)
                .expect("allocating hash sparse-block shared up"),
            library
                .alloc_device_buffer(shared_workspace_bytes)
                .expect("allocating hash sparse-block shared activation"),
            library
                .alloc_device_buffer(hidden_bytes)
                .expect("allocating hash sparse-block shared delta"),
        ];
        let token_bytes = token_ids
            .iter()
            .flat_map(|&token_id| (token_id as i64).to_ne_bytes())
            .collect::<Vec<_>>();
        library
            .copy_h2d(allocations[0], &token_bytes)
            .expect("uploading hash sparse-block token IDs");
        let pre_dispatch = Ds4FlashSparseBlockPreDispatchDeviceBuffers {
            router: Ds4FlashRouterTopKDeviceBuffers {
                indices: allocations[1],
                scores: allocations[2],
                weights: allocations[3],
            },
            shared: Ds4FlashSharedExpertDeviceBuffers {
                gate: allocations[4],
                up: allocations[5],
                activated: allocations[6],
                output: allocations[7],
            },
        };
        let stream = library
            .cuda_stream_create()
            .expect("creating hash sparse-block graph stream");
        let validate = || {
            let mut id_bytes = vec![0_u8; route_bytes];
            let mut weight_bytes = vec![0_u8; route_bytes];
            let mut shared_bytes = vec![0_u8; hidden_bytes];
            library
                .copy_d2h(&mut id_bytes, pre_dispatch.router.indices)
                .expect("copying hash sparse-block route IDs");
            library
                .copy_d2h(&mut weight_bytes, pre_dispatch.router.weights)
                .expect("copying hash sparse-block route weights");
            library
                .copy_d2h(&mut shared_bytes, pre_dispatch.shared.output)
                .expect("copying hash sparse-block shared delta");
            let ids = id_bytes
                .chunks_exact(std::mem::size_of::<u32>())
                .map(|chunk| u32::from_ne_bytes(chunk.try_into().unwrap()))
                .collect::<Vec<_>>();
            let weights = weight_bytes
                .chunks_exact(std::mem::size_of::<f32>())
                .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
                .collect::<Vec<_>>();
            assert_eq!(ids, route_ids);
            assert_eq!(weights, route_weights);
            assert_eq!(shared_bytes, shared_oracle_bytes);
        };

        ds4_flash_sparse_block_pre_dispatch_into(
            &weights,
            &hidden,
            Some(allocations[0]),
            pre_dispatch,
            stream,
        )
        .expect("launching real hash sparse-block pre-dispatch");
        unsafe {
            library
                .cuda_stream_synchronize(stream)
                .expect("synchronizing real hash sparse-block pre-dispatch");
        }
        validate();
        unsafe {
            library
                .cuda_graph_begin_capture(stream)
                .expect("beginning hash sparse-block capture");
        }
        ds4_flash_sparse_block_pre_dispatch_into(
            &weights,
            &hidden,
            Some(allocations[0]),
            pre_dispatch,
            stream,
        )
        .expect("capturing real hash sparse-block pre-dispatch");
        let graph = unsafe {
            library
                .cuda_graph_end_capture_retained(stream)
                .expect("ending hash sparse-block capture")
        };
        assert!(
            graph.kernel_node_count >= 4,
            "hash router plus shared expert captured only {} kernels",
            graph.kernel_node_count
        );
        assert_eq!(graph.memcpy_node_count, 0);
        for _ in 0..3 {
            unsafe {
                library
                    .cuda_graph_launch(graph.graph_exec, stream)
                    .expect("replaying hash sparse-block graph");
            }
        }
        unsafe {
            library
                .cuda_stream_synchronize(stream)
                .expect("synchronizing hash sparse-block graph replays");
        }
        validate();

        unsafe {
            library
                .cuda_graph_destroy(graph.graph)
                .expect("destroying hash sparse-block graph");
            library
                .cuda_graph_exec_destroy(graph.graph_exec)
                .expect("destroying hash sparse-block graph exec");
            library
                .cuda_stream_destroy(stream)
                .expect("destroying hash sparse-block graph stream");
        }
        for buffer in allocations.iter_mut().rev() {
            library
                .free_device_buffer(buffer)
                .expect("freeing hash sparse-block buffer");
        }
    }

    #[test]
    #[ignore = "requires a real Flash checkpoint and CUDA"]
    fn real_flash_decode_sparse_block_preserves_tp4_split_graph_on_gpu0() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(true);
        let snapshot = std::env::var("DS41RT_DS4_FLASH_SNAPSHOT")
            .expect("DS41RT_DS4_FLASH_SNAPSHOT must name the real Flash snapshot");
        let catalog =
            build_catalog_for_snapshot("deepseek-ai/DeepSeek-V4-Flash-0731", Path::new(&snapshot))
                .expect("building Flash checkpoint catalog");
        let layer_id = 3_usize;
        let rows = 1_usize;
        let hidden_values = (0..DS4_FLASH_HIDDEN)
            .map(|index| ((index * 17 % 257) as f32 - 128.0) / 1024.0)
            .collect::<Vec<_>>();
        let hidden_bf16 = f32_values_to_bf16_bytes(&hidden_values);
        let hidden = device_bf16_output_from_bf16_bytes(
            &hidden_bf16,
            rows,
            DS4_FLASH_HIDDEN,
            "real Flash TP4 sparse-block hidden",
        )
        .expect("uploading real Flash TP4 sparse-block hidden");
        let weights = preload_ds4_flash_sparse_layer_resident_weights(&catalog, layer_id)
            .expect("preloading coordinator sparse-block weights");

        let mut router_cache = RouterTensorCache::default();
        let router_oracle = score_real_router_routes_bf16_cached_device_input(
            &catalog,
            layer_id,
            &hidden,
            Some(&hidden_bf16),
            None,
            DS4_FLASH_HIDDEN,
            6,
            &mut router_cache,
        )
        .expect("executing real Flash router oracle");
        let route_ids = router_oracle.row_routes[0]
            .iter()
            .map(|route| route.expert_id as u32)
            .collect::<Vec<_>>();
        let route_weights = router_oracle.row_routes[0]
            .iter()
            .map(|route| route.normalized_weight)
            .collect::<Vec<_>>();
        assert_eq!(route_ids.len(), 6);
        assert_eq!(route_weights.len(), 6);
        let shared_oracle = ds4_flash_shared_expert_fp8_bf16_device_output(
            &weights.shared_w1_weight_name,
            &weights.shared_w1_scale_name,
            &weights.shared_w3_weight_name,
            &weights.shared_w3_scale_name,
            &weights.shared_w2_weight_name,
            &weights.shared_w2_scale_name,
            &hidden,
            rows,
        )
        .expect("executing real Flash shared-expert oracle");
        let shared_oracle_bytes = shared_oracle
            .copy_to_host_bytes()
            .expect("copying real Flash shared-expert oracle");

        let mut rank_caches = (0..DS4_FLASH_TP)
            .map(|_| RouteTensorCache::default())
            .collect::<Vec<_>>();
        for (rank, cache) in rank_caches.iter_mut().enumerate() {
            let shard = ExpertIntermediateShard::new(DS4_FLASH_TP, rank)
                .unwrap_or_else(|error| panic!("constructing Flash TP4 rank {rank}: {error:#}"));
            let preload =
                preload_native_flash_layer_cuda_cache_local_tp4(&catalog, layer_id, shard, cache)
                    .unwrap_or_else(|error| panic!("preloading Flash TP4 rank {rank}: {error:#}"));
            assert_eq!(preload.projection_groups, 256 * 3);
        }
        let rank_oracles = rank_caches
            .iter_mut()
            .enumerate()
            .map(|(rank, cache)| {
                execute_native_flash_route_bf16_cached(
                    rank as u64,
                    layer_id,
                    rows,
                    &hidden_bf16,
                    Some(hidden.buffer()),
                    &route_ids,
                    &route_weights,
                    false,
                    false,
                    None,
                    cache,
                )
                .unwrap_or_else(|error| {
                    panic!("executing existing Flash TP4 rank {rank} oracle: {error:#}")
                })
                .output_bf16
            })
            .collect::<Vec<_>>();

        let library = cuda_native_library().expect("loading sparse-block CUDA library");
        let route_bytes = 6 * std::mem::size_of::<u32>();
        let route_score_workspace_bytes =
            (6 + DS4_FLASH_ROUTED_EXPERTS) * std::mem::size_of::<f32>();
        let shared_workspace_bytes = 8 * 2_048 * std::mem::size_of::<u16>();
        let mut coordinator_allocations = vec![
            library
                .alloc_device_buffer(route_bytes)
                .expect("allocating sparse-block route indices"),
            library
                .alloc_device_buffer(route_score_workspace_bytes)
                .expect("allocating sparse-block route scores"),
            library
                .alloc_device_buffer(route_bytes)
                .expect("allocating sparse-block route weights"),
            library
                .alloc_device_buffer(shared_workspace_bytes)
                .expect("allocating sparse-block shared gate"),
            library
                .alloc_device_buffer(shared_workspace_bytes)
                .expect("allocating sparse-block shared up"),
            library
                .alloc_device_buffer(shared_workspace_bytes)
                .expect("allocating sparse-block shared activation"),
            library
                .alloc_device_buffer(DS4_FLASH_DECODE_M1_HIDDEN_BYTES)
                .expect("allocating sparse-block shared delta"),
            library
                .alloc_device_buffer(DS4_FLASH_HIDDEN * std::mem::size_of::<f32>())
                .expect("allocating sparse-block FP32 reduction"),
            library
                .alloc_device_buffer(DS4_FLASH_DECODE_M1_HIDDEN_BYTES)
                .expect("allocating sparse-block FFN delta"),
        ];
        let pre_dispatch = Ds4FlashSparseBlockPreDispatchDeviceBuffers {
            router: Ds4FlashRouterTopKDeviceBuffers {
                indices: coordinator_allocations[0],
                scores: coordinator_allocations[1],
                weights: coordinator_allocations[2],
            },
            shared: Ds4FlashSharedExpertDeviceBuffers {
                gate: coordinator_allocations[3],
                up: coordinator_allocations[4],
                activated: coordinator_allocations[5],
                output: coordinator_allocations[6],
            },
        };
        let mut rank_allocations = Vec::<Vec<Ds41rtDeviceBuffer>>::new();
        let mut rank_buffers = Vec::new();
        for rank in 0..DS4_FLASH_TP {
            let allocations = vec![
                library
                    .alloc_device_buffer(DS4_FLASH_DECODE_M1_FC1_BYTES)
                    .unwrap_or_else(|error| panic!("allocating rank {rank} FC1: {error:#}")),
                library
                    .alloc_device_buffer(DS4_FLASH_DECODE_M1_ACTIVATED_BYTES)
                    .unwrap_or_else(|error| panic!("allocating rank {rank} activation: {error:#}")),
                library
                    .alloc_device_buffer(DS4_FLASH_DECODE_M1_HIDDEN_BYTES)
                    .unwrap_or_else(|error| panic!("allocating rank {rank} output: {error:#}")),
                library
                    .alloc_device_buffer(std::mem::size_of::<i32>())
                    .unwrap_or_else(|error| {
                        panic!("allocating rank {rank} block expert: {error:#}")
                    }),
                library
                    .alloc_device_buffer(std::mem::size_of::<i32>())
                    .unwrap_or_else(|error| {
                        panic!("allocating rank {rank} route count: {error:#}")
                    }),
                library
                    .alloc_device_buffer(DS4_FLASH_DECODE_M1_FC1_SCRATCH_BYTES)
                    .unwrap_or_else(|error| {
                        panic!("allocating rank {rank} FC1 scratch: {error:#}")
                    }),
                library
                    .alloc_device_buffer(DS4_FLASH_DECODE_M1_FC2_SCRATCH_BYTES)
                    .unwrap_or_else(|error| {
                        panic!("allocating rank {rank} FC2 scratch: {error:#}")
                    }),
                library
                    .alloc_device_buffer(DS4_FLASH_DECODE_M1_LOCK_BYTES)
                    .unwrap_or_else(|error| panic!("allocating rank {rank} locks: {error:#}")),
                library
                    .alloc_device_buffer(DS4_FLASH_DECODE_M1_ROUTED_OUTPUT_BYTES)
                    .unwrap_or_else(|error| {
                        panic!("allocating rank {rank} gate rotation: {error:#}")
                    }),
                library
                    .alloc_device_buffer(DS4_FLASH_DECODE_M1_ROUTED_OUTPUT_BYTES)
                    .unwrap_or_else(|error| {
                        panic!("allocating rank {rank} up rotation: {error:#}")
                    }),
                library
                    .alloc_device_buffer(DS4_FLASH_DECODE_M1_ROUTED_OUTPUT_BYTES)
                    .unwrap_or_else(|error| {
                        panic!("allocating rank {rank} routed output: {error:#}")
                    }),
                library
                    .alloc_device_buffer(DS4_FLASH_DECODE_M1_OUTPUT_F32_BYTES)
                    .unwrap_or_else(|error| {
                        panic!("allocating rank {rank} FP32 output: {error:#}")
                    }),
                library
                    .alloc_device_buffer(DS4_FLASH_PREFILL_EXPERT_COUNT_BYTES)
                    .unwrap_or_else(|error| {
                        panic!("allocating rank {rank} expert counts: {error:#}")
                    }),
                library
                    .alloc_device_buffer(DS4_FLASH_PREFILL_EXPERT_OFFSET_BYTES)
                    .unwrap_or_else(|error| {
                        panic!("allocating rank {rank} expert offsets: {error:#}")
                    }),
            ];
            library
                .copy_h2d(allocations[3], &0_i32.to_ne_bytes())
                .expect("initializing decode block expert scratch");
            library
                .copy_h2d(allocations[4], &6_i32.to_ne_bytes())
                .expect("initializing decode route count scratch");
            rank_buffers.push(Ds4FlashSparkDecodeM1DeviceBuffers {
                input: hidden.buffer(),
                route_indices: pre_dispatch.router.indices,
                route_weights: pre_dispatch.router.weights,
                fc1: allocations[0],
                activated: allocations[1],
                rotation_gate: allocations[8],
                rotation_up: allocations[9],
                routed_output: allocations[10],
                output_f32: allocations[11],
                output: allocations[2],
                block_expert_ids: allocations[3],
                packed_route_count: allocations[4],
                expert_counts: allocations[12],
                expert_offsets: allocations[13],
                fc1_scratch: allocations[5],
                fc2_scratch: allocations[6],
                locks: allocations[7],
            });
            rank_allocations.push(allocations);
        }
        assert!(rank_buffers.iter().all(|buffers| {
            buffers.input.ptr == hidden.buffer().ptr
                && buffers.route_indices.ptr == pre_dispatch.router.indices.ptr
                && buffers.route_weights.ptr == pre_dispatch.router.weights.ptr
        }));
        let reduction = Ds4FlashTp4ReductionDeviceBuffers {
            shared_delta: pre_dispatch.shared.output,
            rank_partials: [
                rank_buffers[0].output,
                rank_buffers[1].output,
                rank_buffers[2].output,
                rank_buffers[3].output,
            ],
            reduction_f32: coordinator_allocations[7],
            ffn_delta: coordinator_allocations[8],
        };
        let stream = library
            .cuda_stream_create()
            .expect("creating sparse-block graph stream");

        ds4_flash_sparse_block_pre_dispatch_into(&weights, &hidden, None, pre_dispatch, stream)
            .expect("launching real sparse-block segment one");
        unsafe {
            library
                .cuda_stream_synchronize(stream)
                .expect("synchronizing real sparse-block segment one");
        }
        let validate_pre_dispatch = || {
            let mut id_bytes = vec![0_u8; route_bytes];
            let mut weight_bytes = vec![0_u8; route_bytes];
            let mut shared_bytes = vec![0_u8; DS4_FLASH_DECODE_M1_HIDDEN_BYTES];
            library
                .copy_d2h(&mut id_bytes, pre_dispatch.router.indices)
                .expect("copying global route IDs");
            library
                .copy_d2h(&mut weight_bytes, pre_dispatch.router.weights)
                .expect("copying global route weights");
            library
                .copy_d2h(&mut shared_bytes, pre_dispatch.shared.output)
                .expect("copying shared delta");
            let ids = id_bytes
                .chunks_exact(std::mem::size_of::<u32>())
                .map(|chunk| u32::from_ne_bytes(chunk.try_into().unwrap()))
                .collect::<Vec<_>>();
            let weights = weight_bytes
                .chunks_exact(std::mem::size_of::<f32>())
                .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
                .collect::<Vec<_>>();
            assert_eq!(ids, route_ids);
            assert_eq!(weights, route_weights);
            assert_eq!(shared_bytes, shared_oracle_bytes);
        };
        validate_pre_dispatch();

        unsafe {
            library
                .cuda_graph_begin_capture(stream)
                .expect("beginning sparse-block segment-one capture");
        }
        ds4_flash_sparse_block_pre_dispatch_into(&weights, &hidden, None, pre_dispatch, stream)
            .expect("capturing sparse-block segment one");
        let pre_dispatch_graph = unsafe {
            library
                .cuda_graph_end_capture_retained(stream)
                .expect("ending sparse-block segment-one capture")
        };
        assert!(pre_dispatch_graph.kernel_node_count >= 6);
        assert_eq!(pre_dispatch_graph.memcpy_node_count, 0);
        for _ in 0..3 {
            unsafe {
                library
                    .cuda_graph_launch(pre_dispatch_graph.graph_exec, stream)
                    .expect("replaying sparse-block segment one");
            }
        }
        unsafe {
            library
                .cuda_stream_synchronize(stream)
                .expect("synchronizing sparse-block segment-one replays");
        }
        validate_pre_dispatch();

        let assert_rank_close = |rank: usize, actual_buffer: Ds41rtDeviceBuffer| {
            let mut actual = vec![0_u8; DS4_FLASH_DECODE_M1_HIDDEN_BYTES];
            library
                .copy_d2h(&mut actual, actual_buffer)
                .unwrap_or_else(|error| panic!("copying rank {rank} partial: {error:#}"));
            assert_bf16_vectors_close(&format!("rank {rank}"), &actual, &rank_oracles[rank]);
        };
        let mut rank_graphs = Vec::new();
        for (rank, (buffers, cache)) in rank_buffers
            .iter()
            .copied()
            .zip(rank_caches.iter_mut())
            .enumerate()
        {
            ds4_flash_spark_decode_m1_preloaded_device_input_into(layer_id, buffers, stream, cache)
                .unwrap_or_else(|error| panic!("launching raw TP4 rank {rank}: {error:#}"));
            unsafe {
                library
                    .cuda_stream_synchronize(stream)
                    .expect("synchronizing raw TP4 rank");
                library
                    .cuda_graph_begin_capture(stream)
                    .expect("beginning TP4 rank capture");
            }
            ds4_flash_spark_decode_m1_preloaded_device_input_into(layer_id, buffers, stream, cache)
                .unwrap_or_else(|error| panic!("capturing TP4 rank {rank}: {error:#}"));
            let graph = unsafe {
                library
                    .cuda_graph_end_capture_retained(stream)
                    .expect("ending TP4 rank capture")
            };
            assert!(graph.kernel_node_count >= 1);
            assert!(graph.memset_node_count >= 1);
            assert_eq!(graph.memcpy_node_count, 0);
            for _ in 0..3 {
                unsafe {
                    library
                        .cuda_graph_launch(graph.graph_exec, stream)
                        .expect("replaying TP4 rank graph");
                }
            }
            unsafe {
                library
                    .cuda_stream_synchronize(stream)
                    .expect("synchronizing TP4 rank graph replays");
            }
            assert_rank_close(rank, buffers.output);
            rank_graphs.push(graph);
        }

        ds4_flash_tp4_reduce_bf16_device_inputs_into(rows, reduction, stream)
            .expect("launching sparse-block post-dispatch reduction");
        unsafe {
            library
                .cuda_stream_synchronize(stream)
                .expect("synchronizing sparse-block post-dispatch reduction");
            library
                .cuda_graph_begin_capture(stream)
                .expect("beginning sparse-block post-dispatch capture");
        }
        ds4_flash_tp4_reduce_bf16_device_inputs_into(rows, reduction, stream)
            .expect("capturing sparse-block post-dispatch reduction");
        let post_dispatch_graph = unsafe {
            library
                .cuda_graph_end_capture_retained(stream)
                .expect("ending sparse-block post-dispatch capture")
        };
        assert!(post_dispatch_graph.kernel_node_count >= 6);
        assert!(post_dispatch_graph.memset_node_count >= 1);
        assert_eq!(post_dispatch_graph.memcpy_node_count, 0);
        for _ in 0..3 {
            unsafe {
                library
                    .cuda_graph_launch(post_dispatch_graph.graph_exec, stream)
                    .expect("replaying sparse-block post-dispatch graph");
            }
        }
        unsafe {
            library
                .cuda_stream_synchronize(stream)
                .expect("synchronizing sparse-block post-dispatch replays");
        }
        let mut expected = bf16_values_to_f32(&shared_oracle_bytes);
        for buffers in &rank_buffers {
            let mut partial = vec![0_u8; DS4_FLASH_DECODE_M1_HIDDEN_BYTES];
            library
                .copy_d2h(&mut partial, buffers.output)
                .expect("copying TP4 partial for reduction oracle");
            for (dst, src) in expected.iter_mut().zip(bf16_values_to_f32(&partial)) {
                *dst += src;
            }
        }
        let mut reduction_bytes = vec![0_u8; DS4_FLASH_HIDDEN * std::mem::size_of::<f32>()];
        let mut delta_bytes = vec![0_u8; DS4_FLASH_DECODE_M1_HIDDEN_BYTES];
        library
            .copy_d2h(&mut reduction_bytes, reduction.reduction_f32)
            .expect("copying composed TP4 FP32 reduction");
        library
            .copy_d2h(&mut delta_bytes, reduction.ffn_delta)
            .expect("copying composed TP4 BF16 delta");
        let actual = reduction_bytes
            .chunks_exact(std::mem::size_of::<f32>())
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
        assert_eq!(delta_bytes, f32_values_to_bf16_bytes(&expected));

        unsafe {
            library
                .cuda_graph_destroy(post_dispatch_graph.graph)
                .expect("destroying sparse-block post-dispatch graph");
            library
                .cuda_graph_exec_destroy(post_dispatch_graph.graph_exec)
                .expect("destroying sparse-block post-dispatch graph exec");
            for graph in rank_graphs.into_iter().rev() {
                library
                    .cuda_graph_destroy(graph.graph)
                    .expect("destroying TP4 rank graph");
                library
                    .cuda_graph_exec_destroy(graph.graph_exec)
                    .expect("destroying TP4 rank graph exec");
            }
            library
                .cuda_graph_destroy(pre_dispatch_graph.graph)
                .expect("destroying sparse-block pre-dispatch graph");
            library
                .cuda_graph_exec_destroy(pre_dispatch_graph.graph_exec)
                .expect("destroying sparse-block pre-dispatch graph exec");
            library
                .cuda_stream_destroy(stream)
                .expect("destroying sparse-block graph stream");
        }
        for allocations in rank_allocations.iter_mut().rev() {
            for buffer in allocations.iter_mut().rev() {
                library
                    .free_device_buffer(buffer)
                    .expect("freeing TP4 rank arena buffer");
            }
        }
        for buffer in coordinator_allocations.iter_mut().rev() {
            library
                .free_device_buffer(buffer)
                .expect("freeing coordinator sparse-block buffer");
        }
    }

    #[test]
    #[ignore = "requires a real Flash checkpoint and CUDA"]
    fn real_flash_prefill_sparse_block_preserves_tp4_split_graph_on_gpu0() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(true);
        let snapshot = std::env::var("DS41RT_DS4_FLASH_SNAPSHOT")
            .expect("DS41RT_DS4_FLASH_SNAPSHOT must name the real Flash snapshot");
        let catalog =
            build_catalog_for_snapshot("deepseek-ai/DeepSeek-V4-Flash-0731", Path::new(&snapshot))
                .expect("building Flash checkpoint catalog");
        let layer_id = 3_usize;
        let rows = 7_usize;
        let active_hidden_bytes = rows * DS4_FLASH_HIDDEN * std::mem::size_of::<u16>();
        let active_route_bytes = rows * 6 * std::mem::size_of::<u32>();
        let hidden_values = (0..rows * DS4_FLASH_HIDDEN)
            .map(|index| ((index * 29 % 521) as f32 - 260.0) / 2048.0)
            .collect::<Vec<_>>();
        let hidden_bf16 = f32_values_to_bf16_bytes(&hidden_values);
        let library = cuda_native_library().expect("loading sparse-block CUDA library");
        let hidden = DeviceBf16Output {
            buffer: OwnedCoordinatorDeviceBuffer::new(
                library,
                DS4_FLASH_PREFILL_HIDDEN_BYTES,
                "real Flash TP4 sparse-block prefill hidden",
            )
            .expect("allocating max-capacity sparse-block prefill hidden"),
            bytes: active_hidden_bytes,
            rows,
            values_per_row: DS4_FLASH_HIDDEN,
            backend: "real-flash-tp4-sparse-block-prefill-test",
        };
        library
            .copy_h2d(hidden.buffer(), &hidden_bf16)
            .expect("uploading real Flash TP4 sparse-block prefill hidden");
        let weights = preload_ds4_flash_sparse_layer_resident_weights(&catalog, layer_id)
            .expect("preloading coordinator sparse-block weights");

        let mut router_cache = RouterTensorCache::default();
        let router_oracle = score_real_router_routes_bf16_cached_device_input(
            &catalog,
            layer_id,
            &hidden,
            Some(&hidden_bf16),
            None,
            DS4_FLASH_HIDDEN,
            6,
            &mut router_cache,
        )
        .expect("executing real Flash prefill router oracle");
        let route_ids = router_oracle
            .row_routes
            .iter()
            .flatten()
            .map(|route| route.expert_id as u32)
            .collect::<Vec<_>>();
        let route_weights = router_oracle
            .row_routes
            .iter()
            .flatten()
            .map(|route| route.normalized_weight)
            .collect::<Vec<_>>();
        assert_eq!(route_ids.len(), rows * 6);
        assert_eq!(route_weights.len(), rows * 6);
        let shared_oracle = ds4_flash_shared_expert_fp8_bf16_device_output(
            &weights.shared_w1_weight_name,
            &weights.shared_w1_scale_name,
            &weights.shared_w3_weight_name,
            &weights.shared_w3_scale_name,
            &weights.shared_w2_weight_name,
            &weights.shared_w2_scale_name,
            &hidden,
            rows,
        )
        .expect("executing real Flash prefill shared-expert oracle");
        let shared_oracle_bytes = shared_oracle
            .copy_to_host_bytes()
            .expect("copying real Flash prefill shared-expert oracle");

        let mut rank_caches = (0..DS4_FLASH_TP)
            .map(|_| RouteTensorCache::default())
            .collect::<Vec<_>>();
        for (rank, cache) in rank_caches.iter_mut().enumerate() {
            let shard = ExpertIntermediateShard::new(DS4_FLASH_TP, rank)
                .unwrap_or_else(|error| panic!("constructing Flash TP4 rank {rank}: {error:#}"));
            let preload =
                preload_native_flash_layer_cuda_cache_local_tp4(&catalog, layer_id, shard, cache)
                    .unwrap_or_else(|error| panic!("preloading Flash TP4 rank {rank}: {error:#}"));
            assert_eq!(preload.projection_groups, 256 * 3);
        }
        let rank_oracles = rank_caches
            .iter_mut()
            .enumerate()
            .map(|(rank, cache)| {
                execute_native_flash_route_bf16_cached(
                    rank as u64,
                    layer_id,
                    rows,
                    &hidden_bf16,
                    Some(hidden.buffer()),
                    &route_ids,
                    &route_weights,
                    false,
                    false,
                    None,
                    cache,
                )
                .unwrap_or_else(|error| {
                    panic!("executing existing Flash prefill TP4 rank {rank} oracle: {error:#}")
                })
                .output_bf16
            })
            .collect::<Vec<_>>();

        let shared_workspace_bytes = 8 * 2_048 * std::mem::size_of::<u16>();
        let route_score_workspace_bytes = DS4_FLASH_PREFILL_TOPK_WEIGHT_BYTES / DS4_FLASH_TOP_K
            * (DS4_FLASH_TOP_K + DS4_FLASH_ROUTED_EXPERTS);
        let mut coordinator_allocations = vec![
            library
                .alloc_device_buffer(DS4_FLASH_PREFILL_TOPK_ID_BYTES)
                .expect("allocating prefill route indices"),
            library
                .alloc_device_buffer(route_score_workspace_bytes)
                .expect("allocating prefill route scores"),
            library
                .alloc_device_buffer(DS4_FLASH_PREFILL_TOPK_WEIGHT_BYTES)
                .expect("allocating prefill route weights"),
            library
                .alloc_device_buffer(shared_workspace_bytes)
                .expect("allocating prefill shared gate"),
            library
                .alloc_device_buffer(shared_workspace_bytes)
                .expect("allocating prefill shared up"),
            library
                .alloc_device_buffer(shared_workspace_bytes)
                .expect("allocating prefill shared activation"),
            library
                .alloc_device_buffer(DS4_FLASH_PREFILL_HIDDEN_BYTES)
                .expect("allocating prefill shared delta"),
            library
                .alloc_device_buffer(DS4_FLASH_PREFILL_HIDDEN_BYTES * 2)
                .expect("allocating prefill FP32 reduction"),
            library
                .alloc_device_buffer(DS4_FLASH_PREFILL_HIDDEN_BYTES)
                .expect("allocating prefill FFN delta"),
        ];
        let pre_dispatch = Ds4FlashSparseBlockPreDispatchDeviceBuffers {
            router: Ds4FlashRouterTopKDeviceBuffers {
                indices: coordinator_allocations[0],
                scores: coordinator_allocations[1],
                weights: coordinator_allocations[2],
            },
            shared: Ds4FlashSharedExpertDeviceBuffers {
                gate: coordinator_allocations[3],
                up: coordinator_allocations[4],
                activated: coordinator_allocations[5],
                output: coordinator_allocations[6],
            },
        };
        let mut rank_allocations = Vec::<Vec<Ds41rtDeviceBuffer>>::new();
        let mut rank_buffers = Vec::new();
        for rank in 0..DS4_FLASH_TP {
            let allocate = |bytes, label| {
                library
                    .alloc_device_buffer(bytes)
                    .unwrap_or_else(|error| panic!("allocating rank {rank} {label}: {error:#}"))
            };
            let allocations = vec![
                allocate(DS4_FLASH_PREFILL_PACKED_ROUTE_BYTES, "packed routes"),
                allocate(DS4_FLASH_PREFILL_BLOCK_EXPERT_BYTES, "block experts"),
                allocate(std::mem::size_of::<i32>(), "packed route count"),
                allocate(DS4_FLASH_PREFILL_EXPERT_COUNT_BYTES, "expert counts"),
                allocate(DS4_FLASH_PREFILL_EXPERT_OFFSET_BYTES, "expert offsets"),
                allocate(DS4_FLASH_PREFILL_FC1_BYTES, "FC1"),
                allocate(DS4_FLASH_PREFILL_ACTIVATED_BYTES, "activation"),
                allocate(DS4_FLASH_PREFILL_ROUTED_OUTPUT_BYTES, "routed output"),
                allocate(DS4_FLASH_PREFILL_HIDDEN_BYTES, "rank partial"),
                allocate(DS4_FLASH_PREFILL_COMPUTE_SCRATCH_BYTES, "FC1 scratch"),
                allocate(DS4_FLASH_PREFILL_COMPUTE_SCRATCH_BYTES, "FC2 scratch"),
                allocate(DS4_FLASH_DECODE_M1_LOCK_BYTES, "locks"),
                allocate(DS4_FLASH_PREFILL_ROUTED_OUTPUT_BYTES, "gate rotation"),
                allocate(DS4_FLASH_PREFILL_ROUTED_OUTPUT_BYTES, "up rotation"),
                allocate(DS4_FLASH_PREFILL_OUTPUT_F32_BYTES, "FP32 output"),
            ];
            rank_buffers.push(Ds4FlashSparkPrefillDeviceBuffers {
                input: hidden.buffer(),
                topk_ids: pre_dispatch.router.indices,
                topk_weights: pre_dispatch.router.weights,
                packed_route_indices: allocations[0],
                block_expert_ids: allocations[1],
                packed_route_count: allocations[2],
                expert_counts: allocations[3],
                expert_offsets: allocations[4],
                fc1: allocations[5],
                activated: allocations[6],
                rotation_gate: allocations[12],
                rotation_up: allocations[13],
                routed_output: allocations[7],
                output_f32: allocations[14],
                output: allocations[8],
                fc1_scratch: allocations[9],
                fc2_scratch: allocations[10],
                locks: allocations[11],
            });
            rank_allocations.push(allocations);
        }
        assert!(rank_buffers.iter().all(|buffers| {
            buffers.input.ptr == hidden.buffer().ptr
                && buffers.topk_ids.ptr == pre_dispatch.router.indices.ptr
                && buffers.topk_weights.ptr == pre_dispatch.router.weights.ptr
        }));
        let reduction = Ds4FlashTp4ReductionDeviceBuffers {
            shared_delta: pre_dispatch.shared.output,
            rank_partials: [
                rank_buffers[0].output,
                rank_buffers[1].output,
                rank_buffers[2].output,
                rank_buffers[3].output,
            ],
            reduction_f32: coordinator_allocations[7],
            ffn_delta: coordinator_allocations[8],
        };
        let stream = library
            .cuda_stream_create()
            .expect("creating prefill sparse-block graph stream");

        ds4_flash_sparse_block_pre_dispatch_into(&weights, &hidden, None, pre_dispatch, stream)
            .expect("launching real prefill sparse-block segment one");
        unsafe {
            library
                .cuda_stream_synchronize(stream)
                .expect("synchronizing real prefill sparse-block segment one");
        }
        let validate_pre_dispatch = || {
            let mut id_bytes = vec![0_u8; active_route_bytes];
            let mut weight_bytes = vec![0_u8; active_route_bytes];
            let mut shared_bytes = vec![0_u8; active_hidden_bytes];
            library
                .copy_d2h(&mut id_bytes, pre_dispatch.router.indices)
                .expect("copying prefill global route IDs");
            library
                .copy_d2h(&mut weight_bytes, pre_dispatch.router.weights)
                .expect("copying prefill global route weights");
            library
                .copy_d2h(&mut shared_bytes, pre_dispatch.shared.output)
                .expect("copying prefill shared delta");
            let ids = id_bytes
                .chunks_exact(std::mem::size_of::<u32>())
                .map(|chunk| u32::from_ne_bytes(chunk.try_into().unwrap()))
                .collect::<Vec<_>>();
            let weights = weight_bytes
                .chunks_exact(std::mem::size_of::<f32>())
                .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
                .collect::<Vec<_>>();
            assert_eq!(ids, route_ids);
            assert_eq!(weights, route_weights);
            assert_eq!(shared_bytes, shared_oracle_bytes);
        };
        validate_pre_dispatch();

        unsafe {
            library
                .cuda_graph_begin_capture(stream)
                .expect("beginning prefill sparse-block segment-one capture");
        }
        ds4_flash_sparse_block_pre_dispatch_into(&weights, &hidden, None, pre_dispatch, stream)
            .expect("capturing prefill sparse-block segment one");
        let pre_dispatch_graph = unsafe {
            library
                .cuda_graph_end_capture_retained(stream)
                .expect("ending prefill sparse-block segment-one capture")
        };
        assert!(pre_dispatch_graph.kernel_node_count >= 6);
        assert_eq!(pre_dispatch_graph.memcpy_node_count, 0);
        for _ in 0..3 {
            unsafe {
                library
                    .cuda_graph_launch(pre_dispatch_graph.graph_exec, stream)
                    .expect("replaying prefill sparse-block segment one");
            }
        }
        unsafe {
            library
                .cuda_stream_synchronize(stream)
                .expect("synchronizing prefill sparse-block segment-one replays");
        }
        validate_pre_dispatch();

        let assert_rank_close = |rank: usize, actual_buffer: Ds41rtDeviceBuffer| {
            let mut actual = vec![0_u8; active_hidden_bytes];
            library
                .copy_d2h(&mut actual, actual_buffer)
                .unwrap_or_else(|error| panic!("copying prefill rank {rank} partial: {error:#}"));
            assert_bf16_vectors_close(
                &format!("prefill rank {rank}"),
                &actual,
                &rank_oracles[rank],
            );
        };
        let mut rank_graphs = Vec::new();
        for (rank, (buffers, cache)) in rank_buffers
            .iter()
            .copied()
            .zip(rank_caches.iter_mut())
            .enumerate()
        {
            ds4_flash_spark_prefill_topk6_preloaded_device_input_into(
                layer_id, rows, buffers, stream, cache,
            )
            .unwrap_or_else(|error| panic!("launching raw prefill TP4 rank {rank}: {error:#}"));
            unsafe {
                library
                    .cuda_stream_synchronize(stream)
                    .expect("synchronizing raw prefill TP4 rank");
                library
                    .cuda_graph_begin_capture(stream)
                    .expect("beginning prefill TP4 rank capture");
            }
            ds4_flash_spark_prefill_topk6_preloaded_device_input_into(
                layer_id, rows, buffers, stream, cache,
            )
            .unwrap_or_else(|error| panic!("capturing prefill TP4 rank {rank}: {error:#}"));
            let graph = unsafe {
                library
                    .cuda_graph_end_capture_retained(stream)
                    .expect("ending prefill TP4 rank capture")
            };
            assert!(graph.kernel_node_count >= 5);
            assert!(graph.memset_node_count >= 2);
            assert_eq!(graph.memcpy_node_count, 0);
            for _ in 0..3 {
                unsafe {
                    library
                        .cuda_graph_launch(graph.graph_exec, stream)
                        .expect("replaying prefill TP4 rank graph");
                }
            }
            unsafe {
                library
                    .cuda_stream_synchronize(stream)
                    .expect("synchronizing prefill TP4 rank graph replays");
            }
            assert_rank_close(rank, buffers.output);
            rank_graphs.push(graph);
        }

        ds4_flash_tp4_reduce_bf16_device_inputs_into(rows, reduction, stream)
            .expect("launching prefill sparse-block post-dispatch reduction");
        unsafe {
            library
                .cuda_stream_synchronize(stream)
                .expect("synchronizing prefill sparse-block post-dispatch reduction");
            library
                .cuda_graph_begin_capture(stream)
                .expect("beginning prefill sparse-block post-dispatch capture");
        }
        ds4_flash_tp4_reduce_bf16_device_inputs_into(rows, reduction, stream)
            .expect("capturing prefill sparse-block post-dispatch reduction");
        let post_dispatch_graph = unsafe {
            library
                .cuda_graph_end_capture_retained(stream)
                .expect("ending prefill sparse-block post-dispatch capture")
        };
        assert!(post_dispatch_graph.kernel_node_count >= 6);
        assert!(post_dispatch_graph.memset_node_count >= 1);
        assert_eq!(post_dispatch_graph.memcpy_node_count, 0);
        for _ in 0..3 {
            unsafe {
                library
                    .cuda_graph_launch(post_dispatch_graph.graph_exec, stream)
                    .expect("replaying prefill sparse-block post-dispatch graph");
            }
        }
        unsafe {
            library
                .cuda_stream_synchronize(stream)
                .expect("synchronizing prefill sparse-block post-dispatch replays");
        }
        let mut expected = bf16_values_to_f32(&shared_oracle_bytes);
        for buffers in &rank_buffers {
            let mut partial = vec![0_u8; active_hidden_bytes];
            library
                .copy_d2h(&mut partial, buffers.output)
                .expect("copying prefill TP4 partial for reduction oracle");
            for (dst, src) in expected.iter_mut().zip(bf16_values_to_f32(&partial)) {
                *dst += src;
            }
        }
        let mut reduction_bytes = vec![0_u8; active_hidden_bytes * 2];
        let mut delta_bytes = vec![0_u8; active_hidden_bytes];
        library
            .copy_d2h(&mut reduction_bytes, reduction.reduction_f32)
            .expect("copying composed prefill TP4 FP32 reduction");
        library
            .copy_d2h(&mut delta_bytes, reduction.ffn_delta)
            .expect("copying composed prefill TP4 BF16 delta");
        let actual = reduction_bytes
            .chunks_exact(std::mem::size_of::<f32>())
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
        assert_eq!(delta_bytes, f32_values_to_bf16_bytes(&expected));

        unsafe {
            library
                .cuda_graph_destroy(post_dispatch_graph.graph)
                .expect("destroying prefill sparse-block post-dispatch graph");
            library
                .cuda_graph_exec_destroy(post_dispatch_graph.graph_exec)
                .expect("destroying prefill sparse-block post-dispatch graph exec");
            for graph in rank_graphs.into_iter().rev() {
                library
                    .cuda_graph_destroy(graph.graph)
                    .expect("destroying prefill TP4 rank graph");
                library
                    .cuda_graph_exec_destroy(graph.graph_exec)
                    .expect("destroying prefill TP4 rank graph exec");
            }
            library
                .cuda_graph_destroy(pre_dispatch_graph.graph)
                .expect("destroying prefill sparse-block pre-dispatch graph");
            library
                .cuda_graph_exec_destroy(pre_dispatch_graph.graph_exec)
                .expect("destroying prefill sparse-block pre-dispatch graph exec");
            library
                .cuda_stream_destroy(stream)
                .expect("destroying prefill sparse-block graph stream");
        }
        for allocations in rank_allocations.iter_mut().rev() {
            for buffer in allocations.iter_mut().rev() {
                library
                    .free_device_buffer(buffer)
                    .expect("freeing prefill TP4 rank arena buffer");
            }
        }
        for buffer in coordinator_allocations.iter_mut().rev() {
            library
                .free_device_buffer(buffer)
                .expect("freeing coordinator prefill sparse-block buffer");
        }
    }
}
