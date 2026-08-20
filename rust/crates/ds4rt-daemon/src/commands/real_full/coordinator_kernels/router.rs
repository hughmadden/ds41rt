use super::*;
use crate::python_graph_capture::coordinator_python_capture_enabled;
use anyhow::{Context, Result};
use ds4rt_core::{
    CoordinatorGraphInstancePlan, CoordinatorGraphKey, CoordinatorGraphShape, LayerId,
    LayerWaveMode, ModelVariant, COORDINATOR_GRAPH_INSTANCE_COUNT, GLM52_FIRST_K_DENSE_REPLACE,
    GLM52_HIDDEN_BF16_BYTES, GLM52_HIDDEN_SIZE, GLM52_NUM_HIDDEN_LAYERS,
    GLM52_ROUTED_SCALING_FACTOR, GLM52_TOP_K,
};
use ds4rt_ffi::{
    Ds4rtCudaGraphCaptureInfo, Ds4rtDeviceBuffer, Ds4rtHostBuffer, NativeLibrary,
    DS4RT_CUDA_ROUTER_TOPK_MAX_K, DS4RT_CUDA_SAMPLE_TOPK_MAX_K,
};
use std::collections::HashMap;
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::slice;
use std::sync::Mutex;

pub(in crate::commands::real_full) const CPU_REFERENCE_ROUTER_TOPK_BACKEND: &str =
    "cpu-reference-router-topk";
#[allow(dead_code)]
pub(in crate::commands::real_full) const CUDA_REFERENCE_ROUTER_TOPK_BACKEND: &str =
    "cuda-reference-router-topk-f32";
pub(in crate::commands::real_full) const CPU_REFERENCE_ROUTER_TOPK_BF16_BACKEND: &str =
    "cpu-reference-router-topk-bf16";
pub(in crate::commands::real_full) const CUDA_REFERENCE_ROUTER_TOPK_BF16_BACKEND: &str =
    "cuda-reference-router-topk-bf16";
pub(in crate::commands::real_full) const CUDA_REFERENCE_ROUTER_TOPK_BF16_RESIDENT_WEIGHT_BACKEND:
    &str = "cuda-reference-router-topk-bf16-resident-weight";
pub(in crate::commands::real_full) const CUDA_REFERENCE_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND:
    &str = "cuda-reference-router-topk-bf16-preloaded-resident-weight";
pub(in crate::commands::real_full) const CUDA_REFERENCE_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_BIAS_BACKEND:
    &str = "cuda-reference-router-topk-bf16-preloaded-resident-weight-bias";
pub(in crate::commands::real_full) const CUDA_REFERENCE_ROUTER_TOPK_BF16_RESIDENT_WEIGHT_DEVICE_INPUT_BACKEND:
    &str = "cuda-reference-router-topk-bf16-resident-weight-device-input";
pub(in crate::commands::real_full) const CUDA_REFERENCE_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_DEVICE_INPUT_BACKEND:
    &str = "cuda-reference-router-topk-bf16-preloaded-resident-weight-device-input";
pub(in crate::commands::real_full) const CUDA_REFERENCE_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_BIAS_DEVICE_INPUT_BACKEND:
    &str = "cuda-reference-router-topk-bf16-preloaded-resident-weight-bias-device-input";
pub(in crate::commands::real_full) const TRITON_ROUTER_TOPK_BF16_RESIDENT_WEIGHT_BACKEND: &str =
    "triton-router-topk-bf16-resident-weight";
pub(in crate::commands::real_full) const TRITON_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND:
    &str = "triton-router-topk-bf16-preloaded-resident-weight";
pub(in crate::commands::real_full) const TRITON_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_BIAS_BACKEND:
    &str = "triton-router-topk-bf16-preloaded-resident-weight-bias";
pub(in crate::commands::real_full) const TRITON_ROUTER_TOPK_BF16_RESIDENT_WEIGHT_DEVICE_INPUT_BACKEND:
    &str = "triton-router-topk-bf16-resident-weight-device-input";
pub(in crate::commands::real_full) const TRITON_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_DEVICE_INPUT_BACKEND:
    &str = "triton-router-topk-bf16-preloaded-resident-weight-device-input";
pub(in crate::commands::real_full) const TRITON_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_BIAS_DEVICE_INPUT_BACKEND:
    &str = "triton-router-topk-bf16-preloaded-resident-weight-bias-device-input";
pub(in crate::commands::real_full) const CUDA_DS4_FLASH_ROUTER_TOPK_BF16_PRELOADED_DEVICE_INPUT_BACKEND:
    &str = "cuda-ds4-flash-router-topk-bf16-preloaded-device-input";
pub(in crate::commands::real_full) const CUDA_DS4_PRO_ROUTER_TOPK_BF16_PRELOADED_DEVICE_INPUT_BACKEND:
    &str = "cuda-ds4-pro-router-topk-bf16-preloaded-device-input";

const DS4_FLASH_ROUTER_HIDDEN: usize = 4_096;
const DS4_FLASH_ROUTER_EXPERTS: usize = 256;
const DS4_FLASH_ROUTER_TOP_K: usize = 6;
const DS4_FLASH_ROUTER_VOCAB: usize = 129_280;
const DS4_FLASH_ROUTER_SHORTLIST_ENV: &str = "DS4RT_DS4_FLASH_ROUTER_SHORTLIST";
const DS4_PRO_ROUTER_HIDDEN: usize = 7_168;
const DS4_PRO_ROUTER_EXPERTS: usize = 384;
const DS4_PRO_ROUTER_TOP_K: usize = 6;
const DS4_PRO_ROUTER_VOCAB: usize = 129_280;

/// Caller-owned outputs for the native DeepSeek V4 router. For learned routing,
/// `scores` contains the top-k result followed by a rows-by-experts deterministic
/// score workspace. Only indices and normalized weights cross the TP barrier.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) struct Ds4FlashRouterTopKDeviceBuffers {
    pub(in crate::commands::real_full) indices: Ds4rtDeviceBuffer,
    pub(in crate::commands::real_full) scores: Ds4rtDeviceBuffer,
    pub(in crate::commands::real_full) weights: Ds4rtDeviceBuffer,
}

fn ds4_flash_router_shortlist_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var(DS4_FLASH_ROUTER_SHORTLIST_ENV)
            .map(|value| !matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "NO"))
            .unwrap_or(true)
    })
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) fn ds4_flash_router_topk_bf16_preloaded_device_input_into(
    router_weight_name: &str,
    router_aux_name: &str,
    hidden: &DeviceBf16Output,
    token_ids: Option<Ds4rtDeviceBuffer>,
    hash_routing: bool,
    outputs: Ds4FlashRouterTopKDeviceBuffers,
    cuda_stream: *mut c_void,
) -> Result<()> {
    let (library, router_weight, router_aux) = preloaded_ds4_flash_router_device_weights(
        router_weight_name,
        router_aux_name,
        hash_routing,
    )?;
    launch_ds4_flash_router_topk_bf16_device_input_into(
        library,
        router_weight,
        router_aux,
        hidden,
        token_ids,
        hash_routing,
        outputs,
        cuda_stream,
    )
}

/// Launch the Flash router from a stable caller-owned BF16 view whose producer
/// is already ordered on `cuda_stream`. This is the arena-backed dSpark path;
/// it intentionally carries no owned-buffer or expert-placement semantics.
#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn ds4_flash_router_topk_bf16_preloaded_fixed_device_input_into(
    router_weight_name: &str,
    router_aux_name: &str,
    hidden: Ds4rtDeviceBuffer,
    rows: usize,
    token_ids: Option<Ds4rtDeviceBuffer>,
    hash_routing: bool,
    outputs: Ds4FlashRouterTopKDeviceBuffers,
    cuda_stream: *mut c_void,
) -> Result<()> {
    anyhow::ensure!(rows > 0, "Flash fixed router input rows must be positive");
    let (library, router_weight, router_aux) = preloaded_ds4_flash_router_device_weights(
        router_weight_name,
        router_aux_name,
        hash_routing,
    )?;
    validate_ds4_flash_router_topk_device_handoff(hidden, token_ids, hash_routing, outputs, rows)?;
    launch_ds4_flash_router_topk_bf16_into(
        library,
        hidden,
        router_weight,
        router_aux,
        token_ids,
        hash_routing,
        outputs,
        rows,
        cuda_stream,
    )
    .context("launching native Flash router from fixed arena input")
}

/// Pro counterpart of the fixed-input dSpark router. It preserves the same
/// one-global-route-image contract while widening only hidden/expert geometry.
#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn ds4_pro_router_topk_bf16_preloaded_fixed_device_input_into(
    router_weight_name: &str,
    router_aux_name: &str,
    hidden: Ds4rtDeviceBuffer,
    rows: usize,
    token_ids: Option<Ds4rtDeviceBuffer>,
    hash_routing: bool,
    outputs: Ds4FlashRouterTopKDeviceBuffers,
    cuda_stream: *mut c_void,
) -> Result<()> {
    anyhow::ensure!(rows > 0, "Pro fixed router input rows must be positive");
    let (library, router_weight, router_aux) =
        preloaded_ds4_pro_router_device_weights(router_weight_name, router_aux_name, hash_routing)?;
    validate_ds4_pro_router_topk_device_handoff(hidden, token_ids, hash_routing, outputs, rows)?;
    unsafe {
        library
            .cuda_ds4_pro_router_topk_bf16_async(
                hidden,
                router_weight,
                (!hash_routing).then_some(router_aux),
                hash_routing.then_some(router_aux),
                token_ids,
                outputs.indices,
                outputs.scores,
                outputs.weights,
                rows,
                hash_routing,
                cuda_stream,
            )
            .context("launching native Pro router from fixed arena input")
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_ds4_flash_router_topk_bf16_device_input_into(
    library: &'static NativeLibrary,
    router_weight: Ds4rtDeviceBuffer,
    router_aux: Ds4rtDeviceBuffer,
    hidden: &DeviceBf16Output,
    token_ids: Option<Ds4rtDeviceBuffer>,
    hash_routing: bool,
    outputs: Ds4FlashRouterTopKDeviceBuffers,
    cuda_stream: *mut c_void,
) -> Result<()> {
    anyhow::ensure!(
        hidden.rows > 0 && hidden.values_per_row == DS4_FLASH_ROUTER_HIDDEN,
        "Flash router device hidden must be non-empty rows x {DS4_FLASH_ROUTER_HIDDEN}, got {}x{}",
        hidden.rows,
        hidden.values_per_row
    );
    validate_ds4_flash_router_topk_device_handoff(
        hidden.buffer(),
        token_ids,
        hash_routing,
        outputs,
        hidden.rows,
    )?;
    hidden
        .wait_ready_on_stream(cuda_stream)
        .context("waiting for Flash router device hidden on its launch stream")?;
    launch_ds4_flash_router_topk_bf16_into(
        library,
        hidden.buffer(),
        router_weight,
        router_aux,
        token_ids,
        hash_routing,
        outputs,
        hidden.rows,
        cuda_stream,
    )
    .context("launching native Flash router into caller-owned buffers")
}

#[allow(clippy::too_many_arguments)]
fn launch_ds4_flash_router_topk_bf16_into(
    library: &'static NativeLibrary,
    hidden: Ds4rtDeviceBuffer,
    router_weight: Ds4rtDeviceBuffer,
    router_aux: Ds4rtDeviceBuffer,
    token_ids: Option<Ds4rtDeviceBuffer>,
    hash_routing: bool,
    outputs: Ds4FlashRouterTopKDeviceBuffers,
    rows: usize,
    cuda_stream: *mut c_void,
) -> Result<()> {
    if !hash_routing && ds4_flash_router_shortlist_enabled() {
        let topk_score_bytes = rows
            .checked_mul(DS4_FLASH_ROUTER_TOP_K)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .context("Flash shortlist top-k score byte count overflows usize")?;
        let approximate_logit_bytes = rows
            .checked_mul(DS4_FLASH_ROUTER_EXPERTS)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .context("Flash shortlist logit byte count overflows usize")?;
        let topk_scores = device_buffer_byte_view(
            outputs.scores,
            0,
            topk_score_bytes,
            "Flash shortlist top-k scores",
        )?;
        let approximate_logits = device_buffer_byte_view(
            outputs.scores,
            topk_score_bytes,
            approximate_logit_bytes,
            "Flash shortlist approximate logits",
        )?;
        unsafe {
            library
                .cuda_linear_bf16_f32_cublas_async(
                    hidden,
                    router_weight,
                    approximate_logits,
                    rows,
                    DS4_FLASH_ROUTER_HIDDEN,
                    DS4_FLASH_ROUTER_EXPERTS,
                    cuda_stream,
                )
                .context("launching Flash shortlist approximate router GEMM")?;
            library
                .cuda_ds4_flash_router_refine_topk_bf16_async(
                    hidden,
                    router_weight,
                    router_aux,
                    approximate_logits,
                    outputs.indices,
                    topk_scores,
                    outputs.weights,
                    rows,
                    cuda_stream,
                )
                .context("launching exact Flash shortlist refinement")
        }
    } else {
        unsafe {
            library
                .cuda_ds4_flash_router_topk_bf16_async(
                    hidden,
                    router_weight,
                    (!hash_routing).then_some(router_aux),
                    hash_routing.then_some(router_aux),
                    token_ids,
                    outputs.indices,
                    outputs.scores,
                    outputs.weights,
                    rows,
                    hash_routing,
                    cuda_stream,
                )
                .context("launching native Flash router")
        }
    }
}

fn preloaded_ds4_flash_router_device_weights(
    router_weight_name: &str,
    router_aux_name: &str,
    hash_routing: bool,
) -> Result<(&'static NativeLibrary, Ds4rtDeviceBuffer, Ds4rtDeviceBuffer)> {
    let library = cuda_native_library()?;
    let weight_bytes =
        DS4_FLASH_ROUTER_EXPERTS * DS4_FLASH_ROUTER_HIDDEN * std::mem::size_of::<u16>();
    let aux_bytes = if hash_routing {
        DS4_FLASH_ROUTER_VOCAB * DS4_FLASH_ROUTER_TOP_K * std::mem::size_of::<i64>()
    } else {
        DS4_FLASH_ROUTER_EXPERTS * std::mem::size_of::<f32>()
    };
    let router_weight = preloaded_resident_weight_device_buffer(router_weight_name, weight_bytes)?;
    let router_aux = preloaded_resident_weight_device_buffer(router_aux_name, aux_bytes)?;
    Ok((library, router_weight, router_aux))
}

fn preloaded_ds4_pro_router_device_weights(
    router_weight_name: &str,
    router_aux_name: &str,
    hash_routing: bool,
) -> Result<(&'static NativeLibrary, Ds4rtDeviceBuffer, Ds4rtDeviceBuffer)> {
    let library = cuda_native_library()?;
    let weight_bytes = DS4_PRO_ROUTER_EXPERTS * DS4_PRO_ROUTER_HIDDEN * std::mem::size_of::<u16>();
    let aux_bytes = if hash_routing {
        DS4_PRO_ROUTER_VOCAB * DS4_PRO_ROUTER_TOP_K * std::mem::size_of::<i64>()
    } else {
        DS4_PRO_ROUTER_EXPERTS * std::mem::size_of::<f32>()
    };
    let router_weight = preloaded_resident_weight_device_buffer(router_weight_name, weight_bytes)?;
    let router_aux = preloaded_resident_weight_device_buffer(router_aux_name, aux_bytes)?;
    Ok((library, router_weight, router_aux))
}

pub(in crate::commands::real_full) fn ds4_flash_router_topk_bf16_preloaded_device_input(
    router_weight_name: &str,
    router_aux_name: &str,
    hidden: &DeviceBf16Output,
    token_ids: Option<&[usize]>,
    hash_routing: bool,
) -> Result<RouterTopKOutput> {
    ds4_router_topk_bf16_preloaded_device_input(
        ModelVariant::Flash,
        router_weight_name,
        router_aux_name,
        hidden,
        token_ids,
        hash_routing,
    )
}

pub(in crate::commands::real_full) fn ds4_pro_router_topk_bf16_preloaded_device_input(
    router_weight_name: &str,
    router_aux_name: &str,
    hidden: &DeviceBf16Output,
    token_ids: Option<&[usize]>,
    hash_routing: bool,
) -> Result<RouterTopKOutput> {
    ds4_router_topk_bf16_preloaded_device_input(
        ModelVariant::Pro,
        router_weight_name,
        router_aux_name,
        hidden,
        token_ids,
        hash_routing,
    )
}

pub(in crate::commands::real_full) fn ds4_flash_router_routes_bf16_preloaded_device_input(
    router_weight_name: &str,
    router_aux_name: &str,
    hidden: &DeviceBf16Output,
    token_ids: Option<&[usize]>,
    hash_routing: bool,
) -> Result<Ds4RouterRouteOutput> {
    ds4_router_routes_bf16_preloaded_device_input(
        ModelVariant::Flash,
        router_weight_name,
        router_aux_name,
        hidden,
        token_ids,
        hash_routing,
    )
}

pub(in crate::commands::real_full) fn ds4_pro_router_routes_bf16_preloaded_device_input(
    router_weight_name: &str,
    router_aux_name: &str,
    hidden: &DeviceBf16Output,
    token_ids: Option<&[usize]>,
    hash_routing: bool,
) -> Result<Ds4RouterRouteOutput> {
    ds4_router_routes_bf16_preloaded_device_input(
        ModelVariant::Pro,
        router_weight_name,
        router_aux_name,
        hidden,
        token_ids,
        hash_routing,
    )
}

fn ds4_router_routes_bf16_preloaded_device_input(
    variant: ModelVariant,
    router_weight_name: &str,
    router_aux_name: &str,
    hidden: &DeviceBf16Output,
    token_ids: Option<&[usize]>,
    hash_routing: bool,
) -> Result<Ds4RouterRouteOutput> {
    let (label, hidden_dim, experts, top_k, backend) = match variant {
        ModelVariant::Flash => (
            "Flash",
            DS4_FLASH_ROUTER_HIDDEN,
            DS4_FLASH_ROUTER_EXPERTS,
            DS4_FLASH_ROUTER_TOP_K,
            CUDA_DS4_FLASH_ROUTER_TOPK_BF16_PRELOADED_DEVICE_INPUT_BACKEND,
        ),
        ModelVariant::Pro => (
            "Pro",
            DS4_PRO_ROUTER_HIDDEN,
            DS4_PRO_ROUTER_EXPERTS,
            DS4_PRO_ROUTER_TOP_K,
            CUDA_DS4_PRO_ROUTER_TOPK_BF16_PRELOADED_DEVICE_INPUT_BACKEND,
        ),
        ModelVariant::Custom => anyhow::bail!("custom DeepSeek V4 router is unsupported"),
    };
    anyhow::ensure!(
        hidden.rows > 0 && hidden.values_per_row == hidden_dim,
        "{label} compact router device hidden must be non-empty rows x {hidden_dim}, got {}x{}",
        hidden.rows,
        hidden.values_per_row
    );
    if hash_routing {
        anyhow::ensure!(
            token_ids.is_some_and(|ids| ids.len() == hidden.rows),
            "{label} compact hash router requires one token ID per hidden row"
        );
    } else {
        anyhow::ensure!(
            token_ids.is_none(),
            "{label} compact learned router does not accept hash token IDs"
        );
    }

    let library = cuda_native_library()?;
    let output_values = hidden
        .rows
        .checked_mul(top_k)
        .with_context(|| format!("{label} compact router output value count overflows usize"))?;
    let index_bytes = output_values
        .checked_mul(std::mem::size_of::<u32>())
        .with_context(|| format!("{label} compact router index byte count overflows usize"))?;
    let weight_bytes = output_values
        .checked_mul(std::mem::size_of::<f32>())
        .with_context(|| format!("{label} compact router weight byte count overflows usize"))?;
    let packed_bytes = index_bytes
        .checked_add(weight_bytes)
        .with_context(|| format!("{label} compact router packed byte count overflows usize"))?;
    let score_workspace_bytes = hidden
        .rows
        .checked_mul(top_k + if hash_routing { 0 } else { experts })
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .with_context(|| {
            format!("{label} compact router score workspace byte count overflows usize")
        })?;
    let token_values = token_ids
        .map(|ids| {
            ids.iter()
                .copied()
                .map(|token_id| {
                    i64::try_from(token_id).with_context(|| {
                        format!("{label} compact hash router token ID does not fit in i64")
                    })
                })
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?;

    let mut workspace = lock_coordinator_cuda_workspace()?;
    let cuda_stream = workspace.stream_ptr(library)?;
    let token_buffer = if let Some(token_values) = token_values.as_ref() {
        let token_bytes = unsafe {
            slice::from_raw_parts(
                token_values.as_ptr().cast::<u8>(),
                std::mem::size_of_val(token_values.as_slice()),
            )
        };
        let token_label = match variant {
            ModelVariant::Flash => "Flash compact hash router token IDs",
            ModelVariant::Pro => "Pro compact hash router token IDs",
            ModelVariant::Custom => unreachable!("custom variant rejected above"),
        };
        let buffer = workspace.buffer(
            library,
            CoordinatorCudaScratchSlot::A,
            token_bytes.len(),
            token_label,
        )?;
        workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::A,
                token_bytes,
                token_label,
                cuda_stream,
            )
            .with_context(|| format!("copying {label} compact hash router token IDs to device"))?;
        Some(buffer)
    } else {
        None
    };

    let packed_label = match variant {
        ModelVariant::Flash => "Flash compact router route metadata",
        ModelVariant::Pro => "Pro compact router route metadata",
        ModelVariant::Custom => unreachable!("custom variant rejected above"),
    };
    let score_label = match variant {
        ModelVariant::Flash => "Flash compact router score scratch",
        ModelVariant::Pro => "Pro compact router score scratch",
        ModelVariant::Custom => unreachable!("custom variant rejected above"),
    };
    let packed_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::D,
        packed_bytes,
        packed_label,
    )?;
    let index_buffer =
        device_buffer_byte_view(packed_buffer, 0, index_bytes, "compact router index view")?;
    let weight_buffer = device_buffer_byte_view(
        packed_buffer,
        index_bytes,
        weight_bytes,
        "compact router weight view",
    )?;
    let score_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::E,
        score_workspace_bytes,
        score_label,
    )?;
    let packed_output = workspace.host_buffer(
        library,
        CoordinatorCudaScratchSlot::D,
        packed_bytes,
        packed_label,
    )?;

    hidden
        .wait_ready_on_stream(cuda_stream)
        .with_context(|| format!("waiting for {label} compact router device hidden"))?;
    let outputs = Ds4FlashRouterTopKDeviceBuffers {
        indices: index_buffer,
        scores: score_buffer,
        weights: weight_buffer,
    };
    match variant {
        ModelVariant::Flash => ds4_flash_router_topk_bf16_preloaded_fixed_device_input_into(
            router_weight_name,
            router_aux_name,
            hidden.buffer(),
            hidden.rows,
            token_buffer,
            hash_routing,
            outputs,
            cuda_stream,
        ),
        ModelVariant::Pro => ds4_pro_router_topk_bf16_preloaded_fixed_device_input_into(
            router_weight_name,
            router_aux_name,
            hidden.buffer(),
            hidden.rows,
            token_buffer,
            hash_routing,
            outputs,
            cuda_stream,
        ),
        ModelVariant::Custom => unreachable!("custom variant rejected above"),
    }?;
    unsafe {
        library
            .copy_d2h_host_buffer_async(packed_output, packed_buffer, packed_bytes, cuda_stream)
            .with_context(|| format!("copying {label} compact router metadata to host"))?;
        library
            .cuda_stream_synchronize(cuda_stream)
            .with_context(|| format!("synchronizing {label} compact router metadata readback"))?;
    }
    let packed_out = unsafe { slice::from_raw_parts(packed_output.ptr.cast::<u8>(), packed_bytes) };
    Ok(Ds4RouterRouteOutput {
        indices: u32_vec_from_bytes(&packed_out[..index_bytes])?
            .into_iter()
            .map(|value| value as usize)
            .collect(),
        weights: f32_vec_from_bytes(&packed_out[index_bytes..])?,
        backend,
    })
}

fn ds4_router_topk_bf16_preloaded_device_input(
    variant: ModelVariant,
    router_weight_name: &str,
    router_aux_name: &str,
    hidden: &DeviceBf16Output,
    token_ids: Option<&[usize]>,
    hash_routing: bool,
) -> Result<RouterTopKOutput> {
    let (label, hidden_dim, experts, top_k, backend) = match variant {
        ModelVariant::Flash => (
            "Flash",
            DS4_FLASH_ROUTER_HIDDEN,
            DS4_FLASH_ROUTER_EXPERTS,
            DS4_FLASH_ROUTER_TOP_K,
            CUDA_DS4_FLASH_ROUTER_TOPK_BF16_PRELOADED_DEVICE_INPUT_BACKEND,
        ),
        ModelVariant::Pro => (
            "Pro",
            DS4_PRO_ROUTER_HIDDEN,
            DS4_PRO_ROUTER_EXPERTS,
            DS4_PRO_ROUTER_TOP_K,
            CUDA_DS4_PRO_ROUTER_TOPK_BF16_PRELOADED_DEVICE_INPUT_BACKEND,
        ),
        ModelVariant::Custom => anyhow::bail!("custom DeepSeek V4 router is unsupported"),
    };
    anyhow::ensure!(
        hidden.rows > 0 && hidden.values_per_row == hidden_dim,
        "{label} router device hidden must be non-empty rows x {hidden_dim}, got {}x{}",
        hidden.rows,
        hidden.values_per_row
    );
    if hash_routing {
        anyhow::ensure!(
            token_ids.is_some_and(|ids| ids.len() == hidden.rows),
            "{label} hash router requires one token ID per hidden row"
        );
    } else {
        anyhow::ensure!(
            token_ids.is_none(),
            "{label} learned router does not accept hash token IDs"
        );
    }
    let library = cuda_native_library()?;
    let output_values = hidden
        .rows
        .checked_mul(top_k)
        .with_context(|| format!("{label} router output value count overflows usize"))?;
    let index_bytes = output_values
        .checked_mul(std::mem::size_of::<u32>())
        .with_context(|| format!("{label} router index byte count overflows usize"))?;
    let score_bytes = output_values
        .checked_mul(std::mem::size_of::<f32>())
        .with_context(|| format!("{label} router score byte count overflows usize"))?;
    let score_workspace_bytes = hidden
        .rows
        .checked_mul(top_k + if hash_routing { 0 } else { experts })
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .with_context(|| format!("{label} router score workspace byte count overflows usize"))?;
    let token_values = token_ids
        .map(|ids| {
            ids.iter()
                .copied()
                .map(|token_id| {
                    i64::try_from(token_id).with_context(|| {
                        format!("{label} hash router token ID does not fit in i64")
                    })
                })
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let token_label = match variant {
        ModelVariant::Flash => "Flash hash router token IDs",
        ModelVariant::Pro => "Pro hash router token IDs",
        ModelVariant::Custom => unreachable!("custom variant rejected above"),
    };
    let cuda_stream = workspace.stream_ptr(library)?;
    let token_buffer = if let Some(token_values) = token_values.as_ref() {
        let token_bytes = unsafe {
            slice::from_raw_parts(
                token_values.as_ptr().cast::<u8>(),
                std::mem::size_of_val(token_values.as_slice()),
            )
        };
        let buffer = workspace.buffer(
            library,
            CoordinatorCudaScratchSlot::A,
            token_bytes.len(),
            token_label,
        )?;
        workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::A,
                token_bytes,
                token_label,
                cuda_stream,
            )
            .with_context(|| format!("copying {label} hash router token IDs to device"))?;
        Some(buffer)
    } else {
        None
    };
    let (index_label, score_label, weight_label) = match variant {
        ModelVariant::Flash => (
            "Flash router top-k indices",
            "Flash router top-k scores",
            "Flash router top-k weights",
        ),
        ModelVariant::Pro => (
            "Pro router top-k indices",
            "Pro router top-k scores",
            "Pro router top-k weights",
        ),
        ModelVariant::Custom => unreachable!("custom variant rejected above"),
    };
    let index_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::D,
        index_bytes,
        index_label,
    )?;
    let score_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::E,
        score_workspace_bytes,
        score_label,
    )?;
    let weight_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::F,
        score_bytes,
        weight_label,
    )?;
    let index_output = workspace.host_buffer(
        library,
        CoordinatorCudaScratchSlot::D,
        index_bytes,
        index_label,
    )?;
    let score_output = workspace.host_buffer(
        library,
        CoordinatorCudaScratchSlot::E,
        score_bytes,
        score_label,
    )?;
    let weight_output = workspace.host_buffer(
        library,
        CoordinatorCudaScratchSlot::F,
        score_bytes,
        weight_label,
    )?;
    hidden
        .wait_ready_on_stream(cuda_stream)
        .with_context(|| format!("waiting for {label} router device hidden"))?;
    let outputs = Ds4FlashRouterTopKDeviceBuffers {
        indices: index_buffer,
        scores: score_buffer,
        weights: weight_buffer,
    };
    match variant {
        ModelVariant::Flash => ds4_flash_router_topk_bf16_preloaded_fixed_device_input_into(
            router_weight_name,
            router_aux_name,
            hidden.buffer(),
            hidden.rows,
            token_buffer,
            hash_routing,
            outputs,
            cuda_stream,
        ),
        ModelVariant::Pro => ds4_pro_router_topk_bf16_preloaded_fixed_device_input_into(
            router_weight_name,
            router_aux_name,
            hidden.buffer(),
            hidden.rows,
            token_buffer,
            hash_routing,
            outputs,
            cuda_stream,
        ),
        ModelVariant::Custom => unreachable!("custom variant rejected above"),
    }?;
    unsafe {
        library
            .copy_d2h_host_buffer_async(index_output, index_buffer, index_bytes, cuda_stream)
            .with_context(|| format!("copying {label} router indices to host"))?;
        library
            .copy_d2h_host_buffer_async(score_output, score_buffer, score_bytes, cuda_stream)
            .with_context(|| format!("copying {label} router scores to host"))?;
        library
            .copy_d2h_host_buffer_async(weight_output, weight_buffer, score_bytes, cuda_stream)
            .with_context(|| format!("copying {label} router weights to host"))?;
        library
            .cuda_stream_synchronize(cuda_stream)
            .with_context(|| format!("synchronizing {label} router metadata readback"))?;
    }
    let index_out = unsafe { slice::from_raw_parts(index_output.ptr.cast::<u8>(), index_bytes) };
    let score_out = unsafe { slice::from_raw_parts(score_output.ptr.cast::<u8>(), score_bytes) };
    let weight_out = unsafe { slice::from_raw_parts(weight_output.ptr.cast::<u8>(), score_bytes) };
    Ok(RouterTopKOutput {
        indices: u32_vec_from_bytes(index_out)?
            .into_iter()
            .map(|value| value as usize)
            .collect(),
        scores: f32_vec_from_bytes(score_out)?,
        weights: f32_vec_from_bytes(weight_out)?,
        backend,
    })
}

fn validate_ds4_flash_router_topk_device_handoff(
    hidden: Ds4rtDeviceBuffer,
    token_ids: Option<Ds4rtDeviceBuffer>,
    hash_routing: bool,
    outputs: Ds4FlashRouterTopKDeviceBuffers,
    rows: usize,
) -> Result<()> {
    validate_ds4_router_topk_device_handoff(
        hidden,
        token_ids,
        hash_routing,
        outputs,
        rows,
        DS4_FLASH_ROUTER_HIDDEN,
        DS4_FLASH_ROUTER_EXPERTS,
        DS4_FLASH_ROUTER_TOP_K,
        "Flash",
    )
}

fn validate_ds4_pro_router_topk_device_handoff(
    hidden: Ds4rtDeviceBuffer,
    token_ids: Option<Ds4rtDeviceBuffer>,
    hash_routing: bool,
    outputs: Ds4FlashRouterTopKDeviceBuffers,
    rows: usize,
) -> Result<()> {
    validate_ds4_router_topk_device_handoff(
        hidden,
        token_ids,
        hash_routing,
        outputs,
        rows,
        DS4_PRO_ROUTER_HIDDEN,
        DS4_PRO_ROUTER_EXPERTS,
        DS4_PRO_ROUTER_TOP_K,
        "Pro",
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_ds4_router_topk_device_handoff(
    hidden: Ds4rtDeviceBuffer,
    token_ids: Option<Ds4rtDeviceBuffer>,
    hash_routing: bool,
    outputs: Ds4FlashRouterTopKDeviceBuffers,
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
    variant: &str,
) -> Result<()> {
    anyhow::ensure!(rows > 0, "{variant} router handoff requires nonzero rows");
    let hidden_bytes = rows
        .checked_mul(hidden_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .with_context(|| format!("{variant} router handoff hidden byte count overflows usize"))?;
    let route_bytes = rows
        .checked_mul(top_k)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u32>()))
        .with_context(|| format!("{variant} router handoff route byte count overflows usize"))?;
    let score_workspace_bytes = rows
        .checked_mul(top_k + if hash_routing { 0 } else { experts })
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .with_context(|| {
            format!("{variant} router handoff score workspace byte count overflows usize")
        })?;
    device_buffer_byte_view(hidden, 0, hidden_bytes, "DeepSeek V4 router handoff hidden")?;
    if hash_routing {
        let token_ids = token_ids
            .with_context(|| format!("{variant} hash router handoff requires device token IDs"))?;
        let token_bytes = rows
            .checked_mul(std::mem::size_of::<i64>())
            .with_context(|| format!("{variant} hash router token byte count overflows usize"))?;
        device_buffer_byte_view(
            token_ids,
            0,
            token_bytes,
            "DeepSeek V4 hash router handoff token IDs",
        )?;
        anyhow::ensure!(
            token_ids.device_id == hidden.device_id,
            "{variant} hash router token IDs must be on hidden device {}",
            hidden.device_id
        );
    } else {
        anyhow::ensure!(
            token_ids.is_none(),
            "{variant} learned router handoff does not accept device token IDs"
        );
    }
    let named = [
        ("indices", outputs.indices, route_bytes),
        ("scores", outputs.scores, score_workspace_bytes),
        ("weights", outputs.weights, route_bytes),
    ];
    for (name, buffer, bytes) in named {
        device_buffer_byte_view(buffer, 0, bytes, name)?;
        anyhow::ensure!(
            buffer.device_id == hidden.device_id,
            "{variant} router {name} must be on hidden device {}",
            hidden.device_id
        );
    }
    for left in 0..named.len() {
        for right in left + 1..named.len() {
            anyhow::ensure!(
                !device_buffer_prefixes_overlap(
                    named[left].1,
                    named[left].2,
                    named[right].1,
                    named[right].2,
                )?,
                "{variant} router {} and {} buffers overlap",
                named[left].0,
                named[right].0
            );
        }
    }
    Ok(())
}

fn device_buffer_prefixes_overlap(
    left: Ds4rtDeviceBuffer,
    left_bytes: usize,
    right: Ds4rtDeviceBuffer,
    right_bytes: usize,
) -> Result<bool> {
    let left_start = left.ptr as usize;
    let right_start = right.ptr as usize;
    let left_end = left_start
        .checked_add(left_bytes)
        .context("DeepSeek V4 router left buffer address overflows usize")?;
    let right_end = right_start
        .checked_add(right_bytes)
        .context("DeepSeek V4 router right buffer address overflows usize")?;
    Ok(left_start < right_end && right_start < left_end)
}

#[cfg(test)]
mod ds4_flash_device_handoff_tests {
    use super::{
        cuda_native_library, cuda_reference_kernels_test_override,
        device_bf16_output_from_bf16_bytes, ds4_pro_router_topk_bf16_preloaded_device_input,
        f32_values_to_bf16_bytes, preload_resident_weight_from_host_staging,
        validate_ds4_flash_router_topk_device_handoff, validate_ds4_pro_router_topk_device_handoff,
        Ds4FlashRouterTopKDeviceBuffers,
        CUDA_DS4_PRO_ROUTER_TOPK_BF16_PRELOADED_DEVICE_INPUT_BACKEND, DS4_FLASH_ROUTER_EXPERTS,
        DS4_FLASH_ROUTER_HIDDEN, DS4_FLASH_ROUTER_TOP_K, DS4_PRO_ROUTER_EXPERTS,
        DS4_PRO_ROUTER_HIDDEN, DS4_PRO_ROUTER_TOP_K,
    };
    use ds4rt_ffi::Ds4rtDeviceBuffer;

    fn buffer(address: usize, bytes: usize, device_id: i32) -> Ds4rtDeviceBuffer {
        Ds4rtDeviceBuffer {
            ptr: address as *mut _,
            bytes,
            device_id,
            flags: 0,
        }
    }

    #[test]
    fn flash_router_handoff_accepts_three_distinct_fixed_device_regions() {
        let rows = 16;
        let hidden = buffer(0x10_0000, rows * DS4_FLASH_ROUTER_HIDDEN * 2, 0);
        let route_bytes = rows * DS4_FLASH_ROUTER_TOP_K * 4;
        let score_workspace_bytes = rows * (DS4_FLASH_ROUTER_TOP_K + DS4_FLASH_ROUTER_EXPERTS) * 4;
        let outputs = Ds4FlashRouterTopKDeviceBuffers {
            indices: buffer(0x20_0000, route_bytes, 0),
            scores: buffer(0x21_0000, score_workspace_bytes, 0),
            weights: buffer(0x22_0000, route_bytes, 0),
        };

        validate_ds4_flash_router_topk_device_handoff(hidden, None, false, outputs, rows)
            .expect("validating learned router fixed handoff");
        validate_ds4_flash_router_topk_device_handoff(
            hidden,
            Some(buffer(0x23_0000, rows * 8, 0)),
            true,
            outputs,
            rows,
        )
        .expect("validating hash router fixed handoff");
    }

    #[test]
    fn pro_router_handoff_accepts_exact_widened_fixed_device_regions() {
        let rows = 5;
        let hidden = buffer(0x10_0000, rows * DS4_PRO_ROUTER_HIDDEN * 2, 0);
        let route_bytes = rows * DS4_PRO_ROUTER_TOP_K * 4;
        let score_workspace_bytes = rows * (DS4_PRO_ROUTER_TOP_K + DS4_PRO_ROUTER_EXPERTS) * 4;
        let outputs = Ds4FlashRouterTopKDeviceBuffers {
            indices: buffer(0x20_0000, route_bytes, 0),
            scores: buffer(0x21_0000, score_workspace_bytes, 0),
            weights: buffer(0x22_0000, route_bytes, 0),
        };

        validate_ds4_pro_router_topk_device_handoff(hidden, None, false, outputs, rows)
            .expect("validating Pro learned router fixed handoff");
        let flash_sized_hidden = buffer(0x10_0000, rows * DS4_FLASH_ROUTER_HIDDEN * 2, 0);
        let error = validate_ds4_pro_router_topk_device_handoff(
            flash_sized_hidden,
            None,
            false,
            outputs,
            rows,
        )
        .expect_err("rejecting a Flash-width hidden view for the Pro router");
        assert!(error.to_string().contains("router handoff hidden"));
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn pro_router_matches_sqrt_softplus_bias_oracle_on_gpu0() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(true);
        let library = cuda_native_library().expect("loading the CUDA native library");
        let rows = 5_usize;
        let route_values = rows * DS4_PRO_ROUTER_TOP_K;

        let mut hidden_values = vec![0.0_f32; rows * DS4_PRO_ROUTER_HIDDEN];
        for row in 0..rows {
            for expert in 0..DS4_PRO_ROUTER_EXPERTS {
                let rotated = (expert + row * 53) % DS4_PRO_ROUTER_EXPERTS;
                hidden_values[row * DS4_PRO_ROUTER_HIDDEN + expert] =
                    (rotated as f32 - 192.0) / 64.0;
            }
        }
        let mut router_weight = vec![0.0_f32; DS4_PRO_ROUTER_EXPERTS * DS4_PRO_ROUTER_HIDDEN];
        for expert in 0..DS4_PRO_ROUTER_EXPERTS {
            router_weight[expert * DS4_PRO_ROUTER_HIDDEN + expert] = 1.0;
        }
        let correction_bias = (0..DS4_PRO_ROUTER_EXPERTS)
            .map(|expert| ((expert * 17 % 31) as f32 - 15.0) / 1_000.0)
            .collect::<Vec<_>>();
        let hidden_bf16 = f32_values_to_bf16_bytes(&hidden_values);
        let router_weight_bf16 = f32_values_to_bf16_bytes(&router_weight);
        let correction_bias_bytes = correction_bias
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect::<Vec<_>>();
        let route_bytes = route_values * std::mem::size_of::<u32>();
        let score_workspace_bytes = rows * (DS4_PRO_ROUTER_TOP_K + DS4_PRO_ROUTER_EXPERTS) * 4;
        let sizes = [
            hidden_bf16.len(),
            router_weight_bf16.len(),
            correction_bias_bytes.len(),
            route_bytes,
            score_workspace_bytes,
            route_bytes,
        ];
        let mut allocations = sizes
            .iter()
            .enumerate()
            .map(|(index, &bytes)| {
                library.alloc_device_buffer(bytes).unwrap_or_else(|error| {
                    panic!("allocating Pro router test buffer {index}: {error:#}")
                })
            })
            .collect::<Vec<_>>();
        library
            .copy_h2d(allocations[0], &hidden_bf16)
            .expect("uploading Pro router hidden");
        library
            .copy_h2d(allocations[1], &router_weight_bf16)
            .expect("uploading Pro router weight");
        library
            .copy_h2d(allocations[2], &correction_bias_bytes)
            .expect("uploading Pro router correction bias");
        unsafe {
            library
                .cuda_ds4_pro_router_topk_bf16_async(
                    allocations[0],
                    allocations[1],
                    Some(allocations[2]),
                    None,
                    None,
                    allocations[3],
                    allocations[4],
                    allocations[5],
                    rows,
                    false,
                    std::ptr::null_mut(),
                )
                .expect("launching Pro router");
        }
        let mut index_bytes = vec![0_u8; route_bytes];
        let mut score_bytes = vec![0_u8; route_bytes];
        let mut weight_bytes = vec![0_u8; route_bytes];
        library
            .copy_d2h(&mut index_bytes, allocations[3])
            .expect("reading Pro router indices");
        library
            .copy_d2h(&mut score_bytes, allocations[4])
            .expect("reading Pro router scores");
        library
            .copy_d2h(&mut weight_bytes, allocations[5])
            .expect("reading Pro router weights");

        let actual_indices = index_bytes
            .chunks_exact(4)
            .map(|bytes| u32::from_ne_bytes(bytes.try_into().unwrap()) as usize)
            .collect::<Vec<_>>();
        let actual_scores = score_bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        let actual_weights = weight_bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        for row in 0..rows {
            let mut oracle = (0..DS4_PRO_ROUTER_EXPERTS)
                .map(|expert| {
                    let logit = hidden_values[row * DS4_PRO_ROUTER_HIDDEN + expert];
                    let softplus = logit.max(0.0) + (-logit.abs()).exp().ln_1p();
                    let score = softplus.sqrt();
                    (expert, score, score + correction_bias[expert])
                })
                .collect::<Vec<_>>();
            oracle.sort_by(|left, right| {
                right
                    .2
                    .total_cmp(&left.2)
                    .then_with(|| left.0.cmp(&right.0))
            });
            let oracle = &oracle[..DS4_PRO_ROUTER_TOP_K];
            let oracle_sum = oracle.iter().map(|entry| entry.1).sum::<f32>();
            for rank in 0..DS4_PRO_ROUTER_TOP_K {
                let offset = row * DS4_PRO_ROUTER_TOP_K + rank;
                assert_eq!(actual_indices[offset], oracle[rank].0);
                assert!((actual_scores[offset] - oracle[rank].1).abs() <= 2.0e-5);
                let expected_weight = oracle[rank].1 / oracle_sum * 2.5;
                assert!((actual_weights[offset] - expected_weight).abs() <= 2.0e-5);
            }
            let actual_sum = actual_weights
                [row * DS4_PRO_ROUTER_TOP_K..(row + 1) * DS4_PRO_ROUTER_TOP_K]
                .iter()
                .sum::<f32>();
            assert!((actual_sum - 2.5).abs() <= 2.0e-5);
        }

        let resident_weight_name = "ds4rt.test.pro.router.weight";
        let resident_bias_name = "ds4rt.test.pro.router.bias";
        preload_resident_weight_from_host_staging(
            resident_weight_name,
            router_weight_bf16.len(),
            "Pro router wrapper test weight",
            |staging| {
                staging.copy_from_slice(&router_weight_bf16);
                Ok(())
            },
        )
        .expect("preloading Pro router wrapper test weight");
        preload_resident_weight_from_host_staging(
            resident_bias_name,
            correction_bias_bytes.len(),
            "Pro router wrapper test bias",
            |staging| {
                staging.copy_from_slice(&correction_bias_bytes);
                Ok(())
            },
        )
        .expect("preloading Pro router wrapper test bias");
        let hidden_device = device_bf16_output_from_bf16_bytes(
            &hidden_bf16,
            rows,
            DS4_PRO_ROUTER_HIDDEN,
            "Pro router wrapper test hidden",
        )
        .expect("uploading Pro router wrapper test hidden");
        let wrapped = ds4_pro_router_topk_bf16_preloaded_device_input(
            resident_weight_name,
            resident_bias_name,
            &hidden_device,
            None,
            false,
        )
        .expect("executing resident Pro router wrapper");
        assert_eq!(
            wrapped.backend,
            CUDA_DS4_PRO_ROUTER_TOPK_BF16_PRELOADED_DEVICE_INPUT_BACKEND
        );
        assert_eq!(wrapped.indices, actual_indices);
        assert_eq!(wrapped.scores, actual_scores);
        assert_eq!(wrapped.weights, actual_weights);

        for (index, buffer) in allocations.iter_mut().enumerate().rev() {
            library.free_device_buffer(buffer).unwrap_or_else(|error| {
                panic!("freeing Pro router test buffer {index}: {error:#}")
            });
        }
    }

    #[test]
    fn flash_router_handoff_rejects_missing_or_aliased_score_scratch() {
        let rows = 1;
        let hidden = buffer(0x10_0000, DS4_FLASH_ROUTER_HIDDEN * 2, 0);
        let route_bytes = DS4_FLASH_ROUTER_TOP_K * 4;
        let score_workspace_bytes = (DS4_FLASH_ROUTER_TOP_K + DS4_FLASH_ROUTER_EXPERTS) * 4;
        let short_scores = Ds4FlashRouterTopKDeviceBuffers {
            indices: buffer(0x20_0000, route_bytes, 0),
            scores: buffer(0x21_0000, score_workspace_bytes - 1, 0),
            weights: buffer(0x22_0000, route_bytes, 0),
        };
        let error =
            validate_ds4_flash_router_topk_device_handoff(hidden, None, false, short_scores, rows)
                .expect_err("rejecting undersized score scratch");
        assert!(error.to_string().contains("scores device buffer view"));

        let aliased_scores = Ds4FlashRouterTopKDeviceBuffers {
            indices: buffer(0x20_0000, route_bytes, 0),
            scores: buffer(0x20_0000, score_workspace_bytes, 0),
            weights: buffer(0x22_0000, route_bytes, 0),
        };
        let error = validate_ds4_flash_router_topk_device_handoff(
            hidden,
            None,
            false,
            aliased_scores,
            rows,
        )
        .expect_err("rejecting aliased score scratch");
        assert!(error
            .to_string()
            .contains("indices and scores buffers overlap"));
    }
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn router_topk(
    hidden: &[f32],
    router_weight: &[f32],
    correction_bias: &[f32],
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
) -> Result<RouterTopKOutput> {
    validate_router_topk_inputs(
        hidden,
        router_weight,
        correction_bias,
        rows,
        hidden_dim,
        experts,
        top_k,
    )?;
    if cuda_reference_kernels_enabled() {
        return cuda_router_topk(
            hidden,
            router_weight,
            correction_bias,
            rows,
            hidden_dim,
            experts,
            top_k,
        );
    }
    Ok(cpu_router_topk(
        hidden,
        router_weight,
        correction_bias,
        rows,
        hidden_dim,
        experts,
        top_k,
    ))
}

pub(in crate::commands::real_full) fn router_topk_bf16(
    hidden_bf16: &[u8],
    router_weight_bf16: &[u8],
    correction_bias: &[f32],
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
) -> Result<RouterTopKOutput> {
    validate_router_topk_bf16_inputs(
        hidden_bf16,
        router_weight_bf16,
        correction_bias,
        rows,
        hidden_dim,
        experts,
        top_k,
    )?;
    if cuda_reference_kernels_enabled() {
        return cuda_router_topk_bf16(
            hidden_bf16,
            router_weight_bf16,
            correction_bias,
            rows,
            hidden_dim,
            experts,
            top_k,
        );
    }
    Ok(cpu_router_topk_bf16(
        hidden_bf16,
        router_weight_bf16,
        correction_bias,
        rows,
        hidden_dim,
        experts,
        top_k,
    ))
}

pub(in crate::commands::real_full) fn router_topk_bf16_resident_weight(
    router_weight_name: &str,
    hidden_bf16: &[u8],
    router_weight_bf16: &[u8],
    correction_bias: &[f32],
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
) -> Result<RouterTopKOutput> {
    validate_resident_weight_name(router_weight_name)?;
    validate_router_topk_bf16_inputs(
        hidden_bf16,
        router_weight_bf16,
        correction_bias,
        rows,
        hidden_dim,
        experts,
        top_k,
    )?;
    if cuda_reference_kernels_enabled() {
        return cuda_router_topk_bf16_resident_weight(
            router_weight_name,
            hidden_bf16,
            router_weight_bf16,
            correction_bias,
            rows,
            hidden_dim,
            experts,
            top_k,
        );
    }
    Ok(cpu_router_topk_bf16(
        hidden_bf16,
        router_weight_bf16,
        correction_bias,
        rows,
        hidden_dim,
        experts,
        top_k,
    ))
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn router_topk_bf16_preloaded_resident_weight(
    router_weight_name: &str,
    hidden_bf16: &[u8],
    correction_bias: &[f32],
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
) -> Result<RouterTopKOutput> {
    validate_resident_weight_name(router_weight_name)?;
    let weight_bytes = validate_router_topk_bf16_preloaded_resident_inputs(
        hidden_bf16,
        correction_bias,
        rows,
        hidden_dim,
        experts,
        top_k,
    )?;
    if !cuda_reference_kernels_enabled() {
        anyhow::bail!("preloaded resident BF16 router top-k requires CUDA reference kernels");
    }
    cuda_router_topk_bf16_preloaded_resident_weight(
        router_weight_name,
        hidden_bf16,
        correction_bias,
        rows,
        hidden_dim,
        experts,
        top_k,
        weight_bytes,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn router_topk_bf16_preloaded_resident_weight_bias(
    router_weight_name: &str,
    correction_bias_name: &str,
    hidden_bf16: &[u8],
    correction_bias: &[f32],
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
) -> Result<RouterTopKOutput> {
    validate_resident_weight_name(router_weight_name)?;
    validate_resident_weight_name(correction_bias_name)?;
    let weight_bytes = validate_router_topk_bf16_preloaded_resident_inputs(
        hidden_bf16,
        correction_bias,
        rows,
        hidden_dim,
        experts,
        top_k,
    )?;
    let bias_bytes = correction_bias
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .context(
            "real full preloaded BF16 router top-k correction bias byte count overflows usize",
        )?;
    if !cuda_reference_kernels_enabled() {
        anyhow::bail!(
            "preloaded resident BF16 router top-k weight+bias requires CUDA reference kernels"
        );
    }
    cuda_router_topk_bf16_preloaded_resident_weight_bias(
        router_weight_name,
        correction_bias_name,
        hidden_bf16,
        rows,
        hidden_dim,
        experts,
        top_k,
        weight_bytes,
        bias_bytes,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn router_topk_bf16_resident_weight_device_input(
    router_weight_name: &str,
    hidden: &DeviceBf16Output,
    router_weight_bf16: &[u8],
    correction_bias: &[f32],
    experts: usize,
    top_k: usize,
) -> Result<RouterTopKOutput> {
    validate_resident_weight_name(router_weight_name)?;
    validate_router_topk_bf16_resident_device_input(
        hidden,
        router_weight_bf16,
        correction_bias,
        experts,
        top_k,
    )?;
    if !cuda_reference_kernels_enabled() {
        anyhow::bail!("resident BF16 router top-k device-input requires CUDA reference kernels");
    }
    cuda_router_topk_bf16_resident_weight_device_input(
        router_weight_name,
        hidden,
        router_weight_bf16,
        correction_bias,
        experts,
        top_k,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn router_topk_bf16_preloaded_resident_weight_device_input(
    router_weight_name: &str,
    hidden: &DeviceBf16Output,
    correction_bias: &[f32],
    experts: usize,
    top_k: usize,
) -> Result<RouterTopKOutput> {
    validate_resident_weight_name(router_weight_name)?;
    let weight_bytes = validate_router_topk_bf16_preloaded_resident_device_input(
        hidden,
        Some(correction_bias),
        experts,
        top_k,
    )?;
    if !cuda_reference_kernels_enabled() {
        anyhow::bail!(
            "preloaded resident BF16 router top-k device-input requires CUDA reference kernels"
        );
    }
    cuda_router_topk_bf16_preloaded_resident_weight_device_input(
        router_weight_name,
        hidden,
        correction_bias,
        experts,
        top_k,
        weight_bytes,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn router_topk_bf16_preloaded_resident_weight_bias_device_input(
    router_weight_name: &str,
    correction_bias_name: &str,
    hidden: &DeviceBf16Output,
    experts: usize,
    top_k: usize,
) -> Result<RouterTopKOutput> {
    validate_resident_weight_name(router_weight_name)?;
    validate_resident_weight_name(correction_bias_name)?;
    let weight_bytes =
        validate_router_topk_bf16_preloaded_resident_device_input(hidden, None, experts, top_k)?;
    let bias_bytes = experts
        .checked_mul(std::mem::size_of::<f32>())
        .context(
            "real full preloaded BF16 router top-k device-input correction bias byte count overflows usize",
        )?;
    if !cuda_reference_kernels_enabled() {
        anyhow::bail!(
            "preloaded resident BF16 router top-k weight+bias device-input requires CUDA reference kernels"
        );
    }
    cuda_router_topk_bf16_preloaded_resident_weight_bias_device_input(
        router_weight_name,
        correction_bias_name,
        hidden,
        experts,
        top_k,
        weight_bytes,
        bias_bytes,
    )
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn validate_router_topk_inputs(
    hidden: &[f32],
    router_weight: &[f32],
    correction_bias: &[f32],
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
) -> Result<()> {
    if rows == 0 || hidden_dim == 0 || experts == 0 {
        anyhow::bail!(
            "real full router top-k requires non-zero shape, got rows={rows} hidden_dim={hidden_dim} experts={experts}"
        );
    }
    if top_k == 0 || top_k > experts || top_k > DS4RT_CUDA_ROUTER_TOPK_MAX_K {
        anyhow::bail!(
            "real full router top-k invalid top_k={top_k} for experts={experts}; max supported top_k={DS4RT_CUDA_ROUTER_TOPK_MAX_K}"
        );
    }
    let expected_hidden = rows.checked_mul(hidden_dim).context(
        "real full router top-k hidden shape overflows usize while validating coordinator kernel input",
    )?;
    if hidden.len() != expected_hidden {
        anyhow::bail!(
            "real full router top-k hidden length mismatch: expected {} got {}",
            expected_hidden,
            hidden.len()
        );
    }
    let expected_weight = experts.checked_mul(hidden_dim).context(
        "real full router top-k weight shape overflows usize while validating coordinator kernel input",
    )?;
    if router_weight.len() != expected_weight {
        anyhow::bail!(
            "real full router top-k weight length mismatch: expected {} got {}",
            expected_weight,
            router_weight.len()
        );
    }
    if correction_bias.len() != experts {
        anyhow::bail!(
            "real full router top-k correction bias length mismatch: expected {} got {}",
            experts,
            correction_bias.len()
        );
    }
    Ok(())
}

pub(in crate::commands::real_full) fn validate_router_topk_bf16_inputs(
    hidden_bf16: &[u8],
    router_weight_bf16: &[u8],
    correction_bias: &[f32],
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
) -> Result<()> {
    if rows == 0 || hidden_dim == 0 || experts == 0 {
        anyhow::bail!(
            "real full BF16 router top-k requires non-zero shape, got rows={rows} hidden_dim={hidden_dim} experts={experts}"
        );
    }
    if top_k == 0 || top_k > experts || top_k > DS4RT_CUDA_ROUTER_TOPK_MAX_K {
        anyhow::bail!(
            "real full BF16 router top-k invalid top_k={top_k} for experts={experts}; max supported top_k={DS4RT_CUDA_ROUTER_TOPK_MAX_K}"
        );
    }
    let expected_hidden = rows
        .checked_mul(hidden_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "real full BF16 router top-k hidden shape overflows usize while validating input",
        )?;
    if hidden_bf16.len() != expected_hidden {
        anyhow::bail!(
            "real full BF16 router top-k hidden byte length mismatch: expected {} got {}",
            expected_hidden,
            hidden_bf16.len()
        );
    }
    let expected_weight = experts
        .checked_mul(hidden_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "real full BF16 router top-k weight shape overflows usize while validating input",
        )?;
    if router_weight_bf16.len() != expected_weight {
        anyhow::bail!(
            "real full BF16 router top-k weight byte length mismatch: expected {} got {}",
            expected_weight,
            router_weight_bf16.len()
        );
    }
    if correction_bias.len() != experts {
        anyhow::bail!(
            "real full BF16 router top-k correction bias length mismatch: expected {} got {}",
            experts,
            correction_bias.len()
        );
    }
    Ok(())
}

pub(in crate::commands::real_full) fn validate_router_topk_bf16_preloaded_resident_inputs(
    hidden_bf16: &[u8],
    correction_bias: &[f32],
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
) -> Result<usize> {
    if rows == 0 || hidden_dim == 0 || experts == 0 {
        anyhow::bail!(
            "real full preloaded BF16 router top-k requires non-zero shape, got rows={rows} hidden_dim={hidden_dim} experts={experts}"
        );
    }
    if top_k == 0 || top_k > experts || top_k > DS4RT_CUDA_ROUTER_TOPK_MAX_K {
        anyhow::bail!(
            "real full preloaded BF16 router top-k invalid top_k={top_k} for experts={experts}; max supported top_k={DS4RT_CUDA_ROUTER_TOPK_MAX_K}"
        );
    }
    let expected_hidden = rows
        .checked_mul(hidden_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "real full preloaded BF16 router top-k hidden shape overflows usize while validating input",
        )?;
    if hidden_bf16.len() != expected_hidden {
        anyhow::bail!(
            "real full preloaded BF16 router top-k hidden byte length mismatch: expected {} got {}",
            expected_hidden,
            hidden_bf16.len()
        );
    }
    let expected_weight = experts
        .checked_mul(hidden_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "real full preloaded BF16 router top-k weight shape overflows usize while validating input",
        )?;
    if correction_bias.len() != experts {
        anyhow::bail!(
            "real full preloaded BF16 router top-k correction bias length mismatch: expected {} got {}",
            experts,
            correction_bias.len()
        );
    }
    Ok(expected_weight)
}

pub(in crate::commands::real_full) fn validate_router_topk_bf16_resident_device_input(
    hidden: &DeviceBf16Output,
    router_weight_bf16: &[u8],
    correction_bias: &[f32],
    experts: usize,
    top_k: usize,
) -> Result<()> {
    let expected_weight = validate_router_topk_bf16_preloaded_resident_device_input(
        hidden,
        Some(correction_bias),
        experts,
        top_k,
    )?;
    if router_weight_bf16.len() != expected_weight {
        anyhow::bail!(
            "real full BF16 router top-k device-input weight byte length mismatch: expected {} got {}",
            expected_weight,
            router_weight_bf16.len()
        );
    }
    Ok(())
}

pub(in crate::commands::real_full) fn validate_router_topk_bf16_preloaded_resident_device_input(
    hidden: &DeviceBf16Output,
    correction_bias: Option<&[f32]>,
    experts: usize,
    top_k: usize,
) -> Result<usize> {
    if hidden.rows == 0 || hidden.values_per_row == 0 || experts == 0 {
        anyhow::bail!(
            "real full preloaded BF16 router top-k device-input requires non-zero shape, got rows={} hidden_dim={} experts={experts}",
            hidden.rows,
            hidden.values_per_row
        );
    }
    if top_k == 0 || top_k > experts || top_k > DS4RT_CUDA_ROUTER_TOPK_MAX_K {
        anyhow::bail!(
            "real full preloaded BF16 router top-k device-input invalid top_k={top_k} for experts={experts}; max supported top_k={DS4RT_CUDA_ROUTER_TOPK_MAX_K}"
        );
    }
    let expected_hidden = hidden
        .rows
        .checked_mul(hidden.values_per_row)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "real full preloaded BF16 router top-k device-input hidden shape overflows usize while validating input",
        )?;
    if hidden.bytes != expected_hidden {
        anyhow::bail!(
            "real full preloaded BF16 router top-k device-input hidden byte length mismatch: expected {} got {}",
            expected_hidden,
            hidden.bytes
        );
    }
    let buffer = hidden.buffer();
    if buffer.ptr.is_null() {
        anyhow::bail!("real full preloaded BF16 router top-k device-input hidden buffer is null");
    }
    if buffer.bytes < expected_hidden {
        anyhow::bail!(
            "real full preloaded BF16 router top-k device-input hidden buffer has {} bytes, needs {expected_hidden}",
            buffer.bytes
        );
    }
    let expected_weight = experts
        .checked_mul(hidden.values_per_row)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context(
            "real full preloaded BF16 router top-k device-input weight shape overflows usize while validating input",
        )?;
    if let Some(correction_bias) = correction_bias {
        if correction_bias.len() != experts {
            anyhow::bail!(
                "real full preloaded BF16 router top-k device-input correction bias length mismatch: expected {} got {}",
                experts,
                correction_bias.len()
            );
        }
    }
    Ok(expected_weight)
}

pub(in crate::commands::real_full) fn cpu_router_topk(
    hidden: &[f32],
    router_weight: &[f32],
    correction_bias: &[f32],
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
) -> RouterTopKOutput {
    let mut indices = vec![0_usize; rows * top_k];
    let mut scores = vec![0.0_f32; rows * top_k];
    let mut weights = vec![0.0_f32; rows * top_k];
    for row in 0..rows {
        let mut best_scores = vec![0.0_f32; top_k];
        let mut best_corrected = vec![f32::NEG_INFINITY; top_k];
        let mut best_indices = vec![0_usize; top_k];
        let hidden_start = row * hidden_dim;
        for expert in 0..experts {
            let weight_start = expert * hidden_dim;
            let mut logit = 0.0_f32;
            for col in 0..hidden_dim {
                logit += hidden[hidden_start + col] * router_weight[weight_start + col];
            }
            let score = 1.0 / (1.0 + (-logit).exp());
            let corrected = score + correction_bias[expert];
            for rank in 0..top_k {
                if corrected > best_corrected[rank] {
                    for shift in (rank + 1..top_k).rev() {
                        best_corrected[shift] = best_corrected[shift - 1];
                        best_scores[shift] = best_scores[shift - 1];
                        best_indices[shift] = best_indices[shift - 1];
                    }
                    best_corrected[rank] = corrected;
                    best_scores[rank] = score;
                    best_indices[rank] = expert;
                    break;
                }
            }
        }
        let score_sum = best_scores.iter().sum::<f32>().max(1.0e-12);
        let out_start = row * top_k;
        for rank in 0..top_k {
            indices[out_start + rank] = best_indices[rank];
            scores[out_start + rank] = best_scores[rank];
            weights[out_start + rank] = best_scores[rank] / score_sum * GLM52_ROUTED_SCALING_FACTOR;
        }
    }
    RouterTopKOutput {
        indices,
        scores,
        weights,
        backend: CPU_REFERENCE_ROUTER_TOPK_BACKEND,
    }
}

pub(in crate::commands::real_full) fn cpu_router_topk_bf16(
    hidden_bf16: &[u8],
    router_weight_bf16: &[u8],
    correction_bias: &[f32],
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
) -> RouterTopKOutput {
    let mut output = cpu_router_topk(
        &bf16_values_to_f32(hidden_bf16),
        &bf16_values_to_f32(router_weight_bf16),
        correction_bias,
        rows,
        hidden_dim,
        experts,
        top_k,
    );
    output.backend = CPU_REFERENCE_ROUTER_TOPK_BF16_BACKEND;
    output
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn cuda_router_topk(
    hidden: &[f32],
    router_weight: &[f32],
    correction_bias: &[f32],
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
) -> Result<RouterTopKOutput> {
    let library = cuda_native_library()?;
    let hidden_bytes = std::mem::size_of_val(hidden);
    let weight_bytes = std::mem::size_of_val(router_weight);
    let bias_bytes = std::mem::size_of_val(correction_bias);
    let output_values = rows
        .checked_mul(top_k)
        .context("CUDA router top-k output shape overflows usize")?;
    let index_bytes = output_values
        .checked_mul(std::mem::size_of::<u32>())
        .context("CUDA router top-k index bytes overflow usize")?;
    let score_bytes = output_values
        .checked_mul(std::mem::size_of::<f32>())
        .context("CUDA router top-k score bytes overflow usize")?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let hidden_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::A,
        hidden_bytes,
        "router hidden",
    )?;
    let weight_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::B,
        weight_bytes,
        "router weight",
    )?;
    let bias_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::C,
        bias_bytes,
        "router correction bias",
    )?;
    let index_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::D,
        index_bytes,
        "router top-k indices",
    )?;
    let score_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::E,
        score_bytes,
        "router top-k scores",
    )?;
    let weight_buffer_out = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::F,
        score_bytes,
        "router top-k weights",
    )?;

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::A,
            f32_bytes(hidden),
            "router hidden",
        )
        .context("copying router hidden to device")?;
    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::B,
            f32_bytes(router_weight),
            "router weight",
        )
        .context("copying router weight to device")?;
    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::C,
            f32_bytes(correction_bias),
            "router correction bias",
        )
        .context("copying router correction bias to device")?;
    library
        .cuda_router_topk_f32(
            hidden_buffer,
            weight_buffer,
            bias_buffer,
            index_buffer,
            score_buffer,
            weight_buffer_out,
            rows,
            hidden_dim,
            experts,
            top_k,
        )
        .context("executing CUDA router top-k")?;
    let mut index_out = vec![0_u8; index_bytes];
    let mut score_out = vec![0_u8; score_bytes];
    let mut weight_out = vec![0_u8; score_bytes];
    library
        .copy_d2h(&mut index_out, index_buffer)
        .context("copying router top-k indices to host")?;
    library
        .copy_d2h(&mut score_out, score_buffer)
        .context("copying router top-k scores to host")?;
    library
        .copy_d2h(&mut weight_out, weight_buffer_out)
        .context("copying router top-k weights to host")?;

    Ok(RouterTopKOutput {
        indices: u32_vec_from_bytes(&index_out)?
            .into_iter()
            .map(|value| value as usize)
            .collect(),
        scores: f32_vec_from_bytes(&score_out)?,
        weights: f32_vec_from_bytes(&weight_out)?,
        backend: CUDA_REFERENCE_ROUTER_TOPK_BACKEND,
    })
}

pub(in crate::commands::real_full) fn cuda_router_topk_bf16(
    hidden_bf16: &[u8],
    router_weight_bf16: &[u8],
    correction_bias: &[f32],
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
) -> Result<RouterTopKOutput> {
    if let Some(graph_key) = coord_sparse_a_graph_key_for_full_hidden_rows(rows, hidden_dim)? {
        return cuda_router_topk_bf16_graph_slot(
            &graph_key,
            hidden_bf16,
            router_weight_bf16,
            correction_bias,
            rows,
            hidden_dim,
            experts,
            top_k,
        );
    }
    cuda_router_topk_bf16_legacy(
        hidden_bf16,
        router_weight_bf16,
        correction_bias,
        rows,
        hidden_dim,
        experts,
        top_k,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_router_topk_bf16_graph_slot(
    graph_key: &CoordinatorGraphKey,
    hidden_bf16: &[u8],
    router_weight_bf16: &[u8],
    correction_bias: &[f32],
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
) -> Result<RouterTopKOutput> {
    let weight_bytes = router_weight_bf16.len();
    let bias_bytes = std::mem::size_of_val(correction_bias);
    let (_, index_bytes, score_bytes) =
        router_topk_output_byte_counts(rows, top_k, "CUDA BF16 graph-slot router top-k")?;
    let hidden_graph_bytes =
        router_topk_graph_hidden_bytes(graph_key, hidden_dim, "CUDA BF16 graph-slot router top-k")?;
    let (_, graph_index_bytes, graph_score_bytes) = router_topk_graph_output_byte_counts(
        graph_key,
        top_k,
        "CUDA BF16 graph-slot router top-k",
    )?;
    let signature = router_topk_graph_signature(graph_key, hidden_dim, experts, top_k);
    with_coordinator_cuda_graph_slot(graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        let hidden_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::A,
            hidden_graph_bytes,
            "BF16 router hidden",
        )?;
        let weight_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::B,
            weight_bytes,
            "BF16 router weight",
        )?;
        let bias_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::C,
            bias_bytes,
            "BF16 router correction bias",
        )?;
        let index_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::D,
            graph_index_bytes,
            "BF16 router top-k indices",
        )?;
        let score_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::E,
            graph_score_bytes,
            "BF16 router top-k scores",
        )?;
        let weight_buffer_out = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::F,
            graph_score_bytes,
            "BF16 router top-k weights",
        )?;

        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::A,
                hidden_bf16,
                "BF16 router hidden",
                cuda_stream,
            )
            .context("async copying BF16 router hidden to device")?;
        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::B,
                router_weight_bf16,
                "BF16 router weight",
                cuda_stream,
            )
            .context("async copying BF16 router weight to device")?;
        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::C,
                f32_bytes(correction_bias),
                "BF16 router correction bias",
                cuda_stream,
            )
            .context("async copying BF16 router correction bias to device")?;
        capture_or_update_sparse_a_router_topk_bf16_graph(
            library,
            slot,
            signature,
            hidden_buffer,
            weight_buffer,
            bias_buffer,
            index_buffer,
            score_buffer,
            weight_buffer_out,
            rows,
            hidden_dim,
            experts,
            top_k,
            "BF16 router top-k",
        )?;
        let mut index_out = vec![0_u8; index_bytes];
        let mut score_out = vec![0_u8; score_bytes];
        let mut weight_out = vec![0_u8; score_bytes];
        unsafe {
            library
                .copy_d2h_async(&mut index_out, index_buffer, cuda_stream)
                .context("async copying BF16 router top-k indices to host")?;
            library
                .copy_d2h_async(&mut score_out, score_buffer, cuda_stream)
                .context("async copying BF16 router top-k scores to host")?;
            library
                .copy_d2h_async(&mut weight_out, weight_buffer_out, cuda_stream)
                .context("async copying BF16 router top-k weights to host")?;
            library
                .cuda_stream_synchronize(cuda_stream)
                .context("synchronizing BF16 router top-k graph slot stream")?;
        }

        Ok(RouterTopKOutput {
            indices: u32_vec_from_bytes(&index_out)?
                .into_iter()
                .map(|value| value as usize)
                .collect(),
            scores: f32_vec_from_bytes(&score_out)?,
            weights: f32_vec_from_bytes(&weight_out)?,
            backend: CUDA_REFERENCE_ROUTER_TOPK_BF16_BACKEND,
        })
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn capture_or_update_sparse_a_router_topk_bf16_graph(
    library: &'static NativeLibrary,
    slot: &mut CoordinatorCudaGraphWorkspaceSlot,
    signature: CoordinatorCudaGraphSignature,
    hidden_buffer: Ds4rtDeviceBuffer,
    weight_buffer: Ds4rtDeviceBuffer,
    bias_buffer: Ds4rtDeviceBuffer,
    index_buffer: Ds4rtDeviceBuffer,
    score_buffer: Ds4rtDeviceBuffer,
    weight_buffer_out: Ds4rtDeviceBuffer,
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
    label: &'static str,
) -> Result<()> {
    if !slot.has_captured_graph(
        CoordinatorCudaGraphProgram::SparseARouterTopKBf16,
        signature,
    ) {
        slot.stream_synchronize()
            .with_context(|| format!("synchronizing {label} inputs before graph capture"))?;
        slot.capture_graph(
            library,
            CoordinatorCudaGraphProgram::SparseARouterTopKBf16,
            signature,
            |library, cuda_stream, _workspace| unsafe {
                library
                    .cuda_router_topk_bf16_async(
                        hidden_buffer,
                        weight_buffer,
                        bias_buffer,
                        index_buffer,
                        score_buffer,
                        weight_buffer_out,
                        rows,
                        hidden_dim,
                        experts,
                        top_k,
                        cuda_stream,
                    )
                    .with_context(|| format!("capturing async CUDA {label}"))?;
                Ok(())
            },
        )?;
    } else {
        let (graph_raw, exec_raw) = slot
            .captured_graph_raw_handles(
                CoordinatorCudaGraphProgram::SparseARouterTopKBf16,
                signature,
            )
            .context(
                "coordinator CUDA graph slot lost captured router top-k graph before update",
            )?;
        unsafe {
            library
                .cuda_graph_update_router_topk_bf16_node(
                    graph_raw,
                    exec_raw,
                    0,
                    hidden_buffer,
                    weight_buffer,
                    bias_buffer,
                    index_buffer,
                    score_buffer,
                    weight_buffer_out,
                    rows,
                    hidden_dim,
                    experts,
                    top_k,
                )
                .with_context(|| format!("updating captured CUDA {label} graph node"))?;
        }
    }
    slot.launch_captured_graph(
        library,
        CoordinatorCudaGraphProgram::SparseARouterTopKBf16,
        signature,
    )
}

#[allow(clippy::too_many_arguments)]
fn capture_or_update_router_topk_bf16_graph_for_slot(
    graph_key: &CoordinatorGraphKey,
    library: &'static NativeLibrary,
    slot: &mut CoordinatorCudaGraphWorkspaceSlot,
    hidden_buffer: Ds4rtDeviceBuffer,
    weight_buffer: Ds4rtDeviceBuffer,
    bias_buffer: Ds4rtDeviceBuffer,
    index_buffer: Ds4rtDeviceBuffer,
    score_buffer: Ds4rtDeviceBuffer,
    weight_buffer_out: Ds4rtDeviceBuffer,
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
    native_backend: &'static str,
    triton_backend: &'static str,
    label: &'static str,
) -> Result<&'static str> {
    if triton_router_topk_bf16_supported(graph_key, rows, hidden_dim, experts, top_k) {
        let capture_rows = graph_key.row_bucket.row_capacity;
        let hidden_bytes = rows
            .checked_mul(hidden_dim)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .with_context(|| format!("{label} Triton hidden bytes overflow usize"))?;
        let capture_hidden_bytes = capture_rows
            .checked_mul(hidden_dim)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .with_context(|| format!("{label} Triton graph hidden bytes overflow usize"))?;
        let graph_hidden_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::A,
            capture_hidden_bytes,
            "Triton BF16 router graph hidden input",
        )?;
        let score_scratch_bytes =
            triton_router_topk_score_scratch_bytes(graph_key, experts, label)?;
        let score_scratch_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::B,
            score_scratch_bytes,
            "Triton BF16 router score scratch",
        )?;
        let cuda_stream = slot.stream_ptr();
        unsafe {
            if hidden_buffer.ptr != graph_hidden_buffer.ptr {
                library
                    .copy_d2d_async(
                        graph_hidden_buffer,
                        hidden_buffer,
                        hidden_bytes,
                        cuda_stream,
                    )
                    .with_context(|| format!("staging {label} Triton graph hidden input"))?;
            }
            if capture_hidden_bytes > hidden_bytes {
                let padding = device_buffer_byte_view(
                    graph_hidden_buffer,
                    hidden_bytes,
                    capture_hidden_bytes - hidden_bytes,
                    "Triton BF16 router padded hidden rows",
                )?;
                library
                    .cuda_zero_bytes_async(padding, padding.bytes, cuda_stream)
                    .with_context(|| format!("zeroing {label} Triton padded hidden rows"))?;
            }
        }
        let signature = triton_router_topk_graph_signature(
            capture_rows,
            hidden_dim,
            experts,
            top_k,
            graph_hidden_buffer,
            weight_buffer,
            bias_buffer,
        );
        capture_or_update_sparse_a_triton_router_topk_bf16_graph(
            library,
            slot,
            signature,
            graph_hidden_buffer,
            weight_buffer,
            bias_buffer,
            score_scratch_buffer,
            index_buffer,
            score_buffer,
            weight_buffer_out,
            capture_rows,
            hidden_dim,
            experts,
            top_k,
            label,
        )?;
        Ok(triton_backend)
    } else {
        let signature = router_topk_graph_signature(graph_key, hidden_dim, experts, top_k);
        capture_or_update_sparse_a_router_topk_bf16_graph(
            library,
            slot,
            signature,
            hidden_buffer,
            weight_buffer,
            bias_buffer,
            index_buffer,
            score_buffer,
            weight_buffer_out,
            rows,
            hidden_dim,
            experts,
            top_k,
            label,
        )?;
        Ok(native_backend)
    }
}

fn triton_router_topk_bf16_supported(
    graph_key: &CoordinatorGraphKey,
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
) -> bool {
    coordinator_python_capture_enabled()
        && graph_key.shape == CoordinatorGraphShape::CoordSparseA
        && rows > 0
        && rows <= graph_key.row_bucket.row_capacity
        && hidden_dim == GLM52_HIDDEN_SIZE
        && experts > 0
        && top_k > 0
        && top_k <= experts
}

fn triton_router_topk_score_scratch_bytes(
    graph_key: &CoordinatorGraphKey,
    experts: usize,
    label: &str,
) -> Result<usize> {
    graph_key
        .row_bucket
        .row_capacity
        .checked_mul(experts)
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .with_context(|| format!("{label} Triton score scratch buffer bytes overflow usize"))
}

pub(in crate::commands::real_full) fn cuda_router_topk_bf16_legacy(
    hidden_bf16: &[u8],
    router_weight_bf16: &[u8],
    correction_bias: &[f32],
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
) -> Result<RouterTopKOutput> {
    let library = cuda_native_library()?;
    let hidden_bytes = hidden_bf16.len();
    let weight_bytes = router_weight_bf16.len();
    let bias_bytes = std::mem::size_of_val(correction_bias);
    let output_values = rows
        .checked_mul(top_k)
        .context("CUDA BF16 router top-k output shape overflows usize")?;
    let index_bytes = output_values
        .checked_mul(std::mem::size_of::<u32>())
        .context("CUDA BF16 router top-k index bytes overflow usize")?;
    let score_bytes = output_values
        .checked_mul(std::mem::size_of::<f32>())
        .context("CUDA BF16 router top-k score bytes overflow usize")?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let hidden_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::A,
        hidden_bytes,
        "BF16 router hidden",
    )?;
    let weight_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::B,
        weight_bytes,
        "BF16 router weight",
    )?;
    let bias_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::C,
        bias_bytes,
        "BF16 router correction bias",
    )?;
    let index_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::D,
        index_bytes,
        "BF16 router top-k indices",
    )?;
    let score_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::E,
        score_bytes,
        "BF16 router top-k scores",
    )?;
    let weight_buffer_out = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::F,
        score_bytes,
        "BF16 router top-k weights",
    )?;

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::A,
            hidden_bf16,
            "BF16 router hidden",
        )
        .context("copying BF16 router hidden to device")?;
    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::B,
            router_weight_bf16,
            "BF16 router weight",
        )
        .context("copying BF16 router weight to device")?;
    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::C,
            f32_bytes(correction_bias),
            "BF16 router correction bias",
        )
        .context("copying BF16 router correction bias to device")?;
    library
        .cuda_router_topk_bf16(
            hidden_buffer,
            weight_buffer,
            bias_buffer,
            index_buffer,
            score_buffer,
            weight_buffer_out,
            rows,
            hidden_dim,
            experts,
            top_k,
        )
        .context("executing CUDA BF16 router top-k")?;
    let mut index_out = vec![0_u8; index_bytes];
    let mut score_out = vec![0_u8; score_bytes];
    let mut weight_out = vec![0_u8; score_bytes];
    library
        .copy_d2h(&mut index_out, index_buffer)
        .context("copying BF16 router top-k indices to host")?;
    library
        .copy_d2h(&mut score_out, score_buffer)
        .context("copying BF16 router top-k scores to host")?;
    library
        .copy_d2h(&mut weight_out, weight_buffer_out)
        .context("copying BF16 router top-k weights to host")?;

    Ok(RouterTopKOutput {
        indices: u32_vec_from_bytes(&index_out)?
            .into_iter()
            .map(|value| value as usize)
            .collect(),
        scores: f32_vec_from_bytes(&score_out)?,
        weights: f32_vec_from_bytes(&weight_out)?,
        backend: CUDA_REFERENCE_ROUTER_TOPK_BF16_BACKEND,
    })
}

pub(in crate::commands::real_full) fn cuda_router_topk_bf16_resident_weight(
    router_weight_name: &str,
    hidden_bf16: &[u8],
    router_weight_bf16: &[u8],
    correction_bias: &[f32],
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
) -> Result<RouterTopKOutput> {
    if let Some(graph_key) = coord_sparse_a_graph_key_for_full_hidden_rows(rows, hidden_dim)? {
        return cuda_router_topk_bf16_resident_weight_graph_slot(
            &graph_key,
            router_weight_name,
            hidden_bf16,
            router_weight_bf16,
            correction_bias,
            rows,
            hidden_dim,
            experts,
            top_k,
        );
    }
    cuda_router_topk_bf16_resident_weight_legacy(
        router_weight_name,
        hidden_bf16,
        router_weight_bf16,
        correction_bias,
        rows,
        hidden_dim,
        experts,
        top_k,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_router_topk_bf16_resident_weight_graph_slot(
    graph_key: &CoordinatorGraphKey,
    router_weight_name: &str,
    hidden_bf16: &[u8],
    router_weight_bf16: &[u8],
    correction_bias: &[f32],
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
) -> Result<RouterTopKOutput> {
    let bias_bytes = std::mem::size_of_val(correction_bias);
    let (_, index_bytes, score_bytes) =
        router_topk_output_byte_counts(rows, top_k, "CUDA BF16 resident router top-k graph-slot")?;
    let hidden_graph_bytes = router_topk_graph_hidden_bytes(
        graph_key,
        hidden_dim,
        "CUDA BF16 resident router top-k graph-slot",
    )?;
    let (_, graph_index_bytes, graph_score_bytes) = router_topk_graph_output_byte_counts(
        graph_key,
        top_k,
        "CUDA BF16 resident router top-k graph-slot",
    )?;
    let weight_buffer = resident_weight_buffer_from_registry(
        router_weight_name,
        router_weight_bf16,
        "BF16 resident router weight",
    )?;

    with_coordinator_cuda_graph_slot(graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        let hidden_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::A,
            hidden_graph_bytes,
            "BF16 resident router hidden",
        )?;
        let bias_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::C,
            bias_bytes,
            "BF16 resident router correction bias",
        )?;
        let index_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::D,
            graph_index_bytes,
            "BF16 resident router top-k indices",
        )?;
        let score_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::E,
            graph_score_bytes,
            "BF16 resident router top-k scores",
        )?;
        let weight_buffer_out = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::F,
            graph_score_bytes,
            "BF16 resident router top-k weights",
        )?;

        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::A,
                hidden_bf16,
                "BF16 resident router hidden",
                cuda_stream,
            )
            .context("async copying BF16 resident router hidden to device")?;
        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::C,
                f32_bytes(correction_bias),
                "BF16 resident router correction bias",
                cuda_stream,
            )
            .context("async copying BF16 resident router correction bias to device")?;
        let backend = capture_or_update_router_topk_bf16_graph_for_slot(
            graph_key,
            library,
            slot,
            hidden_buffer,
            weight_buffer,
            bias_buffer,
            index_buffer,
            score_buffer,
            weight_buffer_out,
            rows,
            hidden_dim,
            experts,
            top_k,
            CUDA_REFERENCE_ROUTER_TOPK_BF16_RESIDENT_WEIGHT_BACKEND,
            TRITON_ROUTER_TOPK_BF16_RESIDENT_WEIGHT_BACKEND,
            "BF16 resident router top-k",
        )?;
        let mut index_out = vec![0_u8; index_bytes];
        let mut score_out = vec![0_u8; score_bytes];
        let mut weight_out = vec![0_u8; score_bytes];
        unsafe {
            library
                .copy_d2h_async(&mut index_out, index_buffer, cuda_stream)
                .context("async copying BF16 resident router top-k indices to host")?;
            library
                .copy_d2h_async(&mut score_out, score_buffer, cuda_stream)
                .context("async copying BF16 resident router top-k scores to host")?;
            library
                .copy_d2h_async(&mut weight_out, weight_buffer_out, cuda_stream)
                .context("async copying BF16 resident router top-k weights to host")?;
            library
                .cuda_stream_synchronize(cuda_stream)
                .context("synchronizing BF16 resident router top-k graph slot stream")?;
        }

        Ok(RouterTopKOutput {
            indices: u32_vec_from_bytes(&index_out)?
                .into_iter()
                .map(|value| value as usize)
                .collect(),
            scores: f32_vec_from_bytes(&score_out)?,
            weights: f32_vec_from_bytes(&weight_out)?,
            backend,
        })
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_router_topk_bf16_resident_weight_legacy(
    router_weight_name: &str,
    hidden_bf16: &[u8],
    router_weight_bf16: &[u8],
    correction_bias: &[f32],
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
) -> Result<RouterTopKOutput> {
    let library = cuda_native_library()?;
    let hidden_bytes = hidden_bf16.len();
    let bias_bytes = std::mem::size_of_val(correction_bias);
    let output_values = rows
        .checked_mul(top_k)
        .context("CUDA BF16 resident router top-k output shape overflows usize")?;
    let index_bytes = output_values
        .checked_mul(std::mem::size_of::<u32>())
        .context("CUDA BF16 resident router top-k index bytes overflow usize")?;
    let score_bytes = output_values
        .checked_mul(std::mem::size_of::<f32>())
        .context("CUDA BF16 resident router top-k score bytes overflow usize")?;
    let weight_buffer = resident_weight_buffer_from_registry(
        router_weight_name,
        router_weight_bf16,
        "BF16 resident router weight",
    )?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let hidden_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::A,
        hidden_bytes,
        "BF16 resident router hidden",
    )?;
    let bias_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::C,
        bias_bytes,
        "BF16 resident router correction bias",
    )?;
    let index_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::D,
        index_bytes,
        "BF16 resident router top-k indices",
    )?;
    let score_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::E,
        score_bytes,
        "BF16 resident router top-k scores",
    )?;
    let weight_buffer_out = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::F,
        score_bytes,
        "BF16 resident router top-k weights",
    )?;

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::A,
            hidden_bf16,
            "BF16 resident router hidden",
        )
        .context("copying BF16 resident router hidden to device")?;
    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::C,
            f32_bytes(correction_bias),
            "BF16 resident router correction bias",
        )
        .context("copying BF16 resident router correction bias to device")?;
    library
        .cuda_router_topk_bf16(
            hidden_buffer,
            weight_buffer,
            bias_buffer,
            index_buffer,
            score_buffer,
            weight_buffer_out,
            rows,
            hidden_dim,
            experts,
            top_k,
        )
        .context("executing CUDA BF16 resident router top-k")?;
    let mut index_out = vec![0_u8; index_bytes];
    let mut score_out = vec![0_u8; score_bytes];
    let mut weight_out = vec![0_u8; score_bytes];
    library
        .copy_d2h(&mut index_out, index_buffer)
        .context("copying BF16 resident router top-k indices to host")?;
    library
        .copy_d2h(&mut score_out, score_buffer)
        .context("copying BF16 resident router top-k scores to host")?;
    library
        .copy_d2h(&mut weight_out, weight_buffer_out)
        .context("copying BF16 resident router top-k weights to host")?;

    Ok(RouterTopKOutput {
        indices: u32_vec_from_bytes(&index_out)?
            .into_iter()
            .map(|value| value as usize)
            .collect(),
        scores: f32_vec_from_bytes(&score_out)?,
        weights: f32_vec_from_bytes(&weight_out)?,
        backend: CUDA_REFERENCE_ROUTER_TOPK_BF16_RESIDENT_WEIGHT_BACKEND,
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_router_topk_bf16_preloaded_resident_weight(
    router_weight_name: &str,
    hidden_bf16: &[u8],
    correction_bias: &[f32],
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
    weight_bytes: usize,
) -> Result<RouterTopKOutput> {
    if let Some(graph_key) = coord_sparse_a_graph_key_for_full_hidden_rows(rows, hidden_dim)? {
        return cuda_router_topk_bf16_preloaded_resident_weight_graph_slot(
            &graph_key,
            router_weight_name,
            hidden_bf16,
            correction_bias,
            rows,
            hidden_dim,
            experts,
            top_k,
            weight_bytes,
        );
    }
    cuda_router_topk_bf16_preloaded_resident_weight_legacy(
        router_weight_name,
        hidden_bf16,
        correction_bias,
        rows,
        hidden_dim,
        experts,
        top_k,
        weight_bytes,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_router_topk_bf16_preloaded_resident_weight_graph_slot(
    graph_key: &CoordinatorGraphKey,
    router_weight_name: &str,
    hidden_bf16: &[u8],
    correction_bias: &[f32],
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
    weight_bytes: usize,
) -> Result<RouterTopKOutput> {
    let bias_bytes = std::mem::size_of_val(correction_bias);
    let (_, index_bytes, score_bytes) = router_topk_output_byte_counts(
        rows,
        top_k,
        "CUDA BF16 preloaded resident router top-k graph-slot",
    )?;
    let hidden_graph_bytes = router_topk_graph_hidden_bytes(
        graph_key,
        hidden_dim,
        "CUDA BF16 preloaded resident router top-k graph-slot",
    )?;
    let (_, graph_index_bytes, graph_score_bytes) = router_topk_graph_output_byte_counts(
        graph_key,
        top_k,
        "CUDA BF16 preloaded resident router top-k graph-slot",
    )?;
    let weight_buffer = preloaded_resident_weight_device_buffer(router_weight_name, weight_bytes)?;

    with_coordinator_cuda_graph_slot(graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        let hidden_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::A,
            hidden_graph_bytes,
            "BF16 preloaded resident router hidden",
        )?;
        let bias_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::C,
            bias_bytes,
            "BF16 preloaded resident router correction bias",
        )?;
        let index_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::D,
            graph_index_bytes,
            "BF16 preloaded resident router top-k indices",
        )?;
        let score_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::E,
            graph_score_bytes,
            "BF16 preloaded resident router top-k scores",
        )?;
        let weight_buffer_out = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::F,
            graph_score_bytes,
            "BF16 preloaded resident router top-k weights",
        )?;

        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::A,
                hidden_bf16,
                "BF16 preloaded resident router hidden",
                cuda_stream,
            )
            .context("async copying BF16 preloaded resident router hidden to device")?;
        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::C,
                f32_bytes(correction_bias),
                "BF16 preloaded resident router correction bias",
                cuda_stream,
            )
            .context("async copying BF16 preloaded resident router correction bias to device")?;
        let backend = capture_or_update_router_topk_bf16_graph_for_slot(
            graph_key,
            library,
            slot,
            hidden_buffer,
            weight_buffer,
            bias_buffer,
            index_buffer,
            score_buffer,
            weight_buffer_out,
            rows,
            hidden_dim,
            experts,
            top_k,
            CUDA_REFERENCE_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
            TRITON_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
            "BF16 preloaded resident router top-k",
        )?;
        let mut index_out = vec![0_u8; index_bytes];
        let mut score_out = vec![0_u8; score_bytes];
        let mut weight_out = vec![0_u8; score_bytes];
        unsafe {
            library
                .copy_d2h_async(&mut index_out, index_buffer, cuda_stream)
                .context("async copying BF16 preloaded resident router top-k indices to host")?;
            library
                .copy_d2h_async(&mut score_out, score_buffer, cuda_stream)
                .context("async copying BF16 preloaded resident router top-k scores to host")?;
            library
                .copy_d2h_async(&mut weight_out, weight_buffer_out, cuda_stream)
                .context("async copying BF16 preloaded resident router top-k weights to host")?;
            library
                .cuda_stream_synchronize(cuda_stream)
                .context("synchronizing BF16 preloaded resident router top-k graph slot stream")?;
        }

        Ok(RouterTopKOutput {
            indices: u32_vec_from_bytes(&index_out)?
                .into_iter()
                .map(|value| value as usize)
                .collect(),
            scores: f32_vec_from_bytes(&score_out)?,
            weights: f32_vec_from_bytes(&weight_out)?,
            backend,
        })
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_router_topk_bf16_preloaded_resident_weight_legacy(
    router_weight_name: &str,
    hidden_bf16: &[u8],
    correction_bias: &[f32],
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
    weight_bytes: usize,
) -> Result<RouterTopKOutput> {
    let library = cuda_native_library()?;
    let hidden_bytes = hidden_bf16.len();
    let bias_bytes = std::mem::size_of_val(correction_bias);
    let output_values = rows
        .checked_mul(top_k)
        .context("CUDA BF16 preloaded resident router top-k output shape overflows usize")?;
    let index_bytes = output_values
        .checked_mul(std::mem::size_of::<u32>())
        .context("CUDA BF16 preloaded resident router top-k index bytes overflow usize")?;
    let score_bytes = output_values
        .checked_mul(std::mem::size_of::<f32>())
        .context("CUDA BF16 preloaded resident router top-k score bytes overflow usize")?;
    let weight_buffer = preloaded_resident_weight_device_buffer(router_weight_name, weight_bytes)?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let hidden_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::A,
        hidden_bytes,
        "BF16 preloaded resident router hidden",
    )?;
    let bias_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::C,
        bias_bytes,
        "BF16 preloaded resident router correction bias",
    )?;
    let index_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::D,
        index_bytes,
        "BF16 preloaded resident router top-k indices",
    )?;
    let score_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::E,
        score_bytes,
        "BF16 preloaded resident router top-k scores",
    )?;
    let weight_buffer_out = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::F,
        score_bytes,
        "BF16 preloaded resident router top-k weights",
    )?;

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::A,
            hidden_bf16,
            "BF16 preloaded resident router hidden",
        )
        .context("copying BF16 preloaded resident router hidden to device")?;
    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::C,
            f32_bytes(correction_bias),
            "BF16 preloaded resident router correction bias",
        )
        .context("copying BF16 preloaded resident router correction bias to device")?;
    library
        .cuda_router_topk_bf16(
            hidden_buffer,
            weight_buffer,
            bias_buffer,
            index_buffer,
            score_buffer,
            weight_buffer_out,
            rows,
            hidden_dim,
            experts,
            top_k,
        )
        .context("executing CUDA BF16 preloaded resident router top-k")?;
    let mut index_out = vec![0_u8; index_bytes];
    let mut score_out = vec![0_u8; score_bytes];
    let mut weight_out = vec![0_u8; score_bytes];
    library
        .copy_d2h(&mut index_out, index_buffer)
        .context("copying BF16 preloaded resident router top-k indices to host")?;
    library
        .copy_d2h(&mut score_out, score_buffer)
        .context("copying BF16 preloaded resident router top-k scores to host")?;
    library
        .copy_d2h(&mut weight_out, weight_buffer_out)
        .context("copying BF16 preloaded resident router top-k weights to host")?;

    Ok(RouterTopKOutput {
        indices: u32_vec_from_bytes(&index_out)?
            .into_iter()
            .map(|value| value as usize)
            .collect(),
        scores: f32_vec_from_bytes(&score_out)?,
        weights: f32_vec_from_bytes(&weight_out)?,
        backend: CUDA_REFERENCE_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_BACKEND,
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_router_topk_bf16_preloaded_resident_weight_bias(
    router_weight_name: &str,
    correction_bias_name: &str,
    hidden_bf16: &[u8],
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
    weight_bytes: usize,
    bias_bytes: usize,
) -> Result<RouterTopKOutput> {
    if let Some(graph_key) = coord_sparse_a_graph_key_for_full_hidden_rows(rows, hidden_dim)? {
        return cuda_router_topk_bf16_preloaded_resident_weight_bias_graph_slot(
            &graph_key,
            router_weight_name,
            correction_bias_name,
            hidden_bf16,
            rows,
            hidden_dim,
            experts,
            top_k,
            weight_bytes,
            bias_bytes,
        );
    }
    cuda_router_topk_bf16_preloaded_resident_weight_bias_legacy(
        router_weight_name,
        correction_bias_name,
        hidden_bf16,
        rows,
        hidden_dim,
        experts,
        top_k,
        weight_bytes,
        bias_bytes,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_router_topk_bf16_preloaded_resident_weight_bias_graph_slot(
    graph_key: &CoordinatorGraphKey,
    router_weight_name: &str,
    correction_bias_name: &str,
    hidden_bf16: &[u8],
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
    weight_bytes: usize,
    bias_bytes: usize,
) -> Result<RouterTopKOutput> {
    let (_, index_bytes, score_bytes) = router_topk_output_byte_counts(
        rows,
        top_k,
        "CUDA BF16 preloaded resident router top-k graph-slot weight+bias",
    )?;
    let hidden_graph_bytes = router_topk_graph_hidden_bytes(
        graph_key,
        hidden_dim,
        "CUDA BF16 preloaded resident router top-k graph-slot weight+bias",
    )?;
    let (_, graph_index_bytes, graph_score_bytes) = router_topk_graph_output_byte_counts(
        graph_key,
        top_k,
        "CUDA BF16 preloaded resident router top-k graph-slot weight+bias",
    )?;
    let weight_buffer = preloaded_resident_weight_device_buffer(router_weight_name, weight_bytes)?;
    let bias_buffer = preloaded_resident_weight_device_buffer(correction_bias_name, bias_bytes)?;

    with_coordinator_cuda_graph_slot(graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        let hidden_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::A,
            hidden_graph_bytes,
            "BF16 preloaded resident router weight+bias hidden",
        )?;
        let index_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::D,
            graph_index_bytes,
            "BF16 preloaded resident router weight+bias top-k indices",
        )?;
        let score_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::E,
            graph_score_bytes,
            "BF16 preloaded resident router weight+bias top-k scores",
        )?;
        let weight_buffer_out = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::F,
            graph_score_bytes,
            "BF16 preloaded resident router weight+bias top-k weights",
        )?;

        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::A,
                hidden_bf16,
                "BF16 preloaded resident router weight+bias hidden",
                cuda_stream,
            )
            .context("async copying BF16 preloaded resident router weight+bias hidden to device")?;
        let backend = capture_or_update_router_topk_bf16_graph_for_slot(
            graph_key,
            library,
            slot,
            hidden_buffer,
            weight_buffer,
            bias_buffer,
            index_buffer,
            score_buffer,
            weight_buffer_out,
            rows,
            hidden_dim,
            experts,
            top_k,
            CUDA_REFERENCE_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_BIAS_BACKEND,
            TRITON_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_BIAS_BACKEND,
            "BF16 preloaded resident router top-k weight+bias",
        )?;
        let mut index_out = vec![0_u8; index_bytes];
        let mut score_out = vec![0_u8; score_bytes];
        let mut weight_out = vec![0_u8; score_bytes];
        unsafe {
            library
                .copy_d2h_async(&mut index_out, index_buffer, cuda_stream)
                .context(
                    "async copying BF16 preloaded resident router top-k weight+bias indices to host",
                )?;
            library
                .copy_d2h_async(&mut score_out, score_buffer, cuda_stream)
                .context(
                    "async copying BF16 preloaded resident router top-k weight+bias scores to host",
                )?;
            library
                .copy_d2h_async(&mut weight_out, weight_buffer_out, cuda_stream)
                .context(
                    "async copying BF16 preloaded resident router top-k weight+bias weights to host",
                )?;
            library
                .cuda_stream_synchronize(cuda_stream)
                .context("synchronizing BF16 preloaded resident router top-k graph slot stream")?;
        }

        Ok(RouterTopKOutput {
            indices: u32_vec_from_bytes(&index_out)?
                .into_iter()
                .map(|value| value as usize)
                .collect(),
            scores: f32_vec_from_bytes(&score_out)?,
            weights: f32_vec_from_bytes(&weight_out)?,
            backend,
        })
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_router_topk_bf16_preloaded_resident_weight_bias_legacy(
    router_weight_name: &str,
    correction_bias_name: &str,
    hidden_bf16: &[u8],
    rows: usize,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
    weight_bytes: usize,
    bias_bytes: usize,
) -> Result<RouterTopKOutput> {
    let library = cuda_native_library()?;
    let hidden_bytes = hidden_bf16.len();
    let output_values = rows.checked_mul(top_k).context(
        "CUDA BF16 preloaded resident router top-k weight+bias output shape overflows usize",
    )?;
    let index_bytes = output_values
        .checked_mul(std::mem::size_of::<u32>())
        .context(
            "CUDA BF16 preloaded resident router top-k weight+bias index bytes overflow usize",
        )?;
    let score_bytes = output_values
        .checked_mul(std::mem::size_of::<f32>())
        .context(
            "CUDA BF16 preloaded resident router top-k weight+bias score bytes overflow usize",
        )?;
    let weight_buffer = preloaded_resident_weight_device_buffer(router_weight_name, weight_bytes)?;
    let bias_buffer = preloaded_resident_weight_device_buffer(correction_bias_name, bias_bytes)?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let hidden_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::A,
        hidden_bytes,
        "BF16 preloaded resident router weight+bias hidden",
    )?;
    let index_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::D,
        index_bytes,
        "BF16 preloaded resident router weight+bias top-k indices",
    )?;
    let score_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::E,
        score_bytes,
        "BF16 preloaded resident router weight+bias top-k scores",
    )?;
    let weight_buffer_out = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::F,
        score_bytes,
        "BF16 preloaded resident router weight+bias top-k weights",
    )?;

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::A,
            hidden_bf16,
            "BF16 preloaded resident router weight+bias hidden",
        )
        .context("copying BF16 preloaded resident router weight+bias hidden to device")?;
    library
        .cuda_router_topk_bf16(
            hidden_buffer,
            weight_buffer,
            bias_buffer,
            index_buffer,
            score_buffer,
            weight_buffer_out,
            rows,
            hidden_dim,
            experts,
            top_k,
        )
        .context("executing CUDA BF16 preloaded resident router top-k weight+bias")?;
    let mut index_out = vec![0_u8; index_bytes];
    let mut score_out = vec![0_u8; score_bytes];
    let mut weight_out = vec![0_u8; score_bytes];
    library
        .copy_d2h(&mut index_out, index_buffer)
        .context("copying BF16 preloaded resident router top-k weight+bias indices to host")?;
    library
        .copy_d2h(&mut score_out, score_buffer)
        .context("copying BF16 preloaded resident router top-k weight+bias scores to host")?;
    library
        .copy_d2h(&mut weight_out, weight_buffer_out)
        .context("copying BF16 preloaded resident router top-k weight+bias weights to host")?;

    Ok(RouterTopKOutput {
        indices: u32_vec_from_bytes(&index_out)?
            .into_iter()
            .map(|value| value as usize)
            .collect(),
        scores: f32_vec_from_bytes(&score_out)?,
        weights: f32_vec_from_bytes(&weight_out)?,
        backend: CUDA_REFERENCE_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_BIAS_BACKEND,
    })
}

pub(in crate::commands::real_full) fn router_topk_output_byte_counts(
    rows: usize,
    top_k: usize,
    context: &str,
) -> Result<(usize, usize, usize)> {
    let output_values = rows
        .checked_mul(top_k)
        .with_context(|| format!("{context} output shape overflows usize"))?;
    let index_bytes = output_values
        .checked_mul(std::mem::size_of::<u32>())
        .with_context(|| format!("{context} index bytes overflow usize"))?;
    let score_bytes = output_values
        .checked_mul(std::mem::size_of::<f32>())
        .with_context(|| format!("{context} score bytes overflow usize"))?;
    Ok((output_values, index_bytes, score_bytes))
}

pub(in crate::commands::real_full) fn router_topk_graph_output_byte_counts(
    graph_key: &CoordinatorGraphKey,
    top_k: usize,
    context: &str,
) -> Result<(usize, usize, usize)> {
    router_topk_output_byte_counts(graph_key.row_bucket.row_capacity, top_k, context)
}

pub(in crate::commands::real_full) fn router_topk_graph_hidden_bytes(
    graph_key: &CoordinatorGraphKey,
    hidden_dim: usize,
    context: &str,
) -> Result<usize> {
    graph_key
        .row_bucket
        .row_capacity
        .checked_mul(hidden_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .with_context(|| format!("{context} hidden graph buffer bytes overflow usize"))
}

pub(in crate::commands::real_full) fn router_topk_graph_signature(
    graph_key: &CoordinatorGraphKey,
    hidden_dim: usize,
    experts: usize,
    top_k: usize,
) -> CoordinatorCudaGraphSignature {
    CoordinatorCudaGraphSignature::router_topk_bf16(
        graph_key.row_bucket.row_capacity,
        hidden_dim,
        experts,
        top_k,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_router_topk_bf16_resident_weight_device_input(
    router_weight_name: &str,
    hidden: &DeviceBf16Output,
    router_weight_bf16: &[u8],
    correction_bias: &[f32],
    experts: usize,
    top_k: usize,
) -> Result<RouterTopKOutput> {
    if let Some(graph_key) =
        coord_sparse_a_graph_key_for_full_hidden_rows(hidden.rows, hidden.values_per_row)?
    {
        return cuda_router_topk_bf16_resident_weight_device_input_graph_slot(
            &graph_key,
            router_weight_name,
            hidden,
            router_weight_bf16,
            correction_bias,
            experts,
            top_k,
        );
    }
    cuda_router_topk_bf16_resident_weight_device_input_legacy(
        router_weight_name,
        hidden,
        router_weight_bf16,
        correction_bias,
        experts,
        top_k,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_router_topk_bf16_resident_weight_device_input_graph_slot(
    graph_key: &CoordinatorGraphKey,
    router_weight_name: &str,
    hidden: &DeviceBf16Output,
    router_weight_bf16: &[u8],
    correction_bias: &[f32],
    experts: usize,
    top_k: usize,
) -> Result<RouterTopKOutput> {
    let bias_bytes = std::mem::size_of_val(correction_bias);
    let (_, index_bytes, score_bytes) = router_topk_output_byte_counts(
        hidden.rows,
        top_k,
        "CUDA BF16 resident router top-k device-input graph-slot",
    )?;
    let (_, graph_index_bytes, graph_score_bytes) = router_topk_graph_output_byte_counts(
        graph_key,
        top_k,
        "CUDA BF16 resident router top-k device-input graph-slot",
    )?;
    let weight_buffer = resident_weight_buffer_from_registry(
        router_weight_name,
        router_weight_bf16,
        "BF16 resident router device-input weight",
    )?;

    with_coordinator_cuda_graph_slot(graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        hidden
            .wait_ready_on_stream(cuda_stream)
            .context("waiting for BF16 resident router device input")?;
        let hidden_buffer = hidden.buffer();
        let bias_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::C,
            bias_bytes,
            "BF16 resident router device-input correction bias",
        )?;
        let index_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::D,
            graph_index_bytes,
            "BF16 resident router device-input top-k indices",
        )?;
        let score_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::E,
            graph_score_bytes,
            "BF16 resident router device-input top-k scores",
        )?;
        let weight_buffer_out = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::F,
            graph_score_bytes,
            "BF16 resident router device-input top-k weights",
        )?;

        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::C,
                f32_bytes(correction_bias),
                "BF16 resident router device-input correction bias",
                cuda_stream,
            )
            .context("async copying BF16 resident router device-input correction bias to device")?;
        let backend = capture_or_update_router_topk_bf16_graph_for_slot(
            graph_key,
            library,
            slot,
            hidden_buffer,
            weight_buffer,
            bias_buffer,
            index_buffer,
            score_buffer,
            weight_buffer_out,
            hidden.rows,
            hidden.values_per_row,
            experts,
            top_k,
            CUDA_REFERENCE_ROUTER_TOPK_BF16_RESIDENT_WEIGHT_DEVICE_INPUT_BACKEND,
            TRITON_ROUTER_TOPK_BF16_RESIDENT_WEIGHT_DEVICE_INPUT_BACKEND,
            "BF16 resident router top-k device-input",
        )?;
        let mut index_out = vec![0_u8; index_bytes];
        let mut score_out = vec![0_u8; score_bytes];
        let mut weight_out = vec![0_u8; score_bytes];
        unsafe {
            library
                .copy_d2h_async(&mut index_out, index_buffer, cuda_stream)
                .context("async copying BF16 resident router device-input top-k indices to host")?;
            library
                .copy_d2h_async(&mut score_out, score_buffer, cuda_stream)
                .context("async copying BF16 resident router device-input top-k scores to host")?;
            library
                .copy_d2h_async(&mut weight_out, weight_buffer_out, cuda_stream)
                .context("async copying BF16 resident router device-input top-k weights to host")?;
            library
                .cuda_stream_synchronize(cuda_stream)
                .context("synchronizing BF16 resident router device-input graph slot stream")?;
        }

        Ok(RouterTopKOutput {
            indices: u32_vec_from_bytes(&index_out)?
                .into_iter()
                .map(|value| value as usize)
                .collect(),
            scores: f32_vec_from_bytes(&score_out)?,
            weights: f32_vec_from_bytes(&weight_out)?,
            backend,
        })
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_router_topk_bf16_resident_weight_device_input_legacy(
    router_weight_name: &str,
    hidden: &DeviceBf16Output,
    router_weight_bf16: &[u8],
    correction_bias: &[f32],
    experts: usize,
    top_k: usize,
) -> Result<RouterTopKOutput> {
    let library = cuda_native_library()?;
    let bias_bytes = std::mem::size_of_val(correction_bias);
    let (_, index_bytes, score_bytes) = router_topk_output_byte_counts(
        hidden.rows,
        top_k,
        "CUDA BF16 resident router top-k device-input",
    )?;
    let weight_buffer = resident_weight_buffer_from_registry(
        router_weight_name,
        router_weight_bf16,
        "BF16 resident router device-input weight",
    )?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let bias_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::C,
        bias_bytes,
        "BF16 resident router device-input correction bias",
    )?;
    let index_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::D,
        index_bytes,
        "BF16 resident router device-input top-k indices",
    )?;
    let score_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::E,
        score_bytes,
        "BF16 resident router device-input top-k scores",
    )?;
    let weight_buffer_out = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::F,
        score_bytes,
        "BF16 resident router device-input top-k weights",
    )?;

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::C,
            f32_bytes(correction_bias),
            "BF16 resident router device-input correction bias",
        )
        .context("copying BF16 resident router device-input correction bias to device")?;
    library
        .cuda_router_topk_bf16(
            hidden.buffer(),
            weight_buffer,
            bias_buffer,
            index_buffer,
            score_buffer,
            weight_buffer_out,
            hidden.rows,
            hidden.values_per_row,
            experts,
            top_k,
        )
        .context("executing CUDA BF16 resident router top-k device-input")?;
    let mut index_out = vec![0_u8; index_bytes];
    let mut score_out = vec![0_u8; score_bytes];
    let mut weight_out = vec![0_u8; score_bytes];
    library
        .copy_d2h(&mut index_out, index_buffer)
        .context("copying BF16 resident router device-input top-k indices to host")?;
    library
        .copy_d2h(&mut score_out, score_buffer)
        .context("copying BF16 resident router device-input top-k scores to host")?;
    library
        .copy_d2h(&mut weight_out, weight_buffer_out)
        .context("copying BF16 resident router device-input top-k weights to host")?;

    Ok(RouterTopKOutput {
        indices: u32_vec_from_bytes(&index_out)?
            .into_iter()
            .map(|value| value as usize)
            .collect(),
        scores: f32_vec_from_bytes(&score_out)?,
        weights: f32_vec_from_bytes(&weight_out)?,
        backend: CUDA_REFERENCE_ROUTER_TOPK_BF16_RESIDENT_WEIGHT_DEVICE_INPUT_BACKEND,
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_router_topk_bf16_preloaded_resident_weight_device_input(
    router_weight_name: &str,
    hidden: &DeviceBf16Output,
    correction_bias: &[f32],
    experts: usize,
    top_k: usize,
    weight_bytes: usize,
) -> Result<RouterTopKOutput> {
    if let Some(graph_key) =
        coord_sparse_a_graph_key_for_full_hidden_rows(hidden.rows, hidden.values_per_row)?
    {
        return cuda_router_topk_bf16_preloaded_resident_weight_device_input_graph_slot(
            &graph_key,
            router_weight_name,
            hidden,
            correction_bias,
            experts,
            top_k,
            weight_bytes,
        );
    }
    cuda_router_topk_bf16_preloaded_resident_weight_device_input_legacy(
        router_weight_name,
        hidden,
        correction_bias,
        experts,
        top_k,
        weight_bytes,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_router_topk_bf16_preloaded_resident_weight_device_input_graph_slot(
    graph_key: &CoordinatorGraphKey,
    router_weight_name: &str,
    hidden: &DeviceBf16Output,
    correction_bias: &[f32],
    experts: usize,
    top_k: usize,
    weight_bytes: usize,
) -> Result<RouterTopKOutput> {
    let bias_bytes = std::mem::size_of_val(correction_bias);
    let (_, index_bytes, score_bytes) = router_topk_output_byte_counts(
        hidden.rows,
        top_k,
        "CUDA BF16 preloaded resident router top-k device-input graph-slot",
    )?;
    let (_, graph_index_bytes, graph_score_bytes) = router_topk_graph_output_byte_counts(
        graph_key,
        top_k,
        "CUDA BF16 preloaded resident router top-k device-input graph-slot",
    )?;
    let weight_buffer = preloaded_resident_weight_device_buffer(router_weight_name, weight_bytes)?;

    with_coordinator_cuda_graph_slot(graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        hidden
            .wait_ready_on_stream(cuda_stream)
            .context("waiting for BF16 preloaded resident router device input")?;
        let hidden_buffer = hidden.buffer();
        let bias_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::C,
            bias_bytes,
            "BF16 preloaded resident router device-input correction bias",
        )?;
        let index_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::D,
            graph_index_bytes,
            "BF16 preloaded resident router device-input top-k indices",
        )?;
        let score_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::E,
            graph_score_bytes,
            "BF16 preloaded resident router device-input top-k scores",
        )?;
        let weight_buffer_out = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::F,
            graph_score_bytes,
            "BF16 preloaded resident router device-input top-k weights",
        )?;

        slot.workspace
            .copy_h2d_to_slot_async(
                library,
                CoordinatorCudaScratchSlot::C,
                f32_bytes(correction_bias),
                "BF16 preloaded resident router device-input correction bias",
                cuda_stream,
            )
            .context(
                "async copying BF16 preloaded resident router device-input correction bias to device",
            )?;
        let backend = capture_or_update_router_topk_bf16_graph_for_slot(
            graph_key,
            library,
            slot,
            hidden_buffer,
            weight_buffer,
            bias_buffer,
            index_buffer,
            score_buffer,
            weight_buffer_out,
            hidden.rows,
            hidden.values_per_row,
            experts,
            top_k,
            CUDA_REFERENCE_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_DEVICE_INPUT_BACKEND,
            TRITON_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_DEVICE_INPUT_BACKEND,
            "BF16 preloaded resident router top-k device-input",
        )?;
        let mut index_out = vec![0_u8; index_bytes];
        let mut score_out = vec![0_u8; score_bytes];
        let mut weight_out = vec![0_u8; score_bytes];
        unsafe {
            library
                .copy_d2h_async(&mut index_out, index_buffer, cuda_stream)
                .context(
                    "async copying BF16 preloaded resident router device-input top-k indices to host",
                )?;
            library
                .copy_d2h_async(&mut score_out, score_buffer, cuda_stream)
                .context(
                    "async copying BF16 preloaded resident router device-input top-k scores to host",
                )?;
            library
                .copy_d2h_async(&mut weight_out, weight_buffer_out, cuda_stream)
                .context(
                    "async copying BF16 preloaded resident router device-input top-k weights to host",
                )?;
            library.cuda_stream_synchronize(cuda_stream).context(
                "synchronizing BF16 preloaded resident router device-input graph slot stream",
            )?;
        }

        Ok(RouterTopKOutput {
            indices: u32_vec_from_bytes(&index_out)?
                .into_iter()
                .map(|value| value as usize)
                .collect(),
            scores: f32_vec_from_bytes(&score_out)?,
            weights: f32_vec_from_bytes(&weight_out)?,
            backend,
        })
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_router_topk_bf16_preloaded_resident_weight_device_input_legacy(
    router_weight_name: &str,
    hidden: &DeviceBf16Output,
    correction_bias: &[f32],
    experts: usize,
    top_k: usize,
    weight_bytes: usize,
) -> Result<RouterTopKOutput> {
    let library = cuda_native_library()?;
    let bias_bytes = std::mem::size_of_val(correction_bias);
    let (_, index_bytes, score_bytes) = router_topk_output_byte_counts(
        hidden.rows,
        top_k,
        "CUDA BF16 preloaded resident router top-k device-input",
    )?;
    let weight_buffer = preloaded_resident_weight_device_buffer(router_weight_name, weight_bytes)?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let bias_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::C,
        bias_bytes,
        "BF16 preloaded resident router device-input correction bias",
    )?;
    let index_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::D,
        index_bytes,
        "BF16 preloaded resident router device-input top-k indices",
    )?;
    let score_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::E,
        score_bytes,
        "BF16 preloaded resident router device-input top-k scores",
    )?;
    let weight_buffer_out = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::F,
        score_bytes,
        "BF16 preloaded resident router device-input top-k weights",
    )?;

    workspace
        .copy_h2d_to_slot(
            library,
            CoordinatorCudaScratchSlot::C,
            f32_bytes(correction_bias),
            "BF16 preloaded resident router device-input correction bias",
        )
        .context("copying BF16 preloaded resident router device-input correction bias to device")?;
    library
        .cuda_router_topk_bf16(
            hidden.buffer(),
            weight_buffer,
            bias_buffer,
            index_buffer,
            score_buffer,
            weight_buffer_out,
            hidden.rows,
            hidden.values_per_row,
            experts,
            top_k,
        )
        .context("executing CUDA BF16 preloaded resident router top-k device-input")?;
    let mut index_out = vec![0_u8; index_bytes];
    let mut score_out = vec![0_u8; score_bytes];
    let mut weight_out = vec![0_u8; score_bytes];
    library
        .copy_d2h(&mut index_out, index_buffer)
        .context("copying BF16 preloaded resident router device-input top-k indices to host")?;
    library
        .copy_d2h(&mut score_out, score_buffer)
        .context("copying BF16 preloaded resident router device-input top-k scores to host")?;
    library
        .copy_d2h(&mut weight_out, weight_buffer_out)
        .context("copying BF16 preloaded resident router device-input top-k weights to host")?;

    Ok(RouterTopKOutput {
        indices: u32_vec_from_bytes(&index_out)?
            .into_iter()
            .map(|value| value as usize)
            .collect(),
        scores: f32_vec_from_bytes(&score_out)?,
        weights: f32_vec_from_bytes(&weight_out)?,
        backend: CUDA_REFERENCE_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_DEVICE_INPUT_BACKEND,
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_router_topk_bf16_preloaded_resident_weight_bias_device_input(
    router_weight_name: &str,
    correction_bias_name: &str,
    hidden: &DeviceBf16Output,
    experts: usize,
    top_k: usize,
    weight_bytes: usize,
    bias_bytes: usize,
) -> Result<RouterTopKOutput> {
    if let Some(graph_key) =
        coord_sparse_a_graph_key_for_full_hidden_rows(hidden.rows, hidden.values_per_row)?
    {
        return cuda_router_topk_bf16_preloaded_resident_weight_bias_device_input_graph_slot(
            &graph_key,
            router_weight_name,
            correction_bias_name,
            hidden,
            experts,
            top_k,
            weight_bytes,
            bias_bytes,
        );
    }
    cuda_router_topk_bf16_preloaded_resident_weight_bias_device_input_legacy(
        router_weight_name,
        correction_bias_name,
        hidden,
        experts,
        top_k,
        weight_bytes,
        bias_bytes,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_router_topk_bf16_preloaded_resident_weight_bias_device_input_graph_slot(
    graph_key: &CoordinatorGraphKey,
    router_weight_name: &str,
    correction_bias_name: &str,
    hidden: &DeviceBf16Output,
    experts: usize,
    top_k: usize,
    weight_bytes: usize,
    bias_bytes: usize,
) -> Result<RouterTopKOutput> {
    let (_, index_bytes, score_bytes) = router_topk_output_byte_counts(
        hidden.rows,
        top_k,
        "CUDA BF16 preloaded resident router top-k weight+bias device-input graph-slot",
    )?;
    let (_, graph_index_bytes, graph_score_bytes) = router_topk_graph_output_byte_counts(
        graph_key,
        top_k,
        "CUDA BF16 preloaded resident router top-k weight+bias device-input graph-slot",
    )?;
    let graph_output_bytes = graph_score_bytes
        .checked_mul(2)
        .and_then(|score_and_weight_bytes| graph_index_bytes.checked_add(score_and_weight_bytes))
        .context(
            "CUDA BF16 preloaded resident router top-k packed graph output bytes overflow usize",
        )?;
    let graph_score_offset = graph_index_bytes;
    let graph_weight_offset = graph_index_bytes
        .checked_add(graph_score_bytes)
        .context("CUDA BF16 preloaded resident router top-k packed weight offset overflow usize")?;
    let weight_buffer = preloaded_resident_weight_device_buffer(router_weight_name, weight_bytes)?;
    let bias_buffer = preloaded_resident_weight_device_buffer(correction_bias_name, bias_bytes)?;

    with_coordinator_cuda_graph_slot(graph_key, |library, slot| {
        let cuda_stream = slot.stream_ptr();
        hidden
            .wait_ready_on_stream(cuda_stream)
            .context("waiting for BF16 preloaded resident router weight+bias device input")?;
        let hidden_buffer = hidden.buffer();
        let packed_output_buffer = slot.buffer(
            library,
            CoordinatorCudaScratchSlot::D,
            graph_output_bytes,
            "BF16 preloaded resident router weight+bias device-input packed top-k output",
        )?;
        let index_buffer = device_buffer_byte_view(
            packed_output_buffer,
            0,
            graph_index_bytes,
            "BF16 preloaded resident router weight+bias device-input top-k index view",
        )?;
        let score_buffer = device_buffer_byte_view(
            packed_output_buffer,
            graph_score_offset,
            graph_score_bytes,
            "BF16 preloaded resident router weight+bias device-input top-k score view",
        )?;
        let weight_buffer_out = device_buffer_byte_view(
            packed_output_buffer,
            graph_weight_offset,
            graph_score_bytes,
            "BF16 preloaded resident router weight+bias device-input top-k weight view",
        )?;

        let backend = capture_or_update_router_topk_bf16_graph_for_slot(
            graph_key,
            library,
            slot,
            hidden_buffer,
            weight_buffer,
            bias_buffer,
            index_buffer,
            score_buffer,
            weight_buffer_out,
            hidden.rows,
            hidden.values_per_row,
            experts,
            top_k,
            CUDA_REFERENCE_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_BIAS_DEVICE_INPUT_BACKEND,
            TRITON_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_BIAS_DEVICE_INPUT_BACKEND,
            "BF16 preloaded resident router top-k weight+bias device-input",
        )?;
        let packed_output = slot.workspace.host_buffer(
            library,
            CoordinatorCudaScratchSlot::D,
            graph_output_bytes,
            "BF16 preloaded resident router weight+bias device-input packed top-k readback",
        )?;
        unsafe {
            library
                .copy_d2h_host_buffer_async(
                    packed_output,
                    packed_output_buffer,
                    graph_output_bytes,
                    cuda_stream,
                )
                .context(
                    "async copying BF16 preloaded resident router weight+bias device-input packed top-k output to host",
                )?;
            library
                .cuda_stream_synchronize(cuda_stream)
                .context(
                    "synchronizing BF16 preloaded resident router weight+bias device-input graph slot stream",
                )?;
        }

        let packed_bytes = unsafe {
            std::slice::from_raw_parts(packed_output.ptr.cast::<u8>(), graph_output_bytes)
        };
        let index_host_bytes = &packed_bytes[..index_bytes];
        let score_host_end = graph_score_offset.checked_add(score_bytes).context(
            "CUDA BF16 preloaded resident router top-k packed score readback end overflow usize",
        )?;
        let score_host_bytes = &packed_bytes[graph_score_offset..score_host_end];
        let weight_host_end = graph_weight_offset.checked_add(score_bytes).context(
            "CUDA BF16 preloaded resident router top-k packed weight readback end overflow usize",
        )?;
        let weight_host_bytes = &packed_bytes[graph_weight_offset..weight_host_end];

        Ok(RouterTopKOutput {
            indices: u32_vec_from_bytes(index_host_bytes)?
                .into_iter()
                .map(|value| value as usize)
                .collect(),
            scores: f32_vec_from_bytes(score_host_bytes)?,
            weights: f32_vec_from_bytes(weight_host_bytes)?,
            backend,
        })
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn cuda_router_topk_bf16_preloaded_resident_weight_bias_device_input_legacy(
    router_weight_name: &str,
    correction_bias_name: &str,
    hidden: &DeviceBf16Output,
    experts: usize,
    top_k: usize,
    weight_bytes: usize,
    bias_bytes: usize,
) -> Result<RouterTopKOutput> {
    let library = cuda_native_library()?;
    let (_, index_bytes, score_bytes) = router_topk_output_byte_counts(
        hidden.rows,
        top_k,
        "CUDA BF16 preloaded resident router top-k weight+bias device-input",
    )?;
    let weight_buffer = preloaded_resident_weight_device_buffer(router_weight_name, weight_bytes)?;
    let bias_buffer = preloaded_resident_weight_device_buffer(correction_bias_name, bias_bytes)?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let index_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::D,
        index_bytes,
        "BF16 preloaded resident router weight+bias device-input top-k indices",
    )?;
    let score_buffer = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::E,
        score_bytes,
        "BF16 preloaded resident router weight+bias device-input top-k scores",
    )?;
    let weight_buffer_out = workspace.buffer(
        library,
        CoordinatorCudaScratchSlot::F,
        score_bytes,
        "BF16 preloaded resident router weight+bias device-input top-k weights",
    )?;

    library
        .cuda_router_topk_bf16(
            hidden.buffer(),
            weight_buffer,
            bias_buffer,
            index_buffer,
            score_buffer,
            weight_buffer_out,
            hidden.rows,
            hidden.values_per_row,
            experts,
            top_k,
        )
        .context("executing CUDA BF16 preloaded resident router top-k weight+bias device-input")?;
    let mut index_out = vec![0_u8; index_bytes];
    let mut score_out = vec![0_u8; score_bytes];
    let mut weight_out = vec![0_u8; score_bytes];
    library.copy_d2h(&mut index_out, index_buffer).context(
        "copying BF16 preloaded resident router weight+bias device-input top-k indices to host",
    )?;
    library.copy_d2h(&mut score_out, score_buffer).context(
        "copying BF16 preloaded resident router weight+bias device-input top-k scores to host",
    )?;
    library
        .copy_d2h(&mut weight_out, weight_buffer_out)
        .context(
            "copying BF16 preloaded resident router weight+bias device-input top-k weights to host",
        )?;

    Ok(RouterTopKOutput {
        indices: u32_vec_from_bytes(&index_out)?
            .into_iter()
            .map(|value| value as usize)
            .collect(),
        scores: f32_vec_from_bytes(&score_out)?,
        weights: f32_vec_from_bytes(&weight_out)?,
        backend:
            CUDA_REFERENCE_ROUTER_TOPK_BF16_PRELOADED_RESIDENT_WEIGHT_BIAS_DEVICE_INPUT_BACKEND,
    })
}
