use super::*;
use anyhow::{Context, Result};

pub(in crate::commands::real_full) const DS4_FLASH_SHARED_EXPERT_FP8_BF16_BACKEND: &str =
    "deepseek-v4-flash-shared-expert-fp8-bf16-aot";
pub(in crate::commands::real_full) const DS4_PRO_SHARED_EXPERT_FP8_BF16_BACKEND: &str =
    "deepseek-v4-pro-shared-expert-fp8-bf16-aot";

const DS4_FLASH_HIDDEN: usize = 4_096;
const DS4_FLASH_SHARED_INTERMEDIATE: usize = 2_048;
const DS4_PRO_HIDDEN: usize = 7_168;
const DS4_PRO_SHARED_INTERMEDIATE: usize = 3_072;
const DS4_FLASH_SHARED_CHUNK_ROWS: usize = 8;
const DS4_FLASH_SHARED_MAX_LAUNCH_ROWS: usize = 2_048;
const DS4_FLASH_SHARED_ALPHA_NAME: &str = "ds4rt#deepseek-v4-flash-shared-expert-alpha";
const DS4_PRO_SHARED_ALPHA_NAME: &str = "ds4rt#deepseek-v4-pro-shared-expert-alpha";
const DS4_FLASH_SHARED_WORKSPACE_BYTES: usize =
    DS4_FLASH_SHARED_CHUNK_ROWS * DS4_FLASH_SHARED_INTERMEDIATE * std::mem::size_of::<u16>();

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ds4SharedExpertVariant {
    Flash,
    Pro,
}

#[derive(Debug, Clone, Copy)]
struct Ds4SharedExpertProfile {
    variant: Ds4SharedExpertVariant,
    hidden: usize,
    intermediate: usize,
    alpha_name: &'static str,
    backend: &'static str,
}

impl Ds4SharedExpertProfile {
    fn for_geometry(hidden: usize, intermediate: usize) -> Result<Self> {
        match (hidden, intermediate) {
            (DS4_FLASH_HIDDEN, DS4_FLASH_SHARED_INTERMEDIATE) => Ok(Self {
                variant: Ds4SharedExpertVariant::Flash,
                hidden,
                intermediate,
                alpha_name: DS4_FLASH_SHARED_ALPHA_NAME,
                backend: DS4_FLASH_SHARED_EXPERT_FP8_BF16_BACKEND,
            }),
            (DS4_PRO_HIDDEN, DS4_PRO_SHARED_INTERMEDIATE) => Ok(Self {
                variant: Ds4SharedExpertVariant::Pro,
                hidden,
                intermediate,
                alpha_name: DS4_PRO_SHARED_ALPHA_NAME,
                backend: DS4_PRO_SHARED_EXPERT_FP8_BF16_BACKEND,
            }),
            _ => anyhow::bail!(
                "DeepSeek V4 shared-expert AOT does not support hidden={hidden} intermediate={intermediate}"
            ),
        }
    }

    fn workspace_bytes(self) -> Result<usize> {
        DS4_FLASH_SHARED_CHUNK_ROWS
            .checked_mul(self.intermediate)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("DeepSeek V4 shared-expert workspace byte count overflow")
    }
}

/// Caller-owned scratch and output for the coordinator-resident Flash shared
/// expert. All four regions have fixed identities for CUDA graph replay.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) struct Ds4FlashSharedExpertDeviceBuffers {
    pub(in crate::commands::real_full) gate: Ds4rtDeviceBuffer,
    pub(in crate::commands::real_full) up: Ds4rtDeviceBuffer,
    pub(in crate::commands::real_full) activated: Ds4rtDeviceBuffer,
    pub(in crate::commands::real_full) output: Ds4rtDeviceBuffer,
}

#[derive(Debug, Clone, Copy)]
struct Ds4FlashSharedExpertResidentWeights {
    w1_weight: Ds4rtDeviceBuffer,
    w1_scale_mma: Ds4rtDeviceBuffer,
    w3_weight: Ds4rtDeviceBuffer,
    w3_scale_mma: Ds4rtDeviceBuffer,
    w2_weight: Ds4rtDeviceBuffer,
    w2_scale_mma: Ds4rtDeviceBuffer,
    alpha: Ds4rtDeviceBuffer,
}

pub(in crate::commands::real_full) fn preload_ds4_flash_shared_expert_fp8_runtime() -> Result<()> {
    preload_ds4_shared_expert_fp8_runtime_for_geometry(
        DS4_FLASH_HIDDEN,
        DS4_FLASH_SHARED_INTERMEDIATE,
    )
}

pub(in crate::commands::real_full) fn preload_ds4_shared_expert_fp8_runtime_for_geometry(
    hidden: usize,
    intermediate: usize,
) -> Result<()> {
    let profile = Ds4SharedExpertProfile::for_geometry(hidden, intermediate)?;
    let library = cuda_native_library()?;
    match profile.variant {
        Ds4SharedExpertVariant::Flash => {
            anyhow::ensure!(
                library
                    .cuda_ds4_flash_spark_aot_available()
                    .context("querying DeepSeek V4 Flash shared-expert AOT availability")?,
                "native library does not contain the DeepSeek V4 Flash shared-expert AOT module"
            );
            library
                .cuda_ds4_flash_spark_aot_init()
                .context("initializing DeepSeek V4 Flash shared-expert AOT kernels")?;
        }
        Ds4SharedExpertVariant::Pro => {
            anyhow::ensure!(
                library
                    .cuda_ds4_pro_spark_aot_available()
                    .context("querying DeepSeek V4 Pro shared-expert AOT availability")?,
                "native library does not contain the DeepSeek V4 Pro shared-expert AOT module"
            );
            library
                .cuda_ds4_pro_spark_aot_init()
                .context("initializing DeepSeek V4 Pro shared-expert AOT kernels")?;
        }
    }
    preload_resident_weight_from_host_staging(
        profile.alpha_name,
        std::mem::size_of::<f32>(),
        "DeepSeek V4 shared-expert alpha",
        |staging| {
            staging.copy_from_slice(&1.0_f32.to_ne_bytes());
            Ok(())
        },
    )
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) fn ds4_flash_shared_expert_fp8_bf16_device_output_into(
    w1_weight_name: &str,
    w1_scale_name: &str,
    w3_weight_name: &str,
    w3_scale_name: &str,
    w2_weight_name: &str,
    w2_scale_name: &str,
    input: &DeviceBf16Output,
    rows: usize,
    buffers: Ds4FlashSharedExpertDeviceBuffers,
    cuda_stream: *mut c_void,
) -> Result<()> {
    ds4_shared_expert_fp8_bf16_device_output_into_for_geometry(
        w1_weight_name,
        w1_scale_name,
        w3_weight_name,
        w3_scale_name,
        w2_weight_name,
        w2_scale_name,
        input,
        rows,
        DS4_FLASH_SHARED_INTERMEDIATE,
        buffers,
        cuda_stream,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn ds4_shared_expert_fp8_bf16_device_output_into_for_geometry(
    w1_weight_name: &str,
    w1_scale_name: &str,
    w3_weight_name: &str,
    w3_scale_name: &str,
    w2_weight_name: &str,
    w2_scale_name: &str,
    input: &DeviceBf16Output,
    rows: usize,
    intermediate: usize,
    buffers: Ds4FlashSharedExpertDeviceBuffers,
    cuda_stream: *mut c_void,
) -> Result<()> {
    let profile = Ds4SharedExpertProfile::for_geometry(input.values_per_row, intermediate)?;
    let (library, weights) = preloaded_ds4_shared_expert_device_weights(
        profile,
        w1_weight_name,
        w1_scale_name,
        w3_weight_name,
        w3_scale_name,
        w2_weight_name,
        w2_scale_name,
    )?;
    input
        .wait_ready_on_stream(cuda_stream)
        .context("waiting for DeepSeek V4 shared-expert input")?;
    launch_ds4_shared_expert_fp8_bf16_preloaded_device_input_into(
        library,
        profile,
        weights,
        input.buffer(),
        input.rows,
        input.values_per_row,
        rows,
        buffers,
        cuda_stream,
    )
}

/// Launch from a fixed device input whose producer dependency is already
/// ordered on `cuda_stream` (for example, by an enclosing CUDA graph).
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) fn ds4_flash_shared_expert_fp8_bf16_preloaded_device_input_into(
    w1_weight_name: &str,
    w1_scale_name: &str,
    w3_weight_name: &str,
    w3_scale_name: &str,
    w2_weight_name: &str,
    w2_scale_name: &str,
    input: Ds4rtDeviceBuffer,
    rows: usize,
    buffers: Ds4FlashSharedExpertDeviceBuffers,
    cuda_stream: *mut c_void,
) -> Result<()> {
    ds4_shared_expert_fp8_bf16_preloaded_device_input_into_for_geometry(
        w1_weight_name,
        w1_scale_name,
        w3_weight_name,
        w3_scale_name,
        w2_weight_name,
        w2_scale_name,
        input,
        rows,
        DS4_FLASH_HIDDEN,
        DS4_FLASH_SHARED_INTERMEDIATE,
        buffers,
        cuda_stream,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn ds4_shared_expert_fp8_bf16_preloaded_device_input_into_for_geometry(
    w1_weight_name: &str,
    w1_scale_name: &str,
    w3_weight_name: &str,
    w3_scale_name: &str,
    w2_weight_name: &str,
    w2_scale_name: &str,
    input: Ds4rtDeviceBuffer,
    rows: usize,
    hidden: usize,
    intermediate: usize,
    buffers: Ds4FlashSharedExpertDeviceBuffers,
    cuda_stream: *mut c_void,
) -> Result<()> {
    let profile = Ds4SharedExpertProfile::for_geometry(hidden, intermediate)?;
    let (library, weights) = preloaded_ds4_shared_expert_device_weights(
        profile,
        w1_weight_name,
        w1_scale_name,
        w3_weight_name,
        w3_scale_name,
        w2_weight_name,
        w2_scale_name,
    )?;
    launch_ds4_shared_expert_fp8_bf16_preloaded_device_input_into(
        library,
        profile,
        weights,
        input,
        rows,
        hidden,
        rows,
        buffers,
        cuda_stream,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn ds4_flash_shared_expert_fp8_bf16_device_output(
    w1_weight_name: &str,
    w1_scale_name: &str,
    w3_weight_name: &str,
    w3_scale_name: &str,
    w2_weight_name: &str,
    w2_scale_name: &str,
    input: &DeviceBf16Output,
    rows: usize,
) -> Result<DeviceBf16Output> {
    ds4_shared_expert_fp8_bf16_device_output_for_geometry(
        w1_weight_name,
        w1_scale_name,
        w3_weight_name,
        w3_scale_name,
        w2_weight_name,
        w2_scale_name,
        input,
        rows,
        DS4_FLASH_SHARED_INTERMEDIATE,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn ds4_shared_expert_fp8_bf16_device_output_for_geometry(
    w1_weight_name: &str,
    w1_scale_name: &str,
    w3_weight_name: &str,
    w3_scale_name: &str,
    w2_weight_name: &str,
    w2_scale_name: &str,
    input: &DeviceBf16Output,
    rows: usize,
    intermediate: usize,
) -> Result<DeviceBf16Output> {
    let profile = Ds4SharedExpertProfile::for_geometry(input.values_per_row, intermediate)?;
    let (library, weights) = preloaded_ds4_shared_expert_device_weights(
        profile,
        w1_weight_name,
        w1_scale_name,
        w3_weight_name,
        w3_scale_name,
        w2_weight_name,
        w2_scale_name,
    )?;
    if profile.variant == Ds4SharedExpertVariant::Pro && rows > DS4_FLASH_SHARED_CHUNK_ROWS {
        return ds4_pro_shared_expert_fp8_batched_prefill_device_output(
            library, profile, weights, input, rows,
        );
    }
    let workspace_bytes = profile.workspace_bytes()?;
    let gate = OwnedCoordinatorDeviceBuffer::new(
        library,
        workspace_bytes,
        "DeepSeek V4 shared-expert gate workspace",
    )?;
    let up = OwnedCoordinatorDeviceBuffer::new(
        library,
        workspace_bytes,
        "DeepSeek V4 shared-expert up workspace",
    )?;
    let activated = OwnedCoordinatorDeviceBuffer::new(
        library,
        workspace_bytes,
        "DeepSeek V4 shared-expert activation workspace",
    )?;
    let mut output = device_bf16_output_uninitialized(
        rows,
        profile.hidden,
        profile.backend,
        "DeepSeek V4 shared-expert output",
    )?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let stream = workspace.stream_ptr(library)?;
    input
        .wait_ready_on_stream(stream)
        .context("waiting for DeepSeek V4 shared-expert input")?;
    launch_ds4_shared_expert_fp8_bf16_preloaded_device_input_into(
        library,
        profile,
        weights,
        input.buffer(),
        input.rows,
        input.values_per_row,
        rows,
        Ds4FlashSharedExpertDeviceBuffers {
            gate: gate.buffer,
            up: up.buffer,
            activated: activated.buffer,
            output: output.buffer(),
        },
        stream,
    )?;
    let ready_event = Arc::new(CoordinatorCudaEvent::create(library)?);
    ready_event.record(stream)?;
    output.set_ready_event(ready_event);
    output
        .buffer
        .retain_dependencies_until_ready([gate, up, activated]);
    Ok(output)
}

fn ds4_pro_shared_expert_fp8_batched_prefill_device_output(
    library: &'static NativeLibrary,
    profile: Ds4SharedExpertProfile,
    weights: Ds4FlashSharedExpertResidentWeights,
    input: &DeviceBf16Output,
    rows: usize,
) -> Result<DeviceBf16Output> {
    anyhow::ensure!(
        profile.variant == Ds4SharedExpertVariant::Pro
            && rows > DS4_FLASH_SHARED_CHUNK_ROWS
            && rows <= DS4_FLASH_SHARED_MAX_LAUNCH_ROWS
            && input.rows == rows
            && input.values_per_row == profile.hidden,
        "DeepSeek V4 Pro batched shared-expert prefill shape mismatch: rows={rows} input={}x{} hidden={}",
        input.rows,
        input.values_per_row,
        profile.hidden,
    );
    let workspace_bytes = rows
        .checked_mul(profile.intermediate)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("DeepSeek V4 Pro batched shared workspace byte count overflow")?;
    let mxfp8_bytes = |size_k: usize| -> Result<(usize, usize, usize)> {
        let values = rows
            .checked_mul(size_k)
            .context("DeepSeek V4 Pro MXFP8 value byte count overflow")?;
        let scale_rows = values / 32;
        let row_tiles = rows.div_ceil(128);
        let k_tiles = size_k.div_ceil(128);
        let scale_mma = row_tiles
            .checked_mul(k_tiles)
            .and_then(|tiles| tiles.checked_mul(512))
            .context("DeepSeek V4 Pro MXFP8 MMA-scale byte count overflow")?;
        Ok((values, scale_rows, scale_mma))
    };
    let (input_q_values_bytes, input_q_scale_rows_bytes, input_q_scale_mma_bytes) =
        mxfp8_bytes(profile.hidden)?;
    let (activated_q_values_bytes, activated_q_scale_rows_bytes, activated_q_scale_mma_bytes) =
        mxfp8_bytes(profile.intermediate)?;

    let gate = OwnedCoordinatorDeviceBuffer::new(
        library,
        workspace_bytes,
        "DeepSeek V4 Pro batched shared-expert gate workspace",
    )?;
    let up = OwnedCoordinatorDeviceBuffer::new(
        library,
        workspace_bytes,
        "DeepSeek V4 Pro batched shared-expert up workspace",
    )?;
    let activated = OwnedCoordinatorDeviceBuffer::new(
        library,
        workspace_bytes,
        "DeepSeek V4 Pro batched shared-expert activation workspace",
    )?;
    let input_q_values = OwnedCoordinatorDeviceBuffer::new(
        library,
        input_q_values_bytes,
        "DeepSeek V4 Pro shared-expert input MXFP8 values",
    )?;
    let input_q_scale_rows = OwnedCoordinatorDeviceBuffer::new(
        library,
        input_q_scale_rows_bytes,
        "DeepSeek V4 Pro shared-expert input MXFP8 row scales",
    )?;
    let input_q_scale_mma = OwnedCoordinatorDeviceBuffer::new(
        library,
        input_q_scale_mma_bytes,
        "DeepSeek V4 Pro shared-expert input MXFP8 MMA scales",
    )?;
    let activated_q_values = OwnedCoordinatorDeviceBuffer::new(
        library,
        activated_q_values_bytes,
        "DeepSeek V4 Pro shared-expert activated MXFP8 values",
    )?;
    let activated_q_scale_rows = OwnedCoordinatorDeviceBuffer::new(
        library,
        activated_q_scale_rows_bytes,
        "DeepSeek V4 Pro shared-expert activated MXFP8 row scales",
    )?;
    let activated_q_scale_mma = OwnedCoordinatorDeviceBuffer::new(
        library,
        activated_q_scale_mma_bytes,
        "DeepSeek V4 Pro shared-expert activated MXFP8 MMA scales",
    )?;
    let mut output = device_bf16_output_uninitialized(
        rows,
        profile.hidden,
        profile.backend,
        "DeepSeek V4 Pro batched shared-expert output",
    )?;
    let mut workspace = lock_coordinator_cuda_workspace()?;
    let stream = workspace.stream_ptr(library)?;
    input
        .wait_ready_on_stream(stream)
        .context("waiting for DeepSeek V4 Pro batched shared-expert input")?;
    let buffers = Ds4rtDs4ProSharedExpertFp8PrefillBuffers {
        base: Ds4rtDs4FlashSharedExpertFp8Buffers {
            input: input.buffer(),
            w1_weight: weights.w1_weight,
            w1_scale_mma: weights.w1_scale_mma,
            w3_weight: weights.w3_weight,
            w3_scale_mma: weights.w3_scale_mma,
            w2_weight: weights.w2_weight,
            w2_scale_mma: weights.w2_scale_mma,
            gate: gate.buffer,
            up: up.buffer,
            activated: activated.buffer,
            output: output.buffer(),
            alpha: weights.alpha,
        },
        input_q_values: input_q_values.buffer,
        input_q_scale_rows: input_q_scale_rows.buffer,
        input_q_scale_mma: input_q_scale_mma.buffer,
        activated_q_values: activated_q_values.buffer,
        activated_q_scale_rows: activated_q_scale_rows.buffer,
        activated_q_scale_mma: activated_q_scale_mma.buffer,
    };
    unsafe {
        library
            .cuda_ds4_pro_shared_expert_fp8_prefill_bf16_async(&buffers, rows, stream)
            .context("running DeepSeek V4 Pro batched shared-expert prefill")?;
    }
    let ready_event = Arc::new(CoordinatorCudaEvent::create(library)?);
    ready_event.record(stream)?;
    output.set_ready_event(ready_event);
    output.buffer.retain_dependencies_until_ready([
        gate,
        up,
        activated,
        input_q_values,
        input_q_scale_rows,
        input_q_scale_mma,
        activated_q_values,
        activated_q_scale_rows,
        activated_q_scale_mma,
    ]);
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn preloaded_ds4_shared_expert_device_weights(
    profile: Ds4SharedExpertProfile,
    w1_weight_name: &str,
    w1_scale_name: &str,
    w3_weight_name: &str,
    w3_scale_name: &str,
    w2_weight_name: &str,
    w2_scale_name: &str,
) -> Result<(&'static NativeLibrary, Ds4FlashSharedExpertResidentWeights)> {
    let library = cuda_native_library()?;
    let up_weight_bytes = profile
        .intermediate
        .checked_mul(profile.hidden)
        .context("DeepSeek V4 shared up-weight byte count overflow")?;
    let down_weight_bytes = profile
        .hidden
        .checked_mul(profile.intermediate)
        .context("DeepSeek V4 shared down-weight byte count overflow")?;
    let w1_weight = preloaded_resident_weight_device_buffer(w1_weight_name, up_weight_bytes)?;
    let w3_weight = preloaded_resident_weight_device_buffer(w3_weight_name, up_weight_bytes)?;
    let w2_weight = preloaded_resident_weight_device_buffer(w2_weight_name, down_weight_bytes)?;
    let w1_scale_mma = preloaded_ds4_flash_fp8_block_scale_mma(
        w1_scale_name,
        profile.intermediate,
        profile.hidden,
    )?;
    let w3_scale_mma = preloaded_ds4_flash_fp8_block_scale_mma(
        w3_scale_name,
        profile.intermediate,
        profile.hidden,
    )?;
    let w2_scale_mma = preloaded_ds4_flash_fp8_block_scale_mma(
        w2_scale_name,
        profile.hidden,
        profile.intermediate,
    )?;
    let alpha =
        preloaded_resident_weight_device_buffer(profile.alpha_name, std::mem::size_of::<f32>())?;
    Ok((
        library,
        Ds4FlashSharedExpertResidentWeights {
            w1_weight,
            w1_scale_mma,
            w3_weight,
            w3_scale_mma,
            w2_weight,
            w2_scale_mma,
            alpha,
        },
    ))
}

fn launch_ds4_shared_expert_fp8_bf16_preloaded_device_input_into(
    library: &'static NativeLibrary,
    profile: Ds4SharedExpertProfile,
    weights: Ds4FlashSharedExpertResidentWeights,
    input_buffer: Ds4rtDeviceBuffer,
    input_rows: usize,
    input_values_per_row: usize,
    rows: usize,
    buffers: Ds4FlashSharedExpertDeviceBuffers,
    stream: *mut c_void,
) -> Result<()> {
    validate_ds4_shared_expert_buffer_layout_for_geometry(
        profile,
        input_buffer,
        input_rows,
        input_values_per_row,
        rows,
        buffers,
    )?;

    let input_row_bytes = profile
        .hidden
        .checked_mul(std::mem::size_of::<u16>())
        .context("DeepSeek V4 shared-expert row byte count overflow")?;
    let mut row_offset = 0_usize;
    while row_offset < rows {
        let chunk_rows = (rows - row_offset).min(DS4_FLASH_SHARED_MAX_LAUNCH_ROWS);
        let chunk_bytes = chunk_rows
            .checked_mul(input_row_bytes)
            .context("DeepSeek V4 Flash shared-expert chunk byte count overflow")?;
        let buffers = Ds4rtDs4FlashSharedExpertFp8Buffers {
            input: device_buffer_byte_view(
                input_buffer,
                row_offset * input_row_bytes,
                chunk_bytes,
                "DeepSeek V4 Flash shared-expert input rows",
            )?,
            w1_weight: weights.w1_weight,
            w1_scale_mma: weights.w1_scale_mma,
            w3_weight: weights.w3_weight,
            w3_scale_mma: weights.w3_scale_mma,
            w2_weight: weights.w2_weight,
            w2_scale_mma: weights.w2_scale_mma,
            gate: buffers.gate,
            up: buffers.up,
            activated: buffers.activated,
            output: device_buffer_byte_view(
                buffers.output,
                row_offset * input_row_bytes,
                chunk_bytes,
                "DeepSeek V4 Flash shared-expert output rows",
            )?,
            alpha: weights.alpha,
        };
        unsafe {
            match profile.variant {
                Ds4SharedExpertVariant::Flash => library
                    .cuda_ds4_flash_shared_expert_fp8_bf16_async(&buffers, chunk_rows, stream),
                Ds4SharedExpertVariant::Pro => {
                    library.cuda_ds4_pro_shared_expert_fp8_bf16_async(&buffers, chunk_rows, stream)
                }
            }
            .with_context(|| {
                format!(
                    "running DeepSeek V4 {:?} shared expert for rows {row_offset}..{}",
                    profile.variant,
                    row_offset + chunk_rows
                )
            })?;
        }
        row_offset += chunk_rows;
    }
    Ok(())
}

fn validate_ds4_flash_shared_expert_buffer_layout(
    input_buffer: Ds4rtDeviceBuffer,
    input_rows: usize,
    input_values_per_row: usize,
    rows: usize,
    buffers: Ds4FlashSharedExpertDeviceBuffers,
) -> Result<()> {
    validate_ds4_shared_expert_buffer_layout_for_geometry(
        Ds4SharedExpertProfile::for_geometry(DS4_FLASH_HIDDEN, DS4_FLASH_SHARED_INTERMEDIATE)?,
        input_buffer,
        input_rows,
        input_values_per_row,
        rows,
        buffers,
    )
}

fn validate_ds4_shared_expert_buffer_layout_for_geometry(
    profile: Ds4SharedExpertProfile,
    input_buffer: Ds4rtDeviceBuffer,
    input_rows: usize,
    input_values_per_row: usize,
    rows: usize,
    buffers: Ds4FlashSharedExpertDeviceBuffers,
) -> Result<()> {
    anyhow::ensure!(
        rows > 0,
        "DeepSeek V4 Flash shared expert requires nonzero rows"
    );
    anyhow::ensure!(
        input_rows == rows && input_values_per_row == profile.hidden,
        "DeepSeek V4 shared-expert input shape mismatch: expected {rows}x{}, got {}x{}",
        profile.hidden,
        input_rows,
        input_values_per_row
    );
    let input_bytes = rows
        .checked_mul(profile.hidden)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("DeepSeek V4 Flash shared-expert input byte count overflow")?;
    device_buffer_byte_view(
        input_buffer,
        0,
        input_bytes,
        "DeepSeek V4 Flash shared-expert input",
    )?;
    let workspace_bytes = profile.workspace_bytes()?;
    let named = [
        ("gate", buffers.gate, workspace_bytes),
        ("up", buffers.up, workspace_bytes),
        ("activated", buffers.activated, workspace_bytes),
        ("output", buffers.output, input_bytes),
    ];
    for (name, buffer, bytes) in named {
        device_buffer_byte_view(buffer, 0, bytes, name)?;
        anyhow::ensure!(
            buffer.device_id == input_buffer.device_id,
            "DeepSeek V4 Flash shared-expert {name} must be on input device {}",
            input_buffer.device_id
        );
        anyhow::ensure!(
            !device_buffer_ranges_overlap(input_buffer, input_bytes, buffer, bytes)?,
            "DeepSeek V4 Flash shared-expert input and {name} buffers overlap"
        );
    }
    for left in 0..named.len() {
        for right in left + 1..named.len() {
            anyhow::ensure!(
                !device_buffer_ranges_overlap(
                    named[left].1,
                    named[left].2,
                    named[right].1,
                    named[right].2,
                )?,
                "DeepSeek V4 Flash shared-expert {} and {} buffers overlap",
                named[left].0,
                named[right].0
            );
        }
    }
    Ok(())
}

fn device_buffer_ranges_overlap(
    left: Ds4rtDeviceBuffer,
    left_bytes: usize,
    right: Ds4rtDeviceBuffer,
    right_bytes: usize,
) -> Result<bool> {
    let left_start = left.ptr as usize;
    let right_start = right.ptr as usize;
    let left_end = left_start
        .checked_add(left_bytes)
        .context("DeepSeek V4 Flash shared-expert left buffer address overflows usize")?;
    let right_end = right_start
        .checked_add(right_bytes)
        .context("DeepSeek V4 Flash shared-expert right buffer address overflows usize")?;
    Ok(left_start < right_end && right_start < left_end)
}

#[cfg(test)]
mod ds4_flash_shared_expert_handoff_tests {
    use super::{
        cuda_native_library, cuda_reference_kernels_test_override,
        device_bf16_output_from_bf16_bytes, ds4_shared_expert_fp8_bf16_device_output_for_geometry,
        f32_values_to_bf16_bytes, preload_ds4_flash_fp8_block_scale_mma,
        preload_ds4_shared_expert_fp8_runtime_for_geometry,
        preload_resident_weight_from_host_staging, validate_ds4_flash_shared_expert_buffer_layout,
        validate_ds4_shared_expert_buffer_layout_for_geometry, Ds4FlashSharedExpertDeviceBuffers,
        Ds4SharedExpertProfile, Ds4rtDs4FlashSharedExpertFp8Buffers, DS4_FLASH_HIDDEN,
        DS4_FLASH_SHARED_WORKSPACE_BYTES, DS4_PRO_HIDDEN, DS4_PRO_SHARED_EXPERT_FP8_BF16_BACKEND,
        DS4_PRO_SHARED_INTERMEDIATE,
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
    fn flash_shared_expert_handoff_accepts_four_fixed_device_regions() {
        let rows = 16;
        let output_bytes = rows * DS4_FLASH_HIDDEN * 2;
        let input = buffer(0x10_0000, output_bytes, 0);
        let buffers = Ds4FlashSharedExpertDeviceBuffers {
            gate: buffer(0x20_0000, DS4_FLASH_SHARED_WORKSPACE_BYTES, 0),
            up: buffer(0x21_0000, DS4_FLASH_SHARED_WORKSPACE_BYTES, 0),
            activated: buffer(0x22_0000, DS4_FLASH_SHARED_WORKSPACE_BYTES, 0),
            output: buffer(0x30_0000, output_bytes, 0),
        };

        validate_ds4_flash_shared_expert_buffer_layout(
            input,
            rows,
            DS4_FLASH_HIDDEN,
            rows,
            buffers,
        )
        .expect("validating fixed Flash shared-expert handoff");
    }

    #[test]
    fn pro_shared_expert_handoff_accepts_four_fixed_device_regions() {
        let rows = 16;
        let profile =
            Ds4SharedExpertProfile::for_geometry(DS4_PRO_HIDDEN, DS4_PRO_SHARED_INTERMEDIATE)
                .expect("resolving Pro shared-expert profile");
        let workspace_bytes = profile
            .workspace_bytes()
            .expect("computing Pro shared-expert workspace bytes");
        let output_bytes = rows * DS4_PRO_HIDDEN * std::mem::size_of::<u16>();
        let input = buffer(0x10_0000, output_bytes, 0);
        let buffers = Ds4FlashSharedExpertDeviceBuffers {
            gate: buffer(0x20_0000, workspace_bytes, 0),
            up: buffer(0x21_0000, workspace_bytes, 0),
            activated: buffer(0x22_0000, workspace_bytes, 0),
            output: buffer(0x30_0000, output_bytes, 0),
        };

        validate_ds4_shared_expert_buffer_layout_for_geometry(
            profile,
            input,
            rows,
            DS4_PRO_HIDDEN,
            rows,
            buffers,
        )
        .expect("validating fixed Pro shared-expert handoff");
    }

    #[test]
    fn flash_shared_expert_handoff_rejects_undersized_or_aliased_regions() {
        let rows = 1;
        let output_bytes = DS4_FLASH_HIDDEN * 2;
        let input = buffer(0x10_0000, output_bytes, 0);
        let short_gate = Ds4FlashSharedExpertDeviceBuffers {
            gate: buffer(0x20_0000, DS4_FLASH_SHARED_WORKSPACE_BYTES - 1, 0),
            up: buffer(0x21_0000, DS4_FLASH_SHARED_WORKSPACE_BYTES, 0),
            activated: buffer(0x22_0000, DS4_FLASH_SHARED_WORKSPACE_BYTES, 0),
            output: buffer(0x30_0000, output_bytes, 0),
        };
        let error = validate_ds4_flash_shared_expert_buffer_layout(
            input,
            rows,
            DS4_FLASH_HIDDEN,
            rows,
            short_gate,
        )
        .expect_err("rejecting undersized shared gate workspace");
        assert!(error.to_string().contains("gate device buffer view"));

        let aliased_output = Ds4FlashSharedExpertDeviceBuffers {
            gate: buffer(0x20_0000, DS4_FLASH_SHARED_WORKSPACE_BYTES, 0),
            up: buffer(0x21_0000, DS4_FLASH_SHARED_WORKSPACE_BYTES, 0),
            activated: buffer(0x22_0000, DS4_FLASH_SHARED_WORKSPACE_BYTES, 0),
            output: input,
        };
        let error = validate_ds4_flash_shared_expert_buffer_layout(
            input,
            rows,
            DS4_FLASH_HIDDEN,
            rows,
            aliased_output,
        )
        .expect_err("rejecting input/output alias");
        assert!(error
            .to_string()
            .contains("input and output buffers overlap"));
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn pro_shared_expert_aot_executes_m1_and_m8_on_gpu0() {
        const HIDDEN: usize = 7_168;
        const INTERMEDIATE: usize = 3_072;
        const ROWS: usize = 8;
        let library = cuda_native_library().expect("loading CUDA native library");
        assert!(library
            .cuda_ds4_pro_spark_aot_available()
            .expect("querying Pro AOT availability"));
        library
            .cuda_ds4_pro_spark_aot_init()
            .expect("initializing Pro shared-expert AOT modules");

        let row_bytes = HIDDEN * std::mem::size_of::<u16>();
        let weight_bytes = HIDDEN * INTERMEDIATE;
        let scale_mma_bytes = (HIDDEN / 128) * (INTERMEDIATE / 128) * 512;
        let workspace_bytes = ROWS * INTERMEDIATE * std::mem::size_of::<u16>();
        let sizes = [
            ROWS * row_bytes,
            weight_bytes,
            scale_mma_bytes,
            weight_bytes,
            scale_mma_bytes,
            weight_bytes,
            scale_mma_bytes,
            workspace_bytes,
            workspace_bytes,
            workspace_bytes,
            ROWS * row_bytes,
            std::mem::size_of::<f32>(),
        ];
        let mut allocations = sizes
            .iter()
            .enumerate()
            .map(|(index, &bytes)| {
                library.alloc_device_buffer(bytes).unwrap_or_else(|error| {
                    panic!("allocating Pro shared-expert test buffer {index}: {error:#}")
                })
            })
            .collect::<Vec<_>>();
        for (&buffer, &bytes) in allocations.iter().zip(&sizes) {
            library
                .cuda_zero_bytes(buffer, bytes)
                .expect("zeroing Pro shared-expert test buffer");
        }
        library
            .copy_h2d(allocations[11], &1.0_f32.to_ne_bytes())
            .expect("uploading Pro shared-expert alpha");

        let buffers = Ds4rtDs4FlashSharedExpertFp8Buffers {
            input: allocations[0],
            w1_weight: allocations[1],
            w1_scale_mma: allocations[2],
            w3_weight: allocations[3],
            w3_scale_mma: allocations[4],
            w2_weight: allocations[5],
            w2_scale_mma: allocations[6],
            gate: allocations[7],
            up: allocations[8],
            activated: allocations[9],
            output: allocations[10],
            alpha: allocations[11],
        };
        for rows in [1_usize, ROWS] {
            unsafe {
                library
                    .cuda_ds4_pro_shared_expert_fp8_bf16_async(&buffers, rows, std::ptr::null_mut())
                    .unwrap_or_else(|error| {
                        panic!("launching Pro shared expert for {rows} rows: {error:#}")
                    });
            }
            let mut output = vec![0_u8; rows * row_bytes];
            library
                .copy_d2h(&mut output, buffers.output)
                .unwrap_or_else(|error| {
                    panic!("reading Pro shared-expert {rows}-row output: {error:#}")
                });
            assert!(output.iter().all(|&byte| byte == 0));
        }

        for (index, buffer) in allocations.iter_mut().enumerate().rev() {
            library
                .free_device_buffer(buffer)
                .unwrap_or_else(|error| panic!("freeing Pro test buffer {index}: {error:#}"));
        }
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn pro_shared_expert_resident_rust_path_executes_production_geometry_on_gpu0() {
        const W1_WEIGHT: &str = "ds4rt#test-pro-shared-w1-weight";
        const W1_SCALE: &str = "ds4rt#test-pro-shared-w1-scale";
        const W3_WEIGHT: &str = "ds4rt#test-pro-shared-w3-weight";
        const W3_SCALE: &str = "ds4rt#test-pro-shared-w3-scale";
        const W2_WEIGHT: &str = "ds4rt#test-pro-shared-w2-weight";
        const W2_SCALE: &str = "ds4rt#test-pro-shared-w2-scale";
        let _cuda_reference_override = cuda_reference_kernels_test_override(true);
        let rows = 9_usize;
        let weight_bytes = DS4_PRO_HIDDEN * DS4_PRO_SHARED_INTERMEDIATE;
        let scale_bytes = (DS4_PRO_HIDDEN / 128) * (DS4_PRO_SHARED_INTERMEDIATE / 128);

        preload_ds4_shared_expert_fp8_runtime_for_geometry(
            DS4_PRO_HIDDEN,
            DS4_PRO_SHARED_INTERMEDIATE,
        )
        .expect("preloading Pro shared-expert runtime");
        for (name, bytes) in [
            (W1_WEIGHT, weight_bytes),
            (W1_SCALE, scale_bytes),
            (W3_WEIGHT, weight_bytes),
            (W3_SCALE, scale_bytes),
            (W2_WEIGHT, weight_bytes),
            (W2_SCALE, scale_bytes),
        ] {
            preload_resident_weight_from_host_staging(
                name,
                bytes,
                "synthetic Pro shared-expert resident tensor",
                |staging| {
                    staging.fill(0);
                    Ok(())
                },
            )
            .unwrap_or_else(|error| panic!("preloading {name}: {error:#}"));
        }
        preload_ds4_flash_fp8_block_scale_mma(
            W1_SCALE,
            DS4_PRO_SHARED_INTERMEDIATE,
            DS4_PRO_HIDDEN,
        )
        .expect("packing Pro shared w1 scales");
        preload_ds4_flash_fp8_block_scale_mma(
            W3_SCALE,
            DS4_PRO_SHARED_INTERMEDIATE,
            DS4_PRO_HIDDEN,
        )
        .expect("packing Pro shared w3 scales");
        preload_ds4_flash_fp8_block_scale_mma(
            W2_SCALE,
            DS4_PRO_HIDDEN,
            DS4_PRO_SHARED_INTERMEDIATE,
        )
        .expect("packing Pro shared w2 scales");

        let input_values = (0..rows * DS4_PRO_HIDDEN)
            .map(|index| ((index * 13 % 127) as f32 - 63.0) / 512.0)
            .collect::<Vec<_>>();
        let input = device_bf16_output_from_bf16_bytes(
            &f32_values_to_bf16_bytes(&input_values),
            rows,
            DS4_PRO_HIDDEN,
            "synthetic Pro shared-expert input",
        )
        .expect("uploading Pro shared-expert input");
        let output = ds4_shared_expert_fp8_bf16_device_output_for_geometry(
            W1_WEIGHT,
            W1_SCALE,
            W3_WEIGHT,
            W3_SCALE,
            W2_WEIGHT,
            W2_SCALE,
            &input,
            rows,
            DS4_PRO_SHARED_INTERMEDIATE,
        )
        .expect("executing resident Pro shared-expert Rust path");
        assert_eq!(output.rows, rows);
        assert_eq!(output.values_per_row, DS4_PRO_HIDDEN);
        assert_eq!(output.backend, DS4_PRO_SHARED_EXPERT_FP8_BF16_BACKEND);
        assert!(output
            .copy_to_host_bytes()
            .expect("reading resident Pro shared-expert output")
            .iter()
            .all(|&byte| byte == 0));
    }
}
