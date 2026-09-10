use anyhow::{Context, Result};
use ds41rt_core::{
    plan_completion_first_routes, CompletionRoutePlanEntry, DType, ModelVariant, TensorCatalog,
    TensorInfo,
};
use ds41rt_ffi::{
    Ds41rtB12xSparkW4a16MoeBuffers, Ds41rtDeviceBuffer, Ds41rtDs4FlashRoutePackBuffers,
    Ds41rtDs4FlashSparkExl3K2MoeBuffers, Ds41rtDs4FlashSparkExl3MixedMoeBuffers,
    Ds41rtDs4FlashSparkW4a16MoeBuffers, Ds41rtHostBuffer, Ds41rtNcclComm,
    Ds41rtNvfp4RouteBatchedMetadata, Ds41rtRouteShardReductionBuffers, NativeLibrary,
    DS41RT_DEVICE_BUFFER_FLAG_MANAGED, DS41RT_ROUTE_SHARD_LOCAL_BF16, DS41RT_ROUTE_SHARD_LOCAL_F32,
    DS41RT_ROUTE_SHARD_WIRE_BF16, DS41RT_ROUTE_SHARD_WIRE_FP8_E4M3_ROW_SCALED,
    DS41RT_ROUTE_SHARD_WIRE_NVFP4_E2M1_FP8_E4M3,
};
use ds41rt_loader::{
    dtype_byte_width, exl3_expert, exl3_expert_trellis_bits, exl3_projection_trellis_bits,
    is_deepseek_v4_exl3_recipe, is_deepseek_v4_mixed_exl3_recipe, load_tensor_bytes,
    load_tensor_rows, native_fp4_expert, Exl3Projection, Exl3Tp4ResidentGeometry, LoadedTensor,
    LoadedTensorRows, DEEPSEEK_V4_EXL3_RECIPE, DEEPSEEK_V4_EXL3_T12_LUT_BYTES,
};
use ds41rt_transport::{protocol_v2_verbs_host_execution_lanes, ExpertProtocolV2StreamPlan};
use io_uring::{opcode, types, IoUring};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    env,
    ffi::c_void,
    fs::{File, OpenOptions},
    ops::{Deref, Range},
    os::unix::fs::OpenOptionsExt,
    os::{fd::AsRawFd, unix::fs::FileExt},
    path::{Path, PathBuf},
    ptr::NonNull,
    slice,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Condvar, Mutex, OnceLock,
    },
    time::{Duration, Instant},
};

use super::super::coordinator_kernels::{
    cuda_native_library, device_bf16_output_uninitialized, require_cuda_enabled_native_library,
    DeviceBf16Output,
};
use super::super::intermediate_sharding::{
    balanced_row_partition, initialize_spark_expert_rdma_reduction_lane,
    initialize_spark_expert_reduction_lane, initialize_spark_weight_preload_communicator,
    spark_expert_intermediate_shard_from_env, ExpertIntermediateReductionDtype,
    ExpertIntermediateShard, SparkExpertReduction,
};
use super::super::rdma_reduction::SparkExpertRdmaReduction;
use super::math::{bf16_bytes_to_f32, dot_packed_nvfp4, first_f32_scalar, silu, tensor_row_bytes};
use super::router::ScoredRoute;

const REAL_FULL_CUDA_REFERENCE_KERNELS_ENV: &str = "DS41RT_REAL_FULL_CUDA_REFERENCE_KERNELS";
const REAL_FULL_CUDA_ROUTE_VALIDATE_ENV: &str = "DS41RT_REAL_FULL_CUDA_ROUTE_VALIDATE";
const REAL_FULL_NVFP4_ROUTE_CUDA_GRAPHS_ENV: &str = "DS41RT_REAL_FULL_NVFP4_ROUTE_CUDA_GRAPHS";
const REAL_FULL_NVFP4_ROUTE_GROUPED_MULTIROW_ENV: &str =
    "DS41RT_REAL_FULL_NVFP4_ROUTE_GROUPED_MULTIROW";
const REAL_FULL_NVFP4_ROUTE_TIMING_ENV: &str = "DS41RT_REAL_FULL_NVFP4_ROUTE_TIMING";
const REAL_FULL_NVFP4_ROUTE_CUDA_EVENT_TIMING_ENV: &str =
    "DS41RT_REAL_FULL_NVFP4_ROUTE_CUDA_EVENT_TIMING";
const REAL_FULL_PROTOCOL_V2_EXECUTOR_TIMING_ENV: &str =
    "DS41RT_REAL_FULL_PROTOCOL_V2_EXECUTOR_TIMING";
const REAL_FULL_B12X_SPARK_DIRECT_ROUTE_ENV: &str = "DS41RT_B12X_SPARK_DIRECT_ROUTE";
const REAL_FULL_B12X_SPARK_GROUPED_DECODE_ENV: &str = "DS41RT_B12X_SPARK_GROUPED_DECODE";
const REAL_FULL_B12X_SPARK_W4A16_SMALL_M_MODE_ENV: &str = "DS41RT_B12X_SPARK_W4A16_SMALL_M_MODE";
const REAL_FULL_B12X_SPARK_W4A16_DEVICE_WEIGHTS_ENV: &str = "DS41RT_B12X_SPARK_W4A16_DEVICE_WEIGHTS";
const REAL_FULL_B12X_SPARK_ROUTE_LANES_ENV: &str = "DS41RT_B12X_SPARK_ROUTE_LANES";
const REAL_FULL_NVFP4_ROUTE_PRELOAD_IO_WORKERS_ENV: &str =
    "DS41RT_REAL_FULL_NVFP4_ROUTE_PRELOAD_IO_WORKERS";
const REAL_FULL_NVFP4_ROUTE_PRELOAD_DIRECT_IO_ENV: &str =
    "DS41RT_REAL_FULL_NVFP4_ROUTE_PRELOAD_DIRECT_IO";
const REAL_FULL_NVFP4_ROUTE_PRELOAD_COOPERATIVE_ENV: &str =
    "DS41RT_REAL_FULL_NVFP4_ROUTE_PRELOAD_COOPERATIVE";
const EXPERT_FUSED_FP8_REDUCTION_ENV: &str = "DS41RT_EXPERT_FUSED_FP8_REDUCTION";
const EXPERT_NCCL_BF16_REDUCE_ENV: &str = "DS41RT_EXPERT_NCCL_BF16_REDUCE";
const CPU_REFERENCE_NVFP4_ROUTE_BACKEND: &str = "cpu-reference-provisional-nvfp4-route";
const CUDA_REFERENCE_NVFP4_ROUTE_BF16_ACCUMULATED_BACKEND: &str =
    "cuda-reference-provisional-nvfp4-route-bf16-staged-accumulated";
const CUDA_REFERENCE_NVFP4_ROUTE_BF16_ACCUMULATED_DEVICE_INPUT_BACKEND: &str =
    "cuda-reference-provisional-nvfp4-route-bf16-staged-accumulated-device-input";
const CUDA_REFERENCE_NVFP4_ROUTE_BF16_ACCUMULATED_DEVICE_OUTPUT_BACKEND: &str =
    "cuda-reference-provisional-nvfp4-route-bf16-staged-accumulated-device-output";
const B12X_SPARK_DIRECT_NVFP4_ROUTE_BF16_ACCUMULATED_BACKEND: &str =
    "b12x-spark-aot-direct-nvfp4-route-bf16-staged-accumulated";
const RETAINED_BF16_MTP_ROUTE_BACKEND: &str = "retained-bf16-mtp-cublas-route-bf16-accumulated";
const DS4_FLASH_SPARK_AOT_BF16_BACKEND: &str = "deepseek-v4-flash-spark-aot-w4a16-bf16-tp4";
const DS4_FLASH_SPARK_EXL3_AOT_BF16_BACKEND: &str = "deepseek-v4-flash-spark-aot-exl3-k2-bf16-tp4";
const DS4_PRO_SPARK_EXL3_AOT_BF16_BACKEND: &str = "deepseek-v4-pro-spark-aot-exl3-k2-bf16-tp4";
const B12X_SPARK_AOT_MAX_ROWS: usize = 256;
const B12X_SPARK_ROUTE_MAX_LANES: usize = 4;
const B12X_W4A16_PREFILL_TOPK8_CAPACITY_ROWS: usize = 2048;
const B12X_W4A16_PREFILL_TOPK8_ROUTES: usize = 8;
const B12X_W4A16_EXPERTS: usize = 256;
const B12X_W4A16_MAX_PACKED_ROUTE_SLOTS: usize = 32_512;
const B12X_W4A16_MAX_ROUTE_BLOCKS: usize = 760;
const B12X_W4A16_SCRATCH_ELEMENTS: usize = 3_145_728;
const B12X_W4A16_LOCK_ELEMENTS: usize = 1_026;
const DS4_FLASH_HIDDEN_SIZE: usize = 4_096;
const DS4_FLASH_LOCAL_INTERMEDIATE_SIZE: usize = 512;
const DS4_FLASH_ROUTED_EXPERTS: usize = 256;
const DS4_FLASH_W13_WEIGHT_EXPERT_BYTES: usize = 2_097_152;
const DS4_FLASH_W13_SCALE_EXPERT_BYTES: usize = 131_072;
const DS4_FLASH_W2_WEIGHT_EXPERT_BYTES: usize = 1_048_576;
const DS4_FLASH_W2_SCALE_EXPERT_BYTES: usize = 65_536;
#[cfg(test)]
const DS4_FLASH_EXL3_TRELLIS_EXPERT_BYTES: usize = 524_288;
const DS4_FLASH_EXL3_MCG_MARKER: u32 = 0xCBAC_1FED;
const SQG_XOR_CHEB_T12_LUT_BYTES: usize = DEEPSEEK_V4_EXL3_T12_LUT_BYTES;
const SQG_CHEB_POSITIVE_TRANSITIONS: [u16; 77] = [
    17, 51, 85, 119, 153, 187, 221, 255, 289, 323, 357, 391, 426, 460, 494, 528, 579, 647, 715,
    783, 851, 919, 987, 1055, 1157, 1293, 1429, 1565, 1701, 1837, 1973, 2108, 2312, 2583, 2854,
    3124, 3395, 3665, 3934, 4203, 4606, 5141, 5674, 6205, 6732, 7258, 7779, 8298, 9070, 10085,
    11084, 12065, 13026, 13967, 14885, 15781, 17081, 18725, 20265, 21696, 23017, 24229, 25332,
    26330, 27637, 29054, 30143, 30957, 31548, 31967, 32255, 32447, 32617, 32717, 32753, 32764,
    32767,
];

fn sqg_xor_cheb_t12_lut_bytes() -> &'static [u8; SQG_XOR_CHEB_T12_LUT_BYTES] {
    static LUT: OnceLock<[u8; SQG_XOR_CHEB_T12_LUT_BYTES]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut lut = [0_u8; SQG_XOR_CHEB_T12_LUT_BYTES];
        for (block, output) in lut.iter_mut().enumerate() {
            let mut counts = [0_u8; 256];
            for offset in 0..16 {
                let rank = (block * 16 + offset) as u16;
                let negative = rank < 0x8000;
                let mut positive_offset = rank & 0x7fff;
                if negative {
                    positive_offset ^= 0x7fff;
                }
                let magnitude = SQG_CHEB_POSITIVE_TRANSITIONS
                    .iter()
                    .take_while(|&&transition| transition <= positive_offset)
                    .count() as u8;
                let sign = if negative && magnitude != 0 { 0x80 } else { 0 };
                counts[(magnitude | sign) as usize] += 1;
            }
            let mut best_count = 0_u8;
            let mut best_code = 0_u8;
            for (code, &count) in counts.iter().enumerate() {
                if count > best_count {
                    best_count = count;
                    best_code = code as u8;
                }
            }
            *output = best_code;
        }
        lut
    })
}
const DS4_FLASH_TOP_K: usize = 6;
const DS4_FLASH_PREFILL_MAX_ROWS: usize = 2_048;
const DS4_FLASH_PREFILL_ROUTE_BLOCK_ROWS: usize = 32;
const DS4_FLASH_PREFILL_MAX_PACKED_ROUTE_SLOTS: usize = 20_224;
const DS4_FLASH_PREFILL_MAX_ROUTE_BLOCKS: usize = 632;
const DS4_FLASH_PREFILL_SCRATCH_ELEMENTS: usize = 3_145_728;
const DS4_FLASH_FC1_SCRATCH_ELEMENTS: usize = 98_304;
const DS4_FLASH_FC2_SCRATCH_ELEMENTS: usize = 393_216;
const DS4_FLASH_LOCK_ELEMENTS: usize = 48 * 4 + 2;
const DS4_PRO_HIDDEN_SIZE: usize = 7_168;
const DS4_PRO_LOCAL_INTERMEDIATE_SIZE: usize = 768;
const DS4_PRO_ROUTED_EXPERTS: usize = 384;
const DS4_PRO_PREFILL_MAX_PACKED_ROUTE_SLOTS: usize = 24_192;
const DS4_PRO_PREFILL_MAX_ROUTE_BLOCKS: usize = 756;
const DS4_PRO_DECODE_PACKED_ROUTE_SLOTS: usize = 48;
const DS4_PRO_DECODE_ROUTE_BLOCKS: usize = 6;
const DS4_PRO_PREFILL_SCRATCH_ELEMENTS: usize = 3_145_728;
const DS4_PRO_FC1_SCRATCH_ELEMENTS: usize = 147_456;
const DS4_PRO_FC2_SCRATCH_ELEMENTS: usize = 688_128;
const SPARKINFER_EXL3_MODE_ENV: &str = "DS41RT_SPARKINFER_EXL3";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SparkinferExl3Mode {
    Auto,
    Disable,
    Force,
}

fn sparkinfer_exl3_mode_from_value(value: Option<&str>) -> Result<SparkinferExl3Mode> {
    match value.map(str::trim).filter(|value| !value.is_empty()) {
        None | Some("auto") => Ok(SparkinferExl3Mode::Auto),
        Some("disable") => Ok(SparkinferExl3Mode::Disable),
        Some("force") => Ok(SparkinferExl3Mode::Force),
        Some(value) => anyhow::bail!(
            "{SPARKINFER_EXL3_MODE_ENV} must be auto, disable, or force; got {value:?}"
        ),
    }
}

fn require_sparkinfer_exl3_mode(catalog: &TensorCatalog) -> Result<()> {
    let mode = sparkinfer_exl3_mode_from_value(env::var(SPARKINFER_EXL3_MODE_ENV).ok().as_deref())?;
    let checkpoint_is_exl3 = is_deepseek_v4_exl3_recipe(&catalog.facts.quantization_recipe);
    match mode {
        SparkinferExl3Mode::Auto => Ok(()),
        SparkinferExl3Mode::Disable => {
            anyhow::ensure!(
                !checkpoint_is_exl3,
                "{SPARKINFER_EXL3_MODE_ENV}=disable cannot load EXL3 recipe {}",
                catalog.facts.quantization_recipe
            );
            Ok(())
        }
        SparkinferExl3Mode::Force => {
            anyhow::ensure!(
                checkpoint_is_exl3,
                "{SPARKINFER_EXL3_MODE_ENV}=force requires recipe {DEEPSEEK_V4_EXL3_RECIPE}; got {}",
                catalog.facts.quantization_recipe
            );
            Ok(())
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Ds4Exl3AotProfile {
    variant: ModelVariant,
    geometry: Exl3Tp4ResidentGeometry,
    prefill_max_packed_route_slots: usize,
    prefill_max_route_blocks: usize,
    decode_packed_route_slots: usize,
    decode_route_blocks: usize,
    prefill_fc1_scratch_elements: usize,
    prefill_fc2_scratch_elements: usize,
    decode_fc1_scratch_elements: usize,
    decode_fc2_scratch_elements: usize,
}

impl Ds4Exl3AotProfile {
    fn from_catalog(catalog: &TensorCatalog) -> Result<Self> {
        anyhow::ensure!(
            is_deepseek_v4_exl3_recipe(&catalog.facts.quantization_recipe),
            "DeepSeek V4 EXL3 AOT requires recipe {DEEPSEEK_V4_EXL3_RECIPE}, got {}",
            catalog.facts.quantization_recipe
        );
        let geometry = Exl3Tp4ResidentGeometry::from_model_facts(&catalog.facts)?;
        Self::from_geometry(catalog, geometry)
    }

    fn from_catalog_layer(catalog: &TensorCatalog, layer_id: usize) -> Result<Self> {
        anyhow::ensure!(
            is_deepseek_v4_exl3_recipe(&catalog.facts.quantization_recipe),
            "DeepSeek V4 EXL3 AOT requires a recognized EXL3 recipe, got {}",
            catalog.facts.quantization_recipe
        );
        let trellis_bits = exl3_expert_trellis_bits(catalog, layer_id, 0)?;
        for expert_id in 1..catalog.facts.routed_experts {
            let expert_bits = exl3_expert_trellis_bits(catalog, layer_id, expert_id)?;
            anyhow::ensure!(
                expert_bits == trellis_bits,
                "EXL3 layer {layer_id} is not uniform: expert 0 is K{trellis_bits}, expert {expert_id} is K{expert_bits}"
            );
        }
        let geometry = Exl3Tp4ResidentGeometry::from_model_facts_with_trellis_bits(
            &catalog.facts,
            trellis_bits,
        )?;
        Self::from_geometry(catalog, geometry)
    }

    fn from_geometry(catalog: &TensorCatalog, geometry: Exl3Tp4ResidentGeometry) -> Result<Self> {
        let (
            prefill_max_packed_route_slots,
            prefill_max_route_blocks,
            decode_packed_route_slots,
            decode_route_blocks,
            prefill_fc1_scratch_elements,
            prefill_fc2_scratch_elements,
            decode_fc1_scratch_elements,
            decode_fc2_scratch_elements,
        ) = match catalog.facts.variant {
            ModelVariant::Flash
                if geometry.hidden_size == DS4_FLASH_HIDDEN_SIZE
                    && geometry.local_intermediate_size == DS4_FLASH_LOCAL_INTERMEDIATE_SIZE
                    && geometry.experts == DS4_FLASH_ROUTED_EXPERTS
                    && geometry.top_k == DS4_FLASH_TOP_K =>
            {
                (
                    DS4_FLASH_PREFILL_MAX_PACKED_ROUTE_SLOTS,
                    DS4_FLASH_PREFILL_MAX_ROUTE_BLOCKS,
                    DS4_FLASH_TOP_K,
                    1,
                    DS4_FLASH_PREFILL_SCRATCH_ELEMENTS,
                    DS4_FLASH_PREFILL_SCRATCH_ELEMENTS,
                    DS4_FLASH_FC1_SCRATCH_ELEMENTS,
                    DS4_FLASH_FC2_SCRATCH_ELEMENTS,
                )
            }
            ModelVariant::Pro
                if geometry.hidden_size == DS4_PRO_HIDDEN_SIZE
                    && geometry.local_intermediate_size == DS4_PRO_LOCAL_INTERMEDIATE_SIZE
                    && geometry.experts == DS4_PRO_ROUTED_EXPERTS
                    && geometry.top_k == DS4_FLASH_TOP_K =>
            {
                (
                    DS4_PRO_PREFILL_MAX_PACKED_ROUTE_SLOTS,
                    DS4_PRO_PREFILL_MAX_ROUTE_BLOCKS,
                    DS4_PRO_DECODE_PACKED_ROUTE_SLOTS,
                    DS4_PRO_DECODE_ROUTE_BLOCKS,
                    DS4_PRO_PREFILL_SCRATCH_ELEMENTS,
                    DS4_PRO_PREFILL_SCRATCH_ELEMENTS,
                    DS4_PRO_FC1_SCRATCH_ELEMENTS,
                    DS4_PRO_FC2_SCRATCH_ELEMENTS,
                )
            }
            variant => anyhow::bail!(
                "unsupported DeepSeek V4 {variant:?} EXL3 AOT geometry hidden={} local_intermediate={} experts={} top_k={}",
                geometry.hidden_size,
                geometry.local_intermediate_size,
                geometry.experts,
                geometry.top_k,
            ),
        };
        Ok(Self {
            variant: catalog.facts.variant,
            geometry,
            prefill_max_packed_route_slots,
            prefill_max_route_blocks,
            decode_packed_route_slots,
            decode_route_blocks,
            prefill_fc1_scratch_elements,
            prefill_fc2_scratch_elements,
            decode_fc1_scratch_elements,
            decode_fc2_scratch_elements,
        })
    }

    fn label(self) -> &'static str {
        match self.variant {
            ModelVariant::Flash => "Flash",
            ModelVariant::Pro => "Pro",
            ModelVariant::Custom => "Custom",
        }
    }

    fn backend(self) -> &'static str {
        match self.variant {
            ModelVariant::Flash => DS4_FLASH_SPARK_EXL3_AOT_BF16_BACKEND,
            ModelVariant::Pro => DS4_PRO_SPARK_EXL3_AOT_BF16_BACKEND,
            ModelVariant::Custom => unreachable!("custom EXL3 profile rejected at construction"),
        }
    }
}
pub(in crate::commands::real_full) const DS4_FLASH_DECODE_M1_HIDDEN_BYTES: usize =
    DS4_FLASH_HIDDEN_SIZE * std::mem::size_of::<u16>();
pub(in crate::commands::real_full) const DS4_FLASH_DECODE_M1_ROUTE_BYTES: usize =
    DS4_FLASH_TOP_K * std::mem::size_of::<u32>();
pub(in crate::commands::real_full) const DS4_FLASH_DECODE_M1_FC1_BYTES: usize =
    DS4_FLASH_TOP_K * 2 * DS4_FLASH_LOCAL_INTERMEDIATE_SIZE * std::mem::size_of::<u16>();
pub(in crate::commands::real_full) const DS4_FLASH_DECODE_M1_ACTIVATED_BYTES: usize =
    DS4_FLASH_TOP_K * DS4_FLASH_LOCAL_INTERMEDIATE_SIZE * std::mem::size_of::<u16>();
pub(in crate::commands::real_full) const DS4_FLASH_DECODE_M1_ROUTED_OUTPUT_BYTES: usize =
    DS4_FLASH_TOP_K * DS4_FLASH_HIDDEN_SIZE * std::mem::size_of::<u16>();
pub(in crate::commands::real_full) const DS4_FLASH_DECODE_M1_OUTPUT_F32_BYTES: usize =
    DS4_FLASH_HIDDEN_SIZE * std::mem::size_of::<f32>();
pub(in crate::commands::real_full) const DS4_FLASH_DECODE_M1_FC1_SCRATCH_BYTES: usize =
    DS4_FLASH_FC1_SCRATCH_ELEMENTS * std::mem::size_of::<f32>();
pub(in crate::commands::real_full) const DS4_FLASH_DECODE_M1_FC2_SCRATCH_BYTES: usize =
    DS4_FLASH_FC2_SCRATCH_ELEMENTS * std::mem::size_of::<f32>();
pub(in crate::commands::real_full) const DS4_FLASH_DECODE_M1_LOCK_BYTES: usize =
    DS4_FLASH_LOCK_ELEMENTS * std::mem::size_of::<i32>();
pub(in crate::commands::real_full) const DS4_FLASH_PREFILL_TOPK_ID_BYTES: usize =
    DS4_FLASH_PREFILL_MAX_ROWS * DS4_FLASH_TOP_K * std::mem::size_of::<i32>();
pub(in crate::commands::real_full) const DS4_FLASH_PREFILL_PACKED_ROUTE_BYTES: usize =
    DS4_FLASH_PREFILL_MAX_PACKED_ROUTE_SLOTS * std::mem::size_of::<i32>();
pub(in crate::commands::real_full) const DS4_FLASH_PREFILL_BLOCK_EXPERT_BYTES: usize =
    DS4_FLASH_PREFILL_MAX_ROUTE_BLOCKS * std::mem::size_of::<i32>();
pub(in crate::commands::real_full) const DS4_FLASH_PREFILL_EXPERT_COUNT_BYTES: usize =
    DS4_FLASH_ROUTED_EXPERTS * std::mem::size_of::<i32>();
pub(in crate::commands::real_full) const DS4_FLASH_PREFILL_EXPERT_OFFSET_BYTES: usize =
    (DS4_FLASH_ROUTED_EXPERTS + 1) * std::mem::size_of::<i32>();
pub(in crate::commands::real_full) const DS4_FLASH_PREFILL_HIDDEN_BYTES: usize =
    DS4_FLASH_PREFILL_MAX_ROWS * DS4_FLASH_HIDDEN_SIZE * std::mem::size_of::<u16>();
pub(in crate::commands::real_full) const DS4_FLASH_PREFILL_TOPK_WEIGHT_BYTES: usize =
    DS4_FLASH_PREFILL_MAX_ROWS * DS4_FLASH_TOP_K * std::mem::size_of::<f32>();
pub(in crate::commands::real_full) const DS4_FLASH_PREFILL_FC1_BYTES: usize =
    DS4_FLASH_PREFILL_MAX_ROWS
        * DS4_FLASH_TOP_K
        * 2
        * DS4_FLASH_LOCAL_INTERMEDIATE_SIZE
        * std::mem::size_of::<u16>();
pub(in crate::commands::real_full) const DS4_FLASH_PREFILL_ACTIVATED_BYTES: usize =
    DS4_FLASH_PREFILL_MAX_ROWS
        * DS4_FLASH_TOP_K
        * DS4_FLASH_LOCAL_INTERMEDIATE_SIZE
        * std::mem::size_of::<u16>();
pub(in crate::commands::real_full) const DS4_FLASH_PREFILL_ROUTED_OUTPUT_BYTES: usize =
    DS4_FLASH_PREFILL_MAX_ROWS
        * DS4_FLASH_TOP_K
        * DS4_FLASH_HIDDEN_SIZE
        * std::mem::size_of::<u16>();
pub(in crate::commands::real_full) const DS4_FLASH_PREFILL_OUTPUT_F32_BYTES: usize =
    DS4_FLASH_PREFILL_MAX_ROWS * DS4_FLASH_HIDDEN_SIZE * std::mem::size_of::<f32>();
pub(in crate::commands::real_full) const DS4_FLASH_PREFILL_COMPUTE_SCRATCH_BYTES: usize =
    DS4_FLASH_PREFILL_SCRATCH_ELEMENTS * std::mem::size_of::<f32>();
const STREAMING_FIRST_RESPONSE_ROWS: usize = 32;
const STREAMING_RESPONSE_MAX_ROWS: usize = 256;
const SPARK_COLLECTIVE_REQUEST_STRIDE: u64 = 65_536;
const SPARK_COLLECTIVE_HOST_REQUEST_WIDTH: u64 = 16;
const SPARK_COLLECTIVE_REORDER_WAIT: Duration = Duration::from_millis(20);
// Provisional f32 sanity envelope for real 6,144-wide NVFP4 route checks. This
// compares CPU libm against CUDA libdevice around near-zero cancellation; the
// BF16 ProtocolV2 executor has separate opt-in validation.
const CUDA_NVFP4_ROUTE_TOLERANCE: f32 = 1.0e-2;

#[derive(Default)]
struct SparkCollectiveLaunchOrderState {
    pending: BTreeSet<u64>,
    expected: Option<u64>,
    active: Option<u64>,
    gap_wait_started: Option<Instant>,
}

pub(in crate::commands::real_full) struct SparkCollectiveLaunchOrder {
    state: Mutex<SparkCollectiveLaunchOrderState>,
    changed: Condvar,
    reorder_wait: Duration,
}

fn canonical_spark_collective_request_id(request_id: u64) -> u64 {
    request_id - request_id % SPARK_COLLECTIVE_HOST_REQUEST_WIDTH
}

fn collective_gap_ready(
    pending: usize,
    reorder_pending: usize,
    elapsed: Duration,
    reorder_wait: Duration,
) -> bool {
    pending >= reorder_pending || elapsed >= reorder_wait
}

impl SparkCollectiveLaunchOrder {
    pub(in crate::commands::real_full) fn new(reorder_wait: Duration) -> Self {
        Self {
            state: Mutex::new(SparkCollectiveLaunchOrderState::default()),
            changed: Condvar::new(),
            reorder_wait,
        }
    }

    pub(in crate::commands::real_full) fn register(
        self: &Arc<Self>,
        request_id: u64,
    ) -> Result<SparkCollectiveLaunchTicket> {
        let request_id = canonical_spark_collective_request_id(request_id);
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("Spark collective launch-order lock is poisoned"))?;
        anyhow::ensure!(
            state.pending.insert(request_id),
            "Spark collective request {request_id} was registered twice"
        );
        self.changed.notify_all();
        drop(state);
        Ok(SparkCollectiveLaunchTicket {
            order: Arc::clone(self),
            request_id,
            owns_turn: false,
            finished: false,
        })
    }
}

pub(in crate::commands::real_full) struct SparkCollectiveLaunchTicket {
    order: Arc<SparkCollectiveLaunchOrder>,
    request_id: u64,
    owns_turn: bool,
    finished: bool,
}

impl SparkCollectiveLaunchTicket {
    fn wait_for_turn(&mut self) -> Result<()> {
        self.wait_for_turn_inner(protocol_v2_verbs_host_execution_lanes()?, None)
    }

    pub(in crate::commands::real_full) fn wait_for_turn_with_quorum(
        &mut self,
        reorder_pending: usize,
        quorum_timeout: Duration,
    ) -> Result<()> {
        anyhow::ensure!(
            !quorum_timeout.is_zero(),
            "Spark collective quorum timeout must be positive"
        );
        self.wait_for_turn_inner(reorder_pending, Some(quorum_timeout))
    }

    fn wait_for_turn_inner(
        &mut self,
        reorder_pending: usize,
        quorum_timeout: Option<Duration>,
    ) -> Result<()> {
        if self.owns_turn {
            return Ok(());
        }
        anyhow::ensure!(
            reorder_pending > 0,
            "Spark collective reorder window requires at least one pending request"
        );
        let quorum_deadline =
            quorum_timeout.and_then(|timeout| Instant::now().checked_add(timeout));
        let mut state = self
            .order
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("Spark collective launch-order lock is poisoned"))?;
        loop {
            anyhow::ensure!(
                state.pending.contains(&self.request_id),
                "Spark collective request {} disappeared while waiting for launch order",
                self.request_id
            );
            if state.active.is_none() {
                let expected_is_pending = state
                    .expected
                    .is_some_and(|expected| state.pending.contains(&expected));
                if !expected_is_pending {
                    let wait_started = *state.gap_wait_started.get_or_insert_with(Instant::now);
                    let quorum_ready = state.pending.len() >= reorder_pending;
                    let legacy_gap_ready = quorum_timeout.is_none()
                        && collective_gap_ready(
                            state.pending.len(),
                            reorder_pending,
                            wait_started.elapsed(),
                            self.order.reorder_wait,
                        );
                    if quorum_ready || legacy_gap_ready {
                        state.expected = state.pending.first().copied();
                        state.gap_wait_started = None;
                        self.order.changed.notify_all();
                    }
                } else {
                    state.gap_wait_started = None;
                }
                if state.expected == Some(self.request_id) {
                    state.active = Some(self.request_id);
                    self.owns_turn = true;
                    return Ok(());
                }
            }

            if let Some(deadline) = quorum_deadline {
                if Instant::now() >= deadline {
                    anyhow::bail!(
                        "timed out waiting for Spark collective request {} quorum: pending={} required={reorder_pending}",
                        self.request_id,
                        state.pending.len()
                    );
                }
            }

            let mut wait_for = state
                .gap_wait_started
                .filter(|_| quorum_timeout.is_none())
                .map(|started| self.order.reorder_wait.saturating_sub(started.elapsed()))
                .filter(|remaining| !remaining.is_zero())
                .unwrap_or(self.order.reorder_wait);
            if let Some(deadline) = quorum_deadline {
                wait_for = wait_for.min(deadline.saturating_duration_since(Instant::now()));
            }
            let (next_state, _) = self
                .order
                .changed
                .wait_timeout(state, wait_for)
                .map_err(|_| anyhow::anyhow!("Spark collective launch-order lock is poisoned"))?;
            state = next_state;
        }
    }

    pub(in crate::commands::real_full) fn finish(&mut self) -> Result<()> {
        anyhow::ensure!(
            self.owns_turn && !self.finished,
            "Spark collective request {} finished without owning the launch turn",
            self.request_id
        );
        let mut state = self
            .order
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("Spark collective launch-order lock is poisoned"))?;
        anyhow::ensure!(
            state.active == Some(self.request_id) && state.pending.remove(&self.request_id),
            "Spark collective request {} lost its active launch turn",
            self.request_id
        );
        state.active = None;
        state.expected = self.request_id.checked_add(SPARK_COLLECTIVE_REQUEST_STRIDE);
        state.gap_wait_started = None;
        self.owns_turn = false;
        self.finished = true;
        self.order.changed.notify_all();
        Ok(())
    }
}

impl Drop for SparkCollectiveLaunchTicket {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        let Ok(mut state) = self.order.state.lock() else {
            return;
        };
        state.pending.remove(&self.request_id);
        if state.active == Some(self.request_id) {
            state.active = None;
        }
        if state.expected == Some(self.request_id) {
            state.expected = None;
        }
        state.gap_wait_started = None;
        self.order.changed.notify_all();
    }
}

fn spark_collective_launch_order() -> &'static Arc<SparkCollectiveLaunchOrder> {
    static ORDER: OnceLock<Arc<SparkCollectiveLaunchOrder>> = OnceLock::new();
    ORDER.get_or_init(|| {
        Arc::new(SparkCollectiveLaunchOrder::new(
            SPARK_COLLECTIVE_REORDER_WAIT,
        ))
    })
}

#[cfg(test)]
thread_local! {
    static CUDA_REFERENCE_KERNELS_TEST_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
    static CUDA_ROUTE_VALIDATION_TEST_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
    static ROUTE_CUDA_GRAPHS_TEST_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

#[derive(Clone)]
struct RoutedQuantProjection {
    weight: LoadedTensorRows,
    weight_scale: LoadedTensorRows,
    input_scale: LoadedTensor,
    weight_scale_2: LoadedTensor,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct RoutedQuantProjectionKey {
    layer_id: usize,
    expert_id: usize,
    projection: &'static str,
    row_count: usize,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct RoutedQuantScalarMetadataKey {
    layer_id: usize,
    expert_id: usize,
    projection: &'static str,
}

#[derive(Clone)]
struct RoutedQuantScalarMetadata {
    input_scale_name: String,
    weight_scale_2_name: String,
    input_scale: f32,
    weight_scale_2: f32,
}

#[derive(Clone)]
struct Bf16RouteProjection {
    key: RoutedQuantProjectionKey,
    host: Option<Arc<RoutedQuantProjection>>,
}

impl Bf16RouteProjection {
    fn host_projection(&self) -> Result<&RoutedQuantProjection> {
        self.host.as_deref().with_context(|| {
            format!(
                "CUDA NVFP4 route validation requires host tensors for layer {} expert {} {}",
                self.key.layer_id, self.key.expert_id, self.key.projection
            )
        })
    }
}

#[derive(Default)]
pub(in crate::commands::real_full) struct RouteTensorCache {
    projections: HashMap<RoutedQuantProjectionKey, Arc<RoutedQuantProjection>>,
    bf16_projection_groups: HashMap<Bf16RouteProjectionGroupCacheKey, Bf16RouteProjections>,
    scalar_metadata: HashMap<RoutedQuantScalarMetadataKey, RoutedQuantScalarMetadata>,
    projection_loads: usize,
    cache_hits: usize,
    projection_evictions: usize,
    scalar_metadata_loads: usize,
    scalar_metadata_cache_hits: usize,
    active_layer: Option<usize>,
    execution_lane: u32,
    cuda: Option<RouteCudaCache>,
}

#[derive(Clone, Copy, Debug, Default)]
#[allow(dead_code)]
pub(in crate::commands::real_full) struct RouteTensorCacheStats {
    pub(in crate::commands::real_full) entries: usize,
    pub(in crate::commands::real_full) projection_loads: usize,
    pub(in crate::commands::real_full) cache_hits: usize,
    pub(in crate::commands::real_full) projection_evictions: usize,
    pub(in crate::commands::real_full) scalar_metadata_entries: usize,
    pub(in crate::commands::real_full) scalar_metadata_loads: usize,
    pub(in crate::commands::real_full) scalar_metadata_cache_hits: usize,
    pub(in crate::commands::real_full) active_layer: Option<usize>,
    pub(in crate::commands::real_full) cuda_projection_entries: usize,
    pub(in crate::commands::real_full) cuda_projection_uploads: usize,
    pub(in crate::commands::real_full) cuda_cache_hits: usize,
    pub(in crate::commands::real_full) cuda_projection_evictions: usize,
    pub(in crate::commands::real_full) cuda_active_layer: Option<usize>,
    pub(in crate::commands::real_full) cuda_managed_projection_entries: usize,
    pub(in crate::commands::real_full) cuda_managed_projection_allocations_enabled: bool,
    pub(in crate::commands::real_full) cuda_graph_entries: usize,
    pub(in crate::commands::real_full) cuda_graph_captures: usize,
    pub(in crate::commands::real_full) cuda_graph_launches: usize,
}

impl RouteTensorCache {
    pub(in crate::commands::real_full) fn for_execution_lane(execution_lane: u32) -> Self {
        Self {
            execution_lane,
            ..Self::default()
        }
    }

    pub(in crate::commands::real_full) fn stats(&self) -> RouteTensorCacheStats {
        let (
            cuda_projection_entries,
            cuda_projection_uploads,
            cuda_cache_hits,
            cuda_projection_evictions,
            cuda_active_layer,
            cuda_managed_projection_entries,
            cuda_managed_projection_allocations_enabled,
            cuda_graph_entries,
            cuda_graph_captures,
            cuda_graph_launches,
        ) = self
            .cuda
            .as_ref()
            .map(|cuda| {
                let projection_entries = cuda
                    .expert_slabs
                    .values()
                    .map(|slab| slab.expert_count * 3)
                    .sum::<usize>()
                    + cuda
                        .ds4_flash_expert_slabs
                        .values()
                        .map(|slab| slab.expert_count * 3)
                        .sum::<usize>()
                    + cuda
                        .ds4_flash_exl3_expert_slabs
                        .values()
                        .map(|slab| slab.expert_count * 3)
                        .sum::<usize>()
                    + cuda
                        .ds4_flash_exl3_mixed_expert_slabs
                        .values()
                        .map(|_| DS4_FLASH_ROUTED_EXPERTS * 3)
                        .sum::<usize>();
                let managed_projection_entries = cuda
                    .expert_slabs
                    .values()
                    .map(|slab| slab.managed_projection_entries())
                    .sum();
                (
                    projection_entries,
                    cuda.projection_uploads,
                    0,
                    0,
                    cuda.active_layer,
                    managed_projection_entries,
                    false,
                    cuda.packed_w4a16_decode_graphs.len()
                        + cuda.packed_w4a16_stream_decode_graphs.len(),
                    cuda.graph_captures,
                    cuda.graph_launches,
                )
            })
            .unwrap_or((0, 0, 0, 0, None, 0, false, 0, 0, 0));
        RouteTensorCacheStats {
            entries: self.projections.len(),
            projection_loads: self.projection_loads,
            cache_hits: self.cache_hits,
            projection_evictions: self.projection_evictions,
            scalar_metadata_entries: self.scalar_metadata.len(),
            scalar_metadata_loads: self.scalar_metadata_loads,
            scalar_metadata_cache_hits: self.scalar_metadata_cache_hits,
            active_layer: self.active_layer,
            cuda_projection_entries,
            cuda_projection_uploads,
            cuda_cache_hits,
            cuda_projection_evictions,
            cuda_active_layer,
            cuda_managed_projection_entries,
            cuda_managed_projection_allocations_enabled,
            cuda_graph_entries,
            cuda_graph_captures,
            cuda_graph_launches,
        }
    }

    fn cuda_cache(&mut self) -> Result<&mut RouteCudaCache> {
        if self.cuda.is_none() {
            self.cuda = Some(RouteCudaCache::new(self.execution_lane)?);
        }
        Ok(self.cuda.as_mut().expect("initialized above"))
    }

    fn cuda_cache_native_flash_tp4(
        &mut self,
        shard: ExpertIntermediateShard,
    ) -> Result<&mut RouteCudaCache> {
        if self.cuda.is_none() {
            self.cuda = Some(RouteCudaCache::new_native_flash_tp4(
                self.execution_lane,
                shard,
            )?);
        }
        let cuda = self.cuda.as_mut().expect("initialized above");
        anyhow::ensure!(
            cuda.intermediate_shard == Some(shard)
                && !cuda.legacy_spark_runtime_initialization_enabled
                && (cuda.spark_reduction.is_none() || cuda.spark_rdma_reduction.is_none()),
            "native Flash CUDA cache lane {} was initialized with a different TP shard or Spark-side reduction policy",
            self.execution_lane
        );
        Ok(cuda)
    }

    pub(in crate::commands::real_full) fn fork_execution_lane(
        &self,
        execution_lane: u32,
    ) -> Result<Self> {
        Ok(Self {
            projections: self.projections.clone(),
            bf16_projection_groups: self.bf16_projection_groups.clone(),
            scalar_metadata: self.scalar_metadata.clone(),
            projection_loads: 0,
            cache_hits: 0,
            projection_evictions: 0,
            scalar_metadata_loads: 0,
            scalar_metadata_cache_hits: 0,
            active_layer: self.active_layer,
            execution_lane,
            cuda: self
                .cuda
                .as_ref()
                .map(|cuda| cuda.fork_execution_lane(execution_lane))
                .transpose()?,
        })
    }

    fn prepare_layer(&mut self, layer_id: usize) {
        self.active_layer = Some(layer_id);
    }
}

#[derive(Clone, Copy, Debug)]
pub(in crate::commands::real_full) struct RouteHostProjectionPreload {
    pub(in crate::commands::real_full) weight_bytes: u64,
    pub(in crate::commands::real_full) quant_metadata_bytes: u64,
}

#[derive(Clone, Copy, Debug)]
pub(in crate::commands::real_full) struct RouteProjectionCachePreloadRequest {
    pub(in crate::commands::real_full) layer_id: usize,
    pub(in crate::commands::real_full) expert_id: usize,
    pub(in crate::commands::real_full) projection: &'static str,
    pub(in crate::commands::real_full) row_count: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub(in crate::commands::real_full) struct RouteCudaProjectionPreload {
    pub(in crate::commands::real_full) projection_groups: usize,
    pub(in crate::commands::real_full) weight_bytes: u64,
    pub(in crate::commands::real_full) weight_scale_bytes: u64,
}

struct RouteCudaCache {
    library: Arc<NativeLibrary>,
    intermediate_shard: Option<ExpertIntermediateShard>,
    legacy_spark_runtime_initialization_enabled: bool,
    spark_reduction: Option<SparkExpertReduction>,
    spark_rdma_reduction: Option<SparkExpertRdmaReduction>,
    weight_preload_communicator: Option<Ds41rtNcclComm>,
    stream: RouteCudaStream,
    b12x_aux_streams: Vec<RouteCudaStream>,
    b12x_lane_events: Vec<RouteCudaEvent>,
    completion_stream: RouteCudaStream,
    metadata_ready_event: RouteCudaEvent,
    expert_slabs: HashMap<usize, Arc<RouteCudaLayerExpertSlab>>,
    ds4_flash_expert_slabs: HashMap<usize, Arc<RouteCudaDs4FlashLayerExpertSlab>>,
    ds4_flash_exl3_expert_slabs: HashMap<usize, Arc<RouteCudaDs4FlashExl3LayerExpertSlab>>,
    ds4_flash_exl3_mixed_expert_slabs:
        HashMap<usize, Arc<RouteCudaDs4FlashExl3MixedLayerExpertSlab>>,
    bf16_expert_slabs: HashMap<usize, Arc<RouteCudaBf16LayerExpertSlab>>,
    projection_uploads: usize,
    active_layer: Option<usize>,
    workspace: RouteCudaWorkspace,
    b12x_aux_workspaces: Vec<RouteCudaWorkspace>,
    packed_w4a16_decode_graphs: HashMap<PackedW4a16DecodeCudaGraphKey, CapturedRouteCudaGraph>,
    packed_w4a16_stream_decode_graphs:
        HashMap<PackedW4a16StreamDecodeCudaGraphKey, CapturedRouteCudaGraph>,
    graph_captures: usize,
    graph_launches: usize,
    b12x_aot_enabled: bool,
    b12x_w4a16_packed: bool,
    ds4_flash_aot_enabled: bool,
    ds4_pro_aot_enabled: bool,
    grouped_decode_observed: bool,
}

impl RouteCudaCache {
    fn new(execution_lane: u32) -> Result<Self> {
        let shard = spark_expert_intermediate_shard_from_env()?;
        Self::new_with_tp_policy(execution_lane, shard, true)
    }

    fn new_native_flash_tp4(execution_lane: u32, shard: ExpertIntermediateShard) -> Result<Self> {
        anyhow::ensure!(
            shard.count == 4,
            "native Flash CUDA cache requires TP=4, got TP={}",
            shard.count
        );
        let mut cache = Self::new_with_tp_policy(execution_lane, Some(shard), false)?;
        cache.spark_reduction = initialize_spark_expert_reduction_lane(
            Arc::clone(&cache.library),
            Some(shard),
            execution_lane,
        )?;
        cache.spark_rdma_reduction =
            initialize_spark_expert_rdma_reduction_lane(Some(shard), execution_lane)?;
        anyhow::ensure!(
            cache.spark_reduction.is_none() || cache.spark_rdma_reduction.is_none(),
            "native Flash route lane {execution_lane} initialized both NCCL and RDMA reduction"
        );
        Ok(cache)
    }

    fn new_with_tp_policy(
        execution_lane: u32,
        shard: Option<ExpertIntermediateShard>,
        legacy_spark_runtime_initialization_enabled: bool,
    ) -> Result<Self> {
        let path = native_library_path().with_context(|| {
            format!(
                "{REAL_FULL_CUDA_REFERENCE_KERNELS_ENV}=1 requires DS41RT_NATIVE_LIB or native/build-cuda/libds41rt_native.so"
            )
        })?;
        let library = unsafe { NativeLibrary::load(&path) }.with_context(|| {
            format!(
                "loading native CUDA NVFP4 routed expert library {}",
                path.display()
            )
        })?;
        require_cuda_enabled_native_library(
            &library,
            &path,
            "real-full NVFP4 routed expert CUDA kernels",
        )?;
        let b12x_aot_enabled = legacy_spark_runtime_initialization_enabled
            && b12x_spark_direct_route_requested()
            && library
                .cuda_b12x_spark_aot_available()
                .context("querying Spark B12X AOT availability")?;
        if b12x_aot_enabled {
            library
                .cuda_b12x_spark_aot_init()
                .context("initializing Spark B12X AOT kernels")?;
        }
        let b12x_w4a16_packed = b12x_aot_enabled;
        let library = Arc::new(library);
        let stream = RouteCudaStream::new(Arc::clone(&library))?;
        let completion_stream = RouteCudaStream::new(Arc::clone(&library))?;
        let metadata_ready_event = RouteCudaEvent::new(Arc::clone(&library))?;
        let spark_reduction = legacy_spark_runtime_initialization_enabled
            .then(|| {
                initialize_spark_expert_reduction_lane(Arc::clone(&library), shard, execution_lane)
            })
            .transpose()?
            .flatten();
        let spark_rdma_reduction = legacy_spark_runtime_initialization_enabled
            .then(|| initialize_spark_expert_rdma_reduction_lane(shard, execution_lane))
            .transpose()?
            .flatten();
        anyhow::ensure!(
            spark_reduction.is_none() || spark_rdma_reduction.is_none(),
            "Spark route lane {execution_lane} initialized both NCCL and RDMA reduction"
        );
        let cooperative_weight_preload = env::var(REAL_FULL_NVFP4_ROUTE_PRELOAD_COOPERATIVE_ENV)
            .map(|value| {
                !matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "0" | "false" | "no" | "off" | "disabled"
                )
            })
            .unwrap_or(true);
        let weight_preload_communicator = if b12x_w4a16_packed && cooperative_weight_preload {
            initialize_spark_weight_preload_communicator(&library, shard)?
        } else {
            None
        };
        let b12x_lane_count = if b12x_aot_enabled {
            b12x_spark_route_lane_count()
        } else {
            1
        };
        let b12x_aux_streams = (1..b12x_lane_count)
            .map(|_| RouteCudaStream::new(Arc::clone(&library)))
            .collect::<Result<Vec<_>>>()?;
        let b12x_lane_events = (0..b12x_lane_count)
            .map(|_| RouteCudaEvent::new(Arc::clone(&library)))
            .collect::<Result<Vec<_>>>()?;
        let b12x_aux_workspaces = (1..b12x_lane_count)
            .map(|_| RouteCudaWorkspace::default())
            .collect();
        Ok(Self {
            library,
            intermediate_shard: shard,
            legacy_spark_runtime_initialization_enabled,
            spark_reduction,
            spark_rdma_reduction,
            weight_preload_communicator,
            stream,
            b12x_aux_streams,
            b12x_lane_events,
            completion_stream,
            metadata_ready_event,
            expert_slabs: HashMap::new(),
            ds4_flash_expert_slabs: HashMap::new(),
            ds4_flash_exl3_expert_slabs: HashMap::new(),
            ds4_flash_exl3_mixed_expert_slabs: HashMap::new(),
            bf16_expert_slabs: HashMap::new(),
            projection_uploads: 0,
            active_layer: None,
            workspace: RouteCudaWorkspace::default(),
            b12x_aux_workspaces,
            packed_w4a16_decode_graphs: HashMap::new(),
            packed_w4a16_stream_decode_graphs: HashMap::new(),
            graph_captures: 0,
            graph_launches: 0,
            b12x_aot_enabled,
            b12x_w4a16_packed,
            ds4_flash_aot_enabled: false,
            ds4_pro_aot_enabled: false,
            grouped_decode_observed: false,
        })
    }

    fn fork_execution_lane(&self, execution_lane: u32) -> Result<Self> {
        let b12x_lane_count = self.b12x_lane_count();
        let stream = RouteCudaStream::new(Arc::clone(&self.library))?;
        let completion_stream = RouteCudaStream::new(Arc::clone(&self.library))?;
        let metadata_ready_event = RouteCudaEvent::new(Arc::clone(&self.library))?;
        let shard = self.intermediate_shard;
        let initialize_reduction =
            self.legacy_spark_runtime_initialization_enabled || self.intermediate_shard.is_some();
        let spark_reduction = initialize_reduction
            .then(|| {
                initialize_spark_expert_reduction_lane(
                    Arc::clone(&self.library),
                    shard,
                    execution_lane,
                )
            })
            .transpose()?
            .flatten();
        let spark_rdma_reduction = initialize_reduction
            .then(|| initialize_spark_expert_rdma_reduction_lane(shard, execution_lane))
            .transpose()?
            .flatten();
        anyhow::ensure!(
            spark_reduction.is_none() || spark_rdma_reduction.is_none(),
            "Spark route lane {execution_lane} initialized both NCCL and RDMA reduction"
        );
        let b12x_aux_streams = (1..b12x_lane_count)
            .map(|_| RouteCudaStream::new(Arc::clone(&self.library)))
            .collect::<Result<Vec<_>>>()?;
        let b12x_lane_events = (0..b12x_lane_count)
            .map(|_| RouteCudaEvent::new(Arc::clone(&self.library)))
            .collect::<Result<Vec<_>>>()?;
        let b12x_aux_workspaces = (1..b12x_lane_count)
            .map(|_| RouteCudaWorkspace::default())
            .collect();
        Ok(Self {
            library: Arc::clone(&self.library),
            intermediate_shard: shard,
            legacy_spark_runtime_initialization_enabled: self
                .legacy_spark_runtime_initialization_enabled,
            spark_reduction,
            spark_rdma_reduction,
            weight_preload_communicator: None,
            stream,
            b12x_aux_streams,
            b12x_lane_events,
            completion_stream,
            metadata_ready_event,
            expert_slabs: self.expert_slabs.clone(),
            ds4_flash_expert_slabs: self.ds4_flash_expert_slabs.clone(),
            ds4_flash_exl3_expert_slabs: self.ds4_flash_exl3_expert_slabs.clone(),
            ds4_flash_exl3_mixed_expert_slabs: self.ds4_flash_exl3_mixed_expert_slabs.clone(),
            bf16_expert_slabs: self.bf16_expert_slabs.clone(),
            projection_uploads: 0,
            active_layer: self.active_layer,
            workspace: RouteCudaWorkspace::default(),
            b12x_aux_workspaces,
            packed_w4a16_decode_graphs: HashMap::new(),
            packed_w4a16_stream_decode_graphs: HashMap::new(),
            graph_captures: 0,
            graph_launches: 0,
            b12x_aot_enabled: self.b12x_aot_enabled,
            b12x_w4a16_packed: self.b12x_w4a16_packed,
            ds4_flash_aot_enabled: self.ds4_flash_aot_enabled,
            ds4_pro_aot_enabled: self.ds4_pro_aot_enabled,
            grouped_decode_observed: false,
        })
    }

    fn b12x_lane_count(&self) -> usize {
        self.b12x_aux_streams.len() + 1
    }

    fn ensure_ds4_flash_aot(&mut self) -> Result<()> {
        if self.ds4_flash_aot_enabled {
            return Ok(());
        }
        anyhow::ensure!(
            self.library
                .cuda_ds4_flash_spark_aot_available()
                .context("querying DeepSeek V4 Flash Spark AOT availability")?,
            "native library does not contain the DeepSeek V4 Flash Spark AOT module"
        );
        self.library
            .cuda_ds4_flash_spark_aot_init()
            .context("initializing DeepSeek V4 Flash Spark AOT kernels")?;
        self.ds4_flash_aot_enabled = true;
        Ok(())
    }

    fn ensure_ds4_exl3_aot(&mut self, variant: ModelVariant) -> Result<()> {
        match variant {
            ModelVariant::Flash => self.ensure_ds4_flash_aot(),
            ModelVariant::Pro => {
                if self.ds4_pro_aot_enabled {
                    return Ok(());
                }
                anyhow::ensure!(
                    self.library
                        .cuda_ds4_pro_spark_aot_available()
                        .context("querying DeepSeek V4 Pro Spark AOT availability")?,
                    "native library does not contain the DeepSeek V4 Pro Spark AOT module"
                );
                self.library
                    .cuda_ds4_pro_spark_aot_init()
                    .context("initializing DeepSeek V4 Pro Spark AOT kernels")?;
                self.ds4_pro_aot_enabled = true;
                Ok(())
            }
            ModelVariant::Custom => anyhow::bail!("custom DeepSeek V4 EXL3 AOT is unsupported"),
        }
    }

    fn spark_reduction_dtype(&self) -> Option<ExpertIntermediateReductionDtype> {
        self.spark_reduction
            .as_ref()
            .map(|reduction| reduction.dtype)
            .or_else(|| {
                self.spark_rdma_reduction
                    .as_ref()
                    .map(|reduction| reduction.dtype)
            })
    }

    fn spark_reduction_world_size(&self) -> Option<usize> {
        self.spark_reduction
            .as_ref()
            .map(|reduction| reduction.communicator().world_size())
            .or_else(|| {
                self.spark_rdma_reduction
                    .as_ref()
                    .map(SparkExpertRdmaReduction::world_size)
            })
    }

    fn spark_reduction_enabled_for_rows(&self, rows: usize) -> bool {
        self.spark_reduction
            .as_ref()
            .is_some_and(|reduction| reduction.enabled_for_rows(rows))
            || self
                .spark_rdma_reduction
                .as_ref()
                .is_some_and(|reduction| reduction.enabled_for_rows(rows))
    }

    fn spark_rdma_reduction_enabled(&self) -> bool {
        self.spark_rdma_reduction.is_some()
    }

    fn b12x_lane_stream(&self, lane: usize) -> *mut c_void {
        if lane == 0 {
            self.stream.as_ptr()
        } else {
            self.b12x_aux_streams[lane - 1].as_ptr()
        }
    }

    fn b12x_lane_event(&self, lane: usize) -> *mut c_void {
        self.b12x_lane_events[lane].as_ptr()
    }

    fn ensure_b12x_route_workspaces(
        &mut self,
        lane_count: usize,
        rows: usize,
        hidden_dim: usize,
        intermediate_dim: usize,
        output_dim: usize,
    ) -> Result<Vec<B12xSparkAotRouteWorkspaceBuffers>> {
        anyhow::ensure!(
            lane_count > 0 && lane_count <= self.b12x_lane_count(),
            "B12X route lane count {lane_count} exceeds configured lanes {}",
            self.b12x_lane_count()
        );
        let mut workspaces = Vec::with_capacity(lane_count);
        workspaces.push(self.workspace.ensure_b12x_aot_route_buffers(
            Arc::clone(&self.library),
            rows,
            hidden_dim,
            intermediate_dim,
            output_dim,
        )?);
        for workspace in self.b12x_aux_workspaces.iter_mut().take(lane_count - 1) {
            workspaces.push(workspace.ensure_b12x_aot_route_buffers(
                Arc::clone(&self.library),
                rows,
                hidden_dim,
                intermediate_dim,
                output_dim,
            )?);
        }
        Ok(workspaces)
    }

    fn prepare_layer(&mut self, layer_id: usize) {
        self.active_layer = Some(layer_id);
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_or_capture_packed_w4a16_decode_graph(
        &mut self,
        layer_id: usize,
        workspace: RouteCudaAccumulationWorkspaceBuffers,
        pinned_payloads: RouteCudaPinnedPayloadBuffers,
        route_metadata_payload: Ds41rtHostBuffer,
        layer_buffers: B12xSparkW4a16LayerBuffers,
        b12x_workspace: B12xSparkAotRouteWorkspaceBuffers,
        input_payload_stride_bytes: usize,
        hidden_bytes: usize,
        route_weight_bytes: usize,
        route_metadata_bytes: usize,
        output_rows: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let key = PackedW4a16DecodeCudaGraphKey::new(
            layer_id,
            workspace,
            pinned_payloads,
            route_metadata_payload,
            layer_buffers,
            b12x_workspace,
            input_payload_stride_bytes,
            hidden_bytes,
            route_weight_bytes,
            route_metadata_bytes,
            output_rows,
        );
        if let Some(graph) = self.packed_w4a16_decode_graphs.get(&key) {
            self.library
                .cuda_graph_launch(graph.graph_exec, cuda_stream)
                .context("launching captured packed W4A16 decode graph")?;
            self.graph_launches += 1;
            return Ok(());
        }

        self.library
            .cuda_graph_begin_capture(cuda_stream)
            .context("beginning packed W4A16 decode graph capture")?;
        enqueue_packed_w4a16_decode_graph_ops(
            &self.library,
            workspace,
            pinned_payloads,
            route_metadata_payload,
            layer_buffers,
            b12x_workspace,
            input_payload_stride_bytes,
            hidden_bytes,
            route_weight_bytes,
            route_metadata_bytes,
            output_rows,
            cuda_stream,
        )?;
        let capture = self
            .library
            .cuda_graph_end_capture_retained(cuda_stream)
            .context("ending packed W4A16 decode graph capture")?;
        let graph = CapturedRouteCudaGraph::new_with_node_requirements(
            Arc::clone(&self.library),
            capture,
            true,
            "packed W4A16 decode graph",
        )?;
        self.packed_w4a16_decode_graphs.insert(key, graph);
        self.graph_captures += 1;
        let graph = self
            .packed_w4a16_decode_graphs
            .get(&key)
            .expect("captured packed W4A16 decode graph was inserted");
        self.library
            .cuda_graph_launch(graph.graph_exec, cuda_stream)
            .context("launching newly captured packed W4A16 decode graph")?;
        self.graph_launches += 1;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_or_capture_packed_w4a16_stream_decode_graph(
        &mut self,
        layer_id: usize,
        workspace: RouteCudaAccumulationWorkspaceBuffers,
        layer_buffers: B12xSparkW4a16LayerBuffers,
        b12x_workspace: B12xSparkAotRouteWorkspaceBuffers,
        input_payload_stride_bytes: usize,
        row_count: usize,
        output_rows: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let key = PackedW4a16StreamDecodeCudaGraphKey::new(
            layer_id,
            workspace,
            layer_buffers,
            b12x_workspace,
            input_payload_stride_bytes,
            row_count,
            output_rows,
        );
        if let Some(graph) = self.packed_w4a16_stream_decode_graphs.get(&key) {
            self.library
                .cuda_graph_launch(graph.graph_exec, cuda_stream)
                .context("launching captured streamed packed W4A16 decode graph")?;
            self.graph_launches += 1;
            return Ok(());
        }

        self.library
            .cuda_graph_begin_capture(cuda_stream)
            .context("beginning streamed packed W4A16 decode graph capture")?;
        enqueue_packed_w4a16_stream_decode_graph_ops(
            &self.library,
            workspace,
            layer_buffers,
            b12x_workspace,
            input_payload_stride_bytes,
            row_count,
            output_rows,
            cuda_stream,
        )?;
        let capture = self
            .library
            .cuda_graph_end_capture_retained(cuda_stream)
            .context("ending streamed packed W4A16 decode graph capture")?;
        let graph = CapturedRouteCudaGraph::new_with_node_requirements(
            Arc::clone(&self.library),
            capture,
            false,
            "streamed packed W4A16 decode graph",
        )?;
        self.packed_w4a16_stream_decode_graphs.insert(key, graph);
        self.graph_captures += 1;
        let graph = self
            .packed_w4a16_stream_decode_graphs
            .get(&key)
            .expect("captured streamed packed W4A16 decode graph was inserted");
        self.library
            .cuda_graph_launch(graph.graph_exec, cuda_stream)
            .context("launching newly captured streamed packed W4A16 decode graph")?;
        self.graph_launches += 1;
        Ok(())
    }
}

struct RouteCudaStream {
    library: Arc<NativeLibrary>,
    stream: *mut c_void,
}

impl RouteCudaStream {
    fn new(library: Arc<NativeLibrary>) -> Result<Self> {
        let stream = library
            .cuda_stream_create()
            .context("creating NVFP4 route CUDA stream")?;
        Ok(Self { library, stream })
    }

    fn as_ptr(&self) -> *mut c_void {
        self.stream
    }
}

impl Drop for RouteCudaStream {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            let _ = unsafe { self.library.cuda_stream_destroy(self.stream) };
            self.stream = std::ptr::null_mut();
        }
    }
}

// RouteCudaStream is only used while holding the route-cache mutex; moving the
// cache between request/runtime threads preserves single-stream ownership.
unsafe impl Send for RouteCudaStream {}

struct RouteCudaEvent {
    library: Arc<NativeLibrary>,
    event: *mut c_void,
}

impl RouteCudaEvent {
    fn new(library: Arc<NativeLibrary>) -> Result<Self> {
        let event = library
            .cuda_event_create()
            .context("creating NVFP4 route CUDA event")?;
        Ok(Self { library, event })
    }

    fn as_ptr(&self) -> *mut c_void {
        self.event
    }
}

impl Drop for RouteCudaEvent {
    fn drop(&mut self) {
        if !self.event.is_null() {
            let _ = unsafe { self.library.cuda_event_destroy(self.event) };
            self.event = std::ptr::null_mut();
        }
    }
}

unsafe impl Send for RouteCudaEvent {}

#[derive(Default)]
struct RouteCudaEventTimeline {
    start: Option<RouteCudaEvent>,
    hidden_ready: Option<RouteCudaEvent>,
    metadata_ready: Option<RouteCudaEvent>,
    routes_ready: Option<RouteCudaEvent>,
    pack_ready: Option<RouteCudaEvent>,
    retained_ready: Option<RouteCudaEvent>,
    host_copy_ready: Option<RouteCudaEvent>,
}

#[derive(Default)]
struct RouteCudaEventElapsed {
    hidden_copy_ms: f64,
    metadata_copy_ms: f64,
    route_kernel_ms: f64,
    bf16_pack_ms: f64,
    retained_copy_ms: f64,
    host_copy_ms: f64,
    total_ms: f64,
}

impl RouteCudaEventTimeline {
    fn enabled() -> bool {
        env_flag_enabled(REAL_FULL_NVFP4_ROUTE_CUDA_EVENT_TIMING_ENV)
    }

    fn new(library: Arc<NativeLibrary>) -> Result<Self> {
        Ok(Self {
            start: Some(RouteCudaEvent::new(Arc::clone(&library))?),
            hidden_ready: Some(RouteCudaEvent::new(Arc::clone(&library))?),
            metadata_ready: Some(RouteCudaEvent::new(Arc::clone(&library))?),
            routes_ready: Some(RouteCudaEvent::new(Arc::clone(&library))?),
            pack_ready: Some(RouteCudaEvent::new(Arc::clone(&library))?),
            retained_ready: Some(RouteCudaEvent::new(Arc::clone(&library))?),
            host_copy_ready: Some(RouteCudaEvent::new(library)?),
        })
    }

    unsafe fn record(
        &self,
        event: Option<&RouteCudaEvent>,
        cuda_stream: *mut c_void,
        label: &'static str,
    ) -> Result<()> {
        if let Some(event) = event {
            self.event_library()
                .cuda_event_record(event.as_ptr(), cuda_stream)
                .with_context(|| format!("recording NVFP4 route CUDA event {label}"))?;
        }
        Ok(())
    }

    fn event_library(&self) -> &NativeLibrary {
        self.start
            .as_ref()
            .expect("route CUDA event timeline has a start event")
            .library
            .as_ref()
    }

    unsafe fn elapsed_between(
        &self,
        start: Option<&RouteCudaEvent>,
        end: Option<&RouteCudaEvent>,
        label: &'static str,
    ) -> Result<f64> {
        match (start, end) {
            (Some(start), Some(end)) => self
                .event_library()
                .cuda_event_elapsed_ms(start.as_ptr(), end.as_ptr())
                .map(|ms| ms as f64)
                .with_context(|| format!("reading NVFP4 route CUDA event elapsed {label}")),
            _ => Ok(0.0),
        }
    }

    unsafe fn elapsed(&self) -> Result<RouteCudaEventElapsed> {
        Ok(RouteCudaEventElapsed {
            hidden_copy_ms: self.elapsed_between(
                self.start.as_ref(),
                self.hidden_ready.as_ref(),
                "hidden copy",
            )?,
            metadata_copy_ms: self.elapsed_between(
                self.hidden_ready.as_ref(),
                self.metadata_ready.as_ref(),
                "metadata copy",
            )?,
            route_kernel_ms: self.elapsed_between(
                self.metadata_ready.as_ref(),
                self.routes_ready.as_ref(),
                "route kernels",
            )?,
            bf16_pack_ms: self.elapsed_between(
                self.routes_ready.as_ref(),
                self.pack_ready.as_ref(),
                "BF16 pack",
            )?,
            retained_copy_ms: self.elapsed_between(
                self.pack_ready.as_ref(),
                self.retained_ready.as_ref(),
                "retained copy",
            )?,
            host_copy_ms: self.elapsed_between(
                self.retained_ready.as_ref(),
                self.host_copy_ready.as_ref(),
                "host copy",
            )?,
            total_ms: self.elapsed_between(
                self.start.as_ref(),
                self.host_copy_ready.as_ref(),
                "total",
            )?,
        })
    }
}

#[derive(Default)]
struct RouteCudaWorkspace {
    hidden: Option<OwnedDeviceAllocation>,
    accumulator: Option<OwnedDeviceAllocation>,
    final_output: Option<OwnedDeviceAllocation>,
    scatter_index: Option<OwnedDeviceAllocation>,
    route_weights: Option<OwnedDeviceAllocation>,
    route_metadata: Option<OwnedDeviceAllocation>,
    completion_indices: Option<OwnedDeviceAllocation>,
    completion_f32: Option<OwnedDeviceAllocation>,
    completion_output: Option<OwnedDeviceAllocation>,
    completion_reduction_send: Option<OwnedDeviceAllocation>,
    completion_reduction_recv: Option<OwnedDeviceAllocation>,
    b12x_compact_hidden: Option<OwnedDeviceAllocation>,
    b12x_group_output: Option<OwnedDeviceAllocation>,
    b12x_w4a16_fc1_output: Option<OwnedDeviceAllocation>,
    b12x_w4a16_activated: Option<OwnedDeviceAllocation>,
    b12x_w4a16_packed_route_indices: Option<OwnedDeviceAllocation>,
    b12x_w4a16_block_expert_ids: Option<OwnedDeviceAllocation>,
    b12x_w4a16_packed_route_count: Option<OwnedDeviceAllocation>,
    b12x_w4a16_topk_weights: Option<OwnedDeviceAllocation>,
    b12x_w4a16_fc1_scratch: Option<OwnedDeviceAllocation>,
    b12x_w4a16_fc2_scratch: Option<OwnedDeviceAllocation>,
    b12x_w4a16_locks: Option<OwnedDeviceAllocation>,
    b12x_w4a16_pack_source: Option<OwnedDeviceAllocation>,
    ds4_flash_packed_hidden: Option<OwnedDeviceAllocation>,
    ds4_flash_fc1_output: Option<OwnedDeviceAllocation>,
    ds4_flash_activated: Option<OwnedDeviceAllocation>,
    ds4_flash_routed_output: Option<OwnedDeviceAllocation>,
    ds4_flash_rotation_gate: Option<OwnedDeviceAllocation>,
    ds4_flash_rotation_up: Option<OwnedDeviceAllocation>,
    ds4_flash_output_f32: Option<OwnedDeviceAllocation>,
    ds4_flash_block_expert_ids: Option<OwnedDeviceAllocation>,
    ds4_flash_packed_route_count: Option<OwnedDeviceAllocation>,
    ds4_flash_expert_counts: Option<OwnedDeviceAllocation>,
    ds4_flash_expert_offsets: Option<OwnedDeviceAllocation>,
    ds4_flash_fc1_scratch: Option<OwnedDeviceAllocation>,
    ds4_flash_fc2_scratch: Option<OwnedDeviceAllocation>,
    ds4_flash_locks: Option<OwnedDeviceAllocation>,
    startup_nvfp4_weight: Option<OwnedDeviceAllocation>,
    startup_nvfp4_scale: Option<OwnedDeviceAllocation>,
    pinned_hidden: Option<OwnedPinnedHostAllocation>,
    pinned_scatter_index: Option<OwnedPinnedHostAllocation>,
    pinned_route_weights: Option<OwnedPinnedHostAllocation>,
    pinned_route_metadata: Option<OwnedPinnedHostAllocation>,
    pinned_projection_weight: Option<OwnedPinnedHostAllocation>,
    pinned_output: Option<OwnedPinnedHostAllocation>,
    pinned_completion_indices: Option<OwnedPinnedHostAllocation>,
    pinned_completion_output: Option<OwnedPinnedHostAllocation>,
    completion_compute_events: Vec<RouteCudaEvent>,
    completion_ready_events: Vec<RouteCudaEvent>,
}

#[derive(Clone, Copy)]
struct RouteCudaAccumulationWorkspaceBuffers {
    hidden: Ds41rtDeviceBuffer,
    accumulator: Ds41rtDeviceBuffer,
    final_output: Ds41rtDeviceBuffer,
    scatter_index: Ds41rtDeviceBuffer,
    route_weights: Ds41rtDeviceBuffer,
    route_metadata: Ds41rtDeviceBuffer,
}

#[derive(Clone, Copy)]
struct RouteCudaPinnedPayloadBuffers {
    hidden: Ds41rtHostBuffer,
    scatter_index: Ds41rtHostBuffer,
    route_weights: Ds41rtHostBuffer,
}

#[derive(Clone, Copy)]
struct RouteCudaPinnedMetadataPayloadBuffers {
    scatter_index: Ds41rtHostBuffer,
    route_weights: Ds41rtHostBuffer,
}

#[derive(Clone, Copy)]
struct RouteCudaCompletionWorkspaceBuffers {
    indices: Ds41rtDeviceBuffer,
    f32_output: Ds41rtDeviceBuffer,
    output: Ds41rtDeviceBuffer,
    pinned_indices: Ds41rtHostBuffer,
    pinned_output: Ds41rtHostBuffer,
}

#[derive(Clone, Copy)]
struct RouteCudaReductionWorkspaceBuffers {
    send: Ds41rtDeviceBuffer,
    recv: Ds41rtDeviceBuffer,
}

struct RouteCudaStreamingCompletionPlan {
    buffers: RouteCudaCompletionWorkspaceBuffers,
    events: Vec<(*mut c_void, *mut c_void)>,
    slice_row_offsets: Vec<usize>,
}

#[derive(Clone, Copy)]
struct B12xSparkAotRouteWorkspaceBuffers {
    compact_hidden: Ds41rtDeviceBuffer,
    group_output: Ds41rtDeviceBuffer,
    w4a16_fc1_output: Ds41rtDeviceBuffer,
    w4a16_activated: Ds41rtDeviceBuffer,
    w4a16_packed_route_indices: Ds41rtDeviceBuffer,
    w4a16_block_expert_ids: Ds41rtDeviceBuffer,
    w4a16_packed_route_count: Ds41rtDeviceBuffer,
    w4a16_topk_weights: Ds41rtDeviceBuffer,
    w4a16_fc1_scratch: Ds41rtDeviceBuffer,
    w4a16_fc2_scratch: Ds41rtDeviceBuffer,
    w4a16_locks: Ds41rtDeviceBuffer,
}

#[derive(Clone, Copy)]
struct Ds4FlashSparkAotRouteWorkspaceBuffers {
    packed_hidden: Ds41rtDeviceBuffer,
    fc1_output: Ds41rtDeviceBuffer,
    activated: Ds41rtDeviceBuffer,
    routed_output: Ds41rtDeviceBuffer,
    rotation_gate: Ds41rtDeviceBuffer,
    rotation_up: Ds41rtDeviceBuffer,
    output_f32: Ds41rtDeviceBuffer,
    block_expert_ids: Ds41rtDeviceBuffer,
    packed_route_count: Ds41rtDeviceBuffer,
    expert_counts: Ds41rtDeviceBuffer,
    expert_offsets: Ds41rtDeviceBuffer,
    fc1_scratch: Ds41rtDeviceBuffer,
    fc2_scratch: Ds41rtDeviceBuffer,
    locks: Ds41rtDeviceBuffer,
}

/// Fixed caller-owned ingress, scratch, and BF16 rank partial for one Flash
/// decode row. Resident weights remain outside this per-lane arena. The same
/// six global expert IDs and weights are delivered to all four TP ranks.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) struct Ds4FlashSparkDecodeM1DeviceBuffers {
    pub(in crate::commands::real_full) input: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) route_indices: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) route_weights: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) fc1: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) activated: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) rotation_gate: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) rotation_up: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) routed_output: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) output_f32: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) output: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) block_expert_ids: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) packed_route_count: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) expert_counts: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) expert_offsets: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) fc1_scratch: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) fc2_scratch: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) locks: Ds41rtDeviceBuffer,
}

/// Fixed caller-owned metadata pack for Flash prefill. All four TP ranks
/// consume the same global top-k IDs and produce equivalent block-32 metadata.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) struct Ds4FlashSparkPrefillRoutePackDeviceBuffers {
    pub(in crate::commands::real_full) topk_ids: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) packed_route_indices: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) block_expert_ids: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) packed_route_count: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) expert_counts: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) expert_offsets: Ds41rtDeviceBuffer,
}

/// Fixed max-capacity Flash prefill ingress, route pack, compute scratch, and
/// BF16 rank partial. Resident expert weights remain external and every rank
/// owns the same shape for all 256 global experts.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) struct Ds4FlashSparkPrefillDeviceBuffers {
    pub(in crate::commands::real_full) input: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) topk_ids: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) topk_weights: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) packed_route_indices: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) block_expert_ids: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) packed_route_count: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) expert_counts: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) expert_offsets: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) fc1: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) activated: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) rotation_gate: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) rotation_up: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) routed_output: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) output_f32: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) output: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) fc1_scratch: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) fc2_scratch: Ds41rtDeviceBuffer,
    pub(in crate::commands::real_full) locks: Ds41rtDeviceBuffer,
}

impl Ds4FlashSparkPrefillDeviceBuffers {
    fn route_pack(self) -> Ds4FlashSparkPrefillRoutePackDeviceBuffers {
        Ds4FlashSparkPrefillRoutePackDeviceBuffers {
            topk_ids: self.topk_ids,
            packed_route_indices: self.packed_route_indices,
            block_expert_ids: self.block_expert_ids,
            packed_route_count: self.packed_route_count,
            expert_counts: self.expert_counts,
            expert_offsets: self.expert_offsets,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct PackedW4a16DecodeCudaGraphKey {
    layer_id: usize,
    pointers: [usize; 22],
    input_payload_stride_bytes: usize,
    hidden_bytes: usize,
    route_weight_bytes: usize,
    route_metadata_bytes: usize,
    output_rows: usize,
}

impl PackedW4a16DecodeCudaGraphKey {
    #[allow(clippy::too_many_arguments)]
    fn new(
        layer_id: usize,
        workspace: RouteCudaAccumulationWorkspaceBuffers,
        pinned_payloads: RouteCudaPinnedPayloadBuffers,
        route_metadata_payload: Ds41rtHostBuffer,
        layer: B12xSparkW4a16LayerBuffers,
        b12x: B12xSparkAotRouteWorkspaceBuffers,
        input_payload_stride_bytes: usize,
        hidden_bytes: usize,
        route_weight_bytes: usize,
        route_metadata_bytes: usize,
        output_rows: usize,
    ) -> Self {
        Self {
            layer_id,
            pointers: [
                workspace.hidden.ptr as usize,
                workspace.route_weights.ptr as usize,
                workspace.route_metadata.ptr as usize,
                pinned_payloads.hidden.ptr as usize,
                pinned_payloads.route_weights.ptr as usize,
                route_metadata_payload.ptr as usize,
                layer.w13_weight.ptr as usize,
                layer.w2_weight.ptr as usize,
                layer.w13_scale.ptr as usize,
                layer.w2_scale.ptr as usize,
                layer.w13_global_scale.ptr as usize,
                layer.w2_global_scale.ptr as usize,
                b12x.compact_hidden.ptr as usize,
                b12x.group_output.ptr as usize,
                b12x.w4a16_fc1_output.ptr as usize,
                b12x.w4a16_activated.ptr as usize,
                b12x.w4a16_packed_route_indices.ptr as usize,
                b12x.w4a16_block_expert_ids.ptr as usize,
                b12x.w4a16_packed_route_count.ptr as usize,
                b12x.w4a16_fc1_scratch.ptr as usize,
                b12x.w4a16_fc2_scratch.ptr as usize,
                b12x.w4a16_locks.ptr as usize,
            ],
            input_payload_stride_bytes,
            hidden_bytes,
            route_weight_bytes,
            route_metadata_bytes,
            output_rows,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct PackedW4a16StreamDecodeCudaGraphKey {
    layer_id: usize,
    pointers: [usize; 21],
    input_payload_stride_bytes: usize,
    row_count: usize,
    output_rows: usize,
}

impl PackedW4a16StreamDecodeCudaGraphKey {
    fn new(
        layer_id: usize,
        workspace: RouteCudaAccumulationWorkspaceBuffers,
        layer: B12xSparkW4a16LayerBuffers,
        b12x: B12xSparkAotRouteWorkspaceBuffers,
        input_payload_stride_bytes: usize,
        row_count: usize,
        output_rows: usize,
    ) -> Self {
        Self {
            layer_id,
            pointers: [
                workspace.hidden.ptr as usize,
                workspace.route_weights.ptr as usize,
                workspace.route_metadata.ptr as usize,
                workspace.scatter_index.ptr as usize,
                workspace.accumulator.ptr as usize,
                layer.w13_weight.ptr as usize,
                layer.w2_weight.ptr as usize,
                layer.w13_scale.ptr as usize,
                layer.w2_scale.ptr as usize,
                layer.w13_global_scale.ptr as usize,
                layer.w2_global_scale.ptr as usize,
                b12x.compact_hidden.ptr as usize,
                b12x.group_output.ptr as usize,
                b12x.w4a16_fc1_output.ptr as usize,
                b12x.w4a16_activated.ptr as usize,
                b12x.w4a16_packed_route_indices.ptr as usize,
                b12x.w4a16_block_expert_ids.ptr as usize,
                b12x.w4a16_packed_route_count.ptr as usize,
                b12x.w4a16_fc1_scratch.ptr as usize,
                b12x.w4a16_fc2_scratch.ptr as usize,
                b12x.w4a16_locks.ptr as usize,
            ],
            input_payload_stride_bytes,
            row_count,
            output_rows,
        }
    }
}

struct CapturedRouteCudaGraph {
    library: Arc<NativeLibrary>,
    graph: *mut c_void,
    graph_exec: *mut c_void,
}

impl CapturedRouteCudaGraph {
    fn new_with_node_requirements(
        library: Arc<NativeLibrary>,
        capture: ds41rt_ffi::Ds41rtCudaGraphCaptureInfo,
        require_memcpy: bool,
        label: &'static str,
    ) -> Result<Self> {
        if capture.graph.is_null() || capture.graph_exec.is_null() {
            anyhow::bail!("{label} capture returned a null graph handle");
        }
        if capture.kernel_node_count == 0 || (require_memcpy && capture.memcpy_node_count == 0) {
            anyhow::bail!(
                "{label} capture missing expected nodes: kernels={} memcpy={} total={}",
                capture.kernel_node_count,
                capture.memcpy_node_count,
                capture.node_count
            );
        }
        Ok(Self {
            library,
            graph: capture.graph,
            graph_exec: capture.graph_exec,
        })
    }
}

impl Drop for CapturedRouteCudaGraph {
    fn drop(&mut self) {
        if !self.graph_exec.is_null() {
            let _ = unsafe { self.library.cuda_graph_exec_destroy(self.graph_exec) };
            self.graph_exec = std::ptr::null_mut();
        }
        if !self.graph.is_null() {
            let _ = unsafe { self.library.cuda_graph_destroy(self.graph) };
            self.graph = std::ptr::null_mut();
        }
    }
}

// Captured CUDA graph handles are opaque native resources owned by the route
// cache mutex and destroyed through NativeLibrary.
unsafe impl Send for CapturedRouteCudaGraph {}

#[allow(clippy::too_many_arguments)]
unsafe fn enqueue_packed_w4a16_stream_decode_graph_ops(
    library: &NativeLibrary,
    workspace: RouteCudaAccumulationWorkspaceBuffers,
    layer_buffers: B12xSparkW4a16LayerBuffers,
    b12x_workspace: B12xSparkAotRouteWorkspaceBuffers,
    input_payload_stride_bytes: usize,
    row_count: usize,
    output_rows: usize,
    cuda_stream: *mut c_void,
) -> Result<()> {
    let input_payload = device_buffer_byte_view(
        workspace.hidden,
        0,
        input_payload_stride_bytes,
        "streamed packed W4A16 decode graph NVFP4 input",
    )?;
    let output = device_buffer_byte_view(
        b12x_workspace.group_output,
        0,
        output_rows * std::mem::size_of::<u16>(),
        "streamed packed W4A16 decode graph output",
    )?;
    let topk_ids = device_buffer_byte_view(
        workspace.route_metadata,
        0,
        8 * std::mem::size_of::<i32>(),
        "streamed packed W4A16 decode graph expert IDs",
    )?;
    let topk_weights = device_buffer_byte_view(
        workspace.route_weights,
        0,
        8 * std::mem::size_of::<f32>(),
        "streamed packed W4A16 decode graph route weights",
    )?;
    let output_index = device_buffer_byte_view(
        workspace.scatter_index,
        0,
        std::mem::size_of::<u32>(),
        "streamed packed W4A16 decode graph output index",
    )?;
    let buffers = b12x_w4a16_moe_buffers(
        layer_buffers,
        b12x_workspace,
        b12x_workspace.compact_hidden,
        output,
        topk_weights,
    );
    unsafe {
        launch_b12x_w4a16_decode(
            library,
            layer_buffers,
            &buffers,
            input_payload,
            input_payload_stride_bytes,
            topk_ids,
            cuda_stream,
        )
        .context("enqueueing streamed packed W4A16 decode graph MoE")?;
        library
            .cuda_scatter_add_rows_bf16_to_f32_async(
                output,
                output_index,
                workspace.accumulator,
                row_count,
                1,
                output_rows,
                cuda_stream,
            )
            .context("enqueueing streamed packed W4A16 decode graph accumulation")?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
unsafe fn enqueue_packed_w4a16_decode_graph_ops(
    library: &NativeLibrary,
    workspace: RouteCudaAccumulationWorkspaceBuffers,
    pinned_payloads: RouteCudaPinnedPayloadBuffers,
    route_metadata_payload: Ds41rtHostBuffer,
    layer_buffers: B12xSparkW4a16LayerBuffers,
    b12x_workspace: B12xSparkAotRouteWorkspaceBuffers,
    input_payload_stride_bytes: usize,
    hidden_bytes: usize,
    route_weight_bytes: usize,
    route_metadata_bytes: usize,
    output_rows: usize,
    cuda_stream: *mut c_void,
) -> Result<()> {
    let input_payload = device_buffer_byte_view(
        workspace.hidden,
        0,
        input_payload_stride_bytes,
        "packed W4A16 decode graph NVFP4 input",
    )?;
    let output = device_buffer_byte_view(
        b12x_workspace.group_output,
        0,
        output_rows * std::mem::size_of::<u16>(),
        "packed W4A16 decode graph output",
    )?;
    let topk_ids = device_buffer_byte_view(
        workspace.route_metadata,
        0,
        8 * std::mem::size_of::<i32>(),
        "packed W4A16 decode graph expert IDs",
    )?;
    let topk_weights = device_buffer_byte_view(
        workspace.route_weights,
        0,
        8 * std::mem::size_of::<f32>(),
        "packed W4A16 decode graph route weights",
    )?;
    let buffers = b12x_w4a16_moe_buffers(
        layer_buffers,
        b12x_workspace,
        b12x_workspace.compact_hidden,
        output,
        topk_weights,
    );
    unsafe {
        library
            .copy_host_buffer_h2d_async(
                workspace.hidden,
                pinned_payloads.hidden,
                hidden_bytes,
                cuda_stream,
            )
            .context("enqueueing packed W4A16 decode graph hidden copy")?;
        library
            .copy_host_buffer_h2d_async(
                workspace.route_weights,
                pinned_payloads.route_weights,
                route_weight_bytes,
                cuda_stream,
            )
            .context("enqueueing packed W4A16 decode graph route weights")?;
        library
            .copy_host_buffer_h2d_async(
                workspace.route_metadata,
                route_metadata_payload,
                route_metadata_bytes,
                cuda_stream,
            )
            .context("enqueueing packed W4A16 decode graph expert IDs")?;
        launch_b12x_w4a16_decode(
            library,
            layer_buffers,
            &buffers,
            input_payload,
            input_payload_stride_bytes,
            topk_ids,
            cuda_stream,
        )
        .context("enqueueing packed W4A16 decode graph MoE")?;
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum RouteCudaProjectionStageSlot {
    Weight,
}

impl RouteCudaWorkspace {
    fn ensure_ds4_flash_aot_buffers(
        &mut self,
        library: Arc<NativeLibrary>,
        rows: usize,
        exl3_profile: Option<Ds4Exl3AotProfile>,
        mixed_exl3: bool,
    ) -> Result<Ds4FlashSparkAotRouteWorkspaceBuffers> {
        let capacity_rows = ds4_flash_w4a16_capacity_rows(rows)?;
        let hidden_size = exl3_profile
            .map(|profile| profile.geometry.hidden_size)
            .unwrap_or(DS4_FLASH_HIDDEN_SIZE);
        let local_intermediate_size = exl3_profile
            .map(|profile| profile.geometry.local_intermediate_size)
            .unwrap_or(DS4_FLASH_LOCAL_INTERMEDIATE_SIZE);
        let experts = exl3_profile
            .map(|profile| profile.geometry.experts)
            .unwrap_or(DS4_FLASH_ROUTED_EXPERTS);
        let top_k = exl3_profile
            .map(|profile| profile.geometry.top_k)
            .unwrap_or(DS4_FLASH_TOP_K);
        let routed_rows = capacity_rows
            .checked_mul(top_k)
            .context("native DeepSeek V4 routed row count overflow")?;
        let (fc1_scratch_elements, fc2_scratch_elements) = if rows == 1 {
            exl3_profile.map_or(
                (
                    DS4_FLASH_FC1_SCRATCH_ELEMENTS,
                    DS4_FLASH_FC2_SCRATCH_ELEMENTS,
                ),
                |profile| {
                    (
                        profile.decode_fc1_scratch_elements,
                        profile.decode_fc2_scratch_elements,
                    )
                },
            )
        } else {
            exl3_profile.map_or(
                (
                    DS4_FLASH_PREFILL_SCRATCH_ELEMENTS,
                    DS4_FLASH_PREFILL_SCRATCH_ELEMENTS,
                ),
                |profile| {
                    (
                        profile.prefill_fc1_scratch_elements,
                        profile.prefill_fc2_scratch_elements,
                    )
                },
            )
        };
        let allocations = [
            (
                &mut self.ds4_flash_packed_hidden,
                capacity_rows * (hidden_size / 2 + hidden_size / 16),
                "native DeepSeek V4 packed hidden ingress",
            ),
            (
                &mut self.ds4_flash_fc1_output,
                routed_rows * 2 * local_intermediate_size * std::mem::size_of::<u16>(),
                "native DeepSeek V4 FC1 output",
            ),
            (
                &mut self.ds4_flash_activated,
                routed_rows * local_intermediate_size * std::mem::size_of::<u16>(),
                "native DeepSeek V4 activated output",
            ),
            (
                &mut self.ds4_flash_routed_output,
                routed_rows * hidden_size * std::mem::size_of::<u16>(),
                "native DeepSeek V4 routed output",
            ),
            (
                &mut self.ds4_flash_block_expert_ids,
                if rows == 1 {
                    (if mixed_exl3 {
                        top_k
                    } else {
                        exl3_profile.map_or(1, |profile| profile.decode_route_blocks)
                    }) * std::mem::size_of::<i32>()
                } else {
                    exl3_profile.map_or(DS4_FLASH_PREFILL_MAX_ROUTE_BLOCKS, |profile| {
                        profile.prefill_max_route_blocks
                    }) * std::mem::size_of::<i32>()
                },
                "native DeepSeek V4 block expert IDs",
            ),
            (
                &mut self.ds4_flash_packed_route_count,
                std::mem::size_of::<i32>(),
                "native DeepSeek V4 packed route count",
            ),
            (
                &mut self.ds4_flash_fc1_scratch,
                fc1_scratch_elements * std::mem::size_of::<f32>(),
                "native DeepSeek V4 FC1 scratch",
            ),
            (
                &mut self.ds4_flash_fc2_scratch,
                fc2_scratch_elements * std::mem::size_of::<f32>(),
                "native DeepSeek V4 FC2 scratch",
            ),
            (
                &mut self.ds4_flash_locks,
                DS4_FLASH_LOCK_ELEMENTS * std::mem::size_of::<i32>(),
                "native DeepSeek V4 locks",
            ),
        ];
        for (slot, bytes, label) in allocations {
            Self::ensure_buffer(slot, Arc::clone(&library), bytes, label)?;
        }
        if exl3_profile.is_some() || mixed_exl3 {
            let allocations = [
                (
                    &mut self.ds4_flash_rotation_gate,
                    routed_rows * hidden_size * std::mem::size_of::<u16>(),
                    "native DeepSeek V4 EXL3 gate rotation",
                ),
                (
                    &mut self.ds4_flash_rotation_up,
                    routed_rows * hidden_size * std::mem::size_of::<u16>(),
                    "native DeepSeek V4 EXL3 up rotation",
                ),
                (
                    &mut self.ds4_flash_output_f32,
                    capacity_rows * hidden_size * std::mem::size_of::<f32>(),
                    "native DeepSeek V4 EXL3 FP32 rank partial",
                ),
                (
                    &mut self.ds4_flash_expert_counts,
                    experts * std::mem::size_of::<i32>(),
                    "native DeepSeek V4 route expert counts",
                ),
                (
                    &mut self.ds4_flash_expert_offsets,
                    (experts + 1) * std::mem::size_of::<i32>(),
                    "native DeepSeek V4 route expert offsets",
                ),
            ];
            for (slot, bytes, label) in allocations {
                Self::ensure_buffer(slot, Arc::clone(&library), bytes, label)?;
            }
        }
        Ok(Ds4FlashSparkAotRouteWorkspaceBuffers {
            packed_hidden: self
                .ds4_flash_packed_hidden
                .as_ref()
                .expect("native DeepSeek V4 packed hidden ingress ensured")
                .buffer(),
            fc1_output: self
                .ds4_flash_fc1_output
                .as_ref()
                .expect("native Flash FC1 output ensured")
                .buffer(),
            activated: self
                .ds4_flash_activated
                .as_ref()
                .expect("native Flash activated output ensured")
                .buffer(),
            routed_output: self
                .ds4_flash_routed_output
                .as_ref()
                .expect("native Flash routed output ensured")
                .buffer(),
            rotation_gate: self
                .ds4_flash_rotation_gate
                .as_ref()
                .map(OwnedDeviceAllocation::buffer)
                .unwrap_or_default(),
            rotation_up: self
                .ds4_flash_rotation_up
                .as_ref()
                .map(OwnedDeviceAllocation::buffer)
                .unwrap_or_default(),
            output_f32: self
                .ds4_flash_output_f32
                .as_ref()
                .map(OwnedDeviceAllocation::buffer)
                .unwrap_or_default(),
            block_expert_ids: self
                .ds4_flash_block_expert_ids
                .as_ref()
                .expect("native Flash block expert IDs ensured")
                .buffer(),
            packed_route_count: self
                .ds4_flash_packed_route_count
                .as_ref()
                .expect("native Flash packed route count ensured")
                .buffer(),
            expert_counts: self
                .ds4_flash_expert_counts
                .as_ref()
                .map(OwnedDeviceAllocation::buffer)
                .unwrap_or_default(),
            expert_offsets: self
                .ds4_flash_expert_offsets
                .as_ref()
                .map(OwnedDeviceAllocation::buffer)
                .unwrap_or_default(),
            fc1_scratch: self
                .ds4_flash_fc1_scratch
                .as_ref()
                .expect("native Flash FC1 scratch ensured")
                .buffer(),
            fc2_scratch: self
                .ds4_flash_fc2_scratch
                .as_ref()
                .expect("native Flash FC2 scratch ensured")
                .buffer(),
            locks: self
                .ds4_flash_locks
                .as_ref()
                .expect("native Flash locks ensured")
                .buffer(),
        })
    }

    fn ensure_accumulation_buffers(
        &mut self,
        library: Arc<NativeLibrary>,
        hidden_bytes: usize,
        accumulator_bytes: usize,
        final_output_bytes: usize,
        scatter_index_bytes: usize,
        route_weight_bytes: usize,
        route_metadata_bytes: usize,
    ) -> Result<RouteCudaAccumulationWorkspaceBuffers> {
        Self::ensure_buffer(
            &mut self.hidden,
            Arc::clone(&library),
            hidden_bytes,
            "NVFP4 BF16 route hidden workspace",
        )?;
        Self::ensure_buffer(
            &mut self.accumulator,
            Arc::clone(&library),
            accumulator_bytes,
            "NVFP4 BF16 route F32 accumulator workspace",
        )?;
        Self::ensure_buffer(
            &mut self.final_output,
            Arc::clone(&library),
            final_output_bytes,
            "NVFP4 BF16 route final output workspace",
        )?;
        if self
            .scatter_index
            .as_ref()
            .map(|allocation| allocation.capacity_bytes() < scatter_index_bytes)
            .unwrap_or(true)
        {
            self.scatter_index = Some(OwnedDeviceAllocation::new(
                Arc::clone(&library),
                scatter_index_bytes,
                "NVFP4 BF16 route scatter index",
            )?);
        }
        Self::ensure_buffer(
            &mut self.route_weights,
            Arc::clone(&library),
            route_weight_bytes,
            "NVFP4 BF16 route weights",
        )?;
        Self::ensure_buffer(
            &mut self.route_metadata,
            library,
            route_metadata_bytes,
            "NVFP4 BF16 batched route metadata",
        )?;

        Ok(RouteCudaAccumulationWorkspaceBuffers {
            hidden: self
                .hidden
                .as_ref()
                .expect("hidden buffer ensured")
                .buffer(),
            accumulator: self
                .accumulator
                .as_ref()
                .expect("accumulator buffer ensured")
                .buffer(),
            final_output: self
                .final_output
                .as_ref()
                .expect("final output buffer ensured")
                .buffer(),
            scatter_index: self
                .scatter_index
                .as_ref()
                .expect("scatter index buffer ensured")
                .buffer(),
            route_weights: self
                .route_weights
                .as_ref()
                .expect("route weights buffer ensured")
                .buffer(),
            route_metadata: self
                .route_metadata
                .as_ref()
                .expect("route metadata buffer ensured")
                .buffer(),
        })
    }

    fn stage_accumulation_payloads(
        &mut self,
        library: Arc<NativeLibrary>,
        hidden: &[u8],
        scatter_index: &[u8],
        route_weights: &[u8],
    ) -> Result<RouteCudaPinnedPayloadBuffers> {
        let metadata = self.stage_accumulation_metadata_payloads(
            Arc::clone(&library),
            scatter_index,
            route_weights,
        )?;
        Ok(RouteCudaPinnedPayloadBuffers {
            hidden: Self::stage_pinned_bytes(
                &mut self.pinned_hidden,
                library,
                hidden,
                "NVFP4 BF16 route hidden pinned payload",
            )?,
            scatter_index: metadata.scatter_index,
            route_weights: metadata.route_weights,
        })
    }

    fn stage_accumulation_metadata_payloads(
        &mut self,
        library: Arc<NativeLibrary>,
        scatter_index: &[u8],
        route_weights: &[u8],
    ) -> Result<RouteCudaPinnedMetadataPayloadBuffers> {
        Ok(RouteCudaPinnedMetadataPayloadBuffers {
            scatter_index: Self::stage_pinned_bytes(
                &mut self.pinned_scatter_index,
                Arc::clone(&library),
                scatter_index,
                "NVFP4 BF16 route scatter-index pinned payload",
            )?,
            route_weights: Self::stage_pinned_bytes(
                &mut self.pinned_route_weights,
                library,
                route_weights,
                "NVFP4 BF16 route weight pinned payload",
            )?,
        })
    }

    fn stage_stream_hidden_payload(
        &mut self,
        library: Arc<NativeLibrary>,
        hidden: &[u8],
    ) -> Result<Ds41rtHostBuffer> {
        Self::stage_pinned_bytes(
            &mut self.pinned_hidden,
            library,
            hidden,
            "NVFP4 streamed route hidden payload",
        )
    }

    fn stage_stream_input_indices(
        &mut self,
        library: Arc<NativeLibrary>,
        input_indices: &[u32],
    ) -> Result<Ds41rtHostBuffer> {
        Self::stage_pinned_bytes(
            &mut self.pinned_route_metadata,
            library,
            u32_bytes(input_indices),
            "NVFP4 streamed route input indices",
        )
    }

    fn stage_route_metadata_payload(
        &mut self,
        library: Arc<NativeLibrary>,
        route_metadata: &[Ds41rtNvfp4RouteBatchedMetadata],
    ) -> Result<Ds41rtHostBuffer> {
        Self::stage_pinned_bytes(
            &mut self.pinned_route_metadata,
            library,
            route_metadata_bytes(route_metadata),
            "NVFP4 BF16 batched route metadata pinned payload",
        )
    }

    fn ensure_output_payload(
        &mut self,
        library: Arc<NativeLibrary>,
        bytes: usize,
    ) -> Result<Ds41rtHostBuffer> {
        Self::ensure_pinned_buffer(
            &mut self.pinned_output,
            library,
            bytes,
            "NVFP4 BF16 route output pinned payload",
        )?;
        Ok(self
            .pinned_output
            .as_ref()
            .expect("route output pinned buffer ensured")
            .buffer())
    }

    fn output_payload_slice(&mut self, bytes: usize) -> Result<&mut [u8]> {
        self.pinned_output
            .as_mut()
            .context("NVFP4 BF16 route output pinned payload missing")?
            .as_mut_slice(bytes)
    }

    fn ensure_completion_buffers(
        &mut self,
        library: Arc<NativeLibrary>,
        completion_indices: &[u32],
        f32_output_bytes: usize,
        output_bytes: usize,
    ) -> Result<RouteCudaCompletionWorkspaceBuffers> {
        let index_bytes = u32_bytes(completion_indices);
        Self::ensure_buffer(
            &mut self.completion_indices,
            Arc::clone(&library),
            index_bytes.len(),
            "NVFP4 route completion indices",
        )?;
        Self::ensure_buffer(
            &mut self.completion_f32,
            Arc::clone(&library),
            f32_output_bytes,
            "NVFP4 route completion F32 output",
        )?;
        Self::ensure_buffer(
            &mut self.completion_output,
            Arc::clone(&library),
            output_bytes,
            "NVFP4 route completion packed output",
        )?;
        let pinned_indices = Self::stage_pinned_bytes(
            &mut self.pinned_completion_indices,
            Arc::clone(&library),
            index_bytes,
            "NVFP4 route completion indices",
        )?;
        Self::ensure_pinned_buffer(
            &mut self.pinned_completion_output,
            library,
            output_bytes,
            "NVFP4 route completion output",
        )?;
        Ok(RouteCudaCompletionWorkspaceBuffers {
            indices: self
                .completion_indices
                .as_ref()
                .expect("completion index buffer ensured")
                .buffer(),
            f32_output: self
                .completion_f32
                .as_ref()
                .expect("completion F32 buffer ensured")
                .buffer(),
            output: self
                .completion_output
                .as_ref()
                .expect("completion output buffer ensured")
                .buffer(),
            pinned_indices,
            pinned_output: self
                .pinned_completion_output
                .as_ref()
                .expect("completion output pinned buffer ensured")
                .buffer(),
        })
    }

    fn ensure_completion_events(
        &mut self,
        library: Arc<NativeLibrary>,
        count: usize,
    ) -> Result<Vec<(*mut c_void, *mut c_void)>> {
        while self.completion_compute_events.len() < count {
            self.completion_compute_events
                .push(RouteCudaEvent::new(Arc::clone(&library))?);
            self.completion_ready_events
                .push(RouteCudaEvent::new(Arc::clone(&library))?);
        }
        Ok(self
            .completion_compute_events
            .iter()
            .zip(&self.completion_ready_events)
            .take(count)
            .map(|(compute, ready)| (compute.as_ptr(), ready.as_ptr()))
            .collect())
    }

    fn ensure_reduction_buffers(
        &mut self,
        library: Arc<NativeLibrary>,
        send_bytes: usize,
        recv_bytes: usize,
    ) -> Result<RouteCudaReductionWorkspaceBuffers> {
        Self::ensure_buffer(
            &mut self.completion_reduction_send,
            Arc::clone(&library),
            send_bytes,
            "Spark reduction send payload",
        )?;
        Self::ensure_buffer(
            &mut self.completion_reduction_recv,
            library,
            recv_bytes.max(1),
            "Spark reduction receive payloads",
        )?;
        Ok(RouteCudaReductionWorkspaceBuffers {
            send: self
                .completion_reduction_send
                .as_ref()
                .expect("Spark reduction send buffer ensured")
                .buffer(),
            recv: self
                .completion_reduction_recv
                .as_ref()
                .expect("Spark reduction receive buffer ensured")
                .buffer(),
        })
    }

    fn completion_output_slice(&mut self, bytes: usize) -> Result<&mut [u8]> {
        self.pinned_completion_output
            .as_mut()
            .context("NVFP4 route completion pinned output missing")?
            .as_mut_slice(bytes)
    }

    fn ensure_b12x_aot_route_buffers(
        &mut self,
        library: Arc<NativeLibrary>,
        rows: usize,
        hidden_dim: usize,
        intermediate_dim: usize,
        output_dim: usize,
    ) -> Result<B12xSparkAotRouteWorkspaceBuffers> {
        let capacity_rows = b12x_w4a16_capacity_rows(rows)?;
        let compact_hidden_bytes =
            checked_matrix_bytes(capacity_rows, hidden_dim, 2, "B12X input")?;
        let group_output_bytes = checked_matrix_bytes(
            capacity_rows * B12X_W4A16_PREFILL_TOPK8_ROUTES,
            output_dim,
            2,
            "B12X output",
        )?;
        let w4a16_routed_rows = capacity_rows
            .checked_mul(8)
            .context("B12X W4A16 routed row count overflow")?;
        let w4a16_fc1_bytes = checked_matrix_bytes(
            w4a16_routed_rows,
            intermediate_dim * 2,
            std::mem::size_of::<u16>(),
            "B12X W4A16 FC1 output",
        )?;
        let w4a16_activated_bytes = checked_matrix_bytes(
            w4a16_routed_rows,
            intermediate_dim,
            std::mem::size_of::<u16>(),
            "B12X W4A16 activated output",
        )?;
        Self::ensure_buffer(
            &mut self.b12x_compact_hidden,
            Arc::clone(&library),
            compact_hidden_bytes,
            "b12x Spark direct route compact hidden workspace",
        )?;
        Self::ensure_buffer(
            &mut self.b12x_group_output,
            Arc::clone(&library),
            group_output_bytes,
            "b12x Spark direct route compact output workspace",
        )?;
        Self::ensure_buffer(
            &mut self.b12x_w4a16_fc1_output,
            Arc::clone(&library),
            w4a16_fc1_bytes,
            "b12x Spark W4A16 FC1 output",
        )?;
        Self::ensure_buffer(
            &mut self.b12x_w4a16_activated,
            Arc::clone(&library),
            w4a16_activated_bytes,
            "b12x Spark W4A16 activated output",
        )?;
        Self::ensure_buffer(
            &mut self.b12x_w4a16_packed_route_indices,
            Arc::clone(&library),
            B12X_W4A16_MAX_PACKED_ROUTE_SLOTS * std::mem::size_of::<i32>(),
            "b12x Spark W4A16 packed routes",
        )?;
        Self::ensure_buffer(
            &mut self.b12x_w4a16_block_expert_ids,
            Arc::clone(&library),
            B12X_W4A16_MAX_ROUTE_BLOCKS * std::mem::size_of::<i32>(),
            "b12x Spark W4A16 route block experts",
        )?;
        Self::ensure_buffer(
            &mut self.b12x_w4a16_packed_route_count,
            Arc::clone(&library),
            std::mem::size_of::<i32>(),
            "b12x Spark W4A16 packed route count",
        )?;
        Self::ensure_buffer(
            &mut self.b12x_w4a16_topk_weights,
            Arc::clone(&library),
            capacity_rows
                .checked_mul(B12X_W4A16_PREFILL_TOPK8_ROUTES)
                .context("B12X W4A16 top-k weight count overflow")?
                * std::mem::size_of::<f32>(),
            "b12x Spark W4A16 top-k weights",
        )?;
        Self::ensure_buffer(
            &mut self.b12x_w4a16_fc1_scratch,
            Arc::clone(&library),
            B12X_W4A16_SCRATCH_ELEMENTS * std::mem::size_of::<f32>(),
            "b12x Spark W4A16 FC1 scratch",
        )?;
        Self::ensure_buffer(
            &mut self.b12x_w4a16_fc2_scratch,
            Arc::clone(&library),
            B12X_W4A16_SCRATCH_ELEMENTS * std::mem::size_of::<f32>(),
            "b12x Spark W4A16 FC2 scratch",
        )?;
        Self::ensure_buffer(
            &mut self.b12x_w4a16_locks,
            Arc::clone(&library),
            B12X_W4A16_LOCK_ELEMENTS * std::mem::size_of::<i32>(),
            "b12x Spark W4A16 locks",
        )?;
        Ok(B12xSparkAotRouteWorkspaceBuffers {
            compact_hidden: self
                .b12x_compact_hidden
                .as_ref()
                .expect("b12x compact hidden buffer ensured")
                .buffer(),
            group_output: self
                .b12x_group_output
                .as_ref()
                .expect("b12x group output buffer ensured")
                .buffer(),
            w4a16_fc1_output: self
                .b12x_w4a16_fc1_output
                .as_ref()
                .expect("b12x W4A16 FC1 output ensured")
                .buffer(),
            w4a16_activated: self
                .b12x_w4a16_activated
                .as_ref()
                .expect("b12x W4A16 activated output ensured")
                .buffer(),
            w4a16_packed_route_indices: self
                .b12x_w4a16_packed_route_indices
                .as_ref()
                .expect("b12x W4A16 packed routes ensured")
                .buffer(),
            w4a16_block_expert_ids: self
                .b12x_w4a16_block_expert_ids
                .as_ref()
                .expect("b12x W4A16 route block experts ensured")
                .buffer(),
            w4a16_packed_route_count: self
                .b12x_w4a16_packed_route_count
                .as_ref()
                .expect("b12x W4A16 packed route count ensured")
                .buffer(),
            w4a16_topk_weights: self
                .b12x_w4a16_topk_weights
                .as_ref()
                .expect("b12x W4A16 top-k weights ensured")
                .buffer(),
            w4a16_fc1_scratch: self
                .b12x_w4a16_fc1_scratch
                .as_ref()
                .expect("b12x W4A16 FC1 scratch ensured")
                .buffer(),
            w4a16_fc2_scratch: self
                .b12x_w4a16_fc2_scratch
                .as_ref()
                .expect("b12x W4A16 FC2 scratch ensured")
                .buffer(),
            w4a16_locks: self
                .b12x_w4a16_locks
                .as_ref()
                .expect("b12x W4A16 locks ensured")
                .buffer(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn quantize_bf16_projection_into_w4a16(
        &mut self,
        library: Arc<NativeLibrary>,
        source_parts: &[&[u8]],
        destination_weight: Ds41rtDeviceBuffer,
        destination_scale: Ds41rtDeviceBuffer,
        rows: usize,
        cols: usize,
        row_rotation: usize,
        global_scale: f32,
        cuda_stream: *mut c_void,
        label: &str,
    ) -> Result<()> {
        let source_bytes = source_parts.iter().try_fold(0_usize, |total, part| {
            total
                .checked_add(part.len())
                .with_context(|| format!("{label} source byte count overflow"))
        })?;
        let expected_source_bytes =
            checked_matrix_bytes(rows, cols, std::mem::size_of::<u16>(), label)?;
        let packed_bytes = checked_matrix_bytes(rows, cols / 2, 1, label)?;
        let scale_bytes = checked_matrix_bytes(rows, cols / 16, 1, label)?;
        anyhow::ensure!(
            rows > 0
                && cols > 0
                && cols % 16 == 0
                && row_rotation < rows
                && source_bytes == expected_source_bytes
                && destination_weight.bytes >= packed_bytes
                && destination_scale.bytes >= scale_bytes
                && global_scale.is_finite()
                && global_scale > 0.0,
            "invalid startup BF16-to-NVFP4 geometry for {label}"
        );
        Self::ensure_pinned_buffer(
            &mut self.pinned_projection_weight,
            Arc::clone(&library),
            source_bytes,
            label,
        )?;
        let staging = self
            .pinned_projection_weight
            .as_mut()
            .expect("startup BF16 weight staging ensured");
        let staging_slice = staging.as_mut_slice(source_bytes)?;
        let mut offset = 0;
        for part in source_parts {
            staging_slice[offset..offset + part.len()].copy_from_slice(part);
            offset += part.len();
        }
        let staging_buffer = staging.buffer();
        Self::ensure_buffer(
            &mut self.b12x_w4a16_pack_source,
            Arc::clone(&library),
            source_bytes,
            "startup BF16 expert weight",
        )?;
        Self::ensure_buffer(
            &mut self.startup_nvfp4_weight,
            Arc::clone(&library),
            packed_bytes,
            "startup raw NVFP4 expert weight",
        )?;
        Self::ensure_buffer(
            &mut self.startup_nvfp4_scale,
            Arc::clone(&library),
            scale_bytes,
            "startup raw NVFP4 expert scales",
        )?;
        let bf16_source = self
            .b12x_w4a16_pack_source
            .as_ref()
            .expect("startup BF16 expert buffer ensured")
            .buffer();
        let raw_weight = self
            .startup_nvfp4_weight
            .as_ref()
            .expect("startup NVFP4 weight buffer ensured")
            .buffer();
        let raw_scale = self
            .startup_nvfp4_scale
            .as_ref()
            .expect("startup NVFP4 scale buffer ensured")
            .buffer();
        unsafe {
            library
                .copy_host_buffer_h2d_async(bf16_source, staging_buffer, source_bytes, cuda_stream)
                .with_context(|| format!("uploading {label}"))?;
            library
                .cuda_quantize_bf16_weight_nvfp4_async(
                    bf16_source,
                    raw_weight,
                    raw_scale,
                    rows,
                    cols,
                    global_scale,
                    cuda_stream,
                )
                .with_context(|| format!("quantizing {label} to NVFP4"))?;
            library
                .cuda_b12x_w4a16_pack_weight_async(
                    raw_weight,
                    destination_weight,
                    cols,
                    rows,
                    row_rotation,
                    cuda_stream,
                )
                .with_context(|| format!("packing startup-quantized {label} weights"))?;
            library
                .cuda_b12x_w4a16_pack_scale_async(
                    raw_scale,
                    destination_scale,
                    cols,
                    rows,
                    row_rotation,
                    1.0,
                    cuda_stream,
                )
                .with_context(|| format!("packing startup-quantized {label} scales"))?;
            library
                .cuda_stream_synchronize(cuda_stream)
                .with_context(|| format!("synchronizing startup quantization for {label}"))?;
        }
        Ok(())
    }

    fn upload_host_bytes_to_existing_device_buffer(
        &mut self,
        library: Arc<NativeLibrary>,
        destination: Ds41rtDeviceBuffer,
        bytes: &[u8],
        label: &str,
        slot: RouteCudaProjectionStageSlot,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        anyhow::ensure!(
            destination.bytes >= bytes.len(),
            "device buffer for {label} has {} bytes, needs {}",
            destination.bytes,
            bytes.len()
        );
        let staging_slot = match slot {
            RouteCudaProjectionStageSlot::Weight => &mut self.pinned_projection_weight,
        };
        let staging = Self::stage_pinned_bytes(staging_slot, library.clone(), bytes, label)?;
        unsafe {
            library
                .copy_host_buffer_h2d_async(destination, staging, bytes.len(), cuda_stream)
                .with_context(|| format!("enqueueing pinned {label} H2D copy"))?;
            library
                .cuda_stream_synchronize(cuda_stream)
                .with_context(|| format!("synchronizing pinned {label} H2D copy"))?;
        }
        Ok(())
    }

    fn ensure_buffer(
        slot: &mut Option<OwnedDeviceAllocation>,
        library: Arc<NativeLibrary>,
        bytes: usize,
        label: &str,
    ) -> Result<()> {
        let bytes = bytes.max(1);
        let needs_allocation = slot
            .as_ref()
            .map(|allocation| allocation.capacity_bytes() < bytes)
            .unwrap_or(true);
        if needs_allocation {
            *slot = Some(OwnedDeviceAllocation::new(library, bytes, label)?);
        }
        Ok(())
    }

    fn stage_pinned_bytes(
        slot: &mut Option<OwnedPinnedHostAllocation>,
        library: Arc<NativeLibrary>,
        bytes: &[u8],
        label: &str,
    ) -> Result<Ds41rtHostBuffer> {
        Self::ensure_pinned_buffer(slot, library, bytes.len(), label)?;
        let staging = slot.as_mut().expect("pinned host buffer ensured");
        staging.as_mut_slice(bytes.len())?.copy_from_slice(bytes);
        Ok(staging.buffer())
    }

    fn ensure_pinned_buffer(
        slot: &mut Option<OwnedPinnedHostAllocation>,
        library: Arc<NativeLibrary>,
        bytes: usize,
        label: &str,
    ) -> Result<()> {
        let bytes = bytes.max(1);
        let needs_allocation = slot
            .as_ref()
            .map(|allocation| allocation.capacity_bytes() < bytes)
            .unwrap_or(true);
        if needs_allocation {
            *slot = Some(OwnedPinnedHostAllocation::new(library, bytes, label)?);
        }
        Ok(())
    }
}

enum RouteCudaTensorStorage {
    Owned(Vec<u8>),
    Direct(RouteCudaAlignedReadBuffer),
}

#[derive(Clone)]
struct RouteCudaTensorBytes {
    storage: Arc<RouteCudaTensorStorage>,
    offset: usize,
    bytes: usize,
}

impl RouteCudaTensorBytes {
    fn owned(bytes: Vec<u8>) -> Self {
        let length = bytes.len();
        Self {
            storage: Arc::new(RouteCudaTensorStorage::Owned(bytes)),
            offset: 0,
            bytes: length,
        }
    }

    fn direct(buffer: RouteCudaAlignedReadBuffer) -> Self {
        let length = buffer.requested_slice().len();
        Self {
            storage: Arc::new(RouteCudaTensorStorage::Direct(buffer)),
            offset: 0,
            bytes: length,
        }
    }

    fn view(&self, offset: usize, bytes: usize) -> Result<Self> {
        let end = offset
            .checked_add(bytes)
            .context("route tensor byte view end overflow")?;
        anyhow::ensure!(
            end <= self.bytes,
            "route tensor byte view {offset}..{end} exceeds {} bytes",
            self.bytes
        );
        Ok(Self {
            storage: Arc::clone(&self.storage),
            offset: self.offset + offset,
            bytes,
        })
    }

    fn as_slice(&self) -> &[u8] {
        let storage = match self.storage.as_ref() {
            RouteCudaTensorStorage::Owned(bytes) => bytes.as_slice(),
            RouteCudaTensorStorage::Direct(bytes) => bytes.requested_slice(),
        };
        &storage[self.offset..self.offset + self.bytes]
    }
}

impl Deref for RouteCudaTensorBytes {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

struct LoadedRouteCudaTensorRows {
    info: TensorInfo,
    source_path: PathBuf,
    start_row: usize,
    row_count: usize,
    row_width: usize,
    source_row_width: usize,
    source_column_start: usize,
    bytes_per_scalar: usize,
    bytes: RouteCudaTensorBytes,
    elapsed_micros: u128,
}

struct LoadedRouteCudaProjectionShard {
    weight: LoadedRouteCudaTensorRows,
    weight_scale: LoadedRouteCudaTensorRows,
}

struct LoadedRouteCudaExpertShard {
    gate: LoadedRouteCudaProjectionShard,
    up: LoadedRouteCudaProjectionShard,
    down: LoadedRouteCudaProjectionShard,
}

fn loaded_route_cuda_tensor_logical_bytes(tensor: &LoadedRouteCudaTensorRows) -> Result<usize> {
    tensor
        .row_count
        .checked_mul(tensor.row_width)
        .and_then(|values| values.checked_mul(tensor.bytes_per_scalar))
        .context("route tensor logical byte count overflow")
}

fn copy_loaded_route_cuda_tensor_compact(
    tensor: &LoadedRouteCudaTensorRows,
    destination: &mut [u8],
) -> Result<()> {
    let logical_row_bytes = tensor
        .row_width
        .checked_mul(tensor.bytes_per_scalar)
        .context("route tensor logical row bytes overflow")?;
    let source_row_bytes = tensor
        .source_row_width
        .checked_mul(tensor.bytes_per_scalar)
        .context("route tensor source row bytes overflow")?;
    let source_column_bytes = tensor
        .source_column_start
        .checked_mul(tensor.bytes_per_scalar)
        .context("route tensor source column bytes overflow")?;
    let logical_bytes = loaded_route_cuda_tensor_logical_bytes(tensor)?;
    anyhow::ensure!(
        destination.len() == logical_bytes
            && tensor.bytes.len() == tensor.row_count * source_row_bytes
            && source_column_bytes + logical_row_bytes <= source_row_bytes,
        "route tensor compact copy geometry is inconsistent"
    );
    if logical_row_bytes == source_row_bytes && source_column_bytes == 0 {
        destination.copy_from_slice(&tensor.bytes);
    } else {
        for (source, output) in tensor
            .bytes
            .chunks_exact(source_row_bytes)
            .zip(destination.chunks_exact_mut(logical_row_bytes))
        {
            output.copy_from_slice(
                &source[source_column_bytes..source_column_bytes + logical_row_bytes],
            );
        }
    }
    Ok(())
}

struct LoadedRouteCudaBf16ExpertShard {
    gate: LoadedTensorRows,
    up: LoadedTensorRows,
    down: LoadedTensorRows,
}

fn bf16_bytes_amax(parts: &[&[u8]], label: &str) -> Result<f32> {
    let mut maximum = 0.0_f32;
    for part in parts {
        anyhow::ensure!(
            part.len() % std::mem::size_of::<u16>() == 0,
            "{label} BF16 byte count is not scalar-aligned"
        );
        for scalar in part.chunks_exact(2) {
            let bits = u16::from_le_bytes([scalar[0], scalar[1]]);
            let value = f32::from_bits((bits as u32) << 16);
            anyhow::ensure!(
                value.is_finite(),
                "{label} contains a non-finite BF16 value"
            );
            maximum = maximum.max(value.abs());
        }
    }
    Ok(maximum)
}

fn nvfp4_global_scale_for_bf16(parts: &[&[u8]], label: &str) -> Result<f32> {
    let maximum = bf16_bytes_amax(parts, label)?;
    if maximum == 0.0 {
        Ok(1.0)
    } else {
        let scale = 448.0_f32 * 6.0 / maximum;
        anyhow::ensure!(
            scale.is_finite() && scale > 0.0,
            "{label} produced invalid NVFP4 global scale {scale}"
        );
        Ok(scale)
    }
}

fn synthetic_quantized_projection_geometry(
    source: &LoadedTensorRows,
) -> Result<LoadedRouteCudaProjectionShard> {
    anyhow::ensure!(
        source.info.dtype == DType::Bf16
            && source.bytes_per_scalar == 2
            && source.row_width % 16 == 0,
        "startup NVFP4 quantization requires BF16 rows divisible by 16"
    );
    let synthetic = |dtype: DType, row_width: usize| {
        let mut info = source.info.clone();
        info.dtype = dtype;
        info.shape = vec![source.row_count, row_width];
        info.byte_length = (source.row_count * row_width) as u64;
        LoadedRouteCudaTensorRows {
            info,
            source_path: source.source_path.clone(),
            start_row: source.start_row,
            row_count: source.row_count,
            row_width,
            source_row_width: row_width,
            source_column_start: 0,
            bytes_per_scalar: 1,
            bytes: RouteCudaTensorBytes::owned(Vec::new()),
            elapsed_micros: source.elapsed_micros,
        }
    };
    Ok(LoadedRouteCudaProjectionShard {
        weight: synthetic(DType::U8, source.row_width / 2),
        weight_scale: synthetic(DType::F8E4M3, source.row_width / 16),
    })
}

fn synthetic_quantized_expert_geometry(
    source: &LoadedRouteCudaBf16ExpertShard,
) -> Result<LoadedRouteCudaExpertShard> {
    Ok(LoadedRouteCudaExpertShard {
        gate: synthetic_quantized_projection_geometry(&source.gate)?,
        up: synthetic_quantized_projection_geometry(&source.up)?,
        down: synthetic_quantized_projection_geometry(&source.down)?,
    })
}

#[derive(Clone, Copy)]
struct RouteCudaBf16ExpertBuffers {
    w13_weight: Ds41rtDeviceBuffer,
    w2_weight: Ds41rtDeviceBuffer,
}

struct RouteCudaBf16LayerExpertSlab {
    layer_id: usize,
    expert_count: usize,
    hidden_dim: usize,
    intermediate_rows: usize,
    output_rows: usize,
    w13_weight: Arc<OwnedDeviceAllocation>,
    w13_expert_stride_bytes: usize,
    w2_weight: Arc<OwnedDeviceAllocation>,
    w2_expert_stride_bytes: usize,
}

struct RouteCudaDs4FlashLayerExpertSlab {
    layer_id: usize,
    expert_count: usize,
    w13_weight: Arc<OwnedDeviceAllocation>,
    w13_scale: Arc<OwnedDeviceAllocation>,
    w2_weight: Arc<OwnedDeviceAllocation>,
    w2_scale: Arc<OwnedDeviceAllocation>,
    w13_global_scale: Arc<OwnedDeviceAllocation>,
    w2_global_scale: Arc<OwnedDeviceAllocation>,
}

struct RouteCudaDs4FlashExl3LayerExpertSlab {
    layer_id: usize,
    expert_count: usize,
    profile: Ds4Exl3AotProfile,
    w13_trellis: Arc<OwnedDeviceAllocation>,
    w2_trellis: Arc<OwnedDeviceAllocation>,
    gate_suh: Arc<OwnedDeviceAllocation>,
    up_suh: Arc<OwnedDeviceAllocation>,
    intermediate_rotations: Arc<OwnedDeviceAllocation>,
    down_svh: Arc<OwnedDeviceAllocation>,
    expert_map: Arc<OwnedDeviceAllocation>,
    dummy_scale: Arc<OwnedDeviceAllocation>,
    trellis_lut: Arc<OwnedDeviceAllocation>,
    global_scale: Arc<OwnedDeviceAllocation>,
}

struct RouteCudaDs4FlashExl3MixedLayerExpertSlab {
    layer_id: usize,
    tier0_gate_expert_ids: Vec<usize>,
    tier1_gate_expert_ids: Vec<usize>,
    tier0_up_expert_ids: Vec<usize>,
    tier1_up_expert_ids: Vec<usize>,
    tier0_down_expert_ids: Vec<usize>,
    tier1_down_expert_ids: Vec<usize>,
    tier0_slots: usize,
    tier1_slots: usize,
    tier0_w13_trellis: Arc<OwnedDeviceAllocation>,
    tier0_w2_trellis: Arc<OwnedDeviceAllocation>,
    tier1_w13_trellis: Arc<OwnedDeviceAllocation>,
    tier1_w2_trellis: Arc<OwnedDeviceAllocation>,
    dummy_scale: Arc<OwnedDeviceAllocation>,
    tier0_global_scale: Arc<OwnedDeviceAllocation>,
    tier1_global_scale: Arc<OwnedDeviceAllocation>,
    gate_suh: Arc<OwnedDeviceAllocation>,
    up_suh: Arc<OwnedDeviceAllocation>,
    intermediate_rotations: Arc<OwnedDeviceAllocation>,
    down_svh: Arc<OwnedDeviceAllocation>,
    trellis_lut: Arc<OwnedDeviceAllocation>,
    global_to_combined: Arc<OwnedDeviceAllocation>,
    descriptor_map: Arc<OwnedDeviceAllocation>,
}

impl RouteCudaDs4FlashLayerExpertSlab {
    fn new(library: Arc<NativeLibrary>, catalog: &TensorCatalog, layer_id: usize) -> Result<Self> {
        anyhow::ensure!(
            catalog.facts.hidden_size == DS4_FLASH_HIDDEN_SIZE
                && catalog.facts.moe_intermediate_size
                    == DS4_FLASH_LOCAL_INTERMEDIATE_SIZE * 4
                && catalog.facts.routed_experts == DS4_FLASH_ROUTED_EXPERTS,
            "DeepSeek V4 Flash AOT layer {layer_id} requires hidden={} intermediate={} experts={}, got hidden={} intermediate={} experts={}",
            DS4_FLASH_HIDDEN_SIZE,
            DS4_FLASH_LOCAL_INTERMEDIATE_SIZE * 4,
            DS4_FLASH_ROUTED_EXPERTS,
            catalog.facts.hidden_size,
            catalog.facts.moe_intermediate_size,
            catalog.facts.routed_experts,
        );
        // Validate the checkpoint contract before committing almost 1 GiB of
        // resident device memory for this layer.
        let exemplar = native_fp4_expert(catalog, layer_id, 0)?
            .tp_shard(0)
            .with_context(|| format!("validating native Flash layer {layer_id} TP geometry"))?;
        anyhow::ensure!(
            exemplar.gate.local_intermediate_size == DS4_FLASH_LOCAL_INTERMEDIATE_SIZE
                && exemplar.up.local_intermediate_size == DS4_FLASH_LOCAL_INTERMEDIATE_SIZE
                && exemplar.down.local_intermediate_size == DS4_FLASH_LOCAL_INTERMEDIATE_SIZE,
            "native Flash layer {layer_id} does not provide equal 512-wide TP4 expert slices"
        );

        let expert_count = DS4_FLASH_ROUTED_EXPERTS;
        let slab_bytes = |stride: usize, label: &str| {
            stride
                .checked_mul(expert_count)
                .with_context(|| format!("native Flash {label} slab byte count overflow"))
        };
        let w13_weight = Arc::new(OwnedDeviceAllocation::new(
            Arc::clone(&library),
            slab_bytes(DS4_FLASH_W13_WEIGHT_EXPERT_BYTES, "W13 weight")?,
            &format!("native Flash layer {layer_id} TP4 W13 weights"),
        )?);
        let w13_scale = Arc::new(OwnedDeviceAllocation::new(
            Arc::clone(&library),
            slab_bytes(DS4_FLASH_W13_SCALE_EXPERT_BYTES, "W13 scale")?,
            &format!("native Flash layer {layer_id} TP4 W13 scales"),
        )?);
        let w2_weight = Arc::new(OwnedDeviceAllocation::new(
            Arc::clone(&library),
            slab_bytes(DS4_FLASH_W2_WEIGHT_EXPERT_BYTES, "W2 weight")?,
            &format!("native Flash layer {layer_id} TP4 W2 weights"),
        )?);
        let w2_scale = Arc::new(OwnedDeviceAllocation::new(
            Arc::clone(&library),
            slab_bytes(DS4_FLASH_W2_SCALE_EXPERT_BYTES, "W2 scale")?,
            &format!("native Flash layer {layer_id} TP4 W2 scales"),
        )?);
        let global_scale_bytes = expert_count * std::mem::size_of::<f32>();
        let w13_global_scale = Arc::new(OwnedDeviceAllocation::new(
            Arc::clone(&library),
            global_scale_bytes,
            &format!("native Flash layer {layer_id} W13 global scales"),
        )?);
        let w2_global_scale = Arc::new(OwnedDeviceAllocation::new(
            Arc::clone(&library),
            global_scale_bytes,
            &format!("native Flash layer {layer_id} W2 global scales"),
        )?);
        let ones = (0..expert_count)
            .flat_map(|_| 1.0_f32.to_le_bytes())
            .collect::<Vec<_>>();
        library
            .copy_h2d(w13_global_scale.buffer(), &ones)
            .context("uploading native Flash W13 global scales")?;
        library
            .copy_h2d(w2_global_scale.buffer(), &ones)
            .context("uploading native Flash W2 global scales")?;

        Ok(Self {
            layer_id,
            expert_count,
            w13_weight,
            w13_scale,
            w2_weight,
            w2_scale,
            w13_global_scale,
            w2_global_scale,
        })
    }

    fn store_layer_experts(
        &self,
        expert_ids: &[usize],
        loaded_experts: &[LoadedRouteCudaExpertShard],
        library: Arc<NativeLibrary>,
        workspace: &mut RouteCudaWorkspace,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        anyhow::ensure!(
            !expert_ids.is_empty() && expert_ids.len() == loaded_experts.len(),
            "native Flash layer {} preload received {} expert ids and {} payloads",
            self.layer_id,
            expert_ids.len(),
            loaded_experts.len()
        );

        #[derive(Clone, Copy)]
        struct PackPlan {
            source_offset: usize,
            source_bytes: usize,
            destination: Ds41rtDeviceBuffer,
            size_k: usize,
            size_n: usize,
            row_rotation: usize,
            scale: bool,
        }

        let validate = |projection: &LoadedRouteCudaProjectionShard,
                        rows: usize,
                        weight_width: usize,
                        scale_width: usize,
                        label: &str|
         -> Result<()> {
            anyhow::ensure!(
                projection.weight.info.dtype == DType::I8
                    && projection.weight_scale.info.dtype == DType::F8E8M0
                    && projection.weight.bytes_per_scalar == 1
                    && projection.weight_scale.bytes_per_scalar == 1,
                "native Flash {label} must contain packed I8 weights and F8E8M0 scales"
            );
            anyhow::ensure!(
                projection.weight.row_count == rows
                    && projection.weight.row_width == weight_width
                    && projection.weight_scale.row_count == rows
                    && projection.weight_scale.row_width == scale_width,
                "native Flash {label} TP geometry is rows={}/{} weight_width={}/{} scale_width={}/{}",
                projection.weight.row_count,
                rows,
                projection.weight.row_width,
                weight_width,
                projection.weight_scale.row_width,
                scale_width,
            );
            Ok(())
        };

        let source_bytes = loaded_experts.iter().try_fold(0_usize, |total, loaded| {
            [
                &loaded.up.weight,
                &loaded.gate.weight,
                &loaded.up.weight_scale,
                &loaded.gate.weight_scale,
                &loaded.down.weight,
                &loaded.down.weight_scale,
            ]
            .into_iter()
            .try_fold(total, |subtotal, tensor| {
                subtotal
                    .checked_add(loaded_route_cuda_tensor_logical_bytes(tensor)?)
                    .context("native Flash preload source byte count overflow")
            })
        })?;
        RouteCudaWorkspace::ensure_pinned_buffer(
            &mut workspace.pinned_projection_weight,
            Arc::clone(&library),
            source_bytes,
            "native Flash batched pack source",
        )?;
        RouteCudaWorkspace::ensure_buffer(
            &mut workspace.b12x_w4a16_pack_source,
            Arc::clone(&library),
            source_bytes,
            "native Flash batched device pack source",
        )?;

        let mut plans = Vec::with_capacity(expert_ids.len() * 4);
        let staging_buffer;
        {
            let staging = workspace
                .pinned_projection_weight
                .as_mut()
                .expect("native Flash pinned pack source ensured above");
            let staging_slice = staging.as_mut_slice(source_bytes)?;
            let mut source_offset = 0_usize;
            for (&expert_id, loaded) in expert_ids.iter().zip(loaded_experts) {
                anyhow::ensure!(
                    expert_id < self.expert_count,
                    "native Flash layer {} expert {expert_id} exceeds {}",
                    self.layer_id,
                    self.expert_count
                );
                validate(
                    &loaded.up,
                    DS4_FLASH_LOCAL_INTERMEDIATE_SIZE,
                    DS4_FLASH_HIDDEN_SIZE / 2,
                    DS4_FLASH_HIDDEN_SIZE / 32,
                    "up projection",
                )?;
                validate(
                    &loaded.gate,
                    DS4_FLASH_LOCAL_INTERMEDIATE_SIZE,
                    DS4_FLASH_HIDDEN_SIZE / 2,
                    DS4_FLASH_HIDDEN_SIZE / 32,
                    "gate projection",
                )?;
                validate(
                    &loaded.down,
                    DS4_FLASH_HIDDEN_SIZE,
                    DS4_FLASH_LOCAL_INTERMEDIATE_SIZE / 2,
                    DS4_FLASH_LOCAL_INTERMEDIATE_SIZE / 32,
                    "down projection",
                )?;

                let w13_weight_parts = [&loaded.up.weight, &loaded.gate.weight];
                let w13_scale_parts = [&loaded.up.weight_scale, &loaded.gate.weight_scale];
                let w2_weight_parts = [&loaded.down.weight];
                let w2_scale_parts = [&loaded.down.weight_scale];
                let destinations = [
                    (
                        device_buffer_byte_view(
                            self.w13_weight.buffer(),
                            expert_id * DS4_FLASH_W13_WEIGHT_EXPERT_BYTES,
                            DS4_FLASH_W13_WEIGHT_EXPERT_BYTES,
                            "native Flash W13 expert weight",
                        )?,
                        DS4_FLASH_HIDDEN_SIZE,
                        DS4_FLASH_LOCAL_INTERMEDIATE_SIZE * 2,
                        DS4_FLASH_LOCAL_INTERMEDIATE_SIZE,
                        false,
                        w13_weight_parts.as_slice(),
                    ),
                    (
                        device_buffer_byte_view(
                            self.w13_scale.buffer(),
                            expert_id * DS4_FLASH_W13_SCALE_EXPERT_BYTES,
                            DS4_FLASH_W13_SCALE_EXPERT_BYTES,
                            "native Flash W13 expert scale",
                        )?,
                        DS4_FLASH_HIDDEN_SIZE,
                        DS4_FLASH_LOCAL_INTERMEDIATE_SIZE * 2,
                        DS4_FLASH_LOCAL_INTERMEDIATE_SIZE,
                        true,
                        w13_scale_parts.as_slice(),
                    ),
                    (
                        device_buffer_byte_view(
                            self.w2_weight.buffer(),
                            expert_id * DS4_FLASH_W2_WEIGHT_EXPERT_BYTES,
                            DS4_FLASH_W2_WEIGHT_EXPERT_BYTES,
                            "native Flash W2 expert weight",
                        )?,
                        DS4_FLASH_LOCAL_INTERMEDIATE_SIZE,
                        DS4_FLASH_HIDDEN_SIZE,
                        0,
                        false,
                        w2_weight_parts.as_slice(),
                    ),
                    (
                        device_buffer_byte_view(
                            self.w2_scale.buffer(),
                            expert_id * DS4_FLASH_W2_SCALE_EXPERT_BYTES,
                            DS4_FLASH_W2_SCALE_EXPERT_BYTES,
                            "native Flash W2 expert scale",
                        )?,
                        DS4_FLASH_LOCAL_INTERMEDIATE_SIZE,
                        DS4_FLASH_HIDDEN_SIZE,
                        0,
                        true,
                        w2_scale_parts.as_slice(),
                    ),
                ];
                for (destination, size_k, size_n, row_rotation, scale, parts) in destinations {
                    let plan_offset = source_offset;
                    for part in parts {
                        let part_bytes = loaded_route_cuda_tensor_logical_bytes(part)?;
                        let end = source_offset
                            .checked_add(part_bytes)
                            .context("native Flash staging offset overflow")?;
                        copy_loaded_route_cuda_tensor_compact(
                            part,
                            &mut staging_slice[source_offset..end],
                        )?;
                        source_offset = end;
                    }
                    plans.push(PackPlan {
                        source_offset: plan_offset,
                        source_bytes: source_offset - plan_offset,
                        destination,
                        size_k,
                        size_n,
                        row_rotation,
                        scale,
                    });
                }
            }
            anyhow::ensure!(
                source_offset == source_bytes,
                "native Flash staged {source_offset} bytes, expected {source_bytes}"
            );
            staging_buffer = staging.buffer();
        }

        let source = workspace
            .b12x_w4a16_pack_source
            .as_ref()
            .expect("native Flash device pack source ensured above")
            .buffer();
        unsafe {
            library
                .copy_host_buffer_h2d_async(source, staging_buffer, source_bytes, cuda_stream)
                .context("uploading native Flash pack source")?;
            for plan in plans {
                let source_view = device_buffer_byte_view(
                    source,
                    plan.source_offset,
                    plan.source_bytes,
                    "native Flash pack source view",
                )?;
                if plan.scale {
                    library
                        .cuda_ds4_flash_w4a16_pack_e8m0_scale_async(
                            source_view,
                            plan.destination,
                            plan.size_k,
                            plan.size_n,
                            plan.row_rotation,
                            cuda_stream,
                        )
                        .context("packing native Flash E8M0 scales")?;
                } else {
                    library
                        .cuda_ds4_flash_w4a16_pack_weight_async(
                            source_view,
                            plan.destination,
                            plan.size_k,
                            plan.size_n,
                            plan.row_rotation,
                            cuda_stream,
                        )
                        .context("packing native Flash expert weights")?;
                }
            }
            library
                .cuda_stream_synchronize(cuda_stream)
                .context("synchronizing native Flash layer pack")?;
        }
        Ok(())
    }

    fn resident_weight_bytes(&self) -> u64 {
        (self.w13_weight.capacity_bytes() + self.w2_weight.capacity_bytes()) as u64
    }

    fn resident_scale_bytes(&self) -> u64 {
        // The strict-TP4 plan accounts the two global-scale vectors as scalar
        // metadata; include their actual CUDA allocations in the resident
        // metadata total as well as the E8M0 scale slabs.
        (self.w13_scale.capacity_bytes()
            + self.w2_scale.capacity_bytes()
            + self.w13_global_scale.capacity_bytes()
            + self.w2_global_scale.capacity_bytes()) as u64
    }

    fn aot_buffers(
        &self,
        workspace: Ds4FlashSparkAotRouteWorkspaceBuffers,
        input: Ds41rtDeviceBuffer,
        routed_output: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        packed_route_indices: Ds41rtDeviceBuffer,
        topk_weights: Ds41rtDeviceBuffer,
    ) -> Ds41rtDs4FlashSparkW4a16MoeBuffers {
        Ds41rtDs4FlashSparkW4a16MoeBuffers {
            input,
            w13_weight: self.w13_weight.buffer(),
            w2_weight: self.w2_weight.buffer(),
            fc1_output: workspace.fc1_output,
            activated: workspace.activated,
            routed_output,
            output,
            w13_scale: self.w13_scale.buffer(),
            w2_scale: self.w2_scale.buffer(),
            w13_global_scale: self.w13_global_scale.buffer(),
            w2_global_scale: self.w2_global_scale.buffer(),
            packed_route_indices,
            block_expert_ids: workspace.block_expert_ids,
            packed_route_count: workspace.packed_route_count,
            topk_weights,
            fc1_scratch: workspace.fc1_scratch,
            fc2_scratch: workspace.fc2_scratch,
            locks: workspace.locks,
        }
    }

    fn decode_m1_aot_buffers(
        &self,
        buffers: Ds4FlashSparkDecodeM1DeviceBuffers,
    ) -> Ds41rtDs4FlashSparkW4a16MoeBuffers {
        Ds41rtDs4FlashSparkW4a16MoeBuffers {
            input: buffers.input,
            w13_weight: self.w13_weight.buffer(),
            w2_weight: self.w2_weight.buffer(),
            fc1_output: buffers.fc1,
            activated: buffers.activated,
            routed_output: Ds41rtDeviceBuffer::default(),
            output: buffers.output,
            w13_scale: self.w13_scale.buffer(),
            w2_scale: self.w2_scale.buffer(),
            w13_global_scale: self.w13_global_scale.buffer(),
            w2_global_scale: self.w2_global_scale.buffer(),
            packed_route_indices: buffers.route_indices,
            block_expert_ids: buffers.block_expert_ids,
            packed_route_count: buffers.packed_route_count,
            topk_weights: buffers.route_weights,
            fc1_scratch: buffers.fc1_scratch,
            fc2_scratch: buffers.fc2_scratch,
            locks: buffers.locks,
        }
    }

    fn prefill_aot_buffers(
        &self,
        buffers: Ds4FlashSparkPrefillDeviceBuffers,
    ) -> Ds41rtDs4FlashSparkW4a16MoeBuffers {
        Ds41rtDs4FlashSparkW4a16MoeBuffers {
            input: buffers.input,
            w13_weight: self.w13_weight.buffer(),
            w2_weight: self.w2_weight.buffer(),
            fc1_output: buffers.fc1,
            activated: buffers.activated,
            routed_output: buffers.routed_output,
            output: buffers.output,
            w13_scale: self.w13_scale.buffer(),
            w2_scale: self.w2_scale.buffer(),
            w13_global_scale: self.w13_global_scale.buffer(),
            w2_global_scale: self.w2_global_scale.buffer(),
            packed_route_indices: buffers.packed_route_indices,
            block_expert_ids: buffers.block_expert_ids,
            packed_route_count: buffers.packed_route_count,
            topk_weights: buffers.topk_weights,
            fc1_scratch: buffers.fc1_scratch,
            fc2_scratch: buffers.fc2_scratch,
            locks: buffers.locks,
        }
    }
}

fn exl3_source_file(
    snapshot: &Path,
    tensor: &TensorInfo,
    files: &mut HashMap<PathBuf, Arc<File>>,
) -> Result<Arc<File>> {
    let path = snapshot.join(&tensor.file);
    if let Some(file) = files.get(&path) {
        return Ok(Arc::clone(file));
    }
    let file = Arc::new(File::open(&path).with_context(|| format!("opening {}", path.display()))?);
    files.insert(path, Arc::clone(&file));
    Ok(file)
}

fn read_exl3_tensor_window(
    snapshot: &Path,
    tensor: &TensorInfo,
    relative_offset: usize,
    files: &mut HashMap<PathBuf, Arc<File>>,
    dst: &mut [u8],
) -> Result<()> {
    let end = relative_offset
        .checked_add(dst.len())
        .context("EXL3 tensor window overflow")?;
    anyhow::ensure!(
        end <= tensor.byte_length as usize,
        "EXL3 tensor window {}..{end} exceeds {} bytes for {}",
        relative_offset,
        tensor.byte_length,
        tensor.name
    );
    let file = exl3_source_file(snapshot, tensor, files)?;
    file.read_exact_at(dst, tensor.byte_offset + relative_offset as u64)
        .with_context(|| format!("reading compact EXL3 tensor window {}", tensor.name))
}

fn exl3_trellis_tile_bytes(trellis_bits: usize) -> Result<usize> {
    anyhow::ensure!(
        matches!(trellis_bits, 2 | 3),
        "EXL3 TP4 read does not support K{trellis_bits}"
    );
    Ok(32 * trellis_bits)
}

fn read_exl3_trellis_tp4(
    snapshot: &Path,
    projection: Exl3Projection<'_>,
    shard: ExpertIntermediateShard,
    hidden_size: usize,
    local_intermediate_size: usize,
    trellis_bits: usize,
    trellis_bytes: usize,
    files: &mut HashMap<PathBuf, Arc<File>>,
    dst: &mut [u8],
) -> Result<()> {
    anyhow::ensure!(
        dst.len() == trellis_bytes,
        "EXL3 compact trellis destination has {} bytes, expected {}",
        dst.len(),
        trellis_bytes
    );
    let tile_bytes = exl3_trellis_tile_bytes(trellis_bits)?;
    let input_tiles = projection.input_features / 16;
    let output_tiles = projection.output_features / 16;
    let local_tiles = local_intermediate_size / 16;
    if projection.input_features == hidden_size {
        anyhow::ensure!(
            projection.output_features == local_intermediate_size * shard.count,
            "EXL3 {:?} projection has unsupported {}x{} TP geometry",
            projection.kind,
            projection.input_features,
            projection.output_features
        );
        let source_row_bytes = output_tiles * tile_bytes;
        let local_row_bytes = local_tiles * tile_bytes;
        for input_tile in 0..input_tiles {
            let source_offset = input_tile * source_row_bytes + shard.rank * local_row_bytes;
            let destination_offset = input_tile * local_row_bytes;
            read_exl3_tensor_window(
                snapshot,
                projection.trellis,
                source_offset,
                files,
                &mut dst[destination_offset..destination_offset + local_row_bytes],
            )?;
        }
    } else {
        anyhow::ensure!(
            projection.input_features == local_intermediate_size * shard.count
                && projection.output_features == hidden_size,
            "EXL3 {:?} projection has unsupported {}x{} TP geometry",
            projection.kind,
            projection.input_features,
            projection.output_features
        );
        let local_input_tiles = input_tiles / shard.count;
        let bytes = local_input_tiles * output_tiles * tile_bytes;
        anyhow::ensure!(bytes == dst.len(), "EXL3 down trellis TP4 byte mismatch");
        read_exl3_tensor_window(snapshot, projection.trellis, shard.rank * bytes, files, dst)?;
    }
    Ok(())
}

struct Exl3ReadRequest {
    file: Arc<File>,
    source_offset: u64,
    destination: *mut u8,
    bytes: usize,
    label: String,
}

fn queue_exl3_tensor_window(
    snapshot: &Path,
    tensor: &TensorInfo,
    relative_offset: usize,
    files: &mut HashMap<PathBuf, Arc<File>>,
    dst: &mut [u8],
    requests: &mut Vec<Exl3ReadRequest>,
) -> Result<()> {
    let end = relative_offset
        .checked_add(dst.len())
        .context("EXL3 queued tensor window overflow")?;
    anyhow::ensure!(
        end <= tensor.byte_length as usize,
        "EXL3 queued tensor window {}..{end} exceeds {} bytes for {}",
        relative_offset,
        tensor.byte_length,
        tensor.name
    );
    requests.push(Exl3ReadRequest {
        file: exl3_source_file(snapshot, tensor, files)?,
        source_offset: tensor.byte_offset + relative_offset as u64,
        destination: dst.as_mut_ptr(),
        bytes: dst.len(),
        label: tensor.name.clone(),
    });
    Ok(())
}

fn queue_exl3_trellis_tp4(
    snapshot: &Path,
    projection: Exl3Projection<'_>,
    shard: ExpertIntermediateShard,
    hidden_size: usize,
    local_intermediate_size: usize,
    trellis_bits: usize,
    trellis_bytes: usize,
    files: &mut HashMap<PathBuf, Arc<File>>,
    dst: &mut [u8],
    requests: &mut Vec<Exl3ReadRequest>,
) -> Result<()> {
    anyhow::ensure!(
        dst.len() == trellis_bytes,
        "EXL3 queued compact trellis destination has {} bytes, expected {}",
        dst.len(),
        trellis_bytes
    );
    let tile_bytes = exl3_trellis_tile_bytes(trellis_bits)?;
    let input_tiles = projection.input_features / 16;
    let output_tiles = projection.output_features / 16;
    let local_tiles = local_intermediate_size / 16;
    if projection.input_features == hidden_size {
        anyhow::ensure!(
            projection.output_features == local_intermediate_size * shard.count,
            "EXL3 {:?} projection has unsupported {}x{} TP geometry",
            projection.kind,
            projection.input_features,
            projection.output_features
        );
        let source_row_bytes = output_tiles * tile_bytes;
        let local_row_bytes = local_tiles * tile_bytes;
        for input_tile in 0..input_tiles {
            let source_offset = input_tile * source_row_bytes + shard.rank * local_row_bytes;
            let destination_offset = input_tile * local_row_bytes;
            queue_exl3_tensor_window(
                snapshot,
                projection.trellis,
                source_offset,
                files,
                &mut dst[destination_offset..destination_offset + local_row_bytes],
                requests,
            )?;
        }
    } else {
        anyhow::ensure!(
            projection.input_features == local_intermediate_size * shard.count
                && projection.output_features == hidden_size,
            "EXL3 {:?} projection has unsupported {}x{} TP geometry",
            projection.kind,
            projection.input_features,
            projection.output_features
        );
        let local_input_tiles = input_tiles / shard.count;
        let bytes = local_input_tiles * output_tiles * tile_bytes;
        anyhow::ensure!(bytes == dst.len(), "EXL3 queued down trellis byte mismatch");
        queue_exl3_tensor_window(
            snapshot,
            projection.trellis,
            shard.rank * bytes,
            files,
            dst,
            requests,
        )?;
    }
    Ok(())
}

fn execute_exl3_read_requests(requests: &[Exl3ReadRequest]) -> Result<()> {
    // A gate/up TP4 projection is stored as one compact rank window per input
    // tile. Pro therefore produces several thousand small reads for one bounded
    // staging chunk. Waiting after only 256 requests leaves the NVMe below its
    // useful queue depth and serializes dozens of submit/completion waves per
    // chunk. Keep the queue bounded, but deep enough to cover the common chunk
    // in one or two waves.
    const QUEUE_DEPTH: usize = 8_192;
    let mut ring = match IoUring::new(QUEUE_DEPTH as u32) {
        Ok(ring) => ring,
        Err(error) => {
            eprintln!("ds4_flash_exl3_io_uring_unavailable error={error}");
            for request in requests {
                let dst = unsafe { slice::from_raw_parts_mut(request.destination, request.bytes) };
                request
                    .file
                    .read_exact_at(dst, request.source_offset)
                    .with_context(|| format!("reading compact EXL3 window {}", request.label))?;
            }
            return Ok(());
        }
    };
    for (chunk_index, chunk) in requests.chunks(QUEUE_DEPTH).enumerate() {
        let request_start = chunk_index * QUEUE_DEPTH;
        for (local_index, request) in chunk.iter().enumerate() {
            let length: u32 = request
                .bytes
                .try_into()
                .context("EXL3 io_uring read length exceeds u32")?;
            let entry = opcode::Read::new(
                types::Fd(request.file.as_raw_fd()),
                request.destination,
                length,
            )
            .offset(request.source_offset)
            .build()
            .user_data((request_start + local_index) as u64);
            unsafe {
                ring.submission()
                    .push(&entry)
                    .map_err(|_| anyhow::anyhow!("EXL3 io_uring submission queue is full"))?;
            }
        }
        ring.submit_and_wait(chunk.len())
            .context("submitting compact EXL3 TP4 reads")?;
        let mut completions = 0_usize;
        for completion in ring.completion() {
            let request_index: usize = completion
                .user_data()
                .try_into()
                .context("EXL3 io_uring completion index exceeds usize")?;
            let request = requests
                .get(request_index)
                .context("EXL3 io_uring completion index is out of range")?;
            let result = completion.result();
            if result < 0 {
                return Err(std::io::Error::from_raw_os_error(-result))
                    .with_context(|| format!("reading compact EXL3 window {}", request.label));
            }
            anyhow::ensure!(
                result as usize == request.bytes,
                "EXL3 io_uring short read for {}: {result}/{} bytes",
                request.label,
                request.bytes
            );
            completions += 1;
        }
        anyhow::ensure!(
            completions == chunk.len(),
            "EXL3 io_uring returned {completions}/{} completions",
            chunk.len()
        );
    }
    Ok(())
}

fn validate_exl3_mcg_marker(
    snapshot: &Path,
    projection: Exl3Projection<'_>,
    files: &mut HashMap<PathBuf, Arc<File>>,
) -> Result<()> {
    let mut marker = [0_u8; std::mem::size_of::<u32>()];
    read_exl3_tensor_window(snapshot, projection.mcg, 0, files, &mut marker)?;
    anyhow::ensure!(
        u32::from_le_bytes(marker) == DS4_FLASH_EXL3_MCG_MARKER,
        "EXL3 projection {} has unsupported MCG marker 0x{:08x}",
        projection.trellis.name,
        u32::from_le_bytes(marker)
    );
    Ok(())
}

const EXL3_FULL_GATE_TRELLIS: usize = 0;
const EXL3_FULL_UP_TRELLIS: usize = 1;
const EXL3_FULL_DOWN_TRELLIS: usize = 2;
const EXL3_FULL_GATE_SUH: usize = 3;
const EXL3_FULL_UP_SUH: usize = 4;
const EXL3_FULL_GATE_SVH: usize = 5;
const EXL3_FULL_UP_SVH: usize = 6;
const EXL3_FULL_DOWN_SUH: usize = 7;
const EXL3_FULL_DOWN_SVH: usize = 8;
const EXL3_FULL_GATE_MCG: usize = 9;
const EXL3_FULL_UP_MCG: usize = 10;
const EXL3_FULL_DOWN_MCG: usize = 11;
const EXL3_FULL_COMPONENTS: usize = 12;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Exl3FullExpertLayout {
    offsets: [usize; EXL3_FULL_COMPONENTS],
    bytes: [usize; EXL3_FULL_COMPONENTS],
    expert_bytes: usize,
}

impl Exl3FullExpertLayout {
    fn for_projections(
        gate: Exl3Projection<'_>,
        up: Exl3Projection<'_>,
        down: Exl3Projection<'_>,
    ) -> Result<Self> {
        let usize_bytes = |tensor: &TensorInfo| {
            usize::try_from(tensor.byte_length)
                .with_context(|| format!("EXL3 tensor {} exceeds usize", tensor.name))
        };
        let bytes = [
            usize_bytes(gate.trellis)?,
            usize_bytes(up.trellis)?,
            usize_bytes(down.trellis)?,
            usize_bytes(gate.suh)?,
            usize_bytes(up.suh)?,
            usize_bytes(gate.svh)?,
            usize_bytes(up.svh)?,
            usize_bytes(down.suh)?,
            usize_bytes(down.svh)?,
            usize_bytes(gate.mcg)?,
            usize_bytes(up.mcg)?,
            usize_bytes(down.mcg)?,
        ];
        anyhow::ensure!(
            bytes[EXL3_FULL_GATE_MCG..]
                .iter()
                .all(|bytes| *bytes == std::mem::size_of::<u32>()),
            "EXL3 MCG marker tensors must contain one u32"
        );
        let mut offsets = [0_usize; EXL3_FULL_COMPONENTS];
        let mut expert_bytes = 0_usize;
        for (index, component_bytes) in bytes.iter().copied().enumerate() {
            offsets[index] = expert_bytes;
            expert_bytes = expert_bytes
                .checked_add(component_bytes)
                .context("full EXL3 expert layout size overflow")?;
        }
        Ok(Self {
            offsets,
            bytes,
            expert_bytes,
        })
    }

    fn range(&self, expert_base: usize, component: usize) -> Result<Range<usize>> {
        let start = expert_base
            .checked_add(self.offsets[component])
            .context("full EXL3 component offset overflow")?;
        let end = start
            .checked_add(self.bytes[component])
            .context("full EXL3 component extent overflow")?;
        Ok(start..end)
    }
}

struct LoadedExl3FullLayer {
    expert_ids: Vec<usize>,
    layout: Exl3FullExpertLayout,
    bytes: LoadedExl3FullLayerBytes,
    source_bytes: u64,
    source_requests: usize,
    source_spans: usize,
    direct_io: bool,
}

enum LoadedExl3FullLayerBytes {
    Packed(Vec<u8>),
    DirectSpans {
        buffers: Vec<RouteCudaAlignedReadBuffer>,
        component_locations: Vec<[Exl3FullComponentLocation; EXL3_FULL_COMPONENTS]>,
    },
}

#[derive(Clone, Copy)]
struct Exl3FullComponentLocation {
    span_index: usize,
    start: usize,
    end: usize,
}

impl Exl3FullComponentLocation {
    const MISSING: Self = Self {
        span_index: usize::MAX,
        start: 0,
        end: 0,
    };
}

impl LoadedExl3FullLayer {
    fn component(&self, source_index: usize, component: usize) -> Result<&[u8]> {
        anyhow::ensure!(
            source_index < self.expert_ids.len() && component < EXL3_FULL_COMPONENTS,
            "EXL3 full-layer component index is out of range"
        );
        match &self.bytes {
            LoadedExl3FullLayerBytes::Packed(bytes) => {
                let source_base = source_index
                    .checked_mul(self.layout.expert_bytes)
                    .context("EXL3 packed source expert offset overflow")?;
                Ok(&bytes[self.layout.range(source_base, component)?])
            }
            LoadedExl3FullLayerBytes::DirectSpans {
                buffers,
                component_locations,
            } => {
                let location = component_locations[source_index][component];
                let buffer = buffers
                    .get(location.span_index)
                    .context("EXL3 direct component is missing its source span")?;
                Ok(&buffer.requested_slice()[location.start..location.end])
            }
        }
    }
}

#[derive(Clone)]
struct Exl3FullTensorWindow {
    source_index: usize,
    component: usize,
    info: TensorInfo,
}

struct Exl3FullPhysicalSpan {
    file: String,
    source_offset: u64,
    bytes: usize,
    window_indices: Vec<usize>,
}

fn validate_exl3_mcg_bytes(marker: &[u8], label: &str) -> Result<()> {
    let marker: [u8; std::mem::size_of::<u32>()] = marker
        .try_into()
        .with_context(|| format!("EXL3 {label} MCG marker has {} bytes", marker.len()))?;
    anyhow::ensure!(
        u32::from_le_bytes(marker) == DS4_FLASH_EXL3_MCG_MARKER,
        "EXL3 {label} has unsupported MCG marker 0x{:08x}",
        u32::from_le_bytes(marker)
    );
    Ok(())
}

fn load_exl3_full_layer_experts(
    catalog: &TensorCatalog,
    layer_id: usize,
    expert_ids: &[usize],
) -> Result<LoadedExl3FullLayer> {
    let first_id = *expert_ids
        .first()
        .context("cooperative EXL3 preload has no source experts")?;
    let first = exl3_expert(catalog, layer_id, first_id)?;
    let layout = Exl3FullExpertLayout::for_projections(first.gate, first.up, first.down)?;
    let total_bytes = layout
        .expert_bytes
        .checked_mul(expert_ids.len())
        .context("cooperative full EXL3 layer byte count overflow")?;
    let snapshot = Path::new(&catalog.snapshot_path);
    let mut windows = Vec::with_capacity(expert_ids.len() * EXL3_FULL_COMPONENTS);
    for (source_index, &expert_id) in expert_ids.iter().enumerate() {
        let expert = exl3_expert(catalog, layer_id, expert_id)?;
        anyhow::ensure!(
            Exl3FullExpertLayout::for_projections(expert.gate, expert.up, expert.down)? == layout,
            "cooperative EXL3 layer {layer_id} expert {expert_id} changed physical layout"
        );
        for (tensor, component) in [
            (expert.gate.trellis, EXL3_FULL_GATE_TRELLIS),
            (expert.up.trellis, EXL3_FULL_UP_TRELLIS),
            (expert.down.trellis, EXL3_FULL_DOWN_TRELLIS),
            (expert.gate.suh, EXL3_FULL_GATE_SUH),
            (expert.up.suh, EXL3_FULL_UP_SUH),
            (expert.gate.svh, EXL3_FULL_GATE_SVH),
            (expert.up.svh, EXL3_FULL_UP_SVH),
            (expert.down.suh, EXL3_FULL_DOWN_SUH),
            (expert.down.svh, EXL3_FULL_DOWN_SVH),
            (expert.gate.mcg, EXL3_FULL_GATE_MCG),
            (expert.up.mcg, EXL3_FULL_UP_MCG),
            (expert.down.mcg, EXL3_FULL_DOWN_MCG),
        ] {
            windows.push(Exl3FullTensorWindow {
                source_index,
                component,
                info: tensor.clone(),
            });
        }
    }
    let requested_bytes = windows.iter().try_fold(0_usize, |total, window| {
        total
            .checked_add(
                usize::try_from(window.info.byte_length)
                    .context("EXL3 full-layer tensor length exceeds usize")?,
            )
            .context("EXL3 full-layer requested bytes overflow")
    })?;
    anyhow::ensure!(
        requested_bytes == total_bytes,
        "EXL3 full-layer requested {requested_bytes} bytes, expected {total_bytes}"
    );
    let mut physical_order = (0..windows.len()).collect::<Vec<_>>();
    physical_order.sort_unstable_by(|&left, &right| {
        windows[left]
            .info
            .file
            .cmp(&windows[right].info.file)
            .then_with(|| {
                windows[left]
                    .info
                    .byte_offset
                    .cmp(&windows[right].info.byte_offset)
            })
    });
    anyhow::ensure!(
        !physical_order.is_empty(),
        "EXL3 full-layer preload has no tensor windows"
    );
    let mut physical_spans = Vec::<Exl3FullPhysicalSpan>::new();
    for index in physical_order {
        let window = &windows[index];
        let window_bytes: usize = window
            .info
            .byte_length
            .try_into()
            .context("EXL3 full-layer tensor length exceeds usize")?;
        let append = if let Some(span) = physical_spans.last() {
            let span_bytes: u64 = span
                .bytes
                .try_into()
                .context("EXL3 physical span length exceeds u64")?;
            span.file == window.info.file
                && span
                    .source_offset
                    .checked_add(span_bytes)
                    .is_some_and(|expected| expected == window.info.byte_offset)
        } else {
            false
        };
        if append {
            let span = physical_spans
                .last_mut()
                .expect("EXL3 physical span exists after adjacency check");
            span.bytes = span
                .bytes
                .checked_add(window_bytes)
                .context("EXL3 physical span byte count overflow")?;
            span.window_indices.push(index);
        } else {
            physical_spans.push(Exl3FullPhysicalSpan {
                file: window.info.file.clone(),
                source_offset: window.info.byte_offset,
                bytes: window_bytes,
                window_indices: vec![index],
            });
        }
    }
    const DIRECT_SPAN_LIMIT: usize = 8;
    let mut direct_buffers = Vec::with_capacity(physical_spans.len());
    let mut component_locations =
        vec![[Exl3FullComponentLocation::MISSING; EXL3_FULL_COMPONENTS]; expert_ids.len()];
    let direct_io = route_preload_direct_io() && physical_spans.len() <= DIRECT_SPAN_LIMIT;
    if direct_io {
        for (span_index, span) in physical_spans.iter().enumerate() {
            let source_path = snapshot.join(&span.file);
            let file = OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_DIRECT)
                .open(&source_path)
                .with_context(|| {
                    format!("opening {} for coalesced direct I/O", source_path.display())
                })?;
            let file_bytes = file
                .metadata()
                .with_context(|| format!("reading metadata for {}", source_path.display()))?
                .len();
            let buffered_file = File::open(&source_path).with_context(|| {
                format!(
                    "opening {} for coalesced buffered tail I/O",
                    source_path.display()
                )
            })?;
            let buffer = RouteCudaAlignedReadBuffer::new_with_buffered_tail(
                &file,
                &buffered_file,
                file_bytes,
                span.source_offset,
                span.bytes,
            )?;
            for &window_index in &span.window_indices {
                let window = &windows[window_index];
                let start: usize = window
                    .info
                    .byte_offset
                    .checked_sub(span.source_offset)
                    .context("EXL3 direct tensor precedes its physical span")?
                    .try_into()
                    .context("EXL3 direct tensor offset exceeds usize")?;
                let bytes: usize = window
                    .info
                    .byte_length
                    .try_into()
                    .context("EXL3 direct tensor length exceeds usize")?;
                let end = start
                    .checked_add(bytes)
                    .context("EXL3 direct tensor extent overflow")?;
                anyhow::ensure!(
                    end <= span.bytes,
                    "EXL3 direct tensor exceeds its physical span"
                );
                component_locations[window.source_index][window.component] =
                    Exl3FullComponentLocation {
                        span_index,
                        start,
                        end,
                    };
            }
            direct_buffers.push(buffer);
        }
    }
    let bytes = if direct_io {
        anyhow::ensure!(
            direct_buffers.len() == physical_spans.len()
                && component_locations
                    .iter()
                    .flatten()
                    .all(|location| location.span_index != usize::MAX),
            "EXL3 direct source map is incomplete"
        );
        LoadedExl3FullLayerBytes::DirectSpans {
            buffers: direct_buffers,
            component_locations,
        }
    } else {
        let mut bytes = vec![0_u8; total_bytes];
        let mut files = HashMap::<PathBuf, Arc<File>>::new();
        let mut requests = Vec::with_capacity(windows.len());
        for window in &windows {
            let base = window.source_index * layout.expert_bytes;
            queue_exl3_tensor_window(
                snapshot,
                &window.info,
                0,
                &mut files,
                &mut bytes[layout.range(base, window.component)?],
                &mut requests,
            )?;
        }
        requests.sort_unstable_by_key(|request| (request.file.as_raw_fd(), request.source_offset));
        execute_exl3_read_requests(&requests)?;
        LoadedExl3FullLayerBytes::Packed(bytes)
    };
    let loaded = LoadedExl3FullLayer {
        expert_ids: expert_ids.to_vec(),
        layout,
        bytes,
        source_bytes: total_bytes as u64,
        source_requests: windows.len(),
        source_spans: physical_spans.len(),
        direct_io,
    };
    for (source_index, &expert_id) in expert_ids.iter().enumerate() {
        for (component, label) in [
            (EXL3_FULL_GATE_MCG, "gate"),
            (EXL3_FULL_UP_MCG, "up"),
            (EXL3_FULL_DOWN_MCG, "down"),
        ] {
            validate_exl3_mcg_bytes(
                loaded.component(source_index, component)?,
                &format!("layer {layer_id} expert {expert_id} {label}"),
            )?;
        }
    }
    Ok(loaded)
}

fn compact_exl3_trellis_tp4_bytes(
    source: &[u8],
    projection: Exl3Projection<'_>,
    shard: ExpertIntermediateShard,
    hidden_size: usize,
    local_intermediate_size: usize,
    trellis_bits: usize,
    destination: &mut [u8],
) -> Result<()> {
    let tile_bytes = exl3_trellis_tile_bytes(trellis_bits)?;
    let input_tiles = projection.input_features / 16;
    let output_tiles = projection.output_features / 16;
    let local_tiles = local_intermediate_size / 16;
    anyhow::ensure!(
        source.len() == input_tiles * output_tiles * tile_bytes,
        "EXL3 full trellis source byte mismatch for {}",
        projection.trellis.name
    );
    if projection.input_features == hidden_size {
        let source_row_bytes = output_tiles * tile_bytes;
        let local_row_bytes = local_tiles * tile_bytes;
        anyhow::ensure!(
            destination.len() == input_tiles * local_row_bytes,
            "EXL3 compact gate/up trellis destination byte mismatch"
        );
        for input_tile in 0..input_tiles {
            let source_start = input_tile * source_row_bytes + shard.rank * local_row_bytes;
            let destination_start = input_tile * local_row_bytes;
            destination[destination_start..destination_start + local_row_bytes]
                .copy_from_slice(&source[source_start..source_start + local_row_bytes]);
        }
    } else {
        let shard_bytes = source.len() / shard.count;
        anyhow::ensure!(
            projection.output_features == hidden_size
                && projection.input_features == local_intermediate_size * shard.count
                && destination.len() == shard_bytes,
            "EXL3 compact down trellis geometry mismatch"
        );
        let source_start = shard.rank * shard_bytes;
        destination.copy_from_slice(&source[source_start..source_start + shard_bytes]);
    }
    Ok(())
}

impl RouteCudaDs4FlashExl3LayerExpertSlab {
    fn new(library: Arc<NativeLibrary>, catalog: &TensorCatalog, layer_id: usize) -> Result<Self> {
        let profile = Ds4Exl3AotProfile::from_catalog_layer(catalog, layer_id)?;
        let geometry = profile.geometry;
        let _ = exl3_expert(catalog, layer_id, 0)
            .with_context(|| format!("validating EXL3 {} layer {layer_id}", profile.label()))?;
        let allocation = |bytes: usize, label: &str| {
            OwnedDeviceAllocation::new(
                Arc::clone(&library),
                bytes,
                &format!("EXL3 {} layer {layer_id} {label}", profile.label()),
            )
            .map(Arc::new)
        };
        let usize_bytes = |bytes: u64, label: &str| {
            usize::try_from(bytes)
                .with_context(|| format!("EXL3 {} {label} exceeds usize", profile.label()))
        };
        let w13_trellis = allocation(
            usize_bytes(geometry.w13_trellis_bytes, "W13 trellis")?,
            &format!("TP4 W13 K{} trellis", geometry.trellis_bits),
        )?;
        let w2_trellis = allocation(
            usize_bytes(geometry.w2_trellis_bytes, "W2 trellis")?,
            &format!("TP4 W2 K{} trellis", geometry.trellis_bits),
        )?;
        let hidden_rotation_bytes = usize_bytes(
            geometry.hidden_rotation_table_bytes,
            "hidden rotation table",
        )?;
        let gate_suh = allocation(hidden_rotation_bytes, "gate SUH rotations")?;
        let up_suh = allocation(hidden_rotation_bytes, "up SUH rotations")?;
        let intermediate_rotations = allocation(
            usize_bytes(
                geometry.intermediate_rotation_bytes,
                "intermediate rotations",
            )?,
            "intermediate rotations",
        )?;
        let down_svh = allocation(hidden_rotation_bytes, "down SVH rotations")?;
        let expert_map = allocation(
            (geometry.experts + 1) * std::mem::size_of::<i32>(),
            "identity expert map",
        )?;
        let dummy_scale = allocation(16, "dummy scale")?;
        let trellis_lut = allocation(SQG_XOR_CHEB_T12_LUT_BYTES, "SQG-XOR-Cheb T12 LUT")?;
        let global_scale = allocation(
            geometry.experts * std::mem::size_of::<f32>(),
            "global scales",
        )?;
        let identity = (0..=geometry.experts as i32)
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        let ones = (0..geometry.experts)
            .flat_map(|_| 1.0_f32.to_le_bytes())
            .collect::<Vec<_>>();
        library
            .copy_h2d(expert_map.buffer(), &identity)
            .context("uploading EXL3 identity expert map")?;
        library
            .copy_h2d(dummy_scale.buffer(), &[0_u8; 16])
            .context("uploading EXL3 dummy scale")?;
        library
            .copy_h2d(trellis_lut.buffer(), sqg_xor_cheb_t12_lut_bytes())
            .context("uploading EXL3 SQG-XOR-Cheb T12 LUT")?;
        library
            .copy_h2d(global_scale.buffer(), &ones)
            .context("uploading EXL3 global scales")?;
        Ok(Self {
            layer_id,
            expert_count: geometry.experts,
            profile,
            w13_trellis,
            w2_trellis,
            gate_suh,
            up_suh,
            intermediate_rotations,
            down_svh,
            expert_map,
            dummy_scale,
            trellis_lut,
            global_scale,
        })
    }

    fn store_layer_experts_direct(
        &self,
        catalog: &TensorCatalog,
        expert_ids: &[usize],
        shard: ExpertIntermediateShard,
        library: Arc<NativeLibrary>,
        workspace: &mut RouteCudaWorkspace,
        cuda_stream: *mut c_void,
    ) -> Result<u64> {
        anyhow::ensure!(
            shard.count == 4 && shard.rank < shard.count,
            "EXL3 Flash preload requires a valid TP4 shard, got rank {}/{}",
            shard.rank,
            shard.count
        );
        let geometry = self.profile.geometry;
        let trellis = usize::try_from(geometry.projection_trellis_bytes)
            .context("EXL3 projection trellis bytes exceed usize")?;
        let hidden_rotation = geometry.hidden_size * std::mem::size_of::<u16>();
        let local_rotation = geometry.local_intermediate_size * std::mem::size_of::<u16>();
        let gate_trellis = 0;
        let up_trellis = gate_trellis + trellis;
        let down_trellis = up_trellis + trellis;
        let gate_suh = down_trellis + trellis;
        let up_suh = gate_suh + hidden_rotation;
        let gate_svh = up_suh + hidden_rotation;
        let up_svh = gate_svh + local_rotation;
        let down_suh = up_svh + local_rotation;
        let down_svh = down_suh + local_rotation;
        let expert_bytes = down_svh + hidden_rotation;
        // Keep the pinned staging window bounded as Pro's per-expert payload
        // grows beyond Flash. It remains a streaming chunk, never a model slab.
        let chunk_experts = ((32 * 1024 * 1024) / expert_bytes).clamp(1, 16);

        RouteCudaWorkspace::ensure_pinned_buffer(
            &mut workspace.pinned_projection_weight,
            Arc::clone(&library),
            chunk_experts * expert_bytes,
            "EXL3 direct resident preload",
        )?;
        let snapshot = Path::new(&catalog.snapshot_path);
        let mut files = HashMap::<PathBuf, Arc<File>>::new();
        let mut source_bytes = 0_u64;
        let mut request_build_elapsed = Duration::ZERO;
        let mut source_read_elapsed = Duration::ZERO;
        let mut host_to_device_elapsed = Duration::ZERO;
        let mut preload_chunks = 0_usize;
        let mut max_chunk_requests = 0_usize;

        for expert_chunk in expert_ids.chunks(chunk_experts) {
            preload_chunks += 1;
            let chunk_bytes = expert_chunk.len() * expert_bytes;
            let staging_buffer;
            {
                let request_build_started = Instant::now();
                let staging = workspace
                    .pinned_projection_weight
                    .as_mut()
                    .expect("EXL3 pinned preload buffer ensured above");
                let dst = staging.as_mut_slice(chunk_bytes)?;
                let mut requests = Vec::with_capacity(expert_chunk.len() * 520);
                for (chunk_index, &expert_id) in expert_chunk.iter().enumerate() {
                    anyhow::ensure!(
                        expert_id < self.expert_count,
                        "EXL3 Flash layer {} expert {expert_id} exceeds {}",
                        self.layer_id,
                        self.expert_count
                    );
                    let expert = exl3_expert(catalog, self.layer_id, expert_id)?;
                    let base = chunk_index * expert_bytes;
                    queue_exl3_trellis_tp4(
                        snapshot,
                        expert.gate,
                        shard,
                        geometry.hidden_size,
                        geometry.local_intermediate_size,
                        geometry.trellis_bits,
                        trellis,
                        &mut files,
                        &mut dst[base + gate_trellis..base + gate_trellis + trellis],
                        &mut requests,
                    )?;
                    queue_exl3_trellis_tp4(
                        snapshot,
                        expert.up,
                        shard,
                        geometry.hidden_size,
                        geometry.local_intermediate_size,
                        geometry.trellis_bits,
                        trellis,
                        &mut files,
                        &mut dst[base + up_trellis..base + up_trellis + trellis],
                        &mut requests,
                    )?;
                    queue_exl3_trellis_tp4(
                        snapshot,
                        expert.down,
                        shard,
                        geometry.hidden_size,
                        geometry.local_intermediate_size,
                        geometry.trellis_bits,
                        trellis,
                        &mut files,
                        &mut dst[base + down_trellis..base + down_trellis + trellis],
                        &mut requests,
                    )?;
                    queue_exl3_tensor_window(
                        snapshot,
                        expert.gate.suh,
                        0,
                        &mut files,
                        &mut dst[base + gate_suh..base + gate_suh + hidden_rotation],
                        &mut requests,
                    )?;
                    queue_exl3_tensor_window(
                        snapshot,
                        expert.up.suh,
                        0,
                        &mut files,
                        &mut dst[base + up_suh..base + up_suh + hidden_rotation],
                        &mut requests,
                    )?;
                    let local_rotation_start = shard.rank * local_rotation;
                    queue_exl3_tensor_window(
                        snapshot,
                        expert.gate.svh,
                        local_rotation_start,
                        &mut files,
                        &mut dst[base + gate_svh..base + gate_svh + local_rotation],
                        &mut requests,
                    )?;
                    queue_exl3_tensor_window(
                        snapshot,
                        expert.up.svh,
                        local_rotation_start,
                        &mut files,
                        &mut dst[base + up_svh..base + up_svh + local_rotation],
                        &mut requests,
                    )?;
                    queue_exl3_tensor_window(
                        snapshot,
                        expert.down.suh,
                        local_rotation_start,
                        &mut files,
                        &mut dst[base + down_suh..base + down_suh + local_rotation],
                        &mut requests,
                    )?;
                    queue_exl3_tensor_window(
                        snapshot,
                        expert.down.svh,
                        0,
                        &mut files,
                        &mut dst[base + down_svh..base + down_svh + hidden_rotation],
                        &mut requests,
                    )?;
                    for projection in [expert.gate, expert.up, expert.down] {
                        validate_exl3_mcg_marker(snapshot, projection, &mut files)?;
                    }
                    source_bytes = source_bytes
                        .checked_add(expert_bytes as u64 + 3 * std::mem::size_of::<u32>() as u64)
                        .context("EXL3 source byte count overflow")?;
                }
                request_build_elapsed += request_build_started.elapsed();
                max_chunk_requests = max_chunk_requests.max(requests.len());
                let source_read_started = Instant::now();
                execute_exl3_read_requests(&requests)?;
                source_read_elapsed += source_read_started.elapsed();
                staging_buffer = staging.buffer();
            }

            let host_to_device_started = Instant::now();
            for (chunk_index, &expert_id) in expert_chunk.iter().enumerate() {
                let host_base = chunk_index * expert_bytes;
                let intermediate_base = expert_id * 3 * local_rotation;
                let copies = [
                    (
                        gate_trellis,
                        trellis,
                        device_buffer_byte_view(
                            self.w13_trellis.buffer(),
                            expert_id * trellis,
                            trellis,
                            "EXL3 W13 gate trellis",
                        )?,
                    ),
                    (
                        up_trellis,
                        trellis,
                        device_buffer_byte_view(
                            self.w13_trellis.buffer(),
                            (self.expert_count + expert_id) * trellis,
                            trellis,
                            "EXL3 W13 up trellis",
                        )?,
                    ),
                    (
                        down_trellis,
                        trellis,
                        device_buffer_byte_view(
                            self.w2_trellis.buffer(),
                            expert_id * trellis,
                            trellis,
                            "EXL3 W2 down trellis",
                        )?,
                    ),
                    (
                        gate_suh,
                        hidden_rotation,
                        device_buffer_byte_view(
                            self.gate_suh.buffer(),
                            expert_id * hidden_rotation,
                            hidden_rotation,
                            "EXL3 gate SUH",
                        )?,
                    ),
                    (
                        up_suh,
                        hidden_rotation,
                        device_buffer_byte_view(
                            self.up_suh.buffer(),
                            expert_id * hidden_rotation,
                            hidden_rotation,
                            "EXL3 up SUH",
                        )?,
                    ),
                    (
                        gate_svh,
                        local_rotation,
                        device_buffer_byte_view(
                            self.intermediate_rotations.buffer(),
                            intermediate_base,
                            local_rotation,
                            "EXL3 gate SVH",
                        )?,
                    ),
                    (
                        up_svh,
                        local_rotation,
                        device_buffer_byte_view(
                            self.intermediate_rotations.buffer(),
                            intermediate_base + local_rotation,
                            local_rotation,
                            "EXL3 up SVH",
                        )?,
                    ),
                    (
                        down_suh,
                        local_rotation,
                        device_buffer_byte_view(
                            self.intermediate_rotations.buffer(),
                            intermediate_base + 2 * local_rotation,
                            local_rotation,
                            "EXL3 down SUH",
                        )?,
                    ),
                    (
                        down_svh,
                        hidden_rotation,
                        device_buffer_byte_view(
                            self.down_svh.buffer(),
                            expert_id * hidden_rotation,
                            hidden_rotation,
                            "EXL3 down SVH",
                        )?,
                    ),
                ];
                for (source_offset, bytes, destination) in copies {
                    let source = host_buffer_byte_view(
                        staging_buffer,
                        host_base + source_offset,
                        bytes,
                        "EXL3 compact pinned source",
                    )?;
                    unsafe {
                        library
                            .copy_host_buffer_h2d_async(destination, source, bytes, cuda_stream)
                            .context("copying compact EXL3 TP4 bytes to resident slab")?;
                    }
                }
            }
            unsafe {
                library
                    .cuda_stream_synchronize(cuda_stream)
                    .context("synchronizing EXL3 direct resident preload chunk")?;
            }
            host_to_device_elapsed += host_to_device_started.elapsed();
        }
        eprintln!(
            "ds4_flash_exl3_direct_preload_breakdown layer_id={} tp_rank={} chunks={} chunk_experts={} max_chunk_requests={} request_build_ms={:.3} source_read_ms={:.3} host_to_device_ms={:.3}",
            self.layer_id,
            shard.rank,
            preload_chunks,
            chunk_experts,
            max_chunk_requests,
            request_build_elapsed.as_secs_f64() * 1_000.0,
            source_read_elapsed.as_secs_f64() * 1_000.0,
            host_to_device_elapsed.as_secs_f64() * 1_000.0,
        );
        Ok(source_bytes)
    }

    #[allow(clippy::too_many_arguments)]
    fn store_loaded_full_layer_experts(
        &self,
        catalog: &TensorCatalog,
        loaded: &LoadedExl3FullLayer,
        destination_expert_ids: &[usize],
        target_shard: ExpertIntermediateShard,
        library: Arc<NativeLibrary>,
        workspace: &mut RouteCudaWorkspace,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        anyhow::ensure!(
            loaded.expert_ids.len() == destination_expert_ids.len(),
            "EXL3 cooperative source/destination expert count mismatch: {}/{}",
            loaded.expert_ids.len(),
            destination_expert_ids.len()
        );
        let geometry = self.profile.geometry;
        let trellis = usize::try_from(geometry.projection_trellis_bytes)
            .context("EXL3 projection trellis bytes exceed usize")?;
        let hidden_rotation = geometry.hidden_size * std::mem::size_of::<u16>();
        let local_rotation = geometry.local_intermediate_size * std::mem::size_of::<u16>();
        let gate_trellis = 0;
        let up_trellis = gate_trellis + trellis;
        let down_trellis = up_trellis + trellis;
        let gate_suh = down_trellis + trellis;
        let up_suh = gate_suh + hidden_rotation;
        let gate_svh = up_suh + hidden_rotation;
        let up_svh = gate_svh + local_rotation;
        let down_suh = up_svh + local_rotation;
        let down_svh = down_suh + local_rotation;
        let expert_bytes = down_svh + hidden_rotation;
        let chunk_experts = ((32 * 1024 * 1024) / expert_bytes).clamp(1, 16);
        RouteCudaWorkspace::ensure_pinned_buffer(
            &mut workspace.pinned_projection_weight,
            Arc::clone(&library),
            chunk_experts * expert_bytes,
            "EXL3 cooperative compact preload",
        )?;

        for chunk_start in (0..loaded.expert_ids.len()).step_by(chunk_experts) {
            let chunk_end = (chunk_start + chunk_experts).min(loaded.expert_ids.len());
            let chunk_len = chunk_end - chunk_start;
            let staging_buffer;
            {
                let staging = workspace
                    .pinned_projection_weight
                    .as_mut()
                    .expect("EXL3 cooperative pinned preload buffer ensured above");
                let dst = staging.as_mut_slice(chunk_len * expert_bytes)?;
                for chunk_index in 0..chunk_len {
                    let source_index = chunk_start + chunk_index;
                    let source_expert_id = loaded.expert_ids[source_index];
                    let expert = exl3_expert(catalog, self.layer_id, source_expert_id)?;
                    let destination_base = chunk_index * expert_bytes;
                    compact_exl3_trellis_tp4_bytes(
                        loaded.component(source_index, EXL3_FULL_GATE_TRELLIS)?,
                        expert.gate,
                        target_shard,
                        geometry.hidden_size,
                        geometry.local_intermediate_size,
                        geometry.trellis_bits,
                        &mut dst[destination_base + gate_trellis
                            ..destination_base + gate_trellis + trellis],
                    )?;
                    compact_exl3_trellis_tp4_bytes(
                        loaded.component(source_index, EXL3_FULL_UP_TRELLIS)?,
                        expert.up,
                        target_shard,
                        geometry.hidden_size,
                        geometry.local_intermediate_size,
                        geometry.trellis_bits,
                        &mut dst[destination_base + up_trellis
                            ..destination_base + up_trellis + trellis],
                    )?;
                    compact_exl3_trellis_tp4_bytes(
                        loaded.component(source_index, EXL3_FULL_DOWN_TRELLIS)?,
                        expert.down,
                        target_shard,
                        geometry.hidden_size,
                        geometry.local_intermediate_size,
                        geometry.trellis_bits,
                        &mut dst[destination_base + down_trellis
                            ..destination_base + down_trellis + trellis],
                    )?;
                    for (component, destination_offset, bytes, source_offset) in [
                        (EXL3_FULL_GATE_SUH, gate_suh, hidden_rotation, 0),
                        (EXL3_FULL_UP_SUH, up_suh, hidden_rotation, 0),
                        (
                            EXL3_FULL_GATE_SVH,
                            gate_svh,
                            local_rotation,
                            target_shard.rank * local_rotation,
                        ),
                        (
                            EXL3_FULL_UP_SVH,
                            up_svh,
                            local_rotation,
                            target_shard.rank * local_rotation,
                        ),
                        (
                            EXL3_FULL_DOWN_SUH,
                            down_suh,
                            local_rotation,
                            target_shard.rank * local_rotation,
                        ),
                        (EXL3_FULL_DOWN_SVH, down_svh, hidden_rotation, 0),
                    ] {
                        let full = loaded.component(source_index, component)?;
                        anyhow::ensure!(
                            source_offset + bytes <= full.len(),
                            "EXL3 cooperative rotation slice exceeds source component"
                        );
                        dst[destination_base + destination_offset
                            ..destination_base + destination_offset + bytes]
                            .copy_from_slice(&full[source_offset..source_offset + bytes]);
                    }
                }
                staging_buffer = staging.buffer();
            }

            for chunk_index in 0..chunk_len {
                let destination_expert_id = destination_expert_ids[chunk_start + chunk_index];
                anyhow::ensure!(
                    destination_expert_id < self.expert_count,
                    "EXL3 cooperative destination expert {destination_expert_id} exceeds {}",
                    self.expert_count
                );
                let host_base = chunk_index * expert_bytes;
                let intermediate_base = destination_expert_id * 3 * local_rotation;
                let copies = [
                    (
                        gate_trellis,
                        trellis,
                        device_buffer_byte_view(
                            self.w13_trellis.buffer(),
                            destination_expert_id * trellis,
                            trellis,
                            "EXL3 cooperative W13 gate trellis",
                        )?,
                    ),
                    (
                        up_trellis,
                        trellis,
                        device_buffer_byte_view(
                            self.w13_trellis.buffer(),
                            (self.expert_count + destination_expert_id) * trellis,
                            trellis,
                            "EXL3 cooperative W13 up trellis",
                        )?,
                    ),
                    (
                        down_trellis,
                        trellis,
                        device_buffer_byte_view(
                            self.w2_trellis.buffer(),
                            destination_expert_id * trellis,
                            trellis,
                            "EXL3 cooperative W2 down trellis",
                        )?,
                    ),
                    (
                        gate_suh,
                        hidden_rotation,
                        device_buffer_byte_view(
                            self.gate_suh.buffer(),
                            destination_expert_id * hidden_rotation,
                            hidden_rotation,
                            "EXL3 cooperative gate SUH",
                        )?,
                    ),
                    (
                        up_suh,
                        hidden_rotation,
                        device_buffer_byte_view(
                            self.up_suh.buffer(),
                            destination_expert_id * hidden_rotation,
                            hidden_rotation,
                            "EXL3 cooperative up SUH",
                        )?,
                    ),
                    (
                        gate_svh,
                        local_rotation,
                        device_buffer_byte_view(
                            self.intermediate_rotations.buffer(),
                            intermediate_base,
                            local_rotation,
                            "EXL3 cooperative gate SVH",
                        )?,
                    ),
                    (
                        up_svh,
                        local_rotation,
                        device_buffer_byte_view(
                            self.intermediate_rotations.buffer(),
                            intermediate_base + local_rotation,
                            local_rotation,
                            "EXL3 cooperative up SVH",
                        )?,
                    ),
                    (
                        down_suh,
                        local_rotation,
                        device_buffer_byte_view(
                            self.intermediate_rotations.buffer(),
                            intermediate_base + 2 * local_rotation,
                            local_rotation,
                            "EXL3 cooperative down SUH",
                        )?,
                    ),
                    (
                        down_svh,
                        hidden_rotation,
                        device_buffer_byte_view(
                            self.down_svh.buffer(),
                            destination_expert_id * hidden_rotation,
                            hidden_rotation,
                            "EXL3 cooperative down SVH",
                        )?,
                    ),
                ];
                for (source_offset, bytes, destination) in copies {
                    let source = host_buffer_byte_view(
                        staging_buffer,
                        host_base + source_offset,
                        bytes,
                        "EXL3 cooperative compact pinned source",
                    )?;
                    unsafe {
                        library
                            .copy_host_buffer_h2d_async(destination, source, bytes, cuda_stream)
                            .context("uploading cooperative compact EXL3 bytes")?;
                    }
                }
            }
            unsafe {
                library
                    .cuda_stream_synchronize(cuda_stream)
                    .context("synchronizing EXL3 cooperative compact upload")?;
            }
        }
        Ok(())
    }

    fn resident_weight_bytes(&self) -> u64 {
        (self.w13_trellis.capacity_bytes() + self.w2_trellis.capacity_bytes()) as u64
    }

    fn resident_scale_bytes(&self) -> u64 {
        (self.gate_suh.capacity_bytes()
            + self.up_suh.capacity_bytes()
            + self.intermediate_rotations.capacity_bytes()
            + self.down_svh.capacity_bytes()
            + self.expert_map.capacity_bytes()
            + self.dummy_scale.capacity_bytes()
            + self.trellis_lut.capacity_bytes()
            + self.global_scale.capacity_bytes()) as u64
    }

    fn aot_buffers(
        &self,
        workspace: Ds4FlashSparkAotRouteWorkspaceBuffers,
        input: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        topk_ids: Ds41rtDeviceBuffer,
        topk_weights: Ds41rtDeviceBuffer,
        packed_route_indices: Ds41rtDeviceBuffer,
    ) -> Ds41rtDs4FlashSparkExl3K2MoeBuffers {
        Ds41rtDs4FlashSparkExl3K2MoeBuffers {
            input,
            w13_trellis: self.w13_trellis.buffer(),
            w2_trellis: self.w2_trellis.buffer(),
            gate_suh: self.gate_suh.buffer(),
            up_suh: self.up_suh.buffer(),
            intermediate_rotations: self.intermediate_rotations.buffer(),
            down_svh: self.down_svh.buffer(),
            expert_map: self.expert_map.buffer(),
            dummy_scale: self.dummy_scale.buffer(),
            trellis_lut: self.trellis_lut.buffer(),
            global_scale: self.global_scale.buffer(),
            topk_ids,
            topk_weights,
            rotation_gate: workspace.rotation_gate,
            rotation_up: workspace.rotation_up,
            fc1_output: workspace.fc1_output,
            activated: workspace.activated,
            routed_output: workspace.routed_output,
            output_f32: workspace.output_f32,
            output_bf16: output,
            packed_route_indices,
            block_expert_ids: workspace.block_expert_ids,
            packed_route_count: workspace.packed_route_count,
            expert_counts: workspace.expert_counts,
            expert_offsets: workspace.expert_offsets,
            fc1_scratch: workspace.fc1_scratch,
            fc2_scratch: workspace.fc2_scratch,
            workspace: workspace.locks,
        }
    }

    fn decode_m1_aot_buffers(
        &self,
        buffers: Ds4FlashSparkDecodeM1DeviceBuffers,
    ) -> Ds41rtDs4FlashSparkExl3K2MoeBuffers {
        self.aot_buffers(
            Ds4FlashSparkAotRouteWorkspaceBuffers {
                packed_hidden: Ds41rtDeviceBuffer::default(),
                fc1_output: buffers.fc1,
                activated: buffers.activated,
                routed_output: buffers.routed_output,
                rotation_gate: buffers.rotation_gate,
                rotation_up: buffers.rotation_up,
                output_f32: buffers.output_f32,
                block_expert_ids: buffers.block_expert_ids,
                packed_route_count: buffers.packed_route_count,
                expert_counts: buffers.expert_counts,
                expert_offsets: buffers.expert_offsets,
                fc1_scratch: buffers.fc1_scratch,
                fc2_scratch: buffers.fc2_scratch,
                locks: buffers.locks,
            },
            buffers.input,
            buffers.output,
            buffers.route_indices,
            buffers.route_weights,
            buffers.route_indices,
        )
    }

    fn prefill_aot_buffers(
        &self,
        buffers: Ds4FlashSparkPrefillDeviceBuffers,
    ) -> Ds41rtDs4FlashSparkExl3K2MoeBuffers {
        self.aot_buffers(
            Ds4FlashSparkAotRouteWorkspaceBuffers {
                packed_hidden: Ds41rtDeviceBuffer::default(),
                fc1_output: buffers.fc1,
                activated: buffers.activated,
                routed_output: buffers.routed_output,
                rotation_gate: buffers.rotation_gate,
                rotation_up: buffers.rotation_up,
                output_f32: buffers.output_f32,
                block_expert_ids: buffers.block_expert_ids,
                packed_route_count: buffers.packed_route_count,
                expert_counts: buffers.expert_counts,
                expert_offsets: buffers.expert_offsets,
                fc1_scratch: buffers.fc1_scratch,
                fc2_scratch: buffers.fc2_scratch,
                locks: buffers.locks,
            },
            buffers.input,
            buffers.output,
            buffers.topk_ids,
            buffers.topk_weights,
            buffers.packed_route_indices,
        )
    }
}

impl RouteCudaDs4FlashExl3MixedLayerExpertSlab {
    fn new(library: Arc<NativeLibrary>, catalog: &TensorCatalog, layer_id: usize) -> Result<Self> {
        anyhow::ensure!(
            catalog.facts.variant == ModelVariant::Flash
                && is_deepseek_v4_mixed_exl3_recipe(&catalog.facts.quantization_recipe),
            "mixed K2/K3 EXL3 slabs require a DeepSeek V4 Flash mixed artifact"
        );
        let mut tier0_gate_expert_ids = Vec::new();
        let mut tier1_gate_expert_ids = Vec::new();
        let mut tier0_up_expert_ids = Vec::new();
        let mut tier1_up_expert_ids = Vec::new();
        let mut tier0_down_expert_ids = Vec::new();
        let mut tier1_down_expert_ids = Vec::new();
        for expert_id in 0..catalog.facts.routed_experts {
            let expert = exl3_expert(catalog, layer_id, expert_id)?;
            for (projection, tier0, tier1) in [
                (
                    expert.gate,
                    &mut tier0_gate_expert_ids,
                    &mut tier1_gate_expert_ids,
                ),
                (
                    expert.up,
                    &mut tier0_up_expert_ids,
                    &mut tier1_up_expert_ids,
                ),
                (
                    expert.down,
                    &mut tier0_down_expert_ids,
                    &mut tier1_down_expert_ids,
                ),
            ] {
                match exl3_projection_trellis_bits(projection)? {
                    2 => tier0.push(expert_id),
                    3 => tier1.push(expert_id),
                    bits => anyhow::bail!("unsupported mixed EXL3 tier K{bits}"),
                }
            }
        }
        for (name, tier0, tier1) in [
            ("gate", &tier0_gate_expert_ids, &tier1_gate_expert_ids),
            ("up", &tier0_up_expert_ids, &tier1_up_expert_ids),
            ("down", &tier0_down_expert_ids, &tier1_down_expert_ids),
        ] {
            anyhow::ensure!(
                !tier0.is_empty()
                    && !tier1.is_empty()
                    && tier0.len() + tier1.len() == catalog.facts.routed_experts,
                "mixed EXL3 layer {layer_id} {name} is not a complete nonempty K2/K3 partition"
            );
        }
        let tier0_slots = tier0_gate_expert_ids.len().max(tier0_up_expert_ids.len());
        let tier1_slots = tier1_gate_expert_ids.len().max(tier1_up_expert_ids.len());
        let total_slots = tier0_slots
            .checked_add(tier1_slots)
            .context("mixed EXL3 projection slot count overflow")?;
        anyhow::ensure!(
            total_slots >= catalog.facts.routed_experts && tier0_slots <= 256 && tier1_slots <= 256,
            "mixed EXL3 layer {layer_id} has invalid projection slot geometry"
        );
        let allocate = |bytes: usize, label: &str| {
            OwnedDeviceAllocation::new(
                Arc::clone(&library),
                bytes,
                &format!("mixed EXL3 layer {layer_id} {label}"),
            )
            .map(Arc::new)
        };
        let projection_bytes = |experts: usize, bits: usize| {
            DS4_FLASH_HIDDEN_SIZE
                .checked_mul(DS4_FLASH_LOCAL_INTERMEDIATE_SIZE)
                .and_then(|values| values.checked_mul(bits))
                .and_then(|bits| bits.checked_div(8))
                .and_then(|bytes| bytes.checked_mul(experts))
                .context("mixed EXL3 tier projection size overflow")
        };
        let tier0_w13_bytes =
            projection_bytes(tier0_gate_expert_ids.len() + tier0_up_expert_ids.len(), 2)?;
        let tier0_w2_bytes = projection_bytes(tier0_down_expert_ids.len(), 2)?;
        let tier1_w13_bytes =
            projection_bytes(tier1_gate_expert_ids.len() + tier1_up_expert_ids.len(), 3)?;
        let tier1_w2_bytes = projection_bytes(tier1_down_expert_ids.len(), 3)?;
        let hidden_rotation_bytes =
            total_slots * DS4_FLASH_HIDDEN_SIZE * std::mem::size_of::<u16>();
        let intermediate_rotation_bytes =
            total_slots * 3 * DS4_FLASH_LOCAL_INTERMEDIATE_SIZE * std::mem::size_of::<u16>();
        let tier0_w13_trellis = allocate(tier0_w13_bytes, "K2 W13 trellis")?;
        let tier0_w2_trellis = allocate(tier0_w2_bytes, "K2 W2 trellis")?;
        let tier1_w13_trellis = allocate(tier1_w13_bytes, "K3 W13 trellis")?;
        let tier1_w2_trellis = allocate(tier1_w2_bytes, "K3 W2 trellis")?;
        let dummy_scale = allocate(16, "dummy scale")?;
        let tier0_scale_count = tier0_slots.max(tier0_down_expert_ids.len());
        let tier1_scale_count = tier1_slots.max(tier1_down_expert_ids.len());
        let tier0_global_scale = allocate(
            tier0_scale_count * std::mem::size_of::<f32>(),
            "K2 global scales",
        )?;
        let tier1_global_scale = allocate(
            tier1_scale_count * std::mem::size_of::<f32>(),
            "K3 global scales",
        )?;
        let gate_suh = allocate(hidden_rotation_bytes, "gate SUH rotations")?;
        let up_suh = allocate(hidden_rotation_bytes, "up SUH rotations")?;
        let intermediate_rotations =
            allocate(intermediate_rotation_bytes, "intermediate rotations")?;
        let down_svh = allocate(hidden_rotation_bytes, "down SVH rotations")?;
        let trellis_lut = allocate(SQG_XOR_CHEB_T12_LUT_BYTES, "SQG-XOR-Cheb T12 LUT")?;
        let global_to_combined = allocate(
            DS4_FLASH_ROUTED_EXPERTS * std::mem::size_of::<i32>(),
            "global-to-combined map",
        )?;
        let descriptor_map = allocate(
            3 * total_slots * std::mem::size_of::<i32>(),
            "tier descriptor map",
        )?;

        // Projection tiering cannot remap one route to one tier. Keep routes
        // in their 256-expert global namespace and let each descriptor row
        // select its own tier and tier-local packed plane.
        let global_map = (0..DS4_FLASH_ROUTED_EXPERTS as i32).collect::<Vec<_>>();
        let mut descriptors = vec![-1_i32; 3 * total_slots];
        for (row, (tier0, tier1)) in [
            (&tier0_gate_expert_ids, &tier1_gate_expert_ids),
            (&tier0_up_expert_ids, &tier1_up_expert_ids),
            (&tier0_down_expert_ids, &tier1_down_expert_ids),
        ]
        .into_iter()
        .enumerate()
        {
            for (local, &global) in tier0.iter().enumerate() {
                descriptors[row * total_slots + global] = local as i32;
            }
            for (local, &global) in tier1.iter().enumerate() {
                descriptors[row * total_slots + global] = ((1 << 8) | local) as i32;
            }
        }
        let encode_i32 = |values: &[i32]| {
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>()
        };
        let ones = |count: usize| {
            (0..count)
                .flat_map(|_| 1.0_f32.to_le_bytes())
                .collect::<Vec<_>>()
        };
        library.copy_h2d(dummy_scale.buffer(), &[0_u8; 16])?;
        library.copy_h2d(trellis_lut.buffer(), sqg_xor_cheb_t12_lut_bytes())?;
        library.copy_h2d(tier0_global_scale.buffer(), &ones(tier0_scale_count))?;
        library.copy_h2d(tier1_global_scale.buffer(), &ones(tier1_scale_count))?;
        library.copy_h2d(global_to_combined.buffer(), &encode_i32(&global_map))?;
        library.copy_h2d(descriptor_map.buffer(), &encode_i32(&descriptors))?;
        Ok(Self {
            layer_id,
            tier0_gate_expert_ids,
            tier1_gate_expert_ids,
            tier0_up_expert_ids,
            tier1_up_expert_ids,
            tier0_down_expert_ids,
            tier1_down_expert_ids,
            tier0_slots,
            tier1_slots,
            tier0_w13_trellis,
            tier0_w2_trellis,
            tier1_w13_trellis,
            tier1_w2_trellis,
            dummy_scale,
            tier0_global_scale,
            tier1_global_scale,
            gate_suh,
            up_suh,
            intermediate_rotations,
            down_svh,
            trellis_lut,
            global_to_combined,
            descriptor_map,
        })
    }

    fn store_layer_experts_direct(
        &self,
        catalog: &TensorCatalog,
        shard: ExpertIntermediateShard,
        library: Arc<NativeLibrary>,
        workspace: &mut RouteCudaWorkspace,
        cuda_stream: *mut c_void,
    ) -> Result<u64> {
        let trellis_bytes =
            |bits: usize| DS4_FLASH_HIDDEN_SIZE * DS4_FLASH_LOCAL_INTERMEDIATE_SIZE * bits / 8;
        let max_trellis = trellis_bytes(3);
        let hidden_rotation = DS4_FLASH_HIDDEN_SIZE * std::mem::size_of::<u16>();
        let local_rotation = DS4_FLASH_LOCAL_INTERMEDIATE_SIZE * std::mem::size_of::<u16>();
        let offsets = [
            0,
            max_trellis,
            2 * max_trellis,
            3 * max_trellis,
            3 * max_trellis + hidden_rotation,
            3 * max_trellis + 2 * hidden_rotation,
            3 * max_trellis + 2 * hidden_rotation + local_rotation,
            3 * max_trellis + 2 * hidden_rotation + 2 * local_rotation,
            3 * max_trellis + 2 * hidden_rotation + 3 * local_rotation,
        ];
        let expert_bytes = offsets[8] + hidden_rotation;
        let chunk_experts = ((32 * 1024 * 1024) / expert_bytes).clamp(1, 16);
        RouteCudaWorkspace::ensure_pinned_buffer(
            &mut workspace.pinned_projection_weight,
            Arc::clone(&library),
            chunk_experts * expert_bytes,
            "mixed EXL3 direct resident preload",
        )?;
        let snapshot = Path::new(&catalog.snapshot_path);
        let mut files = HashMap::<PathBuf, Arc<File>>::new();
        let mut source_bytes = 0_u64;
        let expert_ids = (0..catalog.facts.routed_experts).collect::<Vec<_>>();
        for expert_chunk in expert_ids.chunks(chunk_experts) {
            let staging_buffer;
            {
                let staging = workspace
                    .pinned_projection_weight
                    .as_mut()
                    .expect("mixed EXL3 pinned preload exists");
                let dst = staging.as_mut_slice(expert_chunk.len() * expert_bytes)?;
                let mut requests = Vec::with_capacity(expert_chunk.len() * 520);
                for (chunk_index, &expert_id) in expert_chunk.iter().enumerate() {
                    let expert = exl3_expert(catalog, self.layer_id, expert_id)?;
                    let base = chunk_index * expert_bytes;
                    for (projection, offset) in [
                        (expert.gate, offsets[0]),
                        (expert.up, offsets[1]),
                        (expert.down, offsets[2]),
                    ] {
                        let bits = exl3_projection_trellis_bits(projection)?;
                        let trellis = trellis_bytes(bits);
                        queue_exl3_trellis_tp4(
                            snapshot,
                            projection,
                            shard,
                            DS4_FLASH_HIDDEN_SIZE,
                            DS4_FLASH_LOCAL_INTERMEDIATE_SIZE,
                            bits,
                            trellis,
                            &mut files,
                            &mut dst[base + offset..base + offset + trellis],
                            &mut requests,
                        )?;
                    }
                    for (projection, source_start, offset, bytes) in [
                        (expert.gate, 0, offsets[3], hidden_rotation),
                        (expert.up, 0, offsets[4], hidden_rotation),
                        (
                            expert.gate,
                            shard.rank * local_rotation,
                            offsets[5],
                            local_rotation,
                        ),
                        (
                            expert.up,
                            shard.rank * local_rotation,
                            offsets[6],
                            local_rotation,
                        ),
                        (
                            expert.down,
                            shard.rank * local_rotation,
                            offsets[7],
                            local_rotation,
                        ),
                        (expert.down, 0, offsets[8], hidden_rotation),
                    ] {
                        let tensor = if offset == offsets[3] || offset == offsets[4] {
                            projection.suh
                        } else if offset == offsets[7] {
                            projection.suh
                        } else {
                            projection.svh
                        };
                        queue_exl3_tensor_window(
                            snapshot,
                            tensor,
                            source_start,
                            &mut files,
                            &mut dst[base + offset..base + offset + bytes],
                            &mut requests,
                        )?;
                    }
                    for projection in [expert.gate, expert.up, expert.down] {
                        validate_exl3_mcg_marker(snapshot, projection, &mut files)?;
                    }
                    source_bytes += [expert.gate, expert.up, expert.down]
                        .into_iter()
                        .map(exl3_projection_trellis_bits)
                        .collect::<Result<Vec<_>>>()?
                        .into_iter()
                        .map(trellis_bytes)
                        .sum::<usize>() as u64
                        + (2 * hidden_rotation + 3 * local_rotation + hidden_rotation) as u64
                        + 3 * std::mem::size_of::<u32>() as u64;
                }
                execute_exl3_read_requests(&requests)?;
                staging_buffer = staging.buffer();
            }
            for (chunk_index, &expert_id) in expert_chunk.iter().enumerate() {
                let expert = exl3_expert(catalog, self.layer_id, expert_id)?;
                let host_base = chunk_index * expert_bytes;
                let intermediate_base = expert_id * 3 * local_rotation;
                let mut copies = Vec::with_capacity(9);
                for (projection_index, (projection, source_offset)) in [
                    (expert.gate, offsets[0]),
                    (expert.up, offsets[1]),
                    (expert.down, offsets[2]),
                ]
                .into_iter()
                .enumerate()
                {
                    let bits = exl3_projection_trellis_bits(projection)?;
                    let trellis = trellis_bytes(bits);
                    let (members, buffer, plane_base, label) = match (bits, projection_index) {
                        (2, 0) => (
                            &self.tier0_gate_expert_ids,
                            self.tier0_w13_trellis.buffer(),
                            0,
                            "mixed K2 W13 gate",
                        ),
                        (3, 0) => (
                            &self.tier1_gate_expert_ids,
                            self.tier1_w13_trellis.buffer(),
                            0,
                            "mixed K3 W13 gate",
                        ),
                        (2, 1) => (
                            &self.tier0_up_expert_ids,
                            self.tier0_w13_trellis.buffer(),
                            self.tier0_gate_expert_ids.len(),
                            "mixed K2 W13 up",
                        ),
                        (3, 1) => (
                            &self.tier1_up_expert_ids,
                            self.tier1_w13_trellis.buffer(),
                            self.tier1_gate_expert_ids.len(),
                            "mixed K3 W13 up",
                        ),
                        (2, 2) => (
                            &self.tier0_down_expert_ids,
                            self.tier0_w2_trellis.buffer(),
                            0,
                            "mixed K2 W2 down",
                        ),
                        (3, 2) => (
                            &self.tier1_down_expert_ids,
                            self.tier1_w2_trellis.buffer(),
                            0,
                            "mixed K3 W2 down",
                        ),
                        _ => anyhow::bail!("unsupported mixed EXL3 projection tier K{bits}"),
                    };
                    let local_id = members.binary_search(&expert_id).map_err(|_| {
                        anyhow::anyhow!(
                            "mixed EXL3 layer {} projection membership changed for expert {expert_id}",
                            self.layer_id
                        )
                    })?;
                    copies.push((
                        source_offset,
                        trellis,
                        device_buffer_byte_view(
                            buffer,
                            (plane_base + local_id) * trellis,
                            trellis,
                            label,
                        )?,
                    ));
                }
                copies.extend([
                    (
                        offsets[3],
                        hidden_rotation,
                        device_buffer_byte_view(
                            self.gate_suh.buffer(),
                            expert_id * hidden_rotation,
                            hidden_rotation,
                            "mixed gate SUH",
                        )?,
                    ),
                    (
                        offsets[4],
                        hidden_rotation,
                        device_buffer_byte_view(
                            self.up_suh.buffer(),
                            expert_id * hidden_rotation,
                            hidden_rotation,
                            "mixed up SUH",
                        )?,
                    ),
                    (
                        offsets[5],
                        local_rotation,
                        device_buffer_byte_view(
                            self.intermediate_rotations.buffer(),
                            intermediate_base,
                            local_rotation,
                            "mixed gate SVH",
                        )?,
                    ),
                    (
                        offsets[6],
                        local_rotation,
                        device_buffer_byte_view(
                            self.intermediate_rotations.buffer(),
                            intermediate_base + local_rotation,
                            local_rotation,
                            "mixed up SVH",
                        )?,
                    ),
                    (
                        offsets[7],
                        local_rotation,
                        device_buffer_byte_view(
                            self.intermediate_rotations.buffer(),
                            intermediate_base + 2 * local_rotation,
                            local_rotation,
                            "mixed down SUH",
                        )?,
                    ),
                    (
                        offsets[8],
                        hidden_rotation,
                        device_buffer_byte_view(
                            self.down_svh.buffer(),
                            expert_id * hidden_rotation,
                            hidden_rotation,
                            "mixed down SVH",
                        )?,
                    ),
                ]);
                for (source_offset, bytes, destination) in copies {
                    let source = host_buffer_byte_view(
                        staging_buffer,
                        host_base + source_offset,
                        bytes,
                        "mixed EXL3 compact pinned source",
                    )?;
                    unsafe {
                        library.copy_host_buffer_h2d_async(
                            destination,
                            source,
                            bytes,
                            cuda_stream,
                        )?;
                    }
                }
            }
            unsafe {
                library.cuda_stream_synchronize(cuda_stream)?;
            }
        }
        Ok(source_bytes)
    }

    fn resident_weight_bytes(&self) -> u64 {
        (self.tier0_w13_trellis.capacity_bytes()
            + self.tier0_w2_trellis.capacity_bytes()
            + self.tier1_w13_trellis.capacity_bytes()
            + self.tier1_w2_trellis.capacity_bytes()) as u64
    }

    fn resident_scale_bytes(&self) -> u64 {
        (self.dummy_scale.capacity_bytes()
            + self.tier0_global_scale.capacity_bytes()
            + self.tier1_global_scale.capacity_bytes()
            + self.gate_suh.capacity_bytes()
            + self.up_suh.capacity_bytes()
            + self.intermediate_rotations.capacity_bytes()
            + self.down_svh.capacity_bytes()
            + self.trellis_lut.capacity_bytes()
            + self.global_to_combined.capacity_bytes()
            + self.descriptor_map.capacity_bytes()) as u64
    }

    fn launch_counts(&self) -> [usize; 8] {
        [
            self.tier0_slots,
            self.tier1_slots,
            self.tier0_gate_expert_ids.len(),
            self.tier1_gate_expert_ids.len(),
            self.tier0_up_expert_ids.len(),
            self.tier1_up_expert_ids.len(),
            self.tier0_down_expert_ids.len(),
            self.tier1_down_expert_ids.len(),
        ]
    }

    unsafe fn launch_mixed(
        &self,
        library: &NativeLibrary,
        buffers: &Ds41rtDs4FlashSparkExl3MixedMoeBuffers,
        rows: usize,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let [tier0_slots, tier1_slots, tier0_gate, tier1_gate, tier0_up, tier1_up, tier0_down, tier1_down] =
            self.launch_counts();
        unsafe {
            library.cuda_ds4_flash_spark_exl3_mixed_k2_k3_async(
                buffers,
                tier0_slots,
                tier1_slots,
                tier0_gate,
                tier1_gate,
                tier0_up,
                tier1_up,
                tier0_down,
                tier1_down,
                rows,
                cuda_stream,
            )
        }
    }

    fn aot_buffers(
        &self,
        workspace: Ds4FlashSparkAotRouteWorkspaceBuffers,
        input: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        topk_ids: Ds41rtDeviceBuffer,
        topk_weights: Ds41rtDeviceBuffer,
        packed_route_indices: Ds41rtDeviceBuffer,
    ) -> Ds41rtDs4FlashSparkExl3MixedMoeBuffers {
        Ds41rtDs4FlashSparkExl3MixedMoeBuffers {
            input,
            tier0_w13_trellis: self.tier0_w13_trellis.buffer(),
            tier0_w2_trellis: self.tier0_w2_trellis.buffer(),
            tier1_w13_trellis: self.tier1_w13_trellis.buffer(),
            tier1_w2_trellis: self.tier1_w2_trellis.buffer(),
            dummy_scale: self.dummy_scale.buffer(),
            tier0_global_scale: self.tier0_global_scale.buffer(),
            tier1_global_scale: self.tier1_global_scale.buffer(),
            gate_suh: self.gate_suh.buffer(),
            up_suh: self.up_suh.buffer(),
            intermediate_rotations: self.intermediate_rotations.buffer(),
            down_svh: self.down_svh.buffer(),
            trellis_lut: self.trellis_lut.buffer(),
            global_to_combined: self.global_to_combined.buffer(),
            descriptor_map: self.descriptor_map.buffer(),
            topk_ids,
            mapped_topk_ids: workspace.fc1_scratch,
            topk_weights,
            rotation_gate: workspace.rotation_gate,
            rotation_up: workspace.rotation_up,
            fc1_output: workspace.fc1_output,
            activated: workspace.activated,
            routed_output: workspace.routed_output,
            output_f32: workspace.output_f32,
            output_bf16: output,
            packed_route_indices,
            block_expert_ids: workspace.block_expert_ids,
            packed_route_count: workspace.packed_route_count,
            expert_counts: workspace.expert_counts,
            expert_offsets: workspace.expert_offsets,
            fc1_scratch: workspace.fc1_scratch,
            fc2_scratch: workspace.fc2_scratch,
            workspace: workspace.locks,
        }
    }

    fn decode_m1_aot_buffers(
        &self,
        buffers: Ds4FlashSparkDecodeM1DeviceBuffers,
    ) -> Ds41rtDs4FlashSparkExl3MixedMoeBuffers {
        self.aot_buffers(
            Ds4FlashSparkAotRouteWorkspaceBuffers {
                packed_hidden: Ds41rtDeviceBuffer::default(),
                fc1_output: buffers.fc1,
                activated: buffers.activated,
                routed_output: buffers.routed_output,
                rotation_gate: buffers.rotation_gate,
                rotation_up: buffers.rotation_up,
                output_f32: buffers.output_f32,
                // The uniform direct M1 path reserves one descriptor. Mixed
                // route packing needs up to six, so reuse the activation arena
                // until the one-grid kernel reaches its activation phase.
                block_expert_ids: buffers.activated,
                packed_route_count: buffers.packed_route_count,
                expert_counts: buffers.expert_counts,
                expert_offsets: buffers.expert_offsets,
                fc1_scratch: buffers.fc1_scratch,
                fc2_scratch: buffers.fc2_scratch,
                locks: buffers.locks,
            },
            buffers.input,
            buffers.output,
            buffers.route_indices,
            buffers.route_weights,
            // Route-pack output is dead before the mixed grid writes FC2.
            buffers.routed_output,
        )
    }

    fn prefill_aot_buffers(
        &self,
        buffers: Ds4FlashSparkPrefillDeviceBuffers,
    ) -> Ds41rtDs4FlashSparkExl3MixedMoeBuffers {
        self.aot_buffers(
            Ds4FlashSparkAotRouteWorkspaceBuffers {
                packed_hidden: Ds41rtDeviceBuffer::default(),
                fc1_output: buffers.fc1,
                activated: buffers.activated,
                routed_output: buffers.routed_output,
                rotation_gate: buffers.rotation_gate,
                rotation_up: buffers.rotation_up,
                output_f32: buffers.output_f32,
                block_expert_ids: buffers.block_expert_ids,
                packed_route_count: buffers.packed_route_count,
                expert_counts: buffers.expert_counts,
                expert_offsets: buffers.expert_offsets,
                fc1_scratch: buffers.fc1_scratch,
                fc2_scratch: buffers.fc2_scratch,
                locks: buffers.locks,
            },
            buffers.input,
            buffers.output,
            buffers.topk_ids,
            buffers.topk_weights,
            buffers.packed_route_indices,
        )
    }
}

impl RouteCudaBf16LayerExpertSlab {
    fn new(
        library: Arc<NativeLibrary>,
        layer_id: usize,
        expert_count: usize,
        exemplar: &LoadedRouteCudaBf16ExpertShard,
    ) -> Result<Self> {
        anyhow::ensure!(expert_count > 0, "BF16 expert slab must not be empty");
        for (name, tensor) in [
            ("gate_proj", &exemplar.gate),
            ("up_proj", &exemplar.up),
            ("down_proj", &exemplar.down),
        ] {
            anyhow::ensure!(
                tensor.info.dtype == DType::Bf16 && tensor.bytes_per_scalar == 2,
                "layer {layer_id} retained BF16 {name} has dtype {:?} and scalar width {}, expected BF16",
                tensor.info.dtype,
                tensor.bytes_per_scalar,
            );
        }
        anyhow::ensure!(
            exemplar.gate.row_count == exemplar.up.row_count
                && exemplar.gate.row_width == exemplar.up.row_width,
            "layer {layer_id} retained BF16 gate/up geometry differs"
        );
        let intermediate_rows = exemplar.gate.row_count;
        let hidden_dim = exemplar.gate.row_width;
        let output_rows = exemplar.down.row_count;
        anyhow::ensure!(
            exemplar.down.row_width == intermediate_rows,
            "layer {layer_id} retained BF16 down width {} differs from local intermediate rows {intermediate_rows}",
            exemplar.down.row_width
        );
        let gate_bytes = exemplar.gate.bytes.len();
        let up_bytes = exemplar.up.bytes.len();
        let w13_expert_stride_bytes = gate_bytes
            .checked_add(up_bytes)
            .context("BF16 W13 expert stride overflow")?;
        let w2_expert_stride_bytes = exemplar.down.bytes.len();
        let w13_bytes = w13_expert_stride_bytes
            .checked_mul(expert_count)
            .context("BF16 W13 slab byte count overflow")?;
        let w2_bytes = w2_expert_stride_bytes
            .checked_mul(expert_count)
            .context("BF16 W2 slab byte count overflow")?;
        Ok(Self {
            layer_id,
            expert_count,
            hidden_dim,
            intermediate_rows,
            output_rows,
            w13_weight: Arc::new(OwnedDeviceAllocation::new_with_kind(
                Arc::clone(&library),
                w13_bytes,
                &format!("layer {layer_id} retained BF16 W13 expert slab"),
                false,
            )?),
            w13_expert_stride_bytes,
            w2_weight: Arc::new(OwnedDeviceAllocation::new_with_kind(
                library,
                w2_bytes,
                &format!("layer {layer_id} retained BF16 W2 expert slab"),
                false,
            )?),
            w2_expert_stride_bytes,
        })
    }

    fn store_expert(
        &self,
        expert_id: usize,
        loaded: &LoadedRouteCudaBf16ExpertShard,
        library: Arc<NativeLibrary>,
        workspace: &mut RouteCudaWorkspace,
        cuda_stream: *mut c_void,
    ) -> Result<u64> {
        anyhow::ensure!(
            expert_id < self.expert_count
                && loaded.gate.row_count == self.intermediate_rows
                && loaded.up.row_count == self.intermediate_rows
                && loaded.gate.row_width == self.hidden_dim
                && loaded.up.row_width == self.hidden_dim
                && loaded.down.row_count == self.output_rows
                && loaded.down.row_width == self.intermediate_rows,
            "layer {} expert {expert_id} retained BF16 geometry differs from its slab",
            self.layer_id
        );
        let w13_offset = expert_id
            .checked_mul(self.w13_expert_stride_bytes)
            .context("BF16 W13 expert offset overflow")?;
        let gate_target = device_buffer_byte_view(
            self.w13_weight.buffer(),
            w13_offset,
            loaded.gate.bytes.len(),
            "retained BF16 gate target",
        )?;
        let up_target = device_buffer_byte_view(
            self.w13_weight.buffer(),
            w13_offset
                .checked_add(loaded.gate.bytes.len())
                .context("BF16 up expert offset overflow")?,
            loaded.up.bytes.len(),
            "retained BF16 up target",
        )?;
        let w2_target = device_buffer_byte_view(
            self.w2_weight.buffer(),
            expert_id
                .checked_mul(self.w2_expert_stride_bytes)
                .context("BF16 W2 expert offset overflow")?,
            loaded.down.bytes.len(),
            "retained BF16 down target",
        )?;
        workspace.upload_host_bytes_to_existing_device_buffer(
            Arc::clone(&library),
            gate_target,
            &loaded.gate.bytes,
            "retained BF16 gate weight",
            RouteCudaProjectionStageSlot::Weight,
            cuda_stream,
        )?;
        workspace.upload_host_bytes_to_existing_device_buffer(
            Arc::clone(&library),
            up_target,
            &loaded.up.bytes,
            "retained BF16 up weight",
            RouteCudaProjectionStageSlot::Weight,
            cuda_stream,
        )?;
        workspace.upload_host_bytes_to_existing_device_buffer(
            library,
            w2_target,
            &loaded.down.bytes,
            "retained BF16 down weight",
            RouteCudaProjectionStageSlot::Weight,
            cuda_stream,
        )?;
        Ok((loaded.gate.bytes.len() + loaded.up.bytes.len() + loaded.down.bytes.len()) as u64)
    }

    fn expert_buffers(&self, expert_id: usize) -> Result<RouteCudaBf16ExpertBuffers> {
        anyhow::ensure!(
            expert_id < self.expert_count,
            "retained BF16 expert {expert_id} exceeds layer {} expert count {}",
            self.layer_id,
            self.expert_count
        );
        Ok(RouteCudaBf16ExpertBuffers {
            w13_weight: device_buffer_byte_view(
                self.w13_weight.buffer(),
                expert_id * self.w13_expert_stride_bytes,
                self.w13_expert_stride_bytes,
                "retained BF16 W13 expert",
            )?,
            w2_weight: device_buffer_byte_view(
                self.w2_weight.buffer(),
                expert_id * self.w2_expert_stride_bytes,
                self.w2_expert_stride_bytes,
                "retained BF16 W2 expert",
            )?,
        })
    }
}

struct RouteCudaLayerExpertSlab {
    layer_id: usize,
    expert_count: usize,
    w13_weight: Arc<OwnedDeviceAllocation>,
    w13_weight_expert_stride_bytes: usize,
    w13_scale: Arc<OwnedDeviceAllocation>,
    w13_scale_expert_stride_bytes: usize,
    w2_weight: Arc<OwnedDeviceAllocation>,
    w2_weight_expert_stride_bytes: usize,
    w2_scale: Arc<OwnedDeviceAllocation>,
    w2_scale_expert_stride_bytes: usize,
    w1_alphas: Arc<OwnedDeviceAllocation>,
    w2_alphas: Arc<OwnedDeviceAllocation>,
    w4a16_w13_global_scale: Option<Arc<OwnedDeviceAllocation>>,
    w4a16_w2_global_scale: Option<Arc<OwnedDeviceAllocation>>,
    gate_rows: usize,
    up_rows: usize,
    down_rows: usize,
    gate_weight_row_stride_bytes: usize,
    up_weight_row_stride_bytes: usize,
    down_weight_row_stride_bytes: usize,
}

impl RouteCudaLayerExpertSlab {
    fn new(
        library: Arc<NativeLibrary>,
        layer_id: usize,
        expert_count: usize,
        exemplar: &LoadedRouteCudaExpertShard,
        managed_weights: bool,
        geometry_only_exemplar: bool,
    ) -> Result<Self> {
        anyhow::ensure!(expert_count > 0, "expert slab requires at least one expert");
        let geometry = |projection| {
            loaded_route_cuda_projection_geometry_inner(projection, !geometry_only_exemplar)
        };
        let gate = geometry(&exemplar.gate)?;
        let up = geometry(&exemplar.up)?;
        let down = geometry(&exemplar.down)?;
        anyhow::ensure!(
            gate.rows == up.rows,
            "layer {layer_id} gate/up shard rows differ: {} vs {}",
            gate.rows,
            up.rows
        );
        for (projection, geometry) in [("gate", gate), ("up", up), ("down", down)] {
            anyhow::ensure!(
                b12x_projection_scale_shape_supported(
                    geometry.rows,
                    geometry.scale_row_stride_bytes
                ),
                "layer {layer_id} {projection} scale shape rows={} cols={} is not B12X-compatible",
                geometry.rows,
                geometry.scale_row_stride_bytes
            );
        }
        let w13_weight_expert_stride_bytes = gate
            .weight_bytes
            .checked_add(up.weight_bytes)
            .context("W13 expert weight stride overflow")?;
        let w13_scale_expert_stride_bytes = gate
            .scale_bytes
            .checked_add(up.scale_bytes)
            .context("W13 expert scale stride overflow")?;
        let w13_weight_bytes = w13_weight_expert_stride_bytes
            .checked_mul(expert_count)
            .context("W13 slab weight bytes overflow")?;
        let w13_scale_bytes = w13_scale_expert_stride_bytes
            .checked_mul(expert_count)
            .context("W13 slab scale bytes overflow")?;
        let w2_weight_bytes = down
            .weight_bytes
            .checked_mul(expert_count)
            .context("W2 slab weight bytes overflow")?;
        let w2_scale_bytes = down
            .scale_bytes
            .checked_mul(expert_count)
            .context("W2 slab scale bytes overflow")?;
        let scalar_bytes = expert_count * std::mem::size_of::<f32>();
        let dedicated_w4a16_global_scales = !managed_weights;
        let w4a16_w13_global_scale = if dedicated_w4a16_global_scales {
            Some(Arc::new(OwnedDeviceAllocation::new_with_kind(
                Arc::clone(&library),
                scalar_bytes,
                &format!("layer {layer_id} W4A16 W13 global scales"),
                false,
            )?))
        } else {
            None
        };
        let w4a16_w2_global_scale = if dedicated_w4a16_global_scales {
            Some(Arc::new(OwnedDeviceAllocation::new_with_kind(
                Arc::clone(&library),
                scalar_bytes,
                &format!("layer {layer_id} W4A16 W2 global scales"),
                false,
            )?))
        } else {
            None
        };
        Ok(Self {
            layer_id,
            expert_count,
            w13_weight: Arc::new(OwnedDeviceAllocation::new_with_kind(
                Arc::clone(&library),
                w13_weight_bytes,
                &format!("layer {layer_id} TP4 W13 expert weight slab"),
                managed_weights,
            )?),
            w13_weight_expert_stride_bytes,
            w13_scale: Arc::new(OwnedDeviceAllocation::new_with_kind(
                Arc::clone(&library),
                w13_scale_bytes,
                &format!("layer {layer_id} TP4 W13 expert scale slab"),
                managed_weights,
            )?),
            w13_scale_expert_stride_bytes,
            w2_weight: Arc::new(OwnedDeviceAllocation::new_with_kind(
                Arc::clone(&library),
                w2_weight_bytes,
                &format!("layer {layer_id} TP4 W2 expert weight slab"),
                managed_weights,
            )?),
            w2_weight_expert_stride_bytes: down.weight_bytes,
            w2_scale: Arc::new(OwnedDeviceAllocation::new_with_kind(
                Arc::clone(&library),
                w2_scale_bytes,
                &format!("layer {layer_id} TP4 W2 expert scale slab"),
                managed_weights,
            )?),
            w2_scale_expert_stride_bytes: down.scale_bytes,
            w1_alphas: Arc::new(OwnedDeviceAllocation::new_with_kind(
                Arc::clone(&library),
                scalar_bytes,
                &format!("layer {layer_id} TP4 W1 expert alphas"),
                true,
            )?),
            w2_alphas: Arc::new(OwnedDeviceAllocation::new_with_kind(
                library,
                scalar_bytes,
                &format!("layer {layer_id} TP4 W2 expert alphas"),
                true,
            )?),
            w4a16_w13_global_scale,
            w4a16_w2_global_scale,
            gate_rows: gate.rows,
            up_rows: up.rows,
            down_rows: down.rows,
            gate_weight_row_stride_bytes: gate.weight_row_stride_bytes,
            up_weight_row_stride_bytes: up.weight_row_stride_bytes,
            down_weight_row_stride_bytes: down.weight_row_stride_bytes,
        })
    }

    fn managed_projection_entries(&self) -> usize {
        usize::from(self.w13_weight.is_managed()) * self.expert_count * 3
    }

    fn store_layer_experts_w4a16(
        &self,
        expert_ids: &[usize],
        loaded_experts: &[LoadedRouteCudaExpertShard],
        library: Arc<NativeLibrary>,
        workspace: &mut RouteCudaWorkspace,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        anyhow::ensure!(
            !expert_ids.is_empty() && expert_ids.len() == loaded_experts.len(),
            "layer {} batched W4A16 preload received {} expert ids and {} payloads",
            self.layer_id,
            expert_ids.len(),
            loaded_experts.len()
        );

        #[derive(Clone, Copy)]
        struct PackPlan {
            source_offset: usize,
            source_bytes: usize,
            destination: Ds41rtDeviceBuffer,
            size_k: usize,
            source_size_k: usize,
            source_start_k: usize,
            size_n: usize,
            row_rotation: usize,
            scale_factor: Option<f32>,
        }

        let source_bytes = loaded_experts.iter().try_fold(0_usize, |total, loaded| {
            [
                &loaded.up.weight,
                &loaded.gate.weight,
                &loaded.up.weight_scale,
                &loaded.gate.weight_scale,
                &loaded.down.weight,
                &loaded.down.weight_scale,
            ]
            .into_iter()
            .try_fold(total, |subtotal, tensor| {
                subtotal
                    .checked_add(loaded_route_cuda_tensor_logical_bytes(tensor)?)
                    .context("layer W4A16 preload source byte count overflow")
            })
        })?;
        RouteCudaWorkspace::ensure_pinned_buffer(
            &mut workspace.pinned_projection_weight,
            Arc::clone(&library),
            source_bytes,
            "batched layer W4A16 source",
        )?;
        RouteCudaWorkspace::ensure_buffer(
            &mut workspace.b12x_w4a16_pack_source,
            Arc::clone(&library),
            source_bytes,
            "batched layer W4A16 device source",
        )?;

        let mut plans = Vec::with_capacity(expert_ids.len() * 4);
        let staging_buffer;
        {
            let staging = workspace
                .pinned_projection_weight
                .as_mut()
                .expect("batched layer W4A16 pinned source ensured");
            let staging_slice = staging.as_mut_slice(source_bytes)?;
            let mut source_offset = 0_usize;
            for (&expert_id, loaded) in expert_ids.iter().zip(loaded_experts) {
                anyhow::ensure!(
                    expert_id < self.expert_count,
                    "layer {} batched W4A16 expert {expert_id} exceeds {}",
                    self.layer_id,
                    self.expert_count
                );
                let gate = loaded_route_cuda_projection_geometry(&loaded.gate)?;
                let up = loaded_route_cuda_projection_geometry(&loaded.up)?;
                let down = loaded_route_cuda_projection_geometry(&loaded.down)?;
                anyhow::ensure!(
                    gate.rows == up.rows
                        && gate.weight_row_stride_bytes == up.weight_row_stride_bytes
                        && gate.scale_row_stride_bytes == up.scale_row_stride_bytes,
                    "layer {} expert {expert_id} batched W13 geometry differs",
                    self.layer_id
                );
                let hidden_dim = gate
                    .weight_row_stride_bytes
                    .checked_mul(2)
                    .context("batched W13 hidden dimension overflow")?;
                let intermediate_dim = down
                    .weight_row_stride_bytes
                    .checked_mul(2)
                    .context("batched W2 intermediate dimension overflow")?;
                let down_weight_source_k = loaded
                    .down
                    .weight
                    .source_row_width
                    .checked_mul(2)
                    .context("batched W2 source weight K overflow")?;
                let down_weight_source_start_k = loaded
                    .down
                    .weight
                    .source_column_start
                    .checked_mul(2)
                    .context("batched W2 source weight K offset overflow")?;
                let down_scale_source_k = loaded
                    .down
                    .weight_scale
                    .source_row_width
                    .checked_mul(16)
                    .context("batched W2 source scale K overflow")?;
                let down_scale_source_start_k = loaded
                    .down
                    .weight_scale
                    .source_column_start
                    .checked_mul(16)
                    .context("batched W2 source scale K offset overflow")?;
                anyhow::ensure!(
                    down_weight_source_k == down_scale_source_k
                        && down_weight_source_start_k == down_scale_source_start_k
                        && down_weight_source_start_k + intermediate_dim <= down_weight_source_k,
                    "layer {} expert {expert_id} batched W2 source windows differ or overflow",
                    self.layer_id
                );
                let w13_rows = gate
                    .rows
                    .checked_add(up.rows)
                    .context("batched W13 row count overflow")?;
                let w13_weight_parts = [&loaded.up.weight, &loaded.gate.weight];
                let w13_scale_parts = [&loaded.up.weight_scale, &loaded.gate.weight_scale];
                let w2_weight_parts = [&loaded.down.weight];
                let w2_scale_parts = [&loaded.down.weight_scale];
                let destinations = [
                    (
                        route_device_buffer_slice(
                            self.w13_weight.buffer(),
                            expert_id * self.w13_weight_expert_stride_bytes,
                            self.w13_weight_expert_stride_bytes,
                        )?,
                        hidden_dim,
                        w13_rows,
                        up.rows,
                        None,
                        w13_weight_parts.as_slice(),
                    ),
                    (
                        route_device_buffer_slice(
                            self.w13_scale.buffer(),
                            expert_id * self.w13_scale_expert_stride_bytes,
                            self.w13_scale_expert_stride_bytes,
                        )?,
                        hidden_dim,
                        w13_rows,
                        up.rows,
                        Some(1.0),
                        w13_scale_parts.as_slice(),
                    ),
                    (
                        route_device_buffer_slice(
                            self.w2_weight.buffer(),
                            expert_id * self.w2_weight_expert_stride_bytes,
                            self.w2_weight_expert_stride_bytes,
                        )?,
                        intermediate_dim,
                        down.rows,
                        0,
                        None,
                        w2_weight_parts.as_slice(),
                    ),
                    (
                        route_device_buffer_slice(
                            self.w2_scale.buffer(),
                            expert_id * self.w2_scale_expert_stride_bytes,
                            self.w2_scale_expert_stride_bytes,
                        )?,
                        intermediate_dim,
                        down.rows,
                        0,
                        Some(1.0),
                        w2_scale_parts.as_slice(),
                    ),
                ];
                for (destination, size_k, size_n, row_rotation, scale_factor, parts) in destinations
                {
                    let plan_offset = source_offset;
                    for part in parts {
                        let part_bytes = loaded_route_cuda_tensor_logical_bytes(part)?;
                        let end = source_offset
                            .checked_add(part_bytes)
                            .context("batched W4A16 staging offset overflow")?;
                        copy_loaded_route_cuda_tensor_compact(
                            part,
                            &mut staging_slice[source_offset..end],
                        )?;
                        source_offset = end;
                    }
                    plans.push(PackPlan {
                        source_offset: plan_offset,
                        source_bytes: source_offset - plan_offset,
                        destination,
                        size_k,
                        source_size_k: size_k,
                        source_start_k: 0,
                        size_n,
                        row_rotation,
                        scale_factor,
                    });
                }
            }
            anyhow::ensure!(
                source_offset == source_bytes,
                "batched W4A16 staged {source_offset} bytes, expected {source_bytes}"
            );
            staging_buffer = staging.buffer();
        }

        let source = workspace
            .b12x_w4a16_pack_source
            .as_ref()
            .expect("batched layer W4A16 device source ensured")
            .buffer();
        unsafe {
            library
                .copy_host_buffer_h2d_async(source, staging_buffer, source_bytes, cuda_stream)
                .context("uploading batched layer W4A16 source")?;
            for plan in plans {
                let source_view =
                    route_device_buffer_slice(source, plan.source_offset, plan.source_bytes)?;
                if let Some(scale_factor) = plan.scale_factor {
                    library
                        .cuda_b12x_w4a16_pack_scale_strided_async(
                            source_view,
                            plan.destination,
                            plan.size_k,
                            plan.source_size_k,
                            plan.source_start_k,
                            plan.size_n,
                            plan.row_rotation,
                            scale_factor,
                            cuda_stream,
                        )
                        .context("packing batched layer W4A16 scales")?;
                } else {
                    library
                        .cuda_b12x_w4a16_pack_weight_strided_async(
                            source_view,
                            plan.destination,
                            plan.size_k,
                            plan.source_size_k,
                            plan.source_start_k,
                            plan.size_n,
                            plan.row_rotation,
                            cuda_stream,
                        )
                        .context("packing batched layer W4A16 weights")?;
                }
            }
            library
                .cuda_stream_synchronize(cuda_stream)
                .context("synchronizing batched layer W4A16 preload")?;
        }
        Ok(())
    }

    fn store_expert_scalars_from_cache(
        &self,
        expert_id: usize,
        scalar_metadata: &HashMap<RoutedQuantScalarMetadataKey, RoutedQuantScalarMetadata>,
    ) -> Result<()> {
        let metadata = |projection| {
            scalar_metadata
                .get(&RoutedQuantScalarMetadataKey {
                    layer_id: self.layer_id,
                    expert_id,
                    projection,
                })
                .with_context(|| {
                    format!(
                        "layer {} expert {expert_id} is missing cached {projection} scalar metadata",
                        self.layer_id
                    )
                })
        };
        self.store_expert_scalars_from_metadata(
            expert_id,
            metadata("gate_proj")?,
            metadata("up_proj")?,
            metadata("down_proj")?,
        )
    }

    fn store_expert_scalars_from_metadata(
        &self,
        expert_id: usize,
        gate: &RoutedQuantScalarMetadata,
        up: &RoutedQuantScalarMetadata,
        down: &RoutedQuantScalarMetadata,
    ) -> Result<()> {
        anyhow::ensure!(
            expert_id < self.expert_count,
            "expert {expert_id} exceeds layer {} scalar slab expert count {}",
            self.layer_id,
            self.expert_count
        );
        anyhow::ensure!(
            gate.input_scale.to_bits() == up.input_scale.to_bits()
                && gate.weight_scale_2.to_bits() == up.weight_scale_2.to_bits(),
            "layer {} expert {expert_id} gate/up scalar metadata differ",
            self.layer_id
        );
        anyhow::ensure!(
            gate.input_scale > 0.0 && down.input_scale > 0.0,
            "layer {} expert {expert_id} activation scales must be positive",
            self.layer_id
        );
        let packed_scale = 2.0_f32.powi(119);
        let w1_alpha = gate.weight_scale_2 * packed_scale;
        let w2_alpha = down.weight_scale_2 * packed_scale;
        for (allocation, value, label) in [
            (&self.w1_alphas, w1_alpha, "W1 alpha"),
            (&self.w2_alphas, w2_alpha, "W2 alpha"),
        ] {
            anyhow::ensure!(
                value.is_finite(),
                "layer {} expert {expert_id} {label} is not finite",
                self.layer_id
            );
            allocation.copy_host_bytes_direct_at(
                expert_id * std::mem::size_of::<f32>(),
                &value.to_le_bytes(),
                label,
            )?;
        }
        Ok(())
    }

    fn store_startup_quantized_expert_scalars(
        &self,
        expert_id: usize,
        w13_global_scale: f32,
        w2_global_scale: f32,
    ) -> Result<()> {
        anyhow::ensure!(
            expert_id < self.expert_count
                && w13_global_scale.is_finite()
                && w13_global_scale > 0.0
                && w2_global_scale.is_finite()
                && w2_global_scale > 0.0,
            "invalid startup-quantized scalar metadata for layer {} expert {expert_id}",
            self.layer_id
        );
        let packed_scale = 2.0_f32.powi(119);
        for (allocation, value, label) in [
            (
                &self.w1_alphas,
                w13_global_scale.recip() * packed_scale,
                "startup-quantized W1 alpha",
            ),
            (
                &self.w2_alphas,
                w2_global_scale.recip() * packed_scale,
                "startup-quantized W2 alpha",
            ),
        ] {
            allocation.copy_host_bytes_direct_at(
                expert_id * std::mem::size_of::<f32>(),
                &value.to_le_bytes(),
                label,
            )?;
        }
        Ok(())
    }

    fn store_expert_startup_quantized_bf16(
        &self,
        expert_id: usize,
        loaded: &LoadedRouteCudaBf16ExpertShard,
        library: Arc<NativeLibrary>,
        workspace: &mut RouteCudaWorkspace,
        cuda_stream: *mut c_void,
    ) -> Result<(u64, u64)> {
        anyhow::ensure!(
            expert_id < self.expert_count,
            "layer {} cannot startup-quantize BF16 expert {expert_id}",
            self.layer_id
        );
        anyhow::ensure!(
            loaded.gate.row_count == self.gate_rows
                && loaded.up.row_count == self.up_rows
                && loaded.down.row_count == self.down_rows
                && loaded.gate.row_width == self.gate_weight_row_stride_bytes * 2
                && loaded.up.row_width == self.up_weight_row_stride_bytes * 2
                && loaded.down.row_width == self.down_weight_row_stride_bytes * 2,
            "layer {} expert {expert_id} BF16 geometry differs from startup NVFP4 slab",
            self.layer_id
        );
        let w13_global_scale = nvfp4_global_scale_for_bf16(
            &[&loaded.up.bytes, &loaded.gate.bytes],
            "startup BF16 W13",
        )?;
        let w2_global_scale =
            nvfp4_global_scale_for_bf16(&[&loaded.down.bytes], "startup BF16 W2")?;
        let w13_weight = route_device_buffer_slice(
            self.w13_weight.buffer(),
            expert_id * self.w13_weight_expert_stride_bytes,
            self.w13_weight_expert_stride_bytes,
        )?;
        let w13_scale = route_device_buffer_slice(
            self.w13_scale.buffer(),
            expert_id * self.w13_scale_expert_stride_bytes,
            self.w13_scale_expert_stride_bytes,
        )?;
        let w2_weight = route_device_buffer_slice(
            self.w2_weight.buffer(),
            expert_id * self.w2_weight_expert_stride_bytes,
            self.w2_weight_expert_stride_bytes,
        )?;
        let w2_scale = route_device_buffer_slice(
            self.w2_scale.buffer(),
            expert_id * self.w2_scale_expert_stride_bytes,
            self.w2_scale_expert_stride_bytes,
        )?;
        workspace.quantize_bf16_projection_into_w4a16(
            Arc::clone(&library),
            &[&loaded.up.bytes, &loaded.gate.bytes],
            w13_weight,
            w13_scale,
            self.up_rows + self.gate_rows,
            loaded.up.row_width,
            self.up_rows,
            w13_global_scale,
            cuda_stream,
            "layer-78 BF16 W13",
        )?;
        workspace.quantize_bf16_projection_into_w4a16(
            library,
            &[&loaded.down.bytes],
            w2_weight,
            w2_scale,
            self.down_rows,
            loaded.down.row_width,
            0,
            w2_global_scale,
            cuda_stream,
            "layer-78 BF16 W2",
        )?;
        self.store_startup_quantized_expert_scalars(expert_id, w13_global_scale, w2_global_scale)?;
        Ok((
            (self.w13_weight_expert_stride_bytes + self.w2_weight_expert_stride_bytes) as u64,
            (self.w13_scale_expert_stride_bytes + self.w2_scale_expert_stride_bytes) as u64,
        ))
    }

    fn finalize_w4a16_global_scales(
        &self,
        library: &NativeLibrary,
        cuda_stream: *mut c_void,
    ) -> Result<()> {
        let (Some(w13_global_scale), Some(w2_global_scale)) = (
            self.w4a16_w13_global_scale.as_ref(),
            self.w4a16_w2_global_scale.as_ref(),
        ) else {
            anyhow::ensure!(
                self.w4a16_w13_global_scale.is_none() && self.w4a16_w2_global_scale.is_none(),
                "layer {} has an incomplete W4A16 device global-scale pair",
                self.layer_id
            );
            return Ok(());
        };
        let bytes = self.expert_count * std::mem::size_of::<f32>();
        unsafe {
            library
                .copy_d2d_async(
                    w13_global_scale.buffer(),
                    self.w1_alphas.buffer(),
                    bytes,
                    cuda_stream,
                )
                .context("copying W4A16 W13 global scales to device memory")?;
            library
                .copy_d2d_async(
                    w2_global_scale.buffer(),
                    self.w2_alphas.buffer(),
                    bytes,
                    cuda_stream,
                )
                .context("copying W4A16 W2 global scales to device memory")?;
            library
                .cuda_stream_synchronize(cuda_stream)
                .context("synchronizing W4A16 device global scales")?;
        }
        Ok(())
    }

    fn w4a16_moe_buffers(&self) -> Result<B12xSparkW4a16LayerBuffers> {
        anyhow::ensure!(
            self.expert_count == 256
                && self.gate_rows == 512
                && self.up_rows == 512
                && self.down_rows == 6144,
            "layer {} W4A16 slab has unsupported experts={} rows={}/{}/{}",
            self.layer_id,
            self.expert_count,
            self.gate_rows,
            self.up_rows,
            self.down_rows
        );
        Ok(B12xSparkW4a16LayerBuffers {
            w13_weight: self.w13_weight.buffer(),
            w2_weight: self.w2_weight.buffer(),
            w13_scale: self.w13_scale.buffer(),
            w2_scale: self.w2_scale.buffer(),
            w13_global_scale: self
                .w4a16_w13_global_scale
                .as_deref()
                .unwrap_or(self.w1_alphas.as_ref())
                .buffer(),
            w2_global_scale: self
                .w4a16_w2_global_scale
                .as_deref()
                .unwrap_or(self.w2_alphas.as_ref())
                .buffer(),
        })
    }
}

#[derive(Clone, Copy)]
struct B12xSparkW4a16LayerBuffers {
    w13_weight: Ds41rtDeviceBuffer,
    w2_weight: Ds41rtDeviceBuffer,
    w13_scale: Ds41rtDeviceBuffer,
    w2_scale: Ds41rtDeviceBuffer,
    w13_global_scale: Ds41rtDeviceBuffer,
    w2_global_scale: Ds41rtDeviceBuffer,
}

fn b12x_w4a16_moe_buffers(
    layer: B12xSparkW4a16LayerBuffers,
    workspace: B12xSparkAotRouteWorkspaceBuffers,
    input: Ds41rtDeviceBuffer,
    output: Ds41rtDeviceBuffer,
    topk_weights: Ds41rtDeviceBuffer,
) -> Ds41rtB12xSparkW4a16MoeBuffers {
    let route_output_bytes = B12X_W4A16_PREFILL_TOPK8_ROUTES * 6144 * std::mem::size_of::<u16>();
    let output = if output.bytes < route_output_bytes {
        workspace.group_output
    } else {
        output
    };
    Ds41rtB12xSparkW4a16MoeBuffers {
        input,
        w13_weight: layer.w13_weight,
        w2_weight: layer.w2_weight,
        fc1_output: workspace.w4a16_fc1_output,
        activated: workspace.w4a16_activated,
        output,
        w13_scale: layer.w13_scale,
        w2_scale: layer.w2_scale,
        w13_global_scale: layer.w13_global_scale,
        w2_global_scale: layer.w2_global_scale,
        packed_route_indices: workspace.w4a16_packed_route_indices,
        block_expert_ids: workspace.w4a16_block_expert_ids,
        packed_route_count: workspace.w4a16_packed_route_count,
        topk_weights,
        fc1_scratch: workspace.w4a16_fc1_scratch,
        fc2_scratch: workspace.w4a16_fc2_scratch,
        locks: workspace.w4a16_locks,
    }
}

unsafe fn launch_b12x_w4a16_decode(
    library: &NativeLibrary,
    layer: B12xSparkW4a16LayerBuffers,
    buffers: &Ds41rtB12xSparkW4a16MoeBuffers,
    input_payload: Ds41rtDeviceBuffer,
    input_payload_stride_bytes: usize,
    topk_ids: Ds41rtDeviceBuffer,
    cuda_stream: *mut c_void,
) -> Result<()> {
    let _ = layer;
    unsafe {
        library.cuda_b12x_spark_w4a16_decode_m1_nvfp4_async(
            buffers,
            input_payload,
            input_payload_stride_bytes,
            topk_ids,
            cuda_stream,
        )
    }
}

#[derive(Clone, Copy)]
struct LoadedRouteCudaProjectionGeometry {
    rows: usize,
    weight_row_stride_bytes: usize,
    scale_row_stride_bytes: usize,
    weight_bytes: usize,
    scale_bytes: usize,
}

fn loaded_route_cuda_projection_geometry(
    projection: &LoadedRouteCudaProjectionShard,
) -> Result<LoadedRouteCudaProjectionGeometry> {
    loaded_route_cuda_projection_geometry_inner(projection, true)
}

fn loaded_route_cuda_projection_geometry_inner(
    projection: &LoadedRouteCudaProjectionShard,
    require_loaded_bytes: bool,
) -> Result<LoadedRouteCudaProjectionGeometry> {
    anyhow::ensure!(
        projection.weight.info.dtype == DType::U8
            && projection.weight_scale.info.dtype == DType::F8E4M3,
        "contiguous CUDA slab expects U8 weights and F8E4M3 scales, got {:?} and {:?}",
        projection.weight.info.dtype,
        projection.weight_scale.info.dtype
    );
    anyhow::ensure!(
        projection.weight.row_count == projection.weight_scale.row_count,
        "contiguous CUDA slab weight/scale rows differ: {} vs {}",
        projection.weight.row_count,
        projection.weight_scale.row_count
    );
    let weight_row_stride_bytes = projection
        .weight
        .row_width
        .checked_mul(projection.weight.bytes_per_scalar)
        .context("contiguous slab weight row stride overflow")?;
    let scale_row_stride_bytes = projection
        .weight_scale
        .row_width
        .checked_mul(projection.weight_scale.bytes_per_scalar)
        .context("contiguous slab scale row stride overflow")?;
    let weight_bytes = projection
        .weight
        .row_count
        .checked_mul(weight_row_stride_bytes)
        .context("contiguous slab weight bytes overflow")?;
    let scale_bytes = projection
        .weight_scale
        .row_count
        .checked_mul(scale_row_stride_bytes)
        .context("contiguous slab scale bytes overflow")?;
    if require_loaded_bytes {
        let loaded_weight_bytes = projection
            .weight
            .row_count
            .checked_mul(projection.weight.source_row_width)
            .and_then(|values| values.checked_mul(projection.weight.bytes_per_scalar))
            .context("contiguous slab loaded weight bytes overflow")?;
        let loaded_scale_bytes = projection
            .weight_scale
            .row_count
            .checked_mul(projection.weight_scale.source_row_width)
            .and_then(|values| values.checked_mul(projection.weight_scale.bytes_per_scalar))
            .context("contiguous slab loaded scale bytes overflow")?;
        anyhow::ensure!(
            projection.weight.bytes.len() == loaded_weight_bytes
                && projection.weight_scale.bytes.len() == loaded_scale_bytes,
            "contiguous slab loaded byte lengths differ from projection geometry"
        );
    }
    Ok(LoadedRouteCudaProjectionGeometry {
        rows: projection.weight.row_count,
        weight_row_stride_bytes,
        scale_row_stride_bytes,
        weight_bytes,
        scale_bytes,
    })
}

pub(in crate::commands::real_full) struct RouteExecution {
    pub(in crate::commands::real_full) outputs: Vec<f32>,
    pub(in crate::commands::real_full) weight_bytes_read: u64,
    pub(in crate::commands::real_full) quant_metadata_bytes_read: u64,
    #[allow(dead_code)]
    pub(in crate::commands::real_full) kernel_backend: &'static str,
}

pub(in crate::commands::real_full) struct RouteBf16AccumulatedExecution {
    pub(in crate::commands::real_full) output_bf16: Vec<u8>,
    pub(in crate::commands::real_full) completion_slices: Vec<Vec<usize>>,
    #[allow(dead_code)]
    pub(in crate::commands::real_full) kernel_backend: &'static str,
}

pub(in crate::commands::real_full) struct Ds4FlashRouteExecution {
    pub(in crate::commands::real_full) output_bf16: Vec<u8>,
    pub(in crate::commands::real_full) output_device: Option<Ds41rtDeviceBuffer>,
    pub(in crate::commands::real_full) completed_rows: Vec<usize>,
    pub(in crate::commands::real_full) reduction_follower: bool,
    #[allow(dead_code)]
    pub(in crate::commands::real_full) kernel_backend: &'static str,
}

struct Ds4FlashPackedPrefillPlan {
    packed_route_indices: Vec<u32>,
    block_expert_ids: Vec<u32>,
    packed_route_count: u32,
    topk_weights: Vec<f32>,
}

fn plan_ds4_flash_packed_topk6_routes(
    rows: usize,
    topk_ids: &[u32],
    topk_weights: &[f32],
) -> Result<Ds4FlashPackedPrefillPlan> {
    plan_ds4_packed_topk6_routes(
        rows,
        topk_ids,
        topk_weights,
        DS4_FLASH_ROUTED_EXPERTS,
        DS4_FLASH_PREFILL_MAX_PACKED_ROUTE_SLOTS,
        DS4_FLASH_PREFILL_MAX_ROUTE_BLOCKS,
        "Flash",
    )
}

fn plan_ds4_packed_topk6_routes(
    rows: usize,
    topk_ids: &[u32],
    topk_weights: &[f32],
    experts: usize,
    max_packed_route_slots: usize,
    max_route_blocks: usize,
    variant: &str,
) -> Result<Ds4FlashPackedPrefillPlan> {
    let route_count = rows
        .checked_mul(DS4_FLASH_TOP_K)
        .context("native DeepSeek V4 route count overflow")?;
    anyhow::ensure!(
        rows > 0
            && rows <= DS4_FLASH_PREFILL_MAX_ROWS
            && topk_ids.len() == route_count
            && topk_weights.len() == route_count,
        "native {variant} requires rows=1..={DS4_FLASH_PREFILL_MAX_ROWS} and exactly {DS4_FLASH_TOP_K} routes per row, got rows={rows} ids={} weights={}",
        topk_ids.len(),
        topk_weights.len()
    );
    anyhow::ensure!(
        topk_ids.iter().all(|&expert_id| expert_id < experts as u32),
        "native {variant} route contains an expert outside 0..{experts}"
    );
    anyhow::ensure!(
        topk_weights.iter().all(|weight| weight.is_finite()),
        "native Flash route contains a non-finite weight"
    );
    if rows == 1 {
        return Ok(Ds4FlashPackedPrefillPlan {
            packed_route_indices: topk_ids.to_vec(),
            block_expert_ids: Vec::new(),
            packed_route_count: u32::try_from(route_count)
                .context("native Flash direct route count exceeds u32")?,
            topk_weights: topk_weights.to_vec(),
        });
    }

    let route_count_u32 =
        u32::try_from(route_count).context("native Flash route count exceeds u32")?;
    let mut expert_routes = vec![Vec::<u32>::new(); experts];
    for (route_index, &expert_id) in topk_ids.iter().enumerate() {
        expert_routes[expert_id as usize]
            .push(u32::try_from(route_index).context("native Flash route index exceeds u32")?);
    }
    let mut packed_route_indices = Vec::new();
    let mut block_expert_ids = Vec::new();
    for (expert_id, routes) in expert_routes.into_iter().enumerate() {
        if routes.is_empty() {
            continue;
        }
        let blocks = routes.len().div_ceil(DS4_FLASH_PREFILL_ROUTE_BLOCK_ROWS);
        let padded_routes = blocks
            .checked_mul(DS4_FLASH_PREFILL_ROUTE_BLOCK_ROWS)
            .context("native Flash padded route count overflow")?;
        let padding = padded_routes - routes.len();
        packed_route_indices.extend(routes);
        packed_route_indices.extend(std::iter::repeat_n(route_count_u32, padding));
        block_expert_ids.extend(std::iter::repeat_n(
            u32::try_from(expert_id).context("native Flash expert ID exceeds u32")?,
            blocks,
        ));
    }
    anyhow::ensure!(
        packed_route_indices.len() <= max_packed_route_slots
            && block_expert_ids.len() <= max_route_blocks,
        "native {variant} packed route metadata exceeds AOT capacity: routes={} blocks={} max_routes={} max_blocks={}",
        packed_route_indices.len(),
        block_expert_ids.len(),
        max_packed_route_slots,
        max_route_blocks
    );
    Ok(Ds4FlashPackedPrefillPlan {
        packed_route_count: u32::try_from(packed_route_indices.len())
            .context("native Flash packed route count exceeds u32")?,
        packed_route_indices,
        block_expert_ids,
        topk_weights: topk_weights.to_vec(),
    })
}

pub(in crate::commands::real_full) struct RouteBf16AccumulatedDeviceExecution {
    pub(in crate::commands::real_full) output_device: DeviceBf16Output,
    pub(in crate::commands::real_full) kernel_backend: &'static str,
}

pub(in crate::commands::real_full) struct RouteBf16AccumulatedStreamingExecution {
    pub(in crate::commands::real_full) completion_slices: Vec<Vec<usize>>,
    #[allow(dead_code)]
    pub(in crate::commands::real_full) kernel_backend: &'static str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::commands::real_full) enum RouteStreamingOutputDtype {
    Bf16,
    Fp8E4m3RowScaled,
    Nvfp4E2m1Fp8E4m3,
}

impl RouteStreamingOutputDtype {
    fn row_stride_bytes(self, values_per_row: usize) -> Result<usize> {
        match self {
            Self::Bf16 => values_per_row
                .checked_mul(std::mem::size_of::<u16>())
                .context("streaming BF16 route output row bytes overflow usize"),
            Self::Fp8E4m3RowScaled => values_per_row
                .checked_add(std::mem::size_of::<f32>())
                .context("streaming row-scaled FP8 route output row bytes overflow usize"),
            Self::Nvfp4E2m1Fp8E4m3 => {
                anyhow::ensure!(
                    values_per_row > 0 && values_per_row % 16 == 0,
                    "streaming NVFP4 route output width must be a positive multiple of 16"
                );
                values_per_row
                    .checked_div(2)
                    .and_then(|packed| packed.checked_add(values_per_row / 16))
                    .context("streaming NVFP4 route output row bytes overflow usize")
            }
        }
    }
}

struct RouteBf16AccumulatedInnerExecution {
    output_bf16: Option<Vec<u8>>,
    output_device: Option<DeviceBf16Output>,
    completion_slices: Vec<Vec<usize>>,
    kernel_backend: &'static str,
}

enum RouteHiddenBatch<'a> {
    HostBf16(&'a [u8]),
    HostNvfp4(&'a [u8]),
    DeviceBf16 {
        output: &'a DeviceBf16Output,
        host_for_validation: Option<&'a [u8]>,
    },
}

impl RouteHiddenBatch<'_> {
    fn validate(
        &self,
        row_count: usize,
        hidden_dim: usize,
        hidden_row_stride_bytes: usize,
        hidden_bytes: usize,
    ) -> Result<()> {
        match self {
            Self::HostBf16(bytes) | Self::HostNvfp4(bytes) => {
                validate_route_hidden_host_bytes(bytes, hidden_bytes)
            }
            Self::DeviceBf16 {
                output,
                host_for_validation,
            } => {
                let expected_stride_values = hidden_row_stride_bytes / std::mem::size_of::<u16>();
                if output.rows != row_count || output.values_per_row != expected_stride_values {
                    anyhow::bail!(
                        "BF16 NVFP4 accumulated route device hidden shape mismatch: expected rows={} stride_values={} hidden_dim={} got rows={} values_per_row={}",
                        row_count,
                        expected_stride_values,
                        hidden_dim,
                        output.rows,
                        output.values_per_row
                    );
                }
                let buffer = output.buffer();
                if buffer.ptr.is_null() {
                    anyhow::bail!("BF16 NVFP4 accumulated route device hidden buffer is null");
                }
                if buffer.bytes < hidden_bytes {
                    anyhow::bail!(
                        "BF16 NVFP4 accumulated route device hidden buffer has {} bytes, needs {hidden_bytes}",
                        buffer.bytes
                    );
                }
                if cuda_route_validation_enabled() {
                    let host = host_for_validation.with_context(|| {
                        "CUDA NVFP4 route validation requires host-visible hidden bytes for device-input route execution"
                    })?;
                    validate_route_hidden_host_bytes(host, hidden_bytes)?;
                } else if let Some(host) = host_for_validation {
                    validate_route_hidden_host_bytes(host, hidden_bytes)?;
                }
                Ok(())
            }
        }
    }

    fn device_buffer(&self) -> Option<Ds41rtDeviceBuffer> {
        match self {
            Self::HostBf16(_) | Self::HostNvfp4(_) => None,
            Self::DeviceBf16 { output, .. } => Some(output.buffer()),
        }
    }

    fn host_slice(&self, hidden_bytes: usize) -> Option<&[u8]> {
        match self {
            Self::HostBf16(bytes) | Self::HostNvfp4(bytes) => Some(&bytes[..hidden_bytes]),
            Self::DeviceBf16 {
                host_for_validation,
                ..
            } => host_for_validation.map(|bytes| &bytes[..hidden_bytes]),
        }
    }

    fn kernel_backend(&self) -> &'static str {
        match self {
            Self::HostBf16(_) => CUDA_REFERENCE_NVFP4_ROUTE_BF16_ACCUMULATED_BACKEND,
            Self::HostNvfp4(_) => B12X_SPARK_DIRECT_NVFP4_ROUTE_BF16_ACCUMULATED_BACKEND,
            Self::DeviceBf16 { .. } => {
                CUDA_REFERENCE_NVFP4_ROUTE_BF16_ACCUMULATED_DEVICE_INPUT_BACKEND
            }
        }
    }

    fn is_nvfp4_payload(&self) -> bool {
        matches!(self, Self::HostNvfp4(_))
    }
}

fn validate_route_hidden_host_bytes(bytes: &[u8], hidden_bytes: usize) -> Result<()> {
    if bytes.len() < hidden_bytes {
        anyhow::bail!(
            "BF16 NVFP4 accumulated hidden batch has {} bytes, needs {hidden_bytes}",
            bytes.len()
        );
    }
    Ok(())
}

/// Launch one allocation-free Flash decode row from fixed device ingress into
/// this rank's fixed BF16 partial. The caller has already ordered transport
/// completion on `cuda_stream`; this function performs no host route packing,
/// copies, synchronization, reduction, or readback.
#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) fn ds4_flash_spark_decode_m1_preloaded_device_input_into(
    layer_id: usize,
    buffers: Ds4FlashSparkDecodeM1DeviceBuffers,
    cuda_stream: *mut c_void,
    cache: &mut RouteTensorCache,
) -> Result<()> {
    cache.prepare_layer(layer_id);
    let cuda_cache = cache.cuda_cache()?;
    anyhow::ensure!(
        !cuda_cache.legacy_spark_runtime_initialization_enabled
            && (cuda_cache.spark_reduction.is_none()
                || cuda_cache.spark_rdma_reduction.is_none()),
        "native Flash TP=4 execution requires its dedicated CUDA cache with at most one Spark reduction backend"
    );
    cuda_cache.prepare_layer(layer_id);
    cuda_cache.ensure_ds4_flash_aot()?;
    let slab = cuda_cache.ds4_flash_expert_slabs.get(&layer_id).cloned();
    let exl3_slab = cuda_cache
        .ds4_flash_exl3_expert_slabs
        .get(&layer_id)
        .cloned();
    let mixed_slab = cuda_cache
        .ds4_flash_exl3_mixed_expert_slabs
        .get(&layer_id)
        .cloned();
    anyhow::ensure!(
        [slab.is_some(), exl3_slab.is_some(), mixed_slab.is_some()]
            .into_iter()
            .filter(|value| *value)
            .count()
            == 1,
        "native Flash layer {layer_id} must have exactly one resident expert representation"
    );
    validate_ds4_flash_spark_decode_m1_buffer_layout(
        buffers,
        exl3_slab.is_some() || mixed_slab.is_some(),
    )?;
    unsafe {
        if let Some(slab) = mixed_slab {
            let native_buffers = slab.decode_m1_aot_buffers(buffers);
            slab.launch_mixed(&cuda_cache.library, &native_buffers, 1, cuda_stream)
                .with_context(|| {
                    format!(
                        "launching native Flash mixed EXL3 layer {layer_id} caller-owned M1 TP4 rank partial"
                    )
                })
        } else if let Some(slab) = exl3_slab {
            let native_buffers = slab.decode_m1_aot_buffers(buffers);
            let launch = match slab.profile.geometry.trellis_bits {
                2 => cuda_cache
                    .library
                    .cuda_ds4_flash_spark_exl3_k2_decode_m1_async(&native_buffers, cuda_stream),
                3 => cuda_cache
                    .library
                    .cuda_ds4_flash_spark_exl3_k3_decode_m1_async(&native_buffers, cuda_stream),
                bits => anyhow::bail!("unsupported Flash EXL3 tier K{bits}"),
            };
            launch.with_context(|| {
                format!(
                    "launching native Flash EXL3 layer {layer_id} caller-owned M1 TP4 rank partial"
                )
            })
        } else {
            let native_buffers = slab
                .expect("native Flash W4A16 slab selected")
                .decode_m1_aot_buffers(buffers);
            cuda_cache
                .library
                .cuda_ds4_flash_spark_w4a16_decode_m1_bf16_async(&native_buffers, cuda_stream)
                .with_context(|| {
                    format!(
                        "launching native Flash layer {layer_id} caller-owned M1 TP4 rank partial"
                    )
                })
        }
    }
}

fn validate_ds4_flash_spark_decode_m1_buffer_layout(
    buffers: Ds4FlashSparkDecodeM1DeviceBuffers,
    exl3: bool,
) -> Result<()> {
    let mut named = vec![
        ("input", buffers.input, DS4_FLASH_DECODE_M1_HIDDEN_BYTES),
        (
            "route_indices",
            buffers.route_indices,
            DS4_FLASH_DECODE_M1_ROUTE_BYTES,
        ),
        (
            "route_weights",
            buffers.route_weights,
            DS4_FLASH_DECODE_M1_ROUTE_BYTES,
        ),
        ("fc1", buffers.fc1, DS4_FLASH_DECODE_M1_FC1_BYTES),
        (
            "activated",
            buffers.activated,
            DS4_FLASH_DECODE_M1_ACTIVATED_BYTES,
        ),
        ("output", buffers.output, DS4_FLASH_DECODE_M1_HIDDEN_BYTES),
        (
            "block_expert_ids",
            buffers.block_expert_ids,
            std::mem::size_of::<i32>(),
        ),
        (
            "packed_route_count",
            buffers.packed_route_count,
            std::mem::size_of::<i32>(),
        ),
        (
            "fc1_scratch",
            buffers.fc1_scratch,
            DS4_FLASH_DECODE_M1_FC1_SCRATCH_BYTES,
        ),
        (
            "fc2_scratch",
            buffers.fc2_scratch,
            DS4_FLASH_DECODE_M1_FC2_SCRATCH_BYTES,
        ),
        ("locks", buffers.locks, DS4_FLASH_DECODE_M1_LOCK_BYTES),
    ];
    if exl3 {
        named.extend([
            (
                "rotation_gate",
                buffers.rotation_gate,
                DS4_FLASH_DECODE_M1_ROUTED_OUTPUT_BYTES,
            ),
            (
                "rotation_up",
                buffers.rotation_up,
                DS4_FLASH_DECODE_M1_ROUTED_OUTPUT_BYTES,
            ),
            (
                "routed_output",
                buffers.routed_output,
                DS4_FLASH_DECODE_M1_ROUTED_OUTPUT_BYTES,
            ),
            (
                "output_f32",
                buffers.output_f32,
                DS4_FLASH_DECODE_M1_OUTPUT_F32_BYTES,
            ),
            (
                "expert_counts",
                buffers.expert_counts,
                DS4_FLASH_PREFILL_EXPERT_COUNT_BYTES,
            ),
            (
                "expert_offsets",
                buffers.expert_offsets,
                DS4_FLASH_PREFILL_EXPERT_OFFSET_BYTES,
            ),
        ]);
    }
    let input_device = buffers.input.device_id;
    for &(name, buffer, bytes) in &named {
        anyhow::ensure!(
            !buffer.ptr.is_null(),
            "native Flash Spark decode M1 {name} buffer is null"
        );
        device_buffer_byte_view(buffer, 0, bytes, name)?;
        anyhow::ensure!(
            buffer.device_id == input_device,
            "native Flash Spark decode M1 {name} must be on input device {input_device}"
        );
    }
    for left in 0..named.len() {
        for right in left + 1..named.len() {
            anyhow::ensure!(
                !ds4_flash_device_buffer_ranges_overlap(
                    named[left].1,
                    named[left].2,
                    named[right].1,
                    named[right].2,
                )?,
                "native Flash Spark decode M1 {} and {} buffers overlap",
                named[left].0,
                named[right].0,
            );
        }
    }
    Ok(())
}

fn ds4_flash_device_buffer_ranges_overlap(
    left: Ds41rtDeviceBuffer,
    left_bytes: usize,
    right: Ds41rtDeviceBuffer,
    right_bytes: usize,
) -> Result<bool> {
    let left_start = left.ptr as usize;
    let right_start = right.ptr as usize;
    let left_end = left_start
        .checked_add(left_bytes)
        .context("native Flash Spark left buffer address overflow")?;
    let right_end = right_start
        .checked_add(right_bytes)
        .context("native Flash Spark right buffer address overflow")?;
    Ok(left_start < right_end && right_start < left_end)
}

/// Pack one fixed-capacity Flash prefill route buffer entirely on the Spark
/// device. This mirrors SparkInfer's replicated-expert block-32 pack: every
/// rank sees every global expert ID, and no expert ownership map is applied.
#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) fn ds4_flash_spark_prefill_pack_topk6_device_into(
    rows: usize,
    buffers: Ds4FlashSparkPrefillRoutePackDeviceBuffers,
    cuda_stream: *mut c_void,
    library: &NativeLibrary,
) -> Result<()> {
    anyhow::ensure!(
        (2..=DS4_FLASH_PREFILL_MAX_ROWS).contains(&rows),
        "native Flash Spark prefill route pack rows {rows} are outside 2..={DS4_FLASH_PREFILL_MAX_ROWS}"
    );
    validate_ds4_flash_spark_prefill_route_pack_buffer_layout(buffers)?;
    let native_buffers = Ds41rtDs4FlashRoutePackBuffers {
        topk_ids: buffers.topk_ids,
        packed_route_indices: buffers.packed_route_indices,
        block_expert_ids: buffers.block_expert_ids,
        packed_route_count: buffers.packed_route_count,
        expert_counts: buffers.expert_counts,
        expert_offsets: buffers.expert_offsets,
    };
    unsafe {
        library
            .cuda_ds4_flash_pack_topk6_routes_async(&native_buffers, rows, cuda_stream)
            .with_context(|| {
                format!("packing {rows}-row native Flash top-k=6 routes on the replicated TP4 rank")
            })
    }
}

fn validate_ds4_flash_spark_prefill_route_pack_buffer_layout(
    buffers: Ds4FlashSparkPrefillRoutePackDeviceBuffers,
) -> Result<()> {
    let named = [
        (
            "topk_ids",
            buffers.topk_ids,
            DS4_FLASH_PREFILL_TOPK_ID_BYTES,
        ),
        (
            "packed_route_indices",
            buffers.packed_route_indices,
            DS4_FLASH_PREFILL_PACKED_ROUTE_BYTES,
        ),
        (
            "block_expert_ids",
            buffers.block_expert_ids,
            DS4_FLASH_PREFILL_BLOCK_EXPERT_BYTES,
        ),
        (
            "packed_route_count",
            buffers.packed_route_count,
            std::mem::size_of::<i32>(),
        ),
        (
            "expert_counts",
            buffers.expert_counts,
            DS4_FLASH_PREFILL_EXPERT_COUNT_BYTES,
        ),
        (
            "expert_offsets",
            buffers.expert_offsets,
            DS4_FLASH_PREFILL_EXPERT_OFFSET_BYTES,
        ),
    ];
    let input_device = buffers.topk_ids.device_id;
    for &(name, buffer, bytes) in &named {
        anyhow::ensure!(
            !buffer.ptr.is_null(),
            "native Flash Spark prefill route pack {name} buffer is null"
        );
        device_buffer_byte_view(buffer, 0, bytes, name)?;
        anyhow::ensure!(
            buffer.device_id == input_device,
            "native Flash Spark prefill route pack {name} must be on input device {input_device}"
        );
    }
    for left in 0..named.len() {
        for right in left + 1..named.len() {
            anyhow::ensure!(
                !ds4_flash_device_buffer_ranges_overlap(
                    named[left].1,
                    named[left].2,
                    named[right].1,
                    named[right].2,
                )?,
                "native Flash Spark prefill route pack {} and {} buffers overlap",
                named[left].0,
                named[right].0,
            );
        }
    }
    Ok(())
}

/// Pack and execute one Flash prefill bucket from fixed device ingress into
/// this rank's BF16 partial. The caller orders transport completion on the
/// stream; this boundary performs no host packing, allocation, copies,
/// synchronization, cross-rank reduction, or readback.
#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) fn ds4_flash_spark_prefill_topk6_preloaded_device_input_into(
    layer_id: usize,
    rows: usize,
    buffers: Ds4FlashSparkPrefillDeviceBuffers,
    cuda_stream: *mut c_void,
    cache: &mut RouteTensorCache,
) -> Result<()> {
    anyhow::ensure!(
        (2..=DS4_FLASH_PREFILL_MAX_ROWS).contains(&rows),
        "native Flash Spark prefill rows {rows} are outside 2..={DS4_FLASH_PREFILL_MAX_ROWS}"
    );
    cache.prepare_layer(layer_id);
    let cuda_cache = cache.cuda_cache()?;
    anyhow::ensure!(
        !cuda_cache.legacy_spark_runtime_initialization_enabled
            && (cuda_cache.spark_reduction.is_none()
                || cuda_cache.spark_rdma_reduction.is_none()),
        "native Flash TP=4 execution requires its dedicated CUDA cache with at most one Spark reduction backend"
    );
    cuda_cache.prepare_layer(layer_id);
    cuda_cache.ensure_ds4_flash_aot()?;
    let slab = cuda_cache.ds4_flash_expert_slabs.get(&layer_id).cloned();
    let exl3_slab = cuda_cache
        .ds4_flash_exl3_expert_slabs
        .get(&layer_id)
        .cloned();
    let mixed_slab = cuda_cache
        .ds4_flash_exl3_mixed_expert_slabs
        .get(&layer_id)
        .cloned();
    anyhow::ensure!(
        [slab.is_some(), exl3_slab.is_some(), mixed_slab.is_some()]
            .into_iter()
            .filter(|value| *value)
            .count()
            == 1,
        "native Flash layer {layer_id} must have exactly one resident expert representation"
    );
    validate_ds4_flash_spark_prefill_buffer_layout(
        buffers,
        exl3_slab.is_some() || mixed_slab.is_some(),
    )?;
    let library = Arc::clone(&cuda_cache.library);
    unsafe {
        if let Some(slab) = mixed_slab {
            let native_buffers = slab.prefill_aot_buffers(buffers);
            slab.launch_mixed(&library, &native_buffers, rows, cuda_stream)
                .with_context(|| {
                    format!(
                        "launching native Flash mixed EXL3 layer {layer_id} caller-owned {rows}-row TP4 rank partial"
                    )
                })
        } else if let Some(slab) = exl3_slab {
            let native_buffers = slab.prefill_aot_buffers(buffers);
            let launch = match slab.profile.geometry.trellis_bits {
                2 => library.cuda_ds4_flash_spark_exl3_k2_prefill_topk6_async(
                    &native_buffers,
                    rows,
                    cuda_stream,
                ),
                3 => library.cuda_ds4_flash_spark_exl3_k3_prefill_topk6_async(
                    &native_buffers,
                    rows,
                    cuda_stream,
                ),
                bits => anyhow::bail!("unsupported Flash EXL3 tier K{bits}"),
            };
            launch
                .with_context(|| {
                    format!(
                        "launching native Flash EXL3 layer {layer_id} caller-owned {rows}-row TP4 rank partial"
                    )
                })
        } else {
            ds4_flash_spark_prefill_pack_topk6_device_into(
                rows,
                buffers.route_pack(),
                cuda_stream,
                library.as_ref(),
            )?;
            let native_buffers = slab
                .expect("native Flash W4A16 slab selected")
                .prefill_aot_buffers(buffers);
            library
                .cuda_ds4_flash_spark_w4a16_prefill_topk6_bf16_async(
                    &native_buffers,
                    rows,
                    cuda_stream,
                )
                .with_context(|| {
                    format!(
                        "launching native Flash layer {layer_id} caller-owned {rows}-row TP4 rank partial"
                    )
                })
        }
    }
}

fn validate_ds4_flash_spark_prefill_buffer_layout(
    buffers: Ds4FlashSparkPrefillDeviceBuffers,
    exl3: bool,
) -> Result<()> {
    let mut named = vec![
        ("input", buffers.input, DS4_FLASH_PREFILL_HIDDEN_BYTES),
        (
            "topk_ids",
            buffers.topk_ids,
            DS4_FLASH_PREFILL_TOPK_ID_BYTES,
        ),
        (
            "topk_weights",
            buffers.topk_weights,
            DS4_FLASH_PREFILL_TOPK_WEIGHT_BYTES,
        ),
        (
            "packed_route_indices",
            buffers.packed_route_indices,
            DS4_FLASH_PREFILL_PACKED_ROUTE_BYTES,
        ),
        (
            "block_expert_ids",
            buffers.block_expert_ids,
            DS4_FLASH_PREFILL_BLOCK_EXPERT_BYTES,
        ),
        (
            "packed_route_count",
            buffers.packed_route_count,
            std::mem::size_of::<i32>(),
        ),
        (
            "expert_counts",
            buffers.expert_counts,
            DS4_FLASH_PREFILL_EXPERT_COUNT_BYTES,
        ),
        (
            "expert_offsets",
            buffers.expert_offsets,
            DS4_FLASH_PREFILL_EXPERT_OFFSET_BYTES,
        ),
        ("fc1", buffers.fc1, DS4_FLASH_PREFILL_FC1_BYTES),
        (
            "activated",
            buffers.activated,
            DS4_FLASH_PREFILL_ACTIVATED_BYTES,
        ),
        (
            "routed_output",
            buffers.routed_output,
            DS4_FLASH_PREFILL_ROUTED_OUTPUT_BYTES,
        ),
        ("output", buffers.output, DS4_FLASH_PREFILL_HIDDEN_BYTES),
        (
            "fc1_scratch",
            buffers.fc1_scratch,
            DS4_FLASH_PREFILL_COMPUTE_SCRATCH_BYTES,
        ),
        (
            "fc2_scratch",
            buffers.fc2_scratch,
            DS4_FLASH_PREFILL_COMPUTE_SCRATCH_BYTES,
        ),
        ("locks", buffers.locks, DS4_FLASH_DECODE_M1_LOCK_BYTES),
    ];
    if exl3 {
        named.extend([
            (
                "rotation_gate",
                buffers.rotation_gate,
                DS4_FLASH_PREFILL_ROUTED_OUTPUT_BYTES,
            ),
            (
                "rotation_up",
                buffers.rotation_up,
                DS4_FLASH_PREFILL_ROUTED_OUTPUT_BYTES,
            ),
            (
                "output_f32",
                buffers.output_f32,
                DS4_FLASH_PREFILL_OUTPUT_F32_BYTES,
            ),
        ]);
    }
    let input_device = buffers.input.device_id;
    for &(name, buffer, bytes) in &named {
        anyhow::ensure!(
            !buffer.ptr.is_null(),
            "native Flash Spark prefill {name} buffer is null"
        );
        device_buffer_byte_view(buffer, 0, bytes, name)?;
        anyhow::ensure!(
            buffer.device_id == input_device,
            "native Flash Spark prefill {name} must be on input device {input_device}"
        );
    }
    for left in 0..named.len() {
        for right in left + 1..named.len() {
            anyhow::ensure!(
                !ds4_flash_device_buffer_ranges_overlap(
                    named[left].1,
                    named[left].2,
                    named[right].1,
                    named[right].2,
                )?,
                "native Flash Spark prefill {} and {} buffers overlap",
                named[left].0,
                named[right].0,
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod ds4_flash_spark_decode_m1_handoff_tests {
    use super::{
        validate_ds4_flash_spark_decode_m1_buffer_layout, Ds4FlashSparkDecodeM1DeviceBuffers,
        DS4_FLASH_DECODE_M1_ACTIVATED_BYTES, DS4_FLASH_DECODE_M1_FC1_BYTES,
        DS4_FLASH_DECODE_M1_FC1_SCRATCH_BYTES, DS4_FLASH_DECODE_M1_FC2_SCRATCH_BYTES,
        DS4_FLASH_DECODE_M1_HIDDEN_BYTES, DS4_FLASH_DECODE_M1_LOCK_BYTES,
        DS4_FLASH_DECODE_M1_OUTPUT_F32_BYTES, DS4_FLASH_DECODE_M1_ROUTED_OUTPUT_BYTES,
        DS4_FLASH_DECODE_M1_ROUTE_BYTES, DS4_FLASH_PREFILL_EXPERT_COUNT_BYTES,
        DS4_FLASH_PREFILL_EXPERT_OFFSET_BYTES,
    };
    use ds41rt_ffi::Ds41rtDeviceBuffer;

    fn buffer(address: usize, bytes: usize, device_id: i32) -> Ds41rtDeviceBuffer {
        Ds41rtDeviceBuffer {
            ptr: address as *mut _,
            bytes,
            device_id,
            flags: 0,
        }
    }

    fn fixed_buffers() -> Ds4FlashSparkDecodeM1DeviceBuffers {
        Ds4FlashSparkDecodeM1DeviceBuffers {
            input: buffer(0x10_0000, DS4_FLASH_DECODE_M1_HIDDEN_BYTES, 0),
            route_indices: buffer(0x11_0000, DS4_FLASH_DECODE_M1_ROUTE_BYTES, 0),
            route_weights: buffer(0x12_0000, DS4_FLASH_DECODE_M1_ROUTE_BYTES, 0),
            fc1: buffer(0x13_0000, DS4_FLASH_DECODE_M1_FC1_BYTES, 0),
            activated: buffer(0x14_0000, DS4_FLASH_DECODE_M1_ACTIVATED_BYTES, 0),
            rotation_gate: buffer(0x18_0000, DS4_FLASH_DECODE_M1_ROUTED_OUTPUT_BYTES, 0),
            rotation_up: buffer(0x19_0000, DS4_FLASH_DECODE_M1_ROUTED_OUTPUT_BYTES, 0),
            routed_output: buffer(0x1a_0000, DS4_FLASH_DECODE_M1_ROUTED_OUTPUT_BYTES, 0),
            output_f32: buffer(0x1b_0000, DS4_FLASH_DECODE_M1_OUTPUT_F32_BYTES, 0),
            output: buffer(0x15_0000, DS4_FLASH_DECODE_M1_HIDDEN_BYTES, 0),
            block_expert_ids: buffer(0x16_0000, std::mem::size_of::<i32>(), 0),
            packed_route_count: buffer(0x17_0000, std::mem::size_of::<i32>(), 0),
            expert_counts: buffer(0x1c_0000, DS4_FLASH_PREFILL_EXPERT_COUNT_BYTES, 0),
            expert_offsets: buffer(0x1d_0000, DS4_FLASH_PREFILL_EXPERT_OFFSET_BYTES, 0),
            fc1_scratch: buffer(0x20_0000, DS4_FLASH_DECODE_M1_FC1_SCRATCH_BYTES, 0),
            fc2_scratch: buffer(0x30_0000, DS4_FLASH_DECODE_M1_FC2_SCRATCH_BYTES, 0),
            locks: buffer(0x50_0000, DS4_FLASH_DECODE_M1_LOCK_BYTES, 0),
        }
    }

    #[test]
    fn flash_spark_decode_m1_handoff_accepts_fixed_tp_rank_regions() {
        validate_ds4_flash_spark_decode_m1_buffer_layout(fixed_buffers(), false).unwrap();
    }

    #[test]
    fn flash_spark_decode_m1_handoff_rejects_short_or_aliased_regions() {
        let short_routes = Ds4FlashSparkDecodeM1DeviceBuffers {
            route_indices: buffer(0x11_0000, DS4_FLASH_DECODE_M1_ROUTE_BYTES - 1, 0),
            ..fixed_buffers()
        };
        assert!(validate_ds4_flash_spark_decode_m1_buffer_layout(short_routes, false).is_err());

        let aliased_output = Ds4FlashSparkDecodeM1DeviceBuffers {
            output: fixed_buffers().input,
            ..fixed_buffers()
        };
        assert!(validate_ds4_flash_spark_decode_m1_buffer_layout(aliased_output, false).is_err());

        let wrong_device = Ds4FlashSparkDecodeM1DeviceBuffers {
            route_weights: buffer(0x12_0000, DS4_FLASH_DECODE_M1_ROUTE_BYTES, 1),
            ..fixed_buffers()
        };
        assert!(validate_ds4_flash_spark_decode_m1_buffer_layout(wrong_device, false).is_err());
    }
}

#[cfg(test)]
mod ds4_flash_spark_prefill_route_pack_handoff_tests {
    use super::{
        validate_ds4_flash_spark_prefill_route_pack_buffer_layout,
        Ds4FlashSparkPrefillRoutePackDeviceBuffers, DS4_FLASH_PREFILL_BLOCK_EXPERT_BYTES,
        DS4_FLASH_PREFILL_EXPERT_COUNT_BYTES, DS4_FLASH_PREFILL_EXPERT_OFFSET_BYTES,
        DS4_FLASH_PREFILL_PACKED_ROUTE_BYTES, DS4_FLASH_PREFILL_TOPK_ID_BYTES,
    };
    use ds41rt_ffi::Ds41rtDeviceBuffer;

    fn buffer(address: usize, bytes: usize, device_id: i32) -> Ds41rtDeviceBuffer {
        Ds41rtDeviceBuffer {
            ptr: address as *mut _,
            bytes,
            device_id,
            flags: 0,
        }
    }

    fn fixed_buffers() -> Ds4FlashSparkPrefillRoutePackDeviceBuffers {
        Ds4FlashSparkPrefillRoutePackDeviceBuffers {
            topk_ids: buffer(0x10_0000, DS4_FLASH_PREFILL_TOPK_ID_BYTES, 0),
            packed_route_indices: buffer(0x11_0000, DS4_FLASH_PREFILL_PACKED_ROUTE_BYTES, 0),
            block_expert_ids: buffer(0x13_0000, DS4_FLASH_PREFILL_BLOCK_EXPERT_BYTES, 0),
            packed_route_count: buffer(0x14_0000, std::mem::size_of::<i32>(), 0),
            expert_counts: buffer(0x15_0000, DS4_FLASH_PREFILL_EXPERT_COUNT_BYTES, 0),
            expert_offsets: buffer(0x16_0000, DS4_FLASH_PREFILL_EXPERT_OFFSET_BYTES, 0),
        }
    }

    #[test]
    fn flash_spark_prefill_route_pack_accepts_fixed_tp_rank_regions() {
        validate_ds4_flash_spark_prefill_route_pack_buffer_layout(fixed_buffers()).unwrap();
    }

    #[test]
    fn flash_spark_prefill_route_pack_rejects_short_alias_or_wrong_device() {
        let short_routes = Ds4FlashSparkPrefillRoutePackDeviceBuffers {
            packed_route_indices: buffer(0x11_0000, DS4_FLASH_PREFILL_PACKED_ROUTE_BYTES - 1, 0),
            ..fixed_buffers()
        };
        assert!(validate_ds4_flash_spark_prefill_route_pack_buffer_layout(short_routes).is_err());

        let aliased_counts = Ds4FlashSparkPrefillRoutePackDeviceBuffers {
            expert_counts: fixed_buffers().expert_offsets,
            ..fixed_buffers()
        };
        assert!(validate_ds4_flash_spark_prefill_route_pack_buffer_layout(aliased_counts).is_err());

        let wrong_device = Ds4FlashSparkPrefillRoutePackDeviceBuffers {
            block_expert_ids: buffer(0x13_0000, DS4_FLASH_PREFILL_BLOCK_EXPERT_BYTES, 1),
            ..fixed_buffers()
        };
        assert!(validate_ds4_flash_spark_prefill_route_pack_buffer_layout(wrong_device).is_err());
    }
}

#[cfg(test)]
mod ds4_flash_spark_prefill_handoff_tests {
    use super::{
        device_buffer_ptr_is_aligned, validate_ds4_flash_spark_prefill_buffer_layout,
        Ds4FlashSparkPrefillDeviceBuffers, DS4_FLASH_DECODE_M1_LOCK_BYTES,
        DS4_FLASH_PREFILL_ACTIVATED_BYTES, DS4_FLASH_PREFILL_BLOCK_EXPERT_BYTES,
        DS4_FLASH_PREFILL_COMPUTE_SCRATCH_BYTES, DS4_FLASH_PREFILL_EXPERT_COUNT_BYTES,
        DS4_FLASH_PREFILL_EXPERT_OFFSET_BYTES, DS4_FLASH_PREFILL_FC1_BYTES,
        DS4_FLASH_PREFILL_HIDDEN_BYTES, DS4_FLASH_PREFILL_OUTPUT_F32_BYTES,
        DS4_FLASH_PREFILL_PACKED_ROUTE_BYTES, DS4_FLASH_PREFILL_ROUTED_OUTPUT_BYTES,
        DS4_FLASH_PREFILL_TOPK_ID_BYTES, DS4_FLASH_PREFILL_TOPK_WEIGHT_BYTES,
    };
    use ds41rt_ffi::Ds41rtDeviceBuffer;

    fn buffer(address: usize, bytes: usize, device_id: i32) -> Ds41rtDeviceBuffer {
        Ds41rtDeviceBuffer {
            ptr: address as *mut _,
            bytes,
            device_id,
            flags: 0,
        }
    }

    fn fixed_buffers() -> Ds4FlashSparkPrefillDeviceBuffers {
        Ds4FlashSparkPrefillDeviceBuffers {
            input: buffer(0x1000_0000, DS4_FLASH_PREFILL_HIDDEN_BYTES, 0),
            topk_ids: buffer(0x1200_0000, DS4_FLASH_PREFILL_TOPK_ID_BYTES, 0),
            topk_weights: buffer(0x1300_0000, DS4_FLASH_PREFILL_TOPK_WEIGHT_BYTES, 0),
            packed_route_indices: buffer(0x1400_0000, DS4_FLASH_PREFILL_PACKED_ROUTE_BYTES, 0),
            block_expert_ids: buffer(0x1500_0000, DS4_FLASH_PREFILL_BLOCK_EXPERT_BYTES, 0),
            packed_route_count: buffer(0x1600_0000, std::mem::size_of::<i32>(), 0),
            expert_counts: buffer(0x1700_0000, DS4_FLASH_PREFILL_EXPERT_COUNT_BYTES, 0),
            expert_offsets: buffer(0x1800_0000, DS4_FLASH_PREFILL_EXPERT_OFFSET_BYTES, 0),
            fc1: buffer(0x2000_0000, DS4_FLASH_PREFILL_FC1_BYTES, 0),
            activated: buffer(0x2200_0000, DS4_FLASH_PREFILL_ACTIVATED_BYTES, 0),
            rotation_gate: buffer(0x6000_0000, DS4_FLASH_PREFILL_ROUTED_OUTPUT_BYTES, 0),
            rotation_up: buffer(0x7000_0000, DS4_FLASH_PREFILL_ROUTED_OUTPUT_BYTES, 0),
            routed_output: buffer(0x3000_0000, DS4_FLASH_PREFILL_ROUTED_OUTPUT_BYTES, 0),
            output_f32: buffer(0x8000_0000, DS4_FLASH_PREFILL_OUTPUT_F32_BYTES, 0),
            output: buffer(0x4000_0000, DS4_FLASH_PREFILL_HIDDEN_BYTES, 0),
            fc1_scratch: buffer(0x5000_0000, DS4_FLASH_PREFILL_COMPUTE_SCRATCH_BYTES, 0),
            fc2_scratch: buffer(0x5100_0000, DS4_FLASH_PREFILL_COMPUTE_SCRATCH_BYTES, 0),
            locks: buffer(0x5200_0000, DS4_FLASH_DECODE_M1_LOCK_BYTES, 0),
        }
    }

    #[test]
    fn flash_spark_prefill_handoff_accepts_fixed_max_capacity_tp_rank_regions() {
        validate_ds4_flash_spark_prefill_buffer_layout(fixed_buffers(), false).unwrap();
    }

    #[test]
    fn flash_spark_prefill_handoff_rejects_short_alias_or_wrong_device() {
        let short_routed = Ds4FlashSparkPrefillDeviceBuffers {
            routed_output: buffer(0x3000_0000, DS4_FLASH_PREFILL_ROUTED_OUTPUT_BYTES - 1, 0),
            ..fixed_buffers()
        };
        assert!(validate_ds4_flash_spark_prefill_buffer_layout(short_routed, false).is_err());

        let aliased_output = Ds4FlashSparkPrefillDeviceBuffers {
            output: fixed_buffers().input,
            ..fixed_buffers()
        };
        assert!(validate_ds4_flash_spark_prefill_buffer_layout(aliased_output, false).is_err());

        let wrong_device = Ds4FlashSparkPrefillDeviceBuffers {
            fc2_scratch: buffer(0x5100_0000, DS4_FLASH_PREFILL_COMPUTE_SCRATCH_BYTES, 1),
            ..fixed_buffers()
        };
        assert!(validate_ds4_flash_spark_prefill_buffer_layout(wrong_device, false).is_err());
    }

    #[test]
    fn flash_aot_io_alignment_rejects_packed_protocol_v2_small_batch_offsets() {
        let aligned = buffer(0x1000_0000, DS4_FLASH_PREFILL_HIDDEN_BYTES, 0);
        let request_m1 = buffer(0x1000_0000 + 196, DS4_FLASH_PREFILL_HIDDEN_BYTES, 0);
        let response_m1 = buffer(0x1000_0000 + 100, DS4_FLASH_PREFILL_HIDDEN_BYTES, 0);

        assert!(device_buffer_ptr_is_aligned(aligned, 16));
        assert!(!device_buffer_ptr_is_aligned(request_m1, 16));
        assert!(!device_buffer_ptr_is_aligned(response_m1, 16));
    }
}

fn device_buffer_ptr_is_aligned(buffer: Ds41rtDeviceBuffer, alignment: usize) -> bool {
    alignment.is_power_of_two() && !buffer.ptr.is_null() && (buffer.ptr as usize) % alignment == 0
}

pub(in crate::commands::real_full) fn execute_native_flash_route_bf16_cached(
    request_id: u64,
    layer_id: usize,
    rows: usize,
    hidden_bf16: &[u8],
    hidden_device: Option<Ds41rtDeviceBuffer>,
    topk_ids: &[u32],
    topk_weights: &[f32],
    spark_reduction: bool,
    retain_device_output: bool,
    output_device_target: Option<Ds41rtDeviceBuffer>,
    cache: &mut RouteTensorCache,
) -> Result<Ds4FlashRouteExecution> {
    execute_native_flash_route_cached_inner(
        request_id,
        layer_id,
        rows,
        hidden_bf16,
        hidden_device,
        false,
        topk_ids,
        topk_weights,
        spark_reduction,
        retain_device_output,
        output_device_target,
        cache,
    )
}

pub(in crate::commands::real_full) fn execute_native_flash_route_nvfp4_cached(
    request_id: u64,
    layer_id: usize,
    rows: usize,
    hidden_nvfp4: &[u8],
    hidden_device: Option<Ds41rtDeviceBuffer>,
    topk_ids: &[u32],
    topk_weights: &[f32],
    spark_reduction: bool,
    retain_device_output: bool,
    output_device_target: Option<Ds41rtDeviceBuffer>,
    cache: &mut RouteTensorCache,
) -> Result<Ds4FlashRouteExecution> {
    execute_native_flash_route_cached_inner(
        request_id,
        layer_id,
        rows,
        hidden_nvfp4,
        hidden_device,
        true,
        topk_ids,
        topk_weights,
        spark_reduction,
        retain_device_output,
        output_device_target,
        cache,
    )
}

#[allow(clippy::too_many_arguments)]
fn execute_native_flash_route_cached_inner(
    request_id: u64,
    layer_id: usize,
    rows: usize,
    hidden_payload: &[u8],
    hidden_device: Option<Ds41rtDeviceBuffer>,
    prequantized_hidden: bool,
    topk_ids: &[u32],
    topk_weights: &[f32],
    spark_reduction: bool,
    retain_device_output: bool,
    output_device_target: Option<Ds41rtDeviceBuffer>,
    cache: &mut RouteTensorCache,
) -> Result<Ds4FlashRouteExecution> {
    cache.prepare_layer(layer_id);
    let cuda_cache = cache.cuda_cache()?;
    anyhow::ensure!(
        !cuda_cache.legacy_spark_runtime_initialization_enabled
            && (cuda_cache.spark_reduction.is_none()
                || cuda_cache.spark_rdma_reduction.is_none()),
        "native DeepSeek V4 TP=4 execution requires its dedicated CUDA cache with at most one Spark reduction backend"
    );
    cuda_cache.prepare_layer(layer_id);
    let slab = cuda_cache.ds4_flash_expert_slabs.get(&layer_id).cloned();
    let exl3_slab = cuda_cache
        .ds4_flash_exl3_expert_slabs
        .get(&layer_id)
        .cloned();
    let mixed_slab = cuda_cache
        .ds4_flash_exl3_mixed_expert_slabs
        .get(&layer_id)
        .cloned();
    anyhow::ensure!(
        [slab.is_some(), exl3_slab.is_some(), mixed_slab.is_some()]
            .into_iter()
            .filter(|value| *value)
            .count()
            == 1,
        "native DeepSeek V4 layer {layer_id} must have exactly one resident expert representation"
    );
    let mixed_exl3 = mixed_slab.is_some();
    let exl3_profile = exl3_slab.as_ref().map(|slab| slab.profile);
    if let Some(profile) = exl3_profile {
        cuda_cache.ensure_ds4_exl3_aot(profile.variant)?;
    } else {
        cuda_cache.ensure_ds4_flash_aot()?;
    }
    let hidden_size = exl3_profile
        .map(|profile| profile.geometry.hidden_size)
        .unwrap_or(DS4_FLASH_HIDDEN_SIZE);
    let top_k = exl3_profile
        .map(|profile| profile.geometry.top_k)
        .unwrap_or(DS4_FLASH_TOP_K);
    let experts = exl3_profile
        .map(|profile| profile.geometry.experts)
        .unwrap_or(DS4_FLASH_ROUTED_EXPERTS);
    let variant_label = exl3_profile
        .map(Ds4Exl3AotProfile::label)
        .unwrap_or(if mixed_exl3 {
            "Flash mixed K2/K3"
        } else {
            "Flash"
        });
    let capacity_rows = ds4_flash_w4a16_capacity_rows(rows)?;
    let active_hidden_bytes = rows
        .checked_mul(hidden_size)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("native DeepSeek V4 active hidden byte count overflow")?;
    let input_row_stride_bytes = if prequantized_hidden {
        hidden_size / 2 + hidden_size / 16
    } else {
        hidden_size * std::mem::size_of::<u16>()
    };
    let active_input_bytes = rows
        .checked_mul(input_row_stride_bytes)
        .context("native DeepSeek V4 active input byte count overflow")?;
    let capacity_hidden_bytes = capacity_rows
        .checked_mul(hidden_size)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("native DeepSeek V4 capacity hidden byte count overflow")?;
    let route_plan = plan_ds4_packed_topk6_routes(
        rows,
        topk_ids,
        topk_weights,
        experts,
        exl3_profile.map_or(DS4_FLASH_PREFILL_MAX_PACKED_ROUTE_SLOTS, |profile| {
            profile.prefill_max_packed_route_slots
        }),
        exl3_profile.map_or(DS4_FLASH_PREFILL_MAX_ROUTE_BLOCKS, |profile| {
            profile.prefill_max_route_blocks
        }),
        variant_label,
    )?;
    anyhow::ensure!(
        hidden_payload.len() >= active_input_bytes,
        "native Flash hidden payload has {} bytes, needs {active_input_bytes}",
        hidden_payload.len()
    );
    if let Some(hidden_device) = hidden_device {
        anyhow::ensure!(
            !hidden_device.ptr.is_null() && hidden_device.bytes >= active_input_bytes,
            "native Flash device hidden payload is null or has {} bytes, needs {active_input_bytes}",
            hidden_device.bytes
        );
    }
    if let Some(output_device_target) = output_device_target {
        anyhow::ensure!(
            retain_device_output,
            "native Flash device output target requires retained device output"
        );
        anyhow::ensure!(
            !output_device_target.ptr.is_null()
                && output_device_target.bytes >= active_hidden_bytes,
            "native Flash device output target is null or has {} bytes, needs {active_hidden_bytes}",
            output_device_target.bytes
        );
    }

    let exl3 = exl3_profile.is_some() || mixed_exl3;
    let library = Arc::clone(&cuda_cache.library);
    let cuda_stream = cuda_cache.stream.as_ptr();
    let packed_route_capacity = if rows == 1 {
        if mixed_exl3 {
            top_k
        } else {
            exl3_profile.map_or(top_k, |profile| profile.decode_packed_route_slots)
        }
    } else {
        exl3_profile.map_or(DS4_FLASH_PREFILL_MAX_PACKED_ROUTE_SLOTS, |profile| {
            profile.prefill_max_packed_route_slots
        })
    };
    let accumulation = cuda_cache.workspace.ensure_accumulation_buffers(
        Arc::clone(&library),
        capacity_hidden_bytes,
        1,
        capacity_hidden_bytes,
        if exl3 {
            capacity_rows * top_k * std::mem::size_of::<i32>()
        } else {
            1
        },
        capacity_rows * top_k * std::mem::size_of::<f32>(),
        packed_route_capacity * std::mem::size_of::<i32>(),
    )?;
    let aot_workspace = cuda_cache.workspace.ensure_ds4_flash_aot_buffers(
        Arc::clone(&library),
        rows,
        exl3_profile,
        mixed_exl3,
    )?;
    let rdma_row_sharded_reduction = spark_reduction && cuda_cache.spark_rdma_reduction.is_some();
    let nccl_reduction = spark_reduction && cuda_cache.spark_reduction.is_some();
    anyhow::ensure!(
        !spark_reduction || rdma_row_sharded_reduction != nccl_reduction,
        "native Flash request asked for Spark reduction without exactly one TP4 reduction backend"
    );
    let rdma_shape = rdma_row_sharded_reduction
        .then(|| {
            let dtype = reduction_output_dtype(
                cuda_cache
                    .spark_reduction_dtype()
                    .context("native Flash RDMA reduction lost its configured dtype")?,
            );
            let row_bytes = dtype.row_stride_bytes(hidden_size)?;
            let send_bytes = rows
                .checked_mul(row_bytes)
                .context("native Flash RDMA send bytes overflow")?;
            let recv_bytes = send_bytes
                .checked_mul(
                    cuda_cache
                        .spark_reduction_world_size()
                        .context("native Flash RDMA reduction lost its world size")?
                        - 1,
                )
                .context("native Flash RDMA receive bytes overflow")?;
            Ok::<_, anyhow::Error>((dtype, row_bytes, send_bytes, recv_bytes))
        })
        .transpose()?;
    let rdma_completion = rdma_row_sharded_reduction
        .then(|| {
            cuda_cache.workspace.ensure_completion_buffers(
                Arc::clone(&library),
                &[],
                rows.checked_mul(hidden_size)
                    .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
                    .context("native Flash RDMA completion F32 bytes overflow")?,
                active_hidden_bytes,
            )
        })
        .transpose()?;
    let reduction_buffers = if let Some((_, _, send_bytes, recv_bytes)) = rdma_shape {
        Some(cuda_cache.workspace.ensure_reduction_buffers(
            Arc::clone(&library),
            send_bytes,
            recv_bytes,
        )?)
    } else if nccl_reduction {
        Some(cuda_cache.workspace.ensure_reduction_buffers(
            Arc::clone(&library),
            capacity_hidden_bytes,
            capacity_hidden_bytes,
        )?)
    } else {
        None
    };
    // ProtocolV2 packs row and route metadata immediately before the mapped
    // hidden/response payloads. Small batches therefore produce device views
    // such as base+196 and base+100, which are valid BF16 addresses but do not
    // satisfy the vectorized SparkInfer AOT load/store alignment contract.
    // Keep direct I/O for naturally aligned batches and use this lane's
    // preallocated aligned workspace plus D2D copies for the others.
    const AOT_IO_ALIGNMENT_BYTES: usize = 16;
    let direct_output_target = output_device_target.filter(|target| {
        target.bytes >= capacity_hidden_bytes
            && reduction_buffers.is_none()
            && device_buffer_ptr_is_aligned(*target, AOT_IO_ALIGNMENT_BYTES)
    });
    let device_output = if nccl_reduction {
        reduction_buffers
            .expect("native Flash NCCL reduction buffers exist")
            .send
    } else {
        direct_output_target.unwrap_or(accumulation.final_output)
    };
    let pinned_metadata = cuda_cache.workspace.stage_accumulation_metadata_payloads(
        Arc::clone(&library),
        u32_bytes(&route_plan.block_expert_ids),
        f32_bytes(&route_plan.topk_weights),
    )?;
    let direct_hidden_device = (!prequantized_hidden)
        .then_some(hidden_device)
        .flatten()
        .filter(|buffer| {
            buffer.bytes >= capacity_hidden_bytes
                && device_buffer_ptr_is_aligned(*buffer, AOT_IO_ALIGNMENT_BYTES)
        });
    let pinned_hidden = hidden_device
        .is_none()
        .then(|| {
            cuda_cache.workspace.stage_stream_hidden_payload(
                Arc::clone(&library),
                &hidden_payload[..active_input_bytes],
            )
        })
        .transpose()?;
    let route_indices = if exl3 {
        topk_ids
    } else {
        route_plan.packed_route_indices.as_slice()
    };
    let pinned_routes = RouteCudaWorkspace::stage_pinned_bytes(
        &mut cuda_cache.workspace.pinned_route_metadata,
        Arc::clone(&library),
        u32_bytes(route_indices),
        if exl3 {
            "native Flash EXL3 top-k IDs"
        } else {
            "native Flash packed route indices"
        },
    )?;
    let packed_route_count = [route_plan.packed_route_count];
    let pinned_route_count = RouteCudaWorkspace::stage_pinned_bytes(
        &mut cuda_cache.workspace.pinned_completion_indices,
        Arc::clone(&library),
        u32_bytes(&packed_route_count),
        "native Flash packed route count",
    )?;
    let pinned_output = (!retain_device_output)
        .then(|| {
            cuda_cache
                .workspace
                .ensure_output_payload(Arc::clone(&library), active_hidden_bytes)
        })
        .transpose()?;
    let input = direct_hidden_device.unwrap_or(accumulation.hidden);
    let route_indices_device = if exl3 {
        accumulation.scatter_index
    } else {
        accumulation.route_metadata
    };
    let buffers = slab.as_ref().map(|slab| {
        slab.aot_buffers(
            aot_workspace,
            input,
            aot_workspace.routed_output,
            device_output,
            route_indices_device,
            accumulation.route_weights,
        )
    });
    let exl3_buffers = exl3_slab.as_ref().map(|slab| {
        slab.aot_buffers(
            aot_workspace,
            input,
            device_output,
            route_indices_device,
            accumulation.route_weights,
            accumulation.route_metadata,
        )
    });
    let mixed_buffers = mixed_slab.as_ref().map(|slab| {
        slab.aot_buffers(
            aot_workspace,
            input,
            device_output,
            route_indices_device,
            accumulation.route_weights,
            accumulation.route_metadata,
        )
    });
    let mut reduction_follower = false;
    // NCCL communicators share a process-wide launch order across execution
    // lanes. Spark RDMA owns a distinct endpoint set per lane, so globally
    // ordering RDMA requests can make different ranks enter different lanes
    // when host ingress completion order differs. Match the established
    // streamed RDMA path: preserve the request ID on its assigned lane and
    // reserve the global launch gate for NCCL only.
    let mut collective_ticket = if nccl_reduction {
        Some(
            spark_collective_launch_order()
                .register(canonical_spark_collective_request_id(request_id))?,
        )
    } else {
        None
    };
    let mut completed_device_output = None;
    let mut completed_rows = (0..rows).collect::<Vec<_>>();
    let mut completed_output_bytes = active_hidden_bytes;
    let native_cuda_timeline = RouteCudaEventTimeline::enabled()
        .then(|| RouteCudaEventTimeline::new(Arc::clone(&library)))
        .transpose()?;
    unsafe {
        if let Some(timeline) = native_cuda_timeline.as_ref() {
            timeline.record(
                timeline.start.as_ref(),
                cuda_stream,
                "native Flash execution start",
            )?;
        }
        if prequantized_hidden {
            let packed_source = if let Some(hidden_device) = hidden_device {
                hidden_device
            } else {
                library
                    .copy_host_buffer_h2d_async(
                        aot_workspace.packed_hidden,
                        pinned_hidden.expect("packed host input was staged"),
                        active_input_bytes,
                        cuda_stream,
                    )
                    .context("uploading native Flash packed hidden input")?;
                aot_workspace.packed_hidden
            };
            library
                .cuda_b12x_dequantize_nvfp4_rows_bf16_async(
                    packed_source,
                    input_row_stride_bytes,
                    accumulation.hidden,
                    rows,
                    hidden_size,
                    cuda_stream,
                )
                .context("dequantizing native Flash packed hidden input")?;
        } else if let Some(hidden_device) = hidden_device.filter(|_| direct_hidden_device.is_none())
        {
            library
                .copy_d2d_async(
                    accumulation.hidden,
                    hidden_device,
                    active_hidden_bytes,
                    cuda_stream,
                )
                .context("compacting native Flash device hidden input")?;
        } else if let Some(pinned_hidden) = pinned_hidden {
            library
                .copy_host_buffer_h2d_async(
                    accumulation.hidden,
                    pinned_hidden,
                    active_input_bytes,
                    cuda_stream,
                )
                .context("uploading native Flash hidden input")?;
        }
        if let Some(timeline) = native_cuda_timeline.as_ref() {
            timeline.record(
                timeline.hidden_ready.as_ref(),
                cuda_stream,
                "native Flash input ready",
            )?;
        }
        library
            .copy_host_buffer_h2d_async(
                route_indices_device,
                pinned_routes,
                route_indices.len() * std::mem::size_of::<i32>(),
                cuda_stream,
            )
            .context("uploading native Flash route indices")?;
        library
            .copy_host_buffer_h2d_async(
                accumulation.route_weights,
                pinned_metadata.route_weights,
                route_plan.topk_weights.len() * std::mem::size_of::<f32>(),
                cuda_stream,
            )
            .context("uploading native Flash route weights")?;
        if rows != 1 && exl3_buffers.is_none() && mixed_slab.is_none() {
            library
                .copy_host_buffer_h2d_async(
                    aot_workspace.block_expert_ids,
                    pinned_metadata.scatter_index,
                    route_plan.block_expert_ids.len() * std::mem::size_of::<i32>(),
                    cuda_stream,
                )
                .context("uploading native Flash route block experts")?;
            library
                .copy_host_buffer_h2d_async(
                    aot_workspace.packed_route_count,
                    pinned_route_count,
                    std::mem::size_of::<i32>(),
                    cuda_stream,
                )
                .context("uploading native Flash packed route count")?;
        }
        if let Some(timeline) = native_cuda_timeline.as_ref() {
            timeline.record(
                timeline.metadata_ready.as_ref(),
                cuda_stream,
                "native Flash metadata ready",
            )?;
        }
        if rows == 1 {
            if let (Some(slab), Some(buffers)) = (mixed_slab.as_ref(), mixed_buffers.as_ref()) {
                slab.launch_mixed(&library, buffers, rows, cuda_stream)
                    .context("launching native Flash mixed EXL3 K2/K3 M1 TP4 expert partial")?;
            } else if let Some(buffers) = exl3_buffers.as_ref() {
                let profile = exl3_profile.expect("EXL3 slab has a checked profile");
                match (profile.variant, profile.geometry.trellis_bits) {
                    (ModelVariant::Flash, 2) => library
                        .cuda_ds4_flash_spark_exl3_k2_decode_m1_async(buffers, cuda_stream)
                        .context("launching native Flash EXL3 K2 M1 TP4 expert partial")?,
                    (ModelVariant::Flash, 3) => library
                        .cuda_ds4_flash_spark_exl3_k3_decode_m1_async(buffers, cuda_stream)
                        .context("launching native Flash EXL3 K3 M1 TP4 expert partial")?,
                    (ModelVariant::Pro, 2) => library
                        .cuda_ds4_pro_spark_exl3_k2_decode_m1_async(buffers, cuda_stream)
                        .context("launching native Pro EXL3 K2 M1 TP4 expert partial")?,
                    (ModelVariant::Pro, bits) => {
                        anyhow::bail!("unsupported native Pro EXL3 tier K{bits}")
                    }
                    (ModelVariant::Flash, bits) => {
                        anyhow::bail!("unsupported native Flash EXL3 tier K{bits}")
                    }
                    (ModelVariant::Custom, _) => unreachable!("custom EXL3 profile rejected"),
                }
            } else {
                library
                    .cuda_ds4_flash_spark_w4a16_decode_m1_bf16_async(
                        buffers.as_ref().expect("W4A16 Flash slab selected"),
                        cuda_stream,
                    )
                    .context("launching native Flash M1 TP4 expert partial")?;
            }
        } else {
            if let (Some(slab), Some(buffers)) = (mixed_slab.as_ref(), mixed_buffers.as_ref()) {
                slab.launch_mixed(&library, buffers, rows, cuda_stream)
                    .context(
                        "launching native Flash mixed EXL3 K2/K3 prefill TP4 expert partials",
                    )?;
            } else if let Some(buffers) = exl3_buffers.as_ref() {
                let profile = exl3_profile.expect("EXL3 slab has a checked profile");
                match (profile.variant, profile.geometry.trellis_bits) {
                    (ModelVariant::Flash, 2) => library
                        .cuda_ds4_flash_spark_exl3_k2_prefill_topk6_async(
                            buffers,
                            rows,
                            cuda_stream,
                        )
                        .context("launching native Flash EXL3 K2 prefill TP4 expert partials")?,
                    (ModelVariant::Flash, 3) => library
                        .cuda_ds4_flash_spark_exl3_k3_prefill_topk6_async(
                            buffers,
                            rows,
                            cuda_stream,
                        )
                        .context("launching native Flash EXL3 K3 prefill TP4 expert partials")?,
                    (ModelVariant::Pro, 2) => library
                        .cuda_ds4_pro_spark_exl3_k2_prefill_topk6_async(buffers, rows, cuda_stream)
                        .context("launching native Pro EXL3 K2 prefill TP4 expert partials")?,
                    (ModelVariant::Pro, bits) => {
                        anyhow::bail!("unsupported native Pro EXL3 tier K{bits}")
                    }
                    (ModelVariant::Flash, bits) => {
                        anyhow::bail!("unsupported native Flash EXL3 tier K{bits}")
                    }
                    (ModelVariant::Custom, _) => unreachable!("custom EXL3 profile rejected"),
                }
            } else {
                library
                    .cuda_ds4_flash_spark_w4a16_prefill_topk6_bf16_async(
                        buffers.as_ref().expect("W4A16 Flash slab selected"),
                        rows,
                        cuda_stream,
                    )
                    .context("launching native Flash prefill TP4 expert partials")?;
            }
        }
        if let Some(timeline) = native_cuda_timeline.as_ref() {
            timeline.record(
                timeline.routes_ready.as_ref(),
                cuda_stream,
                "native Flash AOT ready",
            )?;
        }

        let completed_source = if rdma_row_sharded_reduction {
            let (reduction_dtype, reduction_row_bytes, _, _) =
                rdma_shape.expect("native Flash RDMA reduction shape exists");
            let (row_start, local_rows) = execute_rdma_row_sharded_reduction(
                cuda_cache,
                &library,
                canonical_spark_collective_request_id(request_id),
                reduction_dtype,
                reduction_row_bytes,
                accumulation,
                rdma_completion.expect("native Flash RDMA completion buffers exist"),
                reduction_buffers.expect("native Flash RDMA reduction buffers exist"),
                Some(device_output),
                rows,
                hidden_size,
                RouteStreamingOutputDtype::Bf16,
                hidden_size * std::mem::size_of::<u16>(),
                cuda_stream,
                None,
            )?;
            completed_rows = (row_start..row_start + local_rows).collect();
            completed_output_bytes = local_rows
                .checked_mul(hidden_size)
                .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
                .context("native Flash RDMA response bytes overflow")?;
            Some(
                rdma_completion
                    .expect("native Flash RDMA completion buffers exist")
                    .output,
            )
        } else if nccl_reduction {
            collective_ticket
                .as_mut()
                .expect("native Flash reduction ticket registered above")
                .wait_for_turn()
                .context("waiting for native Flash deterministic collective order")?;
            let reduction_buffers =
                reduction_buffers.expect("native Flash NCCL reduction buffers exist");
            let reduction = cuda_cache
                .spark_reduction
                .as_ref()
                .expect("native Flash NCCL reduction checked above");
            reduction
                .communicator()
                .reduce_bf16_async(
                    reduction_buffers.send,
                    reduction_buffers.recv,
                    rows * hidden_size,
                    reduction.root_rank,
                    cuda_stream,
                )
                .context("reducing native Flash TP4 expert partials")?;
            if reduction.is_root() {
                Some(reduction_buffers.recv)
            } else {
                reduction_follower = true;
                completed_rows.clear();
                completed_output_bytes = 0;
                None
            }
        } else {
            Some(device_output)
        };
        if let Some(timeline) = native_cuda_timeline.as_ref() {
            timeline.record(
                timeline.pack_ready.as_ref(),
                cuda_stream,
                "native Flash reduction ready",
            )?;
        }
        if let Some(completed_source) = completed_source {
            if retain_device_output {
                let retained = if let Some(target) = output_device_target {
                    if target.ptr != completed_source.ptr {
                        library
                            .copy_d2d_async(
                                target,
                                completed_source,
                                completed_output_bytes,
                                cuda_stream,
                            )
                            .context("copying native Flash BF16 output to transport slot")?;
                    }
                    route_device_buffer_slice(target, 0, completed_output_bytes)?
                } else {
                    route_device_buffer_slice(completed_source, 0, completed_output_bytes)?
                };
                completed_device_output = Some(retained);
            } else {
                library
                    .copy_d2h_host_buffer_async(
                        pinned_output.expect("host output buffer ensured above"),
                        completed_source,
                        completed_output_bytes,
                        cuda_stream,
                    )
                    .context("copying native Flash BF16 output")?;
            }
        }
        if let Some(timeline) = native_cuda_timeline.as_ref() {
            timeline.record(
                timeline.retained_ready.as_ref(),
                cuda_stream,
                "native Flash retained output ready",
            )?;
            timeline.record(
                timeline.host_copy_ready.as_ref(),
                cuda_stream,
                "native Flash execution complete",
            )?;
        }
        library
            .cuda_stream_synchronize(cuda_stream)
            .context("synchronizing native Flash execution")?;
    }
    if let Some(timeline) = native_cuda_timeline.as_ref() {
        let elapsed = unsafe { timeline.elapsed()? };
        eprintln!(
            "real_native_flash_cuda_event_timing layer_id={} rows={} spark_reduction={} rdma_reduction={} exl3={} input_ms={:.3} metadata_ms={:.3} aot_ms={:.3} post_aot_exchange_combine_ms={:.3} retained_ms={:.3} finalize_ms={:.3} total_ms={:.3}",
            layer_id,
            rows,
            spark_reduction,
            rdma_row_sharded_reduction,
            exl3,
            elapsed.hidden_copy_ms,
            elapsed.metadata_copy_ms,
            elapsed.route_kernel_ms,
            elapsed.bf16_pack_ms,
            elapsed.retained_copy_ms,
            elapsed.host_copy_ms,
            elapsed.total_ms,
        );
    }
    if let Some(ticket) = collective_ticket.as_mut() {
        ticket
            .finish()
            .context("advancing native Flash deterministic collective order")?;
    }
    let output_bf16 = if reduction_follower || retain_device_output {
        Vec::new()
    } else {
        cuda_cache
            .workspace
            .output_payload_slice(completed_output_bytes)?
            .to_vec()
    };
    Ok(Ds4FlashRouteExecution {
        output_bf16,
        output_device: completed_device_output,
        completed_rows,
        reduction_follower,
        kernel_backend: exl3_profile
            .map(Ds4Exl3AotProfile::backend)
            .unwrap_or(DS4_FLASH_SPARK_AOT_BF16_BACKEND),
    })
}

#[derive(Clone)]
struct Bf16RouteProjections {
    gate: Bf16RouteProjection,
    up: Bf16RouteProjection,
    down: Bf16RouteProjection,
    gate_scale_2: f32,
    up_scale_2: f32,
    down_scale_2: f32,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct Bf16RouteProjectionGroupKey {
    expert_id: usize,
    intermediate_rows: usize,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct Bf16RouteProjectionGroupCacheKey {
    layer_id: usize,
    expert_id: usize,
    intermediate_rows: usize,
    output_rows: usize,
    hidden_dim: usize,
    require_host_tensors: bool,
}

struct LoadedBf16Route {
    row_index: usize,
    route: ScoredRoute,
    intermediate_rows: usize,
    projections: Bf16RouteProjections,
}

struct LoadedBf16RouteGroup {
    start: usize,
    intermediate_rows: usize,
    count: usize,
    projections: Bf16RouteProjections,
    completed_rows: Vec<usize>,
}

struct PackedW4a16Topk8PrefillPlan {
    packed_route_indices: Vec<u32>,
    block_expert_ids: Vec<u32>,
    packed_route_count: u32,
    direct_topk_ids: Vec<u32>,
    topk_weights: Vec<f32>,
}

#[derive(Clone, Copy, Debug)]
pub(in crate::commands::real_full) struct PackedW4a16Topk8Route {
    pub(in crate::commands::real_full) expert_id: u32,
    pub(in crate::commands::real_full) weight: f32,
}

fn packed_w4a16_topk8_prefill_eligible(row_count: usize, route_count: usize) -> bool {
    row_count > 0
        && row_count <= B12X_W4A16_PREFILL_TOPK8_CAPACITY_ROWS
        && route_count == row_count * B12X_W4A16_PREFILL_TOPK8_ROUTES
}

fn plan_packed_w4a16_topk8_prefill(
    row_routes: &[Vec<(ScoredRoute, usize)>],
) -> Result<PackedW4a16Topk8PrefillPlan> {
    anyhow::ensure!(
        !row_routes.is_empty()
            && row_routes.len() <= B12X_W4A16_PREFILL_TOPK8_CAPACITY_ROWS,
        "packed W4A16 top-k=8 prefill rows {} are outside 1..={B12X_W4A16_PREFILL_TOPK8_CAPACITY_ROWS}",
        row_routes.len()
    );
    let mut flat_routes = Vec::with_capacity(
        row_routes
            .len()
            .checked_mul(B12X_W4A16_PREFILL_TOPK8_ROUTES)
            .context("packed W4A16 prefill route count overflow")?,
    );
    for (row_index, routes) in row_routes.iter().enumerate() {
        anyhow::ensure!(
            routes.len() == B12X_W4A16_PREFILL_TOPK8_ROUTES,
            "packed W4A16 prefill row {row_index} has {} routes, expected {}",
            routes.len(),
            B12X_W4A16_PREFILL_TOPK8_ROUTES
        );
        for (route_slot, (route, intermediate_rows)) in routes.iter().enumerate() {
            anyhow::ensure!(
                *intermediate_rows == 512 && route.expert_id < B12X_W4A16_EXPERTS,
                "packed W4A16 prefill route row={row_index} slot={route_slot} has expert={} intermediate={intermediate_rows}",
                route.expert_id
            );
            flat_routes.push(PackedW4a16Topk8Route {
                expert_id: u32::try_from(route.expert_id)
                    .context("packed W4A16 prefill expert ID exceeds u32")?,
                weight: route.normalized_weight,
            });
        }
    }

    plan_packed_w4a16_topk8_prefill_flat(row_routes.len(), &flat_routes)
}

fn plan_packed_w4a16_topk8_prefill_flat(
    row_count: usize,
    routes: &[PackedW4a16Topk8Route],
) -> Result<PackedW4a16Topk8PrefillPlan> {
    anyhow::ensure!(
        row_count > 0 && row_count <= B12X_W4A16_PREFILL_TOPK8_CAPACITY_ROWS,
        "packed W4A16 top-k=8 prefill rows {row_count} are outside 1..={B12X_W4A16_PREFILL_TOPK8_CAPACITY_ROWS}"
    );
    let route_count = row_count
        .checked_mul(B12X_W4A16_PREFILL_TOPK8_ROUTES)
        .context("packed W4A16 prefill route count overflow")?;
    anyhow::ensure!(
        routes.len() == route_count,
        "packed W4A16 prefill routes {} did not match {row_count} * {B12X_W4A16_PREFILL_TOPK8_ROUTES}",
        routes.len()
    );
    let route_count_u32 =
        u32::try_from(route_count).context("packed W4A16 prefill route count exceeds u32")?;
    let direct_topk_ids = routes
        .iter()
        .map(|route| {
            anyhow::ensure!(
                route.expert_id < B12X_W4A16_EXPERTS as u32,
                "direct W4A16 prefill expert {} exceeds {}",
                route.expert_id,
                B12X_W4A16_EXPERTS
            );
            Ok(route.expert_id)
        })
        .collect::<Result<Vec<_>>>()?;
    // M=1 consumes the direct top-k expert IDs. Exact M=2..8 execution is
    // expert-packed in block-8 groups so each route retains the M=1 arithmetic
    // shape while repeated experts reuse their resident weights.
    let direct_topk = row_count == 1;
    let topk_weights = routes.iter().map(|route| route.weight).collect::<Vec<_>>();
    if direct_topk {
        return Ok(PackedW4a16Topk8PrefillPlan {
            packed_route_indices: direct_topk_ids.clone(),
            block_expert_ids: Vec::new(),
            packed_route_count: route_count_u32,
            direct_topk_ids,
            topk_weights,
        });
    }

    let route_block_rows = b12x_w4a16_prefill_route_block_rows(row_count);
    let mut expert_counts = [0_usize; B12X_W4A16_EXPERTS];
    for route in routes {
        let expert_id = route.expert_id as usize;
        anyhow::ensure!(
            expert_id < B12X_W4A16_EXPERTS,
            "packed W4A16 prefill expert {expert_id} exceeds {B12X_W4A16_EXPERTS}"
        );
        expert_counts[expert_id] += 1;
    }

    let mut expert_offsets = [0_usize; B12X_W4A16_EXPERTS];
    let mut block_expert_ids = Vec::new();
    let mut packed_route_slots = 0_usize;
    for (expert_id, count) in expert_counts.iter().copied().enumerate() {
        expert_offsets[expert_id] = packed_route_slots;
        if count == 0 {
            continue;
        }
        let blocks = count.div_ceil(route_block_rows);
        packed_route_slots = packed_route_slots
            .checked_add(blocks * route_block_rows)
            .context("packed W4A16 route slot count overflow")?;
        block_expert_ids.extend(std::iter::repeat_n(
            u32::try_from(expert_id).context("packed W4A16 expert ID exceeds u32")?,
            blocks,
        ));
    }
    anyhow::ensure!(
        packed_route_slots <= B12X_W4A16_MAX_PACKED_ROUTE_SLOTS
            && block_expert_ids.len() <= B12X_W4A16_MAX_ROUTE_BLOCKS,
        "packed W4A16 prefill metadata exceeds AOT capacity: routes={} blocks={} max_routes={} max_blocks={}",
        packed_route_slots,
        block_expert_ids.len(),
        B12X_W4A16_MAX_PACKED_ROUTE_SLOTS,
        B12X_W4A16_MAX_ROUTE_BLOCKS
    );
    let mut packed_route_indices = vec![route_count_u32; packed_route_slots];
    let mut expert_cursors = expert_offsets;
    for (route_index, route) in routes.iter().enumerate() {
        let expert_id = route.expert_id as usize;
        let destination = expert_cursors[expert_id];
        packed_route_indices[destination] =
            u32::try_from(route_index).context("packed W4A16 route index exceeds u32")?;
        expert_cursors[expert_id] += 1;
    }

    let packed_route_count = u32::try_from(packed_route_indices.len())
        .context("packed W4A16 padded route count exceeds u32")?;
    Ok(PackedW4a16Topk8PrefillPlan {
        packed_route_indices,
        block_expert_ids,
        packed_route_count,
        direct_topk_ids,
        topk_weights,
    })
}

pub(in crate::commands::real_full) struct RouteNvfp4IngressStream {
    layer_id: usize,
    row_count: usize,
    route_count: usize,
    hidden_dim: usize,
    hidden_row_stride_bytes: usize,
    output_rows: usize,
    output_dtype: RouteStreamingOutputDtype,
    spark_reduction: bool,
    spark_row_sharded_reduction: bool,
    route_groups: Vec<LoadedBf16RouteGroup>,
    group_ready_after_rows: Vec<usize>,
    max_intermediate_rows: usize,
    max_group_rows: usize,
    lane_count: usize,
    packed_w4a16_topk8_prefill: bool,
    m1_parity_grouped_small_m_w4a16: bool,
    split_m1_m2_w4a16: bool,
    w4a16_small_m_mode: W4a16SmallMMode,
    topk8_combined_output_in_group_buffer: bool,
    packed_w4a16_direct_fp8_output: bool,
    grouped_decode: bool,
    collective_request_id: Option<u64>,
    collective_launch_ticket: Option<SparkCollectiveLaunchTicket>,
    lane_used_since_emit: Vec<bool>,
    scheduled_group_count: usize,
    next_group: usize,
    received_rows: usize,
    emitted_rows: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum W4a16SmallMMode {
    Ordered,
    WideOrdered,
    SplitM1,
}

impl RouteNvfp4IngressStream {
    pub(in crate::commands::real_full) fn consumes_request_order(&self) -> bool {
        self.packed_w4a16_topk8_prefill
    }

    pub(in crate::commands::real_full) fn register_collective_request(
        &mut self,
        request_id: u64,
        cache: &RouteTensorCache,
    ) -> Result<()> {
        anyhow::ensure!(
            self.spark_reduction,
            "non-reduced route stream cannot register a collective request"
        );
        anyhow::ensure!(
            self.collective_request_id.is_none() && self.collective_launch_ticket.is_none(),
            "route stream already registered a collective request"
        );
        let cuda_cache = cache
            .cuda
            .as_ref()
            .context("Spark-reduced route stream lost its CUDA cache")?;
        let nccl = cuda_cache.spark_reduction.is_some();
        let rdma = cuda_cache.spark_rdma_reduction.is_some();
        anyhow::ensure!(
            nccl ^ rdma,
            "Spark-reduced route stream requires exactly one reduction backend"
        );
        let request_id = canonical_spark_collective_request_id(request_id);
        self.collective_request_id = Some(request_id);
        if nccl {
            self.collective_launch_ticket =
                Some(spark_collective_launch_order().register(request_id)?);
        }
        Ok(())
    }
}

pub(in crate::commands::real_full) struct RouteNvfp4IngressStreamChunk {
    pub(in crate::commands::real_full) completed_rows: Vec<usize>,
    pub(in crate::commands::real_full) output: Vec<u8>,
    pub(in crate::commands::real_full) device_output: Option<Ds41rtDeviceBuffer>,
    pub(in crate::commands::real_full) reduction_follower: bool,
    pub(in crate::commands::real_full) complete: bool,
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn try_begin_packed_w4a16_topk8_prefill_cached(
    layer_id: usize,
    hidden_dim: usize,
    hidden_row_stride_bytes: usize,
    row_count: usize,
    routes: &[PackedW4a16Topk8Route],
    output_rows: usize,
    output_dtype: RouteStreamingOutputDtype,
    spark_reduction: bool,
    spark_row_sharded_reduction: bool,
    cache: &mut RouteTensorCache,
) -> Result<Option<RouteNvfp4IngressStream>> {
    let route_count = routes.len();
    if row_count == 0
        || row_count > B12X_W4A16_PREFILL_TOPK8_CAPACITY_ROWS
        || route_count != row_count * B12X_W4A16_PREFILL_TOPK8_ROUTES
    {
        return Ok(None);
    }
    anyhow::ensure!(
        cuda_reference_kernels_enabled() && !cuda_route_validation_enabled(),
        "packed W4A16 prefill ingress requires non-validating CUDA execution"
    );
    anyhow::ensure!(
        hidden_dim > 0 && hidden_dim % 16 == 0,
        "packed W4A16 prefill hidden width must be a positive multiple of 16"
    );
    let logical_hidden_row_bytes = hidden_dim / 2 + hidden_dim / 16;
    anyhow::ensure!(
        hidden_row_stride_bytes >= logical_hidden_row_bytes,
        "packed W4A16 prefill hidden stride {hidden_row_stride_bytes} is smaller than {logical_hidden_row_bytes}"
    );
    anyhow::ensure!(
        !spark_row_sharded_reduction || spark_reduction,
        "row-sharded Spark reduction requires Spark reduction"
    );

    cache.prepare_layer(layer_id);
    let cuda_cache = cache.cuda_cache()?;
    cuda_cache.prepare_layer(layer_id);
    let slab = cuda_cache.expert_slabs.get(&layer_id);
    let packed_w4a16 = cuda_cache.b12x_w4a16_packed
        && slab
            .map(|slab| slab.w4a16_moe_buffers().is_ok())
            .unwrap_or(false);
    if !packed_w4a16 {
        return Ok(None);
    }
    anyhow::ensure!(
        cuda_cache.b12x_aot_enabled,
        "packed W4A16 prefill ingress requires the B12X AOT backend"
    );
    if spark_reduction {
        let reduction_dtype = cuda_cache
            .spark_reduction_dtype()
            .context("Spark-reduced request reached an expert without a reduction backend")?;
        anyhow::ensure!(
            cuda_cache.spark_reduction_enabled_for_rows(row_count),
            "Spark-reduced request rows {row_count} are below the configured minimum"
        );
        anyhow::ensure!(
            !spark_row_sharded_reduction
                || reduction_dtype == ExpertIntermediateReductionDtype::Fp8
                || cuda_cache.spark_rdma_reduction_enabled(),
            "NCCL row-sharded reduction currently requires FP8"
        );
    }

    let packed_plan = plan_packed_w4a16_topk8_prefill_flat(row_count, routes)?;
    let m1_parity_grouped_small_m_w4a16 = packed_w4a16 && (2..=8).contains(&row_count);
    let w4a16_small_m_mode = b12x_spark_w4a16_small_m_mode();
    let split_m1_m2_w4a16 = m1_parity_grouped_small_m_w4a16
        && row_count == 2
        && w4a16_small_m_mode == W4a16SmallMMode::SplitM1;
    let lane_count = if split_m1_m2_w4a16 { 2 } else { 1 };
    anyhow::ensure!(
        lane_count <= cuda_cache.b12x_lane_count(),
        "split-M1 physical M=2 requires two B12X route lanes, configured {}",
        cuda_cache.b12x_lane_count()
    );
    let topk8_combined_output_in_group_buffer = false;
    let packed_w4a16_row_sharded_fp8_direct = spark_row_sharded_reduction
        && output_dtype == RouteStreamingOutputDtype::Fp8E4m3RowScaled
        && fused_fp8_reduction_enabled()
        && cuda_cache.spark_reduction_dtype() == Some(ExpertIntermediateReductionDtype::Fp8);
    let packed_w4a16_direct_fp8_output = output_dtype
        == RouteStreamingOutputDtype::Fp8E4m3RowScaled
        && (packed_w4a16_row_sharded_fp8_direct || !spark_reduction);
    if route_stage_timing_enabled() {
        eprintln!(
            "real_nvfp4_route_topk8_prefill_fast_selected layer_id={layer_id} rows={row_count} routes={route_count} layout=w4a16-packed"
        );
    }

    let hidden_bytes = row_count
        .checked_mul(hidden_row_stride_bytes)
        .context("packed W4A16 prefill hidden byte count overflow")?;
    let accumulator_bytes = row_count
        .checked_mul(output_rows)
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .context("packed W4A16 prefill accumulator byte count overflow")?;
    let index_bytes = route_count
        .checked_mul(std::mem::size_of::<u32>())
        .context("packed W4A16 prefill index byte count overflow")?;
    let weight_bytes = route_count
        .checked_mul(std::mem::size_of::<f32>())
        .context("packed W4A16 prefill weight byte count overflow")?;
    let library = Arc::clone(&cuda_cache.library);
    let cuda_stream = cuda_cache.stream.as_ptr();
    let completion_stream = cuda_cache.completion_stream.as_ptr();
    let metadata_ready_event = cuda_cache.metadata_ready_event.as_ptr();
    unsafe {
        library
            .cuda_stream_synchronize(cuda_stream)
            .context("synchronizing packed W4A16 route setup")?;
    }
    let workspace = cuda_cache.workspace.ensure_accumulation_buffers(
        Arc::clone(&library),
        hidden_bytes,
        accumulator_bytes,
        1,
        index_bytes,
        weight_bytes,
        index_bytes,
    )?;
    let b12x_workspace_rows = if split_m1_m2_w4a16 { 1 } else { row_count };
    let b12x_workspaces = cuda_cache.ensure_b12x_route_workspaces(
        lane_count,
        b12x_workspace_rows,
        hidden_dim,
        512,
        output_rows,
    )?;
    let scatter_indices = (0..row_count)
        .map(|row| u32::try_from(row).context("packed prefill row index exceeds u32"))
        .collect::<Result<Vec<_>>>()?;
    let mut packed_metadata = if split_m1_m2_w4a16 {
        packed_plan.direct_topk_ids.clone()
    } else {
        packed_plan.packed_route_indices.clone()
    };
    let block_expert_offset = packed_metadata.len() * std::mem::size_of::<u32>();
    packed_metadata.extend_from_slice(&packed_plan.block_expert_ids);
    let route_count_offset = packed_metadata.len() * std::mem::size_of::<u32>();
    packed_metadata.push(packed_plan.packed_route_count);
    let metadata_payloads = cuda_cache.workspace.stage_accumulation_metadata_payloads(
        Arc::clone(&library),
        u32_bytes(&scatter_indices),
        f32_bytes(&packed_plan.topk_weights),
    )?;
    let input_index_payload = cuda_cache
        .workspace
        .stage_stream_input_indices(Arc::clone(&library), &packed_metadata)?;
    unsafe {
        library
            .copy_host_buffer_h2d_async(
                workspace.scatter_index,
                metadata_payloads.scatter_index,
                scatter_indices.len() * std::mem::size_of::<u32>(),
                completion_stream,
            )
            .context("copying packed W4A16 prefill output indices")?;
        let b12x_workspace = b12x_workspaces[0];
        if split_m1_m2_w4a16 {
            let topk_id_row_bytes = B12X_W4A16_PREFILL_TOPK8_ROUTES * std::mem::size_of::<u32>();
            let topk_weight_row_bytes =
                B12X_W4A16_PREFILL_TOPK8_ROUTES * std::mem::size_of::<f32>();
            for (lane, lane_workspace) in b12x_workspaces.iter().copied().enumerate() {
                library
                    .copy_host_buffer_h2d_async(
                        lane_workspace.w4a16_packed_route_indices,
                        host_buffer_byte_view(
                            input_index_payload,
                            lane * topk_id_row_bytes,
                            topk_id_row_bytes,
                            "split-M1 W4A16 expert IDs",
                        )?,
                        topk_id_row_bytes,
                        completion_stream,
                    )
                    .context("copying split-M1 W4A16 expert IDs")?;
                library
                    .copy_host_buffer_h2d_async(
                        lane_workspace.w4a16_topk_weights,
                        host_buffer_byte_view(
                            metadata_payloads.route_weights,
                            lane * topk_weight_row_bytes,
                            topk_weight_row_bytes,
                            "split-M1 W4A16 route weights",
                        )?,
                        topk_weight_row_bytes,
                        completion_stream,
                    )
                    .context("copying split-M1 W4A16 route weights")?;
            }
        } else {
            library
                .copy_host_buffer_h2d_async(
                    b12x_workspace.w4a16_packed_route_indices,
                    input_index_payload,
                    block_expert_offset,
                    completion_stream,
                )
                .context("copying packed W4A16 prefill route indices")?;
        }
        let block_expert_bytes = route_count_offset - block_expert_offset;
        if block_expert_bytes > 0 {
            library
                .copy_host_buffer_h2d_async(
                    b12x_workspace.w4a16_block_expert_ids,
                    host_buffer_byte_view(
                        input_index_payload,
                        block_expert_offset,
                        block_expert_bytes,
                        "packed W4A16 prefill block expert IDs",
                    )?,
                    block_expert_bytes,
                    completion_stream,
                )
                .context("copying packed W4A16 prefill block expert IDs")?;
        }
        library
            .copy_host_buffer_h2d_async(
                b12x_workspace.w4a16_packed_route_count,
                host_buffer_byte_view(
                    input_index_payload,
                    route_count_offset,
                    std::mem::size_of::<u32>(),
                    "packed W4A16 prefill route count",
                )?,
                std::mem::size_of::<u32>(),
                completion_stream,
            )
            .context("copying packed W4A16 prefill route count")?;
        if !split_m1_m2_w4a16 {
            library
                .copy_host_buffer_h2d_async(
                    b12x_workspace.w4a16_topk_weights,
                    metadata_payloads.route_weights,
                    packed_plan.topk_weights.len() * std::mem::size_of::<f32>(),
                    completion_stream,
                )
                .context("copying packed W4A16 prefill route weights")?;
        }
        if !packed_w4a16_direct_fp8_output {
            library
                .cuda_zero_f32_async(
                    workspace.accumulator,
                    row_count * output_rows,
                    completion_stream,
                )
                .context("zeroing packed W4A16 route accumulator")?;
        }
        library
            .cuda_event_record(metadata_ready_event, completion_stream)
            .context("recording packed W4A16 route metadata readiness")?;
        library
            .cuda_stream_wait_event(cuda_stream, metadata_ready_event)
            .context("waiting for packed W4A16 route metadata")?;
        for lane in 1..lane_count {
            library
                .cuda_stream_wait_event(cuda_cache.b12x_lane_stream(lane), metadata_ready_event)
                .context("waiting for split-M1 W4A16 route metadata")?;
        }
    }

    Ok(Some(RouteNvfp4IngressStream {
        layer_id,
        row_count,
        route_count,
        hidden_dim,
        hidden_row_stride_bytes,
        output_rows,
        output_dtype,
        spark_reduction,
        spark_row_sharded_reduction,
        route_groups: Vec::new(),
        group_ready_after_rows: Vec::new(),
        max_intermediate_rows: 512,
        max_group_rows: b12x_workspace_rows,
        lane_count,
        packed_w4a16_topk8_prefill: true,
        m1_parity_grouped_small_m_w4a16,
        split_m1_m2_w4a16,
        w4a16_small_m_mode,
        topk8_combined_output_in_group_buffer,
        packed_w4a16_direct_fp8_output,
        grouped_decode: false,
        collective_request_id: None,
        collective_launch_ticket: None,
        lane_used_since_emit: vec![false; lane_count],
        scheduled_group_count: 1,
        next_group: 0,
        received_rows: 0,
        emitted_rows: 0,
    }))
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn begin_nvfp4_route_ingress_stream_cached(
    catalog: &TensorCatalog,
    layer_id: usize,
    hidden_dim: usize,
    hidden_row_stride_bytes: usize,
    row_routes: &[Vec<(ScoredRoute, usize)>],
    output_rows: usize,
    output_dtype: RouteStreamingOutputDtype,
    spark_reduction: bool,
    spark_row_sharded_reduction: bool,
    plan: &ExpertProtocolV2StreamPlan,
    cache: &mut RouteTensorCache,
) -> Result<RouteNvfp4IngressStream> {
    anyhow::ensure!(
        cuda_reference_kernels_enabled() && !cuda_route_validation_enabled(),
        "streamed NVFP4 route ingress requires non-validating CUDA execution"
    );
    anyhow::ensure!(
        hidden_dim > 0 && hidden_dim % 16 == 0,
        "streamed NVFP4 route hidden width must be a positive multiple of 16"
    );
    let logical_hidden_row_bytes = hidden_dim / 2 + hidden_dim / 16;
    anyhow::ensure!(
        hidden_row_stride_bytes >= logical_hidden_row_bytes,
        "streamed NVFP4 route hidden stride {hidden_row_stride_bytes} is smaller than {logical_hidden_row_bytes}"
    );
    let row_count = row_routes.len();
    let route_count = row_routes.iter().map(Vec::len).sum::<usize>();
    anyhow::ensure!(
        plan.row_count as usize == row_count && plan.route_count as usize == route_count,
        "streamed NVFP4 route plan shape rows={} routes={} did not match rows={row_count} routes={route_count}",
        plan.row_count,
        plan.route_count
    );
    cache.prepare_layer(layer_id);

    let mut loaded_routes = Vec::with_capacity(route_count);
    let mut loaded_projection_groups = HashMap::new();
    for (row_index, routes) in row_routes.iter().enumerate() {
        for (route, intermediate_rows) in routes {
            let projections = load_bf16_route_projections_for_group_cached(
                catalog,
                layer_id,
                route,
                *intermediate_rows,
                output_rows,
                hidden_dim,
                cache,
                &mut loaded_projection_groups,
            )?;
            loaded_routes.push(LoadedBf16Route {
                row_index,
                route: route.clone(),
                intermediate_rows: *intermediate_rows,
                projections,
            });
        }
    }
    let mut activation_position = vec![0_usize; row_count];
    for (position, row_index) in plan.activation_row_order.iter().enumerate() {
        activation_position[*row_index as usize] = position;
    }
    let mut input_indices = Vec::with_capacity(route_count);
    let mut output_indices = Vec::with_capacity(route_count);
    let mut route_weights = Vec::with_capacity(route_count);
    let mut route_groups = Vec::with_capacity(plan.groups.len());
    for (group_index, planned_group) in plan.groups.iter().enumerate() {
        let start = input_indices.len();
        let first_route_index = *planned_group
            .route_indices
            .first()
            .with_context(|| format!("streamed NVFP4 route group {group_index} is empty"))?
            as usize;
        let first = loaded_routes.get(first_route_index).with_context(|| {
            format!("streamed NVFP4 route group {group_index} first route is out of range")
        })?;
        for route_index in &planned_group.route_indices {
            let loaded = loaded_routes.get(*route_index as usize).with_context(|| {
                format!("streamed NVFP4 route group {group_index} route is out of range")
            })?;
            anyhow::ensure!(
                loaded.route.expert_id == first.route.expert_id
                    && loaded.intermediate_rows == first.intermediate_rows,
                "streamed NVFP4 route group {group_index} mixes expert projection shapes"
            );
            input_indices.push(
                u32::try_from(activation_position[loaded.row_index])
                    .context("streamed NVFP4 route input row exceeds u32")?,
            );
            output_indices.push(
                u32::try_from(loaded.row_index)
                    .context("streamed NVFP4 route output row exceeds u32")?,
            );
            route_weights.push(loaded.route.normalized_weight);
        }
        route_groups.push(LoadedBf16RouteGroup {
            start,
            intermediate_rows: first.intermediate_rows,
            count: input_indices.len() - start,
            projections: first.projections.clone(),
            completed_rows: planned_group
                .completed_rows
                .iter()
                .map(|row| *row as usize)
                .collect(),
        });
    }
    anyhow::ensure!(
        input_indices.len() == route_count
            && output_indices.len() == route_count
            && route_weights.len() == route_count,
        "streamed NVFP4 route metadata did not cover every route"
    );
    let max_intermediate_rows = route_groups
        .iter()
        .map(|group| group.intermediate_rows)
        .max()
        .unwrap_or(0);
    let max_group_rows = route_groups
        .iter()
        .map(|group| group.count)
        .max()
        .unwrap_or(0);
    anyhow::ensure!(
        max_group_rows > 0 && max_group_rows <= B12X_SPARK_AOT_MAX_ROWS,
        "streamed NVFP4 route group rows {max_group_rows} are outside B12X range"
    );

    let hidden_bytes = row_count
        .checked_mul(hidden_row_stride_bytes)
        .context("streamed NVFP4 route hidden byte count overflow")?;
    let accumulator_bytes = row_count
        .checked_mul(output_rows)
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .context("streamed NVFP4 route accumulator byte count overflow")?;
    let index_bytes = route_count
        .checked_mul(std::mem::size_of::<u32>())
        .context("streamed NVFP4 route index byte count overflow")?;
    let weight_bytes = route_count
        .checked_mul(std::mem::size_of::<f32>())
        .context("streamed NVFP4 route weight byte count overflow")?;
    let cuda_cache = cache.cuda_cache()?;
    cuda_cache.prepare_layer(layer_id);
    if spark_reduction {
        let reduction_dtype = cuda_cache
            .spark_reduction_dtype()
            .context("Spark-reduced request reached an expert without a reduction backend")?;
        anyhow::ensure!(
            cuda_cache.spark_reduction_enabled_for_rows(row_count),
            "Spark-reduced request rows {row_count} are below the configured minimum"
        );
        anyhow::ensure!(
            !spark_row_sharded_reduction
                || reduction_dtype == ExpertIntermediateReductionDtype::Fp8
                || cuda_cache.spark_rdma_reduction_enabled(),
            "NCCL row-sharded reduction currently requires FP8"
        );
    }
    anyhow::ensure!(
        !spark_row_sharded_reduction || spark_reduction,
        "row-sharded Spark reduction requires Spark reduction"
    );
    anyhow::ensure!(
        cuda_cache.b12x_aot_enabled,
        "streamed NVFP4 route ingress requires the B12X AOT backend"
    );
    let packed_w4a16 = cuda_cache.b12x_w4a16_packed
        && cuda_cache
            .expert_slabs
            .get(&layer_id)
            .map(|slab| slab.w4a16_moe_buffers().is_ok())
            .unwrap_or(false);
    anyhow::ensure!(
        packed_w4a16,
        "streamed NVFP4 route ingress requires a packed W4A16 expert slab"
    );
    let packed_w4a16_topk8_prefill_plan =
        (packed_w4a16_topk8_prefill_eligible(row_count, route_count))
            .then(|| plan_packed_w4a16_topk8_prefill(row_routes))
            .transpose()?;
    let packed_w4a16_topk8_prefill = packed_w4a16_topk8_prefill_plan.is_some();
    let m1_parity_grouped_small_m_w4a16 =
        packed_w4a16 && packed_w4a16_topk8_prefill && (2..=8).contains(&row_count);
    let w4a16_small_m_mode = b12x_spark_w4a16_small_m_mode();
    let split_m1_m2_w4a16 = m1_parity_grouped_small_m_w4a16
        && row_count == 2
        && w4a16_small_m_mode == W4a16SmallMMode::SplitM1;
    let topk8_combined_output_in_group_buffer = false;
    let packed_w4a16_row_sharded_fp8_direct = packed_w4a16_topk8_prefill
        && spark_row_sharded_reduction
        && output_dtype == RouteStreamingOutputDtype::Fp8E4m3RowScaled
        && fused_fp8_reduction_enabled()
        && cuda_cache.spark_reduction_dtype() == Some(ExpertIntermediateReductionDtype::Fp8);
    let packed_w4a16_direct_fp8_output = packed_w4a16_topk8_prefill
        && output_dtype == RouteStreamingOutputDtype::Fp8E4m3RowScaled
        && (packed_w4a16_row_sharded_fp8_direct || !spark_reduction);
    if packed_w4a16_topk8_prefill && route_stage_timing_enabled() {
        eprintln!(
            "real_nvfp4_route_packed_w4a16_prefill_selected layer_id={layer_id} rows={row_count} routes={route_count}"
        );
    }
    let grouped_decode_shape = b12x_spark_grouped_decode_enabled()
        && row_count == 1
        && route_count == 8
        && route_groups.len() == 8
        && route_groups
            .iter()
            .all(|group| group.count == 1 && group.intermediate_rows == 512);
    let w4a16_decode = grouped_decode_shape
        && cuda_cache
            .expert_slabs
            .get(&layer_id)
            .map(|slab| slab.w4a16_moe_buffers().is_ok())
            .unwrap_or(false);
    let grouped_decode = grouped_decode_shape && w4a16_decode;
    let library = Arc::clone(&cuda_cache.library);
    let cuda_stream = cuda_cache.stream.as_ptr();
    let completion_stream = cuda_cache.completion_stream.as_ptr();
    let workspace = cuda_cache.workspace.ensure_accumulation_buffers(
        Arc::clone(&library),
        hidden_bytes,
        accumulator_bytes,
        1,
        index_bytes,
        weight_bytes,
        index_bytes,
    )?;
    let lane_count = if split_m1_m2_w4a16 {
        2
    } else if packed_w4a16_topk8_prefill {
        1
    } else {
        cuda_cache.b12x_lane_count().min(route_groups.len().max(1))
    };
    anyhow::ensure!(
        lane_count <= cuda_cache.b12x_lane_count(),
        "split-M1 physical M=2 requires two B12X route lanes, configured {}",
        cuda_cache.b12x_lane_count()
    );
    let workspace_rows = if packed_w4a16_topk8_prefill {
        if split_m1_m2_w4a16 {
            1
        } else {
            row_count
        }
    } else {
        max_group_rows
    };
    let b12x_workspaces = cuda_cache.ensure_b12x_route_workspaces(
        lane_count,
        workspace_rows,
        hidden_dim,
        max_intermediate_rows,
        output_rows,
    )?;
    for (group_index, group) in route_groups.iter().enumerate() {
        anyhow::ensure!(
            hidden_dim == 6144
                && group.intermediate_rows == 512
                && output_rows == 6144
                && group.count <= B12X_SPARK_AOT_MAX_ROWS,
            "streamed NVFP4 route group {group_index} is not supported by packed W4A16"
        );
    }
    let packed_scatter_indices = packed_w4a16_topk8_prefill
        .then(|| {
            (0..row_count)
                .map(|row| u32::try_from(row).context("packed prefill row index exceeds u32"))
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?;
    let packed_metadata = packed_w4a16_topk8_prefill_plan.as_ref().map(|plan| {
        let route_indices = if split_m1_m2_w4a16 {
            plan.direct_topk_ids.as_slice()
        } else {
            plan.packed_route_indices.as_slice()
        };
        let block_expert_ids = if split_m1_m2_w4a16 {
            &[][..]
        } else {
            plan.block_expert_ids.as_slice()
        };
        let mut values = Vec::with_capacity(route_indices.len() + block_expert_ids.len() + 1);
        values.extend_from_slice(route_indices);
        let block_expert_offset = values.len() * std::mem::size_of::<u32>();
        values.extend_from_slice(block_expert_ids);
        let route_count_offset = values.len() * std::mem::size_of::<u32>();
        values.push(plan.packed_route_count);
        (values, block_expert_offset, route_count_offset)
    });
    let scatter_indices = packed_scatter_indices
        .as_deref()
        .unwrap_or(output_indices.as_slice());
    let staged_route_weights = packed_w4a16_topk8_prefill_plan
        .as_ref()
        .map(|plan| plan.topk_weights.as_slice())
        .unwrap_or(route_weights.as_slice());
    let metadata_payloads = cuda_cache.workspace.stage_accumulation_metadata_payloads(
        Arc::clone(&library),
        u32_bytes(scatter_indices),
        f32_bytes(staged_route_weights),
    )?;
    let grouped_expert_ids = grouped_decode
        .then(|| {
            route_groups
                .iter()
                .map(|group| {
                    u32::try_from(group.projections.gate.key.expert_id)
                        .context("grouped decode expert ID exceeds u32")
                })
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?;
    let input_index_payload = cuda_cache.workspace.stage_stream_input_indices(
        Arc::clone(&library),
        packed_metadata
            .as_ref()
            .map(|(values, _, _)| values.as_slice())
            .or(grouped_expert_ids.as_deref())
            .unwrap_or(&input_indices),
    )?;
    unsafe {
        library
            .cuda_stream_synchronize(cuda_stream)
            .context("synchronizing streamed NVFP4 route projection setup")?;
        library
            .copy_host_buffer_h2d_async(
                workspace.scatter_index,
                metadata_payloads.scatter_index,
                scatter_indices.len() * std::mem::size_of::<u32>(),
                completion_stream,
            )
            .context("copying streamed NVFP4 route output indices")?;
        if let Some((packed_metadata, block_expert_offset, route_count_offset)) =
            packed_metadata.as_ref()
        {
            let b12x_workspace = b12x_workspaces[0];
            let packed_route_bytes = *block_expert_offset;
            let block_expert_bytes = *route_count_offset - *block_expert_offset;
            if split_m1_m2_w4a16 {
                let topk_id_row_bytes =
                    B12X_W4A16_PREFILL_TOPK8_ROUTES * std::mem::size_of::<u32>();
                let topk_weight_row_bytes =
                    B12X_W4A16_PREFILL_TOPK8_ROUTES * std::mem::size_of::<f32>();
                for (lane, lane_workspace) in b12x_workspaces.iter().copied().enumerate() {
                    library
                        .copy_host_buffer_h2d_async(
                            lane_workspace.w4a16_packed_route_indices,
                            host_buffer_byte_view(
                                input_index_payload,
                                lane * topk_id_row_bytes,
                                topk_id_row_bytes,
                                "split-M1 W4A16 expert IDs",
                            )?,
                            topk_id_row_bytes,
                            completion_stream,
                        )
                        .context("copying split-M1 W4A16 expert IDs")?;
                    library
                        .copy_host_buffer_h2d_async(
                            lane_workspace.w4a16_topk_weights,
                            host_buffer_byte_view(
                                metadata_payloads.route_weights,
                                lane * topk_weight_row_bytes,
                                topk_weight_row_bytes,
                                "split-M1 W4A16 route weights",
                            )?,
                            topk_weight_row_bytes,
                            completion_stream,
                        )
                        .context("copying split-M1 W4A16 route weights")?;
                }
            } else {
                library
                    .copy_host_buffer_h2d_async(
                        b12x_workspace.w4a16_packed_route_indices,
                        input_index_payload,
                        packed_route_bytes,
                        completion_stream,
                    )
                    .context("copying packed W4A16 prefill route indices")?;
            }
            if block_expert_bytes > 0 {
                library
                    .copy_host_buffer_h2d_async(
                        b12x_workspace.w4a16_block_expert_ids,
                        host_buffer_byte_view(
                            input_index_payload,
                            *block_expert_offset,
                            block_expert_bytes,
                            "packed W4A16 prefill block expert IDs",
                        )?,
                        block_expert_bytes,
                        completion_stream,
                    )
                    .context("copying packed W4A16 prefill block expert IDs")?;
            }
            library
                .copy_host_buffer_h2d_async(
                    b12x_workspace.w4a16_packed_route_count,
                    host_buffer_byte_view(
                        input_index_payload,
                        *route_count_offset,
                        std::mem::size_of::<u32>(),
                        "packed W4A16 prefill route count",
                    )?,
                    std::mem::size_of::<u32>(),
                    completion_stream,
                )
                .context("copying packed W4A16 prefill route count")?;
            if !split_m1_m2_w4a16 {
                library
                    .copy_host_buffer_h2d_async(
                        b12x_workspace.w4a16_topk_weights,
                        metadata_payloads.route_weights,
                        staged_route_weights.len() * std::mem::size_of::<f32>(),
                        completion_stream,
                    )
                    .context("copying packed W4A16 prefill route weights")?;
            }
            debug_assert_eq!(
                packed_metadata.len() * std::mem::size_of::<u32>(),
                *route_count_offset + std::mem::size_of::<u32>()
            );
        } else {
            library
                .copy_host_buffer_h2d_async(
                    workspace.route_weights,
                    metadata_payloads.route_weights,
                    weight_bytes,
                    completion_stream,
                )
                .context("copying streamed NVFP4 route weights")?;
            library
                .copy_host_buffer_h2d_async(
                    workspace.route_metadata,
                    input_index_payload,
                    index_bytes,
                    completion_stream,
                )
                .context("copying streamed NVFP4 route input indices")?;
        }
        if !packed_w4a16_direct_fp8_output {
            library
                .cuda_zero_f32_async(
                    workspace.accumulator,
                    row_count * output_rows,
                    completion_stream,
                )
                .context("zeroing streamed NVFP4 route accumulator")?;
        }
        library
            .cuda_stream_synchronize(completion_stream)
            .context("synchronizing streamed NVFP4 route initialization")?;
    }

    let scheduled_group_count = route_groups.len();
    Ok(RouteNvfp4IngressStream {
        layer_id,
        row_count,
        route_count,
        hidden_dim,
        hidden_row_stride_bytes,
        output_rows,
        output_dtype,
        spark_reduction,
        spark_row_sharded_reduction,
        route_groups,
        group_ready_after_rows: plan
            .groups
            .iter()
            .map(|group| group.ready_after_rows as usize)
            .collect(),
        max_intermediate_rows,
        max_group_rows: workspace_rows,
        lane_count,
        packed_w4a16_topk8_prefill,
        m1_parity_grouped_small_m_w4a16,
        split_m1_m2_w4a16,
        w4a16_small_m_mode,
        topk8_combined_output_in_group_buffer,
        packed_w4a16_direct_fp8_output,
        grouped_decode,
        collective_request_id: None,
        collective_launch_ticket: None,
        lane_used_since_emit: vec![false; lane_count],
        scheduled_group_count,
        next_group: 0,
        received_rows: 0,
        emitted_rows: 0,
    })
}

fn reduction_output_dtype(dtype: ExpertIntermediateReductionDtype) -> RouteStreamingOutputDtype {
    match dtype {
        ExpertIntermediateReductionDtype::Bf16 => RouteStreamingOutputDtype::Bf16,
        ExpertIntermediateReductionDtype::Fp8 => RouteStreamingOutputDtype::Fp8E4m3RowScaled,
        ExpertIntermediateReductionDtype::Nvfp4 => RouteStreamingOutputDtype::Nvfp4E2m1Fp8E4m3,
    }
}

fn fused_fp8_reduction_eligible(
    reduction_dtype: RouteStreamingOutputDtype,
    output_dtype: RouteStreamingOutputDtype,
    completed_rows: &[usize],
    row_count: usize,
) -> bool {
    reduction_dtype == RouteStreamingOutputDtype::Fp8E4m3RowScaled
        && output_dtype == RouteStreamingOutputDtype::Fp8E4m3RowScaled
        && completed_rows.len() == row_count
        && completed_rows
            .iter()
            .enumerate()
            .all(|(index, row)| *row == index)
}

#[allow(clippy::too_many_arguments)]
unsafe fn pack_streamed_completion_rows(
    library: &NativeLibrary,
    dtype: RouteStreamingOutputDtype,
    accumulator: Ds41rtDeviceBuffer,
    accumulator_rows: usize,
    indices: Ds41rtDeviceBuffer,
    f32_output: Ds41rtDeviceBuffer,
    output: Ds41rtDeviceBuffer,
    rows: usize,
    row_width: usize,
    output_row_stride_bytes: usize,
    cuda_stream: *mut c_void,
) -> Result<()> {
    match dtype {
        RouteStreamingOutputDtype::Bf16 => unsafe {
            library.cuda_gather_rows_f32_async(
                accumulator,
                accumulator_rows,
                indices,
                f32_output,
                rows,
                row_width,
                cuda_stream,
            )?;
            library.cuda_f32_to_bf16_async(f32_output, output, rows * row_width, cuda_stream)?;
        },
        RouteStreamingOutputDtype::Fp8E4m3RowScaled => unsafe {
            library.cuda_gather_rows_f32_to_fp8_e4m3_row_scaled_async(
                accumulator,
                accumulator_rows,
                indices,
                output,
                rows,
                row_width,
                output_row_stride_bytes,
                cuda_stream,
            )?;
        },
        RouteStreamingOutputDtype::Nvfp4E2m1Fp8E4m3 => unsafe {
            library.cuda_gather_rows_f32_to_nvfp4_e2m1_fp8_e4m3_async(
                accumulator,
                accumulator_rows,
                indices,
                output,
                rows,
                row_width,
                output_row_stride_bytes,
                cuda_stream,
            )?;
        },
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
unsafe fn scatter_add_streamed_reduction_rows(
    library: &NativeLibrary,
    dtype: RouteStreamingOutputDtype,
    input: Ds41rtDeviceBuffer,
    input_row_stride_bytes: usize,
    indices: Ds41rtDeviceBuffer,
    accumulator: Ds41rtDeviceBuffer,
    accumulator_rows: usize,
    rows: usize,
    row_width: usize,
    cuda_stream: *mut c_void,
) -> Result<()> {
    match dtype {
        RouteStreamingOutputDtype::Bf16 => unsafe {
            library.cuda_scatter_add_rows_bf16_to_f32_async(
                input,
                indices,
                accumulator,
                accumulator_rows,
                rows,
                row_width,
                cuda_stream,
            )?;
        },
        RouteStreamingOutputDtype::Fp8E4m3RowScaled => unsafe {
            library.cuda_scatter_add_rows_fp8_e4m3_row_scaled_to_f32_async(
                input,
                input_row_stride_bytes,
                indices,
                accumulator,
                accumulator_rows,
                rows,
                row_width,
                cuda_stream,
            )?;
        },
        RouteStreamingOutputDtype::Nvfp4E2m1Fp8E4m3 => unsafe {
            library.cuda_scatter_add_rows_nvfp4_e2m1_fp8_e4m3_to_f32_async(
                input,
                input_row_stride_bytes,
                indices,
                accumulator,
                accumulator_rows,
                rows,
                row_width,
                cuda_stream,
            )?;
        },
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
unsafe fn execute_rdma_row_sharded_reduction(
    cuda_cache: &mut RouteCudaCache,
    library: &NativeLibrary,
    request_id: u64,
    reduction_dtype: RouteStreamingOutputDtype,
    reduction_row_bytes: usize,
    workspace: RouteCudaAccumulationWorkspaceBuffers,
    completion: RouteCudaCompletionWorkspaceBuffers,
    reduction_buffers: RouteCudaReductionWorkspaceBuffers,
    packed_bf16: Option<Ds41rtDeviceBuffer>,
    rows: usize,
    row_width: usize,
    output_dtype: RouteStreamingOutputDtype,
    output_row_bytes: usize,
    cuda_stream: *mut c_void,
    timeline: Option<&RouteCudaEventTimeline>,
) -> Result<(usize, usize)> {
    let exchange = if let Some(packed_bf16) = packed_bf16 {
        anyhow::ensure!(
            matches!(
                reduction_dtype,
                RouteStreamingOutputDtype::Bf16 | RouteStreamingOutputDtype::Fp8E4m3RowScaled
            ),
            "direct packed Spark RDMA reduction requires BF16 or FP8 wire rows"
        );
        if let Some(timeline) = timeline {
            timeline.record(
                timeline.metadata_ready.as_ref(),
                cuda_stream,
                "direct RDMA wire pack start",
            )?;
        }
        let reduction = cuda_cache
            .spark_rdma_reduction
            .as_mut()
            .context("Spark RDMA route stream lost its reduction backend")?;
        match reduction_dtype {
            RouteStreamingOutputDtype::Fp8E4m3RowScaled => reduction
                .exchange_bf16_row_partitions(
                    library,
                    request_id,
                    packed_bf16,
                    rows,
                    row_width,
                    reduction_row_bytes,
                    cuda_stream,
                )
                .context("packing and exchanging row-sharded Spark RDMA expert outputs")?,
            RouteStreamingOutputDtype::Bf16 => reduction
                .exchange_row_partitions(
                    library,
                    request_id,
                    packed_bf16,
                    rows,
                    row_width,
                    reduction_row_bytes,
                    cuda_stream,
                )
                .context("exchanging BF16 row-sharded Spark RDMA expert outputs")?,
            RouteStreamingOutputDtype::Nvfp4E2m1Fp8E4m3 => {
                unreachable!("direct packed Spark RDMA reduction dtype checked above")
            }
        }
    } else {
        pack_streamed_completion_rows(
            library,
            reduction_dtype,
            workspace.accumulator,
            rows,
            completion.indices,
            completion.f32_output,
            reduction_buffers.send,
            rows,
            row_width,
            reduction_row_bytes,
            cuda_stream,
        )
        .context("packing Spark RDMA reduction rows")?;
        if let Some(timeline) = timeline {
            timeline.record(
                timeline.metadata_ready.as_ref(),
                cuda_stream,
                "packed RDMA wire end",
            )?;
        }
        cuda_cache
            .spark_rdma_reduction
            .as_mut()
            .context("Spark RDMA route stream lost its reduction backend")?
            .exchange_row_partitions(
                library,
                request_id,
                reduction_buffers.send,
                rows,
                row_width,
                reduction_row_bytes,
                cuda_stream,
            )
            .context("exchanging row-sharded Spark RDMA expert outputs")?
    };
    if let Some(timeline) = timeline {
        timeline.record(
            timeline.routes_ready.as_ref(),
            cuda_stream,
            "packed RDMA exchange end",
        )?;
    }

    let local_value_bytes = exchange
        .row_count
        .checked_mul(row_width)
        .context("Spark RDMA local value count overflow")?;
    let (local, local_dtype) = if let Some(packed_bf16) = packed_bf16 {
        let offset = exchange
            .row_start
            .checked_mul(row_width)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("Spark RDMA local BF16 offset overflow")?;
        let bytes = local_value_bytes
            .checked_mul(std::mem::size_of::<u16>())
            .context("Spark RDMA local BF16 byte count overflow")?;
        (
            route_device_buffer_slice(packed_bf16, offset, bytes)?,
            DS41RT_ROUTE_SHARD_LOCAL_BF16,
        )
    } else {
        let offset = exchange
            .row_start
            .checked_mul(row_width)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .context("Spark RDMA local F32 offset overflow")?;
        let bytes = local_value_bytes
            .checked_mul(std::mem::size_of::<f32>())
            .context("Spark RDMA local F32 byte count overflow")?;
        (
            route_device_buffer_slice(workspace.accumulator, offset, bytes)?,
            DS41RT_ROUTE_SHARD_LOCAL_F32,
        )
    };
    let output_f32_bytes = local_value_bytes
        .checked_mul(std::mem::size_of::<f32>())
        .context("Spark RDMA combined F32 byte count overflow")?;
    let output_f32 = route_device_buffer_slice(completion.f32_output, 0, output_f32_bytes)?;
    let wire_dtype = match reduction_dtype {
        RouteStreamingOutputDtype::Bf16 => DS41RT_ROUTE_SHARD_WIRE_BF16,
        RouteStreamingOutputDtype::Fp8E4m3RowScaled => DS41RT_ROUTE_SHARD_WIRE_FP8_E4M3_ROW_SCALED,
        RouteStreamingOutputDtype::Nvfp4E2m1Fp8E4m3 => DS41RT_ROUTE_SHARD_WIRE_NVFP4_E2M1_FP8_E4M3,
    };
    let local_row_bytes = row_width
        .checked_mul(if packed_bf16.is_some() {
            std::mem::size_of::<u16>()
        } else {
            std::mem::size_of::<f32>()
        })
        .context("Spark RDMA local row byte count overflow")?;
    let output_f32_row_bytes = row_width
        .checked_mul(std::mem::size_of::<f32>())
        .context("Spark RDMA output row byte count overflow")?;
    for segment in exchange.segments()? {
        let local_offset = segment
            .row_offset
            .checked_mul(local_row_bytes)
            .context("Spark RDMA local segment offset overflow")?;
        let local_bytes = segment
            .row_count
            .checked_mul(local_row_bytes)
            .context("Spark RDMA local segment byte count overflow")?;
        let output_offset = segment
            .row_offset
            .checked_mul(output_f32_row_bytes)
            .context("Spark RDMA output segment offset overflow")?;
        let output_bytes = segment
            .row_count
            .checked_mul(output_f32_row_bytes)
            .context("Spark RDMA output segment byte count overflow")?;
        library
            .cuda_reduce_route_shards_to_f32_async(
                &Ds41rtRouteShardReductionBuffers {
                    local: route_device_buffer_slice(local, local_offset, local_bytes)?,
                    peers: segment.peer_payloads,
                    output_f32: route_device_buffer_slice(output_f32, output_offset, output_bytes)?,
                },
                segment.row_count,
                row_width,
                reduction_row_bytes,
                local_dtype,
                wire_dtype,
                3,
                cuda_stream,
            )
            .context("combining mapped Spark RDMA route shard rail")?;
    }
    match output_dtype {
        RouteStreamingOutputDtype::Bf16 => library
            .cuda_f32_to_bf16_async(
                output_f32,
                completion.output,
                local_value_bytes,
                cuda_stream,
            )
            .context("packing Spark RDMA BF16 response rows")?,
        RouteStreamingOutputDtype::Fp8E4m3RowScaled
        | RouteStreamingOutputDtype::Nvfp4E2m1Fp8E4m3 => {
            pack_streamed_completion_rows(
                library,
                output_dtype,
                output_f32,
                exchange.row_count,
                completion.indices,
                completion.f32_output,
                completion.output,
                exchange.row_count,
                row_width,
                output_row_bytes,
                cuda_stream,
            )
            .context("packing Spark RDMA response rows")?;
        }
    }
    if let Some(timeline) = timeline {
        timeline.record(
            timeline.pack_ready.as_ref(),
            cuda_stream,
            "packed RDMA combine end",
        )?;
    }
    library
        .cuda_stream_synchronize(cuda_stream)
        .context("synchronizing mapped Spark RDMA route reduction")?;
    let row_start = exchange.row_start;
    let row_count = exchange.row_count;
    cuda_cache
        .spark_rdma_reduction
        .as_mut()
        .expect("Spark RDMA exchange requires its backend")
        .release_exchange(exchange)?;
    Ok((row_start, row_count))
}

pub(in crate::commands::real_full) fn reduce_mapped_route_shards_cached_host_output<'a>(
    local_bf16: Ds41rtDeviceBuffer,
    peer_payloads: &[Ds41rtDeviceBuffer],
    peer_dtype: RouteStreamingOutputDtype,
    rows: usize,
    row_width: usize,
    output_dtype: RouteStreamingOutputDtype,
    cache: &'a mut RouteTensorCache,
) -> Result<&'a [u8]> {
    anyhow::ensure!(
        (1..=3).contains(&peer_payloads.len()),
        "mapped route shard reduction requires one to three peers"
    );
    let local_bytes = rows
        .checked_mul(row_width)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("mapped route shard local BF16 byte count overflow")?;
    anyhow::ensure!(
        !local_bf16.ptr.is_null() && local_bf16.bytes >= local_bytes,
        "mapped route shard local BF16 buffer has {} bytes, needs {local_bytes}",
        local_bf16.bytes
    );
    let peer_row_stride_bytes = peer_dtype.row_stride_bytes(row_width)?;
    let peer_bytes = rows
        .checked_mul(peer_row_stride_bytes)
        .context("mapped route shard peer byte count overflow")?;
    for (index, peer) in peer_payloads.iter().enumerate() {
        anyhow::ensure!(
            !peer.ptr.is_null() && peer.bytes >= peer_bytes,
            "mapped route shard peer {index} has {} bytes, needs {peer_bytes}",
            peer.bytes
        );
    }

    let output_row_stride_bytes = output_dtype.row_stride_bytes(row_width)?;
    let output_bytes = rows
        .checked_mul(output_row_stride_bytes)
        .context("mapped route shard output byte count overflow")?;
    let accumulator_bytes = rows
        .checked_mul(row_width)
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .context("mapped route shard accumulator byte count overflow")?;
    let completion_indices = if output_dtype == RouteStreamingOutputDtype::Bf16 {
        Vec::new()
    } else {
        (0..rows)
            .map(|row| u32::try_from(row).context("mapped route shard row index exceeds u32"))
            .collect::<Result<Vec<_>>>()?
    };
    let wire_dtype = match peer_dtype {
        RouteStreamingOutputDtype::Bf16 => DS41RT_ROUTE_SHARD_WIRE_BF16,
        RouteStreamingOutputDtype::Fp8E4m3RowScaled => DS41RT_ROUTE_SHARD_WIRE_FP8_E4M3_ROW_SCALED,
        RouteStreamingOutputDtype::Nvfp4E2m1Fp8E4m3 => DS41RT_ROUTE_SHARD_WIRE_NVFP4_E2M1_FP8_E4M3,
    };

    let cuda_cache = cache.cuda_cache()?;
    let library = Arc::clone(&cuda_cache.library);
    let stream = cuda_cache.completion_stream.as_ptr();
    let completion = cuda_cache.workspace.ensure_completion_buffers(
        Arc::clone(&library),
        &completion_indices,
        accumulator_bytes,
        output_bytes,
    )?;
    let reduction = cuda_cache.workspace.ensure_reduction_buffers(
        Arc::clone(&library),
        accumulator_bytes,
        1,
    )?;
    let mut peers = [Ds41rtDeviceBuffer::default(); 3];
    peers[..peer_payloads.len()].copy_from_slice(peer_payloads);
    let buffers = Ds41rtRouteShardReductionBuffers {
        local: local_bf16,
        peers,
        output_f32: reduction.send,
    };

    unsafe {
        if !completion_indices.is_empty() {
            library
                .copy_host_buffer_h2d_async(
                    completion.indices,
                    completion.pinned_indices,
                    completion_indices.len() * std::mem::size_of::<u32>(),
                    stream,
                )
                .context("copying mapped route shard row indices")?;
        }
        library
            .cuda_reduce_route_shards_to_f32_async(
                &buffers,
                rows,
                row_width,
                peer_row_stride_bytes,
                DS41RT_ROUTE_SHARD_LOCAL_BF16,
                wire_dtype,
                peer_payloads.len(),
                stream,
            )
            .context("reducing mapped route shards on the owner GPU")?;
        if output_dtype == RouteStreamingOutputDtype::Bf16 {
            library
                .cuda_f32_to_bf16_async(reduction.send, completion.output, rows * row_width, stream)
                .context("packing identity-ordered mapped owner BF16 response")?;
        } else {
            pack_streamed_completion_rows(
                &library,
                output_dtype,
                reduction.send,
                rows,
                completion.indices,
                completion.f32_output,
                completion.output,
                rows,
                row_width,
                output_row_stride_bytes,
                stream,
            )
            .context("packing mapped owner-reduced route response")?;
        }
        library
            .copy_d2h_host_buffer_async(
                completion.pinned_output,
                completion.output,
                output_bytes,
                stream,
            )
            .context("copying mapped owner-reduced route response to host")?;
        library
            .cuda_stream_synchronize(stream)
            .context("synchronizing mapped owner route reduction")?;
    }

    Ok(cuda_cache.workspace.completion_output_slice(output_bytes)?)
}

pub(in crate::commands::real_full) fn reduce_mapped_route_shards_cached(
    local_bf16: Ds41rtDeviceBuffer,
    peer_payloads: &[Ds41rtDeviceBuffer],
    peer_dtype: RouteStreamingOutputDtype,
    rows: usize,
    row_width: usize,
    output_dtype: RouteStreamingOutputDtype,
    cache: &mut RouteTensorCache,
) -> Result<Vec<u8>> {
    Ok(reduce_mapped_route_shards_cached_host_output(
        local_bf16,
        peer_payloads,
        peer_dtype,
        rows,
        row_width,
        output_dtype,
        cache,
    )?
    .to_vec())
}

fn route_device_buffer_slice(
    buffer: Ds41rtDeviceBuffer,
    offset_bytes: usize,
    bytes: usize,
) -> Result<Ds41rtDeviceBuffer> {
    let end = offset_bytes
        .checked_add(bytes)
        .context("route device buffer slice end overflow")?;
    anyhow::ensure!(
        end <= buffer.bytes,
        "route device buffer slice [{offset_bytes}, {end}) exceeds {} bytes",
        buffer.bytes
    );
    Ok(Ds41rtDeviceBuffer {
        ptr: unsafe { buffer.ptr.cast::<u8>().add(offset_bytes).cast() },
        bytes,
        device_id: buffer.device_id,
        flags: buffer.flags,
    })
}

pub(in crate::commands::real_full) fn execute_nvfp4_route_ingress_stream_chunk_cached(
    _catalog: &TensorCatalog,
    state: &mut RouteNvfp4IngressStream,
    hidden_chunk: &[u8],
    hidden_chunk_device: Option<Ds41rtDeviceBuffer>,
    retain_device_output: bool,
    response_device_target: Option<Ds41rtDeviceBuffer>,
    row_offset: usize,
    final_frame: bool,
    cache: &mut RouteTensorCache,
) -> Result<RouteNvfp4IngressStreamChunk> {
    anyhow::ensure!(
        row_offset == state.received_rows,
        "streamed NVFP4 route chunk offset {row_offset} did not match {}",
        state.received_rows
    );
    anyhow::ensure!(
        !hidden_chunk.is_empty() && hidden_chunk.len() % state.hidden_row_stride_bytes == 0,
        "streamed NVFP4 route chunk bytes {} are not a positive multiple of stride {}",
        hidden_chunk.len(),
        state.hidden_row_stride_bytes
    );
    let chunk_rows = hidden_chunk.len() / state.hidden_row_stride_bytes;
    let row_end = row_offset
        .checked_add(chunk_rows)
        .context("streamed NVFP4 route chunk row range overflow")?;
    anyhow::ensure!(
        row_end <= state.row_count,
        "streamed NVFP4 route chunk rows {row_offset}..{row_end} exceed {}",
        state.row_count
    );
    anyhow::ensure!(
        final_frame == (row_end == state.row_count),
        "streamed NVFP4 route final marker={final_frame} at rows {row_end}/{}",
        state.row_count
    );
    if let Some(device) = hidden_chunk_device {
        anyhow::ensure!(
            !device.ptr.is_null() && hidden_chunk.len() <= device.bytes,
            "streamed NVFP4 route device chunk has {} bytes for {} host bytes",
            device.bytes,
            hidden_chunk.len()
        );
    }
    anyhow::ensure!(
        response_device_target.is_none() || retain_device_output,
        "streamed route response target requires retained device output"
    );
    if let Some(target) = response_device_target {
        anyhow::ensure!(
            !target.ptr.is_null(),
            "streamed route response target is null"
        );
    }

    let hidden_bytes = state
        .row_count
        .checked_mul(state.hidden_row_stride_bytes)
        .context("streamed NVFP4 route hidden byte count overflow")?;
    let accumulator_bytes = state
        .row_count
        .checked_mul(state.output_rows)
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .context("streamed NVFP4 route accumulator byte count overflow")?;
    let index_bytes = state
        .route_count
        .checked_mul(std::mem::size_of::<u32>())
        .context("streamed NVFP4 route index byte count overflow")?;
    let weight_bytes = state
        .route_count
        .checked_mul(std::mem::size_of::<f32>())
        .context("streamed NVFP4 route weight byte count overflow")?;
    let cuda_cache = cache.cuda_cache()?;
    cuda_cache.prepare_layer(state.layer_id);
    let library = Arc::clone(&cuda_cache.library);
    let completion_stream = cuda_cache.completion_stream.as_ptr();
    let workspace = cuda_cache.workspace.ensure_accumulation_buffers(
        Arc::clone(&library),
        hidden_bytes,
        accumulator_bytes,
        1,
        index_bytes,
        weight_bytes,
        index_bytes,
    )?;
    let b12x_workspaces = cuda_cache.ensure_b12x_route_workspaces(
        state.lane_count,
        state.max_group_rows,
        state.hidden_dim,
        state.max_intermediate_rows,
        state.output_rows,
    )?;
    let packed_prefill_rdma_bf16_direct = state.packed_w4a16_topk8_prefill
        && state.spark_row_sharded_reduction
        && cuda_cache.spark_rdma_reduction_enabled()
        && cuda_cache.spark_reduction_dtype() == Some(ExpertIntermediateReductionDtype::Fp8);
    let packed_prefill_timeline = (packed_prefill_rdma_bf16_direct
        && RouteCudaEventTimeline::enabled())
    .then(|| RouteCudaEventTimeline::new(Arc::clone(&library)))
    .transpose()?;
    let direct_packed_input =
        (state.packed_w4a16_topk8_prefill && row_offset == 0 && row_end == state.row_count)
            .then_some(hidden_chunk_device)
            .flatten()
            .map(|device| route_device_buffer_slice(device, 0, hidden_chunk.len()))
            .transpose()?;
    if direct_packed_input.is_none() {
        let pinned_hidden = cuda_cache
            .workspace
            .stage_stream_hidden_payload(Arc::clone(&library), hidden_chunk)?;
        let hidden_offset = row_offset
            .checked_mul(state.hidden_row_stride_bytes)
            .context("streamed NVFP4 route hidden offset overflow")?;
        let hidden_target = device_buffer_byte_view(
            workspace.hidden,
            hidden_offset,
            hidden_chunk.len(),
            "streamed NVFP4 route hidden chunk",
        )?;
        unsafe {
            library
                .copy_host_buffer_h2d_async(
                    hidden_target,
                    pinned_hidden,
                    hidden_chunk.len(),
                    completion_stream,
                )
                .context("copying streamed NVFP4 route hidden chunk")?;
            library
                .cuda_stream_synchronize(completion_stream)
                .context("synchronizing streamed NVFP4 route hidden chunk")?;
        }
    }

    let mut completed_rows = Vec::new();
    let mut packed_prefill_bf16_output = None;
    let mut packed_prefill_fp8_output = None;
    if state.packed_w4a16_topk8_prefill {
        if row_end == state.row_count {
            anyhow::ensure!(
                state.next_group == 0,
                "packed W4A16 top-k=8 prefill was launched more than once"
            );
            let slab = cuda_cache
                .expert_slabs
                .get(&state.layer_id)
                .context("top-k=8 prefill lost its expert slab")?;
            let w4a16_layer_buffers = slab.w4a16_moe_buffers()?;
            let group_stream = cuda_cache.b12x_lane_stream(0);
            let b12x_workspace = b12x_workspaces[0];
            let input_payload = direct_packed_input.unwrap_or(device_buffer_byte_view(
                workspace.hidden,
                0,
                state.row_count * state.hidden_row_stride_bytes,
                "packed W4A16 top-k=8 prefill NVFP4 input",
            )?);
            let combined_output = device_buffer_byte_view(
                if state.topk8_combined_output_in_group_buffer {
                    b12x_workspace.group_output
                } else {
                    b12x_workspace.compact_hidden
                },
                0,
                state.row_count * state.output_rows * std::mem::size_of::<u16>(),
                "packed W4A16 top-k=8 prefill combined output",
            )?;
            let output_indices = device_buffer_byte_view(
                workspace.scatter_index,
                0,
                state.row_count * std::mem::size_of::<u32>(),
                "packed W4A16 top-k=8 prefill output indices",
            )?;
            unsafe {
                if let Some(timeline) = packed_prefill_timeline.as_ref() {
                    timeline.record(
                        timeline.start.as_ref(),
                        group_stream,
                        "packed compute start",
                    )?;
                }
                let buffers = b12x_w4a16_moe_buffers(
                    w4a16_layer_buffers,
                    b12x_workspace,
                    b12x_workspace.compact_hidden,
                    b12x_workspace.group_output,
                    b12x_workspace.w4a16_topk_weights,
                );
                let direct_fp8_target = (!state.spark_reduction
                    && state.packed_w4a16_direct_fp8_output)
                    .then_some(response_device_target)
                    .flatten()
                    .map(|target| {
                        let output_bytes = state
                            .row_count
                            .checked_mul(state.output_dtype.row_stride_bytes(state.output_rows)?)
                            .context("direct packed FP8 response byte count overflow")?;
                        route_device_buffer_slice(target, 0, output_bytes)
                    })
                    .transpose()?;
                if state.m1_parity_grouped_small_m_w4a16 {
                    anyhow::ensure!(
                            state.hidden_dim == state.output_rows,
                            "grouped W4A16 M2..8 row parity requires hidden/output equality, got hidden={} output={}",
                            state.hidden_dim,
                            state.output_rows
                        );
                    match state.w4a16_small_m_mode {
                        W4a16SmallMMode::Ordered => library
                            .cuda_b12x_spark_w4a16_m1_parity_grouped_m2_8_nvfp4_async(
                                &buffers,
                                input_payload,
                                state.hidden_row_stride_bytes,
                                state.row_count,
                                group_stream,
                            )
                            .context(
                                "launching expert-grouped block-8 W4A16 M2..8 row-parity MoE",
                            )?,
                        W4a16SmallMMode::WideOrdered => library
                            .cuda_b12x_spark_w4a16_m1_parity_grouped_wide_m2_8_nvfp4_async(
                                &buffers,
                                input_payload,
                                state.hidden_row_stride_bytes,
                                state.row_count,
                                group_stream,
                            )
                            .context(
                                "launching expert-grouped wide-FC2 W4A16 M2..8 row-parity MoE",
                            )?,
                        W4a16SmallMMode::SplitM1 if state.split_m1_m2_w4a16 => {
                            anyhow::ensure!(
                                state.split_m1_m2_w4a16
                                    && state.row_count == 2
                                    && state.lane_count == 2
                                    && b12x_workspaces.len() == 2,
                                "split-M1 W4A16 requires physical M=2 and two route workspaces"
                            );
                            let input_row_bytes = state.hidden_row_stride_bytes;
                            let output_row_bytes = state
                                .output_rows
                                .checked_mul(std::mem::size_of::<u16>())
                                .context("split-M1 W4A16 output row byte count overflow")?;
                            let topk_id_row_bytes = B12X_W4A16_PREFILL_TOPK8_ROUTES
                                .checked_mul(std::mem::size_of::<u32>())
                                .context("split-M1 W4A16 expert-ID byte count overflow")?;
                            let topk_weight_row_bytes = B12X_W4A16_PREFILL_TOPK8_ROUTES
                                .checked_mul(std::mem::size_of::<f32>())
                                .context("split-M1 W4A16 route-weight byte count overflow")?;
                            for (lane, lane_workspace) in
                                b12x_workspaces.iter().copied().enumerate()
                            {
                                let lane_stream = cuda_cache.b12x_lane_stream(lane);
                                let lane_input_payload = route_device_buffer_slice(
                                    input_payload,
                                    lane * input_row_bytes,
                                    input_row_bytes,
                                )?;
                                let lane_topk_ids = route_device_buffer_slice(
                                    lane_workspace.w4a16_packed_route_indices,
                                    0,
                                    topk_id_row_bytes,
                                )?;
                                let lane_topk_weights = route_device_buffer_slice(
                                    lane_workspace.w4a16_topk_weights,
                                    0,
                                    topk_weight_row_bytes,
                                )?;
                                let lane_buffers = b12x_w4a16_moe_buffers(
                                    w4a16_layer_buffers,
                                    lane_workspace,
                                    lane_workspace.compact_hidden,
                                    lane_workspace.group_output,
                                    lane_topk_weights,
                                );
                                library
                                    .cuda_b12x_spark_w4a16_decode_m1_fused_sum_nvfp4_async(
                                        &lane_buffers,
                                        lane_input_payload,
                                        input_row_bytes,
                                        lane_topk_ids,
                                        lane_stream,
                                    )
                                    .context("launching split-M1 W4A16 row")?;
                                library
                                    .copy_d2d_async(
                                        route_device_buffer_slice(
                                            combined_output,
                                            lane * output_row_bytes,
                                            output_row_bytes,
                                        )?,
                                        route_device_buffer_slice(
                                            lane_workspace.group_output,
                                            0,
                                            output_row_bytes,
                                        )?,
                                        output_row_bytes,
                                        lane_stream,
                                    )
                                    .context("joining split-M1 W4A16 row output")?;
                                library
                                    .cuda_event_record(
                                        cuda_cache.b12x_lane_event(lane),
                                        lane_stream,
                                    )
                                    .context("recording split-M1 W4A16 row completion")?;
                            }
                        }
                        W4a16SmallMMode::SplitM1 => library
                            .cuda_b12x_spark_w4a16_m1_parity_grouped_wide_m2_8_nvfp4_async(
                                &buffers,
                                input_payload,
                                state.hidden_row_stride_bytes,
                                state.row_count,
                                group_stream,
                            )
                            .context(
                                "launching split-M1 fallback wide-FC2 W4A16 M3..8 row-parity MoE",
                            )?,
                    }
                    if let Some(target) = direct_fp8_target {
                        if state.split_m1_m2_w4a16 {
                            library
                                .cuda_stream_wait_event(group_stream, cuda_cache.b12x_lane_event(1))
                                .context("joining split-M1 W4A16 rows before direct FP8 pack")?;
                        }
                        library
                            .cuda_bf16_rows_to_fp8_e4m3_row_scaled_async(
                                combined_output,
                                target,
                                state.row_count,
                                state.output_rows,
                                state.output_dtype.row_stride_bytes(state.output_rows)?,
                                group_stream,
                            )
                            .context("packing grouped W4A16 M2..8 row-parity FP8 response")?;
                        packed_prefill_fp8_output = Some(target);
                    }
                } else if let Some(target) = direct_fp8_target {
                    library
                        .cuda_b12x_spark_w4a16_prefill_topk8_nvfp4_fp8_async(
                            &buffers,
                            input_payload,
                            state.hidden_row_stride_bytes,
                            state.row_count,
                            target,
                            state.output_dtype.row_stride_bytes(state.output_rows)?,
                            group_stream,
                        )
                        .context("launching packed B12X W4A16 top-k=8 direct FP8 response")?;
                    packed_prefill_fp8_output = Some(target);
                } else {
                    library
                        .cuda_b12x_spark_w4a16_prefill_topk8_nvfp4_async(
                            &buffers,
                            input_payload,
                            state.hidden_row_stride_bytes,
                            state.row_count,
                            group_stream,
                        )
                        .context("launching packed B12X W4A16 top-k=8 prefill MoE")?;
                }
                if let Some(timeline) = packed_prefill_timeline.as_ref() {
                    timeline.record(
                        timeline.hidden_ready.as_ref(),
                        group_stream,
                        "packed compute end",
                    )?;
                }
                if packed_prefill_fp8_output.is_none() {
                    // The row-sharded RDMA reducer can FP8-pack outgoing BF16
                    // partitions directly and consume the local BF16 shard.
                    // Do not materialize every row through the FP32
                    // accumulator merely because the final coordinator
                    // response uses a different packed dtype.
                    if state.packed_w4a16_direct_fp8_output || packed_prefill_rdma_bf16_direct {
                        packed_prefill_bf16_output = Some(combined_output);
                    } else {
                        library
                            .cuda_scatter_add_rows_bf16_to_f32_async(
                                combined_output,
                                output_indices,
                                workspace.accumulator,
                                state.row_count,
                                state.row_count,
                                state.output_rows,
                                group_stream,
                            )
                            .context("accumulating packed B12X W4A16 top-k=8 prefill output")?;
                    }
                }
                library
                    .cuda_event_record(cuda_cache.b12x_lane_event(0), group_stream)
                    .context("recording packed B12X W4A16 top-k=8 prefill completion")?;
            }
            if state.split_m1_m2_w4a16 {
                state.lane_used_since_emit.fill(true);
            } else {
                state.lane_used_since_emit[0] = true;
            }
            completed_rows.extend(0..state.row_count);
            state.next_group = state.scheduled_group_count;
        }
    } else if state.grouped_decode && state.next_group == 0 && row_end == 1 {
        let w4a16_layer_buffers = cuda_cache
            .expert_slabs
            .get(&state.layer_id)
            .context("W4A16 decode lost its expert slab")?
            .w4a16_moe_buffers()?;
        let group_stream = cuda_cache.b12x_lane_stream(0);
        let b12x_workspace = b12x_workspaces[0];
        let input_payload = device_buffer_byte_view(
            workspace.hidden,
            0,
            state.hidden_row_stride_bytes,
            "grouped decode NVFP4 input payload",
        )?;
        let output = device_buffer_byte_view(
            b12x_workspace.group_output,
            0,
            state.output_rows * std::mem::size_of::<u16>(),
            "grouped decode BF16 output",
        )?;
        let topk_ids = device_buffer_byte_view(
            workspace.route_metadata,
            0,
            8 * std::mem::size_of::<i32>(),
            "grouped decode expert IDs",
        )?;
        let topk_weights = device_buffer_byte_view(
            workspace.route_weights,
            0,
            8 * std::mem::size_of::<f32>(),
            "grouped decode route weights",
        )?;
        let output_index = device_buffer_byte_view(
            workspace.scatter_index,
            0,
            std::mem::size_of::<u32>(),
            "grouped decode output index",
        )?;
        unsafe {
            let mut output_scattered = false;
            if route_cuda_graphs_enabled() {
                cuda_cache.launch_or_capture_packed_w4a16_stream_decode_graph(
                    state.layer_id,
                    workspace,
                    w4a16_layer_buffers,
                    b12x_workspace,
                    state.hidden_row_stride_bytes,
                    state.row_count,
                    state.output_rows,
                    group_stream,
                )?;
                output_scattered = true;
            } else {
                let buffers = b12x_w4a16_moe_buffers(
                    w4a16_layer_buffers,
                    b12x_workspace,
                    b12x_workspace.compact_hidden,
                    output,
                    topk_weights,
                );
                launch_b12x_w4a16_decode(
                    &library,
                    w4a16_layer_buffers,
                    &buffers,
                    input_payload,
                    state.hidden_row_stride_bytes,
                    topk_ids,
                    group_stream,
                )
                .context("launching B12X W4A16 decode MoE")?;
            }
            if !output_scattered {
                library
                    .cuda_scatter_add_rows_bf16_to_f32_async(
                        output,
                        output_index,
                        workspace.accumulator,
                        state.row_count,
                        1,
                        state.output_rows,
                        group_stream,
                    )
                    .context("accumulating grouped B12X TP4 decode output")?;
            }
            library
                .cuda_event_record(cuda_cache.b12x_lane_event(0), group_stream)
                .context("recording grouped B12X TP4 decode completion")?;
        }
        state.lane_used_since_emit[0] = true;
        completed_rows.push(0);
        state.next_group = state.scheduled_group_count;
    } else {
        while state.next_group < state.route_groups.len()
            && state.group_ready_after_rows[state.next_group] <= row_end
        {
            let group_index = state.next_group;
            let group = &state.route_groups[group_index];
            let lane = group_index % state.lane_count;
            let group_stream = cuda_cache.b12x_lane_stream(lane);
            let b12x_workspace = b12x_workspaces[lane];
            let packed_layer = cuda_cache
                .expert_slabs
                .get(&state.layer_id)
                .context("packed W4A16 route lost its expert slab")?
                .w4a16_moe_buffers()?;
            let index_offset = group
                .start
                .checked_mul(std::mem::size_of::<u32>())
                .context("streamed NVFP4 route group index offset overflow")?;
            let group_index_bytes = group
                .count
                .checked_mul(std::mem::size_of::<u32>())
                .context("streamed NVFP4 route group index byte count overflow")?;
            let weight_offset = group
                .start
                .checked_mul(std::mem::size_of::<f32>())
                .context("streamed NVFP4 route group weight offset overflow")?;
            let group_weight_bytes = group
                .count
                .checked_mul(std::mem::size_of::<f32>())
                .context("streamed NVFP4 route group weight byte count overflow")?;
            let input_indices = device_buffer_byte_view(
                workspace.route_metadata,
                index_offset,
                group_index_bytes,
                "streamed NVFP4 route input indices",
            )?;
            let output_indices = device_buffer_byte_view(
                workspace.scatter_index,
                index_offset,
                group_index_bytes,
                "streamed NVFP4 route output indices",
            )?;
            let route_weights = device_buffer_byte_view(
                workspace.route_weights,
                weight_offset,
                group_weight_bytes,
                "streamed NVFP4 route weights",
            )?;
            let group_output_bytes = group
                .count
                .checked_mul(state.output_rows)
                .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
                .context("streamed NVFP4 route group output byte count overflow")?;
            let group_output = device_buffer_byte_view(
                b12x_workspace.group_output,
                0,
                group_output_bytes,
                "streamed NVFP4 route group output",
            )?;
            unsafe {
                library
                    .cuda_b12x_gather_nvfp4_rows_bf16_async(
                        workspace.hidden,
                        state.row_count,
                        state.hidden_row_stride_bytes,
                        input_indices,
                        b12x_workspace.compact_hidden,
                        group.count,
                        state.hidden_dim,
                        group_stream,
                    )
                    .context("gathering packed W4A16 NVFP4 route input rows")?;
                let buffers = b12x_w4a16_moe_buffers(
                    packed_layer,
                    b12x_workspace,
                    b12x_workspace.compact_hidden,
                    group_output,
                    b12x_workspace.w4a16_topk_weights,
                );
                library
                    .cuda_b12x_spark_w4a16_top1_async(
                        &buffers,
                        group.count,
                        b12x_w4a16_capacity_rows(group.count)?,
                        u32::try_from(group.projections.gate.key.expert_id)
                            .context("packed W4A16 expert ID exceeds u32")?,
                        group_stream,
                    )
                    .context("launching streamed packed B12X W4A16 expert")?;
                library
                    .cuda_scatter_add_rows_bf16_weighted_to_f32_async(
                        group_output,
                        output_indices,
                        route_weights,
                        workspace.accumulator,
                        state.row_count,
                        group.count,
                        state.output_rows,
                        group_stream,
                    )
                    .context("accumulating streamed B12X routed expert output")?;
                library
                    .cuda_event_record(cuda_cache.b12x_lane_event(lane), group_stream)
                    .context("recording streamed B12X route lane completion")?;
            }
            state.lane_used_since_emit[lane] = true;
            completed_rows.extend_from_slice(&group.completed_rows);
            state.next_group += 1;
        }
    }
    state.received_rows = row_end;

    let output_row_bytes = state.output_dtype.row_stride_bytes(state.output_rows)?;
    let mut reduction_follower = false;
    let mut device_output = None;
    let mut response_completed_rows = completed_rows.clone();
    let mut response_output_bytes = 0_usize;
    let output = if completed_rows.is_empty() {
        Vec::new()
    } else {
        let completion_indices = completed_rows
            .iter()
            .map(|row| u32::try_from(*row).context("streamed completion row exceeds u32"))
            .collect::<Result<Vec<_>>>()?;
        let completion_f32_bytes = completed_rows
            .len()
            .checked_mul(state.output_rows)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .context("streamed completion F32 byte count overflow")?;
        let completion_output_bytes = completed_rows
            .len()
            .checked_mul(output_row_bytes)
            .context("streamed completion output byte count overflow")?;
        let buffers = cuda_cache.workspace.ensure_completion_buffers(
            Arc::clone(&library),
            &completion_indices,
            completion_f32_bytes,
            completion_output_bytes,
        )?;
        let reduction_shape = if state.spark_reduction {
            let dtype = reduction_output_dtype(
                cuda_cache
                    .spark_reduction_dtype()
                    .context("Spark-reduced route stream lost its backend")?,
            );
            let row_stride_bytes = dtype.row_stride_bytes(state.output_rows)?;
            let send_bytes = completed_rows
                .len()
                .checked_mul(row_stride_bytes)
                .context("Spark reduction send byte count overflow")?;
            let recv_bytes = send_bytes
                .checked_mul(
                    cuda_cache
                        .spark_reduction_world_size()
                        .context("Spark reduction backend has no world size")?
                        - 1,
                )
                .context("Spark reduction receive byte count overflow")?;
            Some((dtype, row_stride_bytes, send_bytes, recv_bytes))
        } else {
            None
        };
        let fused_fp8_reduction = reduction_shape
            .map(|(dtype, _, _, _)| {
                fused_fp8_reduction_enabled()
                    && fused_fp8_reduction_eligible(
                        dtype,
                        state.output_dtype,
                        &completed_rows,
                        state.row_count,
                    )
            })
            .unwrap_or(false);
        let rdma_row_sharded_reduction =
            state.spark_row_sharded_reduction && cuda_cache.spark_rdma_reduction_enabled();
        anyhow::ensure!(
            !state.spark_row_sharded_reduction || rdma_row_sharded_reduction || fused_fp8_reduction,
            "NCCL row-sharded Spark reduction requires full-row fused FP8 reduction"
        );
        let row_sharded_fp8_reduction =
            state.spark_row_sharded_reduction && !rdma_row_sharded_reduction && fused_fp8_reduction;
        let nccl_bf16_reduce = fused_fp8_reduction
            && !rdma_row_sharded_reduction
            && !row_sharded_fp8_reduction
            && nccl_bf16_reduce_enabled();
        let bf16_reduction_bytes = completed_rows
            .len()
            .checked_mul(state.output_rows)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("Spark BF16 reduce byte count overflow")?;
        let reduction_buffers = reduction_shape
            .map(|(_, _, send_bytes, recv_bytes)| {
                let (send_bytes, recv_bytes) = if nccl_bf16_reduce {
                    (bf16_reduction_bytes, bf16_reduction_bytes)
                } else {
                    (send_bytes, recv_bytes)
                };
                cuda_cache.workspace.ensure_reduction_buffers(
                    Arc::clone(&library),
                    send_bytes,
                    recv_bytes,
                )
            })
            .transpose()?;
        unsafe {
            if (packed_prefill_bf16_output.is_none() && packed_prefill_fp8_output.is_none())
                || rdma_row_sharded_reduction
            {
                library
                    .copy_host_buffer_h2d_async(
                        buffers.indices,
                        buffers.pinned_indices,
                        completion_indices.len() * std::mem::size_of::<u32>(),
                        completion_stream,
                    )
                    .context("copying streamed route completion indices")?;
            }
            for (lane, used) in state.lane_used_since_emit.iter_mut().enumerate() {
                if *used {
                    library
                        .cuda_stream_wait_event(completion_stream, cuda_cache.b12x_lane_event(lane))
                        .context("joining streamed B12X route lane")?;
                    *used = false;
                }
            }
            if reduction_shape.is_some() {
                if let Some(ticket) = state.collective_launch_ticket.as_mut() {
                    ticket
                        .wait_for_turn()
                        .context("waiting for deterministic Spark collective launch order")?;
                }
            }
            if rdma_row_sharded_reduction {
                let (reduction_dtype, reduction_row_bytes, _, _) =
                    reduction_shape.expect("Spark RDMA reduction has a shape");
                let reduction_buffers =
                    reduction_buffers.expect("Spark RDMA reduction has workspace buffers");
                let request_id = state
                    .collective_request_id
                    .context("Spark RDMA route stream has no registered collective request")?;
                let (row_start, local_rows) = execute_rdma_row_sharded_reduction(
                    cuda_cache,
                    &library,
                    request_id,
                    reduction_dtype,
                    reduction_row_bytes,
                    workspace,
                    buffers,
                    reduction_buffers,
                    packed_prefill_bf16_output,
                    completed_rows.len(),
                    state.output_rows,
                    state.output_dtype,
                    output_row_bytes,
                    completion_stream,
                    packed_prefill_timeline.as_ref(),
                )?;
                response_completed_rows = (row_start..row_start + local_rows).collect();
                response_output_bytes = local_rows
                    .checked_mul(output_row_bytes)
                    .context("Spark RDMA response byte count overflow")?;
            } else if let (
                Some((reduction_dtype, reduction_row_bytes, reduction_bytes, _)),
                Some(reduction_buffers),
            ) = (reduction_shape, reduction_buffers)
            {
                let reduction = cuda_cache
                    .spark_reduction
                    .as_ref()
                    .expect("NCCL reduction shape requires a communicator");
                if row_sharded_fp8_reduction {
                    if let Some(packed_bf16) = packed_prefill_bf16_output {
                        library
                            .cuda_bf16_rows_to_fp8_e4m3_row_scaled_async(
                                packed_bf16,
                                reduction_buffers.send,
                                completed_rows.len(),
                                state.output_rows,
                                reduction_row_bytes,
                                completion_stream,
                            )
                            .context("packing contiguous BF16 Spark reduction rows")?;
                    } else {
                        pack_streamed_completion_rows(
                            &library,
                            reduction_dtype,
                            workspace.accumulator,
                            state.row_count,
                            buffers.indices,
                            buffers.f32_output,
                            reduction_buffers.send,
                            completed_rows.len(),
                            state.output_rows,
                            reduction_row_bytes,
                            completion_stream,
                        )
                        .context("packing row-sharded Spark reduction rows")?;
                    }
                    if let Some(timeline) = packed_prefill_timeline.as_ref() {
                        timeline.record(
                            timeline.metadata_ready.as_ref(),
                            completion_stream,
                            "packed FP8 end",
                        )?;
                    }
                    reduction
                        .communicator()
                        .row_all_to_all_u8_async(
                            reduction_buffers.send,
                            reduction_buffers.recv,
                            completed_rows.len(),
                            reduction_row_bytes,
                            completion_stream,
                        )
                        .context("exchanging row-sharded Spark expert outputs")?;
                    if let Some(timeline) = packed_prefill_timeline.as_ref() {
                        timeline.record(
                            timeline.routes_ready.as_ref(),
                            completion_stream,
                            "packed collective end",
                        )?;
                    }
                    let rank = reduction.communicator().rank();
                    let world_size = reduction.communicator().world_size();
                    let (row_start, local_rows) =
                        balanced_row_partition(completed_rows.len(), world_size, rank)?;
                    let local_peer_bytes = local_rows
                        .checked_mul(reduction_row_bytes)
                        .context("row-sharded peer payload byte count overflow")?;
                    if let Some(packed_bf16) = packed_prefill_bf16_output {
                        let local_bf16_offset = row_start
                            .checked_mul(state.output_rows)
                            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
                            .context("row-sharded local BF16 offset overflow")?;
                        let local_bf16_bytes = local_rows
                            .checked_mul(state.output_rows)
                            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
                            .context("row-sharded local BF16 byte count overflow")?;
                        let local_bf16 = route_device_buffer_slice(
                            packed_bf16,
                            local_bf16_offset,
                            local_bf16_bytes,
                        )?;
                        library
                            .cuda_combine_bf16_fp8_e4m3_row_scaled_to_fp8_async(
                                local_bf16,
                                reduction_buffers.recv,
                                local_peer_bytes,
                                world_size - 1,
                                reduction_row_bytes,
                                buffers.output,
                                local_rows,
                                state.output_rows,
                                output_row_bytes,
                                completion_stream,
                            )
                            .context("combining BF16 and peer FP8 Spark responses")?;
                    } else {
                        let local_f32_offset = row_start
                            .checked_mul(state.output_rows)
                            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
                            .context("row-sharded local accumulator offset overflow")?;
                        let local_f32_bytes = local_rows
                            .checked_mul(state.output_rows)
                            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
                            .context("row-sharded local accumulator byte count overflow")?;
                        let local_accumulator = route_device_buffer_slice(
                            workspace.accumulator,
                            local_f32_offset,
                            local_f32_bytes,
                        )?;
                        library
                            .cuda_combine_fp8_e4m3_row_scaled_to_fp8_async(
                                local_accumulator,
                                reduction_buffers.recv,
                                local_peer_bytes,
                                world_size - 1,
                                reduction_row_bytes,
                                buffers.output,
                                local_rows,
                                state.output_rows,
                                output_row_bytes,
                                completion_stream,
                            )
                            .context("combining row-sharded Spark FP8 responses")?;
                    }
                    if let Some(timeline) = packed_prefill_timeline.as_ref() {
                        timeline.record(
                            timeline.pack_ready.as_ref(),
                            completion_stream,
                            "packed combine end",
                        )?;
                    }
                    response_completed_rows = (row_start..row_start + local_rows).collect();
                    response_output_bytes = local_rows
                        .checked_mul(output_row_bytes)
                        .context("row-sharded response byte count overflow")?;
                } else if nccl_bf16_reduce {
                    pack_streamed_completion_rows(
                        &library,
                        RouteStreamingOutputDtype::Bf16,
                        workspace.accumulator,
                        state.row_count,
                        buffers.indices,
                        buffers.f32_output,
                        reduction_buffers.send,
                        completed_rows.len(),
                        state.output_rows,
                        state.output_rows * std::mem::size_of::<u16>(),
                        completion_stream,
                    )
                    .context("packing Spark BF16 reduce rows")?;
                    reduction
                        .communicator()
                        .reduce_bf16_async(
                            reduction_buffers.send,
                            reduction_buffers.recv,
                            completed_rows.len() * state.output_rows,
                            reduction.root_rank,
                            completion_stream,
                        )
                        .context("reducing Spark BF16 shard outputs")?;
                    if reduction.is_root() {
                        library
                            .cuda_zero_f32_async(
                                buffers.f32_output,
                                completed_rows.len() * state.output_rows,
                                completion_stream,
                            )
                            .context("zeroing Spark BF16 reduce conversion buffer")?;
                        library
                            .cuda_scatter_add_rows_bf16_to_f32_async(
                                reduction_buffers.recv,
                                buffers.indices,
                                buffers.f32_output,
                                completed_rows.len(),
                                completed_rows.len(),
                                state.output_rows,
                                completion_stream,
                            )
                            .context("converting Spark BF16 reduce output to F32")?;
                        library
                            .cuda_gather_rows_f32_to_fp8_e4m3_row_scaled_async(
                                buffers.f32_output,
                                completed_rows.len(),
                                buffers.indices,
                                buffers.output,
                                completed_rows.len(),
                                state.output_rows,
                                output_row_bytes,
                                completion_stream,
                            )
                            .context("packing Spark BF16 reduce output as FP8")?;
                    } else {
                        reduction_follower = true;
                    }
                } else {
                    if !reduction.is_root() {
                        pack_streamed_completion_rows(
                            &library,
                            reduction_dtype,
                            workspace.accumulator,
                            state.row_count,
                            buffers.indices,
                            buffers.f32_output,
                            reduction_buffers.send,
                            completed_rows.len(),
                            state.output_rows,
                            reduction_row_bytes,
                            completion_stream,
                        )
                        .context("packing Spark reduction completion rows")?;
                    }
                    reduction
                        .communicator()
                        .gather_u8_async(
                            reduction_buffers.send,
                            reduction_buffers.recv,
                            reduction_bytes,
                            reduction.root_rank,
                            completion_stream,
                        )
                        .context("gathering Spark expert shard outputs")?;
                    if reduction.is_root() {
                        if fused_fp8_reduction {
                            library
                                .cuda_combine_fp8_e4m3_row_scaled_to_fp8_async(
                                    workspace.accumulator,
                                    reduction_buffers.recv,
                                    reduction_bytes,
                                    reduction.communicator().world_size() - 1,
                                    reduction_row_bytes,
                                    buffers.output,
                                    completed_rows.len(),
                                    state.output_rows,
                                    output_row_bytes,
                                    completion_stream,
                                )
                                .context("fusing Spark FP8 shard responses")?;
                        } else {
                            for peer_index in 0..reduction.communicator().world_size() - 1 {
                                let peer_payload = route_device_buffer_slice(
                                    reduction_buffers.recv,
                                    peer_index * reduction_bytes,
                                    reduction_bytes,
                                )?;
                                scatter_add_streamed_reduction_rows(
                                    &library,
                                    reduction_dtype,
                                    peer_payload,
                                    reduction_row_bytes,
                                    buffers.indices,
                                    workspace.accumulator,
                                    state.row_count,
                                    completed_rows.len(),
                                    state.output_rows,
                                    completion_stream,
                                )
                                .context("accumulating a Spark expert shard response")?;
                            }
                            pack_streamed_completion_rows(
                                &library,
                                state.output_dtype,
                                workspace.accumulator,
                                state.row_count,
                                buffers.indices,
                                buffers.f32_output,
                                buffers.output,
                                completed_rows.len(),
                                state.output_rows,
                                output_row_bytes,
                                completion_stream,
                            )
                            .context("packing Spark-reduced coordinator response rows")?;
                        }
                    } else {
                        reduction_follower = true;
                    }
                }
            } else if let Some(packed_bf16) = packed_prefill_bf16_output {
                library
                    .cuda_bf16_rows_to_fp8_e4m3_row_scaled_async(
                        packed_bf16,
                        buffers.output,
                        completed_rows.len(),
                        state.output_rows,
                        output_row_bytes,
                        completion_stream,
                    )
                    .context("packing direct packed W4A16 response")?;
            } else if packed_prefill_fp8_output.is_none() {
                pack_streamed_completion_rows(
                    &library,
                    state.output_dtype,
                    workspace.accumulator,
                    state.row_count,
                    buffers.indices,
                    buffers.f32_output,
                    buffers.output,
                    completed_rows.len(),
                    state.output_rows,
                    output_row_bytes,
                    completion_stream,
                )
                .context("packing streamed route completion rows")?;
            }
            if !reduction_follower {
                if response_output_bytes == 0 {
                    response_output_bytes = completion_output_bytes;
                }
                if retain_device_output {
                    let retained_output = if let Some(output) = packed_prefill_fp8_output {
                        route_device_buffer_slice(output, 0, response_output_bytes)?
                    } else if let Some(target) = response_device_target {
                        let target = route_device_buffer_slice(target, 0, response_output_bytes)?;
                        library
                            .copy_d2d_async(
                                target,
                                buffers.output,
                                response_output_bytes,
                                completion_stream,
                            )
                            .context(
                                "copying streamed route completion into mapped response slot",
                            )?;
                        target
                    } else {
                        route_device_buffer_slice(buffers.output, 0, response_output_bytes)?
                    };
                    if let Some(timeline) = packed_prefill_timeline.as_ref() {
                        timeline.record(
                            timeline.host_copy_ready.as_ref(),
                            completion_stream,
                            "packed completion end",
                        )?;
                    }
                    library
                        .cuda_stream_synchronize(completion_stream)
                        .context("synchronizing device-backed streamed route completion rows")?;
                    if let Some(timeline) = packed_prefill_timeline.as_ref() {
                        let compute_ms = timeline.elapsed_between(
                            timeline.start.as_ref(),
                            timeline.hidden_ready.as_ref(),
                            "packed compute",
                        )?;
                        let fp8_pack_ms = timeline.elapsed_between(
                            timeline.hidden_ready.as_ref(),
                            timeline.metadata_ready.as_ref(),
                            "packed FP8 pack",
                        )?;
                        let collective_ms = timeline.elapsed_between(
                            timeline.metadata_ready.as_ref(),
                            timeline.routes_ready.as_ref(),
                            "packed collective",
                        )?;
                        let combine_ms = timeline.elapsed_between(
                            timeline.routes_ready.as_ref(),
                            timeline.pack_ready.as_ref(),
                            "packed combine",
                        )?;
                        let tail_ms = timeline.elapsed_between(
                            timeline.pack_ready.as_ref(),
                            timeline.host_copy_ready.as_ref(),
                            "packed completion tail",
                        )?;
                        let total_ms = timeline.elapsed_between(
                            timeline.start.as_ref(),
                            timeline.host_copy_ready.as_ref(),
                            "packed total",
                        )?;
                        eprintln!(
                            "real_nvfp4_route_packed_prefill_cuda_timing layer_id={} rows={} compute_ms={:.3} fp8_pack_ms={:.3} collective_ms={:.3} combine_ms={:.3} tail_ms={:.3} total_ms={:.3}",
                            state.layer_id,
                            state.row_count,
                            compute_ms,
                            fp8_pack_ms,
                            collective_ms,
                            combine_ms,
                            tail_ms,
                            total_ms,
                        );
                    }
                    device_output = Some(retained_output);
                } else {
                    library
                        .copy_d2h_host_buffer_async(
                            buffers.pinned_output,
                            buffers.output,
                            response_output_bytes,
                            completion_stream,
                        )
                        .context("copying streamed route completion rows to host")?;
                    library
                        .cuda_stream_synchronize(completion_stream)
                        .context("synchronizing streamed route completion rows")?;
                }
            } else if reduction_shape.is_some() && final_frame {
                library
                    .cuda_stream_synchronize(completion_stream)
                    .context("synchronizing follower Spark collective completion")?;
            }
            if reduction_shape.is_some() && final_frame {
                if let Some(ticket) = state.collective_launch_ticket.as_mut() {
                    ticket
                        .finish()
                        .context("advancing deterministic Spark collective launch order")?;
                }
            }
        }
        if reduction_follower || device_output.is_some() {
            Vec::new()
        } else {
            cuda_cache
                .workspace
                .completion_output_slice(response_output_bytes)?
                .to_vec()
        }
    };
    state.emitted_rows = state
        .emitted_rows
        .checked_add(completed_rows.len())
        .context("streamed route emitted row count overflow")?;
    let complete = final_frame;
    if complete {
        anyhow::ensure!(
            state.next_group == state.scheduled_group_count,
            "streamed NVFP4 route completed with {}/{} groups launched",
            state.next_group,
            state.scheduled_group_count
        );
        anyhow::ensure!(
            state.emitted_rows == state.row_count,
            "streamed NVFP4 route emitted {}/{} rows",
            state.emitted_rows,
            state.row_count
        );
        anyhow::ensure!(
            state.lane_used_since_emit.iter().all(|used| !*used),
            "streamed NVFP4 route completed with unjoined lanes"
        );
    }
    Ok(RouteNvfp4IngressStreamChunk {
        completed_rows: response_completed_rows,
        output,
        device_output,
        reduction_follower,
        complete,
    })
}

struct RouteComputation {
    outputs: Vec<f32>,
    kernel_backend: &'static str,
}

pub(in crate::commands::real_full) fn execute_nvfp4_route(
    catalog: &TensorCatalog,
    layer_id: usize,
    hidden: &[f32],
    route: &ScoredRoute,
    intermediate_rows: usize,
    output_rows: usize,
) -> Result<RouteExecution> {
    execute_nvfp4_route_with_projection_loader(
        catalog,
        layer_id,
        hidden,
        route,
        intermediate_rows,
        output_rows,
        &mut |catalog, layer_id, expert_id, projection, row_count| {
            load_routed_quant_projection(catalog, layer_id, expert_id, projection, row_count)
                .map(Arc::new)
        },
    )
}

pub(in crate::commands::real_full) fn execute_nvfp4_route_cached(
    catalog: &TensorCatalog,
    layer_id: usize,
    hidden: &[f32],
    route: &ScoredRoute,
    intermediate_rows: usize,
    output_rows: usize,
    cache: &mut RouteTensorCache,
) -> Result<RouteExecution> {
    cache.prepare_layer(layer_id);
    execute_nvfp4_route_with_projection_loader(
        catalog,
        layer_id,
        hidden,
        route,
        intermediate_rows,
        output_rows,
        &mut |catalog, layer_id, expert_id, projection, row_count| {
            load_routed_quant_projection_cached(
                catalog, layer_id, expert_id, projection, row_count, cache,
            )
        },
    )
}

pub(in crate::commands::real_full) fn execute_nvfp4_route_rows_bf16_accumulated_cached(
    catalog: &TensorCatalog,
    layer_id: usize,
    hidden_bf16: &[u8],
    hidden_dim: usize,
    hidden_row_stride_bytes: usize,
    row_routes: &[Vec<(ScoredRoute, usize)>],
    output_rows: usize,
    cache: &mut RouteTensorCache,
) -> Result<RouteBf16AccumulatedExecution> {
    let execution = execute_nvfp4_route_rows_bf16_accumulated_cached_inner(
        catalog,
        layer_id,
        RouteHiddenBatch::HostBf16(hidden_bf16),
        hidden_dim,
        hidden_row_stride_bytes,
        row_routes,
        output_rows,
        cache,
        false,
        RouteStreamingOutputDtype::Bf16,
        None,
    )?;
    Ok(RouteBf16AccumulatedExecution {
        output_bf16: execution.output_bf16.with_context(|| {
            format!("CUDA NVFP4 route layer {layer_id} did not retain host BF16 output")
        })?,
        completion_slices: execution.completion_slices,
        kernel_backend: execution.kernel_backend,
    })
}

pub(in crate::commands::real_full) fn execute_nvfp4_route_rows_bf16_accumulated_streaming_cached(
    catalog: &TensorCatalog,
    layer_id: usize,
    hidden_bf16: &[u8],
    hidden_dim: usize,
    hidden_row_stride_bytes: usize,
    row_routes: &[Vec<(ScoredRoute, usize)>],
    output_rows: usize,
    cache: &mut RouteTensorCache,
    output_dtype: RouteStreamingOutputDtype,
    emit: &mut dyn FnMut(&[usize], &[u8]) -> Result<()>,
) -> Result<RouteBf16AccumulatedStreamingExecution> {
    let execution = execute_nvfp4_route_rows_bf16_accumulated_cached_inner(
        catalog,
        layer_id,
        RouteHiddenBatch::HostBf16(hidden_bf16),
        hidden_dim,
        hidden_row_stride_bytes,
        row_routes,
        output_rows,
        cache,
        false,
        output_dtype,
        Some(emit),
    )?;
    Ok(RouteBf16AccumulatedStreamingExecution {
        completion_slices: execution.completion_slices,
        kernel_backend: execution.kernel_backend,
    })
}

pub(in crate::commands::real_full) fn execute_nvfp4_route_rows_nvfp4_accumulated_cached(
    catalog: &TensorCatalog,
    layer_id: usize,
    hidden_nvfp4: &[u8],
    hidden_dim: usize,
    hidden_row_stride_bytes: usize,
    row_routes: &[Vec<(ScoredRoute, usize)>],
    output_rows: usize,
    cache: &mut RouteTensorCache,
) -> Result<RouteBf16AccumulatedExecution> {
    let execution = execute_nvfp4_route_rows_bf16_accumulated_cached_inner(
        catalog,
        layer_id,
        RouteHiddenBatch::HostNvfp4(hidden_nvfp4),
        hidden_dim,
        hidden_row_stride_bytes,
        row_routes,
        output_rows,
        cache,
        false,
        RouteStreamingOutputDtype::Bf16,
        None,
    )?;
    Ok(RouteBf16AccumulatedExecution {
        output_bf16: execution.output_bf16.with_context(|| {
            format!("CUDA NVFP4 route layer {layer_id} did not retain host BF16 output")
        })?,
        completion_slices: execution.completion_slices,
        kernel_backend: execution.kernel_backend,
    })
}

pub(in crate::commands::real_full) fn execute_nvfp4_route_rows_nvfp4_accumulated_streaming_cached(
    catalog: &TensorCatalog,
    layer_id: usize,
    hidden_nvfp4: &[u8],
    hidden_dim: usize,
    hidden_row_stride_bytes: usize,
    row_routes: &[Vec<(ScoredRoute, usize)>],
    output_rows: usize,
    cache: &mut RouteTensorCache,
    output_dtype: RouteStreamingOutputDtype,
    emit: &mut dyn FnMut(&[usize], &[u8]) -> Result<()>,
) -> Result<RouteBf16AccumulatedStreamingExecution> {
    let execution = execute_nvfp4_route_rows_bf16_accumulated_cached_inner(
        catalog,
        layer_id,
        RouteHiddenBatch::HostNvfp4(hidden_nvfp4),
        hidden_dim,
        hidden_row_stride_bytes,
        row_routes,
        output_rows,
        cache,
        false,
        output_dtype,
        Some(emit),
    )?;
    Ok(RouteBf16AccumulatedStreamingExecution {
        completion_slices: execution.completion_slices,
        kernel_backend: execution.kernel_backend,
    })
}

pub(in crate::commands::real_full) fn execute_nvfp4_route_rows_nvfp4_accumulated_cached_device_output(
    catalog: &TensorCatalog,
    layer_id: usize,
    hidden_nvfp4: &[u8],
    hidden_dim: usize,
    hidden_row_stride_bytes: usize,
    row_routes: &[Vec<(ScoredRoute, usize)>],
    output_rows: usize,
    cache: &mut RouteTensorCache,
) -> Result<RouteBf16AccumulatedDeviceExecution> {
    let execution = execute_nvfp4_route_rows_bf16_accumulated_cached_inner(
        catalog,
        layer_id,
        RouteHiddenBatch::HostNvfp4(hidden_nvfp4),
        hidden_dim,
        hidden_row_stride_bytes,
        row_routes,
        output_rows,
        cache,
        true,
        RouteStreamingOutputDtype::Bf16,
        None,
    )?;
    Ok(RouteBf16AccumulatedDeviceExecution {
        output_device: execution.output_device.with_context(|| {
            format!("CUDA NVFP4 route layer {layer_id} did not retain device BF16 output")
        })?,
        kernel_backend: execution.kernel_backend,
    })
}

#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) fn execute_nvfp4_route_rows_bf16_accumulated_cached_device_output(
    catalog: &TensorCatalog,
    layer_id: usize,
    hidden_bf16: &[u8],
    hidden_dim: usize,
    hidden_row_stride_bytes: usize,
    row_routes: &[Vec<(ScoredRoute, usize)>],
    output_rows: usize,
    cache: &mut RouteTensorCache,
) -> Result<RouteBf16AccumulatedDeviceExecution> {
    let execution = execute_nvfp4_route_rows_bf16_accumulated_cached_inner(
        catalog,
        layer_id,
        RouteHiddenBatch::HostBf16(hidden_bf16),
        hidden_dim,
        hidden_row_stride_bytes,
        row_routes,
        output_rows,
        cache,
        true,
        RouteStreamingOutputDtype::Bf16,
        None,
    )?;
    Ok(RouteBf16AccumulatedDeviceExecution {
        output_device: execution.output_device.with_context(|| {
            format!("CUDA NVFP4 route layer {layer_id} did not retain device BF16 output")
        })?,
        kernel_backend: execution.kernel_backend,
    })
}

pub(in crate::commands::real_full) fn execute_nvfp4_route_rows_bf16_accumulated_cached_device_input_device_output(
    catalog: &TensorCatalog,
    layer_id: usize,
    hidden_device: &DeviceBf16Output,
    hidden_host_for_validation: Option<&[u8]>,
    hidden_dim: usize,
    hidden_row_stride_bytes: usize,
    row_routes: &[Vec<(ScoredRoute, usize)>],
    output_rows: usize,
    cache: &mut RouteTensorCache,
) -> Result<RouteBf16AccumulatedDeviceExecution> {
    let execution = execute_nvfp4_route_rows_bf16_accumulated_cached_inner(
        catalog,
        layer_id,
        RouteHiddenBatch::DeviceBf16 {
            output: hidden_device,
            host_for_validation: hidden_host_for_validation,
        },
        hidden_dim,
        hidden_row_stride_bytes,
        row_routes,
        output_rows,
        cache,
        true,
        RouteStreamingOutputDtype::Bf16,
        None,
    )?;
    Ok(RouteBf16AccumulatedDeviceExecution {
        output_device: execution.output_device.with_context(|| {
            format!("CUDA NVFP4 route layer {layer_id} did not retain device BF16 output")
        })?,
        kernel_backend: execution.kernel_backend,
    })
}

#[allow(clippy::too_many_arguments)]
fn execute_nvfp4_route_with_projection_loader(
    catalog: &TensorCatalog,
    layer_id: usize,
    hidden: &[f32],
    route: &ScoredRoute,
    intermediate_rows: usize,
    output_rows: usize,
    load_projection: &mut impl FnMut(
        &TensorCatalog,
        usize,
        usize,
        &'static str,
        usize,
    ) -> Result<Arc<RoutedQuantProjection>>,
) -> Result<RouteExecution> {
    let expert_id = route.expert_id;
    let gate = load_projection(catalog, layer_id, expert_id, "gate_proj", intermediate_rows)?;
    let up = load_projection(catalog, layer_id, expert_id, "up_proj", intermediate_rows)?;
    let down = load_projection(catalog, layer_id, expert_id, "down_proj", output_rows)?;

    validate_packed_nvfp4_projection_width(
        "gate_proj",
        &gate.weight,
        &gate.weight_scale,
        hidden.len(),
    )?;
    validate_packed_nvfp4_projection_width("up_proj", &up.weight, &up.weight_scale, hidden.len())?;
    validate_packed_nvfp4_projection_width(
        "down_proj",
        &down.weight,
        &down.weight_scale,
        intermediate_rows,
    )?;

    let gate_input_scale = first_f32_scalar(&gate.input_scale.info.name, &gate.input_scale.bytes)?;
    let up_input_scale = first_f32_scalar(&up.input_scale.info.name, &up.input_scale.bytes)?;
    let down_input_scale = first_f32_scalar(&down.input_scale.info.name, &down.input_scale.bytes)?;
    let gate_scale_2 =
        first_f32_scalar(&gate.weight_scale_2.info.name, &gate.weight_scale_2.bytes)?;
    let up_scale_2 = first_f32_scalar(&up.weight_scale_2.info.name, &up.weight_scale_2.bytes)?;
    let down_scale_2 =
        first_f32_scalar(&down.weight_scale_2.info.name, &down.weight_scale_2.bytes)?;
    for (name, value) in [
        (&gate.input_scale.info.name, gate_input_scale),
        (&up.input_scale.info.name, up_input_scale),
        (&down.input_scale.info.name, down_input_scale),
        (&gate.weight_scale_2.info.name, gate_scale_2),
        (&up.weight_scale_2.info.name, up_scale_2),
        (&down.weight_scale_2.info.name, down_scale_2),
    ] {
        validate_finite_route_scalar(name, value)?;
    }

    let computation = cpu_nvfp4_route_computation(
        hidden,
        route,
        &gate,
        &up,
        &down,
        intermediate_rows,
        output_rows,
        gate_scale_2,
        up_scale_2,
        down_scale_2,
    )?;

    let weight_bytes_read = gate.weight.bytes.len() as u64
        + up.weight.bytes.len() as u64
        + down.weight.bytes.len() as u64;
    let quant_metadata_bytes_read = gate.weight_scale.bytes.len() as u64
        + up.weight_scale.bytes.len() as u64
        + down.weight_scale.bytes.len() as u64
        + gate.input_scale.bytes.len() as u64
        + up.input_scale.bytes.len() as u64
        + down.input_scale.bytes.len() as u64
        + gate.weight_scale_2.bytes.len() as u64
        + up.weight_scale_2.bytes.len() as u64
        + down.weight_scale_2.bytes.len() as u64;
    Ok(RouteExecution {
        outputs: computation.outputs,
        weight_bytes_read,
        quant_metadata_bytes_read,
        kernel_backend: computation.kernel_backend,
    })
}

fn execute_nvfp4_route_rows_bf16_accumulated_cached_inner(
    catalog: &TensorCatalog,
    layer_id: usize,
    hidden: RouteHiddenBatch<'_>,
    hidden_dim: usize,
    hidden_row_stride_bytes: usize,
    row_routes: &[Vec<(ScoredRoute, usize)>],
    output_rows: usize,
    cache: &mut RouteTensorCache,
    retain_device_output: bool,
    streaming_output_dtype: RouteStreamingOutputDtype,
    mut completion_emit: Option<&mut dyn FnMut(&[usize], &[u8]) -> Result<()>>,
) -> Result<RouteBf16AccumulatedInnerExecution> {
    if !cuda_reference_kernels_enabled() {
        anyhow::bail!(
            "BF16 NVFP4 route accumulation requires {REAL_FULL_CUDA_REFERENCE_KERNELS_ENV}=1"
        );
    }
    cache.prepare_layer(layer_id);
    let nvfp4_hidden_payload = hidden.is_nvfp4_payload();
    let logical_hidden_bytes = if nvfp4_hidden_payload {
        anyhow::ensure!(
            hidden_dim > 0 && hidden_dim % 16 == 0,
            "NVFP4 route hidden width must be a nonzero multiple of 16, got {hidden_dim}"
        );
        hidden_dim / 2 + hidden_dim / 16
    } else {
        hidden_dim
            .checked_mul(std::mem::size_of::<u16>())
            .context("BF16 NVFP4 accumulated route hidden byte count overflow")?
    };
    if hidden_row_stride_bytes < logical_hidden_bytes {
        anyhow::bail!(
            "NVFP4 accumulated route hidden row stride {hidden_row_stride_bytes} is smaller than logical row bytes {logical_hidden_bytes}"
        );
    }
    if !nvfp4_hidden_payload && hidden_row_stride_bytes % std::mem::size_of::<u16>() != 0 {
        anyhow::bail!(
            "BF16 NVFP4 accumulated route hidden row stride {hidden_row_stride_bytes} is not BF16-aligned"
        );
    }
    let row_count = row_routes.len();
    let hidden_bytes = row_count
        .checked_mul(hidden_row_stride_bytes)
        .context("CUDA NVFP4 accumulated hidden batch byte count overflow")?;
    hidden.validate(row_count, hidden_dim, hidden_row_stride_bytes, hidden_bytes)?;
    if nvfp4_hidden_payload && cuda_route_validation_enabled() {
        anyhow::bail!("CUDA route CPU validation does not accept prequantized hidden payloads");
    }
    let hidden_logical = hidden.host_slice(hidden_bytes);
    let mut kernel_backend = hidden.kernel_backend();
    let output_row_bytes = output_rows
        .checked_mul(std::mem::size_of::<u16>())
        .context("CUDA NVFP4 accumulated BF16 route output row byte count overflow")?;
    let output_bytes = row_count
        .checked_mul(output_row_bytes)
        .context("CUDA NVFP4 accumulated BF16 route output batch byte count overflow")?;
    let streaming_output_row_bytes = streaming_output_dtype.row_stride_bytes(output_rows)?;
    let streaming_output_bytes = row_count
        .checked_mul(streaming_output_row_bytes)
        .context("CUDA NVFP4 accumulated streaming route output byte count overflow")?;
    anyhow::ensure!(
        completion_emit.is_some() || streaming_output_dtype == RouteStreamingOutputDtype::Bf16,
        "low-precision route output requires streaming completion emission"
    );
    let route_count = row_routes.iter().map(Vec::len).sum::<usize>();
    if route_count == 0 {
        let output_bf16 = if let Some(emit) = completion_emit.as_deref_mut() {
            let completed_rows = (0..row_count).collect::<Vec<_>>();
            let mut output = vec![0_u8; streaming_output_bytes];
            if streaming_output_dtype == RouteStreamingOutputDtype::Fp8E4m3RowScaled {
                for row_index in 0..row_count {
                    let scale_offset = row_index
                        .checked_mul(streaming_output_row_bytes)
                        .and_then(|offset| offset.checked_add(output_rows))
                        .context("empty route FP8 response scale offset overflow")?;
                    output[scale_offset..scale_offset + std::mem::size_of::<f32>()]
                        .copy_from_slice(&1.0_f32.to_le_bytes());
                }
            }
            if !completed_rows.is_empty() {
                emit(&completed_rows, &output)?;
            }
            cuda_route_validation_enabled().then(|| vec![0_u8; output_bytes])
        } else {
            (!retain_device_output || cuda_route_validation_enabled())
                .then(|| vec![0_u8; output_bytes])
        };
        let output_device = if retain_device_output && output_bytes > 0 {
            let retained = device_bf16_output_uninitialized(
                row_count,
                output_rows,
                CUDA_REFERENCE_NVFP4_ROUTE_BF16_ACCUMULATED_DEVICE_OUTPUT_BACKEND,
                "NVFP4 BF16 empty route retained device output",
            )?;
            cuda_native_library()?
                .cuda_zero_bytes(retained.buffer(), output_bytes)
                .context("zeroing empty NVFP4 BF16 route output retained device buffer")?;
            Some(retained)
        } else {
            None
        };
        return Ok(RouteBf16AccumulatedInnerExecution {
            output_bf16,
            output_device,
            completion_slices: (row_count > 0)
                .then(|| (0..row_count).collect())
                .into_iter()
                .collect(),
            kernel_backend,
        });
    }

    let timing_enabled = route_stage_timing_enabled();
    let timing_total_started = Instant::now();
    let hidden_device_input = matches!(&hidden, RouteHiddenBatch::DeviceBf16 { .. });
    let projection_source_started = Instant::now();
    let mut loaded_routes = Vec::with_capacity(route_count);
    let mut loaded_projection_groups = HashMap::new();
    for (row_index, routes) in row_routes.iter().enumerate() {
        for (route, intermediate_rows) in routes {
            let projections = load_bf16_route_projections_for_group_cached(
                catalog,
                layer_id,
                route,
                *intermediate_rows,
                output_rows,
                hidden_dim,
                cache,
                &mut loaded_projection_groups,
            )?;
            loaded_routes.push(LoadedBf16Route {
                row_index,
                route: route.clone(),
                intermediate_rows: *intermediate_rows,
                projections,
            });
        }
    }
    let projection_source_ms = elapsed_ms(projection_source_started);
    let plan_started = Instant::now();
    let completion_plan_entries = loaded_routes
        .iter()
        .map(|loaded_route| CompletionRoutePlanEntry {
            row_index: loaded_route.row_index,
            expert_id: loaded_route.route.expert_id,
            intermediate_rows: loaded_route.intermediate_rows,
        })
        .collect::<Vec<_>>();
    let completion_plan =
        plan_completion_first_routes(&completion_plan_entries, row_count, B12X_SPARK_AOT_MAX_ROWS)?;
    let mut loaded_route_slots = loaded_routes.into_iter().map(Some).collect::<Vec<_>>();
    let mut loaded_routes = Vec::with_capacity(route_count);
    let mut route_groups = Vec::with_capacity(completion_plan.groups.len());
    for planned_group in completion_plan.groups {
        let group_start = loaded_routes.len();
        for route_index in planned_group.route_indices {
            loaded_routes.push(
                loaded_route_slots[route_index]
                    .take()
                    .expect("completion route is scheduled exactly once"),
            );
        }
        let first = &loaded_routes[group_start];
        route_groups.push(LoadedBf16RouteGroup {
            start: group_start,
            intermediate_rows: first.intermediate_rows,
            count: loaded_routes.len() - group_start,
            projections: first.projections.clone(),
            completed_rows: planned_group.completed_rows,
        });
    }
    debug_assert!(loaded_route_slots.iter().all(Option::is_none));
    let mut scatter_indices = Vec::with_capacity(route_count);
    let mut route_weights = Vec::with_capacity(route_count);
    for loaded_route in &loaded_routes {
        scatter_indices.push(u32::try_from(loaded_route.row_index).with_context(|| {
            format!(
                "CUDA NVFP4 route row index {} exceeds u32",
                loaded_route.row_index
            )
        })?);
        route_weights.push(loaded_route.route.normalized_weight);
    }
    let route_group_count = route_groups.len();
    let completion_slice_count = route_groups
        .iter()
        .filter(|group| !group.completed_rows.is_empty())
        .count();
    let first_completion_group = route_groups
        .iter()
        .position(|group| !group.completed_rows.is_empty())
        .map(|index| index + 1)
        .unwrap_or(0);
    let first_completion_rows = route_groups
        .iter()
        .find(|group| !group.completed_rows.is_empty())
        .map(|group| group.completed_rows.len())
        .unwrap_or(0);
    let indexed_completion_slices = route_groups
        .iter()
        .enumerate()
        .filter(|(_, group)| !group.completed_rows.is_empty())
        .map(|(group_index, group)| (group_index, group.completed_rows.clone()))
        .collect::<Vec<_>>();
    let completion_slices = indexed_completion_slices
        .iter()
        .map(|(_, rows)| rows.clone())
        .collect::<Vec<_>>();
    let max_route_group_count = route_groups
        .iter()
        .map(|group| group.count)
        .max()
        .unwrap_or(0);
    let use_grouped_route_launches =
        should_use_grouped_route_launches(row_count, route_group_count);
    let grouped_decode_shape = b12x_spark_grouped_decode_enabled()
        && nvfp4_hidden_payload
        && row_count == 1
        && route_count == 8
        && route_groups.len() == 8
        && route_groups
            .iter()
            .all(|group| group.count == 1 && group.intermediate_rows == 512);
    let streaming_completion_enabled = completion_emit.is_some();
    if streaming_completion_enabled && !use_grouped_route_launches {
        anyhow::bail!("streaming CUDA NVFP4 route completion requires grouped route launches");
    }
    let (streaming_completion_group_indices, response_completion_slices) =
        if streaming_completion_enabled {
            coalesce_streaming_completion_slices(
                &indexed_completion_slices,
                STREAMING_FIRST_RESPONSE_ROWS,
                STREAMING_RESPONSE_MAX_ROWS,
            )?
        } else {
            (Vec::new(), completion_slices.clone())
        };
    let response_completion_slice_count = response_completion_slices.len();
    let completion_indices = response_completion_slices
        .iter()
        .flatten()
        .map(|row_index| {
            u32::try_from(*row_index)
                .with_context(|| format!("route completion row {row_index} exceeds u32"))
        })
        .collect::<Result<Vec<_>>>()?;
    if streaming_completion_enabled {
        let mut sorted_completion_indices = completion_indices.clone();
        sorted_completion_indices.sort_unstable();
        sorted_completion_indices.dedup();
        anyhow::ensure!(
            sorted_completion_indices.len() == row_count && completion_indices.len() == row_count,
            "streaming CUDA NVFP4 route completion covered {} unique rows and {} total rows, expected {row_count}",
            sorted_completion_indices.len(),
            completion_indices.len()
        );
    }
    let max_intermediate_rows = loaded_routes
        .iter()
        .map(|route| route.intermediate_rows)
        .max()
        .unwrap_or(0);
    let output_values = row_count
        .checked_mul(output_rows)
        .context("CUDA NVFP4 accumulated output value count overflow")?;
    let accumulator_bytes = output_values
        .checked_mul(std::mem::size_of::<f32>())
        .context("CUDA NVFP4 accumulated F32 route output byte count overflow")?;
    let scatter_index_bytes = scatter_indices
        .len()
        .checked_mul(std::mem::size_of::<u32>())
        .context("CUDA NVFP4 accumulated route index byte count overflow")?;
    let route_weight_bytes = route_weights
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .context("CUDA NVFP4 accumulated route weight byte count overflow")?;
    let route_metadata_bytes_count = route_count
        .checked_mul(std::mem::size_of::<Ds41rtNvfp4RouteBatchedMetadata>())
        .context("CUDA NVFP4 batched route metadata byte count overflow")?;
    let plan_ms = elapsed_ms(plan_started);

    let needs_host_output =
        (!retain_device_output && !streaming_completion_enabled) || cuda_route_validation_enabled();
    let retained_route_device_output = if retain_device_output {
        Some(device_bf16_output_uninitialized(
            row_count,
            output_rows,
            CUDA_REFERENCE_NVFP4_ROUTE_BF16_ACCUMULATED_DEVICE_OUTPUT_BACKEND,
            "NVFP4 BF16 route retained device output",
        )?)
    } else {
        None
    };
    let hidden_row_stride_elems = if nvfp4_hidden_payload {
        hidden_dim
    } else {
        hidden_row_stride_bytes / std::mem::size_of::<u16>()
    };
    let cuda_cache_prepare_ms: f64;
    let workspace_ms: f64;
    let payload_stage_ms: f64;
    let device_projection_ms: f64;
    let route_metadata_stage_ms: f64;
    let host_output_alloc_ms: f64;
    let mut graph_or_launch_ms = 0.0_f64;
    let mut enqueue_ms = 0.0_f64;
    let mut hidden_copy_enqueue_ms = 0.0_f64;
    let mut scatter_copy_enqueue_ms = 0.0_f64;
    let mut weight_copy_enqueue_ms = 0.0_f64;
    let mut accumulator_zero_enqueue_ms = 0.0_f64;
    let mut metadata_copy_enqueue_ms = 0.0_f64;
    let mut route_kernel_enqueue_ms = 0.0_f64;
    let mut bf16_pack_enqueue_ms = 0.0_f64;
    let mut retained_copy_enqueue_ms = 0.0_f64;
    let mut host_copy_enqueue_ms = 0.0_f64;
    let sync_ms: f64;
    let mut completion_emit_ms = 0.0_f64;
    let mut host_output_copy_ms = 0.0_f64;
    let mut cuda_event_hidden_copy_ms = 0.0_f64;
    let mut cuda_event_metadata_copy_ms = 0.0_f64;
    let mut cuda_event_route_kernel_ms = 0.0_f64;
    let mut cuda_event_bf16_pack_ms = 0.0_f64;
    let mut cuda_event_retained_copy_ms = 0.0_f64;
    let mut cuda_event_host_copy_ms = 0.0_f64;
    let mut cuda_event_total_ms = 0.0_f64;
    let cuda_projection_entries: usize;
    let cuda_projection_uploads: usize;
    let cuda_cache_hits: usize;
    let mut used_graph_path = false;
    let mut used_b12x_route_lanes = 1_usize;
    let (cuda_bytes, retained_route_device_output) = {
        let retained_route_device_output = retained_route_device_output;
        let cuda_cache_prepare_started = Instant::now();
        let cuda_cache = cache.cuda_cache()?;
        cuda_cache.prepare_layer(layer_id);
        cuda_cache_prepare_ms = elapsed_ms(cuda_cache_prepare_started);
        let library = Arc::clone(&cuda_cache.library);
        let cuda_stream = cuda_cache.stream.as_ptr();
        let completion_stream = cuda_cache.completion_stream.as_ptr();
        let retained_bf16_layer = cuda_cache.bf16_expert_slabs.get(&layer_id).cloned();
        let w4a16_layer_buffers = if cuda_cache.b12x_aot_enabled {
            cuda_cache
                .expert_slabs
                .get(&layer_id)
                .map(|slab| slab.w4a16_moe_buffers())
                .transpose()?
        } else {
            None
        };
        let packed_w4a16_layer_buffers = w4a16_layer_buffers;
        let grouped_decode_w4a16_buffers = if grouped_decode_shape {
            w4a16_layer_buffers
        } else {
            None
        };
        let grouped_decode_available = grouped_decode_w4a16_buffers.is_some();
        anyhow::ensure!(
            retained_bf16_layer.is_some() || packed_w4a16_layer_buffers.is_some(),
            "layer {layer_id} has neither retained BF16 nor packed W4A16 expert weights"
        );
        if timing_enabled {
            eprintln!(
                "real_nvfp4_route_stage stage=cuda_cache_ready layer_id={} rows={} routes={} hidden_dim={} output_rows={} elapsed_ms={:.3}",
                layer_id,
                row_count,
                route_count,
                hidden_dim,
                output_rows,
                cuda_cache_prepare_ms
            );
        }
        let hidden_workspace_bytes = hidden_bytes;
        let final_output_workspace_bytes = output_bytes;
        let scatter_index_workspace_bytes = scatter_index_bytes;
        let route_weight_workspace_bytes = route_weight_bytes;
        let workspace_started = Instant::now();
        let workspace = cuda_cache.workspace.ensure_accumulation_buffers(
            Arc::clone(&library),
            hidden_workspace_bytes,
            accumulator_bytes,
            final_output_workspace_bytes,
            scatter_index_workspace_bytes,
            route_weight_workspace_bytes,
            route_metadata_bytes_count,
        )?;
        workspace_ms = elapsed_ms(workspace_started);
        if timing_enabled {
            eprintln!(
                "real_nvfp4_route_stage stage=workspace_ready layer_id={} rows={} routes={} hidden_workspace_bytes={} final_output_bytes={} scatter_bytes={} route_weight_bytes={} metadata_bytes={} elapsed_ms={:.3}",
                layer_id,
                row_count,
                route_count,
                hidden_workspace_bytes,
                final_output_workspace_bytes,
                scatter_index_workspace_bytes,
                route_weight_workspace_bytes,
                route_metadata_bytes_count,
                workspace_ms
            );
        }
        let scatter_index_bytes_view = u32_bytes(&scatter_indices);
        let route_weight_bytes_view = f32_bytes(&route_weights);
        let hidden_device_buffer = hidden.device_buffer();
        let payload_stage_started = Instant::now();
        let host_pinned_payloads = if hidden_device_buffer.is_none() {
            let hidden_logical = hidden_logical
                .context("host hidden bytes missing for host-input route execution")?;
            Some(cuda_cache.workspace.stage_accumulation_payloads(
                Arc::clone(&library),
                hidden_logical,
                scatter_index_bytes_view,
                route_weight_bytes_view,
            )?)
        } else {
            None
        };
        let metadata_payloads = if let Some(pinned_payloads) = host_pinned_payloads {
            RouteCudaPinnedMetadataPayloadBuffers {
                scatter_index: pinned_payloads.scatter_index,
                route_weights: pinned_payloads.route_weights,
            }
        } else {
            cuda_cache.workspace.stage_accumulation_metadata_payloads(
                Arc::clone(&library),
                scatter_index_bytes_view,
                route_weight_bytes_view,
            )?
        };
        payload_stage_ms = elapsed_ms(payload_stage_started);
        if timing_enabled {
            eprintln!(
                "real_nvfp4_route_stage stage=payloads_ready layer_id={} rows={} routes={} host_input={} elapsed_ms={:.3}",
                layer_id,
                row_count,
                route_count,
                hidden_device_buffer.is_none(),
                payload_stage_ms
            );
        }

        let device_projection_started = Instant::now();
        let mut route_metadata = Vec::with_capacity(route_count);
        for group in &route_groups {
            route_metadata.extend(std::iter::repeat_n(
                Ds41rtNvfp4RouteBatchedMetadata::default(),
                group.count,
            ));
        }
        device_projection_ms = elapsed_ms(device_projection_started);
        cuda_projection_entries = if let Some(layer) = retained_bf16_layer.as_ref() {
            layer.expert_count * 3
        } else {
            cuda_cache
                .expert_slabs
                .values()
                .map(|slab| slab.expert_count * 3)
                .sum()
        };
        cuda_projection_uploads = cuda_cache.projection_uploads;
        cuda_cache_hits = 0;
        if timing_enabled {
            eprintln!(
                "real_nvfp4_route_stage stage=device_projections_ready layer_id={} rows={} routes={} metadata={} projection_entries={} projection_uploads={} cache_hits={} elapsed_ms={:.3}",
                layer_id,
                row_count,
                route_count,
                route_metadata.len(),
                cuda_projection_entries,
                cuda_projection_uploads,
                cuda_cache_hits,
                device_projection_ms
            );
        }
        if route_metadata.len() != route_count {
            anyhow::bail!(
                "CUDA NVFP4 batched route metadata length mismatch: expected {route_count}, got {}",
                route_metadata.len()
            );
        }
        let route_metadata_stage_started = Instant::now();
        let (route_metadata_payload, route_metadata_copy_bytes) = if grouped_decode_available {
            let expert_ids = route_groups
                .iter()
                .map(|group| {
                    u32::try_from(group.projections.gate.key.expert_id)
                        .context("grouped decode expert ID exceeds u32")
                })
                .collect::<Result<Vec<_>>>()?;
            (
                cuda_cache
                    .workspace
                    .stage_stream_input_indices(Arc::clone(&library), &expert_ids)?,
                expert_ids.len() * std::mem::size_of::<u32>(),
            )
        } else {
            (
                cuda_cache
                    .workspace
                    .stage_route_metadata_payload(Arc::clone(&library), &route_metadata)?,
                route_metadata_bytes_count,
            )
        };
        route_metadata_stage_ms = elapsed_ms(route_metadata_stage_started);
        if timing_enabled {
            eprintln!(
                "real_nvfp4_route_stage stage=metadata_payload_ready layer_id={} rows={} routes={} bytes={} elapsed_ms={:.3}",
                layer_id,
                row_count,
                route_count,
                route_metadata_copy_bytes,
                route_metadata_stage_ms
            );
        }
        let streaming_completion_plan = if streaming_completion_enabled {
            let mut slice_row_offsets = Vec::with_capacity(response_completion_slices.len());
            let mut row_offset = 0_usize;
            for rows in &response_completion_slices {
                slice_row_offsets.push(row_offset);
                row_offset = row_offset
                    .checked_add(rows.len())
                    .context("route completion row offset overflow")?;
            }
            anyhow::ensure!(
                row_offset == row_count,
                "route completion rows {row_offset} did not match row count {row_count}"
            );
            let buffers = cuda_cache.workspace.ensure_completion_buffers(
                Arc::clone(&library),
                &completion_indices,
                accumulator_bytes,
                streaming_output_bytes,
            )?;
            let events = cuda_cache
                .workspace
                .ensure_completion_events(Arc::clone(&library), response_completion_slices.len())?;
            Some(RouteCudaStreamingCompletionPlan {
                buffers,
                events,
                slice_row_offsets,
            })
        } else {
            None
        };
        let host_output_alloc_started = Instant::now();
        let mut out_bytes = needs_host_output.then(|| vec![0_u8; output_bytes]);
        host_output_alloc_ms = elapsed_ms(host_output_alloc_started);
        let route_cuda_event_timing = RouteCudaEventTimeline::enabled();
        let b12x_workspace_rows = max_route_group_count.min(B12X_SPARK_AOT_MAX_ROWS);
        let b12x_direct_route_candidate = use_grouped_route_launches
            && (retained_bf16_layer.is_some() || cuda_cache.b12x_aot_enabled)
            && (packed_w4a16_layer_buffers.is_none() || max_intermediate_rows == 512)
            && b12x_spark_direct_route_shape_supported(
                b12x_workspace_rows,
                hidden_dim,
                hidden_row_stride_elems,
                max_intermediate_rows,
                output_rows,
            );
        anyhow::ensure!(
            !nvfp4_hidden_payload || b12x_direct_route_candidate,
            "prequantized NVFP4 hidden exchange requires the direct B12X route backend"
        );
        anyhow::ensure!(
            b12x_direct_route_candidate,
            "CUDA expert routing requires the packed W4A16 or retained-BF16 B12X backend"
        );
        {
            let direct_output_payload = if needs_host_output {
                Some(
                    cuda_cache
                        .workspace
                        .ensure_output_payload(Arc::clone(&library), output_bytes)?,
                )
            } else {
                None
            };
            let cuda_event_timeline = if route_cuda_event_timing && !streaming_completion_enabled {
                Some(RouteCudaEventTimeline::new(Arc::clone(&library))?)
            } else {
                None
            };
            let b12x_route_lane_count = if grouped_decode_available {
                1
            } else if b12x_direct_route_candidate
                && nvfp4_hidden_payload
                && streaming_completion_enabled
            {
                cuda_cache.b12x_lane_count().min(route_groups.len().max(1))
            } else {
                1
            };
            let b12x_direct_workspaces = if b12x_direct_route_candidate {
                cuda_cache.ensure_b12x_route_workspaces(
                    b12x_route_lane_count,
                    b12x_workspace_rows,
                    hidden_dim,
                    max_intermediate_rows,
                    output_rows,
                )?
            } else {
                Vec::new()
            };
            let b12x_multistream = b12x_direct_workspaces.len() > 1;
            used_b12x_route_lanes = b12x_direct_workspaces.len().max(1);
            let mut used_b12x_direct_route = false;
            let packed_w4a16_decode_graph = route_cuda_graphs_enabled()
                && grouped_decode_w4a16_buffers.is_some()
                && b12x_direct_workspaces.len() == 1
                && host_pinned_payloads.is_some()
                && hidden_device_buffer.is_none()
                && !streaming_completion_enabled
                && direct_output_payload.is_none()
                && retained_route_device_output.is_some()
                && cuda_event_timeline.is_none();
            if packed_w4a16_decode_graph {
                let enqueue_started = Instant::now();
                let layer_buffers = grouped_decode_w4a16_buffers
                    .expect("packed W4A16 decode graph requires layer buffers");
                let b12x_workspace = b12x_direct_workspaces[0];
                let pinned_payloads = host_pinned_payloads
                    .expect("packed W4A16 decode graph requires pinned payloads");
                if !cuda_cache.grouped_decode_observed {
                    eprintln!(
                        "real_nvfp4_route_packed_w4a16_decode_graph_selected layer_id={layer_id} rows={row_count} routes={route_count}"
                    );
                    cuda_cache.grouped_decode_observed = true;
                }
                let graph_started = Instant::now();
                unsafe {
                    cuda_cache.launch_or_capture_packed_w4a16_decode_graph(
                        layer_id,
                        workspace,
                        pinned_payloads,
                        route_metadata_payload,
                        layer_buffers,
                        b12x_workspace,
                        hidden_row_stride_bytes,
                        hidden_bytes,
                        route_weight_bytes,
                        route_metadata_copy_bytes,
                        output_rows,
                        cuda_stream,
                    )?;
                }
                graph_or_launch_ms = elapsed_ms(graph_started);
                let retained_copy_started = Instant::now();
                unsafe {
                    library
                        .copy_d2d_async(
                            retained_route_device_output
                                .as_ref()
                                .expect("packed W4A16 decode graph retains device output")
                                .buffer(),
                            b12x_workspace.group_output,
                            output_bytes,
                            cuda_stream,
                        )
                        .context("retaining packed W4A16 decode output")?;
                }
                retained_copy_enqueue_ms = elapsed_ms(retained_copy_started);
                enqueue_ms = elapsed_ms(enqueue_started);
                let sync_started = Instant::now();
                unsafe {
                    library
                        .cuda_stream_synchronize(cuda_stream)
                        .context("synchronizing packed W4A16 decode graph")?;
                }
                sync_ms = elapsed_ms(sync_started);
                used_graph_path = true;
                kernel_backend = B12X_SPARK_DIRECT_NVFP4_ROUTE_BF16_ACCUMULATED_BACKEND;
            } else {
                unsafe {
                    let enqueue_started = Instant::now();
                    if let Some(timeline) = cuda_event_timeline.as_ref() {
                        timeline.record(timeline.start.as_ref(), cuda_stream, "start")?;
                    }
                    if let Some(plan) = streaming_completion_plan.as_ref() {
                        library
                            .copy_host_buffer_h2d_async(
                                plan.buffers.indices,
                                plan.buffers.pinned_indices,
                                completion_indices.len() * std::mem::size_of::<u32>(),
                                completion_stream,
                            )
                            .context("enqueueing route completion indices H2D copy")?;
                    }
                    if let Some(hidden_device_buffer) = hidden_device_buffer {
                        let op_started = Instant::now();
                        library
                        .copy_d2d_async(
                            workspace.hidden,
                            hidden_device_buffer,
                            hidden_bytes,
                            cuda_stream,
                        )
                        .context(
                            "enqueueing NVFP4 BF16 accumulated route hidden device-input D2D copy",
                        )?;
                        hidden_copy_enqueue_ms = elapsed_ms(op_started);
                    } else if let Some(pinned_payloads) = host_pinned_payloads {
                        let op_started = Instant::now();
                        library
                        .copy_host_buffer_h2d_async(
                            workspace.hidden,
                            pinned_payloads.hidden,
                            hidden_bytes,
                            cuda_stream,
                        )
                        .context(
                            "enqueueing pinned NVFP4 BF16 accumulated route hidden batch H2D copy",
                        )?;
                        hidden_copy_enqueue_ms = elapsed_ms(op_started);
                    }
                    if let Some(timeline) = cuda_event_timeline.as_ref() {
                        timeline.record(
                            timeline.hidden_ready.as_ref(),
                            cuda_stream,
                            "hidden_ready",
                        )?;
                    }
                    let op_started = Instant::now();
                    library
                        .copy_host_buffer_h2d_async(
                            workspace.scatter_index,
                            metadata_payloads.scatter_index,
                            scatter_index_bytes,
                            cuda_stream,
                        )
                        .context(
                            "enqueueing pinned NVFP4 BF16 accumulated route row-index H2D copy",
                        )?;
                    scatter_copy_enqueue_ms = elapsed_ms(op_started);
                    let op_started = Instant::now();
                    library
                        .copy_host_buffer_h2d_async(
                            workspace.route_weights,
                            metadata_payloads.route_weights,
                            route_weight_bytes,
                            cuda_stream,
                        )
                        .context(
                            "enqueueing pinned NVFP4 BF16 accumulated route weight H2D copy",
                        )?;
                    weight_copy_enqueue_ms = elapsed_ms(op_started);
                    let op_started = Instant::now();
                    library
                        .cuda_zero_f32_async(workspace.accumulator, output_values, cuda_stream)
                        .context("enqueueing NVFP4 BF16 route F32 accumulator zero")?;
                    accumulator_zero_enqueue_ms = elapsed_ms(op_started);
                    let op_started = Instant::now();
                    library
                        .copy_host_buffer_h2d_async(
                            workspace.route_metadata,
                            route_metadata_payload,
                            route_metadata_copy_bytes,
                            cuda_stream,
                        )
                        .context("enqueueing pinned NVFP4 BF16 batched route metadata H2D copy")?;
                    metadata_copy_enqueue_ms = elapsed_ms(op_started);
                    if let Some(timeline) = cuda_event_timeline.as_ref() {
                        timeline.record(
                            timeline.metadata_ready.as_ref(),
                            cuda_stream,
                            "metadata_ready",
                        )?;
                    }
                    if b12x_multistream {
                        let setup_ready = cuda_cache.b12x_lane_event(0);
                        library
                            .cuda_event_record(setup_ready, cuda_stream)
                            .context("recording B12X route setup completion")?;
                        for lane in 1..b12x_route_lane_count {
                            library
                                .cuda_stream_wait_event(
                                    cuda_cache.b12x_lane_stream(lane),
                                    setup_ready,
                                )
                                .context("waiting for B12X route setup on auxiliary lane")?;
                        }
                    }
                    let op_started = Instant::now();
                    anyhow::ensure!(
                        use_grouped_route_launches,
                        "packed W4A16 requires grouped route launches"
                    );
                    if use_grouped_route_launches {
                        let mut completion_slice_index = 0_usize;
                        let mut b12x_lane_used = vec![false; b12x_route_lane_count];
                        for (group_index, group) in route_groups.iter().enumerate() {
                            let b12x_lane = if b12x_multistream {
                                group_index % b12x_route_lane_count
                            } else {
                                0
                            };
                            let group_stream = if b12x_multistream {
                                cuda_cache.b12x_lane_stream(b12x_lane)
                            } else {
                                cuda_stream
                            };
                            let scatter_offset = group
                                .start
                                .checked_mul(std::mem::size_of::<u32>())
                                .context("grouped route scatter offset overflow")?;
                            let scatter_bytes = group
                                .count
                                .checked_mul(std::mem::size_of::<u32>())
                                .context("grouped route scatter bytes overflow")?;
                            let route_weight_offset = group
                                .start
                                .checked_mul(std::mem::size_of::<f32>())
                                .context("grouped route route-weight offset overflow")?;
                            let route_weight_bytes = group
                                .count
                                .checked_mul(std::mem::size_of::<f32>())
                                .context("grouped route route-weight bytes overflow")?;
                            let scatter_view = device_buffer_byte_view(
                                workspace.scatter_index,
                                scatter_offset,
                                scatter_bytes,
                                "grouped route scatter",
                            )?;
                            let route_weight_view = device_buffer_byte_view(
                                workspace.route_weights,
                                route_weight_offset,
                                route_weight_bytes,
                                "grouped route route weights",
                            )?;
                            let b12x_workspace = b12x_direct_workspaces.get(b12x_lane).copied();
                            let b12x_supported = b12x_workspace.is_some()
                                && b12x_spark_direct_route_shape_supported(
                                    group.count,
                                    hidden_dim,
                                    hidden_row_stride_elems,
                                    group.intermediate_rows,
                                    output_rows,
                                )
                                && (retained_bf16_layer.is_some()
                                    || (packed_w4a16_layer_buffers.is_some()
                                        && group.intermediate_rows == 512));
                            let mut launched_b12x = false;
                            if let (Some(layer_buffers), Some(b12x_workspace)) =
                                (retained_bf16_layer.as_ref(), b12x_workspace)
                            {
                                anyhow::ensure!(
                                    !nvfp4_hidden_payload
                                        && group.intermediate_rows
                                            == layer_buffers.intermediate_rows
                                        && hidden_dim == layer_buffers.hidden_dim
                                        && output_rows == layer_buffers.output_rows,
                                    "retained BF16 MTP route geometry or input dtype mismatch"
                                );
                                let compact_hidden_bytes = group
                                    .count
                                    .checked_mul(hidden_dim)
                                    .and_then(|values| {
                                        values.checked_mul(std::mem::size_of::<u16>())
                                    })
                                    .context("retained BF16 compact input bytes overflow")?;
                                let fc1_bytes = group
                                    .count
                                    .checked_mul(group.intermediate_rows * 2)
                                    .and_then(|values| {
                                        values.checked_mul(std::mem::size_of::<u16>())
                                    })
                                    .context("retained BF16 FC1 bytes overflow")?;
                                let activated_bytes = group
                                    .count
                                    .checked_mul(group.intermediate_rows)
                                    .and_then(|values| {
                                        values.checked_mul(std::mem::size_of::<u16>())
                                    })
                                    .context("retained BF16 activation bytes overflow")?;
                                let group_output_bytes = group
                                    .count
                                    .checked_mul(output_rows)
                                    .and_then(|values| {
                                        values.checked_mul(std::mem::size_of::<u16>())
                                    })
                                    .context("retained BF16 output bytes overflow")?;
                                let compact_hidden = device_buffer_byte_view(
                                    b12x_workspace.compact_hidden,
                                    0,
                                    compact_hidden_bytes,
                                    "retained BF16 compact hidden",
                                )?;
                                let fc1_output = device_buffer_byte_view(
                                    b12x_workspace.w4a16_fc1_output,
                                    0,
                                    fc1_bytes,
                                    "retained BF16 FC1 output",
                                )?;
                                let activated = device_buffer_byte_view(
                                    b12x_workspace.w4a16_activated,
                                    0,
                                    activated_bytes,
                                    "retained BF16 activated output",
                                )?;
                                let group_output = device_buffer_byte_view(
                                    b12x_workspace.group_output,
                                    0,
                                    group_output_bytes,
                                    "retained BF16 group output",
                                )?;
                                library
                                    .cuda_gather_rows_bf16_async(
                                        workspace.hidden,
                                        row_count,
                                        scatter_view,
                                        compact_hidden,
                                        group.count,
                                        hidden_dim,
                                        group_stream,
                                    )
                                    .context("gathering retained BF16 MTP route inputs")?;
                                let expert = layer_buffers
                                    .expert_buffers(group.projections.gate.key.expert_id)?;
                                library
                                    .cuda_linear_bf16_cublas_async(
                                        compact_hidden,
                                        expert.w13_weight,
                                        None,
                                        fc1_output,
                                        group.count,
                                        hidden_dim,
                                        group.intermediate_rows * 2,
                                        group_stream,
                                    )
                                    .context("launching retained BF16 MTP W13 GEMM")?;
                                library
                                    .cuda_silu_mul_bf16_async(
                                        fc1_output,
                                        activated,
                                        group.count,
                                        group.intermediate_rows,
                                        group_stream,
                                    )
                                    .context("launching retained BF16 MTP SiLU-mul")?;
                                library
                                    .cuda_linear_bf16_cublas_async(
                                        activated,
                                        expert.w2_weight,
                                        None,
                                        group_output,
                                        group.count,
                                        group.intermediate_rows,
                                        output_rows,
                                        group_stream,
                                    )
                                    .context("launching retained BF16 MTP W2 GEMM")?;
                                library
                                    .cuda_scatter_add_rows_bf16_weighted_to_f32_async(
                                        group_output,
                                        scatter_view,
                                        route_weight_view,
                                        workspace.accumulator,
                                        row_count,
                                        group.count,
                                        output_rows,
                                        group_stream,
                                    )
                                    .context(
                                        "accumulating retained BF16 MTP routed expert output",
                                    )?;
                                launched_b12x = true;
                                used_b12x_direct_route = true;
                                kernel_backend = RETAINED_BF16_MTP_ROUTE_BACKEND;
                            } else if let (Some(layer_buffers), Some(b12x_workspace)) =
                                (grouped_decode_w4a16_buffers, b12x_workspace)
                            {
                                launched_b12x = true;
                                used_b12x_direct_route = true;
                                if group_index == 0 {
                                    if !cuda_cache.grouped_decode_observed {
                                        eprintln!(
                                            "real_nvfp4_route_w4a16_decode_selected layer_id={layer_id} rows={row_count} routes={route_count}"
                                        );
                                        cuda_cache.grouped_decode_observed = true;
                                    }
                                    let input_payload = device_buffer_byte_view(
                                        workspace.hidden,
                                        0,
                                        hidden_row_stride_bytes,
                                        "packed W4A16 decode NVFP4 input payload",
                                    )?;
                                    let group_output = device_buffer_byte_view(
                                        b12x_workspace.group_output,
                                        0,
                                        output_rows * std::mem::size_of::<u16>(),
                                        "packed W4A16 decode BF16 output",
                                    )?;
                                    let topk_ids = device_buffer_byte_view(
                                        workspace.route_metadata,
                                        0,
                                        8 * std::mem::size_of::<i32>(),
                                        "packed W4A16 decode expert IDs",
                                    )?;
                                    let topk_weights = device_buffer_byte_view(
                                        workspace.route_weights,
                                        0,
                                        8 * std::mem::size_of::<f32>(),
                                        "packed W4A16 decode route weights",
                                    )?;
                                    let buffers = b12x_w4a16_moe_buffers(
                                        layer_buffers,
                                        b12x_workspace,
                                        b12x_workspace.compact_hidden,
                                        group_output,
                                        topk_weights,
                                    );
                                    launch_b12x_w4a16_decode(
                                        &library,
                                        layer_buffers,
                                        &buffers,
                                        input_payload,
                                        hidden_row_stride_bytes,
                                        topk_ids,
                                        group_stream,
                                    )
                                    .context("launching B12X W4A16 decode MoE")?;
                                    library
                                        .cuda_scatter_add_rows_bf16_to_f32_async(
                                            group_output,
                                            scatter_view,
                                            workspace.accumulator,
                                            row_count,
                                            1,
                                            output_rows,
                                            group_stream,
                                        )
                                        .context("accumulating packed B12X W4A16 decode output")?;
                                }
                            } else if let (true, Some(b12x_workspace)) =
                                (b12x_supported, b12x_workspace)
                            {
                                let compact_hidden_bytes = group
                                .count
                                .checked_mul(hidden_dim)
                                .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
                                .context(
                                    "b12x Spark direct route compact hidden group byte count overflow",
                                )?;
                                let group_output_bytes = group
                                .count
                                .checked_mul(output_rows)
                                .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
                                .context(
                                    "b12x Spark direct route compact output group byte count overflow",
                                )?;
                                let compact_hidden = device_buffer_byte_view(
                                    b12x_workspace.compact_hidden,
                                    0,
                                    compact_hidden_bytes,
                                    "b12x Spark direct route compact hidden",
                                )?;
                                let group_output = device_buffer_byte_view(
                                    b12x_workspace.group_output,
                                    0,
                                    group_output_bytes,
                                    "b12x Spark direct route compact output",
                                )?;
                                let layer_buffers = packed_w4a16_layer_buffers
                                    .context("packed W4A16 route lost its expert slab")?;
                                {
                                    if nvfp4_hidden_payload {
                                        library
                                            .cuda_b12x_gather_nvfp4_rows_bf16_async(
                                                workspace.hidden,
                                                row_count,
                                                hidden_row_stride_bytes,
                                                scatter_view,
                                                compact_hidden,
                                                group.count,
                                                hidden_dim,
                                                group_stream,
                                            )
                                            .context("gathering packed W4A16 NVFP4 route inputs")?;
                                    } else {
                                        library
                                            .cuda_gather_rows_bf16_async(
                                                workspace.hidden,
                                                row_count,
                                                scatter_view,
                                                compact_hidden,
                                                group.count,
                                                hidden_dim,
                                                group_stream,
                                            )
                                            .context("gathering packed W4A16 BF16 route inputs")?;
                                    }
                                    let buffers = b12x_w4a16_moe_buffers(
                                        layer_buffers,
                                        b12x_workspace,
                                        compact_hidden,
                                        group_output,
                                        b12x_workspace.w4a16_topk_weights,
                                    );
                                    library
                                        .cuda_b12x_spark_w4a16_top1_async(
                                            &buffers,
                                            group.count,
                                            b12x_w4a16_capacity_rows(group.count)?,
                                            u32::try_from(group.projections.gate.key.expert_id)
                                                .context("packed W4A16 expert ID exceeds u32")?,
                                            group_stream,
                                        )
                                        .context("launching packed B12X W4A16 route expert")?;
                                    library
                                        .cuda_scatter_add_rows_bf16_weighted_to_f32_async(
                                            group_output,
                                            scatter_view,
                                            route_weight_view,
                                            workspace.accumulator,
                                            row_count,
                                            group.count,
                                            output_rows,
                                            group_stream,
                                        )
                                        .context("accumulating packed B12X W4A16 route output")?;
                                    used_b12x_direct_route = true;
                                    launched_b12x = true;
                                }
                            }
                            anyhow::ensure!(
                                launched_b12x,
                                "packed W4A16 route was not launched for layer {layer_id}"
                            );
                            if b12x_multistream {
                                let lane_event = cuda_cache.b12x_lane_event(b12x_lane);
                                library
                                    .cuda_event_record(lane_event, group_stream)
                                    .context("recording B12X route lane completion")?;
                                b12x_lane_used[b12x_lane] = true;
                            }
                            if streaming_completion_group_indices
                                .get(completion_slice_index)
                                .copied()
                                == Some(group_index)
                            {
                                if let Some(plan) = streaming_completion_plan.as_ref() {
                                    let slice_rows =
                                        response_completion_slices[completion_slice_index].len();
                                    let row_offset = plan.slice_row_offsets[completion_slice_index];
                                    let index_offset = row_offset
                                        .checked_mul(std::mem::size_of::<u32>())
                                        .context("route completion index offset overflow")?;
                                    let index_bytes = slice_rows
                                        .checked_mul(std::mem::size_of::<u32>())
                                        .context("route completion index byte count overflow")?;
                                    let f32_offset = row_offset
                                        .checked_mul(output_rows)
                                        .and_then(|values| {
                                            values.checked_mul(std::mem::size_of::<f32>())
                                        })
                                        .context("route completion F32 offset overflow")?;
                                    let f32_bytes = slice_rows
                                        .checked_mul(output_rows)
                                        .and_then(|values| {
                                            values.checked_mul(std::mem::size_of::<f32>())
                                        })
                                        .context("route completion F32 byte count overflow")?;
                                    let packed_offset = row_offset
                                        .checked_mul(streaming_output_row_bytes)
                                        .context(
                                            "route completion packed output offset overflow",
                                        )?;
                                    let packed_bytes = slice_rows
                                        .checked_mul(streaming_output_row_bytes)
                                        .context(
                                            "route completion packed output byte count overflow",
                                        )?;
                                    let index_view = device_buffer_byte_view(
                                        plan.buffers.indices,
                                        index_offset,
                                        index_bytes,
                                        "route completion indices",
                                    )?;
                                    let f32_view = device_buffer_byte_view(
                                        plan.buffers.f32_output,
                                        f32_offset,
                                        f32_bytes,
                                        "route completion F32 output",
                                    )?;
                                    let packed_view = device_buffer_byte_view(
                                        plan.buffers.output,
                                        packed_offset,
                                        packed_bytes,
                                        "route completion packed output",
                                    )?;
                                    let pinned_view = host_buffer_byte_view(
                                        plan.buffers.pinned_output,
                                        packed_offset,
                                        packed_bytes,
                                        "route completion pinned output",
                                    )?;
                                    let (compute_event, ready_event) =
                                        plan.events[completion_slice_index];
                                    if b12x_multistream {
                                        for lane in 0..b12x_route_lane_count {
                                            if b12x_lane_used[lane] {
                                                library
                                                .cuda_stream_wait_event(
                                                    completion_stream,
                                                    cuda_cache.b12x_lane_event(lane),
                                                )
                                                .context(
                                                    "joining B12X route lanes for completion pack",
                                                )?;
                                            }
                                        }
                                    } else {
                                        library
                                            .cuda_event_record(compute_event, cuda_stream)
                                            .context("recording route completion compute event")?;
                                        library
                                            .cuda_stream_wait_event(
                                                completion_stream,
                                                compute_event,
                                            )
                                            .context(
                                                "waiting for route completion on copy stream",
                                            )?;
                                    }
                                    match streaming_output_dtype {
                                        RouteStreamingOutputDtype::Bf16 => {
                                            library
                                                .cuda_gather_rows_f32_async(
                                                    workspace.accumulator,
                                                    row_count,
                                                    index_view,
                                                    f32_view,
                                                    slice_rows,
                                                    output_rows,
                                                    completion_stream,
                                                )
                                                .context("enqueueing completed route row gather")?;
                                            library
                                                .cuda_f32_to_bf16_async(
                                                    f32_view,
                                                    packed_view,
                                                    slice_rows * output_rows,
                                                    completion_stream,
                                                )
                                                .context("enqueueing completed route BF16 pack")?;
                                        }
                                        RouteStreamingOutputDtype::Fp8E4m3RowScaled => {
                                            library
                                            .cuda_gather_rows_f32_to_fp8_e4m3_row_scaled_async(
                                                workspace.accumulator,
                                                row_count,
                                                index_view,
                                                packed_view,
                                                slice_rows,
                                                output_rows,
                                                streaming_output_row_bytes,
                                                completion_stream,
                                            )
                                            .context(
                                                "enqueueing completed route row-scaled FP8 pack",
                                            )?;
                                        }
                                        RouteStreamingOutputDtype::Nvfp4E2m1Fp8E4m3 => {
                                            library
                                            .cuda_gather_rows_f32_to_nvfp4_e2m1_fp8_e4m3_async(
                                                workspace.accumulator,
                                                row_count,
                                                index_view,
                                                packed_view,
                                                slice_rows,
                                                output_rows,
                                                streaming_output_row_bytes,
                                                completion_stream,
                                            )
                                            .context(
                                                "enqueueing completed route NVFP4 response pack",
                                            )?;
                                        }
                                    }
                                    library
                                        .copy_d2h_host_buffer_async(
                                            pinned_view,
                                            packed_view,
                                            packed_bytes,
                                            completion_stream,
                                        )
                                        .context("enqueueing completed route D2H copy")?;
                                    library
                                        .cuda_event_record(ready_event, completion_stream)
                                        .context("recording route response-ready event")?;
                                }
                                completion_slice_index += 1;
                            }
                        }
                        if !streaming_completion_enabled && b12x_multistream {
                            for lane in 1..b12x_route_lane_count {
                                if b12x_lane_used[lane] {
                                    library
                                        .cuda_stream_wait_event(
                                            cuda_stream,
                                            cuda_cache.b12x_lane_event(lane),
                                        )
                                        .context("joining B12X route lanes before output pack")?;
                                }
                            }
                        }
                        if streaming_completion_enabled {
                            debug_assert_eq!(
                                completion_slice_index,
                                response_completion_slices.len()
                            );
                        }
                    }
                    route_kernel_enqueue_ms = elapsed_ms(op_started);
                    if used_b12x_direct_route {
                        kernel_backend = B12X_SPARK_DIRECT_NVFP4_ROUTE_BF16_ACCUMULATED_BACKEND;
                    }
                    if let Some(timeline) = cuda_event_timeline.as_ref() {
                        timeline.record(
                            timeline.routes_ready.as_ref(),
                            cuda_stream,
                            "routes_ready",
                        )?;
                    }
                    if !streaming_completion_enabled {
                        let op_started = Instant::now();
                        library
                            .cuda_f32_to_bf16_async(
                                workspace.accumulator,
                                workspace.final_output,
                                output_values,
                                cuda_stream,
                            )
                            .context("enqueueing accumulated CUDA NVFP4 route output BF16 pack")?;
                        bf16_pack_enqueue_ms = elapsed_ms(op_started);
                        if let Some(timeline) = cuda_event_timeline.as_ref() {
                            timeline.record(
                                timeline.pack_ready.as_ref(),
                                cuda_stream,
                                "pack_ready",
                            )?;
                        }
                        if let Some(retained) = retained_route_device_output.as_ref() {
                            let op_started = Instant::now();
                            library
                            .copy_d2d_async(
                                retained.buffer(),
                                workspace.final_output,
                                output_bytes,
                                cuda_stream,
                            )
                            .context(
                                "enqueueing accumulated CUDA NVFP4 BF16 route output retained D2D copy",
                            )?;
                            retained_copy_enqueue_ms = elapsed_ms(op_started);
                        }
                        if let Some(timeline) = cuda_event_timeline.as_ref() {
                            timeline.record(
                                timeline.retained_ready.as_ref(),
                                cuda_stream,
                                "retained_ready",
                            )?;
                        }
                        if let Some(pinned_output) = direct_output_payload {
                            let op_started = Instant::now();
                            library
                            .copy_d2h_host_buffer_async(
                                pinned_output,
                                workspace.final_output,
                                output_bytes,
                                cuda_stream,
                            )
                            .context(
                                "enqueueing accumulated CUDA NVFP4 BF16 route output pinned D2H copy",
                            )?;
                            host_copy_enqueue_ms = elapsed_ms(op_started);
                        }
                        if let Some(timeline) = cuda_event_timeline.as_ref() {
                            timeline.record(
                                timeline.host_copy_ready.as_ref(),
                                cuda_stream,
                                "host_copy_ready",
                            )?;
                        }
                    }
                    enqueue_ms = elapsed_ms(enqueue_started);
                    if let Some(plan) = streaming_completion_plan.as_ref() {
                        let mut event_wait_ms = 0.0_f64;
                        let emit = completion_emit
                            .as_deref_mut()
                            .expect("streaming completion callback is present");
                        for (slice_index, rows) in response_completion_slices.iter().enumerate() {
                            let wait_started = Instant::now();
                            library
                                .cuda_event_synchronize(plan.events[slice_index].1)
                                .context("waiting for route response-ready event")?;
                            event_wait_ms += elapsed_ms(wait_started);
                            let byte_offset = plan.slice_row_offsets[slice_index]
                                .checked_mul(streaming_output_row_bytes)
                                .context("route completion output offset overflow")?;
                            let byte_count = rows
                                .len()
                                .checked_mul(streaming_output_row_bytes)
                                .context("route completion output byte count overflow")?;
                            let output = cuda_cache
                                .workspace
                                .completion_output_slice(streaming_output_bytes)?;
                            let emit_started = Instant::now();
                            if let Err(error) =
                                emit(rows, &output[byte_offset..byte_offset + byte_count])
                            {
                                let _ = library.cuda_stream_synchronize(completion_stream);
                                let _ = library.cuda_stream_synchronize(cuda_stream);
                                for stream in cuda_cache
                                    .b12x_aux_streams
                                    .iter()
                                    .take(b12x_route_lane_count.saturating_sub(1))
                                {
                                    let _ = library.cuda_stream_synchronize(stream.as_ptr());
                                }
                                return Err(error)
                                    .context("emitting completed CUDA NVFP4 route rows");
                            }
                            completion_emit_ms += elapsed_ms(emit_started);
                        }
                        sync_ms = event_wait_ms;
                    } else {
                        let sync_started = Instant::now();
                        library
                            .cuda_stream_synchronize(cuda_stream)
                            .context("synchronizing NVFP4 route CUDA stream")?;
                        sync_ms = elapsed_ms(sync_started);
                        if let Some(timeline) = cuda_event_timeline.as_ref() {
                            let elapsed = timeline.elapsed()?;
                            cuda_event_hidden_copy_ms = elapsed.hidden_copy_ms;
                            cuda_event_metadata_copy_ms = elapsed.metadata_copy_ms;
                            cuda_event_route_kernel_ms = elapsed.route_kernel_ms;
                            cuda_event_bf16_pack_ms = elapsed.bf16_pack_ms;
                            cuda_event_retained_copy_ms = elapsed.retained_copy_ms;
                            cuda_event_host_copy_ms = elapsed.host_copy_ms;
                            cuda_event_total_ms = elapsed.total_ms;
                        }
                    }
                }
            }
            if let Some(out_bytes) = out_bytes.as_mut() {
                let host_copy_started = Instant::now();
                out_bytes.copy_from_slice(
                    cuda_cache
                        .workspace
                        .output_payload_slice(output_bytes)
                        .context("reading direct NVFP4 route pinned output")?,
                );
                host_output_copy_ms = elapsed_ms(host_copy_started);
            }
        }
        (out_bytes, retained_route_device_output)
    };

    if timing_enabled {
        eprintln!(
            "real_nvfp4_route_timing layer_id={} rows={} routes={} route_groups={} max_route_group_count={} completion_slices={} response_slices={} first_completion_group={} first_completion_rows={} grouped_route_launches={} b12x_route_lanes={} hidden_dim={} output_rows={} hidden_device_input={} retain_device_output={} host_output={} graph={} backend={} projection_source_ms={:.3} plan_ms={:.3} cuda_cache_prepare_ms={:.3} workspace_ms={:.3} payload_stage_ms={:.3} device_projection_ms={:.3} route_metadata_stage_ms={:.3} host_output_alloc_ms={:.3} graph_or_launch_ms={:.3} enqueue_ms={:.3} hidden_copy_enqueue_ms={:.3} scatter_copy_enqueue_ms={:.3} weight_copy_enqueue_ms={:.3} accumulator_zero_enqueue_ms={:.3} metadata_copy_enqueue_ms={:.3} route_kernel_enqueue_ms={:.3} bf16_pack_enqueue_ms={:.3} retained_copy_enqueue_ms={:.3} host_copy_enqueue_ms={:.3} sync_ms={:.3} completion_emit_ms={:.3} host_output_copy_ms={:.3} cuda_projection_entries={} cuda_projection_uploads={} cuda_cache_hits={} total_ms={:.3}",
            layer_id,
            row_count,
            route_count,
            route_group_count,
            max_route_group_count,
            completion_slice_count,
            response_completion_slice_count,
            first_completion_group,
            first_completion_rows,
            use_grouped_route_launches,
            used_b12x_route_lanes,
            hidden_dim,
            output_rows,
            hidden_device_input,
            retain_device_output,
            needs_host_output,
            used_graph_path,
            kernel_backend,
            projection_source_ms,
            plan_ms,
            cuda_cache_prepare_ms,
            workspace_ms,
            payload_stage_ms,
            device_projection_ms,
            route_metadata_stage_ms,
            host_output_alloc_ms,
            graph_or_launch_ms,
            enqueue_ms,
            hidden_copy_enqueue_ms,
            scatter_copy_enqueue_ms,
            weight_copy_enqueue_ms,
            accumulator_zero_enqueue_ms,
            metadata_copy_enqueue_ms,
            route_kernel_enqueue_ms,
            bf16_pack_enqueue_ms,
            retained_copy_enqueue_ms,
            host_copy_enqueue_ms,
            sync_ms,
            completion_emit_ms,
            host_output_copy_ms,
            cuda_projection_entries,
            cuda_projection_uploads,
            cuda_cache_hits,
            elapsed_ms(timing_total_started)
        );
        if cuda_event_total_ms > 0.0 {
            eprintln!(
                "real_nvfp4_route_cuda_event_timing layer_id={} rows={} routes={} route_groups={} grouped_route_launches={} hidden_copy_ms={:.3} metadata_copy_ms={:.3} route_kernel_ms={:.3} bf16_pack_ms={:.3} retained_copy_ms={:.3} host_copy_ms={:.3} total_ms={:.3}",
                layer_id,
                row_count,
                route_count,
                route_group_count,
                use_grouped_route_launches,
                cuda_event_hidden_copy_ms,
                cuda_event_metadata_copy_ms,
                cuda_event_route_kernel_ms,
                cuda_event_bf16_pack_ms,
                cuda_event_retained_copy_ms,
                cuda_event_host_copy_ms,
                cuda_event_total_ms
            );
        }
    }

    if cuda_route_validation_enabled() {
        let cuda_bytes = cuda_bytes
            .as_ref()
            .context("CUDA NVFP4 route validation requested host-visible CUDA bytes")?;
        let hidden_logical = hidden_logical
            .context("CUDA NVFP4 route validation requested host-visible hidden bytes")?;
        let mut expected = vec![0.0_f32; output_values];
        for loaded_route in &loaded_routes {
            let row_start = loaded_route
                .row_index
                .checked_mul(hidden_row_stride_bytes)
                .context("CUDA NVFP4 validation hidden row offset overflow")?;
            let row_end = row_start
                .checked_add(logical_hidden_bytes)
                .context("CUDA NVFP4 validation hidden row range overflow")?;
            let hidden = bf16_bytes_to_f32(&hidden_logical[row_start..row_end])?;
            let cpu = cpu_nvfp4_route_computation(
                &hidden,
                &loaded_route.route,
                loaded_route.projections.gate.host_projection()?,
                loaded_route.projections.up.host_projection()?,
                loaded_route.projections.down.host_projection()?,
                loaded_route.intermediate_rows,
                output_rows,
                loaded_route.projections.gate_scale_2,
                loaded_route.projections.up_scale_2,
                loaded_route.projections.down_scale_2,
            )?;
            let mut route_bytes = vec![0_u8; output_row_bytes];
            f32_values_to_bf16_bytes(&cpu.outputs, &mut route_bytes);
            let route_outputs = bf16_bytes_to_f32(&route_bytes)?;
            let output_start = loaded_route
                .row_index
                .checked_mul(output_rows)
                .context("CUDA NVFP4 validation output row offset overflow")?;
            let output_end = output_start
                .checked_add(output_rows)
                .context("CUDA NVFP4 validation output row range overflow")?;
            for (dst, delta) in expected[output_start..output_end]
                .iter_mut()
                .zip(route_outputs.iter())
            {
                *dst += *delta;
            }
        }
        let mut expected_bytes = vec![0_u8; output_bytes];
        f32_values_to_bf16_bytes(&expected, &mut expected_bytes);
        let cuda_outputs = bf16_bytes_to_f32(cuda_bytes)?;
        let expected_outputs = bf16_bytes_to_f32(&expected_bytes)?;
        validate_cuda_route_outputs(&cuda_outputs, &expected_outputs)?;
    }

    let output_device = retained_route_device_output;

    Ok(RouteBf16AccumulatedInnerExecution {
        output_bf16: cuda_bytes,
        output_device,
        completion_slices: if streaming_completion_enabled {
            response_completion_slices
        } else {
            completion_slices
        },
        kernel_backend,
    })
}

#[allow(clippy::too_many_arguments)]
fn load_bf16_route_projections_for_group_cached(
    catalog: &TensorCatalog,
    layer_id: usize,
    route: &ScoredRoute,
    intermediate_rows: usize,
    output_rows: usize,
    hidden_dim: usize,
    cache: &mut RouteTensorCache,
    loaded_projection_groups: &mut HashMap<Bf16RouteProjectionGroupKey, Bf16RouteProjections>,
) -> Result<Bf16RouteProjections> {
    let key = Bf16RouteProjectionGroupKey {
        expert_id: route.expert_id,
        intermediate_rows,
    };
    if let Some(projections) = loaded_projection_groups.get(&key) {
        return Ok(projections.clone());
    }

    let require_host_tensors = cuda_route_validation_enabled();
    let cache_key = Bf16RouteProjectionGroupCacheKey {
        layer_id,
        expert_id: route.expert_id,
        intermediate_rows,
        output_rows,
        hidden_dim,
        require_host_tensors,
    };
    if let Some(projections) = cache.bf16_projection_groups.get(&cache_key) {
        let projections = projections.clone();
        loaded_projection_groups.insert(key, projections.clone());
        return Ok(projections);
    }

    let projections = load_validated_bf16_route_projections(
        catalog,
        layer_id,
        route.expert_id,
        intermediate_rows,
        output_rows,
        hidden_dim,
        cache,
    )?;
    cache
        .bf16_projection_groups
        .insert(cache_key, projections.clone());
    loaded_projection_groups.insert(key, projections.clone());
    Ok(projections)
}

#[allow(clippy::too_many_arguments)]
fn load_validated_bf16_route_projections(
    catalog: &TensorCatalog,
    layer_id: usize,
    expert_id: usize,
    intermediate_rows: usize,
    output_rows: usize,
    hidden_dim: usize,
    cache: &mut RouteTensorCache,
) -> Result<Bf16RouteProjections> {
    let require_host_tensors = cuda_route_validation_enabled();
    let (gate, gate_scale_2) = load_bf16_route_projection_source(
        catalog,
        layer_id,
        expert_id,
        "gate_proj",
        intermediate_rows,
        hidden_dim,
        cache,
        require_host_tensors,
    )?;
    let (up, up_scale_2) = load_bf16_route_projection_source(
        catalog,
        layer_id,
        expert_id,
        "up_proj",
        intermediate_rows,
        hidden_dim,
        cache,
        require_host_tensors,
    )?;
    let (down, down_scale_2) = load_bf16_route_projection_source(
        catalog,
        layer_id,
        expert_id,
        "down_proj",
        output_rows,
        intermediate_rows,
        cache,
        require_host_tensors,
    )?;

    Ok(Bf16RouteProjections {
        gate,
        up,
        down,
        gate_scale_2,
        up_scale_2,
        down_scale_2,
    })
}

#[allow(clippy::too_many_arguments)]
fn load_bf16_route_projection_source(
    catalog: &TensorCatalog,
    layer_id: usize,
    expert_id: usize,
    projection: &'static str,
    row_count: usize,
    input_width: usize,
    cache: &mut RouteTensorCache,
    require_host_tensors: bool,
) -> Result<(Bf16RouteProjection, f32)> {
    let key = RoutedQuantProjectionKey {
        layer_id,
        expert_id,
        projection,
        row_count,
    };
    if require_host_tensors {
        let host = load_routed_quant_projection_cached(
            catalog, layer_id, expert_id, projection, row_count, cache,
        )?;
        validate_packed_nvfp4_projection_width(
            projection,
            &host.weight,
            &host.weight_scale,
            input_width,
        )?;
        let input_scale = first_f32_scalar(&host.input_scale.info.name, &host.input_scale.bytes)?;
        let scale_2 = first_f32_scalar(&host.weight_scale_2.info.name, &host.weight_scale_2.bytes)?;
        validate_finite_route_scalar(&host.input_scale.info.name, input_scale)?;
        validate_finite_route_scalar(&host.weight_scale_2.info.name, scale_2)?;
        return Ok((
            Bf16RouteProjection {
                key,
                host: Some(host),
            },
            scale_2,
        ));
    }

    validate_routed_quant_projection_catalog(
        catalog,
        layer_id,
        expert_id,
        projection,
        row_count,
        input_width,
    )?;
    let scalar_metadata =
        load_routed_quant_scalar_metadata_cached(catalog, layer_id, expert_id, projection, cache)?;
    validate_finite_route_scalar(
        &scalar_metadata.input_scale_name,
        scalar_metadata.input_scale,
    )?;
    validate_finite_route_scalar(
        &scalar_metadata.weight_scale_2_name,
        scalar_metadata.weight_scale_2,
    )?;
    Ok((
        Bf16RouteProjection { key, host: None },
        scalar_metadata.weight_scale_2,
    ))
}

#[allow(clippy::too_many_arguments)]
fn cpu_nvfp4_route_computation(
    hidden: &[f32],
    route: &ScoredRoute,
    gate: &RoutedQuantProjection,
    up: &RoutedQuantProjection,
    down: &RoutedQuantProjection,
    intermediate_rows: usize,
    output_rows: usize,
    gate_scale_2: f32,
    up_scale_2: f32,
    down_scale_2: f32,
) -> Result<RouteComputation> {
    let mut activations = Vec::with_capacity(intermediate_rows);
    for row_index in 0..intermediate_rows {
        let gate_value = dot_packed_nvfp4(
            hidden,
            tensor_row_bytes(&gate.weight, row_index)?,
            tensor_row_bytes(&gate.weight_scale, row_index)?,
            gate_scale_2,
        )?;
        let up_value = dot_packed_nvfp4(
            hidden,
            tensor_row_bytes(&up.weight, row_index)?,
            tensor_row_bytes(&up.weight_scale, row_index)?,
            up_scale_2,
        )?;
        let activation = silu(gate_value) * up_value;
        if !activation.is_finite() {
            anyhow::bail!(
                "real full NVFP4 expert probe produced non-finite activation at row {row_index}"
            );
        }
        activations.push(activation);
    }

    let mut outputs = Vec::with_capacity(output_rows);
    for row_index in 0..output_rows {
        let output = route.normalized_weight
            * dot_packed_nvfp4(
                &activations,
                tensor_row_bytes(&down.weight, row_index)?,
                tensor_row_bytes(&down.weight_scale, row_index)?,
                down_scale_2,
            )?;
        if !output.is_finite() {
            anyhow::bail!(
                "real full NVFP4 expert probe produced non-finite output at row {row_index}"
            );
        }
        outputs.push(output);
    }

    Ok(RouteComputation {
        outputs,
        kernel_backend: CPU_REFERENCE_NVFP4_ROUTE_BACKEND,
    })
}

fn validate_cuda_route_outputs(cuda_outputs: &[f32], cpu_outputs: &[f32]) -> Result<()> {
    if cuda_outputs.len() != cpu_outputs.len() {
        anyhow::bail!(
            "CUDA NVFP4 route output length mismatch: cuda={} cpu={}",
            cuda_outputs.len(),
            cpu_outputs.len()
        );
    }
    let mut first_mismatch = None;
    let mut max_abs_error = 0.0_f32;
    let mut max_abs_index = 0_usize;
    for (idx, (cuda, cpu)) in cuda_outputs.iter().zip(cpu_outputs.iter()).enumerate() {
        let abs_error = (*cuda - *cpu).abs();
        if abs_error > max_abs_error {
            max_abs_error = abs_error;
            max_abs_index = idx;
        }
        let tolerance = CUDA_NVFP4_ROUTE_TOLERANCE * cpu.abs().max(1.0);
        if (!cuda.is_finite() || abs_error > tolerance) && first_mismatch.is_none() {
            first_mismatch = Some((idx, *cuda, *cpu, tolerance, abs_error));
        }
    }
    if let Some((idx, cuda, cpu, tolerance, abs_error)) = first_mismatch {
        anyhow::bail!(
            "CUDA NVFP4 route output mismatch at {idx}: cuda={cuda} cpu={cpu} abs_error={abs_error} tolerance={tolerance} max_abs_error={max_abs_error} max_abs_index={max_abs_index}"
        );
    }
    Ok(())
}

pub(in crate::commands::real_full) fn cuda_reference_kernels_enabled() -> bool {
    #[cfg(test)]
    if let Some(enabled) = CUDA_REFERENCE_KERNELS_TEST_OVERRIDE.with(|value| value.get()) {
        return enabled;
    }

    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env::var(REAL_FULL_CUDA_REFERENCE_KERNELS_ENV)
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "cuda" | "reference"
                )
            })
            .unwrap_or(false)
    })
}

#[cfg(test)]
fn set_cuda_reference_kernels_test_override(enabled: Option<bool>) -> Option<bool> {
    CUDA_REFERENCE_KERNELS_TEST_OVERRIDE.with(|value| {
        let previous = value.get();
        value.set(enabled);
        previous
    })
}

#[cfg(test)]
pub(in crate::commands::real_full) struct CudaReferenceKernelsTestOverride {
    previous: Option<bool>,
}

#[cfg(test)]
impl Drop for CudaReferenceKernelsTestOverride {
    fn drop(&mut self) {
        set_cuda_reference_kernels_test_override(self.previous);
    }
}

#[cfg(test)]
pub(in crate::commands::real_full) fn cuda_reference_kernels_test_override(
    enabled: bool,
) -> CudaReferenceKernelsTestOverride {
    CudaReferenceKernelsTestOverride {
        previous: set_cuda_reference_kernels_test_override(Some(enabled)),
    }
}

pub(in crate::commands::real_full) fn cuda_route_validation_enabled() -> bool {
    #[cfg(test)]
    if let Some(enabled) = CUDA_ROUTE_VALIDATION_TEST_OVERRIDE.with(|value| value.get()) {
        return enabled;
    }

    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env::var(REAL_FULL_CUDA_ROUTE_VALIDATE_ENV)
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "validate"
                )
            })
            .unwrap_or(false)
    })
}

fn b12x_spark_w4a16_device_weights_enabled() -> bool {
    env::var(REAL_FULL_B12X_SPARK_W4A16_DEVICE_WEIGHTS_ENV)
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "device" | "cuda"
            )
        })
        .unwrap_or(false)
}

fn route_cuda_graphs_enabled() -> bool {
    #[cfg(test)]
    if let Some(enabled) = ROUTE_CUDA_GRAPHS_TEST_OVERRIDE.with(|value| value.get()) {
        return enabled;
    }

    env::var(REAL_FULL_NVFP4_ROUTE_CUDA_GRAPHS_ENV)
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

#[cfg(test)]
fn set_route_cuda_graphs_test_override(enabled: Option<bool>) -> Option<bool> {
    ROUTE_CUDA_GRAPHS_TEST_OVERRIDE.with(|value| {
        let previous = value.get();
        value.set(enabled);
        previous
    })
}

#[cfg(test)]
struct RouteCudaGraphsTestOverride {
    previous: Option<bool>,
}

#[cfg(test)]
impl Drop for RouteCudaGraphsTestOverride {
    fn drop(&mut self) {
        set_route_cuda_graphs_test_override(self.previous);
    }
}

#[cfg(test)]
fn route_cuda_graphs_test_override(enabled: bool) -> RouteCudaGraphsTestOverride {
    RouteCudaGraphsTestOverride {
        previous: set_route_cuda_graphs_test_override(Some(enabled)),
    }
}

fn route_grouped_multirow_enabled() -> bool {
    env::var(REAL_FULL_NVFP4_ROUTE_GROUPED_MULTIROW_ENV)
        .map(|value| {
            !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off"
            )
        })
        .unwrap_or(true)
}

fn b12x_spark_direct_route_requested() -> bool {
    env::var(REAL_FULL_B12X_SPARK_DIRECT_ROUTE_ENV)
        .map(|value| {
            !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off" | "disabled"
            )
        })
        .unwrap_or(true)
}

fn b12x_spark_grouped_decode_enabled() -> bool {
    env::var(REAL_FULL_B12X_SPARK_GROUPED_DECODE_ENV)
        .map(|value| {
            !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off" | "disabled"
            )
        })
        .unwrap_or(true)
}

fn b12x_spark_w4a16_small_m_mode() -> W4a16SmallMMode {
    static MODE: OnceLock<W4a16SmallMMode> = OnceLock::new();
    *MODE.get_or_init(|| {
        match env::var(REAL_FULL_B12X_SPARK_W4A16_SMALL_M_MODE_ENV)
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "ordered" => W4a16SmallMMode::Ordered,
            "split-m1" => W4a16SmallMMode::SplitM1,
            "wide" | "wide-ordered" => W4a16SmallMMode::WideOrdered,
            _ => W4a16SmallMMode::WideOrdered,
        }
    })
}

fn fused_fp8_reduction_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env::var(EXPERT_FUSED_FP8_REDUCTION_ENV)
            .map(|value| {
                !matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "0" | "false" | "no" | "off" | "disabled"
                )
            })
            .unwrap_or(true)
    })
}

fn nccl_bf16_reduce_enabled() -> bool {
    env::var(EXPERT_NCCL_BF16_REDUCE_ENV)
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on" | "enabled"
            )
        })
        .unwrap_or(false)
}

fn b12x_w4a16_capacity_rows(rows: usize) -> Result<usize> {
    anyhow::ensure!(
        rows > 0 && rows <= B12X_W4A16_PREFILL_TOPK8_CAPACITY_ROWS,
        "B12X packed W4A16 rows {rows} are outside 1..={B12X_W4A16_PREFILL_TOPK8_CAPACITY_ROWS}"
    );
    Ok(rows.next_power_of_two().max(2))
}

fn ds4_flash_w4a16_capacity_rows(rows: usize) -> Result<usize> {
    anyhow::ensure!(
        rows > 0 && rows <= DS4_FLASH_PREFILL_MAX_ROWS,
        "native Flash W4A16 rows {rows} are outside 1..={DS4_FLASH_PREFILL_MAX_ROWS}"
    );
    Ok(if rows == 1 {
        1
    } else {
        rows.next_power_of_two()
    })
}

fn b12x_w4a16_prefill_route_block_rows(rows: usize) -> usize {
    if (2..=8).contains(&rows) {
        8
    } else if rows <= 2048 {
        32
    } else {
        48
    }
}

fn b12x_spark_route_lane_count() -> usize {
    env::var(REAL_FULL_B12X_SPARK_ROUTE_LANES_ENV)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(B12X_SPARK_ROUTE_MAX_LANES)
        .clamp(1, B12X_SPARK_ROUTE_MAX_LANES)
}

fn should_use_grouped_route_launches(row_count: usize, route_group_count: usize) -> bool {
    row_count == 1 || route_group_count == 1 || (row_count > 1 && route_grouped_multirow_enabled())
}

fn coalesce_streaming_completion_slices(
    indexed_slices: &[(usize, Vec<usize>)],
    first_response_rows: usize,
    max_response_rows: usize,
) -> Result<(Vec<usize>, Vec<Vec<usize>>)> {
    anyhow::ensure!(
        first_response_rows > 0 && first_response_rows <= max_response_rows,
        "streaming first response rows must be in 1..={max_response_rows}"
    );
    let mut emission_group_indices = Vec::new();
    let mut response_slices = Vec::new();
    let mut pending_rows = Vec::new();
    let mut pending_last_group = 0_usize;

    for (slice_index, (group_index, rows)) in indexed_slices.iter().enumerate() {
        anyhow::ensure!(
            !rows.is_empty() && rows.len() <= max_response_rows,
            "streaming completion slice rows {} must be in 1..={max_response_rows}",
            rows.len()
        );
        if !pending_rows.is_empty()
            && pending_rows.len().saturating_add(rows.len()) > max_response_rows
        {
            emission_group_indices.push(pending_last_group);
            response_slices.push(std::mem::take(&mut pending_rows));
        }
        pending_rows.extend_from_slice(rows);
        pending_last_group = *group_index;
        let target_rows = if response_slices.is_empty() {
            first_response_rows
        } else {
            max_response_rows
        };
        let final_input_slice = slice_index + 1 == indexed_slices.len();
        if pending_rows.len() >= target_rows || final_input_slice {
            emission_group_indices.push(pending_last_group);
            response_slices.push(std::mem::take(&mut pending_rows));
        }
    }
    anyhow::ensure!(
        pending_rows.is_empty(),
        "streaming completion coalescer left pending rows"
    );
    Ok((emission_group_indices, response_slices))
}

fn b12x_spark_direct_route_shape_supported(
    rows: usize,
    hidden_dim: usize,
    hidden_row_stride_elems: usize,
    intermediate_rows: usize,
    output_rows: usize,
) -> bool {
    rows > 0
        && rows <= B12X_SPARK_AOT_MAX_ROWS
        && hidden_dim == 6144
        && matches!(intermediate_rows, 512 | 2048)
        && output_rows == 6144
        && hidden_row_stride_elems == hidden_dim
}

fn checked_matrix_bytes(
    rows: usize,
    cols: usize,
    element_bytes: usize,
    label: &str,
) -> Result<usize> {
    rows.checked_mul(cols)
        .and_then(|values| values.checked_mul(element_bytes))
        .with_context(|| format!("{label} byte count overflow"))
}

fn b12x_projection_scale_shape_supported(rows: usize, scale_cols: usize) -> bool {
    matches!(
        (rows, scale_cols),
        (512, 384) | (2048, 384) | (6144, 32) | (6144, 128)
    )
}

fn route_stage_timing_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env_flag_enabled(REAL_FULL_NVFP4_ROUTE_TIMING_ENV)
            || env_flag_enabled(REAL_FULL_PROTOCOL_V2_EXECUTOR_TIMING_ENV)
            || env_flag_enabled(REAL_FULL_NVFP4_ROUTE_CUDA_EVENT_TIMING_ENV)
    })
}

fn env_flag_enabled(name: &str) -> bool {
    env::var(name)
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
}

fn cuda_projection_preload_progress_interval() -> usize {
    env::var("DS41RT_REAL_FULL_NVFP4_ROUTE_PRELOAD_PROGRESS_INTERVAL")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(512)
}

#[cfg(test)]
struct CudaRouteValidationTestOverride {
    previous: Option<bool>,
}

#[cfg(test)]
impl Drop for CudaRouteValidationTestOverride {
    fn drop(&mut self) {
        CUDA_ROUTE_VALIDATION_TEST_OVERRIDE.with(|value| {
            value.set(self.previous);
        });
    }
}

#[cfg(test)]
fn cuda_route_validation_test_override(enabled: bool) -> CudaRouteValidationTestOverride {
    CudaRouteValidationTestOverride {
        previous: CUDA_ROUTE_VALIDATION_TEST_OVERRIDE.with(|value| {
            let previous = value.get();
            value.set(Some(enabled));
            previous
        }),
    }
}

fn f32_values_to_bf16_bytes(values: &[f32], out: &mut [u8]) {
    for (value, dst) in values.iter().zip(out.chunks_exact_mut(2)) {
        let bf16 = (value.to_bits() >> 16) as u16;
        dst.copy_from_slice(&bf16.to_le_bytes());
    }
}

struct OwnedDeviceAllocation {
    library: Arc<NativeLibrary>,
    buffer: Ds41rtDeviceBuffer,
}

impl OwnedDeviceAllocation {
    fn new(library: Arc<NativeLibrary>, bytes: usize, label: &str) -> Result<Self> {
        Self::new_with_kind(library, bytes, label, false)
    }

    fn new_with_kind(
        library: Arc<NativeLibrary>,
        bytes: usize,
        label: &str,
        managed: bool,
    ) -> Result<Self> {
        let buffer = if managed {
            library.alloc_managed_device_buffer(bytes)
        } else {
            library.alloc_device_buffer(bytes)
        }
        .with_context(|| {
            if managed {
                format!("allocating managed device buffer for {label}")
            } else {
                format!("allocating device buffer for {label}")
            }
        })?;
        Ok(Self { library, buffer })
    }

    fn buffer(&self) -> Ds41rtDeviceBuffer {
        self.buffer
    }

    fn capacity_bytes(&self) -> usize {
        self.buffer.bytes
    }

    fn is_managed(&self) -> bool {
        (self.buffer.flags & DS41RT_DEVICE_BUFFER_FLAG_MANAGED) != 0
    }

    fn copy_host_bytes_direct_at(
        &self,
        offset_bytes: usize,
        bytes: &[u8],
        label: &str,
    ) -> Result<()> {
        anyhow::ensure!(
            self.is_managed(),
            "direct host copy for {label} requires a managed device allocation"
        );
        if self.buffer.ptr.is_null() {
            anyhow::bail!("managed device buffer for {label} is null");
        }
        let end = offset_bytes
            .checked_add(bytes.len())
            .with_context(|| format!("managed device buffer offset overflow for {label}"))?;
        if self.buffer.bytes < end {
            anyhow::bail!(
                "managed device buffer for {label} has {} bytes, needs {end}",
                self.buffer.bytes,
            );
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                self.buffer.ptr.cast::<u8>().add(offset_bytes),
                bytes.len(),
            );
        }
        Ok(())
    }
}

impl Drop for OwnedDeviceAllocation {
    fn drop(&mut self) {
        let _ = self.library.free_device_buffer(&mut self.buffer);
    }
}

// CUDA device pointers are opaque runtime handles here. Rust never dereferences
// them directly; allocation, copy, kernel use, and free all go through NativeLibrary.
unsafe impl Send for OwnedDeviceAllocation {}
unsafe impl Sync for OwnedDeviceAllocation {}

struct OwnedPinnedHostAllocation {
    library: Arc<NativeLibrary>,
    buffer: Ds41rtHostBuffer,
}

impl OwnedPinnedHostAllocation {
    fn new(library: Arc<NativeLibrary>, bytes: usize, label: &str) -> Result<Self> {
        let buffer = library
            .alloc_host_buffer(bytes)
            .with_context(|| format!("allocating pinned host staging buffer for {label}"))?;
        if buffer.ptr.is_null() {
            anyhow::bail!("pinned host staging buffer for {label} is null");
        }
        if buffer.bytes < bytes {
            anyhow::bail!(
                "pinned host staging buffer for {label} has {} bytes, needs {}",
                buffer.bytes,
                bytes
            );
        }
        Ok(Self { library, buffer })
    }

    fn as_mut_slice(&mut self, len: usize) -> Result<&mut [u8]> {
        if self.buffer.ptr.is_null() {
            anyhow::bail!("pinned host staging buffer is null");
        }
        if self.buffer.bytes < len {
            anyhow::bail!(
                "pinned host staging buffer has {} bytes, needs {len}",
                self.buffer.bytes
            );
        }
        Ok(unsafe { slice::from_raw_parts_mut(self.buffer.ptr.cast::<u8>(), len) })
    }

    fn buffer(&self) -> Ds41rtHostBuffer {
        self.buffer
    }

    fn capacity_bytes(&self) -> usize {
        self.buffer.bytes
    }
}

impl Drop for OwnedPinnedHostAllocation {
    fn drop(&mut self) {
        let _ = self.library.free_host_buffer(&mut self.buffer);
    }
}

// Pinned host pointers are opaque native allocations. Rust only copies bytes
// into them and passes the handles back through NativeLibrary calls.
unsafe impl Send for OwnedPinnedHostAllocation {}
unsafe impl Sync for OwnedPinnedHostAllocation {}

fn native_library_path() -> Option<PathBuf> {
    if let Ok(path) = env::var("DS41RT_NATIVE_LIB") {
        return Some(PathBuf::from(path));
    }
    if env::var_os("DS41RT_DISABLE_NATIVE_AUTO_DISCOVERY").is_some() {
        return None;
    }
    native_library_path_candidates()
        .into_iter()
        .find(|path| path.exists())
}

fn native_library_path_candidates() -> Vec<PathBuf> {
    let manifest_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .join("native");
    vec![
        manifest_path.join("build-cuda/libds41rt_native.so"),
        PathBuf::from("native/build-cuda/libds41rt_native.so"),
    ]
}

fn f32_bytes(values: &[f32]) -> &[u8] {
    unsafe { slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

fn u32_bytes(values: &[u32]) -> &[u8] {
    unsafe { slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

fn route_metadata_bytes(values: &[Ds41rtNvfp4RouteBatchedMetadata]) -> &[u8] {
    unsafe { slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

fn device_buffer_byte_view(
    buffer: Ds41rtDeviceBuffer,
    offset_bytes: usize,
    required_bytes: usize,
    label: &str,
) -> Result<Ds41rtDeviceBuffer> {
    let end = offset_bytes
        .checked_add(required_bytes)
        .with_context(|| format!("{label} device view byte range overflow"))?;
    if end > buffer.bytes {
        anyhow::bail!(
            "{label} device view byte range {}..{} exceeds buffer bytes {}",
            offset_bytes,
            end,
            buffer.bytes
        );
    }
    let ptr = unsafe { buffer.ptr.cast::<u8>().add(offset_bytes).cast::<c_void>() };
    Ok(Ds41rtDeviceBuffer {
        ptr,
        bytes: buffer.bytes - offset_bytes,
        device_id: buffer.device_id,
        flags: buffer.flags,
    })
}

fn host_buffer_byte_view(
    buffer: Ds41rtHostBuffer,
    offset_bytes: usize,
    required_bytes: usize,
    label: &str,
) -> Result<Ds41rtHostBuffer> {
    let end = offset_bytes
        .checked_add(required_bytes)
        .with_context(|| format!("{label} host view byte range overflow"))?;
    if end > buffer.bytes {
        anyhow::bail!(
            "{label} host view byte range {}..{} exceeds buffer bytes {}",
            offset_bytes,
            end,
            buffer.bytes
        );
    }
    let ptr = unsafe { buffer.ptr.cast::<u8>().add(offset_bytes).cast::<c_void>() };
    Ok(Ds41rtHostBuffer {
        ptr,
        bytes: buffer.bytes - offset_bytes,
        flags: buffer.flags,
    })
}

fn load_routed_quant_projection_cached(
    catalog: &TensorCatalog,
    layer_id: usize,
    expert_id: usize,
    projection: &'static str,
    row_count: usize,
    cache: &mut RouteTensorCache,
) -> Result<Arc<RoutedQuantProjection>> {
    let key = RoutedQuantProjectionKey {
        layer_id,
        expert_id,
        projection,
        row_count,
    };
    if let Some(projection) = cache.projections.get(&key) {
        cache.cache_hits += 1;
        return Ok(Arc::clone(projection));
    }

    let loaded = load_routed_quant_projection(catalog, layer_id, expert_id, projection, row_count)?;
    cache.projection_loads += 1;
    let loaded = Arc::new(loaded);
    seed_routed_quant_scalar_metadata_cache(
        layer_id,
        expert_id,
        projection,
        loaded.as_ref(),
        cache,
    )?;
    cache.projections.insert(key, Arc::clone(&loaded));
    Ok(loaded)
}

fn load_routed_quant_scalar_metadata_cached(
    catalog: &TensorCatalog,
    layer_id: usize,
    expert_id: usize,
    projection: &'static str,
    cache: &mut RouteTensorCache,
) -> Result<RoutedQuantScalarMetadata> {
    let key = RoutedQuantScalarMetadataKey {
        layer_id,
        expert_id,
        projection,
    };
    if let Some(metadata) = cache.scalar_metadata.get(&key) {
        cache.scalar_metadata_cache_hits += 1;
        return Ok(metadata.clone());
    }

    let metadata = load_routed_quant_scalar_metadata(catalog, layer_id, expert_id, projection)?;
    cache.scalar_metadata_loads += 1;
    cache.scalar_metadata.insert(key, metadata.clone());
    Ok(metadata)
}

fn seed_routed_quant_scalar_metadata_cache(
    layer_id: usize,
    expert_id: usize,
    projection: &'static str,
    loaded: &RoutedQuantProjection,
    cache: &mut RouteTensorCache,
) -> Result<()> {
    let key = RoutedQuantScalarMetadataKey {
        layer_id,
        expert_id,
        projection,
    };
    if cache.scalar_metadata.contains_key(&key) {
        return Ok(());
    }
    let metadata =
        routed_quant_scalar_metadata_from_loaded(&loaded.input_scale, &loaded.weight_scale_2)?;
    cache.scalar_metadata.insert(key, metadata);
    Ok(())
}

pub(in crate::commands::real_full) fn preload_routed_quant_projection_host_cache(
    catalog: &TensorCatalog,
    layer_id: usize,
    expert_id: usize,
    projection: &'static str,
    row_count: usize,
    cache: &mut RouteTensorCache,
) -> Result<RouteHostProjectionPreload> {
    cache.prepare_layer(layer_id);
    let loaded = load_routed_quant_projection_cached(
        catalog, layer_id, expert_id, projection, row_count, cache,
    )?;
    Ok(RouteHostProjectionPreload {
        weight_bytes: loaded.weight.bytes.len() as u64,
        quant_metadata_bytes: loaded.weight_scale.bytes.len() as u64
            + loaded.input_scale.bytes.len() as u64
            + loaded.weight_scale_2.bytes.len() as u64,
    })
}

pub(in crate::commands::real_full) fn preload_routed_quant_projection_scalar_cache_parallel(
    catalog: &TensorCatalog,
    requests: &[RouteProjectionCachePreloadRequest],
    cache: &mut RouteTensorCache,
) -> Result<RouteHostProjectionPreload> {
    anyhow::ensure!(
        !requests.is_empty(),
        "parallel routed scalar preload requires projection requests"
    );
    #[derive(Clone)]
    struct ScalarLocation {
        key: RoutedQuantScalarMetadataKey,
        name: String,
        byte_offset: u64,
        input_scale: bool,
    }

    let snapshot = Path::new(&catalog.snapshot_path);
    let mut locations_by_file = BTreeMap::<PathBuf, Vec<ScalarLocation>>::new();
    for request in requests {
        let base_name = routed_quant_projection_base_name(
            request.layer_id,
            request.expert_id,
            request.projection,
        );
        for (suffix, input_scale) in [("input_scale", true), ("weight_scale_2", false)] {
            let name = format!("{base_name}.{suffix}");
            let tensor = catalog_tensor(catalog, &name)?;
            anyhow::ensure!(
                tensor.dtype == DType::F32 && tensor.shape.is_empty() && tensor.byte_length == 4,
                "routed scalar tensor {name} must be a four-byte F32 scalar"
            );
            locations_by_file
                .entry(snapshot.join(&tensor.file))
                .or_default()
                .push(ScalarLocation {
                    key: RoutedQuantScalarMetadataKey {
                        layer_id: request.layer_id,
                        expert_id: request.expert_id,
                        projection: request.projection,
                    },
                    name,
                    byte_offset: tensor.byte_offset,
                    input_scale,
                });
        }
    }

    type ScalarPair = (Option<(String, f32)>, Option<(String, f32)>);
    let mut values = HashMap::<RoutedQuantScalarMetadataKey, ScalarPair>::new();
    for (path, locations) in locations_by_file {
        let first = locations
            .iter()
            .map(|location| location.byte_offset)
            .min()
            .context("routed scalar file group is empty")?;
        let end = locations
            .iter()
            .map(|location| location.byte_offset + 4)
            .max()
            .context("routed scalar file group is empty")?;
        let span: usize = (end - first)
            .try_into()
            .context("routed scalar file span does not fit in memory")?;
        let mut bytes = vec![0_u8; span];
        File::open(&path)
            .with_context(|| format!("opening routed scalar file {}", path.display()))?
            .read_exact_at(&mut bytes, first)
            .with_context(|| {
                format!(
                    "reading routed scalar span {first}..{end} from {}",
                    path.display()
                )
            })?;
        for location in locations {
            let offset: usize = (location.byte_offset - first)
                .try_into()
                .context("routed scalar offset does not fit in memory")?;
            let value = f32::from_le_bytes(
                bytes[offset..offset + 4]
                    .try_into()
                    .expect("routed scalar slice has four bytes"),
            );
            validate_finite_route_scalar(&location.name, value)?;
            let pair = values.entry(location.key).or_default();
            let destination = if location.input_scale {
                &mut pair.0
            } else {
                &mut pair.1
            };
            anyhow::ensure!(
                destination.replace((location.name, value)).is_none(),
                "routed scalar metadata was loaded twice"
            );
        }
    }

    for request in requests {
        let key = RoutedQuantScalarMetadataKey {
            layer_id: request.layer_id,
            expert_id: request.expert_id,
            projection: request.projection,
        };
        let (input_scale, weight_scale_2) = values.remove(&key).with_context(|| {
            format!(
                "bulk scalar preload did not fill layer {} expert {} {}",
                request.layer_id, request.expert_id, request.projection
            )
        })?;
        let (input_scale_name, input_scale) =
            input_scale.context("bulk scalar preload is missing input_scale")?;
        let (weight_scale_2_name, weight_scale_2) =
            weight_scale_2.context("bulk scalar preload is missing weight_scale_2")?;
        let metadata = RoutedQuantScalarMetadata {
            input_scale_name,
            weight_scale_2_name,
            input_scale,
            weight_scale_2,
        };
        validate_finite_route_scalar(&metadata.input_scale_name, metadata.input_scale)?;
        validate_finite_route_scalar(&metadata.weight_scale_2_name, metadata.weight_scale_2)?;
        if cache.scalar_metadata.insert(key, metadata).is_none() {
            cache.scalar_metadata_loads += 1;
        } else {
            cache.scalar_metadata_cache_hits += 1;
        }
        cache.prepare_layer(request.layer_id);
    }
    Ok(RouteHostProjectionPreload {
        weight_bytes: 0,
        quant_metadata_bytes: (requests.len() as u64) * 8,
    })
}

#[allow(clippy::too_many_arguments)]
pub(in crate::commands::real_full) fn preload_bf16_route_projection_group_cache(
    catalog: &TensorCatalog,
    layer_id: usize,
    expert_id: usize,
    intermediate_rows: usize,
    output_rows: usize,
    hidden_dim: usize,
    cache: &mut RouteTensorCache,
) -> Result<()> {
    cache.prepare_layer(layer_id);
    let cache_key = Bf16RouteProjectionGroupCacheKey {
        layer_id,
        expert_id,
        intermediate_rows,
        output_rows,
        hidden_dim,
        require_host_tensors: cuda_route_validation_enabled(),
    };
    if cache.bf16_projection_groups.contains_key(&cache_key) {
        return Ok(());
    }
    let projections = if cuda_route_validation_enabled() {
        load_validated_bf16_route_projections(
            catalog,
            layer_id,
            expert_id,
            intermediate_rows,
            output_rows,
            hidden_dim,
            cache,
        )?
    } else {
        let projection = |projection, row_count, input_width| -> Result<_> {
            let metadata = cache
                .scalar_metadata
                .get(&RoutedQuantScalarMetadataKey {
                    layer_id,
                    expert_id,
                    projection,
                })
                .with_context(|| {
                    format!(
                        "startup route group is missing layer {layer_id} expert {expert_id} {projection} scalar metadata"
                    )
                })?;
            anyhow::ensure!(
                row_count > 0 && input_width > 0 && input_width % 16 == 0,
                "startup route group has invalid {projection} geometry {row_count}x{input_width}"
            );
            Ok((
                Bf16RouteProjection {
                    key: RoutedQuantProjectionKey {
                        layer_id,
                        expert_id,
                        projection,
                        row_count,
                    },
                    host: None,
                },
                metadata.weight_scale_2,
            ))
        };
        let (gate, gate_scale_2) = projection("gate_proj", intermediate_rows, hidden_dim)?;
        let (up, up_scale_2) = projection("up_proj", intermediate_rows, hidden_dim)?;
        let (down, down_scale_2) = projection("down_proj", output_rows, intermediate_rows)?;
        Bf16RouteProjections {
            gate,
            up,
            down,
            gate_scale_2,
            up_scale_2,
            down_scale_2,
        }
    };
    cache.bf16_projection_groups.insert(cache_key, projections);
    Ok(())
}

fn load_route_cuda_bf16_projection_shard(
    catalog: &TensorCatalog,
    request: &RouteProjectionCachePreloadRequest,
    shard: ExpertIntermediateShard,
) -> Result<LoadedTensorRows> {
    let base_name =
        routed_quant_projection_base_name(request.layer_id, request.expert_id, request.projection);
    let loaded = load_routed_projection_rows_for_shard(
        catalog,
        &format!("{base_name}.weight"),
        request.projection,
        request.row_count,
        shard,
    )?;
    anyhow::ensure!(
        loaded.info.dtype == DType::Bf16 && loaded.bytes_per_scalar == 2,
        "retained layer {} expert {} {} has dtype {:?}, expected BF16",
        request.layer_id,
        request.expert_id,
        request.projection,
        loaded.info.dtype,
    );
    Ok(loaded)
}

fn load_route_cuda_bf16_expert_shard(
    catalog: &TensorCatalog,
    layer_requests: &[&RouteProjectionCachePreloadRequest],
    layer_id: usize,
    expert_id: usize,
    shard: ExpertIntermediateShard,
) -> Result<LoadedRouteCudaBf16ExpertShard> {
    let find = |projection| {
        layer_requests
            .iter()
            .copied()
            .find(|request| request.expert_id == expert_id && request.projection == projection)
            .with_context(|| {
                format!("retained BF16 layer {layer_id} expert {expert_id} is missing {projection}")
            })
    };
    Ok(LoadedRouteCudaBf16ExpertShard {
        gate: load_route_cuda_bf16_projection_shard(catalog, find("gate_proj")?, shard)?,
        up: load_route_cuda_bf16_projection_shard(catalog, find("up_proj")?, shard)?,
        down: load_route_cuda_bf16_projection_shard(catalog, find("down_proj")?, shard)?,
    })
}

pub(in crate::commands::real_full) fn preload_routed_bf16_projection_cuda_cache(
    catalog: &TensorCatalog,
    requests: &[RouteProjectionCachePreloadRequest],
    cache: &mut RouteTensorCache,
) -> Result<RouteCudaProjectionPreload> {
    anyhow::ensure!(
        !requests.is_empty(),
        "retained BF16 CUDA resident preload requires projection requests"
    );
    let shard = spark_expert_intermediate_shard_from_env()?
        .context("retained BF16 MTP experts require four intermediate shards")?;
    let cuda_cache = cache.cuda_cache()?;
    let stream = RouteCudaStream::new(Arc::clone(&cuda_cache.library))?;
    let cuda_stream = stream.as_ptr();
    let mut layers = requests
        .iter()
        .map(|request| request.layer_id)
        .collect::<Vec<_>>();
    layers.sort_unstable();
    layers.dedup();
    let mut preload = RouteCudaProjectionPreload::default();
    for layer_id in layers {
        let layer_requests = requests
            .iter()
            .filter(|request| request.layer_id == layer_id)
            .collect::<Vec<_>>();
        let mut expert_ids = layer_requests
            .iter()
            .map(|request| request.expert_id)
            .collect::<Vec<_>>();
        expert_ids.sort_unstable();
        expert_ids.dedup();
        anyhow::ensure!(
            !expert_ids.is_empty()
                && expert_ids.iter().copied().eq(0..expert_ids.len())
                && layer_requests.len() == expert_ids.len() * 3,
            "retained BF16 layer {layer_id} requires three projections for dense expert IDs"
        );
        if let Some(existing) = cuda_cache.bf16_expert_slabs.get(&layer_id) {
            preload.projection_groups += existing.expert_count * 3;
            preload.weight_bytes +=
                (existing.w13_weight.capacity_bytes() + existing.w2_weight.capacity_bytes()) as u64;
            continue;
        }
        let mut first = Some(load_route_cuda_bf16_expert_shard(
            catalog,
            &layer_requests,
            layer_id,
            0,
            shard,
        )?);
        let slab = Arc::new(RouteCudaBf16LayerExpertSlab::new(
            Arc::clone(&cuda_cache.library),
            layer_id,
            expert_ids.len(),
            first.as_ref().expect("first BF16 expert loaded above"),
        )?);
        for expert_id in expert_ids {
            let loaded = if expert_id == 0 {
                first.take().expect("first BF16 expert is consumed once")
            } else {
                load_route_cuda_bf16_expert_shard(
                    catalog,
                    &layer_requests,
                    layer_id,
                    expert_id,
                    shard,
                )?
            };
            preload.weight_bytes += slab.store_expert(
                expert_id,
                &loaded,
                Arc::clone(&cuda_cache.library),
                &mut cuda_cache.workspace,
                cuda_stream,
            )?;
            preload.projection_groups += 3;
            cuda_cache.projection_uploads += 3;
        }
        anyhow::ensure!(
            cuda_cache
                .bf16_expert_slabs
                .insert(layer_id, slab)
                .is_none(),
            "retained BF16 layer {layer_id} was preloaded twice"
        );
    }
    Ok(preload)
}

pub(in crate::commands::real_full) fn preload_startup_quantized_bf16_projection_cuda_cache(
    catalog: &TensorCatalog,
    requests: &[RouteProjectionCachePreloadRequest],
    cache: &mut RouteTensorCache,
) -> Result<RouteCudaProjectionPreload> {
    anyhow::ensure!(
        !requests.is_empty(),
        "startup BF16-to-NVFP4 preload requires projection requests"
    );
    let shard = spark_expert_intermediate_shard_from_env()?
        .context("startup-quantized MTP experts require four intermediate shards")?;
    let cuda_cache = cache.cuda_cache()?;
    anyhow::ensure!(
        cuda_cache.b12x_w4a16_packed,
        "startup BF16-to-NVFP4 experts require the packed W4A16 serving layout"
    );
    let stream = RouteCudaStream::new(Arc::clone(&cuda_cache.library))?;
    let cuda_stream = stream.as_ptr();
    let managed_weights = !b12x_spark_w4a16_device_weights_enabled();
    let mut layers = requests
        .iter()
        .map(|request| request.layer_id)
        .collect::<Vec<_>>();
    layers.sort_unstable();
    layers.dedup();
    let mut preload = RouteCudaProjectionPreload::default();
    for layer_id in layers {
        let layer_requests = requests
            .iter()
            .filter(|request| request.layer_id == layer_id)
            .collect::<Vec<_>>();
        let mut expert_ids = layer_requests
            .iter()
            .map(|request| request.expert_id)
            .collect::<Vec<_>>();
        expert_ids.sort_unstable();
        expert_ids.dedup();
        anyhow::ensure!(
            !expert_ids.is_empty()
                && expert_ids.iter().copied().eq(0..expert_ids.len())
                && layer_requests.len() == expert_ids.len() * 3,
            "startup-quantized BF16 layer {layer_id} requires three projections for dense expert IDs"
        );
        if let Some(existing) = cuda_cache.expert_slabs.get(&layer_id) {
            preload.projection_groups += existing.expert_count * 3;
            preload.weight_bytes +=
                (existing.w13_weight.capacity_bytes() + existing.w2_weight.capacity_bytes()) as u64;
            preload.weight_scale_bytes +=
                (existing.w13_scale.capacity_bytes() + existing.w2_scale.capacity_bytes()) as u64;
            continue;
        }
        let mut first = Some(load_route_cuda_bf16_expert_shard(
            catalog,
            &layer_requests,
            layer_id,
            0,
            shard,
        )?);
        let geometry = synthetic_quantized_expert_geometry(
            first.as_ref().expect("first BF16 expert loaded above"),
        )?;
        let slab = Arc::new(RouteCudaLayerExpertSlab::new(
            Arc::clone(&cuda_cache.library),
            layer_id,
            expert_ids.len(),
            &geometry,
            managed_weights,
            true,
        )?);
        for expert_id in expert_ids {
            let loaded = if expert_id == 0 {
                first.take().expect("first BF16 expert is consumed once")
            } else {
                load_route_cuda_bf16_expert_shard(
                    catalog,
                    &layer_requests,
                    layer_id,
                    expert_id,
                    shard,
                )?
            };
            let (weight_bytes, scale_bytes) = slab.store_expert_startup_quantized_bf16(
                expert_id,
                &loaded,
                Arc::clone(&cuda_cache.library),
                &mut cuda_cache.workspace,
                cuda_stream,
            )?;
            preload.projection_groups += 3;
            preload.weight_bytes += weight_bytes;
            preload.weight_scale_bytes += scale_bytes;
            cuda_cache.projection_uploads += 3;
        }
        slab.finalize_w4a16_global_scales(cuda_cache.library.as_ref(), cuda_stream)?;
        anyhow::ensure!(
            cuda_cache.expert_slabs.insert(layer_id, slab).is_none(),
            "startup-quantized BF16 layer {layer_id} was preloaded twice"
        );
    }
    Ok(preload)
}

fn route_preload_io_workers() -> usize {
    env::var(REAL_FULL_NVFP4_ROUTE_PRELOAD_IO_WORKERS_ENV)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(128)
        .clamp(1, 128)
}

fn route_preload_direct_io() -> bool {
    env::var(REAL_FULL_NVFP4_ROUTE_PRELOAD_DIRECT_IO_ENV)
        .ok()
        .map(|value| !matches!(value.trim(), "0" | "false" | "off"))
        .unwrap_or(true)
}

const ROUTE_PRELOAD_DIRECT_IO_ALIGNMENT: usize = 4096;

struct RouteCudaAlignedAllocation {
    ptr: NonNull<u8>,
    layout: std::alloc::Layout,
}

impl Drop for RouteCudaAlignedAllocation {
    fn drop(&mut self) {
        unsafe {
            std::alloc::dealloc(self.ptr.as_ptr(), self.layout);
        }
    }
}

unsafe impl Send for RouteCudaAlignedAllocation {}
unsafe impl Sync for RouteCudaAlignedAllocation {}

fn route_cuda_aligned_read_pool() -> &'static Mutex<Vec<RouteCudaAlignedAllocation>> {
    static POOL: OnceLock<Mutex<Vec<RouteCudaAlignedAllocation>>> = OnceLock::new();
    POOL.get_or_init(|| Mutex::new(Vec::new()))
}

fn clear_route_cuda_aligned_read_pool() {
    if let Ok(mut pool) = route_cuda_aligned_read_pool().lock() {
        pool.clear();
    }
}

struct RouteCudaAlignedReadBuffer {
    allocation: Option<RouteCudaAlignedAllocation>,
    aligned_bytes: usize,
    requested_offset: usize,
    requested_bytes: usize,
}

impl RouteCudaAlignedReadBuffer {
    fn aligned_geometry(
        file_bytes: u64,
        source_offset: u64,
        source_bytes: usize,
    ) -> Result<Option<(u64, usize, usize)>> {
        let alignment = ROUTE_PRELOAD_DIRECT_IO_ALIGNMENT as u64;
        let aligned_offset = source_offset / alignment * alignment;
        let requested_offset: usize = (source_offset - aligned_offset)
            .try_into()
            .context("direct route read prefix does not fit in usize")?;
        let requested_end = requested_offset
            .checked_add(source_bytes)
            .context("direct route read requested byte end overflow")?;
        let aligned_bytes = requested_end
            .checked_add(ROUTE_PRELOAD_DIRECT_IO_ALIGNMENT - 1)
            .context("direct route read aligned byte length overflow")?
            / ROUTE_PRELOAD_DIRECT_IO_ALIGNMENT
            * ROUTE_PRELOAD_DIRECT_IO_ALIGNMENT;
        let aligned_end = aligned_offset
            .checked_add(aligned_bytes as u64)
            .context("direct route read aligned file end overflow")?;
        Ok(
            (aligned_end <= file_bytes).then_some((
                aligned_offset,
                aligned_bytes,
                requested_offset,
            )),
        )
    }

    fn allocate(
        file_bytes: u64,
        source_offset: u64,
        source_bytes: usize,
    ) -> Result<Option<(Self, u64)>> {
        let Some((aligned_offset, aligned_bytes, requested_offset)) =
            Self::aligned_geometry(file_bytes, source_offset, source_bytes)?
        else {
            return Ok(None);
        };
        Ok(Some((
            Self::allocate_geometry(aligned_bytes, requested_offset, source_bytes)?,
            aligned_offset,
        )))
    }

    fn allocate_geometry(
        aligned_bytes: usize,
        requested_offset: usize,
        requested_bytes: usize,
    ) -> Result<Self> {
        let layout =
            std::alloc::Layout::from_size_align(aligned_bytes, ROUTE_PRELOAD_DIRECT_IO_ALIGNMENT)
                .context("building direct route read allocation layout")?;
        let mut allocation = {
            let mut pool = route_cuda_aligned_read_pool()
                .lock()
                .map_err(|_| anyhow::anyhow!("direct route read allocation pool is poisoned"))?;
            let candidate = pool
                .iter()
                .enumerate()
                .filter(|(_, allocation)| allocation.layout.size() >= aligned_bytes)
                .min_by_key(|(_, allocation)| allocation.layout.size())
                .map(|(index, _)| index);
            candidate.map(|index| pool.swap_remove(index))
        };
        if allocation.is_none() {
            let ptr = NonNull::new(unsafe { std::alloc::alloc(layout) })
                .context("allocating direct route read buffer")?;
            allocation = Some(RouteCudaAlignedAllocation { ptr, layout });
        }
        Ok(Self {
            allocation,
            aligned_bytes,
            requested_offset,
            requested_bytes,
        })
    }

    fn new(
        file: &File,
        file_bytes: u64,
        source_offset: u64,
        source_bytes: usize,
    ) -> Result<Option<Self>> {
        let Some((mut buffer, aligned_offset)) =
            Self::allocate(file_bytes, source_offset, source_bytes)?
        else {
            return Ok(None);
        };
        if let Err(error) = file.read_exact_at(buffer.full_slice_mut(), aligned_offset) {
            return Err(error).with_context(|| {
                format!(
                    "direct-reading {} bytes at aligned offset {aligned_offset}",
                    buffer.full_bytes()
                )
            });
        }
        Ok(Some(buffer))
    }

    fn new_with_buffered_tail(
        direct_file: &File,
        buffered_file: &File,
        file_bytes: u64,
        source_offset: u64,
        source_bytes: usize,
    ) -> Result<Self> {
        let alignment = ROUTE_PRELOAD_DIRECT_IO_ALIGNMENT as u64;
        let aligned_offset = source_offset / alignment * alignment;
        let requested_offset: usize = (source_offset - aligned_offset)
            .try_into()
            .context("coalesced route read prefix exceeds usize")?;
        let requested_end = requested_offset
            .checked_add(source_bytes)
            .context("coalesced route read requested byte end overflow")?;
        let aligned_bytes = requested_end
            .checked_add(ROUTE_PRELOAD_DIRECT_IO_ALIGNMENT - 1)
            .context("coalesced route read aligned byte length overflow")?
            / ROUTE_PRELOAD_DIRECT_IO_ALIGNMENT
            * ROUTE_PRELOAD_DIRECT_IO_ALIGNMENT;
        let source_end = source_offset
            .checked_add(
                source_bytes
                    .try_into()
                    .context("coalesced route source length exceeds u64")?,
            )
            .context("coalesced route source extent overflow")?;
        anyhow::ensure!(
            source_end <= file_bytes,
            "coalesced route source extent exceeds its file"
        );
        let direct_bytes: usize = ((file_bytes - aligned_offset) / alignment * alignment)
            .min(
                aligned_bytes
                    .try_into()
                    .context("coalesced route aligned length exceeds u64")?,
            )
            .try_into()
            .context("coalesced route direct length exceeds usize")?;
        let mut buffer = Self::allocate_geometry(aligned_bytes, requested_offset, source_bytes)?;
        if direct_bytes > 0 {
            direct_file
                .read_exact_at(&mut buffer.full_slice_mut()[..direct_bytes], aligned_offset)
                .with_context(|| {
                    format!(
                        "direct-reading {direct_bytes} coalesced bytes at offset {aligned_offset}"
                    )
                })?;
        }
        let buffered_start = direct_bytes.max(requested_offset);
        if buffered_start < requested_end {
            let buffered_offset = aligned_offset
                .checked_add(
                    buffered_start
                        .try_into()
                        .context("coalesced route buffered offset exceeds u64")?,
                )
                .context("coalesced route buffered file offset overflow")?;
            buffered_file
                .read_exact_at(
                    &mut buffer.full_slice_mut()[buffered_start..requested_end],
                    buffered_offset,
                )
                .with_context(|| {
                    format!(
                        "buffered-reading {} coalesced tail bytes at offset {buffered_offset}",
                        requested_end - buffered_start
                    )
                })?;
        }
        Ok(buffer)
    }

    fn full_slice_mut(&mut self) -> &mut [u8] {
        let allocation = self
            .allocation
            .as_mut()
            .expect("direct route read allocation is present until drop");
        unsafe { slice::from_raw_parts_mut(allocation.ptr.as_ptr(), self.aligned_bytes) }
    }

    fn full_ptr(&mut self) -> *mut u8 {
        self.allocation
            .as_mut()
            .expect("direct route read allocation is present until drop")
            .ptr
            .as_ptr()
    }

    fn full_bytes(&self) -> usize {
        self.aligned_bytes
    }

    fn requested_slice(&self) -> &[u8] {
        let allocation = self
            .allocation
            .as_ref()
            .expect("direct route read allocation is present until drop");
        unsafe {
            slice::from_raw_parts(
                allocation.ptr.as_ptr().add(self.requested_offset),
                self.requested_bytes,
            )
        }
    }

    fn physical_bytes_for(file_bytes: u64, source_offset: u64, source_bytes: usize) -> Result<u64> {
        Ok(
            match Self::aligned_geometry(file_bytes, source_offset, source_bytes)? {
                Some((_, aligned_bytes, _)) => aligned_bytes as u64,
                None => source_bytes as u64,
            },
        )
    }
}

impl Drop for RouteCudaAlignedReadBuffer {
    fn drop(&mut self) {
        if let Some(allocation) = self.allocation.take() {
            if let Ok(mut pool) = route_cuda_aligned_read_pool().lock() {
                pool.push(allocation);
            }
        }
    }
}

unsafe impl Send for RouteCudaAlignedReadBuffer {}
unsafe impl Sync for RouteCudaAlignedReadBuffer {}

#[derive(Clone, Copy)]
enum RouteCudaLayerTensorSlot {
    GateWeight,
    GateWeightScale,
    UpWeight,
    UpWeightScale,
    DownWeight,
    DownWeightScale,
}

impl RouteCudaLayerTensorSlot {
    const COUNT: usize = 6;

    fn index(self) -> usize {
        match self {
            Self::GateWeight => 0,
            Self::GateWeightScale => 1,
            Self::UpWeight => 2,
            Self::UpWeightScale => 3,
            Self::DownWeight => 4,
            Self::DownWeightScale => 5,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::GateWeight => "gate weight",
            Self::GateWeightScale => "gate weight scale",
            Self::UpWeight => "up weight",
            Self::UpWeightScale => "up weight scale",
            Self::DownWeight => "down weight",
            Self::DownWeightScale => "down weight scale",
        }
    }
}

#[derive(Clone, Copy)]
struct RouteCudaLayerColumnWindow {
    start: usize,
    count: usize,
}

struct RouteCudaLayerTensorReadPlan {
    expert_index: usize,
    slot: RouteCudaLayerTensorSlot,
    info: TensorInfo,
    source_path: PathBuf,
    buffered_file: Arc<File>,
    direct_file: Option<Arc<File>>,
    file_bytes: u64,
    source_offset: u64,
    source_bytes: usize,
    output_start_row: usize,
    output_row_count: usize,
    output_row_width: usize,
    bytes_per_scalar: usize,
    column_window: Option<RouteCudaLayerColumnWindow>,
    retain_column_window_for_gpu_pack: bool,
}

impl RouteCudaLayerTensorReadPlan {
    fn load(&self) -> Result<LoadedRouteCudaTensorRows> {
        let started = Instant::now();
        let source = if let Some(file) = &self.direct_file {
            if let Some(buffer) = RouteCudaAlignedReadBuffer::new(
                file,
                self.file_bytes,
                self.source_offset,
                self.source_bytes,
            )
            .with_context(|| {
                format!(
                    "direct-reading {} from {}",
                    self.info.name,
                    self.source_path.display()
                )
            })? {
                RouteCudaTensorBytes::direct(buffer)
            } else {
                let mut bytes = vec![0_u8; self.source_bytes];
                self.buffered_file
                    .read_exact_at(&mut bytes, self.source_offset)
                    .with_context(|| {
                        format!(
                            "reading tail tensor {} from {} at offset {}",
                            self.info.name,
                            self.source_path.display(),
                            self.source_offset
                        )
                    })?;
                RouteCudaTensorBytes::owned(bytes)
            }
        } else {
            let mut bytes = vec![0_u8; self.source_bytes];
            self.buffered_file
                .read_exact_at(&mut bytes, self.source_offset)
                .with_context(|| {
                    format!(
                        "reading {} bytes for {} from {} at offset {}",
                        self.source_bytes,
                        self.info.name,
                        self.source_path.display(),
                        self.source_offset
                    )
                })?;
            RouteCudaTensorBytes::owned(bytes)
        };
        self.finish_load(source, started.elapsed().as_micros())
    }

    fn finish_load(
        &self,
        source: RouteCudaTensorBytes,
        elapsed_micros: u128,
    ) -> Result<LoadedRouteCudaTensorRows> {
        let bytes = if let Some(window) = self
            .column_window
            .filter(|_| !self.retain_column_window_for_gpu_pack)
        {
            let full_row_bytes = self
                .info
                .shape
                .get(1)
                .copied()
                .context("route preload tensor has no row width")?
                .checked_mul(self.bytes_per_scalar)
                .context("route preload full row byte width overflow")?;
            let compact_row_bytes = window
                .count
                .checked_mul(self.bytes_per_scalar)
                .context("route preload compact row byte width overflow")?;
            let column_start_bytes = window
                .start
                .checked_mul(self.bytes_per_scalar)
                .context("route preload column start byte offset overflow")?;
            let mut compact = Vec::with_capacity(
                self.output_row_count
                    .checked_mul(compact_row_bytes)
                    .context("route preload compact tensor byte count overflow")?,
            );
            for row in source.as_slice().chunks_exact(full_row_bytes) {
                compact.extend_from_slice(
                    &row[column_start_bytes..column_start_bytes + compact_row_bytes],
                );
            }
            RouteCudaTensorBytes::owned(compact)
        } else {
            source
        };
        let (source_row_width, source_column_start) = if self.retain_column_window_for_gpu_pack {
            (
                self.info.shape[1],
                self.column_window
                    .map(|window| window.start)
                    .unwrap_or_default(),
            )
        } else {
            (self.output_row_width, 0)
        };
        let mut info = self.info.clone();
        if self.column_window.is_some() {
            info.shape[1] = self.output_row_width;
            info.byte_length =
                (self.output_row_count * self.output_row_width * self.bytes_per_scalar) as u64;
        }
        Ok(LoadedRouteCudaTensorRows {
            info,
            source_path: self.source_path.clone(),
            start_row: self.output_start_row,
            row_count: self.output_row_count,
            row_width: self.output_row_width,
            source_row_width,
            source_column_start,
            bytes_per_scalar: self.bytes_per_scalar,
            bytes,
            elapsed_micros,
        })
    }

    fn physical_source_bytes(&self) -> u64 {
        if self.direct_file.is_none() {
            return self.source_bytes as u64;
        }
        RouteCudaAlignedReadBuffer::physical_bytes_for(
            self.file_bytes,
            self.source_offset,
            self.source_bytes,
        )
        .unwrap_or(self.source_bytes as u64)
    }
}

fn load_route_cuda_read_plans_io_uring(
    read_plans: &[RouteCudaLayerTensorReadPlan],
    slots: &[Mutex<Vec<Option<Result<LoadedRouteCudaTensorRows>>>>],
) -> Result<Option<u128>> {
    let mut buffers = (0..read_plans.len()).map(|_| None).collect::<Vec<_>>();
    let mut aligned_offsets = vec![None; read_plans.len()];
    let mut direct_indices = Vec::with_capacity(read_plans.len());
    let mut buffered_indices = Vec::new();
    for (index, plan) in read_plans.iter().enumerate() {
        let direct = plan
            .direct_file
            .as_ref()
            .map(|_| {
                RouteCudaAlignedReadBuffer::allocate(
                    plan.file_bytes,
                    plan.source_offset,
                    plan.source_bytes,
                )
            })
            .transpose()?
            .flatten();
        if let Some((buffer, aligned_offset)) = direct {
            buffers[index] = Some(buffer);
            aligned_offsets[index] = Some(aligned_offset);
            direct_indices.push(index);
        } else {
            buffered_indices.push(index);
        }
    }
    if direct_indices.is_empty() {
        return Ok(None);
    }

    let entries = direct_indices.len().next_power_of_two().max(2);
    let entries: u32 = entries
        .try_into()
        .context("route preload io_uring entry count exceeds u32")?;
    let mut ring = match IoUring::new(entries) {
        Ok(ring) => ring,
        Err(error) => {
            eprintln!("real_nvfp4_route_preload_io_uring_unavailable error={error}");
            return Ok(None);
        }
    };
    for &index in &direct_indices {
        let plan = &read_plans[index];
        let buffer = buffers[index]
            .as_mut()
            .expect("route preload io_uring buffer allocated above");
        let length: u32 = buffer
            .full_bytes()
            .try_into()
            .context("route preload io_uring read length exceeds u32")?;
        let entry = opcode::Read::new(
            types::Fd(
                plan.direct_file
                    .as_ref()
                    .expect("route preload direct file checked above")
                    .as_raw_fd(),
            ),
            buffer.full_ptr(),
            length,
        )
        .offset(
            aligned_offsets[index].expect("route preload io_uring aligned offset recorded above"),
        )
        .build()
        .user_data(index as u64);
        unsafe {
            ring.submission()
                .push(&entry)
                .map_err(|_| anyhow::anyhow!("route preload io_uring submission queue is full"))?;
        }
    }
    let started = Instant::now();
    std::thread::scope(|scope| -> Result<()> {
        for &index in &buffered_indices {
            scope.spawn(move || {
                let plan = &read_plans[index];
                let loaded = plan.load();
                slots[plan.expert_index]
                    .lock()
                    .expect("mixed route preload result slot is poisoned")[plan.slot.index()] =
                    Some(loaded);
            });
        }
        ring.submit_and_wait(direct_indices.len())
            .context("submitting route preload io_uring reads")?;
        Ok(())
    })?;
    let elapsed_micros = started.elapsed().as_micros();
    let mut completed = vec![false; read_plans.len()];
    for completion in ring.completion() {
        let index: usize = completion
            .user_data()
            .try_into()
            .context("route preload io_uring completion index exceeds usize")?;
        let buffer = buffers
            .get(index)
            .and_then(Option::as_ref)
            .with_context(|| {
                format!("route preload io_uring completion {index} is out of range")
            })?;
        let result = completion.result();
        if result < 0 {
            return Err(std::io::Error::from_raw_os_error(-result)).with_context(|| {
                format!(
                    "route preload io_uring read failed for {}",
                    read_plans[index].info.name
                )
            });
        }
        anyhow::ensure!(
            result as usize == buffer.full_bytes(),
            "route preload io_uring short read for {}: {result}/{} bytes",
            read_plans[index].info.name,
            buffer.full_bytes()
        );
        completed[index] = true;
    }
    anyhow::ensure!(
        direct_indices.iter().all(|index| completed[*index]),
        "route preload io_uring returned {}/{} completions",
        direct_indices
            .iter()
            .filter(|index| completed[**index])
            .count(),
        direct_indices.len()
    );

    for index in direct_indices {
        let plan = &read_plans[index];
        let loaded = plan.finish_load(
            RouteCudaTensorBytes::direct(
                buffers[index]
                    .take()
                    .expect("route preload io_uring buffer retained until completion"),
            ),
            elapsed_micros,
        );
        slots[plan.expert_index]
            .lock()
            .expect("route preload io_uring result slot is poisoned")[plan.slot.index()] =
            Some(loaded);
    }
    Ok(Some(elapsed_micros))
}

struct RouteCudaLayerSourceFile {
    buffered: Arc<File>,
    direct: Option<Arc<File>>,
    bytes: u64,
}

fn route_cuda_layer_tensor_read_plan(
    catalog: &TensorCatalog,
    file_cache: &mut HashMap<PathBuf, RouteCudaLayerSourceFile>,
    expert_index: usize,
    slot: RouteCudaLayerTensorSlot,
    tensor_name: &str,
    projection: &str,
    row_count: usize,
    shard: ExpertIntermediateShard,
    retain_column_window_for_gpu_pack: bool,
    load_full_projection: bool,
) -> Result<RouteCudaLayerTensorReadPlan> {
    let info = catalog_tensor(catalog, tensor_name)?.clone();
    anyhow::ensure!(
        info.shape.len() == 2,
        "route preload tensor {tensor_name} must be rank-2, shape={:?}",
        info.shape
    );
    let full_rows = info.shape[0];
    let full_row_width = info.shape[1];
    let bytes_per_scalar = dtype_byte_width(&info.dtype)?;
    let full_row_bytes = full_row_width
        .checked_mul(bytes_per_scalar)
        .context("route preload full row byte width overflow")?;
    let (
        source_offset,
        source_bytes,
        output_start_row,
        output_row_count,
        output_row_width,
        column_window,
    ) = if load_full_projection {
        (
            info.byte_offset,
            full_rows
                .checked_mul(full_row_bytes)
                .context("route preload full tensor byte length overflow")?,
            0,
            full_rows,
            full_row_width,
            None,
        )
    } else if matches!(projection, "gate_proj" | "up_proj") {
        let local_rows = shard.local_rows(full_rows)?;
        anyhow::ensure!(
            row_count == local_rows,
            "sharded projection {tensor_name} requested {row_count} rows, expected {local_rows}"
        );
        let start_row = shard.row_start(full_rows)?;
        let relative_offset = start_row
            .checked_mul(full_row_bytes)
            .context("route preload row byte offset overflow")?;
        let source_bytes = local_rows
            .checked_mul(full_row_bytes)
            .context("route preload row byte length overflow")?;
        (
            info.byte_offset
                .checked_add(relative_offset as u64)
                .context("route preload absolute byte offset overflow")?,
            source_bytes,
            start_row,
            local_rows,
            full_row_width,
            None,
        )
    } else {
        anyhow::ensure!(
                projection == "down_proj" && row_count == full_rows,
                "unsupported sharded projection window for {tensor_name}: projection={projection} rows={row_count}/{full_rows}"
            );
        anyhow::ensure!(
            full_row_width % shard.count == 0,
            "sharded down projection {tensor_name} width {full_row_width} is not divisible by {}",
            shard.count
        );
        let local_width = full_row_width / shard.count;
        (
            info.byte_offset,
            full_rows
                .checked_mul(full_row_bytes)
                .context("route preload tensor byte length overflow")?,
            0,
            full_rows,
            local_width,
            Some(RouteCudaLayerColumnWindow {
                start: local_width
                    .checked_mul(shard.rank)
                    .context("route preload shard column start overflow")?,
                count: local_width,
            }),
        )
    };
    let source_end = source_offset
        .checked_add(source_bytes as u64)
        .context("route preload source byte end overflow")?;
    let tensor_end = info
        .byte_offset
        .checked_add(info.byte_length)
        .context("route preload tensor byte end overflow")?;
    anyhow::ensure!(
        source_end <= tensor_end,
        "route preload byte window for {tensor_name} exceeds catalog tensor range"
    );
    let source_path = Path::new(&catalog.snapshot_path).join(&info.file);
    let files = if let Some(files) = file_cache.get(&source_path) {
        files
    } else {
        let buffered = Arc::new(
            File::open(&source_path)
                .with_context(|| format!("opening {}", source_path.display()))?,
        );
        let direct = if route_preload_direct_io() {
            Some(Arc::new(
                OpenOptions::new()
                    .read(true)
                    .custom_flags(libc::O_DIRECT)
                    .open(&source_path)
                    .with_context(|| format!("opening {} for direct I/O", source_path.display()))?,
            ))
        } else {
            None
        };
        let bytes = buffered
            .metadata()
            .with_context(|| format!("reading metadata for {}", source_path.display()))?
            .len();
        file_cache.insert(
            source_path.clone(),
            RouteCudaLayerSourceFile {
                buffered,
                direct,
                bytes,
            },
        );
        file_cache
            .get(&source_path)
            .expect("route preload file inserted above")
    };
    Ok(RouteCudaLayerTensorReadPlan {
        expert_index,
        slot,
        info,
        source_path,
        buffered_file: Arc::clone(&files.buffered),
        direct_file: files.direct.as_ref().map(Arc::clone),
        file_bytes: files.bytes,
        source_offset,
        source_bytes,
        output_start_row,
        output_row_count,
        output_row_width,
        bytes_per_scalar,
        column_window,
        retain_column_window_for_gpu_pack,
    })
}

struct LoadedRouteCudaLayer {
    experts: Vec<LoadedRouteCudaExpertShard>,
    source_bytes_read: u64,
    physical_source_bytes_read: u64,
    source_io_micros: Option<u128>,
}

fn cooperative_weight_preload_expert_groups(
    catalog: &TensorCatalog,
    layer_id: usize,
    expert_ids: &[usize],
    world_size: usize,
) -> Result<Vec<Vec<usize>>> {
    anyhow::ensure!(
        world_size > 1 && expert_ids.len() % world_size == 0,
        "cooperative weight preload requires evenly divisible expert groups"
    );
    let mut ordered = expert_ids
        .iter()
        .copied()
        .map(|expert_id| {
            let name = format!(
                "{}.weight",
                routed_quant_projection_base_name(layer_id, expert_id, "gate_proj")
            );
            let info = catalog_tensor(catalog, &name)?;
            Ok((info.file.clone(), info.byte_offset, expert_id))
        })
        .collect::<Result<Vec<_>>>()?;
    ordered.sort_unstable();
    let group_len = expert_ids.len() / world_size;
    Ok(ordered
        .chunks_exact(group_len)
        .map(|group| group.iter().map(|(_, _, expert_id)| *expert_id).collect())
        .collect())
}

fn native_flash_cooperative_weight_preload_expert_groups(
    catalog: &TensorCatalog,
    layer_id: usize,
    expert_ids: &[usize],
    world_size: usize,
) -> Result<Vec<Vec<usize>>> {
    anyhow::ensure!(
        world_size == 4 && expert_ids.len() % world_size == 0,
        "native Flash cooperative preload requires 256 experts split over TP=4"
    );
    let mut ordered = expert_ids
        .iter()
        .copied()
        .map(|expert_id| {
            let expert = native_fp4_expert(catalog, layer_id, expert_id)?;
            Ok((
                expert.gate.weight.file.clone(),
                expert.gate.weight.byte_offset,
                expert_id,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    ordered.sort_unstable();
    let group_len = expert_ids.len() / world_size;
    Ok(ordered
        .chunks_exact(group_len)
        .map(|group| group.iter().map(|(_, _, expert_id)| *expert_id).collect())
        .collect())
}

fn exl3_cooperative_weight_preload_expert_groups(
    catalog: &TensorCatalog,
    layer_id: usize,
    expert_ids: &[usize],
    world_size: usize,
) -> Result<Vec<Vec<usize>>> {
    anyhow::ensure!(
        world_size == 4 && expert_ids.len() % world_size == 0,
        "EXL3 cooperative preload requires evenly divided TP4 expert groups"
    );
    let mut ordered = expert_ids
        .iter()
        .copied()
        .map(|expert_id| {
            let expert = exl3_expert(catalog, layer_id, expert_id)?;
            Ok((
                expert.gate.trellis.file.clone(),
                expert.gate.trellis.byte_offset,
                expert_id,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    ordered.sort_unstable();
    let group_len = expert_ids.len() / world_size;
    Ok(ordered
        .chunks_exact(group_len)
        .map(|group| group.iter().map(|(_, _, expert_id)| *expert_id).collect())
        .collect())
}

fn compact_route_cuda_tensor_for_shard(
    source: &LoadedRouteCudaTensorRows,
    projection: &str,
    shard: ExpertIntermediateShard,
) -> Result<LoadedRouteCudaTensorRows> {
    anyhow::ensure!(
        source.start_row == 0
            && source.row_count == source.info.shape[0]
            && source.row_width == source.info.shape[1]
            && source.source_row_width == source.row_width
            && source.source_column_start == 0,
        "cooperative weight preload source {} is not a full tensor",
        source.info.name
    );
    let source_row_bytes = source
        .row_width
        .checked_mul(source.bytes_per_scalar)
        .context("cooperative source row byte width overflow")?;
    let (start_row, row_count, row_width, source_row_width, source_column_start, bytes) =
        if matches!(projection, "gate_proj" | "up_proj") {
            let row_count = shard.local_rows(source.row_count)?;
            let start_row = shard.row_start(source.row_count)?;
            let start = start_row
                .checked_mul(source_row_bytes)
                .context("cooperative source row byte offset overflow")?;
            let bytes = row_count
                .checked_mul(source_row_bytes)
                .context("cooperative source row byte length overflow")?;
            (
                start_row,
                row_count,
                source.row_width,
                source.row_width,
                0,
                source.bytes.view(start, bytes)?,
            )
        } else {
            anyhow::ensure!(
                projection == "down_proj" && source.row_width % shard.count == 0,
                "unsupported cooperative projection {projection} width {}",
                source.row_width
            );
            let row_width = source.row_width / shard.count;
            let start_column = row_width
                .checked_mul(shard.rank)
                .context("cooperative source column start overflow")?;
            (
                0,
                source.row_count,
                row_width,
                source.row_width,
                start_column,
                source.bytes.clone(),
            )
        };
    let mut info = source.info.clone();
    info.shape = vec![row_count, row_width];
    info.byte_length = row_count
        .checked_mul(row_width)
        .and_then(|values| values.checked_mul(source.bytes_per_scalar))
        .context("cooperative compact tensor logical byte length overflow")?
        as u64;
    Ok(LoadedRouteCudaTensorRows {
        info,
        source_path: source.source_path.clone(),
        start_row,
        row_count,
        row_width,
        source_row_width,
        source_column_start,
        bytes_per_scalar: source.bytes_per_scalar,
        bytes,
        elapsed_micros: source.elapsed_micros,
    })
}

fn compact_route_cuda_expert_for_shard(
    source: &LoadedRouteCudaExpertShard,
    shard: ExpertIntermediateShard,
) -> Result<LoadedRouteCudaExpertShard> {
    let projection = |source: &LoadedRouteCudaProjectionShard,
                      name|
     -> Result<LoadedRouteCudaProjectionShard> {
        Ok(LoadedRouteCudaProjectionShard {
            weight: compact_route_cuda_tensor_for_shard(&source.weight, name, shard)?,
            weight_scale: compact_route_cuda_tensor_for_shard(&source.weight_scale, name, shard)?,
        })
    };
    Ok(LoadedRouteCudaExpertShard {
        gate: projection(&source.gate, "gate_proj")?,
        up: projection(&source.up, "up_proj")?,
        down: projection(&source.down, "down_proj")?,
    })
}

fn compact_route_cuda_experts_for_shard_parallel(
    source: &[LoadedRouteCudaExpertShard],
    shard: ExpertIntermediateShard,
) -> Result<Vec<LoadedRouteCudaExpertShard>> {
    let workers = route_preload_io_workers().min(source.len());
    let next = AtomicUsize::new(0);
    let slots = (0..source.len())
        .map(|_| Mutex::new(None))
        .collect::<Vec<OptionSlot<LoadedRouteCudaExpertShard>>>();
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                let Some(expert) = source.get(index) else {
                    break;
                };
                *slots[index]
                    .lock()
                    .expect("cooperative compact result slot is poisoned") =
                    Some(compact_route_cuda_expert_for_shard(expert, shard));
            });
        }
    });
    slots
        .into_iter()
        .enumerate()
        .map(|(index, slot)| {
            slot.into_inner()
                .map_err(|_| anyhow::anyhow!("cooperative compact result slot is poisoned"))?
                .with_context(|| {
                    format!(
                        "cooperative compact worker did not fill expert index {index} for rank {}",
                        shard.rank
                    )
                })?
        })
        .collect()
}

fn load_route_cuda_layer_expert_shards_parallel(
    catalog: &TensorCatalog,
    layer_requests: &[&RouteProjectionCachePreloadRequest],
    layer_id: usize,
    expert_ids: &[usize],
    shard: ExpertIntermediateShard,
    retain_column_window_for_gpu_pack: bool,
    load_full_projection: bool,
) -> Result<LoadedRouteCudaLayer> {
    let mut requests_by_projection = HashMap::with_capacity(layer_requests.len());
    for &request in layer_requests {
        anyhow::ensure!(
            requests_by_projection
                .insert((request.expert_id, request.projection), request)
                .is_none(),
            "layer {layer_id} has duplicate expert {} {} preload requests",
            request.expert_id,
            request.projection
        );
    }
    let mut file_cache = HashMap::new();
    let mut read_plans = Vec::with_capacity(expert_ids.len() * RouteCudaLayerTensorSlot::COUNT);
    for (expert_index, &expert_id) in expert_ids.iter().enumerate() {
        for (projection, weight_slot, scale_slot) in [
            (
                "gate_proj",
                RouteCudaLayerTensorSlot::GateWeight,
                RouteCudaLayerTensorSlot::GateWeightScale,
            ),
            (
                "up_proj",
                RouteCudaLayerTensorSlot::UpWeight,
                RouteCudaLayerTensorSlot::UpWeightScale,
            ),
            (
                "down_proj",
                RouteCudaLayerTensorSlot::DownWeight,
                RouteCudaLayerTensorSlot::DownWeightScale,
            ),
        ] {
            let request = requests_by_projection
                .get(&(expert_id, projection))
                .with_context(|| {
                    format!("layer {layer_id} expert {expert_id} is missing {projection}")
                })?;
            let base_name = routed_quant_projection_base_name(layer_id, expert_id, projection);
            read_plans.push(route_cuda_layer_tensor_read_plan(
                catalog,
                &mut file_cache,
                expert_index,
                weight_slot,
                &format!("{base_name}.weight"),
                projection,
                request.row_count,
                shard,
                retain_column_window_for_gpu_pack,
                load_full_projection,
            )?);
            read_plans.push(route_cuda_layer_tensor_read_plan(
                catalog,
                &mut file_cache,
                expert_index,
                scale_slot,
                &format!("{base_name}.weight_scale"),
                projection,
                request.row_count,
                shard,
                retain_column_window_for_gpu_pack,
                load_full_projection,
            )?);
        }
    }
    read_plans.sort_unstable_by(|left, right| {
        left.source_path
            .cmp(&right.source_path)
            .then_with(|| left.source_offset.cmp(&right.source_offset))
    });
    let source_bytes_read = read_plans.iter().try_fold(0_u64, |total, plan| {
        total
            .checked_add(plan.source_bytes as u64)
            .context("route preload source byte count overflow")
    })?;
    let physical_source_bytes_read = read_plans.iter().try_fold(0_u64, |total, plan| {
        total
            .checked_add(plan.physical_source_bytes())
            .context("route preload physical source byte count overflow")
    })?;
    let slots = (0..expert_ids.len())
        .map(|_| {
            Mutex::new(
                (0..RouteCudaLayerTensorSlot::COUNT)
                    .map(|_| None)
                    .collect::<Vec<Option<Result<LoadedRouteCudaTensorRows>>>>(),
            )
        })
        .collect::<Vec<_>>();
    let source_io_micros = if load_full_projection && route_preload_direct_io() {
        load_route_cuda_read_plans_io_uring(&read_plans, &slots)?
    } else {
        None
    };
    if source_io_micros.is_some() {
        // Results were filled above from one queued direct-I/O batch, with any
        // unaligned EOF tail reads running concurrently through buffered I/O.
    } else {
        let workers = route_preload_io_workers().min(read_plans.len());
        let next = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..workers {
                scope.spawn(|| loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    let Some(plan) = read_plans.get(index) else {
                        break;
                    };
                    let loaded = plan.load();
                    slots[plan.expert_index]
                        .lock()
                        .expect("parallel route preload result slot is poisoned")
                        [plan.slot.index()] = Some(loaded);
                });
            }
        });
    }
    let experts = slots
        .into_iter()
        .enumerate()
        .map(|(expert_index, slot)| {
            let mut tensors = slot
                .into_inner()
                .map_err(|_| anyhow::anyhow!("parallel route preload result slot is poisoned"))?;
            let mut take =
                |slot: RouteCudaLayerTensorSlot| -> Result<LoadedRouteCudaTensorRows> {
                tensors[slot.index()]
                    .take()
                    .with_context(|| {
                        format!(
                            "parallel route preload did not fill layer {layer_id} expert index {expert_index} {}",
                            slot.label()
                        )
                    })?
                    .with_context(|| {
                        format!(
                            "loading layer {layer_id} expert index {expert_index} {}",
                            slot.label()
                        )
                    })
            };
            Ok(LoadedRouteCudaExpertShard {
                gate: LoadedRouteCudaProjectionShard {
                    weight: take(RouteCudaLayerTensorSlot::GateWeight)?,
                    weight_scale: take(RouteCudaLayerTensorSlot::GateWeightScale)?,
                },
                up: LoadedRouteCudaProjectionShard {
                    weight: take(RouteCudaLayerTensorSlot::UpWeight)?,
                    weight_scale: take(RouteCudaLayerTensorSlot::UpWeightScale)?,
                },
                down: LoadedRouteCudaProjectionShard {
                    weight: take(RouteCudaLayerTensorSlot::DownWeight)?,
                    weight_scale: take(RouteCudaLayerTensorSlot::DownWeightScale)?,
                },
            })
        })
        .collect::<Result<Vec<_>>>()?;
    anyhow::ensure!(
        experts.len() == expert_ids.len(),
        "parallel route preload returned {} experts for layer {layer_id}, expected {}",
        experts.len(),
        expert_ids.len()
    );
    Ok(LoadedRouteCudaLayer {
        experts,
        source_bytes_read,
        physical_source_bytes_read,
        source_io_micros,
    })
}

fn load_native_flash_layer_expert_shards_parallel(
    catalog: &TensorCatalog,
    layer_id: usize,
    expert_ids: &[usize],
    shard: ExpertIntermediateShard,
    load_full_projection: bool,
) -> Result<LoadedRouteCudaLayer> {
    let mut file_cache = HashMap::new();
    let mut read_plans = Vec::with_capacity(expert_ids.len() * RouteCudaLayerTensorSlot::COUNT);
    for (expert_index, &expert_id) in expert_ids.iter().enumerate() {
        let expert = native_fp4_expert(catalog, layer_id, expert_id)
            .with_context(|| format!("loading native Flash layer {layer_id} expert {expert_id}"))?;
        let tp = expert.tp_shard(shard.rank)?;
        for (projection, native, tp_projection, weight_slot, scale_slot) in [
            (
                "gate_proj",
                expert.gate,
                tp.gate,
                RouteCudaLayerTensorSlot::GateWeight,
                RouteCudaLayerTensorSlot::GateWeightScale,
            ),
            (
                "up_proj",
                expert.up,
                tp.up,
                RouteCudaLayerTensorSlot::UpWeight,
                RouteCudaLayerTensorSlot::UpWeightScale,
            ),
            (
                "down_proj",
                expert.down,
                tp.down,
                RouteCudaLayerTensorSlot::DownWeight,
                RouteCudaLayerTensorSlot::DownWeightScale,
            ),
        ] {
            read_plans.push(route_cuda_layer_tensor_read_plan(
                catalog,
                &mut file_cache,
                expert_index,
                weight_slot,
                &native.weight.name,
                projection,
                tp_projection.weight.row_count,
                shard,
                true,
                load_full_projection,
            )?);
            read_plans.push(route_cuda_layer_tensor_read_plan(
                catalog,
                &mut file_cache,
                expert_index,
                scale_slot,
                &native.scale.name,
                projection,
                tp_projection.scale.row_count,
                shard,
                true,
                load_full_projection,
            )?);
        }
    }
    read_plans.sort_unstable_by(|left, right| {
        left.source_path
            .cmp(&right.source_path)
            .then_with(|| left.source_offset.cmp(&right.source_offset))
    });
    let source_bytes_read = read_plans.iter().try_fold(0_u64, |total, plan| {
        total
            .checked_add(plan.source_bytes as u64)
            .context("native Flash preload source byte count overflow")
    })?;
    let physical_source_bytes_read = read_plans.iter().try_fold(0_u64, |total, plan| {
        total
            .checked_add(plan.physical_source_bytes())
            .context("native Flash preload physical source byte count overflow")
    })?;
    let slots = (0..expert_ids.len())
        .map(|_| {
            Mutex::new(
                (0..RouteCudaLayerTensorSlot::COUNT)
                    .map(|_| None)
                    .collect::<Vec<Option<Result<LoadedRouteCudaTensorRows>>>>(),
            )
        })
        .collect::<Vec<_>>();
    let source_io_micros = if load_full_projection && route_preload_direct_io() {
        load_route_cuda_read_plans_io_uring(&read_plans, &slots)?
    } else {
        None
    };
    if source_io_micros.is_none() {
        let workers = route_preload_io_workers().min(read_plans.len());
        let next = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..workers {
                scope.spawn(|| loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    let Some(plan) = read_plans.get(index) else {
                        break;
                    };
                    let loaded = plan.load();
                    slots[plan.expert_index]
                        .lock()
                        .expect("native Flash preload result slot is poisoned")
                        [plan.slot.index()] = Some(loaded);
                });
            }
        });
    }
    let experts = slots
        .into_iter()
        .enumerate()
        .map(|(expert_index, slot)| {
            let mut tensors = slot
                .into_inner()
                .map_err(|_| anyhow::anyhow!("native Flash preload result slot is poisoned"))?;
            let mut take =
                |slot: RouteCudaLayerTensorSlot| -> Result<LoadedRouteCudaTensorRows> {
                    tensors[slot.index()]
                        .take()
                        .with_context(|| {
                            format!(
                                "native Flash preload did not fill layer {layer_id} expert index {expert_index} {}",
                                slot.label()
                            )
                        })?
                        .with_context(|| {
                            format!(
                                "loading native Flash layer {layer_id} expert index {expert_index} {}",
                                slot.label()
                            )
                        })
                };
            Ok(LoadedRouteCudaExpertShard {
                gate: LoadedRouteCudaProjectionShard {
                    weight: take(RouteCudaLayerTensorSlot::GateWeight)?,
                    weight_scale: take(RouteCudaLayerTensorSlot::GateWeightScale)?,
                },
                up: LoadedRouteCudaProjectionShard {
                    weight: take(RouteCudaLayerTensorSlot::UpWeight)?,
                    weight_scale: take(RouteCudaLayerTensorSlot::UpWeightScale)?,
                },
                down: LoadedRouteCudaProjectionShard {
                    weight: take(RouteCudaLayerTensorSlot::DownWeight)?,
                    weight_scale: take(RouteCudaLayerTensorSlot::DownWeightScale)?,
                },
            })
        })
        .collect::<Result<Vec<_>>>()?;
    anyhow::ensure!(
        experts.len() == expert_ids.len(),
        "native Flash preload returned {} experts for layer {layer_id}, expected {}",
        experts.len(),
        expert_ids.len()
    );
    Ok(LoadedRouteCudaLayer {
        experts,
        source_bytes_read,
        physical_source_bytes_read,
        source_io_micros,
    })
}

type OptionSlot<T> = Mutex<Option<Result<T>>>;

struct RouteCudaQuantLayerPreloadPlan {
    layer_id: usize,
    requests: Vec<RouteProjectionCachePreloadRequest>,
    expert_ids: Vec<usize>,
}

struct RouteCudaCooperativeLayerPreloadPlan {
    expert_groups: Vec<Vec<usize>>,
}

#[derive(Default)]
struct RouteCudaCooperativePreloadWorkspace {
    exchange_slab: Option<Arc<RouteCudaLayerExpertSlab>>,
    receive: Option<OwnedDeviceAllocation>,
}

#[derive(Default)]
struct RouteCudaDs4FlashCooperativePreloadWorkspace {
    exchange_slab: Option<Arc<RouteCudaDs4FlashLayerExpertSlab>>,
    receive: Option<OwnedDeviceAllocation>,
}

#[derive(Default)]
struct RouteCudaDs4FlashExl3CooperativePreloadWorkspace {
    exchange_slab: Option<Arc<RouteCudaDs4FlashExl3LayerExpertSlab>>,
    receive: Option<OwnedDeviceAllocation>,
}

fn load_route_cuda_layer_plan(
    catalog: &TensorCatalog,
    plan: &RouteCudaQuantLayerPreloadPlan,
    shard: ExpertIntermediateShard,
    retain_column_window_for_gpu_pack: bool,
) -> Result<LoadedRouteCudaLayer> {
    let request_refs = plan.requests.iter().collect::<Vec<_>>();
    load_route_cuda_layer_expert_shards_parallel(
        catalog,
        &request_refs,
        plan.layer_id,
        &plan.expert_ids,
        shard,
        retain_column_window_for_gpu_pack,
        false,
    )
}

fn load_route_cuda_cooperative_layer_plan(
    catalog: &TensorCatalog,
    plan: &RouteCudaQuantLayerPreloadPlan,
    cooperative: &RouteCudaCooperativeLayerPreloadPlan,
    shard: ExpertIntermediateShard,
) -> Result<LoadedRouteCudaLayer> {
    let request_refs = plan.requests.iter().collect::<Vec<_>>();
    let source_expert_ids = cooperative.expert_groups.get(shard.rank).with_context(|| {
        format!(
            "cooperative layer {} has no expert group for rank {}",
            plan.layer_id, shard.rank
        )
    })?;
    load_route_cuda_layer_expert_shards_parallel(
        catalog,
        &request_refs,
        plan.layer_id,
        source_expert_ids,
        shard,
        false,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
fn store_loaded_route_cuda_layer_cooperative(
    plan: &RouteCudaQuantLayerPreloadPlan,
    cooperative: &RouteCudaCooperativeLayerPreloadPlan,
    loaded_source_experts: &[LoadedRouteCudaExpertShard],
    cuda_cache: &mut RouteCudaCache,
    scalar_metadata: &HashMap<RoutedQuantScalarMetadataKey, RoutedQuantScalarMetadata>,
    shard: ExpertIntermediateShard,
    cuda_stream: *mut c_void,
    workspace: &mut RouteCudaCooperativePreloadWorkspace,
    preload: &mut RouteCudaProjectionPreload,
    load_ms: f64,
    source_io_micros: Option<u128>,
    source_bytes_read: u64,
    physical_source_bytes_read: u64,
) -> Result<()> {
    let world_size = shard.count;
    anyhow::ensure!(
        world_size == 4
            && cooperative.expert_groups.len() == world_size
            && cooperative
                .expert_groups
                .iter()
                .all(|group| group.len() == loaded_source_experts.len()),
        "cooperative layer {} expert groups do not match rank {}/{} source expert count {}",
        plan.layer_id,
        shard.rank,
        world_size,
        loaded_source_experts.len()
    );
    let source_expert_count = loaded_source_experts.len();
    let mut allocation_ms = 0.0;
    let mut final_slab = None;
    let mut compact_ms = 0.0;
    let mut pack_ms = 0.0;
    for target_rank in 0..world_size {
        let target_shard = ExpertIntermediateShard::new(world_size, target_rank)?;
        let compact_started = Instant::now();
        let compact =
            compact_route_cuda_experts_for_shard_parallel(loaded_source_experts, target_shard)?;
        compact_ms += elapsed_ms(compact_started);
        let exemplar = compact
            .first()
            .context("cooperative weight preload produced no compact experts")?;
        if workspace.exchange_slab.is_none() {
            let allocation_started = Instant::now();
            workspace.exchange_slab = Some(Arc::new(RouteCudaLayerExpertSlab::new(
                Arc::clone(&cuda_cache.library),
                plan.layer_id,
                plan.expert_ids.len(),
                exemplar,
                false,
                false,
            )?));
            allocation_ms += elapsed_ms(allocation_started);
        }
        if final_slab.is_none() {
            let allocation_started = Instant::now();
            final_slab = Some(Arc::new(RouteCudaLayerExpertSlab::new(
                Arc::clone(&cuda_cache.library),
                plan.layer_id,
                plan.expert_ids.len(),
                exemplar,
                false,
                false,
            )?));
            allocation_ms += elapsed_ms(allocation_started);
        }
        let target_start = target_rank
            .checked_mul(source_expert_count)
            .context("cooperative exchange expert offset overflow")?;
        let exchange_expert_ids =
            (target_start..target_start + source_expert_count).collect::<Vec<_>>();
        let pack_started = Instant::now();
        workspace
            .exchange_slab
            .as_ref()
            .expect("cooperative exchange slab initialized above")
            .store_layer_experts_w4a16(
                &exchange_expert_ids,
                &compact,
                Arc::clone(&cuda_cache.library),
                &mut cuda_cache.workspace,
                cuda_stream,
            )?;
        pack_ms += elapsed_ms(pack_started);
    }
    let exchange_slab = workspace
        .exchange_slab
        .as_ref()
        .context("cooperative exchange slab was not initialized")?;
    let final_slab = final_slab.context("cooperative final slab was not initialized")?;
    let communicator = cuda_cache
        .weight_preload_communicator
        .as_ref()
        .context("cooperative weight preload has no NCCL communicator")?;
    anyhow::ensure!(
        communicator.world_size() == world_size && communicator.rank() == shard.rank,
        "cooperative weight preload communicator rank {}/{} differs from shard {}/{}",
        communicator.rank(),
        communicator.world_size(),
        shard.rank,
        world_size
    );
    struct ExchangeComponent {
        label: &'static str,
        send: Ds41rtDeviceBuffer,
        destination: Ds41rtDeviceBuffer,
        expert_stride_bytes: usize,
    }
    let components = [
        ExchangeComponent {
            label: "W13 weight",
            send: exchange_slab.w13_weight.buffer(),
            destination: final_slab.w13_weight.buffer(),
            expert_stride_bytes: exchange_slab.w13_weight_expert_stride_bytes,
        },
        ExchangeComponent {
            label: "W13 scale",
            send: exchange_slab.w13_scale.buffer(),
            destination: final_slab.w13_scale.buffer(),
            expert_stride_bytes: exchange_slab.w13_scale_expert_stride_bytes,
        },
        ExchangeComponent {
            label: "W2 weight",
            send: exchange_slab.w2_weight.buffer(),
            destination: final_slab.w2_weight.buffer(),
            expert_stride_bytes: exchange_slab.w2_weight_expert_stride_bytes,
        },
        ExchangeComponent {
            label: "W2 scale",
            send: exchange_slab.w2_scale.buffer(),
            destination: final_slab.w2_scale.buffer(),
            expert_stride_bytes: exchange_slab.w2_scale_expert_stride_bytes,
        },
    ];
    let maximum_row_stride = components
        .iter()
        .map(|component| {
            component
                .expert_stride_bytes
                .checked_mul(source_expert_count)
                .context("cooperative exchange row stride overflow")
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .max()
        .context("cooperative exchange has no components")?;
    let receive_bytes = maximum_row_stride
        .checked_mul(world_size - 1)
        .context("cooperative exchange receive byte count overflow")?;
    let needs_receive_allocation = workspace
        .receive
        .as_ref()
        .map(|receive| receive.capacity_bytes() < receive_bytes)
        .unwrap_or(true);
    if needs_receive_allocation {
        let allocation_started = Instant::now();
        workspace.receive = Some(OwnedDeviceAllocation::new_with_kind(
            Arc::clone(&cuda_cache.library),
            receive_bytes,
            "cooperative weight exchange receive workspace",
            false,
        )?);
        allocation_ms += elapsed_ms(allocation_started);
    }
    let receive = workspace
        .receive
        .as_ref()
        .context("cooperative weight exchange receive workspace was not initialized")?;
    let exchange_started = Instant::now();
    for component in components {
        let row_stride = component
            .expert_stride_bytes
            .checked_mul(source_expert_count)
            .context("cooperative component row stride overflow")?;
        let component_receive_bytes = row_stride
            .checked_mul(world_size - 1)
            .context("cooperative component receive byte count overflow")?;
        let receive_view = route_device_buffer_slice(receive.buffer(), 0, component_receive_bytes)?;
        unsafe {
            communicator
                .row_all_to_all_u8_async(
                    component.send,
                    receive_view,
                    world_size,
                    row_stride,
                    cuda_stream,
                )
                .with_context(|| {
                    format!(
                        "exchanging layer {} cooperative {}",
                        plan.layer_id, component.label
                    )
                })?;
        }
        for source_rank in 0..world_size {
            let source_group = &cooperative.expert_groups[source_rank];
            let segment = if source_rank == shard.rank {
                route_device_buffer_slice(component.send, shard.rank * row_stride, row_stride)?
            } else {
                let receive_index = if source_rank < shard.rank {
                    source_rank
                } else {
                    source_rank - 1
                };
                route_device_buffer_slice(receive_view, receive_index * row_stride, row_stride)?
            };
            for (source_index, &expert_id) in source_group.iter().enumerate() {
                let source = route_device_buffer_slice(
                    segment,
                    source_index * component.expert_stride_bytes,
                    component.expert_stride_bytes,
                )?;
                let destination = route_device_buffer_slice(
                    component.destination,
                    expert_id * component.expert_stride_bytes,
                    component.expert_stride_bytes,
                )?;
                unsafe {
                    cuda_cache
                        .library
                        .copy_d2d_async(
                            destination,
                            source,
                            component.expert_stride_bytes,
                            cuda_stream,
                        )
                        .with_context(|| {
                            format!(
                                "scattering layer {} cooperative {} expert {expert_id}",
                                plan.layer_id, component.label
                            )
                        })?;
                }
            }
        }
    }
    unsafe {
        cuda_cache
            .library
            .cuda_stream_synchronize(cuda_stream)
            .context("synchronizing cooperative weight preload exchange")?;
    }
    let exchange_ms = elapsed_ms(exchange_started);
    for expert_id in &plan.expert_ids {
        final_slab.store_expert_scalars_from_cache(*expert_id, scalar_metadata)?;
    }
    final_slab.finalize_w4a16_global_scales(cuda_cache.library.as_ref(), cuda_stream)?;
    anyhow::ensure!(
        cuda_cache
            .expert_slabs
            .insert(plan.layer_id, Arc::clone(&final_slab))
            .is_none(),
        "cooperative preload inserted duplicate layer {}",
        plan.layer_id
    );
    let projection_groups = plan.expert_ids.len() * 3;
    preload.projection_groups += projection_groups;
    preload.weight_bytes +=
        (final_slab.w13_weight.capacity_bytes() + final_slab.w2_weight.capacity_bytes()) as u64;
    preload.weight_scale_bytes +=
        (final_slab.w13_scale.capacity_bytes() + final_slab.w2_scale.capacity_bytes()) as u64;
    cuda_cache.projection_uploads += projection_groups;
    let source_gbps = physical_source_bytes_read as f64 / (load_ms * 1.0e6).max(1.0);
    let source_io_ms = source_io_micros
        .map(|micros| micros as f64 / 1_000.0)
        .unwrap_or_default();
    let source_io_gbps = source_io_micros
        .map(|micros| physical_source_bytes_read as f64 / (micros as f64 * 1.0e3).max(1.0))
        .unwrap_or_default();
    eprintln!(
        "real_nvfp4_cuda_layer_preload layer_id={} experts={} source_experts={} io_workers={} direct_io={} cooperative=true source_bytes_requested={source_bytes_read} source_bytes_read={physical_source_bytes_read} source_gbps={source_gbps:.3} source_io_gbps={source_io_gbps:.3} load_ms={load_ms:.3} source_io_ms={source_io_ms:.3} allocation_ms={allocation_ms:.3} compact_ms={compact_ms:.3} pack_ms={pack_ms:.3} exchange_ms={exchange_ms:.3}",
        plan.layer_id,
        plan.expert_ids.len(),
        loaded_source_experts.len(),
        route_preload_io_workers(),
        route_preload_direct_io(),
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn store_loaded_route_cuda_layer(
    plan: &RouteCudaQuantLayerPreloadPlan,
    loaded_experts: &[LoadedRouteCudaExpertShard],
    cuda_cache: &mut RouteCudaCache,
    scalar_metadata: &HashMap<RoutedQuantScalarMetadataKey, RoutedQuantScalarMetadata>,
    managed_weights: bool,
    cuda_stream: *mut c_void,
    request_count: usize,
    progress_interval: usize,
    preload: &mut RouteCudaProjectionPreload,
    load_ms: f64,
    source_io_micros: Option<u128>,
    source_bytes_read: u64,
    physical_source_bytes_read: u64,
) -> Result<()> {
    let first = loaded_experts
        .first()
        .context("parallel route preload returned no experts")?;
    let allocation_started = Instant::now();
    let slab = Arc::new(RouteCudaLayerExpertSlab::new(
        Arc::clone(&cuda_cache.library),
        plan.layer_id,
        plan.expert_ids.len(),
        first,
        managed_weights,
        false,
    )?);
    let allocation_ms = elapsed_ms(allocation_started);
    let pack_started = Instant::now();
    slab.store_layer_experts_w4a16(
        &plan.expert_ids,
        loaded_experts,
        Arc::clone(&cuda_cache.library),
        &mut cuda_cache.workspace,
        cuda_stream,
    )?;
    for (expert_id, loaded) in plan.expert_ids.iter().copied().zip(loaded_experts) {
        slab.store_expert_scalars_from_cache(expert_id, scalar_metadata)?;
        preload.projection_groups += 3;
        preload.weight_bytes += (loaded.gate.weight.bytes.len()
            + loaded.up.weight.bytes.len()
            + loaded.down.weight.bytes.len()) as u64;
        preload.weight_scale_bytes += (loaded.gate.weight_scale.bytes.len()
            + loaded.up.weight_scale.bytes.len()
            + loaded.down.weight_scale.bytes.len()) as u64;
        cuda_cache.projection_uploads += 3;
        if progress_interval > 0
            && (preload.projection_groups % progress_interval == 0
                || preload.projection_groups == request_count)
        {
            eprintln!(
                "real_nvfp4_cuda_projection_preload_progress groups={}/{} managed_projection_allocations={managed_weights} contiguous_tp4=true packed_w4a16=true",
                preload.projection_groups,
                request_count
            );
        }
    }
    let pack_ms = elapsed_ms(pack_started);
    let source_gbps = physical_source_bytes_read as f64 / (load_ms * 1.0e6).max(1.0);
    let source_io_ms = source_io_micros
        .map(|micros| micros as f64 / 1_000.0)
        .unwrap_or_default();
    let source_io_gbps = source_io_micros
        .map(|micros| physical_source_bytes_read as f64 / (micros as f64 * 1.0e3).max(1.0))
        .unwrap_or_default();
    eprintln!(
        "real_nvfp4_cuda_layer_preload layer_id={} experts={} io_workers={} direct_io={} source_bytes_requested={source_bytes_read} source_bytes_read={physical_source_bytes_read} source_gbps={source_gbps:.3} source_io_gbps={source_io_gbps:.3} load_ms={load_ms:.3} source_io_ms={source_io_ms:.3} allocation_ms={allocation_ms:.3} pack_and_scalar_ms={pack_ms:.3}",
        plan.layer_id,
        plan.expert_ids.len(),
        route_preload_io_workers(),
        route_preload_direct_io()
    );
    slab.finalize_w4a16_global_scales(cuda_cache.library.as_ref(), cuda_stream)?;
    anyhow::ensure!(
        cuda_cache
            .expert_slabs
            .insert(plan.layer_id, slab)
            .is_none(),
        "contiguous TP4 preload inserted duplicate layer {}",
        plan.layer_id
    );
    Ok(())
}

fn preload_routed_quant_projection_cuda_cache_cooperative_tp4(
    catalog: &TensorCatalog,
    layer_plans: &[RouteCudaQuantLayerPreloadPlan],
    cuda_cache: &mut RouteCudaCache,
    scalar_metadata: &HashMap<RoutedQuantScalarMetadataKey, RoutedQuantScalarMetadata>,
    shard: ExpertIntermediateShard,
    cuda_stream: *mut c_void,
) -> Result<RouteCudaProjectionPreload> {
    let preload_started = Instant::now();
    let world_size = cuda_cache
        .weight_preload_communicator
        .as_ref()
        .context("cooperative preload requires a communicator")?
        .world_size();
    anyhow::ensure!(
        world_size == shard.count,
        "cooperative communicator world size {world_size} differs from shard count {}",
        shard.count
    );
    let cooperative_plans = layer_plans
        .iter()
        .map(|plan| {
            Ok(RouteCudaCooperativeLayerPreloadPlan {
                expert_groups: cooperative_weight_preload_expert_groups(
                    catalog,
                    plan.layer_id,
                    &plan.expert_ids,
                    world_size,
                )?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let mut preload = RouteCudaProjectionPreload::default();
    let mut source_bytes_read = 0_u64;
    let mut workspace = RouteCudaCooperativePreloadWorkspace::default();
    std::thread::scope(|scope| -> Result<()> {
        let first_plan = layer_plans
            .first()
            .context("cooperative TP4 preload has no layer plans")?;
        let first_cooperative = cooperative_plans
            .first()
            .context("cooperative TP4 preload has no expert groups")?;
        let mut pending = Some(scope.spawn(move || {
            let started = Instant::now();
            let loaded = load_route_cuda_cooperative_layer_plan(
                catalog,
                first_plan,
                first_cooperative,
                shard,
            )?;
            Ok::<_, anyhow::Error>((loaded, elapsed_ms(started)))
        }));
        for index in 0..layer_plans.len() {
            let plan = &layer_plans[index];
            let cooperative = &cooperative_plans[index];
            let (loaded_layer, load_ms) = pending
                .take()
                .expect("cooperative current layer preload handle exists")
                .join()
                .map_err(|_| {
                    anyhow::anyhow!("cooperative route layer preload worker panicked")
                })??;
            pending = layer_plans.get(index + 1).map(|next_plan| {
                let next_cooperative = &cooperative_plans[index + 1];
                scope.spawn(move || {
                    let started = Instant::now();
                    let loaded = load_route_cuda_cooperative_layer_plan(
                        catalog,
                        next_plan,
                        next_cooperative,
                        shard,
                    )?;
                    Ok::<_, anyhow::Error>((loaded, elapsed_ms(started)))
                })
            });
            source_bytes_read = source_bytes_read
                .checked_add(loaded_layer.physical_source_bytes_read)
                .context("cooperative preload source byte total overflow")?;
            store_loaded_route_cuda_layer_cooperative(
                plan,
                cooperative,
                &loaded_layer.experts,
                cuda_cache,
                scalar_metadata,
                shard,
                cuda_stream,
                &mut workspace,
                &mut preload,
                load_ms,
                loaded_layer.source_io_micros,
                loaded_layer.source_bytes_read,
                loaded_layer.physical_source_bytes_read,
            )?;
        }
        Ok(())
    })?;
    let total_ms = preload_started.elapsed().as_secs_f64() * 1_000.0;
    let effective_source_gbps = source_bytes_read as f64 / (total_ms * 1.0e6).max(1.0);
    eprintln!(
        "real_nvfp4_cooperative_preload_complete layers={} read_ahead_layers=1 source_bytes={} total_ms={total_ms:.3} effective_source_gbps={effective_source_gbps:.3}",
        layer_plans.len(),
        source_bytes_read,
    );
    let expected = layer_plans
        .iter()
        .map(|plan| plan.expert_ids.len() * 3)
        .sum::<usize>();
    anyhow::ensure!(
        preload.projection_groups == expected,
        "cooperative preload loaded {} projections, expected {expected}",
        preload.projection_groups
    );
    let cleanup_started = Instant::now();
    let read_pool_started = Instant::now();
    clear_route_cuda_aligned_read_pool();
    eprintln!(
        "real_nvfp4_cooperative_preload_cleanup stage=aligned-read-pool elapsed_ms={:.3} total_ms={:.3}",
        read_pool_started.elapsed().as_secs_f64() * 1_000.0,
        cleanup_started.elapsed().as_secs_f64() * 1_000.0,
    );
    let communicator_started = Instant::now();
    drop(cuda_cache.weight_preload_communicator.take());
    eprintln!(
        "real_nvfp4_cooperative_preload_cleanup stage=communicator-drop elapsed_ms={:.3} total_ms={:.3}",
        communicator_started.elapsed().as_secs_f64() * 1_000.0,
        cleanup_started.elapsed().as_secs_f64() * 1_000.0,
    );
    Ok(preload)
}

fn preload_routed_quant_projection_cuda_cache_contiguous_tp4(
    catalog: &TensorCatalog,
    requests: &[RouteProjectionCachePreloadRequest],
    cuda_cache: &mut RouteCudaCache,
    scalar_metadata: &HashMap<RoutedQuantScalarMetadataKey, RoutedQuantScalarMetadata>,
    shard: ExpertIntermediateShard,
    cuda_stream: *mut c_void,
) -> Result<RouteCudaProjectionPreload> {
    anyhow::ensure!(
        shard.count == 4 && cuda_cache.b12x_w4a16_packed,
        "contiguous TP4 expert slabs require four intermediate shards and packed W4A16"
    );
    let managed_weights = !b12x_spark_w4a16_device_weights_enabled();

    let mut layers = requests
        .iter()
        .map(|request| request.layer_id)
        .collect::<Vec<_>>();
    layers.sort_unstable();
    layers.dedup();
    let mut layer_plans = Vec::with_capacity(layers.len());
    for layer_id in layers {
        let layer_requests = requests
            .iter()
            .filter(|request| request.layer_id == layer_id)
            .copied()
            .collect::<Vec<_>>();
        let mut expert_ids = layer_requests
            .iter()
            .map(|request| request.expert_id)
            .collect::<Vec<_>>();
        expert_ids.sort_unstable();
        expert_ids.dedup();
        anyhow::ensure!(
            !expert_ids.is_empty() && expert_ids.iter().copied().eq(0..expert_ids.len()),
            "layer {layer_id} contiguous TP4 expert IDs must be dense from zero"
        );
        anyhow::ensure!(
            layer_requests.len() == expert_ids.len() * 3,
            "layer {layer_id} has {} projection requests for {} experts, expected {}",
            layer_requests.len(),
            expert_ids.len(),
            expert_ids.len() * 3
        );
        layer_plans.push(RouteCudaQuantLayerPreloadPlan {
            layer_id,
            requests: layer_requests,
            expert_ids,
        });
    }
    if cuda_cache.weight_preload_communicator.is_some() {
        return preload_routed_quant_projection_cuda_cache_cooperative_tp4(
            catalog,
            &layer_plans,
            cuda_cache,
            scalar_metadata,
            shard,
            cuda_stream,
        );
    }
    let progress_interval = cuda_projection_preload_progress_interval();
    let mut preload = RouteCudaProjectionPreload::default();
    std::thread::scope(|scope| -> Result<()> {
        let first_plan = layer_plans
            .first()
            .context("contiguous TP4 preload has no layer plans")?;
        let mut pending = Some(scope.spawn(move || {
            let started = Instant::now();
            let loaded = load_route_cuda_layer_plan(catalog, first_plan, shard, true)?;
            Ok::<_, anyhow::Error>((loaded, elapsed_ms(started)))
        }));
        for (index, plan) in layer_plans.iter().enumerate() {
            let (loaded_layer, load_ms) = pending
                .take()
                .expect("current layer preload handle exists")
                .join()
                .map_err(|_| anyhow::anyhow!("parallel route layer preload worker panicked"))??;
            pending = layer_plans.get(index + 1).map(|next_plan| {
                scope.spawn(move || {
                    let started = Instant::now();
                    let loaded = load_route_cuda_layer_plan(catalog, next_plan, shard, true)?;
                    Ok::<_, anyhow::Error>((loaded, elapsed_ms(started)))
                })
            });
            store_loaded_route_cuda_layer(
                plan,
                &loaded_layer.experts,
                cuda_cache,
                scalar_metadata,
                managed_weights,
                cuda_stream,
                requests.len(),
                progress_interval,
                &mut preload,
                load_ms,
                loaded_layer.source_io_micros,
                loaded_layer.source_bytes_read,
                loaded_layer.physical_source_bytes_read,
            )?;
        }
        Ok(())
    })?;
    anyhow::ensure!(
        preload.projection_groups == requests.len(),
        "contiguous TP4 preload loaded {} projections, expected {}",
        preload.projection_groups,
        requests.len()
    );
    clear_route_cuda_aligned_read_pool();
    Ok(preload)
}

#[allow(clippy::too_many_arguments)]
fn store_native_flash_layer_cooperative(
    catalog: &TensorCatalog,
    layer_id: usize,
    expert_ids: &[usize],
    expert_groups: &[Vec<usize>],
    loaded_source_experts: &[LoadedRouteCudaExpertShard],
    cuda_cache: &mut RouteCudaCache,
    shard: ExpertIntermediateShard,
    cuda_stream: *mut c_void,
    workspace: &mut RouteCudaDs4FlashCooperativePreloadWorkspace,
    preload: &mut RouteCudaProjectionPreload,
    load_ms: f64,
    source_io_micros: Option<u128>,
    source_bytes_read: u64,
    physical_source_bytes_read: u64,
) -> Result<()> {
    let world_size = shard.count;
    anyhow::ensure!(
        world_size == 4
            && expert_groups.len() == world_size
            && expert_groups
                .iter()
                .all(|group| group.len() == loaded_source_experts.len()),
        "native Flash cooperative layer {layer_id} groups do not match rank {}/{} source expert count {}",
        shard.rank,
        world_size,
        loaded_source_experts.len()
    );
    let source_expert_count = loaded_source_experts.len();
    let mut allocation_ms = 0.0;
    let mut compact_ms = 0.0;
    let mut pack_ms = 0.0;
    let final_allocation_started = Instant::now();
    let final_slab = Arc::new(RouteCudaDs4FlashLayerExpertSlab::new(
        Arc::clone(&cuda_cache.library),
        catalog,
        layer_id,
    )?);
    allocation_ms += elapsed_ms(final_allocation_started);

    for target_rank in 0..world_size {
        let target_shard = ExpertIntermediateShard::new(world_size, target_rank)?;
        let compact_started = Instant::now();
        let compact =
            compact_route_cuda_experts_for_shard_parallel(loaded_source_experts, target_shard)?;
        compact_ms += elapsed_ms(compact_started);
        if workspace.exchange_slab.is_none() {
            let allocation_started = Instant::now();
            workspace.exchange_slab = Some(Arc::new(RouteCudaDs4FlashLayerExpertSlab::new(
                Arc::clone(&cuda_cache.library),
                catalog,
                layer_id,
            )?));
            allocation_ms += elapsed_ms(allocation_started);
        }
        let target_start = target_rank
            .checked_mul(source_expert_count)
            .context("native Flash cooperative exchange expert offset overflow")?;
        let exchange_expert_ids =
            (target_start..target_start + source_expert_count).collect::<Vec<_>>();
        let pack_started = Instant::now();
        for (id_chunk, expert_chunk) in exchange_expert_ids.chunks(32).zip(compact.chunks(32)) {
            workspace
                .exchange_slab
                .as_ref()
                .expect("native Flash cooperative exchange slab initialized above")
                .store_layer_experts(
                    id_chunk,
                    expert_chunk,
                    Arc::clone(&cuda_cache.library),
                    &mut cuda_cache.workspace,
                    cuda_stream,
                )?;
        }
        pack_ms += elapsed_ms(pack_started);
    }

    let exchange_slab = workspace
        .exchange_slab
        .as_ref()
        .context("native Flash cooperative exchange slab was not initialized")?;
    let communicator = cuda_cache
        .weight_preload_communicator
        .as_ref()
        .context("native Flash cooperative preload has no NCCL communicator")?;
    anyhow::ensure!(
        communicator.world_size() == world_size && communicator.rank() == shard.rank,
        "native Flash cooperative communicator rank {}/{} differs from shard {}/{}",
        communicator.rank(),
        communicator.world_size(),
        shard.rank,
        world_size
    );
    struct ExchangeComponent {
        label: &'static str,
        send: Ds41rtDeviceBuffer,
        destination: Ds41rtDeviceBuffer,
        expert_stride_bytes: usize,
    }
    let components = [
        ExchangeComponent {
            label: "W13 weight",
            send: exchange_slab.w13_weight.buffer(),
            destination: final_slab.w13_weight.buffer(),
            expert_stride_bytes: DS4_FLASH_W13_WEIGHT_EXPERT_BYTES,
        },
        ExchangeComponent {
            label: "W13 scale",
            send: exchange_slab.w13_scale.buffer(),
            destination: final_slab.w13_scale.buffer(),
            expert_stride_bytes: DS4_FLASH_W13_SCALE_EXPERT_BYTES,
        },
        ExchangeComponent {
            label: "W2 weight",
            send: exchange_slab.w2_weight.buffer(),
            destination: final_slab.w2_weight.buffer(),
            expert_stride_bytes: DS4_FLASH_W2_WEIGHT_EXPERT_BYTES,
        },
        ExchangeComponent {
            label: "W2 scale",
            send: exchange_slab.w2_scale.buffer(),
            destination: final_slab.w2_scale.buffer(),
            expert_stride_bytes: DS4_FLASH_W2_SCALE_EXPERT_BYTES,
        },
    ];
    let maximum_row_stride = components
        .iter()
        .map(|component| {
            component
                .expert_stride_bytes
                .checked_mul(source_expert_count)
                .context("native Flash cooperative row stride overflow")
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .max()
        .context("native Flash cooperative exchange has no components")?;
    let receive_bytes = maximum_row_stride
        .checked_mul(world_size - 1)
        .context("native Flash cooperative receive byte count overflow")?;
    if workspace
        .receive
        .as_ref()
        .is_none_or(|receive| receive.capacity_bytes() < receive_bytes)
    {
        let allocation_started = Instant::now();
        workspace.receive = Some(OwnedDeviceAllocation::new(
            Arc::clone(&cuda_cache.library),
            receive_bytes,
            "native Flash cooperative receive workspace",
        )?);
        allocation_ms += elapsed_ms(allocation_started);
    }
    let receive = workspace
        .receive
        .as_ref()
        .context("native Flash cooperative receive workspace was not initialized")?;
    let exchange_started = Instant::now();
    for component in components {
        let row_stride = component
            .expert_stride_bytes
            .checked_mul(source_expert_count)
            .context("native Flash cooperative component row stride overflow")?;
        let component_receive_bytes = row_stride
            .checked_mul(world_size - 1)
            .context("native Flash cooperative component receive byte count overflow")?;
        let receive_view = device_buffer_byte_view(
            receive.buffer(),
            0,
            component_receive_bytes,
            component.label,
        )?;
        unsafe {
            communicator
                .row_all_to_all_u8_async(
                    component.send,
                    receive_view,
                    world_size,
                    row_stride,
                    cuda_stream,
                )
                .with_context(|| {
                    format!(
                        "exchanging native Flash layer {layer_id} {}",
                        component.label
                    )
                })?;
        }
        for source_rank in 0..world_size {
            let source_group = &expert_groups[source_rank];
            let segment = if source_rank == shard.rank {
                device_buffer_byte_view(
                    component.send,
                    shard.rank * row_stride,
                    row_stride,
                    "native Flash local exchange segment",
                )?
            } else {
                let receive_index = if source_rank < shard.rank {
                    source_rank
                } else {
                    source_rank - 1
                };
                device_buffer_byte_view(
                    receive_view,
                    receive_index * row_stride,
                    row_stride,
                    "native Flash remote exchange segment",
                )?
            };
            for (source_index, &expert_id) in source_group.iter().enumerate() {
                let source = device_buffer_byte_view(
                    segment,
                    source_index * component.expert_stride_bytes,
                    component.expert_stride_bytes,
                    "native Flash exchange expert source",
                )?;
                let destination = device_buffer_byte_view(
                    component.destination,
                    expert_id * component.expert_stride_bytes,
                    component.expert_stride_bytes,
                    "native Flash resident expert destination",
                )?;
                unsafe {
                    cuda_cache
                        .library
                        .copy_d2d_async(
                            destination,
                            source,
                            component.expert_stride_bytes,
                            cuda_stream,
                        )
                        .with_context(|| {
                            format!(
                                "scattering native Flash layer {layer_id} {} expert {expert_id}",
                                component.label
                            )
                        })?;
                }
            }
        }
    }
    unsafe {
        cuda_cache
            .library
            .cuda_stream_synchronize(cuda_stream)
            .context("synchronizing native Flash cooperative exchange")?;
    }
    let exchange_ms = elapsed_ms(exchange_started);
    anyhow::ensure!(
        cuda_cache
            .ds4_flash_expert_slabs
            .insert(layer_id, Arc::clone(&final_slab))
            .is_none(),
        "native Flash cooperative preload inserted duplicate layer {layer_id}"
    );
    let projection_groups = expert_ids.len() * 3;
    preload.projection_groups += projection_groups;
    preload.weight_bytes += final_slab.resident_weight_bytes();
    preload.weight_scale_bytes += final_slab.resident_scale_bytes();
    cuda_cache.projection_uploads += projection_groups;
    let source_gbps = physical_source_bytes_read as f64 / (load_ms * 1.0e6).max(1.0);
    let source_io_ms = source_io_micros
        .map(|micros| micros as f64 / 1_000.0)
        .unwrap_or_default();
    let source_io_gbps = source_io_micros
        .map(|micros| physical_source_bytes_read as f64 / (micros as f64 * 1.0e3).max(1.0))
        .unwrap_or_default();
    eprintln!(
        "ds4_flash_cuda_layer_preload layer_id={layer_id} experts={} source_experts={} io_workers={} direct_io={} cooperative=true source_bytes_requested={source_bytes_read} source_bytes_read={physical_source_bytes_read} source_gbps={source_gbps:.3} source_io_gbps={source_io_gbps:.3} load_ms={load_ms:.3} source_io_ms={source_io_ms:.3} allocation_ms={allocation_ms:.3} compact_ms={compact_ms:.3} pack_ms={pack_ms:.3} exchange_ms={exchange_ms:.3}",
        expert_ids.len(),
        loaded_source_experts.len(),
        route_preload_io_workers(),
        route_preload_direct_io(),
    );
    Ok(())
}

fn preload_native_flash_projection_cuda_cache_cooperative_tp4(
    catalog: &TensorCatalog,
    layers: &[usize],
    expert_ids: &[usize],
    cuda_cache: &mut RouteCudaCache,
    shard: ExpertIntermediateShard,
    cuda_stream: *mut c_void,
) -> Result<RouteCudaProjectionPreload> {
    let preload_started = Instant::now();
    let world_size = cuda_cache
        .weight_preload_communicator
        .as_ref()
        .context("native Flash cooperative preload requires a communicator")?
        .world_size();
    anyhow::ensure!(
        world_size == shard.count,
        "native Flash cooperative world size {world_size} differs from TP shard count {}",
        shard.count
    );
    let expert_groups = layers
        .iter()
        .map(|&layer_id| {
            native_flash_cooperative_weight_preload_expert_groups(
                catalog, layer_id, expert_ids, world_size,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let mut preload = RouteCudaProjectionPreload::default();
    let mut physical_source_bytes = 0_u64;
    let mut workspace = RouteCudaDs4FlashCooperativePreloadWorkspace::default();
    std::thread::scope(|scope| -> Result<()> {
        let first_layer = layers
            .first()
            .context("native Flash cooperative preload has no layers")?;
        let first_source_ids = &expert_groups[0][shard.rank];
        let mut pending = Some(scope.spawn(move || {
            let started = Instant::now();
            let loaded = load_native_flash_layer_expert_shards_parallel(
                catalog,
                *first_layer,
                first_source_ids,
                shard,
                true,
            )?;
            Ok::<_, anyhow::Error>((loaded, elapsed_ms(started)))
        }));
        for (index, &layer_id) in layers.iter().enumerate() {
            let (loaded, load_ms) = pending
                .take()
                .expect("native Flash current preload worker exists")
                .join()
                .map_err(|_| anyhow::anyhow!("native Flash preload worker panicked"))??;
            pending = layers.get(index + 1).map(|&next_layer| {
                let next_source_ids = &expert_groups[index + 1][shard.rank];
                scope.spawn(move || {
                    let started = Instant::now();
                    let loaded = load_native_flash_layer_expert_shards_parallel(
                        catalog,
                        next_layer,
                        next_source_ids,
                        shard,
                        true,
                    )?;
                    Ok::<_, anyhow::Error>((loaded, elapsed_ms(started)))
                })
            });
            physical_source_bytes = physical_source_bytes
                .checked_add(loaded.physical_source_bytes_read)
                .context("native Flash cooperative source byte total overflow")?;
            store_native_flash_layer_cooperative(
                catalog,
                layer_id,
                expert_ids,
                &expert_groups[index],
                &loaded.experts,
                cuda_cache,
                shard,
                cuda_stream,
                &mut workspace,
                &mut preload,
                load_ms,
                loaded.source_io_micros,
                loaded.source_bytes_read,
                loaded.physical_source_bytes_read,
            )?;
        }
        Ok(())
    })?;
    let total_ms = elapsed_ms(preload_started);
    let effective_source_gbps = physical_source_bytes as f64 / (total_ms * 1.0e6).max(1.0);
    eprintln!(
        "ds4_flash_cooperative_preload_complete layers={} read_ahead_layers=1 source_bytes={physical_source_bytes} total_ms={total_ms:.3} effective_source_gbps={effective_source_gbps:.3}",
        layers.len(),
    );
    Ok(preload)
}

fn preload_native_flash_projection_cuda_cache_local_tp4(
    catalog: &TensorCatalog,
    layers: &[usize],
    expert_ids: &[usize],
    cuda_cache: &mut RouteCudaCache,
    shard: ExpertIntermediateShard,
    cuda_stream: *mut c_void,
) -> Result<RouteCudaProjectionPreload> {
    let mut preload = RouteCudaProjectionPreload::default();
    std::thread::scope(|scope| -> Result<()> {
        let first_layer = layers
            .first()
            .context("native Flash local preload has no layers")?;
        let mut pending = Some(scope.spawn(move || {
            let started = Instant::now();
            let loaded = load_native_flash_layer_expert_shards_parallel(
                catalog,
                *first_layer,
                expert_ids,
                shard,
                false,
            )?;
            Ok::<_, anyhow::Error>((loaded, elapsed_ms(started)))
        }));
        for (index, &layer_id) in layers.iter().enumerate() {
            let (loaded, load_ms) = pending
                .take()
                .expect("native Flash current local preload worker exists")
                .join()
                .map_err(|_| anyhow::anyhow!("native Flash local preload worker panicked"))??;
            pending = layers.get(index + 1).map(|&next_layer| {
                scope.spawn(move || {
                    let started = Instant::now();
                    let loaded = load_native_flash_layer_expert_shards_parallel(
                        catalog, next_layer, expert_ids, shard, false,
                    )?;
                    Ok::<_, anyhow::Error>((loaded, elapsed_ms(started)))
                })
            });
            let allocation_started = Instant::now();
            let slab = Arc::new(RouteCudaDs4FlashLayerExpertSlab::new(
                Arc::clone(&cuda_cache.library),
                catalog,
                layer_id,
            )?);
            let allocation_ms = elapsed_ms(allocation_started);
            let pack_started = Instant::now();
            for (id_chunk, expert_chunk) in expert_ids.chunks(32).zip(loaded.experts.chunks(32)) {
                slab.store_layer_experts(
                    id_chunk,
                    expert_chunk,
                    Arc::clone(&cuda_cache.library),
                    &mut cuda_cache.workspace,
                    cuda_stream,
                )?;
            }
            let pack_ms = elapsed_ms(pack_started);
            anyhow::ensure!(
                cuda_cache
                    .ds4_flash_expert_slabs
                    .insert(layer_id, Arc::clone(&slab))
                    .is_none(),
                "native Flash local preload inserted duplicate layer {layer_id}"
            );
            let projection_groups = expert_ids.len() * 3;
            preload.projection_groups += projection_groups;
            preload.weight_bytes += slab.resident_weight_bytes();
            preload.weight_scale_bytes += slab.resident_scale_bytes();
            cuda_cache.projection_uploads += projection_groups;
            let source_gbps = loaded.physical_source_bytes_read as f64 / (load_ms * 1.0e6).max(1.0);
            eprintln!(
                "ds4_flash_cuda_layer_preload layer_id={layer_id} experts={} io_workers={} direct_io={} cooperative=false source_bytes_requested={} source_bytes_read={} source_gbps={source_gbps:.3} load_ms={load_ms:.3} allocation_ms={allocation_ms:.3} pack_ms={pack_ms:.3}",
                expert_ids.len(),
                route_preload_io_workers(),
                route_preload_direct_io(),
                loaded.source_bytes_read,
                loaded.physical_source_bytes_read,
            );
        }
        Ok(())
    })?;
    Ok(preload)
}

#[allow(clippy::too_many_arguments)]
fn store_exl3_flash_layer_cooperative(
    catalog: &TensorCatalog,
    layer_id: usize,
    expert_ids: &[usize],
    expert_groups: &[Vec<usize>],
    loaded: &LoadedExl3FullLayer,
    cuda_cache: &mut RouteCudaCache,
    shard: ExpertIntermediateShard,
    cuda_stream: *mut c_void,
    workspace: &mut RouteCudaDs4FlashExl3CooperativePreloadWorkspace,
    preload: &mut RouteCudaProjectionPreload,
    load_ms: f64,
) -> Result<()> {
    let world_size = shard.count;
    anyhow::ensure!(
        world_size == 4
            && expert_groups.len() == world_size
            && expert_groups
                .iter()
                .all(|group| group.len() == loaded.expert_ids.len()),
        "EXL3 cooperative layer {layer_id} groups do not match rank {}/{} source expert count {}",
        shard.rank,
        world_size,
        loaded.expert_ids.len()
    );
    let source_expert_count = loaded.expert_ids.len();
    let mut allocation_ms = 0.0;
    let mut compact_pack_ms = 0.0;
    let final_allocation_started = Instant::now();
    let final_slab = Arc::new(RouteCudaDs4FlashExl3LayerExpertSlab::new(
        Arc::clone(&cuda_cache.library),
        catalog,
        layer_id,
    )?);
    allocation_ms += elapsed_ms(final_allocation_started);
    if workspace.exchange_slab.is_none() {
        let allocation_started = Instant::now();
        workspace.exchange_slab = Some(Arc::new(RouteCudaDs4FlashExl3LayerExpertSlab::new(
            Arc::clone(&cuda_cache.library),
            catalog,
            layer_id,
        )?));
        allocation_ms += elapsed_ms(allocation_started);
    }
    let exchange_slab = workspace
        .exchange_slab
        .as_ref()
        .context("EXL3 cooperative exchange slab was not initialized")?;
    for target_rank in 0..world_size {
        let target_shard = ExpertIntermediateShard::new(world_size, target_rank)?;
        let target_start = target_rank
            .checked_mul(source_expert_count)
            .context("EXL3 cooperative exchange expert offset overflow")?;
        let exchange_expert_ids =
            (target_start..target_start + source_expert_count).collect::<Vec<_>>();
        let compact_pack_started = Instant::now();
        exchange_slab.store_loaded_full_layer_experts(
            catalog,
            loaded,
            &exchange_expert_ids,
            target_shard,
            Arc::clone(&cuda_cache.library),
            &mut cuda_cache.workspace,
            cuda_stream,
        )?;
        compact_pack_ms += elapsed_ms(compact_pack_started);
    }

    let communicator = cuda_cache
        .weight_preload_communicator
        .as_ref()
        .context("EXL3 cooperative preload has no NCCL communicator")?;
    anyhow::ensure!(
        communicator.world_size() == world_size && communicator.rank() == shard.rank,
        "EXL3 cooperative communicator rank {}/{} differs from shard {}/{}",
        communicator.rank(),
        communicator.world_size(),
        shard.rank,
        world_size
    );
    let geometry = final_slab.profile.geometry;
    let trellis = usize::try_from(geometry.projection_trellis_bytes)
        .context("EXL3 cooperative trellis stride exceeds usize")?;
    let hidden_rotation = geometry.hidden_size * std::mem::size_of::<u16>();
    let local_rotation = geometry.local_intermediate_size * std::mem::size_of::<u16>();
    struct ExchangeComponent {
        label: &'static str,
        send: Ds41rtDeviceBuffer,
        destination: Ds41rtDeviceBuffer,
        expert_stride_bytes: usize,
    }
    let all_trellis_bytes = expert_ids.len() * trellis;
    let components = [
        ExchangeComponent {
            label: "gate trellis",
            send: device_buffer_byte_view(
                exchange_slab.w13_trellis.buffer(),
                0,
                all_trellis_bytes,
                "EXL3 cooperative gate trellis send",
            )?,
            destination: device_buffer_byte_view(
                final_slab.w13_trellis.buffer(),
                0,
                all_trellis_bytes,
                "EXL3 cooperative gate trellis destination",
            )?,
            expert_stride_bytes: trellis,
        },
        ExchangeComponent {
            label: "up trellis",
            send: device_buffer_byte_view(
                exchange_slab.w13_trellis.buffer(),
                all_trellis_bytes,
                all_trellis_bytes,
                "EXL3 cooperative up trellis send",
            )?,
            destination: device_buffer_byte_view(
                final_slab.w13_trellis.buffer(),
                all_trellis_bytes,
                all_trellis_bytes,
                "EXL3 cooperative up trellis destination",
            )?,
            expert_stride_bytes: trellis,
        },
        ExchangeComponent {
            label: "down trellis",
            send: exchange_slab.w2_trellis.buffer(),
            destination: final_slab.w2_trellis.buffer(),
            expert_stride_bytes: trellis,
        },
        ExchangeComponent {
            label: "gate SUH",
            send: exchange_slab.gate_suh.buffer(),
            destination: final_slab.gate_suh.buffer(),
            expert_stride_bytes: hidden_rotation,
        },
        ExchangeComponent {
            label: "up SUH",
            send: exchange_slab.up_suh.buffer(),
            destination: final_slab.up_suh.buffer(),
            expert_stride_bytes: hidden_rotation,
        },
        ExchangeComponent {
            label: "intermediate rotations",
            send: exchange_slab.intermediate_rotations.buffer(),
            destination: final_slab.intermediate_rotations.buffer(),
            expert_stride_bytes: 3 * local_rotation,
        },
        ExchangeComponent {
            label: "down SVH",
            send: exchange_slab.down_svh.buffer(),
            destination: final_slab.down_svh.buffer(),
            expert_stride_bytes: hidden_rotation,
        },
    ];
    let maximum_row_stride = components
        .iter()
        .map(|component| {
            component
                .expert_stride_bytes
                .checked_mul(source_expert_count)
                .context("EXL3 cooperative row stride overflow")
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .max()
        .context("EXL3 cooperative exchange has no components")?;
    let receive_bytes = maximum_row_stride
        .checked_mul(world_size - 1)
        .context("EXL3 cooperative receive byte count overflow")?;
    if workspace
        .receive
        .as_ref()
        .is_none_or(|receive| receive.capacity_bytes() < receive_bytes)
    {
        let allocation_started = Instant::now();
        workspace.receive = Some(OwnedDeviceAllocation::new(
            Arc::clone(&cuda_cache.library),
            receive_bytes,
            "EXL3 cooperative receive workspace",
        )?);
        allocation_ms += elapsed_ms(allocation_started);
    }
    let receive = workspace
        .receive
        .as_ref()
        .context("EXL3 cooperative receive workspace was not initialized")?;
    let exchange_started = Instant::now();
    for component in components {
        let row_stride = component
            .expert_stride_bytes
            .checked_mul(source_expert_count)
            .context("EXL3 cooperative component row stride overflow")?;
        let component_receive_bytes = row_stride
            .checked_mul(world_size - 1)
            .context("EXL3 cooperative component receive byte count overflow")?;
        let receive_view = device_buffer_byte_view(
            receive.buffer(),
            0,
            component_receive_bytes,
            component.label,
        )?;
        unsafe {
            communicator
                .row_all_to_all_u8_async(
                    component.send,
                    receive_view,
                    world_size,
                    row_stride,
                    cuda_stream,
                )
                .with_context(|| format!("exchanging EXL3 layer {layer_id} {}", component.label))?;
        }
        for source_rank in 0..world_size {
            let source_group = &expert_groups[source_rank];
            let segment = if source_rank == shard.rank {
                device_buffer_byte_view(
                    component.send,
                    shard.rank * row_stride,
                    row_stride,
                    "EXL3 cooperative local exchange segment",
                )?
            } else {
                let receive_index = if source_rank < shard.rank {
                    source_rank
                } else {
                    source_rank - 1
                };
                device_buffer_byte_view(
                    receive_view,
                    receive_index * row_stride,
                    row_stride,
                    "EXL3 cooperative remote exchange segment",
                )?
            };
            for (source_index, &expert_id) in source_group.iter().enumerate() {
                let source = device_buffer_byte_view(
                    segment,
                    source_index * component.expert_stride_bytes,
                    component.expert_stride_bytes,
                    "EXL3 cooperative expert source",
                )?;
                let destination = device_buffer_byte_view(
                    component.destination,
                    expert_id * component.expert_stride_bytes,
                    component.expert_stride_bytes,
                    "EXL3 cooperative resident expert destination",
                )?;
                unsafe {
                    cuda_cache
                        .library
                        .copy_d2d_async(
                            destination,
                            source,
                            component.expert_stride_bytes,
                            cuda_stream,
                        )
                        .with_context(|| {
                            format!(
                                "scattering EXL3 layer {layer_id} {} expert {expert_id}",
                                component.label
                            )
                        })?;
                }
            }
        }
    }
    unsafe {
        cuda_cache
            .library
            .cuda_stream_synchronize(cuda_stream)
            .context("synchronizing EXL3 cooperative exchange")?;
    }
    let exchange_ms = elapsed_ms(exchange_started);
    anyhow::ensure!(
        cuda_cache
            .ds4_flash_exl3_expert_slabs
            .insert(layer_id, Arc::clone(&final_slab))
            .is_none(),
        "EXL3 cooperative preload inserted duplicate layer {layer_id}"
    );
    let projection_groups = expert_ids.len() * 3;
    preload.projection_groups += projection_groups;
    preload.weight_bytes += final_slab.resident_weight_bytes();
    preload.weight_scale_bytes += final_slab.resident_scale_bytes();
    cuda_cache.projection_uploads += projection_groups;
    let source_gbps = loaded.source_bytes as f64 / (load_ms * 1.0e6).max(1.0);
    eprintln!(
        "ds4_flash_exl3_cuda_layer_preload layer_id={layer_id} experts={} source_experts={} tp_rank={} cooperative=true source_bytes={} source_requests={} source_spans={} direct_io={} source_gbps={source_gbps:.3} load_ms={load_ms:.3} allocation_ms={allocation_ms:.3} compact_pack_ms={compact_pack_ms:.3} exchange_ms={exchange_ms:.3}",
        expert_ids.len(),
        loaded.expert_ids.len(),
        shard.rank,
        loaded.source_bytes,
        loaded.source_requests,
        loaded.source_spans,
        loaded.direct_io,
    );
    Ok(())
}

fn preload_exl3_flash_projection_cuda_cache_cooperative_tp4(
    catalog: &TensorCatalog,
    layers: &[usize],
    expert_ids: &[usize],
    cuda_cache: &mut RouteCudaCache,
    shard: ExpertIntermediateShard,
    cuda_stream: *mut c_void,
) -> Result<RouteCudaProjectionPreload> {
    let preload_started = Instant::now();
    let world_size = cuda_cache
        .weight_preload_communicator
        .as_ref()
        .context("EXL3 cooperative preload requires a communicator")?
        .world_size();
    anyhow::ensure!(
        world_size == shard.count,
        "EXL3 cooperative world size {world_size} differs from TP shard count {}",
        shard.count
    );
    let expert_groups = layers
        .iter()
        .map(|&layer_id| {
            exl3_cooperative_weight_preload_expert_groups(catalog, layer_id, expert_ids, world_size)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut preload = RouteCudaProjectionPreload::default();
    let mut physical_source_bytes = 0_u64;
    let mut workspace = RouteCudaDs4FlashExl3CooperativePreloadWorkspace::default();
    std::thread::scope(|scope| -> Result<()> {
        let first_layer = layers
            .first()
            .context("EXL3 cooperative preload has no layers")?;
        let first_source_ids = &expert_groups[0][shard.rank];
        let mut pending = Some(scope.spawn(move || {
            let started = Instant::now();
            let loaded = load_exl3_full_layer_experts(catalog, *first_layer, first_source_ids)?;
            Ok::<_, anyhow::Error>((loaded, elapsed_ms(started)))
        }));
        for (index, &layer_id) in layers.iter().enumerate() {
            let (loaded, load_ms) = pending
                .take()
                .expect("EXL3 current preload worker exists")
                .join()
                .map_err(|_| anyhow::anyhow!("EXL3 preload worker panicked"))??;
            pending = layers.get(index + 1).map(|&next_layer| {
                let next_source_ids = &expert_groups[index + 1][shard.rank];
                scope.spawn(move || {
                    let started = Instant::now();
                    let loaded =
                        load_exl3_full_layer_experts(catalog, next_layer, next_source_ids)?;
                    Ok::<_, anyhow::Error>((loaded, elapsed_ms(started)))
                })
            });
            physical_source_bytes = physical_source_bytes
                .checked_add(loaded.source_bytes)
                .context("EXL3 cooperative source byte total overflow")?;
            store_exl3_flash_layer_cooperative(
                catalog,
                layer_id,
                expert_ids,
                &expert_groups[index],
                &loaded,
                cuda_cache,
                shard,
                cuda_stream,
                &mut workspace,
                &mut preload,
                load_ms,
            )?;
        }
        Ok(())
    })?;
    let total_ms = elapsed_ms(preload_started);
    let effective_source_gbps = physical_source_bytes as f64 / (total_ms * 1.0e6).max(1.0);
    eprintln!(
        "ds4_flash_exl3_cooperative_preload_complete layers={} read_ahead_layers=1 source_bytes={physical_source_bytes} total_ms={total_ms:.3} effective_source_gbps={effective_source_gbps:.3}",
        layers.len(),
    );
    Ok(preload)
}

fn preload_exl3_flash_projection_cuda_cache_local_tp4(
    catalog: &TensorCatalog,
    layers: &[usize],
    expert_ids: &[usize],
    cuda_cache: &mut RouteCudaCache,
    shard: ExpertIntermediateShard,
    cuda_stream: *mut c_void,
) -> Result<RouteCudaProjectionPreload> {
    let mut preload = RouteCudaProjectionPreload::default();
    for &layer_id in layers {
        let allocation_started = Instant::now();
        let slab = Arc::new(RouteCudaDs4FlashExl3LayerExpertSlab::new(
            Arc::clone(&cuda_cache.library),
            catalog,
            layer_id,
        )?);
        let allocation_ms = elapsed_ms(allocation_started);
        let load_started = Instant::now();
        let source_bytes = slab.store_layer_experts_direct(
            catalog,
            expert_ids,
            shard,
            Arc::clone(&cuda_cache.library),
            &mut cuda_cache.workspace,
            cuda_stream,
        )?;
        let load_ms = elapsed_ms(load_started);
        anyhow::ensure!(
            cuda_cache
                .ds4_flash_exl3_expert_slabs
                .insert(layer_id, Arc::clone(&slab))
                .is_none(),
            "EXL3 Flash local preload inserted duplicate layer {layer_id}"
        );
        let projection_groups = expert_ids.len() * 3;
        preload.projection_groups += projection_groups;
        preload.weight_bytes += slab.resident_weight_bytes();
        preload.weight_scale_bytes += slab.resident_scale_bytes();
        cuda_cache.projection_uploads += projection_groups;
        let source_gbps = source_bytes as f64 / (load_ms * 1.0e6).max(1.0);
        eprintln!(
            "ds4_flash_exl3_cuda_layer_preload layer_id={layer_id} experts={} tp_rank={} source_bytes={source_bytes} source_gbps={source_gbps:.3} allocation_ms={allocation_ms:.3} load_ms={load_ms:.3} direct_resident=true cuda_repack=false",
            expert_ids.len(),
            shard.rank,
        );
    }
    Ok(preload)
}

fn preload_exl3_mixed_flash_projection_cuda_cache_local_tp4(
    catalog: &TensorCatalog,
    layers: &[usize],
    cuda_cache: &mut RouteCudaCache,
    shard: ExpertIntermediateShard,
    cuda_stream: *mut c_void,
) -> Result<RouteCudaProjectionPreload> {
    let mut preload = RouteCudaProjectionPreload::default();
    for &layer_id in layers {
        let projection_bits = (0..catalog.facts.routed_experts)
            .map(|expert_id| {
                let expert = exl3_expert(catalog, layer_id, expert_id)?;
                Ok([
                    exl3_projection_trellis_bits(expert.gate)?,
                    exl3_projection_trellis_bits(expert.up)?,
                    exl3_projection_trellis_bits(expert.down)?,
                ])
            })
            .collect::<Result<Vec<_>>>()?;
        let first_bits = projection_bits[0][0];
        if projection_bits
            .iter()
            .flatten()
            .all(|bits| *bits == first_bits)
        {
            let expert_ids = (0..catalog.facts.routed_experts).collect::<Vec<_>>();
            let uniform = preload_exl3_flash_projection_cuda_cache_local_tp4(
                catalog,
                &[layer_id],
                &expert_ids,
                cuda_cache,
                shard,
                cuda_stream,
            )?;
            preload.projection_groups += uniform.projection_groups;
            preload.weight_bytes += uniform.weight_bytes;
            preload.weight_scale_bytes += uniform.weight_scale_bytes;
            eprintln!(
                "ds4_flash_exl3_mixed_cuda_layer_preload layer_id={layer_id} uniform_k={} experts={} tp_rank={} one_grid=false",
                first_bits,
                expert_ids.len(),
                shard.rank,
            );
            continue;
        }
        let allocation_started = Instant::now();
        let slab = Arc::new(RouteCudaDs4FlashExl3MixedLayerExpertSlab::new(
            Arc::clone(&cuda_cache.library),
            catalog,
            layer_id,
        )?);
        let allocation_ms = elapsed_ms(allocation_started);
        let load_started = Instant::now();
        let source_bytes = slab.store_layer_experts_direct(
            catalog,
            shard,
            Arc::clone(&cuda_cache.library),
            &mut cuda_cache.workspace,
            cuda_stream,
        )?;
        let load_ms = elapsed_ms(load_started);
        anyhow::ensure!(
            cuda_cache
                .ds4_flash_exl3_mixed_expert_slabs
                .insert(layer_id, Arc::clone(&slab))
                .is_none(),
            "mixed EXL3 Flash preload inserted duplicate layer {layer_id}"
        );
        let projection_groups = catalog.facts.routed_experts * 3;
        preload.projection_groups += projection_groups;
        preload.weight_bytes += slab.resident_weight_bytes();
        preload.weight_scale_bytes += slab.resident_scale_bytes();
        cuda_cache.projection_uploads += projection_groups;
        let source_gbps = source_bytes as f64 / (load_ms * 1.0e6).max(1.0);
        eprintln!(
            "ds4_flash_exl3_mixed_cuda_layer_preload layer_id={layer_id} k2_slots={} k3_slots={} gate_k3={} up_k3={} down_k3={} tp_rank={} source_bytes={source_bytes} source_gbps={source_gbps:.3} allocation_ms={allocation_ms:.3} load_ms={load_ms:.3} one_grid=true projection_mixed=true",
            slab.tier0_slots,
            slab.tier1_slots,
            slab.tier1_gate_expert_ids.len(),
            slab.tier1_up_expert_ids.len(),
            slab.tier1_down_expert_ids.len(),
            shard.rank,
        );
    }
    Ok(preload)
}

/// Preload exactly one replicated-expert Flash layer for one TP=4 rank.
/// This is the bounded startup boundary used by inactive end-to-end sparse
/// block qualification; it deliberately does not initialize EP ownership or
/// preload unrelated layers.
#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::commands::real_full) fn preload_native_flash_layer_cuda_cache_local_tp4(
    catalog: &TensorCatalog,
    layer_id: usize,
    shard: ExpertIntermediateShard,
    cache: &mut RouteTensorCache,
) -> Result<RouteCudaProjectionPreload> {
    require_sparkinfer_exl3_mode(catalog)?;
    let mixed_exl3 = is_deepseek_v4_mixed_exl3_recipe(&catalog.facts.quantization_recipe);
    let exl3_profile = (is_deepseek_v4_exl3_recipe(&catalog.facts.quantization_recipe)
        && !mixed_exl3)
        .then(|| Ds4Exl3AotProfile::from_catalog(catalog))
        .transpose()?;
    anyhow::ensure!(
        catalog.facts.model_type == "deepseek_v4"
            && (exl3_profile.is_some()
                || mixed_exl3
                || (catalog.facts.variant == ModelVariant::Flash
                    && catalog.facts.quantization_recipe == "deepseek_v4_native_fp4_fp8_mixed_v1")),
        "native DeepSeek V4 layer preload cannot consume variant={:?} model_type={:?} recipe={:?}",
        catalog.facts.variant,
        catalog.facts.model_type,
        catalog.facts.quantization_recipe
    );
    anyhow::ensure!(
        shard.count == 4,
        "native Flash layer preload requires TP=4, got TP={}",
        shard.count
    );
    anyhow::ensure!(
        layer_id < catalog.facts.total_transformer_blocks(),
        "native Flash layer preload index {layer_id} exceeds checkpoint depth {}",
        catalog.facts.total_transformer_blocks()
    );
    let cuda_cache = cache.cuda_cache_native_flash_tp4(shard)?;
    if let Some(profile) = exl3_profile {
        cuda_cache.ensure_ds4_exl3_aot(profile.variant)?;
    } else if mixed_exl3 {
        cuda_cache.ensure_ds4_flash_aot()?;
    } else {
        cuda_cache.ensure_ds4_flash_aot()?;
    }
    let stream = RouteCudaStream::new(Arc::clone(&cuda_cache.library))?;
    let expert_ids = (0..catalog.facts.routed_experts).collect::<Vec<_>>();
    let preload = if mixed_exl3 {
        preload_exl3_mixed_flash_projection_cuda_cache_local_tp4(
            catalog,
            &[layer_id],
            cuda_cache,
            shard,
            stream.as_ptr(),
        )?
    } else if is_deepseek_v4_exl3_recipe(&catalog.facts.quantization_recipe) {
        preload_exl3_flash_projection_cuda_cache_local_tp4(
            catalog,
            &[layer_id],
            &expert_ids,
            cuda_cache,
            shard,
            stream.as_ptr(),
        )?
    } else {
        preload_native_flash_projection_cuda_cache_local_tp4(
            catalog,
            &[layer_id],
            &expert_ids,
            cuda_cache,
            shard,
            stream.as_ptr(),
        )?
    };
    let expected = expert_ids.len() * 3;
    anyhow::ensure!(
        preload.projection_groups == expected,
        "native Flash layer preload loaded {} projections, expected {expected}",
        preload.projection_groups
    );
    clear_route_cuda_aligned_read_pool();
    Ok(preload)
}

pub(in crate::commands::real_full) fn preload_native_flash_projection_cuda_cache(
    catalog: &TensorCatalog,
    cache: &mut RouteTensorCache,
    shard: ExpertIntermediateShard,
) -> Result<RouteCudaProjectionPreload> {
    require_sparkinfer_exl3_mode(catalog)?;
    let mixed_exl3 = is_deepseek_v4_mixed_exl3_recipe(&catalog.facts.quantization_recipe);
    let exl3_profile = (is_deepseek_v4_exl3_recipe(&catalog.facts.quantization_recipe)
        && !mixed_exl3)
        .then(|| Ds4Exl3AotProfile::from_catalog(catalog))
        .transpose()?;
    anyhow::ensure!(
        catalog.facts.model_type == "deepseek_v4"
            && (exl3_profile.is_some()
                || mixed_exl3
                || (catalog.facts.variant == ModelVariant::Flash
                    && catalog.facts.quantization_recipe == "deepseek_v4_native_fp4_fp8_mixed_v1")),
        "native DeepSeek V4 CUDA preload cannot consume variant={:?} model_type={:?} recipe={:?}",
        catalog.facts.variant,
        catalog.facts.model_type,
        catalog.facts.quantization_recipe
    );
    anyhow::ensure!(
        shard.count == 4,
        "native Flash expert preload requires TP=4, got TP={}",
        shard.count
    );
    let cuda_cache = cache.cuda_cache_native_flash_tp4(shard)?;
    if let Some(profile) = exl3_profile {
        cuda_cache.ensure_ds4_exl3_aot(profile.variant)?;
    } else if mixed_exl3 {
        cuda_cache.ensure_ds4_flash_aot()?;
    } else {
        cuda_cache.ensure_ds4_flash_aot()?;
    }
    let exl3 = is_deepseek_v4_exl3_recipe(&catalog.facts.quantization_recipe);
    let cooperative = !mixed_exl3
        && env::var(REAL_FULL_NVFP4_ROUTE_PRELOAD_COOPERATIVE_ENV)
            .map(|value| {
                !matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "0" | "false" | "no" | "off" | "disabled"
                )
            })
            .unwrap_or(true);
    if cooperative && cuda_cache.weight_preload_communicator.is_none() {
        cuda_cache.weight_preload_communicator =
            initialize_spark_weight_preload_communicator(&cuda_cache.library, Some(shard))?;
    }
    let stream = RouteCudaStream::new(Arc::clone(&cuda_cache.library))?;
    let cuda_stream = stream.as_ptr();
    let layers = (0..catalog.facts.total_transformer_blocks()).collect::<Vec<_>>();
    let expert_ids = (0..catalog.facts.routed_experts).collect::<Vec<_>>();
    let preload = if mixed_exl3 {
        preload_exl3_mixed_flash_projection_cuda_cache_local_tp4(
            catalog,
            &layers,
            cuda_cache,
            shard,
            cuda_stream,
        )?
    } else if exl3 && cuda_cache.weight_preload_communicator.is_some() {
        preload_exl3_flash_projection_cuda_cache_cooperative_tp4(
            catalog,
            &layers,
            &expert_ids,
            cuda_cache,
            shard,
            cuda_stream,
        )?
    } else if exl3 {
        preload_exl3_flash_projection_cuda_cache_local_tp4(
            catalog,
            &layers,
            &expert_ids,
            cuda_cache,
            shard,
            cuda_stream,
        )?
    } else if cuda_cache.weight_preload_communicator.is_some() {
        preload_native_flash_projection_cuda_cache_cooperative_tp4(
            catalog,
            &layers,
            &expert_ids,
            cuda_cache,
            shard,
            cuda_stream,
        )?
    } else {
        preload_native_flash_projection_cuda_cache_local_tp4(
            catalog,
            &layers,
            &expert_ids,
            cuda_cache,
            shard,
            cuda_stream,
        )?
    };
    let expected = layers.len() * expert_ids.len() * 3;
    anyhow::ensure!(
        preload.projection_groups == expected,
        "native Flash preload loaded {} projections, expected {expected}",
        preload.projection_groups
    );
    clear_route_cuda_aligned_read_pool();
    drop(cuda_cache.weight_preload_communicator.take());
    Ok(preload)
}

pub(in crate::commands::real_full) fn preload_routed_quant_projection_cuda_cache(
    catalog: &TensorCatalog,
    requests: &[RouteProjectionCachePreloadRequest],
    cache: &mut RouteTensorCache,
) -> Result<RouteCudaProjectionPreload> {
    if requests.is_empty() {
        anyhow::bail!("real NVFP4 CUDA resident preload requires at least one projection request");
    }
    let scalar_metadata = cache.scalar_metadata.clone();
    let cuda_cache = cache.cuda_cache()?;
    let stream = RouteCudaStream::new(Arc::clone(&cuda_cache.library))?;
    let cuda_stream = stream.as_ptr();
    let shard = spark_expert_intermediate_shard_from_env()?
        .context("packed W4A16 expert preload requires four intermediate shards")?;
    anyhow::ensure!(
        cuda_cache.b12x_aot_enabled && cuda_cache.b12x_w4a16_packed,
        "CUDA expert preload requires the packed W4A16 SparkInfer backend"
    );
    preload_routed_quant_projection_cuda_cache_contiguous_tp4(
        catalog,
        requests,
        cuda_cache,
        &scalar_metadata,
        shard,
        cuda_stream,
    )
}

fn load_routed_quant_projection(
    catalog: &TensorCatalog,
    layer_id: usize,
    expert_id: usize,
    projection: &str,
    row_count: usize,
) -> Result<RoutedQuantProjection> {
    let base_name = routed_quant_projection_base_name(layer_id, expert_id, projection);
    let intermediate_shard = spark_expert_intermediate_shard_from_env()?;
    let weight_name = format!("{base_name}.weight");
    let weight_scale_name = format!("{base_name}.weight_scale");
    let weight = if let Some(shard) = intermediate_shard {
        load_routed_projection_rows_for_shard(catalog, &weight_name, projection, row_count, shard)?
    } else {
        load_tensor_rows(catalog, &weight_name, 0, row_count)?
    };
    let weight_scale = if let Some(shard) = intermediate_shard {
        load_routed_projection_rows_for_shard(
            catalog,
            &weight_scale_name,
            projection,
            row_count,
            shard,
        )?
    } else {
        load_tensor_rows(catalog, &weight_scale_name, 0, row_count)?
    };
    let input_scale = load_tensor_bytes(catalog, &format!("{base_name}.input_scale"))?;
    let weight_scale_2 = load_tensor_bytes(catalog, &format!("{base_name}.weight_scale_2"))?;

    if weight.info.dtype != DType::U8
        || weight_scale.info.dtype != DType::F8E4M3
        || input_scale.info.dtype != DType::F32
        || weight_scale_2.info.dtype != DType::F32
    {
        anyhow::bail!(
            "real full NVFP4 expert probe expects U8 weight, F8E4M3 scale, and F32 scalar metadata for {base_name}, got {:?}, {:?}, {:?}, {:?}",
            weight.info.dtype,
            weight_scale.info.dtype,
            input_scale.info.dtype,
            weight_scale_2.info.dtype
        );
    }
    if !input_scale.info.shape.is_empty() || !weight_scale_2.info.shape.is_empty() {
        anyhow::bail!(
            "real full NVFP4 expert probe expected scalar metadata for {base_name}, got {:?} and {:?}",
            input_scale.info.shape,
            weight_scale_2.info.shape
        );
    }
    if weight.row_count != row_count || weight_scale.row_count != row_count {
        anyhow::bail!(
            "real full NVFP4 expert probe row count mismatch for {base_name}: weight={} scale={} expected={row_count}",
            weight.row_count,
            weight_scale.row_count
        );
    }

    Ok(RoutedQuantProjection {
        weight,
        weight_scale,
        input_scale,
        weight_scale_2,
    })
}

fn load_routed_projection_rows_for_shard(
    catalog: &TensorCatalog,
    tensor_name: &str,
    projection: &str,
    row_count: usize,
    shard: ExpertIntermediateShard,
) -> Result<LoadedTensorRows> {
    let info = catalog_tensor(catalog, tensor_name)?;
    let full_rows = *info
        .shape
        .first()
        .with_context(|| format!("sharded projection tensor {tensor_name} has no row dimension"))?;
    if matches!(projection, "gate_proj" | "up_proj") {
        let local_rows = shard.local_rows(full_rows)?;
        anyhow::ensure!(
            row_count == local_rows,
            "sharded projection {tensor_name} requested {row_count} rows, expected {local_rows}"
        );
        return load_tensor_rows(
            catalog,
            tensor_name,
            shard.row_start(full_rows)?,
            local_rows,
        );
    }
    anyhow::ensure!(
        projection == "down_proj" && row_count == full_rows,
        "unsupported sharded projection window for {tensor_name}: projection={projection} rows={row_count}/{full_rows}"
    );

    let full = load_tensor_rows(catalog, tensor_name, 0, full_rows)?;
    anyhow::ensure!(
        full.row_width % shard.count == 0,
        "sharded down projection {tensor_name} width {} is not divisible by {}",
        full.row_width,
        shard.count
    );
    let local_width = full.row_width / shard.count;
    let column_start = local_width
        .checked_mul(shard.rank)
        .context("sharded down projection column start overflow")?;
    let full_row_bytes = full
        .row_width
        .checked_mul(full.bytes_per_scalar)
        .context("sharded down projection full row byte count overflow")?;
    let local_row_bytes = local_width
        .checked_mul(full.bytes_per_scalar)
        .context("sharded down projection local row byte count overflow")?;
    let column_start_bytes = column_start
        .checked_mul(full.bytes_per_scalar)
        .context("sharded down projection column byte offset overflow")?;
    let mut bytes = Vec::with_capacity(
        full_rows
            .checked_mul(local_row_bytes)
            .context("sharded down projection byte count overflow")?,
    );
    let compact_started = Instant::now();
    for row in full.bytes.chunks_exact(full_row_bytes) {
        bytes.extend_from_slice(&row[column_start_bytes..column_start_bytes + local_row_bytes]);
    }
    let mut sharded_info = full.info;
    if let Some(width) = sharded_info.shape.get_mut(1) {
        *width = local_width;
    }
    sharded_info.byte_length = bytes.len() as u64;
    Ok(LoadedTensorRows {
        info: sharded_info,
        source_path: full.source_path,
        start_row: 0,
        row_count: full_rows,
        row_width: local_width,
        bytes_per_scalar: full.bytes_per_scalar,
        bytes,
        elapsed_micros: full.elapsed_micros + compact_started.elapsed().as_micros(),
        sha256: String::new(),
    })
}

fn load_routed_quant_scalar_metadata(
    catalog: &TensorCatalog,
    layer_id: usize,
    expert_id: usize,
    projection: &'static str,
) -> Result<RoutedQuantScalarMetadata> {
    let base_name = routed_quant_projection_base_name(layer_id, expert_id, projection);
    let loaded_input_scale = load_tensor_bytes(catalog, &format!("{base_name}.input_scale"))?;
    let loaded_weight_scale_2 = load_tensor_bytes(catalog, &format!("{base_name}.weight_scale_2"))?;
    routed_quant_scalar_metadata_from_loaded(&loaded_input_scale, &loaded_weight_scale_2)
}

fn routed_quant_scalar_metadata_from_loaded(
    input_scale: &LoadedTensor,
    weight_scale_2: &LoadedTensor,
) -> Result<RoutedQuantScalarMetadata> {
    if input_scale.info.dtype != DType::F32 || weight_scale_2.info.dtype != DType::F32 {
        anyhow::bail!(
            "real full NVFP4 expert probe expects F32 scalar metadata for {} and {}, got {:?} and {:?}",
            input_scale.info.name,
            weight_scale_2.info.name,
            input_scale.info.dtype,
            weight_scale_2.info.dtype
        );
    }
    if !input_scale.info.shape.is_empty() || !weight_scale_2.info.shape.is_empty() {
        anyhow::bail!(
            "real full NVFP4 expert probe expected scalar metadata for {} and {}, got {:?} and {:?}",
            input_scale.info.name,
            weight_scale_2.info.name,
            input_scale.info.shape,
            weight_scale_2.info.shape
        );
    }
    let input_scale_value = first_f32_scalar(&input_scale.info.name, &input_scale.bytes)?;
    let weight_scale_2_value = first_f32_scalar(&weight_scale_2.info.name, &weight_scale_2.bytes)?;
    validate_finite_route_scalar(&input_scale.info.name, input_scale_value)?;
    validate_finite_route_scalar(&weight_scale_2.info.name, weight_scale_2_value)?;
    Ok(RoutedQuantScalarMetadata {
        input_scale_name: input_scale.info.name.clone(),
        weight_scale_2_name: weight_scale_2.info.name.clone(),
        input_scale: input_scale_value,
        weight_scale_2: weight_scale_2_value,
    })
}

fn routed_quant_projection_base_name(
    layer_id: usize,
    expert_id: usize,
    projection: &str,
) -> String {
    format!("model.layers.{layer_id}.mlp.experts.{expert_id}.{projection}")
}

fn validate_routed_quant_projection_catalog(
    catalog: &TensorCatalog,
    layer_id: usize,
    expert_id: usize,
    projection: &'static str,
    row_count: usize,
    input_width: usize,
) -> Result<()> {
    let base_name = routed_quant_projection_base_name(layer_id, expert_id, projection);
    let weight_name = format!("{base_name}.weight");
    let weight_scale_name = format!("{base_name}.weight_scale");
    let input_scale_name = format!("{base_name}.input_scale");
    let weight_scale_2_name = format!("{base_name}.weight_scale_2");
    let weight = catalog_tensor(catalog, &weight_name)?;
    let weight_scale = catalog_tensor(catalog, &weight_scale_name)?;
    let input_scale = catalog_tensor(catalog, &input_scale_name)?;
    let weight_scale_2 = catalog_tensor(catalog, &weight_scale_2_name)?;

    if weight.dtype != DType::U8
        || weight_scale.dtype != DType::F8E4M3
        || input_scale.dtype != DType::F32
        || weight_scale_2.dtype != DType::F32
    {
        anyhow::bail!(
            "real full NVFP4 expert probe expects U8 weight, F8E4M3 scale, and F32 scalar metadata for {base_name}, got {:?}, {:?}, {:?}, {:?}",
            weight.dtype,
            weight_scale.dtype,
            input_scale.dtype,
            weight_scale_2.dtype
        );
    }
    if !input_scale.shape.is_empty() || !weight_scale_2.shape.is_empty() {
        anyhow::bail!(
            "real full NVFP4 expert probe expected scalar metadata for {base_name}, got {:?} and {:?}",
            input_scale.shape,
            weight_scale_2.shape
        );
    }

    let (weight_row_width, _, _) = tensor_rows_read_plan(weight, &weight_name, row_count)?;
    let (scale_row_width, _, _) =
        tensor_rows_read_plan(weight_scale, &weight_scale_name, row_count)?;
    validate_packed_nvfp4_projection_shape(
        projection,
        weight_row_width,
        scale_row_width,
        input_width,
    )
}

fn validate_finite_route_scalar(name: &str, value: f32) -> Result<()> {
    if !value.is_finite() {
        anyhow::bail!("real full NVFP4 BF16 route scalar {name} is non-finite");
    }
    Ok(())
}

fn catalog_tensor<'a>(catalog: &'a TensorCatalog, tensor_name: &str) -> Result<&'a TensorInfo> {
    catalog
        .tensors
        .binary_search_by(|tensor| tensor.name.as_str().cmp(tensor_name))
        .ok()
        .and_then(|index| catalog.tensors.get(index))
        .or_else(|| {
            catalog
                .tensors
                .iter()
                .find(|tensor| tensor.name == tensor_name)
        })
        .with_context(|| format!("tensor {tensor_name} not found in catalog"))
}

fn tensor_rows_read_plan(
    info: &TensorInfo,
    tensor_name: &str,
    row_count: usize,
) -> Result<(usize, usize, usize)> {
    if row_count == 0 {
        anyhow::bail!("row window for tensor {tensor_name} must contain at least one row");
    }
    if info.shape.len() != 2 {
        anyhow::bail!("tensor {tensor_name} is not rank-2, shape={:?}", info.shape);
    }
    let rows = info.shape[0];
    let row_width = info.shape[1];
    if row_count > rows {
        anyhow::bail!("row window [0, {row_count}) exceeds tensor {tensor_name} row count {rows}");
    }
    let bytes_per_scalar = dtype_byte_width(&info.dtype)?;
    let row_bytes = row_width
        .checked_mul(bytes_per_scalar)
        .context("row byte width overflow")?;
    let bytes_to_read = row_count
        .checked_mul(row_bytes)
        .context("row byte length overflow")?;
    let tensor_byte_length: usize = info
        .byte_length
        .try_into()
        .context("tensor byte length does not fit in usize")?;
    if bytes_to_read > tensor_byte_length {
        anyhow::bail!(
            "row window for tensor {tensor_name} exceeds recorded byte length {}",
            info.byte_length
        );
    }
    Ok((row_width, bytes_per_scalar, bytes_to_read))
}

fn validate_packed_nvfp4_projection_width(
    projection: &str,
    weight: &LoadedTensorRows,
    weight_scale: &LoadedTensorRows,
    input_width: usize,
) -> Result<()> {
    validate_packed_nvfp4_projection_shape(
        projection,
        weight.row_width,
        weight_scale.row_width,
        input_width,
    )
}

fn validate_packed_nvfp4_projection_shape(
    projection: &str,
    packed_row_width: usize,
    scale_row_width: usize,
    input_width: usize,
) -> Result<()> {
    let required_packed_width = input_width.div_ceil(2);
    let required_scale_width = input_width.div_ceil(16);
    if packed_row_width < required_packed_width {
        anyhow::bail!(
            "real full NVFP4 expert {projection} packed row width {packed_row_width} cannot cover input width {input_width}"
        );
    }
    if scale_row_width < required_scale_width {
        anyhow::bail!(
            "real full NVFP4 expert {projection} scale row width {scale_row_width} cannot cover input width {input_width}"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        b12x_projection_scale_shape_supported, b12x_spark_direct_route_shape_supported,
        b12x_w4a16_capacity_rows, b12x_w4a16_prefill_route_block_rows,
        canonical_spark_collective_request_id, coalesce_streaming_completion_slices,
        collective_gap_ready, copy_loaded_route_cuda_tensor_compact,
        cuda_reference_kernels_enabled, cuda_reference_kernels_test_override,
        cuda_route_validation_test_override, device_buffer_byte_view,
        ds4_flash_spark_decode_m1_preloaded_device_input_into,
        ds4_flash_spark_prefill_pack_topk6_device_into,
        ds4_flash_spark_prefill_topk6_preloaded_device_input_into, ds4_flash_w4a16_capacity_rows,
        execute_exl3_read_requests, execute_native_flash_route_bf16_cached,
        execute_nvfp4_route_cached, execute_nvfp4_route_rows_bf16_accumulated_cached,
        execute_nvfp4_route_rows_bf16_accumulated_cached_device_input_device_output,
        execute_nvfp4_route_rows_bf16_accumulated_cached_device_output, exl3_trellis_tile_bytes,
        f32_values_to_bf16_bytes, fused_fp8_reduction_eligible, load_bf16_route_projection_source,
        load_bf16_route_projections_for_group_cached, load_routed_projection_rows_for_shard,
        native_library_path, native_library_path_candidates, packed_w4a16_topk8_prefill_eligible,
        plan_ds4_flash_packed_topk6_routes, plan_packed_w4a16_topk8_prefill,
        plan_packed_w4a16_topk8_prefill_flat, preload_native_flash_layer_cuda_cache_local_tp4,
        preload_native_flash_projection_cuda_cache_local_tp4, queue_exl3_trellis_tp4,
        read_exl3_trellis_tp4, route_cuda_graphs_test_override,
        routed_quant_scalar_metadata_from_loaded, should_use_grouped_route_launches,
        sparkinfer_exl3_mode_from_value, sqg_xor_cheb_t12_lut_bytes, Bf16RouteProjectionGroupKey,
        Bf16RouteProjections, Ds4Exl3AotProfile, Ds4FlashSparkDecodeM1DeviceBuffers,
        Ds4FlashSparkPrefillDeviceBuffers, Ds4FlashSparkPrefillRoutePackDeviceBuffers,
        LoadedRouteCudaTensorRows, LoadedTensor, LoadedTensorRows, OwnedDeviceAllocation,
        PackedW4a16Topk8Route, RouteCudaCache, RouteCudaDs4FlashExl3LayerExpertSlab,
        RouteCudaEvent, RouteCudaStream, RouteCudaTensorBytes, RouteCudaWorkspace,
        RouteStreamingOutputDtype, RouteTensorCache, RoutedQuantProjection,
        RoutedQuantProjectionKey, ScoredRoute, SparkCollectiveLaunchOrder, SparkinferExl3Mode,
        CPU_REFERENCE_NVFP4_ROUTE_BACKEND, CUDA_REFERENCE_NVFP4_ROUTE_BF16_ACCUMULATED_BACKEND,
        CUDA_REFERENCE_NVFP4_ROUTE_BF16_ACCUMULATED_DEVICE_INPUT_BACKEND,
        CUDA_REFERENCE_NVFP4_ROUTE_BF16_ACCUMULATED_DEVICE_OUTPUT_BACKEND,
        DS4_FLASH_DECODE_M1_ACTIVATED_BYTES, DS4_FLASH_DECODE_M1_FC1_BYTES,
        DS4_FLASH_DECODE_M1_FC1_SCRATCH_BYTES, DS4_FLASH_DECODE_M1_FC2_SCRATCH_BYTES,
        DS4_FLASH_DECODE_M1_HIDDEN_BYTES, DS4_FLASH_DECODE_M1_LOCK_BYTES,
        DS4_FLASH_DECODE_M1_OUTPUT_F32_BYTES, DS4_FLASH_DECODE_M1_ROUTED_OUTPUT_BYTES,
        DS4_FLASH_DECODE_M1_ROUTE_BYTES, DS4_FLASH_EXL3_MCG_MARKER,
        DS4_FLASH_EXL3_TRELLIS_EXPERT_BYTES, DS4_FLASH_HIDDEN_SIZE,
        DS4_FLASH_LOCAL_INTERMEDIATE_SIZE, DS4_FLASH_PREFILL_ACTIVATED_BYTES,
        DS4_FLASH_PREFILL_BLOCK_EXPERT_BYTES, DS4_FLASH_PREFILL_COMPUTE_SCRATCH_BYTES,
        DS4_FLASH_PREFILL_EXPERT_COUNT_BYTES, DS4_FLASH_PREFILL_EXPERT_OFFSET_BYTES,
        DS4_FLASH_PREFILL_FC1_BYTES, DS4_FLASH_PREFILL_HIDDEN_BYTES,
        DS4_FLASH_PREFILL_OUTPUT_F32_BYTES, DS4_FLASH_PREFILL_PACKED_ROUTE_BYTES,
        DS4_FLASH_PREFILL_ROUTED_OUTPUT_BYTES, DS4_FLASH_PREFILL_ROUTE_BLOCK_ROWS,
        DS4_FLASH_PREFILL_SCRATCH_ELEMENTS, DS4_FLASH_PREFILL_TOPK_ID_BYTES,
        DS4_FLASH_PREFILL_TOPK_WEIGHT_BYTES, DS4_FLASH_ROUTED_EXPERTS, DS4_FLASH_TOP_K,
        DS4_PRO_DECODE_PACKED_ROUTE_SLOTS, DS4_PRO_DECODE_ROUTE_BLOCKS,
        DS4_PRO_FC1_SCRATCH_ELEMENTS, DS4_PRO_FC2_SCRATCH_ELEMENTS, DS4_PRO_HIDDEN_SIZE,
        DS4_PRO_LOCAL_INTERMEDIATE_SIZE, DS4_PRO_PREFILL_MAX_PACKED_ROUTE_SLOTS,
        DS4_PRO_PREFILL_MAX_ROUTE_BLOCKS, DS4_PRO_PREFILL_SCRATCH_ELEMENTS, DS4_PRO_ROUTED_EXPERTS,
        DS4_PRO_SPARK_EXL3_AOT_BF16_BACKEND,
    };

    #[test]
    fn sparkinfer_exl3_mode_is_explicit_and_fail_closed() {
        assert_eq!(
            sparkinfer_exl3_mode_from_value(None).unwrap(),
            SparkinferExl3Mode::Auto
        );
        assert_eq!(
            sparkinfer_exl3_mode_from_value(Some("auto")).unwrap(),
            SparkinferExl3Mode::Auto
        );
        assert_eq!(
            sparkinfer_exl3_mode_from_value(Some("disable")).unwrap(),
            SparkinferExl3Mode::Disable
        );
        assert_eq!(
            sparkinfer_exl3_mode_from_value(Some("force")).unwrap(),
            SparkinferExl3Mode::Force
        );
        assert!(sparkinfer_exl3_mode_from_value(Some("fallback")).is_err());
    }

    #[test]
    fn exl3_aot_profiles_pin_exact_flash_and_pro_geometry() {
        let catalog = |variant, hidden_size, moe_intermediate_size, routed_experts| {
            let mut facts = ModelFacts::default();
            facts.variant = variant;
            facts.hidden_size = hidden_size;
            facts.moe_intermediate_size = moe_intermediate_size;
            facts.routed_experts = routed_experts;
            facts.top_k = DS4_FLASH_TOP_K;
            facts.quantization_recipe = DEEPSEEK_V4_EXL3_RECIPE.to_owned();
            TensorCatalog {
                model_id: "deepseek-ai/test-exl3-profile".to_owned(),
                snapshot_path: "/tmp/test-exl3-profile".to_owned(),
                facts,
                tensors: Vec::new(),
            }
        };

        let flash = Ds4Exl3AotProfile::from_catalog(&catalog(
            ModelVariant::Flash,
            DS4_FLASH_HIDDEN_SIZE,
            DS4_FLASH_LOCAL_INTERMEDIATE_SIZE * 4,
            DS4_FLASH_ROUTED_EXPERTS,
        ))
        .unwrap();
        assert_eq!(flash.variant, ModelVariant::Flash);
        assert_eq!(flash.decode_packed_route_slots, DS4_FLASH_TOP_K);
        assert_eq!(flash.decode_route_blocks, 1);
        assert_eq!(
            flash.prefill_fc1_scratch_elements,
            DS4_FLASH_PREFILL_SCRATCH_ELEMENTS
        );
        assert_eq!(
            flash.prefill_fc2_scratch_elements,
            DS4_FLASH_PREFILL_SCRATCH_ELEMENTS
        );

        let pro = Ds4Exl3AotProfile::from_catalog(&catalog(
            ModelVariant::Pro,
            DS4_PRO_HIDDEN_SIZE,
            DS4_PRO_LOCAL_INTERMEDIATE_SIZE * 4,
            DS4_PRO_ROUTED_EXPERTS,
        ))
        .unwrap();
        assert_eq!(pro.variant, ModelVariant::Pro);
        assert_eq!(pro.geometry.hidden_size, DS4_PRO_HIDDEN_SIZE);
        assert_eq!(
            pro.geometry.local_intermediate_size,
            DS4_PRO_LOCAL_INTERMEDIATE_SIZE
        );
        assert_eq!(
            pro.prefill_max_packed_route_slots,
            DS4_PRO_PREFILL_MAX_PACKED_ROUTE_SLOTS
        );
        assert_eq!(
            pro.prefill_max_route_blocks,
            DS4_PRO_PREFILL_MAX_ROUTE_BLOCKS
        );
        assert_eq!(
            pro.decode_packed_route_slots,
            DS4_PRO_DECODE_PACKED_ROUTE_SLOTS
        );
        assert_eq!(pro.decode_route_blocks, DS4_PRO_DECODE_ROUTE_BLOCKS);
        assert_eq!(
            pro.prefill_fc1_scratch_elements,
            DS4_PRO_PREFILL_SCRATCH_ELEMENTS
        );
        assert_eq!(
            pro.prefill_fc2_scratch_elements,
            DS4_PRO_PREFILL_SCRATCH_ELEMENTS
        );
        assert_eq!(
            pro.decode_fc1_scratch_elements,
            DS4_PRO_FC1_SCRATCH_ELEMENTS
        );
        assert_eq!(
            pro.decode_fc2_scratch_elements,
            DS4_PRO_FC2_SCRATCH_ELEMENTS
        );

        assert!(Ds4Exl3AotProfile::from_catalog(&catalog(
            ModelVariant::Pro,
            DS4_PRO_HIDDEN_SIZE,
            DS4_PRO_LOCAL_INTERMEDIATE_SIZE * 4,
            DS4_PRO_ROUTED_EXPERTS - 1,
        ))
        .is_err());
    }
    use crate::commands::real_full::coordinator_kernels::{
        device_bf16_output_from_bf16_bytes, device_bf16_output_uninitialized,
        require_cuda_enabled_native_library,
    };
    use crate::commands::real_full::intermediate_sharding::ExpertIntermediateShard;
    use anyhow::{Context, Result};
    use ds41rt_core::{
        plan_completion_first_routes, CompletionRoutePlanEntry, DType, ModelFacts, ModelVariant,
        TensorCatalog, TensorInfo, TensorRole,
    };
    use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary};
    use ds41rt_loader::{
        build_catalog_for_snapshot, exl3_expert, Exl3Projection, Exl3ProjectionKind,
        DEEPSEEK_V4_EXL3_RECIPE,
    };
    use sha2::{Digest, Sha256};
    use std::{collections::HashMap, fs::File, io::Write, time::Duration};
    use std::{path::Path, path::PathBuf, sync::Arc};

    fn synthetic_exl3_single_expert_catalog(
        root: &Path,
        variant: ModelVariant,
        hidden_size: usize,
        global_intermediate_size: usize,
        routed_experts: usize,
    ) -> Result<TensorCatalog> {
        let source_path = root.join("model.raw");
        let mut source = File::create(&source_path)?;
        let mut byte_offset = 0_u64;
        let mut tensors = Vec::new();
        for (stem, input_features, output_features, seed) in [
            ("w1", hidden_size, global_intermediate_size, 11_u8),
            ("w2", global_intermediate_size, hidden_size, 73_u8),
            ("w3", hidden_size, global_intermediate_size, 149_u8),
        ] {
            let trellis = (0..input_features * output_features / 4)
                .map(|index| seed.wrapping_add(((index / 64) % 101) as u8))
                .collect::<Vec<_>>();
            let suh = std::iter::repeat_n([0x80_u8, 0x3f], input_features)
                .flatten()
                .collect::<Vec<_>>();
            let svh = std::iter::repeat_n([0x80_u8, 0x3f], output_features)
                .flatten()
                .collect::<Vec<_>>();
            for (suffix, dtype, shape, payload, metadata) in [
                (
                    "trellis",
                    DType::I16,
                    vec![input_features / 16, output_features / 16, 32],
                    trellis,
                    false,
                ),
                ("suh", DType::F16, vec![input_features], suh, true),
                ("svh", DType::F16, vec![output_features], svh, true),
                (
                    "mcg",
                    DType::I32,
                    vec![],
                    DS4_FLASH_EXL3_MCG_MARKER.to_le_bytes().to_vec(),
                    true,
                ),
            ] {
                source.write_all(&payload)?;
                tensors.push(TensorInfo {
                    name: format!("layers.0.ffn.experts.0.{stem}.{suffix}"),
                    file: "model.raw".to_owned(),
                    dtype,
                    shape,
                    byte_offset,
                    byte_length: payload.len() as u64,
                    role: TensorRole::RoutedExpert,
                    layer_id: Some(0),
                    expert_id: Some(0),
                    is_quantization_metadata: metadata,
                });
                byte_offset += payload.len() as u64;
            }
        }
        drop(source);
        tensors.sort_by(|left, right| left.name.cmp(&right.name));
        let mut facts = ModelFacts::default();
        facts.variant = variant;
        facts.num_hidden_layers = 1;
        facts.dspark_target_layer_ids.clear();
        facts.hidden_size = hidden_size;
        facts.moe_intermediate_size = global_intermediate_size;
        facts.routed_experts = routed_experts;
        facts.top_k = DS4_FLASH_TOP_K;
        facts.quantization_recipe = DEEPSEEK_V4_EXL3_RECIPE.to_owned();
        Ok(TensorCatalog {
            model_id: format!("deepseek-ai/test-{variant:?}-exl3-direct-resident"),
            snapshot_path: root.display().to_string(),
            facts,
            tensors,
        })
    }

    #[test]
    fn native_library_candidates_only_include_cuda_builds() {
        let candidates = native_library_path_candidates();

        assert!(!candidates.is_empty());
        assert!(candidates
            .iter()
            .all(|path| path.to_string_lossy().contains("native/build-cuda/")));
    }

    #[test]
    fn exl3_t12_lut_matches_the_pinned_b12x_table() {
        let lut = sqg_xor_cheb_t12_lut_bytes();
        assert_eq!(lut.len(), 4096);
        assert_eq!(lut.first(), Some(&203));
        assert_eq!(lut.last(), Some(&75));
        assert_eq!(
            format!("{:x}", Sha256::digest(lut)),
            "cca11fe5744c9c93a34f4217f342fbc0f74ecc8a007c076582424a505fc9da5e"
        );
    }

    #[test]
    fn exl3_trellis_tile_bytes_follow_uniform_tier() -> Result<()> {
        assert_eq!(exl3_trellis_tile_bytes(2)?, 64);
        assert_eq!(exl3_trellis_tile_bytes(3)?, 96);
        assert!(exl3_trellis_tile_bytes(4).is_err());
        Ok(())
    }

    #[test]
    fn exl3_direct_preload_reads_only_rank_local_trellis_tiles() -> Result<()> {
        const TILE_BYTES: usize = 32 * std::mem::size_of::<i16>();
        let tempdir = tempfile::tempdir()?;
        let source_path = tempdir.path().join("trellis.bin");
        let gate_bytes = (0..(4_096 / 16) * (2_048 / 16) * TILE_BYTES)
            .map(|offset| ((offset / TILE_BYTES) % 251) as u8)
            .collect::<Vec<_>>();
        let down_bytes = (0..(2_048 / 16) * (4_096 / 16) * TILE_BYTES)
            .map(|offset| (193_u8).wrapping_add(((offset / TILE_BYTES) % 59) as u8))
            .collect::<Vec<_>>();
        let mut source = Vec::with_capacity(gate_bytes.len() + down_bytes.len());
        source.extend_from_slice(&gate_bytes);
        source.extend_from_slice(&down_bytes);
        std::fs::write(&source_path, source)?;
        let tensor = |name: &str, byte_offset: usize, shape: Vec<usize>, bytes: usize| TensorInfo {
            name: name.to_owned(),
            file: "trellis.bin".to_owned(),
            dtype: DType::I16,
            shape,
            byte_offset: byte_offset as u64,
            byte_length: bytes as u64,
            role: TensorRole::RoutedExpert,
            layer_id: Some(0),
            expert_id: Some(0),
            is_quantization_metadata: false,
        };
        let gate = tensor(
            "gate.trellis",
            0,
            vec![4_096 / 16, 2_048 / 16, 32],
            gate_bytes.len(),
        );
        let down = tensor(
            "down.trellis",
            gate_bytes.len(),
            vec![2_048 / 16, 4_096 / 16, 32],
            down_bytes.len(),
        );
        let shard = ExpertIntermediateShard { rank: 2, count: 4 };
        let mut files = HashMap::new();
        let mut compact_gate = vec![0_u8; DS4_FLASH_EXL3_TRELLIS_EXPERT_BYTES];
        let mut requests = Vec::new();
        queue_exl3_trellis_tp4(
            tempdir.path(),
            Exl3Projection {
                kind: Exl3ProjectionKind::Gate,
                trellis: &gate,
                suh: &gate,
                svh: &gate,
                mcg: &gate,
                input_features: 4_096,
                output_features: 2_048,
            },
            shard,
            DS4_FLASH_HIDDEN_SIZE,
            DS4_FLASH_LOCAL_INTERMEDIATE_SIZE,
            2,
            DS4_FLASH_EXL3_TRELLIS_EXPERT_BYTES,
            &mut files,
            &mut compact_gate,
            &mut requests,
        )?;
        execute_exl3_read_requests(&requests)?;
        let full_row_bytes = (2_048 / 16) * TILE_BYTES;
        let local_row_bytes = (512 / 16) * TILE_BYTES;
        for input_tile in 0..4_096 / 16 {
            let source_start = input_tile * full_row_bytes + shard.rank * local_row_bytes;
            let compact_start = input_tile * local_row_bytes;
            assert_eq!(
                &compact_gate[compact_start..compact_start + local_row_bytes],
                &gate_bytes[source_start..source_start + local_row_bytes]
            );
        }

        let mut compact_down = vec![0_u8; DS4_FLASH_EXL3_TRELLIS_EXPERT_BYTES];
        read_exl3_trellis_tp4(
            tempdir.path(),
            Exl3Projection {
                kind: Exl3ProjectionKind::Down,
                trellis: &down,
                suh: &down,
                svh: &down,
                mcg: &down,
                input_features: 2_048,
                output_features: 4_096,
            },
            shard,
            DS4_FLASH_HIDDEN_SIZE,
            DS4_FLASH_LOCAL_INTERMEDIATE_SIZE,
            2,
            DS4_FLASH_EXL3_TRELLIS_EXPERT_BYTES,
            &mut files,
            &mut compact_down,
        )?;
        assert_eq!(
            compact_down,
            down_bytes[shard.rank * DS4_FLASH_EXL3_TRELLIS_EXPERT_BYTES
                ..(shard.rank + 1) * DS4_FLASH_EXL3_TRELLIS_EXPERT_BYTES]
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA native library and physical GPU 0"]
    fn exl3_direct_preload_writes_final_resident_trellis_planes_without_device_repack() -> Result<()>
    {
        let tempdir = tempfile::tempdir()?;
        let source_path = tempdir.path().join("model.raw");
        let mut source = File::create(&source_path)?;
        let mut byte_offset = 0_u64;
        let mut tensors = Vec::new();
        for (stem, input_features, output_features, seed) in [
            ("w1", 4_096, 2_048, 11_u8),
            ("w2", 2_048, 4_096, 73_u8),
            ("w3", 4_096, 2_048, 149_u8),
        ] {
            for (suffix, dtype, shape, payload, metadata) in [
                (
                    "trellis",
                    DType::I16,
                    vec![input_features / 16, output_features / 16, 32],
                    (0..input_features * output_features / 4)
                        .map(|index| seed.wrapping_add(((index / 64) % 101) as u8))
                        .collect::<Vec<_>>(),
                    false,
                ),
                (
                    "suh",
                    DType::F16,
                    vec![input_features],
                    (0..input_features * 2)
                        .map(|index| seed.wrapping_add((index % 37) as u8))
                        .collect::<Vec<_>>(),
                    true,
                ),
                (
                    "svh",
                    DType::F16,
                    vec![output_features],
                    (0..output_features * 2)
                        .map(|index| seed.wrapping_add((index % 43) as u8))
                        .collect::<Vec<_>>(),
                    true,
                ),
                (
                    "mcg",
                    DType::I32,
                    vec![],
                    DS4_FLASH_EXL3_MCG_MARKER.to_le_bytes().to_vec(),
                    true,
                ),
            ] {
                let name = format!("layers.0.ffn.experts.0.{stem}.{suffix}");
                source.write_all(&payload)?;
                tensors.push(TensorInfo {
                    name,
                    file: "model.raw".to_owned(),
                    dtype,
                    shape,
                    byte_offset,
                    byte_length: payload.len() as u64,
                    role: TensorRole::RoutedExpert,
                    layer_id: Some(0),
                    expert_id: Some(0),
                    is_quantization_metadata: metadata,
                });
                byte_offset += payload.len() as u64;
            }
        }
        drop(source);
        tensors.sort_by(|left, right| left.name.cmp(&right.name));
        let mut facts = ModelFacts::default();
        facts.num_hidden_layers = 1;
        facts.dspark_target_layer_ids.clear();
        facts.quantization_recipe = DEEPSEEK_V4_EXL3_RECIPE.to_owned();
        let catalog = TensorCatalog {
            model_id: "deepseek-ai/test-flash-exl3-direct-resident".to_owned(),
            snapshot_path: tempdir.path().display().to_string(),
            facts,
            tensors,
        };
        let path = native_library_path().context("CUDA native library is unavailable")?;
        let library = Arc::new(unsafe { NativeLibrary::load(&path) }?);
        require_cuda_enabled_native_library(
            library.as_ref(),
            &path,
            "EXL3 direct resident preload",
        )?;
        let stream = RouteCudaStream::new(Arc::clone(&library))?;
        let shard = ExpertIntermediateShard::new(4, 1)?;
        let slab = RouteCudaDs4FlashExl3LayerExpertSlab::new(Arc::clone(&library), &catalog, 0)?;
        let mut workspace = RouteCudaWorkspace::default();
        slab.store_layer_experts_direct(
            &catalog,
            &[0],
            shard,
            Arc::clone(&library),
            &mut workspace,
            stream.as_ptr(),
        )?;

        let expert = exl3_expert(&catalog, 0, 0)?;
        let mut files = HashMap::new();
        for (label, projection, resident, resident_offset) in [
            ("gate", expert.gate, slab.w13_trellis.buffer(), 0),
            (
                "up",
                expert.up,
                slab.w13_trellis.buffer(),
                DS4_FLASH_ROUTED_EXPERTS * DS4_FLASH_EXL3_TRELLIS_EXPERT_BYTES,
            ),
            ("down", expert.down, slab.w2_trellis.buffer(), 0),
        ] {
            let mut expected = vec![0_u8; DS4_FLASH_EXL3_TRELLIS_EXPERT_BYTES];
            read_exl3_trellis_tp4(
                tempdir.path(),
                projection,
                shard,
                DS4_FLASH_HIDDEN_SIZE,
                DS4_FLASH_LOCAL_INTERMEDIATE_SIZE,
                2,
                DS4_FLASH_EXL3_TRELLIS_EXPERT_BYTES,
                &mut files,
                &mut expected,
            )?;
            let view = device_buffer_byte_view(
                resident,
                resident_offset,
                expected.len(),
                "EXL3 resident trellis test view",
            )?;
            let mut actual = vec![0_u8; expected.len()];
            library.copy_d2h(&mut actual, view)?;
            assert_eq!(actual, expected, "{label} resident TP4 trellis differs");
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA native library and physical GPU 0"]
    fn exl3_pro_direct_resident_executes_decode_and_prefill_through_rust() -> Result<()> {
        let tempdir = tempfile::tempdir()?;
        let catalog = synthetic_exl3_single_expert_catalog(
            tempdir.path(),
            ModelVariant::Pro,
            DS4_PRO_HIDDEN_SIZE,
            DS4_PRO_LOCAL_INTERMEDIATE_SIZE * 4,
            DS4_PRO_ROUTED_EXPERTS,
        )?;
        let shard = ExpertIntermediateShard::new(4, 1)?;
        let mut cache = RouteTensorCache::default();
        {
            let cuda_cache = cache.cuda_cache_native_flash_tp4(shard)?;
            cuda_cache.ensure_ds4_exl3_aot(ModelVariant::Pro)?;
            let stream = RouteCudaStream::new(Arc::clone(&cuda_cache.library))?;
            let slab = Arc::new(RouteCudaDs4FlashExl3LayerExpertSlab::new(
                Arc::clone(&cuda_cache.library),
                &catalog,
                0,
            )?);
            slab.store_layer_experts_direct(
                &catalog,
                &[0],
                shard,
                Arc::clone(&cuda_cache.library),
                &mut cuda_cache.workspace,
                stream.as_ptr(),
            )?;
            assert_eq!(
                slab.resident_weight_bytes(),
                3 * DS4_PRO_ROUTED_EXPERTS as u64 * 1_376_256
            );
            assert!(cuda_cache
                .ds4_flash_exl3_expert_slabs
                .insert(0, slab)
                .is_none());
        }

        for rows in [1_usize, 16] {
            let hidden_values = (0..rows * DS4_PRO_HIDDEN_SIZE)
                .map(|index| ((index * 17 % 257) as f32 - 128.0) / 4096.0)
                .collect::<Vec<_>>();
            let mut hidden_bf16 =
                vec![0_u8; rows * DS4_PRO_HIDDEN_SIZE * std::mem::size_of::<u16>()];
            f32_values_to_bf16_bytes(&hidden_values, &mut hidden_bf16);
            let route_ids = vec![0_u32; rows * DS4_FLASH_TOP_K];
            let route_weights = vec![1.0_f32 / DS4_FLASH_TOP_K as f32; rows * DS4_FLASH_TOP_K];
            let execution = execute_native_flash_route_bf16_cached(
                rows as u64,
                0,
                rows,
                &hidden_bf16,
                None,
                &route_ids,
                &route_weights,
                false,
                false,
                None,
                &mut cache,
            )?;
            assert_eq!(
                execution.kernel_backend,
                DS4_PRO_SPARK_EXL3_AOT_BF16_BACKEND
            );
            assert_eq!(execution.completed_rows, (0..rows).collect::<Vec<_>>());
            assert_eq!(execution.output_bf16.len(), hidden_bf16.len());
            assert!(execution.output_bf16.chunks_exact(2).all(|bytes| {
                f32::from_bits(u32::from(u16::from_ne_bytes([bytes[0], bytes[1]])) << 16)
                    .is_finite()
            }));
        }
        Ok(())
    }

    #[test]
    fn compact_route_tensor_copy_preserves_contiguous_rows() -> Result<()> {
        let tensor = LoadedRouteCudaTensorRows {
            info: fake_tensor_info("weight", DType::U8, vec![2, 4], 8),
            source_path: PathBuf::from("fake-route.bin"),
            start_row: 0,
            row_count: 2,
            row_width: 4,
            source_row_width: 4,
            source_column_start: 0,
            bytes_per_scalar: 1,
            bytes: RouteCudaTensorBytes::owned(vec![0, 1, 2, 3, 4, 5, 6, 7]),
            elapsed_micros: 0,
        };
        let mut compact = vec![0_u8; 8];

        copy_loaded_route_cuda_tensor_compact(&tensor, &mut compact)?;

        assert_eq!(compact, [0, 1, 2, 3, 4, 5, 6, 7]);
        Ok(())
    }

    #[test]
    fn compact_route_tensor_copy_extracts_strided_column_window() -> Result<()> {
        let tensor = LoadedRouteCudaTensorRows {
            info: fake_tensor_info("weight", DType::U8, vec![2, 3], 6),
            source_path: PathBuf::from("fake-route.bin"),
            start_row: 0,
            row_count: 2,
            row_width: 3,
            source_row_width: 6,
            source_column_start: 2,
            bytes_per_scalar: 1,
            bytes: RouteCudaTensorBytes::owned(vec![0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15]),
            elapsed_micros: 0,
        };
        let mut compact = vec![0_u8; 6];

        copy_loaded_route_cuda_tensor_compact(&tensor, &mut compact)?;

        assert_eq!(compact, [2, 3, 4, 12, 13, 14]);
        Ok(())
    }

    #[test]
    fn single_row_multi_expert_route_uses_grouped_launches() {
        assert!(should_use_grouped_route_launches(1, 2));
        assert!(should_use_grouped_route_launches(1, 1));
    }

    #[test]
    fn spark_collective_launch_order_sorts_pending_requests_and_resets_on_gap() -> Result<()> {
        let order = Arc::new(SparkCollectiveLaunchOrder::new(Duration::from_millis(1)));
        let mut later = order.register(65_539)?;
        let mut earlier = order.register(3)?;
        assert_eq!(later.request_id, 65_536);
        assert_eq!(earlier.request_id, 0);
        earlier.wait_for_turn()?;
        earlier.finish()?;
        later.wait_for_turn()?;
        later.finish()?;

        let mut next_epoch_later = order.register(1_065_538)?;
        let mut next_epoch_earlier = order.register(1_000_002)?;
        next_epoch_earlier.wait_for_turn()?;
        next_epoch_earlier.finish()?;
        next_epoch_later.wait_for_turn()?;
        next_epoch_later.finish()?;
        Ok(())
    }

    #[test]
    fn spark_collective_request_id_is_shared_by_all_host_requests() {
        for host_offset in 0..4 {
            assert_eq!(
                canonical_spark_collective_request_id(0x8000_0000_0002_0000 + host_offset),
                0x8000_0000_0002_0000
            );
        }
    }

    #[test]
    fn spark_collective_gap_waits_for_every_execution_lane() {
        let wait = Duration::from_millis(20);

        assert!(!collective_gap_ready(2, 4, Duration::from_millis(1), wait));
        assert!(collective_gap_ready(4, 4, Duration::from_millis(1), wait));
        assert!(collective_gap_ready(1, 4, wait, wait));
    }

    #[test]
    fn spark_collective_explicit_partition_quorum_orders_opposite_arrivals() -> Result<()> {
        let rank0 = Arc::new(SparkCollectiveLaunchOrder::new(Duration::from_millis(20)));
        let rank1 = Arc::new(SparkCollectiveLaunchOrder::new(Duration::from_millis(20)));
        let mut rank0_later = rank0.register(65_536)?;
        let mut rank0_earlier = rank0.register(0)?;
        let mut rank1_earlier = rank1.register(0)?;
        let mut rank1_later = rank1.register(65_536)?;

        rank0_earlier.wait_for_turn_with_quorum(2, Duration::from_millis(100))?;
        rank1_earlier.wait_for_turn_with_quorum(2, Duration::from_millis(100))?;
        rank0_earlier.finish()?;
        rank1_earlier.finish()?;
        rank0_later.wait_for_turn_with_quorum(2, Duration::from_millis(100))?;
        rank1_later.wait_for_turn_with_quorum(2, Duration::from_millis(100))?;
        rank0_later.finish()?;
        rank1_later.finish()?;
        Ok(())
    }

    #[test]
    fn spark_collective_explicit_quorum_does_not_fall_through_after_reorder_wait() -> Result<()> {
        let order = Arc::new(SparkCollectiveLaunchOrder::new(Duration::from_millis(1)));
        let mut later = order.register(65_536)?;
        let (acquired_tx, acquired_rx) = std::sync::mpsc::channel();
        let waiter = std::thread::spawn(move || -> Result<()> {
            later.wait_for_turn_with_quorum(2, Duration::from_millis(200))?;
            acquired_tx.send(later.request_id)?;
            later.finish()
        });

        std::thread::sleep(Duration::from_millis(10));
        assert!(matches!(
            acquired_rx.try_recv(),
            Err(std::sync::mpsc::TryRecvError::Empty)
        ));

        let mut earlier = order.register(0)?;
        earlier.wait_for_turn_with_quorum(2, Duration::from_millis(200))?;
        earlier.finish()?;
        assert_eq!(
            acquired_rx.recv_timeout(Duration::from_millis(200))?,
            65_536
        );
        waiter.join().expect("collective quorum waiter panicked")?;
        Ok(())
    }

    #[test]
    fn packed_w4a16_topk8_prefill_plan_uses_direct_routes_for_one_row() -> Result<()> {
        let row_routes = (0..1)
            .map(|row| {
                (0..8)
                    .map(|slot| {
                        let route_index = row * 8 + slot;
                        (
                            ScoredRoute {
                                expert_id: slot % 2,
                                score: 0.0,
                                corrected_score: 0.0,
                                normalized_weight: route_index as f32 + 1.0,
                            },
                            512,
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let plan = plan_packed_w4a16_topk8_prefill(&row_routes)?;

        assert_eq!(plan.packed_route_count, 8);
        assert!(plan.block_expert_ids.is_empty());
        assert_eq!(
            plan.packed_route_indices,
            (0..8).map(|route| route % 2).collect::<Vec<_>>()
        );
        assert_eq!(plan.direct_topk_ids, plan.packed_route_indices);
        assert_eq!(
            plan.topk_weights,
            (1..=8).map(|value| value as f32).collect::<Vec<_>>()
        );
        Ok(())
    }

    #[test]
    fn native_flash_decode_plan_preserves_six_direct_routes() -> Result<()> {
        let ids = vec![19, 3, 255, 44, 0, 128];
        let weights = vec![0.1, 0.2, 0.05, 0.25, 0.15, 0.25];

        let plan = plan_ds4_flash_packed_topk6_routes(1, &ids, &weights)?;

        assert_eq!(plan.packed_route_indices, ids);
        assert_eq!(plan.packed_route_count, 6);
        assert!(plan.block_expert_ids.is_empty());
        assert_eq!(plan.topk_weights, weights);
        assert_eq!(ds4_flash_w4a16_capacity_rows(1)?, 1);
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn native_flash_prefill_route_pack_is_device_resident_and_graph_safe() {
        let path = native_library_path().expect("finding the CUDA native library");
        let library = Arc::new(unsafe { NativeLibrary::load(path) }.expect("loading CUDA native"));
        let rows = 37_usize;
        let route_count = rows * 6;
        let topk_ids = (0..route_count)
            .map(|route| ((route * 37 + route / 6 * 11) % 256) as u32)
            .collect::<Vec<_>>();
        let route_weights = vec![1.0_f32 / 6.0; route_count];
        let expected = plan_ds4_flash_packed_topk6_routes(rows, &topk_ids, &route_weights)
            .expect("planning the host route-pack oracle");

        let allocate = |bytes, label| {
            OwnedDeviceAllocation::new(Arc::clone(&library), bytes, label)
                .unwrap_or_else(|error| panic!("allocating {label}: {error:#}"))
        };
        let topk = allocate(
            DS4_FLASH_PREFILL_TOPK_ID_BYTES,
            "Flash route-pack top-k IDs",
        );
        let packed = allocate(
            DS4_FLASH_PREFILL_PACKED_ROUTE_BYTES,
            "Flash route-pack indices",
        );
        let block_experts = allocate(
            DS4_FLASH_PREFILL_BLOCK_EXPERT_BYTES,
            "Flash route-pack block experts",
        );
        let packed_count = allocate(std::mem::size_of::<i32>(), "Flash packed route count");
        let expert_counts = allocate(
            DS4_FLASH_PREFILL_EXPERT_COUNT_BYTES,
            "Flash route-pack expert counts",
        );
        let expert_offsets = allocate(
            DS4_FLASH_PREFILL_EXPERT_OFFSET_BYTES,
            "Flash route-pack expert offsets",
        );
        let buffers = Ds4FlashSparkPrefillRoutePackDeviceBuffers {
            topk_ids: topk.buffer(),
            packed_route_indices: packed.buffer(),
            block_expert_ids: block_experts.buffer(),
            packed_route_count: packed_count.buffer(),
            expert_counts: expert_counts.buffer(),
            expert_offsets: expert_offsets.buffer(),
        };
        let fixed_pointers = [
            buffers.topk_ids.ptr,
            buffers.packed_route_indices.ptr,
            buffers.block_expert_ids.ptr,
            buffers.packed_route_count.ptr,
            buffers.expert_counts.ptr,
            buffers.expert_offsets.ptr,
        ];
        library
            .copy_h2d(buffers.topk_ids, super::u32_bytes(&topk_ids))
            .expect("uploading global Flash top-k IDs");
        let stream = library
            .cuda_stream_create()
            .expect("creating Flash route-pack stream");
        ds4_flash_spark_prefill_pack_topk6_device_into(rows, buffers, stream, library.as_ref())
            .expect("packing Flash routes on device");
        unsafe {
            library
                .cuda_stream_synchronize(stream)
                .expect("synchronizing Flash route pack");
        }

        let read_i32 = |buffer: Ds41rtDeviceBuffer, bytes: usize| {
            let mut payload = vec![0_u8; bytes];
            library
                .copy_d2h(&mut payload, buffer)
                .expect("copying Flash route-pack output");
            payload
                .chunks_exact(std::mem::size_of::<i32>())
                .map(|bytes| i32::from_ne_bytes(bytes.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let validate_output = || {
            let routes = read_i32(
                buffers.packed_route_indices,
                DS4_FLASH_PREFILL_PACKED_ROUTE_BYTES,
            );
            let blocks = read_i32(
                buffers.block_expert_ids,
                DS4_FLASH_PREFILL_BLOCK_EXPERT_BYTES,
            );
            let count =
                read_i32(buffers.packed_route_count, std::mem::size_of::<i32>())[0] as usize;
            let counts = read_i32(buffers.expert_counts, DS4_FLASH_PREFILL_EXPERT_COUNT_BYTES);
            let offsets = read_i32(
                buffers.expert_offsets,
                DS4_FLASH_PREFILL_EXPERT_OFFSET_BYTES,
            );
            assert_eq!(count, expected.packed_route_count as usize);
            assert_eq!(
                &blocks[..count / DS4_FLASH_PREFILL_ROUTE_BLOCK_ROWS],
                expected
                    .block_expert_ids
                    .iter()
                    .map(|&expert| expert as i32)
                    .collect::<Vec<_>>()
            );
            assert!(blocks[count / DS4_FLASH_PREFILL_ROUTE_BLOCK_ROWS..]
                .iter()
                .all(|&expert| expert == -1));
            assert!(routes[count..]
                .iter()
                .all(|&route| route == route_count as i32));

            let mut payload = routes[..count]
                .iter()
                .copied()
                .filter(|&route| route < route_count as i32)
                .collect::<Vec<_>>();
            payload.sort_unstable();
            assert_eq!(payload, (0..route_count as i32).collect::<Vec<_>>());
            for (block, &expert) in blocks[..count / 32].iter().enumerate() {
                for &route in &routes[block * 32..(block + 1) * 32] {
                    if route < route_count as i32 {
                        assert_eq!(topk_ids[route as usize], expert as u32);
                    }
                }
            }

            let mut padded_offset = 0_i32;
            for expert in 0..256 {
                let expected_count = topk_ids
                    .iter()
                    .filter(|&&candidate| candidate as usize == expert)
                    .count() as i32;
                assert_eq!(counts[expert], expected_count);
                assert_eq!(offsets[expert], padded_offset + expected_count);
                padded_offset += ((expected_count + 31) / 32) * 32;
            }
            assert_eq!(offsets[256], count as i32);
        };
        validate_output();

        unsafe {
            library
                .cuda_graph_begin_capture(stream)
                .expect("beginning Flash route-pack graph capture");
        }
        ds4_flash_spark_prefill_pack_topk6_device_into(rows, buffers, stream, library.as_ref())
            .expect("capturing Flash route pack");
        let capture = unsafe {
            library
                .cuda_graph_end_capture_retained(stream)
                .expect("ending Flash route-pack graph capture")
        };
        assert!(capture.node_count >= 4);
        assert!(capture.kernel_node_count >= 3);
        assert!(capture.memset_node_count >= 1);
        assert_eq!(capture.memcpy_node_count, 0);
        for _ in 0..3 {
            unsafe {
                library
                    .cuda_graph_launch(capture.graph_exec, stream)
                    .expect("replaying Flash route-pack graph");
            }
        }
        unsafe {
            library
                .cuda_stream_synchronize(stream)
                .expect("synchronizing Flash route-pack graph replay");
        }
        validate_output();
        assert_eq!(
            fixed_pointers,
            [
                buffers.topk_ids.ptr,
                buffers.packed_route_indices.ptr,
                buffers.block_expert_ids.ptr,
                buffers.packed_route_count.ptr,
                buffers.expert_counts.ptr,
                buffers.expert_offsets.ptr,
            ]
        );
        unsafe {
            library
                .cuda_graph_destroy(capture.graph)
                .expect("destroying Flash route-pack graph");
            library
                .cuda_graph_exec_destroy(capture.graph_exec)
                .expect("destroying Flash route-pack graph exec");
            library
                .cuda_stream_destroy(stream)
                .expect("destroying Flash route-pack stream");
        }
    }

    #[test]
    #[ignore = "requires a real Flash checkpoint and CUDA"]
    fn native_flash_checkpoint_prefill_uses_fixed_caller_owned_tp4_rank_partial() {
        let snapshot = std::env::var("DS41RT_DS4_FLASH_SNAPSHOT")
            .expect("DS41RT_DS4_FLASH_SNAPSHOT must name the real Flash snapshot");
        let catalog =
            build_catalog_for_snapshot("deepseek-ai/DeepSeek-V4-Flash-0731", Path::new(&snapshot))
                .expect("building Flash checkpoint catalog");
        let layer_id = 3_usize;
        let shard = ExpertIntermediateShard::new(4, 0).expect("constructing Flash TP4 rank zero");
        let mut cache = RouteTensorCache::default();
        preload_native_flash_layer_cuda_cache_local_tp4(&catalog, layer_id, shard, &mut cache)
            .expect("preloading one real Flash TP4 rank slab");

        let rows = 7_usize;
        let active_hidden_bytes = rows * 4_096 * std::mem::size_of::<u16>();
        let hidden_values = (0..rows * 4_096)
            .map(|index| ((index * 29 % 521) as f32 - 260.0) / 2048.0)
            .collect::<Vec<_>>();
        let mut hidden_bf16 = vec![0_u8; active_hidden_bytes];
        f32_values_to_bf16_bytes(&hidden_values, &mut hidden_bf16);
        let route_indices = (0..rows * 6)
            .map(|route| ((route * 37 + route / 6 * 11) % 256) as u32)
            .collect::<Vec<_>>();
        let route_weights = (0..rows * 6)
            .map(|route| ((route % 6) + 1) as f32 / 21.0)
            .collect::<Vec<_>>();
        let expected = execute_native_flash_route_bf16_cached(
            0,
            layer_id,
            rows,
            &hidden_bf16,
            None,
            &route_indices,
            &route_weights,
            false,
            false,
            None,
            &mut cache,
        )
        .expect("executing existing Flash prefill TP4 rank wrapper")
        .output_bf16;

        let library = Arc::clone(
            &cache
                .cuda
                .as_ref()
                .expect("Flash route CUDA cache exists")
                .library,
        );
        let allocate = |bytes, label| {
            OwnedDeviceAllocation::new(Arc::clone(&library), bytes, label)
                .unwrap_or_else(|error| panic!("allocating {label}: {error:#}"))
        };
        let input = allocate(DS4_FLASH_PREFILL_HIDDEN_BYTES, "Flash prefill rank input");
        let topk_ids = allocate(DS4_FLASH_PREFILL_TOPK_ID_BYTES, "Flash prefill top-k IDs");
        let topk_weights = allocate(
            DS4_FLASH_PREFILL_TOPK_WEIGHT_BYTES,
            "Flash prefill top-k weights",
        );
        let packed_routes = allocate(
            DS4_FLASH_PREFILL_PACKED_ROUTE_BYTES,
            "Flash prefill packed routes",
        );
        let block_experts = allocate(
            DS4_FLASH_PREFILL_BLOCK_EXPERT_BYTES,
            "Flash prefill block experts",
        );
        let packed_count = allocate(std::mem::size_of::<i32>(), "Flash prefill route count");
        let expert_counts = allocate(
            DS4_FLASH_PREFILL_EXPERT_COUNT_BYTES,
            "Flash prefill expert counts",
        );
        let expert_offsets = allocate(
            DS4_FLASH_PREFILL_EXPERT_OFFSET_BYTES,
            "Flash prefill expert offsets",
        );
        let fc1 = allocate(DS4_FLASH_PREFILL_FC1_BYTES, "Flash prefill FC1");
        let activated = allocate(
            DS4_FLASH_PREFILL_ACTIVATED_BYTES,
            "Flash prefill activation",
        );
        let rotation_gate = allocate(
            DS4_FLASH_PREFILL_ROUTED_OUTPUT_BYTES,
            "Flash prefill gate rotation",
        );
        let rotation_up = allocate(
            DS4_FLASH_PREFILL_ROUTED_OUTPUT_BYTES,
            "Flash prefill up rotation",
        );
        let routed_output = allocate(
            DS4_FLASH_PREFILL_ROUTED_OUTPUT_BYTES,
            "Flash prefill routed output",
        );
        let output_f32 = allocate(
            DS4_FLASH_PREFILL_OUTPUT_F32_BYTES,
            "Flash prefill FP32 output",
        );
        let output = allocate(DS4_FLASH_PREFILL_HIDDEN_BYTES, "Flash prefill rank partial");
        let fc1_scratch = allocate(
            DS4_FLASH_PREFILL_COMPUTE_SCRATCH_BYTES,
            "Flash prefill FC1 scratch",
        );
        let fc2_scratch = allocate(
            DS4_FLASH_PREFILL_COMPUTE_SCRATCH_BYTES,
            "Flash prefill FC2 scratch",
        );
        let locks = allocate(DS4_FLASH_DECODE_M1_LOCK_BYTES, "Flash prefill locks");
        let buffers = Ds4FlashSparkPrefillDeviceBuffers {
            input: input.buffer(),
            topk_ids: topk_ids.buffer(),
            topk_weights: topk_weights.buffer(),
            packed_route_indices: packed_routes.buffer(),
            block_expert_ids: block_experts.buffer(),
            packed_route_count: packed_count.buffer(),
            expert_counts: expert_counts.buffer(),
            expert_offsets: expert_offsets.buffer(),
            fc1: fc1.buffer(),
            activated: activated.buffer(),
            rotation_gate: rotation_gate.buffer(),
            rotation_up: rotation_up.buffer(),
            routed_output: routed_output.buffer(),
            output_f32: output_f32.buffer(),
            output: output.buffer(),
            fc1_scratch: fc1_scratch.buffer(),
            fc2_scratch: fc2_scratch.buffer(),
            locks: locks.buffer(),
        };
        let fixed_pointers = [
            buffers.input.ptr,
            buffers.topk_ids.ptr,
            buffers.topk_weights.ptr,
            buffers.packed_route_indices.ptr,
            buffers.block_expert_ids.ptr,
            buffers.packed_route_count.ptr,
            buffers.expert_counts.ptr,
            buffers.expert_offsets.ptr,
            buffers.fc1.ptr,
            buffers.activated.ptr,
            buffers.routed_output.ptr,
            buffers.output.ptr,
            buffers.fc1_scratch.ptr,
            buffers.fc2_scratch.ptr,
            buffers.locks.ptr,
        ];
        library
            .copy_h2d(buffers.input, &hidden_bf16)
            .expect("uploading fixed Flash prefill hidden input");
        library
            .copy_h2d(buffers.topk_ids, super::u32_bytes(&route_indices))
            .expect("uploading fixed Flash prefill global route IDs");
        library
            .copy_h2d(buffers.topk_weights, super::f32_bytes(&route_weights))
            .expect("uploading fixed Flash prefill route weights");

        let stream = library
            .cuda_stream_create()
            .expect("creating caller-owned Flash prefill graph stream");
        ds4_flash_spark_prefill_topk6_preloaded_device_input_into(
            layer_id, rows, buffers, stream, &mut cache,
        )
        .expect("launching caller-owned Flash prefill TP4 rank partial");
        unsafe {
            library
                .cuda_stream_synchronize(stream)
                .expect("synchronizing caller-owned Flash prefill launch");
        }
        let mut eager = vec![0_u8; active_hidden_bytes];
        library
            .copy_d2h(&mut eager, buffers.output)
            .expect("copying caller-owned Flash prefill rank partial");
        let assert_same_rank_partial = |label: &str, actual: &[u8], wanted: &[u8]| {
            let decode = |bytes: &[u8]| {
                bytes
                    .chunks_exact(std::mem::size_of::<u16>())
                    .map(|bytes| {
                        f32::from_bits(
                            u32::from(u16::from_ne_bytes(bytes.try_into().unwrap())) << 16,
                        )
                    })
                    .collect::<Vec<_>>()
            };
            let actual = decode(actual);
            let wanted = decode(wanted);
            let dot = actual
                .iter()
                .zip(&wanted)
                .map(|(left, right)| left * right)
                .sum::<f32>();
            let actual_norm = actual.iter().map(|value| value * value).sum::<f32>();
            let wanted_norm = wanted.iter().map(|value| value * value).sum::<f32>();
            let error_norm = actual
                .iter()
                .zip(&wanted)
                .map(|(left, right)| (left - right) * (left - right))
                .sum::<f32>();
            let cosine = dot / (actual_norm.sqrt() * wanted_norm.sqrt());
            let relative_l2 = (error_norm / wanted_norm).sqrt();
            assert!(
                cosine >= 0.999_98 && relative_l2 <= 0.006,
                "{label} differs from existing wrapper: cosine={cosine:.9} relative_l2={relative_l2:.9}",
            );
        };
        assert_same_rank_partial("caller-owned prefill eager output", &eager, &expected);

        unsafe {
            library
                .cuda_graph_begin_capture(stream)
                .expect("beginning caller-owned Flash prefill graph capture");
        }
        ds4_flash_spark_prefill_topk6_preloaded_device_input_into(
            layer_id, rows, buffers, stream, &mut cache,
        )
        .expect("capturing caller-owned Flash prefill TP4 rank partial");
        let capture = unsafe {
            library
                .cuda_graph_end_capture_retained(stream)
                .expect("ending caller-owned Flash prefill graph capture")
        };
        assert!(capture.node_count >= 7);
        assert!(capture.kernel_node_count >= 5);
        assert!(capture.memset_node_count >= 2);
        assert_eq!(capture.memcpy_node_count, 0);
        for _ in 0..3 {
            unsafe {
                library
                    .cuda_graph_launch(capture.graph_exec, stream)
                    .expect("replaying caller-owned Flash prefill graph");
            }
        }
        unsafe {
            library
                .cuda_stream_synchronize(stream)
                .expect("synchronizing caller-owned Flash prefill graph replay");
        }
        let mut replay = vec![0_u8; active_hidden_bytes];
        library
            .copy_d2h(&mut replay, buffers.output)
            .expect("copying replayed Flash prefill rank partial");
        assert_same_rank_partial("caller-owned prefill graph output", &replay, &expected);
        assert_eq!(
            fixed_pointers,
            [
                buffers.input.ptr,
                buffers.topk_ids.ptr,
                buffers.topk_weights.ptr,
                buffers.packed_route_indices.ptr,
                buffers.block_expert_ids.ptr,
                buffers.packed_route_count.ptr,
                buffers.expert_counts.ptr,
                buffers.expert_offsets.ptr,
                buffers.fc1.ptr,
                buffers.activated.ptr,
                buffers.routed_output.ptr,
                buffers.output.ptr,
                buffers.fc1_scratch.ptr,
                buffers.fc2_scratch.ptr,
                buffers.locks.ptr,
            ]
        );
        unsafe {
            library
                .cuda_graph_destroy(capture.graph)
                .expect("destroying caller-owned Flash prefill graph");
            library
                .cuda_graph_exec_destroy(capture.graph_exec)
                .expect("destroying caller-owned Flash prefill graph exec");
            library
                .cuda_stream_destroy(stream)
                .expect("destroying caller-owned Flash prefill graph stream");
        }
    }

    #[test]
    #[ignore = "requires a real Flash checkpoint and CUDA"]
    fn native_flash_checkpoint_decode_m1_uses_fixed_caller_owned_tp4_rank_partial() {
        let snapshot = std::env::var("DS41RT_DS4_FLASH_SNAPSHOT")
            .expect("DS41RT_DS4_FLASH_SNAPSHOT must name the real Flash snapshot");
        let catalog =
            build_catalog_for_snapshot("deepseek-ai/DeepSeek-V4-Flash-0731", Path::new(&snapshot))
                .expect("building Flash checkpoint catalog");
        let layer_id = 3_usize;
        let shard = ExpertIntermediateShard::new(4, 0).expect("constructing Flash TP4 rank zero");
        let mut cache = RouteTensorCache::default();
        {
            let cuda_cache = cache
                .cuda_cache_native_flash_tp4(shard)
                .expect("creating Flash route CUDA cache");
            cuda_cache
                .ensure_ds4_flash_aot()
                .expect("initializing Flash Spark AOT");
            let preload_stream = RouteCudaStream::new(Arc::clone(&cuda_cache.library))
                .expect("creating Flash rank preload stream");
            preload_native_flash_projection_cuda_cache_local_tp4(
                &catalog,
                &[layer_id],
                &(0..256).collect::<Vec<_>>(),
                cuda_cache,
                shard,
                preload_stream.as_ptr(),
            )
            .expect("preloading one real Flash TP4 rank slab");
        }

        let hidden_values = (0..4_096)
            .map(|index| ((index * 17 % 257) as f32 - 128.0) / 1024.0)
            .collect::<Vec<_>>();
        let mut hidden_bf16 = vec![0_u8; DS4_FLASH_DECODE_M1_HIDDEN_BYTES];
        f32_values_to_bf16_bytes(&hidden_values, &mut hidden_bf16);
        let route_indices = [19_u32, 3, 255, 44, 0, 128];
        let route_weights = [0.1_f32, 0.2, 0.05, 0.25, 0.15, 0.25];
        let expected = execute_native_flash_route_bf16_cached(
            0,
            layer_id,
            1,
            &hidden_bf16,
            None,
            &route_indices,
            &route_weights,
            false,
            false,
            None,
            &mut cache,
        )
        .expect("executing existing Flash M1 TP4 rank wrapper")
        .output_bf16;

        let library = Arc::clone(
            &cache
                .cuda
                .as_ref()
                .expect("Flash route CUDA cache exists")
                .library,
        );
        let input = OwnedDeviceAllocation::new(
            Arc::clone(&library),
            DS4_FLASH_DECODE_M1_HIDDEN_BYTES,
            "caller-owned Flash rank input",
        )
        .expect("allocating Flash rank input");
        let indices = OwnedDeviceAllocation::new(
            Arc::clone(&library),
            DS4_FLASH_DECODE_M1_ROUTE_BYTES,
            "caller-owned Flash rank route indices",
        )
        .expect("allocating Flash rank route indices");
        let weights = OwnedDeviceAllocation::new(
            Arc::clone(&library),
            DS4_FLASH_DECODE_M1_ROUTE_BYTES,
            "caller-owned Flash rank route weights",
        )
        .expect("allocating Flash rank route weights");
        let fc1 = OwnedDeviceAllocation::new(
            Arc::clone(&library),
            DS4_FLASH_DECODE_M1_FC1_BYTES,
            "caller-owned Flash rank FC1",
        )
        .expect("allocating Flash rank FC1");
        let activated = OwnedDeviceAllocation::new(
            Arc::clone(&library),
            DS4_FLASH_DECODE_M1_ACTIVATED_BYTES,
            "caller-owned Flash rank activation",
        )
        .expect("allocating Flash rank activation");
        let rotation_gate = OwnedDeviceAllocation::new(
            Arc::clone(&library),
            DS4_FLASH_DECODE_M1_ROUTED_OUTPUT_BYTES,
            "caller-owned Flash gate rotation",
        )
        .expect("allocating Flash gate rotation");
        let rotation_up = OwnedDeviceAllocation::new(
            Arc::clone(&library),
            DS4_FLASH_DECODE_M1_ROUTED_OUTPUT_BYTES,
            "caller-owned Flash up rotation",
        )
        .expect("allocating Flash up rotation");
        let routed_output = OwnedDeviceAllocation::new(
            Arc::clone(&library),
            DS4_FLASH_DECODE_M1_ROUTED_OUTPUT_BYTES,
            "caller-owned Flash routed output",
        )
        .expect("allocating Flash routed output");
        let output_f32 = OwnedDeviceAllocation::new(
            Arc::clone(&library),
            DS4_FLASH_DECODE_M1_OUTPUT_F32_BYTES,
            "caller-owned Flash FP32 output",
        )
        .expect("allocating Flash FP32 output");
        let output = OwnedDeviceAllocation::new(
            Arc::clone(&library),
            DS4_FLASH_DECODE_M1_HIDDEN_BYTES,
            "caller-owned Flash rank partial",
        )
        .expect("allocating Flash rank partial");
        let block_expert_ids = OwnedDeviceAllocation::new(
            Arc::clone(&library),
            std::mem::size_of::<i32>(),
            "caller-owned Flash decode block expert IDs",
        )
        .expect("allocating Flash decode block expert IDs");
        let packed_route_count = OwnedDeviceAllocation::new(
            Arc::clone(&library),
            std::mem::size_of::<i32>(),
            "caller-owned Flash decode route count",
        )
        .expect("allocating Flash decode route count");
        let expert_counts = OwnedDeviceAllocation::new(
            Arc::clone(&library),
            DS4_FLASH_PREFILL_EXPERT_COUNT_BYTES,
            "caller-owned Flash expert counts",
        )
        .expect("allocating Flash expert counts");
        let expert_offsets = OwnedDeviceAllocation::new(
            Arc::clone(&library),
            DS4_FLASH_PREFILL_EXPERT_OFFSET_BYTES,
            "caller-owned Flash expert offsets",
        )
        .expect("allocating Flash expert offsets");
        let fc1_scratch = OwnedDeviceAllocation::new(
            Arc::clone(&library),
            DS4_FLASH_DECODE_M1_FC1_SCRATCH_BYTES,
            "caller-owned Flash rank FC1 scratch",
        )
        .expect("allocating Flash rank FC1 scratch");
        let fc2_scratch = OwnedDeviceAllocation::new(
            Arc::clone(&library),
            DS4_FLASH_DECODE_M1_FC2_SCRATCH_BYTES,
            "caller-owned Flash rank FC2 scratch",
        )
        .expect("allocating Flash rank FC2 scratch");
        let locks = OwnedDeviceAllocation::new(
            Arc::clone(&library),
            DS4_FLASH_DECODE_M1_LOCK_BYTES,
            "caller-owned Flash rank locks",
        )
        .expect("allocating Flash rank locks");
        let buffers = Ds4FlashSparkDecodeM1DeviceBuffers {
            input: input.buffer(),
            route_indices: indices.buffer(),
            route_weights: weights.buffer(),
            fc1: fc1.buffer(),
            activated: activated.buffer(),
            rotation_gate: rotation_gate.buffer(),
            rotation_up: rotation_up.buffer(),
            routed_output: routed_output.buffer(),
            output_f32: output_f32.buffer(),
            output: output.buffer(),
            block_expert_ids: block_expert_ids.buffer(),
            packed_route_count: packed_route_count.buffer(),
            expert_counts: expert_counts.buffer(),
            expert_offsets: expert_offsets.buffer(),
            fc1_scratch: fc1_scratch.buffer(),
            fc2_scratch: fc2_scratch.buffer(),
            locks: locks.buffer(),
        };
        let fixed_pointers = [
            buffers.input.ptr,
            buffers.route_indices.ptr,
            buffers.route_weights.ptr,
            buffers.fc1.ptr,
            buffers.activated.ptr,
            buffers.output.ptr,
            buffers.block_expert_ids.ptr,
            buffers.packed_route_count.ptr,
            buffers.fc1_scratch.ptr,
            buffers.fc2_scratch.ptr,
            buffers.locks.ptr,
        ];
        library
            .copy_h2d(buffers.input, &hidden_bf16)
            .expect("uploading fixed Flash rank input");
        library
            .copy_h2d(buffers.route_indices, super::u32_bytes(&route_indices))
            .expect("uploading fixed Flash global route indices");
        library
            .copy_h2d(buffers.route_weights, super::f32_bytes(&route_weights))
            .expect("uploading fixed Flash global route weights");
        library
            .copy_h2d(buffers.block_expert_ids, &0_i32.to_ne_bytes())
            .expect("initializing fixed Flash block expert scratch");
        library
            .copy_h2d(buffers.packed_route_count, &6_i32.to_ne_bytes())
            .expect("initializing fixed Flash route count scratch");

        let stream = library
            .cuda_stream_create()
            .expect("creating caller-owned Flash rank graph stream");
        ds4_flash_spark_decode_m1_preloaded_device_input_into(
            layer_id, buffers, stream, &mut cache,
        )
        .expect("launching caller-owned Flash M1 TP4 rank partial");
        unsafe {
            library
                .cuda_stream_synchronize(stream)
                .expect("synchronizing caller-owned Flash rank launch");
        }
        let mut eager = vec![0_u8; DS4_FLASH_DECODE_M1_HIDDEN_BYTES];
        library
            .copy_d2h(&mut eager, buffers.output)
            .expect("copying caller-owned Flash rank partial");
        let assert_same_rank_partial = |label: &str, actual: &[u8], wanted: &[u8]| {
            let decode = |bytes: &[u8]| {
                bytes
                    .chunks_exact(std::mem::size_of::<u16>())
                    .map(|bytes| {
                        f32::from_bits(
                            u32::from(u16::from_ne_bytes(bytes.try_into().unwrap())) << 16,
                        )
                    })
                    .collect::<Vec<_>>()
            };
            let actual = decode(actual);
            let wanted = decode(wanted);
            let dot = actual
                .iter()
                .zip(&wanted)
                .map(|(left, right)| left * right)
                .sum::<f32>();
            let actual_norm = actual.iter().map(|value| value * value).sum::<f32>();
            let wanted_norm = wanted.iter().map(|value| value * value).sum::<f32>();
            let error_norm = actual
                .iter()
                .zip(&wanted)
                .map(|(left, right)| (left - right) * (left - right))
                .sum::<f32>();
            let cosine = dot / (actual_norm.sqrt() * wanted_norm.sqrt());
            let relative_l2 = (error_norm / wanted_norm).sqrt();
            assert!(
                cosine >= 0.999_98 && relative_l2 <= 0.006,
                "{label} differs from existing wrapper: cosine={cosine:.9} relative_l2={relative_l2:.9}",
            );
        };
        assert_same_rank_partial("caller-owned eager output", &eager, &expected);

        unsafe {
            library
                .cuda_graph_begin_capture(stream)
                .expect("beginning caller-owned Flash rank graph capture");
        }
        ds4_flash_spark_decode_m1_preloaded_device_input_into(
            layer_id, buffers, stream, &mut cache,
        )
        .expect("capturing caller-owned Flash M1 TP4 rank partial");
        let capture = unsafe {
            library
                .cuda_graph_end_capture_retained(stream)
                .expect("ending caller-owned Flash rank graph capture")
        };
        assert!(capture.node_count >= 2);
        assert!(capture.kernel_node_count >= 1);
        assert!(capture.memset_node_count >= 1);
        assert_eq!(capture.memcpy_node_count, 0);
        for _ in 0..3 {
            unsafe {
                library
                    .cuda_graph_launch(capture.graph_exec, stream)
                    .expect("replaying caller-owned Flash rank graph");
            }
        }
        unsafe {
            library
                .cuda_stream_synchronize(stream)
                .expect("synchronizing caller-owned Flash rank graph replay");
        }
        let mut replay = vec![0_u8; DS4_FLASH_DECODE_M1_HIDDEN_BYTES];
        library
            .copy_d2h(&mut replay, buffers.output)
            .expect("copying replayed Flash rank partial");
        assert_same_rank_partial("caller-owned graph output", &replay, &expected);
        assert_eq!(
            fixed_pointers,
            [
                buffers.input.ptr,
                buffers.route_indices.ptr,
                buffers.route_weights.ptr,
                buffers.fc1.ptr,
                buffers.activated.ptr,
                buffers.output.ptr,
                buffers.block_expert_ids.ptr,
                buffers.packed_route_count.ptr,
                buffers.fc1_scratch.ptr,
                buffers.fc2_scratch.ptr,
                buffers.locks.ptr,
            ]
        );
        unsafe {
            library
                .cuda_graph_destroy(capture.graph)
                .expect("destroying caller-owned Flash rank graph");
            library
                .cuda_graph_exec_destroy(capture.graph_exec)
                .expect("destroying caller-owned Flash rank graph exec");
            library
                .cuda_stream_destroy(stream)
                .expect("destroying caller-owned Flash rank graph stream");
        }
    }

    #[test]
    fn native_flash_prefill_plan_groups_routes_in_block32_without_losing_order() -> Result<()> {
        let rows = 7;
        let ids = (0..rows * 6)
            .map(|route| ((route * 17 + route / 6) % 256) as u32)
            .collect::<Vec<_>>();
        let weights = (0..rows * 6)
            .map(|route| route as f32 + 0.5)
            .collect::<Vec<_>>();

        let plan = plan_ds4_flash_packed_topk6_routes(rows, &ids, &weights)?;

        assert_eq!(plan.packed_route_indices.len() % 32, 0);
        assert_eq!(
            plan.packed_route_indices.len(),
            plan.block_expert_ids.len() * DS4_FLASH_PREFILL_ROUTE_BLOCK_ROWS
        );
        assert_eq!(plan.topk_weights, weights);
        let sentinel = (rows * 6) as u32;
        let mut restored = plan
            .packed_route_indices
            .iter()
            .copied()
            .filter(|&route| route != sentinel)
            .collect::<Vec<_>>();
        restored.sort_unstable();
        assert_eq!(restored, (0..sentinel).collect::<Vec<_>>());
        for (block, &expert_id) in plan.block_expert_ids.iter().enumerate() {
            for &route in &plan.packed_route_indices[block * 32..(block + 1) * 32] {
                assert!(route == sentinel || ids[route as usize] == expert_id);
            }
        }
        assert_eq!(ds4_flash_w4a16_capacity_rows(rows)?, 8);
        assert_eq!(ds4_flash_w4a16_capacity_rows(1025)?, 2048);
        Ok(())
    }

    #[test]
    fn packed_w4a16_topk8_prefill_plan_packs_eight_rows_in_m1_shaped_blocks() -> Result<()> {
        let routes = (0..64)
            .map(|route| PackedW4a16Topk8Route {
                expert_id: (route % 2) as u32,
                weight: route as f32 + 1.0,
            })
            .collect::<Vec<_>>();

        let plan = plan_packed_w4a16_topk8_prefill_flat(8, &routes)?;

        assert_eq!(plan.packed_route_count, 64);
        assert_eq!(
            plan.direct_topk_ids,
            (0..64).map(|route| (route % 2) as u32).collect::<Vec<_>>()
        );
        assert_eq!(plan.block_expert_ids, vec![0, 0, 0, 0, 1, 1, 1, 1]);
        assert_eq!(
            &plan.packed_route_indices[..32],
            &(0..64).step_by(2).collect::<Vec<_>>()
        );
        assert_eq!(
            &plan.packed_route_indices[32..],
            &(1..64).step_by(2).collect::<Vec<_>>()
        );
        Ok(())
    }

    #[test]
    fn packed_w4a16_topk8_prefill_plan_groups_larger_batches_by_expert() -> Result<()> {
        let row_routes = (0..16)
            .map(|row| {
                (0..8)
                    .map(|slot| {
                        let route_index = row * 8 + slot;
                        (
                            ScoredRoute {
                                expert_id: slot % 2,
                                score: 0.0,
                                corrected_score: 0.0,
                                normalized_weight: route_index as f32 + 1.0,
                            },
                            512,
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let plan = plan_packed_w4a16_topk8_prefill(&row_routes)?;

        assert_eq!(plan.packed_route_count, 128);
        assert_eq!(plan.block_expert_ids, vec![0, 0, 1, 1]);
        assert_eq!(
            &plan.packed_route_indices[..64],
            &(0..128).step_by(2).collect::<Vec<_>>()
        );
        assert_eq!(
            &plan.packed_route_indices[64..],
            &(1..128).step_by(2).collect::<Vec<_>>()
        );
        assert_eq!(
            plan.topk_weights,
            (1..=128).map(|value| value as f32).collect::<Vec<_>>()
        );
        Ok(())
    }

    #[test]
    fn packed_w4a16_topk8_prefill_accepts_small_complete_batches() {
        assert!(packed_w4a16_topk8_prefill_eligible(2, 16));
        assert!(packed_w4a16_topk8_prefill_eligible(7, 56));
        assert!(packed_w4a16_topk8_prefill_eligible(1, 8));
        assert!(!packed_w4a16_topk8_prefill_eligible(7, 55));
        assert!(packed_w4a16_topk8_prefill_eligible(1025, 8200));
        assert!(!packed_w4a16_topk8_prefill_eligible(2049, 16392));
    }

    #[test]
    fn packed_w4a16_workspace_capacity_matches_native_buckets() -> Result<()> {
        assert_eq!(b12x_w4a16_capacity_rows(1)?, 2);
        assert_eq!(b12x_w4a16_capacity_rows(2)?, 2);
        assert_eq!(b12x_w4a16_capacity_rows(3)?, 4);
        assert_eq!(b12x_w4a16_capacity_rows(256)?, 256);
        assert_eq!(b12x_w4a16_capacity_rows(257)?, 512);
        assert_eq!(b12x_w4a16_capacity_rows(1024)?, 1024);
        assert_eq!(b12x_w4a16_capacity_rows(1025)?, 2048);
        assert_eq!(b12x_w4a16_capacity_rows(2048)?, 2048);
        Ok(())
    }

    #[test]
    fn packed_w4a16_prefill_uses_exported_route_block_regimes() {
        assert_eq!(b12x_w4a16_prefill_route_block_rows(2), 8);
        assert_eq!(b12x_w4a16_prefill_route_block_rows(8), 8);
        assert_eq!(b12x_w4a16_prefill_route_block_rows(9), 32);
        assert_eq!(b12x_w4a16_prefill_route_block_rows(512), 32);
        assert_eq!(b12x_w4a16_prefill_route_block_rows(1024), 32);
        assert_eq!(b12x_w4a16_prefill_route_block_rows(2048), 32);
        assert_eq!(b12x_w4a16_prefill_route_block_rows(2049), 48);
    }

    #[test]
    fn persistent_b12x_scale_layout_only_selects_model_projection_shapes() {
        assert!(b12x_projection_scale_shape_supported(512, 384));
        assert!(b12x_projection_scale_shape_supported(2048, 384));
        assert!(b12x_projection_scale_shape_supported(6144, 32));
        assert!(b12x_projection_scale_shape_supported(6144, 128));
        assert!(!b12x_projection_scale_shape_supported(2048, 128));
        assert!(!b12x_projection_scale_shape_supported(64, 384));
    }

    #[test]
    fn intermediate_shard_loads_logical_weight_and_scale_coordinates() -> Result<()> {
        let tempdir = tempfile::tempdir()?;
        let mut shard_bytes = Vec::new();
        let mut tensors = Vec::new();
        let gate_bytes = (0_u8..32).collect::<Vec<_>>();
        let down_bytes = (0_u8..32).collect::<Vec<_>>();
        let gate_scale_bytes = (32_u8..64).collect::<Vec<_>>();
        let down_scale_bytes = (64_u8..96).collect::<Vec<_>>();
        push_tensor(
            &mut shard_bytes,
            &mut tensors,
            3,
            0,
            "gate_proj",
            "weight",
            DType::U8,
            vec![8, 4],
            &gate_bytes,
        );
        push_tensor(
            &mut shard_bytes,
            &mut tensors,
            3,
            0,
            "down_proj",
            "weight",
            DType::U8,
            vec![4, 8],
            &down_bytes,
        );
        push_tensor(
            &mut shard_bytes,
            &mut tensors,
            3,
            0,
            "gate_proj",
            "weight_scale",
            DType::F8E4M3,
            vec![8, 4],
            &gate_scale_bytes,
        );
        push_tensor(
            &mut shard_bytes,
            &mut tensors,
            3,
            0,
            "down_proj",
            "weight_scale",
            DType::F8E4M3,
            vec![4, 8],
            &down_scale_bytes,
        );
        File::create(tempdir.path().join("route.bin"))?.write_all(&shard_bytes)?;
        let catalog = TensorCatalog {
            model_id: "test/model".to_owned(),
            snapshot_path: tempdir.path().display().to_string(),
            facts: ModelFacts::default(),
            tensors,
        };
        let shard = ExpertIntermediateShard::new(4, 2)?;
        let gate = load_routed_projection_rows_for_shard(
            &catalog,
            "model.layers.3.mlp.experts.0.gate_proj.weight",
            "gate_proj",
            2,
            shard,
        )?;
        let down = load_routed_projection_rows_for_shard(
            &catalog,
            "model.layers.3.mlp.experts.0.down_proj.weight",
            "down_proj",
            4,
            shard,
        )?;
        let gate_scale = load_routed_projection_rows_for_shard(
            &catalog,
            "model.layers.3.mlp.experts.0.gate_proj.weight_scale",
            "gate_proj",
            2,
            shard,
        )?;
        let down_scale = load_routed_projection_rows_for_shard(
            &catalog,
            "model.layers.3.mlp.experts.0.down_proj.weight_scale",
            "down_proj",
            4,
            shard,
        )?;

        assert_eq!(gate.start_row, 4);
        assert_eq!(gate.row_width, 4);
        assert_eq!(gate.bytes, gate_bytes[16..24]);
        assert_eq!(down.row_width, 2);
        assert_eq!(down.bytes, vec![4, 5, 12, 13, 20, 21, 28, 29]);
        assert_eq!(gate_scale.start_row, 4);
        assert_eq!(gate_scale.row_width, 4);
        assert_eq!(gate_scale.bytes, gate_scale_bytes[16..24]);
        assert_eq!(down_scale.row_width, 2);
        assert_eq!(down_scale.bytes, vec![68, 69, 76, 77, 84, 85, 92, 93]);
        Ok(())
    }

    #[test]
    fn fused_fp8_reduction_requires_full_identity_fp8_rows() {
        let fp8 = RouteStreamingOutputDtype::Fp8E4m3RowScaled;
        assert!(fused_fp8_reduction_eligible(fp8, fp8, &[0, 1, 2, 3], 4));
        assert!(!fused_fp8_reduction_eligible(
            RouteStreamingOutputDtype::Bf16,
            fp8,
            &[0, 1, 2, 3],
            4,
        ));
        assert!(!fused_fp8_reduction_eligible(
            fp8,
            RouteStreamingOutputDtype::Bf16,
            &[0, 1, 2, 3],
            4,
        ));
        assert!(!fused_fp8_reduction_eligible(fp8, fp8, &[0, 1, 2], 4));
        assert!(!fused_fp8_reduction_eligible(fp8, fp8, &[0, 2, 1, 3], 4));
    }

    #[test]
    fn completion_route_plan_prioritizes_low_remaining_rows_then_expert_reuse() -> Result<()> {
        let entries = [
            CompletionRoutePlanEntry {
                row_index: 0,
                expert_id: 1,
                intermediate_rows: 2048,
            },
            CompletionRoutePlanEntry {
                row_index: 1,
                expert_id: 1,
                intermediate_rows: 2048,
            },
            CompletionRoutePlanEntry {
                row_index: 1,
                expert_id: 2,
                intermediate_rows: 2048,
            },
            CompletionRoutePlanEntry {
                row_index: 2,
                expert_id: 2,
                intermediate_rows: 2048,
            },
            CompletionRoutePlanEntry {
                row_index: 3,
                expert_id: 2,
                intermediate_rows: 2048,
            },
            CompletionRoutePlanEntry {
                row_index: 3,
                expert_id: 3,
                intermediate_rows: 2048,
            },
            CompletionRoutePlanEntry {
                row_index: 4,
                expert_id: 3,
                intermediate_rows: 2048,
            },
            CompletionRoutePlanEntry {
                row_index: 4,
                expert_id: 4,
                intermediate_rows: 2048,
            },
            CompletionRoutePlanEntry {
                row_index: 4,
                expert_id: 5,
                intermediate_rows: 2048,
            },
        ];

        let groups = plan_completion_first_routes(&entries, 5, 256)?.groups;
        let experts = groups
            .iter()
            .map(|group| entries[group.route_indices[0]].expert_id)
            .collect::<Vec<_>>();
        assert_eq!(experts, vec![2, 1, 3, 4, 5]);
        assert_eq!(groups[0].completed_rows, vec![2]);
        assert_eq!(groups[1].completed_rows, vec![0, 1]);
        assert_eq!(groups[2].completed_rows, vec![3]);
        assert!(groups[3].completed_rows.is_empty());
        assert_eq!(groups[4].completed_rows, vec![4]);
        Ok(())
    }

    #[test]
    fn completion_route_plan_caps_expert_groups_at_256_rows() -> Result<()> {
        let entries = (0..300)
            .map(|row_index| CompletionRoutePlanEntry {
                row_index,
                expert_id: 7,
                intermediate_rows: 2048,
            })
            .collect::<Vec<_>>();

        let groups = plan_completion_first_routes(&entries, entries.len(), 256)?.groups;
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].route_indices.len(), 256);
        assert_eq!(groups[0].completed_rows.len(), 256);
        assert_eq!(groups[1].route_indices.len(), 44);
        assert_eq!(groups[1].completed_rows.len(), 44);
        Ok(())
    }

    #[test]
    fn streaming_completion_keeps_first_32_rows_then_coalesces_to_256() -> Result<()> {
        let indexed_slices = (0..32)
            .map(|slice_index| {
                (
                    slice_index * 2 + 1,
                    (slice_index * 32..slice_index * 32 + 32).collect(),
                )
            })
            .collect::<Vec<_>>();

        let (group_indices, response_slices) =
            coalesce_streaming_completion_slices(&indexed_slices, 32, 256)?;

        assert_eq!(group_indices, vec![1, 17, 33, 49, 63]);
        assert_eq!(
            response_slices.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![32, 256, 256, 256, 224]
        );
        assert_eq!(
            response_slices.into_iter().flatten().collect::<Vec<_>>(),
            (0..1024).collect::<Vec<_>>()
        );
        Ok(())
    }

    #[test]
    fn b12x_spark_direct_route_shape_gate_requires_contiguous_supported_rows() {
        assert!(b12x_spark_direct_route_shape_supported(
            16, 6144, 6144, 2048, 6144
        ));
        assert!(b12x_spark_direct_route_shape_supported(
            256, 6144, 6144, 2048, 6144
        ));
        assert!(!b12x_spark_direct_route_shape_supported(
            0, 6144, 6144, 2048, 6144
        ));
        assert!(!b12x_spark_direct_route_shape_supported(
            257, 6144, 6144, 2048, 6144
        ));
        assert!(!b12x_spark_direct_route_shape_supported(
            16, 6144, 6160, 2048, 6144
        ));
        assert!(!b12x_spark_direct_route_shape_supported(
            16, 6144, 6144, 2047, 6144
        ));
    }

    #[test]
    fn route_tensor_cache_reuses_loaded_projection_rows() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);
        let tempdir = tempfile::tempdir().unwrap();
        let catalog = tiny_route_catalog(tempdir.path(), &[3]);
        let route = ScoredRoute {
            expert_id: 0,
            score: 1.0,
            corrected_score: 1.0,
            normalized_weight: 1.0,
        };
        let mut cache = RouteTensorCache::default();

        let first =
            execute_nvfp4_route_cached(&catalog, 3, &[1.0_f32, 1.0], &route, 1, 1, &mut cache)
                .unwrap();
        let second =
            execute_nvfp4_route_cached(&catalog, 3, &[1.0_f32, 1.0], &route, 1, 1, &mut cache)
                .unwrap();

        assert_eq!(first.outputs, second.outputs);
        assert_eq!(first.kernel_backend, CPU_REFERENCE_NVFP4_ROUTE_BACKEND);
        assert_eq!(first.weight_bytes_read, second.weight_bytes_read);
        assert_eq!(
            first.quant_metadata_bytes_read,
            second.quant_metadata_bytes_read
        );
        let stats = cache.stats();
        assert_eq!(stats.entries, 3);
        assert_eq!(stats.projection_loads, 3);
        assert_eq!(stats.cache_hits, 3);
        assert_eq!(stats.projection_evictions, 0);
        assert_eq!(stats.active_layer, Some(3));
    }

    #[test]
    fn route_tensor_cache_reuses_bf16_source_scalar_metadata() {
        let tempdir = tempfile::tempdir().unwrap();
        let catalog = tiny_route_catalog(tempdir.path(), &[3]);
        let mut cache = RouteTensorCache::default();

        let (first, first_scale_2) =
            load_bf16_route_projection_source(&catalog, 3, 0, "gate_proj", 1, 2, &mut cache, false)
                .unwrap();
        let first_stats = cache.stats();
        let (second, second_scale_2) =
            load_bf16_route_projection_source(&catalog, 3, 0, "gate_proj", 1, 2, &mut cache, false)
                .unwrap();
        let second_stats = cache.stats();

        assert!(first.host.is_none());
        assert!(second.host.is_none());
        assert_eq!(first_scale_2, 1.0);
        assert_eq!(second_scale_2, first_scale_2);
        assert_eq!(first_stats.entries, 0);
        assert_eq!(first_stats.projection_loads, 0);
        assert_eq!(first_stats.scalar_metadata_entries, 1);
        assert_eq!(first_stats.scalar_metadata_loads, 1);
        assert_eq!(first_stats.scalar_metadata_cache_hits, 0);
        assert_eq!(second_stats.scalar_metadata_entries, 1);
        assert_eq!(second_stats.scalar_metadata_loads, 1);
        assert_eq!(second_stats.scalar_metadata_cache_hits, 1);
    }

    #[test]
    fn repeated_bf16_routes_reuse_request_projection_group() {
        let tempdir = tempfile::tempdir().unwrap();
        let catalog = tiny_route_catalog(tempdir.path(), &[3]);
        let route = ScoredRoute {
            expert_id: 0,
            score: 1.0,
            corrected_score: 1.0,
            normalized_weight: 1.0,
        };
        let mut cache = RouteTensorCache::default();
        let mut request_projection_groups: HashMap<
            Bf16RouteProjectionGroupKey,
            Bf16RouteProjections,
        > = HashMap::new();

        load_bf16_route_projections_for_group_cached(
            &catalog,
            3,
            &route,
            1,
            1,
            2,
            &mut cache,
            &mut request_projection_groups,
        )
        .unwrap();
        let first_stats = cache.stats();
        load_bf16_route_projections_for_group_cached(
            &catalog,
            3,
            &route,
            1,
            1,
            2,
            &mut cache,
            &mut request_projection_groups,
        )
        .unwrap();
        let second_stats = cache.stats();

        assert_eq!(request_projection_groups.len(), 1);
        assert_eq!(first_stats.scalar_metadata_loads, 3);
        assert_eq!(second_stats.scalar_metadata_loads, 3);
        assert_eq!(
            second_stats.scalar_metadata_cache_hits,
            first_stats.scalar_metadata_cache_hits
        );
    }

    #[test]
    fn repeated_bf16_routes_reuse_cross_request_projection_group() {
        let tempdir = tempfile::tempdir().unwrap();
        let catalog = tiny_route_catalog(tempdir.path(), &[3]);
        let route = ScoredRoute {
            expert_id: 0,
            score: 1.0,
            corrected_score: 1.0,
            normalized_weight: 1.0,
        };
        let mut cache = RouteTensorCache::default();
        let mut first_request_groups = HashMap::new();
        let mut second_request_groups = HashMap::new();

        let first = load_bf16_route_projections_for_group_cached(
            &catalog,
            3,
            &route,
            1,
            1,
            2,
            &mut cache,
            &mut first_request_groups,
        )
        .unwrap();
        let first_stats = cache.stats();
        let second = load_bf16_route_projections_for_group_cached(
            &catalog,
            3,
            &route,
            1,
            1,
            2,
            &mut cache,
            &mut second_request_groups,
        )
        .unwrap();
        let second_stats = cache.stats();

        assert_eq!(first.gate_scale_2, second.gate_scale_2);
        assert_eq!(first.up_scale_2, second.up_scale_2);
        assert_eq!(first.down_scale_2, second.down_scale_2);
        assert_eq!(cache.bf16_projection_groups.len(), 1);
        assert_eq!(first_request_groups.len(), 1);
        assert_eq!(second_request_groups.len(), 1);
        assert_eq!(first_stats.scalar_metadata_loads, 3);
        assert_eq!(second_stats.scalar_metadata_loads, 3);
        assert_eq!(
            second_stats.scalar_metadata_cache_hits,
            first_stats.scalar_metadata_cache_hits
        );
    }

    #[test]
    fn routed_quant_scalar_metadata_accepts_non_unity_input_scale() {
        let input_scale = fake_loaded_tensor("input_scale", DType::F32, 0.5_f32.to_le_bytes());
        let weight_scale_2 =
            fake_loaded_tensor("weight_scale_2", DType::F32, 1.0_f32.to_le_bytes());

        let metadata =
            routed_quant_scalar_metadata_from_loaded(&input_scale, &weight_scale_2).unwrap();

        assert_eq!(metadata.input_scale, 0.5);
        assert_eq!(metadata.weight_scale_2, 1.0);
    }

    #[test]
    fn route_tensor_cache_keeps_host_layers_resident() {
        let tempdir = tempfile::tempdir().unwrap();
        let catalog = tiny_route_catalog(tempdir.path(), &[3, 4]);
        let route = ScoredRoute {
            expert_id: 0,
            score: 1.0,
            corrected_score: 1.0,
            normalized_weight: 1.0,
        };
        let mut cache = RouteTensorCache::default();

        execute_nvfp4_route_cached(&catalog, 3, &[1.0_f32, 1.0], &route, 1, 1, &mut cache).unwrap();
        let layer3_stats = cache.stats();
        assert_eq!(layer3_stats.entries, 3);
        assert_eq!(layer3_stats.projection_loads, 3);
        assert_eq!(layer3_stats.projection_evictions, 0);
        assert_eq!(layer3_stats.active_layer, Some(3));

        execute_nvfp4_route_cached(&catalog, 4, &[1.0_f32, 1.0], &route, 1, 1, &mut cache).unwrap();
        let layer4_stats = cache.stats();
        assert_eq!(layer4_stats.entries, 6);
        assert_eq!(layer4_stats.projection_loads, 6);
        assert_eq!(layer4_stats.cache_hits, 0);
        assert_eq!(layer4_stats.projection_evictions, 0);
        assert_eq!(layer4_stats.active_layer, Some(4));

        execute_nvfp4_route_cached(&catalog, 4, &[1.0_f32, 1.0], &route, 1, 1, &mut cache).unwrap();
        let reused_layer4_stats = cache.stats();
        assert_eq!(reused_layer4_stats.entries, 6);
        assert_eq!(reused_layer4_stats.projection_loads, 6);
        assert_eq!(reused_layer4_stats.cache_hits, 3);
        assert_eq!(reused_layer4_stats.projection_evictions, 0);

        execute_nvfp4_route_cached(&catalog, 3, &[1.0_f32, 1.0], &route, 1, 1, &mut cache).unwrap();
        let layer3_again_stats = cache.stats();
        assert_eq!(layer3_again_stats.entries, 6);
        assert_eq!(layer3_again_stats.projection_loads, 6);
        assert_eq!(layer3_again_stats.cache_hits, 6);
        assert_eq!(layer3_again_stats.projection_evictions, 0);
        assert_eq!(layer3_again_stats.active_layer, Some(3));
    }

    #[test]
    fn retained_route_device_output_reuses_shape_pool() -> Result<()> {
        if native_library_path().is_none() || !cuda_reference_kernels_enabled() {
            return Ok(());
        }
        let first = device_bf16_output_uninitialized(
            1,
            32,
            CUDA_REFERENCE_NVFP4_ROUTE_BF16_ACCUMULATED_DEVICE_OUTPUT_BACKEND,
            "test retained route output pool",
        )?;
        let first_ptr = first.buffer().ptr;
        drop(first);
        let second = device_bf16_output_uninitialized(
            1,
            32,
            CUDA_REFERENCE_NVFP4_ROUTE_BF16_ACCUMULATED_DEVICE_OUTPUT_BACKEND,
            "test retained route output pool",
        )?;
        assert_eq!(second.buffer().ptr, first_ptr);
        Ok(())
    }

    #[test]
    fn route_bf16_accumulated_device_output_retains_zero_route_rows_when_cuda_enabled() -> Result<()>
    {
        if native_library_path().is_none() || !cuda_reference_kernels_enabled() {
            return Ok(());
        }
        let tempdir = tempfile::tempdir()?;
        let catalog = tiny_route_catalog(tempdir.path(), &[3]);
        let mut hidden = vec![0_u8; 2 * std::mem::size_of::<u16>()];
        f32_values_to_bf16_bytes(&[1.0, -2.0], &mut hidden);
        let row_routes: Vec<Vec<(ScoredRoute, usize)>> = vec![Vec::new()];
        let mut cache = RouteTensorCache::default();

        let execution = execute_nvfp4_route_rows_bf16_accumulated_cached_device_output(
            &catalog,
            3,
            &hidden,
            2,
            hidden.len(),
            &row_routes,
            2,
            &mut cache,
        )?;

        assert_eq!(
            execution.kernel_backend,
            CUDA_REFERENCE_NVFP4_ROUTE_BF16_ACCUMULATED_BACKEND
        );
        assert_eq!(
            execution.output_device.backend,
            CUDA_REFERENCE_NVFP4_ROUTE_BF16_ACCUMULATED_DEVICE_OUTPUT_BACKEND
        );
        assert_eq!(execution.output_device.rows, 1);
        assert_eq!(execution.output_device.values_per_row, 2);
        assert_eq!(execution.output_device.copy_to_host_bytes()?, vec![0_u8; 4]);
        Ok(())
    }

    #[test]
    fn route_bf16_accumulated_device_input_matches_host_input_when_cuda_enabled() -> Result<()> {
        if native_library_path().is_none() || !cuda_reference_kernels_enabled() {
            return Ok(());
        }
        let tempdir = tempfile::tempdir()?;
        let catalog = tiny_route_catalog(tempdir.path(), &[3]);
        let mut hidden = vec![0_u8; std::mem::size_of::<u16>()];
        f32_values_to_bf16_bytes(&[1.0], &mut hidden);
        let hidden_device =
            device_bf16_output_from_bf16_bytes(&hidden, 1, 1, "test route device hidden")?;
        let row_routes = vec![vec![(
            ScoredRoute {
                expert_id: 0,
                score: 1.0,
                corrected_score: 1.0,
                normalized_weight: 1.0,
            },
            1,
        )]];

        let mut host_cache = RouteTensorCache::default();
        let host_execution = execute_nvfp4_route_rows_bf16_accumulated_cached_device_output(
            &catalog,
            3,
            &hidden,
            1,
            hidden.len(),
            &row_routes,
            1,
            &mut host_cache,
        )?;
        let first_host_stream = host_cache
            .cuda
            .as_ref()
            .expect("host CUDA cache initialized")
            .stream
            .as_ptr();
        let second_host_execution = execute_nvfp4_route_rows_bf16_accumulated_cached_device_output(
            &catalog,
            3,
            &hidden,
            1,
            hidden.len(),
            &row_routes,
            1,
            &mut host_cache,
        )?;
        let second_host_stream = host_cache
            .cuda
            .as_ref()
            .expect("host CUDA cache retained")
            .stream
            .as_ptr();
        let mut device_cache = RouteTensorCache::default();
        let device_execution =
            execute_nvfp4_route_rows_bf16_accumulated_cached_device_input_device_output(
                &catalog,
                3,
                &hidden_device,
                Some(&hidden),
                1,
                hidden.len(),
                &row_routes,
                1,
                &mut device_cache,
            )?;

        assert_eq!(
            host_execution.kernel_backend,
            CUDA_REFERENCE_NVFP4_ROUTE_BF16_ACCUMULATED_BACKEND
        );
        assert_eq!(
            device_execution.kernel_backend,
            CUDA_REFERENCE_NVFP4_ROUTE_BF16_ACCUMULATED_DEVICE_INPUT_BACKEND
        );
        assert_eq!(
            host_execution.output_device.copy_to_host_bytes()?,
            device_execution.output_device.copy_to_host_bytes()?
        );
        assert_eq!(
            second_host_execution.output_device.copy_to_host_bytes()?,
            device_execution.output_device.copy_to_host_bytes()?
        );
        assert_eq!(first_host_stream, second_host_stream);
        assert_eq!(
            device_execution.output_device.backend,
            CUDA_REFERENCE_NVFP4_ROUTE_BF16_ACCUMULATED_DEVICE_OUTPUT_BACKEND
        );
        Ok(())
    }

    #[test]
    fn route_bf16_accumulated_batches_multiple_expert_groups_when_cuda_enabled() -> Result<()> {
        if native_library_path().is_none() || !cuda_reference_kernels_enabled() {
            return Ok(());
        }
        let tempdir = tempfile::tempdir()?;
        let catalog = tiny_route_catalog_for_experts(tempdir.path(), &[3], &[0, 1]);
        let mut hidden = vec![0_u8; std::mem::size_of::<u16>()];
        f32_values_to_bf16_bytes(&[1.0], &mut hidden);
        let row_routes = vec![vec![
            (
                ScoredRoute {
                    expert_id: 0,
                    score: 1.0,
                    corrected_score: 1.0,
                    normalized_weight: 1.0,
                },
                1,
            ),
            (
                ScoredRoute {
                    expert_id: 1,
                    score: 0.5,
                    corrected_score: 0.5,
                    normalized_weight: 0.5,
                },
                1,
            ),
        ]];
        let mut cache = RouteTensorCache::default();

        let execution = execute_nvfp4_route_rows_bf16_accumulated_cached_device_output(
            &catalog,
            3,
            &hidden,
            1,
            hidden.len(),
            &row_routes,
            1,
            &mut cache,
        )?;

        let output = execution.output_device.copy_to_host_bytes()?;
        assert_ne!(output, vec![0_u8; output.len()]);
        let stats = cache.stats();
        assert_eq!(stats.cuda_projection_entries, 6);
        assert_eq!(stats.cuda_projection_uploads, 6);
        Ok(())
    }

    #[test]
    fn route_bf16_accumulated_host_input_captures_single_row_cuda_graph_when_enabled() -> Result<()>
    {
        if native_library_path().is_none() || !cuda_reference_kernels_enabled() {
            return Ok(());
        }
        let _graph_override = route_cuda_graphs_test_override(true);
        let tempdir = tempfile::tempdir()?;
        let catalog = tiny_route_catalog(tempdir.path(), &[3]);
        let mut hidden = vec![0_u8; std::mem::size_of::<u16>()];
        f32_values_to_bf16_bytes(&[1.0], &mut hidden);
        let row_routes = vec![vec![(
            ScoredRoute {
                expert_id: 0,
                score: 1.0,
                corrected_score: 1.0,
                normalized_weight: 1.0,
            },
            1,
        )]];
        let mut cache = RouteTensorCache::default();

        let first = execute_nvfp4_route_rows_bf16_accumulated_cached(
            &catalog,
            3,
            &hidden,
            1,
            hidden.len(),
            &row_routes,
            1,
            &mut cache,
        )?;
        assert_eq!(
            first.kernel_backend,
            CUDA_REFERENCE_NVFP4_ROUTE_BF16_ACCUMULATED_BACKEND
        );
        assert_ne!(first.output_bf16, vec![0_u8; first.output_bf16.len()]);
        let first_stats = cache.stats();
        assert_eq!(first_stats.cuda_graph_entries, 1);
        assert_eq!(first_stats.cuda_graph_captures, 1);
        assert_eq!(first_stats.cuda_graph_launches, 1);

        let second = execute_nvfp4_route_rows_bf16_accumulated_cached(
            &catalog,
            3,
            &hidden,
            1,
            hidden.len(),
            &row_routes,
            1,
            &mut cache,
        )?;
        assert_eq!(second.output_bf16, first.output_bf16);
        let second_stats = cache.stats();
        assert_eq!(second_stats.cuda_graph_entries, 1);
        assert_eq!(second_stats.cuda_graph_captures, 1);
        assert_eq!(second_stats.cuda_graph_launches, 2);
        Ok(())
    }

    #[test]
    fn route_bf16_accumulated_device_input_allows_no_host_hidden_when_validation_disabled(
    ) -> Result<()> {
        if native_library_path().is_none() || !cuda_reference_kernels_enabled() {
            return Ok(());
        }
        let _route_validation_override = cuda_route_validation_test_override(false);

        let tempdir = tempfile::tempdir()?;
        let catalog = tiny_route_catalog(tempdir.path(), &[3]);
        let mut hidden = vec![0_u8; std::mem::size_of::<u16>()];
        f32_values_to_bf16_bytes(&[1.0], &mut hidden);
        let hidden_device =
            device_bf16_output_from_bf16_bytes(&hidden, 1, 1, "test route device hidden")?;
        let row_routes = vec![vec![(
            ScoredRoute {
                expert_id: 0,
                score: 1.0,
                corrected_score: 1.0,
                normalized_weight: 1.0,
            },
            1,
        )]];

        let mut cache = RouteTensorCache::default();
        let execution =
            execute_nvfp4_route_rows_bf16_accumulated_cached_device_input_device_output(
                &catalog,
                3,
                &hidden_device,
                None,
                1,
                hidden.len(),
                &row_routes,
                1,
                &mut cache,
            )?;

        assert_eq!(
            execution.kernel_backend,
            CUDA_REFERENCE_NVFP4_ROUTE_BF16_ACCUMULATED_DEVICE_INPUT_BACKEND
        );
        assert_eq!(
            execution.output_device.backend,
            CUDA_REFERENCE_NVFP4_ROUTE_BF16_ACCUMULATED_DEVICE_OUTPUT_BACKEND
        );
        assert_eq!(
            execution.output_device.copy_to_host_bytes()?.len(),
            hidden.len()
        );
        Ok(())
    }

    #[test]
    fn route_workspace_reuses_pinned_payload_buffers_when_native_available() -> Result<()> {
        let Some(path) = native_library_path() else {
            return Ok(());
        };
        let library = Arc::new(unsafe { NativeLibrary::load(path)? });
        let mut workspace = RouteCudaWorkspace::default();
        let buffer = OwnedDeviceAllocation::new(Arc::clone(&library), 4, "test route payload")?;
        let cuda_stream = std::ptr::null_mut();
        let first = workspace.stage_accumulation_payloads(
            Arc::clone(&library),
            &[0x01_u8, 0x02, 0x03, 0x04],
            &[0x05_u8, 0x06, 0x07, 0x08],
            &[0x09_u8, 0x0a, 0x0b, 0x0c],
        )?;
        let first_hidden_ptr = first.hidden.ptr;
        let first_index_ptr = first.scatter_index.ptr;
        let first_weight_ptr = first.route_weights.ptr;
        let second_payload = [0x11_u8, 0x22, 0x33, 0x44];
        let second = workspace.stage_accumulation_payloads(
            Arc::clone(&library),
            &second_payload,
            &[0x55_u8, 0x66, 0x77, 0x88],
            &[0x99_u8, 0xaa, 0xbb, 0xcc],
        )?;
        assert_eq!(first_hidden_ptr, second.hidden.ptr);
        assert_eq!(first_index_ptr, second.scatter_index.ptr);
        assert_eq!(first_weight_ptr, second.route_weights.ptr);

        unsafe {
            library.copy_host_buffer_h2d_async(
                buffer.buffer(),
                second.hidden,
                second_payload.len(),
                cuda_stream,
            )?;
            library.cuda_stream_synchronize(cuda_stream)?;
        }
        let mut out = vec![0_u8; second_payload.len()];
        library.copy_d2h(&mut out, buffer.buffer())?;

        assert_eq!(out, second_payload);
        Ok(())
    }

    fn fake_loaded_tensor<const N: usize>(
        suffix: &str,
        dtype: DType,
        bytes: [u8; N],
    ) -> LoadedTensor {
        LoadedTensor {
            info: fake_tensor_info(suffix, dtype, Vec::new(), N),
            source_path: PathBuf::from("fake-route.bin"),
            bytes: bytes.to_vec(),
            elapsed_micros: 0,
            sha256: String::new(),
        }
    }

    fn fake_tensor_info(
        suffix: &str,
        dtype: DType,
        shape: Vec<usize>,
        byte_length: usize,
    ) -> TensorInfo {
        TensorInfo {
            name: format!("model.layers.3.mlp.experts.0.gate_proj.{suffix}"),
            file: "fake-route.bin".to_owned(),
            dtype,
            shape,
            byte_offset: 0,
            byte_length: byte_length as u64,
            role: TensorRole::RoutedExpert,
            layer_id: Some(3),
            expert_id: Some(0),
            is_quantization_metadata: suffix != "weight",
        }
    }

    fn tiny_route_catalog(tempdir: &Path, layers: &[usize]) -> TensorCatalog {
        tiny_route_catalog_for_experts(tempdir, layers, &[0])
    }

    fn tiny_route_catalog_for_experts(
        tempdir: &Path,
        layers: &[usize],
        experts: &[usize],
    ) -> TensorCatalog {
        let shard_path = tempdir.join("route.bin");
        let mut shard_bytes = Vec::new();
        let mut tensors = Vec::new();
        for layer_id in layers {
            for expert_id in experts {
                for projection in ["gate_proj", "up_proj", "down_proj"] {
                    push_tensor(
                        &mut shard_bytes,
                        &mut tensors,
                        *layer_id,
                        *expert_id,
                        projection,
                        "weight",
                        DType::U8,
                        vec![1, 1],
                        &[0xaa],
                    );
                    push_tensor(
                        &mut shard_bytes,
                        &mut tensors,
                        *layer_id,
                        *expert_id,
                        projection,
                        "weight_scale",
                        DType::F8E4M3,
                        vec![1, 1],
                        &[0x38],
                    );
                    push_tensor(
                        &mut shard_bytes,
                        &mut tensors,
                        *layer_id,
                        *expert_id,
                        projection,
                        "input_scale",
                        DType::F32,
                        Vec::new(),
                        &1.0_f32.to_le_bytes(),
                    );
                    push_tensor(
                        &mut shard_bytes,
                        &mut tensors,
                        *layer_id,
                        *expert_id,
                        projection,
                        "weight_scale_2",
                        DType::F32,
                        Vec::new(),
                        &1.0_f32.to_le_bytes(),
                    );
                }
            }
        }
        File::create(&shard_path)
            .unwrap()
            .write_all(&shard_bytes)
            .unwrap();
        TensorCatalog {
            model_id: "test/model".to_owned(),
            snapshot_path: tempdir.display().to_string(),
            facts: ModelFacts::default(),
            tensors,
        }
    }

    fn push_tensor(
        shard_bytes: &mut Vec<u8>,
        tensors: &mut Vec<TensorInfo>,
        layer_id: usize,
        expert_id: usize,
        projection: &str,
        suffix: &str,
        dtype: DType,
        shape: Vec<usize>,
        bytes: &[u8],
    ) {
        let byte_offset = shard_bytes.len() as u64;
        shard_bytes.extend_from_slice(bytes);
        tensors.push(TensorInfo {
            name: format!("model.layers.{layer_id}.mlp.experts.{expert_id}.{projection}.{suffix}"),
            file: "route.bin".to_owned(),
            dtype,
            shape,
            byte_offset,
            byte_length: bytes.len() as u64,
            role: TensorRole::RoutedExpert,
            layer_id: Some(layer_id as u32),
            expert_id: Some(expert_id as u32),
            is_quantization_metadata: suffix != "weight",
        });
    }
}
