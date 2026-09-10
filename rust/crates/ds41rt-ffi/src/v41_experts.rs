//! Native V4.1 expert AOT launch handles; weight/scratch ownership stays with the caller.
use crate::NativeLibrary;
use anyhow::{ensure, Context, Result};
use std::ffi::c_void;
use std::ptr::NonNull;

pub const V41_EXPERT_POINTER_COUNT: usize = 44;

/// Pointer order is checked against generated C headers by the AOT exporter.
#[repr(usize)]
#[derive(Debug, Clone, Copy)]
pub enum V41ExpertPointer {
    Hidden = 0,
    RouteIds,
    RouteWeights,
    PackedA,
    Sfa,
    PackedAStorage,
    ScaleStorage,
    Intermediate,
    BarrierCount,
    BarrierEpoch,
    PairHead,
    ProducersDoneCount,
    AllWorkPublished,
    TaskHead,
    TaskTail,
    TaskReady,
    TaskExpert,
    TaskMTile,
    TaskSliceBegin,
    TaskSliceCount,
    TaskValidRows,
    TileWriteCount,
    WeightW13,
    ScaleW13,
    WeightDown,
    ScaleDown,
    ScaleW13Mx,
    ScaleDownMx,
    ResidualW13,
    ResidualDown,
    W13Repacked,
    W13ScaleRepacked,
    DownRepacked,
    DownScaleRepacked,
    RowCounts,
    ExpertWriteRows,
    ExpertTileBase,
    InputGlobalScale,
    Alpha,
    DownAlpha,
    GlobalScale,
    RoutePartials,
    TokenMap,
    TokenWeights,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct V41ExpertInfo {
    pub abi_version: u32,
    pub role: u32,
    pub experts: u32,
    pub hidden_size: u32,
    pub logical_intermediate: u32,
    pub kernel_intermediate: u32,
    pub topk: u32,
    pub capacity_rows: u32,
    pub scratch_bytes: u64,
    pub max_rows: i32,
    pub rows_padded: i32,
    pub max_tasks: i32,
    pub max_phys_tiles: i32,
    pub max_active_clusters: i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct V41ExpertLaunchArgs {
    pub tensors: [*mut c_void; V41_EXPERT_POINTER_COUNT],
    pub num_tokens: i32,
    pub max_rows: i32,
    pub scatter_rows: i32,
    pub rows_padded: i32,
    pub max_tasks: i32,
    pub max_phys_tiles: i32,
    pub max_active_clusters: i32,
    pub stream: *mut c_void,
}

impl V41ExpertLaunchArgs {
    pub fn new(
        info: &V41ExpertInfo,
        tensors: [*mut c_void; V41_EXPERT_POINTER_COUNT],
        rows: u32,
        stream: *mut c_void,
    ) -> Result<Self> {
        ensure!(
            rows > 0 && rows <= info.capacity_rows,
            "expert launch exceeds planned token capacity"
        );
        ensure!(
            tensors.iter().all(|pointer| !pointer.is_null()),
            "expert launch contains a null tensor slot"
        );
        Ok(Self {
            tensors,
            num_tokens: i32::try_from(rows)?,
            max_rows: info.max_rows,
            scatter_rows: i32::try_from(
                rows.checked_mul(info.topk)
                    .context("route count overflow")?,
            )?,
            rows_padded: info.rows_padded,
            max_tasks: info.max_tasks,
            max_phys_tiles: info.max_phys_tiles,
            max_active_clusters: info.max_active_clusters,
            stream,
        })
    }
}

type InfoFn = unsafe extern "C" fn(i32, *mut V41ExpertInfo) -> i32;
type InitializeFn = unsafe extern "C" fn(i32, *mut *mut c_void) -> i32;
type LaunchFn = unsafe extern "C" fn(*mut c_void, *const V41ExpertLaunchArgs) -> i32;

pub struct V41ExpertKernel<'a> {
    // Keep all exported code and its CUDA modules loaded until this handle drops.
    _library: &'a NativeLibrary,
    handle: NonNull<c_void>,
    launch: LaunchFn,
    info: V41ExpertInfo,
}

impl NativeLibrary {
    pub fn v41_expert_info(&self, capacity: u32) -> Result<V41ExpertInfo> {
        let function = unsafe { self.lib.get::<InfoFn>(b"ds41rt_v41_expert_info") }
            .context("native library must be built with DS41RT_ENABLE_V41_EXPERT_AOT=ON")?;
        let mut info = V41ExpertInfo::default();
        let status = unsafe { function(i32::try_from(capacity)?, &mut info) };
        ensure!(
            status == 0,
            "V4.1 expert metadata failed with CUDA status {status}"
        );
        ensure!(
            info.abi_version == 1 && info.hidden_size == 5120,
            "unsupported V4.1 native expert ABI"
        );
        let expected = match info.role {
            0 => (128, 2304, 2304, 3),
            1 => (384, 576, 640, 6),
            _ => anyhow::bail!("unknown V4.1 expert role {}", info.role),
        };
        ensure!(
            (
                info.experts,
                info.logical_intermediate,
                info.kernel_intermediate,
                info.topk
            ) == expected,
            "native V4.1 expert geometry does not match the official checkpoint"
        );
        ensure!(
            info.capacity_rows == capacity && info.scratch_bytes > 0,
            "native expert capacity does not match requested variant"
        );
        Ok(info)
    }

    /// Load the chosen variant on the current CUDA device before graph capture.
    pub fn v41_expert_kernel(&self, capacity: u32) -> Result<V41ExpertKernel<'_>> {
        let info = self.v41_expert_info(capacity)?;
        let initialize = unsafe {
            self.lib
                .get::<InitializeFn>(b"ds41rt_v41_expert_initialize")?
        };
        let launch = unsafe { *self.lib.get::<LaunchFn>(b"ds41rt_v41_expert_launch")? };
        let mut handle = std::ptr::null_mut();
        let status = unsafe { initialize(i32::try_from(capacity)?, &mut handle) };
        ensure!(
            status == 0,
            "V4.1 expert initialization failed with CUDA status {status}"
        );
        Ok(V41ExpertKernel {
            _library: self,
            handle: NonNull::new(handle).context("native expert returned a null kernel handle")?,
            launch,
            info,
        })
    }
}

impl V41ExpertKernel<'_> {
    pub fn info(&self) -> &V41ExpertInfo {
        &self.info
    }

    /// # Safety
    /// Every slot must reference correctly typed/packed CUDA storage of the
    /// capacity described by this variant's AOT manifest on its initialized device;
    /// scratch must be initialized and exclusively owned for this launch, and all
    /// buffers plus this kernel/library must remain alive through completion and
    /// every captured graph replay, with stream ordering enforced by the caller.
    pub unsafe fn launch(&self, args: &V41ExpertLaunchArgs) -> Result<()> {
        let status = unsafe { (self.launch)(self.handle.as_ptr(), args) };
        ensure!(
            status == 0,
            "V4.1 expert launch failed with CUDA status {status}"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_abi_layout_and_capacity_checks() {
        assert_eq!(std::mem::size_of::<V41ExpertInfo>(), 64);
        assert_eq!(std::mem::offset_of!(V41ExpertInfo, scratch_bytes), 32);
        assert_eq!(std::mem::size_of::<V41ExpertLaunchArgs>(), 392);
        assert_eq!(std::mem::offset_of!(V41ExpertLaunchArgs, stream), 384);
        assert_eq!(
            V41ExpertPointer::TokenWeights as usize + 1,
            V41_EXPERT_POINTER_COUNT
        );
        let info = V41ExpertInfo {
            capacity_rows: 16,
            topk: 6,
            ..Default::default()
        };
        let slots = [NonNull::<u8>::dangling().as_ptr().cast(); 44];
        assert!(V41ExpertLaunchArgs::new(&info, slots, 17, std::ptr::null_mut()).is_err());
        assert!(V41ExpertLaunchArgs::new(&info, slots, 0, std::ptr::null_mut()).is_err());
        assert!(V41ExpertLaunchArgs::new(
            &info,
            [std::ptr::null_mut(); 44],
            16,
            std::ptr::null_mut()
        )
        .is_err());
        assert_eq!(
            V41ExpertLaunchArgs::new(&info, slots, 16, std::ptr::null_mut())
                .unwrap()
                .scatter_rows,
            96
        );
    }
}
