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
type PackedSizesFn = unsafe extern "C" fn(u32, *mut u64) -> i32;
type PackFn = unsafe extern "C" fn(*const *const u8, *const *mut u8, u32, *mut c_void) -> i32;

/// Per-expert checkpoint staging avoids a second full layer of logical weights.
pub struct V41ExpertPacker<'a> {
    _library: &'a NativeLibrary,
    pack: PackFn,
    intermediate: u32,
    bytes: [u64; 4],
}

impl V41ExpertPacker<'_> {
    /// Per-expert byte strides: W13, W13 scales, W2, W2 scales.
    pub fn packed_bytes(&self) -> [u64; 4] {
        self.bytes
    }

    /// # Safety
    /// Sources must be contiguous native CUDA W1,W3,W2,S1,S3,S2 for one expert,
    /// with hidden size 5120 and this packer's logical intermediate size.
    /// Destinations must have the advertised byte sizes and 16-byte alignment,
    /// be mutually disjoint and not overlap any source, on the same device.
    /// All buffers must remain valid through stream completion; order input
    /// copies before this operation and expert execution after it.
    pub unsafe fn pack(
        &self,
        sources: [*const u8; 6],
        destinations: [*mut u8; 4],
        stream: *mut c_void,
    ) -> Result<()> {
        let status = unsafe {
            (self.pack)(
                sources.as_ptr(),
                destinations.as_ptr(),
                self.intermediate,
                stream,
            )
        };
        ensure!(
            status == 0,
            "V4.1 expert packing failed with CUDA status {status}"
        );
        Ok(())
    }
}
type BindScratchFn = unsafe extern "C" fn(*mut c_void, *mut c_void, u64, *mut *mut c_void) -> i32;
type InitScratchFn = unsafe extern "C" fn(*mut c_void, *mut c_void, u64, *mut c_void) -> i32;
type ReduceFn = unsafe extern "C" fn(
    *const *const f32,
    *const u16,
    *mut u16,
    u32,
    u32,
    u32,
    *mut c_void,
) -> i32;

/// Preloaded, allocation-free reduction entry point for native expert outputs.
pub struct V41RouteReducer<'a> {
    _library: &'a NativeLibrary,
    reduce: ReduceFn,
}

impl V41RouteReducer<'_> {
    /// # Safety
    /// Active planes must be contiguous CUDA FP32 [rows,topk,5120] in identical
    /// route order; output and optional shared must be CUDA BF16 [rows,5120].
    /// All storage must be on the current device and remain valid through stream
    /// completion and graph replays, with writes ordered before this operation.
    /// Output must not overlap planes; shared may alias output only exactly.
    /// Unused plane slots must be null. The library must outlive captured graphs.
    pub unsafe fn launch(
        &self,
        planes: [*const f32; 4],
        shared: *const u16,
        output: *mut u16,
        rows: u32,
        ranks: u32,
        topk: u32,
        stream: *mut c_void,
    ) -> Result<()> {
        let status =
            unsafe { (self.reduce)(planes.as_ptr(), shared, output, rows, ranks, topk, stream) };
        ensure!(
            status == 0,
            "V4.1 route reduction failed with CUDA status {status}"
        );
        Ok(())
    }
}

pub struct V41ExpertKernel<'a> {
    // Keep all exported code and its CUDA modules loaded until this handle drops.
    _library: &'a NativeLibrary,
    handle: NonNull<c_void>,
    launch: LaunchFn,
    bind_scratch: BindScratchFn,
    initialize_scratch: InitScratchFn,
    info: V41ExpertInfo,
}

impl NativeLibrary {
    pub fn v41_expert_packer(&self, intermediate: u32) -> Result<V41ExpertPacker<'_>> {
        let sizes = unsafe {
            self.lib
                .get::<PackedSizesFn>(b"ds41rt_v41_expert_packed_sizes")?
        };
        let pack = unsafe { *self.lib.get::<PackFn>(b"ds41rt_v41_pack_expert_async")? };
        let mut bytes = [0; 4];
        let status = unsafe { sizes(intermediate, bytes.as_mut_ptr()) };
        ensure!(
            status == 0,
            "V4.1 packed sizes failed with CUDA status {status}"
        );
        Ok(V41ExpertPacker {
            _library: self,
            pack,
            intermediate,
            bytes,
        })
    }

    pub fn v41_route_reducer(&self) -> Result<V41RouteReducer<'_>> {
        let reduce = unsafe {
            *self
                .lib
                .get::<ReduceFn>(b"ds41rt_v41_reduce_routes_async")?
        };
        Ok(V41RouteReducer {
            _library: self,
            reduce,
        })
    }

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
        let bind_scratch = unsafe {
            *self
                .lib
                .get::<BindScratchFn>(b"ds41rt_v41_expert_bind_scratch")?
        };
        let initialize_scratch = unsafe {
            *self
                .lib
                .get::<InitScratchFn>(b"ds41rt_v41_expert_initialize_scratch_async")?
        };
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
            bind_scratch,
            initialize_scratch,
            info,
        })
    }
}

impl V41ExpertKernel<'_> {
    pub fn info(&self) -> &V41ExpertInfo {
        &self.info
    }

    /// Bind the exported scratch views; external tensor slots remain unchanged.
    /// # Safety
    /// Storage must be a live, aligned CUDA allocation of at least `bytes` bytes.
    /// The resulting pointers borrow storage and do not extend its lifetime.
    pub unsafe fn bind_scratch(
        &self,
        storage: *mut c_void,
        bytes: u64,
        tensors: &mut [*mut c_void; V41_EXPERT_POINTER_COUNT],
    ) -> Result<()> {
        let status = unsafe {
            (self.bind_scratch)(self.handle.as_ptr(), storage, bytes, tensors.as_mut_ptr())
        };
        ensure!(
            status == 0,
            "V4.1 scratch binding failed with CUDA status {status}"
        );
        Ok(())
    }

    /// Initialize the native recipe's scratch once before use and graph capture.
    /// # Safety
    /// Storage must be exclusively owned CUDA memory of at least `bytes` bytes
    /// on this kernel's device; it must remain alive until stream completion.
    /// Order this initialization before every launch that uses the storage.
    pub unsafe fn initialize_scratch(
        &self,
        storage: *mut c_void,
        bytes: u64,
        stream: *mut c_void,
    ) -> Result<()> {
        let status =
            unsafe { (self.initialize_scratch)(self.handle.as_ptr(), storage, bytes, stream) };
        ensure!(
            status == 0,
            "V4.1 scratch initialization failed with CUDA status {status}"
        );
        Ok(())
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
