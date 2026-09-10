//! Native CSA2 projection and immutable-state ratio-two pooling.
use crate::{Ds41rtDeviceBuffer, NativeLibrary};
use anyhow::{ensure, Result};
use std::ffi::c_void;
type Create = unsafe extern "C" fn(*mut c_void, u64, *mut *mut c_void) -> i32;
type Destroy = unsafe extern "C" fn(*mut c_void) -> i32;
type Project = unsafe extern "C" fn(
    *mut c_void,
    *const u16,
    *const u16,
    *mut c_void,
    i32,
    i32,
    *mut c_void,
) -> i32;
type IndexProject =
    unsafe extern "C" fn(*mut c_void, *const u16, *const u16, *mut u16, i32, *mut c_void) -> i32;
type IndexPack = unsafe extern "C" fn(*const u16, *mut u8, *mut u8, i32, *mut c_void) -> i32;
type IndexStore = unsafe extern "C" fn(
    *const u8,
    *const u8,
    *const u64,
    *mut u8,
    *mut u8,
    i32,
    u64,
    *mut c_void,
) -> i32;
type Pool = unsafe extern "C" fn(
    *const f32,
    *const f32,
    *const f32,
    *const f32,
    *const u64,
    *const u16,
    *mut u16,
    i32,
    i32,
    *mut c_void,
) -> i32;
pub struct V41Compressor<'a> {
    _library: &'a NativeLibrary,
    handle: *mut c_void,
    destroy: Destroy,
    project: Project,
    index_project: IndexProject,
    index_pack: IndexPack,
    index_store: IndexStore,
    pool: Pool,
}
fn buffer(b: Ds41rtDeviceBuffer, bytes: usize) -> Result<()> {
    ensure!(
        !b.ptr.is_null() && b.bytes >= bytes,
        "compressor buffer is null or undersized"
    );
    Ok(())
}
impl NativeLibrary {
    /// # Safety
    /// Workspace stays on the current device, exclusively used and live through
    /// handle/graph destruction. Drain all work before dropping this handle.
    pub unsafe fn v41_compressor(
        &self,
        workspace: Ds41rtDeviceBuffer,
    ) -> Result<V41Compressor<'_>> {
        buffer(workspace, V41Compressor::WORKSPACE_BYTES)?;
        let create: Create = unsafe { *self.lib.get(b"ds41rt_v41_compressor_create")? };
        let destroy: Destroy = unsafe { *self.lib.get(b"ds41rt_v41_compressor_destroy")? };
        let project: Project = unsafe { *self.lib.get(b"ds41rt_v41_compressor_project")? };
        let index_project: IndexProject =
            unsafe { *self.lib.get(b"ds41rt_v41_index_key_project")? };
        let index_pack: IndexPack = unsafe { *self.lib.get(b"ds41rt_v41_index_pack")? };
        let index_store: IndexStore = unsafe { *self.lib.get(b"ds41rt_v41_index_store")? };
        let pool: Pool = unsafe { *self.lib.get(b"ds41rt_v41_compressor_pool")? };
        let mut handle = std::ptr::null_mut();
        let status = unsafe { create(workspace.ptr, workspace.bytes as u64, &mut handle) };
        ensure!(status == 0, "native compressor create status {status}");
        Ok(V41Compressor {
            _library: self,
            handle,
            destroy,
            project,
            index_project,
            index_pack,
            index_store,
            pool,
        })
    }
}
impl V41Compressor<'_> {
    /// # Safety
    /// All disjoint spans are live on the stream device. Each in-range U64
    /// destination is unique and owned by the committing request; other values
    /// skip writes. Validate acceptance and reserve physical pages before launch.
    pub unsafe fn index_store(
        &self,
        packed: Ds41rtDeviceBuffer,
        scales: Ds41rtDeviceBuffer,
        destinations: Ds41rtDeviceBuffer,
        cache: Ds41rtDeviceBuffer,
        cache_scales: Ds41rtDeviceBuffer,
        rows: usize,
        capacity: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            (1..=4096).contains(&rows) && (1..=16 * 1048576).contains(&capacity),
            "invalid index store shape"
        );
        buffer(packed, rows * 64)?;
        buffer(scales, rows * 4)?;
        buffer(destinations, rows * 8)?;
        buffer(cache, capacity * 64)?;
        buffer(cache_scales, capacity * 4)?;
        let status = unsafe {
            (self.index_store)(
                packed.ptr.cast(),
                scales.ptr.cast(),
                destinations.ptr.cast(),
                cache.ptr.cast(),
                cache_scales.ptr.cast(),
                rows as i32,
                capacity as u64,
                stream,
            )
        };
        ensure!(status == 0, "native index store status {status}");
        Ok(())
    }
    /// # Safety
    /// Finite BF16 vectors [rows,128] and disjoint packed [rows,64] / scale
    /// [rows,4] byte outputs are live on the stream device. Flatten query heads
    /// into rows when needed. This produces proposals, not committed cache rows.
    pub unsafe fn index_pack(
        &self,
        input: Ds41rtDeviceBuffer,
        packed: Ds41rtDeviceBuffer,
        scales: Ds41rtDeviceBuffer,
        rows: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!((1..=131072).contains(&rows), "invalid index packing rows");
        buffer(input, rows * 256)?;
        buffer(packed, rows * 64)?;
        buffer(scales, rows * 4)?;
        let status = unsafe {
            (self.index_pack)(
                input.ptr.cast(),
                packed.ptr.cast(),
                scales.ptr.cast(),
                rows as i32,
                stream,
            )
        };
        ensure!(status == 0, "native index packing status {status}");
        Ok(())
    }
    /// # Safety
    /// Initialized unrotated BF16 latents [rows,512], weight [128,512] and
    /// disjoint output [rows,128] are live on the handle's device. Serialize use
    /// with all other operations sharing the handle/workspace.
    pub unsafe fn index_project(
        &self,
        input: Ds41rtDeviceBuffer,
        weight: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        rows: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!((1..=4096).contains(&rows), "invalid index projection rows");
        buffer(input, rows * 1024)?;
        buffer(weight, 128 * 512 * 2)?;
        buffer(output, rows * 256)?;
        let status = unsafe {
            (self.index_project)(
                self.handle,
                input.ptr.cast(),
                weight.ptr.cast(),
                output.ptr.cast(),
                rows as i32,
                stream,
            )
        };
        ensure!(status == 0, "native index projection status {status}");
        Ok(())
    }

    pub const WORKSPACE_BYTES: usize = 4 * 1024 * 1024;
    /// # Safety
    /// BF16 input [rows,5120] and weight [512,5120] are initialized on this
    /// device. Output is BF16 for ratio one, FP32 for ratio two. Buffers and
    /// exclusive workspace remain live through completion, with disjoint output.
    pub unsafe fn project(
        &self,
        input: Ds41rtDeviceBuffer,
        weight: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        rows: usize,
        ratio: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            (1..=4096).contains(&rows) && (1..=2).contains(&ratio),
            "invalid compressor projection shape"
        );
        buffer(input, rows * 10240)?;
        buffer(weight, 512 * 10240)?;
        buffer(output, rows * 512 * if ratio == 2 { 4 } else { 2 })?;
        let status = unsafe {
            (self.project)(
                self.handle,
                input.ptr.cast(),
                weight.ptr.cast(),
                output.ptr,
                rows as i32,
                ratio as i32,
                stream,
            )
        };
        ensure!(status == 0, "native compressor projection status {status}");
        Ok(())
    }
    /// # Safety
    /// FP32 projections [rows,512], committed pending values [slots,512], U64
    /// predecessors [rows], BF16 weight [512] and output [rows,512] are live on
    /// the stream device. Output is disjoint. A completed row references its
    /// chronological predecessor: pending slot, or slots + earlier input row.
    /// Incomplete rows use u64::MAX. Validate request leases and causal positions
    /// before preparing descriptors; commit only accepted projections afterward.
    pub unsafe fn pool(
        &self,
        kv: Ds41rtDeviceBuffer,
        scores: Ds41rtDeviceBuffer,
        pending_kv: Ds41rtDeviceBuffer,
        pending_scores: Ds41rtDeviceBuffer,
        predecessors: Ds41rtDeviceBuffer,
        norm_weight: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        rows: usize,
        slots: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            (1..=4096).contains(&rows) && (1..=16).contains(&slots),
            "invalid compressor pooling shape"
        );
        for b in [kv, scores] {
            buffer(b, rows * 2048)?;
        }
        for b in [pending_kv, pending_scores] {
            buffer(b, slots * 2048)?;
        }
        buffer(predecessors, rows * 8)?;
        buffer(norm_weight, 1024)?;
        buffer(output, rows * 1024)?;
        let status = unsafe {
            (self.pool)(
                kv.ptr.cast(),
                scores.ptr.cast(),
                pending_kv.ptr.cast(),
                pending_scores.ptr.cast(),
                predecessors.ptr.cast(),
                norm_weight.ptr.cast(),
                output.ptr.cast(),
                rows as i32,
                slots as i32,
                stream,
            )
        };
        ensure!(status == 0, "native compressor pool status {status}");
        Ok(())
    }
}
impl Drop for V41Compressor<'_> {
    fn drop(&mut self) {
        let status = unsafe { (self.destroy)(self.handle) };
        if status != 0 {
            eprintln!("native compressor destruction status {status}");
        }
    }
}
