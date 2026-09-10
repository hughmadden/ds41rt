use crate::{Ds41rtDeviceBuffer, NativeLibrary};
use anyhow::{ensure, Result};
use std::ffi::c_void;
type Create = unsafe extern "C" fn(*mut c_void, u64, *mut *mut c_void) -> i32;
type Destroy = unsafe extern "C" fn(*mut c_void) -> i32;
type Launch =
    unsafe extern "C" fn(*mut c_void, *const u16, *const u16, *mut u16, i32, *mut c_void) -> i32;
pub const V41_GROUPED_OUTPUT_WORKSPACE: usize = 4 * 1024 * 1024;
pub struct V41GroupedOutput<'a> {
    _library: &'a NativeLibrary,
    handle: *mut c_void,
    launch: Launch,
    destroy: Destroy,
}
impl NativeLibrary {
    /// # Safety
    /// Native FP8 weights and UE8 K32/N32 scales are initialized on the stream
    /// device; BF16 output is disjoint and buffers live through completion.
    pub unsafe fn v41_grouped_output_dequant(
        &self,
        input: Ds41rtDeviceBuffer,
        scales: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        stream: *mut c_void,
    ) -> Result<()> {
        for (value, bytes) in [(input, 33554432), (scales, 32768), (output, 67108864)] {
            ensure!(
                !value.ptr.is_null() && value.bytes >= bytes,
                "invalid grouped dequant buffer"
            );
        }
        type Dequant = unsafe extern "C" fn(*const u8, *const u8, *mut u16, *mut c_void) -> i32;
        let launch = unsafe {
            *self
                .lib
                .get::<Dequant>(b"ds41rt_v41_grouped_output_dequant")?
        };
        let status = unsafe {
            launch(
                input.ptr.cast(),
                scales.ptr.cast(),
                output.ptr.cast(),
                stream,
            )
        };
        ensure!(status == 0, "grouped dequant CUDA status {status}");
        Ok(())
    }
    /// # Safety
    /// Workspace is distinct, on the current device, and lives through handle
    /// destruction and all queued operations/graphs; serialize handle use.
    pub unsafe fn v41_grouped_output(
        &self,
        workspace: Ds41rtDeviceBuffer,
    ) -> Result<V41GroupedOutput<'_>> {
        ensure!(
            !workspace.ptr.is_null() && workspace.bytes >= V41_GROUPED_OUTPUT_WORKSPACE,
            "invalid grouped output workspace"
        );
        let create = unsafe {
            *self
                .lib
                .get::<Create>(b"ds41rt_v41_grouped_output_create")?
        };
        let launch = unsafe {
            *self
                .lib
                .get::<Launch>(b"ds41rt_v41_grouped_output_launch")?
        };
        let destroy = unsafe {
            *self
                .lib
                .get::<Destroy>(b"ds41rt_v41_grouped_output_destroy")?
        };
        let mut handle = std::ptr::null_mut();
        let status = unsafe { create(workspace.ptr, workspace.bytes as u64, &mut handle) };
        ensure!(
            status == 0 && !handle.is_null(),
            "grouped output initialization status {status}"
        );
        Ok(V41GroupedOutput {
            _library: self,
            handle,
            launch,
            destroy,
        })
    }
}
impl V41GroupedOutput<'_> {
    /// # Safety
    /// Finite BF16 input [rows,8,4096], weight [8,1024,4096], output [rows,8,1024]
    /// are on the handle/stream device. Output and workspace are disjoint from
    /// other buffers. Keep all storage live and serialize use through completion.
    pub unsafe fn launch(
        &mut self,
        input: Ds41rtDeviceBuffer,
        weight: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        rows: u32,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!((1..=4096).contains(&rows), "invalid grouped output rows");
        for (value, bytes) in [
            (input, rows as usize * 65536),
            (weight, 67108864),
            (output, rows as usize * 16384),
        ] {
            ensure!(
                !value.ptr.is_null() && value.bytes >= bytes,
                "invalid grouped output buffer"
            );
        }
        let status = unsafe {
            (self.launch)(
                self.handle,
                input.ptr.cast(),
                weight.ptr.cast(),
                output.ptr.cast(),
                rows as i32,
                stream,
            )
        };
        ensure!(status == 0, "grouped output launch status {status}");
        Ok(())
    }
}
impl Drop for V41GroupedOutput<'_> {
    fn drop(&mut self) {
        let status = unsafe { (self.destroy)(self.handle) };
        if status != 0 {
            eprintln!("grouped output destruction status {status}");
        }
    }
}
