//! Pre-resolved operations used by asynchronous dual-device chains.
use crate::{Ds41rtDeviceBuffer, NativeLibrary};
use anyhow::{ensure, Result};
use std::ffi::c_void;

type AddFn = unsafe extern "C" fn(*const u16, *const u16, *mut u16, usize, *mut c_void) -> i32;
pub struct V41Bf16Add<'a> { _library: &'a NativeLibrary, launch: AddFn }
impl NativeLibrary {
    pub fn v41_bf16_add(&self) -> Result<V41Bf16Add<'_>> {
        Ok(V41Bf16Add { _library: self,
            launch: unsafe { *self.lib.get(b"ds41rt_v41_add_tp2_shared_async")? } })
    }
}
impl V41Bf16Add<'_> {
    /// # Safety
    /// Buffers must be live on the current stream device, with producers ordered
    /// before this operation. No conflicting aliases may access them until done.
    pub unsafe fn launch(&self, a: Ds41rtDeviceBuffer, b: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer, count: usize, stream: *mut c_void) -> Result<()> {
        ensure!(count > 0 && count <= 4096*5120, "invalid BF16 addition extent");
        ensure!(a.device_id == output.device_id && b.device_id == output.device_id,
            "BF16 addition device mismatch");
        for buffer in [a,b,output] {
            ensure!(!buffer.ptr.is_null() && buffer.ptr as usize % 2 == 0 && buffer.bytes >= count*2,
                "invalid BF16 addition buffer");
        }
        let status = unsafe {
            (self.launch)(a.ptr.cast(), b.ptr.cast(), output.ptr.cast(), count, stream)
        };
        ensure!(status == 0, "shared TP2 addition failed with CUDA status {status}");
        Ok(())
    }
}
