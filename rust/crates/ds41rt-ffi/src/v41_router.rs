use crate::{Ds41rtDeviceBuffer, NativeLibrary};
use anyhow::{ensure, Result};
use std::ffi::c_void;
type Launch = unsafe extern "C" fn(
    *const u16,
    *const u16,
    *const f32,
    *const f32,
    *const u8,
    *mut f32,
    *mut u32,
    *mut f32,
    i32,
    i32,
    *mut c_void,
) -> i32;
pub struct V41Router<'a> {
    _library: &'a NativeLibrary,
    launch: Launch,
}
impl NativeLibrary {
    pub fn v41_router(&self) -> Result<V41Router<'_>> {
        let initialize = unsafe {
            *self
                .lib
                .get::<unsafe extern "C" fn() -> i32>(b"ds41rt_v41_router_initialize")?
        };
        let status = unsafe { initialize() };
        ensure!(
            status == 0,
            "native router initialization CUDA status {status}"
        );
        Ok(V41Router {
            _library: self,
            launch: unsafe { *self.lib.get::<Launch>(b"ds41rt_v41_router")? },
        })
    }
}
impl V41Router<'_> {
    /// # Safety
    /// Initialized inputs must belong to the stream device, with finite logits
    /// and biases; all buffers must remain live through completion. Outputs must
    /// not overlap inputs or each other. image_mask requires a valid VL bias.
    pub unsafe fn launch(
        &self,
        hidden: Ds41rtDeviceBuffer,
        weight: Ds41rtDeviceBuffer,
        bias: Ds41rtDeviceBuffer,
        bias_vl: Ds41rtDeviceBuffer,
        image_mask: Option<Ds41rtDeviceBuffer>,
        scores: Ds41rtDeviceBuffer,
        ids: Ds41rtDeviceBuffer,
        routing: Ds41rtDeviceBuffer,
        rows: usize,
        experts: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            (1..=4096).contains(&rows) && matches!(experts, 128 | 384),
            "invalid router geometry"
        );
        let topk = if experts == 128 { 3 } else { 6 };
        let views = [
            (hidden, rows * 10240),
            (weight, experts * 10240),
            (bias, experts * 4),
            (scores, rows * experts * 4),
            (ids, rows * topk * 4),
            (routing, rows * topk * 4),
        ];
        for (view, bytes) in views {
            ensure!(
                !view.ptr.is_null() && view.bytes >= bytes,
                "invalid router buffer"
            );
        }
        if let Some(mask) = image_mask {
            ensure!(
                !mask.ptr.is_null()
                    && mask.bytes >= rows
                    && !bias_vl.ptr.is_null()
                    && bias_vl.bytes >= experts * 4,
                "invalid router image mask or VL bias"
            );
        }
        let status = unsafe {
            (self.launch)(
                hidden.ptr.cast(),
                weight.ptr.cast(),
                bias.ptr.cast(),
                bias_vl.ptr.cast(),
                image_mask.map_or(std::ptr::null(), |m| m.ptr.cast()),
                scores.ptr.cast(),
                ids.ptr.cast(),
                routing.ptr.cast(),
                rows as i32,
                experts as i32,
                stream,
            )
        };
        ensure!(status == 0, "native router CUDA status {status}");
        Ok(())
    }
}
