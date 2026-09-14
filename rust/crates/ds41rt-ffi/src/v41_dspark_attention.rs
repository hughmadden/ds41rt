use crate::{Ds41rtDeviceBuffer, NativeLibrary, V41DsparkCache};
use anyhow::{ensure, Result};
use std::ffi::c_void;
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct V41AttentionWindow {
    pub slot: u32,
    pub valid_rows: u32,
}
const _: [(); 8] = [(); std::mem::size_of::<V41AttentionWindow>()];
type Launch = unsafe extern "C" fn(
    *const u16,
    *const u8,
    *const u16,
    *const f32,
    *const V41AttentionWindow,
    *mut u16,
    i32,
    i32,
    *mut c_void,
) -> i32;
type LaunchWidth = unsafe extern "C" fn(
    *const u16, *const u8, *const u16, *const f32, *const V41AttentionWindow,
    *mut u16, i32, i32, i32, *mut c_void,
) -> i32;
pub struct V41DsparkAttention<'a> {
    _library: &'a NativeLibrary,
    launch: Launch,
    launch_width: Option<LaunchWidth>,
    width: usize,
}
impl NativeLibrary {
    pub fn v41_dspark_attention(&self) -> Result<V41DsparkAttention<'_>> {
        self.v41_dspark_attention_width(5)
    }
    /// Select a fixed draft width before allocating buffers or capturing graphs.
    pub fn v41_dspark_attention_width(&self, width: usize) -> Result<V41DsparkAttention<'_>> {
        ensure!(matches!(width, 5 | 7), "draft width must be five or seven");
        let initialize = unsafe {
            *self
                .lib
                .get::<unsafe extern "C" fn() -> i32>(b"ds41rt_v41_dspark_attention_initialize")?
        };
        let status = if width == 5 { unsafe { initialize() } } else {
            let initialize_width = unsafe { *self.lib.get::<unsafe extern "C" fn(i32) -> i32>(
                b"ds41rt_v41_dspark_attention_initialize_width")? };
            unsafe { initialize_width(width as i32) }
        };
        ensure!(
            status == 0,
            "dSpark attention initialization CUDA status {status}"
        );
        Ok(V41DsparkAttention {
            _library: self,
            width,
            launch_width: if width == 5 { None } else {
                Some(unsafe { *self.lib.get(b"ds41rt_v41_dspark_attention_fp8_width")? })
            },
            launch: unsafe { *self.lib.get(b"ds41rt_v41_dspark_attention_fp8")? },
        })
    }
}
impl V41DsparkAttention<'_> {
    /// # Safety
    /// BF16 queries are rotated; committed KV is packed E4M3/E8M0 K32.
    /// Private BF16 KV has completed normalization, rotation and FP8
    /// quantize/dequantize. Per-request descriptors select valid
    /// initialized committed ring prefixes; all configured private draft entries are
    /// initialized. Inputs/sinks are finite where read. Output is disjoint from
    /// inputs; buffers live on the initialized stream device through completion
    /// and graph replay, with serialized writes and request ownership.
    pub unsafe fn launch(
        &self,
        query: Ds41rtDeviceBuffer,
        ring: Ds41rtDeviceBuffer,
        draft: Ds41rtDeviceBuffer,
        sink: Ds41rtDeviceBuffer,
        windows: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        requests: u32,
        slots: u32,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            (1..=16).contains(&requests) && (1..=16).contains(&slots),
            "invalid dSpark attention geometry"
        );
        for (value, bytes) in [
            (query, requests as usize * self.width * 65536),
            (ring, slots as usize * V41DsparkCache::SLOT_BYTES),
            (draft, requests as usize * self.width * 1024),
            (sink, 256),
            (windows, requests as usize * 8),
            (output, requests as usize * self.width * 65536),
        ] {
            ensure!(
                !value.ptr.is_null() && value.bytes >= bytes,
                "invalid dSpark attention buffer"
            );
        }
        let status = unsafe {
            if let Some(launch) = self.launch_width {
                launch(query.ptr.cast(), ring.ptr.cast(), draft.ptr.cast(), sink.ptr.cast(),
                    windows.ptr.cast(), output.ptr.cast(), requests as i32, slots as i32,
                    self.width as i32, stream)
            } else { (self.launch)(
                query.ptr.cast(),
                ring.ptr.cast(),
                draft.ptr.cast(),
                sink.ptr.cast(),
                windows.ptr.cast(),
                output.ptr.cast(),
                requests as i32,
                slots as i32,
                stream,
            ) }
        };
        ensure!(status == 0, "dSpark attention CUDA status {status}");
        Ok(())
    }
}
