//! Official dSpark confidence is a raw FP32 projection, without sigmoid.
use crate::{Ds41rtDeviceBuffer, NativeLibrary};
use anyhow::{ensure, Result};
use std::ffi::c_void;

type ConfidenceFn =
    unsafe extern "C" fn(*const u16, *const u16, *const u16, *mut f32, i32, *mut c_void) -> i32;
pub struct V41DsparkConfidence<'a> {
    _library: &'a NativeLibrary,
    launch: ConfidenceFn,
}
impl NativeLibrary {
    pub fn v41_dspark_confidence(&self) -> Result<V41DsparkConfidence<'_>> {
        let launch = unsafe {
            *self
                .lib
                .get::<ConfidenceFn>(b"ds41rt_v41_dspark_confidence")?
        };
        Ok(V41DsparkConfidence {
            _library: self,
            launch,
        })
    }
}
impl V41DsparkConfidence<'_> {
    /// # Safety
    /// Initialized BF16 inputs and FP32 output must belong to the stream's device
    /// and remain live through completion; output must not overlap inputs.
    pub unsafe fn launch(
        &self,
        hidden: Ds41rtDeviceBuffer,
        markov: Ds41rtDeviceBuffer,
        weight: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        rows: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!((1..=4096).contains(&rows), "invalid dSpark confidence rows");
        for (buffer, bytes) in [
            (hidden, rows * 10240),
            (markov, rows * 512),
            (weight, 10752),
            (output, rows * 4),
        ] {
            ensure!(
                !buffer.ptr.is_null() && buffer.bytes >= bytes,
                "dSpark confidence buffer is null or undersized"
            );
        }
        let status = unsafe {
            (self.launch)(
                hidden.ptr.cast(),
                markov.ptr.cast(),
                weight.ptr.cast(),
                output.ptr.cast(),
                rows as i32,
                stream,
            )
        };
        ensure!(status == 0, "native dSpark confidence CUDA status {status}");
        Ok(())
    }
}
