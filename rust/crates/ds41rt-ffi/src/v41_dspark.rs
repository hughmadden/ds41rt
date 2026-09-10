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

type MarkovCreate = unsafe extern "C" fn(*mut c_void, u64, *mut *mut c_void) -> i32;
type MarkovDestroy = unsafe extern "C" fn(*mut c_void) -> i32;
type MarkovLaunch =
    unsafe extern "C" fn(*mut c_void, *const u16, *const u16, *mut f32, i32, *mut c_void) -> i32;
pub struct V41DsparkMarkov<'a> {
    _library: &'a NativeLibrary,
    handle: *mut c_void,
    launch: MarkovLaunch,
    destroy: MarkovDestroy,
}
impl NativeLibrary {
    /// # Safety
    /// Workspace must remain allocated on the current device until this handle
    /// and all graphs captured with it are destroyed; serialize handle use.
    pub unsafe fn v41_dspark_markov(
        &self,
        workspace: Ds41rtDeviceBuffer,
    ) -> Result<V41DsparkMarkov<'_>> {
        let create = unsafe { *self.lib.get::<MarkovCreate>(b"ds41rt_v41_markov_create")? };
        let launch = unsafe { *self.lib.get::<MarkovLaunch>(b"ds41rt_v41_markov_launch")? };
        let destroy = unsafe {
            *self
                .lib
                .get::<MarkovDestroy>(b"ds41rt_v41_markov_destroy")?
        };
        let mut handle = std::ptr::null_mut();
        let status = unsafe { create(workspace.ptr, workspace.bytes as u64, &mut handle) };
        ensure!(
            status == 0 && !handle.is_null(),
            "Markov initialization status {status}"
        );
        Ok(V41DsparkMarkov {
            _library: self,
            handle,
            launch,
            destroy,
        })
    }
}
impl V41DsparkMarkov<'_> {
    pub const WORKSPACE_BYTES: usize = 4 * 1024 * 1024;
    /// # Safety
    /// Initialized inputs and disjoint output must remain valid on the handle's
    /// device through stream completion, and must not overlap its workspace.
    pub unsafe fn launch(
        &self,
        embedding: Ds41rtDeviceBuffer,
        weight: Ds41rtDeviceBuffer,
        logits: Ds41rtDeviceBuffer,
        rows: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!((1..=16).contains(&rows), "invalid Markov rows");
        for (buffer, bytes) in [
            (embedding, rows * 512),
            (weight, 129280 * 512),
            (logits, rows * 129280 * 4),
        ] {
            ensure!(
                !buffer.ptr.is_null() && buffer.bytes >= bytes,
                "invalid Markov buffer"
            );
        }
        let status = unsafe {
            (self.launch)(
                self.handle,
                embedding.ptr.cast(),
                weight.ptr.cast(),
                logits.ptr.cast(),
                rows as i32,
                stream,
            )
        };
        ensure!(status == 0, "Markov launch status {status}");
        Ok(())
    }
}
impl Drop for V41DsparkMarkov<'_> {
    fn drop(&mut self) {
        let status = unsafe { (self.destroy)(self.handle) };
        if status != 0 {
            eprintln!("Markov handle destruction status {status}");
        }
    }
}

type DraftStepFn = unsafe extern "C" fn(
    *const f32,
    *const f32,
    *const f32,
    *const f32,
    *mut f32,
    *mut u32,
    i32,
    *mut c_void,
) -> i32;
pub struct V41DraftStep<'a> {
    _library: &'a NativeLibrary,
    launch: DraftStepFn,
}
impl NativeLibrary {
    pub fn v41_draft_step(&self) -> Result<V41DraftStep<'_>> {
        let launch = unsafe { *self.lib.get::<DraftStepFn>(b"ds41rt_v41_draft_step")? };
        Ok(V41DraftStep {
            _library: self,
            launch,
        })
    }
}
impl V41DraftStep<'_> {
    /// # Safety
    /// All buffers must live on the stream device through completion. Shared+bias
    /// must be finite, temperatures finite/nonnegative and stochastic noise must
    /// be positive finite Exp(1) variates, independently supplied per vocabulary
    /// entry. Output spans must not overlap each other or any input.
    pub unsafe fn launch(
        &self,
        shared: Ds41rtDeviceBuffer,
        bias: Ds41rtDeviceBuffer,
        noise: Ds41rtDeviceBuffer,
        temperatures: Ds41rtDeviceBuffer,
        adjusted: Ds41rtDeviceBuffer,
        tokens: Ds41rtDeviceBuffer,
        rows: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!((1..=16).contains(&rows), "invalid draft sampling rows");
        for (buffer, bytes) in [
            (shared, rows * 129280 * 4),
            (bias, rows * 129280 * 4),
            (noise, rows * 129280 * 4),
            (temperatures, rows * 4),
            (adjusted, rows * 129280 * 4),
            (tokens, rows * 4),
        ] {
            ensure!(
                !buffer.ptr.is_null() && buffer.bytes >= bytes,
                "invalid draft sampling buffer"
            );
        }
        let status = unsafe {
            (self.launch)(
                shared.ptr.cast(),
                bias.ptr.cast(),
                noise.ptr.cast(),
                temperatures.ptr.cast(),
                adjusted.ptr.cast(),
                tokens.ptr.cast(),
                rows as i32,
                stream,
            )
        };
        ensure!(status == 0, "draft sampling CUDA status {status}");
        Ok(())
    }
}
