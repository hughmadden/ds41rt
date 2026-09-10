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
pub struct V41VocabularyProjection<'a> {
    _library: &'a NativeLibrary,
    handle: *mut c_void,
    launch: MarkovLaunch,
    destroy: MarkovDestroy,
    width: usize,
    max_rows: usize,
}
impl NativeLibrary {
    /// # Safety
    /// Workspace must remain allocated on the current device until this handle
    /// and all graphs captured with it are destroyed; serialize handle use.
    pub unsafe fn v41_dspark_markov(
        &self,
        workspace: Ds41rtDeviceBuffer,
    ) -> Result<V41VocabularyProjection<'_>> {
        unsafe { self.v41_head_projection(workspace, false) }
    }
    /// # Safety
    /// Same workspace/device/graph lifetime contract as v41_dspark_markov.
    pub unsafe fn v41_vocabulary_head(
        &self,
        workspace: Ds41rtDeviceBuffer,
    ) -> Result<V41VocabularyProjection<'_>> {
        unsafe { self.v41_head_projection(workspace, true) }
    }
    unsafe fn v41_head_projection(
        &self,
        workspace: Ds41rtDeviceBuffer,
        full: bool,
    ) -> Result<V41VocabularyProjection<'_>> {
        let create_name: &[u8] = if full {
            b"ds41rt_v41_vocabulary_head_create"
        } else {
            b"ds41rt_v41_markov_create"
        };
        let launch_name: &[u8] = if full {
            b"ds41rt_v41_vocabulary_head_launch"
        } else {
            b"ds41rt_v41_markov_launch"
        };
        let create = unsafe { *self.lib.get::<MarkovCreate>(create_name)? };
        let launch = unsafe { *self.lib.get::<MarkovLaunch>(launch_name)? };
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
        Ok(V41VocabularyProjection {
            _library: self,
            handle,
            launch,
            destroy,
            width: if full { 5120 } else { 256 },
            max_rows: if full { 80 } else { 16 },
        })
    }
}
impl V41VocabularyProjection<'_> {
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
        ensure!(
            (1..=self.max_rows).contains(&rows),
            "invalid vocabulary projection rows"
        );
        for (buffer, bytes) in [
            (embedding, rows * self.width * 2),
            (weight, 129280 * self.width * 2),
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
impl Drop for V41VocabularyProjection<'_> {
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
    *const u64,
    *const f32,
    *mut f32,
    *mut u32,
    i32,
    i32,
    *mut c_void,
) -> i32;
pub struct V41DraftStep<'a> {
    _library: &'a NativeLibrary,
    launch: DraftStepFn,
}
impl NativeLibrary {
    pub fn v41_draft_step(&self) -> Result<V41DraftStep<'_>> {
        let launch = unsafe { *self.lib.get::<DraftStepFn>(b"ds41rt_v41_draft_step_rng")? };
        Ok(V41DraftStep {
            _library: self,
            launch,
        })
    }
}
impl V41DraftStep<'_> {
    /// # Safety
    /// All buffers must live on the stream device through completion. Shared+bias
    /// must be finite and temperatures finite/nonnegative. RNG [rows,2] contains
    /// seeds and base Philox subsequences with room for 1280 subsequences each.
    /// Output spans must not overlap each other or any input.
    pub unsafe fn launch(
        &self,
        shared: Ds41rtDeviceBuffer,
        bias: Ds41rtDeviceBuffer,
        rng: Ds41rtDeviceBuffer,
        temperatures: Ds41rtDeviceBuffer,
        adjusted: Ds41rtDeviceBuffer,
        tokens: Ds41rtDeviceBuffer,
        rows: usize,
        position: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!((1..=16).contains(&rows), "invalid draft sampling rows");
        ensure!(position < 5, "invalid draft position");
        for (buffer, bytes) in [
            (shared, rows * 129280 * 4),
            (bias, rows * 129280 * 4),
            (rng, rows * 16),
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
                rng.ptr.cast(),
                temperatures.ptr.cast(),
                adjusted.ptr.cast(),
                tokens.ptr.cast(),
                rows as i32,
                position as i32,
                stream,
            )
        };
        ensure!(status == 0, "draft sampling CUDA status {status}");
        Ok(())
    }
}
