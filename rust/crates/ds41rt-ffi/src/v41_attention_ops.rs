use crate::{Ds41rtDeviceBuffer, NativeLibrary};
use anyhow::{ensure, Result};
use std::ffi::c_void;
type Norm = unsafe extern "C" fn(
    *const u16,
    *const u16,
    *const f32,
    *mut u16,
    i32,
    i32,
    *mut c_void,
) -> i32;
type Kv =
    unsafe extern "C" fn(*const u16, *const u16, *const f32, *mut u16, i32, *mut c_void) -> i32;
type Rope =
    unsafe extern "C" fn(*const u16, *const f32, *mut u16, i32, i32, i32, *mut c_void) -> i32;
type Frequencies = unsafe extern "C" fn(*const u64, *mut f32, i32, *mut c_void) -> i32;
pub struct V41AttentionOps<'a> {
    _library: &'a NativeLibrary,
    norm: Norm,
    frequencies: Frequencies,
    kv: Kv,
    rope: Rope,
}
fn buffer(value: Ds41rtDeviceBuffer, bytes: usize) -> Result<()> {
    ensure!(
        !value.ptr.is_null() && value.bytes >= bytes,
        "invalid attention operation buffer"
    );
    Ok(())
}
impl NativeLibrary {
    pub fn v41_attention_ops(&self) -> Result<V41AttentionOps<'_>> {
        Ok(V41AttentionOps {
            _library: self,
            frequencies: unsafe { *self.lib.get(b"ds41rt_v41_dspark_frequencies")? },
            kv: unsafe { *self.lib.get(b"ds41rt_v41_attention_kv")? },
            norm: unsafe { *self.lib.get(b"ds41rt_v41_attention_norm")? },
            rope: unsafe { *self.lib.get(b"ds41rt_v41_attention_rope")? },
        })
    }
}
impl V41AttentionOps<'_> {
    /// # Safety
    /// Initialized U64 absolute positions and disjoint FP32 output are on the
    /// stream device and stay live through completion and graph replay.
    pub unsafe fn frequencies(
        &self,
        positions: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        rows: u32,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!((1..=4096).contains(&rows), "invalid frequency rows");
        buffer(positions, rows as usize * 8)?;
        buffer(output, rows as usize * 256)?;
        let status = unsafe {
            (self.frequencies)(positions.ptr.cast(), output.ptr.cast(), rows as i32, stream)
        };
        ensure!(status == 0, "native frequency CUDA status {status}");
        Ok(())
    }

    /// # Safety
    /// Same initialized, disjoint and live device-buffer contract as norm;
    /// produces official quantized/dequantized BF16 KV without intermediate storage.
    pub unsafe fn kv(
        &self,
        input: Ds41rtDeviceBuffer,
        weight: Ds41rtDeviceBuffer,
        frequencies: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        rows: u32,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!((1..=4096).contains(&rows), "invalid KV rows");
        buffer(input, rows as usize * 1024)?;
        buffer(weight, 1024)?;
        buffer(frequencies, rows as usize * 256)?;
        buffer(output, rows as usize * 1024)?;
        let status = unsafe {
            (self.kv)(
                input.ptr.cast(),
                weight.ptr.cast(),
                frequencies.ptr.cast(),
                output.ptr.cast(),
                rows as i32,
                stream,
            )
        };
        ensure!(status == 0, "native KV CUDA status {status}");
        Ok(())
    }
    /// # Safety
    /// Finite initialized BF16 input/weight and optional per-row complex FP32
    /// frequencies are on the stream device; output is disjoint from inputs.
    /// Keep storage live and serialize mutation through completion and replay.
    pub unsafe fn norm(
        &self,
        input: Ds41rtDeviceBuffer,
        weight: Ds41rtDeviceBuffer,
        frequencies: Option<Ds41rtDeviceBuffer>,
        output: Ds41rtDeviceBuffer,
        rows: u32,
        dim: u32,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            (1..=4096).contains(&rows) && [512, 1280, 5120].contains(&dim),
            "invalid attention norm geometry"
        );
        ensure!(
            frequencies.is_none() || dim == 512,
            "rotated norm requires KV geometry"
        );
        buffer(input, rows as usize * dim as usize * 2)?;
        buffer(output, rows as usize * dim as usize * 2)?;
        buffer(weight, dim as usize * 2)?;
        if let Some(f) = frequencies {
            buffer(f, rows as usize * 256)?;
        }
        let status = unsafe {
            (self.norm)(
                input.ptr.cast(),
                weight.ptr.cast(),
                frequencies.map_or(std::ptr::null(), |f| f.ptr.cast()),
                output.ptr.cast(),
                rows as i32,
                dim as i32,
                stream,
            )
        };
        ensure!(status == 0, "native attention norm CUDA status {status}");
        Ok(())
    }
    /// # Safety
    /// Finite initialized BF16 vectors and per-row complex FP32 frequencies are
    /// on the stream device; output is disjoint and all buffers live through
    /// completion/replay, with serialized producers and consumers.
    pub unsafe fn rope(
        &self,
        input: Ds41rtDeviceBuffer,
        frequencies: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        rows: u32,
        heads: u32,
        inverse: bool,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            (1..=4096).contains(&rows) && [1, 64].contains(&heads),
            "invalid attention RoPE geometry"
        );
        buffer(input, rows as usize * heads as usize * 1024)?;
        buffer(output, rows as usize * heads as usize * 1024)?;
        buffer(frequencies, rows as usize * 256)?;
        let status = unsafe {
            (self.rope)(
                input.ptr.cast(),
                frequencies.ptr.cast(),
                output.ptr.cast(),
                rows as i32,
                heads as i32,
                inverse as i32,
                stream,
            )
        };
        ensure!(status == 0, "native attention RoPE CUDA status {status}");
        Ok(())
    }
}
