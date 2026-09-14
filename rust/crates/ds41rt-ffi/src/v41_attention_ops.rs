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
type BackboneFrequencies = unsafe extern "C" fn(*const u64, *mut f32, i32, i32, *mut c_void) -> i32;
type Frequencies = unsafe extern "C" fn(*const u64, *mut f32, i32, *mut c_void) -> i32;
type Tap = unsafe extern "C" fn(*const u16, *mut u16, i32, i32, *mut c_void) -> i32;
type Embed =
    unsafe extern "C" fn(*const u16, *const i32, *mut u16, *mut f32, i32, *mut c_void) -> i32;
type TerminalLayout =
    unsafe extern "C" fn(*const u16, *const f32, *mut u16, *mut f32, i32, *mut c_void) -> i32;
pub struct V41AttentionOps<'a> {
    _library: &'a NativeLibrary,
    norm: Norm,
    embed: Embed,
    target_embed: Embed,
    terminal_layout: TerminalLayout,
    frequencies: Frequencies,
    backbone_frequencies: BackboneFrequencies,
    tap: Tap,
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
            terminal_layout: unsafe { *self.lib.get(b"ds41rt_v41_dspark_terminal_layout")? },
            embed: unsafe { *self.lib.get(b"ds41rt_v41_dspark_embed")? },
            target_embed: unsafe { *self.lib.get(b"ds41rt_v41_target_embed")? },
            tap: unsafe { *self.lib.get(b"ds41rt_v41_dspark_tap")? },
            backbone_frequencies: unsafe { *self.lib.get(b"ds41rt_v41_backbone_frequencies")? },
            frequencies: unsafe { *self.lib.get(b"ds41rt_v41_dspark_frequencies")? },
            kv: unsafe { *self.lib.get(b"ds41rt_v41_attention_kv")? },
            norm: unsafe { *self.lib.get(b"ds41rt_v41_attention_norm")? },
            rope: unsafe { *self.lib.get(b"ds41rt_v41_attention_rope")? },
        })
    }
}
impl V41AttentionOps<'_> {
    /// # Safety
    /// U64 absolute token positions and disjoint FP32 output [rows,32,2] are
    /// initialized/live on the stream device. Compressed groups use their first
    /// token position, not their latent index; layer selects the official RoPE.
    pub unsafe fn backbone_frequencies(
        &self,
        positions: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        rows: u32,
        layer: u32,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            (1..=4096).contains(&rows) && layer < 40,
            "invalid backbone frequency geometry"
        );
        buffer(positions, rows as usize * 8)?;
        buffer(output, rows as usize * 256)?;
        let status = unsafe {
            (self.backbone_frequencies)(
                positions.ptr.cast(),
                output.ptr.cast(),
                rows as i32,
                layer as i32,
                stream,
            )
        };
        ensure!(status == 0, "native backbone frequency status {status}");
        Ok(())
    }

    /// # Safety
    /// Initialized request-major residual/pre inputs and disjoint position-major
    /// outputs are on the stream device and live through completion.
    pub unsafe fn terminal_layout(
        &self,
        residual: Ds41rtDeviceBuffer,
        pre: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        output_pre: Ds41rtDeviceBuffer,
        requests: u32,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            (1..=16).contains(&requests),
            "invalid terminal layout request count"
        );
        for b in [residual, output] {
            buffer(b, requests as usize * 5 * 40960)?;
        }
        for b in [pre, output_pre] {
            buffer(b, requests as usize * 5 * 16)?;
        }
        let status = unsafe {
            (self.terminal_layout)(
                residual.ptr.cast(),
                pre.ptr.cast(),
                output.ptr.cast(),
                output_pre.ptr.cast(),
                requests as i32,
                stream,
            )
        };
        ensure!(status == 0, "native terminal layout CUDA status {status}");
        Ok(())
    }

    /// # Safety
    /// BF16 embedding rows are initialized and immutable for the launch. The
    /// table may reside on a peer GPU whose access was enabled during planning;
    /// its producer must complete before this stream reads it. I32 seed IDs and
    /// outputs belong to the stream device. All storage stays live and outputs
    /// are disjoint until completion, including captured graph replays.
    pub unsafe fn embed(
        &self,
        table: Ds41rtDeviceBuffer,
        tokens: Ds41rtDeviceBuffer,
        residual: Ds41rtDeviceBuffer,
        pre: Ds41rtDeviceBuffer,
        requests: u32,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            (1..=16).contains(&requests),
            "invalid embedding request count"
        );
        buffer(table, 129280 * 5120 * 2)?;
        buffer(tokens, requests as usize * 4)?;
        buffer(residual, requests as usize * 5 * 40960)?;
        buffer(pre, requests as usize * 5 * 16)?;
        ensure!(tokens.device_id == residual.device_id && pre.device_id == residual.device_id,
            "draft embedding token/output devices differ");
        let status = unsafe {
            (self.embed)(
                table.ptr.cast(),
                tokens.ptr.cast(),
                residual.ptr.cast(),
                pre.ptr.cast(),
                requests as i32,
                stream,
            )
        };
        ensure!(status == 0, "native embedding CUDA status {status}");
        Ok(())
    }

    /// # Safety
    /// Shared BF16 embedding table and I32 token IDs are initialized on the
    /// stream device. Outputs are exclusive and disjoint from inputs. All
    /// storage remains live until queued work and graph replays complete.
    pub unsafe fn target_embed(
        &self,
        table: Ds41rtDeviceBuffer,
        tokens: Ds41rtDeviceBuffer,
        residual: Ds41rtDeviceBuffer,
        pre: Ds41rtDeviceBuffer,
        rows: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!((1..=4096).contains(&rows), "invalid target embedding rows");
        buffer(table, 129280 * 5120 * 2)?;
        buffer(tokens, rows * 4)?;
        buffer(residual, rows * 40960)?;
        buffer(pre, rows * 16)?;
        ensure!(
            [tokens, residual, pre].iter().all(|b| b.device_id == table.device_id),
            "target embedding devices differ"
        );
        let status = unsafe {
            (self.target_embed)(table.ptr.cast(), tokens.ptr.cast(), residual.ptr.cast(),
                pre.ptr.cast(), rows as i32, stream)
        };
        ensure!(status == 0, "native target embedding CUDA status {status}");
        Ok(())
    }

    /// # Safety
    /// Input is initialized finite BF16 [rows,4,5120] after the target layer's
    /// engram update and before attention; output is disjoint [rows,15360].
    /// Storage is on the stream device and stays live through completion/replay.
    pub unsafe fn tap(
        &self,
        input: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        rows: u32,
        layer: u32,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            (1..=4096).contains(&rows) && (37..=39).contains(&layer),
            "invalid dSpark tap geometry"
        );
        buffer(input, rows as usize * 40960)?;
        buffer(output, rows as usize * 30720)?;
        let status = unsafe {
            (self.tap)(
                input.ptr.cast(),
                output.ptr.cast(),
                rows as i32,
                layer as i32,
                stream,
            )
        };
        ensure!(status == 0, "native tap CUDA status {status}");
        Ok(())
    }

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
            (1..=4096).contains(&rows) && [128, 512, 1280, 5120].contains(&dim),
            "invalid attention norm geometry"
        );
        ensure!(
            frequencies.is_none() || dim == 128 || dim == 512,
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
    /// on the stream device; output is disjoint or exactly aliases input. All buffers live through
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
