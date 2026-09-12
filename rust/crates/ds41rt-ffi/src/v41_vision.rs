//! Native BF16 vision operations; the caller owns every device buffer.
use crate::{Ds41rtDeviceBuffer as Buffer, NativeLibrary};
use anyhow::{ensure, Result};
use std::ffi::c_void;

type Create = unsafe extern "C" fn(*mut c_void, u64, *mut *mut c_void) -> i32;
type Destroy = unsafe extern "C" fn(*mut c_void) -> i32;
type Linear = unsafe extern "C" fn(
    *mut c_void,
    *const u16,
    *const u16,
    *const u16,
    *mut f32,
    *mut u16,
    i32,
    i32,
    i32,
    *mut c_void,
) -> i32;
type Norm = unsafe extern "C" fn(*const u16, *const u16, *mut u16, i32, *mut c_void) -> i32;
type Element =
    unsafe extern "C" fn(*const u16, *const u16, *mut u16, i32, i32, i32, *mut c_void) -> i32;
type Rope = unsafe extern "C" fn(
    *const u16,
    *const f32,
    *mut u16,
    *mut u16,
    *mut u16,
    i32,
    i32,
    *mut c_void,
) -> i32;
type Attention = unsafe extern "C" fn(
    *mut c_void,
    *const u16,
    *const u16,
    *const u16,
    *mut f32,
    *mut f32,
    *mut f32,
    *mut u16,
    i32,
    i32,
    *mut c_void,
) -> i32;
type Merge = unsafe extern "C" fn(*const u16, *mut u16, i32, i32, *mut c_void) -> i32;
type Span = unsafe extern "C" fn(
    *const u16,
    *const u16,
    *const u16,
    *const u16,
    *mut u16,
    i32,
    i32,
    *mut c_void,
) -> i32;
pub struct V41VisionOps<'a> {
    _library: &'a NativeLibrary,
    handle: *mut c_void,
    device: i32,
    destroy: Destroy,
    linear: Linear,
    norm: Norm,
    element: Element,
    rope: Rope,
    attention: Attention,
    merge: Merge,
    span: Span,
}
impl NativeLibrary {
    /// # Safety
    /// Workspace is an exclusive current-device allocation, live until this
    /// handle and its pending stream/graph work are destroyed. Serialize calls.
    pub unsafe fn v41_vision_ops(&self, workspace: Buffer) -> Result<V41VisionOps<'_>> {
        ensure!(
            !workspace.ptr.is_null() && workspace.bytes >= V41VisionOps::WORKSPACE_BYTES,
            "invalid vision BLAS workspace"
        );
        unsafe {
            let create = *self.lib.get::<Create>(b"ds41rt_v41_vision_create")?;
            let mut ops = V41VisionOps {
                _library: self,
                handle: std::ptr::null_mut(),
                device: workspace.device_id,
                destroy: *self.lib.get::<Destroy>(b"ds41rt_v41_vision_destroy")?,
                linear: *self.lib.get::<Linear>(b"ds41rt_v41_vision_linear")?,
                norm: *self.lib.get::<Norm>(b"ds41rt_v41_vision_norm")?,
                element: *self.lib.get::<Element>(b"ds41rt_v41_vision_elementwise")?,
                rope: *self.lib.get::<Rope>(b"ds41rt_v41_vision_rope")?,
                attention: *self.lib.get::<Attention>(b"ds41rt_v41_vision_attention")?,
                merge: *self.lib.get::<Merge>(b"ds41rt_v41_vision_merge")?,
                span: *self.lib.get::<Span>(b"ds41rt_v41_vision_span")?,
            };
            let status = create(workspace.ptr, workspace.bytes as u64, &mut ops.handle);
            ensure!(
                status == 0 && !ops.handle.is_null(),
                "vision handle status {status}"
            );
            Ok(ops)
        }
    }
}
impl V41VisionOps<'_> {
    pub const MAX_PATCHES: usize = 9216;
    pub const WORKSPACE_BYTES: usize = 4 * 1024 * 1024;
    pub const QUERY_TILE: usize = 128;
    fn buffers(&self, buffers: &[(Buffer, usize)]) -> Result<()> {
        for &(buffer, bytes) in buffers {
            ensure!(
                !buffer.ptr.is_null() && buffer.bytes >= bytes && buffer.device_id == self.device,
                "vision buffer extent/device mismatch: need {bytes}, got {} on {}",
                buffer.bytes,
                buffer.device_id
            );
        }
        Ok(())
    }
    fn rows(rows: usize) -> Result<()> {
        ensure!(
            (1..=Self::MAX_PATCHES).contains(&rows),
            "invalid vision rows"
        );
        Ok(())
    }
    fn status(status: i32) -> Result<()> {
        ensure!(status == 0, "native vision CUDA status {status}");
        Ok(())
    }
    /// # Safety
    /// All calls require initialized inputs on the current device and disjoint
    /// output/scratch, retained through stream completion. See the native header
    /// for exact in-place exceptions. No concurrent use of the shared handle.
    pub unsafe fn linear(
        &self,
        x: Buffer,
        w: Buffer,
        bias: Option<Buffer>,
        scratch: Buffer,
        y: Buffer,
        rows: usize,
        input: usize,
        output: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        Self::rows(rows)?;
        ensure!(
            (1..=9216).contains(&input) && (1..=5632).contains(&output),
            "invalid vision linear geometry"
        );
        self.buffers(&[
            (x, rows * input * 2),
            (w, output * input * 2),
            (y, rows * output * 2),
        ])?;
        if let Some(bias) = bias {
            self.buffers(&[(bias, output * 2), (scratch, rows * output * 4)])?;
        }
        Self::status(unsafe {
            (self.linear)(
                self.handle,
                x.ptr.cast(),
                w.ptr.cast(),
                bias.map_or(std::ptr::null(), |b| b.ptr.cast()),
                scratch.ptr.cast(),
                y.ptr.cast(),
                rows as i32,
                input as i32,
                output as i32,
                stream,
            )
        })
    }
    /// # Safety
    /// Same stream/device/lifetime contract as linear; output is disjoint.
    pub unsafe fn norm(
        &self,
        x: Buffer,
        w: Buffer,
        y: Buffer,
        rows: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        Self::rows(rows)?;
        self.buffers(&[(x, rows * 2048), (w, 2048), (y, rows * 2048)])?;
        Self::status(unsafe {
            (self.norm)(
                x.ptr.cast(),
                w.ptr.cast(),
                y.ptr.cast(),
                rows as i32,
                stream,
            )
        })
    }
    /// # Safety
    /// Same contract as linear. Add and GELU permit exact input/output aliasing.
    pub unsafe fn element(
        &self,
        x: Buffer,
        other: Option<Buffer>,
        y: Buffer,
        rows: usize,
        mode: i32,
        stream: *mut c_void,
    ) -> Result<()> {
        Self::rows(rows)?;
        let width = match mode {
            0 => 1024,
            1 => 2816,
            2 => 5120,
            _ => anyhow::bail!("invalid vision element operation"),
        };
        self.buffers(&[
            (x, rows * width * 2 * if mode == 1 { 2 } else { 1 }),
            (y, rows * width * 2),
        ])?;
        if mode == 0 {
            self.buffers(&[(
                other.ok_or_else(|| anyhow::anyhow!("vision add needs second input"))?,
                rows * width * 2,
            )])?;
        }
        Self::status(unsafe {
            (self.element)(
                x.ptr.cast(),
                other.map_or(std::ptr::null(), |b| b.ptr.cast()),
                y.ptr.cast(),
                rows as i32,
                width as i32,
                mode,
                stream,
            )
        })
    }
    /// # Safety
    /// Same contract as linear; inputs, outputs and frequency buffer are disjoint.
    pub unsafe fn rope(
        &self,
        qkv: Buffer,
        inv: Buffer,
        q: Buffer,
        k: Buffer,
        v: Buffer,
        height: usize,
        width: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            height > 0 && width > 0 && height <= 9216 && width <= 9216,
            "invalid vision grid"
        );
        let rows = height * width;
        Self::rows(rows)?;
        self.buffers(&[
            (qkv, rows * 6144),
            (inv, 64),
            (q, rows * 2048),
            (k, rows * 2048),
            (v, rows * 2048),
        ])?;
        Self::status(unsafe {
            (self.rope)(
                qkv.ptr.cast(),
                inv.ptr.cast(),
                q.ptr.cast(),
                k.ptr.cast(),
                v.ptr.cast(),
                height as i32,
                width as i32,
                stream,
            )
        })
    }
    /// # Safety
    /// Same contract as linear; all scratch and Q/K/V/output buffers are disjoint.
    pub unsafe fn attention(
        &self,
        q: Buffer,
        k: Buffer,
        v: Buffer,
        scores: Buffer,
        values: Buffer,
        output: Buffer,
        y: Buffer,
        rows: usize,
        fp32_probabilities: bool,
        stream: *mut c_void,
    ) -> Result<()> {
        Self::rows(rows)?;
        let n = rows * 16 * Self::QUERY_TILE;
        self.buffers(&[
            (q, rows * 2048),
            (k, rows * 2048),
            (v, rows * 2048),
            (scores, n * 4),
            (values, rows * 4096),
            (output, rows * 4096),
            (y, rows * 2048),
        ])?;
        Self::status(unsafe {
            (self.attention)(
                self.handle,
                q.ptr.cast(),
                k.ptr.cast(),
                v.ptr.cast(),
                scores.ptr.cast(),
                values.ptr.cast(),
                output.ptr.cast(),
                y.ptr.cast(),
                rows as i32,
                i32::from(fp32_probabilities),
                stream,
            )
        })
    }
    /// # Safety
    /// Same contract as linear; output is disjoint from the image features.
    pub unsafe fn merge(
        &self,
        x: Buffer,
        y: Buffer,
        height: usize,
        width: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            height > 0 && width > 0 && height <= 9216 && width <= 9216,
            "invalid vision grid"
        );
        let rows = height * width;
        Self::rows(rows)?;
        let merged = height.div_ceil(3) * width.div_ceil(3);
        ensure!(merged <= 1024, "image merge exceeds capacity");
        self.buffers(&[(x, rows * 2048), (y, merged * 18432)])?;
        Self::status(unsafe {
            (self.merge)(
                x.ptr.cast(),
                y.ptr.cast(),
                height as i32,
                width as i32,
                stream,
            )
        })
    }
    /// # Safety
    /// Same contract as linear; output is disjoint from all four embedding inputs.
    pub unsafe fn span(
        &self,
        x: Buffer,
        start: Buffer,
        newline: Buffer,
        end: Buffer,
        y: Buffer,
        height: usize,
        width: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            height > 0 && width > 0 && height <= 1024 && width <= 1024,
            "invalid image token grid"
        );
        let tokens = height * (width + 1) + 2;
        ensure!(tokens <= 1024, "image span exceeds token limit");
        self.buffers(&[
            (x, height * width * 10240),
            (start, 10240),
            (newline, 10240),
            (end, 10240),
            (y, tokens * 10240),
        ])?;
        Self::status(unsafe {
            (self.span)(
                x.ptr.cast(),
                start.ptr.cast(),
                newline.ptr.cast(),
                end.ptr.cast(),
                y.ptr.cast(),
                height as i32,
                width as i32,
                stream,
            )
        })
    }
}
impl Drop for V41VisionOps<'_> {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            let status = unsafe { (self.destroy)(self.handle) };
            if status != 0 {
                eprintln!("destroying native vision handle: CUDA status {status}");
            }
        }
    }
}
