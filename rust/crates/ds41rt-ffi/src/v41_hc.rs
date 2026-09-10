use crate::{Ds41rtDeviceBuffer, NativeLibrary};
use anyhow::{ensure, Result};
use std::ffi::c_void;
type Mixes = unsafe extern "C" fn(
    *const u16,
    *const f32,
    *const f32,
    *const f32,
    *mut f32,
    *mut f32,
    *mut f32,
    i32,
    *mut c_void,
) -> i32;
type Pre = unsafe extern "C" fn(*const u16, *const f32, *mut u16, i32, *mut c_void) -> i32;
type Post = unsafe extern "C" fn(
    *const u16,
    *const u16,
    *const f32,
    *const f32,
    *mut u16,
    i32,
    *mut c_void,
) -> i32;
pub struct V41Hc<'a> {
    _library: &'a NativeLibrary,
    pre: Pre,
    post: Post,
    mixes: Mixes,
}
impl NativeLibrary {
    pub fn v41_hc(&self) -> Result<V41Hc<'_>> {
        Ok(V41Hc {
            _library: self,
            mixes: unsafe { *self.lib.get::<Mixes>(b"ds41rt_v41_hc_mixes")? },
            pre: unsafe { *self.lib.get::<Pre>(b"ds41rt_v41_hc_pre")? },
            post: unsafe { *self.lib.get::<Post>(b"ds41rt_v41_hc_post")? },
        })
    }
}
fn buffers(rows: usize, views: &[(Ds41rtDeviceBuffer, usize)]) -> Result<()> {
    ensure!((1..=4096).contains(&rows), "invalid mHC rows");
    for (view, stride) in views {
        ensure!(
            !view.ptr.is_null() && view.bytes >= rows * stride,
            "invalid mHC buffer"
        );
    }
    Ok(())
}
impl V41Hc<'_> {
    /// # Safety
    /// Inputs must be initialized on the stream device and all buffers remain
    /// live through completion; outputs must be mutually disjoint and disjoint
    /// from inputs. Coefficients produced here belong to the following sublayer.
    pub unsafe fn mixes(
        &self,
        residual: Ds41rtDeviceBuffer,
        projection: Ds41rtDeviceBuffer,
        scale: Ds41rtDeviceBuffer,
        base: Ds41rtDeviceBuffer,
        pre: Ds41rtDeviceBuffer,
        post: Ds41rtDeviceBuffer,
        comb: Ds41rtDeviceBuffer,
        rows: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        buffers(
            rows,
            &[(residual, 40960), (pre, 16), (post, 16), (comb, 64)],
        )?;
        buffers(1, &[(projection, 24 * 20480 * 4), (scale, 12), (base, 96)])?;
        let status = unsafe {
            (self.mixes)(
                residual.ptr.cast(),
                projection.ptr.cast(),
                scale.ptr.cast(),
                base.ptr.cast(),
                pre.ptr.cast(),
                post.ptr.cast(),
                comb.ptr.cast(),
                rows as i32,
                stream,
            )
        };
        ensure!(status == 0, "mHC mixes CUDA status {status}");
        Ok(())
    }

    /// # Safety
    /// Inputs must be initialized on the stream's device and all buffers remain
    /// live through completion; output must be disjoint from both inputs.
    pub unsafe fn pre(
        &self,
        residual: Ds41rtDeviceBuffer,
        pre: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        rows: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        buffers(rows, &[(residual, 40960), (pre, 16), (output, 10240)])?;
        let status = unsafe {
            (self.pre)(
                residual.ptr.cast(),
                pre.ptr.cast(),
                output.ptr.cast(),
                rows as i32,
                stream,
            )
        };
        ensure!(status == 0, "mHC pre CUDA status {status}");
        Ok(())
    }
    /// # Safety
    /// Same device/lifetime/output-disjointness contract as pre. Comb is FP32
    /// [rows,source,destination], not destination-major.
    pub unsafe fn post(
        &self,
        sublayer: Ds41rtDeviceBuffer,
        residual: Ds41rtDeviceBuffer,
        post: Ds41rtDeviceBuffer,
        comb: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        rows: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        buffers(
            rows,
            &[
                (sublayer, 10240),
                (residual, 40960),
                (post, 16),
                (comb, 64),
                (output, 40960),
            ],
        )?;
        let status = unsafe {
            (self.post)(
                sublayer.ptr.cast(),
                residual.ptr.cast(),
                post.ptr.cast(),
                comb.ptr.cast(),
                output.ptr.cast(),
                rows as i32,
                stream,
            )
        };
        ensure!(status == 0, "mHC post CUDA status {status}");
        Ok(())
    }
}
