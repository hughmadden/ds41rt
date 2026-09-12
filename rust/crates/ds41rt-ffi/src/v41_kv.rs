//! Architectural window FP8 and compressed FP4 KV packing and accepted-row scatter.
use crate::{Ds41rtDeviceBuffer, NativeLibrary};
use anyhow::{ensure, Result};
use std::ffi::c_void;
type Pack = unsafe extern "C" fn(*const u16, *const f32, *mut u8, *mut u8, i32, *mut c_void) -> i32;
type Store = unsafe extern "C" fn(
    *const u8,
    *const u8,
    *const u64,
    *mut u8,
    *mut u8,
    i32,
    u64,
    *mut c_void,
) -> i32;
pub struct V41Kv<'a> {
    _library: &'a NativeLibrary,
    pack: Pack,
    store: Store,
    value_bytes: usize,
    scale_bytes: usize,
}
fn check(buffers: &[(Ds41rtDeviceBuffer, usize)]) -> Result<()> {
    for &(b, n) in buffers {
        ensure!(
            !b.ptr.is_null() && b.bytes >= n && b.device_id == buffers[0].0.device_id,
            "KV buffer is null, undersized or on another device"
        );
    }
    Ok(())
}
impl NativeLibrary {
    pub fn v41_kv(&self) -> Result<V41Kv<'_>> {
        Ok(V41Kv {
            _library: self,
            pack: unsafe { *self.lib.get(b"ds41rt_v41_kv_pack")? },
            store: unsafe { *self.lib.get(b"ds41rt_v41_kv_store")? },
            value_bytes: 512,
            scale_bytes: 16,
        })
    }
    /// Native compressed format: E2M1 values and one E4M3 scale per 16 coordinates.
    pub fn v41_compressed_kv(&self) -> Result<V41Kv<'_>> {
        Ok(V41Kv {
            _library: self,
            pack: unsafe { *self.lib.get(b"ds41rt_v41_compressed_kv_pack")? },
            store: unsafe { *self.lib.get(b"ds41rt_v41_compressed_kv_store")? },
            value_bytes: 256,
            scale_bytes: 32,
        })
    }
}
impl V41Kv<'_> {
    /// # Safety
    /// Finite BF16 input [rows,512] and optional FP32 complex frequencies
    /// [rows,32,2] are live on the stream device. Rotated values remain finite.
    /// Disjoint values/scales receive output in the format selected at construction:
    /// window E4M3/E8M0 [rows,512/16], or compressed E2M1/E4M3 [rows,256/32].
    /// Compressed input and rotated magnitudes must be <=2688.
    pub unsafe fn pack(
        &self,
        input: Ds41rtDeviceBuffer,
        frequencies: Option<Ds41rtDeviceBuffer>,
        values: Ds41rtDeviceBuffer,
        scales: Ds41rtDeviceBuffer,
        rows: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!((1..=4096).contains(&rows), "invalid KV pack rows");
        check(&[
            (input, rows * 1024),
            (values, rows * self.value_bytes),
            (scales, rows * self.scale_bytes),
        ])?;
        if let Some(f) = frequencies {
            check(&[(input, rows * 1024), (f, rows * 256)])?;
        }
        let status = unsafe {
            (self.pack)(
                input.ptr.cast(),
                frequencies.map_or(std::ptr::null(), |f| f.ptr.cast()),
                values.ptr.cast(),
                scales.ptr.cast(),
                rows as i32,
                stream,
            )
        };
        ensure!(status == 0, "native KV packing status {status}");
        Ok(())
    }
    /// # Safety
    /// Initialized proposals and disjoint cache use the construction-selected format
    /// (512/16 or 256/32 value/scale bytes per row); U64 destinations [rows]
    /// and all buffers are live on the stream device. Valid
    /// destinations are unique; values >=capacity skip. Serialize cache writes
    /// and publish lengths/page metadata only after all accepted writes drain.
    pub unsafe fn store(
        &self,
        values: Ds41rtDeviceBuffer,
        scales: Ds41rtDeviceBuffer,
        destinations: Ds41rtDeviceBuffer,
        cache: Ds41rtDeviceBuffer,
        cache_scales: Ds41rtDeviceBuffer,
        rows: usize,
        capacity: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            (1..=4096).contains(&rows) && (1..=67108864).contains(&capacity),
            "invalid KV store shape"
        );
        check(&[
            (values, rows * self.value_bytes),
            (scales, rows * self.scale_bytes),
            (destinations, rows * 8),
            (cache, capacity * self.value_bytes),
            (cache_scales, capacity * self.scale_bytes),
        ])?;
        let status = unsafe {
            (self.store)(
                values.ptr.cast(),
                scales.ptr.cast(),
                destinations.ptr.cast(),
                cache.ptr.cast(),
                cache_scales.ptr.cast(),
                rows as i32,
                capacity as u64,
                stream,
            )
        };
        ensure!(status == 0, "native KV store status {status}");
        Ok(())
    }
}
