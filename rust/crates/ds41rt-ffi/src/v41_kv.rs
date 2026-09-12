//! Fixed E4M3/E8M0 K32 serving KV encoding and accepted-row scatter.
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
        })
    }
}
impl V41Kv<'_> {
    /// # Safety
    /// Finite BF16 input [rows,512] and optional FP32 complex frequencies
    /// [rows,32,2] are live on the stream device. Rotated values remain finite.
    /// Disjoint E4M3 values [rows,512] and E8M0 scales [rows,16] receive output.
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
            (values, rows * 512),
            (scales, rows * 16),
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
    /// Initialized E4M3/E8M0 proposals [rows,512/16], U64 destinations [rows],
    /// disjoint cache [capacity,512/16] are live on the stream device. Valid
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
            (values, rows * 512),
            (scales, rows * 16),
            (destinations, rows * 8),
            (cache, capacity * 512),
            (cache_scales, capacity * 16),
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
