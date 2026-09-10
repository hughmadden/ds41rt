use crate::{Ds41rtDeviceBuffer, NativeLibrary};
use anyhow::{ensure, Result};
use std::ffi::c_void;
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct V41KvWrite {
    pub position: u64,
    pub source_row: u32,
    pub token_count: u32,
    pub slot: u32,
    pub reserved: u32,
}
const _: [(); 24] = [(); std::mem::size_of::<V41KvWrite>()];
type Launch =
    unsafe extern "C" fn(*const u16, *const V41KvWrite, *mut u8, i32, i32, *mut c_void) -> i32;
pub struct V41DsparkCache<'a> {
    _library: &'a NativeLibrary,
    launch: Launch,
}
impl NativeLibrary {
    pub fn v41_dspark_cache(&self) -> Result<V41DsparkCache<'_>> {
        Ok(V41DsparkCache {
            _library: self,
            launch: unsafe { *self.lib.get(b"ds41rt_v41_dspark_cache_write_fp8")? },
        })
    }
}
impl V41DsparkCache<'_> {
    /// E4M3 values followed by one E8M0 scale per 32 coordinates.
    pub const ROW_BYTES: usize = 528;
    pub const SLOT_BYTES: usize = 128 * Self::ROW_BYTES;
    /// # Safety
    /// Source is normalized and rotated finite BF16 KV. Sixteen initialized
    /// descriptors have valid source spans, unique active slots, and positions
    /// that do not overflow. Storage is distinct on the stream device and remains
    /// live and exclusively writable through completion and graph replay.
    pub unsafe fn launch(
        &self,
        source: Ds41rtDeviceBuffer,
        writes: Ds41rtDeviceBuffer,
        ring: Ds41rtDeviceBuffer,
        source_rows: u32,
        slots: u32,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            (1..=4096).contains(&source_rows) && (1..=16).contains(&slots),
            "invalid dSpark cache geometry"
        );
        for (buffer, bytes) in [
            (source, source_rows as usize * 1024),
            (writes, 384),
            (ring, slots as usize * Self::SLOT_BYTES),
        ] {
            ensure!(
                !buffer.ptr.is_null() && buffer.bytes >= bytes,
                "invalid dSpark cache buffer"
            );
        }
        let status = unsafe {
            (self.launch)(
                source.ptr.cast(),
                writes.ptr.cast(),
                ring.ptr.cast(),
                source_rows as i32,
                slots as i32,
                stream,
            )
        };
        ensure!(status == 0, "native dSpark cache CUDA status {status}");
        Ok(())
    }
}
