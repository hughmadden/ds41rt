//! Bounded top-512 reduction across disjoint index candidate tiles.
use crate::{Ds41rtDeviceBuffer, NativeLibrary};
use anyhow::{ensure, Result};
use std::ffi::c_void;
type TopK = unsafe extern "C" fn(
    *const f32,
    *const u64,
    *mut u64,
    *mut c_void,
    u64,
    *mut i32,
    i32,
    i32,
    i32,
    *mut c_void,
) -> i32;
pub struct V41IndexTopK<'a> {
    _library: &'a NativeLibrary,
    topk: TopK,
}
impl NativeLibrary {
    pub fn v41_index_topk(&self) -> Result<V41IndexTopK<'_>> {
        Ok(V41IndexTopK {
            _library: self,
            topk: unsafe { *self.lib.get(b"ds41rt_v41_index_top512")? },
        })
    }
}
impl V41IndexTopK<'_> {
    pub fn scratch_bytes(queries: usize, candidates: usize) -> Result<usize> {
        ensure!(
            (1..=4096).contains(&queries) && (1..=16384).contains(&candidates),
            "invalid index top-k shape"
        );
        Ok(queries * candidates.div_ceil(1024) * 512 * 8 * 2)
    }
    /// # Safety
    /// FP32 scores/U64 logical positions [queries,candidates], opaque U64 carry
    /// [queries,512], scratch and I32 output [queries,512] are disjoint, live on
    /// the stream device. All valid candidate positions are unique across the
    /// complete search. reset=true starts a new search; reset=false requires
    /// carry initialized by a prior update with identical query bindings/count.
    /// Serialize carry/scratch use, and reset after changing requests or inputs.
    /// Output contains ascending selected positions followed by -1 padding.
    pub unsafe fn update(
        &self,
        scores: Ds41rtDeviceBuffer,
        positions: Ds41rtDeviceBuffer,
        carry: Ds41rtDeviceBuffer,
        scratch: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        queries: usize,
        candidates: usize,
        reset: bool,
        stream: *mut c_void,
    ) -> Result<()> {
        let bytes = Self::scratch_bytes(queries, candidates)?;
        for (b, n) in [
            (scores, queries * candidates * 4),
            (positions, queries * candidates * 8),
            (carry, queries * 4096),
            (scratch, bytes),
            (output, queries * 2048),
        ] {
            ensure!(
                !b.ptr.is_null() && b.bytes >= n && b.device_id == scores.device_id,
                "index top-k buffer is null, undersized or on another device"
            );
        }
        let status = unsafe {
            (self.topk)(
                scores.ptr.cast(),
                positions.ptr.cast(),
                carry.ptr.cast(),
                scratch.ptr,
                scratch.bytes as u64,
                output.ptr.cast(),
                queries as i32,
                candidates as i32,
                i32::from(reset),
                stream,
            )
        };
        ensure!(status == 0, "native index top-k status {status}");
        Ok(())
    }
}
