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
    blocks: TopK,
}
impl NativeLibrary {
    pub fn v41_index_topk(&self) -> Result<V41IndexTopK<'_>> {
        Ok(V41IndexTopK {
            _library: self,
            topk: unsafe { *self.lib.get(b"ds41rt_v41_index_top512")? },
            blocks: unsafe { *self.lib.get(b"ds41rt_v41_index_top2048_blocks")? },
        })
    }
}
impl V41IndexTopK<'_> {
    pub fn scratch_bytes(queries: usize, candidates: usize) -> Result<usize> {
        Self::scratch_for(queries, candidates, 512)
    }
    pub fn block_scratch_bytes(queries: usize, candidates: usize) -> Result<usize> {
        Self::scratch_for(queries, candidates, 2048)
    }
    fn scratch_for(queries: usize, candidates: usize, k: usize) -> Result<usize> {
        ensure!(
            (1..=4096).contains(&queries) && (1..=16384).contains(&candidates),
            "invalid index top-k shape"
        );
        Ok(queries * candidates.div_ceil(2 * k) * k * 8 * 2)
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
        unsafe {
            self.update_impl(
                scores, positions, carry, scratch, output, queries, candidates, reset, stream,
                self.topk, 512,
            )
        }
    }
    /// # Safety
    /// Same lifetime/disjointness/unique-candidate requirements as update, but
    /// positions are block IDs below 131072 and carry/output have width 2048.
    /// Never mix row-selection and block-selection accumulators. The newest
    /// reachable block's score must already be pinned by candidate_block_max.
    pub unsafe fn update_blocks(
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
        unsafe {
            self.update_impl(
                scores,
                positions,
                carry,
                scratch,
                output,
                queries,
                candidates,
                reset,
                stream,
                self.blocks,
                2048,
            )
        }
    }
    unsafe fn update_impl(
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
        op: TopK,
        k: usize,
    ) -> Result<()> {
        let bytes = Self::scratch_for(queries, candidates, k)?;
        for (b, n) in [
            (scores, queries * candidates * 4),
            (positions, queries * candidates * 8),
            (carry, queries * k * 8),
            (scratch, bytes),
            (output, queries * k * 4),
        ] {
            ensure!(
                !b.ptr.is_null() && b.bytes >= n && b.device_id == scores.device_id,
                "index top-k buffer is null, undersized or on another device"
            );
        }
        let status = unsafe {
            (op)(
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
