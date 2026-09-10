//! Causal candidate-block maxima and expansion for the official hierarchy.
use crate::{Ds41rtDeviceBuffer, NativeLibrary};
use anyhow::{ensure, Result};
use std::ffi::c_void;
type Max = unsafe extern "C" fn(
    *const f32,
    *const u64,
    *const u64,
    *mut f32,
    *mut u64,
    i32,
    i32,
    *mut c_void,
) -> i32;
type Tile = unsafe extern "C" fn(*mut u64, *mut u64, i32, i32, u64, *mut c_void) -> i32;
type Expand = unsafe extern "C" fn(*const i32, *const u64, *mut u64, i32, *mut c_void) -> i32;
pub struct V41CandidateBlocks<'a> {
    _library: &'a NativeLibrary,
    max: Max,
    expand: Expand,
    tile: Tile,
}
fn check(buffers: &[(Ds41rtDeviceBuffer, usize)]) -> Result<()> {
    for &(b, n) in buffers {
        ensure!(
            !b.ptr.is_null() && b.bytes >= n && b.device_id == buffers[0].0.device_id,
            "candidate block buffer is null, undersized or on another device"
        );
    }
    Ok(())
}
impl NativeLibrary {
    pub fn v41_candidate_blocks(&self) -> Result<V41CandidateBlocks<'_>> {
        Ok(V41CandidateBlocks {
            _library: self,
            max: unsafe { *self.lib.get(b"ds41rt_v41_candidate_block_max")? },
            expand: unsafe { *self.lib.get(b"ds41rt_v41_candidate_expand")? },
            tile: unsafe { *self.lib.get(b"ds41rt_v41_candidate_tile")? },
        })
    }
}
impl V41CandidateBlocks<'_> {
    /// # Safety
    /// Disjoint U64 positions [queries,width] and first [queries] are live on
    /// the stream device. Generate each history tile once, on aligned boundaries.
    pub unsafe fn tile(
        &self,
        positions: Ds41rtDeviceBuffer,
        first: Ds41rtDeviceBuffer,
        queries: usize,
        width: usize,
        begin: u64,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            (1..=4096).contains(&queries)
                && (1..=16384).contains(&width)
                && begin < 1048576
                && begin % 8 == 0,
            "invalid candidate tile shape"
        );
        check(&[(positions, queries * width * 8), (first, queries * 8)])?;
        let status = unsafe {
            (self.tile)(
                positions.ptr.cast(),
                first.ptr.cast(),
                queries as i32,
                width as i32,
                begin,
                stream,
            )
        };
        ensure!(status == 0, "native candidate tile status {status}");
        Ok(())
    }
    /// # Safety
    /// Scores [queries,width], U64 aligned first positions/causal lengths
    /// [queries] and disjoint FP32 maxima/U64 IDs [queries,ceil(width/8)] are live
    /// on the stream device. Tiles partition eight-row blocks; only the last
    /// tile can end mid-block. Inputs represent the same query/source snapshot.
    /// The newest reachable block is pinned to +inf. Malformed metadata skips.
    pub unsafe fn maxima(
        &self,
        scores: Ds41rtDeviceBuffer,
        first: Ds41rtDeviceBuffer,
        lengths: Ds41rtDeviceBuffer,
        maxima: Ds41rtDeviceBuffer,
        ids: Ds41rtDeviceBuffer,
        queries: usize,
        width: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            (1..=4096).contains(&queries) && (1..=16384).contains(&width),
            "invalid candidate block shape"
        );
        let blocks = width.div_ceil(8);
        check(&[
            (scores, queries * width * 4),
            (first, queries * 8),
            (lengths, queries * 8),
            (maxima, queries * blocks * 4),
            (ids, queries * blocks * 8),
        ])?;
        let status = unsafe {
            (self.max)(
                scores.ptr.cast(),
                first.ptr.cast(),
                lengths.ptr.cast(),
                maxima.ptr.cast(),
                ids.ptr.cast(),
                queries as i32,
                width as i32,
                stream,
            )
        };
        ensure!(status == 0, "native candidate maxima status {status}");
        Ok(())
    }
    /// # Safety
    /// Selected I32 blocks [queries,2048], U64 causal lengths [queries] and
    /// disjoint U64 output [queries,16384] are live on the stream device.
    /// Invalid/unreachable rows expand to u64::MAX. Selected blocks must come
    /// from the same query's candidate-source selection and be unique.
    pub unsafe fn expand(
        &self,
        blocks: Ds41rtDeviceBuffer,
        lengths: Ds41rtDeviceBuffer,
        positions: Ds41rtDeviceBuffer,
        queries: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            (1..=4096).contains(&queries),
            "invalid candidate expansion shape"
        );
        check(&[
            (blocks, queries * 8192),
            (lengths, queries * 8),
            (positions, queries * 131072),
        ])?;
        let status = unsafe {
            (self.expand)(
                blocks.ptr.cast(),
                lengths.ptr.cast(),
                positions.ptr.cast(),
                queries as i32,
                stream,
            )
        };
        ensure!(status == 0, "native candidate expansion status {status}");
        Ok(())
    }
}
