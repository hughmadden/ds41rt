//! Paged committed-history FP4 index scoring over explicit logical candidates.
use crate::{Ds41rtDeviceBuffer, NativeLibrary};
use anyhow::{ensure, Result};
use std::ffi::c_void;
type Scores = unsafe extern "C" fn(
    *const u8,
    *const u8,
    *const u16,
    *const u8,
    *const u8,
    *const u32,
    *const u64,
    *const u64,
    *const u64,
    *mut f32,
    i32,
    i32,
    i32,
    i32,
    u64,
    *mut c_void,
) -> i32;
type Overlay = unsafe extern "C" fn(
    *const u8,
    *const u8,
    *const u16,
    *const u8,
    *const u8,
    *const u32,
    *const u64,
    *const u64,
    *const u64,
    *mut f32,
    *const u8,
    *const u8,
    i32,
    i32,
    i32,
    i32,
    u64,
    u64,
    *mut c_void,
) -> i32;
pub struct V41IndexScores<'a> {
    _library: &'a NativeLibrary,
    scores: Scores,
    overlay: Overlay,
}
impl NativeLibrary {
    pub fn v41_index_scores(&self) -> Result<V41IndexScores<'_>> {
        // Optional in older ABI-compatible native libraries. Prewarm AOT before capture.
        if let Ok(initialize) = unsafe { self.lib.get::<unsafe extern "C" fn() -> i32>(b"ds41rt_v41_index_scores_initialize") } {
            let status = unsafe { initialize() };
            anyhow::ensure!(status == 0, "V4.1 index scorer initialization failed: {status}");
        }
        Ok(V41IndexScores {
            _library: self,
            scores: unsafe { *self.lib.get(b"ds41rt_v41_index_scores")? },
            overlay: unsafe { *self.lib.get(b"ds41rt_v41_index_scores_overlay")? },
        })
    }
}
impl V41IndexScores<'_> {
    /// # Safety
    /// Initialized packed Q [queries,32,64]/E8M0 [queries,32,4], BF16 head
    /// weights [queries,32], key values/scales [capacity,64/4], U32 pages
    /// [slots,stride], U64 committed lengths [slots], metadata [queries,2]
    /// (slot,causal rows) and logical candidates [queries,candidates] are live
    /// on the stream device. Output FP32 [queries,candidates] is disjoint.
    /// Dequantized values and weights must be finite. Validate request leases
    /// before launch, and serialize with cache publication/release. Candidates
    /// outside the causal/committed/table bounds produce -infinity.
    pub unsafe fn execute(
        &self,
        q: Ds41rtDeviceBuffer,
        qs: Ds41rtDeviceBuffer,
        weights: Ds41rtDeviceBuffer,
        keys: Ds41rtDeviceBuffer,
        ks: Ds41rtDeviceBuffer,
        pages: Ds41rtDeviceBuffer,
        lengths: Ds41rtDeviceBuffer,
        metadata: Ds41rtDeviceBuffer,
        positions: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        queries: usize,
        candidates: usize,
        slots: usize,
        stride: usize,
        capacity: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            (1..=4096).contains(&queries)
                && (1..=16384).contains(&candidates)
                && (1..=16).contains(&slots)
                && (1..=4096).contains(&stride)
                && (1..=67108864).contains(&capacity),
            "invalid index scoring shape"
        );
        for (b, n) in [
            (q, queries * 2048),
            (qs, queries * 128),
            (weights, queries * 64),
            (keys, capacity * 64),
            (ks, capacity * 4),
            (pages, slots * stride * 4),
            (lengths, slots * 8),
            (metadata, queries * 16),
            (positions, queries * candidates * 8),
            (output, queries * candidates * 4),
        ] {
            ensure!(
                !b.ptr.is_null() && b.bytes >= n && b.device_id == q.device_id,
                "index score buffer is null, undersized or on another device"
            );
        }
        let status = unsafe {
            (self.scores)(
                q.ptr.cast(),
                qs.ptr.cast(),
                weights.ptr.cast(),
                keys.ptr.cast(),
                ks.ptr.cast(),
                pages.ptr.cast(),
                lengths.ptr.cast(),
                metadata.ptr.cast(),
                positions.ptr.cast(),
                output.ptr.cast(),
                queries as i32,
                candidates as i32,
                slots as i32,
                stride as i32,
                capacity as u64,
                stream,
            )
        };
        ensure!(status == 0, "native index scoring status {status}");
        Ok(())
    }
    /// # Safety
    /// Same finite device inputs and cache serialization as execute, plus immutable
    /// proposal values/scales [proposal_capacity,64/4]. Metadata [queries,6] is
    /// (slot,causal rows,committed start,proposal count,proposal offset,proposal step). The caller
    /// validates lease, owner, version and row mapping before launch; start must
    /// equal the device committed length. Malformed descriptors yield -infinity.
    /// Unaccepted proposal rows remain separate from committed storage.
    pub unsafe fn execute_overlay(
        &self,
        q: Ds41rtDeviceBuffer,
        qs: Ds41rtDeviceBuffer,
        weights: Ds41rtDeviceBuffer,
        keys: Ds41rtDeviceBuffer,
        ks: Ds41rtDeviceBuffer,
        pages: Ds41rtDeviceBuffer,
        lengths: Ds41rtDeviceBuffer,
        metadata: Ds41rtDeviceBuffer,
        positions: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        proposals: Ds41rtDeviceBuffer,
        proposal_scales: Ds41rtDeviceBuffer,
        proposal_capacity: usize,
        queries: usize,
        candidates: usize,
        slots: usize,
        stride: usize,
        capacity: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            (1..=4096).contains(&queries)
                && (1..=16384).contains(&candidates)
                && (1..=16).contains(&slots)
                && (1..=4096).contains(&stride)
                && (1..=67108864).contains(&capacity)
                && (1..=4096).contains(&proposal_capacity),
            "invalid index scoring shape"
        );
        for (b, n) in [
            (q, queries * 2048),
            (qs, queries * 128),
            (weights, queries * 64),
            (keys, capacity * 64),
            (ks, capacity * 4),
            (pages, slots * stride * 4),
            (lengths, slots * 8),
            (metadata, queries * 48),
            (proposals, proposal_capacity * 64),
            (proposal_scales, proposal_capacity * 4),
            (positions, queries * candidates * 8),
            (output, queries * candidates * 4),
        ] {
            ensure!(
                !b.ptr.is_null() && b.bytes >= n && b.device_id == q.device_id,
                "index score buffer is null, undersized or on another device"
            );
        }
        let status = unsafe {
            (self.overlay)(
                q.ptr.cast(),
                qs.ptr.cast(),
                weights.ptr.cast(),
                keys.ptr.cast(),
                ks.ptr.cast(),
                pages.ptr.cast(),
                lengths.ptr.cast(),
                metadata.ptr.cast(),
                positions.ptr.cast(),
                output.ptr.cast(),
                proposals.ptr.cast(),
                proposal_scales.ptr.cast(),
                queries as i32,
                candidates as i32,
                slots as i32,
                stride as i32,
                capacity as u64,
                proposal_capacity as u64,
                stream,
            )
        };
        ensure!(status == 0, "native index scoring status {status}");
        Ok(())
    }
}
