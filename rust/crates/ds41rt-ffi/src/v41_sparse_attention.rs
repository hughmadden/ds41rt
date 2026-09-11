//! Direct FP8 window/paged-source sparse attention with private proposal overlays.
use crate::{Ds41rtDeviceBuffer, NativeLibrary};
use anyhow::{ensure, Result};
use std::ffi::c_void;
#[repr(C)]
struct RawView {
    values: [*const u8; 4],
    scales: [*const u8; 4],
    window_end: *const u64,
    pages: *const u32,
    source_end: *const u64,
    window_proposal_capacity: u64,
    source_capacity: u64,
    source_proposal_capacity: u64,
    page_stride: u32,
    compressed: u32,
}
const _: [(); 120] = [(); std::mem::size_of::<RawView>()];
pub struct V41SparseSource {
    pub values: Ds41rtDeviceBuffer,
    pub scales: Ds41rtDeviceBuffer,
    pub proposals: Ds41rtDeviceBuffer,
    pub proposal_scales: Ds41rtDeviceBuffer,
    pub pages: Ds41rtDeviceBuffer,
    pub end: Ds41rtDeviceBuffer,
    pub capacity: usize,
    pub proposal_capacity: usize,
    pub page_stride: usize,
}
pub struct V41SparseWindow {
    pub values: Ds41rtDeviceBuffer,
    pub scales: Ds41rtDeviceBuffer,
    pub proposals: Ds41rtDeviceBuffer,
    pub proposal_scales: Ds41rtDeviceBuffer,
    pub end: Ds41rtDeviceBuffer,
    pub proposal_capacity: usize,
}
type Launch = unsafe extern "C" fn(
    *const u16,
    *const f32,
    *const u64,
    *const i32,
    *mut u16,
    i32,
    i32,
    *const RawView,
    *mut c_void,
) -> i32;
pub struct V41SparseAttention<'a> {
    _library: &'a NativeLibrary,
    launch: Launch,
}
impl NativeLibrary {
    pub fn v41_sparse_attention(&self) -> Result<V41SparseAttention<'_>> {
        let initialize = unsafe {
            *self
                .lib
                .get::<unsafe extern "C" fn() -> i32>(b"ds41rt_v41_sparse_attention_initialize")?
        };
        let status = unsafe { initialize() };
        ensure!(
            status == 0,
            "sparse attention initialization status {status}"
        );
        Ok(V41SparseAttention {
            _library: self,
            launch: unsafe { *self.lib.get(b"ds41rt_v41_sparse_attention")? },
        })
    }
}
impl V41SparseAttention<'_> {
    /// # Safety
    /// Rotated BF16 queries [rows,64,512], finite FP32 sinks [64], U64 metadata
    /// [rows,10] and optional I32 source IDs [rows,512] follow the native header.
    /// FP8 KV dequantizes to finite BF16. Caller binds every row to one request's
    /// live window/source leases and proposal snapshots; causal source lengths
    /// and unique IDs are correct. Inputs remain immutable on the initialized
    /// stream device through completion/replay, with disjoint BF16 output.
    /// Width zero derives the causal maximum from the last metadata row; this
    /// mode requires ascending query positions within the request.
    pub unsafe fn launch(
        &self,
        query: Ds41rtDeviceBuffer,
        sink: Ds41rtDeviceBuffer,
        metadata: Ds41rtDeviceBuffer,
        selected: Option<Ds41rtDeviceBuffer>,
        window: &V41SparseWindow,
        source: Option<&V41SparseSource>,
        output: Ds41rtDeviceBuffer,
        rows: usize,
        window_width: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            (1..=4096).contains(&rows) && window_width <= 128,
            "invalid sparse attention shape"
        );
        ensure!(
            (1..=4096).contains(&window.proposal_capacity),
            "invalid window proposal capacity"
        );
        ensure!(
            source.is_some() == selected.is_some(),
            "source and selection must be paired"
        );
        let check = |b: Ds41rtDeviceBuffer, n: usize| -> Result<()> {
            ensure!(
                !b.ptr.is_null() && b.bytes >= n && b.device_id == query.device_id,
                "sparse attention buffer is null, undersized or on another device"
            );
            Ok(())
        };
        for (b, n) in [
            (query, rows * 65536),
            (sink, 256),
            (metadata, rows * 80),
            (output, rows * 65536),
            (window.values, 128 * 512),
            (window.scales, 128 * 16),
            (window.end, 8),
            (window.proposals, window.proposal_capacity * 512),
            (window.proposal_scales, window.proposal_capacity * 16),
        ] {
            check(b, n)?;
        }
        let mut view = RawView {
            values: [
                window.values.ptr.cast(),
                window.proposals.ptr.cast(),
                std::ptr::null(),
                std::ptr::null(),
            ],
            scales: [
                window.scales.ptr.cast(),
                window.proposal_scales.ptr.cast(),
                std::ptr::null(),
                std::ptr::null(),
            ],
            window_end: window.end.ptr.cast(),
            pages: std::ptr::null(),
            source_end: std::ptr::null(),
            window_proposal_capacity: window.proposal_capacity as u64,
            source_capacity: 0,
            source_proposal_capacity: 0,
            page_stride: 0,
            compressed: 0,
        };
        if let Some(s) = source {
            ensure!(
                (1..=16777216).contains(&s.capacity)
                    && (1..=4096).contains(&s.proposal_capacity)
                    && (1..=4096).contains(&s.page_stride),
                "invalid sparse source shape"
            );
            for (b, n) in [
                (s.values, s.capacity * 512),
                (s.scales, s.capacity * 16),
                (s.proposals, s.proposal_capacity * 512),
                (s.proposal_scales, s.proposal_capacity * 16),
                (s.pages, s.page_stride * 4),
                (s.end, 8),
                (
                    selected.expect("checked paired source selection"),
                    rows * 2048,
                ),
            ] {
                check(b, n)?;
            }
            view.values[2] = s.values.ptr.cast();
            view.values[3] = s.proposals.ptr.cast();
            view.scales[2] = s.scales.ptr.cast();
            view.scales[3] = s.proposal_scales.ptr.cast();
            view.pages = s.pages.ptr.cast();
            view.source_end = s.end.ptr.cast();
            view.source_capacity = s.capacity as u64;
            view.source_proposal_capacity = s.proposal_capacity as u64;
            view.page_stride = s.page_stride as u32;
            view.compressed = 1;
        }
        let status = unsafe {
            (self.launch)(
                query.ptr.cast(),
                sink.ptr.cast(),
                metadata.ptr.cast(),
                selected.map_or(std::ptr::null(), |b| b.ptr.cast()),
                output.ptr.cast(),
                rows as i32,
                window_width as i32,
                &view,
                stream,
            )
        };
        ensure!(status == 0, "native sparse attention status {status}");
        Ok(())
    }
}
