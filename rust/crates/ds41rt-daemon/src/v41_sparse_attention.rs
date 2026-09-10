//! Captured sparse attention bound to live window/source/selection proposals.
use crate::v41_attention_binding::QueryBinding;
use crate::v41_attention_query::AttentionQueryOutput;
use crate::v41_compressor::IndexProposal;
use crate::v41_index_selection::IndexSelectionOutput;
use crate::v41_memory::{DeviceAllocation, HostAllocation, LoadStream};
use crate::v41_window::WindowProposal;
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{
    Ds41rtDeviceBuffer, NativeLibrary, V41SparseAttention, V41SparseSource, V41SparseWindow,
};
use std::{ffi::c_void, marker::PhantomData};

pub(crate) struct AttentionRequest<'a> {
    pub window: &'a WindowProposal<'a>,
    pub source: Option<&'a IndexProposal<'a>>,
    pub positions: &'a [u64],
}
pub(crate) struct SparseAttentionOutput<'a> {
    query: Option<QueryBinding>,
    tokens: Option<&'a [u64]>,
    pub values: Ds41rtDeviceBuffer,
    pub layer: usize,
    pub rows: usize,
    _inputs: PhantomData<&'a ()>,
}
impl SparseAttentionOutput<'_> {
    pub fn tokens(&self) -> Result<&[u64]> {
        let query = self.query.context("attention has no query origin")?;
        ensure!(query.layer() == self.layer, "attention query layer differs");
        self.tokens.context("attention tokens missing")
    }
}
pub(crate) struct SparseAttentionWave<'a> {
    stream: LoadStream<'a>,
    kernel: V41SparseAttention<'a>,
    query: DeviceAllocation<'a>,
    output: DeviceAllocation<'a>,
    metadata: DeviceAllocation<'a>,
    staging: HostAllocation<'a>,
    capacity: usize,
    graph: Option<(*mut c_void, Vec<usize>)>,
}
struct RequestLaunch {
    window: V41SparseWindow,
    source: Option<V41SparseSource>,
    rows: usize,
    width: usize,
}
fn slice(mut b: Ds41rtDeviceBuffer, offset: usize, bytes: usize) -> Ds41rtDeviceBuffer {
    debug_assert!(offset + bytes <= b.bytes);
    b.ptr = unsafe { b.ptr.cast::<u8>().add(offset).cast() };
    b.bytes = bytes;
    b
}
impl<'a> SparseAttentionWave<'a> {
    pub fn device_bytes(capacity: usize) -> Result<usize> {
        ensure!(
            (1..=4096).contains(&capacity),
            "invalid sparse attention capacity"
        );
        Ok(capacity * 131152)
    }
    pub fn new(library: &'a NativeLibrary, capacity: usize, budget: usize) -> Result<Self> {
        ensure!(
            Self::device_bytes(capacity)? <= budget,
            "sparse attention exceeds budget"
        );
        Ok(Self {
            stream: LoadStream {
                library,
                raw: library.cuda_stream_create()?,
            },
            kernel: library.v41_sparse_attention()?,
            query: DeviceAllocation::new(library, capacity * 65536)?,
            output: DeviceAllocation::new(library, capacity * 65536)?,
            metadata: DeviceAllocation::new(library, capacity * 80)?,
            staging: HostAllocation::new(library, capacity * 80)?,
            capacity,
            graph: None,
        })
    }
    pub fn input(&self) -> Ds41rtDeviceBuffer {
        self.query.buffer
    }
    fn synchronize(&self) -> Result<()> {
        unsafe { self.stream.library.cuda_stream_synchronize(self.stream.raw) }
    }
    pub fn clear_graph(&mut self) -> Result<()> {
        self.synchronize()?;
        if let Some((g, _)) = self.graph.take() {
            unsafe {
                self.stream.library.cuda_graph_exec_destroy(g)?;
            }
        }
        Ok(())
    }
    unsafe fn enqueue(
        &self,
        sink: Ds41rtDeviceBuffer,
        launches: &[RequestLaunch],
        selected: Option<Ds41rtDeviceBuffer>,
    ) -> Result<()> {
        let mut offset = 0;
        for l in launches {
            unsafe {
                self.kernel.launch(
                    slice(self.query.buffer, offset * 65536, l.rows * 65536),
                    sink,
                    slice(self.metadata.buffer, offset * 80, l.rows * 80),
                    selected.map(|b| slice(b, offset * 2048, l.rows * 2048)),
                    &l.window,
                    l.source.as_ref(),
                    slice(self.output.buffer, offset * 65536, l.rows * 65536),
                    l.rows,
                    l.width,
                    self.stream.raw,
                )?;
            }
            offset += l.rows;
        }
        Ok(())
    }
    /// # Safety
    /// No external writes race query, cache/selection views or this wave. Sink
    /// is the matching layer's finite checkpoint sink on this device.
    pub unsafe fn execute_query<'s>(
        &'s mut self,
        query: &'s AttentionQueryOutput<'s>,
        sink: Ds41rtDeviceBuffer,
        requests: &'s [AttentionRequest<'s>],
        selection: Option<&'s IndexSelectionOutput<'s>>,
    ) -> Result<SparseAttentionOutput<'s>> {
        let binding = query.binding()?;
        let tokens = query.tokens()?;
        ensure!(
            query.rows == tokens.len()
                && query.rows <= self.capacity
                && query.rotated.device_id == self.query.buffer.device_id
                && requests
                    .iter()
                    .flat_map(|r| r.positions.iter().copied())
                    .eq(tokens.iter().copied()),
            "attention query token order or device differs"
        );
        if let Some(s) = selection {
            s.validate_query(binding)?;
        }
        self.stream
            .library
            .copy_d2d(self.query.buffer, query.rotated, query.rotated.bytes)?;
        let mut out = unsafe { self.execute(query.layer, sink, requests, selection)? };
        out.query = Some(binding);
        out.tokens = Some(tokens);
        Ok(out)
    }
    /// # Safety
    /// Input contains finite, rotated BF16 attention queries in request/token
    /// order, with producer writes drained. Sink is this layer's finite FP32
    /// checkpoint tensor on the initialized device. No external writes race
    /// inputs or cache owners. Output retains proposal/selection borrows until
    /// its consumers finish, preventing accepted commits or producer reuse.
    pub unsafe fn execute<'s>(
        &'s mut self,
        layer: usize,
        sink: Ds41rtDeviceBuffer,
        requests: &'s [AttentionRequest<'s>],
        selection: Option<&'s IndexSelectionOutput<'s>>,
    ) -> Result<SparseAttentionOutput<'s>> {
        ensure!(
            layer < 40 && !requests.is_empty() && requests.len() <= 16,
            "invalid attention layer or request count"
        );
        let rows: usize = requests.iter().map(|r| r.positions.len()).sum();
        ensure!(
            rows > 0 && rows <= self.capacity,
            "attention rows exceed capacity"
        );
        ensure!(
            sink.bytes >= 256
                && !sink.ptr.is_null()
                && sink.device_id == self.query.buffer.device_id,
            "attention sink differs"
        );
        ensure!(
            selection.is_some() == (layer >= 2),
            "attention selection presence differs"
        );
        let source_layer = if layer >= 20 {
            20
        } else if layer >= 14 {
            14
        } else if layer >= 8 {
            8
        } else {
            2
        };
        let mut bindings = Vec::with_capacity(rows);
        let mut launches = Vec::with_capacity(requests.len());
        let mut metadata = Vec::with_capacity(rows * 10);
        let mut fingerprint = vec![layer, rows, sink.ptr as usize];
        for (i, r) in requests.iter().enumerate() {
            let w = r.window;
            ensure!(
                w.layer == layer && w.binding.same_pool(requests[0].window.binding),
                "attention window layer or owner differs"
            );
            ensure!(
                !r.positions.is_empty() && r.positions.windows(2).all(|p| p[0] < p[1]),
                "empty or unordered attention request"
            );
            ensure!(
                requests[..i].iter().all(|p| p.window.request != w.request),
                "duplicate attention request"
            );
            ensure!(
                r.source.is_some() == (layer >= 2),
                "attention source presence differs"
            );
            let mut width = 0;
            for &position in r.positions {
                let wm = w.metadata(position)?;
                width = width.max((position + 1).min(128) as usize);
                metadata.extend(wm);
                if let Some(s) = r.source {
                    ensure!(
                        s.request_id() == w.request
                            && s.source_layer == source_layer
                            && s.first_token() == wm[0],
                        "attention source request, layer or committed position differs"
                    );
                    ensure!(
                        s.binding()
                            .same_pool(requests[0].source.context("source absent")?.binding()),
                        "attention source owner differs"
                    );
                    metadata.extend(s.metadata(position)?);
                    bindings.push((s.binding(), position));
                } else {
                    metadata.extend([0; 6]);
                }
            }
            let window = V41SparseWindow {
                values: w.cache.values,
                scales: w.cache.scales,
                proposals: w.values,
                proposal_scales: w.scales,
                end: w.cache.device_end,
                proposal_capacity: w.capacity,
            };
            let source = r.source.map(|s| V41SparseSource {
                values: s.kv_cache.values,
                scales: s.kv_cache.scales,
                proposals: s.kv_values,
                proposal_scales: s.kv_scales,
                pages: s.kv_cache.device_pages,
                end: s.kv_cache.device_rows,
                capacity: s.kv_cache.values.bytes / 512,
                proposal_capacity: s.capacity,
                page_stride: s.kv_cache.device_pages.bytes / 4,
            });
            fingerprint.extend([r.positions.len(), width, window.proposal_capacity]);
            let mut buffers = vec![
                window.values,
                window.scales,
                window.proposals,
                window.proposal_scales,
                window.end,
            ];
            if let Some(s) = &source {
                fingerprint.extend([s.capacity, s.proposal_capacity, s.page_stride]);
                buffers.extend([
                    s.values,
                    s.scales,
                    s.proposals,
                    s.proposal_scales,
                    s.pages,
                    s.end,
                ]);
            }
            for b in buffers {
                ensure!(
                    b.device_id == self.query.buffer.device_id && !b.ptr.is_null(),
                    "attention proposal device differs"
                );
                fingerprint.extend([b.ptr as usize, b.bytes]);
            }
            launches.push(RequestLaunch {
                window,
                source,
                rows: r.positions.len(),
                width,
            });
        }
        let selected = if let Some(s) = selection {
            s.validate_attention(layer, &bindings)?;
            ensure!(
                s.selected.device_id == self.query.buffer.device_id
                    && s.selected.bytes == rows * 2048,
                "attention selection device or size differs"
            );
            fingerprint.push(s.selected.ptr as usize);
            Some(s.selected)
        } else {
            None
        };
        for (i, m) in metadata.into_iter().enumerate() {
            self.staging.bytes_mut()[i * 8..i * 8 + 8].copy_from_slice(&m.to_ne_bytes());
        }
        self.stream
            .library
            .copy_h2d(self.metadata.buffer, &self.staging.bytes_mut()[..rows * 80])?;
        if self.graph.as_ref().is_none_or(|(_, f)| f != &fingerprint) {
            self.clear_graph()?;
            let launched = unsafe { self.enqueue(sink, &launches, selected) };
            launched.and(self.synchronize())?;
            unsafe {
                self.stream
                    .library
                    .cuda_graph_begin_capture(self.stream.raw)?;
            }
            let launched = unsafe { self.enqueue(sink, &launches, selected) };
            let captured = unsafe { self.stream.library.cuda_graph_end_capture(self.stream.raw) };
            match (launched, captured) {
                (Ok(()), Ok(g)) => self.graph = Some((g, fingerprint)),
                (Err(e), Ok(g)) => {
                    unsafe {
                        self.stream.library.cuda_graph_exec_destroy(g)?;
                    }
                    return Err(e);
                }
                (Err(e), Err(_)) | (Ok(()), Err(e)) => return Err(e),
            }
        }
        let launched = unsafe {
            self.stream
                .library
                .cuda_graph_launch(self.graph.as_ref().unwrap().0, self.stream.raw)
        };
        launched.and(self.synchronize())?;
        Ok(SparseAttentionOutput {
            query: None,
            tokens: None,
            values: slice(self.output.buffer, 0, rows * 65536),
            layer,
            rows,
            _inputs: PhantomData,
        })
    }
}
impl Drop for SparseAttentionWave<'_> {
    fn drop(&mut self) {
        if let Err(e) = self.clear_graph() {
            tracing::error!(%e,"draining sparse attention graph");
        }
    }
}
