//! Bounded, captured index selection and snapshot-checked candidate sharing.
use crate::v41_compressor::{IndexBinding, IndexProposal};
use crate::v41_index_query::IndexQueryOutput;
use crate::v41_memory::{DeviceAllocation, HostAllocation, LoadStream};
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{
    Ds41rtDeviceBuffer, NativeLibrary, V41CandidateBlocks, V41IndexScores, V41IndexTopK,
};
use std::{ffi::c_void, marker::PhantomData};
const WIDTH: usize = 16384;
const BLOCKS: usize = WIDTH / 8;

pub(crate) struct SelectionRequest<'a> {
    pub proposal: &'a IndexProposal<'a>,
    /// Increasing query token positions, in the query producer's row order.
    pub positions: &'a [u64],
}
pub(crate) struct IndexSelectionOutput<'a> {
    /// Logical compressed row IDs, sorted ascending, padded with -1. No window offset.
    pub selected: Ds41rtDeviceBuffer,
    pub rows: usize,
    layer: usize,
    blocks: Option<Ds41rtDeviceBuffer>,
    bindings: &'a [(IndexBinding, u64)],
    _sources: PhantomData<&'a ()>,
}
struct Ready {
    layer: usize,
    rows: usize,
    bindings: Vec<(IndexBinding, u64)>,
}
pub(crate) struct IndexSelectionWave<'a> {
    stream: LoadStream<'a>,
    buffers: Vec<DeviceAllocation<'a>>,
    staging: HostAllocation<'a>,
    capacity: usize,
    score: V41IndexScores<'a>,
    top: V41IndexTopK<'a>,
    candidates: V41CandidateBlocks<'a>,
    graph: Option<(*mut c_void, Vec<usize>)>,
    ready: Option<Ready>,
}
fn slice(mut b: Ds41rtDeviceBuffer, offset: usize, bytes: usize) -> Ds41rtDeviceBuffer {
    debug_assert!(offset + bytes <= b.bytes);
    b.ptr = unsafe { b.ptr.cast::<u8>().add(offset).cast() };
    b.bytes = bytes;
    b
}
impl<'a> IndexSelectionWave<'a> {
    fn sizes(capacity: usize) -> Result<[usize; 13]> {
        ensure!(
            (1..=4096).contains(&capacity),
            "invalid index selection capacity"
        );
        Ok([
            48,
            8,
            8,
            WIDTH * 8,
            WIDTH * 4,
            BLOCKS * 4,
            BLOCKS * 8,
            4096,
            V41IndexTopK::scratch_bytes(1, WIDTH)?,
            2048,
            16384,
            V41IndexTopK::block_scratch_bytes(1, BLOCKS)?,
            8192,
        ]
        .map(|n| n * capacity))
    }
    pub fn device_bytes(capacity: usize) -> Result<usize> {
        Ok(Self::sizes(capacity)?.iter().sum())
    }
    pub fn new(library: &'a NativeLibrary, capacity: usize, budget: usize) -> Result<Self> {
        ensure!(
            Self::device_bytes(capacity)? <= budget,
            "index selection exceeds budget"
        );
        Ok(Self {
            stream: LoadStream {
                library,
                raw: library.cuda_stream_create()?,
            },
            buffers: Self::sizes(capacity)?
                .into_iter()
                .map(|n| DeviceAllocation::new(library, n))
                .collect::<Result<_>>()?,
            staging: HostAllocation::new(library, capacity * 56)?,
            capacity,
            score: library.v41_index_scores()?,
            top: library.v41_index_topk()?,
            candidates: library.v41_candidate_blocks()?,
            graph: None,
            ready: None,
        })
    }
    fn b(&self, i: usize) -> Ds41rtDeviceBuffer {
        self.buffers[i].buffer
    }
    fn synchronize(&self) -> Result<()> {
        unsafe { self.stream.library.cuda_stream_synchronize(self.stream.raw) }
    }
    pub fn clear_graph(&mut self) -> Result<()> {
        self.ready = None;
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
        query: &IndexQueryOutput<'_>,
        requests: &[SelectionRequest<'_>],
        shared: Option<&IndexSelectionOutput<'_>>,
        rows: usize,
        tiles: usize,
        width: usize,
        use_candidates: bool,
    ) -> Result<()> {
        let source = query.layer == 20;
        let follower = use_candidates;
        let stream = self.stream.raw;
        let passes = if follower { 1 } else { tiles };
        for tile in 0..passes {
            unsafe {
                if follower {
                    self.candidates.expand(
                        shared
                            .context("candidate source missing")?
                            .blocks
                            .context("candidate blocks missing")?,
                        self.b(1),
                        self.b(3),
                        rows,
                        stream,
                    )?;
                } else {
                    self.candidates.tile(
                        self.b(3),
                        self.b(2),
                        rows,
                        width,
                        (tile * width) as u64,
                        stream,
                    )?;
                }
                let mut offset = 0;
                for request in requests {
                    let n = request.positions.len();
                    let p = request.proposal;
                    self.score.execute_overlay(
                        slice(query.packed, offset * 2048, n * 2048),
                        slice(query.scales, offset * 128, n * 128),
                        slice(query.head_weights, offset * 64, n * 64),
                        p.cache.packed,
                        p.cache.scales,
                        p.cache.device_pages,
                        p.cache.device_rows,
                        slice(self.b(0), offset * 48, n * 48),
                        slice(self.b(3), offset * width * 8, n * width * 8),
                        slice(self.b(4), offset * width * 4, n * width * 4),
                        p.packed,
                        p.scales,
                        p.capacity,
                        n,
                        width,
                        1,
                        p.cache.device_pages.bytes / 4,
                        p.cache.packed.bytes / 64,
                        stream,
                    )?;
                    offset += n;
                }
                self.top.update(
                    self.b(4),
                    self.b(3),
                    self.b(7),
                    self.b(8),
                    self.b(9),
                    rows,
                    width,
                    tile == 0,
                    stream,
                )?;
                if source {
                    self.candidates.maxima(
                        self.b(4),
                        self.b(2),
                        self.b(1),
                        self.b(5),
                        self.b(6),
                        rows,
                        width,
                        stream,
                    )?;
                    self.top.update_blocks(
                        self.b(5),
                        self.b(6),
                        self.b(10),
                        self.b(11),
                        self.b(12),
                        rows,
                        width / 8,
                        tile == 0,
                        stream,
                    )?;
                }
            }
        }
        Ok(())
    }
    /// # Safety
    /// Query rows are finite and correspond exactly to the supplied request/token
    /// order; their producers have drained. Proposal borrows are from the matching
    /// source's current wave. No external writes race inputs, caches or outputs.
    /// Results borrow the proposal views until their consumers have drained.
    pub unsafe fn execute<'s>(
        &'s mut self,
        query: &IndexQueryOutput<'_>,
        requests: &'s [SelectionRequest<'_>],
        shared: Option<&IndexSelectionOutput<'_>>,
    ) -> Result<IndexSelectionOutput<'s>> {
        self.ready = None;
        ensure!(
            [2, 8, 14, 20, 24, 28, 32, 36].contains(&query.layer),
            "invalid selection producer layer"
        );
        ensure!(
            !requests.is_empty() && requests.len() <= 16,
            "invalid selection request count"
        );
        let rows: usize = requests.iter().map(|r| r.positions.len()).sum();
        ensure!(
            rows > 0 && rows <= self.capacity,
            "selection rows exceed capacity"
        );
        ensure!(
            query.packed.bytes == rows * 2048
                && query.scales.bytes == rows * 128
                && query.head_weights.bytes == rows * 64,
            "query output row count differs"
        );
        ensure!(
            query.packed.device_id == self.b(0).device_id,
            "selection query device differs"
        );
        let expected_source = if query.layer >= 20 { 20 } else { query.layer };
        let mut bindings = Vec::with_capacity(rows);
        let mut max_length = 0u64;
        let mut fingerprint = vec![
            query.layer,
            rows,
            query.packed.ptr as usize,
            query.scales.ptr as usize,
            query.head_weights.ptr as usize,
        ];
        let mut metadata = Vec::with_capacity(rows);
        for (i, r) in requests.iter().enumerate() {
            ensure!(
                !r.positions.is_empty() && r.positions.windows(2).all(|x| x[0] < x[1]),
                "empty or unordered selection request"
            );
            ensure!(
                r.proposal.source_layer == expected_source,
                "index key source layer differs"
            );
            ensure!(
                requests[..i]
                    .iter()
                    .all(|x| !x.proposal.binding().same_request(r.proposal.binding())),
                "duplicate selection request"
            );
            ensure!(
                requests[0]
                    .proposal
                    .binding()
                    .same_pool(r.proposal.binding()),
                "selection state owner differs"
            );
            let p = r.proposal;
            fingerprint.extend([
                r.positions.len(),
                p.cache.packed.ptr as usize,
                p.cache.scales.ptr as usize,
                p.cache.device_pages.ptr as usize,
                p.cache.device_rows.ptr as usize,
                p.cache.device_pages.bytes,
                p.cache.packed.bytes,
                p.packed.ptr as usize,
                p.scales.ptr as usize,
                p.capacity,
            ]);
            for &position in r.positions {
                let m = p.metadata(position)?;
                max_length = max_length.max(m[1]);
                metadata.push(m);
                bindings.push((p.binding(), position));
            }
        }
        if query.layer > 20 {
            let s = shared.context("later index layer requires source candidates")?;
            ensure!(
                s.layer == 20 && s.rows == rows && s.bindings == bindings && s.blocks.is_some(),
                "candidate source snapshot or row order differs"
            );
            fingerprint.push(s.blocks.unwrap().ptr as usize);
        } else {
            ensure!(
                shared.is_none(),
                "candidate source is unexpected for this layer"
            );
        }
        let width = (max_length as usize).max(1).div_ceil(8).min(WIDTH / 8) * 8;
        let use_candidates = query.layer > 20 && max_length > WIDTH as u64;
        let tiles = (max_length as usize).div_ceil(width).max(1);
        fingerprint.extend([width, usize::from(use_candidates)]);
        fingerprint.push(if query.layer > 20 { 1 } else { tiles });
        let staging = self.staging.bytes_mut();
        for (i, m) in metadata.iter().enumerate() {
            for (j, value) in m.iter().enumerate() {
                staging[i * 48 + j * 8..i * 48 + j * 8 + 8].copy_from_slice(&value.to_ne_bytes());
            }
            staging[rows * 48 + i * 8..rows * 48 + i * 8 + 8].copy_from_slice(&m[1].to_ne_bytes());
        }
        self.stream
            .library
            .copy_h2d(self.b(0), &self.staging.bytes_mut()[..rows * 48])?;
        self.stream
            .library
            .copy_h2d(self.b(1), &self.staging.bytes_mut()[rows * 48..rows * 56])?;
        if self.graph.as_ref().map(|(_, f)| f) != Some(&fingerprint) {
            self.clear_graph()?;
            unsafe {
                self.stream
                    .library
                    .cuda_graph_begin_capture(self.stream.raw)?;
            }
            let launched = unsafe {
                self.enqueue(query, requests, shared, rows, tiles, width, use_candidates)
            };
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
        let g = self.graph.as_ref().context("selection graph missing")?.0;
        let launched = unsafe { self.stream.library.cuda_graph_launch(g, self.stream.raw) };
        launched.and(self.synchronize())?;
        self.ready = Some(Ready {
            layer: query.layer,
            rows,
            bindings,
        });
        let r = self.ready.as_ref().unwrap();
        Ok(IndexSelectionOutput {
            selected: slice(self.b(9), 0, rows * 2048),
            rows: r.rows,
            layer: r.layer,
            blocks: (r.layer == 20).then(|| slice(self.b(12), 0, rows * 8192)),
            bindings: &r.bindings,
            _sources: PhantomData,
        })
    }
}
impl Drop for IndexSelectionWave<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.clear_graph() {
            tracing::error!(%error,"draining index selection graph");
        }
    }
}
