//! Learned index-query projections and fused rotary/FP4 preparation on the RTX.
use crate::v41_memory::{DeviceAllocation, LoadStream};
use crate::v41_tensors::NativeRtxTensors;
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary, V41AttentionOps, V41Compressor, V41Fp8Kernel};
use ds41rt_loader::OfficialV41Catalog;
use std::{ffi::c_void, marker::PhantomData};

pub(crate) struct IndexQueryWeights<'a> {
    library: &'a NativeLibrary,
    layer: usize,
    names: [String; 3],
    tensors: NativeRtxTensors<'a>,
    scales: DeviceAllocation<'a>,
}
impl<'a> IndexQueryWeights<'a> {
    fn names(layer: usize) -> Result<[String; 3]> {
        ensure!(
            [2, 8, 14, 20, 24, 28, 32, 36].contains(&layer),
            "layer has no learned index query producer"
        );
        Ok([
            format!("layers.{layer}.attn.indexer.wq_b.weight"),
            format!("layers.{layer}.attn.indexer.wq_b.scale"),
            format!("layers.{layer}.attn.indexer.weights_proj.weight"),
        ])
    }
    pub fn device_bytes(
        library: &NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer: usize,
    ) -> Result<usize> {
        let resident = NativeRtxTensors::plan(catalog, &Self::names(layer)?)?;
        ensure!(
            resident == 5_575_680,
            "unexpected native index query tensor sizes"
        );
        Ok(resident
            + library
                .v41_fp8_matrix_info(1, 1280, 4096)?
                .packed_weight_scale_bytes as usize)
    }
    pub fn load(
        library: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer: usize,
        budget: usize,
        staging_bytes: usize,
    ) -> Result<Self> {
        ensure!(
            Self::device_bytes(library, catalog, layer)? <= budget,
            "index query weights exceed budget"
        );
        let names = Self::names(layer)?;
        let tensors = NativeRtxTensors::load(library, catalog, &names, budget, staging_bytes)?;
        let kernel = library.v41_fp8_matrix_kernel(1, 1280, 4096)?;
        let scales =
            DeviceAllocation::new(library, kernel.info().packed_weight_scale_bytes as usize)?;
        let stream = LoadStream {
            library,
            raw: library.cuda_stream_create()?,
        };
        let launched =
            unsafe { kernel.pack_scales(tensors.get(&names[1])?, scales.buffer, stream.raw) };
        let drained = unsafe { library.cuda_stream_synchronize(stream.raw) };
        launched.and(drained)?;
        Ok(Self {
            library,
            layer,
            names,
            tensors,
            scales,
        })
    }
    pub fn wave(&self, capacity: u32, budget: usize) -> Result<IndexQueryWave<'_, 'a>> {
        ensure!(
            IndexQueryWave::device_bytes(self.library, capacity)? <= budget,
            "index query wave exceeds budget"
        );
        let fp8 = self.library.v41_fp8_matrix_kernel(capacity, 1280, 4096)?;
        let scratch = DeviceAllocation::new(self.library, fp8.info().scratch_bytes as usize)?;
        let workspace = DeviceAllocation::new(self.library, V41Compressor::WORKSPACE_BYTES)?;
        ensure!(
            workspace.buffer.device_id == self.tensors.get(&self.names[0])?.device_id,
            "index query weight device differs"
        );
        let dense = unsafe { self.library.v41_compressor(workspace.buffer)? };
        let rows = capacity as usize;
        let value = IndexQueryWave {
            stream: LoadStream {
                library: self.library,
                raw: self.library.cuda_stream_create()?,
            },
            fp8,
            dense,
            _workspace: workspace,
            scratch,
            alpha: DeviceAllocation::new(self.library, 4)?,
            norm: self.library.v41_attention_ops()?,
            weights: self,
            qr: DeviceAllocation::new(self.library, rows * 2560)?,
            hidden: DeviceAllocation::new(self.library, rows * 10240)?,
            positions: DeviceAllocation::new(self.library, rows * 8)?,
            frequencies: DeviceAllocation::new(self.library, rows * 256)?,
            projected: DeviceAllocation::new(self.library, rows * 8192)?,
            head_projected: DeviceAllocation::new(self.library, rows * 64)?,
            packed: DeviceAllocation::new(self.library, rows * 2048)?,
            scales: DeviceAllocation::new(self.library, rows * 128)?,
            head_weights: DeviceAllocation::new(self.library, rows * 64)?,
            capacity,
            graph: None,
            ready: None,
        };
        let launched = unsafe {
            value
                .fp8
                .initialize_scratch(value.scratch.buffer, value.alpha.buffer, value.stream.raw)
        };
        launched.and(value.synchronize())?;
        Ok(value)
    }
}
pub(crate) struct IndexQueryOutput<'a> {
    pub layer: usize,
    pub packed: Ds41rtDeviceBuffer,
    pub scales: Ds41rtDeviceBuffer,
    pub head_weights: Ds41rtDeviceBuffer,
    _owner: PhantomData<&'a ()>,
}
pub(crate) struct IndexQueryWave<'w, 'a> {
    stream: LoadStream<'a>,
    fp8: V41Fp8Kernel<'a>,
    dense: V41Compressor<'a>,
    _workspace: DeviceAllocation<'a>,
    scratch: DeviceAllocation<'a>,
    alpha: DeviceAllocation<'a>,
    norm: V41AttentionOps<'a>,
    weights: &'w IndexQueryWeights<'a>,
    qr: DeviceAllocation<'a>,
    hidden: DeviceAllocation<'a>,
    positions: DeviceAllocation<'a>,
    frequencies: DeviceAllocation<'a>,
    projected: DeviceAllocation<'a>,
    head_projected: DeviceAllocation<'a>,
    packed: DeviceAllocation<'a>,
    scales: DeviceAllocation<'a>,
    head_weights: DeviceAllocation<'a>,
    capacity: u32,
    graph: Option<(*mut c_void, u32)>,
    ready: Option<u32>,
}
impl IndexQueryWave<'_, '_> {
    pub fn device_bytes(library: &NativeLibrary, capacity: u32) -> Result<usize> {
        let info = library.v41_fp8_matrix_info(capacity, 1280, 4096)?;
        Ok(V41Compressor::WORKSPACE_BYTES
            + info.scratch_bytes as usize
            + 4
            + capacity as usize * 23560)
    }
    /// Normalized BF16 query-rank input [capacity,1280], shared with target attention.
    pub fn query_rank(&self) -> Ds41rtDeviceBuffer {
        self.qr.buffer
    }
    /// BF16 attention-input hidden states [capacity,5120], in the same row order.
    pub fn hidden(&self) -> Ds41rtDeviceBuffer {
        self.hidden.buffer
    }
    /// U64 token positions below the official context limit, in the same row order.
    pub fn positions(&self) -> Ds41rtDeviceBuffer {
        self.positions.buffer
    }
    fn synchronize(&self) -> Result<()> {
        unsafe { self.stream.library.cuda_stream_synchronize(self.stream.raw) }
    }
    fn validate(&mut self, rows: u32) -> Result<()> {
        self.ready = None;
        ensure!(
            rows > 0 && rows <= self.capacity,
            "index query rows exceed capacity"
        );
        Ok(())
    }
    unsafe fn enqueue(&self, rows: u32) -> Result<()> {
        unsafe {
            self.norm.backbone_frequencies(
                self.positions.buffer,
                self.frequencies.buffer,
                rows,
                self.weights.layer as u32,
                self.stream.raw,
            )?;
            self.fp8.launch(
                self.qr.buffer,
                self.weights.tensors.get(&self.weights.names[0])?,
                self.weights.scales.buffer,
                self.scratch.buffer,
                self.alpha.buffer,
                self.projected.buffer,
                rows,
                self.stream.raw,
            )?;
            self.dense.weights_project(
                self.hidden.buffer,
                self.weights.tensors.get(&self.weights.names[2])?,
                self.head_projected.buffer,
                rows as usize,
                self.stream.raw,
            )?;
            self.dense.query_prepare(
                self.projected.buffer,
                self.frequencies.buffer,
                self.head_projected.buffer,
                self.packed.buffer,
                self.scales.buffer,
                self.head_weights.buffer,
                rows as usize,
                self.stream.raw,
            )?;
        }
        Ok(())
    }
    /// # Safety
    /// Inputs are finite, initialized in matching row order on this device, with
    /// producer writes complete. Positions are below 1048576. No writes may race
    /// this wave; callers bind outputs to the matching request/query snapshot.
    pub unsafe fn execute(&mut self, rows: u32) -> Result<IndexQueryOutput<'_>> {
        self.validate(rows)?;
        let launched = unsafe { self.enqueue(rows) };
        launched.and(self.synchronize())?;
        self.ready = Some(rows);
        self.output()
    }
    /// # Safety
    /// Same inputs as execute. Warmup is drained but not published after capture.
    pub unsafe fn capture(&mut self, rows: u32) -> Result<()> {
        self.ready = None;
        ensure!(self.graph.is_none(), "index query graph already captured");
        unsafe {
            self.execute(rows)?;
        }
        self.ready = None;
        unsafe {
            self.stream
                .library
                .cuda_graph_begin_capture(self.stream.raw)?;
        }
        let launched = unsafe { self.enqueue(rows) };
        let captured = unsafe { self.stream.library.cuda_graph_end_capture(self.stream.raw) };
        match (launched, captured) {
            (Ok(()), Ok(graph)) => {
                self.graph = Some((graph, rows));
                Ok(())
            }
            (Err(e), Ok(graph)) => {
                unsafe {
                    self.stream.library.cuda_graph_exec_destroy(graph)?;
                }
                Err(e)
            }
            (Err(e), Err(_)) | (Ok(()), Err(e)) => Err(e),
        }
    }
    /// # Safety
    /// Same matching initialized inputs as execute. Captured live row count is fixed.
    pub unsafe fn replay(&mut self, rows: u32) -> Result<IndexQueryOutput<'_>> {
        self.validate(rows)?;
        let (graph, count) = self.graph.context("index query graph missing")?;
        ensure!(count == rows, "index query capture row count differs");
        let launched = unsafe {
            self.stream
                .library
                .cuda_graph_launch(graph, self.stream.raw)
        };
        launched.and(self.synchronize())?;
        self.ready = Some(rows);
        self.output()
    }
    pub fn output(&self) -> Result<IndexQueryOutput<'_>> {
        let rows = self.ready.context("index query output unpublished")? as usize;
        let sized = |mut b: Ds41rtDeviceBuffer, n: usize| {
            b.bytes = rows * n;
            b
        };
        Ok(IndexQueryOutput {
            layer: self.weights.layer,
            packed: sized(self.packed.buffer, 2048),
            scales: sized(self.scales.buffer, 128),
            head_weights: sized(self.head_weights.buffer, 64),
            _owner: PhantomData,
        })
    }
    pub fn clear_graph(&mut self) -> Result<()> {
        self.ready = None;
        self.synchronize()?;
        if let Some((graph, _)) = self.graph.take() {
            unsafe {
                self.stream.library.cuda_graph_exec_destroy(graph)?;
            }
        }
        Ok(())
    }
}
impl Drop for IndexQueryWave<'_, '_> {
    fn drop(&mut self) {
        if let Err(error) = self.clear_graph() {
            tracing::error!(%error,"draining index query graph");
        }
    }
}
