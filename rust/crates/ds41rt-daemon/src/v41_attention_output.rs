//! Backbone inverse rotary, grouped BF16 wo_a and native FP8 wo_b.
use crate::v41_memory::{DeviceAllocation, LoadStream};
use crate::v41_tensors::NativeRtxTensors;
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{
    Ds41rtDeviceBuffer, NativeLibrary, V41AttentionOps, V41Fp8Kernel, V41GroupedOutput,
    V41_GROUPED_OUTPUT_WORKSPACE,
};
use ds41rt_loader::OfficialV41Catalog;
use std::{ffi::c_void, marker::PhantomData};
const ROW_BYTES: [usize; 6] = [65536, 65536, 16384, 10240, 8, 256];
const GROUPED_WEIGHT: usize = 67108864;
pub(crate) struct AttentionOutputWeights<'a> {
    library: &'a NativeLibrary,
    layer: usize,
    names: [String; 4],
    tensors: NativeRtxTensors<'a>,
    grouped: DeviceAllocation<'a>,
    scales: DeviceAllocation<'a>,
}
impl<'a> AttentionOutputWeights<'a> {
    fn names(layer: usize) -> Result<[String; 4]> {
        ensure!(layer < 40, "invalid backbone output layer");
        Ok([
            format!("layers.{layer}.attn.wo_a.weight"),
            format!("layers.{layer}.attn.wo_a.scale"),
            format!("layers.{layer}.attn.wo_b.weight"),
            format!("layers.{layer}.attn.wo_b.scale"),
        ])
    }
    /// Maximum of transient dequantization allocation and final resident storage.
    pub fn device_bytes(
        library: &NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer: usize,
    ) -> Result<usize> {
        let names = Self::names(layer)?;
        let transient = NativeRtxTensors::plan(catalog, &names[..2])? + GROUPED_WEIGHT;
        let resident = NativeRtxTensors::plan(catalog, &names[2..])?
            + GROUPED_WEIGHT
            + library
                .v41_fp8_matrix_info(1, 8192, 5120)?
                .packed_weight_scale_bytes as usize;
        Ok(transient.max(resident))
    }
    pub fn load(
        library: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer: usize,
        budget: usize,
        staging: usize,
    ) -> Result<Self> {
        ensure!(
            Self::device_bytes(library, catalog, layer)? <= budget,
            "attention output weights exceed budget"
        );
        let names = Self::names(layer)?;
        let grouped = DeviceAllocation::new(library, GROUPED_WEIGHT)?;
        let stream = LoadStream {
            library,
            raw: library.cuda_stream_create()?,
        };
        {
            let source = NativeRtxTensors::load(
                library,
                catalog,
                &names[..2],
                budget - GROUPED_WEIGHT,
                staging,
            )?;
            let launched = unsafe {
                library.v41_grouped_output_dequant(
                    source.get(&names[0])?,
                    source.get(&names[1])?,
                    grouped.buffer,
                    stream.raw,
                )
            };
            launched.and(unsafe { library.cuda_stream_synchronize(stream.raw) })?;
        }
        // The FP8 wo_a source is released after dequantization drains, before wo_b loads.
        let kernel = library.v41_fp8_matrix_kernel(1, 8192, 5120)?;
        let scales =
            DeviceAllocation::new(library, kernel.info().packed_weight_scale_bytes as usize)?;
        let tensors = NativeRtxTensors::load(
            library,
            catalog,
            &names[2..],
            budget - GROUPED_WEIGHT - scales.buffer.bytes,
            staging,
        )?;
        let launched =
            unsafe { kernel.pack_scales(tensors.get(&names[3])?, scales.buffer, stream.raw) };
        launched.and(unsafe { library.cuda_stream_synchronize(stream.raw) })?;
        Ok(Self {
            library,
            layer,
            names,
            tensors,
            grouped,
            scales,
        })
    }
    pub fn wave(&self, capacity: u32, budget: usize) -> Result<AttentionOutputWave<'_, 'a>> {
        ensure!(
            AttentionOutputWave::device_bytes(self.library, capacity)? <= budget,
            "attention output wave exceeds budget"
        );
        let stream = LoadStream {
            library: self.library,
            raw: self.library.cuda_stream_create()?,
        };
        let kernel = self.library.v41_fp8_matrix_kernel(capacity, 8192, 5120)?;
        let workspace = DeviceAllocation::new(self.library, V41_GROUPED_OUTPUT_WORKSPACE)?;
        let grouped = unsafe { self.library.v41_grouped_output(workspace.buffer)? };
        let value = AttentionOutputWave {
            stream,
            grouped,
            _workspace: workspace,
            scratch: DeviceAllocation::new(self.library, kernel.info().scratch_bytes as usize)?,
            kernel,
            alpha: DeviceAllocation::new(self.library, 4)?,
            norm: self.library.v41_attention_ops()?,
            buffers: ROW_BYTES
                .into_iter()
                .map(|n| DeviceAllocation::new(self.library, n * capacity as usize))
                .collect::<Result<Vec<_>>>()?,
            weights: self,
            capacity,
            graph: None,
            ready: None,
        };
        ensure!(
            value.b(0).device_id == self.grouped.buffer.device_id,
            "output weight device differs"
        );
        let launched = unsafe {
            value.kernel.initialize_scratch(
                value.scratch.buffer,
                value.alpha.buffer,
                value.stream.raw,
            )
        };
        launched.and(value.synchronize())?;
        Ok(value)
    }
}
pub(crate) struct AttentionOutput<'a> {
    pub layer: usize,
    pub rows: usize,
    pub input: Ds41rtDeviceBuffer,
    pub rotated: Ds41rtDeviceBuffer,
    pub grouped: Ds41rtDeviceBuffer,
    pub projected: Ds41rtDeviceBuffer,
    pub positions: Ds41rtDeviceBuffer,
    pub frequencies: Ds41rtDeviceBuffer,
    _owner: PhantomData<&'a ()>,
}
pub(crate) struct AttentionOutputWave<'w, 'a> {
    stream: LoadStream<'a>,
    grouped: V41GroupedOutput<'a>,
    _workspace: DeviceAllocation<'a>,
    kernel: V41Fp8Kernel<'a>,
    scratch: DeviceAllocation<'a>,
    alpha: DeviceAllocation<'a>,
    norm: V41AttentionOps<'a>,
    buffers: Vec<DeviceAllocation<'a>>,
    weights: &'w AttentionOutputWeights<'a>,
    capacity: u32,
    graph: Option<(*mut c_void, u32)>,
    ready: Option<u32>,
}
impl AttentionOutputWave<'_, '_> {
    pub fn device_bytes(library: &NativeLibrary, capacity: u32) -> Result<usize> {
        Ok(V41_GROUPED_OUTPUT_WORKSPACE
            + 4
            + capacity as usize * ROW_BYTES.iter().sum::<usize>()
            + library
                .v41_fp8_matrix_info(capacity, 8192, 5120)?
                .scratch_bytes as usize)
    }
    fn b(&self, i: usize) -> Ds41rtDeviceBuffer {
        self.buffers[i].buffer
    }
    pub fn input(&self) -> Ds41rtDeviceBuffer {
        self.b(0)
    }
    pub fn positions(&self) -> Ds41rtDeviceBuffer {
        self.b(4)
    }
    fn synchronize(&self) -> Result<()> {
        unsafe { self.stream.library.cuda_stream_synchronize(self.stream.raw) }
    }
    fn validate(&mut self, rows: u32) -> Result<()> {
        self.ready = None;
        ensure!(
            rows > 0 && rows <= self.capacity,
            "attention output rows exceed capacity"
        );
        Ok(())
    }
    unsafe fn enqueue(&mut self, rows: u32) -> Result<()> {
        unsafe {
            self.norm.backbone_frequencies(
                self.b(4),
                self.b(5),
                rows,
                self.weights.layer as u32,
                self.stream.raw,
            )?;
            self.norm.rope(
                self.b(0),
                self.b(5),
                self.b(1),
                rows,
                64,
                true,
                self.stream.raw,
            )?;
            self.grouped.launch(
                self.b(1),
                self.weights.grouped.buffer,
                self.b(2),
                rows,
                self.stream.raw,
            )?;
            self.kernel.launch(
                self.b(2),
                self.weights.tensors.get(&self.weights.names[2])?,
                self.weights.scales.buffer,
                self.scratch.buffer,
                self.alpha.buffer,
                self.b(3),
                rows,
                self.stream.raw,
            )?;
        }
        Ok(())
    }
    /// # Safety
    /// Inputs are finite, initialized in matching row order on this device, with
    /// producer writes complete. Positions are below 1048576. No writes may race
    /// this wave; callers bind outputs to the matching request/query snapshot.
    pub unsafe fn execute(&mut self, rows: u32) -> Result<AttentionOutput<'_>> {
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
        ensure!(
            self.graph.is_none(),
            "attention output graph already captured"
        );
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
    pub unsafe fn replay(&mut self, rows: u32) -> Result<AttentionOutput<'_>> {
        self.validate(rows)?;
        let (graph, count) = self.graph.context("attention output graph missing")?;
        ensure!(count == rows, "attention output capture row count differs");
        let launched = unsafe {
            self.stream
                .library
                .cuda_graph_launch(graph, self.stream.raw)
        };
        launched.and(self.synchronize())?;
        self.ready = Some(rows);
        self.output()
    }
    pub fn output(&self) -> Result<AttentionOutput<'_>> {
        let rows = self.ready.context("attention output unpublished")? as usize;
        let b = |i| {
            let mut b = self.b(i);
            b.bytes = rows * ROW_BYTES[i];
            b
        };
        Ok(AttentionOutput {
            layer: self.weights.layer,
            rows,
            input: b(0),
            rotated: b(1),
            grouped: b(2),
            projected: b(3),
            positions: b(4),
            frequencies: b(5),
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
impl Drop for AttentionOutputWave<'_, '_> {
    fn drop(&mut self) {
        if let Err(error) = self.clear_graph() {
            tracing::error!(%error,"draining attention output graph");
        }
    }
}
