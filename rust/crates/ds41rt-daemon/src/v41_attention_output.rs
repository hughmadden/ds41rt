//! Backbone inverse rotary, grouped FP8 wo_a and native FP8 wo_b.
use crate::v41_attention_binding::QueryBinding;
use crate::v41_layer_graphs::LayerGraphs;
use crate::v41_memory::{DeviceAllocation, LoadStream};
use crate::v41_sparse_attention::SparseAttentionOutput;
use crate::v41_tensors::NativeRtxTensors;
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{
    Ds41rtDeviceBuffer, NativeLibrary, V41AttentionOps, V41Fp8Plan,
};
use ds41rt_loader::OfficialV41Catalog;
use std::marker::PhantomData;
const ROW_BYTES: [usize; 5] = [65536, 16384, 10240, 8, 256];
pub(crate) struct AttentionOutputWeights<'a> {
    library: &'a NativeLibrary,
    layer: usize,
    names: [String; 4],
    tensors: NativeRtxTensors<'a>,
    grouped_scales: DeviceAllocation<'a>,
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
    /// Official FP8 weights and packed scales for both output projections.
    pub fn device_bytes(
        library: &NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer: usize,
    ) -> Result<usize> {
        let names = Self::names(layer)?;
        Ok(NativeRtxTensors::plan(catalog, &names)?
            + library.v41_fp8_matrix_info(1, 32768, 8192)?.packed_weight_scale_bytes as usize
            + library.v41_fp8_matrix_info(1, 8192, 5120)?.packed_weight_scale_bytes as usize)
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
        let grouped_kernel = library.v41_fp8_matrix_kernel(1, 32768, 8192)?;
        let grouped_scales = DeviceAllocation::new(library, grouped_kernel.info().packed_weight_scale_bytes as usize)?;
        let stream = LoadStream { library, raw: library.cuda_stream_create()? };
        let kernel = library.v41_fp8_matrix_kernel(1, 8192, 5120)?;
        let scales =
            DeviceAllocation::new(library, kernel.info().packed_weight_scale_bytes as usize)?;
        let tensors = NativeRtxTensors::load(
            library,
            catalog,
            &names,
            budget - grouped_scales.buffer.bytes - scales.buffer.bytes,
            staging,
        )?;
        let launched = (|| unsafe {
            grouped_kernel.pack_scales(tensors.get(&names[1])?, grouped_scales.buffer, stream.raw)?;
            kernel.pack_scales(tensors.get(&names[3])?, scales.buffer, stream.raw)
        })();
        launched.and(unsafe { library.cuda_stream_synchronize(stream.raw) })?;
        Ok(Self {
            library,
            layer,
            names,
            tensors,
            grouped_scales,
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
        let kernel = self.library.v41_fp8_matrix_plan(capacity, 8192, 5120)?;
        let grouped = self.library.v41_fp8_matrix_plan(capacity, 32768, 8192)?;
        let grouped_scratch = DeviceAllocation::new(self.library, grouped.info().scratch_bytes as usize)?;
        let value = AttentionOutputWave {
            stream,
            grouped,
            grouped_scratch,
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
            graphs: LayerGraphs::new(self.library),
            ready: None,
            origin: None,
        };
        ensure!(
            value.b(0).device_id == self.grouped_scales.buffer.device_id,
            "output weight device differs"
        );
        let launched = (|| unsafe {
            value.grouped.initialize_scratch(value.grouped_scratch.buffer, value.alpha.buffer, value.stream.raw)?;
            value.kernel.initialize_scratch(
                value.scratch.buffer,
                value.alpha.buffer,
                value.stream.raw,
            )
        })();
        launched.and(value.synchronize())?;
        Ok(value)
    }
}
pub(crate) struct AttentionOutput<'a> {
    origin: Option<QueryBinding>,
    pub layer: usize,
    pub rows: usize,
    pub input: Ds41rtDeviceBuffer,
    pub grouped: Ds41rtDeviceBuffer,
    pub projected: Ds41rtDeviceBuffer,
    pub positions: Ds41rtDeviceBuffer,
    pub frequencies: Ds41rtDeviceBuffer,
    _owner: PhantomData<&'a ()>,
}
impl AttentionOutput<'_> {
    pub fn binding(&self) -> Result<QueryBinding> {
        let b = self
            .origin
            .context("attention projection has no query origin")?;
        ensure!(
            b.layer() == self.layer,
            "attention projection layer differs"
        );
        Ok(b)
    }
}
pub(crate) struct AttentionOutputWave<'w, 'a> {
    stream: LoadStream<'a>,
    grouped: V41Fp8Plan<'a>,
    grouped_scratch: DeviceAllocation<'a>,
    kernel: V41Fp8Plan<'a>,
    scratch: DeviceAllocation<'a>,
    alpha: DeviceAllocation<'a>,
    norm: V41AttentionOps<'a>,
    buffers: Vec<DeviceAllocation<'a>>,
    weights: &'w AttentionOutputWeights<'a>,
    capacity: u32,
    graphs: LayerGraphs<'w, 'a, AttentionOutputWeights<'a>>,
    ready: Option<u32>,
    origin: Option<QueryBinding>,
}
impl<'w, 'a> AttentionOutputWave<'w, 'a> {
    /// Reuse this lane's storage with another layer's weights. Existing output
    /// borrows must end first. Cached graphs retain their original weight owners.
    pub fn rebind(&mut self, weights: &'w AttentionOutputWeights<'a>) -> Result<()> {
        self.ready = None;
        self.origin = None;
        ensure!(
            std::ptr::eq(self.stream.library, weights.library)
                && weights.grouped_scales.buffer.device_id == self.b(0).device_id,
            "attention rebound weight library or device differs"
        );
        self.synchronize()?;
        self.weights = weights;
        Ok(())
    }
}
impl AttentionOutputWave<'_, '_> {
    pub fn device_bytes(library: &NativeLibrary, capacity: u32) -> Result<usize> {
        Ok(library.v41_fp8_matrix_plan_info(capacity, 32768, 8192)?.scratch_bytes as usize
            + 4
            + capacity as usize * ROW_BYTES.iter().sum::<usize>()
            + library
                .v41_fp8_matrix_plan_info(capacity, 8192, 5120)?
                .scratch_bytes as usize)
    }
    fn b(&self, i: usize) -> Ds41rtDeviceBuffer {
        self.buffers[i].buffer
    }
    pub fn input(&self) -> Ds41rtDeviceBuffer {
        self.b(0)
    }
    pub fn positions(&self) -> Ds41rtDeviceBuffer {
        self.b(3)
    }
    fn synchronize(&self) -> Result<()> {
        unsafe { self.stream.library.cuda_stream_synchronize(self.stream.raw) }
    }
    fn validate(&mut self, rows: u32) -> Result<()> {
        self.ready = None;
        self.origin = None;
        ensure!(
            rows > 0 && rows <= self.capacity,
            "attention output rows exceed capacity"
        );
        Ok(())
    }
    unsafe fn enqueue(&mut self, rows: u32) -> Result<()> {
        unsafe {
            self.norm.backbone_frequencies(
                self.b(3),
                self.b(4),
                rows,
                self.weights.layer as u32,
                self.stream.raw,
            )?;
            self.grouped.launch_rope(
                self.b(0),
                self.b(4),
                self.weights.tensors.get(&self.weights.names[0])?,
                self.weights.grouped_scales.buffer,
                self.grouped_scratch.buffer,
                self.alpha.buffer,
                self.b(1),
                rows,
                self.stream.raw,
            )?;
            self.kernel.launch(
                self.b(1),
                self.weights.tensors.get(&self.weights.names[2])?,
                self.weights.scales.buffer,
                self.scratch.buffer,
                self.alpha.buffer,
                self.b(2),
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
        self.origin = None;
        ensure!(
            self.graphs.get(self.weights.layer, self.weights).is_none(),
            "attention output graph already captured"
        );
        unsafe {
            self.execute(rows)?;
        }
        self.ready = None;
        self.origin = None;
        unsafe {
            self.stream
                .library
                .cuda_graph_begin_capture(self.stream.raw)?;
        }
        let launched = unsafe { self.enqueue(rows) };
        let captured = unsafe { self.stream.library.cuda_graph_end_capture(self.stream.raw) };
        match (launched, captured) {
            (Ok(()), Ok(graph)) => {
                if let Err(error) = unsafe {
                    self.graphs
                        .insert(self.weights.layer, self.weights, rows, graph)
                } {
                    unsafe {
                        self.stream.library.cuda_graph_exec_destroy(graph)?;
                    }
                    return Err(error);
                }
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
        let (graph, count) = self
            .graphs
            .get(self.weights.layer, self.weights)
            .context("attention output graph missing")?;
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
    /// # Safety
    /// No external writes race this wave or the completed attention output.
    pub unsafe fn execute_attention(
        &mut self,
        attention: &SparseAttentionOutput<'_>,
    ) -> Result<AttentionOutput<'_>> {
        self.ready = None;
        self.origin = None;
        let tokens = attention.tokens()?;
        ensure!(
            attention.layer == self.weights.layer
                && attention.rows == tokens.len()
                && attention.rows <= self.capacity as usize
                && attention.values.device_id == self.b(0).device_id,
            "attention output origin differs"
        );
        self.stream
            .library
            .copy_d2d(self.b(0), attention.values, attention.values.bytes)?;
        self.stream.library.copy_h2d(
            self.positions(),
            &tokens
                .iter()
                .flat_map(|p| p.to_ne_bytes())
                .collect::<Vec<_>>(),
        )?;
        let rows = attention.rows as u32;
        if self
            .graphs
            .get(self.weights.layer, self.weights)
            .is_none_or(|(_, n)| n != rows)
        {
            self.clear_graph()?;
            unsafe {
                self.capture(rows)?;
            }
        }
        unsafe {
            self.replay(rows)?;
        }
        self.origin = Some(attention.binding()?);
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
            origin: self.origin,
            layer: self.weights.layer,
            rows,
            input: b(0),
            grouped: b(1),
            projected: b(2),
            positions: b(3),
            frequencies: b(4),
            _owner: PhantomData,
        })
    }
    /// Evict only the current layer; other layers retain one captured shape each.
    pub fn clear_graph(&mut self) -> Result<()> {
        self.ready = None;
        self.origin = None;
        self.synchronize()?;
        unsafe {
            self.graphs.remove(self.weights.layer)?;
        }
        Ok(())
    }
}
impl Drop for AttentionOutputWave<'_, '_> {
    fn drop(&mut self) {
        if let Err(error) = self
            .synchronize()
            .and_then(|()| unsafe { self.graphs.clear() })
        {
            tracing::error!(%error,"draining attention output graph");
        }
    }
}
