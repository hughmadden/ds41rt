//! Real backbone low-rank query projection, normalization and rotary graphs.
use crate::v41_attention_binding::QueryBinding;
use crate::v41_memory::{DeviceAllocation, LoadStream};
use crate::v41_tensors::NativeRtxTensors;
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary, V41AttentionOps, V41Fp8Kernel};
use ds41rt_loader::OfficialV41Catalog;
use std::{ffi::c_void, marker::PhantomData};
const MATRICES: [(u32, u32); 2] = [(5120, 1280), (1280, 32768)];
const ROW_BYTES: [usize; 7] = [10240, 2560, 2560, 65536, 65536, 8, 256];
fn part(mut b: Ds41rtDeviceBuffer, offset: usize, bytes: usize) -> Ds41rtDeviceBuffer {
    debug_assert!(offset + bytes <= b.bytes);
    b.ptr = unsafe { b.ptr.cast::<u8>().add(offset).cast() };
    b.bytes = bytes;
    b
}
pub(crate) struct AttentionQueryWeights<'a> {
    library: &'a NativeLibrary,
    layer: usize,
    names: [String; 5],
    tensors: NativeRtxTensors<'a>,
    scales: Vec<DeviceAllocation<'a>>,
}
impl<'a> AttentionQueryWeights<'a> {
    fn names(layer: usize) -> Result<[String; 5]> {
        ensure!(layer < 40, "invalid backbone query layer");
        Ok([
            format!("layers.{layer}.attn.wq_a.weight"),
            format!("layers.{layer}.attn.wq_a.scale"),
            format!("layers.{layer}.attn.wq_b.weight"),
            format!("layers.{layer}.attn.wq_b.scale"),
            format!("layers.{layer}.attn.q_norm.weight"),
        ])
    }
    pub fn device_bytes(
        library: &NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer: usize,
    ) -> Result<usize> {
        let mut bytes = NativeRtxTensors::plan(catalog, &Self::names(layer)?)?;
        for (k, n) in MATRICES {
            bytes += library
                .v41_fp8_matrix_info(1, k, n)?
                .packed_weight_scale_bytes as usize;
        }
        Ok(bytes)
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
            "attention query weights exceed budget"
        );
        let names = Self::names(layer)?;
        let tensors = NativeRtxTensors::load(library, catalog, &names, budget, staging)?;
        let stream = LoadStream {
            library,
            raw: library.cuda_stream_create()?,
        };
        let mut scales = vec![];
        for (i, (k, n)) in MATRICES.into_iter().enumerate() {
            let kernel = library.v41_fp8_matrix_kernel(1, k, n)?;
            let scale =
                DeviceAllocation::new(library, kernel.info().packed_weight_scale_bytes as usize)?;
            let launched = unsafe {
                kernel.pack_scales(tensors.get(&names[i * 2 + 1])?, scale.buffer, stream.raw)
            };
            launched.and(unsafe { library.cuda_stream_synchronize(stream.raw) })?;
            scales.push(scale);
        }
        Ok(Self {
            library,
            layer,
            names,
            tensors,
            scales,
        })
    }
    pub fn wave(&self, capacity: u32, budget: usize) -> Result<AttentionQueryWave<'_, 'a>> {
        ensure!(
            AttentionQueryWave::device_bytes(self.library, capacity)? <= budget,
            "attention query wave exceeds budget"
        );
        let stream = LoadStream {
            library: self.library,
            raw: self.library.cuda_stream_create()?,
        };
        let kernels = MATRICES
            .into_iter()
            .map(|(k, n)| self.library.v41_fp8_matrix_kernel(capacity, k, n))
            .collect::<Result<Vec<_>>>()?;
        let scratch = kernels
            .iter()
            .map(|k| DeviceAllocation::new(self.library, k.info().scratch_bytes as usize))
            .collect::<Result<Vec<_>>>()?;
        let value = AttentionQueryWave {
            stream,
            kernels,
            scratch,
            alpha: DeviceAllocation::new(self.library, 4)?,
            buffers: ROW_BYTES
                .into_iter()
                .map(|n| DeviceAllocation::new(self.library, n * capacity as usize))
                .collect::<Result<Vec<_>>>()?,
            norm: self.library.v41_attention_ops()?,
            weights: self,
            capacity,
            graph: None,
            ready: None,
            binding: None,
            tokens: Vec::new(),
        };
        ensure!(
            value.b(0).device_id == self.tensors.get(&self.names[0])?.device_id,
            "query weight device differs"
        );
        for i in 0..2 {
            let launched = unsafe {
                value.kernels[i].initialize_scratch(
                    value.scratch[i].buffer,
                    value.alpha.buffer,
                    value.stream.raw,
                )
            };
            launched.and(value.synchronize())?;
        }
        Ok(value)
    }
}
pub(crate) struct AttentionQueryOutput<'a> {
    binding: Option<QueryBinding>,
    tokens: &'a [u64],
    pub layer: usize,
    pub rows: usize,
    pub hidden: Ds41rtDeviceBuffer,
    pub raw_rank: Ds41rtDeviceBuffer,
    pub normalized_rank: Ds41rtDeviceBuffer,
    pub projected: Ds41rtDeviceBuffer,
    pub rotated: Ds41rtDeviceBuffer,
    pub positions: Ds41rtDeviceBuffer,
    pub frequencies: Ds41rtDeviceBuffer,
    _owner: PhantomData<&'a ()>,
}
impl AttentionQueryOutput<'_> {
    pub fn binding(&self) -> Result<QueryBinding> {
        self.binding.context("query has no token binding")
    }
    pub fn tokens(&self) -> Result<&[u64]> {
        self.binding()?;
        Ok(self.tokens)
    }
}
pub(crate) struct AttentionQueryWave<'w, 'a> {
    stream: LoadStream<'a>,
    kernels: Vec<V41Fp8Kernel<'a>>,
    scratch: Vec<DeviceAllocation<'a>>,
    alpha: DeviceAllocation<'a>,
    buffers: Vec<DeviceAllocation<'a>>,
    norm: V41AttentionOps<'a>,
    weights: &'w AttentionQueryWeights<'a>,
    capacity: u32,
    graph: Option<(*mut c_void, u32)>,
    ready: Option<u32>,
    binding: Option<QueryBinding>,
    tokens: Vec<u64>,
}
impl AttentionQueryWave<'_, '_> {
    pub fn device_bytes(library: &NativeLibrary, capacity: u32) -> Result<usize> {
        let mut bytes = 4 + capacity as usize * ROW_BYTES.iter().sum::<usize>();
        for (k, n) in MATRICES {
            bytes += library.v41_fp8_matrix_info(capacity, k, n)?.scratch_bytes as usize;
        }
        Ok(bytes)
    }
    fn b(&self, i: usize) -> Ds41rtDeviceBuffer {
        self.buffers[i].buffer
    }
    pub fn input(&self) -> Ds41rtDeviceBuffer {
        self.b(0)
    }
    pub fn positions(&self) -> Ds41rtDeviceBuffer {
        self.b(5)
    }
    fn synchronize(&self) -> Result<()> {
        unsafe { self.stream.library.cuda_stream_synchronize(self.stream.raw) }
    }
    fn validate(&mut self, rows: u32) -> Result<()> {
        self.ready = None;
        self.binding = None;
        self.tokens.clear();
        ensure!(
            rows > 0 && rows <= self.capacity,
            "query rows exceed capacity"
        );
        Ok(())
    }
    unsafe fn enqueue(&self, rows: u32) -> Result<()> {
        unsafe {
            self.norm.backbone_frequencies(
                self.b(5),
                self.b(6),
                rows,
                self.weights.layer as u32,
                self.stream.raw,
            )?;
            self.kernels[0].launch(
                self.b(0),
                self.weights.tensors.get(&self.weights.names[0])?,
                self.weights.scales[0].buffer,
                self.scratch[0].buffer,
                self.alpha.buffer,
                self.b(1),
                rows,
                self.stream.raw,
            )?;
            self.norm.norm(
                self.b(1),
                self.weights.tensors.get(&self.weights.names[4])?,
                None,
                self.b(2),
                rows,
                1280,
                self.stream.raw,
            )?;
            self.kernels[1].launch(
                self.b(2),
                self.weights.tensors.get(&self.weights.names[2])?,
                self.weights.scales[1].buffer,
                self.scratch[1].buffer,
                self.alpha.buffer,
                self.b(3),
                rows,
                self.stream.raw,
            )?;
            self.norm.rope(
                self.b(3),
                self.b(6),
                self.b(4),
                rows,
                64,
                false,
                self.stream.raw,
            )?;
        }
        Ok(())
    }
    /// # Safety
    /// Inputs are finite, initialized in matching row order on this device, with
    /// producer writes complete. Positions are below 1048576. No writes may race
    /// this wave; callers bind outputs to the matching request/query snapshot.
    pub unsafe fn execute(&mut self, rows: u32) -> Result<AttentionQueryOutput<'_>> {
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
        self.binding = None;
        self.tokens.clear();
        ensure!(
            self.graph.is_none(),
            "attention query graph already captured"
        );
        unsafe {
            self.execute(rows)?;
        }
        self.ready = None;
        self.binding = None;
        self.tokens.clear();
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
    pub unsafe fn replay(&mut self, rows: u32) -> Result<AttentionQueryOutput<'_>> {
        self.validate(rows)?;
        let (graph, count) = self.graph.context("attention query graph missing")?;
        ensure!(count == rows, "attention query capture row count differs");
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
    /// Hidden rows are finite attention inputs in the supplied token order, with
    /// producer writes drained. No external writes race this wave.
    pub unsafe fn execute_tokens(&mut self, tokens: &[u64]) -> Result<AttentionQueryOutput<'_>> {
        self.ready = None;
        self.binding = None;
        self.tokens.clear();
        ensure!(
            !tokens.is_empty()
                && tokens.len() <= self.capacity as usize
                && tokens.iter().all(|&p| p < 1048576),
            "invalid query tokens"
        );
        let binding = QueryBinding::new(self.weights.layer)?;
        self.stream.library.copy_h2d(
            self.positions(),
            &tokens
                .iter()
                .flat_map(|p| p.to_ne_bytes())
                .collect::<Vec<_>>(),
        )?;
        let rows = tokens.len() as u32;
        if self.graph.as_ref().is_none_or(|(_, n)| *n != rows) {
            self.clear_graph()?;
            unsafe {
                self.capture(rows)?;
            }
        }
        unsafe {
            self.replay(rows)?;
        }
        self.tokens.extend_from_slice(tokens);
        self.binding = Some(binding);
        self.output()
    }
    pub fn output(&self) -> Result<AttentionQueryOutput<'_>> {
        let rows = self.ready.context("attention query output unpublished")? as usize;
        let b = |i| part(self.b(i), 0, rows * ROW_BYTES[i]);
        Ok(AttentionQueryOutput {
            binding: self.binding,
            tokens: &self.tokens,
            layer: self.weights.layer,
            rows,
            hidden: b(0),
            raw_rank: b(1),
            normalized_rank: b(2),
            projected: b(3),
            rotated: b(4),
            positions: b(5),
            frequencies: b(6),
            _owner: PhantomData,
        })
    }
    pub fn clear_graph(&mut self) -> Result<()> {
        self.ready = None;
        self.binding = None;
        self.tokens.clear();
        self.synchronize()?;
        if let Some((graph, _)) = self.graph.take() {
            unsafe {
                self.stream.library.cuda_graph_exec_destroy(graph)?;
            }
        }
        Ok(())
    }
}
impl Drop for AttentionQueryWave<'_, '_> {
    fn drop(&mut self) {
        if let Err(error) = self.clear_graph() {
            tracing::error!(%error,"draining attention query graph");
        }
    }
}
