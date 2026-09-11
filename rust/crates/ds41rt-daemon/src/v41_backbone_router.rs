//! Official backbone routing with block-bound inputs and canonical TP4 requests.
use crate::v41_attention_binding::QueryBinding;
use crate::v41_block::FfnInput;
use crate::v41_layer_graphs::LayerGraphs;
use crate::v41_memory::{DeviceAllocation, LoadStream};
use crate::v41_tensors::NativeRtxTensors;
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary, V41ExpertInputQuantizer, V41Router};
use ds41rt_loader::OfficialV41Catalog;
use std::marker::PhantomData;
pub(crate) struct BackboneRouterWeights<'a> {
    library: &'a NativeLibrary,
    layer: usize,
    tensors: NativeRtxTensors<'a>,
}
impl<'a> BackboneRouterWeights<'a> {
    fn names(layer: usize) -> Result<Vec<String>> {
        ensure!(layer < 40, "invalid backbone router layer");
        Ok(["weight", "bias", "bias_vl"]
            .map(|n| format!("layers.{layer}.ffn.gate.{n}"))
            .to_vec())
    }
    pub fn device_bytes(catalog: &OfficialV41Catalog, layer: usize) -> Result<usize> {
        let bytes = NativeRtxTensors::plan(catalog, &Self::names(layer)?)?;
        ensure!(
            bytes == 3_935_232,
            "unexpected backbone router weight geometry"
        );
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
            Self::device_bytes(catalog, layer)? <= budget,
            "backbone router weights exceed budget"
        );
        Ok(Self {
            library,
            layer,
            tensors: NativeRtxTensors::load(
                library,
                catalog,
                &Self::names(layer)?,
                budget,
                staging,
            )?,
        })
    }

    pub fn wave(&self, capacity: u32, budget: usize) -> Result<BackboneRouterWave<'_, 'a>> {
        ensure!(
            BackboneRouterWave::device_bytes(capacity)? <= budget,
            "backbone router wave exceeds budget"
        );
        Ok(BackboneRouterWave {
            stream: LoadStream {
                library: self.library,
                raw: self.library.cuda_stream_create()?,
            },
            kernel: self.library.v41_router()?,
            input_quantizer: self.library.v41_expert_input_quantizer()?,
            buffers: [10240, 1, 1536, 24, 24, 5280]
                .into_iter()
                .map(|n| DeviceAllocation::new(self.library, n * capacity as usize))
                .collect::<Result<Vec<_>>>()?,
            weights: self,
            tokens: Vec::new(),
            layer: self.layer,
            capacity,
            graphs: LayerGraphs::new(self.library),
            ready: None,
            origin: None,
        })
    }
}
pub(crate) struct RouterOutput<'a> {
    pub layer: usize,
    pub rows: u32,
    pub input: Ds41rtDeviceBuffer,
    pub expert_input: Ds41rtDeviceBuffer,
    pub mask: Ds41rtDeviceBuffer,
    pub scores: Ds41rtDeviceBuffer,
    pub ids: Ds41rtDeviceBuffer,
    pub routing: Ds41rtDeviceBuffer,
    pub tokens: &'a [u64],
    origin: Option<QueryBinding>,
    _owner: PhantomData<&'a ()>,
}
impl RouterOutput<'_> {
    pub fn binding(&self) -> Result<QueryBinding> {
        self.origin
            .context("backbone router output has no block origin")
    }
}

#[cfg(test)]
mod reuse_tests {
    use super::*;
    use ds41rt_loader::{read_official_v41_catalog, OFFICIAL_V41_MODEL_ID};
    fn bytes(library: &NativeLibrary, buffer: Ds41rtDeviceBuffer) -> Result<Vec<u8>> {
        let mut result = vec![0; buffer.bytes];
        library.copy_d2h(&mut result, buffer)?;
        Ok(result)
    }
    #[test]
    fn real_router_rebinding_matches_fresh_owners() -> Result<()> {
        let Some(path) = std::env::var_os("DS41RT_ROUTER_REUSE_LIBRARY") else {
            eprintln!("skip router reuse GPU test: DS41RT_ROUTER_REUSE_LIBRARY unset");
            return Ok(());
        };
        let model = std::env::var_os("DS41RT_ROUTER_REUSE_MODEL")
            .context("DS41RT_ROUTER_REUSE_MODEL required")?;
        let library = unsafe { NativeLibrary::load(path)? };
        let catalog =
            read_official_v41_catalog(OFFICIAL_V41_MODEL_ID, std::path::Path::new(&model))?;
        let weights = (0..40)
            .map(|layer| {
                BackboneRouterWeights::load(&library, &catalog, layer, 3_935_232, 1024 * 1024)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut comparisons = 0;
        for rows in [1u32, 80, 4096] {
            let layers: Vec<usize> = if rows == 4096 {
                vec![0, 20, 39]
            } else {
                (0..40).collect()
            };
            let mut lane = weights[0].wave(rows, BackboneRouterWave::device_bytes(rows)?)?;
            let pointers = lane.inputs().map(|b| b.ptr);
            let mut handles = std::collections::HashMap::new();
            for cycle in 0..2usize {
                for &layer in &layers {
                    lane.rebind(&weights[layer])?;
                    assert!(lane.output().is_err());
                    assert_eq!(lane.inputs().map(|b| b.ptr), pointers);
                    let hidden: Vec<u8> = (0..rows as usize * 5120)
                        .flat_map(|i| {
                            let value =
                                (((i * 7 + cycle * 19 + layer * 3) % 31) as i32 - 15) as f32 / 128.;
                            ((value.to_bits() >> 16) as u16).to_ne_bytes()
                        })
                        .collect();
                    let mask: Vec<u8> = (0..rows as usize)
                        .map(|i| ((i + cycle + layer) % 2) as u8)
                        .collect();
                    let [input, modality] = lane.inputs();
                    library.copy_h2d(input, &hidden)?;
                    library.copy_h2d(modality, &mask)?;
                    let mut fresh =
                        weights[layer].wave(rows, BackboneRouterWave::device_bytes(rows)?)?;
                    let [fresh_input, fresh_mask] = fresh.inputs();
                    library.copy_d2d(fresh_input, input, input.bytes)?;
                    library.copy_d2d(fresh_mask, modality, modality.bytes)?;
                    let actual = unsafe { lane.execute_captured(rows)? };
                    let reference = unsafe { fresh.execute(rows)? };
                    assert_eq!(actual.layer, layer);
                    for (a, b) in [
                        (actual.scores, reference.scores),
                        (actual.ids, reference.ids),
                        (actual.routing, reference.routing),
                        (actual.expert_input, reference.expert_input),
                    ] {
                        assert_eq!(bytes(&library, a)?, bytes(&library, b)?);
                    }
                    for buffer in [actual.scores, actual.routing] {
                        assert!(bytes(&library, buffer)?
                            .chunks_exact(4)
                            .all(|b| f32::from_ne_bytes(b.try_into().unwrap()).is_finite()));
                    }
                    let handle = lane.graphs.get(layer, &weights[layer]).unwrap().0;
                    if cycle == 0 {
                        handles.insert(layer, handle);
                    } else {
                        assert_eq!(handles[&layer], handle);
                    }
                    comparisons += 1;
                }
                eprintln!("PASS real router rows={rows} cycle={cycle} layers={} scores/ids/routing/encoded-input exact; stable inputs and cached handles",layers.len());
            }
            assert!(unsafe { lane.execute_captured(0) }.is_err());
            assert!(lane.output().is_err());
            lane.rebind(&weights[0])?;
            unsafe {
                lane.execute_captured(rows)?;
            }
            lane.clear_graph()?;
            assert!(lane.output().is_err());
        }
        assert_eq!(comparisons, 166);
        eprintln!("PASS 166 real-weight router comparisons and invalid-row recovery");
        Ok(())
    }
}
pub(crate) struct BackboneRouterWave<'w, 'a> {
    stream: LoadStream<'a>,
    kernel: V41Router<'a>,
    input_quantizer: V41ExpertInputQuantizer<'a>,
    weights: &'w BackboneRouterWeights<'a>,
    buffers: Vec<DeviceAllocation<'a>>,
    tokens: Vec<u64>,
    layer: usize,
    capacity: u32,
    graphs: LayerGraphs<'w, 'a, BackboneRouterWeights<'a>>,
    ready: Option<u32>,
    origin: Option<QueryBinding>,
}
impl<'w, 'a> BackboneRouterWave<'w, 'a> {
    pub fn rebind(&mut self, weights: &'w BackboneRouterWeights<'a>) -> Result<()> {
        self.invalidate();
        ensure!(
            std::ptr::eq(self.stream.library, weights.library),
            "router rebound library differs"
        );
        for name in BackboneRouterWeights::names(weights.layer)? {
            ensure!(
                weights.tensors.get(&name)?.device_id == self.b(0).device_id,
                "router rebound weight device differs"
            );
        }
        self.synchronize()?;
        self.weights = weights;
        self.layer = weights.layer;
        Ok(())
    }
}
impl BackboneRouterWave<'_, '_> {
    pub fn device_bytes(capacity: u32) -> Result<usize> {
        ensure!(
            (1..=4096).contains(&capacity),
            "invalid backbone router capacity"
        );
        Ok(capacity as usize * (11825 + 5280))
    }
    fn b(&self, i: usize) -> Ds41rtDeviceBuffer {
        self.buffers[i].buffer
    }
    pub fn inputs(&self) -> [Ds41rtDeviceBuffer; 2] {
        [self.b(0), self.b(1)]
    }
    fn synchronize(&self) -> Result<()> {
        unsafe { self.stream.library.cuda_stream_synchronize(self.stream.raw) }
    }
    fn invalidate(&mut self) {
        self.ready = None;
        self.origin = None;
        self.tokens.clear();
    }
    fn validate(&mut self, rows: u32) -> Result<()> {
        self.invalidate();
        ensure!(
            rows > 0 && rows <= self.capacity,
            "backbone router rows exceed capacity"
        );
        Ok(())
    }
    unsafe fn enqueue(&mut self, rows: u32) -> Result<()> {
        let names = BackboneRouterWeights::names(self.layer)?;
        unsafe {
            self.kernel.launch(
                self.b(0),
                self.weights.tensors.get(&names[0])?,
                self.weights.tensors.get(&names[1])?,
                self.weights.tensors.get(&names[2])?,
                Some(self.b(1)),
                self.b(2),
                self.b(3),
                self.b(4),
                rows as usize,
                384,
                self.stream.raw,
            )?;
            self.input_quantizer
                .launch(self.b(0), self.b(5), rows, self.stream.raw)
        }
    }
    /// # Safety
    /// Hidden input is finite and initialized; mask bytes are 0 or 1 in the same
    /// row order. Both inputs are exclusively owned until the call drains.
    pub unsafe fn execute(&mut self, rows: u32) -> Result<RouterOutput<'_>> {
        self.validate(rows)?;
        let launched = unsafe { self.enqueue(rows) };
        launched.and(self.synchronize())?;
        self.ready = Some(rows);
        self.output()
    }
    /// # Safety
    /// Same contract as execute. Warmup is drained before capture.
    pub unsafe fn capture(&mut self, rows: u32) -> Result<()> {
        self.invalidate();
        ensure!(
            self.graphs.get(self.layer, self.weights).is_none(),
            "backbone router graph already captured"
        );
        unsafe {
            self.execute(rows)?;
        }
        self.invalidate();
        unsafe {
            self.stream
                .library
                .cuda_graph_begin_capture(self.stream.raw)?;
        }
        let launched = unsafe { self.enqueue(rows) };
        let captured = unsafe { self.stream.library.cuda_graph_end_capture(self.stream.raw) };
        match (launched, captured) {
            (Ok(()), Ok(graph)) => {
                if let Err(error) =
                    unsafe { self.graphs.insert(self.layer, self.weights, rows, graph) }
                {
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
    /// Same contract as execute; graph live row count must match.
    pub unsafe fn replay(&mut self, rows: u32) -> Result<RouterOutput<'_>> {
        self.validate(rows)?;
        let (graph, count) = self
            .graphs
            .get(self.layer, self.weights)
            .context("backbone router graph missing")?;
        ensure!(count == rows, "backbone router captured rows differ");
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
    /// The completed block input stays immutable through the copy. This wave has
    /// exclusive storage; image mask matches those tokens (0=text, 1=image).
    pub unsafe fn execute_ffn(
        &mut self,
        input: &FfnInput<'_>,
        image_mask: &[u8],
    ) -> Result<RouterOutput<'_>> {
        self.invalidate();
        ensure!(
            input.layer == self.layer
                && input.binding().layer() == self.layer
                && !input.tokens.is_empty()
                && input.tokens.len() <= self.capacity as usize
                && input.values.bytes == input.tokens.len() * 10240
                && input.values.device_id == self.b(0).device_id
                && image_mask.len() == input.tokens.len()
                && image_mask.iter().all(|&v| v <= 1),
            "backbone router block input differs"
        );
        self.stream
            .library
            .copy_d2d(self.b(0), input.values, input.values.bytes)?;
        self.stream.library.copy_h2d(self.b(1), image_mask)?;
        let rows = input.tokens.len() as u32;
        unsafe {
            self.execute_captured(rows)?;
        }
        self.origin = Some(input.binding());
        self.tokens.extend_from_slice(input.tokens);
        self.output()
    }
    /// # Safety
    /// Same initialized finite hidden and binary-mask contract as execute.
    pub unsafe fn execute_captured(&mut self, rows: u32) -> Result<RouterOutput<'_>> {
        self.invalidate();
        if self
            .graphs
            .get(self.layer, self.weights)
            .is_none_or(|(_, n)| n != rows)
        {
            self.clear_graph()?;
            unsafe {
                self.capture(rows)?;
            }
        }
        unsafe { self.replay(rows) }
    }
    pub fn output(&self) -> Result<RouterOutput<'_>> {
        let rows = self.ready.context("backbone router output unpublished")?;
        let b = |i: usize, n: usize| {
            let mut b = self.b(i);
            b.bytes = rows as usize * n;
            b
        };
        Ok(RouterOutput {
            layer: self.layer,
            rows,
            input: b(0, 10240),
            expert_input: b(5, 5280),
            mask: b(1, 1),
            scores: b(2, 1536),
            ids: b(3, 24),
            routing: b(4, 24),
            tokens: &self.tokens,
            origin: self.origin,
            _owner: PhantomData,
        })
    }
    /// Clear only this layer; other layers retain their captured shape.
    pub fn clear_graph(&mut self) -> Result<()> {
        self.invalidate();
        self.synchronize()?;
        unsafe {
            self.graphs.remove(self.layer)?;
        }
        Ok(())
    }
}
impl Drop for BackboneRouterWave<'_, '_> {
    fn drop(&mut self) {
        if let Err(e) = self
            .synchronize()
            .and_then(|()| unsafe { self.graphs.clear() })
        {
            tracing::error!(%e,"draining backbone router");
        }
    }
}

/// Scheduler metadata in exactly the block's token order.
pub(crate) struct ExpertRow {
    pub request_id: u64,
    pub position: u64,
    pub kind: ds41rt_transport::ExpertV2SourceKind,
}
/// An immutable host request owns its BF16 input and router result after D2H.
/// The private binding survives asynchronous transport and router-wave reuse.
pub(crate) struct BoundExpertRequest {
    request: ds41rt_transport::ExpertProtocolV2Request,
    binding: QueryBinding,
}
impl BoundExpertRequest {
    pub fn request(&self) -> &ds41rt_transport::ExpertProtocolV2Request {
        &self.request
    }
    pub fn binding(&self) -> QueryBinding {
        self.binding
    }
}
impl RouterOutput<'_> {
    /// # Safety
    /// Device views still hold this completed router execution with no external
    /// writes. Metadata identifies the actual requests represented by the block.
    pub unsafe fn expert_request(
        &self,
        library: &NativeLibrary,
        placement: u64,
        rows: &[ExpertRow],
    ) -> Result<BoundExpertRequest> {
        use ds41rt_transport::{
            ExpertProtocolV2Request, ExpertProtocolV2RouteEntry, ExpertProtocolV2RowDescriptor,
            ExpertV2Dtype,
        };
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let binding = self.binding()?;
        ensure!(
            binding.layer() == self.layer
                && rows.len() == self.rows as usize
                && self.tokens.len() == rows.len()
                && rows.iter().zip(self.tokens).all(|(r, &p)| r.position == p),
            "router request rows differ from block"
        );
        let request_id = NEXT
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .ok()
            .context("native expert request IDs exhausted")?;
        let mut hidden = vec![0; rows.len() * 5280];
        let mut ids = vec![0; rows.len() * 24];
        let mut weights = vec![0; rows.len() * 24];
        library.copy_d2h(&mut hidden, self.expert_input)?;
        library.copy_d2h(&mut ids, self.ids)?;
        library.copy_d2h(&mut weights, self.routing)?;
        let descriptors = rows
            .iter()
            .enumerate()
            .map(|(i, r)| ExpertProtocolV2RowDescriptor {
                row_id: i as u64,
                source_kind: r.kind,
                source_request_id: r.request_id,
                token_position: r.position,
                route_offset: i as u32 * 6,
                route_count: 6,
            })
            .collect();
        let routes = ids
            .chunks_exact(4)
            .zip(weights.chunks_exact(4))
            .enumerate()
            .map(|(i, (id, w))| ExpertProtocolV2RouteEntry {
                row_index: (i / 6) as u32,
                expert_id: u32::from_ne_bytes(id.try_into().unwrap()),
                gate_weight: f32::from_ne_bytes(w.try_into().unwrap()),
            })
            .collect();
        let mut request = ExpertProtocolV2Request::new(
            request_id,
            placement,
            self.layer as u32,
            5120,
            ExpertV2Dtype::Fp8E4m3Ue8m0K32,
            descriptors,
            routes,
            hidden,
        )?;
        request.header.flags |=
            ds41rt_transport::v41_expert::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
        // Prove the same complete-batch contract used by every Spark receiver.
        ds41rt_transport::v41_expert::V41BackboneRequest::parse(&request.encode()?, self.rows)?;
        Ok(BoundExpertRequest { request, binding })
    }
}
