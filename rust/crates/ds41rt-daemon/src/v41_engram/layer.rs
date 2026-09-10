//! Native engram projection carriers and fused residual gate ownership.
use super::EngramDeviceView;
use crate::{
    v41_memory::{DeviceAllocation, LoadStream},
    v41_tensors::NativeRtxTensors,
};
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary, V41Fp8Kernel};
use ds41rt_loader::OfficialV41Catalog;

pub(crate) struct EngramLayerWeights<'a> {
    tensors: NativeRtxTensors<'a>,
    packed_scales: DeviceAllocation<'a>,
    library: &'a NativeLibrary,
    layer_index: usize,
    names: [String; 4],
}
impl<'a> EngramLayerWeights<'a> {
    pub fn load(
        library: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer_index: usize,
        device_budget: usize,
        staging_bytes: usize,
    ) -> Result<Self> {
        let layer = *ds41rt_core::ENGRAM_LAYERS
            .get(layer_index)
            .context("invalid engram layer index")?;
        let names = ["wkv.weight", "wkv.scale", "q_weight", "k_weight"]
            .map(|suffix| format!("layers.{layer}.engram.{suffix}"));
        let kernel = library.v41_fp8_kernel(16)?;
        let packed_bytes = usize::try_from(kernel.info().packed_weight_scale_bytes)?;
        let required = NativeRtxTensors::plan(catalog, &names)?
            .checked_add(packed_bytes)
            .context("engram resident budget overflow")?;
        ensure!(
            required <= device_budget,
            "engram weights and packed scales exceed device budget"
        );
        let tensors = NativeRtxTensors::load(
            library,
            catalog,
            &names,
            device_budget - packed_bytes,
            staging_bytes,
        )?;
        let packed_scales = DeviceAllocation::new(library, packed_bytes)?;
        let stream = LoadStream {
            library,
            raw: library.cuda_stream_create()?,
        };
        unsafe {
            kernel.pack_scales(tensors.get(&names[1])?, packed_scales.buffer, stream.raw)?;
            library.cuda_stream_synchronize(stream.raw)?;
        }
        Ok(Self {
            tensors,
            packed_scales,
            library,
            layer_index,
            names,
        })
    }
    pub fn resident_bytes(&self) -> usize {
        self.tensors.resident_bytes() + self.packed_scales.buffer.bytes
    }
}

pub(crate) struct EngramGate<'weights, 'library> {
    stream: LoadStream<'library>,
    output: DeviceAllocation<'library>,
    projected: DeviceAllocation<'library>,
    scratch: DeviceAllocation<'library>,
    alpha: DeviceAllocation<'library>,
    kernel: V41Fp8Kernel<'library>,
    weights: &'weights EngramLayerWeights<'library>,
    capacity: usize,
    ready_rows: Option<usize>,
}
impl<'weights, 'library> EngramGate<'weights, 'library> {
    pub fn new(
        weights: &'weights EngramLayerWeights<'library>,
        capacity: usize,
        device_budget: usize,
    ) -> Result<Self> {
        ensure!(
            capacity > 0 && capacity <= 4096,
            "invalid engram gate capacity"
        );
        let kernel = weights.library.v41_fp8_kernel(u32::try_from(capacity)?)?;
        let scratch_bytes = usize::try_from(kernel.info().scratch_bytes)?;
        let output_bytes = capacity * 4 * 5120 * 2;
        let projected_bytes = capacity * 5 * 5120 * 2;
        let bytes = output_bytes
            .checked_add(projected_bytes)
            .and_then(|n| n.checked_add(scratch_bytes))
            .and_then(|n| n.checked_add(4))
            .context("engram execution budget overflow")?;
        ensure!(
            bytes <= device_budget,
            "engram execution exceeds device budget"
        );
        let value = Self {
            stream: LoadStream {
                library: weights.library,
                raw: weights.library.cuda_stream_create()?,
            },
            output: DeviceAllocation::new(weights.library, output_bytes)?,
            projected: DeviceAllocation::new(weights.library, projected_bytes)?,
            scratch: DeviceAllocation::new(weights.library, scratch_bytes)?,
            alpha: DeviceAllocation::new(weights.library, 4)?,
            kernel,
            weights,
            capacity,
            ready_rows: None,
        };
        unsafe {
            value.kernel.initialize_scratch(
                value.scratch.buffer,
                value.alpha.buffer,
                value.stream.raw,
            )?;
        }
        value.synchronize()?;
        Ok(value)
    }
    /// Quantize gathered rows by K32, project with native FP8 weights, then gate.
    ///
    /// # Safety
    /// Residual BF16 [rows,4,5120], gathered BF16 [rows,24,256] and text mask
    /// must belong to this device and remain valid through this call; producer
    /// writes must be complete and input buffers must not alias owned workspace.
    pub unsafe fn execute(
        &mut self,
        residual: Ds41rtDeviceBuffer,
        gathered: &EngramDeviceView,
    ) -> Result<Ds41rtDeviceBuffer> {
        self.ready_rows = None;
        ensure!(
            gathered.layer_index == self.weights.layer_index,
            "engram gate layer mismatch"
        );
        ensure!(
            gathered.rows > 0 && gathered.rows <= self.capacity,
            "engram gate exceeds capacity"
        );
        self.synchronize()?;
        unsafe {
            self.kernel.launch(
                gathered.embeddings,
                self.weights.tensors.get(&self.weights.names[0])?,
                self.weights.packed_scales.buffer,
                self.scratch.buffer,
                self.alpha.buffer,
                self.projected.buffer,
                u32::try_from(gathered.rows)?,
                self.stream.raw,
            )?;
            self.weights.library.cuda_engram_gate_bf16_async(
                residual,
                self.projected.buffer,
                self.weights.tensors.get(&self.weights.names[2])?,
                self.weights.tensors.get(&self.weights.names[3])?,
                Some(gathered.text_mask),
                self.output.buffer,
                i32::try_from(gathered.rows)?,
                self.stream.raw,
            )?;
        }
        self.synchronize()?;
        self.ready_rows = Some(gathered.rows);
        self.output()
    }
    /// Borrowed BF16 residual; never free or retain across reuse/drop.
    pub fn output(&self) -> Result<Ds41rtDeviceBuffer> {
        let rows = self
            .ready_rows
            .context("engram gate output is not complete")?;
        let mut output = self.output.buffer;
        output.bytes = rows * 4 * 5120 * 2;
        Ok(output)
    }
    fn synchronize(&self) -> Result<()> {
        unsafe {
            self.weights
                .library
                .cuda_stream_synchronize(self.stream.raw)
        }
    }
}
impl Drop for EngramGate<'_, '_> {
    fn drop(&mut self) {
        if let Err(error) = self.synchronize() {
            tracing::error!(%error, "draining native engram residual gate");
        }
    }
}
