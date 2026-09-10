//! Native engram projection carriers and fused residual gate ownership.
use super::EngramDeviceView;
use crate::{
    v41_memory::{DeviceAllocation, LoadStream},
    v41_tensors::NativeRtxTensors,
};
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary};
use ds41rt_loader::OfficialV41Catalog;

pub(crate) struct EngramLayerWeights<'a> {
    tensors: NativeRtxTensors<'a>,
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
        let tensors =
            NativeRtxTensors::load(library, catalog, &names, device_budget, staging_bytes)?;
        Ok(Self {
            tensors,
            library,
            layer_index,
            names,
        })
    }
    /// Native FP8 [25600,6144] and UE8M0 [800,192] checkpoint carriers.
    /// They require K32 activation quantization and a native 32x32-scale GEMM.
    pub fn projection(&self) -> Result<(Ds41rtDeviceBuffer, Ds41rtDeviceBuffer)> {
        Ok((
            self.tensors.get(&self.names[0])?,
            self.tensors.get(&self.names[1])?,
        ))
    }
    pub fn resident_bytes(&self) -> usize {
        self.tensors.resident_bytes()
    }
}

pub(crate) struct EngramGate<'weights, 'library> {
    stream: LoadStream<'library>,
    output: DeviceAllocation<'library>,
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
        let bytes = capacity
            .checked_mul(4 * 5120 * 2)
            .context("engram gate budget overflow")?;
        ensure!(bytes <= device_budget, "engram gate exceeds device budget");
        Ok(Self {
            stream: LoadStream {
                library: weights.library,
                raw: weights.library.cuda_stream_create()?,
            },
            output: DeviceAllocation::new(weights.library, bytes)?,
            weights,
            capacity,
            ready_rows: None,
        })
    }
    /// Apply the official fused gate after native FP8 projection has completed.
    ///
    /// # Safety
    /// Residual BF16 [rows,4,5120], projected KV BF16 [rows,5,5120] and the
    /// gathered text mask must belong to this device and remain valid through
    /// this call. All producer writes must be complete; projected KV must come
    /// from this layer's wkv applied to the supplied gathered embeddings.
    pub unsafe fn apply(
        &mut self,
        residual: Ds41rtDeviceBuffer,
        projected_kv: Ds41rtDeviceBuffer,
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
            self.weights.library.cuda_engram_gate_bf16_async(
                residual,
                projected_kv,
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
