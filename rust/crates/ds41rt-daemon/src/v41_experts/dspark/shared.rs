//! Native shared-expert projections, packed scales and exclusive per-wave scratch.
use super::DsparkWeights;
use crate::v41_memory::{DeviceAllocation, LoadStream};
use crate::v41_tensors::NativeRtxTensors;
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary, V41Fp8Kernel, V41SharedSwiGlu};
use std::ffi::c_void;

pub(super) fn packed_scale_bytes(library: &NativeLibrary) -> Result<usize> {
    let up = library.v41_fp8_matrix_info(16, 5120, 2304)?;
    let down = library.v41_fp8_matrix_info(16, 2304, 5120)?;
    usize::try_from((2 * up.packed_weight_scale_bytes + down.packed_weight_scale_bytes) * 3)
        .context("shared FFN packed scale budget overflow")
}
pub(super) fn pack_scales<'a>(
    library: &'a NativeLibrary,
    tensors: &NativeRtxTensors<'a>,
) -> Result<[DeviceAllocation<'a>; 9]> {
    let up = library.v41_fp8_matrix_kernel(16, 5120, 2304)?;
    let down = library.v41_fp8_matrix_kernel(16, 2304, 5120)?;
    // The stream is declared after storage so error unwinding drains before free.
    let mut packed = Vec::with_capacity(9);
    let stream = LoadStream {
        library,
        raw: library.cuda_stream_create()?,
    };
    for stage in 0..3 {
        for (name, kernel) in [("w1", &up), ("w3", &up), ("w2", &down)] {
            packed.push(DeviceAllocation::new(
                library,
                kernel.info().packed_weight_scale_bytes as usize,
            )?);
            unsafe {
                kernel.pack_scales(
                    tensors.get(&format!("mtp.{stage}.ffn.shared_experts.{name}.scale"))?,
                    packed.last().unwrap().buffer,
                    stream.raw,
                )?;
            }
        }
    }
    unsafe {
        library.cuda_stream_synchronize(stream.raw)?;
    }
    packed
        .try_into()
        .ok()
        .context("expected nine shared FFN scale buffers")
}

pub(crate) struct DsparkSharedFfn<'weights, 'library> {
    weights: &'weights DsparkWeights<'library>,
    stage: usize,
    tensors: [Ds41rtDeviceBuffer; 3],
    up_kernel: V41Fp8Kernel<'library>,
    down_kernel: V41Fp8Kernel<'library>,
    activation: V41SharedSwiGlu<'library>,
    up_scratch: DeviceAllocation<'library>,
    down_scratch: DeviceAllocation<'library>,
    alpha: DeviceAllocation<'library>,
    gate: DeviceAllocation<'library>,
    up: DeviceAllocation<'library>,
    intermediate: DeviceAllocation<'library>,
    capacity: u32,
}
impl<'library> DsparkWeights<'library> {
    pub fn shared_ffn(
        &self,
        stage: usize,
        capacity: u32,
        budget: usize,
    ) -> Result<DsparkSharedFfn<'_, 'library>> {
        ensure!(stage < 3, "invalid shared FFN stage");
        let library = self.experts[stage].buffers[0].library;
        ensure!(
            DsparkSharedFfn::device_bytes(library, capacity)? <= budget,
            "shared FFN exceeds budget"
        );
        let up_kernel = library.v41_fp8_matrix_kernel(capacity, 5120, 2304)?;
        let down_kernel = library.v41_fp8_matrix_kernel(capacity, 2304, 5120)?;
        let value = DsparkSharedFfn {
            weights: self,
            stage,
            tensors: [
                self.tensor(&format!("mtp.{stage}.ffn.shared_experts.w1.weight"))?,
                self.tensor(&format!("mtp.{stage}.ffn.shared_experts.w3.weight"))?,
                self.tensor(&format!("mtp.{stage}.ffn.shared_experts.w2.weight"))?,
            ],
            up_scratch: DeviceAllocation::new(library, up_kernel.info().scratch_bytes as usize)?,
            down_scratch: DeviceAllocation::new(
                library,
                down_kernel.info().scratch_bytes as usize,
            )?,
            up_kernel,
            down_kernel,
            activation: library.v41_shared_swiglu()?,
            alpha: DeviceAllocation::new(library, 4)?,
            gate: DeviceAllocation::new(library, capacity as usize * 2304 * 2)?,
            up: DeviceAllocation::new(library, capacity as usize * 2304 * 2)?,
            intermediate: DeviceAllocation::new(library, capacity as usize * 2304 * 2)?,
            capacity,
        };
        let stream = LoadStream {
            library,
            raw: library.cuda_stream_create()?,
        };
        unsafe {
            value.up_kernel.initialize_scratch(
                value.up_scratch.buffer,
                value.alpha.buffer,
                stream.raw,
            )?;
            value.down_kernel.initialize_scratch(
                value.down_scratch.buffer,
                value.alpha.buffer,
                stream.raw,
            )?;
            library.cuda_stream_synchronize(stream.raw)?;
        }
        Ok(value)
    }
}
impl DsparkSharedFfn<'_, '_> {
    pub fn device_bytes(library: &NativeLibrary, capacity: u32) -> Result<usize> {
        let up = library.v41_fp8_matrix_info(capacity, 5120, 2304)?;
        let down = library.v41_fp8_matrix_info(capacity, 2304, 5120)?;
        usize::try_from(up.scratch_bytes + down.scratch_bytes + u64::from(capacity) * 2304 * 6 + 4)
            .context("shared FFN scratch budget overflow")
    }
    pub(in crate::v41_experts) fn matches(
        &self,
        weights: &crate::v41_experts::ExpertWeights<'_>,
    ) -> bool {
        std::ptr::eq(&self.weights.experts[self.stage], weights)
    }
    /// Enqueue on the owning expert wave stream; the caller must drain even on
    /// failure before releasing this exclusive borrow of scratch.
    pub(in crate::v41_experts) unsafe fn enqueue(
        &mut self,
        input: Ds41rtDeviceBuffer,
        output: Ds41rtDeviceBuffer,
        rows: u32,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            rows > 0 && rows <= self.capacity,
            "shared FFN rows exceed capacity"
        );
        let scales = &self.weights.shared_scales[self.stage * 3..self.stage * 3 + 3];
        unsafe {
            self.up_kernel.launch(
                input,
                self.tensors[0],
                scales[0].buffer,
                self.up_scratch.buffer,
                self.alpha.buffer,
                self.gate.buffer,
                rows,
                stream,
            )?;
            self.up_kernel.launch(
                input,
                self.tensors[1],
                scales[1].buffer,
                self.up_scratch.buffer,
                self.alpha.buffer,
                self.up.buffer,
                rows,
                stream,
            )?;
            self.activation.launch(
                self.gate.buffer,
                self.up.buffer,
                self.intermediate.buffer,
                rows,
                stream,
            )?;
            self.down_kernel.launch(
                self.intermediate.buffer,
                self.tensors[2],
                scales[2].buffer,
                self.down_scratch.buffer,
                self.alpha.buffer,
                output,
                rows,
                stream,
            )
        }
    }
}
