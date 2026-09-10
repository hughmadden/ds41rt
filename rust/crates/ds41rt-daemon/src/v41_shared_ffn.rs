//! Shared native FP8 SwiGLU execution for backbone and dSpark weights.
use crate::v41_memory::{DeviceAllocation, LoadStream};
use crate::v41_tensors::NativeRtxTensors;
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary, V41Fp8Kernel, V41SharedSwiGlu};
use std::ffi::c_void;

pub(crate) struct SharedFfn<'weights, 'library> {
    _weights: &'weights NativeRtxTensors<'library>,
    scales: &'weights [DeviceAllocation<'library>],
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
impl<'weights, 'library> SharedFfn<'weights, 'library> {
    pub fn new(
        library: &'library NativeLibrary,
        weights: &'weights NativeRtxTensors<'library>,
        prefix: &str,
        scales: &'weights [DeviceAllocation<'library>],
        capacity: u32,
        budget: usize,
    ) -> Result<Self> {
        ensure!(scales.len() == 3, "shared FFN requires three packed scales");
        ensure!(
            Self::device_bytes(library, capacity)? <= budget,
            "shared FFN exceeds budget"
        );
        let up_kernel = library.v41_fp8_matrix_kernel(capacity, 5120, 2304)?;
        let down_kernel = library.v41_fp8_matrix_kernel(capacity, 2304, 5120)?;
        let value = SharedFfn {
            _weights: weights,
            scales,
            tensors: [
                weights.get(&format!("{prefix}.w1.weight"))?,
                weights.get(&format!("{prefix}.w3.weight"))?,
                weights.get(&format!("{prefix}.w2.weight"))?,
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
impl SharedFfn<'_, '_> {
    pub fn device_bytes(library: &NativeLibrary, capacity: u32) -> Result<usize> {
        let up = library.v41_fp8_matrix_info(capacity, 5120, 2304)?;
        let down = library.v41_fp8_matrix_info(capacity, 2304, 5120)?;
        usize::try_from(up.scratch_bytes + down.scratch_bytes + u64::from(capacity) * 2304 * 6 + 4)
            .context("shared FFN scratch budget overflow")
    }
    pub fn intermediates(&self) -> [Ds41rtDeviceBuffer; 3] {
        [self.gate.buffer, self.up.buffer, self.intermediate.buffer]
    }
    /// Enqueue on the owning expert wave stream; the caller must drain even on
    /// failure before releasing this exclusive borrow of scratch.
    pub(crate) unsafe fn enqueue(
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
        let scales = self.scales;
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
