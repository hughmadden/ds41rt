//! Device-owned TP2 shared-expert weights and one rank's lane workspace.
use crate::v41_memory::{HostAllocation, device::{Allocation, Device, Event, Stream}};
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, V41Fp8Plan, V41SharedSwiGlu};
use ds41rt_loader::OfficialV41Catalog;
use std::rc::Rc;

struct Matrix<'a> { weight: Allocation<'a>, scales: Allocation<'a> }
pub(crate) struct Weights<'a> {
    device: Device<'a>,
    layer: usize,
    matrices: [Matrix<'a>; 3],
}
impl<'a> Weights<'a> {
    pub fn device_bytes() -> usize { 3 * (5120 * 1152 + 5120 * 1152 / 32) }
    pub fn load_peak_device_bytes() -> usize { Self::device_bytes() + 5120 * 1152 / 1024 }
    pub fn load(device: Device<'a>, catalog: &OfficialV41Catalog, layer: usize, budget: usize) -> Result<Self> {
        ensure!(layer < 40 && matches!(device.id, 0 | 1) && budget >= Self::load_peak_device_bytes(),
            "invalid TP2 shared weight layer, rank, or budget");
        let stream = Stream::new(device)?;
        let mut staging = HostAllocation::new(device.library, 5120 * 1152)?;
        let mut scratch = vec![0u8; 2304 * 64];
        let mut matrices = Vec::with_capacity(3);
        for name in ["w1", "w3", "w2"] {
            let (axis, k, n) = if name == "w2" { (1, 1152, 5120) } else { (0, 5120, 1152) };
            let kernel = device.run(|| device.library.v41_fp8_matrix_kernel(1, k, n))?;
            let prefix = format!("layers.{layer}.ffn.shared_experts.{name}");
            let weight = Allocation::new(device, (k*n) as usize)?;
            let scales = Allocation::new(device, (k*n/32) as usize)?;
            let source_scales = Allocation::new(device, (k*n/1024) as usize)?;
            for (suffix, target) in [("weight", &weight), ("scale", &source_scales)] {
                let bytes = catalog.read_coordinator_tp2_into(&format!("{prefix}.{suffix}"), axis,
                    device.id as usize, staging.bytes_mut(), &mut scratch)?;
                ensure!(bytes == target.buffer.bytes, "shared TP2 tensor has unexpected size");
                device.run(|| device.library.copy_h2d(target.buffer, &staging.bytes_mut()[..bytes]))?;
            }
            let packed = device.run(|| unsafe { kernel.pack_scales(source_scales.buffer, scales.buffer, stream.raw) });
            let drained = stream.drain();
            packed.and(drained)?;
            matrices.push(Matrix { weight, scales });
        }
        Ok(Self { device, layer, matrices: matrices.try_into().ok().unwrap() })
    }
}

pub(crate) struct RankWave<'a> {
    pub stream: Stream<'a>,
    ready: Event<'a>,
    weights: Rc<Vec<Weights<'a>>>,
    up_kernel: V41Fp8Plan<'a>,
    down_kernel: V41Fp8Plan<'a>,
    activation: V41SharedSwiGlu<'a>,
    up_scratch: Allocation<'a>,
    down_scratch: Allocation<'a>,
    alpha: Allocation<'a>,
    gate: Allocation<'a>,
    up: Allocation<'a>,
    intermediate: Allocation<'a>,
    pub output: Allocation<'a>,
    capacity: u32,
}
impl<'a> RankWave<'a> {
    pub fn new(weights: Rc<Vec<Weights<'a>>>, capacity: u32) -> Result<Self> {
        ensure!(!weights.is_empty() && weights.len() <= 40, "shared TP2 needs resident layers");
        let device = weights[0].device;
        ensure!(weights.iter().enumerate().all(|(layer,w)| w.layer == layer && w.device.id == device.id
            && std::ptr::eq(w.device.library, device.library)), "shared TP2 weight owners differ");
        let up_kernel = device.run(|| device.library.v41_fp8_matrix_plan(capacity, 5120, 1152))?;
        let down_kernel = device.run(|| device.library.v41_fp8_matrix_plan(capacity, 1152, 5120))?;
        let stream = Stream::new(device)?;
        let up_scratch = Allocation::new(device, up_kernel.info().scratch_bytes as usize)?;
        let down_scratch = Allocation::new(device, down_kernel.info().scratch_bytes as usize)?;
        let alpha = Allocation::new(device, 16)?;
        let initialized = device.run(|| unsafe {
            up_kernel.initialize_scratch(up_scratch.buffer, alpha.buffer, stream.raw)?;
            down_kernel.initialize_scratch(down_scratch.buffer, alpha.buffer, stream.raw)
        });
        let drained = stream.drain();
        initialized.and(drained)?;
        Ok(Self { stream, ready: Event::new(device)?, weights, up_kernel, down_kernel,
            activation: device.library.v41_shared_tp2_swiglu()?, up_scratch, down_scratch, alpha,
            gate: Allocation::new(device, capacity as usize * 1152 * 2)?,
            up: Allocation::new(device, capacity as usize * 1152 * 2)?,
            intermediate: Allocation::new(device, capacity as usize * 1152 * 2)?,
            output: Allocation::new(device, capacity as usize * 5120 * 2)?, capacity })
    }
    /// # Safety
    /// Previous use has completed. Input writes are ordered on producer, and
    /// input storage stays live/unchanged until this stream finishes.
    pub unsafe fn enqueue(&mut self, layer: usize, rows: u32, input: Ds41rtDeviceBuffer,
        producer: &Stream<'a>) -> Result<()> {
        let device = self.output.device;
        ensure!(rows > 0 && rows <= self.capacity && layer < self.weights.len()
            && input.device_id == device.id && input.bytes >= rows as usize * 5120 * 2,
            "invalid shared TP2 input or layer");
        self.ready.record(producer)?;
        let matrices = &self.weights[layer].matrices;
        let queued = device.run(|| unsafe {
            device.library.cuda_stream_wait_event(self.stream.raw, self.ready.raw)?;
            for (matrix, output) in [(&matrices[0], &self.gate), (&matrices[1], &self.up)] {
                self.up_kernel.launch(input, matrix.weight.buffer, matrix.scales.buffer,
                    self.up_scratch.buffer, self.alpha.buffer, output.buffer, rows, self.stream.raw)
                    .with_context(|| format!("shared TP2 up/gate on GPU {}", device.id))?;
            }
            self.activation.launch(self.gate.buffer, self.up.buffer, self.intermediate.buffer, rows, self.stream.raw)?;
            self.down_kernel.launch(self.intermediate.buffer, matrices[2].weight.buffer, matrices[2].scales.buffer,
                self.down_scratch.buffer, self.alpha.buffer, self.output.buffer, rows, self.stream.raw)
                .with_context(|| format!("shared TP2 down on GPU {}", device.id))
        });
        if let Err(error) = queued { self.stream.drain()?; return Err(error); }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires shared TP2 DS41RT_NATIVE_LIB, DS41RT_SNAPSHOT, two GPUs"]
    fn real_shared_tp2_rank_loading_and_execution() -> Result<()> {
        let lib = unsafe { ds41rt_ffi::NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        let catalog = ds41rt_loader::read_official_v41_catalog(ds41rt_loader::OFFICIAL_V41_MODEL_ID,
            std::path::Path::new(&std::env::var("DS41RT_SNAPSHOT")?))?;
        lib.cuda_set_device(0)?;
        let mut ranks = Vec::new();
        for id in 0..2 {
            let device = Device { library: &lib, id };
            let weights = Rc::new(vec![Weights::load(device, &catalog, 0, Weights::load_peak_device_bytes())?]);
            ranks.push((RankWave::new(weights.clone(), 16)?, RankWave::new(weights, 16)?,
                Allocation::new(device, 16*5120*2)?, Stream::new(device)?));
        }
        for nonzero in [false, true] {
            for (first, second, input, producer) in &mut ranks {
                let value = if nonzero { 0x3f00u16 } else { 0 };
                let host: Vec<u8> = value.to_ne_bytes().into_iter().cycle().take(input.buffer.bytes).collect();
                input.device.run(|| lib.copy_h2d(input.buffer, &host))?;
                unsafe {
                    first.enqueue(0, 16, input.buffer, producer)?;
                    second.enqueue(0, 16, input.buffer, producer)?;
                }
            }
            for (first, second, input, _) in &ranks {
                first.stream.drain()?;
                second.stream.drain()?;
                let mut a = vec![0; first.output.buffer.bytes];
                let mut b = vec![0; second.output.buffer.bytes];
                input.device.run(|| {
                    lib.copy_d2h(&mut a, first.output.buffer)?;
                    lib.copy_d2h(&mut b, second.output.buffer)
                })?;
                assert_eq!(a,b);
                assert!(a.chunks_exact(2).all(|v| u16::from_ne_bytes([v[0],v[1]]) & 0x7f80 != 0x7f80));
                assert_eq!(a.chunks_exact(2).any(|v| u16::from_ne_bytes([v[0],v[1]]) & 0x7fff != 0), nonzero);
            }
            assert_eq!(lib.cuda_get_device()?, 0);
        }
        Ok(())
    }
}
