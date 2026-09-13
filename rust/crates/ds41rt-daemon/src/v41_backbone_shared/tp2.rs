//! Device-owned TP2 shared-expert weights and one rank's lane workspace.
use crate::v41_memory::{HostAllocation, device::{Allocation, Device, Event, PeerTransfer, Stream}};
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, V41Bf16Add, V41Fp8Plan, V41SharedSwiGlu};
use ds41rt_loader::OfficialV41Catalog;
use std::rc::Rc;

struct Matrix<'a> { weight: Allocation<'a>, scales: Allocation<'a> }

/// Complete shared TP2 operation for one lane, with either GPU as destination.
pub(crate) struct Wave<'a> {
    transfers: [PeerTransfer<'a>; 2],
    ready: [Event<'a>; 2],
    staging: [Allocation<'a>; 2],
    output: [Allocation<'a>; 2],
    ranks: [RankWave<'a>; 2],
    add: V41Bf16Add<'a>,
    capacity: u32,
}
impl<'a> Wave<'a> {
    pub fn new(weights: [Rc<Vec<Weights<'a>>>; 2], capacity: u32) -> Result<Self> {
        ensure!(!weights[0].is_empty() && weights[0].len() == weights[1].len(), "shared TP2 rank layers differ");
        let d0 = weights[0][0].device;
        let d1 = weights[1][0].device;
        ensure!(d0.id == 0 && d1.id == 1 && std::ptr::eq(d0.library, d1.library), "shared TP2 rank devices differ");
        let ranks = [RankWave::new(weights[0].clone(), capacity)?, RankWave::new(weights[1].clone(), capacity)?];
        let bytes = capacity as usize * 5120 * 2;
        Ok(Self {
            transfers: [PeerTransfer::new(d1,d0)?, PeerTransfer::new(d0,d1)?],
            ready: [Event::new(d0)?, Event::new(d1)?],
            staging: [Allocation::new(d0,bytes)?, Allocation::new(d1,bytes)?],
            output: [Allocation::new(d0,bytes)?, Allocation::new(d1,bytes)?],
            ranks, add: d0.library.v41_bf16_add()?, capacity,
        })
    }
    /// # Safety
    /// Inputs are produced on the supplied streams and remain unchanged through
    /// completion. Each lane owns a separate Wave. Returned output remains valid
    /// only until the next execute or destruction of this owner.
    pub async unsafe fn execute(&mut self, layer: usize, rows: u32, destination: usize,
        inputs: [&Allocation<'a>; 2], producers: [&Stream<'a>; 2]) -> Result<Ds41rtDeviceBuffer> {
        ensure!(destination < 2 && rows > 0 && rows <= self.capacity, "invalid shared TP2 destination/rows");
        struct Drain<'s, 'a> { ranks: &'s mut [RankWave<'a>; 2], complete: bool }
        impl Drop for Drain<'_, '_> {
            fn drop(&mut self) {
                if !self.complete {
                    for rank in self.ranks.iter() {
                        if let Err(error) = rank.stream.drain() { tracing::error!(%error, "draining shared TP2 rank"); }
                    }
                }
            }
        }
        let mut drain = Drain { ranks: &mut self.ranks, complete: false };
        for rank in 0..2 {
            unsafe { drain.ranks[rank].enqueue(layer, rows, inputs[rank].buffer, producers[rank])?; }
        }
        let local = &drain.ranks[destination];
        let remote = &drain.ranks[1-destination];
        self.ready[destination].record(&local.stream)?;
        let ready = &self.ready[destination];
        let output = &mut self.output[destination];
        let add = &self.add;
        let count = rows as usize * 5120;
        unsafe { self.transfers[destination].copy_then(&remote.output, &mut self.staging[destination],
            &remote.stream, count*2, |peer, stream| {
                output.device.library.cuda_stream_wait_event(stream, ready.raw)?;
                add.launch(local.output.buffer, peer, output.buffer, count, stream)
            }).await?; }
        drain.complete = true;
        let mut result = self.output[destination].buffer;
        result.bytes = count*2;
        Ok(result)
    }
}

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
    fn real_shared_tp2_complete_independent_lanes() -> Result<()> {
        let lib = unsafe { ds41rt_ffi::NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        let catalog = ds41rt_loader::read_official_v41_catalog(ds41rt_loader::OFFICIAL_V41_MODEL_ID,
            std::path::Path::new(&std::env::var("DS41RT_SNAPSHOT")?))?;
        lib.cuda_set_device(0)?;
        let devices = [Device { library: &lib, id: 0 }, Device { library: &lib, id: 1 }];
        let weights = [
            Rc::new(vec![Weights::load(devices[0], &catalog, 0, Weights::load_peak_device_bytes())?]),
            Rc::new(vec![Weights::load(devices[1], &catalog, 0, Weights::load_peak_device_bytes())?]),
        ];
        let mut lanes = [Wave::new(weights.clone(), 16)?, Wave::new(weights, 16)?];
        let inputs = [Allocation::new(devices[0], 16*5120*2)?, Allocation::new(devices[1], 16*5120*2)?];
        let producers = [Stream::new(devices[0])?, Stream::new(devices[1])?];
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        for (rows, nonzero) in [(1, false), (16, true), (1, true), (16, false)] {
            let host: Vec<u8> = (0..16*5120).flat_map(|i| {
                let value = if nonzero { (i % 31) as f32 / 32.0 - 0.5 } else { 0.0 };
                ((value.to_bits() >> 16) as u16).to_ne_bytes()
            }).collect();
            for input in &inputs { input.device.run(|| lib.copy_h2d(input.buffer, &host))?; }
            let [left, right] = &mut lanes;
            let (a,b) = runtime.block_on(async { tokio::join!(
                unsafe { left.execute(0, rows, 0, [&inputs[0], &inputs[1]], [&producers[0], &producers[1]]) },
                unsafe { right.execute(0, rows, 1, [&inputs[0], &inputs[1]], [&producers[0], &producers[1]]) }
            ) });
            let outputs = [a?,b?];
            let mut actual = [vec![0; rows as usize*5120*2], vec![0; rows as usize*5120*2]];
            for rank in 0..2 { devices[rank].run(|| lib.copy_d2h(&mut actual[rank], outputs[rank]))?; }
            assert_eq!(actual[0], actual[1], "destination changed shared output");
            let mut partials = [vec![0; actual[0].len()], vec![0; actual[0].len()]];
            for rank in 0..2 {
                let mut buffer = lanes[0].ranks[rank].output.buffer;
                buffer.bytes = partials[rank].len();
                devices[rank].run(|| lib.copy_d2h(&mut partials[rank], buffer))?;
            }
            for ((a,b),actual) in partials[0].chunks_exact(2).zip(partials[1].chunks_exact(2)).zip(actual[0].chunks_exact(2)) {
                let decode = |v: &[u8]| f32::from_bits((u16::from_ne_bytes([v[0],v[1]]) as u32) << 16);
                let sum = decode(a) + decode(b);
                assert!(sum.is_finite());
                let bits = sum.to_bits();
                let rounded = ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16;
                assert_eq!(u16::from_ne_bytes([actual[0],actual[1]]), rounded);
            }
            assert_eq!(actual[0].chunks_exact(2).any(|v| u16::from_ne_bytes([v[0],v[1]]) & 0x7fff != 0), nonzero);
            assert_eq!(lib.cuda_get_device()?, 0);
        }
        Ok(())
    }
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
