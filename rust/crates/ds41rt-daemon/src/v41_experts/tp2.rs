//! One lane's preallocated peer-copy and TP2 reduction chain.
use crate::v41_memory::device::{Allocation, Device, Event, PeerTransfer, Stream};
use anyhow::{ensure, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, V41Tp2ExpertReducer};

pub(crate) struct PeerReduction<'a> {
    // Transfer drains before any referenced staging or output can be freed.
    transfer: PeerTransfer<'a>,
    local_ready: Event<'a>,
    staging: Allocation<'a>,
    output: Allocation<'a>,
    reducer: V41Tp2ExpertReducer<'a>,
    capacity: u32,
}
impl<'a> PeerReduction<'a> {
    pub fn device_bytes(capacity: u32) -> Result<usize> {
        ensure!((1..=4096).contains(&capacity), "invalid TP2 reduction capacity");
        Ok(capacity as usize * 5120 * (6 * 4 + 2))
    }
    pub fn new(remote: Device<'a>, local: Device<'a>, capacity: u32) -> Result<Self> {
        Self::device_bytes(capacity)?;
        ensure!(matches!((local.id, remote.id), (0, 1) | (1, 0)), "TP2 requires devices 0/1");
        Ok(Self {
            transfer: PeerTransfer::new(remote, local)?,
            local_ready: Event::new(local)?,
            staging: Allocation::new(local, capacity as usize * 5120 * 6 * 4)?,
            output: Allocation::new(local, capacity as usize * 5120 * 2)?,
            reducer: local.library.v41_tp2_expert_reducer()?,
            capacity,
        })
    }
    /// # Safety
    /// Input writes are ordered on their producer streams. No conflicting
    /// aliases may touch either input until this call completes or cancellation
    /// drains the chain. The returned view lives until this owner is reused.
    pub async unsafe fn reduce(&mut self, local: &Allocation<'a>, remote: &Allocation<'a>,
        local_producer: &Stream<'a>, remote_producer: &Stream<'a>, rows: u32,
        token_sums: bool) -> Result<Ds41rtDeviceBuffer> {
        ensure!(rows > 0 && rows <= self.capacity, "TP2 reduction exceeds capacity");
        let bytes = rows as usize * 5120 * if token_sums { 4 } else { 24 };
        ensure!(local.device.id == self.output.device.id && local.buffer.bytes >= bytes
            && std::ptr::eq(local.device.library, self.output.device.library), "local rank owner mismatch");
        self.local_ready.record(local_producer)?;
        let ready = &self.local_ready;
        let output = &mut self.output;
        let reducer = &self.reducer;
        unsafe { self.transfer.copy_then(remote, &mut self.staging, remote_producer, bytes,
            |peer, stream| {
                output.device.library.cuda_stream_wait_event(stream, ready.raw)?;
                let (rank0, rank1) = if local.device.id == 0 { (local.buffer, peer) } else { (peer, local.buffer) };
                reducer.reduce(rank0, rank1, output.buffer, rows, token_sums, stream)
            }).await?; }
        let mut result = self.output.buffer;
        result.bytes = rows as usize * 5120 * 2;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ds41rt_ffi::NativeLibrary;
    #[test]
    #[ignore = "requires DS41RT_NATIVE_LIB and two CUDA devices"]
    fn opposite_lane_peer_reductions_preserve_results_and_device() -> Result<()> {
        let lib = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        lib.cuda_set_device(0)?;
        let d0 = Device { library: &lib, id: 0 };
        let d1 = Device { library: &lib, id: 1 };
        let capacity = 16;
        let bytes = capacity * 5120 * 6 * 4;
        let a = Allocation::new(d0, bytes)?;
        let b = Allocation::new(d1, bytes)?;
        let p0 = Stream::new(d0)?;
        let p1 = Stream::new(d1)?;
        let mut lane0 = PeerReduction::new(d1, d0, capacity as u32)?;
        let mut lane1 = PeerReduction::new(d0, d1, capacity as u32)?;
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        for (rows, token_sums, va, vb) in [(1, false, 1f32, 2f32), (16, true, 3., -1.), (16, false, -2., 4.)] {
            let host_a: Vec<u8> = va.to_ne_bytes().into_iter().cycle().take(bytes).collect();
            let host_b: Vec<u8> = vb.to_ne_bytes().into_iter().cycle().take(bytes).collect();
            d0.run(|| lib.copy_h2d(a.buffer, &host_a))?;
            d1.run(|| lib.copy_h2d(b.buffer, &host_b))?;
            let (x, y) = runtime.block_on(async { tokio::join!(
                unsafe { lane0.reduce(&a, &b, &p0, &p1, rows, token_sums) },
                unsafe { lane1.reduce(&b, &a, &p1, &p0, rows, token_sums) }) });
            let expected = (((va + vb) * if token_sums { 1. } else { 6. }).to_bits() >> 16) as u16;
            for (device, output) in [(d0, x?), (d1, y?)] {
                let mut host = vec![0; output.bytes];
                device.run(|| lib.copy_d2h(&mut host, output))?;
                assert!(host.chunks_exact(2).all(|v| u16::from_ne_bytes([v[0],v[1]]) == expected));
            }
            assert_eq!(lib.cuda_get_device()?, 0);
        }
        Ok(())
    }
}
