//! Queued writes reserve only their request slots; publication follows GPU completion.
use super::*;

pub(crate) struct WindowWrite {
    owner: u64,
    descriptors: [V41KvWrite; 16],
    leases: Vec<WindowLease>,
    seen: [bool; 16],
    ends: [Option<u64>; 16],
    source_rows: u32,
    _reservation: WriteReservation,
}
impl WindowWrite {
    pub fn descriptor_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.descriptors.as_ptr().cast::<u8>(), 384) }
    }
}
impl DsparkWindow<'_> {
    /// Reserve a validated append without changing the committed frontier. Dropping
    /// an unqueued ticket cancels it. After enqueue, retain it through stream drain.
    pub fn prepare_async_write(&self, chunks: &[WindowChunk], source_rows: u32) -> Result<WindowWrite> {
        let (descriptors, seen, ends) = self.prepare_write(chunks, source_rows)?;
        Ok(WindowWrite {
            owner: self.owner, descriptors, seen, ends, source_rows,
            leases: chunks.iter().map(|chunk| chunk.lease).collect(),
            _reservation: self.access.reserve_write(seen)?,
        })
    }
    pub(crate) fn validate_async_write(&self, write: &WindowWrite) -> Result<()> {
        ensure!(write.owner == self.owner, "foreign queued draft write");
        for &lease in &write.leases {
            let slot = self.validate(lease)?;
            ensure!(self.access.0.get()[slot] == WRITE_RESERVED, "draft write reservation lost");
        }
        Ok(())
    }
    /// # Safety
    /// Source is finite accepted main KV. Descriptors contain exactly this ticket's
    /// descriptor bytes, and both buffers/window/ticket survive stream completion.
    /// Ordered producers must finish before the kernel reads them. After enqueue,
    /// drain even on error before publishing, revoking or dropping the ticket.
    pub unsafe fn enqueue_write(&self, write: &WindowWrite, source: Ds41rtDeviceBuffer,
        descriptors: Ds41rtDeviceBuffer, stream: *mut c_void) -> Result<()> {
        self.validate_async_write(write)?;
        unsafe { self.kernel.launch(source, descriptors, self.ring.buffer,
            write.source_rows, self.slot_count as u32, stream) }
    }
    /// # Safety
    /// The complete write has succeeded on its stream; no queued consumer remains.
    pub unsafe fn publish_write(&mut self, write: WindowWrite) -> Result<()> {
        self.validate_async_write(&write)?;
        for slot in 0..self.slot_count {
            if write.seen[slot] { self.slots[slot].end = write.ends[slot]; }
        }
        Ok(())
    }
    /// # Safety
    /// All queued work touching this transaction has drained. A partial write must
    /// revoke its requests rather than expose the old frontier over changed bytes.
    pub unsafe fn revoke_write(&mut self, write: WindowWrite) -> Result<()> {
        self.validate_async_write(&write)?;
        for slot in 0..self.slot_count {
            if write.seen[slot] { self.slots[slot].request = None; self.slots[slot].end = None; }
        }
        Ok(())
    }
}
