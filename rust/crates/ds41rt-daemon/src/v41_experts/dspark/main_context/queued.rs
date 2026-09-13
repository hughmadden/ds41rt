//! Direct accepted-KV publication on the producer stream, without shared staging.
use super::*;
impl DsparkMainContext<'_, '_> {
    /// # Safety
    /// Target input is complete and belongs to these validated committed rows.
    /// Window owners and input survive polling/draining. On ANY error the caller
    /// must abort this transaction before recycling target or cache storage.
    pub unsafe fn enqueue_commit(&mut self, input: Ds41rtDeviceBuffer, positions: &[u64],
        windows: [&DsparkWindow<'_>; 3], writes: [WindowWrite; 3]) -> Result<()> {
        let rows = u32::try_from(positions.len())?;
        self.prepare(rows)?;
        ensure!(input.bytes == positions.len()*30720 && input.device_id == self.input().device_id
            && input.ptr != self.input().ptr, "queued main-context input differs");
        for stage in 0..3 { windows[stage].validate_async_write(&writes[stage])?; }
        let host = self.commit_staging.bytes_mut();
        for (i, position) in positions.iter().enumerate() {
            host[i*8..i*8+8].copy_from_slice(&position.to_ne_bytes());
        }
        for stage in 0..3 {
            host[self.capacity as usize*8+stage*384..self.capacity as usize*8+(stage+1)*384]
                .copy_from_slice(writes[stage].descriptor_bytes());
        }
        self.pending_writes = Some(writes); // Own reservations before any launch.
        unsafe {
            let lib = self.stream.library;
            lib.copy_d2d_async(self.input(), input, input.bytes, self.stream.raw)?;
            lib.copy_h2d_async(self.positions.buffer, &self.commit_staging.bytes_mut()[..positions.len()*8], self.stream.raw)?;
            self.enqueue(rows)?;
            for stage in 0..3 {
                let start = self.capacity as usize*8 + stage*384;
                lib.copy_h2d_async(self.commit_descriptors[stage].buffer,
                    &self.commit_staging.bytes_mut()[start..start+384], self.stream.raw)?;
                windows[stage].enqueue_write(&self.pending_writes.as_ref().unwrap()[stage],
                    self.rotated[stage].buffer, self.commit_descriptors[stage].buffer, self.stream.raw)?;
            }
        }
        Ok(())
    }
    pub fn poll_commit(&self) -> Result<bool> {
        ensure!(self.pending_writes.is_some(), "main-context commit not pending");
        unsafe { self.stream.library.cuda_stream_query(self.stream.raw) }
    }
    pub fn publish_commit(&mut self, windows: &mut [DsparkWindow<'_>; 3]) -> Result<()> {
        ensure!(self.poll_commit()?, "main-context commit has not completed");
        let writes = self.pending_writes.as_ref().unwrap();
        // Validate every owner before publishing any stage.
        for stage in 0..3 { windows[stage].validate_async_write(&writes[stage])?; }
        for (window, write) in windows.iter_mut().zip(self.pending_writes.take().unwrap()) {
            unsafe { window.publish_write(write)?; }
        }
        Ok(())
    }
    pub fn abort_commit(&mut self, windows: &mut [DsparkWindow<'_>; 3]) -> Result<()> {
        let drained = self.synchronize();
        let mut result = Ok(());
        if let Some(writes) = self.pending_writes.take() {
            for (window, write) in windows.iter_mut().zip(writes) {
                if let Err(error) = unsafe { window.revoke_write(write) } { result = Err(error); }
            }
        }
        drained.and(result)
    }
}
