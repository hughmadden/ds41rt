use super::*;
use crate::v41_memory::{SnapshotPool, SnapshotStorage};

pub(crate) struct DsparkPrefix<'a> {
    owner: u64,
    end: u64,
    ring: SnapshotStorage<'a>,
}
impl DsparkPrefix<'_> {
    pub fn end(&self) -> u64 {
        self.end
    }
}
fn slice(mut buffer: Ds41rtDeviceBuffer, offset: usize, bytes: usize) -> Ds41rtDeviceBuffer {
    debug_assert!(offset + bytes <= buffer.bytes);
    buffer.ptr = unsafe { buffer.ptr.cast::<u8>().add(offset).cast() };
    buffer.bytes = bytes;
    buffer
}
impl<'a> DsparkWindow<'a> {
    pub fn reserve_prefixes(&mut self, slots: usize) -> Result<usize> {
        ensure!(self.prefix_pool.is_none() && self.slots.iter().all(|s| s.request.is_none()),
            "draft snapshot arena must be installed before admission");
        if slots > 0 { self.prefix_pool = Some(SnapshotPool::new(self.stream.library, V41DsparkCache::SLOT_BYTES, slots)?); }
        Ok(self.prefix_pool.as_ref().map_or(0, SnapshotPool::device_bytes))
    }
    pub fn retain_prefix(&mut self, lease: WindowLease) -> Result<DsparkPrefix<'a>> {
        let slot = self.validate(lease)?;
        self.access.readable(slot)?;
        let end = self.slots[slot]
            .end
            .context("cannot retain an unseeded draft window")?;
        let bytes = end.min(128) as usize * V41DsparkCache::ROW_BYTES;
        ensure!(bytes > 0, "cannot retain an empty draft window");
        let ring = SnapshotStorage::new(self.stream.library, bytes, self.prefix_pool.as_ref())?;
        let copied = unsafe {
            self.stream.library.copy_d2d_async(
                ring.buffer,
                slice(self.ring.buffer, slot * V41DsparkCache::SLOT_BYTES, bytes),
                bytes,
                self.stream.raw,
            )
        };
        let drained = self.synchronize();
        copied.and(drained)?;
        Ok(DsparkPrefix {
            owner: self.owner,
            end,
            ring,
        })
    }
    pub fn restore_prefix(&mut self, lease: WindowLease, prefix: &DsparkPrefix<'a>) -> Result<()> {
        let slot = self.validate(lease)?;
        self.access.writable(slot)?;
        ensure!(
            prefix.owner == self.owner && self.slots[slot].end.is_none(),
            "foreign draft prefix or nonfresh window"
        );
        let copied = unsafe {
            self.stream.library.copy_d2d_async(
                slice(
                    self.ring.buffer,
                    slot * V41DsparkCache::SLOT_BYTES,
                    prefix.ring.buffer.bytes,
                ),
                prefix.ring.buffer,
                prefix.ring.buffer.bytes,
                self.stream.raw,
            )
        };
        let drained = self.synchronize();
        if let Err(error) = copied.and(drained) {
            self.slots[slot].request = None;
            return Err(error);
        }
        self.slots[slot].end = Some(prefix.end);
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires DS41RT_NATIVE_LIB and CUDA"]
    fn native_draft_prefix_survives_slot_reuse() -> Result<()> {
        let lib = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        let mut window = DsparkWindow::new(&lib, 2, 128, usize::MAX)?;
        window.reserve_prefixes(1)?;
        for end in [5u64, 128, 129, 1000000] {
            let old = window.begin_request(0, 1)?;
            assert!(window.retain_prefix(old).is_err());
            window.slots[0].end = Some(end);
            let bytes = end.min(128) as usize * V41DsparkCache::ROW_BYTES;
            let original: Vec<u8> = (0..bytes).map(|i| (i % 251) as u8).collect();
            lib.copy_h2d(slice(window.ring.buffer, 0, bytes), &original)?;
            let prefix = window.retain_prefix(old)?;
            window.release(old)?;
            let replacement = window.begin_request(0, 2)?;
            lib.copy_h2d(slice(window.ring.buffer, 0, bytes), &vec![0xff; bytes])?;
            let resumed = window.begin_request(1, 3)?;
            window.restore_prefix(resumed, &prefix)?;
            let mut restored = vec![0; bytes];
            lib.copy_d2h(
                &mut restored,
                slice(window.ring.buffer, V41DsparkCache::SLOT_BYTES, bytes),
            )?;
            assert_eq!(restored, original);
            assert_eq!(window.committed_end(resumed)?, Some(end));
            assert!(window.validate(old).is_err());
            assert!(window.restore_prefix(resumed, &prefix).is_err());
            window.release(replacement)?;
            window.release(resumed)?;
        }
        Ok(())
    }
}
