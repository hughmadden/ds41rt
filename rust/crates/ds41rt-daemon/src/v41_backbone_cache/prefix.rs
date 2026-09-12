use super::*;
use crate::v41_compressor::{CompressorPrefix, COMPRESSOR_PREFIX_BYTES};
use crate::v41_memory::DeviceAllocation;
use crate::v41_window::{WindowPrefix, WINDOW_PREFIX_BYTES};
use ds41rt_ffi::Ds41rtDeviceBuffer;

pub(crate) struct BackbonePrefix<'a> {
    owner: u64,
    end: u64,
    tail: DeviceAllocation<'a>,
    windows: Vec<WindowPrefix>,
    sources: Vec<CompressorPrefix>,
}
impl BackbonePrefix<'_> {
    pub fn device_bytes() -> usize {
        40 * WINDOW_PREFIX_BYTES + 4 * COMPRESSOR_PREFIX_BYTES
    }
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
impl<'a> BackboneCache<'a> {
    /// Snapshot only a fully committed request, after all producer/consumer
    /// streams have drained. Global KV/index pages remain shared; only bounded
    /// SWA and pending compressor state is copied into the retained GPU arena.
    pub fn retain_prefix(
        &mut self,
        lease: CacheLease,
        budget: usize,
    ) -> Result<BackbonePrefix<'a>> {
        let end = self.committed_end(lease)?;
        let request = self.request(lease)?;
        ensure!(
            end > 0 && request.phase == CachePhase::Full && request.publication.is_empty(),
            "prefix retention requires a complete request frontier"
        );
        ensure!(
            BackbonePrefix::device_bytes() <= budget,
            "retained backbone tail exceeds budget"
        );
        let tail =
            DeviceAllocation::new(self.prefix_stream.library, BackbonePrefix::device_bytes())?;
        let result = (|| -> Result<_> {
            let windows = self
                .windows
                .iter()
                .zip(request.windows)
                .enumerate()
                .map(|(i, (state, lease))| unsafe {
                    state.retain_prefix(
                        lease,
                        slice(tail.buffer, i * WINDOW_PREFIX_BYTES, WINDOW_PREFIX_BYTES),
                        self.prefix_stream.raw,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            let sources = self
                .sources
                .iter()
                .zip(request.sources)
                .enumerate()
                .map(|(i, (state, lease))| unsafe {
                    state.retain_prefix(
                        lease,
                        slice(
                            tail.buffer,
                            40 * WINDOW_PREFIX_BYTES + i * COMPRESSOR_PREFIX_BYTES,
                            COMPRESSOR_PREFIX_BYTES,
                        ),
                        self.prefix_stream.raw,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            Ok((windows, sources))
        })();
        // Always drain before tail is dropped, including an enqueue failure.
        let drained = unsafe {
            self.prefix_stream
                .library
                .cuda_stream_synchronize(self.prefix_stream.raw)
        };
        let (windows, sources) = result.and_then(|saved| {
            drained?;
            Ok(saved)
        })?;
        Ok(BackbonePrefix {
            owner: self.owner,
            end,
            tail,
            windows,
            sources,
        })
    }

    /// Restore into a fresh admission. Any partially applied failure revokes
    /// the whole request after draining; no mixed-layer frontier is observable.
    pub fn restore_prefix(&mut self, lease: CacheLease, prefix: &BackbonePrefix<'a>) -> Result<()> {
        let request = self.request(lease)?;
        ensure!(
            prefix.owner == self.owner
                && request.end == 0
                && request.version == 0
                && request.phase == CachePhase::Full
                && request.publication.is_empty(),
            "foreign backbone prefix or nonfresh admission"
        );
        let windows = request.windows;
        let sources = request.sources;
        let result = (|| -> Result<()> {
            for (i, ((state, lease), saved)) in self
                .windows
                .iter_mut()
                .zip(windows)
                .zip(&prefix.windows)
                .enumerate()
            {
                unsafe {
                    state.restore_prefix(
                        lease,
                        saved,
                        slice(
                            prefix.tail.buffer,
                            i * WINDOW_PREFIX_BYTES,
                            WINDOW_PREFIX_BYTES,
                        ),
                        self.prefix_stream.raw,
                    )?;
                }
            }
            for (i, ((state, lease), saved)) in self
                .sources
                .iter_mut()
                .zip(sources)
                .zip(&prefix.sources)
                .enumerate()
            {
                unsafe {
                    state.restore_prefix(
                        lease,
                        saved,
                        slice(
                            prefix.tail.buffer,
                            40 * WINDOW_PREFIX_BYTES + i * COMPRESSOR_PREFIX_BYTES,
                            COMPRESSOR_PREFIX_BYTES,
                        ),
                        self.prefix_stream.raw,
                    )?;
                }
            }
            Ok(())
        })();
        let drained = unsafe {
            self.prefix_stream
                .library
                .cuda_stream_synchronize(self.prefix_stream.raw)
        };
        if let Err(error) = result.and(drained) {
            if let Err(cleanup) = self.release(&[lease]) {
                tracing::error!(%cleanup, "releasing failed prefix restore");
            }
            return Err(error);
        }
        let request = self.requests[lease.slot]
            .as_mut()
            .expect("validated prefix admission");
        request.end = prefix.end;
        request.version = 1;
        self.committed_end(lease)?;
        Ok(())
    }
}
