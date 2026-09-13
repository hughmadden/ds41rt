//! Reusable pinned output storage with cooperative, owner-local completion.
use super::*;
use anyhow::ensure;

pub(crate) struct RowDownload<'a> {
    stream: LoadStream<'a>,
    staging: HostAllocation<'a>,
}
impl<'a> RowDownload<'a> {
    pub fn new(library: &'a NativeLibrary, bytes: usize) -> Result<Self> {
        Ok(Self { stream: LoadStream { library, raw: library.cuda_stream_create()? },
            staging: HostAllocation::new(library, bytes)? })
    }
    /// # Safety
    /// Source producers are complete. Keep source storage alive and unchanged
    /// until return or cancellation; the wait guard drains before relinquishing it.
    pub async unsafe fn rows(&mut self, source: Ds41rtDeviceBuffer, width: usize,
        selected: &[usize]) -> Result<Vec<u8>> {
        ensure!(width > 0 && source.bytes % width == 0 && !selected.is_empty()
            && selected.len() <= self.staging.buffer.bytes / width
            && selected.iter().all(|&row| row < source.bytes / width), "invalid download row extent");
        let bytes = selected.len() * width;
        let queued = (|| -> Result<()> {
            let mut first = 0;
            while first < selected.len() {
                let mut count = 1;
                while first+count < selected.len() && selected[first+count] == selected[first]+count {
                    count += 1;
                }
                let mut part = source;
                part.ptr = unsafe { part.ptr.cast::<u8>().add(selected[first]*width).cast() };
                part.bytes = count * width;
                unsafe { self.stream.library.copy_d2h_async(
                    &mut self.staging.bytes_mut()[first*width..(first+count)*width], part, self.stream.raw)?; }
                first += count;
            }
            Ok(())
        })();
        let drained = self.stream.wait().await;
        queued.and(drained)?;
        Ok(self.staging.bytes_mut()[..bytes].to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires DS41RT_NATIVE_LIB and CUDA"]
    fn native_row_downloads_preserve_selection_and_peer_storage() -> Result<()> {
        let lib = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        let width = 129280 * 4;
        let source = DeviceAllocation::new(&lib, 4*width)?;
        let original: Vec<u8> = (0..4*width).map(|i| ((i*13+i/width)%251) as u8).collect();
        lib.copy_h2d(source.buffer, &original)?;
        let mut first = RowDownload::new(&lib, 4*width)?;
        let mut second = RowDownload::new(&lib, 4*width)?;
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        runtime.block_on(async {
            let (a, b) = tokio::join!(unsafe { first.rows(source.buffer, width, &[0,1,2,3]) },
                unsafe { second.rows(source.buffer, width, &[3,0,2]) });
            assert_eq!(a?, original);
            let expected: Vec<u8> = [3,0,2].into_iter().flat_map(|r| original[r*width..(r+1)*width].iter().copied()).collect();
            assert_eq!(b?, expected);
            assert!(unsafe { first.rows(source.buffer, width, &[4]).await }.is_err());
            assert_eq!(unsafe { first.rows(source.buffer, width, &[1]).await }?, original[width..2*width]);
            Ok::<_, anyhow::Error>(())
        })
    }
}
