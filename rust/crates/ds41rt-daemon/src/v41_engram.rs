//! Owned native gathered-row upload and BF16 engram embedding production.
pub(crate) mod layer;
use crate::v41_memory::{DeviceAllocation, HostAllocation, LoadStream};
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary};
use ds41rt_loader::{
    EngramGatherPoll, EngramGatherTicket, EngramGatherView, EngramPipeline, EngramWave,
};

pub(crate) struct EngramDeviceRows<'a> {
    stream: LoadStream<'a>,
    staging: HostAllocation<'a>,
    weights: DeviceAllocation<'a>,
    scales: DeviceAllocation<'a>,
    embeddings: DeviceAllocation<'a>,
    text_mask: DeviceAllocation<'a>,
    library: &'a NativeLibrary,
    capacity: usize,
    ready: Option<(usize, usize)>,
}
/// Borrowed buffers: never free or retain across owner reuse/drop.
pub(crate) struct EngramDeviceView {
    pub embeddings: Ds41rtDeviceBuffer,
    pub text_mask: Ds41rtDeviceBuffer,
    pub rows: usize,
    pub layer_index: usize,
}
pub(crate) enum EngramUploadPoll {
    Pending,
    Cancelled,
    Ready(EngramDeviceView),
}
impl<'a> EngramDeviceRows<'a> {
    /// Validate current request generations before consuming an early gather.
    pub fn poll_wave(
        &mut self,
        pipeline: &EngramPipeline,
        wave: &mut EngramWave,
        histories: &[&ds41rt_core::EngramHistory],
        layer: usize,
    ) -> Result<EngramUploadPoll> {
        self.ready = None;
        Ok(match pipeline.poll(wave, histories, layer)? {
            EngramGatherPoll::Pending => {
                tracing::debug!(target: "ds41rt::timing", layer, upload_owner=self as *const Self as usize, "engram IO pending");
                EngramUploadPoll::Pending
            }
            EngramGatherPoll::Cancelled => EngramUploadPoll::Cancelled,
            EngramGatherPoll::Ready(lease) => {
                let gathered = lease.view()?;
                if let Some(timing) = lease.timing() {
                    tracing::debug!(target: "ds41rt::timing", layer, rows=gathered.rows,
                        upload_owner=self as *const Self as usize,
                        queue_us=timing.queued.as_micros() as u64,
                        gather_us=timing.gather.as_micros() as u64,
                        ready_age_us=timing.completed.elapsed().as_micros() as u64,
                        minor_faults=timing.minor_faults, major_faults=timing.major_faults,
                        input_blocks=timing.input_blocks, "engram IO delivery");
                }
                EngramUploadPoll::Ready(self.upload(&gathered)?)
            }
        })
    }
    /// Poll on the CUDA-owning thread and recycle ready staging after upload.
    /// The scheduler must cancel tickets whose request history has been invalidated.
    pub fn poll_upload(&mut self, ticket: &mut EngramGatherTicket) -> Result<EngramUploadPoll> {
        self.ready = None;
        Ok(match ticket.poll()? {
            EngramGatherPoll::Pending => EngramUploadPoll::Pending,
            EngramGatherPoll::Cancelled => EngramUploadPoll::Cancelled,
            EngramGatherPoll::Ready(lease) => {
                let output = self.upload(&lease.view()?)?;
                // upload synchronizes before the lease returns its storage to the pool.
                EngramUploadPoll::Ready(output)
            }
        })
    }
    pub fn device_bytes(capacity: usize) -> Result<usize> {
        ensure!(
            capacity > 0 && capacity <= 4096,
            "invalid engram device capacity"
        );
        capacity
            .checked_mul(24 * (256 + 8 + 512) + 1)
            .context("engram device budget overflow")
    }
    pub fn new(
        library: &'a NativeLibrary,
        capacity: usize,
        available_bytes: usize,
    ) -> Result<Self> {
        ensure!(
            Self::device_bytes(capacity)? <= available_bytes,
            "engram rows exceed device budget"
        );
        Ok(Self {
            stream: LoadStream {
                library,
                raw: library.cuda_stream_create()?,
            },
            staging: HostAllocation::new(library, capacity * (24 * (256 + 8) + 1))?,
            weights: DeviceAllocation::new(library, capacity * 24 * 256)?,
            scales: DeviceAllocation::new(library, capacity * 24 * 8)?,
            embeddings: DeviceAllocation::new(library, capacity * 24 * 512)?,
            text_mask: DeviceAllocation::new(library, capacity)?,
            library,
            capacity,
            ready: None,
        })
    }
    /// Consume a completed I/O-worker gather; this method never reads mapped tables.
    /// Output is contiguous BF16 [token,24,256] plus a byte-per-token text mask.
    pub fn upload(&mut self, gathered: &EngramGatherView<'_>) -> Result<EngramDeviceView> {
        self.ready = None;
        ensure!(
            gathered.rows > 0 && gathered.rows <= self.capacity && gathered.layer_index < 2,
            "invalid gathered engram dimensions"
        );
        let hash_rows = gathered.rows * 24;
        ensure!(
            gathered.weights.len() == hash_rows * 256
                && gathered.scales.len() == hash_rows * 8
                && gathered.text_mask.len() == gathered.rows
                && gathered.text_mask.iter().all(|value| *value <= 1),
            "invalid gathered engram storage"
        );
        self.synchronize()?;
        self.library
            .copy_h2d(self.weights.buffer, gathered.weights)?;
        self.library.copy_h2d(self.scales.buffer, gathered.scales)?;
        self.library
            .copy_h2d(self.text_mask.buffer, gathered.text_mask)?;
        unsafe {
            self.library.cuda_engram_dequant_bf16_async(
                self.weights.buffer,
                self.scales.buffer,
                self.embeddings.buffer,
                i32::try_from(hash_rows)?,
                self.stream.raw,
            )?;
        }
        self.synchronize()?;
        self.ready = Some((gathered.rows, gathered.layer_index));
        self.view()
    }
    /// Pinned staging and this upload owner remain exclusive through completion;
    /// cancellation drains before either can be reused. No request owner is borrowed.
    pub async fn upload_cooperative(&mut self, gathered: &EngramGatherView<'_>) -> Result<EngramDeviceView> {
        self.ready = None;
        ensure!(
            gathered.rows > 0 && gathered.rows <= self.capacity && gathered.layer_index < 2,
            "invalid gathered engram dimensions"
        );
        let hash_rows = gathered.rows * 24;
        ensure!(
            gathered.weights.len() == hash_rows * 256
                && gathered.scales.len() == hash_rows * 8
                && gathered.text_mask.len() == gathered.rows
                && gathered.text_mask.iter().all(|value| *value <= 1),
            "invalid gathered engram storage"
        );
        let mut offset = 0;
        for bytes in [gathered.weights, gathered.scales, gathered.text_mask] {
            self.staging.bytes_mut()[offset..offset + bytes.len()].copy_from_slice(bytes);
            offset += bytes.len();
        }
        let launched = (|| -> Result<()> {
            let mut offset = 0;
            for (destination, bytes) in [(self.weights.buffer, gathered.weights.len()),
                (self.scales.buffer, gathered.scales.len()), (self.text_mask.buffer, gathered.text_mask.len())] {
                let mut host = self.staging.buffer;
                host.ptr = unsafe { host.ptr.cast::<u8>().add(offset).cast() }; host.bytes = bytes;
                unsafe { self.library.copy_host_buffer_h2d_async(destination, host, bytes, self.stream.raw)?; }
                offset += bytes;
            }
            unsafe { self.library.cuda_engram_dequant_bf16_async(self.weights.buffer, self.scales.buffer,
                self.embeddings.buffer, i32::try_from(hash_rows)?, self.stream.raw) }
        })();
        if let Err(error) = launched { self.synchronize()?; return Err(error); }
        self.stream.wait().await?;
        self.ready = Some((gathered.rows, gathered.layer_index));
        self.view()
    }
    #[cfg(test)]
    pub async fn check_cooperative(&mut self, gathered: &EngramGatherView<'_>) -> Result<()> {
        use std::{future::Future, task::Poll};
        let read = |lib: &NativeLibrary, view: EngramDeviceView| -> Result<Vec<Vec<u8>>> {
            [view.embeddings, view.text_mask].into_iter().map(|buffer| {
                let mut bytes = vec![0; buffer.bytes]; lib.copy_d2h(&mut bytes, buffer)?; Ok(bytes)
            }).collect()
        };
        let lib = self.library;
        let expected = read(lib, self.upload(gathered)?)?;
        let cancelled = {
            let mut work = std::pin::pin!(self.upload_cooperative(gathered));
            std::future::poll_fn(|cx| Poll::Ready(work.as_mut().poll(cx).is_pending())).await
        };
        if cancelled { assert!(self.view().is_err()); }
        for _ in 0..2 { assert_eq!(read(lib, self.upload_cooperative(gathered).await?)?, expected); }
        eprintln!("PASS queued Engram upload {}: exact dequant/mask parity, pending cancellation={cancelled}, reuse", gathered.layer_index);
        Ok(())
    }
    pub fn view(&self) -> Result<EngramDeviceView> {
        let (rows, layer_index) = self.ready.context("engram device rows are not complete")?;
        let mut embeddings = self.embeddings.buffer;
        embeddings.bytes = rows * 24 * 512;
        let mut text_mask = self.text_mask.buffer;
        text_mask.bytes = rows;
        Ok(EngramDeviceView {
            embeddings,
            text_mask,
            rows,
            layer_index,
        })
    }
    pub fn synchronize(&self) -> Result<()> {
        unsafe { self.library.cuda_stream_synchronize(self.stream.raw) }
    }
}
impl Drop for EngramDeviceRows<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.synchronize() {
            tracing::error!(%error, "draining native engram row production");
        }
    }
}
