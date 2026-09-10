//! Owned native gathered-row upload and BF16 engram embedding production.
use crate::v41_memory::{DeviceAllocation, LoadStream};
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary};
use ds41rt_loader::EngramGatherView;

pub(crate) struct EngramDeviceRows<'a> {
    stream: LoadStream<'a>,
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
impl<'a> EngramDeviceRows<'a> {
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
