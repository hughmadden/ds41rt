//! Shifted mHC boundaries around one native dSpark attention or FFN sublayer.
use super::DsparkWeights;
use crate::v41_memory::{DeviceAllocation, LoadStream};
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, V41Hc};

pub(crate) struct HcSublayer<'weights, 'library> {
    stream: LoadStream<'library>,
    kernel: V41Hc<'library>,
    weights: &'weights DsparkWeights<'library>,
    names: [String; 4],
    residual: DeviceAllocation<'library>,
    incoming_pre: DeviceAllocation<'library>,
    next_pre: DeviceAllocation<'library>,
    post: DeviceAllocation<'library>,
    comb: DeviceAllocation<'library>,
    collapsed: DeviceAllocation<'library>,
    normalized: DeviceAllocation<'library>,
    sublayer: DeviceAllocation<'library>,
    output: DeviceAllocation<'library>,
    capacity: usize,
    begun: Option<usize>,
    ready: Option<usize>,
}
impl<'library> DsparkWeights<'library> {
    pub fn hc_sublayer(
        &self,
        stage: usize,
        attention: bool,
        capacity: usize,
        budget: usize,
    ) -> Result<HcSublayer<'_, 'library>> {
        ensure!(stage < 3, "invalid dSpark stage");
        ensure!(
            HcSublayer::device_bytes(capacity)? <= budget,
            "mHC sublayer exceeds budget"
        );
        let kind = if attention { "attn" } else { "ffn" };
        let names = [
            format!("mtp.{stage}.hc_{kind}_fn"),
            format!("mtp.{stage}.hc_{kind}_scale"),
            format!("mtp.{stage}.hc_{kind}_base"),
            format!("mtp.{stage}.{kind}_norm.weight"),
        ];
        for name in &names {
            self.tensor(name)?;
        }
        let library = self.experts[0].buffers[0].library;
        Ok(HcSublayer {
            stream: LoadStream {
                library,
                raw: library.cuda_stream_create()?,
            },
            kernel: library.v41_hc()?,
            weights: self,
            names,
            residual: DeviceAllocation::new(library, capacity * 40960)?,
            incoming_pre: DeviceAllocation::new(library, capacity * 16)?,
            next_pre: DeviceAllocation::new(library, capacity * 16)?,
            post: DeviceAllocation::new(library, capacity * 16)?,
            comb: DeviceAllocation::new(library, capacity * 64)?,
            collapsed: DeviceAllocation::new(library, capacity * 10240)?,
            normalized: DeviceAllocation::new(library, capacity * 10240)?,
            sublayer: DeviceAllocation::new(library, capacity * 10240)?,
            output: DeviceAllocation::new(library, capacity * 40960)?,
            capacity,
            begun: None,
            ready: None,
        })
    }
}
impl HcSublayer<'_, '_> {
    pub fn device_bytes(capacity: usize) -> Result<usize> {
        ensure!((1..=4096).contains(&capacity), "invalid mHC capacity");
        Ok(capacity * 112752)
    }
    /// Stable BF16 residual [capacity,4,5120] and incoming FP32 pre [capacity,4].
    /// The first stage starts with one-hot [1,0,0,0]; subsequent boundaries receive
    /// the preceding sublayer's next_pre. Never free or retain after owner drop.
    pub fn inputs(&self) -> [Ds41rtDeviceBuffer; 2] {
        [self.residual.buffer, self.incoming_pre.buffer]
    }
    /// Destination for the attention/FFN result; finish producer writes before finish.
    pub fn sublayer_result(&self) -> Ds41rtDeviceBuffer {
        self.sublayer.buffer
    }
    fn synchronize(&self) -> Result<()> {
        unsafe { self.stream.library.cuda_stream_synchronize(self.stream.raw) }
    }
    /// # Safety
    /// Both inputs must be initialized on this device with completed producer
    /// writes. Serialize use; do not overwrite residual/coefficient storage until finish.
    pub unsafe fn begin(&mut self, rows: usize) -> Result<Ds41rtDeviceBuffer> {
        self.begun = None;
        self.ready = None;
        ensure!(
            rows > 0 && rows <= self.capacity,
            "mHC rows exceed capacity"
        );
        unsafe {
            self.kernel.mixes(
                self.residual.buffer,
                self.weights.tensor(&self.names[0])?,
                self.weights.tensor(&self.names[1])?,
                self.weights.tensor(&self.names[2])?,
                self.next_pre.buffer,
                self.post.buffer,
                self.comb.buffer,
                rows,
                self.stream.raw,
            )?;
            // The newly generated pre belongs to the NEXT sublayer.
            self.kernel.pre(
                self.residual.buffer,
                self.incoming_pre.buffer,
                self.collapsed.buffer,
                rows,
                self.stream.raw,
            )?;
            self.stream.library.cuda_ds4_rmsnorm_bf16_rne_async(
                self.collapsed.buffer,
                self.weights.tensor(&self.names[3])?,
                self.normalized.buffer,
                rows as i32,
                5120,
                1e-6,
                self.stream.raw,
            )?;
        }
        self.synchronize()?;
        self.begun = Some(rows);
        let mut result = self.normalized.buffer;
        result.bytes = rows * 10240;
        Ok(result)
    }
    /// # Safety
    /// The BF16 sublayer result must be initialized for the rows returned by begin
    /// and all producer writes complete. Residual/coefficients must remain unchanged.
    pub unsafe fn finish(&mut self) -> Result<[Ds41rtDeviceBuffer; 2]> {
        self.ready = None;
        let rows = self.begun.take().context("mHC sublayer was not begun")?;
        unsafe {
            self.kernel.post(
                self.sublayer.buffer,
                self.residual.buffer,
                self.post.buffer,
                self.comb.buffer,
                self.output.buffer,
                rows,
                self.stream.raw,
            )?;
        }
        self.synchronize()?;
        self.ready = Some(rows);
        self.output()
    }
    /// BF16 residual [rows,4,5120] and FP32 next_pre [rows,4], borrowed until reuse.
    pub fn output(&self) -> Result<[Ds41rtDeviceBuffer; 2]> {
        let rows = self.ready.context("mHC sublayer output is incomplete")?;
        let mut residual = self.output.buffer;
        residual.bytes = rows * 40960;
        let mut pre = self.next_pre.buffer;
        pre.bytes = rows * 16;
        Ok([residual, pre])
    }
}
impl Drop for HcSublayer<'_, '_> {
    fn drop(&mut self) {
        if let Err(error) = self.synchronize() {
            tracing::error!(%error,"draining mHC sublayer");
        }
    }
}
