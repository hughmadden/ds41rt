//! Shifted mHC boundaries around one native dSpark attention or FFN sublayer.
use super::DsparkWeights;
use crate::v41_memory::{DeviceAllocation, LoadStream};
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, V41Hc};
use std::ffi::c_void;

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
    pending: Option<usize>,
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
            pending: None,
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
        let launched = unsafe { self.enqueue_begin(rows, None, self.stream.raw) };
        let drained = self.synchronize();
        match launched.and_then(|output| drained.map(|()| output)) {
            Ok(output) => Ok(output),
            Err(error) => {
                self.invalidate();
                Err(error)
            }
        }
    }
    pub(super) fn invalidate(&mut self) {
        self.begun = None;
        self.ready = None;
        self.pending = None;
    }
    /// Caller owns the external stream and must drain it before releasing this
    /// borrow, including on error; an external normalized output must be distinct.
    pub(super) unsafe fn enqueue_begin(
        &mut self,
        rows: usize,
        normalized: Option<Ds41rtDeviceBuffer>,
        stream: *mut c_void,
    ) -> Result<Ds41rtDeviceBuffer> {
        self.invalidate();
        let normalized = normalized.unwrap_or(self.normalized.buffer);
        ensure!(
            !normalized.ptr.is_null() && normalized.bytes >= rows.saturating_mul(10240),
            "mHC normalized output is too small"
        );
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
                stream,
            )?;
            // The newly generated pre belongs to the NEXT sublayer.
            self.kernel.pre(
                self.residual.buffer,
                self.incoming_pre.buffer,
                self.collapsed.buffer,
                rows,
                stream,
            )?;
            self.stream.library.cuda_ds4_rmsnorm_bf16_rne_async(
                self.collapsed.buffer,
                self.weights.tensor(&self.names[3])?,
                normalized,
                rows as i32,
                5120,
                1e-6,
                stream,
            )?;
        }
        self.begun = Some(rows);
        let mut result = normalized;
        result.bytes = rows * 10240;
        Ok(result)
    }
    /// # Safety
    /// The BF16 sublayer result must be initialized for the rows returned by begin
    /// and all producer writes complete. Residual/coefficients must remain unchanged.
    pub unsafe fn finish(&mut self) -> Result<[Ds41rtDeviceBuffer; 2]> {
        let launched = unsafe { self.enqueue_finish(None, self.stream.raw) };
        let drained = self.synchronize();
        if let Err(error) = launched.and(drained) {
            self.invalidate();
            return Err(error);
        }
        unsafe { self.complete() }
    }
    /// External result producer must be ordered on stream; drain before release.
    pub(super) unsafe fn enqueue_finish(
        &mut self,
        result: Option<Ds41rtDeviceBuffer>,
        stream: *mut c_void,
    ) -> Result<()> {
        self.ready = None;
        self.pending = None;
        let rows = self.begun.take().context("mHC sublayer was not begun")?;
        let result = result.unwrap_or(self.sublayer.buffer);
        unsafe {
            self.kernel.post(
                result,
                self.residual.buffer,
                self.post.buffer,
                self.comb.buffer,
                self.output.buffer,
                rows,
                stream,
            )?;
        }
        self.pending = Some(rows);
        Ok(())
    }
    /// # Safety
    /// The stream used by enqueue_finish must have completed successfully.
    pub(super) unsafe fn complete(&mut self) -> Result<[Ds41rtDeviceBuffer; 2]> {
        self.ready = Some(self.pending.take().context("mHC post was not submitted")?);
        self.output()
    }
    /// # Safety
    /// A graph containing this boundary's post for exactly rows has completed.
    pub(super) unsafe fn complete_replay(
        &mut self,
        rows: usize,
    ) -> Result<[Ds41rtDeviceBuffer; 2]> {
        ensure!(rows > 0 && rows <= self.capacity, "invalid mHC replay rows");
        self.pending = Some(rows);
        unsafe { self.complete() }
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
