//! Final target mHC collapse, RMS norm and the vocabulary shared with dSpark.
use crate::v41_attention_binding::QueryBinding;
use crate::v41_block::BlockOutput;
use crate::v41_memory::{DeviceAllocation, HostAllocation, LoadStream};
use crate::v41_tensors::{NativeRtxTensors, VocabularyHead};
use anyhow::{Context, Result, ensure};
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary, V41Hc, V41VocabularyProjection};
use ds41rt_loader::OfficialV41Catalog;
use std::{ffi::c_void, marker::PhantomData};
const STRIDES: [usize; 7] = [40960, 16, 10240, 10240, 517120, 4, 4];
pub(crate) struct TargetHeadWeights<'a> {
    library: &'a NativeLibrary,
    norm: NativeRtxTensors<'a>,
}
impl<'a> TargetHeadWeights<'a> {
    pub fn device_bytes(catalog: &OfficialV41Catalog) -> Result<usize> {
        let bytes = NativeRtxTensors::plan(catalog, &["norm.weight".into()])?;
        ensure!(bytes == 10240, "unexpected target norm geometry");
        Ok(bytes)
    }
    pub fn load(
        library: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        budget: usize,
        staging: usize,
    ) -> Result<Self> {
        ensure!(
            Self::device_bytes(catalog)? <= budget,
            "target norm exceeds budget"
        );
        Ok(Self {
            library,
            norm: NativeRtxTensors::load(
                library,
                catalog,
                &["norm.weight".into()],
                budget,
                staging,
            )?,
        })
    }
    pub fn wave<'w>(
        &'w self,
        head: &'w VocabularyHead<'a>,
        capacity: usize,
        budget: usize,
    ) -> Result<TargetHeadWave<'w, 'a>> {
        ensure!(
            TargetHeadWave::device_bytes(capacity)? <= budget,
            "target head wave exceeds budget"
        );
        let workspace =
            DeviceAllocation::new(self.library, V41VocabularyProjection::WORKSPACE_BYTES)?;
        let projection = unsafe { self.library.v41_vocabulary_head(workspace.buffer)? };
        let buffers = STRIDES
            .into_iter()
            .map(|n| DeviceAllocation::new(self.library, n * capacity))
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            head.weight()?.device_id == buffers[0].buffer.device_id
                && self.norm.get("norm.weight")?.device_id == buffers[0].buffer.device_id,
            "target head weight device differs"
        );
        Ok(TargetHeadWave {
            stream: LoadStream {
                library: self.library,
                raw: self.library.cuda_stream_create()?,
            },
            projection,
            _workspace: workspace,
            hc: self.library.v41_hc()?,
            buffers,
            weights: self,
            head,
            capacity,
            graph: None,
            ready: None,
            origin: None,
            selected: Vec::new(),
            tokens: Vec::new(),
            greedy_staging: HostAllocation::new(self.library, capacity * 8)?,
            greedy_ready: false,
        })
    }
}
pub(crate) struct TargetLogits<'a> {
    pub rows: usize,
    pub collapsed: Ds41rtDeviceBuffer,
    pub normalized: Ds41rtDeviceBuffer,
    pub logits: Ds41rtDeviceBuffer,
    pub selected_rows: &'a [usize],
    pub token_positions: &'a [u64],
    origin: Option<QueryBinding>,
    _owner: PhantomData<&'a ()>,
}
impl TargetLogits<'_> {
    pub fn binding(&self) -> Result<QueryBinding> {
        self.origin.context("target logits have no block origin")
    }
}
pub(crate) struct TargetHeadWave<'w, 'a> {
    stream: LoadStream<'a>,
    projection: V41VocabularyProjection<'a>,
    _workspace: DeviceAllocation<'a>,
    hc: V41Hc<'a>,
    buffers: Vec<DeviceAllocation<'a>>,
    weights: &'w TargetHeadWeights<'a>,
    head: &'w VocabularyHead<'a>,
    capacity: usize,
    graph: Option<(*mut c_void, usize)>,
    ready: Option<usize>,
    origin: Option<QueryBinding>,
    selected: Vec<usize>,
    tokens: Vec<u64>,
    greedy_staging: HostAllocation<'a>,
    greedy_ready: bool,
}
impl TargetHeadWave<'_, '_> {
    /// Up to 16 decode/prefill-last rows or 80 verification rows per head wave.
    pub fn device_bytes(capacity: usize) -> Result<usize> {
        ensure!(
            (1..=80).contains(&capacity),
            "target head capacity must be 1..80"
        );
        Ok(V41VocabularyProjection::WORKSPACE_BYTES + capacity * STRIDES.iter().sum::<usize>())
    }
    fn b(&self, i: usize) -> Ds41rtDeviceBuffer {
        self.buffers[i].buffer
    }
    pub fn inputs(&self) -> [Ds41rtDeviceBuffer; 2] {
        [self.b(0), self.b(1)]
    }
    fn synchronize(&self) -> Result<()> {
        unsafe { self.stream.library.cuda_stream_synchronize(self.stream.raw) }
    }
    fn invalidate(&mut self) {
        self.greedy_ready = false;
        self.ready = None;
        self.origin = None;
        self.selected.clear();
        self.tokens.clear();
    }
    fn validate(&mut self, rows: usize) -> Result<()> {
        self.invalidate();
        ensure!(
            rows > 0 && rows <= self.capacity,
            "target head rows exceed capacity"
        );
        Ok(())
    }
    unsafe fn enqueue(&self, rows: usize) -> Result<()> {
        unsafe {
            self.hc
                .pre(self.b(0), self.b(1), self.b(2), rows, self.stream.raw)?;
            self.stream.library.cuda_ds4_rmsnorm_bf16_rne_async(
                self.b(2),
                self.weights.norm.get("norm.weight")?,
                self.b(3),
                rows as i32,
                5120,
                1e-20,
                self.stream.raw,
            )?;
            self.projection.launch(
                self.b(3),
                self.head.weight()?,
                self.b(4),
                rows,
                self.stream.raw,
            )?;
            Ok(())
        }
    }
    /// # Safety
    /// Residual and pre inputs are finite and initialized, with exclusive storage
    /// until the call drains. Raw outputs carry no block identity.
    pub unsafe fn execute(&mut self, rows: usize) -> Result<TargetLogits<'_>> {
        self.validate(rows)?;
        let launched = unsafe { self.enqueue(rows) };
        launched.and(self.synchronize())?;
        self.ready = Some(rows);
        self.output()
    }
    /// # Safety
    /// Same initialized-input contract as execute; capture drains its warmup.
    pub unsafe fn capture(&mut self, rows: usize) -> Result<()> {
        self.invalidate();
        ensure!(self.graph.is_none(), "target head graph already captured");
        unsafe {
            self.execute(rows)?;
        }
        self.invalidate();
        unsafe {
            self.stream
                .library
                .cuda_graph_begin_capture(self.stream.raw)?;
        }
        let launched = unsafe { self.enqueue(rows) };
        let captured = unsafe { self.stream.library.cuda_graph_end_capture(self.stream.raw) };
        match (launched, captured) {
            (Ok(()), Ok(graph)) => {
                self.graph = Some((graph, rows));
                Ok(())
            }
            (Err(e), Ok(graph)) => {
                unsafe {
                    self.stream.library.cuda_graph_exec_destroy(graph)?;
                }
                Err(e)
            }
            (Err(e), Err(_)) | (Ok(()), Err(e)) => Err(e),
        }
    }
    /// # Safety
    /// Same inputs as execute; live row count must match the captured graph.
    pub unsafe fn replay(&mut self, rows: usize) -> Result<TargetLogits<'_>> {
        self.validate(rows)?;
        let (graph, count) = self.graph.context("target head graph missing")?;
        ensure!(rows == count, "target head captured rows differ");
        let launched = unsafe {
            self.stream
                .library
                .cuda_graph_launch(graph, self.stream.raw)
        };
        launched.and(self.synchronize())?;
        self.ready = Some(rows);
        self.output()
    }
    fn slice(mut b: Ds41rtDeviceBuffer, offset: usize, bytes: usize) -> Result<Ds41rtDeviceBuffer> {
        ensure!(
            offset <= b.bytes && bytes <= b.bytes - offset,
            "target head row slice exceeds buffer"
        );
        b.ptr = unsafe { b.ptr.cast::<u8>().add(offset).cast() };
        b.bytes = bytes;
        Ok(b)
    }
    /// # Safety
    /// Completed layer-39 residual/pre storage remains immutable until all row
    /// copies drain. Selection order defines output order; each row occurs once.
    unsafe fn copy_block(
        &mut self,
        block: &BlockOutput<'_>,
        selected: &[usize],
    ) -> Result<()> {
        self.invalidate();
        ensure!(
            block.layer == 39
                && block.binding().layer() == 39
                && block.tokens.len() <= 4096
                && !selected.is_empty()
                && selected.len() <= self.capacity
                && selected.iter().all(|&i| i < block.tokens.len())
                && selected
                    .iter()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len()
                    == selected.len()
                && block.residual.bytes == block.tokens.len() * 40960
                && block.pre.bytes == block.tokens.len() * 16
                && block.residual.device_id == self.b(0).device_id
                && block.pre.device_id == self.b(0).device_id,
            "target head block or selected rows differ"
        );
        let copied = (|| -> Result<()> {
            let mut first = 0;
            while first < selected.len() {
                let mut count = 1;
                while first + count < selected.len()
                    && selected[first + count] == selected[first] + count
                {
                    count += 1;
                }
                for (i, src, stride) in [(0, block.residual, 40960), (1, block.pre, 16)] {
                    unsafe {
                        self.stream.library.copy_d2d_async(
                            Self::slice(self.b(i), first * stride, count * stride)?,
                            Self::slice(src, selected[first] * stride, count * stride)?,
                            count * stride,
                            self.stream.raw,
                        )?;
                    }
                }
                first += count;
            }
            Ok(())
        })();
        if let Err(error) = copied {
            self.synchronize()?;
            return Err(error);
        }
        Ok(())
    }
    unsafe fn capture_block_head(&mut self, rows: usize) -> Result<()> {
        if self.graph.as_ref().is_none_or(|(_, n)| *n != rows) {
            self.clear_graph()?;
            unsafe { self.capture(rows)?; }
        }
        Ok(())
    }
    fn publish_block(&mut self, block: &BlockOutput<'_>, selected: &[usize]) {
        self.ready = Some(selected.len());
        self.origin = Some(block.binding());
        self.selected.extend_from_slice(selected);
        self.tokens.extend(selected.iter().map(|&i| block.tokens[i]));
    }
    /// Completed block inputs remain immutable through the drained head pass.
    pub unsafe fn execute_block(&mut self, block: &BlockOutput<'_>, selected: &[usize])
        -> Result<TargetLogits<'_>> {
        unsafe { self.copy_block(block, selected)?; }
        self.synchronize()?;
        unsafe { self.capture_block_head(selected.len())?; self.replay(selected.len())?; }
        self.publish_block(block, selected);
        self.output()
    }
    /// Same ownership contract, yielding the owner thread during GPU completion.
    /// First-use graph capture still drains its warmup; steady replay is cooperative.
    pub async unsafe fn execute_block_cooperative(&mut self, block: &BlockOutput<'_>, selected: &[usize])
        -> Result<TargetLogits<'_>> {
        unsafe { self.copy_block(block, selected)?; }
        // A warm graph consumes the copies on this same stream; only first-use
        // capture needs a completed input before its synchronous warmup.
        if self.graph.as_ref().is_none_or(|(_, n)| *n != selected.len()) {
            self.stream.wait().await?;
            unsafe { self.capture_block_head(selected.len())?; }
        }
        self.invalidate();
        let graph = self.graph.context("target head graph missing")?.0;
        let launched = unsafe { self.stream.library.cuda_graph_launch(graph, self.stream.raw) };
        let drained = self.stream.wait().await;
        launched.and(drained)?;
        self.publish_block(block, selected);
        self.output()
    }
    /// Keep logits on the device, downloading only checked greedy IDs/scores.
    pub async unsafe fn execute_block_greedy(&mut self, block: &BlockOutput<'_>,
        selected: &[usize], cooperative: bool) -> Result<()> {
        unsafe { self.copy_block(block, selected)?; }
        let rows = selected.len();
        if self.graph.as_ref().is_none_or(|(_, n)| *n != rows) {
            if cooperative { self.stream.wait().await?; } else { self.synchronize()?; }
            unsafe { self.capture_block_head(rows)?; }
        }
        self.invalidate();
        let graph = self.graph.context("target head graph missing")?.0;
        let logits = Self::slice(self.b(4), 0, rows * STRIDES[4])?;
        let indices = Self::slice(self.b(5), 0, rows * 4)?;
        let scores = Self::slice(self.b(6), 0, rows * 4)?;
        let launched = (|| -> Result<()> { unsafe {
            let lib = self.stream.library;
            lib.cuda_graph_launch(graph, self.stream.raw)?;
            lib.cuda_logits_argmax_checked_f32_async(logits, indices, scores, rows, 129280, self.stream.raw)?;
            let host = self.greedy_staging.bytes_mut();
            lib.copy_d2h_async(&mut host[..rows*4], indices, self.stream.raw)?;
            lib.copy_d2h_async(&mut host[rows*4..rows*8], scores, self.stream.raw)?;
            Ok(())
        } })();
        let drained = if cooperative { self.stream.wait().await } else { self.synchronize() };
        launched.and(drained)?;
        self.publish_block(block, selected);
        self.greedy_ready = true;
        Ok(())
    }
    pub fn greedy_output(&mut self) -> Result<Vec<(u32, f32)>> {
        ensure!(self.greedy_ready, "compact head output unpublished");
        let rows = self.ready.context("compact head rows unpublished")?;
        let host = self.greedy_staging.bytes_mut();
        Ok((0..rows).map(|i| (
            u32::from_ne_bytes(host[i*4..i*4+4].try_into().unwrap()),
            f32::from_ne_bytes(host[rows*4+i*4..rows*4+i*4+4].try_into().unwrap()),
        )).collect())
    }
    pub fn output(&self) -> Result<TargetLogits<'_>> {
        let rows = self.ready.context("target logits unpublished")?;
        let b = |i| {
            let mut b = self.b(i);
            b.bytes = rows * STRIDES[i];
            b
        };
        Ok(TargetLogits {
            rows,
            collapsed: b(2),
            normalized: b(3),
            logits: b(4),
            selected_rows: &self.selected,
            token_positions: &self.tokens,
            origin: self.origin,
            _owner: PhantomData,
        })
    }
    pub fn clear_graph(&mut self) -> Result<()> {
        self.invalidate();
        self.synchronize()?;
        if let Some((graph, _)) = self.graph.take() {
            unsafe {
                self.stream.library.cuda_graph_exec_destroy(graph)?;
            }
        }
        Ok(())
    }
}
impl Drop for TargetHeadWave<'_, '_> {
    fn drop(&mut self) {
        if let Err(e) = self.clear_graph() {
            tracing::error!(%e,"draining target head");
        }
    }
}
