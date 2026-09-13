//! Token initialization and image replacement before the first target mHC block.
use crate::v41_memory::{DeviceAllocation, HostAllocation, LoadStream};
use crate::v41_tensors::NativeRtxTensors;
use anyhow::{Context, Result, ensure};
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary, V41AttentionOps};
use std::{ffi::c_void, marker::PhantomData};

pub(crate) struct TargetEmbedding<'a> {
    pub residual: Ds41rtDeviceBuffer,
    pub pre: Ds41rtDeviceBuffer,
    pub token_ids: &'a [u32],
    pub positions: &'a [u64],
    _owner: PhantomData<&'a ()>,
}

pub(crate) struct TargetEmbeddingWave<'w, 'a> {
    stream: LoadStream<'a>,
    ops: V41AttentionOps<'a>,
    table: &'w NativeRtxTensors<'a>,
    ids: DeviceAllocation<'a>,
    staging: HostAllocation<'a>,
    residual: DeviceAllocation<'a>,
    pre: DeviceAllocation<'a>,
    image_features: DeviceAllocation<'a>,
    image_indices: DeviceAllocation<'a>,
    capacity: usize,
    graph: Option<(*mut c_void, usize)>,
    tokens: Vec<u32>,
    positions: Vec<u64>,
    ready: bool,
}
impl<'w, 'a> TargetEmbeddingWave<'w, 'a> {
    pub fn device_bytes(capacity: usize) -> Result<usize> {
        ensure!(
            (1..=4096).contains(&capacity),
            "invalid target embedding capacity"
        );
        Ok(capacity * (4 + 40960 + 16 + 10240 + 4))
    }
    pub fn new(
        library: &'a NativeLibrary,
        table: &'w NativeRtxTensors<'a>,
        capacity: usize,
        budget: usize,
    ) -> Result<Self> {
        ensure!(
            Self::device_bytes(capacity)? <= budget,
            "target embedding exceeds budget"
        );
        ensure!(
            table.get("embed.weight")?.bytes == 129280 * 5120 * 2,
            "invalid shared embedding extent"
        );
        let ids = DeviceAllocation::new(library, capacity * 4)?;
        ensure!(
            table.get("embed.weight")?.device_id == ids.buffer.device_id,
            "target embedding table device differs"
        );
        Ok(Self {
            stream: LoadStream {
                library,
                raw: library.cuda_stream_create()?,
            },
            ops: library.v41_attention_ops()?,
            table,
            ids,
            staging: HostAllocation::new(library, capacity * 4)?,
            residual: DeviceAllocation::new(library, capacity * 40960)?,
            pre: DeviceAllocation::new(library, capacity * 16)?,
            image_features: DeviceAllocation::new(library, capacity * 10240)?,
            image_indices: DeviceAllocation::new(library, capacity * 4)?,
            capacity,
            graph: None,
            tokens: Vec::with_capacity(capacity),
            positions: Vec::with_capacity(capacity),
            ready: false,
        })
    }
    fn invalidate(&mut self) {
        self.ready = false;
        self.tokens.clear();
        self.positions.clear();
    }
    fn synchronize(&self) -> Result<()> {
        unsafe { self.stream.library.cuda_stream_synchronize(self.stream.raw) }
    }
    unsafe fn enqueue(&self, rows: usize) -> Result<()> {
        unsafe {
            self.ops.target_embed(
                self.table.get("embed.weight")?,
                self.ids.buffer,
                self.residual.buffer,
                self.pre.buffer,
                rows,
                self.stream.raw,
            )
        }
    }
    /// Initializes text rows in caller order. Positions may repeat across
    /// requests; request ownership remains the scheduler's responsibility.
    /// Captures when the live shape changes, then reuses stable graph storage.
    pub fn execute(&mut self, tokens: &[u32], positions: &[u64]) -> Result<TargetEmbedding<'_>> {
        self.invalidate();
        let rows = tokens.len();
        ensure!(
            rows > 0
                && rows <= self.capacity
                && rows == positions.len()
                && tokens.iter().all(|&id| id < 129280)
                && positions.iter().all(|&p| p < 1048576),
            "invalid target embedding tokens or positions"
        );
        let bytes: Vec<u8> = tokens.iter().flat_map(|id| id.to_ne_bytes()).collect();
        self.stream.library.copy_h2d(self.ids.buffer, &bytes)?;
        if self.graph.as_ref().is_none_or(|(_, n)| *n != rows) {
            self.clear_graph()?;
            let warmup = unsafe { self.enqueue(rows) };
            warmup.and(self.synchronize())?;
            unsafe {
                self.stream
                    .library
                    .cuda_graph_begin_capture(self.stream.raw)?;
            }
            let launched = unsafe { self.enqueue(rows) };
            let captured = unsafe { self.stream.library.cuda_graph_end_capture(self.stream.raw) };
            match (launched, captured) {
                (Ok(()), Ok(graph)) => self.graph = Some((graph, rows)),
                (Err(e), Ok(graph)) => {
                    unsafe {
                        self.stream.library.cuda_graph_exec_destroy(graph)?;
                    }
                    return Err(e);
                }
                (Err(e), Err(_)) | (Ok(()), Err(e)) => return Err(e),
            }
        }
        let graph = self.graph.context("target embedding graph missing")?.0;
        let launched = unsafe {
            self.stream
                .library
                .cuda_graph_launch(graph, self.stream.raw)
        };
        launched.and(self.synchronize())?;
        self.tokens.extend_from_slice(tokens);
        self.positions.extend_from_slice(positions);
        self.ready = true;
        self.output()
    }
    /// # Safety
    /// The caller owns the supplied stream and destinations and drains it before
    /// reusing this embedding owner, its staging, or either destination. Text
    /// embeddings are produced directly into the block, without an intermediate copy.
    pub unsafe fn enqueue_into(&mut self, tokens: &[u32], stream: *mut c_void,
        destination: [Ds41rtDeviceBuffer; 2]) -> Result<()> {
        self.invalidate();
        ensure!(!tokens.is_empty() && tokens.len() <= self.capacity
            && tokens.iter().all(|&id| id < 129280)
            && destination[0].bytes >= tokens.len()*40960 && destination[1].bytes >= tokens.len()*16
            && destination.iter().all(|b| b.device_id == self.ids.buffer.device_id),
            "invalid direct embedding producer input");
        for (dst, token) in self.staging.bytes_mut().chunks_exact_mut(4).zip(tokens) {
            dst.copy_from_slice(&token.to_ne_bytes());
        }
        unsafe {
            self.stream.library.copy_host_buffer_h2d_async(self.ids.buffer, self.staging.buffer,
                tokens.len()*4, stream)?;
            self.ops.target_embed(self.table.get("embed.weight")?, self.ids.buffer,
                destination[0], destination[1], tokens.len(), stream)
        }
    }
    /// Complete image span rows include learned delimiters. The row indices are
    /// sorted, unique positions within this flattened multi-request batch.
    pub fn execute_with_images(&mut self, tokens: &[u32], positions: &[u64],
        images: &[(usize, &[u8])]) -> Result<TargetEmbedding<'_>> {
        self.invalidate();
        ensure!(images.iter().enumerate().all(|(i, (row, bytes))|
            *row < tokens.len() && tokens[*row] == ds41rt_loader::V41_IMAGE_TOKEN_ID
                && bytes.len() == 10240 && (i == 0 || images[i-1].0 < *row)),
            "invalid target image rows or feature extent");
        self.execute(tokens, positions)?;
        if images.is_empty() { return self.output(); }
        self.ready = false;
        let mut features = Vec::with_capacity(images.len()*10240);
        let mut indices = Vec::with_capacity(images.len()*4);
        for &(row, data) in images {
            features.extend_from_slice(data);
            indices.extend_from_slice(&(row as u32).to_ne_bytes());
        }
        self.stream.library.copy_h2d(self.image_features.buffer, &features)?;
        self.stream.library.copy_h2d(self.image_indices.buffer, &indices)?;
        let launched = unsafe { self.stream.library.v41_vision_embed(
            self.image_features.buffer, self.image_indices.buffer, self.residual.buffer,
            images.len(), tokens.len(), self.stream.raw) };
        launched.and(self.synchronize())?;
        self.ready = true;
        self.output()
    }
    pub fn output(&self) -> Result<TargetEmbedding<'_>> {
        ensure!(self.ready, "target embeddings unpublished");
        let mut residual = self.residual.buffer;
        let mut pre = self.pre.buffer;
        residual.bytes = self.tokens.len() * 40960;
        pre.bytes = self.tokens.len() * 16;
        Ok(TargetEmbedding {
            residual,
            pre,
            token_ids: &self.tokens,
            positions: &self.positions,
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
impl Drop for TargetEmbeddingWave<'_, '_> {
    fn drop(&mut self) {
        if let Err(e) = self.clear_graph() {
            tracing::error!(%e, "draining target embedding");
        }
    }
}

#[cfg(test)]
mod tests;
