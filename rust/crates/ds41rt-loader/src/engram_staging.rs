//! Bounded gathered native rows; run page-faulting gathers on an I/O worker.
use crate::EngramTable;
use anyhow::{ensure, Context, Result};
use ds41rt_core::{EngramBatch, ENGRAM_ROWS};

pub struct EngramBatchStaging {
    capacity: usize,
    weights: Vec<u8>,
    scales: Vec<u8>,
    text_mask: Vec<u8>,
    order: Vec<(u64, usize)>,
    ready: Option<(usize, usize)>,
}
/// Token-major native rows: 24 embeddings of width 256 per token.
/// Image embeddings are zero, scales are one, and the text mask is zero.
pub struct EngramGatherView<'a> {
    pub weights: &'a [u8],
    pub scales: &'a [u8],
    pub text_mask: &'a [u8],
    pub rows: usize,
    pub layer_index: usize,
}
impl EngramBatchStaging {
    /// Vec payload budget, excluding allocator metadata and request-owned batches.
    pub fn storage_bytes(capacity: usize) -> Result<usize> {
        ensure!(
            capacity > 0 && capacity <= 4096,
            "invalid engram staging capacity"
        );
        capacity
            .checked_mul(24 * (256 + 8 + std::mem::size_of::<(u64, usize)>()) + 1)
            .context("engram staging budget overflow")
    }
    pub fn new(capacity: usize) -> Result<Self> {
        ensure!(
            capacity > 0 && capacity <= 4096,
            "invalid engram staging capacity"
        );
        Ok(Self {
            capacity,
            weights: vec![0; capacity * 24 * 256],
            scales: vec![127; capacity * 24 * 8],
            text_mask: vec![0; capacity],
            order: Vec::with_capacity(capacity * 24),
            ready: None,
        })
    }
    /// Concatenate request-owned batches without committing their histories.
    /// Read each distinct table row once, in address order, then scatter duplicates
    /// back to their canonical token/head slots using preallocated storage.
    /// Any failure invalidates the view; no eager table load is performed.
    pub fn gather(
        &mut self,
        table: &EngramTable,
        batches: &[&EngramBatch],
        layer_index: usize,
    ) -> Result<EngramGatherView<'_>> {
        self.ready = None;
        ensure!(
            !batches.is_empty() && batches.len() <= 16,
            "engram wave requires 1..16 request batches"
        );
        let expected_rows = *ENGRAM_ROWS
            .get(layer_index)
            .context("invalid engram layer index")?;
        ensure!(
            table.weights().rows() == expected_rows,
            "engram table belongs to another layer"
        );
        let rows = batches.iter().try_fold(0usize, |sum, batch| {
            sum.checked_add(batch.hashes().len())
                .context("engram batch length overflow")
        })?;
        ensure!(
            rows > 0 && rows <= self.capacity,
            "engram wave exceeds staging capacity"
        );
        self.order.clear();
        let mut token = 0;
        for batch in batches {
            for (row, hashes) in batch.hashes().iter().enumerate() {
                let text = batch.is_image(row) == Some(false);
                self.text_mask[token] = u8::from(text);
                if text {
                    for (head, &address) in hashes[layer_index].iter().enumerate() {
                        ensure!(address < expected_rows, "engram hash is outside its table");
                        self.order.push((address, token * 24 + head));
                    }
                }
                token += 1;
            }
        }
        self.weights[..rows * 24 * 256].fill(0);
        self.scales[..rows * 24 * 8].fill(127);
        self.order.sort_unstable();
        let mut previous = None;
        for &(address, slot) in &self.order {
            if let Some((last_address, source)) = previous {
                if last_address == address {
                    self.weights
                        .copy_within(source * 256..(source + 1) * 256, slot * 256);
                    self.scales
                        .copy_within(source * 8..(source + 1) * 8, slot * 8);
                    continue;
                }
            }
            table
                .weights()
                .gather_into(&[address], &mut self.weights[slot * 256..(slot + 1) * 256])?;
            table
                .scales()
                .gather_into(&[address], &mut self.scales[slot * 8..(slot + 1) * 8])?;
            previous = Some((address, slot));
        }
        self.ready = Some((rows, layer_index));
        self.view()
    }
    pub fn view(&self) -> Result<EngramGatherView<'_>> {
        let (rows, layer_index) = self.ready.context("engram staging is not complete")?;
        Ok(EngramGatherView {
            weights: &self.weights[..rows * 24 * 256],
            scales: &self.scales[..rows * 24 * 8],
            text_mask: &self.text_mask[..rows],
            rows,
            layer_index,
        })
    }
}
