//! One target text pass from request-owned tokens through layer 39 and logits.
use crate::v41_backbone_execution::BackboneExecution;
use crate::v41_backbone_lane::BackboneLane;
use crate::v41_engram::{layer::EngramGate, EngramDeviceRows};
use crate::v41_experts::coordinator::NativeTp4Wave;
use crate::v41_index_lane::IndexLane;
use crate::v41_requests::{RequestBatch, Requests};
use crate::v41_target_embedding::TargetEmbeddingWave;
use crate::v41_target_head::{TargetHeadWave, TargetLogits};
use anyhow::{ensure, Result};
use std::time::{Duration, Instant};
mod taps;
pub(crate) use taps::{TargetTapWave, TargetTaps};

#[derive(Default, Debug, PartialEq, Eq)]
enum State {
    #[default]
    Idle,
    Running,
    Ready(u64),
}
impl State {
    fn begin(&mut self) -> Result<()> {
        ensure!(
            *self == Self::Idle,
            "target pass must be committed or discarded before reuse"
        );
        *self = Self::Running;
        Ok(())
    }
    fn ready(&self, batch: u64) -> Result<()> {
        ensure!(
            *self == Self::Ready(batch),
            "target pass has no completed logits for this batch"
        );
        Ok(())
    }
}
/// Cancelling the async future must cancel mapped I/O and invalidate the batch.
struct BatchGuard<'a> {
    batch: &'a mut RequestBatch,
    completed: bool,
}
impl Drop for BatchGuard<'_> {
    fn drop(&mut self) {
        if !self.completed {
            self.batch.cancel();
        }
    }
}

/// Each alternating wave owns a separate instance; immutable weights are shared.
/// The CUDA-owning executor must poll this owner on one thread. Transport remains
/// external so a scheduler can assign a separate connection set to each wave.
pub(crate) struct TargetPass<'w, 'a> {
    embedding: TargetEmbeddingWave<'w, 'a>,
    lane: BackboneLane<'w, 'a>,
    index: IndexLane<'w, 'a>,
    execution: BackboneExecution<'w, 'a>,
    upload: EngramDeviceRows<'a>,
    gates: [EngramGate<'w, 'a>; 2],
    head: TargetHeadWave<'w, 'a>,
    taps: TargetTapWave<'a>,
    engram_timeout: Duration,
    state: State,
}
impl<'w, 'a> TargetPass<'w, 'a> {
    pub fn new(
        embedding: TargetEmbeddingWave<'w, 'a>,
        lane: BackboneLane<'w, 'a>,
        index: IndexLane<'w, 'a>,
        execution: BackboneExecution<'w, 'a>,
        upload: EngramDeviceRows<'a>,
        gates: [EngramGate<'w, 'a>; 2],
        head: TargetHeadWave<'w, 'a>,
        taps: TargetTapWave<'a>,
        engram_timeout: Duration,
    ) -> Result<Self> {
        ensure!(
            gates[0].layer() == 1 && gates[1].layer() == 14,
            "target engram gates out of order"
        );
        ensure!(!engram_timeout.is_zero(), "engram timeout must be positive");
        Ok(Self {
            embedding,
            lane,
            index,
            execution,
            upload,
            gates,
            head,
            taps,
            engram_timeout,
            state: State::Idle,
        })
    }
    /// # Safety
    /// All components belong to the same device, capacity and official model.
    /// The caller exclusively owns CUDA buffers and polls on the owning thread.
    /// Selected rows are in this batch's flattened request order, at most 80.
    /// Produces private dSpark taps for every input row; image replacement is separate.
    pub async unsafe fn execute(
        &mut self,
        requests: &Requests<'a>,
        batch: &mut RequestBatch,
        transport: &mut NativeTp4Wave<'a>,
        placement: u64,
        selected: &[usize],
    ) -> Result<TargetLogits<'_>> {
        requests.validate(batch)?;
        let id = batch.cache()?.identity();
        let rows = batch.cache()?.positions().len();
        ensure!(
            !selected.is_empty()
                && selected.len() <= 80
                && selected.iter().all(|&i| i < rows)
                && selected
                    .iter()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len()
                    == selected.len(),
            "invalid target head row selection"
        );
        self.state.begin()?;
        let mut guard = BatchGuard {
            batch,
            completed: false,
        };
        self.taps.begin(guard.batch.cache()?)?;
        self.execution.restart();
        self.lane.restart()?;
        self.index.restart()?;
        unsafe {
            requests.begin_text(guard.batch, &mut self.embedding, &mut self.lane)?;
        }
        for layer in 0..40 {
            if layer != 0 {
                self.lane.advance()?;
                if let Some(gate) = [1, 14].iter().position(|&l| l == layer) {
                    let start = Instant::now();
                    while !unsafe {
                        requests.poll_engram(
                            guard.batch,
                            &mut self.upload,
                            &mut self.gates[gate],
                            &mut self.lane,
                        )?
                    } {
                        ensure!(
                            start.elapsed() < self.engram_timeout,
                            "target engram gather timed out at layer {layer}"
                        );
                        tokio::time::sleep(Duration::from_millis(1)).await;
                    }
                }
                if layer >= 37 {
                    unsafe {
                        self.taps
                            .capture(guard.batch.cache()?, &self.lane.prepared_input()?)?;
                    }
                }
                unsafe {
                    self.lane.begin_prepared()?;
                }
            }
            unsafe {
                self.execution
                    .execute_layer(
                        requests.cache(),
                        guard.batch.cache()?,
                        &mut self.lane,
                        &mut self.index,
                        transport,
                        placement,
                        guard.batch.image_mask(),
                    )
                    .await?;
            }
        }
        self.taps.output(guard.batch.cache()?)?;
        let output = self.lane.output()?;
        unsafe {
            self.head.execute_block(&output, selected)?;
        }
        self.state = State::Ready(id);
        guard.completed = true;
        self.head.output()
    }
    pub fn output(&self, batch: &RequestBatch) -> Result<TargetLogits<'_>> {
        self.state.ready(batch.cache()?.identity())?;
        self.head.output()
    }
    pub fn taps(&self, batch: &RequestBatch) -> Result<TargetTaps<'_>> {
        self.state.ready(batch.cache()?.identity())?;
        self.taps.output(batch.cache()?)
    }
    /// The scheduler samples/validates logits before publishing accepted input
    /// prefixes. A failed commit consumes this pass; discard before reuse.
    pub fn commit(
        &mut self,
        requests: &mut Requests<'a>,
        batch: &mut RequestBatch,
        accepted: &[u32],
    ) -> Result<()> {
        self.state.ready(batch.cache()?.identity())?;
        self.state = State::Running;
        self.taps.reset();
        requests.commit(batch, &mut self.execution, accepted)?;
        self.state = State::Idle;
        Ok(())
    }
    /// Call only after execution future/borrowed outputs and transport consumers
    /// have been dropped. Request admission survives discarded private proposals.
    pub fn discard(&mut self, batch: &mut RequestBatch) -> Result<()> {
        batch.cancel();
        self.taps.reset();
        self.state = State::Running;
        self.lane.restart()?;
        self.index.restart()?;
        self.execution.restart();
        self.state = State::Idle;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::State;
    #[test]
    fn incomplete_foreign_and_consumed_passes_cannot_publish() {
        let mut state = State::default();
        assert!(state.ready(1).is_err());
        state.begin().unwrap();
        assert!(state.begin().is_err());
        assert!(state.ready(1).is_err());
        state = State::Ready(1);
        state.ready(1).unwrap();
        assert!(state.ready(2).is_err());
        assert!(state.begin().is_err());
        state = State::Idle;
        assert!(state.ready(1).is_err());
        state.begin().unwrap();
    }
}

#[cfg(test)]
mod distributed_tests;
