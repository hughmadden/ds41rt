//! Request-owned early engram I/O and accepted-prefix history transactions.
use crate::{
    EngramGatherPoll, EngramGatherTicket, EngramGatherer, EngramPrefetcher, EngramTable,
    EngramTokenMap, OfficialV41Catalog, PrefetchTicket,
};
use anyhow::{ensure, Context, Result};
use ds41rt_core::{EngramBatch, EngramHistory, ENGRAM_LAYERS};
use std::sync::Arc;

pub struct EngramRequestTokens<'a> {
    pub history: &'a EngramHistory,
    pub token_ids: &'a [u32],
    pub image_mask: Option<&'a [bool]>,
}
enum LayerIo {
    AwaitingAdmission,
    Queued(EngramGatherTicket),
    Consumed,
}
pub struct EngramWave {
    pipeline: Arc<()>,
    batches: Vec<Arc<EngramBatch>>,
    prefetch: Vec<PrefetchTicket>,
    layers: [LayerIo; 2],
    finished: bool,
}
impl EngramWave {
    pub fn batches(&self) -> &[Arc<EngramBatch>] {
        &self.batches
    }
    /// Cancel early I/O without committing any token history.
    pub fn cancel(&mut self) {
        self.finished = true;
        self.prefetch.clear();
        self.layers = [LayerIo::Consumed, LayerIo::Consumed];
    }
    /// Preflight history publication before an enclosing device-cache commit.
    pub fn validate_commit(&self, histories: &[&EngramHistory], accepted: &[usize]) -> Result<()> {
        ensure!(!self.finished, "engram wave is already finished");
        ensure!(
            histories.len() == self.batches.len() && accepted.len() == self.batches.len(),
            "engram wave acceptance dimensions differ"
        );
        for ((history, batch), &count) in histories.iter().zip(&self.batches).zip(accepted) {
            history.validate_commit(batch, count)?;
        }
        Ok(())
    }
    /// Commit accepted prefixes only after validating the complete physical wave.
    /// Rejected speculative rows never enter history; remaining I/O is cancelled.
    pub fn commit(
        &mut self,
        histories: &mut [&mut EngramHistory],
        accepted: &[usize],
    ) -> Result<()> {
        self.validate_commit(
            &histories.iter().map(|h| &**h).collect::<Vec<_>>(),
            accepted,
        )?;
        // Exclusive history borrows prevent changes between validation and commit.
        for ((history, batch), &count) in histories.iter_mut().zip(&self.batches).zip(accepted) {
            history.commit(batch, count)?;
        }
        self.cancel();
        Ok(())
    }
}

pub struct EngramPipeline {
    identity: Arc<()>,
    tables: [Arc<EngramTable>; 2],
    token_map: EngramTokenMap,
    prefetch: EngramPrefetcher,
    gatherer: EngramGatherer,
    capacity: usize,
}
impl EngramPipeline {
    /// # Safety
    /// Official checkpoint files must stay immutable while any mapping/job is alive.
    pub unsafe fn new(
        catalog: &OfficialV41Catalog,
        token_map: EngramTokenMap,
        capacity: usize,
        gather_slots: usize,
        staging_budget: usize,
    ) -> Result<Self> {
        ensure!(
            capacity > 0 && capacity <= 4096,
            "invalid engram pipeline capacity"
        );
        let tables = [
            Arc::new(unsafe { catalog.map_engram(ENGRAM_LAYERS[0] as usize)? }),
            Arc::new(unsafe { catalog.map_engram(ENGRAM_LAYERS[1] as usize)? }),
        ];
        Ok(Self {
            identity: Arc::new(()),
            tables,
            token_map,
            prefetch: EngramPrefetcher::new(32, capacity * 24, capacity * 48)?,
            gatherer: EngramGatherer::new(gather_slots, capacity, staging_budget)?,
            capacity,
        })
    }
    pub fn new_history(&self) -> Result<EngramHistory> {
        Ok(EngramHistory::new(self.token_map.pad_id())?)
    }
    /// Call as soon as decode, prefill or verification token IDs and image spans are known.
    /// Prepare all hashes before starting I/O, preserving committed request histories.
    pub fn prepare(&self, requests: &[EngramRequestTokens<'_>]) -> Result<EngramWave> {
        ensure!(
            !requests.is_empty() && requests.len() <= 16,
            "engram wave requires 1..16 requests"
        );
        let total = requests.iter().try_fold(0usize, |sum, request| {
            ensure!(
                !request.token_ids.is_empty(),
                "engram request batch is empty"
            );
            sum.checked_add(request.token_ids.len())
                .context("engram wave row overflow")
        })?;
        ensure!(
            total <= self.capacity,
            "engram wave exceeds pipeline capacity"
        );
        // Reject repeated histories before any work is admitted; one request owns
        // exactly one contiguous batch in a physical wave.
        for (index, request) in requests.iter().enumerate() {
            ensure!(
                !requests[..index]
                    .iter()
                    .any(|prior| std::ptr::eq(prior.history, request.history)),
                "engram wave repeats a request history"
            );
        }
        let batches: Result<Vec<_>> = requests
            .iter()
            .map(|request| {
                self.token_map
                    .prepare_batch(
                        request.history,
                        request.history.position(),
                        request.token_ids,
                        request.image_mask,
                        self.capacity,
                    )
                    .map(Arc::new)
            })
            .collect();
        let mut wave = EngramWave {
            pipeline: Arc::clone(&self.identity),
            batches: batches?,
            prefetch: Vec::with_capacity(32),
            layers: [LayerIo::AwaitingAdmission, LayerIo::AwaitingAdmission],
            finished: false,
        };
        for layer in 0..2 {
            for batch in &wave.batches {
                if let Some(ticket) =
                    self.prefetch
                        .try_submit_batch(Arc::clone(&self.tables[layer]), batch, layer)?
                {
                    wave.prefetch.push(ticket);
                }
            }
            self.admit(&mut wave, layer)?;
        }
        Ok(wave)
    }
    fn admit(&self, wave: &mut EngramWave, layer: usize) -> Result<()> {
        if matches!(wave.layers[layer], LayerIo::AwaitingAdmission) {
            if let Some(ticket) =
                self.gatherer
                    .try_submit(Arc::clone(&self.tables[layer]), &wave.batches, layer)?
            {
                wave.layers[layer] = LayerIo::Queued(ticket);
            }
        }
        Ok(())
    }
    /// Retry bounded admission and poll without waiting for mapped pages.
    /// A returned lease must be uploaded or dropped before its staging can be reused.
    pub fn poll(
        &self,
        wave: &mut EngramWave,
        histories: &[&EngramHistory],
        layer: usize,
    ) -> Result<EngramGatherPoll> {
        ensure!(!wave.finished, "engram wave is already finished");
        ensure!(layer < 2, "invalid engram layer index");
        let result = (|| {
            ensure!(
                Arc::ptr_eq(&self.identity, &wave.pipeline),
                "engram wave belongs to another pipeline"
            );
            ensure!(
                histories.len() == wave.batches.len(),
                "engram wave history count differs"
            );
            for (history, batch) in histories.iter().zip(&wave.batches) {
                history.validate_commit(batch, 0)?;
            }
            self.admit(wave, layer)?;
            let result = match &mut wave.layers[layer] {
                LayerIo::AwaitingAdmission => return Ok(EngramGatherPoll::Pending),
                LayerIo::Queued(ticket) => ticket.poll()?,
                LayerIo::Consumed => anyhow::bail!("engram layer result already consumed"),
            };
            if let EngramGatherPoll::Ready(lease) = &result {
                ensure!(
                    lease.batches().len() == wave.batches.len()
                        && lease
                            .batches()
                            .iter()
                            .zip(&wave.batches)
                            .all(|(a, b)| Arc::ptr_eq(a, b)),
                    "engram gather result belongs to a different wave"
                );
            }
            if !matches!(result, EngramGatherPoll::Pending) {
                wave.layers[layer] = LayerIo::Consumed;
            }
            Ok(result)
        })();
        if result.is_err() {
            wave.cancel();
        }
        result
    }
}
