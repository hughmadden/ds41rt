//! One admission identity for persistent backbone caches and mapped engram history.
use crate::v41_backbone_cache::{BackboneCache, CacheBatch, CacheLease, CacheWork};
use crate::v41_backbone_execution::BackboneExecution;
use crate::v41_backbone_lane::BackboneLane;
use crate::v41_engram::layer::EngramGate;
use crate::v41_engram::{EngramDeviceRows, EngramUploadPoll};
use crate::v41_target_embedding::TargetEmbeddingWave;
use anyhow::{ensure, Context, Result};
use ds41rt_core::EngramHistory;
use ds41rt_ffi::NativeLibrary;
use ds41rt_loader::{EngramPipeline, EngramRequestTokens, EngramWave};
use ds41rt_transport::ExpertV2SourceKind;

struct Request {
    lease: CacheLease,
    history: EngramHistory,
}
pub(crate) struct RequestTokens<'a> {
    pub lease: CacheLease,
    pub tokens: &'a [u32],
    pub image_mask: Option<&'a [bool]>,
    pub kind: ExpertV2SourceKind,
}
pub(crate) struct RequestBatch {
    cache: CacheBatch,
    engram: EngramWave,
    leases: Vec<CacheLease>,
    tokens: Vec<u32>,
    image_mask: Vec<u8>,
    finished: bool,
}
impl RequestBatch {
    pub fn cache(&self) -> Result<&CacheBatch> {
        ensure!(!self.finished, "request batch finished");
        Ok(&self.cache)
    }
    pub fn image_mask(&self) -> &[u8] {
        &self.image_mask
    }
    pub fn cancel(&mut self) {
        self.finished = true;
        self.engram.cancel();
    }
}
impl Drop for RequestBatch {
    fn drop(&mut self) {
        self.engram.cancel();
    }
}
pub(crate) struct Requests<'a> {
    cache: BackboneCache<'a>,
    pipeline: EngramPipeline,
    slots: Vec<Option<Request>>,
}
impl<'a> Requests<'a> {
    pub fn new(
        library: &'a NativeLibrary,
        pipeline: EngramPipeline,
        slots: usize,
        pages: [usize; 4],
        cache_budget: usize,
    ) -> Result<Self> {
        Ok(Self {
            cache: BackboneCache::new(library, slots, pages, cache_budget)?,
            pipeline,
            slots: (0..slots).map(|_| None).collect(),
        })
    }
    pub fn cache(&self) -> &BackboneCache<'a> {
        &self.cache
    }
    fn request(&self, lease: CacheLease) -> Result<&Request> {
        self.cache.request_id(lease)?;
        self.slots
            .iter()
            .flatten()
            .find(|r| r.lease == lease)
            .context("request history missing")
    }
    pub fn admit(&mut self, slot: usize, id: u64) -> Result<CacheLease> {
        ensure!(
            slot < self.slots.len() && self.slots[slot].is_none(),
            "request slot occupied"
        );
        let history = self.pipeline.new_history()?;
        let lease = self.cache.begin_request(slot, id)?;
        self.slots[slot] = Some(Request { lease, history });
        Ok(lease)
    }
    pub fn release(&mut self, lease: CacheLease) -> Result<()> {
        let slot = self
            .slots
            .iter()
            .position(|r| r.as_ref().is_some_and(|r| r.lease == lease))
            .context("request already released")?;
        // Revoke the history owner even if device-cache cleanup fails.
        self.slots[slot] = None;
        // Cache failure invalidation may already have revoked its root lease.
        if self.cache.request_id(lease).is_ok() {
            self.cache.release(&[lease])?;
        }
        Ok(())
    }
    pub fn validate(&self, batch: &RequestBatch) -> Result<()> {
        ensure!(!batch.finished, "request batch finished");
        self.cache.validate_batch(&batch.cache)?;
        for &lease in &batch.leases {
            ensure!(
                self.request(lease)?.history.position() == self.cache.committed_end(lease)?,
                "engram and backbone history differ"
            );
        }
        Ok(())
    }
    /// Start hashes, prefetch and bounded gather as soon as token IDs are known.
    pub fn prepare(&self, requests: &[RequestTokens<'_>]) -> Result<RequestBatch> {
        let work = requests
            .iter()
            .map(|r| {
                Ok(CacheWork {
                    lease: r.lease,
                    tokens: u32::try_from(r.tokens.len())?,
                    kind: r.kind,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let cache = self.cache.plan(&work)?;
        let mut inputs = Vec::with_capacity(requests.len());
        let mut tokens = Vec::new();
        let mut mask = Vec::new();
        for r in requests {
            let history = &self.request(r.lease)?.history;
            ensure!(
                history.position() == self.cache.committed_end(r.lease)?,
                "request histories differ"
            );
            ensure!(
                r.image_mask.is_none_or(|m| m.len() == r.tokens.len()),
                "image mask length differs"
            );
            inputs.push(EngramRequestTokens {
                history,
                token_ids: r.tokens,
                image_mask: r.image_mask,
            });
            tokens.extend_from_slice(r.tokens);
            mask.extend((0..r.tokens.len()).map(|i| u8::from(r.image_mask.is_some_and(|m| m[i]))));
        }
        let engram = self.pipeline.prepare(&inputs)?;
        Ok(RequestBatch {
            cache,
            engram,
            leases: requests.iter().map(|r| r.lease).collect(),
            tokens,
            image_mask: mask,
            finished: false,
        })
    }
    /// # Safety
    /// No external writes race embedding or lane buffers. This text entry rejects
    /// images; vision replacement must be implemented before admitting them here.
    pub unsafe fn begin_text(
        &self,
        batch: &RequestBatch,
        embedding: &mut TargetEmbeddingWave<'_, '_>,
        lane: &mut BackboneLane<'_, '_>,
    ) -> Result<()> {
        self.validate(batch)?;
        ensure!(
            batch.image_mask.iter().all(|&b| b == 0),
            "image batch requires vision embedding replacement"
        );
        let positions = batch.cache.positions();
        let embedded = embedding.execute(&batch.tokens, &positions)?;
        unsafe {
            lane.begin_embedded(&embedded)?;
        }
        Ok(())
    }
    /// # Safety
    /// This lane was initialized/advanced for batch and all device producers have
    /// completed. Poll on the CUDA-owning thread. False means I/O is still pending.
    pub unsafe fn poll_engram(
        &self,
        batch: &mut RequestBatch,
        upload: &mut EngramDeviceRows<'_>,
        gate: &mut EngramGate<'_, '_>,
        lane: &mut BackboneLane<'_, '_>,
    ) -> Result<bool> {
        self.validate(batch)?;
        let (pending_layer, positions) = lane.pending_engram()?;
        let layer = ds41rt_core::ENGRAM_LAYERS
            .iter()
            .position(|&l| l as usize == pending_layer)
            .context("prepared layer has no engram")?;
        ensure!(
            positions == batch.cache.positions(),
            "engram lane positions differ"
        );
        let histories = batch
            .leases
            .iter()
            .map(|&l| Ok(&self.request(l)?.history))
            .collect::<Result<Vec<_>>>()?;
        match upload.poll_wave(&self.pipeline, &mut batch.engram, &histories, layer)? {
            EngramUploadPoll::Pending => Ok(false),
            EngramUploadPoll::Cancelled => anyhow::bail!("engram gather cancelled"),
            EngramUploadPoll::Ready(rows) => {
                unsafe {
                    lane.apply_engram(gate, &rows)?;
                }
                Ok(true)
            }
        }
    }
    /// Preflight engram histories before any device commit. Publication failure
    /// revokes participating requests; the enclosing scheduler still owns dSpark.
    pub fn commit(
        &mut self,
        batch: &mut RequestBatch,
        execution: &mut BackboneExecution<'_, '_>,
        accepted: &[u32],
    ) -> Result<()> {
        self.validate(batch)?;
        let counts = accepted.iter().map(|&n| n as usize).collect::<Vec<_>>();
        let histories = batch
            .leases
            .iter()
            .map(|&l| Ok(&self.request(l)?.history))
            .collect::<Result<Vec<_>>>()?;
        batch.engram.validate_commit(&histories, &counts)?;
        let result = (|| -> Result<()> {
            execution.commit(&mut self.cache, &batch.cache, accepted)?;
            let mut histories = self
                .slots
                .iter_mut()
                .flatten()
                .filter_map(|r| {
                    batch
                        .leases
                        .iter()
                        .position(|&l| l == r.lease)
                        .map(|i| (i, &mut r.history))
                })
                .collect::<Vec<_>>();
            histories.sort_by_key(|(i, _)| *i);
            let mut histories = histories.into_iter().map(|(_, h)| h).collect::<Vec<_>>();
            batch.engram.commit(&mut histories, &counts)
        })();
        batch.cancel();
        if let Err(error) = result {
            for &lease in &batch.leases {
                if let Err(cleanup) = self.release(lease) {
                    tracing::error!(%cleanup,"releasing failed request transaction");
                }
            }
            return Err(error);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v41_backbone_execution::CacheProducerWeights;
    #[test]
    fn real_request_prefetch_upload_cancel_and_failed_commit() -> Result<()> {
        let Some(path) = std::env::var_os("DS41RT_REQUESTS_LIBRARY") else {
            eprintln!("skip request integration GPU test: DS41RT_REQUESTS_LIBRARY unset");
            return Ok(());
        };
        let model = std::path::PathBuf::from(
            std::env::var_os("DS41RT_REQUESTS_MODEL").context("DS41RT_REQUESTS_MODEL required")?,
        );
        let lib = unsafe { NativeLibrary::load(path)? };
        let catalog =
            ds41rt_loader::read_official_v41_catalog(ds41rt_loader::OFFICIAL_V41_MODEL_ID, &model)?;
        let map = ds41rt_loader::EngramTokenMap::from_file(&model.join("tokenizer.json"))?;
        let pipeline = unsafe { EngramPipeline::new(&catalog, map, 80, 2, 8 * 1024 * 1024)? };
        let mut requests = Requests::new(
            &lib,
            pipeline,
            16,
            [16; 4],
            BackboneCache::device_bytes(16, [16; 4])?,
        )?;
        let leases = (0..16)
            .map(|slot| requests.admit(slot, 700 + slot as u64))
            .collect::<Result<Vec<_>>>()?;
        let tokens = [17, 29, 31, 47, 61];
        let mask = [false, false, true, false, false];
        let input = leases
            .iter()
            .map(|&lease| RequestTokens {
                lease,
                tokens: &tokens,
                image_mask: Some(&mask),
                kind: ExpertV2SourceKind::Prefill,
            })
            .collect::<Vec<_>>();
        let mut batch = requests.prepare(&input)?;
        requests.validate(&batch)?;
        assert_eq!(batch.tokens.len(), 80);
        assert_eq!(
            batch.cache.positions(),
            (0..16).flat_map(|_| 0..5).collect::<Vec<_>>()
        );
        let mut cancelled = requests.prepare(&input)?;
        cancelled.cancel();
        assert!(requests.validate(&cancelled).is_err());
        assert!(cancelled.cache().is_err());
        let mut upload = EngramDeviceRows::new(&lib, 80, EngramDeviceRows::device_bytes(80)?)?;
        for layer in 0..2 {
            let histories = batch
                .leases
                .iter()
                .map(|&l| Ok(&requests.request(l)?.history))
                .collect::<Result<Vec<_>>>()?;
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
            loop {
                match upload.poll_wave(&requests.pipeline, &mut batch.engram, &histories, layer)? {
                    EngramUploadPoll::Ready(view) => {
                        assert_eq!(view.rows, 80);
                        assert_eq!(view.layer_index, layer);
                        let mut text_mask = vec![0; 80];
                        lib.copy_d2h(&mut text_mask, view.text_mask)?;
                        assert_eq!(
                            text_mask,
                            batch.image_mask.iter().map(|&m| 1 - m).collect::<Vec<_>>()
                        );
                        eprintln!("PASS request batch layer={layer} mapped prefetch/gather/upload and image barriers");
                        break;
                    }
                    EngramUploadPoll::Cancelled => {
                        anyhow::bail!("live batch unexpectedly cancelled")
                    }
                    EngramUploadPoll::Pending => {
                        ensure!(
                            std::time::Instant::now() < deadline,
                            "engram gather timed out"
                        );
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    }
                }
            }
        }
        let weights = CacheProducerWeights::load(
            &lib,
            &catalog,
            CacheProducerWeights::device_bytes(&lib, &catalog)?,
            1024 * 1024,
        )?;
        let mut execution =
            BackboneExecution::new(&weights, 80, BackboneExecution::workspace_bytes(&lib, 80)?)?;
        // Acceptance preflight must preserve a usable batch on invalid counts.
        assert!(requests
            .commit(&mut batch, &mut execution, &[6; 16])
            .is_err());
        requests.validate(&batch)?;
        // An incomplete model pass must not publish engram-only acceptance.
        assert!(requests
            .commit(&mut batch, &mut execution, &[2; 16])
            .is_err());
        assert!(requests.validate(&batch).is_err());
        for &lease in &leases {
            assert!(requests.request(lease).is_err());
        }
        for slot in 0..16 {
            let lease = requests.admit(slot, 700 + slot as u64)?;
            assert_ne!(lease, leases[slot]);
            assert_eq!(requests.request(lease)?.history.position(), 0);
            assert_eq!(requests.cache.committed_end(lease)?, 0);
            requests.release(lease)?;
        }
        eprintln!("PASS 16 request owners: cancelled batch rejected; incomplete commit revoked cache/engram admission; fresh generations recovered");
        Ok(())
    }
}
