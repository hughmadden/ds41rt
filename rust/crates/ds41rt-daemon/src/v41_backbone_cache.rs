//! One request lease spans every backbone window and compressed source.
use crate::v41_backbone_router::ExpertRow;
use crate::v41_compressor::{
    CompressorChunk, CompressorLease, CompressorState, CompressorWave, IndexProposal,
};
use crate::v41_index_selection::SelectionRequest;
use crate::v41_sparse_attention::AttentionRequest;
use crate::v41_window::{WindowChunk, WindowLease, WindowProposal, WindowState, WindowWave};
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::NativeLibrary;
use ds41rt_transport::ExpertV2SourceKind;
use std::sync::atomic::{AtomicU64, Ordering};

mod ced;
use ced::CachePhase;
pub(crate) use ced::CacheStage;

const SOURCES: [usize; 4] = [2, 8, 14, 20];
static NEXT_BATCH: AtomicU64 = AtomicU64::new(1);
static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CacheLease {
    owner: u64,
    slot: usize,
    generation: u64,
}
struct Request {
    id: u64,
    version: u64,
    end: u64,
    phase: CachePhase,
    windows: [WindowLease; 40],
    sources: [CompressorLease; 4],
}
#[derive(Clone, Copy)]
pub(crate) struct CacheWork {
    pub lease: CacheLease,
    pub tokens: u32,
    pub kind: ExpertV2SourceKind,
}
struct BatchRequest {
    work: CacheWork,
    id: u64,
    version: u64,
    position: u64,
    windows: [WindowLease; 40],
    sources: [CompressorLease; 4],
}
/// Owned metadata can survive producer execution. Every cache access/commit
/// revalidates it against the originating bank and live request versions.
pub(crate) struct CacheBatch {
    stage: CacheStage,
    replay_snapshot: Option<u64>,
    identity: u64,
    owner: u64,
    requests: Vec<BatchRequest>,
}
impl CacheBatch {
    pub fn stage(&self) -> CacheStage { self.stage }
    pub fn identity(&self) -> u64 {
        self.identity
    }
    pub fn positions(&self) -> Vec<u64> {
        self.requests
            .iter()
            .flat_map(|r| r.position..r.position + u64::from(r.work.tokens))
            .collect()
    }
    pub fn expert_rows(&self) -> Vec<ExpertRow> {
        self.requests
            .iter()
            .flat_map(|r| {
                (0..r.work.tokens).map(move |i| ExpertRow {
                    request_id: r.id,
                    position: r.position + u64::from(i),
                    kind: r.work.kind,
                })
            })
            .collect()
    }
    pub fn window_chunks(&self, layer: usize) -> Result<Vec<WindowChunk>> {
        ensure!(self.stage.windows().contains(&layer), "invalid batch window layer for phase");
        Ok(self
            .requests
            .iter()
            .map(|r| WindowChunk {
                lease: r.windows[layer],
                position: r.position,
                tokens: r.work.tokens,
            })
            .collect())
    }
    pub fn source_chunks(&self, layer: usize) -> Result<Vec<CompressorChunk>> {
        ensure!(self.stage.source_count() != 0, "decoder replay cannot produce global sources");
        let source = SOURCES
            .iter()
            .position(|&l| l == layer)
            .context("invalid batch source layer")?;
        Ok(self
            .requests
            .iter()
            .map(|r| CompressorChunk {
                lease: r.sources[source],
                position: r.position,
                tokens: r.work.tokens,
            })
            .collect())
    }
}

/// Borrows the bank and completed producers for the whole attention use. The
/// request vectors are built from one validated batch, never independently zipped
/// scheduler arrays. Drop all views before advancing or releasing cache history.
pub(crate) struct CacheAttention<'a> {
    windows: Vec<WindowProposal<'a>>,
    sources: Vec<IndexProposal<'a>>,
    positions: Vec<Vec<u64>>,
}
impl CacheAttention<'_> {
    pub fn attention_requests(&self) -> Vec<AttentionRequest<'_>> {
        self.windows
            .iter()
            .enumerate()
            .map(|(i, window)| AttentionRequest {
                window,
                source: self.sources.get(i),
                positions: &self.positions[i],
            })
            .collect()
    }
    pub fn selection_requests(&self) -> Result<Vec<SelectionRequest<'_>>> {
        ensure!(
            self.sources.len() == self.windows.len(),
            "window-only layers have no index selection"
        );
        Ok(self
            .sources
            .iter()
            .zip(&self.positions)
            .map(|(proposal, positions)| SelectionRequest {
                proposal,
                positions,
            })
            .collect())
    }
}

pub(crate) struct BackboneCache<'a> {
    windows: Vec<WindowState<'a>>,
    sources: Vec<CompressorState<'a>>,
    requests: Vec<Option<Request>>,
    generations: Vec<u64>,
    owner: u64,
    poisoned: bool,
}
impl<'a> BackboneCache<'a> {
    pub fn pages_for_context(slots: usize, context: usize) -> Result<[usize; 4]> {
        let mut pages = [0; 4];
        for (i, layer) in SOURCES.into_iter().enumerate() {
            pages[i] = CompressorState::pages_for_context(layer, slots, context)?;
        }
        Ok(pages)
    }

    pub fn device_bytes(slots: usize, source_pages: [usize; 4]) -> Result<usize> {
        let mut total = 40 * WindowState::device_bytes(0, slots)?;
        for (layer, pages) in SOURCES.into_iter().zip(source_pages) {
            total = total
                .checked_add(CompressorState::device_bytes(layer, slots, pages)?)
                .context("backbone cache budget overflow")?;
        }
        Ok(total)
    }
    pub fn new(
        library: &'a NativeLibrary,
        slots: usize,
        source_pages: [usize; 4],
        budget: usize,
    ) -> Result<Self> {
        ensure!(
            Self::device_bytes(slots, source_pages)? <= budget,
            "backbone cache exceeds budget"
        );
        let windows = (0..40)
            .map(|layer| {
                WindowState::new(
                    library,
                    layer,
                    slots,
                    WindowState::device_bytes(layer, slots)?,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let sources = SOURCES
            .into_iter()
            .zip(source_pages)
            .map(|(layer, pages)| {
                CompressorState::new(
                    library,
                    layer,
                    slots,
                    pages,
                    CompressorState::device_bytes(layer, slots, pages)?,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let owner = NEXT_OWNER
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| anyhow::anyhow!("backbone cache IDs exhausted"))?;
        Ok(Self {
            windows,
            sources,
            requests: (0..slots).map(|_| None).collect(),
            generations: vec![0; slots],
            owner,
            poisoned: false,
        })
    }
    fn healthy(&self) -> Result<()> {
        ensure!(
            !self.poisoned,
            "backbone cache must be recreated after initialization or cleanup failure"
        );
        Ok(())
    }
    fn request(&self, lease: CacheLease) -> Result<&Request> {
        self.healthy()?;
        ensure!(
            lease.owner == self.owner
                && lease.slot < self.requests.len()
                && self.generations[lease.slot] == lease.generation,
            "foreign or stale backbone cache lease"
        );
        self.requests[lease.slot]
            .as_ref()
            .context("backbone cache request released")
    }
    pub fn begin_request(&mut self, slot: usize, id: u64) -> Result<CacheLease> {
        self.healthy()?;
        ensure!(
            slot < self.requests.len() && self.requests[slot].is_none(),
            "backbone cache slot unavailable"
        );
        ensure!(
            !self.requests.iter().flatten().any(|r| r.id == id),
            "duplicate backbone cache request"
        );
        let generation = self.generations[slot]
            .checked_add(1)
            .context("cache generation exhausted")?;
        let mut windows = Vec::with_capacity(40);
        let mut sources = Vec::with_capacity(4);
        let acquired = (|| -> Result<()> {
            for state in &mut self.windows {
                windows.push(state.begin_request(slot, id)?);
            }
            for state in &mut self.sources {
                sources.push(state.begin_request(slot, id)?);
            }
            Ok(())
        })();
        if let Err(error) = acquired {
            // A failed device initialization can leave unknown component state.
            // Revoke all acquired leases and require reconstruction of the bank.
            self.poisoned = true;
            for (state, lease) in self.windows.iter_mut().zip(windows) {
                let _ = state.release(lease);
            }
            for (state, lease) in self.sources.iter_mut().zip(sources) {
                let _ = state.release(lease);
            }
            return Err(error);
        }
        self.generations[slot] = generation;
        self.requests[slot] = Some(Request {
            id,
            version: 0,
            end: 0,
            phase: CachePhase::Full,
            windows: windows.try_into().ok().expect("40 windows"),
            sources: sources.try_into().ok().expect("four sources"),
        });
        Ok(CacheLease {
            owner: self.owner,
            slot,
            generation,
        })
    }
    pub fn request_id(&self, lease: CacheLease) -> Result<u64> {
        Ok(self.request(lease)?.id)
    }
    pub fn committed_end(&self, lease: CacheLease) -> Result<u64> {
        let r = self.request(lease)?;
        for (layer, (state, &l)) in self.windows.iter().zip(&r.windows).enumerate() {
            ensure!(
                state.request_id(l)? == r.id && state.end(l)? == r.phase.window_end(layer, r.end),
                "window request history differs"
            );
        }
        for (state, &l) in self.sources.iter().zip(&r.sources) {
            ensure!(
                state.request_id(l)? == r.id && state.committed_end(l)? == r.end,
                "source request history differs"
            );
        }
        Ok(r.end)
    }
    pub fn plan(&self, work: &[CacheWork]) -> Result<CacheBatch> {
        self.plan_stage(work, false)
    }
    pub fn plan_replay(&self, work: &[CacheWork]) -> Result<CacheBatch> {
        self.plan_stage(work, true)
    }
    fn plan_stage(&self, work: &[CacheWork], replay: bool) -> Result<CacheBatch> {
        self.healthy()?;
        ensure!(
            !work.is_empty() && work.len() <= 16,
            "invalid cache batch request count"
        );
        let mut stage = None;
        let mut rows = 0usize;
        let mut requests = Vec::with_capacity(work.len());
        for (i, &item) in work.iter().enumerate() {
            ensure!(
                item.tokens > 0 && !work[..i].iter().any(|w| w.lease == item.lease),
                "empty or duplicate cache batch request"
            );
            rows = rows
                .checked_add(item.tokens as usize)
                .context("cache batch rows overflow")?;
            ensure!(rows <= 4096, "cache batch exceeds lane capacity");
            self.committed_end(item.lease)?;
            let live = self.request(item.lease)?;
            let current = live.phase.stage();
            ensure!(current == CacheStage::Full || item.kind == ExpertV2SourceKind::Prefill,
                "CED phase requires prefill work");
            ensure!((current == CacheStage::Replay) == replay, "wrong cache planning phase");
            ensure!(stage.is_none_or(|s| s == current), "mixed cache phases in batch");
            stage = Some(current);
            let position = live.phase.position(live.end);
            live.phase.validate_tokens(position, item.tokens)?;
            ensure!(
                position
                    .checked_add(u64::from(item.tokens))
                    .is_some_and(|end| end <= 1048576),
                "cache batch exceeds model context"
            );
            let r = self.request(item.lease)?;
            requests.push(BatchRequest {
                work: item,
                id: r.id,
                version: r.version,
                position,
                windows: r.windows,
                sources: r.sources,
            });
        }
        let identity = NEXT_BATCH
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| anyhow::anyhow!("cache batch IDs exhausted"))?;
        Ok(CacheBatch {
            stage: stage.context("empty cache batch")?,
            replay_snapshot: if replay { Some(crate::v41_compressor::reserve_source_snapshot()?) } else { None },
            identity,
            owner: self.owner,
            requests,
        })
    }
    pub fn validate_batch(&self, batch: &CacheBatch) -> Result<()> {
        self.healthy()?;
        ensure!(batch.owner == self.owner, "foreign backbone cache batch");
        for r in &batch.requests {
            self.committed_end(r.work.lease)?;
            let live = self.request(r.work.lease)?;
            ensure!(
                live.id == r.id
                    && live.version == r.version
                    && live.phase.stage() == batch.stage
                    && live.phase.position(live.end) == r.position,
                "stale backbone cache batch"
            );
        }
        Ok(())
    }
    pub fn window(&self, batch: &CacheBatch, layer: usize) -> Result<&WindowState<'a>> {
        self.validate_batch(batch)?;
        ensure!(batch.stage.windows().contains(&layer), "window outside cache phase");
        self.windows
            .get(layer)
            .context("invalid backbone window layer")
    }
    pub fn source(&self, batch: &CacheBatch, layer: usize) -> Result<&CompressorState<'a>> {
        self.validate_batch(batch)?;
        let source = SOURCES
            .iter()
            .position(|&l| l == layer)
            .context("invalid backbone source layer")?;
        Ok(&self.sources[source])
    }
    /// # Safety
    /// Query rows correspond to this admitted batch and all query writes have
    /// completed. No external writes race query, producer or persistent cache.
    pub unsafe fn produce_window(
        &self,
        batch: &CacheBatch,
        query: &crate::v41_attention_query::AttentionQueryOutput<'_>,
        wave: &mut WindowWave<'_, '_>,
    ) -> Result<()> {
        let state = self.window(batch, query.layer)?;
        let chunks = batch.window_chunks(query.layer)?;
        unsafe {
            wave.execute_query(state, &chunks, query)?;
        }
        Ok(())
    }
    /// # Safety
    /// Same query/batch association and completed-producer contract as
    /// produce_window. Only the four source layers may produce compressed KV.
    pub unsafe fn produce_source(
        &self,
        batch: &CacheBatch,
        query: &crate::v41_attention_query::AttentionQueryOutput<'_>,
        wave: &mut CompressorWave<'_, '_>,
    ) -> Result<()> {
        let state = self.source(batch, query.layer)?;
        let chunks = batch.source_chunks(query.layer)?;
        unsafe {
            wave.execute_query(state, &chunks, query)?;
        }
        Ok(())
    }
    /// Bind the matching layer's completed window and nearest compressed source
    /// to this batch. Later reindex layers continue using source 20's proposal.
    pub fn attention<'s>(
        &'s self,
        batch: &CacheBatch,
        layer: usize,
        window: &'s WindowWave<'_, '_>,
        source: Option<&'s CompressorWave<'_, '_>>,
    ) -> Result<CacheAttention<'s>> {
        let state = self.window(batch, layer)?;
        let chunks = batch.window_chunks(layer)?;
        window.validate_batch(state, &chunks)?;
        let source_layer = SOURCES.iter().copied().rev().find(|&n| n <= layer);
        ensure!(
            source.is_some() == (source_layer.is_some() && batch.stage != CacheStage::Replay),
            "attention batch source presence differs"
        );
        let mut sources = Vec::new();
        if batch.stage == CacheStage::Replay {
            ensure!(source_layer == Some(20), "decoder replay global source differs");
            let state = self.source(batch, 20)?;
            let snapshot = batch.replay_snapshot.context("missing decoder source snapshot")?;
            sources = batch.requests.iter().map(|r| state.committed_proposal(
                r.sources[3], r.position..r.position + u64::from(r.work.tokens), snapshot))
                .collect::<Result<Vec<_>>>()?;
        } else if let Some(source_layer) = source_layer {
            let wave = source.context("attention source absent")?;
            let state = self.source(batch, source_layer)?;
            let chunks = batch.source_chunks(source_layer)?;
            wave.validate_batch(state, &chunks)?;
            sources = chunks.iter().map(|c| wave.index_proposal(state, c.lease))
                .collect::<Result<Vec<_>>>()?;
        }
        let windows = chunks
            .iter()
            .map(|c| window.proposal(state, c.lease))
            .collect::<Result<Vec<_>>>()?;
        let positions = chunks
            .iter()
            .map(|c| (c.position..c.position + u64::from(c.tokens)).collect())
            .collect();
        Ok(CacheAttention {
            windows,
            sources,
            positions,
        })
    }
    /// Revoke host leases first, then release every associated device cache. All
    /// proposal/attention consumers must have drained before this mutable call.
    pub fn release(&mut self, leases: &[CacheLease]) -> Result<()> {
        for (i, &lease) in leases.iter().enumerate() {
            self.request(lease)?;
            ensure!(!leases[..i].contains(&lease), "duplicate cache release");
        }
        let removed = leases
            .iter()
            .map(|l| self.requests[l.slot].take().unwrap())
            .collect::<Vec<_>>();
        let mut first_error = None;
        for r in removed {
            for (state, lease) in self.windows.iter_mut().zip(r.windows) {
                // Component commit failure may already have revoked this lease.
                if state.request_id(lease).is_ok() {
                    if let Err(e) = state.release(lease) {
                        first_error.get_or_insert(e);
                    }
                }
            }
            for (state, lease) in self.sources.iter_mut().zip(r.sources) {
                if state.request_id(lease).is_ok() {
                    if let Err(e) = state.release(lease) {
                        first_error.get_or_insert(e);
                    }
                }
            }
        }
        if let Some(e) = first_error {
            self.poisoned = true;
            return Err(e);
        }
        Ok(())
    }
    /// Commit the accepted prefix to this phase’s window/source owners. Any
    /// execution failure revokes all participating requests; partial device
    /// changes are never published as a usable bank history. Engram/dSpark
    /// acceptance remains the enclosing scheduler transaction's responsibility.
    pub fn commit(
        &mut self,
        batch: &CacheBatch,
        windows: &mut [WindowWave<'_, '_>],
        sources: &mut [CompressorWave<'_, '_>],
        accepted: &[u32],
    ) -> Result<()> {
        self.validate_batch(batch)?;
        ensure!(
            windows.len() == 40 && sources.len() == 4 && accepted.len() == batch.requests.len(),
            "backbone commit owner/count differs"
        );
        for (r, &n) in batch.requests.iter().zip(accepted) {
            ensure!(
                n <= r.work.tokens,
                "backbone accepted prefix exceeds proposal"
            );
            r.version
                .checked_add(1)
                .context("backbone cache version exhausted")?;
        }
        for layer in batch.stage.windows() {
            let wave = &windows[layer];
            wave.validate_batch(&self.windows[layer], &batch.window_chunks(layer)?)?;
        }
        for i in 0..batch.stage.source_count() {
            let wave = &sources[i];
            wave.validate_batch(&self.sources[i], &batch.source_chunks(SOURCES[i])?)?;
        }
        let committed = (|| -> Result<()> {
            for layer in batch.stage.windows() {
                windows[layer].commit(&mut self.windows[layer], accepted)?;
            }
            for i in 0..batch.stage.source_count() {
                sources[i].commit(&mut self.sources[i], accepted)?;
            }
            Ok(())
        })();
        if let Err(error) = committed {
            let leases = batch
                .requests
                .iter()
                .map(|r| r.work.lease)
                .collect::<Vec<_>>();
            if let Err(cleanup) = self.release(&leases) {
                tracing::error!(%cleanup,"invalidating failed backbone cache transaction");
            }
            return Err(error);
        }
        for (r, &n) in batch.requests.iter().zip(accepted) {
            let live = self.requests[r.work.lease.slot]
                .as_mut()
                .expect("validated live request");
            live.phase.advance(&mut live.end, r.position + u64::from(n));
            live.version += 1;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_bank_lifecycle_and_batch_isolation() -> Result<()> {
        let Some(path) = std::env::var_os("DS41RT_BACKBONE_CACHE_LIBRARY") else {
            eprintln!("skip GPU cache bank test: DS41RT_BACKBONE_CACHE_LIBRARY unset");
            return Ok(());
        };
        let library = unsafe { NativeLibrary::load(path)? };
        let pages = [1, 2, 3, 4];
        let budget = BackboneCache::device_bytes(16, pages)?;
        assert!(BackboneCache::device_bytes(0, pages).is_err());
        assert!(BackboneCache::device_bytes(17, pages).is_err());
        assert!(BackboneCache::device_bytes(16, [0, 1, 1, 1]).is_err());
        assert!(BackboneCache::device_bytes(16, [1, 1, 1, 65537]).is_err());
        assert!(BackboneCache::new(&library, 16, pages, budget - 1).is_err());
        let mut bank = BackboneCache::new(&library, 16, pages, budget)?;
        let mut other = BackboneCache::new(&library, 16, pages, budget)?;
        let foreign = other.begin_request(0, 100)?;
        for cycle in 0..2 {
            let leases = (0..16)
                .map(|slot| bank.begin_request(slot, 100 + slot as u64))
                .collect::<Result<Vec<_>>>()?;
            assert!(bank.begin_request(16, 999).is_err());
            assert!(bank.begin_request(0, 999).is_err());
            let work = leases
                .iter()
                .enumerate()
                .map(|(i, &lease)| CacheWork {
                    lease,
                    tokens: (i + 1) as u32,
                    kind: ExpertV2SourceKind::Prefill,
                })
                .collect::<Vec<_>>();
            let batch = bank.plan(&work)?;
            assert_ne!(batch.identity(), bank.plan(&work)?.identity());
            bank.validate_batch(&batch)?;
            assert!(other.validate_batch(&batch).is_err());
            assert!(bank.request_id(foreign).is_err());
            assert!(bank.plan(&[]).is_err());
            assert!(bank.plan(&[work[0], work[0]]).is_err());
            assert!(bank
                .plan(&[CacheWork {
                    tokens: 0,
                    ..work[0]
                }])
                .is_err());
            assert!(bank
                .plan(&[CacheWork {
                    tokens: 4097,
                    ..work[0]
                }])
                .is_err());
            assert!(bank
                .plan(&[
                    CacheWork {
                        tokens: 4096,
                        ..work[0]
                    },
                    work[1]
                ])
                .is_err());
            assert_eq!(
                bank.plan(&[CacheWork {
                    tokens: 4096,
                    ..work[0]
                }])?
                .positions()
                .len(),
                4096
            );
            let positions = work
                .iter()
                .flat_map(|w| 0..u64::from(w.tokens))
                .collect::<Vec<_>>();
            assert_eq!(batch.positions(), positions);
            let rows = batch.expert_rows();
            assert_eq!(rows.len(), 136);
            let mut offset = 0;
            for (i, w) in work.iter().enumerate() {
                assert_eq!(bank.request_id(w.lease)?, 100 + i as u64);
                assert_eq!(bank.committed_end(w.lease)?, 0);
                for row in &rows[offset..offset + w.tokens as usize] {
                    assert_eq!(row.request_id, 100 + i as u64);
                    assert_eq!(row.kind, ExpertV2SourceKind::Prefill);
                }
                offset += w.tokens as usize;
            }
            for layer in 0..40 {
                let state = bank.window(&batch, layer)?;
                for (i, chunk) in batch.window_chunks(layer)?.iter().enumerate() {
                    assert_eq!(state.request_id(chunk.lease)?, 100 + i as u64);
                    assert_eq!(state.end(chunk.lease)?, chunk.position);
                    assert_eq!(chunk.tokens, (i + 1) as u32);
                }
            }
            for layer in SOURCES {
                let state = bank.source(&batch, layer)?;
                for (i, chunk) in batch.source_chunks(layer)?.iter().enumerate() {
                    assert_eq!(state.request_id(chunk.lease)?, 100 + i as u64);
                    assert_eq!(state.committed_end(chunk.lease)?, chunk.position);
                    assert_eq!(chunk.tokens, (i + 1) as u32);
                }
            }
            assert!(batch.window_chunks(40).is_err());
            assert!(batch.source_chunks(21).is_err());
            assert!(bank.window(&batch, 40).is_err());
            assert!(bank.source(&batch, 21).is_err());
            // All release arguments must validate before any live slot changes.
            assert!(bank.release(&[leases[0], foreign]).is_err());
            assert!(bank.release(&[leases[0], leases[0]]).is_err());
            bank.validate_batch(&batch)?;
            // Commit preflight rejects missing producers without consuming history.
            assert!(bank.commit(&batch, &mut [], &mut [], &[0; 16]).is_err());
            bank.validate_batch(&batch)?;
            bank.release(&leases[..1])?;
            assert!(bank.validate_batch(&batch).is_err());
            assert!(bank.window(&batch, 0).is_err());
            assert!(bank.source(&batch, 2).is_err());
            assert!(bank.begin_request(0, 101).is_err());
            let replacement = bank.begin_request(0, 100)?;
            assert_ne!(replacement, leases[0]);
            assert!(bank.request_id(leases[0]).is_err());
            assert!(bank.plan(&work).is_err());
            assert_eq!(bank.committed_end(replacement)?, 0);
            let surviving = bank.plan(&work[1..])?;
            bank.validate_batch(&surviving)?;
            bank.release(&leases[1..])?;
            bank.release(&[replacement])?;
            assert!(bank.validate_batch(&surviving).is_err());
            eprintln!("cache bank cycle {cycle}: 16 requests, all 44 component leases, batch/release/reuse guards passed");
        }
        other.release(&[foreign])?;
        Ok(())
    }
}

#[cfg(test)]
#[path = "v41_backbone_cache/commit_tests.rs"]
mod commit_tests;

#[cfg(test)]
mod context_geometry_tests {
    use super::BackboneCache;
    #[test]
    fn context_pool_provisions_each_slot_and_compression_ratio() {
        assert_eq!(BackboneCache::pages_for_context(16, 32768).unwrap(), [1024, 1024, 1024, 2048]);
        assert_eq!(BackboneCache::pages_for_context(3, 513).unwrap(), [6, 6, 6, 9]);
        assert_eq!(BackboneCache::pages_for_context(16, 1048576).unwrap(), [32768, 32768, 32768, 65536]);
        assert!(BackboneCache::pages_for_context(0, 32768).is_err());
        assert!(BackboneCache::pages_for_context(17, 32768).is_err());
        assert!(BackboneCache::pages_for_context(1, 0).is_err());
        assert!(BackboneCache::pages_for_context(1, 1048577).is_err());
    }
}
