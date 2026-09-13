//! Cache producers and the complete per-layer backbone execution handoff.
use crate::v41_backbone_cache::{BackboneCache, CacheBatch, CacheStage};
use crate::v41_backbone_lane::{BackboneLane, LaneFfn, PendingLaneFfn};
use crate::v41_backbone_router::ExpertRow;
use crate::v41_compressor::{CompressorWave, CompressorWeights};
use crate::v41_experts::coordinator::{NativeTp4Wave, NativeFfnOutput};
use crate::v41_index_lane::IndexLane;
use crate::v41_tensors::NativeRtxTensors;
use crate::v41_window::{WindowWave, WindowWeights};
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::NativeLibrary;
use ds41rt_loader::OfficialV41Catalog;
const SOURCES: [usize; 4] = [2, 8, 14, 20];
const INDEX: [usize; 8] = [2, 8, 14, 20, 24, 28, 32, 36];

enum PreparedFfn<'l, 'w, 'a> {
    Ready(LaneFfn<'l, 'w, 'a>),
    Pending(PendingLaneFfn<'l, 'w, 'a>),
}
/// Owns a lane borrow, but no cache-bank or index borrow. The unsafe queued
/// preparation contract retains external producers through attention completion.
pub(crate) struct PreparedLayer<'l, 'w, 'a> {
    ffn: PreparedFfn<'l, 'w, 'a>,
    rows: Vec<ExpertRow>,
    batch: u64,
    layer: usize,
    started: std::time::Instant,
    produced_us: u64,
    indexed_us: u64,
    attended_us: u64,
}
pub(crate) struct CompletedLayer<'t> {
    result: NativeFfnOutput<'t>,
    batch: u64,
    layer: usize,
    rows: usize,
    started: std::time::Instant,
    produced_us: u64,
    indexed_us: u64,
    attended_us: u64,
    experts_us: u64,
}
impl PreparedLayer<'_, '_, '_> {
    /// # Safety
    /// The modality mask and placement describe this prepared batch. Poll on
    /// the CUDA owner; no external writes may race the borrowed lane.
    pub async unsafe fn execute<'t>(self, transport: &'t mut NativeTp4Wave<'_>,
        placement: u64, image_mask: &[u8]) -> Result<CompletedLayer<'t>> {
        let (mut ffn, attended_us) = match self.ffn {
            PreparedFfn::Ready(ffn) => (ffn, self.attended_us),
            PreparedFfn::Pending(pending) => {
                let ffn = pending.complete().await?;
                (ffn, self.started.elapsed().as_micros() as u64)
            }
        };
        let result = unsafe { ffn.execute_tp4(transport, placement, image_mask, &self.rows).await? };
        Ok(CompletedLayer { result, batch: self.batch, layer: self.layer, rows: self.rows.len(),
            started: self.started, produced_us: self.produced_us, indexed_us: self.indexed_us,
            attended_us, experts_us: self.started.elapsed().as_micros() as u64 })
    }
}

enum LayerCache<'b, 'a> {
    Ordinary(&'b BackboneCache<'a>),
    Encoder(&'b mut BackboneCache<'a>),
}
impl<'a> LayerCache<'_, 'a> {
    fn bank(&self) -> &BackboneCache<'a> {
        match self { Self::Ordinary(bank) => bank, Self::Encoder(bank) => bank }
    }
}

pub(crate) struct CacheProducerWeights<'a> {
    library: &'a NativeLibrary,
    windows: Vec<WindowWeights<'a>>,
    sources: Vec<CompressorWeights<'a>>,
    sinks: NativeRtxTensors<'a>,
}
impl<'a> CacheProducerWeights<'a> {
    fn sinks() -> Vec<String> {
        (0..40)
            .map(|layer| format!("layers.{layer}.attn.attn_sink"))
            .collect()
    }
    pub fn device_bytes(library: &NativeLibrary, catalog: &OfficialV41Catalog) -> Result<usize> {
        let mut total = NativeRtxTensors::plan(catalog, &Self::sinks())?;
        ensure!(total == 40 * 256, "unexpected backbone attention sink size");
        for layer in 0..40 {
            total = total
                .checked_add(WindowWeights::device_bytes(library, catalog, layer)?)
                .context("cache producer weights overflow")?;
        }
        for layer in SOURCES {
            total = total
                .checked_add(CompressorWeights::device_bytes(catalog, layer)?)
                .context("cache producer weights overflow")?;
        }
        Ok(total)
    }
    pub fn load(
        library: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        budget: usize,
        staging: usize,
    ) -> Result<Self> {
        ensure!(
            Self::device_bytes(library, catalog)? <= budget,
            "cache producer weights exceed budget"
        );
        let windows = (0..40)
            .map(|layer| {
                WindowWeights::load(
                    library,
                    catalog,
                    layer,
                    WindowWeights::device_bytes(library, catalog, layer)?,
                    staging,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let sources = SOURCES
            .into_iter()
            .map(|layer| {
                CompressorWeights::load(
                    library,
                    catalog,
                    layer,
                    CompressorWeights::device_bytes(catalog, layer)?,
                    staging,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let names = Self::sinks();
        let sinks = NativeRtxTensors::load(
            library,
            catalog,
            &names,
            NativeRtxTensors::plan(catalog, &names)?,
            staging,
        )?;
        Ok(Self {
            library,
            windows,
            sources,
            sinks,
        })
    }
}

#[derive(Default)]
struct PassProgress {
    stage: CacheStage,
    source_ready: bool,
    batch: Option<u64>,
    next: usize,
    invalid: bool,
}
impl PassProgress {
    fn begin(&mut self, batch: u64, layer: usize) -> Result<()> {
        let valid = !std::mem::replace(&mut self.invalid, true);
        ensure!(
            valid && layer == self.next && self.stage.windows().contains(&layer) && self.batch.is_none_or(|id| id == batch),
            "backbone pass batch or layer differs; restart required"
        );
        self.batch = Some(batch);
        Ok(())
    }
    fn for_stage(stage: CacheStage) -> Self {
        Self { stage, next: stage.windows().start, ..Self::default() }
    }
    fn begin_decoder_source(&mut self, batch: u64) -> Result<()> {
        let valid = !std::mem::replace(&mut self.invalid, true);
        ensure!(valid && self.stage == CacheStage::Encoder && self.next == 20
            && self.batch == Some(batch) && !self.source_ready,
            "decoder source requires this batch's complete encoder pass");
        Ok(())
    }
    fn finish_decoder_source(&mut self) {
        self.source_ready = true;
        self.invalid = false;
    }
    fn finish(&mut self) {
        self.next += 1;
        self.invalid = false;
    }
    fn commit(&mut self, batch: u64) -> Result<()> {
        let valid = !std::mem::replace(&mut self.invalid, true);
        ensure!(
            valid && self.next == self.stage.windows().end && self.batch == Some(batch)
                && (self.stage != CacheStage::Encoder || self.source_ready),
            "cache commit requires this batch's complete execution phase"
        );
        Ok(())
    }
}

/// Every layer retains its private proposal until accepted-prefix commit. Two
/// alternating passes need independent producer storage and progress owners.
pub(crate) struct BackboneExecution<'w, 'a> {
    weights: &'w CacheProducerWeights<'a>,
    windows: Vec<WindowWave<'w, 'a>>,
    sources: Vec<CompressorWave<'w, 'a>>,
    progress: PassProgress,
    queued_production: Option<(u64, usize, std::time::Instant)>,
}
/// Scratch producers retain their wave owner; the caller retains the matching
/// query and admitted cache slots through completion or drained cancellation.
pub(crate) struct PendingProduction<'p, 'w, 'a> {
    execution: &'p mut BackboneExecution<'w, 'a>,
    batch: u64,
    layer: usize,
    source: Option<usize>,
    window_ready: bool,
    source_ready: bool,
    complete: bool,
    started: std::time::Instant,
}
impl PendingProduction<'_, '_, '_> {
    pub unsafe fn poll(&mut self, bank: &BackboneCache<'_>, batch: &CacheBatch) -> Result<bool> {
        ensure!(!self.complete && batch.identity() == self.batch, "queued production batch differs");
        bank.validate_batch(batch)?;
        if !self.window_ready {
            self.window_ready = unsafe { self.execution.windows[self.layer].poll_query(bank.window(batch, self.layer)?)? };
        }
        if !self.source_ready {
            let source = self.source.unwrap();
            self.source_ready = unsafe { self.execution.sources[source].poll_query(bank.source(batch, self.layer)?)? };
        }
        if self.window_ready && self.source_ready {
            self.execution.queued_production = Some((self.batch, self.layer, self.started));
            self.complete = true;
        }
        Ok(self.complete)
    }
}
impl Drop for PendingProduction<'_, '_, '_> {
    fn drop(&mut self) {
        if !self.complete {
            if let Err(error) = self.execution.windows[self.layer].abort_query() {
                tracing::error!(%error, "draining cancelled window production");
            }
            if let Some(source) = self.source {
                if let Err(error) = self.execution.sources[source].abort_query() {
                    tracing::error!(%error, "draining cancelled compressed production");
                }
            }
            self.execution.queued_production = None;
        }
    }
}
impl<'w, 'a> BackboneExecution<'w, 'a> {
    pub fn workspace_bytes(library: &NativeLibrary, capacity: u32) -> Result<usize> {
        let mut total = WindowWave::device_bytes(library, capacity)?
            .checked_mul(40)
            .context("window workspace budget overflow")?;
        for layer in SOURCES {
            total = total
                .checked_add(CompressorWave::device_bytes(layer, capacity as usize)?)
                .context("source workspace budget overflow")?;
        }
        Ok(total)
    }
    pub fn new(
        weights: &'w CacheProducerWeights<'a>,
        capacity: u32,
        budget: usize,
    ) -> Result<Self> {
        ensure!(
            Self::workspace_bytes(weights.library, capacity)? <= budget,
            "cache producer workspace exceeds budget"
        );
        ensure!(
            weights.windows.len() == 40 && weights.sources.len() == 4,
            "cache producer weight owners incomplete"
        );
        let windows = weights
            .windows
            .iter()
            .map(|w| {
                w.wave(
                    capacity,
                    WindowWave::device_bytes(weights.library, capacity)?,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let sources = weights
            .sources
            .iter()
            .zip(SOURCES)
            .map(|(w, layer)| {
                w.wave(
                    capacity as usize,
                    CompressorWave::device_bytes(layer, capacity as usize)?,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            weights,
            windows,
            sources,
            progress: PassProgress::default(),
            queued_production: None,
        })
    }
    /// Discard pass progress only after all consumers finish. The caller also
    /// restarts the backbone/index lanes and initializes the next query batch.
    pub fn restart(&mut self) {
        self.queued_production = None;
        self.progress = PassProgress::default();
    }

    pub fn restart_for(&mut self, stage: CacheStage) {
        self.queued_production = None;
        self.progress = PassProgress::for_stage(stage);
    }
    /// Publish source 20 after a complete reserved encoder chunk.
    /// # Safety
    /// The layer-20 query belongs to this chunk and all earlier readers drained.
    pub unsafe fn publish_encoder_boundary(&mut self, bank: &mut BackboneCache<'_>,
        batch: &CacheBatch, lane: &BackboneLane<'_, '_>) -> Result<()> {
        self.progress.begin_decoder_source(batch.identity())?;
        ensure!(batch.is_reserved() && batch.stage() == CacheStage::Encoder,
            "encoder boundary publication requires reserved prefill");
        bank.validate_batch(batch)?;
        let query = lane.query_output()?;
        ensure!(query.layer == 20, "encoder boundary projection layer differs");
        unsafe { bank.produce_source(batch, &query, &mut self.sources[3])?; }
        bank.publish_encoder_source(batch, 20, &mut self.sources[3])?;
        self.progress.finish_decoder_source();
        Ok(())
    }
    /// Produce global source 20 from the prepared encoder boundary, without
    /// running decoder attention, routing, shared FFN or remote experts.
    /// # Safety
    /// The completed layer-20 query preparation belongs to this encoder batch;
    /// its source input and all device owners remain exclusive through completion.
    pub unsafe fn produce_decoder_source(&mut self, bank: &BackboneCache<'_>,
        batch: &CacheBatch, lane: &BackboneLane<'_, '_>) -> Result<()> {
        self.progress.begin_decoder_source(batch.identity())?;
        ensure!(batch.stage() == CacheStage::Encoder, "decoder source batch phase differs");
        bank.validate_batch(batch)?;
        let query = lane.query_output()?;
        ensure!(query.layer == 20, "decoder source requires layer-20 projection input");
        unsafe { bank.produce_source(batch, &query, &mut self.sources[3])?; }
        self.progress.finish_decoder_source();
        Ok(())
    }

    /// # Safety
    /// Keep the lane query and this batch's admitted cache slots immutable and
    /// alive until the returned producer completes or drains on drop.
    pub unsafe fn enqueue_production(&mut self, bank: &BackboneCache<'_>, batch: &CacheBatch,
        lane: &BackboneLane<'_, '_>) -> Result<PendingProduction<'_, 'w, 'a>> {
        ensure!(self.queued_production.is_none(), "prior queued production not consumed");
        bank.validate_batch(batch)?;
        let query = lane.query_output()?;
        let layer = query.layer;
        ensure!(layer == self.progress.next && batch.stage() == self.progress.stage,
            "queued production layer or stage differs");
        let source = SOURCES.iter().position(|&l| l == layer).filter(|_| !batch.stage().reuses_sources());
        let pending = PendingProduction { execution: self, batch: batch.identity(), layer, source,
            window_ready: false, source_ready: source.is_none(), complete: false,
            started: std::time::Instant::now() };
        unsafe {
            pending.execution.windows[layer].enqueue_query(bank.window(batch, layer)?,
                &batch.window_chunks(layer)?, &query)?;
            if let Some(i) = source {
                pending.execution.sources[i].enqueue_query(bank.source(batch, layer)?,
                    &batch.source_chunks(layer)?, &query)?;
            }
        }
        Ok(pending)
    }
    /// Execute one already-prepared layer through completed FFN/mHC output.
    /// # Safety
    /// Lane query rows, modality mask and cache batch identify the same requests.
    /// All producers have completed; no external writes race any passed owner.
    /// The caller handles engram, decoder taps and next-layer preparation between
    /// calls. Cancellation after polling requires restarting these lane owners.
    pub async unsafe fn execute_layer(
        &mut self,
        bank: &BackboneCache<'_>,
        batch: &CacheBatch,
        lane: &mut BackboneLane<'_, '_>,
        index: &mut IndexLane<'_, '_>,
        transport: &mut NativeTp4Wave<'_>,
        placement: u64,
        image_mask: &[u8],
    ) -> Result<()> {
        let prepared = unsafe { self.prepare_layer(bank, batch, lane, index)? };
        let completed = unsafe { prepared.execute(transport, placement, image_mask).await? };
        unsafe { self.complete_layer(batch, lane, completed) }
    }
    /// Finish all cache production, indexing and attention without holding the
    /// bank across remote execution. Dropping prepared work leaves progress
    /// invalid until restart, just as cancelling ordinary layer execution does.
    /// # Safety
    /// Same completed-query, cache ownership and device contract as execute_layer.
    pub unsafe fn prepare_layer<'l, 'lw, 'la>(&mut self, bank: &BackboneCache<'_>,
        batch: &CacheBatch, lane: &'l mut BackboneLane<'lw, 'la>,
        index: &mut IndexLane<'_, '_>) -> Result<PreparedLayer<'l, 'lw, 'la>> {
        unsafe { self.prepare_layer_with_cache(LayerCache::Ordinary(bank), batch, lane, index, false) }
    }
    /// # Safety
    /// Same contract as prepare_layer, extended through returned work completion:
    /// preserve batch cache slots, this execution's producers and index storage.
    /// Cancellation must drop the returned work before releasing those owners.
    pub unsafe fn prepare_layer_cooperative<'l, 'lw, 'la>(&mut self, bank: &BackboneCache<'_>,
        batch: &CacheBatch, lane: &'l mut BackboneLane<'lw, 'la>,
        index: &mut IndexLane<'_, '_>) -> Result<PreparedLayer<'l, 'lw, 'la>> {
        unsafe { self.prepare_layer_with_cache(LayerCache::Ordinary(bank), batch, lane, index, true) }
    }
    /// Prepare a reserved encoder layer and publish source/window KV before
    /// returning its FFN owner. Published sources feed all later index consumers.
    /// # Safety
    /// Same query/owner contract as prepare_layer; earlier readers have drained.
    pub unsafe fn prepare_encoder_layer<'l, 'lw, 'la>(&mut self, bank: &mut BackboneCache<'_>,
        batch: &CacheBatch, lane: &'l mut BackboneLane<'lw, 'la>,
        index: &mut IndexLane<'_, '_>) -> Result<PreparedLayer<'l, 'lw, 'la>> {
        ensure!(batch.is_reserved() && batch.stage() == CacheStage::Encoder,
            "published execution requires a reserved encoder batch");
        unsafe { self.prepare_layer_with_cache(LayerCache::Encoder(bank), batch, lane, index, false) }
    }
    unsafe fn prepare_layer_with_cache<'l, 'lw, 'la>(&mut self, mut bank: LayerCache<'_, '_>,
        batch: &CacheBatch, lane: &'l mut BackboneLane<'lw, 'la>,
        index: &mut IndexLane<'_, '_>, cooperative: bool) -> Result<PreparedLayer<'l, 'lw, 'la>> {
        let publishing = matches!(&bank, LayerCache::Encoder(_));
        let queued_production = self.queued_production.take();
        let timing = queued_production.map_or_else(std::time::Instant::now, |p| p.2);
        // Invalidate even if obtaining the completed query or bank check fails.
        let layer = self.progress.next;
        self.progress.begin(batch.identity(), layer)?;
        ensure!(batch.stage() == self.progress.stage, "backbone execution/cache phase differs");
        bank.bank().validate_batch(batch)?;
        let query = lane.query_output()?;
        ensure!(
            query.layer == layer,
            "backbone query layer differs from pass"
        );
        if let Some((id, produced_layer, _)) = queued_production {
            ensure!(id == batch.identity() && produced_layer == layer && !publishing,
                "queued cache production identity differs");
        } else {
            unsafe { bank.bank().produce_window(batch, &query, &mut self.windows[layer])?; }
        }
        if let Some(i) = SOURCES
            .iter()
            .position(|&l| l == layer)
            .filter(|_| !batch.stage().reuses_sources())
        {
            if queued_production.is_none() {
                unsafe { bank.bank().produce_source(batch, &query, &mut self.sources[i])?; }
            }
            if let LayerCache::Encoder(bank) = &mut bank {
                bank.publish_encoder_source(batch, layer, &mut self.sources[i])?;
            }
        }
        let produced_us = timing.elapsed().as_micros() as u64;
        let source = SOURCES
            .iter()
            .rposition(|&l| l <= layer)
            .filter(|_| !batch.stage().reuses_sources() && !publishing)
            .map(|i| &self.sources[i]);
        let cache = bank.bank().attention(batch, layer, &self.windows[layer], source)?;
        if INDEX.contains(&layer) {
            unsafe {
                lane.select_index(index, &cache)?;
            }
        }
        let indexed_us = timing.elapsed().as_micros() as u64;
        let sink = self
            .weights
            .sinks
            .get(&format!("layers.{layer}.attn.attn_sink"))?;
        let rows = batch.expert_rows();
        let ffn = if cooperative {
            PreparedFfn::Pending(unsafe { lane.enqueue_attention_indexed_ffn(sink, &cache, index)? })
        } else {
            PreparedFfn::Ready(unsafe { lane.attention_indexed_ffn(sink, &cache, index)? })
        };
        drop(cache);
        if let LayerCache::Encoder(bank) = &mut bank {
            bank.publish_encoder_window(batch, layer, &mut self.windows[layer])?;
        }
        Ok(PreparedLayer { ffn, rows, batch: batch.identity(), layer, started: timing,
            produced_us, indexed_us, attended_us: timing.elapsed().as_micros() as u64 })
    }
    /// # Safety
    /// The completed transport output and lane belong to this prepared layer;
    /// no external writes race the final mHC operation.
    pub unsafe fn complete_layer(&mut self, batch: &CacheBatch,
        lane: &mut BackboneLane<'_, '_>, completed: CompletedLayer<'_>) -> Result<()> {
        ensure!(self.progress.invalid && self.progress.batch == Some(completed.batch)
            && batch.identity() == completed.batch && self.progress.next == completed.layer
            && self.progress.stage == batch.stage(), "completed backbone layer identity differs");
        unsafe { lane.finish_ffn(completed.result.binding(), completed.result.values)?; }
        tracing::debug!(target: "ds41rt::timing", layer=completed.layer, rows=completed.rows,
            produced_us=completed.produced_us, index_us=completed.indexed_us-completed.produced_us,
            attention_us=completed.attended_us-completed.indexed_us,
            experts_us=completed.experts_us-completed.attended_us,
            finish_us=completed.started.elapsed().as_micros() as u64-completed.experts_us, "target layer");
        self.progress.finish();
        Ok(())
    }
    /// Commit after the same batch completed its full or CED phase. The caller
    /// determines acceptance after target-head/sampling/verification and includes
    /// engram/dSpark history in the enclosing scheduler transaction.
    pub unsafe fn enqueue_cache_commit(&mut self, bank: &BackboneCache<'_>,
        batch: &CacheBatch, accepted: &[u32]) -> Result<()> {
        unsafe { bank.enqueue_cache_commit(batch, &mut self.windows, &mut self.sources, accepted) }
    }
    pub fn poll_cache_commit(&self) -> Result<bool> {
        for window in &self.windows { if !window.poll_commit()? { return Ok(false); } }
        for source in &self.sources { if !source.poll_commit()? { return Ok(false); } }
        Ok(true)
    }
    pub fn abort_cache_commit(&mut self, bank: &mut BackboneCache<'_>) -> Result<()> {
        bank.abort_cache_commit(&mut self.windows, &mut self.sources)
    }
    pub fn commit(
        &mut self,
        bank: &mut BackboneCache<'_>,
        batch: &CacheBatch,
        accepted: &[u32],
    ) -> Result<()> {
        self.progress.commit(batch.identity())?;
        ensure!(batch.stage() == self.progress.stage, "backbone commit/cache phase differs");
        bank.commit(batch, &mut self.windows, &mut self.sources, accepted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn official_cache_producer_allocation_plan() -> Result<()> {
        let Some(path) = std::env::var_os("DS41RT_LANE_PLAN_LIBRARY") else {
            eprintln!("skip cache producer planning: DS41RT_LANE_PLAN_LIBRARY unset");
            return Ok(());
        };
        let model = std::env::var_os("DS41RT_LANE_PLAN_MODEL")
            .context("DS41RT_LANE_PLAN_MODEL required")?;
        let library = unsafe { NativeLibrary::load(path)? };
        let catalog = ds41rt_loader::read_official_v41_catalog(
            ds41rt_loader::OFFICIAL_V41_MODEL_ID,
            std::path::Path::new(&model),
        )?;
        let bytes = CacheProducerWeights::device_bytes(&library, &catalog)?;
        assert_eq!(bytes, 145_517_568);
        assert!(
            CacheProducerWeights::load(&library, &catalog, bytes - 1, 1024 * 1024)
                .err()
                .unwrap()
                .to_string()
                .contains("weights exceed budget")
        );
        for capacity in [1, 80, 4096] {
            let workspace = BackboneExecution::workspace_bytes(&library, capacity)?;
            eprintln!("cache producer capacity={capacity} workspace_bytes={workspace}");
        }
        for capacity in [0, 4097, u32::MAX] {
            assert!(BackboneExecution::workspace_bytes(&library, capacity).is_err());
        }
        Ok(())
    }
    #[test]
    fn ced_pass_requires_encoder_source_and_exact_decoder_range() -> Result<()> {
        let mut encoder = PassProgress::for_stage(CacheStage::Encoder);
        for layer in 0..20 { encoder.begin(7, layer)?; encoder.finish(); }
        let mut missing_source = PassProgress::for_stage(CacheStage::Encoder);
        for layer in 0..20 { missing_source.begin(7, layer)?; missing_source.finish(); }
        assert!(missing_source.commit(7).is_err());
        encoder.begin_decoder_source(7)?;
        assert!(encoder.invalid);
        encoder.finish_decoder_source();
        encoder.commit(7)?;
        assert!(encoder.commit(7).is_err());
        let mut interrupted = PassProgress::for_stage(CacheStage::Encoder);
        for layer in 0..20 { interrupted.begin(7, layer)?; interrupted.finish(); }
        interrupted.begin_decoder_source(7)?;
        assert!(interrupted.commit(7).is_err());
        let mut replay = PassProgress::for_stage(CacheStage::Replay);
        assert_eq!(replay.next, 20);
        for layer in 20..40 { replay.begin(8, layer)?; replay.finish(); }
        replay.commit(8)?;
        let mut wrong = PassProgress::for_stage(CacheStage::Replay);
        assert!(wrong.begin(8, 0).is_err());
        let mut wrong = PassProgress::for_stage(CacheStage::Encoder);
        assert!(wrong.begin_decoder_source(7).is_err());
        Ok(())
    }
    #[test]
    fn pass_commit_requires_all_layers_and_one_batch() -> Result<()> {
        let mut pass = PassProgress::default();
        assert!(pass.commit(1).is_err());
        assert!(pass.begin(1, 0).is_err());
        pass = PassProgress::default();
        pass.begin(1, 0)?;
        // An errored or cancelled operation never calls finish and cannot resume.
        assert!(pass.begin(1, 0).is_err());
        pass = PassProgress::default();
        for layer in 0..40 {
            pass.begin(1, layer)?;
            pass.finish();
        }
        assert!(pass.commit(2).is_err());
        assert!(pass.commit(1).is_err());
        pass = PassProgress::default();
        for layer in 0..40 {
            pass.begin(1, layer)?;
            pass.finish();
        }
        pass.commit(1)?;
        assert!(pass.commit(1).is_err());
        pass = PassProgress::default();
        pass.begin(1, 0)?;
        pass.finish();
        assert!(pass.begin(2, 1).is_err());
        pass = PassProgress::default();
        assert!(pass.begin(1, 1).is_err());
        Ok(())
    }
}

#[cfg(test)]
#[path = "v41_backbone_execution/distributed_tests.rs"]
mod distributed_tests;
