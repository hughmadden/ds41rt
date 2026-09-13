//! Load each cache producer beside its SWA/compressed storage.
use super::*;
use crate::v41_backbone_cache::CacheWave;

/// One lane's per-layer producer workspaces, owned by the assigned GPUs.
pub(crate) struct PlacedProducerWaves<'w, 'a> {
    pub windows: Vec<DeviceOwner<'a, WindowWave<'w, 'a>>>,
    pub sources: Vec<DeviceOwner<'a, CompressorWave<'w, 'a>>>,
    pending_commit: Option<(u64, [u32; 16], usize)>,
}
/// Borrows only this lane. The caller retains the query and admitted bank
/// storage until completion; cancellation drains only this layer's producers.
pub(crate) struct PendingPlacedProduction<'p, 'w, 'i, 'a> {
    waves: &'p mut PlacedProducerWaves<'w, 'a>,
    batch: u64,
    layer: usize,
    window: bool,
    source: Option<usize>,
    window_ready: bool,
    source_ready: bool,
    complete: bool,
    index: Option<&'p mut DeviceOwner<'a, IndexLane<'i, 'a>>>,
    projection_ready: bool,
    selection_started: bool,
}
impl PendingPlacedProduction<'_, '_, '_, '_> {
    /// # Safety
    /// The original bank, query buffers and admitted slots remain alive and
    /// immutable. No other producer may overwrite this lane's proposals.
    pub unsafe fn poll(&mut self, bank: &BackboneCache<'_>, batch: &CacheBatch) -> Result<bool> {
        ensure!(
            !self.complete && batch.identity() == self.batch,
            "placed production batch differs or already complete"
        );
        bank.validate_batch(batch)?;
        if !self.window_ready {
            let state = bank.window(batch, self.layer)?;
            self.window_ready = self.waves.windows[self.layer]
                .on_device_mut(|wave| unsafe { wave.poll_query(state) })?;
        }
        if !self.source_ready {
            let source = self.source.context("missing placed source")?;
            let state = bank.source(batch, self.layer)?;
            self.source_ready = self.waves.sources[source]
                .on_device_mut(|wave| unsafe { wave.poll_query(state) })?;
        }
        if let Some(index) = self.index.as_deref_mut() {
            let device = index.device;
            if !self.projection_ready {
                self.projection_ready = device.run(|| index.poll_projection())?;
            }
            if !(self.window_ready && self.source_ready && self.projection_ready) {
                return Ok(false);
            }
            if !self.selection_started {
                let source = SOURCES
                    .iter()
                    .rposition(|&layer| layer <= self.layer)
                    .filter(|_| !batch.stage().reuses_sources())
                    .map(|i| self.waves.sources[i].get());
                let cache =
                    bank.attention(batch, self.layer, &self.waves.windows[self.layer], source)?;
                device.run(|| unsafe { index.enqueue_selection(&cache) })?;
                self.selection_started = true;
            }
            if !device.run(|| index.poll_selection())? {
                return Ok(false);
            }
        }
        self.complete = self.window_ready && self.source_ready;
        Ok(self.complete)
    }
}
impl Drop for PendingPlacedProduction<'_, '_, '_, '_> {
    fn drop(&mut self) {
        if self.complete {
            return;
        }
        if let Some(index) = self.index.as_deref_mut() {
            let device = index.device;
            if let Err(error) = device.run(|| index.get_mut().abort_pending()) {
                tracing::error!(%error, "draining cancelled placed index work");
            }
        }
        if self.window {
            if let Err(error) =
                self.waves.windows[self.layer].on_device_mut(|wave| wave.abort_query())
            {
                tracing::error!(%error, "draining cancelled placed window production");
            }
        }
        if let Some(source) = self.source {
            if let Err(error) = self.waves.sources[source].on_device_mut(|wave| wave.abort_query())
            {
                tracing::error!(%error, "draining cancelled placed compressed production");
            }
        }
    }
}
impl<'w, 'a> PlacedProducerWaves<'w, 'a> {
    /// Enqueue the layer's SWA and, where applicable, compressed source.
    /// Encoder boundary layer 20 produces only its compressed source.
    /// # Safety
    /// Query writes are complete and correspond to this admitted batch. Retain
    /// the query buffers and bank slots until completion or drained cancellation.
    pub unsafe fn enqueue_production<'p, 'i>(
        &'p mut self,
        bank: &BackboneCache<'_>,
        batch: &CacheBatch,
        query: &crate::v41_attention_query::AttentionQueryOutput<'_>,
    ) -> Result<PendingPlacedProduction<'p, 'w, 'i, 'a>> {
        ensure!(
            self.pending_commit.is_none(),
            "placed cache commit still pending"
        );
        bank.validate_batch(batch)?;
        let layer = query.layer;
        let window = batch.stage().windows().contains(&layer);
        ensure!(
            window || (batch.stage() == CacheStage::Encoder && layer == 20),
            "placed production layer outside batch stage"
        );
        let source = SOURCES
            .iter()
            .position(|&l| l == layer)
            .filter(|_| !batch.stage().reuses_sources());
        let device = bank.attention_device(layer)?;
        ensure!(
            query.hidden.device_id == device.id && self.windows[layer].device.id == device.id,
            "placed query, window and cache GPUs differ"
        );
        if let Some(i) = source {
            ensure!(
                self.sources[i].device.id == device.id,
                "placed source GPU differs"
            );
        }
        let pending = PendingPlacedProduction {
            waves: self,
            batch: batch.identity(),
            layer,
            window,
            source,
            window_ready: !window,
            source_ready: source.is_none(),
            complete: false,
            index: None,
            projection_ready: true,
            selection_started: false,
        };
        if window {
            let state = bank.window(batch, layer)?;
            let chunks = batch.window_chunks(layer)?;
            pending.waves.windows[layer]
                .on_device_mut(|wave| unsafe { wave.enqueue_query(state, &chunks, query) })?;
        }
        if let Some(i) = source {
            let state = bank.source(batch, layer)?;
            let chunks = batch.source_chunks(layer)?;
            pending.waves.sources[i]
                .on_device_mut(|wave| unsafe { wave.enqueue_query(state, &chunks, query) })?;
        }
        Ok(pending)
    }
    /// Project the learned index concurrently with SWA/compression, then select
    /// when this layer's producers are ready. This owner borrows only one lane.
    /// # Safety
    /// Same retained query/cache contract as enqueue_production. Index storage
    /// remains exclusive until completion or cancellation has drained consumers.
    pub unsafe fn enqueue_production_and_index<'p, 'i>(
        &'p mut self,
        bank: &BackboneCache<'_>,
        batch: &CacheBatch,
        query: &crate::v41_attention_query::AttentionQueryOutput<'_>,
        index: &'p mut DeviceOwner<'a, IndexLane<'i, 'a>>,
    ) -> Result<PendingPlacedProduction<'p, 'w, 'i, 'a>> {
        ensure!(
            batch.stage().windows().contains(&query.layer) && INDEX.contains(&query.layer),
            "placed index requires an attention index layer"
        );
        ensure!(
            index.device.id == bank.attention_device(query.layer)?.id,
            "placed index and cache GPUs differ"
        );
        let mut pending = unsafe { self.enqueue_production(bank, batch, query)? };
        pending.projection_ready = false;
        pending.index = Some(index);
        let index = pending.index.as_deref_mut().unwrap();
        let device = index.device;
        device.run(|| unsafe { index.get_mut().enqueue_projection(query) })?;
        Ok(pending)
    }
    pub fn device_bytes(
        library: &NativeLibrary,
        placement: CachePlacement,
        capacity: u32,
    ) -> Result<[usize; 2]> {
        let mut bytes = [0usize; 2];
        let window = WindowWave::device_bytes(library, capacity)?;
        for layer in 0..40 {
            let gpu = placement.attention(layer)?;
            bytes[gpu] = bytes[gpu]
                .checked_add(window)
                .context("placed SWA workspace overflow")?;
        }
        for layer in SOURCES {
            let gpu = placement.attention(layer)?;
            bytes[gpu] = bytes[gpu]
                .checked_add(CompressorWave::device_bytes(layer, capacity as usize)?)
                .context("placed compressor workspace overflow")?;
        }
        Ok(bytes)
    }
    pub fn new(
        weights: &'w CacheProducerWeights<'a>,
        capacity: u32,
        budgets: [usize; 2],
    ) -> Result<Self> {
        let placement = weights
            .placement
            .context("placed producer workspaces need a placement map")?;
        let bytes = Self::device_bytes(weights.library, placement, capacity)?;
        ensure!(
            bytes
                .into_iter()
                .zip(budgets)
                .all(|(need, budget)| need <= budget),
            "cache producer workspaces exceed a GPU budget"
        );
        let windows = weights
            .windows
            .iter()
            .map(|weights| {
                weights.device.own(|| {
                    weights.wave(
                        capacity,
                        WindowWave::device_bytes(weights.device.library, capacity)?,
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let sources = weights
            .sources
            .iter()
            .zip(SOURCES)
            .map(|(weights, layer)| {
                weights.device.own(|| {
                    weights.wave(
                        capacity as usize,
                        CompressorWave::device_bytes(layer, capacity as usize)?,
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            windows,
            sources,
            pending_commit: None,
        })
    }
    /// # Safety
    /// Retain the bank and this lane through publication or explicit abort.
    /// Producers/attention consumers have completed; no conflicting writes.
    pub unsafe fn enqueue_cache_commit(
        &mut self,
        bank: &BackboneCache<'_>,
        batch: &CacheBatch,
        accepted: &[u32],
    ) -> Result<()> {
        ensure!(
            self.pending_commit.is_none() && accepted.len() <= 16,
            "placed commit already pending or too many requests"
        );
        let mut counts = [0; 16];
        counts[..accepted.len()].copy_from_slice(accepted);
        self.pending_commit = Some((batch.identity(), counts, accepted.len()));
        unsafe { bank.enqueue_cache_commit(batch, &mut self.windows, &mut self.sources, accepted) }
    }
    pub fn poll_cache_commit(&self) -> Result<bool> {
        ensure!(self.pending_commit.is_some(), "placed commit not enqueued");
        for wave in &self.windows {
            if !wave.on_device(|wave| wave.poll_commit())? {
                return Ok(false);
            }
        }
        for wave in &self.sources {
            if !wave.on_device(|wave| wave.poll_commit())? {
                return Ok(false);
            }
        }
        Ok(true)
    }
    pub fn finish_cache_commit(
        &mut self,
        bank: &mut BackboneCache<'_>,
        batch: &CacheBatch,
        accepted: &[u32],
    ) -> Result<()> {
        let (identity, counts, len) = self.pending_commit.context("placed commit not enqueued")?;
        ensure!(
            identity == batch.identity() && &counts[..len] == accepted,
            "placed commit identity or acceptance changed"
        );
        ensure!(
            self.poll_cache_commit()?,
            "placed cache writes remain pending"
        );
        bank.commit(batch, &mut self.windows, &mut self.sources, accepted)?;
        self.pending_commit = None;
        Ok(())
    }
    pub fn abort_cache_commit(&mut self, bank: &mut BackboneCache<'_>) -> Result<()> {
        let result = bank.abort_cache_commit(&mut self.windows, &mut self.sources);
        self.pending_commit = None;
        result
    }
}

impl<'a> CacheProducerWeights<'a> {
    pub fn distributed_device_bytes(
        library: &NativeLibrary,
        catalog: &OfficialV41Catalog,
        placement: CachePlacement,
    ) -> Result<[usize; 2]> {
        let mut bytes = [0usize; 2];
        for layer in 0..40 {
            let gpu = placement.attention(layer)?;
            let sink =
                NativeRtxTensors::plan(catalog, &[format!("layers.{layer}.attn.attn_sink")])?;
            ensure!(sink == 256, "unexpected attention sink size");
            bytes[gpu] = bytes[gpu]
                .checked_add(WindowWeights::device_bytes(library, catalog, layer)?)
                .and_then(|n| n.checked_add(sink))
                .context("placed window weight budget overflow")?;
        }
        for layer in SOURCES {
            let gpu = placement.attention(layer)?;
            bytes[gpu] = bytes[gpu]
                .checked_add(CompressorWeights::device_bytes(catalog, layer)?)
                .context("placed compressor weight budget overflow")?;
        }
        Ok(bytes)
    }
    pub fn load_distributed(
        library: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        placement: CachePlacement,
        budgets: [usize; 2],
        staging: usize,
    ) -> Result<Self> {
        let bytes = Self::distributed_device_bytes(library, catalog, placement)?;
        ensure!(
            bytes
                .into_iter()
                .zip(budgets)
                .all(|(need, budget)| need <= budget),
            "cache producer weights exceed a GPU budget"
        );
        Self::load_placed(library, catalog, staging, Some(placement))
    }
    pub(super) fn load_placed(
        library: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        staging: usize,
        placement: Option<CachePlacement>,
    ) -> Result<Self> {
        let original = library.cuda_get_device()?;
        let device_for = |layer| -> Result<Device<'a>> {
            Ok(Device {
                library,
                id: match placement {
                    Some(p) => p.attention(layer)? as i32,
                    None => original,
                },
            })
        };
        let windows = (0..40)
            .map(|layer| {
                device_for(layer)?.own(|| {
                    WindowWeights::load(
                        library,
                        catalog,
                        layer,
                        WindowWeights::device_bytes(library, catalog, layer)?,
                        staging,
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let sources = SOURCES
            .into_iter()
            .map(|layer| {
                device_for(layer)?.own(|| {
                    CompressorWeights::load(
                        library,
                        catalog,
                        layer,
                        CompressorWeights::device_bytes(catalog, layer)?,
                        staging,
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let devices = if placement.is_some() {
            vec![0, 1]
        } else {
            vec![original]
        };
        let mut sinks = Vec::new();
        for id in devices {
            let names = (0..40)
                .filter_map(|layer| match device_for(layer) {
                    Ok(device) if device.id == id => {
                        Some(Ok(format!("layers.{layer}.attn.attn_sink")))
                    }
                    Ok(_) => None,
                    Err(error) => Some(Err(error)),
                })
                .collect::<Result<Vec<_>>>()?;
            if names.is_empty() {
                continue;
            }
            sinks.push(Device { library, id }.own(|| {
                NativeRtxTensors::load(
                    library,
                    catalog,
                    &names,
                    NativeRtxTensors::plan(catalog, &names)?,
                    staging,
                )
            })?);
        }
        let sink_views = (0..40)
            .map(|layer| {
                let id = device_for(layer)?.id;
                sinks
                    .iter()
                    .find(|group| group.device.id == id)
                    .context("attention sink GPU missing")?
                    .get()
                    .get(&format!("layers.{layer}.attn.attn_sink"))
            })
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .ok()
            .expect("40 attention sinks");
        Ok(Self {
            library,
            windows,
            sources,
            _sinks: sinks,
            sink_views,
            placement,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v41_window::{WindowChunk, WindowState};
    #[test]
    #[ignore = "requires DS41RT_NATIVE_LIB, DS41RT_SNAPSHOT, and two CUDA GPUs"]
    fn placed_query_production_poll_cancel_and_reuse() -> Result<()> {
        use crate::v41_attention_query::{AttentionQueryWave, AttentionQueryWeights};
        use crate::v41_backbone_cache::CacheWork;
        use ds41rt_transport::ExpertV2SourceKind;
        let lib = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        let catalog = ds41rt_loader::read_official_v41_catalog(
            ds41rt_loader::OFFICIAL_V41_MODEL_ID,
            std::path::Path::new(&std::env::var("DS41RT_SNAPSHOT")?),
        )?;
        lib.cuda_set_device(0)?;
        let placement = CachePlacement::new(std::array::from_fn(|layer| usize::from(layer >= 14)))?;
        let weights = CacheProducerWeights::load_distributed(
            &lib,
            &catalog,
            placement,
            CacheProducerWeights::distributed_device_bytes(&lib, &catalog, placement)?,
            1024 * 1024,
        )?;
        let mut waves = PlacedProducerWaves::new(
            &weights,
            16,
            PlacedProducerWaves::device_bytes(&lib, placement, 16)?,
        )?;
        for (layer, source_id) in [(2, 0), (14, 2), (20, 3)] {
            let pages = [2, 2, 2, 4];
            let mut bank = BackboneCache::new_distributed(
                &lib,
                placement,
                2,
                pages,
                BackboneCache::distributed_device_bytes(placement, 2, pages)?,
            )?;
            let lease = bank.begin_request(0, 1)?;
            let peer = bank.begin_request(1, 2)?;
            if layer == 20 {
                bank.begin_encoder(lease, 5)?;
            }
            let batch = bank.plan(&[CacheWork {
                lease,
                tokens: 5,
                kind: ExpertV2SourceKind::Prefill,
            }])?;
            let wrong = bank.plan(&[CacheWork {
                lease: peer,
                tokens: 5,
                kind: ExpertV2SourceKind::Prefill,
            }])?;
            let device = Device {
                library: &lib,
                id: placement.attention(layer)? as i32,
            };
            let qw = device.own(|| {
                AttentionQueryWeights::load(
                    &lib,
                    &catalog,
                    layer,
                    AttentionQueryWeights::device_bytes(&lib, &catalog, layer)?,
                    1024 * 1024,
                )
            })?;
            let mut query =
                device.own(|| qw.wave(16, AttentionQueryWave::device_bytes(&lib, 16)?))?;
            let mut reference_window = device
                .own(|| weights.windows[layer].wave(16, WindowWave::device_bytes(&lib, 16)?))?;
            let mut reference_source = device.own(|| {
                weights.sources[source_id].wave(16, CompressorWave::device_bytes(layer, 16)?)
            })?;
            let runtime = tokio::runtime::Builder::new_current_thread().build()?;
            for seed in [0, 7] {
                let host: Vec<u8> = (0..5 * 5120)
                    .flat_map(|i| {
                        let value = ((i + seed) % 17) as f32 / 32.0 - 0.25;
                        ((value.to_bits() >> 16) as u16).to_ne_bytes()
                    })
                    .collect();
                device.run(|| {
                    lib.copy_h2d(query.input(), &host)?;
                    unsafe {
                        query.execute_tokens(&[0, 1, 2, 3, 4])?;
                    }
                    Ok(())
                })?;
                let output = query.output()?;
                // Cancellation drains the owned streams and allows immediate reuse.
                drop(unsafe { waves.enqueue_production(&bank, &batch, &output)? });
                assert_eq!(lib.cuda_get_device()?, 0);
                let mut pending = unsafe { waves.enqueue_production(&bank, &batch, &output)? };
                assert!(unsafe { pending.poll(&bank, &wrong) }.is_err());
                runtime.block_on(async {
                    loop {
                        let complete = unsafe { pending.poll(&bank, &batch)? };
                        assert_eq!(lib.cuda_get_device()?, 0);
                        // Shared bank remains available while this lane is in flight.
                        assert_eq!(bank.committed_end(peer)?, 0);
                        if complete {
                            break;
                        }
                        tokio::task::yield_now().await;
                    }
                    Ok::<_, anyhow::Error>(())
                })?;
                drop(pending);
                device.run(|| {
                    unsafe {
                        if layer != 20 {
                            bank.produce_window(&batch, &output, &mut reference_window)?;
                        }
                        bank.produce_source(&batch, &output, &mut reference_source)?;
                    }
                    let source = waves.sources[source_id].output(bank.source(&batch, layer)?)?;
                    let expected_source = reference_source.output(bank.source(&batch, layer)?)?;
                    let mut pairs = vec![
                        (source.kv_values, expected_source.kv_values),
                        (source.kv_scales, expected_source.kv_scales),
                        (source.index_packed, expected_source.index_packed),
                        (source.index_scales, expected_source.index_scales),
                    ];
                    if layer != 20 {
                        let actual = waves.windows[layer].output(bank.window(&batch, layer)?)?;
                        let expected = reference_window.output(bank.window(&batch, layer)?)?;
                        pairs.extend([
                            (actual.values, expected.values),
                            (actual.scales, expected.scales),
                        ]);
                    } else {
                        assert!(bank.window(&batch, layer).is_err());
                    }
                    for (a, b) in pairs {
                        let mut left = vec![0; a.bytes];
                        let mut right = vec![0; b.bytes];
                        lib.copy_d2h(&mut left, a)?;
                        lib.copy_d2h(&mut right, b)?;
                        assert_eq!(
                            left, right,
                            "placed query production differs from direct execution"
                        );
                    }
                    Ok(())
                })?;
            }
            assert_eq!(lib.cuda_get_device()?, 0);
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires DS41RT_NATIVE_LIB, DS41RT_SNAPSHOT, and two CUDA GPUs"]
    fn real_placed_producers_and_window_graphs_match_across_devices() -> Result<()> {
        let lib = unsafe { NativeLibrary::load(std::env::var("DS41RT_NATIVE_LIB")?)? };
        let catalog = ds41rt_loader::read_official_v41_catalog(
            ds41rt_loader::OFFICIAL_V41_MODEL_ID,
            std::path::Path::new(&std::env::var("DS41RT_SNAPSHOT")?),
        )?;
        lib.cuda_set_device(0)?;
        let placement = CachePlacement::encoder_decoder();
        let budgets = CacheProducerWeights::distributed_device_bytes(&lib, &catalog, placement)?;
        assert_eq!(
            budgets.iter().sum::<usize>(),
            CacheProducerWeights::device_bytes(&lib, &catalog)?
        );
        let weights = CacheProducerWeights::load_distributed(
            &lib,
            &catalog,
            placement,
            budgets,
            16 * 1024 * 1024,
        )?;
        let devices = [
            Device {
                library: &lib,
                id: 0,
            },
            Device {
                library: &lib,
                id: 1,
            },
        ];
        let reference = devices[0].own(|| {
            WindowWeights::load(
                &lib,
                &catalog,
                20,
                WindowWeights::device_bytes(&lib, &catalog, 20)?,
                16 * 1024 * 1024,
            )
        })?;
        let reference_compressor = devices[0].own(|| {
            CompressorWeights::load(
                &lib,
                &catalog,
                20,
                CompressorWeights::device_bytes(&catalog, 20)?,
                16 * 1024 * 1024,
            )
        })?;
        let workspace = PlacedProducerWaves::device_bytes(&lib, placement, 16)?;
        assert_eq!(
            workspace.iter().sum::<usize>(),
            BackboneExecution::workspace_bytes(&lib, 16)?
        );
        let mut placed = PlacedProducerWaves::new(&weights, 16, workspace)?;
        for layer in 0..40 {
            assert_eq!(
                weights.windows[layer].device.id,
                placement.attention(layer)? as i32
            );
            assert_eq!(
                weights.sink_views[layer].device_id,
                placement.attention(layer)? as i32
            );
            assert_eq!(
                placed.windows[layer].input().device_id,
                placement.attention(layer)? as i32
            );
        }
        for (source, gpu) in weights.sources.iter().zip(placement.sources()) {
            assert_eq!(source.device.id, gpu as i32);
        }
        for (source, gpu) in placed.sources.iter().zip(placement.sources()) {
            assert_eq!(source.input().device_id, gpu as i32);
        }
        assert!(BackboneExecution::new(&weights, 16, usize::MAX).is_err());
        let mut states = [
            devices[0].own(|| WindowState::new(&lib, 20, 1, WindowState::device_bytes(20, 1)?))?,
            devices[1].own(|| WindowState::new(&lib, 20, 1, WindowState::device_bytes(20, 1)?))?,
        ];
        let leases = [
            states[0].begin_request(0, 1)?,
            states[1].begin_request(0, 1)?,
        ];
        let mut reference_wave =
            devices[0].own(|| reference.wave(16, WindowWave::device_bytes(&lib, 16)?))?;
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        let mut previous = None;
        for changed in [false, true] {
            let host: Vec<u8> = (0..16 * 5120)
                .flat_map(|i| {
                    let value: f32 = if changed {
                        (i % 17) as f32 / 32.0 - 0.25
                    } else {
                        0.0
                    };
                    ((value.to_bits() >> 16) as u16).to_ne_bytes()
                })
                .collect();
            let mut output = Vec::new();
            for rank in 0..2 {
                let wave = if rank == 0 {
                    reference_wave.get_mut()
                } else {
                    placed.windows[20].get_mut()
                };
                let state = states[rank].get();
                let chunks = [WindowChunk {
                    lease: leases[rank],
                    position: 0,
                    tokens: 16,
                }];
                let bytes = runtime.block_on(devices[rank].future(async {
                    lib.copy_h2d(wave.input(), &host)?;
                    tokio::task::yield_now().await;
                    if !changed {
                        unsafe {
                            wave.capture(state, &chunks)?;
                        }
                    }
                    unsafe {
                        wave.replay(state, &chunks)?;
                    }
                    let result = wave.output(state)?;
                    [
                        result.values,
                        result.scales,
                        result.projected,
                        result.normalized,
                        result.frequencies,
                    ]
                    .into_iter()
                    .map(|buffer| {
                        let mut bytes = vec![0; buffer.bytes];
                        lib.copy_d2h(&mut bytes, buffer)?;
                        Ok(bytes)
                    })
                    .collect::<Result<Vec<_>>>()
                }))?;
                output.push(bytes);
                assert_eq!(lib.cuda_get_device()?, 0);
            }
            assert!(
                output[0] == output[1],
                "placed SWA producer graph output differs between GPUs"
            );
            if let Some(old) = &previous {
                assert!(*old != output[0], "SWA graph ignored changed input");
            }
            previous = Some(output.remove(0));
        }
        let mut source_states = [
            devices[0]
                .own(|| crate::v41_compressor::CompressorState::new(&lib, 20, 1, 2, usize::MAX))?,
            devices[1]
                .own(|| crate::v41_compressor::CompressorState::new(&lib, 20, 1, 2, usize::MAX))?,
        ];
        let source_leases = [
            source_states[0].begin_request(0, 1)?,
            source_states[1].begin_request(0, 1)?,
        ];
        let mut reference_compressor_wave = devices[0]
            .own(|| reference_compressor.wave(16, CompressorWave::device_bytes(20, 16)?))?;
        let mut previous = None;
        for changed in [false, true] {
            let host: Vec<u8> = (0..16 * 5120)
                .flat_map(|i| {
                    let value: f32 = if changed {
                        (i % 17) as f32 / 32.0 - 0.25
                    } else {
                        0.0
                    };
                    ((value.to_bits() >> 16) as u16).to_ne_bytes()
                })
                .collect();
            let mut output = Vec::new();
            for rank in 0..2 {
                let wave = if rank == 0 {
                    reference_compressor_wave.get_mut()
                } else {
                    placed.sources[3].get_mut()
                };
                let state = source_states[rank].get();
                let chunks = [crate::v41_compressor::CompressorChunk {
                    lease: source_leases[rank],
                    position: 0,
                    tokens: 16,
                }];
                output.push(runtime.block_on(devices[rank].future(async {
                    lib.copy_h2d(wave.input(), &host)?;
                    tokio::task::yield_now().await;
                    if !changed {
                        unsafe {
                            wave.capture(state, &chunks)?;
                        }
                    }
                    unsafe {
                        wave.replay(state, &chunks)?;
                    }
                    let result = wave.output(state)?;
                    [
                        result.buffer,
                        result.index_key,
                        result.index_packed,
                        result.index_scales,
                        result.kv_values,
                        result.kv_scales,
                    ]
                    .into_iter()
                    .map(|buffer| {
                        let mut bytes = vec![0; buffer.bytes];
                        lib.copy_d2h(&mut bytes, buffer)?;
                        Ok(bytes)
                    })
                    .collect::<Result<Vec<_>>>()
                }))?);
                assert_eq!(lib.cuda_get_device()?, 0);
            }
            assert!(
                output[0] == output[1],
                "placed source-20 graph output differs between GPUs"
            );
            if let Some(old) = &previous {
                assert!(*old != output[0], "compressor graph ignored changed input");
            }
            previous = Some(output.remove(0));
        }
        eprintln!(
            "placed cache-producer weight budgets: GPU0={} GPU1={}",
            budgets[0], budgets[1]
        );
        eprintln!(
            "placed C16 producer workspace budgets per lane: GPU0={} GPU1={}",
            workspace[0], workspace[1]
        );
        let single = CacheProducerWeights::load(
            &lib,
            &catalog,
            CacheProducerWeights::device_bytes(&lib, &catalog)?,
            16 * 1024 * 1024,
        )?;
        let single_execution =
            BackboneExecution::new(&single, 16, BackboneExecution::workspace_bytes(&lib, 16)?)?;
        assert!(single.sink_views.iter().all(|buffer| buffer.device_id == 0));
        assert!(
            single_execution
                .windows
                .iter()
                .all(|wave| wave.input().device_id == 0)
        );
        assert!(
            single_execution
                .sources
                .iter()
                .all(|wave| wave.input().device_id == 0)
        );
        assert_eq!(lib.cuda_get_device()?, 0);
        Ok(())
    }
}
