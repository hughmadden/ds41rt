//! Cache producers and the complete per-layer backbone execution handoff.
use crate::v41_backbone_cache::{BackboneCache, CacheBatch};
use crate::v41_backbone_lane::BackboneLane;
use crate::v41_compressor::{CompressorWave, CompressorWeights};
use crate::v41_experts::coordinator::NativeTp4Wave;
use crate::v41_index_lane::IndexLane;
use crate::v41_tensors::NativeRtxTensors;
use crate::v41_window::{WindowWave, WindowWeights};
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::NativeLibrary;
use ds41rt_loader::OfficialV41Catalog;
const SOURCES: [usize; 4] = [2, 8, 14, 20];
const INDEX: [usize; 8] = [2, 8, 14, 20, 24, 28, 32, 36];

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
    batch: Option<u64>,
    next: usize,
    invalid: bool,
}
impl PassProgress {
    fn begin(&mut self, batch: u64, layer: usize) -> Result<()> {
        let valid = !std::mem::replace(&mut self.invalid, true);
        ensure!(
            valid && layer == self.next && layer < 40 && self.batch.is_none_or(|id| id == batch),
            "backbone pass batch or layer differs; restart required"
        );
        self.batch = Some(batch);
        Ok(())
    }
    fn finish(&mut self) {
        self.next += 1;
        self.invalid = false;
    }
    fn commit(&mut self, batch: u64) -> Result<()> {
        let valid = !std::mem::replace(&mut self.invalid, true);
        ensure!(
            valid && self.next == 40 && self.batch == Some(batch),
            "cache commit requires this batch's complete backbone pass"
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
        })
    }
    /// Discard pass progress only after all consumers finish. The caller also
    /// restarts the backbone/index lanes and initializes the next query batch.
    pub fn restart(&mut self) {
        self.progress = PassProgress::default();
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
        // Invalidate even if obtaining the completed query or bank check fails.
        let layer = self.progress.next;
        self.progress.begin(batch.identity(), layer)?;
        bank.validate_batch(batch)?;
        let query = lane.query_output()?;
        ensure!(
            query.layer == layer,
            "backbone query layer differs from pass"
        );
        unsafe {
            bank.produce_window(batch, &query, &mut self.windows[layer])?;
        }
        if let Some(i) = SOURCES.iter().position(|&l| l == layer) {
            unsafe {
                bank.produce_source(batch, &query, &mut self.sources[i])?;
            }
        }
        let source = SOURCES
            .iter()
            .rposition(|&l| l <= layer)
            .map(|i| &self.sources[i]);
        let cache = bank.attention(batch, layer, &self.windows[layer], source)?;
        if INDEX.contains(&layer) {
            unsafe {
                lane.select_index(index, &cache)?;
            }
        }
        let sink = self
            .weights
            .sinks
            .get(&format!("layers.{layer}.attn.attn_sink"))?;
        let rows = batch.expert_rows();
        let mut ffn = unsafe { lane.attention_indexed_ffn(sink, &cache, index)? };
        let result = unsafe {
            ffn.execute_tp4(transport, placement, image_mask, &rows)
                .await?
        };
        drop(ffn);
        unsafe {
            lane.finish_ffn(result.binding(), result.values)?;
        }
        self.progress.finish();
        Ok(())
    }
    /// Commit only after the same batch completed all forty layers. The caller
    /// determines acceptance after target-head/sampling/verification and includes
    /// engram/dSpark history in the enclosing scheduler transaction.
    pub fn commit(
        &mut self,
        bank: &mut BackboneCache<'_>,
        batch: &CacheBatch,
        accepted: &[u32],
    ) -> Result<()> {
        self.progress.commit(batch.identity())?;
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
