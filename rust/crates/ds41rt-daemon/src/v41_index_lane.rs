//! One lane's learned index query and retained source-20 candidates.
use crate::v41_attention_query::AttentionQueryOutput;
use crate::v41_backbone_cache::CacheAttention;
use crate::v41_index_query::{IndexQueryWave, IndexQueryWeights};
use crate::v41_index_selection::{IndexSelectionOutput, IndexSelectionWave};
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::NativeLibrary;
use ds41rt_loader::OfficialV41Catalog;

const LAYERS: [usize; 8] = [2, 8, 14, 20, 24, 28, 32, 36];
pub(crate) struct IndexLaneWeights<'a> {
    library: &'a NativeLibrary,
    weights: Vec<IndexQueryWeights<'a>>,
}
impl<'a> IndexLaneWeights<'a> {
    pub fn device_bytes(library: &NativeLibrary, catalog: &OfficialV41Catalog) -> Result<usize> {
        LAYERS.into_iter().try_fold(0usize, |total, layer| {
            total
                .checked_add(IndexQueryWeights::device_bytes(library, catalog, layer)?)
                .context("index weight budget overflow")
        })
    }
    pub fn load(
        library: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        budget: usize,
        staging: usize,
    ) -> Result<Self> {
        ensure!(
            Self::device_bytes(library, catalog)? <= budget,
            "index lane weights exceed budget"
        );
        let weights = LAYERS
            .into_iter()
            .map(|layer| {
                IndexQueryWeights::load(
                    library,
                    catalog,
                    layer,
                    IndexQueryWeights::device_bytes(library, catalog, layer)?,
                    staging,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { library, weights })
    }
}

pub(crate) struct IndexLane<'w, 'a> {
    weights: &'w IndexLaneWeights<'a>,
    query: IndexQueryWave<'w, 'a>,
    source: IndexSelectionWave<'a>,
    reindex: IndexSelectionWave<'a>,
    next: usize,
    ready: Option<usize>,
    invalid: bool,
}
impl<'w, 'a> IndexLane<'w, 'a> {
    pub fn workspace_bytes(library: &NativeLibrary, capacity: u32) -> Result<[usize; 3]> {
        Ok([
            IndexQueryWave::device_bytes(library, capacity)?,
            IndexSelectionWave::device_bytes(capacity as usize)?,
            IndexSelectionWave::device_bytes(capacity as usize)?,
        ])
    }
    pub fn new(weights: &'w IndexLaneWeights<'a>, capacity: u32, budget: usize) -> Result<Self> {
        let bytes = Self::workspace_bytes(weights.library, capacity)?;
        let total = bytes.iter().try_fold(0usize, |n, &b| {
            n.checked_add(b).context("index workspace budget overflow")
        })?;
        ensure!(total <= budget, "index lane workspace exceeds budget");
        ensure!(
            weights.weights.len() == 8,
            "index lane requires all eight weight owners"
        );
        Ok(Self {
            weights,
            query: weights.weights[0].wave(capacity, bytes[0])?,
            source: IndexSelectionWave::new(weights.library, capacity as usize, bytes[1])?,
            reindex: IndexSelectionWave::new(weights.library, capacity as usize, bytes[2])?,
            next: 0,
            ready: None,
            invalid: false,
        })
    }
    /// Begin a new batch after all consumers finish; also recovers failed work.
    pub fn restart(&mut self) -> Result<()> {
        self.restart_at(0)
    }
    /// Replay begins at decoder source 20; earlier encoder selections are absent.
    pub fn restart_decoder(&mut self) -> Result<()> {
        self.restart_at(3)
    }
    fn restart_at(&mut self, first: usize) -> Result<()> {
        self.invalid = true;
        self.ready = None;
        self.source.clear_graph()?;
        self.reindex.clear_graph()?;
        self.query.rebind(&self.weights.weights[first])?;
        self.next = first;
        self.invalid = false;
        Ok(())
    }
    /// # Safety
    /// Query and cache belong to the same admitted batch. All producers have
    /// completed and no external writes race these owners. Call once per index
    /// producer in layer order; intermediate attention layers reuse output().
    pub unsafe fn select(
        &mut self,
        query: &AttentionQueryOutput<'_>,
        cache: &CacheAttention<'_>,
    ) -> Result<()> {
        let valid = !std::mem::replace(&mut self.invalid, true);
        self.ready = None;
        ensure!(
            valid && LAYERS.get(self.next) == Some(&query.layer),
            "index producer order differs; restart lane"
        );
        self.query.rebind(&self.weights.weights[self.next])?;
        let requests = cache.selection_requests()?;
        let projected = unsafe { self.query.execute_attention(query)? };
        if query.layer <= 20 {
            unsafe {
                self.source.execute(&projected, &requests, None)?;
            }
        } else {
            let candidates = self.source.output()?;
            unsafe {
                self.reindex
                    .execute(&projected, &requests, Some(&candidates))?;
            }
        }
        self.ready = Some(query.layer);
        self.next += 1;
        self.invalid = false;
        Ok(())
    }
    /// Revalidate exact proposal snapshots and row order for the attention layer.
    pub fn output(
        &self,
        layer: usize,
        cache: &CacheAttention<'_>,
    ) -> Result<IndexSelectionOutput<'_>> {
        ensure!(!self.invalid, "index lane requires restart");
        let producer = self.ready.context("index lane output unavailable")?;
        let output = if producer <= 20 {
            self.source.output()?
        } else {
            self.reindex.output()?
        };
        let requests = cache.selection_requests()?;
        let bindings = requests
            .iter()
            .flat_map(|r| r.positions.iter().map(|&p| (r.proposal.binding(), p)))
            .collect::<Vec<_>>();
        output.validate_attention(layer, &bindings)?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn official_index_lane_budgets_reject_before_allocation() -> Result<()> {
        let Some(path) = std::env::var_os("DS41RT_LANE_PLAN_LIBRARY") else {
            eprintln!("skip index planning test: DS41RT_LANE_PLAN_LIBRARY unset");
            return Ok(());
        };
        let model = std::env::var_os("DS41RT_LANE_PLAN_MODEL")
            .context("DS41RT_LANE_PLAN_MODEL required")?;
        let library = unsafe { NativeLibrary::load(path)? };
        let catalog = ds41rt_loader::read_official_v41_catalog(
            ds41rt_loader::OFFICIAL_V41_MODEL_ID,
            std::path::Path::new(&model),
        )?;
        let bytes = IndexLaneWeights::device_bytes(&library, &catalog)?;
        assert_eq!(bytes, 45_916_160);
        assert!(
            IndexLaneWeights::load(&library, &catalog, bytes - 1, 1024 * 1024)
                .err()
                .unwrap()
                .to_string()
                .contains("weights exceed budget")
        );
        let empty = IndexLaneWeights {
            library: &library,
            weights: Vec::new(),
        };
        for capacity in [1, 80, 4096] {
            let groups = IndexLane::workspace_bytes(&library, capacity)?;
            let total = groups.iter().sum::<usize>();
            assert!(IndexLane::new(&empty, capacity, total - 1)
                .err()
                .unwrap()
                .to_string()
                .contains("workspace exceeds budget"));
            assert!(IndexLane::new(&empty, capacity, total)
                .err()
                .unwrap()
                .to_string()
                .contains("all eight weight owners"));
            eprintln!("index lane capacity={capacity} workspace_groups={groups:?} total={total}");
        }
        for capacity in [0, 4097, u32::MAX] {
            assert!(IndexLane::workspace_bytes(&library, capacity).is_err());
        }
        Ok(())
    }
}
