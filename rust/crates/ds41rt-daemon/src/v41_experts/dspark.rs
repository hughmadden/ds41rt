//! RTX-only ownership for every native dSpark tensor and independent stage experts.
mod attention_output;
pub(crate) use attention_output::DsparkAttentionOutput;
mod confidence;
mod projection;
pub(crate) use projection::{DsparkProjection, ProjectionKind};
mod ffn;
pub(crate) use ffn::DsparkFfn;
mod hc;
pub(crate) use hc::HcSublayer;
mod markov;
mod router;
mod shared;
pub(crate) use shared::DsparkSharedFfn;
mod terminal;
pub(crate) use confidence::DsparkConfidence;
pub(crate) use markov::DsparkMarkov;
pub(crate) use router::DsparkRouter;
pub(crate) use terminal::DsparkTerminal;

use super::{ExpertExecution, ExpertLayer, ExpertWeights};
use crate::v41_dspark_cache::DsparkWindow;
use crate::v41_tensors::NativeRtxTensors;
use anyhow::{ensure, Context, Result};
use ds41rt_ffi::{Ds41rtDeviceBuffer, NativeLibrary};
use ds41rt_loader::OfficialV41Catalog;

#[derive(Debug, Clone, Copy)]
pub(crate) struct DsparkBudget {
    pub expert_resident_bytes: usize,
    pub auxiliary_resident_bytes: usize,
    pub shared_packed_scale_bytes: usize,
    pub projection_packed_scale_bytes: usize,
    pub grouped_output_resident_bytes: usize,
    pub projection_bytes_per_wave: usize,
    pub attention_output_additional_bytes_per_wave: usize,
    pub shared_execution_bytes_per_wave: usize,
    pub load_staging_bytes: usize,
    pub window_cache_bytes: usize,
    pub execution_bytes_per_wave: usize,
    pub hc_bytes_per_wave: usize,
    pub router_bytes_per_wave: usize,
    pub confidence_bytes_per_wave: usize,
    pub markov_bytes_per_wave: usize,
    pub terminal_additional_bytes_per_wave: usize,
}
impl DsparkBudget {
    pub fn resident_bytes(self) -> Result<usize> {
        self.expert_resident_bytes
            .checked_add(self.auxiliary_resident_bytes)
            .and_then(|bytes| bytes.checked_add(self.shared_packed_scale_bytes))
            .and_then(|bytes| bytes.checked_add(self.projection_packed_scale_bytes))
            .and_then(|bytes| bytes.checked_add(self.grouped_output_resident_bytes))
            .context("dSpark residency overflow")
    }
    /// Experts load serially, then native auxiliary tensors, then execution waves.
    /// Committed dSpark windows are shared across waves and counted once below.
    /// Remaining draft-attention/norm scratch, shared head,
    /// driver and graph allocations must be budgeted separately by the coordinator.
    pub fn peak_device_bytes(self, waves: usize) -> Result<usize> {
        ensure!((1..=2).contains(&waves), "dSpark needs one or two waves");
        let execution = self
            .execution_bytes_per_wave
            .checked_add(self.shared_execution_bytes_per_wave)
            .and_then(|bytes| bytes.checked_add(self.projection_bytes_per_wave))
            .and_then(|bytes| bytes.checked_add(self.attention_output_additional_bytes_per_wave))
            .and_then(|bytes| bytes.checked_add(self.hc_bytes_per_wave))
            .and_then(|bytes| bytes.checked_add(self.router_bytes_per_wave))
            .and_then(|bytes| bytes.checked_add(self.confidence_bytes_per_wave))
            .and_then(|bytes| bytes.checked_add(self.markov_bytes_per_wave))
            .and_then(|bytes| bytes.checked_add(self.terminal_additional_bytes_per_wave))
            .context("dSpark combined wave budget overflow")?
            .checked_mul(waves)
            .context("dSpark wave budget overflow")?;
        let loading = self
            .expert_resident_bytes
            .checked_add(self.load_staging_bytes)
            .context("dSpark load peak overflow")?;
        let serving = self
            .resident_bytes()?
            .checked_add(execution)
            .and_then(|bytes| bytes.checked_add(self.window_cache_bytes))
            .context("dSpark serving peak overflow")?;
        Ok(loading.max(serving))
    }
}

pub(crate) struct DsparkWeights<'library> {
    experts: [ExpertWeights<'library>; 3],
    auxiliary: NativeRtxTensors<'library>,
    budget: DsparkBudget,
    shared_scales: [crate::v41_memory::DeviceAllocation<'library>; 9],
    grouped_output_weights: [crate::v41_memory::DeviceAllocation<'library>; 3],
    projection_scales: [crate::v41_memory::DeviceAllocation<'library>; 13],
}
impl<'library> DsparkWeights<'library> {
    fn auxiliary_names(catalog: &OfficialV41Catalog) -> Vec<String> {
        catalog
            .tensors()
            .iter()
            .map(|tensor| &tensor.metadata.name)
            .filter(|name| name.starts_with("mtp.") && !name.contains(".ffn.experts."))
            .cloned()
            .collect()
    }
    pub fn plan(
        library: &NativeLibrary,
        catalog: &OfficialV41Catalog,
        capacity: u32,
    ) -> Result<DsparkBudget> {
        let mut expert_resident_bytes = 0usize;
        let mut load_staging_bytes = 0usize;
        for stage in 0..3 {
            let budget = ExpertWeights::plan(library, catalog, ExpertLayer::Dspark { stage })?;
            expert_resident_bytes = expert_resident_bytes
                .checked_add(budget.resident_bytes)
                .context("dSpark expert residency overflow")?;
            load_staging_bytes = load_staging_bytes.max(budget.device_staging_bytes);
        }
        let auxiliary_resident_bytes =
            NativeRtxTensors::plan(catalog, &Self::auxiliary_names(catalog))?;
        let execution_bytes_per_wave = ExpertWeights::plan_execution(library, capacity)?
            .total()?
            .checked_mul(3)
            .context("dSpark stage workspace overflow")?;
        Ok(DsparkBudget {
            expert_resident_bytes,
            auxiliary_resident_bytes,
            shared_packed_scale_bytes: shared::packed_scale_bytes(library)?,
            projection_packed_scale_bytes: projection::packed_bytes(library)?,
            grouped_output_resident_bytes: 3 * 67108864,
            projection_bytes_per_wave: projection::wave_bytes(library, capacity)?,
            attention_output_additional_bytes_per_wave: DsparkAttentionOutput::additional_bytes(capacity)? * 3,
            shared_execution_bytes_per_wave: DsparkSharedFfn::device_bytes(library, capacity)? * 3,
            load_staging_bytes,
            window_cache_bytes: DsparkWindow::device_bytes(16, 4096)? * 3,
            execution_bytes_per_wave,
            router_bytes_per_wave: DsparkRouter::device_bytes(capacity as usize)? * 3,
            hc_bytes_per_wave: HcSublayer::device_bytes(capacity as usize)?
                .checked_mul(6)
                .context("mHC wave budget overflow")?,
            confidence_bytes_per_wave: DsparkConfidence::device_bytes((capacity as usize).max(80))?,
            markov_bytes_per_wave: DsparkMarkov::device_bytes(16)?,
            terminal_additional_bytes_per_wave: DsparkTerminal::additional_bytes(16)?,
        })
    }
    /// Admit all three stages and the requested expert wave workspaces before
    /// reading payloads; this does not allocate the wave workspaces themselves.
    pub fn load(
        library: &'library NativeLibrary,
        catalog: &OfficialV41Catalog,
        capacity: u32,
        waves: usize,
        device_budget: usize,
        pinned_staging_bytes: usize,
    ) -> Result<Self> {
        let budget = Self::plan(library, catalog, capacity)?;
        ensure!(
            budget.peak_device_bytes(waves)? <= device_budget,
            "dSpark RTX residency and expert waves exceed device budget"
        );
        ensure!(
            (1..=64 * 1024 * 1024).contains(&pinned_staging_bytes),
            "dSpark auxiliary pinned staging must be 1 byte through 64 MiB"
        );
        let mut experts = Vec::with_capacity(3);
        let mut resident = 0usize;
        for stage in 0..3 {
            let weights = ExpertWeights::load(
                library,
                catalog,
                ExpertLayer::Dspark { stage },
                device_budget
                    .checked_sub(resident)
                    .context("dSpark remaining budget underflow")?,
            )?;
            resident = resident
                .checked_add(weights.budget().resident_bytes)
                .context("dSpark loaded residency overflow")?;
            experts.push(weights);
        }
        let auxiliary = NativeRtxTensors::load(
            library,
            catalog,
            &Self::auxiliary_names(catalog),
            device_budget
                .checked_sub(resident)
                .context("dSpark auxiliary budget underflow")?,
            pinned_staging_bytes,
        )?;
        let experts = experts
            .try_into()
            .ok()
            .context("dSpark requires three expert stages")?;
        let shared_scales = shared::pack_scales(library, &auxiliary)?;
        let projection_scales = projection::pack_scales(library, &auxiliary)?;
        let grouped_output_weights = attention_output::dequant_weights(library, &auxiliary)?;
        Ok(Self {
            grouped_output_weights,
            shared_scales,
            projection_scales,
            experts,
            auxiliary,
            budget,
        })
    }
    /// Three independent committed windows, shared across alternating waves.
    /// Each admits sixteen requests and a total of 4096 source KV rows per batch.
    pub fn windows(&self, budget: usize) -> Result<[DsparkWindow<'library>; 3]> {
        let per_stage = DsparkWindow::device_bytes(16, 4096)?;
        ensure!(
            per_stage * 3 <= budget,
            "dSpark windows exceed device budget"
        );
        let library = self.experts[0].buffers[0].library;
        Ok([
            DsparkWindow::new(library, 16, 4096, per_stage)?,
            DsparkWindow::new(library, 16, 4096, per_stage)?,
            DsparkWindow::new(library, 16, 4096, per_stage)?,
        ])
    }
    pub fn budget(&self) -> DsparkBudget {
        self.budget
    }

    /// Includes stage-zero target projection, all attention/shared FFN/router/mHC
    /// tensors and the final Markov/confidence heads in native representations.
    pub fn tensor(&self, name: &str) -> Result<Ds41rtDeviceBuffer> {
        self.auxiliary.get(name)
    }

    /// Each stage gets stable independent buffers so its graph retains its own
    /// weights. The caller budgets each live wave and destroys it before weights.
    pub fn execution_wave(
        &self,
        capacity: u32,
        available_device_bytes: usize,
    ) -> Result<[ExpertExecution<'_, 'library>; 3]> {
        let per_stage = self.experts[0].execution_budget(capacity)?.total()?;
        let total = per_stage
            .checked_mul(3)
            .context("dSpark execution budget overflow")?;
        ensure!(
            total <= available_device_bytes,
            "dSpark expert wave exceeds device budget"
        );
        Ok([
            self.experts[0].execution(capacity, per_stage)?,
            self.experts[1].execution(capacity, per_stage)?,
            self.experts[2].execution(capacity, per_stage)?,
        ])
    }
}
