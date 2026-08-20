#![allow(dead_code)]

use std::cmp::Ordering;
use std::collections::{BTreeMap, VecDeque};
use std::sync::OnceLock;

use anyhow::{Context, Result};
use ds4rt_core::{
    DeepseekV4AttentionLayerSource, DeepseekV4AttentionPlan, DeepseekV4KvCacheFormat, ModelFacts,
    TensorCatalog, DS4_KV_NVFP4_BYTES_PER_ROW,
};
use ds4rt_loader::validate_native_deepseek_v4_dspark_catalog;

use super::coordinator_kernels::DeviceBf16Output;

const NATIVE_DSPARK_TARGET_TAPS: usize = 3;
const NATIVE_DSPARK_TARGET_RING_SLOTS: usize = 128;
pub(super) const NATIVE_DSPARK_PROPOSAL_TOKENS: usize = 5;

/// Hardware-qualified end-to-end target verification latency for the public
/// Flash 0731 K2 1xRTX + 4xSpark TP4 AFD topology. C1 uses warmed, routed
/// serving-corpus observations; C2-C4 use the routing-diverse startup sweep.
/// Each slice is indexed by request count minus one and contains every legal
/// `(total_rows, ms)` cell.
///
/// Regenerate this profile whenever the expert AOT kernels/layout, SparkInfer
/// or B12x/CUDA libraries, transport/reduction path, scheduler execution path,
/// GPU/Spark topology or clock policy, or target/router checkpoint geometry
/// changes. Run with `DS4RT_REAL_FULL_DSPARK_PROFILE_AT_STARTUP=1` for a raw
/// diagnostic curve, then qualify affected cells on the routed production
/// corpus before replacing these rows and advancing the profile identity.
pub(super) const FLASH_0731_K2_AFD_SPS_PROFILE_ID: &str =
    "flash0731-k2-1rtx-4spark-direct-m2-m3-corpus-v2";
pub(super) static FLASH_0731_K2_AFD_SPS_PROFILE_MS: [&[(usize, f64)]; 4] = [
    &[
        (1, 23.221),
        (2, 26.077),
        (3, 27.138),
        (4, 28.257),
        (5, 32.338),
        (6, 32.689),
    ],
    &[
        (2, 36.659),
        (3, 37.045),
        (4, 36.327),
        (5, 37.153),
        (6, 36.684),
        (7, 37.555),
        (8, 36.540),
        (9, 37.305),
        (10, 37.325),
        (11, 36.551),
        (12, 37.000),
    ],
    &[
        (3, 51.174),
        (4, 51.874),
        (5, 51.396),
        (6, 51.559),
        (7, 52.220),
        (8, 52.225),
        (9, 52.903),
        (10, 52.835),
        (11, 53.059),
        (12, 52.893),
        (13, 53.234),
        (14, 53.337),
        (15, 53.287),
        (16, 56.772),
        (17, 56.872),
        (18, 57.114),
    ],
    &[
        (4, 61.767),
        (5, 63.046),
        (6, 63.492),
        (7, 64.659),
        (8, 64.293),
        (9, 65.133),
        (10, 65.437),
        (11, 66.125),
        (12, 66.140),
        (13, 66.362),
        (14, 66.274),
        (15, 67.568),
        (16, 66.679),
        (17, 67.488),
        (18, 67.580),
        (19, 67.771),
        (20, 68.284),
        (21, 68.550),
        (22, 68.744),
        (23, 68.802),
        (24, 68.963),
    ],
];
const NATIVE_DSPARK_CACHE_PAGE_TOKENS: usize = 256;
pub(super) const NATIVE_DSPARK_CACHE_PAGE_BYTES: usize = 149_760;

pub(super) fn native_dspark_cache_page_bytes(format: DeepseekV4KvCacheFormat) -> usize {
    match format {
        DeepseekV4KvCacheFormat::Fp8Ue8m0 => NATIVE_DSPARK_CACHE_PAGE_BYTES,
        DeepseekV4KvCacheFormat::Nvfp4 => {
            NATIVE_DSPARK_CACHE_PAGE_TOKENS * DS4_KV_NVFP4_BYTES_PER_ROW
        }
    }
}
static ACTIVE_DSPARK_TARGET_TAPS: OnceLock<[usize; NATIVE_DSPARK_TARGET_TAPS]> = OnceLock::new();
static ACTIVE_DSPARK_MAX_VERIFY_DRAFTS: OnceLock<usize> = OnceLock::new();

fn native_dspark_target_hidden_boundaries(
    facts: &ModelFacts,
) -> Result<[usize; NATIVE_DSPARK_TARGET_TAPS]> {
    facts
        .dspark_target_layer_ids
        .iter()
        .map(|layer_id| {
            anyhow::ensure!(
                *layer_id < facts.num_hidden_layers,
                "native DeepSeek dSpark target layer {layer_id} exceeds the {}-layer target",
                facts.num_hidden_layers,
            );
            layer_id
                .checked_add(1)
                .context("native DeepSeek dSpark target hidden boundary overflow")
        })
        .collect::<Result<Vec<_>>>()?
        .try_into()
        .map_err(|boundaries: Vec<usize>| {
            anyhow::anyhow!(
                "native DeepSeek dSpark requires exactly {NATIVE_DSPARK_TARGET_TAPS} target hidden boundaries, got {boundaries:?}"
            )
        })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NativeDsparkExpertTopology {
    StrictTensorParallel4,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct NativeDsparkBlockPlan {
    block_index: usize,
    physical_layer_id: usize,
    target_layer_id: usize,
    storage_prefix: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct NativeDsparkExecutionPlan {
    target_taps: [usize; NATIVE_DSPARK_TARGET_TAPS],
    blocks: Vec<NativeDsparkBlockPlan>,
    proposal_tokens: usize,
    sliding_window: usize,
    noise_token_id: usize,
    hidden_size: usize,
    vocab_size: usize,
    expert_topology: NativeDsparkExpertTopology,
}

impl NativeDsparkExecutionPlan {
    fn from_facts(facts: &ModelFacts) -> Result<Self> {
        let attention = DeepseekV4AttentionPlan::from_model_facts(facts)
            .context("building native dSpark physical block plan")?;
        // The checkpoint stores zero-based transformer layer indices. Target
        // hidden-state capture is named by the post-layer boundary, matching
        // HF hidden_states[layer_id + 1] and vLLM's dSpark contract.
        let target_taps = native_dspark_target_hidden_boundaries(facts)?;
        let blocks = attention
            .dspark_layers()
            .iter()
            .map(|layer| {
                let DeepseekV4AttentionLayerSource::Dspark {
                    block_index,
                    target_layer_id,
                } = layer.source
                else {
                    anyhow::bail!(
                        "native dSpark physical layer {} resolved to a target block",
                        layer.logical_layer_id
                    );
                };
                anyhow::ensure!(
                    layer.compress_ratio == 0,
                    "native dSpark block {block_index} must use sliding attention, got compression ratio {}",
                    layer.compress_ratio
                );
                Ok(NativeDsparkBlockPlan {
                    block_index,
                    physical_layer_id: layer.logical_layer_id,
                    target_layer_id,
                    storage_prefix: layer.source.checkpoint_block_prefix(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        anyhow::ensure!(
            blocks.len() == NATIVE_DSPARK_TARGET_TAPS,
            "native dSpark has {} physical blocks for {NATIVE_DSPARK_TARGET_TAPS} target taps",
            blocks.len()
        );
        anyhow::ensure!(
            facts.sliding_window == NATIVE_DSPARK_TARGET_RING_SLOTS,
            "native dSpark requires a {NATIVE_DSPARK_TARGET_RING_SLOTS}-slot target ring, got {}",
            facts.sliding_window,
        );
        anyhow::ensure!(
            facts.dspark_block_size == NATIVE_DSPARK_PROPOSAL_TOKENS,
            "native dSpark requires {NATIVE_DSPARK_PROPOSAL_TOKENS} proposal tokens, got {}",
            facts.dspark_block_size,
        );
        Ok(Self {
            target_taps,
            blocks,
            proposal_tokens: facts.dspark_block_size,
            sliding_window: facts.sliding_window,
            noise_token_id: facts.dspark_noise_token_id,
            hidden_size: facts.hidden_size,
            vocab_size: facts.vocab_size,
            expert_topology: NativeDsparkExpertTopology::StrictTensorParallel4,
        })
    }
}

/// One faithful invocation of DeepSeek's integrated dSpark decoder. The
/// checkpoint calls these blocks `mtp.*`, but none of the GLM recurrent-MTP
/// state belongs here: the target taps are projected together, all three
/// physical blocks see the same five-row proposal block, and every routed MoE
/// invocation uses strict TP=4.
#[derive(Clone, Debug, Eq, PartialEq)]
struct NativeDsparkReplayPlan {
    context_tokens_before: usize,
    main_context_start: usize,
    main_context_end: usize,
    committed_main_rows: usize,
    target_row_start: usize,
    cache_context_tokens_after: usize,
    cache_window_start: usize,
    proposal_position_start: usize,
    proposal_position_end: usize,
    draft_input_token_ids: Vec<usize>,
    blocks: Vec<NativeDsparkBlockPlan>,
    expert_topology: NativeDsparkExpertTopology,
}

impl NativeDsparkReplayPlan {
    fn for_request(
        execution: &NativeDsparkExecutionPlan,
        state: &DsparkRequestState,
        target_row_start: usize,
        committed_rows: usize,
        absolute_context_start: Option<usize>,
        anchor_token: usize,
    ) -> Result<Self> {
        anyhow::ensure!(
            committed_rows > 0,
            "native dSpark requires committed target rows"
        );
        anyhow::ensure!(
            anchor_token < execution.vocab_size,
            "native dSpark anchor token {anchor_token} exceeds vocabulary {}",
            execution.vocab_size,
        );
        let main_context_start = absolute_context_start.unwrap_or(state.context_tokens);
        let main_context_end = main_context_start
            .checked_add(committed_rows)
            .context("native dSpark committed context overflow")?;
        anyhow::ensure!(
            absolute_context_start.is_some() || main_context_start == state.context_tokens,
            "native dSpark incremental replay must begin at context {}, got {main_context_start}",
            state.context_tokens,
        );
        anyhow::ensure!(
            main_context_end >= state.context_tokens,
            "native dSpark replay end {main_context_end} precedes request context {}",
            state.context_tokens,
        );
        let cache_context_tokens_after =
            if absolute_context_start.is_some() && main_context_start != state.context_tokens {
                committed_rows.min(execution.sliding_window)
            } else {
                state
                    .cache_context_tokens
                    .checked_add(committed_rows)
                    .context("native dSpark cache context overflow")?
                    .min(execution.sliding_window)
            };
        let cache_window_start = main_context_end.saturating_sub(cache_context_tokens_after);
        let proposal_position_end = main_context_end
            .checked_add(execution.proposal_tokens)
            .context("native dSpark proposal position overflow")?;
        let mut draft_input_token_ids = vec![execution.noise_token_id; execution.proposal_tokens];
        draft_input_token_ids[0] = anchor_token;
        Ok(Self {
            context_tokens_before: state.context_tokens,
            main_context_start,
            main_context_end,
            committed_main_rows: committed_rows,
            target_row_start,
            cache_context_tokens_after,
            cache_window_start,
            proposal_position_start: main_context_end,
            proposal_position_end,
            draft_input_token_ids,
            blocks: execution.blocks.clone(),
            expert_topology: execution.expert_topology,
        })
    }

    fn validate_target_taps(
        &self,
        execution: &NativeDsparkExecutionPlan,
        target_hidden_taps: [&DeviceBf16Output; NATIVE_DSPARK_TARGET_TAPS],
    ) -> Result<()> {
        let target_row_end = self
            .target_row_start
            .checked_add(self.committed_main_rows)
            .context("native dSpark target tap row range overflow")?;
        for (tap_index, tap) in target_hidden_taps.into_iter().enumerate() {
            anyhow::ensure!(
                tap.rows >= target_row_end && tap.values_per_row == execution.hidden_size,
                "native dSpark target tap {tap_index} must cover rows {}..{} at width {}, got {}x{}",
                self.target_row_start,
                target_row_end,
                execution.hidden_size,
                tap.rows,
                tap.values_per_row,
            );
            anyhow::ensure!(
                tap.buffer().device_id == 0,
                "native dSpark target tap {tap_index} must reside on coordinator GPU0, got device {}",
                tap.buffer().device_id,
            );
        }
        Ok(())
    }
}

pub(super) fn dspark_target_hidden_tap_layer_ids() -> [usize; NATIVE_DSPARK_TARGET_TAPS] {
    ACTIVE_DSPARK_TARGET_TAPS
        .get()
        .copied()
        .unwrap_or([41, 42, 43])
}

pub(super) fn dspark_active_max_verify_drafts() -> usize {
    ACTIVE_DSPARK_MAX_VERIFY_DRAFTS.get().copied().unwrap_or(5)
}

fn activate_native_dspark_contract(facts: &ModelFacts) -> Result<()> {
    let target_taps = native_dspark_target_hidden_boundaries(facts)?;
    if let Some(active) = ACTIVE_DSPARK_TARGET_TAPS.get() {
        anyhow::ensure!(
            *active == target_taps,
            "native dSpark target taps are already active as {active:?}, cannot switch to {target_taps:?}"
        );
    } else {
        let _ = ACTIVE_DSPARK_TARGET_TAPS.set(target_taps);
    }
    if let Some(active) = ACTIVE_DSPARK_MAX_VERIFY_DRAFTS.get() {
        anyhow::ensure!(
            *active == facts.dspark_block_size,
            "native dSpark proposal width is already active as {active}, cannot switch to {}",
            facts.dspark_block_size,
        );
    } else {
        let _ = ACTIVE_DSPARK_MAX_VERIFY_DRAFTS.set(facts.dspark_block_size);
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub(super) struct DsparkDraftStep {
    pub(super) context_tokens: usize,
    pub(super) committed_rows: usize,
    pub(super) anchor_token: usize,
    pub(super) proposal_token_ids: Vec<usize>,
    pub(super) conditional_confidence: Vec<f32>,
    pub(super) update_ms: f64,
    pub(super) suffix_ms: f64,
    pub(super) readback_ms: f64,
    pub(super) total_ms: f64,
}

pub(super) struct DsparkRequestEngine {
    execution_plan: NativeDsparkExecutionPlan,
    max_verify_drafts: usize,
    max_cache_context_tokens: usize,
    kv_bytes_per_request: usize,
    free_slots: Vec<usize>,
    next_replay_id: u64,
}

pub(super) struct DsparkRequestState {
    slot: usize,
    context_tokens: usize,
    cache_context_tokens: usize,
    pending_replay_id: Option<u64>,
}

impl DsparkRequestState {
    pub(super) fn context_tokens(&self) -> usize {
        self.context_tokens
    }

    pub(super) fn request_slot(&self) -> usize {
        self.slot
    }

    pub(super) fn cache_context_tokens(&self) -> usize {
        self.cache_context_tokens
    }
}

pub(super) struct DsparkRequestCacheSnapshot {
    pub(super) context_tokens: usize,
    pub(super) cache_context_tokens: usize,
    pub(super) kv_bytes: Vec<u8>,
}

/// Request reservation created under the dSpark state lock and consumed after
/// coordinator/Spark execution finishes outside that lock.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct DsparkPreparedReplay {
    replay_id: u64,
    request_slot: usize,
    hidden_size: usize,
    vocab_size: usize,
    replay: NativeDsparkReplayPlan,
}

impl DsparkPreparedReplay {
    pub(super) fn request_slot(&self) -> usize {
        self.request_slot
    }

    pub(super) fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub(super) fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub(super) fn target_row_start(&self) -> usize {
        self.replay.target_row_start
    }

    pub(super) fn committed_main_rows(&self) -> usize {
        self.replay.committed_main_rows
    }

    pub(super) fn main_context_range(&self) -> std::ops::Range<usize> {
        self.replay.main_context_start..self.replay.main_context_end
    }

    pub(super) fn proposal_position_range(&self) -> std::ops::Range<usize> {
        self.replay.proposal_position_start..self.replay.proposal_position_end
    }

    pub(super) fn cache_window_start(&self) -> usize {
        self.replay.cache_window_start
    }

    pub(super) fn cache_context_tokens_after(&self) -> usize {
        self.replay.cache_context_tokens_after
    }

    pub(super) fn draft_input_token_ids(&self) -> &[usize] {
        &self.replay.draft_input_token_ids
    }

    pub(super) fn physical_layer_ids(&self) -> Vec<usize> {
        self.replay
            .blocks
            .iter()
            .map(|block| block.physical_layer_id)
            .collect()
    }

    pub(super) fn storage_prefixes(&self) -> Vec<&str> {
        self.replay
            .blocks
            .iter()
            .map(|block| block.storage_prefix.as_str())
            .collect()
    }

    pub(super) fn uses_strict_tp4(&self) -> bool {
        self.replay.expert_topology == NativeDsparkExpertTopology::StrictTensorParallel4
    }

    /// Physical compressed-MLA slots shared by all five proposal queries.
    /// Each request owns one 256-slot page per dSpark block: the target ring
    /// occupies 0..128 and the non-causal proposal KV suffix occupies 128..133.
    pub(super) fn attention_physical_selection(&self) -> Result<Vec<i32>> {
        let page_base = self
            .request_slot
            .checked_mul(NATIVE_DSPARK_CACHE_PAGE_TOKENS)
            .context("native dSpark request page offset overflow")?;
        let mut main_slots = (self.replay.cache_window_start..self.replay.main_context_end)
            .map(|position| position % NATIVE_DSPARK_TARGET_RING_SLOTS)
            .collect::<Vec<_>>();
        main_slots.sort_unstable();
        anyhow::ensure!(
            main_slots.len() == self.replay.cache_context_tokens_after
                && main_slots.len() <= NATIVE_DSPARK_TARGET_RING_SLOTS,
            "native dSpark active target selection has {} slots for cache length {}",
            main_slots.len(),
            self.replay.cache_context_tokens_after,
        );
        let proposal_slots = NATIVE_DSPARK_TARGET_RING_SLOTS
            ..NATIVE_DSPARK_TARGET_RING_SLOTS + self.replay.draft_input_token_ids.len();
        main_slots
            .into_iter()
            .chain(proposal_slots)
            .map(|slot| {
                let physical = page_base
                    .checked_add(slot)
                    .context("native dSpark physical cache slot overflow")?;
                i32::try_from(physical).context("native dSpark physical cache slot exceeds i32")
            })
            .collect()
    }
}

/// Numeric results produced by the integrated dSpark executor. Confidence is
/// normalized to conditional probabilities before this boundary.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct DsparkReplayOutput {
    pub(super) proposal_token_ids: Vec<usize>,
    pub(super) conditional_confidence: Vec<f32>,
    pub(super) update_ms: f64,
    pub(super) suffix_ms: f64,
    pub(super) readback_ms: f64,
    pub(super) total_ms: f64,
}

impl DsparkRequestCacheSnapshot {
    pub(super) fn resident_bytes(&self) -> usize {
        self.kv_bytes.len()
    }
}

impl DsparkRequestEngine {
    pub(super) fn max_verify_drafts(&self) -> usize {
        self.max_verify_drafts
    }

    pub(super) fn load(
        target_catalog: &TensorCatalog,
        kv_capacity_tokens: usize,
        max_active_requests: usize,
    ) -> Result<Self> {
        Self::load_with_cache_page_bytes(
            target_catalog,
            kv_capacity_tokens,
            max_active_requests,
            NATIVE_DSPARK_CACHE_PAGE_BYTES,
        )
    }

    pub(super) fn load_with_cache_page_bytes(
        target_catalog: &TensorCatalog,
        kv_capacity_tokens: usize,
        max_active_requests: usize,
        cache_page_bytes: usize,
    ) -> Result<Self> {
        let summary = validate_native_deepseek_v4_dspark_catalog(target_catalog)
            .context("validating integrated DeepSeek V4 dSpark checkpoint blocks")?;
        let execution_plan = NativeDsparkExecutionPlan::from_facts(&target_catalog.facts)?;
        activate_native_dspark_contract(&target_catalog.facts)?;
        anyhow::ensure!(
            summary.blocks == execution_plan.blocks.len()
                && summary.proposal_tokens == execution_plan.proposal_tokens,
            "native dSpark catalog summary blocks/width {}/{} disagrees with execution plan {}/{}",
            summary.blocks,
            summary.proposal_tokens,
            execution_plan.blocks.len(),
            execution_plan.proposal_tokens,
        );
        anyhow::ensure!(
            max_active_requests > 0,
            "native dSpark requires at least one request slot"
        );
        anyhow::ensure!(
            matches!(cache_page_bytes, 149_760 | 110_592),
            "native dSpark cache page must use the qualified FP8 or NVFP4 width, got {cache_page_bytes}"
        );
        anyhow::ensure!(
            kv_capacity_tokens
                >= target_catalog.facts.sliding_window + target_catalog.facts.dspark_block_size,
            "native dSpark KV capacity {kv_capacity_tokens} cannot hold its {}-token sliding window and {}-token proposal block",
            target_catalog.facts.sliding_window,
            target_catalog.facts.dspark_block_size,
        );
        let kv_bytes_per_request = summary
            .blocks
            .checked_mul(cache_page_bytes)
            .context("native dSpark packed KV bytes/request overflow")?;
        eprintln!(
            "real_full_native_dspark_contract blocks={} target_taps={} proposal_tokens={} coordinator_tensors={} routed_expert_tensors={} kv_bytes_per_request={}",
            summary.blocks,
            summary.target_taps,
            summary.proposal_tokens,
            summary.coordinator_tensors,
            summary.routed_expert_tensors,
            kv_bytes_per_request,
        );
        Ok(Self {
            execution_plan,
            max_verify_drafts: summary.proposal_tokens,
            max_cache_context_tokens: target_catalog.facts.sliding_window,
            kv_bytes_per_request,
            free_slots: (0..max_active_requests).rev().collect(),
            next_replay_id: 1,
        })
    }

    pub(super) fn allocate_request_state(&mut self) -> Result<DsparkRequestState> {
        let slot = self
            .free_slots
            .pop()
            .context("native dSpark request slots are exhausted")?;
        Ok(DsparkRequestState {
            slot,
            context_tokens: 0,
            cache_context_tokens: 0,
            pending_replay_id: None,
        })
    }

    pub(super) fn reset_request_state(&mut self, state: &mut DsparkRequestState) -> Result<()> {
        anyhow::ensure!(
            state.pending_replay_id.is_none(),
            "cannot reset native dSpark request slot {} during replay {:?}",
            state.slot,
            state.pending_replay_id,
        );
        state.context_tokens = 0;
        state.cache_context_tokens = 0;
        Ok(())
    }

    pub(super) fn release_request_state(&mut self, state: DsparkRequestState) {
        debug_assert!(!self.free_slots.contains(&state.slot));
        debug_assert!(state.pending_replay_id.is_none());
        self.free_slots.push(state.slot);
    }

    pub(super) fn snapshot_request_state(
        &self,
        state: &DsparkRequestState,
        kv_bytes: Vec<u8>,
    ) -> Result<Option<DsparkRequestCacheSnapshot>> {
        anyhow::ensure!(
            state.pending_replay_id.is_none(),
            "cannot snapshot native dSpark request slot {} during replay {:?}",
            state.slot,
            state.pending_replay_id,
        );
        if state.cache_context_tokens == 0 {
            anyhow::ensure!(
                kv_bytes.is_empty(),
                "empty native dSpark request state supplied {} snapshot bytes",
                kv_bytes.len(),
            );
            return Ok(None);
        }
        anyhow::ensure!(
            kv_bytes.len() == self.kv_bytes_per_request,
            "native dSpark snapshot has {} bytes, expected {}",
            kv_bytes.len(),
            self.kv_bytes_per_request,
        );
        Ok(Some(DsparkRequestCacheSnapshot {
            context_tokens: state.context_tokens,
            cache_context_tokens: state.cache_context_tokens,
            kv_bytes,
        }))
    }

    pub(super) fn snapshot_request_state_at_prefix(
        &self,
        state: &DsparkRequestState,
        prefix_tokens: usize,
        kv_bytes: Vec<u8>,
    ) -> Result<Option<DsparkRequestCacheSnapshot>> {
        anyhow::ensure!(
            state.pending_replay_id.is_none(),
            "cannot snapshot native dSpark request slot {} during replay {:?}",
            state.slot,
            state.pending_replay_id,
        );
        anyhow::ensure!(
            prefix_tokens <= state.context_tokens,
            "native dSpark reusable prefix {prefix_tokens} exceeds request context {}",
            state.context_tokens,
        );
        // The packed pages are a sliding ring at the current request frontier.
        // Once the request advances, those same bytes no longer describe an
        // arbitrary earlier prefix, so only the exact frontier is reusable.
        if prefix_tokens == 0
            || state.cache_context_tokens == 0
            || prefix_tokens != state.context_tokens
        {
            anyhow::ensure!(
                kv_bytes.is_empty(),
                "unavailable native dSpark prefix snapshot supplied {} KV bytes",
                kv_bytes.len(),
            );
            return Ok(None);
        }
        self.snapshot_request_state(state, kv_bytes)
    }

    pub(super) fn restore_request_state(
        &mut self,
        state: &mut DsparkRequestState,
        snapshot: &DsparkRequestCacheSnapshot,
    ) -> Result<()> {
        anyhow::ensure!(
            state.pending_replay_id.is_none(),
            "cannot restore native dSpark request slot {} during replay {:?}",
            state.slot,
            state.pending_replay_id,
        );
        anyhow::ensure!(
            snapshot.cache_context_tokens <= self.max_cache_context_tokens,
            "native dSpark snapshot retains {} tokens beyond its {}-token window",
            snapshot.cache_context_tokens,
            self.max_cache_context_tokens,
        );
        let expected_bytes = if snapshot.cache_context_tokens == 0 {
            0
        } else {
            self.kv_bytes_per_request
        };
        anyhow::ensure!(
            snapshot.kv_bytes.len() == expected_bytes,
            "native dSpark snapshot has {} bytes, expected {expected_bytes}",
            snapshot.kv_bytes.len(),
        );
        state.context_tokens = snapshot.context_tokens;
        state.cache_context_tokens = snapshot.cache_context_tokens;
        Ok(())
    }

    pub(super) fn prepare_replay(
        &mut self,
        state: &mut DsparkRequestState,
        target_hidden_taps: [&DeviceBf16Output; NATIVE_DSPARK_TARGET_TAPS],
        target_row_start: usize,
        committed_rows: usize,
        absolute_context_start: Option<usize>,
        anchor_token: usize,
    ) -> Result<DsparkPreparedReplay> {
        anyhow::ensure!(
            state.pending_replay_id.is_none(),
            "native dSpark request slot {} already has replay {:?} in flight",
            state.slot,
            state.pending_replay_id,
        );
        let replay = NativeDsparkReplayPlan::for_request(
            &self.execution_plan,
            state,
            target_row_start,
            committed_rows,
            absolute_context_start,
            anchor_token,
        )?;
        replay.validate_target_taps(&self.execution_plan, target_hidden_taps)?;
        let replay_id = self.next_replay_id;
        self.next_replay_id = self
            .next_replay_id
            .checked_add(1)
            .context("native dSpark replay ID overflow")?;
        state.pending_replay_id = Some(replay_id);
        Ok(DsparkPreparedReplay {
            replay_id,
            request_slot: state.slot,
            hidden_size: self.execution_plan.hidden_size,
            vocab_size: self.execution_plan.vocab_size,
            replay,
        })
    }

    pub(super) fn abort_replay(
        &mut self,
        state: &mut DsparkRequestState,
        prepared: &DsparkPreparedReplay,
    ) -> Result<()> {
        self.validate_pending_replay(state, prepared)?;
        state.pending_replay_id = None;
        Ok(())
    }

    pub(super) fn validate_replay_output(
        &self,
        state: &DsparkRequestState,
        prepared: &DsparkPreparedReplay,
        output: DsparkReplayOutput,
    ) -> Result<DsparkDraftStep> {
        self.validate_pending_replay(state, prepared)?;
        anyhow::ensure!(
            output.proposal_token_ids.len() == self.max_verify_drafts
                && output.conditional_confidence.len() == self.max_verify_drafts,
            "native dSpark replay returned {} proposals and {} confidences, expected {} each",
            output.proposal_token_ids.len(),
            output.conditional_confidence.len(),
            self.max_verify_drafts,
        );
        anyhow::ensure!(
            output
                .proposal_token_ids
                .iter()
                .all(|token_id| *token_id < self.execution_plan.vocab_size),
            "native dSpark replay returned a token outside vocabulary {}: {:?}",
            self.execution_plan.vocab_size,
            output.proposal_token_ids,
        );
        anyhow::ensure!(
            output
                .conditional_confidence
                .iter()
                .all(|confidence| confidence.is_finite() && (0.0..=1.0).contains(confidence)),
            "native dSpark replay returned invalid conditional confidence {:?}",
            output.conditional_confidence,
        );
        anyhow::ensure!(
            [
                output.update_ms,
                output.suffix_ms,
                output.readback_ms,
                output.total_ms,
            ]
            .into_iter()
            .all(|milliseconds| milliseconds.is_finite() && milliseconds >= 0.0),
            "native dSpark replay returned invalid timings update={} suffix={} readback={} total={}",
            output.update_ms,
            output.suffix_ms,
            output.readback_ms,
            output.total_ms,
        );
        anyhow::ensure!(
            state.context_tokens == prepared.replay.context_tokens_before,
            "native dSpark request slot {} advanced from context {} to {} during replay {}",
            state.slot,
            prepared.replay.context_tokens_before,
            state.context_tokens,
            prepared.replay_id,
        );
        Ok(DsparkDraftStep {
            context_tokens: prepared.replay.context_tokens_before,
            committed_rows: prepared.replay.committed_main_rows,
            anchor_token: prepared.replay.draft_input_token_ids[0],
            proposal_token_ids: output.proposal_token_ids,
            conditional_confidence: output.conditional_confidence,
            update_ms: output.update_ms,
            suffix_ms: output.suffix_ms,
            readback_ms: output.readback_ms,
            total_ms: output.total_ms,
        })
    }

    pub(super) fn commit_replay(
        &mut self,
        state: &mut DsparkRequestState,
        prepared: &DsparkPreparedReplay,
    ) -> Result<()> {
        self.validate_pending_replay(state, prepared)?;
        anyhow::ensure!(
            state.context_tokens == prepared.replay.context_tokens_before,
            "native dSpark request slot {} advanced from context {} to {} during replay {}",
            state.slot,
            prepared.replay.context_tokens_before,
            state.context_tokens,
            prepared.replay_id,
        );
        state.context_tokens = prepared.replay.main_context_end;
        state.cache_context_tokens = prepared.replay.cache_context_tokens_after;
        state.pending_replay_id = None;
        Ok(())
    }

    fn validate_pending_replay(
        &self,
        state: &DsparkRequestState,
        prepared: &DsparkPreparedReplay,
    ) -> Result<()> {
        anyhow::ensure!(
            state.slot == prepared.request_slot,
            "native dSpark replay {} belongs to slot {}, got slot {}",
            prepared.replay_id,
            prepared.request_slot,
            state.slot,
        );
        anyhow::ensure!(
            state.pending_replay_id == Some(prepared.replay_id),
            "native dSpark request slot {} expects replay {:?}, got {}",
            state.slot,
            state.pending_replay_id,
            prepared.replay_id,
        );
        Ok(())
    }

    pub(super) fn plan_verification(
        &self,
        step: &DsparkDraftStep,
        max_drafts: usize,
        confidence_logit_bias: f64,
        position_logit_bias: &[f64],
        confidence_context_tokens: usize,
        force_probe: bool,
        sps: &DsparkSpsProfile,
    ) -> Result<DsparkDraftPlan> {
        let max_drafts = max_drafts
            .min(self.max_verify_drafts)
            .min(step.proposal_token_ids.len());
        anyhow::ensure!(
            position_logit_bias.is_empty() || position_logit_bias.len() >= max_drafts,
            "dSpark position confidence bias has {} entries for {max_drafts} proposals",
            position_logit_bias.len(),
        );
        let raw_confidence = step
            .conditional_confidence
            .iter()
            .take(max_drafts)
            .copied()
            .collect::<Vec<_>>();
        let confidence = raw_confidence
            .iter()
            .copied()
            .enumerate()
            .map(|(position, value)| {
                apply_dspark_confidence_logit_bias(
                    f64::from(value),
                    confidence_logit_bias
                        + position_logit_bias.get(position).copied().unwrap_or(0.0),
                )
            })
            .collect::<Vec<_>>();
        let schedule = schedule_dspark_verification(
            &[confidence.clone()],
            sps,
            DsparkScheduleSearch::GlobalMaximum,
        )?;
        let selected_drafts = if force_probe && schedule.prefix_lengths[0] == 0 && max_drafts > 0 {
            1
        } else {
            schedule.prefix_lengths[0]
        };
        let (target_batch_rows, expected_committed_tokens, expected_tokens_per_second) =
            if selected_drafts == schedule.prefix_lengths[0] {
                (
                    schedule.target_batch_rows,
                    schedule.expected_committed_tokens,
                    schedule.expected_tokens_per_second,
                )
            } else {
                let expected_tokens = 1.0 + confidence[0];
                (2, expected_tokens, expected_tokens * sps.get(2)?)
            };
        Ok(DsparkDraftPlan {
            proposal_token_ids: step.proposal_token_ids[..selected_drafts].to_vec(),
            conditional_confidence: raw_confidence[..selected_drafts].to_vec(),
            candidate_proposal_token_ids: step.proposal_token_ids[..max_drafts].to_vec(),
            candidate_conditional_confidence: raw_confidence,
            candidate_adjusted_confidence: confidence,
            selected_drafts,
            minimum_drafts: usize::from(
                force_probe && schedule.prefix_lengths[0] == 0 && max_drafts > 0,
            ),
            target_batch_rows,
            expected_committed_tokens,
            expected_tokens_per_second,
            confidence_logit_bias,
            confidence_context_tokens,
            calibration_eligible: true,
        })
    }
}

unsafe impl Send for DsparkRequestEngine {}

#[derive(Clone, Debug)]
pub(super) struct DsparkDraftPlan {
    pub(super) proposal_token_ids: Vec<usize>,
    pub(super) conditional_confidence: Vec<f32>,
    pub(super) candidate_proposal_token_ids: Vec<usize>,
    pub(super) candidate_conditional_confidence: Vec<f32>,
    pub(super) candidate_adjusted_confidence: Vec<f64>,
    pub(super) selected_drafts: usize,
    pub(super) minimum_drafts: usize,
    pub(super) target_batch_rows: usize,
    pub(super) expected_committed_tokens: f64,
    pub(super) expected_tokens_per_second: f64,
    pub(super) confidence_logit_bias: f64,
    pub(super) confidence_context_tokens: usize,
    pub(super) calibration_eligible: bool,
}

const DSPARK_CONFIDENCE_CALIBRATION_WINDOW: usize = 16;
const DSPARK_CONFIDENCE_LOGIT_BIAS_LIMIT: f64 = 13.0;
const DSPARK_CONFIDENCE_PRIOR_PRECISION: f64 = 0.25;
const DSPARK_CONFIDENCE_MIN_RECENCY_DECAY: f64 = 0.70;
const DSPARK_CONFIDENCE_MAX_RECENCY_DECAY: f64 = 0.96;
const DSPARK_CONFIDENCE_MIN_PROBE_INTERVAL: usize = 4;
const DSPARK_CONFIDENCE_MAX_PROBE_INTERVAL: usize = 16;
const DSPARK_CONFIDENCE_RESIDUAL_POSITIONS: usize = 15;
const DSPARK_CONFIDENCE_RESIDUAL_GLOBAL_DECAY: f64 = 0.90;
const DSPARK_CONFIDENCE_RESIDUAL_GLOBAL_RATE: f64 = 0.40;
const DSPARK_CONFIDENCE_RESIDUAL_POSITION_DECAY: f64 = 0.98;
const DSPARK_CONFIDENCE_RESIDUAL_POSITION_RATE: f64 = 0.01;
const DSPARK_CONFIDENCE_RESIDUAL_BIAS_LIMIT: f64 = 4.0;
pub(super) const DSPARK_RUNTIME_CONTEXT_BUCKET_TOKENS: usize = 32 * 1024;
const DSPARK_RUNTIME_COST_EXACT_PRIOR_WEIGHT: f64 = 2.0;
const DSPARK_RUNTIME_COST_ROW_PRIOR_WEIGHT: f64 = 4.0;
const DSPARK_RUNTIME_COST_CONCURRENCY_PRIOR_WEIGHT: f64 = 4.0;
const DSPARK_RUNTIME_CONTEXT_BUCKET_PRIOR_MS: f64 = 2.0;
// Native dSpark starts with a monotonic batching prior and immediately learns
// complete-cycle costs online. Imported GLM timing measurements are not a
// valid prior for DeepSeek's integrated transformer/MoE blocks.
const DSPARK_NATIVE_BOOTSTRAP_BATCH_EXPONENT: f64 = 0.70;

#[derive(Clone, Debug)]
struct DsparkConfidenceObservation {
    conditional_confidence: Vec<f32>,
    accepted_drafts: usize,
}

#[derive(Clone, Debug, Default)]
pub(super) struct DsparkConfidenceCalibrator {
    observations: VecDeque<DsparkConfidenceObservation>,
    logit_bias: f64,
    posterior_variance: f64,
    consecutive_zero_draft_plans: usize,
}

impl DsparkConfidenceCalibrator {
    pub(super) fn reset(&mut self) {
        self.observations.clear();
        self.logit_bias = 0.0;
        self.posterior_variance = 1.0 / DSPARK_CONFIDENCE_PRIOR_PRECISION;
        self.consecutive_zero_draft_plans = 0;
    }

    pub(super) fn logit_bias(&self) -> f64 {
        self.logit_bias
    }

    pub(super) fn observation_cycles(&self) -> usize {
        self.observations.len()
    }

    pub(super) fn posterior_variance(&self) -> f64 {
        self.posterior_variance
    }

    pub(super) fn force_probe_due(&self) -> bool {
        let uncertainty = (self.posterior_variance / 4.0).clamp(0.0, 1.0);
        let interval = ((DSPARK_CONFIDENCE_MAX_PROBE_INTERVAL as f64)
            - uncertainty
                * (DSPARK_CONFIDENCE_MAX_PROBE_INTERVAL - DSPARK_CONFIDENCE_MIN_PROBE_INTERVAL)
                    as f64)
            .round() as usize;
        self.consecutive_zero_draft_plans >= interval.max(1)
    }

    pub(super) fn record_selected_drafts(&mut self, selected_drafts: usize) {
        if selected_drafts == 0 {
            self.consecutive_zero_draft_plans = self.consecutive_zero_draft_plans.saturating_add(1);
        } else {
            self.consecutive_zero_draft_plans = 0;
        }
    }

    pub(super) fn observe(&mut self, conditional_confidence: &[f32], accepted_drafts: usize) {
        if conditional_confidence.is_empty() || accepted_drafts > conditional_confidence.len() {
            return;
        }
        self.observations.push_back(DsparkConfidenceObservation {
            conditional_confidence: conditional_confidence.to_vec(),
            accepted_drafts,
        });
        while self.observations.len() > DSPARK_CONFIDENCE_CALIBRATION_WINDOW {
            self.observations.pop_front();
        }
        let fit = fit_dspark_confidence_logit_bias(&self.observations, self.logit_bias);
        self.logit_bias = fit.logit_bias;
        self.posterior_variance = fit.posterior_variance;
    }
}

/// Cheap request-local correction for the raw dSpark confidence chain.
///
/// Shadow-trace replay found that stream/category drift dominates a single
/// global calibration fit, while later proposal positions have only a small
/// repeatable residual.  The controller therefore maintains one fast pooled
/// residual and fifteen much slower position residuals.  Checkpoint metadata
/// supplies a low-capacity continuous context prior only when trace evidence
/// shows a systematic context drift.
#[derive(Clone, Debug)]
pub(super) struct DsparkConfidenceResidual {
    dynamic_bias: f64,
    position_bias: [f64; DSPARK_CONFIDENCE_RESIDUAL_POSITIONS],
    observation_cycles: usize,
}

impl Default for DsparkConfidenceResidual {
    fn default() -> Self {
        Self {
            dynamic_bias: 0.0,
            position_bias: [0.0; DSPARK_CONFIDENCE_RESIDUAL_POSITIONS],
            observation_cycles: 0,
        }
    }
}

impl DsparkConfidenceResidual {
    pub(super) fn reset(&mut self) {
        *self = Self::default();
    }

    fn context_prior(_context_tokens: usize) -> f64 {
        // DeepSeek's integrated checkpoint supplies its own confidence head.
        // External-checkpoint context priors are intentionally not inherited.
        0.0
    }

    pub(super) fn global_logit_bias(&self, context_tokens: usize) -> f64 {
        Self::context_prior(context_tokens) + self.dynamic_bias
    }

    pub(super) fn position_logit_bias(&self) -> &[f64] {
        &self.position_bias
    }

    pub(super) fn observation_cycles(&self) -> usize {
        self.observation_cycles
    }

    pub(super) fn record_selected_drafts(&mut self, selected_drafts: usize) {
        if selected_drafts == 0 {
            self.dynamic_bias *= DSPARK_CONFIDENCE_RESIDUAL_GLOBAL_DECAY;
        }
    }

    pub(super) fn observe(
        &mut self,
        conditional_confidence: &[f32],
        accepted_drafts: usize,
        context_tokens: usize,
    ) {
        if conditional_confidence.is_empty()
            || conditional_confidence.len() > self.position_bias.len()
            || accepted_drafts > conditional_confidence.len()
        {
            return;
        }
        let observed_positions = if accepted_drafts < conditional_confidence.len() {
            accepted_drafts + 1
        } else {
            accepted_drafts
        };
        if observed_positions == 0 {
            return;
        }
        let global_bias = self.global_logit_bias(context_tokens);
        let mut error_sum = 0.0;
        for (position, raw_probability) in conditional_confidence
            .iter()
            .copied()
            .take(observed_positions)
            .enumerate()
        {
            let predicted = apply_dspark_confidence_logit_bias(
                f64::from(raw_probability),
                global_bias + self.position_bias[position],
            );
            let outcome = f64::from(position < accepted_drafts);
            let error = outcome - predicted;
            error_sum += error;
            self.position_bias[position] = (DSPARK_CONFIDENCE_RESIDUAL_POSITION_DECAY
                * self.position_bias[position]
                + DSPARK_CONFIDENCE_RESIDUAL_POSITION_RATE * error)
                .clamp(
                    -DSPARK_CONFIDENCE_RESIDUAL_BIAS_LIMIT,
                    DSPARK_CONFIDENCE_RESIDUAL_BIAS_LIMIT,
                );
        }
        let mean_error = error_sum / observed_positions as f64;
        self.dynamic_bias = (DSPARK_CONFIDENCE_RESIDUAL_GLOBAL_DECAY * self.dynamic_bias
            + DSPARK_CONFIDENCE_RESIDUAL_GLOBAL_RATE * mean_error)
            .clamp(
                -DSPARK_CONFIDENCE_RESIDUAL_BIAS_LIMIT,
                DSPARK_CONFIDENCE_RESIDUAL_BIAS_LIMIT,
            );
        self.observation_cycles = self.observation_cycles.saturating_add(1);
    }
}

#[derive(Clone, Copy, Debug)]
struct DsparkConfidenceFit {
    logit_bias: f64,
    posterior_variance: f64,
}

fn clamp_dspark_probability(probability: f64) -> f64 {
    probability.clamp(1.0e-6, 1.0 - 1.0e-6)
}

fn apply_dspark_confidence_logit_bias(probability: f64, logit_bias: f64) -> f64 {
    let probability = clamp_dspark_probability(probability);
    let logit = (probability / (1.0 - probability)).ln() + logit_bias;
    if logit >= 0.0 {
        1.0 / (1.0 + (-logit).exp())
    } else {
        let exponential = logit.exp();
        exponential / (1.0 + exponential)
    }
}

fn fit_dspark_confidence_logit_bias(
    observations: &VecDeque<DsparkConfidenceObservation>,
    previous_bias: f64,
) -> DsparkConfidenceFit {
    let newest_surprise = observations.back().map_or(0.0, |observation| {
        let observed_positions =
            if observation.accepted_drafts < observation.conditional_confidence.len() {
                observation.accepted_drafts + 1
            } else {
                observation.accepted_drafts
            };
        let (error, count) = observation
            .conditional_confidence
            .iter()
            .copied()
            .take(observed_positions)
            .enumerate()
            .fold((0.0, 0_usize), |(error, count), (position, raw)| {
                let predicted = apply_dspark_confidence_logit_bias(f64::from(raw), previous_bias);
                let outcome = f64::from(position < observation.accepted_drafts);
                (error + (predicted - outcome).abs(), count + 1)
            });
        if count == 0 {
            0.0
        } else {
            error / count as f64
        }
    });
    let recency_decay = (DSPARK_CONFIDENCE_MAX_RECENCY_DECAY
        - newest_surprise
            * (DSPARK_CONFIDENCE_MAX_RECENCY_DECAY - DSPARK_CONFIDENCE_MIN_RECENCY_DECAY))
        .clamp(
            DSPARK_CONFIDENCE_MIN_RECENCY_DECAY,
            DSPARK_CONFIDENCE_MAX_RECENCY_DECAY,
        );
    let evaluate = |bias: f64| {
        let mut gradient = DSPARK_CONFIDENCE_PRIOR_PRECISION * bias;
        let mut curvature = DSPARK_CONFIDENCE_PRIOR_PRECISION;
        for (age, observation) in observations.iter().rev().enumerate() {
            let weight = recency_decay.powi(age as i32);
            let observed_positions =
                if observation.accepted_drafts < observation.conditional_confidence.len() {
                    observation.accepted_drafts + 1
                } else {
                    observation.accepted_drafts
                };
            for (position, raw_probability) in observation
                .conditional_confidence
                .iter()
                .copied()
                .take(observed_positions)
                .enumerate()
            {
                let calibrated =
                    apply_dspark_confidence_logit_bias(f64::from(raw_probability), bias);
                let outcome = if position < observation.accepted_drafts {
                    1.0
                } else {
                    0.0
                };
                gradient += weight * (calibrated - outcome);
                curvature += weight * calibrated * (1.0 - calibrated);
            }
        }
        (gradient, curvature)
    };
    // The one-dimensional logistic objective is convex. Bisection is both
    // cheaper and more robust than an undamped Newton step when a stream
    // temporarily has near-perfect matches or misses at p≈0/1.
    let mut lower = -DSPARK_CONFIDENCE_LOGIT_BIAS_LIMIT;
    let mut upper = DSPARK_CONFIDENCE_LOGIT_BIAS_LIMIT;
    let lower_gradient = evaluate(lower).0;
    let upper_gradient = evaluate(upper).0;
    let bias = if lower_gradient >= 0.0 {
        lower
    } else if upper_gradient <= 0.0 {
        upper
    } else {
        for _ in 0..40 {
            let midpoint = 0.5 * (lower + upper);
            if evaluate(midpoint).0 < 0.0 {
                lower = midpoint;
            } else {
                upper = midpoint;
            }
        }
        0.5 * (lower + upper)
    };
    let final_curvature = evaluate(bias).1;
    DsparkConfidenceFit {
        logit_bias: bias,
        posterior_variance: 1.0 / final_curvature.max(DSPARK_CONFIDENCE_PRIOR_PRECISION),
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct DsparkRuntimeCostCell {
    samples: u64,
    mean_value: f64,
    value_m2: f64,
}

impl DsparkRuntimeCostCell {
    fn observe(&mut self, value: f64) {
        self.samples = self.samples.saturating_add(1);
        let delta = value - self.mean_value;
        self.mean_value += delta / self.samples as f64;
        let delta_after = value - self.mean_value;
        self.value_m2 += delta * delta_after;
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct DsparkRuntimeCostKey {
    request_count: usize,
    context_work_bucket: usize,
    max_context_bucket: usize,
    target_rows: usize,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct DsparkRuntimeGlobalCostKey {
    request_count: usize,
    target_rows: usize,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct DsparkRuntimeCostObservation {
    pub(super) request_count: usize,
    pub(super) context_work_bucket: usize,
    pub(super) max_context_bucket: usize,
    pub(super) target_rows: usize,
    pub(super) observed_ms: f64,
    pub(super) predicted_ms_before: f64,
    pub(super) exact_samples: u64,
}

#[derive(Clone, Debug)]
pub(super) struct DsparkRuntimeCostModel {
    max_requests: usize,
    max_drafts_per_request: usize,
    exact: BTreeMap<DsparkRuntimeCostKey, DsparkRuntimeCostCell>,
    row: BTreeMap<DsparkRuntimeGlobalCostKey, DsparkRuntimeCostCell>,
    concurrency: BTreeMap<usize, DsparkRuntimeCostCell>,
    profiled_ms: BTreeMap<DsparkRuntimeGlobalCostKey, f64>,
}

impl DsparkRuntimeCostModel {
    pub(super) fn new(max_requests: usize, max_drafts_per_request: usize) -> Result<Self> {
        anyhow::ensure!(max_requests > 0, "dSpark cost model requires requests");
        anyhow::ensure!(
            max_drafts_per_request > 0,
            "native dSpark cost model requires at least one proposal"
        );
        Ok(Self {
            max_requests,
            max_drafts_per_request,
            exact: BTreeMap::new(),
            row: BTreeMap::new(),
            concurrency: BTreeMap::new(),
            profiled_ms: BTreeMap::new(),
        })
    }

    pub(super) fn install_profile(
        &mut self,
        request_count: usize,
        rows: &[(usize, f64)],
    ) -> Result<()> {
        let max_rows = self.max_target_rows(request_count)?;
        anyhow::ensure!(
            rows.len() == max_rows - request_count + 1,
            "dSpark SPS profile for {request_count} requests has {} rows, expected {}",
            rows.len(),
            max_rows - request_count + 1,
        );
        for (expected_rows, (target_rows, observed_ms)) in
            (request_count..=max_rows).zip(rows.iter().copied())
        {
            anyhow::ensure!(
                target_rows == expected_rows,
                "dSpark SPS profile for {request_count} requests expected target row {expected_rows}, got {target_rows}",
            );
            anyhow::ensure!(
                observed_ms.is_finite() && observed_ms > 0.0,
                "invalid profiled dSpark latency {observed_ms} ms at request count {request_count}, target rows {target_rows}",
            );
            self.profiled_ms.insert(
                DsparkRuntimeGlobalCostKey {
                    request_count,
                    target_rows,
                },
                observed_ms,
            );
        }
        Ok(())
    }

    fn has_complete_profile(&self, request_count: usize) -> Result<bool> {
        let max_rows = self.max_target_rows(request_count)?;
        Ok((request_count..=max_rows).all(|target_rows| {
            self.profiled_ms.contains_key(&DsparkRuntimeGlobalCostKey {
                request_count,
                target_rows,
            })
        }))
    }

    pub(super) fn context_buckets(context_tokens: &[usize]) -> Result<(usize, usize)> {
        let max_context = context_tokens
            .iter()
            .copied()
            .max()
            .context("dSpark cost model requires at least one context")?;
        let context_work = context_tokens
            .iter()
            .try_fold(0_usize, |total, tokens| total.checked_add(*tokens))
            .context("dSpark aggregate attention context overflow")?;
        Ok((
            context_work / DSPARK_RUNTIME_CONTEXT_BUCKET_TOKENS,
            max_context / DSPARK_RUNTIME_CONTEXT_BUCKET_TOKENS,
        ))
    }

    fn max_target_rows(&self, request_count: usize) -> Result<usize> {
        anyhow::ensure!(
            (1..=self.max_requests).contains(&request_count),
            "dSpark cost request count {request_count} is outside 1..={}",
            self.max_requests,
        );
        request_count
            .checked_mul(self.max_drafts_per_request + 1)
            .context("dSpark runtime cost row limit overflow")
    }

    fn balanced_prior_ms(target_rows: usize, context_bucket: usize) -> Result<f64> {
        anyhow::ensure!(
            target_rows > 0,
            "dSpark physical target rows must be positive"
        );
        let short_context_ms = (target_rows as f64).powf(DSPARK_NATIVE_BOOTSTRAP_BATCH_EXPONENT);
        Ok(short_context_ms + context_bucket as f64 * DSPARK_RUNTIME_CONTEXT_BUCKET_PRIOR_MS)
    }

    fn predicted_ms_for_bucket(
        &self,
        request_count: usize,
        context_work_bucket: usize,
        max_context_bucket: usize,
        target_rows: usize,
    ) -> Result<f64> {
        let max_rows = self.max_target_rows(request_count)?;
        anyhow::ensure!(
            (request_count..=max_rows).contains(&target_rows),
            "dSpark target rows {target_rows} are outside {request_count}..={max_rows}",
        );
        let row_key = DsparkRuntimeGlobalCostKey {
            request_count,
            target_rows,
        };
        if let Some(profiled_ms) = self.profiled_ms.get(&row_key).copied() {
            return Ok((profiled_ms
                + context_work_bucket as f64 * DSPARK_RUNTIME_CONTEXT_BUCKET_PRIOR_MS)
                .max(0.001));
        }
        let prior = Self::balanced_prior_ms(target_rows, context_work_bucket)?;
        let concurrency = self
            .concurrency
            .get(&request_count)
            .copied()
            .unwrap_or_default();
        let concurrency_weight = concurrency.samples as f64
            / (concurrency.samples as f64 + DSPARK_RUNTIME_COST_CONCURRENCY_PRIOR_WEIGHT);
        let mut log_ratio = concurrency_weight * concurrency.mean_value;
        let row = self.row.get(&row_key).copied().unwrap_or_default();
        let row_weight =
            row.samples as f64 / (row.samples as f64 + DSPARK_RUNTIME_COST_ROW_PRIOR_WEIGHT);
        log_ratio = (1.0 - row_weight) * log_ratio + row_weight * row.mean_value;
        let exact_key = DsparkRuntimeCostKey {
            request_count,
            context_work_bucket,
            max_context_bucket,
            target_rows,
        };
        let exact = self.exact.get(&exact_key).copied().unwrap_or_default();
        let exact_weight =
            exact.samples as f64 / (exact.samples as f64 + DSPARK_RUNTIME_COST_EXACT_PRIOR_WEIGHT);
        log_ratio = (1.0 - exact_weight) * log_ratio + exact_weight * exact.mean_value;
        Ok((prior * log_ratio.exp()).max(0.001))
    }

    pub(super) fn profile(
        &self,
        request_count: usize,
        context_tokens: &[usize],
    ) -> Result<DsparkSpsProfile> {
        anyhow::ensure!(
            context_tokens.len() == request_count,
            "dSpark cost profile has {} contexts for {request_count} requests",
            context_tokens.len(),
        );
        let (context_work_bucket, max_context_bucket) = Self::context_buckets(context_tokens)?;
        let max_rows = self.max_target_rows(request_count)?;
        let mut steps_per_second = vec![0.0; max_rows + 1];
        let mut previous_ms = 0.0_f64;
        for (target_rows, value) in steps_per_second.iter_mut().enumerate().skip(request_count) {
            // A larger verification pack cannot complete before all work in
            // its smaller prefix. Runtime noise in sparsely sampled cells must
            // not manufacture a false latency dip that attracts the global
            // scheduler.
            let predicted_ms = self
                .predicted_ms_for_bucket(
                    request_count,
                    context_work_bucket,
                    max_context_bucket,
                    target_rows,
                )?
                .max(previous_ms);
            previous_ms = predicted_ms;
            *value = 1_000.0 / predicted_ms;
        }
        for target_rows in 1..request_count {
            steps_per_second[target_rows] = steps_per_second[request_count];
        }
        DsparkSpsProfile::new(steps_per_second)
    }

    pub(super) fn observe(
        &mut self,
        request_count: usize,
        context_tokens: &[usize],
        target_rows: usize,
        observed_ms: f64,
    ) -> Result<DsparkRuntimeCostObservation> {
        anyhow::ensure!(
            observed_ms.is_finite() && observed_ms > 0.0,
            "invalid dSpark runtime cost observation {observed_ms}",
        );
        let (context_work_bucket, max_context_bucket) = Self::context_buckets(context_tokens)?;
        let predicted_ms_before = self.predicted_ms_for_bucket(
            request_count,
            context_work_bucket,
            max_context_bucket,
            target_rows,
        )?;
        if self.has_complete_profile(request_count)? {
            return Ok(DsparkRuntimeCostObservation {
                request_count,
                context_work_bucket,
                max_context_bucket,
                target_rows,
                observed_ms,
                predicted_ms_before,
                exact_samples: 0,
            });
        }
        // Preserve real regime changes while preventing one host stall from
        // permanently poisoning a rarely visited row/context cell.
        let robust_observation =
            observed_ms.clamp(predicted_ms_before * 0.25, predicted_ms_before * 4.0);
        let prior = Self::balanced_prior_ms(target_rows, context_work_bucket)?;
        let log_ratio = (robust_observation / prior).ln();
        let exact_key = DsparkRuntimeCostKey {
            request_count,
            context_work_bucket,
            max_context_bucket,
            target_rows,
        };
        let exact = self.exact.entry(exact_key).or_default();
        exact.observe(log_ratio);
        let exact_samples = exact.samples;
        self.row
            .entry(DsparkRuntimeGlobalCostKey {
                request_count,
                target_rows,
            })
            .or_default()
            .observe(log_ratio);
        self.concurrency
            .entry(request_count)
            .or_default()
            .observe(log_ratio);
        Ok(DsparkRuntimeCostObservation {
            request_count,
            context_work_bucket,
            max_context_bucket,
            target_rows,
            observed_ms,
            predicted_ms_before,
            exact_samples,
        })
    }
}

impl DsparkDraftPlan {
    pub(super) fn calibrated_candidate_confidence(&self, max_drafts: usize) -> Vec<f64> {
        self.candidate_adjusted_confidence
            .iter()
            .copied()
            .take(max_drafts)
            .collect()
    }

    pub(super) fn apply_joint_selection(
        &mut self,
        selected_drafts: usize,
        target_batch_rows: usize,
        expected_committed_tokens: f64,
        expected_tokens_per_second: f64,
    ) -> Result<()> {
        anyhow::ensure!(
            selected_drafts >= self.minimum_drafts
                && selected_drafts <= self.candidate_proposal_token_ids.len()
                && selected_drafts <= self.candidate_conditional_confidence.len(),
            "joint dSpark selection {selected_drafts} is outside minimum {} and candidate proposal/confidence lengths {}/{}",
            self.minimum_drafts,
            self.candidate_proposal_token_ids.len(),
            self.candidate_conditional_confidence.len(),
        );
        self.proposal_token_ids = self.candidate_proposal_token_ids[..selected_drafts].to_vec();
        self.conditional_confidence =
            self.candidate_conditional_confidence[..selected_drafts].to_vec();
        self.selected_drafts = selected_drafts;
        self.target_batch_rows = target_batch_rows;
        self.expected_committed_tokens = expected_committed_tokens;
        self.expected_tokens_per_second = expected_tokens_per_second;
        Ok(())
    }
}

pub(super) struct DsparkSpsProfile {
    /// Index is the total target verification row count. Index zero is unused.
    steps_per_second: Vec<f64>,
}

impl DsparkSpsProfile {
    fn new(steps_per_second: Vec<f64>) -> Result<Self> {
        anyhow::ensure!(
            steps_per_second.len() >= 2,
            "dSpark SPS profile must cover at least target batch size one"
        );
        anyhow::ensure!(
            steps_per_second[0] == 0.0,
            "dSpark SPS profile index zero must be an unused zero"
        );
        for (batch_rows, value) in steps_per_second.iter().copied().enumerate().skip(1) {
            anyhow::ensure!(
                value.is_finite() && value > 0.0,
                "invalid dSpark SPS value {value} at target batch size {batch_rows}"
            );
        }
        Ok(Self { steps_per_second })
    }

    pub(super) fn get(&self, batch_rows: usize) -> Result<f64> {
        self.steps_per_second
            .get(batch_rows)
            .copied()
            .with_context(|| format!("dSpark SPS profile does not cover batch size {batch_rows}"))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum DsparkScheduleSearch {
    /// Algorithm 1: stop at the first non-improving admission.
    CausalEarlyStop,
    /// Search every prefix when measured kernel/reduction costs are jagged.
    GlobalMaximum,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct DsparkVerificationSchedule {
    pub(super) prefix_lengths: Vec<usize>,
    pub(super) target_batch_rows: usize,
    pub(super) expected_committed_tokens: f64,
    pub(super) expected_tokens_per_second: f64,
}

#[derive(Clone, Copy, Debug)]
struct DsparkPrefixCandidate {
    request_index: usize,
    prefix_length: usize,
    survival_probability: f64,
}

pub(super) fn schedule_dspark_verification(
    conditional_confidence: &[Vec<f64>],
    sps: &DsparkSpsProfile,
    search: DsparkScheduleSearch,
) -> Result<DsparkVerificationSchedule> {
    schedule_dspark_verification_with_minimums(
        conditional_confidence,
        &vec![0; conditional_confidence.len()],
        sps,
        search,
    )
}

pub(super) fn schedule_dspark_verification_with_minimums(
    conditional_confidence: &[Vec<f64>],
    minimum_prefix_lengths: &[usize],
    sps: &DsparkSpsProfile,
    search: DsparkScheduleSearch,
) -> Result<DsparkVerificationSchedule> {
    anyhow::ensure!(
        !conditional_confidence.is_empty(),
        "dSpark scheduling requires an active request"
    );
    let request_count = conditional_confidence.len();
    anyhow::ensure!(
        minimum_prefix_lengths.len() == request_count,
        "dSpark scheduling has {} minimum widths for {request_count} requests",
        minimum_prefix_lengths.len(),
    );
    let max_target_rows =
        request_count + conditional_confidence.iter().map(Vec::len).sum::<usize>();
    sps.get(max_target_rows)?;

    let mut candidates = Vec::new();
    let mut mandatory_expected_tokens = request_count as f64;
    for (request_index, confidence) in conditional_confidence.iter().enumerate() {
        anyhow::ensure!(
            minimum_prefix_lengths[request_index] <= confidence.len(),
            "dSpark minimum prefix {} exceeds request {request_index} confidence length {}",
            minimum_prefix_lengths[request_index],
            confidence.len(),
        );
        let mut survival = 1.0;
        for (position, value) in confidence.iter().copied().enumerate() {
            anyhow::ensure!(
                value.is_finite() && (0.0..=1.0).contains(&value),
                "invalid dSpark confidence {value} for request {request_index} position {position}"
            );
            survival *= value;
            if position < minimum_prefix_lengths[request_index] {
                mandatory_expected_tokens += survival;
            } else if survival > 0.0 {
                candidates.push(DsparkPrefixCandidate {
                    request_index,
                    prefix_length: position + 1,
                    survival_probability: survival,
                });
            }
        }
    }
    candidates.sort_by(|left, right| {
        right
            .survival_probability
            .partial_cmp(&left.survival_probability)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.prefix_length.cmp(&right.prefix_length))
            .then_with(|| left.request_index.cmp(&right.request_index))
    });

    let mut current_lengths = minimum_prefix_lengths.to_vec();
    let mut best_lengths = current_lengths.clone();
    let mut target_batch_rows = request_count + minimum_prefix_lengths.iter().sum::<usize>();
    let mut expected_committed_tokens = mandatory_expected_tokens;
    let mut best_throughput = expected_committed_tokens * sps.get(target_batch_rows)?;
    let mut best_expected_tokens = expected_committed_tokens;
    let mut best_target_rows = target_batch_rows;

    for candidate in candidates {
        anyhow::ensure!(
            candidate.prefix_length == current_lengths[candidate.request_index] + 1,
            "dSpark confidence ordering violated prefix dependency for request {}: next {}, candidate {}",
            candidate.request_index,
            current_lengths[candidate.request_index] + 1,
            candidate.prefix_length
        );
        current_lengths[candidate.request_index] = candidate.prefix_length;
        target_batch_rows += 1;
        expected_committed_tokens += candidate.survival_probability;
        let throughput = expected_committed_tokens * sps.get(target_batch_rows)?;
        if throughput > best_throughput {
            best_throughput = throughput;
            best_expected_tokens = expected_committed_tokens;
            best_target_rows = target_batch_rows;
            best_lengths.clone_from(&current_lengths);
        } else if search == DsparkScheduleSearch::CausalEarlyStop {
            break;
        }
    }

    Ok(DsparkVerificationSchedule {
        prefix_lengths: best_lengths,
        target_batch_rows: best_target_rows,
        expected_committed_tokens: best_expected_tokens,
        expected_tokens_per_second: best_throughput,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_contract_uses_native_flash_taps_and_width() {
        assert_eq!(dspark_target_hidden_tap_layer_ids(), [41, 42, 43]);
        assert_eq!(dspark_active_max_verify_drafts(), 5);
    }

    #[test]
    fn native_execution_plan_maps_storage_names_to_physical_tp4_blocks() {
        let plan = NativeDsparkExecutionPlan::from_facts(&ModelFacts::default()).unwrap();
        assert_eq!(plan.target_taps, [41, 42, 43]);
        assert_eq!(plan.proposal_tokens, 5);
        assert_eq!(plan.sliding_window, 128);
        assert_eq!(plan.noise_token_id, 128_799);
        assert_eq!(plan.hidden_size, 4_096);
        assert_eq!(plan.vocab_size, 129_280);
        assert_eq!(
            plan.expert_topology,
            NativeDsparkExpertTopology::StrictTensorParallel4
        );
        assert_eq!(
            plan.blocks,
            vec![
                NativeDsparkBlockPlan {
                    block_index: 0,
                    physical_layer_id: 43,
                    target_layer_id: 40,
                    storage_prefix: "mtp.0".to_owned(),
                },
                NativeDsparkBlockPlan {
                    block_index: 1,
                    physical_layer_id: 44,
                    target_layer_id: 41,
                    storage_prefix: "mtp.1".to_owned(),
                },
                NativeDsparkBlockPlan {
                    block_index: 2,
                    physical_layer_id: 45,
                    target_layer_id: 42,
                    storage_prefix: "mtp.2".to_owned(),
                },
            ]
        );
    }

    #[test]
    fn pro_execution_plan_uses_model_taps_and_strict_tp4() {
        let mut facts = ModelFacts::default();
        facts.variant = ds4rt_core::ModelVariant::Pro;
        facts.hidden_size = ds4rt_core::DS4_PRO_HIDDEN_SIZE;
        facts.num_hidden_layers = ds4rt_core::DS4_PRO_NUM_HIDDEN_LAYERS;
        facts.routed_experts = ds4rt_core::DS4_PRO_ROUTED_EXPERTS;
        facts.moe_intermediate_size = ds4rt_core::DS4_PRO_MOE_INTERMEDIATE_SIZE;
        facts.attention_heads = 128;
        facts.q_lora_rank = ds4rt_core::DS4_PRO_Q_LORA_RANK;
        facts.compress_ratios = ds4rt_core::DS4_PRO_COMPRESS_RATIOS.to_vec();
        facts.dspark_markov_rank = ds4rt_core::DS4_PRO_DSPARK_MARKOV_RANK;
        facts.dspark_target_layer_ids = vec![58, 59, 60];

        let plan = NativeDsparkExecutionPlan::from_facts(&facts).unwrap();
        assert_eq!(plan.target_taps, [59, 60, 61]);
        assert_eq!(plan.hidden_size, ds4rt_core::DS4_PRO_HIDDEN_SIZE);
        assert_eq!(
            plan.expert_topology,
            NativeDsparkExpertTopology::StrictTensorParallel4
        );
        assert_eq!(
            plan.blocks
                .iter()
                .map(|block| (block.physical_layer_id, block.target_layer_id))
                .collect::<Vec<_>>(),
            [(61, 58), (62, 59), (63, 60)]
        );
    }

    #[test]
    fn replay_plan_uses_anchor_plus_noise_and_absolute_prompt_tail() {
        let execution = NativeDsparkExecutionPlan::from_facts(&ModelFacts::default()).unwrap();
        let state = DsparkRequestState {
            slot: 0,
            context_tokens: 0,
            cache_context_tokens: 0,
            pending_replay_id: None,
        };
        let replay = NativeDsparkReplayPlan::for_request(
            &execution,
            &state,
            0,
            execution.sliding_window,
            Some(8_192 - execution.sliding_window),
            42,
        )
        .unwrap();
        assert_eq!(replay.context_tokens_before, 0);
        assert_eq!(replay.main_context_start, 8_064);
        assert_eq!(replay.main_context_end, 8_192);
        assert_eq!(replay.cache_context_tokens_after, 128);
        assert_eq!(replay.cache_window_start, 8_064);
        assert_eq!(replay.proposal_position_start, 8_192);
        assert_eq!(replay.proposal_position_end, 8_197);
        assert_eq!(
            replay.draft_input_token_ids,
            [42, 128_799, 128_799, 128_799, 128_799]
        );
        assert_eq!(
            replay.expert_topology,
            NativeDsparkExpertTopology::StrictTensorParallel4
        );
        assert_eq!(
            replay
                .blocks
                .iter()
                .map(|block| block.storage_prefix.as_str())
                .collect::<Vec<_>>(),
            ["mtp.0", "mtp.1", "mtp.2"]
        );
    }

    #[test]
    fn replay_plan_advances_incremental_swa_without_recurrent_mtp_state() {
        let execution = NativeDsparkExecutionPlan::from_facts(&ModelFacts::default()).unwrap();
        let state = DsparkRequestState {
            slot: 0,
            context_tokens: 8_192,
            cache_context_tokens: 128,
            pending_replay_id: None,
        };
        let replay =
            NativeDsparkReplayPlan::for_request(&execution, &state, 2, 3, None, 7).unwrap();
        assert_eq!(replay.main_context_start, 8_192);
        assert_eq!(replay.main_context_end, 8_195);
        assert_eq!(replay.committed_main_rows, 3);
        assert_eq!(replay.target_row_start, 2);
        assert_eq!(replay.cache_context_tokens_after, 128);
        assert_eq!(replay.cache_window_start, 8_067);
        assert_eq!(replay.proposal_position_start, 8_195);
        assert_eq!(replay.proposal_position_end, 8_200);
    }

    #[test]
    fn prepared_replay_maps_target_ring_and_noncausal_proposals_into_one_page() {
        let execution = NativeDsparkExecutionPlan::from_facts(&ModelFacts::default()).unwrap();
        let replay = NativeDsparkReplayPlan::for_request(
            &execution,
            &DsparkRequestState {
                slot: 2,
                context_tokens: 8_192,
                cache_context_tokens: 128,
                pending_replay_id: None,
            },
            0,
            1,
            None,
            42,
        )
        .unwrap();
        let prepared = DsparkPreparedReplay {
            replay_id: 1,
            request_slot: 2,
            hidden_size: execution.hidden_size,
            vocab_size: execution.vocab_size,
            replay,
        };

        let selection = prepared.attention_physical_selection().unwrap();
        assert_eq!(&selection[..128], &(512_i32..640).collect::<Vec<_>>());
        assert_eq!(&selection[128..], &[640, 641, 642, 643, 644]);
        assert_eq!(selection.len(), 133);
    }

    #[test]
    fn two_phase_replay_commits_context_only_after_numeric_output() {
        let execution = NativeDsparkExecutionPlan::from_facts(&ModelFacts::default()).unwrap();
        let replay = NativeDsparkReplayPlan::for_request(
            &execution,
            &DsparkRequestState {
                slot: 3,
                context_tokens: 8_192,
                cache_context_tokens: 128,
                pending_replay_id: None,
            },
            0,
            1,
            None,
            42,
        )
        .unwrap();
        let prepared = DsparkPreparedReplay {
            replay_id: 17,
            request_slot: 3,
            hidden_size: execution.hidden_size,
            vocab_size: execution.vocab_size,
            replay,
        };
        let mut engine = DsparkRequestEngine {
            execution_plan: execution,
            max_verify_drafts: 5,
            max_cache_context_tokens: 128,
            kv_bytes_per_request: 3 * NATIVE_DSPARK_CACHE_PAGE_BYTES,
            free_slots: vec![],
            next_replay_id: 18,
        };
        let mut state = DsparkRequestState {
            slot: 3,
            context_tokens: 8_192,
            cache_context_tokens: 128,
            pending_replay_id: Some(17),
        };
        let step = engine
            .validate_replay_output(
                &state,
                &prepared,
                DsparkReplayOutput {
                    proposal_token_ids: vec![10, 11, 12, 13, 14],
                    conditional_confidence: vec![0.9, 0.8, 0.7, 0.6, 0.5],
                    update_ms: 1.0,
                    suffix_ms: 2.0,
                    readback_ms: 0.5,
                    total_ms: 3.5,
                },
            )
            .unwrap();
        assert_eq!(state.context_tokens, 8_192);
        assert_eq!(state.pending_replay_id, Some(17));
        engine.commit_replay(&mut state, &prepared).unwrap();
        assert_eq!(step.context_tokens, 8_192);
        assert_eq!(step.committed_rows, 1);
        assert_eq!(step.anchor_token, 42);
        assert_eq!(step.proposal_token_ids, [10, 11, 12, 13, 14]);
        assert_eq!(state.context_tokens, 8_193);
        assert_eq!(state.cache_context_tokens, 128);
        assert_eq!(state.pending_replay_id, None);
    }

    #[test]
    fn two_phase_replay_abort_preserves_request_frontier() {
        let execution = NativeDsparkExecutionPlan::from_facts(&ModelFacts::default()).unwrap();
        let replay = NativeDsparkReplayPlan::for_request(
            &execution,
            &DsparkRequestState {
                slot: 1,
                context_tokens: 256,
                cache_context_tokens: 128,
                pending_replay_id: None,
            },
            0,
            1,
            None,
            7,
        )
        .unwrap();
        let prepared = DsparkPreparedReplay {
            replay_id: 23,
            request_slot: 1,
            hidden_size: execution.hidden_size,
            vocab_size: execution.vocab_size,
            replay,
        };
        let mut engine = DsparkRequestEngine {
            execution_plan: execution,
            max_verify_drafts: 5,
            max_cache_context_tokens: 128,
            kv_bytes_per_request: 3 * NATIVE_DSPARK_CACHE_PAGE_BYTES,
            free_slots: vec![],
            next_replay_id: 24,
        };
        let mut state = DsparkRequestState {
            slot: 1,
            context_tokens: 256,
            cache_context_tokens: 128,
            pending_replay_id: Some(23),
        };
        engine.abort_replay(&mut state, &prepared).unwrap();
        assert_eq!(state.context_tokens, 256);
        assert_eq!(state.cache_context_tokens, 128);
        assert_eq!(state.pending_replay_id, None);
    }

    #[test]
    fn request_snapshot_owns_three_complete_packed_pages() {
        let execution = NativeDsparkExecutionPlan::from_facts(&ModelFacts::default()).unwrap();
        let packed_request_bytes = 3 * NATIVE_DSPARK_CACHE_PAGE_BYTES;
        let mut engine = DsparkRequestEngine {
            execution_plan: execution,
            max_verify_drafts: 5,
            max_cache_context_tokens: 128,
            kv_bytes_per_request: packed_request_bytes,
            free_slots: vec![],
            next_replay_id: 1,
        };
        let mut state = DsparkRequestState {
            slot: 0,
            context_tokens: 0,
            cache_context_tokens: 0,
            pending_replay_id: None,
        };
        let snapshot = DsparkRequestCacheSnapshot {
            context_tokens: 100,
            cache_context_tokens: 100,
            kv_bytes: vec![0; packed_request_bytes],
        };

        engine.restore_request_state(&mut state, &snapshot).unwrap();
        assert_eq!(state.context_tokens, 100);
        assert_eq!(state.cache_context_tokens, 100);
        let roundtrip = engine
            .snapshot_request_state(&state, snapshot.kv_bytes.clone())
            .unwrap()
            .expect("restored request has a reusable snapshot");
        assert_eq!(roundtrip.context_tokens, 100);
        assert_eq!(roundtrip.cache_context_tokens, 100);
        assert_eq!(roundtrip.kv_bytes, snapshot.kv_bytes);
        assert!(engine
            .snapshot_request_state_at_prefix(&state, 99, Vec::new())
            .unwrap()
            .is_none());
        let exact_prefix = engine
            .snapshot_request_state_at_prefix(&state, 100, vec![0; packed_request_bytes])
            .unwrap()
            .expect("exact request frontier is reusable");
        assert_eq!(exact_prefix.context_tokens, 100);
        let short_snapshot = DsparkRequestCacheSnapshot {
            context_tokens: 100,
            cache_context_tokens: 100,
            kv_bytes: vec![0; packed_request_bytes - 1],
        };
        assert!(engine
            .restore_request_state(&mut state, &short_snapshot)
            .unwrap_err()
            .to_string()
            .contains("expected 449280"));
    }

    #[test]
    fn joint_scheduler_preserves_a_required_probe() {
        let profile =
            DsparkSpsProfile::new(vec![0.0, 100.0, 100.0, 40.0, 30.0, 20.0, 10.0]).unwrap();
        let schedule = schedule_dspark_verification_with_minimums(
            &[vec![0.01, 0.01], vec![0.01, 0.01]],
            &[1, 0],
            &profile,
            DsparkScheduleSearch::GlobalMaximum,
        )
        .unwrap();
        assert_eq!(schedule.prefix_lengths, [1, 0]);
        assert_eq!(schedule.target_batch_rows, 3);
    }

    #[test]
    fn installed_sps_profile_is_immutable_to_runtime_observations() {
        let mut model = DsparkRuntimeCostModel::new(1, 2).unwrap();
        model
            .install_profile(1, &[(1, 10.0), (2, 20.0), (3, 30.0)])
            .unwrap();
        let before = model.profile(1, &[0]).unwrap();
        let observation = model.observe(1, &[0], 2, 1.0).unwrap();
        let after = model.profile(1, &[0]).unwrap();
        assert_eq!(observation.exact_samples, 0);
        assert_eq!(before.get(1).unwrap(), after.get(1).unwrap());
        assert_eq!(before.get(2).unwrap(), after.get(2).unwrap());
        assert_eq!(before.get(3).unwrap(), after.get(3).unwrap());
    }

    #[test]
    fn native_confidence_has_no_external_context_prior() {
        let residual = DsparkConfidenceResidual::default();
        assert_eq!(residual.global_logit_bias(1), 0.0);
        assert_eq!(residual.global_logit_bias(300_000), 0.0);
    }

    #[test]
    fn runtime_cost_model_covers_three_native_blocks_with_five_proposals() {
        let model = DsparkRuntimeCostModel::new(16, 5).unwrap();
        let profile = model.profile(16, &vec![400_000; 16]).unwrap();
        assert!(profile.get(96).unwrap().is_finite());
    }
}
