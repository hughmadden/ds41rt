use anyhow::{bail, Context, Result};
use ds4rt_core::{
    coordinator_graph_bucket_for_active_rows, DeepseekV4KvCacheFormat, KvCacheConfig, KvCacheDType,
    MlaKvCacheRepresentation, ModelFacts, TensorCatalog, TensorRole, DS4_KV_SOURCE_PAGE_TOKENS,
    EXPERT_HOSTS,
};
use ds4rt_loader::{decode_tokenizer_ids, LoadedTokenizer};
use ds4rt_transport::{expert_protocol_v2_compact_id, TcpProtocolV2HostBatchTarget};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, VecDeque};
use std::env;
use std::fs::{self, File};
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::ops::{Deref, DerefMut, Range};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::cli::CoordinatorArgs;
use crate::python_graph_capture::coordinator_python_capture_enabled;

use super::constants::REAL_DS4_FULL_BLOCKER;
use super::constraint::{RealFullConstraintCompiler, RealFullConstraintState};
use super::coordinator_kernels::{
    clear_transient_coordinator_owned_device_buffers, coordinator_owned_device_buffer_bank_scope,
    plan_deepseek_v4_dspark_device_storage_with_cache_format,
    plan_deepseek_v4_target_device_storage_with_cache_format,
    prewarm_flashinfer_cudnn_mla_suffix_graphs_for_worker,
    seal_coordinator_owned_device_buffer_pool, with_coordinator_owned_device_buffer_bank,
    DeepseekV4DsparkDeviceStorage, DeepseekV4DsparkTp4DispatchInput, DeepseekV4TargetDeviceStorage,
    DeviceBf16Output,
};
use super::dspark::{
    dspark_active_max_verify_drafts, dspark_target_hidden_tap_layer_ids,
    native_dspark_cache_page_bytes, schedule_dspark_verification_with_minimums,
    DsparkConfidenceCalibrator, DsparkConfidenceResidual, DsparkDraftPlan, DsparkDraftStep,
    DsparkPreparedReplay, DsparkReplayOutput, DsparkRequestCacheSnapshot, DsparkRequestEngine,
    DsparkRequestState, DsparkRuntimeCostModel, DsparkRuntimeCostObservation, DsparkScheduleSearch,
    DsparkVerificationSchedule, FLASH_0731_K2_AFD_SPS_PROFILE_ID, FLASH_0731_K2_AFD_SPS_PROFILE_MS,
    NATIVE_DSPARK_PROPOSAL_TOKENS,
};
use super::execution_plan::real_full_execution_plan;
use super::expert_probe::REAL_NVFP4_PROTOCOL_V2_EXECUTOR;
use super::kv::device::RealFullDeviceKvStorageHandle;
use super::prefix_cache::{
    TargetKvExactSubtreeEviction, TargetKvRadixManager, TargetKvRadixReservation,
};
use super::preflight::{
    real_ds4_full_preflight_report, real_full_kv_cache_config_for_model,
    real_full_sparse_transport_plan,
};
use super::residency::preload_real_full_coordinator_resident_weights;
use super::sampling::{RealFullLmHeadSamplingOptions, RealLmHeadBatchScoreForHidden};
use super::scheduler::{
    load_real_full_kv_snapshot,
    real_full_scheduler_execution_for_batched_shapes_with_shared_sparse_tcp_and_state_device_hidden,
    real_full_scheduler_execution_for_shape_with_shared_sparse_tcp_and_state_device_hidden,
    save_real_full_kv_snapshot, scheduler_prefill_chunk_count_for_rows, RealFullKvSnapshot,
    RealFullSchedulerBatchedInput, RealFullSchedulerDeviceExecution,
    RealFullSchedulerDsparkTp4PendingDispatch, RealFullSchedulerExecutionShape,
    RealFullSchedulerExecutionState, RealFullSchedulerNativeTargetContext,
    RealFullSchedulerNativeTargetIdentity, RealFullSchedulerSparseDispatchTransport,
    RealFullSchedulerSparseTcpDispatchProbe, RealFullSchedulerSparseTcpDispatchWorker,
    RealFullSchedulerTargetHiddenTaps,
};
use super::target_sampling::{
    prewarm_real_full_paired_target_token_sample_rows, prewarm_real_full_target_sampler_capacity,
    real_full_target_token_samples, real_full_target_token_samples_constrained,
    real_full_target_token_samples_pair, real_full_target_token_samples_with_options,
};
use super::types::{
    RealDs4FullPreflightReport, RealFullCoordinatorResidentPreloadPlan,
    RealFullSchedulerExecutionDryRun, RealFullSchedulerTerminalLmHeadSample,
};

const DEFAULT_REAL_FULL_REQUEST_PREFILL_CHUNK_TOKENS: usize = 2 * 1024;
const REAL_FULL_NATIVE_TARGET_PREFILL_MAX_QUERY_ROWS: usize = 2 * 1024;
const DEFAULT_REAL_FULL_REQUEST_FRESH_SMALL_PREFILL_CHUNK_TOKENS: usize = 512;
const DEFAULT_REAL_FULL_REQUEST_SMALL_PREFILL_CHUNK_TOKENS: usize = 256;
const DEFAULT_REAL_FULL_REQUEST_CACHED_WIDE_SUFFIX_MIN_TOKENS: usize = 1_024 + 1;
const DEFAULT_REAL_FULL_REQUEST_LARGE_PREFILL_MIN_TOKENS: usize = 4 * 1024;
const DEFAULT_REAL_FULL_REQUEST_LONG_PREFIX_MIN_TOKENS: usize = 32 * 1024;
const DEFAULT_REAL_FULL_REQUEST_LONG_PREFIX_SMALL_PREFILL_CHUNK_TOKENS: usize = 512;

fn request_lm_head_sampling_options(
    request: &ds4rt_api::RealFullRequest,
) -> RealFullLmHeadSamplingOptions {
    request_lm_head_sampling_options_at(request, request.decode_step_index)
}

fn request_lm_head_sampling_options_at(
    request: &ds4rt_api::RealFullRequest,
    decode_step_index: usize,
) -> RealFullLmHeadSamplingOptions {
    if request.greedy_sampling {
        return RealFullLmHeadSamplingOptions::diagnostic();
    }
    RealFullLmHeadSamplingOptions {
        random_uniform: request.sampling.random_uniform(decode_step_index),
        temperature: request.sampling.temperature(),
        top_k: request.sampling.top_k(),
        top_p: request.sampling.top_p(),
    }
}

fn request_sampling_uniforms(
    request: &ds4rt_api::RealFullRequest,
    start_decode_step: usize,
    rows: usize,
) -> Vec<f32> {
    (0..rows)
        .map(|row| {
            request
                .sampling
                .random_uniform(start_decode_step.saturating_add(row))
        })
        .collect()
}

fn real_full_constraint_target_samples(
    catalog: &TensorCatalog,
    state: &BudgetedRealFullSchedulerExecutionState,
    request: &ds4rt_api::RealFullRequest,
    target_hidden: &DeviceBf16Output,
    suffix_rows: usize,
    draft_token_ids: &[usize],
) -> Result<Option<RealLmHeadBatchScoreForHidden>> {
    let Some(constraint) = state.constraint.as_ref() else {
        return Ok(None);
    };
    anyhow::ensure!(
        suffix_rows == draft_token_ids.len() + 1,
        "constrained target suffix has {suffix_rows} rows for {} speculative drafts",
        draft_token_ids.len()
    );
    let masks = constraint
        .masks_for_draft(draft_token_ids)
        .context("building constrained target sampling masks")?;
    let options = request_lm_head_sampling_options_at(request, request.decode_step_index);
    let random_uniforms = if request.greedy_sampling {
        vec![options.random_uniform; suffix_rows]
    } else {
        request_sampling_uniforms(request, request.decode_step_index, suffix_rows)
    };
    real_full_target_token_samples_constrained(
        catalog,
        target_hidden,
        suffix_rows,
        options,
        &random_uniforms,
        &masks,
    )
    .map(Some)
}
const DEFAULT_REAL_FULL_REQUEST_LONG_PREFIX_TAIL_MERGE_NUMERATOR: usize = 7;
const DEFAULT_REAL_FULL_REQUEST_LONG_PREFIX_TAIL_MERGE_DENOMINATOR: usize = 4;
const DEFAULT_REAL_FULL_REQUEST_MIN_STREAMING_TAIL_PREFILL_TOKENS: usize = 15;
const REAL_FULL_PREFILL_PIPELINE_LANES: usize = 4;
const REAL_FULL_REQUEST_LARGE_PREFILL_MIN_TOKENS_ENV: &str =
    "DS4RT_REAL_FULL_REQUEST_LARGE_PREFILL_MIN_TOKENS";
const REAL_FULL_REQUEST_LONG_PREFIX_SMALL_PREFILL_CHUNK_TOKENS_ENV: &str =
    "DS4RT_REAL_FULL_REQUEST_LONG_PREFIX_SMALL_PREFILL_CHUNK_TOKENS";
const REAL_FULL_SEQUENCE_EXTENSION_HEADROOM_TOKENS: usize = 4 * 1024;
const REAL_FULL_SHARED_KV_PAGE_TOKENS: usize = 64;
const REAL_FULL_MAX_ACTIVE_REQUESTS: usize = 16;
const REAL_FULL_DSPARK_MAX_MAIN_ROWS: usize = 2_048;
const REAL_FULL_MAX_EXECUTION_LANES_ENV: &str = "DS4RT_REAL_FULL_MAX_EXECUTION_LANES";
const REAL_FULL_DIAGNOSTIC_MAX_EXECUTION_LANES: usize = 8;
const REAL_FULL_REQUEST_MAX_MTP_VERIFY_ROWS: usize = 4;
const REAL_FULL_REQUEST_MTP_ACCEPTED_ROWS: usize = 2;
const REAL_FULL_SERVE_PREWARM_REQUEST_ENV: &str = "DS4RT_REAL_FULL_SERVE_PREWARM_REQUEST";
const REAL_FULL_STARTUP_SEAL_OWNED_BUFFER_POOL_PREFIX: &str =
    "real-full-startup-seal-owned-buffer-pool-";
const REAL_FULL_STARTUP_PREWARM_PAIRED_LM_HEAD_PREFIX: &str =
    "real-full-startup-prewarm-paired-lm-head-";
const REAL_FULL_STARTUP_PREWARM_BATCHED_DSPARK_PREFIX: &str =
    "real-full-startup-prewarm-batched-dspark-";
const REAL_FULL_STARTUP_MAX_PREFILL_CHUNK_PREFIX: &str =
    "real-full-startup-capture-arena-max-prefill-chunk-";
const REAL_FULL_STARTUP_CANONICAL_PREFILL_CHUNK_PREFIX: &str =
    "real-full-startup-capture-arena-canonical-prefill-chunk-";
const REAL_FULL_STARTUP_CANONICAL_PREFILL_CHUNK_TOKENS: usize = 1_024;
const REAL_FULL_STARTUP_TARGET_RADIX_PUBLISH_PREFIX: &str =
    "real-full-startup-capture-arena-radix-publish-";
const REAL_FULL_STARTUP_TARGET_RADIX_EVICT_PREFIX: &str =
    "real-full-startup-evict-target-radix-prefix-";
const REAL_FULL_STARTUP_BATCHED_DSPARK_BANK_MARKER: &str = "-batched-bank-";
const REAL_FULL_STARTUP_SCALAR_DSPARK_COHORT_MARKER: &str = "-scalar-cohort-";
const REAL_FULL_SERVE_PREFIX_PREFILL_PROBE_ENV: &str = "DS4RT_REAL_FULL_SERVE_PREFIX_PREFILL_PROBE";
const REAL_FULL_SERVE_PREFIX_PREFILL_PROBE_REPEATS_ENV: &str =
    "DS4RT_REAL_FULL_SERVE_PREFIX_PREFILL_PROBE_REPEATS";
const REAL_FULL_SERVE_PREFIX_PREFILL_PROBE_PREFIX_ROWS_ENV: &str =
    "DS4RT_REAL_FULL_SERVE_PREFIX_PREFILL_PROBE_PREFIX_ROWS";
const REAL_FULL_SERVE_PREFIX_PREFILL_PROBE_NEW_ROWS_ENV: &str =
    "DS4RT_REAL_FULL_SERVE_PREFIX_PREFILL_PROBE_NEW_ROWS";
const MAX_REAL_FULL_SERVE_PREFIX_PREFILL_PROBE_REPEATS: usize = 8;
const REAL_FULL_SERVE_PREWARM_PROMPT_TOKEN: &str = "alpha ";
const REAL_FULL_SERVE_PREWARM_BOUNDARY_TOKEN: &str = "beta ";
const REAL_FULL_SERVE_NO_SELECTOR_DSA_BOUNDARY_PROMPT_TOKENS: usize = 2_049;
const REAL_FULL_SERVE_NO_SELECTOR_DSA_BOUNDARY_QUERY_ROWS: usize = 512;
// The 512-row boundary is the canonical graph identity for this bucket. A
// historical 1,008-row sizing request was balanced into two 504-row chunks;
// its 99 exact-row BF16 linear captures were immediately replaced here, while
// its retained packed-attention capture set is equally seeded by this request.
const REAL_FULL_SERVE_PREWARM_PREFILL_ROWS: &[usize] = &[2_048, 512, 144, 72, 36, 18, 9, 8];
const REAL_FULL_SERVE_DSA_SELECTOR_PREWARM_QUERY_ROWS: &[usize] = &[8, 16, 32, 64, 128, 256, 512];

const DEFAULT_REAL_FULL_SERVE_PREFIX_PREFILL_PROBE_ROWS: usize = 1_008;
const REAL_FULL_SERVE_PREWARM_DECODE_BUDGET: usize = 2;
const REAL_FULL_REQUEST_TIMING_ENV: &str = "DS4RT_REAL_FULL_REQUEST_TIMING";
const REAL_FULL_REQUEST_THREAD_PINNED_ENV: &str = "DS4RT_REAL_FULL_REQUEST_THREAD_PINNED";
const REAL_FULL_REQUEST_THREAD_PINNED_WORKERS_ENV: &str =
    "DS4RT_REAL_FULL_REQUEST_THREAD_PINNED_WORKERS";
const REAL_FULL_REQUEST_WORKER_CPUS_ENV: &str = "DS4RT_REAL_FULL_REQUEST_WORKER_CPUS";
const REAL_FULL_SCHEDULER_WORKER_CPU_ENV: &str = "DS4RT_REAL_FULL_SCHEDULER_WORKER_CPU";
const REAL_FULL_REQUEST_MTP_VERIFY_ENV: &str = "DS4RT_REAL_FULL_REQUEST_MTP_VERIFY";
const REAL_FULL_DSPARK_ENV: &str = "DS4RT_REAL_FULL_DSPARK";
const REAL_FULL_DSPARK_SHADOW_ENV: &str = "DS4RT_REAL_FULL_DSPARK_SHADOW";
const REAL_FULL_DSPARK_CACHE_MODE_ENV: &str = "DS4RT_REAL_FULL_DSPARK_CACHE_MODE";
const REAL_FULL_DSPARK_TAIL_CACHE_BYTES_ENV: &str = "DS4RT_REAL_FULL_DSPARK_TAIL_CACHE_BYTES";
const REAL_FULL_DSPARK_TRACE_ENV: &str = "DS4RT_REAL_FULL_DSPARK_TRACE";
const REAL_FULL_DSPARK_CONFIDENCE_POLICY_ENV: &str = "DS4RT_REAL_FULL_DSPARK_CONFIDENCE_POLICY";
const REAL_FULL_DSPARK_FIXED_DRAFTS_ENV: &str = "DS4RT_REAL_FULL_DSPARK_FIXED_DRAFTS";
const REAL_FULL_DSPARK_PROFILE_AT_STARTUP_ENV: &str = "DS4RT_REAL_FULL_DSPARK_PROFILE_AT_STARTUP";
const REAL_FULL_EXPERT_READY_TIMEOUT_SECS_ENV: &str =
    "DS4RT_REAL_FULL_SERVE_EXPERT_READY_TIMEOUT_SECS";
const REAL_FULL_EXPERT_WARMUP_STATUS_FILE_ENV: &str =
    "DS4RT_REAL_FULL_SERVE_EXPERT_WARMUP_STATUS_FILE";
const DEFAULT_REAL_FULL_EXPERT_READY_TIMEOUT_SECS: u64 = 900;
const REAL_FULL_KV_POOL_TOKENS_ENV: &str = "DS4RT_REAL_FULL_KV_POOL_TOKENS";
const REAL_FULL_DSPARK_PAGE_SIZE: usize = 64;
const REAL_FULL_DSPARK_ADAPTIVE_MAX_VERIFY_DRAFTS: usize = 5;
const REAL_FULL_DSPARK_MAX_VERIFY_DRAFTS: usize = 5;
const REAL_FULL_KV_SNAPSHOT_LOAD_ENV: &str = "DS4RT_REAL_FULL_KV_SNAPSHOT_LOAD";
const REAL_FULL_KV_SNAPSHOT_SAVE_ENV: &str = "DS4RT_REAL_FULL_KV_SNAPSHOT_SAVE";
const REAL_FULL_KV_SNAPSHOT_SAVE_TOKENS_ENV: &str = "DS4RT_REAL_FULL_KV_SNAPSHOT_SAVE_TOKENS";
const REAL_FULL_KV_SNAPSHOT_SAVE_POINTS_ENV: &str = "DS4RT_REAL_FULL_KV_SNAPSHOT_SAVE_POINTS";
const REAL_FULL_ENGINE_COMMIT_ENV: &str = "DS4RT_ENGINE_COMMIT";

pub(crate) struct LoadedRealFullServing {
    pub(crate) info: ds4rt_api::RealFullInfo,
    pub(crate) kv_config: KvCacheConfig,
    pub(crate) executor: Arc<dyn ds4rt_api::RealFullRequestExecutor>,
}

struct RealFullPrefixPrefillProbe {
    cases: Vec<RealFullPrefixPrefillProbeCase>,
    repeats: usize,
}

struct RealFullPrefixPrefillProbeCase {
    prefix_prompt: String,
    prefix_prompt_tokens: usize,
    new_prompt_rows: usize,
}

struct RealFullSchedulerRequestExecutor {
    base_info: ds4rt_api::RealFullInfo,
    catalog: TensorCatalog,
    kv_config: KvCacheConfig,
    device_kv_pool_config: KvCacheConfig,
    sparse_tcp_targets: Vec<TcpProtocolV2HostBatchTarget>,
    sparse_tcp_dispatch_worker: Arc<RealFullSchedulerSparseTcpDispatchWorker>,
    scheduler_states: Mutex<HashMap<String, BudgetedRealFullSchedulerExecutionState>>,
    recycled_scheduler_states: Mutex<Vec<BudgetedRealFullSchedulerExecutionState>>,
    max_execution_lanes: usize,
    device_kv_storage: Mutex<Option<RealFullDeviceKvStorageHandle>>,
    context_budget: Arc<RealFullContextTokenBudget>,
    target_kv_radix: Arc<TargetKvRadixManager>,
    sampled_token_text_cache: Mutex<HashMap<usize, String>>,
    tokenizer: Mutex<LoadedTokenizer>,
    constraint_compiler: RealFullConstraintCompiler,
    kv_snapshot_load: Option<Arc<RealFullKvSnapshot>>,
    kv_snapshot_saves: Vec<RealFullKvSnapshotSave>,
    kv_snapshot_saved: AtomicBool,
    dspark: Option<Mutex<RealFullDsparkRuntime>>,
    dspark_device_storage: Option<Mutex<DeepseekV4DsparkDeviceStorage>>,
    target_device_storage: Arc<Mutex<DeepseekV4TargetDeviceStorage>>,
    target_device_identity: RealFullSchedulerNativeTargetIdentity,
    engine_commit: String,
}

struct RealFullDsparkRuntime {
    mode: RealFullDsparkServingMode,
    confidence_policy: RealFullDsparkConfidencePolicy,
    cache_mode: RealFullDsparkCacheMode,
    context_tokens: usize,
    engine: DsparkRequestEngine,
    requests: HashMap<String, RealFullDsparkRequestRuntime>,
    tail_cache: RealFullDsparkTailCache,
    cost_model: DsparkRuntimeCostModel,
}

struct RealFullDsparkRequestRuntime {
    cache: DsparkRequestState,
    confidence_calibrator: DsparkConfidenceCalibrator,
    confidence_residual: DsparkConfidenceResidual,
    pending_verification: Option<DsparkDraftPlan>,
    pending_windows: VecDeque<RealFullDsparkShadowWindow>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RealFullDsparkTailKey {
    prefix_tokens: usize,
    prefix_sha256: [u8; 32],
}

struct RealFullDsparkTailEntry {
    key: RealFullDsparkTailKey,
    snapshot: DsparkRequestCacheSnapshot,
    confidence_calibrator: DsparkConfidenceCalibrator,
    confidence_residual: DsparkConfidenceResidual,
}

struct RealFullDsparkTailCache {
    entries: VecDeque<RealFullDsparkTailEntry>,
    resident_bytes: usize,
    max_bytes: usize,
}

impl RealFullDsparkTailCache {
    fn new(max_bytes: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            resident_bytes: 0,
            max_bytes,
        }
    }

    fn take_exact_prefix(
        &mut self,
        prompt_token_ids: &[usize],
        matched_target_tokens: usize,
    ) -> Option<RealFullDsparkTailEntry> {
        let matched_target_tokens = matched_target_tokens.min(prompt_token_ids.len());
        let fingerprint =
            real_full_dspark_prefix_fingerprint(&prompt_token_ids[..matched_target_tokens]);
        let index = self.entries.iter().position(|entry| {
            entry.key.prefix_tokens == matched_target_tokens
                && entry.key.prefix_sha256 == fingerprint
        })?;
        let entry = self
            .entries
            .remove(index)
            .expect("the selected dSpark tail entry came from this deque");
        self.resident_bytes = self
            .resident_bytes
            .saturating_sub(entry.snapshot.resident_bytes());
        Some(entry)
    }

    fn longest_exact_prefix_tokens(
        &self,
        prompt_token_ids: &[usize],
        maximum_prefix_tokens: usize,
    ) -> usize {
        let maximum_prefix_tokens = maximum_prefix_tokens.min(prompt_token_ids.len());
        let mut candidate_lengths = self
            .entries
            .iter()
            .map(|entry| entry.key.prefix_tokens)
            .filter(|prefix_tokens| *prefix_tokens != 0 && *prefix_tokens <= maximum_prefix_tokens)
            .collect::<Vec<_>>();
        candidate_lengths.sort_unstable();
        candidate_lengths.dedup();
        let mut candidate_lengths = candidate_lengths.into_iter().peekable();
        let mut hasher = Sha256::new();
        let mut longest_match = 0;
        for (token_index, token_id) in prompt_token_ids[..maximum_prefix_tokens].iter().enumerate()
        {
            hasher.update((*token_id as u64).to_le_bytes());
            let prefix_tokens = token_index + 1;
            if candidate_lengths.peek().copied() != Some(prefix_tokens) {
                continue;
            }
            let prefix_sha256: [u8; 32] = hasher.clone().finalize().into();
            if self.entries.iter().any(|entry| {
                entry.key.prefix_tokens == prefix_tokens && entry.key.prefix_sha256 == prefix_sha256
            }) {
                longest_match = prefix_tokens;
            }
            candidate_lengths.next();
        }
        longest_match
    }

    fn insert(&mut self, entry: RealFullDsparkTailEntry) -> bool {
        let entry_bytes = entry.snapshot.resident_bytes();
        if self.max_bytes == 0 || entry_bytes > self.max_bytes {
            return false;
        }
        if let Some(index) = self
            .entries
            .iter()
            .position(|current| current.key == entry.key)
        {
            let replaced = self
                .entries
                .remove(index)
                .expect("the duplicate dSpark tail entry came from this deque");
            self.resident_bytes = self
                .resident_bytes
                .saturating_sub(replaced.snapshot.resident_bytes());
        }
        while self
            .resident_bytes
            .checked_add(entry_bytes)
            .is_none_or(|bytes| bytes > self.max_bytes)
        {
            let Some(evicted) = self.entries.pop_front() else {
                return false;
            };
            self.resident_bytes = self
                .resident_bytes
                .saturating_sub(evicted.snapshot.resident_bytes());
        }
        self.resident_bytes += entry_bytes;
        self.entries.push_back(entry);
        true
    }
}

fn real_full_dspark_prefix_fingerprint(token_ids: &[usize]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for token_id in token_ids {
        hasher.update((*token_id as u64).to_le_bytes());
    }
    hasher.finalize().into()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RealFullDsparkServingMode {
    Active,
    Shadow,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RealFullDsparkCacheMode {
    RequestLocal,
    PromptSwa,
}

fn real_full_dspark_cross_request_tail_reuse_enabled(mode: RealFullDsparkCacheMode) -> bool {
    mode == RealFullDsparkCacheMode::PromptSwa
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RealFullDsparkConfidencePolicy {
    Calibrated,
    Raw,
    Residual,
}

struct RealFullDsparkShadowWindow {
    origin_context: usize,
    proposal_token_ids: Vec<usize>,
    conditional_confidence: Vec<f32>,
    matched: usize,
}

struct RealFullDsparkReplayResult {
    step: DsparkDraftStep,
    plan: DsparkDraftPlan,
    mode: RealFullDsparkServingMode,
    context_tokens: usize,
}

impl RealFullDsparkRuntime {
    fn trace_shadow_window(
        sequence_id: &str,
        window: &RealFullDsparkShadowWindow,
        resolution: &'static str,
        target_token: Option<usize>,
    ) {
        if !real_full_dspark_trace_enabled() {
            return;
        }
        let observed_positions = match resolution {
            "full_match" => window.proposal_token_ids.len(),
            "mismatch" => window.matched.saturating_add(1),
            _ => window.matched,
        };
        eprintln!(
            "real_full_dspark_shadow_policy_trace {}",
            serde_json::json!({
                "schema": "ds4rt-dspark-shadow-policy-trace-v1",
                "sequence_id": sequence_id,
                "origin_context": window.origin_context,
                "proposal_token_ids": &window.proposal_token_ids,
                "conditional_confidence": &window.conditional_confidence,
                "accepted_prefix": window.matched,
                "observed_positions": observed_positions,
                "resolution": resolution,
                "target_token": target_token,
            })
        );
    }

    fn release_internal_sequences(&mut self) -> usize {
        let stale_internal_sequences = self
            .requests
            .keys()
            .filter(|candidate| real_full_internal_sequence(candidate))
            .cloned()
            .collect::<Vec<_>>();
        let released = stale_internal_sequences.len();
        for stale_sequence in stale_internal_sequences {
            let stale_request = self
                .requests
                .remove(&stale_sequence)
                .expect("the stale dSpark startup sequence came from this map");
            self.engine.release_request_state(stale_request.cache);
        }
        released
    }

    fn prepare_cycle(
        &mut self,
        sequence_id: &str,
        initial_decode: bool,
        reusable_target_prefix: Option<(&[usize], usize)>,
        startup_draft_tokens: Option<usize>,
        device_storage: Option<&Mutex<DeepseekV4DsparkDeviceStorage>>,
    ) -> Result<Option<DsparkDraftPlan>> {
        if !self.requests.contains_key(sequence_id) {
            if real_full_internal_sequence(sequence_id)
                && !real_full_batched_dspark_prewarm_sequence(sequence_id)
            {
                self.release_internal_sequences();
            }
            let cache = self.engine.allocate_request_state()?;
            self.requests.insert(
                sequence_id.to_owned(),
                RealFullDsparkRequestRuntime {
                    cache,
                    confidence_calibrator: DsparkConfidenceCalibrator::default(),
                    confidence_residual: DsparkConfidenceResidual::default(),
                    pending_verification: None,
                    pending_windows: VecDeque::new(),
                },
            );
        } else if initial_decode {
            let request = self
                .requests
                .get_mut(sequence_id)
                .expect("the dSpark request was checked above");
            self.engine.reset_request_state(&mut request.cache)?;
        }
        let reusable_tail = if initial_decode
            && real_full_dspark_cross_request_tail_reuse_enabled(self.cache_mode)
        {
            reusable_target_prefix.and_then(|(prompt_token_ids, matched_target_tokens)| {
                let uncached_suffix_tokens =
                    prompt_token_ids.len().saturating_sub(matched_target_tokens);
                (uncached_suffix_tokens < self.context_tokens).then(|| {
                    self.tail_cache
                        .take_exact_prefix(prompt_token_ids, matched_target_tokens)
                })?
            })
        } else {
            None
        };
        let request = self
            .requests
            .get_mut(sequence_id)
            .expect("the dSpark request was inserted above");
        if initial_decode {
            request.pending_verification = None;
            request.pending_windows.clear();
            if let Some(entry) = reusable_tail {
                let prefix_tokens = entry.key.prefix_tokens;
                let cache_tokens = entry.snapshot.cache_context_tokens;
                let snapshot_bytes = entry.snapshot.resident_bytes();
                let restore_result = (|| {
                    if entry.snapshot.cache_context_tokens > 0 {
                        device_storage
                            .context("restoring a reusable dSpark tail requires GPU0 storage")?
                            .lock()
                            .map_err(|error| {
                                anyhow::anyhow!(
                                    "locking dSpark device storage for restore failed: {error}"
                                )
                            })?
                            .restore_request_kv(
                                request.cache.request_slot(),
                                &entry.snapshot.kv_bytes,
                            )
                            .context("restoring reusable dSpark mtp.* KV pages")?;
                    }
                    self.engine
                        .restore_request_state(&mut request.cache, &entry.snapshot)
                })();
                if let Err(error) = restore_result {
                    let _ = self.tail_cache.insert(entry);
                    return Err(error).context("restoring a reusable dSpark tail");
                }
                request.confidence_calibrator = entry.confidence_calibrator.clone();
                request.confidence_residual = entry.confidence_residual.clone();
                let retained = self.tail_cache.insert(entry);
                anyhow::ensure!(
                    retained,
                    "restored dSpark tail no longer fits its configured host cache"
                );
                eprintln!(
                    "real_full_dspark_tail_restore sequence_id={} prefix_tokens={} cache_tokens={} snapshot_bytes={} cached_entries={} cached_bytes={}",
                    sequence_id,
                    prefix_tokens,
                    cache_tokens,
                    snapshot_bytes,
                    self.tail_cache.entries.len(),
                    self.tail_cache.resident_bytes,
                );
            } else {
                request.confidence_calibrator.reset();
                request.confidence_residual.reset();
            }
        }
        if self.mode == RealFullDsparkServingMode::Active && !initial_decode {
            if let Some(draft_tokens) = startup_draft_tokens {
                // Startup width capture must replace the ordinary adaptive
                // plan, including the common empty plan produced by the seed
                // step. Otherwise every nominal width sweep remains scalar.
                request.pending_verification = Some(DsparkDraftPlan {
                    proposal_token_ids: vec![0; draft_tokens],
                    conditional_confidence: Vec::new(),
                    candidate_proposal_token_ids: vec![0; draft_tokens],
                    candidate_conditional_confidence: Vec::new(),
                    candidate_adjusted_confidence: Vec::new(),
                    selected_drafts: draft_tokens,
                    minimum_drafts: draft_tokens,
                    target_batch_rows: draft_tokens + 1,
                    expected_committed_tokens: 1.0,
                    expected_tokens_per_second: 0.0,
                    confidence_logit_bias: request.confidence_calibrator.logit_bias(),
                    confidence_context_tokens: request.cache.context_tokens(),
                    calibration_eligible: false,
                });
            }
        }
        Ok((self.mode == RealFullDsparkServingMode::Active)
            .then(|| request.pending_verification.take())
            .flatten())
    }

    fn restore_verification(&mut self, sequence_id: &str, plan: DsparkDraftPlan) {
        if self.mode == RealFullDsparkServingMode::Active {
            if let Some(request) = self.requests.get_mut(sequence_id) {
                request.pending_verification = Some(plan);
            }
        }
    }

    fn prepare_replay(
        &mut self,
        sequence_id: &str,
        target_hidden_taps: [&DeviceBf16Output; 3],
        target_row_start: usize,
        committed_rows: usize,
        absolute_context_start: Option<usize>,
        anchor_token: usize,
    ) -> Result<DsparkPreparedReplay> {
        let request = self
            .requests
            .get_mut(sequence_id)
            .with_context(|| format!("dSpark request state is missing for {sequence_id}"))?;
        if self.mode == RealFullDsparkServingMode::Shadow {
            let mut retained = VecDeque::with_capacity(request.pending_windows.len() + 1);
            while let Some(mut window) = request.pending_windows.pop_front() {
                let expected = window.proposal_token_ids[window.matched];
                let confidence = window.conditional_confidence[window.matched];
                if expected == anchor_token {
                    window.matched += 1;
                    if window.matched == window.proposal_token_ids.len() {
                        Self::trace_shadow_window(sequence_id, &window, "full_match", None);
                        eprintln!(
                            "real_full_dspark_shadow_acceptance sequence_id={} origin_context={} accepted={} full_match=true",
                            sequence_id, window.origin_context, window.matched
                        );
                    } else {
                        retained.push_back(window);
                    }
                } else {
                    Self::trace_shadow_window(sequence_id, &window, "mismatch", Some(anchor_token));
                    eprintln!(
                        "real_full_dspark_shadow_acceptance sequence_id={} origin_context={} accepted={} full_match=false mismatch_position={} expected_token={} target_token={} mismatch_confidence={:.6}",
                        sequence_id,
                        window.origin_context,
                        window.matched,
                        window.matched,
                        expected,
                        anchor_token,
                        confidence,
                    );
                }
            }
            request.pending_windows = retained;
        }
        self.engine.prepare_replay(
            &mut request.cache,
            target_hidden_taps,
            target_row_start,
            committed_rows,
            absolute_context_start,
            anchor_token,
        )
    }

    fn abort_replay(&mut self, sequence_id: &str, prepared: &DsparkPreparedReplay) -> Result<()> {
        let request = self
            .requests
            .get_mut(sequence_id)
            .with_context(|| format!("dSpark request state is missing for {sequence_id}"))?;
        self.engine.abort_replay(&mut request.cache, prepared)
    }

    fn commit_replay(
        &mut self,
        sequence_id: &str,
        prepared: &DsparkPreparedReplay,
        output: DsparkReplayOutput,
    ) -> Result<(DsparkDraftStep, DsparkDraftPlan)> {
        let request = self
            .requests
            .get_mut(sequence_id)
            .with_context(|| format!("dSpark request state is missing for {sequence_id}"))?;
        let step = self
            .engine
            .validate_replay_output(&request.cache, prepared, output)?;
        let target_context_tokens = prepared.main_context_range().end;
        let sps = self.cost_model.profile(1, &[target_context_tokens])?;
        let (confidence_logit_bias, position_logit_bias, force_probe) = match self.confidence_policy
        {
            RealFullDsparkConfidencePolicy::Calibrated => (
                request.confidence_calibrator.logit_bias(),
                &[][..],
                request.confidence_calibrator.force_probe_due(),
            ),
            RealFullDsparkConfidencePolicy::Raw => (0.0, &[][..], false),
            RealFullDsparkConfidencePolicy::Residual => (
                request
                    .confidence_residual
                    .global_logit_bias(target_context_tokens),
                request.confidence_residual.position_logit_bias(),
                false,
            ),
        };
        let mut plan = self.engine.plan_verification(
            &step,
            REAL_FULL_DSPARK_ADAPTIVE_MAX_VERIFY_DRAFTS,
            confidence_logit_bias,
            position_logit_bias,
            target_context_tokens,
            force_probe,
            &sps,
        )?;
        if let Some(fixed_drafts) = real_full_dspark_fixed_drafts()? {
            plan.proposal_token_ids = step.proposal_token_ids[..fixed_drafts].to_vec();
            plan.conditional_confidence = step.conditional_confidence[..fixed_drafts].to_vec();
            plan.candidate_proposal_token_ids = plan.proposal_token_ids.clone();
            plan.candidate_conditional_confidence = plan.conditional_confidence.clone();
            plan.candidate_adjusted_confidence.truncate(fixed_drafts);
            plan.selected_drafts = fixed_drafts;
            plan.minimum_drafts = fixed_drafts;
            plan.target_batch_rows = fixed_drafts + 1;
            plan.expected_committed_tokens = 0.0;
            plan.expected_tokens_per_second = 0.0;
            plan.calibration_eligible = false;
        }
        self.engine.commit_replay(&mut request.cache, prepared)?;
        if self.mode == RealFullDsparkServingMode::Active {
            request.pending_verification = Some(plan.clone());
        } else {
            request
                .pending_windows
                .push_back(RealFullDsparkShadowWindow {
                    origin_context: step.context_tokens,
                    proposal_token_ids: step.proposal_token_ids.clone(),
                    conditional_confidence: step.conditional_confidence.clone(),
                    matched: 0,
                });
        }
        Ok((step, plan))
    }

    fn record_issued_plan(&mut self, sequence_id: &str, plan: &DsparkDraftPlan) {
        if !plan.calibration_eligible {
            return;
        }
        if let Some(request) = self.requests.get_mut(sequence_id) {
            match self.confidence_policy {
                RealFullDsparkConfidencePolicy::Calibrated => {
                    request
                        .confidence_calibrator
                        .record_selected_drafts(plan.selected_drafts);
                }
                RealFullDsparkConfidencePolicy::Residual => request
                    .confidence_residual
                    .record_selected_drafts(plan.selected_drafts),
                RealFullDsparkConfidencePolicy::Raw => {}
            }
        }
    }

    fn joint_schedule(
        &self,
        plans: &[(&DsparkDraftPlan, usize)],
        context_tokens: &[usize],
    ) -> Result<DsparkVerificationSchedule> {
        anyhow::ensure!(
            plans.len() == context_tokens.len(),
            "joint dSpark scheduling has {} plans and {} contexts",
            plans.len(),
            context_tokens.len(),
        );
        let conditional_confidence = plans
            .iter()
            .map(|(plan, max_useful_drafts)| {
                plan.calibrated_candidate_confidence(*max_useful_drafts)
            })
            .collect::<Vec<_>>();
        let minimum_prefix_lengths = plans
            .iter()
            .map(|(plan, max_useful_drafts)| plan.minimum_drafts.min(*max_useful_drafts))
            .collect::<Vec<_>>();
        let sps = self.cost_model.profile(plans.len(), context_tokens)?;
        schedule_dspark_verification_with_minimums(
            &conditional_confidence,
            &minimum_prefix_lengths,
            &sps,
            DsparkScheduleSearch::GlobalMaximum,
        )
    }

    fn observe_runtime_cost(
        &mut self,
        context_tokens: &[usize],
        target_rows: usize,
        observed_ms: f64,
    ) -> Result<DsparkRuntimeCostObservation> {
        self.cost_model.observe(
            context_tokens.len(),
            context_tokens,
            target_rows,
            observed_ms,
        )
    }

    fn install_runtime_cost_profile(
        &mut self,
        request_count: usize,
        rows: &[(usize, f64)],
    ) -> Result<()> {
        self.cost_model.install_profile(request_count, rows)
    }

    fn observe_verification(
        &mut self,
        sequence_id: &str,
        plan: &DsparkDraftPlan,
        accepted_drafts: usize,
    ) -> (f64, f64, usize) {
        let Some(request) = self.requests.get_mut(sequence_id) else {
            return (0.0, 0.0, 0);
        };
        if plan.calibration_eligible && accepted_drafts <= plan.conditional_confidence.len() {
            match self.confidence_policy {
                RealFullDsparkConfidencePolicy::Calibrated => request
                    .confidence_calibrator
                    .observe(&plan.conditional_confidence, accepted_drafts),
                RealFullDsparkConfidencePolicy::Residual => request.confidence_residual.observe(
                    &plan.conditional_confidence,
                    accepted_drafts,
                    plan.confidence_context_tokens,
                ),
                RealFullDsparkConfidencePolicy::Raw => {}
            }
        }
        match self.confidence_policy {
            RealFullDsparkConfidencePolicy::Residual => (
                request
                    .confidence_residual
                    .global_logit_bias(plan.confidence_context_tokens),
                0.0,
                request.confidence_residual.observation_cycles(),
            ),
            RealFullDsparkConfidencePolicy::Calibrated | RealFullDsparkConfidencePolicy::Raw => (
                request.confidence_calibrator.logit_bias(),
                request.confidence_calibrator.posterior_variance(),
                request.confidence_calibrator.observation_cycles(),
            ),
        }
    }

    fn request_context_tokens(&self, sequence_id: &str) -> Option<usize> {
        self.requests
            .get(sequence_id)
            .map(|request| request.cache.context_tokens())
    }

    fn publish_reusable_prefix(
        &mut self,
        sequence_id: &str,
        prompt_token_ids: &[usize],
        prefix_tokens: usize,
        device_storage: Option<&Mutex<DeepseekV4DsparkDeviceStorage>>,
    ) -> Result<bool> {
        anyhow::ensure!(
            prefix_tokens <= prompt_token_ids.len(),
            "dSpark reusable prefix {prefix_tokens} exceeds prompt length {}",
            prompt_token_ids.len(),
        );
        let request = self
            .requests
            .get(sequence_id)
            .with_context(|| format!("dSpark request state is missing for {sequence_id}"))?;
        if prefix_tokens != request.cache.context_tokens() {
            return Ok(false);
        }
        let kv_bytes = if request.cache.cache_context_tokens() > 0 {
            device_storage
                .context("publishing a reusable dSpark prefix requires GPU0 storage")?
                .lock()
                .map_err(|error| {
                    anyhow::anyhow!("locking dSpark device storage for snapshot failed: {error}")
                })?
                .snapshot_request_kv(request.cache.request_slot())
                .context("snapshotting reusable dSpark mtp.* prefix pages")?
        } else {
            Vec::new()
        };
        let Some(snapshot) = self.engine.snapshot_request_state_at_prefix(
            &request.cache,
            prefix_tokens,
            kv_bytes,
        )?
        else {
            return Ok(false);
        };
        anyhow::ensure!(
            snapshot.context_tokens == prefix_tokens,
            "dSpark reusable snapshot ends at token {}, expected {prefix_tokens}",
            snapshot.context_tokens,
        );
        let key = RealFullDsparkTailKey {
            prefix_tokens,
            prefix_sha256: real_full_dspark_prefix_fingerprint(&prompt_token_ids[..prefix_tokens]),
        };
        let cache_tokens = snapshot.cache_context_tokens;
        let snapshot_bytes = snapshot.resident_bytes();
        let retained = self.tail_cache.insert(RealFullDsparkTailEntry {
            key,
            snapshot,
            confidence_calibrator: request.confidence_calibrator.clone(),
            confidence_residual: request.confidence_residual.clone(),
        });
        eprintln!(
            "real_full_dspark_prefix_publish sequence_id={} prefix_tokens={} cache_tokens={} snapshot_bytes={} retained={} cached_entries={} cached_bytes={} cache_limit_bytes={}",
            sequence_id,
            prefix_tokens,
            cache_tokens,
            snapshot_bytes,
            retained,
            self.tail_cache.entries.len(),
            self.tail_cache.resident_bytes,
            self.tail_cache.max_bytes,
        );
        Ok(retained)
    }

    fn finish_sequence(
        &mut self,
        sequence_id: &str,
        committed_token_ids: Option<&[usize]>,
        device_storage: Option<&Mutex<DeepseekV4DsparkDeviceStorage>>,
    ) -> Result<Option<usize>> {
        let Some(request) = self.requests.remove(sequence_id) else {
            return Ok(None);
        };
        let RealFullDsparkRequestRuntime {
            cache,
            confidence_calibrator,
            confidence_residual,
            pending_windows,
            ..
        } = request;
        for window in &pending_windows {
            Self::trace_shadow_window(sequence_id, window, "request_end", None);
        }
        let snapshot_result = (|| -> Result<Option<usize>> {
            // Request-local dSpark positions begin at the recurrent target
            // tail, not at token zero of the target sequence. They therefore
            // cannot be fingerprinted or aligned against an absolute target
            // radix prefix. Only PromptSwa maintains that shared coordinate
            // system across requests.
            if !real_full_dspark_cross_request_tail_reuse_enabled(self.cache_mode) {
                return Ok(None);
            }
            let Some(committed_token_ids) = committed_token_ids else {
                return Ok(None);
            };
            let kv_bytes = if cache.cache_context_tokens() > 0 {
                device_storage
                    .context("finishing a dSpark request requires GPU0 storage")?
                    .lock()
                    .map_err(|error| {
                        anyhow::anyhow!(
                            "locking dSpark device storage for final snapshot failed: {error}"
                        )
                    })?
                    .snapshot_request_kv(cache.request_slot())
                    .context("snapshotting final dSpark mtp.* KV pages")?
            } else {
                Vec::new()
            };
            self.engine
                .snapshot_request_state(&cache, kv_bytes)
                .and_then(|snapshot| {
                    let Some(snapshot) = snapshot else {
                        return Ok(None);
                    };
                    anyhow::ensure!(
                        snapshot.context_tokens <= committed_token_ids.len(),
                        "dSpark tail ends at token {} but the target committed frontier has only {} tokens",
                        snapshot.context_tokens,
                        committed_token_ids.len(),
                    );
                    let key = RealFullDsparkTailKey {
                        prefix_tokens: snapshot.context_tokens,
                        prefix_sha256: real_full_dspark_prefix_fingerprint(
                            &committed_token_ids[..snapshot.context_tokens],
                        ),
                    };
                    let snapshot_bytes = snapshot.resident_bytes();
                    let cache_tokens = snapshot.cache_context_tokens;
                    let retained = self.tail_cache.insert(RealFullDsparkTailEntry {
                        key,
                        snapshot,
                        confidence_calibrator,
                        confidence_residual,
                    });
                    eprintln!(
                        "real_full_dspark_tail_publish sequence_id={} prefix_tokens={} cache_tokens={} snapshot_bytes={} retained={} cached_entries={} cached_bytes={} cache_limit_bytes={}",
                        sequence_id,
                        key.prefix_tokens,
                        cache_tokens,
                        snapshot_bytes,
                        retained,
                        self.tail_cache.entries.len(),
                        self.tail_cache.resident_bytes,
                        self.tail_cache.max_bytes,
                    );
                    Ok(retained.then_some(key.prefix_tokens))
                })
        })();
        self.engine.release_request_state(cache);
        snapshot_result
    }
}

struct RealFullKvSnapshotSave {
    root: PathBuf,
    token_count: Option<usize>,
}

struct RealFullContextTokenBudget {
    max_tokens: usize,
    inner: Mutex<RealFullContextTokenBudgetInner>,
}

struct RealFullContextTokenBudgetInner {
    free_extents: Vec<RealFullContextTokenExtent>,
    used_tokens: usize,
    active_reservations: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RealFullContextTokenExtent {
    token_base: usize,
    tokens: usize,
}

impl RealFullContextTokenBudget {
    fn new(max_tokens: usize) -> Self {
        debug_assert!(max_tokens > 0);
        Self {
            max_tokens,
            inner: Mutex::new(RealFullContextTokenBudgetInner {
                free_extents: vec![RealFullContextTokenExtent {
                    token_base: 0,
                    tokens: max_tokens,
                }],
                used_tokens: 0,
                active_reservations: 0,
            }),
        }
    }

    fn reserve(self: &Arc<Self>, tokens: usize) -> Result<RealFullContextTokenReservation> {
        anyhow::ensure!(tokens > 0, "real-full context reservation is empty");
        let reserved_tokens = tokens
            .checked_add(REAL_FULL_SHARED_KV_PAGE_TOKENS - 1)
            .context("real-full context reservation page rounding overflow")?
            / REAL_FULL_SHARED_KV_PAGE_TOKENS
            * REAL_FULL_SHARED_KV_PAGE_TOKENS;
        let mut inner = self
            .inner
            .lock()
            .map_err(|error| anyhow::anyhow!("locking real-full context budget failed: {error}"))?;
        anyhow::ensure!(
            inner.active_reservations < REAL_FULL_MAX_ACTIVE_REQUESTS,
            "real-full active request limit exhausted: active={} max={REAL_FULL_MAX_ACTIVE_REQUESTS}",
            inner.active_reservations
        );
        let extent_index = inner
            .free_extents
            .iter()
            .position(|extent| extent.tokens >= reserved_tokens)
            .with_context(|| {
                format!(
                    "real-full global context budget exhausted: requested_sequence_capacity={tokens} reserved_tokens={reserved_tokens} used={} max={}",
                    inner.used_tokens, self.max_tokens
                )
            })?;
        let token_base = inner.free_extents[extent_index].token_base;
        if inner.free_extents[extent_index].tokens == reserved_tokens {
            inner.free_extents.remove(extent_index);
        } else {
            inner.free_extents[extent_index].token_base += reserved_tokens;
            inner.free_extents[extent_index].tokens -= reserved_tokens;
        }
        inner.used_tokens = inner
            .used_tokens
            .checked_add(reserved_tokens)
            .context("real-full context token accounting overflow")?;
        inner.active_reservations += 1;
        Ok(RealFullContextTokenReservation {
            budget: Arc::clone(self),
            token_base,
            reserved_tokens,
        })
    }

    fn release(&self, extent: RealFullContextTokenExtent) {
        let mut inner = self
            .inner
            .lock()
            .expect("real-full context budget lock poisoned during release");
        let insert_at = inner
            .free_extents
            .partition_point(|candidate| candidate.token_base < extent.token_base);
        inner.free_extents.insert(insert_at, extent);
        let mut index = insert_at.saturating_sub(1);
        while index + 1 < inner.free_extents.len() {
            let left = inner.free_extents[index];
            let right = inner.free_extents[index + 1];
            if left.token_base.checked_add(left.tokens) == Some(right.token_base) {
                inner.free_extents[index].tokens = left
                    .tokens
                    .checked_add(right.tokens)
                    .expect("coalesced real-full context extent overflows usize");
                inner.free_extents.remove(index + 1);
            } else {
                index += 1;
            }
        }
        inner.used_tokens = inner
            .used_tokens
            .checked_sub(extent.tokens)
            .expect("real-full context token accounting underflow");
        inner.active_reservations = inner
            .active_reservations
            .checked_sub(1)
            .expect("real-full active request accounting underflow");
    }
}

struct RealFullContextTokenReservation {
    budget: Arc<RealFullContextTokenBudget>,
    token_base: usize,
    reserved_tokens: usize,
}

impl RealFullContextTokenReservation {
    fn token_base(&self) -> usize {
        self.token_base
    }
}

impl Drop for RealFullContextTokenReservation {
    fn drop(&mut self) {
        self.budget.release(RealFullContextTokenExtent {
            token_base: self.token_base,
            tokens: self.reserved_tokens,
        });
    }
}

struct BudgetedRealFullSchedulerExecutionState {
    state: RealFullSchedulerExecutionState,
    execution_lane_id: usize,
    _context_reservation: Option<RealFullContextTokenReservation>,
    target_radix_reservation: Option<TargetKvRadixReservation>,
    bound_target_pages: usize,
    graph_bound_arena: bool,
    snapshot_save_ready: bool,
    snapshot_restore_ms: f64,
    constraint: Option<RealFullConstraintState>,
}

impl Deref for BudgetedRealFullSchedulerExecutionState {
    type Target = RealFullSchedulerExecutionState;

    fn deref(&self) -> &Self::Target {
        &self.state
    }
}

impl DerefMut for BudgetedRealFullSchedulerExecutionState {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.state
    }
}

impl BudgetedRealFullSchedulerExecutionState {
    fn bind_target_radix_reservation(
        &mut self,
        mut reservation: TargetKvRadixReservation,
        prompt_token_ids: &[usize],
        native_target_storage: &Arc<Mutex<DeepseekV4TargetDeviceStorage>>,
    ) -> Result<()> {
        let matched_prefix_tokens = reservation.matched_prefix_tokens();
        anyhow::ensure!(
            matched_prefix_tokens <= prompt_token_ids.len(),
            "target KV radix match {matched_prefix_tokens} exceeds prompt token count {}",
            prompt_token_ids.len()
        );
        let logical_capacity_tokens = reservation.logical_capacity_tokens();
        reservation
            .ensure_materialized_through(prompt_token_ids.len().max(1).min(logical_capacity_tokens))
            .context("materializing initial target KV radix page table")?;
        let source_page_tokens = reservation.page_tokens();
        let device_pages = target_radix_device_pages_for_capacity(
            reservation.physical_pages(),
            source_page_tokens,
            logical_capacity_tokens,
        )?;
        self.state
            .rebind_sequence_physical_pages(&device_pages, logical_capacity_tokens)
            .context("binding scheduler state to target KV radix pages")?;
        if let Some(boundary) = reservation.take_boundary_copy() {
            if source_page_tokens == REAL_FULL_SHARED_KV_PAGE_TOKENS {
                self.state
                    .copy_target_kv_boundary_page(
                        boundary.source_page,
                        boundary.destination_page,
                        boundary.valid_tokens,
                    )
                    .context("copying target KV radix branch boundary")?;
            } else {
                let storage = native_target_storage.lock().map_err(|error| {
                    anyhow::anyhow!("locking native target boundary-copy storage failed: {error}")
                })?;
                anyhow::ensure!(
                    source_page_tokens == storage.plan().source_page_tokens,
                    "target radix page width {source_page_tokens} differs from native target width {}",
                    storage.plan().source_page_tokens,
                );
                storage
                    .copy_physical_boundary_page(
                        boundary.source_page,
                        boundary.destination_page,
                        boundary.valid_tokens,
                    )
                    .context("copying native target KV radix branch boundary")?;
            }
        }
        self.state
            .seed_processed_token_ids(&prompt_token_ids[..matched_prefix_tokens])
            .context("seeding target KV radix processed-token frontier")?;
        self.bound_target_pages = reservation.physical_pages().len();
        self.target_radix_reservation = Some(reservation);
        Ok(())
    }

    fn ensure_target_radix_materialized_through(&mut self, tokens: usize) -> Result<()> {
        let Some(reservation) = self.target_radix_reservation.as_mut() else {
            return Ok(());
        };
        let capacity_tokens = reservation.logical_capacity_tokens();
        let source_page_tokens = reservation.page_tokens();
        let pages = reservation
            .ensure_materialized_through(tokens.min(capacity_tokens))
            .context("materializing target KV radix pages for decode cycle")?;
        if pages.len() != self.bound_target_pages {
            let device_pages =
                target_radix_device_pages_for_capacity(pages, source_page_tokens, capacity_tokens)?;
            self.state
                .extend_sequence_physical_pages(&device_pages, capacity_tokens)
                .context("publishing extended target KV radix page table to device")?;
            self.bound_target_pages = pages.len();
        }
        Ok(())
    }
}

fn target_radix_device_pages(source_pages: &[u32], source_page_tokens: usize) -> Result<Vec<u32>> {
    anyhow::ensure!(
        source_page_tokens >= REAL_FULL_SHARED_KV_PAGE_TOKENS
            && source_page_tokens % REAL_FULL_SHARED_KV_PAGE_TOKENS == 0,
        "target radix source page width {source_page_tokens} is incompatible with the {}-token device page ABI",
        REAL_FULL_SHARED_KV_PAGE_TOKENS,
    );
    let device_pages_per_source = source_page_tokens / REAL_FULL_SHARED_KV_PAGE_TOKENS;
    let mut device_pages = Vec::with_capacity(
        source_pages
            .len()
            .checked_mul(device_pages_per_source)
            .context("expanded target device page-table length overflow")?,
    );
    for source_page in source_pages.iter().copied() {
        let first_device_page = source_page
            .checked_mul(
                u32::try_from(device_pages_per_source)
                    .context("target device pages per source page exceed u32")?,
            )
            .context("expanded target device page ID overflow")?;
        for offset in 0..device_pages_per_source {
            device_pages.push(
                first_device_page
                    .checked_add(u32::try_from(offset).context("target page offset exceeds u32")?)
                    .context("expanded target device page ID overflow")?,
            );
        }
    }
    Ok(device_pages)
}

fn target_radix_device_pages_for_capacity(
    source_pages: &[u32],
    source_page_tokens: usize,
    logical_capacity_tokens: usize,
) -> Result<Vec<u32>> {
    anyhow::ensure!(
        logical_capacity_tokens > 0,
        "target radix logical capacity must be nonzero"
    );
    let mut device_pages = target_radix_device_pages(source_pages, source_page_tokens)?;
    let logical_device_pages = logical_capacity_tokens.div_ceil(REAL_FULL_SHARED_KV_PAGE_TOKENS);
    device_pages.truncate(logical_device_pages);
    Ok(device_pages)
}

fn retain_graph_bound_scheduler_arena(
    graph_bound_arena: bool,
    arena_capacity_tokens: usize,
    max_context_tokens: usize,
) -> bool {
    graph_bound_arena && arena_capacity_tokens == max_context_tokens
}

#[derive(Debug, PartialEq, Eq)]
struct RealFullRequestTokenRows {
    prefix_tokens: usize,
    prefill_tokens: usize,
    prefill_token_ids: Option<Vec<usize>>,
    decode_token_ids: Vec<usize>,
}

struct PreparedBatchedDsparkCycle {
    request: ds4rt_api::RealFullRequest,
    request_start: Instant,
    request_timing: bool,
    sequence_id: String,
    generated_tokens: usize,
    request_id_base: u64,
    token_prefix_tokens: usize,
    token_prefill_tokens: usize,
    committed_input_token_ids: Vec<usize>,
    decode_rows: usize,
    scheduler_start: Instant,
    shape: RealFullSchedulerExecutionShape,
    state: BudgetedRealFullSchedulerExecutionState,
    buffer_bank: usize,
    pending_dspark_plan: Option<DsparkDraftPlan>,
    pending_dspark_draft_token_ids: Vec<usize>,
    dspark_target_hidden_tap_rows: usize,
    snapshot_restore_ms: f64,
}

struct RealFullDsparkReplayLaunch<'a> {
    prepared: &'a DsparkPreparedReplay,
    sequence_id: &'a str,
    request_id: &'a str,
    placement_version: &'a str,
    request_id_base: u64,
    target_hidden_taps: [&'a DeviceBf16Output; 3],
}

struct PreparedBatchedDsparkFinish {
    prepared: PreparedBatchedDsparkCycle,
    target_submit_ms: f64,
    report: RealFullSchedulerExecutionDryRun,
    probe: RealFullSchedulerSparseTcpDispatchProbe,
    target_hidden_taps: Option<RealFullSchedulerTargetHiddenTaps>,
    cycle_token_ids: Vec<usize>,
    terminal_sample: Option<RealFullSpeculativeTerminalSample>,
    final_decode_step: bool,
    replay_prepared: Option<DsparkPreparedReplay>,
    replay_result: Option<RealFullDsparkReplayResult>,
}

struct RealFullSpeculativeTerminalSample {
    hidden_dim: usize,
    vocab_size: usize,
    top_token_id: usize,
    sampled_token_id: usize,
    sample_top_k: usize,
    sample_top_p: f32,
    argmax_backend: &'static str,
    sampler_backend: &'static str,
    accepted_draft_tokens: usize,
    report_mtp_acceptance: bool,
}

fn apply_speculative_terminal_sample_to_report(
    report: &mut RealFullSchedulerExecutionDryRun,
    sample: &RealFullSpeculativeTerminalSample,
) {
    report.terminal_lm_head_sample = speculative_terminal_lm_head_sample(sample);
}

fn speculative_terminal_lm_head_sample(
    sample: &RealFullSpeculativeTerminalSample,
) -> RealFullSchedulerTerminalLmHeadSample {
    RealFullSchedulerTerminalLmHeadSample {
        status: "sampled",
        scope: "sample the terminal row from the retained speculative target hidden batch",
        uses_final_decode_device_hidden: true,
        covers_full_vocabulary: true,
        hidden_dim: sample.hidden_dim,
        vocab_size: sample.vocab_size,
        logits_evaluated: sample.vocab_size,
        top_token_id: Some(sample.top_token_id),
        sampled_token_id: Some(sample.sampled_token_id),
        sample_top_k: Some(sample.sample_top_k),
        sample_top_p: Some(sample.sample_top_p),
        argmax_kernel_backend: Some(sample.argmax_backend),
        sampler_kernel_backend: Some(sample.sampler_backend),
        passed: true,
        blocker: None,
    }
}

#[derive(Debug, PartialEq, Eq)]
struct RealFullSpeculativeAcceptance {
    accepted_draft_tokens: usize,
    terminal_target_index: usize,
    full_match_bonus: bool,
}

fn constrain_dspark_plan(
    plan: &mut DsparkDraftPlan,
    constraint: &RealFullConstraintState,
) -> Result<()> {
    let valid = constraint
        .valid_draft_prefix(&plan.proposal_token_ids)
        .context("validating dSpark proposal against the active output grammar")?;
    plan.proposal_token_ids.truncate(valid);
    plan.conditional_confidence.truncate(valid);
    plan.selected_drafts = valid;
    plan.minimum_drafts = plan.minimum_drafts.min(valid);
    plan.target_batch_rows = valid + 1;
    Ok(())
}

fn batched_dspark_pair_ranges(active_requests: usize) -> Option<Vec<Range<usize>>> {
    matches!(active_requests, 4 | 8).then(|| {
        (0..active_requests)
            .step_by(2)
            .map(|start| start..start + 2)
            .collect()
    })
}

fn batched_dspark_replay_shape(active_requests: usize) -> Option<(usize, usize, usize, bool)> {
    if active_requests == 0 {
        return None;
    }
    let cohort_ranges = batched_dspark_pair_ranges(active_requests);
    let cohort_width = cohort_ranges
        .as_ref()
        .and_then(|ranges| ranges.first())
        .map_or(active_requests, Range::len);
    let cohort_count = cohort_ranges.as_ref().map_or(1, Vec::len);
    Some((
        active_requests,
        cohort_width,
        cohort_count,
        active_requests == 4 && cohort_width == 2 && cohort_count == 2,
    ))
}

impl RealFullSchedulerRequestExecutor {
    fn replay_dspark_step(
        &self,
        sequence_id: &str,
        request_id: &str,
        placement_version: &str,
        request_id_base: u64,
        target_hidden_taps: [&DeviceBf16Output; 3],
        target_row_start: usize,
        committed_rows: usize,
        absolute_context_start: Option<usize>,
        anchor_token: usize,
    ) -> Result<RealFullDsparkReplayResult> {
        let dspark = self
            .dspark
            .as_ref()
            .context("dSpark replay requires an active request runtime")?;
        let prepared = {
            let mut runtime = dspark
                .lock()
                .map_err(|error| anyhow::anyhow!("locking dSpark prepare phase failed: {error}"))?;
            runtime.prepare_replay(
                sequence_id,
                target_hidden_taps,
                target_row_start,
                committed_rows,
                absolute_context_start,
                anchor_token,
            )?
        };

        // This launch deliberately happens after the request-state guard has
        // been dropped. Three coordinator/Spark barriers must never serialize
        // unrelated requests behind the dSpark bookkeeping mutex.
        let output = match self.execute_prepared_dspark_replay(
            &prepared,
            sequence_id,
            request_id,
            placement_version,
            request_id_base,
            target_hidden_taps,
        ) {
            Ok(output) => output,
            Err(execution_error) => {
                let abort_result = dspark
                    .lock()
                    .map_err(|error| anyhow::anyhow!("locking dSpark abort phase failed: {error}"))?
                    .abort_replay(sequence_id, &prepared);
                if let Err(abort_error) = abort_result {
                    return Err(execution_error.context(format!(
                        "aborting failed dSpark replay also failed: {abort_error:#}"
                    )));
                }
                return Err(execution_error);
            }
        };

        let mut runtime = dspark
            .lock()
            .map_err(|error| anyhow::anyhow!("locking dSpark commit phase failed: {error}"))?;
        let (step, plan) = match runtime.commit_replay(sequence_id, &prepared, output) {
            Ok(result) => result,
            Err(commit_error) => {
                let abort_result = runtime.abort_replay(sequence_id, &prepared);
                if let Err(abort_error) = abort_result {
                    return Err(commit_error.context(format!(
                        "aborting rejected dSpark replay also failed: {abort_error:#}"
                    )));
                }
                return Err(commit_error);
            }
        };
        let mode = runtime.mode;
        let context_tokens = runtime
            .request_context_tokens(sequence_id)
            .context("committed dSpark request lost its context state")?;
        Ok(RealFullDsparkReplayResult {
            step,
            plan,
            mode,
            context_tokens,
        })
    }

    fn execute_prepared_dspark_replay(
        &self,
        prepared: &DsparkPreparedReplay,
        sequence_id: &str,
        request_id: &str,
        placement_version: &str,
        request_id_base: u64,
        target_hidden_taps: [&DeviceBf16Output; 3],
    ) -> Result<DsparkReplayOutput> {
        let replay_started = Instant::now();
        anyhow::ensure!(
            prepared.hidden_size() == self.catalog.facts.hidden_size
                && prepared.vocab_size() == self.catalog.facts.vocab_size,
            "prepared dSpark geometry {}/{} disagrees with target catalog {}/{}",
            prepared.hidden_size(),
            prepared.vocab_size(),
            self.catalog.facts.hidden_size,
            self.catalog.facts.vocab_size,
        );
        anyhow::ensure!(
            prepared.uses_strict_tp4(),
            "prepared dSpark replay lost strict TP4 expert topology"
        );
        anyhow::ensure!(
            self.sparse_tcp_targets.len() == 4,
            "integrated dSpark requires one attached dispatch worker for exactly four TP ranks"
        );
        let target_row_end = prepared
            .target_row_start()
            .checked_add(prepared.committed_main_rows())
            .context("prepared dSpark target row range overflow")?;
        for (tap_index, tap) in target_hidden_taps.iter().enumerate() {
            anyhow::ensure!(
                tap.rows >= target_row_end
                    && tap.values_per_row == prepared.hidden_size()
                    && tap.buffer().device_id == 0,
                "prepared dSpark target tap {tap_index} must cover rows {}..{} at width {} on GPU0, got {}x{} on device {}",
                prepared.target_row_start(),
                target_row_end,
                prepared.hidden_size(),
                tap.rows,
                tap.values_per_row,
                tap.buffer().device_id,
            );
        }
        let storage = self
            .dspark_device_storage
            .as_ref()
            .context("native DeepSeek dSpark numeric launch requires startup-owned GPU0 storage")?;
        let mut storage = storage
            .lock()
            .map_err(|error| anyhow::anyhow!("locking dSpark device storage failed: {error}"))?;
        let storage_plan = storage.plan();
        anyhow::ensure!(
            prepared.request_slot() < storage_plan.max_batch,
            "prepared dSpark request slot {} exceeds startup storage batch {}",
            prepared.request_slot(),
            storage_plan.max_batch,
        );
        let arena = storage.arena();
        let persistent_kv = storage.persistent_kv();
        let entry_buffers = storage
            .entry_buffers()
            .context("binding stable dSpark entry arena views")?;
        let proposal_buffers = storage
            .proposal_buffers()
            .context("binding stable dSpark proposal arena views")?;
        let terminal_buffers = storage
            .terminal_buffers()
            .context("binding stable dSpark joint-terminal arena views")?;
        anyhow::ensure!(
            arena.device_id == 0
                && arena.bytes >= storage_plan.arena_bytes
                && persistent_kv.device_id == 0
                && persistent_kv.bytes >= storage_plan.persistent_kv_bytes
                && entry_buffers.projected_target_main.device_id == 0
                && entry_buffers.projected_target_main.bytes
                    == storage_plan.entry.projected_target_main_bytes
                && entry_buffers.block_workspace.device_id == 0
                && entry_buffers.block_workspace.bytes == storage_plan.entry.block_workspace_bytes
                && entry_buffers.target_tap_concat.device_id == 0
                && entry_buffers.target_tap_concat.bytes
                    == storage_plan.entry.target_tap_concat_bytes
                && entry_buffers.projection_scratch.device_id == 0
                && entry_buffers.projection_scratch.bytes
                    == storage_plan.entry.projection_scratch_bytes
                && proposal_buffers.draft_token_ids.device_id == 0
                && proposal_buffers.draft_token_ids.bytes
                    == storage_plan.proposal.draft_token_ids_bytes
                && proposal_buffers.residual_ping.device_id == 0
                && proposal_buffers.residual_ping.bytes
                    == storage_plan.proposal.residual_ping_bytes
                && proposal_buffers.residual_pong.device_id == 0
                && proposal_buffers.residual_pong.bytes
                    == storage_plan.proposal.residual_pong_bytes
                && proposal_buffers.collapsed_hidden.device_id == 0
                && proposal_buffers.collapsed_hidden.bytes
                    == storage_plan.proposal.collapsed_hidden_bytes
                && proposal_buffers.normalized_hidden.device_id == 0
                && proposal_buffers.normalized_hidden.bytes
                    == storage_plan.proposal.normalized_hidden_bytes
                && proposal_buffers.positions.device_id == 0
                && proposal_buffers.positions.bytes == storage_plan.proposal.positions_bytes
                && proposal_buffers.main_slots.device_id == 0
                && proposal_buffers.main_slots.bytes == storage_plan.proposal.main_slots_bytes
                && proposal_buffers.cos_sin.device_id == 0
                && proposal_buffers.cos_sin.bytes == storage_plan.proposal.cos_sin_bytes
                && proposal_buffers.post_ping.device_id == 0
                && proposal_buffers.post_ping.bytes == storage_plan.proposal.post_ping_bytes
                && proposal_buffers.comb_ping.device_id == 0
                && proposal_buffers.comb_ping.bytes == storage_plan.proposal.comb_ping_bytes
                && proposal_buffers.post_pong.device_id == 0
                && proposal_buffers.post_pong.bytes == storage_plan.proposal.post_pong_bytes
                && proposal_buffers.comb_pong.device_id == 0
                && proposal_buffers.comb_pong.bytes == storage_plan.proposal.comb_pong_bytes
                && proposal_buffers.selected_indices.device_id == 0
                && proposal_buffers.selected_indices.bytes
                    == storage_plan.proposal.selected_indices_bytes
                && proposal_buffers.selected_lengths.device_id == 0
                && proposal_buffers.selected_lengths.bytes
                    == storage_plan.proposal.selected_lengths_bytes
                && terminal_buffers.compact_normalized_hidden.device_id == 0
                && terminal_buffers.compact_normalized_hidden.bytes
                    == storage_plan.terminal.compact_normalized_hidden_bytes
                && terminal_buffers.shared_logits.device_id == 0
                && terminal_buffers.shared_logits.bytes
                    == storage_plan.terminal.shared_logits_bytes
                && terminal_buffers.markov_logits.device_id == 0
                && terminal_buffers.markov_logits.bytes
                    == storage_plan.terminal.markov_logits_bytes
                && terminal_buffers.markov_embeddings.device_id == 0
                && terminal_buffers.markov_embeddings.bytes
                    == storage_plan.terminal.markov_embeddings_bytes
                && terminal_buffers.confidence.device_id == 0
                && terminal_buffers.confidence.bytes == storage_plan.terminal.confidence_bytes
                && terminal_buffers.active_slot_ids.device_id == 0
                && terminal_buffers.active_slot_ids.bytes
                    == storage_plan.terminal.active_slot_ids_bytes
                && terminal_buffers.anchor_token_ids.device_id == 0
                && terminal_buffers.anchor_token_ids.bytes
                    == storage_plan.terminal.anchor_token_ids_bytes
                && terminal_buffers.output_token_ids.device_id == 0
                && terminal_buffers.output_token_ids.bytes
                    == storage_plan.terminal.output_token_ids_bytes
                && storage.entry_projection_graphs_prepared()
                && storage.prompt_prime_graphs_prepared()
                && storage.proposal_entry_graphs_prepared()
                && storage.block_attention_graphs_prepared()
                && storage.block_post_dispatch_graphs_prepared()
                && storage.terminal_collapse_graphs_prepared()
                && storage.terminal_head_graphs_prepared()
                && storage.block_sparse_pre_dispatch_prepared()
                && !storage.stream_ptr().is_null(),
            "native DeepSeek dSpark startup storage lost its GPU0 entry pointer contract"
        );
        let main_context = prepared.main_context_range();
        anyhow::ensure!(
            main_context.len() == prepared.committed_main_rows(),
            "prepared dSpark main context {:?} does not cover {} committed rows",
            main_context,
            prepared.committed_main_rows(),
        );
        storage
            .project_and_prime_target_taps_in_chunks(
                target_hidden_taps,
                prepared.target_row_start(),
                prepared.committed_main_rows(),
                main_context.start,
                prepared.request_slot(),
                self.catalog.facts.rope_theta,
                |_| Ok(()),
            )
            .context("projecting target taps and priming integrated dSpark target-main KV")?;
        let draft_inputs = prepared.draft_input_token_ids();
        anyhow::ensure!(
            draft_inputs.len() == NATIVE_DSPARK_PROPOSAL_TOKENS
                && draft_inputs[1..]
                    .iter()
                    .all(|token| *token == draft_inputs[1]),
            "prepared dSpark proposal inputs must be one anchor followed by four identical noise tokens, got {draft_inputs:?}"
        );
        let proposal_positions = prepared.proposal_position_range();
        anyhow::ensure!(
            proposal_positions.start == main_context.end
                && proposal_positions.len() == NATIVE_DSPARK_PROPOSAL_TOKENS
                && prepared.cache_window_start() <= main_context.end
                && prepared.cache_context_tokens_after()
                    == main_context.end - prepared.cache_window_start(),
            "prepared dSpark proposal/cache ranges drifted from the five-row native ABI"
        );
        storage
            .prepare_proposal_inputs(
                draft_inputs[0],
                draft_inputs[1],
                prepared.request_slot(),
                prepared.cache_window_start(),
                main_context.end,
                prepared.vocab_size(),
                self.catalog.facts.rope_theta,
            )
            .context("preparing integrated dSpark proposal inputs and attention selection")?;
        storage
            .run_proposal_entry(prepared.request_slot())
            .context("running integrated dSpark block-zero proposal mHC pre-mix")?;
        let dispatch_worker = self.sparse_tcp_dispatch_worker.as_ref();
        let block_count = prepared.physical_layer_ids().len();
        anyhow::ensure!(
            block_count == 3,
            "integrated dSpark replay requires exactly three sparse blocks"
        );
        for block_index in 0..block_count {
            storage
                .run_block_attention(block_index, prepared.request_slot())
                .with_context(|| {
                    format!(
                        "running integrated dSpark block {block_index} attention to TP4 dispatch input"
                    )
                })?;
            let pre_dispatch = storage
                .run_block_sparse_pre_dispatch(block_index, prepared.request_slot())
                .with_context(|| {
                    format!(
                        "running integrated dSpark block {block_index} global router and shared expert"
                    )
                })?;
            let dispatch_input = storage
                .read_block_tp4_dispatch_input(pre_dispatch, self.catalog.facts.routed_experts)
                .with_context(|| {
                    format!("reading integrated dSpark block {block_index} TP4 dispatch envelope")
                })?;
            dispatch_worker
                .dispatch_dspark_tp4_block_and_reduce(
                    &self.catalog.facts,
                    block_index,
                    request_id,
                    sequence_id,
                    placement_version,
                    prepared.proposal_position_range().start,
                    request_id_base,
                    dispatch_input,
                )
                .with_context(|| {
                    format!(
                        "dispatching and reducing integrated dSpark block {block_index} experts"
                    )
                })?;
            if block_index + 1 < block_count {
                storage
                    .run_block_post_dispatch(block_index, prepared.request_slot())
                    .with_context(|| {
                        format!(
                            "running integrated dSpark block {block_index} FFN HC-post into block {} attention HC-pre",
                            block_index + 1
                        )
                    })?;
            }
        }
        storage
            .run_terminal_collapse(prepared.request_slot())
            .context("running integrated dSpark terminal FFN HC-post and dual-output collapse")?;
        let terminal_started = Instant::now();
        // Scalar decode still uses the same batch-shaped terminal API at
        // width one. The multi-request serving path enters the joint replay
        // executor and calls this terminal once for all active requests.
        let terminal = storage
            .run_terminal_head_batch(&[prepared.request_slot()], &[draft_inputs[0]])
            .context("running integrated dSpark shared LM/Markov/confidence terminal")?;
        anyhow::ensure!(
            terminal.active_requests == 1
                && terminal.proposal_token_ids.len() == NATIVE_DSPARK_PROPOSAL_TOKENS
                && terminal.conditional_confidence.len() == NATIVE_DSPARK_PROPOSAL_TOKENS,
            "native dSpark single-slot bring-up returned malformed terminal batch"
        );
        Ok(DsparkReplayOutput {
            proposal_token_ids: terminal
                .proposal_token_ids
                .into_iter()
                .map(|token| token as usize)
                .collect(),
            conditional_confidence: terminal.conditional_confidence,
            update_ms: 0.0,
            suffix_ms: elapsed_ms(terminal_started),
            readback_ms: 0.0,
            total_ms: elapsed_ms(replay_started),
        })
    }

    fn start_prepared_dspark_joint_block(
        &self,
        storage: &mut DeepseekV4DsparkDeviceStorage,
        dispatch_worker: &RealFullSchedulerSparseTcpDispatchWorker,
        launches: &[RealFullDsparkReplayLaunch<'_>],
        active_request_range: Range<usize>,
        block_index: usize,
    ) -> Result<RealFullSchedulerDsparkTp4PendingDispatch> {
        anyhow::ensure!(
            active_request_range.start < active_request_range.end
                && active_request_range.end <= launches.len(),
            "integrated dSpark cohort range {:?} exceeds {} launches",
            active_request_range,
            launches.len(),
        );
        let mut hidden_bf16 = Vec::new();
        let mut route_indices = Vec::new();
        let mut route_weights = Vec::new();
        for active_request_index in active_request_range.clone() {
            let launch = &launches[active_request_index];
            let request_slot = launch.prepared.request_slot();
            storage
                .run_block_attention(block_index, request_slot)
                .with_context(|| {
                    format!(
                        "running joint dSpark block {block_index} attention slot {request_slot}"
                    )
                })?;
            let pre_dispatch = storage
                .run_block_sparse_pre_dispatch(block_index, request_slot)
                .with_context(|| {
                    format!(
                        "running joint dSpark block {block_index} router/shared expert slot {request_slot}"
                    )
                })?;
            let dispatch_input = storage
                .read_block_tp4_dispatch_input(pre_dispatch, self.catalog.facts.routed_experts)
                .with_context(|| {
                    format!(
                        "reading joint dSpark block {block_index} TP4 envelope slot {request_slot}"
                    )
                })?;
            storage
                .stage_joint_block_shared_delta(active_request_index, &dispatch_input)
                .with_context(|| {
                    format!(
                        "staging joint dSpark block {block_index} shared delta slot {request_slot}"
                    )
                })?;
            hidden_bf16.extend_from_slice(&dispatch_input.hidden_bf16);
            route_indices.extend_from_slice(&dispatch_input.route_indices);
            route_weights.extend_from_slice(&dispatch_input.route_weights);
        }

        let cohort = &launches[active_request_range.clone()];
        let request_ids = cohort
            .iter()
            .map(|launch| launch.request_id)
            .collect::<Vec<_>>();
        let sequence_ids = cohort
            .iter()
            .map(|launch| launch.sequence_id)
            .collect::<Vec<_>>();
        let token_starts = cohort
            .iter()
            .map(|launch| launch.prepared.proposal_position_range().start)
            .collect::<Vec<_>>();
        let rows = active_request_range
            .len()
            .checked_mul(NATIVE_DSPARK_PROPOSAL_TOKENS)
            .context("joint dSpark block row count overflow")?;
        let (shared_delta, ffn_delta) =
            storage.joint_block_tp4_dispatch_buffers_for_range(active_request_range.clone())?;
        dispatch_worker.start_dspark_tp4_joint_block_dispatch(
            &self.catalog.facts,
            block_index,
            &request_ids,
            &sequence_ids,
            &token_starts,
            cohort[0].placement_version,
            cohort[0].request_id_base,
            DeepseekV4DsparkTp4DispatchInput {
                rows,
                hidden_bf16,
                route_indices,
                route_weights,
                shared_delta,
                ffn_delta,
            },
        )
    }

    fn execute_prepared_dspark_replay_batch(
        &self,
        launches: &[RealFullDsparkReplayLaunch<'_>],
    ) -> Result<Vec<DsparkReplayOutput>> {
        let replay_started = Instant::now();
        anyhow::ensure!(!launches.is_empty(), "joint dSpark replay batch is empty");
        anyhow::ensure!(
            self.sparse_tcp_targets.len() == 4,
            "integrated dSpark requires one attached dispatch worker for exactly four TP ranks"
        );
        for (launch_index, launch) in launches.iter().enumerate() {
            anyhow::ensure!(
                launch.prepared.hidden_size() == self.catalog.facts.hidden_size
                    && launch.prepared.vocab_size() == self.catalog.facts.vocab_size,
                "prepared dSpark launch {launch_index} geometry {}/{} disagrees with target catalog {}/{}",
                launch.prepared.hidden_size(),
                launch.prepared.vocab_size(),
                self.catalog.facts.hidden_size,
                self.catalog.facts.vocab_size,
            );
            anyhow::ensure!(
                launch.prepared.uses_strict_tp4(),
                "prepared dSpark launch {launch_index} lost strict TP4 expert topology"
            );
            anyhow::ensure!(
                !launches[..launch_index]
                    .iter()
                    .any(|prior| prior.prepared.request_slot() == launch.prepared.request_slot()),
                "joint dSpark request slot {} is duplicated",
                launch.prepared.request_slot(),
            );
            let target_row_end = launch
                .prepared
                .target_row_start()
                .checked_add(launch.prepared.committed_main_rows())
                .context("prepared joint dSpark target row range overflow")?;
            for (tap_index, tap) in launch.target_hidden_taps.iter().enumerate() {
                anyhow::ensure!(
                    tap.rows >= target_row_end
                        && tap.values_per_row == launch.prepared.hidden_size()
                        && tap.buffer().device_id == 0,
                    "prepared joint dSpark launch {launch_index} tap {tap_index} must cover rows {}..{} at width {} on GPU0, got {}x{} on device {}",
                    launch.prepared.target_row_start(),
                    target_row_end,
                    launch.prepared.hidden_size(),
                    tap.rows,
                    tap.values_per_row,
                    tap.buffer().device_id,
                );
            }
        }

        let storage = self
            .dspark_device_storage
            .as_ref()
            .context("native DeepSeek dSpark numeric launch requires startup-owned GPU0 storage")?;
        let mut storage = storage
            .lock()
            .map_err(|error| anyhow::anyhow!("locking dSpark device storage failed: {error}"))?;
        let storage_plan = storage.plan();
        anyhow::ensure!(
            launches.len() <= storage_plan.max_batch
                && launches
                    .iter()
                    .all(|launch| launch.prepared.request_slot() < storage_plan.max_batch)
                && storage.entry_projection_graphs_prepared()
                && storage.prompt_prime_graphs_prepared()
                && storage.proposal_entry_graphs_prepared()
                && storage.block_attention_graphs_prepared()
                && storage.block_post_dispatch_graphs_prepared()
                && storage.terminal_collapse_graphs_prepared()
                && storage.terminal_head_graphs_prepared()
                && storage.block_sparse_pre_dispatch_prepared()
                && !storage.stream_ptr().is_null(),
            "native DeepSeek dSpark joint replay lost its startup graph/storage contract"
        );

        let mut active_slots = Vec::with_capacity(launches.len());
        let mut anchor_token_ids = Vec::with_capacity(launches.len());
        for (launch_index, launch) in launches.iter().enumerate() {
            let prepared = launch.prepared;
            let main_context = prepared.main_context_range();
            anyhow::ensure!(
                main_context.len() == prepared.committed_main_rows(),
                "prepared joint dSpark launch {launch_index} main context {:?} does not cover {} committed rows",
                main_context,
                prepared.committed_main_rows(),
            );
            storage
                .project_and_prime_target_taps_in_chunks(
                    launch.target_hidden_taps,
                    prepared.target_row_start(),
                    prepared.committed_main_rows(),
                    main_context.start,
                    prepared.request_slot(),
                    self.catalog.facts.rope_theta,
                    |_| Ok(()),
                )
                .with_context(|| {
                    format!(
                        "projecting target taps and priming joint dSpark launch {launch_index} target-main KV"
                    )
                })?;
            let draft_inputs = prepared.draft_input_token_ids();
            anyhow::ensure!(
                draft_inputs.len() == NATIVE_DSPARK_PROPOSAL_TOKENS
                    && draft_inputs[1..]
                        .iter()
                        .all(|token| *token == draft_inputs[1]),
                "prepared joint dSpark launch {launch_index} inputs must be one anchor followed by four identical noise tokens, got {draft_inputs:?}"
            );
            let proposal_positions = prepared.proposal_position_range();
            anyhow::ensure!(
                proposal_positions.start == main_context.end
                    && proposal_positions.len() == NATIVE_DSPARK_PROPOSAL_TOKENS
                    && prepared.cache_window_start() <= main_context.end
                    && prepared.cache_context_tokens_after()
                        == main_context.end - prepared.cache_window_start(),
                "prepared joint dSpark launch {launch_index} proposal/cache ranges drifted from the five-row native ABI"
            );
            storage
                .prepare_proposal_inputs(
                    draft_inputs[0],
                    draft_inputs[1],
                    prepared.request_slot(),
                    prepared.cache_window_start(),
                    main_context.end,
                    prepared.vocab_size(),
                    self.catalog.facts.rope_theta,
                )
                .with_context(|| {
                    format!("preparing joint dSpark launch {launch_index} proposal inputs")
                })?;
            storage
                .run_proposal_entry(prepared.request_slot())
                .with_context(|| {
                    format!("running joint dSpark launch {launch_index} proposal entry")
                })?;
            active_slots.push(prepared.request_slot());
            anchor_token_ids.push(draft_inputs[0]);
        }

        let dispatch_worker = self.sparse_tcp_dispatch_worker.as_ref();
        let block_count = launches[0].prepared.physical_layer_ids().len();
        anyhow::ensure!(
            block_count == 3
                && launches
                    .iter()
                    .all(|launch| launch.prepared.physical_layer_ids().len() == block_count),
            "integrated joint dSpark replay requires exactly three sparse blocks"
        );
        let cohort_ranges =
            batched_dspark_pair_ranges(launches.len()).unwrap_or_else(|| vec![0..launches.len()]);
        // Preserve the production C4 shape used by GLMRT. Each two-request
        // cohort is jointly issued to every TP rank, but its transport remains
        // pending while GPU0 prepares the other pair. The same pair can then
        // advance to the next dSpark block while its sibling finishes the
        // previous one, producing a real 2x2 wavefront rather than two serial
        // calls or one latency-heavy four-request barrier.
        let mut pending_cohorts = cohort_ranges
            .iter()
            .cloned()
            .map(|range| {
                self.start_prepared_dspark_joint_block(
                    &mut storage,
                    dispatch_worker,
                    launches,
                    range,
                    0,
                )
                .map(Some)
            })
            .collect::<Result<Vec<_>>>()?;
        for block_index in 0..block_count {
            let next_block = block_index + 1;
            for (cohort_index, active_request_range) in cohort_ranges.iter().enumerate() {
                let pending = pending_cohorts[cohort_index].take().with_context(|| {
                    format!(
                        "integrated dSpark cohort {cohort_index} block {block_index} dispatch is missing"
                    )
                })?;
                dispatch_worker
                    .finish_dspark_tp4_joint_block_dispatch(pending)
                    .with_context(|| {
                        format!(
                            "finishing jointly issued dSpark cohort {cohort_index} block {block_index}"
                        )
                    })?;
                if next_block < block_count {
                    for active_request_index in active_request_range.clone() {
                        let launch = &launches[active_request_index];
                        storage
                            .run_block_post_dispatch_from_joint(
                                block_index,
                                active_request_index,
                                launch.prepared.request_slot(),
                            )
                            .with_context(|| {
                                format!(
                                    "running dSpark cohort {cohort_index} block {block_index} post-dispatch slot {}",
                                    launch.prepared.request_slot()
                                )
                            })?;
                    }
                    pending_cohorts[cohort_index] = Some(self.start_prepared_dspark_joint_block(
                        &mut storage,
                        dispatch_worker,
                        launches,
                        active_request_range.clone(),
                        next_block,
                    )?);
                }
            }
        }
        for (active_request_index, launch) in launches.iter().enumerate() {
            storage
                .run_terminal_collapse_from_joint(
                    active_request_index,
                    launch.prepared.request_slot(),
                )
                .with_context(|| {
                    format!(
                        "running joint dSpark terminal collapse slot {}",
                        launch.prepared.request_slot()
                    )
                })?;
        }
        let terminal_started = Instant::now();
        let terminal = storage
            .run_terminal_head_batch(&active_slots, &anchor_token_ids)
            .context("running integrated dSpark joint shared LM/Markov/confidence terminal")?;
        anyhow::ensure!(
            terminal.active_requests == launches.len()
                && terminal.proposal_token_ids.len()
                    == launches.len() * NATIVE_DSPARK_PROPOSAL_TOKENS
                && terminal.conditional_confidence.len()
                    == launches.len() * NATIVE_DSPARK_PROPOSAL_TOKENS,
            "native dSpark joint terminal returned malformed request-major output"
        );
        let suffix_ms = elapsed_ms(terminal_started);
        let total_ms = elapsed_ms(replay_started);
        Ok(terminal
            .proposal_token_ids
            .chunks_exact(NATIVE_DSPARK_PROPOSAL_TOKENS)
            .zip(
                terminal
                    .conditional_confidence
                    .chunks_exact(NATIVE_DSPARK_PROPOSAL_TOKENS),
            )
            .map(|(tokens, confidence)| DsparkReplayOutput {
                proposal_token_ids: tokens.iter().map(|token| *token as usize).collect(),
                conditional_confidence: confidence.to_vec(),
                update_ms: 0.0,
                suffix_ms,
                readback_ms: 0.0,
                total_ms,
            })
            .collect())
    }

    fn take_scheduler_state(
        &self,
        request: &ds4rt_api::RealFullRequest,
        target_radix: Option<(TargetKvRadixReservation, &[usize])>,
    ) -> std::result::Result<BudgetedRealFullSchedulerExecutionState, String> {
        let sequence_id = request.sequence_id.as_str();
        let mut states = self
            .scheduler_states
            .lock()
            .map_err(|err| format!("locking real-full scheduler state map failed: {err}"))?;
        if let Some(mut state) = states.remove(sequence_id) {
            if target_radix.is_some() {
                states.insert(sequence_id.to_owned(), state);
                return Err(format!(
                    "active real-full sequence {sequence_id} received a second target KV radix reservation"
                ));
            }
            match (state.constraint.as_ref(), request.constraint.as_ref()) {
                (None, None) => {}
                (Some(active), Some(requested)) if active.matches_spec(requested) => {}
                _ => {
                    states.insert(sequence_id.to_owned(), state);
                    return Err(format!(
                        "active real-full sequence {sequence_id} changed its constrained-decoding specification"
                    ));
                }
            }
            drop(states);
            let materialization_tokens = request
                .prompt_tokens
                .checked_add(request.generated_token_ids.len())
                .and_then(|tokens| tokens.checked_add(REAL_FULL_DSPARK_MAX_VERIFY_DRAFTS))
                .ok_or_else(|| "target KV materialization frontier overflow".to_owned())?;
            state
                .ensure_target_radix_materialized_through(materialization_tokens)
                .map_err(format_error_chain)?;
            return Ok(state);
        }
        drop(states);
        let capacity_tokens = real_full_sequence_capacity_tokens(
            request.prompt_tokens,
            request.decode_budget,
            self.kv_config.max_tokens,
        )
        .map_err(format_error_chain)?;
        let context_reservation =
            if real_full_internal_sequence(sequence_id) || target_radix.is_some() {
                None
            } else {
                Some(
                    self.context_budget
                        .reserve(capacity_tokens)
                        .map_err(format_error_chain)?,
                )
            };
        let physical_token_base = context_reservation
            .as_ref()
            .map_or(0, RealFullContextTokenReservation::token_base);
        let canonical_capture_arena = real_full_capture_arena_sequence(sequence_id)
            || !real_full_internal_sequence(sequence_id);
        let startup_execution_lane_id = real_full_batched_dspark_prewarm_buffer_bank(sequence_id)
            .map(|buffer_bank| {
                buffer_bank
                    .checked_sub(1)
                    .expect("batched dSpark startup buffer banks are nonzero")
            });
        if let Some(execution_lane_id) = startup_execution_lane_id {
            if execution_lane_id >= self.max_execution_lanes {
                return Err(format!(
                    "batched dSpark startup execution lane {execution_lane_id} exceeds configured lane count {}",
                    self.max_execution_lanes
                ));
            }
        }
        let mut state = if canonical_capture_arena {
            let mut recycled = self.recycled_scheduler_states.lock().map_err(|err| {
                format!("locking recycled real-full scheduler states failed: {err}")
            })?;
            if !real_full_internal_sequence(sequence_id) {
                recycled.retain(|candidate| {
                    !candidate.owned_by_current_thread()
                        || candidate.arena_capacity_tokens() == self.kv_config.max_tokens
                });
            }
            let candidate = recycled
                .iter()
                .enumerate()
                .filter(|(_, candidate)| {
                    candidate.owned_by_current_thread()
                        && candidate.arena_capacity_tokens() == self.kv_config.max_tokens
                })
                .filter(|(_, candidate)| {
                    startup_execution_lane_id
                        .is_none_or(|lane_id| candidate.execution_lane_id == lane_id)
                })
                .min_by_key(|(_, candidate)| candidate.execution_lane_id)
                .map(|(index, _)| index);
            candidate.map(|index| recycled.swap_remove(index))
        } else {
            None
        };
        if state.is_none() && canonical_capture_arena && startup_execution_lane_id.is_none() {
            let mut states = self
                .scheduler_states
                .lock()
                .map_err(|err| format!("locking real-full scheduler state map failed: {err}"))?;
            let candidate_key = states
                .iter()
                .find(|(candidate_sequence_id, candidate)| {
                    real_full_capture_arena_sequence(candidate_sequence_id)
                        && candidate.owned_by_current_thread()
                        && candidate.arena_capacity_tokens() == self.kv_config.max_tokens
                })
                .map(|(candidate_sequence_id, _)| candidate_sequence_id.clone());
            state = candidate_key
                .and_then(|candidate_sequence_id| states.remove(&candidate_sequence_id));
        }
        let mut new_execution_lane_id = startup_execution_lane_id.unwrap_or(0);
        if state.is_none() && !real_full_internal_sequence(sequence_id) {
            let mut occupied_execution_lanes = vec![false; self.max_execution_lanes];
            self
                .scheduler_states
                .lock()
                .map_err(|err| format!("locking real-full scheduler state map failed: {err}"))?
                .values()
                .filter(|candidate| {
                    retain_graph_bound_scheduler_arena(
                        candidate.graph_bound_arena,
                        candidate.arena_capacity_tokens(),
                        self.kv_config.max_tokens,
                    )
                })
                .try_for_each(|candidate| {
                    let occupied = occupied_execution_lanes
                        .get_mut(candidate.execution_lane_id)
                        .ok_or_else(|| {
                            format!(
                                "resident real-full execution lane {} exceeds configured lane count {}",
                                candidate.execution_lane_id, self.max_execution_lanes
                            )
                        })?;
                    *occupied = true;
                    Ok::<(), String>(())
                })?;
            self
                .recycled_scheduler_states
                .lock()
                .map_err(|err| {
                    format!("locking recycled real-full scheduler states failed: {err}")
                })?
                .iter()
                .filter(|candidate| {
                    retain_graph_bound_scheduler_arena(
                        candidate.graph_bound_arena,
                        candidate.arena_capacity_tokens(),
                        self.kv_config.max_tokens,
                    )
                })
                .try_for_each(|candidate| {
                    let occupied = occupied_execution_lanes
                        .get_mut(candidate.execution_lane_id)
                        .ok_or_else(|| {
                            format!(
                                "recycled real-full execution lane {} exceeds configured lane count {}",
                                candidate.execution_lane_id, self.max_execution_lanes
                            )
                        })?;
                    *occupied = true;
                    Ok::<(), String>(())
                })?;
            new_execution_lane_id = occupied_execution_lanes
                .iter()
                .position(|occupied| !occupied)
                .ok_or_else(|| format!(
                    "all {} configured real-full execution lanes are resident; request must remain pending",
                    self.max_execution_lanes
                ))?;
        }
        let mut state = match state {
            Some(mut state) => {
                state
                    .rebind_sequence(sequence_id.to_owned(), capacity_tokens, physical_token_base)
                    .map_err(format_error_chain)?;
                state
            }
            None => {
                let arena_capacity_tokens = if canonical_capture_arena {
                    self.kv_config.max_tokens
                } else {
                    capacity_tokens
                };
                let shared_storage = self
                    .device_kv_storage
                    .lock()
                    .map_err(|err| format!("locking shared device KV storage failed: {err}"))?
                    .clone();
                BudgetedRealFullSchedulerExecutionState {
                    state: RealFullSchedulerExecutionState::new_with_arena_capacity_and_storage(
                        self.kv_config.clone(),
                        &self.catalog.facts,
                        sequence_id.to_owned(),
                        capacity_tokens,
                        arena_capacity_tokens,
                        self.device_kv_pool_config.clone(),
                        shared_storage,
                        physical_token_base,
                    )
                    .map_err(format_error_chain)?,
                    execution_lane_id: new_execution_lane_id,
                    _context_reservation: None,
                    target_radix_reservation: None,
                    bound_target_pages: 0,
                    graph_bound_arena: canonical_capture_arena,
                    snapshot_save_ready: false,
                    snapshot_restore_ms: 0.0,
                    constraint: None,
                }
            }
        };
        state.constraint = request
            .constraint
            .as_ref()
            .map(|spec| self.constraint_compiler.matcher(Arc::clone(spec)))
            .transpose()
            .map_err(format_error_chain)?;
        state._context_reservation = context_reservation;
        state.target_radix_reservation = None;
        state.bound_target_pages = 0;
        state.graph_bound_arena = canonical_capture_arena;
        state.snapshot_save_ready = false;
        state.snapshot_restore_ms = 0.0;
        {
            let mut shared_storage = self
                .device_kv_storage
                .lock()
                .map_err(|err| format!("locking shared device KV storage failed: {err}"))?;
            if shared_storage.is_none() {
                *shared_storage = state.device_kv_storage_handle();
            }
        }
        if !real_full_internal_sequence(sequence_id)
            && request.cached_prompt_tokens > 0
            && target_radix.is_none()
        {
            let snapshot = self.kv_snapshot_load.as_ref().ok_or_else(|| {
                "an external cached-prefix request requires DS4RT_REAL_FULL_KV_SNAPSHOT_LOAD"
                    .to_owned()
            })?;
            if request.cached_prompt_tokens != snapshot.token_count() {
                return Err(format!(
                    "cached-prefix request declares {} tokens but the loaded KV snapshot has {}",
                    request.cached_prompt_tokens,
                    snapshot.token_count()
                ));
            }
            let restore_start = Instant::now();
            snapshot
                .restore(&mut state.state)
                .map_err(format_error_chain)?;
            state.snapshot_restore_ms = elapsed_ms(restore_start);
            eprintln!(
                "real_full_kv_snapshot_restore path={} tokens={} elapsed_ms={:.3}",
                snapshot.root().display(),
                snapshot.token_count(),
                state.snapshot_restore_ms,
            );
        }
        if let Some((reservation, prompt_token_ids)) = target_radix {
            state
                .bind_target_radix_reservation(
                    reservation,
                    prompt_token_ids,
                    &self.target_device_storage,
                )
                .map_err(format_error_chain)?;
        }
        let materialization_tokens = request
            .prompt_tokens
            .checked_add(request.generated_token_ids.len())
            .and_then(|tokens| tokens.checked_add(REAL_FULL_DSPARK_MAX_VERIFY_DRAFTS))
            .ok_or_else(|| "target KV materialization frontier overflow".to_owned())?;
        state
            .ensure_target_radix_materialized_through(materialization_tokens)
            .map_err(format_error_chain)?;
        Ok(state)
    }

    fn store_scheduler_state(
        &self,
        sequence_id: &str,
        state: BudgetedRealFullSchedulerExecutionState,
    ) -> std::result::Result<(), String> {
        let mut states = self
            .scheduler_states
            .lock()
            .map_err(|err| format!("locking real-full scheduler state map failed: {err}"))?;
        if states.insert(sequence_id.to_owned(), state).is_some() {
            return Err(format!(
                "real-full scheduler state for sequence {sequence_id} is already active"
            ));
        }
        Ok(())
    }

    fn finish_scheduler_state(
        &self,
        sequence_id: &str,
        mut state: BudgetedRealFullSchedulerExecutionState,
        final_decode_step: bool,
    ) -> std::result::Result<(), String> {
        if final_decode_step {
            let startup_radix_publish = real_full_startup_target_radix_publish_tokens(sequence_id)
                .is_some()
                && state.target_radix_reservation.is_some();
            if ((!self.kv_snapshot_saves.is_empty() || state.target_radix_reservation.is_some())
                && !real_full_internal_sequence(sequence_id))
                || startup_radix_publish
            {
                state.snapshot_save_ready = true;
                self.store_scheduler_state(sequence_id, state)
            } else {
                self.recycle_scheduler_state(state)
            }
        } else {
            self.store_scheduler_state(sequence_id, state)
        }
    }

    fn recycle_scheduler_state(
        &self,
        mut state: BudgetedRealFullSchedulerExecutionState,
    ) -> std::result::Result<(), String> {
        state._context_reservation = None;
        state.target_radix_reservation = None;
        state.bound_target_pages = 0;
        // Only the max-context arena owns graph-bound KV/DSA addresses worth
        // retaining. Startup also creates short diagnostic states; pooling
        // those would keep one extra device arena per probe until the first
        // external request happened to prune them.
        if retain_graph_bound_scheduler_arena(
            state.graph_bound_arena,
            state.arena_capacity_tokens(),
            self.kv_config.max_tokens,
        ) {
            let mut recycled = self.recycled_scheduler_states.lock().map_err(|err| {
                format!("locking recycled real-full scheduler states failed: {err}")
            })?;
            recycled.push(state);
        }
        Ok(())
    }
}

impl RealFullSchedulerRequestExecutor {
    fn profile_batched_dspark_sps(
        &self,
        sequence_ids: &[String],
        prompts: &[String],
        prompt_tokens: &[usize],
        decode_budget: usize,
        generated_token_ids: &mut [Vec<usize>],
        max_draft_tokens: usize,
    ) -> std::result::Result<(), String> {
        const MEASURED_SAMPLES: usize = 4;
        debug_assert_eq!(sequence_ids.len(), prompts.len());
        debug_assert_eq!(sequence_ids.len(), prompt_tokens.len());
        debug_assert_eq!(sequence_ids.len(), generated_token_ids.len());
        let profile_start = Instant::now();
        eprintln!(
            "real_full_dspark_sps_profile_start lanes={} samples={} source=startup-opt-in",
            self.max_execution_lanes, MEASURED_SAMPLES,
        );
        for request_count in 1..=self.max_execution_lanes {
            let mut rows = Vec::with_capacity(request_count * max_draft_tokens + 1);
            for target_rows in request_count..=request_count * (max_draft_tokens + 1) {
                let total_drafts = target_rows - request_count;
                let base_drafts = total_drafts / request_count;
                let extra_drafts = total_drafts % request_count;
                let widths = (0..request_count)
                    .map(|lane_index| base_drafts + usize::from(lane_index < extra_drafts))
                    .collect::<Vec<_>>();
                debug_assert!(widths.iter().all(|width| *width <= max_draft_tokens));
                let mut samples = Vec::with_capacity(MEASURED_SAMPLES);
                for sample_index in 0..=MEASURED_SAMPLES {
                    // Rotate the active lanes so C1-C3 profile real routing
                    // diversity instead of repeatedly timing one prompt.
                    let lane_indices = (0..request_count)
                        .map(|offset| (sample_index + offset) % sequence_ids.len())
                        .collect::<Vec<_>>();
                    let requests = lane_indices
                        .iter()
                        .zip(&widths)
                        .enumerate()
                        .map(|(request_index, (lane_index, draft_tokens))| {
                            ds4rt_api::RealFullRequest::new_decode_step_for_sequence(
                                REAL_FULL_BATCHED_DSPARK_PREWARM_WIDTH_REQUEST_BASE
                                    + (*draft_tokens as u64)
                                        * REAL_FULL_BATCHED_DSPARK_PREWARM_WIDTH_REQUEST_STRIDE
                                    + request_index as u64,
                                &sequence_ids[*lane_index],
                                &prompts[*lane_index],
                                prompt_tokens[*lane_index],
                                1,
                                generated_token_ids[*lane_index].clone(),
                                generated_token_ids[*lane_index].len(),
                                decode_budget,
                            )
                        })
                        .collect::<Vec<_>>();
                    let sample_start = Instant::now();
                    let cycles =
                        ds4rt_api::RealFullRequestExecutor::execute_real_full_decode_cycle_batch(
                            self, requests,
                        )
                        .into_iter()
                        .collect::<std::result::Result<Vec<_>, _>>()?;
                    let observed_ms = elapsed_ms(sample_start);
                    for (lane_index, (cycle, expected_drafts)) in
                        cycles.iter().zip(&widths).enumerate()
                    {
                        if cycle.info.status != "ready"
                            || cycle.info.request_mtp_verify_rows != *expected_drafts
                        {
                            return Err(format!(
                                "dSpark SPS profile C={request_count} rows={target_rows} lane={lane_index} failed: status={} verify_rows={} expected_rows={} blocker={} failed={:?}",
                                cycle.info.status,
                                cycle.info.request_mtp_verify_rows,
                                expected_drafts,
                                cycle.info.blocker,
                                cycle.info.failed_requirements,
                            ));
                        }
                        if sample_index > 0 && cycle.info.request_coordinator_graph_captures != 0 {
                            return Err(format!(
                                "dSpark SPS measured sample captured {} coordinator graphs at C={request_count} rows={target_rows}",
                                cycle.info.request_coordinator_graph_captures,
                            ));
                        }
                    }
                    for (lane_index, cycle) in lane_indices.into_iter().zip(cycles) {
                        generated_token_ids[lane_index].extend(
                            cycle
                                .generated_tokens
                                .into_iter()
                                .map(|token| token.token_id),
                        );
                    }
                    if sample_index > 0 {
                        samples.push(observed_ms);
                    }
                }
                samples.sort_by(f64::total_cmp);
                let median_ms =
                    (samples[MEASURED_SAMPLES / 2 - 1] + samples[MEASURED_SAMPLES / 2]) * 0.5;
                rows.push((target_rows, median_ms));
                eprintln!(
                    "real_full_dspark_sps_profile requests={} target_rows={} latency_ms={:.3} samples={} source=startup-opt-in",
                    request_count, target_rows, median_ms, MEASURED_SAMPLES,
                );
            }
            self.dspark
                .as_ref()
                .expect("dSpark SPS profiling requires a runtime")
                .lock()
                .map_err(|error| format!("locking dSpark SPS profile failed: {error}"))?
                .install_runtime_cost_profile(request_count, &rows)
                .map_err(format_error_chain)?;
        }
        eprintln!(
            "real_full_dspark_sps_profile_done lanes={} samples={} elapsed_ms={:.3} source=startup-opt-in",
            self.max_execution_lanes,
            MEASURED_SAMPLES,
            elapsed_ms(profile_start),
        );
        Ok(())
    }

    fn batched_dspark_cycles_eligible(&self, requests: &[ds4rt_api::RealFullRequest]) -> bool {
        let batched_startup_prewarm = requests
            .iter()
            .all(|request| real_full_batched_dspark_prewarm_sequence(&request.sequence_id));
        if !(2..=8).contains(&requests.len())
            || self.kv_snapshot_load.is_some()
            || !self.kv_snapshot_saves.is_empty()
            || requests.iter().any(|request| {
                request.generated_token_ids.is_empty()
                    || !request.greedy_sampling
                    || request.disable_speculation
                    || request.constraint.is_some()
                    || (real_full_internal_sequence(&request.sequence_id)
                        && !batched_startup_prewarm)
            })
        {
            return false;
        }
        self.dspark.as_ref().is_some_and(|runtime| {
            runtime
                .lock()
                .map(|runtime| runtime.mode == RealFullDsparkServingMode::Active)
                .unwrap_or(false)
        })
    }

    fn prepare_batched_dspark_cycle(
        &self,
        request: ds4rt_api::RealFullRequest,
    ) -> std::result::Result<PreparedBatchedDsparkCycle, String> {
        prewarm_flashinfer_cudnn_mla_suffix_graphs_for_worker().map_err(format_error_chain)?;
        let request_start = Instant::now();
        let request_timing = real_full_request_timing_enabled();
        let sequence_id = request.sequence_id.clone();
        let generated_tokens = request.generated_token_ids.len();
        let request_id_base = request.request_index.saturating_mul(1_000_000);
        let token_rows =
            real_full_request_token_rows(&request, None).map_err(format_error_chain)?;
        let decode_rows = token_rows.decode_token_ids.len();
        let prefill_chunk_tokens = real_full_prefill_chunk_tokens_for_native_target(
            real_full_request_prefill_chunk_tokens_for_sequence(
                &sequence_id,
                token_rows.prefix_tokens,
                token_rows.prefill_tokens,
            ),
        );
        let dspark_mode = self
            .dspark
            .as_ref()
            .context("paired dSpark cycle requires a request runtime")
            .map_err(format_error_chain)?
            .lock()
            .map(|runtime| runtime.mode)
            .map_err(|error| format!("locking dSpark request executor failed: {error}"))?;
        if dspark_mode != RealFullDsparkServingMode::Active {
            return Err("paired dSpark cycle requires active serving mode".to_owned());
        }
        let mut pending_dspark_plan = self
            .dspark
            .as_ref()
            .expect("paired dSpark cycle resolved its runtime")
            .lock()
            .map_err(|error| format!("locking dSpark request executor failed: {error}"))?
            .prepare_cycle(
                &sequence_id,
                false,
                None,
                real_full_batched_dspark_prewarm_requested_draft_tokens(
                    &sequence_id,
                    request.request_index,
                )
                .or_else(|| real_full_dspark_startup_draft_tokens(&sequence_id)),
                self.dspark_device_storage.as_ref(),
            )
            .map_err(format_error_chain)?;
        if let Some(plan) = pending_dspark_plan.as_mut() {
            let max_useful_drafts = request
                .decode_budget
                .saturating_sub(generated_tokens)
                .saturating_sub(1);
            plan.proposal_token_ids.truncate(max_useful_drafts);
            plan.selected_drafts = plan.proposal_token_ids.len();
            plan.target_batch_rows = plan.selected_drafts + 1;
        }
        let pending_dspark_draft_token_ids = pending_dspark_plan
            .as_ref()
            .map(|plan| plan.proposal_token_ids.clone())
            .unwrap_or_default();
        let mtp_rows = pending_dspark_draft_token_ids.len();
        let dspark_target_hidden_tap_rows =
            if request.decode_budget.saturating_sub(generated_tokens) > 1 {
                decode_rows + mtp_rows
            } else {
                0
            };
        let mut scheduler_token_ids = token_rows.decode_token_ids.clone();
        let committed_input_token_ids = token_rows.decode_token_ids.clone();
        scheduler_token_ids.extend_from_slice(&pending_dspark_draft_token_ids);
        let shape = RealFullSchedulerExecutionShape {
            request_id: request.request_id.clone(),
            sequence_id: sequence_id.clone(),
            placement_version: format!("real-full-api-request-{}", request.request_index),
            prefix_tokens: token_rows.prefix_tokens,
            prefill_tokens: token_rows.prefill_tokens,
            prefill_chunk_tokens,
            decode_rows,
            mtp_rows,
            mtp_accepted_rows: 0,
            prefill_token_ids: token_rows.prefill_token_ids,
            decode_token_ids: Some(scheduler_token_ids),
            lm_head_sampling: request_lm_head_sampling_options(&request),
        };
        let state = match self.take_scheduler_state(&request, None) {
            Ok(state) => state,
            Err(error) => {
                if let Some(plan) = pending_dspark_plan.take() {
                    if let Some(dspark) = self.dspark.as_ref() {
                        if let Ok(mut dspark) = dspark.lock() {
                            dspark.restore_verification(&sequence_id, plan);
                        }
                    }
                }
                return Err(error);
            }
        };
        let snapshot_restore_ms = state.snapshot_restore_ms;
        let buffer_bank = state.execution_lane_id + 1;
        if request_timing {
            eprintln!(
                "real_full_request_timing request_id={} stage=paired_prepare total_ms={:.3} prefix_tokens={} decode_rows={} mtp_rows={} execution_lane={} buffer_bank={}",
                request.request_id,
                elapsed_ms(request_start),
                token_rows.prefix_tokens,
                decode_rows,
                mtp_rows,
                state.execution_lane_id,
                buffer_bank,
            );
        }
        Ok(PreparedBatchedDsparkCycle {
            request,
            request_start,
            request_timing,
            sequence_id,
            generated_tokens,
            request_id_base,
            token_prefix_tokens: token_rows.prefix_tokens,
            token_prefill_tokens: token_rows.prefill_tokens,
            committed_input_token_ids,
            decode_rows,
            scheduler_start: Instant::now(),
            shape,
            state,
            buffer_bank,
            pending_dspark_plan,
            pending_dspark_draft_token_ids,
            dspark_target_hidden_tap_rows,
            snapshot_restore_ms,
        })
    }

    fn restore_prepared_batched_dspark_cycle(&self, mut prepared: PreparedBatchedDsparkCycle) {
        if let Some(plan) = prepared.pending_dspark_plan.take() {
            if let Some(dspark) = self.dspark.as_ref() {
                if let Ok(mut dspark) = dspark.lock() {
                    dspark.restore_verification(&prepared.sequence_id, plan);
                }
            }
        }
        let _ = self.store_scheduler_state(&prepared.sequence_id, prepared.state);
    }

    fn refresh_prepared_batched_dspark_shape(prepared: &mut PreparedBatchedDsparkCycle) {
        prepared.pending_dspark_draft_token_ids = prepared
            .pending_dspark_plan
            .as_ref()
            .map(|plan| plan.proposal_token_ids.clone())
            .unwrap_or_default();
        let mtp_rows = prepared.pending_dspark_draft_token_ids.len();
        prepared.shape.mtp_rows = mtp_rows;
        let mut scheduler_token_ids = prepared.committed_input_token_ids.clone();
        scheduler_token_ids.extend_from_slice(&prepared.pending_dspark_draft_token_ids);
        prepared.shape.decode_token_ids = Some(scheduler_token_ids);
        prepared.dspark_target_hidden_tap_rows = if prepared
            .request
            .decode_budget
            .saturating_sub(prepared.generated_tokens)
            > 1
        {
            prepared.decode_rows + mtp_rows
        } else {
            0
        };
    }

    fn replan_prepared_batched_dspark_cycles(
        &self,
        prepared: &mut [PreparedBatchedDsparkCycle],
    ) -> std::result::Result<(), String> {
        let context_tokens = prepared
            .iter()
            .map(|cycle| cycle.token_prefix_tokens + cycle.token_prefill_tokens)
            .collect::<Vec<_>>();
        let max_useful_drafts = prepared
            .iter()
            .map(|cycle| {
                cycle
                    .request
                    .decode_budget
                    .saturating_sub(cycle.generated_tokens)
                    .saturating_sub(1)
            })
            .collect::<Vec<_>>();
        let jointly_adaptive = prepared.iter().all(|cycle| {
            cycle
                .pending_dspark_plan
                .as_ref()
                .is_some_and(|plan| plan.calibration_eligible)
        });

        if jointly_adaptive {
            let schedule = {
                let plans = prepared
                    .iter()
                    .zip(&max_useful_drafts)
                    .map(|(cycle, max_useful)| {
                        (
                            cycle
                                .pending_dspark_plan
                                .as_ref()
                                .expect("jointly adaptive dSpark cycle has a plan"),
                            *max_useful,
                        )
                    })
                    .collect::<Vec<_>>();
                self.dspark
                    .as_ref()
                    .expect("joint dSpark scheduling requires a runtime")
                    .lock()
                    .map_err(|error| format!("locking dSpark joint scheduler failed: {error}"))?
                    .joint_schedule(&plans, &context_tokens)
                    .map_err(format_error_chain)?
            };
            for ((cycle, max_useful), selected_drafts) in prepared
                .iter_mut()
                .zip(&max_useful_drafts)
                .zip(&schedule.prefix_lengths)
            {
                let plan = cycle
                    .pending_dspark_plan
                    .as_mut()
                    .expect("joint dSpark selection has a request plan");
                plan.minimum_drafts = plan.minimum_drafts.min(*max_useful);
                plan.apply_joint_selection(
                    *selected_drafts,
                    schedule.target_batch_rows,
                    schedule.expected_committed_tokens,
                    schedule.expected_tokens_per_second,
                )
                .map_err(format_error_chain)?;
                Self::refresh_prepared_batched_dspark_shape(cycle);
            }
            if real_full_dspark_trace_enabled() {
                eprintln!(
                    "real_full_dspark_joint_plan requests={} contexts={:?} selected_drafts={:?} target_rows={} expected_tokens={:.4} expected_tps={:.3}",
                    prepared.len(),
                    context_tokens,
                    schedule.prefix_lengths,
                    schedule.target_batch_rows,
                    schedule.expected_committed_tokens,
                    schedule.expected_tokens_per_second,
                );
            }
        } else {
            for cycle in prepared.iter_mut() {
                Self::refresh_prepared_batched_dspark_shape(cycle);
            }
        }

        let mut dspark = self
            .dspark
            .as_ref()
            .expect("issued dSpark plans require a runtime")
            .lock()
            .map_err(|error| format!("locking dSpark issued-plan tracker failed: {error}"))?;
        for cycle in prepared {
            if let Some(plan) = cycle.pending_dspark_plan.as_ref() {
                dspark.record_issued_plan(&cycle.sequence_id, plan);
            }
        }
        Ok(())
    }

    fn prepare_batched_dspark_finish(
        &self,
        mut prepared: PreparedBatchedDsparkCycle,
        execution: RealFullSchedulerDeviceExecution,
        paired_target_samples: Option<RealLmHeadBatchScoreForHidden>,
    ) -> std::result::Result<PreparedBatchedDsparkFinish, String> {
        let target_submit_ms = elapsed_ms(prepared.scheduler_start);
        let mut report = execution.report;
        let probe = execution.sparse_tcp_dispatch;
        let mut target_hidden = execution.final_target_device_hidden;
        let target_hidden_taps = execution.target_device_hidden_taps;
        let mut cycle_token_ids = Vec::new();
        let mut terminal_sample = None;
        let dspark_cache_update;
        if prepared.pending_dspark_draft_token_ids.is_empty() {
            let anchor_token = match report
                .terminal_lm_head_sample
                .top_token_id
                .context("paired dSpark scalar step requires a target token")
            {
                Ok(anchor_token) => anchor_token,
                Err(error) => {
                    self.restore_prepared_batched_dspark_cycle(prepared);
                    return Err(format_error_chain(error));
                }
            };
            cycle_token_ids.push(anchor_token);
            dspark_cache_update = Some((1, anchor_token));
        } else {
            let target_hidden = match target_hidden.take() {
                Some(hidden) => hidden,
                None => {
                    self.restore_prepared_batched_dspark_cycle(prepared);
                    return Err(
                        "paired dSpark verification has no retained target hidden batch".to_owned(),
                    );
                }
            };
            let suffix_rows = prepared.decode_rows + prepared.pending_dspark_draft_token_ids.len();
            let target_sampling_start = Instant::now();
            let target_samples = match paired_target_samples {
                Some(samples) => Ok(samples),
                None => real_full_target_token_samples(&self.catalog, &target_hidden, suffix_rows),
            };
            let target_samples = match target_samples {
                Ok(samples) => samples,
                Err(error) => {
                    self.restore_prepared_batched_dspark_cycle(prepared);
                    return Err(format_error_chain(error));
                }
            };
            let target_token_ids = target_samples.top_token_ids.as_slice();
            let acceptance = match real_full_speculative_acceptance(
                prepared.pending_dspark_draft_token_ids.as_slice(),
                target_token_ids,
                true,
                prepared
                    .request
                    .decode_budget
                    .saturating_sub(prepared.generated_tokens),
            ) {
                Ok(acceptance) => acceptance,
                Err(error) => {
                    self.restore_prepared_batched_dspark_cycle(prepared);
                    return Err(format_error_chain(error));
                }
            };
            let accepted_draft_tokens = acceptance.accepted_draft_tokens;
            let terminal_target_index = acceptance.terminal_target_index;
            cycle_token_ids.extend_from_slice(&target_token_ids[..=terminal_target_index]);
            prepared.committed_input_token_ids.extend_from_slice(
                &prepared.pending_dspark_draft_token_ids[..accepted_draft_tokens],
            );
            let tentative_token_start =
                prepared.token_prefix_tokens + prepared.token_prefill_tokens + prepared.decode_rows;
            if let Err(error) = prepared.state.resolve_mtp_tentative_writes(
                tentative_token_start,
                prepared.pending_dspark_draft_token_ids.len(),
                accepted_draft_tokens,
            ) {
                self.restore_prepared_batched_dspark_cycle(prepared);
                return Err(format_error_chain(error));
            }
            report.committed_mtp_writes =
                accepted_draft_tokens * self.catalog.facts.num_hidden_layers;
            report.discarded_mtp_writes = (prepared.pending_dspark_draft_token_ids.len()
                - accepted_draft_tokens)
                * self.catalog.facts.num_hidden_layers;
            report.request_mtp_accepted_rows = accepted_draft_tokens;
            terminal_sample = Some(RealFullSpeculativeTerminalSample {
                hidden_dim: self.catalog.facts.hidden_size,
                vocab_size: target_samples.vocab_size,
                top_token_id: target_samples.top_token_ids[terminal_target_index],
                sampled_token_id: target_token_ids[terminal_target_index],
                sample_top_k: 1,
                sample_top_p: 1.0,
                argmax_backend: target_samples.argmax_kernel_backend,
                sampler_backend: target_samples.argmax_kernel_backend,
                accepted_draft_tokens,
                report_mtp_acceptance: true,
            });
            dspark_cache_update = Some((
                accepted_draft_tokens + 1,
                target_token_ids[terminal_target_index],
            ));
            let plan = prepared
                .pending_dspark_plan
                .as_ref()
                .expect("paired dSpark drafts came from a pending plan");
            let calibration = self
                .dspark
                .as_ref()
                .expect("paired dSpark cycle has a runtime")
                .lock()
                .map_err(|error| format!("locking dSpark confidence calibrator failed: {error}"));
            let (next_confidence_logit_bias, calibration_variance, calibration_cycles) =
                match calibration {
                    Ok(mut runtime) => runtime.observe_verification(
                        &prepared.sequence_id,
                        plan,
                        accepted_draft_tokens,
                    ),
                    Err(error) => {
                        self.restore_prepared_batched_dspark_cycle(prepared);
                        return Err(error);
                    }
                };
            if real_full_dspark_trace_enabled() {
                eprintln!(
                    "real_full_dspark_acceptance request_id={} sequence_id={} target_context={} drafts={} accepted={} emitted={} full_match={} target_rows={} expected_tokens={:.4} expected_tps={:.3} confidence_logit_bias={:.4} next_confidence_logit_bias={:.4} calibration_variance={:.6} calibration_cycles={} target_submit_ms={:.3} target_sampling_ms={:.3} paired=true",
                    prepared.request.request_id,
                    prepared.sequence_id,
                    prepared.token_prefix_tokens + prepared.token_prefill_tokens,
                    prepared.pending_dspark_draft_token_ids.len(),
                    accepted_draft_tokens,
                    cycle_token_ids.len(),
                    acceptance.full_match_bonus,
                    plan.target_batch_rows,
                    plan.expected_committed_tokens,
                    plan.expected_tokens_per_second,
                    plan.confidence_logit_bias,
                    next_confidence_logit_bias,
                    calibration_variance,
                    calibration_cycles,
                    target_submit_ms,
                    elapsed_ms(target_sampling_start),
                );
            }
        }
        report.request_mtp_verify_rows = prepared.pending_dspark_draft_token_ids.len();
        let final_decode_step = {
            let emitted_tokens = cycle_token_ids.len().max(1);
            prepared.generated_tokens + emitted_tokens >= prepared.request.decode_budget
        };
        let mut replay_prepared = None;
        if let Some(taps) = target_hidden_taps.as_ref() {
            if !final_decode_step {
                let (committed_rows, anchor_token) = match dspark_cache_update
                    .context("paired dSpark target taps require a cache update")
                {
                    Ok(update) => update,
                    Err(error) => {
                        self.restore_prepared_batched_dspark_cycle(prepared);
                        return Err(format_error_chain(error));
                    }
                };
                if taps.rows != prepared.dspark_target_hidden_tap_rows
                    || committed_rows > taps.rows
                    || taps.layer_ids != dspark_target_hidden_tap_layer_ids()
                {
                    let error = format!(
                        "paired dSpark expected {} target rows with {} committed, got rows={} layers={:?}",
                        prepared.dspark_target_hidden_tap_rows,
                        committed_rows,
                        taps.rows,
                        taps.layer_ids
                    );
                    self.restore_prepared_batched_dspark_cycle(prepared);
                    return Err(error);
                }
                let tap_refs = std::array::from_fn(|index| &taps.values[index]);
                let dspark = self
                    .dspark
                    .as_ref()
                    .expect("paired dSpark replay requires an active runtime");
                let replay = match dspark
                    .lock()
                    .map_err(|error| format!("locking joint dSpark prepare phase failed: {error}"))
                    .and_then(|mut runtime| {
                        runtime
                            .prepare_replay(
                                &prepared.sequence_id,
                                tap_refs,
                                0,
                                committed_rows,
                                None,
                                anchor_token,
                            )
                            .map_err(format_error_chain)
                    }) {
                    Ok(replay) => replay,
                    Err(error) => {
                        self.restore_prepared_batched_dspark_cycle(prepared);
                        return Err(error);
                    }
                };
                replay_prepared = Some(replay);
            }
        }
        Ok(PreparedBatchedDsparkFinish {
            prepared,
            target_submit_ms,
            report,
            probe,
            target_hidden_taps,
            cycle_token_ids,
            terminal_sample,
            final_decode_step,
            replay_prepared,
            replay_result: None,
        })
    }

    fn complete_batched_dspark_finish(
        &self,
        finish: PreparedBatchedDsparkFinish,
    ) -> std::result::Result<ds4rt_api::RealFullDecodeCycle, String> {
        let PreparedBatchedDsparkFinish {
            mut prepared,
            target_submit_ms,
            mut report,
            probe,
            target_hidden_taps: _,
            cycle_token_ids,
            terminal_sample,
            final_decode_step,
            replay_prepared: _,
            replay_result,
        } = finish;
        if let Some(replay) = replay_result.as_ref() {
            if real_full_dspark_trace_enabled() {
                eprintln!(
                    "real_full_dspark_step request_id={} sequence_id={} mode={:?} target_context={} draft_context_before={} committed_rows={} draft_context_after={} anchor_token={} selected_drafts={} target_batch_rows={} expected_tokens={:.4} expected_tps={:.3} update_ms={:.3} suffix_ms={:.3} readback_ms={:.3} dspark_total_ms={:.3} selected_proposals={:?} proposals={:?} confidence={:?} paired=true joint=true",
                    prepared.request.request_id,
                    prepared.sequence_id,
                    replay.mode,
                    prepared.token_prefix_tokens + prepared.token_prefill_tokens,
                    replay.step.context_tokens,
                    replay.step.committed_rows,
                    replay.context_tokens,
                    replay.step.anchor_token,
                    replay.plan.selected_drafts,
                    replay.plan.target_batch_rows,
                    replay.plan.expected_committed_tokens,
                    replay.plan.expected_tokens_per_second,
                    replay.step.update_ms,
                    replay.step.suffix_ms,
                    replay.step.readback_ms,
                    replay.step.total_ms,
                    replay.plan.proposal_token_ids,
                    replay.step.proposal_token_ids,
                    replay.step.conditional_confidence,
                );
            }
        }
        if let Err(error) = prepared.state.record_processed_token_ids(
            prepared.token_prefix_tokens,
            &prepared.committed_input_token_ids,
        ) {
            self.restore_prepared_batched_dspark_cycle(prepared);
            return Err(format_error_chain(error));
        }
        if let Err(error) =
            self.finish_scheduler_state(&prepared.sequence_id, prepared.state, final_decode_step)
        {
            return Err(error);
        }
        if prepared.request.greedy_sampling {
            report.terminal_lm_head_sample.sampled_token_id =
                report.terminal_lm_head_sample.top_token_id;
            report.terminal_lm_head_sample.sample_top_k = Some(1);
            report.terminal_lm_head_sample.sample_top_p = Some(1.0);
            report.terminal_lm_head_sample.sampler_kernel_backend =
                report.terminal_lm_head_sample.argmax_kernel_backend;
        }
        if let Some(sample) = terminal_sample.as_ref() {
            apply_speculative_terminal_sample_to_report(&mut report, sample);
        }
        let sampled_token_text =
            self.decode_sampled_token_text_cached(report.terminal_lm_head_sample.sampled_token_id);
        let mut info = real_full_info_from_request_execution(
            &self.base_info,
            &self.catalog.snapshot_path,
            &report,
            sampled_token_text,
        );
        info.request_kv_snapshot_restore_ms = prepared.snapshot_restore_ms;
        apply_sparse_tcp_dispatch_probe(&mut info, self.sparse_tcp_targets.len(), &probe);
        if let Some(sample) = terminal_sample.as_ref() {
            if sample.report_mtp_acceptance {
                info.request_mtp_accepted_rows = sample.accepted_draft_tokens;
            }
            info.scheduler_terminal_lm_head_top_token_id = Some(sample.top_token_id);
            info.scheduler_terminal_lm_head_sampled_token_id = Some(sample.sampled_token_id);
            info.scheduler_terminal_lm_head_sampled_text =
                self.decode_sampled_token_text_cached(Some(sample.sampled_token_id));
            info.scheduler_terminal_lm_head_sample_top_k = Some(sample.sample_top_k);
            info.scheduler_terminal_lm_head_sample_top_p = Some(sample.sample_top_p);
            info.scheduler_terminal_lm_head_argmax_backend = Some(sample.argmax_backend.to_owned());
            info.scheduler_terminal_lm_head_sampler_backend =
                Some(sample.sampler_backend.to_owned());
        }
        if prepared.request_timing {
            eprintln!(
                "real_full_request_timing request_id={} stage=paired_finish scheduler_ms={:.3} total_ms={:.3} status={} sampled_token_id={:?}",
                prepared.request.request_id,
                target_submit_ms,
                elapsed_ms(prepared.request_start),
                info.status,
                info.scheduler_terminal_lm_head_sampled_token_id,
            );
        }
        let generated_tokens = cycle_token_ids
            .into_iter()
            .map(|token_id| ds4rt_api::RealFullGeneratedToken {
                token_id,
                text: self.decode_sampled_token_text_cached(Some(token_id)),
            })
            .collect();
        Ok(ds4rt_api::RealFullDecodeCycle {
            info,
            generated_tokens,
        })
    }

    fn abort_batched_dspark_replays(&self, finishes: &[PreparedBatchedDsparkFinish]) -> Result<()> {
        if finishes
            .iter()
            .all(|finish| finish.replay_prepared.is_none())
        {
            return Ok(());
        }
        let mut runtime = self
            .dspark
            .as_ref()
            .context("prepared joint dSpark replays lost their request runtime")?
            .lock()
            .map_err(|error| anyhow::anyhow!("locking joint dSpark abort phase failed: {error}"))?;
        let mut first_error = None;
        for finish in finishes {
            let Some(replay) = finish.replay_prepared.as_ref() else {
                continue;
            };
            if let Err(error) = runtime.abort_replay(&finish.prepared.sequence_id, replay) {
                first_error.get_or_insert_with(|| {
                    error.context(format!(
                        "aborting joint dSpark replay for {}",
                        finish.prepared.sequence_id
                    ))
                });
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    fn commit_batched_dspark_replays(
        &self,
        finishes: &mut [PreparedBatchedDsparkFinish],
        replay_indices: &[usize],
        replay_outputs: Vec<DsparkReplayOutput>,
    ) -> Result<()> {
        anyhow::ensure!(
            !replay_indices.is_empty() && replay_indices.len() == replay_outputs.len(),
            "joint dSpark commit has {} replay indices and {} outputs",
            replay_indices.len(),
            replay_outputs.len(),
        );
        let mut runtime = self
            .dspark
            .as_ref()
            .context("joint dSpark commit lost its request runtime")?
            .lock()
            .map_err(|error| {
                anyhow::anyhow!("locking joint dSpark commit phase failed: {error}")
            })?;

        // Validate every terminal result before advancing any request cache.
        // This keeps malformed joint output from partially committing a batch.
        for (&finish_index, output) in replay_indices.iter().zip(&replay_outputs) {
            let finish = finishes
                .get(finish_index)
                .context("joint dSpark replay index exceeds prepared finishes")?;
            let replay = finish
                .replay_prepared
                .as_ref()
                .context("joint dSpark replay index has no prepared replay")?;
            let request = runtime
                .requests
                .get(&finish.prepared.sequence_id)
                .with_context(|| {
                    format!(
                        "dSpark request state is missing for {}",
                        finish.prepared.sequence_id
                    )
                })?;
            runtime
                .engine
                .validate_replay_output(&request.cache, replay, output.clone())
                .with_context(|| {
                    format!(
                        "validating joint dSpark output for {}",
                        finish.prepared.sequence_id
                    )
                })?;
        }

        for (commit_index, (&finish_index, output)) in
            replay_indices.iter().zip(replay_outputs).enumerate()
        {
            let finish = finishes
                .get_mut(finish_index)
                .context("joint dSpark replay index exceeds prepared finishes")?;
            let replay = finish
                .replay_prepared
                .as_ref()
                .context("joint dSpark replay index has no prepared replay")?;
            let (step, plan) = match runtime.commit_replay(
                &finish.prepared.sequence_id,
                replay,
                output,
            ) {
                Ok(result) => result,
                Err(commit_error) => {
                    let mut abort_error = None;
                    for remaining_index in &replay_indices[commit_index..] {
                        let remaining = &finishes[*remaining_index];
                        let remaining_replay = remaining
                            .replay_prepared
                            .as_ref()
                            .expect("remaining joint replay is prepared");
                        if let Err(error) =
                            runtime.abort_replay(&remaining.prepared.sequence_id, remaining_replay)
                        {
                            abort_error.get_or_insert(error);
                        }
                    }
                    return match abort_error {
                        Some(abort_error) => Err(commit_error.context(format!(
                            "aborting remaining rejected joint dSpark replays also failed: {abort_error:#}"
                        ))),
                        None => Err(commit_error),
                    };
                }
            };
            let mode = runtime.mode;
            let context_tokens = runtime
                .request_context_tokens(&finish.prepared.sequence_id)
                .context("committed joint dSpark request lost its context state")?;
            finish.replay_result = Some(RealFullDsparkReplayResult {
                step,
                plan,
                mode,
                context_tokens,
            });
        }
        Ok(())
    }

    fn execute_real_full_decode_cycle_inner(
        &self,
        mut request: ds4rt_api::RealFullRequest,
    ) -> std::result::Result<ds4rt_api::RealFullDecodeCycle, String> {
        prewarm_flashinfer_cudnn_mla_suffix_graphs_for_worker().map_err(format_error_chain)?;
        let request_start = Instant::now();
        let request_id = request.request_id.clone();
        let request_index = request.request_index;
        let prompt_tokens_hint = request.prompt_tokens;
        let max_tokens = request.max_tokens;
        let generated_tokens = request.generated_token_ids.len();
        let request_timing = real_full_request_timing_enabled();
        let sequence_id = request.sequence_id.clone();
        let final_decode_step = request.decode_step_index + 1 >= request.decode_budget;
        if request_timing {
            eprintln!(
                "real_full_request_timing request_id={} request_index={} stage=start prompt_tokens_hint={} generated_tokens={} max_tokens={}",
                request_id, request_index, prompt_tokens_hint, generated_tokens, max_tokens
            );
        }
        let should_tokenize_prompt = request.generated_token_ids.is_empty();
        let tokenize_start = Instant::now();
        let prompt_token_ids = if should_tokenize_prompt {
            let tokenizer = self
                .tokenizer
                .lock()
                .map_err(|err| format!("locking real-full tokenizer failed: {err}"))?;
            request_prompt_token_ids(&tokenizer, &request).map_err(format_error_chain)?
        } else {
            None
        };
        if should_tokenize_prompt && !real_full_internal_sequence(&sequence_id) {
            if let Some(snapshot) = self.kv_snapshot_load.as_ref() {
                let prompt_token_ids = prompt_token_ids
                    .as_ref()
                    .expect("an initial request was tokenized");
                let snapshot_tokens = snapshot.token_count();
                if prompt_token_ids.len() <= snapshot_tokens {
                    return Err(format!(
                        "loaded KV snapshot has {snapshot_tokens} tokens but request {} has only {}; at least one uncached suffix token is required",
                        request.request_id,
                        prompt_token_ids.len()
                    ));
                }
                if prompt_token_ids[..snapshot_tokens] != *snapshot.token_ids() {
                    let mismatch = prompt_token_ids[..snapshot_tokens]
                        .iter()
                        .zip(snapshot.token_ids())
                        .position(|(request_token, snapshot_token)| request_token != snapshot_token)
                        .expect("unequal equal-length token slices have a mismatch");
                    return Err(format!(
                        "request {} does not match loaded KV snapshot {} at token {mismatch}: request={} snapshot={}",
                        request.request_id,
                        snapshot.root().display(),
                        prompt_token_ids[mismatch],
                        snapshot.token_ids()[mismatch],
                    ));
                }
                request.cached_prompt_tokens = snapshot_tokens;
            }
        }
        let mut target_radix_reservation = None;
        if should_tokenize_prompt
            && request.cached_prompt_tokens == 0
            && (!real_full_internal_sequence(&sequence_id)
                || (real_full_capture_arena_sequence(&sequence_id)
                    && !real_full_batched_dspark_prewarm_sequence(&sequence_id)
                    // Workspace sizing requests must execute their full
                    // physical shape even if an earlier synthetic startup
                    // seed has populated the radix. Only the designated
                    // publisher may bind a reservation in this namespace.
                    && (!real_full_startup_workspace_sizing_sequence(&sequence_id)
                        || real_full_startup_target_radix_publish_tokens(&sequence_id).is_some())))
            && self.kv_snapshot_load.is_none()
            && self.kv_snapshot_saves.is_empty()
        {
            let prompt_ids = prompt_token_ids
                .as_ref()
                .expect("an initial external request was tokenized");
            let capacity_tokens = real_full_sequence_capacity_tokens(
                request.prompt_tokens,
                request.decode_budget,
                self.kv_config.max_tokens,
            )
            .map_err(format_error_chain)?;
            // DeepSeek's compressed-attention KV pages are radix-owned, but
            // the C4/C128 compressor accumulators are execution-lane-local.
            // Until those accumulators are restored or replayed at a hit, an
            // external continuation would consume stale state from the lane's
            // previous sequence. Keep publishing exact target prefixes, but
            // admit a hit only for models without sequence-local compressor
            // state. Internal startup sequences retain cache-hit coverage for
            // graph and allocator qualification.
            let reusable_prompt_tokens = prompt_ids.len().saturating_sub(1);
            let reusable_prompt_ids = if real_full_internal_sequence(&sequence_id)
                || real_full_target_radix_reuse_supported(&self.catalog.facts)
            {
                &prompt_ids[..reusable_prompt_tokens]
            } else {
                &prompt_ids[..0]
            };
            let mut reservation = self
                .target_kv_radix
                .reserve(reusable_prompt_ids, capacity_tokens)
                .map_err(format_error_chain)?;
            if request.decode_budget > 1 && !request.disable_speculation {
                if let Some(dspark) = self.dspark.as_ref() {
                    let dspark = dspark
                        .lock()
                        .map_err(|error| format!("locking dSpark prefix cache failed: {error}"))?;
                    if dspark.mode == RealFullDsparkServingMode::Active
                        && dspark.cache_mode == RealFullDsparkCacheMode::PromptSwa
                    {
                        let target_match = reservation.matched_prefix_tokens();
                        let aligned_match = dspark
                            .tail_cache
                            .longest_exact_prefix_tokens(prompt_ids, target_match);
                        if aligned_match < target_match {
                            drop(dspark);
                            drop(reservation);
                            reservation = self
                                .target_kv_radix
                                .reserve(&prompt_ids[..aligned_match], capacity_tokens)
                                .map_err(format_error_chain)?;
                            if reservation.matched_prefix_tokens() != aligned_match {
                                return Err(format!(
                                    "target/dSpark aligned radix reservation matched {} tokens, expected {aligned_match}",
                                    reservation.matched_prefix_tokens(),
                                ));
                            }
                            eprintln!(
                                "real_full_dspark_radix_align sequence_id={} target_match={} aligned_match={} recompute_tokens={}",
                                sequence_id,
                                target_match,
                                aligned_match,
                                target_match - aligned_match,
                            );
                        }
                    }
                }
            }
            request.cached_prompt_tokens = reservation.matched_prefix_tokens();
            target_radix_reservation = Some(reservation);
        }
        let stateful_decode_step = generated_tokens > 0;
        let token_rows = real_full_request_token_rows(&request, prompt_token_ids.clone())
            .map_err(format_error_chain)?;
        let committed_prefix_tokens = token_rows.prefix_tokens;
        let mut committed_input_token_ids = token_rows
            .prefill_token_ids
            .iter()
            .flatten()
            .copied()
            .chain(token_rows.decode_token_ids.iter().copied())
            .collect::<Vec<_>>();
        let decode_rows = token_rows.decode_token_ids.len();
        let planned_prefill_chunk_tokens = if request.cached_prompt_tokens > 0
            && sequence_id.starts_with("real-full-startup-dsa-selector-seed-")
        {
            // The selector sweep is graph-capture infrastructure, not a
            // serving policy probe. Preserve its requested physical query
            // bucket instead of applying the ordinary cached-suffix 256-row
            // cap; otherwise the log says "512" while only two 256-row
            // identities are captured.
            token_rows.prefill_tokens.max(1)
        } else {
            real_full_request_prefill_chunk_tokens_for_sequence(
                &sequence_id,
                token_rows.prefix_tokens,
                token_rows.prefill_tokens,
            )
        };
        let prefill_chunk_tokens =
            real_full_prefill_chunk_tokens_for_native_target(planned_prefill_chunk_tokens);
        let request_id_base = request.request_index.saturating_mul(1_000_000);
        let placement_version = format!("real-full-api-request-{}", request.request_index);
        let dspark_configuration = self
            .dspark
            .as_ref()
            .map(|runtime| {
                runtime
                    .lock()
                    .map(|runtime| (runtime.mode, runtime.cache_mode, runtime.context_tokens))
                    .map_err(|error| format!("locking dSpark request executor failed: {error}"))
            })
            .transpose()?;
        let dspark_mode = dspark_configuration.map(|configuration| configuration.0);
        let dspark_cache_mode = dspark_configuration.map(|configuration| configuration.1);
        let dspark_context_tokens = dspark_configuration.map(|configuration| configuration.2);
        let dspark_active = dspark_mode == Some(RealFullDsparkServingMode::Active)
            && !request.disable_speculation
            && request.decode_budget.saturating_sub(generated_tokens) > 1
            && (!real_full_internal_sequence(&sequence_id)
                || real_full_dspark_startup_draft_tokens(&sequence_id).is_some());
        let dspark_shadow = dspark_mode == Some(RealFullDsparkServingMode::Shadow)
            && request.greedy_sampling
            && !request.disable_speculation
            && request.decode_budget.saturating_sub(generated_tokens) > 1
            && !real_full_internal_sequence(&sequence_id);
        let dspark_participating = dspark_active || dspark_shadow;
        let radix_binding = target_radix_reservation.take().map(|reservation| {
            (
                reservation,
                prompt_token_ids
                    .as_deref()
                    .expect("target KV radix request retained prompt token IDs"),
            )
        });
        let mut persistent_state = Some(self.take_scheduler_state(&request, radix_binding)?);
        let snapshot_restore_ms = persistent_state
            .as_ref()
            .map_or(0.0, |state| state.snapshot_restore_ms);
        // A persistent scheduler state is permanently assigned to one
        // execution lane. Use that lane's production buffer bank for scalar
        // as well as batched execution so C=1 startup does not capture a
        // duplicate bank-0 graph set that live lane 0 can never reuse.
        let execution_buffer_bank = persistent_state
            .as_ref()
            .map_or(0, |state| state.execution_lane_id + 1);
        let _execution_buffer_bank_scope =
            coordinator_owned_device_buffer_bank_scope(execution_buffer_bank);
        let requested_startup_dspark_drafts =
            real_full_scalar_dspark_prewarm_requested_draft_tokens(
                &sequence_id,
                request.request_index,
            )
            .or_else(|| {
                real_full_batched_dspark_prewarm_requested_draft_tokens(
                    &sequence_id,
                    request.request_index,
                )
            })
            .or_else(|| real_full_dspark_startup_draft_tokens(&sequence_id));
        let mut pending_dspark_plan = if dspark_participating {
            self.dspark
                .as_ref()
                .expect("dSpark participation requires a request executor")
                .lock()
                .map_err(|error| format!("locking dSpark request executor failed: {error}"))?
                .prepare_cycle(
                    &sequence_id,
                    generated_tokens == 0,
                    (generated_tokens == 0)
                        .then(|| {
                            prompt_token_ids
                                .as_deref()
                                .map(|token_ids| (token_ids, request.cached_prompt_tokens))
                        })
                        .flatten(),
                    requested_startup_dspark_drafts,
                    self.dspark_device_storage.as_ref(),
                )
                .map_err(format_error_chain)?
        } else {
            None
        };
        if let Some(plan) = pending_dspark_plan.as_mut() {
            if requested_startup_dspark_drafts.is_none() {
                let max_useful_drafts = request
                    .decode_budget
                    .saturating_sub(generated_tokens)
                    .saturating_sub(1);
                plan.proposal_token_ids.truncate(max_useful_drafts);
            }
            if let Some(constraint) = persistent_state
                .as_ref()
                .and_then(|state| state.constraint.as_ref())
            {
                constrain_dspark_plan(plan, constraint).map_err(format_error_chain)?;
            }
            plan.selected_drafts = plan.proposal_token_ids.len();
            plan.target_batch_rows = plan.selected_drafts + 1;
        }
        if let Some(plan) = pending_dspark_plan.as_ref() {
            self.dspark
                .as_ref()
                .expect("issued dSpark plan requires a request runtime")
                .lock()
                .map_err(|error| format!("locking dSpark issued-plan tracker failed: {error}"))?
                .record_issued_plan(&sequence_id, plan);
        }
        let pending_dspark_draft_token_ids = pending_dspark_plan
            .as_ref()
            .map(|plan| plan.proposal_token_ids.clone())
            .unwrap_or_default();
        let synthetic_mtp_rows = if dspark_active {
            0
        } else {
            real_full_request_mtp_rows(&request, stateful_decode_step)
        };
        let mtp_rows = if dspark_active {
            pending_dspark_draft_token_ids.len()
        } else {
            synthetic_mtp_rows
        };
        let retain_target_hidden =
            // D=0 is a valid adaptive decision, but its scalar target row must
            // still seed the next proposal window. Otherwise one low-confidence
            // decision permanently disables drafting and the recovery probe
            // can never run.
            dspark_active
            || persistent_state
                .as_ref()
                .is_some_and(|state| state.constraint.is_some());
        let dspark_target_hidden_tap_rows =
            if dspark_participating && request.decode_budget.saturating_sub(generated_tokens) > 1 {
                if generated_tokens == 0
                    && dspark_cache_mode == Some(RealFullDsparkCacheMode::PromptSwa)
                {
                    token_rows
                        .prefill_tokens
                        .checked_add(decode_rows)
                        .and_then(|rows| rows.checked_add(mtp_rows))
                        .ok_or_else(|| "dSpark prompt-tail tap row count overflow".to_owned())?
                        .min(
                            dspark_context_tokens
                                .expect("dSpark participation has a resolved context window"),
                        )
                } else if generated_tokens > 0 {
                    decode_rows + mtp_rows
                } else {
                    // Request-local mode intentionally excludes the final prompt
                    // token. Generated token zero is captured on the next pass.
                    0
                }
            } else {
                0
            };
        let mut scheduler_token_ids = token_rows.decode_token_ids.clone();
        scheduler_token_ids.extend_from_slice(&pending_dspark_draft_token_ids);
        if request_timing {
            eprintln!(
                "real_full_request_timing request_id={} stage=tokenize elapsed_ms={:.3} prefill_tokens={} prefill_chunk_tokens={} decode_rows={} speculative_rows={} dspark_active={}",
                request_id,
                elapsed_ms(tokenize_start),
                token_rows.prefill_tokens,
                prefill_chunk_tokens,
                decode_rows,
                mtp_rows,
                dspark_active,
            );
        }

        let shape = RealFullSchedulerExecutionShape {
            request_id: request.request_id.clone(),
            sequence_id: sequence_id.clone(),
            placement_version: placement_version.clone(),
            prefix_tokens: token_rows.prefix_tokens,
            prefill_tokens: token_rows.prefill_tokens,
            prefill_chunk_tokens,
            decode_rows,
            mtp_rows,
            mtp_accepted_rows: if dspark_active {
                0
            } else {
                mtp_rows.min(REAL_FULL_REQUEST_MTP_ACCEPTED_ROWS)
            },
            prefill_token_ids: token_rows.prefill_token_ids,
            decode_token_ids: Some(scheduler_token_ids),
            lm_head_sampling: request_lm_head_sampling_options(&request),
        };
        let scheduler_start = Instant::now();
        let (mut report, sparse_tcp_dispatch, cycle_token_ids, mtp_terminal_sample) = {
            let dispatch_worker = &self.sparse_tcp_dispatch_worker;
            let mut state = persistent_state
                .take()
                .expect("shared sparse dispatch has persistent scheduler state");
            let constrained_sampling = state.constraint.is_some();
            let native_target = RealFullSchedulerNativeTargetContext::new(
                Arc::clone(&self.target_device_storage),
                self.target_device_identity,
                state.execution_lane_id,
                self.catalog.facts.rope_theta,
            );
            let execution =
                    real_full_scheduler_execution_for_shape_with_shared_sparse_tcp_and_state_device_hidden(
                        self.kv_config.clone(),
                        &self.catalog,
                        shape,
                        Arc::clone(dispatch_worker),
                        request_id_base,
                        &mut state,
                        native_target,
                        retain_target_hidden,
                        false,
                        constrained_sampling,
                        dspark_target_hidden_tap_rows,
                    );
            let execution = match execution {
                Ok(result) => result,
                Err(error) => {
                    if let Some(plan) = pending_dspark_plan.take() {
                        if let Some(dspark) = self.dspark.as_ref() {
                            if let Ok(mut dspark) = dspark.lock() {
                                dspark.restore_verification(&sequence_id, plan);
                            }
                        }
                    }
                    let _ = self.store_scheduler_state(&sequence_id, state);
                    return Err(format_error_chain(error));
                }
            };
            let target_submit_ms = elapsed_ms(scheduler_start);
            let mut report = execution.report;
            let probe = execution.sparse_tcp_dispatch;
            let mut target_hidden = execution.final_target_device_hidden;
            let dspark_target_hidden_taps = execution.target_device_hidden_taps;
            let mut cycle_token_ids = Vec::new();
            let mut mtp_terminal_sample = None;
            let mut dspark_cache_update = None;
            if state.constraint.is_some() && pending_dspark_draft_token_ids.is_empty() {
                let constrained_hidden = target_hidden
                    .as_ref()
                    .context("constrained scalar decode has no retained target hidden row")
                    .map_err(format_error_chain)?;
                let constrained_samples = real_full_constraint_target_samples(
                    &self.catalog,
                    &state,
                    &request,
                    constrained_hidden,
                    1,
                    &[],
                )
                .map_err(format_error_chain)?
                .expect("active constraint produces target samples");
                let sampled_token_id = if request.greedy_sampling {
                    constrained_samples.top_token_ids[0]
                } else {
                    constrained_samples.sampled_token_ids[0]
                };
                apply_speculative_terminal_sample_to_report(
                    &mut report,
                    &RealFullSpeculativeTerminalSample {
                        hidden_dim: self.catalog.facts.hidden_size,
                        vocab_size: constrained_samples.vocab_size,
                        top_token_id: constrained_samples.top_token_ids[0],
                        sampled_token_id,
                        sample_top_k: if request.greedy_sampling {
                            1
                        } else {
                            constrained_samples.sample_top_k
                        },
                        sample_top_p: if request.greedy_sampling {
                            1.0
                        } else {
                            constrained_samples.sample_top_p
                        },
                        argmax_backend: constrained_samples.argmax_kernel_backend,
                        sampler_backend: if request.greedy_sampling {
                            constrained_samples.argmax_kernel_backend
                        } else {
                            constrained_samples.sampler_kernel_backend
                        },
                        accepted_draft_tokens: 0,
                        report_mtp_acceptance: false,
                    },
                );
            }
            if dspark_active {
                if pending_dspark_draft_token_ids.is_empty() {
                    let anchor_token = if request.greedy_sampling {
                        report.terminal_lm_head_sample.top_token_id
                    } else {
                        report.terminal_lm_head_sample.sampled_token_id
                    }
                    .context("real-full dSpark scalar step requires a target token")
                    .map_err(format_error_chain)?;
                    let committed_rows = if generated_tokens == 0
                        && dspark_cache_mode == Some(RealFullDsparkCacheMode::PromptSwa)
                    {
                        dspark_target_hidden_tap_rows
                    } else {
                        1
                    };
                    cycle_token_ids.push(anchor_token);
                    dspark_cache_update = Some((committed_rows, anchor_token));
                } else {
                    let target_hidden = match target_hidden.take() {
                        Some(hidden) => hidden,
                        None => {
                            if let Some(plan) = pending_dspark_plan.take() {
                                if let Some(dspark) = self.dspark.as_ref() {
                                    if let Ok(mut dspark) = dspark.lock() {
                                        dspark.restore_verification(&sequence_id, plan);
                                    }
                                }
                            }
                            let _ = self.store_scheduler_state(&sequence_id, state);
                            return Err(format!(
                                "real-full dSpark request {} has no target hidden batch for verification",
                                request.request_id
                            ));
                        }
                    };
                    let suffix_rows = decode_rows + pending_dspark_draft_token_ids.len();
                    let target_sampling_start = Instant::now();
                    let sampled_uniforms = (!request.greedy_sampling).then(|| {
                        request_sampling_uniforms(&request, request.decode_step_index, suffix_rows)
                    });
                    let target_samples = match real_full_constraint_target_samples(
                        &self.catalog,
                        &state,
                        &request,
                        &target_hidden,
                        suffix_rows,
                        &pending_dspark_draft_token_ids,
                    )
                    .map_err(format_error_chain)?
                    {
                        Some(samples) => samples,
                        None => if let Some(random_uniforms) = sampled_uniforms.as_deref() {
                            real_full_target_token_samples_with_options(
                                &self.catalog,
                                &target_hidden,
                                suffix_rows,
                                request_lm_head_sampling_options_at(
                                    &request,
                                    request.decode_step_index,
                                ),
                                random_uniforms,
                            )
                        } else {
                            real_full_target_token_samples(
                                &self.catalog,
                                &target_hidden,
                                suffix_rows,
                            )
                        }
                        .map_err(format_error_chain)?,
                    };
                    let target_token_ids = if request.greedy_sampling {
                        target_samples.top_token_ids.as_slice()
                    } else {
                        target_samples.sampled_token_ids.as_slice()
                    };
                    let acceptance = real_full_speculative_acceptance(
                        pending_dspark_draft_token_ids.as_slice(),
                        target_token_ids,
                        true,
                        request.decode_budget.saturating_sub(generated_tokens),
                    )
                    .map_err(format_error_chain)?;
                    let accepted_draft_tokens = acceptance.accepted_draft_tokens;
                    let terminal_target_index = acceptance.terminal_target_index;
                    cycle_token_ids.extend_from_slice(&target_token_ids[..=terminal_target_index]);
                    committed_input_token_ids.extend_from_slice(
                        &pending_dspark_draft_token_ids[..accepted_draft_tokens],
                    );
                    let tentative_token_start =
                        token_rows.prefix_tokens + token_rows.prefill_tokens + decode_rows;
                    state
                        .resolve_mtp_tentative_writes(
                            tentative_token_start,
                            pending_dspark_draft_token_ids.len(),
                            accepted_draft_tokens,
                        )
                        .map_err(format_error_chain)?;
                    report.committed_mtp_writes =
                        accepted_draft_tokens * self.catalog.facts.num_hidden_layers;
                    report.discarded_mtp_writes = (pending_dspark_draft_token_ids.len()
                        - accepted_draft_tokens)
                        * self.catalog.facts.num_hidden_layers;
                    // The public counters retain their historical MTP names,
                    // but describe target-verified dSpark proposal rows.
                    report.request_mtp_accepted_rows = accepted_draft_tokens;
                    mtp_terminal_sample = Some(RealFullSpeculativeTerminalSample {
                        hidden_dim: self.catalog.facts.hidden_size,
                        vocab_size: target_samples.vocab_size,
                        top_token_id: target_samples.top_token_ids[terminal_target_index],
                        sampled_token_id: target_token_ids[terminal_target_index],
                        sample_top_k: if request.greedy_sampling {
                            1
                        } else {
                            target_samples.sample_top_k
                        },
                        sample_top_p: if request.greedy_sampling {
                            1.0
                        } else {
                            target_samples.sample_top_p
                        },
                        argmax_backend: target_samples.argmax_kernel_backend,
                        sampler_backend: if request.greedy_sampling {
                            target_samples.argmax_kernel_backend
                        } else {
                            target_samples.sampler_kernel_backend
                        },
                        accepted_draft_tokens,
                        report_mtp_acceptance: true,
                    });
                    dspark_cache_update = Some((
                        accepted_draft_tokens + 1,
                        target_token_ids[terminal_target_index],
                    ));
                    let plan = pending_dspark_plan
                        .as_ref()
                        .expect("dSpark drafts came from a pending verification plan");
                    let (next_confidence_logit_bias, calibration_variance, calibration_cycles) =
                        self.dspark
                            .as_ref()
                            .expect("active dSpark verification has a request runtime")
                            .lock()
                            .map_err(|error| {
                                format!("locking dSpark confidence calibrator failed: {error}")
                            })?
                            .observe_verification(&sequence_id, plan, accepted_draft_tokens);
                    if real_full_dspark_trace_enabled() {
                        eprintln!(
                            "real_full_dspark_acceptance request_id={} sequence_id={} target_context={} drafts={} accepted={} emitted={} full_match={} target_rows={} expected_tokens={:.4} expected_tps={:.3} confidence_logit_bias={:.4} next_confidence_logit_bias={:.4} calibration_variance={:.6} calibration_cycles={} target_submit_ms={:.3} target_sampling_ms={:.3}",
                            request_id,
                            sequence_id,
                            token_rows.prefix_tokens + token_rows.prefill_tokens,
                            pending_dspark_draft_token_ids.len(),
                            accepted_draft_tokens,
                            cycle_token_ids.len(),
                            acceptance.full_match_bonus,
                            plan.target_batch_rows,
                            plan.expected_committed_tokens,
                            plan.expected_tokens_per_second,
                            plan.confidence_logit_bias,
                            next_confidence_logit_bias,
                            calibration_variance,
                            calibration_cycles,
                            target_submit_ms,
                            elapsed_ms(target_sampling_start),
                        );
                    }
                }
            }
            if dspark_active {
                // Public acceptance metrics describe logical drafts. Lower-level
                // scheduler/expert counters retain the physical padded work.
                report.request_mtp_verify_rows = pending_dspark_draft_token_ids.len();
            }
            let final_decode_step = if dspark_active {
                let emitted_tokens = if cycle_token_ids.is_empty() {
                    1
                } else {
                    cycle_token_ids.len()
                };
                generated_tokens + emitted_tokens >= request.decode_budget
            } else {
                final_decode_step
            };
            if let Some(taps) = dspark_target_hidden_taps {
                if !final_decode_step {
                    let (committed_rows, anchor_token) = if let Some(update) = dspark_cache_update {
                        update
                    } else {
                        let anchor_token = report
                            .terminal_lm_head_sample
                            .top_token_id
                            .context("real-full dSpark shadow step requires a target token")
                            .map_err(format_error_chain)?;
                        (
                            if generated_tokens == 0
                                && dspark_cache_mode == Some(RealFullDsparkCacheMode::PromptSwa)
                            {
                                taps.rows
                            } else {
                                1
                            },
                            anchor_token,
                        )
                    };
                    if taps.rows != dspark_target_hidden_tap_rows
                        || committed_rows > taps.rows
                        || taps.layer_ids != dspark_target_hidden_tap_layer_ids()
                    {
                        let _ = self.store_scheduler_state(&sequence_id, state);
                        return Err(format!(
                            "real-full dSpark expected {} physical target rows with {} committed at layers {:?}, got rows={} layers={:?}",
                            dspark_target_hidden_tap_rows,
                            committed_rows,
                            dspark_target_hidden_tap_layer_ids(),
                            taps.rows,
                            taps.layer_ids
                        ));
                    }
                    let tap_refs = std::array::from_fn(|index| &taps.values[index]);
                    let absolute_context_start = (generated_tokens == 0
                        && dspark_cache_mode == Some(RealFullDsparkCacheMode::PromptSwa))
                    .then(|| token_rows.prefix_tokens + taps.row_start);
                    let replay = match self.replay_dspark_step(
                        &sequence_id,
                        &request_id,
                        &placement_version,
                        request_id_base,
                        tap_refs,
                        0,
                        committed_rows,
                        absolute_context_start,
                        anchor_token,
                    ) {
                        Ok(replay) => replay,
                        Err(error) => {
                            let _ = self.store_scheduler_state(&sequence_id, state);
                            return Err(format_error_chain(error));
                        }
                    };
                    if generated_tokens == 0
                        && dspark_cache_mode == Some(RealFullDsparkCacheMode::PromptSwa)
                        && !real_full_internal_sequence(&sequence_id)
                    {
                        let prompt_ids = prompt_token_ids
                            .as_deref()
                            .expect("fresh external dSpark request retained prompt token IDs");
                        let reusable_prefix_tokens = prompt_ids.len().saturating_sub(1);
                        if token_rows.prefix_tokens < reusable_prefix_tokens {
                            let publish_result = self
                                .dspark
                                .as_ref()
                                .expect("replayed dSpark request has a runtime")
                                .lock()
                                .map_err(|error| {
                                    anyhow::anyhow!(
                                        "locking dSpark prefix-publish phase failed: {error}"
                                    )
                                })
                                .and_then(|mut runtime| {
                                    runtime.publish_reusable_prefix(
                                        &sequence_id,
                                        prompt_ids,
                                        reusable_prefix_tokens,
                                        self.dspark_device_storage.as_ref(),
                                    )
                                });
                            if let Err(error) = publish_result {
                                let _ = self.store_scheduler_state(&sequence_id, state);
                                return Err(format_error_chain(error));
                            }
                        }
                    }
                    if real_full_dspark_trace_enabled() {
                        eprintln!(
                            "real_full_dspark_step request_id={} sequence_id={} mode={:?} target_context={} draft_context_before={} committed_rows={} draft_context_after={} anchor_token={} selected_drafts={} target_batch_rows={} expected_tokens={:.4} expected_tps={:.3} update_ms={:.3} suffix_ms={:.3} readback_ms={:.3} dspark_total_ms={:.3} selected_proposals={:?} proposals={:?} confidence={:?}",
                            request_id,
                            sequence_id,
                            replay.mode,
                            token_rows.prefix_tokens + token_rows.prefill_tokens,
                            replay.step.context_tokens,
                            replay.step.committed_rows,
                            replay.context_tokens,
                            replay.step.anchor_token,
                            replay.plan.selected_drafts,
                            replay.plan.target_batch_rows,
                            replay.plan.expected_committed_tokens,
                            replay.plan.expected_tokens_per_second,
                            replay.step.update_ms,
                            replay.step.suffix_ms,
                            replay.step.readback_ms,
                            replay.step.total_ms,
                            replay.plan.proposal_token_ids,
                            replay.step.proposal_token_ids,
                            replay.step.conditional_confidence,
                        );
                    }
                }
            }
            if let Some(constraint) = state.constraint.as_mut() {
                if cycle_token_ids.is_empty() {
                    let emitted_token = report
                        .terminal_lm_head_sample
                        .sampled_token_id
                        .context("constrained decode produced no emitted token")
                        .map_err(format_error_chain)?;
                    constraint
                        .commit(std::slice::from_ref(&emitted_token))
                        .map_err(format_error_chain)?;
                } else {
                    constraint
                        .commit(&cycle_token_ids)
                        .map_err(format_error_chain)?;
                }
            }
            state
                .record_processed_token_ids(committed_prefix_tokens, &committed_input_token_ids)
                .map_err(format_error_chain)?;
            self.finish_scheduler_state(&sequence_id, state, final_decode_step)?;
            (report, Some(probe), cycle_token_ids, mtp_terminal_sample)
        };
        if request.greedy_sampling {
            report.terminal_lm_head_sample.sampled_token_id =
                report.terminal_lm_head_sample.top_token_id;
            report.terminal_lm_head_sample.sample_top_k = Some(1);
            report.terminal_lm_head_sample.sample_top_p = Some(1.0);
            report.terminal_lm_head_sample.sampler_kernel_backend =
                report.terminal_lm_head_sample.argmax_kernel_backend;
        }
        if request_timing {
            eprintln!(
                "real_full_request_timing request_id={} stage=scheduler elapsed_ms={:.3} total_ms={:.3} status={} sparse_batches={} host_batches={} sample_status={}",
                request_id,
                elapsed_ms(scheduler_start),
                elapsed_ms(request_start),
                report.status,
                report.sparse_expert_batches,
                report.sparse_expert_host_batches,
                report.terminal_lm_head_sample.status
            );
        }
        if let Some(sample) = mtp_terminal_sample.as_ref() {
            apply_speculative_terminal_sample_to_report(&mut report, sample);
        }
        let info_start = Instant::now();
        let sampled_token_text =
            self.decode_sampled_token_text_cached(report.terminal_lm_head_sample.sampled_token_id);
        let mut info = real_full_info_from_request_execution(
            &self.base_info,
            &self.catalog.snapshot_path,
            &report,
            sampled_token_text,
        );
        info.request_kv_snapshot_restore_ms = snapshot_restore_ms;
        if let Some(probe) = sparse_tcp_dispatch.as_ref() {
            apply_sparse_tcp_dispatch_probe(&mut info, self.sparse_tcp_targets.len(), probe);
        }
        if let Some(sample) = mtp_terminal_sample.as_ref() {
            if sample.report_mtp_acceptance {
                info.request_mtp_accepted_rows = sample.accepted_draft_tokens;
            }
            info.scheduler_terminal_lm_head_top_token_id = Some(sample.top_token_id);
            info.scheduler_terminal_lm_head_sampled_token_id = Some(sample.sampled_token_id);
            info.scheduler_terminal_lm_head_sampled_text =
                self.decode_sampled_token_text_cached(Some(sample.sampled_token_id));
            info.scheduler_terminal_lm_head_sample_top_k = Some(sample.sample_top_k);
            info.scheduler_terminal_lm_head_sample_top_p = Some(sample.sample_top_p);
            info.scheduler_terminal_lm_head_argmax_backend = Some(sample.argmax_backend.to_owned());
            info.scheduler_terminal_lm_head_sampler_backend =
                Some(sample.sampler_backend.to_owned());
        }
        if request_timing {
            eprintln!(
                "real_full_request_timing request_id={} stage=info elapsed_ms={:.3} total_ms={:.3} status={} sampled_token_id={:?}",
                request_id,
                elapsed_ms(info_start),
                elapsed_ms(request_start),
                info.status,
                info.scheduler_terminal_lm_head_sampled_token_id
            );
        }
        let generated_tokens = cycle_token_ids
            .into_iter()
            .map(|token_id| ds4rt_api::RealFullGeneratedToken {
                token_id,
                text: self.decode_sampled_token_text_cached(Some(token_id)),
            })
            .collect();
        if dspark_active
            && pending_dspark_plan
                .as_ref()
                .is_some_and(|plan| plan.calibration_eligible)
            && info.request_coordinator_graph_captures == 0
        {
            let observation = self
                .dspark
                .as_ref()
                .expect("adaptive dSpark cost observation requires a runtime")
                .lock()
                .map_err(|error| format!("locking dSpark runtime cost model failed: {error}"))
                .and_then(|mut dspark| {
                    dspark
                        .observe_runtime_cost(
                            &[token_rows.prefix_tokens + token_rows.prefill_tokens],
                            decode_rows + pending_dspark_draft_token_ids.len(),
                            elapsed_ms(request_start),
                        )
                        .map_err(format_error_chain)
                });
            match observation {
                Ok(observation) if real_full_dspark_trace_enabled() => {
                    eprintln!(
                        "real_full_dspark_runtime_cost requests={} context_work_bucket={} max_context_bucket={} target_rows={} observed_ms={:.3} predicted_ms_before={:.3} exact_samples={}",
                        observation.request_count,
                        observation.context_work_bucket,
                        observation.max_context_bucket,
                        observation.target_rows,
                        observation.observed_ms,
                        observation.predicted_ms_before,
                        observation.exact_samples,
                    );
                }
                Ok(_) => {}
                Err(error) => {
                    eprintln!("real_full_dspark_runtime_cost_ignored error={error}");
                }
            }
        }
        Ok(ds4rt_api::RealFullDecodeCycle {
            info,
            generated_tokens,
        })
    }
}

fn real_full_sequence_capacity_tokens(
    prompt_tokens: usize,
    decode_budget: usize,
    max_context_tokens: usize,
) -> Result<usize> {
    let required_tokens = prompt_tokens
        .checked_add(decode_budget.max(1))
        .context("real-full sequence token capacity overflow")?;
    anyhow::ensure!(
        required_tokens <= max_context_tokens,
        "real-full sequence requires {required_tokens} context tokens but the global maximum is {max_context_tokens}"
    );
    let extension_headroom = prompt_tokens.min(REAL_FULL_SEQUENCE_EXTENSION_HEADROOM_TOKENS);
    Ok(required_tokens
        .saturating_add(extension_headroom)
        .min(max_context_tokens))
}

fn real_full_internal_sequence(sequence_id: &str) -> bool {
    sequence_id.starts_with("real-full-startup-")
}

fn real_full_startup_target_radix_publish_tokens(sequence_id: &str) -> Option<usize> {
    sequence_id
        .strip_prefix(REAL_FULL_STARTUP_TARGET_RADIX_PUBLISH_PREFIX)
        .or_else(|| sequence_id.strip_prefix(REAL_FULL_STARTUP_CANONICAL_PREFILL_CHUNK_PREFIX))
        .and_then(|suffix| suffix.split_once("-sequence-").map(|(tokens, _)| tokens))
        .and_then(|tokens| tokens.parse::<usize>().ok())
        .filter(|tokens| *tokens > 0)
}

fn real_full_startup_target_radix_evict_tokens(sequence_id: &str) -> Option<usize> {
    sequence_id
        .strip_prefix(REAL_FULL_STARTUP_TARGET_RADIX_EVICT_PREFIX)
        .and_then(|suffix| suffix.split_once("-worker-").map(|(tokens, _)| tokens))
        .and_then(|tokens| tokens.parse::<usize>().ok())
        .filter(|tokens| *tokens > 0)
}

fn real_full_startup_workspace_sizing_sequence(sequence_id: &str) -> bool {
    sequence_id.starts_with("real-full-startup-capture-arena-")
}

fn real_full_seal_owned_buffer_pool_sequence(sequence_id: &str) -> bool {
    sequence_id.starts_with(REAL_FULL_STARTUP_SEAL_OWNED_BUFFER_POOL_PREFIX)
}

fn real_full_prewarm_paired_lm_head_sequence(sequence_id: &str) -> bool {
    sequence_id.starts_with(REAL_FULL_STARTUP_PREWARM_PAIRED_LM_HEAD_PREFIX)
}

fn real_full_prewarm_batched_dspark_sequence(sequence_id: &str) -> bool {
    sequence_id.starts_with(REAL_FULL_STARTUP_PREWARM_BATCHED_DSPARK_PREFIX)
}

fn real_full_paired_lm_head_prewarm_range(fixed_drafts: Option<usize>) -> Option<(usize, usize)> {
    let max_single_rows = match fixed_drafts {
        Some(0) => return None,
        Some(fixed_drafts) => fixed_drafts + 1,
        None => REAL_FULL_DSPARK_MAX_VERIFY_DRAFTS + 1,
    };
    Some((max_single_rows + 1, 2 * max_single_rows))
}

fn real_full_dspark_startup_draft_tokens(sequence_id: &str) -> Option<usize> {
    sequence_id
        .strip_prefix("real-full-startup-dspark-width-")
        .and_then(|suffix| suffix.split('-').next())
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|draft_tokens| *draft_tokens <= REAL_FULL_DSPARK_MAX_VERIFY_DRAFTS)
}

fn real_full_batched_dspark_prewarm_buffer_bank(sequence_id: &str) -> Option<usize> {
    sequence_id
        .split_once(REAL_FULL_STARTUP_BATCHED_DSPARK_BANK_MARKER)
        .and_then(|(_, suffix)| suffix.split('-').next())
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|buffer_bank| *buffer_bank > 0)
}

fn real_full_batched_dspark_prewarm_sequence(sequence_id: &str) -> bool {
    real_full_dspark_startup_draft_tokens(sequence_id).is_some_and(|draft_tokens| draft_tokens > 0)
        && real_full_batched_dspark_prewarm_buffer_bank(sequence_id).is_some()
}

const REAL_FULL_BATCHED_DSPARK_PREWARM_WIDTH_REQUEST_BASE: u64 = 92_000;
const REAL_FULL_BATCHED_DSPARK_PREWARM_WIDTH_REQUEST_STRIDE: u64 = 100;
const REAL_FULL_SCALAR_DSPARK_PREWARM_WIDTH_REQUEST_BASE: u64 = 91_000;
const REAL_FULL_SCALAR_DSPARK_PREWARM_WIDTH_REQUEST_STRIDE: u64 = 100;
// Paired LM-head sampling has a different live-buffer lifetime than the
// per-lane recurrent scheduler. Keeping it in a disjoint pool namespace makes
// both sets of CUDA graph pointer identities stable without a final recapture
// pass. Production supports at most eight execution lanes, so this range does
// not overlap the ordinary lane banks (1..=8).
const REAL_FULL_PAIRED_LM_HEAD_BUFFER_BANK_BASE: usize = 16;

fn real_full_paired_lm_head_buffer_bank(execution_buffer_bank: usize) -> usize {
    REAL_FULL_PAIRED_LM_HEAD_BUFFER_BANK_BASE + execution_buffer_bank
}

fn real_full_scalar_dspark_prewarm_requested_draft_tokens(
    sequence_id: &str,
    request_index: u64,
) -> Option<usize> {
    sequence_id
        .contains(REAL_FULL_STARTUP_SCALAR_DSPARK_COHORT_MARKER)
        .then_some(())?;
    let encoded = request_index.checked_sub(REAL_FULL_SCALAR_DSPARK_PREWARM_WIDTH_REQUEST_BASE)?
        / REAL_FULL_SCALAR_DSPARK_PREWARM_WIDTH_REQUEST_STRIDE;
    usize::try_from(encoded)
        .ok()
        .filter(|draft_tokens| *draft_tokens <= REAL_FULL_DSPARK_MAX_VERIFY_DRAFTS)
}

fn real_full_batched_dspark_prewarm_requested_draft_tokens(
    sequence_id: &str,
    request_index: u64,
) -> Option<usize> {
    if !real_full_batched_dspark_prewarm_sequence(sequence_id) {
        return None;
    }
    let encoded = request_index.checked_sub(REAL_FULL_BATCHED_DSPARK_PREWARM_WIDTH_REQUEST_BASE)?
        / REAL_FULL_BATCHED_DSPARK_PREWARM_WIDTH_REQUEST_STRIDE;
    usize::try_from(encoded)
        .ok()
        .filter(|draft_tokens| *draft_tokens <= REAL_FULL_DSPARK_MAX_VERIFY_DRAFTS)
}

fn real_full_capture_arena_sequence(sequence_id: &str) -> bool {
    sequence_id.starts_with("real-full-startup-capture-arena-")
        || sequence_id.starts_with("real-full-startup-dsa-selector-seed-")
        || sequence_id.starts_with("real-full-startup-prefix-prefill-seed-")
        || sequence_id.starts_with("real-full-startup-dspark-width-")
}

impl ds4rt_api::RealFullRequestExecutor for RealFullSchedulerRequestExecutor {
    fn execute_real_full_request(
        &self,
        request: ds4rt_api::RealFullRequest,
    ) -> std::result::Result<ds4rt_api::RealFullInfo, String> {
        self.execute_real_full_decode_cycle_inner(request)
            .map(|cycle| cycle.info)
    }

    fn execute_real_full_decode_cycle(
        &self,
        request: ds4rt_api::RealFullRequest,
    ) -> std::result::Result<ds4rt_api::RealFullDecodeCycle, String> {
        self.execute_real_full_decode_cycle_inner(request)
    }

    fn real_full_decode_cycle_batch_coalesce_timeout(
        &self,
        request: &ds4rt_api::RealFullRequest,
    ) -> Option<Duration> {
        let dspark_active = self.dspark.as_ref().is_some_and(|runtime| {
            runtime
                .lock()
                .map(|runtime| runtime.mode == RealFullDsparkServingMode::Active)
                .unwrap_or(false)
        });
        (dspark_active
            && self.max_execution_lanes > 1
            && request.greedy_sampling
            && !request.disable_speculation
            && !real_full_internal_sequence(&request.sequence_id))
        .then(|| {
            // Near-simultaneous initial requests may enter one persistent prefill
            // wave. This is an idle-start admission quantum; recurrent work is
            // coordinator-owned and never depends on this arrival window.
            if request.generated_token_ids.is_empty() {
                Duration::from_millis(50)
            } else {
                Duration::from_micros(100)
            }
        })
    }

    fn real_full_decode_cycle_batch_max_size(
        &self,
        _request: &ds4rt_api::RealFullRequest,
    ) -> usize {
        self.max_execution_lanes
    }

    fn real_full_max_concurrent_sequences(&self) -> usize {
        REAL_FULL_MAX_ACTIVE_REQUESTS
    }

    fn real_full_retryable_admission_error(
        &self,
        _request: &ds4rt_api::RealFullRequest,
        error: &str,
    ) -> bool {
        error.contains("target KV guaranteed capacity exhausted")
            || error.contains("target KV active request limit exhausted")
            || error.contains("configured real-full execution lanes are resident")
            || error.contains("real-full global context budget exhausted")
    }

    fn execute_real_full_decode_cycle_batch(
        &self,
        requests: Vec<ds4rt_api::RealFullRequest>,
    ) -> Vec<std::result::Result<ds4rt_api::RealFullDecodeCycle, String>> {
        if !self.batched_dspark_cycles_eligible(&requests) {
            return requests
                .into_iter()
                .map(|request| self.execute_real_full_decode_cycle_inner(request))
                .collect();
        }

        let batch_start = Instant::now();
        let retry_requests = requests.clone();
        let mut prepared = Vec::with_capacity(requests.len());
        for (request_index, request) in requests.into_iter().enumerate() {
            match self.prepare_batched_dspark_cycle(request) {
                Ok(cycle) => prepared.push(cycle),
                Err(error) => {
                    for cycle in prepared.drain(..) {
                        self.restore_prepared_batched_dspark_cycle(cycle);
                    }
                    return retry_requests
                        .into_iter()
                        .enumerate()
                        .map(|(retry_index, request)| {
                            if retry_index == request_index {
                                Err(error.clone())
                            } else {
                                self.execute_real_full_decode_cycle_inner(request)
                            }
                        })
                        .collect();
                }
            }
        }
        if let Err(error) = self.replan_prepared_batched_dspark_cycles(&mut prepared) {
            let request_count = prepared.len();
            for cycle in prepared {
                self.restore_prepared_batched_dspark_cycle(cycle);
            }
            return (0..request_count).map(|_| Err(error.clone())).collect();
        }
        let runtime_cost_contexts = prepared
            .iter()
            .map(|cycle| cycle.token_prefix_tokens + cycle.token_prefill_tokens)
            .collect::<Vec<_>>();
        let runtime_cost_target_rows = prepared
            .iter()
            .map(|cycle| cycle.decode_rows + cycle.pending_dspark_draft_token_ids.len())
            .sum::<usize>();
        let runtime_cost_eligible = prepared.iter().all(|cycle| {
            cycle
                .pending_dspark_plan
                .as_ref()
                .is_some_and(|plan| plan.calibration_eligible)
        });

        let scheduler_inputs = prepared
            .iter_mut()
            .map(|cycle| {
                let execution_lane_id = cycle.state.execution_lane_id;
                let native_target = RealFullSchedulerNativeTargetContext::new(
                    Arc::clone(&self.target_device_storage),
                    self.target_device_identity,
                    execution_lane_id,
                    self.catalog.facts.rope_theta,
                );
                RealFullSchedulerBatchedInput {
                    shape: cycle.shape.clone(),
                    request_id_base: cycle.request_id_base,
                    state: &mut cycle.state,
                    buffer_bank: cycle.buffer_bank,
                    retain_final_target_device_hidden: cycle.pending_dspark_plan.is_some(),
                    target_device_hidden_tap_rows: cycle.dspark_target_hidden_tap_rows,
                    native_target,
                }
            })
            .collect();
        let execution =
            real_full_scheduler_execution_for_batched_shapes_with_shared_sparse_tcp_and_state_device_hidden(
            self.kv_config.clone(),
            &self.catalog,
            scheduler_inputs,
            Arc::clone(&self.sparse_tcp_dispatch_worker),
        );
        let executions = match execution {
            Ok(executions) => executions,
            Err(error) => {
                let error = format_error_chain(error);
                let request_count = prepared.len();
                for cycle in prepared {
                    self.restore_prepared_batched_dspark_cycle(cycle);
                }
                return (0..request_count).map(|_| Err(error.clone())).collect();
            }
        };

        let mut paired_target_samples = (0..prepared.len()).map(|_| None).collect::<Vec<_>>();
        for pair_start in (0..prepared.len()).step_by(2) {
            let pair_end = pair_start + 1;
            if pair_end >= prepared.len() {
                break;
            }
            let cycle_a = &prepared[pair_start];
            let cycle_b = &prepared[pair_end];
            let suffix_rows_a = cycle_a.decode_rows + cycle_a.pending_dspark_draft_token_ids.len();
            let suffix_rows_b = cycle_b.decode_rows + cycle_b.pending_dspark_draft_token_ids.len();
            if cycle_a.pending_dspark_draft_token_ids.is_empty()
                || cycle_b.pending_dspark_draft_token_ids.is_empty()
                || suffix_rows_a.saturating_add(suffix_rows_b) > 32
            {
                continue;
            }
            let samples = with_coordinator_owned_device_buffer_bank(
                real_full_paired_lm_head_buffer_bank(cycle_a.buffer_bank),
                || {
                    let hidden_a = executions[pair_start]
                        .final_target_device_hidden
                        .as_ref()
                        .with_context(|| {
                            format!(
                                "batched dSpark request {pair_start} has no retained target hidden rows"
                            )
                        })?;
                    let hidden_b = executions[pair_end]
                        .final_target_device_hidden
                        .as_ref()
                        .with_context(|| {
                            format!(
                            "batched dSpark request {pair_end} has no retained target hidden rows"
                        )
                        })?;
                    real_full_target_token_samples_pair(
                        &self.catalog,
                        hidden_a,
                        suffix_rows_a,
                        hidden_b,
                        suffix_rows_b,
                    )
                },
            );
            match samples {
                Ok((sample_a, sample_b)) => {
                    paired_target_samples[pair_start] = Some(sample_a);
                    paired_target_samples[pair_end] = Some(sample_b);
                }
                Err(error) => {
                    let error = format_error_chain(error);
                    let request_count = prepared.len();
                    for cycle in prepared {
                        self.restore_prepared_batched_dspark_cycle(cycle);
                    }
                    return (0..request_count).map(|_| Err(error.clone())).collect();
                }
            }
        }

        let request_count = prepared.len();
        let finish_results = prepared
            .into_iter()
            .zip(executions)
            .zip(paired_target_samples)
            .map(|((cycle, execution), target_samples)| {
                with_coordinator_owned_device_buffer_bank(cycle.buffer_bank, || {
                    self.prepare_batched_dspark_finish(cycle, execution, target_samples)
                })
            })
            .collect::<Vec<_>>();
        let preparation_error = finish_results
            .iter()
            .find_map(|result| result.as_ref().err().cloned());
        let mut finishes = finish_results
            .into_iter()
            .filter_map(std::result::Result::ok)
            .collect::<Vec<_>>();
        if let Some(mut error) = preparation_error {
            if let Err(abort_error) = self.abort_batched_dspark_replays(&finishes) {
                error.push_str(&format!(
                    "; aborting prepared joint replays failed: {abort_error:#}"
                ));
            }
            for finish in finishes {
                self.restore_prepared_batched_dspark_cycle(finish.prepared);
            }
            return (0..request_count).map(|_| Err(error.clone())).collect();
        }
        debug_assert_eq!(finishes.len(), request_count);
        let replay_indices = finishes
            .iter()
            .enumerate()
            .filter_map(|(index, finish)| finish.replay_prepared.as_ref().map(|_| index))
            .collect::<Vec<_>>();
        if !replay_indices.is_empty() {
            let replay_outputs = {
                let launches = replay_indices
                    .iter()
                    .map(|index| {
                        let finish = &finishes[*index];
                        let taps = finish
                            .target_hidden_taps
                            .as_ref()
                            .expect("prepared joint replay retained target hidden taps");
                        RealFullDsparkReplayLaunch {
                            prepared: finish
                                .replay_prepared
                                .as_ref()
                                .expect("joint replay index retained a prepared replay"),
                            sequence_id: &finish.prepared.sequence_id,
                            request_id: &finish.prepared.request.request_id,
                            placement_version: &finish.prepared.shape.placement_version,
                            request_id_base: finish.prepared.request_id_base,
                            target_hidden_taps: std::array::from_fn(|tap| &taps.values[tap]),
                        }
                    })
                    .collect::<Vec<_>>();
                self.execute_prepared_dspark_replay_batch(&launches)
            };
            let replay_outputs = match replay_outputs {
                Ok(outputs) => outputs,
                Err(replay_error) => {
                    let mut error = format_error_chain(replay_error);
                    if let Err(abort_error) = self.abort_batched_dspark_replays(&finishes) {
                        error.push_str(&format!(
                            "; aborting failed joint replays also failed: {abort_error:#}"
                        ));
                    }
                    for finish in finishes {
                        self.restore_prepared_batched_dspark_cycle(finish.prepared);
                    }
                    return (0..request_count).map(|_| Err(error.clone())).collect();
                }
            };
            if let Err(commit_error) =
                self.commit_batched_dspark_replays(&mut finishes, &replay_indices, replay_outputs)
            {
                let mut error = format_error_chain(commit_error);
                if let Err(abort_error) = self.abort_batched_dspark_replays(&finishes) {
                    error.push_str(&format!(
                        "; cleaning up rejected joint replays also failed: {abort_error:#}"
                    ));
                }
                for finish in finishes {
                    self.restore_prepared_batched_dspark_cycle(finish.prepared);
                }
                return (0..request_count).map(|_| Err(error.clone())).collect();
            }
        }
        let mut results = finishes
            .into_iter()
            .map(|finish| self.complete_batched_dspark_finish(finish))
            .collect::<Vec<_>>();
        if let Some((replay_width, cohort_width, cohort_count, is_2x2_wavefront)) =
            batched_dspark_replay_shape(replay_indices.len())
        {
            for replay_index in &replay_indices {
                if let Ok(cycle) = &mut results[*replay_index] {
                    cycle.info.request_dspark_joint_batch_width = replay_width;
                    cycle.info.request_dspark_joint_cohort_width = cohort_width;
                    cycle.info.request_dspark_joint_cohort_count = cohort_count;
                    cycle.info.request_dspark_2x2_wavefront = is_2x2_wavefront;
                }
            }
        }
        let runtime_cost_clean = results.iter().all(|result| {
            result
                .as_ref()
                .is_ok_and(|cycle| cycle.info.request_coordinator_graph_captures == 0)
        });
        if runtime_cost_eligible && runtime_cost_clean {
            if let Some(dspark) = self.dspark.as_ref() {
                match dspark
                    .lock()
                    .map_err(|error| format!("locking dSpark runtime cost model failed: {error}"))
                    .and_then(|mut dspark| {
                        dspark
                            .observe_runtime_cost(
                                &runtime_cost_contexts,
                                runtime_cost_target_rows,
                                elapsed_ms(batch_start),
                            )
                            .map_err(format_error_chain)
                    }) {
                    Ok(observation) if real_full_dspark_trace_enabled() => {
                        eprintln!(
                            "real_full_dspark_runtime_cost requests={} context_work_bucket={} max_context_bucket={} target_rows={} observed_ms={:.3} predicted_ms_before={:.3} exact_samples={}",
                            observation.request_count,
                            observation.context_work_bucket,
                            observation.max_context_bucket,
                            observation.target_rows,
                            observation.observed_ms,
                            observation.predicted_ms_before,
                            observation.exact_samples,
                        );
                    }
                    Ok(_) => {}
                    Err(error) => {
                        eprintln!("real_full_dspark_runtime_cost_ignored error={error}");
                    }
                }
            }
        }
        results
    }

    fn prewarm_batched_dspark_graphs(&self) -> std::result::Result<(), String> {
        let profile_at_startup = real_full_dspark_profile_at_startup_enabled();
        let checkpoint_max_drafts = dspark_active_max_verify_drafts();
        let max_draft_tokens = match real_full_dspark_fixed_drafts().map_err(format_error_chain)? {
            Some(draft_tokens) => {
                if draft_tokens > checkpoint_max_drafts {
                    return Err(format!(
                            "fixed dSpark width {draft_tokens} exceeds the active checkpoint maximum {checkpoint_max_drafts}"
                        ));
                }
                draft_tokens
            }
            None => checkpoint_max_drafts,
        };
        // Serial width capture explicitly finishes each sequence. Defensively
        // drain any other internal prewarm states before materializing the
        // production lanes; a retained target-radix reservation would
        // silently reduce the serving admission pool.
        let stale_internal_sequences = self
            .scheduler_states
            .lock()
            .map_err(|error| format!("locking startup scheduler states failed: {error}"))?
            .keys()
            .filter(|sequence_id| real_full_internal_sequence(sequence_id))
            .cloned()
            .collect::<Vec<_>>();
        for stale_sequence in &stale_internal_sequences {
            self.finish_real_full_sequence(stale_sequence)?;
        }
        if !stale_internal_sequences.is_empty() {
            eprintln!(
                "real_full_startup_internal_scheduler_release released_sequences={}",
                stale_internal_sequences.len()
            );
        }
        if self.max_execution_lanes <= 1 {
            return Ok(());
        }
        if self.max_execution_lanes > 8 {
            return Err(format!(
                "batched dSpark startup prewarm supports at most 8 execution lanes, got {}",
                self.max_execution_lanes
            ));
        }
        let Some(dspark) = self.dspark.as_ref() else {
            return Ok(());
        };
        {
            let mut dspark = dspark
                .lock()
                .map_err(|error| format!("locking dSpark startup prewarm failed: {error}"))?;
            let released = dspark.release_internal_sequences();
            if released > 0 {
                eprintln!(
                    "real_full_startup_internal_dspark_release released_sequences={released}"
                );
            }
        }

        let prompts = [
            "Implement a Rust function that merges overlapping integer intervals and explain its complexity.",
            "A product costs $240, receives a 25% discount, then 8% sales tax. Calculate the final price carefully.",
            "請用四個簡短條列解釋寫入時複製，並舉一個 fork 後修改記憶體頁面的例子。",
            "Write a short fable about two parrots sharing a mango tree, with a clear moral.",
        ]
        .into_iter()
        .take(self.max_execution_lanes)
        .map(str::to_owned)
        .collect::<Vec<_>>();
        let prompt_tokens = {
            let tokenizer = self
                .tokenizer
                .lock()
                .map_err(|error| format!("locking startup prewarm tokenizer failed: {error}"))?;
            prompts
                .iter()
                .map(|prompt| {
                    tokenizer
                        .encode_text(prompt, false)
                        .map_err(format_error_chain)
                        .map(|encoded| encoded.token_count)
                })
                .collect::<std::result::Result<Vec<_>, _>>()?
        };
        let decode_budget = if profile_at_startup { 4_096 } else { 1_024 };
        let sequence_ids = (1..=self.max_execution_lanes)
            .map(|buffer_bank| {
                format!(
                    "real-full-startup-dspark-width-{max_draft_tokens}-batched-bank-{buffer_bank}-sequence"
                )
            })
            .collect::<Vec<_>>();
        let start = Instant::now();
        eprintln!(
            "real_full_startup_prewarm_start stage=batched-dspark-widths lanes={} prompt_tokens={} max_drafts={} max_physical_m={} profile_at_startup={}",
            self.max_execution_lanes,
            prompt_tokens.iter().sum::<usize>(),
            max_draft_tokens,
            max_draft_tokens + 1,
            profile_at_startup,
        );

        let prewarm_result = (|| {
            let seed_start = Instant::now();
            let seed_requests = sequence_ids
                .iter()
                .zip(&prompts)
                .zip(&prompt_tokens)
                .enumerate()
                .map(|(lane_index, ((sequence_id, prompt), prompt_tokens))| {
                    ds4rt_api::RealFullRequest::new_decode_step_for_sequence(
                        90_000 + lane_index as u64,
                        sequence_id,
                        prompt,
                        *prompt_tokens,
                        1,
                        Vec::new(),
                        0,
                        decode_budget,
                    )
                })
                .collect::<Vec<_>>();
            let initial_cycles = self
                .execute_real_full_decode_cycle_batch(seed_requests)
                .into_iter()
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let reported_capture_delta = initial_cycles
                .iter()
                .map(|cycle| cycle.info.request_coordinator_graph_captures)
                .max()
                .unwrap_or(0);
            let mut generated_token_ids = Vec::with_capacity(sequence_ids.len());
            for (lane_index, initial_cycle) in initial_cycles.into_iter().enumerate() {
                if initial_cycle.info.status != "ready" {
                    return Err(format!(
                        "batched dSpark startup seed for lane {lane_index} failed: status={} blocker={} failed={:?}",
                        initial_cycle.info.status,
                        initial_cycle.info.blocker,
                        initial_cycle.info.failed_requirements,
                    ));
                }
                let token_id = initial_cycle
                    .generated_tokens
                    .first()
                    .map(|token| token.token_id)
                    .or(initial_cycle
                        .info
                        .scheduler_terminal_lm_head_sampled_token_id)
                    .ok_or_else(|| {
                        format!(
                            "batched dSpark startup seed for lane {lane_index} produced no token"
                        )
                    })?;
                generated_token_ids.push(vec![token_id]);
            }
            eprintln!(
                "real_full_startup_prewarm_step_done stage=batched-dspark-seed lanes={} reported_capture_delta={} elapsed_ms={:.3} total_ms={:.3}",
                self.max_execution_lanes,
                reported_capture_delta,
                elapsed_ms(seed_start),
                elapsed_ms(start),
            );

            // Each width uses the same live request/lane state, avoiding 4*C
            // redundant seed prefills. Layer-parity fused-hidden pools keep
            // each recurrent graph input address stable, so one pass captures
            // the complete working set for every physical M=1..max.
            for (width_index, draft_tokens) in (0..=max_draft_tokens).rev().enumerate() {
                let requests = sequence_ids
                    .iter()
                    .zip(&prompts)
                    .zip(&prompt_tokens)
                    .zip(generated_token_ids.iter())
                    .enumerate()
                    .map(
                        |(
                            lane_index,
                            (((sequence_id, prompt), prompt_tokens), generated_tokens),
                        )| {
                            ds4rt_api::RealFullRequest::new_decode_step_for_sequence(
                                REAL_FULL_BATCHED_DSPARK_PREWARM_WIDTH_REQUEST_BASE
                                    + (draft_tokens as u64)
                                        * REAL_FULL_BATCHED_DSPARK_PREWARM_WIDTH_REQUEST_STRIDE
                                    + lane_index as u64,
                                sequence_id,
                                prompt,
                                *prompt_tokens,
                                1,
                                generated_tokens.clone(),
                                1 + width_index,
                                decode_budget,
                            )
                        },
                    )
                    .collect::<Vec<_>>();
                let width_start = Instant::now();
                let cycles = self
                    .execute_real_full_decode_cycle_batch(requests)
                    .into_iter()
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                let reported_capture_delta = cycles
                    .iter()
                    .map(|cycle| cycle.info.request_coordinator_graph_captures)
                    .max()
                    .unwrap_or(0);
                for (lane_index, cycle) in cycles.iter().enumerate() {
                    if cycle.info.status != "ready"
                        || cycle.info.request_mtp_verify_rows != draft_tokens
                    {
                        return Err(format!(
                            "batched dSpark M={} startup lane {} failed: status={} verify_rows={} expected_rows={} blocker={} failed={:?}",
                            draft_tokens + 1,
                            lane_index,
                            cycle.info.status,
                            cycle.info.request_mtp_verify_rows,
                            draft_tokens,
                            cycle.info.blocker,
                            cycle.info.failed_requirements,
                        ));
                    }
                    if cycle.generated_tokens.is_empty() {
                        return Err(format!(
                            "batched dSpark M={} startup lane {} emitted no tokens",
                            draft_tokens + 1,
                            lane_index,
                        ));
                    }
                }
                for (generated_tokens, cycle) in
                    generated_token_ids.iter_mut().zip(cycles.into_iter())
                {
                    generated_tokens.extend(
                        cycle
                            .generated_tokens
                            .into_iter()
                            .map(|token| token.token_id),
                    );
                }
                eprintln!(
                    "real_full_startup_prewarm_step_done stage=batched-dspark-widths lanes={} physical_m={} reported_capture_delta={} elapsed_ms={:.3} total_ms={:.3}",
                    self.max_execution_lanes,
                    draft_tokens + 1,
                    reported_capture_delta,
                    elapsed_ms(width_start),
                    elapsed_ms(start),
                );
            }
            if profile_at_startup {
                self.profile_batched_dspark_sps(
                    &sequence_ids,
                    &prompts,
                    &prompt_tokens,
                    decode_budget,
                    &mut generated_token_ids,
                    max_draft_tokens,
                )?;
            }
            Ok(())
        })();

        let cleanup_errors = sequence_ids
            .iter()
            .filter_map(|sequence_id| {
                self.finish_real_full_sequence(sequence_id)
                    .err()
                    .map(|error| format!("{sequence_id}: {error}"))
            })
            .collect::<Vec<_>>();
        if let Err(error) = prewarm_result {
            if cleanup_errors.is_empty() {
                return Err(error);
            }
            return Err(format!(
                "{error}; startup cleanup also failed: {}",
                cleanup_errors.join("; ")
            ));
        }
        if !cleanup_errors.is_empty() {
            return Err(format!(
                "batched dSpark startup cleanup failed: {}",
                cleanup_errors.join("; ")
            ));
        }
        eprintln!(
            "real_full_startup_prewarm_done stage=batched-dspark-widths lanes={} max_physical_m={} elapsed_ms={:.3}",
            self.max_execution_lanes,
            max_draft_tokens + 1,
            elapsed_ms(start),
        );
        Ok(())
    }

    fn finish_real_full_sequence(&self, sequence_id: &str) -> std::result::Result<(), String> {
        if let Some(prefix_tokens) = real_full_startup_target_radix_evict_tokens(sequence_id) {
            let start = Instant::now();
            let prompt = REAL_FULL_SERVE_PREWARM_PROMPT_TOKEN.repeat(prefix_tokens);
            let encoding = self
                .tokenizer
                .lock()
                .map_err(|error| format!("locking startup radix tokenizer failed: {error}"))?
                .encode_text(&prompt, false)
                .map_err(format_error_chain)?;
            if encoding.token_count != prefix_tokens + 1 {
                return Err(format!(
                    "startup radix eviction prompt produced {} tokens for {prefix_tokens} reusable tokens",
                    encoding.token_count
                ));
            }
            let prefix_token_ids = encoding.token_ids[..prefix_tokens]
                .iter()
                .map(|token_id| *token_id as usize)
                .collect::<Vec<_>>();
            let eviction = self
                .target_kv_radix
                .evict_exact_inactive_subtree_if_present(&prefix_token_ids)
                .map_err(format_error_chain)?;
            let (status, evicted_pages, matched_tokens) = match eviction {
                TargetKvExactSubtreeEviction::Evicted { pages } => {
                    ("evicted", pages, prefix_tokens)
                }
                TargetKvExactSubtreeEviction::AlreadyAbsent { matched_tokens } => {
                    ("already-absent", 0, matched_tokens)
                }
            };
            let stats = self.target_kv_radix.stats();
            eprintln!(
                "real_full_startup_target_kv_radix_evict prefix_tokens={} status={} matched_tokens={} evicted_pages={} cached_pages={} free_pages={} radix_nodes={} elapsed_ms={:.3}",
                prefix_tokens,
                status,
                matched_tokens,
                evicted_pages,
                stats.cached_pages,
                stats.free_pages,
                stats.radix_nodes,
                elapsed_ms(start),
            );
            return Ok(());
        }
        if real_full_prewarm_batched_dspark_sequence(sequence_id) {
            return self.prewarm_batched_dspark_graphs();
        }
        if real_full_prewarm_paired_lm_head_sequence(sequence_id) {
            if let Some(dspark) = self.dspark.as_ref() {
                let released = dspark
                    .lock()
                    .map_err(|err| format!("locking dSpark request executor failed: {err}"))?
                    .release_internal_sequences();
                if released > 0 {
                    eprintln!(
                        "real_full_startup_internal_dspark_release released_sequences={released}"
                    );
                }
            }
            let max_sampler_rows = 2 * (REAL_FULL_DSPARK_MAX_VERIFY_DRAFTS + 1);
            for buffer_bank in 1..=self.max_execution_lanes {
                let start = Instant::now();
                eprintln!(
                    "real_full_startup_prewarm_start stage=lm-head-sampler-capacity buffer_bank={} max_rows={} top_k=64",
                    buffer_bank, max_sampler_rows,
                );
                with_coordinator_owned_device_buffer_bank(buffer_bank, || {
                    prewarm_real_full_target_sampler_capacity(&self.catalog, max_sampler_rows)
                })
                .map_err(format_error_chain)?;
                eprintln!(
                    "real_full_startup_prewarm_step_done stage=lm-head-sampler-capacity buffer_bank={} max_rows={} top_k=64 elapsed_ms={:.3}",
                    buffer_bank,
                    max_sampler_rows,
                    elapsed_ms(start),
                );
            }
            let Some((min_paired_rows, max_paired_rows)) = real_full_paired_lm_head_prewarm_range(
                real_full_dspark_fixed_drafts().map_err(format_error_chain)?,
            ) else {
                return Ok(());
            };
            for buffer_bank in (1..self.max_execution_lanes).step_by(2) {
                let start = Instant::now();
                eprintln!(
                    "real_full_startup_prewarm_start stage=paired-lm-head buffer_bank={} min_rows={} max_rows={}",
                    buffer_bank, min_paired_rows, max_paired_rows,
                );
                with_coordinator_owned_device_buffer_bank(
                    real_full_paired_lm_head_buffer_bank(buffer_bank),
                    || {
                        prewarm_real_full_paired_target_token_sample_rows(
                            &self.catalog,
                            min_paired_rows,
                            max_paired_rows,
                        )
                    },
                )
                .map_err(format_error_chain)?;
                eprintln!(
                    "real_full_startup_prewarm_step_done stage=paired-lm-head buffer_bank={} min_rows={} max_rows={} elapsed_ms={:.3}",
                    buffer_bank,
                    min_paired_rows,
                    max_paired_rows,
                    elapsed_ms(start),
                );
            }
            return Ok(());
        }
        if real_full_seal_owned_buffer_pool_sequence(sequence_id) {
            return seal_coordinator_owned_device_buffer_pool().map_err(format_error_chain);
        }
        let state = {
            let mut states = self
                .scheduler_states
                .lock()
                .map_err(|err| format!("locking real-full scheduler state map failed: {err}"))?;
            if let Some(state) = states.get(sequence_id) {
                if !state.owned_by_current_thread() {
                    return Err(format!(
                        "real-full sequence {sequence_id} must be finished on its graph-owner thread"
                    ));
                }
            }
            states.remove(sequence_id)
        };
        let finished_token_ids = state
            .as_ref()
            .filter(|_| !real_full_internal_sequence(sequence_id))
            .map(|state| state.processed_token_ids().to_vec());
        let dspark_finish_result = if let Some(dspark) = self.dspark.as_ref() {
            dspark
                .lock()
                .map_err(|err| format!("locking dSpark request executor failed: {err}"))?
                .finish_sequence(
                    sequence_id,
                    finished_token_ids.as_deref(),
                    self.dspark_device_storage.as_ref(),
                )
                .map_err(format_error_chain)
        } else {
            Ok(None)
        };
        let dspark_retained_prefix_tokens = dspark_finish_result.as_ref().ok().copied().flatten();
        let mut state_finish_result = Ok(());
        if let Some(mut state) = state {
            let committed_token_ids = state.processed_token_ids().to_vec();
            let snapshot_result = if state.snapshot_save_ready {
                if !self.kv_snapshot_saves.is_empty() {
                    if self
                        .kv_snapshot_saved
                        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                        .is_ok()
                    {
                        let committed_tokens = state.processed_token_ids().len();
                        let result = (|| {
                            for save in &self.kv_snapshot_saves {
                                let token_count = save.token_count.unwrap_or(committed_tokens);
                                anyhow::ensure!(
                                    token_count <= committed_tokens,
                                    "KV snapshot save point {token_count} exceeds committed sequence frontier {committed_tokens}"
                                );
                            }
                            for save in &self.kv_snapshot_saves {
                                let token_count = save.token_count.unwrap_or(committed_tokens);
                                let save_start = Instant::now();
                                save_real_full_kv_snapshot(
                                    &mut state.state,
                                    &save.root,
                                    &self.catalog,
                                    &self.engine_commit,
                                    token_count,
                                )?;
                                eprintln!(
                                    "real_full_kv_snapshot_save path={} tokens={} elapsed_ms={:.3}",
                                    save.root.display(),
                                    token_count,
                                    elapsed_ms(save_start),
                                );
                            }
                            Ok(())
                        })();
                        result.map_err(format_error_chain)
                    } else {
                        Ok(())
                    }
                } else {
                    Ok(())
                }
            } else {
                Ok(())
            };
            let radix_publish_result = if let Some(reservation) =
                state.target_radix_reservation.take()
            {
                // A reusable draft tail can end a handful of tokens behind
                // the final target pass because no next speculation cycle is
                // needed. Publish only the common exact frontier so a later
                // radix hit can restore the tail and replay that tiny suffix.
                let committed_tokens = real_full_startup_target_radix_publish_tokens(sequence_id)
                    .or(dspark_retained_prefix_tokens)
                    .unwrap_or(committed_token_ids.len());
                reservation
                        .commit_prefix(&committed_token_ids, committed_tokens)
                        .map(|published| {
                            let stats = self.target_kv_radix.stats();
                            eprintln!(
                                "real_full_target_kv_radix_publish sequence_id={} committed_tokens={} matched_existing_tokens={} published_pages={} duplicate_pages_freed={} cached_pages={} free_pages={} radix_nodes={} evicted_nodes={} evicted_pages={}",
                                sequence_id,
                                committed_tokens,
                                published.matched_existing_tokens,
                                published.published_pages,
                                published.duplicate_pages_freed,
                                stats.cached_pages,
                                stats.free_pages,
                                stats.radix_nodes,
                                stats.evicted_nodes,
                                stats.evicted_pages,
                            );
                        })
                        .map_err(format_error_chain)
            } else {
                Ok(())
            };
            let recycle_result = self.recycle_scheduler_state(state);
            state_finish_result = snapshot_result
                .and(radix_publish_result)
                .and(recycle_result);
        }
        state_finish_result?;
        dspark_finish_result?;
        if !real_full_internal_sequence(sequence_id) {
            clear_transient_coordinator_owned_device_buffers().map_err(format_error_chain)?;
        }
        Ok(())
    }
}

impl RealFullSchedulerRequestExecutor {
    fn decode_sampled_token_text_cached(&self, token_id: Option<usize>) -> Option<String> {
        let token_id = token_id?;
        if let Ok(cache) = self.sampled_token_text_cache.lock() {
            if let Some(text) = cache.get(&token_id) {
                return Some(text.clone());
            }
        }

        let tokenizer = self.tokenizer.lock().ok()?;
        let text = decode_sampled_token_text_with_tokenizer(&tokenizer, Some(token_id))?;
        if let Ok(mut cache) = self.sampled_token_text_cache.lock() {
            cache.insert(token_id, text.clone());
        }
        Some(text)
    }
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn real_full_request_timing_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env::var(REAL_FULL_REQUEST_TIMING_ENV)
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false)
    })
}

fn real_full_speculative_acceptance(
    draft_token_ids: &[usize],
    target_sampled_token_ids: &[usize],
    full_match_bonus_enabled: bool,
    max_emitted_tokens: usize,
) -> Result<RealFullSpeculativeAcceptance> {
    anyhow::ensure!(
        target_sampled_token_ids.len() == draft_token_ids.len() + 1,
        "real-full speculative target sample count {} must equal draft count {} plus one fallback",
        target_sampled_token_ids.len(),
        draft_token_ids.len()
    );
    let matching_prefix = draft_token_ids
        .iter()
        .zip(target_sampled_token_ids)
        .take_while(|(draft, target)| draft == target)
        .count();
    anyhow::ensure!(
        max_emitted_tokens > 0,
        "real-full speculative acceptance requires a positive emission budget"
    );
    let full_match_bonus = full_match_bonus_enabled
        && matching_prefix == draft_token_ids.len()
        && draft_token_ids.len().saturating_add(1) <= max_emitted_tokens;
    if full_match_bonus {
        return Ok(RealFullSpeculativeAcceptance {
            accepted_draft_tokens: matching_prefix,
            terminal_target_index: matching_prefix,
            full_match_bonus: true,
        });
    }
    // The opt-out path retains the final matching draft as the fallback so the
    // next proposal chain begins after a contiguous speculative cache prefix.
    // Full consumption is the default; this path remains only as a diagnostic.
    let accepted_draft_tokens = matching_prefix
        .min(draft_token_ids.len().saturating_sub(1))
        .min(max_emitted_tokens.saturating_sub(1));
    Ok(RealFullSpeculativeAcceptance {
        accepted_draft_tokens,
        terminal_target_index: accepted_draft_tokens,
        full_match_bonus: false,
    })
}

fn real_full_request_thread_pinned_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env::var(REAL_FULL_REQUEST_THREAD_PINNED_ENV)
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or_else(|_| crate::python_graph_capture::coordinator_python_capture_enabled())
    })
}

fn real_full_request_thread_pinned_workers() -> Result<usize> {
    let worker_count = match env::var(REAL_FULL_REQUEST_THREAD_PINNED_WORKERS_ENV) {
        Ok(value) => value.parse::<usize>().with_context(|| {
            format!("{REAL_FULL_REQUEST_THREAD_PINNED_WORKERS_ENV} must be a positive integer")
        })?,
        Err(_) => 1,
    };
    anyhow::ensure!(
        worker_count > 0,
        "{REAL_FULL_REQUEST_THREAD_PINNED_WORKERS_ENV} must be a positive integer"
    );
    Ok(worker_count)
}

fn optional_cpu_list_env(name: &str) -> Result<Vec<usize>> {
    let value = match env::var(name) {
        Ok(value) => value,
        Err(env::VarError::NotPresent) => return Ok(Vec::new()),
        Err(error) => return Err(error).with_context(|| format!("reading {name}")),
    };
    anyhow::ensure!(!value.trim().is_empty(), "{name} must not be empty");
    value
        .split(',')
        .enumerate()
        .map(|(index, value)| {
            let value = value.trim();
            anyhow::ensure!(!value.is_empty(), "{name} entry {index} must not be empty");
            value
                .parse::<usize>()
                .with_context(|| format!("{name} entry {index} must be a non-negative CPU index"))
        })
        .collect()
}

fn real_full_request_worker_cpus(worker_count: usize) -> Result<Vec<usize>> {
    let cpus = optional_cpu_list_env(REAL_FULL_REQUEST_WORKER_CPUS_ENV)?;
    anyhow::ensure!(
        cpus.is_empty() || cpus.len() == worker_count,
        "{REAL_FULL_REQUEST_WORKER_CPUS_ENV} has {} CPU assignments for {worker_count} request workers",
        cpus.len()
    );
    Ok(cpus)
}

fn real_full_scheduler_worker_cpu() -> Result<Option<usize>> {
    let cpus = optional_cpu_list_env(REAL_FULL_SCHEDULER_WORKER_CPU_ENV)?;
    anyhow::ensure!(
        cpus.len() <= 1,
        "{REAL_FULL_SCHEDULER_WORKER_CPU_ENV} must contain exactly one CPU index"
    );
    Ok(cpus.into_iter().next())
}

fn format_error_chain(error: anyhow::Error) -> String {
    format!("{error:#}")
}

fn optional_nonempty_env_path(name: &str) -> Result<Option<PathBuf>> {
    match env::var(name) {
        Ok(value) => {
            let value = value.trim();
            anyhow::ensure!(!value.is_empty(), "{name} must not be empty");
            Ok(Some(PathBuf::from(value)))
        }
        Err(env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(error).with_context(|| format!("reading {name}")),
    }
}

fn optional_positive_env_usize(name: &str) -> Result<Option<usize>> {
    match env::var(name) {
        Ok(value) => {
            let parsed = value
                .parse::<usize>()
                .with_context(|| format!("{name} must be a positive integer"))?;
            anyhow::ensure!(parsed > 0, "{name} must be a positive integer");
            Ok(Some(parsed))
        }
        Err(env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(error).with_context(|| format!("reading {name}")),
    }
}

fn default_real_full_kv_pool_tokens(max_context_tokens: usize) -> Result<usize> {
    anyhow::ensure!(
        max_context_tokens > 0,
        "real-full target KV pool needs a positive context bound"
    );
    let page_remainder = max_context_tokens % DS4_KV_SOURCE_PAGE_TOKENS;
    if page_remainder == 0 {
        return Ok(max_context_tokens);
    }
    max_context_tokens
        .checked_add(DS4_KV_SOURCE_PAGE_TOKENS - page_remainder)
        .context("real-full page-aligned target KV pool token count overflow")
}

fn real_full_kv_pool_tokens(max_context_tokens: usize) -> Result<usize> {
    let pool_tokens = match optional_positive_env_usize(REAL_FULL_KV_POOL_TOKENS_ENV)? {
        Some(pool_tokens) => pool_tokens,
        None => default_real_full_kv_pool_tokens(max_context_tokens)?,
    };
    anyhow::ensure!(
        pool_tokens >= max_context_tokens,
        "{REAL_FULL_KV_POOL_TOKENS_ENV}={pool_tokens} is smaller than max context {max_context_tokens}"
    );
    anyhow::ensure!(
        pool_tokens % REAL_FULL_SHARED_KV_PAGE_TOKENS == 0,
        "{REAL_FULL_KV_POOL_TOKENS_ENV}={pool_tokens} must be divisible by {REAL_FULL_SHARED_KV_PAGE_TOKENS}"
    );
    anyhow::ensure!(
        u32::try_from(pool_tokens).is_ok(),
        "{REAL_FULL_KV_POOL_TOKENS_ENV}={pool_tokens} exceeds the physical position format"
    );
    Ok(pool_tokens)
}

fn real_full_max_execution_lanes() -> Result<usize> {
    let lanes = optional_positive_env_usize(REAL_FULL_MAX_EXECUTION_LANES_ENV)?.unwrap_or(1);
    anyhow::ensure!(
        lanes <= REAL_FULL_DIAGNOSTIC_MAX_EXECUTION_LANES,
        "{REAL_FULL_MAX_EXECUTION_LANES_ENV}={lanes} exceeds the current diagnostic maximum {REAL_FULL_DIAGNOSTIC_MAX_EXECUTION_LANES}"
    );
    Ok(lanes)
}

fn real_full_kv_snapshot_save_points(name: &str) -> Result<Vec<RealFullKvSnapshotSave>> {
    let value = match env::var(name) {
        Ok(value) => value,
        Err(env::VarError::NotPresent) => return Ok(Vec::new()),
        Err(error) => return Err(error).with_context(|| format!("reading {name}")),
    };
    anyhow::ensure!(!value.trim().is_empty(), "{name} must not be empty");
    value
        .split(',')
        .enumerate()
        .map(|(index, entry)| {
            let entry = entry.trim();
            let (tokens, path) = entry
                .split_once('=')
                .with_context(|| format!("{name} entry {index} must use the form TOKENS=PATH"))?;
            let token_count = tokens.trim().parse::<usize>().with_context(|| {
                format!("{name} entry {index} token count must be a positive integer")
            })?;
            anyhow::ensure!(
                token_count > 0,
                "{name} entry {index} token count must be positive"
            );
            let path = path.trim();
            anyhow::ensure!(
                !path.is_empty(),
                "{name} entry {index} path must not be empty"
            );
            Ok(RealFullKvSnapshotSave {
                root: PathBuf::from(path),
                token_count: Some(token_count),
            })
        })
        .collect()
}

fn validate_real_full_kv_snapshot_saves(saves: &[RealFullKvSnapshotSave]) -> Result<()> {
    for (index, save) in saves.iter().enumerate() {
        for prior in &saves[..index] {
            anyhow::ensure!(
                save.root != prior.root,
                "duplicate KV snapshot save destination {}",
                save.root.display()
            );
            if let (Some(tokens), Some(prior_tokens)) = (save.token_count, prior.token_count) {
                anyhow::ensure!(
                    tokens != prior_tokens,
                    "duplicate KV snapshot save token cutoff {tokens}"
                );
            }
        }
    }
    Ok(())
}

fn real_full_request_prefill_chunk_tokens() -> usize {
    env::var("DS4RT_REAL_FULL_REQUEST_PREFILL_CHUNK_TOKENS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|tokens| *tokens > 0)
        .unwrap_or(DEFAULT_REAL_FULL_REQUEST_PREFILL_CHUNK_TOKENS)
}

fn real_full_request_large_prefill_min_tokens() -> usize {
    env::var(REAL_FULL_REQUEST_LARGE_PREFILL_MIN_TOKENS_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|tokens| *tokens > 0)
        .unwrap_or(DEFAULT_REAL_FULL_REQUEST_LARGE_PREFILL_MIN_TOKENS)
}

fn real_full_request_long_prefix_small_prefill_chunk_tokens() -> usize {
    env::var(REAL_FULL_REQUEST_LONG_PREFIX_SMALL_PREFILL_CHUNK_TOKENS_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|tokens| *tokens > 0)
        .unwrap_or(DEFAULT_REAL_FULL_REQUEST_LONG_PREFIX_SMALL_PREFILL_CHUNK_TOKENS)
}

fn balanced_prefill_chunk_tokens(prefill_tokens: usize, max_chunk_tokens: usize) -> usize {
    if prefill_tokens == 0 {
        return max_chunk_tokens;
    }
    let chunk_count = prefill_tokens.div_ceil(max_chunk_tokens).max(1);
    // A measured 530-row request benefits from replacing 512+18 with two
    // balanced waves. At three or more useful waves, preserving full-width
    // expert launches wins. The exception is a third prefill tail below 15
    // rows: even after the joined decode row it cannot use the incremental
    // Spark-reduction path and serializes another complete 75-layer wave.
    if chunk_count == 2 {
        prefill_tokens.div_ceil(chunk_count)
    } else if tiny_non_streaming_third_prefill_chunk(prefill_tokens, max_chunk_tokens) {
        prefill_tokens.div_ceil(2)
    } else {
        max_chunk_tokens
    }
}

fn tiny_non_streaming_third_prefill_chunk(prefill_tokens: usize, max_chunk_tokens: usize) -> bool {
    // This exception is qualified only for the production 512-row stride.
    // Applying it to a wider operator-selected cap would both exceed that cap
    // and substitute an unmeasured kernel geometry.
    if max_chunk_tokens != DEFAULT_REAL_FULL_REQUEST_FRESH_SMALL_PREFILL_CHUNK_TOKENS {
        return false;
    }
    let chunk_count = prefill_tokens.div_ceil(max_chunk_tokens);
    let tail_tokens = prefill_tokens % max_chunk_tokens;
    chunk_count == 3
        && tail_tokens > 0
        && tail_tokens < DEFAULT_REAL_FULL_REQUEST_MIN_STREAMING_TAIL_PREFILL_TOKENS
}

fn real_full_request_prefill_chunk_tokens_for_shape_with(
    configured_chunk_tokens: usize,
    large_prefill_min_tokens: usize,
    long_prefix_small_prefill_chunk_tokens: usize,
    prefix_tokens: usize,
    prefill_tokens: usize,
) -> usize {
    if prefill_tokens >= large_prefill_min_tokens {
        let minimum_chunks = prefill_tokens.div_ceil(configured_chunk_tokens).max(1);
        let balanced_chunks =
            minimum_chunks.max(prefill_tokens.min(REAL_FULL_PREFILL_PIPELINE_LANES).max(1));
        prefill_tokens.div_ceil(balanced_chunks)
    } else if prefix_tokens >= DEFAULT_REAL_FULL_REQUEST_LONG_PREFIX_MIN_TOKENS {
        let base_chunk_tokens = configured_chunk_tokens
            .min(long_prefix_small_prefill_chunk_tokens)
            .min(REAL_FULL_NATIVE_TARGET_PREFILL_MAX_QUERY_ROWS);
        // A tiny second chunk costs another complete 75-layer sparse wave at
        // long context. The measured one-wave advantage is robust through
        // 7/4 of the 512-row production stride and disappears near 1K.
        let tail_merge_ceiling = base_chunk_tokens
            .saturating_mul(DEFAULT_REAL_FULL_REQUEST_LONG_PREFIX_TAIL_MERGE_NUMERATOR)
            / DEFAULT_REAL_FULL_REQUEST_LONG_PREFIX_TAIL_MERGE_DENOMINATOR;
        let tail_merge_ceiling = tail_merge_ceiling.min(
            DEFAULT_REAL_FULL_REQUEST_LONG_PREFIX_SMALL_PREFILL_CHUNK_TOKENS
                * DEFAULT_REAL_FULL_REQUEST_LONG_PREFIX_TAIL_MERGE_NUMERATOR
                / DEFAULT_REAL_FULL_REQUEST_LONG_PREFIX_TAIL_MERGE_DENOMINATOR,
        );
        if prefill_tokens > base_chunk_tokens && prefill_tokens <= tail_merge_ceiling {
            prefill_tokens
        } else if tiny_non_streaming_third_prefill_chunk(prefill_tokens, base_chunk_tokens) {
            prefill_tokens.div_ceil(2)
        } else {
            base_chunk_tokens
        }
    } else if prefix_tokens == 0
        || prefill_tokens >= DEFAULT_REAL_FULL_REQUEST_CACHED_WIDE_SUFFIX_MIN_TOKENS
    {
        balanced_prefill_chunk_tokens(
            prefill_tokens,
            configured_chunk_tokens.min(DEFAULT_REAL_FULL_REQUEST_FRESH_SMALL_PREFILL_CHUNK_TOKENS),
        )
    } else {
        balanced_prefill_chunk_tokens(
            prefill_tokens,
            configured_chunk_tokens.min(DEFAULT_REAL_FULL_REQUEST_SMALL_PREFILL_CHUNK_TOKENS),
        )
    }
}

fn real_full_request_prefill_chunk_tokens_for_shape(
    prefix_tokens: usize,
    prefill_tokens: usize,
) -> usize {
    real_full_request_prefill_chunk_tokens_for_shape_with(
        real_full_request_prefill_chunk_tokens(),
        real_full_request_large_prefill_min_tokens(),
        real_full_request_long_prefix_small_prefill_chunk_tokens(),
        prefix_tokens,
        prefill_tokens,
    )
}

fn real_full_request_prefill_chunk_tokens_for_sequence(
    sequence_id: &str,
    prefix_tokens: usize,
    prefill_tokens: usize,
) -> usize {
    if sequence_id.starts_with(REAL_FULL_STARTUP_MAX_PREFILL_CHUNK_PREFIX) {
        return prefill_tokens
            .min(real_full_request_prefill_chunk_tokens())
            .max(1);
    }
    if sequence_id.starts_with(REAL_FULL_STARTUP_CANONICAL_PREFILL_CHUNK_PREFIX) {
        return prefill_tokens
            .min(real_full_request_prefill_chunk_tokens())
            .min(REAL_FULL_STARTUP_CANONICAL_PREFILL_CHUNK_TOKENS)
            .max(1);
    }
    real_full_request_prefill_chunk_tokens_for_shape(prefix_tokens, prefill_tokens)
}

fn real_full_target_radix_reuse_supported(facts: &ModelFacts) -> bool {
    facts.compress_ratios.len() >= facts.num_hidden_layers
        && facts
            .compress_ratios
            .iter()
            .take(facts.num_hidden_layers)
            .all(|ratio| *ratio == 0)
}

fn real_full_prefill_chunk_tokens_for_native_target(planned_chunk_tokens: usize) -> usize {
    planned_chunk_tokens.min(REAL_FULL_NATIVE_TARGET_PREFILL_MAX_QUERY_ROWS)
}

fn real_full_request_token_rows(
    request: &ds4rt_api::RealFullRequest,
    prompt_token_ids: Option<Vec<usize>>,
) -> Result<RealFullRequestTokenRows> {
    if request.cached_prompt_tokens > 0 {
        anyhow::ensure!(
            request.generated_token_ids.is_empty(),
            "real-full cached-prefix request cannot also contain generated tokens"
        );
        let prompt_token_ids = prompt_token_ids
            .context("real-full cached-prefix request is missing prompt token ids")?;
        anyhow::ensure!(
            prompt_token_ids.len() == request.prompt_tokens,
            "real-full cached-prefix prompt token id count {} does not match prompt tokens {}",
            prompt_token_ids.len(),
            request.prompt_tokens
        );
        anyhow::ensure!(
            request.cached_prompt_tokens < prompt_token_ids.len(),
            "real-full cached-prefix token count {} leaves no uncached prompt tokens out of {}",
            request.cached_prompt_tokens,
            prompt_token_ids.len()
        );
        let uncached_token_ids = &prompt_token_ids[request.cached_prompt_tokens..];
        let decode_token_id = *uncached_token_ids
            .last()
            .context("real-full cached-prefix request has no uncached prompt tokens")?;
        let prefill_tokens = uncached_token_ids.len() - 1;
        return Ok(RealFullRequestTokenRows {
            prefix_tokens: request.cached_prompt_tokens,
            prefill_tokens,
            prefill_token_ids: (prefill_tokens > 0)
                .then(|| uncached_token_ids[..prefill_tokens].to_vec()),
            decode_token_ids: vec![decode_token_id],
        });
    }
    if let Some(decode_token_id) = request.generated_token_ids.last().copied() {
        let prefix_tokens = request
            .prompt_tokens
            .checked_add(request.generated_token_ids.len().saturating_sub(1))
            .context("real-full request recurrent decode prefix token count overflows usize")?;
        return Ok(RealFullRequestTokenRows {
            prefix_tokens,
            prefill_tokens: 0,
            prefill_token_ids: None,
            decode_token_ids: vec![decode_token_id],
        });
    }

    let prompt_token_ids =
        prompt_token_ids.context("real-full initial decode request is missing prompt token ids")?;
    anyhow::ensure!(
        !prompt_token_ids.is_empty(),
        "real-full initial decode request has no prompt tokens"
    );
    anyhow::ensure!(
        prompt_token_ids.len() == request.prompt_tokens,
        "real-full initial decode prompt token id count {} does not match prompt tokens {}",
        prompt_token_ids.len(),
        request.prompt_tokens
    );
    let decode_token_id = *prompt_token_ids
        .last()
        .context("real-full initial decode prompt token ids unexpectedly empty")?;
    let prefill_tokens = prompt_token_ids.len() - 1;
    let prefill_token_ids = if prefill_tokens == 0 {
        None
    } else {
        Some(prompt_token_ids[..prefill_tokens].to_vec())
    };

    Ok(RealFullRequestTokenRows {
        prefix_tokens: 0,
        prefill_tokens,
        prefill_token_ids,
        decode_token_ids: vec![decode_token_id],
    })
}

fn real_full_request_mtp_rows(
    request: &ds4rt_api::RealFullRequest,
    stateful_decode_step: bool,
) -> usize {
    real_full_request_mtp_rows_for_policy(
        request,
        stateful_decode_step,
        real_full_request_mtp_verify_enabled(),
    )
}

fn real_full_request_mtp_rows_for_policy(
    request: &ds4rt_api::RealFullRequest,
    stateful_decode_step: bool,
    mtp_verify_enabled: bool,
) -> usize {
    if stateful_decode_step
        || !mtp_verify_enabled
        || request.decode_budget <= 1
        || request.decode_budget > request.max_tokens
    {
        0
    } else {
        request
            .max_tokens
            .max(1)
            .min(REAL_FULL_REQUEST_MAX_MTP_VERIFY_ROWS)
    }
}

fn real_full_request_mtp_verify_enabled() -> bool {
    env::var(REAL_FULL_REQUEST_MTP_VERIFY_ENV)
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn real_full_dspark_enabled() -> bool {
    env::var(REAL_FULL_DSPARK_ENV)
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(true)
}

fn real_full_dspark_shadow_enabled() -> bool {
    env::var(REAL_FULL_DSPARK_SHADOW_ENV)
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn real_full_dspark_trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env::var(REAL_FULL_DSPARK_TRACE_ENV)
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false)
    })
}

fn parse_real_full_dspark_confidence_policy(
    value: Option<&str>,
) -> Result<RealFullDsparkConfidencePolicy> {
    match value.map(str::trim).filter(|value| !value.is_empty()) {
        None | Some("residual") => Ok(RealFullDsparkConfidencePolicy::Residual),
        Some("calibrated") => Ok(RealFullDsparkConfidencePolicy::Calibrated),
        Some("raw") => Ok(RealFullDsparkConfidencePolicy::Raw),
        Some(value) => {
            bail!(
                "{REAL_FULL_DSPARK_CONFIDENCE_POLICY_ENV} must be calibrated, raw, or residual, got {value}"
            )
        }
    }
}

fn real_full_dspark_confidence_policy() -> Result<RealFullDsparkConfidencePolicy> {
    parse_real_full_dspark_confidence_policy(
        env::var(REAL_FULL_DSPARK_CONFIDENCE_POLICY_ENV)
            .ok()
            .as_deref(),
    )
}

fn real_full_dspark_profile_at_startup_enabled() -> bool {
    env::var(REAL_FULL_DSPARK_PROFILE_AT_STARTUP_ENV)
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn real_full_dspark_fixed_drafts() -> Result<Option<usize>> {
    let Some(value) =
        env::var_os(REAL_FULL_DSPARK_FIXED_DRAFTS_ENV).filter(|value| !value.is_empty())
    else {
        return Ok(None);
    };
    let value = value
        .into_string()
        .map_err(|_| anyhow::anyhow!("{REAL_FULL_DSPARK_FIXED_DRAFTS_ENV} is not valid UTF-8"))?;
    let drafts = value
        .parse::<usize>()
        .with_context(|| format!("parsing {REAL_FULL_DSPARK_FIXED_DRAFTS_ENV}={value}"))?;
    anyhow::ensure!(
        drafts <= REAL_FULL_DSPARK_MAX_VERIFY_DRAFTS,
        "{REAL_FULL_DSPARK_FIXED_DRAFTS_ENV} must be in 0..={REAL_FULL_DSPARK_MAX_VERIFY_DRAFTS}, got {drafts}"
    );
    Ok(Some(drafts))
}

fn real_full_dspark_mode() -> Result<Option<RealFullDsparkServingMode>> {
    let active = real_full_dspark_enabled();
    let shadow = real_full_dspark_shadow_enabled();
    anyhow::ensure!(
        !(active && shadow),
        "{REAL_FULL_DSPARK_ENV} and {REAL_FULL_DSPARK_SHADOW_ENV} cannot both be enabled"
    );
    if !active && !shadow {
        return Ok(None);
    }
    let mode = if active {
        RealFullDsparkServingMode::Active
    } else {
        RealFullDsparkServingMode::Shadow
    };
    Ok(Some(mode))
}

fn real_full_dspark_tail_cache_bytes(kv_bytes_per_request: usize) -> Result<usize> {
    let default_bytes = kv_bytes_per_request
        .checked_mul(REAL_FULL_MAX_ACTIVE_REQUESTS)
        .context("dSpark tail-cache default byte count overflow")?;
    match env::var(REAL_FULL_DSPARK_TAIL_CACHE_BYTES_ENV) {
        Ok(value) => value.parse::<usize>().with_context(|| {
            format!("{REAL_FULL_DSPARK_TAIL_CACHE_BYTES_ENV}={value} must be a byte count")
        }),
        Err(env::VarError::NotPresent) => Ok(default_bytes),
        Err(error) => {
            Err(error).with_context(|| format!("reading {REAL_FULL_DSPARK_TAIL_CACHE_BYTES_ENV}"))
        }
    }
}

fn real_full_dspark_cache_mode() -> Result<RealFullDsparkCacheMode> {
    match env::var(REAL_FULL_DSPARK_CACHE_MODE_ENV)
        .unwrap_or_else(|_| "request-local".to_owned())
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "request-local" | "request_local" | "mode1" | "1" => {
            Ok(RealFullDsparkCacheMode::RequestLocal)
        }
        "prompt-swa" | "prompt_swa" | "mode2" | "2" => Ok(RealFullDsparkCacheMode::PromptSwa),
        value => bail!(
            "{REAL_FULL_DSPARK_CACHE_MODE_ENV} must be request-local/mode1 or prompt-swa/mode2, got {value}"
        ),
    }
}

fn request_prompt_token_ids(
    tokenizer: &LoadedTokenizer,
    request: &ds4rt_api::RealFullRequest,
) -> Result<Option<Vec<usize>>> {
    if let Some(token_ids) = request.prompt_token_ids.as_ref() {
        anyhow::ensure!(
            token_ids.len() == request.prompt_tokens,
            "real-full explicit prompt token count differs from request: explicit={} request={}",
            token_ids.len(),
            request.prompt_tokens,
        );
        return Ok(Some(token_ids.as_ref().clone()));
    }
    let encoding = tokenizer
        .encode_text(&request.prompt, false)
        .with_context(|| {
            format!(
                "tokenizing real-full request prompt for {}",
                request.request_id
            )
        })?;
    anyhow::ensure!(
        encoding.token_count == request.prompt_tokens,
        "real-full request tokenizer count changed between API and daemon: api={} daemon={}",
        request.prompt_tokens,
        encoding.token_count
    );
    let token_ids = encoding
        .token_ids
        .into_iter()
        .map(|token_id| token_id as usize)
        .collect::<Vec<_>>();
    Ok(Some(token_ids))
}

pub(crate) fn run_real_ds4_full_preflight(args: &CoordinatorArgs) -> Result<()> {
    let (catalog_path, catalog) = load_real_ds4_full_catalog(args)?;
    validate_real_full_strict_tp4_args(args)?;
    let report = real_ds4_full_preflight_report(args, &catalog_path, &catalog)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    bail!("{}", REAL_DS4_FULL_BLOCKER)
}

fn real_full_expert_ready_timeout_secs() -> Result<u64> {
    let timeout_secs = env::var(REAL_FULL_EXPERT_READY_TIMEOUT_SECS_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(|value| {
            value.parse::<u64>().with_context(|| {
                format!("parsing {REAL_FULL_EXPERT_READY_TIMEOUT_SECS_ENV}={value}")
            })
        })
        .transpose()?
        .unwrap_or(DEFAULT_REAL_FULL_EXPERT_READY_TIMEOUT_SECS);
    anyhow::ensure!(
        timeout_secs > 0,
        "{REAL_FULL_EXPERT_READY_TIMEOUT_SECS_ENV} must be positive"
    );
    Ok(timeout_secs)
}

fn wait_for_real_full_sparse_targets(targets: &[TcpProtocolV2HostBatchTarget]) -> Result<()> {
    let timeout_secs = real_full_expert_ready_timeout_secs()?;
    let started = Instant::now();
    let timeout = Duration::from_secs(timeout_secs);
    let connect_timeout = Duration::from_millis(200);
    let mut pending = targets.iter().collect::<Vec<_>>();
    eprintln!(
        "real_full_expert_readiness_wait targets={} timeout_secs={timeout_secs}",
        pending.len(),
    );
    while !pending.is_empty() {
        pending.retain(|target| TcpStream::connect_timeout(&target.addr, connect_timeout).is_err());
        if pending.is_empty() {
            break;
        }
        if started.elapsed() >= timeout {
            let pending_targets = pending
                .iter()
                .map(|target| format!("{}={}", target.host, target.addr))
                .collect::<Vec<_>>()
                .join(",");
            anyhow::bail!(
                "expert daemons did not become ready within {timeout_secs}s: {pending_targets}"
            );
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    eprintln!(
        "real_full_expert_readiness_ready targets={} elapsed_ms={:.3}",
        targets.len(),
        started.elapsed().as_secs_f64() * 1_000.0,
    );
    Ok(())
}

fn wait_for_real_full_expert_warmup() -> Result<()> {
    let Some(status_file) = env::var_os(REAL_FULL_EXPERT_WARMUP_STATUS_FILE_ENV)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
    else {
        return Ok(());
    };
    let timeout_secs = real_full_expert_ready_timeout_secs()?;
    let started = Instant::now();
    let timeout = Duration::from_secs(timeout_secs);
    eprintln!(
        "real_full_expert_warmup_wait status_file={} timeout_secs={timeout_secs}",
        status_file.display(),
    );
    loop {
        match fs::read_to_string(&status_file) {
            Ok(status) if status.trim() == "ready" => break,
            Ok(status) if status.trim().starts_with("failed") => {
                anyhow::bail!(
                    "expert precompile warmup failed: {}",
                    status.trim().replace('\n', " ")
                );
            }
            Ok(status) => {
                anyhow::bail!(
                    "invalid expert precompile warmup status in {}: {:?}",
                    status_file.display(),
                    status.trim()
                );
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "reading expert precompile warmup status {}",
                        status_file.display()
                    )
                });
            }
        }
        if started.elapsed() >= timeout {
            anyhow::bail!(
                "expert precompile warmup did not finish within {timeout_secs}s: {}",
                status_file.display()
            );
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    eprintln!(
        "real_full_expert_warmup_ready elapsed_ms={:.3}",
        started.elapsed().as_secs_f64() * 1_000.0,
    );
    Ok(())
}

fn report_real_full_startup_phase(
    stage: &str,
    startup_started: Instant,
    phase_started: &mut Instant,
) {
    let now = Instant::now();
    eprintln!(
        "real_full_startup_phase stage={stage} elapsed_ms={:.3} total_ms={:.3}",
        now.duration_since(*phase_started).as_secs_f64() * 1_000.0,
        now.duration_since(startup_started).as_secs_f64() * 1_000.0,
    );
    *phase_started = now;
}

fn prepare_real_full_startup_cuda_graphs(
    catalog: &TensorCatalog,
    target_device_storage: &Arc<Mutex<DeepseekV4TargetDeviceStorage>>,
    dspark_device_storage: Option<&Mutex<DeepseekV4DsparkDeviceStorage>>,
) -> Result<()> {
    let capture_started = Instant::now();
    let report_component = |component: &str, started: Instant| {
        eprintln!(
            "real_full_cuda_graph_capture_component component={component} elapsed_ms={:.3} total_ms={:.3}",
            started.elapsed().as_secs_f64() * 1_000.0,
            capture_started.elapsed().as_secs_f64() * 1_000.0,
        );
    };
    {
        let mut storage = target_device_storage
            .lock()
            .map_err(|error| anyhow::anyhow!("locking target device storage failed: {error}"))?;
        let started = Instant::now();
        storage
            .prepare_entry_graphs(
                catalog,
                catalog.facts.rms_norm_eps,
                catalog.facts.hyper_connection_eps,
                catalog.facts.hyper_connection_sinkhorn_iters,
            )
            .context("capturing native DeepSeek V4 target entry graphs")?;
        report_component("target-entry", started);
        let started = Instant::now();
        storage
            .prepare_sliding_attention_graphs(
                catalog,
                catalog.facts.rms_norm_eps,
                catalog.facts.rms_norm_eps,
                catalog.facts.hyper_connection_eps,
                catalog.facts.hyper_connection_sinkhorn_iters,
            )
            .context("capturing native DeepSeek V4 target C0 attention graphs")?;
        report_component("target-c0-attention", started);
        let started = Instant::now();
        storage
            .prepare_c4_attention_graphs(
                catalog,
                catalog.facts.rms_norm_eps,
                catalog.facts.rms_norm_eps,
                catalog.facts.hyper_connection_eps,
                catalog.facts.hyper_connection_sinkhorn_iters,
            )
            .context("capturing native DeepSeek V4 target C4 attention graphs")?;
        report_component("target-c4-attention", started);
        let started = Instant::now();
        storage
            .prepare_c128_attention_graphs(
                catalog,
                catalog.facts.rms_norm_eps,
                catalog.facts.rms_norm_eps,
                catalog.facts.hyper_connection_eps,
                catalog.facts.hyper_connection_sinkhorn_iters,
            )
            .context("capturing native DeepSeek V4 target C128 attention graphs")?;
        report_component("target-c128-attention", started);
        let started = Instant::now();
        storage
            .prepare_post_pre_graphs(
                catalog,
                catalog.facts.rms_norm_eps,
                catalog.facts.hyper_connection_eps,
                catalog.facts.hyper_connection_sinkhorn_iters,
            )
            .context("capturing native DeepSeek V4 target post/pre graphs")?;
        report_component("target-post-pre", started);
        let started = Instant::now();
        storage
            .prepare_terminal_graphs(
                catalog.facts.rms_norm_eps,
                catalog.facts.hyper_connection_eps,
            )
            .context("capturing native DeepSeek V4 target terminal graphs")?;
        report_component("target-terminal", started);
        let started = Instant::now();
        storage
            .smoke_replay_connected_attention(catalog.facts.rope_theta)
            .context("replaying native DeepSeek V4 connected target lifecycles")?;
        report_component("target-smoke-replay", started);
    }
    if let Some(storage) = dspark_device_storage {
        let mut storage = storage
            .lock()
            .map_err(|error| anyhow::anyhow!("locking dSpark device storage failed: {error}"))?;
        let started = Instant::now();
        storage
            .prepare_entry_projection_graphs(catalog.facts.rms_norm_eps)
            .context("capturing integrated dSpark entry projection graphs")?;
        report_component("dspark-entry-projection", started);
        let started = Instant::now();
        storage
            .prepare_prompt_prime_graphs(catalog.facts.rms_norm_eps, catalog.facts.rope_theta)
            .context("capturing integrated dSpark prompt KV graphs")?;
        report_component("dspark-prompt-prime", started);
        let started = Instant::now();
        storage
            .prepare_proposal_entry_graphs(
                catalog,
                catalog.facts.rms_norm_eps,
                catalog.facts.hyper_connection_eps,
                catalog.facts.hyper_connection_sinkhorn_iters,
            )
            .context("capturing integrated dSpark proposal-entry graphs")?;
        report_component("dspark-proposal-entry", started);
        let started = Instant::now();
        storage
            .prepare_block_attention_graphs(
                catalog.facts.rms_norm_eps,
                catalog.facts.rms_norm_eps,
                catalog.facts.hyper_connection_eps,
                catalog.facts.hyper_connection_sinkhorn_iters,
            )
            .context("capturing integrated dSpark block-attention graphs")?;
        report_component("dspark-block-attention", started);
        let started = Instant::now();
        storage
            .prepare_block_post_dispatch_graphs(
                catalog.facts.rms_norm_eps,
                catalog.facts.hyper_connection_eps,
                catalog.facts.hyper_connection_sinkhorn_iters,
            )
            .context("capturing integrated dSpark inter-block post-dispatch graphs")?;
        report_component("dspark-block-post-dispatch", started);
        let started = Instant::now();
        storage
            .prepare_terminal_collapse_graphs(
                catalog.facts.rms_norm_eps,
                catalog.facts.hyper_connection_eps,
            )
            .context("capturing integrated dSpark terminal collapse graphs")?;
        report_component("dspark-terminal-collapse", started);
        let started = Instant::now();
        storage
            .prepare_terminal_head_graphs()
            .context("capturing integrated dSpark joint terminal head width buckets")?;
        report_component("dspark-terminal-head", started);
        let started = Instant::now();
        storage
            .prepare_block_sparse_pre_dispatch(catalog)
            .context("preparing integrated dSpark global router and shared-expert launch")?;
        report_component("dspark-sparse-pre-dispatch", started);
    }
    Ok(())
}

pub(crate) fn load_real_full_serving(
    args: &CoordinatorArgs,
    python_capture_barrier: impl FnOnce() -> Result<()>,
) -> Result<LoadedRealFullServing> {
    let startup_started = Instant::now();
    let mut phase_started = startup_started;
    anyhow::ensure!(
        args.backend == "real-ds4-full",
        "DeepSeek V4 serving requires --backend real-ds4-full; legacy backend {:?} is not a serving mode",
        args.backend
    );
    anyhow::ensure!(
        coordinator_python_capture_enabled(),
        "real-ds4-full serving requires DS4RT_B12X=1 for startup-captured production kernels"
    );
    anyhow::ensure!(
        real_full_serve_prewarm_request_enabled(),
        "real-ds4-full serving requires startup prewarm so all production CUDA graphs are captured before serving"
    );
    report_real_full_startup_phase("validation", startup_started, &mut phase_started);
    let (_catalog_path, catalog) = load_real_ds4_full_catalog(args)?;
    anyhow::ensure!(
        catalog.facts.model_type == "deepseek_v4",
        "real-ds4-full serving requires model_type deepseek_v4, got {:?}",
        catalog.facts.model_type
    );
    let dspark_mode = real_full_dspark_mode()?;
    let mut kv_config = real_full_kv_cache_config_for_model(args, &catalog.facts)?;
    if dspark_mode.is_some() {
        anyhow::ensure!(
            catalog.facts.model_type == "deepseek_v4",
            "native dSpark requires a DeepSeek V4 target catalog, got {}",
            catalog.facts.model_type
        );
        kv_config = kv_config.with_deepseek_v4_dspark_layers(&catalog.facts);
    }
    let kv_config = kv_config.with_mla_representation(MlaKvCacheRepresentation::NormalizedRotated);
    println!(
        "real-ds4-full MLA KV representation={}",
        kv_config.mla_representation.label()
    );
    report_real_full_startup_phase("catalog-kv-config", startup_started, &mut phase_started);
    let sparse_dispatch_transport =
        RealFullSchedulerSparseDispatchTransport::from_label(args.transport.as_str());
    validate_real_full_strict_tp4_args(args)?;
    let sparse_tcp_targets = real_full_sparse_tcp_targets_from_args(args)?;
    let tokenizer = LoadedTokenizer::from_snapshot(Path::new(&catalog.snapshot_path))
        .context("loading real-full serving tokenizer")?;
    let constraint_vocab_size = catalog
        .tensors
        .iter()
        .find(|tensor| tensor.role == TensorRole::LmHead)
        .and_then(|tensor| tensor.shape.first().copied())
        .context("real-full constrained decoding requires a 2D lm_head tensor")?;
    let constraint_compiler = RealFullConstraintCompiler::new(
        Path::new(&catalog.snapshot_path).join("tokenizer.json"),
        constraint_vocab_size,
    )
    .context("configuring lazy real-full constrained decoding")?;
    report_real_full_startup_phase("targets-tokenizer", startup_started, &mut phase_started);
    let kv_snapshot_load_path = optional_nonempty_env_path(REAL_FULL_KV_SNAPSHOT_LOAD_ENV)?;
    let kv_snapshot_save_path = optional_nonempty_env_path(REAL_FULL_KV_SNAPSHOT_SAVE_ENV)?;
    let kv_snapshot_save_tokens =
        optional_positive_env_usize(REAL_FULL_KV_SNAPSHOT_SAVE_TOKENS_ENV)?;
    anyhow::ensure!(
        kv_snapshot_save_path.is_some() || kv_snapshot_save_tokens.is_none(),
        "{REAL_FULL_KV_SNAPSHOT_SAVE_TOKENS_ENV} requires {REAL_FULL_KV_SNAPSHOT_SAVE_ENV}"
    );
    let mut kv_snapshot_saves =
        real_full_kv_snapshot_save_points(REAL_FULL_KV_SNAPSHOT_SAVE_POINTS_ENV)?;
    if let Some(root) = kv_snapshot_save_path {
        kv_snapshot_saves.push(RealFullKvSnapshotSave {
            root,
            token_count: kv_snapshot_save_tokens,
        });
    }
    validate_real_full_kv_snapshot_saves(&kv_snapshot_saves)?;
    let snapshot_enabled = kv_snapshot_load_path.is_some() || !kv_snapshot_saves.is_empty();
    anyhow::ensure!(
        !snapshot_enabled,
        "packed KV snapshots are unavailable for native DeepSeek target serving until the format persists the authoritative physical KV/index pages and sequence-local compressor state"
    );
    let kv_snapshot_load = kv_snapshot_load_path
        .as_deref()
        .map(|path| load_real_full_kv_snapshot(path, &catalog, &kv_config))
        .transpose()?
        .map(Arc::new);
    if let Some(snapshot) = kv_snapshot_load.as_ref() {
        println!(
            "real-full packed KV snapshot loaded path={} tokens={} dspark_layer_tokens={} (device restore is deferred until request admission)",
            snapshot.root().display(),
            snapshot.token_count(),
            snapshot.dspark_layer_token_count(),
        );
    }
    for save in &kv_snapshot_saves {
        anyhow::ensure!(
            !save.root.exists(),
            "KV snapshot save destination already exists: {}",
            save.root.display()
        );
    }
    report_real_full_startup_phase("kv-snapshot-config", startup_started, &mut phase_started);
    let engine_commit = env::var(REAL_FULL_ENGINE_COMMIT_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "unknown".to_owned());
    let prefix_prefill_probe =
        real_full_serve_prefix_prefill_probe(&tokenizer, kv_config.max_tokens)?;
    let prewarm_prompts = if real_full_serve_prewarm_request_enabled()
        || prefix_prefill_probe.is_some()
    {
        let prefill_chunk_tokens = real_full_request_prefill_chunk_tokens();
        let prewarm_prefill_rows =
            real_full_serve_prewarm_prefill_rows(kv_config.max_tokens, prefill_chunk_tokens)?;
        let prompts = prewarm_prefill_rows
            .iter()
            .map(|prefill_rows| {
                let prompt = REAL_FULL_SERVE_PREWARM_PROMPT_TOKEN.repeat(*prefill_rows);
                let prompt_tokens = tokenizer
                    .encode_text(&prompt, false)
                    .context("tokenizing real-full serving prewarm prompt")?
                    .token_count;
                anyhow::ensure!(
                    prompt_tokens == prefill_rows + 1,
                    "real-full serving prewarm prompt produced {prompt_tokens} tokens for {prefill_rows} requested prefill rows"
                );
                Ok((prompt, prompt_tokens))
            })
            .collect::<Result<Vec<_>>>()?;
        let largest_prompt_tokens = prompts
            .first()
            .map(|(_, prompt_tokens)| *prompt_tokens)
            .context("real-full serving prewarm prompt set is empty")?;
        let prefill_tokens = largest_prompt_tokens.saturating_sub(1);
        anyhow::ensure!(
            prefill_tokens >= prefill_chunk_tokens,
            "real-full serving largest prewarm prompt token count {largest_prompt_tokens} does not include an exact {}-row prefill chunk",
            prefill_chunk_tokens,
        );
        Some(prompts)
    } else {
        None
    };
    report_real_full_startup_phase("prewarm-prompts", startup_started, &mut phase_started);
    let preload = preload_real_full_coordinator_resident_weights(&catalog)?;
    println!(
        "real-ds4-full coordinator resident preload status={} tensors={} bytes={}",
        preload.status, preload.selected_tensor_count, preload.loaded_tensor_bytes
    );
    report_real_full_startup_phase(
        "coordinator-resident-preload",
        startup_started,
        &mut phase_started,
    );
    let max_execution_lanes = real_full_max_execution_lanes()?;
    let max_context_tokens = kv_config.max_tokens;
    let kv_pool_tokens = real_full_kv_pool_tokens(max_context_tokens)?;
    anyhow::ensure!(
        kv_pool_tokens % DS4_KV_SOURCE_PAGE_TOKENS == 0,
        "{REAL_FULL_KV_POOL_TOKENS_ENV}={kv_pool_tokens} must be divisible by the native DeepSeek source-page width {DS4_KV_SOURCE_PAGE_TOKENS}"
    );
    let native_kv_cache_format = match KvCacheDType::parse_cache_dtype(&args.kv_cache_dtype)
        .context("parsing validated native DeepSeek V4 KV cache dtype")?
    {
        KvCacheDType::Nvfp4 => DeepseekV4KvCacheFormat::Nvfp4,
        KvCacheDType::Bf16 | KvCacheDType::Fp8 => DeepseekV4KvCacheFormat::Fp8Ue8m0,
        dtype => anyhow::bail!(
            "unsupported validated DeepSeek V4 KV dtype {}",
            dtype.label()
        ),
    };
    // Draft KV is an execution lease, not queued-request or target-radix
    // residency. Size this pool for the lanes that can actually execute;
    // submitted requests beyond that limit must wait without owning pages.
    let dspark_load_started = Instant::now();
    let dspark = dspark_mode
        .map(|mode| {
            let cache_mode = real_full_dspark_cache_mode()?;
            let confidence_policy = real_full_dspark_confidence_policy()?;
            let context_tokens = catalog.facts.sliding_window;
            let kv_bytes_per_request = catalog
                .facts
                .dspark_target_layer_ids
                .len()
                .checked_mul(native_dspark_cache_page_bytes(native_kv_cache_format))
                .context("native dSpark packed KV bytes/request overflow")?;
            let tail_cache_bytes = real_full_dspark_tail_cache_bytes(kv_bytes_per_request)?;
            let kv_capacity_tokens = context_tokens
                .checked_add(REAL_FULL_DSPARK_PAGE_SIZE)
                .context("dSpark request page-slop capacity overflow")?
                .checked_add(catalog.facts.dspark_block_size)
                .context("dSpark request KV capacity overflow")?;
            let engine = DsparkRequestEngine::load_with_cache_page_bytes(
                &catalog,
                kv_capacity_tokens,
                max_execution_lanes,
                native_dspark_cache_page_bytes(native_kv_cache_format),
            )
            .context("loading integrated DeepSeek dSpark request executor")?;
            let max_verify_drafts = engine.max_verify_drafts();
            if let Some(fixed_drafts) = real_full_dspark_fixed_drafts()? {
                anyhow::ensure!(
                    fixed_drafts <= max_verify_drafts,
                    "fixed dSpark width {fixed_drafts} exceeds the active checkpoint maximum {max_verify_drafts}"
                );
            }
            let mut cost_model = DsparkRuntimeCostModel::new(
                REAL_FULL_MAX_ACTIVE_REQUESTS,
                max_verify_drafts,
            )?;
            let embedded_sps_profile = if catalog.facts.hidden_size == 4_096 {
                anyhow::ensure!(
                    max_verify_drafts == NATIVE_DSPARK_PROPOSAL_TOKENS,
                    "embedded Flash dSpark SPS profile expects {} proposals, got {max_verify_drafts}",
                    NATIVE_DSPARK_PROPOSAL_TOKENS,
                );
                for (request_index, rows) in
                    FLASH_0731_K2_AFD_SPS_PROFILE_MS.iter().enumerate()
                {
                    cost_model.install_profile(request_index + 1, rows)?;
                }
                Some(FLASH_0731_K2_AFD_SPS_PROFILE_ID)
            } else {
                None
            };
            println!(
                "real-full native dSpark ready mode={mode:?} confidence_policy={confidence_policy:?} cache_mode={cache_mode:?} algorithm=integrated-dspark storage_namespace=mtp.* context_tokens={} kv_capacity_tokens={} max_verify_drafts={} gpu_request_slots={} host_tail_cache_bytes={} runtime_cost_max_rows={} runtime_context_bucket_tokens={} sps_profile={}",
                context_tokens,
                kv_capacity_tokens,
                max_verify_drafts,
                max_execution_lanes,
                tail_cache_bytes,
                REAL_FULL_MAX_ACTIVE_REQUESTS * (max_verify_drafts + 1),
                super::dspark::DSPARK_RUNTIME_CONTEXT_BUCKET_TOKENS,
                embedded_sps_profile.unwrap_or("unqualified-runtime-prior"),
            );
            Ok::<_, anyhow::Error>(Mutex::new(RealFullDsparkRuntime {
                mode,
                confidence_policy,
                cache_mode,
                context_tokens,
                engine,
                requests: HashMap::new(),
                tail_cache: RealFullDsparkTailCache::new(tail_cache_bytes),
                cost_model,
            }))
        })
        .transpose()?;
    eprintln!(
        "real_full_dspark_preload elapsed_ms={:.3} enabled={}",
        dspark_load_started.elapsed().as_secs_f64() * 1_000.0,
        dspark.is_some(),
    );
    report_real_full_startup_phase("dspark-preload", startup_started, &mut phase_started);
    let dspark_device_storage = if dspark.is_some() {
        let variant = match catalog.facts.hidden_size {
            4_096 => "flash",
            7_168 => "pro",
            hidden => anyhow::bail!(
                "integrated dSpark device storage does not support hidden width {hidden}"
            ),
        };
        let plan = plan_deepseek_v4_dspark_device_storage_with_cache_format(
            variant,
            REAL_FULL_MAX_ACTIVE_REQUESTS,
            REAL_FULL_DSPARK_MAX_MAIN_ROWS,
            native_kv_cache_format,
        )
        .context("planning integrated dSpark GPU0 storage")?;
        let storage = DeepseekV4DsparkDeviceStorage::new(plan)
            .context("allocating integrated dSpark GPU0 storage")?;
        eprintln!(
            "real_full_dspark_device_storage variant={variant} max_batch={} max_main_rows={} arena_bytes={} persistent_kv_bytes={} device=0",
            plan.max_batch,
            plan.max_main_rows,
            plan.arena_bytes,
            plan.persistent_kv_bytes,
        );
        Some(Mutex::new(storage))
    } else {
        None
    };
    report_real_full_startup_phase("dspark-device-storage", startup_started, &mut phase_started);
    let (target_device_storage, target_device_identity) = {
        let plan = plan_deepseek_v4_target_device_storage_with_cache_format(
            &catalog.facts,
            max_execution_lanes,
            DEFAULT_REAL_FULL_REQUEST_PREFILL_CHUNK_TOKENS.min(kv_config.max_tokens),
            kv_config.max_tokens,
            kv_pool_tokens,
            native_kv_cache_format,
        )
        .context("planning native DeepSeek V4 target GPU0 storage")?;
        eprintln!(
            "real_full_target_device_plan variant={} cache_format={} target_layers={} execution_lanes={} max_graph_rows={} max_sequence_rows={} max_sequence_pages={} hc_active_rows_per_lane={} physical_pool_rows={} source_page_tokens={} source_pages={} physical_page_bytes={} workspace_bytes_per_lane={} lane_stride_bytes={} lane_storage_bytes={} persistent_hc_bytes={} physical_kv_bytes={} compressor_state_bytes={} total_bytes={} expert_tp={} expert_parallel={} status={}",
            plan.variant,
            plan.cache_format,
            plan.target_layers,
            plan.execution_lanes,
            plan.max_graph_rows,
            plan.max_sequence_rows,
            plan.max_sequence_pages,
            plan.hc_active_rows_per_lane,
            plan.physical_pool_rows,
            plan.source_page_tokens,
            plan.source_pages,
            plan.physical_kv.bytes / plan.source_pages,
            plan.workspace.bytes,
            plan.lane_stride_bytes,
            plan.lane_storage.bytes,
            plan.residual_ping.bytes * 2 + plan.post_ping.bytes * 2 + plan.comb_ping.bytes * 2,
            plan.physical_kv.bytes,
            plan.compressor_state.bytes,
            plan.total_bytes,
            plan.expert_tensor_parallel,
            plan.expert_parallel,
            plan.status,
        );
        let identity = RealFullSchedulerNativeTargetIdentity::from_plan(&plan);
        let storage = DeepseekV4TargetDeviceStorage::new(plan)
            .context("allocating native DeepSeek V4 target GPU0 storage")?;
        eprintln!(
            "real_full_target_device_storage allocated_bytes={} device=0 capture_streams={} status={}",
            storage.plan().total_bytes,
            storage.plan().execution_lanes,
            storage.plan().status,
        );
        (Arc::new(Mutex::new(storage)), identity)
    };
    report_real_full_startup_phase("target-device-plan", startup_started, &mut phase_started);
    // Spark expert residency is independent of coordinator CUDA graph capture.
    // The expert processes are dispatched before this loader starts, so do the
    // coordinator-only work while their disks and GPUs are busy instead of
    // waiting for all four TCP readiness barriers first.
    python_capture_barrier().context("waiting for coordinator Python capture initialization")?;
    report_real_full_startup_phase(
        "python-capture-barrier",
        startup_started,
        &mut phase_started,
    );
    prepare_real_full_startup_cuda_graphs(
        &catalog,
        &target_device_storage,
        dspark_device_storage.as_ref(),
    )?;
    report_real_full_startup_phase(
        "coordinator-cuda-graph-capture",
        startup_started,
        &mut phase_started,
    );
    wait_for_real_full_sparse_targets(&sparse_tcp_targets)?;
    report_real_full_startup_phase("sparse-target-connect", startup_started, &mut phase_started);
    wait_for_real_full_expert_warmup()?;
    report_real_full_startup_phase("expert-warmup", startup_started, &mut phase_started);
    let sparse_dispatch_worker_cpu = real_full_scheduler_worker_cpu()?;
    let sparse_tcp_dispatch_worker = Arc::new(
        RealFullSchedulerSparseTcpDispatchWorker::new_with_transport_and_cpu_affinity_for_model(
            sparse_dispatch_transport
                .expect("configured sparse dispatch target has supported transport"),
            sparse_tcp_targets.clone(),
            None,
            sparse_dispatch_worker_cpu,
            &catalog.facts,
        )?,
    );
    sparse_tcp_dispatch_worker.preallocate_ds4_flash_tp4_reduction_arena()?;
    report_real_full_startup_phase("dispatch-worker", startup_started, &mut phase_started);
    let mut info = real_full_info_from_startup(args, &catalog, preload)?;
    info.kv_bytes_per_token = kv_config.bytes_per_token();
    initialize_sparse_tcp_dispatch_status(&mut info, &sparse_tcp_targets);
    let serving_kv_config = kv_config.clone();
    let mut device_kv_pool_config = kv_config.clone();
    device_kv_pool_config.max_tokens = kv_pool_tokens;
    let target_kv_page_tokens = target_device_storage
        .lock()
        .map(|storage| storage.plan().source_page_tokens)
        .map_err(|error| anyhow::anyhow!("locking native target plan failed: {error}"))?;
    let device_page_mapping_bytes_per_lane = kv_pool_tokens
        .div_ceil(REAL_FULL_SHARED_KV_PAGE_TOKENS)
        .checked_mul(std::mem::size_of::<u32>())
        .context("native target device page mapping byte count overflows usize")?;
    eprintln!(
        "real_full_kv_pool logical_max_tokens={} physical_pool_tokens={} mapping_page_tokens={} target_page_tokens={} mapping_bytes_per_lane={} generic_payload_allocated_bytes=0 generic_payload_avoided_bytes={} execution_lanes={}",
        max_context_tokens,
        kv_pool_tokens,
        REAL_FULL_SHARED_KV_PAGE_TOKENS,
        target_kv_page_tokens,
        device_page_mapping_bytes_per_lane,
        device_kv_pool_config.capacity_bytes(),
        max_execution_lanes,
    );
    let scheduler_executor = RealFullSchedulerRequestExecutor {
        base_info: info.clone(),
        catalog,
        kv_config,
        device_kv_pool_config,
        sparse_tcp_targets,
        sparse_tcp_dispatch_worker,
        scheduler_states: Mutex::new(HashMap::new()),
        recycled_scheduler_states: Mutex::new(Vec::new()),
        max_execution_lanes,
        device_kv_storage: Mutex::new(None),
        context_budget: Arc::new(RealFullContextTokenBudget::new(kv_pool_tokens)),
        target_kv_radix: Arc::new(
            TargetKvRadixManager::new_with_page_tokens(
                kv_pool_tokens,
                max_execution_lanes,
                target_kv_page_tokens,
            )
            .context("creating shared target KV radix manager")?,
        ),
        sampled_token_text_cache: Mutex::new(HashMap::new()),
        tokenizer: Mutex::new(tokenizer),
        constraint_compiler,
        kv_snapshot_load,
        kv_snapshot_saves,
        kv_snapshot_saved: AtomicBool::new(false),
        dspark,
        dspark_device_storage,
        target_device_storage,
        target_device_identity,
        engine_commit,
    };
    report_real_full_startup_phase("executor-assembly", startup_started, &mut phase_started);
    let executor: Arc<dyn ds4rt_api::RealFullRequestExecutor> =
        if real_full_request_thread_pinned_enabled() {
            let worker_count = real_full_request_thread_pinned_workers()?;
            let worker_cpus = real_full_request_worker_cpus(worker_count)?;
            let executor =
                ds4rt_api::ThreadPinnedRealFullRequestExecutor::spawn_pool_with_cpu_affinity(
                    "ds4rt-real-full-request-worker",
                    scheduler_executor,
                    worker_count,
                    &worker_cpus,
                )
                .context("spawning thread-pinned real-full request worker")?;
            report_real_full_startup_phase(
                "request-worker-spawn",
                startup_started,
                &mut phase_started,
            );
            if let Some(prompts) = prewarm_prompts.as_ref() {
                for worker_index in 0..executor.worker_count() {
                    if max_execution_lanes > 1 && real_full_dspark_enabled() {
                        executor
                            .finish_real_full_sequence_on_worker(
                                worker_index,
                                format!(
                                    "{REAL_FULL_STARTUP_PREWARM_PAIRED_LM_HEAD_PREFIX}{worker_index}"
                                ),
                            )
                            .map_err(anyhow::Error::msg)
                            .with_context(|| {
                                format!(
                                    "prewarming paired LM-head graphs for worker {worker_index}"
                                )
                            })?;
                    }
                    report_real_full_startup_phase(
                        "prewarm-paired-lm-head-initial",
                        startup_started,
                        &mut phase_started,
                    );
                    prewarm_real_full_serving_requests(
                        |request| {
                            executor.execute_real_full_decode_cycle_on_worker(worker_index, request)
                        },
                        |sequence_id| {
                            executor.finish_real_full_sequence_on_worker(
                                worker_index,
                                sequence_id.to_owned(),
                            )
                        },
                        prompts,
                        prefix_prefill_probe.as_ref(),
                        worker_index,
                        max_context_tokens,
                    )?;
                    report_real_full_startup_phase(
                        "prewarm-main",
                        startup_started,
                        &mut phase_started,
                    );
                    if max_execution_lanes > 1 && real_full_dspark_enabled() {
                        executor
                            .finish_real_full_sequence_on_worker(
                                worker_index,
                                format!(
                                    "{REAL_FULL_STARTUP_PREWARM_BATCHED_DSPARK_PREFIX}{worker_index}"
                                ),
                            )
                            .map_err(anyhow::Error::msg)
                            .with_context(|| {
                                format!(
                                    "prewarming batched dSpark graphs for worker {worker_index}"
                                )
                            })?;
                        report_real_full_startup_phase(
                            "prewarm-batched-dspark",
                            startup_started,
                            &mut phase_started,
                        );
                    }
                    executor
                        .finish_real_full_sequence_on_worker(
                            worker_index,
                            format!(
                                "{REAL_FULL_STARTUP_SEAL_OWNED_BUFFER_POOL_PREFIX}{worker_index}"
                            ),
                        )
                        .map_err(anyhow::Error::msg)
                        .with_context(|| {
                            format!(
                                "sealing coordinator owned device-buffer pool for worker {worker_index}"
                            )
                        })?;
                }
            }
            report_real_full_startup_phase(
                "prewarm-audit-seal",
                startup_started,
                &mut phase_started,
            );
            Arc::new(executor)
        } else {
            report_real_full_startup_phase(
                "request-worker-inline",
                startup_started,
                &mut phase_started,
            );
            if let Some(prompts) = prewarm_prompts.as_ref() {
                if max_execution_lanes > 1 && real_full_dspark_enabled() {
                    ds4rt_api::RealFullRequestExecutor::finish_real_full_sequence(
                        &scheduler_executor,
                        REAL_FULL_STARTUP_PREWARM_PAIRED_LM_HEAD_PREFIX,
                    )
                    .map_err(anyhow::Error::msg)
                    .context("prewarming paired LM-head graphs")?;
                }
                report_real_full_startup_phase(
                    "prewarm-paired-lm-head-initial",
                    startup_started,
                    &mut phase_started,
                );
                prewarm_real_full_serving_requests(
                    |request| {
                        ds4rt_api::RealFullRequestExecutor::execute_real_full_decode_cycle(
                            &scheduler_executor,
                            request,
                        )
                    },
                    |sequence_id| {
                        ds4rt_api::RealFullRequestExecutor::finish_real_full_sequence(
                            &scheduler_executor,
                            sequence_id,
                        )
                    },
                    prompts,
                    prefix_prefill_probe.as_ref(),
                    0,
                    max_context_tokens,
                )?;
                report_real_full_startup_phase("prewarm-main", startup_started, &mut phase_started);
                if max_execution_lanes > 1 && real_full_dspark_enabled() {
                    ds4rt_api::RealFullRequestExecutor::finish_real_full_sequence(
                        &scheduler_executor,
                        REAL_FULL_STARTUP_PREWARM_BATCHED_DSPARK_PREFIX,
                    )
                    .map_err(anyhow::Error::msg)
                    .context("prewarming batched dSpark graphs")?;
                    report_real_full_startup_phase(
                        "prewarm-batched-dspark",
                        startup_started,
                        &mut phase_started,
                    );
                }
                seal_coordinator_owned_device_buffer_pool()
                    .context("sealing coordinator owned device-buffer pool")?;
            }
            report_real_full_startup_phase(
                "prewarm-audit-seal",
                startup_started,
                &mut phase_started,
            );
            Arc::new(scheduler_executor)
        };
    report_real_full_startup_phase("complete", startup_started, &mut phase_started);
    Ok(LoadedRealFullServing {
        info,
        kv_config: serving_kv_config,
        executor,
    })
}

fn real_full_serve_prewarm_request_enabled() -> bool {
    env::var(REAL_FULL_SERVE_PREWARM_REQUEST_ENV)
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or_else(|_| coordinator_python_capture_enabled())
}

fn real_full_serve_prewarm_prefill_rows(
    max_context_tokens: usize,
    prefill_chunk_tokens: usize,
) -> Result<Vec<usize>> {
    let max_prefill_rows = max_context_tokens.checked_sub(2).context(
        "real-full serving prewarm needs room for the tokenizer boundary and one output token",
    )?;
    let max_canonical_rows =
        max_prefill_rows / REAL_FULL_SHARED_KV_PAGE_TOKENS * REAL_FULL_SHARED_KV_PAGE_TOKENS;
    anyhow::ensure!(
        max_canonical_rows >= prefill_chunk_tokens,
        "real-full serving context {max_context_tokens} cannot prewarm one {prefill_chunk_tokens}-row prefill chunk"
    );
    let mut rows = REAL_FULL_SERVE_PREWARM_PREFILL_ROWS
        .iter()
        .copied()
        .filter(|rows| *rows <= max_prefill_rows)
        .collect::<Vec<_>>();
    rows.push(prefill_chunk_tokens.min(max_canonical_rows));
    rows.sort_unstable_by(|left, right| right.cmp(left));
    rows.dedup();
    Ok(rows)
}

fn real_full_serve_dsa_selector_seed_prompt<'a>(
    prompts: &'a [(String, usize)],
    query_rows: &[usize],
    decode_budget: usize,
    max_context_tokens: usize,
) -> Result<&'a (String, usize)> {
    let selector_extension_tokens = query_rows.iter().try_fold(0_usize, |tokens, query_rows| {
        query_rows
            .checked_add(1)
            .and_then(|query_tokens| tokens.checked_add(query_tokens))
            .context("DSA selector prewarm extension token count overflow")
    })?;
    let max_seed_tokens = max_context_tokens
        .checked_sub(decode_budget)
        .and_then(|tokens| tokens.checked_sub(selector_extension_tokens))
        .context("real-full serving context is too short for the DSA selector prewarm sweep")?;
    prompts
        .iter()
        .find(|(_, prompt_tokens)| {
            *prompt_tokens >= REAL_FULL_SERVE_NO_SELECTOR_DSA_BOUNDARY_PROMPT_TOKENS
                && *prompt_tokens <= max_seed_tokens
        })
        .with_context(|| {
            format!(
                "real-full serving DSA selector prewarm has no canonical prompt in {}..={max_seed_tokens} tokens for context {max_context_tokens}",
                REAL_FULL_SERVE_NO_SELECTOR_DSA_BOUNDARY_PROMPT_TOKENS,
            )
        })
}

fn real_full_serve_prefix_prefill_probe_enabled() -> bool {
    env::var(REAL_FULL_SERVE_PREFIX_PREFILL_PROBE_ENV)
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn real_full_serve_prefix_prefill_probe_repeats() -> Result<usize> {
    if !real_full_serve_prefix_prefill_probe_enabled() {
        return Ok(1);
    }
    let repeats = env::var(REAL_FULL_SERVE_PREFIX_PREFILL_PROBE_REPEATS_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(|value| {
            value.parse::<usize>().with_context(|| {
                format!("parsing {REAL_FULL_SERVE_PREFIX_PREFILL_PROBE_REPEATS_ENV} value {value}")
            })
        })
        .transpose()?
        .unwrap_or(1);
    anyhow::ensure!(
        (1..=MAX_REAL_FULL_SERVE_PREFIX_PREFILL_PROBE_REPEATS).contains(&repeats),
        "{REAL_FULL_SERVE_PREFIX_PREFILL_PROBE_REPEATS_ENV} must be in 1..={MAX_REAL_FULL_SERVE_PREFIX_PREFILL_PROBE_REPEATS}, got {repeats}"
    );
    Ok(repeats)
}

fn real_full_serve_prefix_prefill_probe_rows(name: &str) -> Result<Vec<usize>> {
    let values = env::var(name).ok().filter(|value| !value.trim().is_empty());
    let rows = if let Some(values) = values {
        values
            .split(',')
            .map(str::trim)
            .map(|value| {
                anyhow::ensure!(!value.is_empty(), "{name} contains an empty row count");
                let rows = value
                    .parse::<usize>()
                    .with_context(|| format!("parsing {name} value {value}"))?;
                anyhow::ensure!(rows > 0, "{name} row counts must be greater than zero");
                Ok(rows)
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        vec![DEFAULT_REAL_FULL_SERVE_PREFIX_PREFILL_PROBE_ROWS]
    };
    anyhow::ensure!(
        !rows.is_empty(),
        "{name} must contain at least one row count"
    );
    Ok(rows)
}

fn real_full_serve_prefix_prefill_probe(
    tokenizer: &LoadedTokenizer,
    max_context_tokens: usize,
) -> Result<Option<RealFullPrefixPrefillProbe>> {
    if !real_full_serve_prefix_prefill_probe_enabled() {
        return Ok(None);
    }
    let prefix_rows = real_full_serve_prefix_prefill_probe_rows(
        REAL_FULL_SERVE_PREFIX_PREFILL_PROBE_PREFIX_ROWS_ENV,
    )?;
    let new_prompt_rows = real_full_serve_prefix_prefill_probe_rows(
        REAL_FULL_SERVE_PREFIX_PREFILL_PROBE_NEW_ROWS_ENV,
    )?;
    anyhow::ensure!(
        new_prompt_rows.iter().all(|rows| *rows > 1),
        "{REAL_FULL_SERVE_PREFIX_PREFILL_PROBE_NEW_ROWS_ENV} must be greater than one"
    );
    let case_count = prefix_rows
        .len()
        .checked_mul(new_prompt_rows.len())
        .context("real-full serving prefix-prefill probe case count overflow")?;
    let mut cases = Vec::with_capacity(case_count);
    for prefix_rows in prefix_rows {
        let prefix_prompt = REAL_FULL_SERVE_PREWARM_PROMPT_TOKEN.repeat(prefix_rows);
        let prefix_prompt_tokens = tokenizer
            .encode_text(&prefix_prompt, false)
            .context("tokenizing real-full serving prefix-prefill probe prefix")?
            .token_count;
        anyhow::ensure!(
            prefix_prompt_tokens == prefix_rows + 1,
            "real-full serving prefix-prefill probe prefix produced {prefix_prompt_tokens} tokens for {prefix_rows} requested rows"
        );
        for new_prompt_rows in new_prompt_rows.iter().copied() {
            let seed_decode_budget = new_prompt_rows
                .checked_add(1)
                .context("real-full serving prefix-prefill probe decode budget overflow")?;
            real_full_sequence_capacity_tokens(
                prefix_prompt_tokens,
                seed_decode_budget,
                max_context_tokens,
            )
            .context("reserving real-full serving prefix-prefill probe context")?;
            cases.push(RealFullPrefixPrefillProbeCase {
                prefix_prompt: prefix_prompt.clone(),
                prefix_prompt_tokens,
                new_prompt_rows,
            });
        }
    }
    Ok(Some(RealFullPrefixPrefillProbe {
        cases,
        repeats: real_full_serve_prefix_prefill_probe_repeats()?,
    }))
}

fn real_full_startup_workspace_is_final_capture_set(prefix_prefill_probe_enabled: bool) -> bool {
    !prefix_prefill_probe_enabled
}

fn prewarm_real_full_serving_requests(
    mut execute: impl FnMut(
        ds4rt_api::RealFullRequest,
    ) -> std::result::Result<ds4rt_api::RealFullDecodeCycle, String>,
    mut finish_sequence: impl FnMut(&str) -> std::result::Result<(), String>,
    prompts: &[(String, usize)],
    prefix_prefill_probe: Option<&RealFullPrefixPrefillProbe>,
    worker_index: usize,
    max_context_tokens: usize,
) -> Result<()> {
    let start = Instant::now();
    let canonical_workspace_complete =
        real_full_startup_workspace_is_final_capture_set(prefix_prefill_probe.is_some());
    let dsa_selector_query_rows = REAL_FULL_SERVE_DSA_SELECTOR_PREWARM_QUERY_ROWS
        .iter()
        .copied()
        .filter(|query_rows| *query_rows < real_full_request_prefill_chunk_tokens())
        .collect::<Vec<_>>();
    let dsa_selector_decode_budget = dsa_selector_query_rows.len() + 1;
    let mut recurrent_seed = None;
    let configured_prefill_chunk_tokens = real_full_request_prefill_chunk_tokens();
    let max_prefill_rows = max_context_tokens.checked_sub(2).context(
        "real-full serving max-chunk prewarm needs a tokenizer boundary and output token",
    )?;
    // Two full-width chunks are enough to enter the pipelined incremental
    // prefill path and establish its maximum-sized permanent buffers.
    let max_chunk_sizing_rows = configured_prefill_chunk_tokens
        .checked_mul(2)
        .filter(|rows| *rows <= max_prefill_rows)
        .unwrap_or(configured_prefill_chunk_tokens);
    let max_chunk_prompt = REAL_FULL_SERVE_PREWARM_PROMPT_TOKEN.repeat(max_chunk_sizing_rows);
    let max_chunk_prompt_tokens = max_chunk_sizing_rows
        .checked_add(1)
        .context("real-full serving max-chunk prewarm prompt token count overflow")?;
    let max_chunk_sequence_id = format!(
        "{REAL_FULL_STARTUP_MAX_PREFILL_CHUNK_PREFIX}{max_chunk_prompt_tokens}-sequence-{worker_index}"
    );
    let mut max_chunk_request = ds4rt_api::RealFullRequest::new_decode_step_for_sequence(
        0,
        &max_chunk_sequence_id,
        &max_chunk_prompt,
        max_chunk_prompt_tokens,
        1,
        Vec::new(),
        0,
        1,
    );
    max_chunk_request.disable_speculation = true;
    let max_chunk_start = Instant::now();
    eprintln!(
        "real_full_startup_prewarm_start worker={} stage=max-prefill-chunk prompt_tokens={} chunk_rows={}",
        worker_index, max_chunk_prompt_tokens, configured_prefill_chunk_tokens,
    );
    let max_chunk_cycle = execute(max_chunk_request)
        .map_err(anyhow::Error::msg)
        .context("capturing the configured maximum prefill chunk")?;
    let max_chunk_info = max_chunk_cycle.info;
    let expected_max_chunk_waves = scheduler_prefill_chunk_count_for_rows(
        max_chunk_sizing_rows,
        configured_prefill_chunk_tokens,
        1,
    );
    anyhow::ensure!(
        max_chunk_info.status == "ready"
            && max_chunk_info.request_prefill_tokens == max_chunk_sizing_rows
            && max_chunk_info.request_prefill_chunks == expected_max_chunk_waves,
        "maximum prefill-chunk startup sizing failed: status={} prefill_rows={} chunks={} expected_rows={} expected_chunks={} blocker={} failed={:?}",
        max_chunk_info.status,
        max_chunk_info.request_prefill_tokens,
        max_chunk_info.request_prefill_chunks,
        max_chunk_sizing_rows,
        expected_max_chunk_waves,
        max_chunk_info.blocker,
        max_chunk_info.failed_requirements,
    );
    eprintln!(
        "real_full_startup_prewarm_step_done worker={} stage=max-prefill-chunk prompt_tokens={} elapsed_ms={:.3} total_ms={:.3} expert_batches={} expert_rows={} graph_captures={} captured_graphs={}",
        worker_index,
        max_chunk_prompt_tokens,
        elapsed_ms(max_chunk_start),
        elapsed_ms(start),
        max_chunk_info.sparse_expert_batches,
        max_chunk_info.request_expert_batch_rows,
        max_chunk_info.request_coordinator_graph_captures,
        max_chunk_info.request_coordinator_graph_captured_graphs,
    );
    finish_sequence(&max_chunk_sequence_id)
        .map_err(anyhow::Error::msg)
        .context("finishing the maximum prefill-chunk startup sequence")?;

    // Exercise both the initial and continuation lifecycles at the production
    // four-lane 1K geometry without replaying the historical four-chunk 4K
    // request. The 2K max-width request above and this 2K canonical request
    // retain the same serving geometries with two fewer full sparse waves.
    let canonical_chunk_tokens = configured_prefill_chunk_tokens
        .min(REAL_FULL_STARTUP_CANONICAL_PREFILL_CHUNK_TOKENS)
        .max(1);
    if canonical_chunk_tokens < configured_prefill_chunk_tokens {
        let canonical_sizing_rows = canonical_chunk_tokens
            .checked_mul(2)
            .filter(|rows| *rows <= max_prefill_rows)
            .unwrap_or(canonical_chunk_tokens);
        let canonical_prompt = REAL_FULL_SERVE_PREWARM_PROMPT_TOKEN.repeat(canonical_sizing_rows);
        let canonical_prompt_tokens = canonical_sizing_rows
            .checked_add(1)
            .context("real-full serving canonical-chunk prewarm token count overflow")?;
        let canonical_sequence_id = format!(
            "{REAL_FULL_STARTUP_CANONICAL_PREFILL_CHUNK_PREFIX}{canonical_sizing_rows}-sequence-{worker_index}"
        );
        let mut canonical_request = ds4rt_api::RealFullRequest::new_decode_step_for_sequence(
            0,
            &canonical_sequence_id,
            &canonical_prompt,
            canonical_prompt_tokens,
            1,
            Vec::new(),
            0,
            1,
        );
        canonical_request.disable_speculation = true;
        let canonical_start = Instant::now();
        eprintln!(
            "real_full_startup_prewarm_start worker={} stage=canonical-prefill-chunk prompt_tokens={} chunk_rows={}",
            worker_index, canonical_prompt_tokens, canonical_chunk_tokens,
        );
        let canonical_cycle = execute(canonical_request)
            .map_err(anyhow::Error::msg)
            .context("capturing the canonical prefill chunk")?;
        let canonical_info = canonical_cycle.info;
        let expected_canonical_waves = scheduler_prefill_chunk_count_for_rows(
            canonical_sizing_rows,
            canonical_chunk_tokens,
            1,
        );
        anyhow::ensure!(
            canonical_info.status == "ready"
                && canonical_info.request_prefill_tokens == canonical_sizing_rows
                && canonical_info.request_prefill_chunks == expected_canonical_waves,
            "canonical prefill-chunk startup sizing failed: status={} prefill_rows={} chunks={} expected_rows={} expected_chunks={} blocker={} failed={:?}",
            canonical_info.status,
            canonical_info.request_prefill_tokens,
            canonical_info.request_prefill_chunks,
            canonical_sizing_rows,
            expected_canonical_waves,
            canonical_info.blocker,
            canonical_info.failed_requirements,
        );
        eprintln!(
            "real_full_startup_prewarm_step_done worker={} stage=canonical-prefill-chunk prompt_tokens={} elapsed_ms={:.3} total_ms={:.3} expert_batches={} expert_rows={} graph_captures={} captured_graphs={}",
            worker_index,
            canonical_prompt_tokens,
            elapsed_ms(canonical_start),
            elapsed_ms(start),
            canonical_info.sparse_expert_batches,
            canonical_info.request_expert_batch_rows,
            canonical_info.request_coordinator_graph_captures,
            canonical_info.request_coordinator_graph_captured_graphs,
        );
        finish_sequence(&canonical_sequence_id)
            .map_err(anyhow::Error::msg)
            .context("finishing the canonical prefill-chunk startup sequence")?;
    }
    for (prompt_index, (prompt, prompt_tokens)) in prompts.iter().enumerate() {
        // The original prewarm ran every bucket twice: the first request grew
        // workspaces and the second recaptured graphs against the stable
        // pointers. The canonical-arena sweep below now performs that final
        // capture after every other startup sizing operation, so retaining the
        // historical middle traversal only creates graphs that are replaced.
        // In the ordinary dSpark path, bind every sizing request to the same
        // max-context arena used by production. With no intervening prefix
        // probes, these exact packed-KV identities are already the final
        // serving set and the historical verification sweep is pure replay.
        // Optional probes keep the conservative two-stage layout.
        let stage = "workspace-sizing";
        let stage_start = Instant::now();
        let canonical_sizing_request = canonical_workspace_complete || prompt_index == 0;
        let startup_radix_publish_tokens = (canonical_workspace_complete
            && prompt_index == 0
            && !dsa_selector_query_rows.is_empty())
        .then_some(prompt_tokens.saturating_sub(1));
        let sequence_id = if let Some(publish_tokens) = startup_radix_publish_tokens {
            format!(
                "{REAL_FULL_STARTUP_TARGET_RADIX_PUBLISH_PREFIX}{publish_tokens}-sequence-{worker_index}"
            )
        } else if canonical_sizing_request {
            format!("real-full-startup-capture-arena-{prompt_tokens}-sequence-{worker_index}")
        } else {
            format!("real-full-startup-prewarm-{stage}-{prompt_tokens}-sequence-{worker_index}")
        };
        let recurrent_candidate = if canonical_workspace_complete {
            prompt_index + 1 == prompts.len()
        } else {
            prompt_index == 0
        };
        let decode_budget = if recurrent_candidate {
            REAL_FULL_SERVE_PREWARM_DECODE_BUDGET
        } else {
            1
        };
        if canonical_workspace_complete
            && !dsa_selector_query_rows.is_empty()
            && *prompt_tokens == REAL_FULL_SERVE_NO_SELECTOR_DSA_BOUNDARY_PROMPT_TOKENS
        {
            let cached_prompt_tokens = prompt_tokens
                .checked_sub(REAL_FULL_SERVE_NO_SELECTOR_DSA_BOUNDARY_QUERY_ROWS + 1)
                .context("no-selector DSA boundary cached prefix underflow")?;
            anyhow::ensure!(
                cached_prompt_tokens % REAL_FULL_SHARED_KV_PAGE_TOKENS == 0,
                "no-selector DSA boundary prefix {cached_prompt_tokens} is not page aligned"
            );
            // Branch from the long alpha radix seed at exactly the requested
            // cached frontier. Leaving the whole prompt as alpha would match
            // all 2,048 reusable tokens, while declaring a cached prefix on a
            // fresh sequence would bypass radix binding and leave its
            // processed-token frontier at zero.
            let boundary_prompt = format!(
                "{}{}",
                REAL_FULL_SERVE_PREWARM_PROMPT_TOKEN.repeat(cached_prompt_tokens),
                REAL_FULL_SERVE_PREWARM_BOUNDARY_TOKEN
                    .repeat(REAL_FULL_SERVE_NO_SELECTOR_DSA_BOUNDARY_QUERY_ROWS),
            );
            let sequence_id = format!(
                "real-full-startup-dsa-selector-seed-no-selector-boundary-{prompt_tokens}-sequence-{worker_index}"
            );
            let mut request = ds4rt_api::RealFullRequest::new_decode_step_for_sequence(
                0,
                &sequence_id,
                &boundary_prompt,
                *prompt_tokens,
                1,
                Vec::new(),
                0,
                1,
            );
            request.disable_speculation = true;
            eprintln!(
                "real_full_startup_prewarm_start worker={} stage=workspace-sizing-cached-boundary prompt_tokens={} cached_prompt_tokens={} query_rows={}",
                worker_index,
                prompt_tokens,
                cached_prompt_tokens,
                REAL_FULL_SERVE_NO_SELECTOR_DSA_BOUNDARY_QUERY_ROWS,
            );
            let cycle = execute(request)
                .map_err(anyhow::Error::msg)
                .context("capturing the cached no-selector DSA boundary")?;
            let info = cycle.info;
            let expected_prefill_chunks = scheduler_prefill_chunk_count_for_rows(
                REAL_FULL_SERVE_NO_SELECTOR_DSA_BOUNDARY_QUERY_ROWS,
                REAL_FULL_SERVE_NO_SELECTOR_DSA_BOUNDARY_QUERY_ROWS,
                1,
            );
            anyhow::ensure!(
                info.status == "ready"
                    && info.request_prefill_tokens
                        == REAL_FULL_SERVE_NO_SELECTOR_DSA_BOUNDARY_QUERY_ROWS
                    && info.request_prefill_chunks == expected_prefill_chunks,
                "cached no-selector DSA boundary failed: status={} prefill_rows={} chunks={} expected_rows={} expected_chunks={} blocker={} failed={:?}",
                info.status,
                info.request_prefill_tokens,
                info.request_prefill_chunks,
                REAL_FULL_SERVE_NO_SELECTOR_DSA_BOUNDARY_QUERY_ROWS,
                expected_prefill_chunks,
                info.blocker,
                info.failed_requirements,
            );
            eprintln!(
                "real_full_startup_prewarm_step_done worker={} stage=workspace-sizing-cached-boundary prompt_tokens={} cached_prompt_tokens={} elapsed_ms={:.3} total_ms={:.3} expert_batches={} expert_rows={} graph_captures={} captured_graphs={}",
                worker_index,
                prompt_tokens,
                cached_prompt_tokens,
                elapsed_ms(stage_start),
                elapsed_ms(start),
                info.sparse_expert_batches,
                info.request_expert_batch_rows,
                info.request_coordinator_graph_captures,
                info.request_coordinator_graph_captured_graphs,
            );
            finish_sequence(&sequence_id)
                .map_err(anyhow::Error::msg)
                .context("finishing the cached no-selector DSA boundary sequence")?;
            continue;
        }
        let request = ds4rt_api::RealFullRequest::new_decode_step_for_sequence(
            0,
            &sequence_id,
            prompt,
            *prompt_tokens,
            1,
            Vec::new(),
            0,
            decode_budget,
        );
        eprintln!(
            "real_full_startup_prewarm_start worker={} stage={} prompt_tokens={} max_tokens=1 decode_budget={}",
            worker_index, stage, prompt_tokens, decode_budget
        );
        let cycle = execute(request)
            .map_err(|err| anyhow::anyhow!(err))
            .with_context(|| {
                format!("executing real-full serving startup {stage} prewarm request")
            })?;
        let info = cycle.info;
        eprintln!(
            "real_full_startup_prewarm_step_done worker={} stage={} prompt_tokens={} elapsed_ms={:.3} total_ms={:.3} status={} sample_status={} sampled_token_id={:?} full_context_device_attention_complete={} sparse_dispatch_status={} prefill_tokens={} prefill_chunks={} expert_batches={} expert_rows={} graph_captures={} graph_launches={} captured_graphs={}",
            worker_index, stage, prompt_tokens, elapsed_ms(stage_start), elapsed_ms(start),
            info.status,
            info.scheduler_terminal_lm_head_sample_status,
            info.scheduler_terminal_lm_head_sampled_token_id,
            info.scheduler_full_context_device_attention_complete,
            info.scheduler_sparse_tcp_dispatch_status,
            info.request_prefill_tokens,
            info.request_prefill_chunks,
            info.sparse_expert_batches,
            info.request_expert_batch_rows,
            info.request_coordinator_graph_captures,
            info.request_coordinator_graph_launches,
            info.request_coordinator_graph_captured_graphs,
        );
        if info.status != "ready" {
            anyhow::bail!(
                "real-full serving startup {stage} prewarm did not produce a ready scheduler sample: status={} sample_status={} blocker={} failed={:?}",
                info.status,
                info.scheduler_terminal_lm_head_sample_status,
                info.blocker,
                info.failed_requirements
            );
        }
        if startup_radix_publish_tokens.is_some() {
            finish_sequence(&sequence_id)
                .map_err(anyhow::Error::msg)
                .context("publishing long-context startup target KV radix seed")?;
        }
        if recurrent_candidate {
            anyhow::ensure!(
                startup_radix_publish_tokens.is_none(),
                "startup recurrent seed cannot also publish the long-context target radix seed"
            );
            recurrent_seed = Some((sequence_id, prompt, *prompt_tokens, info));
        }
    }
    let (sequence_id, prompt, prompt_tokens, info) = recurrent_seed
        .context("real-full serving startup recurrent seed prewarm was not executed")?;
    let sampled_token_id = info
        .scheduler_terminal_lm_head_sampled_token_id
        .context("real-full serving startup recurrent seed produced no sampled token")?;
    let recurrent_start = Instant::now();
    let recurrent = ds4rt_api::RealFullRequest::new_decode_step_for_sequence(
        u64::try_from(prompts.len()).context("recurrent prewarm prompt count exceeds u64")?,
        &sequence_id,
        prompt,
        prompt_tokens,
        1,
        vec![sampled_token_id],
        1,
        REAL_FULL_SERVE_PREWARM_DECODE_BUDGET,
    );
    eprintln!(
        "real_full_startup_prewarm_start worker={} stage=recurrent prompt_tokens={} generated_tokens=1 max_tokens=1 decode_budget={}",
        worker_index, prompt_tokens,
        REAL_FULL_SERVE_PREWARM_DECODE_BUDGET
    );
    let recurrent_info = execute(recurrent)
        .map_err(|err| anyhow::anyhow!(err))
        .context("executing real-full serving startup recurrent prewarm request")?
        .info;
    eprintln!(
        "real_full_startup_prewarm_step_done worker={} stage=recurrent elapsed_ms={:.3} total_ms={:.3} status={} sample_status={} sampled_token_id={:?} full_context_device_attention_complete={} sparse_dispatch_status={}",
        worker_index, elapsed_ms(recurrent_start),
        elapsed_ms(start),
        recurrent_info.status,
        recurrent_info.scheduler_terminal_lm_head_sample_status,
        recurrent_info.scheduler_terminal_lm_head_sampled_token_id,
        recurrent_info.scheduler_full_context_device_attention_complete,
        recurrent_info.scheduler_sparse_tcp_dispatch_status
    );
    if recurrent_info.status != "ready" {
        anyhow::bail!(
            "real-full serving startup recurrent prewarm did not produce a ready scheduler sample: status={} sample_status={} blocker={} failed={:?}",
            recurrent_info.status,
            recurrent_info.scheduler_terminal_lm_head_sample_status,
            recurrent_info.blocker,
            recurrent_info.failed_requirements
        );
    }
    if let Some(probe) = prefix_prefill_probe {
        let worker_request_stride = probe
            .cases
            .len()
            .checked_mul(MAX_REAL_FULL_SERVE_PREFIX_PREFILL_PROBE_REPEATS)
            .context("real-full serving prefix-prefill worker request stride overflow")?;
        let worker_request_base = worker_index
            .checked_mul(worker_request_stride)
            .context("real-full serving prefix-prefill worker request base overflow")?;
        for (case_index, case) in probe.cases.iter().enumerate() {
            let seed_decode_budget = case
                .new_prompt_rows
                .checked_add(1)
                .context("real-full serving prefix-prefill probe decode budget overflow")?;
            let suffix_prompt = REAL_FULL_SERVE_PREWARM_PROMPT_TOKEN.repeat(case.new_prompt_rows);
            let full_prompt = format!("{}{suffix_prompt}", case.prefix_prompt);
            let full_prompt_tokens = case
                .prefix_prompt_tokens
                .checked_add(case.new_prompt_rows)
                .context("real-full serving prefix-prefill probe token count overflow")?;
            let case_request_base = case_index
                .checked_mul(MAX_REAL_FULL_SERVE_PREFIX_PREFILL_PROBE_REPEATS)
                .and_then(|offset| worker_request_base.checked_add(offset))
                .context("real-full serving prefix-prefill case request base overflow")?;
            let mut probe_elapsed_samples = Vec::with_capacity(probe.repeats);
            for repeat_index in 0..probe.repeats {
                let request_offset = case_request_base
                    .checked_add(repeat_index)
                    .context("real-full serving prefix-prefill request offset overflow")?;
                let request_offset = u64::try_from(request_offset)
                    .context("real-full serving prefix-prefill request offset exceeds u64")?;
                let probe_sequence_id = format!(
                    "real-full-startup-prefix-prefill-seed-{}-{}-repeat-{repeat_index}-sequence-{worker_index}",
                    case.prefix_prompt_tokens, case.new_prompt_rows
                );
                let seed_start = Instant::now();
                eprintln!(
                    "real_full_startup_prewarm_start worker={} stage=prefix-prefill-seed repeat={} prefix_tokens={} new_prompt_tokens={} decode_budget={}",
                    worker_index,
                    repeat_index,
                    case.prefix_prompt_tokens,
                    case.new_prompt_rows,
                    seed_decode_budget
                );
                let seed_info = execute(ds4rt_api::RealFullRequest::new_decode_step_for_sequence(
                    20_000_u64
                        .checked_add(request_offset)
                        .context("real-full serving prefix-prefill seed request index overflow")?,
                    &probe_sequence_id,
                    &case.prefix_prompt,
                    case.prefix_prompt_tokens,
                    1,
                    Vec::new(),
                    0,
                    seed_decode_budget,
                ))
                .map_err(|err| anyhow::anyhow!(err))
                .with_context(|| {
                    format!("executing real-full serving prefix-prefill seed {repeat_index}")
                })?
                .info;
                anyhow::ensure!(
                    seed_info.status == "ready",
                    "real-full serving prefix-prefill seed {repeat_index} was not ready: status={} blocker={} failed={:?}",
                    seed_info.status,
                    seed_info.blocker,
                    seed_info.failed_requirements
                );
                eprintln!(
                    "real_full_startup_prewarm_step_done worker={} stage=prefix-prefill-seed repeat={} prefix_tokens={} elapsed_ms={:.3} total_ms={:.3}",
                    worker_index,
                    repeat_index,
                    case.prefix_prompt_tokens,
                    elapsed_ms(seed_start),
                    elapsed_ms(start)
                );
                let request_index = 10_000_u64
                    .checked_add(request_offset)
                    .context("real-full serving prefix-prefill probe request index overflow")?;
                let probe_start = Instant::now();
                eprintln!(
                    "real_full_startup_prewarm_start worker={} stage=prefix-prefill repeat={} prefix_tokens={} new_prompt_tokens={} full_prompt_tokens={}",
                    worker_index,
                    repeat_index,
                    case.prefix_prompt_tokens,
                    case.new_prompt_rows,
                    full_prompt_tokens
                );
                let probe_info = execute(
                    ds4rt_api::RealFullRequest::new_decode_step_for_sequence(
                        request_index,
                        &probe_sequence_id,
                        &full_prompt,
                        full_prompt_tokens,
                        1,
                        Vec::new(),
                        1,
                        REAL_FULL_SERVE_PREWARM_DECODE_BUDGET,
                    )
                    .with_cached_prompt_tokens(case.prefix_prompt_tokens),
                )
                .map_err(|err| anyhow::anyhow!(err))
                .with_context(|| {
                    format!("executing real-full serving prefix-prefill probe {repeat_index}")
                })?
                .info;
                let probe_elapsed_ms = elapsed_ms(probe_start);
                anyhow::ensure!(
                    probe_info.status == "ready",
                    "real-full serving prefix-prefill probe {repeat_index} was not ready: status={} sample_status={} blocker={} failed={:?}",
                    probe_info.status,
                    probe_info.scheduler_terminal_lm_head_sample_status,
                    probe_info.blocker,
                    probe_info.failed_requirements
                );
                anyhow::ensure!(
                    probe_info.request_prefill_tokens + 1 == case.new_prompt_rows,
                    "real-full serving prefix-prefill probe {repeat_index} reported {} prefill rows for {} new prompt tokens",
                    probe_info.request_prefill_tokens,
                    case.new_prompt_rows
                );
                eprintln!(
                    "real_full_startup_prewarm_step_done worker={} stage=prefix-prefill repeat={} prefix_tokens={} new_prompt_tokens={} prefill_rows={} elapsed_ms={:.3} tokens_per_sec={:.3} total_ms={:.3} status={} sample_status={} sampled_token_id={:?}",
                    worker_index,
                    repeat_index,
                    case.prefix_prompt_tokens,
                    case.new_prompt_rows,
                    probe_info.request_prefill_tokens,
                    probe_elapsed_ms,
                    case.new_prompt_rows as f64 * 1_000.0 / probe_elapsed_ms,
                    elapsed_ms(start),
                    probe_info.status,
                    probe_info.scheduler_terminal_lm_head_sample_status,
                    probe_info.scheduler_terminal_lm_head_sampled_token_id,
                );
                probe_elapsed_samples.push(probe_elapsed_ms);
            }
            probe_elapsed_samples.sort_by(f64::total_cmp);
            let sample_midpoint = probe_elapsed_samples.len() / 2;
            let median_elapsed_ms = if probe_elapsed_samples.len() % 2 == 0 {
                (probe_elapsed_samples[sample_midpoint - 1]
                    + probe_elapsed_samples[sample_midpoint])
                    / 2.0
            } else {
                probe_elapsed_samples[sample_midpoint]
            };
            eprintln!(
                "real_full_startup_prefix_prefill_summary worker={} repeats={} prefix_tokens={} new_prompt_tokens={} median_elapsed_ms={:.3} median_tokens_per_sec={:.3} min_elapsed_ms={:.3} max_elapsed_ms={:.3}",
                worker_index,
                probe_elapsed_samples.len(),
                case.prefix_prompt_tokens,
                case.new_prompt_rows,
                median_elapsed_ms,
                case.new_prompt_rows as f64 * 1_000.0 / median_elapsed_ms,
                probe_elapsed_samples[0],
                probe_elapsed_samples[probe_elapsed_samples.len() - 1]
            );
        }
    }
    // Every direct packed-KV attention graph captures the physical cache and
    // query-arena addresses. Ordinary dSpark sizing already uses the canonical
    // max-context arena and no later optional probe can replace those C=1
    // identities, so a second sweep would only replay the same graphs. Prefix
    // probes can still perturb the capture set; retain the conservative final
    // sweep for them and run its largest prompt last. The
    // selector seed is created separately after dSpark width prewarm because
    // those graph-bound requests intentionally recycle the one max-context
    // arena; retaining a seed here would let width prewarm rebind its state
    // while the selector sweep still believed the old sequence was active.
    if canonical_workspace_complete {
        eprintln!(
            "real_full_startup_prewarm_step_done worker={} stage=canonical-capture-skip reason=canonical-workspace-complete elapsed_ms=0.000 total_ms={:.3}",
            worker_index,
            elapsed_ms(start),
        );
    } else {
        for (prompt_index, (prompt, prompt_tokens)) in prompts.iter().rev().enumerate() {
            let sequence_id = format!(
                "real-full-startup-capture-arena-final-{prompt_tokens}-sequence-{worker_index}"
            );
            let decode_budget = 1;
            let capture_start = Instant::now();
            eprintln!(
                "real_full_startup_prewarm_start worker={} stage=canonical-capture prompt_tokens={} max_tokens=1 decode_budget={}",
                worker_index, prompt_tokens, decode_budget
            );
            let capture_info = execute(ds4rt_api::RealFullRequest::new_decode_step_for_sequence(
                u64::try_from(prompt_index)
                    .context("canonical capture prompt index exceeds u64")?,
                &sequence_id,
                prompt,
                *prompt_tokens,
                1,
                Vec::new(),
                0,
                decode_budget,
            ))
            .map_err(|err| anyhow::anyhow!(err))
            .with_context(|| {
                format!("executing canonical max-context capture for {prompt_tokens} prompt tokens")
            })?
            .info;
            anyhow::ensure!(
                capture_info.status == "ready",
                "canonical max-context capture for {prompt_tokens} prompt tokens was not ready: status={} sample_status={} blocker={} failed={:?}",
                capture_info.status,
                capture_info.scheduler_terminal_lm_head_sample_status,
                capture_info.blocker,
                capture_info.failed_requirements
            );
            eprintln!(
                "real_full_startup_prewarm_step_done worker={} stage=canonical-capture prompt_tokens={} elapsed_ms={:.3} total_ms={:.3} sampled_token_id={:?}",
                worker_index,
                prompt_tokens,
                elapsed_ms(capture_start),
                elapsed_ms(start),
                capture_info.scheduler_terminal_lm_head_sampled_token_id,
            );
        }
    }
    // Grow the shared multirow workspaces before capturing long-context DSA
    // identities. The dSpark M=2..8 sweep can enlarge bucket-8 scratch; if it
    // runs afterward, the pointer change correctly clears the earlier DSA
    // graphs and leaves the first >2K serving request unable to recapture.
    prewarm_real_full_dspark_widths(
        &mut execute,
        &mut finish_sequence,
        prompts,
        worker_index,
        start,
        None,
    )?;
    let dsa_selector_seed = if dsa_selector_query_rows.is_empty() {
        None
    } else {
        let (prompt, prompt_tokens) = real_full_serve_dsa_selector_seed_prompt(
            prompts,
            &dsa_selector_query_rows,
            dsa_selector_decode_budget,
            max_context_tokens,
        )?;
        let sequence_id =
            format!("real-full-startup-dsa-selector-seed-{prompt_tokens}-sequence-{worker_index}");
        let seed_start = Instant::now();
        eprintln!(
            "real_full_startup_prewarm_start worker={} stage=dsa-selector-seed prompt_tokens={} decode_budget={}",
            worker_index, prompt_tokens, dsa_selector_decode_budget,
        );
        let mut seed_request = ds4rt_api::RealFullRequest::new_decode_step_for_sequence(
            29_999,
            &sequence_id,
            prompt,
            *prompt_tokens,
            1,
            Vec::new(),
            0,
            dsa_selector_decode_budget,
        );
        // This pass captures target-attention selector buckets. It does not
        // need a draft proposal, and target-only admission lets it reuse the
        // long prefix produced by workspace sizing.
        seed_request.disable_speculation = true;
        let seed_info = execute(seed_request)
            .map_err(|err| anyhow::anyhow!(err))
            .context("executing long-context DSA selector seed")?
            .info;
        anyhow::ensure!(
            seed_info.status == "ready",
            "long-context DSA selector seed was not ready: status={} blocker={} failed={:?}",
            seed_info.status,
            seed_info.blocker,
            seed_info.failed_requirements,
        );
        eprintln!(
            "real_full_startup_prewarm_step_done worker={} stage=dsa-selector-seed prompt_tokens={} elapsed_ms={:.3} total_ms={:.3} sampled_token_id={:?}",
            worker_index,
            prompt_tokens,
            elapsed_ms(seed_start),
            elapsed_ms(start),
            seed_info.scheduler_terminal_lm_head_sampled_token_id,
        );
        Some((sequence_id, prompt.clone(), *prompt_tokens))
    };
    if let Some((sequence_id, mut prompt, mut prompt_tokens)) = dsa_selector_seed {
        for (step_index, query_rows) in dsa_selector_query_rows.iter().copied().enumerate() {
            let cached_prompt_tokens = prompt_tokens;
            let uncached_prompt_rows = query_rows + 1;
            prompt.push_str(&REAL_FULL_SERVE_PREWARM_PROMPT_TOKEN.repeat(uncached_prompt_rows));
            prompt_tokens = prompt_tokens
                .checked_add(uncached_prompt_rows)
                .context("DSA selector prewarm prompt token count overflow")?;
            let sweep_start = Instant::now();
            eprintln!(
                "real_full_startup_prewarm_start worker={} stage=dsa-selector-bucket query_rows={} prefix_tokens={} prompt_tokens={} decode_budget={}",
                worker_index,
                query_rows,
                cached_prompt_tokens,
                prompt_tokens,
                dsa_selector_decode_budget,
            );
            let mut sweep_request = ds4rt_api::RealFullRequest::new_decode_step_for_sequence(
                30_000_u64
                    .checked_add(
                        u64::try_from(step_index)
                            .context("DSA selector prewarm step index exceeds u64")?,
                    )
                    .context("DSA selector prewarm request index overflow")?,
                &sequence_id,
                &prompt,
                prompt_tokens,
                1,
                Vec::new(),
                step_index + 1,
                dsa_selector_decode_budget,
            )
            .with_cached_prompt_tokens(cached_prompt_tokens);
            sweep_request.disable_speculation = true;
            let sweep_info = execute(sweep_request)
                .map_err(|err| anyhow::anyhow!(err))
                .with_context(|| {
                    format!("capturing long-context DSA selector query bucket {query_rows}")
                })?
                .info;
            anyhow::ensure!(
                sweep_info.status == "ready"
                    && sweep_info.request_prefill_tokens == query_rows,
                "long-context DSA selector query bucket {query_rows} was not ready: status={} prefill_rows={} blocker={} failed={:?}",
                sweep_info.status,
                sweep_info.request_prefill_tokens,
                sweep_info.blocker,
                sweep_info.failed_requirements,
            );
            eprintln!(
                "real_full_startup_prewarm_step_done worker={} stage=dsa-selector-bucket query_rows={} elapsed_ms={:.3} total_ms={:.3} sampled_token_id={:?}",
                worker_index,
                query_rows,
                elapsed_ms(sweep_start),
                elapsed_ms(start),
                sweep_info.scheduler_terminal_lm_head_sampled_token_id,
            );
        }
        if canonical_workspace_complete {
            let radix_prefix_tokens = prompts
                .first()
                .map(|(_, prompt_tokens)| prompt_tokens.saturating_sub(1))
                .context("real-full serving radix cleanup prompt set is empty")?;
            let boundary_prefix_tokens = REAL_FULL_SERVE_NO_SELECTOR_DSA_BOUNDARY_PROMPT_TOKENS
                .checked_sub(REAL_FULL_SERVE_NO_SELECTOR_DSA_BOUNDARY_QUERY_ROWS + 1)
                .context("no-selector DSA boundary cleanup prefix underflow")?;
            // The cached no-selector pass splits the long alpha tree at its
            // 1,536-token branch point. Evict the longest suffix first, then
            // remove the complete synthetic branch subtree. With a larger KV
            // pool the selector sweep remains below that branch instead of
            // being displaced by LRU pressure.
            for eviction_tokens in [radix_prefix_tokens, boundary_prefix_tokens] {
                let eviction_sequence_id = format!(
                    "{REAL_FULL_STARTUP_TARGET_RADIX_EVICT_PREFIX}{eviction_tokens}-worker-{worker_index}"
                );
                finish_sequence(&eviction_sequence_id)
                    .map_err(anyhow::Error::msg)
                    .with_context(|| {
                        format!(
                            "evicting {eviction_tokens}-token synthetic startup target KV radix subtree"
                        )
                    })?;
            }
        }
    }
    // The first width pass and target-selector sweep establish the union of
    // required scratch capacities. Exercise one width from each remaining
    // physical query bucket after scratch reaches its final size. The first
    // width pass already captures all other exact-width kernels.
    if real_full_dspark_enabled() {
        let required_attention_prompt_tokens = &[2_049];
        let attention_prompts = required_attention_prompt_tokens
            .into_iter()
            .map(|required_prompt_tokens| {
                prompts
                    .iter()
                    .find(|(_, prompt_tokens)| prompt_tokens == required_prompt_tokens)
                    .cloned()
                    .with_context(|| {
                        format!(
                            "real-full serving dSpark prewarm has no canonical {required_prompt_tokens}-token prompt"
                        )
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let checkpoint_max_drafts = dspark_active_max_verify_drafts();
        let maximum_drafts = match real_full_dspark_fixed_drafts()? {
            Some(drafts) => {
                anyhow::ensure!(
                    drafts <= checkpoint_max_drafts,
                    "fixed dSpark width {drafts} exceeds the active checkpoint maximum {checkpoint_max_drafts}"
                );
                drafts
            }
            None => checkpoint_max_drafts,
        };
        let mut attention_query_bucket_drafts = Vec::new();
        let query_bucket_representatives = [maximum_drafts, 7, 0];
        for representative_drafts in query_bucket_representatives {
            if representative_drafts <= maximum_drafts
                && !attention_query_bucket_drafts.contains(&representative_drafts)
            {
                attention_query_bucket_drafts.push(representative_drafts);
            }
        }
        for attention_prompt in &attention_prompts {
            prewarm_real_full_dspark_widths(
                &mut execute,
                &mut finish_sequence,
                std::slice::from_ref(attention_prompt),
                worker_index,
                start,
                Some(&attention_query_bucket_drafts),
            )?;
        }
    }
    eprintln!(
        "real_full_startup_prewarm_done worker={} elapsed_ms={:.3} initial_sampled_token_id={:?} recurrent_sampled_token_id={:?}",
        worker_index, elapsed_ms(start),
        info.scheduler_terminal_lm_head_sampled_token_id,
        recurrent_info.scheduler_terminal_lm_head_sampled_token_id
    );
    Ok(())
}

fn finish_real_full_dspark_width_prewarm_sequence(
    finish_sequence: &mut impl FnMut(&str) -> std::result::Result<(), String>,
    sequence_id: &str,
    physical_m: usize,
) -> Result<()> {
    finish_sequence(sequence_id)
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("finishing dSpark M={physical_m} startup capture sequence"))
}

#[allow(clippy::too_many_arguments)]
fn prewarm_real_full_dspark_width_cohort(
    execute: &mut impl FnMut(
        ds4rt_api::RealFullRequest,
    ) -> std::result::Result<ds4rt_api::RealFullDecodeCycle, String>,
    finish_sequence: &mut impl FnMut(&str) -> std::result::Result<(), String>,
    dspark_prompt: &str,
    dspark_prompt_tokens: usize,
    draft_widths: &[usize],
    worker_index: usize,
    prewarm_start: Instant,
) -> Result<()> {
    let maximum_drafts = draft_widths
        .iter()
        .copied()
        .max()
        .context("dSpark scalar width cohort is empty")?;
    let sequence_id = format!(
        "real-full-startup-dspark-width-{maximum_drafts}{REAL_FULL_STARTUP_SCALAR_DSPARK_COHORT_MARKER}{dspark_prompt_tokens}-sequence-{worker_index}"
    );
    let capture_budget = draft_widths
        .iter()
        .try_fold(2_usize, |budget, draft_tokens| {
            budget
                .checked_add(draft_tokens.saturating_add(1))
                .context("dSpark scalar width cohort decode budget overflow")
        })?;
    let cohort_start = Instant::now();
    eprintln!(
        "real_full_startup_prewarm_start worker={} stage=dspark-width-cohort prompt_tokens={} widths={:?} max_physical_m={} decode_budget={}",
        worker_index,
        dspark_prompt_tokens,
        draft_widths,
        maximum_drafts + 1,
        capture_budget,
    );
    let initial_cycle = execute(ds4rt_api::RealFullRequest::new_decode_step_for_sequence(
        REAL_FULL_SCALAR_DSPARK_PREWARM_WIDTH_REQUEST_BASE - 1,
        &sequence_id,
        dspark_prompt,
        dspark_prompt_tokens,
        1,
        Vec::new(),
        0,
        capture_budget,
    ))
    .map_err(anyhow::Error::msg)
    .context("seeding a reusable dSpark scalar width cohort")?;
    anyhow::ensure!(
        initial_cycle.info.status == "ready",
        "dSpark scalar width cohort seed failed: status={} blocker={} failed={:?}",
        initial_cycle.info.status,
        initial_cycle.info.blocker,
        initial_cycle.info.failed_requirements,
    );
    let mut generated_token_ids = initial_cycle
        .generated_tokens
        .iter()
        .map(|token| token.token_id)
        .collect::<Vec<_>>();
    if generated_token_ids.is_empty() {
        generated_token_ids.push(
            initial_cycle
                .info
                .scheduler_terminal_lm_head_sampled_token_id
                .context("dSpark scalar width cohort seed produced no token")?,
        );
    }
    for (width_index, draft_tokens) in draft_widths.iter().copied().enumerate() {
        let width_start = Instant::now();
        let request_index = REAL_FULL_SCALAR_DSPARK_PREWARM_WIDTH_REQUEST_BASE
            .checked_add(
                u64::try_from(draft_tokens)
                    .context("dSpark scalar cohort width exceeds u64")?
                    .saturating_mul(REAL_FULL_SCALAR_DSPARK_PREWARM_WIDTH_REQUEST_STRIDE),
            )
            .and_then(|index| index.checked_add(u64::try_from(width_index).ok()?))
            .context("dSpark scalar cohort request index overflow")?;
        let decode_step_index = generated_token_ids.len();
        let verify_cycle = execute(ds4rt_api::RealFullRequest::new_decode_step_for_sequence(
            request_index,
            &sequence_id,
            dspark_prompt,
            dspark_prompt_tokens,
            1,
            generated_token_ids.clone(),
            decode_step_index,
            capture_budget,
        ))
        .map_err(anyhow::Error::msg)
        .with_context(|| {
            format!(
                "executing dSpark M={} scalar cohort capture",
                draft_tokens + 1
            )
        })?;
        anyhow::ensure!(
            verify_cycle.info.status == "ready"
                && verify_cycle.info.request_mtp_verify_rows == draft_tokens,
            "dSpark M={} scalar cohort capture failed: status={} verify_rows={} expected_rows={} blocker={} failed={:?}",
            draft_tokens + 1,
            verify_cycle.info.status,
            verify_cycle.info.request_mtp_verify_rows,
            draft_tokens,
            verify_cycle.info.blocker,
            verify_cycle.info.failed_requirements,
        );
        anyhow::ensure!(
            !verify_cycle.generated_tokens.is_empty(),
            "dSpark M={} scalar cohort capture emitted no continuation token",
            draft_tokens + 1,
        );
        generated_token_ids.extend(
            verify_cycle
                .generated_tokens
                .iter()
                .map(|token| token.token_id),
        );
        eprintln!(
            "real_full_startup_prewarm_step_done worker={} stage=dspark-width-cohort prompt_tokens={} drafts={} physical_m={} verify_rows={} generated_tokens={} elapsed_ms={:.3} total_ms={:.3}",
            worker_index,
            dspark_prompt_tokens,
            draft_tokens,
            draft_tokens + 1,
            verify_cycle.info.request_mtp_verify_rows,
            generated_token_ids.len(),
            elapsed_ms(width_start),
            elapsed_ms(prewarm_start),
        );
    }
    finish_real_full_dspark_width_prewarm_sequence(
        finish_sequence,
        &sequence_id,
        maximum_drafts + 1,
    )?;
    eprintln!(
        "real_full_startup_prewarm_done worker={} stage=dspark-width-cohort prompt_tokens={} widths={:?} elapsed_ms={:.3} total_ms={:.3}",
        worker_index,
        dspark_prompt_tokens,
        draft_widths,
        elapsed_ms(cohort_start),
        elapsed_ms(prewarm_start),
    );
    Ok(())
}

fn prewarm_real_full_dspark_widths(
    execute: &mut impl FnMut(
        ds4rt_api::RealFullRequest,
    ) -> std::result::Result<ds4rt_api::RealFullDecodeCycle, String>,
    finish_sequence: &mut impl FnMut(&str) -> std::result::Result<(), String>,
    prompts: &[(String, usize)],
    worker_index: usize,
    prewarm_start: Instant,
    draft_widths_override: Option<&[usize]>,
) -> Result<()> {
    if !real_full_dspark_enabled() {
        return Ok(());
    }

    // Width identities are keyed by physical rows/layer, not context length.
    // Use the smallest canonical prompt after the base canonical sweep has
    // established the serving arena. Long-context DSA capture runs afterward.
    let (dspark_prompt, dspark_prompt_tokens) = prompts
        .last()
        .context("real-full serving dSpark prewarm prompt set is empty")?;
    // Fixed-width diagnostics can only reach their configured M, so avoid
    // spending every coordinator restart recapturing unused wider widths.
    // Adaptive serving still captures widest-first: several coordinator
    // scratch buffers grow with M, and growing them invalidates every graph
    // identity in the shared slot. Descending widths establish maximum
    // capacity first, then retain all narrower DSA and attention identities.
    let draft_widths = match draft_widths_override {
        Some(draft_widths) => draft_widths.to_vec(),
        None => match real_full_dspark_fixed_drafts()? {
            // The final cycle truncates to the remaining output budget, so a
            // fixed-width request can still reach every narrower M. Include
            // D=0/M=1 explicitly: the adaptive policy can choose target-only
            // decode, and that decode graph must exist before capture closes.
            Some(draft_tokens) => (0..=draft_tokens).rev().collect(),
            None => (0..=dspark_active_max_verify_drafts()).rev().collect(),
        },
    };
    if draft_widths_override.is_some() && draft_widths.len() > 1 {
        return prewarm_real_full_dspark_width_cohort(
            execute,
            finish_sequence,
            dspark_prompt,
            *dspark_prompt_tokens,
            &draft_widths,
            worker_index,
            prewarm_start,
        );
    }
    for draft_tokens in draft_widths {
        let request_index = 30_000 + draft_tokens as u64 * 3;
        let capture_budget = dspark_active_max_verify_drafts() + 4;
        let sequence_id = format!(
            "real-full-startup-dspark-width-{draft_tokens}-{dspark_prompt_tokens}-sequence-{worker_index}"
        );
        let width_start = Instant::now();
        eprintln!(
            "real_full_startup_prewarm_start worker={} stage=dspark-width prompt_tokens={} drafts={} physical_m={}",
            worker_index,
            dspark_prompt_tokens,
            draft_tokens,
            draft_tokens + 1,
        );
        let initial_cycle = execute(ds4rt_api::RealFullRequest::new_decode_step_for_sequence(
            request_index,
            &sequence_id,
            dspark_prompt,
            *dspark_prompt_tokens,
            1,
            Vec::new(),
            0,
            capture_budget,
        ))
        .map_err(|error| anyhow::anyhow!(error))
        .with_context(|| {
            format!(
                "executing dSpark M={} scalar capture seed",
                draft_tokens + 1
            )
        })?;
        anyhow::ensure!(
            initial_cycle.info.status == "ready",
            "dSpark M={} scalar capture seed failed: status={} blocker={} failed={:?}",
            draft_tokens + 1,
            initial_cycle.info.status,
            initial_cycle.info.blocker,
            initial_cycle.info.failed_requirements,
        );
        let initial_token_id = initial_cycle
            .generated_tokens
            .first()
            .map(|token| token.token_id)
            .or(initial_cycle
                .info
                .scheduler_terminal_lm_head_sampled_token_id)
            .with_context(|| {
                format!(
                    "dSpark M={} scalar capture seed has no token",
                    draft_tokens + 1
                )
            })?;
        let verify_cycle = execute(ds4rt_api::RealFullRequest::new_decode_step_for_sequence(
            request_index + 1,
            &sequence_id,
            dspark_prompt,
            *dspark_prompt_tokens,
            1,
            vec![initial_token_id],
            1,
            capture_budget,
        ))
        .map_err(|error| anyhow::anyhow!(error))
        .with_context(|| format!("executing dSpark M={} target capture", draft_tokens + 1))?;
        anyhow::ensure!(
            verify_cycle.info.status == "ready"
                && verify_cycle.info.request_mtp_verify_rows == draft_tokens,
            "dSpark M={} target capture failed: status={} verify_rows={} expected_rows={} blocker={} failed={:?}",
            draft_tokens + 1,
            verify_cycle.info.status,
            verify_cycle.info.request_mtp_verify_rows,
            draft_tokens,
            verify_cycle.info.blocker,
            verify_cycle.info.failed_requirements,
        );
        // Speculative acceptance recomputes final_decode_step from the tokens
        // actually emitted, so the synthetic decode_step_index above does not
        // guarantee that this state was recycled. End every width
        // explicitly before the next width reserves target KV: at C=1 there
        // is no spare reservation slot for arena rebind to drop the old one.
        finish_real_full_dspark_width_prewarm_sequence(
            finish_sequence,
            &sequence_id,
            draft_tokens + 1,
        )?;
        eprintln!(
            "real_full_startup_prewarm_step_done worker={} stage=dspark-width prompt_tokens={} drafts={} physical_m={} verify_rows={} elapsed_ms={:.3} total_ms={:.3}",
            worker_index,
            dspark_prompt_tokens,
            draft_tokens,
            draft_tokens + 1,
            verify_cycle.info.request_mtp_verify_rows,
            elapsed_ms(width_start),
            elapsed_ms(prewarm_start),
        );
    }
    Ok(())
}

fn validate_real_full_strict_tp4_args(args: &CoordinatorArgs) -> Result<()> {
    anyhow::ensure!(
        args.loadplan.is_none(),
        "strict DeepSeek V4 TP=4 serving rejects --loadplan: expert owner maps describe EP placement, while every routed expert must execute on all four Spark ranks"
    );
    Ok(())
}

fn real_full_sparse_tcp_targets_from_args(
    args: &CoordinatorArgs,
) -> Result<Vec<TcpProtocolV2HostBatchTarget>> {
    let dispatch_transport =
        RealFullSchedulerSparseDispatchTransport::from_label(args.transport.as_str())
            .with_context(|| {
                format!(
                    "strict DeepSeek V4 TP4 serving requires tcp, tcp-debug-json, or verbs-host sparse dispatch; transport {:?} cannot execute routed experts",
                    args.transport
                )
            })?;
    if dispatch_transport == RealFullSchedulerSparseDispatchTransport::VerbsHost {
        ds4rt_transport::verbs_host_preflight()
            .context("real-ds4-full verbs-host sparse dispatch RDMA preflight failed")?;
    }
    let entries = args
        .expert_hosts
        .split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .collect::<Vec<_>>();
    if entries.is_empty() {
        bail!(
            "real-ds4-full {} transport requires --expert-hosts",
            dispatch_transport.label()
        );
    }

    if entries.len() == 1 && !entries[0].contains('=') {
        let addr = resolve_real_full_sparse_tcp_addr(entries[0], dispatch_transport.label())?;
        return Ok(EXPERT_HOSTS
            .iter()
            .map(|host| TcpProtocolV2HostBatchTarget {
                host: (*host).to_owned(),
                addr,
            })
            .collect());
    }

    let mut targets = Vec::with_capacity(entries.len());
    for entry in entries {
        let (host, raw_addr) = if let Some((host, raw_addr)) = entry.split_once('=') {
            (host.trim(), raw_addr.trim())
        } else {
            let host = entry.split_once(':').map_or(entry, |(host, _)| host).trim();
            (host, entry)
        };
        if host.is_empty() {
            bail!(
                "real-ds4-full {} expert target {entry:?} has empty host",
                dispatch_transport.label()
            );
        }
        targets.push(TcpProtocolV2HostBatchTarget {
            host: host.to_owned(),
            addr: resolve_real_full_sparse_tcp_addr(raw_addr, dispatch_transport.label())?,
        });
    }

    let missing_hosts = EXPERT_HOSTS
        .iter()
        .filter(|host| !targets.iter().any(|target| target.host.as_str() == **host))
        .copied()
        .collect::<Vec<_>>();
    if !missing_hosts.is_empty() {
        bail!(
            "real-ds4-full {} sparse dispatch is missing expert targets for [{}]; pass host=ip:port entries or a single target to mirror to all expert hosts",
            dispatch_transport.label(),
            missing_hosts.join(",")
        );
    }
    anyhow::ensure!(
        targets.len() == EXPERT_HOSTS.len(),
        "strict DeepSeek V4 TP4 sparse dispatch requires exactly {} named ranks, got {}",
        EXPERT_HOSTS.len(),
        targets.len()
    );
    Ok(targets)
}

fn resolve_real_full_sparse_tcp_addr(raw_target: &str, transport: &str) -> Result<SocketAddr> {
    let raw_target = raw_target.trim();
    if raw_target.is_empty() {
        bail!("real-ds4-full {transport} expert target address is empty");
    }
    let with_port = if raw_target.contains(':') {
        raw_target.to_owned()
    } else {
        format!("{raw_target}:9100")
    };
    with_port
        .to_socket_addrs()
        .with_context(|| format!("resolving real-ds4-full {transport} expert target {with_port}"))?
        .next()
        .ok_or_else(|| {
            anyhow::anyhow!(
                "real-ds4-full {transport} expert target {with_port} resolved to no addresses"
            )
        })
}

fn initialize_sparse_tcp_dispatch_status(
    info: &mut ds4rt_api::RealFullInfo,
    targets: &[TcpProtocolV2HostBatchTarget],
) {
    info.scheduler_sparse_tcp_dispatch_status = "configured-not-run".to_owned();
    info.scheduler_sparse_tcp_dispatch_targets = targets.len();
}

fn scheduler_sparse_tcp_expected_real_executor_id() -> u64 {
    expert_protocol_v2_compact_id(REAL_NVFP4_PROTOCOL_V2_EXECUTOR)
}

fn apply_sparse_tcp_dispatch_probe(
    info: &mut ds4rt_api::RealFullInfo,
    target_count: usize,
    probe: &RealFullSchedulerSparseTcpDispatchProbe,
) {
    info.scheduler_sparse_tcp_dispatch_status = probe.status.to_owned();
    info.scheduler_sparse_tcp_dispatch_targets = target_count;
    info.scheduler_sparse_tcp_dispatch_sparse_layers = probe.sparse_layers;
    info.scheduler_sparse_tcp_dispatch_iterations_per_sparse_layer =
        probe.scheduler_iterations_per_sparse_layer;
    info.scheduler_sparse_tcp_dispatch_batches = probe.sparse_batches;
    info.scheduler_sparse_tcp_dispatch_host_batches = probe.host_batches;
    info.scheduler_sparse_tcp_dispatch_global_rows = probe.global_rows;
    info.scheduler_sparse_tcp_dispatch_host_rows = probe.host_rows;
    info.scheduler_sparse_tcp_dispatch_routes = probe.routes;
    info.scheduler_sparse_tcp_dispatch_request_wire_bytes = probe.request_wire_bytes;
    info.scheduler_sparse_tcp_dispatch_response_wire_bytes = probe.response_wire_bytes;
    info.scheduler_sparse_tcp_dispatch_output_values = probe.output_values;
    info.scheduler_sparse_tcp_dispatch_output_finite_values = probe.output_finite_values;
    info.scheduler_sparse_tcp_dispatch_output_nonzero_values = probe.output_nonzero_values;
    info.scheduler_sparse_tcp_dispatch_output_checksum = probe.output_checksum;
    info.scheduler_sparse_tcp_dispatch_passed = probe.passed;
    info.scheduler_sparse_tcp_dispatch_expected_real_executor_id = probe.expected_real_executor_id;
    info.scheduler_sparse_tcp_dispatch_response_executor_ids_observed =
        probe.response_executor_ids_observed;
    info.scheduler_sparse_tcp_dispatch_real_executor_responses = probe.real_executor_responses;
    info.scheduler_sparse_tcp_dispatch_non_real_executor_responses =
        probe.non_real_executor_responses;
    info.scheduler_sparse_tcp_dispatch_all_responses_real_checkpoint_experts =
        probe.all_responses_real_nvfp4;
    info.scheduler_sparse_tcp_dispatch_all_responses_real_nvfp4 = probe.all_responses_real_nvfp4;
    info.scheduler_sparse_tcp_dispatch_consumed_by_residual =
        sparse_tcp_dispatch_consumed_by_residual(info, probe);
    if !probe.passed {
        info.status = "blocked".to_owned();
        if info.blocker.trim().is_empty() {
            info.blocker = format!("real-full sparse TCP dispatch status={}", probe.status);
        }
        if !info
            .failed_requirements
            .iter()
            .any(|requirement| requirement == "scheduler_sparse_tcp_dispatch")
        {
            info.failed_requirements
                .push("scheduler_sparse_tcp_dispatch".to_owned());
        }
    }
}

fn sparse_tcp_dispatch_consumed_by_residual(
    info: &ds4rt_api::RealFullInfo,
    probe: &RealFullSchedulerSparseTcpDispatchProbe,
) -> bool {
    probe.passed
        && probe.output_values > 0
        && info.scheduler_numeric_progression_passed
        && info.request_numeric_progression_mlp_value_updates > 0
}

fn load_real_ds4_full_catalog(args: &CoordinatorArgs) -> Result<(String, TensorCatalog)> {
    if let Some(catalog_path) = args.catalog.as_deref() {
        let catalog = load_catalog(catalog_path)?;
        anyhow::ensure!(
            catalog.model_id == args.model_id,
            "catalog model_id {} does not match requested {}",
            catalog.model_id,
            args.model_id
        );
        crate::commands::model_artifacts::validate_production_runtime_catalog(&catalog)?;
        return Ok((catalog_path.display().to_string(), catalog));
    }
    let catalog = crate::commands::model_artifacts::build_runtime_catalog(&args.model_id)?;
    Ok((format!("hf://{}", args.model_id), catalog))
}

pub(in crate::commands::real_full) fn real_full_info_from_startup(
    args: &CoordinatorArgs,
    catalog: &TensorCatalog,
    preload: RealFullCoordinatorResidentPreloadPlan,
) -> Result<ds4rt_api::RealFullInfo> {
    let expert_hosts = args
        .expert_hosts
        .split(',')
        .map(str::trim)
        .filter(|target| !target.is_empty())
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let kv_config = real_full_kv_cache_config_for_model(args, &catalog.facts)?;
    let execution_plan =
        real_full_execution_plan(&expert_hosts, kv_config.bytes_per_token(), &catalog.facts);

    Ok(ds4rt_api::RealFullInfo {
        status: "blocked".to_owned(),
        model_id: args.model_id.clone(),
        model_facts: catalog.facts.clone(),
        snapshot_path: Some(catalog.snapshot_path.clone()),
        catalog_hash: catalog.content_hash(),
        quantization_recipe: catalog.facts.quantization_recipe.clone(),
        tensor_count: catalog.tensors.len(),
        startup_diagnostic_mode: "serving-startup-residency-only".to_owned(),
        coordinator_resident_preload_status: preload.status.to_owned(),
        coordinator_resident_preload_selected_tensors: preload.selected_tensor_count,
        coordinator_resident_preload_selected_bytes: preload.selected_tensor_bytes,
        coordinator_resident_preload_loaded_bytes: preload.loaded_tensor_bytes,
        layer_count: execution_plan.layer_count,
        dense_layer_count: execution_plan.dense_layer_count,
        sparse_layer_count: execution_plan.sparse_layer_count,
        kv_layout: kv_config.layout_label().to_owned(),
        kv_bytes_per_token: kv_config.bytes_per_token(),
        request_prefill_tokens: 0,
        request_prefill_chunks: 0,
        request_kv_snapshot_restore_ms: 0.0,
        request_decode_budget: 0,
        request_mtp_verify_rows: 0,
        request_mtp_accepted_rows: 0,
        request_dspark_joint_batch_width: 0,
        request_dspark_joint_cohort_width: 0,
        request_dspark_joint_cohort_count: 0,
        request_dspark_2x2_wavefront: false,
        request_coordinator_graph_slots: 0,
        request_coordinator_graph_captured_graphs: 0,
        request_coordinator_graph_captures: 0,
        request_coordinator_graph_launches: 0,
        request_candidate_layerwaves: 0,
        request_deferred_layerwaves: 0,
        scheduler_iterations: 0,
        selected_layerwaves: 0,
        sparse_expert_batches: 0,
        request_expert_batch_rows: 0,
        request_expert_batch_routes: 0,
        request_expert_prefill_rows: 0,
        request_expert_decode_rows: 0,
        request_expert_mtp_verify_rows: 0,
        request_expert_prefill_routes: 0,
        request_expert_decode_routes: 0,
        request_expert_mtp_verify_routes: 0,
        kv_read_blocks: 0,
        committed_kv_writes: 0,
        tentative_kv_writes: 0,
        request_committed_mtp_writes: 0,
        request_discarded_mtp_writes: 0,
        request_backed_kv_writes: 0,
        request_backed_kv_bytes: 0,
        request_kv_reservation_bytes: 0,
        request_byte_backed_scheduler_trace: false,
        scheduler_numeric_progression_passed: false,
        scheduler_numeric_progression_source_rows: 0,
        scheduler_numeric_progression_hidden_dim: 0,
        scheduler_numeric_progression_visible_checksum: 0.0,
        scheduler_numeric_progression_rejected_mtp_checksum: 0.0,
        request_numeric_progression_selected_prefill_rows: 0,
        request_numeric_progression_selected_decode_rows: 0,
        request_numeric_progression_selected_mtp_rows: 0,
        request_numeric_progression_attention_value_updates: 0,
        request_numeric_progression_mlp_value_updates: 0,
        scheduler_full_context_device_attention_complete: false,
        scheduler_terminal_lm_head_sample_status: "not-run".to_owned(),
        scheduler_terminal_lm_head_sample_passed: false,
        scheduler_terminal_lm_head_uses_final_decode_device_hidden: false,
        scheduler_terminal_lm_head_covers_full_vocabulary: false,
        scheduler_terminal_lm_head_logits_evaluated: 0,
        scheduler_terminal_lm_head_vocab_size: 0,
        scheduler_terminal_lm_head_top_token_id: None,
        scheduler_terminal_lm_head_sampled_token_id: None,
        scheduler_terminal_lm_head_sampled_text: None,
        scheduler_terminal_lm_head_sample_top_k: None,
        scheduler_terminal_lm_head_sample_top_p: None,
        scheduler_terminal_lm_head_argmax_backend: None,
        scheduler_terminal_lm_head_sampler_backend: None,
        scheduler_terminal_lm_head_blocker: Some(
            "serving startup only preloads resident weights; scheduler terminal lm_head sampling is only populated by preflight execution"
                .to_owned(),
        ),
        protocol: execution_plan.protocol_payloads.protocol.to_owned(),
        decode_wire_request_bytes_per_touched_host: execution_plan
            .protocol_payloads
            .decode_wire_request_bytes_per_touched_host,
        decode_wire_response_bytes_per_touched_host: execution_plan
            .protocol_payloads
            .decode_wire_response_bytes_per_touched_host,
        prefill_wire_request_bytes_per_touched_host: execution_plan
            .protocol_payloads
            .prefill_wire_request_bytes_per_touched_host,
        prefill_wire_response_bytes_per_touched_host: execution_plan
            .protocol_payloads
            .prefill_wire_response_bytes_per_touched_host,
        mtp_wire_request_bytes_per_touched_host: execution_plan
            .protocol_payloads
            .mtp_wire_request_bytes_per_touched_host,
        mtp_wire_response_bytes_per_touched_host: execution_plan
            .protocol_payloads
            .mtp_wire_response_bytes_per_touched_host,
        decode_full_sparse_roundtrip_wire_bytes: execution_plan
            .protocol_payloads
            .decode_full_sparse_roundtrip_wire_bytes,
        prefill_full_sparse_roundtrip_wire_bytes: execution_plan
            .protocol_payloads
            .prefill_full_sparse_roundtrip_wire_bytes,
        mtp_full_sparse_roundtrip_wire_bytes: execution_plan
            .protocol_payloads
            .mtp_full_sparse_roundtrip_wire_bytes,
        scheduler_sparse_tcp_dispatch_status: "not-configured".to_owned(),
        scheduler_sparse_tcp_dispatch_targets: 0,
        scheduler_sparse_tcp_dispatch_sparse_layers: 0,
        scheduler_sparse_tcp_dispatch_iterations_per_sparse_layer: 0,
        scheduler_sparse_tcp_dispatch_batches: 0,
        scheduler_sparse_tcp_dispatch_host_batches: 0,
        scheduler_sparse_tcp_dispatch_global_rows: 0,
        scheduler_sparse_tcp_dispatch_host_rows: 0,
        scheduler_sparse_tcp_dispatch_routes: 0,
        scheduler_sparse_tcp_dispatch_request_wire_bytes: 0,
        scheduler_sparse_tcp_dispatch_response_wire_bytes: 0,
        scheduler_sparse_tcp_dispatch_output_values: 0,
        scheduler_sparse_tcp_dispatch_output_finite_values: 0,
        scheduler_sparse_tcp_dispatch_output_nonzero_values: 0,
        scheduler_sparse_tcp_dispatch_output_checksum: 0.0,
        scheduler_sparse_tcp_dispatch_passed: false,
        scheduler_sparse_tcp_dispatch_expected_real_executor_id:
            scheduler_sparse_tcp_expected_real_executor_id(),
        scheduler_sparse_tcp_dispatch_response_executor_ids_observed: 0,
        scheduler_sparse_tcp_dispatch_real_executor_responses: 0,
        scheduler_sparse_tcp_dispatch_non_real_executor_responses: 0,
        scheduler_sparse_tcp_dispatch_all_responses_real_checkpoint_experts: false,
        scheduler_sparse_tcp_dispatch_all_responses_real_nvfp4: false,
        scheduler_sparse_tcp_dispatch_consumed_by_residual: false,
        sampling_default_lm_head_chunk_passed: false,
        sampling_default_lm_head_chunk_rows_scored: 0,
        sampling_default_lm_head_chunk_lm_head_bytes_read: 0,
        sampling_default_lm_head_chunk_top_token_id: None,
        sampling_default_lm_head_chunk_top_logit: None,
        sampling_default_lm_head_chunk_uses_real_dense_prefix: false,
        sampling_default_lm_head_chunk_residual_source_dense_layers: 0,
        sampling_default_lm_head_chunk_residual_source_dense_weight_bytes_read: 0,
        sampling_default_lm_head_chunk_residual_after_checksum: None,
        blocker: REAL_DS4_FULL_BLOCKER.to_owned(),
        failed_requirements: vec![
            "full_residual_stream_execution".to_owned(),
            "full_vocab_sampling".to_owned(),
        ],
    })
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn real_full_info_from_report(
    report: &RealDs4FullPreflightReport,
) -> ds4rt_api::RealFullInfo {
    let lm_head_chunk = &report.sampling_dry_run.real_lm_head_default_chunk_probe;
    let terminal_lm_head_sample = &report.scheduler_execution_dry_run.terminal_lm_head_sample;
    ds4rt_api::RealFullInfo {
        status: report.status.to_owned(),
        model_id: report.model_id.clone(),
        model_facts: report.model_facts.clone(),
        snapshot_path: Some(report.snapshot_path.clone()),
        catalog_hash: report.catalog_hash.clone(),
        quantization_recipe: report.model_facts.quantization_recipe.clone(),
        tensor_count: report.tensor_count,
        startup_diagnostic_mode: "preflight-report".to_owned(),
        coordinator_resident_preload_status: report.coordinator_resident_preload.status.to_owned(),
        coordinator_resident_preload_selected_tensors: report
            .coordinator_resident_preload
            .selected_tensor_count,
        coordinator_resident_preload_selected_bytes: report
            .coordinator_resident_preload
            .selected_tensor_bytes,
        coordinator_resident_preload_loaded_bytes: report
            .coordinator_resident_preload
            .loaded_tensor_bytes,
        layer_count: report.execution_plan.layer_count,
        dense_layer_count: report.execution_plan.dense_layer_count,
        sparse_layer_count: report.execution_plan.sparse_layer_count,
        kv_layout: report.kv_plan.layout.to_owned(),
        kv_bytes_per_token: report.kv_plan.bytes_per_token,
        request_prefill_tokens: report.scheduler_execution_dry_run.request_prefill_tokens,
        request_prefill_chunks: report.scheduler_execution_dry_run.request_prefill_chunks,
        request_kv_snapshot_restore_ms: 0.0,
        request_decode_budget: report.scheduler_execution_dry_run.request_decode_rows,
        request_mtp_verify_rows: report.scheduler_execution_dry_run.request_mtp_verify_rows,
        request_mtp_accepted_rows: report.scheduler_execution_dry_run.request_mtp_accepted_rows,
        request_dspark_joint_batch_width: 0,
        request_dspark_joint_cohort_width: 0,
        request_dspark_joint_cohort_count: 0,
        request_dspark_2x2_wavefront: false,
        request_coordinator_graph_slots: report
            .scheduler_execution_dry_run
            .request_coordinator_graph_slots,
        request_coordinator_graph_captured_graphs: report
            .scheduler_execution_dry_run
            .request_coordinator_graph_captured_graphs,
        request_coordinator_graph_captures: report
            .scheduler_execution_dry_run
            .request_coordinator_graph_captures,
        request_coordinator_graph_launches: report
            .scheduler_execution_dry_run
            .request_coordinator_graph_launches,
        request_candidate_layerwaves: report.scheduler_execution_dry_run.candidate_layerwaves,
        request_deferred_layerwaves: report.scheduler_execution_dry_run.deferred_layerwaves,
        scheduler_iterations: report.scheduler_execution_dry_run.iterations,
        selected_layerwaves: report.scheduler_execution_dry_run.selected_layerwaves,
        sparse_expert_batches: report.scheduler_execution_dry_run.sparse_expert_batches,
        request_expert_batch_rows: report.scheduler_execution_dry_run.sparse_expert_batch_rows,
        request_expert_batch_routes: report
            .scheduler_execution_dry_run
            .sparse_expert_batch_routes,
        request_expert_prefill_rows: report
            .scheduler_execution_dry_run
            .sparse_expert_prefill_rows,
        request_expert_decode_rows: report.scheduler_execution_dry_run.sparse_expert_decode_rows,
        request_expert_mtp_verify_rows: report
            .scheduler_execution_dry_run
            .sparse_expert_mtp_verify_rows,
        request_expert_prefill_routes: report
            .scheduler_execution_dry_run
            .sparse_expert_prefill_routes,
        request_expert_decode_routes: report
            .scheduler_execution_dry_run
            .sparse_expert_decode_routes,
        request_expert_mtp_verify_routes: report
            .scheduler_execution_dry_run
            .sparse_expert_mtp_verify_routes,
        kv_read_blocks: report.scheduler_execution_dry_run.kv_read_blocks,
        committed_kv_writes: report.scheduler_execution_dry_run.committed_kv_writes,
        tentative_kv_writes: report.scheduler_execution_dry_run.tentative_kv_writes,
        request_committed_mtp_writes: report.scheduler_execution_dry_run.committed_mtp_writes,
        request_discarded_mtp_writes: report.scheduler_execution_dry_run.discarded_mtp_writes,
        request_backed_kv_writes: report.scheduler_execution_dry_run.backed_kv_writes,
        request_backed_kv_bytes: report
            .scheduler_execution_dry_run
            .backed_bytes_after_discard,
        request_kv_reservation_bytes: report.scheduler_execution_dry_run.kv_reservation_bytes,
        request_byte_backed_scheduler_trace: report
            .scheduler_execution_dry_run
            .byte_backed_scheduler_trace,
        scheduler_numeric_progression_passed: report
            .scheduler_execution_dry_run
            .numeric_progression_self_test
            .passed,
        scheduler_numeric_progression_source_rows: report
            .scheduler_execution_dry_run
            .numeric_progression_self_test
            .unique_source_rows,
        scheduler_numeric_progression_hidden_dim: report
            .scheduler_execution_dry_run
            .numeric_progression_self_test
            .hidden_dim,
        scheduler_numeric_progression_visible_checksum: report
            .scheduler_execution_dry_run
            .numeric_progression_self_test
            .final_visible_checksum,
        scheduler_numeric_progression_rejected_mtp_checksum: report
            .scheduler_execution_dry_run
            .numeric_progression_self_test
            .rejected_mtp_checksum,
        request_numeric_progression_selected_prefill_rows: report
            .scheduler_execution_dry_run
            .numeric_progression_self_test
            .selected_prefill_rows,
        request_numeric_progression_selected_decode_rows: report
            .scheduler_execution_dry_run
            .numeric_progression_self_test
            .selected_decode_rows,
        request_numeric_progression_selected_mtp_rows: report
            .scheduler_execution_dry_run
            .numeric_progression_self_test
            .selected_mtp_rows,
        request_numeric_progression_attention_value_updates: report
            .scheduler_execution_dry_run
            .numeric_progression_self_test
            .attention_value_updates,
        request_numeric_progression_mlp_value_updates: report
            .scheduler_execution_dry_run
            .numeric_progression_self_test
            .mlp_value_updates,
        scheduler_full_context_device_attention_complete: report
            .scheduler_execution_dry_run
            .full_context_device_attention_complete,
        scheduler_terminal_lm_head_sample_status: terminal_lm_head_sample.status.to_owned(),
        scheduler_terminal_lm_head_sample_passed: terminal_lm_head_sample.passed,
        scheduler_terminal_lm_head_uses_final_decode_device_hidden: terminal_lm_head_sample
            .uses_final_decode_device_hidden,
        scheduler_terminal_lm_head_covers_full_vocabulary: terminal_lm_head_sample
            .covers_full_vocabulary,
        scheduler_terminal_lm_head_logits_evaluated: terminal_lm_head_sample.logits_evaluated,
        scheduler_terminal_lm_head_vocab_size: terminal_lm_head_sample.vocab_size,
        scheduler_terminal_lm_head_top_token_id: terminal_lm_head_sample.top_token_id,
        scheduler_terminal_lm_head_sampled_token_id: terminal_lm_head_sample.sampled_token_id,
        scheduler_terminal_lm_head_sampled_text: decode_sampled_token_text(
            &report.snapshot_path,
            terminal_lm_head_sample.sampled_token_id,
        ),
        scheduler_terminal_lm_head_sample_top_k: terminal_lm_head_sample.sample_top_k,
        scheduler_terminal_lm_head_sample_top_p: terminal_lm_head_sample.sample_top_p,
        scheduler_terminal_lm_head_argmax_backend: terminal_lm_head_sample
            .argmax_kernel_backend
            .map(str::to_owned),
        scheduler_terminal_lm_head_sampler_backend: terminal_lm_head_sample
            .sampler_kernel_backend
            .map(str::to_owned),
        scheduler_terminal_lm_head_blocker: terminal_lm_head_sample.blocker.clone(),
        protocol: report.execution_plan.protocol_payloads.protocol.to_owned(),
        decode_wire_request_bytes_per_touched_host: report
            .execution_plan
            .protocol_payloads
            .decode_wire_request_bytes_per_touched_host,
        decode_wire_response_bytes_per_touched_host: report
            .execution_plan
            .protocol_payloads
            .decode_wire_response_bytes_per_touched_host,
        prefill_wire_request_bytes_per_touched_host: report
            .execution_plan
            .protocol_payloads
            .prefill_wire_request_bytes_per_touched_host,
        prefill_wire_response_bytes_per_touched_host: report
            .execution_plan
            .protocol_payloads
            .prefill_wire_response_bytes_per_touched_host,
        mtp_wire_request_bytes_per_touched_host: report
            .execution_plan
            .protocol_payloads
            .mtp_wire_request_bytes_per_touched_host,
        mtp_wire_response_bytes_per_touched_host: report
            .execution_plan
            .protocol_payloads
            .mtp_wire_response_bytes_per_touched_host,
        decode_full_sparse_roundtrip_wire_bytes: report
            .execution_plan
            .protocol_payloads
            .decode_full_sparse_roundtrip_wire_bytes,
        prefill_full_sparse_roundtrip_wire_bytes: report
            .execution_plan
            .protocol_payloads
            .prefill_full_sparse_roundtrip_wire_bytes,
        mtp_full_sparse_roundtrip_wire_bytes: report
            .execution_plan
            .protocol_payloads
            .mtp_full_sparse_roundtrip_wire_bytes,
        scheduler_sparse_tcp_dispatch_status: "not-configured".to_owned(),
        scheduler_sparse_tcp_dispatch_targets: 0,
        scheduler_sparse_tcp_dispatch_sparse_layers: 0,
        scheduler_sparse_tcp_dispatch_iterations_per_sparse_layer: 0,
        scheduler_sparse_tcp_dispatch_batches: 0,
        scheduler_sparse_tcp_dispatch_host_batches: 0,
        scheduler_sparse_tcp_dispatch_global_rows: 0,
        scheduler_sparse_tcp_dispatch_host_rows: 0,
        scheduler_sparse_tcp_dispatch_routes: 0,
        scheduler_sparse_tcp_dispatch_request_wire_bytes: 0,
        scheduler_sparse_tcp_dispatch_response_wire_bytes: 0,
        scheduler_sparse_tcp_dispatch_output_values: 0,
        scheduler_sparse_tcp_dispatch_output_finite_values: 0,
        scheduler_sparse_tcp_dispatch_output_nonzero_values: 0,
        scheduler_sparse_tcp_dispatch_output_checksum: 0.0,
        scheduler_sparse_tcp_dispatch_passed: false,
        scheduler_sparse_tcp_dispatch_expected_real_executor_id:
            scheduler_sparse_tcp_expected_real_executor_id(),
        scheduler_sparse_tcp_dispatch_response_executor_ids_observed: 0,
        scheduler_sparse_tcp_dispatch_real_executor_responses: 0,
        scheduler_sparse_tcp_dispatch_non_real_executor_responses: 0,
        scheduler_sparse_tcp_dispatch_all_responses_real_checkpoint_experts: false,
        scheduler_sparse_tcp_dispatch_all_responses_real_nvfp4: false,
        scheduler_sparse_tcp_dispatch_consumed_by_residual: false,
        sampling_default_lm_head_chunk_passed: lm_head_chunk.passed,
        sampling_default_lm_head_chunk_rows_scored: lm_head_chunk.rows_scored,
        sampling_default_lm_head_chunk_lm_head_bytes_read: lm_head_chunk.lm_head_bytes_read,
        sampling_default_lm_head_chunk_top_token_id: lm_head_chunk.top_token_id,
        sampling_default_lm_head_chunk_top_logit: lm_head_chunk.top_logit,
        sampling_default_lm_head_chunk_uses_real_dense_prefix: lm_head_chunk.uses_real_dense_prefix,
        sampling_default_lm_head_chunk_residual_source_dense_layers: lm_head_chunk
            .residual_source_dense_layers,
        sampling_default_lm_head_chunk_residual_source_dense_weight_bytes_read: lm_head_chunk
            .residual_source_dense_weight_bytes_read,
        sampling_default_lm_head_chunk_residual_after_checksum: lm_head_chunk
            .residual_after_checksum,
        blocker: report.blocker.to_owned(),
        failed_requirements: report
            .requirements
            .iter()
            .filter(|requirement| !requirement.passed)
            .map(|requirement| requirement.name.to_owned())
            .collect(),
    }
}

fn real_full_info_from_request_execution(
    base_info: &ds4rt_api::RealFullInfo,
    snapshot_path: &str,
    report: &RealFullSchedulerExecutionDryRun,
    sampled_token_text: Option<String>,
) -> ds4rt_api::RealFullInfo {
    let terminal_lm_head_sample = &report.terminal_lm_head_sample;
    let completed = report.full_context_device_attention_complete && terminal_lm_head_sample.passed;
    let mut info = base_info.clone();
    info.status = if completed { "ready" } else { "blocked" }.to_owned();
    info.startup_diagnostic_mode = "request-scheduler-execution".to_owned();
    info.request_prefill_tokens = report.request_prefill_tokens;
    info.request_prefill_chunks = report.request_prefill_chunks;
    info.request_decode_budget = report.request_decode_rows;
    info.request_mtp_verify_rows = report.request_mtp_verify_rows;
    info.request_mtp_accepted_rows = report.request_mtp_accepted_rows;
    info.request_coordinator_graph_slots = report.request_coordinator_graph_slots;
    info.request_coordinator_graph_captured_graphs =
        report.request_coordinator_graph_captured_graphs;
    info.request_coordinator_graph_captures = report.request_coordinator_graph_captures;
    info.request_coordinator_graph_launches = report.request_coordinator_graph_launches;
    info.request_candidate_layerwaves = report.candidate_layerwaves;
    info.request_deferred_layerwaves = report.deferred_layerwaves;
    info.scheduler_iterations = report.iterations;
    info.selected_layerwaves = report.selected_layerwaves;
    info.sparse_expert_batches = report.sparse_expert_batches;
    info.request_expert_batch_rows = report.sparse_expert_batch_rows;
    info.request_expert_batch_routes = report.sparse_expert_batch_routes;
    info.request_expert_prefill_rows = report.sparse_expert_prefill_rows;
    info.request_expert_decode_rows = report.sparse_expert_decode_rows;
    info.request_expert_mtp_verify_rows = report.sparse_expert_mtp_verify_rows;
    info.request_expert_prefill_routes = report.sparse_expert_prefill_routes;
    info.request_expert_decode_routes = report.sparse_expert_decode_routes;
    info.request_expert_mtp_verify_routes = report.sparse_expert_mtp_verify_routes;
    info.kv_read_blocks = report.kv_read_blocks;
    info.committed_kv_writes = report.committed_kv_writes;
    info.tentative_kv_writes = report.tentative_kv_writes;
    info.request_committed_mtp_writes = report.committed_mtp_writes;
    info.request_discarded_mtp_writes = report.discarded_mtp_writes;
    info.request_backed_kv_writes = report.backed_kv_writes;
    info.request_backed_kv_bytes = report.backed_bytes_after_discard;
    info.request_kv_reservation_bytes = report.kv_reservation_bytes;
    info.request_byte_backed_scheduler_trace = report.byte_backed_scheduler_trace;
    info.scheduler_numeric_progression_passed = report.numeric_progression_self_test.passed;
    info.scheduler_numeric_progression_source_rows =
        report.numeric_progression_self_test.unique_source_rows;
    info.scheduler_numeric_progression_hidden_dim = report.numeric_progression_self_test.hidden_dim;
    info.scheduler_numeric_progression_visible_checksum =
        report.numeric_progression_self_test.final_visible_checksum;
    info.scheduler_numeric_progression_rejected_mtp_checksum =
        report.numeric_progression_self_test.rejected_mtp_checksum;
    info.request_numeric_progression_selected_prefill_rows =
        report.numeric_progression_self_test.selected_prefill_rows;
    info.request_numeric_progression_selected_decode_rows =
        report.numeric_progression_self_test.selected_decode_rows;
    info.request_numeric_progression_selected_mtp_rows =
        report.numeric_progression_self_test.selected_mtp_rows;
    info.request_numeric_progression_attention_value_updates =
        report.numeric_progression_self_test.attention_value_updates;
    info.request_numeric_progression_mlp_value_updates =
        report.numeric_progression_self_test.mlp_value_updates;
    info.scheduler_full_context_device_attention_complete =
        report.full_context_device_attention_complete;
    info.scheduler_terminal_lm_head_sample_status = terminal_lm_head_sample.status.to_owned();
    info.scheduler_terminal_lm_head_sample_passed = terminal_lm_head_sample.passed;
    info.scheduler_terminal_lm_head_uses_final_decode_device_hidden =
        terminal_lm_head_sample.uses_final_decode_device_hidden;
    info.scheduler_terminal_lm_head_covers_full_vocabulary =
        terminal_lm_head_sample.covers_full_vocabulary;
    info.scheduler_terminal_lm_head_logits_evaluated = terminal_lm_head_sample.logits_evaluated;
    info.scheduler_terminal_lm_head_vocab_size = terminal_lm_head_sample.vocab_size;
    info.scheduler_terminal_lm_head_top_token_id = terminal_lm_head_sample.top_token_id;
    info.scheduler_terminal_lm_head_sampled_token_id = terminal_lm_head_sample.sampled_token_id;
    info.scheduler_terminal_lm_head_sampled_text = sampled_token_text.or_else(|| {
        decode_sampled_token_text(snapshot_path, terminal_lm_head_sample.sampled_token_id)
    });
    info.scheduler_terminal_lm_head_sample_top_k = terminal_lm_head_sample.sample_top_k;
    info.scheduler_terminal_lm_head_sample_top_p = terminal_lm_head_sample.sample_top_p;
    info.scheduler_terminal_lm_head_argmax_backend = terminal_lm_head_sample
        .argmax_kernel_backend
        .map(str::to_owned);
    info.scheduler_terminal_lm_head_sampler_backend = terminal_lm_head_sample
        .sampler_kernel_backend
        .map(str::to_owned);
    info.scheduler_terminal_lm_head_blocker = terminal_lm_head_sample.blocker.clone();
    if completed {
        info.blocker.clear();
        info.failed_requirements.clear();
    } else {
        info.blocker = terminal_lm_head_sample
            .blocker
            .clone()
            .unwrap_or_else(|| REAL_DS4_FULL_BLOCKER.to_owned());
        info.failed_requirements = real_full_request_failed_requirements(report);
    }
    info
}

fn real_full_request_failed_requirements(report: &RealFullSchedulerExecutionDryRun) -> Vec<String> {
    let mut failed = Vec::new();
    if !report.numeric_progression_self_test.passed {
        failed.push("scheduler_numeric_progression".to_owned());
    }
    if !report.full_context_device_attention_complete {
        failed.push("full_residual_stream_execution".to_owned());
    }
    if !report.terminal_lm_head_sample.passed {
        failed.push("full_vocab_sampling".to_owned());
    }
    failed
}

fn decode_sampled_token_text(snapshot_path: &str, token_id: Option<usize>) -> Option<String> {
    let token_id = token_id?;
    let token_id = u32::try_from(token_id).ok()?;
    decode_tokenizer_ids(Path::new(snapshot_path), &[token_id], false)
        .ok()
        .map(|summary| summary.text)
}

fn decode_sampled_token_text_with_tokenizer(
    tokenizer: &LoadedTokenizer,
    token_id: Option<usize>,
) -> Option<String> {
    let token_id = token_id?;
    let token_id = u32::try_from(token_id).ok()?;
    tokenizer
        .decode_ids(&[token_id], false)
        .ok()
        .map(|summary| summary.text)
}

fn load_catalog(path: &Path) -> Result<TensorCatalog> {
    serde_json::from_reader(
        File::open(path).with_context(|| format!("opening {}", path.display()))?,
    )
    .with_context(|| format!("parsing {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::{
        batched_dspark_pair_ranges, batched_dspark_replay_shape, default_real_full_kv_pool_tokens,
        finish_real_full_dspark_width_prewarm_sequence, parse_real_full_dspark_confidence_policy,
        real_full_batched_dspark_prewarm_buffer_bank,
        real_full_batched_dspark_prewarm_requested_draft_tokens,
        real_full_batched_dspark_prewarm_sequence, real_full_capture_arena_sequence,
        real_full_dspark_cross_request_tail_reuse_enabled, real_full_dspark_prefix_fingerprint,
        real_full_dspark_startup_draft_tokens, real_full_paired_lm_head_buffer_bank,
        real_full_paired_lm_head_prewarm_range, real_full_prefill_chunk_tokens_for_native_target,
        real_full_request_mtp_rows_for_policy, real_full_request_prefill_chunk_tokens_for_sequence,
        real_full_request_prefill_chunk_tokens_for_shape_with, real_full_request_token_rows,
        real_full_scalar_dspark_prewarm_requested_draft_tokens, real_full_sequence_capacity_tokens,
        real_full_serve_dsa_selector_seed_prompt, real_full_serve_prewarm_prefill_rows,
        real_full_sparse_tcp_targets_from_args, real_full_speculative_acceptance,
        real_full_startup_target_radix_evict_tokens, real_full_startup_target_radix_publish_tokens,
        real_full_startup_workspace_is_final_capture_set,
        real_full_startup_workspace_sizing_sequence, real_full_target_radix_reuse_supported,
        request_prompt_token_ids, retain_graph_bound_scheduler_arena,
        speculative_terminal_lm_head_sample, target_radix_device_pages,
        target_radix_device_pages_for_capacity, validate_real_full_strict_tp4_args,
        DsparkConfidenceCalibrator, DsparkConfidenceResidual, DsparkRequestCacheSnapshot,
        RealFullContextTokenBudget, RealFullContextTokenExtent, RealFullDsparkCacheMode,
        RealFullDsparkConfidencePolicy, RealFullDsparkTailCache, RealFullDsparkTailEntry,
        RealFullDsparkTailKey, RealFullSpeculativeTerminalSample, TargetKvRadixManager,
        REAL_FULL_MAX_ACTIVE_REQUESTS, REAL_FULL_SERVE_DSA_SELECTOR_PREWARM_QUERY_ROWS,
        REAL_FULL_SHARED_KV_PAGE_TOKENS,
    };
    use crate::cli::CoordinatorArgs;
    use crate::commands::real_full::preflight::real_full_sparse_transport_plan;
    use anyhow::Result;
    use ds4rt_core::{KvCacheDType, DEFAULT_MODEL_ID, DS4_KV_SOURCE_PAGE_TOKENS, EXPERT_HOSTS};
    use ds4rt_loader::LoadedTokenizer;
    use std::fs;
    use std::sync::Arc;

    fn coordinator_args(transport: &str, expert_hosts: &str) -> CoordinatorArgs {
        CoordinatorArgs {
            backend: "real-ds4-full".to_owned(),
            transport: transport.to_owned(),
            kv_cache_dtype: "bf16".to_owned(),
            max_context_tokens: crate::cli::DEFAULT_REAL_FULL_MAX_CONTEXT_TOKENS,
            listen: "127.0.0.1:8000".to_owned(),
            model_id: DEFAULT_MODEL_ID.to_owned(),
            expert_hosts: expert_hosts.to_owned(),
            catalog: None,
            loadplan: None,
            preflight_only: false,
        }
    }

    #[test]
    fn default_target_kv_pool_is_one_shared_context_not_lane_multiplied() -> Result<()> {
        assert_eq!(default_real_full_kv_pool_tokens(8_192)?, 8_192);
        assert_eq!(default_real_full_kv_pool_tokens(400_000)?, 400_128);
        assert!(default_real_full_kv_pool_tokens(0).is_err());
        assert!(default_real_full_kv_pool_tokens(usize::MAX).is_err());
        assert!(default_real_full_kv_pool_tokens(usize::MAX - 1).is_err());
        Ok(())
    }

    #[test]
    fn speculative_terminal_report_preserves_active_model_hidden_width() {
        let sample = RealFullSpeculativeTerminalSample {
            hidden_dim: ds4rt_core::DS4_FLASH_HIDDEN_SIZE,
            vocab_size: 129_280,
            top_token_id: 17,
            sampled_token_id: 19,
            sample_top_k: 8,
            sample_top_p: 0.95,
            argmax_backend: "test-argmax",
            sampler_backend: "test-sampler",
            accepted_draft_tokens: 3,
            report_mtp_acceptance: true,
        };

        let report = speculative_terminal_lm_head_sample(&sample);
        assert_eq!(report.hidden_dim, ds4rt_core::DS4_FLASH_HIDDEN_SIZE);
        assert_eq!(report.vocab_size, sample.vocab_size);
        assert_eq!(report.sampled_token_id, Some(sample.sampled_token_id));
        assert!(report.passed);
    }

    #[test]
    fn native_source_pages_expand_to_contiguous_device_page_quartets() {
        assert_eq!(
            target_radix_device_pages(&[5, 2], DS4_KV_SOURCE_PAGE_TOKENS).unwrap(),
            vec![20, 21, 22, 23, 8, 9, 10, 11],
        );
        assert_eq!(
            target_radix_device_pages(&[5, 2], REAL_FULL_SHARED_KV_PAGE_TOKENS).unwrap(),
            vec![5, 2],
        );
        assert_eq!(
            target_radix_device_pages_for_capacity(&[5], DS4_KV_SOURCE_PAGE_TOKENS, 27).unwrap(),
            vec![20],
        );
        assert_eq!(
            target_radix_device_pages_for_capacity(&[5], DS4_KV_SOURCE_PAGE_TOKENS, 65).unwrap(),
            vec![20, 21],
        );
        assert_eq!(
            target_radix_device_pages_for_capacity(&[5, 2], DS4_KV_SOURCE_PAGE_TOKENS, 257)
                .unwrap(),
            vec![20, 21, 22, 23, 8],
        );
    }

    #[test]
    fn dspark_confidence_policy_defaults_to_residual_and_retains_diagnostic_modes() {
        assert_eq!(
            parse_real_full_dspark_confidence_policy(None).unwrap(),
            RealFullDsparkConfidencePolicy::Residual
        );
        assert_eq!(
            parse_real_full_dspark_confidence_policy(Some("calibrated")).unwrap(),
            RealFullDsparkConfidencePolicy::Calibrated
        );
        assert_eq!(
            parse_real_full_dspark_confidence_policy(Some("raw")).unwrap(),
            RealFullDsparkConfidencePolicy::Raw
        );
        assert_eq!(
            parse_real_full_dspark_confidence_policy(Some("residual")).unwrap(),
            RealFullDsparkConfidencePolicy::Residual
        );
        assert!(parse_real_full_dspark_confidence_policy(Some("legacy")).is_err());
    }

    #[test]
    fn dspark_cross_request_tail_reuse_requires_absolute_prompt_swa_positions() {
        assert!(!real_full_dspark_cross_request_tail_reuse_enabled(
            RealFullDsparkCacheMode::RequestLocal
        ));
        assert!(real_full_dspark_cross_request_tail_reuse_enabled(
            RealFullDsparkCacheMode::PromptSwa
        ));
    }

    #[test]
    fn target_radix_reuse_waits_for_compressor_state_replay() {
        let flash = ds4rt_core::ModelFacts::default();
        assert!(flash.compress_ratios.iter().any(|ratio| *ratio > 0));
        assert!(!real_full_target_radix_reuse_supported(&flash));

        let mut uncompressed = flash;
        uncompressed.compress_ratios.fill(0);
        assert!(real_full_target_radix_reuse_supported(&uncompressed));
    }

    fn dspark_tail_entry(token_ids: &[usize], bytes: usize) -> RealFullDsparkTailEntry {
        RealFullDsparkTailEntry {
            key: RealFullDsparkTailKey {
                prefix_tokens: token_ids.len(),
                prefix_sha256: real_full_dspark_prefix_fingerprint(token_ids),
            },
            snapshot: DsparkRequestCacheSnapshot {
                context_tokens: token_ids.len(),
                cache_context_tokens: token_ids.len(),
                kv_bytes: vec![0_u8; bytes],
            },
            confidence_calibrator: DsparkConfidenceCalibrator::default(),
            confidence_residual: DsparkConfidenceResidual::default(),
        }
    }

    #[test]
    fn dspark_tail_cache_requires_the_exact_target_radix_frontier() {
        let token_ids = [11, 12, 13, 14];
        let mut cache = RealFullDsparkTailCache::new(16);
        assert!(cache.insert(dspark_tail_entry(&token_ids[..3], 4)));
        assert!(cache.take_exact_prefix(&token_ids, 4).is_none());
        assert_eq!(cache.entries.len(), 1);
        assert_eq!(
            cache
                .take_exact_prefix(&token_ids, 3)
                .expect("the exact dSpark tail is reusable")
                .key
                .prefix_tokens,
            3
        );
        assert_eq!(cache.resident_bytes, 0);
    }

    #[test]
    fn dspark_tail_cache_evicts_lru_entries_by_bytes() {
        let mut cache = RealFullDsparkTailCache::new(8);
        assert!(cache.insert(dspark_tail_entry(&[1], 4)));
        assert!(cache.insert(dspark_tail_entry(&[1, 2], 4)));
        assert!(cache.insert(dspark_tail_entry(&[1, 2, 3], 4)));
        assert_eq!(cache.entries.len(), 2);
        assert_eq!(cache.entries[0].key.prefix_tokens, 2);
        assert_eq!(cache.entries[1].key.prefix_tokens, 3);
        assert_eq!(cache.resident_bytes, 8);
    }

    #[test]
    fn dspark_tail_cache_finds_the_longest_target_aligned_prefix() {
        let prompt = [1, 2, 3, 4];
        let mut cache = RealFullDsparkTailCache::new(16);
        assert!(cache.insert(dspark_tail_entry(&prompt[..2], 4)));
        assert!(cache.insert(dspark_tail_entry(&prompt[..3], 4)));
        assert_eq!(cache.longest_exact_prefix_tokens(&prompt, 4), 3);
        assert_eq!(cache.longest_exact_prefix_tokens(&prompt, 2), 2);
        assert_eq!(cache.longest_exact_prefix_tokens(&[9, 2, 3, 4], 4), 0);
    }

    #[test]
    fn flash_smoke_prewarm_rows_respect_the_context_ceiling() -> Result<()> {
        let rows = real_full_serve_prewarm_prefill_rows(4_096, 2_048)?;
        assert_eq!(rows[0], 2_048);
        assert!(rows.contains(&2_048));
        assert!(!rows.contains(&4_096));
        assert!(rows.iter().all(|rows| rows + 2 <= 4_096));
        Ok(())
    }

    #[test]
    fn production_prewarm_rows_leave_wide_chunks_to_targeted_requests() -> Result<()> {
        let rows = real_full_serve_prewarm_prefill_rows(114_688, 2_048)?;
        assert_eq!(&rows[..2], &[2_048, 512]);
        assert!(!rows.contains(&4_096));
        assert!(!rows.contains(&1_024));
        Ok(())
    }

    #[test]
    fn startup_targeted_prefill_sequences_force_both_production_widths() {
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_sequence(
                "real-full-startup-capture-arena-max-prefill-chunk-4097-sequence-0",
                0,
                4_096,
            ),
            2_048,
        );
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_sequence(
                "real-full-startup-capture-arena-canonical-prefill-chunk-2048-sequence-0",
                0,
                2_048,
            ),
            1_024,
        );
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_sequence(
                "real-full-startup-capture-arena-2049-sequence-0",
                0,
                2_048,
            ),
            512,
        );
    }

    #[test]
    fn flash_smoke_dsa_selector_uses_a_seed_that_leaves_sweep_headroom() -> Result<()> {
        let rows = real_full_serve_prewarm_prefill_rows(4_096, 2_048)?;
        let prompts = rows
            .into_iter()
            .map(|rows| (String::new(), rows + 1))
            .collect::<Vec<_>>();
        let query_rows = REAL_FULL_SERVE_DSA_SELECTOR_PREWARM_QUERY_ROWS;
        let decode_budget = query_rows.len() + 1;
        let (_, seed_tokens) =
            real_full_serve_dsa_selector_seed_prompt(&prompts, query_rows, decode_budget, 4_096)?;
        assert_eq!(*seed_tokens, 2_049);
        let required_tokens =
            seed_tokens + query_rows.iter().map(|rows| rows + 1).sum::<usize>() + decode_budget;
        assert!(required_tokens <= 4_096);
        Ok(())
    }

    #[test]
    fn cached_prefix_startup_probe_uses_graph_stable_capture_arena() {
        assert!(real_full_capture_arena_sequence(
            "real-full-startup-capture-arena-4097-sequence-0"
        ));
        assert!(real_full_capture_arena_sequence(
            "real-full-startup-prefix-prefill-seed-1009-1008-repeat-0-sequence-0"
        ));
        assert!(real_full_capture_arena_sequence(
            "real-full-startup-dspark-width-5-2049-sequence-0"
        ));
        assert!(real_full_capture_arena_sequence(
            "real-full-startup-dsa-selector-seed-8193-sequence-0"
        ));
        assert!(!real_full_capture_arena_sequence(
            "real-full-startup-prewarm-initial-1009-sequence-0"
        ));
    }

    #[test]
    fn startup_target_radix_control_sequences_are_strictly_parsed() {
        let publish = "real-full-startup-capture-arena-radix-publish-8192-sequence-3";
        assert_eq!(
            real_full_startup_target_radix_publish_tokens(publish),
            Some(8192)
        );
        assert_eq!(
            real_full_startup_target_radix_publish_tokens(
                "real-full-startup-capture-arena-canonical-prefill-chunk-2048-sequence-3"
            ),
            Some(2048)
        );
        assert!(real_full_startup_workspace_sizing_sequence(publish));
        assert!(real_full_capture_arena_sequence(publish));

        assert_eq!(
            real_full_startup_target_radix_evict_tokens(
                "real-full-startup-evict-target-radix-prefix-8192-worker-3"
            ),
            Some(8192)
        );
        assert_eq!(
            real_full_startup_target_radix_publish_tokens(
                "real-full-startup-capture-arena-radix-publish-0-sequence-3"
            ),
            None
        );
        assert_eq!(
            real_full_startup_target_radix_evict_tokens(
                "real-full-startup-evict-target-radix-prefix-nope-worker-3"
            ),
            None
        );
    }

    #[test]
    fn paired_lm_head_prewarm_starts_after_the_single_request_widths() {
        assert_eq!(real_full_paired_lm_head_prewarm_range(Some(0)), None);
        assert_eq!(
            real_full_paired_lm_head_prewarm_range(Some(1)),
            Some((3, 4))
        );
        assert_eq!(
            real_full_paired_lm_head_prewarm_range(Some(7)),
            Some((9, 16))
        );
        assert_eq!(real_full_paired_lm_head_prewarm_range(None), Some((7, 12)));
    }

    #[test]
    fn paired_lm_head_uses_disjoint_owned_buffer_banks() {
        assert_eq!(real_full_paired_lm_head_buffer_bank(1), 17);
        assert_eq!(real_full_paired_lm_head_buffer_bank(8), 24);
        assert!((1..=8).all(|bank| real_full_paired_lm_head_buffer_bank(bank) > 8));
    }

    #[test]
    fn c4_dspark_replay_preserves_two_joint_pair_ranges() {
        assert_eq!(batched_dspark_pair_ranges(1), None);
        assert_eq!(batched_dspark_pair_ranges(2), None);
        assert_eq!(batched_dspark_pair_ranges(3), None);
        assert_eq!(batched_dspark_pair_ranges(4), Some(vec![0..2, 2..4]));
        assert_eq!(
            batched_dspark_pair_ranges(8),
            Some(vec![0..2, 2..4, 4..6, 6..8])
        );
    }

    #[test]
    fn dspark_replay_shape_reports_only_a_real_c4_2x2_wavefront() {
        assert_eq!(batched_dspark_replay_shape(0), None);
        assert_eq!(batched_dspark_replay_shape(1), Some((1, 1, 1, false)));
        assert_eq!(batched_dspark_replay_shape(3), Some((3, 3, 1, false)));
        assert_eq!(batched_dspark_replay_shape(4), Some((4, 2, 2, true)));
        assert_eq!(batched_dspark_replay_shape(8), Some((8, 2, 4, false)));
    }

    #[test]
    fn batched_dspark_prewarm_sequences_bind_the_requested_production_bank() {
        let sequence_id = "real-full-startup-dspark-width-5-batched-bank-4-sequence";
        assert_eq!(
            real_full_dspark_startup_draft_tokens("real-full-startup-dspark-width-0-9-sequence-0"),
            Some(0)
        );
        assert_eq!(
            real_full_batched_dspark_prewarm_buffer_bank(sequence_id),
            Some(4)
        );
        assert!(real_full_batched_dspark_prewarm_sequence(sequence_id));
        assert_eq!(
            real_full_batched_dspark_prewarm_requested_draft_tokens(sequence_id, 92_511),
            Some(5)
        );
        assert_eq!(
            real_full_batched_dspark_prewarm_requested_draft_tokens(sequence_id, 92_011),
            Some(0)
        );
        assert_eq!(
            real_full_batched_dspark_prewarm_requested_draft_tokens(sequence_id, 93_611),
            None
        );
        assert!(!real_full_batched_dspark_prewarm_sequence(
            "real-full-startup-dspark-width-5-9-sequence-0"
        ));
        assert!(!real_full_batched_dspark_prewarm_sequence(
            "real-full-startup-dspark-width-0-batched-bank-4-sequence"
        ));
    }

    #[test]
    fn scalar_dspark_width_cohort_decodes_each_request_width() {
        let sequence_id = "real-full-startup-dspark-width-5-scalar-cohort-2049-sequence-0";
        assert_eq!(
            real_full_scalar_dspark_prewarm_requested_draft_tokens(sequence_id, 91_500),
            Some(5)
        );
        assert_eq!(
            real_full_scalar_dspark_prewarm_requested_draft_tokens(sequence_id, 91_001),
            Some(0)
        );
        assert_eq!(
            real_full_scalar_dspark_prewarm_requested_draft_tokens(sequence_id, 90_999),
            None
        );
        assert_eq!(
            real_full_scalar_dspark_prewarm_requested_draft_tokens(
                "real-full-startup-dspark-width-5-2049-sequence-0",
                91_700,
            ),
            None
        );
    }

    #[test]
    fn serial_dspark_width_finish_releases_single_target_kv_slot() {
        let manager =
            Arc::new(TargetKvRadixManager::new(4 * REAL_FULL_SHARED_KV_PAGE_TOKENS, 1).unwrap());
        let first = manager.reserve(&[1, 2], REAL_FULL_SHARED_KV_PAGE_TOKENS);
        let mut first = Some(first.unwrap());
        let exhausted = manager
            .reserve(&[3, 4], REAL_FULL_SHARED_KV_PAGE_TOKENS)
            .unwrap_err();
        assert!(format!("{exhausted:#}")
            .contains("target KV active request limit exhausted: active=1 max=1"));

        let mut finished = Vec::new();
        {
            let mut finish_sequence = |sequence_id: &str| {
                finished.push(sequence_id.to_owned());
                drop(first.take());
                Ok(())
            };
            finish_real_full_dspark_width_prewarm_sequence(
                &mut finish_sequence,
                "real-full-startup-dspark-width-7-9-sequence-0",
                8,
            )
            .unwrap();
        }

        assert_eq!(finished, ["real-full-startup-dspark-width-7-9-sequence-0"]);
        assert_eq!(manager.stats().active_reservations, 0);
        let next = manager
            .reserve(&[3, 4], REAL_FULL_SHARED_KV_PAGE_TOKENS)
            .unwrap();
        assert_eq!(manager.stats().active_reservations, 1);
        drop(next);
    }

    #[test]
    fn only_explicit_graph_bound_max_context_arenas_are_recycled() {
        assert!(retain_graph_bound_scheduler_arena(true, 8_192, 8_192));
        assert!(!retain_graph_bound_scheduler_arena(false, 8_192, 8_192));
        assert!(!retain_graph_bound_scheduler_arena(true, 4_096, 8_192));
    }

    #[test]
    fn real_full_sequence_capacity_is_bounded_and_leaves_small_extension_headroom() {
        let max_context_tokens = crate::cli::DEFAULT_REAL_FULL_MAX_CONTEXT_TOKENS;
        assert_eq!(
            real_full_sequence_capacity_tokens(1_009, 2, max_context_tokens).unwrap(),
            2_020
        );
        assert_eq!(
            real_full_sequence_capacity_tokens(max_context_tokens - 1, 1, max_context_tokens)
                .unwrap(),
            max_context_tokens
        );
        assert!(
            real_full_sequence_capacity_tokens(max_context_tokens, 1, max_context_tokens).is_err()
        );
    }

    #[test]
    fn request_prefill_chunk_width_distinguishes_small_suffixes_and_balances_large_ones() {
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_shape_with(512, 4_096, 1_024, 0, 1_008),
            504
        );
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_shape_with(512, 4_096, 1_024, 1_009, 1_008,),
            256
        );
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_shape_with(
                2_048, 4_096, 1_024, 1_024, 1_994,
            ),
            512
        );
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_shape_with(1_024, 4_096, 1_024, 0, 4_095,),
            512
        );
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_shape_with(2_048, 4_096, 1_024, 0, 529,),
            265
        );
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_shape_with(2_048, 4_096, 1_024, 0, 1_041,),
            512
        );
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_shape_with(2_048, 4_096, 1_024, 0, 1_033,),
            517
        );
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_shape_with(2_048, 4_096, 1_024, 0, 1_038,),
            519
        );
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_shape_with(2_048, 4_096, 1_024, 0, 1_039,),
            512
        );
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_shape_with(1_024, 4_096, 1_024, 0, 4_096,),
            1_024
        );
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_shape_with(512, 4_096, 1_024, 1_009, 8_192,),
            512
        );
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_shape_with(2_048, 4_096, 1_024, 0, 4_096,),
            1_024
        );
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_shape_with(2_048, 4_096, 1_024, 0, 5_157,),
            1_290
        );
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_shape_with(
                2_048, 4_096, 1_024, 32_768, 13_584,
            ),
            1_941
        );
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_shape_with(
                2_048, 4_096, 1_024, 32_767, 1_008,
            ),
            256
        );
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_shape_with(
                2_048, 4_096, 1_024, 32_768, 1_008,
            ),
            1_024
        );
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_shape_with(
                2_048, 4_096, 512, 100_000, 1_008,
            ),
            512
        );
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_shape_with(2_048, 4_096, 512, 100_000, 530,),
            530
        );
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_shape_with(2_048, 4_096, 512, 100_000, 896,),
            896
        );
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_shape_with(2_048, 4_096, 512, 100_000, 897,),
            512
        );
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_shape_with(
                2_048, 4_096, 512, 100_000, 1_032,
            ),
            516
        );
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_shape_with(
                2_048, 4_096, 512, 100_000, 1_038,
            ),
            519
        );
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_shape_with(
                2_048, 4_096, 512, 100_000, 1_039,
            ),
            512
        );
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_shape_with(
                2_048, 4_096, 1_024, 100_000, 2_050,
            ),
            1_024
        );
    }

    #[test]
    fn production_prefill_stays_within_native_target_query_capacity() {
        assert_eq!(
            real_full_prefill_chunk_tokens_for_native_target(4_096),
            2_048
        );
        assert_eq!(
            real_full_prefill_chunk_tokens_for_native_target(2_048),
            2_048
        );
        assert_eq!(
            real_full_prefill_chunk_tokens_for_native_target(1_024),
            1_024
        );
        assert_eq!(real_full_prefill_chunk_tokens_for_native_target(512), 512);
    }

    #[test]
    fn real_full_context_budget_releases_dropped_sequence_reservations() {
        let budget = Arc::new(RealFullContextTokenBudget::new(3 * 64));
        let first = budget.reserve(65).unwrap();
        assert_eq!(first.token_base(), 0);
        assert_eq!(first.reserved_tokens, 128);
        let second = budget.reserve(1).unwrap();
        assert_eq!(second.token_base(), 128);
        assert!(budget.reserve(1).is_err());
        drop(first);
        let replacement = budget.reserve(64).unwrap();
        assert_eq!(replacement.token_base(), 0);
        drop((second, replacement));
        let inner = budget.inner.lock().unwrap();
        assert_eq!(inner.used_tokens, 0);
        assert_eq!(inner.active_reservations, 0);
        assert_eq!(
            inner.free_extents,
            vec![RealFullContextTokenExtent {
                token_base: 0,
                tokens: 3 * 64
            }]
        );
    }

    #[test]
    fn real_full_context_budget_caps_the_active_resident_set() {
        let budget = Arc::new(RealFullContextTokenBudget::new(
            (REAL_FULL_MAX_ACTIVE_REQUESTS + 1) * 64,
        ));
        let reservations = (0..REAL_FULL_MAX_ACTIVE_REQUESTS)
            .map(|_| budget.reserve(1).unwrap())
            .collect::<Vec<_>>();
        assert!(budget.reserve(1).is_err());
        drop(reservations);
    }

    #[test]
    fn real_full_sparse_tcp_targets_reject_inproc() {
        let error =
            real_full_sparse_tcp_targets_from_args(&coordinator_args("inproc", "127.0.0.1:9100"))
                .unwrap_err();

        assert!(error.to_string().contains("strict DeepSeek V4 TP4"));
        assert!(error.to_string().contains("cannot execute routed experts"));
    }

    #[test]
    fn real_full_sparse_tcp_targets_expand_single_addr_to_all_expert_hosts() {
        let targets =
            real_full_sparse_tcp_targets_from_args(&coordinator_args("tcp", "127.0.0.1:9100"))
                .unwrap();

        assert_eq!(targets.len(), EXPERT_HOSTS.len());
        for host in EXPERT_HOSTS {
            let target = targets
                .iter()
                .find(|target| target.host == host)
                .expect("expanded target for expert host");
            assert_eq!(target.addr.port(), 9100);
        }
    }

    #[test]
    fn real_full_sparse_tcp_targets_accept_owner_mapped_entries() {
        let targets = real_full_sparse_tcp_targets_from_args(&coordinator_args(
            "tcp",
            "spark-0=127.0.0.1:9101,spark-1=127.0.0.1:9102,spark-2=127.0.0.1:9103,spark-3=127.0.0.1:9104",
        ))
        .unwrap();

        assert_eq!(targets.len(), 4);
        assert_eq!(targets[0].host, "spark-0");
        assert_eq!(targets[0].addr.port(), 9101);
        assert_eq!(targets[3].host, "spark-3");
        assert_eq!(targets[3].addr.port(), 9104);
    }

    #[test]
    fn real_full_sparse_tcp_targets_require_all_expert_hosts() {
        let err = real_full_sparse_tcp_targets_from_args(&coordinator_args(
            "tcp",
            "spark-0=127.0.0.1:9101,spark-1=127.0.0.1:9102",
        ))
        .unwrap_err();

        assert!(err.to_string().contains("missing expert targets"));
        assert!(err.to_string().contains("spark-2,spark-3"));
    }

    #[test]
    fn real_full_sparse_tcp_targets_reject_extra_rank_identity() {
        let err = real_full_sparse_tcp_targets_from_args(&coordinator_args(
            "tcp",
            "spark-0=127.0.0.1:9101,spark-1=127.0.0.1:9102,spark-2=127.0.0.1:9103,spark-3=127.0.0.1:9104,ep-extra=127.0.0.1:9105",
        ))
        .unwrap_err();

        assert!(err.to_string().contains(
            "strict DeepSeek V4 TP4 sparse dispatch requires exactly 4 named ranks, got 5"
        ));
    }

    #[test]
    fn deepseek_v4_tp4_rejects_legacy_owner_loadplan() {
        let mut args = coordinator_args("tcp", "127.0.0.1:9100");
        args.loadplan = Some("legacy-owner-loadplan.json".into());

        let err = validate_real_full_strict_tp4_args(&args).unwrap_err();
        assert!(err.to_string().contains("strict DeepSeek V4 TP=4"));
        assert!(err.to_string().contains("rejects --loadplan"));

        args.loadplan = None;
        validate_real_full_strict_tp4_args(&args).unwrap();
    }

    #[test]
    fn real_full_sparse_verbs_host_targets_and_plan_follow_rdma_preflight() {
        let args = coordinator_args(
            "verbs-host",
            "spark-0=127.0.0.1:9100,spark-1=127.0.0.1:9100,spark-2=127.0.0.1:9100,spark-3=127.0.0.1:9100",
        );
        let preflight_ok = ds4rt_transport::verbs_host_preflight().is_ok();
        let targets = real_full_sparse_tcp_targets_from_args(&args);
        if preflight_ok {
            let targets = targets.unwrap();
            assert_eq!(targets.len(), EXPERT_HOSTS.len());
            assert!(targets.iter().all(|target| target.addr.port() == 9100));
        } else {
            assert!(targets
                .unwrap_err()
                .to_string()
                .contains("RDMA preflight failed"));
        }

        let plan = real_full_sparse_transport_plan(&args);
        assert_eq!(plan.transport, "verbs-host");
        assert_eq!(plan.supports_rdma, true);
        assert_eq!(plan.supports_host_registered_buffers, true);
        assert_eq!(plan.app_transport_implemented, true);
        assert_eq!(
            plan.app_transport_status,
            ds4rt_transport::VERBS_HOST_APP_TRANSPORT_STATUS
        );
        assert_eq!(plan.sparse_dispatch_available, preflight_ok);
        assert_eq!(
            plan.scheduler_dispatch_backend.as_deref(),
            preflight_ok.then_some("verbs-host-protocol-v2-rc-qp")
        );
        assert_eq!(
            plan.frame_protocol.as_deref(),
            Some(ds4rt_transport::EXPERT_PROTOCOL_V2_FRAME_PROTOCOL)
        );
        assert_eq!(plan.blocker.is_none(), preflight_ok);
        assert_eq!(plan.preflight_ok, preflight_ok);
    }

    #[test]
    fn real_full_request_mtp_rows_default_to_disabled_for_live_serve() {
        let first_decode_loop_step =
            ds4rt_api::RealFullRequest::new_decode_step(1, "user: hi", 3, 4, Vec::new(), 0, 4);

        assert_eq!(
            real_full_request_mtp_rows_for_policy(&first_decode_loop_step, false, false),
            0
        );
    }

    #[test]
    fn ordinary_dspark_workspace_is_the_final_startup_capture_set() {
        assert!(real_full_startup_workspace_is_final_capture_set(false));
    }

    #[test]
    fn optional_startup_probes_require_the_conservative_final_capture_sweep() {
        assert!(!real_full_startup_workspace_is_final_capture_set(true));
    }

    #[test]
    fn real_full_request_mtp_rows_are_opt_in_and_disabled_for_recurrent_or_single_token_decode() {
        let first_decode_loop_step =
            ds4rt_api::RealFullRequest::new_decode_step(1, "user: hi", 3, 4, Vec::new(), 0, 4);
        let later_decode_loop_step =
            ds4rt_api::RealFullRequest::new_decode_step(2, "user: hi", 3, 4, vec![13], 1, 4);
        let standalone_single_step =
            ds4rt_api::RealFullRequest::new_decode_step(3, "user: hi", 3, 1, Vec::new(), 0, 1);

        assert_eq!(
            real_full_request_mtp_rows_for_policy(&first_decode_loop_step, false, true),
            4
        );
        assert_eq!(
            real_full_request_mtp_rows_for_policy(&later_decode_loop_step, true, true),
            0
        );
        assert_eq!(
            real_full_request_mtp_rows_for_policy(&standalone_single_step, false, true),
            0
        );
    }

    #[test]
    fn real_full_speculative_acceptance_without_bonus_keeps_a_cache_backed_leading_run() {
        assert_eq!(
            real_full_speculative_acceptance(&[11, 12, 13], &[11, 12, 99, 100], false, 4).unwrap(),
            super::RealFullSpeculativeAcceptance {
                accepted_draft_tokens: 2,
                terminal_target_index: 2,
                full_match_bonus: false,
            }
        );
        assert_eq!(
            real_full_speculative_acceptance(&[11, 12], &[99, 100, 101], false, 3)
                .unwrap()
                .accepted_draft_tokens,
            0,
        );
        assert_eq!(
            real_full_speculative_acceptance(&[11, 12], &[11, 12, 13], false, 3)
                .unwrap()
                .accepted_draft_tokens,
            1,
        );
        assert_eq!(
            real_full_speculative_acceptance(&[11], &[11, 12], false, 2)
                .unwrap()
                .accepted_draft_tokens,
            0,
        );
    }

    #[test]
    fn real_full_speculative_acceptance_emits_the_full_match_bonus() {
        assert_eq!(
            real_full_speculative_acceptance(&[11, 12, 13], &[11, 12, 13, 14], true, 4).unwrap(),
            super::RealFullSpeculativeAcceptance {
                accepted_draft_tokens: 3,
                terminal_target_index: 3,
                full_match_bonus: true,
            }
        );
        assert!(
            !real_full_speculative_acceptance(&[11, 12, 99], &[11, 12, 13, 14], true, 4)
                .unwrap()
                .full_match_bonus
        );
    }

    #[test]
    fn real_full_speculative_acceptance_requires_one_target_fallback_sample() {
        let error = real_full_speculative_acceptance(&[11, 12], &[11, 12], true, 3).unwrap_err();
        assert!(error.to_string().contains("plus one fallback"));
    }

    #[test]
    fn real_full_speculative_acceptance_clamps_fixed_width_tail_to_output_budget() {
        assert_eq!(
            real_full_speculative_acceptance(
                &[11, 12, 13, 14, 15, 16, 17],
                &[11, 12, 13, 14, 15, 16, 17, 18],
                true,
                6,
            )
            .unwrap(),
            super::RealFullSpeculativeAcceptance {
                accepted_draft_tokens: 5,
                terminal_target_index: 5,
                full_match_bonus: false,
            }
        );
    }

    #[test]
    fn real_full_request_token_rows_seed_initial_decode_from_last_prompt_token() {
        let request = ds4rt_api::RealFullRequest::new_decode_step_for_sequence(
            11,
            "split-initial-sequence",
            "prompt",
            5,
            1,
            Vec::new(),
            0,
            4,
        );

        let rows = real_full_request_token_rows(&request, Some(vec![10, 20, 30, 40, 50]))
            .expect("splitting initial token rows");

        assert_eq!(rows.prefix_tokens, 0);
        assert_eq!(rows.prefill_tokens, 4);
        assert_eq!(rows.prefill_token_ids, Some(vec![10, 20, 30, 40]));
        assert_eq!(rows.decode_token_ids, vec![50]);
    }

    #[test]
    fn real_full_request_token_rows_split_uncached_prompt_suffix() {
        let request = ds4rt_api::RealFullRequest::new_decode_step_for_sequence(
            13,
            "split-cached-prefix-sequence",
            "prompt",
            8,
            1,
            Vec::new(),
            1,
            2,
        )
        .with_cached_prompt_tokens(4);

        let rows =
            real_full_request_token_rows(&request, Some(vec![10, 20, 30, 40, 50, 60, 70, 80]))
                .expect("splitting uncached prompt suffix");

        assert_eq!(rows.prefix_tokens, 4);
        assert_eq!(rows.prefill_tokens, 3);
        assert_eq!(rows.prefill_token_ids, Some(vec![50, 60, 70]));
        assert_eq!(rows.decode_token_ids, vec![80]);
    }

    #[test]
    fn real_full_request_token_rows_seed_recurrent_decode_from_latest_generated_token() {
        let request = ds4rt_api::RealFullRequest::new_decode_step_for_sequence(
            12,
            "split-recurrent-sequence",
            "prompt",
            5,
            1,
            vec![101, 102],
            1,
            4,
        );

        let rows =
            real_full_request_token_rows(&request, None).expect("splitting recurrent token rows");

        assert_eq!(rows.prefix_tokens, 6);
        assert_eq!(rows.prefill_tokens, 0);
        assert_eq!(rows.prefill_token_ids, None);
        assert_eq!(rows.decode_token_ids, vec![102]);
    }

    #[test]
    fn request_prompt_token_ids_tokenizes_prompt_with_loaded_tokenizer() {
        let snapshot = tempfile::tempdir().expect("creating tokenizer snapshot");
        fs::write(
            snapshot.path().join("tokenizer.json"),
            r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"[UNK]":0,"user: Use real full.":7},"unk_token":"[UNK]"}}"#,
        )
        .expect("writing tokenizer fixture");
        let tokenizer =
            LoadedTokenizer::from_snapshot(snapshot.path()).expect("loading tokenizer fixture");
        let request = ds4rt_api::RealFullRequest::new_decode_step(
            9,
            "user: Use real full.",
            1,
            1,
            vec![42, 43],
            2,
            4,
        );

        let token_ids = request_prompt_token_ids(&tokenizer, &request)
            .expect("tokenizing request")
            .expect("token ids");

        assert_eq!(token_ids, vec![7]);
    }
}
