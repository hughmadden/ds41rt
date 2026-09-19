use super::*;
use crate::{v41_backbone_cache::CacheStage, v41_experts::coordinator::NativeTp4Wave,
    v41_requests::{RequestBatch, RequestTokens, Requests}, v41_target_pass::TargetPass};
use std::{net::SocketAddr, path::PathBuf};

#[path = "stream.rs"]
mod stream;
pub use stream::{StreamInput, StreamingResult};
#[path = "draft.rs"]
mod draft;
use draft::{NativeDraft, SharedDraft};

/// Retained greedy draft generation; target sampling remains external.
#[derive(Clone, Copy, Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DsparkConfig {
    pub draft_limit: u8,
    pub adaptive: bool,
    pub confidence_cutoff: Option<f64>,
}
impl DsparkConfig {
    pub(super) fn validate(&self) -> Result<()> {
        ensure!((1..=5).contains(&self.draft_limit), "native draft limit must be 1..5");
        ensure!(self.confidence_cutoff.is_none_or(|p| p.is_finite() && p>0. && p<=1.), "invalid draft confidence cutoff");
        ensure!(!(self.adaptive && self.confidence_cutoff.is_some()), "adaptive and confidence-only policies are exclusive");
        Ok(())
    }
}

/// Explicit construction with optional retained dSpark. vLLM must delegate its
/// physical KV and backbone weight allocation before invoking this factory.
/// This does not run a sampler, create a listener, or start the HTTP scheduler.
pub struct TargetConfig {
    /// Nonzero executor-incarnation nonce, never reused across worker restarts.
    pub owner: u64,
    pub snapshot: PathBuf,
    pub native_lib: PathBuf,
    pub peers: [SocketAddr; 4],
    pub batch_tokens: u32,
    pub max_context_tokens: u32,
    pub slots: u32,
    /// Paired compressed-source payload budget, rounded down to page groups.
    /// Fixed windows and cache metadata are additional; see CacheInfo.cache_bytes.
    pub cache_bytes: usize,
    /// None retains target-only construction and allocation.
    pub dspark: Option<DsparkConfig>,
}
impl TargetConfig {
    fn args(self) -> Result<crate::cli::NativeServeArgs> {
        ensure!(self.owner != 0, "invalid target owner");
        ensure!((80..=4096).contains(&self.batch_tokens), "invalid target batch capacity");
        ensure!((1..=1048576).contains(&self.max_context_tokens), "invalid target context bound");
        ensure!((1..=16).contains(&self.slots) && self.cache_bytes > 0, "invalid target cache plan");
        if let Some(draft)=self.dspark {
            draft.validate()?;
            ensure!(self.slots >= 2, "native speculative facade requires two draft lanes");
        }
        use crate::{cli::HostCacheStore, v41_native_serve::memory::{ByteSize, LocalLayers}};
        Ok(crate::cli::NativeServeArgs {
            rtx_gpus: 1, prefill_batch_tokens: self.batch_tokens,
            max_context_tokens: self.max_context_tokens, max_output_tokens: 1,
            kv_pool_size: Some(ByteSize(self.cache_bytes)), memory_reservation: None,
            rtx_expert_layers: LocalLayers::Count(0), concurrency: self.slots,
            prefix_cache_entries: 0, host_cache_bytes: 0, host_cache_chunk_bytes: 256 << 20,
            host_cache_store: HostCacheStore::OnRetain, host_cache_copy_budget_ms: 1000,
            host_cache_restore_budget_ms: 500, host_cache_store_pace_ms: 0,
            host_cache_min_tokens: 512, host_cache_max_tokens: self.max_context_tokens,
            host_cache_kinds: "prompt,turn".into(), dspark: self.dspark.is_some(),
            dspark_draft_limit: self.dspark.map_or(5,|d|d.draft_limit),
            dspark_adaptive: self.dspark.is_some_and(|d|d.adaptive),
            dspark_fixed: self.dspark.is_some_and(|d|!d.adaptive && d.confidence_cutoff.is_none()),
            dspark_confidence_cutoff: self.dspark.and_then(|d|d.confidence_cutoff),
            dspark_reuse_floor: None, independent_decode_lanes: true,
            snapshot: self.snapshot, native_lib: self.native_lib,
            peers: self.peers.to_vec(), listen: String::new(),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct CacheInfo {
    pub owner: u64,
    pub capacity_rows: u32,
    pub source_page_capacity: [usize; 4],
    pub source_pages_free: [usize; 4],
    pub source_payload_bytes: usize,
    pub cache_bytes: usize,
}

/// Direct attachment to the actual Requests -> BackboneCache -> SourceCache
/// chain. The only cache ledger is inside that retained bank.
pub struct NativeBank<'s, 'a> {
    requests: Rc<RefCell<&'s mut Requests<'a>>>,
    active: Rc<Active>,
    info: CacheInfo,
    max_context: u64,
    draft: SharedDraft<'s, 'a>,
}
impl<'s, 'a> NativeBank<'s, 'a> {
    pub(crate) fn library(&self) -> &'a ds41rt_ffi::NativeLibrary {
        self.requests.borrow().cache().prefix_library()
    }
    pub fn admit(&self, slot: usize, request_id: u64) -> Result<RequestHandle> {
        self.active.healthy()?;
        let request=self.requests.borrow_mut().admit(slot,request_id)?;
        if let Some(draft)=self.draft.borrow_mut().as_deref_mut() {
            if let Err(error)=draft.admit(request_id) {
                if self.requests.borrow_mut().release(request).is_err(){self.active.poisoned.set(true);}
                return Err(error);
            }
        }
        Ok(request)
    }
    pub fn release(&self, request: RequestHandle) -> Result<()> {
        self.active.idle(request)?;
        let id=self.requests.borrow().cache().request_id(request)?;
        let draft=self.draft.borrow_mut().as_deref_mut().map(|draft|draft.release(id)).transpose();
        let target=self.requests.borrow_mut().release(request);
        let result=target.and(draft.map(|_|()));
        if result.is_err(){self.active.poisoned.set(true);}
        result
    }
    pub fn committed_end(&self, request: RequestHandle) -> Result<u64> {
        self.active.healthy()?;
        self.requests.borrow().cache().committed_end(request)
    }
    /// Actual matched three-window frontier; enabled but unseeded windows have
    /// logical end zero. None means the draft runtime was not constructed.
    pub fn draft_committed_end(&self, request: RequestHandle) -> Result<Option<u64>> {
        self.active.healthy()?;
        let id = self.requests.borrow().cache().request_id(request)?;
        self.draft.borrow().as_deref().map(|draft| draft.committed_end(id)
            .map(|end| end.unwrap_or(0))).transpose()
    }
    /// Read-only future append feasibility against the real native page bank.
    /// This does not retain reservations or authorize execution by itself.
    pub fn can_prepare(&self, work: &[(RequestHandle, u32)]) -> Result<bool> {
        ensure!(!work.is_empty() && work.len() <= 16, "invalid capacity query count");
        let requests = self.requests.borrow();
        for (i, &(request, tokens)) in work.iter().enumerate() {
            self.active.idle(request)?;
            ensure!(!work[..i].iter().any(|&(r, _)| r == request), "duplicate capacity query request");
            let end = requests.cache().committed_end(request)?;
            ensure!(end.checked_add(u64::from(tokens)).is_some_and(|end| end <= self.max_context),
                "capacity query exceeds context bound");
            ensure!(requests.cache().stage(request)? == CacheStage::Full, "capacity query requires full-target stage");
        }
        match requests.cache().check_append_capacity(work) {
            Ok(()) => Ok(true),
            Err(error) if error.is::<crate::v41_compressor::source_cache::SourcePoolExhausted>() => Ok(false),
            Err(error) => Err(error),
        }
    }
    pub fn info(&self) -> CacheInfo {
        let requests = self.requests.borrow();
        CacheInfo { source_page_capacity: std::array::from_fn(|s|
            requests.cache().sources()[s].source_cache().capacity / 256),
            source_pages_free: std::array::from_fn(|s|
            requests.cache().sources()[s].source_cache().free_pages()), ..self.info.clone() }
    }
}

/// This batch owns native Engram staging and canonical request/cache bindings.
pub struct NativeBatch { batch: RequestBatch, request: RequestHandle }
pub struct NativeDriver<'s, 'w, 'a> {
    pass: &'s mut TargetPass<'w, 'a>,
    transport: &'s mut NativeTp4Wave<'a>,
    requests: Rc<RefCell<&'s mut Requests<'a>>>,
    max_context: u64,
    draft: SharedDraft<'s, 'a>,
    lane: usize,
}
// Safety: all work uses retained native futures and their drain guards; the
// context prevents publication/reuse until futures and output borrows end.
unsafe impl TargetDriver for NativeDriver<'_, '_, '_> {
    type Batch = NativeBatch;
    fn validate(&self, input: &TargetInput) -> Result<()> {
        let requests = self.requests.borrow();
        let end = requests.cache().committed_end(input.request)?;
        ensure!(end.checked_add(input.tokens.len() as u64).is_some_and(|n| n <= self.max_context),
            "target context bound exceeded");
        ensure!(requests.cache().stage(input.request)? == CacheStage::Full,
            "target-only seam does not yet accept encoder/replay work");
        if let Some(draft)=self.draft.borrow().as_deref() {
            let id=requests.cache().request_id(input.request)?;
            draft.validate_position(id,end)?;
        }
        Ok(())
    }
    fn prepare(&mut self, input: &TargetInput) -> Result<NativeBatch> {
        let batch = self.requests.borrow().prepare(&[RequestTokens {
            lease: input.request, tokens: &input.tokens, image_mask: None, kind: input.kind,
        }])?;
        Ok(NativeBatch { batch, request: input.request })
    }
    async fn execute(&mut self, batch: &mut NativeBatch, input: &TargetInput) -> Result<()> {
        self.pass.set_route_capture(self.draft.borrow().as_deref().is_some_and(NativeDraft::capture_routes));
        unsafe { self.pass.execute_shared(&self.requests, &mut batch.batch,
            self.transport, input.placement, &input.selected).await?; }
        Ok(())
    }
    fn logits<'b>(&'b self, batch: &'b NativeBatch) -> Result<Logits<'b>> {
        let output = self.pass.output(&batch.batch)?;
        Ok(Logits { rows: output.rows, selected: output.selected_rows,
            positions: output.token_positions, device: Some(output.logits), host: None })
    }
    fn commit(&mut self, batch: &mut NativeBatch, accepted: u32) -> Result<u64> {
        let mut requests = self.requests.borrow_mut();
        if let Some(draft)=self.draft.borrow_mut().as_deref_mut() {
            draft.commit(self.pass,&mut requests,&mut batch.batch,accepted,true)?;
        } else {self.pass.commit(&mut requests,&mut batch.batch,&[accepted])?;}
        self.pass.set_route_capture(false);
        requests.cache().committed_end(batch.request)
    }
    fn drain(&mut self, batch: &mut NativeBatch) -> Result<()> {
        // Evaluate every drain even if one reports an error. No borrow of the
        // request bank survives transport execution or a CUDA completion wait.
        self.pass.set_route_capture(false);
        let transport = self.transport.synchronize();
        let commit = self.pass.abort_cache_commit(&mut self.requests.borrow_mut());
        let discard = self.pass.discard(&mut batch.batch);
        transport.and(commit).and(discard)
    }
}

/// Owned computing-thread scope. Lifetimes retain the library, immutable weights,
/// native bank and both transport/lane owners; no self-referential object or Send.
/// ```compile_fail
/// use ds41rt_daemon::native_executor::target::NativeTarget;
/// fn require_send<T: Send>() {}
/// require_send::<NativeTarget<'static, 'static, 'static>>();
/// ```
pub struct NativeTarget<'s, 'w, 'a> {
    contexts: [TargetContext<NativeDriver<'s, 'w, 'a>>; 2],
    bank: NativeBank<'s, 'a>,
    runtime: &'s tokio::runtime::Runtime,
    stream_chunk_rows: usize,
}
impl<'s, 'w, 'a> NativeTarget<'s, 'w, 'a> {
    pub fn bank(&self) -> &NativeBank<'s, 'a> { &self.bank }
    /// Poll both futures on this runtime/constructing thread, for example with
    /// tokio::join!. No Python per-rank executor or shared lane lock is needed.
    pub fn split(&mut self) -> (&tokio::runtime::Runtime, &NativeBank<'s, 'a>,
        [&mut TargetContext<NativeDriver<'s, 'w, 'a>>; 2]) {
        let [first, second] = &mut self.contexts;
        (self.runtime, &self.bank, [first, second])
    }
}

impl TargetContext<NativeDriver<'_, '_, '_>> {
    /// Explicit diagnostic D2H copy using the retained head downloader. Serving
    /// consumers should borrow logits instead. Row indices address the compact
    /// selected output, not the original token input.
    pub async fn download_logits(&mut self, ticket: Ticket, rows: &[usize]) -> Result<Vec<u8>> {
        self.check(ticket)?;
        let job = self.job.as_ref().unwrap();
        ensure!(job.phase == Phase::Ready, "target logits not complete");
        self.driver.pass.download_logits(&job.batch.batch, rows).await
    }
}

/// Call from a dedicated CUDA owner thread after delegating physical model/cache
/// ownership. Return values cannot borrow this scope. A future C ABI actor can
/// run its command loop here, keeping every borrowed native resource on-stack.
/// ```no_run
/// use ds41rt_daemon::native_executor::target::{with_target, TargetConfig, TargetInput, SourceKind};
/// fn two_requests(config: TargetConfig) -> anyhow::Result<()> {
///     with_target(config, |mut target| {
///         let (runtime, bank, [first, second]) = target.split();
///         let a = bank.admit(0, 100)?;
///         let b = bank.admit(1, 101)?;
///         let input = |request| TargetInput { request, tokens: vec![1, 2, 3],
///             selected: vec![2], kind: SourceKind::Prefill, placement: 1 };
///         let x = first.submit(input(a))?;
///         let y = second.submit(input(b))?;
///         let (x_done, y_done) = runtime.block_on(async {
///             tokio::join!(first.execute(x), second.execute(y))
///         });
///         x_done?; y_done?;
///         let logits = first.logits(x)?;
///         assert_eq!(logits.positions, &[2]);
///         drop(logits); // external CUDA sampling must also finish before this
///         first.commit(x, 3)?;
///         second.commit(y, 3)?;
///         bank.release(a)?; bank.release(b)?;
///         Ok(())
///     })
/// }
/// ```
/// ```compile_fail
/// use ds41rt_daemon::native_executor::target::{with_target, TargetConfig};
/// fn escape(config: TargetConfig) {
///     let _dangling = with_target(config, |target| Ok(target));
/// }
/// ```
pub fn with_target<R>(config: TargetConfig,
    run: impl for<'s, 'w, 'a> FnOnce(NativeTarget<'s, 'w, 'a>) -> Result<R>,
) -> Result<R> {
    let owner = config.owner;
    let args = config.args()?;
    crate::v41_native_serve::with_components(&args, |parts| {
        let active = Rc::new(Active::default());
        parts.requests.bind_executor_owner(owner)?;
        let owner = parts.requests.cache().owner();
        let requests = Rc::new(RefCell::new(parts.requests));
        let draft:SharedDraft<'_, '_>=Rc::new(RefCell::new(parts.draft.map(|draft|draft as &mut dyn NativeDraft<'_>)));
        let first = NativeDriver { pass: parts.pass, transport: parts.transport,
            requests: requests.clone(), max_context: args.max_context_tokens as u64, draft:draft.clone(),lane:0 };
        let second = NativeDriver { pass: parts.prefill_pass, transport: parts.prefill_transport,
            requests: requests.clone(), max_context: args.max_context_tokens as u64, draft:draft.clone(),lane:1 };
        let bank = NativeBank { requests, draft, active: active.clone(), max_context: args.max_context_tokens as u64, info: CacheInfo {
            owner, capacity_rows: parts.capacity, source_page_capacity: parts.source_pages,
            source_pages_free: parts.source_pages,
            source_payload_bytes: parts.source_pages.iter().sum::<usize>() * 256
                * (68 + ds41rt_ffi::V41Kv::COMPRESSED_ROW_BYTES),
            cache_bytes: parts.cache_bytes,
        } };
        run(NativeTarget { bank, runtime: parts.runtime, stream_chunk_rows: args.prefill_batch_tokens as usize, contexts: [
            TargetContext::new(first, active.clone(), 0, parts.capacity as usize, 48),
            TargetContext::new(second, active, 1, parts.capacity as usize, 48),
        ] })
    })
}

#[cfg(test)]
mod config_tests {
    use super::*;
    fn config() -> TargetConfig {
        TargetConfig { owner: 77, snapshot: "missing-model".into(), native_lib: "missing-library".into(),
            peers: ["127.0.0.1:1".parse().unwrap(); 4], batch_tokens: 1024,
            max_context_tokens: 8192, slots: 2, cache_bytes: 512 << 20, dspark:None }
    }
    #[test]
    fn explicit_target_factory_never_enables_native_server_or_duplicate_draft() {
        let args = config().args().unwrap();
        assert!(!args.dspark && args.listen.is_empty());
        assert_eq!(args.prefix_cache_entries, 0);
        assert_eq!(args.host_cache_bytes, 0);
        assert_eq!(args.rtx_expert_layers, crate::v41_native_serve::memory::LocalLayers::Count(0));
        assert_eq!(args.kv_pool_size.unwrap().0, 512 << 20);
    }
    #[test]
    fn optional_draft_reuses_retained_policy_flags_and_keeps_target_default() {
        for (adaptive, cutoff) in [(false, None), (true, None), (false, Some(0.6))] {
            let mut cfg = config();
            cfg.dspark = Some(DsparkConfig { draft_limit: 3, adaptive, confidence_cutoff: cutoff });
            let args = cfg.args().unwrap();
            assert!(args.dspark);
            assert_eq!(args.dspark_draft_limit, 3);
            assert_eq!(args.dspark_fixed, !adaptive && cutoff.is_none());
            assert_eq!(args.dspark_confidence_cutoff, cutoff);
        }
        let mut cfg = config();
        cfg.dspark = Some(DsparkConfig { draft_limit: 5, adaptive: false, confidence_cutoff: Some(f64::NAN) });
        assert!(cfg.args().is_err());
    }
    #[test]
    fn invalid_geometry_rejected_before_loading_any_native_library() {
        for mutate in [|c: &mut TargetConfig| c.owner = 0,
            |c: &mut TargetConfig| c.batch_tokens = 79,
            |c: &mut TargetConfig| c.slots = 17,
            |c: &mut TargetConfig| c.max_context_tokens = 0,
            |c: &mut TargetConfig| c.cache_bytes = 0] {
            let mut c = config(); mutate(&mut c);
            let error = with_target(c, |_| Ok(())).unwrap_err().to_string();
            assert!(error.starts_with("invalid target"), "{error}");
        }
    }
}
