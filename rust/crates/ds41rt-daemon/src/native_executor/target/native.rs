use super::*;
use crate::{v41_backbone_cache::CacheStage, v41_experts::coordinator::NativeTp4Wave,
    v41_requests::{RequestBatch, RequestTokens, Requests}, v41_target_pass::TargetPass};
use std::{net::SocketAddr, path::PathBuf};

/// Explicit target-only construction. vLLM must delegate its physical KV and
/// backbone weight allocation before invoking this factory. This does not load
/// dSpark, run a sampler, create a listener, or start the native HTTP scheduler.
pub struct TargetConfig {
    /// Nonzero executor-incarnation nonce, never reused across worker restarts.
    pub owner: u64,
    pub snapshot: PathBuf,
    pub native_lib: PathBuf,
    pub peers: [SocketAddr; 4],
    pub batch_tokens: u32,
    pub max_context_tokens: u32,
    pub slots: u32,
    /// Exact native physical cache budget, including windows and source pools.
    pub cache_bytes: usize,
}
impl TargetConfig {
    fn args(self) -> Result<crate::cli::NativeServeArgs> {
        ensure!(self.owner != 0, "invalid target owner");
        ensure!((80..=4096).contains(&self.batch_tokens), "invalid target batch capacity");
        ensure!((1..=1048576).contains(&self.max_context_tokens), "invalid target context bound");
        ensure!((1..=16).contains(&self.slots) && self.cache_bytes > 0, "invalid target cache plan");
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
            host_cache_kinds: "prompt,turn".into(), dspark: false, dspark_draft_limit: 5,
            dspark_adaptive: false, dspark_fixed: false, dspark_confidence_cutoff: None,
            dspark_reuse_floor: None, independent_decode_lanes: true,
            snapshot: self.snapshot, native_lib: self.native_lib,
            peers: self.peers.to_vec(), listen: String::new(),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CacheInfo {
    pub owner: u64,
    pub capacity_rows: u32,
    pub source_page_capacity: [usize; 4],
    pub source_pages_free: [usize; 4],
    pub cache_bytes: usize,
}

/// Direct attachment to the actual Requests -> BackboneCache -> SourceCache
/// chain. The only cache ledger is inside that retained bank.
pub struct NativeBank<'s, 'a> {
    requests: Rc<RefCell<&'s mut Requests<'a>>>,
    active: Rc<Active>,
    info: CacheInfo,
}
impl NativeBank<'_, '_> {
    pub fn admit(&self, slot: usize, request_id: u64) -> Result<RequestHandle> {
        self.active.healthy()?;
        self.requests.borrow_mut().admit(slot, request_id)
    }
    pub fn release(&self, request: RequestHandle) -> Result<()> {
        self.active.idle(request)?;
        self.requests.borrow_mut().release(request)
    }
    pub fn committed_end(&self, request: RequestHandle) -> Result<u64> {
        self.active.healthy()?;
        self.requests.borrow().cache().committed_end(request)
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
        Ok(())
    }
    fn prepare(&mut self, input: &TargetInput) -> Result<NativeBatch> {
        let batch = self.requests.borrow().prepare(&[RequestTokens {
            lease: input.request, tokens: &input.tokens, image_mask: None, kind: input.kind,
        }])?;
        Ok(NativeBatch { batch, request: input.request })
    }
    async fn execute(&mut self, batch: &mut NativeBatch, input: &TargetInput) -> Result<()> {
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
        self.pass.commit(&mut requests, &mut batch.batch, &[accepted])?;
        requests.cache().committed_end(batch.request)
    }
    fn drain(&mut self, batch: &mut NativeBatch) -> Result<()> {
        // Evaluate every drain even if one reports an error. No borrow of the
        // request bank survives transport execution or a CUDA completion wait.
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
        let first = NativeDriver { pass: parts.pass, transport: parts.transport,
            requests: requests.clone(), max_context: args.max_context_tokens as u64 };
        let second = NativeDriver { pass: parts.prefill_pass, transport: parts.prefill_transport,
            requests: requests.clone(), max_context: args.max_context_tokens as u64 };
        let bank = NativeBank { requests, active: active.clone(), info: CacheInfo {
            owner, capacity_rows: parts.capacity, source_page_capacity: parts.source_pages,
            source_pages_free: parts.source_pages, cache_bytes: parts.cache_bytes,
        } };
        run(NativeTarget { bank, runtime: parts.runtime, contexts: [
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
            max_context_tokens: 8192, slots: 2, cache_bytes: 512 << 20 }
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
