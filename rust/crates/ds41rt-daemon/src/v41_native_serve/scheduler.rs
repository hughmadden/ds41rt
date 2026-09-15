use super::*;
mod independent;
use super::scores::BatchScores;
use crate::v41_backbone_cache::CacheLease;
use crate::v41_requests::RequestBatch;
use super::prefix::{ImageKeys, PrefixCache, SnapshotKind};

struct Active<'a> {
    constraint: Option<super::constraints::State<'a>>,
    id: u64,
    lease: CacheLease,
    job: NativeRequest,
    decoder: ds41rt_loader::StreamingTokenDecoder,
    anchor: u32,
    generated: usize,
    buffered: usize,
    lane: usize,
    finished: bool,
    cacheable: bool,
    tokens: Vec<u32>,
    image_keys: ImageKeys,
    next_after_commit: Option<TokenScores>,
}
impl Active<'_> {
    fn emit_one(&mut self, token: u32) -> Result<[Option<InferenceChunk>; 3]> {
        ensure!(!self.job.events.is_closed(), "client disconnected");
        if let Some(constraint) = &mut self.constraint { constraint.accept(token)?; }
        self.anchor = token;
        self.tokens.push(token);
        self.generated += 1;
        self.buffered += 1;
        let mut chunks = [None, None, None];
        let mut count = 0;
        let mut push = |chunk| { chunks[count] = Some(chunk); count += 1; };
        if token != 1 {
            if let Some(content) = self.decoder.step(token)? {
                push(InferenceChunk::Text { content, content_tokens: self.buffered });
                self.buffered = 0;
            }
        }
        if token == 1 || self.generated == self.job.max_tokens {
            if let Some(content) = self.decoder.finish()? {
                push(InferenceChunk::Text { content, content_tokens: self.buffered });
                self.buffered = 0;
            }
            if self.buffered > 0 {
                push(InferenceChunk::Text { content: String::new(), content_tokens: self.buffered });
                self.buffered = 0;
            }
            push(InferenceChunk::Finish { finish_reason: if token == 1 { InferenceFinishReason::Stop }
                else { InferenceFinishReason::Length } });
            self.finished = true;
            self.cacheable = true;
        }
        Ok(chunks)
    }
    fn emit(&mut self, tokens: &[u32]) -> Result<()> {
        for &token in tokens {
            for chunk in self.emit_one(token)?.into_iter().flatten() { self.job.events.blocking_send(Ok(chunk))?; }
            if self.finished { break; }
        }
        Ok(())
    }
}

fn retire_request<'a>(request: Active<'a>, requests: &mut Requests<'a>,
    prefixes: &mut PrefixCache<'a>, mut draft: Option<&mut DraftRuntime<'_, 'a>>) -> Result<()> {
    if request.cacheable && requests.cache().request_id(request.lease).is_ok() {
        let retained = request.next_after_commit.as_ref().context("finished request has no retained logits")
            .and_then(|next| prefixes.retain(SnapshotKind::Turn, &request.tokens, &request.image_keys,
                next, request.id, request.lease, requests, draft.as_deref_mut()));
        if let Err(error) = retained { tracing::warn!(%error, "completed request prefix was not retained"); }
    }
    // Release both owners even if one cleanup reports an error.
    let target = requests.release_if_present(request.lease);
    let speculative = draft.map(|draft| draft.release(request.id)).transpose();
    target.and(speculative.map(|_| ()))
}

/// Prompt-side state of one admission attempt, prepared exactly once so a
/// deferred retry never re-tokenises the prompt or re-expands its images.
/// Nothing here borrows a lease, the prefix cache, or any scheduler resource
/// mutably: only `constraint` ties into `'a` (the library), which outlives the
/// serve loop, so a parked entry is safe to hold across loop iterations.
struct PreparedAdmission<'a> {
    job: NativeRequest,
    constraint: Option<super::constraints::State<'a>>,
    prompt: Vec<u32>,
    images: Vec<ds41rt_loader::V41ImageSpan>,
    decoder: ds41rt_loader::StreamingTokenDecoder,
    image_keys: ImageKeys,
}

/// A prepared admission parked because the compressed-KV pool was exhausted.
/// Deliberately holds no lease, no draft admission, and no device state:
/// everything allocated for the failed attempt is released before parking, so
/// a retry mints a fresh request id/lease pair exactly like a new job.
struct DeferredAdmission<'a> {
    prepared: PreparedAdmission<'a>,
    /// `format!("{error:#}")` of the first `SourcePoolExhausted`, kept for the
    /// terminal timeout message.
    error: String,
}

/// Result of one admission attempt: either the request is running, or the pool
/// starved and the prepared state is handed back intact for parking.
enum Admission<'a> {
    Running(Active<'a>),
    Starved { prepared: PreparedAdmission<'a>, error: String,
        needed: usize, available: usize },
}

/// Tokenise and expand a fresh request once, before any lease state exists.
/// Every fallible step here maps to the ordinary client-visible admission
/// failure; the expensive host-side work (tokeniser, vision expand, grammar
/// matcher, prefix key) never re-runs for a deferred retry.
fn prepare_admission<'a>(mut job: NativeRequest,
    compiler: &mut super::constraints::Compiler<'a>, snapshot: &std::path::Path,
    limits: &ds41rt_api::native_v41::NativeLimits, prefixes: &mut PrefixCache<'a>,
) -> Result<PreparedAdmission<'a>> {
    let constraint = job.constraint.as_ref().map(|spec| compiler.matcher(spec)).transpose()?;
    ensure!(!job.events.is_closed(), "client disconnected");
    let prompt = ds41rt_loader::encode_tokenizer_text(snapshot, &job.prompt, false)?.token_ids;
    let (prompt, images) = if job.images.is_empty() { (prompt, Vec::new()) } else {
        let expanded = ds41rt_loader::V41VisionPrompt::expand(&prompt,
            std::mem::take(&mut job.images), limits.context() as usize)?;
        (expanded.tokens, expanded.images)
    };
    job.max_tokens = limits.output_for_prompt(prompt.len(), job.max_tokens)?;
    let decoder = ds41rt_loader::streaming_token_decoder(snapshot, false)?;
    let image_keys = prefixes.prepare_key(&prompt, &images)?;
    Ok(PreparedAdmission { job, constraint, prompt, images, decoder, image_keys })
}

/// Run one admission attempt against prepared prompt state. Exactly one
/// `make_room` pass runs here, so a retry against a still-exhausted pool costs
/// one capacity check (plus any evictions it triggers) and nothing else; a
/// starved attempt releases nothing itself — the caller releases lease and
/// draft admission, then either parks the entry or fails it.
// The argument list is the verbatim-extracted serve-loop closure body and
// this is its single call site, so grouping the engine handles into a context
// struct would only re-wrap the same borrows; keep the allow instead.
#[allow(clippy::too_many_arguments)]
fn admit_prepared<'a, 'w>(prepared: PreparedAdmission<'a>, id: u64, lease: CacheLease, lane: usize,
    lib: &'a NativeLibrary, runtime: &tokio::runtime::Runtime,
    first: &mut TargetPass<'w, 'a>, second: &mut TargetPass<'w, 'a>,
    requests: &mut Requests<'a>, first_transport: &mut NativeTp4Wave<'a>,
    second_transport: &mut NativeTp4Wave<'a>, mut draft: Option<&mut DraftRuntime<'_, 'a>>,
    vision: &mut crate::v41_vision::VisionRuntime<'a>, prefixes: &mut PrefixCache<'a>,
    prefill_batch_tokens: usize,
) -> Result<Admission<'a>> {
    let PreparedAdmission { job, mut constraint, prompt, images, decoder, image_keys } = prepared;
    if let Some(draft) = draft.as_deref_mut() { draft.admit(id)?; }
    if !images.is_empty() {
        requests.attach_images(lease, crate::v41_requests::RequestImages::new(&images)?)?;
    }
    let hit = prefixes.restore(&prompt, &image_keys, id, lease, requests, draft.as_deref_mut())?;
    let cached = hit.as_ref().map_or(0, |(end, _)| *end);
    let source_end = requests.cache().committed_end(lease)? as usize;
    if let Err(error) = prefixes.make_room(requests, &[(lease, (prompt.len() - source_end) as u32)]) {
        let Some(pressure) = error.downcast_ref::<crate::v41_compressor::SourcePoolExhausted>()
        else { return Err(error) };
        // Nothing past the prepare phase was consumed, so the prepared state
        // returns intact for parking.
        return Ok(Admission::Starved { prepared: PreparedAdmission { job, constraint, prompt,
            images, decoder, image_keys }, error: format!("{error:#}"),
            needed: pressure.needed, available: pressure.available });
    }
    if !images.is_empty() {
        let start = if requests.cache().stage(lease)? == crate::v41_backbone_cache::CacheStage::EncoderReplay {
            requests.cache().history_end(lease)? as usize
        } else { source_end };
        let needed = requests.images(lease)?.needed(start, prompt.len())?;
        let started = Instant::now();
        for &index in &needed {
            ensure!(!job.events.is_closed(), "client disconnected");
            let features = vision.encode(&images[index].image)?;
            let mut bytes = vec![0; features.bytes];
            lib.copy_d2h(&mut bytes, features)?;
            requests.install_image_features(lease, index, bytes)?;
        }
        tracing::info!(request_id=id, images=images.len(), encoded_images=needed.len(),
            encoder_ms=started.elapsed().as_secs_f64()*1000.0, "native vision preparation");
    }
    drop(images);
    job.events.blocking_send(Ok(InferenceChunk::Ready {
        system_fingerprint: Some(if draft.is_some() { "ds41rt-native-fp4-kv-dspark" }
            else { "ds41rt-native-fp4-kv" }.into()),
        prompt_usage: PromptUsage { prompt_tokens: prompt.len(), prompt_cache_hit_tokens: cached },
    }))?;
    first_transport.begin_request(); second_transport.begin_request();
    let scores = if cached == prompt.len() { hit.expect("complete prefix hit").1.context("exact prefix has no logits")? }
    else { prefill(lib, runtime, first, second, requests, first_transport,
        second_transport, lease, &prompt, prefill_batch_tokens, &job,
        draft.as_deref_mut())? };
    if cached != prompt.len() {
      if let Err(error) = prefixes.retain(SnapshotKind::Prompt, &prompt, &image_keys, &scores, id, lease, requests, draft) {
        tracing::warn!(%error, "prompt prefix was not retained");
      }
    }
    tracing::debug!(request_id=id, prompt_tokens=prompt.len(), cached_tokens=cached, "native prefix admission");
    let mask = constraint.as_mut().map(|state| state.mask()).transpose()?.flatten();
    let anchor = scores.select(mask)?;
    Ok(Admission::Running(Active { constraint, id, lease, job, decoder, anchor, generated: 0, buffered: 0, lane,
        finished: false, cacheable: false, tokens: prompt, image_keys, next_after_commit: Some(scores) }))
}

/// Scheduler-local FIFO of admissions deferred on `SourcePoolExhausted`.
/// Generic and pure — the clock and the disconnect probe are injected by the
/// caller — so the park/timeout policy is unit-testable without the engine.
struct AdmissionQueue<T> {
    entries: std::collections::VecDeque<QueuedAdmission<T>>,
}

#[derive(Debug)]
struct QueuedAdmission<T> {
    value: T,
    /// First-park time, preserved across re-parks so the timeout bounds the
    /// total wait, not the time since the last attempt.
    queued_at: Instant,
}

#[derive(Debug)]
enum QueueOutcome<T> {
    /// No entry remains (the queue was empty or every front client had disconnected).
    Empty,
    /// Front entry is within its wait bound and was removed for a retry.
    Ready(QueuedAdmission<T>),
    /// Front entry waited at least `timeout` and was removed for a terminal failure.
    TimedOut(QueuedAdmission<T>, Duration),
}

impl<T> AdmissionQueue<T> {
    fn new() -> Self { Self { entries: std::collections::VecDeque::new() } }
    fn len(&self) -> usize { self.entries.len() }
    fn is_empty(&self) -> bool { self.entries.is_empty() }
    /// Park a job; `queued_at` is the first-park clock, so a re-park after a
    /// failed retry passes the original timestamp in.
    fn push(&mut self, value: T, queued_at: Instant) {
        self.entries.push_back(QueuedAdmission { value, queued_at });
    }

    /// Remove the front entry for one admission attempt, sweeping every
    /// disconnected entry out of the deque — wherever it sits — as the loop
    /// already does for fresh jobs. The sweep is whole-queue because a parked
    /// entry holds its multi-megabyte prepared prompt, so a dead client buried
    /// mid-queue must not pin memory (or image-key pins) until it reaches the
    /// front; the order of the surviving entries is preserved. The front
    /// entry is `TimedOut` once it has waited at least `timeout` (a zero
    /// timeout times out immediately, so callers treating zero as fail-fast
    /// must never park in the first place).
    fn dequeue(
        &mut self,
        timeout: Duration,
        now: Instant,
        disconnected: impl Fn(&T) -> bool,
    ) -> QueueOutcome<T> {
        self.entries.retain(|entry| !disconnected(&entry.value));
        let Some(entry) = self.entries.pop_front() else { return QueueOutcome::Empty };
        let waited = now.saturating_duration_since(entry.queued_at);
        if waited >= timeout { QueueOutcome::TimedOut(entry, waited) } else { QueueOutcome::Ready(entry) }
    }
}

/// True when a starved fresh admission may be parked instead of failed. A
/// zero wait bound is fail-fast: parking would time the entry out on the very
/// next dequeue, so the caller must send the client the pool error instead.
fn may_defer(timeout: Duration) -> bool {
    !timeout.is_zero()
}

/// True when a serve-loop iteration that ends with no active request and a
/// non-empty deferred queue must block on the job channel for a bounded
/// quantum before retrying: no decode round will free pool pages (nothing is
/// active) and the queue front was just re-parked, so `try_recv` would
/// hot-loop on a starved pool. Never true while any request is active — a
/// decode round must not be delayed — or after the channel closed (the
/// shutdown drain handles that path).
fn needs_idle_pace(no_active: bool, deferred_nonempty: bool, channel_open: bool) -> bool {
    no_active && deferred_nonempty && channel_open
}

/// Bounded channel wait for the paced idle loop: long enough to stop a spin
/// on a starved pool, short enough that a newly arrived job is admitted well
/// inside one scheduler quantum.
const IDLE_PACE: Duration = Duration::from_millis(20);

/// Terminal client-facing message for a deferred job that leaves the queue
/// without admission; the wait is measured from its first park. The same
/// suffix shapes the timeout and the shutdown-drain paths.
fn deferred_failure(error: &str, waited: Duration) -> String {
    format!(
        "{error} after {:.3} s queued for device KV pages",
        waited.as_secs_f64()
    )
}

/// Counters published beside `host_cache` in `/v1/stats`.
#[derive(Default)]
struct AdmissionStats {
    /// Jobs parked at least once; incremented on first park only.
    deferred: u64,
    /// Total queued wait of jobs that left the queue, admitted or timed out.
    deferred_ns_sum: u64,
    deferred_timeouts: u64,
    /// Fresh arrivals failed immediately because the deferred queue was at
    /// `admission_queue_max`; they never park, so they do not count as
    /// `deferred` either.
    queue_rejects: u64,
}
impl AdmissionStats {
    fn record_departure(&mut self, waited: Duration) {
        self.deferred_ns_sum = self.deferred_ns_sum.saturating_add(waited.as_nanos() as u64);
    }
}

/// Scheduler-owned keys of the `/v1/stats` payload, assembled in one place so
/// the published shape — `hc-rotate.py` among other consumers reads it — is
/// unit-testable. `host_cache` is an explicit null when the cache is off, so
/// clients can tell "cache off" from "no stats yet".
fn stats_json(
    host_cache: Option<ds41rt_hostcache::metrics::Snapshot>,
    queue_len: usize,
    admission: &AdmissionStats,
) -> serde_json::Value {
    serde_json::json!({
        "host_cache": host_cache,
        "admission_deferred": admission.deferred,
        "admission_deferred_ns_sum": admission.deferred_ns_sum,
        "admission_deferred_timeouts": admission.deferred_timeouts,
        "admission_queue_rejects": admission.queue_rejects,
        "admission_queue_len": queue_len,
    })
}

pub(super) fn serve<'w, 'a>(lib: &'a NativeLibrary, args: &crate::cli::NativeServeArgs,
    runtime: &tokio::runtime::Runtime, receive: &mut mpsc::Receiver<NativeRequest>,
    first: &mut TargetPass<'w, 'a>, second: &mut TargetPass<'w, 'a>,
    requests: &mut Requests<'a>, first_transport: &mut NativeTp4Wave<'a>,
    second_transport: &mut NativeTp4Wave<'a>, mut draft: Option<&mut DraftRuntime<'_, 'a>>,
    vision: &mut crate::v41_vision::VisionRuntime<'a>,
    stats: std::sync::Arc<std::sync::Mutex<serde_json::Value>>,
) -> Result<()> {
    let mut active: Vec<Option<Active<'a>>> = (0..args.concurrency).map(|_| None).collect();
    let mut compiler = super::constraints::Compiler::new(lib, args.snapshot.join("tokenizer.json"));
    let mut id = 0u64;
    let mut closed = false;
    let template = requests.cache().sources()[0].source_cache().page_segments(0)[0];
    let host_cache = super::prefix::HostCacheBinding::new(lib, args.host_cache_config()?, template)?;
    let mut prefixes = PrefixCache::new(args.prefix_cache_entries as usize).with_host_cache(host_cache);
    let mut stats_published = Instant::now();
    let limits = ds41rt_api::native_v41::NativeLimits::new(args.max_context_tokens, args.max_output_tokens)?;
    let admission_timeout = Duration::from_millis(args.admission_queue_timeout_ms);
    let mut deferred: AdmissionQueue<DeferredAdmission<'a>> = AdmissionQueue::new();
    let mut admission_stats = AdmissionStats::default();
    // A job pulled from the channel by the paced idle wait below; the next
    // iteration's admission phase consumes it before any channel receive.
    let mut pending: Option<NativeRequest> = None;
    loop {
        prefixes.tick();
        if stats_published.elapsed() >= std::time::Duration::from_secs(1) {
            stats_published = Instant::now();
            if let Ok(mut slot) = stats.lock() {
                *slot = stats_json(prefixes.host_metrics(), deferred.len(), &admission_stats);
            }
        }
        // This point is reached only after both complete stacks have drained and
        // committed. No cache owner is migrated or retired inside a layer stack.
        for entry in &mut active {
            if entry.as_ref().is_some_and(|r| r.finished || r.job.events.is_closed()) {
                retire_request(entry.take().unwrap(), requests, &mut prefixes, draft.as_deref_mut())?;
            }
        }
        let mut loads = [0usize; 2];
        for request in active.iter().flatten() { loads[request.lane] += 1; }
        while loads[0].abs_diff(loads[1]) > 1 {
            let heavy = usize::from(loads[1] > loads[0]);
            let request = active.iter_mut().flatten().find(|r| r.lane == heavy).unwrap();
            request.lane = 1 - heavy;
            loads[heavy] -= 1; loads[1 - heavy] += 1;
        }
        // Admit available work at a completed boundary. Prefill currently owns
        // both lanes; mixed prefill/decode interleaving is a subsequent policy.
        // Deferred admissions retry before the channel, in FIFO order; a retry
        // that starves is re-parked with its original wait clock and the loop
        // moves on to the channel, so a still-exhausted pool costs one
        // make_room pass per deferred job per loop iteration and never spins.
        'admission: while let Some(slot) = active.iter().position(Option::is_none) {
            enum Candidate<'a> { Queued(Box<QueuedAdmission<DeferredAdmission<'a>>>), Fresh(NativeRequest) }
            let mut deferred_tried = deferred.is_empty();
            'candidate: loop {
                let candidate = if !deferred_tried {
                    deferred_tried = true;
                    match deferred.dequeue(admission_timeout, Instant::now(),
                        |job| job.prepared.job.events.is_closed()) {
                        QueueOutcome::Empty => None,
                        QueueOutcome::TimedOut(job, waited) => {
                            admission_stats.record_departure(waited);
                            admission_stats.deferred_timeouts += 1;
                            let failure = deferred_failure(&job.value.error, waited);
                            let _ = job.value.prepared.job.events.blocking_send(Err(failure.clone().into()));
                            tracing::warn!(%failure, waited_s = waited.as_secs_f64(),
                                "native request admission timed out while queued");
                            continue 'admission;
                        }
                        QueueOutcome::Ready(job) => Some(Candidate::Queued(Box::new(job))),
                    }
                } else { None };
                let candidate = match candidate {
                    Some(candidate) => candidate,
                    None => {
                        // Deferred jobs never block behind a quiet channel: the
                        // blocking fetch runs only when nothing is queued.
                        let job = if let Some(job) = pending.take() {
                            Some(job)
                        } else if active.iter().all(Option::is_none) && !closed && deferred.is_empty() {
                            match receive.blocking_recv() {
                                Some(job) => Some(job), None => { closed = true; None }
                            }
                        } else {
                            match receive.try_recv() {
                                Ok(job) => Some(job),
                                Err(mpsc::error::TryRecvError::Empty) => None,
                                Err(mpsc::error::TryRecvError::Disconnected) => { closed = true; None }
                            }
                        };
                        match job {
                            Some(job) if job.events.is_closed() => continue 'admission,
                            Some(job) => Candidate::Fresh(job),
                            None => break 'admission,
                        }
                    }
                };
                id = id.checked_add(1).context("request ID exhausted")?;
                let lease = requests.admit(slot, id)?;
                let lane = usize::from(loads[1] < loads[0]);
                let (prepared, queued_at, first_error) = match candidate {
                    Candidate::Queued(job) => (job.value.prepared, Some(job.queued_at), job.value.error),
                    Candidate::Fresh(job) => {
                        let events = job.events.clone();
                        match prepare_admission(job, &mut compiler, &args.snapshot, &limits, &mut prefixes) {
                            Ok(prepared) => (prepared, None, String::new()),
                            Err(error) => {
                                // Nothing lease-bound exists yet; draft release
                                // below tolerates a never-admitted id.
                                let failure = error.downcast_ref::<ds41rt_api::native_v41::NativeFailure>()
                                    .cloned().unwrap_or_else(|| format!("{error:#}").into());
                                let _ = events.blocking_send(Err(failure));
                                tracing::warn!(%error, "native request admission failed");
                                requests.release_if_present(lease)?;
                                if let Some(draft) = draft.as_deref_mut() { draft.release(id)?; }
                                first_transport.reset_connections(); second_transport.reset_connections();
                                continue 'admission;
                            }
                        }
                    }
                };
                let events = prepared.job.events.clone();
                let result = admit_prepared(prepared, id, lease, lane, lib, runtime, first, second,
                    requests, first_transport, second_transport, draft.as_deref_mut(), vision,
                    &mut prefixes, args.prefill_batch_tokens as usize);
                match result {
                    Ok(Admission::Running(mut request)) => {
                        if let Some(queued_at) = queued_at {
                            let queued = queued_at.elapsed();
                            admission_stats.record_departure(queued);
                            tracing::info!(request_id=id, queued_ms = queued.as_millis() as u64,
                                "native request admitted after deferral");
                        }
                        if let Err(error) = request.emit(&[request.anchor]) {
                            let _ = request.job.events.blocking_send(Err(format!("{error:#}").into()));
                            request.finished = true;
                        }
                        active[slot] = Some(request); loads[lane] += 1;
                        break 'candidate;
                    }
                    Ok(Admission::Starved { prepared, error, needed, available }) => {
                        // Release exactly what the failure branch releases;
                        // deferral differs only in that the client hears nothing
                        // and the transports keep their connections.
                        requests.release_if_present(lease)?;
                        if let Some(draft) = draft.as_deref_mut() { draft.release(id)?; }
                        let (error, queued_at) = match queued_at {
                            // Re-park with the original wait clock and error.
                            // The entry was already admitted to the queue, so
                            // the cap check below (fresh arrivals only) never
                            // applies to it.
                            Some(queued_at) => (first_error, queued_at),
                            None if !may_defer(admission_timeout) => {
                                tracing::warn!(%error, "native request admission failed");
                                let _ = events.blocking_send(Err(error.into()));
                                first_transport.reset_connections(); second_transport.reset_connections();
                                continue 'admission;
                            }
                            None if deferred.len() >= args.admission_queue_max as usize => {
                                // The queue holds fully prepared prompts
                                // (~MBs each), so beyond the cap the newest
                                // arrival fails fast instead of growing RSS
                                // without a bound. See the
                                // `admission_queue_max` flag help.
                                admission_stats.queue_rejects += 1;
                                let failure =
                                    format!("{error} (admission queue full {})", deferred.len());
                                tracing::warn!(%failure, "native request admission rejected");
                                let _ = events.blocking_send(Err(failure.into()));
                                first_transport.reset_connections(); second_transport.reset_connections();
                                continue 'admission;
                            }
                            None => {
                                admission_stats.deferred += 1;
                                tracing::info!(request_id=id, needed_pages=needed,
                                    available_pages=available, "native request admission deferred");
                                (error, Instant::now())
                            }
                        };
                        deferred.push(DeferredAdmission { prepared, error }, queued_at);
                        // One attempt per deferred job per loop iteration: with
                        // the pool still full this slot falls through to the
                        // channel; a younger deferred job needs at least as
                        // many pages in practice.
                        continue 'candidate;
                    }
                    Err(error) => {
                        // Other completed requests retain their caches.
                        let failure = error.downcast_ref::<ds41rt_api::native_v41::NativeFailure>()
                            .cloned().unwrap_or_else(|| format!("{error:#}").into());
                        let _ = events.blocking_send(Err(failure));
                        tracing::warn!(%error, "native request admission failed");
                        requests.release_if_present(lease)?;
                        if let Some(draft) = draft.as_deref_mut() { draft.release(id)?; }
                        first_transport.reset_connections(); second_transport.reset_connections();
                        continue 'admission;
                    }
                }
            }
        }
        let no_active = active.iter().all(Option::is_none);
        if no_active {
            if closed {
                // No decode round will run again; fail every still-queued job
                // through the same dequeue decision, message shape and stats
                // bookkeeping as a queue timeout (a zero bound times the front
                // out immediately, so this drains the whole queue). Clients
                // are gone with the channel in practice, so this normally
                // sends nothing.
                while let QueueOutcome::TimedOut(job, waited) =
                    deferred.dequeue(Duration::ZERO, Instant::now(), |_| false)
                {
                    admission_stats.record_departure(waited);
                    admission_stats.deferred_timeouts += 1;
                    let failure = deferred_failure(&job.value.error, waited);
                    let events = &job.value.prepared.job.events;
                    let _ = events.blocking_send(Err(failure.into()));
                }
                break;
            }
            if needs_idle_pace(no_active, !deferred.is_empty(), !closed) {
                // The queue front was just re-parked and no decode round will
                // free pages, so without a wait this loop would spin on
                // `try_recv` — and, with the retained bank drained, log a
                // restore-fallback warn on every iteration. Block on the
                // channel for one bounded quantum instead: a new job arrives
                // immediately; otherwise the queue retries after at most
                // `IDLE_PACE`. Never reached while any request is active.
                match runtime.block_on(tokio::time::timeout(IDLE_PACE, receive.recv())) {
                    Ok(Some(job)) => pending = Some(job),
                    Ok(None) => closed = true,
                    Err(_) => {}
                }
            }
            continue;
        }
        let members: [Vec<usize>; 2] = std::array::from_fn(|lane| active.iter().enumerate()
            .filter_map(|(slot, request)| request.as_ref().filter(|r| r.lane == lane && !r.finished
                && !r.job.events.is_closed()).map(|_| slot)).collect());
        if members.iter().all(Vec::is_empty) { continue; }
        let capacity: Vec<_> = members.iter().flatten().map(|&slot| {
            let r = active[slot].as_ref().unwrap();
            (r.lease, if draft.is_some() { (r.job.max_tokens - r.generated).min(6) as u32 } else { 1 })
        }).collect();
        let room = prefixes.make_room(requests, &capacity);
        if let Err(error) = &room {
            if let Some(pressure) = error.downcast_ref::<crate::v41_compressor::SourcePoolExhausted>() {
                let slot = *members.iter().flatten().nth(pressure.work_index)
                    .context("pool pressure references an invalid append participant")?;
                let request = active[slot].as_mut().unwrap();
                let _ = request.job.events.blocking_send(Err(format!("{error:#}").into()));
                request.finished = true;
                request.cacheable = false;
                // No layer stack started. The next completed-boundary cleanup
                // releases only this owner's pages, then retries remaining work.
                continue;
            }
        }
        // With one active lane there is no peer to overlap. Retain the
        // ordinary C1 path and avoid shared-bank/async-delivery overhead.
        let result = room.and_then(|_| if members.iter().all(|lane| !lane.is_empty()) {
            independent::run(lib, runtime, first, second, requests, first_transport,
                second_transport, &mut active, draft.as_deref_mut(), &mut prefixes, receive)
        } else {
            let lane = usize::from(members[0].is_empty());
            let (pass, transport) = if lane == 0 { (&mut *first, &mut *first_transport) }
                else { (&mut *second, &mut *second_transport) };
            single_lane_round(lib, runtime, lane, pass, requests, transport,
                &mut active, &members[lane], draft.as_deref_mut())
        });
        if let Err(error) = result {
            first_transport.reset_connections(); second_transport.reset_connections();
            for request in active.iter_mut().flatten() {
                let _ = request.job.events.blocking_send(Err(format!("{error:#}").into()));
                request.finished = true;
                request.cacheable = false;
            }
        }
    }
    Ok(())
}

async fn execute_logits<'a>(lib: &'a NativeLibrary, pass: &mut TargetPass<'_, 'a>,
    requests: &Requests<'a>, batch: &mut Option<RequestBatch>, transport: &mut NativeTp4Wave<'a>,
    capture_routes: bool, compact: bool,
) -> Result<BatchScores> {
    let Some(batch) = batch else { return BatchScores::new(Vec::new()); };
    pass.set_route_capture(capture_routes);
    let result = async {
        let selected: Vec<_> = (0..batch.cache()?.positions().len()).collect();
        if compact {
            return BatchScores::from_greedy(unsafe { pass.execute_greedy(requests, batch, transport, 0, &selected).await? });
        }
        let logits = unsafe { pass.execute(requests, batch, transport, 0, &selected).await? };
        let mut bytes = vec![0; logits.logits.bytes];
        lib.copy_d2h(&mut bytes, logits.logits)?;
        BatchScores::new(bytes)
    }.await;
    pass.set_route_capture(false);
    result
}

fn prepare_decode_lane<'a>(requests: &mut Requests<'a>, active: &[Option<Active<'a>>],
    members: &[usize], inputs: &[Vec<u32>], speculative: bool) -> Result<RequestBatch> {
    let work: Vec<_> = members.iter().zip(inputs).map(|(&slot, tokens)| RequestTokens {
        lease: active[slot].as_ref().unwrap().lease, tokens, image_mask: None,
        kind: if speculative { ExpertV2SourceKind::MtpVerify } else { ExpertV2SourceKind::Decode },
    }).collect();
    requests.prepare(&work)
}

fn single_lane_round<'w, 'a>(lib: &'a NativeLibrary, runtime: &tokio::runtime::Runtime,
    lane: usize, pass: &mut TargetPass<'w, 'a>, requests: &mut Requests<'a>,
    transport: &mut NativeTp4Wave<'a>, active: &mut [Option<Active<'a>>],
    members: &[usize], mut draft: Option<&mut DraftRuntime<'_, 'a>>,
) -> Result<()> {
    let started = Instant::now();
    ensure!(!members.is_empty() && members.len() <= 8, "invalid single decode lane");
    let speculative = draft.is_some();
    let capture_routes = draft.as_deref().is_some_and(DraftRuntime::capture_routes);
    let seeds = members.iter().map(|&slot| {
        let r = active[slot].as_ref().unwrap();
        Ok((r.id, r.anchor, requests.cache().committed_end(r.lease)?, r.job.max_tokens-r.generated))
    }).collect::<Result<Vec<_>>>()?;
    let draft_start = Instant::now();
    let mut inputs = if let Some(draft) = draft.as_deref_mut() { draft.propose(lib, &seeds)? }
        else { seeds.iter().map(|r| vec![r.1]).collect() };
    for (&slot, input) in members.iter().zip(&mut inputs) {
        let r = active[slot].as_ref().unwrap();
        if let Some(constraint) = &r.constraint { constraint.truncate_proposal(input)?; }
        else if let Some(draft) = draft.as_deref() {
            input.truncate(draft.confidence_prefix(r.id, input.len()-1)? + 1);
        }
    }
    if members.iter().all(|&slot| active[slot].as_ref().unwrap().constraint.is_none()) {
        if let Some(draft) = draft.as_deref().filter(|d| d.reuse_enabled() || d.adaptive_enabled()) {
            let candidates: Vec<_> = members.iter().zip(&inputs).map(|(&slot, input)|
                (active[slot].as_ref().unwrap().id, lane, input.len()-1)).collect();
            let lengths = if draft.adaptive_enabled() {
                draft.select_prefixes(&candidates, draft_start.elapsed().as_micros() as u64)?
            } else { draft.select_reuse_prefixes(&candidates)? };
            if let Some(lengths) = lengths {
                for (input, length) in inputs.iter_mut().zip(lengths) { input.truncate(length+1); }
            }
        }
    }
    let draft_us = draft_start.elapsed().as_micros() as u64;
    let prepare_start = Instant::now();
    let mut batch = Some(prepare_decode_lane(requests, active, members, &inputs, speculative)?);
    let prepare_us = prepare_start.elapsed().as_micros() as u64;
    let prepared_us = started.elapsed().as_micros() as u64;
    let compact = !tracing::enabled!(target: "ds41rt::logit_trace", tracing::Level::DEBUG)
        && members.iter().all(|&slot| active[slot].as_ref().unwrap().constraint.is_none());
    let next = runtime.block_on(execute_logits(lib, pass, requests, &mut batch, transport, capture_routes, compact));
    let executed_us = started.elapsed().as_micros() as u64;
    let result = (|| -> Result<()> {
        let next = next?;
        let (accepted, emitted, emissions) = commit_lane(lib, lane, pass, requests, active, members,
            &inputs, &mut batch, &next, draft.as_deref_mut(), capture_routes, executed_us-prepared_us)?;
        for (&slot, tokens) in members.iter().zip(emissions) {
            let request = active[slot].as_mut().unwrap();
            if let Err(error) = request.emit(&tokens) {
                let _ = request.job.events.blocking_send(Err(format!("{error:#}").into()));
                request.finished = true;
            }
        }
        tracing::debug!(target: "ds41rt::timing", speculative,
            requests=members.len(), lane0=if lane == 0 { members.len() } else { 0 },
            lane1=if lane == 1 { members.len() } else { 0 },
            proposed=inputs.iter().map(|r| r.len()-1).sum::<usize>(), accepted, emitted,
            draft_us, prepare_us, verify_us=executed_us-prepared_us,
            total_us=started.elapsed().as_micros() as u64, "native scheduler round");
        Ok(())
    })();
    if result.is_err() {
        // The sole execution future has returned before discarding private state.
        if let Some(batch) = &mut batch { pass.discard(batch)?; }
    }
    result
}

struct CommitDecision {
    accepted_drafts: u32,
    emitted: usize,
    accepted: Vec<u32>,
    emissions: Vec<Vec<u32>>,
    next_after_commit: Vec<Option<TokenScores>>,
    frontier_downloads: Vec<(usize, usize)>,
}
fn prepare_commit_lane<'a>(lane: usize,
    requests: &Requests<'a>, active: &[Option<Active<'a>>], members: &[usize], inputs: &[Vec<u32>],
    next: &BatchScores, draft: Option<&DraftRuntime<'_, 'a>>, verify_us: u64,
) -> Result<CommitDecision> {
    let mut accepted_drafts = 0u32;
    let mut emitted = 0usize;
    let mut offset = 0;
    let mut accepted = Vec::new();
    let mut emissions = Vec::new();
    let mut next_after_commit = Vec::new();
    let mut frontier_downloads = Vec::new();
    for (&slot, input) in members.iter().zip(inputs) {
        let request = active[slot].as_ref().unwrap();
        let constrained = request.constraint.as_ref().map(|state|
            state.select_verification(next, offset, input)).transpose()?;
        let selected = constrained.as_deref().unwrap_or(&next.best[offset..offset + input.len()]);
        let decision = ds41rt_core::verify_dspark_greedy(input,
            selected, 1, request.job.max_tokens - request.generated)
            .map_err(anyhow::Error::msg)?;
        if tracing::enabled!(target: "ds41rt::logit_trace", tracing::Level::DEBUG) {
            let top_two = (offset..offset + input.len())
                .map(|row| next.top_two(row)).collect::<Result<Vec<_>>>()?;
            tracing::debug!(target: "ds41rt::logit_trace",
                request_id=request.id, lane, generated=request.generated,
                context_tokens=requests.cache().committed_end(request.lease)?,
                input=?input, selected=?selected, top_two=?top_two,
                accepted_inputs=decision.accepted_inputs,
                emitted=?decision.emitted,
                constrained=request.constraint.is_some(),
                "native verification logits");
        }
        if let Some(confidence) = draft
            .and_then(|draft| draft.confidence_trace(request.id)) {
            // Agreement after the first mismatch is conditional on a
            // rejected history and must not be treated as acceptance.
            let matched = input.iter().skip(1).zip(selected.iter())
                .take_while(|(proposal, target)| proposal == target).count();
            tracing::debug!(target: "ds41rt::draft_policy",
                request_id=request.id, lane, generated=request.generated,
                context_tokens=requests.cache().committed_end(request.lease)?,
                verifier_rows=input.len(), lane_rows=inputs.iter().map(Vec::len).sum::<usize>(),
                constrained=request.constraint.is_some(), raw_confidence=?confidence,
                matched_prefix=matched, accepted_inputs=decision.accepted_inputs,
                eos=decision.eos, length_limit=decision.length_limit,
                verify_us,
                "native draft policy observation");
        }
        accepted_drafts += decision.accepted_inputs - 1;
        emitted += decision.emitted.len();
        let finishing = decision.emitted.contains(&1)
            || request.generated + decision.emitted.len() >= request.job.max_tokens;
        let frontier = offset + decision.accepted_inputs as usize - 1;
        next_after_commit.push(if finishing && next.has_full_logits() {
            Some(next.retain(frontier)?)
        } else {
            if finishing { frontier_downloads.push((next_after_commit.len(), frontier)); }
            None
        });
        offset += input.len(); accepted.push(decision.accepted_inputs); emissions.push(decision.emitted);
    }
    Ok(CommitDecision { accepted_drafts, emitted, accepted, emissions, next_after_commit, frontier_downloads })
}
fn publish_commit_lane<'a>(pass: &TargetPass<'_, 'a>, active: &mut [Option<Active<'a>>],
    members: &[usize], inputs: &[Vec<u32>], owned_batch: &mut Option<RequestBatch>,
    mut draft: Option<&mut DraftRuntime<'_, 'a>>, capture_routes: bool, decision: CommitDecision,
) -> Result<(u32, usize, Vec<Vec<u32>>)> {
    let CommitDecision { accepted_drafts, emitted, accepted, emissions, next_after_commit, frontier_downloads } = decision;
    ensure!(frontier_downloads.is_empty(), "retained frontier downloads are incomplete");
    if let Some(draft) = draft.as_deref_mut() {
        if capture_routes {
            let mut offset = 0;
            for ((&slot, input), &count) in members.iter().zip(inputs).zip(&accepted) {
                draft.observe_accepted_routes(active[slot].as_ref().unwrap().id, offset,
                    count as usize, pass.captured_routes())?;
                offset += input.len();
            }
        }
    }
    *owned_batch = None;
    for (&slot, next_token) in members.iter().zip(next_after_commit) {
        active[slot].as_mut().unwrap().next_after_commit = next_token;
    }
    Ok((accepted_drafts, emitted, emissions))
}
fn commit_lane<'w, 'a>(lib: &'a NativeLibrary, lane: usize, pass: &mut TargetPass<'w, 'a>, requests: &mut Requests<'a>,
    active: &mut [Option<Active<'a>>], members: &[usize], inputs: &[Vec<u32>],
    owned_batch: &mut Option<RequestBatch>, next: &BatchScores,
    mut draft: Option<&mut DraftRuntime<'_, 'a>>, capture_routes: bool, verify_us: u64,
) -> Result<(u32, usize, Vec<Vec<u32>>)> {
    let Some(batch) = owned_batch else { return Ok((0, 0, Vec::new())); };
    let mut decision = prepare_commit_lane(lane, requests, active, members, inputs,
        next, draft.as_deref(), verify_us)?;
    for (member, row) in decision.frontier_downloads.drain(..) {
        decision.next_after_commit[member] = Some(next.retain_from_device(lib, pass.output(batch)?.logits, row)?);
    }
    if let Some(draft) = draft.as_deref_mut() { draft.commit_batch(pass, requests, batch, &decision.accepted)?; }
    else { pass.commit(requests, batch, &decision.accepted)?; }
    publish_commit_lane(pass, active, members, inputs, owned_batch, draft, capture_routes, decision)
}

#[cfg(test)]
mod admission_queue_tests {
    use super::*;

    type Probe = (u32, bool);
    const LIVE: fn(&Probe) -> bool = |(_, gone)| *gone;

    #[test]
    fn empty_queue_reports_empty() {
        let mut queue = AdmissionQueue::new();
        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);
        assert!(matches!(queue.dequeue(Duration::from_secs(1), Instant::now(), LIVE), QueueOutcome::Empty));
    }

    #[test]
    fn push_len_and_fifo_order() {
        let mut queue = AdmissionQueue::new();
        let now = Instant::now();
        for id in 0..3 { queue.push((id, false), now); }
        assert_eq!(queue.len(), 3);
        for id in 0..3 {
            match queue.dequeue(Duration::from_secs(3600), Instant::now(), LIVE) {
                QueueOutcome::Ready(entry) => assert_eq!(entry.value.0, id),
                other => panic!("expected entry {id}, got {other:?}"),
            }
        }
        assert!(queue.is_empty());
    }

    #[test]
    fn disconnected_front_is_dropped_silently() {
        let mut queue = AdmissionQueue::new();
        let now = Instant::now();
        queue.push((0, true), now);
        queue.push((1, false), now);
        queue.push((2, false), now);
        match queue.dequeue(Duration::from_secs(3600), Instant::now(), LIVE) {
            QueueOutcome::Ready(entry) => assert_eq!(entry.value.0, 1),
            other => panic!("expected entry 1, got {other:?}"),
        }
        assert_eq!(queue.len(), 1);
    }

    #[test]
    fn consecutive_disconnected_fronts_are_all_dropped() {
        let mut queue = AdmissionQueue::new();
        let now = Instant::now();
        queue.push((0, true), now);
        queue.push((1, true), now);
        queue.push((2, true), now);
        queue.push((3, false), now);
        match queue.dequeue(Duration::from_secs(3600), Instant::now(), LIVE) {
            QueueOutcome::Ready(entry) => assert_eq!(entry.value.0, 3),
            other => panic!("expected entry 3, got {other:?}"),
        }
        assert!(queue.is_empty());
    }

    #[test]
    fn disconnected_entry_buried_mid_queue_is_dropped_with_order_kept() {
        let mut queue = AdmissionQueue::new();
        let now = Instant::now();
        queue.push((0, false), now);
        queue.push((1, true), now);
        queue.push((2, false), now);
        // First call dequeues the live front; the buried dead entry must not
        // survive to the front (it would pin its prepared prompt for the
        // whole wait), and the live tail keeps its place.
        match queue.dequeue(Duration::from_secs(3600), Instant::now(), LIVE) {
            QueueOutcome::Ready(entry) => assert_eq!(entry.value.0, 0),
            other => panic!("expected entry 0, got {other:?}"),
        }
        match queue.dequeue(Duration::from_secs(3600), Instant::now(), LIVE) {
            QueueOutcome::Ready(entry) => assert_eq!(entry.value.0, 2),
            other => panic!("expected entry 2, got {other:?}"),
        }
        assert!(queue.is_empty());
    }

    #[test]
    fn zero_timeout_never_parks() {
        // The fail-fast invariant behind the serve loop's starved branch: a
        // zero bound must send the client the pool error instead of parking,
        // because a parked entry would time out on the very next dequeue.
        assert!(!may_defer(Duration::ZERO));
        assert!(may_defer(Duration::from_millis(1)));
        assert!(may_defer(Duration::from_secs(300)));
    }

    #[test]
    fn idle_pace_only_when_idle_starved_and_open() {
        // Pacing decision behind the serve loop's bounded channel wait.
        assert!(needs_idle_pace(true, true, true));
        assert!(
            !needs_idle_pace(false, true, true),
            "decode must not be delayed"
        );
        assert!(
            !needs_idle_pace(true, false, true),
            "blocking_recv paces the empty queue"
        );
        assert!(
            !needs_idle_pace(true, true, false),
            "the shutdown drain handles a closed channel"
        );
    }

    #[test]
    fn zero_timeout_times_out_front_and_keeps_order() {
        let mut queue = AdmissionQueue::new();
        let now = Instant::now();
        queue.push((0, false), now);
        queue.push((1, false), now);
        match queue.dequeue(Duration::ZERO, now, LIVE) {
            QueueOutcome::TimedOut(entry, _) => assert_eq!(entry.value.0, 0),
            other => panic!("expected timeout for entry 0, got {other:?}"),
        }
        // The timed-out entry is gone; later entries stay queued in order.
        match queue.dequeue(Duration::ZERO, now, LIVE) {
            QueueOutcome::TimedOut(entry, _) => assert_eq!(entry.value.0, 1),
            other => panic!("expected timeout for entry 1, got {other:?}"),
        }
        assert!(queue.is_empty());
    }

    #[test]
    fn within_bound_entry_is_ready() {
        let mut queue = AdmissionQueue::new();
        let now = Instant::now();
        queue.push((0, false), now);
        match queue.dequeue(Duration::from_secs(3600), now, LIVE) {
            QueueOutcome::Ready(entry) => assert_eq!(entry.value.0, 0),
            other => panic!("expected ready entry, got {other:?}"),
        }
    }

    #[test]
    fn reparked_entry_keeps_its_original_wait_clock() {
        let mut queue = AdmissionQueue::new();
        let parked_at = Instant::now();
        queue.push((0, false), parked_at);
        // A retry attempt dequeues the job and, still starved, re-parks it with
        // the original first-park timestamp behind whatever arrived meanwhile.
        let attempted_at = parked_at + Duration::from_secs(1);
        let attempt = match queue.dequeue(Duration::from_secs(3600), attempted_at, LIVE) {
            QueueOutcome::Ready(entry) => entry,
            other => panic!("expected ready entry, got {other:?}"),
        };
        assert_eq!(attempt.queued_at, parked_at);
        queue.push((1, false), attempted_at);
        queue.push(attempt.value, attempt.queued_at);
        // At a two-second bound, the fresher job (one second old at the first
        // call) times out after another second, while the re-parked job — two
        // seconds old already and queued behind it — times out on the following
        // call with its wait measured from first park, not from the re-park.
        let later = attempted_at + Duration::from_secs(2);
        match queue.dequeue(Duration::from_secs(2), later, LIVE) {
            QueueOutcome::TimedOut(entry, waited) => {
                assert_eq!(entry.value.0, 1);
                assert_eq!(waited, Duration::from_secs(2));
            }
            other => panic!("expected timeout for entry 1, got {other:?}"),
        }
        match queue.dequeue(Duration::from_secs(2), later, LIVE) {
            QueueOutcome::TimedOut(entry, waited) => {
                assert_eq!(entry.value.0, 0);
                assert_eq!(entry.queued_at, parked_at);
                assert_eq!(waited, Duration::from_secs(3));
            }
            other => panic!("expected timeout for entry 0, got {other:?}"),
        }
    }

    #[test]
    fn reparked_entry_goes_to_the_back() {
        // A starved retry re-parks behind whatever arrived since its first
        // park; the next dequeue must serve the newer job first.
        let mut queue = AdmissionQueue::new();
        let parked_at = Instant::now();
        queue.push((0, false), parked_at);
        let attempted_at = parked_at + Duration::from_secs(1);
        let attempt = match queue.dequeue(Duration::from_secs(3600), attempted_at, LIVE) {
            QueueOutcome::Ready(entry) => entry,
            other => panic!("expected ready entry, got {other:?}"),
        };
        queue.push((1, false), attempted_at);
        queue.push(attempt.value, attempt.queued_at);
        match queue.dequeue(Duration::from_secs(3600), attempted_at, LIVE) {
            QueueOutcome::Ready(entry) => assert_eq!(entry.value.0, 1),
            other => panic!("expected entry 1 ahead of the re-parked entry 0, got {other:?}"),
        }
        match queue.dequeue(Duration::from_secs(3600), attempted_at, LIVE) {
            QueueOutcome::Ready(entry) => assert_eq!(entry.value.0, 0),
            other => panic!("expected re-parked entry 0, got {other:?}"),
        }
        assert!(queue.is_empty());
    }

    #[test]
    fn zero_timeout_drains_in_fifo_order() {
        // The shutdown drain dequeues with a zero bound, which times the front
        // out immediately; the whole queue must come out in FIFO order.
        let mut queue = AdmissionQueue::new();
        let now = Instant::now();
        queue.push((0, false), now);
        queue.push((1, false), now);
        for id in 0..2 {
            match queue.dequeue(Duration::ZERO, now, LIVE) {
                QueueOutcome::TimedOut(entry, _) => assert_eq!(entry.value.0, id),
                other => panic!("expected entry {id} to drain, got {other:?}"),
            }
        }
        assert!(matches!(
            queue.dequeue(Duration::ZERO, now, LIVE),
            QueueOutcome::Empty
        ));
    }
}

#[cfg(test)]
mod admission_stats_tests {
    use super::*;

    #[test]
    fn stats_json_publishes_null_host_cache_and_every_counter() {
        // Shape pinned for the consumers of /v1/stats (hc-rotate.py among
        // them): `host_cache` is an explicit null when the cache is off, and
        // every admission counter — including queue rejects — is present.
        let stats = AdmissionStats {
            deferred: 1,
            deferred_ns_sum: 2,
            deferred_timeouts: 3,
            queue_rejects: 4,
        };
        let value = stats_json(None, 5, &stats);
        assert_eq!(value["host_cache"], serde_json::Value::Null);
        assert_eq!(value["admission_deferred"], 1);
        assert_eq!(value["admission_deferred_ns_sum"], 2);
        assert_eq!(value["admission_deferred_timeouts"], 3);
        assert_eq!(value["admission_queue_rejects"], 4);
        assert_eq!(value["admission_queue_len"], 5);
    }
}
