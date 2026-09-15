//! Official V4.1 protocol conversion and bounded handoff to a CUDA owner.
use axum::{
    body::Body,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
pub use deepseek_recipe::stream::{InferenceChunk, InferenceFinishReason, PromptUsage};
use deepseek_recipe::{
    openai::ChatCompletionRequest,
    request::{ConversionOptions, ProtocolRequest},
    response::ProtocolResponse,
    stream::StreamProcessor,
    util::append_delta::AppendDelta,
};
use deepseek_recipe_encoding::{v4::dsv41::DeepseekV41Encoding, PromptEncoding};
use futures::StreamExt;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::mpsc;

pub const MODEL: &str = "deepseek-ai/DeepSeek-V4.1-Flash";
mod limits;
mod constraints;
mod tools;
pub use constraints::NativeConstraint;
mod images;
#[cfg(test)]
mod unicode_tests;
pub use limits::{NativeLimits, MAX_CONTEXT_TOKENS, MAX_OUTPUT_TOKENS};
#[derive(Debug, Clone)]
pub enum NativeFailure {
    BadRequest(String),
    Worker(String),
}
impl std::fmt::Display for NativeFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self { Self::BadRequest(message) | Self::Worker(message) => f.write_str(message) }
    }
}
impl std::error::Error for NativeFailure {}
impl From<String> for NativeFailure {
    fn from(message: String) -> Self { Self::Worker(message) }
}
impl From<&str> for NativeFailure {
    fn from(message: &str) -> Self { Self::Worker(message.into()) }
}
pub struct NativeRequest {
    pub prompt: String,
    pub constraint: Option<NativeConstraint>,
    pub images: Vec<ds41rt_loader::V41Image>,
    pub max_tokens: usize,
    pub events: mpsc::Sender<Result<InferenceChunk, NativeFailure>>,
}
/// Serving statistics the CUDA owner publishes (a JSON object; `null` until the first publish).
pub type SharedStats = Arc<Mutex<Value>>;
/// HTTP queue admission policy: how long a submission may wait for a free slot before the
/// front door answers 429, and the `Retry-After` hint (whole seconds) sent with that answer.
/// `wait == 0` keeps the immediate `try_send` behaviour (no waiting), but a full queue still
/// answers 429 with `Retry-After`; only a closed channel earns 503.
///
/// Fairness note: every submission attempts an immediate `try_send` before it will wait, so
/// a fresh arrival can leapfrog requests already parked in the send-waiter FIFO (the waiters
/// are FIFO only among themselves). That is deliberate: the fast path never queues behind
/// parked requests, the leapfrog window is a single channel slot, and the bounded wait still
/// expires deterministically.
#[derive(Debug, Clone, Copy)]
pub struct QueuePolicy {
    /// How long a submission may wait for a free queue slot before the front door answers
    /// 429; `Duration::ZERO` fails immediately (the old try-send behaviour, but a full queue
    /// still answers 429, not 503).
    pub wait: Duration,
    /// `Retry-After` hint for 429 queue-full answers, in whole seconds. The header rounds any
    /// positive duration up to at least one second, so even a sub-second value yields `1`
    /// rather than a useless `Retry-After: 0` ("retry immediately").
    pub retry_after: Duration,
}
impl Default for QueuePolicy {
    fn default() -> Self {
        Self {
            wait: Duration::ZERO,
            retry_after: Duration::from_secs(2),
        }
    }
}
/// Atomic counters for the HTTP admission queue, merged into `/v1/stats` on every read.
#[derive(Debug, Default)]
struct HttpQueueMetrics {
    /// Submissions that found the queue full and waited for a slot.
    waits: AtomicU64,
    /// Total milliseconds spent waiting for a slot (accepted and rejected waits).
    wait_ms_sum: AtomicU64,
    /// Submissions refused with 429 because the queue stayed full.
    rejects_429: AtomicU64,
}
/// HTTP job queue capacity in jobs. `None` (the flag default) is `4 × concurrency`;
/// `Some(0)` preserves the pre-HC-13 size of exactly `concurrency` jobs.
pub fn http_queue_depth(concurrency: u32, depth: Option<u32>) -> usize {
    match depth {
        None => 4 * concurrency as usize,
        Some(0) => concurrency as usize,
        Some(depth) => depth as usize,
    }
}
#[derive(Clone)]
struct NativeState {
    queue: mpsc::Sender<NativeRequest>,
    limits: NativeLimits,
    images: images::ImageDecoder,
    stats: SharedStats,
    policy: QueuePolicy,
    metrics: Arc<HttpQueueMetrics>,
}
/// Where a queue submission ended up: delivered, refused after waiting, or the channel closed.
#[derive(Debug)]
enum Submission {
    Accepted,
    Full { waited_ms: u64 },
    Closed,
}
impl NativeState {
    /// Hand `job` to the CUDA owner under the policy. A full queue either waits up to
    /// `policy.wait` for a slot or is refused immediately; a closed channel is reported
    /// separately so the caller can keep 503 for shutdown only.
    ///
    /// The immediate `try_send` prefers fresh arrivals over already-parked waiters (see
    /// [`QueuePolicy`]'s fairness note). Cancellation-safety: the bounded `send` is a
    /// `reserve().await` + `permit.send`, so a future dropped by the timeout or by a
    /// disconnected client never takes a slot and never delivers its job.
    async fn submit(&self, job: NativeRequest) -> Submission {
        use mpsc::error::TrySendError;
        let job = match self.queue.try_send(job) {
            Ok(()) => return Submission::Accepted,
            Err(TrySendError::Closed(_)) => return Submission::Closed,
            Err(TrySendError::Full(job)) => job,
        };
        if self.policy.wait.is_zero() {
            self.metrics.rejects_429.fetch_add(1, Ordering::Relaxed);
            return Submission::Full { waited_ms: 0 };
        }
        self.metrics.waits.fetch_add(1, Ordering::Relaxed);
        let started = std::time::Instant::now();
        match tokio::time::timeout(self.policy.wait, self.queue.send(job)).await {
            Ok(Ok(())) => {
                self.metrics
                    .wait_ms_sum
                    .fetch_add(started.elapsed().as_millis() as u64, Ordering::Relaxed);
                Submission::Accepted
            }
            Ok(Err(_)) => Submission::Closed,
            Err(_) => {
                let waited_ms = started.elapsed().as_millis() as u64;
                self.metrics.rejects_429.fetch_add(1, Ordering::Relaxed);
                self.metrics
                    .wait_ms_sum
                    .fetch_add(waited_ms, Ordering::Relaxed);
                Submission::Full { waited_ms }
            }
        }
    }
}
pub fn router(queue: mpsc::Sender<NativeRequest>) -> Router {
    router_with_limits(queue, NativeLimits::default())
}
pub fn router_with_limits(queue: mpsc::Sender<NativeRequest>, limits: NativeLimits) -> Router {
    router_with_limits_and_stats(queue, limits, Arc::new(Mutex::new(Value::Null)))
}
pub fn router_with_limits_and_stats(queue: mpsc::Sender<NativeRequest>, limits: NativeLimits, stats: SharedStats) -> Router {
    router_with_limits_stats_and_policy(queue, limits, stats, QueuePolicy::default())
}
/// Serve with an explicit HTTP queue admission policy (see [`QueuePolicy`]).
pub fn router_with_limits_stats_and_policy(
    queue: mpsc::Sender<NativeRequest>,
    limits: NativeLimits,
    stats: SharedStats,
    policy: QueuePolicy,
) -> Router {
    let images = images::ImageDecoder::new(queue.max_capacity());
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/stats", get(stats_route))
        .route("/v1/chat/completions", post(chat))
        .layer(axum::extract::DefaultBodyLimit::max(images::BODY_BYTES))
        .with_state(NativeState {
            queue,
            limits,
            images,
            stats,
            policy,
            metrics: Arc::default(),
        })
}
async fn stats_route(State(state): State<NativeState>) -> Json<Value> {
    let mut value = match state.stats.lock() {
        Ok(stats) if stats.is_object() => stats.clone(),
        _ => json!({}),
    };
    let Some(stats) = value.as_object_mut() else {
        return Json(value);
    };
    stats.insert(
        "http_queue_waits".into(),
        json!(state.metrics.waits.load(Ordering::Relaxed)),
    );
    stats.insert(
        "http_queue_wait_ms_sum".into(),
        json!(state.metrics.wait_ms_sum.load(Ordering::Relaxed)),
    );
    stats.insert(
        "http_queue_rejects_429".into(),
        json!(state.metrics.rejects_429.load(Ordering::Relaxed)),
    );
    // tokio's bounded Sender has no `len()`; occupancy is the complement of its live capacity.
    stats.insert(
        "http_queue_len".into(),
        json!(state.queue.max_capacity() - state.queue.capacity()),
    );
    Json(value)
}
async fn models(State(state): State<NativeState>) -> Json<Value> {
    Json(json!({"object":"list","data":[{"id":MODEL,"object":"model","owned_by":"deepseek-ai",
        "max_context_tokens":state.limits.context(),"max_output_tokens":state.limits.output()}]}))
}
async fn health(State(state): State<NativeState>) -> StatusCode {
    if state.queue.is_closed() {
        StatusCode::SERVICE_UNAVAILABLE
    } else {
        StatusCode::OK
    }
}
fn error(status: StatusCode, message: impl ToString) -> Response {
    (
        status,
        Json(json!({"error":{"message":message.to_string(),"type":"native_v41_error"}})),
    )
        .into_response()
}
/// 429 for a queue that stayed full: the JSON error body plus a `Retry-After` hint so
/// bursting clients back off instead of reading the refusal as backend failure.
fn queue_full(state: &NativeState, waited_ms: u64) -> Response {
    let message = format!(
        "request queue full ({}) after {} ms",
        state.queue.max_capacity(),
        waited_ms
    );
    let mut response = error(StatusCode::TOO_MANY_REQUESTS, message);
    // Ceil to whole seconds: any positive duration must hint at least one second, because
    // `Retry-After: 0` means "retry immediately" and would defeat the hint.
    let retry_after_secs =
        state.policy.retry_after.as_secs() + u64::from(state.policy.retry_after.subsec_nanos() > 0);
    if let Ok(retry_after) = retry_after_secs.to_string().parse() {
        response
            .headers_mut()
            .insert(axum::http::header::RETRY_AFTER, retry_after);
    }
    response
}
async fn chat(State(state): State<NativeState>, Json(mut body): Json<Value>) -> Response {
    let assistance = match body.get("tool_decoding_assistance") {
        None | Some(Value::Null) => true,
        Some(Value::Bool(value)) => *value,
        _ => return error(StatusCode::BAD_REQUEST, "tool_decoding_assistance must be boolean"),
    };
    let response_format = body.get("response_format").cloned().filter(|v| !v.is_null());
    // The recipe rejects its regex variant, while native XGrammar supports it.
    // Keep the original format for enforcement and render it as ordinary text.
    if response_format.as_ref().and_then(|v| v.get("type")).and_then(Value::as_str) == Some("regex") {
        body["response_format"] = json!({"type":"text"});
    }
    let mut parsed: ChatCompletionRequest = match serde_json::from_value(body) {
        Ok(r) => r,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let include_usage = parsed.include_usage();
    let parallel = parsed.parallel_tool_calls.unwrap_or(true);
    let selection = tools::Selection::extract(&mut parsed);
    // Native serving defaults to thinking at the adapter's high effort. Explicit
    // thinking/effort settings retain the official conversion precedence.
    let mut converted = match parsed.convert(ConversionOptions::default().with_default_thinking_mode(true)) {
        Ok(r) => r,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    if let Err(e) = selection.apply(&mut converted.conversation.tools) {
        return error(StatusCode::BAD_REQUEST, e);
    }
    if converted.model.as_deref() != Some(MODEL) {
        return error(StatusCode::BAD_REQUEST, format!("model must be {MODEL}"));
    }
    if converted.inference_options.temperature.unwrap_or(0.0) != 0.0 {
        return error(
            StatusCode::BAD_REQUEST,
            "native target sampling currently requires temperature=0",
        );
    }
    let max_tokens = match state.limits.requested_output(converted.inference_options.max_tokens) {
        Ok(limit) => limit,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let response_validator = match constraints::response_validator(response_format.as_ref()) {
        Ok(validator) => validator,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let tool_constraints = match tools::ToolConstraints::new(&converted.conversation.tools,
        converted.conversation.tool_choice, selection.required, parallel,
        assistance || response_format.as_ref().is_some_and(|v| v["type"] != "text")) {
        Ok(tools) => tools,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let constraint = match constraints::response_constraint(response_format, converted.conversation.thinking_mode, tool_constraints.as_ref()) {
        Ok(constraint) => constraint,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let mut validator = tools::CompletionValidator::new(response_validator, tool_constraints);
    let rendered = DeepseekV41Encoding::new().render_conversation(&converted.conversation);
    let streaming = converted.stream;
    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let generator = ChatCompletionRequest::chunk_generator(&converted, id.clone(), MODEL.into())
        .with_include_usage(!streaming || include_usage);
    let processor = StreamProcessor::new(generator, converted.parsing_options);
    // Rendered sources own the image payloads needed by preprocessing. Do not
    // retain another copy of their data URLs throughout the generated response.
    drop(converted.conversation);
    let (prepared, permit) = if rendered.image_sources.is_empty() { (Vec::new(), None) } else {
        if rendered.image_sources.len() > ds41rt_loader::V41_MAX_IMAGES {
            return error(StatusCode::BAD_REQUEST, "at most 16 images are supported");
        }
        let permit = match state.queue.clone().try_reserve_owned() {
            Ok(permit) => permit,
            Err(e) => return error(StatusCode::SERVICE_UNAVAILABLE, e),
        };
        // The queue permit bounds waiters while up to four decoders run. A C16
        // burst should wait here instead of imposing a hidden C4 image limit.
        let slot = match state.images.slots.clone().acquire_owned().await {
            Ok(slot) => slot,
            Err(_) => return error(StatusCode::SERVICE_UNAVAILABLE, "image preparation is closed"),
        };
        let decoder = state.images.clone();
        let result = tokio::task::spawn_blocking(move || {
            let _slot = slot;
            decoder.decode(rendered.image_sources)
        }).await;
        match result {
            Ok(Ok(images)) => (images, Some(permit)),
            Ok(Err(e)) => return error(StatusCode::BAD_REQUEST, format!("{e:#}")),
            Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, e),
        }
    };
    let (events, mut receive) = mpsc::channel(16);
    // Recipe 0.1.0 uses a protocol placeholder; the pinned model tokenizer
    // spells token 129264 differently. Preserve the text-only prompt verbatim.
    let prompt = if prepared.is_empty() { rendered.prompt }
        else { rendered.prompt.replace("<｜image｜>", "<｜deepseek_image｜>") };
    let job = NativeRequest {
        prompt,
        constraint,
        images: prepared,
        max_tokens,
        events,
    };
    if let Some(permit) = permit {
        permit.send(job);
    } else {
        match state.submit(job).await {
            Submission::Accepted => {}
            Submission::Full { waited_ms } => return queue_full(&state, waited_ms),
            Submission::Closed => {
                return error(StatusCode::SERVICE_UNAVAILABLE, "request queue is closed")
            }
        }
    }
    // Admission errors must retain their cause and HTTP status, including for
    // SSE, before a protocol processor can turn early EOF into a finish chunk.
    let first = match receive.recv().await {
        Some(Ok(chunk)) => chunk,
        Some(Err(NativeFailure::BadRequest(message))) => return error(StatusCode::BAD_REQUEST, message),
        Some(Err(message)) => return error(StatusCode::INTERNAL_SERVER_ERROR, message),
        None => return error(StatusCode::INTERNAL_SERVER_ERROR, "native worker ended without completion"),
    };
    // Never let a failed/disconnected backend be converted to a successful EOF.
    let failure = Arc::new(Mutex::new(None::<String>));
    let input_failure = failure.clone();
    let input = async_stream::stream! {
        let mut finished = matches!(first, InferenceChunk::Finish { .. });
        yield first;
        while !finished {
            let Some(event) = receive.recv().await else { break; };
            match event {
                Ok(chunk) => {
                    finished = matches!(chunk,InferenceChunk::Finish { .. });
                    yield chunk;
                    if finished { break; }
                }
                Err(message) => { *input_failure.lock().unwrap() = Some(message.to_string()); break; }
            }
        }
        if !finished {
            input_failure.lock().unwrap().get_or_insert_with(|| "native worker ended without completion".into());
        }
    };
    let chunks = processor.process(input);
    let chunks = async_stream::stream! {
        futures::pin_mut!(chunks);
        while let Some(chunk) = chunks.next().await {
            match chunk {
                Ok(chunk) => {
                    if validator.enabled() {
                        if let Err(e) = validator.observe(&serde_json::to_value(&chunk).unwrap()) {
                            yield Err(e); return;
                        }
                    }
                    yield Ok(chunk);
                }
                Err(e) => { yield Err(anyhow::anyhow!(e.to_string())); return; }
            }
        }
    };
    if streaming {
        let stream = async_stream::stream! {
            futures::pin_mut!(chunks);
            while let Some(chunk) = chunks.next().await {
                let failed = failure.lock().unwrap().clone();
                if let Some(message) = failed {
                    yield Err::<String,std::io::Error>(std::io::Error::other(message)); return;
                }
                match chunk {
                    Ok(chunk) => {
                        yield Ok(format!("data: {}\n\n",serde_json::to_string(&chunk).unwrap()));
                    },
                    Err(e) => { yield Err(std::io::Error::other(e.to_string())); return; }
                }
            }
            let failed = failure.lock().unwrap().clone();
            if let Some(message) = failed { yield Err(std::io::Error::other(message)); return; }
            yield Ok("data: [DONE]\n\n".to_owned());
        };
        return (
            [
                ("content-type", "text/event-stream"),
                ("cache-control", "no-cache"),
            ],
            Body::from_stream(stream),
        )
            .into_response();
    }
    type ChatResponse = <ChatCompletionRequest as ProtocolRequest>::Response;
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let mut response = ChatResponse::new(id, MODEL.into(), created, 0, 0);
    futures::pin_mut!(chunks);
    while let Some(chunk) = chunks.next().await {
        match chunk {
            Ok(chunk) => response.append(chunk),
            Err(e) => {
                let message = failure.lock().unwrap().clone().unwrap_or_else(|| e.to_string());
                return error(StatusCode::INTERNAL_SERVER_ERROR, message);
            }
        }
    }
    if let Some(message) = failure.lock().unwrap().clone() {
        return error(StatusCode::INTERNAL_SERVER_ERROR, message);
    }
    Json(response).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tower::ServiceExt;
    fn request(stream: bool) -> axum::http::Request<Body> {
        axum::http::Request::post("/v1/chat/completions").header("content-type","application/json").body(Body::from(json!({"model":MODEL,"messages":[{"role":"user","content":"What is 2 + 2? Answer with just the number."}],"thinking":{"type":"disabled"},"temperature":0,"max_tokens":16,"stream":stream}).to_string())).unwrap()
    }
    #[tokio::test]
    async fn output_limits_reach_worker_and_model_metadata() {
        for (limits, requested, expected) in [
            (NativeLimits::default(), None, 393_216),
            (NativeLimits::default(), Some(8192), 8192),
            (NativeLimits::default(), Some(u32::MAX), 393_216),
            (NativeLimits::new(256, 128).unwrap(), None, 128),
            (NativeLimits::new(256, 128).unwrap(), Some(8192), 128),
            (NativeLimits::new(256, 128).unwrap(), Some(8), 8),
        ] {
            let (tx, mut rx) = mpsc::channel::<NativeRequest>(1);
            let app = router_with_limits(tx, limits);
            let response = app.clone().oneshot(axum::http::Request::get("/v1/models")
                .body(Body::empty()).unwrap()).await.unwrap();
            let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
            let value: Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(value["data"][0]["max_context_tokens"], limits.context());
            assert_eq!(value["data"][0]["max_output_tokens"], limits.output());
            let worker = tokio::spawn(async move {
                let job = rx.recv().await.unwrap();
                assert_eq!(job.max_tokens, expected);
                job.events.send(Ok(InferenceChunk::Ready { system_fingerprint: None,
                    prompt_usage: PromptUsage { prompt_tokens: 1, prompt_cache_hit_tokens: 0 } })).await.unwrap();
                job.events.send(Ok(InferenceChunk::Finish {
                    finish_reason: InferenceFinishReason::Length,
                })).await.unwrap();
            });
            let body = json!({"model":MODEL,"messages":[{"role":"user","content":"Count."}],
                "max_tokens":requested});
            let response = app.oneshot(axum::http::Request::post("/v1/chat/completions")
                .header("content-type", "application/json").body(Body::from(body.to_string())).unwrap()).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            worker.await.unwrap();
        }
        let (tx, mut rx) = mpsc::channel::<NativeRequest>(1);
        let body = json!({"model":MODEL,"messages":[{"role":"user","content":"Count."}],"max_tokens":0});
        let response = router(tx).oneshot(axum::http::Request::post("/v1/chat/completions")
            .header("content-type", "application/json").body(Body::from(body.to_string())).unwrap()).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert!(rx.try_recv().is_err());
    }
    #[tokio::test]
    async fn thinking_defaults_high_and_honors_explicit_overrides() {
        for (options, score) in [
            (json!({}), Some(75)),
            (json!({"thinking":{"type":"enabled"}}), Some(75)),
            (json!({"reasoning_effort":"low"}), Some(50)),
            (json!({"reasoning_effort":"high"}), Some(75)),
            (json!({"reasoning_effort":"max"}), Some(100)),
            (json!({"reasoning_effort":"none"}), None),
            (json!({"thinking":{"type":"disabled"},"reasoning_effort":"max"}), None),
        ] {
            let (tx, mut rx) = mpsc::channel::<NativeRequest>(1);
            let worker = tokio::spawn(async move {
                let job = rx.recv().await.unwrap();
                if let Some(score) = score {
                    assert!(job.prompt.contains(&format!("Reasoning Effort: {score} (range 1-100")));
                    assert!(job.prompt.ends_with("<think>"));
                } else {
                    assert!(!job.prompt.contains("Reasoning Effort:"));
                    assert!(job.prompt.ends_with("</think>"));
                }
                job.events.send(Ok(InferenceChunk::Ready { system_fingerprint: None,
                    prompt_usage: PromptUsage { prompt_tokens: 1, prompt_cache_hit_tokens: 0 } })).await.unwrap();
                job.events.send(Ok(InferenceChunk::Text {
                    content: if score.is_some() { "Compute. </think>4" } else { "4" }.into(),
                    content_tokens: 1,
                })).await.unwrap();
                job.events.send(Ok(InferenceChunk::Finish {
                    finish_reason: InferenceFinishReason::Stop,
                })).await.unwrap();
            });
            let mut body = json!({"model":MODEL,"messages":[{"role":"user","content":"2+2?"}],
                "max_tokens":16,"stream":false});
            body.as_object_mut().unwrap().extend(options.as_object().unwrap().clone());
            let request = axum::http::Request::post("/v1/chat/completions")
                .header("content-type", "application/json").body(Body::from(body.to_string())).unwrap();
            let response = router(tx).oneshot(request).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
            let value: Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(value["choices"][0]["message"]["content"], "4");
            if score.is_some() {
                assert_eq!(value["choices"][0]["message"]["reasoning_content"], "Compute. ");
            }
            worker.await.unwrap();
        }
    }
    #[tokio::test]
    async fn official_prompt_and_both_response_modes() {
        for streaming in [false, true] {
            let (tx, mut rx) = mpsc::channel::<NativeRequest>(1);
            let worker = tokio::spawn(async move {
                let job = rx.recv().await.unwrap();
                assert_eq!(job.prompt,"<｜begin▁of▁sentence｜><｜User｜>What is 2 + 2? Answer with just the number.<｜Assistant｜></think>");
                assert_eq!(job.max_tokens, 16);
                for event in [
                    InferenceChunk::Ready {
                        system_fingerprint: None,
                        prompt_usage: PromptUsage {
                            prompt_tokens: 18,
                            prompt_cache_hit_tokens: 0,
                        },
                    },
                    InferenceChunk::Text {
                        content: "4".into(),
                        content_tokens: 1,
                    },
                    InferenceChunk::Finish {
                        finish_reason: InferenceFinishReason::Stop,
                    },
                ] {
                    job.events.send(Ok(event)).await.unwrap();
                }
            });
            let response = router(tx).oneshot(request(streaming)).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
                .await
                .unwrap();
            if streaming {
                let text = std::str::from_utf8(&body).unwrap();
                assert!(text.contains("\"content\":\"4\""));
                assert!(text.ends_with("data: [DONE]\n\n"));
            } else {
                let value: Value = serde_json::from_slice(&body).unwrap();
                assert_eq!(value["choices"][0]["message"]["content"], "4");
                assert_eq!(value["usage"]["prompt_tokens"], 18);
            }
            worker.await.unwrap();
        }
    }
    #[tokio::test]
    async fn admission_errors_preserve_status_and_cause_before_json_or_sse() {
        for streaming in [false, true] {
            for bad_request in [false, true] {
                let (tx, mut rx) = mpsc::channel::<NativeRequest>(1);
                let worker = tokio::spawn(async move {
                    let job = rx.recv().await.unwrap();
                    let message = "required parameter has incompatible value constraints: nx".to_string();
                    job.events.send(Err(if bad_request { NativeFailure::BadRequest(message) }
                        else { NativeFailure::Worker(message) })).await.unwrap();
                });
                let body = json!({"model":MODEL,"messages":[{"role":"user","content":"Call lookup."}],
                    "tools":[{"type":"function","function":{"name":"lookup","strict":true,
                        "parameters":{"type":"object","properties":{"nx":{"const":1}},"required":["nx"]}}}],
                    "tool_choice":"required","stream":streaming});
                let request = axum::http::Request::post("/v1/chat/completions")
                    .header("content-type", "application/json").body(Body::from(body.to_string())).unwrap();
                let response = router(tx).oneshot(request).await.unwrap();
                assert_eq!(response.status(), if bad_request { StatusCode::BAD_REQUEST }
                    else { StatusCode::INTERNAL_SERVER_ERROR });
                let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
                let value: Value = serde_json::from_slice(&bytes).unwrap();
                assert_eq!(value["error"]["message"], "required parameter has incompatible value constraints: nx");
                worker.await.unwrap();
            }
        }
    }
    #[tokio::test]
    async fn late_worker_failure_is_not_replaced_by_required_tool_validation() {
        for streaming in [false, true] {
            let (tx, mut rx) = mpsc::channel::<NativeRequest>(1);
            tokio::spawn(async move {
                let job = rx.recv().await.unwrap();
                job.events.send(Ok(InferenceChunk::Ready { system_fingerprint: None,
                    prompt_usage: PromptUsage { prompt_tokens: 1, prompt_cache_hit_tokens: 0 } })).await.unwrap();
                job.events.send(Err(NativeFailure::Worker("late execution failure".into()))).await.unwrap();
            });
            let body = json!({"model":MODEL,"messages":[{"role":"user","content":"Call lookup."}],
                "tools":[{"type":"function","function":{"name":"lookup","strict":true,
                    "parameters":{"type":"object","properties":{"n":{"const":1}},"required":["n"]}}}],
                "tool_choice":"required","stream":streaming});
            let request = axum::http::Request::post("/v1/chat/completions")
                .header("content-type", "application/json").body(Body::from(body.to_string())).unwrap();
            let response = router(tx).oneshot(request).await.unwrap();
            if streaming {
                assert!(axum::body::to_bytes(response.into_body(), 1024 * 1024).await.is_err());
            } else {
                assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
                let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
                let value: Value = serde_json::from_slice(&bytes).unwrap();
                assert_eq!(value["error"]["message"], "late execution failure");
            }
        }
    }
    #[tokio::test]
    async fn worker_error_and_missing_finish_are_not_success() {
        for explicit_error in [false, true] {
            let (tx, mut rx) = mpsc::channel::<NativeRequest>(1);
            tokio::spawn(async move {
                let job = rx.recv().await.unwrap();
                if explicit_error {
                    job.events
                        .send(Err("execution failed".into()))
                        .await
                        .unwrap();
                }
            });
            let response = router(tx).oneshot(request(false)).await.unwrap();
            assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        }
    }
    fn dummy_job() -> NativeRequest {
        let (events, _receive) = mpsc::channel(1);
        NativeRequest {
            prompt: String::new(),
            constraint: None,
            images: Vec::new(),
            max_tokens: 1,
            events,
        }
    }
    /// A dummy job whose prompt identifies it, so FIFO service order can be asserted.
    fn tagged_job(tag: &str) -> NativeRequest {
        let mut job = dummy_job();
        job.prompt = tag.to_string();
        job
    }
    async fn stats_json(app: Router) -> Value {
        let response = app
            .oneshot(
                axum::http::Request::get("/v1/stats")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }
    #[test]
    fn http_queue_depth_defaults_to_four_times_concurrency_and_zero_means_concurrency() {
        assert_eq!(http_queue_depth(16, None), 64);
        assert_eq!(http_queue_depth(1, None), 4);
        assert_eq!(http_queue_depth(16, Some(0)), 16);
        assert_eq!(http_queue_depth(16, Some(8)), 8);
        assert_eq!(http_queue_depth(16, Some(256)), 256);
    }
    #[tokio::test]
    async fn full_queue_without_wait_returns_429_with_retry_after_and_body_message() {
        let stats = Arc::new(Mutex::new(Value::Null));
        let (send, mut receive) = mpsc::channel::<NativeRequest>(1);
        let app = router_with_limits_stats_and_policy(
            send.clone(),
            NativeLimits::default(),
            stats.clone(),
            QueuePolicy::default(),
        );
        // Occupy the only slot; no consumer drains it.
        send.try_send(dummy_job()).unwrap();
        let response = app.clone().oneshot(request(false)).await.unwrap();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            response
                .headers()
                .get(axum::http::header::RETRY_AFTER)
                .unwrap(),
            "2"
        );
        let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let value: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            value["error"]["message"],
            "request queue full (1) after 0 ms"
        );
        assert_eq!(value["error"]["type"], "native_v41_error");
        let published = stats_json(app).await;
        assert_eq!(published["http_queue_rejects_429"], 1);
        assert_eq!(published["http_queue_waits"], 0);
        assert_eq!(published["http_queue_wait_ms_sum"], 0);
        assert_eq!(published["http_queue_len"], 1);
        // The parked job was never consumed and the refusal is counted once.
        assert!(receive.try_recv().is_ok());
    }
    #[tokio::test]
    async fn closed_queue_returns_503_not_429() {
        let (tx, rx) = mpsc::channel::<NativeRequest>(1);
        let app = router_with_limits_stats_and_policy(
            tx,
            NativeLimits::default(),
            Arc::new(Mutex::new(Value::Null)),
            QueuePolicy::default(),
        );
        drop(rx);
        let response = app.oneshot(request(false)).await.unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(response
            .headers()
            .get(axum::http::header::RETRY_AFTER)
            .is_none());
    }
    #[tokio::test]
    async fn queued_submission_waits_for_a_slot_then_proceeds_and_counts_the_wait() {
        let stats = Arc::new(Mutex::new(Value::Null));
        let (tx, mut rx) = mpsc::channel::<NativeRequest>(1);
        // The budget is far above any scheduling latency because the consumer drains
        // event-driven (as soon as the submission has parked), not after a fixed sleep.
        let app = router_with_limits_stats_and_policy(
            tx.clone(),
            NativeLimits::default(),
            stats.clone(),
            QueuePolicy {
                wait: Duration::from_secs(5),
                retry_after: Duration::from_secs(2),
            },
        );
        tx.try_send(dummy_job()).unwrap();
        let consumer = tokio::spawn({
            let app = app.clone();
            async move {
                // Wait until the submission has parked in the wait list (the `waits` metric
                // is bumped just before), then drain the parked dummy and serve the queued
                // real job. If the drain lands before the send's first poll, the send takes
                // the freed slot immediately; either way the outcome is a delivery.
                let mut observed = false;
                for _ in 0..100 {
                    let published = stats_json(app.clone()).await;
                    if published["http_queue_waits"].as_u64().unwrap_or(0) >= 1 {
                        observed = true;
                        break;
                    }
                    tokio::task::yield_now().await;
                }
                assert!(observed, "submission never reached the wait list");
                let parked = rx.recv().await.unwrap();
                assert!(parked.prompt.is_empty());
                let job = rx.recv().await.unwrap();
                job.events
                    .send(Ok(InferenceChunk::Ready {
                        system_fingerprint: None,
                        prompt_usage: PromptUsage {
                            prompt_tokens: 1,
                            prompt_cache_hit_tokens: 0,
                        },
                    }))
                    .await
                    .unwrap();
                job.events
                    .send(Ok(InferenceChunk::Finish {
                        finish_reason: InferenceFinishReason::Stop,
                    }))
                    .await
                    .unwrap();
            }
        });
        let response = app.clone().oneshot(request(false)).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        consumer.await.unwrap();
        let published = stats_json(app).await;
        assert_eq!(published["http_queue_waits"], 1);
        assert_eq!(published["http_queue_rejects_429"], 0);
    }
    #[tokio::test]
    async fn queued_submission_times_out_with_429_after_the_wait() {
        let stats = Arc::new(Mutex::new(Value::Null));
        let (tx, _rx) = mpsc::channel::<NativeRequest>(1);
        let app = router_with_limits_stats_and_policy(
            tx.clone(),
            NativeLimits::default(),
            stats.clone(),
            QueuePolicy {
                wait: Duration::from_millis(20),
                retry_after: Duration::from_secs(7),
            },
        );
        tx.try_send(dummy_job()).unwrap();
        let started = std::time::Instant::now();
        let response = app.clone().oneshot(request(false)).await.unwrap();
        assert!(started.elapsed() >= Duration::from_millis(20));
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            response
                .headers()
                .get(axum::http::header::RETRY_AFTER)
                .unwrap(),
            "7"
        );
        let published = stats_json(app).await;
        assert_eq!(published["http_queue_rejects_429"], 1);
        assert_eq!(published["http_queue_waits"], 1);
        assert!(published["http_queue_wait_ms_sum"].as_u64().unwrap() >= 20);
    }
    #[tokio::test]
    async fn stats_route_merges_http_queue_keys_without_disturbing_owner_keys() {
        let owner = json!({
            "host_cache": {"bytes": 1},
            "host_cache_config": {"store": "on_retain"},
            "admission_waits": 3,
            "admission_rejects": 1,
        });
        let (tx, _rx) = mpsc::channel::<NativeRequest>(1);
        let app = router_with_limits_stats_and_policy(
            tx,
            NativeLimits::default(),
            Arc::new(Mutex::new(owner)),
            QueuePolicy::default(),
        );
        let published = stats_json(app).await;
        assert_eq!(published["host_cache"]["bytes"], 1);
        assert_eq!(published["host_cache_config"]["store"], "on_retain");
        assert_eq!(published["admission_waits"], 3);
        assert_eq!(published["admission_rejects"], 1);
        for key in [
            "http_queue_waits",
            "http_queue_wait_ms_sum",
            "http_queue_rejects_429",
            "http_queue_len",
        ] {
            assert!(published[key].is_u64(), "missing numeric key {key}");
        }
    }
    #[tokio::test]
    async fn stats_route_serves_http_queue_keys_even_before_the_owner_publishes() {
        let (tx, _rx) = mpsc::channel::<NativeRequest>(1);
        let app = router_with_limits_stats_and_policy(
            tx,
            NativeLimits::default(),
            Arc::new(Mutex::new(Value::Null)),
            QueuePolicy::default(),
        );
        let published = stats_json(app).await;
        assert_eq!(published["http_queue_waits"], 0);
        assert_eq!(published["http_queue_len"], 0);
    }
    #[tokio::test]
    async fn timed_out_submission_is_never_delivered_and_the_client_gets_one_outcome() {
        let stats = Arc::new(Mutex::new(Value::Null));
        let (tx, mut rx) = mpsc::channel::<NativeRequest>(1);
        let app = router_with_limits_stats_and_policy(
            tx.clone(),
            NativeLimits::default(),
            stats.clone(),
            QueuePolicy {
                wait: Duration::from_millis(20),
                retry_after: Duration::from_secs(3),
            },
        );
        // Occupy the only slot; no consumer drains it while the submission waits.
        tx.try_send(dummy_job()).unwrap();
        // The bounded wait expires on the real clock; the single-outcome and no-delivery
        // assertions below hold from the moment the 429 is produced, with no timing margin.
        let response = app.clone().oneshot(request(false)).await.unwrap();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            response
                .headers()
                .get(axum::http::header::RETRY_AFTER)
                .unwrap(),
            "3"
        );
        let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let value: Value = serde_json::from_slice(&bytes).unwrap();
        assert!(value["error"]["message"]
            .as_str()
            .unwrap()
            .starts_with("request queue full (1)"));
        // The engine saw only the parked dummy; the timed-out job never arrived.
        let parked = rx.recv().await.unwrap();
        assert!(parked.prompt.is_empty());
        assert!(rx.try_recv().is_err());
        // Exactly one client outcome was produced and counted.
        let published = stats_json(app).await;
        assert_eq!(published["http_queue_rejects_429"], 1);
        assert_eq!(published["http_queue_waits"], 1);
    }
    #[tokio::test]
    async fn retry_after_rounds_up_sub_second_policies_to_one_second() {
        let (tx, _rx) = mpsc::channel::<NativeRequest>(1);
        let app = router_with_limits_stats_and_policy(
            tx.clone(),
            NativeLimits::default(),
            Arc::new(Mutex::new(Value::Null)),
            QueuePolicy {
                wait: Duration::ZERO,
                retry_after: Duration::from_millis(250),
            },
        );
        tx.try_send(dummy_job()).unwrap();
        let response = app.oneshot(request(false)).await.unwrap();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            response
                .headers()
                .get(axum::http::header::RETRY_AFTER)
                .unwrap(),
            "1"
        );
    }
    #[tokio::test]
    async fn multi_waiter_fifo_service_and_mid_wait_disconnect_cancels_cleanly() {
        const WAITERS: usize = 4;
        let (tx, mut rx) = mpsc::channel::<NativeRequest>(1);
        let state = NativeState {
            queue: tx.clone(),
            limits: NativeLimits::default(),
            images: images::ImageDecoder::new(1),
            stats: Arc::new(Mutex::new(Value::Null)),
            policy: QueuePolicy {
                wait: Duration::from_secs(60),
                retry_after: Duration::from_secs(2),
            },
            metrics: Arc::default(),
        };
        // Occupy the only slot, then park WAITERS submissions in spawn order.
        tx.try_send(dummy_job()).unwrap();
        let mut handles = Vec::new();
        for index in 0..WAITERS {
            let state = state.clone();
            let tag = format!("w{index}");
            handles.push(tokio::spawn(
                async move { state.submit(tagged_job(&tag)).await },
            ));
        }
        // Let every waiter register in the send-waiter FIFO before any slot frees.
        for _ in 0..WAITERS * 4 {
            tokio::task::yield_now().await;
        }
        // Simulated client disconnect: the front door drops the handler future mid-wait.
        // `abort` is the same cancellation a dropped axum handler applies to its future;
        // the cancelled reservation must leave the FIFO without ever delivering its job.
        handles[1].abort();
        assert!(handles.remove(1).await.unwrap_err().is_cancelled());
        // Free one slot at a time; the remaining waiters must be served in FIFO order.
        let parked = rx.recv().await.unwrap();
        assert!(parked.prompt.is_empty());
        for expected in ["w0", "w2", "w3"] {
            let job = rx.recv().await.unwrap();
            assert_eq!(job.prompt, expected);
        }
        assert!(rx.try_recv().is_err());
        for handle in handles {
            assert!(matches!(handle.await.unwrap(), Submission::Accepted));
        }
        // The disconnected waiter registered its wait but was never delivered or refused.
        assert_eq!(state.metrics.waits.load(Ordering::Relaxed), WAITERS as u64);
        assert_eq!(state.metrics.rejects_429.load(Ordering::Relaxed), 0);
    }
}
