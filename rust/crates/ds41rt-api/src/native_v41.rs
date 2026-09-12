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
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

pub const MODEL: &str = "deepseek-ai/DeepSeek-V4.1-Flash";
mod limits;
mod constraints;
pub use constraints::NativeConstraint;
mod images;
#[cfg(test)]
mod unicode_tests;
pub use limits::{NativeLimits, MAX_CONTEXT_TOKENS, MAX_OUTPUT_TOKENS};
pub struct NativeRequest {
    pub prompt: String,
    pub constraint: Option<NativeConstraint>,
    pub images: Vec<ds41rt_loader::V41Image>,
    pub max_tokens: usize,
    pub events: mpsc::Sender<Result<InferenceChunk, String>>,
}
#[derive(Clone)]
struct NativeState {
    queue: mpsc::Sender<NativeRequest>,
    limits: NativeLimits,
    images: images::ImageDecoder,
}
pub fn router(queue: mpsc::Sender<NativeRequest>) -> Router {
    router_with_limits(queue, NativeLimits::default())
}
pub fn router_with_limits(queue: mpsc::Sender<NativeRequest>, limits: NativeLimits) -> Router {
    let images = images::ImageDecoder::new(queue.max_capacity());
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .layer(axum::extract::DefaultBodyLimit::max(images::BODY_BYTES))
        .with_state(NativeState { queue, limits, images })
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
async fn chat(State(state): State<NativeState>, Json(mut body): Json<Value>) -> Response {
    let response_format = body.get("response_format").cloned().filter(|v| !v.is_null());
    // The recipe rejects its regex variant, while native XGrammar supports it.
    // Keep the original format for enforcement and render it as ordinary text.
    if response_format.as_ref().and_then(|v| v.get("type")).and_then(Value::as_str) == Some("regex") {
        body["response_format"] = json!({"type":"text"});
    }
    let parsed: ChatCompletionRequest = match serde_json::from_value(body) {
        Ok(r) => r,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let include_usage = parsed.include_usage();
    // Native serving defaults to thinking at the adapter's high effort. Explicit
    // thinking/effort settings retain the official conversion precedence.
    let converted = match parsed.convert(ConversionOptions::default().with_default_thinking_mode(true)) {
        Ok(r) => r,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
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
    let required_tools = if converted.conversation.tools.is_empty()
        || matches!(converted.conversation.tool_choice, deepseek_recipe_core::tools::ToolChoice::None) { None }
        else { Some(matches!(converted.conversation.tool_choice, deepseek_recipe_core::tools::ToolChoice::Required)) };
    let constraint = match constraints::response_constraint(response_format, converted.conversation.thinking_mode, required_tools) {
        Ok(constraint) => constraint,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
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
    if let Some(permit) = permit { permit.send(job); }
    else if let Err(e) = state.queue.try_send(job) {
        return error(StatusCode::SERVICE_UNAVAILABLE, e);
    }
    // Never let a failed/disconnected backend be converted to a successful EOF.
    let failure = Arc::new(Mutex::new(None::<String>));
    let input_failure = failure.clone();
    let input = async_stream::stream! {
        let mut finished = false;
        while let Some(event) = receive.recv().await {
            match event {
                Ok(chunk) => {
                    finished = matches!(chunk,InferenceChunk::Finish { .. });
                    yield chunk;
                    if finished { break; }
                }
                Err(message) => { *input_failure.lock().unwrap() = Some(message); break; }
            }
        }
        if !finished {
            input_failure.lock().unwrap().get_or_insert_with(|| "native worker ended without completion".into());
        }
    };
    let chunks = processor.process(input);
    if streaming {
        let stream = async_stream::stream! {
            futures::pin_mut!(chunks);
            let mut content = String::new();
            while let Some(chunk) = chunks.next().await {
                let failed = failure.lock().unwrap().clone();
                if let Some(message) = failed {
                    yield Err::<String,std::io::Error>(std::io::Error::other(message)); return;
                }
                match chunk {
                    Ok(chunk) => {
                        if let Some(validator) = &response_validator {
                            let value = serde_json::to_value(&chunk).unwrap();
                            if let Some(choices) = value["choices"].as_array() {
                                for choice in choices {
                                    if let Some(text) = choice["delta"]["content"].as_str() { content.push_str(text); }
                                    if choice["finish_reason"].as_str() == Some("stop") {
                                        if let Err(error) = constraints::validate_complete(validator, &content) {
                                            yield Err(std::io::Error::other(error.to_string())); return;
                                        }
                                    }
                                }
                            }
                        }
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
            Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, e),
        }
    }
    if let Some(message) = failure.lock().unwrap().clone() {
        return error(StatusCode::INTERNAL_SERVER_ERROR, message);
    }
    if let Some(validator) = response_validator {
        let value = serde_json::to_value(response).unwrap();
        if let Some(choices) = value["choices"].as_array() {
            for choice in choices {
                if choice["finish_reason"].as_str() == Some("stop") {
                    if let Err(e) = constraints::validate_complete(&validator, choice["message"]["content"].as_str().unwrap_or("")) {
                        return error(StatusCode::INTERNAL_SERVER_ERROR, e);
                    }
                }
            }
        }
        return Json(value).into_response();
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
}
