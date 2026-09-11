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
pub struct NativeRequest {
    pub prompt: String,
    pub max_tokens: usize,
    pub events: mpsc::Sender<Result<InferenceChunk, String>>,
}
#[derive(Clone)]
struct NativeState {
    queue: mpsc::Sender<NativeRequest>,
}
pub fn router(queue: mpsc::Sender<NativeRequest>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(|| async { Json(json!({"object":"list","data":[{"id":MODEL,"object":"model","owned_by":"deepseek-ai"}]})) }))
        .route("/v1/chat/completions", post(chat))
        .with_state(NativeState { queue })
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
async fn chat(State(state): State<NativeState>, Json(body): Json<Value>) -> Response {
    let parsed: ChatCompletionRequest = match serde_json::from_value(body) {
        Ok(r) => r,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let include_usage = parsed.include_usage();
    let converted = match parsed.convert(ConversionOptions::default()) {
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
    let max_tokens = converted.inference_options.max_tokens.unwrap_or(128) as usize;
    if !(1..=4096).contains(&max_tokens) {
        return error(StatusCode::BAD_REQUEST, "max_tokens must be 1..4096");
    }
    let rendered = DeepseekV41Encoding::new().render_conversation(&converted.conversation);
    if !rendered.image_sources.is_empty() {
        return error(
            StatusCode::BAD_REQUEST,
            "native vision input is not integrated yet",
        );
    }
    let streaming = converted.stream;
    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let generator = ChatCompletionRequest::chunk_generator(&converted, id.clone(), MODEL.into())
        .with_include_usage(!streaming || include_usage);
    let processor = StreamProcessor::new(generator, converted.parsing_options);
    let (events, mut receive) = mpsc::channel(16);
    if let Err(e) = state.queue.try_send(NativeRequest {
        prompt: rendered.prompt,
        max_tokens,
        events,
    }) {
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
            while let Some(chunk) = chunks.next().await {
                let failed = failure.lock().unwrap().clone();
                if let Some(message) = failed {
                    yield Err::<String,std::io::Error>(std::io::Error::other(message)); return;
                }
                match chunk {
                    Ok(chunk) => yield Ok(format!("data: {}\n\n",serde_json::to_string(&chunk).unwrap())),
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
