use axum::http::{Method, StatusCode};
use ds41rt_core::DEFAULT_MODEL_ID;
use serde_json::json;

use super::{request_json, request_json_with_config, request_text};

#[tokio::test]
async fn health_and_models_routes_are_openai_smoke_compatible() {
    let (health_status, health_body) = request_json(Method::GET, "/health", None).await;
    assert_eq!(health_status, StatusCode::OK);
    assert_eq!(health_body["status"], "ok");
    assert_eq!(health_body["service"], "ds41rt");
    assert_eq!(health_body["backend"], "tiny");
    assert_eq!(health_body["transport"], "inproc");

    let (models_status, models_body) = request_json(Method::GET, "/v1/models", None).await;
    assert_eq!(models_status, StatusCode::OK);
    assert_eq!(models_body["object"], "list");
    let ids = models_body["data"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|model| model["id"].as_str())
        .collect::<Vec<_>>();
    assert!(ids.contains(&"ds41rt-tiny"));
    assert!(ids.contains(&"ds41rt-synthetic-ds4-layer"));
    assert!(ids.contains(&DEFAULT_MODEL_ID));
    let full_model_id = format!("{}-full", DEFAULT_MODEL_ID);
    assert!(ids.contains(&full_model_id.as_str()));
}

#[tokio::test]
async fn custom_checkpoint_identity_is_advertised_and_enforced() {
    let model_id = "tpurtell/DeepSeek-V4-Flash-0731-EXL3-2bpw-v4";
    let mut config = crate::ApiConfig::default();
    config.model_id = model_id.to_owned();

    let (models_status, models_body) =
        request_json_with_config(config.clone(), Method::GET, "/v1/models", None).await;
    assert_eq!(models_status, StatusCode::OK);
    let ids = models_body["data"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|model| model["id"].as_str())
        .collect::<Vec<_>>();
    assert!(ids.contains(&model_id));
    let custom_full_model = format!("{model_id}-full");
    assert!(ids.contains(&custom_full_model.as_str()));
    assert!(!ids.contains(&format!("{}-full", DEFAULT_MODEL_ID).as_str()));

    let (status, body) = request_json_with_config(
        config,
        Method::POST,
        "/v1/chat/completions",
        Some(json!({
            "model": format!("{}-full", DEFAULT_MODEL_ID),
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 1
        })),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["param"], "model");
    assert!(body["error"]["message"]
        .as_str()
        .is_some_and(|message| message.contains("not served")));
}

#[tokio::test]
async fn chat_completion_route_smokes_non_streaming_stop_and_length() {
    let (status, body) = request_json(
        Method::POST,
        "/v1/chat/completions",
        Some(json!({
            "model": "ds41rt-tiny",
            "messages": [{"role": "user", "content": "Say hello in five words."}],
            "max_tokens": 16
        })),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(
        body["choices"][0]["message"]["content"],
        "hello from ds41rt tiny backend"
    );
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    assert_eq!(body["metrics"]["backend_mode"], "tiny");

    let (stop_status, stop_body) = request_json(
        Method::POST,
        "/v1/chat/completions",
        Some(json!({
            "model": "ds41rt-tiny",
            "messages": [{"role": "user", "content": "Say hello."}],
            "stop": "tiny",
            "max_tokens": 16
        })),
    )
    .await;
    assert_eq!(stop_status, StatusCode::OK);
    assert_eq!(
        stop_body["choices"][0]["message"]["content"],
        "hello from ds41rt "
    );
    assert_eq!(stop_body["choices"][0]["finish_reason"], "stop");

    let (length_status, length_body) = request_json(
        Method::POST,
        "/v1/chat/completions",
        Some(json!({
            "model": "ds41rt-tiny",
            "messages": [{"role": "user", "content": "Say hello."}],
            "max_tokens": 1
        })),
    )
    .await;
    assert_eq!(length_status, StatusCode::OK);
    assert_eq!(length_body["choices"][0]["message"]["content"], "hello");
    assert_eq!(length_body["choices"][0]["finish_reason"], "length");
}

#[tokio::test]
async fn chat_completion_accepts_json_larger_than_axums_two_mib_default() {
    let (status, body) = request_json(
        Method::POST,
        "/v1/chat/completions",
        Some(json!({
            "model": "ds41rt-tiny",
            "messages": [{"role": "user", "content": "Say hello."}],
            "max_completion_tokens": 1,
            // Long prompts and tool schemas can exceed Axum's default body
            // limit; an ignored field isolates the route-level regression.
            "client_metadata_padding": "x".repeat(2 * 1024 * 1024)
        })),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["choices"][0]["message"]["content"], "hello");
}

#[tokio::test]
async fn chat_completion_rejects_images_for_text_only_ds4_models() {
    let (status, body) = request_json(
        Method::POST,
        "/v1/chat/completions",
        Some(json!({
            "model": "ds41rt-tiny",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
                ]
            }],
            "max_completion_tokens": 1
        })),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["param"], "messages");
    assert_eq!(body["error"]["code"], "invalid_request");
    assert!(body["error"]["message"]
        .as_str()
        .is_some_and(|message| message.contains("text-only")));
}

#[tokio::test]
async fn chat_completion_route_smokes_streaming_sse_text() {
    let (status, text) = request_text(
        Method::POST,
        "/v1/chat/completions",
        json!({
            "model": "ds41rt-tiny",
            "stream": true,
            "messages": [{"role": "user", "content": "Count to three."}],
            "max_tokens": 16
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(text.contains("data:"));
    assert!(text.contains("\"object\":\"chat.completion.chunk\""));
    assert!(text.contains("\"role\":\"assistant\""));
    assert!(text.contains("one"));
    assert!(text.contains("three"));
    assert!(text.contains("[DONE]"));
}

#[tokio::test]
async fn tool_choice_does_not_fabricate_tiny_backend_calls() {
    let (status, text) = request_text(
        Method::POST,
        "/v1/chat/completions",
        json!({
            "model": "ds41rt-tiny",
            "stream": true,
            "messages": [{"role": "user", "content": "Use the function."}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Lookup a value.",
                    "parameters": {"type": "object", "properties": {}}
                }
            }],
            "tool_choice": "required"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(text.contains("\"role\":\"assistant\""));
    assert!(text.contains("\"content\":\"hello\""));
    assert!(text.contains("\"content\":\" tiny\""));
    assert!(!text.contains("\"tool_calls\":"));
    assert!(text.contains("\"finish_reason\":\"stop\""));
    assert!(text.contains("[DONE]"));
}

#[tokio::test]
async fn chat_completion_route_smokes_tools_and_structured_errors() {
    let (tool_status, tool_body) = request_json(
        Method::POST,
        "/v1/chat/completions",
        Some(json!({
            "model": "ds41rt-tiny",
            "messages": [{"role": "user", "content": "Use the function."}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Lookup a value.",
                    "parameters": {"type": "object", "properties": {}}
                }
            }],
            "tool_choice": "required"
        })),
    )
    .await;
    assert_eq!(tool_status, StatusCode::OK);
    assert_eq!(tool_body["choices"][0]["finish_reason"], "stop");
    assert_eq!(
        tool_body["choices"][0]["message"]["content"],
        "hello from ds41rt tiny backend"
    );
    assert!(tool_body["choices"][0]["message"]["tool_calls"].is_null());

    let (bad_tool_status, bad_tool_body) = request_json(
        Method::POST,
        "/v1/chat/completions",
        Some(json!({
            "model": "ds41rt-tiny",
            "messages": [{"role": "user", "content": "Use the function."}],
            "tools": [{
                "type": "retrieval",
                "function": {"name": "lookup"}
            }]
        })),
    )
    .await;
    assert_eq!(bad_tool_status, StatusCode::BAD_REQUEST);
    assert_eq!(bad_tool_body["error"]["param"], "tools[0].type");
    assert_eq!(bad_tool_body["error"]["code"], "invalid_request");

    let (tool_choice_status, tool_choice_body) = request_json(
        Method::POST,
        "/v1/chat/completions",
        Some(json!({
            "model": "ds41rt-tiny",
            "messages": [{"role": "user", "content": "Use a function."}],
            "tool_choice": "required"
        })),
    )
    .await;
    assert_eq!(tool_choice_status, StatusCode::BAD_REQUEST);
    assert_eq!(tool_choice_body["error"]["param"], "tool_choice");

    let (role_status, role_body) = request_json(
        Method::POST,
        "/v1/chat/completions",
        Some(json!({
            "model": "ds41rt-tiny",
            "messages": [{"role": "invalid", "content": "hello"}]
        })),
    )
    .await;
    assert_eq!(role_status, StatusCode::BAD_REQUEST);
    assert_eq!(role_body["error"]["type"], "invalid_request_error");
    assert_eq!(role_body["error"]["param"], "messages[0].role");
    assert_eq!(role_body["error"]["code"], "invalid_request");
}
