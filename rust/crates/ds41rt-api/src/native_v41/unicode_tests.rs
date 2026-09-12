use super::*;
use tower::ServiceExt;

#[tokio::test]
async fn unicode_reasoning_and_content_survive_character_chunks_in_both_modes() {
    let reasoning = "思考 café e\u{301} 🦜�\0\n段落";
    let content = "台北 👩🏽\u{200d}💻 �";
    let input = "עברית العربية हिन्दी ไทย e\u{301} �\0";
    for streaming in [false, true] {
        let (queue, mut receive) = mpsc::channel::<NativeRequest>(1);
        let worker = tokio::spawn(async move {
            let job = receive.recv().await.unwrap();
            assert!(job.prompt.contains(input));
            job.events
                .send(Ok(InferenceChunk::Ready {
                    system_fingerprint: None,
                    prompt_usage: PromptUsage {
                        prompt_tokens: 42,
                        prompt_cache_hit_tokens: 0,
                    },
                }))
                .await
                .unwrap();
            // The protocol consumes newlines adjoining </think>, while the
            // newline and every Unicode/control scalar inside reasoning stay.
            for character in format!("{reasoning}\n</think>{content}").chars() {
                job.events
                    .send(Ok(InferenceChunk::Text {
                        content: character.to_string(),
                        content_tokens: 1,
                    }))
                    .await
                    .unwrap();
            }
            job.events
                .send(Ok(InferenceChunk::Finish {
                    finish_reason: InferenceFinishReason::Stop,
                }))
                .await
                .unwrap();
        });
        let mut body = json!({"model":MODEL,"messages":[{"role":"user","content":input}],
            "stream":streaming});
        if streaming {
            body["stream_options"] = json!({"include_usage":true});
        }
        let request = axum::http::Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let response = router(queue).oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
            .await
            .unwrap();
        let wire = std::str::from_utf8(&bytes).unwrap();
        let (actual_reasoning, actual_content) = if streaming {
            let mut thoughts = String::new();
            let mut answer = String::new();
            let mut done = false;
            for line in wire.lines().filter_map(|line| line.strip_prefix("data: ")) {
                if line == "[DONE]" {
                    done = true;
                    continue;
                }
                let event: Value = serde_json::from_str(line).unwrap();
                for choice in event["choices"].as_array().unwrap() {
                    thoughts.push_str(choice["delta"]["reasoning_content"].as_str().unwrap_or(""));
                    answer.push_str(choice["delta"]["content"].as_str().unwrap_or(""));
                }
            }
            assert!(done);
            (thoughts, answer)
        } else {
            let response: Value = serde_json::from_str(wire).unwrap();
            let message = &response["choices"][0]["message"];
            (
                message["reasoning_content"].as_str().unwrap().to_owned(),
                message["content"].as_str().unwrap().to_owned(),
            )
        };
        assert_eq!(actual_reasoning, reasoning);
        assert_eq!(actual_content, content);
        worker.await.unwrap();
    }
}
