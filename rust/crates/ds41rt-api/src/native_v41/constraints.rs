use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};

/// Serialized structural grammar; worker compilation is bounded and cached.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NativeConstraint(pub String);

pub(super) fn response_constraint(format: Option<Value>, thinking: bool, tools: Option<&super::tools::ToolConstraints>) -> Result<Option<NativeConstraint>> {
    let content = match format.as_ref().and_then(|v| v.get("type")).and_then(Value::as_str) {
        None if format.is_none() => None,
        Some("text") => None,
        Some("json_object") => Some(json!({"type":"ds41_json_schema", "strict":false,
            "json_schema":{"type":"object", "additionalProperties":true}})),
        Some("json_schema") => {
            let definition = format.as_ref().unwrap().get("json_schema").context("response_format.json_schema is required")?;
            let schema = definition.get("schema").context("response_format.json_schema.schema is required")?;
            ensure!(schema.is_object() || schema.is_boolean(), "JSON schema must be an object or boolean");
            compile_schema(schema)?;
            let strict = match definition.get("strict") {
                None | Some(Value::Null) => false,
                Some(Value::Bool(strict)) => *strict,
                _ => anyhow::bail!("response_format.json_schema.strict must be boolean"),
            };
            Some(json!({"type":"ds41_json_schema", "strict":strict, "json_schema":schema}))
        }
        Some("regex") => {
            let regex = format.as_ref().unwrap().get("regex").and_then(Value::as_str).context("response_format.regex must be a string")?;
            Some(json!({"type":"regex", "pattern":regex}))
        }
        _ => anyhow::bail!("unsupported response_format type"),
    };
    let content = match (content, tools.and_then(|t| t.format.as_ref())) {
        (Some(answer), Some(calls)) if !tools.unwrap().required => json!({"type":"or", "elements":[answer,calls]}),
        (_, Some(calls)) if tools.unwrap().required => calls.clone(),
        (None, Some(calls)) => json!({"type":"triggered_tags", "triggers":["<｜DSML｜ calls>"],
            "tags":[calls], "at_least_one":false, "stop_after_first":true}),
        (Some(answer), _) => answer,
        (None, None) => return Ok(None),
    };
    let format = if thinking {
        json!({"type":"sequence", "elements":[
            {"type":"any_tokens", "exclude_tokens":[128822]},
            {"type":"token", "token":128822}, content]})
    } else { content };
    Ok(Some(NativeConstraint(serde_json::to_string(&json!({"type":"structural_tag", "format":format}))?)))
}

pub(super) fn response_validator(format: Option<&Value>) -> Result<Option<jsonschema::JSONSchema>> {
    let Some(format) = format else { return Ok(None); };
    let schema = match format.get("type").and_then(Value::as_str) {
        Some("json_schema") => format.get("json_schema").and_then(|v| v.get("schema"))
            .context("response_format.json_schema.schema is required")?.clone(),
        Some("json_object") => json!({"type":"object"}),
        _ => return Ok(None),
    };
    compile_schema(&schema).map(Some)
}

pub(super) fn compile_schema(schema: &Value) -> Result<jsonschema::JSONSchema> {
    let mut options = jsonschema::JSONSchema::options();
    // Match modern prefixItems/$defs schemas unless the caller declares an
    // older dialect. HTTP/file resolution remains disabled by crate features.
    if schema.get("$schema").is_none() { options.with_draft(jsonschema::Draft::Draft202012); }
    options.compile(schema).map_err(|error| anyhow::anyhow!("invalid JSON schema: {error}"))
}

pub(super) fn validate_complete(validator: &jsonschema::JSONSchema, content: &str) -> Result<()> {
    let value: Value = serde_json::from_str(content).context("native response is not complete JSON")?;
    ensure!(validator.is_valid(&value), "native response does not satisfy the requested JSON schema");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn schema_survives_conversion_and_reasoning_wrap() {
        let schema = json!({"type":"object", "properties":{"value":{"const":"allowed"}}, "required":["value"]});
        for thinking in [false, true] {
            let result = response_constraint(Some(json!({"type":"json_schema", "json_schema":{"schema":schema,"strict":false}})), thinking, None).unwrap().unwrap();
            let encoded: Value = serde_json::from_str(&result.0).unwrap();
            let content = if thinking { &encoded["format"]["elements"][2] } else { &encoded["format"] };
            assert_eq!(content["json_schema"], schema);
            assert_eq!(content["strict"], false);
        }
        assert!(response_constraint(Some(json!({"type":"json_schema"})), false, None).is_err());
        assert!(response_constraint(Some(json!({"type":"json_schema","json_schema":{"schema":{"type":"invalid"}}})), false, None).is_err());
    }
    #[test]
    fn validation_respects_modern_and_declared_legacy_tuple_dialects() {
        for schema in [
            json!({"type":"array", "prefixItems":[{"const":true},{"const":null}], "items":false,"minItems":2}),
            json!({"$schema":"http://json-schema.org/draft-07/schema#", "type":"array", "items":[{"const":true},{"const":null}], "additionalItems":false,"minItems":2}),
        ] {
            let validator = compile_schema(&schema).unwrap();
            assert!(validate_complete(&validator, "[true,null]").is_ok());
            assert!(validate_complete(&validator, "[true,null,1]").is_err());
            assert!(validate_complete(&validator, "[null,true]").is_err());
        }
    }
    #[tokio::test]
    async fn invalid_completed_schema_output_is_never_successful() {
        use crate::native_v41::*;
        use axum::{body::{Body, to_bytes}, http::{Request, StatusCode}};
        use tower::ServiceExt;
        for streaming in [false, true] {
            let (send, mut receive) = tokio::sync::mpsc::channel::<NativeRequest>(1);
            let worker = tokio::spawn(async move {
                let job = receive.recv().await.unwrap();
                assert!(job.constraint.is_some());
                for event in [
                    InferenceChunk::Ready { system_fingerprint:None, prompt_usage:PromptUsage { prompt_tokens:1, prompt_cache_hit_tokens:0 } },
                    InferenceChunk::Text { content:"{\"wrong\":true}".into(), content_tokens:5 },
                    InferenceChunk::Finish { finish_reason:InferenceFinishReason::Stop },
                ] { if job.events.send(Ok(event)).await.is_err() { break; } }
            });
            let body = json!({"model":MODEL,"messages":[{"role":"user","content":"Return JSON."}],
                "thinking":{"type":"disabled"},"stream":streaming,
                "response_format":{"type":"json_schema","json_schema":{"schema":{"const":{"correct":true}}}}});
            let request = Request::post("/v1/chat/completions").header("content-type","application/json")
                .body(Body::from(body.to_string())).unwrap();
            let response = router(send).oneshot(request).await.unwrap();
            if streaming { assert!(to_bytes(response.into_body(), 1<<20).await.is_err()); }
            else { assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR); }
            worker.await.unwrap();
        }
    }

}
