use serde_json::Value;

use crate::request::{message_content_text, request_thinking_enabled, tool_calls_enabled};
use crate::{ChatCompletionRequest, ChatMessage, ChatTool, ResponseFormat, ToolCall, ToolChoice};

pub(crate) const DS4_BOS: &str = "<｜begin▁of▁sentence｜>";
pub(crate) const DS4_EOS: &str = "<｜end▁of▁sentence｜>";
pub(crate) const DS4_USER: &str = "<｜User｜>";
pub(crate) const DS4_ASSISTANT: &str = "<｜Assistant｜>";
pub(crate) const DS4_LATEST_REMINDER: &str = "<｜latest_reminder｜>";
pub(crate) const DS4_THINK_OPEN: &str = "<think>";
pub(crate) const DS4_THINK_CLOSE: &str = "</think>";
const DS4_ACTION: &str = "<｜action｜>";
const DS4_QUERY: &str = "<｜query｜>";
const DS4_AUTHORITY: &str = "<｜authority｜>";
const DS4_DOMAIN: &str = "<｜domain｜>";
const DS4_TITLE: &str = "<｜title｜>";
const DS4_READ_URL: &str = "<｜read_url｜>";
const HIGH_REASONING_PREFIX: &str = "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n\
You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n\
Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n";

const MAX_REASONING_PREFIX: &str = "Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\n\
You MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively decompose the problem into its most fundamental components, trace every causal chain to its root, and resolve the underlying cause rather than any surface symptom.\n\
Do not stop reasoning until you have independently verified the solution from multiple angles and are certain that no assumption remains unchecked and no error remains undiscovered.\n\n";

const TOOLS_PREAMBLE: &str = "## Tools\n\n\
You have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<｜DSML｜tool_calls>\" block like the following:\n\n\
<｜DSML｜tool_calls>\n\
<｜DSML｜invoke name=\"$TOOL_NAME\">\n\
<｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜parameter>\n\
...\n\
</｜DSML｜invoke>\n\
<｜DSML｜invoke name=\"$TOOL_NAME2\">\n\
...\n\
</｜DSML｜invoke>\n\
</｜DSML｜tool_calls>\n\n\
String parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\n\
If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n\n\
Otherwise, output directly after </think> with tool calls or final response.\n\n\
### Available Tool Schemas\n\n";

const TOOLS_SUFFIX: &str =
    "\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n";

#[derive(Debug, Clone)]
enum UserPart {
    Text(String),
    ToolResult { id: Option<String>, content: String },
}

#[derive(Debug, Clone)]
enum EncodedMessage {
    System {
        content: String,
        tools: Vec<ChatTool>,
        response_format: Option<Value>,
    },
    User {
        parts: Vec<UserPart>,
        task: Option<String>,
    },
    Developer {
        content: String,
        tools: Vec<ChatTool>,
        response_format: Option<Value>,
        task: Option<String>,
    },
    LatestReminder(String),
    Assistant {
        content: String,
        reasoning: String,
        tool_calls: Vec<ToolCall>,
        wo_eos: bool,
    },
}

impl EncodedMessage {
    fn is_user_like(&self) -> bool {
        matches!(self, Self::User { .. } | Self::Developer { .. })
    }

    fn task(&self) -> Option<&str> {
        match self {
            Self::User { task, .. } | Self::Developer { task, .. } => task.as_deref(),
            _ => None,
        }
    }

    fn has_tools(&self) -> bool {
        matches!(
            self,
            Self::System { tools, .. } | Self::Developer { tools, .. } if !tools.is_empty()
        )
    }
}

pub(crate) fn render_ds4_request_prompt(request: &ChatCompletionRequest) -> String {
    let tools = selected_tools(request);
    let thinking = request_thinking_enabled(request);
    let mut messages = merge_openai_messages(&request.messages);
    let drop_thinking = tools.is_empty() && !messages.iter().any(EncodedMessage::has_tools);
    let preamble = request_preamble(request, &tools);
    if !preamble.is_empty() {
        if let Some(EncodedMessage::System { content, .. }) = messages
            .iter_mut()
            .find(|message| matches!(message, EncodedMessage::System { .. }))
        {
            if !content.is_empty() {
                content.push_str("\n\n");
            }
            content.push_str(&preamble);
        } else {
            messages.insert(
                0,
                EncodedMessage::System {
                    content: preamble,
                    tools: Vec::new(),
                    response_format: None,
                },
            );
        }
    }

    let last_user = messages.iter().rposition(EncodedMessage::is_user_like);
    let mut prompt = String::from(DS4_BOS);
    if thinking {
        prompt.push_str(reasoning_prefix(request));
    }

    for (index, message) in messages.iter().enumerate() {
        if thinking
            && drop_thinking
            && matches!(message, EncodedMessage::Developer { .. })
            && last_user.is_some_and(|last| index < last)
        {
            continue;
        }
        match message {
            EncodedMessage::System {
                content,
                tools,
                response_format,
            } => {
                prompt.push_str(content);
                prompt.push_str(&message_preamble(tools, response_format.as_ref()));
            }
            EncodedMessage::User { parts, .. } => {
                prompt.push_str(DS4_USER);
                for (part_index, part) in parts.iter().enumerate() {
                    if part_index != 0 {
                        prompt.push_str("\n\n");
                    }
                    match part {
                        UserPart::Text(content) => prompt.push_str(content),
                        UserPart::ToolResult { content, .. } => {
                            prompt.push_str("<tool_result>");
                            prompt.push_str(content);
                            prompt.push_str("</tool_result>");
                        }
                    }
                }
            }
            EncodedMessage::Developer {
                content,
                tools,
                response_format,
                ..
            } => {
                prompt.push_str(DS4_USER);
                prompt.push_str(content);
                prompt.push_str(&message_preamble(tools, response_format.as_ref()));
            }
            EncodedMessage::LatestReminder(content) => {
                prompt.push_str(DS4_LATEST_REMINDER);
                prompt.push_str(content);
            }
            EncodedMessage::Assistant {
                content,
                reasoning,
                tool_calls,
                wo_eos,
            } => {
                let previous_has_task = index > 0 && messages[index - 1].task().is_some();
                if thinking
                    && !previous_has_task
                    && (!drop_thinking || last_user.is_some_and(|last| index > last))
                {
                    prompt.push_str(reasoning);
                    prompt.push_str(DS4_THINK_CLOSE);
                }
                prompt.push_str(content);
                if !tool_calls.is_empty() {
                    prompt.push_str("\n\n");
                    prompt.push_str(&render_ds4_tool_calls(tool_calls));
                }
                if !wo_eos {
                    prompt.push_str(DS4_EOS);
                }
            }
        }

        if message.is_user_like() && next_accepts_transition(&messages, index) {
            if let Some(task) = message.task() {
                if task == "action" {
                    prompt.push_str(DS4_ASSISTANT);
                    prompt.push_str(if thinking {
                        DS4_THINK_OPEN
                    } else {
                        DS4_THINK_CLOSE
                    });
                }
                prompt.push_str(ds4_task_token(task));
            } else {
                prompt.push_str(DS4_ASSISTANT);
                let open_thinking =
                    thinking && (!drop_thinking || last_user.is_some_and(|last| index >= last));
                prompt.push_str(if open_thinking {
                    DS4_THINK_OPEN
                } else {
                    DS4_THINK_CLOSE
                });
            }
        }
    }
    prompt
}

fn merge_openai_messages(messages: &[ChatMessage]) -> Vec<EncodedMessage> {
    let mut merged = Vec::with_capacity(messages.len());
    let mut tool_order = Vec::<String>::new();
    let mut index = 0;
    while index < messages.len() {
        let message = &messages[index];
        match message.role.as_str() {
            "system" => merged.push(EncodedMessage::System {
                content: message_content_text(&message.content),
                tools: message.tools.clone().unwrap_or_default(),
                response_format: message.response_format.clone(),
            }),
            "developer" => merged.push(EncodedMessage::Developer {
                content: message_content_text(&message.content),
                tools: message.tools.clone().unwrap_or_default(),
                response_format: message.response_format.clone(),
                task: message.task.clone(),
            }),
            "latest_reminder" => merged.push(EncodedMessage::LatestReminder(message_content_text(
                &message.content,
            ))),
            "assistant" => {
                let tool_calls = message.tool_calls.clone().unwrap_or_default();
                tool_order = tool_calls.iter().map(|call| call.id.clone()).collect();
                merged.push(EncodedMessage::Assistant {
                    content: assistant_visible_content(&message_content_text(&message.content)),
                    reasoning: message.reasoning_content.clone().unwrap_or_default(),
                    tool_calls,
                    wo_eos: message.wo_eos,
                });
            }
            "tool" => {
                let run_start = index;
                while index + 1 < messages.len() && messages[index + 1].role == "tool" {
                    index += 1;
                }
                let mut results = messages[run_start..=index]
                    .iter()
                    .map(|tool| UserPart::ToolResult {
                        id: tool.tool_call_id.clone(),
                        content: message_content_text(&tool.content),
                    })
                    .collect::<Vec<_>>();
                results.sort_by_key(|result| match result {
                    UserPart::ToolResult { id: Some(id), .. } => tool_order
                        .iter()
                        .position(|candidate| candidate == id)
                        .unwrap_or(usize::MAX),
                    _ => usize::MAX,
                });
                append_user_parts(&mut merged, results, None);
            }
            "user" => append_user_parts(
                &mut merged,
                vec![UserPart::Text(message_content_text(&message.content))],
                message.task.clone(),
            ),
            _ => {}
        }
        index += 1;
    }
    merged
}

fn append_user_parts(
    messages: &mut Vec<EncodedMessage>,
    parts: Vec<UserPart>,
    task: Option<String>,
) {
    if let Some(EncodedMessage::User {
        parts: existing,
        task: existing_task,
    }) = messages.last_mut()
    {
        if existing_task.is_none() {
            existing.extend(parts);
            *existing_task = task;
            return;
        }
    }
    messages.push(EncodedMessage::User { parts, task });
}

fn next_accepts_transition(messages: &[EncodedMessage], index: usize) -> bool {
    index + 1 == messages.len()
        || matches!(
            messages[index + 1],
            EncodedMessage::Assistant { .. } | EncodedMessage::LatestReminder(_)
        )
}

fn ds4_task_token(task: &str) -> &'static str {
    match task {
        "action" => DS4_ACTION,
        "query" => DS4_QUERY,
        "authority" => DS4_AUTHORITY,
        "domain" => DS4_DOMAIN,
        "title" => DS4_TITLE,
        "read_url" => DS4_READ_URL,
        _ => "",
    }
}

fn assistant_visible_content(content: &str) -> String {
    content
        .split_once(DS4_THINK_CLOSE)
        .map_or(content, |(_, visible)| visible)
        .to_owned()
}

fn reasoning_prefix(request: &ChatCompletionRequest) -> &'static str {
    match request
        .reasoning_effort
        .as_ref()
        .and_then(Value::as_str)
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("high") => HIGH_REASONING_PREFIX,
        Some("max") => MAX_REASONING_PREFIX,
        _ => "",
    }
}

fn request_preamble(request: &ChatCompletionRequest, tools: &[&ChatTool]) -> String {
    let mut sections = Vec::new();
    if !tools.is_empty() {
        let mut rendered = String::from(TOOLS_PREAMBLE);
        rendered.push_str(
            &tools
                .iter()
                .map(|tool| ds4_tool_schema_json(tool))
                .collect::<Vec<_>>()
                .join("\n"),
        );
        rendered.push_str(TOOLS_SUFFIX);
        match &request.tool_choice {
            Some(ToolChoice::Mode(mode)) if mode == "required" => {
                rendered.push_str("\n\nYou MUST invoke at least one available tool.");
            }
            Some(ToolChoice::Specific { function, .. }) => {
                rendered.push_str("\n\nYou MUST invoke the tool named \"");
                rendered.push_str(&function.name);
                rendered.push_str("\".");
            }
            _ => {}
        }
        sections.push(rendered);
    }
    if let Some(format) = request.response_format.as_ref() {
        let schema = match format {
            ResponseFormat::Text => None,
            ResponseFormat::JsonObject => Some(serde_json::json!({"type": "object"})),
            ResponseFormat::JsonSchema { json_schema } => Some(json_schema.schema.clone()),
        };
        if let Some(schema) = schema {
            sections.push(format!(
                "## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n{}",
                dsv4_json(&schema)
            ));
        }
    }
    sections.join("\n\n")
}

fn message_preamble(tools: &[ChatTool], response_format: Option<&Value>) -> String {
    let mut sections = Vec::new();
    if !tools.is_empty() {
        let mut rendered = String::from(TOOLS_PREAMBLE);
        rendered.push_str(
            &tools
                .iter()
                .map(ds4_tool_schema_json)
                .collect::<Vec<_>>()
                .join("\n"),
        );
        rendered.push_str(TOOLS_SUFFIX);
        sections.push(rendered);
    }
    if let Some(response_format) = response_format {
        sections.push(format!(
            "## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n{}",
            dsv4_json(response_format)
        ));
    }
    if sections.is_empty() {
        String::new()
    } else {
        format!("\n\n{}", sections.join("\n\n"))
    }
}

fn selected_tools(request: &ChatCompletionRequest) -> Vec<&ChatTool> {
    if !tool_calls_enabled(request) {
        return Vec::new();
    }
    request
        .tools
        .as_deref()
        .unwrap_or_default()
        .iter()
        .filter(|tool| match &request.tool_choice {
            Some(ToolChoice::Specific { function, .. }) => tool.function.name == function.name,
            _ => true,
        })
        .collect()
}

pub(crate) fn ds4_tool_schema_json(tool: &ChatTool) -> String {
    let mut fields = vec![format!(
        "\"name\": {}",
        dsv4_json(&Value::String(tool.function.name.clone()))
    )];
    if let Some(description) = tool.function.description.as_ref() {
        fields.push(format!(
            "\"description\": {}",
            dsv4_json(&Value::String(description.clone()))
        ));
    }
    if let Some(parameters) = tool.function.parameters.as_ref() {
        fields.push(format!("\"parameters\": {}", dsv4_json(parameters)));
    }
    if let Some(strict) = tool.function.strict {
        fields.push(format!("\"strict\": {strict}"));
    }
    format!("{{{}}}", fields.join(", "))
}

pub(crate) fn render_ds4_tool_call(tool_call: &ToolCall) -> String {
    let mut rendered = format!("<｜DSML｜invoke name=\"{}\">", tool_call.function.name);
    if let Ok(Value::Object(arguments)) =
        serde_json::from_str::<Value>(&tool_call.function.arguments)
    {
        for (key, value) in arguments {
            rendered.push('\n');
            rendered.push_str("<｜DSML｜parameter name=\"");
            rendered.push_str(&key);
            match value {
                Value::String(value) => {
                    rendered.push_str("\" string=\"true\">");
                    rendered.push_str(&value);
                }
                value => {
                    rendered.push_str("\" string=\"false\">");
                    rendered.push_str(&dsv4_json(&value));
                }
            }
            rendered.push_str("</｜DSML｜parameter>");
        }
    }
    rendered.push_str("\n</｜DSML｜invoke>");
    rendered
}

pub(crate) fn render_ds4_tool_calls(tool_calls: &[ToolCall]) -> String {
    format!(
        "<｜DSML｜tool_calls>\n{}\n</｜DSML｜tool_calls>",
        tool_calls
            .iter()
            .map(render_ds4_tool_call)
            .collect::<Vec<_>>()
            .join("\n")
    )
}

fn dsv4_json(value: &Value) -> String {
    match value {
        Value::Null => "null".to_owned(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => serde_json::to_string(value)
            .expect("serializing a DeepSeek V4 JSON string should not fail"),
        Value::Array(values) => format!(
            "[{}]",
            values.iter().map(dsv4_json).collect::<Vec<_>>().join(", ")
        ),
        Value::Object(values) => format!(
            "{{{}}}",
            values
                .iter()
                .map(|(key, value)| format!(
                    "{}: {}",
                    serde_json::to_string(key)
                        .expect("serializing a DeepSeek V4 JSON key should not fail"),
                    dsv4_json(value)
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn request(value: Value) -> ChatCompletionRequest {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn renders_native_chat_and_thinking_transitions() {
        let chat = request(json!({
            "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
            "messages": [
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "Hello"}
            ]
        }));
        assert_eq!(
            render_ds4_request_prompt(&chat),
            "<｜begin▁of▁sentence｜>Be terse.<｜User｜>Hello<｜Assistant｜></think>"
        );

        let thinking = request(json!({
            "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
            "thinking": {"type": "enabled"},
            "messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "reasoning_content": "Old thought.", "content": "Hi."},
                {"role": "user", "content": "Continue."}
            ]
        }));
        assert_eq!(
            render_ds4_request_prompt(&thinking),
            "<｜begin▁of▁sentence｜>Be helpful.<｜User｜>Hello<｜Assistant｜></think>Hi.<｜end▁of▁sentence｜><｜User｜>Continue.<｜Assistant｜><think>"
        );
    }

    #[test]
    fn renders_dsml_tools_and_sorted_results() {
        let request = request(json!({
            "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
            "thinking": {"type": "enabled"},
            "tools": [{
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look up a value.",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}
                }
            }],
            "messages": [
                {"role": "system", "content": "Use tools."},
                {"role": "user", "content": "Find it."},
                {"role": "assistant", "reasoning_content": "I should look.", "tool_calls": [
                    {"id": "second", "type": "function", "function": {"name": "lookup", "arguments": {"query": "B"}}},
                    {"id": "first", "type": "function", "function": {"name": "lookup", "arguments": {"query": "A"}}}
                ]},
                {"role": "tool", "tool_call_id": "first", "content": "A result"},
                {"role": "tool", "tool_call_id": "second", "content": "B result"}
            ]
        }));
        let prompt = render_ds4_request_prompt(&request);
        assert!(prompt.starts_with("<｜begin▁of▁sentence｜>Use tools.\n\n## Tools\n"));
        assert!(
            prompt.contains("<｜Assistant｜><think>I should look.</think>\n\n<｜DSML｜tool_calls>")
        );
        assert!(prompt
            .contains("<｜DSML｜parameter name=\"query\" string=\"true\">B</｜DSML｜parameter>"));
        assert!(prompt.contains(
            "<｜User｜><tool_result>B result</tool_result>\n\n<tool_result>A result</tool_result><｜Assistant｜><think>"
        ));
    }

    #[test]
    fn matches_bundled_deepseek_v4_reference_when_snapshot_is_available() {
        let Ok(snapshot) = std::env::var("DS4RT_FLASH_SNAPSHOT") else {
            return;
        };
        let encoding = std::path::Path::new(&snapshot).join("encoding/tests");
        if !encoding.is_dir() {
            return;
        }

        let mut case1: Value =
            serde_json::from_slice(&std::fs::read(encoding.join("test_input_1.json")).unwrap())
                .unwrap();
        case1["model"] = Value::String("deepseek-ai/DeepSeek-V4-Flash-0731".to_owned());
        case1["thinking"] = json!({"type": "enabled"});
        let case1: ChatCompletionRequest = serde_json::from_value(case1).unwrap();
        let expected1 = std::fs::read_to_string(encoding.join("test_output_1.txt")).unwrap();
        assert_eq!(render_ds4_request_prompt(&case1), expected1);

        let messages2: Value =
            serde_json::from_slice(&std::fs::read(encoding.join("test_input_2.json")).unwrap())
                .unwrap();
        let case2 = request(json!({
            "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
            "thinking": {"type": "enabled"},
            "messages": messages2
        }));
        let expected2 = std::fs::read_to_string(encoding.join("test_output_2.txt")).unwrap();
        assert_eq!(render_ds4_request_prompt(&case2), expected2);

        let messages3: Value =
            serde_json::from_slice(&std::fs::read(encoding.join("test_input_3.json")).unwrap())
                .unwrap();
        let case3 = request(json!({
            "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
            "thinking": {"type": "enabled"},
            "messages": messages3
        }));
        let expected3 = std::fs::read_to_string(encoding.join("test_output_3.txt")).unwrap();
        assert_eq!(render_ds4_request_prompt(&case3), expected3);

        let messages4: Value =
            serde_json::from_slice(&std::fs::read(encoding.join("test_input_4.json")).unwrap())
                .unwrap();
        let case4 = request(json!({
            "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
            "messages": messages4
        }));
        let expected4 = std::fs::read_to_string(encoding.join("test_output_4.txt")).unwrap();
        assert_eq!(render_ds4_request_prompt(&case4), expected4);
    }
}
