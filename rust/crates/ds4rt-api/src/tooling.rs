use serde_json::{Map, Value};
use uuid::Uuid;

use crate::{ToolCall, ToolCallFunction};

pub(crate) const DS4_TOOL_CALLS_START: &str = "<｜DSML｜tool_calls>";
pub(crate) const DS4_TOOL_CALLS_END: &str = "</｜DSML｜tool_calls>";
const DS4_STREAM_TOOL_CALLS_START: &str = "\n\n<｜DSML｜tool_calls>";
const DS4_INVOKE_START: &str = "<｜DSML｜invoke name=\"";
const DS4_INVOKE_END: &str = "</｜DSML｜invoke>";
const DS4_PARAMETER_START: &str = "<｜DSML｜parameter name=\"";
const DS4_PARAMETER_END: &str = "</｜DSML｜parameter>";

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ParsedToolOutput {
    pub(crate) content: Option<String>,
    pub(crate) tool_calls: Vec<ToolCall>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Ds4ToolStreamDelta {
    Content(String),
    ToolCall {
        index: usize,
        id: Option<String>,
        name: Option<String>,
        arguments: Option<String>,
    },
}

#[derive(Debug, Default)]
pub(crate) struct Ds4ToolCallStreamParser {
    pending: String,
    inside_block: bool,
    saw_tool_syntax: bool,
    completed_tool_call_ids: Vec<String>,
    completed_tool_call_values: Vec<ToolCall>,
}

impl Ds4ToolCallStreamParser {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn push(&mut self, chunk: &str) -> Vec<Ds4ToolStreamDelta> {
        self.pending.push_str(chunk);
        self.advance(false)
    }

    pub(crate) fn finish(&mut self) -> Vec<Ds4ToolStreamDelta> {
        self.advance(true)
    }

    pub(crate) fn completed_tool_calls(&self) -> usize {
        self.completed_tool_call_values.len()
    }

    pub(crate) fn completed_tool_call_ids(&self) -> &[String] {
        &self.completed_tool_call_ids
    }

    pub(crate) fn completed_tool_call_values(&self) -> &[ToolCall] {
        &self.completed_tool_call_values
    }

    fn advance(&mut self, finishing: bool) -> Vec<Ds4ToolStreamDelta> {
        let mut deltas = Vec::new();
        loop {
            if !self.inside_block {
                if let Some(start) = self.pending.find(DS4_STREAM_TOOL_CALLS_START) {
                    if !self.saw_tool_syntax && start != 0 {
                        deltas.push(Ds4ToolStreamDelta::Content(
                            self.pending[..start].to_owned(),
                        ));
                    }
                    self.pending
                        .drain(..start + DS4_STREAM_TOOL_CALLS_START.len());
                    self.inside_block = true;
                    self.saw_tool_syntax = true;
                    continue;
                }
                if self.pending.is_empty() {
                    break;
                }
                let retained = if finishing {
                    0
                } else {
                    longest_suffix_matching_prefix(&self.pending, DS4_STREAM_TOOL_CALLS_START)
                };
                let emit_len = self.pending.len() - retained;
                if emit_len == 0 {
                    break;
                }
                let visible = self.pending[..emit_len].to_owned();
                self.pending.drain(..emit_len);
                if !self.saw_tool_syntax {
                    deltas.push(Ds4ToolStreamDelta::Content(visible));
                }
                continue;
            }

            let Some(end) = self.pending.find(DS4_TOOL_CALLS_END) else {
                break;
            };
            let body = self.pending[..end].to_owned();
            self.pending.drain(..end + DS4_TOOL_CALLS_END.len());
            self.inside_block = false;
            let parsed = parse_ds4_block_body(&body);
            for tool_call in parsed {
                let index = self.completed_tool_call_values.len();
                deltas.push(Ds4ToolStreamDelta::ToolCall {
                    index,
                    id: Some(tool_call.id.clone()),
                    name: Some(tool_call.function.name.clone()),
                    arguments: Some(tool_call.function.arguments.clone()),
                });
                self.completed_tool_call_ids.push(tool_call.id.clone());
                self.completed_tool_call_values.push(tool_call);
            }
        }
        deltas
    }
}

pub(crate) fn parse_ds4_tool_calls(output: &str) -> ParsedToolOutput {
    let Some(block_start) = output.find(DS4_TOOL_CALLS_START) else {
        return ParsedToolOutput {
            content: nonempty_content(output),
            tool_calls: Vec::new(),
        };
    };
    let body_start = block_start + DS4_TOOL_CALLS_START.len();
    let Some(relative_end) = output[body_start..].find(DS4_TOOL_CALLS_END) else {
        return ParsedToolOutput {
            content: nonempty_content(output),
            tool_calls: Vec::new(),
        };
    };
    let body_end = body_start + relative_end;
    let tool_calls = parse_ds4_block_body(&output[body_start..body_end]);
    if tool_calls.is_empty() {
        return ParsedToolOutput {
            content: nonempty_content(output),
            tool_calls,
        };
    }
    ParsedToolOutput {
        content: nonempty_content(output[..block_start].trim_end()),
        tool_calls,
    }
}

fn parse_ds4_block_body(mut body: &str) -> Vec<ToolCall> {
    let mut tool_calls = Vec::new();
    while let Some(relative_start) = body.find(DS4_INVOKE_START) {
        body = &body[relative_start + DS4_INVOKE_START.len()..];
        let Some(header_end) = body.find("\">") else {
            break;
        };
        let name = &body[..header_end];
        body = &body[header_end + 2..];
        let Some(invoke_end) = body.find(DS4_INVOKE_END) else {
            break;
        };
        let arguments_body = &body[..invoke_end];
        if !name.is_empty()
            && !name
                .chars()
                .any(|character| matches!(character, '<' | '>' | '"'))
        {
            if let Some(arguments) = parse_ds4_arguments(arguments_body) {
                tool_calls.push(ToolCall {
                    id: format!("call_{}", Uuid::new_v4().simple()),
                    tool_type: "function".to_owned(),
                    function: ToolCallFunction {
                        name: name.to_owned(),
                        arguments,
                    },
                });
            }
        }
        body = &body[invoke_end + DS4_INVOKE_END.len()..];
    }
    tool_calls
}

fn parse_ds4_arguments(mut body: &str) -> Option<String> {
    let mut arguments = Map::new();
    loop {
        body = body.trim();
        if body.is_empty() {
            break;
        }
        body = body.strip_prefix(DS4_PARAMETER_START)?;
        let name_end = body.find("\" string=\"")?;
        let name = &body[..name_end];
        if name.is_empty() || arguments.contains_key(name) {
            return None;
        }
        body = &body[name_end + "\" string=\"".len()..];
        let string_end = body.find("\">")?;
        let string = &body[..string_end];
        if !matches!(string, "true" | "false") {
            return None;
        }
        body = &body[string_end + 2..];
        let value_end = body.find(DS4_PARAMETER_END)?;
        let raw_value = &body[..value_end];
        let value = if string == "true" {
            Value::String(raw_value.to_owned())
        } else {
            serde_json::from_str(raw_value.trim()).ok()?
        };
        arguments.insert(name.to_owned(), value);
        body = &body[value_end + DS4_PARAMETER_END.len()..];
    }
    serde_json::to_string(&arguments).ok()
}

fn longest_suffix_matching_prefix(text: &str, marker: &str) -> usize {
    let max_len = text.len().min(marker.len().saturating_sub(1));
    (1..=max_len)
        .rev()
        .find(|length| {
            let start = text.len() - length;
            text.is_char_boundary(start) && marker.starts_with(&text[start..])
        })
        .unwrap_or(0)
}

fn nonempty_content(content: &str) -> Option<String> {
    let content = content.trim();
    (!content.is_empty()).then(|| content.to_owned())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use serde_json::json;

    use super::*;

    const OUTPUT: &str = "I'll check.\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"lookup\">\n<｜DSML｜parameter name=\"query\" string=\"true\">Taipei</｜DSML｜parameter>\n<｜DSML｜parameter name=\"limit\" string=\"false\">3</｜DSML｜parameter>\n</｜DSML｜invoke>\n<｜DSML｜invoke name=\"refresh\">\n\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";

    #[test]
    fn parses_typed_and_zero_argument_dsml_calls() {
        let parsed = parse_ds4_tool_calls(OUTPUT);
        assert_eq!(parsed.content.as_deref(), Some("I'll check."));
        assert_eq!(parsed.tool_calls.len(), 2);
        assert_eq!(parsed.tool_calls[0].function.name, "lookup");
        assert_eq!(
            serde_json::from_str::<Value>(&parsed.tool_calls[0].function.arguments).unwrap(),
            json!({"query": "Taipei", "limit": 3})
        );
        assert_eq!(parsed.tool_calls[1].function.arguments, "{}");
    }

    #[test]
    fn malformed_or_incomplete_dsml_remains_content() {
        let output = "before <｜DSML｜tool_calls><｜DSML｜invoke name=\"lookup\">";
        let parsed = parse_ds4_tool_calls(output);
        assert!(parsed.tool_calls.is_empty());
        assert_eq!(parsed.content.as_deref(), Some(output));
    }

    #[test]
    fn incremental_parser_is_independent_of_utf8_chunk_boundaries() {
        let expected = (
            "I'll check.".to_owned(),
            vec![
                ("lookup".to_owned(), json!({"query": "Taipei", "limit": 3})),
                ("refresh".to_owned(), json!({})),
            ],
        );
        let mut boundaries = OUTPUT
            .char_indices()
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        boundaries.push(OUTPUT.len());
        for split in boundaries.iter().copied() {
            let mut parser = Ds4ToolCallStreamParser::new();
            let mut deltas = parser.push(&OUTPUT[..split]);
            deltas.extend(parser.push(&OUTPUT[split..]));
            deltas.extend(parser.finish());
            assert_eq!(stream_projection(&deltas), expected, "split={split}");
            assert_eq!(parser.completed_tool_calls(), 2, "split={split}");
        }
    }

    #[test]
    fn incremental_parser_withholds_incomplete_control_syntax() {
        let mut parser = Ds4ToolCallStreamParser::new();
        let mut deltas = parser.push("visible \n\n<｜DSML｜tool_");
        deltas.extend(parser.push("calls>\n<｜DSML｜invoke name=\"lookup\">"));
        deltas.extend(parser.finish());
        assert_eq!(stream_projection(&deltas).0, "visible ");
        assert_eq!(parser.completed_tool_calls(), 0);
    }

    fn stream_projection(deltas: &[Ds4ToolStreamDelta]) -> (String, Vec<(String, Value)>) {
        let mut content = String::new();
        let mut calls = BTreeMap::<usize, (String, String)>::new();
        for delta in deltas {
            match delta {
                Ds4ToolStreamDelta::Content(chunk) => content.push_str(chunk),
                Ds4ToolStreamDelta::ToolCall {
                    index,
                    name,
                    arguments,
                    ..
                } => {
                    let call = calls.entry(*index).or_default();
                    if let Some(name) = name {
                        call.0 = name.clone();
                    }
                    if let Some(arguments) = arguments {
                        call.1.push_str(arguments);
                    }
                }
            }
        }
        let calls = calls
            .into_values()
            .map(|(name, arguments)| {
                let arguments = serde_json::from_str(&arguments).unwrap();
                (name, arguments)
            })
            .collect();
        (content, calls)
    }
}
