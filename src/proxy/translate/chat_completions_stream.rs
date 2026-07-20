use bytes::Bytes;
use futures::stream::{Stream, StreamExt};
use serde_json::{json, Value};
use std::error::Error;
use std::pin::Pin;

use crate::config::{ProviderType, ReasoningBridge};
use crate::proxy::error_translation;
use crate::proxy::translate::{tool_id, tool_text_parser::parse_textual_tool_call};
use crate::proxy::util::{format_sse, ToolNameMap};

const UPSTREAM_STREAM_READ_TIMEOUT_SECS: u64 = 300;

/// Translates an OpenAI SSE stream to Anthropic SSE format.
///
/// OpenAI format:  `data: {"choices":[{"delta":{"content":"..."}}]}`
/// Anthropic format: multiple event types (message_start, content_block_start, content_block_delta, etc.)
pub fn translate_sse_stream<S>(
    input: S,
    tool_name_map: ToolNameMap,
) -> Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
{
    translate_sse_stream_with_reasoning(input, tool_name_map, ReasoningBridge::Off)
}

pub fn translate_sse_stream_with_reasoning<S>(
    input: S,
    tool_name_map: ToolNameMap,
    reasoning_bridge: ReasoningBridge,
) -> Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
{
    let mut state = StreamState::new(tool_name_map);
    state.reasoning_bridge = reasoning_bridge;

    let output = async_stream::stream! {
        let mut stream = std::pin::pin!(input);
        let mut buffer = String::new();
        let mut message_started = false;
        let mut saw_translatable_event = false;

        while let Some(chunk_result) = stream.next().await {
            match chunk_result {
                Ok(chunk) => {
                    buffer.push_str(&String::from_utf8_lossy(&chunk));

                    // Process complete SSE lines
                    while let Some(pos) = buffer.find("\n\n") {
                        let line = buffer[..pos].to_string();
                        buffer = buffer[pos + 2..].to_string();

                        if let Some(events) = state.process_openai_line(&line) {
                            for event in events {
                                if event.starts_with("event: error") {
                                    yield Ok(Bytes::from(event));
                                    return;
                                }
                                if !message_started {
                                    yield Ok(Bytes::from(message_start_event()));
                                    message_started = true;
                                }
                                saw_translatable_event = true;
                                yield Ok(Bytes::from(event));
                            }
                        }
                    }
                    // Also handle single newline delimited chunks
                    while let Some(pos) = buffer.find('\n') {
                        let line = buffer[..pos].to_string();
                        buffer = buffer[pos + 1..].to_string();

                        if line.is_empty() {
                            continue;
                        }

                        if let Some(events) = state.process_openai_line(&line) {
                            for event in events {
                                if event.starts_with("event: error") {
                                    yield Ok(Bytes::from(event));
                                    return;
                                }
                                if !message_started {
                                    yield Ok(Bytes::from(message_start_event()));
                                    message_started = true;
                                }
                                saw_translatable_event = true;
                                yield Ok(Bytes::from(event));
                            }
                        }
                    }
                }
                Err(e) => {
                    log_stream_read_error(&e, &state);
                    if !state.buffered_text.is_empty() {
                        for event in state.flush_buffered_text_or_tool() {
                            if !message_started {
                                yield Ok(Bytes::from(message_start_event()));
                                message_started = true;
                            }
                            yield Ok(Bytes::from(event));
                        }
                    }
                    if state.thinking_block_started {
                        yield Ok(Bytes::from(format_sse("content_block_stop", &json!({
                            "type": "content_block_stop",
                            "index": state.block_index,
                        }))));
                        state.block_index += 1;
                        state.thinking_block_started = false;
                    }
                    if state.block_started {
                        yield Ok(Bytes::from(format_sse("content_block_stop", &json!({
                            "type": "content_block_stop",
                            "index": state.block_index,
                        }))));
                    }
                    yield Ok(Bytes::from(format_stream_read_error_event(&e, &state)));
                    return;
                }
            }
        }

        if !state.buffered_text.is_empty() {
            for event in state.flush_buffered_text_or_tool() {
                if !message_started {
                    yield Ok(Bytes::from(message_start_event()));
                    message_started = true;
                }
                saw_translatable_event = true;
                yield Ok(Bytes::from(event));
            }
        }

        if !saw_translatable_event {
            yield Ok(Bytes::from(error_translation::from_empty_stream(ProviderType::OpenAICompatible, None).sse()));
            return;
        }

        // Send final events
        if state.thinking_block_started {
            let block_stop = format_sse("content_block_stop", &json!({
                "type": "content_block_stop",
                "index": state.block_index,
            }));
            yield Ok(Bytes::from(block_stop));
            state.block_index += 1;
            state.thinking_block_started = false;
        }
        if state.block_started {
            let block_stop = format_sse("content_block_stop", &json!({
                "type": "content_block_stop",
                "index": state.block_index,
            }));
            yield Ok(Bytes::from(block_stop));
        }

        let msg_delta = format_sse("message_delta", &json!({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": null},
            "usage": {"output_tokens": state.output_tokens}
        }));
        yield Ok(Bytes::from(msg_delta));

        yield Ok(Bytes::from(format_sse("message_stop", &json!({"type": "message_stop"}))));
    };

    Box::pin(output)
}

fn message_start_event() -> String {
    format_sse(
        "message_start",
        &json!({
            "type": "message_start",
            "message": {
                "id": format!("msg_{}", uuid::Uuid::new_v4()),
                "type": "message",
                "role": "assistant",
                "model": "claudex-proxy",
                "content": [],
                "stop_reason": null,
                "stop_sequence": null,
                "usage": {"input_tokens": 0, "output_tokens": 0}
            }
        }),
    )
}

fn format_stream_read_error_event(error: &reqwest::Error, state: &StreamState) -> String {
    error_translation::from_stream_transport(&stream_read_error_message(error, state), None).sse()
}

fn stream_read_error_message(error: &reqwest::Error, state: &StreamState) -> String {
    if error.is_timeout() {
        return format!(
            "upstream stream read error: claudex upstream stream read timed out after {UPSTREAM_STREAM_READ_TIMEOUT_SECS}s without receiving data; {}. This is usually an upstream stream that stopped producing data long enough to hit claudex's idle read timeout, not an auth, rate-limit, or upstream HTTP status failure",
            stream_progress_message(state)
        );
    }

    let root = source_chain(error)
        .into_iter()
        .last()
        .unwrap_or_else(|| error.to_string());

    if state.saw_upstream_data {
        return format!(
            "upstream stream read error: upstream connection closed before the stream completed; {}; root cause: {root}. This is usually a mid-stream transport interruption after a successful upstream response, not an auth, rate-limit, or upstream HTTP status failure",
            stream_progress_message(state)
        );
    }

    format!("upstream stream read error: {root}")
}

fn stream_progress_message(state: &StreamState) -> &'static str {
    if state.saw_text_delta {
        "upstream returned 200 OK and was still streaming text"
    } else if state.saw_upstream_data {
        "upstream returned 200 OK and started streaming"
    } else {
        "upstream returned 200 OK but no translatable stream content was received"
    }
}

fn source_chain(error: &(dyn Error + 'static)) -> Vec<String> {
    let mut chain = Vec::new();
    let mut source = error.source();
    while let Some(err) = source {
        chain.push(err.to_string());
        source = err.source();
    }
    chain
}

fn log_stream_read_error(error: &reqwest::Error, state: &StreamState) {
    tracing::warn!(
        error = %error,
        error_debug = ?error,
        is_decode = error.is_decode(),
        is_timeout = error.is_timeout(),
        is_body = error.is_body(),
        is_connect = error.is_connect(),
        source_chain = ?source_chain(error),
        block_started = state.block_started,
        current_tool_name = ?state.current_tool_call.as_ref().map(|tool| tool.name.as_str()),
        saw_upstream_data = state.saw_upstream_data,
        saw_text_delta = state.saw_text_delta,
        "Chat Completions stream read error"
    );
}

fn sanitize_tool_input(tool_name: &str, mut input: Value) -> Value {
    if tool_name == "Read" {
        sanitize_read_pages(&mut input);
    }
    input
}

fn sanitize_read_pages(input: &mut Value) {
    let Some(obj) = input.as_object_mut() else {
        return;
    };
    let pages = obj.get("pages").and_then(|v| v.as_str());
    let file_path = obj.get("file_path").and_then(|v| v.as_str()).unwrap_or("");
    let is_pdf = file_path.to_ascii_lowercase().ends_with(".pdf");
    if pages == Some("") || !is_pdf {
        obj.remove("pages");
    }
}

struct StreamState {
    block_index: usize,
    block_started: bool,
    output_tokens: u64,
    current_tool_call: Option<ToolCallState>,
    tool_name_map: ToolNameMap,
    saw_upstream_data: bool,
    saw_text_delta: bool,
    reasoning_bridge: ReasoningBridge,
    thinking_block_started: bool,
    buffered_text: String,
}

struct ToolCallState {
    id: String,
    name: String,
    arguments_buffer: String,
}

impl StreamState {
    fn new(tool_name_map: ToolNameMap) -> Self {
        Self {
            block_index: 0,
            block_started: false,
            output_tokens: 0,
            current_tool_call: None,
            tool_name_map,
            saw_upstream_data: false,
            saw_text_delta: false,
            reasoning_bridge: ReasoningBridge::Off,
            thinking_block_started: false,
            buffered_text: String::new(),
        }
    }

    fn process_openai_line(&mut self, line: &str) -> Option<Vec<String>> {
        let data = line.strip_prefix("data: ")?.trim();

        if data == "[DONE]" {
            if !self.buffered_text.is_empty() {
                return Some(self.flush_buffered_text_or_tool());
            }
            return self.finalize_tool_call();
        }

        let parsed: Value = serde_json::from_str(data).ok()?;
        self.saw_upstream_data = true;
        if parsed.get("error").is_some() {
            let err =
                error_translation::from_http_status(axum::http::StatusCode::BAD_GATEWAY, data);
            return Some(vec![err.sse()]);
        }
        let choice = parsed.get("choices")?.as_array()?.first()?;
        let delta = choice.get("delta")?;

        let mut events = Vec::new();

        // Track usage
        if let Some(usage) = parsed.get("usage") {
            if let Some(tokens) = usage.get("completion_tokens").and_then(|t| t.as_u64()) {
                self.output_tokens = tokens;
            }
        }

        if self.reasoning_bridge == ReasoningBridge::VisibleThinking {
            if let Some(reasoning) = delta
                .get("reasoning_content")
                .or_else(|| delta.get("reasoning"))
                .or_else(|| delta.get("thinking"))
                .and_then(|c| c.as_str())
            {
                if !reasoning.is_empty() {
                    if self.block_started {
                        events.push(format_sse(
                            "content_block_stop",
                            &json!({"type":"content_block_stop","index": self.block_index}),
                        ));
                        self.block_index += 1;
                        self.block_started = false;
                    }
                    if !self.thinking_block_started {
                        events.push(format_sse(
                            "content_block_start",
                            &json!({
                                "type": "content_block_start",
                                "index": self.block_index,
                                "content_block": {"type": "thinking", "thinking": ""}
                            }),
                        ));
                        self.thinking_block_started = true;
                    }
                    events.push(format_sse(
                        "content_block_delta",
                        &json!({
                            "type": "content_block_delta",
                            "index": self.block_index,
                            "delta": {"type": "thinking_delta", "thinking": reasoning}
                        }),
                    ));
                }
            }
        }

        // Handle text content
        if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
            if !content.is_empty() {
                // Finalize any pending tool call first
                if let Some(tool_events) = self.finalize_tool_call() {
                    events.extend(tool_events);
                }

                if !self.block_started
                    && self.current_tool_call.is_none()
                    && should_buffer_textual_tool_candidate(&self.buffered_text, content)
                {
                    self.buffered_text.push_str(content);
                } else if !self.buffered_text.is_empty() {
                    self.buffered_text.push_str(content);
                    events.extend(self.flush_buffered_text_or_tool());
                } else {
                    self.saw_text_delta = true;
                    if self.thinking_block_started {
                        events.push(format_sse(
                            "content_block_stop",
                            &json!({"type":"content_block_stop","index": self.block_index}),
                        ));
                        self.block_index += 1;
                        self.thinking_block_started = false;
                    }
                    if !self.block_started || self.current_tool_call.is_some() {
                        let block_start = format_sse(
                            "content_block_start",
                            &json!({
                                "type": "content_block_start",
                                "index": self.block_index,
                                "content_block": {"type": "text", "text": ""}
                            }),
                        );
                        events.push(block_start);
                        self.block_started = true;
                    }

                    let block_delta = format_sse(
                        "content_block_delta",
                        &json!({
                            "type": "content_block_delta",
                            "index": self.block_index,
                            "delta": {"type": "text_delta", "text": content}
                        }),
                    );
                    events.push(block_delta);
                }
            }
        }

        // Handle tool calls
        if let Some(tool_calls) = delta.get("tool_calls").and_then(|tc| tc.as_array()) {
            for tc in tool_calls {
                let empty_func = json!({});
                let func = tc.get("function").unwrap_or(&empty_func);

                // New tool call starts
                if let Some(id) = tc.get("id").and_then(|id| id.as_str()) {
                    // Finalize previous blocks
                    if self.thinking_block_started {
                        events.push(format_sse(
                            "content_block_stop",
                            &json!({"type":"content_block_stop","index": self.block_index}),
                        ));
                        self.block_index += 1;
                        self.thinking_block_started = false;
                    }
                    if self.block_started {
                        events.push(format_sse(
                            "content_block_stop",
                            &json!({
                                "type": "content_block_stop",
                                "index": self.block_index,
                            }),
                        ));
                        self.block_index += 1;
                        self.block_started = false;
                    }
                    if let Some(prev_events) = self.finalize_tool_call() {
                        events.extend(prev_events);
                    }

                    let truncated_name = func.get("name").and_then(|n| n.as_str()).unwrap_or("");
                    // 还原被截断的工具名
                    let name = self
                        .tool_name_map
                        .get(truncated_name)
                        .cloned()
                        .unwrap_or_else(|| truncated_name.to_string());

                    self.current_tool_call = Some(ToolCallState {
                        id: id.to_string(),
                        name: name.clone(),
                        arguments_buffer: String::new(),
                    });

                    events.push(format_sse(
                        "content_block_start",
                        &json!({
                            "type": "content_block_start",
                            "index": self.block_index,
                            "content_block": {
                                "type": "tool_use",
                                "id": tool_id::to_anthropic_tool_id(id),
                                "name": name,
                                "input": {}
                            }
                        }),
                    ));
                    self.block_started = true;
                }

                // Accumulate arguments
                if let Some(args) = func.get("arguments").and_then(|a| a.as_str()) {
                    if let Some(ref mut tool_state) = self.current_tool_call {
                        tool_state.arguments_buffer.push_str(args);
                        if tool_state.name != "Read" {
                            events.push(format_sse(
                                "content_block_delta",
                                &json!({
                                    "type": "content_block_delta",
                                    "index": self.block_index,
                                    "delta": {
                                        "type": "input_json_delta",
                                        "partial_json": args
                                    }
                                }),
                            ));
                        }
                    }
                }
            }
        }

        // Handle finish_reason
        if let Some(finish) = choice.get("finish_reason").and_then(|f| f.as_str()) {
            if finish == "tool_calls" {
                if let Some(tool_events) = self.finalize_tool_call() {
                    events.extend(tool_events);
                }
            } else if !self.buffered_text.is_empty() {
                events.extend(self.flush_buffered_text_or_tool());
            }
        }

        if events.is_empty() {
            None
        } else {
            Some(events)
        }
    }

    fn flush_buffered_text_or_tool(&mut self) -> Vec<String> {
        let text = std::mem::take(&mut self.buffered_text);
        if let Some(parsed) = parse_textual_tool_call(&text) {
            self.has_textual_tool_use();
            let events = vec![
                format_sse(
                    "content_block_start",
                    &json!({
                        "type":"content_block_start",
                        "index": self.block_index,
                        "content_block": {
                            "type":"tool_use",
                            "id": tool_id::to_anthropic_tool_id("call_text_1"),
                            "name": parsed.name,
                            "input": sanitize_tool_input(&parsed.name, parsed.input)
                        }
                    }),
                ),
                format_sse(
                    "content_block_stop",
                    &json!({"type":"content_block_stop","index": self.block_index}),
                ),
            ];
            self.block_index += 1;
            return events;
        }
        self.saw_text_delta = true;
        self.block_started = true;
        vec![
            format_sse(
                "content_block_start",
                &json!({"type":"content_block_start","index": self.block_index,"content_block":{"type":"text","text":""}}),
            ),
            format_sse(
                "content_block_delta",
                &json!({"type":"content_block_delta","index": self.block_index,"delta":{"type":"text_delta","text": text}}),
            ),
        ]
    }

    fn has_textual_tool_use(&mut self) {
        self.saw_text_delta = true;
        self.block_started = false;
    }

    fn finalize_tool_call(&mut self) -> Option<Vec<String>> {
        let tool_state = self.current_tool_call.take()?;
        let mut events = Vec::new();

        if tool_state.name == "Read" {
            let input = sanitize_tool_input(
                &tool_state.name,
                serde_json::from_str(&tool_state.arguments_buffer).unwrap_or_else(|_| json!({})),
            );
            if input != json!({}) {
                events.push(format_sse(
                    "content_block_delta",
                    &json!({
                        "type": "content_block_delta",
                        "index": self.block_index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": serde_json::to_string(&input).unwrap_or_default(),
                        }
                    }),
                ));
            }
        }

        if self.block_started {
            events.push(format_sse(
                "content_block_stop",
                &json!({
                    "type": "content_block_stop",
                    "index": self.block_index,
                }),
            ));
            self.block_index += 1;
            self.block_started = false;
        }

        Some(events)
    }
}

fn should_buffer_textual_tool_candidate(existing: &str, delta: &str) -> bool {
    let candidate = format!("{existing}{delta}");
    let text = candidate.trim_start();
    if text.starts_with("<tool_use") {
        return !text.contains("</tool_use>");
    }
    if text.starts_with("<function=") {
        return !text.contains("</function>");
    }
    if text.starts_with("<tool_call>") {
        return !text.contains("</tool_call>");
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_untranslatable_stream_returns_error_before_message_start() {
        let input = futures::stream::iter(vec![Ok(Bytes::from_static(
            b"data: {\"choices\":[{\"delta\":{}}]}\n\n",
        ))]);
        let output = translate_sse_stream(input, ToolNameMap::new())
            .collect::<Vec<_>>()
            .await;
        let body = String::from_utf8_lossy(output[0].as_ref().unwrap()).to_string();
        assert!(body.contains("event: error"));
        assert!(body.contains("empty or untranslatable stream"));
        assert!(!body.contains("message_start"));
    }

    #[test]
    fn test_error_json_maps_to_anthropic_error() {
        let mut state = StreamState::new(std::collections::HashMap::new());
        let line = format!(
            "data: {}",
            json!({"error": {"message": "rate_limit_exceeded", "type": "rate_limit_error"}})
        );
        let events = state.process_openai_line(&line).unwrap();
        assert_eq!(events.len(), 1);
        assert!(events[0].contains("event: error"));
        assert!(events[0].contains("rate_limit_error"));
    }

    #[test]
    fn test_process_text_delta() {
        let mut state = StreamState::new(std::collections::HashMap::new());
        let line = format!(
            "data: {}",
            json!({
                "choices": [{"delta": {"content": "Hello"}}]
            })
        );
        let events = state.process_openai_line(&line).unwrap();
        // Should emit content_block_start + content_block_delta
        assert_eq!(events.len(), 2);
        assert!(events[0].contains("content_block_start"));
        assert!(events[1].contains("text_delta"));
        assert!(events[1].contains("Hello"));
        assert!(state.block_started);
    }

    #[test]
    fn test_subsequent_text_delta_no_block_start() {
        let mut state = StreamState::new(std::collections::HashMap::new());
        state.block_started = true; // simulate already started
        let line = format!(
            "data: {}",
            json!({"choices": [{"delta": {"content": "world"}}]})
        );
        let events = state.process_openai_line(&line).unwrap();
        // Only content_block_delta, no start
        assert_eq!(events.len(), 1);
        assert!(events[0].contains("text_delta"));
    }

    #[test]
    fn test_empty_content_ignored() {
        let mut state = StreamState::new(std::collections::HashMap::new());
        let line = format!("data: {}", json!({"choices": [{"delta": {"content": ""}}]}));
        assert!(state.process_openai_line(&line).is_none());
    }

    #[test]
    fn test_done_marker() {
        let mut state = StreamState::new(std::collections::HashMap::new());
        let result = state.process_openai_line("data: [DONE]");
        // No tool call pending, so None
        assert!(result.is_none());
    }

    #[test]
    fn test_invalid_json_returns_none() {
        let mut state = StreamState::new(std::collections::HashMap::new());
        assert!(state.process_openai_line("data: {invalid}").is_none());
    }

    #[test]
    fn test_no_data_prefix_returns_none() {
        let mut state = StreamState::new(std::collections::HashMap::new());
        assert!(state.process_openai_line("not a data line").is_none());
    }

    #[test]
    fn test_tool_call_start() {
        let mut state = StreamState::new(std::collections::HashMap::new());
        let line = format!(
            "data: {}",
            json!({
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "id": "call_1",
                            "function": {"name": "search", "arguments": "{\"q\":"}
                        }]
                    }
                }]
            })
        );
        let events = state.process_openai_line(&line).unwrap();
        // Should have content_block_start (tool_use) + content_block_delta (input_json_delta)
        assert!(events.iter().any(|e| e.contains("tool_use")));
        assert!(events.iter().any(|e| e.contains("input_json_delta")));
        assert!(state.current_tool_call.is_some());
    }

    #[test]
    fn test_tool_call_argument_accumulation() {
        let mut state = StreamState::new(std::collections::HashMap::new());
        state.current_tool_call = Some(ToolCallState {
            id: "call_1".to_string(),
            name: "search".to_string(),
            arguments_buffer: "{\"q\":".to_string(),
        });
        state.block_started = true;

        let line = format!(
            "data: {}",
            json!({
                "choices": [{
                    "delta": {
                        "tool_calls": [{"function": {"arguments": "\"rust\"}"}}]
                    }
                }]
            })
        );
        let events = state.process_openai_line(&line).unwrap();
        assert!(events.iter().any(|e| e.contains("input_json_delta")));
        assert_eq!(
            state.current_tool_call.as_ref().unwrap().arguments_buffer,
            "{\"q\":\"rust\"}"
        );
    }

    #[test]
    fn test_finish_reason_tool_calls_finalizes() {
        let mut state = StreamState::new(std::collections::HashMap::new());
        state.current_tool_call = Some(ToolCallState {
            id: "call_1".to_string(),
            name: "search".to_string(),
            arguments_buffer: "{}".to_string(),
        });
        state.block_started = true;

        let line = format!(
            "data: {}",
            json!({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
        );
        let events = state.process_openai_line(&line).unwrap();
        assert!(events.iter().any(|e| e.contains("content_block_stop")));
        assert!(state.current_tool_call.is_none());
    }

    #[test]
    fn test_edit_stream_preserves_pages_field_and_streams_deltas() {
        let mut state = StreamState::new(std::collections::HashMap::new());
        let start = format!(
            "data: {}",
            json!({
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "id": "call_edit",
                            "function": {"name": "Edit", "arguments": "{\"file_path\":\"/tmp/a.md\","}
                        }]
                    }
                }]
            })
        );
        let delta = format!(
            "data: {}",
            json!({
                "choices": [{
                    "delta": {
                        "tool_calls": [{"function": {"arguments": "\"old_string\":\"a\",\"new_string\":\"b\",\"pages\":\"\"}"}}]
                    }
                }]
            })
        );
        let done = format!(
            "data: {}",
            json!({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
        );

        let mut output = Vec::new();
        output.extend(state.process_openai_line(&start).unwrap());
        output.extend(state.process_openai_line(&delta).unwrap_or_default());
        output.extend(state.process_openai_line(&done).unwrap());
        let rendered = output.join("\n");
        assert!(rendered.contains("Edit"));
        assert!(rendered.contains("input_json_delta"));
        assert!(rendered.contains("old_string"));
        assert!(rendered.contains("new_string"));
        assert!(rendered.contains("pages"));
    }

    #[test]
    fn test_read_stream_strips_invalid_pages() {
        let mut state = StreamState::new(std::collections::HashMap::new());
        let start = format!(
            "data: {}",
            json!({
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "id": "call_read",
                            "function": {"name": "Read", "arguments": "{\"file_path\":\"/tmp/a.md\","}
                        }]
                    }
                }]
            })
        );
        let delta = format!(
            "data: {}",
            json!({
                "choices": [{
                    "delta": {
                        "tool_calls": [{"function": {"arguments": "\"pages\":\"\",\"limit\":10}"}}]
                    }
                }]
            })
        );
        let done = format!(
            "data: {}",
            json!({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
        );

        let mut output = Vec::new();
        output.extend(state.process_openai_line(&start).unwrap());
        output.extend(state.process_openai_line(&delta).unwrap_or_default());
        output.extend(state.process_openai_line(&done).unwrap());
        let rendered = output.join("\n");
        assert!(rendered.contains("/tmp/a.md"));
        assert!(!rendered.contains("pages"));
    }

    #[test]
    fn test_read_pdf_stream_keeps_pages() {
        let mut state = StreamState::new(std::collections::HashMap::new());
        let start = format!(
            "data: {}",
            json!({
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "id": "call_read",
                            "function": {"name": "Read", "arguments": "{\"file_path\":\"/tmp/a.pdf\","}
                        }]
                    }
                }]
            })
        );
        let delta = format!(
            "data: {}",
            json!({
                "choices": [{
                    "delta": {
                        "tool_calls": [{"function": {"arguments": "\"pages\":\"1-2\"}"}}]
                    }
                }]
            })
        );
        let done = format!(
            "data: {}",
            json!({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
        );

        let mut output = Vec::new();
        output.extend(state.process_openai_line(&start).unwrap());
        output.extend(state.process_openai_line(&delta).unwrap_or_default());
        output.extend(state.process_openai_line(&done).unwrap());
        let rendered = output.join("\n");
        assert!(rendered.contains("/tmp/a.pdf"));
        assert!(rendered.contains("pages"));
        assert!(rendered.contains("1-2"));
    }

    #[tokio::test]
    async fn test_stream_timeout_error_explains_claudex_upstream_timeout() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let _accepted = listener.accept().await.unwrap();
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
        });

        let error = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(10))
            .build()
            .unwrap()
            .get(format!("http://{addr}/"))
            .send()
            .await
            .unwrap_err();
        assert!(error.is_timeout());
        server.abort();

        let input = futures::stream::iter(vec![
            Ok(Bytes::from(
                "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n",
            )),
            Err(error),
        ]);
        let mut stream = translate_sse_stream(input, ToolNameMap::new());

        let mut output = String::new();
        while let Some(chunk) = stream.next().await {
            output.push_str(std::str::from_utf8(&chunk.unwrap()).unwrap());
        }

        assert!(output.contains("event: error"));
        assert!(output.contains("claudex upstream stream read timed out after 300s"));
        assert!(output.contains("upstream returned 200 OK and was still streaming text"));
        assert!(output.contains("idle read timeout"));
    }

    #[test]
    fn test_reasoning_content_visible_thinking() {
        let mut state = StreamState::new(std::collections::HashMap::new());
        state.reasoning_bridge = ReasoningBridge::VisibleThinking;
        let line = format!(
            "data: {}",
            json!({"choices":[{"delta":{"reasoning_content":"think"}}]})
        );
        let events = state.process_openai_line(&line).unwrap();
        assert!(events.join("\n").contains("thinking_delta"));
        assert!(events.join("\n").contains("think"));
    }

    #[test]
    fn test_streamed_textual_tool_call_recovers_tool_use() {
        let mut state = StreamState::new(std::collections::HashMap::new());
        state
            .process_openai_line(&format!(
                "data: {}",
                json!({"choices":[{"delta":{"content":"<function=Read>{\"file_path\":"}}]})
            ))
            .unwrap_or_default();
        let events = state
            .process_openai_line(&format!(
                "data: {}",
                json!({"choices":[{"delta":{"content":"\"/tmp/a.md\"}</function>"},"finish_reason":"stop"}]})
            ))
            .unwrap();
        let rendered = events.join("\n");
        assert!(rendered.contains("tool_use"));
        assert!(rendered.contains("toolu_call_text_1"));
        assert!(rendered.contains("/tmp/a.md"));
    }

    #[test]
    fn test_function_like_text_streams_as_text() {
        let mut state = StreamState::new(std::collections::HashMap::new());
        let events = state
            .process_openai_line(&format!(
                "data: {}",
                json!({"choices":[{"delta":{"content":"Read({not a tool yet"}}]})
            ))
            .unwrap();
        let rendered = events.join("\n");
        assert!(rendered.contains("text_delta"));
        assert!(rendered.contains("Read({not a tool yet"));
    }

    #[test]
    fn test_usage_tracking() {
        let mut state = StreamState::new(std::collections::HashMap::new());
        let line = format!(
            "data: {}",
            json!({
                "choices": [{"delta": {"content": "hi"}}],
                "usage": {"completion_tokens": 42}
            })
        );
        state.process_openai_line(&line);
        assert_eq!(state.output_tokens, 42);
    }

    #[test]
    fn test_finalize_tool_call_no_pending() {
        let mut state = StreamState::new(std::collections::HashMap::new());
        assert!(state.finalize_tool_call().is_none());
    }

    #[test]
    fn test_block_index_increments() {
        let mut state = StreamState::new(std::collections::HashMap::new());
        assert_eq!(state.block_index, 0);

        // Start a text block
        let line1 = format!(
            "data: {}",
            json!({"choices": [{"delta": {"content": "hi"}}]})
        );
        state.process_openai_line(&line1);
        assert_eq!(state.block_index, 0); // still 0 during first block

        // Start a tool call (should close text block and increment)
        let line2 = format!(
            "data: {}",
            json!({
                "choices": [{
                    "delta": {
                        "tool_calls": [{"id": "c1", "function": {"name": "f"}}]
                    }
                }]
            })
        );
        state.process_openai_line(&line2);
        assert_eq!(state.block_index, 1); // incremented after closing text block
    }
}
