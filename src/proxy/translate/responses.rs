use std::collections::HashMap;

use anyhow::{bail, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::config::ReasoningBridge;
use crate::proxy::translate::tool_id;
use crate::proxy::translate::tool_text_parser::parse_textual_tool_call;
use crate::proxy::util::{truncate_tool_name, ToolNameMap};

const DEDUPE_FUNCTION_OUTPUT_MIN_BYTES: usize = 4096;
const DEDUPE_KEEP_RECENT_INPUT_ITEMS: usize = 64;

pub fn request_has_current_image(anthropic: &Value) -> bool {
    anthropic
        .get("messages")
        .and_then(|m| m.as_array())
        .and_then(|messages| messages.last())
        .and_then(|message| message.get("content"))
        .is_some_and(content_has_image)
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ResponsesTranslationOptions {
    pub reasoning_bridge: ReasoningBridge,
}

/// Convert Anthropic Messages API request → OpenAI Responses API request
pub fn anthropic_to_responses(
    anthropic: &Value,
    default_model: &str,
) -> Result<(Value, ToolNameMap)> {
    anthropic_to_responses_with_options(
        anthropic,
        default_model,
        ResponsesTranslationOptions::default(),
    )
}

pub fn anthropic_to_responses_with_options(
    anthropic: &Value,
    default_model: &str,
    options: ResponsesTranslationOptions,
) -> Result<(Value, ToolNameMap)> {
    let mut tool_name_map: ToolNameMap = HashMap::new();
    let mut input: Vec<Value> = Vec::new();

    // Convert messages → input items
    if let Some(msgs) = anthropic.get("messages").and_then(|m| m.as_array()) {
        for msg in msgs {
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
            let content = msg.get("content");

            match role {
                "user" => {
                    // Check if this is a tool_result message
                    let has_tool_result = content.and_then(|c| c.as_array()).is_some_and(|arr| {
                        arr.iter()
                            .any(|b| b.get("type").and_then(|t| t.as_str()) == Some("tool_result"))
                    });

                    if has_tool_result {
                        if let Some(blocks) = content.and_then(|c| c.as_array()) {
                            for block in blocks {
                                if block.get("type").and_then(|t| t.as_str()) == Some("tool_result")
                                {
                                    let call_id = block
                                        .get("tool_use_id")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("call_0");
                                    let output = extract_tool_result_content(block);
                                    let images = extract_tool_result_images(block);
                                    input.push(json!({
                                        "type": "function_call_output",
                                        "call_id": tool_id::to_upstream_tool_id(call_id),
                                        "output": output,
                                    }));
                                    if !images.is_empty() {
                                        input.push(json!({
                                            "role": "user",
                                            "type": "message",
                                            "content": images,
                                        }));
                                    }
                                }
                            }
                        }
                    } else {
                        let parts = convert_user_content(content)?;
                        input.push(json!({
                            "role": "user",
                            "type": "message",
                            "content": parts,
                        }));
                    }
                }
                "assistant" => {
                    // Assistant messages may contain text and tool_use blocks
                    let content_array = match content {
                        Some(Value::Array(arr)) => arr.clone(),
                        Some(Value::String(s)) => vec![json!({"type": "text", "text": s})],
                        _ => vec![],
                    };

                    let mut text_parts = Vec::new();
                    for block in &content_array {
                        let block_type = block.get("type").and_then(|t| t.as_str());
                        match block_type {
                            Some("text") => {
                                if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                                    text_parts.push(json!({
                                        "type": "output_text",
                                        "text": text,
                                        "annotations": [],
                                    }));
                                }
                            }
                            Some("tool_use") => {
                                // tool_use → function_call (separate input item)
                                // First, flush text parts as a message
                                if !text_parts.is_empty() {
                                    input.push(json!({
                                        "type": "message",
                                        "role": "assistant",
                                        "status": "completed",
                                        "content": text_parts,
                                    }));
                                    text_parts = Vec::new();
                                }

                                let name = block
                                    .get("name")
                                    .and_then(|n| n.as_str())
                                    .unwrap_or("unknown");
                                let id =
                                    block.get("id").and_then(|i| i.as_str()).unwrap_or("call_0");
                                let truncated = truncate_tool_name(name);
                                if truncated != name {
                                    tool_name_map.insert(truncated.clone(), name.to_string());
                                }
                                let arguments = block
                                    .get("input")
                                    .map(|v| serde_json::to_string(v).unwrap_or_default())
                                    .unwrap_or_else(|| "{}".to_string());

                                input.push(json!({
                                    "type": "function_call",
                                    "call_id": tool_id::to_upstream_tool_id(id),
                                    "name": truncated,
                                    "arguments": arguments,
                                    "status": "completed",
                                }));
                            }
                            _ => {}
                        }
                    }
                    // Flush remaining text parts
                    if !text_parts.is_empty() {
                        input.push(json!({
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": text_parts,
                        }));
                    }
                }
                _ => {
                    // Generic user message fallback
                    let text = match content {
                        Some(Value::String(s)) => s.clone(),
                        _ => String::new(),
                    };
                    if !text.is_empty() {
                        input.push(json!({
                            "role": "user",
                            "type": "message",
                            "content": [{"type": "input_text", "text": text}],
                        }));
                    }
                }
            }
        }
    }

    dedupe_old_function_call_outputs(&mut input);

    // System prompt → instructions
    let instructions = anthropic
        .get("system")
        .map(|s| match s {
            Value::String(s) => s.clone(),
            Value::Array(parts) => parts
                .iter()
                .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("\n"),
            _ => String::new(),
        })
        .unwrap_or_default();

    // Model
    let model = anthropic
        .get("model")
        .and_then(|m| m.as_str())
        .map(strip_context_window_suffix)
        .unwrap_or(default_model);

    // Build request body
    let mut body = json!({
        "model": model,
        "input": input,
        "stream": anthropic.get("stream").and_then(|s| s.as_bool()).unwrap_or(false),
        "store": false,
    });

    if !instructions.is_empty() {
        body["instructions"] = json!(instructions);
    }

    apply_reasoning_bridge(&mut body, anthropic, options.reasoning_bridge);
    apply_text_format(&mut body, anthropic)?;
    apply_prompt_cache_key(&mut body, anthropic);

    // 注意：ChatGPT 后端不支持 max_output_tokens，跳过该参数

    // temperature, top_p
    if let Some(temp) = anthropic.get("temperature") {
        body["temperature"] = temp.clone();
    }
    if let Some(top_p) = anthropic.get("top_p") {
        body["top_p"] = top_p.clone();
    }

    // Tools
    if let Some(tools) = anthropic.get("tools").and_then(|t| t.as_array()) {
        let mut resp_tools: Vec<Value> = Vec::new();
        for tool in tools {
            reject_unsupported_server_tool(tool)?;
            let name = tool
                .get("name")
                .and_then(|n| n.as_str())
                .unwrap_or("unknown");
            let truncated = truncate_tool_name(name);
            if truncated != name {
                tool_name_map.insert(truncated.clone(), name.to_string());
            }
            resp_tools.push(json!({
                    "type": "function",
                    "name": truncated,
                    "description": tool.get("description").cloned().unwrap_or(json!("")),
                    "parameters": tool.get("input_schema").cloned().unwrap_or(json!({"type": "object"})),
                }));
        }
        body["tools"] = json!(resp_tools);
    }

    // tool_choice
    if let Some(tc) = anthropic.get("tool_choice") {
        let tc_type = tc.get("type").and_then(|t| t.as_str()).unwrap_or("auto");
        body["tool_choice"] = match tc_type {
            "auto" => json!("auto"),
            "any" => json!("required"),
            "none" => json!("none"),
            "tool" => {
                let name = tc.get("name").and_then(|n| n.as_str()).unwrap_or("");
                let truncated = truncate_tool_name(name);
                json!({"type": "function", "name": truncated})
            }
            _ => json!("auto"),
        };
    }

    Ok((body, tool_name_map))
}

fn apply_reasoning_bridge(body: &mut Value, anthropic: &Value, mode: ReasoningBridge) {
    if mode == ReasoningBridge::Off {
        return;
    }

    let effort = anthropic
        .pointer("/output_config/effort")
        .and_then(|value| value.as_str())
        .filter(|effort| matches!(*effort, "low" | "medium" | "high"));

    if let Some(effort) = effort {
        body["reasoning"] = json!({"effort": effort});
    }
}

fn apply_text_format(body: &mut Value, anthropic: &Value) -> Result<()> {
    if let Some(format) = anthropic.pointer("/output_config/format") {
        validate_text_format(format)?;
        let mut format = format.clone();
        if format.get("type").and_then(|v| v.as_str()) == Some("json_schema")
            && format
                .get("name")
                .is_none_or(|v| v.as_str().is_none_or(|s| s.is_empty()))
        {
            format["name"] = json!("claude_code_schema");
        }
        body["text"] = json!({"format": format});
    }
    Ok(())
}

fn validate_text_format(format: &Value) -> Result<()> {
    let format_type = format.get("type").and_then(|v| v.as_str()).unwrap_or("");

    match format_type {
        "text" | "json_object" => Ok(()),
        "json_schema" => {
            if !format.get("schema").is_some_and(|v| v.is_object()) {
                bail!("json_schema format requires schema object");
            }
            Ok(())
        }
        _ => bail!("unsupported text format type '{format_type}'"),
    }
}

fn apply_prompt_cache_key(body: &mut Value, anthropic: &Value) {
    let key = anthropic
        .pointer("/metadata/session_id")
        .and_then(|v| v.as_str())
        .or_else(|| {
            anthropic
                .pointer("/metadata/conversation_id")
                .and_then(|v| v.as_str())
        })
        .or_else(|| anthropic.get("container").and_then(|v| v.as_str()));

    if let Some(key) = key {
        if !key.is_empty() {
            body["prompt_cache_key"] = json!(key);
        }
    }
}

fn reject_unsupported_server_tool(tool: &Value) -> Result<()> {
    let tool_type = tool.get("type").and_then(|v| v.as_str()).unwrap_or("");
    let name = tool.get("name").and_then(|v| v.as_str()).unwrap_or("");
    let is_server_tool = matches!(
        tool_type,
        "web_search_20250305"
            | "web_fetch_20250910"
            | "code_execution_20250522"
            | "bash_20250124"
            | "text_editor_20250124"
            | "mcp_connector_20250910"
    ) || matches!(
        name,
        "web_search" | "web_fetch" | "code_execution" | "bash" | "text_editor" | "mcp"
    );

    if is_server_tool {
        bail!("unsupported Anthropic server tool '{name}' of type '{tool_type}'");
    }

    Ok(())
}

/// Convert OpenAI Responses API response → Anthropic Messages API response
pub fn responses_to_anthropic(resp: &Value, tool_name_map: &ToolNameMap) -> Result<Value> {
    let mut content = Vec::new();
    let mut has_tool_use = false;

    if let Some(output) = resp.get("output").and_then(|o| o.as_array()) {
        for item in output {
            let item_type = item.get("type").and_then(|t| t.as_str()).unwrap_or("");

            match item_type {
                "message" => {
                    if let Some(parts) = item.get("content").and_then(|c| c.as_array()) {
                        for part in parts {
                            let part_type = part.get("type").and_then(|t| t.as_str()).unwrap_or("");
                            if part_type == "output_text" || part_type == "text" {
                                if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                                    content.push(json!({
                                        "type": "text",
                                        "text": text,
                                    }));
                                }
                            }
                        }
                    }
                }
                "function_call" => {
                    has_tool_use = true;
                    let name = item
                        .get("name")
                        .and_then(|n| n.as_str())
                        .unwrap_or("unknown");
                    let original_name =
                        tool_name_map.get(name).cloned().unwrap_or(name.to_string());
                    let call_id = item
                        .get("call_id")
                        .and_then(|i| i.as_str())
                        .unwrap_or("call_0");
                    let arguments = item
                        .get("arguments")
                        .and_then(|a| a.as_str())
                        .unwrap_or("{}");
                    let input = sanitize_tool_input(
                        &original_name,
                        serde_json::from_str(arguments).unwrap_or_else(|_| json!({})),
                    );

                    content.push(json!({
                        "type": "tool_use",
                        "id": tool_id::to_anthropic_tool_id(call_id),
                        "name": original_name,
                        "input": input,
                    }));
                }
                _ => {}
            }
        }
    }
    if !has_tool_use {
        has_tool_use = recover_textual_tool_call(&mut content);
    }
    if content.is_empty() {
        if let Some(text) = resp.get("output_text").and_then(|t| t.as_str()) {
            if !text.is_empty() {
                content.push(json!({
                    "type": "text",
                    "text": text,
                }));
            }
        }
    }
    if !has_tool_use {
        has_tool_use = recover_textual_tool_call(&mut content);
    }

    // stop_reason
    let status = resp
        .get("status")
        .and_then(|s| s.as_str())
        .unwrap_or("completed");
    let stop_reason = if has_tool_use {
        "tool_use"
    } else {
        match status {
            "completed" => "end_turn",
            "incomplete" => "max_tokens",
            _ => "end_turn",
        }
    };

    // usage
    let usage = resp.get("usage").cloned().unwrap_or(json!({}));
    let mut anthropic_usage = json!({
        "input_tokens": usage.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0),
        "output_tokens": usage.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0),
    });
    if let Some(cached) = usage
        .pointer("/input_tokens_details/cached_tokens")
        .and_then(|v| v.as_u64())
    {
        anthropic_usage["cache_read_input_tokens"] = json!(cached);
    }

    let model = resp
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("unknown");
    let id = resp.get("id").and_then(|i| i.as_str()).unwrap_or("resp_0");

    Ok(json!({
        "id": id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": null,
        "usage": anthropic_usage,
    }))
}

fn recover_textual_tool_call(content: &mut [Value]) -> bool {
    if content.len() != 1 {
        return false;
    }
    let Some(text) = content[0].get("text").and_then(|value| value.as_str()) else {
        return false;
    };
    let Some(parsed) = parse_textual_tool_call(text) else {
        return false;
    };
    let input = sanitize_tool_input(&parsed.name, parsed.input);
    content[0] = json!({
        "type": "tool_use",
        "id": tool_id::to_anthropic_tool_id("call_text_1"),
        "name": parsed.name,
        "input": input,
    });
    true
}

fn strip_context_window_suffix(model: &str) -> &str {
    model
        .strip_suffix("[1m]")
        .or_else(|| model.strip_suffix("[1M]"))
        .unwrap_or(model)
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

fn convert_image_block(block: &Value) -> Option<Value> {
    let source = block.get("source")?;
    match source.get("type").and_then(|t| t.as_str()) {
        Some("file") => {
            let file_id = source.get("file_id")?.as_str()?;
            if file_id.is_empty() {
                return None;
            }
            Some(json!({"type": "input_image", "file_id": file_id}))
        }
        Some("base64") => {
            let data = source.get("data")?.as_str()?;
            if data.is_empty() {
                return None;
            }
            let media_type = source
                .get("media_type")
                .and_then(|m| m.as_str())
                .unwrap_or("image/png");
            Some(json!({
                "type": "input_image",
                "image_url": format!("data:{media_type};base64,{data}"),
            }))
        }
        _ => None,
    }
}

fn convert_document_block(block: &Value) -> Result<Option<Value>> {
    let Some(source) = block.get("source") else {
        return Ok(None);
    };
    match source.get("type").and_then(|t| t.as_str()) {
        Some("file") => {
            let Some(file_id) = source.get("file_id").and_then(|v| v.as_str()) else {
                return Ok(None);
            };
            if file_id.is_empty() {
                return Ok(None);
            }
            Ok(Some(json!({"type": "input_file", "file_id": file_id})))
        }
        Some("base64") => {
            if let Some(media_type) = source.get("media_type").and_then(|v| v.as_str()) {
                if !is_supported_document_media_type(media_type) {
                    bail!("unsupported document media type '{media_type}'");
                }
            }
            let Some(data) = source.get("data").and_then(|v| v.as_str()) else {
                return Ok(None);
            };
            if data.is_empty() {
                return Ok(None);
            }
            let filename = block
                .get("title")
                .and_then(|v| v.as_str())
                .unwrap_or("document.pdf");
            Ok(Some(json!({
                "type": "input_file",
                "filename": filename,
                "file_data": data,
            })))
        }
        Some(other) => bail!("unsupported document source type '{other}'"),
        None => Ok(None),
    }
}

fn is_supported_document_media_type(media_type: &str) -> bool {
    media_type == "application/pdf"
        || media_type == "application/json"
        || media_type.starts_with("text/")
}

fn extract_tool_result_images(block: &Value) -> Vec<Value> {
    block
        .get("content")
        .and_then(|c| c.as_array())
        .map(|parts| {
            parts
                .iter()
                .filter(|p| p.get("type").and_then(|t| t.as_str()) == Some("image"))
                .filter_map(convert_image_block)
                .collect()
        })
        .unwrap_or_default()
}

fn content_has_image(content: &Value) -> bool {
    match content {
        Value::Array(parts) => parts.iter().any(|part| {
            part.get("type").and_then(|t| t.as_str()) == Some("image")
                || part.get("content").is_some_and(content_has_image)
        }),
        _ => false,
    }
}

fn convert_user_content(content: Option<&Value>) -> Result<Vec<Value>> {
    match content {
        Some(Value::String(s)) => Ok(vec![json!({"type": "input_text", "text": s})]),
        Some(Value::Array(parts)) => {
            let mut converted = Vec::new();
            for p in parts {
                let block_type = p.get("type").and_then(|t| t.as_str());
                let item = match block_type {
                    Some("text") => {
                        let text = p.get("text").and_then(|t| t.as_str()).unwrap_or("");
                        Some(json!({"type": "input_text", "text": text}))
                    }
                    Some("image") => convert_image_block(p),
                    Some("document") => convert_document_block(p)?,
                    Some("tool_result") => {
                        // tool_result at user level -> function_call_output.
                        // This should not normally appear here but handle it.
                        None
                    }
                    _ => None,
                };
                if let Some(item) = item {
                    converted.push(item);
                }
            }
            Ok(converted)
        }
        _ => Ok(vec![]),
    }
}

fn extract_tool_result_content(block: &Value) -> String {
    let content = block.get("content");
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn dedupe_old_function_call_outputs(input: &mut [Value]) {
    let recent_start = input.len().saturating_sub(DEDUPE_KEEP_RECENT_INPUT_ITEMS);
    let mut first_seen: HashMap<String, String> = HashMap::new();
    let mut omitted = 0usize;
    let mut omitted_bytes = 0usize;

    for (index, item) in input.iter_mut().enumerate() {
        if item.get("type").and_then(|v| v.as_str()) != Some("function_call_output") {
            continue;
        }
        let Some(output) = item.get("output").and_then(|v| v.as_str()) else {
            continue;
        };
        let output_bytes = output.len();
        if output_bytes < DEDUPE_FUNCTION_OUTPUT_MIN_BYTES {
            continue;
        }

        let hash = format!("{:x}", Sha256::digest(output.as_bytes()));
        if let Some(first_call_id) = first_seen.get(&hash) {
            if index < recent_start {
                let call_id = item
                    .get("call_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("<unknown>");
                item["output"] = json!(format!(
                    "[Duplicate tool output omitted by claudex: same content as {first_call_id}, sha256={hash}, original_bytes={output_bytes}, call_id={call_id}.]"
                ));
                omitted += 1;
                omitted_bytes += output_bytes;
            }
        } else {
            let call_id = item
                .get("call_id")
                .and_then(|v| v.as_str())
                .unwrap_or("<unknown>")
                .to_string();
            first_seen.insert(hash, call_id);
        }
    }

    if omitted > 0 {
        tracing::info!(
            omitted_function_outputs = omitted,
            omitted_function_output_bytes = omitted_bytes,
            "deduplicated historical function_call_output payloads"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn function_output(call_id: &str, output: &str) -> Value {
        json!({
            "type": "function_call_output",
            "call_id": tool_id::to_upstream_tool_id(call_id),
            "output": output,
        })
    }

    #[test]
    fn test_dedupe_old_large_duplicate_function_outputs() {
        let duplicate = "x".repeat(DEDUPE_FUNCTION_OUTPUT_MIN_BYTES);
        let mut input = vec![
            function_output("call_1", &duplicate),
            function_output("call_2", &duplicate),
        ];
        input.extend((0..DEDUPE_KEEP_RECENT_INPUT_ITEMS).map(|i| {
            json!({"role":"user","type":"message","content":[{"type":"input_text","text":format!("recent {i}")}]})
        }));

        dedupe_old_function_call_outputs(&mut input);

        assert_eq!(input[0]["output"], duplicate);
        let deduped = input[1]["output"].as_str().unwrap();
        assert!(deduped.contains("Duplicate tool output omitted by claudex"));
        assert!(deduped.contains("same content as call_1"));
    }

    #[test]
    fn test_dedupe_preserves_small_duplicate_function_outputs() {
        let mut input = vec![
            function_output("call_1", "same"),
            function_output("call_2", "same"),
        ];
        dedupe_old_function_call_outputs(&mut input);

        assert_eq!(input[0]["output"], "same");
        assert_eq!(input[1]["output"], "same");
    }

    #[test]
    fn test_dedupe_preserves_recent_duplicate_function_outputs() {
        let duplicate = "x".repeat(DEDUPE_FUNCTION_OUTPUT_MIN_BYTES);
        let mut input = vec![
            function_output("call_1", &duplicate),
            function_output("call_2", &duplicate),
        ];

        dedupe_old_function_call_outputs(&mut input);

        assert_eq!(input[0]["output"], duplicate);
        assert_eq!(input[1]["output"], duplicate);
    }

    #[test]
    fn test_basic_user_message() {
        let anthropic = json!({
            "model": "gpt-4o",
            "system": "You are helpful.",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "max_tokens": 1024,
            "stream": false,
        });
        let (body, map) = anthropic_to_responses(&anthropic, "gpt-4o").unwrap();
        assert_eq!(body["model"], "gpt-4o");
        assert_eq!(body["instructions"], "You are helpful.");
        assert!(body.get("max_output_tokens").is_none());
        assert_eq!(body["store"], false);
        assert_eq!(body["stream"], false);
        let input = body["input"].as_array().unwrap();
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[0]["content"][0]["type"], "input_text");
        assert_eq!(input[0]["content"][0]["text"], "Hello");
        assert!(map.is_empty());
    }

    #[test]
    fn test_tool_use_roundtrip() {
        let anthropic = json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {"location": "Paris"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "call_1", "content": "Sunny, 25°C"}
                ]},
            ],
            "tools": [
                {"name": "get_weather", "description": "Get weather", "input_schema": {"type": "object", "properties": {"location": {"type": "string"}}}}
            ],
            "max_tokens": 1024,
        });

        let (body, _map) = anthropic_to_responses(&anthropic, "gpt-4o").unwrap();
        let input = body["input"].as_array().unwrap();
        // user message + function_call + function_call_output
        assert_eq!(input.len(), 3);
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[1]["type"], "function_call");
        assert_eq!(input[1]["name"], "get_weather");
        assert_eq!(input[2]["type"], "function_call_output");
        assert_eq!(input[2]["call_id"], "call_1");

        // Tools
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["name"], "get_weather");
    }

    #[test]
    fn test_read_tool_use_strips_invalid_pages() {
        let resp = json!({
            "id": "resp_123",
            "model": "gpt-5.5",
            "status": "completed",
            "output": [{
                "type": "function_call",
                "call_id": "call_read",
                "name": "Read",
                "arguments": "{\"file_path\":\"/tmp/a.md\",\"limit\":10,\"offset\":0,\"pages\":\"\"}"
            }]
        });

        let result = responses_to_anthropic(&resp, &ToolNameMap::new()).unwrap();
        let input = &result["content"][0]["input"];
        assert_eq!(input["file_path"], "/tmp/a.md");
        assert!(input.get("pages").is_none());
    }

    #[test]
    fn test_read_pdf_tool_use_keeps_pages() {
        let resp = json!({
            "id": "resp_123",
            "model": "gpt-5.5",
            "status": "completed",
            "output": [{
                "type": "function_call",
                "call_id": "call_read",
                "name": "Read",
                "arguments": "{\"file_path\":\"/tmp/a.pdf\",\"pages\":\"1-2\"}"
            }]
        });

        let result = responses_to_anthropic(&resp, &ToolNameMap::new()).unwrap();
        assert_eq!(result["content"][0]["input"]["pages"], "1-2");
    }

    #[test]
    fn test_tool_result_image_content_becomes_user_image_message() {
        let anthropic = json!({
            "model": "gpt-5.5",
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "/tmp/image.png"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": [
                        {"type": "text", "text": "Image loaded."},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "/9j/AA=="}}
                    ]}
                ]}
            ],
            "max_tokens": 1024,
        });

        let (body, _) = anthropic_to_responses(&anthropic, "gpt-5.5").unwrap();
        let input = body["input"].as_array().unwrap();
        assert_eq!(input.len(), 3);
        assert_eq!(input[1]["type"], "function_call_output");
        assert_eq!(input[1]["output"], "Image loaded.");
        assert_eq!(input[2]["role"], "user");
        assert_eq!(input[2]["content"][0]["type"], "input_image");
        assert_eq!(
            input[2]["content"][0]["image_url"],
            "data:image/jpeg;base64,/9j/AA=="
        );
    }

    #[test]
    fn test_empty_image_source_is_not_forwarded() {
        let anthropic = json!({
            "model": "gpt-5.5",
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "Describe this."},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": ""}}
                ]}
            ],
            "max_tokens": 1024,
        });

        let (body, _) = anthropic_to_responses(&anthropic, "gpt-5.5").unwrap();
        let content = body["input"][0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 1);
        assert_eq!(content[0]["type"], "input_text");
    }

    #[test]
    fn test_responses_to_anthropic_text() {
        let resp = json!({
            "id": "resp_123",
            "model": "gpt-4o",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": "Hello!", "annotations": []}
                    ]
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        });
        let result = responses_to_anthropic(&resp, &HashMap::new()).unwrap();
        assert_eq!(result["stop_reason"], "end_turn");
        assert_eq!(result["content"][0]["type"], "text");
        assert_eq!(result["content"][0]["text"], "Hello!");
        assert_eq!(result["usage"]["input_tokens"], 10);
    }

    #[test]
    fn test_responses_to_anthropic_text_part_shape() {
        let resp = json!({
            "id": "resp_123",
            "model": "gpt-5.5",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "text", "text": "Compact summary"}
                    ]
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        });
        let result = responses_to_anthropic(&resp, &HashMap::new()).unwrap();
        assert_eq!(result["content"][0]["type"], "text");
        assert_eq!(result["content"][0]["text"], "Compact summary");
    }

    #[test]
    fn test_responses_to_anthropic_output_text_fallback() {
        let resp = json!({
            "id": "resp_123",
            "model": "gpt-5.5",
            "status": "completed",
            "output_text": "Compact summary",
            "output": [],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        });
        let result = responses_to_anthropic(&resp, &HashMap::new()).unwrap();
        assert_eq!(result["content"][0]["type"], "text");
        assert_eq!(result["content"][0]["text"], "Compact summary");
    }

    #[test]
    fn test_responses_to_anthropic_tool_call() {
        let resp = json!({
            "id": "resp_456",
            "model": "gpt-4o",
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_abc",
                    "name": "get_weather",
                    "arguments": "{\"location\":\"Paris\"}",
                    "status": "completed",
                }
            ],
            "usage": {"input_tokens": 20, "output_tokens": 10},
        });
        let result = responses_to_anthropic(&resp, &HashMap::new()).unwrap();
        assert_eq!(result["stop_reason"], "tool_use");
        assert_eq!(result["content"][0]["type"], "tool_use");
        assert_eq!(result["content"][0]["id"], "toolu_call_abc");
        assert_eq!(result["content"][0]["name"], "get_weather");
        assert_eq!(result["content"][0]["input"]["location"], "Paris");
    }

    #[test]
    fn test_responses_to_anthropic_textual_tool_call() {
        let resp = json!({
            "id": "resp_text_tool",
            "model": "gpt-5.5",
            "status": "completed",
            "output": [{
                "type": "message",
                "content": [{"type": "output_text", "text": "Read({\"file_path\":\"/tmp/a.md\",\"pages\":\"\"})"}]
            }],
            "usage": {"input_tokens": 10, "output_tokens": 5}
        });

        let result = responses_to_anthropic(&resp, &HashMap::new()).unwrap();
        assert_eq!(result["stop_reason"], "tool_use");
        assert_eq!(result["content"][0]["type"], "tool_use");
        assert_eq!(result["content"][0]["id"], "toolu_call_text_1");
        assert_eq!(result["content"][0]["name"], "Read");
        assert_eq!(result["content"][0]["input"]["file_path"], "/tmp/a.md");
        assert!(result["content"][0]["input"].get("pages").is_none());
    }

    #[test]
    fn test_responses_to_anthropic_prose_tool_like_text_remains_text() {
        let resp = json!({
            "id": "resp_text",
            "model": "gpt-5.5",
            "status": "completed",
            "output_text": "I will call Read({\"file_path\":\"/tmp/a.md\"}) if needed.",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        });

        let result = responses_to_anthropic(&resp, &HashMap::new()).unwrap();
        assert_eq!(result["stop_reason"], "end_turn");
        assert_eq!(result["content"][0]["type"], "text");
    }

    #[test]
    fn test_tool_choice_mapping() {
        let test_cases = vec![
            (json!({"type": "auto"}), json!("auto")),
            (json!({"type": "any"}), json!("required")),
            (json!({"type": "none"}), json!("none")),
            (
                json!({"type": "tool", "name": "fn1"}),
                json!({"type": "function", "name": "fn1"}),
            ),
        ];
        for (anthropic_tc, expected) in test_cases {
            let anthropic = json!({
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "test"}],
                "tool_choice": anthropic_tc,
                "max_tokens": 100,
            });
            let (body, _) = anthropic_to_responses(&anthropic, "gpt-4o").unwrap();
            assert_eq!(body["tool_choice"], expected);
        }
    }

    #[test]
    fn test_system_prompt_array() {
        let anthropic = json!({
            "model": "gpt-4o",
            "system": [
                {"type": "text", "text": "Part 1."},
                {"type": "text", "text": "Part 2."},
            ],
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 100,
        });
        let (body, _) = anthropic_to_responses(&anthropic, "gpt-4o").unwrap();
        assert_eq!(body["instructions"], "Part 1.\nPart 2.");
    }

    #[test]
    fn test_image_file_block_maps_to_input_image_file() {
        let anthropic = json!({
            "model": "gpt-5.5",
            "messages": [{"role":"user","content":[
                {"type":"image","source":{"type":"file","file_id":"file_img"}}
            ]}]
        });
        let (body, _) = anthropic_to_responses(&anthropic, "gpt-5.5").unwrap();
        let content = body["input"][0]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "input_image");
        assert_eq!(content[0]["file_id"], "file_img");
    }

    #[test]
    fn test_document_file_block_maps_to_input_file() {
        let anthropic = json!({
            "model": "gpt-5.5",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Read this PDF."},
                    {"type": "document", "source": {"type": "file", "file_id": "file_abc"}}
                ]
            }]
        });

        let (body, _) = anthropic_to_responses(&anthropic, "gpt-5.5").unwrap();
        let content = body["input"][0]["content"].as_array().unwrap();
        assert_eq!(content[1]["type"], "input_file");
        assert_eq!(content[1]["file_id"], "file_abc");
    }

    #[test]
    fn test_unsafe_document_format_errors() {
        let anthropic = json!({
            "model": "gpt-5.5",
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/octet-stream",
                            "data": "AAE="
                        }
                    }
                ]
            }]
        });

        let err = anthropic_to_responses(&anthropic, "gpt-5.5").unwrap_err();
        assert!(err.to_string().contains("unsupported document media type"));
    }

    #[test]
    fn test_structured_output_and_prompt_cache_mapping() {
        let anthropic = json!({
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "Return JSON"}],
            "thinking": {"type": "enabled", "budget_tokens": 4096},
            "output_config": {
                "effort": "high",
                "format": {
                    "type": "json_schema",
                    "name": "result",
                    "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
                }
            },
            "metadata": {"session_id": "claude-session-1"}
        });

        let (body, _) = anthropic_to_responses(&anthropic, "gpt-5.5").unwrap();
        // thinking/output_config.effort are NOT bridged to OpenAI reasoning.
        assert!(body.get("reasoning").is_none());
        assert_eq!(body["text"]["format"]["type"], "json_schema");
        assert_eq!(body["text"]["format"]["name"], "result");
        assert_eq!(body["prompt_cache_key"], "claude-session-1");
    }

    #[test]
    fn test_reasoning_bridge_effort_only_is_opt_in() {
        let anthropic = json!({
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "Think briefly"}],
            "thinking": {"type": "enabled", "budget_tokens": 4096},
            "output_config": {"effort": "high"}
        });

        let (body, _) = anthropic_to_responses_with_options(
            &anthropic,
            "gpt-5.5",
            ResponsesTranslationOptions {
                reasoning_bridge: ReasoningBridge::EffortOnly,
            },
        )
        .unwrap();

        assert_eq!(body["reasoning"]["effort"], "high");
        assert!(body["reasoning"].get("summary").is_none());
    }

    #[test]
    fn test_reasoning_bridge_ignores_unknown_effort() {
        let anthropic = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "output_config": {"effort": "maximum"}
        });

        let (body, _) = anthropic_to_responses_with_options(
            &anthropic,
            "gpt-5.5",
            ResponsesTranslationOptions {
                reasoning_bridge: ReasoningBridge::EffortOnly,
            },
        )
        .unwrap();

        assert!(body.get("reasoning").is_none());
    }

    #[test]
    fn test_plain_request_does_not_request_reasoning_by_default() {
        let anthropic = json!({
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "Summarize this session for compact"}],
        });

        let (body, _) = anthropic_to_responses(&anthropic, "gpt-5.5").unwrap();
        assert!(body.get("reasoning").is_none());
    }

    #[test]
    fn test_json_schema_format_without_name_gets_default_name() {
        let anthropic = json!({
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "Return JSON"}],
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": {
                        "type": "object",
                        "properties": {"title": {"type": "string"}},
                        "required": ["title"],
                        "additionalProperties": false
                    }
                }
            }
        });

        let (body, _) = anthropic_to_responses(&anthropic, "gpt-5.5").unwrap();
        assert_eq!(body["text"]["format"]["type"], "json_schema");
        assert_eq!(body["text"]["format"]["name"], "claude_code_schema");
    }

    #[test]
    fn test_invalid_structured_output_schema_errors() {
        let anthropic = json!({
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "Return JSON"}],
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "name": "result"
                }
            }
        });

        let err = anthropic_to_responses(&anthropic, "gpt-5.5").unwrap_err();
        assert!(err
            .to_string()
            .contains("json_schema format requires schema"));
    }

    #[test]
    fn test_server_tool_without_opt_in_errors() {
        let anthropic = json!({
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "Search web"}],
            "tools": [{"type": "web_search_20250305", "name": "web_search"}]
        });

        let err = anthropic_to_responses(&anthropic, "gpt-5.5").unwrap_err();
        assert!(err
            .to_string()
            .contains("unsupported Anthropic server tool"));
    }

    #[test]
    fn test_usage_cache_details_are_mapped() {
        let resp = json!({
            "id": "resp_123",
            "model": "gpt-5.5",
            "status": "completed",
            "output_text": "ok",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "input_tokens_details": {"cached_tokens": 40},
                "output_tokens_details": {"reasoning_tokens": 7}
            }
        });

        let result = responses_to_anthropic(&resp, &HashMap::new()).unwrap();
        assert_eq!(result["usage"]["input_tokens"], 100);
        assert_eq!(result["usage"]["output_tokens"], 20);
        assert_eq!(result["usage"]["cache_read_input_tokens"], 40);
        assert!(result["usage"].get("reasoning_tokens").is_none());
    }

    #[test]
    fn test_anthropic_to_responses_strips_1m_model_suffix() {
        let req = json!({
            "model": "gpt-5.5[1M]",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
        });

        let (body, _) = anthropic_to_responses(&req, "gpt-5.5").unwrap();
        assert_eq!(body["model"], "gpt-5.5");
    }
}
