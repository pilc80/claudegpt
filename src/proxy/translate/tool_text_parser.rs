use serde_json::Value;

#[derive(Debug, Clone, PartialEq)]
pub struct ParsedToolCall {
    pub name: String,
    pub input: Value,
}

pub fn parse_textual_tool_call(text: &str) -> Option<ParsedToolCall> {
    let trimmed = text.trim();
    parse_xml_tool_use(trimmed)
        .or_else(|| parse_function_tag(trimmed))
        .or_else(|| parse_tool_call_json_tag(trimmed))
        .or_else(|| parse_json_call(trimmed))
        .or_else(|| parse_function_call(trimmed))
}

fn parse_xml_tool_use(text: &str) -> Option<ParsedToolCall> {
    let prefix = "<tool_use name=\"";
    let name_start = text.strip_prefix(prefix)?;
    let (name, rest) = name_start.split_once("\">")?;
    let json_text = rest.strip_suffix("</tool_use>")?.trim();
    parse_call(name, json_text)
}

fn parse_function_tag(text: &str) -> Option<ParsedToolCall> {
    let prefix = "<function=";
    let name_start = text.strip_prefix(prefix)?;
    let (name, rest) = name_start.split_once('>')?;
    let json_text = rest.strip_suffix("</function>")?.trim();
    parse_call(name.trim(), json_text)
}

fn parse_tool_call_json_tag(text: &str) -> Option<ParsedToolCall> {
    let json_text = text
        .strip_prefix("<tool_call>")?
        .strip_suffix("</tool_call>")?
        .trim();
    parse_json_call(json_text)
}

fn parse_json_call(text: &str) -> Option<ParsedToolCall> {
    let value: Value = serde_json::from_str(text).ok()?;
    let name = value
        .get("name")
        .or_else(|| value.get("tool"))
        .and_then(|v| v.as_str())?;
    let input = value
        .get("arguments")
        .or_else(|| value.get("input"))
        .cloned()?;
    if !input.is_object() {
        return None;
    }
    Some(ParsedToolCall {
        name: name.to_string(),
        input,
    })
}

fn parse_function_call(text: &str) -> Option<ParsedToolCall> {
    let (name, rest) = text.split_once('(')?;
    let json_text = rest.strip_suffix(')')?.trim();
    if name.is_empty()
        || !name
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-')
    {
        return None;
    }
    parse_call(name, json_text)
}

fn parse_call(name: &str, json_text: &str) -> Option<ParsedToolCall> {
    if name.is_empty() || json_text.is_empty() {
        return None;
    }
    let input = serde_json::from_str::<Value>(json_text).ok()?;
    if !input.is_object() {
        return None;
    }
    Some(ParsedToolCall {
        name: name.to_string(),
        input,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_xml_tool_use_envelope() {
        let parsed = parse_textual_tool_call(
            "<tool_use name=\"Read\">{\"file_path\":\"/tmp/a.md\"}</tool_use>",
        )
        .unwrap();
        assert_eq!(parsed.name, "Read");
        assert_eq!(parsed.input["file_path"], "/tmp/a.md");
    }

    #[test]
    fn parses_function_call_envelope() {
        let parsed = parse_textual_tool_call("Read({\"file_path\":\"/tmp/a.md\"})").unwrap();
        assert_eq!(parsed.name, "Read");
        assert_eq!(parsed.input, json!({"file_path": "/tmp/a.md"}));
    }

    #[test]
    fn parses_additional_provider_formats() {
        assert_eq!(
            parse_textual_tool_call("<function=Read>{\"file_path\":\"/tmp/a.md\"}</function>")
                .unwrap()
                .name,
            "Read"
        );
        assert_eq!(
            parse_textual_tool_call("<tool_call>{\"name\":\"Read\",\"arguments\":{\"file_path\":\"/tmp/a.md\"}}</tool_call>")
                .unwrap()
                .name,
            "Read"
        );
        assert_eq!(
            parse_textual_tool_call("{\"tool\":\"Read\",\"input\":{\"file_path\":\"/tmp/a.md\"}}")
                .unwrap()
                .name,
            "Read"
        );
    }

    #[test]
    fn ignores_prose_and_malformed_json() {
        assert!(parse_textual_tool_call("I will call Read({}) now").is_none());
        assert!(parse_textual_tool_call("Read({not json})").is_none());
        assert!(parse_textual_tool_call("Read([])").is_none());
    }
}
