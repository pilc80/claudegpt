use serde_json::Value;

const UNSUPPORTED_SCHEMA_KEYS: &[&str] = &[
    "$schema",
    "$id",
    "$defs",
    "definitions",
    "patternProperties",
    "unevaluatedProperties",
    "dependentSchemas",
    "if",
    "then",
    "else",
    "allOf",
    "anyOf",
    "oneOf",
    "not",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixerWarning {
    pub code: &'static str,
    pub message: String,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct FixerWarnings(Vec<FixerWarning>);

impl FixerWarnings {
    pub fn push(&mut self, code: &'static str, message: impl Into<String>) {
        self.0.push(FixerWarning {
            code,
            message: message.into(),
        });
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &FixerWarning> {
        self.0.iter()
    }
}

pub fn apply_openai_compatibility_fixers(request: &mut Value) -> FixerWarnings {
    let mut warnings = FixerWarnings::default();
    let removed = strip_cache_control_fields(request);
    if removed > 0 {
        warnings.push(
            "strip_cache_control",
            format!("removed {removed} Anthropic cache_control field(s)"),
        );
    }

    let lowered = lower_tool_input_schemas(request);
    if lowered > 0 {
        warnings.push(
            "lower_tool_schema",
            format!("removed {lowered} unsupported JSON Schema field(s) from tool schemas"),
        );
    }
    warnings
}

fn strip_cache_control_fields(request: &mut Value) -> usize {
    let mut removed = 0;
    if let Some(system) = request.get_mut("system") {
        removed += strip_cache_control_from_content(system);
    }
    if let Some(messages) = request.get_mut("messages") {
        removed += strip_cache_control_from_messages(messages);
    }
    if let Some(tools) = request
        .get_mut("tools")
        .and_then(|value| value.as_array_mut())
    {
        for tool in tools {
            if let Some(obj) = tool.as_object_mut() {
                removed += usize::from(obj.remove("cache_control").is_some());
            }
        }
    }
    removed
}

fn strip_cache_control_from_messages(value: &mut Value) -> usize {
    let Some(messages) = value.as_array_mut() else {
        return 0;
    };
    messages
        .iter_mut()
        .filter_map(|message| message.get_mut("content"))
        .map(strip_cache_control_from_content)
        .sum()
}

fn strip_cache_control_from_content(value: &mut Value) -> usize {
    match value {
        Value::Object(obj) => {
            let mut removed = usize::from(obj.remove("cache_control").is_some());
            if let Some(content) = obj.get_mut("content") {
                removed += strip_cache_control_from_content(content);
            }
            removed
        }
        Value::Array(items) => items.iter_mut().map(strip_cache_control_from_content).sum(),
        _ => 0,
    }
}

fn lower_tool_input_schemas(request: &mut Value) -> usize {
    let Some(tools) = request
        .get_mut("tools")
        .and_then(|value| value.as_array_mut())
    else {
        return 0;
    };

    tools
        .iter_mut()
        .filter_map(|tool| tool.get_mut("input_schema"))
        .map(lower_json_schema)
        .sum()
}

fn lower_json_schema(schema: &mut Value) -> usize {
    match schema {
        Value::Object(obj) => {
            let mut removed = 0usize;
            for key in UNSUPPORTED_SCHEMA_KEYS {
                removed += usize::from(obj.remove(*key).is_some());
            }
            for value in obj.values_mut() {
                removed += lower_json_schema(value);
            }
            removed
        }
        Value::Array(items) => items.iter_mut().map(lower_json_schema).sum(),
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn strips_anthropic_cache_control_without_touching_schema_fields() {
        let mut request = json!({
            "system": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral"}}],
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": "hi",
                    "cache_control": {"type": "ephemeral"},
                    "nested": {"cache_control": {"type": "schema-field"}}
                }]
            }],
            "tools": [{
                "name": "t",
                "input_schema": {
                    "type": "object",
                    "cache_control": {"type": "schema-field"}
                },
                "cache_control": {"type": "ephemeral"}
            }],
            "metadata": {"session_id": "s1"}
        });

        let warnings = apply_openai_compatibility_fixers(&mut request);

        assert!(warnings
            .iter()
            .any(|warning| warning.code == "strip_cache_control"));
        assert!(request["system"][0].get("cache_control").is_none());
        assert!(request["messages"][0]["content"][0]
            .get("cache_control")
            .is_none());
        assert!(request["tools"][0].get("cache_control").is_none());
        assert_eq!(
            request["tools"][0]["input_schema"]["cache_control"]["type"],
            "schema-field"
        );
        assert_eq!(
            request["messages"][0]["content"][0]["nested"]["cache_control"]["type"],
            "schema-field"
        );
        assert_eq!(request["metadata"]["session_id"], "s1");
    }

    #[test]
    fn lowers_tool_input_schema_subset() {
        let mut request = json!({
            "tools": [{
                "name": "lookup",
                "input_schema": {
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "type": "object",
                    "allOf": [{"required": ["q"]}],
                    "properties": {
                        "q": {
                            "type": "string",
                            "patternProperties": {".*": {"type": "string"}}
                        }
                    },
                    "required": ["q"],
                    "additionalProperties": false
                }
            }]
        });

        let warnings = apply_openai_compatibility_fixers(&mut request);
        let text = serde_json::to_string(&request).unwrap();

        assert!(warnings
            .iter()
            .any(|warning| warning.code == "lower_tool_schema"));
        assert!(!text.contains("$schema"));
        assert!(!text.contains("allOf"));
        assert!(!text.contains("patternProperties"));
        assert_eq!(request["tools"][0]["input_schema"]["type"], "object");
        assert_eq!(
            request["tools"][0]["input_schema"]["properties"]["q"]["type"],
            "string"
        );
        assert_eq!(
            request["tools"][0]["input_schema"]["additionalProperties"],
            false
        );
    }

    #[test]
    fn no_warning_when_nothing_changes() {
        let mut request = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "lookup", "input_schema": {"type": "object"}}]
        });
        let warnings = apply_openai_compatibility_fixers(&mut request);
        assert!(warnings.is_empty());
    }
}
