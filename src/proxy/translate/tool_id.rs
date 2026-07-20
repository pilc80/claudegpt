pub fn to_anthropic_tool_id(upstream: &str) -> String {
    if let Some(rest) = upstream.strip_prefix("call_") {
        format!("toolu_call_{rest}")
    } else if let Some(rest) = upstream.strip_prefix("toolu_") {
        format!("toolu_openai_toolu_{rest}")
    } else {
        format!("toolu_openai_{upstream}")
    }
}

pub fn to_upstream_tool_id(anthropic: &str) -> String {
    if let Some(rest) = anthropic.strip_prefix("toolu_openai_toolu_") {
        format!("toolu_{rest}")
    } else if let Some(rest) = anthropic.strip_prefix("toolu_openai_") {
        rest.to_string()
    } else if let Some(rest) = anthropic.strip_prefix("toolu_call_") {
        format!("call_{rest}")
    } else {
        anthropic.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_call_ids_statelessly() {
        assert_eq!(to_anthropic_tool_id("call_abc"), "toolu_call_abc");
        assert_eq!(to_upstream_tool_id("toolu_call_abc"), "call_abc");
    }

    #[test]
    fn preserves_existing_anthropic_ids_outward() {
        assert_eq!(to_anthropic_tool_id("toolu_abc"), "toolu_openai_toolu_abc");
        assert_eq!(to_upstream_tool_id("toolu_openai_toolu_abc"), "toolu_abc");
        assert_eq!(to_upstream_tool_id("toolu_abc"), "toolu_abc");
    }

    #[test]
    fn upstream_tool_ids_roundtrip_without_collisions() {
        for upstream in ["call_abc", "toolu_abc", "abc"] {
            assert_eq!(
                to_upstream_tool_id(&to_anthropic_tool_id(upstream)),
                upstream
            );
        }
        assert_ne!(
            to_anthropic_tool_id("toolu_abc"),
            to_anthropic_tool_id("call_abc")
        );
    }
}
