use anyhow::Result;
use reqwest::RequestBuilder;
use serde_json::Value;

use super::{ByteStream, ProviderAdapter, TranslatedRequest};
use crate::config::ProfileConfig;
use crate::proxy::util::ToolNameMap;

pub struct ResponsesAdapter;

const CHATGPT_CODEX_DEFAULT_INSTRUCTIONS: &str =
    "You are Claude Code, a software engineering agent running in a terminal.";

impl ProviderAdapter for ResponsesAdapter {
    fn endpoint_path(&self) -> &str {
        "/responses"
    }

    fn translate_request(
        &self,
        body: &Value,
        profile: &ProfileConfig,
    ) -> Result<TranslatedRequest> {
        let mut body = body.clone();
        let warnings =
            crate::proxy::translate::fixers::apply_openai_compatibility_fixers(&mut body);
        for warning in warnings.iter() {
            tracing::warn!(
                profile = %profile.name,
                code = warning.code,
                message = %warning.message,
                "applied OpenAI-compatible request fixer"
            );
        }
        let options = crate::proxy::translate::responses::ResponsesTranslationOptions {
            reasoning_bridge: profile.reasoning_bridge,
        };
        let (mut responses_body, tool_name_map) =
            crate::proxy::translate::responses::anthropic_to_responses_with_options(
                &body,
                &profile.default_model,
                options,
            )?;
        if let Some(image_model) = &profile.image_model {
            if crate::proxy::translate::responses::request_has_current_image(&body) {
                responses_body["model"] = serde_json::json!(image_model);
            }
        }
        if profile.base_url.contains("chatgpt.com/backend-api/codex") {
            responses_body["stream"] = serde_json::json!(true);
            if responses_body.get("instructions").is_none() {
                responses_body["instructions"] =
                    serde_json::json!(CHATGPT_CODEX_DEFAULT_INSTRUCTIONS);
            }
        }
        Ok(TranslatedRequest {
            body: responses_body,
            tool_name_map,
        })
    }

    fn apply_auth(&self, builder: RequestBuilder, profile: &ProfileConfig) -> RequestBuilder {
        if !profile.api_key.is_empty() {
            builder.header("Authorization", format!("Bearer {}", profile.api_key))
        } else {
            builder
        }
    }

    fn apply_extra_headers(
        &self,
        builder: RequestBuilder,
        profile: &ProfileConfig,
    ) -> RequestBuilder {
        if let Some(account_id) = profile.extra_env.get("CHATGPT_ACCOUNT_ID") {
            builder.header("ChatGPT-Account-ID", account_id.as_str())
        } else {
            builder
        }
    }

    fn translate_response(&self, body: &Value, tool_name_map: &ToolNameMap) -> Result<Value> {
        crate::proxy::translate::responses::responses_to_anthropic(body, tool_name_map)
    }

    fn translate_stream(
        &self,
        stream: ByteStream,
        tool_name_map: ToolNameMap,
        profile: &ProfileConfig,
    ) -> ByteStream {
        crate::proxy::translate::responses_stream::translate_responses_stream_with_reasoning(
            stream,
            tool_name_map,
            profile.reasoning_bridge,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ProfileConfig;
    use serde_json::json;

    #[test]
    fn test_current_image_uses_image_model() {
        let profile = ProfileConfig {
            default_model: "gpt-5.5".to_string(),
            image_model: Some("gpt-5.5-mini".to_string()),
            base_url: "https://api.openai.com/v1".to_string(),
            ..ProfileConfig::default()
        };

        let body = json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe"},
                    {"type": "image", "source": {"type": "base64", "data": "abc", "media_type": "image/png"}}
                ]
            }]
        });

        let translated = ResponsesAdapter.translate_request(&body, &profile).unwrap();
        assert_eq!(translated.body["model"], "gpt-5.5-mini");
    }

    #[test]
    fn test_old_image_history_does_not_force_image_model() {
        let profile = ProfileConfig {
            default_model: "gpt-5.5".to_string(),
            image_model: Some("gpt-5.5-mini".to_string()),
            base_url: "https://api.openai.com/v1".to_string(),
            ..ProfileConfig::default()
        };

        let body = json!({
            "messages": [
                {"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "data": "abc", "media_type": "image/png"}}
                ]},
                {"role": "user", "content": "Now answer text-only."}
            ]
        });

        let translated = ResponsesAdapter.translate_request(&body, &profile).unwrap();
        assert_eq!(translated.body["model"], "gpt-5.5");
    }

    #[test]
    fn test_chatgpt_codex_backend_gets_default_instructions() {
        let profile = ProfileConfig {
            default_model: "gpt-5.5".to_string(),
            base_url: "https://chatgpt.com/backend-api/codex".to_string(),
            ..ProfileConfig::default()
        };

        let body = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
        });

        let translated = ResponsesAdapter.translate_request(&body, &profile).unwrap();
        assert_eq!(
            translated.body["instructions"],
            CHATGPT_CODEX_DEFAULT_INSTRUCTIONS
        );
    }

    #[test]
    fn test_reasoning_bridge_is_profile_gated() {
        let body = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "output_config": {"effort": "high"}
        });
        let off_profile = ProfileConfig {
            default_model: "gpt-5.5".to_string(),
            ..ProfileConfig::default()
        };
        let on_profile = ProfileConfig {
            default_model: "gpt-5.5".to_string(),
            reasoning_bridge: crate::config::ReasoningBridge::EffortOnly,
            ..ProfileConfig::default()
        };

        let off = ResponsesAdapter
            .translate_request(&body, &off_profile)
            .unwrap();
        let on = ResponsesAdapter
            .translate_request(&body, &on_profile)
            .unwrap();

        assert!(off.body.get("reasoning").is_none());
        assert_eq!(on.body["reasoning"]["effort"], "high");
    }

    #[test]
    fn test_non_chatgpt_responses_backend_keeps_missing_instructions() {
        let profile = ProfileConfig {
            default_model: "gpt-5.5".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            ..ProfileConfig::default()
        };

        let body = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
        });

        let translated = ResponsesAdapter.translate_request(&body, &profile).unwrap();
        assert!(translated.body.get("instructions").is_none());
    }
}
