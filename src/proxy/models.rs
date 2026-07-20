use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use reqwest::Client;
use serde_json::{json, Value};
use tokio::sync::RwLock;

use crate::config::{ProfileConfig, ProviderType};
use crate::proxy::ProxyState;

const MODEL_CACHE_TTL: Duration = Duration::from_secs(300);

#[derive(Debug, Clone, Default)]
pub struct ModelCache {
    profiles: HashMap<String, CachedProfileModels>,
}

#[derive(Debug, Clone)]
struct CachedProfileModels {
    provider: ProviderType,
    base_url: String,
    models: Vec<String>,
    case_index: HashMap<String, String>,
    fetched_at: SystemTime,
    expires_at: SystemTime,
    last_error: Option<String>,
}

pub type SharedModelCache = Arc<RwLock<ModelCache>>;

pub fn new_model_cache() -> SharedModelCache {
    Arc::new(RwLock::new(ModelCache::default()))
}

pub async fn list_models(State(state): State<Arc<ProxyState>>) -> impl IntoResponse {
    let profiles = {
        let config = state.config.read().await;
        config
            .enabled_profiles()
            .into_iter()
            .cloned()
            .collect::<Vec<_>>()
    };

    let mut models = Vec::new();

    for profile in &profiles {
        let discovered = fresh_cached_models(&state.model_cache, profile).await;
        models.extend(profile_model_entries_with_discovered(
            profile,
            discovered.as_deref(),
        ));
    }

    (
        StatusCode::OK,
        Json(json!({
            "object": "list",
            "data": models,
        })),
    )
}

pub async fn model_cache_debug(State(state): State<Arc<ProxyState>>) -> impl IntoResponse {
    let cache = state.model_cache.read().await;
    (StatusCode::OK, Json(cache.diagnostic_json()))
}

pub async fn repair_model_case(
    _client: &Client,
    cache: &SharedModelCache,
    profile: &ProfileConfig,
    body: &mut Value,
) {
    let Some(model) = body
        .get("model")
        .and_then(|value| value.as_str())
        .map(ToOwned::to_owned)
    else {
        return;
    };
    if repair_model_case_value(cache, profile, &model)
        .await
        .is_none()
    {
        return;
    }
    let Some(repaired) = repair_model_case_value(cache, profile, &model).await else {
        return;
    };
    if repaired != model {
        tracing::info!(
            profile = %profile.name,
            requested_model = %model,
            repaired_model = %repaired,
            "repaired upstream model casing"
        );
        body["model"] = json!(repaired);
    }
}

async fn discover_or_cached_models(
    client: &Client,
    cache: &SharedModelCache,
    profile: &ProfileConfig,
) -> Option<Vec<String>> {
    if let Some(models) = fresh_cached_models(cache, profile).await {
        return Some(models);
    }

    match fetch_upstream_models(client, profile).await {
        Ok(models) if !models.is_empty() => {
            cache.write().await.insert_models(profile, models.clone());
            Some(models)
        }
        Ok(_) => {
            cached_models_with_error(cache, profile, "upstream /models returned no model ids").await
        }
        Err(err) => cached_models_with_error(cache, profile, &err.to_string()).await,
    }
}

async fn fetch_upstream_models(
    client: &Client,
    profile: &ProfileConfig,
) -> anyhow::Result<Vec<String>> {
    let url = upstream_models_url(profile);
    let mut request = client.get(url);

    request = match profile.provider_type {
        ProviderType::DirectAnthropic => {
            let request = request.header("anthropic-version", "2023-06-01");
            if profile.api_key.is_empty() {
                request
            } else {
                request.header("x-api-key", &profile.api_key)
            }
        }
        ProviderType::OpenAICompatible | ProviderType::OpenAIResponses => {
            let request = if profile.api_key.is_empty() {
                request
            } else {
                request.header("Authorization", format!("Bearer {}", profile.api_key))
            };
            if let Some(account_id) = profile.extra_env.get("CHATGPT_ACCOUNT_ID") {
                request.header("ChatGPT-Account-ID", account_id.as_str())
            } else {
                request
            }
        }
    };

    for (key, value) in &profile.custom_headers {
        request = request.header(key.as_str(), value.as_str());
    }

    let response = request.send().await?;
    if !response.status().is_success() {
        anyhow::bail!("upstream /models returned HTTP {}", response.status());
    }
    let value = response.json::<Value>().await?;
    Ok(parse_model_ids(&value))
}

fn upstream_models_url(profile: &ProfileConfig) -> String {
    let mut url = match profile.provider_type {
        ProviderType::DirectAnthropic => {
            format!("{}/v1/models", profile.base_url.trim_end_matches('/'))
        }
        ProviderType::OpenAICompatible | ProviderType::OpenAIResponses => {
            format!("{}/models", profile.base_url.trim_end_matches('/'))
        }
    };
    append_query_params(&mut url, &profile.query_params);
    url
}

fn append_query_params(url: &mut String, query_params: &HashMap<String, String>) {
    if query_params.is_empty() {
        return;
    }
    let qs = query_params
        .iter()
        .map(|(key, value)| {
            format!(
                "{}={}",
                urlencoding::encode(key),
                urlencoding::encode(value)
            )
        })
        .collect::<Vec<_>>()
        .join("&");
    if url.contains('?') {
        url.push('&');
    } else {
        url.push('?');
    }
    url.push_str(&qs);
}

async fn fresh_cached_models(
    cache: &SharedModelCache,
    profile: &ProfileConfig,
) -> Option<Vec<String>> {
    let cache = cache.read().await;
    cache
        .profiles
        .get(&cache_key(profile))
        .filter(|cached| cached.expires_at > SystemTime::now())
        .map(|cached| cached.models.clone())
}

async fn cached_models_with_error(
    cache: &SharedModelCache,
    profile: &ProfileConfig,
    error: &str,
) -> Option<Vec<String>> {
    let mut cache = cache.write().await;
    let cached = cache.profiles.get_mut(&cache_key(profile))?;
    cached.last_error = Some(error.to_string());
    Some(cached.models.clone())
}

async fn repair_model_case_value(
    cache: &SharedModelCache,
    profile: &ProfileConfig,
    model: &str,
) -> Option<String> {
    let cache = cache.read().await;
    cache
        .profiles
        .get(&cache_key(profile))
        .and_then(|cached| cached.case_index.get(&model.to_ascii_lowercase()).cloned())
}

fn profile_model_entries(profile: &ProfileConfig) -> Vec<Value> {
    profile_model_entries_with_discovered(profile, None)
}

fn profile_model_entries_with_discovered(
    profile: &ProfileConfig,
    discovered: Option<&[String]>,
) -> Vec<Value> {
    let mut seen = HashSet::new();
    let mut ids = Vec::new();
    let candidates = [
        Some(profile.default_model.as_str()),
        profile.models.haiku.as_deref(),
        profile.models.sonnet.as_deref(),
        profile.models.opus.as_deref(),
    ];

    for id in candidates.into_iter().flatten().filter(|id| !id.is_empty()) {
        if seen.insert(id.to_ascii_lowercase()) {
            ids.push(id.to_string());
        }
    }

    if let Some(discovered) = discovered {
        for id in discovered.iter().filter(|id| !id.is_empty()) {
            let key = id.to_ascii_lowercase();
            if let Some(existing) = ids
                .iter_mut()
                .find(|existing| existing.to_ascii_lowercase() == key)
            {
                *existing = id.clone();
            } else if seen.insert(key) {
                ids.push(id.clone());
            }
        }
    }

    ids.into_iter()
        .map(|id| {
            json!({
                "id": id,
                "object": "model",
                "created": 0,
                "owned_by": profile.name,
                "x-claudex-profile": profile.name,
                "x-claudex-provider": match profile.provider_type {
                    ProviderType::DirectAnthropic => "anthropic",
                    ProviderType::OpenAICompatible => "openai-compatible",
                    ProviderType::OpenAIResponses => "openai-responses",
                },
            })
        })
        .collect()
}

pub fn parse_model_ids(value: &Value) -> Vec<String> {
    value
        .get("data")
        .and_then(|data| data.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.get("id").and_then(|id| id.as_str()))
                .filter(|id| !id.is_empty())
                .map(ToString::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn cache_key(profile: &ProfileConfig) -> String {
    format!(
        "{}|{:?}|{}",
        profile.name,
        profile.provider_type,
        profile.base_url.trim_end_matches('/')
    )
}

fn case_index(models: &[String]) -> HashMap<String, String> {
    let mut index = HashMap::new();
    for model in models {
        index
            .entry(model.to_ascii_lowercase())
            .or_insert_with(|| model.clone());
    }
    index
}

impl ModelCache {
    fn insert_models(&mut self, profile: &ProfileConfig, models: Vec<String>) {
        let now = SystemTime::now();
        self.profiles.insert(
            cache_key(profile),
            CachedProfileModels {
                provider: profile.provider_type.clone(),
                base_url: profile.base_url.clone(),
                case_index: case_index(&models),
                models,
                fetched_at: now,
                expires_at: now + MODEL_CACHE_TTL,
                last_error: None,
            },
        );
    }

    fn diagnostic_json(&self) -> Value {
        let profiles = self
            .profiles
            .iter()
            .map(|(key, cached)| {
                json!({
                    "key": key,
                    "provider": match cached.provider {
                        ProviderType::DirectAnthropic => "anthropic",
                        ProviderType::OpenAICompatible => "openai-compatible",
                        ProviderType::OpenAIResponses => "openai-responses",
                    },
                    "base_url": cached.base_url,
                    "status": if cached.expires_at > SystemTime::now() { "fresh" } else { "stale" },
                    "fetched_at_unix_ms": unix_ms(cached.fetched_at),
                    "expires_at_unix_ms": unix_ms(cached.expires_at),
                    "model_count": cached.models.len(),
                    "models": cached.models,
                    "case_index": cached.case_index,
                    "last_error": cached.last_error,
                })
            })
            .collect::<Vec<_>>();

        json!({
            "object": "claudex.model_cache",
            "profiles": profiles,
        })
    }
}

fn unix_ms(time: SystemTime) -> u128 {
    time.duration_since(SystemTime::UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ProfileConfig, ProfileModels};
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[test]
    fn test_profile_model_entries_include_slots_without_duplicates() {
        let profile = ProfileConfig {
            name: "codex".to_string(),
            default_model: "gpt-5.5".to_string(),
            models: ProfileModels {
                haiku: Some("gpt-5.5-mini".to_string()),
                sonnet: Some("gpt-5.5".to_string()),
                opus: Some("gpt-5.5-pro".to_string()),
            },
            ..ProfileConfig::default()
        };

        let entries = profile_model_entries(&profile);
        let ids = entries
            .iter()
            .map(|entry| entry["id"].as_str().unwrap())
            .collect::<Vec<_>>();

        assert_eq!(ids, vec!["gpt-5.5", "gpt-5.5-mini", "gpt-5.5-pro"]);
    }

    #[test]
    fn parse_model_ids_extracts_data_ids() {
        let value =
            json!({"object": "list", "data": [{"id": "GPT-5.5-Pro"}, {"id": "gpt-5.5"}, {}]});
        assert_eq!(parse_model_ids(&value), vec!["GPT-5.5-Pro", "gpt-5.5"]);
    }

    #[test]
    fn upstream_model_urls_match_provider_shapes() {
        let anthropic = ProfileConfig {
            provider_type: ProviderType::DirectAnthropic,
            base_url: "https://api.anthropic.com".to_string(),
            ..ProfileConfig::default()
        };
        let openai = ProfileConfig {
            provider_type: ProviderType::OpenAICompatible,
            base_url: "https://api.openai.com/v1".to_string(),
            query_params: HashMap::from([("api-version".to_string(), "2026-01-01".to_string())]),
            ..ProfileConfig::default()
        };

        assert_eq!(
            upstream_models_url(&anthropic),
            "https://api.anthropic.com/v1/models"
        );
        assert_eq!(
            upstream_models_url(&openai),
            "https://api.openai.com/v1/models?api-version=2026-01-01"
        );
    }

    #[test]
    fn merge_models_prefers_upstream_case_for_duplicates() {
        let profile = ProfileConfig {
            name: "local".to_string(),
            default_model: "gpt-5.5-pro".to_string(),
            ..ProfileConfig::default()
        };
        let discovered = vec!["GPT-5.5-Pro".to_string(), "Other".to_string()];
        let entries = profile_model_entries_with_discovered(&profile, Some(&discovered));
        let ids = entries
            .iter()
            .map(|entry| entry["id"].as_str().unwrap())
            .collect::<Vec<_>>();

        assert_eq!(ids, vec!["GPT-5.5-Pro", "Other"]);
    }

    #[tokio::test]
    async fn repair_model_case_uses_cached_models_only() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "object": "list",
                "data": [{"id": "GPT-5.5-Pro"}]
            })))
            .expect(0)
            .mount(&server)
            .await;
        let profile = ProfileConfig {
            name: "local".to_string(),
            provider_type: ProviderType::OpenAICompatible,
            base_url: format!("{}/v1", server.uri()),
            default_model: "gpt-5.5-pro".to_string(),
            ..ProfileConfig::default()
        };
        let cache = new_model_cache();
        cache
            .write()
            .await
            .insert_models(&profile, vec!["GPT-5.5-Pro".to_string()]);
        let mut body = json!({"model": "gpt-5.5-pro"});

        repair_model_case(&Client::new(), &cache, &profile, &mut body).await;

        assert_eq!(body["model"], "GPT-5.5-Pro");
    }

    #[tokio::test]
    async fn repair_model_case_fails_open_when_discovery_fails() {
        let profile = ProfileConfig {
            name: "local".to_string(),
            provider_type: ProviderType::OpenAICompatible,
            base_url: "http://127.0.0.1:9/v1".to_string(),
            default_model: "gpt-5.5-pro".to_string(),
            ..ProfileConfig::default()
        };
        let cache = new_model_cache();
        let mut body = json!({"model": "gpt-5.5-pro"});

        repair_model_case(&Client::new(), &cache, &profile, &mut body).await;

        assert_eq!(body["model"], "gpt-5.5-pro");
    }
    #[tokio::test]
    async fn case_index_repairs_case_only() {
        let profile = ProfileConfig {
            name: "local".to_string(),
            provider_type: ProviderType::OpenAICompatible,
            base_url: "http://127.0.0.1:8000/v1".to_string(),
            default_model: "gpt-5.5-pro".to_string(),
            ..ProfileConfig::default()
        };
        let cache = new_model_cache();
        cache.write().await.insert_models(
            &profile,
            vec!["GPT-5.5-Pro".to_string(), "gpt-5.5".to_string()],
        );

        assert_eq!(
            repair_model_case_value(&cache, &profile, "gpt-5.5-pro").await,
            Some("GPT-5.5-Pro".to_string())
        );
        assert_eq!(
            repair_model_case_value(&cache, &profile, "gpt-5.5x").await,
            None
        );
    }

    #[tokio::test]
    async fn stale_cache_is_returned_with_discovery_error() {
        let profile = ProfileConfig {
            name: "local".to_string(),
            provider_type: ProviderType::OpenAICompatible,
            base_url: "http://127.0.0.1:9/v1".to_string(),
            default_model: "gpt-5.5".to_string(),
            ..ProfileConfig::default()
        };
        let cache = new_model_cache();
        cache
            .write()
            .await
            .insert_models(&profile, vec!["GPT-5.5".to_string()]);
        let models = discover_or_cached_models(&Client::new(), &cache, &profile).await;
        assert_eq!(models, Some(vec!["GPT-5.5".to_string()]));
        assert!(cache
            .read()
            .await
            .profiles
            .get(&cache_key(&profile))
            .unwrap()
            .last_error
            .is_none());
    }

    #[test]
    fn diagnostic_cache_does_not_include_secrets() {
        let profile = ProfileConfig {
            name: "local".to_string(),
            provider_type: ProviderType::OpenAICompatible,
            base_url: "http://127.0.0.1:8000/v1".to_string(),
            api_key: "secret-key".to_string(),
            default_model: "gpt-5.5".to_string(),
            ..ProfileConfig::default()
        };
        let mut cache = ModelCache::default();
        cache.insert_models(&profile, vec!["gpt-5.5".to_string()]);

        let text = cache.diagnostic_json().to_string();
        assert!(!text.contains("secret-key"));
        assert!(!text.contains("api_key"));
        assert!(!text.contains("authorization"));
    }
}
