use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use anyhow::Context;
use base64::Engine;
use reqwest::{multipart, Url};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tokio::sync::RwLock;

use crate::config::{ProfileConfig, ProviderType};

const FILE_CACHE_TTL: Duration = Duration::from_secs(6 * 60 * 60);
const FILE_CACHE_MAX_ENTRIES: usize = 256;

#[derive(Debug, Default)]
pub struct ProviderFileCache {
    entries: HashMap<String, CachedFile>,
}

#[derive(Debug, Clone)]
struct CachedFile {
    file_id: String,
    expires_at: SystemTime,
    last_used_at: SystemTime,
    cleanup: FileCleanupTarget,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FileCleanupTarget {
    base_url: String,
    api_key: String,
    account_id: Option<String>,
    custom_headers: HashMap<String, String>,
}

#[derive(Debug, Clone)]
struct CachedFileCleanup {
    file_id: String,
    target: FileCleanupTarget,
}

pub type SharedProviderFileCache = Arc<RwLock<ProviderFileCache>>;

pub fn new_provider_file_cache() -> SharedProviderFileCache {
    Arc::new(RwLock::new(ProviderFileCache::default()))
}

pub async fn apply_provider_file_cache(
    client: &reqwest::Client,
    cache: &SharedProviderFileCache,
    profile: &ProfileConfig,
    body: &mut Value,
) {
    if !supports_openai_responses_files(profile) {
        return;
    }

    let Some(messages) = body
        .get_mut("messages")
        .and_then(|value| value.as_array_mut())
    else {
        return;
    };

    for message in messages {
        if let Some(content) = message.get_mut("content") {
            rewrite_media_blocks(client, cache, profile, content).await;
        }
    }
}

fn supports_openai_responses_files(profile: &ProfileConfig) -> bool {
    profile.provider_type == ProviderType::OpenAIResponses
        && !profile.api_key.is_empty()
        && openai_files_base_url(profile).is_some()
}

fn openai_files_base_url(profile: &ProfileConfig) -> Option<String> {
    let url = Url::parse(&profile.base_url).ok()?;
    if url.scheme() != "https" || url.host_str() != Some("api.openai.com") {
        return None;
    }
    Some(profile.base_url.trim_end_matches('/').to_string())
}

fn cleanup_target(profile: &ProfileConfig) -> Option<FileCleanupTarget> {
    Some(FileCleanupTarget {
        base_url: openai_files_base_url(profile)?,
        api_key: profile.api_key.clone(),
        account_id: profile.extra_env.get("CHATGPT_ACCOUNT_ID").cloned(),
        custom_headers: profile.custom_headers.clone(),
    })
}

fn rewrite_media_blocks<'a>(
    client: &'a reqwest::Client,
    cache: &'a SharedProviderFileCache,
    profile: &'a ProfileConfig,
    value: &'a mut Value,
) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
    Box::pin(async move {
        let Some(items) = value.as_array_mut() else {
            return;
        };

        for item in items {
            match item.get("type").and_then(|value| value.as_str()) {
                Some("document") => {
                    if let Err(err) = rewrite_document_block(client, cache, profile, item).await {
                        tracing::warn!(
                            profile = %profile.name,
                            error = %err,
                            "provider file_id cache upload/rewrite failed; using original document payload"
                        );
                    }
                }
                Some("image") => {
                    if let Err(err) = rewrite_image_block(client, cache, profile, item).await {
                        tracing::warn!(
                            profile = %profile.name,
                            error = %err,
                            "provider file_id cache upload/rewrite failed; using original image payload"
                        );
                    }
                }
                _ => {
                    if let Some(nested) = item.get_mut("content") {
                        rewrite_media_blocks(client, cache, profile, nested).await;
                    }
                }
            }
        }
    })
}

async fn rewrite_document_block(
    client: &reqwest::Client,
    cache: &SharedProviderFileCache,
    profile: &ProfileConfig,
    block: &mut Value,
) -> anyhow::Result<()> {
    let source_view = block
        .get("source")
        .and_then(|value| value.as_object())
        .context("document has no source object")?;
    if source_view.get("type").and_then(|value| value.as_str()) != Some("base64") {
        return Ok(());
    }
    let media_type = source_view
        .get("media_type")
        .and_then(|value| value.as_str())
        .unwrap_or("application/pdf")
        .to_string();
    let data = source_view
        .get("data")
        .and_then(|value| value.as_str())
        .filter(|value| !value.is_empty())
        .context("document has no base64 data")?
        .to_string();
    let filename = block
        .get("title")
        .and_then(|value| value.as_str())
        .unwrap_or("document.pdf")
        .to_string();
    let key = cache_key(profile, &media_type, &data);
    let source = block
        .get_mut("source")
        .and_then(|value| value.as_object_mut())
        .context("document has no source object")?;

    if let Some(file_id) = cache_lookup(client, cache, profile, &key).await {
        source.clear();
        source.insert("type".to_string(), json!("file"));
        source.insert("file_id".to_string(), json!(file_id));
        return Ok(());
    }

    let file_id = upload_file(client, profile, &filename, &media_type, &data).await?;
    cache_insert(client, cache, profile, key, file_id.clone()).await;
    source.clear();
    source.insert("type".to_string(), json!("file"));
    source.insert("file_id".to_string(), json!(file_id));
    Ok(())
}

async fn rewrite_image_block(
    client: &reqwest::Client,
    cache: &SharedProviderFileCache,
    profile: &ProfileConfig,
    block: &mut Value,
) -> anyhow::Result<()> {
    let source_view = block
        .get("source")
        .and_then(|value| value.as_object())
        .context("image has no source object")?;
    if source_view.get("type").and_then(|value| value.as_str()) != Some("base64") {
        return Ok(());
    }
    let media_type = source_view
        .get("media_type")
        .and_then(|value| value.as_str())
        .unwrap_or("image/png")
        .to_string();
    let data = source_view
        .get("data")
        .and_then(|value| value.as_str())
        .filter(|value| !value.is_empty())
        .context("image has no base64 data")?
        .to_string();
    let extension = media_type.rsplit('/').next().unwrap_or("png");
    let filename = format!("image.{extension}");
    let key = cache_key(profile, &media_type, &data);
    let source = block
        .get_mut("source")
        .and_then(|value| value.as_object_mut())
        .context("image has no source object")?;

    if let Some(file_id) = cache_lookup(client, cache, profile, &key).await {
        source.clear();
        source.insert("type".to_string(), json!("file"));
        source.insert("file_id".to_string(), json!(file_id));
        return Ok(());
    }

    let file_id = upload_file(client, profile, &filename, &media_type, &data).await?;
    cache_insert(client, cache, profile, key, file_id.clone()).await;
    source.clear();
    source.insert("type".to_string(), json!("file"));
    source.insert("file_id".to_string(), json!(file_id));
    Ok(())
}

async fn upload_file(
    client: &reqwest::Client,
    profile: &ProfileConfig,
    filename: &str,
    media_type: &str,
    base64_data: &str,
) -> anyhow::Result<String> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(base64_data)
        .context("failed to decode base64 document")?;
    let part = multipart::Part::bytes(bytes)
        .file_name(filename.to_string())
        .mime_str(media_type)
        .context("failed to build multipart file part")?;
    let form = multipart::Form::new()
        .text("purpose", "user_data")
        .part("file", part);
    let url = format!(
        "{}/files",
        openai_files_base_url(profile).context("unsupported OpenAI files base URL")?
    );
    let mut request = client
        .post(url)
        .header("Authorization", format!("Bearer {}", profile.api_key));
    if let Some(account_id) = profile.extra_env.get("CHATGPT_ACCOUNT_ID") {
        request = request.header("ChatGPT-Account-ID", account_id.as_str());
    }
    for (key, value) in &profile.custom_headers {
        request = request.header(key.as_str(), value.as_str());
    }
    let response = request
        .multipart(form)
        .send()
        .await
        .context("file upload request failed")?;
    if !response.status().is_success() {
        anyhow::bail!("file upload returned HTTP {}", response.status());
    }
    let value = response
        .json::<Value>()
        .await
        .context("file upload response was not JSON")?;
    value
        .get("id")
        .and_then(|value| value.as_str())
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .context("file upload response did not contain id")
}

async fn cache_lookup(
    client: &reqwest::Client,
    cache: &SharedProviderFileCache,
    profile: &ProfileConfig,
    key: &str,
) -> Option<String> {
    let mut guard = cache.write().await;
    let expired = guard.evict_expired();
    drop(guard);
    delete_remote_files(client, profile, expired).await;

    let mut guard = cache.write().await;
    let entry = guard.entries.get_mut(key)?;
    entry.last_used_at = SystemTime::now();
    Some(entry.file_id.clone())
}

async fn cache_insert(
    client: &reqwest::Client,
    cache: &SharedProviderFileCache,
    profile: &ProfileConfig,
    key: String,
    file_id: String,
) {
    let Some(cleanup) = cleanup_target(profile) else {
        return;
    };
    let mut cache = cache.write().await;
    let mut evicted = cache.evict_expired();
    cache.entries.insert(
        key,
        CachedFile {
            file_id,
            expires_at: SystemTime::now() + FILE_CACHE_TTL,
            last_used_at: SystemTime::now(),
            cleanup,
        },
    );
    evicted.extend(cache.evict_over_limit());
    drop(cache);
    delete_remote_files(client, profile, evicted).await;
}

async fn delete_remote_files(
    client: &reqwest::Client,
    profile: &ProfileConfig,
    files: Vec<CachedFileCleanup>,
) {
    for file in files {
        let url = format!("{}/files/{}", file.target.base_url, file.file_id);
        let mut request = client
            .delete(url)
            .header("Authorization", format!("Bearer {}", file.target.api_key));
        if let Some(account_id) = file.target.account_id.as_deref() {
            request = request.header("ChatGPT-Account-ID", account_id);
        }
        for (key, value) in &file.target.custom_headers {
            request = request.header(key.as_str(), value.as_str());
        }
        match request.send().await {
            Ok(response) if response.status().is_success() || response.status().as_u16() == 404 => {
            }
            Ok(response) => tracing::warn!(
                profile = %profile.name,
                file_id = %file.file_id,
                status = %response.status(),
                "provider file_id cache remote cleanup failed"
            ),
            Err(err) => tracing::warn!(
                profile = %profile.name,
                file_id = %file.file_id,
                error = %err,
                "provider file_id cache remote cleanup request failed"
            ),
        }
    }
}

fn cache_key(profile: &ProfileConfig, media_type: &str, data: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data.as_bytes());
    hasher.update(profile.custom_headers.len().to_string().as_bytes());
    let mut custom_headers = profile.custom_headers.iter().collect::<Vec<_>>();
    custom_headers.sort_by(|a, b| a.0.cmp(b.0));
    for (key, value) in custom_headers {
        hasher.update(key.as_bytes());
        hasher.update(b"\0");
        hasher.update(value.as_bytes());
        hasher.update(b"\0");
    }
    format!(
        "{}|{}|{}|{}|{}|{:x}",
        profile.name,
        profile.base_url.trim_end_matches('/'),
        profile.api_key,
        profile
            .extra_env
            .get("CHATGPT_ACCOUNT_ID")
            .map_or("", String::as_str),
        media_type,
        hasher.finalize()
    )
}

impl ProviderFileCache {
    fn evict_expired(&mut self) -> Vec<CachedFileCleanup> {
        let now = SystemTime::now();
        let expired = self
            .entries
            .iter()
            .filter(|(_, entry)| entry.expires_at <= now)
            .map(|(_, entry)| CachedFileCleanup {
                file_id: entry.file_id.clone(),
                target: entry.cleanup.clone(),
            })
            .collect::<Vec<_>>();
        self.entries.retain(|_, entry| entry.expires_at > now);
        expired
    }

    fn evict_over_limit(&mut self) -> Vec<CachedFileCleanup> {
        let mut evicted = Vec::new();
        while self.entries.len() > FILE_CACHE_MAX_ENTRIES {
            let Some(key) = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used_at)
                .map(|(key, _)| key.clone())
            else {
                return evicted;
            };
            if let Some(entry) = self.entries.remove(&key) {
                evicted.push(CachedFileCleanup {
                    file_id: entry.file_id,
                    target: entry.cleanup,
                });
            }
        }
        evicted
    }

    #[cfg(test)]
    fn insert_for_test(
        &mut self,
        profile: &ProfileConfig,
        media_type: &str,
        data: &str,
        file_id: &str,
    ) {
        self.entries.insert(
            cache_key(profile, media_type, data),
            CachedFile {
                file_id: file_id.to_string(),
                expires_at: SystemTime::now() + FILE_CACHE_TTL,
                last_used_at: SystemTime::now(),
                cleanup: cleanup_target(profile).unwrap(),
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn openai_profile() -> ProfileConfig {
        ProfileConfig {
            name: "openai".to_string(),
            provider_type: ProviderType::OpenAIResponses,
            base_url: "https://api.openai.com/v1".to_string(),
            api_key: "test-key".to_string(),
            default_model: "gpt-5.5".to_string(),
            ..ProfileConfig::default()
        }
    }

    #[test]
    fn cache_key_includes_profile_account_and_content() {
        let profile = openai_profile();
        let other = ProfileConfig {
            name: "other".to_string(),
            ..profile.clone()
        };
        let other_key = ProfileConfig {
            api_key: "other-key".to_string(),
            ..profile.clone()
        };
        let other_headers = ProfileConfig {
            custom_headers: HashMap::from([("OpenAI-Project".to_string(), "proj_2".to_string())]),
            ..profile.clone()
        };
        assert_ne!(
            cache_key(&profile, "application/pdf", "AAAA"),
            cache_key(&other, "application/pdf", "AAAA")
        );
        assert_ne!(
            cache_key(&profile, "application/pdf", "AAAA"),
            cache_key(&other_key, "application/pdf", "AAAA")
        );
        assert_ne!(
            cache_key(&profile, "application/pdf", "AAAA"),
            cache_key(&other_headers, "application/pdf", "AAAA")
        );
        assert_ne!(
            cache_key(&profile, "application/pdf", "AAAA"),
            cache_key(&profile, "application/pdf", "BBBB")
        );
    }

    #[test]
    fn file_cache_support_requires_exact_openai_https_host() {
        let profile = openai_profile();
        assert!(supports_openai_responses_files(&profile));
        assert!(!supports_openai_responses_files(&ProfileConfig {
            base_url: "https://api.openai.com.evil.test/v1".to_string(),
            ..profile.clone()
        }));
        assert!(!supports_openai_responses_files(&ProfileConfig {
            base_url: "http://api.openai.com/v1".to_string(),
            ..profile
        }));
    }

    #[tokio::test]
    async fn cached_document_rewrites_to_file_source() {
        let profile = openai_profile();
        let cache = new_provider_file_cache();
        cache
            .write()
            .await
            .insert_for_test(&profile, "application/pdf", "QUJD", "file_abc");
        let mut body = json!({
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "document",
                    "title": "a.pdf",
                    "source": {"type": "base64", "media_type": "application/pdf", "data": "QUJD"}
                }]
            }]
        });

        apply_provider_file_cache(&reqwest::Client::new(), &cache, &profile, &mut body).await;

        let source = &body["messages"][0]["content"][0]["source"];
        assert_eq!(source["type"], "file");
        assert_eq!(source["file_id"], "file_abc");
        assert!(source.get("data").is_none());
    }

    #[tokio::test]
    async fn cached_image_rewrites_to_file_source() {
        let profile = openai_profile();
        let cache = new_provider_file_cache();
        cache
            .write()
            .await
            .insert_for_test(&profile, "image/png", "QUJD", "file_img");
        let mut body = json!({
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}
                }]
            }]
        });

        apply_provider_file_cache(&reqwest::Client::new(), &cache, &profile, &mut body).await;

        let source = &body["messages"][0]["content"][0]["source"];
        assert_eq!(source["type"], "file");
        assert_eq!(source["file_id"], "file_img");
        assert!(source.get("data").is_none());
    }

    #[tokio::test]
    async fn unsupported_provider_leaves_document_base64() {
        let profile = ProfileConfig {
            name: "local".to_string(),
            provider_type: ProviderType::OpenAICompatible,
            base_url: "http://127.0.0.1:8000/v1".to_string(),
            api_key: "test-key".to_string(),
            default_model: "gpt-5.5".to_string(),
            ..ProfileConfig::default()
        };
        let cache = new_provider_file_cache();
        let mut body = json!({"messages": [{"content": [{"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "QUJD"}}]}]});

        apply_provider_file_cache(&reqwest::Client::new(), &cache, &profile, &mut body).await;

        assert_eq!(
            body["messages"][0]["content"][0]["source"]["type"],
            "base64"
        );
        assert_eq!(body["messages"][0]["content"][0]["source"]["data"], "QUJD");
    }
}
