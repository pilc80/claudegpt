pub mod adapter;
pub mod context_engine;
pub mod error;
pub mod error_translation;
pub mod fallback;
pub mod file_cache;
pub mod handler;
pub mod health;
pub mod metrics;
pub mod models;
pub mod translate;
pub mod util;

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use axum::extract::DefaultBodyLimit;
use axum::http::{HeaderMap, HeaderValue};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::Router;
use tokio::sync::RwLock;

use crate::config::ClaudexConfig;
use crate::context::rag::RagIndex;
use crate::context::sharing::SharedContext;
use metrics::MetricsStore;

pub struct ProxyState {
    pub config: Arc<RwLock<ClaudexConfig>>,
    pub metrics: MetricsStore,
    pub http_client: reqwest::Client,
    pub health_status: Arc<RwLock<health::HealthMap>>,
    pub circuit_breakers: fallback::CircuitBreakerMap,
    pub shared_context: SharedContext,
    pub rag_index: Option<RagIndex>,
    pub token_manager: crate::oauth::manager::TokenManager,
    pub model_cache: models::SharedModelCache,
    pub file_cache: file_cache::SharedProviderFileCache,
}

pub const DEFAULT_REQUEST_BODY_LIMIT_BYTES: usize = 32 * 1024 * 1024;
pub const REQUEST_BODY_LIMIT_ENV: &str = "CLAUDEX_BODY_LIMIT_BYTES";
pub const HEALTH_VERSION_HEADER: &str = "x-claudex-version";
pub const HEALTH_BODY_LIMIT_HEADER: &str = "x-claudex-body-limit";
pub const UPSTREAM_CONNECT_TIMEOUT_SECS: u64 = 30;
pub const UPSTREAM_STREAM_READ_TIMEOUT_SECS: u64 = 300;

fn build_upstream_http_client() -> Result<reqwest::Client> {
    Ok(reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(UPSTREAM_CONNECT_TIMEOUT_SECS))
        .read_timeout(Duration::from_secs(UPSTREAM_STREAM_READ_TIMEOUT_SECS))
        .build()?)
}

pub fn request_body_limit_bytes_from_env(value: Option<&str>) -> Result<usize> {
    match value {
        Some(value) => value.parse::<usize>().with_context(|| {
            format!("{REQUEST_BODY_LIMIT_ENV} must be an unsigned integer byte count")
        }),
        None => Ok(DEFAULT_REQUEST_BODY_LIMIT_BYTES),
    }
}

/// 获取 proxy 日志文件路径（~/.cache/claudex/proxy-{timestamp}-{pid}.log）
/// 每次启动生成独立日志文件，支持多实例并行
pub fn proxy_log_path() -> Option<std::path::PathBuf> {
    dirs::cache_dir().map(|d| {
        let ts = chrono::Local::now().format("%Y%m%d-%H%M%S");
        let pid = std::process::id();
        d.join("claudex").join(format!("proxy-{ts}-{pid}.log"))
    })
}

pub async fn start_proxy(config: ClaudexConfig, port_override: Option<u16>) -> Result<()> {
    let port = port_override.unwrap_or(config.proxy_port);
    let host = config.proxy_host.clone();
    let request_body_limit =
        request_body_limit_bytes_from_env(std::env::var(REQUEST_BODY_LIMIT_ENV).ok().as_deref())?;

    let http_client = build_upstream_http_client()?;

    // Build RAG index if enabled
    let rag_index = if config.context.rag.enabled {
        let index = RagIndex::new(config.context.rag.clone());
        if let Some((base_url, api_key, _)) = crate::context::resolve_profile_endpoint(
            &config,
            &config.context.rag.profile,
            &config.context.rag.model,
        ) {
            if let Err(e) = index.build_index(&http_client, &base_url, &api_key).await {
                tracing::warn!("failed to build RAG index: {e}");
            }
        } else {
            tracing::warn!(
                profile = %config.context.rag.profile,
                "RAG profile not found, skipping index build"
            );
        }
        Some(index)
    } else {
        None
    };

    let token_manager = crate::oauth::manager::TokenManager::new(http_client.clone());

    let state = Arc::new(ProxyState {
        config: Arc::new(RwLock::new(config)),
        metrics: MetricsStore::new(),
        http_client,
        health_status: Arc::new(RwLock::new(health::HealthMap::new())),
        circuit_breakers: fallback::new_circuit_breaker_map(),
        shared_context: SharedContext::new(),
        rag_index,
        token_manager,
        model_cache: models::new_model_cache(),
        file_cache: file_cache::new_provider_file_cache(),
    });

    health::spawn_health_checker(state.clone());

    let app = Router::new()
        .route(
            "/v1/messages/count_tokens",
            post(handler::handle_count_tokens_root),
        )
        .route("/v1/models", get(models::list_models))
        .route("/debug/model-cache", get(models::model_cache_debug))
        .route(
            "/proxy/{profile}/v1/messages/count_tokens",
            post(handler::handle_count_tokens),
        )
        .route(
            "/proxy/{profile}/v1/messages",
            post(handler::handle_messages),
        )
        .route("/health", get(move || health_handler(request_body_limit)));

    let app = if request_body_limit > 0 {
        app.layer(DefaultBodyLimit::max(request_body_limit))
    } else {
        app.layer(DefaultBodyLimit::disable())
    }
    .with_state(state);

    let bind_addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;

    tracing::info!(
        body_limit_bytes = request_body_limit,
        "proxy listening on {bind_addr}"
    );

    crate::process::daemon::write_pid(std::process::id())?;

    axum::serve(listener, app).await?;

    crate::process::daemon::remove_pid()?;
    Ok(())
}

async fn health_handler(request_body_limit: usize) -> impl IntoResponse {
    let mut headers = HeaderMap::new();
    headers.insert(
        HEALTH_VERSION_HEADER,
        HeaderValue::from_static(env!("CARGO_PKG_VERSION")),
    );
    let body_limit = HeaderValue::from_str(&request_body_limit.to_string())
        .expect("usize string should be a valid header value");
    headers.insert(HEALTH_BODY_LIMIT_HEADER, body_limit);
    (headers, "ok")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_body_limit_defaults_to_anthropic_limit() {
        assert_eq!(
            request_body_limit_bytes_from_env(None).unwrap(),
            DEFAULT_REQUEST_BODY_LIMIT_BYTES
        );
    }

    #[test]
    fn request_body_limit_accepts_zero_for_provider_decides() {
        assert_eq!(request_body_limit_bytes_from_env(Some("0")).unwrap(), 0);
    }

    #[test]
    fn request_body_limit_accepts_custom_byte_count() {
        assert_eq!(
            request_body_limit_bytes_from_env(Some("67108864")).unwrap(),
            67_108_864
        );
    }

    #[test]
    fn request_body_limit_rejects_invalid_values() {
        let err = request_body_limit_bytes_from_env(Some("64mb")).unwrap_err();
        assert!(err.to_string().contains(REQUEST_BODY_LIMIT_ENV));
    }

    #[test]
    fn upstream_http_client_uses_idle_read_timeout_instead_of_total_deadline() {
        let client = build_upstream_http_client().unwrap();
        let debug = format!("{client:?}");

        assert!(debug.contains("read_timeout"));
        assert!(!debug.contains("total_timeout"));
    }
}
