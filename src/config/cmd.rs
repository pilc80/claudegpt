use std::io::IsTerminal;

use anyhow::{Context, Result};

use crate::cli::ConfigAction;
use crate::oauth::{AuthType, OAuthProvider, OAuthToken};

use super::{ClaudexConfig, ProfileConfig};

pub async fn dispatch(action: ConfigAction, config: &mut ClaudexConfig) -> Result<()> {
    match action {
        ConfigAction::Show => cmd_show(config),
        ConfigAction::Doctor {
            json,
            fix,
            profile,
            connectivity,
        } => cmd_doctor(config, json, fix, &profile, connectivity).await,
        ConfigAction::Migrate { yes } => cmd_migrate(config, yes).await,
    }
}

/// On-request, non-silenced migration to the canonical config format.
///
/// Backs up the active config file, reports any keys present in the file that are
/// not part of the current schema (legacy/renamed fields that would otherwise be
/// silently dropped), then rewrites the file in the canonical format.
async fn cmd_migrate(config: &mut ClaudexConfig, yes: bool) -> Result<()> {
    let path = match config.config_source.clone() {
        Some(path) => path,
        None => {
            println!("No config file found; nothing to migrate.");
            return Ok(());
        }
    };

    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read config: {}", path.display()))?;
    let unknown = detect_unknown_config_keys(&raw, &path)?;

    // Credentials/keys must NEVER be dropped. If any unknown key looks like a
    // secret, refuse to migrate rather than risk losing it on the canonical rewrite.
    let credentials: Vec<&String> = unknown
        .iter()
        .filter(|key| looks_like_credential(key))
        .collect();
    if !credentials.is_empty() {
        println!("WARNING: credential/secret-like keys are present but not in the current schema:");
        for key in &credentials {
            println!("  - {key}");
        }
        println!(
            "Refusing to migrate: all credentials must be kept. Port these to the current\n\
             schema (or confirm they are obsolete), then re-run `config migrate`."
        );
        anyhow::bail!("migration aborted to preserve credential-like keys");
    }

    if unknown.is_empty() {
        println!("No unknown/deprecated keys found; config is already canonical.");
    } else {
        println!("Unknown/deprecated keys that will be dropped on rewrite:");
        for key in &unknown {
            println!("  - {key}");
        }
        tracing::warn!(
            count = unknown.len(),
            keys = ?unknown,
            "config migration detected unknown/deprecated keys"
        );
    }

    if !yes
        && !prompt_yes_no(
            "Rewrite this config in the canonical format (a backup will be created)?",
            false,
        )?
    {
        println!("Aborted; config unchanged.");
        return Ok(());
    }

    let backup = backup_config_path(&path);
    std::fs::copy(&path, &backup)
        .with_context(|| format!("failed to back up config to {}", backup.display()))?;
    println!("Backed up to {}", backup.display());

    // Re-load from the file only (without CLAUDEX_* env overrides merged in by
    // discover_config) so transient environment values are not baked into the file.
    let canonical = ClaudexConfig::load_from(&path)?;
    canonical.save()?;
    println!(
        "Migrated {} to canonical format ({} profile(s) kept, {} unknown key(s) dropped).",
        path.display(),
        canonical.profiles.len(),
        unknown.len()
    );
    Ok(())
}

/// Backup name: `<file>.bak-YYYYMMDD-HHMMSS-{pid}` (never retimestamped).
/// PID suffix avoids collisions when multiple migrate runs happen in the same second.
fn backup_config_path(path: &std::path::Path) -> std::path::PathBuf {
    let ts = chrono::Local::now().format("%Y%m%d-%H%M%S");
    let stem = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "config.toml".to_string());
    path.with_file_name(format!("{stem}.bak-{ts}-{}", std::process::id()))
}

/// Compare the raw config file against the current schema, returning dotted paths
/// of keys present in the file but absent from `ClaudexConfig` (the "oldfugs").
fn detect_unknown_config_keys(raw: &str, path: &std::path::Path) -> Result<Vec<String>> {
    let ext = path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_else(|| "toml".to_string());

    let have: serde_json::Value = match ext.as_str() {
        "yaml" | "yml" => serde_yml::from_str(raw).context("failed to parse YAML config")?,
        _ => toml::from_str(raw).context("failed to parse TOML config")?,
    };

    // Canonical schema; inject a sample profile so [[profiles]] element shape is known.
    let mut known = serde_json::to_value(ClaudexConfig::default())
        .context("failed to serialize canonical config")?;
    if let Some(profiles) = known
        .get_mut("profiles")
        .and_then(|value| value.as_array_mut())
    {
        profiles.push(
            serde_json::to_value(ProfileConfig::default())
                .context("failed to serialize canonical profile")?,
        );
    }

    let mut unknown = Vec::new();
    diff_unknown_keys(&have, &known, "", &mut unknown);
    unknown.sort();
    Ok(unknown)
}

fn diff_unknown_keys(
    have: &serde_json::Value,
    known: &serde_json::Value,
    prefix: &str,
    out: &mut Vec<String>,
) {
    match (have, known) {
        // An empty `known` object marks a map/dictionary field (HashMap) whose keys
        // are user data, not schema fields — never flag its entries as unknown.
        (serde_json::Value::Object(have_map), serde_json::Value::Object(known_map))
            if !known_map.is_empty() =>
        {
            for (key, value) in have_map {
                let path = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };
                match known_map.get(key) {
                    Some(known_value) => diff_unknown_keys(value, known_value, &path, out),
                    None => out.push(path),
                }
            }
        }
        (serde_json::Value::Array(have_arr), serde_json::Value::Array(known_arr)) => {
            if let Some(known_element) = known_arr.first() {
                for (index, element) in have_arr.iter().enumerate() {
                    diff_unknown_keys(element, known_element, &format!("{prefix}[{index}]"), out);
                }
            }
        }
        _ => {}
    }
}

/// Heuristic: does this config key look like a credential/secret?
/// Conservative on purpose — a false positive makes migration refuse (safe side).
fn looks_like_credential(key: &str) -> bool {
    let leaf = key.rsplit('.').next().unwrap_or(key).to_ascii_lowercase();
    const MARKERS: &[&str] = &[
        "secret",
        "token",
        "password",
        "passwd",
        "credential",
        "apikey",
        "api_key",
        "access_key",
        "cookie",
        "authorization",
        "bearer",
        "jwt",
        "session",
    ];
    MARKERS.iter().any(|marker| leaf.contains(marker)) || leaf == "key" || leaf.ends_with("_key")
}

fn cmd_show(config: &ClaudexConfig) -> Result<()> {
    println!("Config:");
    println!(
        "  active: {}",
        config
            .config_source
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "(none)".to_string())
    );
    println!("  global: {}", ClaudexConfig::config_path()?.display());
    println!();
    println!("Runtime defaults:");
    println!("  claude binary: {}", config.claude_binary);
    println!();
    println!("Profiles:");
    if config.profiles.is_empty() {
        println!("  (none)");
        return Ok(());
    }
    for profile in &config.profiles {
        println!(
            "  {:<16} {:<8} {:<10} {:<18} {}",
            profile.name,
            if profile.enabled {
                "enabled"
            } else {
                "disabled"
            },
            format!("{:?}", profile.provider_type),
            profile.default_model,
            profile.base_url
        );
    }
    Ok(())
}

#[derive(Debug, serde::Serialize)]
struct DoctorReport {
    status: DoctorStatus,
    errors: Vec<String>,
    warnings: Vec<String>,
    actions: Vec<String>,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "snake_case")]
enum DoctorStatus {
    Ok,
    NeedsSetup,
    Error,
}

async fn cmd_doctor(
    config: &mut ClaudexConfig,
    json: bool,
    fix: bool,
    profile: &str,
    connectivity: bool,
) -> Result<()> {
    let report = build_doctor_report(config, connectivity).await;

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&report).context("failed to serialize doctor report")?
        );
    } else {
        print_doctor_report(config, &report);
    }

    if !json && (fix || std::io::stdin().is_terminal()) {
        offer_doctor_fix(config, &report, profile).await?;
    }

    match report.status {
        DoctorStatus::Ok => Ok(()),
        DoctorStatus::NeedsSetup => std::process::exit(2),
        DoctorStatus::Error => std::process::exit(1),
    }
}

async fn build_doctor_report(config: &ClaudexConfig, connectivity: bool) -> DoctorReport {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();
    let mut actions = Vec::new();

    if config.config_source.is_none() {
        errors.push("config file was not found".to_string());
        actions
            .push("run `claudex-config config doctor --fix` to set up ChatGPT/Codex".to_string());
    }

    if config.profiles.is_empty() {
        errors.push("no profiles configured".to_string());
        actions.push(
            "create a profile with `claudex-config auth login chatgpt --profile codex-sub`"
                .to_string(),
        );
    } else if config.enabled_profiles().is_empty() {
        errors.push("no enabled profiles configured".to_string());
        actions.push("enable a profile or create one with `claudex-config auth login chatgpt --profile codex-sub`".to_string());
    }

    let mut seen_names = std::collections::HashSet::new();
    let checks_oauth_sources = config
        .profiles
        .iter()
        .any(|p| p.enabled && p.auth_type == AuthType::OAuth);
    if checks_oauth_sources {
        print_oauth_source_notice(
            "doctor checks OAuth token health using configured provider credential files and environment variables",
        );
    }

    for p in &config.profiles {
        if !seen_names.insert(&p.name) {
            errors.push(format!("duplicate profile name: '{}'", p.name));
        }
        for backup in &p.backup_providers {
            if config.find_profile(backup).is_none() {
                errors.push(format!(
                    "profile '{}': backup_provider '{}' does not exist",
                    p.name, backup
                ));
            }
        }
        if p.auth_type == AuthType::OAuth && p.oauth_provider.is_none() {
            errors.push(format!(
                "profile '{}': auth_type is 'oauth' but oauth_provider is not set",
                p.name
            ));
        }
        if !p.base_url.starts_with("http://") && !p.base_url.starts_with("https://") {
            errors.push(format!(
                "profile '{}': base_url must start with http:// or https://",
                p.name
            ));
        }
        if p.enabled && p.auth_type == AuthType::ApiKey && p.api_key.is_empty() {
            warnings.push(format!(
                "profile '{}': enabled with auth_type=ApiKey but no api_key",
                p.name
            ));
        }
        if p.enabled && p.auth_type == AuthType::ApiKey && p.api_key_keyring.is_some() {
            warnings.push(format!(
                "profile '{}': api_key_keyring is configured but keyring storage is disabled",
                p.name
            ));
        }
        if p.enabled && p.auth_type == AuthType::OAuth {
            let provider = p
                .oauth_provider
                .as_ref()
                .map(|provider| provider.normalize());
            add_oauth_token_health_warnings(
                &p.name,
                provider.as_ref().map(load_oauth_token_without_keyring),
                &mut warnings,
                &mut actions,
            );
        }
    }

    if config.router.enabled
        && !config.router.profile.is_empty()
        && config.find_profile(&config.router.profile).is_none()
    {
        warnings.push(format!(
            "router.profile '{}' does not match any profile",
            config.router.profile
        ));
    }
    if config.context.compression.enabled
        && !config.context.compression.profile.is_empty()
        && config
            .find_profile(&config.context.compression.profile)
            .is_none()
    {
        warnings.push(format!(
            "context.compression.profile '{}' does not match any profile",
            config.context.compression.profile
        ));
    }
    if config.context.rag.enabled
        && !config.context.rag.profile.is_empty()
        && config.find_profile(&config.context.rag.profile).is_none()
    {
        warnings.push(format!(
            "context.rag.profile '{}' does not match any profile",
            config.context.rag.profile
        ));
    }

    // Warn if any profile has openai_compatible_auto_compact enabled and
    // reserve_tokens is still at the old low default (4096).
    for p in &config.profiles {
        if p.openai_compatible_auto_compact.enabled
            && p.openai_compatible_auto_compact.reserve_tokens == 4096
        {
            warnings.push(format!(
                "profile '{}': openai_compatible_auto_compact is enabled with reserve_tokens=4096 (default). \
                 Consider raising it (e.g. to 64000) for large-window models.",
                p.name
            ));
        }
        if p.openai_compatible_auto_compact.enabled {
            // Note: claudex also forces CLAUDE_CODE_AUTO_COMPACT_WINDOW=262k on launch.
            // If reserve_tokens is set low, the proxy-side compaction may fire before
            // Claude Code's client-side compaction, producing unexpected truncation.
        }
    }

    if which::which(&config.claude_binary).is_err() {
        warnings.push(format!(
            "Claude Code binary '{}' was not found in PATH",
            config.claude_binary
        ));
    }

    match crate::process::daemon::read_pid() {
        Ok(Some(pid)) => match crate::process::daemon::is_proxy_running() {
            Ok(true) => actions.push(format!("proxy daemon is running with PID {pid}")),
            Ok(false) => warnings.push(format!("stale proxy PID file for PID {pid}; run `claudex-config proxy status` to clean it up")),
            Err(e) => warnings.push(format!("could not check proxy PID {pid}: {e}")),
        },
        Ok(None) => {}
        Err(e) => warnings.push(format!("could not read proxy PID file: {e}")),
    }

    if connectivity {
        for p in config.enabled_profiles() {
            if let Err(e) = super::profile::test_connectivity(p).await {
                warnings.push(format!("profile '{}': connectivity failed: {e}", p.name));
            }
        }
    }

    let status = if config.profiles.is_empty() || config.enabled_profiles().is_empty() {
        DoctorStatus::NeedsSetup
    } else if errors.is_empty() {
        DoctorStatus::Ok
    } else {
        DoctorStatus::Error
    };

    DoctorReport {
        status,
        errors,
        warnings,
        actions,
    }
}

const OAUTH_EXPIRY_WARNING_DAYS: i64 = 7;

fn add_oauth_token_health_warnings(
    profile_name: &str,
    token: Option<Result<OAuthToken>>,
    warnings: &mut Vec<String>,
    actions: &mut Vec<String>,
) {
    match token {
        Some(Ok(token)) => {
            add_oauth_expiry_warnings(profile_name, token.expires_at, warnings, actions)
        }
        Some(Err(e)) => {
            warnings.push(format!(
                "profile '{profile_name}': OAuth token is not available from configured non-keyring sources: {e}"
            ));
            actions.push(format!(
                "reauthenticate with `claudex-config auth login chatgpt --profile {profile_name}`"
            ));
        }
        None => {}
    }
}

fn load_oauth_token_without_keyring(provider: &OAuthProvider) -> Result<OAuthToken> {
    crate::oauth::source::load_credential_chain(provider).map(|cred| cred.into_oauth_token())
}

fn add_oauth_expiry_warnings(
    profile_name: &str,
    expires_at: Option<i64>,
    warnings: &mut Vec<String>,
    actions: &mut Vec<String>,
) {
    let Some(expires_at) = expires_at else {
        return;
    };
    let now = chrono::Utc::now().timestamp_millis();
    let remaining_ms = expires_at - now;
    if remaining_ms <= 0 {
        warnings.push(format!("profile '{profile_name}': OAuth token is expired"));
        actions.push(format!(
            "reauthenticate with `claudex-config auth login chatgpt --profile {profile_name}`"
        ));
        return;
    }
    let warning_ms = OAUTH_EXPIRY_WARNING_DAYS * 24 * 60 * 60 * 1000;
    if remaining_ms <= warning_ms {
        let days = (remaining_ms + 86_399_999) / 86_400_000;
        warnings.push(format!(
            "profile '{profile_name}': OAuth token expires in {days} day(s)"
        ));
        actions.push(format!(
            "reauthenticate with `claudex-config auth login chatgpt --profile {profile_name}`"
        ));
    }
}

fn print_oauth_source_notice(message: &str) {
    if std::io::stderr().is_terminal() {
        eprintln!("\x1b[33mNote:\x1b[0m {message}.");
    } else {
        eprintln!("Note: {message}.");
    }
}

fn print_doctor_report(config: &ClaudexConfig, report: &DoctorReport) {
    println!("Claudex doctor");
    println!();
    println!("Config:");
    println!(
        "  path: {}",
        config
            .config_source
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "not found".to_string())
    );
    println!("  version: {}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("Profiles:");
    if config.profiles.is_empty() {
        println!("  none");
    } else {
        for profile in &config.profiles {
            println!(
                "  {} ({}, {:?}, {})",
                profile.name,
                if profile.enabled {
                    "enabled"
                } else {
                    "disabled"
                },
                profile.provider_type,
                profile.default_model
            );
        }
    }
    println!();
    println!("Checks:");
    if report.errors.is_empty() && report.warnings.is_empty() {
        println!("  OK: setup looks usable");
    }
    for error in &report.errors {
        println!("  ERROR: {error}");
    }
    for warning in &report.warnings {
        println!("  WARNING: {warning}");
    }
    if !report.actions.is_empty() {
        println!();
        println!("Info / next actions:");
        for action in &report.actions {
            println!("  - {action}");
        }
    }
}

async fn offer_doctor_fix(
    config: &mut ClaudexConfig,
    report: &DoctorReport,
    profile: &str,
) -> Result<()> {
    if matches!(report.status, DoctorStatus::NeedsSetup) {
        if prompt_yes_no("Set up a ChatGPT/Codex OAuth profile now?", false)? {
            crate::oauth::providers::login(config, "chatgpt", profile, false, false, None).await?;
        }
        return Ok(());
    }

    let needs_reauth = report
        .actions
        .iter()
        .any(|action| action.contains("reauthenticate with"));
    if needs_reauth && prompt_yes_no("Re-authenticate ChatGPT/Codex now?", false)? {
        crate::oauth::providers::login(config, "chatgpt", profile, true, false, None).await?;
    }
    Ok(())
}

fn prompt_yes_no(prompt: &str, default: bool) -> Result<bool> {
    if !std::io::stdin().is_terminal() {
        return Ok(default);
    }
    let suffix = if default { "[Y/n]" } else { "[y/N]" };
    println!("{prompt} {suffix} ");
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    let answer = input.trim();
    if answer.is_empty() {
        return Ok(default);
    }
    Ok(matches!(answer, "y" | "Y" | "yes" | "YES" | "Yes"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ClaudexConfig, ProfileConfig, ProviderType};

    fn make_profile(name: &str, enabled: bool) -> ProfileConfig {
        ProfileConfig {
            name: name.to_string(),
            provider_type: ProviderType::OpenAIResponses,
            base_url: "https://example.com".to_string(),
            api_key: "test-key".to_string(),
            api_key_keyring: None,
            default_model: "gpt-5.5".to_string(),
            backup_providers: Vec::new(),
            custom_headers: Default::default(),
            extra_env: Default::default(),
            priority: 100,
            enabled,
            auth_type: AuthType::ApiKey,
            oauth_provider: None,
            models: Default::default(),
            image_model: None,
            max_tokens: None,
            strip_params: Default::default(),
            query_params: Default::default(),
            reasoning_bridge: Default::default(),
            openai_compatible_auto_compact: Default::default(),
        }
    }

    #[tokio::test]
    async fn doctor_reports_needs_setup_without_profiles() {
        let config = ClaudexConfig::default();
        let report = build_doctor_report(&config, false).await;
        assert!(matches!(report.status, DoctorStatus::NeedsSetup));
        assert!(report.errors.iter().any(|e| e.contains("no profiles")));
    }

    #[tokio::test]
    async fn doctor_accepts_enabled_oauth_profile() {
        let mut config = ClaudexConfig {
            config_source: Some(std::path::PathBuf::from("/tmp/config.toml")),
            ..Default::default()
        };
        config.profiles.push(make_profile("codex-sub", true));
        let report = build_doctor_report(&config, false).await;
        assert!(matches!(report.status, DoctorStatus::Ok));
        assert!(report.errors.is_empty());
    }

    #[test]
    fn doctor_adds_reauth_action_when_oauth_token_is_missing() {
        let mut warnings = Vec::new();
        let mut actions = Vec::new();

        add_oauth_token_health_warnings(
            "codex-sub",
            Some(Err(anyhow::anyhow!(
                "no OAuth token found in configured sources"
            ))),
            &mut warnings,
            &mut actions,
        );

        assert!(warnings.iter().any(|warning| warning
            .contains("OAuth token is not available from configured non-keyring sources")));
        assert!(actions.iter().any(|action| {
            action.contains("reauthenticate with")
                && action.contains("claudex-config auth login chatgpt --profile codex-sub")
        }));
    }

    #[tokio::test]
    async fn doctor_reports_duplicate_profile_names() {
        let mut config = ClaudexConfig {
            config_source: Some(std::path::PathBuf::from("/tmp/config.toml")),
            ..Default::default()
        };
        config.profiles.push(make_profile("codex-sub", true));
        config.profiles.push(make_profile("codex-sub", true));
        let report = build_doctor_report(&config, false).await;
        assert!(matches!(report.status, DoctorStatus::Error));
        assert!(report
            .errors
            .iter()
            .any(|e| e.contains("duplicate profile")));
    }

    #[test]
    fn detect_unknown_config_keys_flags_legacy_fields() {
        let path =
            std::env::temp_dir().join(format!("claudex-migrate-{}-flags.toml", std::process::id()));
        std::fs::write(
            &path,
            r#"
legacy_global = "dropped"

[[profiles]]
name = "test"
default_model = "gpt-5.5"
base_url = "https://example.com"
provider_type = "OpenAIResponses"
removed_field = "oldfug"
"#,
        )
        .unwrap();

        let raw = std::fs::read_to_string(&path).unwrap();
        let unknown = detect_unknown_config_keys(&raw, &path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert!(unknown.iter().any(|k| k == "legacy_global"));
        assert!(unknown.iter().any(|k| k == "profiles[0].removed_field"));
    }

    #[test]
    fn credential_heuristic_matches_secret_names() {
        assert!(looks_like_credential("profiles[0].legacy_api_key"));
        assert!(looks_like_credential("refresh_token"));
        assert!(looks_like_credential("client_secret"));
        assert!(looks_like_credential("password"));
        assert!(!looks_like_credential("default_model"));
        assert!(!looks_like_credential("base_url"));
    }

    #[test]
    fn detect_flags_credential_like_unknown_keys() {
        let path =
            std::env::temp_dir().join(format!("claudex-migrate-{}-cred.toml", std::process::id()));
        std::fs::write(
            &path,
            r#"
[[profiles]]
name = "test"
default_model = "gpt-5.5"
base_url = "https://example.com"
provider_type = "OpenAIResponses"
legacy_api_key = "sk-secret"
legacy_refresh_token = "tok"
"#,
        )
        .unwrap();
        let raw = std::fs::read_to_string(&path).unwrap();
        let unknown = detect_unknown_config_keys(&raw, &path).unwrap();
        let _ = std::fs::remove_file(&path);

        let cred_like: Vec<&String> = unknown
            .iter()
            .filter(|k| looks_like_credential(k))
            .collect();
        assert!(cred_like.iter().any(|k| k.contains("legacy_api_key")));
        assert!(cred_like.iter().any(|k| k.contains("legacy_refresh_token")));
    }

    #[test]
    fn detect_does_not_flag_map_entries_as_unknown() {
        let path =
            std::env::temp_dir().join(format!("claudex-migrate-{}-maps.toml", std::process::id()));
        std::fs::write(
            &path,
            r#"
[[profiles]]
name = "test"
default_model = "gpt-5.5"
base_url = "https://example.com"
provider_type = "OpenAIResponses"

[profiles.custom_headers]
X-Custom = "value"

[profiles.extra_env]
OPENAI_API_KEY = "sk-x"
FOO = "bar"

[profiles.query_params]
api-version = "2026-01-01"
"#,
        )
        .unwrap();
        let raw = std::fs::read_to_string(&path).unwrap();
        let unknown = detect_unknown_config_keys(&raw, &path).unwrap();
        let _ = std::fs::remove_file(&path);

        // HashMap entries are user data, not schema keys — must never be flagged.
        assert!(
            unknown.is_empty(),
            "legitimate map entries flagged as unknown: {unknown:?}"
        );
    }

    #[test]
    fn detect_unknown_config_keys_clean_for_canonical_config() {
        let canonical = toml::to_string(&ClaudexConfig::default()).unwrap();
        let path =
            std::env::temp_dir().join(format!("claudex-migrate-{}-clean.toml", std::process::id()));
        std::fs::write(&path, &canonical).unwrap();

        let unknown = detect_unknown_config_keys(&canonical, &path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert!(
            unknown.is_empty(),
            "canonical config flagged unknown keys: {unknown:?}"
        );
    }
}
