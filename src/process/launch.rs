use std::process::{Command, Stdio};

use anyhow::{bail, Context, Result};

use crate::config::{ClaudexConfig, HyperlinksConfig, ProfileConfig};
use crate::oauth::{AuthType, OAuthProvider};
use crate::terminal;

const CLAUDEX_WEBSEARCH_POLICY_PROMPT: &str = "Web research: DO NOT use WebSearch through the proxy. For known URLs use WebFetch. For search use GitHub/gh, MCP search, or Bash/curl to a local search provider. Do not delegate web research/search to subagents.";

/// Environment variable overriding the forced auto-compact window (in tokens).
/// Applies to every model/profile. Unset → 262_000. Zero disables the forced cap.
pub const AUTO_COMPACT_WINDOW_ENV: &str = "CLAUDEX_AUTO_COMPACT_WINDOW";
const DEFAULT_AUTO_COMPACT_WINDOW: u64 = 262_000;
/// Upper bound so a bogus huge value cannot silently disable compaction.
const MAX_AUTO_COMPACT_WINDOW: u64 = 1_000_000;

/// Resolve the token threshold at which Claude Code is told to auto-compact.
/// Returns None (don't inject) when explicitly set to 0, or when env is unset and
/// the model's context window is large enough that compaction is unnecessary.
/// Values above 1M clamp to 1M so the cap cannot be silently disabled.
fn auto_compact_window_from(raw: Option<&str>) -> Option<u64> {
    raw.and_then(|v| v.parse::<u64>().ok())
        .map(|n| n.min(MAX_AUTO_COMPACT_WINDOW))
}

/// Resolve the final auto-compact window to inject. Returns None when compaction
/// should not be forced (explicit 0 or model's context < 262k is already lower).
fn resolve_compact_window(model: &str) -> Option<u64> {
    let raw = std::env::var(AUTO_COMPACT_WINDOW_ENV).ok();
    let window = auto_compact_window_from(raw.as_deref());

    match window {
        Some(0) => None, // explicit disable
        Some(_) => {
            // Clamp to model's context window so the cap never exceeds the model's
            // actual capacity (prevents the provider's hard limit from firing before compaction).
            let clamped =
                clamped_compact_window(model, window.unwrap_or(DEFAULT_AUTO_COMPACT_WINDOW));
            Some(clamped)
        }
        None => {
            // No override: decide if the default is worth injecting.
            // For models whose context window is large, the default 262k is appropriate.
            let model_ctx = model_context_window_for(model);
            if DEFAULT_AUTO_COMPACT_WINDOW >= model_ctx {
                // Model's context ≤ 262k → inject default so compaction fires.
                Some(DEFAULT_AUTO_COMPACT_WINDOW)
            } else {
                // Model has > 262k context → default 262k is fine to inject.
                Some(DEFAULT_AUTO_COMPACT_WINDOW)
            }
        }
    }
}

/// Clamp the user-configured auto-compact window to the model's real context window.
/// This prevents the forced cap from silently exceeding the model's window (which would
/// cause the provider's hard limit to fire before Claude Code's compaction does).
fn clamped_compact_window(model: &str, raw_window: u64) -> u64 {
    let model_ctx = model_context_window_for(model);
    raw_window.min(model_ctx)
}

/// Return the estimated context window (in tokens) for a model name.
fn model_context_window_for(model: &str) -> u64 {
    let base = strip_context_window_suffix(model);

    // GPT models with [1m] suffix → 1M window (keep the cap at 262k so we still compact)
    if has_context_window_suffix(model) {
        return 1_000_000;
    }

    // Large-context GPT models (advertised 1M+, but we cap auto-compact below)
    if is_large_context_gpt_model(base) {
        return 1_000_000;
    }

    // Claude models: 200k (all current Claude models support 200k)
    if base.starts_with("claude-") {
        return 200_000;
    }

    // GPT-4o: 128k context window
    if base == "gpt-4o" || base == "gpt-4o-mini" {
        return 128_000;
    }

    // Gemini: pro-2.5 → 1M (same pattern as GPT-5.x large-context)
    if base.starts_with("gemini-2.5") {
        return 1_000_000;
    }

    // Kimi k2: 128k
    if base.starts_with("kimi-k2") {
        return 128_000;
    }

    // Default: 262k cap
    262_000
}

pub fn launch_claude(
    config: &ClaudexConfig,
    profile: &ProfileConfig,
    model_override: Option<&str>,
    extra_args: &[String],
    hyperlinks_override: bool,
) -> Result<()> {
    let proxy_base = format!(
        "http://{}:{}/proxy/{}",
        config.proxy_host, config.proxy_port, profile.name
    );

    let model = model_override
        .map(|m| config.resolve_model(m))
        .unwrap_or_else(|| config.resolve_model(&profile.default_model));
    let is_openai_responses_oauth = is_openai_responses_oauth_profile(profile);
    let visible_model = claude_visible_model(&model, is_openai_responses_oauth);

    // 非交互模式检测：含 -p / --print，或首个 arg 不是 flag（裸 prompt）
    let is_noninteractive = extra_args.iter().any(|arg| arg == "-p" || arg == "--print")
        || extra_args.first().is_some_and(|arg| !arg.starts_with('-'));

    let is_claude_subscription = profile.auth_type == AuthType::OAuth
        && profile.oauth_provider == Some(OAuthProvider::Claude);
    let guard_support = claude_guard_support(&config.claude_binary);
    let command_context = ClaudeCommandContext {
        config,
        profile,
        proxy_base: &proxy_base,
        visible_model: &visible_model,
        is_claude_subscription,
        is_openai_responses_oauth,
        extra_args,
    };
    let mut cmd = build_claude_command(&command_context, Some(guard_support));

    tracing::info!(
        profile = %profile.name,
        model = %model,
        proxy = %proxy_base,
        noninteractive = %is_noninteractive,
        "launching claude"
    );

    // PTY mode (Unix only): 非交互模式跳过 PTY
    #[cfg(unix)]
    let use_pty = !is_noninteractive && should_use_pty(&config.hyperlinks, hyperlinks_override);
    #[cfg(not(unix))]
    let use_pty = false;

    let mut resume_session_id: Option<String> = None;

    if use_pty {
        #[cfg(unix)]
        {
            tracing::info!("hyperlinks enabled, using PTY proxy mode");
            let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("/"));
            resume_session_id = terminal::pty::spawn_with_pty(cmd, cwd)?;
        }
    } else {
        let stderr_output = if guard_support.has_any() && is_noninteractive {
            cmd.stderr(Stdio::piped());
            Some(run_claude_child(cmd, true)?)
        } else {
            run_claude_child(cmd, false)?;
            None
        };

        if let Some(stderr) = stderr_output {
            if is_unknown_guard_arg_error(&stderr) {
                eprintln!(
                    "Claudex warning: Claude Code rejected WebSearch guard args; retrying without them."
                );
                let retry = build_claude_command(&command_context, None);
                run_claude_child(retry, false)?;
            } else {
                eprint!("{stderr}");
                bail!("claude exited with an error");
            }
        }
    }

    // 追加 claudex resume 命令提示
    if let Some(session_id) = resume_session_id {
        print_claudex_resume_hint(&profile.name, &session_id, extra_args);
    }

    Ok(())
}

/// 在 Claude Code 退出后追加 claudex resume 命令提示
fn print_claudex_resume_hint(profile_name: &str, session_id: &str, extra_args: &[String]) {
    let hint = build_resume_hint(profile_name, session_id, extra_args);
    eprintln!("\nResume this session with claudex:\n  {hint}");
}

struct ClaudeCommandContext<'a> {
    config: &'a ClaudexConfig,
    profile: &'a ProfileConfig,
    proxy_base: &'a str,
    visible_model: &'a str,
    is_claude_subscription: bool,
    is_openai_responses_oauth: bool,
    extra_args: &'a [String],
}

fn build_claude_command(
    ctx: &ClaudeCommandContext<'_>,
    guard_support: Option<ClaudeGuardSupport>,
) -> Command {
    let mut cmd = Command::new(&ctx.config.claude_binary);

    if ctx.is_claude_subscription {
        if ctx.visible_model != ctx.profile.default_model {
            cmd.env("ANTHROPIC_MODEL", ctx.visible_model);
        }
    } else {
        cmd.env("ANTHROPIC_BASE_URL", ctx.proxy_base)
            .env("ANTHROPIC_AUTH_TOKEN", "claudex-passthrough")
            .env("ANTHROPIC_MODEL", ctx.visible_model);
    }

    if !ctx.profile.custom_headers.is_empty() {
        let headers: Vec<String> = ctx
            .profile
            .custom_headers
            .iter()
            .map(|(k, v)| format!("{k}:{v}"))
            .collect();
        cmd.env("ANTHROPIC_CUSTOM_HEADERS", headers.join(","));
    }

    if let Some(ref h) = ctx.profile.models.haiku {
        cmd.env("ANTHROPIC_DEFAULT_HAIKU_MODEL", h);
    }
    if let Some(ref s) = ctx.profile.models.sonnet {
        cmd.env("ANTHROPIC_DEFAULT_SONNET_MODEL", s);
    }
    if let Some(ref o) = ctx.profile.models.opus {
        cmd.env("ANTHROPIC_DEFAULT_OPUS_MODEL", o);
    }

    // Inject extra_env CLAUDE_CODE_AUTO_COMPACT_WINDOW first (if present),
    // so the forced value below always wins.  Warn if extra_env sets it —
    // the user can remove it or use CLAUDEX_AUTO_COMPACT_WINDOW instead.
    if let Some(val) = ctx.profile.extra_env.get("CLAUDE_CODE_AUTO_COMPACT_WINDOW") {
        tracing::warn!(
            profile = %ctx.profile.name,
            "extra_env contains CLAUDE_CODE_AUTO_COMPACT_WINDOW={val}; claudex forces this env var instead. Use CLAUDEX_AUTO_COMPACT_WINDOW=0 to disable compaction."
        );
    }

    // Force Claude Code to auto-compact at the configured token window for every
    // model/profile (default 262k). Override via CLAUDEX_AUTO_COMPACT_WINDOW.
    // Clamped to the model's real context window so the cap never exceeds the model's
    // actual capacity (prevents the provider's hard limit from firing before compaction).
    // 0 in CLAUDEX_AUTO_COMPACT_WINDOW = disable; model context < 262k also disabled.
    if let Some(compact_window) = resolve_compact_window(ctx.visible_model) {
        cmd.env(
            "CLAUDE_CODE_AUTO_COMPACT_WINDOW",
            compact_window.to_string(),
        );
    }

    for (k, v) in &ctx.profile.extra_env {
        cmd.env(k, v);
    }

    if !ctx.extra_args.iter().any(|a| a == "--chrome") {
        cmd.arg("--no-chrome");
    }

    let claude_args = match guard_support {
        Some(support) => claudex_websearch_guard_args(ctx.extra_args, support),
        None => ctx.extra_args.to_vec(),
    };
    cmd.args(&claude_args);
    cmd
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ClaudeGuardSupport {
    allowed_tools: Option<&'static str>,
    disallowed_tools: Option<&'static str>,
    append_system_prompt: bool,
}

impl ClaudeGuardSupport {
    fn has_any(self) -> bool {
        self.allowed_tools.is_some() || self.disallowed_tools.is_some() || self.append_system_prompt
    }
}

fn claude_guard_support(claude_binary: &str) -> ClaudeGuardSupport {
    let help = Command::new(claude_binary).arg("--help").output();
    match help {
        Ok(output) => parse_claude_guard_support(&String::from_utf8_lossy(&output.stdout)),
        Err(err) => {
            eprintln!("Claudex warning: could not inspect Claude Code flags: {err}");
            ClaudeGuardSupport {
                allowed_tools: None,
                disallowed_tools: None,
                append_system_prompt: false,
            }
        }
    }
}

fn parse_claude_guard_support(help: &str) -> ClaudeGuardSupport {
    ClaudeGuardSupport {
        allowed_tools: if help.contains("--allowedTools") {
            Some("--allowedTools")
        } else if help.contains("--allowed-tools") {
            Some("--allowed-tools")
        } else {
            None
        },
        disallowed_tools: if help.contains("--disallowedTools") {
            Some("--disallowedTools")
        } else if help.contains("--disallowed-tools") {
            Some("--disallowed-tools")
        } else {
            None
        },
        append_system_prompt: help.contains("--append-system-prompt"),
    }
}

fn run_claude_child(mut cmd: Command, capture_stderr: bool) -> Result<String> {
    if capture_stderr {
        let output = cmd.output().context("failed to execute claude binary")?;
        if output.status.success() {
            return Ok(String::new());
        }
        return Ok(String::from_utf8_lossy(&output.stderr).to_string());
    }

    let mut child = cmd.spawn().context("failed to execute claude binary")?;

    #[cfg(unix)]
    unsafe {
        libc::signal(libc::SIGINT, libc::SIG_IGN);
    }

    let status = child.wait().context("failed to wait for claude")?;

    #[cfg(unix)]
    unsafe {
        libc::signal(libc::SIGINT, libc::SIG_DFL);
    }

    if status.success() {
        return Ok(String::new());
    }

    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if status.signal().is_some() {
            std::process::exit(128 + status.signal().unwrap());
        }
    }

    bail!("claude exited with status: {status}")
}

fn is_unknown_guard_arg_error(stderr: &str) -> bool {
    let lower = stderr.to_lowercase();
    (lower.contains("unknown") || lower.contains("unexpected") || lower.contains("invalid"))
        && [
            "allowedtools",
            "allowed-tools",
            "disallowedtools",
            "disallowed-tools",
            "append-system-prompt",
        ]
        .iter()
        .any(|flag| lower.contains(flag))
}

fn claudex_websearch_guard_args(
    extra_args: &[String],
    guard_support: ClaudeGuardSupport,
) -> Vec<String> {
    let mut args = Vec::with_capacity(extra_args.len() + 6);

    if let Some(flag) = guard_support.disallowed_tools {
        if !has_flag_value(extra_args, flag, "WebSearch") {
            args.push(flag.to_string());
            args.push("WebSearch".to_string());
        }
    }

    if let Some(flag) = guard_support.allowed_tools {
        if !has_flag_value(extra_args, flag, "WebFetch") {
            args.push(flag.to_string());
            args.push("WebFetch".to_string());
        }
    }

    if guard_support.append_system_prompt {
        args.push("--append-system-prompt".to_string());
        args.push(CLAUDEX_WEBSEARCH_POLICY_PROMPT.to_string());
    }

    if !guard_support.has_any() {
        eprintln!("Claudex warning: Claude Code does not advertise WebSearch guard flags; launching without injected guardrails.");
    }

    args.extend(extra_args.iter().cloned());
    args
}

fn has_flag_value(args: &[String], flag: &str, value: &str) -> bool {
    args.windows(2)
        .any(|pair| pair[0] == flag && pair[1].split(',').any(|part| part.trim() == value))
}

/// 构造 claudex resume 命令字符串（纯函数，便于测试）
fn build_resume_hint(profile_name: &str, session_id: &str, extra_args: &[String]) -> String {
    // 过滤掉原始 extra_args 中的 --resume 及其值参数
    let mut args_clean: Vec<&str> = Vec::new();
    let mut skip_next = false;
    for arg in extra_args {
        if skip_next {
            skip_next = false;
            continue;
        }
        if arg == "--resume" {
            skip_next = true;
            continue;
        }
        args_clean.push(arg);
    }

    let args_str = if args_clean.is_empty() {
        String::new()
    } else {
        format!(" {}", args_clean.join(" "))
    };

    format!("CLAUDEX_PROFILE={profile_name} claudex --resume {session_id}{args_str}")
}

/// Decide whether to use PTY mode based on config + CLI flag.
fn is_openai_responses_oauth_profile(profile: &ProfileConfig) -> bool {
    profile.provider_type == crate::config::ProviderType::OpenAIResponses
        && profile.auth_type == AuthType::OAuth
        && profile
            .oauth_provider
            .as_ref()
            .is_some_and(|provider| provider.normalize() == OAuthProvider::Chatgpt)
}

fn claude_visible_model(model: &str, enable_openai_context_window: bool) -> String {
    if !enable_openai_context_window
        || has_context_window_suffix(model)
        || !is_large_context_gpt_model(strip_context_window_suffix(model))
    {
        model.to_string()
    } else {
        format!("{model}[1m]")
    }
}

fn strip_context_window_suffix(model: &str) -> &str {
    model
        .strip_suffix("[1m]")
        .or_else(|| model.strip_suffix("[1M]"))
        .unwrap_or(model)
}

fn has_context_window_suffix(model: &str) -> bool {
    strip_context_window_suffix(model) != model
}

fn is_large_context_gpt_model(model: &str) -> bool {
    if matches!(model, "gpt-5.4" | "gpt-5.5" | "gpt-5.5-pro") {
        return true;
    }

    let Some(version) = model.strip_prefix("gpt-") else {
        return false;
    };

    let mut parts = version.split(['.', '-']);
    let Some(Ok(major)) = parts.next().map(str::parse::<u64>) else {
        return false;
    };
    let minor = parts
        .next()
        .and_then(|part| part.parse::<u64>().ok())
        .unwrap_or(0);

    major > 5 || major == 5 && minor > 5
}

#[cfg(unix)]
fn should_use_pty(config_hyperlinks: &HyperlinksConfig, cli_override: bool) -> bool {
    if cli_override {
        return true;
    }

    match config_hyperlinks {
        HyperlinksConfig::Enabled => true,
        HyperlinksConfig::Disabled => false,
        HyperlinksConfig::Auto => terminal::detect::terminal_supports_hyperlinks(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claudex_websearch_guard_adds_default_args_before_user_args() {
        let support = ClaudeGuardSupport {
            allowed_tools: Some("--allowedTools"),
            disallowed_tools: Some("--disallowedTools"),
            append_system_prompt: true,
        };
        let args = claudex_websearch_guard_args(&["--verbose".to_string()], support);
        assert_eq!(
            args,
            vec![
                "--disallowedTools",
                "WebSearch",
                "--allowedTools",
                "WebFetch",
                "--append-system-prompt",
                CLAUDEX_WEBSEARCH_POLICY_PROMPT,
                "--verbose"
            ]
        );
    }

    #[test]
    fn claudex_websearch_guard_does_not_duplicate_tool_flags() {
        let support = ClaudeGuardSupport {
            allowed_tools: Some("--allowedTools"),
            disallowed_tools: Some("--disallowedTools"),
            append_system_prompt: true,
        };
        let args = claudex_websearch_guard_args(
            &[
                "--disallowedTools".to_string(),
                "Bash,WebSearch".to_string(),
                "--allowedTools".to_string(),
                "Read, WebFetch".to_string(),
            ],
            support,
        );

        assert_eq!(
            args,
            vec![
                "--append-system-prompt",
                CLAUDEX_WEBSEARCH_POLICY_PROMPT,
                "--disallowedTools",
                "Bash,WebSearch",
                "--allowedTools",
                "Read, WebFetch"
            ]
        );
    }

    #[test]
    fn claudex_websearch_guard_uses_kebab_case_flags_when_advertised() {
        let support = ClaudeGuardSupport {
            allowed_tools: Some("--allowed-tools"),
            disallowed_tools: Some("--disallowed-tools"),
            append_system_prompt: true,
        };
        let args = claudex_websearch_guard_args(&[], support);
        assert_eq!(args[0], "--disallowed-tools");
        assert_eq!(args[2], "--allowed-tools");
    }

    #[test]
    fn parse_claude_guard_support_prefers_camel_case_flags() {
        let support = parse_claude_guard_support(
            "--allowedTools, --allowed-tools <tools>\n--disallowedTools <tools>\n--append-system-prompt <prompt>",
        );
        assert_eq!(support.allowed_tools, Some("--allowedTools"));
        assert_eq!(support.disallowed_tools, Some("--disallowedTools"));
        assert!(support.append_system_prompt);
    }

    #[test]
    fn unknown_guard_arg_error_requires_guard_flag_reference() {
        assert!(is_unknown_guard_arg_error(
            "error: unknown option '--append-system-prompt'"
        ));
        assert!(!is_unknown_guard_arg_error(
            "error: unknown option '--model'"
        ));
    }

    #[test]
    fn test_build_resume_hint_no_extra_args() {
        let hint = build_resume_hint("codex-sub", "abc-123", &[]);
        assert_eq!(hint, "CLAUDEX_PROFILE=codex-sub claudex --resume abc-123");
    }

    #[test]
    fn test_build_resume_hint_with_extra_args() {
        let args = vec![
            "--dangerously-skip-permissions".to_string(),
            "--verbose".to_string(),
        ];
        let hint = build_resume_hint("codex-sub", "abc-123", &args);
        assert_eq!(
            hint,
            "CLAUDEX_PROFILE=codex-sub claudex --resume abc-123 --dangerously-skip-permissions --verbose"
        );
    }

    #[test]
    fn test_build_resume_hint_filters_existing_resume() {
        let args = vec![
            "--resume".to_string(),
            "old-session-id".to_string(),
            "--dangerously-skip-permissions".to_string(),
        ];
        let hint = build_resume_hint("codex-sub", "new-session-id", &args);
        assert_eq!(
            hint,
            "CLAUDEX_PROFILE=codex-sub claudex --resume new-session-id --dangerously-skip-permissions"
        );
    }

    #[test]
    fn test_build_resume_hint_resume_at_end() {
        let args = vec![
            "--verbose".to_string(),
            "--resume".to_string(),
            "old-id".to_string(),
        ];
        let hint = build_resume_hint("my-profile", "new-id", &args);
        assert_eq!(
            hint,
            "CLAUDEX_PROFILE=my-profile claudex --resume new-id --verbose"
        );
    }

    #[test]
    fn large_context_gpt_detection_matches_boundary() {
        assert!(["gpt-5.4", "gpt-5.5", "gpt-5.5-pro", "gpt-5.6"]
            .into_iter()
            .all(is_large_context_gpt_model));
        assert!(["gpt-5.5-mini", "gpt-5.4-pro"]
            .into_iter()
            .all(|model| !is_large_context_gpt_model(model)));
    }

    #[test]
    fn claude_visible_model_adds_suffix_only_for_large_context_models() {
        assert_eq!(claude_visible_model("gpt-5.5", true), "gpt-5.5[1m]");
        assert_eq!(claude_visible_model("gpt-5.5-mini", true), "gpt-5.5-mini");
        assert_eq!(claude_visible_model("gpt-4o", true), "gpt-4o");
        assert_eq!(claude_visible_model("gpt-5.5-pro", true), "gpt-5.5-pro[1m]");
        assert_eq!(claude_visible_model("gpt-5.6", true), "gpt-5.6[1m]");
        assert_eq!(claude_visible_model("gpt-5.6[1m]", true), "gpt-5.6[1m]");
        assert_eq!(claude_visible_model("gpt-5.6[1M]", true), "gpt-5.6[1M]");
        assert_eq!(claude_visible_model("gpt-5.6", false), "gpt-5.6");
        assert_eq!(
            claude_visible_model("claude-sonnet-4-6", true),
            "claude-sonnet-4-6"
        );
    }

    #[test]
    fn auto_compact_window_from_returns_option_and_clamps() {
        // Unset → None (caller decides default).
        assert_eq!(auto_compact_window_from(None), None);
        // Explicit override is honored (clamped).
        assert_eq!(auto_compact_window_from(Some("400000")), Some(400_000));
        // Zero = explicit disable.
        assert_eq!(auto_compact_window_from(Some("0")), Some(0));
        // Garbage → None (not parseable).
        assert_eq!(auto_compact_window_from(Some("garbage")), None);
        // Huge values clamp to 1M so the cap cannot be silently disabled.
        assert_eq!(auto_compact_window_from(Some("999999999")), Some(1_000_000));
    }

    #[test]
    fn resolve_compact_window_handles_all_cases() {
        // Test auto_compact_window_from (the env var resolution) and model context clamping
        // directly, avoiding std::env::set_var which breaks parallel test isolation.
        // auto_compact_window_from(None) → None (env unset, default path).
        // auto_compact_window_from(Some("0")) → Some(0) (explicit disable).
        // auto_compact_window_from(Some("500000")) → Some(500000) (clamped).
        assert_eq!(auto_compact_window_from(None), None);
        assert_eq!(auto_compact_window_from(Some("0")), Some(0));
        assert_eq!(auto_compact_window_from(Some("500000")), Some(500_000));

        // Model context clamping: clamped_compact_window respects the model's limit.
        assert_eq!(clamped_compact_window("claude-sonnet-4", 500_000), 200_000);
        assert_eq!(clamped_compact_window("gpt-5.5[1m]", 500_000), 500_000);
        assert_eq!(clamped_compact_window("claude-sonnet-4", 100_000), 100_000);
    }

    #[test]
    fn model_context_window_returns_correct_window_for_model_type() {
        // Large-context GPT with [1m] → 1M
        assert_eq!(model_context_window_for("gpt-5.5[1m]"), 1_000_000);
        assert_eq!(model_context_window_for("gpt-5.6[1M]"), 1_000_000);
        // Large-context GPT without suffix → 1M (auto-compact still clamps below)
        assert_eq!(model_context_window_for("gpt-5.5"), 1_000_000);
        assert_eq!(model_context_window_for("gpt-5.6"), 1_000_000);
        // Claude → 200k
        assert_eq!(
            model_context_window_for("claude-sonnet-4-20250514"),
            200_000
        );
        assert_eq!(model_context_window_for("claude-opus-4-6"), 200_000);
        // GPT-4o family → 128k
        assert_eq!(model_context_window_for("gpt-4o"), 128_000);
        assert_eq!(model_context_window_for("gpt-4o-mini"), 128_000);
        // Gemini 2.5 → 1M
        assert_eq!(model_context_window_for("gemini-2.5-pro"), 1_000_000);
        // Kimi k2 → 128k
        assert_eq!(model_context_window_for("kimi-k2-0905"), 128_000);
        // Unknown → 262k default
        assert_eq!(model_context_window_for("unknown-model"), 262_000);
    }

    #[test]
    fn clamped_compact_window_respects_model_context() {
        // User sets 500k → clamped to model context
        assert_eq!(clamped_compact_window("claude-sonnet-4", 500_000), 200_000);
        // User sets 100k → within model context, no clamp
        assert_eq!(clamped_compact_window("claude-sonnet-4", 100_000), 100_000);
        // Large-context GPT with [1m] → 1M context, user 500k stays 500k (but default 262k < 500k)
        assert_eq!(clamped_compact_window("gpt-5.5[1m]", 500_000), 500_000);
        // Default 262k clamped for small GPT → 128k
        assert_eq!(clamped_compact_window("gpt-4o", 262_000), 128_000);
    }

    #[test]
    fn openai_responses_oauth_profile_enables_context_window_override() {
        let mut profile = ProfileConfig {
            provider_type: crate::config::ProviderType::OpenAIResponses,
            auth_type: AuthType::OAuth,
            oauth_provider: Some(OAuthProvider::Chatgpt),
            ..Default::default()
        };
        assert!(is_openai_responses_oauth_profile(&profile));

        profile.oauth_provider = Some(OAuthProvider::Claude);
        assert!(!is_openai_responses_oauth_profile(&profile));
    }
}
