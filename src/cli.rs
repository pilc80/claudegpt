use std::path::PathBuf;

use clap::{builder::styling, Args, Parser, Subcommand};

fn plain_help_styles() -> styling::Styles {
    styling::Styles::plain()
}

#[derive(Parser)]
#[command(
    name = "claudex-config",
    version,
    about = "Configure and manage the Claudex Claude Code translation proxy",
    styles = plain_help_styles()
)]
pub struct Cli {
    /// Override config file path
    #[arg(long, global = true, value_name = "PATH")]
    pub config: Option<PathBuf>,
    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Run Claude Code with a specific profile
    Run {
        /// Profile name to use
        profile: String,
        /// Override the model for this session
        #[arg(short, long)]
        model: Option<String>,
        /// Enable terminal hyperlinks (OSC 8) for clickable paths and URLs
        #[arg(long)]
        hyperlinks: bool,
        /// Extra arguments passed to claude
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },

    /// Manage profiles
    Profile {
        #[command(subcommand)]
        action: ProfileAction,
    },

    /// Manage the translation proxy
    Proxy {
        #[command(subcommand)]
        action: ProxyAction,
    },

    /// Manage configuration
    Config(ConfigCommand),

    /// Manage OAuth authentication for subscription services
    Auth {
        #[command(subcommand)]
        action: AuthAction,
    },

    /// Manage Claude Code configuration sets
    Sets {
        #[command(subcommand)]
        action: SetsAction,
    },
}

#[derive(Subcommand)]
pub enum ProfileAction {
    /// List all profiles
    List,
    /// Add a new profile interactively
    Add,
    /// Remove a profile
    Remove {
        /// Profile name
        name: String,
    },
    /// Test connectivity of a profile
    Test {
        /// Profile name (or "all")
        name: String,
    },
    /// Show profile details
    Show {
        /// Profile name
        name: String,
    },
}

#[derive(Subcommand)]
pub enum AuthAction {
    /// Login to an OAuth provider (claude, chatgpt/openai, google, qwen, kimi, github/copilot, gitlab)
    Login {
        /// OAuth provider to log in to (WHO you authenticate with): claude,
        /// chatgpt/openai/codex, google/gemini, qwen, kimi/moonshot,
        /// github/copilot, gitlab. Omit to list available providers + profiles.
        provider: Option<String>,
        /// Profile (from config.toml) to bind this login to (HOW claudex uses
        /// the token: base_url/model). Defaults to <provider>. Multiple profiles
        /// can share one provider login.
        #[arg(short, long)]
        profile: Option<String>,
        /// REQUIRED to re-authenticate when a (stale/invalidated) token already
        /// exists on disk; otherwise login is a no-op ("credentials available").
        #[arg(short, long)]
        force: bool,
        /// Use headless device code flow (for SSH/no-browser environments)
        #[arg(long)]
        headless: bool,
        /// GitHub Enterprise host (e.g. company.ghe.com) for Copilot Enterprise
        #[arg(long, value_name = "DOMAIN")]
        enterprise_url: Option<String>,
    },
    /// Show OAuth token status
    Status {
        /// Profile name (show all if omitted)
        #[arg(short, long)]
        profile: Option<String>,
    },
    /// Remove OAuth token for a profile
    Logout {
        /// Profile name
        profile: String,
    },
    /// Force refresh OAuth token
    Refresh {
        /// Profile name
        profile: String,
    },
}

#[derive(Subcommand)]
pub enum SetsAction {
    /// Install a configuration set from git repo, local path, or URL
    Add {
        /// Source: git URL, local path, or HTTP URL
        source: String,
        /// Install globally (~/.claude/)
        #[arg(long)]
        global: bool,
        /// Pin to a specific git ref (tag/branch/commit)
        #[arg(long)]
        r#ref: Option<String>,
    },
    /// Remove an installed configuration set
    Remove {
        /// Set name
        name: String,
        /// Remove from global scope
        #[arg(long)]
        global: bool,
    },
    /// List installed configuration sets
    List {
        /// List global sets
        #[arg(long)]
        global: bool,
    },
    /// Update configuration sets to latest version
    Update {
        /// Set name (omit to update all)
        name: Option<String>,
        /// Update global sets
        #[arg(long)]
        global: bool,
    },
    /// Show details of an installed configuration set
    Show {
        /// Set name
        name: String,
        /// Show global set
        #[arg(long)]
        global: bool,
    },
}

#[derive(Args)]
pub struct ConfigCommand {
    #[command(subcommand)]
    pub action: Option<ConfigAction>,
}

#[derive(Subcommand)]
pub enum ConfigAction {
    /// Show config paths and loaded config summary
    Show,
    /// Check and repair Claudex setup health
    Doctor {
        /// Output machine-readable diagnostics
        #[arg(long)]
        json: bool,
        /// Offer interactive fixes for detected setup problems
        #[arg(long)]
        fix: bool,
        /// Profile name to create or repair
        #[arg(long, default_value = "codex-sub")]
        profile: String,
        /// Also test enabled provider connectivity
        #[arg(long)]
        connectivity: bool,
    },
    /// Migrate config to the current canonical format (backs up first, reports dropped keys)
    Migrate {
        /// Apply the rewrite without prompting
        #[arg(long)]
        yes: bool,
    },
}

#[derive(Subcommand)]
pub enum ProxyAction {
    /// Start the proxy server (foreground)
    Start {
        /// Port override
        #[arg(short, long)]
        port: Option<u16>,
    },
    /// Stop the proxy daemon
    Stop,
    /// Show proxy status
    Status,
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::{CommandFactory, Parser};

    fn root_help() -> String {
        Cli::command().render_long_help().to_string()
    }

    #[test]
    fn root_help_exposes_core_management_commands() {
        let help = root_help();
        assert!(help.contains("run"));
        assert!(help.contains("profile"));
        assert!(help.contains("proxy"));
        assert!(help.contains("config"));
        assert!(help.contains("auth"));
        assert!(help.contains("sets"));
    }

    #[test]
    fn root_help_omits_obsolete_side_commands() {
        let help = root_help();
        assert!(!help.contains("dashboard"));
        assert!(!help.contains("Self-update claudex binary"));
    }

    #[test]
    fn proxy_start_rejects_daemon_option() {
        assert!(Cli::try_parse_from(["claudex-config", "proxy", "start", "--daemon"]).is_err());
    }

    #[test]
    fn auth_login_help_explains_provider_profile_and_force() {
        let mut login_cmd = Cli::command()
            .get_subcommands()
            .find(|c| c.get_name() == "auth")
            .and_then(|auth| auth.get_subcommands().find(|c| c.get_name() == "login"))
            .expect("auth login subcommand exists")
            .clone();
        let help = login_cmd.render_long_help().to_string();

        // Provider (WHO) + profile (HOW) distinction + the re-auth foot-gun note.
        assert!(help.contains("WHO you authenticate"));
        assert!(help.contains("chatgpt/openai/codex"));
        assert!(help.contains("REQUIRED to re-authenticate"));
    }

    #[test]
    fn auth_login_accepts_no_provider() {
        // provider is now optional so omitting it lists options instead of erroring.
        assert!(Cli::try_parse_from(["claudex-config", "auth", "login"]).is_ok());
    }
}
