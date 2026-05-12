//! `kanad-compute` CLI — Rust port of the Click commands in
//! `kanad_compute/cli.py`. Phase 3.1 implements `creds`, `status`, and
//! `login`; `connect` lands once the gateway loop is wired in 3.2.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use kanad_auth::DeviceFlow;
use kanad_vault::{Vault, CANONICAL_KEYS};
use std::io::{self, Write};
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "kanad-compute", version, about = "Local compute node for Kanad")]
struct Cli {
    /// Override the kanad-app base URL.
    #[arg(long, env = "KANAD_APP_URL", default_value = "https://app.kanad.dev")]
    url: String,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Show vault + connection status.
    Status,
    /// Manage stored credentials.
    Creds {
        #[command(subcommand)]
        action: CredsAction,
    },
    /// Device-authorization login flow.
    Login {
        /// Don't open the browser automatically.
        #[arg(long)]
        no_browser: bool,
        #[arg(long, default_value_t = 900)]
        timeout: u64,
    },
    /// Print version + build info.
    Version,
}

#[derive(Subcommand, Debug)]
enum CredsAction {
    /// Store a credential (canonical key).
    Set { key: String, value: String },
    /// Print a credential.
    Get {
        key: String,
        #[arg(long)]
        reveal: bool,
    },
    /// List all stored credentials (presence only).
    List,
    /// Remove a credential.
    Clear { key: String },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .init();

    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Status => cmd_status().await,
        Cmd::Creds { action } => cmd_creds(action).await,
        Cmd::Login { no_browser, timeout } => cmd_login(&cli.url, no_browser, timeout).await,
        Cmd::Version => {
            println!("kanad-compute {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
    }
}

fn redact(s: &str) -> String {
    if s.len() <= 4 {
        "****".into()
    } else {
        format!("****{}", &s[s.len() - 4..])
    }
}

async fn cmd_status() -> Result<()> {
    let v = Vault::new();
    println!("Vault status:");
    for (logical, present) in v.status() {
        println!("  {logical:<10}: {}", if present { "set" } else { "missing" });
    }
    Ok(())
}

async fn cmd_creds(action: CredsAction) -> Result<()> {
    let v = Vault::new();
    match action {
        CredsAction::Set { key, value } => {
            v.set(&key, &value).context("vault set failed")?;
            println!("stored {key}");
        }
        CredsAction::Get { key, reveal } => match v.get(&key) {
            Some(val) => {
                if reveal {
                    println!("{val}");
                } else {
                    println!("{}", redact(&val));
                }
            }
            None => {
                println!("(unset)");
            }
        },
        CredsAction::List => {
            for k in CANONICAL_KEYS {
                let present = v.has(k);
                println!("  {k:<22}: {}", if present { "set" } else { "" });
            }
        }
        CredsAction::Clear { key } => {
            if v.clear(&key) {
                println!("cleared {key}");
            } else {
                println!("{key} was not set");
            }
        }
    }
    Ok(())
}

async fn cmd_login(base_url: &str, no_browser: bool, timeout: u64) -> Result<()> {
    let flow = DeviceFlow::new(base_url, "kanad-compute-cli");
    let code = flow.request_code().await.context("device/code failed")?;

    println!();
    println!("To authorize this device, visit:");
    let verify = code
        .verification_uri_complete
        .clone()
        .unwrap_or_else(|| code.verification_uri.clone());
    println!("  {verify}");
    println!();
    println!("Code: {}", code.user_code);
    println!();
    io::stdout().flush().ok();

    if !no_browser {
        let _ = open_browser(&verify);
    }

    // Apply the user-supplied timeout as a hard upper bound.
    let expires_in = code.expires_in.min(timeout);
    let mut code = code;
    code.expires_in = expires_in;

    let token = flow.poll_token(&code).await.context("device/token poll failed")?;
    let v = Vault::new();
    v.set("kanad_access_token", &token.access_token)
        .context("vault set kanad_access_token failed")?;
    println!("Login successful — token stored in vault.");
    Ok(())
}

fn open_browser(url: &str) -> std::io::Result<()> {
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open").arg(url).status().map(|_| ())
    }
    #[cfg(target_os = "linux")]
    {
        std::process::Command::new("xdg-open").arg(url).status().map(|_| ())
    }
    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("cmd")
            .args(["/C", "start", "", url])
            .status()
            .map(|_| ())
    }
}
