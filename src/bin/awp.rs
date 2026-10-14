//! `awp` — Agents with Profiles document tool and profiled runner.
//!
//! - `awp inspect  --config <path> [--overlay <path>]...` — resolved registry as JSON.
//! - `awp validate --config <path> [--overlay <path>]...` — load + validate only.
//! - `awp resolve  --config <path> [--overlay <path>]... (--profile <id> | --binding <name>)`
//!   emits the materialization as JSON: the effective `a4aa` spec, the
//!   session defaults and the turn plan. Foreign-language hosts feed the
//!   spec to `a4aa serve` and apply the turn plan per prompt — identical
//!   semantics without reimplementing the merge rules.
//! - `awp run      --config <path> [--overlay <path>]... (--profile <id> | --binding <name>)`
//!   `[--cwd <dir>] [--prompt <text> | --prompt-file <path|->]` — one
//!   session, one prompt; events stream to stderr, the agent's full text
//!   lands on stdout.
//! - `awp fixture-agent` — a deterministic ACP echo agent for tests and
//!   registry smoke checks. Echoes each prompt back as message chunks;
//!   `wait-for-cancel` streams until `session/cancel`.

use awp::{AgentEvent, LaneKind, Registry, RegistryDocument, SessionRequest, VirtualAgent};
use serde_json::{json, Value};
use std::io::{BufRead, Read, Write};
use std::path::PathBuf;
use tokio::sync::mpsc;

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let code = match std::env::args().nth(1).as_deref() {
        Some("inspect") => inspect().await,
        Some("validate") => validate().await,
        Some("resolve") => resolve().await,
        Some("run") => run().await,
        Some("fixture-agent") => fixture_agent(),
        _ => {
            eprintln!(
                "usage: awp inspect|validate|resolve|run --config <path> [--overlay <path>]... \
                 [--profile <id> | --binding <name>]\n       awp fixture-agent"
            );
            2
        }
    };
    std::process::exit(code);
}

// --------------------------- argument parsing ---------------------------

#[derive(Default)]
struct Args {
    config: Option<PathBuf>,
    overlays: Vec<PathBuf>,
    profile: Option<String>,
    binding: Option<String>,
    cwd: Option<PathBuf>,
    prompt: Option<String>,
    prompt_file: Option<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args::default();
    let mut argv = std::env::args().skip(2);
    while let Some(flag) = argv.next() {
        let mut value = |name: &str| -> Result<String, String> {
            argv.next()
                .ok_or_else(|| format!("{name} requires a value"))
        };
        match flag.as_str() {
            "--config" => args.config = Some(PathBuf::from(value("--config")?)),
            "--overlay" => args.overlays.push(PathBuf::from(value("--overlay")?)),
            "--profile" => args.profile = Some(value("--profile")?),
            "--binding" => args.binding = Some(value("--binding")?),
            "--cwd" => args.cwd = Some(PathBuf::from(value("--cwd")?)),
            "--prompt" => args.prompt = Some(value("--prompt")?),
            "--prompt-file" => args.prompt_file = Some(PathBuf::from(value("--prompt-file")?)),
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    Ok(args)
}

fn load_registry(args: &Args) -> Result<Registry, String> {
    let config = args.config.as_ref().ok_or("--config is required")?;
    let mut registry = Registry::from_json(&read_file(config)?).map_err(|e| e.to_string())?;
    for overlay in &args.overlays {
        let document: RegistryDocument =
            serde_json::from_str(&read_file(overlay)?).map_err(|e| e.to_string())?;
        registry.apply(document).map_err(|e| e.to_string())?;
    }
    Ok(registry)
}

fn read_file(path: &PathBuf) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))
}

fn select_agent(args: &Args, registry: &Registry) -> Result<VirtualAgent, String> {
    match (&args.profile, &args.binding) {
        (Some(profile), None) => registry.resolve(profile).map_err(|e| e.to_string()),
        (None, Some(binding)) => registry.resolve_binding(binding).map_err(|e| e.to_string()),
        (None, None) => Err("one of --profile or --binding is required".to_owned()),
        (Some(_), Some(_)) => Err("--profile and --binding are exclusive".to_owned()),
    }
}

// ------------------------------- inspect --------------------------------

async fn inspect() -> i32 {
    let args = match parse_args() {
        Ok(args) => args,
        Err(e) => return usage_error(&e),
    };
    let registry = match load_registry(&args) {
        Ok(registry) => registry,
        Err(e) => return failure(&e),
    };
    let agents: Vec<Value> = registry
        .agents()
        .map(|spec| {
            json!({
                "id": spec.id,
                "lane": match spec.lane {
                    a4aa::LaneSpec::Acp { .. } => "acp",
                    a4aa::LaneSpec::Rpc(_) => "rpc",
                    a4aa::LaneSpec::Cli(_) => "cli",
                },
                "program": spec.command.program,
            })
        })
        .collect();
    let profiles: Vec<Value> = registry
        .profiles()
        .map(|profile| serde_json::to_value(profile).unwrap_or(Value::Null))
        .collect();
    let bindings: Vec<Value> = registry
        .bindings()
        .map(|(name, binding)| {
            json!({
                "name": name,
                "profile": binding.profile(),
                "prompt": binding.prompt(),
            })
        })
        .collect();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "agents": agents,
            "profiles": profiles,
            "bindings": bindings,
        }))
        .unwrap_or_default()
    );
    0
}

// ------------------------------- validate -------------------------------

async fn validate() -> i32 {
    let args = match parse_args() {
        Ok(args) => args,
        Err(e) => return usage_error(&e),
    };
    match load_registry(&args) {
        Ok(registry) => {
            eprintln!(
                "ok: {} agent(s), {} profile(s), {} binding(s)",
                registry.agents().len(),
                registry.profiles().len(),
                registry.bindings().len()
            );
            0
        }
        Err(e) => failure(&e),
    }
}

// ------------------------------- resolve --------------------------------

async fn resolve() -> i32 {
    let args = match parse_args() {
        Ok(args) => args,
        Err(e) => return usage_error(&e),
    };
    let registry = match load_registry(&args) {
        Ok(registry) => registry,
        Err(e) => return failure(&e),
    };
    let agent = match select_agent(&args, &registry) {
        Ok(agent) => agent,
        Err(e) => return failure(&e),
    };
    let profile = agent.profile();
    let output = json!({
        "profile": serde_json::to_value(profile).unwrap_or(Value::Null),
        "lane": match agent.lane_kind() {
            LaneKind::Acp => "acp",
            LaneKind::Rpc => "rpc",
            LaneKind::Cli => "cli",
        },
        "spec": serde_json::to_value(agent.spec()).unwrap_or(Value::Null),
        "session": {
            "cwd": profile.cwd.clone(),
        },
        "turn": {
            "model": profile.model.clone(),
            "effort": profile.effort.clone(),
            "instructions": profile.instructions.clone(),
            "instructionMode": profile.instruction_mode,
        },
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&output).unwrap_or_default()
    );
    0
}

// --------------------------------- run ----------------------------------

async fn run() -> i32 {
    let args = match parse_args() {
        Ok(args) => args,
        Err(e) => return usage_error(&e),
    };
    let registry = match load_registry(&args) {
        Ok(registry) => registry,
        Err(e) => return failure(&e),
    };
    let agent = match select_agent(&args, &registry) {
        Ok(agent) => agent,
        Err(e) => return failure(&e),
    };
    let text = match prompt_text(&args) {
        Ok(text) => text,
        Err(e) => return usage_error(&e),
    };

    let (events, mut rx) = mpsc::unbounded_channel::<AgentEvent>();
    let mut request = SessionRequest::default();
    if let Some(cwd) = args.cwd.clone() {
        request.cwd = Some(cwd);
    }
    let session = match agent.open_session(request, events).await {
        Ok(session) => session,
        Err(e) => return failure(&e.to_string()),
    };

    // Stream turn events to stderr in real time; stdout stays clean for the
    // final text.
    let drainer = tokio::spawn(async move {
        while let Some(event) = rx.recv().await {
            match event {
                AgentEvent::MessageChunk { text } => {
                    eprint!("{text}");
                    let _ = std::io::stderr().flush();
                }
                AgentEvent::ToolCall { summary, .. } => eprintln!("\n[tool] {summary}"),
                AgentEvent::Plan { .. } => {}
                AgentEvent::Truncated => eprintln!("\n[truncated]"),
                AgentEvent::Raw { .. } => {}
            }
        }
    });

    let outcome = session.prompt(text).await;
    let close = session.close().await;
    let _ = drainer.await;
    match outcome {
        Ok(outcome) => {
            // Session lanes deliver text through events (already streamed);
            // stateless lanes carry it on the outcome.
            if !outcome.text.is_empty() {
                println!("{}", outcome.text);
            }
            if let Err(e) = close {
                return failure(&e.to_string());
            }
            0
        }
        Err(e) => failure(&e.to_string()),
    }
}

fn prompt_text(args: &Args) -> Result<String, String> {
    match (&args.prompt, &args.prompt_file) {
        (Some(text), None) => Ok(text.clone()),
        (None, Some(path)) if path.as_os_str() == "-" => {
            let mut text = String::new();
            std::io::stdin()
                .read_to_string(&mut text)
                .map_err(|e| e.to_string())?;
            Ok(text)
        }
        (None, Some(path)) => read_file(path),
        (None, None) => Err("one of --prompt or --prompt-file is required".to_owned()),
        (Some(_), Some(_)) => Err("--prompt and --prompt-file are exclusive".to_owned()),
    }
}

// ------------------------------- helpers --------------------------------

fn usage_error(message: &str) -> i32 {
    eprintln!("error: {message}");
    2
}

fn failure(message: &str) -> i32 {
    eprintln!("error: {message}");
    1
}

// ----------------------------- fixture agent -----------------------------

/// Minimal deterministic ACP agent (protocol v1 over NDJSON stdio).
///
/// - `initialize` → protocolVersion 1, loadSession true.
/// - `session/new|load|resume` → fixed/echoed session id.
/// - `session/prompt` → `agent_message_chunk` updates echoing the prompt
///   text, then `{"stopReason":"end_turn"}`. Magic words:
///   `wait-for-cancel` streams chunks until `session/cancel`, answered with
///   `{"stopReason":"cancelled"}`.
fn fixture_agent() -> i32 {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut deferred_prompt: Option<(Value, String)> = None;

    let write = |value: Value| {
        let mut out = stdout.lock();
        let _ = out.write_all(value.to_string().as_bytes());
        let _ = out.write_all(b"\n");
        let _ = out.flush();
    };

    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let Ok(frame) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        let id = frame.get("id").cloned();
        let method = frame.get("method").and_then(Value::as_str).unwrap_or("");
        let params = frame.get("params").cloned().unwrap_or(Value::Null);

        match method {
            "initialize" => write(json!({
                "jsonrpc": "2.0", "id": id,
                "result": {
                    "protocolVersion": 1,
                    "agentCapabilities": {
                        "loadSession": true,
                        "promptCapabilities": {"image": false, "audio": false, "embeddedContext": true},
                        "mcpCapabilities": {"http": false, "sse": false}
                    },
                    "agentInfo": {"name": "awp-fixture", "version": VERSION},
                    "authMethods": []
                }
            })),
            "session/new" => write(json!({
                "jsonrpc": "2.0", "id": id,
                "result": {"sessionId": "fixture-session-1"}
            })),
            "session/load" | "session/resume" => write(json!({
                "jsonrpc": "2.0", "id": id,
                "result": {"sessionId": params.get("sessionId").cloned().unwrap_or(json!("fixture-resumed"))}
            })),
            "session/cancel" => {
                if let Some((prompt_id, _)) = deferred_prompt.take() {
                    write(json!({
                        "jsonrpc": "2.0", "id": prompt_id,
                        "result": {"stopReason": "cancelled"}
                    }));
                }
            }
            "session/prompt" => {
                let prompt_text = params
                    .get("prompt")
                    .and_then(Value::as_array)
                    .and_then(|blocks| blocks.first())
                    .and_then(|b| b.get("text"))
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned();
                let session_id = params
                    .get("sessionId")
                    .and_then(Value::as_str)
                    .unwrap_or("fixture-session-1")
                    .to_owned();
                if prompt_text.contains("wait-for-cancel") {
                    deferred_prompt = Some((id.unwrap_or(Value::Null), session_id));
                    continue;
                }
                let chunk = format!("echo: {prompt_text}");
                let mut mid = chunk.len() / 2;
                while !chunk.is_char_boundary(mid) {
                    mid -= 1;
                }
                for piece in [&chunk[..mid], &chunk[mid..]] {
                    write(json!({
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": {
                            "sessionId": session_id.clone(),
                            "update": {"sessionUpdate": "agent_message_chunk",
                                       "content": {"type": "text", "text": piece}}
                        }
                    }));
                }
                write(json!({
                    "jsonrpc": "2.0", "id": id.unwrap_or(Value::Null),
                    "result": {"stopReason": "end_turn"}
                }));
            }
            _ => {
                if let Some(id) = id {
                    write(json!({
                        "jsonrpc": "2.0", "id": id,
                        "error": {"code": -32601, "message": "method not found"}
                    }));
                }
            }
        }
    }
    0
}
