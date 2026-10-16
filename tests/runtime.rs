//! End-to-end profile semantics: instruction composition, model/effort
//! defaults, session cwd and cancellation — through real spawned lanes.
//!
//! The `acp` cases drive `awp fixture-agent`; the `cli` cases use ubiquitous
//! Unix tools, and the `rpc` cases use a small local JSON-RPC child fixture.

use a4aa::SessionConfig;
use awp::{
    AgentEvent, AgentSpec, CliLaneSpec, CommandSpec, InstructionMode, LaneSpec, ProfilePatch,
    PromptChannel, Registry, RegistryDocument, RpcCallSpec, RpcLaneSpec, SessionRequest,
    StreamMode, TurnRequest,
};
#[cfg(unix)]
use serde_json::json;
#[cfg(unix)]
use std::path::{Path, PathBuf};
#[cfg(unix)]
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;

#[cfg(unix)]
const RPC_AGENT: &str = r#"
import json, sys

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

def shown(value):
    return value if isinstance(value, str) and value else "unset"

session_model = None
session_effort = None
for line in sys.stdin:
    try:
        request = json.loads(line)
    except Exception:
        continue
    method = request.get("method")
    request_id = request.get("id")
    params = request.get("params") or {}
    if method == "session/open":
        session_model = params.get("model")
        session_effort = params.get("effort")
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        model = params.get("model") or session_model
        effort = params.get("effort") or session_effort
        prompt = params.get("text", "")
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "{}|{}|{}".format(
                  shown(model), shown(effort), prompt)}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

fn cli_spec(id: &str, program: &str, args: &[&str], channel: PromptChannel) -> AgentSpec {
    AgentSpec {
        id: id.to_owned(),
        client_name: None,
        lane: LaneSpec::Cli(Box::new(CliLaneSpec {
            args: args.iter().map(|a| a.to_string()).collect(),
            prompt_channel: channel,
            stream: StreamMode::Stdio,
            ..CliLaneSpec::default()
        })),
        command: CommandSpec {
            program: program.to_owned(),
            args: vec![],
            cwd: None,
            env: Default::default(),
        },
        max_output_bytes: None,
    }
}

#[cfg(unix)]
fn rpc_spec(
    id: &str,
    script: &std::path::Path,
    session_params: serde_json::Value,
    prompt_params: serde_json::Value,
) -> AgentSpec {
    AgentSpec {
        id: id.to_owned(),
        client_name: None,
        lane: LaneSpec::Rpc(Box::new(RpcLaneSpec {
            session_open: Some(RpcCallSpec {
                method: "session/open".to_owned(),
                params: session_params,
            }),
            prompt: RpcCallSpec {
                method: "session/prompt".to_owned(),
                params: prompt_params,
            },
            result_text_path: Some("text".to_owned()),
            ..RpcLaneSpec::default()
        })),
        command: CommandSpec {
            program: "python3".to_owned(),
            args: vec![script.display().to_string()],
            cwd: script.parent().map(std::path::Path::to_path_buf),
            env: Default::default(),
        },
        max_output_bytes: None,
    }
}

#[cfg(unix)]
fn write_rpc_agent(dir: &std::path::Path) -> std::path::PathBuf {
    let script = dir.join("rpc_agent.py");
    std::fs::write(&script, RPC_AGENT).unwrap();
    script
}

#[cfg(unix)]
struct TestDirectory(PathBuf);

#[cfg(unix)]
impl TestDirectory {
    fn new(label: &str) -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("awp-{label}-{}-{nonce}", std::process::id()));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

#[cfg(unix)]
impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn fixture_spec() -> AgentSpec {
    AgentSpec {
        id: "fixture".to_owned(),
        client_name: Some("awp-test".to_owned()),
        lane: LaneSpec::Acp {
            mcp_servers: vec![],
            prefer_load: false,
        },
        command: CommandSpec {
            program: env!("CARGO_BIN_EXE_awp").to_owned(),
            args: vec!["fixture-agent".to_owned()],
            cwd: None,
            env: Default::default(),
        },
        max_output_bytes: None,
    }
}

fn registry_with(spec: AgentSpec, patch: ProfilePatch) -> Registry {
    let mut document = RegistryDocument::default();
    document.agents.insert(spec.id.clone(), spec);
    document.profiles.insert("p".to_owned(), patch);
    Registry::from_document(document).unwrap()
}

fn events() -> (
    mpsc::UnboundedSender<AgentEvent>,
    mpsc::UnboundedReceiver<AgentEvent>,
) {
    mpsc::unbounded_channel()
}

// ------------------------------ cli lane ---------------------------------

#[cfg(unix)]
#[tokio::test]
async fn cli_instructions_arrive_on_every_turn() {
    let registry = registry_with(
        cli_spec("cat-agent", "cat", &[], PromptChannel::Stdin),
        ProfilePatch {
            agent: Some("cat-agent".to_owned()),
            instructions: Some("PERSONA".to_owned()),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    // A stateless lane has no memory: both turns carry the instructions —
    // InstructionMode::FirstTurn cannot apply where there is no session.
    for _ in 0..2 {
        let outcome = session.prompt("task".to_owned()).await.unwrap();
        assert_eq!(outcome.text.trim_end(), "PERSONA\n\ntask");
    }
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn cli_turn_defaults_fill_model_and_effort() {
    let registry = registry_with(
        cli_spec(
            "echo-agent",
            "echo",
            &["{model}|{effort}|{prompt}"],
            PromptChannel::ArgOnly,
        ),
        ProfilePatch {
            agent: Some("echo-agent".to_owned()),
            model: Some("m7".to_owned()),
            effort: Some("low".to_owned()),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    let outcome = session.prompt("work".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "m7|low|work");

    // An explicit turn value beats the profile default.
    let outcome = session
        .prompt_with(TurnRequest {
            text: "work".to_owned(),
            model: Some("m9".to_owned()),
            effort: None,
        })
        .await
        .unwrap();
    assert_eq!(outcome.text.trim_end(), "m9|low|work");
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn rpc_profile_defaults_apply_at_session_open_when_supported() {
    let dir = TestDirectory::new("init-only-rpc");
    let script = write_rpc_agent(dir.path());
    let registry = registry_with(
        rpc_spec(
            "init-only-rpc",
            &script,
            json!({"model": "{model}", "effort": "{effort}"}),
            json!({"sessionId": "{sessionId}", "text": "{prompt}"}),
        ),
        ProfilePatch {
            agent: Some("init-only-rpc".to_owned()),
            model: Some("profile-model".to_owned()),
            effort: Some("profile-effort".to_owned()),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    let unsupported_override = session
        .prompt_with(TurnRequest {
            text: "must-not-run".to_owned(),
            model: Some("turn-model".to_owned()),
            effort: None,
        })
        .await
        .expect_err("session-only selectors cannot accept an operation override");
    assert_eq!(unsupported_override.code, "agent_model_unsupported");

    let outcome = session.prompt("work".to_owned()).await.unwrap();
    session.close().await.unwrap();

    // The RPC has no turn selectors. The profile defaults arrive through
    // session/open and AwP leaves the per-turn request unset.
    assert_eq!(outcome.text, "profile-model|profile-effort|work");
}

#[cfg(unix)]
#[tokio::test]
async fn rpc_turn_only_profile_defaults_and_overrides_use_turn_templates() {
    let dir = TestDirectory::new("turn-only-rpc");
    let script = write_rpc_agent(dir.path());
    let registry = registry_with(
        rpc_spec(
            "turn-only-rpc",
            &script,
            json!({"cwd": "{cwd}"}),
            json!({
                "sessionId": "{sessionId}",
                "text": "{prompt}",
                "model": "{model}",
                "effort": "{effort}"
            }),
        ),
        ProfilePatch {
            agent: Some("turn-only-rpc".to_owned()),
            model: Some("profile-model".to_owned()),
            effort: Some("profile-effort".to_owned()),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    let defaults = session.prompt("work".to_owned()).await.unwrap();
    let override_turn = session
        .prompt_with(TurnRequest {
            text: "work".to_owned(),
            model: Some("turn-model".to_owned()),
            effort: Some("turn-effort".to_owned()),
        })
        .await
        .unwrap();
    session.close().await.unwrap();

    assert_eq!(defaults.text, "profile-model|profile-effort|work");
    assert_eq!(override_turn.text, "turn-model|turn-effort|work");
}

#[cfg(unix)]
#[tokio::test]
async fn explicit_session_selectors_precede_profile_defaults() {
    let dir = TestDirectory::new("rpc-session-overrides");
    let script = write_rpc_agent(dir.path());
    let registry = registry_with(
        rpc_spec(
            "session-rpc",
            &script,
            json!({"model": "{model}", "effort": "{effort}"}),
            json!({
                "sessionId": "{sessionId}",
                "text": "{prompt}",
                "model": "{model}",
                "effort": "{effort}"
            }),
        ),
        ProfilePatch {
            agent: Some("session-rpc".to_owned()),
            model: Some("profile-model".to_owned()),
            effort: Some("profile-effort".to_owned()),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(
            SessionRequest {
                config: SessionConfig {
                    model: Some("session-model".to_owned()),
                    effort: Some("session-effort".to_owned()),
                    ..Default::default()
                },
                ..Default::default()
            },
            tx,
        )
        .await
        .unwrap();

    let session_override = session.prompt("work".to_owned()).await.unwrap();
    let turn_override = session
        .prompt_with(TurnRequest {
            text: "work".to_owned(),
            model: Some("turn-model".to_owned()),
            effort: Some("turn-effort".to_owned()),
        })
        .await
        .unwrap();
    session.close().await.unwrap();

    assert_eq!(session_override.text, "session-model|session-effort|work");
    assert_eq!(turn_override.text, "turn-model|turn-effort|work");
}

#[cfg(unix)]
#[tokio::test]
async fn cli_session_uses_profile_cwd() {
    let dir = std::env::temp_dir().join(format!("awp-cwd-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let registry = registry_with(
        cli_spec(
            "pwd-agent",
            "sh",
            &["-c", "pwd", "{prompt}"],
            PromptChannel::ArgOnly,
        ),
        ProfilePatch {
            agent: Some("pwd-agent".to_owned()),
            cwd: Some(dir.clone()),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();
    let outcome = session.prompt("ignored".to_owned()).await.unwrap();
    session.close().await.unwrap();

    // /var vs /private/var: compare canonicalized (before removing the dir).
    let reported = std::fs::canonicalize(outcome.text.trim_end()).unwrap();
    let expected = std::fs::canonicalize(&dir).unwrap();
    let _ = std::fs::remove_dir(&dir);
    assert_eq!(reported, expected);
}

// ------------------------------ acp lane ---------------------------------

#[tokio::test]
async fn acp_first_turn_delivers_instructions_once() {
    let registry = registry_with(
        fixture_spec(),
        ProfilePatch {
            agent: Some("fixture".to_owned()),
            instructions: Some("PERSONA".to_owned()),
            instruction_mode: Some(InstructionMode::FirstTurn),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, mut rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();
    assert_eq!(session.session_id(), "fixture-session-1");

    session.prompt("one".to_owned()).await.unwrap();
    session.prompt("two".to_owned()).await.unwrap();
    session.close().await.unwrap();

    let mut echoed = String::new();
    while let Ok(event) = rx.try_recv() {
        if let AgentEvent::MessageChunk { text } = event {
            echoed.push_str(&text);
        }
    }
    assert!(
        echoed.contains("echo: PERSONA\n\none"),
        "first turn composed: {echoed:?}"
    );
    assert!(
        echoed.contains("echo: two"),
        "second turn verbatim: {echoed:?}"
    );
    assert!(!echoed.contains("PERSONA\n\nPERSONA"));
}

#[tokio::test]
async fn acp_every_turn_and_direct_prompt() {
    let registry = registry_with(
        fixture_spec(),
        ProfilePatch {
            agent: Some("fixture".to_owned()),
            instructions: Some("P".to_owned()),
            instruction_mode: Some(InstructionMode::EveryTurn),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, mut rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    session.prompt("a".to_owned()).await.unwrap();
    session.prompt("b".to_owned()).await.unwrap();
    // prompt_direct bypasses composition entirely.
    session
        .prompt_direct(TurnRequest {
            text: "raw".to_owned(),
            model: None,
            effort: None,
        })
        .await
        .unwrap();
    session.close().await.unwrap();

    let mut echoed = String::new();
    while let Ok(event) = rx.try_recv() {
        if let AgentEvent::MessageChunk { text } = event {
            echoed.push_str(&text);
        }
    }
    assert!(echoed.contains("echo: P\n\na"), "{echoed:?}");
    assert!(echoed.contains("echo: P\n\nb"), "{echoed:?}");
    assert!(echoed.contains("echo: raw"), "{echoed:?}");
}

#[tokio::test]
async fn acp_explicit_cancel_reaches_the_agent() {
    let registry = registry_with(
        fixture_spec(),
        ProfilePatch {
            agent: Some("fixture".to_owned()),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    let outcome = {
        let turn = session.prompt("wait-for-cancel".to_owned());
        tokio::pin!(turn);
        tokio::select! {
            outcome = &mut turn => panic!("turn finished before cancel: {outcome:?}"),
            _ = tokio::time::sleep(std::time::Duration::from_millis(500)) => {}
        }
        session.cancel().await.unwrap();
        turn.await.unwrap()
    };
    assert_eq!(outcome.stop_reason, "cancelled");
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn rpc_initial_cancellation_keeps_first_turn_guidance_pending() {
    let dir = TestDirectory::new("cancel-guidance");
    let script = dir.path().join("rpc_cancel.py");
    std::fs::write(
        &script,
        r#"
import json,sys
first=True
for line in sys.stdin:
    m=json.loads(line)
    if m.get('method')=='session/open': result={'sessionId':'fixture'}
    elif first:
        first=False
        result={'stopReason':'cancelled','text':''}
    else: result={'stopReason':'end_turn','text':m['params']['text']}
    print(json.dumps({'jsonrpc':'2.0','id':m['id'],'result':result}),flush=True)
"#,
    )
    .unwrap();
    let registry = registry_with(
        rpc_spec("fixture", &script, json!({}), json!({"text":"{prompt}"})),
        ProfilePatch {
            agent: Some("fixture".into()),
            instructions: Some("PERSONA".into()),
            instruction_mode: Some(InstructionMode::FirstTurn),
            ..Default::default()
        },
    );
    let (tx, _rx) = events();
    let session = registry
        .resolve("p")
        .unwrap()
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();
    assert_eq!(
        session
            .prompt("cancelled attempt")
            .await
            .unwrap()
            .stop_reason,
        "cancelled"
    );
    assert_eq!(session.composed_prompt("after"), "PERSONA\n\nafter");
    assert_eq!(
        session.prompt("after").await.unwrap().text,
        "PERSONA\n\nafter"
    );
    assert_eq!(session.prompt("later").await.unwrap().text, "later");
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn concurrent_turns_compose_guidance_after_the_previous_outcome() {
    for abandon_first_waiter in [false, true] {
        tokio::time::timeout(std::time::Duration::from_secs(10), async {
        let dir = TestDirectory::new("ordered-guidance");
        let script = dir.path().join("ordered.py");
        let barrier = dir.path().join("release");
        std::fs::write(&script, r#"
import json,sys,os
first=True
for line in sys.stdin:
    m=json.loads(line)
    if m.get('method')=='session/open': result={'sessionId':'fixture'}
    else:
        if first:
            first=False
            os.mkfifo('release')
            print(json.dumps({'jsonrpc':'2.0','method':'ready','params':{}}),flush=True)
            with open('release') as gate: gate.read()
        result={'stopReason':'end_turn','text':m['params']['text']}
    print(json.dumps({'jsonrpc':'2.0','id':m['id'],'result':result}),flush=True)
"#).unwrap();
        let registry = registry_with(rpc_spec("fixture", &script, json!({}), json!({"text":"{prompt}"})),
            ProfilePatch { agent: Some("fixture".into()), instructions: Some("PERSONA".into()),
                instruction_mode: Some(InstructionMode::FirstTurn), ..Default::default() });
        let (tx, mut rx) = events();
        let session = registry.resolve("p").unwrap().open_session(SessionRequest::default(), tx).await.unwrap();
        let mut first = Some(Box::pin(session.prompt("first")));
        loop {
            tokio::select! {
                result = first.as_mut().unwrap() => panic!("first prompt crossed the fixture barrier: {result:?}"),
                event = rx.recv() => if matches!(event, Some(AgentEvent::Raw { ref raw }) if raw["method"]=="ready") { break; }
            }
        }
        if abandon_first_waiter { drop(first.take()); }
        let second = session.prompt("second");
        tokio::pin!(second);
        // Poll exactly once while the first turn is unresolved. The old
        // implementation composed PERSONA here, before A4AA serialized I/O.
        tokio::select! { biased;
            result = &mut second => panic!("second prompt overtook first: {result:?}"),
            _ = std::future::ready(()) => {}
        }
        tokio::task::spawn_blocking(move || std::fs::write(barrier, b"release")).await.unwrap().unwrap();
        if let Some(first) = first { assert_eq!(first.await.unwrap().text, "PERSONA\n\nfirst"); }
        assert_eq!(second.await.unwrap().text, "second");
    }).await.expect("concurrent turn ordering did not finish");
    }
}
