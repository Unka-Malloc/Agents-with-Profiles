//! End-to-end profile semantics: instruction composition, model/effort
//! defaults, session cwd and cancellation — through real spawned lanes.
//!
//! The `acp` cases drive `awp fixture-agent`; the `cli` cases use ubiquitous
//! Unix tools, and the `rpc` cases use a small local JSON-RPC child fixture.

#[cfg(unix)]
use a4aa::SessionConfig;
use awp::{
    AgentEvent, AgentSpec, CommandSpec, InstructionMode, LaneSpec, ProfilePatch, Registry,
    RegistryDocument, SessionRequest, TurnRequest,
};
#[cfg(unix)]
use awp::{EnvPolicy, EnvValue, PromptChannel, RpcCallSpec, RpcLaneSpec};
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

mod common;
#[cfg(unix)]
use common::cli_spec;

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
        seatbelted_permission: None,
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
            auth_key: None,
            enforced_permission: None,
        },
        command: CommandSpec {
            program: env!("CARGO_BIN_EXE_awp").to_owned(),
            args: vec!["fixture-agent".to_owned()],
            cwd: None,
            env: Default::default(),
        },
        max_output_bytes: None,
        seatbelted_permission: None,
    }
}

fn registry_with(spec: AgentSpec, patch: ProfilePatch) -> Registry {
    let mut document = RegistryDocument::default();
    document.agents.insert(spec.id.clone(), spec.into());
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
    assert_eq!(session.session_id().await, "fixture-session-1");

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

// --------------------------- model fallback -----------------------------

/// A fixture spec whose agent advertises a select `model` configOption and
/// answers prompts under `quota` models with a provider rate-limit error.
#[cfg(unix)]
fn quota_fixture_spec(models: &str, quota: &str) -> AgentSpec {
    let mut spec = fixture_spec();
    spec.command.env = awp::EnvPolicy::Sanitized {
        inherit: vec![],
        set: [
            (
                "AWP_FIXTURE_MODELS".to_owned(),
                EnvValue::Literal(models.to_owned()),
            ),
            (
                "AWP_FIXTURE_QUOTA".to_owned(),
                EnvValue::Literal(quota.to_owned()),
            ),
        ]
        .into_iter()
        .collect(),
    };
    spec
}

#[cfg(unix)]
#[tokio::test]
async fn acp_model_chain_advances_on_quota_failure() {
    let registry = registry_with(
        quota_fixture_spec("prime,fallback", "prime"),
        ProfilePatch {
            agent: Some("fixture".to_owned()),
            model: Some("prime".to_owned()),
            models: Some(vec!["fallback".to_owned()]),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();
    assert_eq!(session.active_model().as_deref(), Some("prime"));

    // The quota-rejected turn is retried on the same session with the next
    // chain entry applied through session/set_config_option.
    let outcome = session.prompt("work".to_owned()).await.unwrap();
    assert_eq!(outcome.stop_reason, "end_turn");
    assert_eq!(session.active_model().as_deref(), Some("fallback"));

    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn acp_exhausted_model_chain_propagates_the_error() {
    let registry = registry_with(
        quota_fixture_spec("only", "only"),
        ProfilePatch {
            agent: Some("fixture".to_owned()),
            model: Some("only".to_owned()),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    let error = session.prompt("work".to_owned()).await.unwrap_err();
    assert_eq!(error.code, "agent_quota_exhausted");
    session.close().await.unwrap();
}

// ---------------------------- lane fallback -----------------------------

/// One document whose agent declares two lanes in fallback order: the
/// primary first, one alternate.
#[cfg(unix)]
fn registry_with_lanes(primary: AgentSpec, alternate: AgentSpec, patch: ProfilePatch) -> Registry {
    let mut document = RegistryDocument::default();
    document.agents.insert(
        primary.id.clone(),
        awp::AgentEntry {
            spec: primary,
            alternates: vec![alternate],
        },
    );
    document.profiles.insert("p".to_owned(), patch);
    Registry::from_document(document).unwrap()
}

#[cfg(unix)]
#[tokio::test]
async fn dead_primary_lane_falls_back_to_alternate() {
    // `false` exits 1 on every prompt — a transport-level failure no model
    // change can fix. The alternate echo lane must serve the same turn.
    let registry = registry_with_lanes(
        cli_spec("dead", "false", &[], PromptChannel::ArgOnly),
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("dead".to_owned()),
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
    assert_eq!(outcome.text.trim_end(), "alt|work");
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn mid_turn_lane_death_replays_on_alternate_with_instructions() {
    // The primary works exactly once: a marker file makes the second spawn
    // fail. The replayed turn on the alternate lane must re-compose the
    // profile's instructions — the fresh session never heard them.
    let dir = TestDirectory::new("lane-death");
    let marker = dir.path().join("primary-ran");
    let primary = cli_spec(
        "once",
        "sh",
        &[
            "-c",
            &format!(
                "test ! -f {0} && touch {0} && echo primary:$1 || exit 1",
                marker.display()
            ),
            "sh",
            "{prompt}",
        ],
        PromptChannel::ArgOnly,
    );
    let alternate = cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly);
    let registry = registry_with_lanes(
        primary,
        alternate,
        ProfilePatch {
            agent: Some("once".to_owned()),
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

    let normalized = |text: &str| text.split_whitespace().collect::<Vec<_>>().join(" ");
    let first = session.prompt("one".to_owned()).await.unwrap();
    assert_eq!(normalized(&first.text), "primary:PERSONA one");

    let second = session.prompt("two".to_owned()).await.unwrap();
    assert_eq!(normalized(&second.text), "alt|PERSONA two");
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn exhausted_model_chain_never_reaches_alternate_lane() {
    // Provider-side exhaustion is account-level: every lane answers from
    // the same account, so the chain end propagates to the caller's next
    // profile rather than reopening on the alternate.
    let registry = registry_with_lanes(
        quota_fixture_spec("only", "only"),
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("fixture".to_owned()),
            model: Some("only".to_owned()),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    let error = session.prompt("work".to_owned()).await.unwrap_err();
    assert_eq!(error.code, "agent_quota_exhausted");
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn cli_error_outcome_with_output_does_not_replay_on_alternate_lane() {
    // The primary exits non-zero after printing a turn. Its output is
    // already paid and visible in the event stream, so the alternate lane
    // must not run the same prompt again and replace that result.
    let registry = registry_with_lanes(
        cli_spec(
            "paid",
            "sh",
            &["-c", "exec 0<&-; echo paid-output; exit 1"],
            PromptChannel::Stdin,
        ),
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("paid".to_owned()),
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
    assert_eq!(outcome.stop_reason, "error");
    assert!(
        outcome.text.contains("paid-output"),
        "completed output is retained: {outcome:?}"
    );
    assert!(!outcome.text.contains("alt|"), "alternate did not replay");
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn cli_truncated_error_outcome_does_not_replay_on_alternate_lane() {
    // The retained cap drops the entire turn text, yet truncation itself
    // proves the primary produced output: replaying the prompt on the
    // alternate would duplicate a turn that already ran.
    let mut primary = cli_spec(
        "paid",
        "sh",
        &["-c", "exec 0<&-; echo paid-output; exit 1"],
        PromptChannel::Stdin,
    );
    primary.max_output_bytes = Some(0);
    let registry = registry_with_lanes(
        primary,
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("paid".to_owned()),
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
    assert_eq!(outcome.stop_reason, "error");
    assert!(outcome.truncated, "dropped output is still marked");
    assert!(!outcome.text.contains("alt|"), "alternate did not replay");
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn cli_ndjson_progress_with_empty_result_does_not_replay_on_alternate_lane() {
    // A json_lines lane delivers every record as an event; an empty
    // `result` record then presents an outcome with no text and no usage.
    // The emitted stream is still proof the paid turn ran — the alternate
    // lane must not replay the prompt.
    let mut primary = cli_spec(
        "paid",
        "sh",
        &[
            "-c",
            "exec 0<&-; \
             echo '{\"type\":\"progress\",\"message\":\"working\"}'; \
             echo '{\"type\":\"result\",\"result\":\"\"}'; \
             exit 1",
        ],
        PromptChannel::Stdin,
    );
    let LaneSpec::Cli(lane) = &mut primary.lane else {
        panic!("cli_spec builds a cli lane")
    };
    lane.json_lines = true;
    lane.result_selector = Some("result".to_owned());
    lane.result_path = Some("result".to_owned());
    let registry = registry_with_lanes(
        primary,
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("paid".to_owned()),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, mut rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    let outcome = session.prompt("work".to_owned()).await.unwrap();
    assert_eq!(outcome.stop_reason, "error");
    assert!(
        outcome.text.trim().is_empty(),
        "the empty result record hides the stream: {outcome:?}"
    );
    let mut forwarded = Vec::new();
    while let Ok(event) = rx.try_recv() {
        forwarded.push(event);
    }
    assert!(
        forwarded
            .iter()
            .any(|event| matches!(event, AgentEvent::Raw { .. })),
        "the lane's emitted records reached the caller: {forwarded:?}"
    );
    assert!(
        !forwarded.iter().any(
            |event| matches!(event, AgentEvent::MessageChunk { text } if text.contains("alt|"))
        ),
        "alternate did not replay"
    );
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn uncertain_prompt_failure_does_not_replay_on_alternate_lane() {
    // The primary prints a chunk and then reports a usage limit. The
    // refusal is uncertain-effect: replaying it on the alternate would
    // duplicate a turn that may already have been billed.
    let registry = registry_with_lanes(
        cli_spec(
            "uncertain",
            "sh",
            &[
                "-c",
                "exec 0<&-; echo partial; echo 'usage limit reached' >&2; exit 1",
            ],
            PromptChannel::Stdin,
        ),
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("uncertain".to_owned()),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    let error = session.prompt("work".to_owned()).await.unwrap_err();
    assert_eq!(error.code, "agent_quota_exhausted");
    assert!(error.uncertain_effect);
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn uncertain_prompt_failure_does_not_advance_the_model_chain() {
    // A quota refusal after streamed output is uncertain-effect even when
    // the model chain has another entry. Advancing it would resubmit the
    // full prompt and duplicate the paid turn.
    let mut primary = quota_fixture_spec("prime,fallback", "prime");
    let EnvPolicy::Sanitized { set, .. } = &mut primary.command.env else {
        panic!("quota fixture uses a sanitized environment");
    };
    set.insert(
        "AWP_FIXTURE_QUOTA_AFTER_OUTPUT".to_owned(),
        EnvValue::Literal("1".to_owned()),
    );
    let registry = registry_with_lanes(
        primary,
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("fixture".to_owned()),
            model: Some("prime".to_owned()),
            models: Some(vec!["fallback".to_owned()]),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    let error = session.prompt("work".to_owned()).await.unwrap_err();
    assert_eq!(error.code, "agent_quota_exhausted");
    assert!(error.uncertain_effect);
    assert_eq!(session.active_model().as_deref(), Some("prime"));
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn all_lanes_dead_propagates_the_last_error() {
    let registry = registry_with_lanes(
        cli_spec("dead", "false", &[], PromptChannel::ArgOnly),
        cli_spec("dead2", "false", &[], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("dead".to_owned()),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    session.prompt("work".to_owned()).await.unwrap_err();
    session.close().await.unwrap();
}

/// An agent that dies when the `AWP_KILL_METHOD` method arrives — the
/// peer then reports a transport failure to whichever call was in flight,
/// the same `rpc/call` surface before and after prompt dispatch.
#[cfg(unix)]
const RPC_DIES_ON_METHOD: &str = r#"
import json, os, sys

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

kill = os.environ.get("AWP_KILL_METHOD", "")
for line in sys.stdin:
    try:
        request = json.loads(line)
    except Exception:
        continue
    method = request.get("method")
    request_id = request.get("id")
    if method == kill:
        sys.stdout.flush()
        os._exit(1)
    if method == "session/open":
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "primary|{}".format(
                  (request.get("params") or {}).get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

/// An rpc lane whose `session/select` runs as a preTurn call and whose
/// child exits when `kill_method` arrives.
#[cfg(unix)]
fn rpc_kill_spec(id: &str, script: &Path, kill_method: &str) -> AgentSpec {
    let mut spec = rpc_spec(id, script, json!({}), json!({"text": "{prompt}"}));
    let LaneSpec::Rpc(lane) = &mut spec.lane else {
        panic!("rpc_spec builds an rpc lane")
    };
    lane.pre_turn = vec![RpcCallSpec {
        method: "session/select".to_owned(),
        params: json!({"model": "selected"}),
    }];
    spec.command.env = EnvPolicy::Sanitized {
        inherit: vec![],
        set: [(
            "AWP_KILL_METHOD".to_owned(),
            EnvValue::Literal(kill_method.to_owned()),
        )]
        .into_iter()
        .collect(),
    };
    spec
}

#[cfg(unix)]
#[tokio::test]
async fn rpc_pre_turn_disconnect_falls_back_to_alternate() {
    // The peer dies while the preTurn selector call is in flight: the
    // `rpc/call` error carries uncertain effect, yet the prompt itself was
    // never dispatched — the alternate lane must serve it.
    let dir = TestDirectory::new("rpc-pre-dispatch");
    let script = dir.path().join("rpc_kill.py");
    std::fs::write(&script, RPC_DIES_ON_METHOD).unwrap();
    let registry = registry_with_lanes(
        rpc_kill_spec("fragile", &script, "session/select"),
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("fragile".to_owned()),
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
    assert_eq!(outcome.text.trim_end(), "alt|work");
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn rpc_post_dispatch_disconnect_never_replays_on_alternate() {
    // The peer dies while the dispatched prompt is in flight: the same
    // uncertain `rpc/call` surface now covers a possibly billed turn —
    // the alternate must not replay it and the error must propagate.
    let dir = TestDirectory::new("rpc-post-dispatch");
    let script = dir.path().join("rpc_kill.py");
    std::fs::write(&script, RPC_DIES_ON_METHOD).unwrap();
    let registry = registry_with_lanes(
        rpc_kill_spec("fragile", &script, "session/prompt"),
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("fragile".to_owned()),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    let error = session.prompt("work".to_owned()).await.unwrap_err();
    assert_eq!(error.code, "agent_transport_failed");
    let _ = session.close().await;
}

#[cfg(unix)]
#[tokio::test]
async fn cli_whitespace_error_outcome_does_not_replay_on_alternate_lane() {
    // A json_lines lane drops non-record lines from the event stream, yet
    // the retained text still carries them: whitespace output is evidence
    // the paid turn ran — the alternate must not replay the prompt.
    let mut primary = cli_spec(
        "paid",
        "sh",
        &["-c", "exec 0<&-; printf ' \\n'; exit 1"],
        PromptChannel::Stdin,
    );
    let LaneSpec::Cli(lane) = &mut primary.lane else {
        panic!("cli_spec builds a cli lane")
    };
    lane.json_lines = true;
    let registry = registry_with_lanes(
        primary,
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("paid".to_owned()),
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
    assert_eq!(outcome.stop_reason, "error");
    assert!(
        !outcome.text.contains("alt|"),
        "alternate did not replay: {outcome:?}"
    );
    session.close().await.unwrap();
}

/// A primary that answers `session/prompt` with a notification followed by
/// a remote error — emitted evidence plus a lane failure — so the next
/// lane's replay decision must stand on its own attempt, not the dead
/// lane's events.
#[cfg(unix)]
const RPC_EMITS_THEN_REFUSES: &str = r#"
import json, sys

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

for line in sys.stdin:
    try:
        request = json.loads(line)
    except Exception:
        continue
    method = request.get("method")
    request_id = request.get("id")
    if method == "session/open":
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        send({"jsonrpc": "2.0", "method": "progress",
              "params": {"note": "ran"}})
        send({"jsonrpc": "2.0", "id": request_id,
              "error": {"code": -32000, "message": "boom"}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn lane_output_evidence_is_per_attempt() {
    // Lane one emits an event then fails its turn; lane two fails with no
    // output of its own. The first lane's emitted evidence must not
    // suppress the second lane's fallback — evidence is per attempt.
    let dir = TestDirectory::new("per-attempt");
    let script = dir.path().join("rpc_refuse.py");
    std::fs::write(&script, RPC_EMITS_THEN_REFUSES).unwrap();
    let primary = rpc_spec("emitting", &script, json!({}), json!({"text": "{prompt}"}));
    let mut document = RegistryDocument::default();
    document.agents.insert(
        primary.id.clone(),
        awp::AgentEntry {
            spec: primary,
            alternates: vec![
                cli_spec("silent", "sh", &["-c", "exit 1"], PromptChannel::ArgOnly),
                cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
            ],
        },
    );
    document.profiles.insert(
        "p".to_owned(),
        ProfilePatch {
            agent: Some("emitting".to_owned()),
            ..Default::default()
        },
    );
    let registry = Registry::from_document(document).unwrap();
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    let outcome = session.prompt("work".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "alt|work");
    session.close().await.unwrap();
}

/// A `BeforeTurn` hook that refuses every turn with an uncertain error —
/// the reservation may already stand, so the failure must propagate.
#[cfg(unix)]
struct RefusingHook;

#[cfg(unix)]
#[async_trait::async_trait]
impl awp::BeforeTurn for RefusingHook {
    async fn run(&self) -> awp::Result<()> {
        Err(awp::AdapterError::uncertain(
            "reservation_unavailable",
            "session/prompt",
        ))
    }
}

/// An rpc agent that logs every method it receives: the dispatch check
/// must not send a probe prompt, so `session/prompt` may never appear in
/// the log when the turn never reached the lane.
#[cfg(unix)]
const RPC_LOG_METHODS: &str = r#"
import json, os, sys

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

log = open(os.environ["AWP_CALL_LOG"], "a")
for line in sys.stdin:
    try:
        request = json.loads(line)
    except Exception:
        continue
    method = request.get("method")
    request_id = request.get("id")
    if method:
        log.write(method + "\n")
        log.flush()
    if method == "session/open":
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "primary|{}".format(
                  (request.get("params") or {}).get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
fn rpc_logging_spec(id: &str, script: &Path, log: &Path) -> AgentSpec {
    let mut spec = rpc_spec(id, script, json!({}), json!({"text": "{prompt}"}));
    spec.command.env = EnvPolicy::Sanitized {
        inherit: vec![],
        set: [(
            "AWP_CALL_LOG".to_owned(),
            EnvValue::Literal(log.display().to_string()),
        )]
        .into_iter()
        .collect(),
    };
    spec
}

#[cfg(unix)]
#[tokio::test]
async fn before_turn_refusal_never_probes_or_replays_the_lane() {
    // The reservation hook refuses with an uncertain error while the rpc
    // peer is healthy. Probing the failed session with a real prompt
    // would bypass the hook and dispatch an empty paid turn; the refusal
    // must propagate before any lane is invoked instead.
    let dir = TestDirectory::new("reservation-refusal");
    let script = dir.path().join("rpc_log.py");
    std::fs::write(&script, RPC_LOG_METHODS).unwrap();
    let log = dir.path().join("calls.log");
    let registry = registry_with_lanes(
        rpc_logging_spec("reserved", &script, &log),
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("reserved".to_owned()),
            ..Default::default()
        },
    );
    let agent = registry
        .resolve("p")
        .unwrap()
        .before_turn(std::sync::Arc::new(RefusingHook));
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    let error = session.prompt("work".to_owned()).await.unwrap_err();
    assert_eq!(error.code, "reservation_unavailable");
    let calls = std::fs::read_to_string(&log).unwrap_or_default();
    assert!(
        !calls.lines().any(|method| method == "session/prompt"),
        "no probe prompt reached the wire: {calls}"
    );
    session.close().await.unwrap();
}

/// An rpc lane that announces a session-open notification and then
/// answers its prompt with a silent error — queued open traffic is not
/// the turn's output, so the next lane must still be tried.
#[cfg(unix)]
const RPC_OPENS_THEN_FAILS: &str = r#"
import json, sys

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

for line in sys.stdin:
    try:
        request = json.loads(line)
    except Exception:
        continue
    method = request.get("method")
    request_id = request.get("id")
    if method == "session/open":
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
        send({"jsonrpc": "2.0", "method": "session/ready",
              "params": {"ready": True}})
    elif method == "session/prompt":
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"stopReason": "error", "text": ""}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn session_open_events_on_a_new_lane_do_not_suppress_fallback() {
    // Lane two opens with a session-open notification queued on its tap,
    // then returns a silent error. That notification is not the active
    // turn's output — the healthy third lane must serve the prompt.
    let dir = TestDirectory::new("open-events");
    let script = dir.path().join("rpc_open.py");
    std::fs::write(&script, RPC_OPENS_THEN_FAILS).unwrap();
    let primary = cli_spec("dead", "false", &[], PromptChannel::ArgOnly);
    let mut document = RegistryDocument::default();
    document.agents.insert(
        primary.id.clone(),
        awp::AgentEntry {
            spec: primary,
            alternates: vec![
                rpc_spec("opening", &script, json!({}), json!({"text": "{prompt}"})),
                cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
            ],
        },
    );
    document.profiles.insert(
        "p".to_owned(),
        ProfilePatch {
            agent: Some("dead".to_owned()),
            ..Default::default()
        },
    );
    let registry = Registry::from_document(document).unwrap();
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    let outcome = session.prompt("work".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "alt|work");
    session.close().await.unwrap();
}

/// A primary that streams one text chunk and then answers the prompt
/// with a generic remote error: the turn produced output already, so the
/// failure must propagate instead of replaying on the alternate.
#[cfg(unix)]
const RPC_STREAMS_THEN_REFUSES: &str = r#"
import json, sys

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

for line in sys.stdin:
    try:
        request = json.loads(line)
    except Exception:
        continue
    method = request.get("method")
    request_id = request.get("id")
    if method == "session/open":
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        send({"jsonrpc": "2.0", "method": "delta",
              "params": {"text": "paid-chunk"}})
        send({"jsonrpc": "2.0", "id": request_id,
              "error": {"code": -32000, "message": "boom"}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_error_after_streamed_text_does_not_replay_on_alternate() {
    // A streamed chunk is evidence the paid turn ran; replaying the
    // generic remote error on the alternate would duplicate it.
    let dir = TestDirectory::new("streamed-refusal");
    let script = dir.path().join("rpc_stream.py");
    std::fs::write(&script, RPC_STREAMS_THEN_REFUSES).unwrap();
    let mut primary = rpc_spec("streaming", &script, json!({}), json!({"text": "{prompt}"}));
    let LaneSpec::Rpc(lane) = &mut primary.lane else {
        panic!("rpc_spec builds an rpc lane")
    };
    lane.text_event_method = Some("delta".to_owned());
    let registry = registry_with_lanes(
        primary,
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("streaming".to_owned()),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    let error = session.prompt("work".to_owned()).await.unwrap_err();
    assert_eq!(error.code, "agent_remote_error");
    session.close().await.unwrap();
}

/// An rpc peer that answers `session/open`, closes its own stdin and then
/// sleeps with stdout still open: the prompt's write fails on the
/// transport while the reader stays alive, so the failure is
/// deterministically `writer_failed` rather than a racing `peer_closed`.
#[cfg(unix)]
const RPC_CLOSES_STDIN: &str = r#"
import json, os, sys, time

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

request = json.loads(sys.stdin.readline())
send({"jsonrpc": "2.0", "id": request["id"],
      "result": {"sessionId": "awp-rpc-session"}})
os.close(0)
time.sleep(30)
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_prompt_write_failure_falls_back_to_alternate() {
    // The transport reports `writer_failed` and the wire journal shows the
    // prompt was the last frame the writer attempted — a confirmed
    // undelivered dispatch the alternate lane may replay.
    let dir = TestDirectory::new("rpc-write-failed");
    let script = dir.path().join("rpc_close_stdin.py");
    std::fs::write(&script, RPC_CLOSES_STDIN).unwrap();
    let registry = registry_with_lanes(
        rpc_spec(
            "broken-pipe",
            &script,
            json!({}),
            json!({"text": "{prompt}"}),
        ),
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("broken-pipe".to_owned()),
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
    assert_eq!(outcome.text.trim_end(), "alt|work");
    session.close().await.unwrap();
}

/// A `BeforeTurn` hook that counts its invocations and returns a fixed
/// verdict — the count proves the reservation runs once per logical turn.
#[cfg(unix)]
struct CountingHook {
    calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    error: Option<awp::AdapterError>,
}

#[cfg(unix)]
#[async_trait::async_trait]
impl awp::BeforeTurn for CountingHook {
    async fn run(&self) -> awp::Result<()> {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        match &self.error {
            Some(error) => Err(error.clone()),
            None => Ok(()),
        }
    }
}

#[cfg(unix)]
#[tokio::test]
async fn before_turn_permanent_refusal_never_reprobes_or_replays() {
    // A permanent refusal is still the reservation's verdict: it must
    // propagate before the next lane invokes the hook for the same turn.
    let dir = TestDirectory::new("reservation-permanent");
    let script = dir.path().join("rpc_log.py");
    std::fs::write(&script, RPC_LOG_METHODS).unwrap();
    let log = dir.path().join("calls.log");
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let registry = registry_with_lanes(
        rpc_logging_spec("reserved", &script, &log),
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("reserved".to_owned()),
            ..Default::default()
        },
    );
    let agent = registry
        .resolve("p")
        .unwrap()
        .before_turn(std::sync::Arc::new(CountingHook {
            calls: calls.clone(),
            error: Some(awp::AdapterError::permanent(
                "reservation_denied",
                "session/prompt",
            )),
        }));
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    let error = session.prompt("work".to_owned()).await.unwrap_err();
    assert_eq!(error.code, "reservation_denied");
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    let calls_log = std::fs::read_to_string(&log).unwrap_or_default();
    assert!(
        !calls_log.lines().any(|method| method == "session/prompt"),
        "no prompt reached the wire: {calls_log}"
    );
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn before_turn_quota_refusal_does_not_retry_the_model_chain() {
    // A refusal the classifier would read as provider-side quota is still
    // the reservation's verdict — it propagates before the chain advances
    // and the hook runs for the same turn a second time.
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let registry = registry_with_lanes(
        quota_fixture_spec("prime,fallback", ""),
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("fixture".to_owned()),
            model: Some("prime".to_owned()),
            models: Some(vec!["fallback".to_owned()]),
            ..Default::default()
        },
    );
    let agent = registry
        .resolve("p")
        .unwrap()
        .before_turn(std::sync::Arc::new(CountingHook {
            calls: calls.clone(),
            error: Some(awp::AdapterError::retryable(
                "agent_quota_exhausted",
                "session/prompt",
            )),
        }));
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    let error = session.prompt("work".to_owned()).await.unwrap_err();
    assert_eq!(error.code, "agent_quota_exhausted");
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn granted_reservation_is_reused_across_lane_fallback() {
    // The hook grants once; the first lane dies during its pre-turn call,
    // before the prompt is dispatched. The replayed turn on the alternate
    // reuses that grant — invoking the hook again could double-book the
    // reservation for one turn.
    let dir = TestDirectory::new("reservation-reuse");
    let script = dir.path().join("rpc_kill.py");
    std::fs::write(&script, RPC_DIES_ON_METHOD).unwrap();
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let registry = registry_with_lanes(
        rpc_kill_spec("fragile", &script, "session/select"),
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("fragile".to_owned()),
            ..Default::default()
        },
    );
    let agent = registry
        .resolve("p")
        .unwrap()
        .before_turn(std::sync::Arc::new(CountingHook {
            calls: calls.clone(),
            error: None,
        }));
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    let outcome = session.prompt("work".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "alt|work");
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    session.close().await.unwrap();
}

/// An rpc agent that reports usage at a configured carrier and then
/// answers the prompt with a generic remote error — the usage proves the
/// paid turn ran, so the failure must propagate.
#[cfg(unix)]
const RPC_USAGE_PATH_THEN_REFUSES: &str = r#"
import json, sys

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

for line in sys.stdin:
    try:
        request = json.loads(line)
    except Exception:
        continue
    method = request.get("method")
    request_id = request.get("id")
    if method == "session/open":
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        send({"jsonrpc": "2.0", "method": "meter",
              "params": {"bill": {"tokens": {"output_tokens": 7}}}})
        send({"jsonrpc": "2.0", "id": request_id,
              "error": {"code": -32000, "message": "boom"}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_usage_at_configured_path_does_not_replay_on_alternate() {
    // Usage at the lane's declared `usage_path` is billable evidence; a
    // generic error after it cannot replay the turn on the alternate.
    let dir = TestDirectory::new("usage-path");
    let script = dir.path().join("rpc_meter.py");
    std::fs::write(&script, RPC_USAGE_PATH_THEN_REFUSES).unwrap();
    let mut primary = rpc_spec("metered", &script, json!({}), json!({"text": "{prompt}"}));
    let LaneSpec::Rpc(lane) = &mut primary.lane else {
        panic!("rpc_spec builds an rpc lane")
    };
    lane.usage_path = Some("bill.tokens".to_owned());
    let registry = registry_with_lanes(
        primary,
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("metered".to_owned()),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    let error = session.prompt("work".to_owned()).await.unwrap_err();
    assert_eq!(error.code, "agent_remote_error");
    session.close().await.unwrap();
}

/// An rpc peer that answers `session/open`, closes its own stdout and
/// keeps reading stdin — a peer that stopped answering yet still consumes
/// whatever the lane's writer sends.
#[cfg(unix)]
const RPC_CLOSES_STDOUT: &str = r#"
import json, os, sys

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

log = open(os.environ["AWP_CALL_LOG"], "a")
request = json.loads(sys.stdin.readline())
if request.get("method"):
    log.write(request["method"] + "\n")
    log.flush()
send({"jsonrpc": "2.0", "id": request["id"],
      "result": {"sessionId": "awp-rpc-session"}})
os.close(1)
for line in sys.stdin:
    try:
        request = json.loads(line)
    except Exception:
        continue
    if request.get("method"):
        log.write(request["method"] + "\n")
        log.flush()
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_queued_prompt_peer_closed_confirms_before_replay() {
    // The peer closes its answer stream while the pre-turn call is in
    // flight but keeps reading stdin: `peer_closed` reaches the turn
    // while the lane's independent writer may still hold queued frames —
    // a peer that stopped answering could still receive a late write.
    // The turn must stop the transport and confirm the prompt never left
    // before the alternate replays it.
    let dir = TestDirectory::new("rpc-queued-prompt");
    let script = dir.path().join("rpc_closes_stdout.py");
    std::fs::write(&script, RPC_CLOSES_STDOUT).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec("closing", &script, json!({}), json!({"text": "{prompt}"}));
    let LaneSpec::Rpc(lane) = &mut primary.lane else {
        panic!("rpc_spec builds an rpc lane")
    };
    lane.pre_turn = vec![RpcCallSpec {
        method: "session/select".to_owned(),
        params: json!({"model": "selected"}),
    }];
    primary.command.env = EnvPolicy::Sanitized {
        inherit: vec![],
        set: [(
            "AWP_CALL_LOG".to_owned(),
            EnvValue::Literal(log.display().to_string()),
        )]
        .into_iter()
        .collect(),
    };
    let registry = registry_with_lanes(
        primary,
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("closing".to_owned()),
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
    assert_eq!(outcome.text.trim_end(), "alt|work");
    let calls_log = std::fs::read_to_string(&log).unwrap_or_default();
    assert!(
        !calls_log.lines().any(|method| method == "session/prompt"),
        "queued prompt reached the closed lane: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer that refuses prompts for the `prime` model with a quota
/// error and serves every other model — the refusal must advance the
/// session's model chain instead of jumping lanes or giving up.
#[cfg(unix)]
const RPC_QUOTA_ON_PRIME: &str = r#"
import json, sys

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

session_model = None
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
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        model = params.get("model") or session_model
        if model == "prime":
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
        else:
            send({"jsonrpc": "2.0", "id": request_id,
                  "result": {"text": "served|{}|{}".format(
                      model or "unset", params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_quota_refusal_advances_the_model_chain() {
    // A4AA reports the rpc prompt's explicit quota response as
    // `agent_quota_exhausted` at `rpc/call` — a prompt-admission refusal
    // that must advance the model chain on the same session, not skip to
    // the alternate lane or give up.
    let dir = TestDirectory::new("rpc-quota-chain");
    let script = dir.path().join("rpc_quota.py");
    std::fs::write(&script, RPC_QUOTA_ON_PRIME).unwrap();
    let registry = registry_with_lanes(
        rpc_spec(
            "quota-rpc",
            &script,
            json!({}),
            json!({"text": "{prompt}", "model": "{model}"}),
        ),
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("quota-rpc".to_owned()),
            model: Some("prime".to_owned()),
            models: Some(vec!["fallback".to_owned()]),
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
    assert_eq!(outcome.text.trim_end(), "served|fallback|work");
    assert_eq!(session.active_model().as_deref(), Some("fallback"));

    // The accepted model is the session's default now: a second default
    // prompt must be served by the fallback, not return to the entry the
    // provider just exhausted.
    let outcome = session.prompt("more".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "served|fallback|more");
    session.close().await.unwrap();
}

/// An rpc peer that binds its model at `session/open` and has no per-turn
/// model surface: the chain's retry must re-establish the session with
/// the accepted entry rather than being rejected before dispatch.
#[cfg(unix)]
const RPC_QUOTA_SESSION_MODEL: &str = r#"
import json, os, sys

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

log = open(os.environ["AWP_CALL_LOG"], "a")
session_model = None
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
        log.write("session/open|{}\n".format(session_model))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        model = params.get("model") or session_model
        log.write("session/prompt|{}\n".format(model))
        log.flush()
        if model == "prime":
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
        else:
            send({"jsonrpc": "2.0", "id": request_id,
                  "result": {"text": "served|{}|{}".format(
                      model or "unset", params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_session_model_retry_reopens_the_session() {
    // The lane's only model surface is the session-open call: a quota
    // refusal on `prime` cannot carry the next chain entry on the turn —
    // A4AA would reject the request selector before dispatch. The retry
    // re-opens this lane's session bound to `fallback`, and the fresh
    // session serves the prompt.
    let dir = TestDirectory::new("rpc-session-model");
    let script = dir.path().join("rpc_session_model.py");
    std::fs::write(&script, RPC_QUOTA_SESSION_MODEL).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "session-model",
        &script,
        json!({"model": "{model}"}),
        json!({"text": "{prompt}"}),
    );
    primary.command.env = EnvPolicy::Sanitized {
        inherit: vec![],
        set: [(
            "AWP_CALL_LOG".to_owned(),
            EnvValue::Literal(log.display().to_string()),
        )]
        .into_iter()
        .collect(),
    };
    let registry = registry_with_lanes(
        primary,
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("session-model".to_owned()),
            model: Some("prime".to_owned()),
            models: Some(vec!["fallback".to_owned()]),
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
    assert_eq!(outcome.text.trim_end(), "served|fallback|work");
    assert_eq!(session.active_model().as_deref(), Some("fallback"));
    let calls_log = std::fs::read_to_string(&log).unwrap_or_default();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "session/open|prime",
            "session/prompt|prime",
            "session/open|fallback",
            "session/prompt|fallback",
        ],
        "the retry re-established the session with the accepted model: {calls_log}"
    );

    // A second default turn stays on the model bound at the re-open.
    let outcome = session.prompt("more".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "served|fallback|more");
    session.close().await.unwrap();
}

/// An rpc peer whose `session/open` itself refuses an exhausted model:
/// the session-bound chain keeps walking — a per-turn surface would
/// have advanced to the next entry the same way.
#[cfg(unix)]
const RPC_QUOTA_SESSION_OPEN_MODEL: &str = r#"
import json, os, sys

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

log = open(os.environ["AWP_CALL_LOG"], "a")
session_model = None
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
        log.write("session/open|{}\n".format(session_model))
        log.flush()
        if session_model == "fallback":
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
            continue
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        model = params.get("model") or session_model
        log.write("session/prompt|{}\n".format(model))
        log.flush()
        if model == "prime":
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
        else:
            send({"jsonrpc": "2.0", "id": request_id,
                  "result": {"text": "served|{}|{}".format(
                      model or "unset", params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_session_model_retry_walks_the_chain_past_a_refused_open() {
    // `fallback` is refused at session establishment, not at prompt: the
    // session-bound retry keeps the lane and binds the next chain entry
    // instead of condemning the turn — `tertiary` serves it.
    let dir = TestDirectory::new("rpc-session-open-model");
    let script = dir.path().join("rpc_session_open_model.py");
    std::fs::write(&script, RPC_QUOTA_SESSION_OPEN_MODEL).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "session-open-model",
        &script,
        json!({"model": "{model}"}),
        json!({"text": "{prompt}"}),
    );
    primary.command.env = EnvPolicy::Sanitized {
        inherit: vec![],
        set: [(
            "AWP_CALL_LOG".to_owned(),
            EnvValue::Literal(log.display().to_string()),
        )]
        .into_iter()
        .collect(),
    };
    let registry = registry_with_lanes(
        primary,
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("session-open-model".to_owned()),
            model: Some("prime".to_owned()),
            models: Some(vec!["fallback".to_owned(), "tertiary".to_owned()]),
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
    assert_eq!(outcome.text.trim_end(), "served|tertiary|work");
    assert_eq!(session.active_model().as_deref(), Some("tertiary"));
    let calls_log = std::fs::read_to_string(&log).unwrap_or_default();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "session/open|prime",
            "session/prompt|prime",
            "session/open|fallback",
            "session/open|tertiary",
            "session/prompt|tertiary",
        ],
        "a refused session/open advances the chain on the same lane: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer that answers the first pre-turn selector call, then exits
/// on the second: model one's prompt is dispatched and explicitly
/// refused, so its dispatch record must not poison model two's
/// pre-dispatch failure — the alternate lane still serves the turn.
#[cfg(unix)]
const RPC_REFUSE_THEN_DIE_ON_SECOND_SELECT: &str = r#"
import json, os, sys

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

selects = 0
session_model = None
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
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/select":
        selects += 1
        if selects > 1:
            sys.stdout.flush()
            os._exit(1)
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
    elif method == "session/prompt":
        model = params.get("model") or session_model
        if model == "prime":
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
        else:
            send({"jsonrpc": "2.0", "id": request_id,
                  "result": {"text": "served|{}|{}".format(
                      model or "unset", params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_model_retry_disconnect_scopes_dispatch_evidence() {
    // Attempt one dispatches its prompt and is explicitly refused; the
    // wire journal recorded the `Sent` frame. The chain advances and the
    // peer dies during attempt two's pre-turn call — a fresh attempt
    // whose own prompt never left. A stale dispatch flag from attempt
    // one would condemn the safe lane fallback.
    let dir = TestDirectory::new("rpc-retry-disconnect");
    let script = dir.path().join("rpc_refuse_die.py");
    std::fs::write(&script, RPC_REFUSE_THEN_DIE_ON_SECOND_SELECT).unwrap();
    let mut primary = rpc_spec(
        "two-attempt",
        &script,
        json!({}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    let LaneSpec::Rpc(lane) = &mut primary.lane else {
        panic!("rpc_spec builds an rpc lane")
    };
    lane.pre_turn = vec![RpcCallSpec {
        method: "session/select".to_owned(),
        params: json!({}),
    }];
    let registry = registry_with_lanes(
        primary,
        cli_spec(
            "alive",
            "echo",
            &["alt|{model}|{prompt}"],
            PromptChannel::ArgOnly,
        ),
        ProfilePatch {
            agent: Some("two-attempt".to_owned()),
            model: Some("prime".to_owned()),
            models: Some(vec!["fallback".to_owned()]),
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
    assert_eq!(outcome.text.trim_end(), "alt|fallback|work");
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn rpc_session_bound_alternate_clears_the_stale_turn_model() {
    // The retry scheduled `fallback` on the primary's per-turn surface,
    // then the peer died before its prompt could dispatch. The alternate
    // binds its model at session/open: replaying the turn with the
    // previous lane's per-turn override would be rejected before
    // dispatch — the replay must carry the caller's fields alone.
    let dir = TestDirectory::new("rpc-bound-alternate");
    let dying = dir.path().join("rpc_refuse_die.py");
    std::fs::write(&dying, RPC_REFUSE_THEN_DIE_ON_SECOND_SELECT).unwrap();
    let mut primary = rpc_spec(
        "turn-model",
        &dying,
        json!({}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    let LaneSpec::Rpc(lane) = &mut primary.lane else {
        panic!("rpc_spec builds an rpc lane")
    };
    lane.pre_turn = vec![RpcCallSpec {
        method: "session/select".to_owned(),
        params: json!({}),
    }];
    let bound = dir.path().join("rpc_bound.py");
    std::fs::write(&bound, RPC_QUOTA_SESSION_MODEL).unwrap();
    let log = dir.path().join("calls.log");
    let mut alternate = rpc_spec(
        "session-bound",
        &bound,
        json!({"model": "{model}"}),
        json!({"text": "{prompt}"}),
    );
    alternate.command.env = EnvPolicy::Sanitized {
        inherit: vec![],
        set: [(
            "AWP_CALL_LOG".to_owned(),
            EnvValue::Literal(log.display().to_string()),
        )]
        .into_iter()
        .collect(),
    };
    let registry = registry_with_lanes(
        primary,
        alternate,
        ProfilePatch {
            agent: Some("turn-model".to_owned()),
            model: Some("prime".to_owned()),
            models: Some(vec!["fallback".to_owned()]),
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
    assert_eq!(outcome.text.trim_end(), "served|fallback|work");
    assert_eq!(session.active_model().as_deref(), Some("fallback"));
    let calls_log = std::fs::read_to_string(&log).unwrap_or_default();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec!["session/open|fallback", "session/prompt|fallback"],
        "the replay reached the session-bound lane without a turn selector: {calls_log}"
    );

    // The bound model stays the session's: a second default turn is
    // served by `fallback` through the session binding, not a selector.
    let outcome = session.prompt("more".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "served|fallback|more");
    session.close().await.unwrap();
}

/// An rpc peer that reports a positive total-only usage record — the
/// `total_tokens` counter the token normalizer recognizes — and then
/// refuses the prompt with quota: the record is billed work, so the
/// refusal must propagate instead of replaying on the fallback model.
#[cfg(unix)]
const RPC_TOTAL_USAGE_THEN_QUOTA: &str = r#"
import json, sys

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

for line in sys.stdin:
    try:
        request = json.loads(line)
    except Exception:
        continue
    method = request.get("method")
    request_id = request.get("id")
    if method == "session/open":
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        send({"jsonrpc": "2.0", "method": "meter",
              "params": {"usage": {"total_tokens": 7}}})
        send({"jsonrpc": "2.0", "id": request_id,
              "error": {"code": -32000, "message":
                        "resource_exhausted: quota exceeded"}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_total_only_usage_does_not_advance_the_model_chain() {
    // A positive `total_tokens` is a recognized usage counter — billed
    // work like any other — not bookkeeping. The quota refusal after it
    // must not resubmit a possibly paid turn; `prime` stays active.
    let dir = TestDirectory::new("rpc-total-usage");
    let script = dir.path().join("rpc_meter.py");
    std::fs::write(&script, RPC_TOTAL_USAGE_THEN_QUOTA).unwrap();
    let registry = registry_with_lanes(
        rpc_spec(
            "metered",
            &script,
            json!({}),
            json!({"text": "{prompt}", "model": "{model}"}),
        ),
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("metered".to_owned()),
            model: Some("prime".to_owned()),
            models: Some(vec!["fallback".to_owned()]),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    let error = session.prompt("work".to_owned()).await.unwrap_err();
    assert_eq!(error.code, "agent_quota_exhausted");
    assert_eq!(session.active_model().as_deref(), Some("prime"));
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn acp_refused_chain_candidate_keeps_the_prior_default() {
    // `fallback` is the last chain entry and the fixture does not
    // advertise it: A4AA refuses the selector before dispatch. The
    // refused candidate must not become the session default — a later
    // prompt submits the prior default, and the provider's own answer
    // (quota) comes back rather than another selector rejection.
    let mut spec = fixture_spec();
    spec.command.env = EnvPolicy::Sanitized {
        inherit: vec![],
        set: [
            (
                "AWP_FIXTURE_MODELS".to_owned(),
                EnvValue::Literal("prime".to_owned()),
            ),
            (
                "AWP_FIXTURE_QUOTA".to_owned(),
                EnvValue::Literal("prime".to_owned()),
            ),
        ]
        .into_iter()
        .collect(),
    };
    let registry = registry_with(
        spec,
        ProfilePatch {
            agent: Some("fixture".to_owned()),
            model: Some("prime".to_owned()),
            models: Some(vec!["fallback".to_owned()]),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    let error = session.prompt("work".to_owned()).await.unwrap_err();
    assert_eq!(error.code, "agent_configuration_unavailable");
    assert_eq!(session.active_model().as_deref(), Some("prime"));

    // A refused selector never publishes: the next default turn goes
    // back on the wire with `prime`, and the refusal it gets is the
    // provider's quota answer — not another validation rejection.
    let error = session.prompt("again".to_owned()).await.unwrap_err();
    assert_eq!(error.code, "agent_quota_exhausted");
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn rpc_refused_reopen_keeps_the_installed_session() {
    // The session-bound retry re-opens this lane to bind `fallback`, and
    // `fallback` is refused at establishment — the last chain entry.
    // The installed session must stay the live `prime` one: a refused
    // replacement neither closes it nor reports a model that never
    // opened.
    let dir = TestDirectory::new("rpc-refused-reopen");
    let script = dir.path().join("rpc_session_open_model.py");
    std::fs::write(&script, RPC_QUOTA_SESSION_OPEN_MODEL).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "session-open-model",
        &script,
        json!({"model": "{model}"}),
        json!({"text": "{prompt}"}),
    );
    primary.command.env = EnvPolicy::Sanitized {
        inherit: vec![],
        set: [(
            "AWP_CALL_LOG".to_owned(),
            EnvValue::Literal(log.display().to_string()),
        )]
        .into_iter()
        .collect(),
    };
    let registry = registry_with_lanes(
        primary,
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("session-open-model".to_owned()),
            model: Some("prime".to_owned()),
            models: Some(vec!["fallback".to_owned()]),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    let error = session.prompt("work".to_owned()).await.unwrap_err();
    assert_eq!(error.code, "agent_quota_exhausted");
    assert_eq!(session.active_model().as_deref(), Some("prime"));

    // The installed session is still the live one: the next turn
    // dispatches on it and gets the provider's answer — not a
    // closed-session failure — and no extra open appears in the log.
    let error = session.prompt("again".to_owned()).await.unwrap_err();
    assert_eq!(error.code, "agent_quota_exhausted");
    let calls_log = std::fs::read_to_string(&log).unwrap_or_default();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "session/open|prime",
            "session/prompt|prime",
            "session/open|fallback",
            "session/prompt|prime",
        ],
        "the refused reopen left the installed session in place: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer that reports bookkeeping under the conventional usage
/// carrier — a sequence number, not a token counter — and then refuses
/// the prompt with quota: raw bookkeeping is not billed work, so the
/// refusal must still advance the model chain.
#[cfg(unix)]
const RPC_BOOKKEEPING_THEN_QUOTA: &str = r#"
import json, sys

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

session_model = None
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
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        model = params.get("model") or session_model
        if model == "prime":
            send({"jsonrpc": "2.0", "method": "meter",
                  "params": {"usage": {"seq": 7, "cursor": "c-41"}}})
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
        else:
            send({"jsonrpc": "2.0", "id": request_id,
                  "result": {"text": "served|{}|{}".format(
                      model or "unset", params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_bookkeeping_usage_still_advances_the_model_chain() {
    // A positive numeric field under `params.usage` is a sequence cursor,
    // not a token counter: the relayed event must read billable evidence
    // with the same normalized counters the wire journal applies —
    // otherwise a quota refusal after pure bookkeeping would skip the
    // model chain entirely.
    let dir = TestDirectory::new("rpc-bookkeeping");
    let script = dir.path().join("rpc_bookkeeping.py");
    std::fs::write(&script, RPC_BOOKKEEPING_THEN_QUOTA).unwrap();
    let registry = registry_with_lanes(
        rpc_spec(
            "metered",
            &script,
            json!({}),
            json!({"text": "{prompt}", "model": "{model}"}),
        ),
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("metered".to_owned()),
            model: Some("prime".to_owned()),
            models: Some(vec!["fallback".to_owned()]),
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
    assert_eq!(outcome.text.trim_end(), "served|fallback|work");
    assert_eq!(session.active_model().as_deref(), Some("fallback"));
    session.close().await.unwrap();
}

/// An rpc peer that reports nonzero usage, replaces it with an all-zero
/// record, then answers the prompt with a quota refusal: the lane's own
/// latest usage says nothing was served, yet the wire journal holds the
/// earlier billable evidence.
#[cfg(unix)]
const RPC_ZEROED_USAGE_THEN_QUOTA: &str = r#"
import json, sys

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

for line in sys.stdin:
    try:
        request = json.loads(line)
    except Exception:
        continue
    method = request.get("method")
    request_id = request.get("id")
    if method == "session/open":
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        send({"jsonrpc": "2.0", "method": "meter",
              "params": {"usage": {"output_tokens": 7}}})
        send({"jsonrpc": "2.0", "method": "meter",
              "params": {"usage": {"output_tokens": 0}}})
        send({"jsonrpc": "2.0", "id": request_id,
              "error": {"code": -32000, "message":
                        "resource_exhausted: quota exceeded"}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_zeroed_usage_does_not_advance_the_model_chain() {
    // The lane's final usage record reads zero, so a4aa classifies the
    // quota refusal as retryable — but the wire journal already recorded
    // the earlier nonzero usage. The chain must not resubmit a possibly
    // billed turn; the refusal propagates with `prime` still active.
    let dir = TestDirectory::new("rpc-zeroed-usage");
    let script = dir.path().join("rpc_meter.py");
    std::fs::write(&script, RPC_ZEROED_USAGE_THEN_QUOTA).unwrap();
    let registry = registry_with_lanes(
        rpc_spec(
            "metered",
            &script,
            json!({}),
            json!({"text": "{prompt}", "model": "{model}"}),
        ),
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("metered".to_owned()),
            model: Some("prime".to_owned()),
            models: Some(vec!["fallback".to_owned()]),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    let error = session.prompt("work".to_owned()).await.unwrap_err();
    assert_eq!(error.code, "agent_quota_exhausted");
    assert_eq!(session.active_model().as_deref(), Some("prime"));
    session.close().await.unwrap();
}

/// An rpc peer that emits a notification outside prompt execution — one
/// right after `session/open` and one right after each prompt's result.
#[cfg(unix)]
const RPC_EVENTS_BETWEEN_TURNS: &str = r#"
import json, sys

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

for line in sys.stdin:
    try:
        request = json.loads(line)
    except Exception:
        continue
    method = request.get("method")
    request_id = request.get("id")
    if method == "session/open":
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
        send({"jsonrpc": "2.0", "method": "session/ready",
              "params": {"note": "opened"}})
    elif method == "session/prompt":
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "done|{}".format(
                  (request.get("params") or {}).get("text", ""))}})
        send({"jsonrpc": "2.0", "method": "turn/idle",
              "params": {"note": "settled"}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_events_between_turns_reach_the_caller() {
    // Events must flow for the session's whole lifetime, not only while a
    // prompt is in flight: the session-ready notification arrives before
    // the first prompt, and the idle notification lands after the turn's
    // outcome without a following prompt to drain it.
    let dir = TestDirectory::new("rpc-between-turns");
    let script = dir.path().join("rpc_events.py");
    std::fs::write(&script, RPC_EVENTS_BETWEEN_TURNS).unwrap();
    let registry = registry_with(
        rpc_spec("chatty", &script, json!({}), json!({"text": "{prompt}"})),
        ProfilePatch {
            agent: Some("chatty".to_owned()),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, mut rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    // Before any prompt: the ready notification is already delivered.
    let ready = tokio::time::timeout(std::time::Duration::from_secs(5), rx.recv())
        .await
        .expect("session/ready stalled outside prompt execution")
        .expect("event stream closed");
    assert!(
        matches!(ready, AgentEvent::Raw { ref raw } if raw.get("method") == Some(&json!("session/ready"))),
        "unexpected first event: {ready:?}"
    );

    let outcome = session.prompt("work".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "done|work");

    // After the outcome resolves — and without another prompt — the idle
    // notification is still delivered.
    let idle = tokio::time::timeout(std::time::Duration::from_secs(5), rx.recv())
        .await
        .expect("turn/idle stalled outside prompt execution")
        .expect("event stream closed");
    assert!(
        matches!(idle, AgentEvent::Raw { ref raw } if raw.get("method") == Some(&json!("turn/idle"))),
        "unexpected post-turn event: {idle:?}"
    );
    session.close().await.unwrap();
}

/// An rpc peer whose pre-turn call shares the prompt method: the first
/// `session/prompt` frame is the configuration call, not the prompt.
/// `AWP_KILL_ON_TEXT=1` makes the peer die only when the frame carries
/// the turn's own `text` — the real prompt call.
#[cfg(unix)]
const RPC_SHARED_PROMPT_METHOD: &str = r#"
import json, os, sys

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

kill_on_text = os.environ.get("AWP_KILL_ON_TEXT", "") == "1"
for line in sys.stdin:
    try:
        request = json.loads(line)
    except Exception:
        continue
    method = request.get("method")
    request_id = request.get("id")
    params = request.get("params") or {}
    if method == "session/prompt" and (not kill_on_text or params.get("text") is not None):
        sys.stdout.flush()
        os._exit(1)
    if method == "session/open":
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

/// One shared-method fixture: a pre-turn `session/prompt` call without a
/// `text` field stands beside the real prompt call.
#[cfg(unix)]
fn rpc_shared_method_spec(id: &str, script: &Path, kill_on_text: bool) -> AgentSpec {
    let mut spec = rpc_spec(id, script, json!({}), json!({"text": "{prompt}"}));
    let LaneSpec::Rpc(lane) = &mut spec.lane else {
        panic!("rpc_spec builds an rpc lane")
    };
    lane.pre_turn = vec![RpcCallSpec {
        method: "session/prompt".to_owned(),
        params: json!({"select": true}),
    }];
    spec.command.env = EnvPolicy::Sanitized {
        inherit: vec![],
        set: [(
            "AWP_KILL_ON_TEXT".to_owned(),
            EnvValue::Literal(if kill_on_text { "1" } else { "0" }.to_owned()),
        )]
        .into_iter()
        .collect(),
    };
    spec
}

#[cfg(unix)]
#[tokio::test]
async fn rpc_shared_method_pre_turn_disconnect_falls_back() {
    // The peer dies while the pre-turn call is in flight — the frame that
    // left was the configuration call sharing the prompt's method, never
    // the prompt itself. A method-name-only dispatch reading would block
    // the safe failover: the alternate must serve the turn.
    let dir = TestDirectory::new("rpc-shared-preturn");
    let script = dir.path().join("rpc_shared.py");
    std::fs::write(&script, RPC_SHARED_PROMPT_METHOD).unwrap();
    let registry = registry_with_lanes(
        rpc_shared_method_spec("shared", &script, false),
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("shared".to_owned()),
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
    assert_eq!(outcome.text.trim_end(), "alt|work");
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn rpc_shared_method_prompt_dispatch_never_replays() {
    // The same shared method now carries the real prompt: the pre-turn
    // call answers, the frame holding the turn's text leaves, and the
    // peer dies on it. The prompt provably dispatched — the uncertain
    // failure must propagate instead of replaying on the alternate.
    let dir = TestDirectory::new("rpc-shared-dispatch");
    let script = dir.path().join("rpc_shared.py");
    std::fs::write(&script, RPC_SHARED_PROMPT_METHOD).unwrap();
    let registry = registry_with_lanes(
        rpc_shared_method_spec("shared", &script, true),
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("shared".to_owned()),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    let error = session.prompt("work".to_owned()).await.unwrap_err();
    assert_eq!(error.code, "agent_transport_failed");
    let _ = session.close().await;
}

#[cfg(unix)]
#[tokio::test]
async fn rpc_retry_disconnect_before_dispatch_keeps_the_prior_model() {
    // `fallback` was scheduled on the turn surface, then the peer died
    // during the retry's pre-turn call — the candidate's prompt never
    // dispatched, and every alternate fails to open. The transport error
    // is no acceptance evidence: `active_model` must stay `prime`.
    let dir = TestDirectory::new("rpc-retry-unaccepted");
    let script = dir.path().join("rpc_refuse_die.py");
    std::fs::write(&script, RPC_REFUSE_THEN_DIE_ON_SECOND_SELECT).unwrap();
    let mut primary = rpc_spec(
        "unaccepted",
        &script,
        json!({}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    let LaneSpec::Rpc(lane) = &mut primary.lane else {
        panic!("rpc_spec builds an rpc lane")
    };
    lane.pre_turn = vec![RpcCallSpec {
        method: "session/select".to_owned(),
        params: json!({}),
    }];
    let dead = dir.path().join("rpc_noopen.py");
    std::fs::write(&dead, RPC_DIES_ON_METHOD).unwrap();
    let registry = registry_with_lanes(
        primary,
        rpc_kill_spec("noopen", &dead, "session/open"),
        ProfilePatch {
            agent: Some("unaccepted".to_owned()),
            model: Some("prime".to_owned()),
            models: Some(vec!["fallback".to_owned()]),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    let error = session.prompt("work".to_owned()).await.unwrap_err();
    assert_eq!(error.code, "agent_transport_failed");
    assert_eq!(session.active_model().as_deref(), Some("prime"));
    session.close().await.unwrap();
}

/// An rpc peer that binds its model at `session/open`: the first prompt
/// it serves fails with a certain remote error — a lane failure the
/// session survives — and later prompts refuse `prime` with quota while
/// serving every other model.
#[cfg(unix)]
const RPC_BOOM_THEN_QUOTA_SESSION_MODEL: &str = r#"
import json, os, sys

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

log = open(os.environ["AWP_CALL_LOG"], "a")
session_model = None
prompts = 0
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
        log.write("session/open|{}\n".format(session_model))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        prompts += 1
        model = params.get("model") or session_model
        log.write("session/prompt|{}\n".format(model))
        log.flush()
        if model == "prime" and prompts == 1:
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message": "permanent boom"}})
        elif model == "prime":
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
        else:
            send({"jsonrpc": "2.0", "id": request_id,
                  "result": {"text": "served|{}|{}".format(
                      model or "unset", params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_failed_alternate_open_keeps_the_installed_lane_index() {
    // Turn one fails on the primary and every alternate fails to open:
    // the installed session survives, so its lane index must stay put.
    // Turn two's quota refusal then triggers the same-lane model retry —
    // an index left pointing past the lane list would skip the reopen.
    let dir = TestDirectory::new("rpc-lane-index");
    let script = dir.path().join("rpc_boom_quota.py");
    std::fs::write(&script, RPC_BOOM_THEN_QUOTA_SESSION_MODEL).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "surviving",
        &script,
        json!({"model": "{model}"}),
        json!({"text": "{prompt}"}),
    );
    primary.command.env = EnvPolicy::Sanitized {
        inherit: vec![],
        set: [(
            "AWP_CALL_LOG".to_owned(),
            EnvValue::Literal(log.display().to_string()),
        )]
        .into_iter()
        .collect(),
    };
    let dead = dir.path().join("rpc_noopen.py");
    std::fs::write(&dead, RPC_DIES_ON_METHOD).unwrap();
    let registry = registry_with_lanes(
        primary,
        rpc_kill_spec("noopen", &dead, "session/open"),
        ProfilePatch {
            agent: Some("surviving".to_owned()),
            model: Some("prime".to_owned()),
            models: Some(vec!["fallback".to_owned()]),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    let error = session.prompt("work".to_owned()).await.unwrap_err();
    assert_eq!(error.code, "agent_remote_error");

    // The installed primary session still serves: its quota refusal must
    // reopen this same lane bound to `fallback`, not skip the retry.
    let outcome = session.prompt("more".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "served|fallback|more");
    assert_eq!(session.active_model().as_deref(), Some("fallback"));
    let calls_log = std::fs::read_to_string(&log).unwrap_or_default();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "session/open|prime",
            "session/prompt|prime",
            "session/prompt|prime",
            "session/open|fallback",
            "session/prompt|fallback",
        ],
        "the same-lane model retry re-opened the installed lane: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer that binds its model at `session/open`, refuses `prime`
/// with quota at the prompt, and then fails its next prompt with a
/// certain remote error — the lane failure that sends the turn looking
/// for an alternate.
#[cfg(unix)]
const RPC_QUOTA_SESSION_MODEL_THEN_BOOM: &str = r#"
import json, os, sys

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

log = open(os.environ["AWP_CALL_LOG"], "a")
session_model = None
prompts = 0
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
        log.write("session/open|{}\n".format(session_model))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        prompts += 1
        model = params.get("model") or session_model
        log.write("session/prompt|{}\n".format(model))
        log.flush()
        if model == "prime":
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
        elif prompts >= 2:
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message": "permanent boom"}})
        else:
            send({"jsonrpc": "2.0", "id": request_id,
                  "result": {"text": "served|{}|{}".format(
                      model or "unset", params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

/// An rpc peer that binds and serves whatever model its session opened
/// with, logging both calls.
#[cfg(unix)]
const RPC_SESSION_MODEL_LOG: &str = r#"
import json, os, sys

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

log = open(os.environ["AWP_CALL_LOG"], "a")
session_model = None
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
        log.write("session/open|{}\n".format(session_model))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        model = params.get("model") or session_model
        log.write("session/prompt|{}\n".format(model))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "alt|{}|{}".format(
                  model or "unset", params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

/// Build a session-bound rpc lane on `script` that logs calls to `log`.
#[cfg(unix)]
fn rpc_session_bound_spec(id: &str, script: &Path, log: &Path) -> AgentSpec {
    let mut spec = rpc_spec(
        id,
        script,
        json!({"model": "{model}"}),
        json!({"text": "{prompt}"}),
    );
    spec.command.env = EnvPolicy::Sanitized {
        inherit: vec![],
        set: [(
            "AWP_CALL_LOG".to_owned(),
            EnvValue::Literal(log.display().to_string()),
        )]
        .into_iter()
        .collect(),
    };
    spec
}

#[cfg(unix)]
#[tokio::test]
async fn rpc_lane_switch_projects_the_accepted_model_not_the_requested() {
    // The caller opened with an explicit `prime`, the session-bound retry
    // moved the chain to `fallback`, then the lane failed. The alternate
    // must open bound to the accepted entry — replaying the caller's
    // original model would have the replacement serve `prime` while the
    // session reports `fallback` active.
    let dir = TestDirectory::new("rpc-switch-model");
    let script = dir.path().join("rpc_quota_boom.py");
    std::fs::write(&script, RPC_QUOTA_SESSION_MODEL_THEN_BOOM).unwrap();
    let primary_log = dir.path().join("primary.log");
    let alt_script = dir.path().join("rpc_alt.py");
    std::fs::write(&alt_script, RPC_SESSION_MODEL_LOG).unwrap();
    let alternate_log = dir.path().join("alternate.log");
    let registry = registry_with_lanes(
        rpc_session_bound_spec("binds", &script, &primary_log),
        rpc_session_bound_spec("alternate", &alt_script, &alternate_log),
        ProfilePatch {
            agent: Some("binds".to_owned()),
            model: Some("prime".to_owned()),
            models: Some(vec!["fallback".to_owned()]),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let mut request = SessionRequest::default();
    request.config.model = Some("prime".to_owned());
    let session = agent.open_session(request, tx).await.unwrap();

    // `prime` is refused at the prompt; the retry re-opens this lane
    // bound to `fallback`.
    let outcome = session.prompt("work".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "served|fallback|work");
    assert_eq!(session.active_model().as_deref(), Some("fallback"));

    // The lane then fails mid-session: the alternate opens bound to the
    // accepted `fallback`, not the caller's original `prime`.
    let outcome = session.prompt("more".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "alt|fallback|more");
    assert_eq!(session.active_model().as_deref(), Some("fallback"));
    let calls_log = std::fs::read_to_string(&alternate_log).unwrap_or_default();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec!["session/open|fallback", "session/prompt|fallback"],
        "the alternate opened with the accepted chain model: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer that fails every prompt with a certain remote error — a
/// lane failure the session survives.
#[cfg(unix)]
const RPC_BOOM: &str = r#"
import json, sys

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

for line in sys.stdin:
    try:
        request = json.loads(line)
    except Exception:
        continue
    method = request.get("method")
    request_id = request.get("id")
    if method == "session/open":
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        send({"jsonrpc": "2.0", "id": request_id,
              "error": {"code": -32000, "message": "permanent boom"}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

/// An rpc peer that delays its `session/open` answer — long enough for a
/// caller's close or cancel to land while the replacement opens — then
/// serves prompts, logging every method it sees.
#[cfg(unix)]
const RPC_SLOW_OPEN: &str = r#"
import json, os, sys, time

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

log = open(os.environ["AWP_CALL_LOG"], "a")
for line in sys.stdin:
    try:
        request = json.loads(line)
    except Exception:
        continue
    method = request.get("method")
    request_id = request.get("id")
    if method:
        log.write(method + "\n")
        log.flush()
    if method == "session/open":
        time.sleep(1.0)
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "alt|{}".format(
                  (request.get("params") or {}).get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn close_during_replacement_open_never_dispatches_the_replay() {
    // The turn's lane fails and the alternate's session/open is still in
    // flight when `close` returns: the driver must not install the
    // replacement or dispatch the replay after close finished — the turn
    // answers with the failure it already had.
    let dir = TestDirectory::new("rpc-close-reopen");
    let boom = dir.path().join("rpc_boom.py");
    std::fs::write(&boom, RPC_BOOM).unwrap();
    let slow = dir.path().join("rpc_slow.py");
    std::fs::write(&slow, RPC_SLOW_OPEN).unwrap();
    let log = dir.path().join("calls.log");
    let registry = registry_with_lanes(
        rpc_spec("boom", &boom, json!({}), json!({"text": "{prompt}"})),
        rpc_logging_spec("slow", &slow, &log),
        ProfilePatch {
            agent: Some("boom".to_owned()),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = std::sync::Arc::new(
        agent
            .open_session(SessionRequest::default(), tx)
            .await
            .unwrap(),
    );

    let driver = {
        let session = session.clone();
        tokio::spawn(async move { session.prompt("work".to_owned()).await })
    };
    // The alternate's `session/open` answer is a second out; the close
    // lands mid-flight by a wide margin.
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    session.close().await.unwrap();

    let error = driver.await.unwrap().unwrap_err();
    assert_eq!(error.code, "agent_remote_error");
    // The replacement opened — and was closed — but its prompt call must
    // never have reached the wire after close returned.
    let calls_log = std::fs::read_to_string(&log).unwrap_or_default();
    assert!(
        !calls_log.lines().any(|method| method == "session/prompt"),
        "the replacement served a replay after close returned: {calls_log}"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn cancel_during_replacement_open_cancels_the_turn() {
    // `cancel` on the idle old session returns while a replacement is
    // still opening: the turn must end cancelled, and the replacement
    // must never see the replayed prompt.
    let dir = TestDirectory::new("rpc-cancel-reopen");
    let boom = dir.path().join("rpc_boom.py");
    std::fs::write(&boom, RPC_BOOM).unwrap();
    let slow = dir.path().join("rpc_slow.py");
    std::fs::write(&slow, RPC_SLOW_OPEN).unwrap();
    let log = dir.path().join("calls.log");
    let registry = registry_with_lanes(
        rpc_spec("boom", &boom, json!({}), json!({"text": "{prompt}"})),
        rpc_logging_spec("slow", &slow, &log),
        ProfilePatch {
            agent: Some("boom".to_owned()),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = std::sync::Arc::new(
        agent
            .open_session(SessionRequest::default(), tx)
            .await
            .unwrap(),
    );

    let driver = {
        let session = session.clone();
        tokio::spawn(async move { session.prompt("work".to_owned()).await })
    };
    // Wait until the alternate logged its `session/open` — the
    // replacement open is provably in-flight, so the cancel lands inside
    // the window the turn driver is watching rather than racing the
    // turn's own startup under parallel load.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        let contents = std::fs::read_to_string(&log).unwrap_or_default();
        if contents.lines().any(|method| method == "session/open") {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the replacement's session/open never dispatched: {contents}"
        );
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    session.cancel().await.unwrap();

    let outcome = driver.await.unwrap().unwrap();
    assert_eq!(outcome.stop_reason, "cancelled");
    let calls_log = std::fs::read_to_string(&log).unwrap_or_default();
    assert!(
        !calls_log.lines().any(|method| method == "session/prompt"),
        "the replacement served a replay after cancel returned: {calls_log}"
    );
    let _ = session.close().await;
}

#[cfg(unix)]
#[tokio::test]
async fn rpc_shared_method_short_prompt_still_falls_back() {
    // The armed prompt "prompt" collides with the shared method name: a
    // text search over the frame would find it inside `session/prompt`
    // and read the pre-turn configuration call as dispatched. The probe
    // matches the template's own text payload instead — the pre-turn
    // frame carries no `text` field, the prompt never left, and the
    // alternate must serve the turn.
    let dir = TestDirectory::new("rpc-shared-short");
    let script = dir.path().join("rpc_shared.py");
    std::fs::write(&script, RPC_SHARED_PROMPT_METHOD).unwrap();
    let registry = registry_with_lanes(
        rpc_shared_method_spec("shared", &script, false),
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("shared".to_owned()),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    let outcome = session.prompt("prompt".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "alt|prompt");
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn rpc_shared_method_empty_prompt_still_falls_back() {
    // An empty armed prompt must not collapse the payload check into a
    // match-everything: the pre-turn frame still carries no `text` field,
    // so the prompt is confirmed never dispatched and the alternate
    // serves the empty turn.
    let dir = TestDirectory::new("rpc-shared-empty");
    let script = dir.path().join("rpc_shared.py");
    std::fs::write(&script, RPC_SHARED_PROMPT_METHOD).unwrap();
    let registry = registry_with_lanes(
        rpc_shared_method_spec("shared", &script, false),
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("shared".to_owned()),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    let outcome = session.prompt("".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "alt|");
    session.close().await.unwrap();
}

/// An rpc peer that answers `session/open`, refuses the first prompt with
/// a provider quota error, then closes its own stdin and sleeps: the
/// model retry's prompt write deterministically fails on the transport —
/// `writer_failed`, a confirmed-undelivered dispatch the probe reads from
/// the last `Sent` frame.
#[cfg(unix)]
const RPC_QUOTA_THEN_CLOSES_STDIN: &str = r#"
import json, os, sys, time

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

request = json.loads(sys.stdin.readline())
send({"jsonrpc": "2.0", "id": request["id"],
      "result": {"sessionId": "awp-rpc-session"}})
request = json.loads(sys.stdin.readline())
os.close(0)
send({"jsonrpc": "2.0", "id": request["id"],
      "error": {"code": -32000, "message":
                "resource_exhausted: quota exceeded for this model"}})
time.sleep(30)
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_failed_prompt_write_keeps_the_prior_model() {
    // `fallback` was scheduled on the turn surface, then its prompt's own
    // write failed — `Sent` was recorded before the write, so dispatch
    // intent alone would publish the never-delivered candidate. With no
    // acceptance evidence and no reachable alternate, `active_model`
    // must stay `prime`.
    let dir = TestDirectory::new("rpc-write-unaccepted");
    let script = dir.path().join("rpc_quota_close.py");
    std::fs::write(&script, RPC_QUOTA_THEN_CLOSES_STDIN).unwrap();
    let primary = rpc_spec(
        "unaccepted",
        &script,
        json!({}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    let dead = dir.path().join("rpc_noopen.py");
    std::fs::write(&dead, RPC_DIES_ON_METHOD).unwrap();
    let registry = registry_with_lanes(
        primary,
        rpc_kill_spec("noopen", &dead, "session/open"),
        ProfilePatch {
            agent: Some("unaccepted".to_owned()),
            model: Some("prime".to_owned()),
            models: Some(vec!["fallback".to_owned()]),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    let error = session.prompt("work".to_owned()).await.unwrap_err();
    assert_eq!(error.code, "agent_transport_failed");
    assert_eq!(session.active_model().as_deref(), Some("prime"));
    session.close().await.unwrap();
}

/// An rpc peer that binds nothing at open and refuses every prompt with
/// quota, logging the model each call carried — a lane that applies the
/// model inside the prompt call, where opening only stores a default.
#[cfg(unix)]
const RPC_REFUSES_MODEL_AT_PROMPT: &str = r#"
import json, os, sys

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

log = open(os.environ["AWP_CALL_LOG"], "a")
for line in sys.stdin:
    try:
        request = json.loads(line)
    except Exception:
        continue
    method = request.get("method")
    request_id = request.get("id")
    params = request.get("params") or {}
    if method == "session/open":
        log.write("session/open|{}\n".format(params.get("model") or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        log.write("session/prompt|{}\n".format(params.get("model") or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "error": {"code": -32000, "message":
                        "resource_exhausted: quota exceeded for this model"}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_lane_switch_keeps_the_carried_model_pending() {
    // The retry scheduled `fallback` on the primary's per-turn surface,
    // then the prompt's write failed — the alternate opens carrying the
    // candidate, but opening only stores it as a default. The replay's
    // quota refusal proves the lane never took it: `active_model` must
    // stay `prime` instead of publishing the refused candidate.
    let dir = TestDirectory::new("rpc-switch-pending");
    let script = dir.path().join("rpc_quota_close.py");
    std::fs::write(&script, RPC_QUOTA_THEN_CLOSES_STDIN).unwrap();
    let primary = rpc_spec(
        "unaccepted",
        &script,
        json!({}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    let refusing = dir.path().join("rpc_refuse.py");
    std::fs::write(&refusing, RPC_REFUSES_MODEL_AT_PROMPT).unwrap();
    let log = dir.path().join("calls.log");
    let mut alternate = rpc_spec(
        "refusing",
        &refusing,
        json!({}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    alternate.command.env = EnvPolicy::Sanitized {
        inherit: vec![],
        set: [(
            "AWP_CALL_LOG".to_owned(),
            EnvValue::Literal(log.display().to_string()),
        )]
        .into_iter()
        .collect(),
    };
    let registry = registry_with_lanes(
        primary,
        alternate,
        ProfilePatch {
            agent: Some("unaccepted".to_owned()),
            model: Some("prime".to_owned()),
            models: Some(vec!["fallback".to_owned()]),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    let error = session.prompt("work".to_owned()).await.unwrap_err();
    assert_eq!(error.code, "agent_quota_exhausted");
    assert_eq!(session.active_model().as_deref(), Some("prime"));

    // The refused candidate never became the adapter's persistent
    // default: the next implicit prompt rides the last accepted model —
    // `prime` — not the refused `fallback` the session would otherwise
    // have retained.
    let error = session.prompt("again".to_owned()).await.unwrap_err();
    assert_eq!(error.code, "agent_quota_exhausted");
    assert_eq!(session.active_model().as_deref(), Some("prime"));
    let calls_log = std::fs::read_to_string(&log).unwrap_or_default();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "session/open|unset",
            "session/prompt|fallback",
            "session/prompt|prime",
        ],
        "the refused candidate must not remain the session's default: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer that never answers `session/open`: a replacement open on
/// this lane stalls inside the handshake — close and cancel must cut the
/// open short instead of waiting on it.
#[cfg(unix)]
const RPC_STALL_OPEN: &str = r#"
import json, sys, time

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

for line in sys.stdin:
    try:
        request = json.loads(line)
    except Exception:
        continue
    method = request.get("method")
    request_id = request.get("id")
    if method == "session/open":
        time.sleep(600)
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn close_during_stalled_replacement_open_ends_the_turn() {
    // The lane fails and the alternate's session/open never answers:
    // close lands mid-handshake — the driver must abandon the open and
    // end the turn with the failure it already had, not wait out the
    // stalled replacement.
    let dir = TestDirectory::new("rpc-close-stall");
    let boom = dir.path().join("rpc_boom.py");
    std::fs::write(&boom, RPC_BOOM).unwrap();
    let stall = dir.path().join("rpc_stall.py");
    std::fs::write(&stall, RPC_STALL_OPEN).unwrap();
    let registry = registry_with_lanes(
        rpc_spec("boom", &boom, json!({}), json!({"text": "{prompt}"})),
        rpc_spec("stall", &stall, json!({}), json!({"text": "{prompt}"})),
        ProfilePatch {
            agent: Some("boom".to_owned()),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = std::sync::Arc::new(
        agent
            .open_session(SessionRequest::default(), tx)
            .await
            .unwrap(),
    );

    let driver = {
        let session = session.clone();
        tokio::spawn(async move { session.prompt("work".to_owned()).await })
    };
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    session.close().await.unwrap();

    let error = tokio::time::timeout(std::time::Duration::from_secs(10), driver)
        .await
        .expect("the turn outlived close during a stalled replacement open")
        .unwrap()
        .unwrap_err();
    assert_eq!(error.code, "agent_remote_error");
}

#[cfg(unix)]
#[tokio::test]
async fn cancel_during_stalled_replacement_open_ends_the_turn() {
    // `cancel` on the idle old session returns while the replacement open
    // is stalled mid-handshake: the turn must end cancelled instead of
    // waiting on the abandoned replacement.
    let dir = TestDirectory::new("rpc-cancel-stall");
    let boom = dir.path().join("rpc_boom.py");
    std::fs::write(&boom, RPC_BOOM).unwrap();
    let stall = dir.path().join("rpc_stall.py");
    std::fs::write(&stall, RPC_STALL_OPEN).unwrap();
    let registry = registry_with_lanes(
        rpc_spec("boom", &boom, json!({}), json!({"text": "{prompt}"})),
        rpc_spec("stall", &stall, json!({}), json!({"text": "{prompt}"})),
        ProfilePatch {
            agent: Some("boom".to_owned()),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = std::sync::Arc::new(
        agent
            .open_session(SessionRequest::default(), tx)
            .await
            .unwrap(),
    );

    let driver = {
        let session = session.clone();
        tokio::spawn(async move { session.prompt("work".to_owned()).await })
    };
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    session.cancel().await.unwrap();

    let outcome = tokio::time::timeout(std::time::Duration::from_secs(10), driver)
        .await
        .expect("the turn outlived cancel during a stalled replacement open")
        .unwrap()
        .unwrap();
    assert_eq!(outcome.stop_reason, "cancelled");
    let _ = session.close().await;
}

/// An enveloped rpc peer: frames carry `cmd` instead of `method`, and
/// the child exits upon receiving `session/prompt` — the prompt frame
/// was delivered, then the transport died.
#[cfg(unix)]
const RPC_ENVELOPE_DIES_ON_PROMPT: &str = r#"
import json, os, sys

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

for line in sys.stdin:
    try:
        request = json.loads(line)
    except Exception:
        continue
    method = request.get("cmd")
    request_id = request.get("id")
    if method == "session/prompt":
        sys.stdout.flush()
        os._exit(1)
    if method == "session/open":
        send({"cmd": "response", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif request_id is not None:
        send({"cmd": "response", "id": request_id, "result": None})
"#;

/// Build a non-stock-envelope rpc lane on `script`.
#[cfg(unix)]
fn rpc_envelope_spec(
    id: &str,
    script: &Path,
    prompt_params: serde_json::Value,
    envelope: a4aa::RpcEnvelopeSpec,
) -> AgentSpec {
    let mut spec = rpc_spec(id, script, json!({}), prompt_params);
    let LaneSpec::Rpc(lane) = &mut spec.lane else {
        panic!("rpc_spec builds an rpc lane")
    };
    lane.envelope = Some(envelope);
    spec
}

#[cfg(unix)]
#[tokio::test]
async fn rpc_inline_envelope_post_dispatch_loss_never_replays() {
    // An inline envelope merges the call's object params into the frame
    // root — no `params` member at all. The prompt frame left, then the
    // peer died: the wire journal must read dispatch from the outbound
    // encoder's layout or the turn replays on the alternate — a
    // duplicated, possibly billed, turn.
    let dir = TestDirectory::new("rpc-inline-loss");
    let script = dir.path().join("rpc_inline.py");
    std::fs::write(&script, RPC_ENVELOPE_DIES_ON_PROMPT).unwrap();
    let envelope = a4aa::RpcEnvelopeSpec {
        method_field: "cmd".to_owned(),
        params_inline: true,
        response_method: Some("response".to_owned()),
        ..Default::default()
    };
    let registry = registry_with_lanes(
        rpc_envelope_spec("inline", &script, json!({"text": "{prompt}"}), envelope),
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("inline".to_owned()),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    let error = session.prompt("work".to_owned()).await.unwrap_err();
    assert_eq!(error.code, "agent_transport_failed");
    let _ = session.close().await;
}

#[cfg(unix)]
#[tokio::test]
async fn rpc_nested_envelope_post_dispatch_loss_never_replays() {
    // The same transport loss after delivery under a nested envelope:
    // `data` is declared for inbound payloads, yet outbound parameters
    // still sit under `params` — reading the prompt frame through the
    // inbound field hides the dispatch and replays a billed turn.
    let dir = TestDirectory::new("rpc-nested-loss");
    let script = dir.path().join("rpc_nested.py");
    std::fs::write(&script, RPC_ENVELOPE_DIES_ON_PROMPT).unwrap();
    let envelope = a4aa::RpcEnvelopeSpec {
        method_field: "cmd".to_owned(),
        params_inline: false,
        response_method: Some("response".to_owned()),
        data_field: Some("data".to_owned()),
        ..Default::default()
    };
    let registry = registry_with_lanes(
        rpc_envelope_spec("nested", &script, json!({"text": "{prompt}"}), envelope),
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("nested".to_owned()),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    let error = session.prompt("work".to_owned()).await.unwrap_err();
    assert_eq!(error.code, "agent_transport_failed");
    let _ = session.close().await;
}

/// An rpc peer with no model selector at all: it logs the model member
/// each call carried — proving the replay omitted the pending
/// candidate — and serves prompts without one.
#[cfg(unix)]
const RPC_SELECTORLESS_LOG: &str = r#"
import json, os, sys

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

log = open(os.environ["AWP_CALL_LOG"], "a")
for line in sys.stdin:
    try:
        request = json.loads(line)
    except Exception:
        continue
    method = request.get("method")
    request_id = request.get("id")
    params = request.get("params") or {}
    if method == "session/open":
        log.write("session/open|{}\n".format(params.get("model") or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        log.write("session/prompt|{}\n".format(params.get("model") or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "alt|{}".format(params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_lane_switch_to_selectorless_lane_keeps_the_prior_model() {
    // The retry scheduled `fallback` on the primary's per-turn surface,
    // then the prompt's write failed. The alternate has no model
    // selector: the replay omits the candidate on the wire, so the
    // served turn is no evidence the lane took it — `active_model` must
    // stay `prime` instead of publishing a model the lane cannot apply.
    let dir = TestDirectory::new("rpc-selectorless-switch");
    let script = dir.path().join("rpc_quota_close.py");
    std::fs::write(&script, RPC_QUOTA_THEN_CLOSES_STDIN).unwrap();
    let primary = rpc_spec(
        "unaccepted",
        &script,
        json!({}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    let serving = dir.path().join("rpc_plain.py");
    std::fs::write(&serving, RPC_SELECTORLESS_LOG).unwrap();
    let log = dir.path().join("calls.log");
    let mut alternate = rpc_spec(
        "selectorless",
        &serving,
        json!({}),
        json!({"text": "{prompt}"}),
    );
    alternate.command.env = EnvPolicy::Sanitized {
        inherit: vec![],
        set: [(
            "AWP_CALL_LOG".to_owned(),
            EnvValue::Literal(log.display().to_string()),
        )]
        .into_iter()
        .collect(),
    };
    let registry = registry_with_lanes(
        primary,
        alternate,
        ProfilePatch {
            agent: Some("unaccepted".to_owned()),
            model: Some("prime".to_owned()),
            models: Some(vec!["fallback".to_owned()]),
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
    assert_eq!(outcome.text.trim_end(), "alt|work");
    assert_eq!(session.active_model().as_deref(), Some("prime"));
    let calls_log = std::fs::read_to_string(&log).unwrap_or_default();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec!["session/open|unset", "session/prompt|unset"],
        "the pending candidate never reached the wire: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer that refuses prompts and stalls its `session/close`
/// shutdown call forever.
#[cfg(unix)]
const RPC_BOOM_STALLED_CLOSE: &str = r#"
import json, sys, time

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

for line in sys.stdin:
    try:
        request = json.loads(line)
    except Exception:
        continue
    method = request.get("method")
    request_id = request.get("id")
    if method == "session/open":
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        send({"jsonrpc": "2.0", "id": request_id,
              "error": {"code": -32000, "message": "permanent boom"}})
    elif method == "session/close":
        time.sleep(600)
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn stalled_lane_retirement_blocks_neither_turn_nor_close() {
    // The old lane's shutdown call never answers: retiring it under the
    // session lock would block both the replay and any later close.
    // Retirement is interruptible — the turn serves on the replacement
    // and close proceeds while the stalled shutdown drains alone.
    let dir = TestDirectory::new("rpc-retire-stall");
    let boom = dir.path().join("rpc_boom.py");
    std::fs::write(&boom, RPC_BOOM_STALLED_CLOSE).unwrap();
    let mut primary = rpc_spec("boom", &boom, json!({}), json!({"text": "{prompt}"}));
    let LaneSpec::Rpc(lane) = &mut primary.lane else {
        panic!("rpc_spec builds an rpc lane")
    };
    lane.session_close = Some(RpcCallSpec {
        method: "session/close".to_owned(),
        params: json!({}),
    });
    let serving = write_rpc_agent(dir.path());
    let registry = registry_with_lanes(
        primary,
        rpc_spec("serving", &serving, json!({}), json!({"text": "{prompt}"})),
        ProfilePatch {
            agent: Some("boom".to_owned()),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    let outcome = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        session.prompt("work".to_owned()),
    )
    .await
    .expect("the turn waited on the old lane's stalled shutdown")
    .unwrap();
    assert_eq!(outcome.text.trim_end(), "unset|unset|work");
    tokio::time::timeout(std::time::Duration::from_secs(10), session.close())
        .await
        .expect("close waited on the old lane's stalled shutdown")
        .unwrap();
}

/// A `BeforeTurn` hook that parks until released — modelling a
/// reservation that stays in flight while the caller cancels.
#[cfg(unix)]
struct PausingHook {
    entered: std::sync::Arc<tokio::sync::Notify>,
    release: std::sync::Arc<tokio::sync::Notify>,
}

#[cfg(unix)]
#[async_trait::async_trait]
impl awp::BeforeTurn for PausingHook {
    async fn run(&self) -> awp::Result<()> {
        self.entered.notify_one();
        self.release.notified().await;
        Ok(())
    }
}

#[cfg(unix)]
#[tokio::test]
async fn cancel_during_blocked_reservation_never_dispatches() {
    // `cancel` lands while the reservation hook is parked: the lane is
    // idle, so cancel returns having marked nothing — and when the hook
    // then succeeds the prompt must still not dispatch.
    let dir = TestDirectory::new("rpc-cancel-admission");
    let script = dir.path().join("rpc_log.py");
    std::fs::write(&script, RPC_LOG_METHODS).unwrap();
    let log = dir.path().join("calls.log");
    let registry = registry_with(
        rpc_logging_spec("admission", &script, &log),
        ProfilePatch {
            agent: Some("admission".to_owned()),
            ..Default::default()
        },
    );
    let hook = std::sync::Arc::new(PausingHook {
        entered: std::sync::Arc::new(tokio::sync::Notify::new()),
        release: std::sync::Arc::new(tokio::sync::Notify::new()),
    });
    let agent = registry.resolve("p").unwrap().before_turn(hook.clone());
    let (tx, _rx) = events();
    let session = std::sync::Arc::new(
        agent
            .open_session(SessionRequest::default(), tx)
            .await
            .unwrap(),
    );

    let driver = {
        let session = session.clone();
        tokio::spawn(async move { session.prompt("work".to_owned()).await })
    };
    tokio::time::timeout(std::time::Duration::from_secs(5), hook.entered.notified())
        .await
        .expect("the reservation hook never ran");
    session.cancel().await.unwrap();
    hook.release.notify_one();

    let outcome = tokio::time::timeout(std::time::Duration::from_secs(10), driver)
        .await
        .expect("the turn outlived cancel during admission")
        .unwrap()
        .unwrap();
    assert_eq!(outcome.stop_reason, "cancelled");
    let calls_log = std::fs::read_to_string(&log).unwrap_or_default();
    assert!(
        !calls_log.lines().any(|method| method == "session/prompt"),
        "the prompt dispatched after cancel returned: {calls_log}"
    );
    let _ = session.close().await;
}

/// An rpc peer whose `session/close` shutdown call sleeps forever:
/// retiring the lane must not wait on the peer — the dead lane's
/// senders release and its process dies, so the caller's event stream
/// ends once the session is gone.
#[cfg(unix)]
#[tokio::test]
async fn retired_lane_releases_the_event_stream_and_terminates() {
    let dir = TestDirectory::new("rpc-retire-release");
    let boom = dir.path().join("rpc_boom.py");
    std::fs::write(&boom, RPC_BOOM_STALLED_CLOSE).unwrap();
    let mut primary = rpc_spec("boom", &boom, json!({}), json!({"text": "{prompt}"}));
    let LaneSpec::Rpc(lane) = &mut primary.lane else {
        panic!("rpc_spec builds an rpc lane")
    };
    lane.session_close = Some(RpcCallSpec {
        method: "session/close".to_owned(),
        params: json!({}),
    });
    let serving = write_rpc_agent(dir.path());
    let registry = registry_with_lanes(
        primary,
        rpc_spec("serving", &serving, json!({}), json!({"text": "{prompt}"})),
        ProfilePatch {
            agent: Some("boom".to_owned()),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    let outcome = session.prompt("work".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "unset|unset|work");
    session.close().await.unwrap();
    drop(session);

    tokio::time::timeout(std::time::Duration::from_secs(10), async move {
        let mut rx = rx;
        while rx.recv().await.is_some() {}
    })
    .await
    .expect("the retired lane's senders outlived its teardown");
}

/// An rpc peer with a model selector at both `session/open` and
/// `session/prompt`: it quota-refuses `prime` at open, logs the model
/// each call carried, and serves prompts with the model they name.
#[cfg(unix)]
const RPC_QUOTA_OPEN_DUAL_MODEL: &str = r#"
import json, os, sys

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

log = open(os.environ["AWP_CALL_LOG"], "a")
for line in sys.stdin:
    try:
        request = json.loads(line)
    except Exception:
        continue
    method = request.get("method")
    request_id = request.get("id")
    params = request.get("params") or {}
    if method == "session/open":
        model = params.get("model")
        log.write("session/open|{}\n".format(model or "unset"))
        log.flush()
        if model == "prime":
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
        else:
            send({"jsonrpc": "2.0", "id": request_id,
                  "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        log.write("session/prompt|{}\n".format(params.get("model") or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "served|{}".format(params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_dual_selector_switch_routes_the_candidate_to_the_turn() {
    // The retry scheduled `fallback` on the primary's per-turn surface,
    // then the prompt's write failed. The alternate selects the model at
    // both open and prompt — an actual open-time selector, not a stored
    // default: the pending candidate binds at `session/open`, where the
    // peer accepts it — restoring the exhausted `prime` there would be
    // quota-refused before the fallback could reach the prompt.
    let dir = TestDirectory::new("rpc-dual-selector");
    let script = dir.path().join("rpc_quota_close.py");
    std::fs::write(&script, RPC_QUOTA_THEN_CLOSES_STDIN).unwrap();
    let primary = rpc_spec(
        "unaccepted",
        &script,
        json!({}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    let dual = dir.path().join("rpc_dual.py");
    std::fs::write(&dual, RPC_QUOTA_OPEN_DUAL_MODEL).unwrap();
    let log = dir.path().join("calls.log");
    let mut alternate = rpc_spec(
        "dual",
        &dual,
        json!({"model": "{model}"}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    alternate.command.env = EnvPolicy::Sanitized {
        inherit: vec![],
        set: [(
            "AWP_CALL_LOG".to_owned(),
            EnvValue::Literal(log.display().to_string()),
        )]
        .into_iter()
        .collect(),
    };
    let registry = registry_with_lanes(
        primary,
        alternate,
        ProfilePatch {
            agent: Some("unaccepted".to_owned()),
            model: Some("prime".to_owned()),
            models: Some(vec!["fallback".to_owned()]),
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
    assert_eq!(outcome.text.trim_end(), "served|work");
    assert_eq!(session.active_model().as_deref(), Some("fallback"));
    let calls_log = std::fs::read_to_string(&log).unwrap_or_default();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec!["session/open|fallback", "session/prompt|fallback"],
        "the exhausted model must not be retried at open: {calls_log}"
    );
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn rpc_inline_envelope_overwritten_slot_still_marks_dispatch() {
    // The inline encoder merges object params into the frame root, then
    // stamps `cmd` with the method name — the templated `cmd` leaf never
    // reaches the wire, but `text` does: the peer received the prompt
    // through `text` and died, so the post-dispatch transport loss must
    // propagate instead of replaying a possibly billed turn.
    let dir = TestDirectory::new("rpc-inline-overwrite");
    let script = dir.path().join("rpc_inline.py");
    std::fs::write(&script, RPC_ENVELOPE_DIES_ON_PROMPT).unwrap();
    let envelope = a4aa::RpcEnvelopeSpec {
        method_field: "cmd".to_owned(),
        params_inline: true,
        response_method: Some("response".to_owned()),
        ..Default::default()
    };
    let registry = registry_with_lanes(
        rpc_envelope_spec(
            "inline",
            &script,
            json!({"cmd": "{prompt}", "text": "{prompt}"}),
            envelope,
        ),
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("inline".to_owned()),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    let error = session.prompt("work".to_owned()).await.unwrap_err();
    assert_eq!(error.code, "agent_transport_failed");
    let _ = session.close().await;
}

#[cfg(unix)]
#[tokio::test]
async fn cli_cancelled_turn_keeps_the_lane_outcome() {
    // The cli lane's own cancelled outcome carries the text the turn
    // already produced — the driver's synthesized empty outcome for
    // admission cancellation must not erase an in-flight lane result.
    let registry = registry_with(
        cli_spec(
            "slow",
            "sh",
            &["-c", "printf 'partial-output\\n'; sleep 60", "x|{prompt}"],
            PromptChannel::ArgOnly,
        ),
        ProfilePatch {
            agent: Some("slow".to_owned()),
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
        let turn = session.prompt("work".to_owned());
        tokio::pin!(turn);
        tokio::select! {
            outcome = &mut turn => panic!("turn finished before cancel: {outcome:?}"),
            _ = tokio::time::sleep(std::time::Duration::from_millis(500)) => {}
        }
        session.cancel().await.unwrap();
        turn.await.unwrap()
    };
    assert_eq!(outcome.stop_reason, "cancelled");
    assert!(
        outcome.text.contains("partial-output"),
        "the lane's cancelled outcome lost its partial text: {outcome:?}"
    );
    session.close().await.unwrap();
}

/// An rpc peer that takes one second to answer each prompt: a cancel
/// that lands mid-turn finds no cancel surface, and the turn completes
/// normally — an unsupported cancel must not rewrite the lane's result.
#[cfg(unix)]
const RPC_SLOW_PROMPT: &str = r#"
import json, sys, time

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

for line in sys.stdin:
    try:
        request = json.loads(line)
    except Exception:
        continue
    method = request.get("method")
    request_id = request.get("id")
    params = request.get("params") or {}
    if method == "session/open":
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        time.sleep(1)
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "answered|{}".format(params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn cancel_unsupported_lane_keeps_its_served_outcome() {
    // `cancel` reports `agent_cancel_unsupported` — the lane finished
    // the turn anyway. Its served result must reach the caller, not the
    // synthesized empty cancellation.
    let dir = TestDirectory::new("rpc-cancel-unsupported");
    let script = dir.path().join("rpc_slow.py");
    std::fs::write(&script, RPC_SLOW_PROMPT).unwrap();
    let registry = registry_with(
        rpc_spec("slow", &script, json!({}), json!({"text": "{prompt}"})),
        ProfilePatch {
            agent: Some("slow".to_owned()),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = std::sync::Arc::new(
        agent
            .open_session(SessionRequest::default(), tx)
            .await
            .unwrap(),
    );

    let driver = {
        let session = session.clone();
        tokio::spawn(async move { session.prompt("work".to_owned()).await })
    };
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    let error = session.cancel().await.unwrap_err();
    assert_eq!(error.code, "agent_cancel_unsupported");

    let outcome = tokio::time::timeout(std::time::Duration::from_secs(10), driver)
        .await
        .expect("the turn outlived its own completion")
        .unwrap()
        .unwrap();
    assert_eq!(outcome.text.trim_end(), "answered|work");
    session.close().await.unwrap();
}

// -------------------- release-repair regressions --------------------

/// An rpc peer that answers `session/open` after a delay but never
/// answers `session/close`: a replacement that finished opening just as
/// the caller closes or cancels must be discarded without awaiting its
/// teardown call — that call would stall the driver and the caller's
/// event stream forever.
#[cfg(unix)]
const RPC_SLOW_OPEN_STALL_CLOSE: &str = r#"
import json, os, sys, time

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

log = open(os.environ["AWP_CALL_LOG"], "a")
for line in sys.stdin:
    try:
        request = json.loads(line)
    except Exception:
        continue
    method = request.get("method")
    request_id = request.get("id")
    if method:
        log.write(method + "\n")
        log.flush()
    if method == "session/open":
        time.sleep(0.8)
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/close":
        time.sleep(600)
    elif method == "session/prompt":
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "alt|{}".format(
                  (request.get("params") or {}).get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

/// An rpc peer that logs the model and session id of `session/open` and
/// `session/resume` and serves prompts with the effective model — used to
/// prove which setup call a request actually selected.
#[cfg(unix)]
const RPC_RESUME_BIND_LOG: &str = r#"
import json, os, sys

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

log = open(os.environ["AWP_CALL_LOG"], "a")
session_model = None
for line in sys.stdin:
    try:
        request = json.loads(line)
    except Exception:
        continue
    method = request.get("method")
    request_id = request.get("id")
    params = request.get("params") or {}
    if method == "session/resume":
        session_model = params.get("model")
        log.write("session/resume|{}|{}\n".format(
            session_model, params.get("sessionId")))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": params.get("sessionId")}})
    elif method == "session/open":
        session_model = params.get("model")
        log.write("session/open|{}\n".format(session_model))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        model = params.get("model") or session_model
        log.write("session/prompt|{}\n".format(model))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "served|{}|{}".format(
                  model or "unset", params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

/// An rpc peer that sleeps a second, then answers its prompt with a
/// permanent error: a cancel that lands mid-turn finds no cancel surface
/// and the lane still completes — with its real failure.
#[cfg(unix)]
const RPC_SLOW_PROMPT_ERROR: &str = r#"
import json, sys, time

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

for line in sys.stdin:
    try:
        request = json.loads(line)
    except Exception:
        continue
    method = request.get("method")
    request_id = request.get("id")
    if method == "session/open":
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        time.sleep(1)
        send({"jsonrpc": "2.0", "id": request_id,
              "error": {"code": -32000, "message": "permanent boom"}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

/// Add a `session_resume` call carrying `{sessionId}` and `{model}` to an
/// rpc spec built by `rpc_spec`.
#[cfg(unix)]
fn rpc_add_resume(spec: &mut AgentSpec) {
    let LaneSpec::Rpc(lane) = &mut spec.lane else {
        panic!("rpc_spec builds an rpc lane")
    };
    lane.session_resume = Some(RpcCallSpec {
        method: "session/resume".to_owned(),
        params: json!({"sessionId": "{sessionId}", "model": "{model}"}),
    });
}

/// Attach the `AWP_CALL_LOG` environment to a spec so its pebble logs
/// every method it receives.
#[cfg(unix)]
fn attach_call_log(spec: &mut AgentSpec, log: &Path) {
    spec.command.env = EnvPolicy::Sanitized {
        inherit: vec![],
        set: [(
            "AWP_CALL_LOG".to_owned(),
            EnvValue::Literal(log.display().to_string()),
        )]
        .into_iter()
        .collect(),
    };
}

#[cfg(unix)]
#[tokio::test]
async fn rpc_resume_only_placeholder_is_a_stored_default_for_fresh_sessions() {
    // The alternate's only session-call model placeholder sits in
    // `session_resume` — a call a fresh session never makes. The carried
    // candidate must ride the replayed prompt, and the open stores only
    // the last accepted entry: a refused candidate cannot stick as the
    // session default for the next implicit prompt.
    let dir = TestDirectory::new("rpc-resume-only-fresh");
    let script = dir.path().join("rpc_quota_close.py");
    std::fs::write(&script, RPC_QUOTA_THEN_CLOSES_STDIN).unwrap();
    let primary = rpc_spec(
        "unaccepted",
        &script,
        json!({}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    let refusing = dir.path().join("rpc_refuse.py");
    std::fs::write(&refusing, RPC_REFUSES_MODEL_AT_PROMPT).unwrap();
    let log = dir.path().join("calls.log");
    let mut alternate = rpc_spec(
        "refusing",
        &refusing,
        json!({}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    rpc_add_resume(&mut alternate);
    attach_call_log(&mut alternate, &log);
    let registry = registry_with_lanes(
        primary,
        alternate,
        ProfilePatch {
            agent: Some("unaccepted".to_owned()),
            model: Some("prime".to_owned()),
            models: Some(vec!["fallback".to_owned()]),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    let error = session.prompt("work".to_owned()).await.unwrap_err();
    assert_eq!(error.code, "agent_quota_exhausted");
    assert_eq!(session.active_model().as_deref(), Some("prime"));

    // The refused candidate never became the adapter's persistent
    // default: the next implicit prompt rides the last accepted model —
    // `prime` — not the refused `fallback`.
    let error = session.prompt("again".to_owned()).await.unwrap_err();
    assert_eq!(error.code, "agent_quota_exhausted");
    assert_eq!(session.active_model().as_deref(), Some("prime"));
    let calls_log = std::fs::read_to_string(&log).unwrap_or_default();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "session/open|unset",
            "session/prompt|fallback",
            "session/prompt|prime",
        ],
        "a fresh session must not bind a resume-only placeholder: {calls_log}"
    );
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn rpc_resumed_session_binds_the_carried_model_through_session_resume() {
    // A resumed request selects `session_resume` as its setup call: the
    // carried candidate binds on that call — the wire shows
    // `session/resume` with the requested binding, never `session/open` —
    // and the bound candidate is the session's model.
    let dir = TestDirectory::new("rpc-resume-bind");
    let script = dir.path().join("rpc_quota_close.py");
    std::fs::write(&script, RPC_QUOTA_THEN_CLOSES_STDIN).unwrap();
    let mut primary = rpc_spec(
        "unaccepted",
        &script,
        json!({}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    rpc_add_resume(&mut primary);
    let resumed = dir.path().join("rpc_resume.py");
    std::fs::write(&resumed, RPC_RESUME_BIND_LOG).unwrap();
    let log = dir.path().join("calls.log");
    let mut alternate = rpc_spec("resuming", &resumed, json!({}), json!({"text": "{prompt}"}));
    rpc_add_resume(&mut alternate);
    attach_call_log(&mut alternate, &log);
    let registry = registry_with_lanes(
        primary,
        alternate,
        ProfilePatch {
            agent: Some("unaccepted".to_owned()),
            model: Some("prime".to_owned()),
            models: Some(vec!["fallback".to_owned()]),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(
            SessionRequest {
                resume: Some("sess-42".to_owned()),
                ..Default::default()
            },
            tx,
        )
        .await
        .unwrap();

    let outcome = session.prompt("work".to_owned()).await.unwrap();
    let calls_log = std::fs::read_to_string(&log).unwrap_or_default();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec!["session/resume|fallback|sess-42", "session/prompt|fallback"],
        "a resumed session must bind the model on session/resume: {calls_log}"
    );
    assert_eq!(outcome.text.trim_end(), "served|fallback|work");
    assert_eq!(session.active_model().as_deref(), Some("fallback"));
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn rpc_alternate_open_refusal_continues_the_model_chain() {
    // The primary loses transport while retrying `fallback`; the
    // alternate quota-refuses `fallback` at session/open. A verified
    // setup refusal is an answer about the model, not the lane — the
    // chain keeps walking on that lane and `tertiary` serves the turn.
    let dir = TestDirectory::new("rpc-alt-open-refusal");
    let script = dir.path().join("rpc_quota_close.py");
    std::fs::write(&script, RPC_QUOTA_THEN_CLOSES_STDIN).unwrap();
    let primary = rpc_spec(
        "dying",
        &script,
        json!({}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    let bound = dir.path().join("rpc_bound.py");
    std::fs::write(&bound, RPC_QUOTA_SESSION_OPEN_MODEL).unwrap();
    let log = dir.path().join("calls.log");
    let alternate = rpc_session_bound_spec("bound", &bound, &log);
    let registry = registry_with_lanes(
        primary,
        alternate,
        ProfilePatch {
            agent: Some("dying".to_owned()),
            model: Some("prime".to_owned()),
            models: Some(vec!["fallback".to_owned(), "tertiary".to_owned()]),
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
    assert_eq!(outcome.text.trim_end(), "served|tertiary|work");
    assert_eq!(session.active_model().as_deref(), Some("tertiary"));
    let calls_log = std::fs::read_to_string(&log).unwrap_or_default();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "session/open|fallback",
            "session/open|tertiary",
            "session/prompt|tertiary",
        ],
        "a refused alternate open must keep walking the chain: {calls_log}"
    );
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn rpc_bound_candidate_stays_active_when_its_replay_is_refused() {
    // The replacement binds `prime` at session/open and is installed; the
    // replay then quota-refuses it. The installed session genuinely holds
    // `prime` — `active_model` must agree with the binding the next
    // implicit prompt will keep sending, not the entry it replaced.
    let dir = TestDirectory::new("rpc-bound-active");
    let script = dir.path().join("rpc_quota_close.py");
    std::fs::write(&script, RPC_QUOTA_THEN_CLOSES_STDIN).unwrap();
    let primary = rpc_spec(
        "dying",
        &script,
        json!({}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    let bound = dir.path().join("rpc_bound.py");
    std::fs::write(&bound, RPC_QUOTA_SESSION_MODEL).unwrap();
    let log = dir.path().join("calls.log");
    let alternate = rpc_session_bound_spec("bound", &bound, &log);
    let registry = registry_with_lanes(
        primary,
        alternate,
        ProfilePatch {
            agent: Some("dying".to_owned()),
            model: Some("fallback".to_owned()),
            models: Some(vec!["prime".to_owned()]),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    let error = session.prompt("work".to_owned()).await.unwrap_err();
    assert_eq!(error.code, "agent_quota_exhausted");
    assert_eq!(session.active_model().as_deref(), Some("prime"));

    // The installed binding is still `prime`: the next implicit prompt
    // sends it and `active_model` keeps reporting it.
    let error = session.prompt("again".to_owned()).await.unwrap_err();
    assert_eq!(error.code, "agent_quota_exhausted");
    assert_eq!(session.active_model().as_deref(), Some("prime"));
    let calls_log = std::fs::read_to_string(&log).unwrap_or_default();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "session/open|prime",
            "session/prompt|prime",
            "session/prompt|prime",
        ],
        "the installed binding and active_model must agree: {calls_log}"
    );
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn cancel_unsupported_mid_prompt_preserves_the_lane_error() {
    // `cancel` reports `agent_cancel_unsupported` while the prompt is in
    // flight; the lane completes with a permanent error anyway. The
    // fallback branch must not dispatch the alternate nor rewrite that
    // error as an empty cancellation — the lane's real result stands.
    let dir = TestDirectory::new("rpc-cancel-error");
    let script = dir.path().join("rpc_slow_error.py");
    std::fs::write(&script, RPC_SLOW_PROMPT_ERROR).unwrap();
    let registry = registry_with_lanes(
        rpc_spec("slowboom", &script, json!({}), json!({"text": "{prompt}"})),
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("slowboom".to_owned()),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = std::sync::Arc::new(
        agent
            .open_session(SessionRequest::default(), tx)
            .await
            .unwrap(),
    );

    let driver = {
        let session = session.clone();
        tokio::spawn(async move { session.prompt("work".to_owned()).await })
    };
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    let error = session.cancel().await.unwrap_err();
    assert_eq!(error.code, "agent_cancel_unsupported");

    let error = tokio::time::timeout(std::time::Duration::from_secs(10), driver)
        .await
        .expect("the turn outlived its own completion")
        .unwrap()
        .unwrap_err();
    assert_eq!(error.code, "agent_remote_error");
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn close_after_replacement_opened_drops_the_replacement() {
    // The alternate's `session/open` resolves while the caller's `close`
    // is already landing: the install guard rejects the finished
    // replacement and discarding it must not await a `session/close` this
    // peer never answers — the turn keeps its failure and the caller's
    // event stream still ends.
    let dir = TestDirectory::new("rpc-close-opened-replacement");
    let boom = dir.path().join("rpc_boom.py");
    std::fs::write(&boom, RPC_BOOM).unwrap();
    let stall = dir.path().join("rpc_stall_close.py");
    std::fs::write(&stall, RPC_SLOW_OPEN_STALL_CLOSE).unwrap();
    let log = dir.path().join("calls.log");
    let mut alternate = rpc_logging_spec("stalling", &stall, &log);
    let LaneSpec::Rpc(lane) = &mut alternate.lane else {
        panic!("rpc_spec builds an rpc lane")
    };
    lane.session_close = Some(RpcCallSpec {
        method: "session/close".to_owned(),
        params: json!({}),
    });
    let registry = registry_with_lanes(
        rpc_spec("boom", &boom, json!({}), json!({"text": "{prompt}"})),
        alternate,
        ProfilePatch {
            agent: Some("boom".to_owned()),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, rx) = events();
    let session = std::sync::Arc::new(
        agent
            .open_session(SessionRequest::default(), tx)
            .await
            .unwrap(),
    );

    let driver = {
        let session = session.clone();
        tokio::spawn(async move { session.prompt("work".to_owned()).await })
    };
    // Let the driver reach the replacement select with the alternate's
    // open in flight, then freeze the (single-threaded) runtime so the
    // peer's open reply is buffered — resolved but never polled — before
    // `close` marks the session closing. After the freeze the open arm
    // and the lifecycle signal are both ready; the select resolves the
    // open and the install guard rejects it under the write lock.
    tokio::time::sleep(std::time::Duration::from_millis(400)).await;
    std::thread::sleep(std::time::Duration::from_millis(1500));
    session.close().await.unwrap();

    let error = tokio::time::timeout(std::time::Duration::from_secs(10), driver)
        .await
        .expect("the turn waited on the discarded replacement's close")
        .unwrap()
        .unwrap_err();
    assert_eq!(error.code, "agent_remote_error");
    let calls_log = std::fs::read_to_string(&log).unwrap_or_default();
    assert!(
        !calls_log.lines().any(|method| method == "session/prompt"),
        "the discarded replacement served a replay: {calls_log}"
    );
    drop(session);
    tokio::time::timeout(std::time::Duration::from_secs(10), async move {
        let mut rx = rx;
        while rx.recv().await.is_some() {}
    })
    .await
    .expect("the discarded replacement outlived the session");
}

#[cfg(unix)]
#[tokio::test]
async fn cancel_after_replacement_opened_drops_the_replacement() {
    // Same teardown on the cancel arm: the replacement finished opening
    // while the runtime was frozen and the cancel landed before it could
    // be installed — discarding it drops the process rather than awaiting
    // a `session/close` the peer never answers.
    let dir = TestDirectory::new("rpc-cancel-opened-replacement");
    let boom = dir.path().join("rpc_boom.py");
    std::fs::write(&boom, RPC_BOOM).unwrap();
    let stall = dir.path().join("rpc_stall_close.py");
    std::fs::write(&stall, RPC_SLOW_OPEN_STALL_CLOSE).unwrap();
    let log = dir.path().join("calls.log");
    let mut alternate = rpc_logging_spec("stalling", &stall, &log);
    let LaneSpec::Rpc(lane) = &mut alternate.lane else {
        panic!("rpc_spec builds an rpc lane")
    };
    lane.session_close = Some(RpcCallSpec {
        method: "session/close".to_owned(),
        params: json!({}),
    });
    let registry = registry_with_lanes(
        rpc_spec("boom", &boom, json!({}), json!({"text": "{prompt}"})),
        alternate,
        ProfilePatch {
            agent: Some("boom".to_owned()),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = std::sync::Arc::new(
        agent
            .open_session(SessionRequest::default(), tx)
            .await
            .unwrap(),
    );

    let driver = {
        let session = session.clone();
        tokio::spawn(async move { session.prompt("work".to_owned()).await })
    };
    tokio::time::sleep(std::time::Duration::from_millis(400)).await;
    std::thread::sleep(std::time::Duration::from_millis(1500));
    session.cancel().await.unwrap();

    let outcome = tokio::time::timeout(std::time::Duration::from_secs(10), driver)
        .await
        .expect("the turn waited on the discarded replacement's close")
        .unwrap()
        .unwrap();
    assert_eq!(outcome.stop_reason, "cancelled");
    let calls_log = std::fs::read_to_string(&log).unwrap_or_default();
    assert!(
        !calls_log.lines().any(|method| method == "session/prompt"),
        "the discarded replacement served a replay: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer that quota-refuses `session/open` outright and logs every
/// call it receives: the refusal carries no model because the call
/// receives none — the lane's only model placeholder lives in its prompt.
#[cfg(unix)]
const RPC_QUOTA_REFUSES_OPEN: &str = r#"
import json, os, sys

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

log = open(os.environ["AWP_CALL_LOG"], "a")
for line in sys.stdin:
    try:
        request = json.loads(line)
    except Exception:
        continue
    method = request.get("method")
    request_id = request.get("id")
    if method == "session/open":
        log.write("session/open\n")
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "error": {"code": -32000, "message":
                        "resource_exhausted: quota exceeded for this model"}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_open_quota_without_model_binding_walks_to_the_next_lane() {
    // The alternate's only model placeholder is in its prompt: the
    // `session/open` it fails carries no model, so no chain entry can
    // change that call. The refusal is the lane's answer — the walk must
    // continue to the healthy third lane instead of spending the chain
    // on identical retries against the same alternate.
    let dir = TestDirectory::new("rpc-open-quota-walk");
    let script = dir.path().join("rpc_quota_close.py");
    std::fs::write(&script, RPC_QUOTA_THEN_CLOSES_STDIN).unwrap();
    let primary = rpc_spec(
        "dying",
        &script,
        json!({}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    let refusing = dir.path().join("rpc_refuse_open.py");
    std::fs::write(&refusing, RPC_QUOTA_REFUSES_OPEN).unwrap();
    let refusing_log = dir.path().join("refusing.log");
    let mut alternate = rpc_spec(
        "refusing",
        &refusing,
        json!({}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut alternate, &refusing_log);
    let serving = dir.path().join("rpc_serving.py");
    std::fs::write(&serving, RPC_SESSION_MODEL_LOG).unwrap();
    let serving_log = dir.path().join("serving.log");
    let third = rpc_session_bound_spec("serving", &serving, &serving_log);
    let mut document = RegistryDocument::default();
    document.agents.insert(
        primary.id.clone(),
        awp::AgentEntry {
            spec: primary,
            alternates: vec![alternate, third],
        },
    );
    document.profiles.insert(
        "p".to_owned(),
        ProfilePatch {
            agent: Some("dying".to_owned()),
            model: Some("prime".to_owned()),
            models: Some(vec!["fallback".to_owned(), "tertiary".to_owned()]),
            ..Default::default()
        },
    );
    let registry = Registry::from_document(document).unwrap();
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    let outcome = session.prompt("work".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "alt|fallback|work");
    assert_eq!(session.active_model().as_deref(), Some("fallback"));
    let calls_log = std::fs::read_to_string(&refusing_log).unwrap_or_default();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec!["session/open"],
        "a setup call that carries no model must not consume the chain: {calls_log}"
    );
    let calls_log = std::fs::read_to_string(&serving_log).unwrap_or_default();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec!["session/open|fallback", "session/prompt|fallback"],
        "the third lane binds the carried candidate and serves: {calls_log}"
    );
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn rpc_fresh_open_never_projects_a_resume_only_model() {
    // The lane's only model placeholder sits in `session_resume` — a
    // call this fresh request never makes. Projecting the profile model
    // into session config anyway makes the adapter refuse the open with
    // `agent_model_unsupported`; with the selected-call classification
    // the lane opens cleanly and serves.
    let dir = TestDirectory::new("rpc-resume-only-open");
    let script = write_rpc_agent(dir.path());
    let mut spec = rpc_spec(
        "resume-only",
        &script,
        json!({}),
        json!({"text": "{prompt}"}),
    );
    rpc_add_resume(&mut spec);
    let registry = registry_with(
        spec,
        ProfilePatch {
            agent: Some("resume-only".to_owned()),
            model: Some("prime".to_owned()),
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
    assert_eq!(outcome.text.trim_end(), "unset|unset|work");
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn rpc_blank_resume_id_opens_fresh_and_skips_resume_only_model() {
    // An empty-of-content resume id is a fresh session to the adapter:
    // `session/open` is the setup call, so a `session_resume`-only model
    // placeholder must not read as support either — the same fresh
    // classification, the same clean open.
    let dir = TestDirectory::new("rpc-blank-resume-open");
    let script = write_rpc_agent(dir.path());
    let mut spec = rpc_spec(
        "blank-resume",
        &script,
        json!({}),
        json!({"text": "{prompt}"}),
    );
    rpc_add_resume(&mut spec);
    let registry = registry_with(
        spec,
        ProfilePatch {
            agent: Some("blank-resume".to_owned()),
            model: Some("prime".to_owned()),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(
            SessionRequest {
                resume: Some("   ".to_owned()),
                ..Default::default()
            },
            tx,
        )
        .await
        .unwrap();

    let outcome = session.prompt("work".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "unset|unset|work");
    session.close().await.unwrap();
}

/// An rpc lane that applies the model only inside its prompt: `prime` is
/// always quota-refused, `fallback` serves once then refuses, and
/// `tertiary` fails with a certain remote error — the lane failure that
/// walks the turn to the alternate with the candidate still pending.
#[cfg(unix)]
const RPC_TURN_MODEL_THEN_BOOM: &str = r#"
import json, sys

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

session_model = None
fallbacks = 0
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
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        model = params.get("model") or session_model
        if model == "tertiary":
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message": "permanent boom"}})
        elif model == "fallback":
            fallbacks += 1
            if fallbacks > 1:
                send({"jsonrpc": "2.0", "id": request_id,
                      "error": {"code": -32000, "message":
                                "resource_exhausted: quota exceeded for this model"}})
            else:
                send({"jsonrpc": "2.0", "id": request_id,
                      "result": {"text": "served|{}|{}".format(
                          model, params.get("text", ""))}})
        else:
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

/// An rpc peer with a model selector at both `session/open` and
/// `session/prompt`: it binds the open's model as the session default
/// and serves every model but `tertiary`, which its prompt quota-refuses.
#[cfg(unix)]
const RPC_DUAL_REFUSES_TERTIARY_AT_PROMPT: &str = r#"
import json, os, sys

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

log = open(os.environ["AWP_CALL_LOG"], "a")
session_model = None
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
        log.write("session/open|{}\n".format(session_model or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        model = params.get("model") or session_model
        log.write("session/prompt|{}\n".format(model or "unset"))
        log.flush()
        if model == "tertiary":
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
        else:
            send({"jsonrpc": "2.0", "id": request_id,
                  "result": {"text": "served|{}|{}".format(
                      model or "unset", params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_bound_install_clears_the_replaced_turn_default() {
    // Turn one accepts `fallback` on the primary's per-turn surface; turn
    // two's `tertiary` retry dies on the lane and the replacement binds
    // `tertiary` at `session/open` — the lane also selects models per
    // turn. The replay is quota-refused, so the stale `fallback` default
    // must not survive: the next implicit prompt sends the bound model,
    // not the replaced one `active_model` no longer reports.
    let dir = TestDirectory::new("rpc-bound-clears-default");
    let script = dir.path().join("rpc_turn_boom.py");
    std::fs::write(&script, RPC_TURN_MODEL_THEN_BOOM).unwrap();
    let primary = rpc_spec(
        "turnmodel",
        &script,
        json!({}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    let dual = dir.path().join("rpc_dual.py");
    std::fs::write(&dual, RPC_DUAL_REFUSES_TERTIARY_AT_PROMPT).unwrap();
    let log = dir.path().join("calls.log");
    let mut alternate = rpc_spec(
        "dual",
        &dual,
        json!({"model": "{model}"}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut alternate, &log);
    let registry = registry_with_lanes(
        primary,
        alternate,
        ProfilePatch {
            agent: Some("turnmodel".to_owned()),
            model: Some("prime".to_owned()),
            models: Some(vec!["fallback".to_owned(), "tertiary".to_owned()]),
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
    assert_eq!(outcome.text.trim_end(), "served|fallback|work");
    assert_eq!(session.active_model().as_deref(), Some("fallback"));

    let error = session.prompt("more".to_owned()).await.unwrap_err();
    assert_eq!(error.code, "agent_quota_exhausted");
    assert_eq!(session.active_model().as_deref(), Some("tertiary"));

    // The installed session holds `tertiary`: an implicit prompt must
    // ride that binding — the replaced `fallback` default cannot resend.
    let error = session.prompt("again".to_owned()).await.unwrap_err();
    assert_eq!(error.code, "agent_quota_exhausted");
    let calls_log = std::fs::read_to_string(&log).unwrap_or_default();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "session/open|tertiary",
            "session/prompt|tertiary",
            "session/prompt|tertiary",
        ],
        "the implicit prompt must not override the installed binding: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer that quota-refuses the `prime` model after a beat and
/// serves every other model — the refusal lands after `cancel` ran, so
/// the driver must not spend the fallback its retry never dispatched.
#[cfg(unix)]
const RPC_SLOW_QUOTA_ON_PRIME: &str = r#"
import json, os, sys, time

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

log = open(os.environ["AWP_CALL_LOG"], "a")
session_model = None
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
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        model = params.get("model") or session_model
        log.write("session/prompt|{}\n".format(model or "unset"))
        log.flush()
        if model == "prime":
            time.sleep(1)
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
        else:
            send({"jsonrpc": "2.0", "id": request_id,
                  "result": {"text": "served|{}|{}".format(
                      model or "unset", params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn cancel_during_quota_refusal_keeps_the_untried_model() {
    // `cancel` lands while the prompt is in flight; the lane answers
    // with a quota refusal anyway. The turn keeps the lane's real
    // refusal — and the chain entry its retry never dispatched stays in
    // the chain: the next turn still gets to try it.
    let dir = TestDirectory::new("rpc-cancel-unspent-model");
    let script = dir.path().join("rpc_slow_quota.py");
    std::fs::write(&script, RPC_SLOW_QUOTA_ON_PRIME).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "slowquota",
        &script,
        json!({}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut primary, &log);
    let registry = registry_with_lanes(
        primary,
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("slowquota".to_owned()),
            model: Some("prime".to_owned()),
            models: Some(vec!["fallback".to_owned()]),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = std::sync::Arc::new(
        agent
            .open_session(SessionRequest::default(), tx)
            .await
            .unwrap(),
    );

    let driver = {
        let session = session.clone();
        tokio::spawn(async move { session.prompt("work".to_owned()).await })
    };
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    let error = session.cancel().await.unwrap_err();
    assert_eq!(error.code, "agent_cancel_unsupported");

    let error = tokio::time::timeout(std::time::Duration::from_secs(10), driver)
        .await
        .expect("the turn outlived its own completion")
        .unwrap()
        .unwrap_err();
    assert_eq!(error.code, "agent_quota_exhausted");

    // `fallback` was never dispatched: it is still in the chain, so the
    // next turn's `prime` refusal retries it — and it serves.
    let outcome = session.prompt("more".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "served|fallback|more");
    let calls_log = std::fs::read_to_string(&log).unwrap_or_default();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "session/prompt|prime",
            "session/prompt|prime",
            "session/prompt|fallback",
        ],
        "the untried model must still be reachable: {calls_log}"
    );
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn rpc_initial_open_quota_without_binding_walks_to_next_lane() {
    // The primary's only model placeholder is in its prompt: the
    // `session/open` it fails carries no model, so retrying the
    // identical call cannot answer a quota refusal. The initial open
    // must not spend the chain on it — the walk moves straight to the
    // alternate, which binds the chain head and serves.
    let dir = TestDirectory::new("rpc-initial-open-quota");
    let refusing = dir.path().join("rpc_refuse_open.py");
    std::fs::write(&refusing, RPC_QUOTA_REFUSES_OPEN).unwrap();
    let refusing_log = dir.path().join("refusing.log");
    let mut primary = rpc_spec(
        "refusing",
        &refusing,
        json!({}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut primary, &refusing_log);
    let serving = dir.path().join("rpc_serving.py");
    std::fs::write(&serving, RPC_SESSION_MODEL_LOG).unwrap();
    let serving_log = dir.path().join("serving.log");
    let alternate = rpc_session_bound_spec("serving", &serving, &serving_log);
    let registry = registry_with_lanes(
        primary,
        alternate,
        ProfilePatch {
            agent: Some("refusing".to_owned()),
            model: Some("prime".to_owned()),
            models: Some(vec!["fallback".to_owned(), "tertiary".to_owned()]),
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
    assert_eq!(outcome.text.trim_end(), "alt|prime|work");
    assert_eq!(session.active_model().as_deref(), Some("prime"));
    let calls_log = std::fs::read_to_string(&refusing_log).unwrap_or_default();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec!["session/open"],
        "an initial open call that binds no model must not consume the chain: {calls_log}"
    );
    let calls_log = std::fs::read_to_string(&serving_log).unwrap_or_default();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec!["session/open|prime", "session/prompt|prime"],
        "the alternate binds the chain head and serves: {calls_log}"
    );
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn rpc_fresh_session_never_retries_through_a_resume_only_selector() {
    // The lane's only model placeholder sits in `session_resume` — a
    // call this fresh session never makes. The aggregate session
    // support still reports model, so a session-bound retry would
    // reopen the identical fresh session and replay the identical
    // prompt once per chain entry. With the selected setup call
    // deciding, the refusal propagates and the chain stays untouched.
    let dir = TestDirectory::new("rpc-resume-only-retry");
    let script = dir.path().join("rpc_refuse.py");
    std::fs::write(&script, RPC_REFUSES_MODEL_AT_PROMPT).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec("refusing", &script, json!({}), json!({"text": "{prompt}"}));
    rpc_add_resume(&mut primary);
    attach_call_log(&mut primary, &log);
    let registry = registry_with(
        primary,
        ProfilePatch {
            agent: Some("refusing".to_owned()),
            model: Some("prime".to_owned()),
            models: Some(vec!["fallback".to_owned()]),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    let error = session.prompt("work".to_owned()).await.unwrap_err();
    assert_eq!(error.code, "agent_quota_exhausted");
    assert_eq!(session.active_model().as_deref(), Some("prime"));

    // The refusal propagated without consuming `fallback`: the next
    // turn walks the same single prompt — never a pointless reopen.
    let error = session.prompt("again".to_owned()).await.unwrap_err();
    assert_eq!(error.code, "agent_quota_exhausted");
    assert_eq!(session.active_model().as_deref(), Some("prime"));
    let calls_log = std::fs::read_to_string(&log).unwrap_or_default();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "session/open|unset",
            "session/prompt|unset",
            "session/prompt|unset",
        ],
        "a fresh session must not spend the chain on a resume-only selector: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer whose `initialize` handshake sleeps before answering and
/// whose `session/open` binds the model: a cancel mid-initialize
/// interrupts the replacement open before the model-carrying call can
/// dispatch.
#[cfg(unix)]
const RPC_SLOW_INIT_SESSION_MODEL: &str = r#"
import json, os, sys, time

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

log = open(os.environ["AWP_CALL_LOG"], "a")
session_model = None
for line in sys.stdin:
    try:
        request = json.loads(line)
    except Exception:
        continue
    method = request.get("method")
    request_id = request.get("id")
    params = request.get("params") or {}
    if method == "initialize":
        log.write("initialize\n")
        log.flush()
        time.sleep(1.0)
        send({"jsonrpc": "2.0", "id": request_id, "result": {}})
    elif method == "session/open":
        session_model = params.get("model")
        log.write("session/open|{}\n".format(session_model))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        model = params.get("model") or session_model
        log.write("session/prompt|{}\n".format(model))
        log.flush()
        if model == "prime":
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
        else:
            send({"jsonrpc": "2.0", "id": request_id,
                  "result": {"text": "served|{}|{}".format(
                      model or "unset", params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn cancel_during_replacement_initialize_restores_the_untried_model() {
    // `prime` is refused at the prompt; the session-bound retry removes
    // `fallback` from the chain and re-opens this lane — but `cancel`
    // lands while the model-less `initialize` is still in flight,
    // before the model-carrying `session/open` can dispatch. The
    // untried candidate goes back to the chain: the next turn's retry
    // binds it and is served.
    let dir = TestDirectory::new("rpc-cancel-init-reopen");
    let script = dir.path().join("rpc_slow_init.py");
    std::fs::write(&script, RPC_SLOW_INIT_SESSION_MODEL).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_session_bound_spec("init-model", &script, &log);
    let LaneSpec::Rpc(lane) = &mut primary.lane else {
        panic!("rpc_spec builds an rpc lane")
    };
    lane.initialize = Some(RpcCallSpec {
        method: "initialize".to_owned(),
        params: json!({}),
    });
    let registry = registry_with_lanes(
        primary,
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("init-model".to_owned()),
            model: Some("prime".to_owned()),
            models: Some(vec!["fallback".to_owned()]),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = std::sync::Arc::new(
        agent
            .open_session(SessionRequest::default(), tx)
            .await
            .unwrap(),
    );

    let driver = {
        let session = session.clone();
        tokio::spawn(async move { session.prompt("work".to_owned()).await })
    };
    // The replacement's `initialize` answer is a second out; cancel once
    // the peer logged it — the model-carrying `session/open` is still a
    // full handshake away.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        let contents = std::fs::read_to_string(&log).unwrap_or_default();
        if contents.matches("initialize").count() >= 2 {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the replacement initialize never dispatched: {contents}"
        );
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    session.cancel().await.unwrap();

    let outcome = tokio::time::timeout(std::time::Duration::from_secs(10), driver)
        .await
        .expect("the turn outlived cancel during the replacement handshake")
        .unwrap()
        .unwrap();
    assert_eq!(outcome.stop_reason, "cancelled");

    // `fallback` was never dispatched on a model-carrying call: the next
    // turn's `prime` refusal retries it — and it serves.
    let outcome = session.prompt("more".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "served|fallback|more");
    assert_eq!(session.active_model().as_deref(), Some("fallback"));
    let calls_log = std::fs::read_to_string(&log).unwrap_or_default();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "initialize",
            "session/open|prime",
            "session/prompt|prime",
            "initialize",
            "session/prompt|prime",
            "initialize",
            "session/open|fallback",
            "session/prompt|fallback",
        ],
        "the cancelled handshake never carried the candidate: {calls_log}"
    );
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn cli_empty_error_outcome_never_publishes_the_candidate() {
    // The retry carried `fallback` onto the alternate's per-turn
    // surface, but the cli process exits before producing anything —
    // an `error` outcome with no text and no usage is the same
    // unserved answer the lane-fallback check reads. It must not
    // publish the candidate: `active_model` and the next implicit
    // prompt keep the last accepted entry.
    let dir = TestDirectory::new("cli-error-no-publish");
    let script = dir.path().join("rpc_quota_close.py");
    std::fs::write(&script, RPC_QUOTA_THEN_CLOSES_STDIN).unwrap();
    let primary = rpc_spec(
        "unaccepted",
        &script,
        json!({}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    let log = dir.path().join("models.log");
    let record = format!("echo \"$0\" >> {}; exit 1", log.display());
    let alternate = cli_spec(
        "fails",
        "sh",
        &["-c", &record, "{model}", "{prompt}"],
        PromptChannel::ArgOnly,
    );
    let registry = registry_with_lanes(
        primary,
        alternate,
        ProfilePatch {
            agent: Some("unaccepted".to_owned()),
            model: Some("prime".to_owned()),
            models: Some(vec!["fallback".to_owned()]),
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
    assert_eq!(outcome.stop_reason, "error");
    assert_eq!(session.active_model().as_deref(), Some("prime"));

    // The unserved candidate never became the default: the next
    // implicit prompt rides the last accepted model — `prime` — not
    // the `fallback` the empty error outcome would have published.
    let outcome = session.prompt("again".to_owned()).await.unwrap();
    assert_eq!(outcome.stop_reason, "error");
    assert_eq!(session.active_model().as_deref(), Some("prime"));
    let models_log = std::fs::read_to_string(&log).unwrap_or_default();
    assert_eq!(
        models_log.lines().collect::<Vec<_>>(),
        vec!["fallback", "prime"],
        "an unserved candidate must not publish as the session default: {models_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer whose `initialize` always refuses with provider quota
/// while its `session/open` binds the model: the refusal comes from a
/// call that never carried the model, so no chain entry can change it.
#[cfg(unix)]
const RPC_INIT_QUOTA_SESSION_MODEL_LOG: &str = r#"
import json, os, sys

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

log = open(os.environ["AWP_CALL_LOG"], "a")
session_model = None
for line in sys.stdin:
    try:
        request = json.loads(line)
    except Exception:
        continue
    method = request.get("method")
    request_id = request.get("id")
    params = request.get("params") or {}
    if method == "initialize":
        log.write("initialize\n")
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "error": {"code": -32000, "message":
                        "resource_exhausted: quota exceeded for this model"}})
    elif method == "session/open":
        session_model = params.get("model")
        log.write("session/open|{}\n".format(session_model or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        model = params.get("model") or session_model
        log.write("session/prompt|{}\n".format(model or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "alt|{}|{}".format(
                  model or "unset", params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

/// Build a session-bound rpc lane on `script` whose `initialize` carries
/// no model — the model-bearing setup call is `session/open`.
#[cfg(unix)]
fn rpc_init_bound_spec(id: &str, script: &Path, log: &Path) -> AgentSpec {
    let mut spec = rpc_session_bound_spec(id, script, log);
    let LaneSpec::Rpc(lane) = &mut spec.lane else {
        panic!("rpc_session_bound_spec builds an rpc lane")
    };
    lane.initialize = Some(RpcCallSpec {
        method: "initialize".to_owned(),
        params: json!({}),
    });
    spec
}

/// An rpc peer that answers `session/prompt` with a plain remote error —
/// a lane fault the model chain cannot explain, so the walk falls to the
/// next lane.
#[cfg(unix)]
const RPC_REMOTE_ERROR_AT_PROMPT: &str = r#"
import json, os, sys

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

log = open(os.environ["AWP_CALL_LOG"], "a")
for line in sys.stdin:
    try:
        request = json.loads(line)
    except Exception:
        continue
    method = request.get("method")
    request_id = request.get("id")
    params = request.get("params") or {}
    if method == "session/open":
        log.write("session/open|{}\n".format(params.get("model") or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        log.write("session/prompt|{}\n".format(params.get("model") or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "error": {"code": -32001, "message": "boom"}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

/// An rpc peer whose model-binding `session/open` logs the model then
/// sleeps before replying — the window a cancel can land in — and whose
/// `session/prompt` always refuses with provider quota.
#[cfg(unix)]
const RPC_SLOW_SESSION_OPEN_MODEL: &str = r#"
import json, os, sys, time

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

log = open(os.environ["AWP_CALL_LOG"], "a")
session_model = None
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
        log.write("session/open|{}\n".format(session_model or "unset"))
        log.flush()
        time.sleep(1.0)
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        model = params.get("model") or session_model
        log.write("session/prompt|{}\n".format(model or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "error": {"code": -32000, "message":
                        "resource_exhausted: quota exceeded for this model"}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

/// An rpc peer whose `session/prompt` emits a lone turn-start
/// notification — bookkeeping, not work — then settles an `error`
/// outcome with empty text and no usage.
#[cfg(unix)]
const RPC_BOOKKEEPING_THEN_ERROR: &str = r#"
import json, os, sys

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

log = open(os.environ["AWP_CALL_LOG"], "a")
for line in sys.stdin:
    try:
        request = json.loads(line)
    except Exception:
        continue
    method = request.get("method")
    request_id = request.get("id")
    params = request.get("params") or {}
    if method == "session/open":
        log.write("session/open|{}\n".format(params.get("model") or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        log.write("session/prompt|{}\n".format(params.get("model") or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "method": "turn/start",
              "params": {"turnId": "t1"}})
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "", "status": "error"}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_initial_initialize_quota_walks_to_next_lane() {
    // The primary's `session/open` binds the model, but its `initialize`
    // — a call that never carried the model — answers quota first. The
    // refusal came from a model-free call: spending chain entries
    // re-sends the identical `initialize`, and exhausting the chain
    // would report provider exhaustion without ever trying the healthy
    // alternate. The wire journal names the failing call, so the walk
    // moves straight on.
    let dir = TestDirectory::new("rpc-init-quota-open");
    let refusing = dir.path().join("rpc_init_quota.py");
    std::fs::write(&refusing, RPC_INIT_QUOTA_SESSION_MODEL_LOG).unwrap();
    let refusing_log = dir.path().join("refusing.log");
    let primary = rpc_init_bound_spec("init-quota", &refusing, &refusing_log);
    let serving = dir.path().join("rpc_serving.py");
    std::fs::write(&serving, RPC_SESSION_MODEL_LOG).unwrap();
    let serving_log = dir.path().join("serving.log");
    let alternate = rpc_session_bound_spec("serving", &serving, &serving_log);
    let registry = registry_with_lanes(
        primary,
        alternate,
        ProfilePatch {
            agent: Some("init-quota".to_owned()),
            model: Some("prime".to_owned()),
            models: Some(vec!["fallback".to_owned(), "tertiary".to_owned()]),
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
    assert_eq!(outcome.text.trim_end(), "alt|prime|work");
    assert_eq!(session.active_model().as_deref(), Some("prime"));
    let calls_log = std::fs::read_to_string(&refusing_log).unwrap_or_default();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec!["initialize"],
        "a model-free initialize refusal must not consume the chain: {calls_log}"
    );
    let calls_log = std::fs::read_to_string(&serving_log).unwrap_or_default();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec!["session/open|prime", "session/prompt|prime"],
        "the alternate binds the chain head and serves: {calls_log}"
    );
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn rpc_replacement_initialize_quota_walks_to_next_lane() {
    // The same model-free-call rule on a fallback's replacement open: the
    // alternate's `initialize` refuses quota while its `session/open`
    // binds the model. That refusal is the alternate's answer — the walk
    // must reach the third lane instead of burning the chain on
    // identical `initialize` retries and reporting exhaustion.
    let dir = TestDirectory::new("rpc-reopen-init-quota");
    let boom = dir.path().join("rpc_boom.py");
    std::fs::write(&boom, RPC_REMOTE_ERROR_AT_PROMPT).unwrap();
    let boom_log = dir.path().join("boom.log");
    let mut primary = rpc_spec(
        "boom",
        &boom,
        json!({}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut primary, &boom_log);
    let refusing = dir.path().join("rpc_init_quota.py");
    std::fs::write(&refusing, RPC_INIT_QUOTA_SESSION_MODEL_LOG).unwrap();
    let refusing_log = dir.path().join("refusing.log");
    let alternate = rpc_init_bound_spec("refusing", &refusing, &refusing_log);
    let serving = dir.path().join("rpc_serving.py");
    std::fs::write(&serving, RPC_SESSION_MODEL_LOG).unwrap();
    let serving_log = dir.path().join("serving.log");
    let third = rpc_session_bound_spec("serving", &serving, &serving_log);
    let mut document = RegistryDocument::default();
    document.agents.insert(
        primary.id.clone(),
        awp::AgentEntry {
            spec: primary,
            alternates: vec![alternate, third],
        },
    );
    document.profiles.insert(
        "p".to_owned(),
        ProfilePatch {
            agent: Some("boom".to_owned()),
            model: Some("prime".to_owned()),
            models: Some(vec!["fallback".to_owned()]),
            ..Default::default()
        },
    );
    let registry = Registry::from_document(document).unwrap();
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    let outcome = session.prompt("work".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "alt|prime|work");
    assert_eq!(session.active_model().as_deref(), Some("prime"));
    let calls_log = std::fs::read_to_string(&refusing_log).unwrap_or_default();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec!["initialize"],
        "a model-free initialize refusal must not consume the chain: {calls_log}"
    );
    let calls_log = std::fs::read_to_string(&serving_log).unwrap_or_default();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec!["session/open|prime", "session/prompt|prime"],
        "the third lane binds the chain head and serves: {calls_log}"
    );
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn cancel_after_bound_setup_sent_keeps_the_candidate_spent() {
    // `prime` is refused at the prompt; the session-bound retry removes
    // `fallback` from the chain and re-opens — the peer receives the
    // model-carrying `session/open` and sleeps before replying. `cancel`
    // lands in that window: the candidate already left on the wire, so it
    // is spent — restoring it would let the next turn resend the same
    // setup call.
    let dir = TestDirectory::new("rpc-cancel-sent-setup");
    let script = dir.path().join("rpc_slow_open.py");
    std::fs::write(&script, RPC_SLOW_SESSION_OPEN_MODEL).unwrap();
    let log = dir.path().join("calls.log");
    let primary = rpc_session_bound_spec("slowopen", &script, &log);
    let registry = registry_with(
        primary,
        ProfilePatch {
            agent: Some("slowopen".to_owned()),
            model: Some("prime".to_owned()),
            models: Some(vec!["fallback".to_owned()]),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = std::sync::Arc::new(
        agent
            .open_session(SessionRequest::default(), tx)
            .await
            .unwrap(),
    );

    let driver = {
        let session = session.clone();
        tokio::spawn(async move { session.prompt("work".to_owned()).await })
    };
    // Wait until the peer logged the model-bound `session/open` — the
    // `Sent` record precedes the write, so a logged receipt proves the
    // candidate already left on the wire. The reply is a second out.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        let contents = std::fs::read_to_string(&log).unwrap_or_default();
        if contents.contains("session/open|fallback") {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the model-bound session/open never dispatched: {contents}"
        );
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    session.cancel().await.unwrap();

    let outcome = tokio::time::timeout(std::time::Duration::from_secs(10), driver)
        .await
        .expect("the turn outlived cancel during the replacement open")
        .unwrap()
        .unwrap();
    assert_eq!(outcome.stop_reason, "cancelled");

    // The candidate was already sent: it stays spent, so the next turn's
    // `prime` refusal has no chain entry left to resend it.
    let error = session.prompt("more".to_owned()).await.unwrap_err();
    assert_eq!(error.code, "agent_quota_exhausted");
    let calls_log = std::fs::read_to_string(&log).unwrap_or_default();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "session/open|prime",
            "session/prompt|prime",
            "session/open|fallback",
            "session/prompt|prime",
        ],
        "a sent candidate must not be restored and resent: {calls_log}"
    );
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn rpc_bookkeeping_only_turn_never_publishes_or_blocks_the_walk() {
    // The carried candidate's retry emits only a turn-start notification
    // and then settles an `error` outcome with empty text, no truncation
    // and no usage. Bookkeeping wire activity is not served work: the
    // candidate must not publish as `active_model`, and the same reading
    // must keep the lane walk moving to a lane that actually serves.
    let dir = TestDirectory::new("rpc-bookkeeping-error");
    let dying = dir.path().join("rpc_quota_close.py");
    std::fs::write(&dying, RPC_QUOTA_THEN_CLOSES_STDIN).unwrap();
    let primary = rpc_spec(
        "dying",
        &dying,
        json!({}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    let bookkeeping = dir.path().join("rpc_bookkeeping.py");
    std::fs::write(&bookkeeping, RPC_BOOKKEEPING_THEN_ERROR).unwrap();
    let bookkeeping_log = dir.path().join("bookkeeping.log");
    let mut alternate = rpc_spec(
        "bookkeeping",
        &bookkeeping,
        json!({}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut alternate, &bookkeeping_log);
    let serving = dir.path().join("rpc_serving.py");
    std::fs::write(&serving, RPC_SESSION_MODEL_LOG).unwrap();
    let serving_log = dir.path().join("serving.log");
    let third = rpc_session_bound_spec("serving", &serving, &serving_log);
    let mut document = RegistryDocument::default();
    document.agents.insert(
        primary.id.clone(),
        awp::AgentEntry {
            spec: primary,
            alternates: vec![alternate, third],
        },
    );
    document.profiles.insert(
        "p".to_owned(),
        ProfilePatch {
            agent: Some("dying".to_owned()),
            model: Some("prime".to_owned()),
            models: Some(vec!["fallback".to_owned()]),
            ..Default::default()
        },
    );
    let registry = Registry::from_document(document).unwrap();
    let agent = registry.resolve("p").unwrap();
    let (tx, mut rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    let outcome = session.prompt("work".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "alt|fallback|work");
    assert_eq!(session.active_model().as_deref(), Some("fallback"));
    let mut forwarded = Vec::new();
    while let Ok(event) = rx.try_recv() {
        forwarded.push(event);
    }
    assert!(
        forwarded.iter().any(|event| matches!(
            event,
            AgentEvent::Raw { raw }
                if raw.get("method").and_then(|method| method.as_str())
                    == Some("turn/start")
        )),
        "the bookkeeping notification reached the caller: {forwarded:?}"
    );
    let calls_log = std::fs::read_to_string(&bookkeeping_log).unwrap_or_default();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec!["session/open|unset", "session/prompt|fallback"],
        "the bookkeeping-only retry is not served work: {calls_log}"
    );
    let calls_log = std::fs::read_to_string(&serving_log).unwrap_or_default();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec!["session/open|fallback", "session/prompt|fallback"],
        "the walk continues to a lane that serves the candidate: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An enveloped rpc peer whose `initialize` and `session/open` share one
/// wire command: a `negotiate` frame carrying a `model` member is the
/// open, a `negotiate` frame without one is the handshake — and only the
/// handshake refuses quota.
#[cfg(unix)]
const RPC_ENVELOPE_SHARED_METHOD: &str = r#"
import json, os, sys

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

log = open(os.environ["AWP_CALL_LOG"], "a")
session_model = None
for line in sys.stdin:
    try:
        request = json.loads(line)
    except Exception:
        continue
    method = request.get("cmd")
    request_id = request.get("id")
    params = request.get("params") or {}
    if method == "negotiate" and "model" in params:
        session_model = params.get("model")
        log.write("negotiate-open|{}\n".format(session_model or "unset"))
        log.flush()
        send({"cmd": "response", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "negotiate":
        log.write("negotiate-init\n")
        log.flush()
        send({"cmd": "response", "id": request_id,
              "error": {"code": -32000, "message":
                        "resource_exhausted: quota exceeded for this model"}})
    elif method == "session/prompt":
        model = params.get("model") or session_model
        log.write("session/prompt|{}\n".format(model or "unset"))
        log.flush()
        send({"cmd": "response", "id": request_id,
              "result": {"text": "alt|{}|{}".format(
                  model or "unset", params.get("text", ""))}})
    elif request_id is not None:
        send({"cmd": "response", "id": request_id, "result": None})
"#;

/// An rpc peer that interleaves a server-initiated callback inside the
/// model-bearing `session/open`: it parks a `prime` open, issues
/// `peer/lookup`, and only once the host's callback reply arrives does it
/// refuse the parked call with provider quota.
#[cfg(unix)]
const RPC_CALLBACK_THEN_OPEN_QUOTA: &str = r#"
import json, os, sys

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

log = open(os.environ["AWP_CALL_LOG"], "a")
session_model = None
parked_open = None
for line in sys.stdin:
    try:
        request = json.loads(line)
    except Exception:
        continue
    if "method" not in request:
        if parked_open is not None and request.get("id") == "awp-cb":
            send({"jsonrpc": "2.0", "id": parked_open,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
            parked_open = None
        continue
    method = request.get("method")
    request_id = request.get("id")
    params = request.get("params") or {}
    if method == "session/open":
        session_model = params.get("model")
        log.write("session/open|{}\n".format(session_model or "unset"))
        log.flush()
        if session_model == "prime":
            parked_open = request_id
            send({"jsonrpc": "2.0", "id": "awp-cb",
                  "method": "peer/lookup", "params": {}})
        else:
            send({"jsonrpc": "2.0", "id": request_id,
                  "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        model = params.get("model") or session_model
        log.write("session/prompt|{}\n".format(model or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "served|{}|{}".format(
                  model or "unset", params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

/// An rpc peer whose `initialize` carries the model: it answers the
/// first `session/open`, refuses every later one with provider quota,
/// and refuses `session/prompt` the same way.
#[cfg(unix)]
const RPC_INIT_MODEL_THEN_QUOTA_OPEN: &str = r#"
import json, os, sys

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

log = open(os.environ["AWP_CALL_LOG"], "a")
log_path = os.environ["AWP_CALL_LOG"]
session_model = None
for line in sys.stdin:
    try:
        request = json.loads(line)
    except Exception:
        continue
    method = request.get("method")
    request_id = request.get("id")
    params = request.get("params") or {}
    if method == "initialize":
        session_model = params.get("model")
        log.write("initialize|{}\n".format(session_model or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id, "result": {}})
    elif method == "session/open":
        log.write("session/open\n")
        log.flush()
        # A replacement open is a fresh process, so the count lives in
        # the log itself: only the lane's first ever open is answered.
        seen = sum(
            1 for entry in open(log_path) if entry.startswith("session/open")
        )
        if seen == 1:
            send({"jsonrpc": "2.0", "id": request_id,
                  "result": {"sessionId": "awp-rpc-session"}})
        else:
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
    elif method == "session/prompt":
        model = params.get("model") or session_model
        log.write("session/prompt|{}\n".format(model or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "error": {"code": -32000, "message":
                        "resource_exhausted: quota exceeded for this model"}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

/// An rpc peer whose `session/prompt` emits bookkeeping and empty
/// billable-looking signals — a turn-start, an empty text delta, an
/// all-zero usage notification — then settles an `error` outcome whose
/// own usage record is all zero.
#[cfg(unix)]
const RPC_EMPTY_SIGNALS_THEN_ERROR: &str = r#"
import json, os, sys

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

log = open(os.environ["AWP_CALL_LOG"], "a")
for line in sys.stdin:
    try:
        request = json.loads(line)
    except Exception:
        continue
    method = request.get("method")
    request_id = request.get("id")
    params = request.get("params") or {}
    if method == "session/open":
        log.write("session/open|{}\n".format(params.get("model") or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        log.write("session/prompt|{}\n".format(params.get("model") or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "method": "turn/start",
              "params": {"turnId": "t1"}})
        send({"jsonrpc": "2.0", "method": "turn/delta",
              "params": {"text": ""}})
        send({"jsonrpc": "2.0", "method": "turn/usage",
              "params": {"usage": {"inputTokens": 0, "outputTokens": 0}}})
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "", "status": "error",
                         "usage": {"inputTokens": 0, "outputTokens": 0}}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_shared_method_initialize_quota_walks_to_next_lane() {
    // `initialize` and `session/open` ride one wire command — only the
    // params distinguish them. The model-free handshake refuses quota:
    // a method-name reading would call the refused call model-carrying
    // (the shared command does bind the model at the open) and burn the
    // chain on identical `initialize` retries. The journaled request id
    // names the call exactly, under a custom envelope and string ids.
    let dir = TestDirectory::new("rpc-shared-init-quota");
    let refusing = dir.path().join("rpc_shared_quota.py");
    std::fs::write(&refusing, RPC_ENVELOPE_SHARED_METHOD).unwrap();
    let refusing_log = dir.path().join("refusing.log");
    let mut primary = rpc_spec(
        "shared",
        &refusing,
        json!({"model": "{model}"}),
        json!({"text": "{prompt}"}),
    );
    {
        let LaneSpec::Rpc(lane) = &mut primary.lane else {
            panic!("rpc_spec builds an rpc lane")
        };
        lane.initialize = Some(RpcCallSpec {
            method: "negotiate".to_owned(),
            params: json!({}),
        });
        lane.session_open = Some(RpcCallSpec {
            method: "negotiate".to_owned(),
            params: json!({"model": "{model}"}),
        });
        lane.envelope = Some(a4aa::RpcEnvelopeSpec {
            method_field: "cmd".to_owned(),
            string_ids: true,
            response_method: Some("response".to_owned()),
            ..Default::default()
        });
    }
    attach_call_log(&mut primary, &refusing_log);
    let serving = dir.path().join("rpc_serving.py");
    std::fs::write(&serving, RPC_SESSION_MODEL_LOG).unwrap();
    let serving_log = dir.path().join("serving.log");
    let alternate = rpc_session_bound_spec("serving", &serving, &serving_log);
    let registry = registry_with_lanes(
        primary,
        alternate,
        ProfilePatch {
            agent: Some("shared".to_owned()),
            model: Some("prime".to_owned()),
            models: Some(vec!["fallback".to_owned(), "tertiary".to_owned()]),
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
    assert_eq!(outcome.text.trim_end(), "alt|prime|work");
    assert_eq!(session.active_model().as_deref(), Some("prime"));
    let calls_log = std::fs::read_to_string(&refusing_log).unwrap_or_default();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec!["negotiate-init"],
        "a model-free shared-method refusal must not consume the chain: {calls_log}"
    );
    let calls_log = std::fs::read_to_string(&serving_log).unwrap_or_default();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec!["session/open|prime", "session/prompt|prime"],
        "the alternate binds the chain head and serves: {calls_log}"
    );
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn rpc_callback_before_open_refusal_still_serves_the_candidate() {
    // The peer parks the model-bearing `session/open`, issues a callback
    // request, and refuses the open only after the host's callback reply
    // left the wire. That outbound reply must not clear the refusal's
    // evidence — it is a response frame, never a setup request — so the
    // journaled request id still names the model-carrying call and the
    // chain advances on this same lane.
    let dir = TestDirectory::new("rpc-callback-open-quota");
    let script = dir.path().join("rpc_callback_quota.py");
    std::fs::write(&script, RPC_CALLBACK_THEN_OPEN_QUOTA).unwrap();
    let log = dir.path().join("calls.log");
    let primary = rpc_session_bound_spec("callback", &script, &log);
    let registry = registry_with(
        primary,
        ProfilePatch {
            agent: Some("callback".to_owned()),
            model: Some("prime".to_owned()),
            models: Some(vec!["fallback".to_owned()]),
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
    assert_eq!(outcome.text.trim_end(), "served|fallback|work");
    assert_eq!(session.active_model().as_deref(), Some("fallback"));
    let calls_log = std::fs::read_to_string(&log).unwrap_or_default();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "session/open|prime",
            "session/open|fallback",
            "session/prompt|fallback",
        ],
        "the callback-interleaved refusal still advances the chain: {calls_log}"
    );
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn cancel_after_model_sent_on_earlier_lane_keeps_the_candidate_spent() {
    // `prime` is refused at the prompt; the session-bound retry removes
    // `fallback` and re-opens — this lane's `initialize` puts the
    // candidate on the wire, then its model-free `session/open` refuses
    // and the walk advances. The cancel lands while the next lane's
    // model-free `initialize` still sleeps: that lane's probe has no
    // send record, but the earlier attempt already carried the
    // candidate, so it stays spent instead of riding a restore.
    let dir = TestDirectory::new("rpc-cancel-earlier-lane");
    let first = dir.path().join("rpc_init_model_quota.py");
    std::fs::write(&first, RPC_INIT_MODEL_THEN_QUOTA_OPEN).unwrap();
    let first_log = dir.path().join("first.log");
    let mut primary = rpc_spec("initmodel", &first, json!({}), json!({"text": "{prompt}"}));
    {
        let LaneSpec::Rpc(lane) = &mut primary.lane else {
            panic!("rpc_spec builds an rpc lane")
        };
        lane.initialize = Some(RpcCallSpec {
            method: "initialize".to_owned(),
            params: json!({"model": "{model}"}),
        });
    }
    attach_call_log(&mut primary, &first_log);
    let second = dir.path().join("rpc_slow_init.py");
    std::fs::write(&second, RPC_SLOW_INIT_SESSION_MODEL).unwrap();
    let second_log = dir.path().join("second.log");
    let mut alternate = rpc_session_bound_spec("slowinit", &second, &second_log);
    {
        let LaneSpec::Rpc(lane) = &mut alternate.lane else {
            panic!("rpc_session_bound_spec builds an rpc lane")
        };
        lane.initialize = Some(RpcCallSpec {
            method: "initialize".to_owned(),
            params: json!({}),
        });
    }
    let registry = registry_with_lanes(
        primary,
        alternate,
        ProfilePatch {
            agent: Some("initmodel".to_owned()),
            model: Some("prime".to_owned()),
            models: Some(vec!["fallback".to_owned()]),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = std::sync::Arc::new(
        agent
            .open_session(SessionRequest::default(), tx)
            .await
            .unwrap(),
    );

    let driver = {
        let session = session.clone();
        tokio::spawn(async move { session.prompt("work".to_owned()).await })
    };
    // Wait until the next lane logged its model-free `initialize` — the
    // candidate's earlier `initialize|fallback` send is behind it, so a
    // restore at this point could only come from the fresh probe's empty
    // record, never from evidence that the wire carried the model.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        let contents = std::fs::read_to_string(&second_log).unwrap_or_default();
        if contents.contains("initialize") {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the next lane's initialize never ran: {contents} | first: {}",
            std::fs::read_to_string(&first_log).unwrap_or_default()
        );
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    session.cancel().await.unwrap();

    let outcome = tokio::time::timeout(std::time::Duration::from_secs(10), driver)
        .await
        .expect("the turn outlived cancel during the next lane's open")
        .unwrap()
        .unwrap();
    assert_eq!(outcome.stop_reason, "cancelled");

    // The earlier lane already sent the candidate: it stays spent, so the
    // next turn's `prime` refusal has no chain entry left to resend it.
    let error = session.prompt("more".to_owned()).await.unwrap_err();
    assert_eq!(error.code, "agent_quota_exhausted");
    let calls_log = std::fs::read_to_string(&first_log).unwrap_or_default();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "initialize|prime",
            "session/open",
            "session/prompt|prime",
            "initialize|fallback",
            "session/open",
            "session/prompt|prime",
        ],
        "a candidate sent on an earlier lane must not be restored and resent: {calls_log}"
    );
    let calls_log = std::fs::read_to_string(&second_log).unwrap_or_default();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec!["initialize"],
        "the abandoned open stopped at its model-free initialize: {calls_log}"
    );
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn rpc_empty_signals_turn_never_publishes_or_blocks_the_walk() {
    // The carried candidate's retry emits bookkeeping plus signals that
    // only look billable — an empty text delta and an all-zero usage
    // notification — then settles an `error` outcome whose usage record
    // is all zero too. None of it is served work: the candidate must not
    // publish, and the same reading must keep the walk moving to a lane
    // that actually serves.
    let dir = TestDirectory::new("rpc-empty-signals-error");
    let dying = dir.path().join("rpc_quota_close.py");
    std::fs::write(&dying, RPC_QUOTA_THEN_CLOSES_STDIN).unwrap();
    let primary = rpc_spec(
        "dying",
        &dying,
        json!({}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    let bookkeeping = dir.path().join("rpc_empty_signals.py");
    std::fs::write(&bookkeeping, RPC_EMPTY_SIGNALS_THEN_ERROR).unwrap();
    let bookkeeping_log = dir.path().join("bookkeeping.log");
    let mut alternate = rpc_spec(
        "bookkeeping",
        &bookkeeping,
        json!({}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut alternate, &bookkeeping_log);
    let serving = dir.path().join("rpc_serving.py");
    std::fs::write(&serving, RPC_SESSION_MODEL_LOG).unwrap();
    let serving_log = dir.path().join("serving.log");
    let third = rpc_session_bound_spec("serving", &serving, &serving_log);
    let mut document = RegistryDocument::default();
    document.agents.insert(
        primary.id.clone(),
        awp::AgentEntry {
            spec: primary,
            alternates: vec![alternate, third],
        },
    );
    document.profiles.insert(
        "p".to_owned(),
        ProfilePatch {
            agent: Some("dying".to_owned()),
            model: Some("prime".to_owned()),
            models: Some(vec!["fallback".to_owned()]),
            ..Default::default()
        },
    );
    let registry = Registry::from_document(document).unwrap();
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    let outcome = session.prompt("work".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "alt|fallback|work");
    assert_eq!(session.active_model().as_deref(), Some("fallback"));
    let calls_log = std::fs::read_to_string(&bookkeeping_log).unwrap_or_default();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec!["session/open|unset", "session/prompt|fallback"],
        "empty signals must not hold the walk on the lane: {calls_log}"
    );
    let calls_log = std::fs::read_to_string(&serving_log).unwrap_or_default();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec!["session/open|fallback", "session/prompt|fallback"],
        "the third lane binds the candidate and serves: {calls_log}"
    );
    session.close().await.unwrap();
}
