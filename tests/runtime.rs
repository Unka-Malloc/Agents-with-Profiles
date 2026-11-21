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
