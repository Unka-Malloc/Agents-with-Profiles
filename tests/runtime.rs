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
# stdin closes before the answer flushes, so a prompt the client writes
# after the open resolves is guaranteed to land on a dead pipe — never
# in the gap where the transport could still accept it.
os.close(0)
send({"jsonrpc": "2.0", "id": request["id"],
      "result": {"sessionId": "awp-rpc-session"}})
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
    let ready = tokio::time::timeout(std::time::Duration::from_secs(20), rx.recv())
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
    let idle = tokio::time::timeout(std::time::Duration::from_secs(20), rx.recv())
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
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
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
    tokio::time::timeout(std::time::Duration::from_secs(20), hook.entered.notified())
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
    let dir = TestDirectory::new("cli-cancel-partial");
    let marker = dir.path().join("printed.marker");
    let registry = registry_with(
        cli_spec(
            "slow",
            "sh",
            &[
                "-c",
                &format!(
                    "printf 'partial-output\\n'; touch {}; sleep 60",
                    marker.display()
                ),
                "x|{prompt}",
            ],
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
        // Wait until the child has actually emitted its partial output —
        // the marker is written after the printf, so the text is on the
        // pipe before cancel lands.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        loop {
            tokio::select! {
                outcome = &mut turn => panic!("turn finished before cancel: {outcome:?}"),
                _ = tokio::time::sleep(std::time::Duration::from_millis(10)) => {}
            }
            if marker.exists() {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "the lane never emitted its partial output"
            );
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
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
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
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
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
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
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

/// An inline-params envelope whose `session/open` and `session/prompt`
/// params are scalars — the model and the prompt text travel as the bare
/// `params` value, so neither params document offers object positions.
#[cfg(unix)]
const RPC_INLINE_SCALAR_SETUP: &str = r#"
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
    method = request.get("cmd")
    request_id = request.get("id")
    params = request.get("params")
    if method == "session/open":
        session_model = params if isinstance(params, str) else None
        send({"cmd": "response", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        text = params if isinstance(params, str) else ""
        send({"cmd": "response", "id": request_id,
              "result": {"text": "served|{}|{}".format(
                  session_model or "unset", text)}})
    elif request_id is not None:
        send({"cmd": "response", "id": request_id, "result": None})
"#;

/// An rpc peer that shares `negotiate` between a fixed-model
/// `initialize` handshake and the model-bound `session/open`: the
/// handshake is quota-refused while the open call would serve.
#[cfg(unix)]
const RPC_FIXED_MODEL_INIT_SHARED_METHOD: &str = r#"
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
    if method == "negotiate":
        model = params.get("model")
        if model == "handshake":
            log.write("negotiate|handshake\n")
            log.flush()
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
        else:
            session_model = model
            log.write("negotiate|{}\n".format(model or "unset"))
            log.flush()
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

/// An rpc peer on a discriminated command envelope whose replies echo
/// the answered command: it parks a `prime` `negotiate` open, issues its
/// own `negotiate` callback, and once the host's reply — a response
/// frame that echoes `negotiate` — arrives, refuses the parked call.
#[cfg(unix)]
const RPC_ENVELOPE_CALLBACK_REPLY: &str = r#"
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
    if request.get("type") == "response":
        # The host's reply to our callback releases the parked open's
        # refusal.
        if parked_open is not None and request.get("id") == "cb-1":
            send({"type": "response", "id": parked_open,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
            parked_open = None
        continue
    method = request.get("cmd")
    request_id = request.get("id")
    params = request.get("params") or {}
    if method == "negotiate":
        model = params.get("model")
        session_model = model
        log.write("negotiate|{}\n".format(model or "unset"))
        log.flush()
        if model == "prime":
            parked_open = request_id
            send({"type": "request", "cmd": "negotiate",
                  "id": "cb-1", "params": {}})
        else:
            send({"type": "response", "id": request_id,
                  "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        model = params.get("model") or session_model
        log.write("session/prompt|{}\n".format(model or "unset"))
        log.flush()
        send({"type": "response", "id": request_id,
              "result": {"text": "served|{}|{}".format(
                  model or "unset", params.get("text", ""))}})
    elif request_id is not None:
        send({"type": "response", "id": request_id, "result": None})
"#;

/// An rpc peer whose `session/prompt` settles an `error` result whose
/// only billable evidence is the usage record inside the synchronous
/// reply payload — no notification ever carries it.
#[cfg(unix)]
const RPC_RESULT_ONLY_USAGE: &str = r#"
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
        log.write("session/open\n")
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        log.write("session/prompt\n")
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "", "status": "error",
                         "usage": {"inputTokens": 3, "outputTokens": 7}}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

/// An rpc peer that dies on every `initialize` after the first: the
/// model-free handshake is answered once, then each replacement open's
/// process exits before its `session/open` ever reaches the wire.
#[cfg(unix)]
const RPC_DIE_ON_REOPEN_QUOTA: &str = r#"
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
        # A replacement open is a fresh process, so the count lives in
        # the log itself: only the lane's first handshake is answered.
        seen = sum(
            1 for entry in open(log_path) if entry.startswith("initialize")
        )
        if seen >= 1:
            sys.stdout.flush()
            os._exit(0)
        log.write("initialize\n")
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id, "result": {}})
    elif method == "session/open":
        session_model = params.get("model")
        log.write("session/open|{}\n".format(session_model or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        model = session_model
        log.write("session/prompt|{}\n".format(model or "unset"))
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

/// An rpc peer whose model-carrying `initialize` dies when it carries
/// `fallback`: the candidate's setup call reaches the wire, then the
/// process exits before `session/open` can follow.
#[cfg(unix)]
const RPC_DIE_ON_FALLBACK_QUOTA: &str = r#"
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
        session_model = params.get("model")
        log.write("initialize|{}\n".format(session_model or "unset"))
        log.flush()
        if session_model == "fallback":
            sys.stdout.flush()
            os._exit(0)
        send({"jsonrpc": "2.0", "id": request_id, "result": {}})
    elif method == "session/open":
        log.write("session/open\n")
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        model = session_model
        log.write("session/prompt|{}\n".format(model or "unset"))
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

/// An rpc peer whose `session/open` stalls — long enough for a cancel to
/// land mid-open — then logs the model each prompt carried.
#[cfg(unix)]
const RPC_SLOW_OPEN_PROMPT_LOG: &str = r#"
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
    params = request.get("params") or {}
    if method == "session/open":
        log.write("session/open\n")
        log.flush()
        time.sleep(0.8)
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        model = params.get("model")
        log.write("session/prompt|{}\n".format(model or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "served|{}|{}".format(
                  model or "unset", params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_inline_envelope_scalar_setup_params_open_serves() {
    // `paramsInline` merges object params into the frame root; scalar
    // params stay under `params` untouched, so the control-field filter
    // must apply only where the merge happened — a scalar document has
    // no root positions to strip, and the wire's `params` value remains
    // the model slot's verifiable position.
    let dir = TestDirectory::new("rpc-inline-scalar-setup");
    let script = dir.path().join("rpc_inline_scalar.py");
    std::fs::write(&script, RPC_INLINE_SCALAR_SETUP).unwrap();
    let mut spec = rpc_spec("scalar", &script, json!("{model}"), json!("{prompt}"));
    let LaneSpec::Rpc(lane) = &mut spec.lane else {
        panic!("rpc_spec builds an rpc lane")
    };
    lane.envelope = Some(a4aa::RpcEnvelopeSpec {
        method_field: "cmd".to_owned(),
        params_inline: true,
        response_method: Some("response".to_owned()),
        ..Default::default()
    });
    let mut document = RegistryDocument::default();
    document.agents.insert(
        spec.id.clone(),
        awp::AgentEntry {
            spec,
            alternates: vec![],
        },
    );
    document.profiles.insert(
        "p".to_owned(),
        ProfilePatch {
            agent: Some("scalar".to_owned()),
            model: Some("prime".to_owned()),
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
    assert_eq!(outcome.text.trim_end(), "served|prime|work");
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn rpc_fixed_model_handshake_quota_is_not_the_model_bound_open() {
    // `initialize` and `session/open` ride one wire method, and the
    // handshake's params even populate the field the open binds the
    // model through — a fixed `"handshake"` value, never the chain's
    // `{model}` slot. Matching populated positions instead of the
    // selected call would read its quota refusal as a model rejection
    // and burn every chain entry on identical handshakes. Each request
    // attaches to the call the adapter actually selected, so the
    // refusal stays a lane refusal and the walk reaches the next lane
    // with the chain untouched.
    let dir = TestDirectory::new("rpc-fixed-model-init");
    let refusing = dir.path().join("rpc_fixed_init.py");
    std::fs::write(&refusing, RPC_FIXED_MODEL_INIT_SHARED_METHOD).unwrap();
    let refusing_log = dir.path().join("refusing.log");
    let mut primary = rpc_spec(
        "fixedinit",
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
            params: json!({"model": "handshake"}),
        });
        lane.session_open = Some(RpcCallSpec {
            method: "negotiate".to_owned(),
            params: json!({"model": "{model}"}),
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
            agent: Some("fixedinit".to_owned()),
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
    assert_eq!(
        session.active_model().as_deref(),
        Some("prime"),
        "a fixed-model handshake's refusal is not a model rejection"
    );
    let calls_log = std::fs::read_to_string(&refusing_log).unwrap_or_default();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec!["negotiate|handshake"],
        "the handshake refusal must not burn chain entries on retries: {calls_log}"
    );
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn rpc_callback_reply_never_claims_the_setup_call() {
    // The peer parks the `prime` open, issues its own `negotiate`
    // callback, and refuses the parked call once the host's reply
    // arrives. That reply is a response frame that echoes the answered
    // command — under a method-name reading it is indistinguishable
    // from a setup request — so a journal that claimed slots for it
    // would leave the real open unattributed and call its quota refusal
    // model-free. Responses are never setup requests: the journaled id
    // still names the model-bound call, the chain advances, and the
    // retry binds `fallback`.
    let dir = TestDirectory::new("rpc-callback-echo");
    let script = dir.path().join("rpc_envelope_callback.py");
    std::fs::write(&script, RPC_ENVELOPE_CALLBACK_REPLY).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "callback",
        &script,
        json!({"model": "{model}"}),
        json!({"text": "{prompt}"}),
    );
    {
        let LaneSpec::Rpc(lane) = &mut primary.lane else {
            panic!("rpc_spec builds an rpc lane")
        };
        lane.session_open = Some(RpcCallSpec {
            method: "negotiate".to_owned(),
            params: json!({"model": "{model}"}),
        });
        lane.envelope = Some(a4aa::RpcEnvelopeSpec {
            method_field: "cmd".to_owned(),
            string_ids: true,
            discriminant_field: Some("type".to_owned()),
            request_marker: Some("request".to_owned()),
            response_method: Some("response".to_owned()),
            command_field: Some("cmd".to_owned()),
            ..Default::default()
        });
    }
    attach_call_log(&mut primary, &log);
    let mut document = RegistryDocument::default();
    document.agents.insert(
        primary.id.clone(),
        awp::AgentEntry {
            spec: primary,
            alternates: vec![],
        },
    );
    document.profiles.insert(
        "p".to_owned(),
        ProfilePatch {
            agent: Some("callback".to_owned()),
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
    assert_eq!(outcome.text.trim_end(), "served|fallback|work");
    assert_eq!(session.active_model().as_deref(), Some("fallback"));
    let calls_log = std::fs::read_to_string(&log).unwrap_or_default();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "negotiate|prime",
            "negotiate|fallback",
            "session/prompt|fallback"
        ],
        "the refused call was the model-bound open: {calls_log}"
    );
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn rpc_result_only_usage_marks_the_error_served() {
    // The turn's only billable evidence is the usage record inside the
    // prompt's synchronous reply — no notification ever carries one, so
    // the outcome's usage is empty and its text is too. The wire probe
    // correlates the reply to the prompt's request id: a nonzero record
    // there still proves the paid turn ran, so the `error` outcome
    // stands instead of replaying on the alternate lane.
    let dir = TestDirectory::new("rpc-result-only-usage");
    let script = dir.path().join("rpc_result_usage.py");
    std::fs::write(&script, RPC_RESULT_ONLY_USAGE).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec("resultuse", &script, json!({}), json!({"text": "{prompt}"}));
    attach_call_log(&mut primary, &log);
    let registry = registry_with_lanes(
        primary,
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("resultuse".to_owned()),
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
        "result-only usage still proves the turn ran: {outcome:?}"
    );
    let calls_log = std::fs::read_to_string(&log).unwrap_or_default();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec!["session/open", "session/prompt"],
        "the served error must not replay the prompt: {calls_log}"
    );
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn cli_zero_usage_records_never_block_the_walk() {
    // A json_lines lane that emits only an empty result with an all-zero
    // usage record did no billable work: the record is bookkeeping, not
    // output evidence, so the `error` outcome cannot hold the walk —
    // the alternate lane runs the turn.
    let mut primary = cli_spec(
        "zerouse",
        "sh",
        &[
            "-c",
            "exec 0<&-; \
             echo '{\"type\":\"result\",\"result\":\"\",\"usage\":{\"inputTokens\":0,\"outputTokens\":0}}'; \
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
    lane.usage_path = Some("usage".to_owned());
    let registry = registry_with_lanes(
        primary,
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("zerouse".to_owned()),
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
    assert_eq!(
        outcome.text.trim_end(),
        "alt|work",
        "zero-usage bookkeeping is not served work: {outcome:?}"
    );
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn cancel_during_model_free_replacement_restores_the_unsent_candidate() {
    // `prime` is quota-refused at the prompt; the session-bound retry
    // removes `fallback` and re-opens — but the replacement's
    // model-free `initialize` dies mid-handshake, so no setup call ever
    // carried the candidate. The cancel lands while the next lane's
    // model-free `session/open` still stalls: with no wire evidence the
    // candidate returns to the chain, and the next turn retries it.
    let dir = TestDirectory::new("rpc-cancel-unsent-reopen");
    let first = dir.path().join("rpc_die_reopen.py");
    std::fs::write(&first, RPC_DIE_ON_REOPEN_QUOTA).unwrap();
    let first_log = dir.path().join("first.log");
    let mut primary = rpc_spec(
        "diereopen",
        &first,
        json!({"model": "{model}"}),
        json!({"text": "{prompt}"}),
    );
    {
        let LaneSpec::Rpc(lane) = &mut primary.lane else {
            panic!("rpc_spec builds an rpc lane")
        };
        lane.initialize = Some(RpcCallSpec {
            method: "initialize".to_owned(),
            params: json!({}),
        });
    }
    attach_call_log(&mut primary, &first_log);
    let second = dir.path().join("rpc_slow_open.py");
    std::fs::write(&second, RPC_SLOW_OPEN_PROMPT_LOG).unwrap();
    let second_log = dir.path().join("second.log");
    let mut alternate = rpc_spec(
        "slowopen",
        &second,
        json!({}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut alternate, &second_log);
    let registry = registry_with_lanes(
        primary,
        alternate,
        ProfilePatch {
            agent: Some("diereopen".to_owned()),
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
    // Wait until the next lane's open is on the wire and stalled — the
    // earlier attempt died at its model-free `initialize`, so the
    // candidate never rode a setup frame anywhere.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    loop {
        let contents = std::fs::read_to_string(&second_log).unwrap_or_default();
        if contents.contains("session/open") {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the next lane's open never ran: {contents} | first: {}",
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

    // No wire ever carried `fallback`: it went back to the chain, so the
    // next turn's `prime` refusal retries it — the walk reaches the
    // alternate, whose per-turn selector serves it.
    let outcome = session.prompt("more".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "served|fallback|more");
    let calls_log = std::fs::read_to_string(&first_log).unwrap_or_default();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "initialize",
            "session/open|prime",
            "session/prompt|prime",
            "session/prompt|prime",
        ],
        "the candidate was never sent, so no reopen binds it: {calls_log}"
    );
    let calls_log = std::fs::read_to_string(&second_log).unwrap_or_default();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec!["session/open", "session/open", "session/prompt|fallback"],
        "the next turn's retry binds the restored candidate: {calls_log}"
    );
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn cancel_after_candidate_sent_then_died_keeps_it_spent() {
    // `prime` is refused at the prompt; the session-bound retry removes
    // `fallback` and re-opens — this lane's model-carrying `initialize`
    // puts the candidate on the wire, then the process dies before its
    // model-free `session/open`. The cancel lands while the next lane's
    // open still stalls: the folded send record keeps the candidate
    // spent instead of restoring it on the fresh probe's empty record.
    let dir = TestDirectory::new("rpc-cancel-sent-died");
    let first = dir.path().join("rpc_die_fallback.py");
    std::fs::write(&first, RPC_DIE_ON_FALLBACK_QUOTA).unwrap();
    let first_log = dir.path().join("first.log");
    let mut primary = rpc_spec(
        "diefallback",
        &first,
        json!({}),
        json!({"text": "{prompt}"}),
    );
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
    let second = dir.path().join("rpc_slow_open.py");
    std::fs::write(&second, RPC_SLOW_OPEN_PROMPT_LOG).unwrap();
    let second_log = dir.path().join("second.log");
    let mut alternate = rpc_spec(
        "slowopen",
        &second,
        json!({}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut alternate, &second_log);
    let registry = registry_with_lanes(
        primary,
        alternate,
        ProfilePatch {
            agent: Some("diefallback".to_owned()),
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
    // Wait until the next lane's open is on the wire and stalled — the
    // earlier attempt's `initialize|fallback` send is behind it, so a
    // restore at this point could only come from the fresh probe's
    // empty record, never from evidence that the wire carried the model.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    loop {
        let contents = std::fs::read_to_string(&second_log).unwrap_or_default();
        if contents.contains("session/open") {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the next lane's open never ran: {contents} | first: {}",
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

    // The candidate already reached the wire: it stays spent, so the
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
            "session/prompt|prime",
        ],
        "a candidate sent then abandoned must not be restored and resent: {calls_log}"
    );
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn cli_empty_result_after_zero_usage_never_blocks_the_walk() {
    // A json_lines lane's terminal bookkeeping is two records here: a
    // zero-counter usage document at the configured `usage_path`, then an
    // empty result under the declared selector. Neither is served work —
    // the `error` outcome must not hold the walk.
    let mut primary = cli_spec(
        "bookkeep",
        "sh",
        &[
            "-c",
            "exec 0<&-; \
             echo '{\"type\":\"usage\",\"stats\":{\"tokens\":{\"inputTokens\":0,\"outputTokens\":0}}}'; \
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
    lane.usage_path = Some("stats.tokens".to_owned());
    let registry = registry_with_lanes(
        primary,
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("bookkeep".to_owned()),
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
    assert_eq!(
        outcome.text.trim_end(),
        "alt|work",
        "bookkeeping-only records are not served work: {outcome:?}"
    );
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn cli_progress_with_zero_usage_never_replays_on_alternate_lane() {
    // A progress record that carries real content beside an all-zero
    // usage document is output evidence — the empty result record hides
    // the stream text, but the paid turn still ran: the alternate lane
    // must not replay it.
    let mut primary = cli_spec(
        "progress",
        "sh",
        &[
            "-c",
            "exec 0<&-; \
             echo '{\"type\":\"assistant\",\"message\":{\"role\":\"assistant\",\"content\":\"real work\",\"usage\":{\"inputTokens\":0,\"outputTokens\":0}}}'; \
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
    lane.usage_path = Some("message.usage".to_owned());
    let registry = registry_with_lanes(
        primary,
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("progress".to_owned()),
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
        "alternate did not replay a served turn: {outcome:?}"
    );
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn cli_session_init_record_does_not_suppress_fallback() {
    // A json_lines lane that emits only its session-init record and exits
    // non-zero did no served work: the record's only claim is the session
    // binding at the lane's declared `session_id_path` — bookkeeping, not
    // turn output. The alternate must serve the turn, exactly as it does
    // when the primary emits no record at all.
    let mut primary = cli_spec(
        "init-only",
        "sh",
        &[
            "-c",
            "exec 0<&-; \
             echo '{\"type\":\"system\",\"subtype\":\"init\",\"session_id\":\"sess-1\"}'; \
             exit 7",
        ],
        PromptChannel::Stdin,
    );
    let LaneSpec::Cli(lane) = &mut primary.lane else {
        panic!("cli_spec builds a cli lane")
    };
    lane.json_lines = true;
    lane.session_id_path = Some("session_id".to_owned());
    let registry = registry_with_lanes(
        primary,
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("init-only".to_owned()),
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
async fn cli_tool_use_record_preserves_replay_protection() {
    // A json_lines lane that emits an assistant record carrying its
    // session binding beside a `message.content` tool_use block, then
    // exits non-zero: the session-init exemption covers records whose
    // whole claim is the binding — a tool call is activity the caller
    // already saw, so replaying the turn on the alternate would
    // duplicate it. The tool event has no `text` leaf; it must still
    // count as produced output.
    let mut primary = cli_spec(
        "tooling",
        "sh",
        &[
            "-c",
            "exec 0<&-; \
             echo '{\"type\":\"assistant\",\"session_id\":\"sess-1\",\"message\":{\"role\":\"assistant\",\"content\":[{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"Bash\",\"input\":{\"command\":\"ls\"}}]}}'; \
             exit 7",
        ],
        PromptChannel::Stdin,
    );
    let LaneSpec::Cli(lane) = &mut primary.lane else {
        panic!("cli_spec builds a cli lane")
    };
    lane.json_lines = true;
    lane.session_id_path = Some("session_id".to_owned());
    let registry = registry_with_lanes(
        primary,
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("tooling".to_owned()),
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
        "alternate must not replay a turn whose tool activity reached the caller: {outcome:?}"
    );
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn cli_quota_refusal_advances_the_model_chain() {
    // A stderr-only quota refusal is a certain-effect admission
    // rejection: nothing ran, so the turn retries the chain's next entry
    // on the same lane — the model rides the cli lane's args per spawn.
    // Walking lanes instead would leave quota-checked work unspent.
    let primary = cli_spec(
        "quota-model",
        "sh",
        &[
            "-c",
            "exec 0<&-; if [ \"$1\" = \"backup\" ]; then \
             echo \"served|$1|$2\"; else echo 'quota exceeded' >&2; exit 1; fi",
            "sh0",
            "{model}",
            "{prompt}",
        ],
        PromptChannel::ArgOnly,
    );
    let registry = registry_with_lanes(
        primary,
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("quota-model".to_owned()),
            model: Some("prime".to_owned()),
            models: Some(vec!["backup".to_owned()]),
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
    assert_eq!(
        outcome.text.trim_end(),
        "served|backup|work",
        "the next chain entry retried on the same lane: {outcome:?}"
    );
    session.close().await.unwrap();
}

/// An rpc peer whose model-free `session/select` pre-turn call refuses
/// every request with a provider quota error — the identical answer for
/// every model the chain could try.
#[cfg(unix)]
const RPC_PRE_TURN_MODEL_FREE_QUOTA: &str = r#"
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
        log.write("session/open\n")
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/select":
        log.write("session/select\n")
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "error": {"code": -32000, "message":
                        "resource_exhausted: quota exceeded"}})
    elif method == "session/prompt":
        model = params.get("model")
        log.write("session/prompt|{}\n".format(model or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "served|{}|{}".format(
                  model or "unset", params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_model_free_pre_turn_refusal_walks_lanes() {
    // The pre-turn preparation call carries no model, so its quota
    // refusal answers the lane, not the chain entry: retrying it with
    // every remaining model sends the identical request before the walk
    // must move to the alternate lane.
    let dir = TestDirectory::new("rpc-pre-turn-quota");
    let script = dir.path().join("rpc_pre_turn.py");
    std::fs::write(&script, RPC_PRE_TURN_MODEL_FREE_QUOTA).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "preturn",
        &script,
        json!({}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    {
        let LaneSpec::Rpc(lane) = &mut primary.lane else {
            panic!("rpc_spec builds an rpc lane")
        };
        lane.pre_turn = vec![RpcCallSpec {
            method: "session/select".to_owned(),
            params: json!({"select": "x"}),
        }];
    }
    attach_call_log(&mut primary, &log);
    let registry = registry_with_lanes(
        primary,
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("preturn".to_owned()),
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
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec!["session/open", "session/select"],
        "a model-free refusal never spends another chain entry: {calls_log}"
    );
    session.close().await.unwrap();
}

/// A cmd-envelope peer that refuses every `session/prompt` with a quota
/// error — the model it never received cannot change the answer.
#[cfg(unix)]
const RPC_INLINE_PROMPT_QUOTA: &str = r#"
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
    method = request.get("cmd")
    request_id = request.get("id")
    if method == "session/open":
        log.write("session/open\n")
        log.flush()
        send({"cmd": "response", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        log.write("session/prompt\n")
        log.flush()
        send({"cmd": "response", "id": request_id,
              "error": {"code": -32000, "message":
                        "resource_exhausted: quota exceeded for this model"}})
    elif request_id is not None:
        send({"cmd": "response", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_inline_overwritten_model_slot_never_retries_the_exchange() {
    // The only `{model}` position in `session/open` sits under `id` —
    // the inline envelope stamps its request id over the merged params,
    // so the model never reaches the wire and no chain entry could
    // change the call's answer. The refusal is this lane's answer, not a
    // model verdict: the turn replays on the alternate — but only after
    // one identical exchange, never a reopened second.
    let dir = TestDirectory::new("rpc-inline-masked-model");
    let script = dir.path().join("rpc_inline_masked.py");
    std::fs::write(&script, RPC_INLINE_PROMPT_QUOTA).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "masked",
        &script,
        json!({"id": "{model}", "note": "x"}),
        json!({"text": "{prompt}"}),
    );
    {
        let LaneSpec::Rpc(lane) = &mut primary.lane else {
            panic!("rpc_spec builds an rpc lane")
        };
        lane.envelope = Some(a4aa::RpcEnvelopeSpec {
            method_field: "cmd".to_owned(),
            params_inline: true,
            response_method: Some("response".to_owned()),
            ..Default::default()
        });
    }
    attach_call_log(&mut primary, &log);
    let registry = registry_with_lanes(
        primary,
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("masked".to_owned()),
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
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec!["session/open", "session/prompt"],
        "a slot the wire never carries is no model binding: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer whose model-free `session/select` pre-turn call answers
/// once, then kills the process on its second invocation — a per-turn
/// model retry dies at its preparation call before the candidate could
/// ride the prompt.
#[cfg(unix)]
const RPC_PRE_TURN_DIES_SECOND: &str = r#"
import json, os, sys

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

log = open(os.environ["AWP_CALL_LOG"], "a")
log_path = os.environ["AWP_CALL_LOG"]
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
    elif method == "session/select":
        seen = sum(
            1 for entry in open(log_path)
            if entry.startswith("session/select")
        )
        log.write("session/select\n")
        log.flush()
        if seen == 1:
            sys.stdout.flush()
            os._exit(1)
        send({"jsonrpc": "2.0", "id": request_id, "result": {}})
    elif method == "session/prompt":
        model = params.get("model")
        log.write("session/prompt|{}\n".format(model or "unset"))
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

/// An rpc peer whose `session/open` stalls — long enough for a cancel to
/// land mid-open — then refuses `prime` at the prompt and serves every
/// other model.
#[cfg(unix)]
const RPC_SLOW_OPEN_REFUSES_PRIME: &str = r#"
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
    params = request.get("params") or {}
    if method == "session/open":
        log.write("session/open\n")
        log.flush()
        time.sleep(0.8)
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        model = params.get("model")
        log.write("session/prompt|{}\n".format(model or "unset"))
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
async fn cancel_during_alternate_open_restores_the_unsent_turn_candidate() {
    // `prime` is quota-refused at the prompt; the per-turn retry removes
    // `fallback` from the chain — but its attempt dies at the model-free
    // pre-turn call, so the candidate never rode a frame anywhere. The
    // cancel lands while the next lane's open stalls: the candidate only
    // survives in the turn's driver state, and it must return to the
    // chain — the next turn's `prime` refusal retries it.
    let dir = TestDirectory::new("rpc-cancel-unsent-turn");
    let first = dir.path().join("rpc_pre_turn_die.py");
    std::fs::write(&first, RPC_PRE_TURN_DIES_SECOND).unwrap();
    let first_log = dir.path().join("first.log");
    let mut primary = rpc_spec(
        "dieturn",
        &first,
        json!({"model": "{model}"}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    {
        let LaneSpec::Rpc(lane) = &mut primary.lane else {
            panic!("rpc_spec builds an rpc lane")
        };
        lane.pre_turn = vec![RpcCallSpec {
            method: "session/select".to_owned(),
            params: json!({"select": "x"}),
        }];
    }
    attach_call_log(&mut primary, &first_log);
    let second = dir.path().join("rpc_slow_open.py");
    std::fs::write(&second, RPC_SLOW_OPEN_REFUSES_PRIME).unwrap();
    let second_log = dir.path().join("second.log");
    let mut alternate = rpc_spec(
        "slowopen",
        &second,
        json!({}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut alternate, &second_log);
    let registry = registry_with_lanes(
        primary,
        alternate,
        ProfilePatch {
            agent: Some("dieturn".to_owned()),
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
    // Wait until the next lane's open is on the wire and stalled — the
    // earlier attempt died at its model-free `session/select`, so the
    // candidate never rode a frame anywhere.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    loop {
        let contents = std::fs::read_to_string(&second_log).unwrap_or_default();
        if contents.contains("session/open") {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the next lane's open never ran: {contents} | first: {}",
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

    // No wire ever carried `fallback`: it went back to the chain, so the
    // next turn's walk reaches the alternate and retries it there.
    let outcome = session.prompt("more".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "served|fallback|more");
    let calls_log = std::fs::read_to_string(&first_log).unwrap_or_default();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "session/open|prime",
            "session/select",
            "session/prompt|prime",
            "session/select",
        ],
        "the per-turn candidate died at the model-free call: {calls_log}"
    );
    let calls_log = std::fs::read_to_string(&second_log).unwrap_or_default();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "session/open",
            "session/open",
            "session/prompt|prime",
            "session/prompt|fallback",
        ],
        "the next turn's retry binds the restored candidate: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer whose `session/select` preparation call refuses `prime`
/// with quota and serves everything else; the prompt serves the model
/// it was handed.
#[cfg(unix)]
const RPC_SELECT_REFUSES_PRIME: &str = r#"
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
        log.write("session/open\n")
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/select":
        model = params.get("model")
        log.write("session/select|{}\n".format(model or "unset"))
        log.flush()
        if model == "prime":
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
        else:
            send({"jsonrpc": "2.0", "id": request_id, "result": {}})
    elif method == "session/prompt":
        model = params.get("model")
        log.write("session/prompt|{}\n".format(model or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "served|{}|{}".format(
                  model or "unset", params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_skipped_pre_turn_call_keeps_the_refusal_on_the_model() {
    // Two `session/select` pre-turn calls share one wire method: the
    // first's `{effort}` leaf resolves empty so the lane skips it, and
    // the second carries the model. The refusal the second call draws
    // must attribute to it — not to the skipped template — or a model
    // quota refusal reads as the lane's answer and the chain never
    // moves.
    let dir = TestDirectory::new("rpc-skip-select");
    let script = dir.path().join("rpc_select_quota.py");
    std::fs::write(&script, RPC_SELECT_REFUSES_PRIME).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "skipselect",
        &script,
        json!({}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    {
        let LaneSpec::Rpc(lane) = &mut primary.lane else {
            panic!("rpc_spec builds an rpc lane")
        };
        lane.pre_turn = vec![
            RpcCallSpec {
                method: "session/select".to_owned(),
                params: json!({"effort": "{effort}"}),
            },
            RpcCallSpec {
                method: "session/select".to_owned(),
                params: json!({"model": "{model}", "keep": "x"}),
            },
        ];
    }
    attach_call_log(&mut primary, &log);
    let registry = registry_with_lanes(
        primary,
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("skipselect".to_owned()),
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
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "session/open",
            "session/select|prime",
            "session/select|fallback",
            "session/prompt|fallback",
        ],
        "the refusal the model call drew must retry the chain: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer whose `session/select` pauses before answering — long
/// enough for a cancel to land mid-call — then serves every model but
/// `prime`, which is quota-refused at the prompt.
#[cfg(unix)]
const RPC_SLOW_SELECT_REFUSES_PRIME: &str = r#"
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
    params = request.get("params") or {}
    if method == "session/open":
        log.write("session/open\n")
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/select":
        log.write("session/select\n")
        log.flush()
        time.sleep(0.8)
        send({"jsonrpc": "2.0", "id": request_id, "result": {}})
    elif method == "session/prompt":
        model = params.get("model")
        log.write("session/prompt|{}\n".format(model or "unset"))
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
async fn cancelled_model_free_pre_turn_restores_the_unsent_turn_candidate() {
    // `prime` is quota-refused at the prompt; the per-turn retry
    // schedules `fallback` — but the cancel lands inside the model-free
    // pre-turn call, so the lane returns `cancelled` before the
    // candidate's prompt frame is ever written. The candidate must
    // return to the chain on its own send record: the installed probe's
    // setup flag describes the open's model, not this one.
    let dir = TestDirectory::new("rpc-cancel-pre-turn");
    let script = dir.path().join("rpc_slow_select.py");
    std::fs::write(&script, RPC_SLOW_SELECT_REFUSES_PRIME).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "slowselect",
        &script,
        json!({}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    {
        let LaneSpec::Rpc(lane) = &mut primary.lane else {
            panic!("rpc_spec builds an rpc lane")
        };
        lane.pre_turn = vec![RpcCallSpec {
            method: "session/select".to_owned(),
            params: json!({"select": "x"}),
        }];
    }
    attach_call_log(&mut primary, &log);
    let registry = registry_with(
        primary,
        ProfilePatch {
            agent: Some("slowselect".to_owned()),
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
    // Wait until the retry's model-free `session/select` is in flight —
    // the cancel lands inside its pause, before the candidate's prompt
    // frame could be written.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    loop {
        let contents = std::fs::read_to_string(&log).unwrap_or_default();
        if contents.matches("session/select").count() >= 2 {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the retry's pre-turn call never ran: {contents}"
        );
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    session.cancel().await.unwrap();

    let outcome = tokio::time::timeout(std::time::Duration::from_secs(10), driver)
        .await
        .expect("the turn outlived cancel inside the pre-turn call")
        .unwrap()
        .unwrap();
    assert_eq!(outcome.stop_reason, "cancelled");

    // `fallback` never rode a frame: it went back to the chain, so the
    // next turn's `prime` refusal retries it on the same lane.
    let outcome = session.prompt("more".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "served|fallback|more");
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "session/open",
            "session/select",
            "session/prompt|prime",
            "session/select",
            "session/select",
            "session/prompt|prime",
            "session/select",
            "session/prompt|fallback",
        ],
        "the cancelled pre-turn attempt must restore the unsent candidate: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer that binds `prime` at `session/open`, quota-refuses the
/// session's prompt, and dies — after logging — when the retry binds
/// `fallback`, so the candidate's open frame reached the wire but its
/// session never came up.
#[cfg(unix)]
const RPC_OPEN_QUOTA_DIES_ON_FALLBACK: &str = r#"
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
        model = params.get("model")
        session_model = model
        log.write("session/open|{}\n".format(model or "unset"))
        log.flush()
        if model == "fallback":
            sys.stdout.flush()
            os._exit(1)
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

/// An rpc peer for the selector-less middle lane: its open succeeds
/// and its prompt is never asked — the lane's empty prompt method
/// refuses before any dispatch.
#[cfg(unix)]
const RPC_OPEN_ONLY: &str = r#"
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
              "result": {"sessionId": "awp-rpc-session"}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

/// An rpc peer that quota-refuses every `session/prompt` — on the third
/// lane the retry it would enable exists only while the chain still
/// holds the candidate.
#[cfg(unix)]
const RPC_PROMPT_ALWAYS_QUOTA: &str = r#"
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
        log.write("session/open\n")
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        model = params.get("model")
        log.write("session/prompt|{}\n".format(model or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "error": {"code": -32000, "message":
                        "resource_exhausted: quota exceeded for this model"}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn sent_candidate_stays_spent_across_a_model_free_install() {
    // `prime` binds at the primary's open and is quota-refused; the
    // session-bound retry's `open|fallback` reaches the wire before the
    // peer dies — the candidate is sent, not pending. The walk then
    // installs a lane with no model selector: that lane cannot deliver
    // the candidate, and restoring it there would requeue an entry the
    // peer may already have bound — it stays spent, so the third lane's
    // refusal has nothing left to retry.
    let dir = TestDirectory::new("rpc-spent-install");
    let first = dir.path().join("rpc_die_fallback_open.py");
    std::fs::write(&first, RPC_OPEN_QUOTA_DIES_ON_FALLBACK).unwrap();
    let first_log = dir.path().join("first.log");
    let mut primary = rpc_spec(
        "diesopen",
        &first,
        json!({"model": "{model}"}),
        json!({"text": "{prompt}"}),
    );
    attach_call_log(&mut primary, &first_log);
    let second = dir.path().join("rpc_open_only.py");
    std::fs::write(&second, RPC_OPEN_ONLY).unwrap();
    let second_log = dir.path().join("second.log");
    let mut model_free = rpc_spec(
        "nofreesurface",
        &second,
        json!({}),
        json!({"text": "{prompt}"}),
    );
    {
        let LaneSpec::Rpc(lane) = &mut model_free.lane else {
            panic!("rpc_spec builds an rpc lane")
        };
        lane.prompt.method.clear();
    }
    attach_call_log(&mut model_free, &second_log);
    let third = dir.path().join("rpc_always_quota.py");
    std::fs::write(&third, RPC_PROMPT_ALWAYS_QUOTA).unwrap();
    let third_log = dir.path().join("third.log");
    let mut surfaced = rpc_spec(
        "surfaced",
        &third,
        json!({}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut surfaced, &third_log);
    let mut document = RegistryDocument::default();
    document.agents.insert(
        primary.id.clone(),
        awp::AgentEntry {
            spec: primary,
            alternates: vec![model_free, surfaced],
        },
    );
    document.profiles.insert(
        "p".to_owned(),
        ProfilePatch {
            agent: Some("diesopen".to_owned()),
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

    let error = session.prompt("work".to_owned()).await.unwrap_err();
    assert_eq!(error.code, "agent_quota_exhausted");
    let calls_log = std::fs::read_to_string(&first_log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "session/open|prime",
            "session/prompt|prime",
            "session/open|fallback"
        ],
        "the candidate's open frame reached the wire: {calls_log}"
    );
    let calls_log = std::fs::read_to_string(&third_log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec!["session/open", "session/prompt|prime"],
        "a sent candidate never requeues on the model-free lane: {calls_log}"
    );
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn cli_result_record_with_params_member_is_bookkeeping() {
    // A failed json_lines turn whose terminal record carries an empty
    // `params` sibling did no billable work: the selector reads the
    // record itself, so the empty result is bookkeeping — not the
    // progress a `params` member lookup would fabricate.
    let mut primary = cli_spec(
        "paramsibling",
        "sh",
        &[
            "-c",
            "exec 0<&-; \
             echo '{\"type\":\"result\",\"result\":\"\",\"params\":{}}'; \
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
    lane.usage_path = Some("usage".to_owned());
    let registry = registry_with_lanes(
        primary,
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("paramsibling".to_owned()),
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
    assert_eq!(
        outcome.text.trim_end(),
        "alt|work",
        "an empty result at the record root is bookkeeping: {outcome:?}"
    );
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn cli_bookkeeping_only_stream_never_counts_as_served() {
    // A json_lines lane that exits nonzero after only zero-usage records
    // returns the retained stream as outcome text — every line a
    // record the relay already classified as bookkeeping. The raw
    // fallback is not result text: the error outcome cannot hold the
    // walk.
    let mut primary = cli_spec(
        "streamonly",
        "sh",
        &[
            "-c",
            "exec 0<&-; \
             echo '{\"type\":\"usage\",\"usage\":{\"inputTokens\":0,\"outputTokens\":0}}'; \
             echo '{\"type\":\"summary\",\"usage\":{\"inputTokens\":0,\"outputTokens\":0}}'; \
             exit 1",
        ],
        PromptChannel::Stdin,
    );
    let LaneSpec::Cli(lane) = &mut primary.lane else {
        panic!("cli_spec builds a cli lane")
    };
    lane.json_lines = true;
    lane.usage_path = Some("usage".to_owned());
    let registry = registry_with_lanes(
        primary,
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("streamonly".to_owned()),
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
    assert_eq!(
        outcome.text.trim_end(),
        "alt|work",
        "retained bookkeeping records are not served output: {outcome:?}"
    );
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn cli_blank_lines_inside_a_bookkeeping_stream_still_fall_back() {
    // The same bookkeeping-only stream, now with blank separators the
    // lane itself ignored between records: the retained text is still
    // nothing but records, so the failed lane's outcome cannot hold
    // the walk.
    let mut primary = cli_spec(
        "blankstream",
        "sh",
        &[
            "-c",
            "exec 0<&-; \
             echo '{\"type\":\"usage\",\"usage\":{\"inputTokens\":0,\"outputTokens\":0}}'; \
             echo; \
             echo '{\"type\":\"summary\",\"usage\":{\"inputTokens\":0,\"outputTokens\":0}}'; \
             exit 1",
        ],
        PromptChannel::Stdin,
    );
    let LaneSpec::Cli(lane) = &mut primary.lane else {
        panic!("cli_spec builds a cli lane")
    };
    lane.json_lines = true;
    lane.usage_path = Some("usage".to_owned());
    let registry = registry_with_lanes(
        primary,
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("blankstream".to_owned()),
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
    assert_eq!(
        outcome.text.trim_end(),
        "alt|work",
        "blank separators inside the retained stream are not output: {outcome:?}"
    );
    session.close().await.unwrap();
}

/// An rpc peer whose `session/select` carries the selector under one
/// shared `choice` key: it refuses `prime` with quota and serves every
/// other value; the prompt serves the model it was handed.
#[cfg(unix)]
const RPC_CHOICE_SELECT_REFUSES_PRIME: &str = r#"
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
        log.write("session/open\n")
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/select":
        choice = params.get("choice")
        log.write("session/select|{}\n".format(choice or "unset"))
        log.flush()
        if choice == "prime":
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
        else:
            send({"jsonrpc": "2.0", "id": request_id, "result": {}})
    elif method == "session/prompt":
        model = params.get("model")
        log.write("session/prompt|{}\n".format(model or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "served|{}|{}".format(
                  model or "unset", params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_same_key_skipped_call_keeps_the_refusal_on_the_model() {
    // Both `session/select` calls use one `choice` key: the effort
    // call's `{effort}` resolves absent so the lane emits nothing for
    // it, and the model call's resolved `choice` is the only select
    // frame on the wire. Its quota refusal must stay on the model —
    // if the skipped template claimed the frame, the verdict would
    // read as the lane's answer and the chain would never move.
    let dir = TestDirectory::new("rpc-same-key-skip");
    let script = dir.path().join("rpc_choice_quota.py");
    std::fs::write(&script, RPC_CHOICE_SELECT_REFUSES_PRIME).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "samekeyskip",
        &script,
        json!({}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    {
        let LaneSpec::Rpc(lane) = &mut primary.lane else {
            panic!("rpc_spec builds an rpc lane")
        };
        lane.pre_turn = vec![
            RpcCallSpec {
                method: "session/select".to_owned(),
                params: json!({"choice": "{effort}"}),
            },
            RpcCallSpec {
                method: "session/select".to_owned(),
                params: json!({"choice": "{model}"}),
            },
        ];
    }
    attach_call_log(&mut primary, &log);
    let registry = registry_with_lanes(
        primary,
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("samekeyskip".to_owned()),
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
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "session/open",
            "session/select|prime",
            "session/select|fallback",
            "session/prompt|fallback",
        ],
        "the skipped call cannot claim the model call's frame: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer whose `session/open` refuses `prime` with quota and
/// binds whatever model opened the session for the prompt.
#[cfg(unix)]
const RPC_OPEN_OPTIONS_REFUSES_PRIME: &str = r#"
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
        options = params.get("options")
        log.write("session/open|{}|{}\n".format(session_model, options))
        log.flush()
        if session_model == "prime":
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
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "served|{}|{}".format(
                  model or "unset", params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_nested_empty_open_object_keeps_the_refusal_on_the_model() {
    // The open template nests `{effort}` inside `options`: with effort
    // absent the leaf drops but the object stays, and the lane emits
    // `options={}` — the frame must still claim the session call or a
    // quota refusal at open reads as the lane's answer and skips the
    // model retry entirely.
    let dir = TestDirectory::new("rpc-nested-empty");
    let script = dir.path().join("rpc_open_options.py");
    std::fs::write(&script, RPC_OPEN_OPTIONS_REFUSES_PRIME).unwrap();
    let log = dir.path().join("calls.log");
    let primary = rpc_spec(
        "nestedempty",
        &script,
        json!({"model": "{model}", "options": {"effort": "{effort}"}}),
        json!({"text": "{prompt}"}),
    );
    let mut primary = primary;
    attach_call_log(&mut primary, &log);
    let registry = registry_with_lanes(
        primary,
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("nestedempty".to_owned()),
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
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "session/open|prime|{}",
            "session/open|fallback|{}",
            "session/prompt|fallback",
        ],
        "the empty nested object is the lane's own emission: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer that refuses a prompt carrying a `provider` member —
/// the qualified candidate's carrier — and serves the provider-free
/// prompt with its own default model.
#[cfg(unix)]
const RPC_PROVIDER_ONLY_REFUSES_CARRIED: &str = r#"
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
        log.write("session/open\n")
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        provider = params.get("provider")
        log.write("session/prompt|{}\n".format(provider or "unset"))
        log.flush()
        if provider is not None:
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
        else:
            send({"jsonrpc": "2.0", "id": request_id,
                  "result": {"text": "served|peerdefault|{}".format(
                      params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_unsent_fallback_never_publishes_as_the_served_model() {
    // The prompt's only model surface is `{modelProvider}`: a qualified
    // `provider/prime` resolves it and the refusal it draws advances
    // the chain, but the unqualified `fallback` entry resolves it
    // absent — the member drops, the peer serves with its own default,
    // and nothing on the wire ever carried `fallback`. A successful
    // outcome is not carry evidence: the unsent candidate must not
    // publish as the session's active model.
    let dir = TestDirectory::new("rpc-unsent-fallback");
    let script = dir.path().join("rpc_provider_only.py");
    std::fs::write(&script, RPC_PROVIDER_ONLY_REFUSES_CARRIED).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "provideronly",
        &script,
        json!({}),
        json!({"text": "{prompt}", "provider": "{modelProvider}"}),
    );
    attach_call_log(&mut primary, &log);
    let registry = registry_with_lanes(
        primary,
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("provideronly".to_owned()),
            model: Some("provider/prime".to_owned()),
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
    assert_eq!(outcome.text.trim_end(), "served|peerdefault|work");
    assert_eq!(
        session.active_model().as_deref(),
        Some("provider/prime"),
        "the wire never carried `fallback`: it cannot publish as active"
    );
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "session/open",
            "session/prompt|provider",
            "session/prompt|unset"
        ],
        "the fallback attempt carried no selector: {calls_log}"
    );
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn rpc_inline_overwritten_turn_slot_never_retries_the_exchange() {
    // The prompt's only `{model}` position sits under `id` — the inline
    // envelope stamps the request id over the merged params, so the
    // model never reaches the wire and no chain entry could change the
    // answer. The refusal is this lane's answer: the turn replays on the
    // alternate — but only after one identical exchange, never a second
    // try on the same model-free payload.
    let dir = TestDirectory::new("rpc-inline-masked-turn");
    let script = dir.path().join("rpc_inline_masked.py");
    std::fs::write(&script, RPC_INLINE_PROMPT_QUOTA).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "maskedturn",
        &script,
        json!({}),
        json!({"text": "{prompt}", "id": "{model}"}),
    );
    {
        let LaneSpec::Rpc(lane) = &mut primary.lane else {
            panic!("rpc_spec builds an rpc lane")
        };
        lane.envelope = Some(a4aa::RpcEnvelopeSpec {
            method_field: "cmd".to_owned(),
            params_inline: true,
            response_method: Some("response".to_owned()),
            ..Default::default()
        });
    }
    attach_call_log(&mut primary, &log);
    let registry = registry_with_lanes(
        primary,
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("maskedturn".to_owned()),
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
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec!["session/open", "session/prompt"],
        "a turn slot the wire never carries is no retry surface: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer that logs the open's marker verbatim: a fresh session's
/// `open-{sessionId}` has no id to substitute, so the literal token
/// reaches the wire — and `prime` draws a quota refusal.
#[cfg(unix)]
const RPC_MARKER_OPEN_REFUSES_PRIME: &str = r#"
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
        session_model = params.get("model")
        log.write("session/open|{}|{}\n".format(
            params.get("marker"), session_model))
        log.flush()
        if session_model == "prime":
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
            continue
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        log.write("session/prompt|{}\n".format(params.get("model") or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "served|{}|{}".format(
                  params.get("model") or "unset", params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_embedded_absent_id_keeps_the_open_refusal_attributed() {
    // The open embeds `{sessionId}` inside a larger marker: a fresh
    // session has no id to substitute, so the pinned adapter emits the
    // literal token `open-{sessionId}`. If the probe substitutes `""`
    // instead, the journaled frame claims no declared call — the quota
    // refusal loses its model attribution, the walk treats it as the
    // lane's answer, and the fallback model never runs.
    let dir = TestDirectory::new("rpc-embedded-id");
    let script = dir.path().join("rpc_marker_open.py");
    std::fs::write(&script, RPC_MARKER_OPEN_REFUSES_PRIME).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "embeddedid",
        &script,
        json!({"marker": "open-{sessionId}", "model": "{model}"}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut primary, &log);
    let registry = registry_with_lanes(
        primary,
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("embeddedid".to_owned()),
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
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "session/open|open-{sessionId}|prime",
            "session/open|open-{sessionId}|fallback",
            "session/prompt|fallback",
        ],
        "the literal embedded id must still claim the open call: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer whose `session/open` answers with no session id and
/// whose `session/select` refuses `prime` under one shared `session`
/// key.
#[cfg(unix)]
const RPC_SESSIONLESS_SELECT_REFUSES_PRIME: &str = r#"
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
        log.write("session/open\n")
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id, "result": {}})
    elif method == "session/select":
        selected = params.get("session")
        log.write("session/select|{}\n".format(selected or "unset"))
        log.flush()
        if selected == "prime":
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
        else:
            send({"jsonrpc": "2.0", "id": request_id, "result": {}})
    elif method == "session/prompt":
        model = params.get("model")
        log.write("session/prompt|{}\n".format(model or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "served|{}|{}".format(
                  model or "unset", params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_sessionless_open_keeps_the_refusal_on_the_model() {
    // The peer assigns no session id, so the id-only `session/select`
    // call resolves `{}` and the lane skips it — the lone select frame
    // on the wire is the model call's. If the journaled presence stays
    // "maybe present", the skipped template claims the frame and the
    // quota refusal reads as the lane's answer: the turn fails with
    // quota while `fallback` was still available.
    let dir = TestDirectory::new("rpc-sessionless-open");
    let script = dir.path().join("rpc_sessionless.py");
    std::fs::write(&script, RPC_SESSIONLESS_SELECT_REFUSES_PRIME).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "sessionless",
        &script,
        json!({}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    {
        let LaneSpec::Rpc(lane) = &mut primary.lane else {
            panic!("rpc_spec builds an rpc lane")
        };
        lane.pre_turn = vec![
            RpcCallSpec {
                method: "session/select".to_owned(),
                params: json!({"session": "{sessionId}"}),
            },
            RpcCallSpec {
                method: "session/select".to_owned(),
                params: json!({"session": "{model}"}),
            },
        ];
    }
    attach_call_log(&mut primary, &log);
    let registry = registry_with(
        primary,
        ProfilePatch {
            agent: Some("sessionless".to_owned()),
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
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "session/open",
            "session/select|prime",
            "session/select|fallback",
            "session/prompt|fallback",
        ],
        "the skipped id call cannot claim the model frame: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer whose `session/select` refuses `prime` under a shared
/// `choice` key — the secret-only call ahead of it resolves to nothing
/// when the referenced secret is unset.
#[cfg(unix)]
const RPC_SECRET_SELECT_REFUSES_PRIME: &str = r#"
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
        log.write("session/open\n")
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/select":
        choice = params.get("choice")
        log.write("session/select|{}\n".format(choice or "unset"))
        log.flush()
        if choice == "prime":
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
        else:
            send({"jsonrpc": "2.0", "id": request_id, "result": {}})
    elif method == "session/prompt":
        model = params.get("model")
        log.write("session/prompt|{}\n".format(model or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "served|{}|{}".format(
                  model or "unset", params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_unresolved_secret_keeps_the_refusal_on_the_model() {
    // The first `session/select` template carries only a `{secret:...}`
    // leaf the resolver cannot produce: the call resolves `{}` and the
    // lane skips it. If the probe guesses at presence instead of
    // reading the adapter's own resolution, the skipped call claims the
    // model frame and the quota refusal reads as the lane's answer.
    let dir = TestDirectory::new("rpc-secret-skip");
    let script = dir.path().join("rpc_secret_select.py");
    std::fs::write(&script, RPC_SECRET_SELECT_REFUSES_PRIME).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "secretskip",
        &script,
        json!({}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    {
        let LaneSpec::Rpc(lane) = &mut primary.lane else {
            panic!("rpc_spec builds an rpc lane")
        };
        lane.pre_turn = vec![
            RpcCallSpec {
                method: "session/select".to_owned(),
                params: json!({"choice": "{secret:AWP_TEST_UNSET_SELECTOR_SECRET}"}),
            },
            RpcCallSpec {
                method: "session/select".to_owned(),
                params: json!({"choice": "{model}"}),
            },
        ];
    }
    attach_call_log(&mut primary, &log);
    let registry = registry_with(
        primary,
        ProfilePatch {
            agent: Some("secretskip".to_owned()),
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
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "session/open",
            "session/select|prime",
            "session/select|fallback",
            "session/prompt|fallback",
        ],
        "the skipped secret call cannot claim the model frame: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer that refuses a `vnd:`-tagged prompt and serves the
/// scaffolded `v:` prompt with its own default model.
#[cfg(unix)]
const RPC_VENDOR_TAG_REFUSES_CARRIED: &str = r#"
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
        log.write("session/open\n")
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        vendor = params.get("vendor")
        log.write("session/prompt|{}\n".format(vendor or "unset"))
        log.flush()
        if vendor == "v:vnd":
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
        else:
            send({"jsonrpc": "2.0", "id": request_id,
                  "result": {"text": "served|peerdefault|{}".format(
                      params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_scaffolded_slot_never_counts_as_model_carry() {
    // The prompt's vendor field is `v:{modelProvider}`: the qualified
    // `vnd/prime` resolves the token and the refusal it draws advances
    // the chain, but the unqualified `fallback` still emits a populated
    // `v:` — scaffolding, not a selector the peer could have bound. The
    // peer serves its own default; nothing on the wire carried
    // `fallback`, so it must not publish as the active model.
    let dir = TestDirectory::new("rpc-scaffolded-slot");
    let script = dir.path().join("rpc_vendor_tag.py");
    std::fs::write(&script, RPC_VENDOR_TAG_REFUSES_CARRIED).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "scaffolded",
        &script,
        json!({}),
        json!({"vendor": "v:{modelProvider}", "text": "{prompt}"}),
    );
    attach_call_log(&mut primary, &log);
    let registry = registry_with_lanes(
        primary,
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("scaffolded".to_owned()),
            model: Some("vnd/prime".to_owned()),
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
    assert_eq!(outcome.text.trim_end(), "served|peerdefault|work");
    assert_eq!(
        session.active_model().as_deref(),
        Some("vnd/prime"),
        "a populated scaffolding field is not carry evidence"
    );
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec!["session/open", "session/prompt|v:vnd", "session/prompt|v:"],
        "the fallback attempt emitted only the literal scaffold: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer that serves its process's first open, refuses every
/// prompt with quota, and exits mid-handshake on a *reopened* session —
/// the marker file survives across spawns, so the candidate's setup
/// send is journaled before the lane dies.
#[cfg(unix)]
const RPC_DIES_ON_REOPEN: &str = r#"
import json, os, sys

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

log = open(os.environ["AWP_CALL_LOG"], "a")
marker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "opened")
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
        if os.path.exists(marker):
            sys.exit(0)
        open(marker, "w").close()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        log.write("session/prompt\n")
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "error": {"code": -32000, "message":
                        "resource_exhausted: quota exceeded for this model"}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

/// An rpc peer on the second lane: a model-free open and a prompt whose
/// vendor field emits only scaffolding for an unqualified candidate.
#[cfg(unix)]
const RPC_VENDOR_TAG_ALWAYS_SERVES: &str = r#"
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
        log.write("session/open\n")
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        vendor = params.get("vendor")
        log.write("session/prompt|{}\n".format(vendor or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "served|peerdefault|{}".format(
                  params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_dead_lane_send_never_publishes_the_candidate() {
    // The primary lane binds its model at session establishment only:
    // the prompt refusal advances the chain into a session-bound retry,
    // and the reopen puts `fallback` on the wire — then the lane dies
    // mid-handshake. The alternate's turn surface is `v:{modelProvider}`
    // only: the unqualified candidate emits scaffolding, the peer
    // serves its own default, and the dead lane's send must not satisfy
    // the publication guard — a send on a lane that never served proves
    // nothing about this session.
    let dir = TestDirectory::new("rpc-dead-lane-send");
    let dead_script = dir.path().join("rpc_dead_open.py");
    std::fs::write(&dead_script, RPC_DIES_ON_REOPEN).unwrap();
    let serve_script = dir.path().join("rpc_serving.py");
    std::fs::write(&serve_script, RPC_VENDOR_TAG_ALWAYS_SERVES).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "bindlane",
        &dead_script,
        json!({"model": "{model}"}),
        json!({"text": "{prompt}"}),
    );
    attach_call_log(&mut primary, &log);
    let mut alternate = rpc_spec(
        "omitlane",
        &serve_script,
        json!({"session": "new"}),
        json!({"vendor": "v:{modelProvider}", "text": "{prompt}"}),
    );
    attach_call_log(&mut alternate, &log);
    let registry = registry_with_lanes(
        primary,
        alternate,
        ProfilePatch {
            agent: Some("bindlane".to_owned()),
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

    let outcome = match session.prompt("work".to_owned()).await {
        Ok(outcome) => outcome,
        Err(error) => panic!(
            "prompt failed: {error:?}\ncalls: {}",
            std::fs::read_to_string(&log).unwrap_or_default()
        ),
    };
    assert_eq!(outcome.text.trim_end(), "served|peerdefault|work");
    assert_eq!(
        session.active_model().as_deref(),
        Some("prime"),
        "the serving lane never carried `fallback` — the dead lane's \
         send is requeue history, not publication evidence"
    );
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "session/open|prime",
            "session/prompt",
            "session/open|fallback",
            "session/open",
            "session/prompt|v:",
        ],
        "the dead lane's send stays spent; the serving lane emitted \
         only scaffolding: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer that refuses a `vnd` provider at open and opens its own
/// default when the provider member is omitted.
#[cfg(unix)]
const RPC_PROVIDER_OPEN_REFUSES_VENDOR: &str = r#"
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
        provider = params.get("provider")
        log.write("session/open|{}\n".format(provider or "unset"))
        log.flush()
        if provider == "vnd":
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
            continue
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        log.write("session/prompt\n")
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "served|peerdefault|{}".format(
                  params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_omitted_open_fallback_never_publishes() {
    // The open's only model surface is a whole `{modelProvider}` member:
    // the qualified `vnd/prime` resolves it, the refusal advances the
    // chain, and the unqualified `fallback` drops the member entirely —
    // the peer opens with its own default. The refused entry is spent
    // and the requested fallback was never on the wire: nothing
    // publishes — the omitted candidate stays pending until a serving
    // turn's frame provably carries it, and this prompt carries none.
    let dir = TestDirectory::new("rpc-omitted-open");
    let script = dir.path().join("rpc_provider_open.py");
    std::fs::write(&script, RPC_PROVIDER_OPEN_REFUSES_VENDOR).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "omittedopen",
        &script,
        json!({"provider": "{modelProvider}"}),
        json!({"text": "{prompt}"}),
    );
    attach_call_log(&mut primary, &log);
    let registry = registry_with(
        primary,
        ProfilePatch {
            agent: Some("omittedopen".to_owned()),
            model: Some("vnd/prime".to_owned()),
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
    assert_eq!(outcome.text.trim_end(), "served|peerdefault|work");
    assert_eq!(
        session.active_model().as_deref(),
        None,
        "the refused `vnd/prime` is spent and the omitted fallback \
         never reached the wire — neither publishes as the session's \
         model"
    );
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec!["session/open|vnd", "session/open|unset", "session/prompt"],
        "the unqualified open omitted the provider member: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer that refuses a `vnd` provider at open and echoes the
/// `model` the prompt actually carries — the serving turn's own proof.
#[cfg(unix)]
const RPC_PROVIDER_OPEN_SERVES_CARRIED: &str = r#"
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
        provider = params.get("provider")
        log.write("session/open|{}\n".format(provider or "unset"))
        log.flush()
        if provider == "vnd":
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
            continue
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        model = params.get("model")
        log.write("session/prompt|{}\n".format(model or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "served|{}|{}".format(
                  model or "unset", params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_serving_turn_commits_the_omitted_open_fallback() {
    // The open's model surface is a whole `{modelProvider}` member: the
    // qualified `vnd/prime` resolves it and is refused, the unqualified
    // `fallback` drops the member and opens the peer's default — the
    // fallback the session stores was never on the wire. The prompt's
    // `{model}` slot then carries the stored candidate on the first
    // implicit turn: a serving turn on the frame that provably carried
    // it is the commit evidence — the refused head never publishes and
    // the fallback no longer stays queued.
    let dir = TestDirectory::new("rpc-omitted-open-carry");
    let script = dir.path().join("rpc_provider_open.py");
    std::fs::write(&script, RPC_PROVIDER_OPEN_SERVES_CARRIED).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "carriedopen",
        &script,
        json!({"provider": "{modelProvider}"}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut primary, &log);
    let registry = registry_with(
        primary,
        ProfilePatch {
            agent: Some("carriedopen".to_owned()),
            model: Some("vnd/prime".to_owned()),
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
    assert_eq!(
        session.active_model().as_deref(),
        Some("fallback"),
        "the serving turn's frame provably carried the fallback — the \
         pending candidate commits and the refused head stays spent"
    );
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "session/open|vnd",
            "session/open|unset",
            "session/prompt|fallback"
        ],
        "the fallback reached the wire on the prompt, not the open: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer that refuses `prime` at `session/open` and dies on any
/// later open — the primary's establishment walk spends `prime`, then
/// its fallback open never completes.
#[cfg(unix)]
const RPC_REFUSES_OPEN_MODEL_THEN_DIES: &str = r#"
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
        log.write("primary/open|{}\n".format(model or "unset"))
        log.flush()
        if model == "prime":
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
        else:
            sys.exit(0)
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

/// An rpc peer that logs the model each `session/open` and
/// `session/prompt` carried and serves every call — the alternate that
/// accepts whatever the chain cursor hands it.
#[cfg(unix)]
const RPC_LOGGED_MODEL_SERVE: &str = r#"
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
        log.write("alt/open|{}\n".format(session_model or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        model = params.get("model") or session_model
        log.write("alt/prompt|{}\n".format(model or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "served|{}|{}".format(
                  model or "unset", params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_alternate_lane_resumes_the_chain_at_the_unrefused_entry() {
    // The primary's establishment walk already spent `prime` — a refused
    // binding open — and died opening the fallback. Starting the
    // alternate's walk at the chain head again resends the refused entry:
    // the alternate must resume where the primary's walk left the cursor.
    let dir = TestDirectory::new("rpc-chain-cursor");
    let dying = dir.path().join("rpc_die.py");
    std::fs::write(&dying, RPC_REFUSES_OPEN_MODEL_THEN_DIES).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "cursor",
        &dying,
        json!({"model": "{model}"}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut primary, &log);
    let serving = dir.path().join("rpc_serve.py");
    std::fs::write(&serving, RPC_LOGGED_MODEL_SERVE).unwrap();
    let mut alternate = rpc_spec(
        "serving",
        &serving,
        json!({"model": "{model}"}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut alternate, &log);
    let registry = registry_with_lanes(
        primary,
        alternate,
        ProfilePatch {
            agent: Some("cursor".to_owned()),
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
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "primary/open|prime",
            "primary/open|fallback",
            "alt/open|fallback",
            "alt/prompt|fallback",
        ],
        "the refused head must never reach the alternate's wire: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer that refuses `vnd` at `session/open`, answers an open
/// whose provider member is omitted, then closes its stdin — the
/// established lane dies before the first prompt and the turn replays
/// on the alternate.
#[cfg(unix)]
const RPC_PROVIDER_OPEN_THEN_CLOSES_STDIN: &str = r#"
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
    params = request.get("params") or {}
    if method == "session/open":
        provider = params.get("provider")
        log.write("primary/open|{}\n".format(provider or "unset"))
        log.flush()
        if provider == "vnd":
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
            continue
        # stdin closes before the answer flushes, so the prompt write is
        # guaranteed to land on a dead pipe
        os.close(0)
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
        time.sleep(30)
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

/// An rpc peer that logs the `provider` its open and the `vendor`
/// scaffold its prompt carried, and serves every call.
#[cfg(unix)]
const RPC_LOGGED_PROVIDER_VENDOR_SERVE: &str = r#"
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
        log.write("alt/open|{}\n".format(params.get("provider") or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        vendor = params.get("vendor") or "unset"
        log.write("alt/prompt|{}\n".format(vendor))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "served|{}|{}".format(
                  vendor, params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_pending_candidate_stays_pending_across_an_omitting_alternate() {
    // `fallback` opened the primary with its `provider` member omitted —
    // pending, never carried. The lane dies before the prompt, so the
    // replacement open rides the same pending candidate: the alternate's
    // open and prompt both offer only an unqualified `{modelProvider}`
    // surface, so the wire carries no selector at all. Nothing ever
    // carried `fallback` on the serving lane — publishing it as the
    // session's active model would name a model the peer never bound.
    let dir = TestDirectory::new("rpc-alt-pending");
    let dying = dir.path().join("rpc_close.py");
    std::fs::write(&dying, RPC_PROVIDER_OPEN_THEN_CLOSES_STDIN).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "pending",
        &dying,
        json!({"provider": "{modelProvider}"}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut primary, &log);
    let serving = dir.path().join("rpc_serve.py");
    std::fs::write(&serving, RPC_LOGGED_PROVIDER_VENDOR_SERVE).unwrap();
    let mut alternate = rpc_spec(
        "serving",
        &serving,
        json!({"provider": "{modelProvider}"}),
        json!({"vendor": "v:{modelProvider}", "text": "{prompt}"}),
    );
    attach_call_log(&mut alternate, &log);
    let registry = registry_with_lanes(
        primary,
        alternate,
        ProfilePatch {
            agent: Some("pending".to_owned()),
            model: Some("vnd/prime".to_owned()),
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
    assert_eq!(outcome.text.trim_end(), "served|v:|work");
    assert_eq!(
        session.active_model().as_deref(),
        None,
        "no frame on the serving lane ever carried the omitted candidate"
    );
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "primary/open|vnd",
            "primary/open|unset",
            "alt/open|unset",
            "alt/prompt|v:",
        ],
        "the alternate carries only scaffolding — never the candidate: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer that refuses `vnd` at `session/open`, accepts an omitted
/// provider, refuses a prompt carrying `fallback`, and serves a prompt
/// whose model member is dropped with its own default.
#[cfg(unix)]
const RPC_PROVIDER_OPEN_REFUSES_CARRIED_PROMPT: &str = r#"
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
        provider = params.get("provider")
        log.write("session/open|{}\n".format(provider or "unset"))
        log.flush()
        if provider == "vnd":
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
            continue
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        model = params.get("model")
        log.write("session/prompt|{}\n".format(model or "unset"))
        log.flush()
        if model == "fallback":
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
        else:
            send({"jsonrpc": "2.0", "id": request_id,
                  "result": {"text": "served|{}|{}".format(
                      model or "peerdefault", params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_refused_pending_candidate_retires_the_session_default() {
    // `fallback` opened the session with its `provider` member omitted —
    // the unconfirmed candidate stays parked out of the stored slot, so
    // no session default ever inherits it. The first prompt carries it
    // on the `{model}` slot and the peer refuses it; with the chain
    // exhausted the next implicit prompt must dispatch model-free on
    // the session that never stored the refused value — no repair
    // re-open is needed and nothing is resent.
    let dir = TestDirectory::new("rpc-retire-default");
    let script = dir.path().join("rpc_provider.py");
    std::fs::write(&script, RPC_PROVIDER_OPEN_REFUSES_CARRIED_PROMPT).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "retiring",
        &script,
        json!({"provider": "{modelProvider}"}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut primary, &log);
    let registry = registry_with(
        primary,
        ProfilePatch {
            agent: Some("retiring".to_owned()),
            model: Some("vnd/prime".to_owned()),
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
    assert_eq!(session.active_model().as_deref(), None);

    // The session stores no model: the next implicit prompt's `{model}`
    // member drops and the peer serves its own default — the refused
    // candidate is never resent and no session is re-established.
    let outcome = session.prompt("again".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "served|peerdefault|again");
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "session/open|vnd",
            "session/open|unset",
            "session/prompt|fallback",
            "session/prompt|unset",
        ],
        "the refused candidate must never become the stored default: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer that refuses the explicit `session/open` model and dies
/// on any later open — the primary's establishment walk spends the
/// refused entry, then its fallback open never completes.
#[cfg(unix)]
const RPC_REFUSES_EXPLICIT_OPEN_THEN_DIES: &str = r#"
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
        log.write("primary/open|{}\n".format(model or "unset"))
        log.flush()
        if model == "explicit":
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
        else:
            sys.exit(0)
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_alternate_lane_resumes_the_chain_past_the_refused_explicit() {
    // The caller's explicit model is the primary's first candidate: the
    // peer refuses it at open and the lane dies opening the chain head
    // the walk advanced to. Replaying the original request on the
    // alternate hands it the refused explicit again — the cursor the
    // primary's walk left is the model that must reach the wire.
    let dir = TestDirectory::new("rpc-explicit-cursor");
    let dying = dir.path().join("rpc_die.py");
    std::fs::write(&dying, RPC_REFUSES_EXPLICIT_OPEN_THEN_DIES).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "cursor",
        &dying,
        json!({"model": "{model}"}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut primary, &log);
    let serving = dir.path().join("rpc_serve.py");
    std::fs::write(&serving, RPC_LOGGED_MODEL_SERVE).unwrap();
    let mut alternate = rpc_spec(
        "serving",
        &serving,
        json!({"model": "{model}"}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut alternate, &log);
    let registry = registry_with_lanes(
        primary,
        alternate,
        ProfilePatch {
            agent: Some("cursor".to_owned()),
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
                config: SessionConfig {
                    model: Some("explicit".to_owned()),
                    ..Default::default()
                },
                ..Default::default()
            },
            tx,
        )
        .await
        .unwrap();

    let outcome = session.prompt("work".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "served|prime|work");
    assert_eq!(session.active_model().as_deref(), Some("prime"));
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "primary/open|explicit",
            "primary/open|prime",
            "alt/open|prime",
            "alt/prompt|prime",
        ],
        "the refused explicit must never reach the alternate's wire: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer that refuses `vnd` at `session/open`, accepts an omitted
/// provider, refuses prompts carrying `fallback` or `third`, and serves
/// a prompt whose model member is dropped with its own default.
#[cfg(unix)]
const RPC_PROVIDER_OPEN_REFUSES_TWO_CARRIED: &str = r#"
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
        provider = params.get("provider")
        log.write("session/open|{}\n".format(provider or "unset"))
        log.flush()
        if provider == "vnd":
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
            continue
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        model = params.get("model")
        log.write("session/prompt|{}\n".format(model or "unset"))
        log.flush()
        if model == "fallback" or model == "third":
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
        else:
            send({"jsonrpc": "2.0", "id": request_id,
                  "result": {"text": "served|{}|{}".format(
                      model or "peerdefault", params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_refused_stored_default_retires_after_a_later_refusal() {
    // `fallback` opened the session with its `provider` member omitted —
    // parked, pending, never stored. The prompt carrying it is refused,
    // and the retry's per-turn candidate `third` is refused too. The
    // chain's exhaustion must leave no trace of either candidate in the
    // session's stored default: the next implicit prompt goes out
    // model-free without a repair re-open.
    let dir = TestDirectory::new("rpc-retire-two");
    let script = dir.path().join("rpc_provider.py");
    std::fs::write(&script, RPC_PROVIDER_OPEN_REFUSES_TWO_CARRIED).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "retiring",
        &script,
        json!({"provider": "{modelProvider}"}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut primary, &log);
    let registry = registry_with(
        primary,
        ProfilePatch {
            agent: Some("retiring".to_owned()),
            model: Some("vnd/prime".to_owned()),
            models: Some(vec!["fallback".to_owned(), "third".to_owned()]),
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
    assert_eq!(session.active_model().as_deref(), None);

    // The session stores no model: the next implicit prompt's `{model}`
    // member drops and the peer serves its own default.
    let outcome = session.prompt("again".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "served|peerdefault|again");
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "session/open|vnd",
            "session/open|unset",
            "session/prompt|fallback",
            "session/prompt|third",
            "session/prompt|unset",
        ],
        "the refused candidate must never become the stored default: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer that accepts a model-free `session/open` — the stored
/// default its request carries — logs each open's model-or-provider,
/// refuses every prompt, then closes its stdin so the lane dies for
/// writes. Open calls are counted in a state file: `AWP_OPEN_DIE` names
/// the count on which the fresh process exits, `AWP_OPEN_PARK` the one
/// it parks for a minute.
#[cfg(unix)]
const RPC_STORED_REFUSES_PROMPT_THEN_DIES: &str = r#"
import json, os, sys, time

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

log = open(os.environ["AWP_CALL_LOG"], "a")
state = os.environ["AWP_OPEN_STATE"]
die_on = int(os.environ.get("AWP_OPEN_DIE", "0") or "0")
park_on = int(os.environ.get("AWP_OPEN_PARK", "0") or "0")
count = 0
if os.path.exists(state):
    try:
        count = int(open(state).read().strip() or "0")
    except Exception:
        count = 0
for line in sys.stdin:
    try:
        request = json.loads(line)
    except Exception:
        continue
    method = request.get("method")
    request_id = request.get("id")
    params = request.get("params") or {}
    if method == "session/open":
        count += 1
        with open(state, "w") as handle:
            handle.write(str(count))
        model = params.get("model") or params.get("provider")
        log.write("primary/open|{}\n".format(model or "unset"))
        log.flush()
        if count == die_on:
            sys.exit(0)
        if count == park_on:
            time.sleep(60)
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        model = params.get("model")
        log.write("primary/prompt|{}\n".format(model or "unset"))
        log.flush()
        # stdin closes before the answer flushes, so the next write is
        # guaranteed to land on a dead pipe
        os.close(0)
        send({"jsonrpc": "2.0", "id": request_id,
              "error": {"code": -32000, "message":
                        "resource_exhausted: quota exceeded for this model"}})
        time.sleep(30)
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_failed_retire_never_resends_the_poisoned_default() {
    // `fallback` is the stored session default an open that never
    // carried it left in place — the session bound it as its active
    // model. The prompt carrying it is refused and the lane dies; the
    // lane walk's replacement open binds it on the wire and the
    // alternate refuses it, poisoning the installed session's stored
    // slot. The repair reopen then dies too, leaving the mark behind:
    // the walk must continue past the dead repair instead of resending
    // `fallback`, and the alternate replays model-free.
    let dir = TestDirectory::new("rpc-retire-fail");
    let script = dir.path().join("rpc_provider.py");
    std::fs::write(&script, RPC_STORED_REFUSES_PROMPT_THEN_DIES).unwrap();
    let log = dir.path().join("calls.log");
    let state = dir.path().join("opens.count");
    let mut primary = rpc_spec(
        "retiring",
        &script,
        json!({"note": "stored"}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    primary.command.env = EnvPolicy::Sanitized {
        inherit: vec![],
        set: [
            (
                "AWP_CALL_LOG".to_owned(),
                EnvValue::Literal(log.display().to_string()),
            ),
            (
                "AWP_OPEN_STATE".to_owned(),
                EnvValue::Literal(state.display().to_string()),
            ),
            ("AWP_OPEN_DIE".to_owned(), EnvValue::Literal("2".to_owned())),
        ]
        .into_iter()
        .collect(),
    };
    let refusing = dir.path().join("rpc_refuse.py");
    std::fs::write(&refusing, RPC_ALT_OPEN_REFUSES_FALLBACK).unwrap();
    let mut alternate = rpc_spec(
        "refusing",
        &refusing,
        json!({"model": "{model}"}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut alternate, &log);
    let registry = registry_with_lanes(
        primary,
        alternate,
        ProfilePatch {
            agent: Some("retiring".to_owned()),
            model: Some("fallback".to_owned()),
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

    // The lane switch's open binds `fallback` on the wire and the
    // alternate refuses it: the installed primary's stored default is
    // marked poisoned and the entry is spent.
    let error = session.prompt("two".to_owned()).await.unwrap_err();
    assert_eq!(error.code, "agent_quota_exhausted");

    // The repair reopen on the marked primary dies — the walk carries
    // on to the alternate, which opens model-free and replays with no
    // model at all: the poisoned default is never resent.
    let outcome = session.prompt("again".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "served|unset|again");
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "primary/open|unset",
            "primary/prompt|fallback",
            "alt/open|fallback",
            "primary/open|unset",
            "alt/open|unset",
            "alt/prompt|unset",
        ],
        "the poisoned stored default must never be resent: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer that logs the provider its open carried, parks a
/// model-free `session/prepare` pre-turn call for seconds, and serves
/// prompts with the model they name.
#[cfg(unix)]
const RPC_SLOW_PREPARE_SERVE: &str = r#"
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
    params = request.get("params") or {}
    if method == "session/open":
        log.write("alt/open|{}\n".format(params.get("provider") or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prepare":
        log.write("alt/prepare\n")
        log.flush()
        time.sleep(2.0)
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
    elif method == "session/prompt":
        model = params.get("model")
        log.write("alt/prompt|{}\n".format(model or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "served|{}|{}".format(
                  model or "unset", params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_pending_candidate_survives_a_cancelled_replay() {
    // `fallback` opened the primary with its `provider` member omitted —
    // stored, pending. The lane dies before the first prompt, so the
    // replacement open rides the same pending candidate: the alternate
    // can carry it only on turns, so installation keeps the shared
    // pending claim while the replay holds the candidate turn-locally.
    // A cancel that stops the replay before dispatch must preserve the
    // candidate across turns — the next implicit prompt still projects
    // it.
    let dir = TestDirectory::new("rpc-pending-cancel");
    let dying = dir.path().join("rpc_close.py");
    std::fs::write(&dying, RPC_PROVIDER_OPEN_THEN_CLOSES_STDIN).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "pending",
        &dying,
        json!({"provider": "{modelProvider}"}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut primary, &log);
    let serving = dir.path().join("rpc_prepare.py");
    std::fs::write(&serving, RPC_SLOW_PREPARE_SERVE).unwrap();
    let mut alternate = rpc_spec(
        "serving",
        &serving,
        json!({"provider": "{modelProvider}"}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    let LaneSpec::Rpc(lane) = &mut alternate.lane else {
        panic!("rpc_spec builds an rpc lane")
    };
    lane.pre_turn = vec![RpcCallSpec {
        method: "session/prepare".to_owned(),
        params: json!({"prepare": true}),
    }];
    attach_call_log(&mut alternate, &log);
    let registry = registry_with_lanes(
        primary,
        alternate,
        ProfilePatch {
            agent: Some("pending".to_owned()),
            model: Some("vnd/prime".to_owned()),
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
    // The replay's model-free pre-turn call parks for seconds: once it
    // is on the wire the cancel lands inside the window before the
    // prompt frame can dispatch.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    loop {
        let contents = std::fs::read_to_string(&log).unwrap_or_default();
        if contents.lines().any(|line| line == "alt/prepare") {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the replay's pre-turn call never dispatched: {contents}"
        );
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    session.cancel().await.unwrap();

    let outcome = driver.await.unwrap().unwrap();
    assert_eq!(outcome.stop_reason, "cancelled");

    // The candidate the cancelled replay never sent is still the
    // chain's awaiting entry: the next implicit prompt projects it —
    // the wire finally carries `fallback` and the peer accepts it.
    let outcome = session.prompt("again".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "served|fallback|again");
    assert_eq!(session.active_model().as_deref(), Some("fallback"));
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "primary/open|vnd",
            "primary/open|unset",
            "alt/open|unset",
            "alt/prepare",
            "alt/prepare",
            "alt/prompt|fallback",
        ],
        "the unsent pending candidate must ride the next implicit prompt: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer that logs the provider each open carried, refuses a
/// prompt carrying `fallback`, serves any other prompt exactly once —
/// then closes its stdin, so the next turn's write lands on a dead pipe.
#[cfg(unix)]
const RPC_ALT_REFUSES_THEN_DIES: &str = r#"
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
    params = request.get("params") or {}
    if method == "session/open":
        log.write("alt/open|{}\n".format(params.get("provider") or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        model = params.get("model")
        log.write("alt/prompt|{}\n".format(model or "unset"))
        log.flush()
        if model == "fallback":
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
        else:
            # stdin closes before the answer flushes, so the next write
            # is guaranteed to land on a dead pipe
            os.close(0)
            send({"jsonrpc": "2.0", "id": request_id,
                  "result": {"text": "served|{}|{}".format(
                      model or "unset", params.get("text", ""))}})
            time.sleep(30)
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

/// An rpc peer that logs the provider each open and the model each
/// prompt carried under a `third/` tag, and serves every call.
#[cfg(unix)]
const RPC_LOGGED_MODEL_SERVE_THIRD: &str = r#"
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
        log.write("third/open|{}\n".format(params.get("provider") or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        model = params.get("model")
        log.write("third/prompt|{}\n".format(model or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "served|{}|{}".format(
                  model or "unset", params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_refused_pending_never_reenters_a_later_lane() {
    // `fallback` opened the primary with its `provider` member omitted —
    // parked, pending, never stored. The alternate can carry it only on
    // turns, so the shared pending claim stays set while the replay
    // holds it turn-locally. The replay's `fallback` refusal must clear
    // that claim: a later lane switch would otherwise pick the refused
    // value again — and the alternate's own session, whose stored slot
    // the unconfirmed candidate never entered, replays model-free with
    // no repair re-open.
    let dir = TestDirectory::new("rpc-pending-refused");
    let dying = dir.path().join("rpc_close.py");
    std::fs::write(&dying, RPC_PROVIDER_OPEN_THEN_CLOSES_STDIN).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "pending",
        &dying,
        json!({"provider": "{modelProvider}"}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut primary, &log);
    let alt_script = dir.path().join("rpc_alt.py");
    std::fs::write(&alt_script, RPC_ALT_REFUSES_THEN_DIES).unwrap();
    let mut alternate = rpc_spec(
        "alternate",
        &alt_script,
        json!({"provider": "{modelProvider}"}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut alternate, &log);
    let third_script = dir.path().join("rpc_third.py");
    std::fs::write(&third_script, RPC_LOGGED_MODEL_SERVE_THIRD).unwrap();
    let mut third = rpc_spec(
        "third",
        &third_script,
        json!({"provider": "{modelProvider}"}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut third, &log);
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
            agent: Some("pending".to_owned()),
            model: Some("vnd/prime".to_owned()),
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

    // The replay carries `fallback` on the alternate's turn surface and
    // the peer refuses it — the shared pending claim must be spent.
    let error = session.prompt("work".to_owned()).await.unwrap_err();
    assert_eq!(error.code, "agent_quota_exhausted");

    // The alternate's session never stored the refused candidate — the
    // unconfirmed claim stayed out of its stored slot — so the next
    // implicit prompt replays model-free with no repair at all.
    let outcome = session.prompt("two".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "served|unset|two");

    // The alternate dies on the next write: the lane switch must not
    // pick the refused `fallback` out of a stale pending claim — the
    // replay on the third lane carries no model at all.
    let outcome = session.prompt("three".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "served|unset|three");
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "primary/open|vnd",
            "primary/open|unset",
            "alt/open|unset",
            "alt/prompt|fallback",
            "alt/prompt|unset",
            "third/open|unset",
            "third/prompt|unset",
        ],
        "the refused pending candidate must never re-enter a lane: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer that logs the model each open and prompt carried under
/// an `alt/` tag, refuses an open carrying `fallback`, and serves
/// everything else.
#[cfg(unix)]
const RPC_ALT_OPEN_REFUSES_FALLBACK: &str = r#"
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
        log.write("alt/open|{}\n".format(model or "unset"))
        log.flush()
        if model == "fallback":
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
            continue
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        model = params.get("model")
        log.write("alt/prompt|{}\n".format(model or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "served|{}|{}".format(
                  model or "unset", params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_open_refused_pending_never_reenters_the_next_turn() {
    // `fallback` opened the primary with its `provider` member omitted —
    // parked, pending, never stored. The lane dies before the first
    // prompt, so the replacement open rides the shared pending claim —
    // and the alternate's open binds it on the wire, where the peer
    // refuses it. The chain is exhausted, so no replacement installs:
    // the claim must be spent instead of re-entering on the next
    // implicit prompt.
    let dir = TestDirectory::new("rpc-open-refused-pending");
    let dying = dir.path().join("rpc_close.py");
    std::fs::write(&dying, RPC_PROVIDER_OPEN_THEN_CLOSES_STDIN).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "pending",
        &dying,
        json!({"provider": "{modelProvider}"}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut primary, &log);
    let refusing = dir.path().join("rpc_refuse.py");
    std::fs::write(&refusing, RPC_ALT_OPEN_REFUSES_FALLBACK).unwrap();
    let mut alternate = rpc_spec(
        "refusing",
        &refusing,
        json!({"model": "{model}"}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut alternate, &log);
    let registry = registry_with_lanes(
        primary,
        alternate,
        ProfilePatch {
            agent: Some("pending".to_owned()),
            model: Some("vnd/prime".to_owned()),
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

    // The alternate's open carries `fallback` on the wire and the peer
    // refuses it — the shared pending claim is spent with the open.
    let error = session.prompt("work".to_owned()).await.unwrap_err();
    assert_eq!(error.code, "agent_quota_exhausted");

    // The primary's stored slot stayed clean — the unconfirmed claim
    // never entered it — so no repair is owed: the dead lane's walk
    // opens the alternate model-free and the prompt replays with no
    // model at all.
    let outcome = session.prompt("again".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "served|unset|again");
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "primary/open|vnd",
            "primary/open|unset",
            "alt/open|fallback",
            "alt/open|unset",
            "alt/prompt|unset",
        ],
        "the open-refused pending candidate must never re-enter: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer that logs the provider its open and prompt carried under
/// an `alt/` tag, refuses the first prompt with a quota error, then
/// closes its stdin so the next write lands on a dead pipe.
#[cfg(unix)]
const RPC_ALT_REFUSES_PROMPT_THEN_DIES: &str = r#"
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
    params = request.get("params") or {}
    if method == "session/open":
        log.write("alt/open|{}\n".format(params.get("provider") or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        log.write("alt/prompt|{}\n".format(params.get("provider") or "unset"))
        log.flush()
        # stdin closes before the answer flushes, so the next write is
        # guaranteed to land on a dead pipe
        os.close(0)
        send({"jsonrpc": "2.0", "id": request_id,
              "error": {"code": -32000, "message":
                        "resource_exhausted: quota exceeded for this model"}})
        time.sleep(30)
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_model_free_refusal_keeps_the_unsent_candidate() {
    // `fallback` opened the primary with its `provider` member omitted —
    // stored, pending. The alternate prepares it for a per-turn slot
    // that resolves absent on an unqualified model: the replayed prompt
    // is model-free, so its quota refusal is the lane's answer — never
    // proof the peer refused `fallback`. The shared pending claim must
    // survive for a lane that can actually carry it — and the walk
    // reaches that lane inside the same turn.
    let dir = TestDirectory::new("rpc-model-free-refusal");
    let dying = dir.path().join("rpc_close.py");
    std::fs::write(&dying, RPC_PROVIDER_OPEN_THEN_CLOSES_STDIN).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "pending",
        &dying,
        json!({"provider": "{modelProvider}"}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut primary, &log);
    let refusing = dir.path().join("rpc_refuse.py");
    std::fs::write(&refusing, RPC_ALT_REFUSES_PROMPT_THEN_DIES).unwrap();
    let mut alternate = rpc_spec(
        "refusing",
        &refusing,
        json!({"provider": "{modelProvider}"}),
        json!({"provider": "{modelProvider}", "text": "{prompt}"}),
    );
    attach_call_log(&mut alternate, &log);
    let serving = dir.path().join("rpc_serve.py");
    std::fs::write(&serving, RPC_LOGGED_MODEL_SERVE_THIRD).unwrap();
    let mut third = rpc_spec(
        "serving",
        &serving,
        json!({"provider": "{model}"}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut third, &log);
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
            agent: Some("pending".to_owned()),
            model: Some("vnd/prime".to_owned()),
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

    // The replayed prompt is model-free — the candidate's
    // `{modelProvider}` resolves absent — so the refusal cannot spend a
    // claim the peer never saw: the lane answer walks the surviving
    // claim to the third lane, whose open and prompt carry `fallback`
    // to a peer that serves it in the same turn.
    let outcome = session.prompt("work".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "served|fallback|work");
    assert_eq!(session.active_model().as_deref(), Some("fallback"));

    // The bound session keeps serving the model it accepted.
    let outcome = session.prompt("again".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "served|fallback|again");
    assert_eq!(session.active_model().as_deref(), Some("fallback"));
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "primary/open|vnd",
            "primary/open|unset",
            "alt/open|unset",
            "alt/prompt|unset",
            "third/open|fallback",
            "third/prompt|fallback",
            "third/prompt|fallback",
        ],
        "an unsent candidate must survive a model-free refusal: {calls_log}"
    );
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn rpc_direct_prompt_never_inherits_the_pending_candidate() {
    // `fallback` opened the primary with its `provider` member omitted —
    // stored, pending. The alternate's open offers no model slot, so the
    // replacement stores nothing: the pending claim can only ride the
    // replayed request, and a cancel stops the replay before the prompt
    // dispatches. `prompt_direct` fills no defaults — the queued
    // candidate must not leak into its request — while the claim still
    // rides the next implicit prompt.
    let dir = TestDirectory::new("rpc-direct-pending");
    let dying = dir.path().join("rpc_close.py");
    std::fs::write(&dying, RPC_PROVIDER_OPEN_THEN_CLOSES_STDIN).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "pending",
        &dying,
        json!({"provider": "{modelProvider}"}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut primary, &log);
    let serving = dir.path().join("rpc_prepare.py");
    std::fs::write(&serving, RPC_SLOW_PREPARE_SERVE).unwrap();
    let mut alternate = rpc_spec(
        "serving",
        &serving,
        json!({"note": "alternate"}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    let LaneSpec::Rpc(lane) = &mut alternate.lane else {
        panic!("rpc_spec builds an rpc lane")
    };
    lane.pre_turn = vec![RpcCallSpec {
        method: "session/prepare".to_owned(),
        params: json!({"prepare": true}),
    }];
    attach_call_log(&mut alternate, &log);
    let registry = registry_with_lanes(
        primary,
        alternate,
        ProfilePatch {
            agent: Some("pending".to_owned()),
            model: Some("vnd/prime".to_owned()),
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
    // The replay's model-free pre-turn call parks for seconds: once it
    // is on the wire the cancel lands inside the window before the
    // prompt frame can dispatch.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    loop {
        let contents = std::fs::read_to_string(&log).unwrap_or_default();
        if contents.lines().any(|line| line == "alt/prepare") {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the replay's pre-turn call never dispatched: {contents}"
        );
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    session.cancel().await.unwrap();
    let outcome = driver.await.unwrap().unwrap();
    assert_eq!(outcome.stop_reason, "cancelled");

    // The direct prompt inherits nothing: no `{model}` member reaches
    // the wire, and the peer answers with its own default.
    let outcome = session
        .prompt_direct(TurnRequest {
            text: "raw".to_owned(),
            model: None,
            effort: None,
        })
        .await
        .unwrap();
    assert_eq!(outcome.text.trim_end(), "served|unset|raw");
    assert_eq!(session.active_model().as_deref(), None);

    // The claim a direct turn never consumed still rides the next
    // implicit prompt — the wire finally carries `fallback`.
    let outcome = session.prompt("again".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "served|fallback|again");
    assert_eq!(session.active_model().as_deref(), Some("fallback"));
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "primary/open|vnd",
            "primary/open|unset",
            "alt/open|unset",
            "alt/prepare",
            "alt/prepare",
            "alt/prompt|unset",
            "alt/prepare",
            "alt/prompt|fallback",
        ],
        "a direct prompt must never inherit the queued model: {calls_log}"
    );
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn rpc_closed_session_skips_the_poisoned_repair() {
    // `fallback` is the stored session default — bound active by an
    // open that never carried it. The prompt carrying it is refused
    // and the lane dies; the lane walk's replacement open binds it on
    // the wire and the alternate refuses it, so the installed session
    // keeps the poisoned stored default with its mark. A prompt on a
    // closed session must report the close: the repair iteration
    // dispatches nothing, so the lifecycle answer lands before the
    // synthesized refusal.
    let dir = TestDirectory::new("rpc-poisoned-close");
    let script = dir.path().join("rpc_provider.py");
    std::fs::write(&script, RPC_STORED_REFUSES_PROMPT_THEN_DIES).unwrap();
    let log = dir.path().join("calls.log");
    let state = dir.path().join("opens.count");
    let mut primary = rpc_spec(
        "retiring",
        &script,
        json!({"note": "stored"}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    primary.command.env = EnvPolicy::Sanitized {
        inherit: vec![],
        set: [
            (
                "AWP_CALL_LOG".to_owned(),
                EnvValue::Literal(log.display().to_string()),
            ),
            (
                "AWP_OPEN_STATE".to_owned(),
                EnvValue::Literal(state.display().to_string()),
            ),
        ]
        .into_iter()
        .collect(),
    };
    let refusing = dir.path().join("rpc_refuse.py");
    std::fs::write(&refusing, RPC_ALT_OPEN_REFUSES_FALLBACK).unwrap();
    let mut alternate = rpc_spec(
        "refusing",
        &refusing,
        json!({"model": "{model}"}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut alternate, &log);
    let registry = registry_with_lanes(
        primary,
        alternate,
        ProfilePatch {
            agent: Some("retiring".to_owned()),
            model: Some("fallback".to_owned()),
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

    // The lane switch's open binds `fallback` on the wire and the
    // alternate refuses it: the installed primary keeps its poisoned
    // stored default under the mark.
    let error = session.prompt("two".to_owned()).await.unwrap_err();
    assert_eq!(error.code, "agent_quota_exhausted");

    // The lane still holds the poisoned stored default — but the close
    // answers first: no repair open is even attempted.
    session.close().await.unwrap();
    let error = session.prompt("again".to_owned()).await.unwrap_err();
    assert_eq!(error.code, "agent_session_closed");
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "primary/open|unset",
            "primary/prompt|fallback",
            "alt/open|fallback",
        ],
        "a closed session must not surface the synthesized refusal: {calls_log}"
    );
}

/// A secret resolver whose first lookup returns a value and every later
/// lookup comes back empty — one call's two `{secret:NAME}` placeholders
/// resolve to different answers inside a single substitution.
#[cfg(unix)]
struct FlipSecrets(std::sync::atomic::AtomicUsize);

#[cfg(unix)]
impl awp::SecretResolver for FlipSecrets {
    fn resolve<'life0, 'life1, 'async_trait>(
        &'life0 self,
        _reference: &'life1 str,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Option<String>> + Send + 'async_trait>>
    where
        'life0: 'async_trait,
        'life1: 'async_trait,
        Self: 'async_trait,
    {
        let count = self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Box::pin(async move { (count == 0).then(|| "v".to_owned()) })
    }
}

/// An rpc peer that refuses the secret-carrying pre-turn frame — marked
/// by its `hdr` member — and serves real prompts, which carry `payload`.
#[cfg(unix)]
const RPC_SHARED_METHOD_SECRET_HDR: &str = r#"
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
        log.write("session/open\n")
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        if "hdr" in params:
            log.write("pre|{}\n".format(params.get("hdr") or "unset"))
            log.flush()
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded"}})
        else:
            payload = params.get("payload", "")
            log.write("prompt|{}\n".format(payload))
            log.flush()
            send({"jsonrpc": "2.0", "id": request_id,
                  "result": {"text": "served|{}".format(payload)}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_mixed_secret_resolution_keeps_the_frame_on_its_call() {
    // One call's two `{secret:AWP_TOKEN}` placeholders resolve inside the
    // same substitution: the whole-leaf lookup returns `v` and the
    // embedded lookup comes back empty — the `auth` member drops and the
    // `hdr` token still substitutes the retained value. Collapsing both
    // facts into absent reads the frame's `hdr` as a literal token,
    // rejects the call's own frame, and hands a same-method prompt the
    // call's dispatch and refusal — binding the model to a refusal that
    // never touched it.
    let dir = TestDirectory::new("rpc-mixed-secret");
    let script = dir.path().join("rpc_hdr.py");
    std::fs::write(&script, RPC_SHARED_METHOD_SECRET_HDR).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "secrets",
        &script,
        json!({}),
        json!({"payload": "{prompt}|{model}"}),
    );
    {
        let LaneSpec::Rpc(lane) = &mut primary.lane else {
            panic!("rpc_spec builds an rpc lane")
        };
        lane.pre_turn = vec![RpcCallSpec {
            method: "session/prompt".to_owned(),
            params: json!({"auth": "{secret:AWP_TOKEN}", "hdr": "Bearer {secret:AWP_TOKEN}"}),
        }];
    }
    attach_call_log(&mut primary, &log);
    let registry = registry_with_lanes(
        primary,
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("secrets".to_owned()),
            model: Some("prime".to_owned()),
            ..Default::default()
        },
    );
    let agent = registry
        .resolve("p")
        .unwrap()
        .secrets(std::sync::Arc::new(FlipSecrets(
            std::sync::atomic::AtomicUsize::new(0),
        )));
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    // The refusal is the model-free pre-turn call's own answer — a lane
    // failure, not the prompt's model verdict — so the turn replays on
    // the alternate instead of advancing the chain.
    let outcome = session.prompt("work".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "alt|work");
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec!["session/open", "pre|Bearer v"],
        "the frame's substituted value keeps it on its own call: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer whose `session/prompt` method doubles as a configuration
/// call: a `note` member means the frame is the pre-turn call — served
/// until the peer has refused a real prompt, failed afterwards. Frames
/// carrying `payload` are prompts; `prime` payloads are quota-refused.
#[cfg(unix)]
const RPC_SHARED_METHOD_PRE_TURN_ERRORS: &str = r#"
import json, os, sys

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

log = open(os.environ["AWP_CALL_LOG"], "a")
refused = False
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
        if "note" in params:
            log.write("note|{}\n".format(params.get("note") or "unset"))
            log.flush()
            if refused:
                send({"jsonrpc": "2.0", "id": request_id,
                      "error": {"code": -32603, "message":
                                "internal failure in configure"}})
            else:
                send({"jsonrpc": "2.0", "id": request_id, "result": {}})
        else:
            payload = params.get("payload", "")
            log.write("session/prompt|{}\n".format(payload))
            log.flush()
            if payload.endswith("prime"):
                refused = True
                send({"jsonrpc": "2.0", "id": request_id,
                      "error": {"code": -32000, "message":
                                "resource_exhausted: quota exceeded for this model"}})
            else:
                send({"jsonrpc": "2.0", "id": request_id,
                      "result": {"text": "served|{}".format(payload)}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_pre_turn_frame_never_counts_as_the_prompt() {
    // A model-free pre-turn call shares the prompt's method, and the
    // prompt's own template mixes `{prompt}` with `{model}` — no
    // verifiable text position tells the frames apart on the wire. If
    // the probe owns the frame as the prompt anyway, the pre-turn
    // call's non-quota failure lands on a dispatched-looking record and
    // publishes the candidate the prompt never ran. Attributing the
    // frame to its actual call first keeps dispatch unset: the error
    // is the lane's answer, the turn falls to the alternate lane, and
    // the refused `prime` stays the published model.
    let dir = TestDirectory::new("rpc-shared-method");
    let script = dir.path().join("rpc_shared_method.py");
    std::fs::write(&script, RPC_SHARED_METHOD_PRE_TURN_ERRORS).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "sharedmethod",
        &script,
        json!({"model": "{model}"}),
        json!({"payload": "{prompt}|{model}"}),
    );
    {
        let LaneSpec::Rpc(lane) = &mut primary.lane else {
            panic!("rpc_spec builds an rpc lane")
        };
        lane.pre_turn = vec![RpcCallSpec {
            method: "session/prompt".to_owned(),
            params: json!({"note": "configure"}),
        }];
    }
    attach_call_log(&mut primary, &log);
    let registry = registry_with_lanes(
        primary,
        cli_spec("alive", "echo", &["alt|{prompt}"], PromptChannel::ArgOnly),
        ProfilePatch {
            agent: Some("sharedmethod".to_owned()),
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
    assert_eq!(
        session.active_model().as_deref(),
        Some("prime"),
        "the prompt never ran for `fallback` — a pre-turn frame's error \
         cannot publish it"
    );
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "session/open|prime",
            "note|configure",
            "session/prompt|work|prime",
            "note|configure",
        ],
        "the retry's only `session/prompt` frame was the pre-turn call: \
         {calls_log}"
    );
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn rpc_agent_lane_without_session_call_keeps_the_id_absent() {
    // An agent-scoped lane with no session-open call substitutes the
    // session id it holds — none: `{sessionId}` is known-absent, so the
    // id-only `session/select` resolves `{}` and the lane skips it. If
    // the probe leaves the id "maybe present" waiting on a response
    // that cannot exist, the skipped call claims the model frame and
    // the quota refusal reads as the lane's answer.
    let dir = TestDirectory::new("rpc-no-session-call");
    let script = dir.path().join("rpc_sessionless.py");
    std::fs::write(&script, RPC_SESSIONLESS_SELECT_REFUSES_PRIME).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "nosessioncall",
        &script,
        json!({}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    {
        let LaneSpec::Rpc(lane) = &mut primary.lane else {
            panic!("rpc_spec builds an rpc lane")
        };
        lane.session_open = None;
        lane.pre_turn = vec![
            RpcCallSpec {
                method: "session/select".to_owned(),
                params: json!({"session": "{sessionId}"}),
            },
            RpcCallSpec {
                method: "session/select".to_owned(),
                params: json!({"session": "{model}"}),
            },
        ];
    }
    attach_call_log(&mut primary, &log);
    let registry = registry_with(
        primary,
        ProfilePatch {
            agent: Some("nosessioncall".to_owned()),
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
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "session/select|prime",
            "session/select|fallback",
            "session/prompt|fallback",
        ],
        "no session call exists — the skipped id call cannot claim the \
         model frame: {calls_log}"
    );
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn rpc_null_params_open_binds_the_session_answer() {
    // The session call's params template is the default null: the lane
    // emits `params: null`, and the frame must claim that call — a
    // matcher that normalizes null to `{}` loses the journaled
    // identity, the open's answer never settles the session id's
    // presence, and the skipped id-only select claims the model frame
    // instead.
    let dir = TestDirectory::new("rpc-null-open");
    let script = dir.path().join("rpc_sessionless.py");
    std::fs::write(&script, RPC_SESSIONLESS_SELECT_REFUSES_PRIME).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "nullopen",
        &script,
        serde_json::Value::Null,
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    {
        let LaneSpec::Rpc(lane) = &mut primary.lane else {
            panic!("rpc_spec builds an rpc lane")
        };
        lane.pre_turn = vec![
            RpcCallSpec {
                method: "session/select".to_owned(),
                params: json!({"session": "{sessionId}"}),
            },
            RpcCallSpec {
                method: "session/select".to_owned(),
                params: json!({"session": "{model}"}),
            },
        ];
    }
    attach_call_log(&mut primary, &log);
    let registry = registry_with(
        primary,
        ProfilePatch {
            agent: Some("nullopen".to_owned()),
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
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "session/open",
            "session/select|prime",
            "session/select|fallback",
            "session/prompt|fallback",
        ],
        "the null-params open claimed its call and bound its answer: \
         {calls_log}"
    );
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn rpc_direct_prompt_leaves_the_pending_candidate_queued() {
    // `fallback` opened the primary with its `provider` member omitted —
    // stored, pending. The lane dies before the first turn, so the
    // replacement open happens inside a `prompt_direct` — which fills
    // no defaults: the queued claim must not bind at the replacement
    // open or ride the replayed request, and it still belongs to the
    // next implicit prompt.
    let dir = TestDirectory::new("rpc-direct-switch");
    let dying = dir.path().join("rpc_close.py");
    std::fs::write(&dying, RPC_PROVIDER_OPEN_THEN_CLOSES_STDIN).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "pending",
        &dying,
        json!({"provider": "{modelProvider}"}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut primary, &log);
    let serving = dir.path().join("rpc_serve.py");
    std::fs::write(&serving, RPC_LOGGED_MODEL_SERVE_THIRD).unwrap();
    let mut alternate = rpc_spec(
        "serving",
        &serving,
        json!({"note": "alternate"}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut alternate, &log);
    let registry = registry_with_lanes(
        primary,
        alternate,
        ProfilePatch {
            agent: Some("pending".to_owned()),
            model: Some("vnd/prime".to_owned()),
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

    // The direct turn replays on the alternate: no `{model}` member
    // reaches the wire — the queued claim stays queued.
    let outcome = session
        .prompt_direct(TurnRequest {
            text: "raw".to_owned(),
            model: None,
            effort: None,
        })
        .await
        .unwrap();
    assert_eq!(outcome.text.trim_end(), "served|unset|raw");
    assert_eq!(session.active_model().as_deref(), None);

    // The claim a direct turn never consumed still rides the next
    // implicit prompt — the wire finally carries `fallback`.
    let outcome = session.prompt("again".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "served|fallback|again");
    assert_eq!(session.active_model().as_deref(), Some("fallback"));
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "primary/open|vnd",
            "primary/open|unset",
            "third/open|unset",
            "third/prompt|unset",
            "third/prompt|fallback",
        ],
        "a direct prompt must leave the queued claim for implicit turns: {calls_log}"
    );
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn rpc_close_during_the_poisoned_repair_reports_session_closed() {
    // `fallback` is the stored session default — bound active by an
    // open that never carried it. The prompt carrying it is refused
    // and the lane dies; the lane walk's replacement open binds it on
    // the wire and the alternate refuses it, leaving the installed
    // session marked with the poisoned stored default. The next
    // prompt's repair parks inside the replacement open: a close
    // landing there must report the close — the repair dispatched no
    // prompt, so the synthesized model refusal is not the turn's
    // answer.
    let dir = TestDirectory::new("rpc-repair-close");
    let script = dir.path().join("rpc_provider.py");
    std::fs::write(&script, RPC_STORED_REFUSES_PROMPT_THEN_DIES).unwrap();
    let log = dir.path().join("calls.log");
    let state = dir.path().join("opens.count");
    let mut primary = rpc_spec(
        "parking",
        &script,
        json!({"note": "stored"}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    primary.command.env = EnvPolicy::Sanitized {
        inherit: vec![],
        set: [
            (
                "AWP_CALL_LOG".to_owned(),
                EnvValue::Literal(log.display().to_string()),
            ),
            (
                "AWP_OPEN_STATE".to_owned(),
                EnvValue::Literal(state.display().to_string()),
            ),
            (
                "AWP_OPEN_PARK".to_owned(),
                EnvValue::Literal("2".to_owned()),
            ),
        ]
        .into_iter()
        .collect(),
    };
    let refusing = dir.path().join("rpc_refuse.py");
    std::fs::write(&refusing, RPC_ALT_OPEN_REFUSES_FALLBACK).unwrap();
    let mut alternate = rpc_spec(
        "refusing",
        &refusing,
        json!({"model": "{model}"}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut alternate, &log);
    let registry = registry_with_lanes(
        primary,
        alternate,
        ProfilePatch {
            agent: Some("parking".to_owned()),
            model: Some("fallback".to_owned()),
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

    let error = session.prompt("work".to_owned()).await.unwrap_err();
    assert_eq!(error.code, "agent_quota_exhausted");

    // The lane switch's open binds `fallback` on the wire and the
    // alternate refuses it: the installed primary keeps its poisoned
    // stored default under the mark.
    let error = session.prompt("two".to_owned()).await.unwrap_err();
    assert_eq!(error.code, "agent_quota_exhausted");

    // The repair iteration parks inside the replacement open: the
    // second logged primary open is the handshake the close interrupts.
    let driver = {
        let session = session.clone();
        tokio::spawn(async move { session.prompt("again".to_owned()).await })
    };
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    loop {
        let contents = std::fs::read_to_string(&log).unwrap_or_default();
        if contents
            .lines()
            .filter(|line| line.starts_with("primary/open"))
            .count()
            >= 2
        {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the repair open never parked: {contents}"
        );
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    session.close().await.unwrap();

    let error = driver.await.unwrap().unwrap_err();
    assert_eq!(error.code, "agent_session_closed");
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "primary/open|unset",
            "primary/prompt|fallback",
            "alt/open|fallback",
            "primary/open|unset",
        ],
        "a close during the repair must not surface the synthesized refusal: {calls_log}"
    );
}

/// An rpc peer that logs each open's provider and each prompt's model,
/// refuses a `vnd` open, serves exactly one prompt then closes its
/// stdin, and exits on the third open across restarts so a repair walk
/// continues to the alternate.
#[cfg(unix)]
const RPC_PROVIDER_OPEN_DIES_ON_THIRD_SERVE: &str = r#"
import json, os, sys, time

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

log = open(os.environ["AWP_CALL_LOG"], "a")
state = os.environ["AWP_OPEN_STATE"]
count = 0
if os.path.exists(state):
    try:
        count = int(open(state).read().strip() or "0")
    except Exception:
        count = 0
for line in sys.stdin:
    try:
        request = json.loads(line)
    except Exception:
        continue
    method = request.get("method")
    request_id = request.get("id")
    params = request.get("params") or {}
    if method == "session/open":
        count += 1
        with open(state, "w") as handle:
            handle.write(str(count))
        provider = params.get("provider")
        log.write("primary/open|{}\n".format(provider or "unset"))
        log.flush()
        if count == 3:
            sys.exit(0)
        if provider == "vnd":
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
            continue
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        model = params.get("model")
        log.write("primary/prompt|{}\n".format(model or "unset"))
        log.flush()
        # stdin closes before the answer flushes, so the next write is
        # guaranteed to land on a dead pipe
        os.close(0)
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "served|{}|{}".format(
                  model or "peerdefault", params.get("text", ""))}})
        time.sleep(30)
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_refused_active_never_reenters_the_repair() {
    // `fallback` opened the primary parked, pending — and the first
    // prompt carries it on the wire, where it is served: the candidate
    // publishes as the chain's last accepted entry. The alternate's
    // open binds it on the wire and the peer refuses it — the entry is
    // spent everywhere, `active` included. The walk must not select it
    // again: the replacement open and the prompt that follows carry no
    // model at all.
    let dir = TestDirectory::new("rpc-active-refused");
    let script = dir.path().join("rpc_provider.py");
    std::fs::write(&script, RPC_PROVIDER_OPEN_DIES_ON_THIRD_SERVE).unwrap();
    let log = dir.path().join("calls.log");
    let state = dir.path().join("opens.count");
    let mut primary = rpc_spec(
        "serving",
        &script,
        json!({"provider": "{modelProvider}"}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    primary.command.env = EnvPolicy::Sanitized {
        inherit: vec![],
        set: [
            (
                "AWP_CALL_LOG".to_owned(),
                EnvValue::Literal(log.display().to_string()),
            ),
            (
                "AWP_OPEN_STATE".to_owned(),
                EnvValue::Literal(state.display().to_string()),
            ),
        ]
        .into_iter()
        .collect(),
    };
    let refusing = dir.path().join("rpc_refuse.py");
    std::fs::write(&refusing, RPC_ALT_OPEN_REFUSES_FALLBACK).unwrap();
    let mut alternate = rpc_spec(
        "refusing",
        &refusing,
        json!({"model": "{model}"}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut alternate, &log);
    let registry = registry_with_lanes(
        primary,
        alternate,
        ProfilePatch {
            agent: Some("serving".to_owned()),
            model: Some("vnd/prime".to_owned()),
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

    // The prompt carries `fallback` and the peer serves it — the
    // candidate confirms and publishes as the chain's active entry.
    let outcome = session.prompt("work".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "served|fallback|work");
    assert_eq!(session.active_model().as_deref(), Some("fallback"));

    // The dead lane's replacement open binds `fallback` on the wire and
    // the peer refuses it: the refused entry leaves `active`, and the
    // parked candidate never entered the installed stored slot.
    let error = session.prompt("two".to_owned()).await.unwrap_err();
    assert_eq!(error.code, "agent_quota_exhausted");

    // The walk moves straight past the dead primary to the alternate —
    // which must open unset instead of rebinding the refused model,
    // then the prompt carries no model at all.
    let outcome = session.prompt("three".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "served|unset|three");
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "primary/open|vnd",
            "primary/open|unset",
            "primary/prompt|fallback",
            "alt/open|fallback",
            "alt/open|unset",
            "alt/prompt|unset",
        ],
        "the open-refused active entry must never re-enter the walk: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer that logs each open's provider and each prompt's
/// provider, accepts every open, and refuses every prompt with a quota
/// error — after the second refusal it closes its stdin and sleeps, so
/// the next write lands on a dead pipe.
#[cfg(unix)]
const RPC_PRIMARY_OPEN_PROVIDER_REFUSES_PROMPTS: &str = r#"
import json, os, sys, time

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

def refused(request_id):
    send({"jsonrpc": "2.0", "id": request_id,
          "error": {"code": -32000, "message":
                    "resource_exhausted: quota exceeded for this model"}})

log = open(os.environ["AWP_CALL_LOG"], "a")
refusals = 0
for line in sys.stdin:
    try:
        request = json.loads(line)
    except Exception:
        continue
    method = request.get("method")
    request_id = request.get("id")
    params = request.get("params") or {}
    if method == "session/open":
        log.write("primary/open|{}\n".format(params.get("provider") or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        refusals += 1
        log.write("primary/prompt|{}\n".format(params.get("provider") or "unset"))
        log.flush()
        # stdin is closed before the second refusal is flushed, so the
        # next write is guaranteed to land on a dead pipe
        if refusals == 2:
            os.close(0)
        refused(request_id)
        if refusals == 2:
            time.sleep(30)
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

/// An rpc peer that logs each open's model and each prompt's model and
/// serves everything.
#[cfg(unix)]
const RPC_ALT_OPEN_MODEL_SERVES: &str = r#"
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
        log.write("alt/open|{}\n".format(params.get("model") or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        model = params.get("model")
        log.write("alt/prompt|{}\n".format(model or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "served|{}|{}".format(
                  model or "unset", params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_unsent_retry_candidate_survives_the_turn() {
    // `vnd/prime` binds at open through its provider, so `fallback`
    // waits in `remaining`. The explicit prompt carries `vnd` on the
    // wire and the peer refuses it: the per-turn retry schedules
    // `fallback` — but the lane's only slot on either surface is an
    // unqualified `{modelProvider}` and `fallback` has no provider
    // part, so the retry sends no selector at all. The refusal is this
    // lane's answer, not the chain's: the unsent candidate was never
    // refused, so the walk must keep it and reach the alternate whose
    // `{model}` slots carry it — open and prompt alike — inside the
    // same turn.
    let dir = TestDirectory::new("rpc-unsent-candidate");
    let script = dir.path().join("rpc_refuse.py");
    std::fs::write(&script, RPC_PRIMARY_OPEN_PROVIDER_REFUSES_PROMPTS).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "refusing",
        &script,
        json!({"provider": "{modelProvider}"}),
        json!({"provider": "{modelProvider}", "text": "{prompt}"}),
    );
    attach_call_log(&mut primary, &log);
    let serving = dir.path().join("rpc_serve.py");
    std::fs::write(&serving, RPC_ALT_OPEN_MODEL_SERVES).unwrap();
    let mut alternate = rpc_spec(
        "serving",
        &serving,
        json!({"model": "{model}"}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut alternate, &log);
    let registry = registry_with_lanes(
        primary,
        alternate,
        ProfilePatch {
            agent: Some("refusing".to_owned()),
            model: Some("vnd/prime".to_owned()),
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

    // The explicit model's refusal schedules `fallback`; the retry's
    // unqualified slot drops it from the frame, and `remaining` is
    // empty — the candidate exists only in the awaiting claim, so the
    // walk must carry it to the alternate instead of ending the turn.
    let outcome = session
        .prompt_with(TurnRequest {
            text: "work".to_owned(),
            model: Some("vnd/prime".to_owned()),
            effort: None,
        })
        .await
        .unwrap();
    assert_eq!(outcome.text.trim_end(), "served|fallback|work");
    assert_eq!(session.active_model().as_deref(), Some("fallback"));
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "primary/open|vnd",
            "primary/prompt|vnd",
            "primary/prompt|unset",
            "alt/open|fallback",
            "alt/prompt|fallback",
        ],
        "an unsent scheduled candidate must survive the walk: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer that accepts every `session/open`, logging the provider
/// it carried, and refuses every `session/prompt` with a quota error —
/// a lane whose turn surface can never serve any model.
#[cfg(unix)]
const RPC_PROVIDER_PROMPT_REFUSES_ALL: &str = r#"
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
        log.write("session/open|{}\n".format(params.get("provider") or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        log.write("session/prompt|{}\n".format(params.get("provider") or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "error": {"code": -32000, "message":
                        "resource_exhausted: quota exceeded for this model"}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_unsent_fallbacks_wait_for_a_lane_that_can_carry() {
    // A prompt whose only model slot is `{modelProvider}` emits the same
    // model-free frame for every unqualified candidate — the refusal
    // binds the model claim but proves nothing about which entry was
    // attempted. A refused-but-unsent candidate must wait out the retry
    // queue instead of re-entering `remaining`, where the retry loop
    // would immediately re-select the identical unsent entry on this
    // same lane — forever. With it held aside, the turn terminates once
    // every chain entry has been refused, and the held entries return
    // to the chain for the next turn rather than disappearing.
    let dir = TestDirectory::new("rpc-unsent-hold");
    let script = dir.path().join("rpc_provider.py");
    std::fs::write(&script, RPC_PROVIDER_PROMPT_REFUSES_ALL).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "holding",
        &script,
        json!({"provider": "{modelProvider}"}),
        json!({"text": "{prompt}", "provider": "{modelProvider}"}),
    );
    attach_call_log(&mut primary, &log);
    let registry = registry_with(
        primary,
        ProfilePatch {
            agent: Some("holding".to_owned()),
            model: Some("first".to_owned()),
            models: Some(vec!["second".to_owned(), "third".to_owned()]),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    // Three unqualified candidates — the parked head plus the two
    // scheduled retries — all produce the identical model-free request.
    // The turn must end after one refusal each instead of re-selecting
    // an unsent entry forever.
    let error = session.prompt("work".to_owned()).await.unwrap_err();
    assert_eq!(error.code, "agent_quota_exhausted");

    // The held candidates returned to the chain at the turn's end: the
    // next prompt retries the whole chain — bounded to one attempt per
    // entry again — rather than losing them.
    let error = session.prompt("again".to_owned()).await.unwrap_err();
    assert_eq!(error.code, "agent_quota_exhausted");
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "session/open|unset",
            "session/prompt|unset",
            "session/prompt|unset",
            "session/prompt|unset",
            "session/prompt|unset",
            "session/prompt|unset",
            "session/prompt|unset",
        ],
        "unsent candidates must not re-enter the same lane's retry: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer that logs each open's provider and each prompt's model,
/// refuses a `vnd` open, and serves every prompt.
#[cfg(unix)]
const RPC_PROVIDER_OPEN_SERVE_ALL: &str = r#"
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
        provider = params.get("provider")
        log.write("session/open|{}\n".format(provider or "unset"))
        log.flush()
        if provider == "vnd":
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
            continue
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        model = params.get("model")
        log.write("session/prompt|{}\n".format(model or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "served|{}|{}".format(
                  model or "peerdefault", params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_direct_prompt_never_inherits_the_parked_open_candidate() {
    // `fallback` opened the healthy session with its `provider` member
    // omitted — the candidate is parked pending, never stored, so the
    // adapter's session model stays empty. A direct prompt naming no
    // model inherits nothing through the `{model}` substitution and
    // cannot promote the unconfirmed claim: only an implicit prompt
    // supplies it.
    let dir = TestDirectory::new("rpc-parked-direct");
    let script = dir.path().join("rpc_provider.py");
    std::fs::write(&script, RPC_PROVIDER_OPEN_SERVE_ALL).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "parked",
        &script,
        json!({"provider": "{modelProvider}"}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut primary, &log);
    let registry = registry_with(
        primary,
        ProfilePatch {
            agent: Some("parked".to_owned()),
            model: Some("vnd/prime".to_owned()),
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

    // The direct prompt inherits nothing: the stored slot held no
    // unconfirmed candidate, so no `{model}` member reaches the wire
    // and the peer answers with its own default.
    let outcome = session
        .prompt_direct(TurnRequest {
            text: "raw".to_owned(),
            model: None,
            effort: None,
        })
        .await
        .unwrap();
    assert_eq!(outcome.text.trim_end(), "served|peerdefault|raw");
    assert_eq!(session.active_model().as_deref(), None);

    // The claim a direct turn never consumed still rides the next
    // implicit prompt — the wire finally carries `fallback`.
    let outcome = session.prompt("again".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "served|fallback|again");
    assert_eq!(session.active_model().as_deref(), Some("fallback"));
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "session/open|vnd",
            "session/open|unset",
            "session/prompt|unset",
            "session/prompt|fallback",
        ],
        "a direct prompt must never inherit the parked candidate: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer that accepts a model-free `session/open` — the stored
/// default its request carries — logs each open and prompt, refuses a
/// `vnd/prime` prompt, serves a `fallback` prompt once, then closes
/// its stdin so the lane dies for writes.
#[cfg(unix)]
const RPC_STORED_SERVES_FALLBACK_THEN_DIES: &str = r#"
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
    params = request.get("params") or {}
    if method == "session/open":
        log.write("primary/open|{}\n".format(params.get("provider") or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        model = params.get("model")
        log.write("primary/prompt|{}\n".format(model or "unset"))
        log.flush()
        if model == "fallback":
            # stdin closes before the answer flushes, so the next write
            # is guaranteed to land on a dead pipe
            os.close(0)
            send({"jsonrpc": "2.0", "id": request_id,
                  "result": {"text": "served|{}|{}".format(
                      model, params.get("text", ""))}})
            time.sleep(30)
        else:
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_open_refused_turn_default_never_resends() {
    // A per-turn retry wins `fallback` on a session that still stores
    // `vnd/prime` from its open: the confirmed candidate publishes as
    // the active entry and the per-turn default while the stored slot
    // keeps its own model. A later lane's open binds `fallback` on the
    // wire and refuses it — `active` and the turn default are both
    // spent, or the still-installed lane's next implicit prompt would
    // resend the refused model.
    let dir = TestDirectory::new("rpc-retire-default");
    let script = dir.path().join("rpc_provider.py");
    std::fs::write(&script, RPC_STORED_SERVES_FALLBACK_THEN_DIES).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "retiring",
        &script,
        json!({"note": "stored"}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut primary, &log);
    let refusing = dir.path().join("rpc_refuse.py");
    std::fs::write(&refusing, RPC_ALT_OPEN_REFUSES_FALLBACK).unwrap();
    let mut alternate = rpc_spec(
        "refusing",
        &refusing,
        json!({"model": "{model}"}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut alternate, &log);
    let registry = registry_with_lanes(
        primary,
        alternate,
        ProfilePatch {
            agent: Some("retiring".to_owned()),
            model: Some("vnd/prime".to_owned()),
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

    // The stored `vnd/prime` is refused on the prompt; the per-turn
    // retry carries `fallback` and the lane serves it — the candidate
    // confirms as the turn default and the chain's active entry.
    let outcome = session.prompt("work".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "served|fallback|work");
    assert_eq!(session.active_model().as_deref(), Some("fallback"));

    // The dead lane's walk opens the alternate with `fallback` on the
    // wire and the peer refuses it: the entry is spent from `active`
    // and the per-turn default alike.
    let error = session.prompt("two".to_owned()).await.unwrap_err();
    assert_eq!(error.code, "agent_quota_exhausted");
    assert_eq!(session.active_model().as_deref(), None);

    // The lane walk must not resend the refused default: the alternate
    // opens model-free and the replay carries no model at all.
    let outcome = session.prompt("three".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "served|unset|three");
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "primary/open|unset",
            "primary/prompt|vnd/prime",
            "primary/prompt|fallback",
            "alt/open|fallback",
            "alt/open|unset",
            "alt/prompt|unset",
        ],
        "the open-refused turn default must never resend: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer that refuses a `vnd` provider at `session/open` and exits
/// on any other open — the transport dies mid-open before it can answer.
#[cfg(unix)]
const RPC_PROVIDER_OPEN_REFUSES_VND_THEN_EXITS: &str = r#"
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
        provider = params.get("provider")
        log.write("primary/open|{}\n".format(provider or "unset"))
        log.flush()
        if provider == "vnd":
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
        else:
            sys.exit(0)
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_parked_candidate_carries_past_a_failed_open() {
    // `vnd/prime` is refused at the primary's open and the chain
    // advances to unqualified `fallback`, which the `{modelProvider}`
    // slot cannot resolve — it waits parked out of the stored slot.
    // The peer then dies mid-open, and the failure handler must read
    // the parked candidate — not the cleared slot — as the walk's
    // cursor: resuming at the refused head resends it on the alternate.
    let dir = TestDirectory::new("rpc-parked-carry");
    let dying = dir.path().join("rpc_die.py");
    std::fs::write(&dying, RPC_PROVIDER_OPEN_REFUSES_VND_THEN_EXITS).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "parking",
        &dying,
        json!({"provider": "{modelProvider}"}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut primary, &log);
    let serving = dir.path().join("rpc_serve.py");
    std::fs::write(&serving, RPC_LOGGED_MODEL_SERVE).unwrap();
    let mut alternate = rpc_spec(
        "serving",
        &serving,
        json!({"model": "{model}"}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut alternate, &log);
    let registry = registry_with_lanes(
        primary,
        alternate,
        ProfilePatch {
            agent: Some("parking".to_owned()),
            model: Some("vnd/prime".to_owned()),
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
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "primary/open|vnd",
            "primary/open|unset",
            "alt/open|fallback",
            "alt/prompt|fallback",
        ],
        "the parked candidate — not the refused head — must carry: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer that refuses a `vnd` provider at `session/open` and also
/// refuses an open whose provider member was dropped — the model-free
/// parked open whose refusal is the lane's answer, not the model's.
#[cfg(unix)]
const RPC_PROVIDER_OPEN_REFUSES_VND_AND_UNSET: &str = r#"
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
        provider = params.get("provider")
        log.write("primary/open|{}\n".format(provider or "unset"))
        log.flush()
        if provider == "vnd" or provider is None:
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
        else:
            send({"jsonrpc": "2.0", "id": request_id,
                  "result": {"sessionId": "awp-rpc-session"}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_refused_parked_open_carries_its_candidate() {
    // Same parked walk as the transport-death case, but the peer
    // refuses the model-free parked open outright: a refusal the
    // refused call could not have answered about the model is the
    // lane's answer, and the walk still carries the parked candidate —
    // never the refused head — onto the alternate.
    let dir = TestDirectory::new("rpc-parked-refused");
    let refusing = dir.path().join("rpc_refuse.py");
    std::fs::write(&refusing, RPC_PROVIDER_OPEN_REFUSES_VND_AND_UNSET).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "parking",
        &refusing,
        json!({"provider": "{modelProvider}"}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut primary, &log);
    let serving = dir.path().join("rpc_serve.py");
    std::fs::write(&serving, RPC_LOGGED_MODEL_SERVE).unwrap();
    let mut alternate = rpc_spec(
        "serving",
        &serving,
        json!({"model": "{model}"}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut alternate, &log);
    let registry = registry_with_lanes(
        primary,
        alternate,
        ProfilePatch {
            agent: Some("parking".to_owned()),
            model: Some("vnd/prime".to_owned()),
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
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "primary/open|vnd",
            "primary/open|unset",
            "alt/open|fallback",
            "alt/prompt|fallback",
        ],
        "a refused parked open still carries its candidate: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer that accepts every `session/open`, logging the provider
/// it carried, refuses the first `session/prompt` with a quota error —
/// closing its stdin before the refusal flushes so the next write lands
/// on a dead pipe — and sleeps.
#[cfg(unix)]
const RPC_PROVIDER_OPEN_REFUSES_FIRST_PROMPT_THEN_DIES: &str = r#"
import json, os, sys, time

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

def refused(request_id):
    send({"jsonrpc": "2.0", "id": request_id,
          "error": {"code": -32000, "message":
                    "resource_exhausted: quota exceeded for this model"}})

log = open(os.environ["AWP_CALL_LOG"], "a")
refusals = 0
for line in sys.stdin:
    try:
        request = json.loads(line)
    except Exception:
        continue
    method = request.get("method")
    request_id = request.get("id")
    params = request.get("params") or {}
    if method == "session/open":
        log.write("primary/open|{}\n".format(params.get("provider") or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        refusals += 1
        log.write("primary/prompt|{}\n".format(params.get("provider") or "unset"))
        log.flush()
        # stdin is closed before the refusal is flushed, so the next
        # prompt write is guaranteed to land on a dead pipe
        if refusals == 1:
            os.close(0)
        refused(request_id)
        if refusals == 1:
            time.sleep(30)
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

/// An rpc peer that logs each open's provider and each prompt's model,
/// refuses a `vnd` provider at `session/open`, and serves every other
/// call — the alternate whose provider-only setup must not resend the
/// provider the chain just exhausted.
#[cfg(unix)]
const RPC_PROVIDER_OPEN_REFUSES_VND_MODEL_PROMPT_SERVES: &str = r#"
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
        provider = params.get("provider")
        log.write("alt/open|{}\n".format(provider or "unset"))
        log.flush()
        if provider == "vnd":
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
        else:
            send({"jsonrpc": "2.0", "id": request_id,
                  "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        model = params.get("model")
        log.write("alt/prompt|{}\n".format(model or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "served|{}|{}".format(
                  model or "peerdefault", params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_unresolvable_candidate_never_resends_the_exhausted_provider() {
    // `vnd/prime` binds the primary's open through its provider and is
    // refused on the prompt; the chain schedules unqualified `fallback`,
    // then the lane dies before the retry can dispatch. On the
    // alternate — `{modelProvider}`-only at setup — the replacement open
    // must not fill the slot the candidate cannot resolve with the
    // exhausted entry: its provider would reach the wire, be refused as
    // a model answer, and end the turn before `fallback` ever reached
    // the prompt.
    let dir = TestDirectory::new("rpc-parked-provider");
    let dying = dir.path().join("rpc_die.py");
    std::fs::write(&dying, RPC_PROVIDER_OPEN_REFUSES_FIRST_PROMPT_THEN_DIES).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "exhausting",
        &dying,
        json!({"provider": "{modelProvider}"}),
        json!({"provider": "{modelProvider}", "text": "{prompt}"}),
    );
    attach_call_log(&mut primary, &log);
    let refusing = dir.path().join("rpc_refuse.py");
    std::fs::write(&refusing, RPC_PROVIDER_OPEN_REFUSES_VND_MODEL_PROMPT_SERVES).unwrap();
    let mut alternate = rpc_spec(
        "serving",
        &refusing,
        json!({"provider": "{modelProvider}"}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut alternate, &log);
    let registry = registry_with_lanes(
        primary,
        alternate,
        ProfilePatch {
            agent: Some("exhausting".to_owned()),
            model: Some("vnd/prime".to_owned()),
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

    // `fallback` waits for the prompt surface it can ride: the
    // alternate's setup sends no model at all, and the replayed turn
    // carries the candidate to a peer that serves it.
    let outcome = session.prompt("work".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "served|fallback|work");
    assert_eq!(session.active_model().as_deref(), Some("fallback"));
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "primary/open|vnd",
            "primary/prompt|vnd",
            "alt/open|unset",
            "alt/prompt|fallback",
        ],
        "the exhausted provider must never fill the setup slot: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer that accepts every `session/open`, logging the provider
/// it carried, refuses three `session/prompt` calls with a quota
/// error — closing its stdin before the third refusal flushes so the
/// next write lands on a dead pipe — and sleeps.
#[cfg(unix)]
const RPC_PROVIDER_OPEN_REFUSES_THREE_PROMPTS_THEN_DIES: &str = r#"
import json, os, sys, time

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

def refused(request_id):
    send({"jsonrpc": "2.0", "id": request_id,
          "error": {"code": -32000, "message":
                    "resource_exhausted: quota exceeded for this model"}})

log = open(os.environ["AWP_CALL_LOG"], "a")
refusals = 0
for line in sys.stdin:
    try:
        request = json.loads(line)
    except Exception:
        continue
    method = request.get("method")
    request_id = request.get("id")
    params = request.get("params") or {}
    if method == "session/open":
        log.write("primary/open|{}\n".format(params.get("provider") or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        refusals += 1
        log.write("primary/prompt|{}\n".format(params.get("provider") or "unset"))
        log.flush()
        # stdin is closed before the third refusal is flushed, so the
        # next prompt write is guaranteed to land on a dead pipe
        if refusals == 3:
            os.close(0)
        refused(request_id)
        if refusals == 3:
            time.sleep(30)
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

/// An rpc peer that logs each open's model and each prompt's model,
/// refuses a `three` model at `session/open`, and serves everything
/// else — the alternate that can bind the candidates the dead lane
/// held unsent.
#[cfg(unix)]
const RPC_ALT_OPEN_REFUSES_THREE_MODEL_PROMPT_SERVES: &str = r#"
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
        log.write("alt/open|{}\n".format(model or "unset"))
        log.flush()
        if model == "three":
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
        else:
            send({"jsonrpc": "2.0", "id": request_id,
                  "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        model = params.get("model")
        log.write("alt/prompt|{}\n".format(model or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "served|{}|{}".format(
                  model or "peerdefault", params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_held_candidates_enter_the_alternate_setup_walk() {
    // `vnd/prime` binds the primary's open through its provider; prompt
    // refusals spend it and park unqualified `one` on the shared pending
    // claim, then `two` in the held list — unsent, since neither surface
    // resolves an unqualified entry. The lane dies mid-retry carrying
    // `three`, and the alternate's setup refuses it. The chain walk on
    // that lane must reach the held `two`: a candidate the failed
    // lane could never send is exactly one the alternate can bind —
    // holding it past the lane change reports exhaustion the chain
    // disproves.
    let dir = TestDirectory::new("rpc-held-walk");
    let dying = dir.path().join("rpc_die.py");
    std::fs::write(&dying, RPC_PROVIDER_OPEN_REFUSES_THREE_PROMPTS_THEN_DIES).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "holding",
        &dying,
        json!({"provider": "{modelProvider}"}),
        json!({"provider": "{modelProvider}", "text": "{prompt}"}),
    );
    attach_call_log(&mut primary, &log);
    let refusing = dir.path().join("rpc_refuse.py");
    std::fs::write(&refusing, RPC_ALT_OPEN_REFUSES_THREE_MODEL_PROMPT_SERVES).unwrap();
    let mut alternate = rpc_spec(
        "serving",
        &refusing,
        json!({"model": "{model}"}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut alternate, &log);
    let registry = registry_with_lanes(
        primary,
        alternate,
        ProfilePatch {
            agent: Some("holding".to_owned()),
            model: Some("vnd/prime".to_owned()),
            models: Some(vec!["one".to_owned(), "two".to_owned(), "three".to_owned()]),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    // The held `two` re-enters the chain when the search reaches the
    // alternate: `three`'s refused open walks the chain on that lane,
    // binds `two`, and the replayed prompt carries it to a peer that
    // serves it.
    let outcome = session.prompt("work".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "served|two|work");
    assert_eq!(session.active_model().as_deref(), Some("two"));
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "primary/open|vnd",
            "primary/prompt|vnd",
            "primary/prompt|unset",
            "primary/prompt|unset",
            "alt/open|three",
            "alt/open|two",
            "alt/prompt|two",
        ],
        "a held candidate must be eligible on the new lane: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer that logs each open's model and each prompt's model,
/// refuses `three` and `two` at `session/open`, and serves everything
/// else — the alternate whose setup walk must reach the shared pending
/// claim after the queued fallbacks are spent.
#[cfg(unix)]
const RPC_ALT_OPEN_REFUSES_THREE_TWO_MODEL_PROMPT_SERVES: &str = r#"
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
        log.write("alt/open|{}\n".format(model or "unset"))
        log.flush()
        if model == "three" or model == "two":
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
        else:
            send({"jsonrpc": "2.0", "id": request_id,
                  "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        model = params.get("model")
        log.write("alt/prompt|{}\n".format(model or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "served|{}|{}".format(
                  model or "peerdefault", params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_pending_candidate_enters_the_setup_walk_before_exhaustion() {
    // `vnd/prime` binds the primary's open through its provider; prompt
    // refusals spend it, park unqualified `one` on the shared pending
    // claim, and hold `two` unsent. The lane dies mid-retry carrying
    // `three`, and the alternate's setup refuses both `three` and `two`.
    // The walk must offer the parked `one` — a claim awaiting its carry
    // evidence was never refused — instead of declaring exhaustion with
    // an eligible candidate still standing.
    let dir = TestDirectory::new("rpc-pending-walk");
    let dying = dir.path().join("rpc_die.py");
    std::fs::write(&dying, RPC_PROVIDER_OPEN_REFUSES_THREE_PROMPTS_THEN_DIES).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "holding",
        &dying,
        json!({"provider": "{modelProvider}"}),
        json!({"provider": "{modelProvider}", "text": "{prompt}"}),
    );
    attach_call_log(&mut primary, &log);
    let refusing = dir.path().join("rpc_refuse.py");
    std::fs::write(
        &refusing,
        RPC_ALT_OPEN_REFUSES_THREE_TWO_MODEL_PROMPT_SERVES,
    )
    .unwrap();
    let mut alternate = rpc_spec(
        "serving",
        &refusing,
        json!({"model": "{model}"}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut alternate, &log);
    let registry = registry_with_lanes(
        primary,
        alternate,
        ProfilePatch {
            agent: Some("holding".to_owned()),
            model: Some("vnd/prime".to_owned()),
            models: Some(vec!["one".to_owned(), "two".to_owned(), "three".to_owned()]),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    // The shared pending claim joins the setup walk after `remaining`
    // empties: the alternate binds `one` at open and the replayed prompt
    // carries it to a peer that serves it.
    let outcome = session.prompt("work".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "served|one|work");
    assert_eq!(session.active_model().as_deref(), Some("one"));
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "primary/open|vnd",
            "primary/prompt|vnd",
            "primary/prompt|unset",
            "primary/prompt|unset",
            "alt/open|three",
            "alt/open|two",
            "alt/open|one",
            "alt/prompt|one",
        ],
        "the shared pending claim must be offered before exhaustion: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer that accepts every `session/open` — its only setup slot is
/// `{modelProvider}`, so unqualified candidates reach the wire unset —
/// and refuses every `session/prompt` with a quota error, exiting once a
/// third prompt is logged so a requeueing driver cannot loop forever.
#[cfg(unix)]
const RPC_PROVIDER_OPEN_ACCEPTS_PROMPT_REFUSES_THREE: &str = r#"
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
        log.write("alt/open|{}\n".format(params.get("provider") or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        log.write("alt/prompt|{}\n".format(params.get("provider") or "unset"))
        log.flush()
        log.close()
        seen = sum(1 for logged in open(os.environ["AWP_CALL_LOG"])
                   if logged.startswith("alt/prompt|"))
        if seen >= 3:
            sys.exit(0)
        send({"jsonrpc": "2.0", "id": request_id,
              "error": {"code": -32000, "message":
                        "resource_exhausted: quota exceeded for this model"}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_undeliverable_candidates_leave_the_setup_only_retry_queue() {
    // Same parked chain as the setup-walk cases, but the alternate's
    // only model surface is `{modelProvider}` at setup: no slot on it
    // can carry the unqualified fallbacks. Its model-free open succeeds
    // and the default prompt returns quota exhaustion — candidates the
    // lane cannot deliver must wait out its retry queue instead of
    // reopening the same lane with the identical request until the
    // chain drains.
    let dir = TestDirectory::new("rpc-undeliverable-hold");
    let dying = dir.path().join("rpc_die.py");
    std::fs::write(&dying, RPC_PROVIDER_OPEN_REFUSES_THREE_PROMPTS_THEN_DIES).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "holding",
        &dying,
        json!({"provider": "{modelProvider}"}),
        json!({"provider": "{modelProvider}", "text": "{prompt}"}),
    );
    attach_call_log(&mut primary, &log);
    let refusing = dir.path().join("rpc_refuse.py");
    std::fs::write(&refusing, RPC_PROVIDER_OPEN_ACCEPTS_PROMPT_REFUSES_THREE).unwrap();
    let mut alternate = rpc_spec(
        "provideronly",
        &refusing,
        json!({"provider": "{modelProvider}"}),
        json!({"text": "{prompt}"}),
    );
    attach_call_log(&mut alternate, &log);
    let registry = registry_with_lanes(
        primary,
        alternate,
        ProfilePatch {
            agent: Some("holding".to_owned()),
            model: Some("vnd/prime".to_owned()),
            models: Some(vec!["one".to_owned(), "two".to_owned(), "three".to_owned()]),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    // One model-free open and one refused prompt prove the lane: the
    // undeliverable fallbacks leave its retry queue instead of reopening
    // it with identical requests, and the refusal answers the turn.
    let error = session.prompt("work".to_owned()).await.unwrap_err();
    assert_eq!(error.code, "agent_quota_exhausted");
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "primary/open|vnd",
            "primary/prompt|vnd",
            "primary/prompt|unset",
            "primary/prompt|unset",
            "alt/open|unset",
            "alt/prompt|unset",
        ],
        "an undeliverable candidate must not re-enter the retry queue: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer that accepts every `session/open`, logging the model it
/// carried, refuses one `session/prompt` with a quota error — closing
/// its stdin before the refusal flushes so the next write lands on a
/// dead pipe — and sleeps.
#[cfg(unix)]
const RPC_OPEN_MODEL_REFUSES_ONE_PROMPT_THEN_DIES: &str = r#"
import json, os, sys, time

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

def refused(request_id):
    send({"jsonrpc": "2.0", "id": request_id,
          "error": {"code": -32000, "message":
                    "resource_exhausted: quota exceeded for this model"}})

log = open(os.environ["AWP_CALL_LOG"], "a")
refusals = 0
for line in sys.stdin:
    try:
        request = json.loads(line)
    except Exception:
        continue
    method = request.get("method")
    request_id = request.get("id")
    params = request.get("params") or {}
    if method == "session/open":
        log.write("primary/open|{}\n".format(params.get("model") or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        refusals += 1
        log.write("primary/prompt|{}\n".format(params.get("provider") or "unset"))
        log.flush()
        # stdin is closed before the refusal is flushed, so the next
        # prompt write is guaranteed to land on a dead pipe
        if refusals == 1:
            os.close(0)
        refused(request_id)
        if refusals == 1:
            time.sleep(30)
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

/// An rpc peer whose open's only model slot is `{modelProvider}`: it
/// logs the provider each open resolved, refuses `first`, accepts
/// everything else, and serves every prompt — the setup-only lane whose
/// same-lane retry must not reopen with a model-free request.
#[cfg(unix)]
const RPC_PROVIDER_OPEN_REFUSES_FIRST_PROMPT_SERVES: &str = r#"
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
        provider = params.get("provider")
        log.write("alt/open|{}\n".format(provider or "unset"))
        log.flush()
        if provider == "first":
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
        else:
            send({"jsonrpc": "2.0", "id": request_id,
                  "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        model = params.get("model")
        log.write("alt/prompt|{}\n".format(model or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "served|{}|{}".format(
                  model or "peerdefault", params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_setup_walk_skips_candidates_the_lane_cannot_carry() {
    // `vnd/prime` binds the primary's open; one prompt refusal spends it
    // and the lane dies while `first/one` heads the chain. The
    // alternate's only model surface is `{modelProvider}` at setup — no
    // slot on it can carry the unqualified `plain`. Its open refuses
    // `first`: the same-lane walk must hold `plain` — a retry would
    // emit a model-free request the peer accepts on its own default —
    // and reach `good/two`, the deliverable candidate behind it.
    let dir = TestDirectory::new("rpc-setup-walk-hold");
    let dying = dir.path().join("rpc_die.py");
    std::fs::write(&dying, RPC_OPEN_MODEL_REFUSES_ONE_PROMPT_THEN_DIES).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "holding",
        &dying,
        json!({"model": "{model}"}),
        json!({"provider": "{modelProvider}", "text": "{prompt}"}),
    );
    attach_call_log(&mut primary, &log);
    let refusing = dir.path().join("rpc_refuse.py");
    std::fs::write(&refusing, RPC_PROVIDER_OPEN_REFUSES_FIRST_PROMPT_SERVES).unwrap();
    let mut alternate = rpc_spec(
        "provideronly",
        &refusing,
        json!({"provider": "{modelProvider}"}),
        json!({"text": "{prompt}"}),
    );
    attach_call_log(&mut alternate, &log);
    let registry = registry_with_lanes(
        primary,
        alternate,
        ProfilePatch {
            agent: Some("holding".to_owned()),
            model: Some("vnd/prime".to_owned()),
            models: Some(vec![
                "first/one".to_owned(),
                "plain".to_owned(),
                "good/two".to_owned(),
            ]),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    // `first/one` is refused at the alternate's open; `plain` waits out
    // the same-lane walk — no setup call can carry it — and `good/two`
    // binds the next reopen.
    let outcome = session.prompt("work".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "served|peerdefault|work");
    assert_eq!(session.active_model().as_deref(), Some("good/two"));
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "primary/open|vnd/prime",
            "primary/prompt|vnd",
            "alt/open|first",
            "alt/open|good",
            "alt/prompt|unset",
        ],
        "an undeliverable candidate must not reopen the same lane: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer that accepts every `session/open` — logging the provider
/// it carried — refuses two `session/prompt` calls with a quota error,
/// closing its stdin before the second refusal flushes so the next
/// write lands on a dead pipe, and sleeps.
#[cfg(unix)]
const RPC_PROVIDER_OPEN_REFUSES_TWO_PROMPTS_THEN_DIES: &str = r#"
import json, os, sys, time

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

def refused(request_id):
    send({"jsonrpc": "2.0", "id": request_id,
          "error": {"code": -32000, "message":
                    "resource_exhausted: quota exceeded for this model"}})

log = open(os.environ["AWP_CALL_LOG"], "a")
refusals = 0
for line in sys.stdin:
    try:
        request = json.loads(line)
    except Exception:
        continue
    method = request.get("method")
    request_id = request.get("id")
    params = request.get("params") or {}
    if method == "session/open":
        log.write("primary/open|{}\n".format(params.get("provider") or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        refusals += 1
        log.write("primary/prompt|{}\n".format(params.get("provider") or "unset"))
        log.flush()
        # stdin is closed before the second refusal is flushed, so the
        # next prompt write is guaranteed to land on a dead pipe
        if refusals == 2:
            os.close(0)
        refused(request_id)
        if refusals == 2:
            time.sleep(30)
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

/// An rpc peer with a `{modelProvider}` open that accepts everything
/// and answers its first `session/prompt` — stdin already closed, so
/// the next write lands on a dead pipe: the lane that parks an
/// unqualified candidate as a stored claim while it still serves, then
/// dies for the turn after.
#[cfg(unix)]
const RPC_PROVIDER_OPEN_SERVES_FIRST_THEN_DIES: &str = r#"
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
    params = request.get("params") or {}
    if method == "session/open":
        log.write("alt/open|{}\n".format(params.get("provider") or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        log.write("alt/prompt|{}\n".format(params.get("provider") or "unset"))
        log.flush()
        # stdin is closed before the answer flushes, so the next write
        # is guaranteed to land on a dead pipe
        os.close(0)
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "served|{}|{}".format(
                  params.get("provider") or "peerdefault",
                  params.get("text", ""))}})
        time.sleep(30)
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_direct_prompt_never_spends_the_pending_claim() {
    // `vnd/prime` binds the primary's open; two prompt refusals spend it
    // and park the unqualified `fallback` on the shared pending claim —
    // the `{modelProvider}` prompt emits it absent — then the refusal
    // walks to an alternate whose provider-only slots can likewise only
    // park it: the model-free serve answers the turn with the claim
    // still queued. A `prompt_direct` fills no defaults — the dead pipe
    // carries it to the third lane, which opens and serves model-free —
    // so the claim rides the next implicit prompt to a lane that can
    // finally carry it.
    let dir = TestDirectory::new("rpc-direct-pending-walk");
    let dying = dir.path().join("rpc_die.py");
    std::fs::write(&dying, RPC_PROVIDER_OPEN_REFUSES_TWO_PROMPTS_THEN_DIES).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "pending",
        &dying,
        json!({"provider": "{modelProvider}"}),
        json!({"provider": "{modelProvider}", "text": "{prompt}"}),
    );
    attach_call_log(&mut primary, &log);
    let parking = dir.path().join("rpc_park.py");
    std::fs::write(&parking, RPC_PROVIDER_OPEN_SERVES_FIRST_THEN_DIES).unwrap();
    let mut alternate = rpc_spec(
        "parking",
        &parking,
        json!({"provider": "{modelProvider}"}),
        json!({"provider": "{modelProvider}", "text": "{prompt}"}),
    );
    attach_call_log(&mut alternate, &log);
    let serving = dir.path().join("rpc_serve.py");
    std::fs::write(&serving, RPC_LOGGED_MODEL_SERVE_THIRD).unwrap();
    let mut third = rpc_spec(
        "serving",
        &serving,
        json!({"provider": "{model}"}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut third, &log);
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
            agent: Some("pending".to_owned()),
            model: Some("vnd/prime".to_owned()),
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

    // The walk past the spent `vnd/prime` can only park `fallback` on
    // the alternate's absent provider slot: the model-free serve ends
    // the turn with the claim intact.
    let outcome = session.prompt("work".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "served|peerdefault|work");

    // The direct turn's replay lands on the third lane: no default to
    // fill, so the open and the prompt both carry nothing — the claim
    // is not its to spend.
    let outcome = session
        .prompt_direct(TurnRequest {
            text: "raw".to_owned(),
            model: None,
            effort: None,
        })
        .await
        .unwrap();
    assert_eq!(outcome.text.trim_end(), "served|unset|raw");

    // The claim a direct turn never consumed rides the next implicit
    // prompt — the third lane's prompt carries it at last.
    let outcome = session.prompt("again".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "served|fallback|again");
    assert_eq!(session.active_model().as_deref(), Some("fallback"));
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "primary/open|vnd",
            "primary/prompt|vnd",
            "primary/prompt|unset",
            "alt/open|unset",
            "alt/prompt|unset",
            "third/open|unset",
            "third/prompt|unset",
            "third/prompt|fallback",
        ],
        "a direct prompt must not consume the queued claim: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer whose every model slot is `{modelProvider}` — open and
/// prompt alike: it logs the provider each call resolved, refuses
/// `first` at `session/open`, and serves every prompt on whichever
/// provider the wire carried.
#[cfg(unix)]
const RPC_PROVIDER_ONLY_OPEN_REFUSES_FIRST_PROMPT_SERVES: &str = r#"
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
        provider = params.get("provider")
        log.write("alt/open|{}\n".format(provider or "unset"))
        log.flush()
        if provider == "first":
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
        else:
            send({"jsonrpc": "2.0", "id": request_id,
                  "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        provider = params.get("provider")
        log.write("alt/prompt|{}\n".format(provider or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "served|{}|{}".format(
                  provider or "peerdefault", params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_setup_walk_holds_the_turn_undeliverable_candidate() {
    // `vnd/prime` binds the primary's open; one prompt refusal spends it
    // and the lane dies while `first/one` heads the chain. The
    // alternate's only model surface is `{modelProvider}` — at setup
    // and on prompts alike — so `plain` reaches neither call: it
    // resolves no setup slot and no turn slot. Refused `first/one`, the
    // same-lane walk must not stop at `plain` — a reopen it cannot
    // express emits a model-free request the peer accepts on its own
    // default — it holds `plain` and reaches `good/two`.
    let dir = TestDirectory::new("rpc-setup-walk-turn-hold");
    let dying = dir.path().join("rpc_die.py");
    std::fs::write(&dying, RPC_OPEN_MODEL_REFUSES_ONE_PROMPT_THEN_DIES).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "holding",
        &dying,
        json!({"model": "{model}"}),
        json!({"provider": "{modelProvider}", "text": "{prompt}"}),
    );
    attach_call_log(&mut primary, &log);
    let provider_only = dir.path().join("rpc_provider_only.py");
    std::fs::write(
        &provider_only,
        RPC_PROVIDER_ONLY_OPEN_REFUSES_FIRST_PROMPT_SERVES,
    )
    .unwrap();
    let mut alternate = rpc_spec(
        "provideronly",
        &provider_only,
        json!({"provider": "{modelProvider}"}),
        json!({"provider": "{modelProvider}", "text": "{prompt}"}),
    );
    attach_call_log(&mut alternate, &log);
    let registry = registry_with_lanes(
        primary,
        alternate,
        ProfilePatch {
            agent: Some("holding".to_owned()),
            model: Some("vnd/prime".to_owned()),
            models: Some(vec![
                "first/one".to_owned(),
                "plain".to_owned(),
                "good/two".to_owned(),
            ]),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    // `first/one` is refused at the alternate's open; `plain` — its
    // value resolving neither the provider slot at setup nor the
    // provider slot on prompts — waits out the walk, and `good/two`
    // binds the next reopen and rides the replayed prompt.
    let outcome = session.prompt("work".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "served|good|work");
    assert_eq!(session.active_model().as_deref(), Some("good/two"));
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "primary/open|vnd/prime",
            "primary/prompt|vnd",
            "alt/open|first",
            "alt/open|good",
            "alt/prompt|good",
        ],
        "a candidate no turn slot resolves must not stop the walk: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer with a `{modelProvider}`-only open and prompt: refuses
/// `first` at the open, serves a single prompt on whichever provider
/// the wire carried, then closes stdin before flushing so the next
/// write lands on a dead pipe.
#[cfg(unix)]
const RPC_PROVIDER_ONLY_OPEN_REFUSES_FIRST_SERVES_THEN_DIES: &str = r#"
import json, os, sys, time

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

log = open(os.environ["AWP_CALL_LOG"], "a")
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
        provider = params.get("provider")
        log.write("a/open|{}\n".format(provider or "unset"))
        log.flush()
        if provider == "first":
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
        else:
            send({"jsonrpc": "2.0", "id": request_id,
                  "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        prompts += 1
        provider = params.get("provider")
        log.write("a/prompt|{}\n".format(provider or "unset"))
        log.flush()
        # stdin closes before the served answer flushes, so the next
        # prompt write is guaranteed to land on a dead pipe
        os.close(0)
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "served|{}|{}".format(
                  provider or "peerdefault", params.get("text", ""))}})
        time.sleep(30)
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

/// An rpc peer whose open and prompt carry `{model}`: refuses
/// `good/two` at `session/open`, and serves every prompt on whichever
/// model the wire carried.
#[cfg(unix)]
const RPC_MODEL_OPEN_REFUSES_GOOD_PROMPT_SERVES: &str = r#"
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
        log.write("b/open|{}\n".format(model or "unset"))
        log.flush()
        if model == "good/two":
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
        else:
            send({"jsonrpc": "2.0", "id": request_id,
                  "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        model = params.get("model")
        log.write("b/prompt|{}\n".format(model or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "served|{}|{}".format(
                  model or "peerdefault", params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_initial_setup_walk_holds_the_undeliverable_fallback() {
    // The provider-only lane's open refuses `first/one` in the chain
    // `first/one` → `plain` → `good/two`. `plain` resolves neither the
    // lane's setup slot nor its prompt slot: the establishment walk must
    // not issue a model-free reopen for it — the peer would accept on
    // its own default and `good/two` would never be tried. The walk
    // holds `plain`, binds `good/two`, and keeps `plain` in the chain
    // for the alternate lane whose `{model}` slots can carry it.
    let dir = TestDirectory::new("rpc-initial-walk-hold");
    let provider_only = dir.path().join("rpc_provider_only.py");
    std::fs::write(
        &provider_only,
        RPC_PROVIDER_ONLY_OPEN_REFUSES_FIRST_SERVES_THEN_DIES,
    )
    .unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "provideronly",
        &provider_only,
        json!({"provider": "{modelProvider}"}),
        json!({"provider": "{modelProvider}", "text": "{prompt}"}),
    );
    attach_call_log(&mut primary, &log);
    let carrying = dir.path().join("rpc_carrying.py");
    std::fs::write(&carrying, RPC_MODEL_OPEN_REFUSES_GOOD_PROMPT_SERVES).unwrap();
    let mut alternate = rpc_spec(
        "carrying",
        &carrying,
        json!({"model": "{model}"}),
        json!({"model": "{model}", "text": "{prompt}"}),
    );
    attach_call_log(&mut alternate, &log);
    let registry = registry_with_lanes(
        primary,
        alternate,
        ProfilePatch {
            agent: Some("provideronly".to_owned()),
            model: Some("first/one".to_owned()),
            models: Some(vec!["plain".to_owned(), "good/two".to_owned()]),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    // `first/one` refused at the open; `plain` held — `good/two` binds
    // the deliverable open behind it and the stored session model fills
    // the prompt's provider slot.
    let outcome = session.prompt("work".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "served|good|work");
    assert_eq!(session.active_model().as_deref(), Some("good/two"));

    // The held entry never left the chain: once the serving lane dies
    // and the alternate's open refuses `good/two`, the walk offers
    // `plain` — a candidate the `{model}` lane can carry — and it binds.
    let outcome = session.prompt("again".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "served|plain|again");
    assert_eq!(session.active_model().as_deref(), Some("plain"));
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "a/open|first",
            "a/open|good",
            "a/prompt|good",
            "b/open|good/two",
            "b/open|plain",
            "b/prompt|plain",
        ],
        "an undeliverable fallback must not stop the initial walk: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer whose `{model}` open refuses `vnd/prime`, dies
/// mid-handshake on `second`, and serves every prompt on whichever model
/// the wire carried — the alternate whose transport loss leaves the
/// parked claim's lane walk exhausted.
#[cfg(unix)]
const RPC_ALT_OPEN_REFUSES_PRIME_KILLS_SECOND_PROMPT_SERVES: &str = r#"
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
        log.write("alt/open|{}\n".format(model or "unset"))
        log.flush()
        if model == "second":
            # Dying mid-open is a transport loss, not a refusal — the
            # open never answers and the candidate is never spent.
            sys.exit(0)
        if model == "vnd/prime":
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
        else:
            send({"jsonrpc": "2.0", "id": request_id,
                  "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        model = params.get("model")
        log.write("alt/prompt|{}\n".format(model or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "served|{}|{}".format(
                  model or "peerdefault", params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_direct_retry_preserves_the_queued_claim() {
    // `vnd/prime` binds the primary's open; two prompt refusals park the
    // unqualified `why` on the shared pending claim — the
    // `{modelProvider}` prompt emits it absent — before the lane dies,
    // and the alternate's open dies mid-handshake on `second`, the entry
    // the lane switch then carried. The turn exhausts with `why` still
    // the queued default and `zee` untouched in `remaining` behind it.
    let dir = TestDirectory::new("rpc-direct-claim-preserves");
    let dying = dir.path().join("rpc_die.py");
    std::fs::write(&dying, RPC_PROVIDER_OPEN_REFUSES_TWO_PROMPTS_THEN_DIES).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "pending",
        &dying,
        json!({"provider": "{modelProvider}"}),
        json!({"provider": "{modelProvider}", "text": "{prompt}"}),
    );
    attach_call_log(&mut primary, &log);
    let serving = dir.path().join("rpc_serving.py");
    std::fs::write(
        &serving,
        RPC_ALT_OPEN_REFUSES_PRIME_KILLS_SECOND_PROMPT_SERVES,
    )
    .unwrap();
    let mut alternate = rpc_spec(
        "serving",
        &serving,
        json!({"model": "{model}"}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut alternate, &log);
    let registry = registry_with_lanes(
        primary,
        alternate,
        ProfilePatch {
            agent: Some("pending".to_owned()),
            model: Some("vnd/prime".to_owned()),
            models: Some(vec![
                "why".to_owned(),
                "second".to_owned(),
                "zee".to_owned(),
            ]),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    // The implicit prompt spends `vnd/prime`, parks `why` on the shared
    // claim — refused while the wire never carried it — and dies with
    // `second` mid-open on the alternate.
    let error = session.prompt("work".to_owned()).await.unwrap_err();
    assert_eq!(error.code, "agent_transport_failed");

    // The direct turn retries on the alternate: `vnd/prime` is refused
    // at the open, `zee` binds the replacement and serves — but `why`
    // was never this turn's to spend. The claim survives the install
    // and `zee`'s confirmation.
    let outcome = session
        .prompt_direct(TurnRequest {
            text: "raw".to_owned(),
            model: None,
            effort: None,
        })
        .await
        .unwrap();
    assert_eq!(outcome.text.trim_end(), "served|zee|raw");
    assert_eq!(session.active_model().as_deref(), Some("zee"));

    // The preserved claim fills the next implicit prompt — `why` rides
    // the wire at last.
    let outcome = session.prompt("again".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "served|why|again");
    assert_eq!(session.active_model().as_deref(), Some("why"));
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "primary/open|vnd",
            "primary/prompt|vnd",
            "primary/prompt|unset",
            "alt/open|second",
            "alt/open|vnd/prime",
            "alt/open|zee",
            "alt/prompt|zee",
            "alt/prompt|why",
        ],
        "a direct retry must not erase the queued claim: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer with a `{modelProvider}`-only open and prompt: refuses
/// `first` at the open, dies mid-handshake on `good`, and accepts
/// everything else — the transport loss that leaves a held candidate
/// behind for the next lane's walk.
#[cfg(unix)]
const RPC_PROVIDER_OPEN_REFUSES_FIRST_DIES_ON_GOOD: &str = r#"
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
        provider = params.get("provider")
        log.write("a/open|{}\n".format(provider or "unset"))
        log.flush()
        if provider == "first":
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
        elif provider == "good":
            # Dying mid-open is a transport loss, not a refusal — the
            # open never answers and the candidate is never spent.
            sys.exit(0)
        else:
            send({"jsonrpc": "2.0", "id": request_id,
                  "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        provider = params.get("provider")
        log.write("a/prompt|{}\n".format(provider or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "served|{}|{}".format(
                  provider or "peerdefault", params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

/// An rpc peer with a `{modelProvider}`-only open and prompt that
/// accepts every open and echoes whichever provider the wire carried.
#[cfg(unix)]
const RPC_PROVIDER_OPEN_ACCEPTS_ALL_PROMPT_ECHOES: &str = r#"
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
        provider = params.get("provider")
        log.write("b/open|{}\n".format(provider or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        provider = params.get("provider")
        log.write("b/prompt|{}\n".format(provider or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "served|{}|{}".format(
                  provider or "peerdefault", params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_alternate_lane_rescans_held_candidates() {
    // The provider-only lane's open refuses `first/one` and dies
    // mid-handshake on `good/two`, leaving `plain` held as undeliverable
    // for the resumed walk. The alternate is provider-only too: `plain`
    // still resolves nothing there, and a model-free open would let the
    // peer's default serve while `good/two` was never tried. The resumed
    // cursor must be re-checked against the alternate's own slots —
    // `plain` re-holds for a later lane or turn and the walk binds
    // `good/two`.
    let dir = TestDirectory::new("rpc-alternate-rescan");
    let dying = dir.path().join("rpc_dying.py");
    std::fs::write(&dying, RPC_PROVIDER_OPEN_REFUSES_FIRST_DIES_ON_GOOD).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "holding",
        &dying,
        json!({"provider": "{modelProvider}"}),
        json!({"provider": "{modelProvider}", "text": "{prompt}"}),
    );
    attach_call_log(&mut primary, &log);
    let serving = dir.path().join("rpc_serving.py");
    std::fs::write(&serving, RPC_PROVIDER_OPEN_ACCEPTS_ALL_PROMPT_ECHOES).unwrap();
    let mut alternate = rpc_spec(
        "serving",
        &serving,
        json!({"provider": "{modelProvider}"}),
        json!({"provider": "{modelProvider}", "text": "{prompt}"}),
    );
    attach_call_log(&mut alternate, &log);
    let registry = registry_with_lanes(
        primary,
        alternate,
        ProfilePatch {
            agent: Some("holding".to_owned()),
            model: Some("first/one".to_owned()),
            models: Some(vec!["plain".to_owned(), "good/two".to_owned()]),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    // `first/one` refused, `good/two` lost to transport — the resumed
    // walk must not start at `plain`: the alternate cannot deliver it
    // either, and opening on the peer's default would bypass the
    // deliverable `good/two` behind it.
    let outcome = session.prompt("work".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "served|good|work");
    assert_eq!(session.active_model().as_deref(), Some("good/two"));
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "a/open|first",
            "a/open|good",
            "b/open|good",
            "b/prompt|good",
        ],
        "the alternate must rescan the held cursor before opening on \
         the peer default: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer whose `{model}` open accepts and records every model and
/// whose `{modelProvider}` prompt refuses a `vnd` provider and serves
/// the session's bound model otherwise — the lane that can carry a
/// fallback at setup but not on the prompt.
#[cfg(unix)]
const RPC_SETUP_MODEL_PROMPT_PROVIDER: &str = r#"
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
        log.write("open|{}\n".format(session_model or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        provider = params.get("provider")
        log.write("prompt|{}\n".format(provider or "unset"))
        log.flush()
        if provider == "vnd":
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
        else:
            send({"jsonrpc": "2.0", "id": request_id,
                  "result": {"text": "served|{}|{}".format(
                      session_model or "peerdefault",
                      params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_setup_binds_the_fallback_the_prompt_cannot_carry() {
    // The lane binds `{model}` at the open but carries only
    // `{modelProvider}` on prompts: `vnd/prime`'s refusal advances the
    // chain, and the unqualified `plain` resolves no turn slot. The
    // retry must not send the identical model-free request — setup can
    // carry `plain`, so the retry re-establishes the session bound to
    // it, and the replprompt's dropped provider member still serves the
    // bound model.
    let dir = TestDirectory::new("rpc-setup-bound-retry");
    let script = dir.path().join("rpc_setup_model.py");
    std::fs::write(&script, RPC_SETUP_MODEL_PROMPT_PROVIDER).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "binding",
        &script,
        json!({"model": "{model}"}),
        json!({"provider": "{modelProvider}", "text": "{prompt}"}),
    );
    attach_call_log(&mut primary, &log);
    let registry = registry_with(
        primary,
        ProfilePatch {
            agent: Some("binding".to_owned()),
            model: Some("vnd/prime".to_owned()),
            models: Some(vec!["plain".to_owned()]),
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
    assert_eq!(outcome.text.trim_end(), "served|plain|work");
    assert_eq!(session.active_model().as_deref(), Some("plain"));
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec!["open|vnd/prime", "prompt|vnd", "open|plain", "prompt|unset",],
        "a fallback the prompt cannot carry must retry through setup: \
         {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer with a `{modelProvider}`-only open that refuses `second`,
/// accepts everything else, and refuses the model-free prompt — the
/// lane whose replacement open is refused with candidates it cannot
/// express still behind it.
#[cfg(unix)]
const RPC_PROVIDER_OPEN_REFUSES_SECOND_PROMPT_REFUSES: &str = r#"
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
        provider = params.get("provider")
        log.write("a/open|{}\n".format(provider or "unset"))
        log.flush()
        if provider == "second":
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
        else:
            send({"jsonrpc": "2.0", "id": request_id,
                  "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        log.write("a/prompt\n")
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "error": {"code": -32000, "message":
                        "resource_exhausted: quota exceeded for this model"}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

/// An rpc peer with a `{modelProvider}`-only open that refuses `first`,
/// accepts everything else, and a `{modelProvider}` prompt that echoes
/// whichever provider the wire carried — a lane unable to express an
/// unqualified candidate on either surface.
#[cfg(unix)]
const RPC_PROVIDER_OPEN_REFUSES_FIRST_PROMPT_PROVIDER_ECHOES: &str = r#"
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
        provider = params.get("provider")
        log.write("b/open|{}\n".format(provider or "unset"))
        log.flush()
        if provider == "first":
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
        else:
            send({"jsonrpc": "2.0", "id": request_id,
                  "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        provider = params.get("provider")
        log.write("b/prompt|{}\n".format(provider or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "served|{}|{}".format(
                  provider or "peerdefault", params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

/// An rpc peer with a `{model}` open and prompt: accepts every open and
/// echoes whichever model the wire carried — the lane whose slots can
/// express the candidate the earlier lanes could not.
#[cfg(unix)]
const RPC_MODEL_OPEN_ACCEPTS_ALL_PROMPT_ECHOES: &str = r#"
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
        log.write("c/open|{}\n".format(model or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        model = params.get("model")
        log.write("c/prompt|{}\n".format(model or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "served|{}|{}".format(
                  model or "peerdefault", params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_lane_fallback_tries_the_held_candidates() {
    // The prompt refusal advances the chain into a session-bound retry
    // on the provider-only lane, whose open refuses `second/two` while
    // `plain` — unqualified — resolves none of its slots. The refusal
    // must not stand as exhaustion: `plain` stays untried, so the walk
    // restores it with the lane change and the `{model}` lane behind
    // the provider-only alternate gets its chance to bind it.
    let dir = TestDirectory::new("rpc-held-lane-fallback");
    let refusing_a = dir.path().join("rpc_refusing_a.py");
    std::fs::write(&refusing_a, RPC_PROVIDER_OPEN_REFUSES_SECOND_PROMPT_REFUSES).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "providery",
        &refusing_a,
        json!({"provider": "{modelProvider}"}),
        json!({"text": "{prompt}"}),
    );
    attach_call_log(&mut primary, &log);
    let refusing_b = dir.path().join("rpc_refusing_b.py");
    std::fs::write(
        &refusing_b,
        RPC_PROVIDER_OPEN_REFUSES_FIRST_PROMPT_PROVIDER_ECHOES,
    )
    .unwrap();
    let mut alternate = rpc_spec(
        "providery2",
        &refusing_b,
        json!({"provider": "{modelProvider}"}),
        json!({"provider": "{modelProvider}", "text": "{prompt}"}),
    );
    attach_call_log(&mut alternate, &log);
    let serving_c = dir.path().join("rpc_serving_c.py");
    std::fs::write(&serving_c, RPC_MODEL_OPEN_ACCEPTS_ALL_PROMPT_ECHOES).unwrap();
    let mut third = rpc_spec(
        "carrying",
        &serving_c,
        json!({"model": "{model}"}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut third, &log);
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
            agent: Some("providery".to_owned()),
            model: Some("first/one".to_owned()),
            models: Some(vec!["second/two".to_owned(), "plain".to_owned()]),
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

    // `first/one` refused on the bound prompt, `second/two` refused on
    // the replacement open, `plain` held out of that provider-only
    // lane's walk — then refused again on the alternate. The third
    // lane's `{model}` slots bind `plain`, the candidate every earlier
    // lane held.
    let outcome = session.prompt("work".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "served|plain|work");
    assert_eq!(session.active_model().as_deref(), Some("plain"));
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "a/open|first",
            "a/prompt",
            "a/open|second",
            "b/open|first",
            "c/open|plain",
            "c/prompt|plain",
        ],
        "untried candidates must follow the walk to the next lane: \
         {calls_log}"
    );
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn rpc_prompt_refusal_walks_the_untried_candidate_to_the_alternate() {
    // The setup-only primary binds `vendor/prime` through `{modelProvider}`
    // and its model-free prompt is refused: the unqualified `backup`
    // resolves no slot on this lane, so it stays queued untried — never
    // refused. The refusal is this lane's answer, not the chain's: the
    // walk must preserve `backup` and reach the `{model}` alternate,
    // which binds it at open and serves the replay.
    let dir = TestDirectory::new("rpc-untried-lane-fallback");
    let refusing = dir.path().join("rpc_refusing.py");
    std::fs::write(&refusing, RPC_PROVIDER_OPEN_ACCEPTS_PROMPTS_REFUSE_THEN_DIE).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "setuponly",
        &refusing,
        json!({"provider": "{modelProvider}"}),
        json!({"text": "{prompt}"}),
    );
    attach_call_log(&mut primary, &log);
    let serving = dir.path().join("rpc_serving.py");
    std::fs::write(&serving, RPC_MODEL_OPEN_ACCEPTS_ALL_PROMPT_ECHOES).unwrap();
    let mut alternate = rpc_spec(
        "carrying",
        &serving,
        json!({"model": "{model}"}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut alternate, &log);
    let registry = registry_with_lanes(
        primary,
        alternate,
        ProfilePatch {
            agent: Some("setuponly".to_owned()),
            model: Some("vendor/prime".to_owned()),
            models: Some(vec!["backup".to_owned()]),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    // `vendor/prime` refused on the bound prompt, `backup` untried behind
    // it — the alternate carries the candidate the primary could not.
    let outcome = session.prompt("work".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "served|backup|work");
    assert_eq!(session.active_model().as_deref(), Some("backup"));
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "a/open|vendor",
            "a/prompt|unset",
            "c/open|backup",
            "c/prompt|backup",
        ],
        "an untried candidate must follow the walk to a carrying lane: \
         {calls_log}"
    );
    session.close().await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn rpc_refused_driver_pick_never_heads_the_alternate() {
    // The provider-only primary binds `vendor/prime` and can carry the
    // qualified candidates on its prompt too: `vendor/prime` refused,
    // the retry sends `vendor2/next` on the same session — refused as
    // well — while unqualified `backup` waits held out of the lane's
    // walk, never sent. The walk to the alternate must drop every
    // refused entry: the still-held `vendor2/next` must not head the
    // replacement's open and prompt while `backup` stays untried.
    let dir = TestDirectory::new("rpc-refused-pick-fallback");
    let refusing = dir.path().join("rpc_refusing.py");
    std::fs::write(&refusing, RPC_PROVIDER_OPEN_ACCEPTS_PROMPTS_REFUSE_THEN_DIE).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "provideronly",
        &refusing,
        json!({"provider": "{modelProvider}"}),
        json!({"provider": "{modelProvider}", "text": "{prompt}"}),
    );
    attach_call_log(&mut primary, &log);
    let serving = dir.path().join("rpc_serving.py");
    std::fs::write(&serving, RPC_MODEL_OPEN_ACCEPTS_ALL_PROMPT_ECHOES).unwrap();
    let mut alternate = rpc_spec(
        "carrying",
        &serving,
        json!({"model": "{model}"}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut alternate, &log);
    let registry = registry_with_lanes(
        primary,
        alternate,
        ProfilePatch {
            agent: Some("provideronly".to_owned()),
            model: Some("vendor/prime".to_owned()),
            models: Some(vec!["backup".to_owned(), "vendor2/next".to_owned()]),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    // Both qualified entries refused on the primary; the untried
    // `backup` — not the just-refused `vendor2/next` — is what the
    // alternate binds and serves.
    let outcome = session.prompt("work".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "served|backup|work");
    assert_eq!(session.active_model().as_deref(), Some("backup"));
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "a/open|vendor",
            "a/prompt|vendor",
            "a/prompt|vendor2",
            "c/open|backup",
            "c/prompt|backup",
        ],
        "a refused model must never head the alternate's projection: \
         {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer with a `{modelProvider}`-only open that accepts
/// everything and refuses its first three prompts, then closes stdin so
/// the next write lands on a dead pipe — the lane whose death carries
/// the parked claim to a setup-only alternate.
#[cfg(unix)]
const RPC_PROVIDER_OPEN_ACCEPTS_PROMPTS_REFUSE_THEN_DIE: &str = r#"
import json, os, sys, time

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

log = open(os.environ["AWP_CALL_LOG"], "a")
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
        provider = params.get("provider")
        log.write("a/open|{}\n".format(provider or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        prompts += 1
        provider = params.get("provider")
        log.write("a/prompt|{}\n".format(provider or "unset"))
        log.flush()
        if prompts >= 3:
            # stdin closes before the answer flushes, so the next prompt
            # write is guaranteed to land on a dead pipe
            os.close(0)
        send({"jsonrpc": "2.0", "id": request_id,
              "error": {"code": -32000, "message":
                        "resource_exhausted: quota exceeded for this model"}})
        if prompts >= 3:
            time.sleep(30)
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

/// An rpc peer with a `{model}` open that refuses `vnd/prime` and a
/// model-free prompt that serves the session's bound model — the
/// setup-only lane the queued claim can bind on but never ride.
#[cfg(unix)]
const RPC_SETUP_ONLY_OPEN_REFUSES_PRIME_PROMPT_SERVES_BOUND: &str = r#"
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
        log.write("b/open|{}\n".format(session_model or "unset"))
        log.flush()
        if session_model == "vnd/prime":
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
        else:
            send({"jsonrpc": "2.0", "id": request_id,
                  "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        log.write("b/prompt\n")
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "served|{}|{}".format(
                  session_model or "peerdefault",
                  params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_setup_only_lane_delivers_the_queued_claim() {
    // `vnd/prime` binds the primary's open; prompt refusals park the
    // unqualified `why` on the shared pending claim — the
    // `{modelProvider}` prompt emits it absent — while `zee` waits
    // held behind it. The refusal is this lane's answer: the walk
    // installs the setup-only alternate inside the direct turn, where
    // `zee` binds and serves — and a direct turn fills no defaults, so
    // the claim stays queued. The lane has no turn selector, so the
    // next implicit prompt would answer on `zee` forever and leave the
    // claim queued: the session must first re-establish bound to
    // `why`, whose setup slot can carry it.
    let dir = TestDirectory::new("rpc-setup-only-claim");
    let dying = dir.path().join("rpc_die.py");
    std::fs::write(&dying, RPC_PROVIDER_OPEN_ACCEPTS_PROMPTS_REFUSE_THEN_DIE).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "pending",
        &dying,
        json!({"provider": "{modelProvider}"}),
        json!({"provider": "{modelProvider}", "text": "{prompt}"}),
    );
    attach_call_log(&mut primary, &log);
    let serving = dir.path().join("rpc_serving.py");
    std::fs::write(
        &serving,
        RPC_SETUP_ONLY_OPEN_REFUSES_PRIME_PROMPT_SERVES_BOUND,
    )
    .unwrap();
    let mut alternate = rpc_spec(
        "setuponly",
        &serving,
        json!({"model": "{model}"}),
        json!({"text": "{prompt}"}),
    );
    attach_call_log(&mut alternate, &log);
    let registry = registry_with_lanes(
        primary,
        alternate,
        ProfilePatch {
            agent: Some("pending".to_owned()),
            model: Some("vnd/prime".to_owned()),
            models: Some(vec!["why".to_owned(), "zee".to_owned()]),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    // The direct turn spends `vnd/prime` and parks `why` on the shared
    // claim — refused while the wire never carried it — with `zee`
    // held behind it: the walk binds `zee` on the setup-only alternate
    // and serves, and the claim a direct turn never consumed stays
    // queued.
    let outcome = session
        .prompt_direct(TurnRequest {
            text: "raw".to_owned(),
            model: None,
            effort: None,
        })
        .await
        .unwrap();
    assert_eq!(outcome.text.trim_end(), "served|zee|raw");
    assert_eq!(session.active_model().as_deref(), Some("zee"));

    // The next implicit prompt re-establishes the session bound to the
    // claim first — setup can carry what the lane's turn surface
    // cannot — and `why` is served at last.
    let outcome = session.prompt("again".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "served|why|again");
    assert_eq!(session.active_model().as_deref(), Some("why"));
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "a/open|vnd",
            "a/prompt|vnd",
            "a/prompt|unset",
            "a/prompt|unset",
            "b/open|zee",
            "b/prompt",
            "b/open|why",
            "b/prompt",
        ],
        "a setup-only lane must reopen bound to the queued claim: \
         {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer with an `{modelId}`-only open that accepts everything —
/// the slot resolves empty for a `provider/` candidate, so the wire
/// carries no model — and a model-free prompt that serves the session's
/// stored id.
#[cfg(unix)]
const RPC_MODEL_ID_OPEN_ACCEPTS_PROMPT_SERVES_STORED: &str = r#"
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
        session_model = params.get("id")
        log.write("a/open|{}\n".format(session_model or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        log.write("a/prompt\n")
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "served|{}|{}".format(
                  session_model or "peerdefault",
                  params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_empty_model_id_claim_never_enters_repair() {
    // The `{modelId}` slot resolves `vnd/` to an empty id — the adapter
    // emits nothing for it, so the candidate the open accepted without
    // carry evidence can never ride this lane's setup. The queued claim
    // must not re-enter the same repair reopen forever: the prompt
    // dispatches on the session the peer actually bound.
    let dir = TestDirectory::new("rpc-empty-model-id-claim");
    let script = dir.path().join("rpc_model_id.py");
    std::fs::write(&script, RPC_MODEL_ID_OPEN_ACCEPTS_PROMPT_SERVES_STORED).unwrap();
    let log = dir.path().join("calls.log");
    let mut spec = rpc_spec(
        "modelid",
        &script,
        json!({"id": "{modelId}"}),
        json!({"text": "{prompt}"}),
    );
    attach_call_log(&mut spec, &log);
    let registry = registry_with(
        spec,
        ProfilePatch {
            agent: Some("modelid".to_owned()),
            model: Some("vnd/".to_owned()),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    // The open emitted an empty id — the claim waits pending. The
    // implicit prompt must not re-open the identical binding: the lane
    // provably cannot carry `vnd/` anywhere, so the turn serves on the
    // peer's own default with exactly one open on the log.
    let outcome = session.prompt("work".to_owned()).await.unwrap();
    assert_eq!(outcome.text.trim_end(), "served|peerdefault|work");
    assert_eq!(session.active_model(), None);
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec!["a/open|unset", "a/prompt"],
        "a claim no slot can resolve must not re-enter repair: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer whose `{modelProvider}` open accepts everything, then
/// closes stdin so the next write lands on a dead pipe — the lane
/// whose death carries the session to a refusing alternate.
#[cfg(unix)]
const RPC_PROVIDER_OPEN_ACCEPTS_THEN_DIES: &str = r#"
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
    params = request.get("params") or {}
    if method == "session/open":
        log.write("a/open|{}\n".format(params.get("provider") or "unset"))
        log.flush()
        # stdin closes before the answer flushes, so the prompt write is
        # guaranteed to land on a dead pipe
        os.close(0)
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
        time.sleep(30)
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

/// An rpc peer with a `{model}` open that refuses `second` and a
/// model-free prompt that serves the session's bound model — the
/// alternate whose refused head pick must not be re-picked.
#[cfg(unix)]
const RPC_OPEN_REFUSES_SECOND_PROMPT_SERVES_BOUND: &str = r#"
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
        log.write("b/open|{}\n".format(session_model or "unset"))
        log.flush()
        if session_model == "second":
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
        else:
            send({"jsonrpc": "2.0", "id": request_id,
                  "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        log.write("b/prompt\n")
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "served|{}|{}".format(
                  session_model or "peerdefault",
                  params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_alternate_lane_spends_the_refused_head_pick() {
    // Nothing accepted stands and a direct prompt offers no claim: the
    // alternate's open must bind the first deliverable chain entry —
    // and once the peer refuses it at setup, the retry must walk on to
    // the next entry instead of re-picking the model it just sent.
    let dir = TestDirectory::new("rpc-refused-head-pick");
    let dying = dir.path().join("rpc_die.py");
    std::fs::write(&dying, RPC_PROVIDER_OPEN_ACCEPTS_THEN_DIES).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "pending",
        &dying,
        json!({"provider": "{modelProvider}"}),
        json!({"provider": "{modelProvider}", "text": "{prompt}"}),
    );
    attach_call_log(&mut primary, &log);
    let refusing = dir.path().join("rpc_refusing.py");
    std::fs::write(&refusing, RPC_OPEN_REFUSES_SECOND_PROMPT_SERVES_BOUND).unwrap();
    let mut alternate = rpc_spec(
        "refusing",
        &refusing,
        json!({"model": "{model}"}),
        json!({"text": "{prompt}"}),
    );
    attach_call_log(&mut alternate, &log);
    let registry = registry_with_lanes(
        primary,
        alternate,
        ProfilePatch {
            agent: Some("pending".to_owned()),
            model: Some("first".to_owned()),
            models: Some(vec!["second".to_owned(), "third".to_owned()]),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    // `first` parked on the shared claim — the unqualified provider slot
    // emitted nothing — and a direct prompt offers no claim, so the
    // alternate's open attempts the first deliverable remaining entry:
    // `second`, refused, then `third`, accepted and serving.
    let outcome = session
        .prompt_direct(TurnRequest {
            text: "work".to_owned(),
            model: None,
            effort: None,
        })
        .await
        .unwrap();
    assert_eq!(outcome.text.trim_end(), "served|third|work");
    assert_eq!(session.active_model().as_deref(), Some("third"));
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec!["a/open|unset", "b/open|second", "b/open|third", "b/prompt"],
        "a refused head pick must be spent, never re-picked: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer with a `{model}` open that refuses `vnd/prime`, dies
/// mid-open on `why` — the queued claim's repair reopen — and serves
/// every other bound model's prompt.
#[cfg(unix)]
const RPC_SETUP_ONLY_OPEN_DIES_ON_WHY: &str = r#"
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
        log.write("b/open|{}\n".format(session_model or "unset"))
        log.flush()
        if session_model == "why":
            # Dying mid-open is a transport loss, not a refusal.
            sys.exit(0)
        if session_model == "vnd/prime":
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32000, "message":
                            "resource_exhausted: quota exceeded for this model"}})
        else:
            send({"jsonrpc": "2.0", "id": request_id,
                  "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        log.write("b/prompt\n")
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "served|{}|{}".format(
                  session_model or "peerdefault",
                  params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_repair_reopen_transport_loss_returns_the_real_error() {
    // The queued claim's repair reopen on a dead lane must answer with
    // the transport failure the reopen actually reported — with no
    // alternate left, the turn cannot return the synthesized model
    // refusal that only entered the repair.
    let dir = TestDirectory::new("rpc-repair-transport-loss");
    let dying = dir.path().join("rpc_die.py");
    std::fs::write(&dying, RPC_PROVIDER_OPEN_ACCEPTS_PROMPTS_REFUSE_THEN_DIE).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "pending",
        &dying,
        json!({"provider": "{modelProvider}"}),
        json!({"provider": "{modelProvider}", "text": "{prompt}"}),
    );
    attach_call_log(&mut primary, &log);
    let serving = dir.path().join("rpc_serving.py");
    std::fs::write(&serving, RPC_SETUP_ONLY_OPEN_DIES_ON_WHY).unwrap();
    let mut alternate = rpc_spec(
        "setuponly",
        &serving,
        json!({"model": "{model}"}),
        json!({"text": "{prompt}"}),
    );
    attach_call_log(&mut alternate, &log);
    let registry = registry_with_lanes(
        primary,
        alternate,
        ProfilePatch {
            agent: Some("pending".to_owned()),
            model: Some("vnd/prime".to_owned()),
            models: Some(vec!["why".to_owned(), "zee".to_owned()]),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    // Same parked walk as the queued-claim case: `why` waits on the
    // shared claim while `zee` binds the setup-only alternate and
    // serves the direct turn — the alternate's stdin then dies.
    let outcome = session
        .prompt_direct(TurnRequest {
            text: "work".to_owned(),
            model: None,
            effort: None,
        })
        .await
        .unwrap();
    assert_eq!(outcome.text.trim_end(), "served|zee|work");

    // The implicit prompt enters repair for the claim: the same-lane
    // reopen loses its transport mid-open and no alternate remains, so
    // the real reopen error answers the turn — not the marker that
    // entered repair.
    let error = session.prompt("again".to_owned()).await.unwrap_err();
    assert_eq!(error.code, "agent_transport_failed");
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec![
            "a/open|vnd",
            "a/prompt|vnd",
            "a/prompt|unset",
            "a/prompt|unset",
            "b/open|zee",
            "b/prompt",
            "b/open|why",
        ],
        "the claim's repair reopen must die on the dead lane: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer with a `{model}` open that accepts everything, logs each
/// open's and each prompt's resolved model, and serves every prompt —
/// the alternate a transport failover replays onto.
#[cfg(unix)]
const RPC_MODEL_OPEN_PROMPT_LOGS_AND_SERVES: &str = r#"
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
        log.write("b/open|{}\n".format(session_model or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        model = params.get("model") or session_model
        log.write("b/prompt|{}\n".format(params.get("model") or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "served|{}|{}".format(
                  model or "peerdefault", params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_transport_failover_keeps_the_callers_model() {
    // `first` parked on the shared claim — the unqualified provider slot
    // emitted nothing — so no accepted model stands when the primary's
    // pipe dies. A transport failover replays the caller's explicit
    // model on the alternate: the parked claim is a default fill the
    // chain may not promote ahead of a model that was never refused.
    let dir = TestDirectory::new("rpc-failover-caller-model");
    let dying = dir.path().join("rpc_die.py");
    std::fs::write(&dying, RPC_PROVIDER_OPEN_ACCEPTS_THEN_DIES).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "pending",
        &dying,
        json!({"provider": "{modelProvider}"}),
        json!({"provider": "{modelProvider}", "text": "{prompt}"}),
    );
    attach_call_log(&mut primary, &log);
    let serving = dir.path().join("rpc_serve.py");
    std::fs::write(&serving, RPC_MODEL_OPEN_PROMPT_LOGS_AND_SERVES).unwrap();
    let mut alternate = rpc_spec(
        "serving",
        &serving,
        json!({"model": "{model}"}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    attach_call_log(&mut alternate, &log);
    let registry = registry_with_lanes(
        primary,
        alternate,
        ProfilePatch {
            agent: Some("pending".to_owned()),
            model: Some("first".to_owned()),
            models: Some(vec!["second".to_owned(), "third".to_owned()]),
            ..Default::default()
        },
    );
    let agent = registry.resolve("p").unwrap();
    let (tx, _rx) = events();
    let session = agent
        .open_session(SessionRequest::default(), tx)
        .await
        .unwrap();

    // The caller's model was never refused — only the lane died — so the
    // replay carries `callers`, not the parked claim the chain would
    // otherwise promote onto the alternate's per-turn slot.
    let outcome = session
        .prompt_with(TurnRequest {
            text: "work".to_owned(),
            model: Some("callers".to_owned()),
            effort: None,
        })
        .await
        .unwrap();
    assert_eq!(outcome.text.trim_end(), "served|callers|work");
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec!["a/open|unset", "b/open|unset", "b/prompt|callers"],
        "transport failover must not promote a chain entry ahead of the \
         caller's model: {calls_log}"
    );
    session.close().await.unwrap();
}

/// An rpc peer that logs the first request's method and dies before
/// answering — a transport loss during the lane's model-free `initialize`
/// handshake, before the model-carrying `session/open` is ever attempted.
#[cfg(unix)]
const RPC_DIES_ON_FIRST_REQUEST: &str = r#"
import json, os, sys

log = open(os.environ["AWP_CALL_LOG"], "a")
request = json.loads(sys.stdin.readline())
log.write("b/{}\n".format(request.get("method")))
log.flush()
sys.exit(0)
"#;

/// An rpc peer with a `{modelProvider}`-only open and prompt — a lane
/// that can carry only `provider/model` candidates — that accepts
/// everything and serves the bound provider.
#[cfg(unix)]
const RPC_PROVIDER_OPEN_PROMPT_LOGS_AND_SERVES: &str = r#"
import json, os, sys

def send(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()

log = open(os.environ["AWP_CALL_LOG"], "a")
session_provider = None
for line in sys.stdin:
    try:
        request = json.loads(line)
    except Exception:
        continue
    method = request.get("method")
    request_id = request.get("id")
    params = request.get("params") or {}
    if method == "session/open":
        session_provider = params.get("provider")
        log.write("c/open|{}\n".format(session_provider or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"sessionId": "awp-rpc-session"}})
    elif method == "session/prompt":
        provider = params.get("provider") or session_provider
        log.write("c/prompt|{}\n".format(params.get("provider") or "unset"))
        log.flush()
        send({"jsonrpc": "2.0", "id": request_id,
              "result": {"text": "served|{}|{}".format(
                  provider or "peerdefault", params.get("text", ""))}})
    elif request_id is not None:
        send({"jsonrpc": "2.0", "id": request_id, "result": None})
"#;

#[cfg(unix)]
#[tokio::test]
async fn rpc_unsent_alternate_pick_parks_for_a_later_deliverable_entry() {
    // `first` parked on the shared claim as in the refused-pick case; a
    // direct prompt walks to a middle alternate that dies during its
    // model-free `initialize` — its pick `lone` never reached the wire.
    // The last lane can carry `vnd/duo` but cannot resolve `lone`
    // anywhere: the unsent pick parks back into the chain and the next
    // deliverable entry opens and serves — never the peer's own default
    // while a deliverable entry stood untried.
    let dir = TestDirectory::new("rpc-unsent-pick-parks");
    let dying_primary = dir.path().join("rpc_die_primary.py");
    std::fs::write(&dying_primary, RPC_PROVIDER_OPEN_ACCEPTS_THEN_DIES).unwrap();
    let log = dir.path().join("calls.log");
    let mut primary = rpc_spec(
        "pending",
        &dying_primary,
        json!({"provider": "{modelProvider}"}),
        json!({"provider": "{modelProvider}", "text": "{prompt}"}),
    );
    attach_call_log(&mut primary, &log);
    let dying_middle = dir.path().join("rpc_die_middle.py");
    std::fs::write(&dying_middle, RPC_DIES_ON_FIRST_REQUEST).unwrap();
    let mut middle = rpc_spec(
        "dying-init",
        &dying_middle,
        json!({"model": "{model}"}),
        json!({"text": "{prompt}", "model": "{model}"}),
    );
    let LaneSpec::Rpc(lane) = &mut middle.lane else {
        panic!("rpc_spec builds an rpc lane")
    };
    lane.initialize = Some(RpcCallSpec {
        method: "initialize".to_owned(),
        params: json!({}),
    });
    attach_call_log(&mut middle, &log);
    let serving = dir.path().join("rpc_serve.py");
    std::fs::write(&serving, RPC_PROVIDER_OPEN_PROMPT_LOGS_AND_SERVES).unwrap();
    let mut last = rpc_spec(
        "provider-only",
        &serving,
        json!({"provider": "{modelProvider}"}),
        json!({"provider": "{modelProvider}", "text": "{prompt}"}),
    );
    attach_call_log(&mut last, &log);
    let mut document = RegistryDocument::default();
    document.agents.insert(
        primary.id.clone(),
        awp::AgentEntry {
            spec: primary,
            alternates: vec![middle, last],
        },
    );
    document.profiles.insert(
        "p".to_owned(),
        ProfilePatch {
            agent: Some("pending".to_owned()),
            model: Some("first".to_owned()),
            models: Some(vec!["lone".to_owned(), "vnd/duo".to_owned()]),
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

    let outcome = session
        .prompt_direct(TurnRequest {
            text: "work".to_owned(),
            model: None,
            effort: None,
        })
        .await
        .unwrap();
    assert_eq!(outcome.text.trim_end(), "served|vnd|work");
    assert_eq!(session.active_model().as_deref(), Some("vnd/duo"));
    let calls_log = std::fs::read_to_string(&log).unwrap();
    assert_eq!(
        calls_log.lines().collect::<Vec<_>>(),
        vec!["a/open|unset", "b/initialize", "c/open|vnd", "c/prompt|vnd"],
        "an unsent pick must park for the next deliverable entry: {calls_log}"
    );
    session.close().await.unwrap();
}
