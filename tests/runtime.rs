//! End-to-end profile semantics: instruction composition, model/effort
//! defaults, session cwd and cancellation — through real spawned lanes.
//!
//! The `acp` cases drive `awp fixture-agent`; the `cli` cases use ubiquitous
//! Unix tools (`cat`, `echo`, `sh -c pwd`) so they need no agent at all.

use awp::{
    AgentEvent, AgentSpec, CliLaneSpec, CommandSpec, InstructionMode, LaneSpec, ProfilePatch,
    PromptChannel, Registry, RegistryDocument, SessionRequest, StreamMode, TurnRequest,
};
use tokio::sync::mpsc;

fn cli_spec(id: &str, program: &str, args: &[&str], channel: PromptChannel) -> AgentSpec {
    AgentSpec {
        id: id.to_owned(),
        client_name: None,
        lane: LaneSpec::Cli(CliLaneSpec {
            args: args.iter().map(|a| a.to_string()).collect(),
            prompt_channel: channel,
            stream: StreamMode::Stdio,
        }),
        command: CommandSpec {
            program: program.to_owned(),
            args: vec![],
            cwd: None,
            env: Default::default(),
        },
        max_output_bytes: None,
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
