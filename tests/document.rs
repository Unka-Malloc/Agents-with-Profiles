//! RegistryDocument wire format: serde roundtrips, key naming, rejection of
//! unknown fields, and the binding shorthand.

use awp::{Binding, InstructionMode, RegistryDocument};
use serde_json::json;

const DOC: &str = r#"{
  "agents": {
    "fixture": {
      "id": "fixture",
      "clientName": "host",
      "lane": "acp",
      "preferLoad": false,
      "mcpServers": [],
      "command": {"program": "fixture-bin", "args": ["acp"]}
    }
  },
  "profiles": {
    "reviewer": {
      "agent": "fixture",
      "name": "Reviewer",
      "revision": "3",
      "model": "m-large",
      "effort": "high",
      "permission": "prompt",
      "instructions": "Review carefully.",
      "instructionMode": "first-turn",
      "contextWindow": 200000,
      "env": {"FLAG": "1", "TOKEN": {"secret": {"secret": "vault/token"}},
            "INHERITED": {"hostEnv": {"host_env": "HOST_VAR"}}},
      "mcpServers": [{"name": "tools"}],
      "requiredCapabilities": ["fs:read"],
      "preferredCapabilities": ["fs:write"],
      "skillReferences": ["skill/review"],
      "stateKey": "reviewer-state",
      "extensions": {"avatar": "owl"}
    },
    "implementer": {
      "agent": "fixture",
      "instructionMode": "every-turn"
    }
  },
  "bindings": {
    "review": {"profile": "reviewer", "prompt": "Review the diff."},
    "implement": "implementer"
  }
}"#;

#[test]
fn document_roundtrips_and_validates() {
    let registry = awp::Registry::from_json(DOC).expect("valid document");

    let reviewer = registry.profile("reviewer").expect("profile resolves");
    assert_eq!(reviewer.agent, "fixture");
    assert_eq!(reviewer.revision.as_deref(), Some("3"));
    assert_eq!(reviewer.instruction_mode, InstructionMode::FirstTurn);
    assert_eq!(reviewer.context_window, Some(200_000));
    assert_eq!(reviewer.required_capabilities, ["fs:read"]);
    assert_eq!(reviewer.state_key.as_deref(), Some("reviewer-state"));
    assert_eq!(reviewer.extensions["avatar"], json!("owl"));
    match reviewer.env.get("TOKEN") {
        Some(awp::EnvValue::Source(awp::EnvSource::Secret { secret })) => {
            assert_eq!(secret, "vault/token")
        }
        other => panic!("unexpected env value: {other:?}"),
    }
    match reviewer.env.get("INHERITED") {
        Some(awp::EnvValue::Source(awp::EnvSource::HostEnv { host_env })) => {
            assert_eq!(host_env, "HOST_VAR")
        }
        other => panic!("unexpected env value: {other:?}"),
    }

    let implementer = registry.profile("implementer").unwrap();
    assert_eq!(implementer.instruction_mode, InstructionMode::EveryTurn);

    // Binding shorthand deserializes as a bare profile reference.
    let shorthand = registry.binding("implement").unwrap();
    assert_eq!(shorthand.profile(), "implementer");
    assert_eq!(shorthand.prompt(), None);

    let full = registry.binding("review").unwrap();
    assert_eq!(full.prompt(), Some("Review the diff."));
}

#[test]
fn document_surfaces_resolved_profiles() {
    let registry = awp::Registry::from_json(DOC).unwrap();
    let mut ids: Vec<&str> = registry.profiles().map(|p| p.id.as_str()).collect();
    ids.sort_unstable();
    assert_eq!(ids, ["implementer", "reviewer"]);
    assert_eq!(registry.agents().len(), 1);
    assert_eq!(registry.bindings().len(), 2);
}

#[test]
fn unknown_fields_are_rejected() {
    let bad = r#"{"agents": {}, "profiles": {}, "bindings": {}, "surprise": 1}"#;
    assert!(serde_json::from_str::<RegistryDocument>(bad).is_err());
}

#[test]
fn profile_patch_serializes_camel_case() {
    let patch = awp::ProfilePatch {
        agent: Some("a".to_owned()),
        instruction_mode: Some(InstructionMode::EveryTurn),
        context_window: Some(4096),
        ..Default::default()
    };
    let value = serde_json::to_value(&patch).unwrap();
    assert_eq!(value["agent"], "a");
    assert_eq!(value["instructionMode"], "every-turn");
    assert_eq!(value["contextWindow"], 4096);
    // Unset fields stay absent — no null noise on the wire.
    assert!(value.get("model").is_none() || value["model"].is_null());
}

#[test]
fn binding_serde_forms() {
    let short: Binding = serde_json::from_value(json!("p1")).unwrap();
    assert_eq!(short.profile(), "p1");

    let full: Binding = serde_json::from_value(json!({"profile": "p2", "prompt": "do"})).unwrap();
    assert_eq!(full.profile(), "p2");
    assert_eq!(full.prompt(), Some("do"));

    let back = serde_json::to_value(&full).unwrap();
    assert_eq!(back, json!({"profile": "p2", "prompt": "do"}));
}
