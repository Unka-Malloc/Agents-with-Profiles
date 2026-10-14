//! Profile `mechanisms` (the CAwP section): document parsing, overlay merge
//! and registry-level validation. Adaptation itself is covered in a4aa's
//! mechanisms tests — here only the profile-side wiring is exercised.

use awp::{HookEvent, Registry, RegistryDocument, Support};

const DOC: &str = r#"{
  "agents": {
    "codex": {"id": "codex", "lane": "cli", "command": {"program": "codex"}}
  },
  "profiles": {
    "reviewer": {
      "agent": "codex",
      "instructions": "Review.",
      "mechanisms": {
        "hooks": [{"event": "session-start", "command": "load-context.sh"}],
        "rules": [{"name": "core", "content": "Never guess."}],
        "subagents": [{"name": "verifier", "instructions": "Verify."}],
        "permissions": {"mode": "workspace-write", "allow": ["Shell(ls)"]},
        "nativeProfile": "reviewer"
      }
    }
  },
  "bindings": {"review": "reviewer"}
}"#;

#[test]
fn mechanisms_parse_inside_profile_document() {
    let registry = Registry::from_json(DOC).expect("document parses");
    let mechanisms = registry
        .profile("reviewer")
        .unwrap()
        .mechanisms
        .as_ref()
        .expect("mechanisms present");
    assert_eq!(mechanisms.hooks.len(), 1);
    assert_eq!(mechanisms.hooks[0].event, HookEvent::SessionStart);
    assert_eq!(mechanisms.native_profile.as_deref(), Some("reviewer"));
}

#[test]
fn overlay_merges_mechanism_declarations() {
    let mut registry = Registry::from_json(DOC).unwrap();
    let overlay: RegistryDocument = serde_json::from_value(serde_json::json!({
        "profiles": {
            "reviewer": {
                "mechanisms": {
                    "rules": [{"name": "core", "content": "Never guess. Always verify."}],
                    "hooks": [
                        {"event": "session-start", "command": "load-context.sh"},
                        {"event": "stop", "command": "receipt.sh"}
                    ]
                }
            }
        }
    }))
    .unwrap();
    registry.apply(overlay).expect("overlay applies");

    let mechanisms = registry
        .profile("reviewer")
        .unwrap()
        .mechanisms
        .clone()
        .unwrap();
    assert_eq!(mechanisms.rules.len(), 1);
    assert_eq!(mechanisms.rules[0].content, "Never guess. Always verify.");
    assert_eq!(mechanisms.hooks.len(), 2);
    assert_eq!(mechanisms.hooks[1].event, HookEvent::Stop);
    assert!(mechanisms.subagents.iter().any(|s| s.name == "verifier"));
}

#[test]
fn malformed_mechanisms_fail_atomically() {
    let mut registry = Registry::from_json(DOC).unwrap();
    let overlay: RegistryDocument = serde_json::from_value(serde_json::json!({
        "profiles": {
            "reviewer": {
                "mechanisms": {
                    "hooks": [{"event": "stop", "command": "   "}]
                }
            }
        }
    }))
    .unwrap();
    assert!(registry.apply(overlay).is_err());
    // Failed apply leaves the previous profile untouched.
    let mechanisms = registry
        .profile("reviewer")
        .unwrap()
        .mechanisms
        .as_ref()
        .unwrap();
    assert_eq!(mechanisms.hooks.len(), 1);
}

#[test]
fn resolved_agent_carries_mechanisms_for_adaptation() {
    let registry = Registry::from_json(DOC).unwrap();
    let agent = registry
        .resolve_binding("review")
        .expect("binding resolves");
    let mechanisms = agent.profile().mechanisms.as_ref().unwrap();

    // A host driving only Codex depends on `a4aa-codex` directly — no
    // registry, no other adapter compiled in.
    let plan = a4aa_codex::adapt(mechanisms);
    assert!(!plan.artifacts.is_empty());
    assert!(plan
        .report
        .iter()
        .all(|entry| entry.support != Support::Unsupported || entry.item == "permissions"));
}
