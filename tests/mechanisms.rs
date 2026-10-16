//! Profile `mechanisms` (the CAwP section): document parsing, overlay merge
//! and registry-level validation. Adaptation itself is covered in a4aa's
//! mechanisms tests — here only the profile-side wiring is exercised.

use awp::{HookEvent, Registry, RegistryDocument, Support};
use std::path::Path;

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

#[test]
fn semantic_mechanisms_materialize_through_devin() {
    // The same agent-neutral declarations drive a4aa-devin's surfaces:
    // Claude-dialect hooks (incl. PostCompaction), skill-form commands,
    // markdown subagents, `permissions` allow/deny/ask, mcp_config.
    let mechanisms: awp::MechanismSet = serde_json::from_value(serde_json::json!({
        "hooks": [
            {"event": "user-prompt-submit", "command": "inject.sh"},
            {"event": "post-compaction", "command": "compact.sh"},
            {"event": "notification", "command": "notify.sh"}
        ],
        "commands": [{"name": "review", "content": "Run the review."}],
        "subagents": [{"name": "verifier", "model": "opus", "instructions": "Verify."}],
        "permissions": {
            "allow": ["Exec(git)"],
            "deny": ["Exec(rm)"],
            "ask": ["Write(.env*)"]
        },
        "mcpServers": {"fs": {"command": "fs-mcp"}}
    }))
    .unwrap();
    let plan = a4aa_devin::adapt(&mechanisms);
    let has = |path: &str| plan.artifacts.iter().any(|a| a.path == Path::new(path));
    assert!(has(".config/devin/config.json"));
    assert!(has(".config/devin/skills/review/SKILL.md"));
    assert!(has(".config/devin/agents/verifier.md"));
    assert!(has(".config/devin/mcp_config.json"));
    let support = |item: &str| {
        plan.report
            .iter()
            .find(|e| e.item == item)
            .unwrap_or_else(|| panic!("{item} reported"))
            .support
    };
    assert_eq!(support("hooks.post-compaction"), Support::Native);
    // Devin's hook store has no Notification event — reported, not dropped.
    assert_eq!(support("hooks.notification"), Support::Unsupported);
}
