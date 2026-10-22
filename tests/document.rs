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
    assert_eq!(implementer.state_scope, None);

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
fn state_scope_roundtrips_project_and_global() {
    for (raw, expected) in [
        ("project", awp::StateScope::Project),
        ("global", awp::StateScope::Global),
    ] {
        let doc = format!(
            r#"{{
              "agents": {{"a": {{"id": "a", "lane": "cli",
                  "command": {{"program": "true"}}}}}},
              "profiles": {{"p": {{"agent": "a", "stateScope": "{raw}"}}}},
              "bindings": {{}}
            }}"#
        );
        let registry = awp::Registry::from_json(&doc).expect("valid document");
        let profile = registry.profile("p").expect("profile resolves");
        assert_eq!(profile.state_scope, Some(expected));

        // `va set profile` / `va show profile` round-trip the patch form.
        let wire = serde_json::to_value(profile.to_patch()).unwrap();
        assert_eq!(wire["stateScope"], raw);
        let back: awp::ProfilePatch = serde_json::from_value(wire).unwrap();
        assert_eq!(back.state_scope, Some(expected));
    }
}

#[test]
fn state_scope_absent_writes_no_marker() {
    let patch = awp::ProfilePatch {
        agent: Some("a".to_owned()),
        ..Default::default()
    };
    let value = serde_json::to_value(&patch).unwrap();
    assert!(value.get("stateScope").is_none());

    let profile = awp::Profile::from_patch("p", patch).unwrap();
    assert_eq!(profile.state_scope, None);
    let profile_value = serde_json::to_value(&profile).unwrap();
    assert!(profile_value.get("stateScope").is_none());
}

#[test]
fn unknown_state_scope_is_rejected() {
    let err = serde_json::from_value::<awp::ProfilePatch>(json!({
        "agent": "a",
        "stateScope": "everywhere"
    }));
    assert!(err.is_err());
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
    assert!(value.get("stateScope").is_none());
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

#[test]
fn secrets_and_providers_serialize_and_deserialize() {
    let json_text = r#"{
      "agents": {
        "echo": {
          "id": "echo",
          "lane": "cli",
          "command": {"program": "echo"}
        }
      },
      "profiles": {
        "multi": {
          "agent": "echo",
          "providers": {
            "primary": {
              "vendor": "anthropic",
              "models": ["claude-3-7-sonnet", "claude-3-5-haiku"]
            }
          }
        }
      },
      "bindings": {},
      "secrets": {
        "API_KEY": [
          {"env": "CUSTOM_KEY"},
          {"json": {"path": "{home}/.config/app/config.json", "key": "auth.apiKey"}},
          {"yaml": {"path": "{home}/.config/app/config.yaml", "key": "auth.api_key"}},
          {"toml": {"path": "{home}/.config/app/config.toml", "key": "credentials.token"}}
        ]
      },
      "providers": {
        "deepseek": {
          "vendor": "deepseek",
          "baseUrl": "https://api.deepseek.com/v1",
          "authSecret": "DEEPSEEK_API_KEY",
          "models": ["deepseek-chat", "deepseek-reasoner"]
        }
      }
    }"#;

    let doc: RegistryDocument = serde_json::from_str(json_text).expect("parses document");
    assert_eq!(doc.secrets["API_KEY"].len(), 4);
    assert_eq!(doc.providers["deepseek"].vendor, "deepseek");
    assert_eq!(doc.providers["deepseek"].models.len(), 2);
    let reg = awp::Registry::from_document(doc).expect("valid registry");
    let profile = reg.profile("multi").expect("profile exists");
    assert_eq!(profile.providers["primary"].vendor, "anthropic");
    assert_eq!(profile.providers["primary"].models[0], "claude-3-7-sonnet");
    assert_eq!(reg.provider("deepseek").unwrap().vendor, "deepseek");
}

#[test]
fn secret_recipe_resolution_formats() {
    let temp_dir = std::env::temp_dir().join(format!("awp-sec-test-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&temp_dir);

    let toml_path = temp_dir.join("test.toml");
    std::fs::write(
        &toml_path,
        "[credentials]\ntoken = \"toml-secret-value\"\nother = 123\n",
    )
    .unwrap();

    let json_path = temp_dir.join("test.json");
    std::fs::write(
        &json_path,
        r#"{"auth": {"nested": {"key": "json-secret-value"}}}"#,
    )
    .unwrap();

    let yaml_path = temp_dir.join("test.yaml");
    std::fs::write(
        &yaml_path,
        "auth:\n  nested:\n    key: 'yaml-secret-value'\n",
    )
    .unwrap();

    std::env::set_var("AWP_TEST_ENV_VAR", "env-secret-value");

    let recipes = vec![
        awp::SecretRecipe::Env {
            env: "AWP_NON_EXISTENT_VAR".into(),
        },
        awp::SecretRecipe::Toml {
            toml: awp::FileSecret {
                path: toml_path.to_str().unwrap().into(),
                key: "credentials.token".into(),
            },
        },
    ];
    assert_eq!(
        awp::resolve_secret(&recipes).as_deref(),
        Some("toml-secret-value")
    );

    let json_recipe = awp::SecretRecipe::Json {
        json: awp::FileSecret {
            path: json_path.to_str().unwrap().into(),
            key: "auth.nested.key".into(),
        },
    };
    assert_eq!(json_recipe.resolve().as_deref(), Some("json-secret-value"));

    let yaml_recipe = awp::SecretRecipe::Yaml {
        yaml: awp::FileSecret {
            path: yaml_path.to_str().unwrap().into(),
            key: "auth.nested.key".into(),
        },
    };
    assert_eq!(yaml_recipe.resolve().as_deref(), Some("yaml-secret-value"));

    let env_recipe = awp::SecretRecipe::Env {
        env: "AWP_TEST_ENV_VAR".into(),
    };
    assert_eq!(env_recipe.resolve().as_deref(), Some("env-secret-value"));

    let _ = std::fs::remove_dir_all(temp_dir);
}
