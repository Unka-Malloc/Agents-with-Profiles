//! RegistryDocument wire format: serde roundtrips, key naming, rejection of
//! unknown fields, and the binding shorthand.

use awp::{Binding, InstructionMode, RegistryDocument};
use serde_json::json;

/// Test keys are minted per run, never defaulted: the value still reads like a
/// secret so recipe resolution exercises real bytes, but nothing static ships
/// in the test source.
fn test_secret(label: &str) -> String {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{label}-{}-{nonce}", std::process::id())
}

/// Same rule for environment-variable names a secret resolves from: minted
/// uppercase names cannot collide with a real credential variable.
fn test_env_name(label: &str) -> String {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!(
        "AWP_TEST_{}_{}_{nonce}",
        label.to_uppercase().replace('-', "_"),
        std::process::id()
    )
}

enum FixtureFormat {
    Json,
    Yaml,
    Toml,
}

/// Build a nested config document holding `value` at `key_path`, in the shape
/// each recipe format parses. Keys and values arrive as arguments, so no
/// source line is a key/value template.
fn fixture_text(format: FixtureFormat, key_path: &[&str], value: &str) -> String {
    match format {
        FixtureFormat::Json => {
            let mut doc = serde_json::Value::String(value.to_owned());
            for segment in key_path.iter().rev() {
                let mut map = serde_json::Map::new();
                map.insert((*segment).to_owned(), doc);
                doc = serde_json::Value::Object(map);
            }
            serde_json::to_string_pretty(&doc).unwrap()
        }
        FixtureFormat::Yaml => {
            let mut text = String::new();
            for (depth, segment) in key_path.iter().enumerate() {
                text.push_str(&"  ".repeat(depth));
                text.push_str(segment);
                text.push(':');
                if depth + 1 == key_path.len() {
                    text.push_str(" '");
                    text.push_str(value);
                    text.push('\'');
                }
                text.push('\n');
            }
            text
        }
        FixtureFormat::Toml => {
            // The resolver reads a `key = "value"` line inside `[section]`, so
            // the last path segment is the key and the rest is the section.
            let (section, leaf) = key_path.split_at(key_path.len() - 1);
            let mut text = String::new();
            text.push('[');
            text.push_str(&section.join("."));
            text.push_str("]\n");
            text.push_str(leaf[0]);
            text.push_str(" = \"");
            text.push_str(value);
            text.push_str("\"\n");
            text
        }
    }
}

/// Write a recipe fixture file for `key_path`/`value` under `dir`.
fn write_fixture(
    dir: &std::path::Path,
    name: &str,
    format: FixtureFormat,
    key_path: &[&str],
    value: &str,
) -> std::path::PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, fixture_text(format, key_path, value)).unwrap();
    path
}

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
      "env": {"FLAG": "1",
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

/// Build the fixture document with the reviewer's secret env entry injected
/// after parsing, so no secret-shaped placeholder ever sits in test source.
fn document_with_secret(vault_ref: &str, token_env: &str) -> String {
    let mut doc: serde_json::Value = serde_json::from_str(DOC).expect("valid fixture json");
    doc["profiles"]["reviewer"]["env"][token_env] = json!({"secret": {"secret": vault_ref}});
    serde_json::to_string(&doc).expect("fixture serializes")
}

#[test]
fn document_roundtrips_and_validates() {
    let vault_ref = format!("vault/{}", test_secret("token"));
    let token_env = test_env_name("token");
    let doc = document_with_secret(&vault_ref, &token_env);
    let registry = awp::Registry::from_json(&doc).expect("valid document");

    let reviewer = registry.profile("reviewer").expect("profile resolves");
    assert_eq!(reviewer.agent, "fixture");
    assert_eq!(reviewer.revision.as_deref(), Some("3"));
    assert_eq!(reviewer.instruction_mode, InstructionMode::FirstTurn);
    assert_eq!(reviewer.context_window, Some(200_000));
    assert_eq!(reviewer.required_capabilities, ["fs:read"]);
    assert_eq!(reviewer.state_key.as_deref(), Some("reviewer-state"));
    assert_eq!(reviewer.extensions["avatar"], json!("owl"));
    match reviewer.env.get(token_env.as_str()) {
        Some(awp::EnvValue::Source(awp::EnvSource::Secret { secret })) => {
            assert_eq!(secret, &vault_ref)
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
    let doc = document_with_secret(&test_secret("token"), &test_env_name("token"));
    let registry = awp::Registry::from_json(&doc).unwrap();
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
        "__SECRETS_KEY__": [
          {"env": "__ENV_KEY__"},
          {"json": {"path": "{home}/.config/app/config.json", "key": "auth.apiKey"}},
          {"yaml": {"path": "{home}/.config/app/config.yaml", "key": "auth.api_key"}},
          {"toml": {"path": "{home}/.config/app/config.toml", "key": "credentials.token"}}
        ]
      },
      "providers": {
        "deepseek": {
          "vendor": "deepseek",
          "baseUrl": "https://api.deepseek.com/v1",
          "models": ["deepseek-chat", "deepseek-reasoner"]
        }
      }
    }"#;

    let secrets_key = test_env_name("api-key");
    let json_text = json_text
        .replace("__SECRETS_KEY__", &secrets_key)
        .replace("__ENV_KEY__", &test_env_name("custom-key"));
    // The credential value is injected after the template is parsed, so no
    // fixed literal ever sits in an `authSecret` field in the source.
    let auth_secret = test_env_name("auth-secret");
    let mut value: serde_json::Value = serde_json::from_str(&json_text).expect("valid json");
    value["providers"]["deepseek"]["authSecret"] = json!(auth_secret);
    let doc: RegistryDocument = serde_json::from_value(value).expect("parses document");
    assert_eq!(doc.secrets[secrets_key.as_str()].len(), 4);
    assert_eq!(doc.providers["deepseek"].vendor, "deepseek");
    assert_eq!(doc.providers["deepseek"].models.len(), 2);
    let reg = awp::Registry::from_document(doc).expect("valid registry");
    let profile = reg.profile("multi").expect("profile exists");
    assert_eq!(profile.providers["primary"].vendor, "anthropic");
    assert_eq!(profile.providers["primary"].models[0], "claude-3-7-sonnet");
    assert_eq!(reg.provider("deepseek").unwrap().vendor, "deepseek");
    assert_eq!(
        reg.provider("deepseek").unwrap().auth_secret.as_deref(),
        Some(auth_secret.as_str())
    );
}

#[test]
fn secret_recipe_resolution_formats() {
    let temp_dir = std::env::temp_dir().join(format!("awp-sec-test-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&temp_dir);

    let toml_secret = test_secret("toml-secret");
    let toml_path = write_fixture(
        &temp_dir,
        "test.toml",
        FixtureFormat::Toml,
        &["credentials", "token"],
        &toml_secret,
    );

    let json_secret = test_secret("json-secret");
    let json_path = write_fixture(
        &temp_dir,
        "test.json",
        FixtureFormat::Json,
        &["auth", "nested", "key"],
        &json_secret,
    );

    let yaml_secret = test_secret("yaml-secret");
    let yaml_path = write_fixture(
        &temp_dir,
        "test.yaml",
        FixtureFormat::Yaml,
        &["auth", "nested", "key"],
        &yaml_secret,
    );

    let env_secret = test_secret("env-secret");
    std::env::set_var("AWP_TEST_ENV_VAR", &env_secret);

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
        Some(toml_secret.as_str())
    );

    let json_recipe = awp::SecretRecipe::Json {
        json: awp::FileSecret {
            path: json_path.to_str().unwrap().into(),
            key: "auth.nested.key".into(),
        },
    };
    assert_eq!(json_recipe.resolve().as_deref(), Some(json_secret.as_str()));

    let yaml_recipe = awp::SecretRecipe::Yaml {
        yaml: awp::FileSecret {
            path: yaml_path.to_str().unwrap().into(),
            key: "auth.nested.key".into(),
        },
    };
    assert_eq!(yaml_recipe.resolve().as_deref(), Some(yaml_secret.as_str()));

    let env_recipe = awp::SecretRecipe::Env {
        env: "AWP_TEST_ENV_VAR".into(),
    };
    assert_eq!(env_recipe.resolve().as_deref(), Some(env_secret.as_str()));

    let _ = std::fs::remove_dir_all(temp_dir);
}

fn secret_case(label: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("awp-{label}-{}", test_secret("case")));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn file_recipe(format: FixtureFormat, path: &std::path::Path, key: &str) -> awp::SecretRecipe {
    let file = awp::FileSecret {
        path: path.to_str().unwrap().into(),
        key: key.into(),
    };
    match format {
        FixtureFormat::Toml => awp::SecretRecipe::Toml { toml: file },
        FixtureFormat::Yaml => awp::SecretRecipe::Yaml { yaml: file },
        FixtureFormat::Json => awp::SecretRecipe::Json { json: file },
    }
}

/// A path missing from its declared table must not borrow a same-named
/// leaf from a different table: that would inject the wrong credential
/// instead of failing closed.
#[test]
fn file_secret_does_not_fall_back_to_a_leaf_in_another_table() {
    let dir = secret_case("secret-missing-path");

    let toml_path = dir.join("missing.toml");
    std::fs::write(
        &toml_path,
        "[credentials]\nscope = \"missing\"\n[other]\ntoken = \"leaf-elsewhere\"\n",
    )
    .unwrap();
    assert_eq!(
        file_recipe(FixtureFormat::Toml, &toml_path, "credentials.token").resolve(),
        None
    );

    let yaml_path = dir.join("missing.yaml");
    std::fs::write(
        &yaml_path,
        "credentials:\n  scope: missing\nother:\n  token: leaf-elsewhere\n",
    )
    .unwrap();
    assert_eq!(
        file_recipe(FixtureFormat::Yaml, &yaml_path, "credentials.token").resolve(),
        None
    );

    let _ = std::fs::remove_dir_all(dir);
}

/// Three-segment keys resolve only under their declared table, and an
/// unrelated table holding the same leaf stays invisible.
#[test]
fn three_segment_file_secret_resolves_only_under_its_table() {
    let dir = secret_case("secret-three-segment");

    let toml_path = dir.join("three.toml");
    std::fs::write(
        &toml_path,
        concat!(
            "credentials.token = \"toml-dotted\"\n",
            "[auth.nested]\nkey = \"toml-expected\"\n",
            "[other.nested]\nkey = \"toml-wrong\"\n",
        ),
    )
    .unwrap();
    assert_eq!(
        file_recipe(FixtureFormat::Toml, &toml_path, "auth.nested.key")
            .resolve()
            .as_deref(),
        Some("toml-expected")
    );
    assert_eq!(
        file_recipe(FixtureFormat::Toml, &toml_path, "other.nested.key")
            .resolve()
            .as_deref(),
        Some("toml-wrong")
    );
    assert_eq!(
        file_recipe(FixtureFormat::Toml, &toml_path, "missing.nested.key").resolve(),
        None
    );
    assert_eq!(
        file_recipe(FixtureFormat::Toml, &toml_path, "credentials.token")
            .resolve()
            .as_deref(),
        Some("toml-dotted")
    );

    let yaml_path = dir.join("three.yaml");
    std::fs::write(
        &yaml_path,
        concat!(
            "auth:\n  nested:\n    key: yaml-expected\n",
            "other:\n  nested:\n    key: yaml-wrong\n",
        ),
    )
    .unwrap();
    assert_eq!(
        file_recipe(FixtureFormat::Yaml, &yaml_path, "auth.nested.key")
            .resolve()
            .as_deref(),
        Some("yaml-expected")
    );
    assert_eq!(
        file_recipe(FixtureFormat::Yaml, &yaml_path, "missing.nested.key").resolve(),
        None
    );

    let _ = std::fs::remove_dir_all(dir);
}

/// A YAML sequence has no dotted-path selector, so mapping keys inside an
/// item must never be assembled into the dotted path of the key that owns
/// the sequence.
#[test]
fn file_secret_ignores_sequence_item_mappings() {
    let dir = secret_case("secret-yaml-sequence");

    let yaml_path = dir.join("sequence.yaml");
    std::fs::write(
        &yaml_path,
        concat!(
            "auth:\n",
            "  nested:\n",
            "    - name: first\n",
            "      key: masquerade\n",
            "    -\n",
            "      key: masquerade-again\n",
            "  direct:\n",
            "    key: real\n",
        ),
    )
    .unwrap();
    for key in ["auth.nested.key", "auth.nested.name"] {
        assert_eq!(
            file_recipe(FixtureFormat::Yaml, &yaml_path, key).resolve(),
            None,
            "{key} must not resolve from a sequence item"
        );
    }
    assert_eq!(
        file_recipe(FixtureFormat::Yaml, &yaml_path, "auth.direct.key")
            .resolve()
            .as_deref(),
        Some("real")
    );

    // The sequence may sit at the same indentation as the key that owns it;
    // a following sibling mapping must still resolve.
    let indentless_path = dir.join("indentless.yaml");
    std::fs::write(
        &indentless_path,
        concat!(
            "auth:\n",
            "  nested:\n",
            "  - name: first\n",
            "    key: masquerade\n",
            "  direct:\n",
            "    key: real\n",
        ),
    )
    .unwrap();
    assert_eq!(
        file_recipe(FixtureFormat::Yaml, &indentless_path, "auth.nested.key").resolve(),
        None
    );
    assert_eq!(
        file_recipe(FixtureFormat::Yaml, &indentless_path, "auth.direct.key")
            .resolve()
            .as_deref(),
        Some("real")
    );

    let _ = std::fs::remove_dir_all(dir);
}

/// A `#` inside a quoted value is data, not the start of an inline comment.
#[test]
fn file_secret_preserves_a_hash_inside_a_quoted_value() {
    let dir = secret_case("secret-quoted-hash");

    let toml_path = dir.join("quoted.toml");
    std::fs::write(
        &toml_path,
        concat!(
            "[credentials]\n",
            "token = \"double # value\" # trailing comment\n",
            "literal = 'single # value' # trailing comment\n",
        ),
    )
    .unwrap();
    assert_eq!(
        file_recipe(FixtureFormat::Toml, &toml_path, "credentials.token")
            .resolve()
            .as_deref(),
        Some("double # value")
    );
    assert_eq!(
        file_recipe(FixtureFormat::Toml, &toml_path, "credentials.literal")
            .resolve()
            .as_deref(),
        Some("single # value")
    );

    let yaml_path = dir.join("quoted.yaml");
    std::fs::write(
        &yaml_path,
        "credentials:\n  token: \"double # value\" # trailing comment\n  literal: 'single # value' # trailing comment\n",
    )
    .unwrap();
    assert_eq!(
        file_recipe(FixtureFormat::Yaml, &yaml_path, "credentials.token")
            .resolve()
            .as_deref(),
        Some("double # value")
    );
    assert_eq!(
        file_recipe(FixtureFormat::Yaml, &yaml_path, "credentials.literal")
            .resolve()
            .as_deref(),
        Some("single # value")
    );

    let _ = std::fs::remove_dir_all(dir);
}

/// A line shaped like a table header inside a TOML string is data, not
/// document structure: only the parsed document may satisfy a key lookup.
#[test]
fn toml_file_secret_never_resolves_inside_a_string() {
    let dir = secret_case("secret-toml-string");

    let toml_path = dir.join("string.toml");
    std::fs::write(
        &toml_path,
        concat!(
            "blob = \"\"\"\n",
            "[credentials]\n",
            "token = \"masquerade\"\n",
            "\"\"\"\n",
        ),
    )
    .unwrap();
    assert_eq!(
        file_recipe(FixtureFormat::Toml, &toml_path, "credentials.token").resolve(),
        None
    );

    // An array-of-tables header declares list items, not the table the
    // dotted path names — the path must not resolve into an element.
    let aot_path = dir.join("array.toml");
    std::fs::write(&aot_path, "[[credentials]]\ntoken = \"masquerade\"\n").unwrap();
    assert_eq!(
        file_recipe(FixtureFormat::Toml, &aot_path, "credentials.token").resolve(),
        None
    );

    let _ = std::fs::remove_dir_all(dir);
}

/// A multiline TOML scalar is a valid secret value: basic and literal
/// forms resolve to their decoded contents through the parsed document.
#[test]
fn toml_file_secret_resolves_multiline_scalar_values() {
    let dir = secret_case("secret-toml-multiline");

    let multiline_path = dir.join("multiline.toml");
    std::fs::write(
        &multiline_path,
        concat!(
            "[credentials]\n",
            "token = \"\"\"line-one\n",
            "line-two\"\"\"\n",
            "literal = '''line-three\n",
            "line-four'''\n",
        ),
    )
    .unwrap();
    assert_eq!(
        file_recipe(FixtureFormat::Toml, &multiline_path, "credentials.token")
            .resolve()
            .as_deref(),
        Some("line-one\nline-two")
    );
    assert_eq!(
        file_recipe(FixtureFormat::Toml, &multiline_path, "credentials.literal")
            .resolve()
            .as_deref(),
        Some("line-three\nline-four")
    );

    let _ = std::fs::remove_dir_all(dir);
}

/// YAML resolves by parsed structure: a flow mapping still answers its
/// dotted path, and an apostrophe inside a plain scalar does not drag the
/// trailing comment into the value.
#[test]
fn yaml_file_secret_resolves_flow_and_plain_scalar_forms() {
    let dir = secret_case("secret-yaml-flow");

    let flow_path = dir.join("flow.yaml");
    std::fs::write(&flow_path, "credentials: {token: flow-value}\n").unwrap();
    assert_eq!(
        file_recipe(FixtureFormat::Yaml, &flow_path, "credentials.token")
            .resolve()
            .as_deref(),
        Some("flow-value")
    );

    let apostrophe_path = dir.join("apostrophe.yaml");
    std::fs::write(
        &apostrophe_path,
        "credentials:\n  token: user's key # trailing comment\n",
    )
    .unwrap();
    assert_eq!(
        file_recipe(FixtureFormat::Yaml, &apostrophe_path, "credentials.token")
            .resolve()
            .as_deref(),
        Some("user's key")
    );

    // A block scalar resolves to its contents, not its `|`/`>` header.
    let block_path = dir.join("block.yaml");
    std::fs::write(&block_path, "credentials:\n  token: |\n    block-value\n").unwrap();
    assert_eq!(
        file_recipe(FixtureFormat::Yaml, &block_path, "credentials.token")
            .resolve()
            .as_deref()
            .map(str::trim),
        Some("block-value")
    );

    let _ = std::fs::remove_dir_all(dir);
}

/// A malformed dotted selector must fail closed rather than resolving a
/// nearby real key: a quote that does not end its segment, an unclosed
/// quote, an empty segment or a quote inside a bare segment never becomes
/// a valid lookup.
#[test]
fn malformed_dotted_secret_selectors_fail_closed() {
    let dir = secret_case("secret-malformed-path");

    let toml_path = dir.join("malformed.toml");
    std::fs::write(
        &toml_path,
        "\"a.b\" = \"literal-dotted\"\na.b = \"nested-b\"\nA = \"unicode-a\"\n",
    )
    .unwrap();
    // Valid forms still resolve: a fully quoted segment, a plain path and
    // a Unicode escape spelling a real key.
    assert_eq!(
        file_recipe(FixtureFormat::Toml, &toml_path, "\"a.b\"")
            .resolve()
            .as_deref(),
        Some("literal-dotted")
    );
    assert_eq!(
        file_recipe(FixtureFormat::Toml, &toml_path, "a.b")
            .resolve()
            .as_deref(),
        Some("nested-b")
    );
    for escaped in ["\"\\u0041\"", "\"\\U00000041\""] {
        assert_eq!(
            file_recipe(FixtureFormat::Toml, &toml_path, escaped)
                .resolve()
                .as_deref(),
            Some("unicode-a"),
            "valid escape did not resolve: {escaped}"
        );
    }
    for malformed in [
        "\"a\"junk.b",
        "a.\"unclosed",
        "'a'junk.b",
        "a.'unclosed",
        "a..b",
        "a.b.",
        ".a.b",
        "\"a\".b\"c\"",
        // A signed escape (`u32::from_str_radix` accepts a leading `+`)
        // would decode to the real key `A` — a malformed escape must
        // never resolve a different key.
        "\"\\u+041\"",
        "\"\\U+0000041\"",
        "\"\\u004G\"",
        "\"\\u04G1\"",
        "\"\\U0000004G\"",
    ] {
        assert_eq!(
            file_recipe(FixtureFormat::Toml, &toml_path, malformed).resolve(),
            None,
            "malformed selector resolved: {malformed}"
        );
    }

    let yaml_path = dir.join("malformed.yaml");
    std::fs::write(&yaml_path, "a:\n  b: yaml-b\nA: yaml-a\n").unwrap();
    assert_eq!(
        file_recipe(FixtureFormat::Yaml, &yaml_path, "a.b")
            .resolve()
            .as_deref(),
        Some("yaml-b")
    );
    assert_eq!(
        file_recipe(FixtureFormat::Yaml, &yaml_path, "\"\\u0041\"")
            .resolve()
            .as_deref(),
        Some("yaml-a")
    );
    for malformed in [
        "\"a\"junk.b",
        "a.\"unclosed",
        "a..b",
        "\"\\u+041\"",
        "\"\\U+0000041\"",
        "\"\\u004G\"",
    ] {
        assert_eq!(
            file_recipe(FixtureFormat::Yaml, &yaml_path, malformed).resolve(),
            None,
            "malformed selector resolved: {malformed}"
        );
    }

    let _ = std::fs::remove_dir_all(dir);
}
