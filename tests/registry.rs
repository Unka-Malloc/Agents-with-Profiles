//! Registry merge, overlay and resolution semantics.

use awp::{
    AgentSpec, CliLaneSpec, CommandSpec, EnvValue, LaneSpec, Profile, ProfilePatch, PromptChannel,
    Registry, RegistryDocument, StreamMode,
};
use serde_json::json;
use std::collections::BTreeMap;

fn cli_spec(id: &str, program: &str) -> AgentSpec {
    AgentSpec {
        id: id.to_owned(),
        client_name: None,
        lane: LaneSpec::Cli(Box::new(CliLaneSpec {
            args: vec!["{prompt}".to_owned()],
            prompt_channel: PromptChannel::ArgOnly,
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

fn document(
    agents: &[&str],
    profiles: &[(&str, &str)],
    bindings: &[(&str, &str)],
) -> RegistryDocument {
    let mut document = RegistryDocument::default();
    for id in agents {
        document
            .agents
            .insert(id.to_string(), cli_spec(id, "program"));
    }
    for (id, agent) in profiles {
        document.profiles.insert(
            id.to_string(),
            ProfilePatch {
                agent: Some(agent.to_string()),
                ..Default::default()
            },
        );
    }
    for (name, profile) in bindings {
        document.bindings.insert(
            name.to_string(),
            awp::Binding::Full {
                profile: profile.to_string(),
                prompt: None,
            },
        );
    }
    document
}

#[test]
fn profile_without_agent_fails_closed() {
    let result = Profile::from_patch("p", ProfilePatch::default());
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().code, "profile_agent_missing");
}

#[test]
fn unknown_agent_reference_rejected() {
    let mut document = RegistryDocument::default();
    document.profiles.insert(
        "ghost".to_owned(),
        ProfilePatch {
            agent: Some("nobody".to_owned()),
            ..Default::default()
        },
    );
    let error = Registry::from_document(document).unwrap_err();
    assert_eq!(error.code, "profile_agent_unknown");
}

#[test]
fn agent_key_must_match_spec_id() {
    let mut document = RegistryDocument::default();
    document
        .agents
        .insert("alias".to_owned(), cli_spec("different", "program"));
    let error = Registry::from_document(document).unwrap_err();
    assert_eq!(error.code, "registry_agent_id_mismatch");
}

#[test]
fn binding_to_unknown_profile_rejected() {
    let mut document = document(&["a"], &[("p", "a")], &[]);
    document.bindings.insert(
        "role".to_owned(),
        awp::Binding::Profile("missing".to_owned()),
    );
    let error = Registry::from_document(document).unwrap_err();
    assert_eq!(error.code, "binding_profile_unknown");
}

#[test]
fn overlay_replaces_scalars_merges_env() {
    let mut registry = Registry::from_document(document(&["a", "b"], &[("p", "a")], &[])).unwrap();

    let mut overlay = RegistryDocument::default();
    let mut env = BTreeMap::new();
    env.insert("K".to_owned(), EnvValue::Literal("v".to_owned()));
    overlay.profiles.insert(
        "p".to_owned(),
        ProfilePatch {
            model: Some("m2".to_owned()),
            effort: Some("high".to_owned()),
            env: Some(env),
            instructions: Some("be terse".to_owned()),
            ..Default::default()
        },
    );
    registry.apply(overlay).unwrap();

    let profile = registry.profile("p").unwrap();
    assert_eq!(profile.agent, "a"); // kept: patch did not set it
    assert_eq!(profile.model.as_deref(), Some("m2"));
    assert_eq!(profile.effort.as_deref(), Some("high"));
    assert_eq!(profile.instructions.as_deref(), Some("be terse"));
    assert_eq!(
        profile.env.get("K"),
        Some(&EnvValue::Literal("v".to_owned()))
    );
}

#[test]
fn failed_overlay_leaves_registry_untouched() {
    let mut registry = Registry::from_document(document(&["a"], &[("p", "a")], &[])).unwrap();

    let mut overlay = RegistryDocument::default();
    overlay.profiles.insert(
        "p".to_owned(),
        ProfilePatch {
            agent: Some("ghost".to_owned()),
            model: Some("sneaky".to_owned()),
            ..Default::default()
        },
    );
    assert!(registry.apply(overlay).is_err());

    // The failed apply is atomic: neither the agent rebind nor the model
    // change stuck.
    let profile = registry.profile("p").unwrap();
    assert_eq!(profile.agent, "a");
    assert_eq!(profile.model, None);
}

#[test]
fn binding_prompt_composes_after_profile_instructions() {
    let mut document = document(&["a"], &[("p", "a")], &[]);
    document.profiles.insert(
        "p".to_owned(),
        ProfilePatch {
            agent: Some("a".to_owned()),
            instructions: Some("persona".to_owned()),
            ..Default::default()
        },
    );
    document.bindings.insert(
        "review".to_owned(),
        awp::Binding::Full {
            profile: "p".to_owned(),
            prompt: Some("check the diff".to_owned()),
        },
    );
    let registry = Registry::from_document(document).unwrap();
    let selection = registry.select("review").unwrap();
    assert_eq!(
        selection.instructions.as_deref(),
        Some("persona\n\ncheck the diff")
    );
}

#[test]
fn resolve_materializes_env_and_mcp_into_spec() {
    let mut document = RegistryDocument::default();
    let mut spec = cli_spec("base", "prog");
    spec.lane = LaneSpec::Acp {
        mcp_servers: vec![json!({"name": "base-mcp"})],
        prefer_load: false,
    };
    document.agents.insert("base".to_owned(), spec);

    let mut env = BTreeMap::new();
    env.insert("EXTRA".to_owned(), EnvValue::Literal("1".to_owned()));
    document.profiles.insert(
        "p".to_owned(),
        ProfilePatch {
            agent: Some("base".to_owned()),
            env: Some(env),
            mcp_servers: Some(vec![json!({"name": "profile-mcp"})]),
            ..Default::default()
        },
    );
    let registry = Registry::from_document(document).unwrap();
    let agent = registry.resolve("p").unwrap();

    // Env lands in the command's sanitized policy `set`.
    match &agent.spec().command.env {
        awp::EnvPolicy::Sanitized { set, .. } => {
            assert_eq!(set.get("EXTRA"), Some(&EnvValue::Literal("1".to_owned())));
        }
        other => panic!("unexpected policy: {other:?}"),
    }
    // Profile MCP servers append after the spec's own.
    match &agent.spec().lane {
        LaneSpec::Acp { mcp_servers, .. } => {
            assert_eq!(mcp_servers.len(), 2);
            assert_eq!(mcp_servers[0]["name"], "base-mcp");
            assert_eq!(mcp_servers[1]["name"], "profile-mcp");
        }
        other => panic!("unexpected lane: {other:?}"),
    }
}

#[test]
fn principal_id_uses_profile_prefix() {
    let profile = Profile {
        id: "reviewer".to_owned(),
        agent: "a".to_owned(),
        ..Default::default()
    };
    assert_eq!(profile.principal_id(), "profile:reviewer");
}

#[test]
fn compose_instructions_drops_empty_layers() {
    assert_eq!(awp::compose_instructions(None, None), None);
    assert_eq!(
        awp::compose_instructions(Some("  "), Some("x")).as_deref(),
        Some("x")
    );
    assert_eq!(
        awp::compose_instructions(Some("a"), Some("b")).as_deref(),
        Some("a\n\nb")
    );
}
