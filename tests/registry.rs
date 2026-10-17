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
            .insert(id.to_string(), cli_spec(id, "program").into());
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
                fallbacks: vec![],
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
        .insert("alias".to_owned(), cli_spec("different", "program").into());
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
            fallbacks: vec![],
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
        auth_key: None,
    };
    document.agents.insert("base".to_owned(), spec.into());

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

#[test]
fn binding_fallbacks_validate_against_profiles() {
    let mut document = document(&["a"], &[("p", "a")], &[]);
    document.bindings.insert(
        "role".to_owned(),
        awp::Binding::Full {
            profile: "p".to_owned(),
            prompt: None,
            fallbacks: vec!["missing".to_owned()],
        },
    );
    let error = Registry::from_document(document).unwrap_err();
    assert_eq!(error.code, "binding_profile_unknown");
}

#[test]
fn resolve_chain_orders_primary_then_fallbacks_with_shared_prompt() {
    let mut document = document(&["a", "b"], &[("p", "a"), ("q", "b")], &[]);
    document.profiles.insert(
        "q".to_owned(),
        ProfilePatch {
            agent: Some("b".to_owned()),
            instructions: Some("stand-in".to_owned()),
            ..Default::default()
        },
    );
    document.bindings.insert(
        "role".to_owned(),
        awp::Binding::Full {
            profile: "p".to_owned(),
            prompt: Some("the task".to_owned()),
            fallbacks: vec!["q".to_owned()],
        },
    );
    let registry = Registry::from_document(document).unwrap();
    let chain = registry.resolve_chain("role").unwrap();

    assert_eq!(chain.len(), 2);
    assert_eq!(chain[0].profile().id, "p");
    assert_eq!(chain[0].spec().id, "a");
    assert_eq!(chain[1].profile().id, "q");
    assert_eq!(chain[1].spec().id, "b");
    // The binding's standing prompt rides every candidate, not just the
    // primary — the role is identical while the agent differs.
    assert_eq!(
        chain[1].profile().instructions.as_deref(),
        Some("stand-in\n\nthe task")
    );
}

#[test]
fn short_form_bindings_resolve_to_single_candidate_chains() {
    let mut document = document(&["a"], &[("p", "a")], &[]);
    document
        .bindings
        .insert("role".to_owned(), awp::Binding::Profile("p".to_owned()));
    let registry = Registry::from_document(document).unwrap();
    let chain = registry.resolve_chain("role").unwrap();
    assert_eq!(chain.len(), 1);
    assert_eq!(chain[0].profile().id, "p");
}

#[test]
fn a_covering_window_leads_the_effective_chain() {
    let patch = ProfilePatch {
        agent: Some("a".to_owned()),
        model: Some("peak-model".to_owned()),
        models: Some(vec!["tail-model".to_owned()]),
        windows: Some(vec![
            awp::ModelWindow {
                timezone: "Asia/Shanghai".to_owned(),
                from: "00:00".to_owned(),
                to: "12:00".to_owned(),
                model: Some("offpeak-model".to_owned()),
                models: vec!["offpeak-alt".to_owned()],
            },
            awp::ModelWindow {
                timezone: "Asia/Shanghai".to_owned(),
                from: "12:00".to_owned(),
                to: "00:00".to_owned(),
                model: Some("peak-window-model".to_owned()),
                models: vec![],
            },
        ]),
        ..Default::default()
    };
    // 06:00 Asia/Shanghai == 22:00 UTC the day before; 16:00 == 08:00 UTC.
    let morning = "2026-10-17T22:00:00Z"
        .parse::<jiff::Timestamp>()
        .unwrap()
        .to_zoned(jiff::tz::TimeZone::UTC);
    let evening = "2026-10-18T08:00:00Z"
        .parse::<jiff::Timestamp>()
        .unwrap()
        .to_zoned(jiff::tz::TimeZone::UTC);

    let window = &patch.windows.clone().unwrap()[0];
    assert!(window.covers(window.minute_of_day(&morning).unwrap()));
    let second = &patch.windows.clone().unwrap()[1];
    assert!(second.covers(second.minute_of_day(&evening).unwrap()));

    let profile = awp::Profile::from_patch("p", patch).unwrap();
    assert_eq!(
        profile.effective_chain(&morning),
        vec!["offpeak-model", "offpeak-alt", "peak-model", "tail-model"]
    );
    assert_eq!(
        profile.effective_chain(&evening),
        vec!["peak-window-model", "peak-model", "tail-model"]
    );
}

#[test]
fn window_bounds_cover_normal_wrapped_and_whole_day_ranges() {
    let w = |from: &str, to: &str| awp::ModelWindow {
        timezone: "Asia/Shanghai".to_owned(),
        from: from.to_owned(),
        to: to.to_owned(),
        ..Default::default()
    };
    // Normal range: from inclusive, to exclusive.
    assert!(w("09:00", "17:00").covers(9 * 60));
    assert!(!w("09:00", "17:00").covers(17 * 60));
    // Wrapped range covers the late evening and the early morning.
    let night = w("22:00", "06:00");
    assert!(night.covers(23 * 60));
    assert!(night.covers(5 * 60));
    assert!(!night.covers(12 * 60));
    // Equal bounds cover the entire day.
    assert!(w("08:00", "08:00").covers(0));
    assert!(w("08:00", "08:00").covers(23 * 60 + 59));
    // Malformed bounds never match.
    assert!(!w("9:00", "17:00").covers(10 * 60));
    assert!(!w("25:00", "26:00").covers(10 * 60));
}

#[test]
fn retry_policy_parses_durations_and_computes_wakeup() {
    let patch: ProfilePatch = serde_json::from_value(serde_json::json!({
        "agent": "a",
        "retry": {"interval": "30m", "resetGrace": "PT1M"}
    }))
    .unwrap();
    let policy = patch.retry.clone().unwrap();
    assert_eq!(policy.interval.unwrap(), jiff::SignedDuration::from_mins(30));
    assert_eq!(policy.reset_grace.unwrap(), jiff::SignedDuration::from_mins(1));

    // Unknown reset window: poll `interval` after now.
    let now: jiff::Zoned = "2025-06-01T00:00:00Z[UTC]".parse().unwrap();
    let wake = policy.wakeup(&now, None).unwrap();
    assert_eq!(wake.timestamp().as_second(), now.timestamp().as_second() + 30 * 60);

    // A known reset wins and takes the declared grace — "reset in 3h"
    // wakes at 3h01m.
    let reset = now.checked_add(jiff::SignedDuration::from_hours(3)).unwrap();
    let wake = policy.wakeup(&now, Some(reset.clone())).unwrap();
    assert_eq!(
        wake.timestamp().as_second(),
        reset.timestamp().as_second() + 60
    );

    // No schedule for the situation at hand yields None — the caller
    // decides what unscheduled exhaustion means.
    let unscheduled = awp::RetryPolicy::default();
    assert!(unscheduled.wakeup(&now, None).is_none());
}
