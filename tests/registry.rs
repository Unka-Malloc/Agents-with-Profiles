//! Registry merge, overlay and resolution semantics.

use awp::{
    EnvValue, LaneSpec, Profile, ProfilePatch, PromptChannel, Registry, RegistryDocument,
    StateScope,
};
use serde_json::json;
use std::collections::BTreeMap;

mod common;
use common::cli_spec;

fn document(
    agents: &[&str],
    profiles: &[(&str, &str)],
    bindings: &[(&str, &str)],
) -> RegistryDocument {
    let mut document = RegistryDocument::default();
    for id in agents {
        document.agents.insert(
            id.to_string(),
            cli_spec(id, "program", &["{prompt}"], PromptChannel::ArgOnly).into(),
        );
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
            awp::Binding::Full(awp::BindingFull {
                profile: profile.to_string(),
                strict: false,
                prompt: None,
                availability: None,
                fallbacks: vec![],
            }),
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
    document.agents.insert(
        "alias".to_owned(),
        cli_spec(
            "different",
            "program",
            &["{prompt}"],
            PromptChannel::ArgOnly,
        )
        .into(),
    );
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
            state_scope: Some(StateScope::Project),
            ..Default::default()
        },
    );
    registry.apply(overlay).unwrap();

    let profile = registry.profile("p").unwrap();
    assert_eq!(profile.agent, "a"); // kept: patch did not set it
    assert_eq!(profile.model.as_deref(), Some("m2"));
    assert_eq!(profile.effort.as_deref(), Some("high"));
    assert_eq!(profile.instructions.as_deref(), Some("be terse"));
    assert_eq!(profile.state_scope, Some(StateScope::Project));
    assert_eq!(
        profile.env.get("K"),
        Some(&EnvValue::Literal("v".to_owned()))
    );
}

/// A layered overlay (base → organization → project) that re-materializes
/// an agent entry keeps the host declarations the regenerated layer
/// cannot name — the `Registry::apply` half of the custody incident where
/// a card rewrite stripped `toolsApprovalServers`.
#[test]
fn overlay_replacement_carries_host_declarations() {
    let rpc_agent = |servers: serde_json::Value, approval_servers: serde_json::Value| {
        json!({
            "id": "a",
            "lane": "rpc",
            "prompt": {"method": "session/prompt", "params": {"text": "{prompt}"}},
            "mcpServers": servers,
            "mcpConfig": {
                "flag": "-c",
                "root": "mcp_servers",
                "envVarsKey": "env_vars",
                "toolsApproval": {"key": "default_tools_approval_mode", "value": "approve"},
                "toolsApprovalServers": approval_servers
            },
            "command": {
                "program": "agent",
                "cwd": "/tmp/work",
                "env": {"inheritAll": {"set": {"HOST_FLAG": "1"}}}
            }
        })
    };
    let base: RegistryDocument = serde_json::from_value(json!({
        "agents": {
            "a": rpc_agent(
                json!([{"name": "project-automation", "command": "pa", "args": [], "env": []}]),
                json!(["project-automation"])
            )
        },
        "profiles": {"p": {"agent": "a"}}
    }))
    .unwrap();
    let mut registry = Registry::from_document(base).unwrap();

    // The project layer regenerates the same agent from card data alone:
    // no host server list, no approval servers, no env, no cwd.
    let overlay: RegistryDocument = serde_json::from_value(json!({
        "agents": {"a": rpc_agent(json!([]), json!([]))}
    }))
    .unwrap();
    registry.apply(overlay).unwrap();

    let agent = registry.agent("a").unwrap();
    let LaneSpec::Rpc(rpc) = &agent.lane else {
        panic!("rpc lane");
    };
    assert_eq!(rpc.mcp_servers.len(), 1);
    assert_eq!(rpc.mcp_servers[0]["name"], "project-automation");
    assert_eq!(
        rpc.mcp_config.as_ref().unwrap().tools_approval_servers,
        ["project-automation"]
    );
    let awp::EnvPolicy::InheritAll { set } = &agent.command.env else {
        panic!("fidelity policy survives");
    };
    assert_eq!(set["HOST_FLAG"], EnvValue::Literal("1".to_owned()));
    assert_eq!(
        agent.command.cwd.as_deref(),
        Some(std::path::Path::new("/tmp/work"))
    );

    // Declared fields on the overlay entry still win outright.
    let overlay: RegistryDocument = serde_json::from_value(json!({
        "agents": {"a": rpc_agent(
            json!([{"name": "other-tools", "command": "ot"}]),
            json!(["other-tools"])
        )}
    }))
    .unwrap();
    registry.apply(overlay).unwrap();
    let agent = registry.agent("a").unwrap();
    let LaneSpec::Rpc(rpc) = &agent.lane else {
        panic!("rpc lane");
    };
    assert_eq!(rpc.mcp_servers.len(), 1);
    assert_eq!(rpc.mcp_servers[0]["name"], "other-tools");
    assert_eq!(
        rpc.mcp_config.as_ref().unwrap().tools_approval_servers,
        ["other-tools"]
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
        awp::Binding::Full(awp::BindingFull {
            profile: "p".to_owned(),
            strict: false,
            prompt: Some("check the diff".to_owned()),
            availability: None,
            fallbacks: vec![],
        }),
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
    let mut spec = cli_spec("base", "prog", &["{prompt}"], PromptChannel::ArgOnly);
    spec.lane = LaneSpec::Acp {
        mcp_servers: vec![json!({"name": "base-mcp"})],
        prefer_load: false,
        auth_key: None,
        enforced_permission: None,
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

    // Env lands in the command's policy `set`.
    match &agent.spec().command.env {
        awp::EnvPolicy::InheritAll { set } | awp::EnvPolicy::Sanitized { set, .. } => {
            assert_eq!(set.get("EXTRA"), Some(&EnvValue::Literal("1".to_owned())));
        }
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
fn resolve_materializes_profile_mcp_into_the_rpc_lane() {
    let mut document = RegistryDocument::default();
    let mut spec = cli_spec("base", "prog", &["{prompt}"], PromptChannel::ArgOnly);
    spec.lane = LaneSpec::Rpc(Box::new(awp::RpcLaneSpec {
        session_open: Some(awp::RpcCallSpec {
            method: "session/new".to_owned(),
            params: json!({}),
        }),
        prompt: awp::RpcCallSpec {
            method: "session/prompt".to_owned(),
            params: json!({}),
        },
        mcp_servers: vec![json!({"name": "base-mcp"})],
        mcp_config: Some(awp::RpcMcpConfig {
            flag: "-c".to_owned(),
            root: "mcp_servers".to_owned(),
            env_vars_key: "env_vars".to_owned(),
            tools_approval: None,
            tools_approval_servers: vec![],
        }),
        ..Default::default()
    }));
    document.agents.insert("base".to_owned(), spec.into());
    document.profiles.insert(
        "p".to_owned(),
        ProfilePatch {
            agent: Some("base".to_owned()),
            mcp_servers: Some(vec![json!({"name": "profile-mcp"})]),
            ..Default::default()
        },
    );

    let registry = Registry::from_document(document).unwrap();
    let agent = registry.resolve("p").unwrap();
    match &agent.spec().lane {
        LaneSpec::Rpc(rpc) => {
            assert_eq!(rpc.mcp_servers.len(), 2);
            assert_eq!(rpc.mcp_servers[0]["name"], "base-mcp");
            assert_eq!(rpc.mcp_servers[1]["name"], "profile-mcp");
            assert_eq!(
                rpc.mcp_config.as_ref().map(|config| config.root.as_str()),
                Some("mcp_servers")
            );
        }
        other => panic!("unexpected lane: {other:?}"),
    }
}

#[test]
fn resolve_materializes_profile_mcp_into_the_cli_lane() {
    let mut document = RegistryDocument::default();
    let mut spec = cli_spec("base", "prog", &["{prompt}"], PromptChannel::ArgOnly);
    if let LaneSpec::Cli(lane) = &mut spec.lane {
        lane.mcp_servers = vec![json!({"name": "base-mcp"})];
    }
    document.agents.insert("base".to_owned(), spec.into());
    document.profiles.insert(
        "p".to_owned(),
        ProfilePatch {
            agent: Some("base".to_owned()),
            mcp_servers: Some(vec![json!({"name": "profile-mcp"})]),
            ..Default::default()
        },
    );

    let registry = Registry::from_document(document).unwrap();
    let agent = registry.resolve("p").unwrap();
    match &agent.spec().lane {
        LaneSpec::Cli(lane) => {
            // The cli lane cannot register MCP servers, but dropping them
            // here would hide the mismatch from the adapter's fail-closed
            // admission. The list must survive materialization.
            assert_eq!(lane.mcp_servers.len(), 2);
            assert_eq!(lane.mcp_servers[0]["name"], "base-mcp");
            assert_eq!(lane.mcp_servers[1]["name"], "profile-mcp");
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
        awp::Binding::Full(awp::BindingFull {
            profile: "p".to_owned(),
            strict: false,
            prompt: None,
            availability: None,
            fallbacks: vec![awp::Fallback::Id("missing".to_owned())],
        }),
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
        awp::Binding::Full(awp::BindingFull {
            profile: "p".to_owned(),
            strict: false,
            prompt: Some("the task".to_owned()),
            availability: None,
            fallbacks: vec![awp::Fallback::Id("q".to_owned())],
        }),
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
fn misspelled_full_binding_field_is_a_load_error() {
    // `promt` — not `prompt` — must not silently drop the standing prompt.
    let doc = r#"{
        "agents": {"a": {"id": "a", "lane": "cli",
            "command": {"program": "true"}}},
        "profiles": {"p": {"agent": "a"}},
        "bindings": {"role": {"profile": "p", "promt": "typo"}}
    }"#;
    assert!(awp::Registry::from_json(doc).is_err());
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
    assert_eq!(
        policy.interval.unwrap(),
        jiff::SignedDuration::from_mins(30)
    );
    assert_eq!(
        policy.reset_grace.unwrap(),
        jiff::SignedDuration::from_mins(1)
    );

    // Unknown reset window: poll `interval` after now.
    let now: jiff::Zoned = "2025-06-01T00:00:00Z[UTC]".parse().unwrap();
    let wake = policy.wakeup(&now, None).unwrap();
    assert_eq!(
        wake.timestamp().as_second(),
        now.timestamp().as_second() + 30 * 60
    );

    // A known reset wins and takes the declared grace — "reset in 3h"
    // wakes at 3h01m.
    let reset = now
        .checked_add(jiff::SignedDuration::from_hours(3))
        .unwrap();
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

#[test]
fn availability_bounds_filter_expired_candidates() {
    let mut document = document(&["a", "b", "c"], &[("p", "a"), ("q", "b"), ("r", "c")], &[]);
    document.bindings.insert(
        "role".to_owned(),
        awp::Binding::Full(awp::BindingFull {
            profile: "p".to_owned(),
            strict: false,
            prompt: None,
            availability: None,
            fallbacks: vec![
                awp::Fallback::Windowed(awp::FallbackEntry {
                    profile: "q".to_owned(),
                    availability: Some(awp::Availability {
                        not_before: None,
                        not_after: Some("2026-11-01".to_owned()),
                        ..Default::default()
                    }),
                }),
                awp::Fallback::Id("r".to_owned()),
            ],
        }),
    );
    let registry = Registry::from_document(document.clone()).unwrap();
    let id_of = |chain: &[awp::VirtualAgent]| {
        chain
            .iter()
            .map(|agent| agent.profile().id.clone())
            .collect::<Vec<_>>()
    };

    // Inside the promo window the lane serves; past `notAfter` it is
    // skipped outright and the unconditional fallback takes the slot.
    let during: jiff::Timestamp = "2026-06-01T12:00:00Z".parse().unwrap();
    assert_eq!(
        id_of(&registry.resolve_chain_at("role", &during).unwrap()),
        vec!["p", "q", "r"]
    );
    let after: jiff::Timestamp = "2026-11-02T00:00:00Z".parse().unwrap();
    assert_eq!(
        id_of(&registry.resolve_chain_at("role", &after).unwrap()),
        vec!["p", "r"]
    );
    // `notAfter` is exclusive: the first instant of that date is already out.
    let edge: jiff::Timestamp = "2026-11-01T00:00:00Z".parse().unwrap();
    assert_eq!(
        id_of(&registry.resolve_chain_at("role", &edge).unwrap()),
        vec!["p", "r"]
    );
    // An expired primary drops the whole chain — resolve_binding reports it.
    document.bindings.insert(
        "solo".to_owned(),
        awp::Binding::Full(awp::BindingFull {
            profile: "p".to_owned(),
            strict: false,
            prompt: None,
            availability: Some(awp::Availability {
                not_before: Some("2026-05-01".to_owned()),
                not_after: Some("2026-06-01".to_owned()),
                ..Default::default()
            }),
            fallbacks: vec![],
        }),
    );
    let registry = Registry::from_document(document).unwrap();
    let error = registry.resolve_binding("solo").err().unwrap();
    assert_eq!(error.code, "binding_chain_expired");
    // And before `notBefore` the lane does not exist either.
    let early: jiff::Timestamp = "2026-04-30T23:59:00Z".parse().unwrap();
    assert!(registry
        .resolve_chain_at("solo", &early)
        .unwrap()
        .is_empty());
}

#[test]
fn retired_profiles_drop_out_of_chains() {
    let mut document = document(&["a", "b"], &[("p", "a"), ("q", "b")], &[]);
    document.profiles.get_mut("p").unwrap().retires_at = Some("2026-10-23".to_owned());
    document.bindings.insert(
        "role".to_owned(),
        awp::Binding::Full(awp::BindingFull {
            profile: "p".to_owned(),
            strict: false,
            prompt: None,
            availability: None,
            fallbacks: vec![awp::Fallback::Id("q".to_owned())],
        }),
    );
    let registry = Registry::from_document(document).unwrap();
    let id_of = |chain: &[awp::VirtualAgent]| {
        chain
            .iter()
            .map(|agent| agent.profile().id.clone())
            .collect::<Vec<_>>()
    };

    // Before the instant the promo lane serves; past it the profile is
    // gone everywhere — the chain only yields the surviving candidate.
    let during: jiff::Timestamp = "2026-10-20T12:00:00Z".parse().unwrap();
    assert_eq!(
        id_of(&registry.resolve_chain_at("role", &during).unwrap()),
        vec!["p", "q"]
    );
    let after: jiff::Timestamp = "2026-10-23T00:00:00Z".parse().unwrap();
    assert_eq!(
        id_of(&registry.resolve_chain_at("role", &after).unwrap()),
        vec!["q"]
    );
    // `retiresAt` is exclusive like `notAfter`: the instant itself is out.
    let edge = after;
    assert_eq!(
        id_of(&registry.resolve_chain_at("role", &edge).unwrap()),
        vec!["q"]
    );
}

#[test]
fn malformed_retires_at_fails_the_document() {
    let mut document = document(&["a"], &[("p", "a")], &[]);
    document.profiles.get_mut("p").unwrap().retires_at = Some("not-a-date".to_owned());
    let error = Registry::from_document(document).unwrap_err();
    assert_eq!(error.code, "profile_retirement_malformed");
}

#[test]
fn retire_profiles_prunes_and_promotes() {
    let retired: std::collections::BTreeSet<String> =
        ["p".to_owned(), "gone".to_owned()].into_iter().collect();
    // A retired primary promotes its first surviving fallback; retired
    // fallbacks drop out of the middle of the chain too.
    let mut binding = awp::Binding::Full(awp::BindingFull {
        profile: "p".to_owned(),
        strict: false,
        prompt: None,
        availability: None,
        fallbacks: vec![
            awp::Fallback::Id("gone".to_owned()),
            awp::Fallback::Windowed(awp::FallbackEntry {
                profile: "q".to_owned(),
                availability: Some(awp::Availability {
                    not_after: Some("2030-01-01".to_owned()),
                    ..Default::default()
                }),
            }),
        ],
    });
    binding.retire_profiles(&retired);
    let awp::Binding::Full(full) = &binding else {
        panic!("chain survives through promotion")
    };
    assert_eq!(full.profile, "q");
    assert_eq!(full.fallbacks.len(), 0);
    assert_eq!(
        full.availability
            .as_ref()
            .and_then(|a| a.not_after.as_deref()),
        Some("2030-01-01")
    );
    // A fully spent chain becomes vacant — resolution then fails closed.
    let mut spent = awp::Binding::Profile("p".to_owned());
    spent.retire_profiles(&retired);
    assert!(matches!(spent, awp::Binding::None));
}

#[test]
fn malformed_availability_fails_the_document() {
    let mut document = document(&["a"], &[("p", "a"), ("q", "a")], &[]);
    document.bindings.insert(
        "role".to_owned(),
        awp::Binding::Full(awp::BindingFull {
            profile: "p".to_owned(),
            strict: false,
            prompt: None,
            availability: None,
            fallbacks: vec![awp::Fallback::Windowed(awp::FallbackEntry {
                profile: "q".to_owned(),
                availability: Some(awp::Availability {
                    not_before: None,
                    not_after: Some("not-a-date".to_owned()),
                    ..Default::default()
                }),
            })],
        }),
    );
    let error = Registry::from_document(document).unwrap_err();
    assert_eq!(error.code, "binding_availability_malformed");
}

#[test]
fn malformed_window_days_fail_the_document() {
    let mut document = document(&["a"], &[("p", "a")], &[]);
    document.bindings.insert(
        "role".to_owned(),
        awp::Binding::Full(awp::BindingFull {
            profile: "p".to_owned(),
            strict: false,
            prompt: None,
            availability: Some(awp::Availability {
                windows: vec![awp::AvailabilityWindow {
                    days: "mon,noday".to_owned(),
                    from: "00:00".to_owned(),
                    to: "00:00".to_owned(),
                    timezone: "UTC".to_owned(),
                }],
                ..Default::default()
            }),
            fallbacks: vec![],
        }),
    );
    let error = Registry::from_document(document).unwrap_err();
    assert_eq!(error.code, "binding_availability_malformed");
}

#[test]
fn non_hh_mm_window_bounds_fail_the_document() {
    for (from, to) in [("9:00", "17:00"), ("09:0", "17:00"), ("09:00", "24:01")] {
        let mut document = document(&["a"], &[("p", "a")], &[]);
        document.bindings.insert(
            "role".to_owned(),
            awp::Binding::Full(awp::BindingFull {
                profile: "p".to_owned(),
                strict: false,
                prompt: None,
                availability: Some(awp::Availability {
                    windows: vec![availability_window("*", from, to, "UTC")],
                    ..Default::default()
                }),
                fallbacks: vec![],
            }),
        );
        let error = Registry::from_document(document).unwrap_err();
        assert_eq!(
            error.code, "binding_availability_malformed",
            "{from}-{to} must be rejected at load"
        );
    }
}

#[test]
fn malformed_model_window_fails_the_document() {
    let mut document = document(&["a"], &[], &[]);
    document.profiles.insert(
        "p".to_owned(),
        ProfilePatch {
            agent: Some("a".to_owned()),
            windows: Some(vec![awp::ModelWindow {
                timezone: "UTC".to_owned(),
                from: "25:00".to_owned(),
                to: "06:00".to_owned(),
                model: Some("peak".to_owned()),
                models: vec![],
            }]),
            ..Default::default()
        },
    );
    let error = Registry::from_document(document).unwrap_err();
    assert_eq!(error.code, "profile_windows_malformed");
}

#[test]
fn select_honors_availability_and_retirement() {
    let mut doc = document(&["a", "b"], &[("p", "a"), ("q", "b")], &[]);
    // Primary "p" serves only Monday 00:00-12:00 UTC; "q" is unbounded.
    doc.bindings.insert(
        "role".to_owned(),
        awp::Binding::Full(awp::BindingFull {
            profile: "p".to_owned(),
            strict: false,
            prompt: None,
            availability: Some(awp::Availability {
                windows: vec![awp::AvailabilityWindow {
                    days: "mon".to_owned(),
                    from: "00:00".to_owned(),
                    to: "12:00".to_owned(),
                    timezone: "UTC".to_owned(),
                }],
                ..Default::default()
            }),
            fallbacks: vec![awp::Fallback::Id("q".to_owned())],
        }),
    );
    let registry = Registry::from_document(doc).unwrap();
    // Sunday noon UTC is outside the primary's window -> "q" answers.
    let sunday: jiff::Timestamp = "2026-10-25T12:00:00Z".parse().unwrap();
    assert_eq!(registry.select_at("role", &sunday).unwrap().profile.id, "q");
    // Monday 06:00 UTC is inside -> "p" answers.
    let monday: jiff::Timestamp = "2026-10-19T06:00:00Z".parse().unwrap();
    assert_eq!(registry.select_at("role", &monday).unwrap().profile.id, "p");

    // A retired primary also hands the slot to the next candidate.
    let mut retired = document(&["a", "b"], &[("p", "a"), ("q", "b")], &[]);
    retired.profiles.get_mut("p").unwrap().retires_at = Some("2020-01-01".to_owned());
    retired.bindings.insert(
        "role".to_owned(),
        awp::Binding::Full(awp::BindingFull {
            profile: "p".to_owned(),
            strict: false,
            prompt: None,
            availability: None,
            fallbacks: vec![awp::Fallback::Id("q".to_owned())],
        }),
    );
    let registry = Registry::from_document(retired).unwrap();
    assert_eq!(registry.select_at("role", &sunday).unwrap().profile.id, "q");

    // When every candidate is gated out, select fails rather than
    // answering with an unavailable profile.
    let mut expired = document(&["a"], &[("p", "a")], &[]);
    expired.bindings.insert(
        "role".to_owned(),
        awp::Binding::Full(awp::BindingFull {
            profile: "p".to_owned(),
            strict: false,
            prompt: None,
            availability: Some(awp::Availability {
                not_before: None,
                not_after: Some("2020-01-01".to_owned()),
                ..Default::default()
            }),
            fallbacks: vec![],
        }),
    );
    let registry = Registry::from_document(expired).unwrap();
    let error = registry.select_at("role", &sunday).err().unwrap();
    assert_eq!(error.code, "binding_chain_expired");
}

#[test]
fn windowed_fallback_deserializes_from_json() {
    let binding: awp::Binding = serde_json::from_str(
        r#"{
            "profile": "p",
            "fallbacks": [
                "bare-id",
                {"profile": "q", "availability": {"notAfter": "2026-11-01"}}
            ]
        }"#,
    )
    .unwrap();
    assert_eq!(binding.fallbacks()[0].profile(), "bare-id");
    let windowed = &binding.fallbacks()[1];
    assert_eq!(windowed.profile(), "q");
    assert_eq!(
        windowed.availability().and_then(|a| a.not_after.as_deref()),
        Some("2026-11-01")
    );
}

#[test]
fn recurring_schedule_windows_filter_chain_based_on_time_and_fallback() {
    let mut document = document(&["a", "b", "c"], &[("p", "a"), ("q", "b"), ("r", "c")], &[]);
    // Primary "p" is only available Mon-Fri 18:00 to 09:00 (Asia/Shanghai)
    document.bindings.insert(
        "general-worker".to_owned(),
        awp::Binding::Full(awp::BindingFull {
            profile: "p".to_owned(),
            strict: false,
            prompt: None,
            availability: Some(awp::Availability {
                windows: vec![awp::AvailabilityWindow {
                    days: "mon,tue,wed,thu,fri".to_owned(),
                    from: "18:00".to_owned(),
                    to: "09:00".to_owned(),
                    timezone: "Asia/Shanghai".to_owned(),
                }],
                ..Default::default()
            }),
            fallbacks: vec![
                awp::Fallback::Id("q".to_owned()),
                awp::Fallback::Id("r".to_owned()),
            ],
        }),
    );
    let registry = Registry::from_document(document).unwrap();
    let id_of = |chain: &[awp::VirtualAgent]| {
        chain
            .iter()
            .map(|agent| agent.profile().id.clone())
            .collect::<Vec<_>>()
    };

    // 1. Off-peak: Tuesday 20:00 Asia/Shanghai (12:00 UTC) -> "p" is available!
    // Result chain: ["p", "q", "r"]
    let offpeak: jiff::Timestamp = "2026-10-20T12:00:00Z".parse().unwrap();
    assert_eq!(
        id_of(
            &registry
                .resolve_chain_at("general-worker", &offpeak)
                .unwrap()
        ),
        vec!["p", "q", "r"]
    );

    // 2. Peak: Tuesday 10:00 Asia/Shanghai (02:00 UTC) -> "p" is skipped!
    // Result chain: ["q", "r"]
    let peak: jiff::Timestamp = "2026-10-20T02:00:00Z".parse().unwrap();
    assert_eq!(
        id_of(&registry.resolve_chain_at("general-worker", &peak).unwrap()),
        vec!["q", "r"]
    );

    // 3. Morning tail: Tuesday 08:00 Asia/Shanghai (00:00 UTC) belongs to
    // Monday's wrapped window — Monday is in mon-fri -> "p" is available.
    // Result chain: ["p", "q", "r"]
    let tail_in: jiff::Timestamp = "2026-10-20T00:00:00Z".parse().unwrap();
    assert_eq!(
        id_of(
            &registry
                .resolve_chain_at("general-worker", &tail_in)
                .unwrap()
        ),
        vec!["p", "q", "r"]
    );

    // 4. Morning tail: Monday 08:00 Asia/Shanghai (00:00 UTC) belongs to
    // Sunday's wrapped window — Sunday is not in mon-fri -> "p" is skipped.
    // Result chain: ["q", "r"]
    let tail_out: jiff::Timestamp = "2026-10-19T00:00:00Z".parse().unwrap();
    assert_eq!(
        id_of(
            &registry
                .resolve_chain_at("general-worker", &tail_out)
                .unwrap()
        ),
        vec!["q", "r"]
    );
}

// -------------------- availability windows: cooldowns & edges ------------

fn instant(raw: &str) -> jiff::Timestamp {
    raw.parse().expect("valid test timestamp")
}

fn availability_window(
    days: &str,
    from: &str,
    to: &str,
    timezone: &str,
) -> awp::AvailabilityWindow {
    awp::AvailabilityWindow {
        days: days.to_owned(),
        from: from.to_owned(),
        to: to.to_owned(),
        timezone: timezone.to_owned(),
    }
}

/// An availability bound that opens exactly when a candidate's cooldown
/// elapses — the shape a host writes after removing a refused candidate.
fn cooling_until(until: jiff::Timestamp) -> awp::Availability {
    awp::Availability {
        not_before: Some(until.to_string()),
        ..Default::default()
    }
}

fn ids_at(registry: &Registry, binding: &str, now: jiff::Timestamp) -> Vec<String> {
    registry
        .resolve_chain_at(binding, &now)
        .unwrap()
        .iter()
        .map(|agent| agent.profile().id.clone())
        .collect()
}

#[test]
fn cooldown_expiry_recovers_rate_limit_before_quota_exhaustion() {
    // A QuotaExhausted refusal pins a candidate for its long reset window
    // (one hour here); a RateLimited refusal only for the provider's
    // `retryAfterMs` (30s here). The availability gate must honour each
    // deadline exactly when it elapses — no cooldown outlives its own
    // deadline and no candidate returns before it.
    const RATE_LIMIT_RETRY_AFTER_MS: i64 = 30_000;
    const QUOTA_COOLDOWN_SECS: i64 = 60 * 60;

    let base = instant("2026-10-20T12:00:00Z");
    let rate_expiry = base
        .checked_add(jiff::SignedDuration::from_millis(RATE_LIMIT_RETRY_AFTER_MS))
        .unwrap();
    let quota_expiry = base
        .checked_add(jiff::SignedDuration::from_secs(QUOTA_COOLDOWN_SECS))
        .unwrap();

    let mut document = document(&["a", "b", "c"], &[("p", "a"), ("q", "b"), ("r", "c")], &[]);
    document.bindings.insert(
        "role".to_owned(),
        awp::Binding::Full(awp::BindingFull {
            profile: "p".to_owned(),
            strict: false,
            prompt: None,
            availability: Some(cooling_until(quota_expiry)),
            fallbacks: vec![
                awp::Fallback::Windowed(awp::FallbackEntry {
                    profile: "q".to_owned(),
                    availability: Some(cooling_until(rate_expiry)),
                }),
                awp::Fallback::Id("r".to_owned()),
            ],
        }),
    );
    let registry = Registry::from_document(document).unwrap();

    // Both candidates are cooling down: only the unconditional candidate
    // serves, and neither deadline has been reached early.
    assert_eq!(ids_at(&registry, "role", base), vec!["r"]);
    let just_before_rate = rate_expiry
        .checked_sub(jiff::SignedDuration::from_millis(1))
        .unwrap();
    assert_eq!(ids_at(&registry, "role", just_before_rate), vec!["r"]);

    // `notBefore` is inclusive: the short-cooldown candidate recovers on the
    // exact millisecond of its deadline while the long one stays out.
    assert_eq!(ids_at(&registry, "role", rate_expiry), vec!["q", "r"]);
    let just_before_quota = quota_expiry
        .checked_sub(jiff::SignedDuration::from_secs(1))
        .unwrap();
    assert_eq!(ids_at(&registry, "role", just_before_quota), vec!["q", "r"]);

    // The quota cooldown expires last and restores the primary; both
    // selection paths agree at that boundary.
    assert_eq!(ids_at(&registry, "role", quota_expiry), vec!["p", "q", "r"]);
    assert_eq!(
        registry
            .select_at("role", &quota_expiry)
            .unwrap()
            .profile
            .id,
        "p"
    );
}

#[test]
fn consecutive_rejections_degrade_the_chain_and_recover_in_order() {
    // Three refusals land in sequence: quota (one hour), rate limit (30s)
    // and overload (five minutes). Every rejection removes one more
    // candidate from the front; recovery happens only when that candidate's
    // own deadline elapses, so the chain rebuilds in deadline order rather
    // than resetting wholesale after the first success.
    let base = instant("2026-10-20T12:00:00Z");
    let rate_expiry = base
        .checked_add(jiff::SignedDuration::from_secs(30))
        .unwrap();
    let overload_expiry = base
        .checked_add(jiff::SignedDuration::from_mins(5))
        .unwrap();
    let quota_expiry = base
        .checked_add(jiff::SignedDuration::from_secs(60 * 60))
        .unwrap();

    let mut document = document(
        &["a", "b", "c", "d"],
        &[("p", "a"), ("q", "b"), ("r", "c"), ("s", "d")],
        &[],
    );
    document.bindings.insert(
        "role".to_owned(),
        awp::Binding::Full(awp::BindingFull {
            profile: "p".to_owned(),
            strict: false,
            prompt: None,
            availability: Some(cooling_until(quota_expiry)),
            fallbacks: vec![
                awp::Fallback::Windowed(awp::FallbackEntry {
                    profile: "q".to_owned(),
                    availability: Some(cooling_until(rate_expiry)),
                }),
                awp::Fallback::Windowed(awp::FallbackEntry {
                    profile: "r".to_owned(),
                    availability: Some(cooling_until(overload_expiry)),
                }),
                awp::Fallback::Id("s".to_owned()),
            ],
        }),
    );
    let registry = Registry::from_document(document).unwrap();

    // Every cooled-down candidate is out; the degradation lands on "s".
    assert_eq!(ids_at(&registry, "role", base), vec!["s"]);
    let before_rate = rate_expiry
        .checked_sub(jiff::SignedDuration::from_millis(1))
        .unwrap();
    assert_eq!(ids_at(&registry, "role", before_rate), vec!["s"]);

    // The rate-limited candidate returns first, ahead of the still-cooling
    // overload and quota candidates.
    assert_eq!(ids_at(&registry, "role", rate_expiry), vec!["q", "s"]);
    let before_overload = overload_expiry
        .checked_sub(jiff::SignedDuration::from_secs(1))
        .unwrap();
    assert_eq!(ids_at(&registry, "role", before_overload), vec!["q", "s"]);
    assert_eq!(
        ids_at(&registry, "role", overload_expiry),
        vec!["q", "r", "s"]
    );

    // The quota cooldown is the last to lift and restores the primary.
    let before_quota = quota_expiry
        .checked_sub(jiff::SignedDuration::from_secs(1))
        .unwrap();
    assert_eq!(ids_at(&registry, "role", before_quota), vec!["q", "r", "s"]);
    assert_eq!(
        ids_at(&registry, "role", quota_expiry),
        vec!["p", "q", "r", "s"]
    );
}

#[test]
fn recurring_window_reopens_after_its_cooldown_and_expires_on_the_clock() {
    // A candidate whose availability is a nightly window is the recurring
    // form of a cooldown: while the window is closed the fallback serves,
    // and the next opening restores the primary at the exact boundary.
    let mut document = document(&["a", "b"], &[("p", "a"), ("q", "b")], &[]);
    document.bindings.insert(
        "role".to_owned(),
        awp::Binding::Full(awp::BindingFull {
            profile: "p".to_owned(),
            strict: false,
            prompt: None,
            availability: Some(awp::Availability {
                windows: vec![availability_window("*", "22:00", "06:00", "UTC")],
                ..Default::default()
            }),
            fallbacks: vec![awp::Fallback::Id("q".to_owned())],
        }),
    );
    let registry = Registry::from_document(document).unwrap();

    let cases = [
        // Monday before the window opens.
        ("2026-10-19T21:59:59Z", "q"),
        // The opening instant is inclusive.
        ("2026-10-19T22:00:00Z", "p"),
        // Tuesday's tail still belongs to Monday's window.
        ("2026-10-20T05:59:59Z", "p"),
        // The closing instant is exclusive.
        ("2026-10-20T06:00:00Z", "q"),
        // Between windows the fallback serves.
        ("2026-10-20T21:59:59Z", "q"),
        // The next window opens on time.
        ("2026-10-20T22:00:00Z", "p"),
    ];
    for (raw, expected) in cases {
        let now = instant(raw);
        assert_eq!(
            registry.select_at("role", &now).unwrap().profile.id,
            expected,
            "{raw}"
        );
        if expected == "p" {
            assert_eq!(ids_at(&registry, "role", now), vec!["p", "q"], "{raw}");
        } else {
            assert_eq!(ids_at(&registry, "role", now), vec!["q"], "{raw}");
        }
    }
}

#[test]
fn availability_window_clock_boundaries_are_half_open_across_zones() {
    // Normal range: `from` inclusive, `to` exclusive, in the declared zone.
    let utc = availability_window("mon", "09:00", "17:00", "UTC");
    assert!(!utc.covers(&instant("2026-10-19T08:59:59Z")));
    assert!(utc.covers(&instant("2026-10-19T09:00:00Z")));
    assert!(utc.covers(&instant("2026-10-19T16:59:59Z")));
    assert!(!utc.covers(&instant("2026-10-19T17:00:00Z")));
    // 2026-10-19 is Monday; Tuesday at the same wall clock is outside the
    // day filter even though the time-of-day matches.
    assert!(!utc.covers(&instant("2026-10-20T09:00:00Z")));

    // The same wall clock in UTC+8 opens at 01:00Z, not 09:00Z.
    let shanghai = availability_window("mon", "09:00", "17:00", "Asia/Shanghai");
    assert!(!shanghai.covers(&instant("2026-10-19T00:59:59Z")));
    assert!(shanghai.covers(&instant("2026-10-19T01:00:00Z")));
    assert!(shanghai.covers(&instant("2026-10-19T08:59:59Z")));
    assert!(!shanghai.covers(&instant("2026-10-19T09:00:00Z")));

    // A wrapped window belongs to the day it opens on: Tuesday 00:30 is
    // Monday's tail, while Monday 00:30 belongs to Sunday and is out.
    let night = availability_window("mon", "22:00", "06:00", "UTC");
    assert!(!night.covers(&instant("2026-10-19T00:30:00Z")));
    assert!(night.covers(&instant("2026-10-20T00:30:00Z")));
    // Equal day/time bounds are the documented whole-day spelling.
    let whole_day = availability_window("*", "00:00", "00:00", "UTC");
    assert!(whole_day.covers(&instant("2026-10-20T00:00:00Z")));
    assert!(whole_day.covers(&instant("2026-10-20T12:00:00Z")));
    assert!(whole_day.covers(&instant("2026-10-20T23:59:59Z")));

    // `to` is strictly exclusive at the last minute: 23:59 is minute 1439,
    // not an alias for 24:00, so only the explicit 24:00 end bound covers
    // the final minute of the day.
    let last_minute = availability_window("*", "23:58", "23:59", "UTC");
    assert!(last_minute.covers(&instant("2026-10-20T23:58:00Z")));
    assert!(last_minute.covers(&instant("2026-10-20T23:58:59Z")));
    assert!(!last_minute.covers(&instant("2026-10-20T23:59:00Z")));
    let through_end = availability_window("*", "00:00", "24:00", "UTC");
    assert!(through_end.is_well_formed());
    assert!(through_end.covers(&instant("2026-10-20T23:59:59Z")));

    // Only exactly two ASCII digits per field parse: these are typos, not
    // alternate spellings, and a malformed window covers nothing.
    for (from, to) in [("9:00", "17:00"), ("09:0", "17:00"), ("09:00", " 17:00")] {
        let malformed = availability_window("*", from, to, "UTC");
        assert!(!malformed.is_well_formed(), "{from}-{to} must be malformed");
        assert!(!malformed.covers(&instant("2026-10-20T10:00:00Z")));
    }
}

#[test]
fn zero_span_bounds_cover_nothing_and_recover_on_their_deadline() {
    let now = instant("2026-10-20T12:00:00Z");

    // A zero-span availability interval is empty at every instant:
    // `notBefore` is inclusive and `notAfter` exclusive, so equal bounds
    // never intersect.
    let empty = awp::Availability {
        not_before: Some(now.to_string()),
        not_after: Some(now.to_string()),
        ..Default::default()
    };
    for probe in [
        now.checked_sub(jiff::SignedDuration::from_millis(1))
            .unwrap(),
        now,
        now.checked_add(jiff::SignedDuration::from_millis(1))
            .unwrap(),
    ] {
        assert!(!empty.covers(&probe), "{probe} must not be covered");
    }

    // A zero-length cooldown (`notBefore == now`) is already over: the
    // candidate is out one millisecond earlier and back exactly at now.
    let reopened = cooling_until(now);
    assert!(!reopened.covers(
        &now.checked_sub(jiff::SignedDuration::from_millis(1))
            .unwrap()
    ));
    assert!(reopened.covers(&now));

    // Conversely a deadline exactly at now is already closed.
    let closed = awp::Availability {
        not_after: Some(now.to_string()),
        ..Default::default()
    };
    assert!(closed.covers(
        &now.checked_sub(jiff::SignedDuration::from_millis(1))
            .unwrap()
    ));
    assert!(!closed.covers(&now));
}

#[test]
fn availability_windows_follow_dst_transitions() {
    // US spring-forward 2026-03-08: 02:00 EST jumps to 03:00 EDT. A window
    // spanning the gap stays open across it and still closes on local wall
    // clock, not on a fixed UTC offset.
    let spring = availability_window("sun", "01:30", "03:30", "America/New_York");
    assert!(!spring.covers(&instant("2026-03-08T06:29:59Z")));
    assert!(spring.covers(&instant("2026-03-08T06:30:00Z"))); // 01:30 EST
    assert!(spring.covers(&instant("2026-03-08T07:00:00Z"))); // 03:00 EDT
    assert!(!spring.covers(&instant("2026-03-08T07:30:00Z"))); // 03:30 EDT

    // US fall-back 2026-11-01: 02:00 EDT returns to 01:00 EST. Both passes
    // of the repeated hour are inside a 01:00-02:00 local window.
    let fall = availability_window("sun", "01:00", "02:00", "America/New_York");
    assert!(!fall.covers(&instant("2026-11-01T04:59:59Z"))); // 00:59:59 EDT
    assert!(fall.covers(&instant("2026-11-01T05:00:00Z"))); // 01:00 EDT
    assert!(fall.covers(&instant("2026-11-01T06:00:00Z"))); // 01:00 EST
    assert!(fall.covers(&instant("2026-11-01T06:59:59Z"))); // 01:59:59 EST
    assert!(!fall.covers(&instant("2026-11-01T07:00:00Z"))); // 02:00 EST
}

// ------------------------------- strict bindings -------------------------

fn strict_binding(profile: &str, fallbacks: Vec<awp::Fallback>) -> awp::Binding {
    awp::Binding::Full(awp::BindingFull {
        profile: profile.to_owned(),
        strict: true,
        prompt: None,
        availability: None,
        fallbacks,
    })
}

#[test]
fn strict_binding_seals_sealable_candidates() {
    let mut document = document(&["pi"], &[("p", "pi")], &[]);
    document
        .bindings
        .insert("role".to_owned(), strict_binding("p", vec![]));
    // The pi program name is what the recipe keys on.
    document.agents.insert(
        "pi".to_owned(),
        cli_spec("pi", "pi", &["--mode", "rpc"], PromptChannel::ArgOnly).into(),
    );
    let registry = Registry::from_document(document).unwrap();
    let chain = registry.resolve_chain("role").unwrap();
    let args = &chain[0].spec().command.args;
    for flag in [
        "--no-extensions",
        "--no-skills",
        "--no-prompt-templates",
        "--no-context-files",
        "builtin:mcp",
        "builtin:codemode",
        "builtin:tool-search",
    ] {
        assert!(args.contains(&flag.to_string()), "missing seal flag {flag}");
    }
}

#[test]
fn strict_binding_refuses_a_chain_nothing_can_seal() {
    let mut document = document(&["a", "b"], &[("p", "a"), ("q", "b")], &[]);
    document.bindings.insert(
        "role".to_owned(),
        strict_binding("p", vec![awp::Fallback::Id("q".to_owned())]),
    );
    let error = Registry::from_document(document)
        .unwrap()
        .resolve_chain("role")
        .err()
        .expect("a chain nothing can seal must refuse");
    assert_eq!(error.code, "agent_unsealable");
}

#[test]
fn strict_binding_degrades_to_a_sealable_fallback() {
    let mut document = document(&["a"], &[("p", "a"), ("q", "pi")], &[]);
    document.agents.insert(
        "pi".to_owned(),
        cli_spec("pi", "pi", &[], PromptChannel::ArgOnly).into(),
    );
    document.bindings.insert(
        "role".to_owned(),
        strict_binding("p", vec![awp::Fallback::Id("q".to_owned())]),
    );
    let registry = Registry::from_document(document).unwrap();
    let chain = registry.resolve_chain("role").unwrap();
    // The unsealable primary drops; the sealable fallback serves — never
    // an ambient primary.
    assert_eq!(chain.len(), 1);
    assert_eq!(chain[0].profile().id, "q");
    assert!(chain[0]
        .spec()
        .command
        .args
        .contains(&"--no-extensions".to_string()));
}

#[test]
fn strict_binding_skips_unsealable_fallbacks() {
    let mut document = document(&["pi", "b"], &[("p", "pi"), ("q", "b")], &[]);
    document.agents.insert(
        "pi".to_owned(),
        cli_spec("pi", "pi", &[], PromptChannel::ArgOnly).into(),
    );
    document.bindings.insert(
        "role".to_owned(),
        strict_binding("p", vec![awp::Fallback::Id("q".to_owned())]),
    );
    let registry = Registry::from_document(document).unwrap();
    let chain = registry.resolve_chain("role").unwrap();
    assert_eq!(chain.len(), 1);
    assert_eq!(chain[0].profile().id, "p");
}

#[test]
fn strict_binding_refuses_specs_reopening_ambient_channels() {
    let mut document = document(&["pi"], &[("p", "pi")], &[]);
    document.agents.insert(
        "pi".to_owned(),
        cli_spec(
            "pi",
            "pi",
            &["-e", "./ambient-extension.ts"],
            PromptChannel::ArgOnly,
        )
        .into(),
    );
    document
        .bindings
        .insert("role".to_owned(), strict_binding("p", vec![]));
    let error = Registry::from_document(document)
        .unwrap()
        .resolve_chain("role")
        .err()
        .expect("ambient-surface args must refuse a strict session");
    assert_eq!(error.code, "agent_unsealable");
}

#[test]
fn strict_binding_refuses_equals_form_ambient_args() {
    for args in [
        &["--skill=./ambient"][..],
        &["--prompt-template=./ambient"][..],
        &["--extension=./ambient.ts"][..],
        &["-e=./ambient.ts"][..],
    ] {
        let mut document = document(&["pi"], &[("p", "pi")], &[]);
        document.agents.insert(
            "pi".to_owned(),
            cli_spec("pi", "pi", args, PromptChannel::ArgOnly).into(),
        );
        document
            .bindings
            .insert("role".to_owned(), strict_binding("p", vec![]));
        let error = Registry::from_document(document)
            .unwrap()
            .resolve_chain("role")
            .err()
            .expect("equals-form ambient args must refuse a strict session");
        assert_eq!(error.code, "agent_unsealable", "{args:?}");
    }
    // A builtin path is the seal's own surface, not an ambient channel.
    let mut document = document(&["pi"], &[("p", "pi")], &[]);
    document.agents.insert(
        "pi".to_owned(),
        cli_spec(
            "pi",
            "pi",
            &["--extension=builtin:mcp"],
            PromptChannel::ArgOnly,
        )
        .into(),
    );
    document
        .bindings
        .insert("role".to_owned(), strict_binding("p", vec![]));
    Registry::from_document(document)
        .unwrap()
        .resolve_chain("role")
        .expect("a builtin extension is not an ambient channel");
}

#[test]
fn non_strict_bindings_keep_the_ambient_surface() {
    let mut document = document(&["pi"], &[("p", "pi")], &[]);
    document.agents.insert(
        "pi".to_owned(),
        cli_spec(
            "pi",
            "pi",
            &["-e", "./ambient-extension.ts"],
            PromptChannel::ArgOnly,
        )
        .into(),
    );
    document.bindings.insert(
        "role".to_owned(),
        awp::Binding::Full(awp::BindingFull {
            profile: "p".to_owned(),
            strict: false,
            prompt: None,
            availability: None,
            fallbacks: vec![],
        }),
    );
    let registry = Registry::from_document(document).unwrap();
    let chain = registry.resolve_chain("role").unwrap();
    let spec = chain[0].spec();
    let LaneSpec::Cli(lane) = &spec.lane else {
        panic!("cli spec expected");
    };
    assert_eq!(
        lane.args,
        vec!["-e".to_string(), "./ambient-extension.ts".to_string()]
    );
    assert!(spec.command.args.is_empty());
}

#[test]
fn strict_defaults_off_and_roundtrips() {
    let doc = r#"{
        "agents": {"a": {"id": "a", "lane": "cli",
            "command": {"program": "true"}}},
        "profiles": {"p": {"agent": "a"}},
        "bindings": {"plain": "p", "full": {"profile": "p"}}
    }"#;
    let registry = awp::Registry::from_json(doc).unwrap();
    assert!(!registry.binding("plain").unwrap().strict());
    assert!(!registry.binding("full").unwrap().strict());
    let strict_doc = r#"{
        "agents": {"a": {"id": "a", "lane": "cli",
            "command": {"program": "true"}}},
        "profiles": {"p": {"agent": "a"}},
        "bindings": {"role": {"profile": "p", "strict": true}}
    }"#;
    let registry = awp::Registry::from_json(strict_doc).unwrap();
    assert!(registry.binding("role").unwrap().strict());
}

#[test]
fn strict_select_returns_the_sealable_fallback() {
    let mut document = document(&["a"], &[("p", "a"), ("q", "pi")], &[]);
    document.agents.insert(
        "pi".to_owned(),
        cli_spec("pi", "pi", &[], PromptChannel::ArgOnly).into(),
    );
    document.bindings.insert(
        "role".to_owned(),
        strict_binding("p", vec![awp::Fallback::Id("q".to_owned())]),
    );
    let registry = Registry::from_document(document).unwrap();
    let now: jiff::Timestamp = "2026-10-20T12:00:00Z".parse().unwrap();
    let selection = registry.select_at("role", &now).unwrap();
    assert_eq!(selection.profile.id, "q");
}

#[test]
fn strict_select_refuses_when_nothing_can_seal() {
    let mut document = document(&["a", "b"], &[("p", "a"), ("q", "b")], &[]);
    document.bindings.insert(
        "role".to_owned(),
        strict_binding("p", vec![awp::Fallback::Id("q".to_owned())]),
    );
    let registry = Registry::from_document(document).unwrap();
    let now: jiff::Timestamp = "2026-10-20T12:00:00Z".parse().unwrap();
    let error = registry.select_at("role", &now).err().unwrap();
    assert_eq!(error.code, "agent_unsealable");
}

#[test]
fn non_strict_select_keeps_the_unsealable_candidate() {
    let document = document(&["a"], &[("p", "a")], &[("role", "p")]);
    let registry = Registry::from_document(document).unwrap();
    let now: jiff::Timestamp = "2026-10-20T12:00:00Z".parse().unwrap();
    assert_eq!(registry.select_at("role", &now).unwrap().profile.id, "p");
}

#[test]
fn select_reports_chain_expired_when_every_candidate_is_gated() {
    let mut document = document(&["a"], &[("p", "a")], &[]);
    document.bindings.insert(
        "role".to_owned(),
        awp::Binding::Full(awp::BindingFull {
            profile: "p".to_owned(),
            strict: false,
            prompt: None,
            availability: Some(awp::Availability {
                windows: vec![awp::AvailabilityWindow {
                    days: "mon".to_owned(),
                    from: "00:00".to_owned(),
                    to: "12:00".to_owned(),
                    timezone: "UTC".to_owned(),
                }],
                ..Default::default()
            }),
            fallbacks: vec![],
        }),
    );
    let registry = Registry::from_document(document).unwrap();
    // A Sunday is never inside a Monday window.
    let sunday: jiff::Timestamp = "2026-10-25T12:00:00Z".parse().unwrap();
    let error = registry.select_at("role", &sunday).err().unwrap();
    assert_eq!(error.code, "binding_chain_expired");
}

#[test]
fn from_patch_rejects_a_malformed_model_window() {
    let patch = ProfilePatch {
        agent: Some("a".to_owned()),
        windows: Some(vec![awp::ModelWindow {
            timezone: "UTC".to_owned(),
            from: "24:00".to_owned(),
            to: "06:00".to_owned(),
            model: Some("peak".to_owned()),
            models: vec![],
        }]),
        ..Default::default()
    };
    let error = Profile::from_patch("p", patch).err().unwrap();
    assert_eq!(error.code, "profile_windows_malformed");
    assert_eq!(error.stage, "profile/resolve");
}

#[test]
fn a_dead_strict_chain_reports_the_first_refusal_on_both_paths() {
    // Primary `a` has no sealing recipe; fallback `b` seals structurally
    // but reopens an ambient channel — the two fail for different reasons,
    // so the reported detail proves which candidate's refusal surfaced.
    let mut document = document(&["a"], &[("p", "a"), ("q", "b")], &[]);
    document.agents.insert(
        "b".to_owned(),
        cli_spec("b", "pi", &["-e", "./ambient.ts"], PromptChannel::ArgOnly).into(),
    );
    document.bindings.insert(
        "role".to_owned(),
        strict_binding("p", vec![awp::Fallback::Id("q".to_owned())]),
    );
    let registry = Registry::from_document(document).unwrap();
    let now: jiff::Timestamp = "2026-10-20T12:00:00Z".parse().unwrap();
    for error in [
        registry.select_at("role", &now).err().unwrap(),
        registry.resolve_chain_at("role", &now).err().unwrap(),
    ] {
        assert_eq!(error.code, "agent_unsealable");
        assert!(
            error.detail.as_deref().unwrap_or_default().contains("`a`"),
            "the first candidate's refusal must surface, got {:?}",
            error.detail
        );
    }
}
