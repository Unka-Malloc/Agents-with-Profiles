//! Canonical headless-agent presets.
//!
//! Operators should not hand-assemble argv. Each [`AgentPreset`] is a
//! verified program, spawn prefix, and required environment resolution —
//! real structs, not prompt text. Lane protocol details come from the
//! matching A4AA card: every declared native surface is materialized, in
//! the card's own capability order, as the primary spec plus its ordered
//! alternates.

use std::collections::BTreeMap;
use std::fmt;

use a4aa::{AgentSpec, EnvPolicy, EnvSource, EnvValue, LaneSpec};

use crate::profile::ProfilePatch;
use crate::registry::{AgentEntry, RegistryDocument};

/// How a required child environment variable is resolved at spawn.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RequiredEnv {
    /// Name injected into the child environment.
    pub name: &'static str,
    /// Host variable read at spawn (`{"hostEnv": ...}`).
    pub host_env: &'static str,
}

/// A verified headless-agent launch template.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AgentPreset {
    /// Preset and backing-agent id (`dsh`, `hermes`, …).
    pub id: &'static str,
    /// Display label.
    pub label: &'static str,
    /// Canonical binary name. A host that resolved an absolute path during
    /// detection substitutes it before spawn.
    pub program: &'static str,
    /// Verified spawn prefix (`hermes acp`, `pi --mode rpc`, …). Lane
    /// templates beyond this prefix live on the A4AA-materialized spec.
    pub args: &'static [&'static str],
    /// Child environment entries that must resolve from the host.
    pub required_env: &'static [RequiredEnv],
}

/// Preset ids for the headless agents currently driven through this layer.
/// `cursor` is included because A4AA already publishes its stream-json card.
pub static PRESETS: &[AgentPreset] = &[
    AgentPreset {
        id: "codex",
        label: "Codex",
        program: "codex",
        args: &["app-server", "--stdio"],
        required_env: &[],
    },
    AgentPreset {
        id: "cursor",
        label: "Cursor",
        program: "cursor-agent",
        args: &[],
        required_env: &[],
    },
    AgentPreset {
        id: "dsh",
        label: "DeepSeek Harness",
        program: "dsh",
        // Capability order ranks SDK above ACP, so the preset's primary
        // spawn is the first-party SDK stdio profile; the card keeps the
        // ACP profile as the ordered alternate.
        args: &["--profile", "sdk"],
        required_env: &[],
    },
    AgentPreset {
        id: "hermes",
        label: "Hermes Agent",
        program: "hermes",
        args: &["acp"],
        required_env: &[],
    },
    AgentPreset {
        id: "opencode",
        label: "OpenCode",
        program: "opencode",
        args: &["serve", "--port", "0"],
        required_env: &[],
    },
    AgentPreset {
        id: "pi",
        label: "Pi",
        program: "pi",
        args: &["--mode", "rpc"],
        required_env: &[],
    },
];

/// Look up a preset by id. Unknown names are `None` — never guessed.
pub fn lookup(id: &str) -> Option<&'static AgentPreset> {
    PRESETS.iter().find(|preset| preset.id == id)
}

/// Known preset ids, in declaration order.
pub fn ids() -> impl Iterator<Item = &'static str> {
    PRESETS.iter().map(|preset| preset.id)
}

/// Failure to materialize a preset into a registry document.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PresetError {
    Unknown { id: String },
    ProfileExists { id: String },
    EmptyName,
}

impl fmt::Display for PresetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unknown { id } => {
                let known: Vec<&str> = ids().collect();
                write!(f, "unknown preset '{id}'; known: {}", known.join(", "))
            }
            Self::ProfileExists { id } => {
                write!(f, "profile {id} already exists")
            }
            Self::EmptyName => f.write_str("profile name is empty"),
        }
    }
}

impl std::error::Error for PresetError {}

/// Agent entry plus the profile patch a `--preset` write stores.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PresetUnit {
    pub preset: &'static AgentPreset,
    pub agent: AgentEntry,
    pub profile: ProfilePatch,
}

/// Materialize one preset into an agent entry and a named profile patch.
pub fn instantiate(preset_id: &str, profile_id: &str) -> Result<PresetUnit, PresetError> {
    if profile_id.trim().is_empty() {
        return Err(PresetError::EmptyName);
    }
    let preset = lookup(preset_id).ok_or_else(|| PresetError::Unknown {
        id: preset_id.to_owned(),
    })?;
    Ok(PresetUnit {
        agent: agent_entry(preset),
        profile: profile_patch(preset, profile_id),
        preset,
    })
}

/// Insert the preset's verified agent and a new profile. An existing agent
/// with the same id is replaced so leftover argv cannot linger; an existing
/// profile is refused. Advanced custody stays on `va set --file`.
pub fn apply_preset(
    doc: &mut RegistryDocument,
    profile_id: &str,
    preset_id: &str,
) -> Result<&'static AgentPreset, PresetError> {
    if doc.profiles.contains_key(profile_id) {
        return Err(PresetError::ProfileExists {
            id: profile_id.to_owned(),
        });
    }
    let unit = instantiate(preset_id, profile_id)?;
    doc.agents.insert(unit.agent.spec.id.clone(), unit.agent);
    doc.profiles.insert(profile_id.to_owned(), unit.profile);
    Ok(unit.preset)
}

fn agent_entry(preset: &AgentPreset) -> AgentEntry {
    let agent_id = agent_id(preset);
    let mut entry = entry_from_card(preset.id, &agent_id);
    entry.spec.client_name = Some(preset.label.to_owned());
    apply_command(&mut entry.spec, preset);
    for alternate in &mut entry.alternates {
        alternate.client_name = Some(preset.label.to_owned());
        apply_program(alternate, preset);
        apply_env(alternate, preset.required_env);
    }
    entry
}

/// A preset's backing agent carries the auto-lane policy name, never the
/// bare binary or a lane-specific sibling id.
fn agent_id(preset: &AgentPreset) -> String {
    format!("{}-auto", preset.id)
}

fn entry_from_card(id: &str, agent_id: &str) -> AgentEntry {
    let card =
        a4aa_dialects::card(id).unwrap_or_else(|| panic!("preset id {id} must match an A4AA card"));
    // Presets run headless inside the workspace this layer dispatches —
    // the custody host's own workspace and environment grants, so
    // trusted-workspace argv (cursor's --trust) and InheritAll
    // materialize here rather than in the card's default spec.
    let mut specs = card.specs_trusting_workspace();
    for spec in &mut specs {
        spec.id = agent_id.to_owned();
        spec.command.env = EnvPolicy::InheritAll {
            set: BTreeMap::new(),
        };
    }
    let spec = specs.remove(0);
    AgentEntry {
        spec,
        alternates: specs,
    }
}

fn apply_command(spec: &mut AgentSpec, preset: &AgentPreset) {
    spec.command.program = preset.program.to_owned();
    spec.command.args = preset.args.iter().map(|arg| (*arg).to_owned()).collect();
    apply_env(spec, preset.required_env);
}

fn apply_program(spec: &mut AgentSpec, preset: &AgentPreset) {
    spec.command.program = preset.program.to_owned();
}

fn apply_env(spec: &mut AgentSpec, required: &[RequiredEnv]) {
    let mut set = match &spec.command.env {
        EnvPolicy::InheritAll { set } | EnvPolicy::Sanitized { set, .. } => set.clone(),
    };
    if let LaneSpec::Acp {
        auth_key: Some(EnvSource::HostEnv { host_env }),
        ..
    } = &spec.lane
    {
        set.entry(host_env.clone()).or_insert_with(|| {
            EnvValue::Source(EnvSource::HostEnv {
                host_env: host_env.clone(),
            })
        });
    }
    for req in required {
        set.insert(
            req.name.to_owned(),
            EnvValue::Source(EnvSource::HostEnv {
                host_env: req.host_env.to_owned(),
            }),
        );
    }
    spec.command.env = EnvPolicy::InheritAll { set };
}

fn profile_patch(preset: &AgentPreset, profile_id: &str) -> ProfilePatch {
    let mut env = BTreeMap::new();
    for req in preset.required_env {
        env.insert(
            req.name.to_owned(),
            EnvValue::Source(EnvSource::HostEnv {
                host_env: req.host_env.to_owned(),
            }),
        );
    }
    let mut extensions = BTreeMap::new();
    extensions.insert(
        "preset".to_owned(),
        serde_json::Value::String(preset.id.to_owned()),
    );
    ProfilePatch {
        agent: Some(agent_id(preset)),
        name: Some(profile_id.to_owned()),
        env: if env.is_empty() { None } else { Some(env) },
        extensions: Some(extensions),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Registry;

    fn args_of(entry: &AgentEntry) -> Vec<String> {
        entry.spec.command.args.clone()
    }

    #[test]
    fn ids_cover_headless_agents_in_use() {
        let ids: Vec<_> = ids().collect();
        assert_eq!(ids, ["codex", "cursor", "dsh", "hermes", "opencode", "pi"]);
    }

    #[test]
    fn unknown_preset_is_none() {
        assert!(lookup("no-such-agent").is_none());
        match instantiate("no-such-agent", "p") {
            Err(PresetError::Unknown { id }) => assert_eq!(id, "no-such-agent"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn dsh_is_sdk_primary_with_the_acp_alternate() {
        let unit = instantiate("dsh", "deepseek").unwrap();
        let specs = a4aa_dialects::card("dsh").unwrap().specs();
        assert_eq!(unit.agent.spec.command.program, "dsh");
        assert_eq!(args_of(&unit.agent), ["--profile", "sdk"]);
        assert!(matches!(unit.agent.spec.lane, LaneSpec::Rpc(_)));
        // Capability order ranks SDK above ACP. The card keeps the richer
        // ACP profile as the ordered alternate; dropping it would make the
        // auto-lane policy look SDK-only even though a second native lane
        // exists.
        assert_eq!(unit.agent.alternates.len(), specs.len() - 1);
        assert_eq!(unit.agent.alternates[0].command.args, ["acp"]);
        assert!(matches!(
            unit.agent.alternates[0].lane,
            LaneSpec::Acp { .. }
        ));
    }

    #[test]
    fn every_preset_copies_every_declared_card_lane() {
        for preset in PRESETS {
            let unit = instantiate(preset.id, "p").unwrap();
            let card = a4aa_dialects::card(preset.id).unwrap();
            let declared = card.specs();
            let materialized = unit.agent.lane_specs();
            assert_eq!(
                materialized.len(),
                declared.len(),
                "{} dropped a declared native lane",
                preset.id
            );
            for (index, spec) in materialized.iter().enumerate() {
                assert_eq!(
                    std::mem::discriminant(&spec.lane),
                    std::mem::discriminant(&declared[index].lane),
                    "{} lane {index} has the wrong protocol",
                    preset.id
                );
            }
        }
    }

    #[test]
    fn hermes_copies_acp_spawn() {
        let unit = instantiate("hermes", "h").unwrap();
        let card = a4aa_dialects::card("hermes").unwrap().spec();
        assert_eq!(unit.agent.spec.command.program, "hermes");
        assert_eq!(unit.agent.spec.command.args, card.command.args);
        assert_eq!(unit.agent.spec.command.args, ["acp"]);
        assert!(matches!(unit.agent.spec.lane, LaneSpec::Acp { .. }));
    }

    #[test]
    fn pi_uses_mode_rpc() {
        let unit = instantiate("pi", "p").unwrap();
        let card = a4aa_dialects::card("pi").unwrap().spec();
        assert_eq!(unit.agent.spec.command.program, "pi");
        assert_eq!(unit.agent.spec.command.args, card.command.args);
        assert_eq!(unit.agent.spec.command.args, ["--mode", "rpc"]);
        assert!(matches!(unit.agent.spec.lane, LaneSpec::Rpc(_)));
        assert!(!unit.agent.spec.command.args.iter().any(|a| a == "--effort"));
    }

    #[test]
    fn opencode_copies_serve_then_acp() {
        let unit = instantiate("opencode", "oc").unwrap();
        let specs = a4aa_dialects::card("opencode").unwrap().specs();
        assert_eq!(unit.agent.spec.command.program, "opencode");
        assert_eq!(unit.agent.spec.command.args, specs[0].command.args);
        assert_eq!(unit.agent.spec.command.args, ["serve", "--port", "0"]);
        assert!(matches!(unit.agent.spec.lane, LaneSpec::Serve { .. }));
        assert_eq!(unit.agent.alternates[0].command.args, specs[1].command.args);
        assert_eq!(unit.agent.alternates[0].command.args, ["acp"]);
    }

    #[test]
    fn codex_copies_app_server() {
        let unit = instantiate("codex", "c").unwrap();
        let specs = a4aa_dialects::card("codex").unwrap().specs();
        assert_eq!(unit.agent.spec.command.program, "codex");
        assert_eq!(unit.agent.spec.command.args, specs[0].command.args);
        assert_eq!(unit.agent.spec.command.args, ["app-server", "--stdio"]);
        assert!(matches!(unit.agent.spec.lane, LaneSpec::Rpc(_)));
        assert_eq!(unit.agent.alternates[0].command.args, specs[1].command.args);
        assert_eq!(unit.agent.alternates[0].command.args, ["exec"]);
    }

    #[test]
    fn cursor_is_stream_json_cli_without_acp() {
        let unit = instantiate("cursor", "cur").unwrap();
        assert_eq!(unit.agent.spec.command.program, "cursor-agent");
        assert!(unit.agent.spec.command.args.is_empty());
        let LaneSpec::Cli(cli) = &unit.agent.spec.lane else {
            panic!("cursor drives the cli lane");
        };
        assert!(cli.args.iter().any(|a| a == "--print"));
        assert!(cli.args.iter().any(|a| a == "stream-json"));
        assert!(!cli.args.iter().any(|a| a == "acp"));
        assert!(!unit.agent.spec.command.args.iter().any(|a| a == "acp"));
    }

    #[test]
    fn inherit_all_is_the_env_policy() {
        for preset in PRESETS {
            let unit = instantiate(preset.id, "p").unwrap();
            assert!(
                matches!(unit.agent.spec.command.env, EnvPolicy::InheritAll { .. }),
                "{}",
                preset.id
            );
        }
    }

    #[test]
    fn apply_writes_a_valid_registry_document() {
        let mut doc = RegistryDocument::default();
        apply_preset(&mut doc, "reviewer", "hermes").unwrap();
        assert!(doc.agents.contains_key("hermes-auto"));
        assert_eq!(
            doc.profiles["reviewer"].agent.as_deref(),
            Some("hermes-auto")
        );
        assert_eq!(
            doc.profiles["reviewer"].extensions.as_ref().unwrap()["preset"],
            "hermes"
        );
        Registry::from_document(doc.clone()).unwrap();

        assert!(matches!(
            apply_preset(&mut doc, "reviewer", "hermes"),
            Err(PresetError::ProfileExists { .. })
        ));
        apply_preset(&mut doc, "implementer", "hermes").unwrap();
        assert_eq!(doc.profiles.len(), 2);
        Registry::from_document(doc).unwrap();
    }

    #[test]
    fn every_preset_backs_its_profile_with_an_auto_agent() {
        for preset in PRESETS {
            let unit = instantiate(preset.id, "p").unwrap();
            assert_eq!(unit.agent.spec.id, format!("{}-auto", preset.id));
            assert_eq!(
                unit.profile.agent.as_deref(),
                Some(format!("{}-auto", preset.id).as_str())
            );
        }
    }

    #[test]
    fn every_preset_materializes_a_valid_document() {
        for preset in PRESETS {
            let mut doc = RegistryDocument::default();
            apply_preset(&mut doc, "p", preset.id).unwrap();
            Registry::from_document(doc).unwrap();
        }
    }
}
