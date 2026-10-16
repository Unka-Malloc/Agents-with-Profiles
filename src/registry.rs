//! Registries: the document form of agents, profiles and bindings.
//!
//! A [`RegistryDocument`] is plain data — the shared interchange format
//! every host reads (`awp.json`, a config section, a synced bundle):
//!
//! ```json
//! {
//!   "agents":   { "codex": { "id": "codex", "lane": "cli", "command": {"program": "codex"} } },
//!   "profiles": { "reviewer": { "agent": "codex", "model": "gpt-5", "instructions": "…" } },
//!   "bindings": { "review": { "profile": "reviewer", "prompt": "Review the diff." } }
//! }
//! ```
//!
//! Agents are [`a4aa::AgentSpec`]s keyed by their spec id — *how to reach
//! the agent*. Profiles are [`ProfilePatch`]es keyed by profile id — *how
//! the agent should behave*. Bindings name a profile plus a standing
//! prompt for one role or task slot.
//!
//! [`Registry`] resolves documents into a validated, layered store.
//! [`Registry::apply`] overlays a second document: agents and bindings are
//! replaced per key, profiles merge field-wise through
//! [`Profile::apply`]. This reproduces the scoped-override pattern
//! (base → organization → project) without any scope vocabulary in the
//! crate — the host decides how many layers it stacks.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use a4aa::{AdapterError, AgentSpec};

use crate::profile::{Profile, ProfilePatch};

/// One registry document: the wire, storage and overlay form.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RegistryDocument {
    /// Agent specs keyed by spec id.
    #[serde(default)]
    pub agents: BTreeMap<String, AgentSpec>,
    /// Profile patches keyed by profile id.
    #[serde(default)]
    pub profiles: BTreeMap<String, ProfilePatch>,
    /// Role/task bindings keyed by binding name.
    #[serde(default)]
    pub bindings: BTreeMap<String, Binding>,
}

/// A role or task slot: which profile serves it, plus a standing prompt
/// appended after the profile's own instructions when the binding is used.
///
/// The short form `"binding": "profile-id"` deserializes as
/// `{profile: "profile-id", prompt: None}`.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Binding {
    /// Bare profile reference.
    Profile(String),
    /// Profile plus an optional standing prompt and fallback profiles.
    Full {
        profile: String,
        #[serde(default)]
        prompt: Option<String>,
        /// Ordered fallback profile ids: when the bound profile's agent
        /// cannot serve (its whole model chain failed, the agent is absent
        /// or its session cannot be established), the next profile — one
        /// that may back a different agent — takes over the role.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        fallbacks: Vec<String>,
    },
    #[default]
    None,
}

impl Binding {
    pub fn profile(&self) -> &str {
        match self {
            Binding::Profile(profile) => profile,
            Binding::Full { profile, .. } => profile,
            Binding::None => "",
        }
    }

    pub fn prompt(&self) -> Option<&str> {
        match self {
            Binding::Full { prompt, .. } => prompt.as_deref(),
            _ => None,
        }
    }

    /// Ordered fallback profile ids, empty for the short forms.
    pub fn fallbacks(&self) -> &[String] {
        match self {
            Binding::Full { fallbacks, .. } => fallbacks,
            _ => &[],
        }
    }

    fn is_vacant(&self) -> bool {
        matches!(self, Binding::None) || self.profile().is_empty()
    }
}

/// What one binding resolution returns: the resolved profile and the
/// composed instructions (profile instructions first, then the binding's
/// standing prompt).
pub struct Selection {
    pub profile: Profile,
    /// Composed standing instructions for the bound role, if any.
    pub instructions: Option<String>,
}

/// A validated, layered store of agents, profiles and bindings.
#[derive(Clone, Debug, Default)]
pub struct Registry {
    agents: BTreeMap<String, AgentSpec>,
    profiles: BTreeMap<String, Profile>,
    bindings: BTreeMap<String, Binding>,
}

impl Registry {
    /// Build a registry from a document, validating references. A profile
    /// patch without `agent` and a binding pointing at an unknown profile
    /// fail closed — a registry that cannot materialize is a config error,
    /// not a runtime surprise.
    pub fn from_document(document: RegistryDocument) -> a4aa::Result<Self> {
        let mut registry = Registry::default();
        registry.apply(document)?;
        Ok(registry)
    }

    /// Load a document from JSON text.
    pub fn from_json(text: &str) -> a4aa::Result<Self> {
        let document: RegistryDocument = serde_json::from_str(text).map_err(|error| {
            AdapterError::permanent("registry_document_invalid", "registry/load")
                .with_detail(error.to_string())
        })?;
        Self::from_document(document)
    }

    /// Overlay one document onto this registry: agents and bindings replace
    /// per key, profiles merge field-wise. Atomic — a document that fails
    /// validation leaves the registry untouched.
    pub fn apply(&mut self, document: RegistryDocument) -> a4aa::Result<()> {
        let mut next = self.clone();
        for (id, spec) in document.agents {
            if spec.id != id {
                return Err(AdapterError::permanent(
                    "registry_agent_id_mismatch",
                    "registry/apply",
                )
                .with_detail(format!(
                    "agent key `{id}` does not match spec id `{}`",
                    spec.id
                )));
            }
            next.agents.insert(id, spec);
        }
        for (id, patch) in document.profiles {
            match next.profiles.get_mut(&id) {
                Some(profile) => profile.apply(patch),
                None => {
                    let profile = Profile::from_patch(&id, patch)?;
                    next.profiles.insert(id.clone(), profile);
                }
            }
            let profile = &next.profiles[&id];
            if !next.agents.contains_key(&profile.agent) {
                return Err(
                    AdapterError::permanent("profile_agent_unknown", "registry/apply").with_detail(
                        format!(
                            "profile `{id}` references unknown agent `{}`",
                            profile.agent
                        ),
                    ),
                );
            }
            if let Some(mechanisms) = &profile.mechanisms {
                mechanisms.validate().map_err(|e| {
                    AdapterError::permanent("profile_mechanisms_invalid", "registry/apply")
                        .with_detail(format!("profile `{id}`: {e}"))
                })?;
            }
        }
        for (name, binding) in document.bindings {
            if binding.is_vacant() {
                next.bindings.remove(&name);
                continue;
            }
            if !next.profiles.contains_key(binding.profile()) {
                return Err(
                    AdapterError::permanent("binding_profile_unknown", "registry/apply")
                        .with_detail(format!(
                            "binding `{name}` references unknown profile `{}`",
                            binding.profile()
                        )),
                );
            }
            for fallback in binding.fallbacks() {
                if !next.profiles.contains_key(fallback) {
                    return Err(AdapterError::permanent(
                        "binding_profile_unknown",
                        "registry/apply",
                    )
                    .with_detail(format!(
                        "binding `{name}` references unknown fallback profile `{fallback}`"
                    )));
                }
            }
            next.bindings.insert(name, binding);
        }
        *self = next;
        Ok(())
    }

    pub fn agent(&self, id: &str) -> Option<&AgentSpec> {
        self.agents.get(id)
    }

    pub fn profile(&self, id: &str) -> Option<&Profile> {
        self.profiles.get(id)
    }

    pub fn binding(&self, name: &str) -> Option<&Binding> {
        self.bindings
            .get(name)
            .filter(|binding| !binding.is_vacant())
    }

    pub fn agents(&self) -> impl ExactSizeIterator<Item = &AgentSpec> {
        self.agents.values()
    }

    pub fn profiles(&self) -> impl ExactSizeIterator<Item = &Profile> {
        self.profiles.values()
    }

    /// Bound role/task slots. Vacant bindings are dropped at `apply`, so
    /// every entry here resolves to a live profile reference.
    pub fn bindings(&self) -> impl ExactSizeIterator<Item = (&String, &Binding)> {
        self.bindings.iter()
    }

    /// Resolve a profile to the virtual agent it materializes.
    pub fn resolve(&self, profile_id: &str) -> a4aa::Result<crate::VirtualAgent> {
        let profile = self.profiles.get(profile_id).ok_or_else(|| {
            AdapterError::permanent("profile_unknown", "registry/resolve")
                .with_detail(format!("unknown profile `{profile_id}`"))
        })?;
        let spec = self.agents.get(&profile.agent).ok_or_else(|| {
            AdapterError::permanent("profile_agent_unknown", "registry/resolve").with_detail(
                format!(
                    "profile `{profile_id}` references unknown agent `{}`",
                    profile.agent
                ),
            )
        })?;
        Ok(crate::VirtualAgent::new(profile.clone(), spec.clone()))
    }

    /// Resolve a binding: the profile plus its composed standing
    /// instructions (profile instructions first, binding prompt second).
    pub fn select(&self, binding_name: &str) -> a4aa::Result<Selection> {
        let binding = self.binding(binding_name).ok_or_else(|| {
            AdapterError::permanent("binding_unknown", "registry/select")
                .with_detail(format!("unknown binding `{binding_name}`"))
        })?;
        let mut profile = self
            .profiles
            .get(binding.profile())
            .expect("bindings validate against profiles at apply")
            .clone();
        profile.instructions =
            compose_instructions(profile.instructions.as_deref(), binding.prompt());
        let instructions = profile.instructions.clone();
        Ok(Selection {
            profile,
            instructions,
        })
    }

    /// Resolve a binding straight to its virtual agent, with the binding's
    /// standing prompt folded into the profile's instructions.
    pub fn resolve_binding(&self, binding_name: &str) -> a4aa::Result<crate::VirtualAgent> {
        Ok(self
            .resolve_chain(binding_name)?
            .into_iter()
            .next()
            .expect("a resolved binding always has a primary candidate"))
    }

    /// Resolve a binding to its ordered candidate chain: the bound profile
    /// first, then each declared fallback profile. Every candidate gets the
    /// binding's standing prompt composed into its instructions, so the
    /// role stays identical while the backing agent may differ.
    pub fn resolve_chain(&self, binding_name: &str) -> a4aa::Result<Vec<crate::VirtualAgent>> {
        let binding = self.binding(binding_name).ok_or_else(|| {
            AdapterError::permanent("binding_unknown", "registry/select")
                .with_detail(format!("unknown binding `{binding_name}`"))
        })?;
        let prompt = binding.prompt().map(str::to_owned);
        let mut candidates = Vec::new();
        for id in
            std::iter::once(binding.profile()).chain(binding.fallbacks().iter().map(String::as_str))
        {
            let mut profile = self
                .profiles
                .get(id)
                .expect("bindings validate against profiles at apply")
                .clone();
            profile.instructions =
                compose_instructions(profile.instructions.as_deref(), prompt.as_deref());
            let spec = self.agents.get(&profile.agent).ok_or_else(|| {
                AdapterError::permanent("profile_agent_unknown", "registry/resolve").with_detail(
                    format!(
                        "profile `{}` references unknown agent `{}`",
                        profile.id, profile.agent
                    ),
                )
            })?;
            candidates.push(crate::VirtualAgent::new(profile, spec.clone()));
        }
        Ok(candidates)
    }
}

/// Compose two instruction layers into one preamble. Empty layers drop out;
/// two set layers join with a blank line.
pub fn compose_instructions(base: Option<&str>, extra: Option<&str>) -> Option<String> {
    let mut parts: Vec<&str> = Vec::new();
    for layer in [base, extra].into_iter().flatten() {
        let trimmed = layer.trim();
        if !trimmed.is_empty() {
            parts.push(trimmed);
        }
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join("\n\n"))
    }
}
