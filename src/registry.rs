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
    /// Agent entries keyed by spec id.
    #[serde(default)]
    pub agents: BTreeMap<String, AgentEntry>,
    /// Profile patches keyed by profile id.
    #[serde(default)]
    pub profiles: BTreeMap<String, ProfilePatch>,
    /// Role/task bindings keyed by binding name.
    #[serde(default)]
    pub bindings: BTreeMap<String, Binding>,
    /// Host secret recipes keyed by the environment variable they feed.
    /// Custody (va) resolves these at spawn time and injects the values
    /// into the `a4aa serve` child environment, so spec `auth_key` /
    /// `hostEnv` references resolve without storing credentials here.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub secrets: BTreeMap<String, Vec<SecretRecipe>>,
}

/// One way to obtain a secret value: a host environment variable, or a
/// `{home}`-expanded TOML file key. Recipes are tried in order and the
/// first that yields a non-empty value wins.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SecretRecipe {
    /// `{"env": "NAME"}` — read the named host variable.
    Env { env: String },
    /// `{"toml": {"path": "...", "key": "..."}}` — read a `key = "value"`
    /// entry from a TOML file (basic `key = value` scanning, no deps).
    Toml { toml: TomlSecret },
}

/// File-key recipe target.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TomlSecret {
    /// TOML file path; `{home}` expands to the user's home directory.
    pub path: String,
    /// Key inside the file.
    pub key: String,
}

/// One agent entry: the primary spec plus ordered protocol alternates for
/// the same agent. When the primary lane cannot carry the work — the
/// transport fails to start, the handshake never completes, or the lane
/// dies mid-turn for reasons a model change cannot fix — the next
/// alternate spec drives the same agent over its own protocol. Every
/// alternate is a complete spec (own lane, own command line), declared
/// rather than merged.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AgentEntry {
    #[serde(flatten)]
    pub spec: AgentSpec,
    /// Ordered lane alternates tried after the primary spec.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub alternates: Vec<AgentSpec>,
}

impl AgentEntry {
    /// An entry with a single lane — the common case in documents and
    /// tests.
    pub fn single(spec: AgentSpec) -> Self {
        Self {
            spec,
            alternates: Vec::new(),
        }
    }

    /// Every lane variant in fallback order: primary first, alternates
    /// next — one entry per protocol the same agent speaks.
    pub fn lane_specs(&self) -> Vec<AgentSpec> {
        std::iter::once(self.spec.clone())
            .chain(self.alternates.iter().cloned())
            .collect()
    }
}

impl From<AgentSpec> for AgentEntry {
    fn from(spec: AgentSpec) -> Self {
        Self::single(spec)
    }
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
    Full(BindingFull),
    #[default]
    None,
}

/// The full binding payload — a named struct so `deny_unknown_fields`
/// applies: `{"profile":"p","promt":"x"}` is a load error, not a silently
/// dropped prompt.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct BindingFull {
    pub profile: String,
    #[serde(default)]
    pub prompt: Option<String>,
    /// When the primary candidate may serve at all — a promo lane whose
    /// free window the vendor published expires on its own; absence is
    /// unbounded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub availability: Option<Availability>,
    /// Ordered fallback candidates: a bare profile id stays unconditional,
    /// a `{"profile": ..., "availability": ...}` entry may carry the same
    /// bounds as the primary. When the bound profile's agent cannot serve
    /// (its whole model chain failed, the agent is absent, its session
    /// cannot be established, or its availability excludes now), the next
    /// profile — one that may back a different agent — takes over the role.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fallbacks: Vec<Fallback>,
}

/// One fallback candidate in a binding chain. The short `"profile-id"`
/// form deserializes as [`Fallback::Id`]; the full form adds availability.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Fallback {
    /// Bare profile reference.
    Id(String),
    /// Profile plus availability bounds.
    Windowed(FallbackEntry),
}

/// The full fallback payload — `deny_unknown_fields` keeps a typo'd key a
/// load error, same contract as [`BindingFull`].
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FallbackEntry {
    pub profile: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub availability: Option<Availability>,
}

impl Fallback {
    pub fn profile(&self) -> &str {
        match self {
            Fallback::Id(id) => id,
            Fallback::Windowed(entry) => &entry.profile,
        }
    }

    pub fn availability(&self) -> Option<&Availability> {
        match self {
            Fallback::Id(_) => None,
            Fallback::Windowed(entry) => entry.availability.as_ref(),
        }
    }
}

/// Time bounds on when a candidate may serve: `notBefore` inclusive,
/// `notAfter` exclusive — a promo "until 2026-11-01" stops being selectable
/// the moment that date arrives. Bounds accept RFC 3339 timestamps or a
/// bare `YYYY-MM-DD`, which reads as that day's 00:00 UTC. An absent bound
/// is unbounded: a free lane whose end the vendor never published stays
/// selectable until a request failure pushes it aside.
///
/// A malformed bound covers nothing: expiry must expire the lane, never
/// accidentally extend it. `apply` rejects the document at load for the
/// same reason — see [`Availability::is_well_formed`].
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Availability {
    /// Inclusive lower bound — candidates are not selectable before it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub not_before: Option<String>,
    /// Exclusive upper bound — the first instant the lane is no longer
    /// selectable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub not_after: Option<String>,
}

impl Availability {
    /// Whether `now` falls inside the bounds; a bound that cannot be parsed
    /// never covers.
    pub fn covers(&self, now: &jiff::Timestamp) -> bool {
        let lower = self
            .not_before
            .as_deref()
            .is_none_or(|raw| parse_bound(raw).is_some_and(|b| *now >= b));
        let upper = self
            .not_after
            .as_deref()
            .is_none_or(|raw| parse_bound(raw).is_some_and(|b| *now < b));
        lower && upper
    }

    /// Every declared bound parses — documents carrying a malformed bound
    /// are rejected at `apply` rather than expiring silently at resolve.
    pub fn is_well_formed(&self) -> bool {
        [&self.not_before, &self.not_after]
            .into_iter()
            .flatten()
            .all(|raw| parse_bound(raw).is_some())
    }
}

/// One availability bound: RFC 3339, or a bare date at 00:00 UTC.
fn parse_bound(raw: &str) -> Option<jiff::Timestamp> {
    if let Ok(timestamp) = raw.parse::<jiff::Timestamp>() {
        return Some(timestamp);
    }
    jiff::civil::Date::strptime("%Y-%m-%d", raw)
        .ok()?
        .to_zoned(jiff::tz::TimeZone::UTC)
        .ok()
        .map(|zoned| zoned.timestamp())
}

impl Binding {
    pub fn profile(&self) -> &str {
        match self {
            Binding::Profile(profile) => profile,
            Binding::Full(full) => &full.profile,
            Binding::None => "",
        }
    }

    pub fn prompt(&self) -> Option<&str> {
        match self {
            Binding::Full(full) => full.prompt.as_deref(),
            _ => None,
        }
    }

    /// Ordered fallback candidates, empty for the short forms.
    pub fn fallbacks(&self) -> &[Fallback] {
        match self {
            Binding::Full(full) => &full.fallbacks,
            _ => &[],
        }
    }

    /// The primary candidate's availability bounds, if the full form set them.
    pub fn availability(&self) -> Option<&Availability> {
        match self {
            Binding::Full(full) => full.availability.as_ref(),
            _ => None,
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
    agents: BTreeMap<String, AgentEntry>,
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
        for (id, entry) in document.agents {
            if entry.spec.id != id {
                return Err(AdapterError::permanent(
                    "registry_agent_id_mismatch",
                    "registry/apply",
                )
                .with_detail(format!(
                    "agent key `{id}` does not match spec id `{}`",
                    entry.spec.id
                )));
            }
            next.agents.insert(id, entry);
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
            let malformed = |what: &str| {
                AdapterError::permanent("binding_availability_malformed", "registry/apply")
                    .with_detail(format!(
                        "binding `{name}` {what} declares an availability bound that is neither RFC 3339 nor YYYY-MM-DD"
                    ))
            };
            if let Some(availability) = binding.availability() {
                if !availability.is_well_formed() {
                    return Err(malformed("the primary candidate"));
                }
            }
            for fallback in binding.fallbacks() {
                if !next.profiles.contains_key(fallback.profile()) {
                    return Err(AdapterError::permanent(
                        "binding_profile_unknown",
                        "registry/apply",
                    )
                    .with_detail(format!(
                        "binding `{name}` references unknown fallback profile `{}`",
                        fallback.profile()
                    )));
                }
                if let Some(availability) = fallback.availability() {
                    if !availability.is_well_formed() {
                        return Err(malformed("a fallback candidate"));
                    }
                }
            }
            next.bindings.insert(name, binding);
        }
        *self = next;
        Ok(())
    }

    /// The primary spec of one agent entry. Lane alternates are part of
    /// materialization, so callers resolving the agent for execution use
    /// [`Registry::resolve`]; this getter exists for inspection.
    pub fn agent(&self, id: &str) -> Option<&AgentSpec> {
        self.agents.get(id).map(|entry| &entry.spec)
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
        self.agents.values().map(|entry| &entry.spec)
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
        let entry = self.agents.get(&profile.agent).ok_or_else(|| {
            AdapterError::permanent("profile_agent_unknown", "registry/resolve").with_detail(
                format!(
                    "profile `{profile_id}` references unknown agent `{}`",
                    profile.agent
                ),
            )
        })?;
        Ok(crate::VirtualAgent::new(
            profile.clone(),
            entry.lane_specs(),
        ))
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
    /// standing prompt folded into the profile's instructions. Errors when
    /// every candidate's availability excludes now — a chain that expired
    /// is a lane that no longer exists, not a panic.
    pub fn resolve_binding(&self, binding_name: &str) -> a4aa::Result<crate::VirtualAgent> {
        self.resolve_chain(binding_name)?
            .into_iter()
            .next()
            .ok_or_else(|| {
                AdapterError::permanent("binding_chain_expired", "registry/resolve").with_detail(
                    format!("binding `{binding_name}` has no candidate available now"),
                )
            })
    }

    /// Resolve a binding to its ordered candidate chain: the bound profile
    /// first, then each declared fallback profile. Candidates whose
    /// availability excludes the current instant are skipped outright —
    /// an expired promo lane fails over before it can spend a request.
    /// Every surviving candidate gets the binding's standing prompt
    /// composed into its instructions, so the role stays identical while
    /// the backing agent may differ.
    pub fn resolve_chain(&self, binding_name: &str) -> a4aa::Result<Vec<crate::VirtualAgent>> {
        self.resolve_chain_at(binding_name, &jiff::Timestamp::now())
    }

    /// `resolve_chain` against an explicit instant — the deterministic
    /// form tests and schedulers use.
    pub fn resolve_chain_at(
        &self,
        binding_name: &str,
        now: &jiff::Timestamp,
    ) -> a4aa::Result<Vec<crate::VirtualAgent>> {
        let binding = self.binding(binding_name).ok_or_else(|| {
            AdapterError::permanent("binding_unknown", "registry/select")
                .with_detail(format!("unknown binding `{binding_name}`"))
        })?;
        let prompt = binding.prompt().map(str::to_owned);
        let mut candidates = Vec::new();
        let entries = std::iter::once((binding.profile(), binding.availability())).chain(
            binding
                .fallbacks()
                .iter()
                .map(|f| (f.profile(), f.availability())),
        );
        for (id, availability) in entries {
            if availability.is_some_and(|a| !a.covers(now)) {
                continue;
            }
            let mut profile = self
                .profiles
                .get(id)
                .expect("bindings validate against profiles at apply")
                .clone();
            profile.instructions =
                compose_instructions(profile.instructions.as_deref(), prompt.as_deref());
            let entry = self.agents.get(&profile.agent).ok_or_else(|| {
                AdapterError::permanent("profile_agent_unknown", "registry/resolve").with_detail(
                    format!(
                        "profile `{}` references unknown agent `{}`",
                        profile.id, profile.agent
                    ),
                )
            })?;
            candidates.push(crate::VirtualAgent::new(profile, entry.lane_specs()));
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
