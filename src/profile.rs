//! Behavior profiles layered on a base agent.
//!
//! A [`Profile`] is a named, operator-owned record that turns one registered
//! [`a4aa::AgentSpec`] into a distinct virtual agent: model and effort
//! defaults, persona instructions, a preferred working directory, extra
//! environment entries, ACP MCP servers, and host-advisory preferences.
//!
//! Two fields are required — `id` and `agent`; every other field is an
//! optional preference. Fields split into two classes:
//!
//! - **Materialized** — applied by the crate when a session opens or a turn
//!   is sent: `instructions`, `instruction_mode`, `model`, `effort`, `cwd`,
//!   `env`, `mcp_servers`.
//! - **Advisory** — carried verbatim for the host to consume (`permission`,
//!   `mode`, `context_window`, `required_capabilities`,
//!   `preferred_capabilities`, `skill_references`, `state_key`,
//!   `state_scope`, `extensions`). The crate never invents a delivery
//!   path for advisory fields; the host decides what its agent or
//!   dispatch layer supports.
//! - **Mechanisms** — `mechanisms` is the CAwP section: semantic
//!   declarations (hooks, rules, skills, commands, subagents, plugins,
//!   permissions, config-file MCP servers, native-profile selection) that
//!   `a4aa`'s adaptation layer translates into agent-native artifacts.
//!   Parsing and merging live here; the dialect knowledge lives in a4aa.
//!
//! [`ProfilePatch`] is the same shape with every field optional — the wire
//! and overlay form. Patches layer onto a profile (`None` keeps the value,
//! `Some` replaces it; `env` merges per key). This mirrors how operators
//! actually maintain profiles: a base definition plus scoped overrides.
//!
//! `revision` is an explicit, operator-maintained configuration version —
//! never inferred from a hash of private configuration.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

use a4aa::EnvValue;

/// Serialization domain for host-side session state.
///
/// Absence (the default) means the profile takes no state lease — sessions
/// run fully parallel. `"project"` serializes only within one project;
/// `"global"` single-flights the whole profile, for agents that genuinely
/// cannot serve concurrent sessions. The crate carries the marker; the
/// session host owns the lease.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum StateScope {
    /// Serialize only within one project.
    Project,
    /// Full single-flight across the machine.
    Global,
}

/// When persona instructions are composed into turn text.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum InstructionMode {
    /// Prepend instructions to the first prompt of each session-bound lane
    /// (`acp`, `rpc`). Stateless lanes (`cli`) always receive them — every
    /// turn there is a fresh process.
    #[default]
    FirstTurn,
    /// Prepend instructions to every prompt, on every lane.
    EveryTurn,
}

/// A peak/off-peak window for model selection: `timezone` is an IANA zone
/// name (`"Asia/Shanghai"`), `from`/`to` are `"HH:MM"` wall-clock bounds in
/// that zone. `from == to` covers the whole day; a range whose `from` is
/// later than `to` wraps midnight. An unparseable window never matches.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelWindow {
    pub timezone: String,
    pub from: String,
    pub to: String,
    /// Window head model: selected first while the window covers now.
    #[serde(default)]
    pub model: Option<String>,
    /// Window-local fallbacks behind `model`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub models: Vec<String>,
}

impl ModelWindow {
    /// Minutes after midnight in this window's timezone at `now`; `None`
    /// when the zone name does not resolve.
    pub fn minute_of_day(&self, now: &jiff::Zoned) -> Option<u16> {
        let zone = jiff::tz::TimeZone::get(&self.timezone).ok()?;
        let local = now.with_time_zone(zone).datetime().time();
        Some(local.hour() as u16 * 60 + local.minute() as u16)
    }

    /// Whether `minute_of_day` (minutes after midnight, zone-local) is
    /// covered. Normal ranges cover `from <= m < to`; wrapped ranges cover
    /// `m >= from || m < to`; equal bounds cover every minute.
    pub fn covers(&self, minute_of_day: u16) -> bool {
        let (Some(from), Some(to)) = (minute(&self.from), minute(&self.to)) else {
            return false;
        };
        if from == to {
            true
        } else if from < to {
            (from..to).contains(&minute_of_day)
        } else {
            minute_of_day >= from || minute_of_day < to
        }
    }

    /// A malformed window silently covers nothing and the declared chain
    /// answers instead — a typo then reads as "the discount never applies".
    /// `apply` rejects it at load rather than letting it degrade quietly.
    pub fn is_well_formed(&self) -> bool {
        jiff::tz::TimeZone::get(&self.timezone).is_ok()
            && minute(&self.from).is_some()
            && minute(&self.to).is_some()
    }

    /// The window's own chain: `model` first, then `models`, deduplicated.
    fn model_chain(&self) -> Vec<String> {
        let mut chain = Vec::new();
        if let Some(model) = &self.model {
            chain.push(model.clone());
        }
        for fallback in &self.models {
            if !chain.contains(fallback) {
                chain.push(fallback.clone());
            }
        }
        chain
    }
}

/// `"HH:MM"` as minutes after midnight; `None` for any other shape.
fn minute(hhmm: &str) -> Option<u16> {
    let (h, m) = hhmm.split_once(':')?;
    if h.len() != 2 || m.len() != 2 {
        return None;
    }
    let (h, m) = (h.parse::<u16>().ok()?, m.parse::<u16>().ok()?);
    (h < 24 && m < 60).then_some(h * 60 + m)
}

/// Quota-recovery policy: how an orchestrator schedules the next attempt
/// after every candidate's model chain exhausted. Declarative — nothing
/// sleeps here; the caller owns the clock and decides whether to poll or
/// wake.
///
/// Durations accept the friendly form (`"3h1m"`, `"45s"`) or ISO 8601
/// (`"PT3H1M"`); both serialize as ISO 8601.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RetryPolicy {
    /// Poll interval for an unknown reset window: the next attempt becomes
    /// eligible `interval` after the last failure.
    #[serde(
        default,
        with = "duration_opt",
        skip_serializing_if = "Option::is_none"
    )]
    pub interval: Option<jiff::SignedDuration>,
    /// Grace added to a known reset instant — provider-reported or
    /// caller-estimated — before resuming: a reset "3 hours from now"
    /// wakes at 3h + `resetGrace`.
    #[serde(
        default,
        with = "duration_opt",
        skip_serializing_if = "Option::is_none"
    )]
    pub reset_grace: Option<jiff::SignedDuration>,
}

/// A declared model provider: vendor identifier, base URL or endpoint,
/// optional auth secret recipe key, and catalog of supported models.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ProviderDecl {
    /// Provider kind or vendor name (e.g. "anthropic", "openai", "deepseek", "kimi", "custom").
    #[serde(default)]
    pub vendor: String,
    /// Base URL or custom API endpoint, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// Secret recipe key or host env variable carrying credentials for this provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth_secret: Option<String>,
    /// Catalog of model identifiers offered by this provider.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub models: Vec<String>,
}

impl RetryPolicy {
    /// The earliest instant the next attempt becomes eligible: a known
    /// `reset` wins and is shifted by `resetGrace`; otherwise `interval`
    /// after `now`. `None` means the policy carries no schedule for the
    /// situation at hand — the caller decides what that means.
    pub fn wakeup(&self, now: &jiff::Zoned, reset: Option<jiff::Zoned>) -> Option<jiff::Zoned> {
        if let Some(reset) = reset {
            let grace = self.reset_grace.unwrap_or_default();
            return reset.checked_add(grace).ok();
        }
        now.checked_add(self.interval?).ok()
    }
}

/// `Option<jiff::SignedDuration>` as a duration string: friendly or
/// ISO 8601 on the way in, ISO 8601 on the way out.
mod duration_opt {
    use jiff::SignedDuration;
    use serde::{Deserialize, Deserializer, Serializer};

    pub(super) fn serialize<S: Serializer>(
        duration: &Option<SignedDuration>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        match duration {
            Some(duration) => serializer.serialize_str(&duration.to_string()),
            None => serializer.serialize_none(),
        }
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<SignedDuration>, D::Error> {
        Option::<String>::deserialize(deserializer)?
            .map(|raw| raw.parse::<SignedDuration>())
            .transpose()
            .map_err(serde::de::Error::custom)
    }
}

/// A resolved profile: `id` + `agent` plus every optional preference.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Profile {
    /// Stable identity. The conventional principal id is `profile:<id>`.
    pub id: String,
    /// Display name.
    #[serde(default)]
    pub name: String,
    /// Identity of the backing agent in the registry that executes this
    /// profile's sessions.
    pub agent: String,
    /// Explicit operator-maintained configuration version.
    #[serde(default)]
    pub revision: Option<String>,
    /// Default model passed to lanes that accept one per turn.
    #[serde(default)]
    pub model: Option<String>,
    /// Ordered model fallbacks behind `model`: when the active model
    /// cannot serve a turn (exhausted quota, unavailable, rejected), the
    /// next entry is selected on the same session and the turn retried.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub models: Vec<String>,
    /// Ordered peak/off-peak windows: the first window covering the current
    /// wall-clock time in its named zone leads the effective model chain
    /// with its own `model`/`models`; the declared chain trails as
    /// fallback. Evaluated at session establishment.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub windows: Vec<ModelWindow>,
    /// Quota-recovery schedule for a fully exhausted attempt — see
    /// [`RetryPolicy`]. Advisory to the orchestrator driving the
    /// candidate chain.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry: Option<RetryPolicy>,
    /// Absolute retirement instant: once reached, the profile can no
    /// longer serve anywhere — a free-tier promo whose end the vendor
    /// published drops out of every chain that references it, and custody
    /// prunes the entry on its next read. RFC 3339 or `YYYY-MM-DD`
    /// (00:00 UTC); absence is unbounded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retires_at: Option<String>,
    /// Default reasoning effort passed to lanes that accept one per turn.
    #[serde(default)]
    pub effort: Option<String>,
    /// Agent-internal mode label (advisory).
    #[serde(default)]
    pub mode: Option<String>,
    /// Requested permission posture (advisory).
    #[serde(default)]
    pub permission: Option<String>,
    /// Persona or standing instructions composed into turn text.
    #[serde(default)]
    pub instructions: Option<String>,
    /// When instructions are composed into turns.
    #[serde(default)]
    pub instruction_mode: InstructionMode,
    /// Declared model providers and their model catalogs (advisory).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub providers: BTreeMap<String, ProviderDecl>,
    /// Preferred working directory for sessions that do not override it.
    #[serde(default)]
    pub cwd: Option<PathBuf>,
    /// Requested context window in tokens (advisory).
    #[serde(default)]
    pub context_window: Option<u64>,
    /// Extra environment entries merged into the agent's environment policy
    /// `set` at materialization — [`EnvValue`] indirections included.
    #[serde(default)]
    pub env: BTreeMap<String, EnvValue>,
    /// MCP servers appended to a lane's native registration surface
    /// (`acp` `mcpServers`, `serve` config document, `rpc` process/session
    /// registration). A lane that cannot register them refuses the list at
    /// open; the profile layer never drops it.
    #[serde(default)]
    pub mcp_servers: Vec<serde_json::Value>,
    /// Capabilities the host requires when selecting this profile (advisory).
    #[serde(default)]
    pub required_capabilities: Vec<String>,
    /// Capabilities the host prefers when selecting this profile (advisory).
    #[serde(default)]
    pub preferred_capabilities: Vec<String>,
    /// Host-defined skill or tool references (advisory).
    #[serde(default)]
    pub skill_references: Vec<String>,
    /// Host-side session-state namespace (advisory).
    #[serde(default)]
    pub state_key: Option<String>,
    /// Serialization domain of [`state_key`](Self::state_key). Absent
    /// from the document — the default — means no state lease. See
    /// [`StateScope`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state_scope: Option<StateScope>,
    /// CAwP section: semantic mechanism declarations handed to a4aa's
    /// adaptation layer for materialization into agent-native artifacts.
    #[serde(default)]
    pub mechanisms: Option<a4aa::MechanismSet>,
    /// Host-specific extras (avatars, sync manifests, UI hints). The crate
    /// never interprets them.
    #[serde(default)]
    pub extensions: BTreeMap<String, serde_json::Value>,
}

impl Profile {
    /// Conventional principal identity for hosts that surface a profile as a
    /// conversation participant: `profile:<id>`.
    pub fn principal_id(&self) -> String {
        format!("profile:{}", self.id)
    }

    /// The ordered model preference list: `model` first, then each
    /// `models` fallback, deduplicated. Empty when no model is declared.
    pub fn model_chain(&self) -> Vec<String> {
        let mut chain = Vec::new();
        if let Some(model) = &self.model {
            chain.push(model.clone());
        }
        for fallback in &self.models {
            if !chain.contains(fallback) {
                chain.push(fallback.clone());
            }
        }
        chain
    }

    /// The chain in effect at `now`: the first covering window leads with
    /// its own `model`/`models`; the declared chain trails, deduplicated.
    /// Equal to [`model_chain`](Self::model_chain) when no window matches.
    pub fn effective_chain(&self, now: &jiff::Zoned) -> Vec<String> {
        let mut chain = self
            .windows
            .iter()
            .find(|window| {
                window
                    .minute_of_day(now)
                    .is_some_and(|minute| window.covers(minute))
            })
            .map(ModelWindow::model_chain)
            .unwrap_or_default();
        for entry in self.model_chain() {
            if !chain.contains(&entry) {
                chain.push(entry);
            }
        }
        chain
    }

    /// Apply an overlay: `Some` replaces, `None` keeps; `env` merges per key.
    /// Merge a patch into this profile. Construction-time validation —
    /// including malformed model windows — lives in [`Profile::from_patch`]
    /// and the registry's document loader; callers mutating through this
    /// infallible merge are expected to apply a patch that already passed
    /// one of those gates.
    pub fn apply(&mut self, patch: ProfilePatch) {
        let ProfilePatch {
            name,
            agent,
            revision,
            model,
            models,
            windows,
            retry,
            retires_at,
            effort,
            mode,
            permission,
            instructions,
            instruction_mode,
            providers,
            cwd,
            context_window,
            env,
            mcp_servers,
            required_capabilities,
            preferred_capabilities,
            skill_references,
            state_key,
            state_scope,
            mechanisms,
            extensions,
        } = patch;
        if let Some(value) = name {
            self.name = value;
        }
        if let Some(value) = agent {
            self.agent = value;
        }
        if let Some(value) = revision {
            self.revision = Some(value);
        }
        if let Some(value) = model {
            self.model = Some(value);
        }
        if let Some(value) = models {
            self.models = value;
        }
        if let Some(value) = windows {
            self.windows = value;
        }
        if let Some(value) = retry {
            self.retry = Some(value);
        }
        if let Some(value) = retires_at {
            self.retires_at = Some(value);
        }
        if let Some(value) = effort {
            self.effort = Some(value);
        }
        if let Some(value) = mode {
            self.mode = Some(value);
        }
        if let Some(value) = permission {
            self.permission = Some(value);
        }
        if let Some(value) = instructions {
            self.instructions = Some(value);
        }
        if let Some(value) = instruction_mode {
            self.instruction_mode = value;
        }
        if let Some(value) = providers {
            self.providers.extend(value);
        }
        if let Some(value) = cwd {
            self.cwd = Some(value);
        }
        if let Some(value) = context_window {
            self.context_window = Some(value);
        }
        if let Some(env_patch) = env {
            self.env.extend(env_patch);
        }
        if let Some(value) = mcp_servers {
            self.mcp_servers = value;
        }
        if let Some(value) = required_capabilities {
            self.required_capabilities = value;
        }
        if let Some(value) = preferred_capabilities {
            self.preferred_capabilities = value;
        }
        if let Some(value) = skill_references {
            self.skill_references = value;
        }
        if let Some(value) = state_key {
            self.state_key = Some(value);
        }
        if let Some(value) = state_scope {
            self.state_scope = Some(value);
        }
        if let Some(patch_set) = mechanisms {
            match &mut self.mechanisms {
                Some(set) => set.merge(patch_set),
                None => self.mechanisms = Some(patch_set),
            }
        }
        if let Some(value) = extensions {
            self.extensions.extend(value);
        }
    }

    /// The patch form of this profile — every set field as `Some`.
    pub fn to_patch(&self) -> ProfilePatch {
        ProfilePatch {
            name: Some(self.name.clone()),
            agent: Some(self.agent.clone()),
            revision: self.revision.clone(),
            model: self.model.clone(),
            models: Some(self.models.clone()),
            windows: Some(self.windows.clone()),
            retry: self.retry.clone(),
            retires_at: self.retires_at.clone(),
            effort: self.effort.clone(),
            mode: self.mode.clone(),
            permission: self.permission.clone(),
            instructions: self.instructions.clone(),
            instruction_mode: Some(self.instruction_mode),
            providers: Some(self.providers.clone()),
            cwd: self.cwd.clone(),
            context_window: self.context_window,
            env: Some(self.env.clone()),
            mcp_servers: Some(self.mcp_servers.clone()),
            required_capabilities: Some(self.required_capabilities.clone()),
            preferred_capabilities: Some(self.preferred_capabilities.clone()),
            skill_references: Some(self.skill_references.clone()),
            state_key: self.state_key.clone(),
            state_scope: self.state_scope,
            mechanisms: self.mechanisms.clone(),
            extensions: Some(self.extensions.clone()),
        }
    }
}

/// The wire and overlay form of a profile: every field optional.
///
/// In a registry document a patch is keyed by the profile id, so the id
/// itself never appears inside it. A patch resolves into a [`Profile`]
/// through [`Profile::from_patch`] or by layering onto an existing profile.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ProfilePatch {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub agent: Option<String>,
    #[serde(default)]
    pub revision: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub models: Option<Vec<String>>,
    #[serde(default)]
    pub windows: Option<Vec<ModelWindow>>,
    #[serde(default)]
    pub retry: Option<RetryPolicy>,
    #[serde(default)]
    pub retires_at: Option<String>,
    #[serde(default)]
    pub effort: Option<String>,
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub permission: Option<String>,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub instruction_mode: Option<InstructionMode>,
    #[serde(default)]
    pub providers: Option<BTreeMap<String, ProviderDecl>>,
    #[serde(default)]
    pub cwd: Option<PathBuf>,
    #[serde(default)]
    pub context_window: Option<u64>,
    #[serde(default)]
    pub env: Option<BTreeMap<String, EnvValue>>,
    #[serde(default)]
    pub mcp_servers: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub required_capabilities: Option<Vec<String>>,
    #[serde(default)]
    pub preferred_capabilities: Option<Vec<String>>,
    #[serde(default)]
    pub skill_references: Option<Vec<String>>,
    #[serde(default)]
    pub state_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state_scope: Option<StateScope>,
    #[serde(default)]
    pub mechanisms: Option<a4aa::MechanismSet>,
    #[serde(default)]
    pub extensions: Option<BTreeMap<String, serde_json::Value>>,
}

impl Profile {
    /// Whether this profile's declared retirement instant has passed. A
    /// malformed `retiresAt` never retires — malformed documents are
    /// rejected at apply, so a surviving profile always parses here.
    pub fn is_retired(&self, now: &jiff::Timestamp) -> bool {
        self.retires_at
            .as_deref()
            .is_some_and(|raw| crate::parse_time_bound(raw).is_some_and(|b| *now >= b))
    }

    /// Resolve a patch into a profile under `id`. `agent` is required — a
    /// profile without a backing agent can never materialize. Malformed
    /// model windows are rejected here too, so a patch that bypasses the
    /// registry fails closed exactly like a registry document.
    pub fn from_patch(id: impl Into<String>, mut patch: ProfilePatch) -> a4aa::Result<Self> {
        let id = id.into();
        let agent = patch.agent.take().ok_or_else(|| {
            a4aa::AdapterError::permanent("profile_agent_missing", "profile/resolve")
        })?;
        if patch
            .windows
            .as_ref()
            .is_some_and(|windows| windows.iter().any(|w| !w.is_well_formed()))
        {
            return Err(a4aa::AdapterError::permanent(
                "profile_windows_malformed",
                "profile/resolve",
            )
            .with_detail(format!(
                "profile `{id}` declares a model window whose timezone or HH:MM bound is malformed"
            )));
        }
        let mut profile = Profile {
            id,
            agent,
            ..Default::default()
        };
        profile.apply(patch);
        Ok(profile)
    }
}
