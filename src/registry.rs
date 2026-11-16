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
//! crate — the host decides how many layers it stacks. An agent entry a
//! later layer replaces still keeps the host declarations the card layer
//! cannot name ([`AgentEntry::carry_host_declarations_from`]), so stacking
//! a regenerated document never silently revokes approved MCP servers,
//! lane registrations or declared environment.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use a4aa::{AdapterError, AgentSpec, EnvPolicy, LaneSpec};

use crate::profile::{Profile, ProfilePatch, ProviderDecl};

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
    /// Global or shared model provider declarations and catalogs.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub providers: BTreeMap<String, ProviderDecl>,
}

/// One way to obtain a secret or configuration value: a host environment
/// variable, or a `{home}`-expanded file key (TOML, YAML, JSON).
/// Recipes are tried in order and the first that yields a non-empty value wins.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SecretRecipe {
    /// `{"env": "NAME"}` — read the named host variable.
    Env { env: String },
    /// `{"toml": {"path": "...", "key": "..."}}` — read an entry from a TOML file.
    Toml { toml: FileSecret },
    /// `{"yaml": {"path": "...", "key": "..."}}` — read an entry from a YAML file.
    Yaml { yaml: FileSecret },
    /// `{"json": {"path": "...", "key": "..."}}` — read an entry from a JSON file.
    Json { json: FileSecret },
}

/// Configuration file key target.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileSecret {
    /// File path; `{home}` expands to the user's home directory.
    pub path: String,
    /// Exact key or dotted key path (`section.sub.key`) inside the file.
    /// A leaf name is never matched outside the table the path declares,
    /// and a YAML sequence item has no dotted-path selector.
    pub key: String,
}

pub type TomlSecret = FileSecret;

impl SecretRecipe {
    /// Attempt to resolve this recipe to a non-empty string value.
    pub fn resolve(&self) -> Option<String> {
        match self {
            SecretRecipe::Env { env } => std::env::var(env).ok().filter(|v| !v.trim().is_empty()),
            SecretRecipe::Toml { toml } => {
                let path = expand_home(&toml.path);
                let text = std::fs::read_to_string(path).ok()?;
                extract_toml_key(&text, &toml.key)
            }
            SecretRecipe::Yaml { yaml } => {
                let path = expand_home(&yaml.path);
                let text = std::fs::read_to_string(path).ok()?;
                extract_yaml_key(&text, &yaml.key)
            }
            SecretRecipe::Json { json } => {
                let path = expand_home(&json.path);
                let text = std::fs::read_to_string(path).ok()?;
                extract_json_key(&text, &json.key)
            }
        }
    }
}

/// Resolve a list of secret recipes in declaration order: the first
/// recipe yielding a non-empty value wins.
pub fn resolve_secret(recipes: &[SecretRecipe]) -> Option<String> {
    recipes.iter().find_map(|r| r.resolve())
}

/// A secret resolver backed by a map of `SecretRecipe` lists.
#[derive(Clone, Debug, Default)]
pub struct RegistrySecretResolver {
    secrets: BTreeMap<String, Vec<SecretRecipe>>,
}

impl RegistrySecretResolver {
    pub fn new(secrets: BTreeMap<String, Vec<SecretRecipe>>) -> Self {
        Self { secrets }
    }
}

#[async_trait::async_trait]
impl a4aa::SecretResolver for RegistrySecretResolver {
    async fn resolve(&self, reference: &str) -> Option<String> {
        self.secrets
            .get(reference)
            .and_then(|recipes| resolve_secret(recipes))
            .or_else(|| {
                std::env::var(reference)
                    .ok()
                    .filter(|v| !v.trim().is_empty())
            })
    }
}

fn expand_home(raw: &str) -> String {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_default();
    raw.replace("{home}", &home)
}

/// Remove an inline comment without treating a `#` inside a quoted scalar
/// as the start of one. TOML starts a comment at any unquoted `#`; YAML
/// requires the `#` to be separated from the preceding character.
fn strip_inline_comment(line: &str, yaml: bool) -> &str {
    let mut quote: Option<char> = None;
    let mut escaped = false;
    for (index, character) in line.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        match quote {
            Some('"') => {
                if character == '\\' {
                    escaped = true;
                } else if character == '"' {
                    quote = None;
                }
            }
            Some('\'') => {
                if character == '\'' {
                    // YAML doubles a single quote to escape it; TOML literal
                    // strings have no escape and close at the first quote.
                    if yaml && line[index + 1..].starts_with('\'') {
                        escaped = true;
                    } else {
                        quote = None;
                    }
                }
            }
            Some(_) => {}
            None => match character {
                '"' => quote = Some('"'),
                '\'' => quote = Some('\''),
                '#' => {
                    let separated = index == 0
                        || line[..index]
                            .chars()
                            .next_back()
                            .is_some_and(char::is_whitespace);
                    if !yaml || separated {
                        return line[..index].trim_end();
                    }
                }
                _ => {}
            },
        }
    }
    line.trim_end()
}

/// Find `separator` outside quoted keys and values. For YAML a `:` only
/// separates a mapping key when whitespace or the end of the line follows.
fn find_separator(line: &str, separator: char, yaml: bool) -> Option<usize> {
    let mut quote: Option<char> = None;
    let mut escaped = false;
    for (index, character) in line.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        match quote {
            Some('"') => {
                if character == '\\' {
                    escaped = true;
                } else if character == '"' {
                    quote = None;
                }
            }
            Some('\'') => {
                if character == '\'' {
                    if yaml && line[index + 1..].starts_with('\'') {
                        escaped = true;
                    } else {
                        quote = None;
                    }
                }
            }
            Some(_) => {}
            None => {
                if character == '"' || character == '\'' {
                    quote = Some(character);
                } else if character == separator {
                    let rests_empty_or_spaced = line[index + 1..]
                        .chars()
                        .next()
                        .is_none_or(char::is_whitespace);
                    if !yaml || rests_empty_or_spaced {
                        return Some(index);
                    }
                }
            }
        }
    }
    None
}

/// Split a dotted path into segments, honoring quoted segments that may
/// contain a dot themselves.
fn dotted_path(raw: &str) -> Vec<String> {
    if raw.trim().is_empty() {
        return Vec::new();
    }
    let mut segments = Vec::new();
    let mut start = 0;
    let mut quote: Option<char> = None;
    let mut escaped = false;
    for (index, character) in raw.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        match quote {
            Some('"') => {
                if character == '\\' {
                    escaped = true;
                } else if character == '"' {
                    quote = None;
                }
            }
            Some('\'') => {
                if character == '\'' {
                    quote = None;
                }
            }
            Some(_) => {}
            None => match character {
                '"' | '\'' => quote = Some(character),
                '.' => {
                    segments.push(unquote_path_segment(&raw[start..index]));
                    start = index + 1;
                }
                _ => {}
            },
        }
    }
    segments.push(unquote_path_segment(&raw[start..]));
    segments
}

fn unquote_path_segment(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.starts_with('"') {
        parse_double_quoted(trimmed).unwrap_or_default()
    } else if let Some(literal) = trimmed.strip_prefix('\'') {
        literal
            .find('\'')
            .map(|end| literal[..end].to_owned())
            .unwrap_or_default()
    } else {
        trimmed.to_owned()
    }
}

/// Decode a double-quoted string with its common escapes. The closing
/// quote ends the scalar; anything after it is syntax the caller ignores.
fn parse_double_quoted(raw: &str) -> Option<String> {
    let mut value = String::new();
    let mut characters = raw.strip_prefix('"')?.chars();
    while let Some(character) = characters.next() {
        match character {
            '"' => return Some(value),
            '\\' => match characters.next()? {
                'b' => value.push('\u{0008}'),
                't' => value.push('\t'),
                'n' => value.push('\n'),
                'f' => value.push('\u{000c}'),
                'r' => value.push('\r'),
                '"' => value.push('"'),
                '\\' => value.push('\\'),
                escape @ ('u' | 'U') => {
                    let width = if escape == 'u' { 4 } else { 8 };
                    let digits: String = characters.by_ref().take(width).collect();
                    let decoded = u32::from_str_radix(&digits, 16)
                        .ok()
                        .and_then(char::from_u32)?;
                    value.push(decoded);
                }
                other => {
                    value.push('\\');
                    value.push(other);
                }
            },
            _ => value.push(character),
        }
    }
    None
}

fn parse_toml_value(raw: &str) -> Option<String> {
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }
    let value = if raw.starts_with('"') {
        parse_double_quoted(raw)?
    } else if let Some(literal) = raw.strip_prefix('\'') {
        let end = literal.find('\'')?;
        literal[..end].to_owned()
    } else {
        raw.to_owned()
    };
    (!value.trim().is_empty()).then_some(value)
}

fn parse_yaml_value(raw: &str) -> Option<String> {
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }
    let value = if raw.starts_with('"') {
        parse_double_quoted(raw)?
    } else if raw.starts_with('\'') {
        let mut value = String::new();
        let mut characters = raw.strip_prefix('\'')?.chars();
        loop {
            match characters.next()? {
                '\'' if characters.as_str().starts_with('\'') => {
                    characters.next();
                    value.push('\'');
                }
                '\'' => break value,
                character => value.push(character),
            }
        }
    } else {
        raw.to_owned()
    };
    (!value.trim().is_empty()).then_some(value)
}

fn extract_json_key(text: &str, key: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(text).ok()?;
    let mut current = &value;
    for part in key.split('.') {
        current = match current {
            serde_json::Value::Object(map) => map.get(part)?,
            serde_json::Value::Array(arr) => {
                let idx: usize = part.parse().ok()?;
                arr.get(idx)?
            }
            _ => return None,
        };
    }
    match current {
        serde_json::Value::String(s) => Some(s.clone()).filter(|s| !s.trim().is_empty()),
        serde_json::Value::Number(n) => Some(n.to_string()),
        serde_json::Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

fn extract_toml_key(text: &str, key: &str) -> Option<String> {
    let target = dotted_path(key);
    if target.is_empty() {
        return None;
    }
    let mut table: Vec<String> = Vec::new();
    for raw in text.lines() {
        let line = strip_inline_comment(raw, false).trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            let name = line.trim_start_matches('[').trim_end_matches(']').trim();
            table = dotted_path(name);
            continue;
        }
        let Some(equals) = find_separator(line, '=', false) else {
            continue;
        };
        let mut path = table.clone();
        path.extend(dotted_path(&line[..equals]));
        if path == target {
            return parse_toml_value(&line[equals + 1..]);
        }
    }
    None
}

fn extract_yaml_key(text: &str, key: &str) -> Option<String> {
    let target = dotted_path(key);
    if target.is_empty() {
        return None;
    }
    let mut stack: Vec<(usize, String)> = Vec::new();
    // Indentation of the block sequence currently being scanned. A sequence
    // index cannot be named by the dotted mapping path this contract
    // supports, so nothing inside an item may be assembled into that path.
    let mut sequence_indent: Option<usize> = None;
    for raw in text.lines() {
        let line = strip_inline_comment(raw, true);
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed == "---" || trimmed == "..." {
            continue;
        }
        let indent = line.len() - line.trim_start().len();
        if let Some(sequence) = sequence_indent {
            if indent > sequence {
                // A descendant of a sequence item, not of the mapping key
                // that owns the sequence.
                continue;
            }
            sequence_indent = None;
        }
        if trimmed == "-" || trimmed.starts_with("- ") {
            while stack
                .last()
                .is_some_and(|(parent_indent, _)| *parent_indent >= indent)
            {
                stack.pop();
            }
            sequence_indent = Some(indent);
            continue;
        }
        while stack
            .last()
            .is_some_and(|(parent_indent, _)| *parent_indent >= indent)
        {
            stack.pop();
        }
        let Some(colon) = find_separator(trimmed, ':', true) else {
            continue;
        };
        let segment = unquote_path_segment(&trimmed[..colon]);
        if segment.is_empty() {
            continue;
        }
        let value = &trimmed[colon + 1..];
        let mut path: Vec<String> = stack.iter().map(|(_, key)| key.clone()).collect();
        path.push(segment.clone());
        if path == target {
            if let Some(value) = parse_yaml_value(value) {
                return Some(value);
            }
        }
        if value.trim().is_empty() {
            stack.push((indent, segment));
        }
    }
    None
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

    /// Carry host declarations forward when this entry replaces `prior`.
    ///
    /// Card materialization and presets emit the mechanism surface only —
    /// the fields only the host can name stay blank: approved MCP server
    /// names (`mcpConfig.toolsApprovalServers`), host-registered MCP
    /// servers on the lane, declared env overrides, a working directory,
    /// an output bound, an enforced permission preset. A rewrite that
    /// regenerates the entry from card data alone must not silently
    /// revoke them. Lane variants pair by lane kind, not position, so a
    /// card that gains or reorders alternates still lands its declaration
    /// on the matching spec; fields the new spec declares itself always
    /// win.
    pub fn carry_host_declarations_from(&mut self, prior: &AgentEntry) {
        let mut priors: Vec<&AgentSpec> = std::iter::once(&prior.spec)
            .chain(prior.alternates.iter())
            .collect();
        for spec in std::iter::once(&mut self.spec).chain(self.alternates.iter_mut()) {
            let Some(index) = priors
                .iter()
                .position(|old| same_lane(&old.lane, &spec.lane))
            else {
                continue;
            };
            carry_spec(spec, priors.remove(index));
        }
    }
}

fn same_lane(a: &LaneSpec, b: &LaneSpec) -> bool {
    std::mem::discriminant(a) == std::mem::discriminant(b)
}

fn carry_spec(new: &mut AgentSpec, old: &AgentSpec) {
    if new.command.cwd.is_none() {
        new.command.cwd.clone_from(&old.command.cwd);
    }
    carry_env(&mut new.command.env, &old.command.env);
    if new.max_output_bytes.is_none() {
        new.max_output_bytes = old.max_output_bytes;
    }
    carry_lane(&mut new.lane, &old.lane);
}

/// Host env declarations ride inside whichever policy the spec declares —
/// `set` entries union under either policy; the sanitized `inherit` list
/// carries only when both sides keep one.
fn carry_env(new: &mut EnvPolicy, old: &EnvPolicy) {
    let old_set = match old {
        EnvPolicy::Sanitized { set, .. } | EnvPolicy::InheritAll { set } => set,
    };
    let new_set = match new {
        EnvPolicy::Sanitized { set, .. } | EnvPolicy::InheritAll { set } => set,
    };
    for (name, value) in old_set {
        new_set.entry(name.clone()).or_insert_with(|| value.clone());
    }
    if let (EnvPolicy::Sanitized { inherit: ni, .. }, EnvPolicy::Sanitized { inherit: oi, .. }) =
        (new, old)
    {
        for name in oi {
            if !ni.contains(name) {
                ni.push(name.clone());
            }
        }
    }
}

fn carry_lane(new: &mut LaneSpec, old: &LaneSpec) {
    match (new, old) {
        (LaneSpec::Rpc(new_rpc), LaneSpec::Rpc(old_rpc)) => {
            // A server list the host registered on the lane itself carries
            // even when the new spec lost the registration surface — the
            // custody verifier's explicit refusal beats a silent drop.
            if new_rpc.mcp_servers.is_empty() {
                new_rpc.mcp_servers.clone_from(&old_rpc.mcp_servers);
            }
            if let (Some(config), Some(old_config)) = (&mut new_rpc.mcp_config, &old_rpc.mcp_config)
            {
                // The approval mechanism is card data; the server names it
                // applies to are host config — the card never emits them.
                // Without the mechanism there is nothing to attach them
                // to, so they move only while it exists.
                if config.tools_approval.is_some() && config.tools_approval_servers.is_empty() {
                    config
                        .tools_approval_servers
                        .clone_from(&old_config.tools_approval_servers);
                }
            }
        }
        (
            LaneSpec::Acp {
                mcp_servers: new_servers,
                enforced_permission: new_permission,
                ..
            },
            LaneSpec::Acp {
                mcp_servers: old_servers,
                enforced_permission: old_permission,
                ..
            },
        ) => {
            if new_servers.is_empty() {
                new_servers.clone_from(old_servers);
            }
            if new_permission.is_none() {
                new_permission.clone_from(old_permission);
            }
        }
        (
            LaneSpec::Serve {
                mcp_servers: new_servers,
            },
            LaneSpec::Serve {
                mcp_servers: old_servers,
            },
        ) if new_servers.is_empty() => {
            new_servers.clone_from(old_servers);
        }
        (LaneSpec::Cli(new_cli), LaneSpec::Cli(old_cli)) if new_cli.mcp_servers.is_empty() => {
            new_cli.mcp_servers.clone_from(&old_cli.mcp_servers);
        }
        _ => {}
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

fn is_false(value: &bool) -> bool {
    !*value
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
    /// Strict mode: sessions for this role must run on a sealed surface —
    /// every ambient code and instruction channel the agent would
    /// otherwise discover (extensions, skills, prompt templates, context
    /// files) is cut so the declared surface equals the actual one. A
    /// candidate that cannot seal is refused rather than served
    /// unsealed. Non-strict sessions run the ambient environment as-is.
    #[serde(default, skip_serializing_if = "is_false")]
    pub strict: bool,
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
/// A recurring schedule window for availability (e.g. peak/off-peak discount periods).
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AvailabilityWindow {
    /// Comma-separated day codes: "mon", "tue", "wed", "thu", "fri", "sat", "sun", or "*"
    #[serde(default = "default_all_days")]
    pub days: String,
    /// Wall-clock start time in "HH:MM"
    pub from: String,
    /// Wall-clock end time in "HH:MM"
    pub to: String,
    /// IANA timezone name, defaults to "UTC"
    #[serde(default = "default_timezone")]
    pub timezone: String,
}

fn default_all_days() -> String {
    "*".to_string()
}

/// The weekday vocabulary — the single definition `covers` matches against
/// and `is_well_formed` validates, so an accepted spelling can always be
/// covered and an uncovered spelling never passes load.
const WEEKDAYS: [&str; 7] = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"];

fn weekday_code(date: jiff::civil::Date) -> &'static str {
    WEEKDAYS[date.weekday().to_monday_one_offset() as usize - 1]
}

fn default_timezone() -> String {
    "UTC".to_string()
}

impl AvailabilityWindow {
    pub fn covers(&self, now: &jiff::Timestamp) -> bool {
        let Ok(zone) = jiff::tz::TimeZone::get(&self.timezone) else {
            return false;
        };
        let zoned = now.to_zoned(zone);
        let time = zoned.datetime().time();
        let minute_of_day = (time.hour() as u16) * 60 + (time.minute() as u16);
        let (Some(from), Some(to)) = (minute_bound(&self.from), minute_bound(&self.to)) else {
            return false;
        };

        let day_matches = |target_day: &str| {
            if self.days == "*" {
                return true;
            }
            self.days
                .split(',')
                .any(|d| d.trim().eq_ignore_ascii_case(target_day))
        };

        if from == to {
            // Equal bounds (e.g. 00:00 to 00:00) cover the entire day.
            let today = weekday_code(zoned.datetime().date());
            return day_matches(today);
        }

        if from < to {
            // Normal range within the same calendar day.
            if !(from..to).contains(&minute_of_day) {
                return false;
            }
            let today = weekday_code(zoned.datetime().date());
            return day_matches(today);
        }

        // Wrapped range across midnight (e.g. 18:00 to 09:00).
        if minute_of_day >= from {
            let today = weekday_code(zoned.datetime().date());
            day_matches(today)
        } else if minute_of_day < to {
            let yesterday = weekday_code(
                zoned
                    .datetime()
                    .date()
                    .checked_sub(jiff::SignedDuration::from_hours(24))
                    .unwrap_or(zoned.datetime().date()),
            );
            day_matches(yesterday)
        } else {
            false
        }
    }

    pub fn is_well_formed(&self) -> bool {
        let days_well_formed = if self.days == "*" {
            true
        } else {
            self.days.split(',').all(|day| {
                let day = day.trim();
                !day.is_empty() && WEEKDAYS.iter().any(|known| day.eq_ignore_ascii_case(known))
            })
        };
        jiff::tz::TimeZone::get(&self.timezone).is_ok()
            && minute_bound(&self.from).is_some()
            && minute_bound(&self.to).is_some()
            && days_well_formed
    }
}

fn minute_bound(raw: &str) -> Option<u16> {
    let (h_raw, m_raw) = raw.split_once(':')?;
    // Exactly two ASCII digits per field: `9:00` and `09:0` are typos, not
    // alternate spellings, and the vocabulary fails closed at load.
    let two_digits =
        |field: &str| field.len() == 2 && field.bytes().all(|byte| byte.is_ascii_digit());
    if !two_digits(h_raw) || !two_digits(m_raw) {
        return None;
    }
    let h: u16 = h_raw.parse().ok()?;
    let m: u16 = m_raw.parse().ok()?;
    if h == 24 && m == 0 {
        return Some(1440);
    }
    if h < 24 && m < 60 {
        Some(h * 60 + m)
    } else {
        None
    }
}

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
    /// Optional recurring schedule windows. When non-empty, now must match at least one window.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub windows: Vec<AvailabilityWindow>,
}

impl Availability {
    /// Whether `now` falls inside the bounds; a bound that cannot be parsed
    /// never covers.
    pub fn covers(&self, now: &jiff::Timestamp) -> bool {
        let lower = self
            .not_before
            .as_deref()
            .is_none_or(|raw| parse_time_bound(raw).is_some_and(|b| *now >= b));
        let upper = self
            .not_after
            .as_deref()
            .is_none_or(|raw| parse_time_bound(raw).is_some_and(|b| *now < b));
        if !(lower && upper) {
            return false;
        }
        if self.windows.is_empty() {
            return true;
        }
        self.windows.iter().any(|w| w.covers(now))
    }

    /// Every declared bound parses — documents carrying a malformed bound
    /// are rejected at `apply` rather than expiring silently at resolve.
    pub fn is_well_formed(&self) -> bool {
        let bounds_ok = [&self.not_before, &self.not_after]
            .into_iter()
            .flatten()
            .all(|raw| parse_time_bound(raw).is_some());
        if !bounds_ok {
            return false;
        }
        self.windows.iter().all(|w| w.is_well_formed())
    }
}

/// One availability/retirement bound: RFC 3339, or a bare date at
/// 00:00 UTC. Custody layers use the same parser for `retiresAt`.
pub fn parse_time_bound(raw: &str) -> Option<jiff::Timestamp> {
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

    /// Whether the binding requires a sealed session surface.
    pub fn strict(&self) -> bool {
        match self {
            Binding::Full(full) => full.strict,
            _ => false,
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

    /// Remove retired profiles from the candidate chain: fallbacks
    /// pointing at them drop out; a retired primary promotes its first
    /// surviving fallback. A chain with no surviving candidate becomes
    /// vacant — the lane no longer exists rather than silently
    /// resurrecting a spent entry. Custody calls this when pruning
    /// retired profiles so the document stays self-consistent.
    pub fn retire_profiles(&mut self, retired: &std::collections::BTreeSet<String>) {
        let vacancy = match self {
            Binding::Profile(id) => retired.contains(id.as_str()),
            Binding::Full(full) => {
                full.fallbacks.retain(|f| !retired.contains(f.profile()));
                if retired.contains(&full.profile) {
                    match full.fallbacks.first().cloned() {
                        Some(next) => {
                            full.profile = next.profile().to_string();
                            full.availability = next.availability().cloned();
                            full.fallbacks.remove(0);
                            false
                        }
                        None => true,
                    }
                } else {
                    false
                }
            }
            Binding::None => false,
        };
        if vacancy {
            *self = Binding::None;
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
    secrets: BTreeMap<String, Vec<SecretRecipe>>,
    providers: BTreeMap<String, ProviderDecl>,
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
    /// per key, profiles merge field-wise. A replaced agent entry carries
    /// the host declarations its replacement left blank — an overlay layer
    /// that re-materializes an agent from card data keeps the base entry's
    /// `mcpConfig.toolsApprovalServers`, lane `mcpServers`, env `set`
    /// entries and the other fields only the host can name; every field the
    /// overlay declares itself still wins. Atomic — a document that fails
    /// validation leaves the registry untouched.
    pub fn apply(&mut self, document: RegistryDocument) -> a4aa::Result<()> {
        let mut next = self.clone();
        for (id, mut entry) in document.agents {
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
            if let Some(prior) = next.agents.get(&id) {
                entry.carry_host_declarations_from(prior);
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
            if profile
                .retires_at
                .as_deref()
                .is_some_and(|raw| parse_time_bound(raw).is_none())
            {
                return Err(
                    AdapterError::permanent("profile_retirement_malformed", "registry/apply")
                        .with_detail(format!(
                            "profile `{id}` declares a retiresAt bound that is neither RFC 3339 nor YYYY-MM-DD"
                        )),
                );
            }
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
            if profile.windows.iter().any(|w| !w.is_well_formed()) {
                return Err(
                    AdapterError::permanent("profile_windows_malformed", "registry/apply")
                        .with_detail(format!(
                            "profile `{id}` declares a model window whose timezone or HH:MM bound is malformed"
                        )),
                );
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
        next.secrets.extend(document.secrets);
        next.providers.extend(document.providers);
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

    pub fn secrets(&self) -> &BTreeMap<String, Vec<SecretRecipe>> {
        &self.secrets
    }

    pub fn providers(&self) -> &BTreeMap<String, ProviderDecl> {
        &self.providers
    }

    pub fn provider(&self, name: &str) -> Option<&ProviderDecl> {
        self.providers.get(name)
    }

    /// Resolve a profile to the virtual agent it materializes.
    pub fn resolve(&self, profile_id: &str) -> a4aa::Result<crate::VirtualAgent> {
        let profile = self.profiles.get(profile_id).ok_or_else(|| {
            AdapterError::permanent("profile_unknown", "registry/resolve")
                .with_detail(format!("unknown profile `{profile_id}`"))
        })?;
        if profile.is_retired(&jiff::Timestamp::now()) {
            return Err(
                AdapterError::permanent("profile_retired", "registry/resolve")
                    .with_detail(format!("profile `{profile_id}` passed its retiresAt bound")),
            );
        }
        let entry = self.agents.get(&profile.agent).ok_or_else(|| {
            AdapterError::permanent("profile_agent_unknown", "registry/resolve").with_detail(
                format!(
                    "profile `{profile_id}` references unknown agent `{}`",
                    profile.agent
                ),
            )
        })?;
        Ok(
            crate::VirtualAgent::new(profile.clone(), entry.lane_specs()).secrets(
                std::sync::Arc::new(RegistrySecretResolver::new(self.secrets.clone())),
            ),
        )
    }

    /// Resolve a binding: the first currently servable candidate's profile
    /// plus its composed standing instructions (profile instructions first,
    /// binding prompt second). The same availability and retirement gates
    /// as `resolve_chain` apply — a bound profile whose window excludes
    /// now or that is retired must not surface here either.
    pub fn select(&self, binding_name: &str) -> a4aa::Result<Selection> {
        self.select_at(binding_name, &jiff::Timestamp::now())
    }

    /// `select` against an explicit instant — the deterministic form tests
    /// and schedulers use. A strict binding skips candidates that cannot
    /// seal their session surface, exactly as `resolve_chain_at` does.
    pub fn select_at(&self, binding_name: &str, now: &jiff::Timestamp) -> a4aa::Result<Selection> {
        let binding = self.require_binding(binding_name, "registry/select")?;
        let strict = binding.strict();
        let mut strict_error = None;
        for profile in self.available_candidates(binding, now) {
            let profile = if strict {
                let agent = self.candidate_agent(profile)?;
                match Self::seal_or_record(agent, &mut strict_error) {
                    Some(agent) => agent.profile().clone(),
                    None => continue,
                }
            } else {
                profile
            };
            let instructions = profile.instructions.clone();
            return Ok(Selection {
                profile,
                instructions,
            });
        }
        if let Some(error) = strict_error {
            return Err(error);
        }
        Err(Self::chain_expired(binding_name, "registry/select"))
    }

    /// The binding `name`, refusing vacant or unknown entries — the single
    /// place the selection paths build `binding_unknown`.
    fn require_binding(&self, binding_name: &str, op: &'static str) -> a4aa::Result<&Binding> {
        self.binding(binding_name).ok_or_else(|| {
            AdapterError::permanent("binding_unknown", op)
                .with_detail(format!("unknown binding `{binding_name}`"))
        })
    }

    /// `binding`'s candidates servable at `now`, in order: availability
    /// bounds and retirement filter each profile, then the binding's
    /// standing prompt composes into the survivor's instructions. Every
    /// selection path walks this one iterator, so a future gate applies
    /// everywhere a candidate can be picked rather than in one loop.
    fn available_candidates<'a>(
        &'a self,
        binding: &'a Binding,
        now: &'a jiff::Timestamp,
    ) -> impl Iterator<Item = Profile> + 'a {
        let prompt = binding.prompt().map(str::to_owned);
        std::iter::once((binding.profile(), binding.availability()))
            .chain(
                binding
                    .fallbacks()
                    .iter()
                    .map(|f| (f.profile(), f.availability())),
            )
            .filter_map(move |(id, availability)| {
                if availability.is_some_and(|a| !a.covers(now)) {
                    return None;
                }
                let mut profile = self
                    .profiles
                    .get(id)
                    .expect("bindings validate against profiles at apply")
                    .clone();
                if profile.is_retired(now) {
                    return None;
                }
                profile.instructions =
                    compose_instructions(profile.instructions.as_deref(), prompt.as_deref());
                Some(profile)
            })
    }

    /// `agent` sealed the way a strict binding requires — or `None` with
    /// the first sealing refusal kept in `first_error`, for the caller to
    /// surface when nothing in the chain can serve. The whole
    /// seal-and-remember step lives here so `select` and `resolve_chain`
    /// cannot drift on which refusal a dead chain reports.
    fn seal_or_record(
        agent: crate::VirtualAgent,
        first_error: &mut Option<a4aa::AdapterError>,
    ) -> Option<crate::VirtualAgent> {
        match agent.strict() {
            Ok(agent) => Some(agent),
            Err(error) => {
                first_error.get_or_insert(error);
                None
            }
        }
    }

    /// `profile` materialized with the registry's secret resolver — the
    /// object `strict()` can seal when the binding demands it.
    fn candidate_agent(&self, profile: Profile) -> a4aa::Result<crate::VirtualAgent> {
        let entry = self.agents.get(&profile.agent).ok_or_else(|| {
            AdapterError::permanent("profile_agent_unknown", "registry/resolve").with_detail(
                format!(
                    "profile `{}` references unknown agent `{}`",
                    profile.id, profile.agent
                ),
            )
        })?;
        Ok(
            crate::VirtualAgent::new(profile, entry.lane_specs()).secrets(std::sync::Arc::new(
                RegistrySecretResolver::new(self.secrets.clone()),
            )),
        )
    }

    fn chain_expired(binding_name: &str, op: &'static str) -> AdapterError {
        AdapterError::permanent("binding_chain_expired", op).with_detail(format!(
            "binding `{binding_name}` has no candidate available now"
        ))
    }

    /// Resolve a binding straight to its virtual agent, with the binding's
    /// standing prompt folded into the profile's instructions. Errors when
    /// every candidate's availability excludes now — a chain that expired
    /// is a lane that no longer exists, not a panic.
    pub fn resolve_binding(&self, binding_name: &str) -> a4aa::Result<crate::VirtualAgent> {
        self.resolve_chain(binding_name)?
            .into_iter()
            .next()
            .ok_or_else(|| Self::chain_expired(binding_name, "registry/resolve"))
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
        let binding = self.require_binding(binding_name, "registry/select")?;
        // A strict binding may only be served by candidates that can
        // seal their session surface: an unsealable candidate is
        // skipped like an unavailable one — the role degrades to the
        // candidates that keep the declared surface instead of
        // silently running ambient — and only a chain where nothing
        // can seal refuses the resolution outright.
        let strict = binding.strict();
        let mut strict_error = None;
        let mut candidates = Vec::new();
        for profile in self.available_candidates(binding, now) {
            let agent = self.candidate_agent(profile)?;
            if !strict {
                candidates.push(agent);
                continue;
            }
            if let Some(agent) = Self::seal_or_record(agent, &mut strict_error) {
                candidates.push(agent);
            }
        }
        if strict && candidates.is_empty() {
            if let Some(error) = strict_error {
                return Err(error);
            }
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
