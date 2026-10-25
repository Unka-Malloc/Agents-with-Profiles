//! # awp — Agents with Profiles
//!
//! Hosts that drive multiple coding agents keep rediscovering the same
//! pattern: a *base agent* (the CLI or service that actually runs) plus a
//! *profile* (model, effort, persona instructions, environment, MCP
//! servers, dispatch preferences) materializes a distinct *virtual agent*.
//! A reviewer profile and an implementer profile can share one base agent
//! and still behave as two different members of a workflow.
//!
//! `awp` is that layer over [`a4aa`]:
//!
//! - [`Profile`] — the resolved, named behavior record; [`ProfilePatch`] is
//!   its optional-everything wire and overlay form.
//! - [`RegistryDocument`] / [`Registry`] — a validated store of agent
//!   specs, profile patches and role bindings, with layered overlay
//!   semantics (base → organization → project, or however many layers the
//!   host stacks).
//! - [`VirtualAgent`] / [`VirtualSession`] — the materialized agent:
//!   profile environment and MCP servers fold into the effective
//!   [`a4aa::AgentSpec`]; sessions compose instructions into prompt text
//!   and fill model/effort defaults per turn.
//! - [`AgentPreset`] — verified headless launch templates (`codex`,
//!   `cursor`, `dsh`, `hermes`, `opencode`, `pi`) so hosts fill program,
//!   argv and required env from structs instead of hand-assembled CLI.
//! - The `awp` binary — `inspect`, `validate`, `resolve` and `run` for
//!   document-driven operation, plus an `awp fixture-agent` test double.
//!   `awp resolve` emits the effective spec and turn plan as JSON, so
//!   hosts in any language can feed it to `a4aa serve` and get identical
//!   profile semantics without reimplementing the merge rules.
//!
//! Design invariants (inherited from a4aa where they apply):
//!
//! - **Zero-assumption output.** The crate never imposes a schema on agent
//!   output; instructions compose into *input* text, nothing more.
//! - **Fail closed.** Unknown profiles, agents and bindings, missing
//!   `agent` references and invalid documents are configuration errors,
//!   never runtime guesses.
//! - **Explicit cancellation only.** `VirtualSession::cancel` is the sole
//!   cancel path — it forwards to the lane verbatim.
//! - **Atomic layering.** A registry overlay validates fully or not at
//!   all; a failed `apply` leaves the previous registry intact.

pub mod agent;
#[cfg(feature = "presets")]
pub mod preset;
pub mod profile;
pub mod registry;

pub use agent::{compose_prompt, VirtualAgent, VirtualSession};
pub use jiff;
#[cfg(feature = "presets")]
pub use preset::{
    apply_preset, ids as preset_ids, instantiate as instantiate_preset, lookup as lookup_preset,
    AgentPreset, PresetError, PresetUnit, RequiredEnv, PRESETS,
};
pub use profile::{
    InstructionMode, ModelWindow, Profile, ProfilePatch, ProviderDecl, RetryPolicy, StateScope,
};
pub use registry::{
    compose_instructions, parse_time_bound, resolve_secret, AgentEntry, Availability,
    AvailabilityWindow, Binding, BindingFull, Fallback, FallbackEntry, FileSecret, Registry,
    RegistryDocument, RegistrySecretResolver, SecretRecipe, Selection, TomlSecret,
};

// Re-export the a4aa surface hosts need to configure and drive sessions,
// so an awp consumer rarely needs a second dependency import.
pub use a4aa::{
    adapt, AdaptationPlan, AdapterError, AgentEvent, AgentSpec, Artifact, ArtifactScope,
    BeforeTurn, Capabilities, CliLaneSpec, CommandDecl, CommandSpec, CommandSurface, Dialect,
    EnvPolicy, EnvSource, EnvValue, ExtensionKind, HookDecl, HookEvent, HookIntent, HookSurface,
    LaneKind, LaneSpec, McpSurface, MechanismSet, NoSecrets, Observer, PermissionDecl,
    PermissionHandler, PermissionSurface, PluginDecl, PluginSurface, PromptChannel, Result,
    RpcCallSpec, RpcLaneSpec, RpcMcpConfig, RuleDecl, SecretResolver, SessionRequest,
    SharedObserver, SkillDecl, StreamMode, SubagentDecl, SubagentSurface, Support, SupportEntry,
    Target, TurnOutcome, TurnRequest, WriteMode,
};
