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
pub mod profile;
pub mod registry;

pub use agent::{compose_prompt, VirtualAgent, VirtualSession};
pub use profile::{InstructionMode, Profile, ProfilePatch};
pub use registry::{compose_instructions, Binding, Registry, RegistryDocument, Selection};

// Re-export the a4aa surface hosts need to configure and drive sessions,
// so an awp consumer rarely needs a second dependency import.
pub use a4aa::{
    adapt, AdaptationPlan, AdapterError, AgentEvent, AgentSpec, Artifact, ArtifactScope,
    BeforeTurn, Capabilities, CliLaneSpec, CommandDecl, CommandSpec, CommandSurface, Dialect,
    EnvPolicy, EnvSource, EnvValue, ExtensionKind, HookDecl, HookEvent, HookIntent, HookSurface,
    LaneKind, LaneSpec, McpSurface, MechanismSet, NoSecrets, Observer, PermissionDecl,
    PermissionHandler, PermissionSurface, PluginDecl, PluginSurface, PromptChannel, Result,
    RpcCallSpec, RpcLaneSpec, RuleDecl, SecretResolver, SessionRequest, SharedObserver, SkillDecl,
    StreamMode, SubagentDecl, SubagentSurface, Support, SupportEntry, Target, TurnOutcome,
    TurnRequest, WriteMode,
};
