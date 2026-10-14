//! The virtual agent: one base spec plus one resolved profile.
//!
//! [`VirtualAgent`] is the materialized form of "agents with profiles": the
//! [`a4aa::AgentSpec`] carries *how to reach the agent*, the [`Profile`]
//! carries *how the agent should behave*. Materialization folds the
//! profile's environment entries into the spec's environment policy and its
//! MCP servers into the `acp` lane — the adapter that opens the session sees
//! one effective spec.
//!
//! [`VirtualSession`] then applies profile semantics per turn: persona
//! instructions are composed into prompt text under the profile's
//! [`InstructionMode`], and model/effort defaults fill unset turn fields
//! for the lanes that accept them.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use a4aa::{
    Adapter, AgentEvent, AgentSession, AgentSpec, BeforeTurn, Capabilities, EnvPolicy, LaneKind,
    LaneSpec, PermissionHandler, SecretResolver, SessionRequest, SharedObserver, TurnOutcome,
    TurnRequest,
};
use tokio::sync::mpsc;

use crate::profile::{InstructionMode, Profile};

/// Compose standing instructions and turn text into one prompt body.
/// Empty or absent instructions pass the text through verbatim.
pub fn compose_prompt(instructions: Option<&str>, text: &str) -> String {
    match instructions
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        Some(instructions) => format!("{instructions}\n\n{text}"),
        None => text.to_owned(),
    }
}

/// One base agent spec materialized with one profile.
///
/// The agent holds the effective spec and the adapter hooks; each
/// `open_session` builds a fresh [`a4aa::Adapter`] on the caller's runtime.
pub struct VirtualAgent {
    profile: Profile,
    spec: AgentSpec,
    secrets: Arc<dyn SecretResolver>,
    observer: SharedObserver,
    permission: Option<Arc<dyn PermissionHandler>>,
    before_turn: Option<Arc<dyn BeforeTurn>>,
}

impl VirtualAgent {
    /// Materialize `profile` onto `spec`: profile environment entries merge
    /// into the command's environment policy `set`, profile MCP servers
    /// append to the `acp` lane (other lanes keep none — they have no MCP
    /// surface).
    pub fn new(profile: Profile, mut spec: AgentSpec) -> Self {
        if !profile.env.is_empty() {
            let set = match &mut spec.command.env {
                EnvPolicy::Sanitized { set, .. } | EnvPolicy::InheritAll { set } => set,
            };
            for (name, value) in &profile.env {
                set.insert(name.clone(), value.clone());
            }
        }
        if let LaneSpec::Acp { mcp_servers, .. } = &mut spec.lane {
            mcp_servers.extend(profile.mcp_servers.iter().cloned());
        }
        Self {
            profile,
            spec,
            secrets: Arc::new(a4aa::NoSecrets),
            observer: a4aa::observe::none(),
            permission: None,
            before_turn: None,
        }
    }

    pub fn secrets(mut self, secrets: Arc<dyn SecretResolver>) -> Self {
        self.secrets = secrets;
        self
    }

    pub fn observer(mut self, observer: SharedObserver) -> Self {
        self.observer = observer;
        self
    }

    pub fn permission(mut self, handler: Arc<dyn PermissionHandler>) -> Self {
        self.permission = Some(handler);
        self
    }

    pub fn before_turn(mut self, hook: Arc<dyn BeforeTurn>) -> Self {
        self.before_turn = Some(hook);
        self
    }

    /// The resolved profile this agent runs.
    pub fn profile(&self) -> &Profile {
        &self.profile
    }

    /// The effective spec after materialization — the same value a
    /// `resolve` call emits for foreign hosts.
    pub fn spec(&self) -> &AgentSpec {
        &self.spec
    }

    /// The lane this agent speaks.
    pub fn lane_kind(&self) -> LaneKind {
        match &self.spec.lane {
            LaneSpec::Acp { .. } => LaneKind::Acp,
            LaneSpec::Rpc(_) => LaneKind::Rpc,
            LaneSpec::Cli(_) => LaneKind::Cli,
        }
    }

    /// Negotiated-static capability picture, straight from the adapter.
    pub fn capabilities(&self) -> Capabilities {
        self.adapter().capabilities()
    }

    fn adapter(&self) -> Adapter {
        let mut adapter = Adapter::new(self.spec.clone())
            .secrets(self.secrets.clone())
            .observer(self.observer.clone());
        if let Some(handler) = &self.permission {
            adapter = adapter.permission(handler.clone());
        }
        if let Some(hook) = &self.before_turn {
            adapter = adapter.before_turn(hook.clone());
        }
        adapter
    }

    /// Open a session. The profile's `cwd` is the default working directory;
    /// an explicit `request.cwd` wins. `request.env` passes through to the
    /// adapter untouched.
    pub async fn open_session(
        &self,
        mut request: SessionRequest,
        events: mpsc::UnboundedSender<AgentEvent>,
    ) -> a4aa::Result<VirtualSession> {
        if request.cwd.is_none() {
            request.cwd = self.profile.cwd.clone();
        }
        let session = self.adapter().open_session(request, events).await?;
        Ok(VirtualSession {
            session,
            profile: self.profile.clone(),
            stateless_lane: matches!(self.spec.lane, LaneSpec::Cli(_)),
            instructions_sent: AtomicBool::new(false),
        })
    }
}

/// An open session on a virtual agent. Profile turn semantics apply here:
/// instructions compose into prompt text, model/effort defaults fill in.
pub struct VirtualSession {
    session: AgentSession,
    profile: Profile,
    /// Stateless lanes (cli) start a fresh process per turn, so instructions
    /// are delivered on every prompt regardless of `InstructionMode`.
    stateless_lane: bool,
    instructions_sent: AtomicBool,
}

impl VirtualSession {
    pub fn session_id(&self) -> &str {
        self.session.session_id()
    }

    /// The profile this session runs.
    pub fn profile(&self) -> &Profile {
        &self.profile
    }

    /// The text a prompt would send for `text` under the profile's
    /// instruction mode right now — what the wire would carry.
    pub fn composed_prompt(&self, text: &str) -> String {
        if self.instructions_due() {
            compose_prompt(self.profile.instructions.as_deref(), text)
        } else {
            text.to_owned()
        }
    }

    fn instructions_due(&self) -> bool {
        self.stateless_lane
            || self.profile.instruction_mode == InstructionMode::EveryTurn
            || !self.instructions_sent.load(Ordering::Relaxed)
    }

    /// Submit one turn. Profile instructions and model/effort defaults
    /// apply.
    pub async fn prompt(&self, text: impl Into<String>) -> a4aa::Result<TurnOutcome> {
        self.prompt_with(TurnRequest {
            text: text.into(),
            model: None,
            effort: None,
        })
        .await
    }

    /// Submit one turn with explicit fields. Set model/effort override the
    /// profile defaults; unset fields inherit them.
    pub async fn prompt_with(&self, mut request: TurnRequest) -> a4aa::Result<TurnOutcome> {
        if self.instructions_due() {
            request.text = compose_prompt(self.profile.instructions.as_deref(), &request.text);
        }
        if request.model.is_none() {
            request.model = self.profile.model.clone();
        }
        if request.effort.is_none() {
            request.effort = self.profile.effort.clone();
        }
        let outcome = self.session.prompt(request).await;
        if outcome.is_ok() {
            self.instructions_sent.store(true, Ordering::Relaxed);
        }
        outcome
    }

    /// Submit one turn exactly as given — no instruction composition, no
    /// default filling. The escape hatch for hosts that deliver profile
    /// context through their own channel (skill files, harness prompts).
    pub async fn prompt_direct(&self, request: TurnRequest) -> a4aa::Result<TurnOutcome> {
        self.session.prompt(request).await
    }

    /// Explicit user cancellation of the in-flight turn.
    pub async fn cancel(&self) -> a4aa::Result<()> {
        self.session.cancel().await
    }

    /// Close the session and terminate the agent process tree.
    pub async fn close(self) -> a4aa::Result<()> {
        self.session.close().await
    }
}
