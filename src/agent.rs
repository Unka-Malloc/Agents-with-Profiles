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
//! [`InstructionMode`], and model/effort defaults use the session or turn
//! selector phase the A4AA lane actually supports.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use a4aa::{
    Adapter, AgentEvent, AgentSession, AgentSpec, BeforeTurn, Capabilities, EnvPolicy, LaneKind,
    LaneSpec, PermissionHandler, SecretResolver, SelectorPhase, SelectorSupport, SessionRequest,
    SharedObserver, TurnOutcome, TurnRequest,
};
use tokio::sync::{mpsc, Mutex};

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
    /// adapter untouched. Profile model/effort defaults enter session config
    /// only when that lane applies them at session establishment; turn-only
    /// defaults are carried by `VirtualSession`. Explicit session config wins.
    /// Mode and permission remain session selectors.
    pub async fn open_session(
        &self,
        mut request: SessionRequest,
        events: mpsc::UnboundedSender<AgentEvent>,
    ) -> a4aa::Result<VirtualSession> {
        if request.cwd.is_none() {
            request.cwd = self.profile.cwd.clone();
        }
        let session_selectors = self.spec.selector_support(SelectorPhase::Session);
        let mut turn_model_default = None;
        let mut turn_effort_default = None;
        if request.config.model.is_none() {
            if session_selectors.model != SelectorSupport::Unsupported {
                request.config.model = self.profile.model.clone();
            } else {
                turn_model_default = self.profile.model.clone();
            }
        }
        if request.config.mode.is_none() {
            request.config.mode = self.profile.mode.clone();
        }
        if request.config.effort.is_none() {
            if session_selectors.effort != SelectorSupport::Unsupported {
                request.config.effort = self.profile.effort.clone();
            } else {
                turn_effort_default = self.profile.effort.clone();
            }
        }
        if request.config.permission.is_none() {
            request.config.permission = self.profile.permission.clone();
        }
        let session = self.adapter().open_session(request, events).await?;
        Ok(VirtualSession {
            session: Arc::new(session),
            profile: self.profile.clone(),
            turn_model_default,
            turn_effort_default,
            stateless_lane: matches!(self.spec.lane, LaneSpec::Cli(_)),
            instructions_sent: Arc::new(AtomicBool::new(false)),
            turn_lock: Arc::new(Mutex::new(())),
        })
    }
}

/// An open session on a virtual agent. Profile turn semantics apply here:
/// instructions compose into prompt text, and turn-only defaults fill in.
pub struct VirtualSession {
    session: Arc<AgentSession>,
    profile: Profile,
    /// Profile defaults for selectors the lane consumes per turn rather than
    /// at session establishment. Session-applied defaults stay with A4AA's
    /// SessionRequest so an explicit open-session value is never replaced.
    turn_model_default: Option<String>,
    turn_effort_default: Option<String>,
    /// Stateless lanes (cli) start a fresh process per turn, so instructions
    /// are delivered on every prompt regardless of `InstructionMode`.
    stateless_lane: bool,
    instructions_sent: Arc<AtomicBool>,
    // Composition and submission share one turn boundary. Concurrent callers
    // must observe the previous turn's confirmed instruction-delivery state.
    turn_lock: Arc<Mutex<()>>,
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
        let turn = self.turn_lock.clone().lock_owned().await;
        if self.instructions_due() {
            request.text = compose_prompt(self.profile.instructions.as_deref(), &request.text);
        }
        if request.model.is_none() {
            request.model = self.turn_model_default.clone();
        }
        if request.effort.is_none() {
            request.effort = self.turn_effort_default.clone();
        }
        self.run_owned_prompt(request, turn, true).await
    }

    /// Submit one turn exactly as given — no instruction composition, no
    /// default filling. The escape hatch for hosts that deliver profile
    /// context through their own channel (skill files, harness prompts).
    pub async fn prompt_direct(&self, request: TurnRequest) -> a4aa::Result<TurnOutcome> {
        let turn = self.turn_lock.clone().lock_owned().await;
        self.run_owned_prompt(request, turn, false).await
    }

    async fn run_owned_prompt(
        &self,
        request: TurnRequest,
        turn: tokio::sync::OwnedMutexGuard<()>,
        record_instructions: bool,
    ) -> a4aa::Result<TurnOutcome> {
        let session = self.session.clone();
        let instructions_sent = self.instructions_sent.clone();
        tokio::spawn(async move {
            let _turn = turn;
            let outcome = session.prompt(request).await;
            // Preserve profile state and ordering even when only the caller's
            // waiter is dropped. A cancelled first attempt remains pending.
            if record_instructions
                && outcome.as_ref().is_ok_and(|result| {
                    !matches!(result.stop_reason.as_str(), "cancelled" | "interrupted")
                })
            {
                instructions_sent.store(true, Ordering::Relaxed);
            }
            outcome
        })
        .await
        .map_err(|_| {
            a4aa::AdapterError::uncertain("agent_profile_driver_failed", "profile/prompt")
        })?
    }

    /// Explicit user cancellation of the in-flight turn.
    pub async fn cancel(&self) -> a4aa::Result<()> {
        self.session.cancel().await
    }

    /// Close the session and terminate the agent process tree.
    pub async fn close(&self) -> a4aa::Result<()> {
        self.session.close().await
    }
}
