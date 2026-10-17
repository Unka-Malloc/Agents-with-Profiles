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

/// One base agent materialized with one profile: the profile's resolved
/// profile plus every lane variant the agent declares, in fallback order.
///
/// The agent holds the effective specs and the adapter hooks; each
/// `open_session` builds a fresh [`a4aa::Adapter`] on the caller's runtime.
pub struct VirtualAgent {
    profile: Profile,
    /// Lane variants in fallback order — index 0 is the declared primary.
    specs: Vec<AgentSpec>,
    secrets: Arc<dyn SecretResolver>,
    observer: SharedObserver,
    permission: Option<Arc<dyn PermissionHandler>>,
    before_turn: Option<Arc<dyn BeforeTurn>>,
}

/// Fold the profile into one spec: environment entries merge into the
/// command's environment policy `set`, profile MCP servers append to any
/// lane that carries an MCP surface (`acp` session/new, `serve` config
/// document) — lanes without one keep none.
fn materialize(profile: &Profile, spec: &mut AgentSpec) {
    if !profile.env.is_empty() {
        let set = match &mut spec.command.env {
            EnvPolicy::Sanitized { set, .. } | EnvPolicy::InheritAll { set } => set,
        };
        for (name, value) in &profile.env {
            set.insert(name.clone(), value.clone());
        }
    }
    let servers = match &mut spec.lane {
        LaneSpec::Acp { mcp_servers, .. } | LaneSpec::Serve { mcp_servers } => mcp_servers,
        _ => return,
    };
    servers.extend(profile.mcp_servers.iter().cloned());
}

impl VirtualAgent {
    /// Materialize `profile` onto every declared lane variant. `specs` is
    /// the agent's lane list in fallback order: primary first, then each
    /// protocol alternate — the first entry is the effective spec reported
    /// by [`VirtualAgent::spec`].
    pub fn new(profile: Profile, specs: Vec<AgentSpec>) -> Self {
        let mut specs = specs;
        for spec in &mut specs {
            materialize(&profile, spec);
        }
        Self {
            profile,
            specs,
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

    /// The effective spec after materialization — the primary lane variant,
    /// the same value a `resolve` call emits for foreign hosts.
    pub fn spec(&self) -> &AgentSpec {
        &self.specs[0]
    }

    /// Every lane variant in fallback order.
    pub fn lane_specs(&self) -> &[AgentSpec] {
        &self.specs
    }

    /// The primary lane this agent speaks.
    pub fn lane_kind(&self) -> LaneKind {
        match &self.specs[0].lane {
            LaneSpec::Acp { .. } => LaneKind::Acp,
            LaneSpec::Rpc(_) => LaneKind::Rpc,
            LaneSpec::Cli(_) => LaneKind::Cli,
            LaneSpec::Serve { .. } => LaneKind::Serve,
        }
    }

    /// Negotiated-static capability picture, straight from the adapter.
    pub fn capabilities(&self) -> Capabilities {
        self.adapter(&self.specs[0]).capabilities()
    }

    fn adapter(&self, spec: &AgentSpec) -> Adapter {
        let mut adapter = Adapter::new(spec.clone())
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
    ///
    /// Lane fallback happens here: each declared lane variant gets the full
    /// model chain, and a failure the model chain cannot explain (transport,
    /// handshake, a lane that cannot express a requested selector) advances
    /// to the next lane. Provider-side model exhaustion skips the remaining
    /// lanes — the same account answers every protocol, so the caller's
    /// profile fallback is the right next step.
    pub async fn open_session(
        &self,
        mut request: SessionRequest,
        events: mpsc::UnboundedSender<AgentEvent>,
    ) -> a4aa::Result<VirtualSession> {
        if request.cwd.is_none() {
            request.cwd = self.profile.cwd.clone();
        }
        let chain = self.profile.effective_chain(&jiff::Zoned::now());
        let mut lane_error = None;
        for (lane_index, spec) in self.specs.iter().enumerate() {
            let (mut lane_request, turn_model_default, turn_effort_default) =
                lane_open(spec, &request, &self.profile, chain.first().cloned());
            // A requested model the lane cannot express at establishment
            // advances the chain before the first turn is even attempted.
            let session = loop {
                let result = self
                    .adapter(spec)
                    .open_session(lane_request.clone(), events.clone())
                    .await;
                match result {
                    Err(error) if model_failure(&error) => {
                        let Some(next) = next_model(&chain, lane_request.config.model.as_deref())
                        else {
                            // Model exhaustion is provider-side: remaining
                            // lanes answer from the same account, so the
                            // failure belongs to the caller's next profile.
                            return Err(error);
                        };
                        lane_request.config.model = Some(next);
                    }
                    Err(error) => {
                        lane_error = Some(error);
                        break None;
                    }
                    Ok(session) => break Some((session, lane_request)),
                }
            };
            let Some((session, effective_request)) = session else {
                continue;
            };
            let active_model = effective_request
                .config
                .model
                .clone()
                .or_else(|| turn_model_default.clone());
            let remaining_models = remaining_models(&chain, active_model.as_deref());
            return Ok(VirtualSession {
                session: Arc::new(tokio::sync::RwLock::new(session)),
                profile: self.profile.clone(),
                turn_model_default: Arc::new(std::sync::Mutex::new(turn_model_default)),
                turn_effort_default: Arc::new(std::sync::Mutex::new(turn_effort_default)),
                model_state: Arc::new(std::sync::Mutex::new(ModelChain {
                    active: active_model,
                    remaining: remaining_models,
                })),
                lane: Arc::new(tokio::sync::Mutex::new(LaneState {
                    specs: self.specs.clone(),
                    index: lane_index,
                    requested: request.clone(),
                    open: effective_request,
                    events: events.clone(),
                    secrets: self.secrets.clone(),
                    observer: self.observer.clone(),
                    permission: self.permission.clone(),
                    before_turn: self.before_turn.clone(),
                })),
                stateless_lane: Arc::new(AtomicBool::new(matches!(spec.lane, LaneSpec::Cli(_)))),
                instructions_sent: Arc::new(AtomicBool::new(false)),
                turn_lock: Arc::new(Mutex::new(())),
            });
        }
        Err(lane_error
            .unwrap_or_else(|| a4aa::AdapterError::permanent("agent_unavailable", "session/open")))
    }
}

/// The lane-fallback state one open session carries: every declared spec
/// plus everything needed to reopen the session on the next lane when the
/// current one fails for non-model reasons.
struct LaneState {
    specs: Vec<AgentSpec>,
    index: usize,
    /// The caller's request before lane normalization — re-projected
    /// against each next lane's selector support on a switch.
    requested: SessionRequest,
    /// The request the active lane opened with — replayed on the next lane.
    open: SessionRequest,
    events: mpsc::UnboundedSender<AgentEvent>,
    secrets: Arc<dyn SecretResolver>,
    observer: SharedObserver,
    permission: Option<Arc<dyn PermissionHandler>>,
    before_turn: Option<Arc<dyn BeforeTurn>>,
}

impl LaneState {
    fn adapter(&self, index: usize) -> Adapter {
        let mut adapter = Adapter::new(self.specs[index].clone())
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
}

/// Model fallback state for one session: the model currently selected and
/// the remaining chain entries a failed turn may advance to.
#[derive(Default)]
struct ModelChain {
    active: Option<String>,
    remaining: Vec<String>,
}

/// Chain entries that remain after `active` — every entry once `active` is
/// not part of the chain.
fn remaining_models(chain: &[String], active: Option<&str>) -> Vec<String> {
    match active.and_then(|value| chain.iter().position(|entry| entry == value)) {
        Some(index) => chain[index + 1..].to_vec(),
        None => chain.to_vec(),
    }
}

/// The next chain entry after `current`, or the chain head when `current`
/// is outside the chain.
fn next_model(chain: &[String], current: Option<&str>) -> Option<String> {
    match current.and_then(|value| chain.iter().position(|entry| entry == value)) {
        Some(index) => chain.get(index + 1).cloned(),
        None => chain.first().cloned(),
    }
}

/// Provider-side model exhaustion or rejection signatures. Matched against
/// the error detail — agents report quota and availability failures as
/// free-form protocol errors, so the patterns stay narrow: a match means
/// the turn provably could not complete on this model. Bare status codes
/// and generic phrases ("insufficient", "capacity", "does not exist") are
/// excluded — they describe permissions and unrelated objects too often
/// to reroute a turn.
const MODEL_EXHAUSTION: &[&str] = &[
    "rate limit",
    "ratelimit",
    "too many requests",
    "resource_exhausted",
    "quota",
    "insufficient_quota",
    "overloaded",
    "model unavailable",
    "model not found",
    "no such model",
    "model does not exist",
    "unsupported model",
];

/// A requested model the agent cannot express (`agent_configuration_unavailable`
/// names the rejected slot in its detail).
fn model_rejected(error: &a4aa::AdapterError) -> bool {
    error.code == "agent_configuration_unavailable"
        && error
            .detail
            .as_deref()
            .is_some_and(|detail| detail.starts_with("model "))
}

/// A failure worth retrying on the session's next model: explicit model
/// rejection, or a protocol failure at the session/model-selection stage
/// whose detail reads as provider-side exhaustion or unavailability. An
/// open-stage failure with that signature means the provider declined the
/// session's model — same treatment, never a lane fault.
fn model_failure(error: &a4aa::AdapterError) -> bool {
    if model_rejected(error) {
        return true;
    }
    error.code == "agent_protocol_failed"
        && matches!(
            error.stage,
            "session/open" | "session/prompt" | "session/set_model" | "session/set_config_option"
        )
        && error.detail.as_deref().is_some_and(|detail| {
            let detail = detail.to_lowercase();
            MODEL_EXHAUSTION
                .iter()
                .any(|pattern| detail.contains(pattern))
        })
}

/// Project a session request onto one lane. Profile defaults (and the
/// chain's head model) enter session config only where the lane applies
/// them at establishment; selectors the lane consumes per turn surface as
/// the returned `(request, turn_model, turn_effort)` defaults instead.
/// Used at open and again on every lane switch — a request normalized for
/// one lane is never replayed verbatim onto another.
fn lane_open(
    spec: &AgentSpec,
    request: &SessionRequest,
    profile: &Profile,
    head_model: Option<String>,
) -> (SessionRequest, Option<String>, Option<String>) {
    let session_selectors = spec.selector_support(SelectorPhase::Session);
    let mut turn_model_default = None;
    let mut turn_effort_default = None;
    let mut lane_request = request.clone();
    if lane_request.config.model.is_none() {
        if session_selectors.model != SelectorSupport::Unsupported {
            lane_request.config.model = head_model;
        } else {
            turn_model_default = head_model;
        }
    }
    if lane_request.config.mode.is_none() {
        lane_request.config.mode = profile.mode.clone();
    }
    if lane_request.config.effort.is_none() {
        if session_selectors.effort != SelectorSupport::Unsupported {
            lane_request.config.effort = profile.effort.clone();
        } else {
            turn_effort_default = profile.effort.clone();
        }
    }
    if lane_request.config.permission.is_none() {
        lane_request.config.permission = profile.permission.clone();
    }
    (lane_request, turn_model_default, turn_effort_default)
}

/// An open session on a virtual agent. Profile turn semantics apply here:
/// instructions compose into prompt text, and turn-only defaults fill in.
pub struct VirtualSession {
    session: Arc<tokio::sync::RwLock<AgentSession>>,
    profile: Profile,
    /// Profile defaults for selectors the lane consumes per turn rather than
    /// at session establishment. Session-applied defaults stay with A4AA's
    /// SessionRequest so an explicit open-session value is never replaced.
    /// A mid-session lane switch recomputes both against the new lane.
    turn_model_default: Arc<std::sync::Mutex<Option<String>>>,
    turn_effort_default: Arc<std::sync::Mutex<Option<String>>>,
    /// The session's model chain: a turn that fails on the active model is
    /// retried against the next entry — same session, same composed prompt.
    model_state: Arc<std::sync::Mutex<ModelChain>>,
    /// The lane variants behind this session and the material needed to
    /// reopen it on the next lane after a transport-level failure. Shared
    /// and locked: a mid-turn switch persists the active lane index for
    /// every later turn instead of rescanning dead lanes.
    lane: Arc<tokio::sync::Mutex<LaneState>>,
    /// Stateless lanes (cli) start a fresh process per turn, so instructions
    /// are delivered on every prompt regardless of `InstructionMode`. A lane
    /// switch re-evaluates this against the new lane's spec.
    stateless_lane: Arc<AtomicBool>,
    instructions_sent: Arc<AtomicBool>,
    // Composition and submission share one turn boundary. Concurrent callers
    // must observe the previous turn's confirmed instruction-delivery state.
    turn_lock: Arc<Mutex<()>>,
}

impl VirtualSession {
    /// The native session id of the currently active lane. Empty while a
    /// lane switch is in flight; callers needing a stable identity read it
    /// once per completed turn.
    pub async fn session_id(&self) -> String {
        self.session.read().await.session_id().to_owned()
    }

    /// The profile this session runs.
    pub fn profile(&self) -> &Profile {
        &self.profile
    }

    /// The model currently serving this session — the profile's first
    /// accepted chain entry at establishment, or the entry a failed turn
    /// most recently fell back to.
    pub fn active_model(&self) -> Option<String> {
        self.model_state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .active
            .clone()
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
        self.stateless_lane.load(Ordering::Relaxed)
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
        let instructions_included = self.instructions_due();
        if instructions_included {
            request.text = compose_prompt(self.profile.instructions.as_deref(), &request.text);
        }
        if request.model.is_none() {
            request.model = self
                .turn_model_default
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .clone();
        }
        if request.effort.is_none() {
            request.effort = self
                .turn_effort_default
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .clone();
        }
        self.run_owned_prompt(request, turn, true, instructions_included)
            .await
    }

    /// Submit one turn exactly as given — no instruction composition, no
    /// default filling. The escape hatch for hosts that deliver profile
    /// context through their own channel (skill files, harness prompts).
    pub async fn prompt_direct(&self, request: TurnRequest) -> a4aa::Result<TurnOutcome> {
        let turn = self.turn_lock.clone().lock_owned().await;
        self.run_owned_prompt(request, turn, false, false).await
    }

    async fn run_owned_prompt(
        &self,
        request: TurnRequest,
        turn: tokio::sync::OwnedMutexGuard<()>,
        record_instructions: bool,
        instructions_included: bool,
    ) -> a4aa::Result<TurnOutcome> {
        let session = self.session.clone();
        let instructions_sent = self.instructions_sent.clone();
        let model_state = self.model_state.clone();
        let lane = self.lane.clone();
        let turn_model_default = self.turn_model_default.clone();
        let turn_effort_default = self.turn_effort_default.clone();
        let stateless = self.stateless_lane.clone();
        let profile = self.profile.clone();
        tokio::spawn(async move {
            let _turn = turn;
            let mut request = request;
            let mut instructions_included = instructions_included;
            let outcome = 'lanes: loop {
                // The model loop on the current lane: provider-side model
                // failures advance the chain and resubmit on the same
                // session.
                let outcome = loop {
                    let outcome = session.read().await.prompt(request.clone()).await;
                    let next = match &outcome {
                        // The model itself could not serve this turn:
                        // advance the chain and resubmit — the lane's own
                        // configuration surface applies the per-turn model
                        // before dispatch.
                        Err(error) if model_failure(error) => model_state
                            .lock()
                            .unwrap_or_else(|error| error.into_inner())
                            .remaining
                            .first()
                            .cloned(),
                        _ => None,
                    };
                    let Some(next) = next else { break outcome };
                    {
                        let mut chain = model_state
                            .lock()
                            .unwrap_or_else(|error| error.into_inner());
                        chain.remaining.remove(0);
                        chain.active = Some(next.clone());
                    }
                    request.model = Some(next);
                };
                let lane_failed = match &outcome {
                    // The whole model chain exhausted on this lane — the
                    // failure is provider-side, so the caller's next
                    // profile is the right escalation, not another lane.
                    Err(error) if model_failure(error) => break 'lanes outcome,
                    Err(_) => true,
                    // A lane that reports the turn as failed without an
                    // error channel — cli's nonzero exit surfaces as
                    // `stop_reason: "error"` — could not serve this turn.
                    Ok(result) => result.stop_reason == "error",
                };
                if !lane_failed {
                    break 'lanes outcome;
                }
                // A failure no model change can fix — transport, protocol,
                // a crashed server. Reopen the session on the next lane
                // and replay the turn there.
                loop {
                    let mut lane = lane.lock().await;
                    lane.index += 1;
                    let Some(spec) = lane.specs.get(lane.index).cloned() else {
                        break 'lanes outcome;
                    };
                    // Re-project the caller's request against this lane:
                    // selectors the previous lane applied at establishment
                    // may belong on turns here, and vice versa.
                    let active = model_state
                        .lock()
                        .unwrap_or_else(|error| error.into_inner())
                        .active
                        .clone();
                    let (new_open, turn_model, turn_effort) =
                        lane_open(&spec, &lane.requested, &profile, active);
                    let adapter = lane.adapter(lane.index);
                    let reopen = adapter.open_session(new_open.clone(), lane.events.clone());
                    {
                        let mut guard = session.write().await;
                        let _ = guard.close().await;
                        match reopen.await {
                            Ok(new_session) => {
                                *guard = new_session;
                                lane.open = new_open;
                            }
                            Err(open_error) if model_failure(&open_error) => {
                                break 'lanes Err(open_error);
                            }
                            Err(_) => continue,
                        }
                    }
                    *turn_model_default
                        .lock()
                        .unwrap_or_else(|error| error.into_inner()) = turn_model;
                    *turn_effort_default
                        .lock()
                        .unwrap_or_else(|error| error.into_inner()) = turn_effort;
                    drop(lane);
                    stateless.store(matches!(spec.lane, LaneSpec::Cli(_)), Ordering::Relaxed);
                    instructions_sent.store(false, Ordering::Relaxed);
                    // The fresh session never heard this profile: if the
                    // composed text skipped instructions on the old lane's
                    // behalf, re-compose them into the replayed turn.
                    if record_instructions && !instructions_included {
                        request.text =
                            compose_prompt(profile.instructions.as_deref(), &request.text);
                        instructions_included = true;
                    }
                    continue 'lanes;
                }
            };
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
        self.session.read().await.cancel().await
    }

    /// Close the session and terminate the agent process tree.
    pub async fn close(&self) -> a4aa::Result<()> {
        self.session.read().await.close().await
    }
}
