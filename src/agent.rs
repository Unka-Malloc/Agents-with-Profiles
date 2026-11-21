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

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use a4aa::{
    Adapter, AgentEvent, AgentSession, AgentSpec, BeforeTurn, Capabilities, Direction, EnvPolicy,
    LaneKind, LaneSpec, Observer, PermissionHandler, SecretResolver, SelectorPhase,
    SelectorSupport, SessionRequest, SharedObserver, TokenUsage, TurnOutcome, TurnRequest,
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

/// The sealing recipe for one agent's command line: the flags that close
/// every ambient code and instruction channel — extension discovery,
/// skill and prompt-template loading, context-file injection — while the
/// builtins the session still needs load explicitly by name. An agent
/// without a recipe cannot serve a strict session at all.
fn strict_recipe(spec: &AgentSpec) -> Option<Vec<String>> {
    let program = std::path::Path::new(&spec.command.program)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    match (spec.id.as_str(), program) {
        ("pi", _) | (_, "pi") => Some(
            [
                "--no-extensions",
                "-e",
                "builtin:mcp",
                "-e",
                "builtin:codemode",
                "-e",
                "builtin:tool-search",
                "--no-skills",
                "--no-prompt-templates",
                "--no-context-files",
            ]
            .iter()
            .map(|flag| flag.to_string())
            .collect(),
        ),
        _ => None,
    }
}

/// Every argv vector a spec can place on the agent's command line: the
/// spawn prefix plus, for the `cli` lane, the turn template and its
/// resume/open variants. `resume_spawn_args` replaces the spawn prefix
/// wholesale on a resumed turn, so it both gets its own seal copy and
/// is scanned like any other vector.
fn spec_arg_vectors(spec: &mut AgentSpec) -> Vec<&mut Vec<String>> {
    let mut vectors: Vec<&mut Vec<String>> = vec![&mut spec.command.args];
    if let LaneSpec::Cli(lane) = &mut spec.lane {
        vectors.extend([
            &mut lane.args,
            &mut lane.resume_args,
            &mut lane.resume_spawn_args,
            &mut lane.open_args,
        ]);
    }
    vectors
}

/// Apply strict sealing to one lane spec: the recipe lands on the spawn
/// prefix — and on `resume_spawn_args` when that vector replaces the
/// prefix on resumed turns — while every argv vector is refused when it
/// declares an ambient channel a seal cannot honor: an explicit
/// non-builtin `-e` extension, `--skill`, or `--prompt-template`.
/// Sealing over such a spec would silently change the document's
/// meaning instead of honoring it.
fn seal_lane_spec(spec: &mut AgentSpec) -> a4aa::Result<()> {
    let recipe = strict_recipe(spec).ok_or_else(|| {
        a4aa::AdapterError::permanent("agent_unsealable", "profile/strict").with_detail(format!(
            "agent `{}` (`{}`) has no declared strict-sealing recipe",
            spec.id, spec.command.program
        ))
    })?;
    let id = spec.id.clone();
    for args in spec_arg_vectors(spec) {
        for (index, arg) in args.iter().enumerate() {
            let ambient = matches!(arg.as_str(), "--skill" | "--prompt-template")
                || arg.starts_with("--skill=")
                || arg.starts_with("--prompt-template=")
                || (matches!(arg.as_str(), "-e" | "--extension")
                    && !args
                        .get(index + 1)
                        .is_some_and(|path| path.starts_with("builtin:")))
                || arg
                    .strip_prefix("--extension=")
                    .or_else(|| arg.strip_prefix("-e="))
                    .is_some_and(|path| !path.starts_with("builtin:"));
            if ambient {
                return Err(a4aa::AdapterError::permanent(
                    "agent_unsealable",
                    "profile/strict",
                )
                .with_detail(format!(
                    "agent `{id}` declares ambient-surface arg `{arg}` a strict session cannot honor"
                )));
            }
        }
    }
    let seal = |args: &mut Vec<String>| {
        if !recipe.iter().all(|flag| args.contains(flag)) {
            args.extend(recipe.iter().cloned());
        }
    };
    seal(&mut spec.command.args);
    if let LaneSpec::Cli(lane) = &mut spec.lane {
        if !lane.resume_spawn_args.is_empty() {
            seal(&mut lane.resume_spawn_args);
        }
    }
    Ok(())
}

/// Fold the profile into one spec: environment entries merge into the
/// command's environment policy `set`, profile MCP servers append to any
/// lane that carries an MCP surface (`acp` session/new, `serve` config
/// document, `rpc` process registration, `cli` admission-refusal list). A
/// lane without a native registration surface keeps the list, so the adapter
/// refuses it at open instead of the profile layer silently dropping its
/// tools.
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
        LaneSpec::Rpc(rpc) => &mut rpc.mcp_servers,
        LaneSpec::Cli(cli) => &mut cli.mcp_servers,
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

    /// Seal every lane variant for a strict session: each spec gets the
    /// agent's isolation recipe — ambient extension, skill, prompt
    /// template and context-file channels closed — so the declared
    /// surface is the surface that actually runs. Every variant must
    /// seal: a mid-session lane switch landing on an unsealed lane would
    /// silently drop the guarantee, so one unsealable variant fails the
    /// whole agent.
    pub fn strict(mut self) -> a4aa::Result<Self> {
        for spec in &mut self.specs {
            seal_lane_spec(spec)?;
        }
        Ok(self)
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
        self.adapter(&self.specs[0], &ReservationSlot::default())
            .capabilities()
    }

    fn adapter(&self, spec: &AgentSpec, reservations: &ReservationSlot) -> Adapter {
        let mut adapter = Adapter::new(spec.clone())
            .secrets(self.secrets.clone())
            .observer(self.observer.clone());
        if let Some(handler) = &self.permission {
            adapter = adapter.permission(handler.clone());
        }
        if let Some(hook) = &self.before_turn {
            adapter = adapter.before_turn(Arc::new(ReservationProbe {
                hook: hook.clone(),
                slot: reservations.clone(),
            }));
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
        // The session's reservation slot: every adapter this session runs
        // reads the current turn's verdict cell through it.
        let reservations = ReservationSlot::default();
        let mut lane_error = None;
        for (lane_index, spec) in self.specs.iter().enumerate() {
            let (mut lane_request, turn_model_default, turn_effort_default) =
                lane_open(spec, &request, &self.profile, chain.first().cloned());
            // A requested model the lane cannot express at establishment
            // advances the chain before the first turn is even attempted.
            let session = loop {
                // Every open attempt gets its own event tap: the session's
                // relay drains it into the caller's stream for the whole
                // lifetime and counts each armed attempt's produced-output
                // evidence for the fallback decision.
                let (tap_tx, tap_rx) = mpsc::unbounded_channel();
                // Every open attempt also gets a dispatch probe: the wire
                // tap the turn driver later reads to learn whether an
                // uncertain failure's prompt frame ever left the machine.
                let probe = DispatchProbe::new(self.observer.clone(), spec);
                let observer: SharedObserver = Some(probe.clone());
                let result = self
                    .adapter(spec, &reservations)
                    .observer(observer)
                    .open_session(lane_request.clone(), tap_tx)
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
                    Ok(session) => break Some((session, lane_request, tap_rx, probe)),
                }
            };
            let Some((session, effective_request, tap_rx, probe)) = session else {
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
                    relay: EventRelay::spawn(tap_rx, events.clone(), probe.clone()),
                    secrets: self.secrets.clone(),
                    observer: self.observer.clone(),
                    permission: self.permission.clone(),
                    before_turn: self.before_turn.clone(),
                    probe,
                    reservations,
                })),
                stateless_lane: Arc::new(AtomicBool::new(matches!(spec.lane, LaneSpec::Cli(_)))),
                instructions_sent: Arc::new(AtomicBool::new(false)),
                turn_lock: Arc::new(Mutex::new(())),
                closing: Arc::new(AtomicBool::new(false)),
                cancellation: Arc::new(AtomicU64::new(0)),
                lifecycle: Arc::new(tokio::sync::Notify::new()),
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
    /// The active lane's event path: a forwarder task drains the lane's
    /// tap into the caller's stream for the session's whole lifetime —
    /// events between turns are delivered instead of stalling until the
    /// next prompt — and marks each armed attempt's output evidence.
    /// Replaced on every lane switch.
    relay: EventRelay,
    secrets: Arc<dyn SecretResolver>,
    observer: SharedObserver,
    permission: Option<Arc<dyn PermissionHandler>>,
    before_turn: Option<Arc<dyn BeforeTurn>>,
    /// The active lane's dispatch probe: installed as the adapter's
    /// observer wrapper and replaced on every lane switch.
    probe: Arc<DispatchProbe>,
    /// The session's reservation slot: the turn driver installs a fresh
    /// [`TurnReservation`] cell per logical turn, and every lane adapter
    /// this session runs reads the current cell through the slot.
    reservations: ReservationSlot,
}

impl LaneState {
    /// Build the adapter for lane `index` with its fresh dispatch probe.
    /// The probe is returned alongside rather than installed: it becomes
    /// this lane's probe only when the session it was built for opens —
    /// a refused open leaves the installed session and probe untouched.
    fn adapter(&self, index: usize) -> (Adapter, Arc<DispatchProbe>) {
        let probe = DispatchProbe::new(self.observer.clone(), &self.specs[index]);
        let observer: SharedObserver = Some(probe.clone());
        let mut adapter = Adapter::new(self.specs[index].clone())
            .secrets(self.secrets.clone())
            .observer(observer);
        if let Some(handler) = &self.permission {
            adapter = adapter.permission(handler.clone());
        }
        if let Some(hook) = &self.before_turn {
            adapter = adapter.before_turn(Arc::new(ReservationProbe {
                hook: hook.clone(),
                slot: self.reservations.clone(),
            }));
        }
        (adapter, probe)
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
    // A rejected credential makes this profile's provider just as unusable
    // as an exhausted one — the turn escalates to the next profile.
    "unauthorized",
    "authentication_error",
    "invalid 'authorization' header",
];

/// A failure whose effect is uncertain may already have produced and billed
/// output. Opening a session or applying a selector happens before a model
/// turn, so those stages keep their existing progression; a prompt-stage
/// failure is propagated instead of replaying a possibly duplicated turn.
fn uncertain_turn(error: &a4aa::AdapterError) -> bool {
    error.uncertain_effect
        && !matches!(
            error.stage,
            "session/open" | "session/set_model" | "session/set_config_option"
        )
}

/// The reservation verdict for one logical turn. The `before_turn` hook
/// runs at most once per turn: its outcome is recorded here and reused by
/// every lane and model attempt of the same turn — a reservation is never
/// booked twice for one turn, and the driver reads the recorded verdict
/// to propagate a refusal before either fallback decision can re-invoke
/// the hook.
#[derive(Default)]
struct TurnReservation {
    /// `None` until the hook first runs this turn.
    outcome: std::sync::Mutex<Option<a4aa::Result<()>>>,
    /// The session's cancellation counter and the mark this turn was
    /// submitted under: a reservation that resolves after cancel moved
    /// past the mark must still refuse the prompt — the lane was idle
    /// when cancel ran, so its own cancel path never saw the turn.
    cancellation: Option<(Arc<AtomicU64>, u64)>,
}

impl TurnReservation {
    /// Whether the reservation refused this turn.
    fn refused(&self) -> bool {
        self.outcome
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .as_ref()
            .is_some_and(|result| result.is_err())
    }

    /// Whether the session's cancellation moved past this turn's mark.
    fn cancelled(&self) -> bool {
        self.cancellation
            .as_ref()
            .is_some_and(|(counter, mark)| counter.load(Ordering::Acquire) != *mark)
    }
}

/// The per-session slot the turn driver swaps each logical turn: every
/// adapter built for the session reads the current turn's cell through
/// it, so a hook verdict never crosses turns — and never crosses
/// sessions, since each `VirtualSession` owns its own slot.
type ReservationSlot = Arc<std::sync::Mutex<Option<Arc<TurnReservation>>>>;

/// Wraps the host's `BeforeTurn` reservation so each verdict binds to the
/// logical turn that produced it: the probe reads the session's current
/// turn cell and reuses a verdict already recorded for this turn — a
/// granted reservation carries across a pre-dispatch lane fallback and a
/// model retry without invoking the hook again.
struct ReservationProbe {
    hook: Arc<dyn BeforeTurn>,
    slot: ReservationSlot,
}

#[async_trait::async_trait]
impl BeforeTurn for ReservationProbe {
    async fn run(&self) -> a4aa::Result<()> {
        let turn = self
            .slot
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone();
        if let Some(turn) = &turn {
            // Admission itself carries cancellation: a verdict already
            // recorded this turn cannot dispatch a prompt the caller
            // cancelled while the lane sat idle.
            if turn.cancelled() {
                return Err(a4aa::AdapterError::permanent(
                    "agent_turn_cancelled",
                    "session/prompt",
                ));
            }
            if let Some(outcome) = turn
                .outcome
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .clone()
            {
                return outcome;
            }
        }
        let outcome = self.hook.run().await;
        // Cancellation that landed while the hook was in flight still
        // refuses the prompt: the lane would dispatch it the moment the
        // reservation returned.
        if turn.as_ref().is_some_and(|turn| turn.cancelled()) {
            return Err(a4aa::AdapterError::permanent(
                "agent_turn_cancelled",
                "session/prompt",
            ));
        }
        if let Some(turn) = turn {
            *turn
                .outcome
                .lock()
                .unwrap_or_else(|error| error.into_inner()) = Some(outcome.clone());
        }
        outcome
    }
}

/// One lane's event path for the session's whole lifetime. The lane's tap
/// is a private channel its adapter fills; the relay task drains it as
/// events arrive — a notification between turns reaches the caller
/// immediately rather than stalling until the next prompt, and nothing
/// queues unclaimed when the lane closes. While a turn attempt is *armed*
/// the relay also marks that attempt's output evidence; events outside
/// the window are delivered but never counted as the turn's.
#[derive(Clone)]
struct EventRelay {
    /// The armed-window flags the forwarder updates and the driver reads.
    /// Held separately so the forwarder task carries no reference back
    /// through the relay's own barrier channel.
    evidence: Arc<RelayEvidence>,
    /// Barrier requests: the driver asks the forwarder to finish relaying
    /// whatever the lane has queued so far, then reads settled evidence.
    flush: mpsc::UnboundedSender<tokio::sync::oneshot::Sender<()>>,
}

/// The armed-window flags one lane attempt produces.
struct RelayEvidence {
    /// Open only while a turn attempt is in flight on this lane.
    armed: AtomicBool,
    /// Any event relayed while armed — the attempt produced output.
    output: AtomicBool,
    /// A billable event relayed while armed — the output was paid work.
    billable: AtomicBool,
}

impl EventRelay {
    /// Start draining `tap` into `events` on the session runtime. The
    /// forwarder keeps running until the lane's tap closes *and* the last
    /// relay handle drops — a dead lane's queued leftovers still reach
    /// the caller. `probe` carries the lane's notification carriers so a
    /// raw event's billable check matches the wire journal's reading.
    fn spawn(
        mut tap: mpsc::UnboundedReceiver<AgentEvent>,
        events: mpsc::UnboundedSender<AgentEvent>,
        probe: Arc<DispatchProbe>,
    ) -> Self {
        let (flush, mut barriers) = mpsc::unbounded_channel::<tokio::sync::oneshot::Sender<()>>();
        let evidence = Arc::new(RelayEvidence {
            armed: AtomicBool::new(false),
            output: AtomicBool::new(false),
            billable: AtomicBool::new(false),
        });
        let forwarding = evidence.clone();
        tokio::spawn(async move {
            let mut tap = Some(&mut tap);
            let mut barriers_open = true;
            loop {
                let event = async {
                    match &mut tap {
                        Some(rx) => rx.recv().await,
                        None => std::future::pending().await,
                    }
                };
                let barrier = async {
                    if barriers_open {
                        barriers.recv().await
                    } else {
                        std::future::pending().await
                    }
                };
                tokio::select! {
                    biased;
                    barrier = barrier => match barrier {
                        // Settle the queue before the driver reads
                        // evidence: relay everything already received.
                        Some(ack) => {
                            if let Some(rx) = &mut tap {
                                while let Ok(event) = rx.try_recv() {
                                    Self::forward(&forwarding, &events, &probe, event);
                                }
                            }
                            let _ = ack.send(());
                        }
                        None => barriers_open = false,
                    },
                    event = event => match event {
                        Some(event) => Self::forward(&forwarding, &events, &probe, event),
                        None => tap = None,
                    },
                }
                if tap.is_none() && !barriers_open {
                    break;
                }
            }
        });
        Self { evidence, flush }
    }

    /// Relay one event to the caller; count it as the armed attempt's
    /// evidence only inside the attempt window — and only once the
    /// attempt's prompt actually dispatched, the same attribution the
    /// wire journal applies: traffic a lane emits at session open can
    /// still be in flight when the window opens and is never the turn's
    /// output. Lanes without an observable dispatch point report
    /// `dispatched` unconditionally.
    fn forward(
        evidence: &RelayEvidence,
        events: &mpsc::UnboundedSender<AgentEvent>,
        probe: &DispatchProbe,
        event: AgentEvent,
    ) {
        if evidence.armed.load(Ordering::Acquire) && probe.dispatched() {
            evidence.output.store(true, Ordering::Release);
            if turn_output(&event, probe) {
                evidence.billable.store(true, Ordering::Release);
            }
        }
        let _ = events.send(event);
    }

    /// Open the next attempt's evidence window.
    fn arm(&self) {
        self.evidence.output.store(false, Ordering::Release);
        self.evidence.billable.store(false, Ordering::Release);
        self.evidence.armed.store(true, Ordering::Release);
    }

    /// Wait until everything the lane queued so far has been relayed, so
    /// evidence read afterwards reflects the complete attempt.
    async fn flush(&self) {
        let (settled, wait) = tokio::sync::oneshot::channel();
        if self.flush.send(settled).is_ok() {
            let _ = wait.await;
        }
    }

    /// The armed window's billable flag — read after `flush` settled the
    /// queue so a just-arrived event cannot be missed.
    fn billable(&self) -> bool {
        self.evidence.billable.load(Ordering::Acquire)
    }

    /// Close the window and read the attempt's `(output, billable)`
    /// evidence.
    fn disarm(&self) -> (bool, bool) {
        self.evidence.armed.store(false, Ordering::Release);
        (
            self.evidence.output.load(Ordering::Acquire),
            self.evidence.billable.load(Ordering::Acquire),
        )
    }
}

/// Side-effect-free wire evidence for an rpc lane. The transport observer
/// sees every outbound frame before it is written and every inbound frame
/// before it reaches the lane's notification queue — both strictly before
/// the corresponding call resolves — so the journal the probe keeps is
/// evidence the lane's own delivery task cannot outrun: a `Sent` frame
/// carrying the lane's prompt method marks dispatch intent, the *last*
/// `Sent` frame names the write the transport failed on a `writer_failed`
/// error, and an inbound notification is attributed to the turn at wire
/// receipt rather than at tap dequeue. Every record forwards to the
/// host's own observer.
struct DispatchProbe {
    inner: SharedObserver,
    /// The lane's prompt method — `None` for lanes whose dispatch is not
    /// an rpc frame (cli, acp, serve).
    method: Option<String>,
    /// The frame field the method name occupies: `method` for stock
    /// JSON-RPC, the envelope's `method_field` for non-stock envelopes.
    method_field: String,
    /// A non-stock envelope is declared: inbound responses are
    /// discriminated by marker rather than the stock `id`/result shape.
    enveloped: bool,
    /// The envelope's `response_method` marker value and the field the
    /// marker occupies (`discriminant_field`, falling back to
    /// `method_field`) — the same response discrimination the peer's
    /// dispatch applies.
    response_method: Option<String>,
    discriminant_field: Option<String>,
    /// A declared notification payload field — the envelope's
    /// `data_field`. When declared but absent on a frame, the frame
    /// itself is the payload document (flat envelopes). Inbound only:
    /// it never locates an outbound call's parameters.
    data_field: Option<String>,
    /// Whether the prompt call's parameters merge into the outbound
    /// frame's top level: the lane's encoder inlines an *object* params
    /// document under an inline envelope (`params_inline`) and puts
    /// every other shape under `params` — nested envelopes included.
    prompt_at_root: bool,
    /// The lane's stream-text notification method (`text_event_method`)
    /// and the dotted path to its payload (`text_path`), mirroring how
    /// the lane normalizes stream text.
    text_event_method: Option<String>,
    text_path: Option<String>,
    /// The lane's configured usage carrier (`usage_path`); when unset the
    /// conventional `params.usage` / `params.message.usage` carriers are
    /// read instead — the same precedence the lane's own ingest applies.
    usage_path: Option<String>,
    /// Whether the lane's prompt params template carries `{prompt}` in at
    /// least one leaf the wire can verify — the prompt call is then the
    /// same-method frame whose configured payload fields hold this
    /// attempt's text. A configured pre-turn call can share the prompt
    /// method; its frame must not read as dispatched. When the template
    /// offers no verifiable prompt position the method name is all the
    /// wire offers and a matching request counts conservatively.
    prompt_carries_text: bool,
    /// Every verifiable `{prompt}` position in the prompt params
    /// template: the path from the params root to a leaf that declares
    /// the placeholder and nothing else, plus the leaf's literal text so
    /// the expected wire value is the leaf with `{prompt}` replaced by
    /// this attempt's armed text.
    prompt_slots: Vec<(Vec<String>, String)>,
    /// The substituted value each `prompt_slots` leaf must carry on the
    /// prompt call's frame — computed at `reset` from this attempt's
    /// prompt text, so the check is an exact payload comparison rather
    /// than a text search over the frame.
    expected_prompts: std::sync::Mutex<Vec<String>>,
    /// Set once this attempt's prompt frame was observed on the wire. The
    /// turn driver resets it per attempt — a stale flag from an earlier
    /// turn or lane would misreport a pre-dispatch failure as dispatched.
    dispatched: AtomicBool,
    /// Whether the most recent `Sent` frame was the prompt. The transport
    /// writer stops at the failed write, so on a `writer_failed` report
    /// the last observed `Sent` names the failed write.
    last_sent_was_prompt: AtomicBool,
    /// An in-turn notification reached the wire: every id-less inbound
    /// frame after dispatch becomes a lane event, so receipt is output
    /// evidence even when the lane's delivery task has not forwarded it
    /// to the tap yet.
    turn_event: AtomicBool,
    /// An in-turn notification carried billable evidence — stream text at
    /// the lane's text carrier or a normalized usage record with a
    /// nonzero counter.
    turn_billable: AtomicBool,
}

impl DispatchProbe {
    fn new(inner: SharedObserver, spec: &AgentSpec) -> Arc<Self> {
        let mut probe = Self {
            inner,
            method: None,
            method_field: "method".to_owned(),
            enveloped: false,
            response_method: None,
            discriminant_field: None,
            data_field: None,
            text_event_method: None,
            text_path: None,
            usage_path: None,
            prompt_carries_text: false,
            prompt_slots: Vec::new(),
            prompt_at_root: false,
            expected_prompts: std::sync::Mutex::new(Vec::new()),
            dispatched: AtomicBool::new(false),
            last_sent_was_prompt: AtomicBool::new(false),
            turn_event: AtomicBool::new(false),
            turn_billable: AtomicBool::new(false),
        };
        if let LaneSpec::Rpc(rpc) = &spec.lane {
            if !rpc.prompt.method.is_empty() {
                probe.method = Some(rpc.prompt.method.clone());
            }
            if let Some(envelope) = &rpc.envelope {
                probe.method_field = envelope.method_field.clone();
                probe.enveloped = true;
                probe.response_method = envelope.response_method.clone();
                probe.discriminant_field = envelope.discriminant_field.clone();
                probe.data_field = envelope.data_field.clone();
            }
            probe.text_event_method = rpc.text_event_method.clone();
            probe.text_path = rpc.text_path.clone();
            probe.usage_path = rpc.usage_path.clone();
            probe.prompt_slots = prompt_slots(&rpc.prompt.params);
            probe.prompt_carries_text = !probe.prompt_slots.is_empty();
            probe.prompt_at_root = rpc
                .envelope
                .as_ref()
                .is_some_and(|envelope| envelope.params_inline)
                && rpc.prompt.params.is_object();
        }
        Arc::new(probe)
    }

    /// Whether this attempt's prompt frame reached the wire. `true` for
    /// lanes without an observable dispatch point — their uncertain
    /// failures keep the conservative reading.
    fn dispatched(&self) -> bool {
        self.method.is_none() || self.dispatched.load(Ordering::Acquire)
    }

    /// Positive evidence a failed attempt's selector reached the lane:
    /// the lane has an observable dispatch point and this error's
    /// verdict cannot rule out delivery of the prompt frame that
    /// carried it. A `Sent` record is dispatch intent, not acceptance —
    /// it precedes the write — so a prompt whose write the transport
    /// confirmed failed, and every failure on a lane with no dispatch
    /// probe at all, offers no such evidence.
    fn dispatch_confirmed(&self, error: &a4aa::AdapterError) -> bool {
        self.method.is_some()
            && matches!(
                dispatch_verdict(error, self),
                DispatchVerdict::PossiblyDispatched
            )
    }

    /// Whether the last frame the transport writer attempted was this
    /// attempt's prompt — on a `writer_failed` report, the write the
    /// transport confirmed failed.
    fn last_write_was_prompt(&self) -> bool {
        self.last_sent_was_prompt.load(Ordering::Acquire)
    }

    /// Whether a same-method request frame is this attempt's prompt call
    /// rather than a configured pre-turn call sharing the method. The
    /// lane builds the prompt frame from the prompt params template:
    /// when that template declares `{prompt}` in a verifiable leaf, the
    /// frame must carry this attempt's substituted text at exactly those
    /// payload positions — the method name and every other string in the
    /// frame are never read as text, so a short prompt that collides
    /// with the method name cannot mark a configuration call as
    /// dispatched. When the template offers no verifiable position the
    /// wire cannot tell the calls apart and the request counts
    /// conservatively.
    fn is_prompt_frame(&self, frame: &serde_json::Value) -> bool {
        frame.get("id").is_some()
            && frame
                .get(self.method_field.as_str())
                .and_then(serde_json::Value::as_str)
                .is_some_and(|method| Some(method) == self.method.as_deref())
            && (!self.prompt_carries_text || self.carries_armed_prompt(frame))
    }

    /// Whether the frame's params carry this attempt's prompt text at
    /// every verifiable template position. Outbound parameter placement
    /// mirrors the lane's own encoder: an inline envelope merges an
    /// object params document into the frame root; every other shape —
    /// a nested envelope or a non-object params value — sits under
    /// `params`.
    fn carries_armed_prompt(&self, frame: &serde_json::Value) -> bool {
        let params = if self.prompt_at_root {
            frame
        } else {
            match frame.get("params") {
                Some(params) => params,
                None => return false,
            }
        };
        let expected = self
            .expected_prompts
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        self.prompt_slots.len() == expected.len()
            && self
                .prompt_slots
                .iter()
                .zip(expected.iter())
                .all(|((path, _), expected)| {
                    slot_at(params, path)
                        .and_then(serde_json::Value::as_str)
                        .is_some_and(|value| value == expected)
                })
    }

    /// In-turn output evidence attributed at wire receipt.
    fn wire_event(&self) -> bool {
        self.turn_event.load(Ordering::Acquire)
    }

    /// In-turn billable evidence attributed at wire receipt.
    fn wire_billable(&self) -> bool {
        self.turn_billable.load(Ordering::Acquire)
    }

    /// Open the next physical attempt's journal: stale dispatch and
    /// event records never carry into it, and the attempt's prompt text
    /// resolves the payload values the prompt call's frame must carry.
    fn reset(&self, prompt: &str) {
        *self
            .expected_prompts
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = self
            .prompt_slots
            .iter()
            .map(|(_, leaf)| leaf.replace("{prompt}", prompt))
            .collect();
        self.dispatched.store(false, Ordering::Release);
        self.last_sent_was_prompt.store(false, Ordering::Release);
        self.turn_event.store(false, Ordering::Release);
        self.turn_billable.store(false, Ordering::Release);
    }

    /// Mirror of the lane's own evidence normalization over a wire frame:
    /// the method and payload sit at the envelope's declared fields when
    /// one is configured, at `method`/`params` otherwise.
    fn billable_frame(&self, frame: &serde_json::Value) -> bool {
        let method = frame
            .get(self.method_field.as_str())
            .and_then(serde_json::Value::as_str);
        let params = match &self.data_field {
            Some(field) => frame.get(field).unwrap_or(frame),
            None => frame.get("params").unwrap_or(&serde_json::Value::Null),
        };
        self.billable_notification(method, params)
    }

    /// The event-tap form of the same normalization: the lane rebuilds a
    /// relayed notification as a `{method, params}` document already
    /// decoded from the frame, so its carriers sit at their conventional
    /// positions regardless of the wire envelope.
    fn billable_raw(&self, raw: &serde_json::Value) -> bool {
        self.billable_notification(
            raw.get("method").and_then(serde_json::Value::as_str),
            raw.get("params").unwrap_or(&serde_json::Value::Null),
        )
    }

    /// An id-less payload is billable when its method carries stream text
    /// at the declared carrier, or a usage record at the configured
    /// `usage_path` — the conventional carriers when unset — has a
    /// nonzero counter.
    fn billable_notification(&self, method: Option<&str>, params: &serde_json::Value) -> bool {
        if method.is_some()
            && method == self.text_event_method.as_deref()
            && notification_text(params, self.text_path.as_deref()).is_some()
        {
            return true;
        }
        self.usage_path
            .as_deref()
            .and_then(|path| extract_path(params, path))
            .or_else(|| params.get("usage"))
            .or_else(|| extract_path(params, "message.usage"))
            .is_some_and(normalized_billable)
    }
}

/// Every `{prompt}`-carrying leaf position in a params template: the
/// path from the params root (object keys and array indices) and the
/// leaf's literal text. A leaf that also declares another placeholder —
/// `{model}`, `{sessionId}`, `{cwd}` — is unusable: its substituted
/// value is unknowable at the probe, so only the leaves that isolate
/// the prompt give the frame check a verifiable position.
fn prompt_slots(template: &serde_json::Value) -> Vec<(Vec<String>, String)> {
    fn walk(
        value: &serde_json::Value,
        path: &mut Vec<String>,
        slots: &mut Vec<(Vec<String>, String)>,
    ) {
        match value {
            serde_json::Value::String(leaf)
                if leaf.contains("{prompt}") && !other_placeholder(leaf) =>
            {
                slots.push((path.clone(), leaf.clone()));
            }
            serde_json::Value::Array(items) => {
                for (index, item) in items.iter().enumerate() {
                    path.push(index.to_string());
                    walk(item, path, slots);
                    path.pop();
                }
            }
            serde_json::Value::Object(map) => {
                for (key, item) in map {
                    path.push(key.clone());
                    walk(item, path, slots);
                    path.pop();
                }
            }
            _ => {}
        }
    }
    let mut path = Vec::new();
    let mut slots = Vec::new();
    walk(template, &mut path, &mut slots);
    slots
}

/// Whether `leaf` declares a placeholder other than `{prompt}` — the
/// same `{name}` tokenization the lane's substitution applies: an
/// unmatched or re-opened brace is literal text, not a token.
fn other_placeholder(leaf: &str) -> bool {
    let mut rest = leaf;
    while let Some(open) = rest.find('{') {
        let tail = &rest[open + 1..];
        match tail.find(['{', '}']) {
            Some(end) if tail.as_bytes()[end] == b'}' => {
                if &tail[..end] != "prompt" {
                    return true;
                }
                rest = &tail[end + 1..];
            }
            _ => rest = tail,
        }
    }
    false
}

/// The value at `path` inside a params document — each segment is an
/// object key, or an array index when the object lookup misses —
/// mirroring the positions the template walk recorded.
fn slot_at<'a>(mut value: &'a serde_json::Value, path: &[String]) -> Option<&'a serde_json::Value> {
    for segment in path {
        value = value.get(segment.as_str()).or_else(|| {
            segment
                .parse::<usize>()
                .ok()
                .and_then(|index| value.get(index))
        })?;
    }
    Some(value)
}

/// Walk a dotted path into a JSON value.
fn extract_path<'a>(value: &'a serde_json::Value, path: &str) -> Option<&'a serde_json::Value> {
    let mut cursor = value;
    for segment in path.split('.') {
        cursor = cursor.get(segment)?;
    }
    Some(cursor)
}

/// Mirror of the lane's notification text extraction, reduced to a
/// presence check: a declared `text_path` resolves a string leaf or an
/// array whose `{type:"text",text}` blocks concatenate to text; otherwise
/// the conventional fields (`text`, `content`, `delta`, `item.text`) or a
/// bare string.
fn notification_text(params: &serde_json::Value, text_path: Option<&str>) -> Option<()> {
    if let Some(path) = text_path {
        return match extract_path(params, path) {
            Some(serde_json::Value::String(_)) => Some(()),
            Some(serde_json::Value::Array(blocks)) => blocks
                .iter()
                .filter(|block| {
                    block
                        .get("type")
                        .is_none_or(|kind| kind.as_str() == Some("text"))
                })
                .any(|block| {
                    block
                        .get("text")
                        .and_then(serde_json::Value::as_str)
                        .is_some_and(|text| !text.is_empty())
                })
                .then_some(()),
            _ => None,
        };
    }
    params
        .get("text")
        .or_else(|| params.get("content"))
        .or_else(|| params.get("delta"))
        .and_then(serde_json::Value::as_str)
        .map(|_| ())
        .or_else(|| extract_path(params, "item.text")?.as_str().map(|_| ()))
        .or_else(|| params.as_str().map(|_| ()))
}

/// Mirror of the lane's billable-usage normalization: the aliased
/// counters folded into `TokenUsage` — input, cached input, cached write,
/// output, thought, total — decide the served question; a record carrying
/// only unknown keys is bookkeeping, not billed work.
fn normalized_billable(usage: &serde_json::Value) -> bool {
    const COUNTERS: &[&[&str]] = &[
        &[
            "input",
            "input_tokens",
            "inputTokens",
            "prompt_tokens",
            "promptTokens",
        ],
        &[
            "cacheRead",
            "cached_input_tokens",
            "cachedInputTokens",
            "cache_read_input_tokens",
            "cacheReadInputTokens",
            "cache_read_tokens",
        ],
        &[
            "cacheWrite",
            "cached_write_tokens",
            "cachedWriteTokens",
            "cache_write_tokens",
        ],
        &[
            "output",
            "output_tokens",
            "outputTokens",
            "completion_tokens",
            "completionTokens",
        ],
        &["thought_tokens", "thoughtTokens", "reasoning_tokens"],
        &["total_tokens", "totalTokens"],
    ];
    usage.as_object().is_some_and(|usage| {
        COUNTERS.iter().any(|aliases| {
            aliases.iter().any(|name| {
                usage
                    .get(*name)
                    .and_then(serde_json::Value::as_u64)
                    .is_some_and(|tokens| tokens > 0)
            })
        })
    })
}

impl Observer for DispatchProbe {
    fn record(&self, source: &'static str, direction: Direction, bytes: &[u8]) {
        if let Some(inner) = &self.inner {
            inner.record(source, direction, bytes);
        }
        if source != "rpc" || self.method.is_none() {
            return;
        }
        match direction {
            // Every `Sent` refreshes which frame the writer is on: the
            // record precedes the write, so the last observed `Sent`
            // names the write a `writer_failed` report refers to.
            Direction::Sent => {
                let prompt = serde_json::from_slice::<serde_json::Value>(bytes)
                    .ok()
                    .is_some_and(|frame| self.is_prompt_frame(&frame));
                self.last_sent_was_prompt.store(prompt, Ordering::Release);
                if prompt {
                    self.dispatched.store(true, Ordering::Release);
                }
            }
            // Inbound frames: a response settles a pending call, a request
            // (id + method) or a notification (id-less) is agent activity.
            // Either is the turn's output only once the prompt reached the
            // wire — receipt cannot precede the `Sent` record, so
            // dispatch-gated attribution is exact even when the lane's
            // delivery task has not forwarded the frame to the tap yet.
            Direction::Received => {
                if !self.dispatched.load(Ordering::Acquire) {
                    return;
                }
                let Ok(frame) = serde_json::from_slice::<serde_json::Value>(bytes) else {
                    return;
                };
                let id = frame.get("id");
                let is_response = if self.enveloped {
                    id.is_some()
                        && self.response_method.as_deref().is_some_and(|marker| {
                            frame
                                .get(
                                    self.discriminant_field
                                        .as_deref()
                                        .unwrap_or(self.method_field.as_str()),
                                )
                                .and_then(serde_json::Value::as_str)
                                == Some(marker)
                        })
                } else {
                    id.is_some()
                        && frame.get("method").is_none()
                        && (frame.get("result").is_some() || frame.get("error").is_some())
                };
                if is_response {
                    return;
                }
                self.turn_event.store(true, Ordering::Release);
                // Only id-less notifications flow through the lane's
                // text/usage normalization; an inbound request is turn
                // activity but carries no billable payload of its own.
                if id.is_none() && self.billable_frame(&frame) {
                    self.turn_billable.store(true, Ordering::Release);
                }
            }
            Direction::Stderr => {}
        }
    }
}

/// The replay verdict for an `uncertain_effect` prompt failure.
enum DispatchVerdict {
    /// Dispatch cannot be ruled out — the turn may already be paid work,
    /// so the failure propagates and the prompt never replays.
    PossiblyDispatched,
    /// Confirmed never dispatched: the lane's own gate refused the prompt
    /// before anything could queue, or the transport reported the
    /// prompt's own write failed — the writer stops at a failed line, so
    /// no peer could have received it. Replays without further checks.
    NeverDispatched,
    /// The journal shows no prompt frame, but a frame still queued on
    /// the lane's writer is invisible to it. Only a final reading taken
    /// after the transport is stopped can confirm the prompt never left.
    Unconfirmed,
}

/// Whether an `uncertain_effect` prompt failure may already have
/// dispatched the turn. A4AA reports `rpc/call` with uncertain effect for
/// a transport failure both before and after prompt dispatch, so the flag
/// alone cannot tell a safe fallback from a duplicated billed turn. The
/// wire journal answers the question exactly — dispatch intent flips only
/// when the prompt frame was written — with no probe call and no
/// reservation hook re-entry. A `agent_session_recovery_required` failure
/// is already the verified pre-dispatch verdict: the lane rejected the
/// prompt at its own gate before anything could be dispatched. A
/// transport-confirmed write failure is the other exact answer: the
/// writer stops at the failed line, so when the prompt is the last
/// `Sent` frame the journal recorded, the prompt's own write is the one
/// that failed — a confirmed undelivered frame the next lane may replay.
/// A frame the peer demonstrably answered — any in-turn activity on the
/// wire — stays uncertain: the write may have delivered despite the
/// reported failure. A `Sent` record whose write was never reported
/// failed keeps the uncertain reading for the same reason: the frame may
/// have reached the agent whole or in part.
fn dispatch_verdict(error: &a4aa::AdapterError, probe: &DispatchProbe) -> DispatchVerdict {
    if error.code == "agent_session_recovery_required" {
        return DispatchVerdict::NeverDispatched;
    }
    if error.code == "agent_transport_failed"
        && error.detail.as_deref() == Some("writer_failed")
        && probe.last_write_was_prompt()
        && !probe.wire_event()
    {
        return DispatchVerdict::NeverDispatched;
    }
    if probe.dispatched() {
        DispatchVerdict::PossiblyDispatched
    } else {
        DispatchVerdict::Unconfirmed
    }
}

/// The turn's own report of an explicit cancellation that landed while
/// the driver was between physical attempts — the same verdict a lane
/// gives when its turn is cancelled before dispatch.
fn cancelled_outcome() -> TurnOutcome {
    TurnOutcome {
        stop_reason: "cancelled".to_owned(),
        text: String::new(),
        truncated: false,
        exit_code: None,
        usage: None,
    }
}

/// `served`-style usage evidence: any nonzero counter proves billed work;
/// an all-zero record does not.
fn billable_usage(usage: &TokenUsage) -> bool {
    [
        usage.input,
        usage.cached_input,
        usage.cached_write,
        usage.output,
        usage.thought,
        usage.total,
    ]
    .into_iter()
    .flatten()
    .any(|tokens| tokens > 0)
}

/// Whether one relayed lane event is evidence the turn produced billable
/// output — streamed text, tool or plan activity, truncation marks —
/// versus bookkeeping-only session and turn lifecycle notifications. A
/// raw event is the lane's decoded reconstruction of a notification, so
/// the dispatch probe answers it with the same normalized counters and
/// configured carriers the wire journal uses — a sequence number under
/// `params.usage` is bookkeeping, not billed work.
fn turn_output(event: &AgentEvent, probe: &DispatchProbe) -> bool {
    match event {
        AgentEvent::MessageChunk { .. }
        | AgentEvent::ToolCall { .. }
        | AgentEvent::Plan { .. }
        | AgentEvent::Truncated => true,
        AgentEvent::Usage { usage, .. } => billable_usage(usage),
        AgentEvent::Raw { raw } => probe.billable_raw(raw),
    }
}

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
    // A4AA normalizes explicit provider quota/capacity refusals across
    // lanes into this code — a certain-effect retryable refusal, treated
    // exactly like a matched exhaustion signature. `rpc/call` covers the
    // generic rpc lane's prompt admission: the refusal arrived in place
    // of a result, before the provider could serve the turn.
    if error.code == "agent_quota_exhausted" {
        return matches!(
            error.stage,
            "session/open"
                | "session/prompt"
                | "session/set_model"
                | "session/set_config_option"
                | "rpc/call"
        );
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
    /// Set once `close` begins — before the lane's session is touched, so
    /// a turn driver still opening a replacement session can never
    /// install or prompt it after `close` returns.
    closing: Arc<AtomicBool>,
    /// Incremented on every `cancel` call. The turn driver snapshots the
    /// value at turn submission and re-checks it before installing or
    /// prompting a replacement, so a cancellation that lands while a
    /// substitute session opens still stops the replay.
    cancellation: Arc<AtomicU64>,
    /// Lifecycle signal for the turn driver's waits: `close` and `cancel`
    /// notify every waiter so a replacement session open can be abandoned
    /// mid-handshake instead of outliving the session or the turn.
    lifecycle: Arc<tokio::sync::Notify>,
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
    /// profile defaults; unset fields inherit them — per lane, inside the
    /// turn, so a mid-turn lane switch re-projects the defaults against
    /// the lane that actually serves the replay.
    pub async fn prompt_with(&self, mut request: TurnRequest) -> a4aa::Result<TurnOutcome> {
        let turn = self.turn_lock.clone().lock_owned().await;
        let instructions_included = self.instructions_due();
        if instructions_included {
            request.text = compose_prompt(self.profile.instructions.as_deref(), &request.text);
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
        let closing = self.closing.clone();
        let cancellation = self.cancellation.clone();
        let lifecycle = self.lifecycle.clone();
        // Cancellation applies to the turn in flight when `cancel` ran:
        // the mark submitted with this turn, compared again before every
        // replacement install or prompt below.
        let cancel_mark = cancellation.load(Ordering::Acquire);
        tokio::spawn(async move {
            let _turn = turn;
            let mut request = request;
            let mut instructions_included = instructions_included;
            // The caller's selector values, captured before the driver
            // manages them. An explicit value stays caller-owned through
            // every lane attempt — the lane's own validation answers it —
            // while everything else is driver state the lane's selector
            // surface decides how to carry.
            let caller_model = request.model.clone();
            let caller_effort = request.effort.clone();
            // The chain candidate a model retry scheduled on a per-turn
            // surface this turn — kept so a lane switch replays it where
            // the next lane carries a turn selector.
            let mut driver_model: Option<String> = None;
            // The chain candidate a session-bound retry is binding at the
            // current reopen; it crosses the install still pending and
            // commits only when the lane's acceptance evidence lands.
            let mut pending_bind: Option<String> = None;
            // The chain candidate scheduled on this turn — on a per-turn
            // surface or carried across a lane switch — waiting for the
            // lane's acceptance evidence. It publishes as the session's
            // default and the chain's active entry only when the attempt
            // that carried it confirms the selector; a refused candidate
            // never publishes, and neither does one whose attempt died
            // before dispatch.
            let mut pending_model: Option<String> = None;
            // Lane events travel through the relay on the lane state: a
            // forwarder task drains the lane's tap into the caller's
            // stream for the session's whole lifetime — a notification
            // between turns is delivered instead of stalling until the
            // next prompt — and marks each armed attempt's output
            // evidence, since `TurnOutcome` alone can lose it: a
            // structured stream may end on an empty result record that
            // hides its earlier records.
            // One logical turn, one reservation cell: the `before_turn`
            // hook runs at most once for the whole turn, however many lane
            // or model attempts the turn makes. The cell also carries the
            // turn's cancellation mark so admission itself can refuse a
            // prompt the caller cancelled while the lane sat idle.
            let turn_reservation = Arc::new(TurnReservation {
                outcome: std::sync::Mutex::new(None),
                cancellation: Some((cancellation.clone(), cancel_mark)),
            });
            {
                let lane = lane.lock().await;
                *lane
                    .reservations
                    .lock()
                    .unwrap_or_else(|error| error.into_inner()) = Some(turn_reservation.clone());
            }
            let outcome = 'lanes: loop {
                // This attempt's dispatch probe and event relay: settle
                // whatever the lane queued outside a turn — session-open
                // markers on a fresh lane, stragglers on a reused one —
                // before opening this attempt's evidence window, so only
                // output produced inside it counts toward the fallback
                // decisions. The wire journal itself resets per physical
                // prompt below — a dispatch record must never carry into
                // the next attempt on this lane. The open session reports
                // its own selector support — negotiated values included —
                // and that decides how a chain retry applies the next
                // entry: a per-turn surface takes it on the request, a
                // session-bound one re-establishes the session.
                let (probe, relay) = {
                    let lane = lane.lock().await;
                    (lane.probe.clone(), lane.relay.clone())
                };
                let (turn_selectors, session_model) = {
                    let guard = session.read().await;
                    (
                        guard.selector_support(SelectorPhase::Turn),
                        guard.selector_support(SelectorPhase::Session).model,
                    )
                };
                let turn_model = turn_selectors.model;
                // Re-project the turn's selectors onto the lane now serving
                // it. A driver-managed value follows this lane's surface:
                // the chain candidate a retry is carrying, the caller's
                // explicit value, then the lane's per-turn default — and
                // where the lane binds a selector at session establishment
                // only the caller's own value may remain: anything else was
                // consumed at open and a stale override is rejected before
                // dispatch.
                request.model = if turn_model == SelectorSupport::Unsupported {
                    caller_model.clone()
                } else {
                    driver_model
                        .clone()
                        .or_else(|| caller_model.clone())
                        .or_else(|| {
                            record_instructions
                                .then(|| {
                                    turn_model_default
                                        .lock()
                                        .unwrap_or_else(|error| error.into_inner())
                                        .clone()
                                })
                                .flatten()
                        })
                };
                request.effort = if turn_selectors.effort == SelectorSupport::Unsupported {
                    caller_effort.clone()
                } else {
                    caller_effort.clone().or_else(|| {
                        record_instructions
                            .then(|| {
                                turn_effort_default
                                    .lock()
                                    .unwrap_or_else(|error| error.into_inner())
                                    .clone()
                            })
                            .flatten()
                    })
                };
                relay.flush().await;
                relay.arm();
                // Set when the next chain entry can only apply by
                // re-establishing this lane's session — a model bound at
                // session/open has no per-turn surface to carry it.
                let mut session_retry = false;
                // The model loop on the current lane: provider-side model
                // failures advance the chain and resubmit on the same
                // session.
                let outcome = loop {
                    // A close or cancel that landed while the driver was
                    // arranging this attempt — including a fallback's
                    // reopen — must stop the turn before anything else
                    // dispatches.
                    if closing.load(Ordering::Acquire) {
                        break 'lanes Err(a4aa::AdapterError::permanent(
                            "agent_session_closed",
                            "session/prompt",
                        ));
                    }
                    if cancellation.load(Ordering::Acquire) != cancel_mark {
                        break 'lanes Ok(cancelled_outcome());
                    }
                    // One iteration is one physical prompt attempt: reset
                    // the wire journal so an earlier attempt's dispatch
                    // record cannot read this attempt's pre-dispatch
                    // failure as dispatched.
                    probe.reset(&request.text);
                    let outcome = {
                        let guard = session.read().await;
                        guard.prompt(request.clone()).await
                    };
                    // A cancel that landed while this attempt sat in
                    // admission or was in flight ends the turn: the lane
                    // may have been idle when cancel ran, so its own
                    // cancel path saw nothing — the driver carries the
                    // cancelled verdict itself.
                    if cancellation.load(Ordering::Acquire) != cancel_mark {
                        break 'lanes Ok(cancelled_outcome());
                    }
                    // The lane may still hold this attempt's tail events:
                    // settle the queue so the chain and fallback decisions
                    // read complete evidence rather than whatever happened
                    // to be relayed already.
                    relay.flush().await;
                    let billed = relay.billable() || probe.wire_billable();
                    // A scheduled candidate the lane provably took is the
                    // accepted replacement: publish it as the session's
                    // default and the chain's active entry. Acceptance
                    // needs positive evidence — a served turn, billable
                    // work this attempt produced, or on a probed lane a
                    // dispatch verdict that cannot rule out delivery of
                    // the prompt frame carrying the selector. A `Sent`
                    // record alone is intent: its write may be the one
                    // the transport confirmed failed, and a lane with no
                    // dispatch probe offers no wire evidence at all.
                    if let Some(accepted) = pending_model.take() {
                        let confirmed = match &outcome {
                            Ok(result) => {
                                !matches!(result.stop_reason.as_str(), "cancelled" | "interrupted")
                            }
                            Err(error) => {
                                !model_failure(error) && (billed || probe.dispatch_confirmed(error))
                            }
                        };
                        if confirmed {
                            *turn_model_default
                                .lock()
                                .unwrap_or_else(|error| error.into_inner()) =
                                Some(accepted.clone());
                            model_state
                                .lock()
                                .unwrap_or_else(|error| error.into_inner())
                                .active = Some(accepted);
                        }
                    }
                    let next = match &outcome {
                        // The reservation refused this turn: its verdict is
                        // already the answer — propagate it before a model
                        // retry or the next lane invokes the reservation
                        // again for the same turn.
                        Err(_) if turn_reservation.refused() => break 'lanes outcome,
                        // The prompt may already have produced billed output.
                        // Advancing the model and resubmitting would duplicate
                        // that turn, so the uncertain failure is propagated.
                        Err(error) if uncertain_turn(error) => break outcome,
                        // The model itself could not serve this turn:
                        // advance the chain and resubmit — the lane's own
                        // configuration surface applies the per-turn model
                        // before dispatch. Billable evidence this attempt
                        // already produced stays decisive even when the
                        // agent's final usage record reads as zero: the
                        // earlier stream is the billed work, so the chain
                        // must not retry a possibly paid turn.
                        Err(error) if model_failure(error) && !billed => model_state
                            .lock()
                            .unwrap_or_else(|error| error.into_inner())
                            .remaining
                            .first()
                            .cloned(),
                        _ => None,
                    };
                    let Some(next) = next else { break outcome };
                    // Decide which surface carries the next entry before
                    // spending it: a per-turn selector takes it on the
                    // request, a session-bound selector re-establishes
                    // the session, and a lane with no model surface keeps
                    // the refused entry active. The entry leaves
                    // `remaining` once scheduled — it is never retried —
                    // but the chain's active entry and the session's
                    // default publish only when the lane accepts it.
                    let session_bound = if turn_model != SelectorSupport::Unsupported {
                        false
                    } else if session_model != SelectorSupport::Unsupported {
                        true
                    } else {
                        break outcome;
                    };
                    model_state
                        .lock()
                        .unwrap_or_else(|error| error.into_inner())
                        .remaining
                        .remove(0);
                    if session_bound {
                        // The lane binds its model at session
                        // establishment and has no per-turn surface — a
                        // request selector would be rejected before
                        // dispatch. The retry re-opens this lane's
                        // session to move the session model forward; the
                        // candidate binds when the replacement opens.
                        pending_bind = Some(next);
                        session_retry = true;
                        break outcome;
                    }
                    // The lane applies the model per turn: the retry
                    // carries the next entry, and once the lane accepts
                    // it the entry becomes the session's default so
                    // later turns inherit the model that actually
                    // served — not the entry the provider just
                    // exhausted.
                    driver_model = Some(next.clone());
                    pending_model = Some(next.clone());
                    request.model = Some(next);
                };
                relay.flush().await;
                let (emitted_output, emitted_billable) = relay.disarm();
                // A session-bound retry keeps its lane: the session is
                // reopened below with the accepted model. Any other
                // outcome answers the lane question from the evidence.
                let lane_failed = if session_retry {
                    true
                } else {
                    match &outcome {
                        // A prompt-stage failure with uncertain effect may have
                        // produced billed output that is already in the event
                        // stream; replaying it on another lane would duplicate
                        // the turn and discard the completed work. A4AA cannot
                        // mark which side of its dispatch boundary a transport
                        // failure fell on — a dead peer during a pre-turn
                        // selector call reports the same `rpc/call` as one
                        // after the prompt left — so the dispatch probe's wire
                        // record answers the dispatch question before this lane
                        // is condemned.
                        Err(error) if uncertain_turn(error) => {
                            match dispatch_verdict(error, &probe) {
                                DispatchVerdict::PossiblyDispatched => break 'lanes outcome,
                                DispatchVerdict::NeverDispatched => true,
                                // No prompt frame reached the wire — but a
                                // frame still queued on the lane's writer is
                                // invisible to the journal, and its write
                                // could yet land on a peer that stopped
                                // answering yet keeps reading. Stop the
                                // transport first, then take the final
                                // reading: only a prompt confirmed never
                                // written may replay on the next lane.
                                DispatchVerdict::Unconfirmed => {
                                    let _ = session.read().await.close().await;
                                    if probe.dispatched() {
                                        break 'lanes outcome;
                                    }
                                    true
                                }
                            }
                        }
                        // The whole model chain exhausted on this lane — the
                        // failure is provider-side, so the caller's next
                        // profile is the right escalation, not another lane.
                        Err(error) if model_failure(error) => break 'lanes outcome,
                        // A lane failure after streamed text or billable usage
                        // may already have produced paid output — replaying it
                        // on the next lane would duplicate the turn. Evidence
                        // is whatever this attempt produced: the wire journal
                        // on lanes with a dispatch frame (attributed at
                        // receipt, so a generic error cannot outrun it), tap
                        // events elsewhere. Bookkeeping-only notifications are
                        // not output evidence: fallback still applies.
                        Err(_) => !(emitted_billable || probe.wire_billable()),
                        // A lane that reports the turn as failed without an
                        // error channel — cli's nonzero exit surfaces as
                        // `stop_reason: "error"` — could not serve this turn
                        // only when it also produced no output and reported no
                        // usage. Output — even when the retained cap dropped
                        // the text entirely or an empty result record hid an
                        // earlier stream — or usage proves the paid turn ran.
                        Ok(result) => {
                            result.stop_reason == "error"
                                && result.text.is_empty()
                                && !result.truncated
                                && result.usage.is_none()
                                && !(emitted_output || probe.wire_event())
                        }
                    }
                };
                if !lane_failed {
                    break 'lanes outcome;
                }
                // A failure no model change can fix — transport, protocol,
                // a crashed server — or a session-bound model retry whose
                // lane re-establishes its session to bind the accepted
                // entry. Reopen on the target lane — this one again for a
                // session retry, the next declared variant otherwise — and
                // replay the turn there; a reopen that fails keeps walking.
                let mut retry_same_lane = session_retry;
                // The search walks with a local index: the installed
                // lane's position commits only when a replacement session
                // actually opens — a candidate that fails to open never
                // advances it, so the installed lane keeps its place for
                // every later turn.
                let mut candidate = lane.lock().await.index;
                loop {
                    let mut lane = lane.lock().await;
                    let same_lane = retry_same_lane;
                    retry_same_lane = false;
                    if !same_lane {
                        candidate += 1;
                    }
                    let Some(spec) = lane.specs.get(candidate).cloned() else {
                        break 'lanes outcome;
                    };
                    // The model this reopen binds: a session-bound retry
                    // pursues its scheduled candidate, a per-turn retry's
                    // candidate is the chain position this turn committed
                    // (the provider already refused the entry before it),
                    // and any other reopen carries the last accepted
                    // chain entry forward.
                    let pending_candidate = pending_bind.clone().or_else(|| driver_model.clone());
                    let head = pending_candidate.clone().or_else(|| {
                        model_state
                            .lock()
                            .unwrap_or_else(|error| error.into_inner())
                            .active
                            .clone()
                    });
                    // Re-project the caller's request against this lane:
                    // selectors the previous lane applied at establishment
                    // may belong on turns here, and vice versa. The model
                    // slot is cleared first so the accepted chain entry —
                    // not the caller's original value — is the one routed:
                    // the caller's model either was refused already or is
                    // the same entry `head` carries.
                    let mut projection = lane.requested.clone();
                    projection.config.model = None;
                    let (mut new_open, mut turn_model, turn_effort) =
                        lane_open(&spec, &projection, &profile, head.clone());
                    // A pending candidate is a per-turn selector only on a
                    // lane with a turn surface: it rides the replayed
                    // request instead of the replacement session's
                    // persistent default — a refusal must not leave the
                    // refused model installed as the session's model,
                    // ready to send again on the next implicit prompt.
                    if let Some(candidate) = &pending_candidate {
                        if new_open.config.model.as_deref() == Some(candidate.as_str())
                            && spec.selector_support(SelectorPhase::Turn).model
                                != SelectorSupport::Unsupported
                        {
                            new_open.config.model = model_state
                                .lock()
                                .unwrap_or_else(|error| error.into_inner())
                                .active
                                .clone();
                            turn_model = Some(candidate.clone());
                        }
                    }
                    let (adapter, probe) = lane.adapter(candidate);
                    let (tap_tx, next_tap_rx) = mpsc::unbounded_channel();
                    // The replacement open is interruptible: register
                    // interest in the lifecycle signal before the flag
                    // reads, so a close or cancel landing between the
                    // check and the wait is never missed — either abandons
                    // a stalled handshake instead of letting the turn and
                    // the half-open replacement outlive them.
                    let notified = lifecycle.notified();
                    tokio::pin!(notified);
                    let _ = notified.as_mut().enable();
                    if closing.load(Ordering::Acquire) {
                        break 'lanes outcome;
                    }
                    if cancellation.load(Ordering::Acquire) != cancel_mark {
                        break 'lanes Ok(cancelled_outcome());
                    }
                    // The replacement session opens while the installed
                    // one stays in place: a refused open leaves the live
                    // session, its probe and the accepted model binding
                    // untouched — all three commit together only when the
                    // replacement actually opens.
                    let result = tokio::select! {
                        result = adapter.open_session(new_open.clone(), tap_tx) => result,
                        _ = notified.as_mut() => {
                            if closing.load(Ordering::Acquire) {
                                break 'lanes outcome;
                            }
                            break 'lanes Ok(cancelled_outcome());
                        }
                    };
                    // Whether the reopen carried a scheduled-but-unaccepted
                    // chain candidate — its turn default commits only when
                    // the replay's acceptance evidence lands.
                    let mut carried = false;
                    match result {
                        Ok(new_session) => {
                            // The model the replacement actually carries —
                            // session-bound in `config.model`, or the turn
                            // default a per-turn lane applies.
                            let accepted =
                                new_open.config.model.clone().or_else(|| turn_model.clone());
                            {
                                let mut guard = session.write().await;
                                // The caller may have closed the session or
                                // cancelled this turn while the replacement
                                // was opening — checked again under the
                                // installation lock so the swap is strictly
                                // ordered against close and cancel: never
                                // install it or let it serve the replay.
                                if closing.load(Ordering::Acquire) {
                                    drop(guard);
                                    let _ = new_session.close().await;
                                    break 'lanes outcome;
                                }
                                if cancellation.load(Ordering::Acquire) != cancel_mark {
                                    drop(guard);
                                    let _ = new_session.close().await;
                                    break 'lanes Ok(cancelled_outcome());
                                }
                                // The lifecycle check stands immediately
                                // before installation; the old lane then
                                // retires outside the lock — its shutdown
                                // call may stall, and holding the write
                                // lock across it would block close and
                                // cancel on the read side.
                                let retiring = std::mem::replace(&mut *guard, new_session);
                                drop(guard);
                                tokio::spawn(async move {
                                    let _ = retiring.close().await;
                                });
                            }
                            lane.index = candidate;
                            lane.probe = probe.clone();
                            lane.open = new_open;
                            // The dead lane's relay keeps draining its
                            // tap to the caller on its own — leftovers
                            // are the dead attempt's evidence, never
                            // counted against the reopened lane.
                            lane.relay = EventRelay::spawn(next_tap_rx, lane.events.clone(), probe);
                            match pending_bind.take().or_else(|| driver_model.clone()) {
                                // A candidate the lane switch carried:
                                // it stays pending until the replay
                                // proves the lane took it — bound in the
                                // open request's session model or
                                // deliverable on the lane's per-turn
                                // surface, which carries it on the
                                // replayed request. A lane with no model
                                // selector never carries the candidate;
                                // its replay proves nothing about it, so
                                // it must not publish.
                                Some(carried_model)
                                    if lane.open.config.model.as_deref()
                                        == Some(carried_model.as_str())
                                        || spec.selector_support(SelectorPhase::Turn).model
                                            != SelectorSupport::Unsupported =>
                                {
                                    driver_model = Some(carried_model.clone());
                                    pending_model = Some(carried_model);
                                    carried = true;
                                }
                                // No scheduled candidate: the model the
                                // replacement carries is the last
                                // accepted chain entry — publish it.
                                None => {
                                    model_state
                                        .lock()
                                        .unwrap_or_else(|error| error.into_inner())
                                        .active = accepted;
                                }
                                Some(_) => {}
                            }
                        }
                        Err(open_error) if model_failure(&open_error) => {
                            // A session-bound retry refused at
                            // establishment still owns its chain: the
                            // next entry binds the model slot and this
                            // same lane re-opens — a per-turn surface
                            // would have kept walking the chain the
                            // same way. The candidate stays pending:
                            // the installed session and the accepted
                            // binding carry no sign of it.
                            if same_lane {
                                let next = {
                                    let mut chain = model_state
                                        .lock()
                                        .unwrap_or_else(|error| error.into_inner());
                                    let next = chain.remaining.first().cloned();
                                    if next.is_some() {
                                        chain.remaining.remove(0);
                                    }
                                    next
                                };
                                if let Some(next) = next {
                                    pending_bind = Some(next);
                                    retry_same_lane = true;
                                    continue;
                                }
                            }
                            break 'lanes Err(open_error);
                        }
                        Err(_) => continue,
                    }
                    // A carried candidate keeps the last accepted turn
                    // default until its acceptance evidence lands; the
                    // effort default is unrelated to the model chain and
                    // always re-projects.
                    if !carried {
                        *turn_model_default
                            .lock()
                            .unwrap_or_else(|error| error.into_inner()) = turn_model;
                    }
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

    /// Explicit user cancellation of the in-flight turn. The mark moves
    /// first, then every waiter wakes: a turn driver between attempts —
    /// or inside a fallback's replacement open — sees the cancellation
    /// before it installs or prompts anything, even while the current
    /// session sits idle.
    pub async fn cancel(&self) -> a4aa::Result<()> {
        self.cancellation.fetch_add(1, Ordering::AcqRel);
        self.lifecycle.notify_waiters();
        self.session.read().await.cancel().await
    }

    /// Close the session and terminate the agent process tree. The flag
    /// is raised first and every waiter wakes: a replacement session
    /// still opening in the turn driver is abandoned, never installed or
    /// prompted after close returns.
    pub async fn close(&self) -> a4aa::Result<()> {
        self.closing.store(true, Ordering::Release);
        self.lifecycle.notify_waiters();
        self.session.read().await.close().await
    }
}
