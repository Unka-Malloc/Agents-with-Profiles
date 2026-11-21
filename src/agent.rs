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
    fn adapter(&mut self, index: usize) -> Adapter {
        let probe = DispatchProbe::new(self.observer.clone(), &self.specs[index]);
        self.probe = probe.clone();
        let observer: SharedObserver = Some(probe);
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
        if let Some(outcome) = turn.as_ref().and_then(|turn| {
            turn.outcome
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .clone()
        }) {
            return outcome;
        }
        let outcome = self.hook.run().await;
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
    /// evidence only inside the attempt window.
    fn forward(
        evidence: &RelayEvidence,
        events: &mpsc::UnboundedSender<AgentEvent>,
        probe: &DispatchProbe,
        event: AgentEvent,
    ) {
        if evidence.armed.load(Ordering::Acquire) {
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
    /// itself is the payload document (flat envelopes).
    data_field: Option<String>,
    /// The lane's stream-text notification method (`text_event_method`)
    /// and the dotted path to its payload (`text_path`), mirroring how
    /// the lane normalizes stream text.
    text_event_method: Option<String>,
    text_path: Option<String>,
    /// The lane's configured usage carrier (`usage_path`); when unset the
    /// conventional `params.usage` / `params.message.usage` carriers are
    /// read instead — the same precedence the lane's own ingest applies.
    usage_path: Option<String>,
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
        }
        Arc::new(probe)
    }

    /// Whether this attempt's prompt frame reached the wire. `true` for
    /// lanes without an observable dispatch point — their uncertain
    /// failures keep the conservative reading.
    fn dispatched(&self) -> bool {
        self.method.is_none() || self.dispatched.load(Ordering::Acquire)
    }

    /// Whether the last frame the transport writer attempted was this
    /// attempt's prompt — on a `writer_failed` report, the write the
    /// transport confirmed failed.
    fn last_write_was_prompt(&self) -> bool {
        self.last_sent_was_prompt.load(Ordering::Acquire)
    }

    /// In-turn output evidence attributed at wire receipt.
    fn wire_event(&self) -> bool {
        self.turn_event.load(Ordering::Acquire)
    }

    /// In-turn billable evidence attributed at wire receipt.
    fn wire_billable(&self) -> bool {
        self.turn_billable.load(Ordering::Acquire)
    }

    fn reset(&self) {
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
/// output, thought — decide the served question; a record carrying only
/// unknown keys is bookkeeping, not billed work.
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
                    .and_then(|frame| {
                        frame
                            .get(self.method_field.as_str())
                            .and_then(serde_json::Value::as_str)
                            .map(str::to_owned)
                    });
                let prompt = prompt.as_deref() == self.method.as_deref();
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

/// `served`-style usage evidence: any nonzero counter proves billed work;
/// an all-zero record does not.
fn billable_usage(usage: &TokenUsage) -> bool {
    [
        usage.input,
        usage.cached_input,
        usage.cached_write,
        usage.output,
        usage.thought,
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
            // or model attempts the turn makes.
            let turn_reservation = Arc::new(TurnReservation::default());
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
                let (turn_model, session_model) = {
                    let guard = session.read().await;
                    (
                        guard.selector_support(SelectorPhase::Turn).model,
                        guard.selector_support(SelectorPhase::Session).model,
                    )
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
                    // One iteration is one physical prompt attempt: reset
                    // the wire journal so an earlier attempt's dispatch
                    // record cannot read this attempt's pre-dispatch
                    // failure as dispatched.
                    probe.reset();
                    let outcome = {
                        let guard = session.read().await;
                        guard.prompt(request.clone()).await
                    };
                    // The lane may still hold this attempt's tail events:
                    // settle the queue so the chain and fallback decisions
                    // read complete evidence rather than whatever happened
                    // to be relayed already.
                    relay.flush().await;
                    let billed = relay.billable() || probe.wire_billable();
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
                    // committing the chain position: a per-turn selector
                    // takes it on the request, a session-bound selector
                    // re-establishes the session, and a lane with no model
                    // surface keeps the refused entry active.
                    let session_bound = if turn_model != SelectorSupport::Unsupported {
                        false
                    } else if session_model != SelectorSupport::Unsupported {
                        true
                    } else {
                        break outcome;
                    };
                    {
                        let mut chain = model_state
                            .lock()
                            .unwrap_or_else(|error| error.into_inner());
                        chain.remaining.remove(0);
                        chain.active = Some(next.clone());
                    }
                    if session_bound {
                        // The lane binds its model at session
                        // establishment and has no per-turn surface — a
                        // request selector would be rejected before
                        // dispatch. The retry re-opens this lane's
                        // session to move the session model forward.
                        session_retry = true;
                        break outcome;
                    }
                    // The lane applies the model per turn: the retry
                    // carries the next entry, and the accepted entry
                    // becomes the session's default so later turns
                    // inherit the model that actually served — not
                    // the entry the provider just exhausted.
                    *turn_model_default
                        .lock()
                        .unwrap_or_else(|error| error.into_inner()) = Some(next.clone());
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
                loop {
                    let mut lane = lane.lock().await;
                    let same_lane = retry_same_lane;
                    retry_same_lane = false;
                    if !same_lane {
                        lane.index += 1;
                    }
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
                    let (mut new_open, turn_model, turn_effort) =
                        lane_open(&spec, &lane.requested, &profile, active.clone());
                    if same_lane {
                        // The refused model was bound at establishment, so
                        // the retry re-opens this lane with the accepted
                        // entry in the model slot — the selector phase
                        // this lane actually supports.
                        new_open.config.model = active;
                    }
                    let index = lane.index;
                    let adapter = lane.adapter(index);
                    let (tap_tx, next_tap_rx) = mpsc::unbounded_channel();
                    let reopen = adapter.open_session(new_open.clone(), tap_tx);
                    {
                        let mut guard = session.write().await;
                        let _ = guard.close().await;
                        match reopen.await {
                            Ok(new_session) => {
                                *guard = new_session;
                                lane.open = new_open;
                                // The dead lane's relay keeps draining its
                                // tap to the caller on its own — leftovers
                                // are the dead attempt's evidence, never
                                // counted against the reopened lane.
                                lane.relay = EventRelay::spawn(
                                    next_tap_rx,
                                    lane.events.clone(),
                                    lane.probe.clone(),
                                );
                            }
                            Err(open_error) if model_failure(&open_error) => {
                                // A session-bound retry refused at
                                // establishment still owns its chain: the
                                // next entry binds the model slot and this
                                // same lane re-opens — a per-turn surface
                                // would have kept walking the chain the
                                // same way.
                                if same_lane {
                                    let mut chain = model_state
                                        .lock()
                                        .unwrap_or_else(|error| error.into_inner());
                                    if let Some(next) = chain.remaining.first().cloned() {
                                        chain.remaining.remove(0);
                                        chain.active = Some(next);
                                        drop(chain);
                                        retry_same_lane = true;
                                        continue;
                                    }
                                }
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
