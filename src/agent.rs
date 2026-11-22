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
    SelectorSupport, SelectorSupportSet, SessionRequest, SharedObserver, TokenUsage, TurnOutcome,
    TurnRequest,
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
        // The chain position the establishment walk reached: candidates a
        // lane already refused stay spent for every later lane, so the
        // alternate resumes with the entry the previous lane was
        // attempting — never the chain head it already proved refused.
        let mut lane_model = chain.first().cloned();
        // Entries a lane's open refused — provider-side refusals stay
        // spent for every later lane: once held entries can rewind the
        // walk behind the cursor, the cursor alone cannot express them.
        let mut spent: Vec<String> = Vec::new();
        // Entries a lane could not deliver at all — no setup call binds
        // them and no turn slot resolves them — held out of that lane's
        // walk. They re-enter in chain order on the next lane, and
        // survive into the session's `remaining` for a later lane or
        // turn.
        let mut held: Vec<String> = Vec::new();
        for (lane_index, spec) in self.specs.iter().enumerate() {
            // Entries a previous lane held as undeliverable re-enter this
            // lane's walk at their chain position — its slots may resolve
            // what theirs could not.
            if let Some(earliest) = held.first().cloned() {
                lane_model = Some(earliest);
                held.clear();
            }
            // A cursor left on an entry the peer already refused never
            // re-projects: refused candidates stay spent on every later
            // lane.
            while lane_model
                .as_deref()
                .is_some_and(|entry| spent.iter().any(|refused| refused == entry))
            {
                lane_model = next_model(&chain, lane_model.as_deref());
            }
            // A resumed cursor must also be one this lane can actually
            // deliver: an entry no setup call binds and no turn slot
            // resolves would open model-free — the peer accepts on its
            // own default while a deliverable entry behind it is never
            // tried. Re-check the cursor against this lane's own slots
            // before opening on the peer's default: undeliverable
            // entries re-hold for a later lane or turn, and the walk
            // resumes at the first one this lane can carry — or parks
            // the earliest held entry pending when none stands. The
            // first lane keeps the parked-head reading: its cursor is
            // the untouched chain head, whose claim rides the peer's
            // default by design.
            if lane_index > 0 {
                if let Some(cursor) = lane_model.clone() {
                    if !setup_carries_candidate(spec, &cursor, resuming(&request))
                        && !turn_carries_candidate(spec, &cursor)
                    {
                        if !held.iter().any(|entry| entry == &cursor) {
                            held.push(cursor.clone());
                        }
                        lane_model = next_deliverable(
                            &chain,
                            Some(&cursor),
                            spec,
                            resuming(&request),
                            &spent,
                            &mut held,
                        )
                        .or_else(|| held.first().cloned());
                    }
                }
            }
            let (mut lane_request, turn_model_default, turn_effort_default) =
                lane_open(spec, &request, &self.profile, lane_model.clone());
            let lane_resuming = resuming(&lane_request);
            // The candidate an open could not bind — every declared slot
            // resolved its value absent — parked aside for the pending
            // claim below. Leaving it in `config.model` would store it
            // as the adapter's session default, where every later
            // prompt's `{model}` substitution inherits it — a direct
            // prompt included — though the wire never carried it.
            let mut parked_model = None;
            // A requested model the lane cannot express at establishment
            // advances the chain before the first turn is even attempted.
            let session = loop {
                if model_binds_at_open(spec, resuming(&lane_request))
                    && !setup_carries_model(spec, &lane_request)
                {
                    // The open's model slots exist but none resolves
                    // this candidate — an unqualified `{modelProvider}`
                    // — so establishment cannot bind it. Out of the
                    // stored slot it waits pending: only an implicit
                    // prompt supplies the queued claim.
                    parked_model = lane_request.config.model.take();
                }
                // Every open attempt gets its own event tap: the session's
                // relay drains it into the caller's stream for the whole
                // lifetime and counts each armed attempt's produced-output
                // evidence for the fallback decision.
                let (tap_tx, tap_rx) = mpsc::unbounded_channel();
                // Every open attempt also gets a dispatch probe: the wire
                // tap the turn driver later reads to learn whether an
                // uncertain failure's prompt frame ever left the machine.
                let probe = DispatchProbe::new(self.observer.clone(), spec, &lane_request);
                let observer: SharedObserver = Some(probe.clone());
                let result = self
                    .adapter(spec, &reservations)
                    .secrets(probe.wrap_secrets(self.secrets.clone()))
                    .observer(observer)
                    .open_session(lane_request.clone(), tap_tx)
                    .await;
                match result {
                    Err(error) if model_failure(&error) => {
                        // Only the setup call that produced the refusal
                        // can answer about the model: where it binds
                        // nothing — a model-free `initialize` ahead of a
                        // model-carrying `session/open`, or an open whose
                        // model is only a stored default feeding later
                        // prompts — every retry re-sends the identical
                        // call, so the refusal is the lane's answer and
                        // the walk moves on.
                        if !refused_call_binds_model(spec, resuming(&lane_request), &probe) {
                            lane_model = lane_request
                                .config
                                .model
                                .clone()
                                .or_else(|| parked_model.clone())
                                .or(lane_model);
                            lane_error = Some(error);
                            break None;
                        }
                        // A refusal that binds the model proves the wire
                        // carried this candidate — the entry is spent:
                        // refused candidates never publish as the
                        // session's model and the chain moves on.
                        if let Some(refused) = lane_request.config.model.clone() {
                            if !spent.iter().any(|entry| entry == &refused) {
                                spent.push(refused);
                            }
                        }
                        if lane_request.config.model == request.config.model {
                            // The caller's explicit model was just refused:
                            // later lanes project the advanced cursor —
                            // never the refused original value.
                            request.config.model = None;
                        }
                        // The entry replacing it must be one this lane
                        // can deliver: an unqualified fallback on a
                        // provider-only lane would open with a model-free
                        // request the peer accepts on its own default
                        // while the entries behind it were never tried —
                        // hold it out and walk on.
                        let Some(next) = next_deliverable(
                            &chain,
                            lane_request.config.model.as_deref(),
                            spec,
                            lane_resuming,
                            &spent,
                            &mut held,
                        ) else {
                            // Model exhaustion is provider-side only once
                            // the chain is fully spent — every entry
                            // refused. Entries merely held out of this
                            // lane stay in the walk: the next chain entry
                            // parks pending and the model-free open
                            // leaves the peer on its own default until a
                            // lane or turn can carry the claim.
                            match held.first().cloned() {
                                Some(parked) => {
                                    lane_request.config.model = Some(parked);
                                    continue;
                                }
                                None => return Err(error),
                            }
                        };
                        lane_request.config.model = Some(next);
                    }
                    Err(error) => {
                        // The candidate this lane's walk last attempted
                        // carries to the next lane: refused entries
                        // already advanced the request past themselves,
                        // and a lane that never got to express a model
                        // leaves the cursor untouched. An open whose
                        // model slot could not resolve its candidate
                        // holds the entry parked outside the slot — it
                        // is the cursor the walk actually reached, and
                        // the cleared slot must not read as the refused
                        // entry before it.
                        lane_model = lane_request
                            .config
                            .model
                            .clone()
                            .or_else(|| parked_model.clone())
                            .or(lane_model);
                        lane_error = Some(error);
                        break None;
                    }
                    Ok(session) => break Some((session, lane_request, tap_rx, probe)),
                }
            };
            let Some((session, effective_request, tap_rx, probe)) = session else {
                continue;
            };
            // The requested model publishes only with carry evidence: an
            // open whose model slot resolves absent — an unqualified
            // `{modelProvider}` — omits the member, the peer binds its own
            // default, and the omitted candidate waits pending until a
            // serving turn's frame provably carried it. A lane whose
            // establishment call binds nothing keeps the stored-default
            // reading, and a refused entry is spent — tracked out of
            // `active`, `pending` and `remaining` alike.
            let bound = !model_binds_at_open(spec, resuming(&effective_request))
                || !probe.has_dispatch_point()
                || probe.model_setup_sent();
            let (active_model, pending_model) = if bound {
                (
                    effective_request
                        .config
                        .model
                        .clone()
                        .or_else(|| turn_model_default.clone()),
                    None,
                )
            } else {
                (
                    None,
                    effective_request.config.model.clone().or(parked_model),
                )
            };
            let remaining_models = remaining_models(
                &chain,
                pending_model.as_deref().or(active_model.as_deref()),
                &spent,
            );
            return Ok(VirtualSession {
                session: Arc::new(tokio::sync::RwLock::new(session)),
                profile: self.profile.clone(),
                turn_model_default: Arc::new(std::sync::Mutex::new(turn_model_default)),
                turn_effort_default: Arc::new(std::sync::Mutex::new(turn_effort_default)),
                model_state: Arc::new(std::sync::Mutex::new(ModelChain {
                    active: active_model,
                    pending: pending_model,
                    remaining: remaining_models,
                })),
                lane: Arc::new(tokio::sync::Mutex::new(LaneState {
                    specs: self.specs.clone(),
                    index: lane_index,
                    requested: request.clone(),
                    open: effective_request,
                    refused_default: None,
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
    /// The stored default the provider refused while it was an
    /// unconfirmed pending candidate — installed by an open that never
    /// carried it. The session still holds it, so every implicit prompt
    /// would resend the refused model; the lane keeps the mark until a
    /// replacement session installs a clean slot.
    refused_default: Option<String>,
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
    fn adapter(&self, index: usize, request: &SessionRequest) -> (Adapter, Arc<DispatchProbe>) {
        let probe = DispatchProbe::new(self.observer.clone(), &self.specs[index], request);
        let observer: SharedObserver = Some(probe.clone());
        let mut adapter = Adapter::new(self.specs[index].clone())
            .secrets(probe.wrap_secrets(self.secrets.clone()))
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

/// Model fallback state for one session: the model currently selected,
/// an accepted-but-unconfirmed candidate, and the remaining chain
/// entries a failed turn may advance to.
#[derive(Default)]
struct ModelChain {
    active: Option<String>,
    /// The candidate the session's open accepted without carry
    /// evidence — an open whose model slot resolved absent bound the
    /// peer's own default, so the stored entry waits here until a
    /// serving turn's frame provably carried it, then publishes as
    /// `active`. A refused entry never lands here: it is spent.
    pending: Option<String>,
    remaining: Vec<String>,
}

/// Chain entries still eligible once `published` is out — every entry
/// the peer never refused and the session did not publish, in chain
/// order. Entries an establishment walk held as undeliverable on the
/// serving lane stay eligible: a later lane or turn may still carry
/// them.
fn remaining_models(chain: &[String], published: Option<&str>, spent: &[String]) -> Vec<String> {
    chain
        .iter()
        .filter(|entry| {
            Some(entry.as_str()) != published && !spent.iter().any(|refused| refused == *entry)
        })
        .cloned()
        .collect()
}

/// The next chain entry after `current`, or the chain head when `current`
/// is outside the chain.
fn next_model(chain: &[String], current: Option<&str>) -> Option<String> {
    match current.and_then(|value| chain.iter().position(|entry| entry == value)) {
        Some(index) => chain.get(index + 1).cloned(),
        None => chain.first().cloned(),
    }
}

/// The next chain entry after `current` this lane can actually deliver —
/// one a setup call binds or a turn slot resolves. Entries the peer
/// already refused stay spent; entries this lane can carry nowhere are
/// held out for a later lane or turn rather than opening a model-free
/// request the peer accepts on its own default while the entries
/// behind them were never tried.
fn next_deliverable(
    chain: &[String],
    current: Option<&str>,
    spec: &AgentSpec,
    resuming: bool,
    spent: &[String],
    held: &mut Vec<String>,
) -> Option<String> {
    let mut cursor = current.map(str::to_owned);
    loop {
        let candidate = next_model(chain, cursor.as_deref())?;
        cursor = Some(candidate.clone());
        if spent.iter().any(|entry| entry == &candidate) {
            continue;
        }
        if setup_carries_candidate(spec, &candidate, resuming)
            || turn_carries_candidate(spec, &candidate)
        {
            return Some(candidate);
        }
        if !held.iter().any(|entry| entry == &candidate) {
            held.push(candidate);
        }
    }
}

/// Return a scheduled candidate to the chain when its attempt never
/// started: the entry left `remaining` at schedule time and no prompt or
/// open ever carried it, so it is still a viable fallback the next turn
/// must be allowed to try. A candidate that already published as the
/// chain's active entry, one still pending the serving lane's carry
/// evidence — it rides the session's stored model into every prompt
/// already — or one still queued, is not re-added.
fn restore_candidate(chain: &std::sync::Mutex<ModelChain>, candidate: Option<String>) {
    let Some(candidate) = candidate else {
        return;
    };
    let mut chain = chain.lock().unwrap_or_else(|error| error.into_inner());
    if chain.active.as_deref() == Some(candidate.as_str())
        || chain.pending.as_deref() == Some(candidate.as_str())
        || chain.remaining.iter().any(|entry| entry == &candidate)
    {
        return;
    }
    chain.remaining.insert(0, candidate);
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
    /// Output evidence relayed while armed — genuine turn output or
    /// progress, never bookkeeping-only signals like an empty chunk or
    /// a usage record with no other content.
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
    /// evidence only inside the attempt window — and only on lanes
    /// without an observable dispatch point. On a probed lane the wire
    /// journal already attributes output at frame receipt: an event
    /// received before the prompt but dequeued only after dispatch is
    /// never the attempt's output. Lanes without an observable dispatch
    /// point keep tap attribution — the armed window is all their
    /// evidence has.
    fn forward(
        evidence: &RelayEvidence,
        events: &mpsc::UnboundedSender<AgentEvent>,
        probe: &DispatchProbe,
        event: AgentEvent,
    ) {
        if evidence.armed.load(Ordering::Acquire) && !probe.has_dispatch_point() {
            if turn_progress(&event, probe) {
                evidence.output.store(true, Ordering::Release);
            }
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

    /// The armed window's output flag — read after `flush` settled the
    /// queue so a just-arrived event cannot be missed.
    fn output(&self) -> bool {
        self.evidence.output.load(Ordering::Acquire)
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

/// One declared setup call's wire identity: the method its request
/// carries and the params-template positions a `{model}`-family
/// placeholder occupies. `params_at_root` mirrors the lane's encoder —
/// an inline envelope merges an object params document into the frame
/// root; every other shape sits under `params`.
struct SetupWire {
    method: String,
    params_at_root: bool,
    /// The call's params template: the journaled request claims the call
    /// whose template could have produced the frame's resolved params —
    /// a same-method call the adapter skipped because its optional leaves
    /// all resolved empty never matches.
    template: serde_json::Value,
    /// Params-root-relative positions whose template leaf declares a
    /// `{model}`-family placeholder, with the leaf text — a slot is
    /// model-carrying only when a model token in it resolved to a value.
    model_slots: Vec<(Vec<String>, String)>,
}

/// One setup request observed on the wire, awaiting its response.
struct SentSetup {
    /// The request id, keyed the same way the peer keys pending calls.
    id: String,
    /// Whether the frame's params populated a declared model slot —
    /// only a model-carrying call's refusal is an answer about the
    /// model a chain entry can change.
    carries_model: bool,
    /// Whether this entry is the session call — its successful response
    /// is where the session id's presence is learned.
    binds_session: bool,
}

/// Sent setup requests still awaiting a response, plus the call claims
/// that matched them. The lane issues its setup calls in one fixed order
/// — `initialize`, then the session call the request selects — so a
/// request claims the first unclaimed call whose method it carries:
/// two calls sharing a wire method attach to the call the adapter
/// actually selected instead of matching every same-method template.
struct SetupJournal {
    /// One claim per entry in `setup_calls`, in adapter call order.
    claimed: Vec<bool>,
    /// One claim per entry in `turn_calls` — reset per attempt, since
    /// the pre-turn calls issue once per prompt.
    turn_claimed: Vec<bool>,
    /// The selector values the pre-turn calls resolve against this
    /// attempt — rebuilt at every `reset` from the attempt's own
    /// model/effort over the session's stored defaults, so a same-method
    /// frame claims the call whose resolved shape it actually is.
    turn_vars: ProbeVars,
    /// The journaled requests in wire order.
    sent: Vec<SentSetup>,
}

/// The adapter's secret resolutions in arrival order — presence only,
/// never a resolved value. `substitute_resolved` resolves a call's
/// `{secret:NAME}` leaves contiguously, right before the call's frame is
/// written or the call is skipped, so the slice one call owns is located
/// by the occurrence counts of the calls issued before it in the same
/// phase. `phase_start` is where the current phase's first call began
/// resolving: spawn-time env resolutions precede the setup calls, and
/// every `reset` opens a fresh turn-phase slice. A presence a later
/// call's resolution produced therefore never rewrites an earlier
/// call's skip decision on the same name.
#[derive(Default)]
struct SecretJournal {
    outcomes: Vec<(String, bool)>,
    phase_start: usize,
}

/// The two facts one call's own resolutions can leave on a name —
/// tracked separately because `substitute_resolved` keeps them in
/// different places: a lookup that returned a value leaves the name in
/// `values`, where every embedded token substitutes it, while any empty
/// lookup lands the name on `absent`, the list a whole-leaf member
/// drops on. Two lookups of one name inside a single call can disagree,
/// so both facts can hold at once.
#[derive(Clone, Copy, Default)]
struct SecretPresence {
    /// At least one lookup in this call's slice returned a value —
    /// embedded `{secret:NAME}` tokens substitute it.
    value: bool,
    /// At least one lookup in this call's slice came back empty —
    /// a whole-leaf `{secret:NAME}` member drops.
    absent: bool,
}

/// The selector values a call's params template resolves against — the
/// subset of the lane's substitution inputs the wire journal can prove.
/// Values the journal cannot know — secrets and peer-assigned session
/// ids — leave their leaves flexible instead of guessing.
#[derive(Default)]
struct ProbeVars {
    /// Placeholders with a known substituted value — the same map the
    /// lane's `vars` populates, including the always-present selectors
    /// that substitute `""` when unset.
    known: std::collections::BTreeMap<String, String>,
    /// Optional-selector names the lane reports absent: a whole-leaf
    /// member drops from the resolved document, a mixed leaf substitutes
    /// `""`.
    absent: std::collections::BTreeSet<String>,
    /// Names present with a value the journal cannot know — a
    /// process-minted session id: the member must be present, but any
    /// string is consistent with it.
    wild: std::collections::BTreeSet<String>,
    /// Names whose presence itself is unknowable — a peer-assigned
    /// session id and every `{secret:…}` leaf: the member may be present
    /// carrying any string, or absent entirely.
    flexible: std::collections::BTreeSet<String>,
}

/// The session-id shape one lane's substitution carries: the resume
/// binding the journal knows verbatim, a process-minted id it cannot
/// know, a peer-assigned id whose presence and value are both
/// unknowable, or no id at all.
#[derive(Clone)]
enum SessionIdVar {
    Known(String),
    Wild,
    Flexible,
    Absent,
}

/// The inputs `selector_vars` resolves — the same fields the lane's
/// `VarSpec` carries, narrowed to what the journal knows.
struct SelectorInput<'a> {
    cwd: &'a str,
    prompt: &'a str,
    model: Option<&'a str>,
    effort: Option<&'a str>,
    mode: Option<&'a str>,
    instructions: Option<&'a str>,
    permission: Option<&'a str>,
    provider: Option<&'a str>,
    session_id: SessionIdVar,
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
    /// The envelope's response failure members (`error_field`,
    /// `success_field`) — the same failure classification the peer's
    /// dispatch applies, needed to tell a refused setup call's response
    /// from an accepted one.
    error_field: Option<String>,
    success_field: Option<String>,
    /// The envelope's response result member (`result_field`) — where
    /// the prompt's synchronous reply carries its payload, usage
    /// included, when `result` is not the field name.
    result_field: Option<String>,
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
    /// A cli lane's declared result-record selection (`result_selector`
    /// under `result_selector_field`, `"type"` when unset) and the path
    /// to its result text (`result_path`) — the same selection the
    /// lane's own terminal extraction applies, so a bookkeeping record
    /// the lane would ignore reads the same way here.
    result_selector: Option<String>,
    result_selector_field: Option<String>,
    result_path: Option<String>,
    /// A cli lane's declared session-id carrier (`session_id_path`): a
    /// record whose only claim is that value is the session's own init
    /// marker — bookkeeping, not turn output.
    session_id_path: Option<String>,
    /// A cli lane's raw stream records are the records themselves — the
    /// lane's own selectors read the full NDJSON document, not a
    /// `params` member inside it.
    cli: bool,
    /// A json-lines cli lane's outcome text is the retained raw stream
    /// when no result record supplied authoritative text — raw NDJSON
    /// bytes, not evidence of their own.
    json_lines: bool,
    /// The frame-root fields an inline envelope stamps over a merged
    /// object params document — positions a journaled template cannot be
    /// held to, since the stamped value replaced whatever the call
    /// resolved there.
    overwritten: std::collections::BTreeSet<String>,
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
    /// The prompt params template's `{model}`-family positions that
    /// survive the lane's encoding, with their leaf text — a slot the
    /// inline envelope stamps over never reaches the wire, and a leaf
    /// whose model tokens all resolved empty (`provider:{modelProvider}`
    /// on an unqualified candidate) emits scaffolding, never a model.
    prompt_model_slots: Vec<(Vec<String>, String)>,
    /// Whether this attempt's prompt frame populated an effective model
    /// slot — the request's model reached the wire below the prompt's
    /// own dispatch.
    prompt_model_sent: AtomicBool,
    /// The prompt template's `{secret:NAME}` occurrence count — the
    /// width of the outcome slice the prompt's own resolution owns
    /// after every declared pre-turn call's.
    prompt_secret_count: usize,
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
    /// The setup calls the adapter actually issues, in issue order —
    /// `initialize`, then `session_open` or `session_resume` as the
    /// open request's resume flag selects — with the params-template
    /// positions a `{model}`-family placeholder occupies. An rpc lane
    /// reports every setup failure as `rpc/call`, so the wire journal
    /// must name the refused call exactly: method names cannot
    /// distinguish calls a lane shares, and an interleaved callback
    /// reply is not a setup request.
    setup_calls: Vec<SetupWire>,
    /// The lane's `pre_turn` call templates in issue order, with the
    /// same model-slot model: those calls run while the turn phase
    /// holds and share the `rpc/call` stage with the prompt, so a
    /// refusal to a model-free preparation call must be attributable
    /// before it can be read as the model's answer.
    turn_calls: Vec<SetupWire>,
    /// The selector values each setup call resolves against, parallel
    /// to `setup_calls` — the open request's session config, with the
    /// resume binding as the session id only for the call that carries
    /// it. The journaled call identity is what the lane emitted, not
    /// every shape the template could take: a call whose placeholders
    /// all resolved absent produces a narrower document than its
    /// template suggests.
    setup_vars: Vec<ProbeVars>,
    /// The session's stored turn defaults — the selectors a pre-turn or
    /// prompt call inherits when the turn request does not override
    /// them — plus the open's working directory.
    session_model: Option<String>,
    session_effort: Option<String>,
    session_instructions: Option<String>,
    cwd: String,
    /// The setup-call index that binds the session id — the session
    /// call this open selects — so its successful response settles
    /// `session_id`'s presence for every later turn-phase call.
    session_call: Option<usize>,
    /// The session id the turn phase's calls substitute — a resume
    /// binding is known verbatim, a process-minted id is present but
    /// unreadable, and a peer-assigned id is unknowable until the
    /// session call's response supplies or omits it.
    session_id: std::sync::Mutex<SessionIdVar>,
    /// Secret presence journaled by the adapter's own resolutions: each
    /// `resolve()` records whether the resolver produced a value — never
    /// the value itself — and the outcome's position names the call
    /// whose substitution asked for it, so a skipped secret-only call
    /// cannot claim a later same-method frame on a newer resolution.
    secrets: Arc<std::sync::Mutex<SecretJournal>>,
    /// Requests journal against `setup_calls` until `reset` opens the
    /// turn phase — the open's calls precede every prompt — and
    /// against `turn_calls` afterwards, so a same-method collision
    /// across phases never attaches a request to the wrong call.
    setup_phase: AtomicBool,
    /// Sent setup requests still awaiting a response, in wire order:
    /// request id and whether the frame's params populated a declared
    /// model slot. The refusal's response id correlates the verdict to
    /// the exact call it answers.
    setup_journal: std::sync::Mutex<SetupJournal>,
    /// Whether a model-carrying pre-turn frame left this attempt —
    /// reset alongside the prompt flags each attempt.
    turn_model_sent: AtomicBool,
    /// The request id of this attempt's prompt frame — the key its
    /// synchronous response correlates on, so usage that lives only
    /// inside the reply payload still reaches the served check.
    prompt_id: std::sync::Mutex<Option<String>>,
    /// The request a journaled error response answered, when one has
    /// been observed — `Some(carries_model)` for journaled setup and
    /// pre-turn calls, `Some(true)` for the prompt itself.
    refusal: std::sync::Mutex<Option<bool>>,
    /// Whether a model-carrying setup call ever reached the wire — an
    /// open abandoned mid-flight may already have delivered the
    /// candidate it binds to a peer that never got to reply.
    model_setup_sent: AtomicBool,
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
    /// `request` is the open this session runs — the same request the
    /// lane's own substitution reads — so the probe's setup list and the
    /// selector values each declared call resolves against are the ones
    /// this open actually issues with, never a template-wide shape.
    fn new(inner: SharedObserver, spec: &AgentSpec, request: &SessionRequest) -> Arc<Self> {
        let resuming = resuming(request);
        let mut probe = Self {
            inner,
            method: None,
            method_field: "method".to_owned(),
            enveloped: false,
            response_method: None,
            discriminant_field: None,
            error_field: None,
            success_field: None,
            result_field: None,
            data_field: None,
            text_event_method: None,
            text_path: None,
            usage_path: None,
            result_selector: None,
            result_selector_field: None,
            result_path: None,
            session_id_path: None,
            cli: false,
            json_lines: false,
            overwritten: std::collections::BTreeSet::new(),
            prompt_carries_text: false,
            prompt_slots: Vec::new(),
            prompt_model_slots: Vec::new(),
            prompt_model_sent: AtomicBool::new(false),
            prompt_secret_count: 0,
            prompt_at_root: false,
            expected_prompts: std::sync::Mutex::new(Vec::new()),
            dispatched: AtomicBool::new(false),
            last_sent_was_prompt: AtomicBool::new(false),
            setup_calls: Vec::new(),
            turn_calls: Vec::new(),
            setup_vars: Vec::new(),
            session_model: None,
            session_effort: None,
            session_instructions: None,
            cwd: String::new(),
            session_call: None,
            session_id: std::sync::Mutex::new(SessionIdVar::Absent),
            secrets: Arc::new(std::sync::Mutex::new(SecretJournal::default())),
            setup_phase: AtomicBool::new(true),
            setup_journal: std::sync::Mutex::new(SetupJournal {
                claimed: Vec::new(),
                turn_claimed: Vec::new(),
                turn_vars: ProbeVars::default(),
                sent: Vec::new(),
            }),
            prompt_id: std::sync::Mutex::new(None),
            refusal: std::sync::Mutex::new(None),
            turn_model_sent: AtomicBool::new(false),
            model_setup_sent: AtomicBool::new(false),
            turn_event: AtomicBool::new(false),
            turn_billable: AtomicBool::new(false),
        };
        // Spawn-time env resolutions precede every call's — they occupy
        // the outcome stream's leading entries in `set`-map order, so
        // the setup phase's first call slice begins after them.
        probe
            .secrets
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .phase_start = env_secret_count(&spec.command.env);
        if let LaneSpec::Cli(cli) = &spec.lane {
            probe.usage_path = cli.usage_path.clone();
            probe.result_selector = cli.result_selector.clone();
            probe.result_selector_field = cli.result_selector_field.clone();
            probe.result_path = cli.result_path.clone();
            probe.session_id_path = cli.session_id_path.clone();
            probe.cli = true;
            probe.json_lines = cli.json_lines;
        }
        if let LaneSpec::Rpc(rpc) = &spec.lane {
            if !rpc.prompt.method.is_empty() {
                probe.method = Some(rpc.prompt.method.clone());
            }
            if let Some(envelope) = &rpc.envelope {
                probe.method_field = envelope.method_field.clone();
                probe.enveloped = true;
                probe.response_method = envelope.response_method.clone();
                probe.discriminant_field = envelope.discriminant_field.clone();
                probe.error_field = envelope.error_field.clone();
                probe.success_field = envelope.success_field.clone();
                probe.result_field = envelope.result_field.clone();
                probe.data_field = envelope.data_field.clone();
            }
            probe.text_event_method = rpc.text_event_method.clone();
            probe.text_path = rpc.text_path.clone();
            probe.usage_path = rpc.usage_path.clone();
            probe.prompt_slots = prompt_slots(&rpc.prompt.params);
            probe.prompt_model_slots = model_slots(&rpc.prompt.params);
            probe.prompt_secret_count = secret_occurrences(&rpc.prompt.params);
            probe.prompt_at_root = rpc
                .envelope
                .as_ref()
                .is_some_and(|envelope| envelope.params_inline)
                && rpc.prompt.params.is_object();
            // The setup calls this request issues and the pre-turn
            // calls every prompt issues — `rpc_setup_calls` mirrors the
            // adapter's own selection — so a shared wire method
            // attaches each request to the call that produced it.
            let setup_wire = |call: &a4aa::RpcCallSpec| SetupWire {
                method: call.method.clone(),
                params_at_root: rpc
                    .envelope
                    .as_ref()
                    .is_some_and(|envelope| envelope.params_inline)
                    && call.params.is_object(),
                template: call.params.clone(),
                model_slots: model_slots(&call.params),
            };
            let setup_calls = rpc_setup_calls(rpc, resuming);
            probe.turn_calls = rpc.pre_turn.iter().map(setup_wire).collect();
            // The selectors the lane substitutes at open and every turn:
            // the open request's session config for setup calls, the
            // session's stored defaults for the turn phase. The cwd the
            // lane resolves — request, then the command's, then the
            // temp dir — matches the lane's own precedence.
            let cwd = request
                .cwd
                .clone()
                .or_else(|| spec.command.cwd.clone())
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| std::env::temp_dir().display().to_string());
            probe.cwd = cwd.clone();
            probe.session_model = request.config.model.clone();
            probe.session_effort = request.config.effort.clone();
            probe.session_instructions = request.config.instructions.clone();
            // The session call is the setup list's tail when one exists;
            // only it substitutes the resume binding — `initialize` runs
            // without a session id. Every other setup var is the open
            // request's session config verbatim.
            let session_index = match rpc.session_scope {
                a4aa::SessionScope::Process => None,
                _ if resuming => rpc
                    .session_resume
                    .as_ref()
                    .or(rpc.session_open.as_ref())
                    .is_some()
                    .then(|| setup_calls.len() - 1),
                _ => rpc.session_open.is_some().then(|| setup_calls.len() - 1),
            };
            probe.session_call = session_index;
            probe.session_id =
                std::sync::Mutex::new(if rpc.session_scope == a4aa::SessionScope::Process {
                    // Process-scoped ids are minted per spawn — present on
                    // the wire but unknowable to the journal.
                    SessionIdVar::Wild
                } else {
                    match (
                        session_index.is_some(),
                        request.resume.as_deref().filter(|v| !v.trim().is_empty()),
                    ) {
                        // A resume binding is the session id verbatim.
                        (true, Some(id)) => SessionIdVar::Known(id.to_owned()),
                        // A fresh session's id is assigned by the peer's
                        // open response — unknowable presence and value
                        // until that response is journaled.
                        (true, None) => SessionIdVar::Flexible,
                        // No session call exists to assign an id: the
                        // adapter substitutes the id it holds — none —
                        // so `{sessionId}` is known-absent, never
                        // flexible, and an id-only call skips like any
                        // other call whose leaves all resolved absent.
                        (false, _) => SessionIdVar::Absent,
                    }
                });
            let config_input = |session_id: SessionIdVar| SelectorInput {
                cwd: &cwd,
                prompt: "",
                model: request.config.model.as_deref(),
                effort: request.config.effort.as_deref(),
                mode: request.config.mode.as_deref(),
                instructions: request.config.instructions.as_deref(),
                permission: request.config.permission.as_deref(),
                provider: request.config.provider.as_deref(),
                session_id,
            };
            // At open the session call substitutes the resume binding
            // itself — a fresh open has none yet, so `{sessionId}` is
            // simply absent there even though the turn phase's id is
            // peer-assigned and flexible.
            let resume_id = request
                .resume
                .as_deref()
                .filter(|v| !v.trim().is_empty())
                .map(|id| SessionIdVar::Known(id.to_owned()))
                .unwrap_or(SessionIdVar::Absent);
            probe.setup_vars = setup_calls
                .iter()
                .enumerate()
                .map(|(index, _)| {
                    selector_vars(config_input(if Some(index) == session_index {
                        resume_id.clone()
                    } else {
                        SessionIdVar::Absent
                    }))
                })
                .collect();
            probe.setup_calls = setup_calls.into_iter().map(setup_wire).collect();
            probe
                .setup_journal
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .claimed = vec![false; probe.setup_calls.len()];
            probe
                .setup_journal
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .turn_claimed = vec![false; probe.turn_calls.len()];
            if let Some(envelope) = rpc
                .envelope
                .as_ref()
                .filter(|envelope| envelope.params_inline)
            {
                // The inline encoder merges an object params document
                // into the frame root, then stamps the envelope's control
                // fields over the merge — the method field, every const
                // field, the request discriminant, and `id`. A template
                // leaf at one of those positions never reaches the wire:
                // it is not a verifiable position, and demanding it would
                // hide the evidence the surviving positions already prove.
                // Non-object params sit under `params` untouched, so the
                // filter applies only where the merge actually happened.
                let overwritten = inline_overwritten(envelope);
                if probe.prompt_at_root {
                    probe
                        .prompt_slots
                        .retain(|(path, _)| !overwritten.contains(path[0].as_str()));
                    probe
                        .prompt_model_slots
                        .retain(|(path, _)| !overwritten.contains(path[0].as_str()));
                }
                probe.overwritten = overwritten.clone();
                for call in probe
                    .setup_calls
                    .iter_mut()
                    .chain(probe.turn_calls.iter_mut())
                {
                    if call.params_at_root {
                        call.model_slots
                            .retain(|(path, _)| !overwritten.contains(path[0].as_str()));
                    }
                }
            }
            probe.prompt_carries_text = !probe.prompt_slots.is_empty();
        }
        Arc::new(probe)
    }

    /// Wrap the host's secret resolver so every resolution the adapter
    /// performs also journals the name's presence — never the value —
    /// into the call-matching journal. Resolution precedes the frame
    /// that uses it, so presence is always settled at claim time.
    fn wrap_secrets(&self, inner: Arc<dyn SecretResolver>) -> Arc<dyn SecretResolver> {
        Arc::new(ProbeSecrets {
            inner,
            journal: self.secrets.clone(),
        })
    }

    /// Whether the lane has an observable dispatch point — the wire
    /// journal then owns output attribution at frame receipt, and tap
    /// events carry no evidence of their own.
    fn has_dispatch_point(&self) -> bool {
        self.method.is_some()
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

    /// Whether the call an open-stage refusal answers carried the model.
    /// The journaled error response names the refused call by request id
    /// — exact even when `initialize` and `session/open` share a method
    /// or a peer callback interleaved before the refusal. Without an
    /// observed refusal response — a refusal the lane synthesized before
    /// any reply — the setup request still outstanding is the call it
    /// answers; a refusal before anything was sent cannot be about the
    /// model at all. A refusal from a model-free call repeats
    /// identically on every retry, so it is the lane's answer, not the
    /// model's.
    fn refused_setup_binds_model(&self) -> bool {
        if let Some(carries) = *self
            .refusal
            .lock()
            .unwrap_or_else(|error| error.into_inner())
        {
            return carries;
        }
        self.setup_journal
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .sent
            .last()
            .is_some_and(|sent| sent.carries_model)
    }

    /// Whether a turn-stage refusal is an answer about the model. The
    /// journaled error response names the refused call by request id —
    /// the prompt always binds the session's bound model, a journaled
    /// call binds exactly when it carried the model. Without an
    /// observed refusal the failed call is whatever was outstanding:
    /// once the prompt left, the refusal answers it; before that, the
    /// last journaled request carries the verdict, and nothing sent
    /// keeps the conservative model reading.
    fn refused_turn_binds_model(&self) -> bool {
        if let Some(binds) = *self
            .refusal
            .lock()
            .unwrap_or_else(|error| error.into_inner())
        {
            return binds;
        }
        if self.dispatched.load(Ordering::Acquire) {
            return true;
        }
        self.setup_journal
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .sent
            .last()
            .is_none_or(|sent| sent.carries_model)
    }

    /// Whether a model-carrying setup call already reached the wire: an
    /// open abandoned mid-flight may have delivered its candidate to the
    /// peer even though no reply arrived.
    fn model_setup_sent(&self) -> bool {
        self.model_setup_sent.load(Ordering::Acquire)
    }

    /// Whether a model-carrying pre-turn frame reached the wire this
    /// attempt — a pending candidate's send evidence below the prompt's
    /// own dispatch.
    fn turn_model_sent(&self) -> bool {
        self.turn_model_sent.load(Ordering::Acquire)
    }

    /// The presence map for one declared call's own resolutions: the
    /// adapter resolves a call's `{secret:NAME}` leaves contiguously in
    /// issue order, so the call owns the `outcomes` slice its occurrence
    /// count spans. A presence a later call's resolution produced on the
    /// same name belongs to that call — it never un-skips an earlier
    /// call whose own resolution came back empty.
    fn call_secrets(
        &self,
        journal: &SecretJournal,
        calls: &[SetupWire],
        index: usize,
    ) -> std::collections::BTreeMap<String, SecretPresence> {
        let offset: usize = calls
            .iter()
            .take(index)
            .map(|call| secret_occurrences(&call.template))
            .sum();
        Self::secret_slice(journal, offset, secret_occurrences(&calls[index].template))
    }

    /// The prompt's own resolutions — it substitutes after every
    /// declared pre-turn call inside the same turn phase.
    fn prompt_secrets_view(
        &self,
        journal: &SecretJournal,
    ) -> std::collections::BTreeMap<String, SecretPresence> {
        let offset: usize = self
            .turn_calls
            .iter()
            .map(|call| secret_occurrences(&call.template))
            .sum();
        Self::secret_slice(journal, offset, self.prompt_secret_count)
    }

    /// The presence records inside one call's outcome slice: a
    /// duplicate name accumulates both facts — the adapter's `values`
    /// keeps a returned value for embedded substitution while its
    /// `absent` list persists once a lookup came back empty, matching
    /// `vars`.
    fn secret_slice(
        journal: &SecretJournal,
        offset: usize,
        count: usize,
    ) -> std::collections::BTreeMap<String, SecretPresence> {
        journal
            .outcomes
            .iter()
            .skip(journal.phase_start + offset)
            .take(count)
            .fold(
                std::collections::BTreeMap::new(),
                |mut view, (name, resolved)| {
                    view.entry(name.clone())
                        .and_modify(|seen| {
                            seen.value |= *resolved;
                            seen.absent |= !*resolved;
                        })
                        .or_insert(SecretPresence {
                            value: *resolved,
                            absent: !*resolved,
                        });
                    view
                },
            )
    }

    /// Whether this attempt's prompt frame populated a surviving model
    /// slot — the request's model reached the wire. `true` for lanes
    /// without an observable dispatch point, whose model rides the spawn
    /// itself rather than a frame the journal can inspect.
    fn prompt_model_sent(&self) -> bool {
        self.method.is_none() || self.prompt_model_sent.load(Ordering::Acquire)
    }

    /// Whether an outcome's text is the retained raw stream rather than
    /// an actual result — a json-lines cli lane falls back to the
    /// retained NDJSON bytes when no result record supplied text, and
    /// every record inside it was already classified through the event
    /// relay. Text that parses as nothing but records is that fallback:
    /// it repeats the stream, it does not add evidence. Blank separators
    /// between records are part of the stream the lane ignored — never
    /// output of their own — but a stream holding no record at all is
    /// the lane's real output, whitespace included.
    fn retained_stream_text(&self, text: &str) -> bool {
        let mut records = false;
        self.json_lines
            && !text.is_empty()
            && text.lines().all(|line| {
                let line = line.trim_end_matches('\r');
                if line.trim().is_empty() {
                    return true;
                }
                let record = serde_json::from_str::<serde_json::Value>(line).is_ok();
                records |= record;
                record
            })
            && records
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
    /// event records never carry into it, the attempt's prompt text
    /// resolves the payload values the prompt call's frame must carry,
    /// and the attempt's own selectors — over the session's stored
    /// defaults — resolve what every pre-turn call emitted this round.
    fn reset(&self, prompt: &str, model: Option<&str>, effort: Option<&str>) {
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
        // The open's setup calls precede every prompt, so the first
        // reset marks the journal's turn phase: from here requests
        // claim `turn_calls`, never another open's setup entries.
        self.setup_phase.store(false, Ordering::Release);
        self.turn_model_sent.store(false, Ordering::Release);
        self.prompt_model_sent.store(false, Ordering::Release);
        {
            let mut journal = self
                .setup_journal
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            // The turn phase's substitution inputs: the attempt's
            // model/effort falling back to the session's stored values —
            // the same `or` chain the lane's own prompt applies — with
            // mode, permission, provider and turn id never populated at
            // turn time.
            journal.turn_vars = selector_vars(SelectorInput {
                cwd: &self.cwd,
                prompt,
                model: model.or(self.session_model.as_deref()),
                effort: effort.or(self.session_effort.as_deref()),
                mode: None,
                instructions: self.session_instructions.as_deref(),
                permission: None,
                provider: None,
                session_id: self
                    .session_id
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .clone(),
            });
            journal.sent.clear();
            journal
                .claimed
                .iter_mut()
                .for_each(|claimed| *claimed = false);
            journal
                .turn_claimed
                .iter_mut()
                .for_each(|claimed| *claimed = false);
        }
        {
            let mut secrets = self
                .secrets
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            // The turn phase's first call resolves from here — the
            // outcomes already journaled belong to env and the open's
            // setup calls.
            secrets.phase_start = secrets.outcomes.len();
        }
        *self
            .prompt_id
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = None;
        *self
            .refusal
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = None;
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

    /// The event-tap form of the same normalization: an rpc lane
    /// rebuilds a relayed notification as a `{method, params}` document
    /// already decoded from the frame, while a cli lane's raw record is
    /// the record itself — either way its carriers sit at their
    /// conventional positions regardless of the wire envelope.
    fn billable_raw(&self, raw: &serde_json::Value) -> bool {
        if self.cli {
            return self.billable_notification(None, raw);
        }
        self.billable_notification(
            raw.get("method").and_then(serde_json::Value::as_str),
            raw.get("params").unwrap_or(raw),
        )
    }

    /// Whether a raw record's only evidence claim is bookkeeping — a
    /// usage document, an empty result at the lane's declared result
    /// position, or the session binding at its declared `session_id_path`
    /// — versus genuine turn progress. The lane's own
    /// selection leads: a record matching the declared result selector
    /// is bookkeeping exactly when its result text is empty, regardless
    /// of a usage claim beside it. Every other record is bookkeeping
    /// only when a usage document or a session-init marker is its whole
    /// claim — one that also carries progress text is real output even
    /// with all-zero counters, and one carrying no usage at all is
    /// progress by default. The declared `usage_path` leads; the
    /// conventional `usage` / `message.usage` positions follow, resolved
    /// where `billable_raw` reads the record.
    fn bookkeeping_raw(&self, raw: &serde_json::Value) -> bool {
        let root = if self.cli {
            raw
        } else {
            raw.get("params").unwrap_or(raw)
        };
        if let (Some(selector), Some(path)) =
            (self.result_selector.as_deref(), self.result_path.as_deref())
        {
            let field = self.result_selector_field.as_deref().unwrap_or("type");
            if root.get(field).and_then(serde_json::Value::as_str) == Some(selector) {
                return extract_path(root, path)
                    .and_then(serde_json::Value::as_str)
                    .is_none_or(|text| text.trim().is_empty());
            }
        }
        // A session-init record is bookkeeping the same way a usage-only
        // record is: its whole claim is the session binding the lane's
        // declared `session_id_path` reads. A record that also carries
        // progress text or a usage document still counts on its own
        // merits — the init marker does not launder real output.
        let session_init = self.session_id_path.as_deref().is_some_and(|path| {
            extract_path(root, path)
                .and_then(serde_json::Value::as_str)
                .is_some_and(|id| !id.trim().is_empty())
        });
        let usage_only = self
            .usage_path
            .as_deref()
            .and_then(|path| extract_path(root, path))
            .or_else(|| root.get("usage"))
            .or_else(|| extract_path(root, "message.usage"))
            .is_some_and(|usage| usage.is_object());
        (usage_only || session_init) && !record_progress(root)
    }

    /// Whether a synchronous prompt response's payload proves billed
    /// work: the result body or the error object may carry a usage
    /// record at the declared or conventional carriers — nested under
    /// `data` on an error payload — that never rode a notification.
    fn usage_billable(&self, payload: &serde_json::Value) -> bool {
        [Some(payload), payload.get("data")]
            .into_iter()
            .flatten()
            .any(|doc| {
                self.usage_path
                    .as_deref()
                    .and_then(|path| extract_path(doc, path))
                    .or_else(|| doc.get("usage"))
                    .or_else(|| extract_path(doc, "message.usage"))
                    .is_some_and(normalized_billable)
            })
    }

    /// Whether an outbound frame answers a peer-initiated request — the
    /// same response discrimination the peer's inbound dispatch applies.
    /// A response is never a setup request: an envelope that echoes the
    /// answered command on replies would otherwise let a callback reply
    /// steal a same-method setup call's journal claim.
    fn outbound_response(&self, frame: &serde_json::Value) -> bool {
        if !self.enveloped {
            return false;
        }
        self.response_method.as_deref().is_some_and(|marker| {
            frame
                .get(
                    self.discriminant_field
                        .as_deref()
                        .unwrap_or(self.method_field.as_str()),
                )
                .and_then(serde_json::Value::as_str)
                == Some(marker)
        })
    }

    /// Whether an inbound response frame reports a failure — the same
    /// classification the peer's dispatch applies: a stock `error`
    /// member, or under a declared envelope the configured error content
    /// with the success marker honoured when one exists.
    fn response_is_error(&self, frame: &serde_json::Value) -> bool {
        if !self.enveloped {
            return frame.get("error").is_some();
        }
        let error_content = frame
            .get(self.error_field.as_deref().unwrap_or("error"))
            .filter(|error| !error.is_null());
        match self.success_field.as_deref() {
            Some(field) => {
                frame.get(field) != Some(&serde_json::Value::Bool(true)) || error_content.is_some()
            }
            None => error_content.is_some(),
        }
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

/// The `{model}`-family placeholder names a model slot can declare —
/// the same set the lane's selector discovery reads.
const MODEL_NAMES: &[&str] = &["model", "modelProvider", "modelId"];

/// Whether a model slot's leaf carried a selector value under `vars`:
/// a populated wire field is model-carrying only when one of the
/// leaf's model tokens resolved to something — `provider:{modelProvider}`
/// still emits `provider:` for an unqualified candidate, and that
/// scaffolding is no model the peer could have bound.
fn leaf_carries_model(leaf: &str, vars: &ProbeVars) -> bool {
    MODEL_NAMES.iter().any(|name| {
        leaf.contains(&format!("{{{name}}}"))
            && vars.known.get(*name).is_some_and(|value| !value.is_empty())
    })
}

/// Whether a model slot's leaf resolves a selector value for this
/// candidate — the same per-token reading `leaf_carries_model` makes of
/// the resolved vars, computed before any wire journal exists:
/// `{model}` resolves for every non-empty candidate, `{modelId}` only
/// for one whose post-slash id part is non-empty — `provider/` resolves
/// it to nothing, the same `split_once('/')` the adapter's vars apply —
/// and `{modelProvider}` only for a `provider/model` selection.
fn leaf_resolves_model(leaf: &str, model: &str) -> bool {
    MODEL_NAMES.iter().any(|name| {
        leaf.contains(&format!("{{{name}}}"))
            && match *name {
                "modelProvider" => model
                    .split_once('/')
                    .is_some_and(|(provider, _)| !provider.is_empty()),
                "modelId" => !model
                    .split_once('/')
                    .map(|(_, id)| id)
                    .unwrap_or(model)
                    .is_empty(),
                _ => !model.is_empty(),
            }
    })
}

fn model_slots(template: &serde_json::Value) -> Vec<(Vec<String>, String)> {
    fn walk(
        value: &serde_json::Value,
        path: &mut Vec<String>,
        slots: &mut Vec<(Vec<String>, String)>,
    ) {
        match value {
            serde_json::Value::String(leaf)
                if MODEL_NAMES
                    .iter()
                    .any(|name| leaf.contains(&format!("{{{name}}}"))) =>
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

/// The `{secret:NAME}` occurrences inside a params template, counted in
/// the document order the lane's `secret_names` walk produces — the
/// number of `resolve()` records the adapter writes when it substitutes
/// this call. An unclosed `secret:` token is literal text, the same
/// read the lane makes.
fn secret_occurrences(template: &serde_json::Value) -> usize {
    match template {
        serde_json::Value::Object(map) => map.values().map(secret_occurrences).sum(),
        serde_json::Value::Array(items) => items.iter().map(secret_occurrences).sum(),
        serde_json::Value::String(leaf) => {
            let mut count = 0;
            let mut rest = leaf.as_str();
            while let Some(open) = rest.find("{secret:") {
                let tail = &rest[open + 8..];
                match tail.find('}') {
                    Some(end) => {
                        count += 1;
                        rest = &tail[end + 1..];
                    }
                    None => break,
                }
            }
            count
        }
        _ => 0,
    }
}

/// The `{"secret": "ref"}` env entries the lane resolves before it
/// issues any call — in `set`-map order, contiguous, so the outcome
/// stream's setup slice begins after them. Extra env entries the open
/// request carries are literals and resolve nothing.
fn env_secret_count(env: &EnvPolicy) -> usize {
    match env {
        EnvPolicy::Sanitized { set, .. } | EnvPolicy::InheritAll { set } => set
            .values()
            .filter(|value| {
                matches!(
                    value,
                    a4aa::EnvValue::Source(a4aa::EnvSource::Secret { .. })
                )
            })
            .count(),
    }
}

/// The params document a journaled request frame carries, normalized
/// the way the lane's encoder produced it: an inline envelope merges an
/// object document into the frame root, and a null or absent `params`
/// member resolves against the template's own shape — an object
/// template's call emits the empty document, a null template's emits
/// null. Keeping the null lets a default-null call claim its own frame.
fn frame_params(
    frame: &serde_json::Value,
    call: &SetupWire,
    overwritten: &std::collections::BTreeSet<String>,
) -> serde_json::Value {
    if call.params_at_root {
        strip_overwritten(frame, overwritten)
    } else {
        match frame.get("params") {
            Some(params) if !params.is_null() => params.clone(),
            _ if call.template.is_object() => serde_json::Value::Object(Default::default()),
            _ => serde_json::Value::Null,
        }
    }
}

/// Whether the frame's params prove a declared model slot carried a
/// selector value: the leaf's `{model}`-family token resolved non-empty
/// under `vars` — a populated field whose model token came back empty
/// is scaffolding — and the string the frame carries at the slot's
/// position is the resolved leaf, or at least contains the resolved
/// token when an unreadable placeholder shares the leaf. The template's
/// declaration alone never credits a send another call produced.
fn slot_carries_model(
    params: &serde_json::Value,
    path: &[String],
    leaf: &str,
    vars: &ProbeVars,
    secrets: &std::collections::BTreeMap<String, SecretPresence>,
) -> bool {
    if !leaf_carries_model(leaf, vars) {
        return false;
    }
    let Some(value) = slot_at(params, path).and_then(serde_json::Value::as_str) else {
        return false;
    };
    match expected_leaf(leaf, vars, secrets) {
        Some(expected) => value == expected,
        None => MODEL_NAMES.iter().any(|name| {
            leaf.contains(&format!("{{{name}}}"))
                && vars
                    .known
                    .get(*name)
                    .is_some_and(|known| !known.is_empty() && value.contains(known.as_str()))
        }),
    }
}

/// The journal key for a request id — the same normalization the peer
/// applies to pending calls: numeric ids by value, string ids verbatim.
fn setup_id_key(id: &serde_json::Value) -> Option<String> {
    id.as_u64()
        .map(|serial| serial.to_string())
        .or_else(|| id.as_str().map(str::to_owned))
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

/// The session id a session call's result binds — the same extraction
/// the lane applies to its open or resume answer: `sessionId`,
/// `session_id` or `id`, one object level deep included.
fn response_binding(result: &serde_json::Value) -> Option<String> {
    fn id_in(map: &serde_json::Map<String, serde_json::Value>) -> Option<String> {
        ["sessionId", "session_id", "id"]
            .iter()
            .find_map(|key| map.get(*key)?.as_str().map(str::to_owned))
    }
    let map = result.as_object()?;
    id_in(map).or_else(|| {
        map.values()
            .find_map(|value| value.as_object().and_then(id_in))
    })
}

/// Walk a dotted path into a JSON value.
fn extract_path<'a>(value: &'a serde_json::Value, path: &str) -> Option<&'a serde_json::Value> {
    let mut cursor = value;
    for segment in path.split('.') {
        cursor = cursor.get(segment)?;
    }
    Some(cursor)
}

/// An object document without the members an inline envelope stamps
/// over the merged params — the journaled call cannot be held to a
/// position the encoder replaced.
fn strip_overwritten(
    value: &serde_json::Value,
    overwritten: &std::collections::BTreeSet<String>,
) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => serde_json::Value::Object(
            map.iter()
                .filter(|(key, _)| !overwritten.contains(key.as_str()))
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// The selector values a phase's calls resolve against, mirroring the
/// lane's `vars`: always-populated placeholders land in `known` — `""`
/// when unset — while an unset optional selector also lands in `absent`
/// so a whole-leaf member drops. A peer-assigned or minted session id
/// the journal cannot read stays out of `known`: present-but-unknowable
/// is `wild`, maybe-present-maybe-absent is `flexible`.
fn selector_vars(input: SelectorInput) -> ProbeVars {
    let mut vars = ProbeVars::default();
    let model = input.model.unwrap_or_default();
    let (model_provider, model_id) = model.split_once('/').unwrap_or(("", model));
    let mut known = |name: &str, value: &str| {
        vars.known.insert(name.to_owned(), value.to_owned());
    };
    known("cwd", input.cwd);
    known("prompt", input.prompt);
    known("model", model);
    known("modelId", model_id);
    known("modelProvider", model_provider);
    known("effort", input.effort.unwrap_or_default());
    known("mode", input.mode.unwrap_or_default());
    known("instructions", input.instructions.unwrap_or_default());
    known("permission", input.permission.unwrap_or_default());
    known("provider", input.provider.unwrap_or_default());
    // The absent-option list the lane reports: a whole-leaf member with
    // one of these names drops from the resolved document. `turnId` is
    // never populated on the calls the journal tracks.
    for (name, value) in [
        ("model", input.model),
        ("modelId", input.model),
        (
            "modelProvider",
            input
                .model
                .and_then(|model| model.split_once('/').map(|(provider, _)| provider)),
        ),
        ("effort", input.effort),
        ("mode", input.mode),
        ("instructions", input.instructions),
        ("permission", input.permission),
        ("provider", input.provider),
    ] {
        if value.is_none() {
            vars.absent.insert(name.to_owned());
        }
    }
    vars.absent.insert("turnId".to_owned());
    match input.session_id {
        SessionIdVar::Known(id) => {
            vars.known.insert("sessionId".to_owned(), id);
        }
        SessionIdVar::Wild => {
            vars.wild.insert("sessionId".to_owned());
        }
        SessionIdVar::Flexible => {
            vars.flexible.insert("sessionId".to_owned());
        }
        SessionIdVar::Absent => {
            vars.absent.insert("sessionId".to_owned());
        }
    }
    vars
}

/// How one `{placeholder}` name resolves in `vars`: a value the journal
/// knows, a known-absent optional selector, a present-but-unknowable
/// value, a maybe-present-maybe-absent unknown, or an undeclared name
/// the lane leaves literal in the document.
enum LeafClass {
    Known,
    Absent,
    Wild,
    Flexible,
    Literal,
}

/// `whole` distinguishes the two questions a secret name answers: a
/// whole-leaf member drops on the `absent` list, while an embedded
/// token substitutes whatever `values` retained — mixed resolutions on
/// one name carry both facts, so neither collapses into the other.
fn leaf_class(
    name: &str,
    vars: &ProbeVars,
    secrets: &std::collections::BTreeMap<String, SecretPresence>,
    whole: bool,
) -> LeafClass {
    if vars.absent.contains(name) {
        LeafClass::Absent
    } else if vars.flexible.contains(name) {
        LeafClass::Flexible
    } else if name.starts_with("secret:") {
        // A secret resolves through the adapter's own resolver — the
        // journaled presence is the answer the substitution used. A
        // whole-leaf member drops when any lookup in the call's slice
        // came back empty — a value another lookup retained does not
        // revive it. An embedded token substitutes whatever a returned
        // value left in `values`; with no value at all it stays
        // literal. Before any resolution the name is merely possible.
        match secrets.get(name) {
            Some(presence) if whole => {
                if presence.absent {
                    LeafClass::Absent
                } else {
                    LeafClass::Wild
                }
            }
            Some(presence) => {
                if presence.value {
                    LeafClass::Wild
                } else {
                    LeafClass::Absent
                }
            }
            None => LeafClass::Flexible,
        }
    } else if vars.wild.contains(name) {
        LeafClass::Wild
    } else if vars.known.contains_key(name) {
        LeafClass::Known
    } else {
        LeafClass::Literal
    }
}

/// The name a whole-leaf `{placeholder}` declares — the only member
/// shape the lane's substitution drops from an object.
fn whole_placeholder(leaf: &serde_json::Value) -> Option<&str> {
    let serde_json::Value::String(text) = leaf else {
        return None;
    };
    text.strip_prefix('{')
        .and_then(|tail| tail.strip_suffix('}'))
        .filter(|name| !name.contains(['{', '}']))
}

/// The string a template leaf resolves to under `vars`, or `None` when
/// a placeholder's value is unknowable — the lane's own
/// `substitute_text` tokenization: unmatched braces stay literal and
/// any name the resolved selector table does not hold stays literal,
/// `sessionId`/`turnId`/secret included — absent values never enter the
/// adapter's `values`, so embedded tokens keep their braces.
fn expected_leaf(
    leaf: &str,
    vars: &ProbeVars,
    secrets: &std::collections::BTreeMap<String, SecretPresence>,
) -> Option<String> {
    let mut output = String::new();
    let mut rest = leaf;
    while let Some(open) = rest.find('{') {
        output.push_str(&rest[..open]);
        let tail = &rest[open + 1..];
        match tail.find(['{', '}']) {
            Some(end) if tail.as_bytes()[end] == b'}' => {
                let name = &tail[..end];
                match leaf_class(name, vars, secrets, false) {
                    LeafClass::Wild | LeafClass::Flexible => return None,
                    _ => match vars.known.get(name) {
                        // A value the lane put in `values` — an empty
                        // string included — contributes exactly it.
                        Some(value) => output.push_str(value),
                        // A name the lane never put in `values` — an
                        // absent optional id or an unresolved secret —
                        // stays literal where it sits.
                        None => {
                            output.push('{');
                            output.push_str(&tail[..=end]);
                        }
                    },
                }
                rest = &tail[end + 1..];
            }
            _ => {
                output.push('{');
                rest = tail;
            }
        }
    }
    output.push_str(rest);
    Some(output)
}

/// Whether a member missing from the frame can be one the lane dropped
/// — a whole-leaf placeholder that is provably absent, or a flexible
/// leaf whose presence was never certain. A member whose placeholder
/// is known present cannot be missing from its call's frame.
fn droppable(
    leaf: &serde_json::Value,
    vars: &ProbeVars,
    secrets: &std::collections::BTreeMap<String, SecretPresence>,
) -> bool {
    whole_placeholder(leaf).is_some_and(|name| {
        matches!(
            leaf_class(name, vars, secrets, true),
            LeafClass::Absent | LeafClass::Flexible
        )
    })
}

/// Whether `template` could have resolved to the params a journaled
/// request carries under `vars` — the resolved call's identity when
/// declared calls share a wire method. Every member the frame carries
/// must trace to a template position: a literal must match exactly and
/// a placeholder leaf must resolve to the frame's value, while a leaf
/// holding an unknowable placeholder accepts any string. Every member
/// it lacks must be a droppable whole-leaf `{placeholder}` — the only
/// shape the lane's substitution drops. Nested objects keep their
/// position the way the lane emits them: an object whose leaves all
/// dropped arrives as `{}`.
fn resolved_call_consistent(
    template: &serde_json::Value,
    params: &serde_json::Value,
    vars: &ProbeVars,
    secrets: &std::collections::BTreeMap<String, SecretPresence>,
) -> bool {
    fn produces(
        template: &serde_json::Value,
        params: &serde_json::Value,
        vars: &ProbeVars,
        secrets: &std::collections::BTreeMap<String, SecretPresence>,
    ) -> bool {
        match (template, params) {
            (serde_json::Value::Object(tmap), serde_json::Value::Object(pmap)) => {
                pmap.keys().all(|key| tmap.contains_key(key))
                    && tmap.iter().all(|(key, leaf)| match pmap.get(key) {
                        Some(value) => produces(leaf, value, vars, secrets),
                        None => droppable(leaf, vars, secrets),
                    })
            }
            (serde_json::Value::String(leaf), serde_json::Value::String(value)) => {
                expected_leaf(leaf, vars, secrets).is_none_or(|expected| expected == *value)
            }
            (serde_json::Value::Array(titems), serde_json::Value::Array(pitems)) => {
                titems.len() == pitems.len()
                    && titems
                        .iter()
                        .zip(pitems.iter())
                        .all(|(template, params)| produces(template, params, vars, secrets))
            }
            _ => template == params,
        }
    }
    produces(template, params, vars, secrets)
}

/// Whether a pre-turn call resolves to the empty params object the lane
/// skips without emitting a frame — every member of a non-empty object
/// template a whole-leaf placeholder that is provably absent. A
/// maybe-present flexible member keeps the call claimable: the lane
/// may have emitted it after all. Setup calls are never skipped — the
/// lane issues whatever the template resolved to, `{}` included.
fn turn_call_skipped(
    template: &serde_json::Value,
    vars: &ProbeVars,
    secrets: &std::collections::BTreeMap<String, SecretPresence>,
) -> bool {
    let serde_json::Value::Object(map) = template else {
        return false;
    };
    !map.is_empty()
        && map.values().all(|leaf| {
            whole_placeholder(leaf).is_some_and(|name| {
                matches!(leaf_class(name, vars, secrets, true), LeafClass::Absent)
            })
        })
}

/// Mirror of the lane's notification text extraction, reduced to a
/// presence check: a declared `text_path` resolves a string leaf or an
/// array whose `{type:"text",text}` blocks concatenate to text; otherwise
/// the conventional fields (`text`, `content`, `delta`, `item.text`) or a
/// bare string.
fn notification_text(params: &serde_json::Value, text_path: Option<&str>) -> Option<()> {
    if let Some(path) = text_path {
        return match extract_path(params, path) {
            Some(serde_json::Value::String(text)) if !text.is_empty() => Some(()),
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
        .filter(|text| !text.is_empty())
        .map(|_| ())
        .or_else(|| {
            extract_path(params, "item.text")?
                .as_str()
                .filter(|text| !text.is_empty())
                .map(|_| ())
        })
        .or_else(|| params.as_str().filter(|text| !text.is_empty()).map(|_| ()))
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

/// Whether a raw stream record carries progress text — a non-empty
/// string at the conventional carriers (`text`, `content`, `delta`,
/// `message`, `result`, `item.text`, `message.content`,
/// `message.text`), where a block array counts by its `text` leaves and
/// by any non-`text` block it holds: a tool call, a tool result or a
/// thinking block is activity of its own whether or not it emits text.
/// A record with no progress claim is bookkeeping-only when a usage
/// document is its only payload.
fn record_progress(record: &serde_json::Value) -> bool {
    fn claim(value: Option<&serde_json::Value>) -> bool {
        match value {
            Some(serde_json::Value::String(text)) => !text.trim().is_empty(),
            Some(serde_json::Value::Array(blocks)) => blocks.iter().any(|block| {
                block
                    .get("text")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|text| !text.trim().is_empty())
                    || block
                        .get("type")
                        .and_then(serde_json::Value::as_str)
                        .is_some_and(|kind| kind != "text")
            }),
            _ => false,
        }
    }
    [
        record.get("text"),
        record.get("content"),
        record.get("delta"),
        record.get("message"),
        record.get("result"),
        extract_path(record, "item.text"),
        extract_path(record, "message.content"),
        extract_path(record, "message.text"),
    ]
    .into_iter()
    .any(claim)
}

/// The adapter's secret resolver, wrapped to journal each resolution's
/// presence for the probe's call matching: `substitute_resolved`
/// resolves a call's `{secret:NAME}` leaves immediately before that
/// call's frame is written — or the call is skipped — so the outcome's
/// position in the stream attributes presence to the call whose
/// substitution asked for it. Only presence is recorded, never the
/// secret value itself.
struct ProbeSecrets {
    inner: Arc<dyn SecretResolver>,
    journal: Arc<std::sync::Mutex<SecretJournal>>,
}

#[async_trait::async_trait]
impl SecretResolver for ProbeSecrets {
    async fn resolve(&self, reference: &str) -> Option<String> {
        let resolved = self.inner.resolve(reference).await;
        self.journal
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .outcomes
            .push((format!("secret:{reference}"), resolved.is_some()));
        resolved
    }
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
                let frame = serde_json::from_slice::<serde_json::Value>(bytes).ok();
                let outbound_response = frame
                    .as_ref()
                    .is_some_and(|frame| self.outbound_response(frame));
                let setup_phase = self.setup_phase.load(Ordering::Acquire);
                let mut prompt = false;
                let mut claimed_model = None;
                if let Some(frame) = frame.as_ref() {
                    if !outbound_response {
                        // A declared setup or pre-turn request is
                        // journaled by request id: the refusal's
                        // response then names the exact call it
                        // answers. Every request claims its actual call
                        // before it may read as the prompt — the adapter
                        // issues the phase's calls ahead of the prompt,
                        // so a same-method frame belongs to the earliest
                        // unclaimed call whose resolved shape produced
                        // it. In the setup phase the prompt never
                        // issues; once the turn phase holds only a
                        // request no pre-turn call produced can be the
                        // prompt. Outbound callback replies are response
                        // frames, never requests. Whether the frame's
                        // params populate a declared model slot decides
                        // if a chain entry could change that call's
                        // answer.
                        let request = frame.get("id").and_then(setup_id_key).zip(
                            frame
                                .get(self.method_field.as_str())
                                .and_then(serde_json::Value::as_str),
                        );
                        if let Some((id, method)) = request {
                            claimed_model = {
                                let mut guard = self
                                    .setup_journal
                                    .lock()
                                    .unwrap_or_else(|error| error.into_inner());
                                let journal = &mut *guard;
                                let (calls, flags) = if setup_phase {
                                    (&self.setup_calls, &mut journal.claimed)
                                } else {
                                    (&self.turn_calls, &mut journal.turn_claimed)
                                };
                                let secret_journal = self
                                    .secrets
                                    .lock()
                                    .unwrap_or_else(|error| error.into_inner());
                                // A request claims the first unclaimed
                                // call whose resolved shape the frame
                                // actually is — the adapter's own issue
                                // order, journaled at issuance against
                                // the selector values this open or turn
                                // substitutes. A pre-turn call whose
                                // leaves all resolved absent emitted no
                                // frame at all: it cannot claim a
                                // same-method request a later call
                                // produced, and a skipped call's slot
                                // shifts onto the call the adapter
                                // really issued.
                                if let Some(index) =
                                    calls.iter().enumerate().find_map(|(index, call)| {
                                        if flags[index] || call.method != method {
                                            return None;
                                        }
                                        let vars = if setup_phase {
                                            &self.setup_vars[index]
                                        } else {
                                            &journal.turn_vars
                                        };
                                        // The call's own resolutions —
                                        // a skipped call's skip decision
                                        // is the presence its
                                        // substitution saw, never a
                                        // newer resolution on the same
                                        // name.
                                        let secrets =
                                            self.call_secrets(&secret_journal, calls, index);
                                        let params = frame_params(frame, call, &self.overwritten);
                                        let template = if call.params_at_root {
                                            strip_overwritten(&call.template, &self.overwritten)
                                        } else {
                                            call.template.clone()
                                        };
                                        ((setup_phase
                                            || !turn_call_skipped(&call.template, vars, &secrets))
                                            && resolved_call_consistent(
                                                &template, &params, vars, &secrets,
                                            ))
                                        .then_some(index)
                                    })
                                {
                                    flags[index] = true;
                                    let call = &calls[index];
                                    let vars = if setup_phase {
                                        &self.setup_vars[index]
                                    } else {
                                        &journal.turn_vars
                                    };
                                    let secrets = self.call_secrets(&secret_journal, calls, index);
                                    let params = frame_params(frame, call, &self.overwritten);
                                    // A claimed call carried the model
                                    // only where the frame's own value
                                    // at a declared slot proves a
                                    // `{model}`-family token resolved
                                    // to a value — a populated field
                                    // whose model token came back empty
                                    // is scaffolding, not a model the
                                    // peer could have bound.
                                    let carries_model =
                                        call.model_slots.iter().any(|(path, leaf)| {
                                            slot_carries_model(&params, path, leaf, vars, &secrets)
                                        });
                                    journal.sent.push(SentSetup {
                                        id,
                                        carries_model,
                                        binds_session: Some(index) == self.session_call,
                                    });
                                    Some(carries_model)
                                } else {
                                    None
                                }
                            };
                        }
                        // Only a request no declared call produced can
                        // be the prompt — and only once the turn phase
                        // holds: the open's calls precede every prompt.
                        if claimed_model.is_none() && !setup_phase && self.is_prompt_frame(frame) {
                            prompt = true;
                            self.dispatched.store(true, Ordering::Release);
                            // The selector must have reached this frame:
                            // a surviving model slot marks the frame
                            // model-carrying only when the frame's own
                            // params prove a `{model}`-family token
                            // resolved to a value —
                            // `provider:{modelProvider}` still emits
                            // `provider:` for an unqualified candidate,
                            // and that scaffolding carried no model the
                            // peer could have bound.
                            let model_sent = {
                                let journal = self
                                    .setup_journal
                                    .lock()
                                    .unwrap_or_else(|error| error.into_inner());
                                let secret_journal = self
                                    .secrets
                                    .lock()
                                    .unwrap_or_else(|error| error.into_inner());
                                let secrets = self.prompt_secrets_view(&secret_journal);
                                let params = if self.prompt_at_root {
                                    strip_overwritten(frame, &self.overwritten)
                                } else {
                                    frame
                                        .get("params")
                                        .cloned()
                                        .unwrap_or(serde_json::Value::Null)
                                };
                                self.prompt_model_slots.iter().any(|(path, leaf)| {
                                    slot_carries_model(
                                        &params,
                                        path,
                                        leaf,
                                        &journal.turn_vars,
                                        &secrets,
                                    )
                                })
                            };
                            if model_sent {
                                self.prompt_model_sent.store(true, Ordering::Release);
                            }
                            // The request id correlates the prompt's
                            // synchronous response — usage that lives
                            // only inside that reply still proves billed
                            // work happened.
                            if let Some(id) = frame.get("id").and_then(setup_id_key) {
                                *self
                                    .prompt_id
                                    .lock()
                                    .unwrap_or_else(|error| error.into_inner()) = Some(id);
                            }
                        }
                    }
                }
                // Every `Sent` refreshes which frame the writer is on:
                // the record precedes the write, so the last observed
                // `Sent` names the write a `writer_failed` report refers
                // to.
                self.last_sent_was_prompt.store(prompt, Ordering::Release);
                if claimed_model == Some(true) {
                    if setup_phase {
                        self.model_setup_sent.store(true, Ordering::Release);
                    } else {
                        self.turn_model_sent.store(true, Ordering::Release);
                    }
                }
            }
            // Inbound frames: a response settles a pending call, a request
            // (id + method) or a notification (id-less) is agent activity.
            // A response is correlated to the journaled setup request its
            // id answers — before the dispatch gate, since setup calls
            // complete before any prompt — while requests and
            // notifications become the turn's output only once the prompt
            // reached the wire: receipt cannot precede the `Sent` record,
            // so dispatch-gated attribution is exact even when the lane's
            // delivery task has not forwarded the frame to the tap yet.
            Direction::Received => {
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
                    // An error response's id names the refused setup
                    // call — the verdict `refused_setup_binds_model`
                    // reads. A successful answer just retires its
                    // journal entry.
                    if let Some(key) = id.and_then(setup_id_key) {
                        let sent = {
                            let mut journal = self
                                .setup_journal
                                .lock()
                                .unwrap_or_else(|error| error.into_inner());
                            journal
                                .sent
                                .iter()
                                .position(|sent| sent.id == key)
                                .map(|position| journal.sent.remove(position))
                        };
                        if let Some(sent) = sent {
                            if self.response_is_error(&frame) {
                                *self
                                    .refusal
                                    .lock()
                                    .unwrap_or_else(|error| error.into_inner()) =
                                    Some(sent.carries_model);
                            } else if sent.binds_session {
                                // The session call's answer settles the
                                // session id's presence — the presence
                                // the adapter substitutes into every
                                // later call. A peer that assigns no id
                                // leaves `{sessionId}` absent, not
                                // flexible.
                                let binding = frame
                                    .get(self.result_field.as_deref().unwrap_or("result"))
                                    .and_then(response_binding);
                                let mut session_id = self
                                    .session_id
                                    .lock()
                                    .unwrap_or_else(|error| error.into_inner());
                                if matches!(*session_id, SessionIdVar::Flexible) {
                                    *session_id = binding
                                        .map(SessionIdVar::Known)
                                        .unwrap_or(SessionIdVar::Absent);
                                }
                            }
                        } else if self
                            .prompt_id
                            .lock()
                            .unwrap_or_else(|error| error.into_inner())
                            .as_deref()
                            == Some(key.as_str())
                        {
                            // The prompt's own reply is in-turn evidence
                            // by correlation — A4AA surfaces
                            // notification-carried usage in the outcome,
                            // but usage that exists only inside this
                            // response payload still proves the paid
                            // turn ran and must reach the served check.
                            self.turn_event.store(true, Ordering::Release);
                            let result =
                                frame.get(self.result_field.as_deref().unwrap_or("result"));
                            let error = frame.get(self.error_field.as_deref().unwrap_or("error"));
                            if [result, error]
                                .into_iter()
                                .flatten()
                                .any(|payload| self.usage_billable(payload))
                            {
                                self.turn_billable.store(true, Ordering::Release);
                            }
                            // A refusal to the prompt always answers the
                            // model the session bound — a journaled call
                            // binds only when it carried the model.
                            if self.response_is_error(&frame) {
                                *self
                                    .refusal
                                    .lock()
                                    .unwrap_or_else(|error| error.into_inner()) = Some(true);
                            }
                        }
                    }
                    return;
                }
                if !self.dispatched.load(Ordering::Acquire) {
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
        AgentEvent::MessageChunk { text } => !text.is_empty(),
        AgentEvent::ToolCall { .. } | AgentEvent::Plan { .. } | AgentEvent::Truncated => true,
        AgentEvent::Usage { usage, .. } => billable_usage(usage),
        AgentEvent::Raw { raw } => probe.billable_raw(raw),
    }
}

/// Whether one relayed lane event proves the attempt produced output
/// the caller saw — the billable evidence of `turn_output`, or genuine
/// turn progress a raw lane stream carries: any record that is not
/// exclusively bookkeeping. An empty chunk, an all-zero usage record
/// and a raw record whose only claim is a usage document never count —
/// a lane that emitted nothing but zero-usage bookkeeping and then
/// errored did no served work.
fn turn_progress(event: &AgentEvent, probe: &DispatchProbe) -> bool {
    match event {
        AgentEvent::Raw { raw } => probe.billable_raw(raw) || !probe.bookkeeping_raw(raw),
        _ => turn_output(event, probe),
    }
}

/// Whether an `error`-reported outcome still proves the attempt produced
/// work: the outcome's own text, truncation mark or nonzero usage
/// record, or in-turn output — every tap event on a lane with no
/// dispatch point, billable wire evidence on a probed one. A bare
/// turn-start notification or an all-zero usage record is bookkeeping,
/// not served work.
fn error_outcome_served(result: &TurnOutcome, emitted_output: bool, probe: &DispatchProbe) -> bool {
    (!result.text.is_empty() && !probe.retained_stream_text(&result.text))
        || result.truncated
        || result.usage.as_ref().is_some_and(billable_usage)
        || emitted_output
        || probe.wire_billable()
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
    // generic rpc lane's prompt admission and `cli/turn` the cli lane's
    // turn: the refusal arrived in place of a result, before the
    // provider could serve the turn.
    if error.code == "agent_quota_exhausted" {
        return matches!(
            error.stage,
            "session/open"
                | "session/prompt"
                | "session/set_model"
                | "session/set_config_option"
                | "rpc/call"
                | "cli/turn"
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
/// Whether the request resumes an existing session — the adapter's own
/// normalization: an empty or whitespace-only id opens a fresh session,
/// so the setup call is `session_open`, never `session_resume`.
fn resuming(request: &SessionRequest) -> bool {
    request
        .resume
        .as_deref()
        .is_some_and(|value| !value.trim().is_empty())
}

/// The setup calls this open actually issues, in issue order —
/// `initialize`, then the session call the request selects. A
/// process-scoped lane has no session calls at all; resuming selects
/// `session_resume` with `session_open` as fallback, and a fresh
/// session only ever calls `session_open`.
fn rpc_setup_calls(rpc: &a4aa::RpcLaneSpec, resuming: bool) -> Vec<&a4aa::RpcCallSpec> {
    let session_call = match rpc.session_scope {
        a4aa::SessionScope::Process => None,
        _ if resuming => rpc.session_resume.as_ref().or(rpc.session_open.as_ref()),
        _ => rpc.session_open.as_ref(),
    };
    [rpc.initialize.as_ref(), session_call]
        .into_iter()
        .flatten()
        .collect()
}

/// Whether one of the rpc lane's setup calls — `initialize`, or the
/// session call this request actually selects — consumes one of `names`.
/// A `session_resume`-only placeholder must not read as used by a fresh
/// open.
fn rpc_setup_uses(rpc: &a4aa::RpcLaneSpec, resuming: bool, names: &[&str]) -> bool {
    rpc_setup_calls(rpc, resuming).into_iter().any(|call| {
        names
            .iter()
            .any(|name| contains_placeholder(&call.params, name))
    })
}

/// The frame-root fields an inline envelope stamps over a merged object
/// params document — the method field, every const field, the request
/// discriminant, and `id`. A template leaf at one of those positions
/// never reaches the wire.
fn inline_overwritten(envelope: &a4aa::RpcEnvelopeSpec) -> std::collections::BTreeSet<String> {
    let mut overwritten: std::collections::BTreeSet<String> =
        envelope.const_fields.keys().cloned().collect();
    overwritten.insert(envelope.method_field.clone());
    overwritten.insert("id".to_owned());
    if envelope.request_marker.is_some() {
        if let Some(field) = &envelope.discriminant_field {
            overwritten.insert(field.clone());
        }
    }
    overwritten
}

/// Whether a setup call's model binding survives onto the wire — the
/// same effective-slot check the probe applies: under an inline
/// envelope a `{model}` leaf whose root position the envelope stamps
/// over is emitted nowhere, so the call cannot bind the model no
/// matter what its template declares.
fn rpc_call_binds_model(rpc: &a4aa::RpcLaneSpec, call: &a4aa::RpcCallSpec) -> bool {
    let slots = model_slots(&call.params);
    if slots.is_empty() {
        return false;
    }
    match rpc
        .envelope
        .as_ref()
        .filter(|envelope| envelope.params_inline)
    {
        Some(envelope) if call.params.is_object() => {
            let overwritten = inline_overwritten(envelope);
            slots
                .iter()
                .any(|(path, _)| !overwritten.contains(path[0].as_str()))
        }
        _ => true,
    }
}

/// Whether a setup call can put this model value on the wire —
/// `rpc_call_binds_model`'s surviving-slot read narrowed by the value's
/// own resolution: a `{modelProvider}`-only slot emits nothing for an
/// unqualified candidate, so the call cannot bind it however the
/// template declares.
fn rpc_call_carries_model(rpc: &a4aa::RpcLaneSpec, call: &a4aa::RpcCallSpec, model: &str) -> bool {
    let slots = model_slots(&call.params)
        .into_iter()
        .filter(|(_, leaf)| leaf_resolves_model(leaf, model))
        .collect::<Vec<_>>();
    if slots.is_empty() {
        return false;
    }
    match rpc
        .envelope
        .as_ref()
        .filter(|envelope| envelope.params_inline)
    {
        Some(envelope) if call.params.is_object() => {
            let overwritten = inline_overwritten(envelope);
            slots
                .iter()
                .any(|(path, _)| !overwritten.contains(path[0].as_str()))
        }
        _ => true,
    }
}

/// Session-phase selector support for the setup call this request
/// actually selects — the same classification the adapter's session
/// validation applies at open, so a projected value never reaches a
/// call that cannot consume it. The session model and effort still
/// count the lane's turn surface: a stored `config` value substitutes
/// into every later prompt template whether or not a setup call binds
/// it. Lanes without a per-call classification keep their declared
/// session support.
fn session_selectors(spec: &AgentSpec, request: &SessionRequest) -> SelectorSupportSet {
    let LaneSpec::Rpc(rpc) = &spec.lane else {
        return spec.selector_support(SelectorPhase::Session);
    };
    let resuming = resuming(request);
    let uses_turn = |names: &[&str]| {
        let uses = |call: &a4aa::RpcCallSpec| {
            names
                .iter()
                .any(|name| contains_placeholder(&call.params, name))
        };
        uses(&rpc.prompt) || rpc.pre_turn.iter().any(uses)
    };
    let support = |applied: bool| {
        if applied {
            SelectorSupport::Applied
        } else {
            SelectorSupport::Unsupported
        }
    };
    SelectorSupportSet {
        // Model support counts only slots that survive the lane's
        // encoding: a `{model}` an inline envelope stamps over never
        // reaches the wire, so it neither binds at open nor carries a
        // turn — the same effective reading the probe enforces.
        model: support(
            rpc_setup_calls(rpc, resuming)
                .into_iter()
                .any(|call| rpc_call_binds_model(rpc, call))
                || rpc_turn_binds_model(rpc),
        ),
        effort: support(rpc_setup_uses(rpc, resuming, &["effort"]) || uses_turn(&["effort"])),
        mode: support(rpc_setup_uses(rpc, resuming, &["mode"])),
        permission: support(rpc_setup_uses(rpc, resuming, &["permission"])),
    }
}

/// Whether the lane's session establishment consumes the model selector
/// on the wire — an actual open-time binding — as opposed to storing the
/// open request's model as the default every later turn inherits. Rpc
/// checks the setup calls the request actually selects. Acp negotiates
/// the model in its session handshake and serve applies it through the
/// server's model endpoint; cli substitutes it into every spawn's argv,
/// so its open model is always a stored default.
fn model_binds_at_open(spec: &AgentSpec, resuming: bool) -> bool {
    match &spec.lane {
        LaneSpec::Cli(_) => false,
        LaneSpec::Rpc(rpc) => rpc_setup_calls(rpc, resuming)
            .into_iter()
            .any(|call| rpc_call_binds_model(rpc, call)),
        _ => spec.selector_support(SelectorPhase::Session).model != SelectorSupport::Unsupported,
    }
}

/// Whether one of the lane's setup calls can put this request's model
/// on the wire — `model_binds_at_open`'s declared slot read resolved
/// against the actual candidate. Non-rpc lanes keep the declared
/// open-binding reading.
fn setup_carries_model(spec: &AgentSpec, request: &SessionRequest) -> bool {
    match &spec.lane {
        LaneSpec::Rpc(rpc) => request.config.model.as_deref().is_some_and(|model| {
            rpc_setup_calls(rpc, resuming(request))
                .into_iter()
                .any(|call| rpc_call_carries_model(rpc, call, model))
        }),
        _ => model_binds_at_open(spec, resuming(request)),
    }
}

/// Whether one of the setup calls this request's reopen issues can put
/// `model` on the wire. Declared support is not enough: a
/// `{modelProvider}`-only lane emits nothing for an unqualified
/// candidate however its slots read, so a session-bound retry for it
/// would only reopen the lane and repeat the identical refused request.
fn setup_carries_candidate(spec: &AgentSpec, model: &str, resuming: bool) -> bool {
    match &spec.lane {
        LaneSpec::Rpc(rpc) => rpc_setup_calls(rpc, resuming)
            .into_iter()
            .any(|call| rpc_call_carries_model(rpc, call, model)),
        _ => spec.selector_support(SelectorPhase::Session).model != SelectorSupport::Unsupported,
    }
}

/// Whether the rpc lane's turn surface can carry the model selector on
/// the wire — a `{model}`-family slot in the prompt or a pre-turn call
/// that survives the encoding, read the same effective way
/// `model_binds_at_open` reads the setup calls.
fn rpc_turn_binds_model(rpc: &a4aa::RpcLaneSpec) -> bool {
    rpc_call_binds_model(rpc, &rpc.prompt)
        || rpc
            .pre_turn
            .iter()
            .any(|call| rpc_call_binds_model(rpc, call))
}

/// Whether a per-turn model retry can actually reach the wire on this
/// lane — its declared per-turn support, reduced for rpc to the slots
/// the frame's encoding leaves standing. A `{model}` leaf the envelope
/// overwrites still reports `Applied` upstream, but sending it changes
/// nothing on the wire.
fn turn_binds_model(spec: &AgentSpec) -> bool {
    match &spec.lane {
        LaneSpec::Rpc(rpc) => rpc_turn_binds_model(rpc),
        _ => spec.selector_support(SelectorPhase::Turn).model != SelectorSupport::Unsupported,
    }
}

/// Whether this lane's per-turn surface can put `model` on the wire —
/// `turn_binds_model`'s effective-slot read narrowed by the value's own
/// resolution, the way `setup_carries_candidate` narrows
/// `model_binds_at_open`: a `{modelProvider}`-only prompt emits nothing
/// for an unqualified candidate, so scheduling a retry for it resubmits
/// the identical model-free request and the peer serves its own
/// default while the entries behind it were never tried.
fn turn_carries_candidate(spec: &AgentSpec, model: &str) -> bool {
    match &spec.lane {
        LaneSpec::Rpc(rpc) => {
            rpc_call_carries_model(rpc, &rpc.prompt, model)
                || rpc
                    .pre_turn
                    .iter()
                    .any(|call| rpc_call_carries_model(rpc, call, model))
        }
        _ => spec.selector_support(SelectorPhase::Turn).model != SelectorSupport::Unsupported,
    }
}

/// Whether the call an open-stage model refusal answers carried the
/// model. Rpc reports every setup failure as `rpc/call`, so the wire
/// journal names the refused call by its request id: only a refusal
/// from a model-carrying setup call — `session/open`, `session_resume`,
/// or a model-consuming `initialize` — is an answer about the model
/// worth a chain entry. A model-free call's refusal repeats identically
/// on every retry, so it is the lane's answer and the walk moves on.
/// Lanes without wire visibility keep the declared open-binding
/// reading.
fn refused_call_binds_model(spec: &AgentSpec, resuming: bool, probe: &DispatchProbe) -> bool {
    match &spec.lane {
        LaneSpec::Rpc(_) if probe.has_dispatch_point() => probe.refused_setup_binds_model(),
        _ => model_binds_at_open(spec, resuming),
    }
}

/// Whether a turn-stage model refusal actually answers the model. A
/// quota refusal reports `rpc/call` for every rpc request — the prompt
/// admission, but also the configured pre-turn calls that share the
/// stage. The wire journal names the refused call by request id: a
/// model-free preparation call's refusal repeats identically for every
/// chain entry, so it is the lane's answer and the walk moves on; the
/// prompt's refusal and every refusal the wire cannot attribute keep
/// the model reading. Lanes without wire visibility keep the declared
/// reading.
fn turn_refusal_binds_model(error: &a4aa::AdapterError, probe: &DispatchProbe) -> bool {
    if error.code == "agent_quota_exhausted"
        && error.stage == "rpc/call"
        && probe.has_dispatch_point()
    {
        probe.refused_turn_binds_model()
    } else {
        true
    }
}

/// The chain entry the peer could actually refuse: the candidate the
/// refused turn call carried on the wire, or — for a model-free call —
/// the session model a setup call bound. A refusal from a lane whose
/// wire never carried a model answers no chain entry: masked slots and
/// absent resolutions are scaffolding, not a model the peer saw.
fn refused_model(
    request: &TurnRequest,
    open: &SessionRequest,
    probe: &DispatchProbe,
) -> Option<String> {
    if probe.prompt_model_sent() || probe.turn_model_sent() {
        request.model.clone().or_else(|| open.config.model.clone())
    } else if probe.model_setup_sent() {
        open.config.model.clone()
    } else {
        None
    }
}

/// Whether a params template consumes one named `{placeholder}` anywhere
/// in its tree — the same check the lane's own selector discovery runs.
fn contains_placeholder(template: &serde_json::Value, name: &str) -> bool {
    match template {
        serde_json::Value::String(text) => text.contains(&format!("{{{name}}}")),
        serde_json::Value::Array(items) => {
            items.iter().any(|item| contains_placeholder(item, name))
        }
        serde_json::Value::Object(fields) => fields
            .values()
            .any(|value| contains_placeholder(value, name)),
        _ => false,
    }
}

fn lane_open(
    spec: &AgentSpec,
    request: &SessionRequest,
    profile: &Profile,
    head_model: Option<String>,
) -> (SessionRequest, Option<String>, Option<String>) {
    let session_support = session_selectors(spec, request);
    let mut turn_model_default = None;
    let mut turn_effort_default = None;
    let mut lane_request = request.clone();
    if lane_request.config.model.is_none() {
        if session_support.model != SelectorSupport::Unsupported {
            lane_request.config.model = head_model;
        } else {
            turn_model_default = head_model;
        }
    }
    if lane_request.config.mode.is_none() {
        lane_request.config.mode = profile.mode.clone();
    }
    if lane_request.config.effort.is_none() {
        if session_support.effort != SelectorSupport::Unsupported {
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
            // Whether a replacement open already put the pending
            // candidate on the wire. The dispatch probe is per-attempt, so
            // each abandoned attempt folds its record in here — a
            // candidate the peer may have received stays sent no matter
            // how fresh the next lane's probe is.
            let mut pending_sent = false;
            // Carry evidence from the lane actually serving this turn —
            // the send record belonging to the session that now holds
            // the work. `pending_sent` folds every lane the candidate
            // rode, including ones that already died: their sends make
            // the candidate spent for requeue, but only the serving
            // lane's own carry can publish it — a send on a dead lane
            // proves nothing about the session that answered.
            let mut serving_sent = false;
            // The chain candidate scheduled on this turn — on a per-turn
            // surface or carried across a lane switch — waiting for the
            // lane's acceptance evidence. It publishes as the session's
            // default and the chain's active entry only when the attempt
            // that carried it confirms the selector; a refused candidate
            // never publishes, and neither does one whose attempt died
            // before dispatch.
            let mut pending_model: Option<String> = None;
            // Chain candidates this lane's refusals never saw — held out
            // of `remaining` so the retry below cannot re-select the
            // identical unsent entry for another go on this same lane.
            // They return to the chain when the lane changes or the
            // turn ends.
            let mut held_unsent: Vec<String> = Vec::new();
            // Every chain entry a refusal answered this turn, in refusal
            // order. The retry arm spends the entry it advances past by
            // scheduling the next one, but the chain itself keeps no
            // record — so each refusal is journaled here and the
            // lane-fallback arm spends them all: a still-held refused
            // entry — the turn-local driver selection above all — would
            // otherwise ride the lane switch and re-open the alternate
            // on a model the peer already refused.
            let mut refused_models: Vec<String> = Vec::new();
            // This turn ends on a re-established session whose stored
            // default is cleared of the refused pending candidate —
            // nothing replays on it; the refusal still answers the turn.
            let mut retire_default = false;
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
            // The most recent completed attempt's result, carried across
            // lane and model retries: a cancellation observed while the
            // driver arranges the next attempt ends the turn with the
            // lane's real answer — never an empty synthesized
            // cancellation that erases it.
            let mut prior: Option<a4aa::Result<TurnOutcome>> = None;
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
                let (probe, relay, active_spec, installed_open, refused_default) = {
                    let lane = lane.lock().await;
                    (
                        lane.probe.clone(),
                        lane.relay.clone(),
                        lane.specs[lane.index].clone(),
                        lane.open.clone(),
                        lane.refused_default.clone(),
                    )
                };
                let (turn_selectors, session_model) = {
                    let guard = session.read().await;
                    (
                        guard.selector_support(SelectorPhase::Turn),
                        guard.selector_support(SelectorPhase::Session).model,
                    )
                };
                // Declared per-turn model support, reduced to the slots
                // the lane's encoding leaves standing: a `{model}` an
                // inline envelope stamps over never reaches the wire, so
                // it cannot carry a chain retry.
                let turn_model = if matches!(&active_spec.lane, LaneSpec::Rpc(_))
                    && !turn_binds_model(&active_spec)
                {
                    SelectorSupport::Unsupported
                } else {
                    turn_selectors.model
                };
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
                            // The chain's awaiting candidate keeps riding
                            // implicit prompts until the wire proves or
                            // refuses it — on a session that only expresses
                            // the model per turn this projection is its
                            // carry, on one that stores it the slot resolves
                            // the same value either way. The claim is a
                            // default: a direct prompt fills none.
                            record_instructions
                                .then(|| {
                                    model_state
                                        .lock()
                                        .unwrap_or_else(|error| error.into_inner())
                                        .pending
                                        .clone()
                                })
                                .flatten()
                        })
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
                // A stored default the provider refused — the lane keeps
                // it marked until a clean replacement session takes over —
                // must never ride another implicit prompt: this iteration
                // exists only to re-establish the session before anything
                // dispatches. An explicit turn model names its own value
                // and needs no repair.
                let poisoned = refused_default.is_some()
                    && installed_open.config.model == refused_default
                    && request
                        .model
                        .as_deref()
                        .is_none_or(|model| refused_default.as_deref() == Some(model));
                // Set when the next chain entry can only apply by
                // re-establishing this lane's session — a model bound at
                // session/open has no per-turn surface to carry it.
                let mut session_retry = poisoned;
                // The queued claim still awaits carry evidence on a lane
                // whose turn surface cannot deliver it: every implicit
                // prompt fills nothing for it and keeps answering on the
                // installed model, so the claim would queue forever.
                // Where a setup call on this lane can carry the claim,
                // re-establish the session bound to it before anything
                // dispatches — the same repair iteration the poisoned
                // default takes, and like it no completed attempt's
                // answer. A caller-owned value owns the prompt instead,
                // and a claim the turn slots resolve simply rides the
                // request.
                let mut repair = poisoned;
                if !repair && record_instructions && pending_bind.is_none() {
                    let claim = model_state
                        .lock()
                        .unwrap_or_else(|error| error.into_inner())
                        .pending
                        .clone();
                    if let Some(claim) = claim {
                        if request
                            .model
                            .as_deref()
                            .is_none_or(|model| model == claim.as_str())
                            && !turn_carries_candidate(&active_spec, &claim)
                            && setup_carries_candidate(
                                &active_spec,
                                &claim,
                                resuming(&installed_open),
                            )
                            // The installed session is already bound to
                            // this claim and its wire provably carried no
                            // model: the identical reopen would repeat an
                            // evidence-free open and the claim would
                            // re-enter repair without ever dispatching —
                            // it waits for a lane that can carry it.
                            && !(probe.has_dispatch_point()
                                && installed_open.config.model.as_deref()
                                    == Some(claim.as_str())
                                && !probe.model_setup_sent())
                        {
                            pending_bind = Some(claim);
                            pending_sent = false;
                            session_retry = true;
                            repair = true;
                        }
                    }
                }
                // The model loop on the current lane: provider-side model
                // failures advance the chain and resubmit on the same
                // session.
                let outcome = loop {
                    // A close or cancel that landed while the driver was
                    // arranging this attempt — including a fallback's
                    // reopen — must stop the turn before anything else
                    // dispatches. The poisoned iteration dispatches
                    // nothing, so lifecycle answers first: a cancel before
                    // the repair open is a cancelled turn, never a model
                    // refusal.
                    if closing.load(Ordering::Acquire) {
                        break 'lanes Err(a4aa::AdapterError::permanent(
                            "agent_session_closed",
                            "session/prompt",
                        ));
                    }
                    if cancellation.load(Ordering::Acquire) != cancel_mark {
                        // A scheduled candidate goes back to the chain
                        // only while no wire ever carried it — the same
                        // restoration rule the replacement-open guards
                        // apply: folded send history plus the installed
                        // lane's own probe record. A candidate an earlier
                        // lane's setup already sent stays spent — the
                        // peer may have bound it.
                        let pending_candidate =
                            pending_model.clone().or_else(|| driver_model.clone());
                        if !(pending_sent
                            || probe.model_setup_sent()
                                && installed_open.config.model == pending_candidate)
                        {
                            restore_candidate(
                                &model_state,
                                pending_model.take().or_else(|| driver_model.take()),
                            );
                        }
                        break 'lanes prior.take().unwrap_or_else(|| Ok(cancelled_outcome()));
                    }
                    // The repair iteration synthesized a model refusal so
                    // the reopen below knows to re-establish the session:
                    // it is not a completed attempt's answer.
                    if repair {
                        break Err(a4aa::AdapterError::permanent(
                            "agent_model_refused",
                            "session/open",
                        ));
                    }
                    // One iteration is one physical prompt attempt: reset
                    // the wire journal so an earlier attempt's dispatch
                    // record cannot read this attempt's pre-dispatch
                    // failure as dispatched.
                    probe.reset(
                        &request.text,
                        request.model.as_deref(),
                        request.effort.as_deref(),
                    );
                    let outcome = {
                        let guard = session.read().await;
                        guard.prompt(request.clone()).await
                    };
                    // Fold this attempt's send evidence into the pending
                    // candidate's record before the guards below read it:
                    // a prompt or pre-turn frame that carried the
                    // request's model reached the wire — the same "sent
                    // stays spent" rule the replacement opens apply. The
                    // session's stored model rides the frame the same way
                    // when the request omits the selector: an open that
                    // left its candidate unconfirmed keeps it pending,
                    // and a frame whose resolved model is that candidate
                    // is its carry evidence — the same serving-lane rule.
                    let sent = probe.prompt_model_sent() || probe.turn_model_sent();
                    if request.model.is_some() && request.model == driver_model {
                        pending_sent |= sent;
                        serving_sent |= sent;
                    }
                    {
                        let chain = model_state
                            .lock()
                            .unwrap_or_else(|error| error.into_inner());
                        let wire_model = request
                            .model
                            .clone()
                            .or_else(|| installed_open.config.model.clone());
                        if sent && chain.pending.is_some() && chain.pending == wire_model {
                            pending_sent = true;
                            serving_sent = true;
                            if pending_model.is_none() {
                                // The shared claim stays in place until a
                                // refusal spends it or a confirmation
                                // publishes it: a `Sent` record is intent,
                                // so the write may yet be confirmed
                                // failed — taking the claim here would
                                // drop a candidate the wire never carried.
                                pending_model = chain.pending.clone();
                            }
                        }
                    }
                    // A cancel that landed while the prompt sat in
                    // admission ends the turn: the lane was idle when
                    // cancel ran, so its own cancel path saw nothing —
                    // the admission refusal is the cancelled verdict.
                    // Once the attempt was in flight the lane's result
                    // stands: a genuinely cancelled turn keeps its
                    // partial text, usage and truncation, and a lane that
                    // completed the prompt normally — a cancel the lane
                    // does not support — is not rewritten as cancelled.
                    if cancellation.load(Ordering::Acquire) != cancel_mark
                        && matches!(&outcome, Err(error) if error.code == "agent_turn_cancelled")
                    {
                        // The candidate whose attempt ended at admission
                        // never dispatched — same restoration rule: it
                        // returns to the chain only while no wire ever
                        // carried it, and the installed probe's setup
                        // send counts only when it carried *this*
                        // candidate — its flag alone can describe the
                        // open's other model.
                        let pending_candidate =
                            pending_model.clone().or_else(|| driver_model.clone());
                        if !(pending_sent
                            || probe.model_setup_sent()
                                && installed_open.config.model == pending_candidate)
                        {
                            restore_candidate(
                                &model_state,
                                pending_model.take().or_else(|| driver_model.take()),
                            );
                        }
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
                    // The serving lane must also have carried the
                    // candidate: a prompt whose only model slot resolved
                    // absent — an unqualified `{modelProvider}` — reaches
                    // the peer with the selector omitted, and whatever
                    // the peer then serves is its own default, never
                    // proof it bound the candidate. The evidence is the
                    // serving lane's own send record — a send on a lane
                    // that already died belongs to `pending_sent`'s
                    // requeue history, not to this session.
                    let carried = !probe.has_dispatch_point() || serving_sent;
                    if let Some(accepted) = pending_model.take() {
                        let confirmed = carried
                            && match &outcome {
                                // A served turn confirms the candidate; an
                                // `error` outcome confirms it only with
                                // positive work evidence — text, truncation,
                                // usage or output the attempt produced — the
                                // same reading the lane-fallback decision
                                // makes of an unserved turn.
                                Ok(result) => {
                                    !matches!(
                                        result.stop_reason.as_str(),
                                        "cancelled" | "interrupted"
                                    ) && (result.stop_reason != "error"
                                        || error_outcome_served(result, relay.output(), &probe))
                                }
                                Err(error) => {
                                    // Dispatch evidence publishes only a
                                    // candidate the wire could have carried —
                                    // send tracking already folded this
                                    // attempt's prompt and pre-turn record.
                                    !model_failure(error)
                                        && (billed
                                            || serving_sent && probe.dispatch_confirmed(error))
                                }
                            };
                        if confirmed {
                            *turn_model_default
                                .lock()
                                .unwrap_or_else(|error| error.into_inner()) =
                                Some(accepted.clone());
                            let mut chain = model_state
                                .lock()
                                .unwrap_or_else(|error| error.into_inner());
                            // The claim is a default fill: when the
                            // confirmed entry is not the claim itself —
                            // a candidate a direct retry carried and the
                            // queued default it never used — the claim
                            // stays parked for the next implicit prompt.
                            let claim_confirmed =
                                chain.pending.as_deref() == Some(accepted.as_str());
                            chain.active = Some(accepted);
                            if record_instructions || claim_confirmed {
                                chain.pending = None;
                            }
                        } else if serving_sent
                            && matches!(&outcome, Err(error)
                                if model_failure(error)
                                    && turn_refusal_binds_model(error, &probe))
                        {
                            // A pending candidate the lane provably
                            // refused is spent: the shared chain must not
                            // keep it as the awaiting entry for a later
                            // lane walk to pick again. The carry evidence
                            // is this lane's own send record — a refusal
                            // on a model-free prompt never used the
                            // candidate, so the claim survives.
                            {
                                let mut chain = model_state
                                    .lock()
                                    .unwrap_or_else(|error| error.into_inner());
                                if chain.pending.as_deref() == Some(accepted.as_str()) {
                                    chain.pending = None;
                                }
                            }
                            // The refused entry installed as this
                            // session's stored default — an open that
                            // never carried it — poisons every later
                            // implicit prompt: mark the lane's slot until
                            // a clean replacement session takes over. A
                            // model the open itself bound is a real
                            // session binding, not a poisoned default.
                            if installed_open.config.model.as_deref() == Some(accepted.as_str())
                                && !probe.model_setup_sent()
                            {
                                lane.lock().await.refused_default = Some(accepted);
                            }
                        } else if !pending_sent
                            && matches!(&outcome, Err(error) if model_failure(error))
                        {
                            // A model refusal that never saw the candidate
                            // spends nothing — the unsent entry exists only
                            // in turn-local state and ends with the turn
                            // otherwise. The awaiting claim keeps it riding
                            // implicit prompts for a lane that can send it;
                            // it must not re-enter `remaining`, where the
                            // retry below would immediately re-select the
                            // identical unsent entry for the same lane —
                            // it holds outside the retry queue until the
                            // lane changes or the turn ends. Cancellations
                            // and transport losses restore through
                            // `restore_candidate` — only a refusal has no
                            // other path back.
                            let mut chain = model_state
                                .lock()
                                .unwrap_or_else(|error| error.into_inner());
                            if chain.active.as_deref() != Some(accepted.as_str())
                                && chain.pending.as_deref() != Some(accepted.as_str())
                                && !chain.remaining.iter().any(|entry| entry == &accepted)
                                && !held_unsent.iter().any(|entry| entry == &accepted)
                            {
                                if chain.pending.is_none() {
                                    chain.pending = Some(accepted);
                                } else {
                                    held_unsent.push(accepted);
                                }
                            }
                        }
                    }
                    // A cancelled outcome can arrive after a model-free
                    // pre-turn call completes and before the candidate
                    // reaches the prompt — the take above already dropped
                    // it. Restore by the candidate's own folded send
                    // record: the installed probe's setup flag alone can
                    // describe the open's other model.
                    if !pending_sent
                        && matches!(&outcome, Ok(result) if matches!(
                            result.stop_reason.as_str(),
                            "cancelled" | "interrupted"
                        ))
                    {
                        restore_candidate(
                            &model_state,
                            pending_model.take().or_else(|| driver_model.take()),
                        );
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
                        // must not retry a possibly paid turn. And the
                        // refusal must answer the model at all: a
                        // model-free pre-turn call shares the `rpc/call`
                        // stage, and its refusal is the lane's answer,
                        // not a chain entry's.
                        Err(error)
                            if model_failure(error)
                                && !billed
                                && turn_refusal_binds_model(error, &probe) =>
                        {
                            // Journal the refused entry before advancing:
                            // the lane-fallback arm spends this turn's
                            // whole refusal record, not only the last
                            // attempt's.
                            if let Some(refused) = refused_model(&request, &installed_open, &probe)
                            {
                                if !refused_models.iter().any(|entry| entry == &refused) {
                                    refused_models.push(refused);
                                }
                            }
                            let mut chain = model_state
                                .lock()
                                .unwrap_or_else(|error| error.into_inner());
                            // Prefer the first entry this lane can
                            // actually deliver on whichever surface
                            // carries it — resolving an effective turn
                            // slot where the lane has one, or bound by a
                            // setup call where the retry re-opens the
                            // session. An entry it can deliver nowhere —
                            // an unqualified candidate on a
                            // provider-only lane — would resubmit the
                            // identical model-free request and serve the
                            // peer's own default while deliverable
                            // entries behind it were never tried; hold
                            // it out of this lane's retry queue and walk
                            // on. When nothing deliverable stands, the
                            // head stays the pick on a lane whose turn
                            // surface carries models — the last-resort
                            // retry serves the same default it always
                            // has. A lane with no turn surface at all
                            // has no such retry: the identical reopen
                            // would only resend what the refusal already
                            // answered.
                            match chain.remaining.iter().position(|candidate| {
                                (turn_model != SelectorSupport::Unsupported
                                    && turn_carries_candidate(&active_spec, candidate))
                                    || setup_carries_candidate(
                                        &active_spec,
                                        candidate,
                                        resuming(&installed_open),
                                    )
                            }) {
                                Some(index) => {
                                    for candidate in chain.remaining.drain(..index) {
                                        if !held_unsent.iter().any(|entry| entry == &candidate) {
                                            held_unsent.push(candidate);
                                        }
                                    }
                                    chain.remaining.first().cloned()
                                }
                                None if turn_model != SelectorSupport::Unsupported => {
                                    chain.remaining.first().cloned()
                                }
                                None => None,
                            }
                        }
                        _ => None,
                    };
                    let Some(next) = next else { break outcome };
                    // A cancellation that landed while the prompt was in
                    // flight ends the turn with the lane's real answer —
                    // before the next chain entry is spent on a retry
                    // that can never dispatch.
                    if cancellation.load(Ordering::Acquire) != cancel_mark {
                        break 'lanes outcome;
                    }
                    // Decide which surface carries the next entry before
                    // spending it: a per-turn selector takes it on the
                    // request where the lane's turn slots resolve this
                    // candidate, a session-bound selector re-establishes
                    // the session for a candidate only a setup call
                    // carries — the prompt cannot express it, and
                    // resending the identical model-free request would
                    // just earn the same refusal — and a lane with no
                    // model surface keeps the refused entry active. The
                    // entry leaves `remaining` once scheduled — it is
                    // never retried — but the chain's active entry and
                    // the session's default publish only when the lane
                    // accepts it.
                    let session_bound = if turn_model != SelectorSupport::Unsupported
                        && turn_carries_candidate(&active_spec, &next)
                    {
                        false
                    } else {
                        // The retry re-establishes this lane's session to
                        // bind the candidate — meaningful only when the
                        // setup call this request actually selects carries
                        // it on the wire. The session's aggregate rpc
                        // support counts `session_resume` even for a
                        // fresh session that never calls it, which would
                        // reopen the identical session once per chain
                        // entry and replay the identical prompt without a
                        // model. An entry no surface carries — the
                        // last-resort head pick — keeps the identical
                        // per-turn resend on a lane whose turn surface
                        // exists, and ends the turn where none does.
                        let binds = match &active_spec.lane {
                            LaneSpec::Rpc(_) => setup_carries_candidate(
                                &active_spec,
                                &next,
                                resuming(&installed_open),
                            ),
                            _ => session_model != SelectorSupport::Unsupported,
                        };
                        if binds {
                            true
                        } else if turn_model != SelectorSupport::Unsupported {
                            false
                        } else {
                            break outcome;
                        }
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
                        pending_sent = false;
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
                    // The send history belongs to the candidate: a fresh
                    // chain entry leaves `remaining` only while no wire
                    // carried it, so its history starts empty.
                    pending_sent = false;
                    serving_sent = false;
                    prior = Some(outcome);
                };
                relay.flush().await;
                let (emitted_output, emitted_billable) = relay.disarm();
                // A refused pending candidate installed as the session's
                // stored default is spent — but the session would resend
                // it on every later implicit prompt. The retire is only
                // for a stored default the open never carried — marked
                // on the lane when the refusal was attributed, so a
                // later candidate's refusal cannot mask it: a model the
                // open itself bound is a real session binding and keeps
                // the "last accepted" reading even when a prompt refuses
                // it. Re-establishing this lane's session with the
                // chain's last accepted entry retires the poisoned
                // default; the refusal still answers this turn, nothing
                // replays on the replacement.
                let refused_default = lane.lock().await.refused_default.clone();
                if !session_retry
                    && matches!(&outcome, Err(error) if model_failure(error)
                        && turn_refusal_binds_model(error, &probe))
                    && refused_default.is_some()
                    && installed_open.config.model == refused_default
                    && !probe.model_setup_sent()
                {
                    session_retry = true;
                    retire_default = true;
                    // The refused entry is spent everywhere: a scheduled
                    // copy rides the same reopen's projection as the
                    // candidate, and the shared `pending` claim was
                    // already dropped when the refusal was attributed —
                    // the re-established session binds the last accepted
                    // one.
                    driver_model = None;
                }
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
                        // A refusal the wire attributes to a model-free
                        // pre-turn call is no model answer at all: it is
                        // the lane's, and the lane-fallback rule below
                        // applies. The chain is only exhausted once every
                        // entry was tried: candidates this lane could
                        // deliver nowhere — still queued behind its pick,
                        // held out of its retry walk, or parked on the
                        // shared claim — were never refused, so a later
                        // lane's slots may still carry them and the refusal
                        // stands only as this lane's answer. Billable
                        // evidence stays decisive: a possibly paid turn
                        // never replays.
                        Err(error)
                            if model_failure(error) && turn_refusal_binds_model(error, &probe) =>
                        {
                            // The entry the peer could actually refuse
                            // joins this turn's refusal record — the
                            // earlier attempts' refusals journaled in the
                            // retry arm are spent alongside it.
                            if let Some(refused) = refused_model(&request, &installed_open, &probe)
                            {
                                if !refused_models.iter().any(|entry| entry == &refused) {
                                    refused_models.push(refused);
                                }
                            }
                            // The chain is only exhausted once every
                            // entry was tried: candidates this lane could
                            // deliver nowhere — still queued behind its
                            // pick, held out of its retry walk, or parked
                            // on the shared claim — were never refused,
                            // so a later lane's slots may still carry
                            // them and the refusal stands only as this
                            // lane's answer. An entry a refusal spent —
                            // wherever the chain still holds it — is not
                            // untried. Billable evidence stays decisive:
                            // a possibly paid turn never replays.
                            let refused_entry = |held: &Option<String>| {
                                held.as_ref().is_some_and(|entry| {
                                    refused_models.iter().any(|refused| refused == entry)
                                })
                            };
                            let untried = !held_unsent.is_empty() || {
                                let chain = model_state
                                    .lock()
                                    .unwrap_or_else(|error| error.into_inner());
                                !chain.remaining.is_empty()
                                    || (chain.active.is_some() && !refused_entry(&chain.active))
                                    || (record_instructions
                                        && chain.pending.is_some()
                                        && !refused_entry(&chain.pending))
                            };
                            if emitted_billable || probe.wire_billable() || !untried {
                                break 'lanes outcome;
                            }
                            // Every refused entry is spent wherever the
                            // chain holds it — the same rule a refused
                            // open applies. Left `active`, the walk's
                            // untried pick could never run and the next
                            // lane would re-open on a model the peer just
                            // refused; left pending, the claim would keep
                            // a refused entry queued.
                            {
                                let mut chain = model_state
                                    .lock()
                                    .unwrap_or_else(|error| error.into_inner());
                                if refused_entry(&chain.active) {
                                    chain.active = None;
                                }
                                if refused_entry(&chain.pending) {
                                    chain.pending = None;
                                }
                            }
                            // The stored turn default is spent with them:
                            // left holding a refused entry, the next
                            // implicit prompt would resend it.
                            {
                                let mut default = turn_model_default
                                    .lock()
                                    .unwrap_or_else(|error| error.into_inner());
                                if refused_entry(&default) {
                                    *default = None;
                                }
                            }
                            // The turn-local driver selection is spent
                            // too: a scheduled candidate the peer refused
                            // must not ride the lane switch into the
                            // alternate's projection ahead of the untried
                            // chain.
                            for slot in [&mut pending_bind, &mut driver_model, &mut pending_model] {
                                if refused_entry(slot) {
                                    *slot = None;
                                }
                            }
                            true
                        }
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
                        // earlier stream — or usage proves the paid turn ran;
                        // bookkeeping-only wire activity does not.
                        Ok(result) => {
                            result.stop_reason == "error"
                                && !error_outcome_served(result, emitted_output, &probe)
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
                // The real answer a repair reopen produced: the repair
                // iteration's own `outcome` is the synthesized model
                // refusal, so a search that runs out of lanes must return
                // the failure the peer actually reported — a lost
                // transport, a protocol fault — never the marker that
                // entered repair.
                let mut reopen_error: Option<a4aa::AdapterError> = None;
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
                        // Candidates a refusal never saw on the failed
                        // lane waited outside the retry queue so it could
                        // not re-select an identical unsent request — the
                        // search has moved on now: this lane's slots may
                        // carry what that lane could not, so they
                        // re-enter the chain before its first setup
                        // attempt, where a refused open's chain walk can
                        // reach them.
                        for held in held_unsent.drain(..).rev() {
                            restore_candidate(&model_state, Some(held));
                        }
                    }
                    let Some(spec) = lane.specs.get(candidate).cloned() else {
                        // The search is over with nothing served — a
                        // scheduled candidate no wire ever carried goes
                        // back to the chain for the next turn, the same
                        // restoration rule the abandoned-open guards
                        // apply.
                        if !pending_sent {
                            restore_candidate(
                                &model_state,
                                pending_bind.take().or_else(|| driver_model.take()),
                            );
                        }
                        break 'lanes match reopen_error {
                            Some(error) if repair => Err(error),
                            _ => outcome,
                        };
                    };
                    // A pick scheduled against an earlier lane's slots
                    // rides the walk only where this lane can carry it:
                    // opening here unable to deliver it would sit the
                    // peer on its own default before the entries this
                    // lane can serve were ever tried. An unsent pick this
                    // lane can deliver nowhere returns to the chain
                    // before this lane's selection runs; one the wire
                    // already carried stays spent.
                    if !pending_sent {
                        for slot in [&mut pending_bind, &mut driver_model] {
                            let undeliverable = slot.as_deref().is_some_and(|candidate| {
                                !turn_carries_candidate(&spec, candidate)
                                    && !setup_carries_candidate(
                                        &spec,
                                        candidate,
                                        resuming(&lane.requested),
                                    )
                            });
                            if undeliverable {
                                let pick = slot.take();
                                if pending_model == pick {
                                    pending_model = None;
                                }
                                restore_candidate(&model_state, pick);
                            }
                        }
                    }
                    // Nothing scheduled rides this reopen and no accepted
                    // entry stands — the untried chain is what the walk
                    // has left. Move the first entry this lane's own slots
                    // can deliver into the scheduled candidate state
                    // before opening: the peer's refusal then spends it
                    // instead of leaving it queued for the next retry to
                    // re-pick, and the restore paths return it only while
                    // no wire carried it. An explicit turn model is the
                    // caller's own: the chain advances only on a verified
                    // model refusal, never on a lane's transport failure.
                    if caller_model.is_none() && pending_bind.is_none() && driver_model.is_none() {
                        let mut chain = model_state
                            .lock()
                            .unwrap_or_else(|error| error.into_inner());
                        if chain.active.is_none()
                            && !(record_instructions && chain.pending.is_some())
                        {
                            if let Some(index) = chain.remaining.iter().position(|candidate| {
                                turn_carries_candidate(&spec, candidate)
                                    || setup_carries_candidate(
                                        &spec,
                                        candidate,
                                        resuming(&lane.requested),
                                    )
                            }) {
                                pending_bind = Some(chain.remaining.remove(index));
                                pending_sent = false;
                            }
                        }
                    }
                    // The model this reopen binds: a session-bound retry
                    // pursues its scheduled candidate, a per-turn retry's
                    // candidate is the chain position this turn committed
                    // (the provider already refused the entry before it),
                    // and any other reopen carries the last accepted
                    // chain entry forward.
                    let pending_candidate = pending_bind
                        .clone()
                        .or_else(|| driver_model.clone())
                        .or_else(|| {
                            // A candidate the installed open accepted
                            // without carry evidence rides the switch
                            // too — the new lane's binding is its next
                            // chance to prove itself. The claim is a
                            // default: a direct prompt fills none, and a
                            // caller-modelled prompt names its own value —
                            // the chain may not promote its entries ahead
                            // of an explicit model that was never refused,
                            // so it stays queued for the next implicit
                            // prompt either way.
                            (record_instructions && caller_model.is_none())
                                .then(|| {
                                    model_state
                                        .lock()
                                        .unwrap_or_else(|error| error.into_inner())
                                        .pending
                                        .clone()
                                })
                                .flatten()
                        });
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
                    // Where the open request's model is only a stored
                    // default the lane re-applies to every turn, a pending
                    // candidate does not belong in it: it rides the
                    // replayed request instead of becoming the replacement
                    // session's persistent model — a refusal must not
                    // leave the refused model installed as the session's
                    // default, ready to send again on the next implicit
                    // prompt. A stored-default slot keeps the last
                    // accepted chain entry. A lane whose establishment
                    // call binds the model on the wire is an actual
                    // open-time selector — the candidate's open binding
                    // is the selector doing its job, and restoring the
                    // exhausted active entry there would abort the
                    // switch on a refusal before the candidate could
                    // reach the prompt. A declared slot that cannot
                    // resolve the candidate — an unqualified
                    // `{modelProvider}` — binds nothing either: left
                    // stored, every later prompt's `{model}` substitution
                    // would inherit the unconfirmed claim, and filling
                    // it with the exhausted entry would resolve the
                    // slot the candidate could not and resend a model
                    // the chain already spent.
                    if let Some(candidate) = &pending_candidate {
                        if new_open.config.model.as_deref() == Some(candidate.as_str())
                            && !setup_carries_model(&spec, &new_open)
                        {
                            // The open's declared slots exist but this
                            // candidate resolves none of them — the
                            // slot stays empty rather than carrying the
                            // exhausted entry onto the wire. Where the
                            // lane's establishment binds nothing at all
                            // the slot is only a stored default, and it
                            // keeps the last accepted entry.
                            new_open.config.model =
                                if model_binds_at_open(&spec, resuming(&new_open)) {
                                    None
                                } else {
                                    model_state
                                        .lock()
                                        .unwrap_or_else(|error| error.into_inner())
                                        .active
                                        .clone()
                                };
                            // The candidate rides the replay only where
                            // a turn slot can actually resolve it — an
                            // unqualified `{modelProvider}` entry emits
                            // nothing there either.
                            if turn_carries_candidate(&spec, candidate) {
                                turn_model = Some(candidate.clone());
                            }
                        }
                    }
                    let (adapter, probe) = lane.adapter(candidate, &new_open);
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
                        // The repair iteration's synthesized refusal is
                        // no completed attempt's answer: the close owns
                        // the verdict when nothing dispatched.
                        break 'lanes if repair {
                            Err(a4aa::AdapterError::permanent(
                                "agent_session_closed",
                                "session/prompt",
                            ))
                        } else {
                            outcome
                        };
                    }
                    // The cancellation mark moved before this candidate's
                    // open even began — the completed attempt's result is
                    // the turn's real answer; replaying nothing is the
                    // point, not rewriting that result as cancelled. A
                    // session-bound candidate whose reopen never ran
                    // returns to the chain for the next turn.
                    if cancellation.load(Ordering::Acquire) != cancel_mark {
                        // The candidate goes back to the chain only while
                        // no wire ever carried it — evidence folded in
                        // from every attempt it already rode, not just
                        // the probe this open happens to use, and on this
                        // probe only a setup send of *this* candidate
                        // counts. Either selector path may hold it: the
                        // session-bound `pending_bind`, or the per-turn
                        // `driver_model` a lane switch carried.
                        if !(pending_sent
                            || probe.model_setup_sent()
                                && new_open.config.model == pending_candidate)
                        {
                            restore_candidate(
                                &model_state,
                                pending_bind.take().or_else(|| driver_model.take()),
                            );
                        }
                        // The repair iteration's synthesized refusal is
                        // no completed attempt's answer: cancellation owns
                        // the verdict when nothing dispatched.
                        break 'lanes if repair {
                            prior.take().unwrap_or_else(|| Ok(cancelled_outcome()))
                        } else {
                            outcome
                        };
                    }
                    // The replacement session opens while the installed
                    // one stays in place: a refused open leaves the live
                    // session, its probe and the accepted model binding
                    // untouched — all three commit together only when the
                    // replacement actually opens.
                    let result = tokio::select! {
                        // A completed open wins over a coincident
                        // lifecycle signal: the installation guards below
                        // make the same ordered decision under the write
                        // lock — preferring the resolved arm keeps the
                        // outcome deterministic instead of racing which
                        // branch polled first.
                        biased;
                        result = adapter.open_session(new_open.clone(), tap_tx) => result,
                        _ = notified.as_mut() => {
                            if closing.load(Ordering::Acquire) {
                                break 'lanes if repair {
                                    Err(a4aa::AdapterError::permanent(
                                        "agent_session_closed",
                                        "session/prompt",
                                    ))
                                } else {
                                    outcome
                                };
                            }
                            // The abandoned open never completed — a
                            // pending candidate goes back to the chain
                            // only when no wire ever carried it: neither
                            // an earlier attempt's folded record nor this
                            // open's setup send of *this* candidate can
                            // show its model on the wire. One already
                            // sent may be bound on the peer, and
                            // restoring it would resend the same request
                            // next turn. Either selector path may hold
                            // the unsent candidate.
                            if !(pending_sent
                                || probe.model_setup_sent()
                                    && new_open.config.model == pending_candidate)
                            {
                                restore_candidate(
                                    &model_state,
                                    pending_bind.take().or_else(|| driver_model.take()),
                                );
                            }
                            break 'lanes Ok(cancelled_outcome());
                        }
                    };
                    // Fold this attempt's dispatch record into the
                    // candidate's: the probe is per-attempt, but whether
                    // a wire ever carried the pending model follows the
                    // candidate across every replacement it rode. Only a
                    // setup send of *this* candidate counts — an open
                    // that carried the still-active model says nothing
                    // about a candidate that rode the request instead.
                    let setup_carried =
                        probe.model_setup_sent() && new_open.config.model == pending_candidate;
                    pending_sent |= setup_carried;
                    // The serving record is this candidate lane's own —
                    // assigned, not folded: an open that failed leaves a
                    // dead lane's send in `pending_sent` for the requeue
                    // question while the next lane's open restarts the
                    // carry evidence its publication must show.
                    serving_sent = setup_carried;
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
                                    // Discarding the opened session is the
                                    // same peer-independent teardown the
                                    // retired lane gets below — its Drop
                                    // kills the process tree. Awaiting its
                                    // `session/close` would pin the turn
                                    // driver and the caller's event stream
                                    // on a peer that may never answer.
                                    drop(new_session);
                                    break 'lanes if repair {
                                        Err(a4aa::AdapterError::permanent(
                                            "agent_session_closed",
                                            "session/prompt",
                                        ))
                                    } else {
                                        outcome
                                    };
                                }
                                if cancellation.load(Ordering::Acquire) != cancel_mark {
                                    drop(guard);
                                    drop(new_session);
                                    // The opened replacement never
                                    // installed — a pending candidate no
                                    // wire ever carried returns to the
                                    // chain; one already sent may be
                                    // bound on the dropped peer.
                                    if !(pending_sent
                                        || probe.model_setup_sent()
                                            && new_open.config.model == pending_candidate)
                                    {
                                        restore_candidate(
                                            &model_state,
                                            pending_bind.take().or_else(|| driver_model.take()),
                                        );
                                    }
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
                                // The retired lane's teardown cannot wait
                                // on its peer: a session/close the peer
                                // never acknowledges would pin the lane's
                                // process and the sender feeding its tap —
                                // the relay draining that tap would hold
                                // the caller's event stream open forever.
                                // Dropping the session is the explicit
                                // teardown: the lane's own Drop kills the
                                // process tree, the transport's EOF closes
                                // its tap, and the dead lane's relay exits
                                // once its queued leftovers settle.
                                drop(retiring);
                            }
                            lane.index = candidate;
                            lane.probe = probe.clone();
                            lane.open = new_open;
                            // The replacement's stored slot is clean: the
                            // retired session carried the refused default
                            // out of service with it.
                            lane.refused_default = None;
                            // The dead lane's relay keeps draining its
                            // tap to the caller on its own — leftovers
                            // are the dead attempt's evidence, never
                            // counted against the reopened lane.
                            lane.relay =
                                EventRelay::spawn(next_tap_rx, lane.events.clone(), probe.clone());
                            // The same candidate the projection routed:
                            // a pending entry the installed open
                            // accepted without carry evidence rides this
                            // switch exactly like a scheduled one — it
                            // publishes only once the serving lane's own
                            // wire carried it.
                            match pending_bind.take().or_else(|| pending_candidate.clone()) {
                                // A candidate the lane switch carried and
                                // bound into the open request is already
                                // the replacement's session model — the
                                // installed lane holds it whether or not
                                // the replay ever confirms it. Publish it
                                // now so a refused or cancelled replay
                                // cannot leave `active_model` disagreeing
                                // with the binding the session keeps. On
                                // a probed lane the wire must have
                                // actually carried it: a slot the
                                // envelope stamped over proves nothing.
                                // It is no longer pending, so the shared
                                // re-projection below replaces the stale
                                // per-turn default: the next implicit
                                // prompt must not send the replaced model
                                // while `active_model` reports the new
                                // binding.
                                Some(carried_model)
                                    if lane.open.config.model.as_deref()
                                        == Some(carried_model.as_str())
                                        && (!probe.has_dispatch_point()
                                            || probe.model_setup_sent()) =>
                                {
                                    let mut chain = model_state
                                        .lock()
                                        .unwrap_or_else(|error| error.into_inner());
                                    chain.active = Some(carried_model.clone());
                                    // The claim is a default fill: a
                                    // direct prompt never consumed it,
                                    // so it stays queued for the next
                                    // implicit prompt.
                                    if record_instructions {
                                        chain.pending = None;
                                    }
                                    drop(chain);
                                    driver_model = Some(carried_model.clone());
                                    pending_model = Some(carried_model);
                                }
                                // A candidate deliverable on the lane's
                                // per-turn surface stays pending until the
                                // replay proves the lane took it — it rides
                                // the replayed request. A lane whose turn
                                // slots cannot resolve it never carries
                                // the candidate; its replay proves nothing
                                // about it, so it must not publish.
                                Some(carried_model)
                                    if turn_carries_candidate(&spec, &carried_model) =>
                                {
                                    driver_model = Some(carried_model.clone());
                                    pending_model = Some(carried_model);
                                    carried = true;
                                }
                                // No scheduled candidate: the model the
                                // replacement carries is the last
                                // accepted chain entry — publish it. A
                                // direct prompt never consumed the
                                // queued claim, so it stays pending for
                                // the next implicit prompt.
                                None => {
                                    let mut chain = model_state
                                        .lock()
                                        .unwrap_or_else(|error| error.into_inner());
                                    chain.active = accepted;
                                    if record_instructions {
                                        chain.pending = None;
                                    }
                                }
                                // A candidate this lane cannot deliver —
                                // its binding never reached the wire and
                                // the per-turn surface cannot carry it —
                                // goes back to the chain rather than
                                // publishing a model the peer never saw.
                                // Only an unsent candidate returns: one
                                // any wire already carried stays spent.
                                // A restored entry stops riding the
                                // driver's own carry — it waits in the
                                // chain instead of also replaying on the
                                // next lane.
                                Some(carried_model) => {
                                    if !pending_sent {
                                        restore_candidate(&model_state, Some(carried_model));
                                        driver_model = None;
                                    }
                                }
                            }
                        }
                        Err(open_error) => {
                            // A close landing while the replacement open
                            // sat in flight owns the verdict — the biased
                            // select can take this arm with the lifecycle
                            // signal already raised, and a failed open is
                            // no completed attempt's answer — least of
                            // all the repair iteration's synthesized
                            // refusal.
                            if closing.load(Ordering::Acquire) {
                                break 'lanes if repair {
                                    Err(a4aa::AdapterError::permanent(
                                        "agent_session_closed",
                                        "session/prompt",
                                    ))
                                } else {
                                    outcome
                                };
                            }
                            // The repair iteration's `outcome` is
                            // synthesized — keep the lane's real answer
                            // so an exhausted search returns it.
                            if repair {
                                reopen_error = Some(open_error.clone());
                            }
                            if !model_failure(&open_error) {
                                continue;
                            }
                            // A verified model refusal at establishment —
                            // the lane's own setup answered about the
                            // model, not about the lane — walks the chain
                            // on this same lane whether the reopen is a
                            // session-bound retry or a fallback's fresh
                            // alternate. The candidate stays pending: the
                            // installed session and the accepted binding
                            // carry no sign of it. But only the setup
                            // call that produced the refusal can answer
                            // about the model: where it binds nothing —
                            // a model-free `initialize` ahead of a
                            // model-carrying `session/open`, or an open
                            // whose model is only a stored default
                            // feeding later prompts — every retry
                            // re-sends the identical call, so the refusal
                            // is the lane's answer and the walk moves on.
                            if !refused_call_binds_model(&spec, resuming(&new_open), &probe) {
                                continue;
                            }
                            // The refused open's model is spent: a shared
                            // `pending` claim still holding it would
                            // re-enter on the next implicit prompt when no
                            // replacement installs, and the driver's own
                            // candidates must not ride this walk again.
                            // The installed session storing it as its
                            // default keeps the poisoned-slot mark — an
                            // open that never carried it — until a clean
                            // replacement takes over.
                            if let Some(refused) = new_open.config.model.clone() {
                                {
                                    let mut chain = model_state
                                        .lock()
                                        .unwrap_or_else(|error| error.into_inner());
                                    if chain.pending.as_deref() == Some(refused.as_str()) {
                                        chain.pending = None;
                                    }
                                    // A refused entry is spent even where
                                    // it was the last accepted one: left
                                    // `active`, the repair reopen or a
                                    // later lane switch would select it
                                    // again and reinstall or resend it.
                                    // The installed session's own record —
                                    // `lane.open`'s stored slot and the
                                    // mark below — still describes it.
                                    if chain.active.as_deref() == Some(refused.as_str()) {
                                        chain.active = None;
                                    }
                                }
                                // The stored turn default is spent with
                                // it: left holding the refused entry, the
                                // still-installed lane's next implicit
                                // prompt would resend it.
                                {
                                    let mut default = turn_model_default
                                        .lock()
                                        .unwrap_or_else(|error| error.into_inner());
                                    if default.as_deref() == Some(refused.as_str()) {
                                        *default = None;
                                    }
                                }
                                if lane.open.config.model.as_deref() == Some(refused.as_str())
                                    && !lane.probe.model_setup_sent()
                                {
                                    lane.refused_default = Some(refused.clone());
                                }
                                for slot in
                                    [&mut pending_bind, &mut driver_model, &mut pending_model]
                                {
                                    if slot.as_deref() == Some(refused.as_str()) {
                                        *slot = None;
                                    }
                                }
                            }
                            let next = {
                                let mut chain = model_state
                                    .lock()
                                    .unwrap_or_else(|error| error.into_inner());
                                // The retry reopens this same lane, so the
                                // delivery check the prompt pick applies
                                // stands here too: an entry the lane can
                                // neither bind at setup nor resolve in a
                                // turn slot would reopen with a model-free
                                // request — accepted on the peer's own
                                // default while the entries behind it were
                                // never tried. Hold it out of this lane's
                                // walk; a later lane or turn may still
                                // carry it.
                                while let Some(candidate) = chain.remaining.first() {
                                    if turn_carries_candidate(&spec, candidate)
                                        || setup_carries_candidate(
                                            &spec,
                                            candidate,
                                            resuming(&new_open),
                                        )
                                    {
                                        break;
                                    }
                                    let candidate = candidate.clone();
                                    chain.remaining.remove(0);
                                    if !held_unsent.iter().any(|entry| entry == &candidate) {
                                        held_unsent.push(candidate);
                                    }
                                }
                                let next = chain.remaining.first().cloned();
                                if next.is_some() {
                                    chain.remaining.remove(0);
                                }
                                // The shared pending claim is an eligible
                                // fallback too: it was parked awaiting its
                                // carry evidence, never refused, so the
                                // walk offers it before declaring the
                                // chain exhausted. The claim is a default
                                // fill — a direct prompt never consumes it —
                                // and a lane whose slots cannot resolve it
                                // has no retry to spend it on.
                                next.or_else(|| {
                                    chain.pending.clone().filter(|pending| {
                                        record_instructions
                                            && (turn_carries_candidate(&spec, pending)
                                                || setup_carries_candidate(
                                                    &spec,
                                                    pending,
                                                    resuming(&new_open),
                                                ))
                                    })
                                })
                            };
                            if let Some(next) = next {
                                pending_bind = Some(next);
                                pending_sent = false;
                                retry_same_lane = true;
                                continue;
                            }
                            // The candidates this lane cannot express
                            // stayed untried — held out of its walk, or
                            // queued on the shared claim. A later lane's
                            // slots may still carry them: advance the
                            // fallback and let the lane change restore
                            // them rather than declaring exhaustion
                            // while candidates remain. Only with nothing
                            // untried is this refusal the chain's
                            // answer.
                            let untried = !held_unsent.is_empty()
                                || (record_instructions
                                    && model_state
                                        .lock()
                                        .unwrap_or_else(|error| error.into_inner())
                                        .pending
                                        .is_some());
                            if untried {
                                continue;
                            }
                            break 'lanes Err(open_error);
                        }
                    }
                    // A candidate still pending acceptance on the
                    // per-turn surface keeps the last accepted turn
                    // default until its evidence lands; every other
                    // install re-projects the default — including a
                    // candidate already bound at open, where the
                    // replaced lane's default is now stale. The effort
                    // default is unrelated to the model chain and
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
                    // A retire reopen only clears the stored default —
                    // the refusal already answered this turn, so nothing
                    // replays on the replacement session. A scheduled
                    // candidate the install left awaiting evidence —
                    // pending on the per-turn surface — never dispatched:
                    // it returns to the chain while no wire carried it,
                    // the same restoration rule the abandoned-open
                    // guards apply.
                    if retire_default {
                        if !pending_sent {
                            restore_candidate(
                                &model_state,
                                pending_model.take().or_else(|| driver_model.take()),
                            );
                        }
                        break 'lanes outcome;
                    }
                    // The repair iteration ran no attempt: its
                    // synthesized outcome is not a completed attempt's
                    // result — after the replacement installs, the
                    // 'lanes loop re-enters and the turn dispatches on
                    // the clean session.
                    if !repair {
                        prior = Some(outcome);
                    }
                    // Held unsent candidates return to the chain with
                    // the lane change — the replacement's own slots may
                    // carry what this lane never could.
                    for candidate in held_unsent.drain(..).rev() {
                        restore_candidate(&model_state, Some(candidate));
                    }
                    continue 'lanes;
                }
            };
            // Held unsent candidates survive the turn: restored to the
            // chain, the next prompt may still try them on a lane that
            // can carry them.
            for candidate in held_unsent.drain(..).rev() {
                restore_candidate(&model_state, Some(candidate));
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn probed_probe() -> Arc<DispatchProbe> {
        DispatchProbe::new(
            None,
            &AgentSpec {
                id: "rpc".to_owned(),
                client_name: None,
                lane: LaneSpec::Rpc(Box::new(a4aa::RpcLaneSpec {
                    prompt: a4aa::RpcCallSpec {
                        method: "session/prompt".to_owned(),
                        params: serde_json::json!({"text": "{prompt}"}),
                    },
                    ..Default::default()
                })),
                command: a4aa::CommandSpec {
                    program: "fixture".to_owned(),
                    args: Vec::new(),
                    cwd: None,
                    env: Default::default(),
                },
                max_output_bytes: None,
                seatbelted_permission: None,
            },
            &SessionRequest::default(),
        )
    }

    fn unprobed_probe() -> Arc<DispatchProbe> {
        DispatchProbe::new(
            None,
            &AgentSpec {
                id: "cli".to_owned(),
                client_name: None,
                lane: LaneSpec::Cli(Box::default()),
                command: a4aa::CommandSpec {
                    program: "fixture".to_owned(),
                    args: Vec::new(),
                    cwd: None,
                    env: Default::default(),
                },
                max_output_bytes: None,
                seatbelted_permission: None,
            },
            &SessionRequest::default(),
        )
    }

    fn armed() -> RelayEvidence {
        RelayEvidence {
            armed: AtomicBool::new(true),
            output: AtomicBool::new(false),
            billable: AtomicBool::new(false),
        }
    }

    /// On a probed lane the wire journal owns output attribution: an
    /// event received before the prompt but relayed only after dispatch
    /// must not read as the attempt's billable output — the tap still
    /// forwards it to the caller.
    #[test]
    fn probed_lane_relay_never_counts_tap_evidence() {
        let probe = probed_probe();
        probe.dispatched.store(true, Ordering::Release);
        let (events, mut rx) = mpsc::unbounded_channel();
        let evidence = armed();
        EventRelay::forward(
            &evidence,
            &events,
            &probe,
            AgentEvent::MessageChunk {
                text: "paid".to_owned(),
            },
        );
        assert!(!evidence.output.load(Ordering::Acquire));
        assert!(!evidence.billable.load(Ordering::Acquire));
        assert!(matches!(rx.try_recv(), Ok(AgentEvent::MessageChunk { .. })));
    }

    /// Lanes without an observable dispatch point keep tap attribution:
    /// inside the armed window a relayed billable event is the attempt's
    /// output — the wire has no journal to attribute it instead.
    #[test]
    fn unprobed_lane_relay_counts_armed_output() {
        let probe = unprobed_probe();
        let (events, mut rx) = mpsc::unbounded_channel();
        let evidence = armed();
        EventRelay::forward(
            &evidence,
            &events,
            &probe,
            AgentEvent::MessageChunk {
                text: "paid".to_owned(),
            },
        );
        assert!(evidence.output.load(Ordering::Acquire));
        assert!(evidence.billable.load(Ordering::Acquire));
        assert!(matches!(rx.try_recv(), Ok(AgentEvent::MessageChunk { .. })));
    }

    /// Bookkeeping-only records never mark an attempt served on an
    /// unprobed lane: an empty chunk, an all-zero usage event and a raw
    /// record whose only claim is a usage document are not output —
    /// while a genuine progress record still is.
    #[test]
    fn unprobed_lane_relay_ignores_bookkeeping_only_records() {
        let probe = unprobed_probe();
        let (events, _rx) = mpsc::unbounded_channel();
        let evidence = armed();
        for event in [
            AgentEvent::MessageChunk {
                text: String::new(),
            },
            AgentEvent::Usage {
                usage: TokenUsage::default(),
                raw: None,
            },
            AgentEvent::Raw {
                raw: serde_json::json!({
                    "type": "result",
                    "result": "",
                    "usage": {"inputTokens": 0, "outputTokens": 0}
                }),
            },
        ] {
            EventRelay::forward(&evidence, &events, &probe, event);
        }
        assert!(!evidence.output.load(Ordering::Acquire));
        assert!(!evidence.billable.load(Ordering::Acquire));
        EventRelay::forward(
            &evidence,
            &events,
            &probe,
            AgentEvent::Raw {
                raw: serde_json::json!({"type": "progress", "message": "working"}),
            },
        );
        assert!(evidence.output.load(Ordering::Acquire));
        assert!(!evidence.billable.load(Ordering::Acquire));
    }

    /// A raw record that is genuine progress but carries a billable
    /// usage document marks both flags; a nonzero usage record alone
    /// still counts as paid output.
    #[test]
    fn unprobed_lane_relay_counts_billable_usage_record() {
        let probe = unprobed_probe();
        let (events, _rx) = mpsc::unbounded_channel();
        let evidence = armed();
        EventRelay::forward(
            &evidence,
            &events,
            &probe,
            AgentEvent::Raw {
                raw: serde_json::json!({
                    "type": "result",
                    "result": "",
                    "usage": {"inputTokens": 3, "outputTokens": 7}
                }),
            },
        );
        assert!(evidence.output.load(Ordering::Acquire));
        assert!(evidence.billable.load(Ordering::Acquire));
    }

    /// A spec builder for probe-only tests: the probe's setup-call list
    /// is the open request's actual selection, so tests declare the
    /// same `resuming` flag the driver would compute.
    fn open_request(model: Option<&str>) -> SessionRequest {
        SessionRequest {
            config: a4aa::SessionConfig {
                model: model.map(str::to_owned),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn probe_for(rpc: a4aa::RpcLaneSpec, model: Option<&str>) -> Arc<DispatchProbe> {
        DispatchProbe::new(
            None,
            &AgentSpec {
                id: "rpc".to_owned(),
                client_name: None,
                lane: LaneSpec::Rpc(Box::new(rpc)),
                command: a4aa::CommandSpec {
                    program: "fixture".to_owned(),
                    args: Vec::new(),
                    cwd: None,
                    env: Default::default(),
                },
                max_output_bytes: None,
                seatbelted_permission: None,
            },
            &open_request(model),
        )
    }

    /// Scalar setup params carry an empty slot path: the inline-envelope
    /// control-field filter applies only where the params merge into the
    /// frame root — an object document — so a scalar `{model}` keeps its
    /// root position instead of indexing past the empty path, and the
    /// journal still reads the substituted `params` value as
    /// model-carrying.
    #[test]
    fn scalar_setup_params_never_panics_under_inline_envelope() {
        let probe = probe_for(
            a4aa::RpcLaneSpec {
                session_open: Some(a4aa::RpcCallSpec {
                    method: "session/open".to_owned(),
                    params: serde_json::json!("{model}"),
                }),
                prompt: a4aa::RpcCallSpec {
                    method: "session/prompt".to_owned(),
                    params: serde_json::json!("{prompt}"),
                },
                envelope: Some(a4aa::RpcEnvelopeSpec {
                    method_field: "cmd".to_owned(),
                    params_inline: true,
                    response_method: Some("response".to_owned()),
                    ..Default::default()
                }),
                ..Default::default()
            },
            Some("prime"),
        );
        probe.record(
            "rpc",
            Direction::Sent,
            br#"{"cmd":"session/open","id":1,"params":"prime"}"#,
        );
        probe.record(
            "rpc",
            Direction::Received,
            br#"{"cmd":"response","id":1,"error":{"code":-32000,"message":"quota"}}"#,
        );
        assert!(probe.refused_setup_binds_model());
    }

    /// Two setup calls sharing one wire method attach to the call the
    /// adapter's own issue order selected — `initialize` first, then the
    /// session call — and an outbound callback reply that echoes the
    /// answered command is a response, never a setup request, so it
    /// claims no journal slot. A fixed-model handshake's refusal stays
    /// model-free while the model-bearing open's refusal binds.
    #[test]
    fn shared_method_journal_attaches_each_request_to_its_call() {
        let probe = probe_for(
            a4aa::RpcLaneSpec {
                initialize: Some(a4aa::RpcCallSpec {
                    method: "negotiate".to_owned(),
                    params: serde_json::json!({"model": "handshake"}),
                }),
                session_open: Some(a4aa::RpcCallSpec {
                    method: "negotiate".to_owned(),
                    params: serde_json::json!({"model": "{model}"}),
                }),
                prompt: a4aa::RpcCallSpec {
                    method: "session/prompt".to_owned(),
                    params: serde_json::json!({"text": "{prompt}"}),
                },
                envelope: Some(a4aa::RpcEnvelopeSpec {
                    method_field: "cmd".to_owned(),
                    discriminant_field: Some("type".to_owned()),
                    request_marker: Some("request".to_owned()),
                    response_method: Some("response".to_owned()),
                    command_field: Some("cmd".to_owned()),
                    ..Default::default()
                }),
                ..Default::default()
            },
            Some("prime"),
        );
        // The host's reply to a peer `negotiate` callback echoes the
        // command — under the method-name reading it is the same frame
        // shape as a setup request.
        probe.record(
            "rpc",
            Direction::Sent,
            br#"{"type":"response","cmd":"negotiate","id":"cb-1","error":{"code":-32601}}"#,
        );
        probe.record(
            "rpc",
            Direction::Sent,
            br#"{"type":"request","cmd":"negotiate","id":1,"params":{"model":"handshake"}}"#,
        );
        probe.record(
            "rpc",
            Direction::Sent,
            br#"{"type":"request","cmd":"negotiate","id":2,"params":{"model":"prime"}}"#,
        );
        probe.record(
            "rpc",
            Direction::Received,
            br#"{"type":"response","id":1,"error":{"code":-32000,"message":"quota"}}"#,
        );
        assert!(!probe.refused_setup_binds_model());
        probe.record(
            "rpc",
            Direction::Received,
            br#"{"type":"response","id":2,"error":{"code":-32000,"message":"quota"}}"#,
        );
        assert!(probe.refused_setup_binds_model());
    }

    /// Usage that exists only inside the prompt's synchronous response
    /// still reaches the served check: the reply is turn evidence by
    /// request-id correlation, and a nonzero usage record in its result
    /// payload proves the paid turn ran — while an all-zero record does
    /// not.
    #[test]
    fn prompt_response_usage_reads_as_wire_billable() {
        let probe = probed_probe();
        probe.reset("work", None, None);
        probe.record(
            "rpc",
            Direction::Sent,
            br#"{"method":"session/prompt","id":7,"params":{"text":"work"}}"#,
        );
        probe.record(
            "rpc",
            Direction::Received,
            br#"{"id":7,"result":{"text":"","status":"error","usage":{"inputTokens":3,"outputTokens":7}}}"#,
        );
        assert!(probe.wire_event());
        assert!(probe.wire_billable());

        probe.reset("work", None, None);
        probe.record(
            "rpc",
            Direction::Sent,
            br#"{"method":"session/prompt","id":9,"params":{"text":"work"}}"#,
        );
        probe.record(
            "rpc",
            Direction::Received,
            br#"{"id":9,"result":{"text":"","status":"error","usage":{"inputTokens":0,"outputTokens":0}}}"#,
        );
        assert!(!probe.wire_billable());
    }

    /// Once `reset` opens the turn phase, requests journal against the
    /// lane's pre-turn calls: a refusal correlates by request id to the
    /// exact call it answered — a model-free preparation call's refusal
    /// is the lane's answer, a model-carrying one's is the model's, and
    /// the prompt's refusal always binds.
    #[test]
    fn pre_turn_refusal_attaches_to_its_own_call() {
        let probe = probe_for(
            a4aa::RpcLaneSpec {
                pre_turn: vec![
                    a4aa::RpcCallSpec {
                        method: "session/select".to_owned(),
                        params: serde_json::json!({"select": "x"}),
                    },
                    a4aa::RpcCallSpec {
                        method: "session/bind".to_owned(),
                        params: serde_json::json!({"model": "{model}"}),
                    },
                ],
                prompt: a4aa::RpcCallSpec {
                    method: "session/prompt".to_owned(),
                    params: serde_json::json!({"text": "{prompt}", "model": "{model}"}),
                },
                ..Default::default()
            },
            Some("prime"),
        );
        probe.reset("work", None, None);
        probe.record(
            "rpc",
            Direction::Sent,
            br#"{"method":"session/select","id":1,"params":{"select":"x"}}"#,
        );
        probe.record(
            "rpc",
            Direction::Received,
            br#"{"id":1,"error":{"code":-32000,"message":"quota"}}"#,
        );
        assert!(!probe.refused_turn_binds_model());
        assert!(!probe.turn_model_sent());

        probe.reset("work", None, None);
        probe.record(
            "rpc",
            Direction::Sent,
            br#"{"method":"session/bind","id":2,"params":{"model":"prime"}}"#,
        );
        assert!(probe.turn_model_sent());
        probe.record(
            "rpc",
            Direction::Received,
            br#"{"id":2,"error":{"code":-32000,"message":"quota"}}"#,
        );
        assert!(probe.refused_turn_binds_model());

        probe.reset("work", None, None);
        probe.record(
            "rpc",
            Direction::Sent,
            br#"{"method":"session/prompt","id":3,"params":{"text":"work","model":"prime"}}"#,
        );
        probe.record(
            "rpc",
            Direction::Received,
            br#"{"id":3,"error":{"code":-32000,"message":"quota"}}"#,
        );
        assert!(probe.refused_turn_binds_model());
    }

    /// A cli probe knows the lane's own bookkeeping surface: a usage
    /// document at the configured `usage_path` and an empty result at
    /// the declared `result_selector`/`result_path` are bookkeeping,
    /// while a record carrying progress text beside all-zero counters
    /// is real output.
    #[test]
    fn cli_probe_reads_bookkeeping_through_the_declared_fields() {
        let probe = DispatchProbe::new(
            None,
            &AgentSpec {
                id: "cli".to_owned(),
                client_name: None,
                lane: LaneSpec::Cli(Box::new(a4aa::CliLaneSpec {
                    json_lines: true,
                    result_selector: Some("result".to_owned()),
                    result_path: Some("result".to_owned()),
                    usage_path: Some("stats.tokens".to_owned()),
                    session_id_path: Some("session_id".to_owned()),
                    ..Default::default()
                })),
                command: a4aa::CommandSpec {
                    program: "fixture".to_owned(),
                    args: Vec::new(),
                    cwd: None,
                    env: Default::default(),
                },
                max_output_bytes: None,
                seatbelted_permission: None,
            },
            &SessionRequest::default(),
        );
        for record in [
            serde_json::json!({"type":"usage","stats":{"tokens":{"inputTokens":0,"outputTokens":0}}}),
            serde_json::json!({"type":"result","result":""}),
            serde_json::json!({"type":"result","result":"","usage":{"inputTokens":0,"outputTokens":0}}),
            serde_json::json!({"type":"system","subtype":"init","session_id":"sess-1"}),
        ] {
            assert!(
                probe.bookkeeping_raw(&record),
                "bookkeeping record: {record}"
            );
        }
        for record in [
            serde_json::json!({"type":"assistant","message":{"content":"real work","usage":{"inputTokens":0,"outputTokens":0}}}),
            serde_json::json!({"type":"result","result":"real work","usage":{"inputTokens":0,"outputTokens":0}}),
            serde_json::json!({"type":"progress","message":"working"}),
            // The session binding does not launder real activity: a
            // textless tool call is still work the caller saw.
            serde_json::json!({"type":"assistant","session_id":"sess-1","message":{"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"Bash","input":{"command":"ls"}}]}}),
        ] {
            assert!(!probe.bookkeeping_raw(&record), "progress record: {record}");
        }
    }

    /// Retry eligibility reads the same effective wire slots the probe
    /// enforces: an inline envelope stamps `id` over the merged params,
    /// so a `{model}` leaf there binds nothing — while a surviving root
    /// position still does.
    #[test]
    fn overwritten_inline_model_slot_never_binds_the_open() {
        let rpc = |session_params: serde_json::Value| a4aa::RpcLaneSpec {
            session_open: Some(a4aa::RpcCallSpec {
                method: "session/open".to_owned(),
                params: session_params,
            }),
            prompt: a4aa::RpcCallSpec {
                method: "session/prompt".to_owned(),
                params: serde_json::json!({"text": "{prompt}"}),
            },
            envelope: Some(a4aa::RpcEnvelopeSpec {
                method_field: "cmd".to_owned(),
                params_inline: true,
                response_method: Some("response".to_owned()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let spec = |rpc| AgentSpec {
            id: "rpc".to_owned(),
            client_name: None,
            lane: LaneSpec::Rpc(Box::new(rpc)),
            command: a4aa::CommandSpec {
                program: "fixture".to_owned(),
                args: Vec::new(),
                cwd: None,
                env: Default::default(),
            },
            max_output_bytes: None,
            seatbelted_permission: None,
        };
        assert!(!model_binds_at_open(
            &spec(rpc(serde_json::json!({"id": "{model}", "note": "x"}))),
            false,
        ));
        assert!(model_binds_at_open(
            &spec(rpc(serde_json::json!({"note": "{model}"}))),
            false,
        ));
    }

    /// A skipped call leaves no frame, so a same-method request must
    /// claim the call the adapter actually issued — not the first
    /// unclaimed template. Here the effort call's leaves all resolve
    /// empty so the lane skips it; the one select frame on the wire is
    /// the model call's, and its refusal is the model's answer.
    #[test]
    fn skipped_same_method_call_keeps_the_later_calls_verdict() {
        let probe = probe_for(
            a4aa::RpcLaneSpec {
                pre_turn: vec![
                    a4aa::RpcCallSpec {
                        method: "session/select".to_owned(),
                        params: serde_json::json!({"effort": "{effort}"}),
                    },
                    a4aa::RpcCallSpec {
                        method: "session/select".to_owned(),
                        params: serde_json::json!({"model": "{model}", "keep": "x"}),
                    },
                ],
                prompt: a4aa::RpcCallSpec {
                    method: "session/prompt".to_owned(),
                    params: serde_json::json!({"text": "{prompt}", "model": "{model}"}),
                },
                ..Default::default()
            },
            Some("prime"),
        );
        probe.reset("work", None, None);
        probe.record(
            "rpc",
            Direction::Sent,
            br#"{"method":"session/select","id":1,"params":{"model":"prime","keep":"x"}}"#,
        );
        assert!(probe.turn_model_sent());
        probe.record(
            "rpc",
            Direction::Received,
            br#"{"id":1,"error":{"code":-32000,"message":"quota"}}"#,
        );
        assert!(probe.refused_turn_binds_model());
    }

    /// The same-key case the issue order cannot separate: two pre-turn
    /// calls share both the method and the parameter name — the effort
    /// call's `{effort}` resolves absent so the lane emits nothing for
    /// it, and the select frame on the wire is the model call's
    /// resolved `"prime"`. The skipped call's journal slot cannot claim
    /// it: the refusal attaches to the model.
    #[test]
    fn same_key_skipped_call_keeps_the_later_calls_verdict() {
        let probe = probe_for(
            a4aa::RpcLaneSpec {
                pre_turn: vec![
                    a4aa::RpcCallSpec {
                        method: "session/select".to_owned(),
                        params: serde_json::json!({"choice": "{effort}"}),
                    },
                    a4aa::RpcCallSpec {
                        method: "session/select".to_owned(),
                        params: serde_json::json!({"choice": "{model}"}),
                    },
                ],
                prompt: a4aa::RpcCallSpec {
                    method: "session/prompt".to_owned(),
                    params: serde_json::json!({"text": "{prompt}"}),
                },
                ..Default::default()
            },
            Some("prime"),
        );
        probe.reset("work", None, None);
        probe.record(
            "rpc",
            Direction::Sent,
            br#"{"method":"session/select","id":1,"params":{"choice":"prime"}}"#,
        );
        assert!(probe.turn_model_sent());
        probe.record(
            "rpc",
            Direction::Received,
            br#"{"id":1,"error":{"code":-32000,"message":"quota"}}"#,
        );
        assert!(probe.refused_turn_binds_model());
    }

    /// A nested object's leaves drop, the object stays: the open
    /// template's `options` keeps its position as `{}` when `{effort}`
    /// resolves absent, exactly the shape the lane emits — the frame
    /// still claims the session call, and its refusal binds the model.
    #[test]
    fn nested_empty_object_keeps_the_setup_calls_verdict() {
        let probe = probe_for(
            a4aa::RpcLaneSpec {
                session_open: Some(a4aa::RpcCallSpec {
                    method: "session/open".to_owned(),
                    params: serde_json::json!({
                        "model": "{model}",
                        "options": {"effort": "{effort}"}
                    }),
                }),
                prompt: a4aa::RpcCallSpec {
                    method: "session/prompt".to_owned(),
                    params: serde_json::json!({"text": "{prompt}"}),
                },
                ..Default::default()
            },
            Some("prime"),
        );
        probe.record(
            "rpc",
            Direction::Sent,
            br#"{"method":"session/open","id":1,"params":{"model":"prime","options":{}}}"#,
        );
        probe.record(
            "rpc",
            Direction::Received,
            br#"{"id":1,"error":{"code":-32000,"message":"quota"}}"#,
        );
        assert!(probe.refused_setup_binds_model());
    }

    /// The same effective reading applies to the turn surface: an
    /// inline envelope stamps `id` over the merged prompt params, so a
    /// `{model}` leaf there reaches the wire as the request id — it
    /// cannot carry a chain retry. A surviving root position still
    /// binds.
    #[test]
    fn overwritten_inline_turn_slot_reports_no_turn_surface() {
        let rpc = |prompt_params: serde_json::Value| a4aa::RpcLaneSpec {
            session_open: Some(a4aa::RpcCallSpec {
                method: "session/open".to_owned(),
                params: serde_json::json!({}),
            }),
            prompt: a4aa::RpcCallSpec {
                method: "session/prompt".to_owned(),
                params: prompt_params,
            },
            envelope: Some(a4aa::RpcEnvelopeSpec {
                method_field: "cmd".to_owned(),
                params_inline: true,
                response_method: Some("response".to_owned()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let spec = |rpc| AgentSpec {
            id: "rpc".to_owned(),
            client_name: None,
            lane: LaneSpec::Rpc(Box::new(rpc)),
            command: a4aa::CommandSpec {
                program: "fixture".to_owned(),
                args: Vec::new(),
                cwd: None,
                env: Default::default(),
            },
            max_output_bytes: None,
            seatbelted_permission: None,
        };
        assert!(!turn_binds_model(&spec(rpc(
            serde_json::json!({"text": "{prompt}", "id": "{model}"})
        ))));
        assert!(turn_binds_model(&spec(rpc(
            serde_json::json!({"text": "{prompt}", "note": "{model}"})
        ))));
    }

    /// The prompt frame itself reports whether the model reached the
    /// wire: a slot the inline envelope stamps over is absent from the
    /// encoded params, so the sent frame populates no surviving slot —
    /// while a frame whose slot carries a value is model-sent.
    #[test]
    fn prompt_frame_reports_only_surviving_model_slots() {
        let spec = |prompt_params: serde_json::Value| AgentSpec {
            id: "rpc".to_owned(),
            client_name: None,
            lane: LaneSpec::Rpc(Box::new(a4aa::RpcLaneSpec {
                session_open: Some(a4aa::RpcCallSpec {
                    method: "session/open".to_owned(),
                    params: serde_json::json!({}),
                }),
                prompt: a4aa::RpcCallSpec {
                    method: "session/prompt".to_owned(),
                    params: prompt_params,
                },
                envelope: Some(a4aa::RpcEnvelopeSpec {
                    method_field: "cmd".to_owned(),
                    params_inline: true,
                    response_method: Some("response".to_owned()),
                    ..Default::default()
                }),
                ..Default::default()
            })),
            command: a4aa::CommandSpec {
                program: "fixture".to_owned(),
                args: Vec::new(),
                cwd: None,
                env: Default::default(),
            },
            max_output_bytes: None,
            seatbelted_permission: None,
        };
        let masked = DispatchProbe::new(
            None,
            &spec(serde_json::json!({"text": "{prompt}", "id": "{model}"})),
            &open_request(Some("prime")),
        );
        masked.reset("work", None, None);
        masked.record(
            "rpc",
            Direction::Sent,
            br#"{"cmd":"session/prompt","id":7,"text":"work"}"#,
        );
        assert!(masked.dispatched());
        assert!(!masked.prompt_model_sent());

        let carried = DispatchProbe::new(
            None,
            &spec(serde_json::json!({"text": "{prompt}", "note": "{model}"})),
            &open_request(Some("prime")),
        );
        carried.reset("work", None, None);
        carried.record(
            "rpc",
            Direction::Sent,
            br#"{"cmd":"session/prompt","id":7,"text":"work","note":"prime"}"#,
        );
        assert!(carried.prompt_model_sent());
    }

    /// A cli record's selectors read the record itself, never an
    /// unwrapped member: a failed turn's terminal record carries an
    /// empty `params` sibling, and reading the member instead of the
    /// record would mistake it for progress.
    #[test]
    fn cli_result_record_reads_selectors_at_the_record_root() {
        let probe = DispatchProbe::new(
            None,
            &AgentSpec {
                id: "cli".to_owned(),
                client_name: None,
                lane: LaneSpec::Cli(Box::new(a4aa::CliLaneSpec {
                    json_lines: true,
                    result_selector: Some("result".to_owned()),
                    result_path: Some("result".to_owned()),
                    usage_path: Some("usage".to_owned()),
                    ..Default::default()
                })),
                command: a4aa::CommandSpec {
                    program: "fixture".to_owned(),
                    args: Vec::new(),
                    cwd: None,
                    env: Default::default(),
                },
                max_output_bytes: None,
                seatbelted_permission: None,
            },
            &SessionRequest::default(),
        );
        assert!(
            probe.bookkeeping_raw(&serde_json::json!({"type":"result","result":"","params":{}}))
        );
    }

    /// A json_lines cli outcome whose text is the retained raw stream —
    /// every line a record — repeats bookkeeping the relay already
    /// classified: it is not served evidence. Any non-record line is
    /// real output.
    #[test]
    fn retained_ndjson_text_is_not_served_evidence() {
        let probe = DispatchProbe::new(
            None,
            &AgentSpec {
                id: "cli".to_owned(),
                client_name: None,
                lane: LaneSpec::Cli(Box::new(a4aa::CliLaneSpec {
                    json_lines: true,
                    ..Default::default()
                })),
                command: a4aa::CommandSpec {
                    program: "fixture".to_owned(),
                    args: Vec::new(),
                    cwd: None,
                    env: Default::default(),
                },
                max_output_bytes: None,
                seatbelted_permission: None,
            },
            &SessionRequest::default(),
        );
        let outcome = |text: &str| a4aa::TurnOutcome {
            stop_reason: "error".to_owned(),
            text: text.to_owned(),
            truncated: false,
            exit_code: Some(1),
            usage: None,
        };
        assert!(!error_outcome_served(
            &outcome("{\"type\":\"usage\",\"usage\":{\"inputTokens\":0,\"outputTokens\":0}}\n{\"type\":\"summary\",\"usage\":{\"inputTokens\":0,\"outputTokens\":0}}\n"),
            false,
            &probe,
        ));
        // Blank separators between records are part of the stream the
        // lane ignored — never output of their own.
        assert!(!error_outcome_served(
            &outcome("{\"type\":\"usage\"}\n\n{\"type\":\"summary\"}\n  \n"),
            false,
            &probe,
        ));
        assert!(error_outcome_served(
            &outcome("{\"type\":\"usage\"}\nplain tail"),
            false,
            &probe,
        ));
    }

    /// An absent `sessionId` inside a larger string stays literal on the
    /// wire — the adapter never puts an absent id into `values`, so
    /// `open-{sessionId}` emits verbatim. The journaled frame must
    /// match the literal shape or the open's refusal loses its model
    /// attribution entirely.
    #[test]
    fn embedded_absent_session_id_stays_literal_in_setup_matching() {
        let probe = probe_for(
            a4aa::RpcLaneSpec {
                session_open: Some(a4aa::RpcCallSpec {
                    method: "session/open".to_owned(),
                    params: serde_json::json!({"marker": "open-{sessionId}", "model": "{model}"}),
                }),
                prompt: a4aa::RpcCallSpec {
                    method: "session/prompt".to_owned(),
                    params: serde_json::json!({"text": "{prompt}"}),
                },
                ..Default::default()
            },
            Some("prime"),
        );
        probe.record(
            "rpc",
            Direction::Sent,
            br#"{"method":"session/open","id":1,"params":{"marker":"open-{sessionId}","model":"prime"}}"#,
        );
        probe.record(
            "rpc",
            Direction::Received,
            br#"{"id":1,"error":{"code":-32000,"message":"quota"}}"#,
        );
        assert!(probe.refused_setup_binds_model());
    }

    /// The same literal rule holds in the turn phase: `{turnId}` is
    /// never supplied, so a marker embedding it emits verbatim and the
    /// frame still claims the call that produced it.
    #[test]
    fn embedded_absent_turn_id_stays_literal_in_turn_matching() {
        let probe = probe_for(
            a4aa::RpcLaneSpec {
                pre_turn: vec![a4aa::RpcCallSpec {
                    method: "session/select".to_owned(),
                    params: serde_json::json!({"marker": "turn-{turnId}", "model": "{model}"}),
                }],
                prompt: a4aa::RpcCallSpec {
                    method: "session/prompt".to_owned(),
                    params: serde_json::json!({"text": "{prompt}"}),
                },
                ..Default::default()
            },
            Some("prime"),
        );
        probe.reset("work", None, None);
        probe.record(
            "rpc",
            Direction::Sent,
            br#"{"method":"session/select","id":1,"params":{"marker":"turn-{turnId}","model":"prime"}}"#,
        );
        assert!(probe.turn_model_sent());
        probe.record(
            "rpc",
            Direction::Received,
            br#"{"id":1,"error":{"code":-32000,"message":"quota"}}"#,
        );
        assert!(probe.refused_turn_binds_model());
    }

    /// A peer that answers the open with no session id leaves
    /// `{sessionId}` absent — not flexible: the id-only pre-turn call
    /// resolves `{}` and the lane skips it, so the lone select frame on
    /// the wire is the model call's and its refusal binds the model.
    #[test]
    fn sessionless_answer_skips_the_id_only_call() {
        let probe = probe_for(
            a4aa::RpcLaneSpec {
                session_open: Some(a4aa::RpcCallSpec {
                    method: "session/open".to_owned(),
                    params: serde_json::json!({}),
                }),
                pre_turn: vec![
                    a4aa::RpcCallSpec {
                        method: "session/select".to_owned(),
                        params: serde_json::json!({"session": "{sessionId}"}),
                    },
                    a4aa::RpcCallSpec {
                        method: "session/select".to_owned(),
                        params: serde_json::json!({"session": "{model}"}),
                    },
                ],
                prompt: a4aa::RpcCallSpec {
                    method: "session/prompt".to_owned(),
                    params: serde_json::json!({"text": "{prompt}"}),
                },
                ..Default::default()
            },
            Some("prime"),
        );
        // The session call's result carries no id — the presence the
        // adapter substitutes into every later call settles to absent.
        probe.record(
            "rpc",
            Direction::Sent,
            br#"{"method":"session/open","id":0,"params":{}}"#,
        );
        probe.record("rpc", Direction::Received, br#"{"id":0,"result":{}}"#);
        probe.reset("work", None, None);
        probe.record(
            "rpc",
            Direction::Sent,
            br#"{"method":"session/select","id":1,"params":{"session":"prime"}}"#,
        );
        assert!(probe.turn_model_sent());
        probe.record(
            "rpc",
            Direction::Received,
            br#"{"id":1,"error":{"code":-32000,"message":"quota"}}"#,
        );
        assert!(probe.refused_turn_binds_model());
    }

    /// An unresolved `{secret:NAME}` drops its leaf like any absent
    /// optional selector — presence the probe learns from the adapter's
    /// own resolution journal. The secret-only call resolves `{}` and
    /// the lane skips it; the select frame on the wire is the model
    /// call's and its refusal binds.
    #[test]
    fn unresolved_secret_skips_the_secret_only_call() {
        let probe = probe_for(
            a4aa::RpcLaneSpec {
                pre_turn: vec![
                    a4aa::RpcCallSpec {
                        method: "session/select".to_owned(),
                        params: serde_json::json!({"choice": "{secret:AWP_TOKEN}"}),
                    },
                    a4aa::RpcCallSpec {
                        method: "session/select".to_owned(),
                        params: serde_json::json!({"choice": "{model}"}),
                    },
                ],
                prompt: a4aa::RpcCallSpec {
                    method: "session/prompt".to_owned(),
                    params: serde_json::json!({"text": "{prompt}"}),
                },
                ..Default::default()
            },
            Some("prime"),
        );
        probe.reset("work", None, None);
        // The adapter's resolver journaled the name absent at the
        // secret-only call's own substitution — the outcome record
        // `wrap_secrets` produces, inside the turn-phase slice that
        // call owns.
        probe
            .secrets
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .outcomes
            .push(("secret:AWP_TOKEN".to_owned(), false));
        probe.record(
            "rpc",
            Direction::Sent,
            br#"{"method":"session/select","id":1,"params":{"choice":"prime"}}"#,
        );
        assert!(probe.turn_model_sent());
        probe.record(
            "rpc",
            Direction::Received,
            br#"{"id":1,"error":{"code":-32000,"message":"quota"}}"#,
        );
        assert!(probe.refused_turn_binds_model());
    }

    /// A slot that stays populated is not automatically model-carrying:
    /// `v:{modelProvider}` still emits `v:` for an unqualified
    /// candidate — the model token resolved to nothing, so the frame
    /// carried no selector the peer could have bound. A qualified
    /// candidate resolves the token and does carry.
    #[test]
    fn populated_scaffold_slot_never_counts_as_model_carry() {
        let probe = probe_for(
            a4aa::RpcLaneSpec {
                prompt: a4aa::RpcCallSpec {
                    method: "session/prompt".to_owned(),
                    params: serde_json::json!({"vendor": "v:{modelProvider}", "text": "{prompt}"}),
                },
                ..Default::default()
            },
            None,
        );
        probe.reset("work", Some("fallback"), None);
        probe.record(
            "rpc",
            Direction::Sent,
            br#"{"method":"session/prompt","id":7,"params":{"vendor":"v:","text":"work"}}"#,
        );
        assert!(!probe.prompt_model_sent());

        probe.reset("work", Some("vnd/prime"), None);
        probe.record(
            "rpc",
            Direction::Sent,
            br#"{"method":"session/prompt","id":9,"params":{"vendor":"v:vnd","text":"work"}}"#,
        );
        assert!(probe.prompt_model_sent());
    }

    /// A same-method pre-turn call's frame is that call's, never the
    /// prompt's — even when the prompt template offers no verifiable
    /// `{prompt}` position for the wire to tell them apart. The frame
    /// claims its call before the prompt check can own it, so a
    /// model-free preparation call marks neither dispatch nor the
    /// candidate's send, and its refusal cannot bind the model.
    #[test]
    fn pre_turn_frame_claims_its_call_before_the_prompt() {
        let probe = probe_for(
            a4aa::RpcLaneSpec {
                pre_turn: vec![a4aa::RpcCallSpec {
                    method: "session/prompt".to_owned(),
                    params: serde_json::json!({"note": "configure"}),
                }],
                prompt: a4aa::RpcCallSpec {
                    method: "session/prompt".to_owned(),
                    params: serde_json::json!({"payload": "{prompt}|{model}"}),
                },
                ..Default::default()
            },
            Some("prime"),
        );
        probe.reset("work", Some("fallback"), None);
        // The pre-turn frame claims its own call — the prompt template's
        // model slot can never credit the candidate a send the prompt
        // never made.
        probe.record(
            "rpc",
            Direction::Sent,
            br#"{"method":"session/prompt","id":1,"params":{"note":"configure"}}"#,
        );
        assert!(!probe.dispatched());
        assert!(!probe.prompt_model_sent());
        assert!(!probe.turn_model_sent());
        probe.record(
            "rpc",
            Direction::Received,
            br#"{"id":1,"error":{"code":-32000,"message":"quota"}}"#,
        );
        // The refusal answers the model-free preparation call — never
        // the prompt's model verdict.
        assert!(!probe.refused_turn_binds_model());
        probe.record(
            "rpc",
            Direction::Sent,
            br#"{"method":"session/prompt","id":2,"params":{"payload":"work|fallback"}}"#,
        );
        assert!(probe.dispatched());
        assert!(probe.prompt_model_sent());
    }

    /// Presence belongs to the call whose substitution asked for it:
    /// the secret-only call resolves `AWP_TOKEN` to nothing and is
    /// skipped, and a later same-method call's own resolution of the
    /// same name — this time producing a value — must not reopen the
    /// skipped call's claim on the frame it produced.
    #[test]
    fn later_secret_resolution_never_unskips_the_call() {
        let probe = probe_for(
            a4aa::RpcLaneSpec {
                pre_turn: vec![
                    a4aa::RpcCallSpec {
                        method: "session/select".to_owned(),
                        params: serde_json::json!({"choice": "{secret:AWP_TOKEN}"}),
                    },
                    a4aa::RpcCallSpec {
                        method: "session/select".to_owned(),
                        params: serde_json::json!({"choice": "{secret:AWP_TOKEN}-{model}"}),
                    },
                ],
                prompt: a4aa::RpcCallSpec {
                    method: "session/prompt".to_owned(),
                    params: serde_json::json!({"text": "{prompt}"}),
                },
                ..Default::default()
            },
            Some("prime"),
        );
        probe.reset("work", None, None);
        // The adapter's own resolutions in substitution order: the
        // first call's lookup failed, the second's produced a value.
        {
            let mut secrets = probe
                .secrets
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            secrets
                .outcomes
                .push(("secret:AWP_TOKEN".to_owned(), false));
            secrets.outcomes.push(("secret:AWP_TOKEN".to_owned(), true));
        }
        probe.record(
            "rpc",
            Direction::Sent,
            br#"{"method":"session/select","id":1,"params":{"choice":"tok-prime"}}"#,
        );
        assert!(probe.turn_model_sent());
        probe.record(
            "rpc",
            Direction::Received,
            br#"{"id":1,"error":{"code":-32000,"message":"quota"}}"#,
        );
        assert!(probe.refused_turn_binds_model());
    }

    /// Two lookups of one name inside a single call can disagree: the
    /// adapter keeps a returned value in `values` for embedded
    /// substitution while its `absent` list still drops the whole-leaf
    /// member — `{"auth": "{secret:T}", "hdr": "Bearer {secret:T}"}`
    /// emits `{"hdr": "Bearer v"}` when the first lookup returned `v`
    /// and the second came back empty. Collapsing both facts into
    /// absent mis-reads the real frame's `hdr` as a literal token and
    /// lets the same-method prompt claim the call's frame.
    #[test]
    fn mixed_secret_resolution_drops_the_member_but_substitutes_the_leaf() {
        let probe = probe_for(
            a4aa::RpcLaneSpec {
                pre_turn: vec![a4aa::RpcCallSpec {
                    method: "session/prompt".to_owned(),
                    params: serde_json::json!({"auth": "{secret:AWP_TOKEN}", "hdr": "Bearer {secret:AWP_TOKEN}"}),
                }],
                prompt: a4aa::RpcCallSpec {
                    method: "session/prompt".to_owned(),
                    params: serde_json::json!({"payload": "{prompt}|{model}"}),
                },
                ..Default::default()
            },
            Some("prime"),
        );
        probe.reset("work", Some("fallback"), None);
        // The adapter's own resolutions in substitution order: the
        // `auth` lookup returned a value, the `hdr` lookup came back
        // empty — the member drops, the token still substitutes.
        {
            let mut secrets = probe
                .secrets
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            secrets.outcomes.push(("secret:AWP_TOKEN".to_owned(), true));
            secrets
                .outcomes
                .push(("secret:AWP_TOKEN".to_owned(), false));
        }
        probe.record(
            "rpc",
            Direction::Sent,
            br#"{"method":"session/prompt","id":1,"params":{"hdr":"Bearer v"}}"#,
        );
        assert!(!probe.dispatched());
        assert!(!probe.prompt_model_sent());
        probe.record(
            "rpc",
            Direction::Received,
            br#"{"id":1,"error":{"code":-32000,"message":"quota"}}"#,
        );
        // The refusal answers the model-free preparation call — never
        // the prompt's model verdict.
        assert!(!probe.refused_turn_binds_model());
    }

    /// A lane with no session call at all leaves `{sessionId}`
    /// known-absent — there is no open response to wait on, so the
    /// id-only pre-turn call resolves `{}` and skips like any other
    /// absent-leaf call.
    #[test]
    fn no_session_call_keeps_the_id_absent() {
        let probe = probe_for(
            a4aa::RpcLaneSpec {
                session_open: None,
                pre_turn: vec![
                    a4aa::RpcCallSpec {
                        method: "session/select".to_owned(),
                        params: serde_json::json!({"session": "{sessionId}"}),
                    },
                    a4aa::RpcCallSpec {
                        method: "session/select".to_owned(),
                        params: serde_json::json!({"session": "{model}"}),
                    },
                ],
                prompt: a4aa::RpcCallSpec {
                    method: "session/prompt".to_owned(),
                    params: serde_json::json!({"text": "{prompt}"}),
                },
                ..Default::default()
            },
            Some("prime"),
        );
        probe.reset("work", None, None);
        probe.record(
            "rpc",
            Direction::Sent,
            br#"{"method":"session/select","id":1,"params":{"session":"prime"}}"#,
        );
        assert!(probe.turn_model_sent());
        probe.record(
            "rpc",
            Direction::Received,
            br#"{"id":1,"error":{"code":-32000,"message":"quota"}}"#,
        );
        assert!(probe.refused_turn_binds_model());
    }

    /// A session call declared with default null params emits `null` —
    /// not `{}` — and still claims its own journal slot, so the open's
    /// answer settles the session id's presence. Normalizing null to
    /// `{}` would lose the binding and leave the id flexible: the
    /// id-only call then claims the model frame.
    #[test]
    fn null_params_open_binds_its_answer() {
        let probe = probe_for(
            a4aa::RpcLaneSpec {
                session_open: Some(a4aa::RpcCallSpec {
                    method: "session/open".to_owned(),
                    params: serde_json::Value::Null,
                }),
                pre_turn: vec![
                    a4aa::RpcCallSpec {
                        method: "session/select".to_owned(),
                        params: serde_json::json!({"session": "{sessionId}"}),
                    },
                    a4aa::RpcCallSpec {
                        method: "session/select".to_owned(),
                        params: serde_json::json!({"session": "{model}"}),
                    },
                ],
                prompt: a4aa::RpcCallSpec {
                    method: "session/prompt".to_owned(),
                    params: serde_json::json!({"text": "{prompt}"}),
                },
                ..Default::default()
            },
            Some("prime"),
        );
        probe.record(
            "rpc",
            Direction::Sent,
            br#"{"method":"session/open","id":0,"params":null}"#,
        );
        probe.record("rpc", Direction::Received, br#"{"id":0,"result":{}}"#);
        probe.reset("work", None, None);
        probe.record(
            "rpc",
            Direction::Sent,
            br#"{"method":"session/select","id":1,"params":{"session":"prime"}}"#,
        );
        assert!(probe.turn_model_sent());
        probe.record(
            "rpc",
            Direction::Received,
            br#"{"id":1,"error":{"code":-32000,"message":"quota"}}"#,
        );
        assert!(probe.refused_turn_binds_model());
    }
}
