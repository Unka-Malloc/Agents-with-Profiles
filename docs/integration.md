# Integrating awp

Two consumption paths, one semantics: Rust hosts embed the library; hosts
in any other language resolve profiles through the `awp` binary and drive
sessions through the `a4aa` sidecar. Both end up with the same effective
spec, the same instruction composition and the same turn defaults.

## Rust: embed the crate

```toml
[dependencies]
agents-with-profiles = "0.1"
```

```rust
use awp::{Registry, SessionRequest};
use tokio::sync::mpsc;

let registry = Registry::from_json(config_text)?;
let agent = registry.resolve("reviewer")?          // profile → VirtualAgent
    .secrets(host_secret_resolver)                 // optional a4aa hooks
    .permission(permission_handler)
    .before_turn(durable_reservation_hook);

let (events, mut rx) = mpsc::unbounded_channel();
let session = agent
    .open_session(SessionRequest::default(), events)
    .await?;

// Profile semantics apply per turn: instructions compose into the text,
// model/effort defaults use the phase supported by the adapter.
let outcome = session.prompt("task".to_owned()).await?;
session.close().await?;
```

- `registry.resolve(profile_id)` — one profile to one virtual agent.
- `registry.resolve_binding(name)` — a role/task slot: the binding's
  standing prompt folds into the profile's instructions
  (`profile instructions + "\n\n" + binding prompt`).
- `session.prompt_with(TurnRequest)` — explicit model/effort override the
  profile defaults.
- `session.prompt_direct(TurnRequest)` — the escape hatch: nothing is
  composed or filled. For hosts that deliver profile context through their
  own channel (e.g. installed skill files).
- `agent.spec()` — the effective spec after materialization; the same value
  `awp resolve` emits.

## Other languages: `awp resolve` + `a4aa serve`

```sh
awp resolve --config awp.json [--overlay layer.json]... --profile reviewer
```

```jsonc
{
  "profile": { "id": "reviewer", "agent": "example-agent", /* …verbatim… */ },
  "lane": "acp",
  "spec":    { /* effective a4aa AgentSpec — profile env + mcpServers folded in */ },
  "alternates": [ /* ordered lane fallbacks after the primary spec, same shape */ ],
  "session": { "cwd": "/workspace/repo" },
  "turn": {
    "model": "m-large",
    "effort": "high",
    "instructions": "Review carefully.\n\nCheck the diff.",
    "instructionMode": "first-turn"
  },
  "mechanisms": { /* semantic CAwP declarations, or null */ }
}
```

Host-side flow against `a4aa serve` (`a4aa.v1`, see the a4aa sidecar doc):

1. `session/open` with `{spec, cwd: session.cwd?, env: {}, options}`.
   Route `turn.mode` and `turn.permission` to session options. Route model
   and effort defaults to session options when the selected adapter supports
   them at session establishment (including negotiated ACP configuration),
   otherwise use its turn selectors. `resolve` returns profile data; it does
   not perform this runtime routing for a foreign-language host.
2. Every `session/prompt`: compose
   `text = turn.instructions ? turn.instructions + "\n\n" + text : text`
   **when instructions are due**, and pass model/effort values only through
   supported turn selectors. A4AA rejects unsupported selections.
   Instructions are due when `lane == "cli"`, or
   `turn.instructionMode == "every-turn"`, or it is the first prompt of the
   session.
3. `session/cancel` / `session/close` as usual.

`awp validate` checks the same load + layering rules ahead of time, and
`awp run` executes one profiled turn end-to-end in-process — a fast smoke
check for a document before wiring it into a service.

## Mechanism materialization (CAwP)

A profile's `mechanisms` section declares hooks, rules, skills, commands,
subagents, plugins, permissions, config-file MCP servers and native-profile
selection in semantic, dialect-free form. Adaptation lives in the
`a4aa-<agent>` crates — each agent's descriptor is its own installable
package, so a host picks exactly one:

```toml
[dependencies]
agents-with-profiles = "0.1"
a4aa-pi = "0.2"   # the only adapter compiled in
```

```rust
// AwP parses and resolves; the adapter crate owns the agent's dialect.
let mechanisms = agent.profile().mechanisms.clone().unwrap_or_default();
let plan = a4aa_pi::adapt(&mechanisms); // or a4aa::adapt(a4aa_pi::dialect(), &set)
```

Multi-agent tooling can instead depend on `a4aa-dialects` (a name→`Dialect`
registry over all adapter crates) — that is what `awp materialize` does:

```sh
awp materialize --config awp.json --profile reviewer --dialect codex
```

The `materialize` subcommand is behind the `dialects` cargo feature, on by
default. Building `agents-with-profiles` with `default-features = false`
drops `a4aa-dialects` entirely — the library itself needs no adapter crate.

```jsonc
{
  "dialect": "codex",
  "artifacts": [
    {"scope": "user-home", "path": ".codex/hooks.json",
     "content": "{…}", "mode": {"kind": "json-merge", "at": "hooks"}},
    {"scope": "project",   "path": "AGENTS.md",
     "content": "Never guess.\n", "mode": {"kind": "managed-block", "name": "core"}},
    {"scope": "user-home", "path": ".codex/agents/verifier.toml",
     "content": "…", "mode": {"kind": "create"}}
  ],
  "report": [
    {"item": "hooks.session-start", "support": "native"},
    {"item": "permissions", "support": "unsupported",
     "note": "no declarative permission surface for this dialect"}
  ]
}
```

The plan is data only — the host decides the scope roots (`user-home`,
`project`) and whether to write. `json-merge` artifacts deep-merge
`content` (a JSON object) into the target file's document at `at`;
`managed-block` content belongs inside
`<!-- awp-managed:<name> -->` … `<!-- /awp-managed:<name> -->` markers in
the target file; `create` writes verbatim. Items the agent cannot express
come back `unsupported` with a note rather than failing the plan — the
host decides whether a missing mechanism is fatal.

## Typical shapes

- **Chat host** — one profile per persona the user talks to. `name`,
  `extensions.avatar`, `contextWindow` and
  the capability lists are advisory metadata the host's member/dispatch
  model consumes; `principal_id()` gives each profile a stable
  `profile:<id>` identity.
- **Automation service** — a registry of agent specs plus role bindings
  (`designer`, `worker`, `reviewer`) with layered overlays per
  organization/project scope; resolve the binding, send the role prompt as
  instructions, deliver the task as turn text.
- **Publication gate** — each checker is a synced profile: `revision`
  pins the published profile version, `env`/`extensions` carry whatever the
  sync manifest requires, and `prompt_direct` leaves prompt delivery to the
  checker's own harness.

## Skills

- `skills/sdk/` — embedding guidance (registry, resolution, session calls).
- `skills/sidecar/` — wire-mode guidance (`awp resolve` + `a4aa serve`).
