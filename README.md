# agents-with-profiles — `awp`

Agents with Profiles: the layer that turns one *base agent* into many
*virtual agents*. A base agent ([`a4aa`](https://crates.io/crates/a4aa)
`AgentSpec`) says **how to reach the agent** — which lane it speaks and the
command to spawn. A `Profile` says **how the agent should behave** — model,
reasoning effort, persona instructions, working directory, environment, MCP
servers and dispatch preferences. One base agent plus one profile
materializes a [`VirtualAgent`]: a reviewer and an implementer can share a
single underlying agent and still behave as two distinct members of a
workflow.

Rust hosts embed the library (`use awp::…`). Hosts in any language run the
`awp` binary: `awp resolve` emits the effective spec and turn plan as JSON,
which can drive `a4aa serve`. Registry merge rules are resolved once; a
foreign-language host still routes session and turn selectors as described
in the integration guide. `awp run` applies that runtime behavior itself.

## What it provides

| Piece | Type | Responsibility |
| --- | --- | --- |
| Profile | `Profile` / `ProfilePatch` | Named, revisioned behavior record; the patch form is all-optional for documents and overlays |
| Registry | `Registry` / `RegistryDocument` | Validated store of agent specs, profile patches and role bindings; atomic layered overlays (`apply`) |
| Binding | `Binding` | A role/task slot → profile + standing prompt, composed after the profile's own instructions |
| Virtual agent | `VirtualAgent` / `VirtualSession` | Materialization: profile env and MCP servers fold into the spec; sessions compose instructions into prompt text and fill model/effort defaults |
| CLI | `src/bin/awp.rs` | `inspect`, `validate`, `resolve`, `run`, `materialize`, `fixture-agent` |

## Invariants

- **Zero-assumption output** — profiles compose into *input* text only; the
  agent's output stays untouched (inherited from a4aa).
- **Fail closed** — unknown profiles, agents, bindings and missing `agent`
  references are configuration errors, never runtime guesses.
- **Atomic layering** — a registry overlay validates fully or not at all; a
  failed `apply` leaves the previous registry intact.
- **Explicit cancellation only** — `VirtualSession::cancel` is the sole
  cancel path.

## Rust usage

```rust
use awp::{Registry, SessionRequest, TurnRequest};
use tokio::sync::mpsc;

let registry = Registry::from_json(r#"{
  "agents": {
    "fixture": {"id": "fixture", "lane": "acp",
                "command": {"program": "agent-cli", "args": ["acp"]}}
  },
  "profiles": {
    "reviewer": {"agent": "fixture", "model": "m-large",
                 "instructions": "Review carefully."}
  },
  "bindings": {"review": {"profile": "reviewer", "prompt": "Check the diff."}}
}"#)?;

let agent = registry.resolve("reviewer")?;        // or resolve_binding("review")
let (events, mut rx) = mpsc::unbounded_channel();
let session = agent.open_session(SessionRequest::default(), events).await?;
let outcome = session.prompt("src/main.rs".to_owned()).await?;
session.close().await?;
```

## CLI usage

```sh
awp inspect   --config awp.json                       # resolved registry as JSON
awp validate  --config awp.json --overlay prod.json   # load + validate layers
awp resolve   --config awp.json --profile reviewer    # materialization as JSON
awp run       --config awp.json --binding review --prompt "Check the diff."
awp materialize --config awp.json --profile reviewer --dialect codex
                                                  # CAwP adaptation plan as JSON
awp fixture-agent                                     # deterministic ACP echo agent
```

For a foreign-language host the flow is: `awp resolve` → feed `spec` to
`a4aa serve` `session/open` → route selectors to their supported phase and,
per `session/prompt`, compose `turn.instructions` into the text per
`instructionMode`. See
[docs/integration.md](docs/integration.md).

## Documentation

Entry point: [docs/README.md](docs/README.md) — the
[profile format](docs/profile-format.md) (including the CAwP `mechanisms`
section), [integration guide](docs/integration.md), and the
[agent capability survey](docs/agent-capabilities.md) behind the
adaptation dialects.

## Verify

```sh
scripts/verify.sh   # fmt + clippy + cargo test --all-targets
```

## Status

The package is `agents-with-profiles`; the library and binary are `awp`
(Apache-2.0). It depends on `a4aa` — publish `a4aa` first when publishing
to crates.io.
