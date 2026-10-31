# Integration modes

Two ways to consume agents-with-profiles, one semantics. A Rust host
embeds the crate and resolves profiles to virtual agents in-process; a
host in any other language runs the `awp` document tool to emit the same
effective spec, then drives sessions through the `a4aa serve` sidecar.
Both paths share the registry merge rules, the instruction composition
and the turn defaults — the full contract lives in
[integration.md](integration.md).

## SDK — embed the crate

```toml
[dependencies]
agents-with-profiles = "0.2"
```

```rust
use awp::{Registry, SessionRequest};
use tokio::sync::mpsc;

let registry = Registry::from_json(config_text)?;   // the host hands over the document
let agent = registry.resolve("reviewer")?;          // profile → VirtualAgent
let (events, mut rx) = mpsc::unbounded_channel();
let session = agent.open_session(SessionRequest::default(), events).await?;
let outcome = session.prompt("task".to_owned()).await?;
```

Library ownership: `awp` carries the registry document (parse, validate,
layered overlays, role bindings), profile resolution and the
virtual-session facade over `a4aa`. The host's tokio runtime drives
everything — there is no daemon, no fixed home and no ambient filesystem
read; the registry operates on the document the host hands it.

## Wire mode — `awp` + `a4aa serve`

```sh
awp validate --config awp.json [--overlay layer.json]...
awp resolve  --config awp.json --profile reviewer   # effective spec as JSON
awp run      --config awp.json --profile reviewer --prompt "..."  # one shot
```

`resolve` is pure: it reads the registry document and prints the
effective spec, the turn plan and the mechanism set as JSON — it never
spawns an agent. The host feeds that spec to `a4aa serve` (`a4aa.v1` over
NDJSON) and applies the composition rules itself: instructions fold into
prompt text when due, model/effort values travel only through selectors
the session advertised. `awp run` executes one profiled turn in-process —
the smoke check for a document before it is wired into a service. `awp`
itself has no serve mode; the sidecar boundary belongs to `a4aa`.

## Machine discovery

`awp::detect_agents()` probes this machine against every known agent
card — `PATH` binaries, install directories outside `PATH`, configuration
files, prior-use markers and OS-gated application bundles — and returns
one `Detection` per card: the resolved executable, the marker paths that
exist, and the binary-name set a host joins against `command.program`
basenames. What exists on a machine is a runtime fact, never an
install-time assumption; a host pairs each detection with
`awp::lookup_preset` to offer one-step registration for agents already
installed. Discovery lives behind the `dialects` feature (on by default)
beside the card registry.

## Skills

- `skills/sdk/` — embedding guidance (registry, resolution, session calls).
- `skills/sidecar/` — wire-mode guidance (`awp resolve` + `a4aa serve`).
