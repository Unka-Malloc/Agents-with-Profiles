---
name: awp-sdk
description: Embed agents-with-profiles in a Rust host — registry resolution, materialized specs, session calls with profile semantics.
---

# awp SDK mode

Embed `agents-with-profiles`; resolve profiles to virtual agents in-process.

## When to use

- The host is Rust (any tokio runtime) and wants profile semantics —
  instruction composition, model/effort defaults, bindings — without a
  subprocess boundary.
- See `docs/integration.md` for the full contract; this file is the
  orientation card.

## Canonical calls

```rust
let registry = awp::Registry::from_json(config_text)?;
let agent = registry.resolve("reviewer")?;      // profile → VirtualAgent
let (events, mut rx) = tokio::sync::mpsc::unbounded_channel();
let session = agent.open_session(SessionRequest::default(), events).await?;
let outcome = session.prompt("task".to_owned()).await?;
```

- `resolve_binding(name)` — role slot; binding prompt folds into the
  profile's instructions.
- `prompt_with(TurnRequest)` — explicit model/effort overrides.
- `prompt_direct(TurnRequest)` — escape hatch; nothing composed.
- `agent.spec()` — the effective spec, identical to `awp resolve` output.

## Rules that keep the contract

- The registry operates on the document the host hands it — no fixed home,
  no ambient filesystem reads.
- Instruction composition belongs to this layer: compose
  `instructions + "\n\n" + text` when due, never pre-bake it into the spec.
