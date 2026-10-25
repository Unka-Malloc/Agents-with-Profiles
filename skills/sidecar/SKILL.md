---
name: awp-sidecar
description: Consume awp profiles from any language — `awp resolve` emits the effective spec, `a4aa serve` drives the sessions.
---

# awp Sidecar mode

`awp resolve` materializes the profile to a spec on stdout; the host feeds
that spec to an `a4aa serve` sidecar over `a4aa.v1`. Same semantics as the
embedded path — the composition rules just execute host-side.

## When to use

- The host is not Rust, or wants resolution as a build/config step
  decoupled from the session runtime.

## Invocation shape

```sh
awp resolve --config awp.json --profile reviewer    # spec JSON on stdout
awp validate --config awp.json                      # pre-flight checks
awp run --config awp.json --profile reviewer --prompt "..."  # one-shot smoke
```

Then drive `a4aa serve` per `docs/integration.md`:

1. `session/open` with the resolved `spec` + `session.cwd`; route
   `turn.model`/`turn.effort`/`turn.mode`/`turn.permission` through
   session options where the adapter supports them.
2. Compose `turn.instructions + "\n\n" + text` into `session/prompt`
   when instructions are due (cli lane, `every-turn` mode, or first prompt).
3. `session/cancel` is an id-bearing request, never a notification — the
   a4aa sidecar contract (`docs/sidecar-protocol.md` in the adapter repo)
   defines it only as a request and ignores the notification form with a
   stderr warning, so a host must send it with an id and await the result.

## Rules that keep the contract

- `resolve` is pure: it reads config files, never spawns agents — runtime
  routing (which selector carries which field) is the host's job against
  the target adapter's `capabilities()`.
- A4AA refuses unsupported model/effort selections — pass values only
  through selectors the session advertised.
