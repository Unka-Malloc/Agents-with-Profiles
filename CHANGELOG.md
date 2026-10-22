# Changelog

## Unreleased

- Optional profile `stateScope`: absent means fully parallel sessions;
  `project` serializes within one project; `global` single-flights the
  profile. Carried as advisory for hosts; omitted from the document when
  unset.
- Canonical headless-agent presets (`codex`, `cursor`, `dsh`, `hermes`,
  `opencode`, `pi`): verified program, spawn argv, and required env
  resolution as structs. `dsh` launches `dsh acp`; leftover `--profile sdk`
  is not a preset.
- `availability` on binding candidates: `notBefore`/`notAfter` bounds (RFC
  3339 or `YYYY-MM-DD` at 00:00 UTC) make time-limited lanes — vendor promo
  windows — expire semantically at resolution instead of failing per
  request or needing scheduled removal. Bounds live on the binding's
  primary and on full-form fallback entries; absent means unbounded and a
  malformed bound rejects the document at load.

## 0.1.0

Initial registry release, built on A4AA 0.2.0.

- Define named, revisioned profiles and role bindings over A4AA agent specs.
- Resolve layered registry documents atomically and reject unknown references.
- Compose profile instructions and binding context into virtual agents.
- Apply model and reasoning defaults at the session or turn phase supported
  by the adapter. Explicit session configuration takes precedence over profile
  defaults; unsupported selections fail before paid turn work starts.
- Apply mode and permission presets through A4AA session configuration.
- Keep first-turn guidance pending when an initial attempt is cancelled,
  including cancellation before the prompt is dispatched.
- Provide the `awp` CLI for validation, resolution, profiled execution, and
  adapter-specific mechanism plans. Library consumers can disable the default
  `dialects` feature to include only their chosen adapter.
- Serialize profile composition with turn submission so concurrent first turns
  do not duplicate standing guidance.
- Retain turn and guidance ownership when a caller drops its waiting future;
  shared session references can still close active work.
