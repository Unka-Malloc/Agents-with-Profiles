# Changelog

## Unreleased

- `Registry::select` honors the same gates as `resolve_chain`: candidates
  gated by availability windows or retirement no longer answer, a strict
  binding skips candidates that cannot seal their session surface, and an
  empty chain fails with `binding_chain_expired` instead of returning the
  bound profile unchecked.
- Malformed availability `windows` — a `days` value outside `*` or the
  `mon`–`sun` vocabulary, an unresolvable `timezone`, or a malformed
  `HH:MM` bound — reject the document at load (`binding_availability_malformed`)
  instead of silently never covering. The same rule binds `windows` on
  profiles: malformed model windows reject at load
  (`profile_windows_malformed`), including profiles built through the
  public `Profile::from_patch` path.
- Release train: `publish.yml` no longer reads `mode`/`rehearse` from the
  verify artifact (candidate code could rewrite it); live publishing
  requires a `publish` dispatch with `mode: live`, the candidate must be
  the verify run's own head, the version decision is recomputed by the
  default branch's train helper inside the OIDC job and compared with the
  bound run, and the release job waits on the `release` environment. The
  `workflow_run` auto-publish path is removed — crates.io Trusted
  Publishing refuses that event, so it could only ever no-op silently.

## 0.2.0

Built on A4AA 0.4.x (developed against 0.4.0; published lockfile resolves 0.4.1).

- Optional profile `stateScope`: absent means fully parallel sessions;
  `project` serializes within one project; `global` single-flights the
  profile. Carried as advisory for hosts; omitted from the document when
  unset.
- Canonical headless-agent presets (`codex`, `cursor`, `dsh`, `hermes`,
  `opencode`, `pi`): verified program, spawn argv, and required env
  resolution as structs. Every preset materializes the card's full
  capability-ordered lane chain; `dsh` is SDK-primary (`dsh --profile sdk`)
  with the ACP profile as its ordered alternate.
- `availability` on binding candidates: `notBefore`/`notAfter` bounds (RFC
  3339 or `YYYY-MM-DD` at 00:00 UTC) make time-limited lanes — vendor promo
  windows — expire semantically at resolution instead of failing per
  request or needing scheduled removal. Bounds live on the binding's
  primary and on full-form fallback entries; absent means unbounded and a
  malformed bound rejects the document at load.
- Strict bindings: `strict` on a binding seals the resolved session surface
  at resolution time; ambient-surface argv (`--skill`, `--prompt-template`,
  `-e`/`--extension`, including equals-form) a sealed agent cannot honor
  refuses the binding instead of silently widening the session.
- Registry-level `secrets` recipes: named references resolve credential
  values from the host environment at spawn; documents never carry the
  secret itself.
- Environment fidelity: specs resolve the user's real environment — an
  agent never launches on an empty or minimal environment.
- Quota refusals route through the profile's model chain: an
  `agent_quota_exhausted` on A4AA 0.3.0+ advances the serving profile's
  model list before the binding falls through to its next candidate.
- Release train: the two-file verify/publish split lands — `verify.yml`
  certifies a candidate under a read-only token and records the release
  inputs as a data artifact; `publish.yml` holds the OIDC job and replays
  only the bound result.

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
