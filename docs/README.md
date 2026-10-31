# awp documentation

- [profile-format.md](profile-format.md) — the registry document schema:
  agent specs, profile fields (materialized vs advisory), bindings, and the
  layered merge rules.
- [integration-modes.md](integration-modes.md) — the two consumption paths
  (embed the crate vs `awp` document tool + `a4aa serve`) and the
  machine-discovery probe.
- [integration.md](integration.md) — embedding the crate in Rust and driving
  profiles from hosts in other languages through `awp resolve` +
  `a4aa serve`.
- [agent-capabilities.md](agent-capabilities.md) — survey of each agent's
  native mechanisms (hook dialects, plugins, subagents, rules, skills, MCP,
  permissions) for the CAwP layer that maps profile declarations onto them.
