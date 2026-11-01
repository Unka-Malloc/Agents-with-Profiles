# awp documentation

- [profile-format.md](profile-format.md) — the registry document schema:
  agent specs, profile fields (materialized vs advisory), bindings, and the
  layered merge rules.
- [integration-modes.md](integration-modes.md) — machine discovery: probing
  this machine for installed agents through the card `detect` surface.
- [integration.md](integration.md) — the consumption contract: embedding
  the crate in Rust and driving profiles from hosts in other languages
  through `awp resolve` + `a4aa serve`.
- [agent-capabilities.md](agent-capabilities.md) — survey of each agent's
  native mechanisms (hook dialects, plugins, subagents, rules, skills, MCP,
  permissions) for the CAwP layer that maps profile declarations onto them.
