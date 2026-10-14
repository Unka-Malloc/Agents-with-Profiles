# The awp registry document

One JSON document carries everything a host needs to name and materialize
virtual agents. It is the shared interchange format: a config file, a
section of a larger config, or one synced bundle — the crate does not care
where it comes from.

```jsonc
{
  "agents": {
    // key = the spec's `id` (validated)
    "codex": {
      "id": "codex",
      "lane": "acp",                    // "acp" | "rpc" | "cli"
      "preferLoad": false,              // acp lane fields
      "mcpServers": [],
      "command": {
        "program": "codex",
        "args": ["acp"],
        "env": {"sanitized": {"inherit": [], "set": {}}}
      }
    }
  },
  "profiles": {
    // key = the profile id; value is a ProfilePatch — every field optional,
    // but a *new* profile must set `agent`
    "reviewer": {
      "agent": "codex",
      "name": "Reviewer",
      "revision": "3",
      "model": "m-large",
      "effort": "high",
      "instructions": "Review carefully.",
      "instructionMode": "first-turn",  // "first-turn" | "every-turn"
      "cwd": "/workspace/repo",
      "env": {"FLAG": "1", "TOKEN": {"secret": {"secret": "vault/ref"}}},
      "mcpServers": [{"name": "tools", "command": "tool-bin"}],
      "permission": "prompt",
      "mode": "review",
      "contextWindow": 200000,
      "requiredCapabilities": ["fs:read"],
      "preferredCapabilities": ["fs:write"],
      "skillReferences": ["skill/review"],
      "stateKey": "reviewer-state",
      "extensions": {"avatar": "owl"}
    }
  },
  "bindings": {
    // full form — profile plus a standing prompt
    "review": {"profile": "reviewer", "prompt": "Check the diff."},
    // shorthand — a bare profile id
    "implement": "implementer"
  }
}
```

`agents` values are a4aa `AgentSpec`s verbatim (see the a4aa model docs).
`bindings` name which profile serves a role or task slot. Unknown fields
are rejected (`deny_unknown_fields`) — a typo is a load error, not a silent
default.

## Profile fields: materialized vs advisory

| Field | Class | Effect |
| --- | --- | --- |
| `agent` (required) | identity | Registry key of the backing `AgentSpec`. |
| `id`, `name`, `revision` | identity | `revision` is an explicit operator-maintained version string — never inferred from content. `principal_id()` returns `profile:<id>`. |
| `instructions`, `instructionMode` | materialized | Composed into prompt text (`instructions\n\n<text>`). `first-turn` sends once per session-bound lane (`acp`, `rpc`); `every-turn` always. Stateless `cli` lanes deliver on every turn — each turn is a fresh process. |
| `model`, `effort` | materialized | Per-turn defaults for lanes that accept them (`rpc` params, `cli` `{model}`/`{effort}` argv placeholders). Explicit turn values win. The `acp` lane carries no such field — the values still appear in `resolve` output for the host to route. |
| `cwd` | materialized | Default session working directory; an explicit `SessionRequest.cwd` wins. |
| `env` | materialized | Merged per key into the spec's environment policy `set` — literals and `{"secret": …}` / `{"hostEnv": …}` indirections resolve at spawn like any spec entry. |
| `mcpServers` | materialized | Appended to the `acp` lane's `mcpServers` after the spec's own. Ignored on lanes without an MCP surface. |
| `permission`, `mode`, `contextWindow`, `requiredCapabilities`, `preferredCapabilities`, `skillReferences`, `stateKey` | advisory | Carried verbatim for the host's dispatch, selection or permission policy. The crate never invents a delivery path for them. |
| `extensions` | advisory | Host-specific extras (avatars, sync manifests, UI hints); merged per key on overlay, never interpreted. |

## Layering

`Registry::apply(document)` (and `--overlay` on the CLI) overlays one
document on the registry — the scoped-override pattern without scope
vocabulary: a host stacks base → organization → project, or however many
layers it needs.

- `agents`: replace per key (overlays carry complete specs).
- `profiles`: merge field-wise — `Some` replaces, `None` keeps; `env` and
  `extensions` merge per key; list fields replace wholesale.
- `bindings`: replace per key; a vacant binding removes the slot.

Validation is atomic and fails closed: a document that mismatches an agent
id, leaves a new profile without `agent`, or points a binding at an unknown
profile applies nothing at all.

## Environment entry wire form

`env` values reuse a4aa's `EnvValue` untagged form:

```jsonc
"env": {
  "LITERAL":   "value",
  "FROM_HOST": {"hostEnv": {"host_env": "HOST_VAR"}},
  "FROM_VAULT": {"secret": {"secret": "vault/ref"}}
}
```

The nested shape is a4aa's externally-tagged enum contract — variant names
are camelCase, fields keep their declared names — reproduced exactly so a
profile document and an `AgentSpec` accept identical entries. Unresolved
secrets fail the spawn closed; literals and host lookups never reach a
secret store.
