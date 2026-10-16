# The awp registry document

One JSON document carries everything a host needs to name and materialize
virtual agents. It is the shared interchange format: a config file, a
section of a larger config, or one synced bundle — the crate does not care
where it comes from.

```jsonc
{
  "agents": {
    // key = the spec's `id` (validated)
    "example-agent": {
      "id": "example-agent",
      "lane": "acp",                    // "acp" | "rpc" | "cli"
      "preferLoad": false,              // acp lane fields
      "mcpServers": [],
      "command": {
        "program": "agent-cli",
        "args": ["acp"],
        "env": {"sanitized": {"inherit": [], "set": {}}}
      }
    }
  },
  "profiles": {
    // key = the profile id; value is a ProfilePatch — every field optional,
    // but a *new* profile must set `agent`
    "reviewer": {
      "agent": "example-agent",
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
The example uses a hypothetical ACP executable. For a supported real agent,
use its `a4aa-<agent>::spec()` descriptor and supported selector values.
`bindings` name which profile serves a role or task slot. Unknown fields
are rejected (`deny_unknown_fields`) — a typo is a load error, not a silent
default.

## Profile fields: materialized vs advisory

| Field | Class | Effect |
| --- | --- | --- |
| `agent` (required) | identity | Registry key of the backing `AgentSpec`. |
| `id`, `name`, `revision` | identity | `revision` is an explicit operator-maintained version string — never inferred from content. `principal_id()` returns `profile:<id>`. |
| `instructions`, `instructionMode` | materialized | Composed into prompt text (`instructions\n\n<text>`). `first-turn` sends once per session-bound lane (`acp`, `rpc`); `every-turn` always. Stateless `cli` lanes deliver on every turn — each turn is a fresh process. |
| `model`, `effort` | materialized | Defaults applied at the session or turn phase supported by the A4AA adapter. Explicit session configuration wins over profile defaults. Explicit turn selectors are honored where the lane supports them; unsupported selections fail before the turn begins. ACP values use negotiated configuration. |
| `cwd` | materialized | Default session working directory; an explicit `SessionRequest.cwd` wins. |
| `env` | materialized | Merged per key into the spec's environment policy `set` — literals and `{"secret": …}` / `{"hostEnv": …}` indirections resolve at spawn like any spec entry. |
| `mcpServers` | materialized | Appended to the `acp` lane's `mcpServers` after the spec's own. Ignored on lanes without an MCP surface. |
| `permission`, `mode` | materialized | Session defaults, applied through A4AA's supported selectors. Explicit `SessionRequest.config` values win. An unsupported preset is an error. |
| `contextWindow`, `requiredCapabilities`, `preferredCapabilities`, `skillReferences`, `stateKey` | advisory | Carried verbatim for the host's dispatch, selection or state policy. The crate never invents a delivery path for them. |
| `mechanisms` | CAwP | Semantic mechanism declarations — hooks, rules, skills, commands, subagents, plugins, permissions, config-file MCP servers, native-profile selection. awp parses, validates and merges them; `a4aa`'s adaptation layer translates them into agent-native artifacts. See "The `mechanisms` section" below. |
| `extensions` | advisory | Host-specific extras (avatars, sync manifests, UI hints); merged per key on overlay, never interpreted. |

First-turn guidance remains pending after a failed or cancelled initial attempt,
because the agent may not have received that prompt. Once a non-cancelled
response confirms submission, later turns do not repeat first-turn guidance.
Concurrent prompt calls are serialized before composition, so queued turns use
the preceding turn’s confirmed guidance state.
The session owns submission and guidance bookkeeping until the turn settles;
dropping a caller's waiting future does not repeat first-turn guidance or admit
an overlapping turn. `close` is callable through a shared session reference.

## The `mechanisms` section (CAwP)

Where the materialized fields above describe *what a session carries*, the
`mechanisms` section describes *what the agent itself should be configured
to do* — lifecycle hooks, standing rules, skills, commands, subagents,
plugins, permission posture, config-file MCP servers, or selecting the
agent's own native profile. Declarations are **semantic and dialect-free**:
no agent-specific schema ever enters a profile. `a4aa` owns the mapping
onto each agent's native surface.

```jsonc
"mechanisms": {
  "hooks": [{
    "event": "pre-tool-use",      // semantic event set, kebab-case
    "command": "guard.sh",        // shell command the agent's hook runner executes
    "intent": "enforce",          // "enforce" | "inject" | "rewrite" | "notify"
    "matcher": "Bash",            // tool filter where the dialect supports it
    "timeout": 10
  }],
  "rules":     [{"name": "core", "content": "Never guess."}],
  "skills":    [{"name": "audit", "content": "---\nname: audit\n---\nAudit."}],
  "commands":  [{"name": "ship", "description": "Ship", "content": "Ship it."}],
  "subagents": [{"name": "verifier", "model": "m", "effort": "high",
                 "sandbox": "workspace-write", "instructions": "Verify."}],
  "plugins":   [{"id": "linter", "source": "marketplace/linter", "enabled": true}],
  "permissions": {"mode": "workspace-write", "allow": ["Shell(ls)"], "deny": []},
  "mcpServers": {"fs": {"command": "fs-mcp"}},   // agent config-file MCP, not the session lane
  "nativeProfile": "reviewer"                    // use the agent's own profile mechanism
}
```

- `hooks.<event>` — semantic events: `session-start`, `session-end`,
  `user-prompt-submit`, `pre-tool-use`, `post-tool-use`, `stop`,
  `subagent-start`, `subagent-stop`, `permission-request`, `notification`.
  `intent` selects the hook's purpose; the adapter reports `adapted` where
  a dialect cannot express it.
- `mcpServers` here targets the agent's **config file** (`.cursor/mcp.json`,
  `.gemini/settings.json`, project `.mcp.json`, `opencode.json`) — distinct
  from the profile-level `mcpServers` list, which feeds the `acp` session
  lane.
- `nativeProfile` emits no artifacts; it records that the host should select
  the agent's own profile (`codex --profile`, a DSH profile name).

Merging on overlay is keyed: `rules`/`skills`/`commands`/`subagents`/`plugins`
merge by name/id (later wins), `hooks` dedupe by `(event, command, matcher)`,
`permissions` and `nativeProfile` replace. Validation is structural only
(non-empty names/bodies, no duplicates); whether a declaration is
**supported** is a per-dialect question the adapter answers — `awp
materialize --dialect <agent>` emits the artifact plan plus the
native/adapted/unsupported report without writing anything.

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
