# Agent capability survey for CAwP

Purpose: CAwP (Configurable Agents with Profiles) is the layer above AwP that
parses a declarative profile into semantic mechanism declarations — hooks,
plugins, subagents, rules, skills, MCP, permissions — instead of only composing
prompts. The boundary: **AwP owns parsing; a4aa owns adaptation.** AwP never
emits agent-native artifacts itself; it hands normalized declarations to the
a4aa adapter, which owns each agent's capability matrix, config dialect, and
materialization. This document surveys which mechanisms each installed agent
actually exposes — i.e. the surface a4aa's adaptation layer must cover.

Method: primary evidence from the agents' own config artifacts and shipped
documentation (`~/.<agent>` directories, per-agent user guides). Where an
agent's surface is only partially observable (IDE-internal state), the table
says so explicitly rather than extrapolating.

## 1. Lifecycle hooks — the CAwP centerpiece

Hooks are the strongest mechanism class: they let a profile *enforce* behavior
(block a tool call, inject context, rewrite a result) rather than merely
*describe* it. Five schema dialects exist:

| Agent | Hook store | Schema dialect | Observed events |
| --- | --- | --- | --- |
| Claude Code | `~/.claude/settings.json` `hooks` | Claude dialect | SessionStart, UserPromptSubmit, PreToolUse, PermissionRequest, PostToolUse, PostToolUseFailure, Notification, Stop, StopFailure, SubagentStart, SubagentStop, PreCompact, PostCompact, TeammateIdle, SessionEnd |
| Codex | `~/.codex/hooks.json` | Claude dialect (+ `statusMessage`, `additionalContextLimit`) | SessionStart, UserPromptSubmit, PreToolUse, PermissionRequest, PostToolUse, Stop, SubagentStart, SubagentStop |
| Devin | `~/.config/devin/config.json` `hooks` | Claude dialect | SessionStart, SessionEnd, UserPromptSubmit, PreToolUse, PostToolUse, Stop, PermissionRequest, PostCompaction |
| Command Code | `~/.commandcode/settings.json` `hooks` | Claude dialect | SessionStart, PreToolUse, PostToolUse, Stop |
| Grok | `~/.grok/hooks/*.json` (one bundle file per hook owner) | Claude dialect + cancelled/failure variants | SessionStart, SessionEnd, UserPromptSubmit, PreToolUse, PostToolUse, PostToolUseFailure, Notification, Stop, StopCancelled, StopFailure |
| Cursor agent | `~/.cursor/hooks.json` | Flat: `{version, hooks:{event:[{command}]}}`; stdin JSON, stdout `{continue:...}` reply | beforeSubmitPrompt and Cursor's agent-hook event set |
| Gemini CLI / Antigravity | `~/.gemini/config/hooks.json` | Named bundles: `{name:{Event:[{command,timeout,type}],enabled}}` | Stop, PostInvocation, PostToolUse, plus Gemini's SessionStart/BeforeTool set |
| Kiro | `.kiro/hooks/*.kiro.hook` (IDE) | Declarative trigger spec | onSave / onMessageSubmit / onStop-style triggers |
| OpenCode | `~/.config/opencode/plugins/*.js` | JS plugin event exports | session/tool/file events handled in plugin code |
| Pi | `~/.pi/agent/extensions/*.ts` | TS extension API: `pi.on('session_start', …)`, `pi.registerTool`, `ctx.ui.*` | session_start, tool lifecycle, UI events |
| Hermes | `~/.hermes/plugins/*/plugin.yaml` `provides_hooks` | Plugin manifest + platform adapters | event-bus style (platform events, cron, wake word) |
| DSH | `profiles/*/cordis.patch.yml` | Component loader DSL | interception via inserted/disabled components rather than named events |

### Hook schema dialects

```
A. Claude dialect      {"hooks":{"Event":[{"matcher":<regex>,
                       "hooks":[{"type":"command","command":<str>,
                                 "timeout":<s>,"statusMessage":<str>,
                                 "additionalContextLimit":<chars>}]}]}}
   Users: Claude Code, Codex, Devin, Command Code, Grok.

B. Named-bundle        {"<bundle>":{"Event":[{"command":<str>,
                       "timeout":<s>,"type":"command"}],
                       "enabled":<bool>}}
   Users: Gemini CLI, Antigravity (~/.gemini/config/hooks.json).

C. Flat                {"version":1,"hooks":{"<event>":[{"command":<str>}]}}
   Users: Cursor agent.

D. Code extension      installed code units, not config:
   Pi  -> ~/.pi/agent/extensions/*.ts    (pi.on(event), registerTool, ctx.ui)
   OpenCode -> ~/.config/opencode/plugins/*.js  (event handler exports)
   DSH -> cordis.patch.yml               (insert/disable/config components,
                                          `!!js` expressions allowed)

E. Manifest            trigger spec inside a package manifest:
   Hermes -> plugin.yaml provides_hooks; Kiro -> .kiro.hook specs.
```

Hook result semantics vary: blocking (deny a tool call), context injection
(`additionalContextLimit`-capped stdout into the model context), result
rewriting (PostToolUse can replace what the model sees), or notify-only. A CAwP
hook declaration therefore needs an `intent` field (`enforce | inject |
rewrite | notify`), not just an event name.

## 2. Other deep mechanisms

| Agent | Rules / memory | Skills | Commands | Subagents | Plugins | MCP | Permissions / sandbox | Native profiles | Headless / session |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Codex | `AGENTS.md` | `~/.codex/skills` + vendor imports | `~/.codex/prompts/*.md` | `~/.codex/agents/*.toml` (`model`, `model_reasoning_effort`, `sandbox_mode`, `developer_instructions`) | `~/.codex/plugins`, `config.toml [plugins.*]` | `config.toml [mcp_servers.*]` | `sandbox_mode`, `approval_policy`, `sandbox_workspace_write` | `config.toml [profiles.*]` | `codex app-server` (JSON-RPC), `codex exec --json` |
| Claude Code | `CLAUDE.md` | `~/.claude/skills` | `commands/*.md` + plugin commands | `~/.claude/agents/*.md` | `~/.claude/plugins` + marketplaces | `.mcp.json` / settings | `permissions` allow/deny/ask + modes | output styles | `claude -p --output-format stream-json`, ACP adapter |
| Cursor agent | `.cursor/rules/*.mdc` | `~/.cursor/skills`, `skills-cursor` | custom commands | `~/.cursor/agents/*.md` | `~/.cursor/plugins` | `~/.cursor/mcp.json` | `cli-config.json` `permissions` (Shell patterns), `sandbox-policies/` | — | `cursor-agent`, ACP (`acp-sessions/`) |
| Devin | `AGENTS.md` (project + `~/.config/devin/AGENTS.md`) | `.devin/skills`, `~/.config/devin/skills` | skill packs (`<skills>/<name>/SKILL.md`, `triggers: [user]`) | `.devin/agents/*.md`, `~/.config/devin/agents/*.md` (name/model/allowed-tools frontmatter) | `devin plugins` (imperative, host-managed) | `mcp_config.json` | `config.json` `permissions` allow/deny/ask + `--permission-mode` auto/accept-edits/smart/dangerous | — | `devin acp` (ACP over stdio; host authenticates via `_meta.api_key`), `devin -p` batch |
| Gemini CLI (unmaintained; shared `~/.gemini` surface adapted by `a4aa-antigravity`) | `GEMINI.md` | extension skills | `~/.gemini/commands/*.toml` | — | `~/.gemini/extensions`, `config/plugins` | `settings.json` `mcpServers` | `approvalMode`, sandbox settings | — | `gemini --output-format json`, ACP adapter |
| Antigravity | `.agent/rules`, `GEMINI.md` | skills | `.agent/workflows` | agents | — | `mcp_config.json` | IDE sandboxed exec | — | IDE Mission Control |
| Grok | project rules | `~/.grok/skills` | slash commands | subagents | plugins | MCP servers | agent mode | — | `grok` CLI |
| Kiro | `.kiro/steering/*` | `~/.kiro/skills` | — | `~/.kiro/agents/*.json` (`name,model,tools,mcpServers,prompt`) | — | `mcp.json` | trusted commands | steering modes | `kiro` CLI |
| DSH | per-profile instructions | skills | — | — | npm-style components via patch tree | — | `command_allowlist` | `cordis.yml` + `cordis.patch.yml` + `--patch` (layered profile tree) | `dsh` CLI |
| Hermes | `config.yaml` `memory`, `skills` | `platform_toolsets`, skills | — | `delegation` | `plugins/*/plugin.yaml` (`provides_tools`, `provides_hooks`, `requires_env` with provisioning) | `gateway` | `command_allowlist` | — | `hermes` CLI / gateway |
| Pi | — | `~/.pi/agent/skills` | — | — | `extensions/*.ts`, `packages` | — | — | — | `sessions/` JSONL |
| OpenCode | `AGENTS.md` | `~/.opencode/skills` | commands | `~/.config/opencode/agents/*.md` | `plugins/*.js` | `opencode.json` `mcp` | `permission` section | — | `opencode run` / serve |
| Factory Droid | `AGENTS.md` | `~/.factory/skills` | slash-command catalogs | `~/.factory/droids/*.md` | `plugins/` | settings `mcp` | autonomy levels, `trustedFolders` | — | `droid` (ACP probe sessions), exec |
| Command Code | imports `CLAUDE.md`/`AGENTS.md` | skills | — | — | — | `projects/*/mcp.json` | — | — | — |
| OpenClaw | `workspace/AGENTS.md` | `~/.openclaw/skills` | — | `~/.openclaw/agents` | — | — | — | — | — |
| Windsurf → Devin Desktop (renamed 2026-06; Cascade EOL → Devin Local) | `memories/global_rules.md`, `.windsurf/rules` | `~/.codeium/windsurf/skills` | workflows | — | — | `mcp_config.json` | — | `profiles/` | Cascade (IDE) |
| Kimi-code | `AGENTS.md` | `~/.kimi-code/skills` | — | — | — | `config.toml` | `[thinking]`, `[secondary_model]`, `[providers.*]` | — | `kimi` CLI sessions |
| Kilo Code / Roo Code | rules + memories | `~/.kilocode/skills`, `~/.roo/skills` | — | custom modes (IDE `globalStorage`) | — | IDE mcp settings | — | modes | VS Code extension |
| Aider (Desk) | `CONVENTIONS.md`, `.aider.conf.yml` | `~/.aider-desk/skills` | — | — | — | — | auto-commit flags | — | `aider` CLI |
| Junie | guidelines | `~/.junie/skills` | — | — | — | IDE mcp | — | — | JetBrains IDE |
| Tabnine | — | `~/.tabnine/agent/skills` | — | — | — | — | — | — | — |
| Warp | `WARP.md` | `~/.warp/skills` | — | agents (app) | — | — | — | — | terminal blocks |
| Craft Agent | workspace config | `SKILL.md` (Claude-SDK-identical, plus globs/`alwaysAllow` fields) | skill slash commands | — | `.claude-plugin` | — | per-skill tool pre-approval | workspaces | `craft-agent` CLI |

## 3. Interoperability: invocable vs config-surface-only

An `a4aa-<agent>` dialect exists only for products that are **installed,
actually used, and invocable** — a real binary or CLI on PATH plus runtime
config/session artifacts. Products that were never run locally, and GUI-only
surfaces with no spawnable lane, are deliberately *not* adapted.

| Class | Agents | What a host can do |
| --- | --- | --- |
| **Invocable** (CLI / headless process surface) | codex, claude, cursor-agent, devin, opencode, pi, grok, kiro, dsh, hermes, kimi-code, factory, command-code, open-claw, kilo-code (`kilo`), antigravity | Spawn sessions (JSON-RPC, ACP, `exec --json`, `-p` style batch, gateway CLIs) *and* adapt config surfaces |

Products such as Trae, Qoder, Lingma, Kilo Code and Roo Code present only a
GUI/extension surface — no process boundary a host could drive — so they get
no adapter rather than a config-only descriptor that would claim adaptation
without evidence.

## 4. Long tail

These agents expose at minimum a `~/.<agent>/skills/` surface (and typically a
rules file); their deeper surfaces are not enumerated here: trae-cn, qoder-cn,
zencoder, neovate, pochi, vibe, rovodev, zcode, codeartsdoer, ona, terramind,
moxby, mux, autohand, bob, astrbot, haystack, jazz, kode, tinycloud,
codestudio, codewhale, inferencesh, reasonix, onwatch, openhands, copilot.

## 5. What cross-agent tooling already proves

1. `~/.<agent>/skills/` exists for 40+ agents, all consuming the same
   `SKILL.md` pack format — skills are the de-facto portable mechanism class.
2. Existing dotfile-sync tooling mirrors one rules document into every agent's
   native rules file (`AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, …) — rules text is
   portable, the *file location and name* is the per-agent variable.
3. Hooks are *installed through* each agent's native store (codex
   `hooks.json`, cursor `hooks.json`, grok per-file bundles, gemini named
   bundles) — the same logical hook lands in five different dialects.
4. Managed-artifact discipline is established practice: marker comments,
   timestamped `.bak` backups, generated-file header banners — adaptation
   should emit marked, restorable artifacts rather than rewriting user config.

## 6. The implemented contract

This split is now implemented: **AwP parses `Profile.mechanisms` into
semantic declarations; the a4aa workspace owns adaptation.** The engine
lives in `a4aa` (`Dialect::adapt(&MechanismSet) -> AdaptationPlan`); each
agent's knowledge is a `Dialect` const in its own `a4aa-<agent>` crate
(`a4aa_codex::DIALECT`, `a4aa_pi::DIALECT`, ...), independently installable.
`a4aa-dialects` is an optional name registry for multi-agent hosts.

**AwP side (parsing).** The profile `mechanisms` section carries *semantic*
declarations. The parser validates and normalizes them — it contains no
per-agent schema knowledge:

- `hooks.<event>` — semantic event set (`session_start`, `user_prompt`,
  `pre_tool`, `post_tool`, `stop`, `session_end`, `subagent_*`, `permission`);
  `intent: enforce | inject | rewrite | notify` selects the semantics;
  `match` carries the caller's filter expression.
- `rules`, `skills`, `commands` — content references, dialect-free.
- `subagents` — declarative role entries (name, model, instructions,
  sandbox), dialect-free.
- `plugins` — installable references (package, marketplace name, local path).
- `mcpServers`, `permissions`, `env`, `nativeProfile` — already-shaped
  declarations that pass through to the adapter.

**A4AA side (adaptation).** `adapt` translates the normalized declarations
into the agent's dialect and returns an `AdaptationPlan` — artifacts plus a
per-item support report:

- Hook dialect emitters per §1 (Claude dialect → `hooks.json`/`settings.json`
  `json-merge` patch; named bundles → Gemini-style `config/hooks.json`; flat
  → Cursor; code-extension → generated `.ts`/`.js` stubs for Pi/OpenCode).
- Rules rendered into the agent's rules file, project scope first, inside a
  `managed-block` marker.
- Skills written into the agent's skills dir; subagents emitted in the
  agent's declarative form (`*.toml` Codex, `*.md` Claude/Cursor/OpenCode/
  Droid/OpenClaw, `*.json` Kiro); commands emitted per agent (`*.md`,
  `*.toml` Gemini).
- `permissions`/`sandbox` mapped onto `permissions` rules (Claude/Cursor);
  unsupported → reported back as `unsupported` with a note.
- `mcpServers` merged into the agent's JSON MCP config; `env` materializes
  through the existing `AgentSpec` env policy; `nativeProfile` selects the
  agent's own profile (`codex --profile`, DSH profile name).

`awp materialize --dialect <agent>` prints the plan; nothing is written —
the host owns the scope roots and the write decision.
