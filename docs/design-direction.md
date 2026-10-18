# Design direction: native-max composition vs runtime-max gates

This workspace family abstracts over agent hosts in **two opposite
directions**. They are complementary, not competing.

## The two directions

| | AwP (this repo) + A4AA | AgentOps-Runtime |
|---|---|---|
| Direction | **Bottom-up**: agent-native capability → common language | **Top-down**: standard runtime language → projected subset per host |
| What is maximal | **The native agent.** The framework expresses what the agent natively offers; it never invents semantics the agent lacks | **The runtime language.** Gates define full semantics; a host receives the subset its native capabilities can express |
| Completeness rule | A capability the base agent lacks resolves as unsupported — never simulated | Every enforcement channel a gate applies to is declared in `profiles`; a host's weaker projection is marked `Incomplete` — never silently removed |
| Failure attribution | Missing coverage is an adapter/profile gap | A weak projection is a host capability boundary — not a defect in the language |

Both directions share one rule: **partial support is an explicit
declaration, never silent removal.**

## What this means for AwP

AwP is a *composition* layer on the native-max axis: a `Profile`
declares how a base agent should behave — model, reasoning effort,
persona instructions, environment, MCP servers, dispatch preferences —
and one base agent plus one profile materializes a `VirtualAgent`.

Because the direction is bottom-up, the same constraint applies one
level up:

- A profile can **select and shape** capabilities (model choice,
  effort, persona, tools) — it cannot **confer** a capability the base
  agent's adapter does not offer. A binding that names an unsupported
  selector resolves as unsupported rather than silently downgrading.
- The registry is a declaration in the common language, not a semantic
  source. Roles, overlays and bindings compose what base agents
  natively expose; the semantics originate with the agent, expressed
  through `a4aa` descriptors.

## Composition

The three projects describe the same hosts on different axes:

- **A4AA** answers *"what does this host natively speak?"* — per-agent
  descriptor crates form a capability map.
- **AwP** answers *"how should this agent behave here?"* — profiles
  compose that native capability into named virtual agents.
- **AgentOps-Runtime** answers *"which subset of the runtime language
  lands on this host?"* — its adapters project gate semantics onto
  whatever the host can express, reporting the remainder as `Incomplete`.

The same capability map feeds both consumers: AwP composes it into
behavior; AgentOps-Runtime projects runtime discipline onto it.
