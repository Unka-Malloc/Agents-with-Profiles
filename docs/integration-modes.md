# Machine discovery

`awp`'s consumption contract — embedding the crate in Rust vs driving
`awp resolve` + `a4aa serve` from any language — is specified once in
[integration.md](integration.md). This page covers only what the crate
learns from the machine itself.

`awp::detect_agents()` probes this machine against every known agent
card — `PATH` binaries, install directories outside `PATH`, configuration
files, prior-use markers and OS-gated application bundles — and returns
one `Detection` per card: the resolved executable, the marker paths that
exist, and the binary-name set a host joins against `command.program`
basenames. On Windows a bare program name also resolves through each
`PATHEXT` extension in order, so `tool.cmd`/`tool.exe` shims count as
hits both on `PATH` and in install directories. What exists on a machine
is a runtime fact, never an
install-time assumption; a host pairs each detection with
`awp::lookup_preset` to offer one-step registration for agents already
installed. Discovery lives behind the `dialects` feature (on by default)
beside the card registry.
