#!/usr/bin/env bash
# Single-shot verification gate for awp: format, lint, test, and the
# trimmed no-adapter / single-adapter builds. The exhaustive adapter
# matrix lives in scripts/feature-matrix.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
cargo clippy --all-targets --no-default-features -- -D warnings
cargo test --all-targets --no-default-features
cargo check --all-targets --no-default-features --features codex
