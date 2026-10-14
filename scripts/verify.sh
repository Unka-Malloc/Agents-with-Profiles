#!/usr/bin/env bash
# Single-shot verification gate for awp: format, lint, test.
set -euo pipefail
cd "$(dirname "$0")/.."

cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
