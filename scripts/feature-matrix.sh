#!/usr/bin/env bash
# Verify the feature-pruning matrix: no defaults, every single adapter,
# presets only, dialects only. Each combination must check and test, and a
# single-adapter build must pull exactly one a4aa-<agent> package.
set -euo pipefail
cd "$(dirname "$0")/.."

adapters=(
  antigravity claude codex commandcode copilot cursor devin dsh factory
  grok hermes kilocode kimicode kiro openclaw opencode pi
)
combinations=("" "presets" "dialects" "${adapters[@]}")

# Count `a4aa-<agent>` packages in the normal (non-dev) dependency tree.
adapter_count() {
  local tree
  local args=(--no-default-features --edges normal --prefix none --format '{p}')
  if [[ -n "${1:-}" ]]; then
    args+=(--features "$1")
  fi
  tree=$(cargo tree "${args[@]}")
  printf '%s\n' "$tree" |
    sed -E 's/ v[0-9].*$//' |
    awk '/^a4aa-/' |
    sort -u |
    wc -l |
    tr -d ' '
}

for combination in "${combinations[@]}"; do
  label=${combination:-none}
  args=(--no-default-features)
  expected=0
  if [[ -n "$combination" ]]; then
    args+=(--features "$combination")
    case "$combination" in
      presets) expected=6 ;;
      dialects) expected=17 ;;
      *) expected=1 ;;
    esac
  fi

  count=$(adapter_count "$combination")
  if [[ "$count" -ne "$expected" ]]; then
    echo "error: --features $label pulled $count a4aa-<agent> packages; expected $expected" >&2
    tree_args=(--no-default-features --edges normal --prefix none --format '{p}')
    if [[ -n "$combination" ]]; then
      tree_args+=(--features "$combination")
    fi
    cargo tree "${tree_args[@]}" >&2
    exit 1
  fi

  echo "== check  $label ($count adapter(s))"
  cargo check --all-targets "${args[@]}"
  echo "== test   $label ($count adapter(s))"
  cargo test --all-targets "${args[@]}" -- --quiet
done

echo "feature matrix: ok"
