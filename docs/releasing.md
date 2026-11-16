# Releasing the crate

This repository publishes the single package `agents-with-profiles` (library
and binary `awp`) manually from a reviewed default-branch commit. There is no
generated release train and no publishing workflow.

The package depends on `a4aa`; publish a new `a4aa` before this crate when the
release needs one.

## Before publishing

1. Bump `version` in `Cargo.toml` — this is one package, so there is no
   separate version file to bump.
2. Move the `Unreleased` notes in `CHANGELOG.md` under the new version.
3. Run `./scripts/verify.sh` — formatting, clippy with warnings denied, and
   the full test suite. `./scripts/feature-matrix.sh` covers the trimmed
   feature combinations.
4. Run `cargo package --list` and `cargo publish --dry-run`; check that the
   packaged surface is the intended one and that every dependency version is
   already available from crates.io.

## Publishing

1. Commit the version bump and changelog, and merge that commit to the default
   branch.
2. Tag that commit `v<version>` and push the tag.
3. From that commit, run `cargo publish --locked` with a crates.io token that
   may publish `agents-with-profiles`; use `--dry-run` first when unsure.
4. Confirm the crates.io page or `cargo info agents-with-profiles` reports the
   new version.

Local checks prove the candidate; only `cargo publish` uploads.

## Credentials

`cargo publish` reads its registry token from Cargo's credentials store
(`~/.cargo/credentials.toml`) or `CARGO_REGISTRY_TOKEN`. Never commit a token,
place one in this repository, or pass it on a command line.
