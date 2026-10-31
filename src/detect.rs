//! Machine-side discovery of installed agents — the Finder probe.
//!
//! A host must never presuppose which agents the user's machine carries:
//! "what exists here" is a runtime fact, not an installation-time
//! assumption. Every [`a4aa`] card declares a `detect` surface — binary
//! names resolvable on `PATH`, install directories outside `PATH`,
//! configuration files and prior-use markers — and [`scan`] resolves
//! those declarations against the real machine. Cards and presets are
//! templates: they describe how to recognize an agent, never require one.

use std::path::{Path, PathBuf};

/// What one known agent's probe found on this machine.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Detection {
    /// Card id (`pi`, `codex`, …) — the agent's canonical name.
    pub id: &'static str,
    /// The binary names the card recognizes — a host joining discovery
    /// results onto registered agents matches `command.program` basenames
    /// against this set.
    pub binaries: &'static [&'static str],
    /// The first executable any of the card's binaries resolved to.
    pub binary: Option<PathBuf>,
    /// `config`, `detection` and `apps` marker paths that exist.
    pub markers: Vec<PathBuf>,
}

impl Detection {
    /// A usable executable was found — the agent can actually serve.
    pub fn installed(&self) -> bool {
        self.binary.is_some()
    }
}

/// Probe every known agent card against this machine, in catalog order.
pub fn scan() -> Vec<Detection> {
    a4aa_dialects::cards()
        .map(|(id, card)| {
            let detect = &card.detect;
            let binary = detect
                .binaries
                .iter()
                .find_map(|name| on_path(name))
                .or_else(|| {
                    detect
                        .binary_dirs
                        .iter()
                        .filter(|dir| os_matches(dir.oses))
                        .find_map(|dir| {
                            expand(dir.path).and_then(|dir| {
                                detect
                                    .binaries
                                    .iter()
                                    .find_map(|name| executable(&dir.join(name)))
                            })
                        })
                });
            let markers = detect
                .config
                .iter()
                .chain(detect.detection.iter())
                .filter_map(|marker| expand(marker).filter(|path| path.exists()))
                .chain(
                    detect
                        .apps
                        .iter()
                        .filter(|app| os_matches(app.oses))
                        .filter_map(|app| expand(app.path).filter(|path| path.exists())),
                )
                .collect();
            Detection {
                id,
                binaries: detect.binaries,
                binary,
                markers,
            }
        })
        .collect()
}

/// Resolve `name` to the first executable reachable on `PATH`.
pub fn on_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).find_map(|dir| executable(&dir.join(name)))
}

fn executable(path: &Path) -> Option<PathBuf> {
    let meta = std::fs::metadata(path).ok()?;
    if !meta.is_file() {
        return None;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        if meta.permissions().mode() & 0o111 == 0 {
            return None;
        }
    }
    Some(path.to_path_buf())
}

/// `{home}`-template expansion; `USERPROFILE` stands in where `HOME` is
/// unset, matching the custody layer's home resolution.
fn expand(template: &str) -> Option<PathBuf> {
    if !template.contains("{home}") {
        return Some(PathBuf::from(template));
    }
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"))?;
    Some(PathBuf::from(
        template.replace("{home}", &home.to_string_lossy()),
    ))
}

/// An `OsPath` with no `oses` applies everywhere.
fn os_matches(oses: &[&str]) -> bool {
    oses.is_empty() || oses.contains(&std::env::consts::OS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scan_covers_every_card_and_guesses_nothing() {
        let detections = scan();
        assert_eq!(detections.len(), a4aa_dialects::cards().count());
        for detection in &detections {
            if let Some(binary) = &detection.binary {
                assert!(binary.is_absolute() || on_path(binary.to_str().unwrap()).is_some());
            }
        }
    }

    #[test]
    fn a_path_marker_expands_only_with_a_home() {
        let expanded = expand("{home}/.pi/agent/settings.json");
        if std::env::var_os("HOME").is_some() || std::env::var_os("USERPROFILE").is_some() {
            let expanded = expanded.expect("a home exists on this machine");
            assert!(expanded.is_absolute());
            assert!(!expanded.to_string_lossy().contains("{home}"));
        } else {
            assert!(expanded.is_none());
        }
    }

    #[test]
    fn a_non_template_path_expands_verbatim() {
        assert_eq!(
            expand("/opt/homebrew/bin"),
            Some(PathBuf::from("/opt/homebrew/bin"))
        );
    }

    #[test]
    fn the_current_os_matches_universal_and_named_paths() {
        assert!(os_matches(&[]));
        assert!(os_matches(&[std::env::consts::OS]));
        assert!(!os_matches(&["some-other-os"]));
    }
}
