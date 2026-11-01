//! Machine-side discovery of installed agents — the Finder probe.
//!
//! A host must never presuppose which agents the user's machine carries:
//! "what exists here" is a runtime fact, not an installation-time
//! assumption. Every [`a4aa`] card declares a `detect` surface — binary
//! names resolvable on `PATH` (through `PATHEXT` shims on Windows),
//! install directories outside `PATH`, configuration files and prior-use
//! markers — and [`scan`] resolves those declarations against the real
//! machine. Cards and presets are templates: they describe how to
//! recognize an agent, never require one.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use a4aa::DetectSurface;

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

/// The machine facts a scan resolves against — `PATH` entries, the home
/// directory, the running OS and, on Windows, the `PATHEXT` extensions a
/// bare name resolves through. [`scan`] snapshots the real machine; tests
/// inject a fabricated one.
struct Machine {
    path: Vec<PathBuf>,
    home: Option<OsString>,
    os: &'static str,
    pathext: Vec<String>,
}

impl Machine {
    /// Snapshot this machine's discovery facts.
    fn local() -> Self {
        Self {
            path: std::env::var_os("PATH")
                .map(|paths| std::env::split_paths(&paths).collect())
                .unwrap_or_default(),
            home: std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")),
            os: std::env::consts::OS,
            pathext: pathext(),
        }
    }

    /// Resolve `name` to the first launchable file reachable on `PATH`.
    fn on_path(&self, name: &str) -> Option<PathBuf> {
        self.path
            .iter()
            .find_map(|dir| self.launchable_in(dir, name))
    }

    /// The launchable form of `name` inside `dir`: the name verbatim, or
    /// on Windows the name plus each `PATHEXT` extension in order — a
    /// `tool.cmd` shim resolves for `tool`.
    fn launchable_in(&self, dir: &Path, name: &str) -> Option<PathBuf> {
        executable(&dir.join(name)).or_else(|| {
            self.pathext
                .iter()
                .find_map(|ext| executable(&dir.join(format!("{name}{ext}"))))
        })
    }

    /// `{home}`-template expansion against this machine's home directory;
    /// a template with no home to bind expands to nothing.
    fn expand(&self, template: &str) -> Option<PathBuf> {
        if !template.contains("{home}") {
            return Some(PathBuf::from(template));
        }
        let home = self.home.as_ref()?;
        Some(PathBuf::from(
            template.replace("{home}", &home.to_string_lossy()),
        ))
    }

    /// An `OsPath` with no `oses` applies everywhere.
    fn os_matches(&self, oses: &[&str]) -> bool {
        oses.is_empty() || oses.contains(&self.os)
    }
}

/// Probe every known agent card against this machine, in catalog order.
pub fn scan() -> Vec<Detection> {
    probe_cards(
        a4aa_dialects::cards().map(|(id, card)| (id, &card.detect)),
        &Machine::local(),
    )
}

/// The probe loop against an explicit machine — the seam tests inject
/// fabricated cards and machine facts through.
fn probe_cards(
    cards: impl Iterator<Item = (&'static str, &'static DetectSurface)>,
    machine: &Machine,
) -> Vec<Detection> {
    cards
        .map(|(id, detect)| {
            let binary = detect
                .binaries
                .iter()
                .find_map(|name| machine.on_path(name))
                .or_else(|| {
                    detect
                        .binary_dirs
                        .iter()
                        .filter(|dir| machine.os_matches(dir.oses))
                        .find_map(|dir| {
                            machine.expand(dir.path).and_then(|dir| {
                                detect
                                    .binaries
                                    .iter()
                                    .find_map(|name| machine.launchable_in(&dir, name))
                            })
                        })
                });
            let markers = detect
                .config
                .iter()
                .chain(detect.detection.iter())
                .filter_map(|marker| machine.expand(marker).filter(|path| path.exists()))
                .chain(
                    detect
                        .apps
                        .iter()
                        .filter(|app| machine.os_matches(app.oses))
                        .filter_map(|app| machine.expand(app.path).filter(|path| path.exists())),
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
    Machine::local().on_path(name)
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

/// The extensions a bare name resolves through, read from `PATHEXT` with
/// the cmd.exe default set as fallback. Non-Windows machines have no such
/// surface: the list stays empty and only the verbatim name resolves.
#[cfg(windows)]
fn pathext() -> Vec<String> {
    let from_env = std::env::var("PATHEXT").ok().map(|value| {
        value
            .split(';')
            .map(str::trim)
            .filter(|ext| !ext.is_empty())
            .map(|ext| {
                if ext.starts_with('.') {
                    ext.to_owned()
                } else {
                    format!(".{ext}")
                }
            })
            .collect::<Vec<_>>()
    });
    match from_env {
        Some(exts) if !exts.is_empty() => exts,
        _ => [".COM", ".EXE", ".BAT", ".CMD"]
            .iter()
            .map(|ext| ext.to_string())
            .collect(),
    }
}

/// Non-Windows machines launch exactly the file the card names.
#[cfg(not(windows))]
fn pathext() -> Vec<String> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use a4aa::OsPath;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A unique temporary directory removed on drop.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(name: &str) -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let dir = std::env::temp_dir().join(format!(
                "awp-detect-{}-{}-{name}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed),
            ));
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Write `dir/name` as a launchable file — an exec-bit binary on
    /// Unix, a `.CMD` shim (matching `PATHEXT` case) on Windows — and
    /// return its path.
    fn install(dir: &Path, name: &str) -> PathBuf {
        let path = dir.join(if cfg!(windows) {
            format!("{name}.CMD")
        } else {
            name.to_owned()
        });
        std::fs::write(&path, b"agent").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            let mut permissions = std::fs::metadata(&path).unwrap().permissions();
            permissions.set_mode(0o755);
            std::fs::set_permissions(&path, permissions).unwrap();
        }
        path
    }

    /// A fabricated machine: `PATH` is exactly `path`, home is exactly
    /// `home`, and the OS name is `test-os` so fixtures gate on it
    /// instead of the host's real OS.
    fn machine(home: Option<&Path>, path: &[PathBuf]) -> Machine {
        Machine {
            path: path.to_vec(),
            home: home.map(|home| home.as_os_str().to_owned()),
            os: "test-os",
            pathext: if cfg!(windows) {
                vec![".CMD".to_owned()]
            } else {
                Vec::new()
            },
        }
    }

    static FAKE: DetectSurface = DetectSurface {
        binaries: &["missing", "fakeagent"],
        processes: &[],
        binary_dirs: &[],
        config: &["{home}/.fakeconfig", "/definitely/missing/config"],
        detection: &["{home}/.fake/used.marker"],
        apps: &[
            OsPath {
                path: "{home}/Applications/Fake.app",
                oses: &["test-os"],
            },
            OsPath {
                path: "{home}/Applications/Fake.app",
                oses: &["other-os"],
            },
        ],
    };

    static BUNDLED: DetectSurface = DetectSurface {
        binaries: &["fakeonly"],
        processes: &[],
        binary_dirs: &[
            OsPath {
                path: "{home}/.gated/bin",
                oses: &["other-os"],
            },
            OsPath {
                path: "{home}/.fake/bin",
                oses: &[],
            },
        ],
        config: &[],
        detection: &[],
        apps: &[],
    };

    static GHOST: DetectSurface = DetectSurface {
        binaries: &["ghost"],
        processes: &[],
        binary_dirs: &[],
        config: &[],
        detection: &[],
        apps: &[],
    };

    #[test]
    fn a_scan_reports_resolved_binaries_and_markers_per_card() {
        let root = TempDir::new("scan");
        let home = root.0.join("home");
        let path_dir = root.0.join("path");
        let bundle = home.join(".fake/bin");
        let gated = home.join(".gated/bin");
        for dir in [&path_dir, &bundle, &gated] {
            std::fs::create_dir_all(dir).unwrap();
        }

        // One PATH hit, one binary that resolves only inside a {home}
        // install dir, and a copy of the same binary in a dir gated to
        // another OS that must never resolve.
        let on_path_exe = install(&path_dir, "fakeagent");
        let bundled_exe = install(&bundle, "fakeonly");
        install(&gated, "fakeonly");

        // Marker files under the fabricated home.
        let config = home.join(".fakeconfig");
        std::fs::write(&config, b"{}").unwrap();
        let used = home.join(".fake/used.marker");
        std::fs::write(&used, b"").unwrap();
        let app = home.join("Applications/Fake.app");
        std::fs::create_dir_all(app.parent().unwrap()).unwrap();
        std::fs::write(&app, b"").unwrap();

        let detections = probe_cards(
            [("fake", &FAKE), ("bundled", &BUNDLED), ("ghost", &GHOST)].into_iter(),
            &machine(Some(&home), &[path_dir]),
        );

        // Card order is catalog order.
        assert_eq!(
            detections.iter().map(|d| d.id).collect::<Vec<_>>(),
            ["fake", "bundled", "ghost"]
        );

        let fake = &detections[0];
        // `missing` falls through to `fakeagent` on PATH.
        assert_eq!(fake.binary.as_deref(), Some(on_path_exe.as_path()));
        assert!(fake.installed());
        assert_eq!(fake.binaries, ["missing", "fakeagent"]);
        // Config and detection markers in declaration order, then the
        // OS-matching app; the absent literal and the other-OS app are
        // not markers.
        assert_eq!(fake.markers, vec![config, used, app]);

        let bundled = &detections[1];
        // The other-OS install dir is skipped although the file exists;
        // the universal {home} dir resolves.
        assert_eq!(bundled.binary.as_deref(), Some(bundled_exe.as_path()));
        assert!(bundled.markers.is_empty());

        let ghost = &detections[2];
        assert!(ghost.binary.is_none());
        assert!(!ghost.installed());
        assert!(ghost.markers.is_empty());
    }

    #[test]
    fn a_home_template_expands_only_against_a_home() {
        let mut machine = machine(None, &[]);
        assert_eq!(machine.expand("{home}/.pi/agent/settings.json"), None);
        machine.home = Some(OsString::from("/somehome"));
        assert_eq!(
            machine.expand("{home}/.pi/agent/settings.json"),
            Some(PathBuf::from("/somehome/.pi/agent/settings.json"))
        );
        assert_eq!(
            machine.expand("/opt/homebrew/bin"),
            Some(PathBuf::from("/opt/homebrew/bin"))
        );
    }

    #[test]
    fn os_gating_uses_the_machine_os() {
        let machine = machine(None, &[]);
        assert!(machine.os_matches(&[]));
        assert!(machine.os_matches(&["test-os"]));
        assert!(!machine.os_matches(&["windows", "linux"]));
    }

    /// A `.cmd` shim on `PATH` is a launchable agent, not a miss — on
    /// Windows every PATH-installed agent was reported absent.
    #[cfg(windows)]
    #[test]
    fn a_cmd_shim_on_path_resolves_through_pathext() {
        let root = TempDir::new("pathext");
        let shim = install(&root.0, "tool");
        let machine = Machine {
            path: vec![root.0.clone()],
            home: None,
            os: "windows",
            pathext: vec![".CMD".to_owned()],
        };
        assert_eq!(machine.on_path("tool"), Some(shim));
        assert_eq!(machine.on_path("ghost"), None);
        // The real reader never ends up extensionless: env or defaults.
        assert!(!pathext().is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn a_file_without_the_exec_bit_is_not_launchable() {
        let root = TempDir::new("exec-bit");
        std::fs::write(root.0.join("tool"), b"agent").unwrap();
        let machine = machine(None, std::slice::from_ref(&root.0));
        assert_eq!(machine.on_path("tool"), None);
    }

    #[test]
    fn scan_covers_every_card_in_catalog_order() {
        let detections = scan();
        let cards: Vec<_> = a4aa_dialects::cards().collect();
        assert_eq!(detections.len(), cards.len());
        for (detection, (id, card)) in detections.iter().zip(cards) {
            assert_eq!(detection.id, id);
            assert_eq!(detection.binaries, card.detect.binaries);
            // A resolved binary is always a launchable file that exists.
            if let Some(binary) = &detection.binary {
                assert!(binary.is_file());
            }
        }
    }
}
