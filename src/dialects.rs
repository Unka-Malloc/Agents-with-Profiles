//! The dialects compiled into this build.
//!
//! Every adapter lives in its own `a4aa-<agent>` crate, and every adapter is
//! an independent Cargo feature here. A multi-agent host enables the
//! `dialects` feature (all adapters) or compiles this crate with
//! `default-features = false` and picks exactly the adapters it drives; the
//! single-adapter build pulls no other descriptor crate into the dependency
//! tree.
//!
//! Lookup is limited to the compiled-in set by construction: a name whose
//! adapter was not enabled behaves like any other unknown name, so a
//! trimmed build cannot silently rely on an adapter it did not compile.

use a4aa::card::AgentCard;
use a4aa::Dialect;

/// The portable baseline has no adapter crate; it is always available.
static GENERIC: Dialect = Dialect::generic();

macro_rules! catalog {
    ($($feature:literal => $name:literal => $krate:ident),+ $(,)?) => {
        /// Look up a compiled-in dialect by its canonical id.
        ///
        /// `generic` is always available. Returns `None` for an adapter that
        /// was not enabled in this build and for unknown names.
        pub fn dialect(name: &str) -> Option<&'static Dialect> {
            if name == "generic" {
                return Some(&GENERIC);
            }
            $(
                #[cfg(feature = $feature)]
                {
                    if name == $name {
                        return Some($krate::dialect());
                    }
                }
            )+
            None
        }

        /// Look up a compiled-in agent card by its canonical id.
        ///
        /// Returns `None` for `generic`, for an adapter that was not enabled
        /// in this build, and for unknown names.
        #[allow(unused_variables)]
        pub fn card(name: &str) -> Option<&'static AgentCard> {
            $(
                #[cfg(feature = $feature)]
                {
                    if name == $name {
                        return Some($krate::card());
                    }
                }
            )+
            None
        }

        /// Every compiled-in `(id, card)` pair, in adapter order.
        #[allow(unused_mut)]
        pub fn cards() -> impl Iterator<Item = (&'static str, &'static AgentCard)> {
            let mut cards = Vec::new();
            $(
                #[cfg(feature = $feature)]
                {
                    cards.push(($name, $krate::card()));
                }
            )+
            cards.into_iter()
        }

        /// The Cargo feature that compiles a catalog adapter — the name a
        /// trimmed build must enable — whether or not this build enabled
        /// it. Diagnostics report the real manifest feature; `generic`
        /// and unknown names return `None`.
        pub fn feature(name: &str) -> Option<&'static str> {
            $(
                if name == $name {
                    return Some($feature);
                }
            )+
            None
        }

        /// Every compiled-in adapter id, in adapter order.
        #[allow(unused_mut)]
        pub fn names() -> impl Iterator<Item = &'static str> {
            let mut names = Vec::new();
            $(
                #[cfg(feature = $feature)]
                {
                    names.push($name);
                }
            )+
            names.into_iter()
        }
    };
}

catalog! {
    "antigravity" => "antigravity" => a4aa_antigravity,
    "claude" => "claude" => a4aa_claude,
    "codex" => "codex" => a4aa_codex,
    "commandcode" => "commandcode" => a4aa_command_code,
    "copilot" => "copilot" => a4aa_copilot,
    "cursor" => "cursor" => a4aa_cursor,
    "devin" => "devin" => a4aa_devin,
    "dsh" => "dsh" => a4aa_dsh,
    "factory" => "factory" => a4aa_factory,
    "grok" => "grok" => a4aa_grok,
    "hermes" => "hermes" => a4aa_hermes,
    "kilocode" => "kilocode" => a4aa_kilo_code,
    "kimicode" => "kimicode" => a4aa_kimi_code,
    "kiro" => "kiro" => a4aa_kiro,
    "openclaw" => "openclaw" => a4aa_open_claw,
    "opencode" => "opencode" => a4aa_opencode,
    "pi" => "pi" => a4aa_pi,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generic_is_always_available() {
        assert_eq!(dialect("generic").map(|d| d.id), Some("generic"));
        assert!(card("generic").is_none());
    }

    #[test]
    fn lookup_is_limited_to_the_compiled_set() {
        for name in names() {
            assert_eq!(dialect(name).map(|d| d.id), Some(name));
            assert_eq!(card(name).map(|c| c.id), Some(name));
        }
        for (name, card) in cards() {
            assert_eq!(name, card.id);
            assert_eq!(dialect(name).map(|d| d.id), Some(card.dialect.id));
        }
    }

    #[test]
    fn unknown_names_stay_unknown() {
        assert!(dialect("no-such-agent").is_none());
        assert!(card("no-such-agent").is_none());
    }

    #[test]
    fn feature_names_report_the_manifest_feature() {
        // Missing-adapter diagnostics name the feature a build must
        // enable — catalog ids double as feature names, so a trimmed
        // build never invents an `a4aa-<agent>` feature.
        for catalog in ["pi", "codex", "dsh"] {
            assert_eq!(feature(catalog), Some(catalog));
        }
        assert_eq!(feature("generic"), None);
        assert_eq!(feature("no-such-agent"), None);
    }
}
