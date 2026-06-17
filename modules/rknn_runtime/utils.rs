//! Runtime path helpers.

use std::path::PathBuf;

/// Single source of truth so discovery and descriptions cannot drift.
enum SearchDirSource {
    Fixed(&'static str),
    /// Env var name holding a colon-separated PATH-style list.
    EnvColonSplit(&'static str),
    /// `$HOME`-relative suffix; skipped when `$HOME` is unset.
    HomeRelative(&'static str),
}

const SEARCH_DIRS: &[SearchDirSource] = &[
    SearchDirSource::Fixed("/usr/lib"),
    SearchDirSource::Fixed("/usr/local/lib"),
    SearchDirSource::Fixed("/usr/lib/aarch64-linux-gnu"),
    SearchDirSource::EnvColonSplit("LD_LIBRARY_PATH"),
    SearchDirSource::HomeRelative(".local/lib"),
];

fn collect_search_dirs() -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::with_capacity(SEARCH_DIRS.len() + 4);
    for src in SEARCH_DIRS {
        match src {
            SearchDirSource::Fixed(p) => dirs.push(PathBuf::from(*p)),
            SearchDirSource::EnvColonSplit(var) => {
                if let Ok(value) = std::env::var(var) {
                    for part in value.split(':').filter(|s| !s.is_empty()) {
                        dirs.push(PathBuf::from(part));
                    }
                }
            }
            SearchDirSource::HomeRelative(suffix) => {
                if let Ok(home) = std::env::var("HOME") {
                    dirs.push(PathBuf::from(home).join(suffix));
                }
            }
        }
    }
    dirs
}

/// Descriptions in discovery order; all axes always render (env as `$VAR`) so an unset env still shows as settable.
pub fn library_search_dir_descriptions() -> Vec<String> {
    SEARCH_DIRS
        .iter()
        .map(|src| match src {
            SearchDirSource::Fixed(p) => (*p).to_string(),
            SearchDirSource::EnvColonSplit(var) => format!("${var} (colon-separated)"),
            SearchDirSource::HomeRelative(suffix) => format!("$HOME/{suffix}"),
        })
        .collect()
}

/// Hits for `librknnrt.so` / `librknnmrt.so` in discovery order (first preferred).
pub fn find_library_candidates() -> Vec<PathBuf> {
    let dirs = collect_search_dirs();
    let mut hits = Vec::new();
    for d in &dirs {
        for name in ["librknnrt.so", "librknnmrt.so"] {
            let p = d.join(name);
            if p.exists() {
                hits.push(p);
            }
        }
    }
    hits
}
