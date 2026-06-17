//! Layer-graph guard: extracts `crate::<top>::` edges from each production
//! file under `modules/<top>/` and asserts every (source, target) is in
//! `ALLOWED_EDGES`. The cut is the first `#[cfg(test)]`, which in idiomatic
//! files is a top-of-file `#[cfg(test)] mod tests;`, so production edges below
//! it go unscanned: the guard can MISS a real edge (false negative) but never
//! INVENTS one, so a clean run is necessary, not sufficient. Line comments are stripped
//! (so `crate::X::` in a doc isn't counted) but block comments are not (crate
//! uses line comments only), so any `crate::X::` inside one needs an allowlist entry.

#![allow(clippy::disallowed_methods)]

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

/// Layer-bearing top-level modules in `modules/lib.rs` (excludes `allocator`,
/// the global allocator, which is not a layer); only a `crate::X::` whose `X`
/// is listed here is a layer edge (others are intra-module sub-paths).
const TOP_MODULES: &[&str] = &[
    "audio_buffer",
    "audio_io",
    "api",
    "common",
    "config",
    "converter",
    "daemon",
    "dsp",
    "file_mgr",
    "inference",
    "model",
    "opus_stream",
    "preproc",
    "proto",
    "rknn_runtime",
    "sched",
    "status",
    "stream_io",
    "training",
];

/// Allowed direct dependencies: edge `(source, target)` is permitted iff
/// `target` is in `ALLOWED_EDGES[source]`.
const ALLOWED_EDGES: &[(&str, &[&str])] = &[
    ("audio_buffer", &[]),
    ("audio_io", &["audio_buffer", "common", "dsp", "sched"]),
    // `api` adapts every application domain except producer-only ones
    // (audio_buffer, proto, opus_stream, stream_io).
    (
        "api",
        &[
            "audio_io",
            "common",
            "config",
            "converter",
            "file_mgr",
            "inference",
            "model",
            "status",
            "training",
        ],
    ),
    ("common", &[]),
    // `config -> file_mgr`: durability edge; config writers delegate to
    // `fs_atomic::put_atomic`, a deliberate lateral to a low-dep primitive.
    (
        "config",
        &["audio_io", "common", "file_mgr", "inference", "stream_io"],
    ),
    ("converter", &["common", "file_mgr", "model"]),
    // `daemon` is the composition root; listed explicitly (not wildcarded) so a
    // new module surfaces as a required allowlist edit.
    (
        "daemon",
        &[
            "api",
            "audio_buffer",
            "audio_io",
            "common",
            "config",
            "converter",
            "dsp",
            "file_mgr",
            "inference",
            "model",
            "opus_stream",
            "preproc",
            "proto",
            "rknn_runtime",
            "sched",
            "status",
            "stream_io",
            "training",
        ],
    ),
    // `dsp -> common`: `Categorized` impl on `StreamingResampleError` reaches
    // into `common::error`, the canonical typed-error trait home.
    ("dsp", &["common"]),
    ("file_mgr", &["common"]),
    (
        "inference",
        &[
            "audio_buffer",
            "common",
            "model",
            "preproc",
            "proto",
            "rknn_runtime",
        ],
    ),
    ("model", &["common"]),
    (
        "opus_stream",
        &["audio_buffer", "audio_io", "common", "dsp", "proto"],
    ),
    // `preproc -> audio_io`: WAV-ingest reuses capture-side caps (MAX_CHANNELS,
    // MIN/MAX_SAMPLE_RATE) to keep capture and WAV admission policy in sync.
    ("preproc", &["audio_io", "common", "dsp"]),
    ("proto", &["common"]),
    ("rknn_runtime", &[]),
    ("sched", &[]),
    ("status", &["common"]),
    ("stream_io", &["common", "proto"]),
    ("training", &["common", "file_mgr", "model", "preproc"]),
];

/// Every Rust file under `modules/`, paired with its top-level module.
fn discover_production_files() -> Vec<(String, PathBuf)> {
    let modules_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("modules");
    let mut out = Vec::new();
    for top in TOP_MODULES {
        // A top module may have both a `<top>.rs` parent and a `<top>/` dir.
        let parent = modules_root.join(format!("{top}.rs"));
        if parent.exists() {
            out.push((top.to_string(), parent));
        }
        let dir = modules_root.join(top);
        if dir.is_dir() {
            walk_dir(&dir, top, &mut out);
        }
    }
    out
}

fn walk_dir(dir: &Path, top: &str, acc: &mut Vec<(String, PathBuf)>) {
    let entries = fs::read_dir(dir).expect("read modules dir");
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk_dir(&path, top, acc);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            // Skip `tests.rs`: test fixtures carry no layer-graph contract.
            if path.file_name().and_then(|n| n.to_str()) == Some("tests.rs") {
                continue;
            }
            acc.push((top.to_string(), path));
        }
    }
}

/// Source head before the first `#[cfg(test)]`, with line comments stripped.
fn production_segment(src: &str) -> String {
    let cut = src.find("#[cfg(test)]").unwrap_or(src.len());
    let head = &src[..cut];
    let mut out = String::with_capacity(head.len());
    for line in head.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("//") {
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// Unique `crate::<ident>::` identifiers in `src`, filtered to `TOP_MODULES`.
fn extract_edges(src: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let needle = "crate::";
    let mut i = 0;
    while let Some(pos) = src[i..].find(needle) {
        let start = i + pos + needle.len();
        let bytes = src.as_bytes();
        let mut end = start;
        while end < bytes.len() {
            let c = bytes[end];
            if c.is_ascii_alphanumeric() || c == b'_' {
                end += 1;
            } else {
                break;
            }
        }
        if end > start {
            let ident = &src[start..end];
            if TOP_MODULES.contains(&ident) {
                out.insert(ident.to_string());
            }
        }
        i = end;
    }
    out
}

/// Fails (naming the file and edge) on any `crate::<top>::` reference outside
/// the allowlist; fix the import, or add the edge if deliberate.
#[test]
fn no_forbidden_layer_edges() {
    let allowed: BTreeMap<&str, BTreeSet<&str>> = ALLOWED_EDGES
        .iter()
        .map(|(src, targets)| (*src, targets.iter().copied().collect()))
        .collect();

    let files = discover_production_files();
    assert!(
        files.len() >= 30,
        "discover_production_files returned only {} files; expected >= 30 across {} top modules",
        files.len(),
        TOP_MODULES.len(),
    );

    let mut violations: Vec<String> = Vec::new();
    for (top, path) in &files {
        let src = match fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => panic!("read {}: {e}", path.display()),
        };
        let prod = production_segment(&src);
        let edges = extract_edges(&prod);
        let allow_for_src = allowed
            .get(top.as_str())
            .unwrap_or_else(|| panic!("no allowlist entry for top module `{top}`"));
        for tgt in &edges {
            if tgt == top {
                // Self-references are not layer edges.
                continue;
            }
            if !allow_for_src.contains(tgt.as_str()) {
                violations.push(format!(
                    "FORBIDDEN EDGE: {} -> {} in {}\n  (allowed targets for `{}`: {:?})",
                    top,
                    tgt,
                    path.strip_prefix(env!("CARGO_MANIFEST_DIR"))
                        .unwrap_or(path)
                        .display(),
                    top,
                    allow_for_src,
                ));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "{} forbidden layer-graph edge(s) found; update either the source \
         file (fix the import) OR `docs/ARCH_BOUNDARIES.md` + the \
         `ALLOWED_EDGES` table in this test (when introducing a deliberate \
         new edge):\n\n{}",
        violations.len(),
        violations.join("\n\n"),
    );
}

/// Guards the table against typos: every top module has an entry, and every
/// target is a known top module.
#[test]
fn allowlist_table_is_well_formed() {
    let allowed_keys: BTreeSet<&str> = ALLOWED_EDGES.iter().map(|(src, _)| *src).collect();
    let top_set: BTreeSet<&str> = TOP_MODULES.iter().copied().collect();

    let missing: Vec<&str> = top_set.difference(&allowed_keys).copied().collect();
    assert!(
        missing.is_empty(),
        "ALLOWED_EDGES is missing entries for top modules: {missing:?}",
    );

    for (src, targets) in ALLOWED_EDGES {
        for tgt in *targets {
            assert!(
                top_set.contains(tgt),
                "ALLOWED_EDGES[{src}] references unknown top module `{tgt}`",
            );
        }
    }
}
