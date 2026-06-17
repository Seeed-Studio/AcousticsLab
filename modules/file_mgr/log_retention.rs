//! Producer-side keep-last-N retention for per-workspace JSONL job logs; callers
//! invoke [`enforce_keep_last_n`] right after opening a new log, when the cap can
//! first be exceeded. Only regular `.jsonl` files are candidates (subdirs/symlinks/
//! other extensions survive, protecting operator artifacts). Future-mtime files are
//! excluded so an operator-touched or NFS-skewed sibling can't demote the just-opened
//! log out of the top-`keep` slots (trade-off: they don't count toward the cap). A
//! racing single-file delete is a benign `NotFound`; an operator whole-tree wipe
//! can't race a running producer because its dispatcher gates on
//! `has_active_train_for`/`has_active_convert_for`.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// `.jsonl` logs kept per workspace per tree.
pub const LOG_RETENTION_KEEP_COUNT: usize = 10;

/// Outcome of one [`enforce_keep_last_n`] sweep, forwarded to the metrics hook.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RetentionReport {
    /// `.jsonl` files unlinked; raced `NotFound` is a no-op, not counted.
    pub pruned: u64,
    /// Per-entry failures (probe or unlink other than `NotFound`).
    pub failures: u64,
}

/// Keep the newest `keep` `.jsonl` files in `dir`, unlinking older ones.
/// Best-effort: never propagates `io::Error` (must not fail a producer hot path);
/// probe/`read_dir` failures land in `failures` rather than removing. `keep == 0`
/// clears every `.jsonl`, so production callers must pass non-zero to avoid
/// unlinking the just-opened log.
pub fn enforce_keep_last_n(dir: &Path, keep: usize) -> RetentionReport {
    enforce_keep_last_n_excluding(dir, keep, None)
}

/// Variant exempting one path: the equal-mtime tiebreak is filesystem-defined
/// dir-iter order and may sort the just-opened log into the delete tail, leaving
/// the producer writing an orphan inode that vanishes on close. `exempt` reserves
/// one slot (candidates sweep to `keep-1` survivors) so total population stays
/// at `keep`.
pub(crate) fn enforce_keep_last_n_excluding(
    dir: &Path,
    keep: usize,
    exempt: Option<&Path>,
) -> RetentionReport {
    // Single snapshot so every per-entry mtime comparison shares one reference.
    let now = SystemTime::now();
    let mut report = RetentionReport::default();
    let candidate_keep = if exempt.is_some() {
        keep.saturating_sub(1)
    } else {
        keep
    };
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return report,
        Err(e) => {
            tracing::warn!(
                target: "file_mgr",
                err = %e,
                path = %dir.display(),
                "log retention: read_dir failed",
            );
            report.failures += 1;
            return report;
        }
    };
    let mut candidates: Vec<(PathBuf, SystemTime)> = Vec::new();
    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!(
                    target: "file_mgr",
                    err = %e,
                    parent = %dir.display(),
                    "log retention: dir-iter entry failed",
                );
                report.failures += 1;
                continue;
            }
        };
        let path = entry.path();
        if !path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|s| s.ends_with(".jsonl"))
        {
            continue;
        }
        let metadata = match entry.metadata() {
            Ok(m) => m,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => {
                tracing::warn!(
                    target: "file_mgr",
                    err = %e,
                    path = %path.display(),
                    "log retention: metadata probe failed",
                );
                report.failures += 1;
                continue;
            }
        };
        // `DirEntry::metadata` does NOT follow symlinks, so a symlink to a regular
        // file is skipped here by `!is_file()`.
        if !metadata.file_type().is_file() {
            continue;
        }
        let mtime = match metadata.modified() {
            Ok(m) => m,
            Err(e) => {
                tracing::warn!(
                    target: "file_mgr",
                    err = %e,
                    path = %path.display(),
                    "log retention: mtime probe failed",
                );
                report.failures += 1;
                continue;
            }
        };
        // Future-mtime would sort above the just-opened log and push it into the
        // delete tail; skip (survives but doesn't count toward `keep`).
        if mtime > now {
            continue;
        }
        // Compare by file_name only: a whole-path compare would let `./<id>.jsonl`
        // vs `read_dir`'s absolute path slip past and sweep the producer's own log.
        if exempt.and_then(|p| p.file_name()) == path.file_name() {
            continue;
        }
        candidates.push((path, mtime));
    }
    if candidates.len() <= candidate_keep {
        return report;
    }
    // mtime DESCENDING (newest first); ties keep dir-iter order.
    candidates.sort_by(|(_, a), (_, b)| b.cmp(a));
    for (path, _) in candidates.into_iter().skip(candidate_keep) {
        match std::fs::remove_file(&path) {
            Ok(()) => report.pruned += 1,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // Raced an operator delete; already gone, no-op.
            }
            Err(e) => {
                tracing::warn!(
                    target: "file_mgr",
                    err = %e,
                    path = %path.display(),
                    "log retention: remove failed",
                );
                report.failures += 1;
            }
        }
    }
    // Hook short-circuits at `(0, 0)`, so no-op sweeps incur zero metrics.
    super::metrics_hooks::emit_logs_pruned(report.pruned, report.failures);
    report
}

#[cfg(test)]
mod tests {
    // Fixtures stage with raw `std::fs::*`; atomic-helper guards don't apply.
    #![allow(clippy::disallowed_methods)]
    use super::*;
    use std::fs;
    use std::time::{Duration, UNIX_EPOCH};

    fn write_log(dir: &Path, name: &str) -> PathBuf {
        let p = dir.join(name);
        fs::write(&p, b"{}\n").expect("write fixture");
        p
    }

    fn set_mtime(path: &Path, offset_from_now: i64) {
        let target = if offset_from_now >= 0 {
            SystemTime::now()
                .checked_add(Duration::from_secs(offset_from_now as u64))
                .expect("forward mtime")
        } else {
            SystemTime::now()
                .checked_sub(Duration::from_secs((-offset_from_now) as u64))
                .expect("back mtime")
        };
        let secs = target
            .duration_since(UNIX_EPOCH)
            .expect("post-epoch")
            .as_secs();
        let ft = filetime::FileTime::from_unix_time(secs as i64, 0);
        filetime::set_file_mtime(path, ft).expect("set mtime");
    }

    #[test]
    fn missing_dir_is_no_op() {
        let tmp = tempfile::tempdir().unwrap();
        let report = enforce_keep_last_n(&tmp.path().join("absent"), 5);
        assert_eq!(report, RetentionReport::default());
    }

    #[test]
    fn empty_dir_is_no_op() {
        let tmp = tempfile::tempdir().unwrap();
        let report = enforce_keep_last_n(tmp.path(), 5);
        assert_eq!(report, RetentionReport::default());
    }

    #[test]
    fn under_cap_is_no_op() {
        let tmp = tempfile::tempdir().unwrap();
        for i in 0..3 {
            write_log(tmp.path(), &format!("job-{i}.jsonl"));
        }
        let report = enforce_keep_last_n(tmp.path(), 5);
        assert_eq!(report.pruned, 0);
        assert_eq!(report.failures, 0);
        assert_eq!(fs::read_dir(tmp.path()).unwrap().count(), 3);
    }

    #[test]
    fn over_cap_unlinks_oldest_first() {
        let tmp = tempfile::tempdir().unwrap();
        let a = write_log(tmp.path(), "a.jsonl");
        let b = write_log(tmp.path(), "b.jsonl");
        let c = write_log(tmp.path(), "c.jsonl");
        let d = write_log(tmp.path(), "d.jsonl");
        let e = write_log(tmp.path(), "e.jsonl");
        set_mtime(&a, -500);
        set_mtime(&b, -400);
        set_mtime(&c, -300);
        set_mtime(&d, -200);
        set_mtime(&e, -100);
        let report = enforce_keep_last_n(tmp.path(), 2);
        assert_eq!(report.pruned, 3);
        assert_eq!(report.failures, 0);
        assert!(!a.exists());
        assert!(!b.exists());
        assert!(!c.exists());
        assert!(d.exists(), "second-newest survives");
        assert!(e.exists(), "newest survives");
    }

    /// `keep == 0` is an explicit "clear all" with no clamp to 1.
    #[test]
    fn keep_zero_removes_every_jsonl() {
        let tmp = tempfile::tempdir().unwrap();
        write_log(tmp.path(), "a.jsonl");
        write_log(tmp.path(), "b.jsonl");
        let report = enforce_keep_last_n(tmp.path(), 0);
        assert_eq!(report.pruned, 2);
        assert!(!tmp.path().join("a.jsonl").exists());
        assert!(!tmp.path().join("b.jsonl").exists());
    }

    /// Gate is exact `.jsonl` suffix, not substring, so `*.jsonl.bak` survives.
    #[test]
    fn non_jsonl_files_survive() {
        let tmp = tempfile::tempdir().unwrap();
        let notes = tmp.path().join("notes.txt");
        let archive = tmp.path().join("archive.tar.gz");
        let weird = tmp.path().join("job.jsonl.bak");
        fs::write(&notes, b"hello").unwrap();
        fs::write(&archive, b"junk").unwrap();
        fs::write(&weird, b"junk").unwrap();
        write_log(tmp.path(), "a.jsonl");
        let report = enforce_keep_last_n(tmp.path(), 0);
        assert_eq!(report.pruned, 1, "only `.jsonl` files count");
        assert!(notes.exists());
        assert!(archive.exists());
        assert!(weird.exists());
    }

    /// File-type is checked after the extension gate, so a subdir named
    /// `*.jsonl` survives.
    #[test]
    fn subdirs_survive() {
        let tmp = tempfile::tempdir().unwrap();
        let sub = tmp.path().join("operator-stash");
        fs::create_dir(&sub).unwrap();
        let weird_dir = tmp.path().join("looks-like-a-log.jsonl");
        fs::create_dir(&weird_dir).unwrap();
        write_log(tmp.path(), "a.jsonl");
        let report = enforce_keep_last_n(tmp.path(), 0);
        assert_eq!(report.pruned, 1, "only the regular .jsonl file is reaped");
        assert!(sub.is_dir());
        assert!(weird_dir.is_dir(), "subdir named *.jsonl survives");
    }

    #[test]
    fn future_mtime_files_are_preserved_and_do_not_count_toward_cap() {
        let tmp = tempfile::tempdir().unwrap();
        let future_a = write_log(tmp.path(), "future-a.jsonl");
        let future_b = write_log(tmp.path(), "future-b.jsonl");
        set_mtime(&future_a, 3600);
        set_mtime(&future_b, 3600);
        let recent = write_log(tmp.path(), "recent.jsonl");
        let old = write_log(tmp.path(), "old.jsonl");
        set_mtime(&recent, -60);
        set_mtime(&old, -3600);
        // futures excluded: candidate set is {recent, old}, so keep=1 reaps `old`.
        let report = enforce_keep_last_n(tmp.path(), 1);
        assert_eq!(report.pruned, 1, "exactly one past file unlinked");
        assert_eq!(report.failures, 0);
        assert!(future_a.exists(), "future-stamped file survives");
        assert!(future_b.exists(), "second future-stamped file survives");
        assert!(recent.exists(), "newest past file survives the cap");
        assert!(!old.exists(), "oldest past file unlinked by the cap");
    }

    #[test]
    fn default_report_is_all_zero() {
        let r = RetentionReport::default();
        assert_eq!(r.pruned, 0);
        assert_eq!(r.failures, 0);
    }
}
