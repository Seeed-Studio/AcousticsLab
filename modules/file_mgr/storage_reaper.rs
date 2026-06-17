//! Periodic sweep of orphan `.tmp/` entries (crashed uploads / unfinished
//! staged deletes) unreachable by the boot-time `recover_all` on a daemon that
//! `kill -9`s mid-operation and never restarts. Swept: `<root>/.tmp/`,
//! `<root>/active/.tmp/<activation_id>/` (pre-publish active-head staging,
//! atomic-renamed into `active/generations/` on success), and every
//! `<workspace>/.tmp/`; JSONL job logs and `acousticsd.log.*` are pruned
//! elsewhere.
//!
//! No `WorkspaceMgr` lock: the 24 h `tmp_age` default sits orders of magnitude
//! above any legitimate operation, and an inode race is benign since
//! `NamedTempFile::persist` sees `NotFound` and the producer rolls back as a
//! failed upload -- no torn write reaches the tree.
//!
//! Known limit: consults no JobRegistry, so a delete drain or activation
//! outliving `tmp_age` gets its fixed-mtime stage dir reaped mid-flight and
//! fails; safe today since real drains/activations are minutes. Lifting the
//! workspace-size cap needs a JobRegistry skip or a per-batch heartbeat touch
//! keeping the stage mtime fresh.
//!
//! Synchronous + blocking (callers wrap in `spawn_blocking`); cost is
//! `O(workspaces * |.tmp entries|)` syscalls, well under the 1 h period.

use crate::file_mgr::error::{FileError, io_err};
use crate::file_mgr::schema::{
    ROOT_TMP_DIR_NAME, active_staging_dir, root_tmp_dir, workspaces_dir,
};
use std::path::Path;
use std::time::{Duration, SystemTime};

#[derive(Clone, Copy, Debug)]
pub struct SweepConfig {
    /// Reap `.tmp/` entries whose mtime is older than this.
    pub tmp_age: Duration,
}

/// Outcome of one [`sweep_once`] pass; numbers feed `WorkspaceMetrics`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SweepReport {
    pub tmp_orphans_reaped: u64,
    pub workspaces_scanned: u64,
    /// Per-entry failures, logged then skipped so one corrupt entry cannot
    /// disable the reaper.
    pub failures: u64,
}

impl SweepReport {
    pub fn did_work(&self) -> bool {
        self.tmp_orphans_reaped > 0
    }
}

/// Per-workspace failures are isolated; only failure of the workspaces-root
/// walk itself propagates as `Err`.
pub fn sweep_once(root: &Path, cfg: &SweepConfig) -> Result<SweepReport, FileError> {
    let now = SystemTime::now();
    let mut report = SweepReport::default();

    sweep_dir_entries(&root_tmp_dir(root), now, cfg.tmp_age, &mut report);
    sweep_dir_entries(&active_staging_dir(root), now, cfg.tmp_age, &mut report);

    let workspaces = workspaces_dir(root);
    let entries = match std::fs::read_dir(&workspaces) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(report),
        Err(e) => return Err(io_err(workspaces.display(), e)),
    };
    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!(
                    target: "file_mgr",
                    err = %e,
                    parent = %workspaces.display(),
                    "storage reaper: workspaces dir-iter entry failed",
                );
                report.failures += 1;
                continue;
            }
        };
        let file_type = match entry.file_type() {
            Ok(ft) => ft,
            Err(e) => {
                tracing::warn!(
                    target: "file_mgr",
                    err = %e,
                    path = %entry.path().display(),
                    "storage reaper: workspace file_type probe failed",
                );
                report.failures += 1;
                continue;
            }
        };
        if !file_type.is_dir() {
            continue;
        }
        report.workspaces_scanned += 1;
        let ws = entry.path();
        sweep_workspace(&ws, now, cfg, &mut report);
    }
    Ok(report)
}

fn sweep_workspace(ws: &Path, now: SystemTime, cfg: &SweepConfig, report: &mut SweepReport) {
    sweep_dir_entries(&ws.join(ROOT_TMP_DIR_NAME), now, cfg.tmp_age, report);
}

/// Remove every direct entry in `dir` older than `age`; missing `dir` (lazy
/// mkdir) is a no-op, per-entry failures are logged and counted, not fatal.
fn sweep_dir_entries(dir: &Path, now: SystemTime, age: Duration, report: &mut SweepReport) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return,
        Err(e) => {
            tracing::warn!(
                target: "file_mgr",
                err = %e,
                path = %dir.display(),
                "storage reaper: read_dir failed",
            );
            report.failures += 1;
            return;
        }
    };
    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!(
                    target: "file_mgr",
                    err = %e,
                    parent = %dir.display(),
                    "storage reaper: dir-iter entry failed",
                );
                report.failures += 1;
                continue;
            }
        };
        let path = entry.path();
        let metadata = match entry.metadata() {
            Ok(m) => m,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => {
                tracing::warn!(
                    target: "file_mgr",
                    err = %e,
                    path = %path.display(),
                    "storage reaper: metadata probe failed",
                );
                report.failures += 1;
                continue;
            }
        };
        let mtime = match metadata.modified() {
            Ok(m) => m,
            Err(e) => {
                tracing::warn!(
                    target: "file_mgr",
                    err = %e,
                    path = %path.display(),
                    "storage reaper: mtime probe failed",
                );
                report.failures += 1;
                continue;
            }
        };
        // Future mtime (clock skew) errs `duration_since`; skip, don't reap.
        if !now.duration_since(mtime).is_ok_and(|d| d > age) {
            continue;
        }
        // `DirEntry::metadata` is `symlink_metadata`, so `file_type()` is the
        // entry not its target: a symlink is unlinked as-is, target untouched.
        // The explicit symlink branch stays safe even if `metadata` ever begins
        // following symlinks.
        let file_type = metadata.file_type();
        let res = if file_type.is_symlink() {
            std::fs::remove_file(&path)
        } else if file_type.is_dir() {
            std::fs::remove_dir_all(&path)
        } else {
            std::fs::remove_file(&path)
        };
        match res {
            Ok(()) => {
                report.tmp_orphans_reaped += 1;
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // Raced a producer that cleaned up between stat and remove.
            }
            Err(e) => {
                tracing::warn!(
                    target: "file_mgr",
                    err = %e,
                    path = %path.display(),
                    "storage reaper: remove failed",
                );
                report.failures += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    // Fixtures stage with raw `std::fs::*` + `filetime`; write-through-file_mgr
    // clippy constraint is setup-exempt.
    #![allow(clippy::disallowed_methods)]
    use super::*;
    use std::fs;
    use std::time::{Duration, UNIX_EPOCH};

    fn backdate(path: &Path, age: Duration) {
        let target = SystemTime::now()
            .checked_sub(age)
            .expect("backdate clock subtraction");
        set_mtime_at(path, target);
    }

    /// Forward-dated mtime exercises the clock-skew safety branch.
    fn forward_date(path: &Path, ahead: Duration) {
        let target = SystemTime::now()
            .checked_add(ahead)
            .expect("forward_date clock addition");
        set_mtime_at(path, target);
    }

    fn set_mtime_at(path: &Path, target: SystemTime) {
        let secs = target
            .duration_since(UNIX_EPOCH)
            .expect("post-epoch")
            .as_secs();
        let ft = filetime::FileTime::from_unix_time(secs as i64, 0);
        filetime::set_file_mtime(path, ft).expect("set mtime");
    }

    fn build_workspace_skeleton(root: &Path, id: &str) -> std::path::PathBuf {
        let ws = root.join("workspaces").join(id);
        fs::create_dir_all(ws.join(".tmp")).unwrap();
        ws
    }

    fn default_cfg() -> SweepConfig {
        SweepConfig {
            tmp_age: Duration::from_secs(24 * 3600),
        }
    }

    #[test]
    fn sweep_reaps_aged_root_tmp_file() {
        let tmp = tempfile::tempdir().unwrap();
        let root_tmp = tmp.path().join(".tmp");
        fs::create_dir_all(&root_tmp).unwrap();
        let stale = root_tmp.join("delete-workspace-stale.json");
        fs::write(&stale, b"{}").unwrap();
        backdate(&stale, Duration::from_secs(48 * 3600));

        let report = sweep_once(tmp.path(), &default_cfg()).expect("sweep");
        assert_eq!(report.tmp_orphans_reaped, 1);
        assert_eq!(report.failures, 0);
        assert!(!stale.exists());
        assert!(root_tmp.is_dir(), "parent .tmp/ itself stays in place");
    }

    #[test]
    fn sweep_skips_future_mtime_entries() {
        let tmp = tempfile::tempdir().unwrap();
        let root_tmp = tmp.path().join(".tmp");
        fs::create_dir_all(&root_tmp).unwrap();
        let future = root_tmp.join("forward-stamped.json");
        fs::write(&future, b"{}").unwrap();
        forward_date(&future, Duration::from_secs(3600));

        // 1 ns threshold: only the future-mtime skip keeps `future` alive.
        let cfg = SweepConfig {
            tmp_age: Duration::from_nanos(1),
        };
        let report = sweep_once(tmp.path(), &cfg).expect("sweep");
        assert_eq!(
            report.tmp_orphans_reaped, 0,
            "future-mtime entry must not be reaped",
        );
        assert_eq!(
            report.failures, 0,
            "future-mtime skip must not surface as a failure",
        );
        assert!(future.exists(), "future-stamped fixture survived sweep");
    }

    #[test]
    fn sweep_skips_fresh_root_tmp_file() {
        let tmp = tempfile::tempdir().unwrap();
        let root_tmp = tmp.path().join(".tmp");
        fs::create_dir_all(&root_tmp).unwrap();
        let fresh = root_tmp.join("delete-workspace-fresh.json");
        fs::write(&fresh, b"{}").unwrap();

        let report = sweep_once(tmp.path(), &default_cfg()).expect("sweep");
        assert_eq!(report.tmp_orphans_reaped, 0);
        assert!(fresh.exists());
    }

    #[test]
    fn sweep_reaps_aged_root_tmp_dir_tree() {
        let tmp = tempfile::tempdir().unwrap();
        let staging = tmp
            .path()
            .join(".tmp")
            .join("delete-workspace-stale")
            .join("payload");
        fs::create_dir_all(&staging).unwrap();
        fs::write(staging.join("a.bin"), b"junk").unwrap();
        let parent = staging.parent().unwrap();
        backdate(parent, Duration::from_secs(48 * 3600));

        let report = sweep_once(tmp.path(), &default_cfg()).expect("sweep");
        assert_eq!(report.tmp_orphans_reaped, 1);
        assert!(!parent.exists(), "whole subtree removed");
    }

    #[test]
    fn sweep_reaps_aged_per_workspace_tmp() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = build_workspace_skeleton(tmp.path(), "ws-a");
        let stale = ws.join(".tmp").join("delete-assets-stale.json");
        fs::write(&stale, b"{}").unwrap();
        backdate(&stale, Duration::from_secs(48 * 3600));

        let report = sweep_once(tmp.path(), &default_cfg()).expect("sweep");
        assert_eq!(report.tmp_orphans_reaped, 1);
        assert_eq!(report.workspaces_scanned, 1);
        assert!(!stale.exists());
    }

    #[test]
    fn sweep_no_op_on_fresh_root() {
        let tmp = tempfile::tempdir().unwrap();
        let report = sweep_once(tmp.path(), &default_cfg()).expect("sweep no-op");
        assert_eq!(report, SweepReport::default());
    }

    #[test]
    fn sweep_reaps_aged_active_staging() {
        let tmp = tempfile::tempdir().unwrap();
        let staging = tmp
            .path()
            .join("active")
            .join(".tmp")
            .join("00000000-0000-4000-8000-000000000001");
        fs::create_dir_all(&staging).unwrap();
        fs::write(staging.join("head.mpk"), b"junk").unwrap();
        backdate(&staging, Duration::from_secs(48 * 3600));

        let report = sweep_once(tmp.path(), &default_cfg()).expect("sweep");
        assert_eq!(report.tmp_orphans_reaped, 1);
        assert!(!staging.exists());
    }

    /// Backdate the symlink's own mtime; `set_file_mtime` follows the link.
    #[cfg(unix)]
    fn backdate_symlink(path: &Path, age: Duration) {
        let target = SystemTime::now()
            .checked_sub(age)
            .expect("backdate clock subtraction");
        let secs = target
            .duration_since(UNIX_EPOCH)
            .expect("post-epoch")
            .as_secs();
        let ft = filetime::FileTime::from_unix_time(secs as i64, 0);
        filetime::set_symlink_file_times(path, ft, ft).expect("set symlink mtime");
    }

    #[cfg(unix)]
    #[test]
    fn sweep_unlinks_symlinks_without_following() {
        let tmp = tempfile::tempdir().unwrap();
        let root_tmp = tmp.path().join(".tmp");
        fs::create_dir_all(&root_tmp).unwrap();
        let target_dir = tempfile::tempdir().unwrap();
        let target_file = target_dir.path().join("must-survive");
        fs::write(&target_file, b"outside-the-staging-tree").unwrap();
        let link_path = root_tmp.join("evil-link");
        std::os::unix::fs::symlink(&target_file, &link_path).unwrap();
        backdate_symlink(&link_path, Duration::from_secs(48 * 3600));

        let report = sweep_once(tmp.path(), &default_cfg()).expect("sweep");
        assert_eq!(report.tmp_orphans_reaped, 1);
        assert_eq!(report.failures, 0);
        // `exists()` follows symlinks; probe the link via `symlink_metadata`.
        assert!(
            fs::symlink_metadata(&link_path).is_err(),
            "symlink itself unlinked",
        );
        assert!(
            target_file.exists(),
            "symlink target outside .tmp/ must survive",
        );
    }

    #[cfg(unix)]
    #[test]
    fn sweep_unlinks_symlink_to_dir_without_recursing() {
        let tmp = tempfile::tempdir().unwrap();
        let root_tmp = tmp.path().join(".tmp");
        fs::create_dir_all(&root_tmp).unwrap();
        let target_dir = tempfile::tempdir().unwrap();
        let target_sentinel = target_dir.path().join("must-survive");
        fs::write(&target_sentinel, b"keep me").unwrap();
        let link_path = root_tmp.join("evil-dir-link");
        std::os::unix::fs::symlink(target_dir.path(), &link_path).unwrap();
        backdate_symlink(&link_path, Duration::from_secs(48 * 3600));

        let report = sweep_once(tmp.path(), &default_cfg()).expect("sweep");
        assert_eq!(report.tmp_orphans_reaped, 1);
        assert!(
            fs::symlink_metadata(&link_path).is_err(),
            "symlink-to-dir unlinked",
        );
        assert!(
            target_sentinel.exists(),
            "symlink-to-dir target must NOT be recursed into",
        );
    }

    #[test]
    fn did_work_predicate_matches_counters() {
        let mut r = SweepReport::default();
        assert!(!r.did_work());
        r.failures = 5;
        assert!(!r.did_work(), "failures alone are not 'work'");
        r.tmp_orphans_reaped = 1;
        assert!(r.did_work());
    }
}
