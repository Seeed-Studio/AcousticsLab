//! SIGTERM triggers clean shutdown within the drain budget.
//!
//! Scope-reduced from "SIGTERM mid-fine-tune" (which needs HTTP
//! client + real backbone/dataset fixtures) to the always-available
//! signal-handler contract; guards the regression "drain
//! orchestration broke and the daemon now hangs on SIGTERM".

#[path = "daemon_helpers/mod.rs"]
mod daemon_helpers;

use std::time::Duration;

use daemon_helpers::{CheckProfile, launch_long_running};

#[tokio::test]
async fn lifecycle_row6_sigterm_clean_shutdown_within_drain_budget() {
    // `launch_long_running` ignores `check_seconds` (daemon runs
    // until signal); the profile `timeout` bounds the BOOT wait only.
    let daemon = launch_long_running(CheckProfile::default())
        .await
        .expect("daemon long-running launch must succeed");
    let cwd = daemon.cwd().to_path_buf();

    // Pin the trust-posture marker so a log rewrite that drops it fails.
    assert!(
        daemon
            .boot_stderr
            .contains("trust posture: no in-daemon auth"),
        "boot log must surface trust posture line; got boot stderr:\n{}",
        daemon.boot_stderr,
    );

    daemon.kill_term().expect("SIGTERM must succeed");

    // 12 s = 10 s drain outer cap (per-task Major tier is 5 s) + 2 s
    // for exit + zombie reap; overrun = a drain regression (task
    // ignoring the cancellation token, non-yielding await, etc.).
    let drain_budget = Duration::from_secs(12);
    let exit = daemon
        .wait_exit_within(drain_budget)
        .await
        .expect("daemon must exit within drain budget after SIGTERM");

    // Clean = exit 0, OR SIGTERM-terminated: tokio's `signal(SIGTERM)`
    // install races the "TCP listener bound" boot marker, so on a busy
    // host SIGTERM may land pre-handler -- still clean (nothing to leak
    // pre-listener). FAIL = nonzero exit OR any other signal (typically
    // SIGABRT from `process::abort`).
    let clean = exit.exit_code == Some(0)
        || exit.terminating_signal == Some(nix::sys::signal::Signal::SIGTERM as i32);
    assert!(
        clean,
        "daemon SIGTERM should produce clean exit (code 0 or SIGTERM-terminated); \
         got exit_code={:?}, terminating_signal={:?}\n\
         ===== BOOT STDERR =====\n{}",
        exit.exit_code, exit.terminating_signal, exit.boot_stderr,
    );

    // This test creates no workspaces, so a clean shutdown leaves the daemon's
    // workspaces dir (under `--workspace <cwd>/workspace`) absent or empty (no orphans).
    let workspaces_root = cwd.join("workspace").join("workspaces");
    if workspaces_root.exists() {
        let entries: Vec<_> = std::fs::read_dir(&workspaces_root)
            .expect("workspaces dir must be readable")
            .collect();
        assert!(
            entries.is_empty(),
            "workspaces/ must be empty after clean shutdown (no operator \
             created any workspaces); got {:?}",
            entries
                .iter()
                .map(|e| e.as_ref().map(|d| d.file_name()))
                .collect::<Vec<_>>(),
        );
    }
}
