//! SIGKILL filesystem-consistency regression: a force-killed daemon must leave
//! its auto-created config files committed-or-absent (never half-written), so a
//! respawn against the same cwd boots cleanly.
//!
//! Minimal viable form (the head-mid-write variant needs an HTTP client to POST
//! `/api/v1/active`): boot once (writes `<workspace>/config.toml` +
//! `misc/launch.toml` via `file_mgr::put_atomic`'s tempfile+atomic-rename),
//! SIGKILL it (no handler runs), respawn against the SAME cwd, assert the second
//! boot reads those files cleanly. Validates the schema gate + `put_atomic`
//! survive a force-kill.

#[path = "daemon_helpers/mod.rs"]
mod daemon_helpers;

use std::time::Duration;

use daemon_helpers::{CheckProfile, launch_check_mode, launch_long_running};

#[tokio::test]
async fn lifecycle_row7_sigkill_then_respawn_clean_boot() {
    let daemon = launch_long_running(CheckProfile::default())
        .await
        .expect("daemon long-running launch must succeed (phase 1)");
    // Snapshot cwd before `daemon` is consumed by wait_exit_within (whose
    // DaemonExit drop deletes the tempdir); phase 2 reuses this same directory.
    let cwd_path = daemon.cwd().to_path_buf();

    // SIGKILL bypasses the handler: kernel terminates immediately, no drain.
    daemon.kill_kill().expect("SIGKILL must succeed");

    // SIGKILL exits near-instantly (no userland cleanup); 2 s is generous.
    let exit = daemon
        .wait_exit_within(Duration::from_secs(2))
        .await
        .expect("daemon must reap within 2 s after SIGKILL");

    // Signal-terminated => no exit_code; signal is SIGKILL (identical macOS/Linux).
    assert_eq!(
        exit.terminating_signal,
        Some(nix::sys::signal::Signal::SIGKILL as i32),
        "expected SIGKILL termination; got exit_code={:?}, signal={:?}\n\
         ===== BOOT STDERR =====\n{}",
        exit.exit_code,
        exit.terminating_signal,
        exit.boot_stderr,
    );

    // `put_atomic`'s tempfile+atomic-rename means a SIGKILL between rename and
    // parent-dir fsync still leaves each file fully committed or fully absent;
    // a half-written file is the failure mode this excludes. Layout: user-pref
    // TOML is workspace-internal; launch TOML is `<cwd>/misc/`.
    let workspace_config = cwd_path.join("workspace/config.toml");
    let etc_launch = cwd_path.join("misc/launch.toml");
    assert!(
        workspace_config.exists(),
        "workspace/config.toml should survive SIGKILL (auto-created during phase 1 boot); \
         cwd={cwd_path:?}",
    );
    assert!(
        etc_launch.exists(),
        "misc/launch.toml should survive SIGKILL (auto-created during phase 1 boot); \
         cwd={cwd_path:?}",
    );
    // Non-zero size: a broken tempfile-then-rename could leave a 0-byte file.
    let config_size = std::fs::metadata(&workspace_config)
        .expect("metadata workspace/config.toml")
        .len();
    let launch_size = std::fs::metadata(&etc_launch)
        .expect("metadata launch.toml")
        .len();
    assert!(
        config_size > 0,
        "workspace/config.toml should be non-empty post-SIGKILL"
    );
    assert!(
        launch_size > 0,
        "misc/launch.toml should be non-empty post-SIGKILL"
    );

    let exit_cwd = exit.cwd;
    // `keep()` suppresses the TempDir's Drop deletion so the dir outlives `exit`
    // and phase 2 can reuse it; otherwise that deletion races phase 2's spawn.
    // Cleaned up manually at the end of the test.
    let leaked_cwd = exit_cwd.keep();
    assert_eq!(
        leaked_cwd, cwd_path,
        "leaked cwd must match snapshotted path"
    );

    // Phase 2: respawn `--check` against the SAME cwd via `cwd_override` so the
    // second boot reads the phase-1 files instead of auto-creating fresh ones.
    // Going through `launch_check_mode` keeps the spawn shape aligned with the
    // other lifecycle rows, so harness flag changes propagate here automatically.
    let phase2_profile = CheckProfile {
        cwd_override: Some(cwd_path.clone()),
        ..CheckProfile::default()
    };
    let phase2 = launch_check_mode(phase2_profile)
        .await
        .expect("phase 2 acousticslabd --check launch must succeed");
    assert_eq!(
        phase2.exit_code, 0,
        "phase 2 acousticslabd must exit 0 (clean re-boot using phase-1 misc/ files); \
         got exit={}\n\
         ===== STDOUT =====\n{}\n\
         ===== STDERR =====\n{}",
        phase2.exit_code, phase2.stdout, phase2.stderr,
    );

    let _ = std::fs::remove_dir_all(&cwd_path);
}
