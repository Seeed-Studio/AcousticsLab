//! Lifecycle smoke matrix Row 1 (cold boot). The plan's acceptance
//! target is all subsystems healthy within 1 s; this test does not
//! time that bound. The daemon's `--check` mode boots, runs for
//! `--check-seconds`, prints one `StatusSnapshot` JSON, and exits 0/1
//! on health; that exit-code is the assertion and the snapshot walk
//! adds a granular regression diagnostic.

#[path = "daemon_helpers/mod.rs"]
mod daemon_helpers;

use std::time::Duration;

use daemon_helpers::{CheckProfile, launch_check_mode};

/// `--check-seconds 3` gives soft margin against the 5 s
/// `HEALTH_STALE_AFTER` floor: even with only the synchronous
/// registration heartbeat the freshest stamp is at worst ~3 s old at
/// the snapshot, and the 1 Hz refresh re-stamping at t~=1,2,3 keeps it
/// far fresher still. Panics dump the daemon's full stdout + stderr
/// for `cargo test` triage.
#[tokio::test]
async fn lifecycle_row1_cold_boot_all_subsystems_healthy() {
    let run = launch_check_mode(CheckProfile::default())
        .await
        .expect("acousticslabd --check launch must succeed");

    if run.exit_code != 0 || run.snapshot.is_none() {
        panic!(
            "acousticslabd --check failed (exit={}, elapsed={:?})\n\
             ===== STDOUT =====\n{}\n\
             ===== STDERR =====\n{}",
            run.exit_code, run.elapsed, run.stdout, run.stderr,
        );
    }

    let snap = run
        .snapshot
        .as_ref()
        .expect("snapshot already non-None per check above");

    // Five subsystems register at boot; inference's heartbeat pump runs
    // even when --no-inference'd. The walk names the offending one
    // instead of leaving a bare exit-1.
    let subsystems = snap
        .get("subsystems")
        .and_then(|v| v.as_object())
        .unwrap_or_else(|| {
            panic!(
                "snapshot missing 'subsystems' object; full snapshot:\n{}",
                serde_json::to_string_pretty(snap).unwrap_or_default(),
            )
        });
    assert!(
        subsystems.len() >= 5,
        "expected >=5 registered subsystems on cold boot, got {}: {:?}",
        subsystems.len(),
        subsystems.keys().collect::<Vec<_>>(),
    );
    for (name, view) in subsystems {
        let healthy = view
            .get("healthy")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let stale = view.get("stale").and_then(|v| v.as_bool()).unwrap_or(true);
        assert!(
            healthy,
            "subsystem {name:?} reported unhealthy on cold boot: {view}\n\
             ===== STDOUT =====\n{}\n\
             ===== STDERR =====\n{}",
            run.stdout, run.stderr,
        );
        assert!(
            !stale,
            "subsystem {name:?} stale on cold boot (heartbeat older than \
             HEALTH_STALE_AFTER): {view}\n\
             ===== STDOUT =====\n{}\n\
             ===== STDERR =====\n{}",
            run.stdout, run.stderr,
        );
    }

    // Process lifetime is dominated by the --check-seconds tail, so the
    // boot phase can't be isolated; this wall-clock cap only catches a
    // hung boot.
    let total_budget = Duration::from_secs(10);
    assert!(
        run.elapsed < total_budget,
        "acousticslabd --check elapsed {:?} > {:?} budget; cold-boot regression?",
        run.elapsed,
        total_budget,
    );
}
