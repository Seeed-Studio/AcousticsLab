//! Second-signal escalator debounce: one Ctrl-C reaches the daemon as a
//! SIGINT-then-SIGTERM pair ms apart (the terminal delivers SIGINT to the whole
//! foreground process group AND the wrapper traps+forwards a SIGTERM) = ONE
//! shutdown event; pre-fix
//! the escalator misread the second as impatience and `process::exit(1)`
//! mid-drain, so the daemon now ignores a second signal within
//! `SECOND_SIGNAL_DEBOUNCE` (1 s) of drain start. Invariant under test: a rapid
//! double-signal never hard-exits (`Some(1)`)/aborts -- it drains (code 0) or,
//! on the handler-install boot race, dies by one of the signals. The pre-fix
//! exit reproduces only when the second signal lands after `wait_for_signal()`
//! arms (else select! swallows the simultaneous SIGTERM) but before drain ends
//! -- a load-drifting few-ms window, so the gap sweep is a best-effort
//! catch-biased probe (never false-fails with the fix), not a repro.

#[path = "daemon_helpers/mod.rs"]
mod daemon_helpers;

use std::time::Duration;

use daemon_helpers::{CheckProfile, RunningDaemon, launch_long_running};

/// One trial; `gap` between the SIGINT and SIGTERM probes a point in the
/// "second signal lands mid-drain" window.
async fn double_signal_exit(gap: Duration) -> (Option<i32>, Option<i32>, String) {
    let daemon: RunningDaemon = launch_long_running(CheckProfile::default())
        .await
        .expect("daemon long-running launch must succeed");

    // Boot marker fires before `wait_for_signal()` installs handlers; settling
    // biases SIGINT toward the trap+escalator path vs racing the install
    // (kernel-default kill) -- a probe bias, not a correctness dependency.
    tokio::time::sleep(Duration::from_millis(500)).await;

    daemon.kill_int().expect("SIGINT must succeed");
    tokio::time::sleep(gap).await;
    daemon.kill_term().expect("SIGTERM must succeed");

    // 10 s drain-registry outer cap + 2 s soft margin.
    let exit = daemon
        .wait_exit_within(Duration::from_secs(12))
        .await
        .expect("daemon must exit within drain budget after the double signal");
    (exit.exit_code, exit.terminating_signal, exit.boot_stderr)
}

#[tokio::test]
async fn rapid_sigint_then_sigterm_debounces_to_clean_drain() {
    let sigint = nix::sys::signal::Signal::SIGINT as i32;
    let sigterm = nix::sys::signal::Signal::SIGTERM as i32;

    // Geometric spread covering idle-fast through loaded-slow drain windows;
    // broad-but-sparse because the probe is best-effort.
    for gap_ms in [1u64, 2, 4, 8, 16, 32, 64] {
        let (exit_code, terminating_signal, boot_stderr) =
            double_signal_exit(Duration::from_millis(gap_ms)).await;

        // The escalator hard-exit is the only `Some(1)` on this path (the
        // arbitrator-wedge / drain-overrun exit(1) sites cannot fire in a
        // healthy sub-second drain), so `Some(1)` is the unique regression flag.
        assert_ne!(
            exit_code,
            Some(1),
            "gap={gap_ms}ms: rapid SIGINT+SIGTERM (one shutdown event) \
             hard-exited via the second-signal escalator instead of \
             debouncing to a clean drain; terminating_signal={terminating_signal:?}\n\
             ===== BOOT STDERR =====\n{boot_stderr}",
        );

        let acceptable = exit_code == Some(0)
            || terminating_signal == Some(sigint)
            || terminating_signal == Some(sigterm);
        assert!(
            acceptable,
            "gap={gap_ms}ms: double-signal shutdown should drain cleanly \
             (code 0) or, on the handler-install race, be terminated by \
             SIGINT/SIGTERM -- never abort or exit non-zero; \
             exit_code={exit_code:?}, terminating_signal={terminating_signal:?}\n\
             ===== BOOT STDERR =====\n{boot_stderr}",
        );
    }
}
