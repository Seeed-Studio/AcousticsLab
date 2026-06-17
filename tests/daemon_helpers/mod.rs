//! Daemon integration-test harness: boots the production `acousticsd` binary as a
//! child process. Out-of-process to exercise the real `main()` (panic-hook/allocator/
//! tokio init an in-process call would inherit pre-set) and because lifecycle tests kill
//! it mid-flight; `--mock-audio` + `--no-inference` run it on macOS/Linux-CI without sound/NPU.

// Each integration-test binary compiles this module independently, so a binary
// using only some helpers flags the rest dead; module-wide allow avoids per-item attrs.
#![allow(dead_code)]

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

/// Untyped JSON view of the daemon's status snapshot: the real type is
/// `Serialize`-only, so the harness parses output as a generic `Value`.
pub type StatusSnapshotJson = serde_json::Value;

#[derive(Debug)]
pub struct CheckRun {
    /// 0 = all subsystems healthy at end of window, 1 = some unhealthy.
    pub exit_code: i32,
    /// `None` when the daemon crashed before the snapshot-print step (a boot
    /// regression, distinct from a subsystem-unhealthy signal).
    pub snapshot: Option<StatusSnapshotJson>,
    pub stderr: String,
    pub stdout: String,
    /// Spawn-to-exit wall-clock; soft regression guard on time-to-first-snapshot.
    pub elapsed: Duration,
}

#[derive(Debug, Clone)]
pub struct CheckProfile {
    /// 3 (vs the daemon's default 5) keeps CI fast; long enough to reach the snapshot tick.
    pub check_seconds: u64,
    /// Synth audio; required on hosts without sound hardware.
    pub mock_audio: bool,
    /// Skip `InferenceEngine`; required on macOS dev hosts (no `librknnrt`).
    pub no_inference: bool,
    /// Kills past this so a `--check`-mode wedge fails fast instead of hanging to cargo's timeout.
    pub timeout: Duration,
    /// Default `127.0.0.1:0` (ephemeral port) so parallel test binaries don't race the
    /// production default and fail boot with "Address already in use".
    pub tcp_bind: String,
    /// Pre-write `<cwd>/misc/launch.toml` before spawn; `None` auto-creates a stock config.
    pub launch_toml_override: Option<String>,
    pub extra_args: Vec<String>,
    /// `Some` lets a second invocation read state a prior daemon wrote, so it must
    /// outlive the first daemon (caller owns cleanup).
    pub cwd_override: Option<PathBuf>,
}

impl Default for CheckProfile {
    fn default() -> Self {
        Self {
            check_seconds: 3,
            mock_audio: true,
            no_inference: true,
            timeout: Duration::from_secs(15),
            tcp_bind: "127.0.0.1:0".into(),
            launch_toml_override: None,
            extra_args: Vec::new(),
            cwd_override: None,
        }
    }
}

/// Spawn `acousticsd --check ...`, wait for exit, return a [`CheckRun`]; with no
/// `cwd_override` the child runs in a fresh tempdir dropped at fn exit.
pub async fn launch_check_mode(profile: CheckProfile) -> Result<CheckRun> {
    // `_tmpdir_guard` owns+deletes the TempDir; override branch is `None` so the
    // caller-owned dir outlives this fn.
    let (cwd, _tmpdir_guard): (PathBuf, Option<tempfile::TempDir>) =
        match profile.cwd_override.as_ref() {
            Some(path) => (path.clone(), None),
            None => {
                let td = tempfile::tempdir().context("tempdir for daemon cwd")?;
                let path = td.path().to_path_buf();
                (path, Some(td))
            }
        };
    let misc_dir = cwd.join("misc");
    std::fs::create_dir_all(&misc_dir).context("create cwd/misc/ for daemon configs")?;
    if let Some(toml) = profile.launch_toml_override.as_deref() {
        // Pre-boot fixture write needs no atomicity (no concurrent reader), so the
        // clippy.toml ban on `std::fs::write` (prod uses `file_mgr::put_atomic`) is allowed here.
        #[allow(clippy::disallowed_methods)]
        std::fs::write(misc_dir.join("launch.toml"), toml)
            .context("pre-write misc/launch.toml fixture")?;
    }
    // `--config` is explicit: the flag has no default, so the harness owns the path.
    let workspace_dir = cwd.join("workspace");
    let launch_path = misc_dir.join("launch.toml");
    let bin = PathBuf::from(env!("CARGO_BIN_EXE_acousticsd"));
    let mut cmd = tokio::process::Command::new(&bin);
    cmd.current_dir(&cwd)
        .arg("--workspace")
        .arg(&workspace_dir)
        .arg("--config")
        .arg(&launch_path)
        .arg("--check")
        .arg("--check-seconds")
        .arg(profile.check_seconds.to_string())
        .arg("--tcp-bind")
        .arg(&profile.tcp_bind)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .stdin(std::process::Stdio::null())
        // Safety net for the panic-unwind path; explicit kill+wait below handles timeout.
        .kill_on_drop(true);
    if profile.mock_audio {
        cmd.arg("--mock-audio");
    }
    if profile.no_inference {
        cmd.arg("--no-inference");
    }
    for extra in &profile.extra_args {
        cmd.arg(extra);
    }

    let started = Instant::now();
    // Explicit spawn (not `cmd.output()`) keeps `child` for synchronous kill()+wait()+reap
    // past the timeout race; `output()`'s `kill_on_drop` doesn't block-wait, leaking a zombie.
    let mut child = cmd
        .spawn()
        .with_context(|| format!("spawn daemon binary at {}", bin.display()))?;
    let mut stdout_pipe = child
        .stdout
        .take()
        .context("spawned daemon child missing stdout pipe (set Stdio::piped above)")?;
    let mut stderr_pipe = child
        .stderr
        .take()
        .context("spawned daemon child missing stderr pipe (set Stdio::piped above)")?;
    let stdout_handle: tokio::task::JoinHandle<std::io::Result<Vec<u8>>> =
        tokio::spawn(async move {
            use tokio::io::AsyncReadExt;
            let mut buf = Vec::new();
            stdout_pipe.read_to_end(&mut buf).await?;
            Ok(buf)
        });
    let stderr_handle: tokio::task::JoinHandle<std::io::Result<Vec<u8>>> =
        tokio::spawn(async move {
            use tokio::io::AsyncReadExt;
            let mut buf = Vec::new();
            stderr_pipe.read_to_end(&mut buf).await?;
            Ok(buf)
        });

    let status = match tokio::time::timeout(profile.timeout, child.wait()).await {
        Ok(r) => r.with_context(|| format!("wait() on daemon binary at {}", bin.display()))?,
        Err(_) => {
            // SIGKILL then wait() to drain the zombie before returning.
            let _ = child.kill().await;
            let _ = child.wait().await;
            // Bound the reader join so a reader stuck mid-read on a just-SIGKILLed
            // pipe writer can't wedge the harness.
            let reader_budget = Duration::from_millis(200);
            let _ = tokio::time::timeout(reader_budget, stdout_handle).await;
            let _ = tokio::time::timeout(reader_budget, stderr_handle).await;
            return Err(anyhow::anyhow!(
                "acousticsd --check exceeded {} ms timeout (cwd {}); \
                 likely a boot regression that wedges in --check mode",
                profile.timeout.as_millis(),
                cwd.display(),
            ));
        }
    };
    let elapsed = started.elapsed();

    // Reader tasks drain to EOF once the kernel closes the pipe write-ends on
    // process exit; surface a reader-task panic as an error rather than swallow it.
    let stdout_bytes = stdout_handle
        .await
        .context("stdout-reader task join")?
        .context("stdout-reader read_to_end")?;
    let stderr_bytes = stderr_handle
        .await
        .context("stderr-reader task join")?
        .context("stderr-reader read_to_end")?;

    let stdout = String::from_utf8_lossy(&stdout_bytes).into_owned();
    let stderr = String::from_utf8_lossy(&stderr_bytes).into_owned();
    let exit_code = status.code().unwrap_or(-1);
    let snapshot = parse_last_status_snapshot(&stdout);
    Ok(CheckRun {
        exit_code,
        snapshot,
        stderr,
        stdout,
        elapsed,
    })
}

/// Extract the snapshot from `stdout` as the span from the last bare-`{` line to the
/// last bare-`}` line; tracing lines prefix a timestamp so never start at column 0.
/// None on parse failure.
fn parse_last_status_snapshot(stdout: &str) -> Option<StatusSnapshotJson> {
    let mut start = None;
    let mut end = None;
    for (idx, line) in stdout.lines().enumerate() {
        if line == "{" {
            start = Some(idx);
        }
        if line == "}" {
            end = Some(idx);
        }
    }
    let (s, e) = (start?, end?);
    if e <= s {
        return None;
    }
    let block: String = stdout
        .lines()
        .skip(s)
        .take(e - s + 1)
        .collect::<Vec<_>>()
        .join("\n");
    serde_json::from_str(&block).ok()
}

// Long-running harness: runs the daemon long enough to be killed mid-flight, then
// inspects workspace state to verify `file_mgr::put_atomic` write discipline holds across the kill.

use std::os::unix::process::ExitStatusExt;
use tokio::io::{AsyncBufReadExt, BufReader};

/// Drop terminates the child via `kill_on_drop` if the test panics before an explicit
/// `kill_*()` + `wait_exit_within()`.
#[derive(Debug)]
pub struct RunningDaemon {
    child: tokio::process::Child,
    /// Boot-wait stderr; diagnostic dump when a test expects clean shutdown but the daemon panicked.
    pub boot_stderr: String,
    /// Daemon cwd, owned here so it survives post-shutdown filesystem inspection.
    tmpdir: tempfile::TempDir,
}

impl RunningDaemon {
    pub fn cwd(&self) -> &std::path::Path {
        self.tmpdir.path()
    }

    pub fn pid(&self) -> nix::unistd::Pid {
        // `id()` is always `Some` between spawn() and wait()'s reap; a missing PID
        // is a tokio regression, not operator-actionable.
        let raw = self
            .child
            .id()
            .expect("daemon child must have a PID before kill_*()");
        nix::unistd::Pid::from_raw(raw as i32)
    }

    /// Send `SIGTERM`, triggering the daemon's trap + drain; pair with
    /// [`Self::wait_exit_within`] to confirm drain completed. `ESRCH` is Ok.
    pub fn kill_term(&self) -> anyhow::Result<()> {
        kill_tolerating_esrch(self.pid(), nix::sys::signal::Signal::SIGTERM, "SIGTERM")
    }

    /// Send `SIGINT` (the `kill(2)` analogue of terminal Ctrl-C), driving the same
    /// drain as [`Self::kill_term`]; reproduces the supervisor double-signal (SIGINT
    /// then forwarded SIGTERM) the escalator must debounce. `ESRCH` is Ok.
    pub fn kill_int(&self) -> anyhow::Result<()> {
        kill_tolerating_esrch(self.pid(), nix::sys::signal::Signal::SIGINT, "SIGINT")
    }

    /// Send `SIGKILL`, bypassing the signal handler; pair with
    /// [`Self::wait_exit_within`] to reap the zombie. `ESRCH` is Ok.
    pub fn kill_kill(&self) -> anyhow::Result<()> {
        kill_tolerating_esrch(self.pid(), nix::sys::signal::Signal::SIGKILL, "SIGKILL")
    }

    /// Wait for exit bounded by `budget`; on timeout, force-kill + reap to leave the OS process table clean.
    pub async fn wait_exit_within(mut self, budget: Duration) -> anyhow::Result<DaemonExit> {
        let outcome = tokio::time::timeout(budget, self.child.wait()).await;
        let status = match outcome {
            Ok(r) => r.map_err(anyhow::Error::from)?,
            Err(_) => {
                let _ = self.child.kill().await;
                let _ = self.child.wait().await;
                return Err(anyhow::anyhow!(
                    "daemon did not exit within {} ms; force-killed (cwd {})\n\
                     ===== BOOT STDERR =====\n{}",
                    budget.as_millis(),
                    self.tmpdir.path().display(),
                    self.boot_stderr,
                ));
            }
        };
        Ok(DaemonExit {
            exit_code: status.code(),
            terminating_signal: status.signal(),
            cwd: self.tmpdir,
            boot_stderr: self.boot_stderr,
        })
    }
}

/// Outcome of a long-running run terminated via signal or natural exit; `cwd` is
/// held here so the test can inspect post-shutdown filesystem state before auto-delete.
#[derive(Debug)]
pub struct DaemonExit {
    /// `None` if signal-terminated.
    pub exit_code: Option<i32>,
    /// `None` if natural exit.
    pub terminating_signal: Option<i32>,
    /// Daemon cwd; daemon-owned subtrees live under its `workspace/` (`--workspace <cwd>/workspace`).
    pub cwd: tempfile::TempDir,
    /// Boot-wait stderr; post-boot lines land in `cwd/workspace/logs/` via the appender.
    pub boot_stderr: String,
}

/// Spawn the daemon WITHOUT `--check` so it runs until signalled; blocks until boot
/// marker `"TCP listener bound"` appears in stderr (`Ok`) or the boot budget elapses
/// (`Err` with captured stderr). `profile.check_seconds` unused; `profile.timeout`
/// bounds the BOOT wait, not the run.
pub async fn launch_long_running(profile: CheckProfile) -> anyhow::Result<RunningDaemon> {
    let tmpdir = tempfile::tempdir().context("tempdir for daemon cwd")?;
    let misc_dir = tmpdir.path().join("misc");
    std::fs::create_dir_all(&misc_dir).context("create cwd/misc/")?;
    if let Some(toml) = profile.launch_toml_override.as_deref() {
        // Pre-boot fixture write needs no atomicity (no concurrent reader), so the
        // clippy.toml ban on `std::fs::write` is allowed here.
        #[allow(clippy::disallowed_methods)]
        std::fs::write(misc_dir.join("launch.toml"), toml)
            .context("pre-write misc/launch.toml fixture")?;
    }
    let workspace_dir = tmpdir.path().join("workspace");
    let launch_path = misc_dir.join("launch.toml");
    let bin = PathBuf::from(env!("CARGO_BIN_EXE_acousticsd"));
    let mut cmd = tokio::process::Command::new(&bin);
    cmd.current_dir(tmpdir.path())
        .arg("--workspace")
        .arg(&workspace_dir)
        .arg("--config")
        .arg(&launch_path)
        .arg("--tcp-bind")
        .arg(&profile.tcp_bind)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .stdin(std::process::Stdio::null())
        // Safety net for tests that panic before kill_*().
        .kill_on_drop(true);
    if profile.mock_audio {
        cmd.arg("--mock-audio");
    }
    if profile.no_inference {
        cmd.arg("--no-inference");
    }
    for extra in &profile.extra_args {
        cmd.arg(extra);
    }

    let mut child = cmd
        .spawn()
        .with_context(|| format!("spawn daemon binary at {}", bin.display()))?;
    let stderr_pipe = child
        .stderr
        .take()
        .context("spawned daemon child missing stderr pipe")?;

    // Read stderr until "TCP listener bound" -- the canonical marker that boot
    // reached the listening state, past which the daemon accepts requests and can
    // be signalled meaningfully.
    let boot_budget = profile.timeout;
    let mut reader = BufReader::new(stderr_pipe).lines();
    let mut boot_stderr = String::new();
    let bound_seen = tokio::time::timeout(boot_budget, async {
        loop {
            match reader.next_line().await {
                Ok(Some(line)) => {
                    boot_stderr.push_str(&line);
                    boot_stderr.push('\n');
                    if line.contains("TCP listener bound") {
                        return Ok::<bool, std::io::Error>(true);
                    }
                }
                Ok(None) => return Ok(false), // EOF -- daemon died
                Err(e) => return Err(e),
            }
        }
    })
    .await;
    match bound_seen {
        Ok(Ok(true)) => {}
        Ok(Ok(false)) => {
            let _ = child.kill().await;
            let _ = child.wait().await;
            return Err(anyhow::anyhow!(
                "daemon stderr closed before boot completed (cwd {})\n\
                 ===== STDERR =====\n{}",
                tmpdir.path().display(),
                boot_stderr,
            ));
        }
        Ok(Err(e)) => {
            let _ = child.kill().await;
            let _ = child.wait().await;
            return Err(anyhow::anyhow!(
                "stderr read error during boot wait (cwd {}): {e}",
                tmpdir.path().display(),
            ));
        }
        Err(_) => {
            let _ = child.kill().await;
            let _ = child.wait().await;
            return Err(anyhow::anyhow!(
                "daemon boot exceeded {} ms timeout; force-killed (cwd {})\n\
                 ===== STDERR =====\n{}",
                boot_budget.as_millis(),
                tmpdir.path().display(),
                boot_stderr,
            ));
        }
    }

    // Drain post-boot stderr to EOF so it doesn't backpressure the daemon; lines are
    // unneeded here (the cwd/workspace/logs/ appender captures them).
    tokio::spawn(async move {
        loop {
            match reader.next_line().await {
                Ok(Some(_)) => continue,
                _ => return,
            }
        }
    });

    Ok(RunningDaemon {
        child,
        boot_stderr,
        tmpdir,
    })
}

/// `ESRCH` (no such process) is silently OK: intent is "not running after this
/// returns", and the follow-up `wait_exit_within` reports the real exit shape, so a
/// crashed-before-signal regression still surfaces.
fn kill_tolerating_esrch(
    pid: nix::unistd::Pid,
    signal: nix::sys::signal::Signal,
    label: &'static str,
) -> anyhow::Result<()> {
    match nix::sys::signal::kill(pid, signal) {
        Ok(()) => Ok(()),
        Err(nix::errno::Errno::ESRCH) => Ok(()),
        Err(errno) => Err(anyhow::anyhow!("{label} via nix::kill: {errno}")),
    }
}
