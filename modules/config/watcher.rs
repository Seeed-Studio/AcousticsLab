//! Hot-reload worker plus shared config-persist / error-mapping helpers for
//! [`crate::config::ConfigCell::watch_with`].

use crate::config::Config;
use crate::config::ReloadCallback;
use crate::config::error::{ConfigError, ConfigValidationError, write_err};
use arc_swap::ArcSwap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::Ordering;

/// Outcome of one [`try_reload`] cycle; each counter bumps once per outcome so
/// a double-catch path cannot double-count.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) enum ReloadOutcome {
    /// Validated + swapped into `inner`; bumps `reload_count`, clears the latch.
    Applied,
    /// Byte-identical to running snapshot; `validate()` skipped (validity
    /// transitive), provably valid so clears the latch; no bump.
    EqualToRunning,
    /// Transient `NotFound` from an atomic-write rename window; no bump, latch
    /// unchanged (validity unknown). A permanently-deleted file is
    /// indistinguishable, so a missing config is NOT flagged.
    NoChange,
    /// Parse/validate/callback rejected; prior snapshot retained. Bumps
    /// `reload_rejection_count` only on TRANSITION into Rejected so a persistent
    /// failure doesn't spam the metric once per notify event.
    Rejected,
}

pub(crate) fn debounce_reload_worker(
    path: Arc<PathBuf>,
    inner: Arc<ArcSwap<Config>>,
    reload_count: Arc<std::sync::atomic::AtomicU64>,
    reload_rejection_count: Arc<std::sync::atomic::AtomicU64>,
    mutate_lock: Arc<parking_lot::Mutex<()>>,
    on_reload: Arc<ReloadCallback>,
    rx: std::sync::mpsc::Receiver<()>,
) {
    use std::sync::mpsc::RecvTimeoutError;
    use std::time::Duration;

    /// Quiet window after the last FS event before the burst settles; absorbs
    /// editor multi-write saves.
    const WATCHER_DEBOUNCE: Duration = Duration::from_millis(100);

    // Latch so one persistent rejection bumps once: provably-valid outcomes
    // clear it; `NoChange` leaves it so a momentary gap doesn't split an episode.
    let mut in_rejection_episode = false;
    loop {
        // `Err` == all senders dropped == clean shutdown.
        if rx.recv().is_err() {
            return;
        }
        // Coalesce: extend the deadline as kicks arrive so reload runs once the
        // FS goes quiet, regardless of burst length.
        loop {
            match rx.recv_timeout(WATCHER_DEBOUNCE) {
                Ok(()) => continue,
                Err(RecvTimeoutError::Timeout) => break,
                Err(RecvTimeoutError::Disconnected) => return,
            }
        }
        // Outer catch covers panics the inner misses (fs read/parse/Arc::new/log
        // alloc): a killed worker kills hot-reload forever and leaks the unbounded
        // kick channel. `AssertUnwindSafe` sound: no shared-mutable borrow crosses.
        let raw = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            try_reload(&path, &inner, &mutate_lock, on_reload.as_ref())
        }));
        let outcome = match raw {
            Ok(o) => o,
            Err(payload) => {
                let msg = crate::common::error::panic_payload_to_string(payload.as_ref());
                tracing::error!(
                    target: "config",
                    path = %path.display(),
                    panic = %msg,
                    "config watcher: try_reload panicked (outside callback catch_unwind); reload discarded, worker continues",
                );
                ReloadOutcome::Rejected
            }
        };
        match outcome {
            ReloadOutcome::Applied => {
                reload_count.fetch_add(1, Ordering::Relaxed);
                in_rejection_episode = false;
            }
            ReloadOutcome::EqualToRunning => {
                in_rejection_episode = false;
            }
            ReloadOutcome::Rejected => {
                if !in_rejection_episode {
                    reload_rejection_count.fetch_add(1, Ordering::Relaxed);
                    in_rejection_episode = true;
                }
            }
            ReloadOutcome::NoChange => {}
        }
    }
}

/// Read-parse-validate-swap cycle; errors are logged not returned (background
/// thread, only action is log + keep previous). Holds `mutate_lock` across the
/// whole cycle (lock-first read-second) so it can't read a file mid-write by an
/// API mutation already holding the lock and clobber its value.
pub(crate) fn try_reload(
    path: &Path,
    inner: &ArcSwap<Config>,
    mutate_lock: &parking_lot::Mutex<()>,
    on_reload: &dyn Fn(&Config) -> Result<(), ConfigValidationError>,
) -> ReloadOutcome {
    let _guard = mutate_lock.lock();
    // IN_MUTATE sentinel under `mutate_lock` like API holders: else an
    // `on_reload` re-entering `config.mutate` deadlocks on the non-reentrant lock
    // instead of surfacing ReentrantMutate. `ResetOnDrop` clears it on all return
    // paths, including the callback catch_unwind panic.
    let _was_in_mutate = crate::config::IN_MUTATE.with(|c| c.replace(true));
    let _reset = crate::config::ResetOnDrop;
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => {
            // NotFound straddles an atomic-write rename window (transient);
            // anything else (EPERM/EROFS/EACCES) is persistent and bumps the
            // rejection count so alerts fire instead of silently serving stale.
            let outcome = if e.kind() == std::io::ErrorKind::NotFound {
                ReloadOutcome::NoChange
            } else {
                ReloadOutcome::Rejected
            };
            tracing::warn!(
                target: "config",
                path = %path.display(),
                err = %e,
                outcome = ?outcome,
                "config reload: read failed; keeping previous snapshot",
            );
            return outcome;
        }
    };
    let cfg: Config = match toml::from_str(&text) {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(
                target: "config",
                path = %path.display(),
                err = %e,
                "config reload: parse failed; keeping previous snapshot",
            );
            return ReloadOutcome::Rejected;
        }
    };
    // Short-circuit unchanged saves before `validate()`/Arc alloc: byte-identical
    // implies transitively valid; `EqualToRunning` (not `NoChange`) clears latch.
    let prev = inner.load_full();
    if *prev == cfg {
        return ReloadOutcome::EqualToRunning;
    }
    drop(prev);
    if let Err(e) = cfg.validate() {
        tracing::warn!(
            target: "config",
            path = %path.display(),
            err = %e,
            "config reload: validation failed; keeping previous snapshot",
        );
        return ReloadOutcome::Rejected;
    }
    // Callback-first (launch-catalogue cross-validation the `config` crate can't
    // reach), store to `inner` only on Ok; `catch_unwind` keeps the worker alive
    // on callback panic (treated as Err), `AssertUnwindSafe` sound by the
    // no-shared-mutable callback contract. Alloc the Arc BEFORE the catch so an
    // OOM-panic aborts before any side effect commits, never stranding ArcSwaps
    // at NEW while `inner` stays at OLD.
    let arc = Arc::new(cfg);
    let cb_result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| on_reload(arc.as_ref())));
    match cb_result {
        Ok(Ok(())) => {
            inner.store(arc);
            tracing::info!(
                target: "config",
                path = %path.display(),
                "config reloaded from disk",
            );
            ReloadOutcome::Applied
        }
        Ok(Err(diag)) => {
            tracing::warn!(
                target: "config",
                path = %path.display(),
                err = %diag,
                "config reload rejected by callback; keeping prior snapshot",
            );
            ReloadOutcome::Rejected
        }
        Err(payload) => {
            let msg = crate::common::error::panic_payload_to_string(payload.as_ref());
            // A callback panic between its first side-effect commit and `Ok`
            // leaves `inner` at OLD while a committed ArcSwap sits at NEW (`GET
            // /config` disagrees with the live cell); pre-allocation shrinks this
            // window to non-allocating internals + a Versioned carrier OOM.
            tracing::error!(
                target: "config",
                panic = %msg,
                "config reload callback panicked after catch_unwind boundary; inner.store(NEW) skipped, prior snapshot retained",
            );
            ReloadOutcome::Rejected
        }
    }
}

/// Serialize TOML and delegate to `file_mgr::fs_atomic::put_atomic` so the
/// durability protocol (tempfile + flush + sync_all + persist + parent fsync)
/// lives in one place.
pub(crate) fn write_toml_atomically(path: &Path, cfg: &Config) -> Result<(), ConfigError> {
    let text = toml::to_string_pretty(cfg)?;
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    if !dir.exists() {
        std::fs::create_dir_all(dir).map_err(|e| write_err(dir.display(), e))?;
    }
    crate::file_mgr::fs_atomic::put_atomic(path, text.as_bytes()).map_err(file_to_config_err)
}

/// Map the `FileError` variants `put_atomic` returns into parallel
/// `ConfigError` shapes; other variants are unreachable today but degrade
/// gracefully (keep path/Display context) if a future widening reaches them.
/// `pub(crate)` so `config::launch` can share the mapping (only out-of-file caller).
pub(crate) fn file_to_config_err(err: crate::file_mgr::FileError) -> ConfigError {
    use crate::file_mgr::FileError;
    match err {
        FileError::Io { path, source } => write_err(path, source),
        FileError::Persist(e) => ConfigError::Persist(e),
        FileError::SchemaTooNew { ref path, .. }
        | FileError::SchemaTooOld { ref path, .. }
        | FileError::MetadataParse { ref path, .. }
        | FileError::MetadataTooLarge { ref path, .. } => {
            write_err(path.clone(), std::io::Error::other(format!("{err}")))
        }
        other => write_err("<no path>", std::io::Error::other(format!("{other}"))),
    }
}
