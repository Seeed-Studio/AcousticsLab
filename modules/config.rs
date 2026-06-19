//! Acoustics Lab configuration: TOML on disk (durable source of truth) ->
//! `Arc<ArcSwap<Config>>` snapshot in memory. Every mutation clones the
//! snapshot, mutates, atomic-writes the TOML, then `ArcSwap::store` -- write
//! BEFORE store so a failed write or a crash between the two leaves memory
//! consistent with disk, never losing the change.
//!
//! `watch` watches the parent directory (not the file) so vim-style
//! write-tmp+rename edits surface across inode swaps; sibling events are
//! filtered by `file_name()`. Self-writes re-enter `try_reload` but the
//! value-equality short-circuit elides the redundant store. Reload errors log
//! at `warn!` leaving the snapshot unchanged. Comments are NOT round-tripped.
//!
//! Type taxonomy: schema DTOs ([`Config`], [`LaunchConfig`] + sub-tables) are
//! `Serialize + Deserialize` with no runtime handles, validated by a
//! `validate(&self)` per leaf aggregated in [`Config::validate`]. Live stores
//! ([`ConfigCell`], [`MicSettingsCell`]) wrap a DTO in `ArcSwap`/`VersionedSwap` + mutate-lock
//! behind an object-safe trait ([`ConfigHandle`]/[`MicSettingsHandle`]) so
//! cross-crate consumers depend on the trait, not the concrete type.

#![warn(missing_debug_implementations)]

use std::cell::Cell;
use std::path::PathBuf;
use std::sync::Arc;

// Re-entrancy sentinel: `mutate_lock` is non-reentrant, so a same-thread callback
// re-entering `mutate_then` would deadlock; set while inside a body, the entry
// guard returns `ReentrantMutate` instead of blocking. Per-thread (a different
// thread waits on the lock rather than failing); drop always clears it (panic-safe).
thread_local! {
    static IN_MUTATE: Cell<bool> = const { Cell::new(false) };
}

mod domain;
mod error;
mod handle;
mod launch;
pub mod mic_settings;
mod watcher;

pub use domain::{ApiCfg, FileCfg, OutputCfg, OutputInferenceCfg, TrainingDefaults};
pub use error::{ConfigError, ConfigValidationError};
use error::{parse_err, read_err};
pub use handle::{ConfigGuard, ConfigHandle};
pub use launch::{
    DefaultHeadRef, HeadLaunchConfig, LaunchConfig, validate_policy_against_catalogue,
};
pub use mic_settings::{MicError, MicSettingsCell, MicSettingsHandle};

use crate::audio_io::mic_arbitrator::{ChannelSelection, MicPolicy, MicSelection};
use crate::inference::InferenceCfg;
use arc_swap::ArcSwap;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Trait-object form of the user-supplied `watch_with` callback.
type ReloadCallback = dyn Fn(&Config) -> Result<(), ConfigValidationError> + Send + Sync;

/// Top-level user-preference config: only fields BOTH hot-reloadable AND
/// API-mutable (boot-time constants live in [`LaunchConfig`]). The workspace root
/// is CLI-supplied, not persisted (a stored copy would drift from the CLI on next
/// boot). `deny_unknown_fields` fails closed on retired keys.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// Cross-validated at every entry point against the immutable launch-time
    /// [`LaunchConfig::mic`] catalogue.
    pub mic: MicPolicy,
    /// Engine reads the live ArcSwap every iteration, so edits take effect within
    /// one frame.
    pub inference: InferenceCfg,
}

impl Config {
    /// Aggregate validator run at every entry point materializing a `Config`
    /// (load/from_value/mutate_then/reload); every new tunable sub-section's
    /// `validate` MUST be chained here. `MicPolicy` cross-validation needs the
    /// catalogue `Arc` (out of crate reach) so runs separately via
    /// [`validate_policy_against_catalogue`].
    pub fn validate(&self) -> Result<(), ConfigValidationError> {
        self.inference
            .validate()
            .map_err(ConfigValidationError::Inference)?;
        Ok(())
    }

    /// Fresh-install user-preference defaults; [`LaunchConfig::default_for`] is
    /// the matching manifest.
    pub fn default_for() -> Self {
        Self {
            mic: MicPolicy {
                mic: MicSelection::FirstAvailable,
                channel: ChannelSelection::Auto,
            },
            inference: InferenceCfg::default(),
        }
    }
}

/// Owned config snapshot + atomic mutation surface; concrete [`ConfigHandle`].
/// `Clone` is cheap (one Arc bump), reads (`snapshot`) are wait-free atomic loads.
/// Writes serialize through `mutate_lock`: without it two callers each clone the
/// OLD snapshot, mutate disjoint fields, and last-writer-wins on disk+memory,
/// dropping the earlier change. Lock held only for clone+closure+rename+store.
#[derive(Clone, Debug)]
pub struct ConfigCell {
    inner: Arc<ArcSwap<Config>>,
    path: Arc<PathBuf>,
    /// Serializes the read-modify-write cycle; never on the hot read path.
    mutate_lock: Arc<parking_lot::Mutex<()>>,
    /// Successful watcher reloads, bumped only when the swapped value differs.
    reload_count: Arc<std::sync::atomic::AtomicU64>,
    /// REJECTED watcher reloads (parse/validation/callback-Err/panic), surfaced via
    /// `/api/v1/status`. A reject means the snapshot still reflects prior disk, so a
    /// later `mutate_then` (reads `inner`=OLD) silently overwrites the operator's
    /// NEW disk content with OLD-derived bytes.
    reload_rejection_count: Arc<std::sync::atomic::AtomicU64>,
}

impl ConfigCell {
    /// Load from `path`. Does NOT spawn the watcher; call `watch()` separately.
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        let text = std::fs::read_to_string(path).map_err(|e| read_err(path.display(), e))?;
        let cfg: Config = toml::from_str(&text).map_err(|e| parse_err(path.display(), e))?;
        Self::from_value(cfg, path.to_path_buf())
    }

    /// Build a handle around an in-memory `Config`; `path` is where `mutate`
    /// writes. Validates so an invalid config fails loudly at boot.
    pub fn from_value(cfg: Config, path: PathBuf) -> Result<Self, ConfigError> {
        cfg.validate().map_err(|e| ConfigError::Invalid {
            path: path.display().to_string(),
            msg: e.to_string(),
        })?;
        Ok(Self {
            inner: Arc::new(ArcSwap::from_pointee(cfg)),
            path: Arc::new(path),
            mutate_lock: Arc::new(parking_lot::Mutex::new(())),
            reload_count: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            reload_rejection_count: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        })
    }

    /// Persist the in-memory snapshot to disk (first-boot default materialization).
    pub fn persist(&self) -> Result<(), ConfigError> {
        let snap = self.snapshot();
        crate::config::watcher::write_toml_atomically(&self.path, &snap)?;
        Ok(())
    }

    /// `Arc<Config>` aliasing the current snapshot (wait-free).
    pub fn snapshot(&self) -> Arc<Config> {
        self.inner.load_full()
    }

    /// Apply `f` to a clone of the snapshot, then atomic-write + store; serialized
    /// against concurrent `mutate` / `mutate_then`.
    pub fn mutate<F>(&self, f: F) -> Result<(), ConfigError>
    where
        F: FnOnce(&mut Config),
    {
        self.mutate_then(
            |c| {
                f(c);
            },
            |_| (),
        )
    }

    /// Like [`Self::mutate`], but the mutator returns `R` and `after` runs while
    /// STILL HOLDING the mutate lock. Use `after` to keep an in-memory runtime
    /// ArcSwap (e.g. `mic_policy`) consistent with disk: collapsing the disk write
    /// and the runtime store into one critical section prevents two handlers
    /// committing disk in one order but `runtime.store` in the reverse (runtime drift).
    pub fn mutate_then<F, G, R>(&self, mutator: F, after: G) -> Result<R, ConfigError>
    where
        F: FnOnce(&mut Config) -> R,
        G: FnOnce(&Config),
    {
        // Structured error instead of a silent deadlock; the RAII guard clears the sentinel.
        let was_in_mutate = IN_MUTATE.with(|c| c.replace(true));
        if was_in_mutate {
            return Err(ConfigError::ReentrantMutate);
        }
        let _reset = ResetOnDrop;
        let _guard = self.mutate_lock.lock();
        let mut cfg = (*self.snapshot()).clone();
        let result = mutator(&mut cfg);
        self.validate_persist_swap(cfg, after)?;
        Ok(result)
    }

    /// Shared validate-persist-swap-after tail for `mutate_then` and
    /// `CellGuard::commit`. Caller MUST hold `mutate_lock` and have set `IN_MUTATE`.
    /// Validate runs FIRST so an invalid mutator never reaches disk.
    ///
    /// `after` runs AFTER persist + inner-store; a panic there leaves disk+`inner`
    /// at NEW while a side-effect ArcSwap the closure owned stays at OLD, diverging
    /// `GET /config` from the live cell. Callers MUST make `after`
    /// panic-safe-after-first-commit: hoist allocations out, leave only infallible
    /// ops (refcount bumps, `ArcSwap::store`) inside.
    fn validate_persist_swap(
        &self,
        cfg: Config,
        after: impl FnOnce(&Config),
    ) -> Result<(), ConfigError> {
        // Catches a future caller building `cfg` outside a mutate-lock section,
        // racing a peer mutator's read-modify-write.
        debug_assert!(
            IN_MUTATE.with(|c| c.get()),
            "validate_persist_swap invoked without IN_MUTATE; mutate_lock contract violated",
        );
        cfg.validate().map_err(|e| ConfigError::Invalid {
            path: self.path.display().to_string(),
            msg: e.to_string(),
        })?;
        // Allocate the Arc BEFORE persist so an OOM-panic aborts before the write
        // (disk+inner both OLD); the reverse would write disk=NEW then panic before
        // `inner.store`, stranding disk=NEW+inner=OLD until reload/restart reconciles.
        let arc = Arc::new(cfg);
        crate::config::watcher::write_toml_atomically(&self.path, arc.as_ref())?;
        self.inner.store(arc.clone());
        after(&arc);
        Ok(())
    }

    /// Spawn a notify watcher reloading the snapshot on each on-disk change;
    /// returns a guard owning the watcher thread (drop to stop). Equivalent to
    /// [`watch_with`](Self::watch_with) with a trivially-accepting callback.
    pub fn watch(&self) -> Result<WatcherGuard, ConfigError> {
        self.watch_with(|_| Ok(()))
    }

    /// Like [`Self::watch`] but invokes `on_reload` BEFORE committing the parsed
    /// config to `inner`; `Err` logs at `warn!` and discards the reload (`inner`
    /// unchanged) -- the daemon's hook to cross-validate user-pref policy against
    /// the launch catalogue before exposing it.
    ///
    /// The callback runs on the debounce worker WITH `mutate_lock` held, so on `Ok`
    /// its side effects (live-ArcSwap stores) commit before `inner` in one critical
    /// section other writers serialize behind. Callback contract: do NOT apply side
    /// effects on the `Err` path, and MUST NOT call back into
    /// `mutate`/`mutate_then`/re-emit a file write (same-thread `mutate_lock`
    /// re-acquire deadlocks); `snapshot()` and callback-owned ArcSwaps are safe.
    ///
    /// Debounce: editors emit several FS events per save (vim:
    /// truncate/write/rename/chmod), so the worker coalesces nudges in a 100 ms
    /// quiet window into one reload, holding `mutate_lock` across
    /// read+parse+validate+store so a file-edit reload can't read stale bytes
    /// mid-`write_toml_atomically` and clobber a concurrent API mutation's store.
    ///
    /// A panicking callback is `catch_unwind`-caught, logged at `error!`, treated
    /// as `Err` (reload discarded); the worker survives. The payload is dropped, so
    /// log inside the callback and design it never to panic (UnwindSafe-violating
    /// types may be left inconsistent).
    pub fn watch_with<F>(&self, on_reload: F) -> Result<WatcherGuard, ConfigError>
    where
        F: Fn(&Config) -> Result<(), ConfigValidationError> + Send + Sync + 'static,
    {
        use notify::{Event, RecursiveMode, Watcher};

        let path = self.path.clone();
        let inner = self.inner.clone();
        let reload_count = self.reload_count.clone();
        let reload_rejection_count = self.reload_rejection_count.clone();
        let mutate_lock = self.mutate_lock.clone();
        let on_reload: Arc<ReloadCallback> = Arc::new(on_reload);

        // Unbounded: bounded depth could stall the notify thread; << 1k events/s.
        let (kick_tx, kick_rx) = std::sync::mpsc::channel::<()>();

        // Gate by `file_name()` not full path: sidesteps platform canonicalization
        // (macOS `/tmp` <-> `/private/tmp`) breaking full-path equality on the
        // parent-dir watch.
        let target_name = self.path.file_name().map(|n| n.to_owned());

        let mut watcher =
            notify::recommended_watcher(move |res: notify::Result<Event>| match res {
                Ok(event) => {
                    let touches_target = match &target_name {
                        // Empty paths (pathless rescan/recovery) mean "you may have
                        // missed an event"; kick conservatively -- the value-equality
                        // short-circuit absorbs the false-positive.
                        Some(name) => {
                            event.paths.is_empty()
                                || event
                                    .paths
                                    .iter()
                                    .any(|p| p.file_name() == Some(name.as_os_str()))
                        }
                        // Target has no file_name (e.g. "/"): forward everything.
                        None => true,
                    };
                    if !touches_target {
                        return;
                    }
                    // Fails only if the worker already exited; ignore.
                    let _ = kick_tx.send(());
                }
                Err(e) => {
                    tracing::warn!(target: "config", err = %e, "watcher error");
                }
            })?;

        // Loops until all senders drop (dropping the `WatcherGuard`'s watcher closes
        // the kick channel).
        let path_for_worker = path.clone();
        let worker = std::thread::Builder::new()
            .name("config-reload-debounce".into())
            .spawn(move || {
                crate::config::watcher::debounce_reload_worker(
                    path_for_worker,
                    inner,
                    reload_count,
                    reload_rejection_count,
                    mutate_lock,
                    on_reload,
                    kick_rx,
                )
            })
            .map_err(|e| ConfigError::ThreadSpawn {
                path: path.display().to_string(),
                source: e,
            })?;

        // Watch the parent dir so rename-replace edits surface across inode swaps.
        let parent = self
            .path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."));
        watcher.watch(&parent, RecursiveMode::NonRecursive)?;
        Ok(WatcherGuard {
            watcher: Some(Box::new(watcher)),
            worker: Some(worker),
        })
    }

    /// Successful watcher reloads (different value swapped in). `Relaxed` suffices:
    /// a lone visibility signal with no other state to synchronize.
    pub fn reload_count(&self) -> u64 {
        self.reload_count.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// REJECTED watcher reloads. Paired with [`Self::reload_count`] via
    /// `/api/v1/status`: a rejected increment with no matching reload increment
    /// means "my edit was discarded".
    pub fn reload_rejection_count(&self) -> u64 {
        self.reload_rejection_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Clone both reload-counter `Arc`s for a process-wide handle
    /// ([`crate::status::config_metrics::install_global`]); shares the underlying
    /// atomics by refcount.
    pub fn reload_counter_arcs(
        &self,
    ) -> (
        Arc<std::sync::atomic::AtomicU64>,
        Arc<std::sync::atomic::AtomicU64>,
    ) {
        (
            self.reload_count.clone(),
            self.reload_rejection_count.clone(),
        )
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Guard from `ConfigCell::open_mutation`; drop without `commit` releases the lock
/// and IN_MUTATE flag without persisting (rollback semantics).
struct CellGuard<'h> {
    cell: &'h ConfigCell,
    _lock: parking_lot::MutexGuard<'h, ()>,
    /// Cloned snapshot mutated via `config()`; `Some` until commit/rollback.
    cfg: Option<Config>,
    _reset: ResetOnDrop,
}

/// Per-thread re-entrancy reset; cleared on drop so a panic mid-guard doesn't lock
/// the thread out of future mutations.
struct ResetOnDrop;
impl Drop for ResetOnDrop {
    fn drop(&mut self) {
        IN_MUTATE.with(|c| c.set(false));
    }
}

impl<'h> handle::ConfigGuard for CellGuard<'h> {
    fn config(&mut self) -> &mut Config {
        self.cfg
            .as_mut()
            .expect("config() after commit/rollback is a guard misuse")
    }

    fn commit(
        mut self: Box<Self>,
        after: Box<dyn FnOnce(&Config) + Send>,
    ) -> Result<(), ConfigError> {
        let cfg = self
            .cfg
            .take()
            .expect("commit on a guard already committed/rolled-back");
        // `_lock` + `_reset` drop at end of scope, releasing the lock and IN_MUTATE.
        self.cell.validate_persist_swap(cfg, |c| after(c))
    }

    fn rollback(self: Box<Self>) {
        drop(self);
    }
}

impl handle::ConfigHandle for ConfigCell {
    fn snapshot(&self) -> Arc<Config> {
        ConfigCell::snapshot(self)
    }

    fn path(&self) -> &Path {
        ConfigCell::path(self)
    }

    fn open_mutation(&self) -> Result<Box<dyn handle::ConfigGuard + '_>, ConfigError> {
        let was_in = IN_MUTATE.with(|c| c.replace(true));
        if was_in {
            return Err(ConfigError::ReentrantMutate);
        }
        // Lock AFTER setting IN_MUTATE so re-entry surfaces as ReentrantMutate, not
        // a deadlock on the lock.
        let _reset = ResetOnDrop;
        let lock = self.mutate_lock.lock();
        let cfg = (*self.snapshot()).clone();
        Ok(Box::new(CellGuard {
            cell: self,
            _lock: lock,
            cfg: Some(cfg),
            _reset,
        }))
    }
}

/// RAII guard for the notify watcher + debounce worker; dropping frees the watcher,
/// closing the kick channel and unblocking the worker's `recv`.
///
/// Teardown SIGNALS-and-DETACHES, never joins: `JoinHandle::join()` from Drop would
/// wedge a tokio worker frame on any unwind path. The worker observes the closed
/// channel within one ~100 ms debounce window and exits; the OS reclaims the
/// detached thread at process exit (tests needing post-drop quiescence must sleep >
/// one debounce window). Drop order is load-bearing: the worker blocks on the kick
/// channel whose only sender lives in the watcher's callback, so the watcher MUST
/// drop before the worker handle is detached.
pub struct WatcherGuard {
    watcher: Option<Box<dyn notify::Watcher + Send + Sync>>,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl Drop for WatcherGuard {
    // Explicit two-step teardown pins the order regardless of field declaration:
    // free the watcher first (closing kick_tx), then detach the worker (no join).
    fn drop(&mut self) {
        if let Some(w) = self.watcher.take() {
            drop(w);
        }
        drop(self.worker.take());
    }
}

impl std::fmt::Debug for WatcherGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WatcherGuard").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    // `std::fs::write` + `std::sync::Mutex` are disallowed in production, fine here.
    #![allow(clippy::disallowed_methods, clippy::disallowed_types)]
    use super::*;
    use crate::common::ids::MicId;
    use crate::inference::{BackboneCatalogue, BackboneKind, BackboneRef};

    fn fresh_default() -> Config {
        Config::default_for()
    }

    /// Default `[api]` is loopback-TCP-only -- valid without filesystem access.
    fn fresh_api() -> ApiCfg {
        ApiCfg::default()
    }

    /// `[api]` with a UDS listener; TCP stays set so validation reaches the UDS arm.
    fn fresh_api_with_uds(uds_path: PathBuf) -> ApiCfg {
        ApiCfg {
            uds_path: Some(uds_path),
            ..ApiCfg::default()
        }
    }

    fn fresh_output_inference(uds_path: PathBuf) -> OutputInferenceCfg {
        OutputInferenceCfg {
            uds_path,
            uds_mode: 0o666,
        }
    }

    fn fresh_launch_for_load() -> LaunchConfig {
        LaunchConfig::default_for()
    }

    #[test]
    fn config_round_trips_through_toml() {
        let cfg = fresh_default();
        let s = toml::to_string_pretty(&cfg).expect("serialize");
        let back: Config = toml::from_str(&s).expect("parse");
        assert_eq!(cfg, back);
    }

    /// TCP default is strict (subprotocol required); UDS relaxes it because
    /// filesystem perms are the auth boundary.
    #[test]
    fn api_cfg_defaults_per_listener_policy() {
        let api = ApiCfg::default();
        assert_eq!(api.tcp_policy.max_connections_per_stream, 32);
        assert!(
            api.tcp_policy.require_subprotocol,
            "TCP default must keep the strict subprotocol gate",
        );
        assert_eq!(api.uds_policy.max_connections_per_stream, 32);
        assert!(
            !api.uds_policy.require_subprotocol,
            "UDS default must relax the subprotocol gate",
        );

        let s = toml::to_string_pretty(&api).expect("serialize");
        let back: ApiCfg = toml::from_str(&s).expect("parse");
        assert_eq!(api.tcp_policy, back.tcp_policy);
        assert_eq!(api.uds_policy, back.uds_policy);
    }

    /// A minimal `[api]` TOML (no policy sub-tables) loads via `#[serde(default)]` to
    /// the same defaults `ApiCfg::default` writes.
    #[test]
    fn api_cfg_minimal_toml_loads_with_default_policies() {
        let api = ApiCfg::default();
        let mut text = toml::to_string_pretty(&api).expect("ser");
        // toml-rs emits scalars first, sub-tables last, so truncating at the first
        // marker drops both policy tables, mimicking a hand-edited file.
        for marker in ["[tcp_policy]", "[uds_policy]"] {
            if let Some(start) = text.find(marker) {
                text.truncate(start);
                break;
            }
        }
        let parsed: ApiCfg = toml::from_str(&text).expect("parse minimal shape");
        assert_eq!(parsed.tcp_policy, api.tcp_policy);
        assert_eq!(parsed.uds_policy, api.uds_policy);
    }

    #[test]
    fn load_round_trip() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        let cfg = fresh_default();
        let text = toml::to_string_pretty(&cfg).expect("ser");
        std::fs::write(&path, text).expect("write");
        let h = ConfigCell::load(&path).expect("load");
        assert_eq!(*h.snapshot(), cfg);
    }

    /// `Config::validate` walks every operator-tunable sub-section; catches a future
    /// loosening or removal of a delegated predicate.
    #[test]
    fn config_validate_walks_every_subsection() {
        fresh_default().validate().expect("default must validate");

        let mut cfg = fresh_default();
        cfg.inference.hop_samples = 0;
        let err = cfg
            .validate()
            .expect_err("zero hop must reject")
            .to_string();
        assert!(err.contains("inference"), "{err}");
    }

    /// Production default `tcp_bind` is loopback (auth lives in the front-proxy).
    #[test]
    fn default_tcp_bind_is_loopback() {
        let launch = LaunchConfig::default_for();
        assert_eq!(launch.api.tcp_bind.as_deref(), Some("127.0.0.1:8787"));
        assert_eq!(launch.api.uds_path, None);
        assert!(
            launch.output.inference.is_none(),
            "first-boot default binds no raw inference socket",
        );
    }

    /// `ApiCfg::validate` accepts any well-formed `host:port` (incl. non-loopback
    /// and IPv6 unspecified); it only rejects shapes that cannot bind at all.
    #[test]
    fn api_cfg_accepts_any_well_formed_tcp_bind() {
        for ok in [
            "127.0.0.1:8787",
            "127.0.0.1:0",
            "127.5.5.5:8787",
            "[::1]:8787",
            "localhost:8787",
            "Localhost:9000",
            "0.0.0.0:8787",
            "[::]:8787",
            "192.168.1.10:8787",
            "8.8.8.8:8787",
            "example.com:8787",
        ] {
            let mut api = fresh_api();
            api.tcp_bind = Some(ok.into());
            if let Err(e) = api.validate() {
                panic!("{ok:?} should validate but got {e}");
            }
        }
    }

    /// `ApiCfg::validate` requires at least one listener, else the daemon boots
    /// with no reachable API surface.
    #[test]
    fn api_cfg_requires_at_least_one_listener() {
        let mut api = fresh_api();
        api.tcp_bind = None;
        api.uds_path = None;
        let err = api
            .validate()
            .expect_err("an [api] with no listener must reject")
            .to_string();
        assert!(
            err.contains("at least one listener"),
            "diagnostic should name the missing-listener shape: {err}",
        );

        fresh_api()
            .validate()
            .expect("TCP-only [api] must validate");

        let mut uds_only = fresh_api_with_uds(std::env::temp_dir().join("acousticslab_api.sock"));
        uds_only.tcp_bind = None;
        uds_only.validate().expect("UDS-only [api] must validate");

        fresh_api_with_uds(std::env::temp_dir().join("acousticslab_api_both.sock"))
            .validate()
            .expect("dual-transport [api] must validate");
    }

    /// `ApiCfg::validate` rejects `broadcast_capacity` above `MAX_BROADCAST_CAPACITY`:
    /// a typo would eagerly allocate a broadcast-ring big enough to OOM-kill the
    /// daemon at boot.
    #[test]
    fn api_cfg_rejects_broadcast_capacity_above_max() {
        let mut api = fresh_api();
        api.broadcast_capacity = ApiCfg::MAX_BROADCAST_CAPACITY + 1;
        let err = api
            .validate()
            .expect_err("broadcast_capacity above the cap must reject")
            .to_string();
        assert!(
            err.contains("broadcast_capacity"),
            "diagnostic should name broadcast_capacity: {err}",
        );
        api.broadcast_capacity = ApiCfg::MAX_BROADCAST_CAPACITY;
        api.validate()
            .expect("broadcast_capacity at the cap must validate");
        api.broadcast_capacity = 6_400_000;
        api.validate()
            .expect_err("hostile/typo'd broadcast_capacity must reject");
    }

    /// `ApiCfg::validate` rejects an empty-host `tcp_bind` (`":8787"`) at
    /// config-time, not deferred to the daemon's bind-future parse.
    #[test]
    fn api_cfg_rejects_empty_host_tcp_bind() {
        let mut api = fresh_api();
        api.tcp_bind = Some(":8787".into());
        let err = api
            .validate()
            .expect_err("empty-host tcp_bind must reject")
            .to_string();
        assert!(
            err.contains("empty host"),
            "diagnostic should name the empty-host shape: {err}",
        );
    }

    /// `validate_uds_path` rejects a regular file at a `uds_path` at config-load,
    /// before the daemon starts.
    #[test]
    fn output_inference_cfg_rejects_uds_path_pointing_at_regular_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let regular_file = dir.path().join("not-a-socket.txt");
        std::fs::write(&regular_file, b"hello").expect("write regular file");
        let err = fresh_output_inference(regular_file.clone())
            .validate()
            .expect_err("regular-file uds_path must reject")
            .to_string();
        assert!(
            err.contains("regular file"),
            "diagnostic should name the regular-file shape: {err}",
        );
        assert!(
            err.contains(&regular_file.display().to_string()),
            "diagnostic should include the offending path: {err}",
        );
        // The same validator backs `[api].uds_path`; only the prefix differs.
        let err_api = fresh_api_with_uds(regular_file.clone())
            .validate()
            .expect_err("regular-file api.uds_path must reject")
            .to_string();
        assert!(
            err_api.contains("api.uds_path") && err_api.contains("regular file"),
            "[api] must reject the same shape with its own prefix: {err_api}",
        );
    }

    /// `validate_uds_path` rejects a symlink at a `uds_path` even when its target
    /// is a socket: following it at bind time exposes the unlink to a TOCTOU.
    #[cfg(unix)]
    #[test]
    fn output_inference_cfg_rejects_uds_path_pointing_at_symlink() {
        let dir = tempfile::tempdir().expect("tempdir");
        let target = dir.path().join("real.sock");
        std::fs::write(&target, b"x").expect("write target");
        let link = dir.path().join("link.sock");
        std::os::unix::fs::symlink(&target, &link).expect("symlink");
        let err = fresh_output_inference(link)
            .validate()
            .expect_err("symlink uds_path must reject")
            .to_string();
        assert!(
            err.contains("symlink"),
            "diagnostic should name the symlink shape: {err}",
        );
    }

    /// `validate_uds_path` rejects a path whose parent directory does not exist,
    /// surfacing the typo at config-load instead of a confusing bind-future ENOENT.
    #[test]
    fn output_inference_cfg_rejects_uds_path_with_missing_parent() {
        let dir = tempfile::tempdir().expect("tempdir");
        let err = fresh_output_inference(dir.path().join("does-not-exist").join("a.sock"))
            .validate()
            .expect_err("missing-parent uds_path must reject")
            .to_string();
        assert!(
            err.contains("parent directory") && err.contains("does not exist"),
            "diagnostic should name the missing-parent shape: {err}",
        );
    }

    /// `validate_uds_path` rejects a bare filename: binding into CWD is undefined
    /// for a daemon-supervised process, so require an explicit parent.
    #[test]
    fn output_inference_cfg_rejects_uds_path_without_parent() {
        let err = fresh_output_inference(PathBuf::from("acousticslab.sock"))
            .validate()
            .expect_err("bare-filename uds_path must reject")
            .to_string();
        assert!(
            err.contains("parent directory"),
            "diagnostic should name the missing-parent shape: {err}",
        );
    }

    /// `validate_uds_path` accepts an existing-parent, not-yet-existing target
    /// (normal first-boot case -- bind creates the socket).
    #[test]
    fn output_inference_cfg_accepts_uds_path_in_existing_dir() {
        let dir = tempfile::tempdir().expect("tempdir");
        fresh_output_inference(dir.path().join("a.sock"))
            .validate()
            .expect("uds_path under existing dir is fine");
    }

    /// `uds_mode` defaults to 0o666 on both UDS surfaces; `[api]` additionally
    /// defaults `broadcast_capacity` (64) and leaves `tcp_bind` absent via
    /// `#[serde(default)]`.
    #[test]
    fn uds_mode_and_api_scalars_default_when_omitted() {
        let out: OutputInferenceCfg =
            toml::from_str("uds_path = \"/run/acousticslab/out.sock\"\n").expect("parse output");
        assert_eq!(out.uds_mode, 0o666);

        let api: ApiCfg =
            toml::from_str("uds_path = \"/run/acousticslab/api.sock\"\n").expect("parse api");
        assert_eq!(api.uds_mode, 0o666);
        assert_eq!(api.broadcast_capacity, 64);
        assert_eq!(api.tcp_bind, None);
    }

    /// `uds_mode` above the 12-bit POSIX range is rejected (only when a `uds_path`
    /// is set, since the mode is otherwise unused).
    #[test]
    fn api_cfg_rejects_oversized_uds_mode() {
        let mut api = fresh_api_with_uds(std::env::temp_dir().join("acousticslab_mode.sock"));
        api.uds_mode = 0o10000; // one bit above 0o7777
        let err = api
            .validate()
            .expect_err("oversized uds_mode must reject")
            .to_string();
        assert!(
            err.contains("uds_mode") && err.contains("POSIX mode"),
            "diagnostic should name the mode-range shape: {err}",
        );
    }

    /// A launch.toml with the retired `[stream]` table is hard-rejected at load
    /// with an actionable rename message (not serde's cryptic "unknown field").
    #[test]
    fn launch_load_rejects_legacy_stream_table() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("launch.toml");
        std::fs::write(
            &path,
            "[stream]\n\
             uds_path = \"misc/share/acousticslabd.sock\"\n\
             tcp_bind = \"127.0.0.1:8787\"\n\
             broadcast_capacity = 64\n",
        )
        .expect("write legacy launch.toml");
        let err = LaunchConfig::load(&path)
            .expect_err("a launch.toml with [stream] must be rejected")
            .to_string();
        assert!(
            err.contains("[stream]") && err.contains("[api]") && err.contains("[output.inference]"),
            "diagnostic must point at the new sections: {err}",
        );
    }

    /// A launch.toml with no `[output]` table loads cleanly (the raw inference
    /// socket is optional); `output.inference` is `None`.
    #[test]
    fn launch_load_accepts_omitted_output_section() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("launch.toml");
        let launch = LaunchConfig::default_for();
        let mut text = toml::to_string_pretty(&launch).expect("ser");
        // Strip the possibly-empty `[output]` table to mimic a hand-edited file.
        text = text.replace("[output]\n", "").replace("\n[output]\n", "\n");
        std::fs::write(&path, text).expect("write");
        let parsed = LaunchConfig::load(&path).expect("load must accept missing [output]");
        assert!(parsed.output.inference.is_none());
    }

    /// `[api].uds_path` and `[output.inference].uds_path` must be distinct: two
    /// listeners on the same path orphan each other at bind time, so `load()`
    /// rejects the collision up front.
    #[test]
    fn launch_load_rejects_colliding_uds_paths() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("launch.toml");
        let shared = dir.path().join("shared.sock");
        let mut launch = LaunchConfig::default_for();
        launch.api.uds_path = Some(shared.clone());
        launch.output.inference = Some(OutputInferenceCfg {
            uds_path: shared,
            uds_mode: 0o666,
        });
        std::fs::write(&path, toml::to_string_pretty(&launch).expect("ser")).expect("write");
        let err = LaunchConfig::load(&path)
            .expect_err("colliding UDS paths must reject")
            .to_string();
        assert!(
            err.contains("collides") && err.contains("distinct paths"),
            "diagnostic should name the collision: {err}",
        );

        launch.output.inference = Some(OutputInferenceCfg {
            uds_path: dir.path().join("result.sock"),
            uds_mode: 0o666,
        });
        std::fs::write(&path, toml::to_string_pretty(&launch).expect("ser")).expect("write");
        LaunchConfig::load(&path).expect("distinct UDS paths must validate");
    }

    /// `TrainingDefaults::validate` rejects every runtime-fatal `0` field at the
    /// boot-time TOML gate, not just at the first `POST /train`.
    #[test]
    fn training_defaults_validator_rejects_zero_fields() {
        let td = TrainingDefaults {
            epochs: 0,
            ..TrainingDefaults::default()
        };
        let err = td.validate().expect_err("zero epochs must reject");
        assert!(err.contains("epochs"), "{err}");

        let td = TrainingDefaults {
            batch_size: 0,
            ..TrainingDefaults::default()
        };
        let err = td.validate().expect_err("zero batch_size must reject");
        assert!(err.contains("batch_size"), "{err}");

        let td = TrainingDefaults {
            learning_rate_e6: 0,
            ..TrainingDefaults::default()
        };
        let err = td
            .validate()
            .expect_err("zero learning_rate_e6 must reject");
        assert!(err.contains("learning_rate"), "{err}");
    }

    /// `TrainingDefaults::validate` rejects values past the sanity caps so a typo
    /// (`epochs = 100000`) surfaces at boot, not as an hour-long stuck job.
    #[test]
    fn training_defaults_validator_rejects_oversized_fields() {
        let td = TrainingDefaults {
            epochs: 1_000_000,
            ..TrainingDefaults::default()
        };
        let err = td.validate().expect_err("absurd epochs must reject");
        assert!(err.contains("epochs"), "{err}");

        let td = TrainingDefaults {
            batch_size: 100_000,
            ..TrainingDefaults::default()
        };
        let err = td.validate().expect_err("absurd batch_size must reject");
        assert!(err.contains("batch_size"), "{err}");

        let td = TrainingDefaults {
            learning_rate_e6: 5_000_000,
            ..TrainingDefaults::default()
        };
        let err = td.validate().expect_err("absurd lr must reject");
        assert!(err.contains("learning_rate"), "{err}");
    }

    #[test]
    fn training_defaults_default_validates() {
        TrainingDefaults::default()
            .validate()
            .expect("default training_defaults must validate");
    }

    /// `LaunchConfig::load` rejects an invalid `[training_defaults]` block at boot,
    /// surfacing the typo in systemd logs.
    #[test]
    fn launch_load_rejects_invalid_training_defaults() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("launch.toml");
        let mut launch = fresh_launch_for_load();
        launch.training_defaults.epochs = 0;
        let text = toml::to_string_pretty(&launch).expect("ser");
        std::fs::write(&path, text).expect("write");
        let err = LaunchConfig::load(&path).expect_err("zero epochs must reject");
        let ConfigError::Invalid { msg, .. } = err else {
            panic!("expected ConfigError::Invalid, got {err:?}");
        };
        assert!(
            msg.contains("training_defaults") && msg.contains("epochs"),
            "diagnostic should name the offending field: {msg}",
        );
    }

    /// Re-entering `mutate_then` from the same thread returns
    /// `Err(ConfigError::ReentrantMutate)` instead of silently deadlocking.
    #[test]
    fn mutate_then_rejects_reentry() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        let h = ConfigCell::from_value(fresh_default(), path.clone()).expect("validate");
        h.persist().expect("persist initial");
        let h_arc = Arc::new(h);

        let inner_result = Arc::new(parking_lot::Mutex::new(None::<Result<(), ConfigError>>));
        let inner_for_outer = inner_result.clone();
        let h_inner = h_arc.clone();
        h_arc
            .mutate(move |c| {
                c.inference.top_k = 5;
                let r = h_inner.mutate(|c2| {
                    c2.inference.top_k = 7;
                });
                *inner_for_outer.lock() = Some(r);
            })
            .expect("outer mutate must succeed");

        let inner = inner_result.lock().take().expect("inner ran");
        match inner {
            Err(ConfigError::ReentrantMutate) => {}
            other => panic!("expected ReentrantMutate from inner re-entry, got {other:?}"),
        }

        // Outer's value committed; the rejected inner's (7) did not.
        assert_eq!(h_arc.snapshot().inference.top_k, 5);

        // Sentinel reset after the outer returns: a fresh top-level mutate succeeds.
        h_arc
            .mutate(|c| {
                c.inference.top_k = 11;
            })
            .expect("post-reentry mutate must succeed");
        assert_eq!(h_arc.snapshot().inference.top_k, 11);
    }

    /// `mutate_then` rejecting an invalid mutator leaves disk + memory at the
    /// prior snapshot; nothing is half-written.
    #[test]
    fn mutate_rejects_invalid_result() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        let h = ConfigCell::from_value(fresh_default(), path.clone()).expect("validate");
        h.persist().expect("persist initial");

        let prior_disk = std::fs::read_to_string(&path).expect("read disk");
        let prior_mem = (*h.snapshot()).clone();

        let err = h
            .mutate(|c| {
                c.inference.top_k = 0;
            })
            .expect_err("invalid mutator result must reject");
        assert!(matches!(err, ConfigError::Invalid { .. }), "got {err:?}");

        let post_disk = std::fs::read_to_string(&path).expect("read disk");
        assert_eq!(prior_disk, post_disk, "disk must not change on rejection");
        assert_eq!(
            prior_mem,
            *h.snapshot(),
            "snapshot must not change on rejection"
        );
    }

    #[test]
    fn load_rejects_invalid_inference_cfg() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        let mut cfg = fresh_default();
        cfg.inference.hop_samples = 0;
        let text = toml::to_string_pretty(&cfg).expect("ser");
        std::fs::write(&path, text).expect("write");
        let err = ConfigCell::load(&path).expect_err("zero hop must reject");
        assert!(
            matches!(err, ConfigError::Invalid { .. }),
            "expected ConfigError::Invalid, got {err:?}"
        );

        let mut cfg = fresh_default();
        cfg.inference.top_k = 0;
        let text = toml::to_string_pretty(&cfg).expect("ser");
        std::fs::write(&path, text).expect("write");
        let err = ConfigCell::load(&path).expect_err("zero top_k must reject");
        assert!(matches!(err, ConfigError::Invalid { .. }));
    }

    /// `LaunchConfig::load` rejects a candidate failing `MicCandidate::validate`
    /// (empty channel whitelist) at load, not as a runtime warn at first open.
    #[test]
    fn launch_load_rejects_invalid_mic_candidate() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("launch.toml");
        let mut launch = fresh_launch_for_load();
        launch.mic.candidates[0].channels.clear();
        let text = toml::to_string_pretty(&launch).expect("ser");
        std::fs::write(&path, text).expect("write");
        let err = LaunchConfig::load(&path).expect_err("empty channels must reject");
        let ConfigError::Invalid { msg, .. } = err else {
            panic!("expected ConfigError::Invalid, got {err:?}");
        };
        assert!(
            msg.contains("launch mic catalogue"),
            "error message should identify the launch catalogue, got {msg}",
        );
    }

    /// `LaunchConfig` round-trips through TOML; catches serde tag/rename issues.
    #[test]
    fn launch_config_round_trips_through_toml() {
        let original = LaunchConfig::default_for();
        let s = toml::to_string_pretty(&original).expect("ser");
        let back: LaunchConfig = toml::from_str(&s).expect("de");
        assert_eq!(original, back);
    }

    /// First-boot defaults carry no backbone paths: artifacts are
    /// deployment-specific, set in launch.toml, not daemon-hardcoded.
    #[test]
    fn launch_default_backbone_catalogue_is_empty() {
        let l = LaunchConfig::default_for();
        assert!(l.backbone.is_empty());
    }

    /// The checked-in dev fixtures (`misc/etc/config.toml` user-prefs +
    /// `misc/etc/launch.toml` catalogues) parse and cross-validate together, so
    /// the dev smoke command can't regress to a policy/catalogue/backbone mismatch.
    #[test]
    fn bundled_etc_configs_parse_and_cross_validate() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let config_path = root.join("misc/etc/config.toml");
        let launch_path = root.join("misc/etc/launch.toml");

        std::fs::create_dir_all(root.join("misc/share")).expect("bundled UDS parent");
        let launch = LaunchConfig::load(&launch_path).expect("load bundled launch.toml");
        let text = std::fs::read_to_string(&config_path).expect("read bundled config.toml");
        let cfg: Config = toml::from_str(&text).expect("parse bundled config.toml");
        cfg.validate().expect("bundled config.toml validates");
        validate_policy_against_catalogue(&cfg.mic, &launch.mic, &config_path)
            .expect("bundled mic policy matches bundled launch catalogue");
        let paths: Vec<_> = launch
            .backbone
            .candidates
            .iter()
            .map(|c| c.path.as_path())
            .collect();
        assert_eq!(
            paths,
            vec![
                Path::new("misc/backbones/backbone.rknn"),
                Path::new("misc/backbones/backbone.mpk"),
            ],
            "bundled launch.toml must specify local dev backbone paths",
        );
        assert_eq!(
            launch.head.default.as_ref().map(|h| h.path.as_path()),
            Some(Path::new("misc/heads/default/head.mpk")),
            "bundled launch.toml must specify the local dev default head mpk",
        );
        assert_eq!(
            launch
                .head
                .default
                .as_ref()
                .map(|h| h.labels_path.as_path()),
            Some(Path::new("misc/heads/default/labels.txt")),
            "bundled launch.toml must specify the local dev default labels",
        );
        assert_eq!(
            launch.api.tcp_bind.as_deref(),
            Some("127.0.0.1:8787"),
            "bundled launch.toml must own the local dev [api] TCP bind",
        );
        assert_eq!(
            launch.api.uds_path, None,
            "bundled [api] ships TCP-only (the UDS option is commented out)",
        );
        assert_eq!(
            launch
                .output
                .inference
                .as_ref()
                .map(|o| o.uds_path.as_path()),
            Some(Path::new("misc/share/acousticslabd-result.sock")),
            "bundled launch.toml must own the local dev raw inference output socket",
        );
        // Pin bundled training_defaults + file caps so drift surfaces in CI.
        assert_eq!(launch.training_defaults, TrainingDefaults::default());
        assert_eq!(launch.file, FileCfg::default());
    }

    /// A launch.toml without `[[backbone.candidates]]` parses to an empty
    /// catalogue (daemon runs without inference), not a hard failure.
    #[test]
    fn launch_load_accepts_missing_backbone_field() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("launch.toml");
        let mut launch = fresh_launch_for_load();
        launch.backbone = BackboneCatalogue::default();
        let mut text = toml::to_string_pretty(&launch).expect("ser");
        // Strip the emitted-even-when-empty `[backbone]` header to mimic a
        // hand-edited file that never had one.
        text = text
            .replace("[backbone]\n", "")
            .replace("\n[backbone]\n", "\n");
        std::fs::write(&path, text).expect("write");
        let parsed = LaunchConfig::load(&path).expect("load must accept missing backbone");
        assert!(parsed.backbone.is_empty());
    }

    #[test]
    fn launch_load_rejects_malformed_backbone_hash() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("launch.toml");
        let mut launch = fresh_launch_for_load();
        launch.backbone.candidates.push(BackboneRef {
            kind: BackboneKind::Burn,
            path: PathBuf::from("operator-supplied/backbone.mpk"),
            hash: None,
        });
        // 63-char hash fails the 64-hex validator.
        launch.backbone.candidates[0].hash = Some("a".repeat(63));
        let text = toml::to_string_pretty(&launch).expect("ser");
        std::fs::write(&path, text).expect("write");
        let err = LaunchConfig::load(&path).expect_err("must reject");
        match err {
            ConfigError::Invalid { msg, .. } => {
                assert!(
                    msg.contains("backbone catalogue") && msg.contains("64"),
                    "diagnostic should name the catalogue + hash length: {msg}",
                );
            }
            other => panic!("expected ConfigError::Invalid, got {other:?}"),
        }
    }

    /// `validate_policy_against_catalogue` (the boot + hot-reload entry point)
    /// rejects a Fixed-id not in the catalogue.
    #[test]
    fn validate_policy_rejects_unknown_mic_id() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("dev.toml");
        let launch = LaunchConfig::default_for();
        let policy = MicPolicy {
            mic: MicSelection::Fixed {
                id: MicId::from_static("not-in-catalogue"),
            },
            channel: ChannelSelection::Auto,
        };
        let err = validate_policy_against_catalogue(&policy, &launch.mic, &path)
            .expect_err("must reject");
        assert!(
            matches!(err, ConfigError::Invalid { .. }),
            "expected Invalid, got {err:?}",
        );
        if let ConfigError::Invalid { msg, .. } = err {
            assert!(
                msg.contains("not-in-catalogue"),
                "diagnostic should name the missing id, got {msg}",
            );
        }
    }

    /// `LaunchConfig::load` gates the `[file]` admission caps at boot: zero on
    /// either field would brick uploads on an otherwise-healthy daemon.
    #[test]
    fn launch_load_rejects_invalid_file_cfg() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("launch.toml");
        let mut launch = fresh_launch_for_load();
        launch.file.max_upload_bytes = 0;
        let text = toml::to_string_pretty(&launch).expect("ser");
        std::fs::write(&path, text).expect("write");
        let err = LaunchConfig::load(&path).expect_err("zero max_upload_bytes must reject");
        assert!(
            matches!(err, ConfigError::Invalid { .. }),
            "expected ConfigError::Invalid, got {err:?}",
        );
        if let ConfigError::Invalid { msg, .. } = err {
            assert!(
                msg.contains("file") && msg.contains("max_upload_bytes"),
                "diagnostic should name the offending field: {msg}",
            );
        }

        let mut launch = fresh_launch_for_load();
        launch.file.max_concurrent_uploads = 0;
        let text = toml::to_string_pretty(&launch).expect("ser");
        std::fs::write(&path, text).expect("write");
        let err = LaunchConfig::load(&path).expect_err("zero max_concurrent_uploads must reject");
        assert!(
            matches!(err, ConfigError::Invalid { .. }),
            "expected ConfigError::Invalid, got {err:?}",
        );
    }

    /// Launch TOMLs without a `[file]` block load the on-device defaults via
    /// `#[serde(default)]`; pinned values guard against a silent cap loosening.
    #[test]
    fn legacy_launch_without_file_block_loads_defaults() {
        let launch = LaunchConfig::default_for();
        let mut value = toml::Value::try_from(&launch).expect("launch to toml value");
        value
            .as_table_mut()
            .expect("launch is a table")
            .remove("file");
        let text = toml::to_string_pretty(&value).expect("serialize legacy launch");

        let parsed: LaunchConfig = toml::from_str(&text).expect("parse legacy launch");
        assert_eq!(parsed.file, FileCfg::default());
        assert_eq!(parsed.file.max_upload_bytes, 256 * 1024 * 1024);
        assert_eq!(parsed.file.max_concurrent_uploads, 4);
    }

    /// Launch TOMLs without `[training_defaults]` load defaults via `#[serde(default)]`.
    #[test]
    fn legacy_launch_without_training_defaults_loads_defaults() {
        let launch = LaunchConfig::default_for();
        let mut value = toml::Value::try_from(&launch).expect("launch to toml value");
        value
            .as_table_mut()
            .expect("launch is a table")
            .remove("training_defaults");
        let text = toml::to_string_pretty(&value).expect("serialize legacy launch");

        let parsed: LaunchConfig = toml::from_str(&text).expect("parse legacy launch");
        assert_eq!(parsed.training_defaults, TrainingDefaults::default());
    }

    #[test]
    fn mutate_persists_and_swaps() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        let h = ConfigCell::from_value(fresh_default(), path.clone()).expect("validate");
        h.persist().expect("persist initial");

        h.mutate(|c| {
            c.inference.hop_samples = 22_050;
            c.inference.top_k = 5;
        })
        .expect("mutate");

        let snap = h.snapshot();
        assert_eq!(snap.inference.hop_samples, 22_050);
        assert_eq!(snap.inference.top_k, 5);

        let on_disk = std::fs::read_to_string(&path).expect("read");
        let parsed: Config = toml::from_str(&on_disk).expect("parse");
        assert_eq!(parsed.inference.hop_samples, 22_050);
        assert_eq!(parsed.inference.top_k, 5);
    }

    /// Concurrent `mutate` on disjoint fields must not lose updates; without the
    /// mutate_lock this FAILS (each mutator clones the OLD snapshot, so the second
    /// writer drops the first's change). Writer ranges MUST NOT overlap the field
    /// defaults (`hop_samples = 44_100`), else a lost-update revert-to-default still
    /// passes the "in writer range" assert; `[22_000, 22_200)` is distinct yet
    /// inside the validator window (`11_025..=44_100`).
    #[test]
    fn concurrent_mutate_preserves_disjoint_field_updates() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::thread;

        let default_hop = fresh_default().inference.hop_samples;
        assert!(
            !(22_000..=22_199).contains(&default_hop),
            "test offset overlaps default; would mask lost-update bug",
        );

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        let h = ConfigCell::from_value(fresh_default(), path.clone()).expect("validate");
        h.persist().expect("initial persist");
        let h = Arc::new(h);
        let mic_seq = Arc::new(AtomicU32::new(0));
        let inf_seq = Arc::new(AtomicU32::new(0));

        let h1 = h.clone();
        let mic_seq_w = mic_seq.clone();
        let writer_mic = thread::spawn(move || {
            for i in 0..200 {
                h1.mutate(|c| {
                    let id = mic_seq_w.fetch_add(1, Ordering::Relaxed);
                    c.mic.mic = MicSelection::Fixed {
                        id: MicId::parse(&format!("mic-{id}")).expect("test mic id"),
                    };
                    let _ = i;
                })
                .expect("mic mutate");
            }
        });
        let h2 = h.clone();
        let inf_seq_w = inf_seq.clone();
        let writer_inf = thread::spawn(move || {
            for i in 0..200 {
                h2.mutate(|c| {
                    let id = inf_seq_w.fetch_add(1, Ordering::Relaxed);
                    // 22_000 base sits in the validator window; id < 200 keeps values unique.
                    c.inference.hop_samples = 22_000 + id as usize;
                    let _ = i;
                })
                .expect("inf mutate");
            }
        });
        writer_mic.join().unwrap();
        writer_inf.join().unwrap();

        let final_mic_seq = mic_seq.load(Ordering::Relaxed);
        let final_inf_seq = inf_seq.load(Ordering::Relaxed);
        assert_eq!(final_mic_seq, 200);
        assert_eq!(final_inf_seq, 200);

        // BOTH writers' last writes must survive (no field reverted to default).
        let mem = h.snapshot();
        assert!(
            (22_000..=22_199).contains(&mem.inference.hop_samples),
            "hop_samples reverted to {} (lost-update bug)",
            mem.inference.hop_samples,
        );
        match &mem.mic.mic {
            MicSelection::Fixed { id } => {
                let s = id.as_str();
                assert!(s.starts_with("mic-"), "mic reverted to non-Fixed name: {s}",);
                let n: u32 = s["mic-".len()..].parse().expect("parse mic id");
                assert!(n < 200, "mic id out of range: {n}");
            }
            other => panic!("mic reverted to default FirstAvailable / lost-update bug: {other:?}"),
        }

        // Disk must EXACTLY match memory (write + store share one critical section).
        let on_disk: Config = toml::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(on_disk, *mem);
    }

    /// A burst of 8 writes must coalesce inside the 100 ms debounce window to <= 2
    /// reloads (one for the burst, up to one more for a late lone event); without
    /// debouncing each FS event would trigger its own reload.
    #[tokio::test(flavor = "current_thread")]
    async fn watcher_debounces_burst_writes() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        let cfg = fresh_default();
        std::fs::write(&path, toml::to_string_pretty(&cfg).unwrap()).expect("write");

        let h = ConfigCell::load(&path).expect("load");
        let _guard = h.watch().expect("watch");

        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        // 8 rapid writes packed inside the 100 ms window, each a complete doc.
        for k in 2..=9 {
            let mut next = cfg.clone();
            next.inference.top_k = k;
            std::fs::write(&path, toml::to_string_pretty(&next).unwrap()).expect("rewrite");
        }

        tokio::time::sleep(std::time::Duration::from_millis(800)).await;

        assert_eq!(h.snapshot().inference.top_k, 9);
        // Allow 2 for FSEvents coalescing edge cases (most platforms give 1).
        let n = h.reload_count();
        assert!(
            n <= 2,
            "burst of 8 writes triggered {n} reloads; debounce \
             expected <= 2"
        );
        assert!(n >= 1, "burst of 8 writes triggered 0 reloads");
    }

    /// Watcher reloads when an external editor rewrites the file. The notify
    /// backend is platform-specific; poll up to 5 s to avoid timing flakiness.
    #[tokio::test(flavor = "current_thread")]
    async fn watcher_reloads_on_external_write() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        let cfg = fresh_default();
        std::fs::write(&path, toml::to_string_pretty(&cfg).unwrap()).expect("write");

        let h = ConfigCell::load(&path).expect("load");
        let _guard = h.watch().expect("watch");

        let mut new_cfg = cfg.clone();
        new_cfg.inference.top_k = 7;
        let new_text = toml::to_string_pretty(&new_cfg).unwrap();

        // Let notify spin up its FSEvents/inotify session before writing.
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        std::fs::write(&path, &new_text).expect("rewrite");

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            if h.snapshot().inference.top_k == 7 {
                return;
            }
            if std::time::Instant::now() > deadline {
                panic!(
                    "watcher did not reload within timeout; current top_k = {}",
                    h.snapshot().inference.top_k
                );
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
    }

    /// A panicking `watch_with` callback must not kill the debounce worker (else
    /// operators silently lose hot-reload).
    #[tokio::test(flavor = "current_thread")]
    async fn watch_with_callback_panic_does_not_kill_watcher() {
        use std::sync::Mutex;

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        let cfg = fresh_default();
        std::fs::write(&path, toml::to_string_pretty(&cfg).unwrap()).expect("write");

        let h = ConfigCell::load(&path).expect("load");
        let calls: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));
        let calls_clone = calls.clone();
        let _guard = h
            .watch_with(move |_c| {
                let n = {
                    let mut g = calls_clone.lock().unwrap();
                    *g += 1;
                    *g
                };
                if n == 1 {
                    panic!("intentional callback panic for test");
                }
                Ok(())
            })
            .expect("watch_with");

        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        let mut cfg2 = cfg.clone();
        cfg2.inference.top_k = 5;
        std::fs::write(&path, toml::to_string_pretty(&cfg2).unwrap()).expect("write 1");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            if *calls.lock().unwrap() >= 1 {
                break;
            }
            if std::time::Instant::now() > deadline {
                panic!("first callback never fired");
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }

        // Second write fires the callback iff the watcher survived the panic.
        let mut cfg3 = cfg.clone();
        cfg3.inference.top_k = 9;
        std::fs::write(&path, toml::to_string_pretty(&cfg3).unwrap()).expect("write 2");

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            if *calls.lock().unwrap() >= 2 {
                return;
            }
            if std::time::Instant::now() > deadline {
                let n = *calls.lock().unwrap();
                panic!(
                    "second callback never fired (calls={n}) -- watcher died after \
                     panicking callback",
                );
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
    }

    /// A callback returning `Err` discards the reload: `inner` keeps its prior value
    /// and `reload_count` does not increment.
    #[tokio::test(flavor = "current_thread")]
    async fn watch_with_callback_err_rejects_reload() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        let cfg = fresh_default();
        std::fs::write(&path, toml::to_string_pretty(&cfg).unwrap()).expect("write");

        let h = ConfigCell::load(&path).expect("load");
        let _guard = h
            .watch_with(|_| Err(ConfigValidationError::Callback("test rejection".into())))
            .expect("watch_with");

        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        let mut new_cfg = cfg.clone();
        new_cfg.inference.top_k = 7;
        std::fs::write(&path, toml::to_string_pretty(&new_cfg).unwrap()).expect("rewrite");

        tokio::time::sleep(std::time::Duration::from_millis(800)).await;

        let snap = h.snapshot();
        assert_ne!(
            snap.inference.top_k, 7,
            "rejected reload was committed anyway: top_k = {}",
            snap.inference.top_k,
        );
        assert_eq!(
            snap.inference.top_k, cfg.inference.top_k,
            "snapshot drifted from the prior valid value despite rejection",
        );
        assert_eq!(
            h.reload_count(),
            0,
            "reload_count incremented on rejected reload",
        );
    }

    /// `watch_with` invokes the callback on each successful reload (the daemon's
    /// hook to update live ArcSwaps downstream hot paths read).
    #[tokio::test(flavor = "current_thread")]
    async fn watch_with_invokes_callback_on_reload() {
        use std::sync::Mutex;

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        let cfg = fresh_default();
        std::fs::write(&path, toml::to_string_pretty(&cfg).unwrap()).expect("write");

        let h = ConfigCell::load(&path).expect("load");
        let observed: Arc<Mutex<Vec<usize>>> = Arc::new(Mutex::new(Vec::new()));
        let observed_clone = observed.clone();
        let _guard = h
            .watch_with(move |c| {
                observed_clone.lock().unwrap().push(c.inference.top_k);
                Ok(())
            })
            .expect("watch_with");

        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        let mut new_cfg = cfg.clone();
        new_cfg.inference.top_k = 9;
        std::fs::write(&path, toml::to_string_pretty(&new_cfg).unwrap()).expect("rewrite");

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            if observed.lock().unwrap().contains(&9) {
                break;
            }
            if std::time::Instant::now() > deadline {
                panic!(
                    "watch_with callback never observed top_k=9; saw {:?}",
                    observed.lock().unwrap(),
                );
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
    }

    /// Pins `try_reload`'s four-state outcome directly (bypassing notify flakiness):
    /// `Applied` on commit, `EqualToRunning` on byte-equal read (recovery signal
    /// clearing the rejection latch), `NoChange` on transient `NotFound`.
    #[test]
    fn try_reload_classifies_applied_equal_and_no_change() {
        use super::watcher::{ReloadOutcome, try_reload};
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        let cfg = fresh_default();
        std::fs::write(&path, toml::to_string_pretty(&cfg).unwrap()).expect("write");
        let inner = arc_swap::ArcSwap::from_pointee(cfg.clone());
        let lock = parking_lot::Mutex::new(());

        // Byte-equal to `inner` -> EqualToRunning (value-equality short-circuit).
        let outcome = try_reload(&path, &inner, &lock, &|_| Ok(()));
        assert_eq!(outcome, ReloadOutcome::EqualToRunning);

        let mut next = cfg.clone();
        next.inference.top_k = next.inference.top_k.saturating_add(1);
        std::fs::write(&path, toml::to_string_pretty(&next).unwrap()).expect("rewrite");
        let outcome = try_reload(&path, &inner, &lock, &|_| Ok(()));
        assert_eq!(outcome, ReloadOutcome::Applied);
        assert_eq!(inner.load().inference.top_k, next.inference.top_k);

        // Same NEW value re-presented -> EqualToRunning; the latch must clear so
        // restoring running content counts as recovery, not a transient skip.
        let outcome = try_reload(&path, &inner, &lock, &|_| Ok(()));
        assert_eq!(outcome, ReloadOutcome::EqualToRunning);

        // Missing file (ENOENT atomic-write race) -> NoChange, NOT Rejected.
        std::fs::remove_file(&path).unwrap();
        let outcome = try_reload(&path, &inner, &lock, &|_| Ok(()));
        assert_eq!(outcome, ReloadOutcome::NoChange);
    }

    /// Persistent read failures (not the transient `NotFound` of an atomic-write
    /// rename window) classify as `Rejected`, not `NoChange`, so
    /// `rejection_count`-gated alerts fire instead of silently serving stale config.
    /// Triggered via a directory at the path (EISDIR regardless of uid; chmod-000
    /// would be root-bypassed in CI).
    #[test]
    fn try_reload_persistent_read_err_is_rejected() {
        use super::watcher::{ReloadOutcome, try_reload};
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        std::fs::create_dir(&path).expect("mkdir at config path");

        let inner = arc_swap::ArcSwap::from_pointee(fresh_default());
        let lock = parking_lot::Mutex::new(());
        let outcome = try_reload(&path, &inner, &lock, &|_| Ok(()));

        assert_eq!(
            outcome,
            ReloadOutcome::Rejected,
            "persistent read err (EISDIR) must classify as Rejected so the rejection counter fires",
        );
    }
}
