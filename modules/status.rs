//! Daemon-wide status aggregator: subsystems [`StatusMonitor::register`] for a
//! `watch::Sender<Heartbeat>`; the snapshot endpoint marks a subsystem healthy
//! only if its latest heartbeat is at most `HEALTH_STALE_AFTER` old AND
//! `healthy = true` (unregistered subsystems are absent).
//!
//! Process-wide metrics (CPU/RSS/disk-free) are sampled by a background tokio task
//! and published via `ArcSwap<MetricsSnapshot>`, so each request is one wait-free
//! `load_full()` (no mutex/syscall/blocking-pool slot). A narrow CPU-usage-only
//! `RefreshKind` plus a single-PID memory-only `ProcessRefreshKind` (for RSS)
//! avoid `new_all()`'s ~5% CPU at 1 Hz on a Pi 5.
//! Memory is the daemon's own RSS (`top` RES), not system totals.

#![warn(missing_debug_implementations)]

use std::borrow::Cow;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;
use dashmap::DashMap;
use parking_lot::Mutex;
use serde::Serialize;
use sysinfo::{Disks, ProcessRefreshKind, ProcessesToUpdate, RefreshKind, System, get_current_pid};
use thiserror::Error;
use tokio::sync::watch;
use tokio::task::AbortHandle;

#[derive(Debug, Error)]
pub enum StatusError {
    /// `Internal`: daemon wiring is the sole caller, so a name collision is a
    /// programmer error.
    #[error("subsystem already registered: {name}")]
    AlreadyRegistered { name: String },
}

impl crate::common::error::Categorized for StatusError {
    fn kind(&self) -> crate::common::error::ErrorKind {
        crate::common::error::ErrorKind::Internal
    }
}

/// Heartbeat goes stale (unhealthy) after this; 5s covers the slowest outer-loop
/// period (opus paused: 1 Hz).
pub const HEALTH_STALE_AFTER: Duration = Duration::from_secs(5);

/// Host-metrics sample goes stale after this; 10x the production 500 ms period,
/// so the flag fires only on a genuinely wedged sampler.
pub const METRICS_STALE_AFTER: Duration = Duration::from_secs(5);

/// Floor on `start_sampler`'s `period`: `interval(Duration::ZERO)` panics, and
/// sub-`MINIMUM_CPU_UPDATE_INTERVAL` (~200 ms) ticks just repeat CPU samples.
pub const MIN_SAMPLER_PERIOD: Duration = Duration::from_millis(50);

#[derive(Clone, Debug, Serialize)]
pub struct Heartbeat {
    /// Monotonic emit instant; snapshot derives `age_ms`/`stale`. Skipped on the
    /// wire (the wire gets the derived `HeartbeatView`).
    #[serde(skip)]
    pub last_tick: Instant,
    pub healthy: bool,
    pub detail: Cow<'static, str>,
    /// Non-fatal degradation (e.g. inference behind real-time), orthogonal to
    /// `healthy`. Omitted from the wire when `None`; carried AS-IS once stale so
    /// the operator sees the last reason alongside `stale: true`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub degraded_reason: Option<Cow<'static, str>>,
}

impl Heartbeat {
    pub fn ok(detail: impl Into<Cow<'static, str>>) -> Self {
        Self {
            last_tick: Instant::now(),
            healthy: true,
            detail: detail.into(),
            degraded_reason: None,
        }
    }

    pub fn unhealthy(detail: impl Into<Cow<'static, str>>) -> Self {
        Self {
            last_tick: Instant::now(),
            healthy: false,
            detail: detail.into(),
            degraded_reason: None,
        }
    }

    /// `healthy: true` but with a non-fatal concern in `reason`.
    pub fn degraded(
        detail: impl Into<Cow<'static, str>>,
        reason: impl Into<Cow<'static, str>>,
    ) -> Self {
        Self {
            last_tick: Instant::now(),
            healthy: true,
            detail: detail.into(),
            degraded_reason: Some(reason.into()),
        }
    }
}

impl Default for Heartbeat {
    fn default() -> Self {
        Heartbeat::unhealthy("not started")
    }
}

/// Process-wide sample published by the background sampler; read wait-free via
/// `ArcSwap::load_full`. Default (zeros) is seen before
/// [`StatusMonitor::start_sampler`] runs.
#[derive(Clone, Debug, Default)]
pub struct MetricsSnapshot {
    /// System-wide CPU %, clamped 0..=100 (NOT 0..=100*N) on the production
    /// aarch64-linux target; other platforms vary.
    pub cpu_pct: f32,
    pub mem_rss_kb: u64,
    /// Free space (KiB) on the mount holding `workspace_root`; `0` when the path
    /// doesn't resolve to a known mount or none was supplied.
    pub disk_free_kb: u64,
    /// Monotonic publish instant; `None` before the first sample. Compare against
    /// `Instant::now()` to detect a wedged sampler.
    pub captured_at: Option<Instant>,
}

/// Aggregator; `Clone` is an Arc bump, all clones share state.
#[derive(Clone)]
pub struct StatusMonitor {
    inner: Arc<Inner>,
}

struct Inner {
    /// Separate `Arc` so the sampler holds a shared handle without owning
    /// `Arc<Inner>`, which would cycle (Inner owns the AbortHandle ending the loop).
    metrics: Arc<ArcSwap<MetricsSnapshot>>,
    /// `Cow` key: static names register zero-alloc, runtime names avoid `Box::leak`.
    heartbeats: DashMap<Cow<'static, str>, watch::Receiver<Heartbeat>>,
    started_at: Instant,
    /// `Option` so `new()` constructs without spawning; aborted on replacement and
    /// in `Drop` (dropping a `JoinHandle` only detaches).
    sampler: Mutex<Option<AbortHandle>>,
}

impl Drop for Inner {
    fn drop(&mut self) {
        // Abort on last-clone drop, else the detached loop runs and holds its
        // `Arc<ArcSwap<_>>` until runtime shutdown. Idempotent.
        if let Some(handle) = self.sampler.lock().take() {
            handle.abort();
        }
    }
}

impl std::fmt::Debug for StatusMonitor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StatusMonitor")
            .field("subsystems", &self.inner.heartbeats.len())
            .field("uptime_s", &self.inner.started_at.elapsed().as_secs())
            .finish_non_exhaustive()
    }
}

impl StatusMonitor {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Inner {
                metrics: Arc::new(ArcSwap::from_pointee(MetricsSnapshot::default())),
                heartbeats: DashMap::new(),
                started_at: Instant::now(),
                sampler: Mutex::new(None),
            }),
        }
    }

    /// Start the background sampler refreshing process-wide CPU + RSS + workspace
    /// disk-free at `period` into `metrics`. Must run inside a tokio runtime.
    /// Idempotent: a second call aborts the prior and starts fresh (e.g. config
    /// reload re-pointing `workspace_root`).
    ///
    /// `workspace_root` is the mount whose free-space is reported (`None` keeps
    /// `disk_free_kb` at 0); re-canonicalized every tick, so a directory created
    /// *after* `start_sampler` (cold-boot wiring) is picked up within one tick.
    /// Periods below `MIN_SAMPLER_PERIOD` (incl. zero, which would panic
    /// `interval`) clamp to that floor. Typical caller period: 500 ms.
    pub fn start_sampler(&self, workspace_root: Option<PathBuf>, period: Duration) {
        // Clamp (not error) keeps the call site infallible: a degenerate period
        // from config drift falls to the floor instead of downing the daemon.
        let requested = period;
        let period = period.max(MIN_SAMPLER_PERIOD);
        if requested != period {
            tracing::warn!(
                target: "status",
                requested_ms = requested.as_millis() as u64,
                clamped_ms = period.as_millis() as u64,
                "requested period below MIN_SAMPLER_PERIOD; clamped",
            );
        }
        let metrics_for_task = Arc::clone(&self.inner.metrics);
        // The one long-lived spawn not in `DrainRegistry`; a `sample_loop` panic dies
        // into tokio's JoinError silently but surfaces as `metrics_stale`. Abort the
        // prior under the lock before spawning; `abort()` is non-blocking, so a prior
        // tick may briefly race the fresh one and make `captured_at` non-monotonic
        // until the prior's next .await.
        let mut slot = self.inner.sampler.lock();
        if let Some(prior) = slot.take() {
            prior.abort();
        }
        let handle = tokio::spawn(sample_loop(metrics_for_task, workspace_root, period));
        *slot = Some(handle.abort_handle());
        // The kept `AbortHandle` is sole owner; dropping the `JoinHandle` detaches.
        drop(handle);
    }

    /// Register a subsystem; returns its heartbeat sender. Fail-fast on collision
    /// so a duplicate-name wiring bug surfaces at boot rather than masquerading as
    /// one healthy subsystem.
    pub fn register(
        &self,
        name: impl Into<Cow<'static, str>>,
    ) -> Result<watch::Sender<Heartbeat>, StatusError> {
        let name = name.into();
        let (tx, rx) = watch::channel(Heartbeat::default());
        // Entry API is atomic insert-iff-absent; check-then-insert could
        // double-register under contention.
        match self.inner.heartbeats.entry(name) {
            dashmap::Entry::Occupied(occ) => Err(StatusError::AlreadyRegistered {
                name: occ.key().to_string(),
            }),
            dashmap::Entry::Vacant(vac) => {
                vac.insert(rx);
                Ok(tx)
            }
        }
    }

    /// One wait-free snapshot of daemon state. `broadcast_lags` (cumulative
    /// WS-broadcast drop counts from `stream_io`) is passed in, `Default` when
    /// `stream_io` isn't wired. Host metrics read 0 if `start_sampler` was never
    /// called.
    pub fn snapshot(&self, broadcast_lags: BroadcastLagSnapshot) -> StatusSnapshot {
        let metrics = self.inner.metrics.load_full();
        let cpu_pct = metrics.cpu_pct;
        let mem_rss_kb = metrics.mem_rss_kb;
        let disk_free_kb = metrics.disk_free_kb;

        let uptime_s = self.inner.started_at.elapsed().as_secs();

        // No tick yet -> `age_ms = 0, stale = true`, so the signal reads "no sample
        // yet" rather than fresh zero data. `saturating_*` guards a future timestamp
        // (raw `now - ts` subtraction would panic on clock skew).
        let now = Instant::now();
        let (metrics_age_ms, metrics_stale) = match metrics.captured_at {
            Some(ts) => {
                let age = now.saturating_duration_since(ts);
                // Saturating cast: naked `as u64` would truncate the u128 millis.
                let age_ms = u64::try_from(age.as_millis()).unwrap_or(u64::MAX);
                (age_ms, age > METRICS_STALE_AFTER)
            }
            None => (0, true),
        };

        // Reuse `now` so heartbeat ages and metrics age share one instant.
        let mut subsystems = std::collections::BTreeMap::new();
        for entry in self.inner.heartbeats.iter() {
            let rx = entry.value();
            let hb = rx.borrow().clone();
            // `saturating_*` against future `last_tick` (cross-core clock skew).
            let age = now.saturating_duration_since(hb.last_tick);
            let stale = age > HEALTH_STALE_AFTER;
            let view = HeartbeatView {
                healthy: hb.healthy && !stale,
                detail: hb.detail.into_owned(),
                age_ms: u64::try_from(age.as_millis()).unwrap_or(u64::MAX),
                stale,
                degraded_reason: hb.degraded_reason.map(Cow::into_owned),
            };
            subsystems.insert(entry.key().to_string(), view);
        }

        let workspace = workspace_metrics::global()
            .map(|m| m.snapshot())
            .unwrap_or_default();
        let config_reload = config_metrics::global()
            .map(|h| h.snapshot())
            .unwrap_or_default();

        StatusSnapshot {
            cpu_pct,
            mem_rss_kb,
            disk_free_kb,
            metrics_age_ms,
            metrics_stale,
            uptime_s,
            subsystems,
            broadcast_audio_messages_dropped: broadcast_lags.audio_messages_dropped,
            broadcast_inference_messages_dropped: broadcast_lags.inference_messages_dropped,
            workspace,
            config_reload,
        }
    }
}

impl Default for StatusMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Background sampler publishing a fresh [`MetricsSnapshot`] every `period` until
/// its spawn handle is aborted. `period` is already clamped by the caller.
async fn sample_loop(
    metrics: Arc<ArcSwap<MetricsSnapshot>>,
    workspace_root: Option<PathBuf>,
    period: Duration,
) {
    debug_assert!(
        period >= MIN_SAMPLER_PERIOD,
        "sample_loop expects period >= MIN_SAMPLER_PERIOD; caller must clamp",
    );
    let refresh =
        RefreshKind::nothing().with_cpu(sysinfo::CpuRefreshKind::nothing().with_cpu_usage());
    let process_refresh = ProcessRefreshKind::nothing().with_memory();
    let pid = get_current_pid().ok();
    let mut sys = System::new_with_specifics(refresh);
    // Prime the CPU baseline: usage is a delta between refreshes, gated by
    // sysinfo's ~200 ms `MINIMUM_CPU_UPDATE_INTERVAL`.
    sys.refresh_specifics(refresh);
    if let Some(pid) = pid {
        sys.refresh_processes_specifics(ProcessesToUpdate::Some(&[pid]), false, process_refresh);
    }

    // Lazy: eager construction runs macOS's full-mount-table walk (>100 ms) in the
    // no-path/cold-boot case, blocking the sampler past its first tick. Reused once
    // built; only the path is re-resolved per tick.
    let mut disks: Option<Disks> = None;

    let mut interval = tokio::time::interval(period);
    // Skip backlog on a missed tick rather than firing back-to-back.
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    // First `tick()` returns immediately, publishing the priming sample.
    loop {
        interval.tick().await;
        sys.refresh_specifics(refresh);
        let mem_rss_kb = match pid {
            Some(pid) => {
                sys.refresh_processes_specifics(
                    ProcessesToUpdate::Some(&[pid]),
                    false,
                    process_refresh,
                );
                sys.process(pid).map(|p| p.memory() / 1024).unwrap_or(0)
            }
            None => 0,
        };
        let cpu_pct = sys.global_cpu_usage();
        // Re-canonicalize per tick (one lstat-class syscall): `start_sampler` may run
        // before `create_dir_all(workspace_root)`, so a once-cached result would pin
        // `disk_free_kb` at 0 forever. The `refresh(true)` mount-table walk runs inline
        // (sub-ms on aarch64-linux; per-tick `spawn_blocking` blew the macOS tick budget).
        let disk_free_kb = match workspace_root
            .as_deref()
            .and_then(|p| std::fs::canonicalize(p).ok())
        {
            Some(canonical) => {
                let disks = disks.get_or_insert_with(Disks::new_with_refreshed_list);
                // `true` re-queries free space; `false` would only refresh the list.
                disks.refresh(true);
                disks
                    .list()
                    .iter()
                    .filter(|d| canonical.starts_with(d.mount_point()))
                    .max_by_key(|d| d.mount_point().as_os_str().len())
                    .map(|d| d.available_space() / 1024)
                    .unwrap_or(0)
            }
            None => 0,
        };
        metrics.store(Arc::new(MetricsSnapshot {
            cpu_pct,
            mem_rss_kb,
            disk_free_kb,
            captured_at: Some(Instant::now()),
        }));
    }
}

/// Wire form of [`Heartbeat`]: the monotonic `Instant` resolved to `age_ms` +
/// `stale`.
#[derive(Clone, Debug, Serialize)]
pub struct HeartbeatView {
    pub healthy: bool,
    pub detail: String,
    pub age_ms: u64,
    /// True once age exceeds `HEALTH_STALE_AFTER`.
    pub stale: bool,
    /// Omitted from JSON when `None`, keeping the wire byte-identical for
    /// subsystems that don't report it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub degraded_reason: Option<String>,
}

/// Aggregated daemon health snapshot for `/api/v1/status`: per-subsystem
/// [`Heartbeat`]s plus host metrics, built on demand by
/// [`StatusMonitor::snapshot`] (metrics half read wait-free).
#[derive(Clone, Debug, Serialize)]
pub struct StatusSnapshot {
    pub cpu_pct: f32,
    /// Daemon RSS in KiB (`top`/`htop` RES, VmRSS on Linux); system totals are
    /// deliberately not reported. 0 if the PID couldn't be determined.
    pub mem_rss_kb: u64,
    pub disk_free_kb: u64,
    /// Millis since the sampler last published; `0` when no sample yet. Pair with
    /// `metrics_stale` to distinguish "no sample" from "fresh sample".
    pub metrics_age_ms: u64,
    /// `true` when the sample is older than [`METRICS_STALE_AFTER`] or none was
    /// published. Operators MUST check this before trusting the metric values: a
    /// wedged sampler reports last-good values + `stale`, not zeroes.
    pub metrics_stale: bool,
    pub uptime_s: u64,
    pub subsystems: std::collections::BTreeMap<String, HeartbeatView>,
    /// Cumulative WS audio-broadcast messages dropped since boot (summed
    /// `RecvError::Lagged(n)`): TOTAL DROPPED MESSAGES, not lag-event count.
    #[serde(default)]
    pub broadcast_audio_messages_dropped: u64,
    /// Same semantics, inference broadcast.
    #[serde(default)]
    pub broadcast_inference_messages_dropped: u64,
    /// Workspace-side counters; zero when no global was installed (tests).
    #[serde(default)]
    pub workspace: WorkspaceMetricsSnapshot,
    /// Cumulative config-reload counters. A `rejected` bump means the watcher
    /// discarded a reload and the in-memory snapshot still reflects prior on-disk
    /// content, so a later API mutation would overwrite the operator's new disk
    /// bytes with old-derived ones. Zero when no global.
    #[serde(default)]
    pub config_reload: ConfigReloadSnapshot,
}

pub use crate::common::traits::lag_source::BroadcastLagSnapshot;

pub mod reporter;
pub use reporter::StatusReporter;

pub mod workspace_metrics;
pub use workspace_metrics::{SseClientGuard, WorkspaceMetrics, WorkspaceMetricsSnapshot};

pub mod config_metrics;
pub use config_metrics::{ConfigReloadHandles, ConfigReloadSnapshot};

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn register_and_snapshot_round_trip() {
        let mon = StatusMonitor::new();
        mon.start_sampler(None, Duration::from_millis(50));
        let tx = mon.register("test_subsystem").expect("register");
        tx.send(Heartbeat::ok("running")).expect("send");
        // Slack against scheduler jitter on busy CI.
        tokio::time::sleep(Duration::from_millis(150)).await;
        let snap = mon.snapshot(BroadcastLagSnapshot::default());
        let view = snap
            .subsystems
            .get("test_subsystem")
            .expect("registered subsystem");
        assert!(view.healthy);
        assert_eq!(view.detail, "running");
        assert!(
            view.age_ms < 1000,
            "fresh heartbeat ages quickly: {}",
            view.age_ms
        );
        assert!(!view.stale);

        assert!(snap.mem_rss_kb > 0, "mem_rss_kb=0 -- sampler broken?");
    }

    #[tokio::test(start_paused = true)]
    async fn stale_heartbeat_is_unhealthy() {
        let mon = StatusMonitor::new();
        let tx = mon.register("stalled").expect("register");
        tx.send(Heartbeat::ok("ok at t=0")).expect("send");

        tokio::time::advance(HEALTH_STALE_AFTER + Duration::from_secs(1)).await;

        // `Instant::now()` is real-clock (unaffected by paused time), so age manually.
        let aged = Heartbeat {
            last_tick: Instant::now() - Duration::from_secs(10),
            healthy: true,
            detail: "should be marked stale".into(),
            degraded_reason: None,
        };
        tx.send(aged).expect("send aged");
        let snap = mon.snapshot(BroadcastLagSnapshot::default());
        let view = snap.subsystems.get("stalled").expect("registered");
        assert!(view.stale, "expected stale: {view:?}");
        assert!(!view.healthy, "stale should imply unhealthy");
    }

    #[test]
    fn unhealthy_heartbeat_propagates() {
        let mon = StatusMonitor::new();
        let tx = mon.register("oncall").expect("register");
        tx.send(Heartbeat::unhealthy("rknn returned -1"))
            .expect("send");
        let snap = mon.snapshot(BroadcastLagSnapshot::default());
        let v = snap.subsystems.get("oncall").unwrap();
        assert!(!v.healthy);
        assert_eq!(v.detail, "rknn returned -1");
    }

    /// No sampler -> zero metrics (request path never touches the filesystem).
    #[tokio::test]
    async fn snapshot_without_sampler_returns_zero_metrics() {
        let mon = StatusMonitor::new();
        let snap = mon.snapshot(BroadcastLagSnapshot::default());
        assert_eq!(snap.cpu_pct, 0.0);
        assert_eq!(snap.mem_rss_kb, 0);
        assert_eq!(snap.disk_free_kb, 0);
    }

    #[tokio::test]
    async fn sampler_publishes_real_rss_within_first_tick() {
        let mon = StatusMonitor::new();
        // 50 ms cadence so the test doesn't wait the production 500 ms.
        mon.start_sampler(None, Duration::from_millis(50));
        tokio::time::sleep(Duration::from_millis(150)).await;
        let snap = mon.snapshot(BroadcastLagSnapshot::default());
        assert!(
            snap.mem_rss_kb > 0,
            "sampler should have published a real RSS sample by now; \
             got {snap:?}"
        );
    }

    #[test]
    fn re_register_returns_already_registered() {
        let mon = StatusMonitor::new();
        let tx1 = mon.register("flappy").expect("first register");
        let err = mon
            .register("flappy")
            .expect_err("second register must fail");
        match err {
            StatusError::AlreadyRegistered { name } => assert_eq!(name, "flappy"),
        }
        // First registration is not poisoned by the failed second.
        tx1.send(Heartbeat::ok("still alive")).expect("send");
        let snap = mon.snapshot(BroadcastLagSnapshot::default());
        let v = snap.subsystems.get("flappy").expect("registered");
        assert_eq!(v.detail, "still alive");
    }

    #[test]
    fn register_accepts_owned_string_name() {
        let mon = StatusMonitor::new();
        let dynamic_name = format!("inference#{}", 7);
        let tx = mon.register(dynamic_name.clone()).expect("register owned");
        tx.send(Heartbeat::ok("gen-7 up")).expect("send");
        let snap = mon.snapshot(BroadcastLagSnapshot::default());
        let v = snap
            .subsystems
            .get(&dynamic_name)
            .expect("registered under owned name");
        assert_eq!(v.detail, "gen-7 up");
    }

    #[test]
    fn heartbeat_degraded_surfaces_reason_on_wire_and_keeps_healthy_true() {
        let mon = StatusMonitor::new();
        let tx = mon.register("inference").expect("register");
        tx.send(Heartbeat::degraded(
            "running but stale",
            "rknn frame interval > 200ms (target 250ms)",
        ))
        .expect("send");
        let snap = mon.snapshot(BroadcastLagSnapshot::default());
        let v = snap.subsystems.get("inference").expect("registered");
        assert!(v.healthy, "degraded does not flip healthy off");
        assert_eq!(
            v.degraded_reason.as_deref(),
            Some("rknn frame interval > 200ms (target 250ms)"),
        );

        let json = serde_json::to_string(v).expect("serialize");
        assert!(
            json.contains("\"degraded_reason\":\"rknn frame interval > 200ms (target 250ms)\""),
            "expected degraded_reason in wire JSON: {json}",
        );
    }

    /// Wire-compat gate: the `ok` path must NOT add a JSON field; baseline
    /// `{healthy, detail, age_ms, stale}` stays byte-identical. The assertion
    /// quotes the key so a `detail` containing the literal text can't satisfy it.
    #[test]
    fn heartbeat_ok_omits_degraded_reason_field_from_wire() {
        let mon = StatusMonitor::new();
        let tx = mon.register("audio_io").expect("register");
        tx.send(Heartbeat::ok("running")).expect("send");
        let snap = mon.snapshot(BroadcastLagSnapshot::default());
        let v = snap.subsystems.get("audio_io").expect("registered");
        assert_eq!(v.degraded_reason, None);
        let json = serde_json::to_string(v).expect("serialize");
        assert!(
            !json.contains("\"degraded_reason\""),
            "ok-path heartbeat must not emit degraded_reason field: {json}",
        );
    }

    /// Pins stale-carry-forward: a heartbeat whose last send was `degraded` keeps
    /// `degraded_reason` past the staleness window, so the dashboard shows both
    /// `stale: true` and the prior reason.
    #[test]
    fn stale_degraded_heartbeat_preserves_reason_on_wire() {
        let mon = StatusMonitor::new();
        let tx = mon.register("inference").expect("register");
        let aged = Heartbeat {
            last_tick: Instant::now() - Duration::from_secs(10),
            healthy: true,
            detail: "rknn slow".into(),
            degraded_reason: Some(Cow::Borrowed("frame > 250ms target")),
        };
        tx.send(aged).expect("send aged degraded");
        let snap = mon.snapshot(BroadcastLagSnapshot::default());
        let v = snap.subsystems.get("inference").expect("registered");
        assert!(v.stale, "expected stale: {v:?}");
        assert_eq!(
            v.degraded_reason.as_deref(),
            Some("frame > 250ms target"),
            "stale heartbeat should carry the prior degraded_reason forward",
        );
        assert!(!v.healthy, "stale entry shows healthy=false");
        let json = serde_json::to_string(v).expect("serialize");
        assert!(
            json.contains("\"degraded_reason\":\"frame > 250ms target\""),
            "stale-but-was-degraded heartbeat must preserve the field: {json}",
        );
    }

    /// Replacing the sampler must abort the prior task; else the dropped
    /// `JoinHandle` detaches (not aborts) and the old loop leaks across every
    /// config reload. Detected by `captured_at` ceasing to advance once both
    /// samplers are aborted -- a leaked one would keep bumping it.
    #[tokio::test]
    async fn replacing_sampler_aborts_predecessor() {
        let mon = StatusMonitor::new();
        // `MIN_SAMPLER_PERIOD` keeps the clamp warn silent; ~6 ticks in the 300 ms
        // poll window below.
        let period = MIN_SAMPLER_PERIOD;
        mon.start_sampler(None, period);
        // Warm-up >= 2 ticks with margin for CI jitter.
        tokio::time::sleep(Duration::from_millis(120)).await;
        let before = mon
            .inner
            .metrics
            .load_full()
            .captured_at
            .expect("first sampler must publish at least one sample");

        mon.start_sampler(None, period);

        // Drop exercises `Drop for Inner` on the replacement sampler.
        let metrics = Arc::clone(&mon.inner.metrics);
        drop(mon);

        // After both aborts `captured_at` must stabilize; 300 ms covers ~6 ticks of
        // a leaked 50 ms sampler.
        tokio::time::sleep(Duration::from_millis(300)).await;
        let stable_marker = metrics
            .load_full()
            .captured_at
            .expect("snapshot retains captured_at across abort");
        // A surviving task would advance `captured_at` within this window.
        tokio::time::sleep(Duration::from_millis(200)).await;
        let after = metrics
            .load_full()
            .captured_at
            .expect("snapshot retains captured_at after second wait");
        assert_eq!(
            stable_marker, after,
            "captured_at advanced after both samplers were aborted -- a sampler leaked: \
             before_replace={before:?} stable={stable_marker:?} after={after:?}",
        );
    }

    /// `disk_free_kb` recovers via per-tick re-canonicalize when the workspace dir
    /// is created *after* `start_sampler` (cold-boot case).
    #[tokio::test]
    async fn disk_free_kb_recovers_after_workspace_created() {
        // Nanos suffix so parallel runs don't collide in the process-wide temp dir.
        let base = std::env::temp_dir();
        let unique = format!(
            "acousticslab_status_cold_boot_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
        );
        let workspace = base.join(unique);
        // Precondition: path must NOT exist, else the test degrades to the easy case.
        assert!(
            !workspace.exists(),
            "test setup expects a non-existent workspace path: {workspace:?}",
        );

        let mon = StatusMonitor::new();
        let period = Duration::from_millis(50);
        mon.start_sampler(Some(workspace.clone()), period);

        // Nothing to canonicalize while the dir is missing -> `disk_free_kb` 0.
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert_eq!(
            mon.snapshot(BroadcastLagSnapshot::default()).disk_free_kb,
            0,
            "disk_free_kb should be 0 while workspace is missing",
        );

        std::fs::create_dir_all(&workspace).expect("create workspace");

        // ~10 ticks for the sampler to observe the new directory.
        let deadline = Instant::now() + Duration::from_millis(500);
        let mut got_nonzero = 0u64;
        while Instant::now() < deadline {
            tokio::time::sleep(period).await;
            let snap = mon.snapshot(BroadcastLagSnapshot::default());
            if snap.disk_free_kb > 0 {
                got_nonzero = snap.disk_free_kb;
                break;
            }
        }
        let _ = std::fs::remove_dir_all(&workspace);

        assert!(
            got_nonzero > 0,
            "disk_free_kb did not recover within bounded poll window after \
             workspace creation; sampler did not re-canonicalize",
        );
    }

    /// `start_sampler(_, Duration::ZERO)` clamps to `MIN_SAMPLER_PERIOD` instead
    /// of panicking `interval`.
    #[tokio::test]
    async fn start_sampler_clamps_zero_period() {
        let mon = StatusMonitor::new();
        mon.start_sampler(None, Duration::ZERO);
        // > MIN_SAMPLER_PERIOD so the clamped sampler has published.
        tokio::time::sleep(MIN_SAMPLER_PERIOD * 4).await;
        let snap = mon.snapshot(BroadcastLagSnapshot::default());
        // Populated RSS proves it clamped and ran (an unclamped period would have
        // panicked at the call above).
        assert!(
            snap.mem_rss_kb > 0,
            "clamped-period sampler should have published a real RSS value: {snap:?}",
        );
    }
}
