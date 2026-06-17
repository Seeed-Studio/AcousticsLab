//! Workspace-side counters for `GET /api/v1/status`, aggregated by
//! [`WorkspaceMetrics`] with a bounded-ring p99 estimator for atomic-rewrite
//! latency.
//!
//! Held as a process-wide `OnceLock` ([`global`]); an absent global (test
//! fixtures) is a zero-cost no-op, so threading `Option<Arc<_>>` through every
//! call site buys nothing - tests [`install_for_tests`] their own handle.
//!
//! `Ordering::Relaxed` everywhere is correct, not lazy: these counters
//! synchronize nothing and the snapshot tolerates per-counter skew.
//! `sse_clients_current` is `AtomicI64` so a stale double-drop goes negative
//! rather than wrapping.

use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::time::Duration;

use parking_lot::Mutex;
use serde::Serialize;

/// Bounded ring depth for the p99 write-duration estimator (~13 s of peak churn).
pub const WRITE_DURATION_RING_DEPTH: usize = 256;

/// Snapshot of [`WorkspaceMetrics`], embedded in
/// [`crate::status::StatusSnapshot`] under `workspace`. `*_total` are
/// cumulative since boot; `*_p99_us` derive on demand from a bounded ring.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct WorkspaceMetricsSnapshot {
    pub assets_uploaded_total: u64,
    pub bytes_uploaded_total: u64,
    pub workspace_core_writes_total: u64,
    pub head_index_writes_total: u64,
    pub dataset_mutations_rejected_total: u64,
    pub converter_mutations_rejected_total: u64,
    pub workspace_core_write_p99_us: u64,
    pub head_index_write_p99_us: u64,
    /// Job-event broadcasts dropped because every receiver lagged out; distinct
    /// from `JobRegistryCounters::events_dropped_total` (ring-overflow drops).
    pub job_events_dropped_total: u64,
    /// Live SSE client count (signed; see module-level rationale).
    pub sse_clients_current: i64,
    pub boot_orphans_swept_total: u64,
    /// Workspace-level recovery failures, kept separate from
    /// `boot_workspace_enumeration_failures_total` (filesystem-level, e.g. EIO)
    /// so the same `workspaces_scanned < expected` symptom triages distinctly.
    pub boot_workspace_recovery_failures_total: u64,
    pub boot_workspace_enumeration_failures_total: u64,
    /// `.tmp/` orphans reaped by the runtime `storage_reaper`; closes the
    /// post-hard-crash gap the boot sweep cannot.
    pub tmp_orphans_reaped_total: u64,
    pub log_files_pruned_total: u64,
    /// Kept apart from `storage_reaper_failures_total` for independent triage.
    pub log_retention_failures_total: u64,
    pub storage_reaper_failures_total: u64,
}

/// Aggregated workspace-side counter surface. Lock-free except the p99 ring,
/// whose `parking_lot::Mutex` covers only push + recompute, never an `.await`.
#[derive(Debug, Default)]
pub struct WorkspaceMetrics {
    assets_uploaded_total: AtomicU64,
    bytes_uploaded_total: AtomicU64,
    workspace_core_writes_total: AtomicU64,
    head_index_writes_total: AtomicU64,
    dataset_mutations_rejected_total: AtomicU64,
    converter_mutations_rejected_total: AtomicU64,
    job_events_dropped_total: AtomicU64,
    sse_clients_current: AtomicI64,
    boot_orphans_swept_total: AtomicU64,
    boot_workspace_recovery_failures_total: AtomicU64,
    boot_workspace_enumeration_failures_total: AtomicU64,
    tmp_orphans_reaped_total: AtomicU64,
    log_files_pruned_total: AtomicU64,
    log_retention_failures_total: AtomicU64,
    storage_reaper_failures_total: AtomicU64,
    workspace_core_write_samples: Mutex<DurationRing>,
    head_index_write_samples: Mutex<DurationRing>,
}

impl WorkspaceMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Recomputes both p99 estimates under short critical sections; the
    /// `parking_lot` locks never bridge an `.await`, so this is request-safe.
    pub fn snapshot(&self) -> WorkspaceMetricsSnapshot {
        WorkspaceMetricsSnapshot {
            assets_uploaded_total: self.assets_uploaded_total.load(Ordering::Relaxed),
            bytes_uploaded_total: self.bytes_uploaded_total.load(Ordering::Relaxed),
            workspace_core_writes_total: self.workspace_core_writes_total.load(Ordering::Relaxed),
            head_index_writes_total: self.head_index_writes_total.load(Ordering::Relaxed),
            dataset_mutations_rejected_total: self
                .dataset_mutations_rejected_total
                .load(Ordering::Relaxed),
            converter_mutations_rejected_total: self
                .converter_mutations_rejected_total
                .load(Ordering::Relaxed),
            workspace_core_write_p99_us: self.workspace_core_write_samples.lock().p99_us(),
            head_index_write_p99_us: self.head_index_write_samples.lock().p99_us(),
            job_events_dropped_total: self.job_events_dropped_total.load(Ordering::Relaxed),
            sse_clients_current: self.sse_clients_current.load(Ordering::Relaxed),
            boot_orphans_swept_total: self.boot_orphans_swept_total.load(Ordering::Relaxed),
            boot_workspace_recovery_failures_total: self
                .boot_workspace_recovery_failures_total
                .load(Ordering::Relaxed),
            boot_workspace_enumeration_failures_total: self
                .boot_workspace_enumeration_failures_total
                .load(Ordering::Relaxed),
            tmp_orphans_reaped_total: self.tmp_orphans_reaped_total.load(Ordering::Relaxed),
            log_files_pruned_total: self.log_files_pruned_total.load(Ordering::Relaxed),
            log_retention_failures_total: self.log_retention_failures_total.load(Ordering::Relaxed),
            storage_reaper_failures_total: self
                .storage_reaper_failures_total
                .load(Ordering::Relaxed),
        }
    }

    pub fn record_upload(&self, bytes: u64) {
        self.assets_uploaded_total.fetch_add(1, Ordering::Relaxed);
        self.bytes_uploaded_total
            .fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn record_workspace_core_write(&self, duration: Duration) {
        self.workspace_core_writes_total
            .fetch_add(1, Ordering::Relaxed);
        self.workspace_core_write_samples
            .lock()
            .push_micros(duration_micros(duration));
    }

    pub fn record_head_index_write(&self, duration: Duration) {
        self.head_index_writes_total.fetch_add(1, Ordering::Relaxed);
        self.head_index_write_samples
            .lock()
            .push_micros(duration_micros(duration));
    }

    pub fn record_dataset_mutation_rejected(&self) {
        self.dataset_mutations_rejected_total
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_converter_mutation_rejected(&self) {
        self.converter_mutations_rejected_total
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_job_events_dropped(&self, n: u64) {
        self.job_events_dropped_total
            .fetch_add(n, Ordering::Relaxed);
    }

    /// +1 on construct, -1 on [`SseClientGuard`] drop.
    pub fn sse_client_guard(self: &Arc<Self>) -> SseClientGuard {
        self.sse_clients_current.fetch_add(1, Ordering::Relaxed);
        SseClientGuard {
            metrics: Arc::clone(self),
        }
    }

    pub fn record_boot_orphans_swept(&self, n: u64) {
        self.boot_orphans_swept_total
            .fetch_add(n, Ordering::Relaxed);
    }

    pub fn record_boot_workspace_recovery_failures(&self, n: u64) {
        self.boot_workspace_recovery_failures_total
            .fetch_add(n, Ordering::Relaxed);
    }

    pub fn record_boot_workspace_enumeration_failures(&self, n: u64) {
        self.boot_workspace_enumeration_failures_total
            .fetch_add(n, Ordering::Relaxed);
    }

    /// Takes bare counters, not the report struct, to keep the layer edge
    /// `status -> file_mgr` out of the dependency graph.
    pub fn record_storage_sweep(&self, tmp_orphans_reaped: u64, failures: u64) {
        self.tmp_orphans_reaped_total
            .fetch_add(tmp_orphans_reaped, Ordering::Relaxed);
        self.storage_reaper_failures_total
            .fetch_add(failures, Ordering::Relaxed);
    }

    /// The hook short-circuits zero-on-zero, so a no-op sweep never reaches here.
    pub fn record_logs_pruned(&self, pruned: u64, failures: u64) {
        self.log_files_pruned_total
            .fetch_add(pruned, Ordering::Relaxed);
        self.log_retention_failures_total
            .fetch_add(failures, Ordering::Relaxed);
    }
}

/// RAII guard from [`WorkspaceMetrics::sse_client_guard`]; decrements
/// `sse_clients_current` on drop.
#[derive(Debug)]
pub struct SseClientGuard {
    metrics: Arc<WorkspaceMetrics>,
}

impl Drop for SseClientGuard {
    fn drop(&mut self) {
        self.metrics
            .sse_clients_current
            .fetch_sub(1, Ordering::Relaxed);
    }
}

static GLOBAL: OnceLock<Arc<WorkspaceMetrics>> = OnceLock::new();

/// One-shot: `Err(existing)` on re-init.
pub fn install_global(metrics: Arc<WorkspaceMetrics>) -> Result<(), Arc<WorkspaceMetrics>> {
    GLOBAL.set(metrics)
}

/// Absent means counters disabled.
pub fn global() -> Option<&'static Arc<WorkspaceMetrics>> {
    GLOBAL.get()
}

/// First-call-wins install; safe under cargo's per-binary fresh-process
/// isolation. Tests needing per-fixture isolation construct their own `Arc`.
#[doc(hidden)]
pub fn install_for_tests(metrics: Arc<WorkspaceMetrics>) {
    let _ = GLOBAL.set(metrics);
}

/// Saturating at `u64::MAX`.
fn duration_micros(d: Duration) -> u64 {
    u64::try_from(d.as_micros()).unwrap_or(u64::MAX)
}

/// Fixed-capacity ([`WRITE_DURATION_RING_DEPTH`]) ring of microsecond
/// durations; `push_micros` evicts the oldest on overflow (constant-time),
/// p99 sorts a snapshot off the hot path.
#[derive(Debug)]
struct DurationRing {
    /// Oldest at front, newest at back.
    samples: std::collections::VecDeque<u64>,
}

impl Default for DurationRing {
    fn default() -> Self {
        Self::new()
    }
}

impl DurationRing {
    /// Pre-sized to capacity so steady-state fill pays no reallocations.
    fn new() -> Self {
        Self {
            samples: std::collections::VecDeque::with_capacity(WRITE_DURATION_RING_DEPTH),
        }
    }

    fn push_micros(&mut self, micros: u64) {
        if self.samples.len() >= WRITE_DURATION_RING_DEPTH {
            self.samples.pop_front();
        }
        self.samples.push_back(micros);
    }

    /// Nearest-rank p99 in microseconds (`ceil(0.99 * n) - 1`); 0 when empty,
    /// effectively the max for `n < 100`.
    fn p99_us(&self) -> u64 {
        if self.samples.is_empty() {
            return 0;
        }
        let mut sorted: Vec<u64> = self.samples.iter().copied().collect();
        sorted.sort_unstable();
        let n = sorted.len();
        let idx = (((n as f64) * 0.99).ceil() as usize)
            .saturating_sub(1)
            .min(n - 1);
        sorted[idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn record_upload_increments_assets_and_bytes() {
        let m = WorkspaceMetrics::new();
        m.record_upload(1024);
        m.record_upload(2048);
        let snap = m.snapshot();
        assert_eq!(snap.assets_uploaded_total, 2);
        assert_eq!(snap.bytes_uploaded_total, 1024 + 2048);
    }

    #[test]
    fn record_workspace_core_write_updates_counter_and_p99() {
        let m = WorkspaceMetrics::new();
        m.record_workspace_core_write(Duration::from_millis(5));
        m.record_workspace_core_write(Duration::from_millis(10));
        let snap = m.snapshot();
        assert_eq!(snap.workspace_core_writes_total, 2);
        assert_eq!(snap.workspace_core_write_p99_us, 10_000);
    }

    #[test]
    fn record_head_index_write_updates_counter_and_p99() {
        let m = WorkspaceMetrics::new();
        m.record_head_index_write(Duration::from_micros(750));
        m.record_head_index_write(Duration::from_micros(1500));
        m.record_head_index_write(Duration::from_micros(500));
        let snap = m.snapshot();
        assert_eq!(snap.head_index_writes_total, 3);
        assert_eq!(snap.head_index_write_p99_us, 1500);
    }

    #[test]
    fn p99_uses_nearest_rank_at_large_sample_sizes() {
        let m = WorkspaceMetrics::new();
        // p99 = sorted[ceil(0.99*100)-1] = sorted[98] = 99 ms.
        for i in 1..=100u64 {
            m.record_workspace_core_write(Duration::from_millis(i));
        }
        let snap = m.snapshot();
        assert_eq!(snap.workspace_core_write_p99_us, 99 * 1000);
    }

    #[test]
    fn duration_ring_evicts_oldest_at_capacity() {
        let m = WorkspaceMetrics::new();
        for i in 0..(WRITE_DURATION_RING_DEPTH + 10) {
            m.record_workspace_core_write(Duration::from_micros(i as u64));
        }
        // Ring holds [10..=265]; p99 = sorted[253] = 10 + 253 = 263.
        let snap = m.snapshot();
        assert_eq!(snap.workspace_core_write_p99_us, 263);
    }

    #[test]
    fn record_dataset_mutation_rejected_monotonic() {
        let m = WorkspaceMetrics::new();
        for _ in 0..7 {
            m.record_dataset_mutation_rejected();
        }
        assert_eq!(m.snapshot().dataset_mutations_rejected_total, 7);
    }

    #[test]
    fn record_converter_mutation_rejected_separate_from_dataset() {
        let m = WorkspaceMetrics::new();
        for _ in 0..3 {
            m.record_dataset_mutation_rejected();
        }
        for _ in 0..5 {
            m.record_converter_mutation_rejected();
        }
        let snap = m.snapshot();
        assert_eq!(snap.dataset_mutations_rejected_total, 3);
        assert_eq!(snap.converter_mutations_rejected_total, 5);
    }

    #[test]
    fn record_job_events_dropped_accumulates() {
        let m = WorkspaceMetrics::new();
        m.record_job_events_dropped(3);
        m.record_job_events_dropped(5);
        assert_eq!(m.snapshot().job_events_dropped_total, 8);
    }

    #[test]
    fn sse_client_guard_increments_then_decrements() {
        let m = Arc::new(WorkspaceMetrics::new());
        let g1 = m.sse_client_guard();
        let g2 = m.sse_client_guard();
        assert_eq!(m.snapshot().sse_clients_current, 2);
        drop(g1);
        assert_eq!(m.snapshot().sse_clients_current, 1);
        drop(g2);
        assert_eq!(m.snapshot().sse_clients_current, 0);
    }

    #[test]
    fn boot_orphans_swept_accumulates() {
        let m = WorkspaceMetrics::new();
        m.record_boot_orphans_swept(3);
        m.record_boot_orphans_swept(2);
        assert_eq!(m.snapshot().boot_orphans_swept_total, 5);
    }

    #[test]
    fn boot_workspace_recovery_failures_accumulate() {
        let m = WorkspaceMetrics::new();
        m.record_boot_workspace_recovery_failures(2);
        m.record_boot_workspace_recovery_failures(0);
        m.record_boot_workspace_recovery_failures(3);
        assert_eq!(m.snapshot().boot_workspace_recovery_failures_total, 5);
    }

    /// Guards that recovery and enumeration counters stay independent surfaces.
    #[test]
    fn boot_workspace_enumeration_failures_accumulate_independently() {
        let m = WorkspaceMetrics::new();
        m.record_boot_workspace_enumeration_failures(2);
        m.record_boot_workspace_enumeration_failures(0);
        m.record_boot_workspace_enumeration_failures(3);
        let s = m.snapshot();
        assert_eq!(s.boot_workspace_enumeration_failures_total, 5);
        assert_eq!(s.boot_workspace_recovery_failures_total, 0);

        m.record_boot_workspace_recovery_failures(7);
        let s = m.snapshot();
        assert_eq!(s.boot_workspace_enumeration_failures_total, 5);
        assert_eq!(s.boot_workspace_recovery_failures_total, 7);
    }

    /// Guards positional arg order (same-typed swap fails CI) and that the
    /// storage sweep leaves the log-retention counters untouched.
    #[test]
    fn record_storage_sweep_accumulates() {
        let m = WorkspaceMetrics::new();
        m.record_storage_sweep(3, 1);
        m.record_storage_sweep(2, 0);
        m.record_storage_sweep(0, 4);
        let s = m.snapshot();
        assert_eq!(s.tmp_orphans_reaped_total, 5);
        assert_eq!(s.storage_reaper_failures_total, 5);
        assert_eq!(s.log_files_pruned_total, 0);
        assert_eq!(s.log_retention_failures_total, 0);
    }

    /// Pins the arg order and independence from the storage-sweep counters.
    #[test]
    fn record_logs_pruned_accumulates() {
        let m = WorkspaceMetrics::new();
        m.record_logs_pruned(3, 0);
        m.record_logs_pruned(2, 1);
        m.record_logs_pruned(0, 4);
        let s = m.snapshot();
        assert_eq!(s.log_files_pruned_total, 5);
        assert_eq!(s.log_retention_failures_total, 5);
        assert_eq!(s.tmp_orphans_reaped_total, 0);
        assert_eq!(s.storage_reaper_failures_total, 0);
    }

    #[test]
    fn snapshot_consistency_under_concurrent_writers() {
        // 8 threads * 1k events: no torn or missed atomic increments.
        let m = Arc::new(WorkspaceMetrics::new());
        let mut handles = Vec::new();
        for _ in 0..8 {
            let m_clone = Arc::clone(&m);
            handles.push(thread::spawn(move || {
                for _ in 0..1000 {
                    m_clone.record_upload(1024);
                    m_clone.record_workspace_core_write(Duration::from_micros(10));
                    m_clone.record_head_index_write(Duration::from_micros(20));
                    m_clone.record_dataset_mutation_rejected();
                    m_clone.record_job_events_dropped(1);
                    m_clone.record_boot_orphans_swept(1);
                }
            }));
        }
        for h in handles {
            h.join().expect("worker join");
        }
        let snap = m.snapshot();
        assert_eq!(snap.assets_uploaded_total, 8 * 1000);
        assert_eq!(snap.bytes_uploaded_total, 8 * 1000 * 1024);
        assert_eq!(snap.workspace_core_writes_total, 8 * 1000);
        assert_eq!(snap.head_index_writes_total, 8 * 1000);
        assert_eq!(snap.dataset_mutations_rejected_total, 8 * 1000);
        assert_eq!(snap.job_events_dropped_total, 8 * 1000);
        assert_eq!(snap.boot_orphans_swept_total, 8 * 1000);
    }

    #[test]
    fn snapshot_default_has_zero_counters() {
        let m = WorkspaceMetrics::new();
        let snap = m.snapshot();
        assert_eq!(snap.assets_uploaded_total, 0);
        assert_eq!(snap.bytes_uploaded_total, 0);
        assert_eq!(snap.workspace_core_writes_total, 0);
        assert_eq!(snap.head_index_writes_total, 0);
        assert_eq!(snap.dataset_mutations_rejected_total, 0);
        assert_eq!(snap.converter_mutations_rejected_total, 0);
        assert_eq!(snap.workspace_core_write_p99_us, 0);
        assert_eq!(snap.head_index_write_p99_us, 0);
        assert_eq!(snap.job_events_dropped_total, 0);
        assert_eq!(snap.sse_clients_current, 0);
        assert_eq!(snap.boot_orphans_swept_total, 0);
        assert_eq!(snap.boot_workspace_recovery_failures_total, 0);
        assert_eq!(snap.boot_workspace_enumeration_failures_total, 0);
    }
}
