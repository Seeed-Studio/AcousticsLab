//! In-process job registry: per-`JobType` concurrency-cap admission gate,
//! `WorkspaceDelete`-exclusion conflict detection (see the `*_conflicts_*`
//! predicates), bounded recent-history, per-job SSE-replay event ring +
//! broadcast channel, and log-line cap. References are workspace-only
//! ([`JobReference::Workspace`]); a dropped handle without an explicit terminate
//! records `Failed`.
//!
//! All public methods take `&self`; inner state is under a `parking_lot::Mutex`
//! never held across `.await`.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::common::asset_path::AssetPath;
use crate::common::ids::{HeadId, JobId, WorkspaceId};
use crate::common::workspace::{JobReference, JobType};
use crate::file_mgr::error::FileError;
use crate::file_mgr::recovery::RecoveryReport;
use crate::file_mgr::time_util::now_rfc3339;

use parking_lot::Mutex;
use serde::Serialize;
use tokio::sync::broadcast;

/// Tuning knobs for the [`JobRegistry`].
#[derive(Debug, Clone, Copy)]
pub struct JobRegistryCfg {
    /// Daemon-wide train cap; a second fails [`RegistryConflict::AnotherTrainRunning`].
    pub max_train_jobs: usize,
    pub max_convert_jobs: usize,
    /// One shared slot across all five delete subtypes.
    pub max_delete_jobs: usize,
    pub max_recent_jobs: usize,
    /// Per-job SSE-replay ring sized for the reconnect-replay window, not full
    /// history (older events backfill from the durable JSONL via `EventGap`);
    /// overflow drops oldest + bumps `events_dropped_total`.
    pub max_job_event_ring: usize,
    /// UTF-8 byte cap on a single log line; longer lines get a `... [truncated]` suffix.
    pub max_log_line_bytes: usize,
    /// Per-job progress rate limit (Hz); faster updates coalesce to the latest per
    /// window. Terminal-state events are never throttled.
    pub progress_throttle_hz: f32,
}

impl Default for JobRegistryCfg {
    fn default() -> Self {
        Self {
            max_train_jobs: 1,
            max_convert_jobs: 1,
            max_delete_jobs: 1,
            // 3 running slots (1+1+1) + 1 spare terminal entry.
            max_recent_jobs: 4,
            max_job_event_ring: 128,
            max_log_line_bytes: 8 * 1024,
            progress_throttle_hz: 4.0,
        }
    }
}

/// Terminal job result surfaced through [`JobSnapshot::result`].
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum JobResult {
    WorkspaceDelete {
        /// Whether the deleted workspace was the active generation's source,
        /// captured under `active_mutex` before the staging rename. NOT live (a
        /// concurrent `POST /active` may have rebound it - cross-reference
        /// `GET /active`); runtime survives the delete since activation copied
        /// bytes into `active/generations/<id>/`.
        active_source_deleted: bool,
    },
    /// Typed so `GET /jobs/{job_id}` surfaces the head id directly for a chained
    /// `POST /active` without reading the JSONL log.
    Convert {
        head_id: HeadId,
        /// Lowercase-hex SHA-256 of the published `head.mpk`.
        sha256: String,
        n_classes: u32,
    },
    Train {
        head_id: HeadId,
        sha256: String,
        n_classes: u32,
    },
}

/// Registry-side job lifecycle states, distinct from per-domain state enums
/// whose domain payloads don't belong in a memory-only registry.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum JobState {
    /// Admission cleared, worker not yet started. Reserved for future scheduling;
    /// producers immediately transition to [`Self::Running`].
    Queued,
    /// Worker executing; references held.
    Running,
    Succeeded,
    /// Failure, or dropped without an explicit terminate.
    Failed,
    Cancelled,
}

impl JobState {
    pub fn is_active(self) -> bool {
        matches!(self, Self::Queued | Self::Running)
    }
}

/// Unitless progress; `total = None` renders as a counter, `Some(_)` as a percent.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, serde::Deserialize)]
pub struct JobProgress {
    pub done: u64,
    pub total: Option<u64>,
}

/// Memory-only job snapshot for `GET /jobs[/{job_id}]`, built from one entry
/// under the registry mutex with no log file I/O.
#[derive(Clone, Debug, Serialize)]
pub struct JobSnapshot {
    pub job_id: JobId,
    pub job_type: JobType,
    /// First reference's workspace.
    pub workspace_id: Option<WorkspaceId>,
    /// Display target only (not conflict detection); set for
    /// `dataset_delete`/`converter_delete`, else omitted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_path: Option<AssetPath>,
    pub state: JobState,
    pub progress: Option<JobProgress>,
    pub result: Option<JobResult>,
    /// Last event seq; clients pass back via `?after_seq=` to resume SSE.
    pub last_seq: u64,
    /// RFC3339 wall-clock at last state change.
    pub updated_at: String,
}

/// Per-job event surfaced over SSE and persisted (train/convert) as JSONL.
#[derive(Clone, Debug, Serialize, serde::Deserialize)]
pub struct JobEvent {
    /// Strictly-increasing per-job sequence; SSE clients' reconnect cursor.
    pub seq: u64,
    /// RFC3339 wall-clock at event emission.
    pub at: String,
    /// State transition, or `None` for a pure-progress / pure-log line.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state: Option<JobState>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub progress: Option<JobProgress>,
    /// Free-form log message; already capped to `max_log_line_bytes`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// 409 body when `?after_seq=N` predates the in-memory ring. Clients backfill
/// from the durable `/{training,converter}_logs/{job_id}` JSONL, then reconnect
/// with a fresher `after_seq`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct EventGap {
    /// Oldest event seq still in the ring.
    pub oldest_seq: u64,
    /// Most recent event seq emitted.
    pub latest_seq: u64,
}

/// Counter snapshot for `GET /api/v1/status`.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct JobRegistryCounters {
    pub admission_conflicts_total: u64,
    /// Events dropped from the in-memory ring (capacity exceeded).
    pub events_dropped_total: u64,
    pub progress_throttled_total: u64,
    pub log_lines_truncated_total: u64,
    pub jobs_running: u64,
    pub jobs_retained: u64,
}

/// Why a [`JobRegistry::try_acquire`] failed. Maps to HTTP 409 via `FileError`.
#[derive(Debug, Clone)]
pub enum RegistryConflict {
    /// Train cap reached; distinct from [`Self::JobConflict`] so the api layer
    /// renders the dedicated `another_train_running` code.
    AnotherTrainRunning,
    /// References overlap an in-flight job, or the convert/delete cap is reached.
    JobConflict {
        job_id: JobId,
        /// Existing job's type (disambiguates self-retry vs sibling race).
        job_type: JobType,
    },
    /// `WorkspaceDelete` blocked by an active same-workspace bare lease (upload or
    /// `delete_head`); distinct from [`Self::JobConflict`] because leases have no
    /// `JobId` to look up via `GET /jobs/{id}`.
    WorkspaceLocked { workspace_id: WorkspaceId },
}

impl From<RegistryConflict> for FileError {
    fn from(c: RegistryConflict) -> Self {
        match c {
            RegistryConflict::AnotherTrainRunning => FileError::AnotherTrainRunning,
            RegistryConflict::JobConflict { job_id, job_type } => FileError::JobConflict {
                message: format!("conflicts with running job {job_id} ({job_type:?})"),
            },
            RegistryConflict::WorkspaceLocked { workspace_id } => FileError::JobConflict {
                message: format!(
                    "workspace {workspace_id} is locked by an in-flight upload or head-delete; \
                     wait for it to finish and retry"
                ),
            },
        }
    }
}

/// One registry entry; references are released on terminate, freeing the
/// conflict slot.
#[derive(Debug)]
struct JobEntry {
    job_id: JobId,
    job_type: JobType,
    references: Vec<JobReference>,
    /// Display target only (references are workspace-only and carry no path).
    target_path: Option<AssetPath>,
    state: JobState,
    progress: Option<JobProgress>,
    result: Option<JobResult>,
    last_seq: u64,
    updated_at: String,
    /// SSE-replay ring, capacity `cfg.max_job_event_ring`.
    ring: VecDeque<JobEvent>,
    /// Progress rate-limit timestamp; `None` until the first progress event.
    last_progress_at: Option<Instant>,
}

impl JobEntry {
    fn snapshot(&self) -> JobSnapshot {
        let workspace_id = self.references.first().map(|r| r.workspace_id());
        JobSnapshot {
            job_id: self.job_id,
            job_type: self.job_type,
            workspace_id,
            target_path: self.target_path.clone(),
            state: self.state,
            progress: self.progress,
            result: self.result.clone(),
            last_seq: self.last_seq,
            updated_at: self.updated_at.clone(),
        }
    }

    fn push_event(&mut self, event: JobEvent, max_ring: usize, events_dropped: &AtomicU64) {
        if self.ring.len() >= max_ring {
            self.ring.pop_front();
            events_dropped.fetch_add(1, Ordering::Relaxed);
        }
        self.ring.push_back(event);
    }
}

/// In-process job registry, held by `AppState` as `Arc<JobRegistry>`.
#[derive(Debug)]
pub struct JobRegistry {
    inner: Arc<RegistryInner>,
}

#[derive(Debug)]
struct RegistryInner {
    /// Retained jobs (active + recent terminal), keyed by [`JobId`].
    jobs: Mutex<JobsState>,
    /// Live event broadcast; older-event replay comes from the per-job ring.
    broadcast: broadcast::Sender<RegistryEvent>,
    counters: Counters,
    /// `Arc` because [`RecoveryReport`] isn't `Clone` and per-status-poll deep
    /// clones are too expensive; reads are Arc bumps. Set once at boot.
    boot_recovery: Mutex<Option<Arc<RecoveryReport>>>,
    cfg: JobRegistryCfg,
}

#[derive(Debug)]
struct JobsState {
    /// Newest-first ordering for `recent()`; `entries` is the source of truth.
    order: VecDeque<JobId>,
    entries: std::collections::HashMap<JobId, JobEntry>,
    /// Bare leases (short-lived upload gates), keyed by a monotonic counter so
    /// [`LeaseGuard`] releases on drop without a content search.
    leases: std::collections::HashMap<u64, Vec<JobReference>>,
    next_lease_id: u64,
}

impl JobsState {
    fn new() -> Self {
        Self {
            order: VecDeque::new(),
            entries: std::collections::HashMap::new(),
            leases: std::collections::HashMap::new(),
            next_lease_id: 1,
        }
    }
}

#[derive(Debug, Default)]
struct Counters {
    admission_conflicts_total: AtomicU64,
    events_dropped_total: AtomicU64,
    progress_throttled_total: AtomicU64,
    log_lines_truncated_total: AtomicU64,
}

/// Live event published on the broadcast channel; subscribers filter by `job_id`.
#[derive(Clone, Debug)]
pub struct RegistryEvent {
    pub job_id: JobId,
    pub event: JobEvent,
}

/// Broadcast slot count. Generous because the per-job ring is the durable replay
/// surface and the broadcast is just the wakeup; a slow subscriber gets
/// `RecvError::Lagged` and reconnects with `after_seq`.
const BROADCAST_CAPACITY: usize = 256;

/// Broadcast `event`; a `SendError` (all receivers dropped/lagged) bumps
/// `job_events_dropped`, distinct from in-ring overflow (`events_dropped_total`).
fn send_or_count_drop(
    broadcast: &broadcast::Sender<RegistryEvent>,
    job_id: JobId,
    event: JobEvent,
) {
    // Skip no-subscriber sends so the metric counts only genuine lag overflow.
    if broadcast.receiver_count() == 0 {
        return;
    }
    if broadcast.send(RegistryEvent { job_id, event }).is_err() {
        crate::file_mgr::metrics_hooks::emit_job_events_dropped(1);
    }
}

impl JobRegistry {
    pub fn new(cfg: JobRegistryCfg) -> Self {
        let (tx, _rx) = broadcast::channel(BROADCAST_CAPACITY);
        Self {
            inner: Arc::new(RegistryInner {
                jobs: Mutex::new(JobsState::new()),
                broadcast: tx,
                counters: Counters::default(),
                boot_recovery: Mutex::new(None),
                cfg,
            }),
        }
    }

    pub fn cfg(&self) -> &JobRegistryCfg {
        &self.inner.cfg
    }

    /// Atomic admission gate: validates the per-`JobType` cap and reference
    /// overlap against every active job, then registers the entry. The returned
    /// [`JobHandle`] releases references on drop and (if not explicitly
    /// terminated) records [`JobState::Failed`].
    pub fn try_acquire(
        self: &Arc<Self>,
        job_type: JobType,
        references: Vec<JobReference>,
        target_path: Option<AssetPath>,
    ) -> Result<JobHandle, RegistryConflict> {
        let mut state = self.inner.jobs.lock();

        // Per-type cap counts only active entries; the delete family shares one
        // `max_delete_jobs` slot spanning every delete subtype.
        let (cap, slot_predicate): (usize, fn(JobType) -> bool) = match job_type {
            JobType::Train => (self.inner.cfg.max_train_jobs, |t| t == JobType::Train),
            JobType::Convert => (self.inner.cfg.max_convert_jobs, |t| t == JobType::Convert),
            JobType::DatasetDelete
            | JobType::ConverterDelete
            | JobType::TrainingLogsDelete
            | JobType::ConverterLogsDelete
            | JobType::WorkspaceDelete => {
                (self.inner.cfg.max_delete_jobs, JobType::is_delete_subtype)
            }
        };
        let active_same_type = state
            .entries
            .values()
            .filter(|e| e.state.is_active() && slot_predicate(e.job_type))
            .count();
        if active_same_type >= cap {
            self.inner
                .counters
                .admission_conflicts_total
                .fetch_add(1, Ordering::Relaxed);
            let conflicting = state
                .entries
                .values()
                .find(|e| e.state.is_active() && slot_predicate(e.job_type))
                .map(|e| (e.job_id, e.job_type));
            // `None` is unreachable: `active_same_type >= cap > 0` came from the
            // same scan `find` replays. Release fallback synthesizes a typed
            // `JobConflict` rather than panicking.
            debug_assert!(
                conflicting.is_some() || job_type == JobType::Train,
                "active_same_type >= cap but no matching active entry found; \
                 type={job_type:?}, active_same_type={active_same_type}, cap={cap}",
            );
            return Err(match (job_type, conflicting) {
                (JobType::Train, _) => RegistryConflict::AnotherTrainRunning,
                (_, Some((job_id, jt))) => RegistryConflict::JobConflict {
                    job_id,
                    job_type: jt,
                },
                (_, None) => RegistryConflict::JobConflict {
                    job_id: JobId::new(),
                    job_type,
                },
            });
        }

        // `WorkspaceDelete` is the only exclusive shape: conflict iff workspaces
        // match and at least one side is a `WorkspaceDelete`; it also blocks active
        // bare leases.
        let new_is_workspace_delete = job_type == JobType::WorkspaceDelete;
        for new_ref in &references {
            let new_ws = new_ref.workspace_id();
            for existing in state.entries.values() {
                if !existing.state.is_active() {
                    continue;
                }
                if job_conflicts_with_existing_job(
                    job_type,
                    new_ws,
                    existing.job_type,
                    &existing.references,
                ) {
                    self.inner
                        .counters
                        .admission_conflicts_total
                        .fetch_add(1, Ordering::Relaxed);
                    return Err(RegistryConflict::JobConflict {
                        job_id: existing.job_id,
                        job_type: existing.job_type,
                    });
                }
            }
            if new_is_workspace_delete {
                for lease_refs in state.leases.values() {
                    if job_conflicts_with_existing_lease(
                        new_is_workspace_delete,
                        new_ws,
                        lease_refs,
                    ) {
                        self.inner
                            .counters
                            .admission_conflicts_total
                            .fetch_add(1, Ordering::Relaxed);
                        // Leases have no `JobId`, so name the workspace.
                        return Err(RegistryConflict::WorkspaceLocked {
                            workspace_id: new_ws,
                        });
                    }
                }
            }
        }

        let job_id = JobId::new();
        let entry = JobEntry {
            job_id,
            job_type,
            references: references.clone(),
            target_path,
            state: JobState::Running,
            progress: None,
            result: None,
            last_seq: 0,
            updated_at: now_rfc3339(),
            ring: VecDeque::with_capacity(self.inner.cfg.max_job_event_ring.min(64)),
            last_progress_at: None,
        };
        state.order.push_front(job_id);
        state.entries.insert(job_id, entry);
        prune_recent(&mut state, self.inner.cfg.max_recent_jobs);

        Ok(JobHandle {
            registry: Arc::clone(self),
            job_id,
            terminated: false,
        })
    }

    /// Update progress, rate-limited to `cfg.progress_throttle_hz` (leading-edge:
    /// latest payload wins, no event emitted until the window elapses). After a
    /// burst the entry can lead the emitted stream by up to one window, so SSE
    /// clients MUST snapshot `/jobs/{id}` once at connect for authoritative state.
    fn update_progress(&self, job_id: JobId, progress: JobProgress) {
        let now = Instant::now();
        let throttle =
            Duration::from_secs_f32(1.0 / self.inner.cfg.progress_throttle_hz.max(0.001));
        // Push under the lock, release before broadcasting, so a slow SSE
        // subscriber can't hold the mutex against concurrent admissions.
        let event_to_broadcast: Option<JobEvent> = {
            let mut state = self.inner.jobs.lock();
            let Some(entry) = state.entries.get_mut(&job_id) else {
                return;
            };
            // Latest payload always wins, even when throttled.
            entry.progress = Some(progress);
            if let Some(prev) = entry.last_progress_at
                && now.duration_since(prev) < throttle
            {
                self.inner
                    .counters
                    .progress_throttled_total
                    .fetch_add(1, Ordering::Relaxed);
                return;
            }
            entry.last_progress_at = Some(now);
            entry.last_seq = entry.last_seq.saturating_add(1);
            entry.updated_at = now_rfc3339();
            let event = JobEvent {
                seq: entry.last_seq,
                at: entry.updated_at.clone(),
                state: None,
                progress: Some(progress),
                message: None,
            };
            entry.push_event(
                event.clone(),
                self.inner.cfg.max_job_event_ring,
                &self.inner.counters.events_dropped_total,
            );
            Some(event)
        };
        if let Some(event) = event_to_broadcast {
            send_or_count_drop(&self.inner.broadcast, job_id, event);
        }
    }

    /// Append a free-form log line, truncating over `cfg.max_log_line_bytes` with
    /// a `... [truncated]` suffix.
    fn append_log(&self, job_id: JobId, message: String) {
        let mut msg = message;
        let max = self.inner.cfg.max_log_line_bytes;
        if msg.len() > max {
            self.inner
                .counters
                .log_lines_truncated_total
                .fetch_add(1, Ordering::Relaxed);
            // Largest UTF-8-boundary prefix fitting cap minus suffix.
            let suffix = " ... [truncated]";
            let target = max.saturating_sub(suffix.len()).max(1);
            let mut cut = target;
            while cut > 0 && !msg.is_char_boundary(cut) {
                cut -= 1;
            }
            msg.truncate(cut);
            msg.push_str(suffix);
        }
        // Push under the lock, broadcast after release (see `update_progress`).
        let event_to_broadcast: JobEvent = {
            let mut state = self.inner.jobs.lock();
            let Some(entry) = state.entries.get_mut(&job_id) else {
                return;
            };
            entry.last_seq = entry.last_seq.saturating_add(1);
            entry.updated_at = now_rfc3339();
            let event = JobEvent {
                seq: entry.last_seq,
                at: entry.updated_at.clone(),
                state: None,
                progress: None,
                message: Some(msg),
            };
            entry.push_event(
                event.clone(),
                self.inner.cfg.max_job_event_ring,
                &self.inner.counters.events_dropped_total,
            );
            event
        };
        send_or_count_drop(&self.inner.broadcast, job_id, event_to_broadcast);
    }

    /// Terminate the job with `state` (idempotent), releasing references and
    /// emitting the terminal event.
    fn terminate(&self, job_id: JobId, state: JobState, result: Option<JobResult>) {
        debug_assert!(
            !state.is_active(),
            "terminate called with non-terminal state"
        );
        // Push + prune under the lock, broadcast after release (see `update_progress`).
        let event_to_broadcast: JobEvent = {
            let mut s = self.inner.jobs.lock();
            let Some(entry) = s.entries.get_mut(&job_id) else {
                return;
            };
            if !entry.state.is_active() {
                return;
            }
            entry.state = state;
            if result.is_some() {
                entry.result = result;
            }
            entry.last_seq = entry.last_seq.saturating_add(1);
            entry.updated_at = now_rfc3339();
            // Free the conflict slot; entry stays in history.
            entry.references.clear();
            let event = JobEvent {
                seq: entry.last_seq,
                at: entry.updated_at.clone(),
                state: Some(state),
                progress: entry.progress,
                message: None,
            };
            entry.push_event(
                event.clone(),
                self.inner.cfg.max_job_event_ring,
                &self.inner.counters.events_dropped_total,
            );
            prune_recent(&mut s, self.inner.cfg.max_recent_jobs);
            event
        };
        send_or_count_drop(&self.inner.broadcast, job_id, event_to_broadcast);
    }

    /// Memory-only snapshot for `GET /jobs/{job_id}`.
    pub fn snapshot(&self, job_id: JobId) -> Option<JobSnapshot> {
        self.inner
            .jobs
            .lock()
            .entries
            .get(&job_id)
            .map(JobEntry::snapshot)
    }

    /// Most-recent-first retained job snapshots, capped at
    /// `min(limit, cfg.max_recent_jobs)`.
    pub fn recent(&self, limit: usize) -> Vec<JobSnapshot> {
        let state = self.inner.jobs.lock();
        let cap = limit.min(self.inner.cfg.max_recent_jobs);
        state
            .order
            .iter()
            .take(cap)
            .filter_map(|id| state.entries.get(id).map(JobEntry::snapshot))
            .collect()
    }

    /// Whether `workspace_id` has an active [`JobType::Train`]. Lets training-logs
    /// delete 409 rather than race a producer's open append-fd against the wipe.
    pub fn has_active_train_for(&self, workspace_id: WorkspaceId) -> bool {
        self.has_active_for(workspace_id, JobType::Train)
    }

    /// Converter-side counterpart to [`Self::has_active_train_for`].
    pub fn has_active_convert_for(&self, workspace_id: WorkspaceId) -> bool {
        self.has_active_for(workspace_id, JobType::Convert)
    }

    fn has_active_for(&self, workspace_id: WorkspaceId, job_type: JobType) -> bool {
        let state = self.inner.jobs.lock();
        state.entries.values().any(|e| {
            e.state.is_active()
                && e.job_type == job_type
                && e.references
                    .iter()
                    .any(|r| r.workspace_id() == workspace_id)
        })
    }

    /// Acquire a bare reference lease for short-lived ops (e.g. dataset upload)
    /// needing conflict gating but no job snapshot/ring. The [`LeaseGuard`]
    /// releases on drop, counts against no cap, and is invisible to
    /// [`Self::recent`] / [`Self::snapshot`].
    pub fn try_acquire_lease(
        self: &Arc<Self>,
        references: Vec<JobReference>,
    ) -> Result<LeaseGuard, RegistryConflict> {
        let mut state = self.inner.jobs.lock();
        // Bare leases are gated only by same-workspace `WorkspaceDelete` jobs;
        // they coexist with everything else, leases included.
        for new_ref in &references {
            let new_ws = new_ref.workspace_id();
            for existing in state.entries.values() {
                if !existing.state.is_active() {
                    continue;
                }
                if lease_conflicts_with_existing_job(
                    new_ws,
                    existing.job_type,
                    &existing.references,
                ) {
                    self.inner
                        .counters
                        .admission_conflicts_total
                        .fetch_add(1, Ordering::Relaxed);
                    return Err(RegistryConflict::JobConflict {
                        job_id: existing.job_id,
                        job_type: existing.job_type,
                    });
                }
            }
        }
        let lease_id = state.next_lease_id;
        state.next_lease_id = state.next_lease_id.wrapping_add(1);
        state.leases.insert(lease_id, references);
        Ok(LeaseGuard {
            registry: Arc::clone(self),
            lease_id,
            released: false,
        })
    }

    fn release_lease(&self, lease_id: u64) {
        let mut state = self.inner.jobs.lock();
        state.leases.remove(&lease_id);
    }

    /// Subscribe to the live event stream for `job_id`, replaying ring events
    /// strictly newer than `after_seq` (oldest-first) before live events. An
    /// `after_seq` older than the ring's oldest yields [`EventGap`] so the caller
    /// backfills from the JSONL log before reconnecting.
    pub fn subscribe_events(&self, job_id: JobId, after_seq: u64) -> Result<EventStream, EventGap> {
        let state = self.inner.jobs.lock();
        let entry = match state.entries.get(&job_id) {
            Some(e) => e,
            None => {
                // Unknown id: close on first poll rather than hang the SSE
                // connection to client timeout.
                return Ok(EventStream {
                    replay: std::collections::VecDeque::new(),
                    live: self.inner.broadcast.subscribe(),
                    job_id,
                    terminal_seen: true,
                    max_replay_seq: 0,
                });
            }
        };
        // Empty ring + after_seq=0 is a fresh subscribe: no gap.
        let oldest = entry.ring.front().map(|e| e.seq).unwrap_or(0);
        let latest = entry.last_seq;
        // Gap = ring dropped the cursor's events (`< oldest-1`) or cursor is from
        // another stream (`> latest`); both 409 to backfill via JSONL.
        if after_seq > 0 && (after_seq < oldest.saturating_sub(1) || after_seq > latest) {
            return Err(EventGap {
                oldest_seq: oldest,
                latest_seq: latest,
            });
        }
        let replay: std::collections::VecDeque<JobEvent> = entry
            .ring
            .iter()
            .filter(|e| e.seq > after_seq)
            .cloned()
            .collect();
        let terminal_seen = !entry.state.is_active();
        // Highest replayed seq lets `recv` dedupe a broadcast whose `send` raced
        // this subscribe; empty replay falls back to `last_seq` so the next live
        // event must exceed it.
        let max_replay_seq = replay
            .iter()
            .map(|e| e.seq)
            .max()
            .unwrap_or(after_seq.max(entry.last_seq));
        Ok(EventStream {
            replay,
            live: self.inner.broadcast.subscribe(),
            job_id,
            terminal_seen,
            max_replay_seq,
        })
    }

    /// Counter snapshot for `GET /api/v1/status`.
    pub fn counters(&self) -> JobRegistryCounters {
        let state = self.inner.jobs.lock();
        let jobs_running = state
            .entries
            .values()
            .filter(|e| e.state.is_active())
            .count() as u64;
        let jobs_retained = state.entries.len() as u64;
        let c = &self.inner.counters;
        JobRegistryCounters {
            admission_conflicts_total: c.admission_conflicts_total.load(Ordering::Relaxed),
            events_dropped_total: c.events_dropped_total.load(Ordering::Relaxed),
            progress_throttled_total: c.progress_throttled_total.load(Ordering::Relaxed),
            log_lines_truncated_total: c.log_lines_truncated_total.load(Ordering::Relaxed),
            jobs_running,
            jobs_retained,
        }
    }

    /// Stash the boot-recovery report for the status surface; no-op once set.
    pub fn record_boot_recovery(&self, report: RecoveryReport) {
        let mut slot = self.inner.boot_recovery.lock();
        if slot.is_none() {
            *slot = Some(Arc::new(report));
        }
    }

    /// Stashed boot-recovery report (cheap `Arc` bump, no deep clone).
    pub fn boot_recovery(&self) -> Option<Arc<RecoveryReport>> {
        self.inner.boot_recovery.lock().clone()
    }
}

/// RAII guard for one admitted job. `Drop` releases the references and records
/// [`JobState::Failed`] if the worker abandoned the job without an explicit
/// terminate.
#[derive(Debug)]
pub struct JobHandle {
    registry: Arc<JobRegistry>,
    job_id: JobId,
    terminated: bool,
}

impl JobHandle {
    pub fn job_id(&self) -> JobId {
        self.job_id
    }

    /// Update progress (rate-limited; trailing-edge gap documented on [`JobRegistry`]).
    pub fn update_progress(&self, progress: JobProgress) {
        self.registry.update_progress(self.job_id, progress);
    }

    /// Append a log line, truncated to `max_log_line_bytes` if needed.
    pub fn append_log<S: Into<String>>(&self, message: S) {
        self.registry.append_log(self.job_id, message.into());
    }

    pub fn succeed(mut self, result: Option<JobResult>) {
        self.registry
            .terminate(self.job_id, JobState::Succeeded, result);
        self.terminated = true;
    }

    pub fn fail<S: Into<String>>(mut self, reason: S) {
        // Log the reason so it's visible without opening JSONL.
        self.registry.append_log(self.job_id, reason.into());
        self.registry.terminate(self.job_id, JobState::Failed, None);
        self.terminated = true;
    }

    pub fn cancel(mut self) {
        self.registry
            .terminate(self.job_id, JobState::Cancelled, None);
        self.terminated = true;
    }
}

impl Drop for JobHandle {
    fn drop(&mut self) {
        if !self.terminated {
            // Abandoned without succeed/fail/cancel; record Failed.
            self.registry.terminate(self.job_id, JobState::Failed, None);
        }
    }
}

/// Short-lived reference-only lease from [`JobRegistry::try_acquire_lease`] for
/// ops needing conflict gating but no full job (e.g. dataset upload). Drop
/// releases it.
#[derive(Debug)]
pub struct LeaseGuard {
    registry: Arc<JobRegistry>,
    lease_id: u64,
    released: bool,
}

impl Drop for LeaseGuard {
    fn drop(&mut self) {
        if !self.released {
            self.registry.release_lease(self.lease_id);
            self.released = true;
        }
    }
}

/// Replay-then-live job-event stream for one subscription: ring events
/// (oldest-first) then live broadcast events filtered to the job. Slow-consumer
/// lag surfaces as [`EventStreamError::Lagged`].
#[derive(Debug)]
pub struct EventStream {
    replay: std::collections::VecDeque<JobEvent>,
    live: broadcast::Receiver<RegistryEvent>,
    job_id: JobId,
    terminal_seen: bool,
    /// Highest `seq` covered by `replay`. Producers release the mutex before
    /// broadcasting, so a racing subscribe can snapshot event `N` into both
    /// `replay` and the pending `live` send; filtering `seq <= max_replay_seq`
    /// collapses that duplicate without holding the lock across the broadcast.
    max_replay_seq: u64,
}

impl EventStream {
    /// Pop the next replay event; drain before falling back to the live channel.
    pub fn next_replay(&mut self) -> Option<JobEvent> {
        self.replay.pop_front()
    }

    pub fn replay_drained(&self) -> bool {
        self.replay.is_empty()
    }

    /// True once a terminal event was observed; SSE streams close after.
    pub fn terminal_seen(&self) -> bool {
        self.terminal_seen
    }

    /// Receive the next live event for this job, skipping other jobs and events
    /// already covered by the replay snapshot (push-then-broadcast race).
    pub async fn recv(&mut self) -> Result<JobEvent, EventStreamError> {
        loop {
            match self.live.recv().await {
                Ok(re) => {
                    if re.job_id != self.job_id {
                        continue;
                    }
                    // Dedupe against replay; see `max_replay_seq` field doc.
                    if re.event.seq <= self.max_replay_seq {
                        continue;
                    }
                    if re.event.state.is_some_and(|s| !s.is_active()) {
                        self.terminal_seen = true;
                    }
                    return Ok(re.event);
                }
                Err(broadcast::error::RecvError::Closed) => {
                    return Err(EventStreamError::Closed);
                }
                Err(broadcast::error::RecvError::Lagged(_)) => {
                    return Err(EventStreamError::Lagged);
                }
            }
        }
    }
}

/// Receive-side error from [`EventStream::recv`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum EventStreamError {
    /// Subscriber lagged; client reconnects with `?after_seq=`.
    #[error("subscriber lagged the broadcast channel")]
    Lagged,
    /// Sender dropped (registry shutdown).
    #[error("broadcast channel closed")]
    Closed,
}

/// Conflict predicate between a new job and an existing in-flight job. Two
/// exclusion shapes (else siblings coexist under their per-type caps):
/// 1. `WorkspaceDelete` excludes everything same-workspace (it renames the whole
///    tree out from under readers/writers).
/// 2. `{Training,Converter}LogsDelete` excludes its producer (`Train`/`Convert`):
///    the producer's open append-fd survives the delete's log-dir rename, so on
///    Linux its writes silently land in a dirent-less inode until the fd closes.
fn job_conflicts_with_existing_job(
    new_job_type: JobType,
    new_workspace: WorkspaceId,
    existing_job_type: JobType,
    existing_refs: &[JobReference],
) -> bool {
    if !existing_refs
        .iter()
        .any(|r| r.workspace_id() == new_workspace)
    {
        return false;
    }
    let new_is_ws_delete = new_job_type == JobType::WorkspaceDelete;
    let existing_is_ws_delete = existing_job_type == JobType::WorkspaceDelete;
    if new_is_ws_delete || existing_is_ws_delete {
        return true;
    }
    // Log-producer / log-delete pairing, either direction.
    matches!(
        (new_job_type, existing_job_type),
        (JobType::TrainingLogsDelete, JobType::Train)
            | (JobType::Train, JobType::TrainingLogsDelete)
            | (JobType::ConverterLogsDelete, JobType::Convert)
            | (JobType::Convert, JobType::ConverterLogsDelete)
    )
}

/// Conflict between a new admission and an existing bare lease: only a
/// same-workspace `WorkspaceDelete` conflicts.
fn job_conflicts_with_existing_lease(
    new_is_workspace_delete: bool,
    new_workspace: WorkspaceId,
    lease_refs: &[JobReference],
) -> bool {
    if !new_is_workspace_delete {
        return false;
    }
    lease_refs.iter().any(|r| r.workspace_id() == new_workspace)
}

/// Conflict for a new bare lease against an existing job: only a same-workspace
/// `WorkspaceDelete`.
fn lease_conflicts_with_existing_job(
    new_workspace: WorkspaceId,
    existing_job_type: JobType,
    existing_refs: &[JobReference],
) -> bool {
    if existing_job_type != JobType::WorkspaceDelete {
        return false;
    }
    existing_refs
        .iter()
        .any(|r| r.workspace_id() == new_workspace)
}

/// Drop oldest terminal entries until `entries.len() <= max_recent`. Active jobs
/// are never displaced; an all-active oversubscription fails admission earlier.
fn prune_recent(state: &mut JobsState, max_recent: usize) {
    if state.entries.len() <= max_recent {
        return;
    }
    let mut i = state.order.len();
    while state.entries.len() > max_recent && i > 0 {
        i -= 1;
        let id = state.order[i];
        let drop_it = matches!(state.entries.get(&id), Some(e) if !e.state.is_active());
        if drop_it {
            state.order.remove(i);
            state.entries.remove(&id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::asset_path::AssetPath;

    fn ws_id() -> WorkspaceId {
        WorkspaceId::parse("11111111-2222-4333-8444-555555555555").unwrap()
    }
    fn ws_id_b() -> WorkspaceId {
        WorkspaceId::parse("22222222-3333-4444-8555-666666666666").unwrap()
    }

    fn fresh() -> Arc<JobRegistry> {
        Arc::new(JobRegistry::new(JobRegistryCfg::default()))
    }

    fn ws_ref(id: WorkspaceId) -> Vec<JobReference> {
        vec![JobReference::Workspace { workspace_id: id }]
    }

    /// Cap fires before any reference check, regardless of workspace.
    #[test]
    fn second_train_returns_another_train_running() {
        let r = fresh();
        let _h = r
            .try_acquire(JobType::Train, ws_ref(ws_id()), None)
            .unwrap();
        let err = r
            .try_acquire(JobType::Train, ws_ref(ws_id_b()), None)
            .unwrap_err();
        assert!(matches!(err, RegistryConflict::AnotherTrainRunning));
    }

    #[test]
    fn workspace_delete_blocks_train_in_same_workspace() {
        let r = fresh();
        let _h = r
            .try_acquire(JobType::WorkspaceDelete, ws_ref(ws_id()), None)
            .unwrap();
        let err = r
            .try_acquire(JobType::Train, ws_ref(ws_id()), None)
            .unwrap_err();
        assert!(matches!(err, RegistryConflict::JobConflict { .. }));
    }

    #[test]
    fn train_blocks_workspace_delete_in_same_workspace() {
        let r = fresh();
        let _h = r
            .try_acquire(JobType::Train, ws_ref(ws_id()), None)
            .unwrap();
        let err = r
            .try_acquire(JobType::WorkspaceDelete, ws_ref(ws_id()), None)
            .unwrap_err();
        assert!(matches!(err, RegistryConflict::JobConflict { .. }));
    }

    #[test]
    fn train_and_dataset_delete_coexist_in_same_workspace() {
        let r = fresh();
        let _h_train = r
            .try_acquire(JobType::Train, ws_ref(ws_id()), None)
            .unwrap();
        let _h_del = r
            .try_acquire(
                JobType::DatasetDelete,
                ws_ref(ws_id()),
                Some(AssetPath::parse("audio/cat").unwrap()),
            )
            .unwrap();
    }

    /// Separate caps coexist same-workspace when neither side is a delete.
    #[test]
    fn train_and_convert_coexist_in_same_workspace() {
        let r = fresh();
        let _h_train = r
            .try_acquire(JobType::Train, ws_ref(ws_id()), None)
            .unwrap();
        let _h_conv = r
            .try_acquire(JobType::Convert, ws_ref(ws_id()), None)
            .unwrap();
    }

    /// Exclusive with an in-flight same-workspace `Train` (open append-fd vs
    /// log-dir rename).
    #[test]
    fn training_logs_delete_conflicts_with_active_train_in_same_workspace() {
        let r = fresh();
        let _h_train = r
            .try_acquire(JobType::Train, ws_ref(ws_id()), None)
            .unwrap();
        let err = r
            .try_acquire(JobType::TrainingLogsDelete, ws_ref(ws_id()), None)
            .expect_err("logs-delete must conflict with active train");
        assert!(matches!(err, RegistryConflict::JobConflict { .. }));
    }

    #[test]
    fn train_conflicts_with_active_training_logs_delete_in_same_workspace() {
        let r = fresh();
        let _h_del = r
            .try_acquire(JobType::TrainingLogsDelete, ws_ref(ws_id()), None)
            .unwrap();
        let err = r
            .try_acquire(JobType::Train, ws_ref(ws_id()), None)
            .expect_err("train must conflict with active logs-delete");
        assert!(matches!(err, RegistryConflict::JobConflict { .. }));
    }

    #[test]
    fn converter_logs_delete_conflicts_with_active_convert_in_same_workspace() {
        let r = fresh();
        let _h_conv = r
            .try_acquire(JobType::Convert, ws_ref(ws_id()), None)
            .unwrap();
        let err = r
            .try_acquire(JobType::ConverterLogsDelete, ws_ref(ws_id()), None)
            .expect_err("logs-delete must conflict with active convert");
        assert!(matches!(err, RegistryConflict::JobConflict { .. }));
    }

    #[test]
    fn logs_delete_does_not_conflict_across_workspaces() {
        let r = fresh();
        let _h_del = r
            .try_acquire(JobType::TrainingLogsDelete, ws_ref(ws_id()), None)
            .unwrap();
        let _h_train = r
            .try_acquire(JobType::Train, ws_ref(ws_id_b()), None)
            .unwrap();
    }

    /// The `WorkspaceDelete` exclusion is workspace-scoped (type caps are global).
    #[test]
    fn workspace_delete_isolated_per_workspace() {
        let r = fresh();
        let _h = r
            .try_acquire(JobType::Train, ws_ref(ws_id()), None)
            .unwrap();
        let _h_del = r
            .try_acquire(JobType::WorkspaceDelete, ws_ref(ws_id_b()), None)
            .unwrap();
    }

    #[test]
    fn upload_lease_blocked_by_active_workspace_delete() {
        let r = fresh();
        let _h = r
            .try_acquire(JobType::WorkspaceDelete, ws_ref(ws_id()), None)
            .unwrap();
        let err = r.try_acquire_lease(ws_ref(ws_id())).unwrap_err();
        assert!(matches!(err, RegistryConflict::JobConflict { .. }));
    }

    #[test]
    fn upload_lease_coexists_with_train_in_same_workspace() {
        let r = fresh();
        let _h = r
            .try_acquire(JobType::Train, ws_ref(ws_id()), None)
            .unwrap();
        let _lease = r.try_acquire_lease(ws_ref(ws_id())).unwrap();
    }

    #[test]
    fn two_upload_leases_coexist_in_same_workspace() {
        let r = fresh();
        let _l1 = r.try_acquire_lease(ws_ref(ws_id())).unwrap();
        let _l2 = r.try_acquire_lease(ws_ref(ws_id())).unwrap();
    }

    /// Active bare lease blocks `WorkspaceDelete`; surfaces `WorkspaceLocked`,
    /// not `JobConflict`, since a lease has no `JobId`.
    #[test]
    fn workspace_delete_blocked_by_in_flight_upload_lease() {
        let r = fresh();
        let _lease = r.try_acquire_lease(ws_ref(ws_id())).unwrap();
        let err = r
            .try_acquire(JobType::WorkspaceDelete, ws_ref(ws_id()), None)
            .unwrap_err();
        match err {
            RegistryConflict::WorkspaceLocked { workspace_id } => {
                assert_eq!(workspace_id, ws_id());
            }
            other => panic!("expected WorkspaceLocked, got {other:?}"),
        }
    }

    #[test]
    fn dataset_and_converter_deletes_share_one_slot() {
        let r = fresh();
        let _h = r
            .try_acquire(
                JobType::DatasetDelete,
                ws_ref(ws_id()),
                Some(AssetPath::parse("a").unwrap()),
            )
            .unwrap();
        let err = r
            .try_acquire(
                JobType::ConverterDelete,
                ws_ref(ws_id_b()),
                Some(AssetPath::parse("tfjs").unwrap()),
            )
            .unwrap_err();
        assert!(matches!(err, RegistryConflict::JobConflict { .. }));
    }

    #[test]
    fn snapshot_target_path_round_trips_through_acquire() {
        let r = fresh();
        let target = AssetPath::parse("audio/cat").unwrap();
        let h = r
            .try_acquire(
                JobType::DatasetDelete,
                ws_ref(ws_id()),
                Some(target.clone()),
            )
            .unwrap();
        let snap = r.snapshot(h.job_id()).unwrap();
        assert_eq!(snap.target_path.as_ref(), Some(&target));
        assert_eq!(snap.workspace_id, Some(ws_id()));
    }

    #[test]
    fn snapshot_target_path_none_for_train() {
        let r = fresh();
        let h = r
            .try_acquire(JobType::Train, ws_ref(ws_id()), None)
            .unwrap();
        let snap = r.snapshot(h.job_id()).unwrap();
        assert!(snap.target_path.is_none());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn update_progress_rate_limited_to_4hz() {
        // 1 Hz for test stability.
        let cfg = JobRegistryCfg {
            progress_throttle_hz: 1.0,
            ..Default::default()
        };
        let r = Arc::new(JobRegistry::new(cfg));
        let h = r
            .try_acquire(JobType::Train, ws_ref(ws_id()), None)
            .unwrap();
        for i in 0..5u64 {
            h.update_progress(JobProgress {
                done: i,
                total: Some(100),
            });
        }
        let snap = r.snapshot(h.job_id()).unwrap();
        // Only the leading-edge update emits, yet latest payload wins.
        assert_eq!(snap.last_seq, 1);
        assert_eq!(snap.progress.unwrap().done, 4);
        let counters = r.counters();
        assert!(counters.progress_throttled_total >= 4);
    }

    #[test]
    fn append_log_caps_line_length() {
        let cfg = JobRegistryCfg {
            max_log_line_bytes: 32,
            ..Default::default()
        };
        let r = Arc::new(JobRegistry::new(cfg));
        let h = r
            .try_acquire(JobType::Convert, ws_ref(ws_id()), None)
            .unwrap();
        let big = "x".repeat(1024);
        h.append_log(big);
        let counters = r.counters();
        assert_eq!(counters.log_lines_truncated_total, 1);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn subscribe_events_replays_from_ring() {
        let r = fresh();
        let h = r
            .try_acquire(JobType::Train, ws_ref(ws_id()), None)
            .unwrap();
        h.append_log("hello");
        h.append_log("world");
        let mut stream = r.subscribe_events(h.job_id(), 0).unwrap();
        let mut got = Vec::new();
        while let Some(e) = stream.next_replay() {
            got.push(e.message.unwrap_or_default());
        }
        assert_eq!(got, vec!["hello".to_string(), "world".to_string()]);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn subscribe_events_dedupes_live_event_already_in_replay() {
        // Guards the push-then-broadcast race: an event in `replay` that also
        // arrives live must be filtered by `max_replay_seq`, not re-yielded.
        let r = fresh();
        let h = r
            .try_acquire(JobType::Train, ws_ref(ws_id()), None)
            .unwrap();
        let job_id = h.job_id();
        h.append_log("first");
        let mut stream = r.subscribe_events(job_id, 0).unwrap();
        let replay: Vec<_> = std::iter::from_fn(|| stream.next_replay()).collect();
        assert_eq!(replay.len(), 1);
        let replayed_seq = replay[0].seq;
        let dup = RegistryEvent {
            job_id,
            event: replay[0].clone(),
        };
        r.inner.broadcast.send(dup).unwrap();
        // New event so recv returns after skipping the dup.
        h.append_log("second");
        let next = stream.recv().await.unwrap();
        assert!(
            next.seq > replayed_seq,
            "recv yielded seq {} which was already in replay (max_replay_seq={})",
            next.seq,
            replayed_seq,
        );
        assert_eq!(next.message.as_deref(), Some("second"));
    }

    #[test]
    fn subscribe_events_returns_event_gap_when_after_seq_too_old() {
        // Tiny ring so few writes produce a gap.
        let cfg = JobRegistryCfg {
            max_job_event_ring: 2,
            ..Default::default()
        };
        let r = Arc::new(JobRegistry::new(cfg));
        let h = r
            .try_acquire(JobType::Train, ws_ref(ws_id()), None)
            .unwrap();
        for _ in 0..5 {
            h.append_log("evt");
        }
        // after_seq=0 is the fresh-subscribe case, so ask for an old seq.
        let res = r.subscribe_events(h.job_id(), 1);
        assert!(matches!(res, Err(EventGap { .. })));
    }

    #[test]
    fn terminate_releases_references() {
        let r = fresh();
        let h = r
            .try_acquire(JobType::Train, ws_ref(ws_id()), None)
            .unwrap();
        let job_id = h.job_id();
        h.succeed(None);
        // Slot freed: same-workspace re-acquire now succeeds.
        let _h2 = r
            .try_acquire(JobType::Train, ws_ref(ws_id()), None)
            .unwrap();
        let snap = r.snapshot(job_id).unwrap();
        assert_eq!(snap.state, JobState::Succeeded);
    }

    #[test]
    fn drop_handle_releases_references_and_records_failed() {
        let r = fresh();
        let job_id = {
            let h = r
                .try_acquire(JobType::Train, ws_ref(ws_id()), None)
                .unwrap();
            h.job_id()
            // h drops without explicit terminate, releasing the slot.
        };
        let _h2 = r
            .try_acquire(JobType::Train, ws_ref(ws_id()), None)
            .unwrap();
        let snap = r.snapshot(job_id).unwrap();
        assert_eq!(snap.state, JobState::Failed);
    }

    #[test]
    fn recent_bounded_to_max_recent_jobs() {
        let cfg = JobRegistryCfg {
            max_recent_jobs: 2,
            ..Default::default()
        };
        let r = Arc::new(JobRegistry::new(cfg));
        for _ in 0..5 {
            // Terminate each so the next acquire reuses the freed slot.
            let h = r
                .try_acquire(JobType::Convert, ws_ref(ws_id()), None)
                .unwrap();
            h.succeed(None);
        }
        let snaps = r.recent(100);
        assert!(snaps.len() <= 2, "got {} > 2 retained", snaps.len());
    }

    #[test]
    fn counters_track_admission_conflicts() {
        let r = fresh();
        let _h = r
            .try_acquire(JobType::Train, ws_ref(ws_id()), None)
            .unwrap();
        let _ = r.try_acquire(JobType::Train, ws_ref(ws_id_b()), None);
        assert!(r.counters().admission_conflicts_total >= 1);
    }

    #[test]
    fn has_active_train_for_filters_by_workspace() {
        let r = fresh();
        let _h = r
            .try_acquire(JobType::Train, ws_ref(ws_id()), None)
            .unwrap();
        assert!(r.has_active_train_for(ws_id()));
        assert!(!r.has_active_train_for(ws_id_b()));
        assert!(!r.has_active_convert_for(ws_id()));
    }

    #[test]
    fn registry_conflict_maps_to_file_error() {
        let err: FileError = RegistryConflict::AnotherTrainRunning.into();
        assert!(matches!(err, FileError::AnotherTrainRunning));
        let err: FileError = RegistryConflict::JobConflict {
            job_id: JobId::new(),
            job_type: JobType::Convert,
        }
        .into();
        assert!(matches!(err, FileError::JobConflict { .. }));
    }
}
