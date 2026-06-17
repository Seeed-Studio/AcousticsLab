//! Metrics publication for `file_mgr` hot paths.
//!
//! `file_mgr` may not reach `status::WorkspaceMetrics` (layering), so the daemon
//! installs typed `Fn` slots at boot; an uninstalled slot is a no-op. Each slot
//! is an `OnceLock<Box<dyn Fn>>`: install is single-shot, the read path lock-free.

use std::sync::OnceLock;
use std::time::Duration;

/// `Duration` is the wall-clock of the timed `put_atomic`: tempfile write, flush, fsync (`sync_all`), rename, parent fsync; the caller's serialize runs before the timer and is excluded.
pub type WriteDurationHook = dyn Fn(Duration) + Send + Sync + 'static;

/// Argument is the upload size in bytes (receipt `size_bytes`).
pub type UploadHook = dyn Fn(u64) + Send + Sync + 'static;

pub type IncrementHook = dyn Fn() + Send + Sync + 'static;

/// Argument is the count of broadcast events dropped because `send` returned `SendError` (every receiver had been dropped); the no-subscriber case is filtered out before the send.
pub type EventsDroppedHook = dyn Fn(u64) + Send + Sync + 'static;

/// Arguments are `(pruned, failures)` from one keep-last-N log sweep.
pub type LogsPrunedHook = dyn Fn(u64, u64) + Send + Sync + 'static;

static WORKSPACE_CORE_WRITE: OnceLock<Box<WriteDurationHook>> = OnceLock::new();
static HEAD_INDEX_WRITE: OnceLock<Box<WriteDurationHook>> = OnceLock::new();
static UPLOAD: OnceLock<Box<UploadHook>> = OnceLock::new();
static DATASET_MUTATION_REJECTED: OnceLock<Box<IncrementHook>> = OnceLock::new();
static CONVERTER_MUTATION_REJECTED: OnceLock<Box<IncrementHook>> = OnceLock::new();
static JOB_EVENTS_DROPPED: OnceLock<Box<EventsDroppedHook>> = OnceLock::new();
static LOGS_PRUNED: OnceLock<Box<LogsPrunedHook>> = OnceLock::new();

/// `debug!` not `warn!`: tests re-install slots, which would spam at `warn!`.
fn note_double_install(slot: &'static str) {
    tracing::debug!(
        target: "file_mgr",
        slot,
        "metrics_hooks: install ignored (slot already set; OnceLock is single-shot)",
    );
}

pub fn install_workspace_core_write_hook<F>(f: F)
where
    F: Fn(Duration) + Send + Sync + 'static,
{
    if WORKSPACE_CORE_WRITE.set(Box::new(f)).is_err() {
        note_double_install("workspace_core_write");
    }
}

pub fn install_head_index_write_hook<F>(f: F)
where
    F: Fn(Duration) + Send + Sync + 'static,
{
    if HEAD_INDEX_WRITE.set(Box::new(f)).is_err() {
        note_double_install("head_index_write");
    }
}

pub fn install_upload_hook<F>(f: F)
where
    F: Fn(u64) + Send + Sync + 'static,
{
    if UPLOAD.set(Box::new(f)).is_err() {
        note_double_install("upload");
    }
}

pub fn install_dataset_mutation_rejected_hook<F>(f: F)
where
    F: Fn() + Send + Sync + 'static,
{
    if DATASET_MUTATION_REJECTED.set(Box::new(f)).is_err() {
        note_double_install("dataset_mutation_rejected");
    }
}

pub fn install_converter_mutation_rejected_hook<F>(f: F)
where
    F: Fn() + Send + Sync + 'static,
{
    if CONVERTER_MUTATION_REJECTED.set(Box::new(f)).is_err() {
        note_double_install("converter_mutation_rejected");
    }
}

pub fn install_job_events_dropped_hook<F>(f: F)
where
    F: Fn(u64) + Send + Sync + 'static,
{
    if JOB_EVENTS_DROPPED.set(Box::new(f)).is_err() {
        note_double_install("job_events_dropped");
    }
}

pub fn install_logs_pruned_hook<F>(f: F)
where
    F: Fn(u64, u64) + Send + Sync + 'static,
{
    if LOGS_PRUNED.set(Box::new(f)).is_err() {
        note_double_install("logs_pruned");
    }
}

pub(crate) fn emit_workspace_core_write(d: Duration) {
    if let Some(h) = WORKSPACE_CORE_WRITE.get() {
        h(d);
    }
}

pub(crate) fn emit_head_index_write(d: Duration) {
    if let Some(h) = HEAD_INDEX_WRITE.get() {
        h(d);
    }
}

pub(crate) fn emit_upload(bytes: u64) {
    if let Some(h) = UPLOAD.get() {
        h(bytes);
    }
}

pub(crate) fn emit_dataset_mutation_rejected() {
    if let Some(h) = DATASET_MUTATION_REJECTED.get() {
        h();
    }
}

pub(crate) fn emit_converter_mutation_rejected() {
    if let Some(h) = CONVERTER_MUTATION_REJECTED.get() {
        h();
    }
}

pub(crate) fn emit_job_events_dropped(n: u64) {
    if let Some(h) = JOB_EVENTS_DROPPED.get() {
        h(n);
    }
}

/// Skipped when both counts are 0 so an idle workspace's empty sweep does not churn the metrics surface.
pub(crate) fn emit_logs_pruned(pruned: u64, failures: u64) {
    if pruned == 0 && failures == 0 {
        return;
    }
    if let Some(h) = LOGS_PRUNED.get() {
        h(pruned, failures);
    }
}
